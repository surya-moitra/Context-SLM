#!/usr/bin/env python3
"""Build a frozen development pack; reference facts never enter inference metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from PRAGMOS_knowledge_update_pack_cases import CASES, EXTRA_FEATURES, PROFILE_GROUPS


DATA_NAME = "pragmos_knowledge_update_validation_40_v1.json"
FIELDS = {
    "question_id", "question_type", "question", "answer", "question_date",
    "answer_session_ids", "haystack_session_ids", "haystack_dates", "haystack_sessions",
}


def payload(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timestamp(value):
    # Weekday is display metadata; compare the actual calendar date and clock.
    return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M")


def messages(session):
    if isinstance(session, list):
        return session
    for key in ("messages", "turns"):
        if isinstance(session.get(key), list):
            return session[key]
    raise ValueError("Unsupported session structure")


def reference_answer(facts, proof):
    """Small authoring-only calculator; it is not imported by a benchmark adapter."""
    op = proof["op"]
    ids = proof.get("ids", [proof["id"]] if "id" in proof else [])
    values = [facts[key]["value"] for key in ids]
    if op == "pick":
        value = values[0]
    elif op in {"min", "max"}:
        value = {"min": min, "max": max}[op](values)
    elif op == "equal":
        value = values[0] == values[1]
    elif op == "pair":
        return f"Initially {values[0]}; now {values[1]}"
    elif op == "set_size":
        members = set(facts[proof["base"]]["value"])
        for key in proof["add"]:
            members.update(facts[key]["value"])
        for key in proof["remove"]:
            members.difference_update(facts[key]["value"])
        value = len(members)
    elif op == "ratio_direction":
        before, after = Fraction(values[0], values[1]), Fraction(values[2], values[3])
        return "more" if after > before else "less" if after < before else "the same"
    elif op == "floor_divide":
        if values[1] <= 0 or values[0] < 0:
            raise ValueError("Invalid entitlement operands")
        value = values[0] // values[1]
    elif op == "subtract":
        value = values[0] - values[1]
    elif op == "distinct_count":
        value = len(set(values))
    elif op == "direction":
        return "increased" if values[1] > values[0] else "decreased" if values[1] < values[0] else "unchanged"
    elif op == "absent":
        return "I do not know."
    elif op == "other_member":
        remaining = [v for v in values[0] if v not in proof["exclude"]]
        if not remaining:
            raise ValueError("An existence fixture must contain an additional member")
        return "Yes, a " + ", ".join(remaining) + "."
    else:
        raise ValueError(f"Unknown reference operation: {op}")
    format_name = proof.get("format")
    if format_name == "duration_seconds":
        minutes, seconds = divmod(value, 60)
        return f"{minutes} minutes {seconds} seconds"
    if format_name == "currency":
        return f"${value:,}"
    if format_name in {"kg", "points"}:
        return f"{value} {format_name}"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def build_record(source, source_audits, fixture):
    if set(source) != FIELDS or source["question_type"] != "knowledge-update":
        raise ValueError("Unexpected inference schema or question type")
    index = fixture["source_profile_index"]
    slots = set()
    for row in source_audits:
        for item in row["evidence"]:
            slots.add((item["session_id"], int(item["turn_index"])))
    session_ids = source["haystack_session_ids"]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("Duplicate session IDs")
    if not (len(session_ids) == len(source["haystack_sessions"]) == len(source["haystack_dates"])):
        raise ValueError("Mismatched haystack arrays")
    positions = {sid: i for i, sid in enumerate(session_ids)}
    latest_slots = {(item["session_id"], int(item["turn_index"])) for item in source_audits[-1]["evidence"]}
    preferred = {(sid, min(turn for slot_sid, turn in latest_slots if slot_sid == sid))
                 for sid, _ in latest_slots}
    restored_slots = []
    for sid in source["answer_session_ids"]:
        if any(slot_sid == sid for slot_sid, _ in preferred):
            continue
        turns = messages(source["haystack_sessions"][positions[sid]])
        user_indices = [i for i, turn in enumerate(turns) if turn["role"] == "user"]
        if not user_indices:
            raise ValueError("Missing user slot in a support session")
        slot = (sid, user_indices[len(user_indices) // 2])
        preferred.add(slot)
        restored_slots.append(slot)
    ordered = sorted(preferred, key=lambda slot: (
        timestamp(source["haystack_dates"][positions[slot[0]]]), slot[1]
    ))
    if not 1 <= len(ordered) <= 2 or len(fixture["episodes"]) != 2:
        raise ValueError("Expected one or two authoritative source slots")
    if len(ordered) == 1:
        # Preserve a single-session profile; its earlier/later account shares one turn.
        ordered *= 2
    result = copy.deepcopy(source)
    result["question_id"] = f"synthetic_ku40_v1_{index:04d}" + ("_abs" if fixture["proof"]["op"] == "absent" else "")
    result["question"] = fixture["question"]
    facts, episode_audit = {}, []
    required_episodes = set(fixture["proof"].get("include_episodes", []))
    obsolete_slots = slots - preferred
    for sid, turn_index in obsolete_slots:
        turns = messages(result["haystack_sessions"][positions[sid]])
        if turns[turn_index]["role"] != "user":
            raise ValueError("Obsolete source slot is not a user turn")
        turns[turn_index]["content"] = "I have kept the earlier conversation notes with my other papers."
        if turn_index + 1 < len(turns) and turns[turn_index + 1]["role"] == "assistant":
            turns[turn_index + 1]["content"] = "Thank you. What else would you like to discuss?"
    rendered = [" ".join(part["text"] if isinstance(part, dict) else part for part in episode)
                for episode in fixture["episodes"]]
    for episode_number, ((sid, turn_index), episode) in enumerate(zip(ordered, fixture["episodes"])):
        session_index = positions[sid]
        turns = messages(result["haystack_sessions"][session_index])
        if turns[turn_index]["role"] != "user":
            raise ValueError("Authoritative slot is not a user turn")
        source_date = result["haystack_dates"][session_index]
        if timestamp(source_date) > timestamp(result["question_date"]):
            raise ValueError("Source lies after the evaluation question")
        text = rendered[episode_number]
        same_turn = ordered[0] == ordered[1]
        turns[turn_index]["content"] = "\n\n".join(rendered) if same_turn else text
        # Remove stale acknowledgements that might repeat the previous fixture's facts.
        neutralized_turn = None
        if turn_index + 1 < len(turns) and turns[turn_index + 1]["role"] == "assistant":
            turns[turn_index + 1]["content"] = "Thank you for the update. What else would you like to discuss?"
            neutralized_turn = turn_index + 1
        for part in episode:
            if not isinstance(part, dict):
                continue
            if part["id"] in facts or text.count(part["text"]) != 1:
                raise ValueError("Duplicate fact ID or ambiguous source span")
            start = text.index(part["text"])
            if same_turn and episode_number == 1:
                start += len(rendered[0]) + 2
            facts[part["id"]] = {
                **part, "episode": episode_number, "source_session_index": session_index,
                "source_turn_index": turn_index, "timestamp": source_date,
                "quote_start": start, "quote_end": start + len(part["text"]),
            }
        episode_audit.append({
            "episode": episode_number, "source_session_index": session_index,
            "source_turn_index": turn_index, "timestamp": source_date,
            "source_quote": text, "words": len(text.split()),
            "neutralized_assistant_turn": neutralized_turn,
        })
    proof = fixture["proof"]
    used = set(proof.get("ids", [])) | set(proof.get("add", [])) | set(proof.get("remove", []))
    used.update(proof[key] for key in ("id", "base") if key in proof)
    if not used or not used.issubset(facts):
        raise ValueError("Unbound reference fact")
    required_episodes.update(facts[key]["episode"] for key in used)
    if not required_episodes.issubset(range(len(ordered))):
        raise ValueError("Invalid required episode")
    if proof["op"] == "absent":
        all_support = " ".join(row["source_quote"] for row in episode_audit)
        if proof["missing_anchor"].casefold() in all_support.casefold():
            raise ValueError("Unsupported scope is present in an authoritative source")
    result["answer"] = reference_answer(facts, proof)
    # All session IDs are opaque, rather than advertising answer/noise identities.
    renamed = {sid: f"ku40_{index:04d}_s{i:04d}" for i, sid in enumerate(session_ids)}
    result["haystack_session_ids"] = [renamed[sid] for sid in session_ids]
    minimal_sessions = list(dict.fromkeys(renamed[ordered[i][0]] for i in sorted(required_episodes)))
    # Knowledge-update relevance includes the competing history, not just the final answer span.
    result["answer_session_ids"] = list(dict.fromkeys(renamed[sid] for sid, _ in ordered))
    for row in facts.values():
        row["session_id"] = result["haystack_session_ids"][row["source_session_index"]]
    for row in episode_audit:
        row["session_id"] = result["haystack_session_ids"][row["source_session_index"]]
    return result, {
        "question_id": result["question_id"], "source_profile_index": index,
        "source_r2_question_id": source["question_id"], "reference": proof,
        "reference_answer": result["answer"], "facts": facts, "episodes": episode_audit,
        "semantic_review_note": fixture["semantic_review_note"],
        "secondary_features": list(EXTRA_FEATURES[index]),
        "source_session_count": len(session_ids),
        "source_turn_count": sum(len(messages(s)) for s in source["haystack_sessions"]),
        "minimal_reference_session_ids": minimal_sessions,
        "retired_source_slots": [{"source_session_index": positions[sid], "source_turn_index": turn}
                                 for sid, turn in sorted(obsolete_slots)],
        "restored_support_slots": [{"source_session_index": positions[sid], "source_turn_index": turn}
                                   for sid, turn in restored_slots],
        "abstention_evidence_note": (
            "Near-miss sessions document scope discrimination, not positive answer evidence; "
            "no closed-world absence is implied by a retrieval result."
        ) if proof["op"] == "absent" else None,
    }


def profile_inventory(traces, labels):
    lookup = {}
    for name, indices in PROFILE_GROUPS.items():
        for index in indices:
            if index in lookup:
                raise ValueError("Overlapping profile groups")
            lookup[index] = name
    official = {int(row["dataset_index"]): row for row in traces if row["question_type"] == "knowledge-update"}
    if set(lookup) != set(official) or len(official) != 78:
        raise ValueError("Profile groups must cover all 78 official knowledge-update records")
    selected = {fixture["source_profile_index"] for fixture in CASES}
    if len(CASES) != 40 or len(selected) != 40:
        raise ValueError("Expected 40 unique fixtures")
    rows = []
    for index, row in sorted(official.items()):
        label = labels[row["question_id"]]["autoeval_label"]["label"]
        if not isinstance(label, bool):
            raise ValueError("Judge label must be boolean")
        rows.append({
            "source_profile_index": index, "official_question_id": row["question_id"],
            "official_question": row["question"], "historical_official_judge_correct": label,
            "primary_profile": lookup[index], "selected": index in selected,
        })
    if sum(row["historical_official_judge_correct"] for row in rows) != 35:
        raise ValueError("Unexpected historical official result")
    selected_rows = [row for row in rows if row["selected"]]
    if {row["primary_profile"] for row in selected_rows} != set(PROFILE_GROUPS):
        raise ValueError("Selected pack omits a primary profile")
    if sum(row["historical_official_judge_correct"] for row in selected_rows) != 21:
        raise ValueError("Expected 21 historical-success profiles and 19 failure profiles")
    successful_profiles = {row["primary_profile"] for row in rows if row["historical_official_judge_correct"]}
    control_profiles = {row["primary_profile"] for row in selected_rows if row["historical_official_judge_correct"]}
    if successful_profiles != control_profiles:
        raise ValueError("Each historically successful primary profile needs a control counterpart")
    return rows


def build_artifacts(project_root):
    root = project_root / "benchmark/longMemEval"
    synthetic = root / "synthetic_v2"
    official = root / "official_publication_v1/pragmos_2k"
    paths = {
        "source_r2": synthetic / "semantic_revision_2/pragmos_synthetic_longmemeval_500_v2_semantic_r2.json",
        "source_audit": synthetic / "pragmos_synthetic_longmemeval_500_v2_audit.jsonl",
        "r2_audit": synthetic / "semantic_revision_2/semantic_r2_audit.jsonl",
        "historical_trace": official / "run/lme_official_pragmos_context_full500_v1_trace.jsonl",
        "historical_judge": official / "official_judge/full_500/pragmos_gpt4o_full500_v1_results.jsonl",
        "builder": Path(__file__).resolve(),
        "authoring_fixtures": Path(__file__).with_name("PRAGMOS_knowledge_update_pack_cases.py").resolve(),
    }
    identities = {key: {"path": str(path.resolve()), "sha256": file_hash(path)} for key, path in paths.items()}
    source = json.loads(paths["source_r2"].read_text())
    source_audit = {row["source_profile_index"]: row for row in load_jsonl(paths["source_audit"])}
    r2_audit = {row["source_profile_index"]: row for row in load_jsonl(paths["r2_audit"])}
    traces = load_jsonl(paths["historical_trace"])
    labels = {row["question_id"]: row for row in load_jsonl(paths["historical_judge"])}
    inventory = profile_inventory(traces, labels)
    profiles = {row["source_profile_index"]: row for row in inventory}
    records, audits = [], []
    for fixture in CASES:
        index = fixture["source_profile_index"]
        row_audits = [source_audit[index]]
        if index in r2_audit:
            row_audits.append(r2_audit[index])
        record, audit = build_record(source[index], row_audits, fixture)
        audit.update(profiles[index])
        audit["record_sha256"] = digest(payload(record))
        records.append(record)
        audits.append(audit)
    if sum(record["question_id"].endswith("_abs") for record in records) != 6:
        raise ValueError("Expected all six unsupported-scope profiles")
    artifacts = {
        DATA_NAME: payload(records),
        "evaluation_only/case_audit.json": payload(audits),
        "evaluation_only/coverage_inventory.json": payload(inventory),
    }
    manifest = {
        "schema_version": 1, "suite": "knowledge_update_40_v1", "question_count": 40,
        "official_benchmark_score": False, "independent_holdout": False,
        "selection_policy": "profile-balanced development; not proportional or a random accuracy estimate",
        "historical_success_profile_count": 21, "historical_failure_profile_count": 19,
        "historical_success_is_not_a_synthetic_baseline_pass": True,
        "abstention_count": 6, "primary_profile_count": len(PROFILE_GROUPS),
        "answer_session_policy": "Both earlier and later source sessions are relevant, matching official update-history evaluation; minimal answer-span sessions are audited separately.",
        "primary_profile_counts": dict(sorted(Counter(row["primary_profile"] for row in audits).items())),
        "source_profile_indices": [case["source_profile_index"] for case in CASES],
        "inputs": identities,
        "artifacts": {name: digest(data) for name, data in artifacts.items()},
        "validation": {
            "preserved_session_turn_counts_dates_and_unedited_background": True,
            "reference_fact_spans_bound_to_user_sources": True,
            "reference_operations_recomputed": True,
            "opaque_session_ids": True,
            "semantic_equivalence_or_equal_difficulty_to_official_proven": False,
            "independent_semantic_review_completed": False,
        },
        "limitations": [
            "The 24 subtypes are an analyst taxonomy, not official LongMemEval labels.",
            "Authored facts/proofs validate internal consistency, not independent semantic fidelity.",
            "Only source slots and adjacent assistant acknowledgements are replaced; full distractor histories remain.",
            "Answers, profile selection and annotations are evaluation-only. Inference must not read this directory's audit files.",
        ],
    }
    for key, path in paths.items():
        if file_hash(path) != identities[key]["sha256"]:
            raise ValueError("An input changed during construction")
    artifacts["manifest.json"] = payload(manifest)
    return artifacts, manifest


def write_artifacts(output, artifacts, *, check_only=False):
    # Preflight every file so a differing frozen artifact cannot be partly overwritten.
    for name, content in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Frozen output differs; choose a new version: {path}")
        if check_only and not path.is_file():
            raise ValueError(f"Missing frozen artifact: {path}")
    if not check_only:
        for name, content in artifacts.items():
            path = output / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    artifacts, manifest = build_artifacts(args.project_root)
    write_artifacts(args.output_dir, artifacts, check_only=args.check_only)
    print(json.dumps({key: manifest[key] for key in (
        "question_count", "historical_success_profile_count", "historical_failure_profile_count",
        "abstention_count", "primary_profile_count",
    )}, indent=2))
    print(f"{'Verified' if args.check_only else 'Built'}: {args.output_dir / DATA_NAME}")


if __name__ == "__main__":
    main()
