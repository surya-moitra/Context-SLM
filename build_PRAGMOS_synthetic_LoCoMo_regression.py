#!/usr/bin/env python3
"""Build a deterministic, coverage-balanced LoCoMo regression suite.

The suite is a frozen subset of the independent 1,986-question synthetic
development set. It retains each complete conversation so retrieval difficulty
is unchanged while selecting questions across every synthetic family.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SEED = 20261008
DEFAULT_SOURCE = Path(
    "benchmark/locomo/synthetic_v1/pragmos_synthetic_locomo_1986_v1.json"
)
DEFAULT_OUTPUT = Path(
    "benchmark/locomo/regression_suite_600_v1/"
    "pragmos_synthetic_locomo_regression_600_v1.json"
)
DEFAULT_MANIFEST = Path(
    "benchmark/locomo/regression_suite_600_v1/"
    "pragmos_synthetic_locomo_regression_600_v1_manifest.json"
)

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# This is intentionally coverage-balanced rather than prevalence-proportional.
# Category 3 is oversampled because it is both rare and model-dependent.
FAMILY_TARGETS = {
    "cross-session-collection-union": 10,
    "cross-session-count-distinct": 10,
    "cross-session-goal-union": 10,
    "cross-session-list-union": 10,
    "cross-session-place-union": 10,
    "cross-session-recommendation-union": 10,
    "long-history-list-union": 10,
    "speaker-set-intersection": 10,
    "cross-session-relative-date-join": 12,
    "explicit-date": 11,
    "explicit-duration": 11,
    "explicit-year": 11,
    "relative-days-ago": 11,
    "relative-last-week": 11,
    "relative-next-month": 11,
    "relative-weekday": 11,
    "relative-yesterday": 11,
    "career-inference": 10,
    "commonsense-gift-inference": 11,
    "genre-inference": 10,
    "geographic-country-inference": 11,
    "geographic-state-inference": 11,
    "multi-premise-behavioral-inference": 7,
    "preference-inference": 11,
    "tool-inference": 9,
    "advice-recall": 22,
    "causal-recall": 22,
    "collection-recall": 22,
    "direct-activity": 22,
    "direct-place": 22,
    "emotion-recall": 22,
    "goal-recall": 22,
    "membership-recall": 22,
    "object-recall": 22,
    "visual-caption-single-hop": 22,
    "adversarial-wrong-attribute": 60,
    "adversarial-wrong-speaker": 60,
}

EXPECTED_CATEGORY_COUNTS = {
    1: 80,
    2: 100,
    3: 80,
    4: 220,
    5: 120,
}


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_rank(seed: int, family: str, question_id: str) -> str:
    payload = f"{seed}|{family}|{question_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_rows(dataset: list[dict]) -> list[dict]:
    rows = []
    for sample_index, sample in enumerate(dataset):
        for qa_index, qa in enumerate(sample.get("qa") or []):
            rows.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample["sample_id"],
                    "qa_index": qa_index,
                    "qa": qa,
                    "evidence_count": len(qa.get("evidence") or []),
                }
            )
    return rows


def select_family_rows(rows: list[dict], target: int, seed: int) -> list[dict]:
    """Select extrema, conversation coverage, then deterministic remainder."""
    if target > len(rows):
        family = rows[0]["qa"]["synthetic_family"] if rows else "unknown"
        raise ValueError(
            f"Family {family!r} has only {len(rows)} rows; target is {target}"
        )
    if target == len(rows):
        return list(rows)

    family = rows[0]["qa"]["synthetic_family"]
    selected = []
    selected_ids = set()

    def add(row: dict) -> None:
        question_id = row["qa"]["question_id"]
        if question_id not in selected_ids and len(selected) < target:
            selected.append(row)
            selected_ids.add(question_id)

    evidence_counts = {row["evidence_count"] for row in rows}
    if len(evidence_counts) > 1 and target >= 2:
        for evidence_count in (min(evidence_counts), max(evidence_counts)):
            candidates = [
                row for row in rows if row["evidence_count"] == evidence_count
            ]
            add(
                min(
                    candidates,
                    key=lambda row: stable_rank(
                        seed,
                        family,
                        row["qa"]["question_id"],
                    ),
                )
            )

    by_sample = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    sample_ids = sorted(
        by_sample,
        key=lambda sample_id: stable_rank(seed, family, sample_id),
    )
    represented_samples = {row["sample_id"] for row in selected}
    for sample_id in sample_ids:
        if len(selected) >= target:
            break
        if sample_id in represented_samples:
            continue
        candidates = [
            row
            for row in by_sample[sample_id]
            if row["qa"]["question_id"] not in selected_ids
        ]
        if not candidates:
            continue
        add(
            min(
                candidates,
                key=lambda row: stable_rank(
                    seed,
                    family,
                    row["qa"]["question_id"],
                ),
            )
        )
        represented_samples.add(sample_id)

    remaining = sorted(
        (
            row
            for row in rows
            if row["qa"]["question_id"] not in selected_ids
        ),
        key=lambda row: stable_rank(seed, family, row["qa"]["question_id"]),
    )
    for row in remaining:
        add(row)
        if len(selected) >= target:
            break
    return selected


def build_regression_suite(dataset: list[dict], seed: int = DEFAULT_SEED) -> list[dict]:
    grouped = defaultdict(list)
    for row in source_rows(dataset):
        grouped[row["qa"]["synthetic_family"]].append(row)

    missing_families = set(FAMILY_TARGETS).difference(grouped)
    unexpected_families = set(grouped).difference(FAMILY_TARGETS)
    if missing_families or unexpected_families:
        raise ValueError(
            "Synthetic family mismatch: "
            f"missing={sorted(missing_families)}, "
            f"unexpected={sorted(unexpected_families)}"
        )

    selected_ids = set()
    for family, target in FAMILY_TARGETS.items():
        selected_ids.update(
            row["qa"]["question_id"]
            for row in select_family_rows(grouped[family], target, seed)
        )

    result = []
    for sample in dataset:
        subset = copy.deepcopy(sample)
        subset["qa"] = [
            copy.deepcopy(qa)
            for qa in sample.get("qa") or []
            if qa["question_id"] in selected_ids
        ]
        subset.setdefault("synthetic_metadata", {})["regression_subset"] = {
            "name": "PRAGMOS Synthetic LoCoMo Regression Suite 600 V1",
            "source_question_count": sum(len(item.get("qa") or []) for item in dataset),
            "selected_question_count": len(subset["qa"]),
            "selection_seed": seed,
        }
        result.append(subset)
    return result


def dataset_counts(dataset: list[dict]) -> tuple[Counter, Counter]:
    categories = Counter()
    families = Counter()
    for sample in dataset:
        for qa in sample.get("qa") or []:
            categories[int(qa["category"])] += 1
            families[qa["synthetic_family"]] += 1
    return categories, families


def evidence_complexity(dataset: list[dict]) -> dict[str, dict[str, int]]:
    grouped = defaultdict(list)
    for sample in dataset:
        for qa in sample.get("qa") or []:
            grouped[qa["synthetic_family"]].append(len(qa.get("evidence") or []))
    return {
        family: {
            "min_evidence_turns": min(counts),
            "max_evidence_turns": max(counts),
        }
        for family, counts in sorted(grouped.items())
    }


def validate_regression_suite(source: list[dict], subset: list[dict]) -> dict:
    errors = []
    if len(source) != len(subset):
        errors.append("conversation count changed")

    source_by_id = {
        qa["question_id"]: qa
        for sample in source
        for qa in sample.get("qa") or []
    }
    selected_ids = []
    selected_sample_ids = set()
    for source_sample, subset_sample in zip(source, subset):
        if source_sample["sample_id"] != subset_sample["sample_id"]:
            errors.append("sample order or identity changed")
            continue
        if source_sample["conversation"] != subset_sample["conversation"]:
            errors.append(f"conversation changed in {source_sample['sample_id']}")
        if subset_sample.get("qa"):
            selected_sample_ids.add(subset_sample["sample_id"])
        dialog_ids = {
            turn["dia_id"]
            for key, turns in subset_sample["conversation"].items()
            if re.fullmatch(r"session_\d+", key) and isinstance(turns, list)
            for turn in turns
        }
        for qa in subset_sample.get("qa") or []:
            selected_ids.append(qa["question_id"])
            source_qa = source_by_id.get(qa["question_id"])
            if source_qa != qa:
                errors.append(f"question content changed: {qa['question_id']}")
            if not set(qa.get("evidence") or []).issubset(dialog_ids):
                errors.append(f"invalid evidence: {qa['question_id']}")

    categories, families = dataset_counts(subset)
    if dict(categories) != EXPECTED_CATEGORY_COUNTS:
        errors.append(
            f"category counts differ: {dict(categories)} != {EXPECTED_CATEGORY_COUNTS}"
        )
    if dict(families) != FAMILY_TARGETS:
        errors.append("family counts differ from targets")
    if len(selected_ids) != 600:
        errors.append(f"expected 600 questions, found {len(selected_ids)}")
    if len(set(selected_ids)) != len(selected_ids):
        errors.append("duplicate question IDs")
    if len(selected_sample_ids) != len(source):
        errors.append("not every conversation contributes selected questions")

    source_complexity = evidence_complexity(source)
    subset_complexity = evidence_complexity(subset)
    for family in (
        "long-history-list-union",
        "cross-session-relative-date-join",
        "multi-premise-behavioral-inference",
    ):
        if source_complexity[family] != subset_complexity[family]:
            errors.append(f"evidence complexity extrema not preserved for {family}")

    if errors:
        raise ValueError("Regression suite validation failed:\n- " + "\n- ".join(errors))
    return {
        "conversation_count": len(subset),
        "question_count": len(selected_ids),
        "category_counts": dict(sorted(categories.items())),
        "category_names": CATEGORY_NAMES,
        "family_counts": dict(sorted(families.items())),
        "evidence_complexity": subset_complexity,
        "all_source_conversations_retained": True,
        "question_content_unchanged": True,
    }


def write_artifacts(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    seed: int,
) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    subset = build_regression_suite(source, seed=seed)
    validation = validate_regression_suite(source, subset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json_bytes(subset)
    output_path.write_bytes(content)
    source_categories, source_families = dataset_counts(source)
    manifest = {
        "schema_version": 1,
        "dataset_name": "PRAGMOS Synthetic LoCoMo Regression Suite 600 V1",
        "purpose": "coverage_balanced_development_regression_only",
        "official_locomo_score": False,
        "selection_seed": seed,
        "builder": Path(__file__).name,
        "builder_sha256": sha256_path(Path(__file__)),
        "source_file": str(source_path),
        "source_sha256": sha256_path(source_path),
        "source_question_count": sum(source_categories.values()),
        "source_category_counts": dict(sorted(source_categories.items())),
        "source_family_counts": dict(sorted(source_families.items())),
        "dataset_file": output_path.name,
        "dataset_sha256": sha256_bytes(content),
        "selection_policy": {
            "kind": "coverage_balanced_stratified_subset",
            "family_targets": FAMILY_TARGETS,
            "conversation_policy": (
                "retain all complete conversations and spread each family across "
                "available speaker pairs before deterministic remainder sampling"
            ),
            "complexity_policy": (
                "preserve minimum and maximum evidence-turn complexity for variable-"
                "length families"
            ),
        },
        "validation": validation,
        "aggregation_note": (
            "The unweighted mean is a regression signal, not an estimate of official "
            "LoCoMo prevalence. Report category and family metrics separately."
        ),
        "publication_note": (
            "This suite is synthetic and must not be reported as an official LoCoMo "
            "result. Use untouched official data for publication claims."
        ),
    }
    manifest_path.write_bytes(json_bytes(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_artifacts(
        source_path=args.source,
        output_path=args.output,
        manifest_path=args.manifest,
        seed=args.seed,
    )
    print(f"Wrote dataset: {args.output}")
    print(f"Wrote manifest: {args.manifest}")
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    main()
