"""Offline comparison of logged shadow candidates; never imported by inference."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean

from PRAGMOS_benchmark_LongMemEval import answer_variants, best_local_metrics
from PRAGMOS_composable_candidate import promotable, supported_typed_abstention


def projected_answer(trace):
    baseline = trace["hypothesis"]
    candidate = copy.deepcopy(trace.get("composable_execution_result") or {})
    if candidate.get("promoted") or candidate.get("abstention_applied"):
        raise ValueError("This utility requires unpromoted shadow results")
    if candidate.get("status") not in {None, "NOT_APPLICABLE"} and not candidate.get("shadow"):
        raise ValueError("Candidate is not marked as shadow")
    candidate.pop("shadow", None)
    if promotable(candidate):
        return candidate["answer"], "would_replace_answer"
    if supported_typed_abstention(candidate):
        return "I do not know.", "would_apply_supported_abstention"
    return baseline, "retain_existing_answer"


def build_report(records, traces, manifest, controls=None):
    config = manifest["config"]
    if config.get("pragmos_composable_mode") != "shadow" or config.get("pragmos_composable_protocol") != "typed_v4":
        raise ValueError("Expected a typed_v4 shadow manifest")
    if config.get("pragmos_temporal_contract_mode", "off") != "off":
        raise ValueError("Paused temporal experiment must remain off")
    data = {row["question_id"]: row for row in records}
    workload = [row["question_id"] for row in manifest["workload"]]
    identifiers = [row["question_id"] for row in traces]
    if len(data) != len(records) or len(set(workload)) != len(workload) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate question IDs")
    if not set(identifiers).issubset(workload) or not set(workload).issubset(data):
        raise ValueError("Trace, workload and dataset do not match")
    if not traces:
        raise ValueError("No completed question traces")
    rows = []
    for trace in traces:
        record = data[trace["question_id"]]
        answer, action = projected_answer(trace)
        references = answer_variants(record["answer"])
        old = best_local_metrics(trace["hypothesis"], references)
        new = best_local_metrics(answer, references)
        candidate = trace.get("composable_execution_result") or {}
        rows.append({"question_id": record["question_id"], "question": record["question"],
                     "reference": record["answer"], "existing_answer": trace["hypothesis"],
                     "projected_candidate_answer": answer, "action": action,
                     "existing_token_f1": old["token_f1"], "projected_token_f1": new["token_f1"],
                     "existing_exact_match": old["exact_match"], "projected_exact_match": new["exact_match"],
                     "candidate_status": candidate.get("status"), "candidate_reason": candidate.get("reason")})
    old_controls = {row["question_id"]: row for row in (controls or {}).get("rows", [])}
    control_regressions = []
    for row in rows:
        prior = old_controls.get(row["question_id"])
        if prior:
            prior_score = best_local_metrics(prior["prediction"], answer_variants(data[row["question_id"]]["answer"]))
            if row["projected_token_f1"] + 1e-12 < prior_score["token_f1"]:
                control_regressions.append(row["question_id"])
    count = len(rows)
    return {
        "scope": "offline_projected_candidate_scores_from_shadow_not_actual_candidate_run_or_llm_judge",
        "processed": count, "target": len(workload), "complete": set(identifiers) == set(workload),
        "existing_mean_token_f1": mean(row["existing_token_f1"] for row in rows),
        "projected_mean_token_f1": mean(row["projected_token_f1"] for row in rows),
        "existing_exact_matches": sum(int(row["existing_exact_match"]) for row in rows),
        "projected_exact_matches": sum(int(row["projected_exact_match"]) for row in rows),
        "would_change_answers": sum(row["action"] != "retain_existing_answer" for row in rows),
        "f1_regressions_vs_existing": [row["question_id"] for row in rows if row["projected_token_f1"] + 1e-12 < row["existing_token_f1"]],
        "previous_v3_controls_checked": sum(row["question_id"] in old_controls for row in rows),
        "projected_f1_regressions_vs_previous_v3_controls": control_regressions,
        "semantic_accuracy_measured": False, "rows": rows,
    }


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--controls")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if digest(args.data_file) != manifest["data_identity"]["sha256"]:
        raise ValueError("Dataset identity differs from the inference manifest")
    for name in ("PRAGMOS_composable_candidate.py", "PRAGMOS_benchmark_LongMemEval.py"):
        if digest(Path(__file__).parent / name) != manifest["source_identities"][name]["sha256"]:
            raise ValueError("Analysis helper contracts changed since inference: " + name)
    traces = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line.strip()]
    records = json.loads(Path(args.data_file).read_text())
    controls = json.loads(Path(args.controls).read_text()) if args.controls else None
    report = build_report(records, traces, manifest, controls)
    report["identity"] = {"dataset_sha256": digest(args.data_file), "trace_sha256": digest(args.trace),
                          "manifest_sha256": digest(args.manifest), "analyzer_sha256": digest(__file__),
                          "controls_sha256": digest(args.controls) if args.controls else None,
                          "inference_sources": manifest["source_identities"]}
    output = Path(args.output)
    if output.resolve() in {Path(path).resolve() for path in (args.data_file, args.trace, args.manifest, *([args.controls] if args.controls else []))}:
        raise ValueError("Analysis cannot overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"rows", "identity"}}, indent=2))
    print("Wrote offline shadow comparison:", output)


if __name__ == "__main__":
    main()
