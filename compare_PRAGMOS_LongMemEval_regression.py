#!/usr/bin/env python3
"""Compare two PRAGMOS regression traces with strict per-question gating."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_trace(path: Path) -> dict[str, dict]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"Missing question_id in {path}:{line_number}")
        if question_id in rows:
            raise ValueError(f"Duplicate question_id {question_id!r} in {path}")
        rows[question_id] = row
    return rows


def metric(row: dict, name: str) -> float:
    return float((row.get("local_metrics") or {}).get(name, 0.0))


def aggregate(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "avg_token_f1": statistics.fmean(metric(row, "token_f1") for row in rows),
        "avg_exact_match": statistics.fmean(
            metric(row, "exact_match") for row in rows
        ),
        "avg_contains_reference": statistics.fmean(
            metric(row, "contains_reference") for row in rows
        ),
    }


def compare_traces(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    *,
    expected_count: int = 200,
    tolerance: float = 0.0,
    max_question_regressions: int = 0,
) -> dict:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        raise ValueError(
            "Trace question IDs differ: "
            f"missing_from_candidate={sorted(baseline_ids - candidate_ids)}; "
            f"extra_in_candidate={sorted(candidate_ids - baseline_ids)}"
        )
    if len(baseline_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} matched questions, found {len(baseline_ids)}"
        )

    ordered_ids = sorted(baseline_ids)
    baseline_rows = [baseline[question_id] for question_id in ordered_ids]
    candidate_rows = [candidate[question_id] for question_id in ordered_ids]
    baseline_overall = aggregate(baseline_rows)
    candidate_overall = aggregate(candidate_rows)

    by_type = defaultdict(list)
    for question_id in ordered_ids:
        baseline_type = baseline[question_id].get("question_type")
        candidate_type = candidate[question_id].get("question_type")
        if baseline_type != candidate_type:
            raise ValueError(
                f"Question type changed for {question_id}: "
                f"{baseline_type!r} != {candidate_type!r}"
            )
        by_type[str(baseline_type)].append(question_id)

    type_metrics = {}
    failing_types = []
    for question_type, question_ids in sorted(by_type.items()):
        baseline_metrics = aggregate([baseline[value] for value in question_ids])
        candidate_metrics = aggregate([candidate[value] for value in question_ids])
        delta = (
            candidate_metrics["avg_token_f1"]
            - baseline_metrics["avg_token_f1"]
        )
        type_metrics[question_type] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "token_f1_delta": delta,
        }
        if delta < -tolerance:
            failing_types.append(question_type)

    regressions = []
    improvements = []
    for question_id in ordered_ids:
        baseline_f1 = metric(baseline[question_id], "token_f1")
        candidate_f1 = metric(candidate[question_id], "token_f1")
        delta = candidate_f1 - baseline_f1
        detail = {
            "question_id": question_id,
            "question_type": baseline[question_id].get("question_type"),
            "baseline_token_f1": baseline_f1,
            "candidate_token_f1": candidate_f1,
            "token_f1_delta": delta,
            "baseline_hypothesis": baseline[question_id].get("hypothesis"),
            "candidate_hypothesis": candidate[question_id].get("hypothesis"),
        }
        if delta < -tolerance:
            regressions.append(detail)
        elif delta > tolerance:
            improvements.append(detail)

    overall_delta = (
        candidate_overall["avg_token_f1"] - baseline_overall["avg_token_f1"]
    )
    failures = []
    if overall_delta < -tolerance:
        failures.append("overall_token_f1_decreased")
    if failing_types:
        failures.append("question_type_token_f1_decreased")
    if len(regressions) > max_question_regressions:
        failures.append("question_regression_limit_exceeded")

    return {
        "gate_passed": not failures,
        "gate_failures": failures,
        "policy": {
            "expected_count": expected_count,
            "tolerance": tolerance,
            "max_question_regressions": max_question_regressions,
        },
        "overall": {
            "baseline": baseline_overall,
            "candidate": candidate_overall,
            "token_f1_delta": overall_delta,
        },
        "question_types": type_metrics,
        "failing_question_types": failing_types,
        "question_regression_count": len(regressions),
        "question_improvement_count": len(improvements),
        "regressions": regressions,
        "improvements": improvements,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--max-question-regressions", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_traces(
        load_trace(args.baseline_trace),
        load_trace(args.candidate_trace),
        expected_count=args.expected_count,
        tolerance=args.tolerance,
        max_question_regressions=args.max_question_regressions,
    )
    content = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote comparison: {args.output}")
    print(
        json.dumps(
            {
                "gate_passed": report["gate_passed"],
                "gate_failures": report["gate_failures"],
                "overall_token_f1_delta": report["overall"]["token_f1_delta"],
                "failing_question_types": report["failing_question_types"],
                "question_regression_count": report[
                    "question_regression_count"
                ],
                "question_improvement_count": report[
                    "question_improvement_count"
                ],
            },
            indent=2,
        )
    )
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
