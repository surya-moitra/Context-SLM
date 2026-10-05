#!/usr/bin/env python3
"""Build a frozen, coverage-balanced LongMemEval synthetic regression suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SEED = 20261004
DEFAULT_SOURCE = Path(
    "benchmark/longMemEval/synthetic_v1/"
    "pragmos_synthetic_longmemeval_500_v1.json"
)
DEFAULT_OUTPUT = Path(
    "benchmark/longMemEval/regression_suite_200_v1/"
    "pragmos_longmemeval_regression_200_v1.json"
)
DEFAULT_MANIFEST = Path(
    "benchmark/longMemEval/regression_suite_200_v1/"
    "pragmos_longmemeval_regression_200_v1_manifest.json"
)

TYPE_TARGET_COUNTS = {
    "single-session-user": 26,
    "single-session-assistant": 18,
    "single-session-preference": 10,
    "knowledge-update": 38,
    "temporal-reasoning": 50,
    "multi-session": 58,
}

SINGLE_USER_FAMILIES = (
    "user-degree",
    "user-place",
    "user-title",
    "user-date",
    "user-quantity",
    "user-model",
    "user-person",
    "user-duration",
)
SINGLE_USER_ABSTENTION_FAMILIES = (
    "near-entity-count",
    "missing-related-place",
    "missing-attribute",
    "underspecified-date",
    "activity-mismatch",
    "relation-mismatch",
)
ASSISTANT_FAMILIES = (
    "assistant-ordinal-list",
    "assistant-recommendation-name",
    "assistant-process",
    "assistant-contact-detail",
    "assistant-visual-attribute",
    "assistant-multi-item",
)
PREFERENCE_FAMILIES = tuple(
    f"preference-template-{index}" for index in range(1, 9)
)
UPDATE_ATTRIBUTES = (
    "favorite-weekend-animal",
    "preferred-morning-drink",
    "home-city",
    "work-laptop",
    "job-title",
    "favorite-running-route",
    "music-subscription",
    "weekly-class",
)
UPDATE_MODES = ("current", "previous", "original")
UPDATE_ABSTENTION_FAMILIES = (
    "update-near-entity-abstention",
    "update-missing-attribute-abstention",
    "update-related-person-abstention",
)
TEMPORAL_FAMILIES = (
    "temporal-date-difference",
    "temporal-relative-weeks",
    "temporal-event-order",
    "temporal-adjacent-day",
    "temporal-clock-duration",
    "temporal-exact-date",
)
TEMPORAL_ABSTENTION_FAMILIES = (
    "temporal-missing-endpoint",
    "temporal-undated-events",
    "temporal-vague-date",
)
MULTI_FAMILIES = (
    "multi-count",
    "multi-count-distinct",
    "multi-sum-money",
    "multi-sum-duration",
    "multi-average",
    "multi-difference",
    "multi-argmax",
    "multi-total-distance",
    "multi-ratio",
    "multi-argmin",
)
MULTI_ABSTENTION_FAMILIES = (
    "multi-missing-operand",
    "multi-near-entity-abstention",
    "multi-planned-not-completed-abstention",
    "multi-missing-measurement-abstention",
)

# These questions had zero local token F1 in the first complete multi-session
# synthetic run. Retaining them makes later LoCoMo work prove that it did not
# reopen already-observed LongMemEval weaknesses.
HISTORICAL_MULTI_FAILURE_INDICES = {
    70,
    71,
    81,
    90,
    91,
    100,
    101,
    106,
    110,
    111,
    120,
    121,
    164,
    169,
    170,
    171,
    174,
    180,
    181,
    190,
    191,
    196,
    199,
    200,
    201,
    210,
    211,
    216,
    220,
    221,
    226,
}

# These update questions exposed previous/original selector failures before the
# state-timeline fixes. They remain regression anchors even when now answered.
HISTORICAL_UPDATE_FAILURE_INDICES = {
    *range(404, 414),
    422,
    424,
    425,
    426,
    427,
    429,
    *range(430, 436),
}

FIX_VALIDATION_ANCHOR_INDICES = {
    235,
    236,
    242,
    248,
    360,
    404,
    406,
    414,
    438,
    439,
    440,
    441,
    444,
    448,
    449,
    450,
    455,
    456,
}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_key(seed: int, source_index: int, question_id: str) -> str:
    value = f"{seed}:{source_index}:{question_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def regression_family(question_type: str, ordinal: int, abstention: bool) -> str:
    if question_type == "single-session-user":
        families = (
            SINGLE_USER_ABSTENTION_FAMILIES if abstention else SINGLE_USER_FAMILIES
        )
        return families[ordinal % len(families)]
    if question_type == "single-session-assistant":
        return ASSISTANT_FAMILIES[ordinal % len(ASSISTANT_FAMILIES)]
    if question_type == "single-session-preference":
        return PREFERENCE_FAMILIES[ordinal % len(PREFERENCE_FAMILIES)]
    if question_type == "knowledge-update":
        if abstention:
            return UPDATE_ABSTENTION_FAMILIES[
                ordinal % len(UPDATE_ABSTENTION_FAMILIES)
            ]
        mode = UPDATE_MODES[(ordinal // len(UPDATE_ATTRIBUTES)) % len(UPDATE_MODES)]
        attribute = UPDATE_ATTRIBUTES[ordinal % len(UPDATE_ATTRIBUTES)]
        return f"update-{mode}-{attribute}"
    if question_type == "temporal-reasoning":
        families = TEMPORAL_ABSTENTION_FAMILIES if abstention else TEMPORAL_FAMILIES
        return families[ordinal % len(families)]
    if question_type == "multi-session":
        families = MULTI_ABSTENTION_FAMILIES if abstention else MULTI_FAMILIES
        return families[ordinal % len(families)]
    raise ValueError(f"Unsupported question type: {question_type!r}")


def annotate_records(records: list[dict]) -> list[dict]:
    type_ordinals: Counter = Counter()
    annotated = []
    for source_index, record in enumerate(records):
        question_type = str(record.get("question_type") or "")
        ordinal = type_ordinals[question_type]
        type_ordinals[question_type] += 1
        question_id = str(record.get("question_id") or "")
        abstention = "_abs" in question_id
        annotated.append(
            {
                "source_index": source_index,
                "question_id": question_id,
                "question_type": question_type,
                "type_ordinal": ordinal,
                "abstention": abstention,
                "family": regression_family(question_type, ordinal, abstention),
                "record": record,
            }
        )
    return annotated


def selection_reasons(row: dict) -> list[str]:
    source_index = row["source_index"]
    reasons = []
    if row["abstention"]:
        reasons.append("all_abstention_cases")
    if source_index in HISTORICAL_MULTI_FAILURE_INDICES:
        reasons.append("historical_multi_session_failure")
    if source_index in HISTORICAL_UPDATE_FAILURE_INDICES:
        reasons.append("historical_knowledge_update_failure")
    if source_index in FIX_VALIDATION_ANCHOR_INDICES:
        reasons.append("prior_fix_validation_anchor")
    return reasons


def select_regression_rows(records: list[dict], seed: int = DEFAULT_SEED) -> list[dict]:
    annotated = annotate_records(records)
    by_type = defaultdict(list)
    for row in annotated:
        by_type[row["question_type"]].append(row)

    if set(by_type) != set(TYPE_TARGET_COUNTS):
        raise ValueError(
            "Question types do not match the frozen suite profile: "
            f"{sorted(by_type)}"
        )

    selected = []
    for question_type, target in TYPE_TARGET_COUNTS.items():
        candidates = by_type[question_type]
        chosen_by_index = {
            row["source_index"]: row
            for row in candidates
            if selection_reasons(row)
        }
        if len(chosen_by_index) > target:
            raise ValueError(
                f"Mandatory {question_type} rows exceed target: "
                f"{len(chosen_by_index)} > {target}"
            )

        positive_families = {
            row["family"] for row in candidates if not row["abstention"]
        }
        represented = {
            row["family"] for row in chosen_by_index.values() if not row["abstention"]
        }
        for family in sorted(positive_families - represented):
            family_candidates = [
                row
                for row in candidates
                if not row["abstention"]
                and row["family"] == family
                and row["source_index"] not in chosen_by_index
            ]
            row = min(
                family_candidates,
                key=lambda item: stable_key(
                    seed,
                    item["source_index"],
                    item["question_id"],
                ),
            )
            chosen_by_index[row["source_index"]] = row

        family_counts = Counter(
            row["family"]
            for row in chosen_by_index.values()
            if not row["abstention"]
        )
        remaining = [
            row
            for row in candidates
            if not row["abstention"]
            and row["source_index"] not in chosen_by_index
        ]
        while len(chosen_by_index) < target:
            row = min(
                remaining,
                key=lambda item: (
                    family_counts[item["family"]],
                    stable_key(seed, item["source_index"], item["question_id"]),
                ),
            )
            remaining.remove(row)
            chosen_by_index[row["source_index"]] = row
            family_counts[row["family"]] += 1

        selected.extend(chosen_by_index.values())

    return sorted(selected, key=lambda row: row["source_index"])


def validate_selection(records: list[dict], selected: list[dict]) -> dict:
    errors = []
    selected_indices = [row["source_index"] for row in selected]
    if len(selected) != 200:
        errors.append(f"expected 200 rows, found {len(selected)}")
    if len(set(selected_indices)) != len(selected_indices):
        errors.append("source indices are not unique")

    type_counts = Counter(row["question_type"] for row in selected)
    if dict(type_counts) != TYPE_TARGET_COUNTS:
        errors.append(f"question type counts differ: {dict(type_counts)}")

    annotated = annotate_records(records)
    expected_abstentions = {
        row["source_index"] for row in annotated if row["abstention"]
    }
    missing_abstentions = expected_abstentions.difference(selected_indices)
    if missing_abstentions:
        errors.append(f"missing abstention indices: {sorted(missing_abstentions)}")

    required = (
        HISTORICAL_MULTI_FAILURE_INDICES
        | HISTORICAL_UPDATE_FAILURE_INDICES
        | FIX_VALIDATION_ANCHOR_INDICES
    )
    missing_required = required.difference(selected_indices)
    if missing_required:
        errors.append(f"missing required regression anchors: {sorted(missing_required)}")

    for question_type in TYPE_TARGET_COUNTS:
        available = {
            row["family"]
            for row in annotated
            if row["question_type"] == question_type and not row["abstention"]
        }
        covered = {
            row["family"]
            for row in selected
            if row["question_type"] == question_type and not row["abstention"]
        }
        if available != covered:
            errors.append(
                f"positive family coverage differs for {question_type}: "
                f"missing={sorted(available - covered)}"
            )

    if errors:
        raise ValueError("Regression suite validation failed:\n- " + "\n- ".join(errors))

    return {
        "record_count": len(selected),
        "question_type_counts": dict(sorted(type_counts.items())),
        "abstention_count": sum(row["abstention"] for row in selected),
        "positive_family_count": len(
            {row["family"] for row in selected if not row["abstention"]}
        ),
        "family_counts": dict(
            sorted(Counter(row["family"] for row in selected).items())
        ),
        "source_indices": selected_indices,
    }


def write_artifacts(
    records: list[dict],
    selected: list[dict],
    validation: dict,
    *,
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    seed: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_content = json_bytes([row["record"] for row in selected])
    output_path.write_bytes(subset_content)
    manifest = {
        "schema_version": 1,
        "suite_name": "PRAGMOS LongMemEval Synthetic Regression Suite 200 V1",
        "purpose": "cross_benchmark_core_regression_gate",
        "official_benchmark_score": False,
        "seed": seed,
        "selector": Path(__file__).name,
        "selector_sha256": sha256_path(Path(__file__)),
        "source_file": str(source_path),
        "source_sha256": sha256_path(source_path),
        "output_file": output_path.name,
        "output_sha256": sha256_bytes(subset_content),
        "selection_policy": {
            "all_synthetic_abstentions": True,
            "historical_failures_retained": True,
            "prior_fix_validation_anchors_retained": True,
            "all_positive_subfamilies_covered": True,
            "question_type_targets": TYPE_TARGET_COUNTS,
        },
        "validation": validation,
        "records": [
            {
                "subset_index": subset_index,
                "source_index": row["source_index"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "regression_family": row["family"],
                "abstention": row["abstention"],
                "selection_reasons": selection_reasons(row)
                or ["balanced_family_coverage"],
            }
            for subset_index, row in enumerate(selected)
        ],
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
    records = json.loads(args.source.read_text(encoding="utf-8"))
    selected = select_regression_rows(records, seed=args.seed)
    validation = validate_selection(records, selected)
    manifest = write_artifacts(
        records,
        selected,
        validation,
        source_path=args.source,
        output_path=args.output,
        manifest_path=args.manifest,
        seed=args.seed,
    )
    print(f"Wrote suite: {args.output}")
    print(f"Wrote manifest: {args.manifest}")
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    main()
