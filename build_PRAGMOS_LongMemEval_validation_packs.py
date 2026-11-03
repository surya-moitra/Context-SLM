#!/usr/bin/env python3
"""Build frozen 30-question LongMemEval V2 validation packs.

The packs contain unmodified records from the synthetic V2 dataset.  They are
development/regression gates, not official LongMemEval scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


MULTI_PREVIOUS_INDICES = (
    70, 75, 76, 78, 79, 80, 82, 83, 84, 85,
    87, 90, 91, 92, 93, 95, 96, 97, 98, 99,
)
MULTI_UNSEEN_INDICES = (
    100, 103, 164, 165, 168, 177, 211, 219, 221, 225,
)
TEMPORAL_INDICES = (
    233, 234, 241, 246, 254, 260, 261, 266, 271, 277,
    282, 285, 288, 292, 295, 299, 302, 308, 315, 316,
    320, 328, 334, 340, 341, 343, 349, 353, 358, 359,
)

EXPECTED_TEMPORAL_SUBTYPES = {
    "date-resolution": 1,
    "elapsed-duration": 12,
    "event-order": 9,
    "relative-or-recent": 8,
}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_audit(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(row["question_id"]): row for row in rows}


def selected_records(records: list[dict], indices: tuple[int, ...]) -> list[dict]:
    if len(indices) != len(set(indices)):
        raise ValueError("Validation-pack source indices must be unique")
    if any(index < 0 or index >= len(records) for index in indices):
        raise ValueError("Validation-pack source index is outside the dataset")
    return [records[index] for index in indices]


def validate_pack(
    records: list[dict],
    indices: tuple[int, ...],
    audit: dict[str, dict],
    *,
    question_type: str,
) -> dict:
    selected = selected_records(records, indices)
    if len(selected) != 30:
        raise ValueError(f"Expected 30 records, found {len(selected)}")
    if any(record.get("question_type") != question_type for record in selected):
        raise ValueError(f"Pack contains a record outside {question_type}")
    if any("_abs" in str(record.get("question_id")) for record in selected):
        raise ValueError("Positive-path validation packs cannot contain abstention cases")

    audit_rows = [audit[str(record["question_id"])] for record in selected]
    subtype_counts = Counter(row["subtype"] for row in audit_rows)
    if question_type == "temporal-reasoning":
        if dict(sorted(subtype_counts.items())) != EXPECTED_TEMPORAL_SUBTYPES:
            raise ValueError(
                "Temporal subtype allocation changed: "
                f"{dict(sorted(subtype_counts.items()))}"
            )
    return {
        "record_count": len(selected),
        "question_type": question_type,
        "abstention_count": 0,
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "source_indices": list(indices),
        "question_ids": [str(record["question_id"]) for record in selected],
        "source_record_sha256": [record_sha256(record) for record in selected],
    }


def write_pack(
    records: list[dict],
    indices: tuple[int, ...],
    validation: dict,
    *,
    source: Path,
    source_audit: Path,
    output: Path,
    manifest: Path,
    suite_name: str,
    selection_policy: dict,
) -> None:
    selected = selected_records(records, indices)
    payload = json_bytes(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest_payload = {
        "schema_version": 1,
        "suite_name": suite_name,
        "purpose": "pre_full_run_positive_path_regression_and_validation",
        "official_benchmark_score": False,
        "source_file": str(source.resolve()),
        "source_sha256": sha256_path(source),
        "source_audit_file": str(source_audit.resolve()),
        "source_audit_sha256": sha256_path(source_audit),
        "output_file": output.name,
        "output_sha256": sha256_bytes(payload),
        "selection_policy": selection_policy,
        "validation": validation,
        "publication_note": (
            "This pack contains unchanged synthetic V2 development records. "
            "It is a regression gate and is not an official LongMemEval score."
        ),
    }
    manifest.write_bytes(json_bytes(manifest_payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.source.read_text(encoding="utf-8"))
    audit = load_audit(args.audit)
    multi_indices = MULTI_PREVIOUS_INDICES + MULTI_UNSEEN_INDICES
    multi_validation = validate_pack(
        records,
        multi_indices,
        audit,
        question_type="multi-session",
    )
    temporal_validation = validate_pack(
        records,
        TEMPORAL_INDICES,
        audit,
        question_type="temporal-reasoning",
    )

    output_dir = args.output_dir
    write_pack(
        records,
        multi_indices,
        multi_validation,
        source=args.source,
        source_audit=args.audit,
        output=output_dir / "pragmos_lme_v2_multisession_validation_30_v1.json",
        manifest=output_dir / "pragmos_lme_v2_multisession_validation_30_v1_manifest.json",
        suite_name="PRAGMOS LongMemEval V2 Multi-session Validation 30 V1",
        selection_policy={
            "previously_exercised_unique_cases": 20,
            "previously_exercised_source_indices": list(MULTI_PREVIOUS_INDICES),
            "previously_unseen_cases": 10,
            "previously_unseen_source_indices": list(MULTI_UNSEEN_INDICES),
            "positive_path_only": True,
        },
    )
    write_pack(
        records,
        TEMPORAL_INDICES,
        temporal_validation,
        source=args.source,
        source_audit=args.audit,
        output=output_dir / "pragmos_lme_v2_temporal_validation_30_v1.json",
        manifest=output_dir / "pragmos_lme_v2_temporal_validation_30_v1_manifest.json",
        suite_name="PRAGMOS LongMemEval V2 Temporal Validation 30 V1",
        selection_policy={
            "coverage_balanced_across_positive_temporal_subtypes": True,
            "positive_path_only": True,
            "source_indices": list(TEMPORAL_INDICES),
        },
    )
    print(f"Wrote validation packs under: {output_dir}")
    print(json.dumps({
        "multi_session": multi_validation,
        "temporal": temporal_validation,
    }, indent=2))


if __name__ == "__main__":
    main()
