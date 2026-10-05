"""Create a stable question-type block from a frozen synthetic dataset.

The transformation changes only record order. It verifies that every serialized
record from the source occurs unchanged in the output and writes a provenance
manifest suitable for repeatable development runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


DEFAULT_SOURCE = Path(
    "benchmark/longMemEval/synthetic_v1/"
    "pragmos_synthetic_longmemeval_500_v1.json"
)
DEFAULT_OUTPUT = Path(
    "benchmark/longMemEval/synthetic_v1/"
    "pragmos_synthetic_longmemeval_500_v1_multisession_block.json"
)
DEFAULT_MANIFEST = Path(
    "benchmark/longMemEval/synthetic_v1/"
    "pragmos_synthetic_longmemeval_500_v1_multisession_block_manifest.json"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "benchmark/longMemEval/synthetic_v1/"
    "pragmos_synthetic_longmemeval_500_v1_manifest.json"
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def record_fingerprint(record: dict) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def record_set_sha256(records: list[dict]) -> str:
    fingerprints = sorted(record_fingerprint(record) for record in records)
    return sha256_bytes(("\n".join(fingerprints) + "\n").encode("ascii"))


def stable_question_type_block(
    records: list[dict],
    *,
    question_type: str,
    start_index: int,
) -> list[dict]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    selected = [
        record for record in records if record.get("question_type") == question_type
    ]
    if not selected:
        raise ValueError(f"No records have question_type={question_type!r}")
    remaining = [
        record for record in records if record.get("question_type") != question_type
    ]
    if start_index > len(remaining):
        raise ValueError(
            f"start_index {start_index} exceeds the {len(remaining)} non-selected "
            "records"
        )
    return remaining[:start_index] + selected + remaining[start_index:]


def validate_reorder(
    source_records: list[dict],
    reordered_records: list[dict],
    *,
    question_type: str,
    start_index: int,
) -> dict:
    source_ids = [str(record.get("question_id") or "") for record in source_records]
    output_ids = [str(record.get("question_id") or "") for record in reordered_records]
    if any(not question_id for question_id in source_ids):
        raise ValueError("Every source record must have a question_id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Source question IDs are not unique")
    if len(source_records) != len(reordered_records):
        raise ValueError("The reordered dataset changed the record count")
    if Counter(source_ids) != Counter(output_ids):
        raise ValueError("The reordered dataset changed the question ID set")

    source_fingerprints = Counter(record_fingerprint(record) for record in source_records)
    output_fingerprints = Counter(record_fingerprint(record) for record in reordered_records)
    if source_fingerprints != output_fingerprints:
        raise ValueError("The reordered dataset changed record content")

    selected_count = sum(
        record.get("question_type") == question_type for record in source_records
    )
    end_index_exclusive = start_index + selected_count
    block = reordered_records[start_index:end_index_exclusive]
    if len(block) != selected_count or any(
        record.get("question_type") != question_type for record in block
    ):
        raise ValueError("The requested question-type block is not contiguous")
    if any(
        record.get("question_type") == question_type
        for record in reordered_records[:start_index]
        + reordered_records[end_index_exclusive:]
    ):
        raise ValueError("Selected question types remain outside the requested block")

    source_types = Counter(record.get("question_type") for record in source_records)
    output_types = Counter(record.get("question_type") for record in reordered_records)
    if source_types != output_types:
        raise ValueError("Question-type counts changed during reordering")

    return {
        "record_count": len(reordered_records),
        "content_unchanged": True,
        "question_type_counts": dict(sorted(output_types.items())),
        "record_set_sha256": record_set_sha256(reordered_records),
        "block": {
            "question_type": question_type,
            "count": selected_count,
            "start_index_0_based": start_index,
            "end_index_inclusive_0_based": end_index_exclusive - 1,
            "end_index_exclusive_0_based": end_index_exclusive,
            "start_position_1_based": start_index + 1,
            "end_position_inclusive_1_based": end_index_exclusive,
        },
    }


def write_artifacts(
    source_records: list[dict],
    reordered_records: list[dict],
    validation: dict,
    *,
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    source_manifest_path: Path | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_payload = json_bytes(reordered_records)
    output_path.write_bytes(dataset_payload)

    manifest = {
        "schema_version": 1,
        "dataset_name": "PRAGMOS Synthetic LongMemEval V1 - Multi-session Block",
        "purpose": "development_and_regression_testing_only",
        "official_longmemeval_score": False,
        "transformation": "stable_question_type_block",
        "transformer": Path(__file__).name,
        "transformer_sha256": sha256_path(Path(__file__)),
        "source_dataset_file": str(source_path),
        "source_dataset_sha256": sha256_path(source_path),
        "source_record_set_sha256": record_set_sha256(source_records),
        "dataset_file": output_path.name,
        "dataset_sha256": sha256_bytes(dataset_payload),
        "validation": validation,
        "publication_note": (
            "This is an order-only view of the frozen synthetic V1 development "
            "set. It is not an official LongMemEval split or an independent "
            "held-out benchmark."
        ),
    }
    if source_manifest_path is not None and source_manifest_path.exists():
        manifest["source_manifest_file"] = str(source_manifest_path)
        manifest["source_manifest_sha256"] = sha256_path(source_manifest_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(json_bytes(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument("--question-type", default="multi-session")
    parser.add_argument("--start-index", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_records = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source_records, list) or not all(
        isinstance(record, dict) for record in source_records
    ):
        raise ValueError("Source dataset must be a JSON list of objects")

    reordered_records = stable_question_type_block(
        source_records,
        question_type=args.question_type,
        start_index=args.start_index,
    )
    validation = validate_reorder(
        source_records,
        reordered_records,
        question_type=args.question_type,
        start_index=args.start_index,
    )
    manifest = write_artifacts(
        source_records,
        reordered_records,
        validation,
        source_path=args.source,
        output_path=args.output,
        manifest_path=args.manifest,
        source_manifest_path=args.source_manifest,
    )

    block = validation["block"]
    print(f"Wrote reordered dataset: {args.output}")
    print(f"Wrote manifest: {args.manifest}")
    print(
        f"{block['question_type']}: {block['count']} records at zero-based "
        f"indices {block['start_index_0_based']}-"
        f"{block['end_index_inclusive_0_based']}"
    )
    print(f"Content unchanged: {validation['content_unchanged']}")
    print(f"Record-set SHA-256: {validation['record_set_sha256']}")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")


if __name__ == "__main__":
    main()
