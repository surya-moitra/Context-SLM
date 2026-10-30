#!/usr/bin/env python3
"""Derive compact context and memory/graph metrics from benchmark traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PRAGMOS_benchmark_LoCoMo import canonical_locomo_source_conversation
from PRAGMOS_benchmark_LongMemEval import (
    aggregate_longmemeval_retrieval_metrics,
    longmemeval_turn_retrieval_metrics,
)
from PRAGMOS_observability import (
    aggregate_measurement_traces,
    measurement_trace_from_row,
)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def canonical_longmemeval_source_conversation(record: dict[str, Any]) -> str:
    sessions = record.get("haystack_sessions") or []
    session_ids = record.get("haystack_session_ids") or []
    dates = record.get("haystack_dates") or []
    blocks = []
    for index, session in enumerate(sessions):
        session_id = session_ids[index] if index < len(session_ids) else index
        timestamp = dates[index] if index < len(dates) else None
        lines = [f"[session_id={session_id}; timestamp={timestamp or 'unknown'}]"]
        messages = (
            session.get("messages") or session.get("turns") or []
            if isinstance(session, dict)
            else session
        )
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict):
                continue
            speaker = str(message.get("speaker") or message.get("role") or "unknown")
            text = str(message.get("content", message.get("text", ""))).strip()
            if text:
                lines.append(f"{speaker}: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def source_texts_by_question(
    records: Iterable[dict[str, Any]],
    family: str,
) -> dict[str, str]:
    result = {}
    if family == "longmemeval":
        for record in records:
            question_id = str(record.get("question_id") or "")
            if question_id:
                result[question_id] = canonical_longmemeval_source_conversation(record)
        return result
    if family != "locomo":
        raise ValueError(f"Unsupported benchmark family: {family}")
    for sample_index, sample in enumerate(records):
        sample_id = str(sample.get("sample_id") or f"sample-{sample_index}")
        source = canonical_locomo_source_conversation(sample)
        for qa_index, qa in enumerate(sample.get("qa") or []):
            question_id = str(
                qa.get("question_id") or f"{sample_id}:q{qa_index + 1:04d}"
            )
            result[question_id] = source
    return result


def infer_benchmark_family(trace_rows: Sequence[dict[str, Any]]) -> str:
    if not trace_rows:
        raise ValueError("Trace is empty")
    return "locomo" if "category" in trace_rows[0] else "longmemeval"


def load_vocab_token_counter(model_path: Path) -> Callable[[str], int]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    from llama_cpp import Llama

    tokenizer = Llama(
        model_path=str(model_path),
        vocab_only=True,
        n_ctx=512,
        n_gpu_layers=0,
        verbose=False,
    )

    def count_tokens(text: str) -> int:
        return len(tokenizer.tokenize(str(text).encode("utf-8"), add_bos=False))

    return count_tokens


def derive_measurement_traces(
    trace_rows: Sequence[dict[str, Any]],
    *,
    family: str,
    data_records: Sequence[dict[str, Any]] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    if all(isinstance(row.get("measurement_trace"), dict) for row in trace_rows):
        return [row["measurement_trace"] for row in trace_rows]
    if data_records is None or token_counter is None:
        raise ValueError(
            "Legacy traces require --data-file and --model-path for exact source token counts"
        )
    source_texts = source_texts_by_question(data_records, family)
    token_cache: dict[str, int] = {}
    measurements = []
    for row in trace_rows:
        question_id = str(row.get("question_id") or "")
        if question_id not in source_texts:
            raise ValueError(f"Question {question_id!r} is missing from the data file")
        source_text = source_texts[question_id]
        if source_text not in token_cache:
            token_cache[source_text] = token_counter(source_text)
        measurements.append(
            measurement_trace_from_row(
                row,
                source_conversation_tokens=token_cache[source_text],
            )
        )
    return measurements


def derive_longmemeval_turn_retrieval_metrics(
    trace_rows: Sequence[dict[str, Any]],
    oracle_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Join private answer-turn labels after inference and aggregate retrieval."""
    oracle_by_id = {
        str(record.get("question_id") or ""): record
        for record in oracle_records
        if record.get("question_id") is not None
    }
    candidate_rows = []
    final_rows = []
    missing_question_ids = []
    for trace in trace_rows:
        question_id = str(trace.get("question_id") or "")
        oracle_record = oracle_by_id.get(question_id)
        if oracle_record is None:
            missing_question_ids.append(question_id)
            continue
        candidate_rows.append(
            longmemeval_turn_retrieval_metrics(
                oracle_record,
                trace.get("retrieval_candidate_pool") or [],
                ranking_name="candidate_pool_turns",
                question_id=question_id,
            )
        )
        final_rows.append(
            longmemeval_turn_retrieval_metrics(
                oracle_record,
                list(trace.get("final_memories") or [])
                + list(trace.get("graph_evidence") or []),
                ranking_name="final_evidence_turns",
                question_id=question_id,
            )
        )
    if missing_question_ids:
        raise ValueError(
            "Turn-retrieval oracle is missing question IDs: "
            + ", ".join(missing_question_ids[:10])
        )
    return {
        "metric_version": 1,
        "label_join_stage": "post_inference_analysis_only",
        "oracle_question_count": len(oracle_by_id),
        "trace_question_count": len(trace_rows),
        "candidate_pool": aggregate_longmemeval_retrieval_metrics(candidate_rows),
        "final_evidence": aggregate_longmemeval_retrieval_metrics(final_rows),
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--benchmark-family",
        choices=("auto", "longmemeval", "locomo"),
        default="auto",
    )
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--retrieval-oracle",
        type=Path,
        help=(
            "Optional private LongMemEval oracle used only after inference to "
            "measure answer-turn retrieval."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    trace_rows = load_json_or_jsonl(args.trace)
    family = (
        infer_benchmark_family(trace_rows)
        if args.benchmark_family == "auto"
        else args.benchmark_family
    )
    needs_derivation = not all(
        isinstance(row.get("measurement_trace"), dict) for row in trace_rows
    )
    data_records = None
    token_counter = None
    if needs_derivation:
        if args.data_file is None or args.model_path is None:
            raise SystemExit(
                "analysis error: legacy traces require --data-file and --model-path"
            )
        data_records = load_json_or_jsonl(args.data_file)
        token_counter = load_vocab_token_counter(args.model_path.resolve())
    try:
        measurements = derive_measurement_traces(
            trace_rows,
            family=family,
            data_records=data_records,
            token_counter=token_counter,
        )
        aggregate = aggregate_measurement_traces(measurements)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"analysis error: {error}") from error
    report = {
        "schema_version": 1,
        "benchmark_family": family,
        "trace_path": str(args.trace.resolve()),
        "processed": len(trace_rows),
        **aggregate,
    }
    if args.retrieval_oracle is not None:
        if family != "longmemeval":
            raise SystemExit(
                "analysis error: --retrieval-oracle is supported only for LongMemEval"
            )
        try:
            oracle_records = load_json_or_jsonl(args.retrieval_oracle)
            report["longmemeval_turn_retrieval_metrics"] = (
                derive_longmemeval_turn_retrieval_metrics(
                    trace_rows,
                    oracle_records,
                )
            )
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(f"analysis error: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote trace measurement report: {args.output}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
