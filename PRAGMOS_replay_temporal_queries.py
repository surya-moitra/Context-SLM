#!/usr/bin/env python3
"""Diagnostic replay of saved retrieved evidence, NOT a fresh benchmark run."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from Phi3_raw_baseline import Phi3RawChat
from PRAGMOS_benchmark_LongMemEval import (
    best_local_metrics, label_blind_inference_record, local_file_identity,
    longmemeval_turns, resolve_temporal_expression, restore_count_source_turns,
)
from PRAGMOS_semantic_query import run_query_candidate
from PRAGMOS_temporal_algebra import prepare_source_rows
from PRAGMOS_temporal_contracts import promotable_temporal_result


class TurnBuilder:
    def __init__(self):
        self.counter = 0

    def create_turn(self, **values):
        self.counter += 1
        return SimpleNamespace(turn_id=self.counter, **values)


def replay_sources(trace, record):
    """Restore only previously retrieved turns; never select gold evidence."""
    if str(trace.get("question")) != str(record.get("question")):
        raise ValueError("question_changed_between_trace_and_dataset")
    turns = longmemeval_turns(TurnBuilder(), label_blind_inference_record(record))
    lookup = {turn.turn_id: turn for turn in turns}
    selected = prepare_source_rows(trace.get("evidence_compiler_ledger") or [])
    for row in selected:
        turn = lookup.get(int(row["turn_id"])) if str(row["turn_id"]).isdigit() else None
        if (turn is None or str(turn.session_id) != str(row["session_id"])
                or row["text"].strip() not in turn.text):
            raise ValueError("saved_source_content_changed_or_identity_invalid")
    rows = [dict(source_turn_id=r["turn_id"], source_session_id=r["session_id"],
                 source_quote=r["text"]) for r in selected]
    restored, integrity = restore_count_source_turns(rows, lookup)
    if integrity["unresolved_row_count"]:
        raise ValueError("saved_source_identity_not_found_in_dataset")
    complete = {str(r["source_turn_id"]) for r in restored} == {str(t.turn_id) for t in turns}
    return restored, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--semantic-checker-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("output already exists; choose a new diagnostic file")
    data = json.loads(args.data_file.read_text())
    records = {str(row["question_id"]): row for row in data}
    wanted = set(args.question_id or [])
    traces = []
    with args.trace.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not wanted or str(row["question_id"]) in wanted:
                traces.append(row)
            if len(traces) == args.limit:
                break
    if wanted - {str(t["question_id"]) for t in traces}:
        parser.error("requested IDs not found within selected trace/limit")
    if not traces:
        parser.error("no records selected")
    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True, device="cpu")
    model = Phi3RawChat(model_path=args.model_path, n_ctx=args.n_ctx,
                       n_threads=args.n_threads, n_gpu_layers=args.n_gpu_layers, seed=args.seed)
    model.context_length = args.n_ctx
    model.encode_normalized_texts = lambda texts: embeddings.encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    identity = dict(diagnostic="recorded_evidence_replay_not_benchmark", schema_version=1,
                    model=local_file_identity(args.model_path), data=local_file_identity(args.data_file),
                    trace=local_file_identity(args.trace), config=vars(args).copy(),
                    sources={name: local_file_identity(Path(__file__).with_name(name)) for name in (
                        "PRAGMOS_semantic_query.py", "PRAGMOS_temporal_algebra.py",
                        "PRAGMOS_semantic_checker.py",
                        "PRAGMOS_temporal_contracts.py", "PRAGMOS_benchmark_LongMemEval.py",
                        "PRAGMOS_evidence_compiler.py", "PRAGMOS_temporal_grounding.py",
                        "Phi3_raw_baseline.py", "PRAGMOS_replay_temporal_queries.py")})
    identity["config"] = {key: str(value) if isinstance(value, Path) else value
                          for key, value in identity["config"].items()}
    if args.semantic_checker_path:
        from PRAGMOS_semantic_checker import semantic_model_identity
        identity["semantic_checker_identity"] = semantic_model_identity(args.semantic_checker_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        with args.output.open("x") as output:
            for index, trace in enumerate(traces, 1):
                record = records[str(trace["question_id"])]
                rows, complete = replay_sources(trace, record)
                started = time.perf_counter()
                result = run_query_candidate(model, str(trace["question"]), rows,
                    temporal_resolver=resolve_temporal_expression, reference_timestamp=record.get("question_date"),
                    source_view_complete=complete, query_intent=trace.get("query_intent"),
                    semantic_checker_path=args.semantic_checker_path,
                    state_selector=trace.get("state_selector_plan"),
                    protected=trace.get("baseline_answer_decision") in {
                        "deterministic_state_history", "deterministic_ordinal_list_recall", "deterministic_typed_answer_slot"}
                        or ((trace.get("evidence_compiler_binding_diagnostics") or {}).get("source_input_validation") or {}).get("blocked", False))
                candidate = (result.get("proof") or {}).get("answer")
                promoted = promotable_temporal_result(result)
                hypothesis = candidate if promoted else trace["hypothesis"]
                # References are first accessed after inference, for diagnostics only.
                references = record.get("answer", "")
                references = references if isinstance(references, list) else [references]
                scored = dict(question_id=trace["question_id"], baseline=trace["hypothesis"],
                    candidate=candidate, hypothesis=hypothesis, would_promote=promoted,
                    elapsed_seconds=time.perf_counter() - started, result=result,
                    local_metrics=best_local_metrics(hypothesis, references))
                output.write(json.dumps(scored, ensure_ascii=True) + "\n")
                output.flush()
                print(f"[{index}/{len(traces)}] {trace['question_id']} {result['status']} "
                      f"promote={promoted} answer={candidate!r} reason={result.get('reason')}", flush=True)
    finally:
        model.llm.close()


if __name__ == "__main__":
    main()
