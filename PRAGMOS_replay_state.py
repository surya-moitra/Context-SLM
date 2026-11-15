"""Development replay of saved retrieval through the production state boundary.

Gold answers are read only after inference for scoring. This is not a full
benchmark, an official judge run, or proof of generalization.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from PRAGMOS_benchmark_LongMemEval import answer_variants, best_local_metrics, evaluate_state_candidate


def restore_trace_sources(record, trace):
    turns, counter = {}, 0
    for sid, stamp, session in zip(record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"]):
        messages = session.get("messages", session.get("turns", [])) if isinstance(session, dict) else session
        for message in messages:
            text = message.get("content", message.get("text", ""))
            if not str(text).strip():
                continue
            counter += 1
            turns[str(counter)] = {"session_id": sid, "timestamp": stamp, "role": message["role"],
                                   "speaker": message.get("speaker", message["role"]), "text": text}
    rows = (list(trace.get("answer_candidate_extractions") or [])
            + list(trace.get("selected_candidate_memories") or [])
            + list(trace.get("retrieval_candidate_pool") or []))
    return rows, turns


def protected_trace(trace):
    diagnostics = trace.get("evidence_compiler_binding_diagnostics") or {}
    temporal = trace.get("temporal_contract_result") or {}
    return bool((diagnostics.get("source_input_validation") or {}).get("blocked")
                or any((trace.get(field) or {}).get("status") == "complete" for field in (
                    "state_history_result", "ordinal_list_result", "typed_answer_result"))
                or temporal.get("promoted"))


def legacy_trace_result(trace):
    proof = trace.get("evidence_compiler_proof") or {}
    operation = trace.get("operation_result") or {}
    if proof.get("status") == "COMPLETE" and proof.get("executable"):
        return proof
    return operation if operation.get("status") == "complete" else None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    from llama_cpp import Llama
    from sentence_transformers import SentenceTransformer
    from PRAGMOS_context_layer_org import EMBEDDING_MODEL
    from PRAGMOS_state_views import accepted_state_candidate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-id", action="append", default=[], help="Evaluation-only filter, never passed to inference")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-protocol", choices=["bound_v3", "evidence_v4"], default="evidence_v4")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    partial = output.with_suffix(".partial.json")
    if output.exists() or partial.exists():
        raise SystemExit("Output already exists; choose a new version")
    records = {r["question_id"]: r for r in json.loads(Path(args.data_file).read_text())}
    traces = [json.loads(line) for path in args.trace for line in Path(path).open() if line.strip()]
    if len({r["question_id"] for r in traces}) != len(traces):
        raise SystemExit("Duplicate question IDs across supplied traces")
    if args.question_id:
        wanted = set(args.question_id)
        traces = [r for r in traces if r["question_id"] in wanted]
        if {r["question_id"] for r in traces} != wanted:
            raise SystemExit("Requested evaluation IDs are absent from traces")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("Limit must be positive")
        traces = traces[:args.limit]
    if not traces:
        raise SystemExit("No saved cases selected")
    root = Path(__file__).resolve().parent
    identity = {"source_sha256": {name: digest(root / name) for name in (
        "PRAGMOS_state_views.py", "PRAGMOS_state_binding.py", "PRAGMOS_state_claims.py", "PRAGMOS_state_contracts.py",
        "PRAGMOS_state_reader.py", "PRAGMOS_state_owners.py", "PRAGMOS_state_program.py", "PRAGMOS_replay_state.py", "PRAGMOS_benchmark_LongMemEval.py",
        "PRAGMOS_evidence_compiler.py", "PRAGMOS_composable_execution.py", "PRAGMOS_composable_candidate.py",
        "PRAGMOS_grounded_computation.py", "PRAGMOS_typed_claims.py", "PRAGMOS_temporal_grounding.py",
        "PRAGMOS_context_layer_org.py")},
        "dataset_sha256": digest(args.data_file), "model_sha256": digest(args.model_path),
        "trace_sha256": {str(path): digest(path) for path in args.trace},
        "configuration": {k: v for k, v in vars(args).items() if k not in {"question_id", "output", "progress"}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    identity["embedding_model"] = {"name": EMBEDDING_MODEL,
                                   "revision": getattr(encoder[0].auto_model.config, "_commit_hash", None)}
    llm = Llama(model_path=args.model_path, n_ctx=args.n_ctx, n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers, seed=args.seed, verbose=False)
    context = SimpleNamespace(llm=llm, embedder=encoder,
                              count_tokens=lambda text: len(llm.tokenize(text.encode(), add_bos=True, special=True)))
    if args.progress:
        import PRAGMOS_composable_candidate as module
        original = module._call
        def logged(*positional, **keywords):
            payload = original(*positional, **keywords)
            print("[state-selection]", json.dumps(payload, ensure_ascii=True), flush=True)
            return payload
        module._call = logged
    options = SimpleNamespace(mode="pragmos_context", pragmos_state_mode="candidate", pragmos_state_protocol=args.state_protocol,
                              n_ctx=args.n_ctx, max_tokens=64)
    report = {"scope": "saved_retrieval_development_replay_not_full_benchmark", "identity": identity,
              "complete": False, "rows": []}
    try:
        for trace in traces:
            changed = [name for name, expected in identity["source_sha256"].items()
                       if digest(root / name) != expected]
            if changed:
                raise RuntimeError("Replay source changed during this run: " + ", ".join(changed))
            record = records[trace["question_id"]]
            rows, turns = restore_trace_sources(record, trace)
            started = time.monotonic()
            result = evaluate_state_candidate(context, record["question"], options,
                                              baseline_answer=trace["hypothesis"],
                                              reference_time=record.get("question_date"),
                                              query_intent=trace["query_intent"], source_rows=rows,
                                              turn_lookup=turns, protected=protected_trace(trace), legacy_result=legacy_trace_result(trace),
                                              required_scopes=[g["text"] for g in (trace.get("query_profile") or {}).get(
                                                  "required_anchor_groups", []) if g.get("text") and g["text"] in record["question"]])
            accepted = accepted_state_candidate(result, "candidate")
            answer = result["answer"] if accepted else trace["hypothesis"]
            # References and metrics never become model inputs.
            reference = record["answer"]
            baseline_metrics = best_local_metrics(trace["hypothesis"], answer_variants(reference))
            candidate_metrics = best_local_metrics(answer, answer_variants(reference))
            item = {"question_id": record["question_id"], "question": record["question"],
                    "baseline": trace["hypothesis"], "prediction": answer, "reference": reference,
                    "baseline_exact": bool(baseline_metrics["exact_match"]),
                    "candidate_exact": bool(candidate_metrics["exact_match"]),
                    "baseline_f1": baseline_metrics["token_f1"], "candidate_f1": candidate_metrics["token_f1"],
                    "accepted": accepted, "result": result, "elapsed_seconds": time.monotonic() - started}
            report["rows"].append(item)
            partial.write_text(json.dumps(report, indent=2) + "\n")
            print(item["question_id"], json.dumps({"answer": answer, "accepted": accepted, "status": result["status"],
                                                   "reason": result["reason"], "exact": item["candidate_exact"]}), flush=True)
        report["complete"] = True
    finally:
        if args.progress:
            module._call = original
        llm.close()
        results = report["rows"]
        report["summary"] = {"processed": len(results),
                             "candidate_proven_count": sum(r["result"].get("status") == "COMPLETE" for r in results),
                             "promoted_count": sum(r["accepted"] for r in results),
                             "not_proven_reasons": dict(Counter(r["result"].get("reason") for r in results
                                                                if r["result"].get("status") != "COMPLETE")),
                             "baseline_exact_count": sum(r["baseline_exact"] for r in results),
                             "candidate_exact_count": sum(r["candidate_exact"] for r in results),
                             "baseline_exact_regressions": [r["question_id"] for r in results
                                                            if r["baseline_exact"] and not r["candidate_exact"]],
                             "note": "Lexical metrics are not semantic judge accuracy."}
        (output if report["complete"] else partial).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
