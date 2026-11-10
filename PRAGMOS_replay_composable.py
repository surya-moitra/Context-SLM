"""Replay saved retrieval, never gold-selected sources, through local inference."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from PRAGMOS_composable_candidate import _call, promotable, supported_typed_abstention
from PRAGMOS_evidence_compiler import composable_source_view, content_tokens
from PRAGMOS_grounded_computation import (
    evidence_units, query_prompt, query_schema, selection_prompt, selection_schema, validate_query,
)


def replay_sources(record, trace):
    turns = {}
    counter = 0
    for sid, date, session in zip(record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"]):
        messages = session.get("messages", session.get("turns", [])) if isinstance(session, dict) else session
        for message in messages:
            text = message.get("content", message.get("text", ""))
            if not str(text).strip(): continue
            counter += 1
            turns[str(counter)] = {"session_id": sid, "timestamp": date, "role": message["role"],
                                   "speaker": message.get("speaker", message["role"]), "text": text}
    coverage = (trace.get("evidence_compiler_binding_diagnostics") or {}).get("count_source_coverage") or {}
    identifiers = (coverage.get("source_integrity") or {}).get("restored_source_turn_ids")
    rows = ([{"source_turn_id": key} for key in identifiers] if identifiers else
            list(trace.get("answer_candidate_extractions") or []) + list(trace.get("selected_candidate_memories") or []))
    sources, diagnostics = composable_source_view(rows, turns)
    if diagnostics["unresolved_row_count"]:
        raise ValueError("Saved evidence references could not be restored")
    return sources


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-id", action="append", default=[], help="Evaluation-only ID filter; never passed to inference")
    parser.add_argument("--probe", action="store_true", help="Development-only first-slot probe, never an accuracy check")
    parser.add_argument("--verify", action="store_true", help="Fail unless every baseline error is repaired and every baseline exact answer is retained")
    parser.add_argument("--protocol", choices=["slots_v3", "typed_v4"], default="slots_v3")
    parser.add_argument("--progress", action="store_true", help="Log local selection progress without altering inference")
    return parser.parse_args()


def main():
    from llama_cpp import Llama
    args = arguments()
    output = Path(args.output)
    if output.exists(): raise SystemExit("Output already exists; choose a new version")
    records = {r["question_id"]: r for r in json.loads(Path(args.data_file).read_text())}
    traces = [json.loads(line) for line in Path(args.trace).open() if line.strip()]
    if args.question_id:
        wanted = set(args.question_id)
        traces = [row for row in traces if row["question_id"] in wanted]
        if {row["question_id"] for row in traces} != wanted:
            raise SystemExit("Requested evaluation IDs are absent from the saved trace")
    if not traces:
        raise SystemExit("Saved trace contains no evaluation cases")
    output.parent.mkdir(parents=True, exist_ok=True)
    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
        return value.hexdigest()
    identity = {"source_sha256": {name: digest(Path(__file__).parent / name) for name in (
        "PRAGMOS_benchmark_LongMemEval.py", "PRAGMOS_evidence_compiler.py", "PRAGMOS_composable_candidate.py",
        "PRAGMOS_composable_execution.py", "PRAGMOS_grounded_computation.py", "PRAGMOS_replay_composable.py")},
        "dataset_sha256": digest(args.data_file), "trace_sha256": digest(args.trace), "model_sha256": digest(args.model_path)}
    if args.protocol == "typed_v4":
        identity["source_sha256"].update({name: digest(Path(__file__).parent / name)
            for name in ("PRAGMOS_typed_claims.py", "PRAGMOS_typed_computation.py", "PRAGMOS_temporal_grounding.py")})
    identity["protocol"] = args.protocol
    llm = Llama(model_path=args.model_path, n_ctx=2048, n_threads=8, n_gpu_layers=40, seed=42, verbose=False)
    context = SimpleNamespace(llm=llm, count_tokens=lambda text: len(llm.tokenize(text.encode(), add_bos=True, special=True)))
    if args.progress:
        import PRAGMOS_composable_candidate as candidate_module
        original_call = candidate_module._call
        def logged_call(*positional, **keywords):
            value = original_call(*positional, **keywords)
            print("[selection]", json.dumps(value, ensure_ascii=True)[:300], flush=True)
            return value
        candidate_module._call = logged_call
    results = []
    try:
        for trace in traces:
            record = records[trace["question_id"]]
            question = record["question"]
            sources = replay_sources(record, trace)
            started = time.monotonic()
            if args.probe:
                audit = {"planner_calls": 0, "extractor_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
                try:
                    query = validate_query(_call(context, query_prompt(question), "planner", 2048, audit,
                                                 schema=query_schema(question), max_tokens=96), question)
                    units = [u for u in evidence_units(sources) if sources[u.source_id].role == "user"]
                    tokens = content_tokens(question)
                    units.sort(key=lambda u: -len(tokens.intersection(content_tokens(u.text))))
                    selected = []
                    for evidence in units[:2]:
                        payload = _call(context, selection_prompt(question, query, evidence, sources[evidence.source_id]),
                                        "extractor", 2048, audit, schema=selection_schema(evidence), max_tokens=256)
                        selected.append({"statement": evidence.text, "selection": payload})
                    result = {"query": query, "selections": selected, "audit": audit}
                except Exception as exc:
                    result = {"error": type(exc).__name__ + ":" + str(exc), "audit": audit}
            else:
                from PRAGMOS_benchmark_LongMemEval import evaluate_composable_candidate
                from PRAGMOS_evidence_compiler import parse_semantic_plan
                plan = trace.get("semantic_query_plan") or {}
                semantic = parse_semantic_plan(json.loads(plan.get("raw_output") or "{}"), question)
                selector = trace.get("state_selector_plan") or {}
                options = SimpleNamespace(mode="pragmos_context", pragmos_composable_mode="candidate",
                                          pragmos_composable_protocol=args.protocol,
                                          pragmos_evidence_compiler_mode="validated", pragmos_temporal_contract_mode="off",
                                          pragmos_ablate=[], n_ctx=2048, max_tokens=64)
                # Route through the production boundary; only retrieval is replayed.
                rows = [{"source_turn_id": s.turn_id, "source_session_id": s.session_id} for s in sources.values()]
                lookup = {s.turn_id: {"session_id": s.session_id, "text": s.text, "role": s.role,
                                     "speaker": s.speaker, "timestamp": s.timestamp} for s in sources.values()}
                result = evaluate_composable_candidate(context, question, options, semantic_plan=semantic,
                            query_intent=trace["query_intent"], source_rows=rows, turn_lookup=lookup,
                            selector_plan=selector, baseline_answer=trace["hypothesis"])
            accepted = promotable(result) or supported_typed_abstention(result)
            answer = result.get("answer") if accepted else trace["hypothesis"]
            row = {"question_id": trace["question_id"], "question": question, "baseline": trace["hypothesis"],
                   "prediction": answer, "candidate": result, "elapsed_seconds": time.monotonic() - started}
            if not args.probe:
                row.update({"reference": record["answer"], "exact_match": str(answer).strip() == str(record["answer"]).strip()})
            results.append(row)
            partial = output.with_suffix(".partial.json")
            with partial.open("w") as stream:
                json.dump({"complete": False, "identity": identity, "rows": results}, stream, indent=2)
            print(row["question_id"], json.dumps(result if args.probe else {"answer": answer, "status": result.get("status"),
                  "reason": result.get("reason"), "match": row["exact_match"]}), flush=True)
    finally:
        if args.progress:
            candidate_module._call = original_call
        llm.close()
        report = {"scope": "development_probe" if args.probe else "saved_retrieval_replay_not_full_benchmark",
                  "n_ctx": 2048, "identity": identity, "complete": len(results) == len(traces), "rows": results}
        if not args.probe:
            failures = [row for row in results if str(row["baseline"]).strip() != str(row["reference"]).strip()]
            correct = [row for row in results if row not in failures]
            repaired = [row for row in failures if row["exact_match"] and (
                promotable(row["candidate"]) or supported_typed_abstention(row["candidate"]))]
            report["verification"] = {
                "baseline_failure_count": len(failures), "repaired_failure_count": len(repaired),
                "baseline_correct_count": len(correct), "retained_correct_count": sum(row["exact_match"] for row in correct),
                "passed": report["complete"] and len(repaired) == len(failures) and all(row["exact_match"] for row in correct),
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    if args.verify and not report.get("verification", {}).get("passed"):
        raise SystemExit("Live saved-evidence verification failed; do not claim the cases are fixed")


if __name__ == "__main__":
    main()
