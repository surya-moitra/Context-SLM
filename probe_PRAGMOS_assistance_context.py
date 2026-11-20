#!/usr/bin/env python3
"""Explicit component probe, NOT automatic routing or an official benchmark.

Selects records only to form a test workload. The helper receives the question
and indexed raw dialogue, never labels, references or authoring source addresses.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def memory_fingerprint(context):
    state = {
        "vector_memory": context.vector_memory,
        "raw_turns": [asdict(t) for t in context.raw_source_turns.values()],
        "graph_nodes": list(context.G.nodes(data=True)),
        "graph_edges": list(context.G.edges(keys=True, data=True)),
    }
    for name in ("entity_registry", "alias_to_entity_id", "speaker_entity_ids", "related_entity_by_owner",
                 "cache_counters", "lexical_doc_freqs", "lexical_total_doc_len", "lexical_term_cache"):
        state[name] = getattr(context, name, None)
    state["embedding_cache"] = [(key, hashlib.sha256(value.tobytes()).hexdigest())
                                for key, value in context.embedding_cache.items()]
    state["reranker_cache"] = list(context.reranker_score_cache.items())
    state["token_cache"] = list(context.token_count_cache.items())
    digest = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=True, default=str).encode())
    for i in range(context.index.ntotal):
        digest.update(context.index.reconstruct(i).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--relevance-mass", type=float, default=0.90,
                        help="Relative source-score mass retained, not calibrated relevance probability")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with Path(args.data_file).open(encoding="utf-8") as handle:
        records = json.load(handle)
    selected = {str(r["question_id"]): r for r in records if str(r["question_id"]) in args.question_ids}
    del records
    gc.collect()
    if len(selected) != len(set(args.question_ids)):
        raise ValueError("probe_question_ids_missing_or_duplicate")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pragmos-matplotlib")

    import numpy as np
    from PRAGMOS_assistance_context import AssistanceRequest, protocol_identity, run_isolated_assistance
    from PRAGMOS_benchmark_LongMemEval import longmemeval_turns
    from PRAGMOS_context_layer_org import ContextLayer

    root = Path(__file__).resolve().parent
    config = {"n_ctx": 2048, "max_tokens": 64, "seed": 42, "n_threads": 8, "n_gpu_layers": 40,
              "retrieval_device": "cpu", "top_k": 12, "min_score": 0.20,
              "relative_relevance_mass": args.relevance_mass}
    manifest = {"scope": "explicit_assistance_component_not_automatic_routing_or_official_accuracy",
                "gold_source_addresses_used": False, "hosted_or_network_calls": False,
                "protocol": protocol_identity(), "config": config, "workload": args.question_ids,
                "model": file_identity(args.model_path), "dataset": file_identity(args.data_file),
                "sources": {name: file_identity(root / name) for name in (
                    "PRAGMOS_assistance_context.py", "PRAGMOS_assistance_selection.py",
                    "PRAGMOS_assistance_boundary.py", "PRAGMOS_evidence_context.py",
                    "PRAGMOS_context_layer_org.py", "PRAGMOS_benchmark_LongMemEval.py", "PRAGMOS_composable_candidate.py",
                    "PRAGMOS_composable_execution.py", Path(__file__).name)},
                "versions": {name: version(name) for name in (
                    "llama-cpp-python", "faiss-cpu", "sentence-transformers", "torch", "numpy")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    context = ContextLayer(model_path=args.model_path, context_length=2048, n_threads=8,
                           n_gpu_layers=40, seed=42, retrieval_device="cpu")
    results = []
    try:
        context.get_local_reranker()
        context.llm("<|user|>\nReply OK.<|end|>\n<|assistant|>\n", max_tokens=4, temperature=0, seed=42)
        warm = context.llm.save_state()
        control_prompt = "<|user|>\nThe studio door is green. What color is the door? Return only its color.<|end|>\n<|assistant|>\n"
        settings = dict(max_tokens=8, temperature=0, seed=42, stop=["<|end|>"])
        control = context.llm(control_prompt, **settings)
        context.llm.load_state(warm)
        with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
            for question_id in args.question_ids:
                record = selected[question_id]
                context.reset_memory(session_id="isolated-assistance-probe")
                # Whitelist source fields before indexing; gold is read only after inference.
                inference = {key: record.get(key) for key in (
                    "haystack_sessions", "haystack_dates", "haystack_session_ids")}
                begin = time.perf_counter()
                turns = longmemeval_turns(context, inference)
                context.index_raw_turns(turns, chunk_words=160, overlap_words=32, batch_size=64)
                index_seconds = time.perf_counter() - begin
                before = memory_fingerprint(context)
                result = run_isolated_assistance(context, AssistanceRequest(record["question"]),
                                                 n_ctx=2048, max_tokens=64, seed=42, top_k=12, min_score=0.20,
                                                 relevance_mass=args.relevance_mass)
                unchanged = before == memory_fingerprint(context)
                assert unchanged, "assistance_experiment_mutated_existing_memory"
                assert context.llm.n_tokens == warm.n_tokens, "warm_token_count_changed"
                assert context.llm._seed == warm.seed, "warm_seed_changed"
                np.testing.assert_array_equal(context.llm.input_ids[:warm.n_tokens], warm.input_ids[:warm.n_tokens])
                np.testing.assert_array_equal(context.llm._scores[warm.n_tokens - 1], warm.scores[warm.n_tokens - 1])
                after_control = context.llm(control_prompt, **settings)
                assert after_control["choices"] == control["choices"], "legacy_control_answer_changed"
                context.llm.load_state(warm)
                row = {"question_id": question_id, "question": record["question"], "result": result,
                       "index_seconds": index_seconds, "memory_unchanged": unchanged,
                       "warm_state_restored": True, "legacy_control_answer_unchanged": True,
                       "reference_for_post_inference_review": record.get("answer")}
                results.append(row)
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"question_id": question_id, "status": result["status"], "answer": result["answer"],
                                  "seconds": result["elapsed_seconds"], "calls": result["model_calls"]}), flush=True)
        summary = {"processed": len(results), "complete": True, "scope": manifest["scope"],
                   "automatic_routing_enabled": False, "semantic_accuracy": None,
                   "memory_and_model_isolation_checks_passed": all(r["memory_unchanged"] and r["warm_state_restored"] for r in results),
                   "generated_candidate_count": sum(r["result"]["status"].startswith("generated_candidate") for r in results),
                   "output_limited_count": sum(r["result"].get("output_limit_reached", False) for r in results),
                   "mean_component_seconds": sum(r["result"]["elapsed_seconds"] for r in results) / len(results)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("Probe artifacts: " + str(output), flush=True)
    finally:
        context.llm.close()


if __name__ == "__main__":
    main()
