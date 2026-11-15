"""Optional offline retrieval-model equivalence and timing probe, not a QA eval."""

import json
import os
import time
import unittest


@unittest.skipUnless(os.environ.get("PRAGMOS_NATIVE_RETRIEVAL_TEST") == "1",
                     "Set PRAGMOS_NATIVE_RETRIEVAL_TEST=1 for installed CPU retrieval models")
class NativeRetrievalExecutionTests(unittest.TestCase):
    def test_serial_and_batched_native_rankings(self):
        # Match the production module's native-library initialization order.
        from PRAGMOS_context_layer_org import ContextLayer
        import faiss
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from test_PRAGMOS_context_layer_latency import cache_only_context

        context = cache_only_context()
        context.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
        context.enable_dense_retrieval = context.enable_lexical_retrieval = context.enable_graph_retrieval = True
        context.execution_policy = "reference"
        context.embedding_cache_size = context.reranker_cache_size = 0
        context.reset_memory("native-execution")
        context.index = faiss.IndexFlatL2(384)
        print("[native-execution] CPU models loaded; indexing", flush=True)
        for person in ("Asha", "Ben", "Dara", "Milo"):
            for index, text in enumerate(("bought a green lantern", "painted the studio door",
                                           "spent $17 on supplies", "completed the project yesterday",
                                           "planned a trip for next month", "recommended a desk and easel")):
                quote = f"{person} {text}."
                context.embed_and_store(quote, metadata={"source_quote": quote,
                    "source_session_id": person, "speaker": person, "role": "user",
                    "source_turn_ids": [person + str(index)]})
        queries = ["What did Asha buy?", "What did Ben paint?", "What did Dara spend?",
                   "When did Milo complete the project?", "Where does Asha plan to travel?",
                   "What did Ben recommend?", "What did Asha buy?"]
        print("[native-execution] indexed; warming serial retrieval", flush=True)
        # Match production's lazy load order after corpus ingestion.
        context.local_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu", local_files_only=True)
        # Warm the models without allowing answer/query score caches to mask work.
        context.retrieve_relevant_memories(queries[0], top_k=4, min_score=0, graph_evidence=[])
        started = time.perf_counter()
        reference = [context.retrieve_relevant_memories(q, top_k=4, min_score=0, graph_evidence=[]) for q in queries]
        serial_seconds = time.perf_counter() - started
        print("[native-execution] serial retrieval completed; batching", flush=True)
        context.execution_policy = "optimized"
        started = time.perf_counter()
        candidate = context.retrieve_relevant_memories_many(queries, top_k=4, min_score=0, graph_evidence=[])
        batch_seconds = time.perf_counter() - started
        max_delta = 0.0
        for old_group, new_group in zip(reference, candidate):
            self.assertEqual([r["memory_id"] for r in old_group], [r["memory_id"] for r in new_group])
            for old, new in zip(old_group, new_group):
                self.assertEqual(old["source_turn_ids"], new["source_turn_ids"])
                for field in ("score", "hybrid_score", "dense_score", "lexical_score", "rerank_score"):
                    delta = abs(old[field] - new[field])
                    max_delta = max(max_delta, delta)
                    self.assertLessEqual(delta, 1e-5)
        print(json.dumps({"scope": "retrieval_microbenchmark_not_end_to_end_or_semantic_accuracy",
            "queries": len(queries), "documents": len(context.vector_memory),
            "serial_seconds": serial_seconds, "batched_seconds": batch_seconds,
            "max_score_difference": max_delta, "identical_ranked_source_ids": True}))


@unittest.skipUnless(os.environ.get("PRAGMOS_NATIVE_LLM_EXECUTION_TEST_MODEL"),
                     "Set PRAGMOS_NATIVE_LLM_EXECUTION_TEST_MODEL for two bounded local planner calls")
class NativePlannerOverlapTests(unittest.TestCase):
    def test_real_planner_and_cpu_indexing_overlap_preserve_outputs(self):
        from PRAGMOS_context_layer_org import ContextLayer
        from PRAGMOS_evidence_compiler import plan_query_with_local_slm
        from PRAGMOS_execution import index_with_planner
        import numpy as np

        context = ContextLayer(
            model_path=os.environ["PRAGMOS_NATIVE_LLM_EXECUTION_TEST_MODEL"],
            context_length=2048, n_threads=8, n_gpu_layers=40,
            execution_policy="optimized", retrieval_device="cpu",
            embedding_cache_size=0,
        )
        question = "What total did I spend on studio supplies?"
        runs = []
        try:
            for overlap in (False, True):
                context.reset_memory("native-overlap")
                context.llm.reset()
                turns = [context.create_turn(role="user", text=f"I spent ${index + 11} on studio supplies.",
                         session_id="studio", turn_id=index, timestamp="2027-01-03T10:00:00Z")
                         for index in range(32)]
                _, planned, timing = index_with_planner(
                    lambda: context.index_raw_turns(turns, batch_size=16),
                    lambda: plan_query_with_local_slm(context, question), overlap=overlap,
                )
                vectors = context.index.reconstruct_n(0, context.index.ntotal).copy()
                runs.append((planned.as_dict(), list(context.vector_memory), vectors, timing))
            self.assertEqual(runs[0][0], runs[1][0])
            self.assertEqual(runs[0][0]["status"], "valid")
            self.assertEqual(runs[0][1], runs[1][1])
            np.testing.assert_array_equal(runs[0][2], runs[1][2])
            print(json.dumps({"scope": "native_planner_overlap_probe_not_full_qa_accuracy",
                "identical_planner_output_and_index": True,
                "serial": runs[0][3], "overlapped": runs[1][3]}), flush=True)
        finally:
            context.llm.close()


if __name__ == "__main__":
    unittest.main()
