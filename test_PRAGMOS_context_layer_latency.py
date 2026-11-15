import unittest
import copy
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np

from PRAGMOS_context_layer_org import ContextLayer


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            base = float(sum(ord(character) for character in text) % 17 + 1)
            vectors.append([base, base + 1.0, base + 2.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeReranker:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return np.asarray(
            [len(query) * 0.1 + len(document) * 0.01 for query, document in pairs],
            dtype=np.float32,
        )


def cache_only_context():
    context = ContextLayer.__new__(ContextLayer)
    context.embedder = FakeEmbedder()
    context.embedding_cache_size = 8
    context.reranker_cache_size = 8
    context.embedding_cache = OrderedDict()
    context.reranker_score_cache = OrderedDict()
    context.cache_counters = {
        "embedding_hits": 0,
        "embedding_misses": 0,
        "reranker_hits": 0,
        "reranker_misses": 0,
    }
    context.enable_reranker = True
    context.local_reranker = FakeReranker()
    context.local_reranker_load_attempted = True
    return context


class ContextLayerLatencyTests(unittest.TestCase):
    def test_token_cache_is_bounded_exact_and_does_not_cache_failures(self):
        context = cache_only_context()
        context.execution_policy = "optimized"
        context.llm = SimpleNamespace(tokenize=Mock(side_effect=lambda text, **kwargs: list(text)))
        self.assertEqual(context.count_tokens("some text"), 9)
        self.assertEqual(context.count_tokens("some text"), 9)
        self.assertEqual(context.llm.tokenize.call_count, 1)
        for index in range(300):
            context.count_tokens(str(index))
        self.assertEqual(len(context.token_count_cache), 256)
        context.llm.tokenize.side_effect = [RuntimeError("temporary"), [1]]
        self.assertEqual(context.count_tokens("temporary"), 3)
        self.assertEqual(context.count_tokens("temporary"), 1)
        context.llm = SimpleNamespace(tokenize=Mock(return_value=[1, 2]))
        self.assertEqual(context.count_tokens("some text"), 2)

    def test_reference_token_counter_keeps_original_execution(self):
        context = cache_only_context()
        context.execution_policy = "reference"
        context.llm = SimpleNamespace(tokenize=Mock(return_value=[1, 2]))
        self.assertEqual(context.count_tokens("x"), context.count_tokens("x"))
        self.assertEqual(context.llm.tokenize.call_count, 2)

    def test_bm25_cached_counts_invalidate_on_token_mutation_and_reset(self):
        context = retrieval_context("optimized")
        expected = retrieval_context("reference")
        for query in ("lantern", "paint", "lantern lantern"):
            self.assertEqual(context.lexical_candidate_scores(query, 5), expected.lexical_candidate_scores(query, 5))
        context.vector_memory[0]["lexical_tokens"] = ["paint"]
        expected.vector_memory[0]["lexical_tokens"] = ["paint"]
        self.assertEqual(context.bm25_score(["paint"], 0), expected.bm25_score(["paint"], 0))
        self.assertTrue(context.lexical_term_cache)
        context.reset_memory("new")
        self.assertEqual(context.lexical_term_cache, {})

    def test_embedding_cache_returns_identical_normalized_vectors(self):
        context = cache_only_context()

        first = context.encode_normalized_texts(["alpha", "alpha", "beta"])
        second = context.encode_normalized_texts(["beta", "alpha"])

        self.assertEqual(context.embedder.calls, [["alpha", "beta"]])
        np.testing.assert_allclose(first[0], first[1], rtol=0, atol=0)
        np.testing.assert_allclose(first[0], second[1], rtol=0, atol=0)
        np.testing.assert_allclose(first[2], second[0], rtol=0, atol=0)
        np.testing.assert_allclose(
            np.linalg.norm(first, axis=1),
            np.ones(3),
            rtol=1e-6,
        )
        self.assertEqual(context.cache_counters["embedding_misses"], 2)
        self.assertEqual(context.cache_counters["embedding_hits"], 3)

    def test_reranker_cache_preserves_raw_scores_and_skips_duplicate_work(self):
        context = cache_only_context()
        pairs = [("question", "document one"), ("question", "document two")]

        first = context.predict_reranker_pairs(pairs + [pairs[0]])
        second = context.predict_reranker_pairs(list(reversed(pairs)))

        self.assertEqual(len(context.local_reranker.calls), 1)
        self.assertEqual(context.local_reranker.calls[0], pairs)
        self.assertEqual(first[0], first[2])
        self.assertEqual(second, [first[1], first[0]])
        self.assertEqual(context.cache_counters["reranker_misses"], 2)
        self.assertEqual(context.cache_counters["reranker_hits"], 3)

    def test_disabled_reranker_keeps_hybrid_order_without_loading_a_model(self):
        context = cache_only_context()
        context.enable_reranker = False

        ranked = context.rerank_memory_candidates(
            "question",
            [
                {"memory_id": "low", "hybrid_score": 0.2},
                {"memory_id": "high", "hybrid_score": 0.8},
            ],
        )

        self.assertEqual([row["memory_id"] for row in ranked], ["high", "low"])
        self.assertTrue(all(row["reranker"] == "disabled" for row in ranked))


def retrieval_context(policy):
    context = cache_only_context()
    context.execution_policy = policy
    context.enable_dense_retrieval = True
    context.enable_lexical_retrieval = True
    context.enable_graph_retrieval = True
    context.reset_memory("fixture")
    context.index = faiss.IndexFlatL2(3)
    for index, (text, speaker) in enumerate([
        ("I bought a lantern", "Asha"), ("I bought a lantern", "Ben"),
        ("Asha painted a lantern", "Asha"), ("Ben bought paint", "Ben"),
        ("Asha bought paint", "Asha"),
    ]):
        context.embed_and_store(text, label="raw_turn", metadata={
            "source_quote": text, "speaker": speaker, "role": "user",
            "source_session_id": str(index // 2), "source_turn_ids": [index],
            "edge_id": "edge-" + str(index),
            "status": "superseded" if index == 4 else "active",
        })
    context.embedder.calls.clear()
    context.embedding_cache.clear()
    context.reranker_score_cache.clear()
    return context


class BatchedRetrievalTests(unittest.TestCase):
    queries = ["Asha lantern", "Ben paint", "Asha lantern"]

    def check_equivalence(self, **kwargs):
        reference, candidate = retrieval_context("reference"), retrieval_context("optimized")
        options = {"top_k": 3, "min_score": 0.0, "graph_evidence": [], **kwargs}
        expected = [reference.retrieve_relevant_memories(query, **options) for query in self.queries]
        actual = candidate.retrieve_relevant_memories_many(self.queries, **options)
        self.assertEqual(actual, expected)
        return reference, candidate

    def test_identical_scores_order_provenance_and_reduced_encoder_invocations(self):
        reference, candidate = self.check_equivalence()
        self.assertEqual(len(reference.embedder.calls), 2)
        self.assertEqual(len(candidate.embedder.calls), 1)
        self.assertEqual(len(reference.local_reranker.calls), 2)
        self.assertEqual(len(candidate.local_reranker.calls), 1)

    def test_all_filters_and_thresholds_remain_per_query(self):
        for top_k in (0, 1, 3, 50):
            for threshold in (0.0, 0.2, 0.8, 1.1):
                with self.subTest(top_k=top_k, threshold=threshold):
                    self.check_equivalence(top_k=top_k, min_score=threshold,
                        exclude_turn_ids={0}, query_time_scope="current")
        self.check_equivalence(query_time_scope="historical", query_relation_keys={"buy"})

    def test_graph_expansion_does_not_change_pool_or_normalization(self):
        self.check_equivalence(graph_evidence=[{"edge_id": "edge-3", "source_turn_ids": [3]}])

    def test_caches_can_be_disabled_without_losing_batch_results(self):
        reference, candidate = retrieval_context("reference"), retrieval_context("optimized")
        for context in (reference, candidate):
            context.embedding_cache_size = context.reranker_cache_size = 0
        expected = [reference.retrieve_relevant_memories(q, graph_evidence=[], min_score=0.0) for q in self.queries]
        actual = candidate.retrieve_relevant_memories_many(self.queries, graph_evidence=[], min_score=0.0)
        self.assertEqual(actual, expected)
        self.assertEqual(len(candidate.embedder.calls), 1)
        self.assertEqual(len(candidate.local_reranker.calls), 1)

    def test_retrieval_ablations_do_not_reenable_models(self):
        for component in ("enable_dense_retrieval", "enable_lexical_retrieval", "enable_graph_retrieval", "enable_reranker"):
            reference, candidate = retrieval_context("reference"), retrieval_context("optimized")
            setattr(reference, component, False)
            setattr(candidate, component, False)
            expected = [reference.retrieve_relevant_memories(q, graph_evidence=[], min_score=0.0) for q in self.queries]
            self.assertEqual(candidate.retrieve_relevant_memories_many(self.queries, graph_evidence=[], min_score=0.0), expected)
            if component == "enable_dense_retrieval":
                self.assertEqual(candidate.embedder.calls, [])
            if component == "enable_reranker":
                self.assertEqual(candidate.local_reranker.calls, [])

    def test_batch_reranker_failure_retries_original_per_query_path(self):
        reference, candidate = retrieval_context("reference"), retrieval_context("optimized")
        original = candidate.local_reranker.predict
        failed = []

        def predict(pairs):
            if not failed:
                failed.append(True)
                raise RuntimeError("batch rejected")
            return original(pairs)

        candidate.local_reranker.predict = predict
        expected = [reference.retrieve_relevant_memories(q, graph_evidence=[], min_score=0.0) for q in self.queries]
        self.assertEqual(candidate.retrieve_relevant_memories_many(self.queries, graph_evidence=[], min_score=0.0), expected)
        self.assertTrue(failed)

    def test_implicit_graph_retrieval_stays_serial(self):
        context = retrieval_context("optimized")
        context.retrieve_relevant_memories = Mock(side_effect=[[1], [2], [3]])
        self.assertEqual(context.retrieve_relevant_memories_many(self.queries), [[1], [2], [3]])
        self.assertEqual(context.embedder.calls, [])

    def test_query_fork_has_isolated_bm25_cache(self):
        context = retrieval_context("optimized")
        context.bm25_score(["lantern"], 0)
        before = copy.deepcopy(context.lexical_term_cache)
        fork = context.fork_for_query()
        fork.vector_memory[0]["lexical_tokens"] = ["changed"]
        fork.bm25_score(["changed"], 0)
        self.assertEqual(context.lexical_term_cache, before)
        self.assertNotEqual(fork.lexical_term_cache, before)


if __name__ == "__main__":
    unittest.main()
