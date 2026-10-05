import unittest
from collections import OrderedDict

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


if __name__ == "__main__":
    unittest.main()
