import unittest
from collections import OrderedDict

import networkx as nx
import numpy as np

from PRAGMOS_context_layer_org import ContextLayer, ConversationTurn


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
    def test_structural_turn_graph_is_complete_and_semantic_graph_is_untouched(self):
        context = ContextLayer.__new__(ContextLayer)
        context.reset_memory(session_id="structural-test")
        turns = [
            ConversationTurn("s1", 1, "user", "Viky", "t1", "I like cats."),
            ConversationTurn("s1", 2, "assistant", "assistant", "t1", "Noted."),
            ConversationTurn("s2", 3, "user", "Viky", "t2", "I like dogs."),
        ]

        context.register_structural_turns(turns)

        self.assertEqual(context.G.number_of_nodes(), 0)
        self.assertEqual(len(context.turn_records), 3)
        self.assertEqual(context.structural_neighbor_turn_ids([1], radius=1), [2])
        self.assertEqual(context.structural_neighbor_turn_ids([2], radius=1), [1])
        relations = {
            edge[3]["relation"]
            for edge in context.turn_graph.edges(keys=True, data=True)
        }
        self.assertEqual(
            relations,
            {"contains_turn", "authored_turn", "next_turn", "previous_turn"},
        )

    def test_semantic_fact_cache_is_provenance_scoped_and_reusable(self):
        context = ContextLayer.__new__(ContextLayer)
        context.semantic_fact_cache_size = 8
        context.semantic_fact_cache = OrderedDict()
        context.cache_counters = {
            "semantic_fact_hits": 0,
            "semantic_fact_misses": 0,
        }
        turn = ConversationTurn("s1", 1, "user", "Viky", "t1", "I like cats.")

        self.assertIsNone(context.get_cached_semantic_facts(turn, turn.text))
        context.cache_semantic_facts(turn, turn.text, [["Viky", "likes", "cats"]])
        cached = context.get_cached_semantic_facts(turn, turn.text)

        self.assertEqual(cached, (("Viky", "likes", "cats"),))
        self.assertEqual(
            context.get_cached_semantic_facts(turn, turn.text),
            (("Viky", "likes", "cats"),),
        )
        self.assertEqual(context.cache_counters["semantic_fact_hits"], 2)

    def test_graph_edges_and_vector_mirror_store_additive_fact_types(self):
        context = ContextLayer.__new__(ContextLayer)
        context.session_id = "typed-test"
        context.edge_counter = 0
        context.G = nx.MultiDiGraph()
        context.entity_registry = {}
        context.resolve_entity_reference = lambda value, source_turn=None: value
        context.should_merge_identity_triple = lambda *_args, **_kwargs: False
        context.add_node_with_provenance = (
            lambda node, _provenance: context.G.add_node(node)
        )
        context.supersede_conflicting_edges = lambda **_kwargs: []
        context.has_named_or_typed_evidence = lambda _provenance: False
        context.remember_entity_mention = lambda *_args, **_kwargs: None
        stored = []
        context.embed_and_store = lambda text, label, metadata: stored.append(
            {"text": text, "label": label, "metadata": metadata}
        )

        context.update_knowledge_graph_from_triples(
            [["speaker user", "completed camping trip", "3 days"]],
            source_quote="I completed a 3-day camping trip.",
        )

        edge = next(iter(context.G.edges(keys=True, data=True)))[3]
        self.assertEqual(edge["fact_schema_version"], 1)
        self.assertEqual(edge["fact_type"], "measurement")
        self.assertEqual(edge["head_type"], "speaker")
        self.assertEqual(edge["tail_type"], "measurement")
        self.assertEqual(edge["predicate_family"], "event")
        self.assertIn("camp", edge["predicate_actions"])
        self.assertEqual(edge["event_status"], "completed")
        self.assertEqual(
            stored[0]["metadata"]["fact_type"],
            edge["fact_type"],
        )
        money_fact = context.classify_graph_fact(
            "speaker user",
            "raised",
            "$5,000",
            source_quote="I raised $5,000 for charity.",
        )
        negated_plan = context.classify_graph_fact(
            "speaker user",
            "planning",
            "a trip",
            source_quote="I do not plan a trip.",
        )
        self.assertEqual(money_fact["fact_type"], "measurement")
        self.assertEqual(negated_plan["event_status"], "negated")

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
