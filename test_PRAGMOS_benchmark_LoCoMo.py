import argparse
import unittest
from collections import OrderedDict

import faiss
import networkx as nx

import PRAGMOS_benchmark_LoCoMo as adapter
from PRAGMOS_benchmark_LongMemEval import (
    actor_bound_evidence_coverage,
    build_query_profile,
    enforce_actor_grounding_abstention,
    evidence_provenance_identity,
    filter_actor_grounded_evidence,
    filter_candidates_by_query_anchors_in_evidence,
)
from PRAGMOS_context_layer_org import ContextLayer


class IdentityStemmer:
    def stem(self, word):
        return word


class LoCoMoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "sample_id": "sample-a",
            "conversation": {
                "speaker_a": "Asha",
                "speaker_b": "Ben",
                "session_1": [
                    {
                        "speaker": "Asha",
                        "dia_id": "D1:1",
                        "text": "I adopted a dog named Comet.",
                    },
                    {
                        "speaker": "Ben",
                        "dia_id": "D1:2",
                        "text": "That is wonderful.",
                        "blip_caption": "A black dog beside a red ball.",
                    },
                ],
                "session_1_date_time": "11:15 am on 5 January, 2027",
                "session_2": [
                    {
                        "speaker": "Asha",
                        "dia_id": "D2:1",
                        "text": "Comet likes the park.",
                    }
                ],
                "session_2_date_time": "2:30 pm on 8 January, 2027",
            },
            "qa": [
                {
                    "question_id": "q-1",
                    "question": "What is Asha's dog's name?",
                    "answer": "Comet",
                    "category": 4,
                    "evidence": ["D1:1"],
                },
                {
                    "question_id": "q-2",
                    "question": "What color was the ball?",
                    "answer": "red",
                    "category": 4,
                    "evidence": ["D1:2"],
                },
            ],
        }

    def bare_context_layer(self):
        layer = ContextLayer.__new__(ContextLayer)
        layer.session_id = "test"
        layer.turn_counter = 0
        return layer

    def test_timestamp_normalization(self):
        self.assertEqual(
            adapter.normalize_session_timestamp(
                "11:15 am on 5 January, 2027"
            ),
            "2027-01-05T11:15:00",
        )
        self.assertEqual(
            adapter.normalize_session_timestamp("2027-01-05T11:15:00"),
            "2027-01-05T11:15:00",
        )

    def test_peer_turns_and_caption_provenance_are_preserved(self):
        prepared = adapter.build_locomo_turns(
            self.bare_context_layer(),
            self.sample,
        )
        turns = prepared["turns"]
        self.assertEqual(len(turns), 4)
        self.assertTrue(all(turn.role == "user" for turn in turns))
        self.assertEqual([turn.speaker for turn in turns[:3]], ["Asha", "Ben", "Ben"])
        self.assertEqual(turns[0].external_turn_id, "D1:1")
        self.assertEqual(turns[0].evidence_type, "dialogue")
        self.assertEqual(turns[0].text, "I adopted a dog named Comet.")
        self.assertEqual(turns[2].evidence_type, "image_caption")
        self.assertEqual(turns[2].parent_turn_id, "D1:2")
        self.assertEqual(prepared["internal_to_dialog"][turns[2].turn_id], "D1:2")

    def test_source_quote_stays_raw_while_speaker_is_separate_metadata(self):
        prepared = adapter.build_locomo_turns(
            self.bare_context_layer(),
            self.sample,
        )
        turn = prepared["turns"][0]

        self.assertEqual(turn.text, "I adopted a dog named Comet.")
        self.assertEqual(turn.speaker, "Asha")
        self.assertNotIn("Asha:", turn.text)

    def test_raw_haystack_includes_caption_and_named_speaker(self):
        record = adapter.locomo_as_haystack_record(self.sample)
        message = record["haystack_sessions"][0][1]
        self.assertEqual(message["role"], "Ben")
        self.assertIn("[Image caption:", message["content"])
        self.assertEqual(record["haystack_dates"][0], "2027-01-05T11:15:00")

    def test_inference_record_contains_no_gold_fields(self):
        for category in adapter.CATEGORY_NAMES:
            record = adapter.inference_record_for_category(category)
            self.assertNotIn("answer", record)
            self.assertNotIn("evidence", record)
            self.assertNotIn("answer_session_ids", record)
        self.assertEqual(
            adapter.inference_record_for_category(1)["question_type"],
            "multi-session",
        )
        self.assertEqual(adapter.inference_record_for_category(5), {})

    def test_dataset_validation_and_workload_selection(self):
        validation = adapter.validate_locomo_dataset([self.sample])
        self.assertEqual(validation["question_count"], 2)
        self.assertEqual(validation["caption_turn_count"], 1)
        args = argparse.Namespace(
            start_index=0,
            limit=1,
            category=[4],
            sample_id=None,
        )
        workload = adapter.flatten_workload([self.sample], args)
        self.assertEqual(len(workload), 1)
        self.assertEqual(workload[0]["question_id"], "q-1")

    def test_official_category_scoring_rules(self):
        stemmer = IdentityStemmer()
        self.assertEqual(
            adapter.official_locomo_score(
                "dog, park",
                "dog, park",
                1,
                stemmer=stemmer,
            ),
            1.0,
        )
        self.assertEqual(
            adapter.official_locomo_score(
                "Paris",
                "Paris; France",
                3,
                stemmer=stemmer,
            ),
            1.0,
        )
        self.assertEqual(
            adapter.official_locomo_score(
                "No information available",
                None,
                5,
                stemmer=stemmer,
            ),
            1.0,
        )
        self.assertEqual(
            adapter.official_locomo_score(
                "I do not know",
                None,
                5,
                stemmer=stemmer,
            ),
            0.0,
        )

    def test_canonical_abstention_is_narrow(self):
        self.assertEqual(
            adapter.canonicalize_locomo_abstention("I do not know."),
            "No information available",
        )
        self.assertEqual(
            adapter.canonicalize_locomo_abstention("There is no red car."),
            "There is no red car.",
        )

    def test_dialog_and_session_retrieval_metrics(self):
        metrics = adapter.retrieval_metrics(
            ["D1:1", "D2:1"],
            ["D1:1", "D9:1", "D2:1"],
        )
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertTrue(metrics["complete_evidence_coverage"])
        sessions = adapter.session_ids_for_dialogs(
            ["D1:1", "D2:1", "D1:2"],
            {"D1:1": "session_1", "D1:2": "session_1", "D2:1": "session_2"},
        )
        self.assertEqual(sessions, ["session_1", "session_2"])

    def test_wrong_speaker_with_identical_fact_does_not_ground_answer(self):
        profile = build_query_profile(
            "When did Asha start painting?",
            known_speakers=["Asha", "Ben"],
        )
        wrong = {
            "source_turn_id": "D1:1",
            "source_session_id": "session_1",
            "source_speaker": "Ben",
            "source_role": "user",
            "source_quote": "I started painting last Tuesday.",
            "triples": [["speaker", "started painting", "last Tuesday"]],
        }
        correct = {
            **wrong,
            "source_turn_id": "D2:1",
            "source_session_id": "session_2",
            "source_speaker": "Asha",
        }

        wrong_coverage = actor_bound_evidence_coverage(profile, [wrong])
        correct_coverage = actor_bound_evidence_coverage(profile, [correct])

        self.assertFalse(wrong_coverage["complete"])
        self.assertTrue(correct_coverage["complete"])
        self.assertFalse(wrong_coverage["per_evidence"][0]["actor_bound"])
        abstain, reason = enforce_actor_grounding_abstention(
            safe_abstention=False,
            abstention_reason=None,
            query_profile=profile,
            actor_grounding=wrong_coverage,
        )
        self.assertTrue(abstain)
        self.assertEqual(
            reason,
            "actor_bound_evidence_missing_or_incomplete",
        )

    def test_reason_question_does_not_treat_why_as_an_entity_anchor(self):
        profile = build_query_profile(
            "Why did Rohan postpone the Emerald Atlas autumn program?",
            known_speakers=["Rohan", "Mira"],
        )
        evidence = {
            "source_turn_id": "D1:1",
            "source_session_id": "session_1",
            "source_speaker": "Rohan",
            "source_role": "user",
            "source_quote": (
                "I postponed the Emerald Atlas autumn program because the "
                "delivery arrived late."
            ),
            "triples": [
                [
                    "speaker",
                    "postponed Emerald Atlas autumn program",
                    "delivery arrived late",
                ]
            ],
        }

        self.assertNotIn(
            "why",
            {
                token
                for group in profile["required_anchor_groups"]
                for token in group["tokens"]
            },
        )
        self.assertTrue(
            actor_bound_evidence_coverage(profile, [evidence])["complete"]
        )

    def test_actor_grounding_handles_derived_and_operation_qualifiers(self):
        speakers = ["Asha", "Rohan"]
        preference_profile = build_query_profile(
            "Would Asha prefer a national park or an indoor arcade after "
            "the Copper Atlas winter project?",
            known_speakers=speakers,
        )
        preference_evidence = {
            "source_turn_id": "D1:1",
            "source_session_id": "session_1",
            "source_speaker": "Asha",
            "source_quote": (
                "I loved the forest hikes during the Copper Atlas winter "
                "project."
            ),
        }
        self.assertNotIn(
            "would",
            {
                token
                for group in preference_profile["required_anchor_groups"]
                for token in group["tokens"]
            },
        )
        self.assertTrue(
            actor_bound_evidence_coverage(
                preference_profile,
                [preference_evidence],
            )["complete"]
        )

        country_profile = build_query_profile(
            "Which country did Asha visit for the Amber Atlas project?",
            known_speakers=speakers,
        )
        country_evidence = {
            "source_turn_id": "D2:1",
            "source_session_id": "session_2",
            "source_speaker": "Asha",
            "source_quote": (
                "For the Amber Atlas project, I spent a week in Kyoto."
            ),
        }
        self.assertTrue(
            actor_bound_evidence_coverage(
                country_profile,
                [country_evidence],
            )["complete"]
        )

        count_profile = build_query_profile(
            "How many distinct workshops did Asha attend in the Copper "
            "Atlas series?",
            known_speakers=speakers,
            allow_distributed_actor_evidence=True,
        )
        self.assertIn(
            "workshops",
            [
                group["text"]
                for group in count_profile["required_anchor_groups"]
            ],
        )
        self.assertNotIn(
            "distinct workshops",
            [
                group["text"]
                for group in count_profile["required_anchor_groups"]
            ],
        )

    def test_primary_actor_is_bound_when_question_mentions_both_peers(self):
        profile = build_query_profile(
            "What gift did Asha buy for Ben?",
            known_speakers=["Asha", "Ben"],
        )

        self.assertTrue(profile["actor_binding_enabled"])
        self.assertEqual(profile["requested_speakers"], ["Asha"])
        self.assertEqual(profile["mentioned_speakers"], ["Asha", "Ben"])

        joint_profile = build_query_profile(
            "Which activities did Asha and Ben both start?",
            known_speakers=["Asha", "Ben"],
        )
        self.assertFalse(joint_profile["actor_binding_enabled"])

    def test_same_speaker_wrong_attribute_is_rejected(self):
        profile = build_query_profile(
            "What is the name of Asha's dog?",
            known_speakers=["Asha", "Ben"],
        )
        wrong_attribute = {
            "source_turn_id": "D1:1",
            "source_session_id": "session_1",
            "source_speaker": "Asha",
            "source_role": "user",
            "source_quote": "My dog's age is four years.",
            "triples": [["my dog", "age", "four years"]],
        }
        correct_attribute = {
            **wrong_attribute,
            "source_turn_id": "D2:1",
            "source_quote": "My dog is named Comet.",
            "triples": [["my dog", "name", "Comet"]],
        }

        self.assertFalse(
            actor_bound_evidence_coverage(profile, [wrong_attribute])["complete"]
        )
        self.assertTrue(
            actor_bound_evidence_coverage(profile, [correct_attribute])["complete"]
        )
        combined_coverage = actor_bound_evidence_coverage(
            profile,
            [wrong_attribute, correct_attribute],
        )
        self.assertTrue(combined_coverage["complete"])
        self.assertEqual(
            filter_actor_grounded_evidence(
                [wrong_attribute, correct_attribute],
                profile,
                combined_coverage,
            ),
            [correct_attribute],
        )

        budget_profile = build_query_profile(
            "What budget did Asha approve for the Copper Atlas program?",
            known_speakers=["Asha", "Ben"],
        )
        venue_only = {
            "source_turn_id": "D3:1",
            "source_session_id": "session_3",
            "source_speaker": "Asha",
            "source_quote": (
                "I held the Copper Atlas program at Juniper Community Hall."
            ),
            "triples": [
                ["Copper Atlas program", "held at", "Juniper Community Hall"]
            ],
        }
        self.assertFalse(
            actor_bound_evidence_coverage(budget_profile, [venue_only])["complete"]
        )
        budget_fact = {
            **venue_only,
            "source_turn_id": "D4:1",
            "source_quote": "I approved a budget of $4,500 for Copper Atlas.",
            "triples": [["Copper Atlas", "approved budget", "$4,500"]],
        }
        self.assertTrue(
            actor_bound_evidence_coverage(budget_profile, [budget_fact])["complete"]
        )

    def test_candidate_filter_keeps_only_actor_bound_provenance(self):
        profile = build_query_profile(
            "When did Asha start painting?",
            known_speakers=["Asha", "Ben"],
        )
        candidates = [
            {
                "value": "Monday",
                "source_turn_id": "D1:1",
                "source_session_id": "session_1",
                "source_speaker": "Ben",
                "source_quote": "I started painting Monday.",
                "head": "speaker",
                "relation": "started painting",
                "tail": "Monday",
            },
            {
                "value": "Tuesday",
                "source_turn_id": "D2:1",
                "source_session_id": "session_2",
                "source_speaker": "Asha",
                "source_quote": "I started painting Tuesday.",
                "head": "speaker",
                "relation": "started painting",
                "tail": "Tuesday",
            },
        ]

        filtered = filter_candidates_by_query_anchors_in_evidence(
            candidates,
            profile,
        )

        self.assertEqual([candidate["value"] for candidate in filtered], ["Tuesday"])

    def test_multi_hop_allows_distributed_facts_but_rejects_other_actor(self):
        profile = build_query_profile(
            "Which activities did Asha start across the conversations?",
            known_speakers=["Asha", "Ben"],
            allow_distributed_actor_evidence=True,
        )
        rows = [
            {
                "source_turn_id": "D1:1",
                "source_session_id": "session_1",
                "source_speaker": "Asha",
                "source_quote": "I started painting.",
                "triples": [["speaker", "started", "painting"]],
            },
            {
                "source_turn_id": "D2:1",
                "source_session_id": "session_2",
                "source_speaker": "Asha",
                "source_quote": "I started yoga too.",
                "triples": [["speaker", "started", "yoga"]],
            },
            {
                "source_turn_id": "D3:1",
                "source_session_id": "session_3",
                "source_speaker": "Ben",
                "source_quote": "I started running.",
                "triples": [["speaker", "started", "running"]],
            },
        ]

        coverage = actor_bound_evidence_coverage(profile, rows)

        self.assertTrue(coverage["complete"])
        self.assertEqual(len(coverage["eligible_provenance_identities"]), 2)
        self.assertEqual(len(coverage["rejected_provenance_identities"]), 1)

    def test_provenance_identity_does_not_merge_identical_peer_quotes(self):
        base = {
            "source_turn_id": "D1:1",
            "source_session_id": "session_1",
            "source_quote": "I started painting.",
            "evidence_type": "dialogue",
        }
        asha = {**base, "source_speaker": "Asha"}
        ben = {**base, "source_turn_id": "D1:2", "source_speaker": "Ben"}

        self.assertNotEqual(
            evidence_provenance_identity(asha),
            evidence_provenance_identity(ben),
        )
        layer = self.bare_context_layer()
        core_asha = {
            **asha,
            "source_turn_ids": [asha["source_turn_id"]],
            "speaker": "Asha",
        }
        core_ben = {
            **ben,
            "source_turn_ids": [ben["source_turn_id"]],
            "speaker": "Ben",
        }
        self.assertNotEqual(
            layer.memory_provenance_identity(core_asha),
            layer.memory_provenance_identity(core_ben),
        )

    def test_compact_context_evidence_displays_actor_envelope(self):
        layer = self.bare_context_layer()
        formatted = layer.format_vector_evidence(
            [
                {
                    "source_turn_ids": [1],
                    "source_session_id": "session_1",
                    "source_quote": "I started painting.",
                    "speaker": "Asha",
                    "role": "user",
                    "evidence_type": "dialogue",
                    "timestamp": "2027-01-05T11:15:00",
                    "score": 0.9,
                    "retrieval_sources": ["dense"],
                }
            ],
            compact=True,
        )

        self.assertIn("speaker=Asha", formatted)
        self.assertIn("evidence_type=dialogue", formatted)
        self.assertIn('evidence: "I started painting."', formatted)

    def test_generic_user_assistant_evidence_keeps_legacy_compact_format(self):
        layer = self.bare_context_layer()
        formatted = layer.format_vector_evidence(
            [
                {
                    "source_turn_ids": [1],
                    "source_session_id": "session-1",
                    "source_quote": "I started painting.",
                    "speaker": "user",
                    "role": "user",
                    "timestamp": "2027-01-05T11:15:00",
                    "score": 0.9,
                    "retrieval_sources": ["dense"],
                }
            ],
            compact=True,
        )

        self.assertIn("role=user; sources=dense", formatted)
        self.assertNotIn("speaker=", formatted)
        self.assertNotIn("evidence_type=", formatted)

    def test_actor_instruction_is_gated_to_named_or_typed_evidence(self):
        layer = self.bare_context_layer()
        layer.count_tokens = lambda text: len(str(text).split())
        layer.trim_text_to_token_budget = lambda text, _budget: text
        generic_memory = {
            "source_turn_ids": [1],
            "source_session_id": "session-1",
            "source_quote": "I started painting.",
            "speaker": "user",
            "role": "user",
            "score": 0.9,
            "retrieval_sources": ["dense"],
        }
        named_memory = {
            **generic_memory,
            "source_turn_ids": [2],
            "speaker": "Asha",
            "evidence_type": "dialogue",
        }

        generic_context = layer.build_context_with_budget(
            "Question",
            [generic_memory],
            [],
            [],
            token_budget=1000,
        )
        named_context = layer.build_context_with_budget(
            "Question",
            [named_memory],
            [],
            [],
            token_budget=1000,
        )

        self.assertNotIn("Bind first-person statements", generic_context)
        self.assertIn("Bind first-person statements", named_context)


class ContextLayerQueryForkTests(unittest.TestCase):
    def test_query_fork_isolates_mutable_memory_and_shares_models(self):
        layer = ContextLayer.__new__(ContextLayer)
        layer.llm = object()
        layer.embedder = object()
        layer.local_reranker = object()
        layer.local_reranker_load_attempted = True
        layer.index = faiss.IndexFlatL2(384)
        layer.vector_memory = [{"metadata": {"value": [1]}}]
        layer.lexical_doc_freqs = {"term": 1}
        layer.lexical_total_doc_len = 1
        layer.G = nx.MultiDiGraph()
        layer.G.add_node("person", values=["base"])
        layer.conversation_history = []
        layer.entity_registry = {"person": {"aliases": {"P"}}}
        layer.alias_to_entity_id = {"p": "person"}
        layer.speaker_entity_ids = {"P": "person"}
        layer.last_entity_by_speaker = {"P": "person"}
        layer.last_person_entity_by_speaker = {"P": "person"}
        layer.last_possessive_owner_by_speaker = {}
        layer.related_entity_by_owner = {"person": {"friend": "other"}}
        layer.embedding_cache = OrderedDict()
        layer.reranker_score_cache = OrderedDict()
        layer.cache_counters = {"embedding_hits": 0}
        layer.session_id = "base"
        layer.turn_counter = 3
        layer.edge_counter = 0
        layer.memory_counter = 1
        layer.entity_counter = 1

        fork = layer.fork_for_query()
        fork.vector_memory[0]["metadata"]["value"].append(2)
        fork.G.nodes["person"]["values"].append("query")
        fork.entity_registry["person"]["aliases"].add("Query")

        self.assertEqual(layer.vector_memory[0]["metadata"]["value"], [1])
        self.assertEqual(layer.G.nodes["person"]["values"], ["base"])
        self.assertEqual(layer.entity_registry["person"]["aliases"], {"P"})
        self.assertIs(fork.llm, layer.llm)
        self.assertIs(fork.embedder, layer.embedder)
        self.assertIs(fork.local_reranker, layer.local_reranker)


if __name__ == "__main__":
    unittest.main()
