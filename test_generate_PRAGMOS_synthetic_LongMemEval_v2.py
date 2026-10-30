import random
import unittest
from datetime import datetime, timedelta

from generate_PRAGMOS_synthetic_LongMemEval_v2 import (
    BackgroundProvider,
    DistributionAccumulator,
    RelevantEvidenceTarget,
    StructuralProfile,
    anchor_coherent_distractor,
    build_record,
    compose_calibrated_evidence_turn,
    distribution_comparison,
    fit_content,
    knowledge_update_scenario,
    numeric_mention_count,
    neutralize_exact_official_turn_overlaps,
    normalized_text_hash,
    question_subtype,
    relevant_evidence_comparison,
    sentence_count,
    synthetic_relevant_evidence_metrics,
    validate_evidence_preservation,
    word_count,
)


class SyntheticLongMemEvalV2Tests(unittest.TestCase):
    def test_knowledge_update_retains_independent_background_topic(self):
        profile = StructuralProfile(
            source_index=8,
            question_type="knowledge-update",
            subtype="current-state",
            abstention=False,
            question_words=16,
            answer_words=4,
            question_date=datetime(2025, 2, 20, 12, 0),
            session_dates=[datetime(2025, 2, 19, 12, 0)],
            session_roles=[["user", "assistant"]],
            session_word_counts=[[80, 120]],
            answer_session_indices=[0],
        )
        scenario = knowledge_update_scenario(
            profile,
            random.Random(31),
            "new product managers",
        )
        self.assertEqual(scenario.topic, "new product managers")

    def test_topic_fragments_are_consumed_without_reuse(self):
        provider = BackgroundProvider.__new__(BackgroundProvider)
        provider.handles = {"user": object(), "assistant": object()}
        provider.topic_fragments = {
            ("harbor workshop", "user"): [
                "The harbor workshop compared several restoration methods.",
                "Participants discussed how tides affected the survey schedule.",
            ],
            ("harbor workshop", "assistant"): [
                "The facilitator summarized the practical tradeoffs for the group."
            ],
        }
        first = provider.take_topic_sentence_fragments(
            "harbor workshop", "user", [8, 9]
        )
        second = provider.take_topic_sentence_fragments(
            "harbor workshop", "user", [10]
        )
        self.assertEqual(len(set(first + second)), 3)
        self.assertEqual(
            provider.topic_context_summary(["harbor workshop"])[
                "topics_with_any_context"
            ],
            0,
        )

    def test_subtypes_cover_operations(self):
        cases = [
            ("multi-session", "How much did I save by taking the train instead of a taxi?", "difference"),
            ("multi-session", "How many different doctors did I visit?", "count-distinct"),
            ("temporal-reasoning", "How many days passed between the two events?", "elapsed-duration"),
            ("knowledge-update", "What is my current job title?", "current-state"),
            ("single-session-assistant", "What was the 7th item in your list?", "ordinal-list"),
        ]
        for question_type, question, expected in cases:
            self.assertEqual(
                question_subtype({"question_type": question_type, "question": question}),
                expected,
            )

    def test_required_evidence_survives_word_fitting(self):
        required = (
            "I paid exactly $47 for the completed repair after comparing three providers, "
            "and that was the final charge rather than the preliminary estimate."
        )
        text = fit_content(
            20,
            ["This is ordinary surrounding conversation without the requested fact."],
            random.Random(3),
            required,
        )
        self.assertIn(required, text)
        self.assertGreaterEqual(word_count(text), word_count(required) + 12)

    def test_record_preserves_structural_shape(self):
        question_date = datetime(2025, 2, 20, 12, 0)
        profile = StructuralProfile(
            source_index=11,
            question_type="multi-session",
            subtype="sum",
            abstention=False,
            question_words=18,
            answer_words=4,
            question_date=question_date,
            session_dates=[question_date - timedelta(days=i + 1) for i in range(3)],
            session_roles=[["user", "assistant"]] * 3,
            session_word_counts=[[40, 180], [35, 200], [50, 160]],
            answer_session_indices=[0, 2],
        )
        record, audit = build_record(profile, seed=9)
        self.assertEqual(len(record["haystack_sessions"]), 3)
        self.assertEqual([len(value) for value in record["haystack_sessions"]], [2, 2, 2])
        self.assertEqual(len(record["answer_session_ids"]), 2)
        self.assertEqual(len(audit["evidence"]), 2)
        self.assertTrue(set(record["answer_session_ids"]).issubset(record["haystack_session_ids"]))
        validate_evidence_preservation(record, audit)

    def test_calibrated_evidence_turn_preserves_fact_and_density(self):
        required = "I completed the harbor workshop and paid the final fee of $47."
        target = RelevantEvidenceTarget(
            words=96,
            sentences=6,
            numeric_mentions=4,
        )
        text = compose_calibrated_evidence_turn(
            required_text=required,
            target=target,
            role="user",
            topic="harbor workshop",
            rng=random.Random(17),
        )
        self.assertIn(required, text)
        self.assertEqual(word_count(text), target.words)
        self.assertEqual(sentence_count(text), target.sentences)
        self.assertEqual(
            numeric_mention_count(text),
            target.numeric_mentions,
        )

    def test_natural_background_evidence_preserves_density(self):
        required = "I completed the harbor workshop and paid the final fee of $47."
        target = RelevantEvidenceTarget(96, 6, 4)
        background = [
            "The planning discussion compared practical alternatives and documented "
            "the reasons behind each recommendation before reviewing the schedule."
        ] * 12
        text = compose_calibrated_evidence_turn(
            required_text=required,
            target=target,
            role="user",
            topic="harbor workshop",
            rng=random.Random(19),
            background_sentences=background,
        )
        self.assertIn(required, text)
        self.assertEqual(word_count(text), target.words)
        self.assertEqual(sentence_count(text), target.sentences)
        self.assertEqual(numeric_mention_count(text), target.numeric_mentions)

    def test_numeric_topic_is_not_repeated_in_evidence_filler(self):
        required = "I completed the 360-degree video workshop and paid $47."
        target = RelevantEvidenceTarget(
            words=96,
            sentences=6,
            numeric_mentions=4,
        )
        text = compose_calibrated_evidence_turn(
            required_text=required,
            target=target,
            role="user",
            topic="360-degree video workshop",
            rng=random.Random(23),
        )
        self.assertIn(required, text)
        self.assertEqual(word_count(text), target.words)
        self.assertEqual(sentence_count(text), target.sentences)
        self.assertEqual(numeric_mention_count(text), target.numeric_mentions)

    def test_exact_official_overlap_is_neutralized_only_for_background(self):
        record = {
            "question_id": "synthetic_v2_overlap_test",
            "haystack_session_ids": ["session-1"],
            "haystack_sessions": [[
                {"role": "user", "content": "can you show me how"},
                {"role": "assistant", "content": "Independent evidence stays unchanged."},
            ]],
        }
        audit = {
            "evidence": [{"session_id": "session-1", "turn_index": 1}],
        }
        count = neutralize_exact_official_turn_overlaps(
            record,
            audit,
            {normalized_text_hash("can you show me how")},
        )
        self.assertEqual(count, 1)
        self.assertTrue(
            record["haystack_sessions"][0][0]["content"].startswith(
                "can you show me how"
            )
        )
        self.assertEqual(
            record["haystack_sessions"][0][1]["content"],
            "Independent evidence stays unchanged.",
        )

    def test_record_uses_full_turn_evidence_density_target(self):
        question_date = datetime(2025, 2, 20, 12, 0)
        target = RelevantEvidenceTarget(
            words=210,
            sentences=16,
            numeric_mentions=6,
        )
        profile = StructuralProfile(
            source_index=12,
            question_type="single-session-assistant",
            subtype="assistant-recall",
            abstention=False,
            question_words=16,
            answer_words=3,
            question_date=question_date,
            session_dates=[question_date - timedelta(days=1)],
            session_roles=[["user", "assistant", "user", "assistant"]],
            session_word_counts=[[40, 180, 35, 200]],
            answer_session_indices=[0],
            relevant_evidence_target=target,
        )
        record, audit = build_record(profile, seed=11)
        metrics, evidence_turns = synthetic_relevant_evidence_metrics(record, audit)
        self.assertEqual(evidence_turns, 1)
        self.assertEqual(metrics, target)
        validate_evidence_preservation(record, audit)

    def test_zero_evidence_profile_keeps_answer_sessions_without_audit_evidence(self):
        question_date = datetime(2025, 2, 20, 12, 0)
        profile = StructuralProfile(
            source_index=13,
            question_type="single-session-user",
            subtype="attribute",
            abstention=True,
            question_words=12,
            answer_words=4,
            question_date=question_date,
            session_dates=[question_date - timedelta(days=1)],
            session_roles=[["user", "assistant"]],
            session_word_counts=[[60, 80]],
            answer_session_indices=[0],
            relevant_evidence_target=RelevantEvidenceTarget(0, 0, 0),
            suppress_relevant_evidence=True,
        )
        record, audit = build_record(profile, seed=13)
        self.assertEqual(len(record["answer_session_ids"]), 1)
        self.assertEqual(audit["evidence"], [])

    def test_hard_distractor_adds_question_anchors_without_answer(self):
        text = anchor_coherent_distractor(
            question="Which current workshop title replaced the earlier draft?",
            answer="Northwind Final",
            topic="workshop record",
            distractor="The old working title was Cedar Draft.",
            rng=random.Random(29),
        )
        self.assertIn("workshop", text.lower())
        self.assertNotIn("northwind final", text.lower())
        self.assertIn("cedar draft", text.lower())

    def test_distribution_gate_rejects_large_word_gap(self):
        section = {"mean": 100.0, "median": 100.0, "p90": 100.0, "min": 1, "max": 200}
        official = {
            "records": 500,
            "question_type_counts": {"x": 500},
            "subtype_counts": {"x::y": 500},
            "abstention_count": 30,
            "session_count": section,
            "turn_count": section,
            "source_words": section,
            "gold_session_words": section,
            "question_words": section,
            "answer_words": section,
            "answer_session_count": section,
            "record_ttr": section,
            "unique_turn_rate": section,
            "trigram_diversity": section,
            "question_lexical_diversity": {
                "token_count": 1000,
                "vocabulary_size": 300,
                "ttr": 0.3,
                "trigram_diversity": 0.8,
            },
        }
        synthetic = dict(official)
        synthetic["source_words"] = dict(section, mean=50.0)
        result = distribution_comparison(official, synthetic)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["source_words_mean_within_8pct"])

    def test_question_lexical_diversity_is_computed(self):
        accumulator = DistributionAccumulator()
        record = {
            "question_id": "synthetic_v2_test",
            "question_type": "single-session-user",
            "question": "Which coastal observatory hosted the astronomy workshop?",
            "answer": "Aster Hall",
            "answer_session_ids": ["session-1"],
            "haystack_session_ids": ["session-1"],
            "haystack_sessions": [[
                {"role": "user", "content": "I attended the astronomy workshop at Aster Hall."},
                {"role": "assistant", "content": "That sounds like a useful workshop."},
            ]],
        }
        accumulator.add(record, "place")
        lexical = accumulator.summary()["question_lexical_diversity"]
        self.assertEqual(lexical["token_count"], 7)
        self.assertGreater(lexical["ttr"], 0.0)
        self.assertGreater(lexical["trigram_diversity"], 0.0)

    def test_relevant_evidence_gate_rejects_sparse_assistant_numbers(self):
        def section(words, sentences, numbers):
            return {
                "records": 10,
                "words": {"mean": words},
                "sentences": {"mean": sentences},
                "numeric_mentions": {"mean": numbers},
            }

        official = {
            "overall": section(116.0, 7.0, 2.1),
            "by_question_type": {
                "single-session-assistant": section(216.0, 16.7, 6.3),
            },
        }
        synthetic = {
            "overall": section(116.0, 7.0, 2.1),
            "by_question_type": {
                "single-session-assistant": section(216.0, 16.7, 3.0),
            },
        }
        result = relevant_evidence_comparison(official, synthetic)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["assistant_numeric_mentions_mean_within_8pct"]
        )


if __name__ == "__main__":
    unittest.main()
