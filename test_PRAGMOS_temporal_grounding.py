import dataclasses
import datetime
import json
import unittest

from PRAGMOS_temporal_grounding import describe_time, number_at, words
from PRAGMOS_evidence_compiler import (
    build_evidence_ledger, execute_frame_operation_proof, execute_temporal_proof,
    ground_temporal_ledger, parse_semantic_plan,
)
from PRAGMOS_benchmark_LongMemEval import (
    build_query_profile, execute_operation_plan, grounded_temporal_evidence_rows,
    infer_multi_session_operation, resolve_temporal_expression,
    semantic_plan_to_operation_plan, temporal_event_status,
    state_history_overrides_compiler, STATE_SELECTOR_LATEST,
)
from test_PRAGMOS_evidence_compiler import FakeFrameByEvidenceContext, frame, row
from test_PRAGMOS_benchmark_LongMemEval import SelectAllOperationContext


def temporal_plan(*operands, operator="ELAPSED_TIME", unit="days"):
    question = "How long between " + " and ".join(operands) + "?"
    return parse_semantic_plan({
        "operator": operator, "answer_type": "scalar", "target_unit": unit,
        "answer_role": "duration" if operator == "ELAPSED_TIME" else "event",
        "group_role": "event", "requested_status": "completed",
        "actor": "user", "confidence": "high",
        "operands": [{"id": str(i), "text": text, "kind": "event",
                      "actor": "user", "hard_entities": []}
                     for i, text in enumerate(operands)],
    }, question)


def temporal_ledger(*rows):
    return build_evidence_ledger(rows, temporal_resolver=resolve_temporal_expression,
                                 event_status_resolver=temporal_event_status)


class TemporalGroundingTests(unittest.TestCase):
    def test_scalar_duration_contract_executes_from_two_events(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit"), temporal_ledger(
                row("I attended the harbor workshop on 2027-04-02.", "t1", "s1"),
                row("I completed the gallery visit on 2027-04-07.", "t2", "s2"),
            ),
        )
        self.assertEqual(proof.answer, "5 days")
        self.assertTrue(proof.executable)
        json.dumps(proof.as_dict())

    def test_unary_duration_uses_question_clock_not_report_clock(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop"), temporal_ledger(
                row("I finished the harbor workshop on 2027-04-02.", "t1", "s1", timestamp="2027-04-10"),
            ), reference_timestamp="2027/04/15 (Thu) 18:00",
        )
        self.assertEqual(proof.answer, "13 days")
        self.assertEqual(len(proof.bindings), 1)
        self.assertEqual(proof.calculation["reference_role"], "question_time")

    def test_elapsed_operator_itself_establishes_a_duration_contract(self):
        plan = dataclasses.replace(temporal_plan("harbor workshop"), answer_role="unknown")
        proof, _ = execute_temporal_proof(plan, temporal_ledger(
            row("I completed the harbor workshop on 2027-04-02.", "t1", "s1"),
        ), reference_timestamp="2027-04-15")
        self.assertEqual(proof.answer, "13 days")

    def test_unary_duration_never_invents_a_missing_question_clock(self):
        proof, _ = execute_temporal_proof(temporal_plan("harbor workshop"), temporal_ledger(
            row("I completed the harbor workshop on 2027-04-02.", "t1", "s1"),
        ))
        self.assertFalse(proof.executable)
        self.assertEqual(proof.reason, "question_reference_time_missing")

    def test_explicit_unresolved_time_cannot_become_report_time(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit"), temporal_ledger(
                row("I attended the harbor workshop on 2027-04-02.", "t1", "s1"),
                row("I completed the gallery visit some time earlier.", "t2", "s2", timestamp="2027-04-20"),
            ),
        )
        self.assertFalse(proof.executable)
        self.assertIsNone(proof.answer)

    def test_existing_undated_completed_event_contract_is_preserved(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit"), temporal_ledger(
                row("I completed the harbor workshop.", "t1", "s1", timestamp="2027-04-02"),
                row("I completed the gallery visit.", "t2", "s2", timestamp="2027-04-07"),
            ),
        )
        self.assertEqual(proof.answer, "5 days")

    def test_duration_attribute_is_not_an_unresolved_event_date(self):
        atoms, _ = ground_temporal_ledger(temporal_ledger(
            row("I completed the three-day harbor workshop.", "t1", "s1", timestamp="2027-04-02"),
        ))
        self.assertEqual(atoms[0].temporal_expression["value"], "2027-04-02")

    def test_planning_mention_does_not_conflict_with_completion(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit"), temporal_ledger(
                row("I had considered the harbor workshop a week earlier, but that was only a tentative plan.", "p", "p1", timestamp="2027-04-01"),
                row("I completed the harbor workshop on 2027-04-02.", "t1", "s1"),
                row("I completed the gallery visit on 2027-04-07.", "t2", "s2"),
            ),
        )
        self.assertEqual(proof.answer, "5 days")
        self.assertEqual({b.source_turn_id for b in proof.bindings}, {"t1", "t2"})

    def test_relative_event_chain_uses_event_dates_and_keeps_provenance(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit"), temporal_ledger(
                row("I attended the harbor workshop twelve days ago.", "t1", "s1", timestamp="2027-04-14"),
                row("I completed the gallery visit three days after that workshop.", "t2", "s2", timestamp="2027-04-20"),
            ),
        )
        self.assertEqual(proof.answer, "3 days")
        self.assertEqual(proof.calculation["right"], "2027-04-05")
        self.assertEqual(len(proof.bindings[1].temporal_expression["provenance_chain"]), 2)

    def test_reference_mention_is_not_a_second_assertion_of_that_event(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit"), temporal_ledger(
                row("I attended the harbor workshop twelve days ago.", "t1", "s1", timestamp="2027-04-14"),
                row("I completed the gallery visit three days after attending the harbor workshop.", "t2", "s2", timestamp="2027-04-20"),
            ),
        )
        self.assertEqual(proof.answer, "3 days")
        self.assertEqual(proof.bindings[0].source_turn_id, "t1")

    def test_multi_hop_offsets_are_composed_without_model_arithmetic(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "studio tour"), temporal_ledger(
                row("I attended the harbor workshop on 2027-04-02.", "t1", "s1"),
                row("I completed the gallery visit three days after that workshop.", "t2", "s2"),
                row("I finished the studio tour five days after the gallery visit.", "t3", "s3"),
            ),
        )
        self.assertEqual(proof.answer, "8 days")
        self.assertEqual(len(proof.bindings[1].temporal_expression["provenance_chain"]), 3)

    def test_competing_reference_occurrences_remain_unresolved(self):
        proof, _ = execute_temporal_proof(temporal_plan("gallery visit"), temporal_ledger(
            row("I attended the harbor workshop on 2027-04-02.", "t1", "s1"),
            row("I attended the harbor workshop on 2027-04-04.", "t2", "s2"),
            row("I completed the gallery visit three days after that workshop.", "t3", "s3"),
        ), reference_timestamp="2027-04-20")
        self.assertFalse(proof.executable)

    def test_cycle_without_an_absolute_anchor_is_not_executable(self):
        atoms, _ = ground_temporal_ledger(temporal_ledger(
            row("I completed the gallery visit three days after the studio tour.", "t1", "s1"),
            row("I finished the studio tour two days after the gallery visit.", "t2", "s2"),
        ))
        self.assertTrue(all(a.temporal_expression is None for a in atoms))

    def test_other_speaker_cannot_supply_the_reference_event(self):
        atoms, _ = ground_temporal_ledger(temporal_ledger(
            row("I attended the harbor workshop on 2027-04-02.", "t1", "s1", speaker="Mira"),
            row("I completed the gallery visit three days after that workshop.", "t2", "s2", speaker="Omar"),
        ))
        self.assertIsNone(atoms[1].temporal_expression)

    def test_complete_cardinal_quantities_are_not_truncated(self):
        for text, value in (("one hundred and ten", 110), ("one hundred and seventy-seven", 177), ("one hundred and six", 106)):
            with self.subTest(text=text):
                self.assertEqual(number_at(words(text), 0)[0], value)
                result = describe_time("I completed the workshop " + text + " days ago.", "2027-12-31")
                expected = datetime.date(2027, 12, 31) - datetime.timedelta(days=value)
                self.assertEqual(result["expression"]["value"], expected.isoformat())

    def test_numeric_labels_cannot_merge_with_the_following_spelled_duration(self):
        for label, quantity, expected_days in (("1", "ninety-three", 93),
                                                ("2", "one hundred and ten", 110),
                                                ("5", "three", 3)):
            with self.subTest(label=label):
                text = f"I completed workshop {label} {quantity} days ago."
                result = describe_time(text, "2027-12-31")
                expected = datetime.date(2027, 12, 31) - datetime.timedelta(days=expected_days)
                self.assertEqual(result["expression"]["value"], expected.isoformat())

    def test_ordinal_day_and_weekday_keep_the_stated_day(self):
        cases = (
            ("I joined the workshop on the thirtieth of last month.", "2027-07-15", "2027-06-30"),
            ("I completed the workshop on Tuesday of the previous calendar week.", "2027-04-15", "2027-04-06"),
            ("I joined the workshop on Monday three weeks ago.", "2027-04-15", "2027-03-22"),
            ("I completed the workshop on the third of two calendar months before this message.", "2027-07-15", "2027-05-03"),
            ("I was born on the twenty-second of July 1981.", "2027-07-15", "1981-07-22"),
        )
        for text, source, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(describe_time(text, source)["expression"]["value"], expected)

    def test_broad_month_does_not_invent_day_precision(self):
        atoms, _ = ground_temporal_ledger(temporal_ledger(
            row("I attended the harbor workshop last month.", "t1", "s1", timestamp="2027-04-15"),
        ))
        self.assertEqual(atoms[0].temporal_expression["granularity"], "month")
        proof, _ = execute_temporal_proof(temporal_plan("harbor workshop"), atoms, reference_timestamp="2027-04-20")
        self.assertFalse(proof.executable)

    def test_invalid_explicit_day_never_falls_back_to_the_report_day(self):
        atoms, _ = ground_temporal_ledger(temporal_ledger(
            row("I completed the workshop on the thirty-first of February 2027.", "t1", "s1", timestamp="2027-04-15"),
        ))
        self.assertIsNone(atoms[0].temporal_expression)

    def test_latest_uses_event_time_not_report_time(self):
        question = "What was the latest workshop I completed?"
        plan = temporal_plan("workshop I completed", operator="LATEST", unit=None)
        proof, _ = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([
                frame("E1", predicate="completed", event_identity="harbor workshop"),
                frame("E2", predicate="completed", event_identity="gallery workshop"),
            ]), question, plan, temporal_ledger(
                row("I completed the harbor workshop on 2027-04-12.", "t1", "s1", timestamp="2027-04-15"),
                row("I completed the gallery workshop on 2027-04-10.", "t2", "s2", timestamp="2027-04-20"),
            ), effective_operation="temporal_latest",
        )
        self.assertEqual(proof.answer, "harbor workshop")
        self.assertEqual(proof.calculation["selected_date"], "2027-04-12")

    def test_equal_day_events_do_not_acquire_an_invented_order(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit", operator="ORDER"), temporal_ledger(
                row("I completed the harbor workshop on 2027-04-02.", "t1", "s1"),
                row("I completed the gallery visit on 2027-04-02.", "t2", "s2"),
            ),
        )
        self.assertFalse(proof.executable)
        self.assertEqual(proof.status, "CONFLICTING")

    def test_a_source_question_cannot_be_certified_as_a_completed_event(self):
        plan = temporal_plan("workshop", operator="LATEST", unit=None)
        proof, _ = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([frame("E1", predicate="completed", event_identity="harbor workshop")]),
            "What was the latest workshop?", plan, temporal_ledger(
                row("Did I complete the harbor workshop on 2027-04-12?", "t1", "s1"),
            ), effective_operation="temporal_latest",
        )
        self.assertFalse(proof.executable)

    def test_overlapping_coarse_extrema_are_not_certified(self):
        plan = temporal_plan("workshop", operator="LATEST", unit=None)
        proof, _ = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([
                frame("E1", predicate="completed", event_identity="harbor workshop"),
                frame("E2", predicate="completed", event_identity="gallery workshop"),
            ]), "What was the latest workshop?", plan, temporal_ledger(
                row("I completed the harbor workshop last month.", "t1", "s1", timestamp="2027-04-15"),
                row("I completed the gallery workshop on 2027-03-12.", "t2", "s2", timestamp="2027-04-20"),
            ), effective_operation="temporal_latest",
        )
        self.assertFalse(proof.executable)
        self.assertEqual(proof.status, "CONFLICTING")

    def test_requested_weeks_do_not_silently_turn_into_days(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("harbor workshop", "gallery visit", unit="weeks"), temporal_ledger(
                row("I completed the harbor workshop on 2027-04-02.", "t1", "s1"),
                row("I completed the gallery visit on 2027-04-12.", "t2", "s2"),
            ),
        )
        self.assertEqual(proof.answer, "1.4286 weeks")

    def test_completed_years_use_calendar_anniversary(self):
        proof, _ = execute_temporal_proof(
            temporal_plan("birth", "gallery visit", unit="years"), temporal_ledger(
                row("My birth was on 1981-07-22.", "t1", "s1"),
                row("I completed the gallery visit on 2027-07-21.", "t2", "s2"),
            ),
        )
        self.assertEqual(proof.answer, "45 years")

    def test_completed_future_event_conflicts_with_question_clock(self):
        proof, _ = execute_temporal_proof(temporal_plan("harbor workshop"), temporal_ledger(
            row("I completed the harbor workshop on 2027-04-22.", "t1", "s1"),
        ), reference_timestamp="2027-04-20")
        self.assertFalse(proof.executable)
        self.assertEqual(proof.reason, "completed_event_after_question_reference")

    def test_coordinated_plan_and_completion_keep_separate_assertions(self):
        quote = "I planned the harbor workshop, but I completed the gallery visit yesterday."
        atoms, _ = ground_temporal_ledger(temporal_ledger(row(quote, "t1", "s1", timestamp="2027-04-15")))
        self.assertEqual([a.event_status for a in atoms], ["planned", "completed"])
        self.assertEqual(atoms[1].temporal_expression["value"], "2027-04-14")
        self.assertTrue(all(a.source_quote == quote for a in atoms))

    def test_legacy_temporal_fallback_reuses_the_same_grounded_chain(self):
        question = "How many days passed between the harbor workshop and the gallery visit?"
        plan = temporal_plan("harbor workshop", "gallery visit")
        translated = semantic_plan_to_operation_plan(plan, infer_multi_session_operation(question, None, "2027-04-30"), question)
        atoms, _ = ground_temporal_ledger(temporal_ledger(
            row("I attended the harbor workshop twelve days ago.", "t1", "s1", timestamp="2027-04-14"),
            row("I completed the gallery visit three days after that workshop.", "t2", "s2", timestamp="2027-04-20"),
        ))
        result = execute_operation_plan(SelectAllOperationContext(), question, translated,
                                        grounded_temporal_evidence_rows(atoms), build_query_profile(question))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "3 days")

    def test_grounding_is_idempotent(self):
        first, _ = ground_temporal_ledger(temporal_ledger(
            row("I completed the harbor workshop yesterday.", "t1", "s1", timestamp="2027-04-15"),
        ))
        second, _ = ground_temporal_ledger(first)
        self.assertEqual(first, second)

    def test_state_route_change_is_confined_to_temporal_queries(self):
        weak = {"selector": STATE_SELECTOR_LATEST, "deterministic_enabled": False,
                "requested_attribute": None}
        temporal = temporal_plan("workshop", operator="LATEST", unit=None)
        factual = dataclasses.replace(temporal, operator="LOOKUP", answer_type="scalar",
                                      answer_role="object")
        self.assertFalse(state_history_overrides_compiler(weak, temporal))
        self.assertTrue(state_history_overrides_compiler(weak, factual))
        self.assertTrue(state_history_overrides_compiler(
            {**weak, "deterministic_enabled": True, "requested_attribute": "preferred editor"},
            temporal,
        ))


if __name__ == "__main__":
    unittest.main()
