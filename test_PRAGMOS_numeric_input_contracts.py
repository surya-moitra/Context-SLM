import dataclasses
import json
import unittest
from types import SimpleNamespace

from PRAGMOS_evidence_compiler import (
    OperationProof,
    QueryOperand,
    SemanticQueryPlan,
    validate_numeric_operation_proof_sources,
)


def numeric_plan(*operands, operation="SUM", status="completed"):
    return SemanticQueryPlan(
        operator=operation, answer_type="scalar", target_unit=None, actor="user",
        operands=tuple(QueryOperand(str(i), text, kind="quantity")
                       for i, text in enumerate(operands)),
        confidence="high", status="valid", requested_status=status,
    )


def numeric_fact(clause, value, turn="t1", **fields):
    return {
        "source_turn_id": turn, "source_session_id": "s1",
        "source_quote": clause, "clause": clause,
        "value": value, "status": "completed", **fields,
    }


def numeric_proof(*facts, operator="SUM"):
    return OperationProof(
        status="COMPLETE", operator=operator, answer="previous answer",
        executable=True,
        calculation={"provenance": list(facts), "proof_certificate": {"scope_bound": True}},
    )


class NumericInputContractTests(unittest.TestCase):
    def validate(self, plan, proof, *source_texts, operation=None):
        sources = {f"t{i}": {"session_id": "s1", "text": text}
                   for i, text in enumerate(source_texts, 1)}
        return validate_numeric_operation_proof_sources(
            plan, proof, sources, effective_operation=operation or plan.operator.lower(),
        )

    def test_raw_qualification_overrules_truncated_completed_status(self):
        clause = "I estimated $90 for the observatory trip"
        proof = numeric_proof(numeric_fact(clause, 90))
        result, audit = self.validate(
            numeric_plan("observatory trip expenses"), proof,
            clause + ", but that amount was only a budget and was never spent.",
        )
        self.assertFalse(result.executable)
        self.assertIsNone(result.answer)
        self.assertEqual(audit["invalid_input_indexes"], [0])
        self.assertIn("never spent", audit["source_checks"][0]["source"]["source_quote"])
        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "previous answer")
        json.dumps(result.as_dict())

    def test_required_item_is_not_imputed_from_a_supported_subtotal(self):
        clause = "I completed the observatory notebook purchase at $47."
        proof = numeric_proof(numeric_fact(clause, 47))
        result, audit = self.validate(
            numeric_plan("observatory expenses", "ticket price"), proof, clause,
        )
        self.assertFalse(result.executable)
        self.assertEqual(audit["invalid_input_indexes"], [])
        self.assertEqual(audit["missing_operand_ids"], ["1"])

    def test_invalid_budget_and_missing_item_are_independent_checks(self):
        clause = "I estimated $90 for the observatory trip"
        paid = "I completed the observatory notebook purchase at $47."
        result, audit = self.validate(
            numeric_plan("observatory expenses", "ticket price"),
            numeric_proof(numeric_fact(clause, 90), numeric_fact(paid, 47, "t2")),
            clause + ", but it was never spent.", paid,
        )
        self.assertFalse(result.executable)
        self.assertEqual(audit["invalid_input_indexes"], [0])
        self.assertEqual(audit["missing_operand_ids"], ["1"])

    def test_zero_is_a_present_value(self):
        clause = "I completed the observatory ticket purchase for $0."
        proof = numeric_proof(numeric_fact(clause, 0))
        result, audit = self.validate(
            numeric_plan("observatory expenses", "ticket price"), proof, clause,
        )
        self.assertIs(result, proof)
        self.assertEqual(audit["missing_operand_ids"], [])

    def test_none_is_not_a_zero_value(self):
        clause = "I discussed the ticket; its price was not recorded."
        proof = numeric_proof(numeric_fact(clause, None))
        result, audit = self.validate(
            numeric_plan("observatory expenses", "ticket price"), proof, clause,
        )
        self.assertFalse(result.executable)
        self.assertEqual(audit["missing_operand_ids"], ["0", "1"])

    def test_non_numeric_and_non_finite_inputs_do_not_cover_an_operand(self):
        clause = "The recorded ticket price is unavailable."
        for value in ("unknown", float("nan"), float("inf")):
            with self.subTest(value=value):
                result, audit = self.validate(
                    numeric_plan("observatory expenses", "ticket price"),
                    numeric_proof(numeric_fact(clause, value)), clause,
                )
                self.assertFalse(result.executable)
                self.assertEqual(audit["missing_operand_ids"], ["0", "1"])

    def test_missing_operand_check_is_not_specific_to_totals_or_money(self):
        for operation in ("SUM", "AVERAGE", "DIFFERENCE", "ARGMAX", "ARGMIN"):
            with self.subTest(operation=operation):
                clause = "I completed the outward walk in 25 minutes."
                result, audit = self.validate(
                    numeric_plan("outward duration", "return duration", operation=operation),
                    numeric_proof(numeric_fact(clause, 25), operator=operation), clause,
                )
                self.assertFalse(result.executable)
                self.assertEqual(audit["missing_operand_ids"], ["1"])

    def test_qualifications_apply_to_other_units_and_aggregates(self):
        for operation in ("SUM", "AVERAGE", "DIFFERENCE", "ARGMAX", "ARGMIN"):
            with self.subTest(operation=operation):
                clause = "The forest route was estimated at 12 kilometers"
                result, audit = self.validate(
                    numeric_plan("forest route", operation=operation),
                    numeric_proof(numeric_fact(clause, 12), operator=operation),
                    clause + ", but that distance was never completed.",
                )
                self.assertFalse(result.executable)
                self.assertEqual(audit["invalid_input_indexes"], [0])

    def test_same_operand_can_have_many_completed_inputs(self):
        first = "My completed observatory notebook expense was $21."
        second = "My completed observatory ticket expense was $34."
        proof = numeric_proof(numeric_fact(first, 21), numeric_fact(second, 34, "t2"))
        result, audit = self.validate(numeric_plan("observatory expenses"), proof, first, second)
        self.assertIs(result, proof)
        self.assertFalse(audit["blocked"])

    def test_broad_operand_and_required_item_can_overlap(self):
        first = "I completed the observatory notebook purchase for $21."
        second = "I completed the observatory ticket purchase for $34."
        proof = numeric_proof(numeric_fact(first, 21), numeric_fact(second, 34, "t2"))
        result, audit = self.validate(
            numeric_plan("observatory expenses", "ticket price"), proof, first, second,
        )
        self.assertIs(result, proof)
        self.assertEqual(audit["operand_coverage"][0]["input_indexes"], [0, 1])
        self.assertEqual(audit["operand_coverage"][1]["input_indexes"], [1])

    def test_existing_semantic_operand_binding_survives_paraphrases(self):
        first = "I completed the taxi trip for $42."
        second = "I completed the train trip for $19."
        proof = numeric_proof(
            numeric_fact(first, 42, bound_operand_index=0),
            numeric_fact(second, 19, "t2", bound_operand_index=1),
            operator="DIFFERENCE",
        )
        result, audit = self.validate(
            numeric_plan("taxi fare", "rail fare", operation="DIFFERENCE"),
            proof, first, second,
        )
        self.assertIs(result, proof)
        self.assertFalse(audit["blocked"])

    def test_another_event_negation_does_not_poison_a_valid_input(self):
        clause = "I spent $21 on the observatory notebook"
        for ending in (
            ", but I never purchased the ticket.",
            ". I never bought the ticket.",
            ", but that ticket was never purchased.",
        ):
            with self.subTest(ending=ending):
                proof = numeric_proof(numeric_fact(clause, 21))
                result, audit = self.validate(numeric_plan("notebook expenses"), proof, clause + ending)
                self.assertIs(result, proof)
                self.assertFalse(audit["blocked"])

    def test_same_value_in_another_event_is_not_the_selected_claim(self):
        clause = "I spent $21 on the observatory notebook"
        proof = numeric_proof(numeric_fact(clause, 21))
        result, audit = self.validate(
            numeric_plan("notebook expenses"), proof,
            clause + ". I estimated $21 for the ticket, but that amount was never spent.",
        )
        self.assertIs(result, proof)
        self.assertFalse(audit["blocked"])

    def test_a_qualification_cannot_absorb_another_events_negation(self):
        clause = "I recorded $21 for the notebook"
        for ending in (
            ", but that amount was spent and I never bought a ticket.",
            ", but that amount was spent while the ticket was never purchased.",
            ", but that amount was spent and Mira never bought a ticket.",
        ):
            with self.subTest(ending=ending):
                proof = numeric_proof(numeric_fact(clause, 21))
                result, audit = self.validate(numeric_plan("notebook expenses"), proof, clause + ending)
                self.assertIs(result, proof)
                self.assertFalse(audit["blocked"])

    def test_guard_does_not_reinterpret_an_existing_multi_event_clause(self):
        clause = "I spent $21 on a notebook, but I never purchased a ticket."
        proof = numeric_proof(numeric_fact(clause, 21))
        result, audit = self.validate(numeric_plan("notebook expenses"), proof, clause)
        self.assertIs(result, proof)
        self.assertFalse(audit["blocked"])

    def test_explicitly_excluded_item_is_not_added_as_a_mandatory_operand(self):
        clause = "I completed the observatory notebook purchase at $21."
        proof = numeric_proof(numeric_fact(clause, 21))
        result, audit = self.validate(
            numeric_plan("observatory notebook expenses"), proof,
            clause + " The ticket price was not recorded.",
        )
        self.assertIs(result, proof)
        self.assertFalse(audit["blocked"])

    def test_mention_in_another_sentence_is_not_operand_coverage(self):
        clause = "I spent $21 on the observatory notebook."
        proof = numeric_proof(numeric_fact(clause, 21))
        result, audit = self.validate(
            numeric_plan("observatory expenses", "ticket price"), proof,
            clause + " I discussed a ticket without recording its price.",
        )
        self.assertFalse(result.executable)
        self.assertEqual(audit["missing_operand_ids"], ["1"])

    def test_an_unpriced_item_in_the_same_sentence_is_not_a_numeric_binding(self):
        clause = ("I completed the observatory notebook purchase for $21 "
                  "and considered a ticket without recording its price.")
        proof = numeric_proof(numeric_fact(
            clause, 21, frame_id="F1", object="notebook", event_context="observatory",
        ))
        result, audit = self.validate(
            numeric_plan("observatory expenses", "ticket price"), proof, clause,
        )
        self.assertFalse(result.executable)
        self.assertEqual(audit["missing_operand_ids"], ["1"])

    def test_estimated_budget_question_is_not_treated_as_completed_spending(self):
        clause = "I estimated $90 for the observatory trip"
        for status in ("planned", "pending"):
            with self.subTest(status=status):
                proof = numeric_proof(numeric_fact(clause, 90, status="planned"))
                result, audit = self.validate(
                    numeric_plan("observatory budget", status=status), proof,
                    clause + ", but that amount was only a budget and was never spent.",
                )
                self.assertIs(result, proof)
                self.assertFalse(audit["blocked"])

    def test_unknown_status_cannot_upgrade_an_explicitly_cancelled_input(self):
        clause = "I estimated $90 for the observatory trip"
        proof = numeric_proof(numeric_fact(clause, 90))
        result, audit = self.validate(
            numeric_plan("observatory expenses", status="unknown"), proof,
            clause + ", but that amount was never spent.",
        )
        self.assertFalse(result.executable)
        self.assertTrue(audit["blocked"])

    def test_explicitly_ineligible_typed_inputs_cannot_cover_completed_operands(self):
        clause = "The observatory ticket price is $21."
        for status in ("negated", "planned", "pending"):
            with self.subTest(status=status):
                result, audit = self.validate(
                    numeric_plan("observatory expenses", "ticket price"),
                    numeric_proof(numeric_fact(clause, 21, status=status)), clause,
                )
                self.assertFalse(result.executable)
                self.assertEqual(audit["invalid_input_indexes"], [0])
                self.assertEqual(audit["missing_operand_ids"], ["0", "1"])

    def test_unresolved_source_is_not_fabricated_or_used_as_negative_evidence(self):
        clause = "I spent $21 on the observatory notebook."
        proof = numeric_proof(numeric_fact(clause, 21))
        result, audit = self.validate(numeric_plan("notebook expenses"), proof)
        self.assertIs(result, proof)
        self.assertFalse(audit["source_checks"][0]["source_resolved"])

    def test_session_identity_must_match_before_applying_source_qualifications(self):
        clause = "I estimated $90 for the observatory trip"
        proof = numeric_proof(numeric_fact(clause, 90, source_session_id="different-session"))
        result, audit = self.validate(
            numeric_plan("observatory expenses"), proof,
            clause + ", but that amount was never spent.",
        )
        self.assertIs(result, proof)
        self.assertFalse(audit["source_checks"][0]["source_resolved"])

    def test_structured_source_objects_and_integer_turn_ids(self):
        clause = "I estimated $90 for the observatory trip"
        proof = numeric_proof(numeric_fact(clause, 90, turn=1))
        result, audit = validate_numeric_operation_proof_sources(
            numeric_plan("observatory expenses"), proof,
            {1: SimpleNamespace(session_id="s1", text=clause + ", but it was never spent.")},
            effective_operation="sum",
        )
        self.assertFalse(result.executable)
        self.assertTrue(audit["blocked"])

    def test_non_numeric_and_existing_failure_paths_remain_identical(self):
        for operation in ("LOOKUP", "COUNT", "COUNT_DISTINCT", "COLLECT", "ELAPSED_TIME"):
            with self.subTest(operation=operation):
                proof = numeric_proof(operator=operation)
                result, audit = self.validate(numeric_plan(operation=operation), proof)
                self.assertIs(result, proof)
                self.assertFalse(audit["applicable"])
        for proof in (None, OperationProof("PARTIAL", "SUM", None)):
            result, audit = self.validate(numeric_plan("expenses"), proof)
            self.assertIs(result, proof)
            self.assertFalse(audit["applicable"])
        proof = numeric_proof()
        plan = dataclasses.replace(numeric_plan("expenses"), status="invalid")
        result, audit = self.validate(plan, proof)
        self.assertIs(result, proof)
        self.assertFalse(audit["applicable"])


if __name__ == "__main__":
    unittest.main()
