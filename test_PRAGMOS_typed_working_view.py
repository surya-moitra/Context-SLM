"""Admission, projection and uncertainty contracts for the opt-in executor."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_typed_claims import actor_choices, source_windows, weekday_literals
from PRAGMOS_typed_computation import (
    established_slots_capability, plan_small_request, run_typed_computation,
    working_windows,
)
from test_PRAGMOS_typed_computation import bind, request, selection


class WorkingViewTests(unittest.TestCase):
    def test_repeated_entity_is_not_proof_of_a_repeated_fact(self):
        with self.assertRaisesRegex(ContractError, "grounded_antecedent"):
            bind(["I paid $8 at Cedar."], [selection(entity="Cedar", key="Cedar", kind="reference",
                                                     numbers=[0], quantity_kind="measure")])

    def test_dependency_expansion_keeps_indirect_inputs_and_untouched_qualifications(self):
        claims, windows, sources = bind(["My studio course workbook has 81 pages.",
                                        "I read 13 pages of that workbook.", "I paid $9 for coffee."], [
            selection(actor="My", entity="workbook", numbers=[0], quantity_kind="measure"),
            selection(entity="workbook", numbers=[0], quantity_kind="measure"),
            selection(entity="coffee", numbers=[0], quantity_kind="measure")])
        raw = list(windows.values())
        contract = {"scope_tokens": ["studio", "course"]}
        initial = working_windows("How many workbook pages remain?", contract, raw, {})
        self.assertEqual([w.source_id for w in initial], ["S1"])
        expanded = working_windows("How many workbook pages remain?", contract, raw, {claims[0].claim_id: claims[0]})
        self.assertEqual([w.source_id for w in expanded], ["S1", "S2"])
        self.assertEqual(sources["S2"].text, "I read 13 pages of that workbook.")

    def test_one_common_entity_word_or_a_measurement_cannot_expand_the_topic(self):
        claims, windows, _ = bind(["My community ceramics workbook has 67 pages.",
                                   "I read 9 pages of that workbook.",
                                   "The community funding report has 9 pages."], [
            selection(actor="My", entity="community ceramics workbook", numbers=[0], quantity_kind="measure"),
            selection(entity="9 pages", numbers=[0], quantity_kind="measure"),
            selection(actor=None, binding="event_property", event="community funding report",
                      entity="community funding report", numbers=[0], quantity_kind="measure")])
        chosen = working_windows("How many pages remain in the workbook?",
            {"scope_tokens": ["community", "ceramics"]}, list(windows.values()), {c.claim_id: c for c in claims[:2]})
        self.assertEqual([w.source_id for w in chosen], ["S1", "S2"])

    def test_delegation_cannot_discard_a_projection_role(self):
        base = request("count_distinct")
        self.assertTrue(established_slots_capability(base))
        for role in ("weekday", "event", "participant", "location"):
            with self.subTest(role=role):
                self.assertFalse(established_slots_capability(replace(base, key_role=role)))

    def test_source_weekdays_are_not_topic_labels_or_substrings(self):
        self.assertEqual(weekday_literals("I attended on Tuesday and on Tuesday."), ("Tuesday",))
        self.assertEqual(weekday_literals("The ThursdayClub met on Friday."), ("Friday",))
        self.assertEqual(weekday_literals("I attended studio sessions."), ())

    def test_prepositions_and_quantifiers_are_not_people(self):
        choices = actor_choices("For the class, every packet cost $4. Every packet was sold.")
        self.assertNotIn("For", choices)
        self.assertNotIn("Every", choices)
        self.assertIn("Mira", actor_choices("Mira paid $4."))

    def test_an_omitted_named_owner_cannot_become_an_impersonal_speaker_property(self):
        from PRAGMOS_typed_computation import select_small_claims
        from PRAGMOS_typed_claims import bind_typed_claim
        answers = {"decision": "support", "facts": [{"entity": "price", "actor": None}],
                   "status": "completed", "anchor": None}
        def choose(_prompt, _stage, schema, _reserve):
            return {key: answers[key] for key in schema["properties"]}
        source = Source("S", "s", "t", "user", "user", None, "Mira's price was $4.")
        window = source_windows({"S": source})[0]
        payload = select_small_claims(choose, "What was my total?", request(), window, source, {})
        with self.assertRaisesRegex(ContractError, "typed_actor_not_bound"):
            bind_typed_claim(window, source, payload["facts"][0], {})

    def test_calendar_numbers_never_become_decimal_precision_choices(self):
        answers = {"operation": "sum", "answer_kind": "a number or measured quantity",
                   "scope": "3 March 2024", "statuses": ["completed"]}
        fields = []
        def call(_prompt, _stage, schema, _reserve):
            fields.extend(schema["properties"])
            return {key: answers[key] for key in schema["properties"]}
        planned = plan_small_request(call, "What was the total on 3 March 2024?", {})
        self.assertIsNone(planned.precision)
        self.assertNotIn("precision", fields)

    def test_an_explicit_small_precision_is_still_available_beside_a_date(self):
        answers = {"operation": "sum", "answer_kind": "a number or measured quantity",
                   "scope": "March 2024", "precision": "two", "statuses": ["completed"]}
        def call(_prompt, _stage, schema, _reserve):
            if "precision" in schema["properties"]:
                self.assertEqual(set(schema["properties"]["precision"]["enum"]), {None, "two"})
            return {key: answers[key] for key in schema["properties"]}
        planned = plan_small_request(call, "Total in March 2024 to two decimal places?", {})
        self.assertEqual(planned.precision, 2)

    def test_unsupported_explicit_precision_is_rejected_not_silently_dropped(self):
        answers = {"operation": "mean", "answer_kind": "a number or measured quantity",
                   "precision": "nine", "statuses": ["completed"]}
        def choose(_prompt, _stage, schema, _reserve):
            return {key: answers[key] for key in schema["properties"]}
        with self.assertRaisesRegex(ContractError, "invalid_requested_precision"):
            plan_small_request(choose, "Average to nine decimal places?", {})

    def test_general_products_keep_their_semantic_factor_descriptions(self):
        answers = {"operation": "multiply", "answer_kind": "a number or measured quantity", "unit": None}
        def choose(_prompt, _stage, schema, _reserve):
            if set(schema["properties"]) == {"left", "right"}:
                if schema["properties"]["left"]["type"] == "string":
                    return {"left": "principal", "right": "interest rate"}
                return {"left": ["completed"], "right": ["completed"]}
            return {key: answers[key] for key in schema["properties"]}
        planned = plan_small_request(choose, "What interest does that principal earn at the stated rate?", {})
        self.assertEqual((planned.left_operand, planned.right_operand), ("principal", "interest rate"))

    def test_product_binding_does_not_relabel_ordinary_measured_factors(self):
        from PRAGMOS_typed_computation import product_factor_side
        self.assertIsNone(product_factor_side(request("multiply"), selection(quantity_kind="measure"), {}))

    def test_working_view_keeps_all_matching_inputs_instead_of_a_top_k_prefix(self):
        sources = {str(i): Source(str(i), "s", str(i), "user", "user", None,
                                 "I paid $8 for studio supplies.") for i in range(100)}
        windows = source_windows(sources)
        self.assertEqual(len(working_windows("What total studio supply cost?", {}, windows, {})), 100)

    def test_raw_view_noise_does_not_exhaust_literal_extraction_or_drop_a_third_transaction(self):
        sources = {"S": Source("S", "s", "t", "user", "user", None,
                              "I paid $8 at Cedar. I paid $9 at Birch. I paid $6 at Cedar.")}
        sources.update({f"N{i}": Source(f"N{i}", "s", str(i), "user", "user", None,
                                      "A completely unrelated note. " * 4) for i in range(40)})
        original = {key: value.text for key, value in sources.items()}
        windows = source_windows(sources)
        specs = [selection(entity=shop, key=shop, numbers=[0], quantity_kind="measure")
                 for shop in ("Cedar", "Birch", "Cedar")]
        report = {"calls": {"planner_calls": 0, "extractor_calls": 0}}
        with patch("PRAGMOS_typed_computation.plan_small_request", return_value=request(
                grouped=True, rank="max", projection="entity", key_role="location")), patch(
                "PRAGMOS_typed_computation.select_small_claims", side_effect=[
                    {"decision": "include", "facts": [spec]} for spec in specs]) as select:
            result = run_typed_computation(SimpleNamespace(count_tokens=lambda _: 10), "Which shop, Cedar or Birch?",
                sources, report, n_ctx=2048, max_answer_tokens=64, contract={"extended_contracts": True})
        self.assertEqual(result["answer"], "Cedar")
        self.assertEqual(select.call_count, 3)
        self.assertEqual(len(result["window_reviews"]), 3)
        self.assertEqual(len(result["working_view"]["outside_working_view"]), len(windows) - 3)
        self.assertFalse(result["working_view"]["global_completeness_proven"])
        self.assertEqual({key: value.text for key, value in sources.items()}, original)

    def test_unknown_relevant_statement_cannot_be_dropped_to_certify_a_partial_total(self):
        sources = {"S": Source("S", "s", "t", "user", "user", None, "I paid $8. I paid an unknown amount.")}
        windows = source_windows(sources)
        report = {"calls": {"planner_calls": 0, "extractor_calls": 0}}
        with patch("PRAGMOS_typed_computation.plan_small_request", return_value=request()), patch(
                "PRAGMOS_typed_computation.select_small_claims", side_effect=[
                    {"decision": "include", "facts": [selection(entity="paid", numbers=[0], quantity_kind="measure")]},
                    {"decision": "unknown", "facts": []}, {"decision": "unknown", "facts": []}]):
            with self.assertRaisesRegex(ContractError, "typed_source_selection_unresolved"):
                run_typed_computation(SimpleNamespace(count_tokens=lambda _: 10), "What total?", sources,
                                      report, n_ctx=2048, max_answer_tokens=64, contract={"extended_contracts": True})
        self.assertFalse(report["extraction_complete"])

    def test_per_member_signature_cannot_request_its_output_as_a_denominator(self):
        answers = {"operation": "divide", "answer_kind": "a number or measured quantity", "per_member": True, "unit": None}
        def choose(_prompt, _stage, schema, _reserve):
            if set(schema["properties"]) == {"left", "right"}:
                self.assertEqual(schema["properties"]["left"]["type"], "array")
                return {"left": ["completed"], "right": ["completed"]}
            return {key: answers[key] for key in schema["properties"]}
        query = plan_small_request(choose, "What was each equal member's price?", {})
        self.assertEqual(query.left_operand, "total measured value for all members of the same group")
        self.assertEqual(query.right_operand, "number of members sharing that total")

    def test_quantified_member_link_requires_one_matching_count_dimension(self):
        claims, windows, sources = bind(["I sold 7 packets.", "Every packet sold for $4."], [
            selection(entity="7 packets", attribute="packets", numbers=[0], quantity_kind="count", operand="left"),
            selection(actor=None, binding="linked_event", entity="packet", attribute="packet", numbers=[0],
                      quantity_kind="rate", operand="right", link_index=0)])
        from PRAGMOS_typed_computation import execute_typed_plan
        self.assertEqual(execute_typed_plan(request("multiply"), claims, windows, sources)["answer"], "$28")
        with self.assertRaisesRegex(ContractError, "no_source_link"):
            bind(["I sold 7 mugs.", "Every packet sold for $4."], [
                selection(entity="7 mugs", attribute="mugs", numbers=[0], quantity_kind="count", operand="left"),
                selection(actor=None, binding="linked_event", entity="packet", attribute="packet", numbers=[0],
                          quantity_kind="rate", operand="right", link_index=0)])

    def test_rate_denominator_cannot_be_the_price_span(self):
        from PRAGMOS_typed_computation import rate_denominators
        claims, _, _ = bind(["I sold 7 packets."], [selection(entity="7 packets", attribute="packets",
            numbers=[0], quantity_kind="count", operand="left")])
        source = Source("S", "s", "t", "user", "user", None, "Every packet sold for $4.")
        choices = rate_denominators(source_windows({"S": source})[0], {c.claim_id: c for c in claims})
        self.assertEqual(choices, ("packet",))


if __name__ == "__main__":
    unittest.main()
