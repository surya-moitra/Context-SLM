"""Contract, ownership, composition and promotion regression counterexamples."""

from dataclasses import replace
from decimal import Decimal
import unittest

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_state_contracts import SourceStateRelation, bind_answer_contract, bind_typed_value
from PRAGMOS_state_owners import OwnerIndex
from PRAGMOS_state_program import (
    Value, execute_program, plan_program, review_promotion, validate_answer, validate_output_type, validate_program,
)
from test_PRAGMOS_state_binding import fact
from test_PRAGMOS_state_contracts import contract_payloads


def contract(q="What is my balance?", prop="balance", **changes):
    return bind_answer_contract(q, *contract_payloads(q, prop, **changes))


def bound(value, day=1, column="P1", role="measurement", kind="quantity", dimension="currency", **changes):
    row = fact(value, day, **changes)
    relation = SourceStateRelation(column, row.actor, "is", row.attribute, None, role,
                                  bind_typed_value(value, kind, dimension), row.quote, (), None)
    return replace(row, column_id=column, relation=relation,
                   claim_review={"history_binding": column, "target_match": "bound", "qualification_match": "bound",
                                 "method": "independent_assertion_binding", **(changes.get("claim_review") or {})})


def select(node="N1", column="P1", owner="speaker:user", selector="latest", marker=None, modality="actual"):
    return {"id": node, "op": "select", "column": column, "owner": owner, "selector": selector,
            "marker": marker, "modality": modality}


def program(q, nodes, rounding=None):
    return {"nodes": nodes, "output": nodes[-1]["id"], "request_ref": q, "rounding_ref": rounding}


def read_program(c, **changes):
    return program(c.question, [select(**changes), {"id": "N2", "op": "read", "input": "N1"}])


class ProgramTests(unittest.TestCase):
    def test_latest_read_uses_complete_column_not_chosen_fact_ids(self):
        c = contract()
        result, _ = execute_program(read_program(c), c, [bound("$20"), bound("$35", 2)], {"speaker:user"})
        self.assertEqual(result["answer"], "$35")

    def test_unrelated_or_planned_uncertainty_does_not_abort_actual_read(self):
        c = contract()
        for unknown in ({"relevance": "irrelevant"}, {"operands": ["P2"]}, {"modality": "planned"}, {"owner_id": "named:mei"}):
            result, _ = execute_program(read_program(c), c, [bound("$20")], {"speaker:user"}, [unknown])
            self.assertEqual(result["answer"], "$20")

    def test_unresolved_required_assertion_cannot_certify_a_prefix(self):
        c = contract()
        with self.assertRaisesRegex(ContractError, "assertion_unresolved"):
            execute_program(read_program(c), c, [bound("$20")], {"speaker:user"}, [{"operands": ["P1"]}])

    def test_source_plan_does_not_override_actual_snapshot(self):
        c = contract()
        result, _ = execute_program(read_program(c), c, [bound("$20"), bound("$90", 2, modality="planned")], {"speaker:user"})
        self.assertEqual(result["answer"], "$20")

    def test_latest_record_is_not_an_arithmetic_operation(self):
        c = contract("What is my current record?", "record", projection="record", input_type="record", dimension="none")
        rows = [bound("3-1", role="record", kind="record", dimension="none"), bound("5-2", 2, role="record", kind="record", dimension="none")]
        result, value = execute_program(read_program(c), c, rows, {"speaker:user"})
        self.assertEqual(result["answer"], "5-2")
        self.assertEqual(validate_answer("3-1", c, value)["status"], "contradicted")

    def test_minimum_duration_excludes_plans_and_converts_units(self):
        c = contract("What is my best time?", "time", projection="duration", dimension="duration", family="reduce", selector="all")
        plan = program(c.question, [select(selector="all"), {"id": "N2", "op": "minimum", "input": "N1"}])
        rows = [bound("27 minutes", role="duration", dimension="duration"),
                bound("26 minutes 15 seconds", 2, role="duration", dimension="duration"),
                bound("24 minutes", 3, role="duration", dimension="duration", modality="planned")]
        result, _ = execute_program(plan, c, rows, {"speaker:user"})
        self.assertEqual(result["answer"], "26 minutes 15 seconds")

    def test_previous_to_latest_direction_does_not_compare_against_original(self):
        c = contract("Did my last change increase or decrease my limit?", "limit", projection="direction", dimension="scalar", family="compare")
        plan = program(c.question, [select(selector="previous"), {"id": "N2", "op": "read", "input": "N1"},
            select("N3"), {"id": "N4", "op": "read", "input": "N3"},
            {"id": "N5", "op": "direction", "left": "N2", "right": "N4"}])
        rows = [bound("5", dimension="scalar"), bound("2", 2, dimension="scalar"), bound("3", 3, dimension="scalar")]
        self.assertEqual(execute_program(plan, c, rows, {"speaker:user"})[0]["answer"], "increased")

    def test_composed_old_and_new_ratios_require_all_four_bound_inputs(self):
        c = contract("Did my water per syrup increase or decrease?", "water", projection="direction", dimension="scalar", family="compare", operand_refs=["water", "syrup"])
        nodes = []
        for col, selector in (("P1", "earliest"), ("P2", "earliest"), ("P1", "latest"), ("P2", "latest")):
            n = len(nodes) + 1
            nodes += [select("N" + str(n), column=col, selector=selector), {"id": "N" + str(n+1), "op": "read", "input": "N" + str(n)}]
        nodes += [{"id": "N9", "op": "divide", "left": "N2", "right": "N4"},
                  {"id": "N10", "op": "divide", "left": "N6", "right": "N8"},
                  {"id": "N11", "op": "direction", "left": "N9", "right": "N10"}]
        rows = [bound("9", dimension="scalar"), bound("3", 2, column="P2", dimension="scalar"),
                bound("14", 3, dimension="scalar"), bound("7", 4, column="P2", dimension="scalar")]
        self.assertEqual(execute_program(program(c.question, nodes), c, rows, {"speaker:user"})[0]["answer"], "decreased")

    def test_planning_never_receives_answers_or_literal_source_values(self):
        c = contract()
        prompts = []
        def ask(prompt, *args):
            prompts.append(prompt)
            return read_program(c)
        plan_program(ask, c, [bound("$987654")], {"speaker:user"})
        self.assertNotIn("987654", prompts[0])
        self.assertNotIn("results:", prompts[0])

    def test_dead_nodes_missing_columns_and_forward_references_are_rejected(self):
        c = contract()
        for change in (lambda p: p["nodes"][1].update(input="N3"),
                       lambda p: p["nodes"].append({"id": "N3", "op": "read", "input": "N1"})):
            plan = read_program(c); change(plan)
            with self.assertRaises(ContractError):
                validate_program(plan, c, [bound("$20")], {"speaker:user"})

    def test_source_ranges_are_read_intact_and_cannot_be_subtracted_as_exact(self):
        c = contract("How long is my duration?", "duration", projection="duration", dimension="duration")
        row = bound("about 12-16 hours", role="duration", dimension="duration")
        result, _ = execute_program(read_program(c), c, [row], {"speaker:user"})
        self.assertEqual(result["answer"], "about 12-16 hours")

    def test_initial_current_pair_does_not_discard_either_output(self):
        c = contract("What was my initial balance, and what is my balance now?", shape="pair")
        plan = program(c.question, [select(selector="earliest"), {"id": "N2", "op": "read", "input": "N1"},
            select("N3"), {"id": "N4", "op": "read", "input": "N3"},
            {"id": "N5", "op": "tuple", "inputs": ["N2", "N4"], "labels": ["initial balance", "balance now"]}])
        result, value = execute_program(plan, c, [bound("$20"), bound("$35", 2)], {"speaker:user"})
        self.assertIn("$20", result["answer"]); self.assertIn("$35", result["answer"])
        self.assertEqual(validate_answer("$20", c, value)["status"], "contradicted")

    def test_program_cannot_drop_a_third_requested_output(self):
        c = contract("What was my initial balance, and what is my balance now?", shape="pair")
        plan = program(c.question, [select(selector="earliest"), {"id": "N2", "op": "read", "input": "N1"},
            select("N3"), {"id": "N4", "op": "read", "input": "N3"},
            {"id": "N5", "op": "tuple", "inputs": ["N2", "N4"], "labels": ["initial balance", "balance now"]}])
        with self.assertRaisesRegex(ContractError, "required_output_count_mismatch"):
            execute_program(plan, c, [bound("$20"), bound("$35", 2)], {"speaker:user"}, required_outputs=3)

    def test_requested_list_cannot_be_replaced_with_a_scalar(self):
        c = contract("Which balances are in my list?", "balances", shape="list")
        with self.assertRaisesRegex(ContractError, "requires_collection"):
            validate_output_type(c, Value("text", "account A", ()))

    def test_one_fact_cannot_certify_two_time_endpoints(self):
        c = contract("How much did my balance change?", family="calculate")
        plan = program(c.question, [select(selector="earliest"), {"id": "N2", "op": "read", "input": "N1"},
            select("N3"), {"id": "N4", "op": "read", "input": "N3"},
            {"id": "N5", "op": "subtract", "left": "N4", "right": "N2"}])
        with self.assertRaisesRegex(ContractError, "distinct_operands"):
            execute_program(plan, c, [bound("$20")], {"speaker:user"})

    def test_cumulative_snapshots_cannot_be_summed_as_independent_contributions(self):
        c = contract("What is my total balance?", family="reduce", selector="all")
        plan = program(c.question, [select(selector="all"), {"id": "N2", "op": "sum", "input": "N1"}])
        with self.assertRaisesRegex(ContractError, "independent_contributions"):
            execute_program(plan, c, [bound("$20"), bound("$35", 2)], {"speaker:user"})


class OwnerTests(unittest.TestCase):
    def test_query_object_name_does_not_become_a_requested_person(self):
        owners = OwnerIndex()
        narrator = owners.reference("my", "user", "speaker")
        owners.reference("Meridian Lab", "user", "named")
        c = contract("What is my balance at Meridian Lab?", target_ref="Meridian Lab")
        self.assertEqual(owners.allowed(c), {narrator})

    def test_source_speaker_metadata_binds_first_person_to_the_named_speaker(self):
        owners = OwnerIndex()
        first_person = owners.reference("I", "Asha", "speaker", source_id="S4")
        named = owners.reference("Asha", "Ben", "named")
        self.assertEqual(owners.root(first_person), owners.root(named))
        c = contract("What is Asha's balance?", subject_ref="Asha", owner_scope="other")
        self.assertEqual(owners.allowed(c), {owners.root(named)})
        self.assertEqual(owners.as_dict()["speaker_links"][0]["source_id"], "S4")

    def test_role_name_is_not_used_as_a_person_name(self):
        owners = OwnerIndex()
        owners.reference("I", "user", "speaker", source_id="S1")
        self.assertEqual(owners.as_dict()["speaker_links"], [])
        self.assertNotIn("named:user", owners.labels)

    def test_two_named_speakers_keep_a_first_person_question_ambiguous(self):
        owners = OwnerIndex()
        owners.reference("I", "Asha", "speaker")
        owners.reference("I", "Ben", "speaker")
        with self.assertRaisesRegex(ContractError, "question_speaker_ambiguous"):
            owners.allowed(contract())

    def test_named_person_is_not_merged_with_narrator(self):
        owners = OwnerIndex()
        a = owners.reference("My", "user", "speaker")
        b = owners.reference("Leena", "user", "named")
        self.assertNotEqual(owners.root(a), owners.root(b))
        c = contract("Where has Leena moved?", "moved", projection="location", input_type="text", dimension="none", owner_scope="other", subject_ref="Leena")
        self.assertEqual(owners.allowed(c), {b})

    def test_explicit_named_relative_link_is_canonicalized_with_provenance(self):
        owners = OwnerIndex()
        s = Source("S1", "s1", "1", "user", "user", "2025-01-01", "My mother is Nora.")
        owners.link(s, "My mother", "related", "Nora", "named", s.text)
        self.assertEqual(owners.root("related:user:my mother"), owners.root("named:nora"))
        self.assertEqual(owners.as_dict()["links"][0]["source_id"], "S1")

    def test_same_relation_phrase_from_two_speakers_is_not_one_person(self):
        owners = OwnerIndex()
        self.assertNotEqual(owners.reference("my mother", "Asha", "related"), owners.reference("my mother", "Ben", "related"))

    def test_pronoun_or_unwitnessed_alias_cannot_create_named_identity(self):
        owners = OwnerIndex()
        with self.assertRaises(ContractError):
            owners.reference("she", "user", "named")
        s = Source("S", "s", "1", "user", "user", None, "Nora went shopping.")
        with self.assertRaises(ContractError):
            owners.link(s, "my mother", "related", "Nora", "named", s.text)


class PromotionTests(unittest.TestCase):
    def test_range_endpoints_cannot_be_added_to_certify_an_exact_measurement(self):
        c = contract("How long is my duration?", "duration", projection="duration", dimension="duration")
        value = Value("quantity", "12 hours to 16 hours", (), display_unit="hours")
        self.assertEqual(validate_answer("28 hours", c, value)["status"], "contradicted")

    def test_answer_range_is_not_equivalent_to_an_exact_source_measurement(self):
        c = contract("How long is my duration?", "duration", projection="duration", dimension="duration")
        value = Value("quantity", "28 hours", (), display_unit="hours")
        self.assertEqual(validate_answer("12 hours to 16 hours", c, value)["status"], "contradicted")

    def test_exact_compound_measurement_unit_conversion_remains_supported(self):
        c = contract("How long is my duration?", "duration", projection="duration", dimension="duration")
        value = Value("quantity", "90 minutes", (), display_unit="minutes")
        self.assertEqual(validate_answer("1 hour 30 minutes", c, value)["status"], "supported")

    def test_equivalent_units_preserve_an_existing_correct_answer(self):
        c = contract("How long is my duration?", "duration", projection="duration", dimension="duration")
        _, value = execute_program(read_program(c), c, [bound("3600 seconds", role="duration", dimension="duration")], {"speaker:user"})
        self.assertEqual(review_promotion("1 hour", "3600 seconds", c, value)["decision"], "preserve_supported_answer")

    def test_approximation_cannot_be_dropped_by_unit_normalization(self):
        c = contract("How long is my duration?", "duration", projection="duration", dimension="duration")
        _, value = execute_program(read_program(c), c, [bound("about 1 hour", role="duration", dimension="duration")], {"speaker:user"})
        self.assertEqual(validate_answer("1 hour", c, value)["reason"], "source_qualification_lost")

    def test_correct_answer_is_preserved_even_when_old_operation_label_differs(self):
        c = contract()
        _, output = execute_program(read_program(c), c, [bound("$20")], {"speaker:user"})
        self.assertEqual(review_promotion("$20", "$20", c, output)["decision"], "preserve_supported_answer")

    def test_unsupported_text_paraphrase_is_preserved_when_not_comparable(self):
        c = contract("Where is my object?", "object", projection="location", input_type="text", dimension="none")
        value = Value("text", "Northern Workshop", ())
        self.assertEqual(review_promotion("the workshop in the north", "Northern Workshop", c, value)["decision"], "preserve_unresolved_answer")

    def test_correct_nights_answer_cannot_be_displaced_by_a_points_balance(self):
        c = contract("How many nights can my points buy?", "points", projection="cardinality", dimension="count", family="calculate")
        c = replace(c, counted_entity="nights")
        points = bound("6800 points", dimension="count", role="cardinality")
        bad = Value("quantity", points.value, (points,), counted_entity="points")
        with self.assertRaisesRegex(ContractError, "wrong_counted_entity"):
            validate_output_type(c, bad)
        good = Value("number", Decimal(2), (), (("count:night", 1),), counted_entity="nights")
        self.assertEqual(review_promotion("2 standard free nights", "2", c, good)["decision"], "preserve_supported_answer")

    def test_missing_baseline_never_enables_promotion(self):
        c = contract()
        value = Value("number", Decimal(2), ())
        self.assertEqual(review_promotion(None, "2", c, value)["decision"], "preserve_unresolved_answer")

    def test_incompatible_boolean_cannot_be_treated_as_a_correct_entity_answer(self):
        c = contract("Do I have my spare tool?", "spare tool", projection="boolean", input_type="text", dimension="none", family="predicate", shape="boolean")
        with self.assertRaisesRegex(ContractError, "requires_boolean"):
            validate_output_type(c, Value("text", "toolkit", ()))


if __name__ == "__main__":
    unittest.main()
