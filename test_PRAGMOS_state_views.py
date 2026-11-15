"""State contracts and source/projection counterexamples, without dataset fixtures."""

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_state_views import (
    Observation, accepted_state_candidate, bind_observations, execute_state_contract,
    contract_schema, direct_literal_projection, extract_window, observation_schema, phrase_present,
    plan_state_contract, run_state_candidate, scoped_sources, select_state, source_windows, validate_contract,
)
from test_PRAGMOS_composable_candidate import context


def slot(**overrides):
    return {"label": "balance", "subject": None, "property": "balance", "anchors": [],
            "selector": "latest", "marker": None, "value_kind": "quantity", "status": "actual", **overrides}


def contract(operation="read", slots=None, **overrides):
    return {"operation": operation, "form": "scalar", "slots": slots or [slot()],
            "whole_units": False, **overrides}


def observation(value, day=1, **overrides):
    return Observation(**{"slot": 0, "source_id": "S" + str(day), "session_id": "session" + str(day),
                          "turn_id": str(day), "speaker": "user", "role": "user",
                          "timestamp": f"2025-01-{day:02d}T12:00:00+00:00",
                          "quote": "My balance is " + value + ".", "value": value,
                          "actor": "My", "property": "balance", "change": "snapshot", "status": "actual",
                          "effective_date": None, "qualification": None, "complete_set": False,
                          "offset": 0, **overrides})


def source_and_payload(value="$29", **overrides):
    text = "My balance is " + value + "."
    source = Source("S1", "opaque-session", "1", "user", "user", "2025-01-01", text)
    row = {"slot": 0, "quote": text, "value": value, "actor": "My", "property": "balance",
           "change": "snapshot", "status": "actual", "effective_date": None,
           "qualification": None, "complete_set": False, **overrides}
    return source, {"observations": [row], "unresolved": False}


class ContractTests(unittest.TestCase):
    def test_measurement_cannot_be_coerced_to_identity_count(self):
        with self.assertRaisesRegex(ContractError, "member_identities_not_measurements"):
            validate_contract(contract("count_distinct"), "What is my balance?")

    def test_read_contract_preserves_literal_range(self):
        plan = validate_contract(contract(slots=[slot(label="hours", property="hours")]), "How many hours?")
        self.assertEqual(execute_state_contract(plan, [observation("11-14 hours")])["answer"], "11-14 hours")

    def test_two_output_slots_cannot_be_scalarized(self):
        with self.assertRaisesRegex(ContractError, "unused_output_slots"):
            validate_contract(contract(slots=[slot(), slot()]), "What balance?")

    def test_every_question_slot_is_literal(self):
        with self.assertRaisesRegex(ContractError, "not_source_grounded"):
            validate_contract(contract(slots=[slot(property="invented attribute")]), "What balance?")

    def test_bare_boolean_is_not_forced_to_numeric_comparison(self):
        plan = validate_contract(contract("exists", [slot(label="spare", property="spare", value_kind="text")],
                                          form="boolean"), "Do I have a spare?")
        self.assertEqual(execute_state_contract(plan, [observation("spare")])["answer"], "Yes")

    def test_boolean_operator_cannot_return_entity(self):
        with self.assertRaisesRegex(ContractError, "boolean_output_required"):
            validate_contract(contract("exists"), "What balance?")

    def test_markers_must_be_question_bound_and_executable(self):
        for marker in ("invented", None):
            with self.subTest(marker=marker), self.assertRaises(ContractError):
                validate_contract(contract(slots=[slot(selector="before_value", marker=marker)]), "Before Sapphire balance?")

    def test_read_cannot_use_list_shape(self):
        with self.assertRaisesRegex(ContractError, "read_output_mismatch"):
            validate_contract(contract(form="list"), "What balance?")

    def test_unauthorized_rounding_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "rounding_requires"):
            validate_contract(contract(whole_units=True), "What balance?")


class BindingTests(unittest.TestCase):
    def test_possessive_actor_and_source_addresses_are_retained(self):
        source, payload = source_and_payload()
        rows = bind_observations(payload, contract(), source, source.text, 0)
        self.assertEqual(rows[0].role, "user")
        self.assertEqual(rows[0].actor, "My")
        self.assertEqual(source.text[rows[0].offset:rows[0].offset + len(rows[0].value)], "$29")

    def test_fabricated_value_is_not_silently_dropped(self):
        source, payload = source_and_payload(value="$29")
        payload["observations"][0]["value"] = "$88"
        with self.assertRaisesRegex(ContractError, "not_source_grounded"):
            bind_observations(payload, contract(), source, source.text, 0)

    def test_unit_word_alone_cannot_replace_the_measured_value(self):
        source, payload = source_and_payload("18 dozen")
        payload["observations"][0]["value"] = "dozen"
        with self.assertRaisesRegex(ContractError, "no_quantity"):
            bind_observations(payload, contract(), source, source.text, 0)

    def test_source_qualification_cannot_be_removed_from_a_value(self):
        source, payload = source_and_payload("approximately $29", qualification="approximately")
        payload["observations"][0]["value"] = "$29"
        with self.assertRaisesRegex(ContractError, "qualification_cannot_be_stripped"):
            bind_observations(payload, contract(), source, source.text, 0)

    def test_wrong_actor_is_not_the_narrator(self):
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", "Nora's balance is $29.")
        _, payload = source_and_payload(actor="Nora", quote=source.text)
        with self.assertRaisesRegex(ContractError, "wrong_actor"):
            bind_observations(payload, contract(), source, source.text, 0)

    def test_named_actors_are_bound_without_becoming_user(self):
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", "Nora's balance is $29.")
        _, payload = source_and_payload(actor="Nora", quote=source.text)
        rows = bind_observations(payload, contract(slots=[slot(subject="Nora")]), source, source.text, 0)
        self.assertEqual(rows[0].actor, "Nora")

    def test_composite_scope_does_not_match_one_shared_word(self):
        source, payload = source_and_payload()
        with self.assertRaisesRegex(ContractError, "mandatory_scope_not_bound"):
            bind_observations(payload, contract(slots=[slot(anchors=["travel balance"])]), source, source.text, 0)
        self.assertFalse(phrase_present("table tennis", "tennis"))
        self.assertFalse(phrase_present("signed posters", "signed postcards"))
        self.assertTrue(phrase_present("North Clinic", "At NORTH CLINIC today"))

    def test_actor_and_value_cannot_be_donated_by_different_assertions(self):
        text = "My balance is $21. Nora's balance is $29."
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", text)
        _, payload = source_and_payload(quote=text)
        with self.assertRaisesRegex(ContractError, "crosses_assertion_boundaries"):
            bind_observations(payload, contract(), source, text, 0)

    def test_extractor_uncertainty_is_not_evidence_of_absence(self):
        source, payload = source_and_payload()
        payload["unresolved"] = True
        with self.assertRaisesRegex(ContractError, "extraction_unresolved"):
            bind_observations(payload, contract(), source, source.text, 0)

    def test_effective_dates_cannot_be_invented(self):
        source, payload = source_and_payload(effective_date="2024-01-01")
        with self.assertRaisesRegex(ContractError, "not_source_grounded"):
            bind_observations(payload, contract(), source, source.text, 0)

    def test_repeated_literal_requires_an_unambiguous_address(self):
        source, payload = source_and_payload()
        text = source.text + " " + source.text
        with self.assertRaisesRegex(ContractError, "ambiguous_literal_address"):
            bind_observations(payload, contract(), replace(source, text=text), text, 0)


class StateSelectionTests(unittest.TestCase):
    def test_latest_earliest_and_previous_do_not_erase_history(self):
        rows = [observation("$20"), observation("$37", 2), observation("$51", 3)]
        before = copy.deepcopy(rows)
        for selector, value in (("latest", "$51"), ("earliest", "$20"), ("previous", "$37")):
            self.assertEqual(select_state(slot(selector=selector), list(reversed(rows)))[0].value, value)
        self.assertEqual(rows, before)

    def test_late_report_of_an_old_value_does_not_reset_state(self):
        rows = [observation("$20"), observation("$37", 2), observation("$20", 3, change="reference")]
        self.assertEqual(select_state(slot(), rows)[0].value, "$37")

    def test_repeated_same_value_is_not_an_extra_update(self):
        rows = [observation("$20"), observation("$37", 2), observation("$37", 3, change="correction")]
        self.assertEqual(select_state(slot(selector="previous"), rows)[0].value, "$20")

    def test_plans_do_not_override_completed_facts(self):
        rows = [observation("East"), observation("West", 2, status="planned")]
        self.assertEqual(select_state(slot(value_kind="text"), rows)[0].value, "East")
        self.assertEqual(select_state(slot(value_kind="text", status="planned"), rows)[0].value, "West")

    def test_explicit_effective_date_precedes_report_time(self):
        rows = [observation("$20"), observation("$37", 2), observation("$10", 3, effective_date="2024-01-01")]
        self.assertEqual(select_state(slot(), rows)[0].value, "$37")
        self.assertEqual(select_state(slot(selector="earliest"), rows)[0].value, "$10")

    def test_structured_turn_timestamp_formats_retain_time_of_day(self):
        rows = [observation("$20", timestamp="2025/01/01 (Wed) 14:00"),
                observation("$37", 2, timestamp="2025/01/01 (Wed) 16:00")]
        self.assertEqual(select_state(slot(), list(reversed(rows)))[0].value, "$37")

    def test_concurrent_conflicting_sources_are_not_arbitrarily_sorted(self):
        rows = [observation("$20"), observation("$37", source_id="other", session_id="other", turn_id="other")]
        with self.assertRaisesRegex(ContractError, "tie_conflict"):
            select_state(slot(), rows)

    def test_marker_selects_immediate_predecessor(self):
        rows = [observation("Amber"), observation("Jade", 2), observation("Sapphire", 3)]
        self.assertEqual(select_state(slot(selector="before_value", marker="Sapphire"), rows)[0].value, "Jade")

    def test_unknown_marker_is_not_latest(self):
        with self.assertRaisesRegex(ContractError, "marker_not_unique"):
            select_state(slot(selector="before_value", marker="Ruby"), [observation("Jade")])

    def test_possession_statements_do_not_erase_other_items(self):
        rows = [observation("city bike"), observation("trail bike", 2),
                observation("folding bike", 3, change="add"), observation("city bike", 4, change="remove")]
        plan = contract("count_distinct", [slot(value_kind="member", selector="all")])
        self.assertEqual(execute_state_contract(plan, rows)["answer"], "2")

    def test_complete_snapshot_can_replace_membership(self):
        rows = [observation("city bike"), observation("trail bike", 2, complete_set=True)]
        self.assertEqual([r.value for r in select_state(slot(value_kind="member"), rows)], ["trail bike"])

    def test_empty_or_reference_only_input_cannot_certify_absence(self):
        for rows in ([], [observation("$20", change="reference")]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ContractError, "operand_unresolved"):
                select_state(slot(), rows)

    def test_increment_is_not_a_replacement_snapshot(self):
        with self.assertRaisesRegex(ContractError, "increment_cannot_replace"):
            select_state(slot(), [observation("$20", change="add")])

    def test_conflicting_annotations_of_same_value_are_not_silently_deduplicated(self):
        row = observation("$20")
        with self.assertRaisesRegex(ContractError, "conflicting_annotations"):
            select_state(slot(), [row, replace(row, change="denied")])


class ExecutionTests(unittest.TestCase):
    def test_initial_current_pair_retains_both_slots(self):
        plan = contract(slots=[slot(selector="earliest"), slot(label="now")], form="tuple")
        rows = [observation("6"), observation("8", 2), observation("6", slot=1), observation("8", 2, slot=1)]
        self.assertEqual(execute_state_contract(plan, rows)["answer"], "balance: 6; now: 8")

    def test_measured_extremum_returns_value_not_entity_count_or_label(self):
        rows = [observation("48 points"), observation("63 points", 2), observation("59 points", 3)]
        proof = execute_state_contract(contract("maximum", [slot(selector="all")]), rows)
        self.assertEqual(proof["answer"], "63 points")

    def test_stated_duration_is_read_not_elapsed_date_arithmetic(self):
        proof = execute_state_contract(contract(), [observation("seven months")])
        self.assertEqual(proof["answer"], "seven months")

    def test_inventory_packaging_is_not_substring_canonicalized(self):
        self.assertEqual(execute_state_contract(contract(), [observation("18 dozen eggs")])["answer"], "18 dozen eggs")

    def test_structured_record_is_not_subtracted(self):
        self.assertEqual(execute_state_contract(contract(), [observation("7-3")])["answer"], "7-3")

    def test_difference_is_python_arithmetic_with_unit_conversion(self):
        plan = contract("difference", [slot(), slot()])
        rows = [observation("90 kg"), observation("76500 g", 2, slot=1)]
        self.assertEqual(execute_state_contract(plan, rows)["answer"], "13.5 kg")

    def test_mismatched_units_cannot_be_combined(self):
        with self.assertRaisesRegex(ContractError, "incompatible"):
            execute_state_contract(contract("difference", [slot(), slot()]),
                                   [observation("90 kg"), observation("12 hours", 2, slot=1)])

    def test_whole_entitlement_uses_current_balance_before_division(self):
        plan = contract("quotient", [slot(), slot()], whole_units=True)
        rows = [observation("12000"), observation("15800", 2), observation("7000", slot=1)]
        self.assertEqual(execute_state_contract(plan, rows)["answer"], "2")

    def test_same_quantity_twice_is_not_dropped_from_an_occurrence_total(self):
        proof = execute_state_contract(contract("sum", [slot(selector="all")]),
                                       [observation("$5"), observation("$5", 2)])
        self.assertEqual(proof["answer"], "10 usd")

    def test_qualification_blocks_certifying_an_exact_total(self):
        with self.assertRaisesRegex(ContractError, "qualification_blocks"):
            execute_state_contract(contract("sum", [slot(selector="all")]),
                                   [observation("$20", qualification="approximately")])

    def test_range_cannot_be_reduced_to_one_arithmetic_operand(self):
        with self.assertRaisesRegex(ContractError, "one_exact_measurement"):
            execute_state_contract(contract("difference", [slot(), slot()]),
                                   [observation("11-14 hours"), observation("3 hours", slot=1)])

    def test_approximation_in_source_blocks_arithmetic_even_if_model_omits_flag(self):
        with self.assertRaisesRegex(ContractError, "interval_contract"):
            execute_state_contract(contract("sum", [slot(selector="all")]),
                                   [observation("$29", quote="My balance is approximately $29.")])

    def test_explicit_denial_is_not_a_positive_exists(self):
        proof = execute_state_contract(contract("exists", [slot(value_kind="text")], form="boolean"),
                                       [observation("spare", change="denied")])
        self.assertEqual(proof["answer"], "No")

    def test_missing_binary_operand_cannot_generate_a_number(self):
        with self.assertRaisesRegex(ContractError, "operand_unresolved"):
            execute_state_contract(contract("difference", [slot(), slot()]), [observation("90 kg")])


class IsolationAndCoverageTests(unittest.TestCase):
    def run_candidate(self, ctx, sources):
        with patch("PRAGMOS_composable_candidate.schema_grammar", return_value=None):
            return run_state_candidate(ctx, "What is my balance?", sources, n_ctx=2048, baseline_answer="$12", protocol="bound_v3")

    def test_complete_result_is_isolated_and_only_candidate_mode_can_accept(self):
        source, payload = source_and_payload()
        from test_PRAGMOS_state_binding import bound_responses
        ctx = context(bound_responses([source]))
        before = copy.deepcopy(source)
        result = self.run_candidate(ctx, {source.source_id: source})
        self.assertEqual(result["answer"], "$29")
        self.assertTrue(ctx.llm.restored)
        self.assertEqual(ctx.llm.position, 0)
        self.assertEqual(source, before)
        self.assertTrue(accepted_state_candidate(result, "candidate"))
        self.assertFalse(accepted_state_candidate(result, "shadow"))
        self.assertFalse(accepted_state_candidate(result, "off"))
        self.assertFalse(result["semantic_equivalence_verified"])

    def test_failed_extraction_restores_state_and_cannot_promote(self):
        source, payload = source_and_payload()
        from test_PRAGMOS_state_binding import bound_responses
        ctx = context(bound_responses([source], unresolved_last=True))
        result = self.run_candidate(ctx, {source.source_id: source})
        self.assertEqual(result["status"], "NOT_PROVEN")
        self.assertIsNone(result["answer"])
        self.assertTrue(ctx.llm.restored)

    def test_restore_failure_is_fatal_not_silent_fallback(self):
        source, payload = source_and_payload()
        from test_PRAGMOS_state_binding import bound_responses
        ctx = context(bound_responses([source]))
        ctx.llm.load_state = lambda state: (_ for _ in ()).throw(RuntimeError("failed"))
        with self.assertRaisesRegex(RuntimeError, "could not restore"):
            self.run_candidate(ctx, {source.source_id: source})

    def test_every_selected_source_is_reviewed(self):
        a, payload = source_and_payload()
        b = replace(a, source_id="S2", turn_id="2", timestamp="2025-01-02")
        from test_PRAGMOS_state_binding import bound_responses
        ctx = context(bound_responses([a, b], unresolved_last=True))
        result = self.run_candidate(ctx, {a.source_id: a, b.source_id: b})
        self.assertFalse(result["source_view_complete"])
        self.assertFalse(accepted_state_candidate(result, "candidate"))

    def test_source_windows_cover_all_nonwhitespace_without_prefix_truncation(self):
        text = "My first note is here. My second note is here. My third note is here."
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", text)
        windows = source_windows({"S1": source}, len, maximum=28)
        self.assertEqual(" ".join(t for _, _, t in windows), text)

    def test_oversized_atom_is_not_silently_truncated(self):
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", "x" * 100)
        with self.assertRaisesRegex(ContractError, "atom_exceeds_budget"):
            source_windows({"S1": source}, len, maximum=20)

    def test_dataset_identifiers_never_enter_model_prompts(self):
        source, payload = source_and_payload()
        from test_PRAGMOS_state_binding import bound_responses
        ctx = context(bound_responses([source]))
        self.run_candidate(ctx, {source.source_id: source})
        for prompt, _ in ctx.llm.calls:
            self.assertNotIn(source.session_id, prompt)
            self.assertNotIn("question_type", prompt)
            self.assertNotIn("reference_answer", prompt)


class LiteralProjectionTests(unittest.TestCase):
    def sources(self):
        a = Source("S1", "a", "1", "user", "user", "2025-01-01", "I had used my band for 3 months.")
        b = Source("S2", "b", "2", "user", "user", "2025-02-01", "I have used my band for 7 months.")
        return {"S1": a, "S2": b}

    def test_literal_value_can_be_projected_without_inventing_property_labels(self):
        proof = direct_literal_projection(lambda prompt: "I have used it for 7 months.",
                                          "How long have I used my band?", self.sources(), lambda s: len(s) // 5)
        self.assertEqual(proof["answer"], "7 months")
        self.assertEqual(proof["selected_observations"][0][0]["turn_id"], "2")
        self.assertFalse(proof["semantic_equivalence_verified"])

    def test_calculation_result_not_stated_in_sources_does_not_become_a_projection(self):
        proof = direct_literal_projection(lambda prompt: "4 months", "What difference did I report?",
                                          self.sources(), lambda s: len(s) // 5)
        self.assertIsNone(proof)

    def test_multi_number_explanation_cannot_be_silently_scalarized(self):
        proof = direct_literal_projection(lambda prompt: "3 months before and 7 months now.",
                                          "How long did I use it?", self.sources(), lambda s: len(s) // 5)
        self.assertIsNone(proof)

    def test_partial_digit_is_not_an_evidence_match(self):
        source = replace(self.sources()["S1"], text="I have 137 points.")
        proof = direct_literal_projection(lambda prompt: "37", "How many points do I have?",
                                          {"S1": source}, lambda s: len(s) // 5)
        self.assertIsNone(proof)

    def test_oversized_source_view_is_not_prefix_truncated(self):
        callback = lambda prompt: self.fail("Oversized view reached inference")
        self.assertIsNone(direct_literal_projection(callback, "How long did I use it?", self.sources(), lambda s: 2000))

    def test_unsupported_answer_is_not_proof_of_absence(self):
        self.assertIsNone(direct_literal_projection(lambda prompt: "I do not know.",
                                                   "How long did I use it?", self.sources(), lambda s: len(s) // 5))

    def test_literal_projection_does_not_strip_source_qualification(self):
        source = replace(self.sources()["S1"], text="I used it for about 7 months.")
        sources = {"S1": source}
        self.assertIsNone(direct_literal_projection(lambda prompt: "7 months", "How long did I use it?",
                                                   sources, lambda s: len(s) // 5))
        proof = direct_literal_projection(lambda prompt: "about 7 months", "How long did I use it?",
                                          sources, lambda s: len(s) // 5)
        self.assertEqual(proof["answer"], "about 7 months")

    def test_scope_hints_cannot_delete_evidence_before_binding(self):
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", "I play tennis weekly.")
        sources = {"S1": source}
        selected, audit = scoped_sources("How often do I play table tennis?", sources, ["table tennis"])
        self.assertEqual(selected, sources)
        self.assertEqual(audit["omitted_source_count"], 0)
        self.assertFalse(phrase_present("table tennis", source.text))
        self.assertEqual(sources, {"S1": source})

    def test_membership_extraction_retains_all_members_from_one_assertion(self):
        from unittest.mock import Mock
        source = Source("S1", "s", "1", "user", "user", "2025-01-01",
                        "My basket contains a red mug, blue mug and green mug.")
        choose = Mock(side_effect=[{"items": [{"slot": 0, "assertion": 0}], "unresolved": False},
                                   {"actor": "My"}, {"values": ["red mug", "blue mug", "green mug"]},
                                   *[{"change": "snapshot", "status": "actual", "complete_set": True}] * 3])
        plan = contract("count_distinct", [slot(value_kind="member", selector="all")])
        rows = extract_window(choose, plan, source, 0, source.text)
        self.assertEqual(execute_state_contract(plan, rows)["answer"], "3")

    def test_separate_source_fields_cannot_invent_an_actor_or_value(self):
        from unittest.mock import Mock
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", "My balance is $29.")
        choose = Mock(side_effect=[{"items": [{"slot": 0, "assertion": 0}], "unresolved": False},
                                   {"actor": "My"}, {"values": ["$999"]},
                                   {"change": "snapshot", "status": "actual", "complete_set": False}])
        with self.assertRaisesRegex(ContractError, "not_source_grounded"):
            extract_window(choose, contract(), source, 0, source.text)


class IntegrationTests(unittest.TestCase):
    def test_off_does_not_read_sources_or_call_model(self):
        from PRAGMOS_benchmark_LongMemEval import evaluate_state_candidate
        result = evaluate_state_candidate(None, "Anything", SimpleNamespace(), query_intent=None,
                                          source_rows=None, turn_lookup=None)
        self.assertIsNone(result)

    def test_protected_families_do_not_call_new_model_path(self):
        from PRAGMOS_benchmark_LongMemEval import (
            ASSISTANT_MEMORY_INTENT, EVIDENCE_SYNTHESIS_INTENT, PREFERENCE_RECOMMENDATION_INTENT,
            evaluate_state_candidate,
        )
        args = SimpleNamespace(mode="pragmos_context", pragmos_state_mode="candidate")
        for intent in (ASSISTANT_MEMORY_INTENT, EVIDENCE_SYNTHESIS_INTENT, PREFERENCE_RECOMMENDATION_INTENT):
            result = evaluate_state_candidate(None, "Anything", args, query_intent={"intent": intent},
                                              source_rows=None, turn_lookup=None)
            self.assertEqual(result["calls"], {"planner_calls": 0, "extractor_calls": 0})
            self.assertEqual(result["status"], "NOT_APPLICABLE")

    def test_raw_and_overlapping_experiments_are_rejected(self):
        from PRAGMOS_benchmark_LongMemEval import validate_state_configuration
        for kwargs in ({"mode": "raw_phi3_haystack"}, {"pragmos_composable_mode": "candidate"},
                       {"pragmos_temporal_contract_mode": "validated"}, {"pragmos_semantic_checker_path": "local"}):
            args = SimpleNamespace(**{"mode": "pragmos_context", "pragmos_state_mode": "candidate", **kwargs})
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_state_configuration(args)


if __name__ == "__main__":
    unittest.main()
