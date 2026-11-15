"""Counterexamples for source-bound state execution; these are not model scores."""

from dataclasses import replace
import copy
import unittest
from unittest.mock import patch

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_state_binding import (
    StateFact, assertion_spans, bind_facts, bind_query, execute_query, ordered_history,
    quantity_choices, select_history, choose_executable_query, literal_names, validate_history_overlap,
    bind_output_obligations, predicate_history,
)
from PRAGMOS_state_views import accepted_state_candidate, scoped_sources, run_state_candidate
from test_PRAGMOS_composable_candidate import context
from test_PRAGMOS_state_contracts import contract_payloads, planner_responses, source_fact, relation_responses


def fact(value, day=1, **changes):
    fields = dict(fact_id="F" + str(day), source_id="S" + str(day), session_id="session" + str(day),
                  turn_id=str(day), speaker="user", role="user", timestamp=f"2025-01-{day:02d}",
                  start=0, end=40, quote="My balance is " + value + ".", actor="My", actor_binding="speaker",
                  entity=None, attribute="balance", value=value, kind="snapshot", modality="actual",
                  previous=None, effective_date=None)
    fields.update(changes)
    return StateFact(**fields)


def query(operation="read", left=None, **changes):
    fields = dict(operation=operation, left=left or ["F1"], right=[], selectors=["latest"], marker=None,
                  exclusions=[], modality="actual", whole_units=False, request_quote="What is my balance?")
    fields.update(changes)
    return fields


def bound_responses(sources, unresolved_last=False):
    from test_PRAGMOS_state_reader import reader_responses
    from test_PRAGMOS_state_program import read_program
    from PRAGMOS_state_contracts import bind_answer_contract
    responses = planner_responses(contract_payloads("What is my balance?"))
    for i, source in enumerate(sources):
        value = "$98" if unresolved_last and i == len(sources) - 1 else "$29"
        responses.extend(reader_responses(source, [source_fact(value)]))
    c = bind_answer_contract("What is my balance?", *contract_payloads("What is my balance?"))
    responses.extend([{"projection": "measurement"}, {"family": "read"}])
    responses.append(read_program(c))
    return responses


class StateSourceReviewTests(unittest.TestCase):
    def test_changed_state_need_not_repeat_old_role_or_location(self):
        sources = {"a": Source("a", "s1", "1", "user", "user", "2025-01-01", "As Build Lead at Fern, I led 4 analysts."),
                   "b": Source("b", "s2", "2", "user", "user", "2025-02-01", "My Fern team now has 7 analysts.")}
        before = copy.deepcopy(sources)
        selected, audit = scoped_sources("How many analysts as Build Lead at Fern and now?", sources,
                                         ["analysts", "Build Lead", "Fern"])
        self.assertEqual(set(selected), set(sources))
        self.assertEqual(audit["omitted_source_count"], 0)
        self.assertEqual(sources, before)

    def test_abbreviated_title_keeps_person_identity_in_one_assertion(self):
        text = "My sessions with Dr. Chen were weekly. I changed the schedule."
        self.assertEqual(assertion_spans(text), ["My sessions with Dr. Chen were weekly.", "I changed the schedule."])

    def test_short_units_do_not_hide_sentence_boundaries(self):
        self.assertEqual(len(assertion_spans("My weight is 62 kg. My friend's weight is 75 kg.")), 2)

    def test_literal_quantities_do_not_include_property_labels(self):
        self.assertEqual(quantity_choices("My latest weight is 62 kg."), ["62 kg"])
        self.assertEqual(quantity_choices("The original starting measurement is unchanged."), [])

    def test_compound_measurement_stays_one_value(self):
        self.assertEqual(quantity_choices("It took 18 minutes 12 seconds."), ["18 minutes 12 seconds"])

    def test_qualifiers_and_ranges_remain_attached_to_literal_values(self):
        self.assertEqual(quantity_choices("My total is about 11-14 hours."), ["about 11-14 hours"])
        self.assertEqual(quantity_choices("I have at least 7 items."), ["at least 7 items"])

    def test_names_are_literals_not_shared_head_words(self):
        self.assertIn("Fern Research", literal_names("What is my Fern Research status before Ruby?"))
        self.assertNotIn("I", literal_names("How many staff did I lead at Fern Research?"))


class StateFactBindingTests(unittest.TestCase):
    def test_implicit_source_property_needs_an_explicit_query_column(self):
        with self.assertRaisesRegex(ContractError, "bound_column"):
            self.bind(self.payload(attribute=None))

    def test_query_column_does_not_fabricate_a_literal_source_predicate(self):
        data = self.payload(attribute=None)
        source = Source("S1", "s1", "1", "user", "user", "2025-01-01", "My balance is $29.")
        result = bind_facts(data, source, 0, source.text, 1, column_id="P1")[0]
        self.assertIsNone(result.attribute)
        self.assertEqual(result.column_id, "P1")

    def payload(self, **changes):
        row = dict(quote="My balance is $29.", actor="My", binding="speaker", entity=None,
                   attribute="balance", value="$29", kind="snapshot", modality="actual", previous=None,
                   effective_date=None)
        row.update(changes)
        return {"facts": [row], "unresolved": False}

    def bind(self, payload, text="My balance is $29."):
        source = Source("S1", "s1", "1", "user", "user", "2025-01-01", text)
        return bind_facts(payload, source, 0, text, 1)

    def test_source_actor_property_value_addresses_are_retained(self):
        row = self.bind(self.payload())[0]
        self.assertEqual((row.value, row.owner, row.source_id), ("$29", "speaker:user", "S1"))

    def test_fabricated_value_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ContractError, "literal_not_in_source"):
            self.bind(self.payload(value="$98"))

    def test_generated_actor_name_is_not_a_source_reference(self):
        with self.assertRaisesRegex(ContractError, "literal_not_in_source"):
            self.bind(self.payload(actor="user"))

    def test_borrowed_actor_from_another_assertion_cannot_own_a_positive_value(self):
        text = "My balance is $29. Mei's balance is $72."
        with self.assertRaisesRegex(ContractError, "literal_not_in_source"):
            self.bind(self.payload(quote="Mei's balance is $72.", actor="My", value="$72"), text)

    def test_named_actor_cannot_be_marked_as_speaker(self):
        text = "Mei has a balance of $29."
        with self.assertRaisesRegex(ContractError, "source_pronoun"):
            self.bind(self.payload(quote=text, actor="Mei", value="$29"), text)

    def test_source_uncertainty_prevents_completion(self):
        data = self.payload(); data["unresolved"] = True
        with self.assertRaisesRegex(ContractError, "source_unresolved"):
            self.bind(data)

    def test_same_value_does_not_create_its_own_transition(self):
        with self.assertRaisesRegex(ContractError, "self_transition"):
            self.bind(self.payload(previous="$29"))


class StateTimelineTests(unittest.TestCase):
    def test_overlapping_history_stitches_without_reversing_an_old_value(self):
        rows = [fact("4"), fact("9", 2), fact("9", 3), fact("7", 4, source_id="S3", timestamp="2025-01-03")]
        validate_history_overlap(rows)

    def test_return_to_an_old_value_requires_an_occurrence_binding(self):
        rows = [fact("4"), fact("9", 2), fact("7", 3), fact("9", 4, source_id="S3", timestamp="2025-01-03")]
        with self.assertRaisesRegex(ContractError, "occurrence_binding"):
            validate_history_overlap(rows)

    def test_initial_and_current_include_intermediate_state_without_scalarizing(self):
        rows = [fact("4"), fact("9", 2), fact("7", 3)]
        result = execute_query(query("pair", ["F1", "F2", "F3"], selectors=["earliest", "latest"]), rows)
        self.assertEqual(result["answer"], "earliest: 4; latest: 7")

    def test_explicit_order_beats_printed_order_of_corrected_value(self):
        old = fact("$150", timestamp="2025-01-01", quote="My letter says $225, correcting $150.", start=30)
        new = fact("$225", 2, timestamp="2025-01-01", previous="$150", quote=old.quote, start=15)
        self.assertEqual(select_history([new, old], "latest")[0].value, "$225")

    def test_multiple_states_in_one_turn_need_an_order_witness(self):
        old = fact("Amber")
        new = replace(old, fact_id="F2", value="Jade", start=20)
        with self.assertRaisesRegex(ContractError, "explicit_order"):
            ordered_history([new, old])

    def test_linked_same_turn_states_support_marker_predecessor(self):
        rows = [fact("Amber"), fact("Jade", 2), fact("Pearl", 3, previous="Jade"),
                fact("Ruby", 4, timestamp="2025-01-03", previous="Pearl")]
        self.assertEqual(select_history(list(reversed(rows)), "before_value", "Ruby")[0].value, "Pearl")

    def test_later_old_photo_does_not_reset_current_state(self):
        rows = [fact("Amber"), fact("Jade", 2), fact("Ruby", 3), fact("Amber", 4, kind="reference")]
        self.assertEqual(select_history(rows, "latest")[0].value, "Ruby")

    def test_planned_goal_is_not_an_actual_measurement(self):
        rows = [fact("78 kg"), fact("72 kg", 2, modality="planned"), fact("75 kg", 3)]
        self.assertEqual(select_history(rows, "latest")[0].value, "75 kg")
        self.assertEqual(select_history(rows, "latest", modality="planned")[0].value, "72 kg")

    def test_cycle_is_rejected_not_sorted_by_text(self):
        rows = [fact("A", previous="B"), fact("B", 2, timestamp="2025-01-01", previous="A")]
        with self.assertRaisesRegex(ContractError, "cycle"):
            ordered_history(rows)

    def test_ambiguous_marker_is_not_arbitrarily_selected(self):
        rows = [fact("A"), fact("B", 2), fact("A", 3)]
        with self.assertRaisesRegex(ContractError, "marker_ambiguous"):
            select_history(rows, "before_value", "A")


class StateQueryTests(unittest.TestCase):
    def test_reused_predicate_planner_receives_the_actual_question(self):
        prompts = []
        def stop_after_prompt(prompt, schema, reserve, stage):
            prompts.append(prompt)
            raise ContractError("diagnostic_stop")
        question = "Did I complete the assignment?"
        with self.assertRaisesRegex(ContractError, "diagnostic_stop"):
            predicate_history(stop_after_prompt, question, [], {})
        self.assertIn(question, prompts[0])

    def test_two_coordinated_requests_have_two_disjoint_source_spans(self):
        question = "What was my opening balance, and what is my balance now?"
        shape, proof = bind_output_obligations(question, "one")
        self.assertEqual(shape, "pair")
        self.assertEqual(len(proof["spans"]), 2)
        self.assertLessEqual(proof["spans"][0]["end"], proof["spans"][1]["start"])

    def test_starting_point_does_not_add_an_extra_requested_output(self):
        shape, proof = bind_output_obligations("How much flour have I used since opening the bag?", "pair")
        self.assertEqual(shape, "one")
        self.assertEqual(proof["model_proposal"], "pair")

    def test_ambiguous_coordinated_attributes_keep_baseline_instead_of_losing_a_slot(self):
        with self.assertRaisesRegex(ContractError, "bound_request_spans"):
            bind_output_obligations("What are my initial and current settings?", "pair")

    def test_boolean_is_not_scalarized(self):
        self.assertEqual(bind_output_obligations("Did I complete the assignment?", "boolean")[0], "boolean")

    def test_explicit_selector_constraint_cannot_be_replaced_by_a_model_pair(self):
        from test_PRAGMOS_state_reader import reader_responses
        from test_PRAGMOS_state_program import select, program
        a = Source("S1", "s1", "1", "user", "user", "2025-01-01", "I began in Amber and later reached Jade.")
        b = Source("S2", "s2", "2", "user", "user", "2025-02-01", "I reached Pearl and then Ruby.")
        def mode(value, predicate, **changes):
            return source_fact(value, actor="I", predicate=predicate, property_ref=None,
                               value_role="attribute", value_type="text", dimension="none", **changes)
        ctx = context([
            *planner_responses(contract_payloads("What was my mode immediately before Ruby?", "mode", projection="attribute",
                               input_type="text", dimension="none", shape="pair")),
            *reader_responses(a, [mode("Amber", "began"), mode("Jade", "reached", previous="Amber", transition=a.text)]),
            *reader_responses(b, [mode("Pearl", "reached"), mode("Ruby", "reached", previous="Pearl", transition=b.text)]),
            {"projection": "attribute"}, {"family": "read"},
            program("What was my mode immediately before Ruby?", [select(selector="before_value", marker="Ruby"),
                {"id": "N2", "op": "read", "input": "N1"}]),
        ])
        with patch("PRAGMOS_composable_candidate.schema_grammar", return_value=None):
            result = run_state_candidate(ctx, "What was my mode immediately before Ruby?", {"S1": a, "S2": b},
                                         n_ctx=2048, baseline_answer="Ruby",
                                         protocol="bound_v3",
                                         selector_constraint={"selector": "predecessor_of_value", "target_value": "Ruby"})
        self.assertEqual(result["answer"], "Pearl")
        self.assertEqual(result["answer_contract"]["shape"], "one")
        self.assertEqual(result["answer_contract"]["time_ref"], "Ruby")
        self.assertTrue(accepted_state_candidate(result, "candidate"))
        self.assertTrue(ctx.llm.restored)

    def test_other_person_is_not_silently_recast_as_the_speaker(self):
        from test_PRAGMOS_state_reader import reader_responses
        from test_PRAGMOS_state_program import contract, read_program
        source = Source("S1", "s", "1", "user", "user", "2025-01-01", "My balance is $29. Mei has $72.")
        q = "What is Mei's balance?"
        payloads = contract_payloads(q, owner_scope="other", subject_ref="Mei")
        ctx = context([*planner_responses(payloads),
            *reader_responses(source, [source_fact("$29"), source_fact("$72", assertion="1", actor="Mei", binding="named")]),
            {"projection": "measurement"}, {"family": "read"},
            read_program(contract(q, owner_scope="other", subject_ref="Mei"), owner="named:mei")])
        with patch("PRAGMOS_composable_candidate.schema_grammar", return_value=None):
            result = run_state_candidate(ctx, q, {"S1": source}, n_ctx=2048, baseline_answer="$29", protocol="bound_v3")
        self.assertEqual(result["answer"], "$72")
        self.assertTrue(accepted_state_candidate(result, "candidate"))
        self.assertTrue(all(row["owner_id"] == "named:mei" for row in result["observations"]))
        self.assertTrue(ctx.llm.restored)

    def test_question_span_cannot_be_generated(self):
        with self.assertRaisesRegex(ContractError, "literal_not_in_source"):
            bind_query(query(request_quote="invented"), "What is my balance?", [fact("29")])

    def test_unknown_fact_id_is_not_accepted(self):
        with self.assertRaisesRegex(ContractError, "unknown_or_duplicate"):
            bind_query(query(left=["invented"]), "What is my balance?", [fact("29")])

    def test_two_owners_cannot_be_folded_into_one_property_history(self):
        rows = [fact("29"), fact("72", 2, actor="Mei", actor_binding="named")]
        with self.assertRaisesRegex(ContractError, "crosses_owners"):
            bind_query(query(left=["F1", "F2"]), "What is my balance?", rows)

    def test_a_pair_needs_both_requested_outputs(self):
        with self.assertRaisesRegex(ContractError, "separate_outputs"):
            bind_query(query("pair"), "What is my balance?", [fact("29")])

    def test_difference_requires_explicit_operand_order(self):
        with self.assertRaisesRegex(ContractError, "two_state_selections"):
            bind_query(query("difference"), "What is my balance?", [fact("29")])

    def test_duplicate_fact_id_is_not_double_counted(self):
        with self.assertRaisesRegex(ContractError, "duplicate_fact_id"):
            bind_query(query(left=["F1", "F1"]), "What is my balance?", [fact("29")])


class StateArithmeticTests(unittest.TestCase):
    def test_one_observation_cannot_be_reused_as_two_time_endpoints(self):
        with self.assertRaisesRegex(ContractError, "distinct_time_operands"):
            execute_query(query("difference", selectors=["earliest", "latest"]), [fact("88 kg")])

    def test_initial_minus_current_excludes_intermediate_target_and_other_owner(self):
        rows = [fact("88 kg"), fact("80 kg", 2, modality="planned"), fact("84 kg", 3), fact("81 kg", 4),
                fact("12 kg", 5, actor="Mei", actor_binding="named")]
        plan = query("difference", ["F1", "F2", "F3", "F4"], selectors=["earliest", "latest"])
        self.assertEqual(execute_query(plan, rows)["answer"], "7 kg")

    def test_different_source_wording_does_not_change_bare_quantity_dimension(self):
        rows = [fact("5", attribute="reports"), fact("9", 2, attribute="team members")]
        self.assertEqual(execute_query(query("difference", ["F1", "F2"], selectors=["latest", "earliest"]), rows)["answer"], "4")

    def test_incompatible_units_cannot_be_subtracted(self):
        rows = [fact("88 kg"), fact("81 hours", 2)]
        with self.assertRaisesRegex(ContractError, "dimensions_incompatible"):
            execute_query(query("difference", ["F1", "F2"], selectors=["earliest", "latest"]), rows)

    def test_source_qualification_blocks_exact_arithmetic(self):
        rows = [fact("$29", quote="My balance is about $29."), fact("$42", 2)]
        with self.assertRaisesRegex(ContractError, "interval_contract"):
            execute_query(query("difference", ["F1", "F2"], selectors=["latest", "earliest"]), rows)

    def test_change_direction_is_comparison_not_a_snapshot(self):
        rows = [fact("2"), fact("5", 2)]
        self.assertEqual(execute_query(query("direction", ["F1", "F2"], selectors=["earliest", "latest"]), rows)["answer"], "increased")

    def test_read_preserves_range_and_structured_record(self):
        for value in ("11-14 hours", "6-2", "the blue box on the top shelf"):
            with self.subTest(value=value):
                self.assertEqual(execute_query(query(), [fact(value)])["answer"], value)

    def test_predecessor_requires_a_supported_marker(self):
        with self.assertRaisesRegex(ContractError, "marker_ambiguous"):
            execute_query(query(selectors=["before_value"], marker="unknown"), [fact("A")])

    def test_surface_name_difference_cannot_certify_inequality(self):
        rows = [fact("MapNote app"), fact("MapNote", 2, actor="Mei", actor_binding="named")]
        with self.assertRaisesRegex(ContractError, "equivalence_link"):
            execute_query(query("equal", ["F1"], right=["F2"], selectors=["latest", "latest"]), rows)


class StatePromotionTests(unittest.TestCase):
    def test_scalar_output_menu_does_not_contain_an_initial_current_pair(self):
        prompts = []
        def choose(prompt, schema, reserve, stage):
            prompts.append(prompt)
            return {"option": "read"}
        result = choose_executable_query(choose, "What is my current balance?", [fact("20"), fact("37", 2)], "one")
        self.assertEqual(result["operation"], "read")
        self.assertNotIn("Both initial AND current", prompts[0])

    def test_pair_output_menu_cannot_silently_return_one_state(self):
        def choose(prompt, schema, reserve, stage):
            self.fail("A uniquely bound pair must not need another interpretation call")
        result = choose_executable_query(choose, "What are my original and current balances?", [fact("20"), fact("37", 2)], "pair")
        self.assertEqual(result["operation"], "pair")

    def test_legacy_literal_projection_cannot_promote(self):
        result = {"version": 1, "status": "COMPLETE", "answer": "29", "source_view_complete": True,
                  "proof": {"source_bound": True, "output_slots_complete": True},
                  "isolation": "llama_state_restored_no_external_cache"}
        self.assertFalse(accepted_state_candidate(result, "candidate"))

    def test_each_missing_binding_obligation_prevents_promotion(self):
        from test_PRAGMOS_state_program import contract, read_program, bound
        from PRAGMOS_state_program import execute_program, review_promotion
        c = contract()
        proof, expected = execute_program(read_program(c), c, [bound("$29")], {"speaker:user"})
        result = {"version": 3, "status": "COMPLETE", "answer": "$29", "source_view_complete": True,
                  "promotion_review": review_promotion("$12", "$29", c, expected),
                  "proof": proof, "isolation": "llama_state_restored_no_external_cache"}
        self.assertTrue(accepted_state_candidate(result, "candidate"))
        for obligation in proof["obligations"]:
            bad = copy.deepcopy(result); bad["proof"]["obligations"][obligation] = False
            self.assertFalse(accepted_state_candidate(bad, "candidate"), obligation)
        self.assertFalse(accepted_state_candidate(result, "shadow"))


if __name__ == "__main__":
    unittest.main()
