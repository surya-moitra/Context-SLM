"""Counterexamples for source contracts; no benchmark IDs or reference data."""
from dataclasses import replace
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PRAGMOS_composable_execution import ContractError
from PRAGMOS_typed_claims import numeric_slots
from PRAGMOS_typed_computation import execute_typed_plan
from test_PRAGMOS_typed_computation import bind, request, selection


class EvidenceContractTests(unittest.TestCase):
    def test_relative_actor_scope_does_not_absorb_the_speakers_own_event(self):
        claims, windows, sources = bind(["I welcomed Ada.", "My cousin welcomed Ben."], [
            selection(entity="Ada", key="Ada"),
            selection(actor="My cousin", binding="requested_relation", relation_anchor="My cousin", entity="Ben", key="Ben")])
        query = replace(request("count_distinct"), actor_scope="related", target_actor="user")
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["answer"], "1")

    def test_equality_quote_is_extracted_from_source_not_a_model_selected_prefix(self):
        from PRAGMOS_typed_claims import uniformity_quote, source_windows, actor_choices
        from PRAGMOS_composable_execution import Source
        source = Source("S", "s", "t", "user", "user", None, "There were 7 mugs in that purchase, all at the same price.")
        window = source_windows({"S": source})[0]
        self.assertEqual(uniformity_quote(window, "mugs"), "all at the same price.")
        self.assertNotIn("There", actor_choices(source.text))

    def test_modified_noun_reference_requires_a_unique_parent_and_explicit_demonstrative(self):
        claims, windows, sources = bind(["I paid $56 altogether for the studio mugs.",
                                        "There were 7 mugs in that purchase, all at the same price."], [
            selection(entity="studio mugs", operand="left", numbers=[0], quantity_kind="measure"),
            selection(actor=None, binding="linked_event", entity="mugs", attribute="mugs", operand="right", numbers=[0],
                      quantity_kind="count", link_index=0, uniformity="all at the same price")])
        self.assertEqual(execute_typed_plan(request("divide", per_member=True), claims, windows, sources)["answer"], "$8")

    def test_quoted_known_fact_is_not_an_explicit_missing_value(self):
        with self.assertRaisesRegex(ContractError, "explicit_fact_bound_uncertainty"):
            bind(["I finished the stage."], [selection(entity="stage", quantity_kind="missing", missing_span="finished")])

    def test_other_entity_missing_clause_cannot_veto_this_entity(self):
        with self.assertRaisesRegex(ContractError, "explicit_fact_bound_uncertainty"):
            bind(["I finished the stage; the camera price is unknown."],
                 [selection(entity="stage", quantity_kind="missing", missing_span="price is unknown")])

    def test_other_purchase_uniformity_cannot_certify_the_requested_purchase(self):
        claims, windows, sources = bind(["I paid $56 for books.", "I received 7 mugs; all at the same price."], [
            selection(entity="books", operand="left", numbers=[0], quantity_kind="measure"),
            selection(entity="mugs", attribute="mugs", operand="right", numbers=[0], quantity_kind="count", uniformity="all at the same price")])
        with self.assertRaisesRegex(ContractError, "uniformity"):
            execute_typed_plan(request("divide", per_member=True), claims, windows, sources)

    def test_arbitrary_quote_cannot_certify_equal_individual_prices(self):
        for quote, suffix in (("I paid", ""), ("same color", "; all the same color")):
            claims, windows, sources = bind(["I paid $56 for mugs" + suffix + ".", "I received 7 mugs."], [
                selection(entity="mugs", operand="left", numbers=[0], quantity_kind="measure", uniformity=quote),
                selection(entity="mugs", attribute="mugs", operand="right", numbers=[0], quantity_kind="count")])
            with self.subTest(quote=quote), self.assertRaisesRegex(ContractError, "uniformity"):
                execute_typed_plan(request("divide", per_member=True), claims, windows, sources)

    def test_equal_price_qualification_is_group_bound_not_from_other_goods(self):
        for suffix, accepted in (("all at the same price", True), ("all shoes cost the same price", False),
                                 ("not all at the same price", False)):
            claims, windows, sources = bind(["I paid $56 for mugs.", "I received 7 mugs; " + suffix + "."], [
                selection(entity="mugs", operand="left", numbers=[0], quantity_kind="measure"),
                selection(entity="mugs", attribute="mugs", operand="right", numbers=[0], quantity_kind="count", uniformity=suffix)])
            if accepted:
                self.assertEqual(execute_typed_plan(request("divide", per_member=True), claims, windows, sources)["answer"], "$8")
            else:
                with self.assertRaisesRegex(ContractError, "uniformity"):
                    execute_typed_plan(request("divide", per_member=True), claims, windows, sources)

    def test_reception_cannot_donate_its_venue_to_the_host_event(self):
        for venue, accepted in (("Cedar University", True), ("Pine College", False)):
            claims, windows, sources = bind(["I presented at the Lantern Symposium.",
                "The Lantern Symposium was hosted at Cedar University; the reception was at Pine College."], [
                selection(entity="Lantern Symposium", key="Lantern Symposium", kind="context", operand="context"),
                selection(actor="Lantern Symposium", binding="linked_event", entity="Lantern Symposium", key=venue, link_index=0)])
            if accepted:
                self.assertEqual(execute_typed_plan(request("lookup", projection="entity", key_role="location"), claims, windows, sources)["answer"], venue)
            else:
                with self.assertRaisesRegex(ContractError, "bound_source_relation"):
                    execute_typed_plan(request("lookup", projection="entity", key_role="location"), claims, windows, sources)

    def test_unmatched_group_is_not_dropped_before_choosing_a_winner(self):
        claims, windows, sources = bind(["My Cedar account had 180 followers.", "My Cedar account had 260 followers.",
                                        "My Birch account had 420 followers."], [
            selection(actor="My", entity=group, key=group, attribute="followers", operand=side, numbers=[0], quantity_kind="count")
            for group, side in (("Cedar", "right"), ("Cedar", "left"), ("Birch", "right"))])
        result = execute_typed_plan(request("subtract", grouped=True, rank="max", projection="entity"), claims, windows, sources)
        self.assertFalse(result["executable"])
        self.assertEqual(result["coverage"]["missing_group_operands"], [{"key": "birch", "operands": ["left"]}])


class BoundContractTests(unittest.TestCase):
    def bound_claims(self, second="at least $9", selection_index=0):
        return bind(["My resale estimate for the camera is at least $35, possibly $47.",
                     "My resale estimate for the lamp is " + second + "."], [
            selection(actor="My", entity="camera", numbers=[selection_index], quantity_kind="measure", bound="lower"),
            selection(actor="My", entity="lamp", numbers=[0], quantity_kind="measure", bound="lower" if second.startswith("at least") else None)])

    def test_minimum_total_uses_lower_bounds_not_possible_values(self):
        claims, windows, sources = self.bound_claims()
        result = execute_typed_plan(replace(request(), answer_bound="lower"), claims, windows, sources)
        self.assertEqual(result["answer"], "$44")
        json.dumps(result)
        with self.assertRaisesRegex(ContractError, "exact_total"):
            execute_typed_plan(request(), claims, windows, sources)

    def test_possible_estimate_is_neither_an_exact_value_nor_a_bound(self):
        with self.assertRaises(ContractError):
            self.bound_claims(selection_index=1)

    def test_exact_and_bounded_contributions_can_be_added(self):
        claims, windows, sources = self.bound_claims("$9")
        self.assertEqual(execute_typed_plan(replace(request(), answer_bound="lower"), claims, windows, sources)["answer"], "$44")

    def test_lower_bound_does_not_invent_an_upper_bound(self):
        claims, windows, sources = self.bound_claims()
        with self.assertRaisesRegex(ContractError, "unbounded"):
            execute_typed_plan(replace(request(), answer_bound="upper"), claims, windows, sources)


class TemporalContractTests(unittest.TestCase):
    def test_completed_years_policy_cannot_silently_answer_an_elapsed_month_request(self):
        from PRAGMOS_typed_computation import validate_result_contract
        query = replace(request("calendar_difference"), time_policy="completed_years", output_unit="months")
        with self.assertRaisesRegex(ContractError, "conflicts_with_temporal"):
            validate_result_contract(query)

    def test_relative_event_date_cannot_use_another_speakers_anchor(self):
        claims, windows, sources = bind(["I submitted the report two days before the workshop.", "My workshop was on 18 March 2020."], [
            selection(entity="report", event="report"),
            selection(actor="My", entity="workshop", event="workshop", time="18 March 2020", kind="context", operand="context")], speakers=["Asha", "Ciara"])
        query = replace(request("date_lookup"), target_actor="Asha")
        with self.assertRaisesRegex(ContractError, "not_uniquely_resolved"):
            execute_typed_plan(query, claims, windows, sources)

    def test_before_event_filter_works_without_inventing_a_contract_date(self):
        claims, windows, sources = bind(["Before I signed the contract, I got the camera.",
                                        "After I signed the contract, I got the lamp."], [
            selection(entity=item, key=item, time_anchor="I signed the contract", time_direction=direction)
            for item, direction in (("camera", "before"), ("lamp", "after"))])
        query = replace(request("count_distinct"), temporal_filter=("before", "I signed the contract"))
        result = execute_typed_plan(query, claims, windows, sources)
        self.assertEqual(result["answer"], "1")
        self.assertEqual(len(result["coverage"]["event_filter_excluded_claim_ids"]), 1)

    def test_unknown_event_order_is_not_silently_excluded_from_a_count(self):
        claims, windows, sources = bind(["I got a camera."], [selection(entity="camera", key="camera")])
        query = replace(request("count_distinct"), temporal_filter=("before", "the contract"))
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["coverage"]["status"], "UNRESOLVED_EVENT_FILTER")

    def test_relative_date_resolution_is_independent_of_retrieval_order(self):
        for reverse in (False, True):
            texts = ["I submitted the report two days before the workshop.", "My workshop was on 18 March 2020."]
            specs = [selection(entity="report", event="report"),
                     selection(actor="My", entity="workshop", event="workshop", time="18 March 2020", kind="context", operand="context")]
            if reverse:
                texts.reverse()
                specs.reverse()
            claims, windows, sources = bind(texts, specs)
            result = execute_typed_plan(request("date_lookup"), claims, windows, sources)
            self.assertEqual(result["answer"], "2020-03-16")
            self.assertEqual(len(result["provenance"]), 2)

    def test_composed_offsets_carry_all_ancestors(self):
        claims, windows, sources = bind(["I submitted the report two days before the workshop.",
                                        "My workshop was three days after the review.", "My review was on 10 March 2020."], [
            selection(entity="report", event="report"),
            selection(actor="My", entity="workshop", event="workshop", kind="context", operand="context"),
            selection(actor="My", entity="review", event="review", time="10 March 2020", kind="context", operand="context")])
        result = execute_typed_plan(request("date_lookup"), claims, windows, sources)
        self.assertEqual(result["answer"], "2020-03-11")
        self.assertEqual(len(result["provenance"]), 3)

    def test_cyclic_offsets_are_not_forced_to_a_date(self):
        claims, windows, sources = bind(["My review was two days before the workshop.", "My workshop was three days after the review."], [
            selection(actor="My", entity="review", event="review"), selection(actor="My", entity="workshop", event="workshop")])
        with self.assertRaisesRegex(ContractError, "cyclic"):
            execute_typed_plan(request("date_lookup"), claims, windows, sources)

    def test_age_uses_completed_years_not_matching_calendar_anniversaries(self):
        claims, windows, sources = bind(["Mira was born on 12 July 1990.", "The celebration was on 18 May 2020."], [
            selection(actor="Mira", binding="named", entity="Mira", event="born", operand="left", time="12 July 1990"),
            selection(actor=None, binding="event_property", entity="celebration", event="celebration", operand="right", time="18 May 2020")])
        query = replace(request("calendar_difference"), actor_scope="named", subject_actor="Mira", time_policy="completed_years")
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["answer"], "29 years")

    def test_clock_is_not_two_counts_or_an_unstated_midnight_rollover(self):
        self.assertFalse(numeric_slots("I start at 07:45."))
        claims, windows, sources = bind(["I normally start at 07:45.", "I start at 06:55 on Fridays."], [
            selection(entity="start", operand="left", clock="07:45"), selection(entity="start", operand="right", clock="06:55")])
        query = replace(request("clock_difference"), time_policy="wall_clock")
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["answer"], "50 minutes")


class StructuredProjectionTests(unittest.TestCase):
    def test_collection_keeps_names_attached_to_their_measurements(self):
        claims, windows, sources = bind(["My Green Book has 281 pages.", "My Blue Book has 317 pages."], [
            selection(actor="My", entity=name, key=name, numbers=[0], quantity_kind="measure") for name in ("Green Book", "Blue Book")])
        query = replace(request("collect", projection="entity"), result_shape="tuple_set")
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["answer"], "Green Book: 281 pages; Blue Book: 317 pages")

    def test_conflicting_values_for_one_record_are_not_collapsed(self):
        claims, windows, sources = bind(["My Green Book has 281 pages.", "My Green Book has 317 pages."], [
            selection(actor="My", entity="Green Book", key="Green Book", numbers=[0], quantity_kind="measure")] * 2)
        with self.assertRaisesRegex(ContractError, "conflicting_attribute"):
            execute_typed_plan(replace(request("collect"), result_shape="tuple_set"), claims, windows, sources)

    def test_named_participant_group_does_not_turn_into_a_speaker_only_average(self):
        claims, windows, sources = bind(["At the studio gathering Mira was 3 years old.", "At the studio gathering Omar was 5 years old."], [
            selection(actor=name, binding="named", entity=name, event="studio gathering", numbers=[0], quantity_kind="measure") for name in ("Mira", "Omar")])
        query = replace(request("mean"), actor_scope="event_participants", event_scope="the studio gathering")
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["answer"], "4 years")


class ProductionContractTests(unittest.TestCase):
    def test_operator_label_requires_a_literal_not_an_invented_query_modifier(self):
        from PRAGMOS_typed_claims import literal_labels
        self.assertEqual(literal_labels("What did I buy?", ("before", "after")), ())
        self.assertEqual(literal_labels("Which purchases were before the contract?", ("before", "after")), ("before",))

    def test_record_planner_cannot_leave_tuple_output_on_scalar_lookup(self):
        from PRAGMOS_typed_computation import plan_small_request
        responses = {"operation": "lookup", "answer_kind": "a list of entities", "paired_values": True,
                     "key_role": "entity", "requested_key": "books", "statuses": ["completed"],
                     "speaker_inputs": True, "event_filtered": False}
        def choose(_prompt, _stage, schema, _reserve):
            return {key: responses[key] for key in schema["properties"]}
        planned = plan_small_request(choose, "Which books and page counts did I mention?", {"extended_contracts": True})
        self.assertEqual((planned.operation, planned.result_shape), ("collect", "tuple_set"))

    def test_single_winner_never_enters_new_paired_list_discriminator(self):
        from PRAGMOS_typed_computation import plan_small_request
        responses = {"operation": "sum", "answer_kind": "one named entity", "rank_needed": True, "rank": "max",
                     "key_role": "location", "requested_key": "shop", "statuses": ["completed"],
                     "speaker_inputs": True, "event_filtered": False}
        def choose(_prompt, _stage, schema, _reserve):
            self.assertNotIn("paired_values", schema["properties"])
            return {key: responses[key] for key in schema["properties"]}
        planned = plan_small_request(choose, "At which shop did I spend the most?", {"extended_contracts": True})
        self.assertEqual((planned.operation, planned.result_shape, planned.rank), ("sum", "scalar", "max"))

    def test_compatible_requests_use_established_engine_but_derived_contracts_do_not(self):
        from PRAGMOS_typed_computation import established_slots_capability
        base = request("count_distinct")
        self.assertTrue(established_slots_capability(base))
        for candidate in (replace(base, temporal_filter=("before", "the contract")),
                          replace(base, actor_scope="related"), replace(base, result_shape="tuple_set"),
                          request("mean"), request("subtract")):
            self.assertFalse(established_slots_capability(candidate))

    def test_delegation_restores_state_before_existing_engine_and_accounts_for_both(self):
        from PRAGMOS_composable_candidate import run_typed_candidate, promotable
        from test_PRAGMOS_composable_candidate import context
        ctx = context([])
        def planning(_ctx, _q, _s, report, **_kwargs):
            ctx.llm.position = 12
            report.update(delegate_to_slots_v3=True, request={"operation": "count_distinct"})
            report["calls"]["planner_calls"] = 4
            report["calls"]["responses"] = [{"stage": "planner", "text": "{}"}]
            report["calls"]["grammar_checks"] = [{"stage": "planner", "accepted": True}]
            return report
        def delegate(*_args, **_kwargs):
            self.assertEqual(ctx.llm.position, 0)
            return {"version": 3, "protocol": "source_linked_slots", "status": "COMPLETE", "answer": "3",
                    "extraction_complete": True, "isolation": "llama_state_restored_no_external_cache",
                    "proof": {"executable": True, "provenance": ["source"]}, "calls": {"planner_calls": 1, "extractor_calls": 2}}
        with patch("PRAGMOS_typed_computation.run_typed_computation", side_effect=planning), patch(
                "PRAGMOS_composable_candidate.run_grounded_candidate", side_effect=delegate):
            result = run_typed_candidate(ctx, "How many?", {}, n_ctx=2048)
        self.assertTrue(promotable(result))
        self.assertEqual(result["selected_engine"], "slots_v3")
        self.assertEqual(result["calls"]["planner_calls"], 5)
        self.assertEqual(result["calls"]["extractor_calls"], 2)
        self.assertEqual(len(result["calls"]["responses"]), 1)
        self.assertEqual(len(result["calls"]["grammar_checks"]), 1)

    def test_long_turns_still_supply_whole_statement_planner_evidence(self):
        from PRAGMOS_typed_computation import planner_evidence
        from PRAGMOS_composable_execution import Source
        text = "Unrelated note. " * 500 + "I paid $17 at Cedar."
        preview = planner_evidence(SimpleNamespace(count_tokens=lambda value: len(value.split())),
                                   "Where did I pay $17?", {"S": Source("S", "s", "t", "user", "user", None, text)})
        self.assertEqual(len(preview), 1)
        self.assertIn("I paid $17 at Cedar.", preview[0]["text"])

    def test_typed_temporal_boundary_is_opt_in_and_explicit_protection_wins(self):
        from PRAGMOS_benchmark_LongMemEval import evaluate_composable_candidate
        from test_PRAGMOS_composable_integration import semantic_plan, arguments
        plan = semantic_plan("When did I complete the studio project?", "DATE_LOOKUP")
        args = arguments()
        args.pragmos_composable_mode, args.pragmos_composable_protocol = "candidate", "typed_v4"
        rows = [{"source_turn_id": "1"}]
        turns = {"1": {"session_id": "s", "role": "user", "speaker": "user", "text": "I completed the project yesterday."}}
        with patch("PRAGMOS_composable_candidate.run_typed_candidate", return_value={"calls": {}}) as run:
            evaluate_composable_candidate(None, "When?", args, semantic_plan=plan, query_intent={}, source_rows=rows, turn_lookup=turns)
            self.assertTrue(run.call_args.kwargs["answer_contract"]["extended_contracts"])
            run.reset_mock()
            evaluate_composable_candidate(None, "When?", args, semantic_plan=plan, query_intent={}, source_rows=rows, turn_lookup=turns, protected=True)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
