import copy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from PRAGMOS_composable_execution import ContractError, Quantity, Source
from PRAGMOS_typed_claims import (
    FIELDS, bind_typed_claim, calendar_value, numeric_slots, source_windows, validate_claims,
)
from PRAGMOS_typed_computation import (
    calendar_difference, compile_typed_request, execute_typed_plan, validate_typed_request,
)
from PRAGMOS_composable_candidate import promotable, run_typed_candidate, supported_typed_abstention
from test_PRAGMOS_composable_candidate import context


def selection(**fields):
    result = {name: None for name in FIELDS}
    result.update(actor="I", binding="self", numbers=[], quantity_kind="none", operand="member",
                  group_side="none", kind="assertion", status="completed", change="none")
    result.update(fields)
    return result


def request(operation="sum", *, question="Compute the result.", **options):
    primary = {"operation": operation, "projection": "value", "rank": "none", "comparison": "none"}
    detail = {"grouped": False, "key_role": "entity", "time_scope": None, "output_unit": None,
              "precision": None, "membership": False, "per_member": False,
              "statuses": ["completed", "ongoing", "pending"]}
    for key, value in options.items():
        (primary if key in primary else detail)[key] = value
    return validate_typed_request(question, primary, detail)


def bind(texts, selections, speakers=None):
    speakers = speakers or ["user"] * len(texts)
    sources = {f"S{i}": Source(f"S{i}", f"session{i}", str(i), "user", speakers[i - 1], f"2028-01-{i:02d}", text)
               for i, text in enumerate(texts, 1)}
    windows = source_windows(sources)
    if len(windows) != len(selections):
        raise AssertionError("Fixture must specify a selection for each focus statement")
    claims, known = [], {}
    for window, spec in zip(windows, selections):
        row = dict(spec)
        index = row.pop("link_index", None)
        if index is not None:
            row["link"] = claims[index].claim_id
        claim = bind_typed_claim(window, sources[window.source_id], row, known)
        known[claim.claim_id] = claim
        claims.append(claim)
    return claims, {w.window_id: w for w in windows}, sources


class TypedClaimTests(unittest.TestCase):
    def test_numeric_slots_include_bare_properties_and_unknown_count_nouns(self):
        for text, value in (("My rating was 3.2.", "3.2"), ("I sold 7 packets.", "7"), ("I gained 180 followers.", "180")):
            self.assertEqual(numeric_slots(text)[0].value, value)

    def test_calendar_numbers_cannot_be_selected_as_measurements(self):
        slots = numeric_slots("On 1 March 2020 I had 180 followers.")
        self.assertEqual([slot.value for slot in slots], ["180"])

    def test_spelled_counts_are_not_lost(self):
        self.assertEqual(numeric_slots("I sold seven packets.")[0].value, "7")

    def test_parent_context_preserves_a_following_calendar_qualification(self):
        source = Source("S", "session", "turn", "user", "user", None,
                        "My visit lasted 3 days. That visit belongs to March 2020.")
        window = source_windows({"S": source})[0]
        claim = bind_typed_claim(window, source, selection(actor="My", entity="visit", attribute="days",
                                  numbers=[0], quantity_kind="measure", time="March 2020"), {})
        self.assertEqual(claim.occurred_at.granularity, "month")

    def test_known_units_cannot_be_erased(self):
        with self.assertRaisesRegex(ContractError, "stated_unit"):
            bind(["I worked 3 hours."], [selection(entity="worked", attribute="hours", numbers=[0], quantity_kind="scalar")])

    def test_unknown_count_nouns_have_distinct_dimensions(self):
        claims, _, _ = bind(["I sold 7 packets.", "I sold 7 mugs."],
                            [selection(entity="packets", attribute="packets", numbers=[0], quantity_kind="count"),
                             selection(entity="mugs", attribute="mugs", numbers=[0], quantity_kind="count")])
        self.assertNotEqual(claims[0].quantity.dimensions, claims[1].quantity.dimensions)

    def test_self_binding_does_not_absorb_a_relative(self):
        with self.assertRaisesRegex(ContractError, "not_source_speaker"):
            bind(["My cousin welcomed a child."], [selection(actor="My cousin", entity="child")])

    def test_related_actor_requires_an_actual_source_relation(self):
        with self.assertRaisesRegex(ContractError, "possessive_source_link"):
            bind(["Mira welcomed a child."], [selection(actor="Mira", binding="requested_relation", entity="child")])
        claims, _, _ = bind(["My cousin welcomed a child."], [selection(actor="My cousin", binding="requested_relation",
                                 relation_anchor="My cousin", entity="child")])
        self.assertEqual(claims[0].actor, "My cousin")

    def test_reference_cannot_replace_an_explicit_other_actor(self):
        with self.assertRaisesRegex(ContractError, "explicit_actor"):
            bind(["I placed an order.", "Mira received an order."],
                 [selection(entity="order", key="order"), selection(actor="Mira", binding="linked_event", entity="order", link_index=0)])

    def test_source_literal_cannot_be_invented(self):
        with self.assertRaisesRegex(ContractError, "not_in_source"):
            bind(["I paid $17 at Cedar."], [selection(entity="invented entity", numbers=[0], quantity_kind="measure")])

    def test_forged_quantity_is_detected_on_execution(self):
        claims, windows, sources = bind(["I paid $17."], [selection(entity="paid", numbers=[0], quantity_kind="measure")])
        forged = replace(claims[0], quantity=Quantity.from_raw(170, "dollars"))
        with self.assertRaisesRegex(ContractError, "not_equal_to_source"):
            execute_typed_plan(request(), [forged], windows, sources)

    def test_missing_value_needs_a_literal_source_qualification(self):
        with self.assertRaisesRegex(ContractError, "source_qualification"):
            bind(["I finished the stage."], [selection(entity="stage", quantity_kind="missing")])

    def test_ranges_and_bounds_cannot_be_certified_as_exact(self):
        for text in ("I worked at least 3 hours.", "I worked 3 to 5 hours."):
            with self.assertRaises(ContractError):
                bind([text], [selection(entity="worked", numbers=[0], quantity_kind="measure")])


class TypedExecutionTests(unittest.TestCase):
    def test_grouped_sum_and_individual_max_remain_distinct(self):
        claims, windows, sources = bind(["I paid $17 at Cedar.", "I paid $38 at Birch.", "I paid $29 at Cedar."],
            [selection(entity=shop, key=shop, numbers=[0], quantity_kind="measure") for shop in ("Cedar", "Birch", "Cedar")])
        result = execute_typed_plan(request(grouped=True, rank="max", projection="entity", key_role="location"), claims, windows, sources)
        self.assertEqual(result["answer"], "Cedar")
        self.assertEqual(len(result["provenance"]), 2)
        self.assertEqual(execute_typed_plan(request("collect", projection="entity"), claims, windows, sources)["answer"], "Cedar; Birch")

    def test_independent_equal_transactions_are_not_deduplicated(self):
        claims, windows, sources = bind(["I paid $17 at Cedar.", "I paid $17 at Cedar."],
            [selection(entity="Cedar", key="Cedar", numbers=[0], quantity_kind="measure")] * 2)
        self.assertEqual(execute_typed_plan(request(), claims, windows, sources)["answer"], "$34")

    def test_grouped_ties_are_not_arbitrarily_resolved(self):
        claims, windows, sources = bind(["I paid $17 at Cedar.", "I paid $17 at Birch."],
            [selection(entity=shop, key=shop, numbers=[0], quantity_kind="measure") for shop in ("Cedar", "Birch")])
        with self.assertRaisesRegex(ContractError, "not_unique"):
            execute_typed_plan(request(grouped=True, rank="max", projection="entity"), claims, windows, sources)

    def test_remaining_quantity_has_ordered_independent_roles(self):
        claims, windows, sources = bind(["My workbook has 81 pages.", "I read 13 pages."],
            [selection(actor="My", entity="workbook", operand="left", numbers=[0], quantity_kind="measure"),
             selection(entity="read", operand="right", numbers=[0], quantity_kind="measure")])
        self.assertEqual(execute_typed_plan(request("subtract"), claims, windows, sources)["answer"], "68 pages")
        reversed_claims = [replace(c, operand="right" if c.operand == "left" else "left") for c in claims]
        # Changing semantic slots after binding must not be accepted.
        with self.assertRaisesRegex(ContractError, "not_equal_to_source"):
            execute_typed_plan(request("subtract"), reversed_claims, windows, sources)

    def test_bare_metric_mean_preserves_property_and_precision(self):
        claims, windows, sources = bind(["My rating was 3.1.", "My rating was 3.8."],
            [selection(actor="My", entity="rating", key="rating", attribute="rating", numbers=[0], quantity_kind="scalar")] * 2)
        self.assertEqual(execute_typed_plan(request("mean", question="Use two decimal places", precision="two"),
                                          claims, windows, sources)["answer"], "3.45")

    def test_mixed_scalar_properties_are_not_averaged_together(self):
        claims, windows, sources = bind(["My rating was 3.1.", "My temperature was 3.8."],
            [selection(actor="My", entity=name, attribute=name, numbers=[0], quantity_kind="scalar") for name in ("rating", "temperature")])
        with self.assertRaisesRegex(ContractError, "incompatible_units"):
            execute_typed_plan(request("mean"), claims, windows, sources)

    def test_compound_durations_bind_components_before_subtraction(self):
        claims, windows, sources = bind(["My earlier trial took 18 minutes and 25 seconds.", "My later trial took 16 minutes and 15 seconds."],
            [selection(actor="My", entity=entity, operand=side, numbers=[0, 1], quantity_kind="measure")
             for entity, side in (("earlier trial", "left"), ("later trial", "right"))])
        result = execute_typed_plan(request("subtract", question="How many seconds?", output_unit="seconds"), claims, windows, sources)
        self.assertEqual(result["answer"], "130 seconds")

    def test_weekday_identity_is_not_activity_identity(self):
        texts = ["I attend on Monday.", "I attend on Tuesday.", "I attend on Monday.", "I attend on Friday."]
        claims, windows, sources = bind(texts, [selection(entity="attend", key=day) for day in ("Monday", "Tuesday", "Monday", "Friday")])
        self.assertEqual(execute_typed_plan(request("count_distinct", key_role="weekday"), claims, windows, sources)["answer"], "3")

    def test_related_birth_occurrences_are_not_merged_by_generic_child_label(self):
        texts = ["My cousin welcomed a child.", "My cousin welcomed another child."]
        claims, windows, sources = bind(texts, [selection(actor="My cousin", binding="requested_relation", relation_anchor="My cousin", entity="child")] * 2)
        self.assertEqual(execute_typed_plan(request("count", key_role="event"), claims, windows, sources)["answer"], "2")

    def test_explicit_missing_contribution_is_not_a_partial_total(self):
        claims, windows, sources = bind(["My initial stage lasted 3 years.", "I finished the final stage; its duration is unrecorded."],
            [selection(actor="My", entity="initial stage", numbers=[0], quantity_kind="measure"),
             selection(entity="final stage", quantity_kind="missing", missing_span="duration is unrecorded")])
        result = execute_typed_plan(request(), claims, windows, sources)
        self.assertEqual(result["coverage"]["status"], "EXPLICIT_MISSING_INPUT")
        self.assertTrue(result["abstention_supported"])
        self.assertFalse(result["executable"])
        json.dumps(result)

    def test_wrong_period_is_not_a_zero_count(self):
        claims, windows, sources = bind(["I used a lamp in March 2020."], [selection(entity="lamp", key="lamp", time="March 2020")])
        result = execute_typed_plan(request("count_distinct", question="During December 2020?", time_scope="December 2020"), claims, windows, sources)
        self.assertEqual(result["coverage"]["status"], "OUTSIDE_REQUESTED_SCOPE")
        self.assertIsNone(result["answer"])
        self.assertFalse(result["abstention_supported"])

    def test_unknown_period_is_not_silently_filtered_out(self):
        claims, windows, sources = bind(["I used a lamp."], [selection(entity="lamp", key="lamp")])
        result = execute_typed_plan(request("count_distinct", question="During December 2020?", time_scope="December 2020"), claims, windows, sources)
        self.assertEqual(result["coverage"]["status"], "UNRESOLVED_TIME")

    def test_missing_arithmetic_operand_is_not_zero_or_supported_abstention(self):
        claims, windows, sources = bind(["I paid $17."], [selection(entity="paid", operand="left", numbers=[0], quantity_kind="measure")])
        result = execute_typed_plan(request("subtract"), claims, windows, sources)
        self.assertEqual(result["coverage"]["required_operands_missing"], ["right"])
        self.assertFalse(result["abstention_supported"])

    def test_source_link_reconstructs_a_host_venue(self):
        claims, windows, sources = bind(["I presented at the Lantern Symposium.", "The Lantern Symposium was hosted at Cedar University."],
            [selection(entity="Lantern Symposium", key="Lantern Symposium", kind="context", operand="context"),
             selection(actor="Lantern Symposium", binding="linked_event", entity="Lantern Symposium", key="Cedar University", link_index=0)])
        result = execute_typed_plan(request("lookup", projection="entity", key_role="location"), claims, windows, sources)
        self.assertEqual(result["answer"], "Cedar University")
        self.assertEqual(len(result["provenance"]), 2)

    def test_direct_factual_answer_is_not_promoted_as_multi_hop(self):
        claims, windows, sources = bind(["I presented at Cedar University."], [selection(entity="presented", key="Cedar University")])
        with self.assertRaisesRegex(ContractError, "multi_edge"):
            execute_typed_plan(request("lookup", projection="entity"), claims, windows, sources)

    def test_calendar_endpoints_are_independent_event_dates(self):
        claims, windows, sources = bind(["I placed my order on 04 February 2020.", "I received my order on 04 October 2020."],
            [selection(entity="order", operand=side, time=date) for side, date in (("left", "04 February 2020"), ("right", "04 October 2020"))])
        result = execute_typed_plan(request("calendar_difference", question="How many months?", output_unit="months"), claims, windows, sources)
        self.assertEqual(result["answer"], "8 months")

    def test_partial_month_does_not_invent_a_rounding_policy(self):
        with self.assertRaisesRegex(ContractError, "policy"):
            calendar_difference(calendar_value("2020-01-31"), calendar_value("2020-02-29"), "months")


class TypedIsolationTests(unittest.TestCase):
    def test_failure_restores_llm_state_and_never_promotes(self):
        ctx = context([])
        sources = {"S": Source("S", "s", "t", "user", "user", None, "I paid $17.")}
        with patch("PRAGMOS_composable_candidate._call", side_effect=ContractError("simulated failure")):
            result = run_typed_candidate(ctx, "Any request", sources, n_ctx=2048)
        self.assertTrue(ctx.llm.restored)
        self.assertEqual(result["reason"], "simulated failure")
        self.assertFalse(promotable(result))

    def test_shadow_cannot_promote_an_answer_or_supported_abstention(self):
        report = {"status": "COMPLETE", "shadow": True, "extraction_complete": True,
                  "isolation": "llama_state_restored_no_external_cache", "answer": "17", "supported_abstention": True,
                  "proof": {"executable": True, "provenance": ["source"], "abstention_supported": True,
                            "coverage": {"status": "EXPLICIT_MISSING_INPUT"}}}
        self.assertFalse(promotable(report))
        self.assertFalse(supported_typed_abstention(report))


class TypedArithmeticSafetyTests(unittest.TestCase):
    def test_typed_rate_product_preserves_then_cancels_item_dimension(self):
        claims, windows, sources = bind(["I sold 9 packets.", "I received $3 per packet."],
            [selection(entity="packets", attribute="packets", operand="left", numbers=[0], quantity_kind="count"),
             selection(entity="packet", attribute="packet", operand="right", numbers=[0], quantity_kind="rate")])
        self.assertIn(("count:packet", -1), claims[1].quantity.dimensions)
        result = execute_typed_plan(request("multiply"), claims, windows, sources)
        self.assertEqual(result["answer"], "$27")

    def test_unit_price_requires_equal_pricing_not_only_a_total(self):
        texts = ["I paid $56 for the mugs.", "I received 7 mugs; all at the same price."]
        rows = [selection(entity="mugs", operand="left", numbers=[0], quantity_kind="measure"),
                selection(entity="mugs", attribute="mugs", operand="right", numbers=[0], quantity_kind="count")]
        claims, windows, sources = bind(texts, rows)
        with self.assertRaisesRegex(ContractError, "uniformity"):
            execute_typed_plan(request("divide", per_member=True), claims, windows, sources)
        rows[1]["uniformity"] = "all at the same price"
        claims, windows, sources = bind(texts, rows)
        self.assertEqual(execute_typed_plan(request("divide", per_member=True), claims, windows, sources)["answer"], "$8")

    def test_zero_denominator_and_wrong_item_type_are_rejected(self):
        for count, noun in ((0, "packets"), (9, "mugs")):
            claims, windows, sources = bind(["I paid $27.", f"I received {count} {noun}."],
                [selection(entity="paid", operand="left", numbers=[0], quantity_kind="measure"),
                 selection(entity=noun, attribute=noun, operand="right", numbers=[0], quantity_kind="count")])
            if count == 0:
                with self.assertRaisesRegex(ContractError, "zero_denominator"):
                    execute_typed_plan(request("divide"), claims, windows, sources)
            else:
                # Multiplication cannot cancel a packet rate against a mug count.
                rate_claims, rate_windows, rate_sources = bind([f"I sold {count} {noun}.", "I received $3 per packet."],
                    [selection(entity=noun, attribute=noun, operand="left", numbers=[0], quantity_kind="count"),
                     selection(entity="packet", attribute="packet", operand="right", numbers=[0], quantity_kind="rate")])
                with self.assertRaisesRegex(ContractError, "safe_display_unit"):
                    execute_typed_plan(request("multiply"), rate_claims, rate_windows, rate_sources)

    def test_percentage_and_absolute_savings_are_different_calculations(self):
        texts = ["I quoted $100 for the first order.", "I paid $80 for the first order.",
                 "I quoted $60 for the second order.", "I paid $42 for the second order."]
        rows = [selection(entity=group, key=group, operand=operand, group_side=side, numbers=[0], quantity_kind="measure")
                for group, operand, side in (("first", "left", "left"), ("first", "right", "left"),
                                              ("second", "left", "right"), ("second", "right", "right"))]
        claims, windows, sources = bind(texts, rows)
        relative = execute_typed_plan(request("relative_difference", grouped=True, comparison="greater"), claims, windows, sources)
        absolute = execute_typed_plan(request("subtract", grouped=True, comparison="greater"), claims, windows, sources)
        self.assertEqual(relative["answer"], "No")
        self.assertEqual(absolute["answer"], "Yes")

    def test_percent_rendering_is_not_scaled_twice_or_truncated(self):
        claims, windows, sources = bind(["I quoted $100.", "I paid $80."],
            [selection(entity="quoted", operand="left", numbers=[0], quantity_kind="measure"),
             selection(entity="paid", operand="right", numbers=[0], quantity_kind="measure")])
        result = execute_typed_plan(request("relative_difference", question="What percent?", output_unit="percent"), claims, windows, sources)
        self.assertEqual(result["answer"], "20%")

    def test_grouped_endpoint_differences_preserve_winning_entity(self):
        texts = ["My Cedar account had 180 followers.", "My Cedar account had 260 followers.",
                 "My Birch account had 420 followers.", "My Birch account had 470 followers."]
        rows = [selection(actor="My", entity=group, key=group, attribute="followers", operand=side, numbers=[0], quantity_kind="count")
                for group, side in (("Cedar", "right"), ("Cedar", "left"), ("Birch", "right"), ("Birch", "left"))]
        claims, windows, sources = bind(texts, rows)
        result = execute_typed_plan(request("subtract", grouped=True, rank="max", projection="entity"), claims, windows, sources)
        self.assertEqual(result["answer"], "Cedar")

    def test_ambiguous_group_join_is_not_resolved_by_first_row(self):
        texts = ["I quoted $100 at Cedar.", "I paid $80 at Cedar.", "I paid $70 at Cedar."]
        rows = [selection(entity="Cedar", key="Cedar", operand=side, numbers=[0], quantity_kind="measure") for side in ("left", "right", "right")]
        claims, windows, sources = bind(texts, rows)
        with self.assertRaisesRegex(ContractError, "ambiguous_right"):
            execute_typed_plan(request("subtract", grouped=True), claims, windows, sources)

    def test_all_missing_inputs_are_not_replaced_by_an_empty_sum(self):
        claims, windows, sources = bind(["I finished my stage; its duration is unrecorded."],
            [selection(entity="stage", quantity_kind="missing", missing_span="duration is unrecorded")])
        result = execute_typed_plan(request(), claims, windows, sources)
        self.assertTrue(result["abstention_supported"])
        self.assertIsNone(result["answer"])

    def test_request_cannot_invent_precision_or_scope(self):
        with self.assertRaisesRegex(ContractError, "question_grounded"):
            request(question="An ordinary question", precision="two")
        with self.assertRaisesRegex(ContractError, "question_grounded"):
            request(question="An ordinary question", time_scope="March 2020")


class TypedCounterexampleTests(unittest.TestCase):
    def test_named_speakers_and_their_relatives_cannot_be_mixed_in_one_total(self):
        for related in (False, True):
            with self.subTest(related=related):
                spec = selection(actor="My cousin", binding="requested_relation", relation_anchor="My cousin",
                                 entity="paid", numbers=[0], quantity_kind="measure") if related else selection(
                                 entity="paid", numbers=[0], quantity_kind="measure")
                texts = ["My cousin paid $17.", "My cousin paid $29."] if related else ["I paid $17.", "I paid $29."]
                claims, windows, sources = bind(texts, [spec] * 2, speakers=["Asha", "Ciara"])
                query = replace(request(), target_actor="Asha", actor_scope="related" if related else "speaker")
                result = execute_typed_plan(query, claims, windows, sources)
                self.assertEqual(result["answer"], "$17")

    def test_unbound_multi_speaker_scope_does_not_call_a_model_or_claim_absence(self):
        from PRAGMOS_typed_computation import run_typed_computation
        from types import SimpleNamespace
        sources = {"A": Source("A", "s", "t", "user", "Asha", None, "I paid $17."),
                   "C": Source("C", "s", "u", "user", "Ciara", None, "I paid $29.")}
        with patch("PRAGMOS_composable_candidate._call", side_effect=AssertionError("No bound actor")):
            with self.assertRaisesRegex(ContractError, "explicit_bound_actor"):
                run_typed_computation(SimpleNamespace(), "What total?", sources, {"calls": {}}, n_ctx=2048,
                                      max_answer_tokens=64, contract={})

    def test_both_adapters_record_typed_sources_and_protocol_only_when_enabled(self):
        from pathlib import Path
        import sys
        import tempfile
        import PRAGMOS_benchmark_LongMemEval as lme
        import PRAGMOS_benchmark_LoCoMo as locomo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model = root / "data.json", root / "model.gguf"
            dataset.write_text("[]\n")
            model.write_bytes(b"test identity, not model weights")
            for adapter in (lme, locomo):
                for mode in ("off", "shadow", "candidate"):
                    with self.subTest(adapter=adapter.__name__, mode=mode), patch.object(sys, "argv", [
                        "fixture", "--data-file", str(dataset), "--model-path", str(model), "--mode", "pragmos_context",
                        "--pragmos-composable-mode", mode, "--pragmos-composable-protocol", "typed_v4",
                    ]):
                        options = adapter.parse_args()
                    manifest = adapter.build_run_manifest(options, str(dataset), []) if adapter is lme else adapter.build_manifest(options, [])
                    self.assertEqual(manifest["config"]["pragmos_composable_protocol"], "typed_v4")
                    for name in ("PRAGMOS_typed_claims.py", "PRAGMOS_typed_computation.py", "PRAGMOS_typed_binding.py"):
                        self.assertEqual(name in manifest["source_identities"], mode != "off")

    def test_small_model_orchestration_uses_literal_selections_not_generated_answers(self):
        from PRAGMOS_typed_computation import run_typed_computation
        from types import SimpleNamespace
        sources = {"S": Source("S", "not-for-model-session", "not-for-model-turn", "user", "user", None,
                               "I paid $17 for studio supplies.")}
        responses = [{"operation": "sum"}, {"answer_kind": "a number or measured quantity"},
                     {"statuses": ["completed", "ongoing"]},
                     {"decision": "support"}, {"facts": [{"entity": "studio supplies", "actor": "I"}]},
                     {"status": "completed"}]
        report = {"calls": {"planner_calls": 0, "extractor_calls": 0}}
        prompts = []
        def generate(_context, prompt, stage, *_args, schema, **_kwargs):
            prompts.append(prompt)
            report["calls"][stage + "_calls"] += 1
            return responses.pop(0)
        with patch("PRAGMOS_composable_candidate._call", side_effect=generate):
            result = run_typed_computation(SimpleNamespace(count_tokens=lambda _: 10),
                "What total did I spend on studio supplies?", sources, report, n_ctx=2048,
                max_answer_tokens=64, contract={})
        self.assertEqual(result["answer"], "$17")
        self.assertTrue(result["proof"]["executable"])
        self.assertEqual(responses, [])
        self.assertTrue(all("not-for-model" not in prompt for prompt in prompts))

    def test_actor_choices_do_not_include_action_adverbs_or_self_action_phrases(self):
        from PRAGMOS_typed_claims import actor_choices
        actors = actor_choices("I've already read 13 pages. My cousin Mira read 8.")
        self.assertIn("I've", actors)
        self.assertNotIn("already", actors)
        self.assertNotIn("I've already read", actors)

    def test_duplicate_extractions_cannot_multiply_one_measurement(self):
        claims, windows, sources = bind(["I paid $17."], [selection(entity="paid", numbers=[0], quantity_kind="measure")])
        with self.assertRaisesRegex(ContractError, "independent_facts"):
            execute_typed_plan(request(), claims + claims, windows, sources)

    def test_scalar_arithmetic_is_independent_of_process_global_decimal_precision(self):
        from decimal import localcontext
        claims, windows, sources = bind(["I paid $17.", "I paid $29."],
            [selection(entity="paid", numbers=[0], quantity_kind="measure")] * 2)
        with localcontext() as ctx:
            ctx.prec = 1
            self.assertEqual(execute_typed_plan(request(), claims, windows, sources)["answer"], "$46")

    def test_pending_and_negated_assertions_do_not_enter_a_completed_total(self):
        claims, windows, sources = bind(["I spent $17.", "I still need to pay $29.", "I did not pay $38."],
            [selection(entity="spent", status="completed", numbers=[0], quantity_kind="measure"),
             selection(entity="pay", status="pending", numbers=[0], quantity_kind="measure"),
             selection(entity="pay", status="negated", numbers=[0], quantity_kind="measure")])
        completed = replace(request(), statuses=("completed", "ongoing"))
        self.assertEqual(execute_typed_plan(completed, claims, windows, sources)["answer"], "$17")

    def test_derived_result_status_does_not_filter_its_different_input_states(self):
        claims, windows, sources = bind(["My workbook has 81 pages.", "I already read 13 pages."],
            [selection(actor="My", entity="workbook", operand="left", numbers=[0], quantity_kind="measure"),
             selection(entity="read", operand="right", numbers=[0], quantity_kind="measure")])
        query = replace(request("subtract"), statuses=("pending",),
                        operand_statuses=(("left", ("completed", "ongoing")), ("right", ("completed",))))
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["answer"], "68 pages")
        pending_read = selection(entity="read", operand="right", status="planned", numbers=[0], quantity_kind="measure")
        future_claims, future_windows, future_sources = bind(["My workbook has 81 pages.", "I plan to read 13 pages."],
            [selection(actor="My", entity="workbook", operand="left", numbers=[0], quantity_kind="measure"), pending_read])
        self.assertEqual(execute_typed_plan(query, future_claims, future_windows, future_sources)["coverage"]["status"], "MISSING_OPERAND")

    def test_only_adjacent_compound_measurements_are_combined(self):
        from PRAGMOS_typed_claims import measurement_groups
        self.assertEqual(measurement_groups("I took 18 minutes and 25 seconds."), [[0, 1]])
        self.assertEqual(measurement_groups("I took 18 minutes, then my friend took 25 seconds."), [[0], [1]])

    def test_missing_value_from_an_unrelated_neighbor_is_not_a_qualification(self):
        source = Source("S", "s", "t", "user", "user", None,
                        "I finished the studio stage. My unrelated trip duration is unrecorded.")
        window = source_windows({"S": source})[0]
        with self.assertRaisesRegex(ContractError, "not_bound_to_focus"):
            bind_typed_claim(window, source, selection(entity="studio stage", quantity_kind="missing",
                             missing_span="duration is unrecorded"), {})

    def test_outside_scope_veto_requires_the_same_positive_baseline_calculation(self):
        claims, windows, sources = bind(["I used a lamp in March 2020."],
            [selection(entity="lamp", key="lamp", time="March 2020")])
        query = request("count_distinct", question="During December 2020?", time_scope="December 2020")
        for answer, supported in ((None, False), ("9", False), ("1", True)):
            with self.subTest(answer=answer):
                result = execute_typed_plan(query, claims, windows, sources, baseline_answer=answer)
                self.assertEqual(result["abstention_supported"], supported)
                self.assertFalse(result["coverage"]["global_retrieval_completeness_proven"])

    def test_only_a_bound_intermediate_can_be_vetoed_for_missing_relation(self):
        claims, windows, sources = bind(["I presented at the Lantern Symposium."],
            [selection(entity="Lantern Symposium", key="Lantern Symposium", kind="context", operand="context")])
        query = request("lookup", projection="entity", key_role="location")
        self.assertTrue(execute_typed_plan(query, claims, windows, sources,
                                          baseline_answer="Lantern Symposium")["abstention_supported"])
        self.assertFalse(execute_typed_plan(query, claims, windows, sources,
                                           baseline_answer="Cedar University")["abstention_supported"])

    def test_opt_in_typed_protocol_cannot_enter_protected_intents(self):
        from PRAGMOS_benchmark_LongMemEval import evaluate_composable_candidate
        from test_PRAGMOS_composable_candidate import aggregate_plan, args
        for intent in ("assistant_memory", "preference_recommendation", "evidence_synthesis", "knowledge_update"):
            with self.subTest(intent=intent), patch("PRAGMOS_composable_candidate.run_typed_candidate",
                                                   side_effect=AssertionError("protected model path")):
                result = evaluate_composable_candidate(None, "Question", args(pragmos_composable_protocol="typed_v4"),
                    semantic_plan=aggregate_plan(), query_intent={"intent": intent}, source_rows=None, turn_lookup=None)
                self.assertFalse(result["promoted"])

    def test_shadow_boundary_marks_an_otherwise_executable_candidate_non_promotable(self):
        from PRAGMOS_benchmark_LongMemEval import evaluate_composable_candidate
        from test_PRAGMOS_composable_candidate import aggregate_plan, args
        report = {"status": "COMPLETE", "answer": "$46", "extraction_complete": True,
                  "isolation": "llama_state_restored_no_external_cache",
                  "proof": {"executable": True, "provenance": ["source"]}}
        with patch("PRAGMOS_composable_candidate.run_typed_candidate", return_value=report):
            result = evaluate_composable_candidate(None, "Question", args("shadow", pragmos_composable_protocol="typed_v4"),
                semantic_plan=aggregate_plan(), query_intent={}, source_rows=[{"source_turn_id": "1"}],
                turn_lookup={"1": {"session_id": "s", "role": "user", "speaker": "user", "text": "I paid $46."}})
        self.assertTrue(result["shadow"])
        self.assertFalse(promotable(result))


if __name__ == "__main__":
    unittest.main()
