"""Generic evidence-first execution counterexamples; fixtures are not model accuracy."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_state_binding import _frame_value, _typed_frame_value
from PRAGMOS_state_reader import assertion_view, bind_evidence_frames, evidence_frame_schema
from PRAGMOS_state_views import accepted_state_candidate, run_state_candidate
from PRAGMOS_state_program import bind_evidence_request, evidence_request_schema
from test_PRAGMOS_composable_candidate import context


def claim(at, value, **changes):
    return {"at": at, "subject": "My", "subject_type": "person", "owner": None,
        "entity": None, "property": "balance", "values": [value], "kind": "state", "previous": None,
        "unit": None, "per": None, "set_complete": False, "time": None, **changes}


def leaf(series="G1", selector="latest", op="read", **changes):
    return {"op": op, "series": series, "selector": selector, "marker": None, "modality": "actual", **changes}


def request(question, expression, groups=None, **changes):
    return {"request": {"subject": "my" if "my" in question else None, "entity": None, "qualifiers": []},
        "groups": groups if groups is not None else [{"id": "G1", "facts": ["F1", "F2"],
            "request": question, "aggregation": "state"}],
        "links": [], "uncertainty": [], "expression": expression, "rounding": None, "read_more": [], **changes}


def sources(texts, speaker="user"):
    return {"S" + str(i): Source("S" + str(i), "s" + str(i), str(i), "user", speaker,
            f"2025-01-{i:02d}", text) for i, text in enumerate(texts, 1)}


class EvidenceFirstTests(unittest.TestCase):
    def run_case(self, question, texts, claims, plan, *, baseline="$12", unresolved=(), source_map=None):
        plan = {**plan, "groups": [{"scope": None, **group} for group in plan["groups"]]}
        data = source_map or sources(texts)
        view = assertion_view([(source, 0, source.text) for source in data.values()])
        keys = list(view)
        batches = [set(keys[i:i+3]) for i in range(0, len(keys), 3)]
        replies = [{"claims": [c for c in claims if c["at"] in batch],
                    "unresolved": [key for key in unresolved if key in batch]} for batch in batches]
        ctx = context([*replies, plan])
        with patch("PRAGMOS_composable_candidate.schema_grammar", return_value=None):
            result = run_state_candidate(ctx, question, data, n_ctx=2048, baseline_answer=baseline,
                                         protocol="evidence_v4")
        self.assertTrue(ctx.llm.restored)
        self.assertEqual(ctx.llm.position, 0)
        return result, ctx

    def test_latest_source_value_without_question_type_classification(self):
        q = "What is my balance?"
        result, ctx = self.run_case(q, ["My balance was $20.", "My balance is $35."],
            [claim("A1", "$20"), claim("A2", "$35")], request(q, leaf()))
        self.assertEqual(result["answer"], "$35")
        self.assertTrue(accepted_state_candidate(result, "candidate"))
        self.assertEqual(len(ctx.llm.calls), 2)
        self.assertEqual(result["version"], 4)
        self.assertNotIn("Select the requested ANSWER TYPE", " ".join(p for p, _ in ctx.llm.calls))

    def test_wrong_person_cannot_replace_an_existing_correct_answer(self):
        q = "What is my balance?"
        plan = request(q, leaf(), [{"id": "G1", "facts": ["F2"], "request": q, "aggregation": "state"}])
        result, _ = self.run_case(q, ["My balance is $20.", "Nora's balance is $35."],
            [claim("A1", "$20"), claim("A2", "$35", subject="Nora", property="balance")], plan, baseline="$20")
        self.assertEqual(result["reason"], "state_requested_holder_mismatch")
        self.assertFalse(accepted_state_candidate(result, "candidate"))

    def test_person_possessive_is_not_flattened_to_the_narrator(self):
        q = "What is my balance?"
        result, _ = self.run_case(q, ["My balance is $20.", "My mother's balance is $35."],
            [claim("A1", "$20"), claim("A2", "$35", subject="My mother", owner="My")],
            request(q, leaf(), [{"id": "G1", "facts": ["F2"], "request": q, "aggregation": "state"}]), baseline="$20")
        self.assertEqual(result["reason"], "state_requested_holder_mismatch")

    def test_general_rule_does_not_need_a_human_owner(self):
        q = "What is the programme fee?"
        result, _ = self.run_case(q, ["The programme fee is $25."],
            [claim("A1", "$25", subject="The programme", subject_type="rule", property="fee")],
            request(q, leaf(), [{"id": "G1", "facts": ["F1"], "request": q, "aggregation": "state"}]))
        self.assertEqual(result["answer"], "$25")

    def test_recollection_and_plan_do_not_replace_actual_state(self):
        q = "What is my balance?"
        result, _ = self.run_case(q, ["My balance is $20.", "My balance is $35.",
            "My old screenshot shows a balance of $20.", "My target balance is $80."],
            [claim("A1", "$20"), claim("A2", "$35"), claim("A3", "$20", kind="reference"),
             claim("A4", "$80", kind="plan")],
            request(q, leaf(), [{"id": "G1", "facts": ["F1", "F2", "F3", "F4"], "request": q, "aggregation": "state"}]))
        self.assertEqual(result["answer"], "$35")

    def test_source_range_and_qualification_are_restored(self):
        q = "What is my effort?"
        result, _ = self.run_case(q, ["My effort is about 9-12 hours."],
            [claim("A1", "9-12 hours", property="effort")],
            request(q, leaf(), [{"id": "G1", "facts": ["F1"], "request": q, "aggregation": "state"}]))
        self.assertEqual(result["answer"], "about 9-12 hours")

    def test_two_requested_outputs_remain_separate(self):
        q = "What was my balance initially, and what is my balance now?"
        expression = {"op": "tuple", "inputs": [leaf(selector="earliest"), leaf()],
                      "labels": ["balance initially", "balance now"]}
        result, _ = self.run_case(q, ["My balance was $20.", "My balance is $35."],
            [claim("A1", "$20"), claim("A2", "$35")], request(q, expression), baseline="$20")
        self.assertIn("$20", result["answer"])
        self.assertIn("$35", result["answer"])

    def test_source_unit_arithmetic_uses_both_endpoints(self):
        q = "What is my weight loss?"
        result, _ = self.run_case(q, ["My weight was 90 kg.", "My weight is 84 kg."],
            [claim("A1", "90 kg", property="weight"), claim("A2", "84 kg", property="weight")],
            request(q, {"op": "subtract", "left": leaf(selector="earliest"), "right": leaf()}), baseline="8 kg")
        self.assertEqual(result["answer"], "6 kg")

    def test_counted_members_are_not_model_generated_answer_numerals(self):
        q = "How many venues have I visited?"
        rows = [claim("A1", "Cedar Hall", subject="I", property="visited"),
                claim("A1", "Larch Inn", subject="I", property="visited"),
                claim("A2", "Cedar Hall", subject="I", property="visited"),
                claim("A3", "Oak Room", subject="I", property="visited"),
                claim("A4", "Pine House", subject="I", property="plan", kind="plan")]
        plan = request(q, {"op": "count", "input": leaf(selector="all", op="distinct")},
            [{"id": "G1", "facts": ["F1", "F2", "F3", "F4", "F5"], "request": q, "aggregation": "state"}],
            request={"subject": "I", "entity": None, "qualifiers": []})
        result, _ = self.run_case(q, ["I visited Cedar Hall and Larch Inn.", "I visited Cedar Hall again.",
            "I visited Oak Room.", "I plan to visit Pine House."], rows, plan, baseline="1")
        self.assertEqual(result["answer"], "3")

    def test_compare_same_property_across_two_people(self):
        q = "Is my mother using the same tool as me?"
        plan = request(q, {"op": "equal", "left": leaf(), "right": leaf("G2")},
            [{"id": "G1", "facts": ["F1"], "request": "tool", "aggregation": "state"},
             {"id": "G2", "facts": ["F2"], "request": "tool", "aggregation": "state"}],
            request={"subject": "my mother", "entity": None, "qualifiers": []})
        result, _ = self.run_case(q, ["I use NoteNest.", "My mother uses NoteNest."],
            [claim("A1", "NoteNest", subject="I", property="use"),
             claim("A2", "NoteNest", subject="My mother", property="uses")], plan, baseline="No")
        self.assertEqual(result["answer"], "Yes")

    def test_unrelated_uncertainty_does_not_block_a_known_value(self):
        q = "What is my balance?"
        plan = request(q, leaf(), [{"id": "G1", "facts": ["F1"], "request": q, "aggregation": "state"}],
            uncertainty=[{"at": "A2", "groups": [], "status": "irrelevant"}])
        result, _ = self.run_case(q, ["My balance is $20.", "My tax estimate may be unclear."],
            [claim("A1", "$20")], plan, unresolved=["A2"])
        self.assertEqual(result["answer"], "$20")

    def test_uncertain_required_operand_cannot_be_ignored(self):
        q = "What is my balance?"
        plan = request(q, leaf(), [{"id": "G1", "facts": ["F1"], "request": q, "aggregation": "state"}],
            uncertainty=[{"at": "A2", "groups": ["G1"], "status": "unresolved"}])
        result, _ = self.run_case(q, ["My balance is $20.", "My balance may have changed."],
            [claim("A1", "$20")], plan, unresolved=["A2"])
        self.assertEqual(result["reason"], "state_program_required_assertion_unresolved")

    def test_snapshot_sum_is_rejected(self):
        q = "What is my total balance?"
        result, _ = self.run_case(q, ["My balance was $20.", "My balance is $35."],
            [claim("A1", "$20"), claim("A2", "$35")], request(q, leaf(op="sum", selector="all")))
        self.assertIn("independent_contributions", result["reason"])

    def test_wrong_entity_absence_is_conditional_not_world_falsehood(self):
        q = "What is my balance at New Bank?"
        plan = request(q, {"op": "unknown", "missing": ["New Bank"]}, [],
            request={"subject": "my", "entity": "New Bank", "qualifiers": []})
        result, _ = self.run_case(q, ["My Old Bank balance is $20."], [claim("A1", "$20", entity="Old Bank")],
                                   plan, baseline="$20")
        self.assertEqual(result["answer"], "I do not know.")
        self.assertFalse(result["proof"]["full_conversation_absence_proven"])
        self.assertFalse(accepted_state_candidate(result, "candidate"))

    def test_entity_name_with_number_is_not_a_measurement(self):
        value, _ = _typed_frame_value("Model 7", None)
        self.assertEqual(value.kind, "text")

    def test_source_assertion_offsets_survive_repeated_text(self):
        source = next(iter(sources(["My balance is $20. My balance is $20."]).values()))
        view = assertion_view([(source, 0, source.text)])
        self.assertEqual(len(view), 2)
        offsets = [a.start for a in view.values()]
        self.assertNotEqual(*offsets)
        self.assertTrue(all(source.text[a.start:a.end] == a.text for a in view.values()))

    def test_fabricated_source_value_is_rejected(self):
        source = next(iter(sources(["My balance is $20."]).values()))
        view = assertion_view([(source, 0, source.text)])
        with self.assertRaises(ContractError):
            bind_evidence_frames({"claims": [claim("A1", "$90")], "unresolved": []}, view)

    def test_array_members_expand_with_one_source_address(self):
        source = next(iter(sources(["My list contains Cedar Hall and Larch Inn."]).values()))
        view = assertion_view([(source, 0, source.text)])
        rows, _ = bind_evidence_frames({"claims": [claim("A1", "Cedar Hall", property="contains",
            values=["Cedar Hall", "Larch Inn"], set_complete=True)], "unresolved": []}, view)
        self.assertEqual([f.value for f in rows], ["Cedar Hall", "Larch Inn"])
        self.assertTrue(all(f.complete and f.assertion_id == "A1" for f in rows))

    def test_literal_choices_belong_to_the_addressed_assertion(self):
        source = next(iter(sources(["My balance is $20. Nora's balance is $35."]).values()))
        view = assertion_view([(source, 0, source.text)])
        schema = evidence_frame_schema(view)
        self.assertIn("$20", schema["$defs"]["A1_local"]["enum"])
        self.assertNotIn("$35", schema["$defs"]["A1_local"]["enum"])
        with self.assertRaises(ContractError):
            bind_evidence_frames({"claims": [claim("A1", "$35")], "unresolved": []}, view)

    def test_ambiguous_bare_measurement_is_not_given_a_nearby_unit(self):
        source = next(iter(sources(["My weight changed by 6 kg during 6 hours."]).values()))
        view = assertion_view([(source, 0, source.text)])
        rows, _ = bind_evidence_frames({"claims": [claim("A1", "6", property="weight")], "unresolved": []}, view)
        with self.assertRaisesRegex(ContractError, "ambiguous_measurement"):
            _frame_value(rows[0], view["A1"])

    def test_count_noun_is_only_restored_when_adjacent(self):
        source = next(iter(sources(["My inventory is 6 boxes; each order has 2 parts."]).values()))
        view = assertion_view([(source, 0, source.text)])
        rows, _ = bind_evidence_frames({"claims": [claim("A1", "6", property="inventory", unit="boxes")], "unresolved": []}, view)
        self.assertEqual(_frame_value(rows[0], view["A1"]), "6 boxes")
        self.assertEqual(_frame_value(replace(rows[0], unit="parts"), view["A1"]), "6")

    def test_a_range_endpoint_cannot_become_an_exact_measurement(self):
        source = next(iter(sources(["My effort is about 9-12 hours."]).values()))
        view = assertion_view([(source, 0, source.text)])
        for endpoint in ("9", "12", "12 hours"):
            with self.subTest(endpoint=endpoint):
                rows, _ = bind_evidence_frames({"claims": [claim("A1", endpoint, property="effort")], "unresolved": []}, view)
                self.assertEqual(_frame_value(rows[0], view["A1"]), "about 9-12 hours")

    def test_repeated_bare_number_cannot_borrow_another_occurrences_unit(self):
        source = next(iter(sources(["My reading count was 6; my weight change was 6 kg."]).values()))
        view = assertion_view([(source, 0, source.text)])
        rows, _ = bind_evidence_frames({"claims": [claim("A1", "6", property="count")], "unresolved": []}, view)
        with self.assertRaisesRegex(ContractError, "ambiguous_measurement"):
            _frame_value(rows[0], view["A1"])

    def test_changing_collection_members_do_not_create_new_owner_scopes(self):
        q = "Which venues have I visited?"
        plan = request(q, leaf(op="distinct", selector="all"),
            [{"id": "G1", "facts": ["F1", "F2"], "request": q, "aggregation": "state"}],
            request={"subject": "I", "entity": None, "qualifiers": []})
        result, _ = self.run_case(q, ["I visited Cedar Hall.", "I visited Larch Inn."],
            [claim("A1", "Cedar Hall", subject="I", entity="Cedar Hall", property="visited"),
             claim("A2", "Larch Inn", subject="I", entity="Larch Inn", property="visited")], plan)
        self.assertEqual(result["answer"], "Cedar Hall; Larch Inn")

    def test_shared_parent_scope_does_not_merge_different_accounts(self):
        q = "What is my New Bank balance?"
        plan = request(q, leaf(), [{"id": "G1", "facts": ["F1", "F2"], "request": q,
            "scope": "New Bank", "aggregation": "state"}])
        result, _ = self.run_case(q, ["My New Bank checking balance is $20.", "My New Bank savings balance is $35."],
            [claim("A1", "$20", entity="checking"), claim("A2", "$35", entity="savings")], plan)
        self.assertEqual(result["reason"], "state_group_crosses_entity_scopes")

    def test_missing_extraction_cannot_overwrite_a_positive_baseline(self):
        q = "Where is my travel card?"
        plan = request(q, {"op": "unknown", "missing": ["travel card"]}, [],
            request={"subject": "my", "entity": "travel card", "qualifiers": []})
        result, _ = self.run_case(q, ["My desk drawer is tidy."],
            [claim("A1", "tidy", entity="desk drawer", property="is")], plan, baseline="desk drawer")
        self.assertFalse(accepted_state_candidate(result, "candidate"))
        self.assertEqual(result["promotion_review"]["baseline"]["status"], "unresolved")

    def test_retracted_report_is_not_an_original_genuine_state(self):
        q = "What was my original balance?"
        result, _ = self.run_case(q, ["My balance was $20.", "My earlier balance of $20 was incorrect; it was $18."],
            [claim("A1", "$20"), claim("A2", "$18", kind="correction", previous="$20")],
            request(q, leaf(selector="earliest")), baseline="$20")
        self.assertEqual(result["answer"], "$18")
        self.assertTrue(result["correction_links"])

    def test_planned_state_uses_a_planned_contract_not_actual_history(self):
        q = "Where is my planned stay?"
        result, _ = self.run_case(q, ["My previous stay was Cedar Hall.", "My planned stay is Larch Inn."],
            [claim("A1", "Cedar Hall", property="stay"), claim("A2", "Larch Inn", property="stay", kind="plan")],
            request(q, leaf(modality="planned")))
        self.assertEqual(result["answer"], "Larch Inn")
        self.assertEqual(result["answer_contract"]["modality"], "planned")

    def test_mixed_actual_and_planned_operands_do_not_silently_coerce(self):
        q = "What is my balance change?"
        result, _ = self.run_case(q, ["My balance is $20.", "My target balance is $35."],
            [claim("A1", "$20"), claim("A2", "$35", kind="plan")],
            request(q, {"op": "subtract", "left": leaf(), "right": leaf(modality="planned")}))
        self.assertEqual(result["reason"], "state_expression_mixed_or_unbound_modality")

    def test_query_literals_cannot_be_copied_from_instructions(self):
        q = "What is my balance?"
        schema = evidence_request_schema(["F1"], ["A1"], q)
        enum = schema["properties"]["request"]["properties"]["subject"]["anyOf"][1]["enum"]
        self.assertIn("my", enum)
        self.assertNotIn("literal requested person", enum)
        leaves = schema["$defs"]["expr0"]["anyOf"]
        latest = next(row for row in leaves if row["properties"]["selector"]["enum"] == ["latest"])
        self.assertEqual(latest["properties"]["marker"]["enum"], [None])

    def test_an_omitted_correct_assertion_is_not_rejected_source_scope(self):
        q = "What is my chosen venue?"
        result, _ = self.run_case(q, ["My chosen venue was Cedar Hall.", "My chosen venue is Larch Inn."],
            [claim("A1", "Cedar Hall", property="venue")],
            request(q, leaf(), [{"id": "G1", "facts": ["F1"], "request": q, "aggregation": "state"}]),
            baseline="Larch Inn")
        self.assertFalse(accepted_state_candidate(result, "candidate"))
        self.assertEqual(result["promotion_review"]["baseline"]["status"], "unresolved")

    def test_numeral_bearing_name_is_not_automatically_a_wrong_quantity(self):
        q = "What is my chosen venue?"
        result, _ = self.run_case(q, ["My chosen venue is Larch Inn."],
            [claim("A1", "Larch Inn", property="venue")],
            request(q, leaf(), [{"id": "G1", "facts": ["F1"], "request": q, "aggregation": "state"}]),
            baseline="Room 7")
        self.assertFalse(accepted_state_candidate(result, "candidate"))
        self.assertEqual(result["promotion_review"]["baseline"]["status"], "unresolved")

    def test_planner_sees_unextracted_assertions_and_scope_explanations(self):
        q = "What is my balance?"
        result, ctx = self.run_case(q, ["My balance is $20.", "My statement explains that savings are separate."],
            [claim("A1", "$20")], request(q, leaf(),
                [{"id": "G1", "facts": ["F1"], "request": q, "aggregation": "state"}]))
        self.assertIn("My statement explains that savings are separate.", ctx.llm.calls[1][0])
        self.assertIsNotNone(result["answer"])

    def test_one_addressed_retry_can_recover_a_model_omitted_fact(self):
        q = "What is my balance?"
        data = sources(["My balance was $20.", "My balance is $35."])
        initial = request(q, {"op": "unknown", "missing": ["balance"]}, [], read_more=["A2"])
        final = request(q, leaf())
        final["groups"][0]["scope"] = None
        ctx = context([{"claims": [claim("A1", "$20")], "unresolved": []}, initial,
                       {"claims": [claim("A2", "$35")], "unresolved": []}, final])
        with patch("PRAGMOS_composable_candidate.schema_grammar", return_value=None):
            result = run_state_candidate(ctx, q, data, n_ctx=2048, baseline_answer="$20", protocol="evidence_v4")
        self.assertEqual(result["answer"], "$35")
        self.assertTrue(accepted_state_candidate(result, "candidate"))
        self.assertEqual(len(ctx.llm.calls), 4)
        self.assertEqual(result["evidence_batches"][-1]["kind"], "targeted_omission_retry")
        self.assertTrue(ctx.llm.restored)

    def test_retry_cannot_reinterpret_an_already_extracted_assertion(self):
        q = "What is my balance?"
        result, _ = self.run_case(q, ["My balance is $20."], [claim("A1", "$20")],
            request(q, {"op": "unknown", "missing": ["balance"]}, [], read_more=["A1"]))
        self.assertEqual(result["reason"], "state_read_more_not_unbound_assertions")

    def test_output_reservation_uses_available_space_without_dropping_sources(self):
        q = "What is my balance?"
        source = next(iter(sources(["My balance is $20."]).values()))
        view = assertion_view([(source, 0, source.text)])
        frames, _ = bind_evidence_frames({"claims": [claim("A1", "$20")], "unresolved": []}, view)
        plan = request(q, leaf(), [{"id": "G1", "facts": ["F1"], "request": q,
                                   "scope": None, "aggregation": "state"}])
        calls, audit = [], {}
        def ask(prompt, schema, reserve, stage):
            calls.append((prompt, reserve))
            return plan
        bind_evidence_request(ask, q, view, frames, 2048, lambda text: 1850, report=audit)
        self.assertEqual(calls[0][1], 134)
        self.assertIn(source.text, calls[0][0])
        self.assertEqual(audit["request_token_budgets"][0]["prompt_tokens"] + calls[0][1] + 64, 2048)
        with self.assertRaisesRegex(ContractError, "context_budget_exceeded"):
            bind_evidence_request(ask, q, view, frames, 2048, lambda text: 1900)


if __name__ == "__main__":
    unittest.main()
