"""Source-read admission and unchanged-route checks; no model downloads or judge calls."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import PRAGMOS_benchmark_LongMemEval as adapter
from test_PRAGMOS_composable_integration import (
    arguments, fixtures, make_context, semantic_plan, semantic_trace,
)


CASES = (
    ("In the workshop equipment list for my Alder plan, what was item 2?",
     "For my Alder plan, give me a workshop equipment list.",
     "Here is the numbered list:\n1. desk lamp\n2. folding easel\n3. drawing stool", [], "folding easel"),
    ("What phone number did you give me for the Alder field office?",
     "Do you have the phone number for the Alder field office?",
     "The Alder field office can be reached at 555-612-9073.",
     [["Alder field office", "provided phone number", "555-612-9073"]], "555-612-9073"),
    ("What color did you say the cinder kite was in the illustration?",
     "Describe the cinder kite in the illustration.",
     "The cinder kite has a moss green body with white markings.", [], "moss green"),
    ("What were the three objectives you outlined for the Orchid Project?",
     "Please outline three objectives for the Orchid Project.",
     "The three objectives are: record water levels; publish flow maps; inspect the sensors.", [],
     "record water levels; publish flow maps; inspect the sensors"),
    ("What was the cafe you recommended to Mira and me for steamed buns?",
     "Mira and I need one cafe for steamed buns. What do you recommend?",
     "I recommend Birch House; it is known for steamed buns.",
     [["speaker assistant", "recommended cafe", "Birch House"]], "Birch House"),
)


def source_case(question, request, response, triples=()):
    context = make_context()
    record = {"haystack_sessions": [[{"role": "user", "content": request},
                                     {"role": "assistant", "content": response}]],
              "haystack_session_ids": ["exchange"], "haystack_dates": ["2027-01-03"]}
    turns = adapter.longmemeval_turns(context, record)
    rows = [{"source_turn_id": turn.turn_id, "source_session_id": turn.session_id,
             "source_role": turn.role, "source_speaker": turn.speaker,
             "source_timestamp": turn.timestamp, "source_quote": turn.text,
             "triples": list(triples) if turn.role == "assistant" else []} for turn in turns]
    return record, turns, rows


def certificate(question, turns, rows, *, profile=None, ordinal=None, typed=None, candidates=None):
    profile = profile or adapter.build_query_profile(question)
    intent, slot = adapter.infer_query_intent(question), adapter.infer_requested_answer_slot(question)
    return adapter.resolve_source_projection(
        question, slot, profile, intent, adapter.infer_multi_session_operation(question, None),
        {"operation": "collect_distinct"}, adapter.infer_state_history_selector(question),
        {turn.turn_id: turn for turn in turns},
        ordinal if ordinal is not None else adapter.resolve_deterministic_ordinal_list_answer(question, rows, profile),
        typed if typed is not None else adapter.resolve_typed_answer_slot(question, slot, rows, profile, intent),
        candidates if candidates is not None else adapter.select_answer_slot_candidates(
            rows, question, slot, query_profile=profile, query_intent=intent),
    )


class SourceProjectionTests(unittest.TestCase):
    def test_all_projection_capabilities_have_literal_source_addresses(self):
        for question, request, response, triples, expected in CASES:
            with self.subTest(question=question):
                _, turns, rows = source_case(question, request, response, triples)
                before = copy.deepcopy(rows)
                proof = certificate(question, turns, rows)
                self.assertEqual(proof["status"], "complete")
                self.assertEqual(proof["answer"], expected)
                self.assertEqual(rows, before)
                for source in proof["candidates"]:
                    self.assertTrue(source["value_spans"])
                    self.assertEqual(source["source_role"], "assistant")
                    self.assertTrue(all(source["source_quote"][start:end] for start, end in source["value_spans"]))

    def test_ordinal_is_not_a_word_count_or_an_arithmetic_result(self):
        response = "The checklist uses short words:\n1. Pack the box\n2. Review the final schedule\n3. Check the lights"
        for question in ("How many words are in item 2 of the checklist you gave me?",
                         "How many words are in the three objectives you outlined?",
                         "What is the total duration of the two steps you listed?",
                         "What is the ratio of the two measurements you provided?"):
            with self.subTest(question=question):
                _, turns, rows = source_case(question, "Please provide the checklist.", response)
                self.assertNotEqual(certificate(question, turns, rows)["status"], "complete")

    def test_parent_cannot_lend_an_unrelated_objects_color(self):
        q = "What color did you say my studio door was?"
        _, turns, rows = source_case(q, "Describe my studio door.",
            "My studio door is beside a cabinet. The cabinet's color is forest green.")
        old = adapter.resolve_typed_answer_slot(q, "color", rows, adapter.build_query_profile(q), adapter.infer_query_intent(q))
        self.assertEqual(old["status"], "complete")
        self.assertNotEqual(certificate(q, turns, rows)["status"], "complete")

    def test_state_and_temporal_views_do_not_borrow_a_later_list(self):
        for boundary in ("before the revision", "immediately before the update", "after the April meeting"):
            q = "What were the three objectives you outlined for Project Elm " + boundary + "?"
            _, turns, rows = source_case(q, "Outline three objectives for Project Elm.",
                "After the revision, the three objectives are: recruit mentors; publish the handbook; hold the exhibition.")
            self.assertNotEqual(certificate(q, turns, rows)["status"], "complete")

    def test_request_placeholder_is_not_an_answer(self):
        q, request, response, _triples, _expected = CASES[4]
        _, turns, rows = source_case(q, request, response)
        rows[0]["triples"] = [["Mira", "needs", "one cafe for steamed buns"]]
        self.assertNotEqual(certificate(q, turns, rows)["status"], "complete")

    def test_adjacent_request_not_any_turn_in_the_session(self):
        q, request, response, triples, _expected = CASES[3]
        _, turns, rows = source_case(q, request, response, triples)
        from dataclasses import replace
        switched = replace(turns[0], text="List three objectives for another project.")
        self.assertNotEqual(certificate(q, [switched, turns[1]], rows)["status"], "complete")
        mismatched = replace(turns[1], parent_turn_id="unrelated-turn")
        self.assertNotEqual(certificate(q, [turns[0], mismatched], rows)["status"], "complete")

    def test_competing_source_values_remain_ambiguous(self):
        q, request, response, triples, _expected = CASES[1]
        _, turns, rows = source_case(q, request, response, triples)
        from dataclasses import replace
        other = replace(turns[1], turn_id=9, session_id="another", text=response.replace("9073", "9074"))
        rows.append({**rows[1], "source_turn_id": 9, "source_session_id": "another", "source_quote": other.text,
                     "triples": [["Alder field office", "provided phone number", "555-612-9074"]]})
        self.assertEqual(certificate(q, [*turns, other], rows)["status"], "ambiguous")

    def test_source_metadata_and_selected_window_are_mandatory(self):
        q, request, response, triples, _expected = CASES[1]
        for field, value in (("source_role", "user"), ("source_speaker", "another person"),
                             ("source_session_id", "other"), ("source_timestamp", "2030-01-01"),
                             ("source_quote", "can be reached")):
            _, turns, rows = source_case(q, request, response, triples)
            rows[1][field] = value
            with self.subTest(field=field):
                self.assertNotEqual(certificate(q, turns, rows)["status"], "complete")

    def test_partial_requested_lists_and_exclusions_are_not_certified(self):
        q, request, response, triples, _expected = CASES[3]
        for question, source in ((q, "The objectives are: record water levels; publish flow maps."),
                                 (q[:-1] + ", excluding the field work?", response)):
            _, turns, rows = source_case(question, request, source, triples)
            self.assertNotEqual(certificate(question, turns, rows)["status"], "complete")

    def test_unconfirmed_suggestions_are_not_user_preferences_or_choices(self):
        for q in ("What cafe did I choose for steamed buns?", "Which cafe do I prefer for steamed buns?",
                  "What cafe would you recommend based on my preferences?"):
            _, turns, rows = source_case(q, "Recommend a cafe for steamed buns.",
                "I recommend Birch House.", [["speaker assistant", "recommended cafe", "Birch House"]])
            self.assertEqual(certificate(q, turns, rows)["status"], "not_applicable")

    def test_unknown_binding_keeps_existing_arbitration(self):
        q, request, response, triples, _expected = CASES[1]
        _, turns, rows = source_case(q, request, response, triples)
        for text in ("Would you call 555-612-9073?", "I never provided 555-612-9073.",
                     "I plan to provide 555-612-9073."):
            from dataclasses import replace
            changed = replace(turns[1], text=text)
            changed_rows = [rows[0], {**rows[1], "source_quote": text}]
            self.assertNotEqual(certificate(q, [turns[0], changed], changed_rows)["status"], "complete")

    def test_explicit_speech_complement_does_not_need_a_generated_object_type(self):
        q, request, response, _triples, expected = CASES[4]
        _, turns, rows = source_case(q, request, response, [["assistant", "recommended", "Birch House"]])
        self.assertEqual(certificate(q, turns, rows)["answer"], expected)

    def test_another_actors_speech_is_not_the_assistants_recommendation(self):
        q, request, _response, _triples, _expected = CASES[4]
        _, turns, rows = source_case(q, request, "Mira recommended Birch House for steamed buns.",
                                    [["assistant", "recommended cafe", "Birch House"]])
        self.assertNotEqual(certificate(q, turns, rows)["status"], "complete")

    def test_negated_speech_cannot_be_promoted_as_a_positive_recommendation(self):
        q, request, _response, _triples, _expected = CASES[4]
        for response in ("I don't recommend Birch House for steamed buns.",
                         "I recommend that you not use Birch House for steamed buns.",
                         "I would recommend Birch House if it reopened for steamed buns.",
                         "I cannot recommend Birch House for steamed buns."):
            _, turns, rows = source_case(q, request, response, [["assistant", "recommended cafe", "Birch House"]])
            self.assertNotEqual(certificate(q, turns, rows)["status"], "complete")

    def test_phone_owner_cannot_be_borrowed_from_a_request_or_another_sentence(self):
        q, request, _response, _triples, _expected = CASES[1]
        for source, head in (("The Cedar field office can be reached at 555-612-9073.", "Cedar field office"),
                             ("The Alder field office is closed. The Cedar office number is 555-612-9073.", "Alder field office")):
            _, turns, rows = source_case(q, request, source, [[head, "provided phone number", "555-612-9073"]])
            self.assertNotEqual(certificate(q, turns, rows)["status"], "complete")

    def test_named_peer_binding_is_not_relaxed(self):
        q, request, response, triples, _expected = CASES[1]
        _, turns, rows = source_case(q, request, response, triples)
        profile = {**adapter.build_query_profile(q), "actor_binding_enabled": True}
        self.assertEqual(certificate(q, turns, rows, profile=profile)["status"], "not_applicable")


class ProjectionPipelineTests(unittest.TestCase):
    def test_only_a_selected_projection_is_protected_from_all_later_promotions(self):
        q, request, response, triples, expected = CASES[1]
        record, _turns, _rows = source_case(q, request, response, triples)
        def extracted(**kwargs):
            rows, audit = adapter.evidence_rows_without_candidate_extraction(kwargs["selected_memories"])
            for row in rows:
                if row["source_role"] == "assistant":
                    row["triples"] = triples
            return rows, audit
        for field, method in (("pragmos_composable_mode", "evaluate_composable_candidate"),
                              ("pragmos_temporal_contract_mode", "run_temporal_candidate"),
                              ("pragmos_state_mode", "evaluate_state_candidate")):
            args = arguments()
            setattr(args, field, "candidate")
            args.pragmos_composable_protocol = "typed_v4"
            with self.subTest(candidate=field), \
                 patch.object(adapter, "plan_query_with_local_slm", return_value=semantic_plan(q, "COLLECT")), \
                 patch.object(adapter, "extract_candidates_from_selected_evidence", side_effect=extracted), \
                 patch.object(adapter, method, wraps=getattr(adapter, method)) as candidate:
                result = adapter.run_pragmos_record(make_context(), record, q, args, "unlabeled")
                self.assertEqual(result["text"], expected)
                candidate.assert_called_once()
                self.assertTrue(candidate.call_args.kwargs["protected"])

    def test_failed_anchor_coverage_does_not_skip_the_existing_operation(self):
        q, request, response, triples, _expected = CASES[1]
        record, _turns, _rows = source_case(q, request, response, triples)
        original = adapter.actor_bound_evidence_coverage
        def incomplete(*args, **kwargs):
            return {**original(*args, **kwargs), "complete": False}
        def extracted(**kwargs):
            rows, audit = adapter.evidence_rows_without_candidate_extraction(kwargs["selected_memories"])
            for row in rows:
                if row["source_role"] == "assistant":
                    row["triples"] = triples
            return rows, audit
        with patch.object(adapter, "plan_query_with_local_slm", return_value=semantic_plan(q, "COLLECT")), \
             patch.object(adapter, "actor_bound_evidence_coverage", side_effect=incomplete), \
             patch.object(adapter, "extract_candidates_from_selected_evidence", side_effect=extracted), \
             patch.object(adapter, "execute_operation_plan", wraps=adapter.execute_operation_plan) as execute:
            result = adapter.run_pragmos_record(make_context(), record, q, arguments(), "unlabeled")
        execute.assert_called_once()
        self.assertFalse(result["execution_diagnostics"]["source_projection"]["selected"])
        self.assertNotEqual(result["answer_decision"], "validated_source_projection")

    def test_wrong_collect_plan_cannot_overrule_a_certified_source_read(self):
        for question, request, response, triples, expected in CASES:
            record, _turns, _rows = source_case(question, request, response, triples)
            def extracted(_context=None, **kwargs):
                rows, diagnostics = adapter.evidence_rows_without_candidate_extraction(kwargs["selected_memories"])
                for row in rows:
                    if row["source_role"] == "assistant":
                        row["triples"] = triples
                return rows, diagnostics
            args = arguments()
            args.pragmos_composable_mode = "candidate"
            args.pragmos_composable_protocol = "typed_v4"
            with self.subTest(question=question), \
                 patch.object(adapter, "plan_query_with_local_slm", return_value=semantic_plan(question, "COLLECT")), \
                 patch.object(adapter, "extract_candidates_from_selected_evidence", side_effect=extracted), \
                 patch.object(adapter, "execute_operation_plan", side_effect=AssertionError("obsolete collection ran")), \
                 patch.object(adapter, "execute_frame_operation_proof", side_effect=AssertionError("obsolete frames ran")):
                result = adapter.run_pragmos_record(make_context(), record, question, args, "opaque")
                self.assertEqual(result["text"], expected)
                self.assertEqual(result["answer_decision"], "validated_source_projection")
                self.assertFalse(result["safe_abstention"])
                self.assertIsNone(result["abstention_reason"])
                self.assertTrue(result["execution_diagnostics"]["source_projection"]["selected"])
                self.assertEqual(result["llm_call_counts"]["evidence_frame_compiler"], 0)
                self.assertEqual(result["llm_call_counts"]["final_generation"], 0)
                self.assertEqual(result["composable_execution_result"]["status"], "NOT_APPLICABLE")
                json.dumps(result)

    def test_unaffected_routes_match_the_checkpoint_in_all_six_families(self):
        revision = "5206a245533835e90b65e78743f75d528dc2ca7d"
        try:
            source = subprocess.check_output(["git", "show", revision + ":PRAGMOS_benchmark_LongMemEval.py"],
                                             cwd=Path(__file__).parent, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.skipTest("Checkpoint Git object is unavailable in this clone")
        old = types.ModuleType("_projection_checkpoint")
        old.__file__ = str(Path(adapter.__file__))
        sys.modules[old.__name__] = old
        exec(compile(source, old.__file__, "exec"), old.__dict__)
        for name, question, operator, sessions in fixtures():
            runs = []
            for module in (old, adapter):
                context, args = make_context(), arguments()
                record = {"haystack_sessions": sessions,
                          "haystack_dates": ["2027-01-03"] * len(sessions),
                          "haystack_session_ids": [str(i) for i in range(len(sessions))]}
                with patch.object(module, "plan_query_with_local_slm", return_value=semantic_plan(question, operator)):
                    result = module.run_pragmos_record(context, record, question, args, "unlabeled")
                runs.append((result, context))
            (before, old_context), (after, new_context) = runs
            with self.subTest(family=name):
                self.assertEqual(semantic_trace(before), semantic_trace(after))
                self.assertEqual(old_context.llm.calls, new_context.llm.calls)
                self.assertEqual(old_context.llm.state, new_context.llm.state)
                self.assertEqual(old_context.vector_memory, new_context.vector_memory)
                self.assertEqual(old_context.alias_to_entity_id, new_context.alias_to_entity_id)
                self.assertEqual(list(old_context.G.edges(data=True, keys=True)),
                                 list(new_context.G.edges(data=True, keys=True)))


class BoundedProjectionRepairTests(unittest.TestCase):
    def test_already_available_answer_keeps_original_repair_schedule(self):
        q, request, response, _triples, _expected = CASES[4]
        _, turns, rows = source_case(q, request, response)
        memories = [{"source_turn_ids": [30 + i], "source_session_id": "noise", "role": "assistant",
                     "source_quote": "Unrelated planning note."} for i in range(3)]
        memories += [{**row, "source_turn_ids": [row["source_turn_id"]]} for row in rows]
        context, calls = make_context(), []
        def model(prompt, **kwargs):
            calls.append(prompt)
            return {"choices": [{"text": "E1 | incomplete\nE2 | incomplete\nE3 | incomplete\n"
                "E5 | assistant | recommended | Birch House\nE5 | malformed"}]}
        context.llm = model
        _, diagnostics = adapter.extract_candidates_from_selected_evidence(
            context, q, "specific fact requested", {"operation": "collect_distinct"}, memories,
            query_intent=adapter.infer_query_intent(q), turn_lookup={t.turn_id: t for t in turns},
            prioritize_source_projection=True)
        self.assertEqual(diagnostics["repair_evidence_ids"], ["E1", "E2"])
        self.assertTrue(all("Repair the incomplete relation extraction" not in prompt for prompt in calls))

    def test_bound_incomplete_source_precedes_noise_without_expanding_budget(self):
        q, request, response, _triples, _expected = CASES[4]
        record, turns, rows = source_case(q, request, response)
        memories = [{"source_turn_ids": [index + 30], "source_session_id": "noise",
                     "role": "assistant", "source_quote": "An unrelated planning note."} for index in range(3)]
        memories += [{**row, "source_turn_ids": [row["source_turn_id"]]} for row in rows]
        context = make_context()
        calls = []
        def model(prompt, **kwargs):
            calls.append(prompt)
            if "Repair the incomplete relation extraction" in prompt:
                return {"choices": [{"text": "E5 | speaker assistant | recommended cafe | Birch House\nEVIDENCE_COMPLETE | E5"}]}
            return {"choices": [{"text": "E1 | incomplete\nE2 | incomplete\nE3 | incomplete\nE5 | Birch House | recommended cafe"}]}
        context.llm = model
        extracted, diagnostics = adapter.extract_candidates_from_selected_evidence(
            context, q, adapter.infer_requested_answer_slot(q), {"operation": "collect_distinct"}, memories,
            query_intent=adapter.infer_query_intent(q), turn_lookup={t.turn_id: t for t in turns},
            prioritize_source_projection=True)
        self.assertEqual(diagnostics["repair_evidence_ids"], ["E5", "E1"])
        self.assertLessEqual(len(diagnostics["repair_evidence_ids"]), diagnostics["max_repair_calls"])
        self.assertIn(["speaker assistant", "recommended cafe", "Birch House"], extracted[-1]["triples"])
        self.assertEqual(extracted[0]["triples"], [])
        self.assertIn("[REQUEST_CONTEXT", calls[1])
        proof = certificate(q, turns, extracted[-2:])
        self.assertEqual(proof["answer"], "Birch House")

    def test_existing_retry_order_is_unchanged_without_an_eligible_read(self):
        context = make_context()
        context.llm = lambda *a, **k: {"choices": [{"text": "E1 | incomplete\nE2 | incomplete\nE3 | incomplete"}]}
        memories = [{"source_turn_ids": [i], "source_session_id": "s", "role": "user",
                     "source_quote": "I spent $17 on equipment."} for i in range(1, 4)]
        _, diagnostics = adapter.extract_candidates_from_selected_evidence(
            context, "What total did I spend on equipment?", "exact quantity or amount", {"operation": "sum"}, memories)
        self.assertEqual(diagnostics["repair_evidence_ids"], ["E1", "E2"])


class SavedProjectionControlsTests(unittest.TestCase):
    def test_saved_assistant_panel_only_admits_the_four_existing_source_projections(self):
        root = Path("/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/assistant_memory_30_v1")
        dataset = root / "pragmos_assistant_memory_validation_30_v1.json"
        trace = root / "runs/baseline_pre_cross_family_v1/pragmos_assistant_memory_30_baseline_pre_cross_family_v1_trace.jsonl"
        if not dataset.is_file() or not trace.is_file():
            self.skipTest("Local saved assistant panel is unavailable")
        records = {r["question_id"]: r for r in json.loads(dataset.read_text())}
        admitted = {}
        for row in map(json.loads, trace.read_text().splitlines()):
            record = records[row["question_id"]]
            turns = adapter.longmemeval_turns(make_context(), adapter.label_blind_inference_record(record))
            result = adapter.resolve_source_projection(
                record["question"], row["requested_answer_slot"], row["query_profile"], row["query_intent"],
                row["inferred_operation_plan"], row["operation_plan"], row["state_selector_plan"],
                {t.turn_id: t for t in turns}, row["ordinal_list_result"], row["typed_answer_result"],
                row["answer_slot_candidates"],
            )
            if result["status"] == "complete":
                admitted[row["question_id"]] = result["answer"]
                self.assertEqual(result["answer"], record["answer"])
        self.assertEqual(set(admitted), {"synthetic_v1_0444", "synthetic_v1_0447",
                                        "synthetic_v1_0448", "synthetic_v1_0449"})

    def test_saved_multisession20_retains_the_existing_route(self):
        root = Path("/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2")
        trace = root / "execution_latency_v2/runs/proof_first/pragmos_diagnostics20_proof_first_candidate_v2_trace.jsonl"
        if not trace.is_file():
            self.skipTest("Local saved multi-session panel is unavailable")
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len(rows), 20)
        for row in rows:
            self.assertFalse(adapter.source_projection_eligible(
                row["question"], row["requested_answer_slot"], row["query_intent"], row["query_profile"],
                row["inferred_operation_plan"], row["operation_plan"], row["state_selector_plan"],
            ))


if __name__ == "__main__":
    unittest.main()
