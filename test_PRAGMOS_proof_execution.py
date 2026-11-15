import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import PRAGMOS_benchmark_LongMemEval as adapter
from PRAGMOS_evidence_compiler import composable_source_view, parse_semantic_plan
from PRAGMOS_execution import validate_execution_arguments
from PRAGMOS_proof_execution import candidate_identity, independent_positive_guard, reuse_candidate
from test_PRAGMOS_composable_integration import arguments, fixtures, make_context, semantic_plan, semantic_trace


def options():
    args = arguments()
    args.pragmos_composable_mode = "candidate"
    args.pragmos_composable_protocol = "typed_v4"
    args.pragmos_proof_first = True
    return args


def positive(sources):
    rows = [{"source": source.as_dict(), "source_id": key, "quote": source.text,
             "status": "completed", "quantity": {"canonical_value": str(amount)},
             "roles": {"label": "studio supplies"}}
            for (key, source), amount in zip(sources.items(), (17, 29))]
    return {"status": "COMPLETE", "answer": "$46", "extraction_complete": True,
            "isolation": "llama_state_restored_no_external_cache", "selected_engine": "slots_v3",
            "proof": {"executable": True, "provenance": rows},
            "calls": {"planner_calls": 1, "extractor_calls": 2}}


class ProofExecutionTests(unittest.TestCase):
    def record(self):
        return {"question_date": "2027-02-01", "haystack_sessions": [
            [{"role": "user", "content": f"I spent ${n} on studio supplies."}] for n in (17, 29)],
            "haystack_dates": ["2027-01-03", "2027-01-04"], "haystack_session_ids": ["s0", "s1"]}

    def run_record(self, args, runner):
        question = "What total did I spend on studio supplies?"
        context = make_context()
        with patch.object(adapter, "plan_query_with_local_slm", return_value=semantic_plan(question, "SUM")), \
             patch("PRAGMOS_composable_candidate.run_typed_candidate", side_effect=runner):
            return adapter.run_pragmos_record(context, self.record(), question, args, "arbitrary-id"), context

    def test_early_answer_skips_duplicate_models_and_preserves_trace_schema(self):
        def runner(context, question, sources, **kwargs):
            return positive(sources)
        with patch.object(adapter, "extract_candidates_from_selected_evidence", side_effect=AssertionError("duplicate extraction")), \
             patch.object(adapter, "execute_operation_plan", side_effect=AssertionError("duplicate operation")):
            fast, _ = self.run_record(options(), runner)
        old_args = options()
        old_args.pragmos_proof_first = False
        old, _ = self.run_record(old_args, runner)
        self.assertEqual(set(fast), set(old))
        self.assertEqual(fast["text"], "$46")
        self.assertEqual(fast["text"], old["text"])
        self.assertIsNone(fast["baseline_prediction"])
        self.assertTrue(fast["execution_diagnostics"]["proof_first"]["accepted"])
        for key in ("candidate_extraction", "operation_selector", "evidence_frame_compiler", "final_generation"):
            self.assertEqual(fast["llm_call_counts"][key], 0)
        self.assertEqual(fast["candidate_extraction_seconds"], 0)
        json.dumps(fast)

    def test_unproven_candidate_reused_after_identical_legacy_fallback(self):
        seen = []
        def runner(context, question, sources, **kwargs):
            seen.append(context.llm.state)
            return {"status": "NOT_PROVEN", "answer": None, "selected_engine": "typed_v4",
                    "calls": {"planner_calls": 1, "extractor_calls": 2}}
        args = options()
        result, _ = self.run_record(args, runner)
        self.assertEqual(len(seen), 1)
        self.assertTrue(result["execution_diagnostics"]["proof_first"]["candidate_reused_after_fallback"])
        self.assertNotEqual(result["answer_decision"], "composable_evidence_execution")
        self.assertEqual(result["llm_call_counts"]["composable_planner"], 1)

    def test_configuration_cannot_accidentally_run_shadow_or_other_experiments(self):
        validate_execution_arguments(options())
        for key, value in (("mode", "raw_phi3_haystack"), ("pragmos_composable_mode", "shadow"),
                           ("pragmos_composable_protocol", "slots_v3"),
                           ("pragmos_evidence_compiler_mode", "shadow"),
                           ("pragmos_temporal_contract_mode", "candidate"),
                           ("pragmos_state_mode", "candidate"), ("pragmos_overlap_planner", True)):
            args = options()
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_execution_arguments(args)

    def test_specialized_paths_keep_calls_state_graphs_and_answers(self):
        for name, question, operator, sessions in fixtures():
            if name == "multisession":
                continue
            record = {"question_date": "2027-02-01", "haystack_sessions": sessions,
                      "haystack_dates": [f"2027-01-0{i + 3}" for i in range(len(sessions))],
                      "haystack_session_ids": [f"s{i}" for i in range(len(sessions))]}
            runs = []
            for enabled in (False, True):
                context, args = make_context(), options()
                args.pragmos_proof_first = enabled
                with patch.object(adapter, "plan_query_with_local_slm", return_value=semantic_plan(question, operator)), \
                     patch("PRAGMOS_composable_candidate.run_typed_candidate", side_effect=lambda *a, **kw: {
                         "status": "NOT_PROVEN", "answer": None,
                         "calls": {"planner_calls": 0, "extractor_calls": 0}}):
                    result = adapter.run_pragmos_record(context, record, question, args, name)
                runs.append((result, context))
            old, new = runs
            with self.subTest(family=name):
                for key in ("text", "answer_decision", "final_memories", "operation_result"):
                    self.assertEqual(semantic_trace(old[0][key]), semantic_trace(new[0][key]))
                self.assertEqual(old[1].llm.calls, new[1].llm.calls)
                self.assertEqual(old[1].llm.state, new[1].llm.state)
                self.assertEqual(old[1].vector_memory, new[1].vector_memory)
                self.assertEqual(list(old[1].G.edges(data=True, keys=True)), list(new[1].G.edges(data=True, keys=True)))

    def test_cache_identity_covers_order_text_role_speaker_time_and_contract(self):
        sources, integrity = composable_source_view([{"source_turn_id": "1"}], {
            "1": {"session_id": "s", "text": "A fact", "role": "user", "speaker": "A", "timestamp": "2027-01-01"}})
        def identity(sources=sources, contract=None):
            return candidate_identity("question", sources, integrity, contract or {}, n_ctx=2048, max_tokens=64)
        self.assertEqual(identity(contract={"baseline_answer": "a"}), identity(contract={"baseline_answer": "b"}))
        self.assertNotEqual(identity(), identity(contract={"actor": "B"}))
        from dataclasses import replace
        for key, value in (("text", "changed"), ("role", "assistant"), ("speaker", "B"), ("timestamp", "2028")):
            self.assertNotEqual(identity(), identity({"CS1": replace(sources["CS1"], **{key: value})}))

    def test_numeric_guard_keeps_source_qualification_and_missing_operand_vetoes(self):
        question = "What total did I spend on supplies?"
        plan = semantic_plan(question, "SUM")
        lookup = {"1": {"session_id": "s", "text": "I spent $17 on supplies.",
                        "role": "user", "speaker": "user", "timestamp": "2027-01-01"}}
        sources, _ = composable_source_view([{"source_turn_id": "1"}], lookup)
        report = positive(sources)
        self.assertTrue(independent_positive_guard(report, plan, "sum", lookup)["accepted"])
        report["proof"]["provenance"][0]["source"]["speaker"] = "someone else"
        self.assertFalse(independent_positive_guard(report, plan, "sum", lookup)["accepted"])
        lookup["1"]["text"] = "I spent $17 on supplies, but that amount was never spent."
        sources, _ = composable_source_view([{"source_turn_id": "1"}], lookup)
        report = positive(sources)
        report["proof"]["provenance"][0]["quote"] = "I spent $17 on supplies"
        self.assertFalse(independent_positive_guard(report, plan, "sum", lookup)["accepted"])
        report["proof"]["baseline_veto"] = True
        self.assertFalse(independent_positive_guard(report, plan, "sum", lookup)["accepted"])
        lookup["1"]["text"] = "I spent $17 on supplies."
        sources, _ = composable_source_view([{"source_turn_id": "1"}], lookup)
        plan = parse_semantic_plan({"operator": "SUM", "confidence": "high", "answer_role": "amount",
            "answer_type": "amount", "actor": "user", "operands": [
                {"id": "O1", "text": "supplies", "kind": "quantity", "hard_entities": ["supplies"]},
                {"id": "O2", "text": "travel", "kind": "quantity", "hard_entities": ["travel"]}]}, question)
        guard = independent_positive_guard(positive(sources), plan, "sum", lookup)
        self.assertFalse(guard["accepted"])
        self.assertEqual(guard["source_input_validation"]["missing_operand_ids"], ["O2"])

    def test_baseline_dependent_reexecution_is_pure_and_does_not_mutate_cache(self):
        context = SimpleNamespace(count_tokens=lambda text: 1)
        cached = {"report": {"selected_engine": "typed_v4", "calls": {"planner_calls": 2}},
                  "prepared": {"request": "r", "claims": (), "windows": {}, "sources": {}}}
        proof = {"status": "PARTIAL", "answer": None, "reason": "missing", "coverage": {},
                 "abstention_supported": True}
        with patch("PRAGMOS_typed_computation.execute_typed_plan", return_value=proof) as execute:
            report = reuse_candidate(context, cached, baseline_answer="bad answer", max_answer_tokens=64)
        self.assertTrue(report["supported_abstention"])
        self.assertEqual(report["answer"], "I do not know.")
        self.assertNotIn("answer", cached["report"])
        self.assertEqual(execute.call_args.kwargs["baseline_answer"], "bad answer")

    def test_changed_source_view_invalidates_model_result_reuse(self):
        context, args, cache = make_context(), options(), {}
        question = "What total did I spend on supplies?"
        plan = semantic_plan(question, "SUM")
        lookup = {"1": {"session_id": "s", "text": "I spent $17 on supplies.",
                        "role": "user", "speaker": "user", "timestamp": "2027-01-01"}}
        def evaluate():
            return adapter.evaluate_composable_candidate(
                context, question, args, semantic_plan=plan,
                query_intent={"intent": "recall", "preferred_source_roles": ["user"]},
                source_rows=[{"source_turn_id": "1"}], turn_lookup=lookup,
                execution_cache=cache)
        with patch("PRAGMOS_composable_candidate.run_typed_candidate", side_effect=lambda *a, **kw: {
                "status": "NOT_PROVEN", "calls": {"planner_calls": 1, "extractor_calls": 0}}) as runner:
            evaluate()
            evaluate()
            self.assertEqual(runner.call_count, 1)
            lookup["1"]["speaker"] = "another speaker"
            evaluate()
            self.assertEqual(runner.call_count, 2)

    def test_pure_recheck_error_preserves_generation_fallback(self):
        cache = {"report": {"selected_engine": "typed_v4", "calls": {}, "answer": "old"},
                 "prepared": {"request": "r", "claims": (), "windows": {}, "sources": {}}}
        with patch("PRAGMOS_typed_computation.execute_typed_plan", side_effect=ValueError("unbound")):
            result = reuse_candidate(SimpleNamespace(), cache, baseline_answer="baseline", max_answer_tokens=64)
        self.assertEqual(result["status"], "NOT_PROVEN")
        self.assertIsNone(result["answer"])
        self.assertFalse(result["supported_abstention"])


class SavedDiagnosticSourceTests(unittest.TestCase):
    def test_all_twenty_raw_source_views_match_established_preparation(self):
        root = Path("/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2")
        trace = root / "execution_latency_v1/runs/optimized/pragmos_diagnostics20_latency_optimized_v1_trace.jsonl"
        data = root / "typed_computation_diagnostics_v1/pragmos_multisession_diagnostics_20_v1.json"
        if not trace.is_file() or not data.is_file():
            self.skipTest("Local diagnostic artifacts unavailable")
        records = {row["question_id"]: row for row in json.loads(data.read_text())}
        traces = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len(traces), 20)
        for row in traces:
            record = records[row["question_id"]]
            turns = adapter.longmemeval_turns(make_context(), adapter.label_blind_inference_record(record))
            lookup = {turn.turn_id: turn for turn in turns}
            plan = parse_semantic_plan(json.loads(row["semantic_query_plan"]["raw_output"]), record["question"])
            raw = adapter.selected_evidence_records(row["selected_candidate_memories"])
            views = []
            for carrier in (raw, row["answer_candidate_extractions"]):
                prepared, _, _ = adapter.prepare_compiler_sources(
                    carrier, row["selected_candidate_memories"], row["retrieval_candidate_pool"],
                    semantic_plan=plan, question=record["question"], operation_plan=row["operation_plan"],
                    compiler_active=row["evidence_compiler_active"], temporal_active=False, turn_lookup=lookup)
                sources, integrity = composable_source_view(prepared, lookup)
                views.append(([s.as_dict() for s in sources.values()], integrity))
            with self.subTest(question=row["question_id"]):
                self.assertEqual(views[0], views[1])
                grounding = adapter.actor_bound_evidence_coverage(row["query_profile"], raw)
                self.assertEqual(json.loads(json.dumps(grounding)), row["actor_grounding"])

    def test_saved_positive_model_selections_replay_through_new_boundary(self):
        root = Path("/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2")
        trace = root / "execution_latency_v1/runs/optimized/pragmos_diagnostics20_latency_optimized_v1_trace.jsonl"
        data = root / "typed_computation_diagnostics_v1/pragmos_multisession_diagnostics_20_v1.json"
        if not trace.is_file() or not data.is_file():
            self.skipTest("Local diagnostic artifacts unavailable")
        records = {row["question_id"]: row for row in json.loads(data.read_text())}
        checked = 0
        for row in map(json.loads, trace.read_text().splitlines()):
            previous = row["composable_execution_result"]
            if previous.get("status") != "COMPLETE":
                continue
            record, context = records[row["question_id"]], make_context()
            turns = adapter.longmemeval_turns(context, adapter.label_blind_inference_record(record))
            lookup = {turn.turn_id: turn for turn in turns}
            plan = parse_semantic_plan(json.loads(row["semantic_query_plan"]["raw_output"]), record["question"])
            rows, _, _ = adapter.prepare_compiler_sources(
                adapter.selected_evidence_records(row["selected_candidate_memories"]),
                row["selected_candidate_memories"], row["retrieval_candidate_pool"],
                semantic_plan=plan, question=record["question"], operation_plan=row["operation_plan"],
                compiler_active=row["evidence_compiler_active"], temporal_active=False, turn_lookup=lookup)
            responses = iter(previous["calls"]["responses"])
            def replay(context, prompt, stage, n_ctx, audit, **kwargs):
                response = next(responses)
                self.assertEqual(stage, response["stage"])
                audit[stage + "_calls"] += 1
                return json.loads(response["text"])
            cache = {}
            with self.subTest(question=row["question_id"]), patch("PRAGMOS_composable_candidate._call", side_effect=replay):
                result = adapter.evaluate_composable_candidate(
                    context, record["question"], options(), semantic_plan=plan, query_intent=row["query_intent"],
                    source_rows=rows, turn_lookup=lookup, selector_plan=row["state_selector_plan"],
                    baseline_answer=None, execution_cache=cache)
                self.assertEqual(result.get("status"), previous["status"], result.get("reason"))
                self.assertEqual(result.get("answer"), previous["answer"])
                self.assertEqual(list(responses), [])
                checked += 1
        self.assertEqual(checked, 14)


if __name__ == "__main__":
    unittest.main()
