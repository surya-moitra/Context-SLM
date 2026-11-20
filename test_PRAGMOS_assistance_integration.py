"""Default-off parity, guarded route isolation and exact source interpretation."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import PRAGMOS_assistance_contracts as contracts
import PRAGMOS_assistance_context as assistance
import PRAGMOS_benchmark_LongMemEval as adapter
import PRAGMOS_assistance_runner as runner
import PRAGMOS_benchmark_LoCoMo as locomo
from PRAGMOS_assistance_boundary import ObservationIsolationError
from PRAGMOS_assistance_runner import run_advice, source_rows
from PRAGMOS_execution import validate_execution_arguments
from test_PRAGMOS_assistance_context import AssistanceModel, RankingTokenizer, memory_snapshot, source_context
from test_PRAGMOS_composable_integration import arguments, fixtures, make_context, semantic_plan, semantic_trace


class ContractModel(AssistanceModel):
    def __init__(self, *, action="create_advice", source_mode="reported", text=None, finish="stop", error=None):
        super().__init__(answer="A source-compatible suggestion.", error=error)
        self.action, self.source_mode, self.text, self.finish = action, source_mode, text, finish

    def __call__(self, prompt, **kwargs):
        output = super().__call__(prompt, **kwargs)
        if self.text is not None:
            output["choices"][0].update(text=self.text, finish_reason=self.finish)
            return output
        if prompt.startswith("<|system|>\n" + contracts.REQUEST_INSTRUCTION):
            data = json.loads(prompt.split("<|user|>\n", 1)[1].split("<|end|>", 1)[0])
            output["choices"][0]["text"] = json.dumps({"request_quote": data["request"], "action": self.action})
        elif prompt.startswith("<|system|>\n" + contracts.SOURCE_INSTRUCTION):
            data = json.loads(prompt.split("<|user|>\n", 1)[1].split("<|end|>", 1)[0])
            output["choices"][0]["text"] = json.dumps({"interpretation": "Explicit fixture assessment, not semantic evaluation.",
                "disclosure_quote": data["past_message"][:100], "source_mode": self.source_mode})
        return output


def layer_fixture(model=None):
    layer = make_context()
    layer.llm = model or ContractModel()
    layer.local_reranker.tokenizer = RankingTokenizer()
    layer.local_reranker.max_length = 512
    return layer


class InterpretationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(contracts, "_grammar", return_value=object()))

    def request(self, model):
        return contracts.interpret_request(SimpleNamespace(llm=model), "Suggest an activity.")

    def test_request_authorizes_only_new_advice_with_literal_clause_and_restored_state(self):
        for action in ("create_advice", "retrieve_record", "derive_fact", "uncertain"):
            model = ContractModel(action=action)
            report = self.request(model)
            self.assertEqual(report["advice_candidate"], action == "create_advice")
            self.assertEqual(model.state, 37)
            self.assertEqual(report["model_calls"], 1)
            self.assertEqual(report["isolation"], "model_state_restored")
        report = self.request(ContractModel(text='{"request_quote":"Invented question","action":"create_advice"}'))
        self.assertFalse(report["advice_candidate"])

    def test_invalid_truncated_duplicate_and_wrong_typed_outputs_do_not_route(self):
        for text in ('null', '{', '{"request_quote":"Suggest an activity.","action":true}',
                     '{"request_quote":"Suggest an activity.","action":"create_advice","action":"derive_fact"}',
                     '{"request_quote":"Suggest an activity.","action":"create_advice","extra":1}'):
            with self.subTest(text=text):
                self.assertFalse(self.request(ContractModel(text=text))["advice_candidate"])
        self.assertFalse(self.request(ContractModel(text='{"request_quote":"Suggest an activity.","action":"create_advice"}',
                                                  finish="length"))["advice_candidate"])

    def test_source_modes_require_reported_and_literal_source_not_generated_summary(self):
        context, _ = source_context(["I enjoy drawing. I avoid crowded places."])
        request = assistance.AssistanceRequest("Suggest an activity.")
        bundles, _ = assistance.build_circumstance_bundles(context.vector_memory, context.raw_source_turns, request)
        for mode in ("reported", "imagined", "quoted", "task", "uncertain"):
            context.llm = ContractModel(source_mode=mode)
            report = contracts.interpret_source(context, request, bundles[0])
            self.assertEqual(report["disclosed_candidate"], mode == "reported")
            self.assertEqual(context.llm.state, 37)
        context.llm = ContractModel(text=json.dumps({"interpretation": "I enjoy drawing.",
            "disclosure_quote": "I enjoy cycling.", "source_mode": "reported"}))
        self.assertFalse(contracts.interpret_source(context, request, bundles[0])["disclosed_candidate"])

    def test_interpretation_never_clips_inputs_or_uses_external_cache(self):
        model = ContractModel()
        report = contracts.interpret_request(SimpleNamespace(llm=model), "word " * 3000)
        self.assertEqual(report["model_calls"], 0)
        self.assertIn("budget", report["reason"])
        model.cache = {}
        self.assertEqual(self.request(model)["model_calls"], 0)

    def test_errors_interrupts_and_restore_failures_are_isolated(self):
        model = ContractModel(error=RuntimeError("fixture"))
        self.assertFalse(self.request(model)["advice_candidate"])
        self.assertEqual(model.state, 37)
        model = ContractModel(error=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.request(model)
        self.assertEqual(model.state, 37)
        model = ContractModel()
        model.load_state = Mock(side_effect=RuntimeError("cannot restore"))
        with self.assertRaises(ObservationIsolationError):
            self.request(model)

    def test_interpreted_helper_excludes_nonpersonal_sources_without_model_generation(self):
        for mode in ("imagined", "quoted", "task", "uncertain"):
            context, _ = source_context(["A fixture source."])
            context.llm = ContractModel(source_mode=mode)
            before = memory_snapshot(context)
            result = assistance.run_isolated_assistance(context, assistance.AssistanceRequest("Suggest an activity."),
                                                        source_policy="interpreted", min_score=0)
            self.assertEqual(result["status"], "no_disclosed_personal_context")
            self.assertEqual(result["model_calls"], {"generation": 0, "source_interpretation": 1, "total": 1})
            self.assertEqual(memory_snapshot(context), before)
            self.assertEqual(context.llm.state, 37)

    def test_accepted_sources_stay_whole_while_generated_interpretation_is_not_answer_evidence(self):
        context, _ = source_context(["I enjoy drawing. I avoid crowded places."])
        context.llm = ContractModel()
        result = assistance.run_isolated_assistance(context, assistance.AssistanceRequest("Suggest an activity."),
                                                    source_policy="interpreted", min_score=0)
        self.assertIn("I avoid crowded places.", result["answer_prompt"])
        self.assertNotIn("Explicit fixture assessment", result["answer_prompt"])
        self.assertEqual(result["model_calls"]["total"], 2)
        self.assertLessEqual(result["answer_packing"]["prompt_tokens"] + 64 + 16, 2048)

    def test_source_cap_and_uncertain_qualifications_never_leave_half_an_atomic_state(self):
        context, _ = source_context(["I own a camera.", "I sold the camera."])
        context.llm = ContractModel()
        request = assistance.AssistanceRequest("Suggest an accessory.")
        bundles, _ = assistance.build_circumstance_bundles(context.vector_memory, context.raw_source_turns, request)
        result = {"model_calls": {"total": 0}}
        with patch.object(assistance, "MAX_SOURCE_INTERPRETATIONS", 1):
            selected = assistance._interpret_sources(context, request, bundles, [[b.bundle_id for b in bundles]],
                                                     result, n_ctx=2048, seed=42)
        self.assertFalse(selected)
        self.assertEqual(result["model_calls"]["total"], 1)
        self.assertTrue(result["source_interpretation"]["blocked_atomic_groups"])

    def test_inner_restoration_failure_is_not_swallowed_by_the_helper(self):
        context, _ = source_context(["I own a camera."])
        context.llm = ContractModel()
        context.llm.load_state = Mock(side_effect=RuntimeError("cannot restore"))
        with self.assertRaises(ObservationIsolationError):
            assistance.run_isolated_assistance(context, assistance.AssistanceRequest("Suggest an accessory."),
                                               source_policy="interpreted", min_score=0)

    def test_source_interpretations_reuse_only_identical_sources_for_the_same_question(self):
        context, _ = source_context(["I enjoy drawing."])
        context.llm = ContractModel()
        cache = {}
        request = assistance.AssistanceRequest("Suggest an activity.")
        first = assistance.run_isolated_assistance(context, request, source_policy="interpreted", min_score=0,
                                                   interpretation_cache=cache)
        second = assistance.run_isolated_assistance(context, request, source_policy="interpreted", min_score=0,
                                                    interpretation_cache=cache, max_interpretation_calls=0)
        self.assertEqual(first["model_calls"]["source_interpretation"], 1)
        self.assertEqual(second["model_calls"]["source_interpretation"], 0)
        third = assistance.run_isolated_assistance(context, assistance.AssistanceRequest("Suggest supplies."),
                source_policy="interpreted", min_score=0, interpretation_cache=cache)
        self.assertEqual(third["model_calls"]["source_interpretation"], 1)

    def test_search_payload_is_bounded_and_is_not_an_asserted_personal_fact(self):
        for payload in ({"queries": []}, {"queries": ["x"] * 3}, {"queries": [True]},
                        {"queries": ["x" * 201]}, {"queries": ["a"], "answer": "invented"}):
            result = contracts.expand_memory_search(SimpleNamespace(llm=ContractModel(text=json.dumps(payload))), "Suggest an activity.")
            self.assertEqual(result["status"], "unresolved")
        result = contracts.expand_memory_search(SimpleNamespace(llm=ContractModel(text='{"queries":["earlier circumstances"]}')),
                                                "Suggest an activity.")
        self.assertEqual(result["status"], "interpreted")
        self.assertEqual(result["model_calls"], 1)


class AdviceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(contracts, "_grammar", return_value=object()))

    def run_case(self, module, case, *, mode="off", action="derive_fact", extra_record=None):
        name, question, operator, sessions = case
        layer, args = layer_fixture(ContractModel(action=action)), arguments()
        args.pragmos_assistance_mode = mode
        record = {"question_date": "2027-02-01", "haystack_sessions": sessions,
            "haystack_dates": [f"2027-01-0{i + 3}" for i in range(len(sessions))],
            "haystack_session_ids": [f"s{i}" for i in range(len(sessions))], **(extra_record or {})}
        with patch.object(module, "plan_query_with_local_slm", return_value=semantic_plan(question, operator)):
            result = module.run_pragmos_record(layer, record, question, args, name)
        return result, layer

    def test_off_preserves_all_six_families_against_exact_prechange_adapter(self):
        path = Path("/tmp/PRAGMOS_pre_assistance_integration_LongMemEval.py")
        if not path.exists():
            self.skipTest("Exact pre-integration source snapshot unavailable")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "c82bac259e4b257b356cb306fe5f893b80339ceaba52bd37b6fd706534ae977c")
        spec = importlib.util.spec_from_file_location("_pre_advice_integration", path)
        before = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before)
        for case in fixtures():
            with self.subTest(family=case[0]):
                old, a = self.run_case(before, case)
                new, b = self.run_case(adapter, case)
                self.assertEqual(semantic_trace(new), semantic_trace(old))
                self.assertEqual(a.llm.calls, b.llm.calls)
                self.assertEqual(memory_snapshot(a), memory_snapshot(b))

    def test_guarded_recall_compute_and_uncertain_use_identical_legacy_path_in_six_families(self):
        for case in fixtures():
            for action in ("derive_fact", "retrieve_record", "uncertain"):
                with self.subTest(family=case[0], action=action):
                    old, a = self.run_case(adapter, case)
                    new, b = self.run_case(adapter, case, mode="guarded", action=action)
                    new["execution_diagnostics"].pop("assistance_route")
                    self.assertEqual(new["llm_call_counts"].pop("assistance_request"), 1)
                    new["llm_call_counts"]["total"] -= 1
                    self.assertEqual(semantic_trace(new), semantic_trace(old))
                    self.assertEqual(memory_snapshot(a), memory_snapshot(b))
                    self.assertEqual(a.llm.state, b.llm.state)
                    self.assertEqual(a.llm.calls, [c for c in b.llm.calls if contracts.REQUEST_INSTRUCTION not in c[0]])

    def test_advice_runs_real_helper_and_records_actual_calls_without_gold_or_category(self):
        case = ("arbitrary-record", "Suggest an activity.", "LOOKUP", [[{"role": "user", "content": "I enjoy drawing."}]])
        result, layer = self.run_case(adapter, case, mode="guarded", action="create_advice",
                                      extra_record={"answer": "secret-gold", "question_type": "secret-category"})
        self.assertEqual(result["answer_decision"], "contextual_assistance")
        report = result["execution_diagnostics"]["assistance_route"]["assistance"]
        self.assertEqual(report["invocation"], "opt_in_guarded_advice_route")
        self.assertTrue(report["automatic_routing_enabled"])
        self.assertEqual(report["interpretation_policy"], "source_mode_interpretation_with_literal_quote_validation")
        self.assertEqual(result["llm_call_counts"]["total"], 3)
        self.assertEqual(result["llm_call_counts"]["graph_materialization"], 0)
        self.assertEqual(len(result["final_memories"]), 1)
        self.assertEqual(layer.G.number_of_edges(), 0)
        for prompt, _ in layer.llm.calls:
            self.assertNotIn("secret-gold", prompt)
            self.assertNotIn("secret-category", prompt)
            self.assertNotIn("arbitrary-record", prompt)
        self.assertLessEqual(result["prompt_tokens_estimate"] + 64 + 16, 2048)
        self.assertLessEqual(result["context_tokens_estimate"], result["context_token_budget"])
        json.dumps(result, allow_nan=False)

    def test_successful_first_pass_never_uses_search_retry(self):
        case = ("success", "Suggest an activity.", "LOOKUP", [[{"role": "user", "content": "I enjoy drawing."}]])
        with patch.object(runner, "expand_memory_search", side_effect=AssertionError("successful answer must not be retried")):
            result, _ = self.run_case(adapter, case, mode="guarded", action="create_advice")
        self.assertEqual(result["answer_decision"], "contextual_assistance")

    def test_missing_context_gets_one_bounded_retry_with_reused_cache_and_full_accounting(self):
        layer, turns = source_context(["I enjoy drawing."])
        layer.llm = ContractModel()
        args = arguments()
        candidate = assistance.run_isolated_assistance(layer, assistance.AssistanceRequest("Suggest an activity."),
                                                      source_policy="interpreted", min_score=0)
        first = {"status": "no_disclosed_personal_context", "model_calls": {"source_interpretation": 1, "generation": 0, "total": 1},
                 "phase_seconds": {"retrieval": .1}}
        expansion = {"status": "interpreted", "payload": {"queries": ["previous personal experiences"]}, "model_calls": 1}
        with patch.object(runner, "run_isolated_assistance", side_effect=[first, candidate]) as attempt, \
             patch.object(runner, "expand_memory_search", return_value=expansion) as expand:
            result, report = runner.run_advice(layer, "Suggest an activity.", args, turns)
        self.assertEqual(attempt.call_count, 2)
        self.assertEqual(expand.call_count, 1)
        calls = attempt.call_args_list
        self.assertIs(calls[0].kwargs["interpretation_cache"], calls[1].kwargs["interpretation_cache"])
        self.assertEqual(calls[1].kwargs["max_interpretation_calls"], assistance.MAX_SOURCE_INTERPRETATIONS - 1)
        self.assertEqual(calls[1].args[1].retrieval_queries, ("previous personal experiences",))
        self.assertEqual(report["model_calls"]["total"], 4)
        self.assertEqual(result["llm_call_counts"]["total"], 4)
        self.assertNotIn("previous personal experiences", report["answer_prompt"])
        json.dumps(report)

    def test_both_adapters_pin_every_active_assistance_source_and_reject_mixed_resume_config(self):
        args = arguments()
        args.pragmos_assistance_mode = "guarded"
        args.model_path, args.data_file = "/tmp/fixture-model", "/tmp/fixture-data"
        def identity(path):
            return {"path": str(path), "sha256": "fixture", "size_bytes": 1}
        for module, build in ((adapter, lambda: adapter.build_run_manifest(args, "fixture", [])),
                              (locomo, lambda: locomo.build_manifest(args, []))):
            with self.subTest(module=module.__name__), patch.object(module, "local_file_identity", side_effect=identity):
                manifest = build()
                self.assertTrue(set(runner.SOURCE_FILES).issubset(manifest["source_identities"]))
                self.assertIn("assistance_protocol", manifest["config"])

    def test_no_personal_context_is_explicit_uncertainty_not_legacy_false_evidence(self):
        case = ("fictional-record", "Suggest an activity.", "LOOKUP", [[{"role": "user", "content": "Write a fictional scene."}]])
        with patch.object(contracts, "interpret_source", return_value={"model_calls": 0, "status": "interpreted", "disclosed_candidate": False}):
            result, _ = self.run_case(adapter, case, mode="guarded", action="create_advice")
        self.assertEqual(result["answer_decision"], "assistance_insufficient_context")
        self.assertEqual(result["final_memories"], [])
        self.assertEqual(result["prompt_tokens_estimate"], 0)
        self.assertEqual(result["generation_seconds"], 0)

    def test_preindexed_peer_sources_are_not_reindexed_and_preserve_provenance(self):
        layer, args = layer_fixture(), arguments()
        args.pragmos_assistance_mode = "guarded"
        turn = layer.create_turn(role="user", speaker="Asha", text="I enjoy drawing.", session_id="peer", timestamp="2027-01-01")
        layer.index_raw_turns([turn])
        before = memory_snapshot(layer)
        layer.index_raw_turns = Mock(side_effect=AssertionError("must not reindex"))
        result = adapter.run_pragmos_record(layer, {}, "Suggest an activity for Asha.", args, "q", prepared_turns=[turn], memory_preindexed=True)
        self.assertEqual(memory_snapshot(layer), before)
        self.assertEqual(result["ingestion_seconds"], 0)
        self.assertEqual(result["final_memories"][0]["speaker"], "Asha")

    def test_technical_failure_falls_back_with_attempt_calls_accounted(self):
        case = ("fallback", "Suggest an activity.", "LOOKUP", [[{"role": "user", "content": "I enjoy drawing."}]])
        with patch("PRAGMOS_assistance_runner.run_isolated_assistance", return_value={"status": "component_failed",
                 "model_calls": {"source_interpretation": 2, "generation": 1, "total": 3}}):
            result, _ = self.run_case(adapter, case, mode="guarded", action="create_advice")
        self.assertNotEqual(result["answer_decision"], "contextual_assistance")
        self.assertEqual(result["llm_call_counts"]["assistance_attempt_generation"], 1)
        self.assertEqual(result["llm_call_counts"]["assistance_source_interpretation"], 2)
        self.assertEqual(result["llm_call_counts"]["total"], sum(v for k, v in result["llm_call_counts"].items() if k != "total"))

    def test_configuration_defaults_off_and_rejects_unsupported_mixes(self):
        self.assertEqual(arguments().pragmos_assistance_mode, "off")
        for field, value in (("mode", "raw_phi3_haystack"), ("pragmos_assistance_boundary", "observe"),
                             ("pragmos_ablate", ["graph"]), ("temperature", .7)):
            args = arguments()
            args.pragmos_assistance_mode = "guarded"
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_execution_arguments(args)


@unittest.skipUnless(os.environ.get("PRAGMOS_ASSISTANCE_INTEGRATION_NATIVE_MODEL"), "Opt-in native guarded-route controls")
class NativeGuardedAssistanceTests(unittest.TestCase):
    def test_native_request_routing_sources_and_state_isolation(self):
        from PRAGMOS_context_layer_org import ContextLayer
        from probe_PRAGMOS_assistance_context import file_identity, memory_fingerprint
        model_path = os.environ["PRAGMOS_ASSISTANCE_INTEGRATION_NATIVE_MODEL"]
        layer = ContextLayer(model_path=model_path, context_length=2048, n_threads=8, n_gpu_layers=40,
                             seed=42, retrieval_device="cpu")
        audit = []
        try:
            layer.get_local_reranker()
            layer.llm("<|user|>\nReply OK.<|end|>\n<|assistant|>\n", max_tokens=4, temperature=0, seed=42)
            warm = layer.llm.save_state()
            control_prompt = "<|user|>\nThe studio door is green. What color is the door? Return only the color.<|end|>\n<|assistant|>\n"
            settings = dict(max_tokens=8, temperature=0, seed=42, stop=["<|end|>"])
            control = layer.llm(control_prompt, **settings)
            layer.llm.load_state(warm)

            def verify_state():
                self.assertEqual(layer.llm.n_tokens, warm.n_tokens)
                self.assertEqual(layer.llm._seed, warm.seed)
                np.testing.assert_array_equal(layer.llm.input_ids[:warm.n_tokens], warm.input_ids[:warm.n_tokens])
                np.testing.assert_array_equal(layer.llm._scores[warm.n_tokens - 1], warm.scores[warm.n_tokens - 1])
                self.assertEqual(layer.llm(control_prompt, **settings)["choices"], control["choices"])
                layer.llm.load_state(warm)

            questions = [("What accessories would make sense for my camera setup?", True),
                ("I've been looking back through our earlier conversations about party drinks. Any thoughts on a drink for the small gathering?", True),
                ("What could I do during my commute?", True),
                ("What kind of hotel should I look for on my next break?", True),
                ("What color is my studio door?", False),
                ("What was the second item you recommended for my studio?", False),
                ("I'm trying to put together the details we discussed about Larch field course and wanted to check something. Which item in your Larch field course comparison was both Tern-compatible and weather-resistant?", False),
                ("What was my original laptop model?", False),
                ("What total did I spend on studio supplies?", False),
                ("When did I complete the studio project?", False),
                ("Which kind of food did I say I prefer?", False)]
            for question, expected in questions:
                with self.subTest(question=question):
                    report = contracts.interpret_request(layer, question)
                    self.assertEqual(report["advice_candidate"], expected)
                    verify_state()
                    audit.append({"scope": "request_control", "question": question, "expected_advice": expected, "report": report})
            cases = [("quoted", "Could you suggest a novel for me?", [
                'Here is a novel excerpt: "I only read horror and hate poetry," the detective said. Analyze the narrator.'], False),
                ("imagined", "What activity might I enjoy?", [
                    "Imagine I were a pirate who loved sailing. Write a scene about that fictional version of me."], False),
                ("update", "What could I make for dinner?", [
                    "I have pasta, tomatoes and basil. I like basil.",
                    "The basil is gone and none is available this week. I still have pasta and tomatoes."], True)]
            args = arguments()
            args.pragmos_assistance_mode = "guarded"
            args.pragmos_retrieval_device = "cpu"
            args.pragmos_min_score = 0  # Supplied-source controls, not a retrieval benchmark.
            for name, question, sources, generates in cases:
                with self.subTest(case=name):
                    layer.reset_memory("native-guard-controls")
                    turns = [layer.create_turn(role="user", speaker="user", text=text, session_id="fixture" + str(i),
                        timestamp="2027-01-" + str(i + 1).zfill(2)) for i, text in enumerate(sources)]
                    layer.index_raw_turns(turns)
                    before = memory_fingerprint(layer)
                    result = adapter.run_pragmos_record(layer, {}, question, args, "control",
                        prepared_turns=turns, memory_preindexed=True)
                    self.assertEqual(memory_fingerprint(layer), before)
                    self.assertEqual(result["llm_call_counts"]["final_generation"], int(generates))
                    if generates:
                        self.assertEqual(len(result["final_memories"]), 2)
                        self.assertTrue(result["context_packing_diagnostics"]["atomic_related_groups"])
                    else:
                        self.assertTrue(result["safe_abstention"])
                        self.assertEqual(result["final_memories"], [])
                    verify_state()
                    audit.append({"scope": "provided_sources_not_benchmark_accuracy", "case": name,
                                  "question": question, "sources": sources, "result": result})
            output = os.environ.get("PRAGMOS_ASSISTANCE_INTEGRATION_CONTROL_OUTPUT")
            if output:
                root = Path(__file__).resolve().parent
                payload = {"model": file_identity(model_path), "protocol": runner.protocol_identity(), "semantic_accuracy": None,
                    "code": {name: file_identity(root / name) for name in (*runner.SOURCE_FILES,
                         "PRAGMOS_benchmark_LongMemEval.py", Path(__file__).name)}, "controls": audit}
                with Path(output).open("x", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                    handle.write("\n")
            print("\nNative guarded controls complete: routing, source exclusions and state isolation verified.", flush=True)
        finally:
            layer.llm.close()


if __name__ == "__main__":
    unittest.main()
