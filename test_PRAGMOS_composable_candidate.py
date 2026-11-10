import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PRAGMOS_benchmark_LongMemEval import (
    evaluate_composable_candidate, validate_composable_configuration,
)
from PRAGMOS_composable_candidate import (
    compile_model_plan, extraction_prompt, extraction_schema, promotable, run_candidate, source_span_choices,
)
from PRAGMOS_evidence_compiler import composable_source_view, parse_semantic_plan
from PRAGMOS_composable_execution import ContractError, execute_plan
from test_PRAGMOS_composable_execution import data, operand, plan, scan, step, total


class FixtureLLM:
    cache = None

    def __init__(self, responses):
        self.responses = responses
        self.position = 0
        self.calls = []
        self.restored = False

    def save_state(self):
        return self.position

    def load_state(self, state):
        self.position = state
        self.restored = True

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = self.responses[self.position]
        self.position += 1
        return {"choices": [{"text": json.dumps(response), "finish_reason": "stop"}]}


def context(responses):
    return SimpleNamespace(llm=FixtureLLM(responses), count_tokens=lambda text: max(1, len(text) // 5))


def payload(fact):
    return {"source_id": fact.source_id, "quote": fact.quote,
            "actor_span": fact.actor_span, "predicate": fact.predicate,
            "status": fact.status, "kind": fact.kind, "operands": list(fact.operands),
            "roles": dict(fact.roles),
            "measurement": {"text": fact.raw_measurement, "value": fact.raw_value, "unit": fact.raw_unit}}


def args(mode="candidate", **overrides):
    return SimpleNamespace(**{
        "mode": "pragmos_context", "pragmos_composable_mode": mode,
        "pragmos_evidence_compiler_mode": "validated", "pragmos_temporal_contract_mode": "off",
        "pragmos_ablate": [], "n_ctx": 2048, **overrides,
    })


def aggregate_plan():
    return parse_semantic_plan({
        "operator": "SUM", "answer_type": "amount", "answer_role": "amount",
        "target_unit": "money", "actor": "user", "confidence": "high",
        "operands": [{"id": "O1", "text": "supplies", "kind": "quantity", "actor": "user", "hard_entities": []}],
    }, "What total did I spend on supplies?")


class ComposableCandidateTests(unittest.TestCase):
    def run_fixture(self, ctx, sources):
        with patch("PRAGMOS_composable_candidate._grammar", return_value=None):
            return run_candidate(ctx, "What total comes from the supported source facts?", sources, n_ctx=2048)

    def test_model_source_envelope_does_not_expose_internal_identifiers(self):
        from dataclasses import replace
        sources, _ = data([17])
        source = replace(sources["S1"], session_id="private-answer-session-identifier",
                         turn_id="private-question-identifier")
        prompt = extraction_prompt("What total?", plan([scan(), total()]), [source])
        self.assertNotIn(source.session_id, prompt)
        self.assertNotIn(source.turn_id, prompt)
        self.assertIn(source.source_id, prompt)
        self.assertIn(source.text, prompt)

    def test_complete_candidate_with_every_source_reviewed_and_state_restored(self):
        sources, facts = data([17, 38, 29], roles=[{"location": "Cedar"}, {"location": "Birch"}, {"location": "Cedar"}])
        request = plan([scan(), total(keys=["location"]), step("c", "rank", ["b"], column="value", direction="max")], column="location")
        ctx = context([request, {"reviewed": list(sources), "facts": [payload(f) for f in facts]}])
        before = copy.deepcopy((sources, facts))
        result = self.run_fixture(ctx, sources)
        self.assertEqual(result["answer"], "Cedar")
        self.assertTrue(promotable(result))
        self.assertEqual(ctx.llm.position, 0)
        self.assertTrue(ctx.llm.restored)
        self.assertEqual(result["calls"]["planner_calls"], 1)
        self.assertEqual(result["calls"]["extractor_calls"], 1)
        self.assertEqual((sources, facts), before)
        self.assertFalse(result["source_view_complete"])
        self.assertFalse(result["semantic_equivalence_verified"])

    def test_missing_source_review_cannot_certify_partial_total(self):
        sources, facts = data([17, 29])
        ctx = context([plan([scan(), total()]), {"reviewed": ["S1"], "facts": [payload(facts[0])]}])
        result = self.run_fixture(ctx, sources)
        self.assertFalse(promotable(result))
        self.assertEqual(result["reason"], "selected_source_review_incomplete")
        self.assertTrue(ctx.llm.restored)

    def test_invalid_fact_is_not_silently_dropped(self):
        sources, facts = data([17, 29])
        rows = [payload(f) for f in facts]
        rows[1]["measurement"]["value"] = "999"
        ctx = context([plan([scan(), total()]), {"reviewed": list(sources), "facts": rows}])
        result = self.run_fixture(ctx, sources)
        self.assertFalse(promotable(result))
        self.assertEqual(result["reason"], "ungrounded_raw_measurement")

    def test_parser_failure_restores_state_and_does_not_certify_absence(self):
        sources, _ = data([17])
        ctx = context([{"this": "is not a program"}])
        result = self.run_fixture(ctx, sources)
        self.assertEqual(result["status"], "NOT_PROVEN")
        self.assertTrue(ctx.llm.restored)
        self.assertIsNone(result["answer"])

    def test_output_truncation_is_not_a_complete_extraction(self):
        sources, _ = data([17])
        ctx = context([])
        class Truncated(FixtureLLM):
            def __call__(self, *_args, **_kwargs):
                return {"choices": [{"text": "{}", "finish_reason": "length"}]}
        ctx.llm = Truncated([])
        result = self.run_fixture(ctx, sources)
        self.assertEqual(result["reason"], "planner_output_truncated")
        self.assertFalse(promotable(result))

    def test_overflow_is_explicit_not_truncation(self):
        sources, _ = data([17])
        ctx = context([plan([scan(), total()])])
        ctx.count_tokens = lambda _text: 4000
        result = self.run_fixture(ctx, sources)
        self.assertEqual(result["reason"], "planner_context_budget_exceeded")
        self.assertEqual(len(ctx.llm.calls), 0)
        self.assertTrue(ctx.llm.restored)

    def test_non_applicable_planner_preserves_baseline(self):
        sources, _ = data([17])
        ctx = context([{"applicable": False}])
        result = self.run_fixture(ctx, sources)
        self.assertEqual(result["reason"], "not_applicable")
        self.assertEqual(len(ctx.llm.calls), 1)
        self.assertFalse(promotable(result))

    def test_live_calls_require_restorable_model_and_no_external_cache(self):
        sources, _ = data([17])
        for llm in (lambda *_a, **_k: self.fail("model must not run"), FixtureLLM([])):
            if isinstance(llm, FixtureLLM):
                llm.cache = object()
            ctx = SimpleNamespace(llm=llm, count_tokens=lambda text: len(text))
            result = self.run_fixture(ctx, sources)
            self.assertEqual(result["reason"], "isolated_model_state_unavailable")

    def test_restore_failure_cannot_silently_taint_later_queries(self):
        sources, _ = data([17])
        ctx = context([{"applicable": False}])
        def broken_restore(_state):
            raise RuntimeError("restore failed")
        ctx.llm.load_state = broken_restore
        with self.assertRaisesRegex(RuntimeError, "could not restore model state"):
            self.run_fixture(ctx, sources)

    def test_default_off_does_not_touch_context_sources_or_new_engine(self):
        with patch("PRAGMOS_composable_candidate.run_grounded_candidate", side_effect=AssertionError("new model path")), \
             patch("PRAGMOS_evidence_compiler.composable_source_view", side_effect=AssertionError("source view touched")):
            result = evaluate_composable_candidate(None, "any question", args("off"),
                semantic_plan=None, query_intent={}, source_rows=None, turn_lookup=None)
        self.assertIsNone(result)

    def test_specialized_and_protected_paths_do_not_run_new_model(self):
        for intent in ("assistant_memory", "knowledge_update", "preference_recommendation", "evidence_synthesis"):
            with self.subTest(intent=intent):
                result = evaluate_composable_candidate(None, "any question", args(),
                    semantic_plan=aggregate_plan(), query_intent={"intent": intent},
                    source_rows=None, turn_lookup=None)
                self.assertEqual(result["reason"], "protected_existing_answer_path")
        result = evaluate_composable_candidate(None, "any question", args(),
            semantic_plan=aggregate_plan(), query_intent={}, source_rows=None, turn_lookup=None, protected=True)
        self.assertEqual(result["calls"]["planner_calls"], 0)

    def test_incompatible_flags_fail_before_running_a_model(self):
        for overrides in ({"mode": "raw_phi3_haystack"}, {"pragmos_temporal_contract_mode": "shadow"},
                          {"pragmos_evidence_compiler_mode": "off"}):
            with self.assertRaises(ValueError):
                validate_composable_configuration(args(**overrides))

    def test_source_view_recovers_whole_turn_and_never_crosses_sessions(self):
        turns = {"t1": {"session_id": "s1", "role": "user", "speaker": "Mira", "text": "Full source with qualifications."}}
        before = copy.deepcopy(turns)
        sources, diagnostic = composable_source_view([
            {"source_session_id": "s1", "source_turn_id": "t1", "source_quote": "Full source"},
        ], turns)
        self.assertEqual(sources["CS1"].text, turns["t1"]["text"])
        self.assertEqual(sources["CS1"].speaker, "Mira")
        self.assertEqual(diagnostic["unresolved_row_count"], 0)
        wrong, diagnostic = composable_source_view([{"source_session_id": "s2", "source_turn_id": "t1"}], turns)
        self.assertEqual(wrong, {})
        self.assertEqual(diagnostic["unresolved_row_count"], 1)
        self.assertEqual(turns, before)

    def test_both_adapters_fingerprint_candidate_sources_and_configuration(self):
        import PRAGMOS_benchmark_LongMemEval as lme
        import PRAGMOS_benchmark_LoCoMo as locomo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model = root / "data.json", root / "model.gguf"
            dataset.write_text("[]\n")
            model.write_bytes(b"fixture model identity only; never loaded")
            for adapter in (lme, locomo):
                for mode in ("off", "candidate"):
                    with self.subTest(adapter=adapter.__name__, mode=mode), patch.object(sys, "argv", [
                        "fixture", "--data-file", str(dataset), "--model-path", str(model),
                        "--mode", "pragmos_context", "--pragmos-composable-mode", mode,
                    ]):
                        options = adapter.parse_args()
                    manifest = (adapter.build_run_manifest(options, str(dataset), [])
                                if adapter is lme else adapter.build_manifest(options, []))
                    self.assertEqual(manifest["config"]["pragmos_composable_mode"], mode)
                    for name in ("PRAGMOS_composable_execution.py", "PRAGMOS_composable_candidate.py"):
                        self.assertEqual(name in manifest["source_identities"], mode == "candidate")

    def test_original_question_type_and_gold_fields_are_not_candidate_inputs(self):
        import inspect
        self.assertEqual(set(inspect.signature(run_candidate).parameters), {"context", "question", "sources", "n_ctx", "max_answer_tokens", "answer_contract"})
        fields = set(inspect.signature(evaluate_composable_candidate).parameters)
        self.assertTrue(fields.isdisjoint({"question_type", "category", "answer", "answer_session_ids", "question_id", "record"}))

    def test_model_expression_compiles_group_then_rank_without_generated_ids(self):
        sources, facts = data([17, 38, 29], roles=[{"location": "Cedar"}, {"location": "Birch"}, {"location": "Cedar"}])
        request = {"version": 2, "operands": [operand()],
                   "expression": {"op": "argmax", "input": {"op": "group", "field": "location", "aggregate": "sum",
                                  "input": {"op": "scan", "operand": "O1"}}},
                   "output": {"shape": "scalar", "field": "location", "unit": None, "decimals": None}}
        compiled = compile_model_plan(request)
        self.assertEqual([s["op"] for s in compiled["steps"]], ["source", "reduce", "rank"])
        self.assertEqual(execute_plan(compiled, facts, sources)["answer"], "Cedar")

    def test_expression_does_not_treat_program_steps_as_evidence_operands(self):
        request = {"version": 2, "operands": [operand(), operand("O2")],
                   "expression": {"op": "sum", "input": {"op": "scan", "operand": "O1"}},
                   "output": {"shape": "scalar", "field": "value", "unit": None, "decimals": None}}
        with self.assertRaisesRegex(ContractError, "unused_mandatory_operand"):
            compile_model_plan(request)

    def test_expression_field_cannot_be_a_formula_or_invented_alias(self):
        request = {"version": 2, "operands": [operand()],
                   "expression": {"op": "group", "field": "sum(value)", "aggregate": "sum", "input": {"op": "scan", "operand": "O1"}},
                   "output": {"shape": "scalar", "field": "value", "unit": None, "decimals": None}}
        with self.assertRaisesRegex(ContractError, "unknown_expression_field"):
            compile_model_plan(request)

    def test_expression_unsupported_preserves_fallback_not_certified_absence(self):
        with self.assertRaisesRegex(ContractError, "not_applicable"):
            compile_model_plan({"version": 2, "operands": [], "expression": {"op": "unsupported"}, "output": {}})

    def test_actor_and_predicate_decoder_uses_source_spans_not_metadata(self):
        sources, _ = data([17])
        choices = source_span_choices(list(sources.values()))
        self.assertIn("I", choices)
        self.assertIn("paid", choices)
        self.assertNotIn("user", choices)
        schema = extraction_schema(tuple(sources), ("O1",), choices)
        item = schema["properties"]["facts"]["items"]["properties"]
        self.assertEqual(item["actor_span"]["enum"], list(choices))
        self.assertEqual(item["source_id"]["enum"], ["S1"])
        measurement = item["measurement"]["anyOf"][0]["properties"]
        self.assertEqual(set(measurement), {"text"})

    def test_pronoun_and_null_metadata_normalization_does_not_change_sources(self):
        request = {"version": 2, "operands": [{**operand(), "actor": "my"}],
                   "expression": {"op": "sum", "input": {"op": "scan", "operand": "O1"}},
                   "output": {"shape": "scalar", "field": "value", "unit": "null", "decimals": None}}
        before = copy.deepcopy(request)
        result = compile_model_plan(request)
        self.assertEqual(result["operands"][0]["actor"], "user")
        self.assertIsNone(result["output"]["unit"])
        self.assertEqual(request, before)

    def test_amount_cannot_replace_a_requested_entity_projection(self):
        sources, _ = data([17])
        ctx = context([plan([scan(), total()])])
        with patch("PRAGMOS_composable_candidate._grammar", return_value=None):
            result = run_candidate(ctx, "What is the requested source location?", sources, n_ctx=2048,
                                   answer_contract={"answer_role": "location", "result_shape": "scalar"})
        self.assertEqual(result["reason"], "candidate_output_conflicts_with_parent_answer_contract")
        self.assertFalse(promotable(result))
        self.assertEqual(result["calls"]["extractor_calls"], 0)

    def test_collection_cannot_replace_parent_scalar_contract(self):
        sources, _ = data([17])
        request = plan([scan(), total()], shape="value_set")
        ctx = context([request])
        with patch("PRAGMOS_composable_candidate._grammar", return_value=None):
            result = run_candidate(ctx, "What total?", sources, n_ctx=2048,
                                   answer_contract={"answer_role": "amount", "result_shape": "scalar"})
        self.assertEqual(result["reason"], "candidate_shape_conflicts_with_parent_answer_contract")
        self.assertFalse(promotable(result))
        self.assertEqual(result["calls"]["extractor_calls"], 0)
        self.assertTrue(ctx.llm.restored)

    def test_tuple_must_retain_all_parent_roles_in_order(self):
        sources, _ = data([17])
        for columns in (["object"], ["location", "object"], ["object", "object"]):
            with self.subTest(columns=columns):
                request = plan([scan(), total()], columns=columns, shape="tuple_set")
                ctx = context([request])
                with patch("PRAGMOS_composable_candidate._grammar", return_value=None):
                    result = run_candidate(ctx, "Which items and locations?", sources, n_ctx=2048,
                                           answer_contract={"result_shape": "tuple_set", "projection_roles": ["object", "location"]})
                self.assertEqual(result["reason"], "candidate_tuple_conflicts_with_parent_answer_contract")
                self.assertFalse(promotable(result))
                self.assertEqual(result["calls"]["extractor_calls"], 0)
                self.assertTrue(ctx.llm.restored)


if __name__ == "__main__":
    unittest.main()
