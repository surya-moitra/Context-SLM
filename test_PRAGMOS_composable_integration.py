"""Pipeline parity against an explicitly supplied pre-change source snapshot."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

import PRAGMOS_benchmark_LongMemEval as current
from PRAGMOS_context_layer_org import ContextLayer
from PRAGMOS_evidence_compiler import parse_semantic_plan
from test_PRAGMOS_context_layer_latency import cache_only_context


class EmbedderFixture:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            values = np.zeros(384, dtype=np.float32)
            values[:3] = [1, 2, float(sum(map(ord, text)) % 17 + 1)]
            vectors.append(values)
        return np.asarray(vectors)


class ModelFixture:
    cache = None

    def __init__(self):
        self.calls = []
        self.state = 0

    def save_state(self):
        return self.state

    def load_state(self, state):
        self.state = state

    def tokenize(self, text, **_kwargs):
        return list(range(len(text.split())))

    def __call__(self, prompt, **kwargs):
        arguments = {key: (getattr(value, "_grammar", type(value).__name__) if key == "grammar" else value)
                     for key, value in kwargs.items()}
        self.calls.append((prompt, arguments))
        self.state += 1
        return {"choices": [{"text": "Unstructured fixture answer."}]}


def make_context():
    layer = cache_only_context()
    layer.llm = ModelFixture()
    layer.embedder = EmbedderFixture()
    layer.enable_dense_retrieval = True
    layer.enable_lexical_retrieval = True
    layer.enable_graph_retrieval = True
    layer.enable_reranker = True
    layer.context_length = 2048
    layer.answer_token_reserve = 64
    layer.context_token_budget = 1800
    layer.reset_memory(session_id="parity")
    return layer


def arguments():
    with patch.object(sys, "argv", ["fixture", "--data-file", "unused.json", "--mode", "pragmos_context"]):
        args = current.parse_args()
    args.pragmos_answer_context_tokens = 2048
    return args


def fixtures():
    return [
        ("user", "What color is my studio door?", "LOOKUP", [
            [{"role": "user", "content": "My studio door is green."}],
        ]),
        ("assistant", "What was the second item you recommended for my studio?", "LOOKUP", [
            [{"role": "user", "content": "Recommend items for my studio."},
             {"role": "assistant", "content": "For your studio: 1. Lamp 2. Easel 3. Stool"}],
        ]),
        ("updates", "What was my original laptop model?", "LOOKUP", [
            [{"role": "user", "content": "My laptop model is Pine 1."}],
            [{"role": "user", "content": "I replaced Pine 1 with Pine 2."}],
        ]),
        ("preference", "Recommend a studio gift based on my preferences.", "LOOKUP", [
            [{"role": "user", "content": "I enjoy painting and dislike scented gifts."}],
        ]),
        ("multisession", "What total did I spend on studio supplies?", "SUM", [
            [{"role": "user", "content": "I spent $17 on studio supplies."}],
            [{"role": "user", "content": "I spent $29 on studio supplies."}],
        ]),
        ("temporal", "When did I complete the studio project?", "DATE_LOOKUP", [
            [{"role": "user", "content": "I completed the studio project on January 2, 2027."}],
        ]),
    ]


def semantic_plan(question, operator):
    return parse_semantic_plan({
        "operator": operator, "actor": "user", "confidence": "high",
        "answer_type": "amount" if operator == "SUM" else "date" if operator == "DATE_LOOKUP" else "text",
        "answer_role": "amount" if operator == "SUM" else "date" if operator == "DATE_LOOKUP" else "text",
        "target_unit": "money" if operator == "SUM" else None,
        "operands": [{"id": "O1", "text": question, "kind": "quantity" if operator == "SUM" else "fact",
                      "actor": "user", "hard_entities": []}],
    }, question)


def semantic_trace(value):
    if isinstance(value, dict):
        return {key: semantic_trace(item) for key, item in value.items() if not key.endswith("_seconds")}
    if isinstance(value, list):
        return [semantic_trace(item) for item in value]
    return value


class ComposablePipelineParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline = os.environ.get("PRAGMOS_PRE_COMPOSABLE_BASELINE")
        if not baseline:
            raise unittest.SkipTest("Set PRAGMOS_PRE_COMPOSABLE_BASELINE to the frozen pre-change adapter for paired pipeline checks")
        path = Path(baseline)
        expected = "48c3a6c61bdcafc7961f34230ad96139def1c25fe8023c08600901141adc70ee"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError("Not the recorded pre-composable inference baseline")
        spec = importlib.util.spec_from_file_location("_pragmos_pre_composable", path)
        cls.baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.baseline)

    def run_record(self, module, name, question, operator, sessions, *, mode="off", protocol="slots_v3"):
        layer, args = make_context(), arguments()
        args.pragmos_composable_mode = mode
        args.pragmos_composable_protocol = protocol
        record = {"question_date": "2027-02-01", "haystack_sessions": sessions,
                  "haystack_dates": [f"2027-01-0{i + 3}" for i in range(len(sessions))],
                  "haystack_session_ids": [f"s{i}" for i in range(len(sessions))]}
        with patch.object(module, "plan_query_with_local_slm", return_value=semantic_plan(question, operator)):
            result = module.run_pragmos_record(layer, record, question, args, name)
        return result, layer

    def test_default_off_pipeline_matches_snapshot_in_all_six_families(self):
        for name, question, operator, sessions in fixtures():
            with self.subTest(name=name):
                old, old_layer = self.run_record(self.baseline, name, question, operator, sessions)
                new, new_layer = self.run_record(current, name, question, operator, sessions)
                self.assertEqual(semantic_trace(new), semantic_trace(old))
                self.assertEqual(new_layer.llm.calls, old_layer.llm.calls)
                self.assertEqual(new_layer.alias_to_entity_id, old_layer.alias_to_entity_id)
                self.assertEqual(new_layer.vector_memory, old_layer.vector_memory)
                self.assertEqual(dict(new_layer.G.nodes(data=True)), dict(old_layer.G.nodes(data=True)))
                self.assertEqual(list(new_layer.G.edges(data=True, keys=True)), list(old_layer.G.edges(data=True, keys=True)))

    def test_opt_in_does_not_call_candidate_for_existing_factual_and_specialized_fixtures(self):
        for name, question, operator, sessions in fixtures():
            if name == "multisession":
                continue
            with self.subTest(name=name):
                old, old_layer = self.run_record(self.baseline, name, question, operator, sessions)
                with patch("PRAGMOS_composable_candidate.run_candidate", side_effect=AssertionError("Protected path ran new model calls")):
                    new, new_layer = self.run_record(current, name, question, operator, sessions, mode="candidate")
                self.assertEqual(new["text"], old["text"])
                self.assertEqual(new["answer_decision"], old["answer_decision"])
                self.assertEqual(new_layer.llm.calls, old_layer.llm.calls)
                self.assertFalse(new["composable_execution_result"]["promoted"])

    def test_opt_in_calculation_replaces_only_answer_and_accounts_for_calls(self):
        from test_PRAGMOS_composable_execution import operand
        question = "What total did I spend on studio supplies?"
        sessions = [[{"role": "user", "content": f"I spent ${amount} on studio supplies."}] for amount in (17, 29)]
        request = {"version": 3, "reduction": "sum"}
        original = ModelFixture.__call__
        def respond(model, prompt, **kwargs):
            output = original(model, prompt, **kwargs)
            if prompt.startswith("<|user|>\nChoose the computation"):
                output["choices"][0]["text"] = json.dumps(request)
            elif "Assess this statement independently" in prompt:
                output["choices"][0]["text"] = json.dumps({"include": True})
            elif prompt.startswith("<|user|>\nExtract answer slots"):
                output["choices"][0]["text"] = json.dumps({"facts": [{"entity": "studio supplies", "actor": "I",
                    "status": "completed", "change": "none", "measurement": 0}]})
            return output
        old, old_layer = self.run_record(self.baseline, "multi", question, "SUM", sessions)
        with patch.object(ModelFixture, "__call__", respond), patch("PRAGMOS_composable_candidate._validate_native_grammar"):
            new, new_layer = self.run_record(current, "multi", question, "SUM", sessions, mode="candidate")
        self.assertEqual(new["text"], "$46")
        self.assertEqual(new["answer_decision"], "composable_evidence_execution")
        self.assertEqual(new["baseline_prediction"], old["text"])
        self.assertTrue(new["composable_execution_result"]["promoted"])
        self.assertEqual(new["llm_call_counts"]["total"] - old["llm_call_counts"]["total"], 5)
        self.assertEqual(new_layer.llm.state, old_layer.llm.state)
        self.assertEqual(new_layer.alias_to_entity_id, old_layer.alias_to_entity_id)
        self.assertEqual(new_layer.vector_memory, old_layer.vector_memory)

    def test_typed_opt_in_preserves_protected_intents_in_the_actual_pipeline(self):
        for name, question, operator, sessions in fixtures():
            if name not in {"assistant", "updates", "preference"}:
                continue
            with self.subTest(name=name):
                old, old_layer = self.run_record(self.baseline, name, question, operator, sessions)
                with patch("PRAGMOS_composable_candidate.run_typed_candidate", side_effect=AssertionError("Protected typed path")):
                    new, new_layer = self.run_record(current, name, question, operator, sessions,
                                                     mode="candidate", protocol="typed_v4")
                self.assertEqual(new["text"], old["text"])
                self.assertEqual(new["answer_decision"], old["answer_decision"])
                self.assertEqual(new_layer.llm.calls, old_layer.llm.calls)
                self.assertEqual(new_layer.llm.state, old_layer.llm.state)
                self.assertEqual(new_layer.alias_to_entity_id, old_layer.alias_to_entity_id)
                self.assertEqual(new_layer.vector_memory, old_layer.vector_memory)


if __name__ == "__main__":
    unittest.main()
