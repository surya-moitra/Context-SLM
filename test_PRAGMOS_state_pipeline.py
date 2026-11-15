"""Paired default-off parity against the committed pre-state implementation."""

import hashlib
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import PRAGMOS_benchmark_LongMemEval as current
from test_PRAGMOS_composable_integration import (
    arguments, fixtures, make_context, semantic_plan, semantic_trace,
)


class StatePipelineParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            source = subprocess.check_output(
                ["git", "show", "18d4519:PRAGMOS_benchmark_LongMemEval.py"], stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            raise unittest.SkipTest("Frozen pre-state revision unavailable in this checkout")
        if hashlib.sha256(source).hexdigest() != "149e70e6ce2c883723a4d6fb40c8b9888f385221d0a510e6fabbbeebf14c4692":
            raise AssertionError("Pre-state snapshot does not match recorded hash")
        cls.baseline = ModuleType("_pragmos_pre_state_snapshot")
        cls.baseline.__file__ = current.__file__
        sys.modules[cls.baseline.__name__] = cls.baseline
        exec(compile(source, "<frozen-pre-state>", "exec"), cls.baseline.__dict__)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("_pragmos_pre_state_snapshot", None)

    def run_record(self, module, name, question, operator, sessions, state_mode="off"):
        layer, args = make_context(), arguments()
        args.pragmos_state_mode = state_mode
        record = {"question_date": "2027-02-01", "haystack_sessions": sessions,
                  "haystack_dates": [f"2027-01-0{i + 3}" for i in range(len(sessions))],
                  "haystack_session_ids": [f"s{i}" for i in range(len(sessions))]}
        with patch.object(module, "plan_query_with_local_slm", return_value=semantic_plan(question, operator)):
            result = module.run_pragmos_record(layer, record, question, args, name)
        return result, layer

    def test_off_keeps_answers_prompts_calls_and_memory_identical_in_six_families(self):
        for name, question, operator, sessions in fixtures():
            with self.subTest(family=name):
                before, old = self.run_record(self.baseline, name, question, operator, sessions)
                with patch("PRAGMOS_state_views.run_state_candidate", side_effect=AssertionError("Off path called state model")):
                    after, new = self.run_record(current, name, question, operator, sessions)
                self.assertEqual(semantic_trace(after), semantic_trace(before))
                self.assertEqual(new.llm.calls, old.llm.calls)
                self.assertEqual(new.llm.state, old.llm.state)
                self.assertEqual(new.vector_memory, old.vector_memory)
                self.assertEqual(new.alias_to_entity_id, old.alias_to_entity_id)
                self.assertEqual(list(new.G.nodes(data=True)), list(old.G.nodes(data=True)))
                self.assertEqual(list(new.G.edges(data=True, keys=True)), list(old.G.edges(data=True, keys=True)))

    def test_shadow_never_promotes_even_a_complete_different_answer(self):
        for name, question, operator, sessions in fixtures():
            with self.subTest(family=name):
                baseline, old = self.run_record(current, name, question, operator, sessions)
                candidate = {"version": 1, "status": "COMPLETE", "answer": "Different candidate", "promoted": False,
                             "source_view_complete": True, "proof": {"output_slots_complete": True, "source_bound": True},
                             "isolation": "llama_state_restored_no_external_cache",
                             "calls": {"planner_calls": 0, "extractor_calls": 0}}
                with patch("PRAGMOS_state_views.run_state_candidate", return_value=candidate):
                    shadow, new = self.run_record(current, name, question, operator, sessions, state_mode="shadow")
                self.assertEqual(shadow["text"], baseline["text"])
                self.assertEqual(shadow["answer_decision"], baseline["answer_decision"])
                self.assertEqual(shadow["safe_abstention"], baseline["safe_abstention"])
                self.assertEqual(new.llm.calls, old.llm.calls)
                self.assertEqual(new.vector_memory, old.vector_memory)
                self.assertEqual(list(new.G.edges(data=True, keys=True)), list(old.G.edges(data=True, keys=True)))

    def test_unvalidated_support_failure_cannot_veto_the_existing_answer(self):
        name, question, operator, sessions = next(iter(fixtures()))
        baseline, _ = self.run_record(current, name, question, operator, sessions)
        candidate = {"version": 2, "status": "COMPLETE", "answer": "I do not know.", "promoted": False,
                     "source_view_complete": True, "proof": {"kind": "abstention",
                     "required_identity_unbound": ["unbound identity"], "reviewed_source_ids": ["S1"],
                     "source_witnesses": [{"source_id": "S1", "quote": "a different identity"}]},
                     "isolation": "llama_state_restored_no_external_cache",
                     "calls": {"planner_calls": 0, "extractor_calls": 0}}
        with patch.object(current, "evaluate_state_candidate", return_value=candidate):
            result, _ = self.run_record(current, name, question, operator, sessions, state_mode="candidate")
        self.assertEqual(result["safe_abstention"], baseline["safe_abstention"])
        self.assertEqual(result["text"], baseline["text"])
        self.assertEqual(result["answer_decision"], baseline["answer_decision"])


if __name__ == "__main__":
    unittest.main()
