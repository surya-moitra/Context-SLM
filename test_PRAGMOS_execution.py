import argparse
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from PRAGMOS_execution import (
    add_execution_arguments, index_with_planner, retrieve_independent_queries,
    validate_execution_arguments,
)


class ExecutionTests(unittest.TestCase):
    def test_serial_order_and_outputs(self):
        order = []
        indexed, planned, audit = index_with_planner(
            lambda: order.append("index") or ["turn"],
            lambda: order.append("planner") or {"contract": "same"},
        )
        self.assertEqual(order, ["index", "planner"])
        self.assertEqual(indexed, ["turn"])
        self.assertEqual(planned, {"contract": "same"})
        self.assertFalse(audit["overlap_enabled"])

    def test_overlap_has_one_planner_and_waits_for_both(self):
        planner_started, index_started = threading.Event(), threading.Event()
        calls = []

        def index():
            index_started.set()
            self.assertTrue(planner_started.wait(2))
            calls.append("index")
            return ["turn"]

        def plan():
            planner_started.set()
            self.assertTrue(index_started.wait(2))
            calls.append("plan")
            return "contract"

        indexed, planned, audit = index_with_planner(index, plan, overlap=True)
        self.assertCountEqual(calls, ["index", "plan"])
        self.assertEqual((indexed, planned), (["turn"], "contract"))
        self.assertTrue(audit["overlap_enabled"])

    def test_index_error_joins_planner_before_propagating(self):
        finished = threading.Event()

        def plan():
            finished.set()

        with self.assertRaisesRegex(RuntimeError, "index failed"):
            index_with_planner(Mock(side_effect=RuntimeError("index failed")), plan, overlap=True)
        self.assertTrue(finished.is_set())

    def test_planner_error_is_propagated(self):
        with self.assertRaisesRegex(RuntimeError, "planner failed"):
            index_with_planner(lambda: [], Mock(side_effect=RuntimeError("planner failed")), overlap=True)

    def test_disabled_planner_does_not_create_work(self):
        indexed, plan, audit = index_with_planner(lambda: [1], None, overlap=True)
        self.assertEqual(indexed, [1])
        self.assertIsNone(plan)
        self.assertEqual(audit["planner_seconds"], 0)
        self.assertFalse(audit["overlap_enabled"])

    def test_reference_defaults_and_explicit_overlap_guards(self):
        parser = argparse.ArgumentParser()
        add_execution_arguments(parser)
        args = parser.parse_args([])
        args.mode = "pragmos_context"
        self.assertEqual(args.pragmos_execution_policy, "reference")
        validate_execution_arguments(args)
        args.pragmos_overlap_planner = True
        for policy, device, mode in [("reference", "cpu", "pragmos_context"),
                                     ("optimized", "auto", "pragmos_context"),
                                     ("optimized", "cpu", "raw_phi3_haystack")]:
            args.pragmos_execution_policy, args.pragmos_retrieval_device, args.mode = policy, device, mode
            with self.assertRaises(ValueError):
                validate_execution_arguments(args)
        args.pragmos_execution_policy, args.pragmos_retrieval_device, args.mode = "optimized", "cpu", "pragmos_context"
        validate_execution_arguments(args)

    def test_batch_routing_keeps_query_order_and_duplicates(self):
        context = SimpleNamespace(execution_policy="optimized",
            retrieve_relevant_memories_many=Mock(return_value=[[1], [2], [1]]))
        self.assertEqual(retrieve_independent_queries(context, ["a", "b", "a"], top_k=4), [[1], [2], [1]])
        context.retrieve_relevant_memories_many.assert_called_once_with(["a", "b", "a"], top_k=4)

    def test_older_context_stays_serial(self):
        context = SimpleNamespace(retrieve_relevant_memories=Mock(side_effect=[[1], [2]]))
        self.assertEqual(retrieve_independent_queries(context, ["a", "b"], top_k=4), [[1], [2]])


class ExecutionPipelineParityTests(unittest.TestCase):
    def test_all_six_families_preserve_model_calls_answers_evidence_and_graphs(self):
        from unittest.mock import patch
        import PRAGMOS_benchmark_LongMemEval as adapter
        from test_PRAGMOS_composable_integration import (
            arguments, fixtures, make_context, semantic_plan, semantic_trace,
        )
        for name, question, operator, sessions in fixtures():
            record = {"question_date": "2027-02-01", "haystack_sessions": sessions,
                      "haystack_dates": [f"2027-01-0{i + 3}" for i in range(len(sessions))],
                      "haystack_session_ids": [f"s{i}" for i in range(len(sessions))]}
            runs = []
            for policy, overlap in [("reference", False), ("optimized", False), ("optimized", True)]:
                context, args = make_context(), arguments()
                context.execution_policy = args.pragmos_execution_policy = policy
                context.retrieval_device = args.pragmos_retrieval_device = "cpu"
                args.pragmos_overlap_planner = overlap
                with patch.object(adapter, "plan_query_with_local_slm", return_value=semantic_plan(question, operator)):
                    result = adapter.run_pragmos_record(context, record, question, args, name)
                runs.append((result, context))
            baseline, old = runs[0]
            for result, new in runs[1:]:
                with self.subTest(family=name, overlap=result["execution_diagnostics"]["overlap_enabled"]):
                    for key in ("text", "answer_decision", "safe_abstention", "operation_plan",
                                "operation_result", "final_memories", "graph_evidence", "query_profile"):
                        self.assertEqual(semantic_trace(result[key]), semantic_trace(baseline[key]), key)
                    self.assertEqual(new.llm.calls, old.llm.calls)
                    self.assertEqual(new.alias_to_entity_id, old.alias_to_entity_id)
                    self.assertEqual(new.vector_memory, old.vector_memory)
                    self.assertEqual(list(new.G.nodes(data=True)), list(old.G.nodes(data=True)))
                    self.assertEqual(list(new.G.edges(data=True, keys=True)), list(old.G.edges(data=True, keys=True)))


if __name__ == "__main__":
    unittest.main()
