import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import PRAGMOS_analyze_typed_shadow as analyzer
from PRAGMOS_analyze_typed_shadow import build_report, projected_answer


def candidate(**overrides):
    report = {"status": "COMPLETE", "shadow": True, "answer": "$46", "extraction_complete": True,
              "isolation": "llama_state_restored_no_external_cache", "proof": {"executable": True, "provenance": ["source"]}}
    report.update(overrides)
    return report


def manifest():
    return {"config": {"pragmos_composable_mode": "shadow", "pragmos_composable_protocol": "typed_v4"},
            "workload": [{"question_id": "fixture"}]}


class ShadowReportTests(unittest.TestCase):
    def test_projection_reuses_promotion_contract_without_mutating_trace(self):
        trace = {"question_id": "fixture", "hypothesis": "$17", "composable_execution_result": candidate()}
        before = copy.deepcopy(trace)
        self.assertEqual(projected_answer(trace), ("$46", "would_replace_answer"))
        self.assertEqual(trace, before)

    def test_unresolved_or_protected_candidate_retains_existing_answer(self):
        for report in (None, {"status": "NOT_APPLICABLE"}, candidate(status="NOT_PROVEN"), candidate(extraction_complete=False)):
            with self.subTest(report=report):
                self.assertEqual(projected_answer({"hypothesis": "$17", "composable_execution_result": report})[0], "$17")

    def test_supported_abstention_uses_existing_veto_rules(self):
        report = candidate(status="NOT_PROVEN", supported_abstention=True,
                           proof={"abstention_supported": True, "provenance": ["source"],
                                  "coverage": {"status": "EXPLICIT_MISSING_INPUT"}})
        self.assertEqual(projected_answer({"hypothesis": "$17", "composable_execution_result": report})[0], "I do not know.")
        report["proof"]["coverage"]["status"] = "MISSING_OPERAND"
        self.assertEqual(projected_answer({"hypothesis": "$17", "composable_execution_result": report})[0], "$17")

    def test_rejects_actual_promoted_or_non_shadow_output(self):
        for report in (candidate(promoted=True), candidate(shadow=False), candidate(abstention_applied=True)):
            with self.subTest(report=report), self.assertRaises(ValueError):
                projected_answer({"hypothesis": "$17", "composable_execution_result": report})

    def test_reports_existing_and_projected_scores_separately(self):
        data = [{"question_id": "fixture", "question": "What total?", "answer": "$46"}]
        rows = [{"question_id": "fixture", "hypothesis": "$17", "composable_execution_result": candidate()}]
        result = build_report(data, rows, manifest())
        self.assertTrue(result["complete"])
        self.assertEqual(result["existing_mean_token_f1"], 0)
        self.assertEqual(result["projected_mean_token_f1"], 1)
        self.assertFalse(result["semantic_accuracy_measured"])

    def test_incomplete_workload_is_not_marked_complete(self):
        data = [{"question_id": key, "question": "Q", "answer": "A"} for key in ("fixture", "later")]
        plan = manifest()
        plan["workload"].append({"question_id": "later"})
        self.assertFalse(build_report(data, [{"question_id": "fixture", "hypothesis": "A"}], plan)["complete"])

    def test_fallback_is_compared_with_previously_correct_v3_control(self):
        data = [{"question_id": "fixture", "question": "Count?", "answer": "5"}]
        controls = {"rows": [{"question_id": "fixture", "prediction": "5"}]}
        result = build_report(data, [{"question_id": "fixture", "hypothesis": "2"}], manifest(), controls)
        self.assertEqual(result["projected_f1_regressions_vs_previous_v3_controls"], ["fixture"])

    def test_wrong_run_mode_duplicates_or_unknown_questions_are_rejected(self):
        data = [{"question_id": "fixture", "question": "Q", "answer": "A"}]
        rows = [{"question_id": "fixture", "hypothesis": "A"}]
        bad = manifest()
        bad["config"]["pragmos_composable_mode"] = "candidate"
        for records, traces, plan in ((data, rows, bad), (data, rows * 2, manifest()),
                                     (data, [{"question_id": "wrong", "hypothesis": "A"}], manifest())):
            with self.subTest(traces=traces), self.assertRaises(ValueError):
                build_report(records, traces, plan)

    def test_cli_verifies_input_hashes_and_writes_only_the_offline_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, trace, plan, output = (root / name for name in ("data.json", "trace.jsonl", "manifest.json", "report.json"))
            data.write_text(json.dumps([{"question_id": "fixture", "question": "What total?", "answer": "$46"}]))
            trace.write_text(json.dumps({"question_id": "fixture", "hypothesis": "$17", "composable_execution_result": candidate()}) + "\n")
            expected = manifest()
            expected["data_identity"] = {"sha256": analyzer.digest(data)}
            code = Path(analyzer.__file__).parent
            expected["source_identities"] = {name: {"sha256": analyzer.digest(code / name)} for name in (
                "PRAGMOS_composable_candidate.py", "PRAGMOS_benchmark_LongMemEval.py")}
            plan.write_text(json.dumps(expected))
            before = [path.read_bytes() for path in (data, trace, plan)]
            arguments = ["analyze", "--data-file", str(data), "--trace", str(trace), "--manifest", str(plan), "--output", str(output)]
            with patch.object(sys, "argv", arguments), patch("builtins.print"):
                analyzer.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["projected_exact_matches"], 1)
            self.assertEqual(before, [path.read_bytes() for path in (data, trace, plan)])


if __name__ == "__main__":
    unittest.main()
