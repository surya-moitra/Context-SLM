import unittest

import compare_PRAGMOS_LongMemEval_regression as comparator


def row(question_id, question_type, token_f1, exact_match=0.0):
    return {
        "question_id": question_id,
        "question_type": question_type,
        "hypothesis": f"answer-{token_f1}",
        "local_metrics": {
            "token_f1": token_f1,
            "exact_match": exact_match,
            "contains_reference": exact_match,
        },
    }


class LongMemEvalRegressionComparatorTests(unittest.TestCase):
    def test_identical_traces_pass_strict_gate(self):
        baseline = {
            "q1": row("q1", "multi-session", 1.0, 1.0),
            "q2": row("q2", "temporal-reasoning", 0.5),
        }
        report = comparator.compare_traces(
            baseline,
            baseline,
            expected_count=2,
        )

        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["question_regression_count"], 0)

    def test_individual_and_type_regression_fail_gate(self):
        baseline = {
            "q1": row("q1", "multi-session", 1.0),
            "q2": row("q2", "temporal-reasoning", 0.5),
        }
        candidate = {
            "q1": row("q1", "multi-session", 0.0),
            "q2": row("q2", "temporal-reasoning", 1.0),
        }
        report = comparator.compare_traces(
            baseline,
            candidate,
            expected_count=2,
        )

        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["failing_question_types"], ["multi-session"])
        self.assertEqual(report["question_regression_count"], 1)

    def test_question_sets_must_match(self):
        baseline = {"q1": row("q1", "multi-session", 1.0)}
        candidate = {"q2": row("q2", "multi-session", 1.0)}

        with self.assertRaisesRegex(ValueError, "question IDs differ"):
            comparator.compare_traces(
                baseline,
                candidate,
                expected_count=1,
            )

    def test_tolerance_can_ignore_tiny_numeric_drift(self):
        baseline = {"q1": row("q1", "multi-session", 1.0)}
        candidate = {"q1": row("q1", "multi-session", 0.9999)}
        report = comparator.compare_traces(
            baseline,
            candidate,
            expected_count=1,
            tolerance=0.001,
        )

        self.assertTrue(report["gate_passed"])


if __name__ == "__main__":
    unittest.main()
