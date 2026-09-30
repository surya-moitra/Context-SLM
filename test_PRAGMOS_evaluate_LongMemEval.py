import json
import tempfile
import unittest
from pathlib import Path

from PRAGMOS_evaluate_LongMemEval import (
    OFFICIAL_METRIC_MODEL,
    fingerprint_manifest,
    official_metrics,
    prepare_checkpoint,
    validate_workload,
)


class OfficialJudgeWrapperTests(unittest.TestCase):
    @staticmethod
    def prediction(question_id, hypothesis="answer"):
        return {"question_id": question_id, "hypothesis": hypothesis}

    @staticmethod
    def reference(question_id, question_type):
        return {
            "question_id": question_id,
            "question_type": question_type,
            "question": "Question?",
            "answer": "Answer",
        }

    @staticmethod
    def judged(question_id, label, hypothesis="answer"):
        return {
            "question_id": question_id,
            "hypothesis": hypothesis,
            "autoeval_label": {
                "model": OFFICIAL_METRIC_MODEL,
                "label": label,
            },
        }

    def test_validate_workload_preserves_prediction_order_and_selection(self):
        predictions = [
            self.prediction("q1"),
            self.prediction("q2"),
            self.prediction("q3"),
        ]
        references = [
            self.reference("q3", "knowledge-update"),
            self.reference("q1", "single-session-user"),
            self.reference("q2", "multi-session"),
        ]

        selected = validate_workload(
            predictions,
            references,
            start_index=1,
            limit=2,
        )

        self.assertEqual([row["question_id"] for row in selected], ["q2", "q3"])

    def test_validate_workload_rejects_duplicates_and_missing_oracle_rows(self):
        references = [self.reference("q1", "single-session-user")]
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            validate_workload(
                [self.prediction("q1"), self.prediction("q1")],
                references,
            )
        with self.assertRaisesRegex(ValueError, "absent from oracle"):
            validate_workload([self.prediction("q2")], references)

    def test_official_metrics_match_macro_and_overall_definitions(self):
        rows = [
            ("user-1", "single-session-user", True),
            ("user-2", "single-session-user", False),
            ("pref", "single-session-preference", True),
            ("assistant", "single-session-assistant", True),
            ("multi", "multi-session", False),
            ("temporal", "temporal-reasoning", True),
            ("update_abs", "knowledge-update", True),
        ]
        references = {
            question_id: self.reference(question_id, question_type)
            for question_id, question_type, _label in rows
        }
        results = [
            self.judged(question_id, label)
            for question_id, _question_type, label in rows
        ]

        metrics = official_metrics(results, references)

        self.assertAlmostEqual(metrics["overall_accuracy"], 5 / 7)
        self.assertAlmostEqual(metrics["task_averaged_accuracy"], 0.75)
        self.assertEqual(metrics["abstention_accuracy"], 1.0)
        self.assertEqual(metrics["abstention_count"], 1)
        self.assertEqual(
            metrics["by_question_type"]["single-session-user"],
            {"accuracy": 0.5, "count": 2},
        )

    def test_resume_repairs_a_partial_final_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            result_path = root / "results.jsonl"
            summary_path = root / "summary.json"
            workload = [
                {
                    "question_id": "q1",
                    "prediction": self.prediction("q1"),
                },
                {
                    "question_id": "q2",
                    "prediction": self.prediction("q2"),
                },
            ]
            manifest = fingerprint_manifest(
                {
                    "schema_version": 1,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "run_name": "judge-test",
                    "config": {"model": OFFICIAL_METRIC_MODEL},
                }
            )
            prepare_checkpoint(
                manifest_path,
                result_path,
                summary_path,
                manifest,
                workload,
                resume=True,
            )
            first = self.judged("q1", True)
            result_path.write_text(
                json.dumps(first) + "\n" + '{"question_id":',
                encoding="utf-8",
            )

            results, repaired = prepare_checkpoint(
                manifest_path,
                result_path,
                summary_path,
                manifest,
                workload,
                resume=True,
            )

            self.assertTrue(repaired)
            self.assertEqual([row["question_id"] for row in results], ["q1"])
            self.assertTrue(result_path.read_text(encoding="utf-8").endswith("\n"))

    def test_resume_rejects_changed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                root / "manifest.json",
                root / "results.jsonl",
                root / "summary.json",
            )
            workload = []
            first = fingerprint_manifest(
                {
                    "schema_version": 1,
                    "created_at": "first",
                    "config": {"limit": 1},
                }
            )
            changed = fingerprint_manifest(
                {
                    "schema_version": 1,
                    "created_at": "second",
                    "config": {"limit": 2},
                }
            )
            prepare_checkpoint(*paths, first, workload, resume=True)

            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_checkpoint(*paths, changed, workload, resume=True)


if __name__ == "__main__":
    unittest.main()
