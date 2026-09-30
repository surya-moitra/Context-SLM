import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PRAGMOS_evaluate_LongMemEval import (
    OFFICIAL_API_BASE_URL,
    OFFICIAL_METRIC_MODEL,
    create_official_client,
    chat_completion_with_bounded_retries,
    fingerprint_manifest,
    is_retryable_api_error,
    official_metrics,
    prepare_checkpoint,
    validate_workload,
)


class FakeAPIError(Exception):
    def __init__(self, message, *, status_code=None, code=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"code": code} if code else None


class FakeConnectionError(FakeAPIError):
    pass


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.call_count = 0

    def create(self, **_request):
        outcome = self.outcomes[self.call_count]
        self.call_count += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def fake_official():
    return SimpleNamespace(
        openai=SimpleNamespace(
            APIConnectionError=FakeConnectionError,
            APITimeoutError=type("FakeTimeoutError", (FakeAPIError,), {}),
        )
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

    def test_bounded_retry_recovers_from_transient_server_error(self):
        completion = SimpleNamespace(_request_id="request-ok")
        completions = FakeCompletions(
            [FakeAPIError("temporary", status_code=500), completion]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        delays = []

        result, audit = chat_completion_with_bounded_retries(
            fake_official(),
            client,
            max_attempts=3,
            retry_base_seconds=2,
            retry_max_seconds=10,
            sleep=delays.append,
            model=OFFICIAL_METRIC_MODEL,
        )

        self.assertIs(result, completion)
        self.assertEqual(audit, {"attempt_count": 2, "request_id": "request-ok"})
        self.assertEqual(delays, [2])
        self.assertEqual(completions.call_count, 2)

    def test_insufficient_quota_is_not_retried(self):
        error = FakeAPIError(
            "quota exhausted",
            status_code=429,
            code="insufficient_quota",
        )
        completions = FakeCompletions([error])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        delays = []

        self.assertFalse(is_retryable_api_error(fake_official(), error))
        with self.assertRaisesRegex(FakeAPIError, "quota exhausted"):
            chat_completion_with_bounded_retries(
                fake_official(),
                client,
                max_attempts=4,
                retry_base_seconds=2,
                retry_max_seconds=10,
                sleep=delays.append,
                model=OFFICIAL_METRIC_MODEL,
            )

        self.assertEqual(delays, [])
        self.assertEqual(completions.call_count, 1)

    def test_connection_error_is_retryable_without_status_code(self):
        error = FakeConnectionError("network unavailable")
        self.assertTrue(is_retryable_api_error(fake_official(), error))

    def test_official_client_ignores_base_url_environment_override(self):
        captured = {}

        def construct_client(**options):
            captured.update(options)
            return SimpleNamespace()

        official = SimpleNamespace(OpenAI=construct_client)
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": "http://127.0.0.1:9999/v1"},
        ):
            create_official_client(official, "test-key", 30)

        self.assertEqual(captured["base_url"], OFFICIAL_API_BASE_URL)
        self.assertEqual(captured["max_retries"], 0)
        captured["http_client"].close()


if __name__ == "__main__":
    unittest.main()
