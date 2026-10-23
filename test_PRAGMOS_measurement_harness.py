import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import PRAGMOS_measurement_harness as harness
import PRAGMOS_measurement_report as reporter


ROOT = Path(__file__).resolve().parent


class MeasurementHarnessTests(unittest.TestCase):
    def test_frozen_core_matches_publication_ref(self):
        result = harness.verify_core_files(ROOT, harness.DEFAULT_CORE_REF)

        self.assertTrue(result["enabled"])
        self.assertEqual(set(result["files"]), set(harness.FROZEN_CORE_FILES))
        self.assertTrue(
            all(item["matches_expected_ref"] for item in result["files"].values())
        )

    def test_local_only_environment_removes_cloud_credentials(self):
        environment = {
            "PATH": "/bin",
            "OPENAI_API_KEY": "must-not-leak",
            "AWS_SECRET_ACCESS_KEY": "must-not-leak-either",
        }

        child, audit = harness.prepare_child_environment(
            environment,
            local_only=True,
        )

        self.assertNotIn("OPENAI_API_KEY", child)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", child)
        self.assertEqual(child["HF_HUB_OFFLINE"], "1")
        self.assertEqual(
            audit["cloud_credential_variables_removed"],
            ["AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY"],
        )

    def test_headline_metrics_are_machine_normalized(self):
        metrics = harness.calculate_headline_metrics(
            peak_rss_bytes=2 * 1024**3,
            cpu_seconds=40.0,
            wall_seconds=10.0,
            logical_cpu_count=8,
            processed_questions=5,
        )

        self.assertEqual(metrics["peak_process_tree_rss_gib"], 2.0)
        self.assertEqual(metrics["mean_cpu_utilization_percent_of_machine"], 50.0)
        self.assertEqual(metrics["cpu_seconds_per_processed_question"], 8.0)

    def test_processed_question_count_prefers_new_work(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text(
                json.dumps({"processed": 200, "newly_processed": 121}),
                encoding="utf-8",
            )

            count, source = harness.resolve_processed_questions(summary_path, 999)

        self.assertEqual(count, 121)
        self.assertEqual(source, "benchmark_summary.newly_processed")

    def test_cli_samples_an_unchanged_child_process(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            command = [
                sys.executable,
                str(ROOT / "PRAGMOS_measurement_harness.py"),
                "--run-name",
                "unit_smoke",
                "--output-dir",
                str(output_dir),
                "--question-count",
                "2",
                "--sample-interval",
                "0.02",
                "--skip-core-check",
                "--",
                sys.executable,
                "-c",
                (
                    "import time; data=bytearray(8*1024*1024); "
                    "sum(i*i for i in range(300000)); time.sleep(0.08)"
                ),
            ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(
                (output_dir / "unit_smoke_resource_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            samples = (
                output_dir / "unit_smoke_resource_samples.jsonl"
            ).read_text(encoding="utf-8").splitlines()

        self.assertTrue(summary["completed"])
        self.assertEqual(summary["processed_questions"], 2)
        self.assertGreater(
            summary["headline_metrics"]["peak_process_tree_rss_gib"],
            0,
        )
        self.assertGreaterEqual(len(samples), 2)


class MeasurementReportTests(unittest.TestCase):
    def resource_summary(self):
        return {
            "run_name": "candidate",
            "completed": True,
            "processed_questions": 200,
            "wall_seconds": 16000.0,
            "headline_metrics": {
                "peak_process_tree_rss_gib": 3.25,
                "mean_cpu_utilization_percent_of_machine": 61.0,
                "cpu_seconds_per_processed_question": 27.0,
            },
            "hardware": {"hardware_model": "Mac", "physical_memory_gib": 8.0},
            "execution_controls": {"local_only_requested": True},
            "process_tree_enumeration_complete": True,
            "measurement_notes": [],
        }

    def test_longmemeval_report_uses_final_evidence_retrieval(self):
        benchmark = {
            "avg_token_f1": 0.83,
            "avg_elapsed_seconds": 75.0,
            "answer_session_full_coverage_rate": 0.995,
            "longmemeval_session_retrieval_metrics": {
                "final_evidence": {
                    "metrics": {"recall@5": 0.98, "recall@10": 1.0}
                }
            },
        }

        report = reporter.build_measurement_report(
            benchmark,
            self.resource_summary(),
            {"overall_accuracy": 0.8, "metric_model": "gpt-4o"},
        )

        self.assertEqual(report["benchmark_family"], "longmemeval")
        self.assertEqual(report["quality"]["f1"], 0.83)
        self.assertEqual(report["quality"]["llm_judge_accuracy"], 0.8)
        self.assertEqual(report["retrieval"]["recall_at_5"], 0.98)
        self.assertEqual(
            report["latency"]["end_to_end_wall_seconds_per_question"],
            80.0,
        )
        self.assertEqual(
            report["latency"][
                "benchmark_internal_mean_seconds_per_question"
            ],
            75.0,
        )
        self.assertEqual(len(report["resources"]), 3)
        self.assertTrue(report["execution"]["process_tree_enumeration_complete"])

    def test_locomo_report_uses_session_retrieval(self):
        benchmark = {
            "avg_official_locomo_f1": 0.9,
            "avg_elapsed_seconds": 60.0,
            "context_efficiency": {
                "mean_injected_context_tokens": 900.0,
                "source_to_injected_token_ratio": 12.5,
            },
            "memory_graph_quality": {
                "selected_memory_provenance_completeness_rate": 1.0,
                "selected_graph_provenance_completeness_rate": 0.99,
                "selected_graph_temporal_consistency_rate": 1.0,
            },
            "session_retrieval": {
                "mean_recall_at_5": 0.91,
                "mean_recall_at_10": 0.97,
                "complete_evidence_coverage_rate": 0.88,
            },
        }

        report = reporter.build_measurement_report(
            benchmark,
            self.resource_summary(),
        )

        self.assertEqual(report["benchmark_family"], "locomo")
        self.assertEqual(report["quality"]["f1_metric"], "official_locomo_f1")
        self.assertEqual(report["retrieval"]["recall_at_10"], 0.97)
        self.assertEqual(
            report["context_efficiency"]["source_to_injected_token_ratio"],
            12.5,
        )
        self.assertEqual(
            report["memory_graph_quality"][
                "selected_graph_provenance_completeness_rate"
            ],
            0.99,
        )

    def test_report_rejects_mismatched_invocation_counts(self):
        benchmark = {
            "avg_token_f1": 0.8,
            "newly_processed": 500,
        }

        with self.assertRaisesRegex(ValueError, "different invocation counts"):
            reporter.build_measurement_report(
                benchmark,
                self.resource_summary(),
            )

    def test_resumed_run_uses_newly_processed_question_count(self):
        benchmark = {
            "avg_token_f1": 0.8,
            "avg_elapsed_seconds": 3.0,
            "processed": 200,
            "newly_processed": 25,
        }
        resources = self.resource_summary()
        resources["processed_questions"] = 25
        resources["wall_seconds"] = 100.0

        report = reporter.build_measurement_report(benchmark, resources)

        self.assertEqual(
            report["latency"]["end_to_end_wall_seconds_per_question"],
            4.0,
        )

    def test_report_rejects_invalid_wall_time_or_question_count(self):
        benchmark = {"avg_token_f1": 0.8}
        missing_wall = self.resource_summary()
        missing_wall.pop("wall_seconds")
        with self.assertRaisesRegex(ValueError, "missing wall_seconds"):
            reporter.build_measurement_report(benchmark, missing_wall)

        invalid_wall = self.resource_summary()
        invalid_wall["wall_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "wall_seconds"):
            reporter.build_measurement_report(benchmark, invalid_wall)

        invalid_count = self.resource_summary()
        invalid_count["processed_questions"] = 0
        with self.assertRaisesRegex(ValueError, "processed_questions"):
            reporter.build_measurement_report(benchmark, invalid_count)

    def test_markdown_labels_end_to_end_latency_as_headline(self):
        benchmark = {
            "avg_token_f1": 0.8,
            "avg_elapsed_seconds": 60.0,
        }

        markdown = reporter.report_markdown(
            reporter.build_measurement_report(benchmark, self.resource_summary())
        )

        self.assertIn("End-to-end wall time per question: 80.000 seconds", markdown)
        self.assertIn(
            "Benchmark-reported internal mean latency (diagnostic): 60.000 seconds",
            markdown,
        )

    def test_sidecar_trace_metrics_are_used_for_longmemeval(self):
        benchmark = {
            "avg_token_f1": 0.8,
            "avg_elapsed_seconds": 60.0,
        }
        sidecar = {
            "context_efficiency": {
                "mean_injected_context_tokens": 1000.0,
                "source_to_injected_token_ratio": 8.0,
            },
            "memory_graph_quality": {
                "selected_memory_provenance_completeness_rate": 1.0,
                "selected_graph_provenance_completeness_rate": 1.0,
                "selected_graph_temporal_consistency_rate": 1.0,
            },
        }

        report = reporter.build_measurement_report(
            benchmark,
            self.resource_summary(),
            trace_metrics_summary=sidecar,
        )

        self.assertEqual(
            report["context_efficiency"]["source_to_injected_token_ratio"],
            8.0,
        )


if __name__ == "__main__":
    unittest.main()
