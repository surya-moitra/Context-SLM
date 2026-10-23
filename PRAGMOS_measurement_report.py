#!/usr/bin/env python3
"""Combine benchmark, judge, and resource summaries into one compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_family(summary: dict[str, Any]) -> str:
    if "avg_official_locomo_f1" in summary:
        return "locomo"
    if "avg_token_f1" in summary:
        return "longmemeval"
    raise ValueError("Benchmark summary is neither recognized LongMemEval nor LoCoMo output")


def retrieval_metrics(summary: dict[str, Any], family: str) -> dict[str, Any]:
    if family == "locomo":
        retrieval = summary.get("session_retrieval") or {}
        return {
            "recall_at_5": retrieval.get("mean_recall_at_5"),
            "recall_at_10": retrieval.get("mean_recall_at_10"),
            "complete_evidence_coverage_rate": retrieval.get(
                "complete_evidence_coverage_rate"
            ),
        }

    retrieval = (
        (summary.get("longmemeval_session_retrieval_metrics") or {})
        .get("final_evidence", {})
        .get("metrics", {})
    )
    return {
        "recall_at_5": retrieval.get("recall@5"),
        "recall_at_10": retrieval.get("recall@10"),
        "complete_evidence_coverage_rate": summary.get(
            "answer_session_full_coverage_rate"
        ),
    }


def build_measurement_report(
    benchmark_summary: dict[str, Any],
    resource_summary: dict[str, Any],
    judge_summary: dict[str, Any] | None = None,
    trace_metrics_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resource_summary.get("completed") is False:
        raise ValueError("Resource measurement did not complete successfully")
    benchmark_count = benchmark_summary.get("newly_processed")
    if benchmark_count is None:
        benchmark_count = benchmark_summary.get("processed")
    resource_count = resource_summary.get("processed_questions")
    if (
        benchmark_count is not None
        and resource_count is not None
        and int(benchmark_count) != int(resource_count)
    ):
        raise ValueError(
            "Benchmark and resource summaries cover different invocation counts: "
            f"{benchmark_count} != {resource_count}"
        )
    if trace_metrics_summary is not None:
        trace_count = trace_metrics_summary.get("processed")
        benchmark_total = benchmark_summary.get("processed")
        if (
            trace_count is not None
            and benchmark_total is not None
            and int(trace_count) != int(benchmark_total)
        ):
            raise ValueError(
                "Benchmark and trace-metric summaries cover different question counts: "
                f"{benchmark_total} != {trace_count}"
            )
    family = benchmark_family(benchmark_summary)
    f1_name = "official_locomo_f1" if family == "locomo" else "token_f1"
    f1_value = (
        benchmark_summary.get("avg_official_locomo_f1")
        if family == "locomo"
        else benchmark_summary.get("avg_token_f1")
    )
    judge_summary = judge_summary or {}
    trace_metrics_summary = trace_metrics_summary or benchmark_summary
    context_efficiency = trace_metrics_summary.get("context_efficiency") or {}
    memory_graph_quality = trace_metrics_summary.get("memory_graph_quality") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_family": family,
        "run_name": resource_summary.get("run_name"),
        "quality": {
            "f1_metric": f1_name,
            "f1": f1_value,
            "llm_judge_accuracy": judge_summary.get("overall_accuracy"),
            "llm_judge_model": judge_summary.get("metric_model"),
        },
        "retrieval": retrieval_metrics(benchmark_summary, family),
        "latency": {
            "mean_seconds_per_question": benchmark_summary.get(
                "avg_elapsed_seconds"
            ),
        },
        "resources": resource_summary.get("headline_metrics") or {},
        "context_efficiency": {
            "mean_injected_context_tokens": context_efficiency.get(
                "mean_injected_context_tokens"
            ),
            "source_to_injected_token_ratio": context_efficiency.get(
                "source_to_injected_token_ratio"
            ),
        },
        "memory_graph_quality": {
            "selected_memory_provenance_completeness_rate": (
                memory_graph_quality.get(
                    "selected_memory_provenance_completeness_rate"
                )
            ),
            "selected_graph_provenance_completeness_rate": (
                memory_graph_quality.get(
                    "selected_graph_provenance_completeness_rate"
                )
            ),
            "selected_graph_temporal_consistency_rate": (
                memory_graph_quality.get(
                    "selected_graph_temporal_consistency_rate"
                )
            ),
        },
        "execution": {
            "processed_questions": resource_summary.get("processed_questions"),
            "completed": resource_summary.get("completed"),
            "hardware": resource_summary.get("hardware") or {},
            "controls": resource_summary.get("execution_controls") or {},
            "process_tree_enumeration_complete": resource_summary.get(
                "process_tree_enumeration_complete"
            ),
            "measurement_notes": resource_summary.get("measurement_notes") or [],
        },
    }


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report_markdown(report: dict[str, Any]) -> str:
    quality = report["quality"]
    retrieval = report["retrieval"]
    latency = report["latency"]
    resources = report["resources"]
    context_efficiency = report["context_efficiency"]
    memory_graph_quality = report["memory_graph_quality"]
    execution = report["execution"]
    hardware = execution["hardware"]
    controls = execution["controls"]
    lines = [
        f"# Measurement Report: {report.get('run_name') or 'unnamed run'}",
        "",
        f"- Benchmark: {report['benchmark_family']}",
        f"- Processed questions: {format_value(execution.get('processed_questions'))}",
        f"- Hardware: {hardware.get('hardware_model') or 'unknown'}; "
        f"{format_value(hardware.get('physical_memory_gib'), 2)} GiB RAM",
        f"- Execution: {controls.get('accelerator_scope') or 'unspecified'}; "
        f"local-only controls={format_value(controls.get('local_only_requested'))}",
        "",
        "## Quality",
        "",
        f"- {quality['f1_metric']}: {format_value(quality.get('f1'))}",
        f"- LLM-judge accuracy: {format_value(quality.get('llm_judge_accuracy'))}",
        "",
        "## Retrieval And Latency",
        "",
        f"- Recall@5: {format_value(retrieval.get('recall_at_5'))}",
        f"- Recall@10: {format_value(retrieval.get('recall_at_10'))}",
        "- Complete evidence coverage: "
        f"{format_value(retrieval.get('complete_evidence_coverage_rate'))}",
        "- Mean latency per question: "
        f"{format_value(latency.get('mean_seconds_per_question'), 3)} seconds",
        "",
        "## Context Efficiency",
        "",
        "- Mean injected context: "
        f"{format_value(context_efficiency.get('mean_injected_context_tokens'), 2)} tokens",
        "- Source-to-injected token ratio: "
        f"{format_value(context_efficiency.get('source_to_injected_token_ratio'), 2)}x",
        "",
        "## Memory And Graph Quality",
        "",
        "- Selected-memory provenance completeness: "
        f"{format_value(memory_graph_quality.get('selected_memory_provenance_completeness_rate'))}",
        "- Selected-graph provenance completeness: "
        f"{format_value(memory_graph_quality.get('selected_graph_provenance_completeness_rate'))}",
        "- Selected-graph temporal consistency: "
        f"{format_value(memory_graph_quality.get('selected_graph_temporal_consistency_rate'))}",
        "",
        "## Local Resources",
        "",
        "- Peak process-tree RAM: "
        f"{format_value(resources.get('peak_process_tree_rss_gib'), 3)} GiB",
        "- Mean CPU utilization: "
        f"{format_value(resources.get('mean_cpu_utilization_percent_of_machine'), 2)}% "
        "of machine capacity",
        "- CPU time per processed question: "
        f"{format_value(resources.get('cpu_seconds_per_processed_question'), 3)} seconds",
        "",
    ]
    if execution.get("measurement_notes"):
        lines.extend(
            [
                "## Measurement Notes",
                "",
                *[f"- {note}" for note in execution["measurement_notes"]],
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--resource-summary", type=Path, required=True)
    parser.add_argument("--judge-summary", type=Path)
    parser.add_argument("--trace-metrics", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    try:
        report = build_measurement_report(
            load_json(args.benchmark_summary),
            load_json(args.resource_summary),
            load_json(args.judge_summary) if args.judge_summary else None,
            load_json(args.trace_metrics) if args.trace_metrics else None,
        )
    except ValueError as error:
        raise SystemExit(f"report error: {error}") from error
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote measurement report: {args.output_json}")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(report_markdown(report), encoding="utf-8")
        print(f"Wrote measurement report: {args.output_markdown}")


if __name__ == "__main__":
    main()
