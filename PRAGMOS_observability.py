#!/usr/bin/env python3
"""Sidecar observability metrics for PRAGMOS benchmark traces."""

from __future__ import annotations

import statistics
from typing import Any, Iterable


MEASUREMENT_TRACE_SCHEMA_VERSION = 1


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def provenance_complete(record: dict[str, Any], *, graph: bool = False) -> bool:
    required = (
        "source_turn_ids",
        "source_quote",
        "timestamp",
    )
    if graph:
        required = (*required, "confidence")
    return all(_nonempty(record.get(field)) for field in required)


def temporal_fields_consistent(edge: dict[str, Any]) -> bool:
    status = edge.get("status")
    if not _nonempty(edge.get("edge_id")) or not _nonempty(edge.get("valid_from")):
        return False
    if status == "active":
        return not _nonempty(edge.get("valid_to")) and not _nonempty(
            edge.get("superseded_by_edge_id")
        )
    if status == "superseded":
        return _nonempty(edge.get("valid_to")) and _nonempty(
            edge.get("superseded_by_edge_id")
        )
    return False


def selected_memory_graph_quality(
    final_memories: Iterable[dict[str, Any]] | None,
    graph_evidence: Iterable[dict[str, Any]] | None,
    *,
    memory_count: int | None = None,
    graph_node_count: int | None = None,
    graph_edge_count: int | None = None,
) -> dict[str, Any]:
    memories = list(final_memories or [])
    edges = list(graph_evidence or [])
    complete_memories = sum(provenance_complete(item) for item in memories)
    complete_edges = sum(provenance_complete(item, graph=True) for item in edges)
    temporal_edges = [item for item in edges if item.get("status") is not None]
    consistent_temporal_edges = sum(
        temporal_fields_consistent(item) for item in temporal_edges
    )
    return {
        "selected_memory_count": len(memories),
        "selected_memory_provenance_complete_count": complete_memories,
        "selected_graph_evidence_count": len(edges),
        "selected_graph_provenance_complete_count": complete_edges,
        "selected_temporal_edge_count": len(temporal_edges),
        "selected_temporal_edge_consistent_count": consistent_temporal_edges,
        "memory_store_record_count": memory_count,
        "graph_node_count": graph_node_count,
        "graph_edge_count": graph_edge_count,
    }


def context_efficiency_record(
    *,
    source_conversation_tokens: int,
    injected_context_tokens: int,
    context_token_budget: int | None = None,
) -> dict[str, Any]:
    source_tokens = max(0, int(source_conversation_tokens))
    injected_tokens = max(0, int(injected_context_tokens))
    compression_ratio = (
        source_tokens / injected_tokens if injected_tokens > 0 else None
    )
    return {
        "source_conversation_tokens": source_tokens,
        "injected_context_tokens": injected_tokens,
        "source_to_injected_token_ratio": compression_ratio,
        "context_token_budget": (
            max(0, int(context_token_budget))
            if context_token_budget is not None
            else None
        ),
        "token_definition": (
            "Active local model tokenizer over a canonical serialization of the "
            "complete source conversation and the exact injected context count "
            "reported by the benchmark. The ratio measures compression only; "
            "retrieval coverage and answer quality measure retained usefulness."
        ),
    }


def build_measurement_trace(
    *,
    source_conversation_tokens: int,
    injected_context_tokens: int,
    context_token_budget: int | None = None,
    final_memories: Iterable[dict[str, Any]] | None = None,
    graph_evidence: Iterable[dict[str, Any]] | None = None,
    memory_count: int | None = None,
    graph_node_count: int | None = None,
    graph_edge_count: int | None = None,
) -> dict[str, Any]:
    memory_graph = None
    if any(
        value is not None
        for value in (memory_count, graph_node_count, graph_edge_count)
    ) or final_memories is not None or graph_evidence is not None:
        memory_graph = selected_memory_graph_quality(
            final_memories,
            graph_evidence,
            memory_count=memory_count,
            graph_node_count=graph_node_count,
            graph_edge_count=graph_edge_count,
        )
    return {
        "schema_version": MEASUREMENT_TRACE_SCHEMA_VERSION,
        "context_efficiency": context_efficiency_record(
            source_conversation_tokens=source_conversation_tokens,
            injected_context_tokens=injected_context_tokens,
            context_token_budget=context_token_budget,
        ),
        "memory_graph_quality": memory_graph,
    }


def measurement_trace_from_row(
    row: dict[str, Any],
    *,
    source_conversation_tokens: int | None = None,
) -> dict[str, Any]:
    existing = row.get("measurement_trace")
    if isinstance(existing, dict):
        return existing
    if source_conversation_tokens is None:
        raise ValueError(
            "Trace row has no measurement_trace and no source token count was supplied"
        )
    injected = row.get("context_tokens_estimate")
    if injected is None:
        injected = row.get("history_tokens_estimate")
    if injected is None:
        raise ValueError("Trace row has no injected context token count")
    mode = row.get("mode")
    is_pragmos = mode == "pragmos_context"
    return build_measurement_trace(
        source_conversation_tokens=source_conversation_tokens,
        injected_context_tokens=int(injected),
        context_token_budget=row.get("context_token_budget"),
        final_memories=row.get("final_memories") if is_pragmos else None,
        graph_evidence=row.get("graph_evidence") if is_pragmos else None,
        memory_count=row.get("memory_count") if is_pragmos else None,
        graph_node_count=row.get("graph_node_count") if is_pragmos else None,
        graph_edge_count=row.get("graph_edge_count") if is_pragmos else None,
    )


def _weighted_rate(complete: int, total: int) -> float | None:
    return complete / total if total else None


def aggregate_measurement_traces(
    traces: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(traces)
    if not rows:
        raise ValueError("No measurement traces were provided")

    contexts = [row["context_efficiency"] for row in rows]
    source_total = sum(int(row["source_conversation_tokens"]) for row in contexts)
    injected_total = sum(int(row["injected_context_tokens"]) for row in contexts)
    context_summary = {
        "question_count": len(contexts),
        "mean_injected_context_tokens": statistics.fmean(
            int(row["injected_context_tokens"]) for row in contexts
        ),
        "source_to_injected_token_ratio": (
            source_total / injected_total if injected_total else None
        ),
        "audit_totals": {
            "source_conversation_tokens": source_total,
            "injected_context_tokens": injected_total,
        },
    }

    quality_rows = [
        row["memory_graph_quality"]
        for row in rows
        if isinstance(row.get("memory_graph_quality"), dict)
    ]
    quality_summary = None
    if quality_rows:
        selected_memories = sum(row["selected_memory_count"] for row in quality_rows)
        complete_memories = sum(
            row["selected_memory_provenance_complete_count"]
            for row in quality_rows
        )
        selected_edges = sum(
            row["selected_graph_evidence_count"] for row in quality_rows
        )
        complete_edges = sum(
            row["selected_graph_provenance_complete_count"]
            for row in quality_rows
        )
        temporal_edges = sum(
            row["selected_temporal_edge_count"] for row in quality_rows
        )
        consistent_edges = sum(
            row["selected_temporal_edge_consistent_count"]
            for row in quality_rows
        )
        quality_summary = {
            "selected_memory_provenance_completeness_rate": _weighted_rate(
                complete_memories,
                selected_memories,
            ),
            "selected_graph_provenance_completeness_rate": _weighted_rate(
                complete_edges,
                selected_edges,
            ),
            "selected_graph_temporal_consistency_rate": _weighted_rate(
                consistent_edges,
                temporal_edges,
            ),
            "audit_counts": {
                "selected_memory_records": selected_memories,
                "selected_graph_edges": selected_edges,
                "selected_temporal_edges": temporal_edges,
            },
        }
    return {
        "schema_version": MEASUREMENT_TRACE_SCHEMA_VERSION,
        "context_efficiency": context_summary,
        "memory_graph_quality": quality_summary,
    }
