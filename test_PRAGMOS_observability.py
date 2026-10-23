import json
import unittest

import PRAGMOS_analyze_trace_metrics as analyzer
import PRAGMOS_observability as observability


def memory(turn_id, quote="A fact"):
    return {
        "source_turn_ids": [turn_id],
        "source_quote": quote,
        "timestamp": "2027-01-01T10:00:00",
    }


def edge(edge_id, status="active"):
    result = {
        **memory(edge_id),
        "edge_id": edge_id,
        "confidence": 0.75,
        "valid_from": "2027-01-01T10:00:00",
        "valid_to": None,
        "status": status,
        "superseded_by_edge_id": None,
    }
    if status == "superseded":
        result["valid_to"] = "2027-02-01T10:00:00"
        result["superseded_by_edge_id"] = "edge-new"
    return result


class ObservabilityMetricTests(unittest.TestCase):
    def test_context_efficiency_uses_exact_token_counts(self):
        record = observability.context_efficiency_record(
            source_conversation_tokens=12000,
            injected_context_tokens=1000,
            context_token_budget=1300,
        )

        self.assertEqual(record["source_to_injected_token_ratio"], 12.0)
        self.assertEqual(record["injected_context_tokens"], 1000)

    def test_selected_quality_checks_provenance_and_temporal_fields(self):
        incomplete_memory = memory(2, quote="")
        inconsistent_edge = edge("edge-bad")
        inconsistent_edge["valid_to"] = "2027-02-01T10:00:00"

        quality = observability.selected_memory_graph_quality(
            [memory(1), incomplete_memory],
            [edge("edge-good"), inconsistent_edge],
            memory_count=100,
            graph_node_count=8,
            graph_edge_count=12,
        )

        self.assertEqual(quality["selected_memory_provenance_complete_count"], 1)
        self.assertEqual(quality["selected_graph_provenance_complete_count"], 2)
        self.assertEqual(quality["selected_temporal_edge_consistent_count"], 1)
        self.assertEqual(quality["memory_store_record_count"], 100)

    def test_aggregate_uses_weighted_counts_not_mean_of_rates(self):
        first = observability.build_measurement_trace(
            source_conversation_tokens=1000,
            injected_context_tokens=100,
            final_memories=[memory(1), memory(2, quote="")],
            graph_evidence=[edge("edge-1")],
            memory_count=2,
            graph_node_count=2,
            graph_edge_count=1,
        )
        second = observability.build_measurement_trace(
            source_conversation_tokens=3000,
            injected_context_tokens=300,
            final_memories=[memory(3)],
            graph_evidence=[],
            memory_count=1,
            graph_node_count=0,
            graph_edge_count=0,
        )

        aggregate = observability.aggregate_measurement_traces([first, second])

        self.assertEqual(
            aggregate["context_efficiency"]["source_to_injected_token_ratio"],
            10.0,
        )
        self.assertEqual(
            aggregate["memory_graph_quality"][
                "selected_memory_provenance_completeness_rate"
            ],
            2 / 3,
        )
        self.assertEqual(
            aggregate["memory_graph_quality"][
                "selected_graph_temporal_consistency_rate"
            ],
            1.0,
        )

    def test_measurement_trace_is_json_serializable(self):
        trace = observability.build_measurement_trace(
            source_conversation_tokens=500,
            injected_context_tokens=100,
            final_memories=[memory(1)],
            graph_evidence=[edge("edge-1")],
            memory_count=20,
            graph_node_count=4,
            graph_edge_count=3,
        )

        serialized = json.dumps(trace)

        self.assertIn('"schema_version": 1', serialized)


class TraceAnalyzerTests(unittest.TestCase):
    def test_legacy_longmemeval_trace_can_be_derived_without_core_changes(self):
        data = [
            {
                "question_id": "q1",
                "haystack_sessions": [
                    [{"role": "user", "content": "I like green tea."}]
                ],
                "haystack_session_ids": ["session-1"],
                "haystack_dates": ["2027-01-01"],
            }
        ]
        trace_rows = [
            {
                "question_id": "q1",
                "mode": "pragmos_context",
                "context_tokens_estimate": 10,
                "context_token_budget": 20,
                "memory_count": 4,
                "graph_node_count": 2,
                "graph_edge_count": 1,
                "final_memories": [memory(1)],
                "graph_evidence": [edge("edge-1")],
            }
        ]

        measurements = analyzer.derive_measurement_traces(
            trace_rows,
            family="longmemeval",
            data_records=data,
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(len(measurements), 1)
        self.assertEqual(
            measurements[0]["context_efficiency"]["injected_context_tokens"],
            10,
        )
        self.assertEqual(
            measurements[0]["memory_graph_quality"]["graph_edge_count"],
            1,
        )

    def test_locomo_source_serialization_includes_all_sessions_and_captions(self):
        sample = {
            "sample_id": "sample-a",
            "conversation": {
                "session_1": [
                    {
                        "speaker": "Asha",
                        "dia_id": "D1:1",
                        "text": "I started painting.",
                        "blip_caption": "A blue canvas.",
                    }
                ],
                "session_1_date_time": "11:15 am on 5 January, 2027",
                "session_2": [
                    {
                        "speaker": "Ben",
                        "dia_id": "D2:1",
                        "text": "I started hiking.",
                    }
                ],
                "session_2_date_time": "2:30 pm on 8 January, 2027",
            },
            "qa": [{"question_id": "q1"}],
        }

        sources = analyzer.source_texts_by_question([sample], "locomo")

        self.assertIn("Asha: I started painting.", sources["q1"])
        self.assertIn("Asha [image_caption]: A blue canvas.", sources["q1"])
        self.assertIn("Ben: I started hiking.", sources["q1"])


if __name__ == "__main__":
    unittest.main()
