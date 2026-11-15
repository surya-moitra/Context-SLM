import math
import json
import unittest
from types import SimpleNamespace

from PRAGMOS_evidence_context import (
    evidence_source, pack_evidence_context, quote_spans, source_units,
    unit_text, word_chunk_spans,
)
from PRAGMOS_context_layer_org import ContextLayer, ConversationTurn
from PRAGMOS_benchmark_LongMemEval import (
    build_pragmos_answer_suffix, canonicalize_generated_scalar_answer,
    extract_candidates_from_selected_evidence,
    materialize_candidate_graph,
)


def turn(text, turn_id=1, speaker="Asha", role="user"):
    return SimpleNamespace(text=text, turn_id=turn_id, speaker=speaker,
                           role=role, session_id="s1", timestamp="2026-01-02")


def memory(source, quote=None):
    return {"source_quote": source.text if quote is None else quote,
            "source_turn_ids": [source.turn_id], "source_session_id": source.session_id,
            "speaker": source.speaker, "role": source.role, "timestamp": source.timestamp}


def pack(sources, records=None, budget=10000, question="What is recorded?", graph=None):
    return pack_evidence_context(
        user_input=question, vector_memories=records if records is not None else [memory(t) for t in sources],
        graph_evidence=graph or [], recent_turns=[], source_turns={t.turn_id: t for t in sources},
        header="[CONTEXT]\n", footer="\n[/CONTEXT]", token_budget=budget,
        count_tokens=len, tokenize=lambda text: text.lower().split(),
    )


class SourcePackingTests(unittest.TestCase):
    def test_late_fact_and_qualification_survive(self):
        source = turn("Background. " * 60 + "The total is $120, excluding mandatory delivery.")
        context, audit = pack([source])
        self.assertIn("$120, excluding mandatory delivery.", context)
        self.assertTrue(audit["question_preserved"])

    def test_original_list_layout_and_ordinals_survive(self):
        source = turn("Suggested order:\n1. Cedar\n2. Willow\n3. Spruce", role="assistant")
        context, _ = pack([source], [memory(source, "2. Willow")])
        self.assertIn(source.text, context)
        self.assertIn("role=assistant", context)

    def test_table_chunk_restores_column_header(self):
        source = turn("Locations:\n| Site | City |\n| --- | --- |\n| North | Ottawa |\n| South | Quito |")
        context, audit = pack([source], [memory(source, "| South | Quito |")])
        self.assertIn(source.text, context)
        self.assertEqual(audit["included_units"][0]["kind"], "table")

    def test_whitespace_normalized_old_chunk_restores_original_layout(self):
        source = turn("Checklist:\n1. Cedar\n2. Willow")
        context, _ = pack([source], [memory(source, "1. Cedar 2. Willow")])
        self.assertIn(source.text, context)

    def test_same_quote_different_speakers_is_not_deduplicated(self):
        a, b = turn("I enjoy painting."), turn("I enjoy painting.", 2, "Ben")
        context, audit = pack([a, b])
        self.assertEqual(len(audit["included_units"]), 2)
        self.assertIn("speaker=Asha", context)
        self.assertIn("speaker=Ben", context)

    def test_duplicate_chunks_do_not_repeat_source(self):
        source = turn("A shared source paragraph.")
        _, audit = pack([source], [memory(source), memory(source)])
        self.assertEqual(len(audit["included_units"]), 1)

    def test_large_first_source_does_not_starve_another_session(self):
        first = turn("Asha recorded a detailed entry. " * 15)
        second = turn("Ben recorded a short entry.", 2, "Ben")
        second.session_id = "s2"
        context, audit = pack([first, second], budget=750)
        self.assertIn(second.text, context)
        self.assertEqual({item["session_id"] for item in audit["included_units"]}, {"s1", "s2"})
        self.assertLessEqual(len(context), 750)

    def test_wrong_actor_or_session_or_timestamp_cannot_expand_source(self):
        source = turn("Asha owns a blue bicycle.")
        for field in ("speaker", "source_session_id", "timestamp"):
            with self.subTest(field=field):
                record = {**memory(source), field: "unrelated"}
                context, audit = pack([source], [record])
                self.assertNotIn(source.text, context)
                self.assertEqual(audit["source_errors"][0]["reason"], "source_identity_conflict")

    def test_unresolved_quote_is_not_guessed(self):
        source = turn("I chose red.")
        _, audit = pack([source], [memory(source, "I chose blue.")])
        self.assertEqual(audit["included_units"], [])
        self.assertEqual(audit["source_errors"][0]["reason"], "source_quote_unresolved")

    def test_small_budget_packs_whole_list_item_with_original_number(self):
        source = turn("Options:\n1. Cedar\n2. " + "Willow " * 100)
        context, audit = pack([source], budget=330)
        self.assertLessEqual(len(context), 330)
        self.assertIn("1. Cedar", context)
        self.assertNotIn("2. Willow", context)
        self.assertTrue(audit["partial_containers"])
        self.assertTrue(audit["skipped_units"])

    def test_question_is_never_prefix_truncated(self):
        question = "Which original and current values were requested?"
        context, _ = pack([], question=question, budget=500)
        self.assertIn(question, context)
        with self.assertRaises(ValueError):
            pack([], question=question, budget=5)

    def test_indivisible_long_source_is_skipped_not_sliced(self):
        source = turn("unbroken " * 100)
        context, audit = pack([source], budget=300)
        self.assertNotIn("unbroken", context)
        self.assertTrue(audit["skipped_units"])
        self.assertLessEqual(len(context), 300)

    def test_graph_status_keeps_source_identity_and_history(self):
        source = turn("I used to prefer cats.")
        edge = {**memory(source), "head": "Asha", "relation": "preferred", "tail": "cats",
                "status": "superseded", "valid_to": "2026-01-03"}
        context, _ = pack([source], [], graph=[edge])
        self.assertIn(source.text, context)
        self.assertIn("status=superseded", context)
        self.assertIn("valid_to=2026-01-03", context)

    def test_multiple_graph_source_ids_remain_distinct(self):
        a = {"source_turn_ids": [1, 2], "source_quote": "Asha owns Cedar.", "speaker": "Asha"}
        b = {**a, "source_turn_ids": [3, 4]}
        context, audit = pack([], [a, b])
        self.assertEqual(len(audit["included_units"]), 2)
        self.assertIn("turn=[1, 2]", context)
        self.assertIn("turn=[3, 4]", context)

    def test_invalid_graph_source_does_not_leak_derived_hint(self):
        source = turn("I prefer cats.")
        edge = {**memory(source), "speaker": "Ben", "path": "Ben likes cats"}
        context, _ = pack([source], [], graph=[edge])
        self.assertNotIn("Ben likes cats", context)

    def test_heading_only_source_is_not_lost(self):
        source = turn("Final selection:")
        context, _ = pack([source])
        self.assertIn(source.text, context)

    def test_pipe_in_prose_is_not_mistaken_for_table(self):
        self.assertEqual(source_units("Use A | B\n\nNext paragraph.")[0].kind, "paragraph")

    def test_source_spans_and_old_embedding_whitespace_are_preserved(self):
        text = "\n".join("word" + str(i) for i in range(40))
        spans = word_chunk_spans(text, 16, 4)
        self.assertEqual(text[spans[0][0]:spans[0][1]].split(), text.split()[:16])
        self.assertIn("\n", text[spans[0][0]:spans[0][1]])
        self.assertEqual(spans[-1][1], len(text))

    def test_indexed_embedding_text_matches_previous_chunk_format(self):
        layer = ContextLayer.__new__(ContextLayer)
        layer.embed_and_store_many = lambda entries, batch_size: entries
        source = ConversationTurn(session_id="s1", turn_id=1, role="assistant", speaker="assistant",
                                  timestamp="2026-01-02", text="\n".join("item" + str(i) for i in range(40)))
        entries = layer.index_raw_turns([source], chunk_words=16, overlap_words=4)
        old_chunks = layer.split_text_for_memory(source.text, 16, 4)
        self.assertEqual(len(entries), len(old_chunks))
        for index, (entry, old_chunk) in enumerate(zip(entries, old_chunks)):
            self.assertEqual(entry["text"], layer.format_turn_chunk_for_memory(source, old_chunk, index, len(old_chunks)))
            meta = entry["metadata"]
            self.assertEqual(meta["source_quote"], source.text[meta["source_start"]:meta["source_end"]])
        self.assertEqual(layer.raw_source_turns[1], source)

    def test_repeated_quote_locations_are_not_guessed(self):
        self.assertEqual(quote_spans("red blue\nred blue", "red blue"), [(0, 8), (9, 17)])

    def test_headings_are_provenance_spans(self):
        text = "# Session\n## Picks\n- Cedar\n- Willow"
        unit = source_units(text)[0]
        self.assertEqual(unit_text(text, unit), text)
        self.assertEqual(unit.context, ((0, 10), (10, 19)))

    def test_caption_and_parent_dialogue_identity_are_visible(self):
        source = turn("A red ball on a desk.")
        source.evidence_type = "image_caption"
        source.external_turn_id = "D1:2#caption"
        source.parent_turn_id = "D1:2"
        context, audit = pack([source])
        self.assertIn("evidence_type=image_caption", context)
        self.assertIn("parent_turn_id=D1:2", context)
        self.assertEqual(audit["included_units"][0]["external_turn_id"], "D1:2#caption")


class ExtractionContext:
    context_length = 2048

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, text):
        return math.ceil(len(text) / 4)

    def llm(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        text, reason = response
        return {"choices": [{"text": text, "finish_reason": reason}]}


def extract(context, memories, existing=None):
    return extract_candidates_from_selected_evidence(
        context, "What does Asha own?", "specific fact requested",
        {"operation": "none"}, memories, existing_extractions=existing,
    )


class ExtractionRecoveryTests(unittest.TestCase):
    def test_truncated_output_is_retried_without_losing_existing_fact(self):
        source = turn("Asha owns Cedar and Willow.")
        context = ExtractionContext([
            ("E1 | Asha | owns | Cedar", "length"),
            ("E1 | Asha | owns | Willow\nEVIDENCE_COMPLETE | E1", "stop"),
        ])
        rows, audit = extract(context, [memory(source)])
        self.assertEqual(rows[0]["triples"], [["Asha", "owns", "Cedar"], ["Asha", "owns", "Willow"]])
        self.assertEqual(audit["candidate_extraction_batch_count"], 2)
        self.assertEqual(audit["repair_evidence_ids"], ["E1"])
        self.assertFalse(audit["semantic_completeness_verified"])
        for prompt, kwargs in context.calls:
            self.assertLessEqual(context.count_tokens(prompt) + kwargs["max_tokens"] + 64, context.context_length)

    def test_failed_call_retains_raw_evidence_and_explicit_failure(self):
        source = turn("Asha owns Cedar.")
        rows, audit = extract(ExtractionContext([RuntimeError("bad response")] * 2), [memory(source)])
        self.assertEqual(rows[0]["source_quote"], source.text)
        self.assertEqual(rows[0]["extraction_status"], "incomplete")
        self.assertEqual(audit["interpretation_status"], "incomplete")
        self.assertEqual(len(audit["batch_errors"]), 2)

    def test_failed_repair_keeps_valid_first_pass_facts(self):
        context = ExtractionContext([("E1 | Asha | owns | Cedar", "length"), RuntimeError("repair failed")])
        rows, audit = extract(context, [memory(turn("Asha owns Cedar and Willow."))])
        self.assertEqual(rows[0]["triples"], [["Asha", "owns", "Cedar"]])
        attempts = rows[0]["extraction_review"]["windows"]["0"]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts], ["output_truncated", "failed"])
        self.assertFalse(audit["evidence_absence_established"])

    def test_preliminary_failure_preserves_source_for_selected_recovery(self):
        source = ConversationTurn(session_id="s1", turn_id=1, role="user", speaker="Asha",
                                  timestamp="2026-01-02", text="Asha owns Cedar.")
        context = ContextLayer.__new__(ContextLayer)
        context.learn_speaker_identity_from_text = lambda _turn: None
        context.format_turn_for_extraction = lambda source: source.text
        context.extract_entities_and_relationships_with_llm = lambda _text: (_ for _ in ()).throw(ValueError("bad extraction"))
        context.update_knowledge_graph_from_triples = lambda *_args, **_kwargs: None
        rows = materialize_candidate_graph(context, [memory(source)], {1: source}, limit=1)
        self.assertEqual(rows[0]["extraction_status"], "incomplete")
        self.assertEqual(rows[0]["source_quote"], source.text)
        recovery = ExtractionContext([("E1 | Asha | owns | Cedar\nEVIDENCE_COMPLETE | E1", "stop")])
        recovered, _ = extract(recovery, [memory(source)], rows)
        self.assertEqual(recovered[0]["triples"], [["Asha", "owns", "Cedar"]])

    def test_ungrounded_output_does_not_become_reviewed_despite_marker(self):
        source = turn("Asha owns Cedar.")
        response = ("E1 | Asha | owns | Mirage\nEVIDENCE_COMPLETE | E1", "stop")
        rows, audit = extract(ExtractionContext([response] * 2), [memory(source)])
        self.assertEqual(rows[0]["triples"], [])
        self.assertEqual(rows[0]["extraction_status"], "incomplete")
        self.assertEqual(audit["repair_evidence_ids"], ["E1"])

    def test_empty_output_is_incomplete_not_evidence_absence(self):
        rows, audit = extract(ExtractionContext([("", "stop")] * 2), [memory(turn("Asha owns Cedar."))])
        self.assertEqual(rows[0]["extraction_review"]["windows"]["0"]["status"], "empty_output")
        self.assertEqual(audit["interpretation_status"], "incomplete")

    def test_legacy_preliminary_facts_are_reused_without_new_calls(self):
        source = turn("Asha owns Cedar.")
        context = ExtractionContext([])
        old = {"source_turn_id": 1, "source_session_id": "s1", "source_role": "user",
               "source_speaker": "Asha", "source_quote": source.text,
               "triples": [["Asha", "owns", "Cedar"]]}
        rows, audit = extract(context, [memory(source)], [old])
        self.assertEqual(context.calls, [])
        self.assertEqual(rows[0]["triples"], old["triples"])
        self.assertEqual(audit["reused_graph_extraction_count"], 1)

    def test_preliminary_truncation_is_observable(self):
        layer = ContextLayer.__new__(ContextLayer)
        layer.llm = lambda *_args, **_kwargs: {"choices": [{"text": "Asha | owns | Cedar", "finish_reason": "length"}]}
        self.assertEqual(layer.extract_entities_and_relationships_with_llm("Asha owns Cedar."), [("Asha", "owns", "Cedar")])
        self.assertEqual(layer.last_graph_extraction_diagnostics["status"], "incomplete")

    def test_long_indivisible_source_is_reported_without_silent_trim(self):
        source = turn("content " * 4000)
        context = ExtractionContext([])
        rows, audit = extract(context, [memory(source)])
        self.assertEqual(context.calls, [])
        self.assertEqual(rows[0]["source_quote"], source.text)
        self.assertTrue(audit["unprocessed_source_spans"])
        self.assertEqual(rows[0]["extraction_status"], "incomplete")

    def test_repair_is_bounded_to_two_evidence_items(self):
        sources = [turn("Asha owns Cedar.", i) for i in range(1, 4)]
        context = ExtractionContext([("broken output", "stop")] * 3)
        _, audit = extract(context, [memory(t) for t in sources])
        self.assertEqual(len(audit["repair_evidence_ids"]), 2)
        self.assertEqual(len(context.calls), 3)

    def test_valid_completed_window_does_not_add_a_repair_call(self):
        context = ExtractionContext([("E1 | Asha | owns | Cedar\nEVIDENCE_COMPLETE | E1", "stop")])
        rows, audit = extract(context, [memory(turn("Asha owns Cedar."))])
        self.assertEqual(rows[0]["extraction_status"], "reviewed")
        self.assertEqual(audit["repair_evidence_ids"], [])
        self.assertFalse(audit["semantic_completeness_verified"])
        self.assertEqual(json.loads(json.dumps(rows)), rows)

    def test_copied_protocol_header_does_not_trigger_a_repair(self):
        context = ExtractionContext([("evidence_id|entity|relation|entity\nE1|Asha|owns|Cedar", "stop")])
        rows, audit = extract(context, [memory(turn("Asha owns Cedar."))])
        self.assertEqual(rows[0]["triples"], [["Asha", "owns", "Cedar"]])
        self.assertEqual(len(context.calls), 1)
        self.assertEqual(audit["repair_evidence_ids"], [])
        self.assertEqual(rows[0]["extraction_status"], "incomplete")

    def test_saturated_window_gets_a_larger_bounded_review(self):
        source = turn("Asha owns " + ", ".join("item" + str(i) for i in range(7)) + ".")
        triples = [f"E1 | Asha | owns | item{i}" for i in range(7)]
        context = ExtractionContext([("\n".join(triples[:6]), "stop"),
                                     ("\n".join(triples) + "\nEVIDENCE_COMPLETE | E1", "stop")])
        rows, audit = extract(context, [memory(source)])
        self.assertEqual(len(rows[0]["triples"]), 7)
        self.assertEqual(len(context.calls), 2)
        self.assertEqual(rows[0]["extraction_status"], "reviewed")

    def test_raw_source_windows_not_prefix_truncation_cover_later_paragraphs(self):
        paragraphs = [f"Asha owns item{i}. " + "Details " * 130 for i in range(6)]
        source = turn("\n\n".join(paragraphs))
        context = ExtractionContext([("EVIDENCE_COMPLETE | E1", "stop")] * 4)
        rows, audit = extract(context, [memory(source)])
        self.assertEqual(len(context.calls), 4)
        self.assertIn("item3", context.calls[-1][0])
        self.assertTrue(audit["unprocessed_source_spans"]["E1"])
        self.assertEqual(rows[0]["extraction_status"], "incomplete")
        self.assertNotIn("TRUNCATED_TO_CONTEXT_BUDGET", "".join(prompt for prompt, _ in context.calls))


class CompleteAnswerTests(unittest.TestCase):
    def test_compound_quantity_is_not_replaced_by_first_candidate(self):
        answer = "Originally 6 hours; now 8 hours."
        self.assertEqual(canonicalize_generated_scalar_answer(answer, "exact quantity or amount", [{"value": "6"}]), answer)

    def test_range_is_not_replaced_by_first_candidate(self):
        answer = "6 to 8 hours"
        self.assertEqual(canonicalize_generated_scalar_answer(answer, "exact duration", [{"value": "6"}]), answer)

    def test_single_value_cleanup_is_unchanged(self):
        self.assertEqual(canonicalize_generated_scalar_answer("20 playlists on SoundWave", "exact quantity or amount", [{"value": "20"}]), "20")

    def test_existing_qualifier_is_not_stripped(self):
        answer = "At least 6 hours"
        self.assertEqual(canonicalize_generated_scalar_answer(answer, "exact duration", [{"value": "6 hours"}]), answer)

    def test_suffix_allows_requested_lists_and_compound_answers(self):
        suffix = build_pragmos_answer_suffix("Which two values were recorded?", "current")
        self.assertIn("complete requested answer", suffix)
        self.assertNotIn("exact contiguous span", suffix)


class ProtectedPipelineBehaviorTests(unittest.TestCase):
    def test_answers_decisions_and_graphs_match_prechange_in_six_families(self):
        from test_PRAGMOS_state_pipeline import StatePipelineParityTests
        from test_PRAGMOS_composable_integration import fixtures, semantic_trace
        import PRAGMOS_benchmark_LongMemEval as current

        comparison = StatePipelineParityTests()
        comparison.setUpClass()
        try:
            for name, question, operator, sessions in fixtures():
                with self.subTest(family=name):
                    before, old = comparison.run_record(comparison.baseline, name, question, operator, sessions)
                    after, new = comparison.run_record(current, name, question, operator, sessions)
                    for key in ("text", "answer_decision", "safe_abstention", "abstention_reason",
                                "operation_plan", "operation_result", "query_profile", "actor_grounding"):
                        self.assertEqual(semantic_trace(after[key]), semantic_trace(before[key]), key)
                    self.assertEqual(new.alias_to_entity_id, old.alias_to_entity_id)
                    self.assertEqual(new.vector_memory, old.vector_memory)
                    self.assertEqual(list(new.G.nodes(data=True)), list(old.G.nodes(data=True)))
                    self.assertEqual(list(new.G.edges(data=True, keys=True)), list(old.G.edges(data=True, keys=True)))
        finally:
            comparison.tearDownClass()


if __name__ == "__main__":
    unittest.main()
