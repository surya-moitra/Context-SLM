import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PRAGMOS_benchmark_LongMemEval import (
    ASSISTANT_MEMORY_INTENT,
    EVIDENCE_SYNTHESIS_INTENT,
    PREFERENCE_RECOMMENDATION_INTENT,
    aggregate_longmemeval_retrieval_metrics,
    assess_state_attribute_mismatch,
    anchor_coverage,
    anchor_coverage_across_evidence,
    answer_session_recall,
    build_query_profile,
    build_preference_profile,
    build_pragmos_answer_suffix,
    build_evidence_synthesis_premises,
    canonicalize_generated_scalar_answer,
    deduplicate_operation_facts,
    determine_safe_abstention,
    evidence_rows_without_candidate_extraction,
    evidence_synthesis_scope_profile,
    execute_operation_plan,
    extract_candidates_from_selected_evidence,
    extract_measurement_facts,
    filter_candidates_by_query_anchors,
    filter_candidates_by_query_anchors_in_evidence,
    infer_query_intent,
    infer_multi_session_operation,
    infer_requested_answer_slot,
    infer_state_history_selector,
    longmemeval_session_retrieval_metrics,
    operation_retrieval_queries,
    parse_number_value,
    parse_operation_selection,
    parse_selected_evidence_triples,
    prepare_run_checkpoint,
    format_answer_slot_candidates,
    format_evidence_synthesis_context,
    format_preference_profile,
    counter_delta,
    pragmos_ablation_set,
    prioritize_memories_for_query_intent,
    rerank_answer_slot_candidates,
    retrieve_temporal_adjacent_date_memories,
    retrieve_collection_memories_until_saturated,
    retrieve_evidence_synthesis_memories,
    resolve_deterministic_ordinal_list_answer,
    resolve_deterministic_state_history_answer,
    resolve_temporal_event_chains,
    resolve_temporal_expression,
    resolve_typed_answer_slot,
    select_answer_slot_candidates,
    select_session_diverse_memories,
    select_operation_user_turn_memories,
    session_neighbor_priority,
    temporal_operation_result,
    temporal_join_result,
    validate_operation_fact_coverage,
    with_manifest_fingerprint,
)


def extraction(
    quote,
    triples,
    role="user",
    turn_id="turn-1",
    session_id="session-1",
    timestamp="2026-01-01T10:00:00Z",
    speaker=None,
):
    return {
        "source_turn_id": turn_id,
        "source_session_id": session_id,
        "source_role": role,
        "source_speaker": speaker or role,
        "source_timestamp": timestamp,
        "source_quote": quote,
        "triples": triples,
    }


def memory(
    quote,
    turn_id,
    session_id="session-1",
    role="user",
    score=0.8,
):
    return {
        "memory_id": f"memory-{turn_id}",
        "source_turn_ids": [turn_id],
        "source_session_id": session_id,
        "source_quote": quote,
        "role": role,
        "speaker": role,
        "timestamp": "2026-01-01T10:00:00Z",
        "score": score,
    }


class FakeCandidateExtractionContext:
    context_length = 2048

    def __init__(self, extraction_output="", selector_output=""):
        self.extraction_output = extraction_output
        self.selector_output = selector_output
        self.prompts = []

    def count_tokens(self, text):
        return max(1, (len(text) + 3) // 4)

    def trim_text_to_token_budget(self, text, budget):
        return text[: max(0, budget * 4)]

    def llm(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        if "strict evidence selector" in prompt:
            text = self.selector_output
        else:
            text = self.extraction_output
        return {"choices": [{"text": text}]}


class SelectAllOperationContext:
    def count_tokens(self, text):
        return max(1, (len(text) + 3) // 4)

    def llm(self, prompt, **_kwargs):
        candidate_ids = re.findall(r"^\[([MC]\d+)\]", prompt, flags=re.MULTILINE)
        prefix = "ITEM" if "ITEM | candidate_id" in prompt else "OPERAND"
        return {
            "choices": [
                {
                    "text": "\n".join(
                        f"{prefix} | {candidate_id} | selected {candidate_id}"
                        for candidate_id in candidate_ids
                    )
                }
            ]
        }


class ResumeCheckpointTests(unittest.TestCase):
    @staticmethod
    def manifest(mode="raw_phi3_haystack"):
        return with_manifest_fingerprint(
            {
                "schema_version": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "run_name": "resume-test",
                "mode": mode,
                "config": {"mode": mode, "n_ctx": 2048},
                "data_identity": {"sha256": "data"},
                "model_identity": {"sha256": "model"},
                "source_identities": {"runner": {"sha256": "source"}},
                "workload": [
                    {"dataset_index": 0, "question_id": "q1"},
                    {"dataset_index": 1, "question_id": "q2"},
                ],
            }
        )

    @staticmethod
    def workload():
        return [
            {"dataset_index": 0, "question_id": "q1"},
            {"dataset_index": 1, "question_id": "q2"},
        ]

    @staticmethod
    def write_jsonl(path, rows, trailing_text=""):
        text = "".join(json.dumps(row) + "\n" for row in rows)
        Path(path).write_text(text + trailing_text, encoding="utf-8")

    def paths(self, directory):
        root = Path(directory)
        return {
            "manifest_path": root / "run_manifest.json",
            "predictions_path": root / "run_predictions.jsonl",
            "trace_path": root / "run_trace.jsonl",
            "summary_path": root / "run_summary.json",
        }

    def prepare(self, paths, manifest=None, resume=True):
        return prepare_run_checkpoint(
            **paths,
            manifest=manifest or self.manifest(),
            workload=self.workload(),
            resume=resume,
        )

    def test_resume_creates_manifest_and_loads_a_paired_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            initial = self.prepare(paths)
            self.assertEqual(initial["completed_count"], 0)
            self.assertTrue(paths["manifest_path"].is_file())

            predictions = [
                {"question_id": "q1", "hypothesis": "one"},
                {"question_id": "q2", "hypothesis": "two"},
            ]
            traces = [
                {**predictions[0], "dataset_index": 0},
                {**predictions[1], "dataset_index": 1},
            ]
            self.write_jsonl(paths["predictions_path"], predictions)
            self.write_jsonl(paths["trace_path"], traces)

            resumed = self.prepare(paths)

            self.assertEqual(resumed["completed_count"], 2)
            self.assertFalse(resumed["repaired"])
            self.assertEqual(
                [row["question_id"] for row in resumed["traces"]],
                ["q1", "q2"],
            )

    def test_resume_truncates_an_unpaired_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            self.prepare(paths)
            predictions = [
                {"question_id": "q1", "hypothesis": "one"},
                {"question_id": "q2", "hypothesis": "two"},
            ]
            traces = [{**predictions[0], "dataset_index": 0}]
            self.write_jsonl(paths["predictions_path"], predictions)
            self.write_jsonl(paths["trace_path"], traces)

            resumed = self.prepare(paths)

            self.assertEqual(resumed["completed_count"], 1)
            self.assertTrue(resumed["repaired"])
            persisted = paths["predictions_path"].read_text(encoding="utf-8")
            self.assertIn('"question_id": "q1"', persisted)
            self.assertNotIn('"question_id": "q2"', persisted)

    def test_resume_truncates_a_partial_final_trace_line(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            self.prepare(paths)
            predictions = [
                {"question_id": "q1", "hypothesis": "one"},
                {"question_id": "q2", "hypothesis": "two"},
            ]
            traces = [{**predictions[0], "dataset_index": 0}]
            self.write_jsonl(paths["predictions_path"], predictions)
            self.write_jsonl(paths["trace_path"], traces, trailing_text='{"question_id":')

            resumed = self.prepare(paths)

            self.assertEqual(resumed["completed_count"], 1)
            self.assertTrue(resumed["repaired"])
            self.assertTrue(
                paths["trace_path"].read_text(encoding="utf-8").endswith("\n")
            )

    def test_resume_rejects_a_changed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            self.prepare(paths)

            with self.assertRaisesRegex(ValueError, "manifest changed"):
                self.prepare(paths, manifest=self.manifest(mode="pragmos_context"))

    def test_resume_rejects_a_non_prefix_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            self.prepare(paths)
            prediction = {"question_id": "q2", "hypothesis": "two"}
            self.write_jsonl(paths["predictions_path"], [prediction])
            self.write_jsonl(
                paths["trace_path"],
                [{**prediction, "dataset_index": 1}],
            )

            with self.assertRaisesRegex(ValueError, "not an exact prefix"):
                self.prepare(paths)

    def test_non_resume_mode_does_not_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            self.prepare(paths, resume=False)

            with self.assertRaises(FileExistsError):
                self.prepare(paths, resume=False)


class QueryRoutingRegressionTests(unittest.TestCase):
    def test_selector_parser_accepts_common_phi3_variants(self):
        candidates = [
            {"candidate_id": "C1", "value": "alpha"},
            {"candidate_id": "C2", "value": "beta"},
            {"candidate_id": "C3", "value": "gamma"},
            {"candidate_id": "C4", "value": "delta"},
        ]
        selected = parse_operation_selection(
            "\n".join(
                [
                    "[C2] second supported choice",
                    "I1 | C3 | third supported choice",
                    "candidate_id=C1 | first supported choice",
                    "REJECT C9",
                    "C4 is unsupported and should not be selected",
                    "ITEM | C2 | duplicate",
                ]
            ),
            candidates,
            "ITEM",
        )

        self.assertEqual(
            [candidate["candidate_id"] for candidate in selected],
            ["C2", "C3", "C1"],
        )
        self.assertEqual(selected[0]["canonical_identity"], "second supported choice")

    def test_selector_parser_respects_a_declared_candidate_id_column(self):
        candidates = [
            {"candidate_id": "C2", "value": "selected"},
            {"candidate_id": "C15", "value": "looks-like-an-id"},
        ]

        selected = parse_operation_selection(
            "ITEM | candidate_id\nC15 | C2",
            candidates,
            "ITEM",
        )

        self.assertEqual(
            [candidate["candidate_id"] for candidate in selected],
            ["C2"],
        )
        self.assertEqual(selected[0]["canonical_identity"], "C15")

    def test_strict_selector_parser_accepts_only_explicit_decisions(self):
        candidates = [
            {"candidate_id": "M1", "value": "supported"},
            {"candidate_id": "M2", "value": "noise"},
            {"candidate_id": "M3", "value": "rejected"},
        ]

        selected = parse_operation_selection(
            "\n".join(
                [
                    "OPERAND | M1 | grounded duration",
                    "[M2] unrelated library-book count",
                    "REJECT | M3 | unrelated",
                ]
            ),
            candidates,
            "OPERAND",
            strict=True,
        )

        self.assertEqual(
            [candidate["candidate_id"] for candidate in selected],
            ["M1"],
        )

    def test_conversation_words_and_sentence_openers_are_not_entity_anchors(self):
        profile = build_query_profile(
            "Can you remind me what was in my previous conversation?"
        )

        self.assertEqual(profile["required_anchor_groups"], [])

    def test_composite_question_retains_each_real_operand_anchor(self):
        profile = build_query_profile(
            "What is the total cost of my recently purchased headphones and the iPad?"
        )
        anchors = {group["text"] for group in profile["required_anchor_groups"]}

        self.assertIn("headphones", anchors)
        self.assertIn("iPad", anchors)
        self.assertNotIn("recently", anchors)

    def test_composite_anchors_drop_relative_clauses_and_single_predicates(self):
        purchase_profile = build_query_profile(
            "What is the total cost of the car cover and detailing spray I purchased?"
        )
        purchase_anchors = {
            group["text"] for group in purchase_profile["required_anchor_groups"]
        }
        earnings_profile = build_query_profile(
            "What is the total amount of money I earned from selling products?"
        )
        earnings_anchors = {
            group["text"] for group in earnings_profile["required_anchor_groups"]
        }

        self.assertIn("car cover", purchase_anchors)
        self.assertIn("detailing spray", purchase_anchors)
        self.assertNotIn("detailing spray I purchased", purchase_anchors)
        self.assertNotIn("money I earned from", earnings_anchors)

    def test_anchor_filter_allows_linked_same_session_evidence(self):
        question = "Which color did you say the Acme bicycle was?"
        profile = build_query_profile(question)
        candidates = [
            {
                "value": "cobalt blue",
                "head": "speaker assistant",
                "relation": "described color as",
                "tail": "cobalt blue",
                "source_role": "assistant",
                "source_session_id": "session-1",
                "source_quote": "It was cobalt blue.",
            }
        ]
        session_evidence = {
            "session-1": (
                "What color is the Acme bicycle? It was cobalt blue."
            )
        }

        filtered = filter_candidates_by_query_anchors_in_evidence(
            candidates,
            profile,
            evidence_text=session_evidence["session-1"],
            evidence_by_session=session_evidence,
        )

        self.assertEqual([candidate["value"] for candidate in filtered], ["cobalt blue"])

    def test_anchor_filter_still_rejects_wrong_category_evidence(self):
        question = "How many autographed footballs do I own?"
        profile = build_query_profile(question)
        candidates = [
            {
                "value": "6",
                "source_session_id": "session-1",
                "source_quote": "I own 6 autographed baseballs.",
            }
        ]

        filtered = filter_candidates_by_query_anchors_in_evidence(
            candidates,
            profile,
            evidence_text="I own 6 autographed baseballs.",
        )

        self.assertEqual(filtered, [])

    def test_multiword_anchor_tokens_cannot_be_scattered_across_turns(self):
        profile = build_query_profile(
            "How long have I been collecting vintage films?"
        )

        unrelated = anchor_coverage_across_evidence(
            profile,
            ["I collect vintage cameras.", "I record which film each camera uses."],
        )
        exact = anchor_coverage_across_evidence(
            profile,
            ["I have been collecting vintage films for several years."],
        )

        self.assertFalse(unrelated["complete"])
        self.assertTrue(exact["complete"])

    def test_abstention_uses_evidence_coverage_not_parser_success(self):
        question = "Which color did you say the Acme bicycle was?"
        profile = build_query_profile(question)
        coverage = anchor_coverage(
            profile,
            "We discussed the Acme bicycle. It was cobalt blue.",
        )

        decision = determine_safe_abstention(
            {"operation": "none"},
            {"status": "not_applicable"},
            profile,
            coverage,
        )

        self.assertEqual(decision, (False, None))

    def test_abstention_still_blocks_missing_anchor_and_incomplete_operation(self):
        profile = build_query_profile("Which color was the Acme bicycle?")
        missing_anchor = determine_safe_abstention(
            {"operation": "none"},
            {"status": "not_applicable"},
            profile,
            anchor_coverage(profile, "The bicycle was cobalt blue."),
        )
        incomplete_operation = determine_safe_abstention(
            {"operation": "sum"},
            {"status": "insufficient_evidence"},
            {"required_anchor_groups": []},
            {"complete": True},
        )

        self.assertEqual(
            missing_anchor,
            (True, "required_query_anchors_missing_from_evidence"),
        )
        self.assertEqual(
            incomplete_operation,
            (True, "operation_evidence_incomplete"),
        )

    def test_abstention_requires_a_raw_scalar_signal_when_parsing_fails(self):
        profile = build_query_profile(
            "How long have I been collecting vintage films?"
        )
        unsupported_text = "I enjoy collecting vintage films."
        supported_text = "I have collected vintage films for over three years."

        unsupported = determine_safe_abstention(
            {"operation": "none"},
            {"status": "not_applicable"},
            profile,
            anchor_coverage(profile, unsupported_text),
            requested_slot="exact duration",
            selected_evidence_text=unsupported_text,
            answer_slot_candidates=[],
        )
        supported = determine_safe_abstention(
            {"operation": "none"},
            {"status": "not_applicable"},
            profile,
            anchor_coverage(profile, supported_text),
            requested_slot="exact duration",
            selected_evidence_text=supported_text,
            answer_slot_candidates=[],
        )

        self.assertEqual(
            unsupported,
            (True, "requested_answer_type_missing_from_evidence"),
        )
        self.assertEqual(supported, (False, None))

    def test_intent_router_distinguishes_assistant_and_user_requests(self):
        assistant_questions = [
            "What was the seventh recommendation you gave me?",
            "What move did you make after 27. Kg2 Bd5+?",
            "What did Borges say in our previous conversation?",
            "Can you remind me what was the average improvement in framerate?",
        ]
        for question in assistant_questions:
            self.assertEqual(
                infer_query_intent(question)["intent"],
                ASSISTANT_MEMORY_INTENT,
            )
        self.assertNotEqual(
            infer_query_intent("What is the name of my rabbit?")["intent"],
            ASSISTANT_MEMORY_INTENT,
        )
        self.assertNotEqual(
            infer_query_intent(
                "Can you recommend a restaurant based on my preferences?"
            )["intent"],
            ASSISTANT_MEMORY_INTENT,
        )

    def test_intent_router_detects_generic_evidence_synthesis_only(self):
        inference_questions = [
            "What career might suit Rohan based on the Emerald Atlas project?",
            "What gift could support Asha's new running habit?",
            "Which genre would Rohan likely enjoy?",
            "Would Asha probably prefer a national park or an indoor arcade?",
            "Which country did Asha visit for the Amber Atlas project?",
            "What can be inferred from Asha's actions?",
        ]
        for question in inference_questions:
            with self.subTest(question=question):
                intent = infer_query_intent(question)
                self.assertEqual(intent["intent"], EVIDENCE_SYNTHESIS_INTENT)
                self.assertTrue(intent["requires_session_diversity"])

        self.assertEqual(
            infer_query_intent("What city did Asha visit?")["intent"],
            "user_memory",
        )
        self.assertEqual(
            infer_query_intent(
                "Could you recommend a restaurant based on my preferences?"
            )["intent"],
            PREFERENCE_RECOMMENDATION_INTENT,
        )

    def test_synthesis_scope_does_not_require_the_inferred_attribute(self):
        question = "Which country did Asha visit for the Amber Atlas project?"
        base = build_query_profile(
            question,
            known_speakers=["Asha", "Rohan"],
        )

        scope = evidence_synthesis_scope_profile(base)

        self.assertIn(
            "country",
            [group["text"] for group in base["required_anchor_groups"]],
        )
        self.assertNotIn(
            "country",
            [group["text"] for group in scope["required_anchor_groups"]],
        )
        self.assertIn(
            "Amber Atlas",
            [group["text"] for group in scope["required_anchor_groups"]],
        )
        self.assertTrue(scope["allow_distributed_actor_evidence"])

    def test_synthesis_retrieval_is_actor_bound_and_session_diverse(self):
        question = (
            "Based on Asha's actions during the Silver Atlas project, would "
            "Asha likely support another community event?"
        )
        base = build_query_profile(
            question,
            known_speakers=["Asha", "Rohan"],
        )
        profile = evidence_synthesis_scope_profile(base)
        rows = []
        for index in range(5):
            rows.append(
                {
                    **memory(
                        f"During the Silver Atlas project, I completed service task {index + 1}.",
                        f"a{index + 1}",
                        session_id=f"s{index + 1}",
                    ),
                    "speaker": "Asha",
                    "source_speaker": "Asha",
                }
            )
        rows.extend(
            [
                {
                    **memory(
                        "During the Silver Atlas project, I skipped the event.",
                        "r1",
                        session_id="wrong-speaker",
                    ),
                    "speaker": "Rohan",
                    "source_speaker": "Rohan",
                },
                {
                    **memory(
                        "I reorganized my kitchen shelves.",
                        "a-noise",
                        session_id="noise",
                    ),
                    "speaker": "Asha",
                    "source_speaker": "Asha",
                },
            ]
        )

        class RetrievalContext:
            def retrieve_relevant_memories(self, _query, top_k, **_kwargs):
                return rows[:top_k]

        selected, diagnostics = retrieve_evidence_synthesis_memories(
            RetrievalContext(),
            question,
            profile,
            retrieval_top_k=12,
            limit=5,
            max_sessions=5,
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(diagnostics["covered_session_count"], 5)
        self.assertTrue(
            all(item["source_speaker"] == "Asha" for item in selected)
        )
        self.assertTrue(
            all("Silver Atlas" in item["source_quote"] for item in selected)
        )

    def test_synthesis_premises_are_exact_deduplicated_and_provenanced(self):
        question = "What career might suit Asha based on the Emerald Atlas project?"
        profile = evidence_synthesis_scope_profile(
            build_query_profile(
                question,
                known_speakers=["Asha", "Rohan"],
            )
        )
        quote = (
            "For the Emerald Atlas project, I enjoyed teaching children about "
            "plants and protecting habitats."
        )
        rows = [
            extraction(
                quote,
                [],
                turn_id="a1",
                session_id="s1",
                speaker="Asha",
            ),
            extraction(
                quote,
                [],
                turn_id="a2",
                session_id="s2",
                speaker="Asha",
            ),
            extraction(
                "For the Emerald Atlas project, I managed a ticket booth.",
                [],
                turn_id="r1",
                session_id="s3",
                speaker="Rohan",
            ),
        ]

        premises = build_evidence_synthesis_premises(
            rows,
            question,
            profile,
            limit=4,
        )

        self.assertEqual(len(premises), 1)
        self.assertEqual(premises[0]["premise_text"], quote)
        self.assertEqual(premises[0]["source_quote"], quote)
        self.assertEqual(len(premises[0]["provenance"]), 2)

    def test_synthesis_context_obeys_budget_and_suffix_delegates_inference(self):
        context = FakeCandidateExtractionContext()
        premises = [
            {
                "premise_id": f"P{index}",
                "premise_text": f"I completed community service task {index}.",
                "source_turn_id": f"t{index}",
                "source_session_id": f"s{index}",
                "source_speaker": "Asha",
                "source_timestamp": "2027-01-01T10:00:00",
                "evidence_type": "dialogue",
            }
            for index in range(1, 8)
        ]

        rendered, included = format_evidence_synthesis_context(
            context,
            premises,
            token_budget=125,
        )
        suffix = build_pragmos_answer_suffix(
            "Would Asha likely support another community event?",
            "current",
            query_profile={"required_anchor_groups": []},
            query_intent={"intent": EVIDENCE_SYNTHESIS_INTENT},
        )

        self.assertLessEqual(context.count_tokens(rendered), 125)
        self.assertGreater(len(included), 0)
        self.assertLess(len(included), len(premises))
        self.assertIn("ordinary background knowledge", suffix)
        self.assertIn("Infer the concise answer", suffix)

    def test_role_routing_reorders_only_assistant_memory_queries(self):
        rows = [
            memory("I prefer Harbor Cafe.", "turn-user", role="user"),
            memory(
                "I recommended Cedar Bistro.",
                "turn-assistant",
                role="assistant",
            ),
        ]
        assistant_intent = infer_query_intent(
            "Which restaurant did you recommend?"
        )
        user_intent = infer_query_intent("Which restaurant do I prefer?")

        assistant_order = prioritize_memories_for_query_intent(
            rows,
            assistant_intent,
        )
        user_order = prioritize_memories_for_query_intent(rows, user_intent)

        self.assertEqual(assistant_order[0]["role"], "assistant")
        self.assertEqual(user_order, rows)

    def test_assistant_intent_ranks_assistant_authored_candidate_first(self):
        question = "Which restaurant did you recommend?"
        rows = [
            extraction(
                "I recommended Harbor Cafe.",
                [["speaker user", "recommended", "Harbor Cafe"]],
                role="user",
                turn_id="turn-user",
            ),
            extraction(
                "I recommended Cedar Bistro.",
                [["speaker assistant", "recommended", "Cedar Bistro"]],
                role="assistant",
                turn_id="turn-assistant",
            ),
        ]

        candidates = select_answer_slot_candidates(
            rows,
            question,
            infer_requested_answer_slot(question),
            limit=2,
            query_intent=infer_query_intent(question),
        )

        self.assertEqual(candidates[0]["value"], "Cedar Bistro")
        self.assertEqual(candidates[0]["source_role"], "assistant")

    def test_assistant_memory_fallback_extracts_requested_numbered_item(self):
        quote = "\n".join(
            [
                "Here are the recommendations you requested:",
                "1. Alpha Market",
                "2. Birch House",
                "3. Copper Cafe",
                "4. Delta Deli",
                "5. Elm Kitchen",
                "6. Fern Grill",
                "7. Cedar Bistro - quiet tables and vegetarian food",
            ]
        )
        context = FakeCandidateExtractionContext(extraction_output="")
        question = "What was the seventh recommendation you gave me?"
        query_intent = infer_query_intent(question)

        rows, diagnostics = extract_candidates_from_selected_evidence(
            context,
            question,
            "specific fact requested",
            infer_multi_session_operation(question, "single-session-assistant"),
            [memory(quote, "turn-assistant", role="assistant")],
            query_intent=query_intent,
        )

        self.assertIn(
            ["speaker assistant", "list item 7", "Cedar Bistro"],
            rows[0]["triples"],
        )
        self.assertEqual(diagnostics["assistant_fallback_triple_count"], 1)
        self.assertIn("role=assistant text as primary evidence", context.prompts[0])

    def test_assistant_numbered_fallback_rejects_ambiguous_lists(self):
        question = "What was the seventh recommendation you gave me?"
        context = FakeCandidateExtractionContext(extraction_output="")
        first = " ".join(f"{index}. Option {index}" for index in range(1, 8))
        second = " ".join(
            f"{index}. Alternative {index}" for index in range(1, 8)
        )

        rows, diagnostics = extract_candidates_from_selected_evidence(
            context,
            question,
            "specific fact requested",
            infer_multi_session_operation(question, "single-session-assistant"),
            [
                memory(first, "turn-a", "session-a", role="assistant"),
                memory(second, "turn-b", "session-b", role="assistant"),
            ],
            query_intent=infer_query_intent(question),
        )

        self.assertFalse(any(row["triples"] for row in rows))
        self.assertEqual(diagnostics["assistant_fallback_triple_count"], 0)
        self.assertEqual(
            diagnostics["assistant_fallback_ambiguous_value_count"],
            2,
        )

    def test_ordinal_list_recall_uses_anchored_assistant_evidence(self):
        question = (
            "In the remote-job list for my Fern Court plan, what was item 6?"
        )
        rows = [
            extraction(
                "For my Fern Court plan, please give me ten unusual remote job "
                "ideas in a numbered list.",
                [],
                turn_id="user-request",
                session_id="answer-session",
            ),
            extraction(
                "Here is the numbered list: 1. archive researcher 2. museum "
                "caption writer 3. accessibility tester 4. community newsletter "
                "editor 5. online language tutor 6. podcast transcript reviewer "
                "7. digital pattern designer.",
                [],
                role="assistant",
                turn_id="assistant-list",
                session_id="answer-session",
            ),
            extraction(
                "Here is another list: 1. one 2. two 3. three 4. four 5. five "
                "6. wrong value.",
                [],
                role="assistant",
                turn_id="noise-list",
                session_id="noise-session",
            ),
        ]

        result = resolve_deterministic_ordinal_list_answer(
            question,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "podcast transcript reviewer")
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["rejected_session_count"], 1)
        self.assertEqual(
            result["candidates"][0]["source_turn_id"],
            "assistant-list",
        )

    def test_ordinal_list_recall_abstains_from_conflicting_anchored_lists(self):
        question = "For my Harbor Annex plan, what was item 2 in the list?"
        rows = []
        for suffix, value in (("a", "Birch House"), ("b", "Copper Cafe")):
            rows.extend(
                [
                    extraction(
                        "For my Harbor Annex plan, create a numbered list.",
                        [],
                        turn_id=f"request-{suffix}",
                        session_id=f"session-{suffix}",
                    ),
                    extraction(
                        f"1. Alpha Market 2. {value} 3. Delta Deli",
                        [],
                        role="assistant",
                        turn_id=f"list-{suffix}",
                        session_id=f"session-{suffix}",
                    ),
                ]
            )

        result = resolve_deterministic_ordinal_list_answer(
            question,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "ambiguous_list_item")
        self.assertIsNone(result["answer"])
        self.assertEqual(
            set(result["candidate_values"]),
            {"Birch House", "Copper Cafe"},
        )

    def test_ordinal_list_fallback_does_not_require_general_assistant_intent(self):
        quote = "1. archive researcher 2. museum writer 3. transcript reviewer"
        question = "In the remote-job list for my Fern Court plan, what was item 3?"
        context = FakeCandidateExtractionContext(extraction_output="")
        query_intent = infer_query_intent(question)

        rows, diagnostics = extract_candidates_from_selected_evidence(
            context,
            question,
            "specific fact requested",
            infer_multi_session_operation(question, "single-session-assistant"),
            [memory(quote, "turn-assistant", role="assistant")],
            query_intent=query_intent,
        )

        self.assertNotEqual(query_intent["intent"], ASSISTANT_MEMORY_INTENT)
        self.assertIn(
            ["speaker assistant", "list item 3", "transcript reviewer"],
            rows[0]["triples"],
        )
        self.assertEqual(diagnostics["assistant_fallback_triple_count"], 1)

    def test_typed_color_answer_returns_only_the_grounded_color_span(self):
        question = (
            "In The Quiet Cartographer, what color did you say the moon moth "
            "was in the illustration?"
        )
        rows = [
            extraction(
                "In The Quiet Cartographer, describe the moon moth in the "
                "illustration.",
                [],
                turn_id="request-turn",
                session_id="answer-session",
            ),
            extraction(
                "In The Quiet Cartographer, the moon moth has a forest green "
                "body with small white markings.",
                [],
                role="assistant",
                turn_id="answer-turn",
                session_id="answer-session",
            ),
        ]

        result = resolve_typed_answer_slot(
            question,
            infer_requested_answer_slot(question),
            rows,
            build_query_profile(question),
            infer_query_intent(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "forest green")
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["source_turn_id"], "answer-turn")

    def test_typed_requested_list_removes_only_a_grounded_intro(self):
        question = (
            "What were the three objectives you outlined for the Cool Streets "
            "Study at Elm Street Library?"
        )
        rows = [
            extraction(
                "Please outline three objectives for the Cool Streets Study at "
                "Elm Street Library.",
                [],
                turn_id="request-turn",
                session_id="answer-session",
            ),
            extraction(
                "The three objectives are: publish an open street-level dataset; "
                "map neighborhood heat islands; compare tree canopy coverage.",
                [],
                role="assistant",
                turn_id="answer-turn",
                session_id="answer-session",
            ),
        ]

        result = resolve_typed_answer_slot(
            question,
            infer_requested_answer_slot(question),
            rows,
            build_query_profile(question),
            infer_query_intent(question),
        )

        self.assertEqual(infer_requested_answer_slot(question), "requested list")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(
            result["answer"],
            "publish an open street-level dataset; map neighborhood heat "
            "islands; compare tree canopy coverage",
        )

    def test_typed_answer_slot_keeps_conflicting_colors_ambiguous(self):
        question = "What color did you say the moon moth was?"
        rows = [
            extraction(
                "The moon moth has a forest green body.",
                [],
                role="assistant",
                turn_id="green-turn",
                session_id="green-session",
            ),
            extraction(
                "The moon moth has a marigold yellow body.",
                [],
                role="assistant",
                turn_id="yellow-turn",
                session_id="yellow-session",
            ),
        ]

        result = resolve_typed_answer_slot(
            question,
            infer_requested_answer_slot(question),
            rows,
            build_query_profile(question),
            infer_query_intent(question),
        )

        self.assertEqual(result["status"], "ambiguous_typed_value")
        self.assertIsNone(result["answer"])
        self.assertEqual(
            set(result["candidate_values"]),
            {"forest green", "marigold yellow"},
        )

    def test_assistant_speaker_entity_is_grounded_by_provenance_role(self):
        records = [
            {
                "evidence_id": "E1",
                "source_role": "assistant",
                "source_speaker": "assistant",
                "source_quote": "Cedar Bistro is the strongest recommendation.",
            }
        ]

        parsed = parse_selected_evidence_triples(
            "E1 | speaker assistant | recommended | Cedar Bistro",
            records,
        )

        self.assertEqual(
            parsed["E1"],
            [["speaker assistant", "recommended", "Cedar Bistro"]],
        )


class PreferenceSynthesisTests(unittest.TestCase):
    def test_profile_extracts_include_and_avoid_only_from_user_evidence(self):
        rows = [
            extraction(
                "I prefer a quiet hotel with an ocean view. Please avoid large chains.",
                [],
                role="user",
                turn_id="turn-1",
            ),
            extraction(
                "I recommend the Large Chain Hotel.",
                [],
                role="assistant",
                turn_id="turn-2",
            ),
        ]

        profile = build_preference_profile(rows, "Can you recommend a hotel?")

        self.assertTrue(profile["applicable"])
        self.assertEqual(
            [item["text"] for item in profile["include_constraints"]],
            ["a quiet hotel with an ocean view"],
        )
        self.assertEqual(
            [item["text"] for item in profile["avoid_constraints"]],
            ["large chains"],
        )
        self.assertTrue(
            all(item["source_turn_id"] == "turn-1" for item in profile["constraints"])
        )

    def test_hypothetical_language_is_not_promoted_to_a_preference(self):
        rows = [
            extraction(
                "I might prefer a rooftop pool someday, but I have not decided.",
                [],
                role="user",
            )
        ]

        profile = build_preference_profile(rows)

        self.assertFalse(profile["applicable"])
        self.assertEqual(profile["constraints"], [])

    def test_explicit_preference_update_retains_superseded_history(self):
        rows = [
            extraction(
                "I used to prefer cats, but now prefer dogs.",
                [],
                role="user",
                turn_id="turn-7",
            )
        ]

        profile = build_preference_profile(rows)

        self.assertEqual(
            [item["text"] for item in profile["active_constraints"]],
            ["dogs"],
        )
        old = next(item for item in profile["constraints"] if item["text"] == "cats")
        new = next(item for item in profile["constraints"] if item["text"] == "dogs")
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(new["supersedes_constraint_id"], old["constraint_id"])

    def test_later_exact_contradiction_wins_by_timestamp(self):
        rows = [
            extraction(
                "Please avoid rooftop pools.",
                [],
                turn_id="turn-2",
                timestamp="2026-02-01T10:00:00Z",
            ),
            extraction(
                "I now like rooftop pools.",
                [],
                turn_id="turn-9",
                timestamp="2026-03-01T10:00:00Z",
            ),
        ]

        profile = build_preference_profile(rows)

        self.assertEqual(profile["conflict_count"], 1)
        self.assertEqual(len(profile["active_constraints"]), 1)
        self.assertEqual(profile["active_constraints"][0]["polarity"], "include")
        self.assertEqual(profile["active_constraints"][0]["text"], "rooftop pools")

    def test_profile_format_and_answer_suffix_preserve_provenance(self):
        profile = build_preference_profile(
            [
                extraction(
                    "I prefer vegetarian Korean food. Please avoid nuts.",
                    [],
                    turn_id="turn-4",
                    session_id="session-dinner",
                )
            ]
        )
        formatted = format_preference_profile(profile)
        suffix = build_pragmos_answer_suffix(
            "Can you recommend a restaurant?",
            "current",
            query_intent=infer_query_intent("Can you recommend a restaurant?"),
            preference_profile=profile,
        )

        self.assertIn("INCLUDE (soft): vegetarian Korean food", formatted)
        self.assertIn("AVOID (hard): nuts", formatted)
        self.assertIn("turn=turn-4; session=session-dinner", formatted)
        self.assertIn("Synthesize a concise recommendation", suffix)

    def test_synthetic_style_constraints_generalize_across_domains(self):
        cases = [
            (
                "advanced color grading and keyboard workflows",
                "generic beginner tutorials",
            ),
            ("low-impact swimming and mobility", "high-impact running"),
            ("an offline app with no social feed", "subscription-only apps"),
            ("interpretable machine learning in healthcare", "marketing conferences"),
        ]
        for index, (positive, negative) in enumerate(cases):
            with self.subTest(index=index):
                profile = build_preference_profile(
                    [
                        extraction(
                            f"For future recommendations, I prefer {positive}. "
                            f"Please avoid {negative}.",
                            [],
                            turn_id=f"turn-{index}",
                        )
                    ]
                )
                self.assertEqual(profile["include_constraints"][0]["text"], positive)
                self.assertEqual(profile["avoid_constraints"][0]["text"], negative)


class LatencyAndAblationTests(unittest.TestCase):
    def test_ablation_set_is_explicit_and_validated(self):
        args = SimpleNamespace(pragmos_ablate=["graph", "reranker"])
        self.assertEqual(pragmos_ablation_set(args), {"graph", "reranker"})
        with self.assertRaises(ValueError):
            pragmos_ablation_set(SimpleNamespace(pragmos_ablate=["unknown"]))

    def test_counter_delta_only_reports_cache_counters(self):
        before = {"embedding_hits": 2, "embedding_misses": 3, "cache_entries": 9}
        after = {"embedding_hits": 5, "embedding_misses": 4, "cache_entries": 10}
        self.assertEqual(
            counter_delta(after, before),
            {"embedding_hits": 3, "embedding_misses": 1},
        )

    def test_candidate_extraction_ablation_keeps_all_evidence_and_graph_rows(self):
        memories = [
            memory("I prefer quiet hotels.", "turn-1"),
            memory("Please avoid large chains.", "turn-2"),
        ]
        existing = [
            extraction(
                "I prefer quiet hotels.",
                [["user", "prefers", "quiet hotels"]],
                turn_id="turn-1",
            )
        ]

        rows, diagnostics = evidence_rows_without_candidate_extraction(
            memories,
            existing_extractions=existing,
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(diagnostics["all_selected_evidence_represented"])
        self.assertEqual(diagnostics["candidate_extraction_batch_count"], 0)
        self.assertEqual(rows[0]["triples"], [["user", "prefers", "quiet hotels"]])
        self.assertEqual(rows[1]["triples"], [])


class StructuredAnswerCandidateTests(unittest.TestCase):
    def test_duration_canonicalization_restores_grounded_qualifier(self):
        candidates = [
            {
                "value": "Over a year",
                "source_role": "user",
                "source_quote": "Over a year of uncertainty was really tough.",
                "quote_supported": True,
            }
        ]

        answer = canonicalize_generated_scalar_answer(
            "a year",
            "exact duration",
            candidates,
        )

        self.assertEqual(answer, "Over a year")

    def test_duration_canonicalization_supports_general_qualifiers(self):
        candidates = [
            {
                "value": "at least three months",
                "source_role": "user",
                "source_quote": "I waited at least three months for approval.",
                "quote_supported": True,
            }
        ]

        answer = canonicalize_generated_scalar_answer(
            "three months",
            "exact duration",
            candidates,
        )

        self.assertEqual(answer, "at least three months")

    def test_duration_canonicalization_does_not_replace_a_different_answer(self):
        candidates = [
            {
                "value": "over a year",
                "source_role": "user",
                "source_quote": "I waited over a year for approval.",
                "quote_supported": True,
            }
        ]

        self.assertEqual(
            canonicalize_generated_scalar_answer(
                "two years",
                "exact duration",
                candidates,
            ),
            "two years",
        )
        self.assertEqual(
            canonicalize_generated_scalar_answer(
                "a year for the decision",
                "exact duration",
                candidates,
            ),
            "a year for the decision",
        )

    def test_duration_canonicalization_requires_direct_user_provenance(self):
        base_candidate = {
            "value": "over a year",
            "source_quote": "I waited over a year for approval.",
            "quote_supported": True,
        }
        for source_role, quote_supported in (("assistant", True), ("user", False)):
            candidate = dict(
                base_candidate,
                source_role=source_role,
                quote_supported=quote_supported,
            )
            self.assertEqual(
                canonicalize_generated_scalar_answer(
                    "a year",
                    "exact duration",
                    [candidate],
                ),
                "a year",
            )

    def test_occupation_question_uses_a_dedicated_answer_slot(self):
        self.assertEqual(
            infer_requested_answer_slot("What was my previous occupation?"),
            "occupation or role",
        )
        self.assertEqual(
            infer_requested_answer_slot("What is my current job title?"),
            "occupation or role",
        )

    def test_previous_occupation_uses_role_as_not_role_at(self):
        question = "What was my previous occupation?"
        rows = [
            extraction(
                "I've used Trello in my previous role as a marketing "
                "specialist at a small startup, and I'm familiar with its boards.",
                [["speaker user", "used in previous role", "Trello"]],
                turn_id="turn-previous",
            ),
            extraction(
                "I started a new role as a senior marketing analyst. In my "
                "previous role at the startup, I used ClickUp.",
                [
                    ["speaker user", "role", "senior marketing analyst"],
                    ["speaker user", "previous role", "startup"],
                ],
                turn_id="turn-current",
            ),
        ]

        candidates = select_answer_slot_candidates(
            rows,
            question,
            infer_requested_answer_slot(question),
            limit=12,
            query_profile=build_query_profile(question),
        )
        candidates = filter_candidates_by_query_anchors(
            candidates,
            build_query_profile(question),
        )

        self.assertEqual(candidates[0]["value"], "marketing specialist at a small startup")
        self.assertEqual(candidates[0]["source_turn_id"], "turn-previous")
        self.assertNotIn("startup", [candidate["value"] for candidate in candidates])
        self.assertNotIn("Trello", [candidate["value"] for candidate in candidates])

    def test_current_occupation_excludes_the_previous_role(self):
        question = "What is my current occupation?"
        rows = [
            extraction(
                "My previous role as a marketing specialist ended last year.",
                [["speaker user", "previous role", "marketing specialist"]],
                turn_id="turn-previous",
            ),
            extraction(
                "I started a new role as a senior marketing analyst.",
                [["speaker user", "new role", "senior marketing analyst"]],
                turn_id="turn-current",
            ),
        ]

        candidates = select_answer_slot_candidates(
            rows,
            question,
            infer_requested_answer_slot(question),
            limit=12,
        )

        self.assertEqual(candidates[0]["value"], "senior marketing analyst")
        self.assertEqual(candidates[0]["source_turn_id"], "turn-current")
        self.assertNotIn(
            "marketing specialist",
            [candidate["value"] for candidate in candidates],
        )

    def test_role_at_workplace_is_not_an_occupation(self):
        question = "What was my previous occupation?"
        rows = [
            extraction(
                "In my previous role at the startup, I used ClickUp.",
                [["speaker user", "previous role", "startup"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            question,
            infer_requested_answer_slot(question),
        )

        self.assertEqual(candidates, [])

    def test_verbose_scalar_answer_is_canonicalized_to_grounded_value(self):
        candidates = [
            {
                "value": "20",
                "source_quote": "I have 20 playlists on SoundWave.",
            }
        ]

        answer = canonicalize_generated_scalar_answer(
            "20 playlists on SoundWave",
            "exact quantity or amount",
            candidates,
        )

        self.assertEqual(answer, "20")

    def test_non_scalar_answer_is_not_canonicalized(self):
        candidates = [
            {
                "value": "Luna",
                "source_quote": "My cat is named Luna.",
            }
        ]

        answer = canonicalize_generated_scalar_answer(
            "My cat is named Luna",
            "exact name or title",
            candidates,
        )

        self.assertEqual(answer, "My cat is named Luna")

    def test_hard_anchor_checks_the_full_quote_not_the_prompt_excerpt(self):
        quote = (
            "I discussed music organization in detail for quite a while. " * 5
            + "I have 20 playlists on SoundWave."
        )
        question = "How many playlists do I have on SoundWave?"
        profile = build_query_profile(question)

        candidates = select_answer_slot_candidates(
            [extraction(quote, [["speaker user", "has", "20 playlists"]])],
            question,
            "exact quantity or amount",
            limit=12,
            query_profile=profile,
        )
        filtered = filter_candidates_by_query_anchors(candidates, profile)

        self.assertEqual(filtered[0]["value"], "20")
        self.assertIn("SoundWave", filtered[0]["source_quote"])
        self.assertNotIn("SoundWave", filtered[0]["source_quote_excerpt"])

    def test_duplicate_value_keeps_the_anchor_supported_provenance(self):
        question = "How long did it take me to assemble the Acme bookshelf?"
        profile = build_query_profile(question)
        rows = [
            extraction(
                "Furniture assembly can take 4 hours.",
                [["Acme bookshelf", "assembly time", "4 hours"]],
                role="assistant",
                turn_id="turn-generic",
            ),
            extraction(
                "I assembled the Acme bookshelf in 4 hours.",
                [["speaker user", "assembled in", "4 hours"]],
                turn_id="turn-supported",
            ),
        ]

        candidates = select_answer_slot_candidates(
            rows,
            question,
            "exact duration",
            limit=12,
            query_profile=profile,
        )
        filtered = filter_candidates_by_query_anchors(candidates, profile)

        self.assertEqual(filtered[0]["value"], "4 hours")
        self.assertEqual(filtered[0]["source_turn_id"], "turn-supported")

    def test_qualified_duration_is_preserved(self):
        rows = [
            extraction(
                "I waited over a year for the application decision.",
                [["speaker user", "waited", "over a year"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "How long did I wait for the application decision?",
            "exact duration",
        )

        self.assertEqual(candidates[0]["value"], "over a year")

    def test_ratio_question_extracts_the_ratio_value(self):
        question = "What is my preferred tea-to-water ratio?"
        rows = [
            extraction(
                "My preferred tea-to-water ratio is 1:4.",
                [["speaker user", "preferred tea-to-water ratio", "1:4"]],
            )
        ]

        self.assertEqual(
            infer_requested_answer_slot(question),
            "exact quantity or amount",
        )
        candidates = select_answer_slot_candidates(
            rows,
            question,
            infer_requested_answer_slot(question),
        )
        self.assertEqual(candidates[0]["value"], "1:4")

    def test_location_relation_supports_a_single_word_place(self):
        question = "Where does my sister Mara live?"
        profile = build_query_profile(question)
        rows = [
            extraction(
                "I am visiting my sister Mara in Denver.",
                [["sister Mara", "located in", "Denver"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            question,
            "place or organization",
            query_profile=profile,
        )
        candidates = filter_candidates_by_query_anchors(candidates, profile)

        self.assertEqual(candidates[0]["value"], "Denver")

    def test_possessive_anchor_skips_modifiers(self):
        occupation_anchors = [
            group["text"]
            for group in build_query_profile(
                "What was my previous occupation?"
            )["required_anchor_groups"]
        ]
        ratio_anchors = [
            group["text"]
            for group in build_query_profile(
                "What is my preferred gin-to-vermouth ratio?"
            )["required_anchor_groups"]
        ]

        self.assertIn("occupation", occupation_anchors)
        self.assertNotIn("previous", occupation_anchors)
        self.assertIn("gin-to-vermouth", ratio_anchors)
        self.assertNotIn("preferred", ratio_anchors)

    def test_quantity_uses_the_query_related_number(self):
        rows = [
            extraction(
                "I packed 9 jackets and 4 scarves for the mountain trip.",
                [["speaker user", "packed 9 jackets and 4 scarves", "mountain trip"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "How many jackets did I pack for the mountain trip?",
            "exact quantity or amount",
        )

        self.assertEqual(candidates[0]["value"], "9")
        self.assertGreater(
            candidates[0]["candidate_strength"],
            next(item for item in candidates if item["value"] == "4")[
                "candidate_strength"
            ],
        )

    def test_duration_preserves_qualifier(self):
        rows = [
            extraction(
                "The train ride takes 35 minutes each way.",
                [["train ride", "takes", "35 minutes each way"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "How long does the train ride take?",
            "exact duration",
        )

        self.assertEqual(candidates[0]["value"], "35 minutes each way")

    def test_specific_attribute_uses_object_value(self):
        rows = [
            extraction(
                "Nova is a Border Collie.",
                [["Nova", "breed", "Border Collie"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "What breed is Nova?",
            "specific fact requested",
        )

        self.assertEqual(candidates[0]["value"], "Border Collie")

    def test_person_question_can_select_the_subject(self):
        rows = [
            extraction(
                "Mara gave me the blue notebook.",
                [["Mara", "gave", "blue notebook"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "Who gave me the blue notebook?",
            "person or group",
        )

        self.assertEqual(candidates[0]["value"], "Mara")

    def test_direct_user_fact_outranks_assistant_example(self):
        rows = [
            extraction(
                "You could pursue the Certified Metrics Specialist credential.",
                [["speaker user", "could pursue", "Certified Metrics Specialist"]],
                role="assistant",
                turn_id="turn-assistant",
            ),
            extraction(
                "I am pursuing a certification in Applied Analytics.",
                [["speaker user", "certification", "Applied Analytics"]],
                role="user",
                turn_id="turn-user",
            ),
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "What certification am I pursuing?",
            "specific fact requested",
        )

        self.assertEqual(candidates[0]["value"], "Applied Analytics")
        self.assertEqual(candidates[0]["source_role"], "user")

    def test_name_and_place_are_not_special_case_only_paths(self):
        rows = [
            extraction(
                "I listen using SoundWave.",
                [["speaker user", "uses", "SoundWave"]],
                turn_id="turn-service",
            ),
            extraction(
                "We attended the celebration at Cedar Hall.",
                [["speaker user", "attended celebration", "Cedar Hall"]],
                turn_id="turn-place",
            ),
        ]

        name_candidates = select_answer_slot_candidates(
            rows,
            "What music service do I use?",
            "exact name or title",
        )
        place_candidates = select_answer_slot_candidates(
            rows,
            "Where was the celebration held?",
            "place or organization",
        )

        self.assertEqual(name_candidates[0]["value"], "SoundWave")
        self.assertEqual(place_candidates[0]["value"], "Cedar Hall")

    def test_formatted_candidate_keeps_fact_and_provenance(self):
        rows = [
            extraction(
                "Nova is a Border Collie.",
                [["Nova", "breed", "Border Collie"]],
            )
        ]
        candidates = select_answer_slot_candidates(
            rows,
            "What breed is Nova?",
            "specific fact requested",
        )

        formatted = format_answer_slot_candidates(
            candidates,
            "specific fact requested",
        )

        self.assertIn("Nova -[breed]-> Border Collie", formatted)
        self.assertIn("source_role: user", formatted)
        self.assertIn('source_quote: "Nova is a Border Collie."', formatted)

    def test_place_candidate_can_be_cleaned_from_a_graph_phrase(self):
        rows = [
            extraction(
                "I completed my degree at Northbridge University.",
                [["speaker user", "completed degree", "degree from Northbridge University"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "Where did I complete my degree?",
            "place or organization",
        )

        self.assertEqual(candidates[0]["value"], "Northbridge University")

    def test_place_candidate_uses_the_last_location_preposition(self):
        rows = [
            extraction(
                "Can you suggest things to do on Amber Island?",
                [["speaker user", "asked for", "things to do on Amber Island"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "Where did I ask about activities?",
            "place or organization",
        )

        self.assertEqual(candidates[0]["value"], "Amber Island")

    def test_academic_field_is_not_treated_as_a_place(self):
        rows = [
            extraction(
                "I am considering a degree in Data Engineering.",
                [["speaker user", "considering degree in", "Data Engineering"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "Where did I complete my degree?",
            "place or organization",
        )

        self.assertEqual(candidates, [])

    def test_acronym_shaped_organization_is_a_place_candidate(self):
        rows = [
            extraction(
                "My new desk is from HFG, and it fits perfectly.",
                [["new desk", "is from", "HFG"]],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "Where did I buy my new desk from?",
            "place or organization",
        )

        self.assertEqual(candidates[0]["value"], "HFG")

    def test_local_semantic_reranker_can_correct_lexical_order(self):
        class FakeReranker:
            def predict(self, pairs):
                return [-5.0, 5.0]

        class FakeContextLayer:
            def get_local_reranker(self):
                return FakeReranker()

        candidates = [
            {
                "value": "VideoBox",
                "head": "speaker user",
                "relation": "uses",
                "tail": "VideoBox",
                "source_role": "user",
                "source_quote": "I use VideoBox to watch films.",
                "candidate_strength": 10.0,
                "quote_supported": True,
            },
            {
                "value": "TuneCloud",
                "head": "speaker user",
                "relation": "listens on",
                "tail": "TuneCloud",
                "source_role": "user",
                "source_quote": "I listen to songs on TuneCloud.",
                "candidate_strength": 6.0,
                "quote_supported": True,
            },
        ]

        reranked = rerank_answer_slot_candidates(
            FakeContextLayer(),
            "What music service do I use?",
            candidates,
            limit=2,
        )

        self.assertEqual(reranked[0]["value"], "TuneCloud")
        self.assertEqual(reranked[0]["candidate_reranker"], "local_cross_encoder")


class SelectedEvidenceCandidateTests(unittest.TestCase):
    def test_batched_extraction_visits_every_selected_evidence_id(self):
        memories = [
            memory(
                f"Evidence item {index} contains " + ("detail " * 45),
                f"turn-{index}",
                session_id=f"session-{index}",
            )
            for index in range(1, 8)
        ]
        context = FakeCandidateExtractionContext()
        context.context_length = 1100

        rows, diagnostics = extract_candidates_from_selected_evidence(
            context,
            "What detail was recorded?",
            "specific fact requested",
            infer_multi_session_operation(
                "What detail was recorded?",
                "single-session-user",
            ),
            memories,
        )

        prompts = "\n".join(context.prompts)
        self.assertGreater(diagnostics["candidate_extraction_batch_count"], 1)
        self.assertEqual(len(rows), len(memories))
        self.assertTrue(diagnostics["all_selected_evidence_represented"])
        for index in range(1, len(memories) + 1):
            self.assertIn(f"[E{index}]", prompts)

    def test_candidate_fact_from_fifth_selected_evidence_is_not_dropped(self):
        memories = [
            memory("I discussed routine topic one.", "turn-1"),
            memory("I discussed routine topic two.", "turn-2"),
            memory("I discussed routine topic three.", "turn-3"),
            memory("I discussed routine topic four.", "turn-4"),
            memory("Nova's preferred snack is mango.", "turn-5", "session-2"),
        ]
        existing = [
            extraction(
                item["source_quote"],
                [["speaker user", "discussed", f"topic {index}"]],
                turn_id=item["source_turn_ids"][0],
                session_id=item["source_session_id"],
            )
            for index, item in enumerate(memories[:4], start=1)
        ]
        context = FakeCandidateExtractionContext(
            extraction_output="E5 | Nova | preferred snack | mango"
        )
        question = "What snack does Nova prefer?"

        rows, diagnostics = extract_candidates_from_selected_evidence(
            context,
            question,
            "specific fact requested",
            infer_multi_session_operation(question, "single-session-user"),
            memories,
            existing_extractions=existing,
        )
        candidates = select_answer_slot_candidates(
            rows,
            question,
            "specific fact requested",
            limit=12,
        )
        candidates = filter_candidates_by_query_anchors(
            candidates,
            build_query_profile(question),
        )

        self.assertEqual(len(rows), 5)
        self.assertTrue(diagnostics["all_selected_evidence_represented"])
        self.assertEqual(diagnostics["reused_graph_extraction_count"], 4)
        self.assertEqual(diagnostics["batch_extracted_evidence_count"], 1)
        self.assertEqual(candidates[0]["value"], "mango")
        self.assertEqual(candidates[0]["source_turn_id"], "turn-5")
        self.assertIn("[E5]", context.prompts[0])

    def test_wrong_entity_and_ungrounded_candidate_still_abstain(self):
        memories = [
            memory("My parrot is named Pico.", "turn-parrot"),
            memory("My rabbit likes fresh hay.", "turn-rabbit", "session-2"),
        ]
        context = FakeCandidateExtractionContext(
            extraction_output=(
                "E1 | Pico | name of | rabbit\n"
                "E2 | my rabbit | name | Zuzu"
            )
        )
        question = "What is the name of my rabbit?"

        rows, _diagnostics = extract_candidates_from_selected_evidence(
            context,
            question,
            "exact name or title",
            infer_multi_session_operation(question, "single-session-user"),
            memories,
        )
        candidates = select_answer_slot_candidates(
            rows,
            question,
            "exact name or title",
            limit=12,
        )
        candidates = filter_candidates_by_query_anchors(
            candidates,
            build_query_profile(question),
        )

        self.assertEqual(candidates, [])

    def test_multisession_sum_uses_operand_beyond_preliminary_four(self):
        memories = [
            memory(
                "Traveling the north route took 2 hours.",
                "turn-1",
                "session-1",
            ),
            memory("I packed water for the trip.", "turn-2", "session-1"),
            memory("I checked the weather before leaving.", "turn-3", "session-1"),
            memory("I brought a paper map.", "turn-4", "session-1"),
            memory(
                "Traveling the south route took 90 minutes.",
                "turn-5",
                "session-2",
            ),
        ]
        existing = [
            extraction(
                item["source_quote"],
                [],
                turn_id=item["source_turn_ids"][0],
                session_id=item["source_session_id"],
            )
            for item in memories[:4]
        ]
        question = "How many hours in total did I spend traveling?"
        plan = infer_multi_session_operation(question, "multi-session")
        context = FakeCandidateExtractionContext(
            extraction_output="",
            selector_output=(
                "OPERAND | M1 | north route\n"
                "OPERAND | M2 | south route"
            ),
        )

        rows, diagnostics = extract_candidates_from_selected_evidence(
            context,
            question,
            "exact quantity or amount",
            plan,
            memories,
            existing_extractions=existing,
        )
        result = execute_operation_plan(
            context,
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(diagnostics["selected_evidence_count"], 5)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "3.5 hours")
        self.assertEqual(
            {fact["source_session_id"] for fact in result["facts"]},
            {"session-1", "session-2"},
        )


class RetrievalSafetyTests(unittest.TestCase):
    def test_longmemeval_retrieval_metrics_match_official_binary_formulas(self):
        record = {
            "question_id": "question-1",
            "question_type": "multi-session",
            "answer_session_ids": ["s2", "s4"],
        }
        ranking = [
            {"source_session_id": "s1"},
            {"source_session_id": "s2"},
            {"source_session_id": "s3"},
            {"source_session_id": "s4"},
        ]

        result = longmemeval_session_retrieval_metrics(record, ranking)

        self.assertEqual(result["metrics"]["recall@1"], 0.0)
        self.assertEqual(result["metrics"]["recall_any@1"], 0.0)
        self.assertEqual(result["metrics"]["recall@5"], 1.0)
        self.assertEqual(result["metrics"]["recall_all@5"], 1.0)
        self.assertEqual(result["metrics"]["recall_any@5"], 1.0)
        self.assertAlmostEqual(result["metrics"]["ndcg@5"], 0.75)
        self.assertEqual(
            result["metrics"]["ndcg@5"],
            result["metrics"]["ndcg_any@5"],
        )

    def test_longmemeval_session_ranking_deduplicates_chunks(self):
        record = {
            "question_id": "question-2",
            "question_type": "single-session-user",
            "answer_session_ids": ["answer-session"],
        }
        ranking = [
            {"source_session_id": "noise-session", "memory_id": "chunk-1"},
            {"source_session_id": "noise-session", "memory_id": "chunk-2"},
            {"source_session_id": "answer-session", "memory_id": "chunk-3"},
        ]

        result = longmemeval_session_retrieval_metrics(
            record,
            ranking,
            ks=(1, 2),
        )

        self.assertEqual(
            result["ranked_session_ids"],
            ["noise-session", "answer-session"],
        )
        self.assertEqual(result["metrics"]["recall@1"], 0.0)
        self.assertEqual(result["metrics"]["recall@2"], 1.0)

    def test_longmemeval_retrieval_metrics_exclude_abstention(self):
        result = longmemeval_session_retrieval_metrics(
            {
                "question_id": "question_abs",
                "question_type": "single-session-user",
                "answer_session_ids": [],
            },
            [],
        )

        self.assertFalse(result["eligible"])
        self.assertEqual(result["exclusion_reason"], "abstention_question")
        self.assertEqual(result["metrics"], {})

    def test_longmemeval_retrieval_aggregation_is_macro_averaged(self):
        first = longmemeval_session_retrieval_metrics(
            {
                "question_id": "q1",
                "question_type": "single-session-user",
                "answer_session_ids": ["s1"],
            },
            [{"source_session_id": "s1"}],
            ks=(1,),
        )
        second = longmemeval_session_retrieval_metrics(
            {
                "question_id": "q2",
                "question_type": "multi-session",
                "answer_session_ids": ["s2"],
            },
            [{"source_session_id": "noise"}],
            ks=(1,),
        )
        excluded = longmemeval_session_retrieval_metrics(
            {
                "question_id": "q3_abs",
                "question_type": "multi-session",
                "answer_session_ids": [],
            },
            [],
            ks=(1,),
        )

        aggregate = aggregate_longmemeval_retrieval_metrics(
            [first, second, excluded]
        )

        self.assertEqual(aggregate["question_count"], 2)
        self.assertEqual(aggregate["excluded_question_count"], 1)
        self.assertEqual(aggregate["metrics"]["recall@1"], 0.5)
        self.assertEqual(
            aggregate["by_question_type"]["multi-session"]["metrics"][
                "recall@1"
            ],
            0.0,
        )

    def test_anchor_filter_rejects_a_nearby_but_different_entity(self):
        rows = [
            extraction(
                "My parrot is named Pico.",
                [["my parrot", "name", "Pico"]],
            )
        ]
        candidates = select_answer_slot_candidates(
            rows,
            "What is the name of my rabbit?",
            "exact name or title",
        )

        filtered = filter_candidates_by_query_anchors(
            candidates,
            build_query_profile("What is the name of my rabbit?"),
        )

        self.assertEqual(filtered, [])

    def test_brand_slot_rejects_a_product_scent(self):
        rows = [
            extraction(
                "I bought FreshMart lavender shampoo.",
                [
                    ["shampoo", "brand bought at", "FreshMart"],
                    ["shampoo", "scent", "lavender"],
                ],
            )
        ]

        candidates = select_answer_slot_candidates(
            rows,
            "What brand of shampoo did I buy?",
            infer_requested_answer_slot("What brand of shampoo did I buy?"),
        )

        self.assertEqual(candidates[0]["value"], "FreshMart")
        self.assertNotIn("lavender", [item["value"] for item in candidates])

    def test_session_diversity_round_robins_before_reusing_a_session(self):
        memories = [
            {
                "memory_id": "m1",
                "source_session_id": "s1",
                "source_quote": "I repaired my bicycle wheel.",
                "score": 0.9,
                "role": "user",
            },
            {
                "memory_id": "m2",
                "source_session_id": "s1",
                "source_quote": "I repaired my bicycle chain.",
                "score": 0.8,
                "role": "user",
            },
            {
                "memory_id": "m3",
                "source_session_id": "s2",
                "source_quote": "I repaired my bicycle brake.",
                "score": 0.7,
                "role": "user",
            },
        ]

        selected = select_session_diverse_memories(
            memories,
            build_query_profile("How many bicycle repairs did I complete?"),
            limit=3,
        )

        self.assertEqual(
            [item["source_session_id"] for item in selected[:2]],
            ["s1", "s2"],
        )

    def test_retrieval_metrics_distinguish_partial_and_complete_coverage(self):
        record = {"answer_session_ids": ["s1", "s2", "s3"]}
        memories = [
            {"source_session_id": "s1"},
            {"source_session_id": "s3"},
        ]

        metrics = answer_session_recall(record, memories)

        self.assertTrue(metrics["retrieval_hit_answer_session"])
        self.assertFalse(metrics["retrieval_hit_all_answer_sessions"])
        self.assertAlmostEqual(metrics["answer_session_coverage"], 2 / 3)
        self.assertEqual(metrics["missing_answer_session_ids"], ["s2"])

    def test_temporal_neighbor_prefers_the_reference_event_clause(self):
        question = "What time did I sleep the day before my flight?"
        schedule = {
            "source_quote": "Tuesday schedule: work at 9 AM and lunch at noon.",
            "role": "assistant",
            "score": 0.9,
        }
        reference = {
            "source_quote": "Your flight on Thursday followed a busy week.",
            "role": "assistant",
            "score": 0.4,
        }

        self.assertGreater(
            session_neighbor_priority(reference, question),
            session_neighbor_priority(schedule, question),
        )


class MultiSessionOperationTests(unittest.TestCase):
    def test_planner_distinguishes_count_count_distinct_sum_and_join(self):
        count_plan = infer_multi_session_operation(
            "How many projects did I lead?",
            "multi-session",
        )
        distinct_plan = infer_multi_session_operation(
            "How many different clinics did I visit?",
            "multi-session",
        )
        sum_plan = infer_multi_session_operation(
            "How many hours in total did I spend traveling?",
            "multi-session",
        )
        join_plan = infer_multi_session_operation(
            "What time did I sleep the day before my flight?",
            "multi-session",
        )

        self.assertEqual(count_plan["operation"], "count")
        self.assertEqual(distinct_plan["operation"], "count_distinct")
        self.assertEqual(sum_plan["operation"], "sum")
        self.assertEqual(join_plan["operation"], "temporal_join")

    def test_planner_separates_money_from_year_to_date_scope(self):
        plan = infer_multi_session_operation(
            "How much total money have I spent on bike-related expenses "
            "since the start of the year?",
            "multi-session",
            "2026/05/05",
        )

        self.assertEqual(plan["operation"], "sum")
        self.assertEqual(plan["answer_dimension"], "money")
        self.assertEqual(plan["target_unit"], "money")
        self.assertEqual(plan["temporal_scope"]["kind"], "year_to_date")

    def test_duration_spend_question_remains_a_duration(self):
        plan = infer_multi_session_operation(
            "How many days did I spend camping this year?",
            "multi-session",
        )

        self.assertEqual(plan["operation"], "sum")
        self.assertEqual(plan["answer_dimension"], "duration")
        self.assertEqual(plan["target_unit"], "day")

    def test_hyphenated_duration_and_us_scope_are_grounded(self):
        question = (
            "How many days did I spend on camping trips in the United States "
            "this year?"
        )
        rows = [
            extraction(
                "I just returned from a 5-day camping trip to Yellowstone while "
                "planning another hike in Colorado.",
                [],
                session_id="s1",
            ),
            extraction(
                "I completed a 3-day solo camping trip to Big Sur.",
                [],
                turn_id="turn-2",
                session_id="s2",
            ),
            extraction(
                "We took a 7-day road trip through Utah. We did plenty of hiking, "
                "but did not go camping.",
                [],
                turn_id="turn-3",
                session_id="s3",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")

        facts = extract_measurement_facts(rows, question, plan)

        self.assertEqual([fact["normalized_value"] for fact in facts], [5.0, 3.0])
        self.assertTrue(
            anchor_coverage(
                build_query_profile(question),
                " ".join(row["source_quote"] for row in rows),
            )["complete"]
        )

    def test_count_coverage_does_not_apply_measurement_unit_validation(self):
        plan = infer_multi_session_operation(
            "How many different doctors did I visit?",
            "multi-session",
        )
        selected = [
            {
                "candidate_id": "C1",
                "canonical_identity": "Dr. Smith",
                "entity": "Dr. Smith",
                "source_turn_id": "t1",
                "source_session_id": "s1",
                "source_quote": "I visited Dr. Smith.",
            },
            {
                "candidate_id": "C2",
                "canonical_identity": "Dr. Lee",
                "entity": "Dr. Lee",
                "source_turn_id": "t2",
                "source_session_id": "s2",
                "source_quote": "I visited Dr. Lee.",
            },
        ]

        coverage = validate_operation_fact_coverage(selected, plan)

        self.assertTrue(coverage["complete"])
        self.assertNotIn("incompatible_units", coverage["reasons"])

    def test_incomplete_count_still_fails_session_coverage(self):
        plan = infer_multi_session_operation(
            "How many projects did I lead?",
            "multi-session",
        )
        coverage = validate_operation_fact_coverage(
            [
                {
                    "candidate_id": "C1",
                    "canonical_identity": "market analysis project",
                    "entity": "market analysis project",
                    "source_turn_id": "t1",
                    "source_session_id": "s1",
                    "source_quote": "I led the market analysis project.",
                }
            ],
            plan,
        )

        self.assertFalse(coverage["complete"])
        self.assertIn("minimum_session_count_not_met", coverage["reasons"])

    def test_operation_queries_and_user_turn_expansion_are_multi_session_only(self):
        question = "How many projects have I led or am currently leading?"
        plan = infer_multi_session_operation(question, "multi-session")
        variants = operation_retrieval_queries(question, plan)
        selected = [
            memory("An assistant summary about the project.", "a1", "s1", "assistant"),
            memory("Another assistant summary.", "a2", "s2", "assistant"),
        ]
        user_one = {
            **memory("I led a market research project.", "u1", "s1", "user"),
            "label": "raw_turn_chunk",
        }
        user_two = {
            **memory("I am working on a solo data project.", "u2", "s2", "user"),
            "label": "raw_turn_chunk",
        }
        context = SimpleNamespace(vector_memory=[*selected, user_one, user_two])

        expanded = select_operation_user_turn_memories(
            context,
            question,
            plan,
            selected,
        )

        self.assertTrue(any("manage" in variant for variant in variants))
        self.assertEqual({item["source_turn_ids"][0] for item in expanded}, {"u1", "u2"})
        single_plan = infer_multi_session_operation(question, "single-session-user")
        self.assertEqual(operation_retrieval_queries(question, single_plan), [])
        self.assertEqual(
            select_operation_user_turn_memories(
                context,
                question,
                single_plan,
                selected,
            ),
            [],
        )

    def test_bike_expenses_are_summed_and_repeated_event_is_deduplicated(self):
        class BikeSelectorContext(SelectAllOperationContext):
            context_length = 2048

            def llm(self, prompt, **_kwargs):
                identities = {
                    "M1": "chain replacement",
                    "M2": "bike lights installation",
                    "M3": "helmet purchase",
                    "M4": "bike lights installation",
                }
                candidate_ids = re.findall(
                    r"^\[([MC]\d+)\]",
                    prompt,
                    flags=re.MULTILINE,
                )
                return {
                    "choices": [
                        {
                            "text": "\n".join(
                                f"OPERAND | {candidate_id} | {identities[candidate_id]}"
                                for candidate_id in candidate_ids
                            )
                        }
                    ]
                }

        question = (
            "How much total money have I spent on bike-related expenses "
            "since the start of the year?"
        )
        rows = [
            extraction(
                "I replaced the bike chain for $25 and got a new set of bike "
                "lights installed for $40.",
                [],
                session_id="s1",
            ),
            extraction(
                "I bought my Bell Zephyr bike helmet for $120.",
                [],
                turn_id="t2",
                session_id="s2",
            ),
            extraction(
                "I recently got the same new set of bike lights installed for $40.",
                [],
                turn_id="t3",
                session_id="s3",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            BikeSelectorContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "$185")
        self.assertEqual(result["calculation"]["operands"], [25.0, 40.0, 120.0])

    def test_count_executor_keeps_all_grounded_items_across_sessions(self):
        question = "How many model kits have I worked on or bought?"
        rows = [
            extraction(
                "I worked on a Revell F-15 model kit.",
                [["speaker user", "worked on", "Revell F-15 model kit"]],
                session_id="s1",
            ),
            extraction(
                "I finished a Tamiya Spitfire model kit.",
                [["speaker user", "finished", "Tamiya Spitfire model kit"]],
                turn_id="t2",
                session_id="s2",
            ),
            extraction(
                "I worked on a German Tiger tank model kit.",
                [["speaker user", "worked on", "German Tiger tank model kit"]],
                turn_id="t3",
                session_id="s3",
            ),
            extraction(
                "I bought a B-29 bomber model kit and a Camaro model kit.",
                [
                    ["speaker user", "bought", "B-29 bomber model kit"],
                    ["speaker user", "bought", "Camaro model kit"],
                ],
                turn_id="t4",
                session_id="s4",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "5")
        self.assertNotIn("incompatible_units", result["coverage"]["reasons"])

    def test_number_parsing_and_duration_unit_normalization(self):
        self.assertEqual(parse_number_value("a week and a half"), 1.5)
        rows = [
            extraction(
                "The first trek lasted 1 week.",
                [["first trek", "lasted", "1 week"]],
                session_id="s1",
            ),
            extraction(
                "The second trek lasted 3 days.",
                [["second trek", "lasted", "3 days"]],
                turn_id="turn-2",
                session_id="s2",
            ),
        ]
        question = "How many days in total did I spend trekking?"
        plan = infer_multi_session_operation(question, "multi-session")

        facts = extract_measurement_facts(rows, question, plan)

        self.assertEqual([fact["normalized_value"] for fact in facts], [7.0, 3.0])
        self.assertTrue(all(fact["target_unit"] == "day" for fact in facts))

    def test_duplicate_operation_facts_use_canonical_identity(self):
        facts = [
            {"canonical_identity": "front brake repair", "entity": "repair"},
            {"canonical_identity": "front brake repair", "entity": "repair"},
            {"canonical_identity": "new tire", "entity": "tire"},
        ]

        deduplicated = deduplicate_operation_facts(facts, "count")

        self.assertEqual(len(deduplicated), 2)

    def test_count_deduplicates_same_turn_aliases_but_preserves_repeated_events(self):
        facts = [
            {
                "grounded_identity": "orion capsule kit",
                "entity": "Orion capsule kit",
                "source_turn_id": "t1",
                "source_quote": "I worked on my Orion capsule kit.",
            },
            {
                "grounded_identity": "capsule kit",
                "entity": "capsule kit",
                "source_turn_id": "t1",
                "source_quote": "I worked on my Orion capsule kit.",
            },
            {
                "grounded_identity": "orion capsule kit",
                "entity": "Orion capsule kit",
                "source_turn_id": "t2",
                "source_quote": "I worked on my Orion capsule kit again.",
            },
            {
                "grounded_identity": "capsule kit",
                "entity": "capsule kit",
                "source_turn_id": "t2",
                "source_quote": "I worked on my Orion capsule kit again.",
            },
        ]

        counted_events = deduplicate_operation_facts(facts, "count")
        distinct_items = deduplicate_operation_facts(facts, "count_distinct")

        self.assertEqual(len(counted_events), 2)
        self.assertEqual(len(distinct_items), 1)

    def test_measurement_dedup_keeps_same_price_events_on_different_dates(self):
        facts = [
            {
                "canonical_identity": "bike tire purchase",
                "normalized_value": 40.0,
                "currency": "USD",
                "source_turn_id": "t1",
                "source_quote": "I bought a bike tire for $40 on April 2nd.",
                "sentence": "I bought a bike tire for $40 on April 2nd.",
                "clause": "I bought a bike tire for $40 on April 2nd",
                "raw_value": "$40",
            },
            {
                "canonical_identity": "bike tire purchase",
                "normalized_value": 40.0,
                "currency": "USD",
                "source_turn_id": "t2",
                "source_quote": "I bought another bike tire for $40 on May 3rd.",
                "sentence": "I bought another bike tire for $40 on May 3rd.",
                "clause": "I bought another bike tire for $40 on May 3rd",
                "raw_value": "$40",
            },
        ]

        self.assertEqual(len(deduplicate_operation_facts(facts, "sum")), 2)

    def test_numeric_range_endpoint_is_not_a_duration_fact(self):
        question = "How many hours in total did I spend driving?"
        plan = infer_multi_session_operation(question, "multi-session")
        facts = extract_measurement_facts(
            [
                extraction(
                    "I drove for 4 hours. The full vacation lasted 7-10 days.",
                    [],
                    session_id="s1",
                ),
                extraction(
                    "I drove for another 5 hours.",
                    [],
                    turn_id="t2",
                    session_id="s2",
                ),
            ],
            question,
            plan,
        )

        self.assertEqual([fact["raw_value"] for fact in facts], ["4 hours", "5 hours"])

    def test_temporal_join_requires_adjacent_target_and_reference_days(self):
        rows = [
            extraction(
                "I did not get to bed until 1:30 AM last Monday.",
                [["speaker user", "went to bed", "1:30 AM last Monday"]],
                session_id="sleep-session",
            ),
            extraction(
                "My flight left at 9 AM last Tuesday.",
                [["flight", "left at", "9 AM last Tuesday"]],
                turn_id="turn-flight",
                session_id="flight-session",
            ),
        ]
        question = "What time did I go to bed the day before my flight?"

        value, evidence = temporal_join_result(
            rows,
            question,
            build_query_profile(question),
        )

        self.assertEqual(value, "1:30 AM")
        self.assertEqual(len(evidence["joined_events"]), 1)

    def test_executor_sums_only_selected_provenance_backed_operands(self):
        class FakeContextLayer:
            def trim_text_to_token_budget(self, text, _budget):
                return text

            def count_tokens(self, text):
                return max(1, len(text) // 4)

            def llm(self, *_args, **_kwargs):
                return {
                    "choices": [
                        {
                            "text": (
                                "OPERAND | M1 | outbound drive\n"
                                "OPERAND | M2 | return drive"
                            )
                        }
                    ]
                }

        rows = [
            extraction(
                "The outbound drive took 2 hours.",
                [["outbound drive", "took", "2 hours"]],
                session_id="s1",
            ),
            extraction(
                "The return drive took 90 minutes.",
                [["return drive", "took", "90 minutes"]],
                turn_id="turn-2",
                session_id="s2",
            ),
        ]
        question = "How many hours in total did I spend driving?"
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            FakeContextLayer(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "3.5 hours")
        self.assertEqual(result["covered_session_count"], 2)

    def test_planner_recognizes_general_operations_without_answer_data(self):
        cases = {
            "What was the average duration of my three reading sessions?": "average",
            "What is the difference between my train and bus commute times?": "difference",
            "What was the ratio of completed to deferred tickets?": "ratio",
            "Which venue did I visit most: Alpha Hall, Birch Center, Cedar Park?": "argmax",
            "Which venue was least expensive: Alpha Hall, Birch Center, Cedar Park?": "argmin",
            "For my training plan, how many minutes did I spend across swimming, cycling, and walking?": "sum",
        }

        for question, expected_operation in cases.items():
            with self.subTest(question=question):
                self.assertEqual(
                    infer_multi_session_operation(question, "multi-session")[
                        "operation"
                    ],
                    expected_operation,
                )

    def test_general_executor_computes_average_difference_ratio_and_extrema(self):
        cases = [
            (
                "What was the average duration of my three Winter reading sessions?",
                [
                    extraction("My Winter session lasted 20 minutes.", [], session_id="s1"),
                    extraction("My Winter session lasted 40 minutes.", [], turn_id="t2", session_id="s2"),
                    extraction("My Winter session lasted 60 minutes.", [], turn_id="t3", session_id="s3"),
                ],
                "40 minutes",
            ),
            (
                "What is the difference between my train and bus commute times?",
                [
                    extraction("My bus commute took 35 minutes.", [], session_id="s1"),
                    extraction("My train commute took 1 hour.", [], turn_id="t2", session_id="s2"),
                ],
                "25 minutes",
            ),
            (
                "What was the ratio of completed to deferred review tickets?",
                [
                    extraction("I completed 12 review tickets.", [], session_id="s1"),
                    extraction("I deferred 3 review tickets.", [], turn_id="t2", session_id="s2"),
                ],
                "12:3",
            ),
            (
                "Which venue did I visit most: Alpha Hall, Birch Center, Cedar Park?",
                [
                    extraction("I visited Alpha Hall 4 times.", [], session_id="s1"),
                    extraction("I visited Birch Center 9 times.", [], turn_id="t2", session_id="s2"),
                    extraction("I visited Cedar Park 6 times.", [], turn_id="t3", session_id="s3"),
                ],
                "Birch Center",
            ),
            (
                "Which venue was least expensive: Alpha Hall, Birch Center, Cedar Park?",
                [
                    extraction("Alpha Hall cost $35.", [], session_id="s1"),
                    extraction("Birch Center cost $20.", [], turn_id="t2", session_id="s2"),
                    extraction("Cedar Park cost $50.", [], turn_id="t3", session_id="s3"),
                ],
                "Birch Center",
            ),
        ]

        for question, rows, expected_answer in cases:
            with self.subTest(question=question):
                plan = infer_multi_session_operation(question, "multi-session")
                result = execute_operation_plan(
                    SelectAllOperationContext(),
                    question,
                    plan,
                    rows,
                    build_query_profile(question),
                )
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["answer"], expected_answer)
                self.assertTrue(result["coverage"]["complete"])
                self.assertTrue(result["calculation"]["operator"])

    def test_operation_coverage_rejects_missing_operands_and_mixed_currencies(self):
        missing_plan = infer_multi_session_operation(
            "What was the total cost of my headphones and tablet?",
            "multi-session",
        )
        missing_result = execute_operation_plan(
            SelectAllOperationContext(),
            "What was the total cost of my headphones and tablet?",
            missing_plan,
            [extraction("I bought my headphones for $80.", [], session_id="s1")],
            build_query_profile("What was the total cost of my headphones and tablet?"),
        )
        mixed_plan = infer_multi_session_operation(
            "What was the total cost of my lamp and monitor?",
            "multi-session",
        )
        mixed_result = execute_operation_plan(
            SelectAllOperationContext(),
            "What was the total cost of my lamp and monitor?",
            mixed_plan,
            [
                extraction("I bought my lamp for $40.", [], session_id="s1"),
                extraction("I bought my monitor for 50 euros.", [], turn_id="t2", session_id="s2"),
            ],
            build_query_profile("What was the total cost of my lamp and monitor?"),
        )

        self.assertEqual(missing_result["status"], "insufficient_evidence")
        self.assertIn(
            "explicit_operands_missing",
            missing_result["coverage"]["reasons"],
        )
        self.assertEqual(mixed_result["status"], "insufficient_evidence")
        self.assertIn("mixed_currencies", mixed_result["coverage"]["reasons"])

    def test_temporal_executor_handles_difference_relative_order_and_date(self):
        cases = [
            (
                "How many days passed between the ceramics fair at Alpha Hall and the harbor tour at Cedar Park?",
                [
                    extraction("I attended the ceramics fair at Alpha Hall today.", [], timestamp="2026/08/01"),
                    extraction("I took the harbor tour at Cedar Park today.", [], turn_id="t2", session_id="s2", timestamp="2026/08/11"),
                ],
                "10 days",
            ),
            (
                "How many weeks ago did I finish the mosaic at Kestrel Arts?",
                [extraction("Today I finished the mosaic at Kestrel Arts.", [], timestamp="2026/09/09")],
                "3",
            ),
            (
                "In what order did these happen: painted the hallway, ordered the notebook, visited the orchid show?",
                [
                    extraction("Today I painted the hallway.", [], timestamp="2026/08/01"),
                    extraction("Today I ordered the notebook.", [], turn_id="t2", session_id="s2", timestamp="2026/08/11"),
                    extraction("Today I visited the orchid show.", [], turn_id="t3", session_id="s3", timestamp="2026/08/21"),
                ],
                "First, painted the hallway; then, ordered the notebook; finally, visited the orchid show.",
            ),
            (
                "On what date did I attend the lantern festival at Birch Center?",
                [extraction("I attended the lantern festival at Birch Center today.", [], timestamp="2026/08/17")],
                "August 17",
            ),
        ]

        for question, rows, expected_answer in cases:
            with self.subTest(question=question):
                plan = infer_multi_session_operation(
                    question,
                    "temporal-reasoning",
                    "2026/09/30 (Wed) 10:00",
                )
                result = execute_operation_plan(
                    SelectAllOperationContext(),
                    question,
                    plan,
                    rows,
                    build_query_profile(question),
                )
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["answer"], expected_answer)

    def test_temporal_executor_handles_adjacent_day_and_clock_duration(self):
        adjacent_question = (
            "What did I do the day after I attended the chess meetup at Alpha Hall?"
        )
        adjacent_rows = [
            extraction("Today I attended the chess meetup at Alpha Hall.", [], timestamp="2026/08/01"),
            extraction("Today I started the bread workshop at Birch Center.", [], turn_id="t2", session_id="s2", timestamp="2026/08/02"),
        ]
        duration_question = "How long did the print workshop at Cedar Park last?"
        duration_rows = [
            extraction(
                "The print workshop at Cedar Park started at 9:00 and ended at 12:00 today.",
                [],
                timestamp="2026/08/01",
            )
        ]

        adjacent_plan = infer_multi_session_operation(
            adjacent_question,
            "temporal-reasoning",
            "2026/09/30",
        )
        duration_plan = infer_multi_session_operation(
            duration_question,
            "temporal-reasoning",
            "2026/09/30",
        )
        adjacent = execute_operation_plan(
            SelectAllOperationContext(),
            adjacent_question,
            adjacent_plan,
            adjacent_rows,
            build_query_profile(adjacent_question),
        )
        duration = execute_operation_plan(
            SelectAllOperationContext(),
            duration_question,
            duration_plan,
            duration_rows,
            build_query_profile(duration_question),
        )

        self.assertEqual(adjacent["answer"], "started the bread workshop")
        self.assertEqual(duration["answer"], "3 hours")

    def test_temporal_order_parser_handles_explicit_a_or_b_alternatives(self):
        question = (
            "Which happened first at Elm Street Library: my print workshop or "
            "my sculpture class?"
        )
        plan = infer_multi_session_operation(
            question,
            "temporal-reasoning",
            "2027/01/20",
        )
        rows = [
            extraction(
                "Today I attended my print workshop at Elm Street Library.",
                [],
                timestamp="2027/01/03",
            ),
            extraction(
                "Today I attended my sculpture class at Elm Street Library.",
                [],
                turn_id="turn-2",
                session_id="session-2",
                timestamp="2027/01/10",
            ),
        ]

        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(plan["operation"], "temporal_order")
        self.assertEqual(
            [spec["text"] for spec in plan["temporal_event_specs"]],
            ["print workshop", "sculpture class"],
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "print workshop")

    def test_temporal_a_or_b_without_dates_deterministically_abstains(self):
        question = (
            "Which happened first at Northwind Lab: my print workshop or my "
            "sculpture class?"
        )
        plan = infer_multi_session_operation(
            question,
            "temporal-reasoning",
            "2027/01/20",
        )
        evidence_text = (
            "I remember enjoying both the print workshop and sculpture class at "
            "Northwind Lab, but I did not record their dates."
        )
        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            [extraction(evidence_text, [], timestamp="2027/01/10")],
            build_query_profile(question),
        )
        safe = determine_safe_abstention(
            plan,
            result,
            build_query_profile(question),
            anchor_coverage(build_query_profile(question), evidence_text),
        )

        self.assertEqual(plan["operation"], "temporal_order")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(safe, (True, "operation_evidence_incomplete"))

    def test_adjacent_day_date_index_retrieves_the_semantically_unrelated_event(self):
        question = (
            "What did I do the day after I bought the blue suitcase at Cedar House?"
        )
        plan = infer_multi_session_operation(
            question,
            "temporal-reasoning",
            "2027/01/10",
        )
        anchor = {
            **memory(
                "Today I bought the blue suitcase at Cedar House.",
                "anchor-turn",
                "anchor-session",
            ),
            "label": "raw_turn_chunk",
            "timestamp": "2026/12/26 (Sat) 03:30",
        }
        target = {
            **memory(
                "Today I visited the sculpture garden at Lantern Theatre.",
                "target-turn",
                "target-session",
            ),
            "label": "raw_turn_chunk",
            "timestamp": "2026/12/27 (Sun) 03:30",
        }
        unrelated = {
            **memory(
                "Today I repaired the garden gate.",
                "other-turn",
                "other-session",
            ),
            "label": "raw_turn_chunk",
            "timestamp": "2026/12/20 (Sun) 03:30",
        }
        context = SimpleNamespace(vector_memory=[anchor, unrelated, target])

        retrieved, diagnostics = retrieve_temporal_adjacent_date_memories(
            context,
            question,
            plan,
            [anchor],
        )
        rows, _diagnostics = evidence_rows_without_candidate_extraction(
            [anchor, *retrieved]
        )
        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(
            [item["source_session_id"] for item in retrieved],
            ["target-session"],
        )
        self.assertEqual(diagnostics["status"], "complete")
        self.assertEqual(diagnostics["anchor_date"], "2026-12-26")
        self.assertEqual(diagnostics["target_date"], "2026-12-27")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "visited the sculpture garden")

    def test_adjacent_day_date_index_preserves_ambiguity_and_operation_scope(self):
        adjacent_question = (
            "What did I do the day after I attended the chess meetup at Aster Hall?"
        )
        adjacent_plan = infer_multi_session_operation(
            adjacent_question,
            "temporal-reasoning",
            "2027/01/20",
        )
        anchors = [
            {
                **memory(
                    "Today I attended the chess meetup at Aster Hall.",
                    "anchor-one",
                    "anchor-session-one",
                ),
                "label": "raw_turn_chunk",
                "timestamp": "2027/01/13 (Wed) 01:27",
            },
            {
                **memory(
                    "Today I attended the chess meetup at Aster Hall.",
                    "anchor-two",
                    "anchor-session-two",
                ),
                "label": "raw_turn_chunk",
                "timestamp": "2027/01/06 (Wed) 01:27",
            },
        ]
        context = SimpleNamespace(vector_memory=anchors)

        ambiguous, diagnostics = retrieve_temporal_adjacent_date_memories(
            context,
            adjacent_question,
            adjacent_plan,
            anchors,
        )
        difference_question = (
            "How many days passed between my chess meetup and choir rehearsal?"
        )
        difference_plan = infer_multi_session_operation(
            difference_question,
            "temporal-reasoning",
            "2027/01/20",
        )
        out_of_scope, out_of_scope_diagnostics = (
            retrieve_temporal_adjacent_date_memories(
                context,
                difference_question,
                difference_plan,
                anchors,
            )
        )

        self.assertEqual(ambiguous, [])
        self.assertEqual(
            diagnostics["status"],
            "anchor_date_missing_or_ambiguous",
        )
        self.assertEqual(out_of_scope, [])
        self.assertFalse(out_of_scope_diagnostics["applicable"])
        self.assertEqual(out_of_scope_diagnostics["status"], "not_applicable")

    def test_temporal_executor_abstains_for_vague_missing_or_ambiguous_dates(self):
        question = "How many weeks ago did I finish the mosaic at Kestrel Arts?"
        plan = infer_multi_session_operation(
            question,
            "temporal-reasoning",
            "2026/09/30",
        )
        vague_value, _ = temporal_operation_result(
            [
                extraction(
                    "I finished the mosaic at Kestrel Arts recently.",
                    [],
                    timestamp="2026/09/20",
                )
            ],
            question,
            plan,
        )
        ambiguous_value, _ = temporal_operation_result(
            [
                extraction("Today I finished the mosaic at Kestrel Arts.", [], timestamp="2026/09/09"),
                extraction("Today I finished the mosaic at Kestrel Arts.", [], turn_id="t2", session_id="s2", timestamp="2026/09/16"),
            ],
            question,
            plan,
        )

        self.assertIsNone(vague_value)
        self.assertIsNone(ambiguous_value)

    def test_exact_date_abstains_when_evidence_has_only_month_precision(self):
        question = "On what exact date did I volunteer at the shelter dinner?"
        plan = infer_multi_session_operation(
            question,
            "single-session-user",
            "2025/08/10 (Sun) 22:54",
        )
        result = execute_operation_plan(
            None,
            question,
            plan,
            [
                extraction(
                    "I volunteered at the shelter fundraising dinner sometime in March.",
                    [],
                    timestamp="2025/07/15 (Tue) 20:54",
                )
            ],
            build_query_profile(question),
        )

        self.assertEqual(plan["requested_temporal_granularity"], "day")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["answer"])
        self.assertEqual(
            result["candidate_facts"]["precision_check"],
            {
                "requested_granularity": "day",
                "available_granularity": "month",
                "satisfied": False,
            },
        )

    def test_generic_when_date_behavior_remains_source_relative(self):
        question = (
            "When did I attend Emiko's neighborhood fundraising dinner at "
            "Harbor Annex?"
        )
        plan = infer_multi_session_operation(
            question,
            "single-session-user",
            "2025/02/17 (Mon) 22:22",
        )
        result = execute_operation_plan(
            None,
            question,
            plan,
            [
                extraction(
                    "I attended Emiko's neighborhood fundraising dinner at "
                    "Harbor Annex on January 7.",
                    [],
                    timestamp="2025/01/07 (Tue) 19:00",
                )
            ],
            build_query_profile(question),
        )

        self.assertIsNone(plan["requested_temporal_granularity"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "7 January 2025")

    def test_coverage_requires_exact_declared_count(self):
        coverage = validate_operation_fact_coverage(
            [
                {"source_session_id": "s1", "normalized_value": 10, "raw_value": "10 minutes", "clause": "10 minutes"},
                {"source_session_id": "s2", "normalized_value": 20, "raw_value": "20 minutes", "clause": "20 minutes"},
            ],
            {
                "expected_fact_count": 3,
                "minimum_sessions": 2,
                "minimum_fact_count": 2,
                "operands": [],
            },
        )

        self.assertFalse(coverage["complete"])
        self.assertIn("expected_fact_count_not_met", coverage["reasons"])

    def test_average_ignores_unrelated_numbers_and_abstains_when_measurements_missing(self):
        question = (
            "With Jonas, what was the average duration of my three practice "
            "sessions at Harbor Annex?"
        )
        rows = [
            extraction(
                "My first practice session at Harbor Annex lasted 37 minutes.",
                [],
                session_id="practice-1",
            ),
            extraction(
                "I returned two library books near Harbor Annex.",
                [],
                turn_id="noise-1",
                session_id="noise-1",
            ),
            extraction(
                "I replaced one worn felt pad before visiting Harbor Annex.",
                [],
                turn_id="noise-2",
                session_id="noise-2",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(plan["answer_dimension"], "duration")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(
            [fact["raw_value"] for fact in result["candidate_facts"]],
            ["37 minutes"],
        )

    def test_count_requires_each_event_to_support_all_identity_anchors(self):
        question = (
            "How many clothing items do I need to pick up for my Northwind Lab "
            "trip with Caleb?"
        )
        rows = [
            extraction(
                "For my Northwind Lab trip with Caleb, I need to pick up my "
                "walking boots from the store.",
                [["speaker user", "need to pick up", "walking boots"]],
                session_id="answer-1",
            ),
            extraction(
                "For my Northwind Lab trip with Caleb, I need to pick up my "
                "raincoat from the store.",
                [["speaker user", "need to pick up", "raincoat"]],
                turn_id="answer-turn-2",
                session_id="answer-2",
            ),
            extraction(
                "For my Northwind Lab trip with Caleb, I need to pick up my "
                "linen jacket from the store.",
                [["speaker user", "need to pick up", "linen jacket"]],
                turn_id="answer-turn-3",
                session_id="answer-3",
            ),
            extraction(
                "I put a compact umbrella in my bag and might stop near "
                "Northwind Lab later.",
                [["speaker user", "put in bag", "compact umbrella"]],
                turn_id="noise-turn",
                session_id="noise",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "3")
        self.assertEqual(
            {fact["source_session_id"] for fact in result["facts"]},
            {"answer-1", "answer-2", "answer-3"},
        )

    def test_count_distinct_uses_grounded_entity_not_graph_edge_wording(self):
        question = (
            "How many distinct model kits have I worked on with Aarav for the "
            "show at Fern Court?"
        )
        models = [
            "Comet racer kit",
            "Falcon glider kit",
            "Tiger tank kit",
            "Harbor tug kit",
            "Falcon glider kit",
        ]
        rows = []
        for index, model in enumerate(models, start=1):
            quote = (
                f"With Aarav for the model show at Fern Court, I worked on my "
                f"{model} today."
            )
            rows.append(
                extraction(
                    quote,
                    [
                        ["speaker user", "worked on", model],
                        [model, "item type", "model kit"],
                        [model, "located at", "Fern Court"],
                        ["Aarav", "show participant", model],
                    ],
                    turn_id=f"turn-{index}",
                    session_id=f"session-{index}",
                )
            )
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "4")
        self.assertEqual(len(result["facts"]), 4)

    def test_count_distinct_collapses_aliases_before_global_identity_dedup(self):
        question = (
            "How many distinct model kits have I worked on with Hana for the "
            "show at Indigo Room?"
        )
        rows = []
        for index, model in enumerate(
            ["Tiger tank kit", "Tiger tank kit", "Falcon glider kit"],
            start=1,
        ):
            subtype = "tank kit" if model.startswith("Tiger") else "glider kit"
            quote = (
                "With Hana for the model show at Indigo Room, I worked on my "
                f"{model} today."
            )
            rows.append(
                extraction(
                    quote,
                    [
                        ["speaker user", "worked on", model],
                        [model, "item type", subtype],
                    ],
                    turn_id=f"turn-{index}",
                    session_id=f"session-{index}",
                )
            )
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "2")
        self.assertEqual(
            {fact["canonical_identity"] for fact in result["facts"]},
            {"tiger tank kit", "falcon glider kit"},
        )

    def test_extrema_ignore_measurements_that_do_not_map_to_explicit_operands(self):
        question = (
            "Which of these venues have I visited most this year: Aster Hall, "
            "Orchid Gallery, Kestrel Arts?"
        )
        rows = [
            extraction("I visited Aster Hall 3 times this year.", [], session_id="s1"),
            extraction(
                "I visited Orchid Gallery 8 times this year.",
                [],
                turn_id="t2",
                session_id="s2",
            ),
            extraction(
                "I visited Kestrel Arts 10 times this year.",
                [],
                turn_id="t3",
                session_id="s3",
            ),
            extraction(
                "I saw two bulbuls near the feeder.",
                [],
                turn_id="noise-1",
                session_id="noise-1",
            ),
            extraction(
                "I replaced one worn felt pad.",
                [],
                turn_id="noise-2",
                session_id="noise-2",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")

        result = execute_operation_plan(
            SelectAllOperationContext(),
            question,
            plan,
            rows,
            build_query_profile(question),
        )

        self.assertIsNone(plan["answer_dimension"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "Kestrel Arts")
        self.assertEqual(len(result["facts"]), 3)

    def test_implicit_duration_and_distance_dimensions_preserve_units(self):
        cases = [
            (
                "What was the average duration of my three reading sessions?",
                [
                    extraction("My first reading session lasted 20 minutes.", [], session_id="s1"),
                    extraction("My second reading session lasted 40 minutes.", [], turn_id="t2", session_id="s2"),
                    extraction("My third reading session lasted 60 minutes.", [], turn_id="t3", session_id="s3"),
                ],
                "40 minutes",
            ),
            (
                "What total distance did I cover on the canal, park, and harbor routes?",
                [
                    extraction("I covered 5 kilometers on the canal route.", [], session_id="s1"),
                    extraction("I covered 7 kilometers on the park route.", [], turn_id="t2", session_id="s2"),
                    extraction("I covered 8 kilometers on the harbor route.", [], turn_id="t3", session_id="s3"),
                ],
                "20 kilometers",
            ),
        ]

        for question, rows, expected in cases:
            with self.subTest(question=question):
                plan = infer_multi_session_operation(question, "multi-session")
                result = execute_operation_plan(
                    SelectAllOperationContext(),
                    question,
                    plan,
                    rows,
                    build_query_profile(question),
                )
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["answer"], expected)

    def test_complete_operation_cannot_remain_marked_as_safe_abstention(self):
        abstain, reason = determine_safe_abstention(
            {"operation": "count"},
            {"status": "complete", "answer": "3"},
            {"required_anchor_groups": [{"tokens": ["alpha"]}]},
            {"complete": False},
            requested_slot="exact quantity or amount",
            selected_evidence_text="",
            selected_evidence_units=[],
            answer_slot_candidates=[],
        )

        self.assertFalse(abstain)
        self.assertIsNone(reason)

    def test_collection_planner_is_generic_and_preserves_singular_lookup(self):
        collect = infer_multi_session_operation(
            "Which activities did Asha add across all sessions?",
            "multi-session",
        )
        intersection = infer_multi_session_operation(
            "Which activities do Asha and Rohan both enjoy?",
            "multi-session",
        )
        singular = infer_multi_session_operation(
            "What is Asha's favorite place?",
            "multi-session",
        )

        self.assertEqual(collect["operation"], "set_union")
        self.assertEqual(intersection["operation"], "set_intersection")
        self.assertEqual(singular["operation"], "none")

    def test_collection_union_returns_all_actor_bound_items_with_provenance(self):
        question = "Which activities did Asha add across all sessions?"
        rows = [
            extraction(
                "For the Atlas plan, I added ceramics to my activities.",
                [],
                turn_id="a1",
                session_id="s1",
                timestamp="2027-01-05T10:00:00",
                speaker="Asha",
            ),
            extraction(
                "For the Atlas plan, I added birdwatching to my activities.",
                [],
                turn_id="a2",
                session_id="s2",
                timestamp="2027-01-12T10:00:00",
                speaker="Asha",
            ),
            extraction(
                "For the Atlas plan, I added ceramics to my activities.",
                [],
                turn_id="a3",
                session_id="s3",
                timestamp="2027-01-19T10:00:00",
                speaker="Asha",
            ),
            extraction(
                "For the Atlas plan, I added kayaking to my activities.",
                [],
                turn_id="r1",
                session_id="s4",
                timestamp="2027-01-26T10:00:00",
                speaker="Rohan",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")
        profile = build_query_profile(question, known_speakers=["Asha", "Rohan"])

        result = execute_operation_plan(None, question, plan, rows, profile)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "ceramics, birdwatching")
        self.assertEqual(len(result["facts"]), 2)
        self.assertEqual(len(result["facts"][0]["provenance"]), 2)
        self.assertTrue(
            all(fact["source_speaker"] == "Asha" for fact in result["facts"])
        )

    def test_collection_intersection_requires_each_requested_speaker(self):
        question = "Which activities do Asha and Rohan both enjoy?"
        rows = [
            extraction(
                "One activity I especially enjoy is bread baking.",
                [],
                turn_id="a1",
                session_id="s1",
                speaker="Asha",
            ),
            extraction(
                "One activity I especially enjoy is bread baking.",
                [],
                turn_id="r1",
                session_id="s2",
                speaker="Rohan",
            ),
            extraction(
                "One activity I especially enjoy is kayaking.",
                [],
                turn_id="a2",
                session_id="s3",
                speaker="Asha",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")
        profile = build_query_profile(question, known_speakers=["Asha", "Rohan"])

        result = execute_operation_plan(None, question, plan, rows, profile)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "bread baking")
        self.assertEqual(
            set(result["facts"][0]["speakers"]),
            {"Asha", "Rohan"},
        )

    def test_collection_retrieval_expands_until_long_list_is_complete(self):
        question = "Which activities did Asha add across all sessions?"
        memories = []
        for index in range(19):
            memories.append(
                {
                    **memory(
                        f"I added activity {index + 1} to my activities.",
                        f"t{index + 1}",
                        session_id=f"s{index + 1}",
                    ),
                    "source_speaker": "Asha",
                    "speaker": "Asha",
                }
            )

        class ExpandingContext:
            vector_memory = memories

            def retrieve_relevant_memories(self, _query, top_k, **_kwargs):
                return self.vector_memory[:top_k]

        plan = infer_multi_session_operation(question, "multi-session")
        profile = build_query_profile(question, known_speakers=["Asha", "Rohan"])

        retrieved, diagnostics = retrieve_collection_memories_until_saturated(
            ExpandingContext(),
            question,
            plan,
            profile,
            initial_top_k=4,
            max_top_k=24,
        )

        self.assertEqual(len(retrieved), 19)
        self.assertEqual(diagnostics["status"], "corpus_exhausted")
        self.assertTrue(diagnostics["saturated"])
        self.assertEqual(diagnostics["discovered_item_count"], 19)

    def test_count_distinct_binds_named_actor_from_speaker_metadata(self):
        question = (
            "How many different workshops did Asha attend for the Copper "
            "Atlas plan?"
        )
        rows = [
            extraction(
                "I attended workshop Copper as part of the Copper Atlas plan series.",
                [],
                turn_id="a1",
                session_id="s1",
                speaker="Asha",
            ),
            extraction(
                "I attended workshop Emerald as part of the Copper Atlas plan series.",
                [],
                turn_id="a2",
                session_id="s2",
                speaker="Asha",
            ),
            extraction(
                "I attended workshop Indigo as part of the Copper Atlas plan series.",
                [],
                turn_id="a3",
                session_id="s3",
                speaker="Asha",
            ),
            extraction(
                "I attended workshop Silver as part of the Copper Atlas plan series.",
                [],
                turn_id="r1",
                session_id="s4",
                speaker="Rohan",
            ),
        ]
        plan = infer_multi_session_operation(question, "multi-session")
        profile = build_query_profile(question, known_speakers=["Asha", "Rohan"])

        result = execute_operation_plan(None, question, plan, rows, profile)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "3")
        self.assertEqual(
            {fact["source_speaker"] for fact in result["facts"]},
            {"Asha"},
        )

    def test_source_relative_temporal_expressions_keep_their_granularity(self):
        last_week = resolve_temporal_expression(
            "I completed the review last week.",
            "2027-05-20T10:00:00",
        )
        next_month = resolve_temporal_expression(
            "I plan to launch it next month.",
            "2027-02-13T10:00:00",
        )
        day_first = resolve_temporal_expression(
            "I completed it on 3 January 2027.",
            "2027-01-04T10:00:00",
        )

        self.assertEqual(last_week["granularity"], "week")
        self.assertEqual(last_week["interval_start"], "2027-05-10")
        self.assertEqual(last_week["interval_end"], "2027-05-16")
        self.assertEqual(next_month["granularity"], "month")
        self.assertEqual(next_month["display"], "March 2027")
        self.assertEqual(next_month["event_status"], "planned")
        self.assertEqual(day_first["value"], "2027-01-03")

    def test_temporal_chain_composes_cross_session_offsets_with_provenance(self):
        rows = [
            extraction(
                "Today I attended the kickoff meeting for the Silver Bridge project.",
                [],
                turn_id="t1",
                session_id="s1",
                timestamp="2027-04-26T10:00:00",
                speaker="Asha",
            ),
            extraction(
                "The planning checkpoint for the Silver Bridge project happened one day after its kickoff meeting.",
                [],
                turn_id="t2",
                session_id="s2",
                timestamp="2027-05-01T10:00:00",
                speaker="Asha",
            ),
            extraction(
                "The field check for the Silver Bridge project happened two days after its planning checkpoint.",
                [],
                turn_id="t3",
                session_id="s3",
                timestamp="2027-05-08T10:00:00",
                speaker="Asha",
            ),
            extraction(
                "The archive review for the Silver Bridge project took place one day after its field check.",
                [],
                turn_id="t4",
                session_id="s4",
                timestamp="2027-05-15T10:00:00",
                speaker="Asha",
            ),
        ]
        question = (
            "When did Asha complete the archive review for the Silver Bridge "
            "project?"
        )
        plan = infer_multi_session_operation(question, "temporal-reasoning")
        profile = build_query_profile(question, known_speakers=["Asha", "Rohan"])

        result = execute_operation_plan(None, question, plan, rows, profile)
        chain_events = resolve_temporal_event_chains(rows)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "30 April 2027")
        review = next(
            event for event in chain_events if "archive" in event["clause"].lower()
        )
        self.assertEqual(len(review["provenance_chain"]), 4)

    def test_stated_duration_is_actor_bound_and_deterministic(self):
        question = (
            "How long has Asha maintained the Indigo Atlas seasonal project "
            "archive?"
        )
        rows = [
            extraction(
                "I have maintained the Indigo Atlas seasonal project archive for 6 years.",
                [],
                turn_id="a1",
                session_id="s1",
                speaker="Asha",
            ),
            extraction(
                "I have maintained the Indigo Atlas seasonal project archive for 3 years.",
                [],
                turn_id="r1",
                session_id="s2",
                speaker="Rohan",
            ),
        ]
        plan = infer_multi_session_operation(question, "temporal-reasoning")
        profile = build_query_profile(question, known_speakers=["Asha", "Rohan"])

        result = execute_operation_plan(None, question, plan, rows, profile)

        self.assertEqual(plan["operation"], "temporal_stated_duration")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "6 years")
        self.assertEqual(result["facts"]["selected_events"][0]["source_speaker"], "Asha")


class StateHistoryResolutionTests(unittest.TestCase):
    question_scope = "Harbor Lab"
    person = "Maya"

    @staticmethod
    def history_rows():
        return [
            extraction(
                "In the Harbor Lab notes I shared with Maya, my favorite "
                "weekend animal used to be cats.",
                [],
                turn_id="turn-original",
                session_id="session-original",
                timestamp="2028/01/01 (Sat) 10:00",
            ),
            extraction(
                "For Harbor Lab, I told Maya that I changed my favorite "
                "weekend animal from cats to dogs.",
                [],
                turn_id="turn-middle",
                session_id="session-middle",
                timestamp="2028/01/15 (Sat) 10:00",
            ),
            extraction(
                "My latest Harbor Lab update to Maya changes it again: my "
                "favorite weekend animal is now parrots, not dogs.",
                [],
                turn_id="turn-latest",
                session_id="session-latest",
                timestamp="2028/02/01 (Tue) 10:00",
            ),
        ]

    def resolve(self, question, rows=None, intent=None, disabled=False):
        return resolve_deterministic_state_history_answer(
            question=question,
            evidence_rows=rows if rows is not None else self.history_rows(),
            query_profile=build_query_profile(question),
            query_intent=intent or infer_query_intent(question),
            selector_plan=infer_state_history_selector(question),
            disabled=disabled,
        )

    def test_original_selector_terms_are_not_hard_entity_anchors(self):
        profile = build_query_profile(
            "For my Harbor Lab notes with Maya, what was my original favorite "
            "weekend animal, before either update?"
        )
        anchors = {group["text"] for group in profile["required_anchor_groups"]}

        self.assertNotIn("original", anchors)
        self.assertIn("weekend", anchors)
        self.assertIn("Harbor Lab", anchors)
        self.assertIn("Maya", anchors)

    def test_selector_parser_distinguishes_predecessor_earliest_and_latest(self):
        predecessor = infer_state_history_selector(
            "What was my favorite weekend animal immediately before parrots?"
        )
        earliest = infer_state_history_selector(
            "What was my original favorite weekend animal, before either update?"
        )
        latest = infer_state_history_selector(
            "What is my current favorite weekend animal?"
        )

        self.assertEqual(predecessor["selector"], "predecessor_of_value")
        self.assertEqual(predecessor["target_value"], "parrots")
        self.assertTrue(predecessor["deterministic_enabled"])
        self.assertEqual(earliest["selector"], "earliest")
        self.assertTrue(earliest["deterministic_enabled"])
        self.assertEqual(latest["selector"], "latest")
        self.assertEqual(latest["requested_attribute"], "favorite weekend animal")
        self.assertFalse(latest["deterministic_enabled"])

    def test_predecessor_is_resolved_from_a_continuous_provenance_chain(self):
        question = (
            "In my Harbor Lab updates with Maya, what was my favorite weekend "
            "animal immediately before parrots?"
        )

        result = self.resolve(question)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "dogs")
        self.assertEqual(
            [state["value"] for state in result["states"]],
            ["cats", "dogs", "parrots"],
        )
        self.assertEqual(
            result["answer_provenance"]["source_turn_id"],
            "turn-middle",
        )
        self.assertFalse(result["fallback_preserved"])

    def test_earliest_is_resolved_from_the_same_chain(self):
        question = (
            "For my Harbor Lab notes with Maya, what was my original favorite "
            "weekend animal, before either update?"
        )

        result = self.resolve(question)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "cats")
        self.assertEqual(
            result["answer_provenance"]["source_turn_id"],
            "turn-original",
        )

    def test_current_selector_preserves_the_existing_answer_path(self):
        question = (
            "According to my Harbor Lab updates with Maya, what is my current "
            "favorite weekend animal?"
        )

        result = self.resolve(question)

        self.assertEqual(result["status"], "selector_not_enabled")
        self.assertTrue(result["fallback_preserved"])

    def test_incoherent_chain_falls_back_without_guessing(self):
        rows = self.history_rows()
        rows[-1] = extraction(
            "My latest Harbor Lab update to Maya changes it again: my favorite "
            "weekend animal is now parrots, not horses.",
            [],
            turn_id="turn-latest",
            session_id="session-latest",
            timestamp="2028/02/01 (Tue) 10:00",
        )
        question = (
            "In my Harbor Lab updates with Maya, what was my favorite weekend "
            "animal immediately before parrots?"
        )

        result = self.resolve(question, rows=rows)

        self.assertEqual(result["status"], "incoherent_transition_chain")
        self.assertIsNone(result["answer"])
        self.assertTrue(result["fallback_preserved"])

    def test_missing_timestamp_or_second_session_preserves_fallback(self):
        rows = self.history_rows()[:1]
        rows[0]["source_timestamp"] = None
        question = (
            "For my Harbor Lab notes with Maya, what was my original favorite "
            "weekend animal, before either update?"
        )

        result = self.resolve(question, rows=rows)

        self.assertEqual(
            result["status"],
            "insufficient_timestamped_update_evidence",
        )
        self.assertTrue(result["fallback_preserved"])

    def test_assistant_memory_intent_cannot_enter_state_history_override(self):
        question = (
            "In our previous conversation, what item did you list immediately "
            "before parrots?"
        )
        plan = infer_state_history_selector(question)
        plan.update(
            {
                "requested_attribute": "item",
                "attribute_tokens": ["item"],
                "deterministic_enabled": True,
            }
        )

        result = resolve_deterministic_state_history_answer(
            question=question,
            evidence_rows=self.history_rows(),
            query_profile=build_query_profile(question),
            query_intent={"intent": ASSISTANT_MEMORY_INTENT},
            selector_plan=plan,
        )

        self.assertEqual(result["status"], "incompatible_source_role_intent")
        self.assertTrue(result["fallback_preserved"])

    def test_state_history_ablation_preserves_fallback(self):
        question = (
            "In my Harbor Lab updates with Maya, what was my favorite weekend "
            "animal immediately before parrots?"
        )

        result = self.resolve(question, disabled=True)

        self.assertEqual(result["status"], "disabled_by_ablation")
        self.assertTrue(result["fallback_preserved"])


class StateAttributeMismatchTests(unittest.TestCase):
    question = (
        "According to my Juniper Pavilion updates with Luca, which coffee "
        "roast do I currently prefer?"
    )

    @staticmethod
    def afternoon_tea_rows():
        return [
            extraction(
                "In the Juniper Pavilion notes I shared with Luca, my "
                "preferred afternoon tea was white tea.",
                [],
                turn_id="tea-original",
                session_id="tea-session-1",
                timestamp="2028/07/11 (Tue) 02:07",
            ),
            extraction(
                "For Juniper Pavilion, I told Luca that I switched my "
                "afternoon tea from white tea to oolong tea.",
                [],
                turn_id="tea-middle",
                session_id="tea-session-2",
                timestamp="2028/07/29 (Sat) 02:07",
            ),
            extraction(
                "My latest Juniper Pavilion update to Luca says I now prefer "
                "green tea as my afternoon tea instead of oolong tea.",
                [],
                turn_id="tea-latest",
                session_id="tea-session-3",
                timestamp="2028/08/15 (Tue) 02:07",
            ),
        ]

    def assess(self, rows=None, question=None, disabled=False, intent=None):
        question = question or self.question
        return assess_state_attribute_mismatch(
            evidence_rows=rows if rows is not None else self.afternoon_tea_rows(),
            query_profile=build_query_profile(question),
            query_intent=intent or infer_query_intent(question),
            selector_plan=infer_state_history_selector(question),
            disabled=disabled,
        )

    def test_latest_selector_extracts_which_question_attribute(self):
        plan = infer_state_history_selector(self.question)

        self.assertEqual(plan["selector"], "latest")
        self.assertEqual(plan["requested_attribute"], "coffee roast")
        self.assertEqual(plan["attribute_tokens"], ["coffee", "roast"])

    def test_coherent_scoped_wrong_attribute_chain_forces_abstention(self):
        result = self.assess()

        self.assertEqual(result["status"], "confirmed_wrong_attribute")
        self.assertTrue(result["abstain"])
        self.assertEqual(result["observed_attribute_tokens"], ["afternoon", "tea"])
        self.assertEqual(result["states"], ["white tea", "oolong tea", "green tea"])

    def test_matching_requested_attribute_preserves_existing_answer_path(self):
        question = (
            "According to my Juniper Pavilion updates with Luca, what is my "
            "current preferred afternoon tea?"
        )

        result = self.assess(question=question)

        self.assertEqual(result["status"], "requested_attribute_mentioned")
        self.assertFalse(result["abstain"])
        self.assertTrue(result["fallback_preserved"])

    def test_unparsed_requested_attribute_mention_prevents_false_abstention(self):
        rows = self.afternoon_tea_rows()
        rows.append(
            extraction(
                "In my Juniper Pavilion notes with Luca, the coffee roast "
                "choice is still undecided.",
                [],
                turn_id="coffee-uncertain",
                session_id="coffee-session",
                timestamp="2028/08/16 (Wed) 02:07",
            )
        )

        result = self.assess(rows=rows)

        self.assertEqual(result["status"], "requested_attribute_mentioned")
        self.assertFalse(result["abstain"])

    def test_single_session_or_broken_chain_preserves_fallback(self):
        single_session = self.afternoon_tea_rows()[:1]
        broken_chain = self.afternoon_tea_rows()
        broken_chain[-1] = extraction(
            "My latest Juniper Pavilion update to Luca says I now prefer "
            "green tea as my afternoon tea instead of black tea.",
            [],
            turn_id="tea-latest",
            session_id="tea-session-3",
            timestamp="2028/08/15 (Tue) 02:07",
        )

        for rows in (single_session, broken_chain):
            with self.subTest(row_count=len(rows)):
                result = self.assess(rows=rows)
                self.assertEqual(
                    result["status"],
                    "insufficient_wrong_attribute_evidence",
                )
                self.assertFalse(result["abstain"])

    def test_ambiguous_wrong_attribute_chains_preserve_fallback(self):
        rows = self.afternoon_tea_rows() + [
            extraction(
                "In the Juniper Pavilion notes I shared with Luca, my work "
                "laptop used to be Aurora K2.",
                [],
                turn_id="laptop-original",
                session_id="laptop-session-1",
                timestamp="2028/07/12 (Wed) 02:07",
            ),
            extraction(
                "For Juniper Pavilion, I told Luca that I changed my work "
                "laptop from Aurora K2 to Fjord Lite.",
                [],
                turn_id="laptop-latest",
                session_id="laptop-session-2",
                timestamp="2028/07/30 (Sun) 02:07",
            ),
        ]

        result = self.assess(rows=rows)

        self.assertEqual(result["status"], "ambiguous_wrong_attribute_chains")
        self.assertFalse(result["abstain"])

    def test_assistant_rows_unscoped_questions_and_ablation_do_not_apply(self):
        assistant_rows = [
            {**row, "source_role": "assistant", "source_speaker": "assistant"}
            for row in self.afternoon_tea_rows()
        ]
        unscoped_question = "Which coffee roast do I currently prefer?"

        assistant_result = self.assess(rows=assistant_rows)
        unscoped_result = self.assess(question=unscoped_question)
        disabled_result = self.assess(disabled=True)

        self.assertEqual(
            assistant_result["status"],
            "insufficient_wrong_attribute_evidence",
        )
        self.assertEqual(unscoped_result["status"], "not_applicable")
        self.assertEqual(disabled_result["status"], "disabled_by_ablation")
        self.assertFalse(assistant_result["abstain"])
        self.assertFalse(unscoped_result["abstain"])
        self.assertFalse(disabled_result["abstain"])


if __name__ == "__main__":
    unittest.main()
