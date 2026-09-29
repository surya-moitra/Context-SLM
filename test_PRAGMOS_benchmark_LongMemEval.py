import json
import tempfile
import unittest
from pathlib import Path

from PRAGMOS_benchmark_LongMemEval import (
    aggregate_longmemeval_retrieval_metrics,
    answer_session_recall,
    build_query_profile,
    canonicalize_generated_scalar_answer,
    deduplicate_operation_facts,
    execute_operation_plan,
    extract_candidates_from_selected_evidence,
    extract_measurement_facts,
    filter_candidates_by_query_anchors,
    infer_multi_session_operation,
    infer_requested_answer_slot,
    longmemeval_session_retrieval_metrics,
    parse_number_value,
    prepare_run_checkpoint,
    format_answer_slot_candidates,
    rerank_answer_slot_candidates,
    select_answer_slot_candidates,
    select_session_diverse_memories,
    session_neighbor_priority,
    temporal_join_result,
    with_manifest_fingerprint,
)


def extraction(
    quote,
    triples,
    role="user",
    turn_id="turn-1",
    session_id="session-1",
    timestamp="2026-01-01T10:00:00Z",
):
    return {
        "source_turn_id": turn_id,
        "source_session_id": session_id,
        "source_role": role,
        "source_speaker": role,
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


if __name__ == "__main__":
    unittest.main()
