import dataclasses
import datetime
import json
import unittest

import numpy as np

from PRAGMOS_evidence_compiler import (
    OperationProof,
    adjudicate_operation_candidates,
    build_evidence_ledger,
    compiler_state,
    derive_event_status_hint,
    effective_semantic_operator,
    execute_event_count_proof,
    execute_frame_operation_proof,
    execute_grounded_candidate_operation_proof,
    execute_temporal_proof,
    execute_adjudicated_operation_proof,
    parse_semantic_plan,
    plan_query_with_local_slm,
    prefer_stricter_more_complete_proof,
    validate_semantic_route,
)
from PRAGMOS_benchmark_LongMemEval import (
    annotate_measurement_operand_semantics,
    extract_measurement_facts,
    infer_multi_session_operation,
    iter_structured_candidate_fact_rows,
    label_blind_inference_record,
    resolve_measurement_dimension_from_evidence,
    semantic_plan_to_operation_plan,
    should_plan_with_evidence_compiler,
    should_run_evidence_compiler,
)


def explicit_date_resolver(text, _timestamp=None):
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
    }
    lowered = text.lower()
    for name, month in months.items():
        marker = name + " "
        if marker not in lowered:
            continue
        suffix = lowered.split(marker, 1)[1]
        digits = "".join(
            character if character.isdigit() else " " for character in suffix
        )
        values = [int(value) for value in digits.split()]
        if not values:
            continue
        value = datetime.date(values[1] if len(values) > 1 else 2027, month, values[0])
        return {
            "value": value.isoformat(),
            "granularity": "day",
            "interval_start": value.isoformat(),
            "interval_end": value.isoformat(),
        }
    return None


def row(
    quote,
    turn_id,
    session_id,
    role="user",
    speaker="user",
    timestamp="2027-04-30T10:00:00",
):
    return {
        "source_turn_id": turn_id,
        "source_session_id": session_id,
        "source_role": role,
        "source_speaker": speaker,
        "source_timestamp": timestamp,
        "source_quote": quote,
    }


def elapsed_plan(question, first, second, first_entities=(), second_entities=()):
    return parse_semantic_plan(
        {
            "operator": "ELAPSED_TIME",
            "answer_type": "duration",
            "target_unit": "days",
            "actor": "user",
            "confidence": "high",
            "operands": [
                {
                    "id": "START",
                    "text": first,
                    "kind": "event",
                    "actor": "user",
                    "hard_entities": list(first_entities),
                },
                {
                    "id": "END",
                    "text": second,
                    "kind": "event",
                    "actor": "user",
                    "hard_entities": list(second_entities),
                },
            ],
        },
        question,
    )


class FakePlannerContext:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""

    def llm(self, prompt, **_kwargs):
        self.prompt = prompt
        return {"choices": [{"text": json.dumps(self.payload)}]}


class FakeAdjudicatorContext:
    def __init__(self, decisions):
        self.decisions = decisions

    def llm(self, _prompt, **_kwargs):
        return {"choices": [{"text": json.dumps({"decisions": self.decisions})}]}


class FakeFrameContext:
    def __init__(self, frames_by_batch):
        self.frames_by_batch = list(frames_by_batch)
        self.calls = 0

    def llm(self, _prompt, **_kwargs):
        if len(self.frames_by_batch) == 1:
            all_frames = self.frames_by_batch[0]
            frames = [all_frames[self.calls]] if self.calls < len(all_frames) else []
        else:
            frames = self.frames_by_batch[self.calls]
        self.calls += 1
        return {"choices": [{"text": json.dumps({"frames": frames})}]}


class FakeFrameByEvidenceContext:
    def __init__(self, frames):
        self.frames = {
            str(frame_row["evidence_id"]).upper(): frame_row
            for frame_row in frames
        }

    def llm(self, prompt, **_kwargs):
        selected = [
            frame_row
            for evidence_id, frame_row in self.frames.items()
            if f'"evidence_id": "{evidence_id}"' in prompt
        ]
        return {"choices": [{"text": json.dumps({"frames": selected})}]}


class FakeEmbeddingContext:
    def __init__(self, vectors_by_text):
        self.vectors_by_text = vectors_by_text

    def encode_normalized_texts(self, texts):
        return np.asarray([self.vectors_by_text[text] for text in texts])


def frame(
    evidence_id,
    *,
    relevant=True,
    actor_reference="source_speaker",
    status="completed",
    predicate=None,
    subject=None,
    object=None,
    instrument=None,
    theme=None,
    location=None,
    label=None,
    event_identity=None,
    event_context=None,
    value=None,
    unit=None,
    quantity=1,
):
    return {
        "evidence_id": evidence_id,
        "relevant": relevant,
        "actor_reference": actor_reference,
        "status": status,
        "predicate": predicate,
        "subject": subject,
        "object": object,
        "instrument": instrument,
        "theme": theme,
        "location": location,
        "label": label,
        "event_identity": event_identity,
        "event_context": event_context,
        "value": value,
        "unit": unit,
        "quantity": quantity,
    }


class EvidenceCompilerTests(unittest.TestCase):
    def test_compiler_gating_uses_validated_semantics_not_benchmark_labels(self):
        eligible = {"eligible": True}
        ineligible = {"eligible": False}

        self.assertFalse(should_plan_with_evidence_compiler("off", set()))
        self.assertFalse(
            should_plan_with_evidence_compiler("validated", {"evidence_compiler"})
        )
        self.assertTrue(should_plan_with_evidence_compiler("validated", set()))
        self.assertTrue(
            should_run_evidence_compiler(eligible, "validated", set())
        )
        self.assertTrue(should_run_evidence_compiler(eligible, "shadow", set()))
        self.assertFalse(
            should_run_evidence_compiler(ineligible, "validated", set())
        )
        self.assertFalse(should_run_evidence_compiler(eligible, "off", set()))

    def test_label_blind_record_excludes_all_gold_and_category_metadata(self):
        record = {
            "question_type": "temporal-reasoning",
            "answer": "June 4",
            "answer_session_ids": ["s2"],
            "category": 2,
            "question_date": "2027-06-10",
            "haystack_sessions": [[{"role": "user", "content": "hello"}]],
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2027-06-01"],
        }
        inference = label_blind_inference_record(record)
        self.assertEqual(
            set(inference),
            {
                "question_date", "haystack_sessions", "haystack_session_ids",
                "haystack_dates",
            },
        )
        self.assertNotIn("question_type", inference)
        self.assertNotIn("answer", inference)
        self.assertNotIn("answer_session_ids", inference)
        self.assertNotIn("category", inference)

    def test_semantic_router_identifies_temporal_order_without_dataset_label(self):
        question = "Which happened first, moving to Pune or graduating?"
        plan = parse_semantic_plan(
            {
                "operator": "ORDER",
                "answer_type": "event",
                "target_unit": None,
                "actor": "user",
                "answer_role": "object",
                "group_role": "event",
                "requested_status": "completed",
                "result_shape": "value_set",
                "projection_roles": ["event"],
                "confidence": "high",
                "operands": [
                    {
                        "id": "O1", "text": "moving to Pune", "kind": "event",
                        "actor": "user", "hard_entities": ["Pune"],
                    },
                    {
                        "id": "O2", "text": "graduating", "kind": "event",
                        "actor": "user", "hard_entities": [],
                    },
                ],
            },
            question,
        )
        validation = validate_semantic_route(plan, question)
        self.assertEqual(plan.intent_family, "temporal")
        self.assertEqual(plan.temporal_operation, "order")
        self.assertTrue(validation["eligible"])

    def test_factual_lookup_preserves_existing_answer_path(self):
        question = "What degree did I graduate with?"
        plan = parse_semantic_plan(
            {
                "operator": "LOOKUP",
                "answer_type": "text",
                "target_unit": None,
                "actor": "user",
                "answer_role": "text",
                "group_role": "unknown",
                "requested_status": "any",
                "result_shape": "scalar",
                "projection_roles": ["text"],
                "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "degree I graduated with", "kind": "fact",
                    "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        validation = validate_semantic_route(plan, question)
        self.assertEqual(plan.intent_family, "factual")
        self.assertFalse(validation["eligible"])
        self.assertEqual(
            validation["reason"],
            "factual_lookup_uses_existing_answer_path",
        )

    def test_date_lookup_and_relative_join_are_generic_temporal_routes(self):
        cases = [
            (
                "When did I graduate from Pune University?",
                "DATE_LOOKUP",
                "date",
                "date_lookup",
                [{
                    "id": "O1", "text": "graduate from Pune University",
                    "kind": "event", "actor": "user",
                    "hard_entities": ["Pune University"],
                }],
            ),
            (
                "What did I do the day after the Pune interview?",
                "RELATIVE_JOIN",
                "text",
                "relative_join",
                [{
                    "id": "O1", "text": "Pune interview", "kind": "event",
                    "actor": "user", "hard_entities": ["Pune"],
                }],
            ),
        ]
        for question, operator, answer_role, temporal_operation, operands in cases:
            plan = parse_semantic_plan(
                {
                    "operator": operator,
                    "answer_type": answer_role,
                    "target_unit": None,
                    "actor": "user",
                    "answer_role": answer_role,
                    "group_role": "event",
                    "requested_status": "completed",
                    "result_shape": "scalar",
                    "projection_roles": [answer_role],
                    "confidence": "high",
                    "operands": operands,
                },
                question,
            )
            validation = validate_semantic_route(plan, question)
            self.assertTrue(validation["eligible"], validation)
            self.assertEqual(validation["intent_family"], "temporal")
            self.assertEqual(
                validation["temporal_operation"], temporal_operation
            )
            translated = semantic_plan_to_operation_plan(
                plan,
                {
                    "operation": "none",
                    "target_unit": None,
                    "answer_dimension": None,
                    "target_tokens": [],
                    "minimum_sessions": 1,
                },
                question,
            )
            self.assertEqual(
                translated["operation"],
                "temporal_date" if operator == "DATE_LOOKUP" else "temporal_join",
            )

    def test_one_event_elapsed_time_requires_only_the_question_clock(self):
        question = "How long has it been since I moved to Pune?"
        plan = parse_semantic_plan(
            {
                "operator": "ELAPSED_TIME",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "answer_role": "duration",
                "group_role": "event",
                "requested_status": "completed",
                "result_shape": "scalar",
                "projection_roles": ["duration"],
                "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "moved to Pune", "kind": "event",
                    "actor": "user", "hard_entities": ["Pune"],
                }],
            },
            question,
        )
        self.assertFalse(validate_semantic_route(plan, question)["eligible"])
        self.assertTrue(
            validate_semantic_route(
                plan,
                question,
                question_timestamp="2027-06-10",
            )["eligible"]
        )

    def test_duration_role_routes_difference_over_events_to_temporal_executor(self):
        question = "What is the day difference between launch and review?"
        plan = parse_semantic_plan(
            {
                "operator": "DIFFERENCE", "answer_type": "quantity",
                "target_unit": "days", "actor": "user",
                "answer_role": "duration", "group_role": "event",
                "requested_status": "completed", "confidence": "medium",
                "operands": [
                    {
                        "id": "O1", "text": "launch", "kind": "event",
                        "actor": "user", "hard_entities": [],
                    },
                    {
                        "id": "O2", "text": "review", "kind": "event",
                        "actor": "user", "hard_entities": [],
                    },
                ],
            },
            question,
        )
        routed = semantic_plan_to_operation_plan(
            plan,
            {
                "operation": "none", "target_unit": None,
                "answer_dimension": None, "target_tokens": [],
                "minimum_sessions": 1,
            },
            question,
        )
        self.assertEqual(routed["operation"], "temporal_difference")
        self.assertEqual(routed["answer_dimension"], "duration")

    def test_low_confidence_frame_plan_is_retained_with_a_warning(self):
        legacy = {"operation": "none", "target_tokens": ["unchanged"]}
        invalid = parse_semantic_plan(
            {
                "operator": "SUM",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "confidence": "low",
                "operands": [{
                    "id": "O1",
                    "text": "camping days",
                    "kind": "quantity",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            "How many camping days were there?",
        )

        translated = semantic_plan_to_operation_plan(
            invalid, legacy, "How many camping days were there?"
        )

        self.assertTrue(invalid.valid)
        self.assertIn("planner_confidence:low", invalid.warnings)
        self.assertEqual(translated["operation"], "sum")

    def test_event_status_hints_are_explicit_and_conservative(self):
        self.assertEqual(
            derive_event_status_hint("I just got back from a five-day trip."),
            "completed",
        )
        self.assertEqual(
            derive_event_status_hint("I am planning a ten-day trek."),
            "planned",
        )
        self.assertEqual(
            derive_event_status_hint("I never attended the conference."),
            "negated",
        )
        self.assertIsNone(
            derive_event_status_hint("The conference is in Berlin."),
        )

    def test_local_planner_accepts_valid_structured_json(self):
        payload = {
            "operator": "COUNT_DISTINCT",
            "answer_type": "count",
            "target_unit": "items",
            "actor": "user",
            "confidence": "high",
            "operands": [{
                "id": "O1",
                "text": "camping trips",
                "kind": "item",
                "actor": "user",
                "hard_entities": [],
            }],
        }
        context = FakePlannerContext(payload)

        plan = plan_query_with_local_slm(
            context,
            "How many distinct camping trips did I take?",
        )

        self.assertTrue(plan.valid)
        self.assertEqual(plan.operator, "COUNT_DISTINCT")
        self.assertEqual(plan.actor, "user")
        self.assertEqual(plan.operands[0].actor, "user")
        self.assertIn("semantic query planner", context.prompt)

    def test_ungrounded_hard_entities_are_dropped_without_losing_the_plan(self):
        question = "How many days passed between visiting MoMA and the harbor?"
        plan = elapsed_plan(
            question,
            "visiting MoMA",
            "visiting the harbor",
            first_entities=("invented museum",),
        )

        self.assertTrue(plan.valid)
        self.assertEqual(plan.operands[0].hard_entities, ())
        self.assertIn(
            "dropped_ungrounded_hard_entity:invented museum",
            plan.warnings,
        )

    def test_semantic_translation_preserves_legacy_target_tokens(self):
        question = "How many distinct camping trips did I take this year?"
        legacy = infer_multi_session_operation(question, "multi-session")
        plan = parse_semantic_plan(
            {
                "operator": "COUNT_DISTINCT",
                "answer_type": "count",
                "target_unit": None,
                "actor": None,
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "camping trips taken this year",
                    "kind": "item",
                    "actor": None,
                    "hard_entities": ["camping trips"],
                }],
            },
            question,
        )

        translated = semantic_plan_to_operation_plan(plan, legacy, question)

        self.assertEqual(translated["operation"], "count_distinct")
        self.assertEqual(translated["target_tokens"], legacy["target_tokens"])
        self.assertTrue(translated["semantic_compiler_routed"])

    def test_event_duration_difference_routes_to_temporal_executor(self):
        question = "How many days passed between visiting MoMA and visiting the harbor?"
        legacy = infer_multi_session_operation(question, "temporal-reasoning")
        plan = parse_semantic_plan(
            {
                "operator": "DIFFERENCE",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "confidence": "high",
                "operands": [
                    {
                        "id": "O1",
                        "text": "visiting MoMA",
                        "kind": "event",
                        "actor": "user",
                        "hard_entities": ["MoMA"],
                    },
                    {
                        "id": "O2",
                        "text": "visiting the harbor",
                        "kind": "event",
                        "actor": "user",
                        "hard_entities": ["harbor"],
                    },
                ],
            },
            question,
        )

        translated = semantic_plan_to_operation_plan(plan, legacy, question)

        self.assertEqual(translated["operation"], "temporal_difference")
        self.assertEqual(len(translated["temporal_event_specs"]), 2)

    def test_count_of_duration_units_routes_to_sum(self):
        question = "How many days did I spend on camping trips this year?"
        legacy = infer_multi_session_operation(question, "multi-session")
        plan = parse_semantic_plan(
            {
                "operator": "COUNT",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "camping trips this year",
                    "kind": "quantity",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )

        translated = semantic_plan_to_operation_plan(plan, legacy, question)

        self.assertEqual(translated["operation"], "sum")
        self.assertEqual(translated["target_unit"], "days")
        self.assertEqual(translated["answer_dimension"], "duration")
        self.assertTrue(translated["semantic_compiler_routed"])

    def test_semantic_average_drops_incompatible_legacy_date_unit(self):
        question = "What arithmetic mean results from the completed session lengths?"
        legacy = {
            "operation": "temporal_date",
            "target_unit": "date",
            "answer_dimension": "date",
            "target_tokens": ["session", "length"],
            "requires_session_diversity": False,
            "minimum_sessions": 1,
        }
        plan = parse_semantic_plan(
            {
                "operator": "AVERAGE", "answer_type": "number",
                "answer_role": "duration", "group_role": "event",
                "requested_status": "completed", "target_unit": None,
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "completed session lengths",
                    "kind": "quantity", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )

        translated = semantic_plan_to_operation_plan(plan, legacy, question)

        self.assertEqual(translated["operation"], "average")
        self.assertIsNone(translated["target_unit"])
        self.assertEqual(translated["answer_dimension"], "duration")

    def test_measurement_dimension_is_resolved_from_grounded_evidence(self):
        question = "What gap did I record between taxi travel and rail travel?"
        plan = {
            "operation": "difference",
            "target_unit": None,
            "answer_dimension": "duration",
            "operands": ["taxi travel", "rail travel"],
            "target_tokens": ["taxi", "rail", "travel"],
            "strict_grounded_selection": True,
        }
        evidence = [
            row("My completed taxi travel fare was $51.", "t1", "s1"),
            row("My completed rail travel fare was $31.", "t2", "s2"),
        ]

        resolved = resolve_measurement_dimension_from_evidence(
            evidence,
            question,
            plan,
        )

        self.assertEqual(resolved["target_unit"], "money")
        self.assertEqual(resolved["answer_dimension"], "amount")
        self.assertEqual(
            resolved["measurement_dimension_resolution"]["selected_dimension"],
            "money",
        )

    def test_unitless_item_sum_is_type_corrected_to_count(self):
        question = "What tally do the conversations give for project items I acquired?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM",
                "answer_type": "quantity",
                "target_unit": None,
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "project items",
                    "kind": "item",
                    "actor": "user",
                    "hard_entities": ["project items"],
                }],
            },
            question,
        )

        translated = semantic_plan_to_operation_plan(
            plan,
            {"operation": "none", "target_tokens": []},
            question,
        )

        self.assertEqual(effective_semantic_operator(plan), "COUNT")
        self.assertEqual(translated["operation"], "count")
        self.assertEqual(translated["semantic_compiler_typed_operator"], "COUNT")

    def test_provenance_event_count_excludes_plans_and_other_actors(self):
        question = (
            "How many separate cancer treatment items moved beyond planning "
            "into completed purchases?"
        )
        plan = parse_semantic_plan(
            {
                "operator": "COUNT_DISTINCT",
                "answer_type": "count",
                "target_unit": "items",
                "actor": None,
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "cancer treatment items",
                    "kind": "item",
                    "actor": None,
                    "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger(
            [
                row(
                    "I completed the acquisition rather than merely planning it: "
                    "I brought home the coral workshop for my cancer treatment setup.",
                    "a1", "answer-1",
                ),
                row(
                    "The discussion continued. I completed the acquisition rather "
                    "than merely planning it: I brought home the forest course for "
                    "my cancer treatment setup.",
                    "a2", "answer-2",
                ),
                row(
                    "Sofia completed a separate cancer treatment purchase that "
                    "should not be counted as mine.",
                    "n1", "noise-1",
                ),
                row(
                    "I estimated 180 dollars for a future cancer treatment plan, "
                    "but it was only a budget.",
                    "n2", "noise-2",
                ),
            ]
        )

        proof, diagnostics = execute_event_count_proof(
            plan,
            question,
            ledger,
            effective_operation="count_distinct",
        )

        self.assertTrue(diagnostics["applicable"])
        self.assertEqual(
            diagnostics["requested_actor_source"],
            "conversation_owner_default",
        )
        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "2")
        self.assertEqual(
            {
                item["source_session_id"]
                for item in proof.calculation["provenance"]
            },
            {"answer-1", "answer-2"},
        )

    def test_event_count_counts_two_distinct_turns_in_one_session(self):
        question = "Counting only things I obtained, how many self-care items are recorded?"
        plan = parse_semantic_plan(
            {
                "operator": "COUNT_DISTINCT",
                "answer_type": "count",
                "target_unit": "items",
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "self-care items",
                    "kind": "item",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger(
            [
                row(
                    "I completed the acquisition: I brought home the silver "
                    "notebook for my self-care setup.",
                    "a1", "same-session",
                ),
                row(
                    "I completed the acquisition: I brought home the plum tripod "
                    "for my self-care setup.",
                    "a2", "same-session",
                ),
            ]
        )

        proof, _ = execute_event_count_proof(
            plan,
            question,
            ledger,
            effective_operation="count_distinct",
        )

        self.assertEqual(proof.answer, "2")
        self.assertEqual(proof.calculation["covered_session_count"], 1)

    def test_elapsed_time_uses_unique_provenance_backed_events(self):
        question = (
            "How many days passed between starting my herb garden and harvesting "
            "the herb garden?"
        )
        plan = elapsed_plan(
            question,
            "starting herb garden",
            "harvesting herb garden",
            first_entities=("herb garden",),
            second_entities=("herb garden",),
        )
        ledger = build_evidence_ledger(
            [
                row("I started my herb garden on March 22, 2027.", "t1", "s1"),
                row("I harvested the herb garden on April 15, 2027.", "t2", "s2"),
            ],
            temporal_resolver=explicit_date_resolver,
        )

        proof, diagnostics = execute_temporal_proof(plan, ledger)

        self.assertEqual(proof.status, "COMPLETE")
        self.assertTrue(proof.executable)
        self.assertEqual(proof.answer, "24 days")
        self.assertEqual(len(proof.bindings), 2)
        self.assertEqual(diagnostics["atom_count"], 2)

    def test_named_event_binding_rejects_generic_semantic_neighbor(self):
        question = (
            "How many days passed between visiting MoMA and visiting the Ancient "
            "Civilizations exhibit?"
        )
        plan = elapsed_plan(
            question,
            "visiting MoMA",
            "visiting Ancient Civilizations exhibit",
            first_entities=("MoMA",),
            second_entities=("Ancient Civilizations exhibit",),
        )
        ledger = build_evidence_ledger(
            [
                row(
                    "I visited an art museum with modern art on January 15, 2027.",
                    "noise",
                    "s0",
                ),
                row("I visited MoMA on January 22, 2027.", "moma", "s1"),
                row(
                    "I visited the Ancient Civilizations exhibit on January 29, 2027.",
                    "ancient",
                    "s2",
                ),
            ],
            temporal_resolver=explicit_date_resolver,
        )

        proof, _ = execute_temporal_proof(plan, ledger)

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "7 days")
        self.assertNotIn("noise", {binding.source_turn_id for binding in proof.bindings})

    def test_bound_completed_event_can_use_its_source_turn_date(self):
        question = (
            "How many days passed between my visit to the Museum of Modern Art "
            "(MoMA) and the 'Ancient Civilizations' exhibit?"
        )
        plan = elapsed_plan(
            question,
            "visit to the Museum of Modern Art (MoMA)",
            "'Ancient Civilizations' exhibit",
            first_entities=("Museum of Modern Art", "MoMA"),
            second_entities=("'Ancient Civilizations' exhibit",),
        )
        ledger = build_evidence_ledger(
            [
                row(
                    "I just got back from a guided tour at MoMA.",
                    "moma",
                    "s1",
                    timestamp="2027/01/08 (Fri) 10:00",
                ),
                row(
                    "I attended the Ancient Civilizations exhibit.",
                    "ancient",
                    "s2",
                    timestamp="2027/01/15 (Fri) 10:00",
                ),
            ],
            temporal_resolver=explicit_date_resolver,
            event_status_resolver=lambda _text: "completed",
        )

        proof, _ = execute_temporal_proof(plan, ledger)

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "7 days")
        self.assertTrue(
            all(
                binding.temporal_expression["basis"]
                == "bound_event_source_timestamp"
                for binding in proof.bindings
            )
        )

    def test_actor_binding_excludes_assistant_authored_event(self):
        question = "How many days passed between my launch and my review?"
        plan = elapsed_plan(question, "my launch", "my review")
        ledger = build_evidence_ledger(
            [
                row(
                    "I launched it on May 1, 2027.",
                    "assistant",
                    "s0",
                    role="assistant",
                    speaker="assistant",
                ),
                row("I launched it on May 3, 2027.", "user-launch", "s1"),
                row("I reviewed it on May 8, 2027.", "user-review", "s2"),
            ],
            temporal_resolver=explicit_date_resolver,
        )

        proof, _ = execute_temporal_proof(plan, ledger)

        self.assertEqual(proof.answer, "5 days")
        self.assertNotIn(
            "assistant",
            {binding.source_turn_id for binding in proof.bindings},
        )

    def test_competing_dates_are_conflicting_not_absent(self):
        question = "How many days passed between visiting MoMA and visiting the harbor?"
        plan = elapsed_plan(
            question,
            "visiting MoMA",
            "visiting the harbor",
            first_entities=("MoMA",),
            second_entities=("harbor",),
        )
        ledger = build_evidence_ledger(
            [
                row("I visited MoMA on January 10, 2027.", "m1", "s1"),
                row("I visited MoMA on January 12, 2027.", "m2", "s2"),
                row("I visited the harbor on January 20, 2027.", "h1", "s3"),
            ],
            temporal_resolver=explicit_date_resolver,
        )

        proof, _ = execute_temporal_proof(plan, ledger)

        self.assertEqual(proof.status, "CONFLICTING")
        self.assertFalse(proof.executable)
        self.assertEqual(compiler_state(plan, ledger, proof), "CONFLICTING")

    def test_adjudicated_sum_excludes_planned_future_event(self):
        question = "How many days did I spend on camping trips this year?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "camping trips this year",
                    "kind": "quantity",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )
        candidates = [
            {
                "candidate_id": "M1",
                "source_turn_id": "t1",
                "source_session_id": "s1",
                "source_role": "user",
                "source_quote": "I just got back from a 5-day camping trip to Yellowstone.",
                "clause": "I just got back from a 5-day camping trip to Yellowstone.",
                "raw_value": "5-day",
                "normalized_value": 5.0,
                "target_unit": "day",
            },
            {
                "candidate_id": "M2",
                "source_turn_id": "t2",
                "source_session_id": "s2",
                "source_role": "user",
                "source_quote": "I completed a 3-day camping trip to Big Sur.",
                "clause": "I completed a 3-day camping trip to Big Sur.",
                "raw_value": "3-day",
                "normalized_value": 3.0,
                "target_unit": "day",
            },
            {
                "candidate_id": "M3",
                "source_turn_id": "t3",
                "source_session_id": "s2",
                "source_role": "user",
                "source_quote": "I am planning a 10-day trek in New Zealand.",
                "clause": "I am planning a 10-day trek in New Zealand.",
                "raw_value": "10-day",
                "normalized_value": 10.0,
                "target_unit": "day",
            },
        ]
        context = FakeAdjudicatorContext(
            [
                {"candidate_id": "M1", "include": True, "canonical_identity": "camping trip to Yellowstone", "item_quantity": 1, "status": "completed"},
                {"candidate_id": "M2", "include": True, "canonical_identity": "camping trip to Big Sur", "item_quantity": 1, "status": "completed"},
                {"candidate_id": "M3", "include": False, "canonical_identity": None, "item_quantity": None, "status": "planned"},
            ]
        )

        adjudication = adjudicate_operation_candidates(
            context, question, plan, candidates
        )
        proof = execute_adjudicated_operation_proof(
            plan, "sum", adjudication, minimum_sessions=2, target_unit="day"
        )

        self.assertEqual(adjudication["status"], "COMPLETE")
        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "8 days")

    def test_adjudicator_accepts_grounded_identity_with_omitted_modifier(self):
        question = "How many days did I spend camping?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "camping trips",
                    "kind": "quantity",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )
        candidates = [{
            "candidate_id": "M1",
            "source_turn_id": "t1",
            "source_session_id": "s1",
            "source_role": "user",
            "source_quote": "I returned from a 3-day solo camping trip.",
            "clause": "I returned from a 3-day solo camping trip.",
            "raw_value": "3-day",
            "normalized_value": 3.0,
            "target_unit": "day",
        }]
        context = FakeAdjudicatorContext([{
            "candidate_id": "M1",
            "include": True,
            "canonical_identity": "3-day camping trip",
            "item_quantity": 1,
            "status": "completed",
        }])

        adjudication = adjudicate_operation_candidates(
            context, question, plan, candidates
        )

        self.assertEqual(adjudication["status"], "COMPLETE")
        self.assertTrue(adjudication["decisions"][0]["include"])

    def test_typed_matches_override_an_inconsistent_binary_include(self):
        question = "How many days did I spend camping this year?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "camping this year",
                    "kind": "quantity",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )
        candidates = [{
            "candidate_id": "M1",
            "source_turn_id": "t1",
            "source_session_id": "s1",
            "source_role": "user",
            "source_timestamp": "2027-04-30T10:00:00",
            "source_quote": "I returned from a 3-day camping trip last month.",
            "clause": "I returned from a 3-day camping trip last month.",
            "raw_value": "3-day",
            "normalized_value": 3.0,
            "target_unit": "day",
        }]
        context = FakeAdjudicatorContext([{
            "candidate_id": "M1",
            "include": False,
            "actor_match": "yes",
            "event_match": "yes",
            "time_match": "yes",
            "scope_match": "not_applicable",
            "value_grounded": "yes",
            "scope_basis": "not_applicable",
            "canonical_identity": "3-day camping trip",
            "item_quantity": 1,
            "status": "completed",
        }])

        adjudication = adjudicate_operation_candidates(
            context,
            question,
            plan,
            candidates,
            question_timestamp="2027-04-30T12:00:00",
            effective_operation="sum",
        )
        proof = execute_adjudicated_operation_proof(
            plan,
            "sum",
            adjudication,
            target_unit="day",
        )

        self.assertFalse(adjudication["decisions"][0]["declared_include"])
        self.assertTrue(adjudication["decisions"][0]["include"])
        self.assertEqual(proof.answer, "3 days")

    def test_reviewed_but_rejected_candidates_are_partial_not_absent(self):
        question = "How many days did I spend camping this year?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM",
                "answer_type": "duration",
                "target_unit": "days",
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "camping this year",
                    "kind": "quantity",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )
        candidate = {
            "candidate_id": "M1",
            "source_turn_id": "t1",
            "source_session_id": "s1",
            "source_role": "user",
            "source_quote": "I may take a 3-day trip next year.",
            "clause": "I may take a 3-day trip next year.",
            "raw_value": "3-day",
            "normalized_value": 3.0,
            "target_unit": "day",
        }
        context = FakeAdjudicatorContext([{
            "candidate_id": "M1",
            "include": False,
            "actor_match": "yes",
            "event_match": "unknown",
            "time_match": "no",
            "scope_match": "not_applicable",
            "value_grounded": "yes",
            "scope_basis": "not_applicable",
            "canonical_identity": "3-day trip",
            "item_quantity": 1,
            "status": "unknown",
        }])

        adjudication = adjudicate_operation_candidates(
            context, question, plan, [candidate], effective_operation="sum"
        )
        proof = execute_adjudicated_operation_proof(
            plan, "sum", adjudication, target_unit="day"
        )

        self.assertEqual(proof.status, "PARTIAL")
        self.assertFalse(proof.executable)

    def test_pending_item_count_deduplicates_mentions_and_honors_pair_quantity(self):
        question = "How many items of clothing do I need to pick up?"
        plan = parse_semantic_plan(
            {
                "operator": "COUNT",
                "answer_type": "count",
                "target_unit": "items",
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "items of clothing to pick up",
                    "kind": "item",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )
        candidates = [
            {"candidate_id": "C1", "source_turn_id": "t1", "source_session_id": "s1", "source_role": "user", "source_quote": "I still need to pick up the new pair of boots.", "entity": "new pair of boots"},
            {"candidate_id": "C2", "source_turn_id": "t2", "source_session_id": "s2", "source_role": "user", "source_quote": "I still need to pick up my dry cleaning for the navy blue blazer.", "entity": "navy blue blazer"},
            {"candidate_id": "C3", "source_turn_id": "t3", "source_session_id": "s2", "source_role": "user", "source_quote": "I will pick up the navy blue blazer from dry cleaning.", "entity": "navy blue blazer"},
        ]
        context = FakeAdjudicatorContext(
            [
                {"candidate_id": "C1", "include": True, "canonical_identity": "pair of boots", "item_quantity": 2, "status": "pending"},
                {"candidate_id": "C2", "include": True, "canonical_identity": "navy blue blazer", "item_quantity": 1, "status": "pending"},
                {"candidate_id": "C3", "include": True, "canonical_identity": "navy blue blazer", "item_quantity": 1, "status": "pending"},
            ]
        )

        adjudication = adjudicate_operation_candidates(
            context, question, plan, candidates
        )
        proof = execute_adjudicated_operation_proof(
            plan, "count", adjudication, minimum_sessions=2
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "3")

    def test_typed_frames_count_distinct_instruments_not_projects(self):
        question = "What is the cardinality of the tool set used for floor robot projects?"
        plan = parse_semantic_plan(
            {
                "operator": "COUNT_DISTINCT",
                "answer_type": "number",
                "answer_role": "count",
                "group_role": "instrument",
                "requested_status": "completed",
                "target_unit": None,
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "tools used for floor robot projects",
                    "kind": "item", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row("I used the notebook in a completed floor robot project.", "t1", "s1"),
            row("I used the ticket in a completed floor robot project.", "t2", "s2"),
            row("I used the notebook in another completed floor robot project.", "t3", "s3"),
            row("Priya used the repair in a completed floor robot project.", "t4", "s4"),
        ])
        context = FakeFrameContext([[
            frame("E1", actor_reference="unknown", predicate="used", subject="notebook", object="floor robot project"),
            frame("E2", actor_reference="other", predicate="used", subject="ticket", object="floor robot project"),
            frame("E3", actor_reference="unknown", predicate="used", subject="notebook", object="floor robot project"),
            frame("E4", actor_reference="other", predicate="used", subject="repair", object="floor robot project"),
        ]])

        proof, diagnostics = execute_frame_operation_proof(
            context, question, plan, ledger,
            effective_operation="count_distinct",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "2")
        self.assertEqual(set(proof.calculation["distinct_values"]), {"notebook", "ticket"})
        self.assertEqual(diagnostics["status"], "COMPLETE")

    def test_count_contract_moves_entity_role_without_disabling_route(self):
        question = "How many distinct tools did I use for the completed projects?"
        plan = parse_semantic_plan(
            {
                "operator": "COUNT_DISTINCT", "answer_type": "number",
                "answer_role": "instrument", "group_role": "event",
                "requested_status": "completed", "target_unit": None,
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "tools used for completed projects",
                    "kind": "item", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )

        self.assertEqual(plan.answer_role, "count")
        self.assertEqual(plan.group_role, "instrument")
        self.assertEqual(plan.projection_roles, ("instrument",))
        self.assertTrue(validate_semantic_route(plan, question)["eligible"])

    def test_count_frames_canonicalize_duplicate_extractions_by_source_turn(self):
        question = "How many separate completed habitat acquisitions did I make?"
        plan = parse_semantic_plan(
            {
                "operator": "COUNT_DISTINCT", "answer_type": "number",
                "answer_role": "count", "group_role": "event",
                "requested_status": "completed", "target_unit": None,
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "completed habitat acquisitions",
                    "kind": "event", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row("I completed the first habitat acquisition.", "t1", "s1"),
            row("I completed the first habitat acquisition with details.", "t1", "s1"),
            row("I completed another habitat acquisition.", "t2", "s2"),
        ])
        context = FakeFrameContext([[
            frame("E1", predicate="completed", event_identity="habitat acquisition"),
            frame("E2", predicate="completed", event_identity="completed acquisition"),
            frame("E3", predicate="completed", event_identity="habitat acquisition"),
        ]])

        proof, diagnostics = execute_frame_operation_proof(
            context, question, plan, ledger,
            effective_operation="count_distinct", minimum_sessions=2,
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "2")
        self.assertEqual(diagnostics["canonical_claim_count"], 2)
        self.assertEqual(diagnostics["duplicate_frame_count"], 1)

    def test_typed_frames_sum_money_and_exclude_planned_budget(self):
        question = "What total amount did I spend on completed habitat repairs?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM", "answer_type": "amount",
                "answer_role": "amount", "group_role": "event",
                "requested_status": "completed", "target_unit": "currency",
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "completed habitat repairs",
                    "kind": "quantity", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row("I spent $79 on a completed habitat repair.", "t1", "s1"),
            row("I spent $56 on another completed habitat repair.", "t2", "s2"),
            row("I spent $18 on the final completed habitat repair.", "t3", "s3"),
            row("I planned a $247 budget for a future habitat repair.", "t4", "s4"),
        ])
        context = FakeFrameContext([[
            frame("E1", predicate="spent", theme="habitat repair", value=79, unit="dollars"),
            frame("E2", predicate="spent", theme="habitat repair", value=56, unit="dollars"),
            frame("E3", predicate="spent", theme="habitat repair", value=18, unit="dollars"),
            frame("E4", status="planned", predicate="planned", theme="habitat repair", value=247, unit="dollars"),
        ]])

        proof, _ = execute_frame_operation_proof(
            context, question, plan, ledger,
            effective_operation="sum", minimum_sessions=3, target_unit="currency",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "$153")
        self.assertEqual(proof.calculation["values"], [79.0, 56.0, 18.0])

    def test_grounded_sum_requires_substantive_operand_scope(self):
        question = "What amount results from every recorded sudden mood swings charge?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM", "answer_type": "amount",
                "answer_role": "amount", "group_role": "label",
                "requested_status": "completed", "target_unit": "money",
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "recorded sudden mood swings charge",
                    "kind": "quantity", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        candidates = [
            {
                "candidate_id": "M1", "source_turn_id": "t1",
                "source_session_id": "s1", "source_role": "user",
                "source_speaker": "user", "clause":
                "I paid a completed sudden mood swings charge of $73.",
                "source_quote": "I paid a completed sudden mood swings charge of $73.",
                "numeric_value": 73, "unit": "money",
                "event_status_hint": "completed",
            },
            {
                "candidate_id": "M2", "source_turn_id": "t2",
                "source_session_id": "s2", "source_role": "user",
                "source_speaker": "user", "clause":
                "I paid another completed sudden mood swings charge of $80.",
                "source_quote": "I paid another completed sudden mood swings charge of $80.",
                "numeric_value": 80, "unit": "money",
                "event_status_hint": "completed",
            },
            {
                "candidate_id": "M3", "source_turn_id": "t3",
                "source_session_id": "s3", "source_role": "user",
                "source_speaker": "user", "clause":
                "A competitor routinely charges $200 for marketing.",
                "source_quote": "A competitor routinely charges $200 for marketing.",
                "numeric_value": 200, "unit": "money",
                "event_status_hint": "completed",
            },
        ]

        proof, diagnostics = execute_grounded_candidate_operation_proof(
            plan, candidates, effective_operation="sum", target_unit="money",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "$153")
        self.assertEqual(diagnostics["rejected_counts"]["scope"], 1)

    def test_difference_binds_remaining_operand_after_lexical_assignment(self):
        question = "What fare gap did I record between taxi and rail travel?"
        plan = parse_semantic_plan(
            {
                "operator": "DIFFERENCE", "answer_type": "scalar",
                "answer_role": "amount", "group_role": "event",
                "requested_status": "completed", "target_unit": "money",
                "actor": "user", "confidence": "high",
                "operands": [
                    {"id": "O1", "text": "taxi fare", "kind": "quantity", "actor": "user", "hard_entities": []},
                    {"id": "O2", "text": "rail fare", "kind": "quantity", "actor": "user", "hard_entities": []},
                ],
            },
            question,
        )
        candidates = [
            {
                "candidate_id": "M1", "source_turn_id": "t1",
                "source_session_id": "s1", "source_role": "user",
                "source_speaker": "user", "clause": "The taxi trip cost $91.",
                "source_quote": "The taxi trip cost $91.", "numeric_value": 91,
                "unit": "money", "event_status_hint": "completed",
                "operand_index": 0,
                "semantic_operand_scores": [
                    {"operand_index": 0, "score": 0.55},
                    {"operand_index": 1, "score": 0.32},
                ],
            },
            {
                "candidate_id": "M2", "source_turn_id": "t2",
                "source_session_id": "s2", "source_role": "user",
                "source_speaker": "user", "clause": "I paid $63 for the train trip.",
                "source_quote": "I paid $63 for the train trip.", "numeric_value": 63,
                "unit": "money", "event_status_hint": "completed",
                "operand_index": None,
                "semantic_operand_scores": [
                    {"operand_index": 0, "score": 0.42},
                    {"operand_index": 1, "score": 0.46},
                ],
            },
        ]

        proof, diagnostics = execute_grounded_candidate_operation_proof(
            plan, candidates, effective_operation="difference", target_unit="money",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "$28")
        self.assertEqual(
            diagnostics["operand_binding"]["semantic_bindings"]["1"]["candidate_index"],
            1,
        )

    def test_typed_frames_select_latest_completed_event_across_sessions(self):
        question = "What was the latest thing I did?"
        plan = parse_semantic_plan(
            {
                "operator": "LATEST", "answer_type": "event",
                "answer_role": "object", "group_role": "event",
                "requested_status": "completed", "target_unit": None,
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "latest thing I did",
                    "kind": "event", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row(
                "I completed the harbor workshop.", "t1", "s1",
                timestamp="2027-04-02T10:00:00",
            ),
            row(
                "I completed the gallery interview.", "t2", "s2",
                timestamp="2027-04-09T10:00:00",
            ),
            row(
                "I planned the future studio visit.", "t3", "s3",
                timestamp="2027-04-12T10:00:00",
            ),
        ])
        context = FakeFrameContext([
            [frame(
                "E3", status="planned", predicate="planned",
                event_identity="future studio visit",
            )],
            [frame(
                "E2", predicate="completed", event_identity="gallery interview",
            )],
            [frame(
                "E1", predicate="completed", event_identity="harbor workshop",
            )],
        ])

        proof, diagnostics = execute_frame_operation_proof(
            context,
            question,
            plan,
            ledger,
            effective_operation="temporal_latest",
            minimum_sessions=1,
        )

        self.assertEqual(proof.status, "COMPLETE", diagnostics)
        self.assertEqual(proof.answer, "gallery interview")
        self.assertEqual(proof.calculation["selected_date"], "2027-04-09")
        self.assertEqual(diagnostics["rejected_counts"]["status"], 1)

    def test_grounded_candidates_sum_without_model_selection(self):
        question = "What aggregate payment did the completed repair expenses produce?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM", "answer_type": "amount",
                "answer_role": "amount", "group_role": "event",
                "requested_status": "completed", "target_unit": None,
                "actor": None, "confidence": "medium",
                "operands": [{
                    "id": "O1", "text": "completed repair expenses",
                    "kind": "quantity", "actor": None, "hard_entities": [],
                }],
            },
            question,
        )
        candidates = [
            {
                "candidate_id": "M1", "source_turn_id": "t1",
                "source_session_id": "s1", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "The completed repair expense was $18.",
                "clause": "The completed repair expense was $18.",
                "normalized_value": 18, "target_unit": "money",
            },
            {
                "candidate_id": "M2", "source_turn_id": "t2",
                "source_session_id": "s2", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "I spent $56 on the completed repair.",
                "clause": "I spent $56 on the completed repair.",
                "normalized_value": 56, "target_unit": "currency",
            },
            {
                "candidate_id": "M3", "source_turn_id": "t3",
                "source_session_id": "s3", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "I estimated $247 for a future repair plan, but it was never spent.",
                "clause": "I estimated $247 for a future repair plan.",
                "normalized_value": 247, "target_unit": "money",
            },
            {
                "candidate_id": "M4", "source_turn_id": "t4",
                "source_session_id": "s4", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "Priya spent $99 on a completed repair.",
                "clause": "Priya spent $99 on a completed repair.",
                "normalized_value": 99, "target_unit": "money",
            },
        ]

        proof, diagnostics = execute_grounded_candidate_operation_proof(
            plan,
            candidates,
            effective_operation="sum",
            minimum_sessions=2,
            target_unit="money",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "$74")
        self.assertEqual(diagnostics["eligible_candidate_count"], 2)
        self.assertEqual(diagnostics["rejected_counts"]["status"], 1)
        self.assertEqual(diagnostics["rejected_counts"]["actor"], 1)

    def test_grounded_candidate_argmax_returns_associated_label(self):
        question = "Which completed supply expense was the highest?"
        plan = parse_semantic_plan(
            {
                "operator": "ARGMAX", "answer_type": "label",
                "answer_role": "label", "group_role": "label",
                "requested_status": "completed", "target_unit": "money",
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "completed supply expenses",
                    "kind": "quantity", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        candidates = [
            {
                "candidate_id": "M1", "source_turn_id": "t1",
                "source_session_id": "s1", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "My completed supply expense at notebook was $42.",
                "clause": "My completed supply expense at notebook was $42.",
                "normalized_value": 42, "target_unit": "money",
            },
            {
                "candidate_id": "M2", "source_turn_id": "t2",
                "source_session_id": "s2", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "My completed supply expense at toolkit was $90.",
                "clause": "My completed supply expense at toolkit was $90.",
                "normalized_value": 90, "target_unit": "money",
            },
        ]

        proof, _ = execute_grounded_candidate_operation_proof(
            plan,
            candidates,
            effective_operation="argmax",
            minimum_sessions=2,
            target_unit="money",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "toolkit")

    def test_semantic_operand_binding_resolves_grounded_paraphrase(self):
        question = "What was the difference between taxi travel and rail travel?"
        plan = parse_semantic_plan(
            {
                "operator": "DIFFERENCE", "answer_type": "amount",
                "answer_role": "amount", "group_role": "event",
                "requested_status": "completed", "target_unit": "money",
                "actor": "user", "confidence": "high",
                "operands": [
                    {
                        "id": "O1", "text": "taxi travel", "kind": "quantity",
                        "actor": "user", "hard_entities": [],
                    },
                    {
                        "id": "O2", "text": "rail travel", "kind": "quantity",
                        "actor": "user", "hard_entities": [],
                    },
                ],
            },
            question,
        )
        candidates = [
            {
                "candidate_id": "M1", "source_turn_id": "t1",
                "source_session_id": "s1", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "The completed taxi portion cost $51.",
                "clause": "The completed taxi portion cost $51.",
                "normalized_value": 51, "target_unit": "money",
                "operand_index": 0, "operand_match_score": 0.5,
                "semantic_operand_scores": [
                    {"operand_index": 0, "score": 0.52},
                    {"operand_index": 1, "score": 0.21},
                ],
            },
            {
                "candidate_id": "M2", "source_turn_id": "t2",
                "source_session_id": "s2", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "The completed train portion cost $31.",
                "clause": "The completed train portion cost $31.",
                "normalized_value": 31, "target_unit": "money",
                "operand_index": None, "operand_match_score": 0.0,
                "semantic_operand_scores": [
                    {"operand_index": 0, "score": 0.17},
                    {"operand_index": 1, "score": 0.36},
                ],
            },
            {
                "candidate_id": "M3", "source_turn_id": "t3",
                "source_session_id": "s3", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "I planned a $300 budget for rail travel.",
                "clause": "I planned a $300 budget for rail travel.",
                "normalized_value": 300, "target_unit": "money",
                "operand_index": 1, "operand_match_score": 1.0,
                "semantic_operand_scores": [
                    {"operand_index": 0, "score": 0.28},
                    {"operand_index": 1, "score": 0.63},
                ],
            },
        ]

        proof, diagnostics = execute_grounded_candidate_operation_proof(
            plan, candidates, effective_operation="difference",
            minimum_sessions=2, target_unit="money",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "$20")
        self.assertEqual(diagnostics["rejected_counts"]["status"], 1)
        self.assertEqual(
            diagnostics["operand_binding"]["semantic_bindings"]["1"][
                "candidate_index"
            ],
            1,
        )

    def test_semantic_operand_binding_abstains_when_candidates_are_ambiguous(self):
        question = "What was the difference between taxi travel and rail travel?"
        plan = parse_semantic_plan(
            {
                "operator": "DIFFERENCE", "answer_type": "amount",
                "answer_role": "amount", "group_role": "event",
                "requested_status": "completed", "target_unit": "money",
                "actor": "user", "confidence": "high",
                "operands": [
                    {"id": "O1", "text": "taxi travel", "kind": "quantity", "actor": "user", "hard_entities": []},
                    {"id": "O2", "text": "rail travel", "kind": "quantity", "actor": "user", "hard_entities": []},
                ],
            },
            question,
        )
        candidates = [
            {
                "candidate_id": "M1", "source_turn_id": "t1",
                "source_session_id": "s1", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "The completed taxi portion cost $51.",
                "clause": "The completed taxi portion cost $51.",
                "normalized_value": 51, "target_unit": "money",
                "operand_index": 0,
            },
            {
                "candidate_id": "M2", "source_turn_id": "t2",
                "source_session_id": "s2", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "The completed train portion cost $31.",
                "clause": "The completed train portion cost $31.",
                "normalized_value": 31, "target_unit": "money",
                "semantic_operand_scores": [
                    {"operand_index": 0, "score": 0.17},
                    {"operand_index": 1, "score": 0.36},
                ],
            },
            {
                "candidate_id": "M3", "source_turn_id": "t3",
                "source_session_id": "s3", "source_role": "user",
                "source_speaker": "user",
                "source_quote": "The completed subway portion cost $29.",
                "clause": "The completed subway portion cost $29.",
                "normalized_value": 29, "target_unit": "money",
                "semantic_operand_scores": [
                    {"operand_index": 0, "score": 0.16},
                    {"operand_index": 1, "score": 0.35},
                ],
            },
        ]

        proof, diagnostics = execute_grounded_candidate_operation_proof(
            plan, candidates, effective_operation="difference",
            minimum_sessions=2, target_unit="money",
        )

        self.assertEqual(proof.status, "PARTIAL")
        self.assertEqual(
            proof.reason,
            "multi_operand_binding_incomplete_or_ambiguous",
        )
        self.assertEqual(
            diagnostics["operand_binding"]["ambiguous_operand_indexes"],
            [1],
        )

    def test_dense_operand_annotations_do_not_replace_lexical_binding(self):
        candidates = [
            {
                "candidate_id": "M1", "clause": "completed taxi fare",
                "operand_index": 0,
            },
            {
                "candidate_id": "M2", "clause": "completed train fare",
                "operand_index": None,
            },
        ]
        vectors = {
            "taxi travel": [1.0, 0.0],
            "rail travel": [0.0, 1.0],
            "completed taxi fare": [0.9, 0.1],
            "completed train fare": [0.1, 0.9],
        }

        annotated = annotate_measurement_operand_semantics(
            FakeEmbeddingContext(vectors),
            candidates,
            {
                "operation": "difference",
                "operands": ["taxi travel", "rail travel"],
            },
        )

        self.assertEqual(annotated[0]["operand_index"], 0)
        self.assertIsNone(annotated[1]["operand_index"])
        self.assertEqual(
            annotated[1]["semantic_operand_binding"]["best_operand_index"],
            1,
        )
        self.assertNotIn("semantic_operand_scores", candidates[0])

    def test_measurement_candidates_reuse_existing_event_status_policy(self):
        operation_plan = {
            "operation": "difference",
            "target_unit": "money",
            "answer_dimension": "amount",
            "operands": ["taxi travel", "rail travel"],
        }
        candidates = extract_measurement_facts(
            [
                row("The taxi portion of the trip cost $51.", "t1", "s1"),
                row("I paid $31 for the train portion of the trip.", "t2", "s2"),
                row(
                    "I estimated $300 for a future trip, but it was never spent.",
                    "t3", "s3",
                ),
            ],
            "What was the trip fare gap between taxi travel and rail travel?",
            operation_plan,
        )

        statuses = {
            candidate["source_turn_id"]: candidate["event_status_hint"]
            for candidate in candidates
        }
        self.assertEqual(statuses["t1"], "completed")
        self.assertEqual(statuses["t2"], "completed")
        self.assertEqual(statuses["t3"], "negated_or_cancelled")

    def test_measurement_status_is_bound_to_exact_clause(self):
        operation_plan = {
            "operation": "sum",
            "target_unit": "money",
            "answer_dimension": "amount",
            "operands": ["completed printing expense"],
        }
        candidates = extract_measurement_facts(
            [
                row(
                    "I paid $69 for the completed printing expense. "
                    "I will budget $300 for a later project.",
                    "t1",
                    "s1",
                )
            ],
            "What was the completed printing expense?",
            operation_plan,
        )

        paid = next(candidate for candidate in candidates if candidate["numeric_value"] == 69)
        self.assertEqual(paid["event_status_hint"], "completed")
        self.assertEqual(paid["source_event_status_hint"], "planned")

    def test_single_set_tuple_contract_corrects_intersection_to_collect(self):
        plan = parse_semantic_plan(
            {
                "operator": "INTERSECTION",
                "answer_type": "paired rows",
                "answer_role": "tuple",
                "group_role": "object",
                "requested_status": "completed",
                "result_shape": "tuple_set",
                "projection_roles": ["object", "location"],
                "target_unit": None,
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "completed project records",
                    "kind": "fact",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            "Which completed project items belong with each location?",
        )

        self.assertTrue(plan.valid)
        self.assertEqual(plan.operator, "COLLECT")
        self.assertEqual(plan.result_shape, "tuple_set")
        self.assertEqual(plan.projection_roles, ("object", "location"))

    def test_mapping_contract_recovers_tuple_shape_and_canonical_role_order(self):
        question = "Which item belongs with each location across the finished activities?"
        plan = parse_semantic_plan(
            {
                "operator": "INTERSECTION", "answer_type": "value_set",
                "answer_role": "location", "group_role": "object",
                "requested_status": "completed", "result_shape": "value_set",
                "projection_roles": ["object"], "target_unit": None,
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "finished protective womb activities",
                    "kind": "event", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )

        self.assertEqual(plan.operator, "COLLECT")
        self.assertEqual(plan.result_shape, "tuple_set")
        self.assertEqual(plan.projection_roles, ("object", "location"))

    def test_tuple_reconstruction_uses_one_actor_bound_frame_per_pair(self):
        question = "Which item belongs with each location across my finished activities?"
        plan = parse_semantic_plan(
            {
                "operator": "COLLECT", "answer_type": "tuple_set",
                "answer_role": "tuple", "group_role": "object",
                "requested_status": "completed", "result_shape": "tuple_set",
                "projection_roles": ["location", "object"], "target_unit": None,
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "finished protective womb activities",
                    "kind": "event", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row(
                "I completed the protective womb activity using the toolkit at Orchid Gallery.",
                "t1", "s1",
            ),
            row(
                "I completed the protective womb activity using the notebook at Harbor Annex.",
                "t2", "s2",
            ),
            row(
                "Another protective womb exchange connected locations and items.",
                "t3", "s3",
            ),
        ])
        context = FakeFrameByEvidenceContext([
            frame(
                "E1", object="protective womb activity", instrument="toolkit",
                location="Orchid Gallery", event_identity="completed protective womb activity",
            ),
            frame(
                "E2", subject="notebook", object="protective womb activity",
                location="Harbor Annex", event_identity="completed protective womb activity",
            ),
            frame(
                "E3", actor_reference="source_speaker",
                subject="protective womb exchange", object="locations",
                location="connected locations",
            ),
        ])

        proof, diagnostics = execute_frame_operation_proof(
            context, question, plan, ledger,
            effective_operation="collect_distinct", minimum_sessions=2,
            strict_contract=True,
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(
            proof.answer,
            "toolkit at Orchid Gallery, notebook at Harbor Annex",
        )
        self.assertEqual(diagnostics["rejected_counts"]["actor_unresolved"], 1)

    def test_irrelevant_frames_cannot_enter_distinct_count_proof(self):
        question = "How many distinct completed project items did I acquire?"
        plan = parse_semantic_plan(
            {
                "operator": "COUNT_DISTINCT",
                "answer_type": "number",
                "answer_role": "count",
                "group_role": "event",
                "requested_status": "completed",
                "result_shape": "scalar",
                "projection_roles": ["event"],
                "target_unit": None,
                "actor": "user",
                "confidence": "high",
                "operands": [{
                    "id": "O1",
                    "text": "completed project acquisitions",
                    "kind": "event",
                    "actor": "user",
                    "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row("I acquired the coral notebook for the project.", "t1", "s1"),
            row("The algorithm returned useful project context.", "t2", "s2"),
        ])
        context = FakeFrameContext([[
            frame(
                "E1",
                predicate="acquired",
                object="coral notebook",
                event_identity="coral notebook",
                event_context="project",
            ),
            frame(
                "E2",
                relevant=False,
                predicate="returned",
                object="useful project context",
                event_identity="useful project context",
                event_context="project",
            ),
        ]])

        proof, diagnostics = execute_frame_operation_proof(
            context,
            question,
            plan,
            ledger,
            effective_operation="count_distinct",
        )

        self.assertEqual(proof.answer, "1")
        self.assertEqual(diagnostics["rejected_counts"]["relevance"], 1)
        self.assertTrue(all(proof.calculation["proof_certificate"].values()))

    def test_more_complete_proof_replacement_requires_strict_certificate(self):
        current = OperationProof(
            status="COMPLETE",
            operator="COUNT_DISTINCT",
            answer="2",
            calculation={
                "covered_session_count": 2,
                "provenance": [{"id": 1}, {"id": 2}],
            },
            executable=True,
        )
        strict = OperationProof(
            status="COMPLETE",
            operator="COUNT_DISTINCT",
            answer="4",
            calculation={
                "covered_session_count": 3,
                "provenance": [{"id": value} for value in range(4)],
                "proof_certificate": {"plan_valid": True, "scope_bound": True},
            },
            executable=True,
        )
        weak = dataclasses.replace(
            strict,
            answer="5",
            calculation={
                **strict.calculation,
                "proof_certificate": {"plan_valid": True, "scope_bound": False},
            },
        )

        self.assertIs(prefer_stricter_more_complete_proof(current, strict), strict)
        self.assertIs(prefer_stricter_more_complete_proof(current, weak), current)

    def test_nested_operation_candidate_facts_are_serialized_safely(self):
        nested = {
            "event_specs": ["metadata-only"],
            "event_candidates": {
                "T1": [
                    {
                        "source_turn_id": "t1",
                        "source_session_id": "s1",
                        "source_quote": "I completed the first event today.",
                    }
                ],
                "T2": [],
            },
            "selected_events": [
                {
                    "source_turn_id": "t2",
                    "source_session_id": "s2",
                    "clause": "I completed the second event today",
                }
            ],
            "calculation": {"operator": "elapsed", "result": 7},
        }

        rows = list(iter_structured_candidate_fact_rows(nested))

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["source_turn_id"] for row in rows},
            {"t1", "t2"},
        )

    def test_typed_frames_average_durations(self):
        question = "What arithmetic mean results from the completed session lengths?"
        plan = parse_semantic_plan(
            {
                "operator": "AVERAGE", "answer_type": "duration",
                "answer_role": "duration", "group_role": "event",
                "requested_status": "completed", "target_unit": "minutes",
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "completed session lengths",
                    "kind": "quantity", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row("The completed session lasted 12 minutes.", "t1", "s1"),
            row("The completed session lasted 24 minutes.", "t2", "s2"),
            row("The completed session lasted 40 minutes.", "t3", "s3"),
        ])
        context = FakeFrameContext([[
            frame("E1", predicate="lasted", theme="session", value=12, unit="minutes"),
            frame("E2", predicate="lasted", theme="session", value=24, unit="minutes"),
            frame("E3", predicate="lasted", theme="session", value=40, unit="minutes"),
        ]])

        proof, _ = execute_frame_operation_proof(
            context, question, plan, ledger,
            effective_operation="average", minimum_sessions=3,
            target_unit="minutes",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "25.3333 minutes")

    def test_typed_frames_extremum_returns_associated_label(self):
        question = "Which supply label had the highest completed ceramics cost?"
        plan = parse_semantic_plan(
            {
                "operator": "ARGMAX", "answer_type": "label",
                "answer_role": "label", "group_role": "label",
                "requested_status": "completed", "target_unit": "money",
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "completed ceramics supply costs",
                    "kind": "quantity", "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row("The completed ceramics toolkit cost $18.", "t1", "s1"),
            row("The completed ceramics course cost $31.", "t2", "s2"),
        ])
        context = FakeFrameContext([[
            frame("E1", predicate="cost", label="toolkit", theme="ceramics", value=18, unit="dollars"),
            frame("E2", predicate="cost", label="course", theme="ceramics", value=31, unit="dollars"),
        ]])

        proof, _ = execute_frame_operation_proof(
            context, question, plan, ledger,
            effective_operation="argmax", minimum_sessions=2, target_unit="money",
        )

        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "course")
        self.assertEqual(proof.calculation["result"], 31.0)

    def test_typed_frames_difference_and_structured_collection(self):
        difference_question = "What is the difference between the two completed repair costs?"
        difference_plan = parse_semantic_plan(
            {
                "operator": "DIFFERENCE", "answer_type": "amount",
                "answer_role": "amount", "group_role": "event",
                "requested_status": "completed", "target_unit": "usd",
                "actor": "user", "confidence": "high",
                "operands": [
                    {"id": "O1", "text": "first repair cost", "kind": "quantity", "actor": "user", "hard_entities": []},
                    {"id": "O2", "text": "second repair cost", "kind": "quantity", "actor": "user", "hard_entities": []},
                ],
            },
            difference_question,
        )
        difference_ledger = build_evidence_ledger([
            row("The first completed repair cost $51.", "t1", "s1"),
            row("The second completed repair cost $31.", "t2", "s2"),
        ])
        difference_context = FakeFrameContext([[
            frame("E1", predicate="cost", theme="repair", value=51, unit="dollars"),
            frame("E2", predicate="cost", theme="repair", value=31, unit="dollars"),
        ]])

        proof, _ = execute_frame_operation_proof(
            difference_context, difference_question, difference_plan,
            difference_ledger, effective_operation="difference",
            minimum_sessions=2, target_unit="usd",
        )
        self.assertEqual(proof.answer, "$20")

        collect_question = "Which project materials and places were completed?"
        collect_plan = parse_semantic_plan(
            {
                "operator": "COLLECT", "answer_type": "tuple",
                "answer_role": "tuple", "group_role": "object",
                "requested_status": "completed", "target_unit": None,
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "completed project materials and places",
                    "kind": "fact", "actor": "user", "hard_entities": [],
                }],
            },
            collect_question,
        )
        collect_ledger = build_evidence_ledger([
            row("I completed the toolkit project at Fern Court.", "t3", "s3"),
            row("I completed the course project at Maple Commons.", "t4", "s4"),
        ])
        collect_context = FakeFrameContext([[
            frame("E1", predicate="completed", object="toolkit", theme="project", location="Fern Court"),
            frame("E2", predicate="completed", object="course", theme="project", location="Maple Commons"),
        ]])

        proof, _ = execute_frame_operation_proof(
            collect_context, collect_question, collect_plan, collect_ledger,
            effective_operation="collect_distinct", minimum_sessions=2,
        )
        self.assertEqual(proof.status, "COMPLETE")
        self.assertEqual(proof.answer, "toolkit at Fern Court, course at Maple Commons")

    def test_incomplete_frame_review_cannot_create_executable_proof(self):
        question = "What total amount did I spend on repairs?"
        plan = parse_semantic_plan(
            {
                "operator": "SUM", "answer_type": "amount",
                "answer_role": "amount", "group_role": "event",
                "requested_status": "completed", "target_unit": "money",
                "actor": "user", "confidence": "high",
                "operands": [{
                    "id": "O1", "text": "repair spending", "kind": "quantity",
                    "actor": "user", "hard_entities": [],
                }],
            },
            question,
        )
        ledger = build_evidence_ledger([
            row("I spent $12 on a completed repair.", "t1", "s1"),
        ])
        context = FakeFrameContext([[]])

        proof, diagnostics = execute_frame_operation_proof(
            context, question, plan, ledger,
            effective_operation="sum", target_unit="money",
        )

        self.assertEqual(diagnostics["status"], "EXTRACTION_FAILED")
        self.assertEqual(proof.status, "EXTRACTION_FAILED")
        self.assertFalse(proof.executable)


if __name__ == "__main__":
    unittest.main()
