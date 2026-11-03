import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PRAGMOS_evidence_compiler import (
    EvidenceFrame,
    build_evidence_ledger,
    count_scope_contract,
    derive_event_status_hint,
    execute_event_count_proof,
    execute_frame_operation_proof,
    expand_count_compiler_sources,
    parse_semantic_plan,
)
from PRAGMOS_benchmark_LongMemEval import restore_count_source_turns
from test_PRAGMOS_evidence_compiler import FakeFrameByEvidenceContext, frame, row


def count_plan(question, operand, *, entities=(), group="instrument"):
    return parse_semantic_plan({
        "operator": "COUNT_DISTINCT", "answer_type": "number",
        "answer_role": "count", "group_role": group,
        "requested_status": "completed", "target_unit": None,
        "actor": "user", "confidence": "high",
        "operands": [{
            "id": "O1", "text": operand, "kind": "item", "actor": "user",
            "hard_entities": list(entities),
        }],
    }, question)


class CountContractTests(unittest.TestCase):
    def test_generic_type_constraints_recover_question_topic(self):
        cases = (
            ("How many distinct tools did I use for seaside restoration projects?",
             "tools", ("distinct tools",), {"seaside", "restoration"}),
            ("What is the cardinality of the tool set used for winter orchard projects?",
             "tool set", ("tool set",), {"winter", "orchard"}),
        )
        for question, operand, entities, topic in cases:
            with self.subTest(question=question):
                plan = count_plan(question, operand, entities=entities)
                contract = count_scope_contract(plan, question)
                self.assertEqual(set(contract["topic_tokens"]), topic)
                self.assertEqual(contract["topic_source"], "question")
                self.assertEqual(contract["counted_role"], "instrument")
                queries = plan.retrieval_queries(question)
                self.assertTrue(all(word in queries[0] for word in topic))

    def test_concrete_named_constraint_remains_required(self):
        question = "How many tools did I use for Orion repairs?"
        plan = count_plan(question, "tools for Orion repairs", entities=("Orion",))
        self.assertEqual(count_scope_contract(plan, question)["topic_tokens"], ["orion"])

    def test_typed_frames_count_role_values_without_literal_type_word(self):
        question = "What is the cardinality of the tool set used for winter orchard projects?"
        plan = count_plan(question, "tool set", entities=("tool set",))
        ledger = build_evidence_ledger([
            row("I used the notebook in a completed winter orchard project.", "t1", "s1"),
            row("I used the ticket in a completed winter orchard project.", "t2", "s2"),
            row("I used the notebook in another winter orchard project.", "t3", "s3"),
            row("I used the tripod in a completed seaside restoration project.", "t4", "s4"),
        ])
        context = FakeFrameByEvidenceContext([
            frame("E1", predicate="used", instrument="notebook", theme="winter orchard project"),
            frame("E2", predicate="used", instrument="ticket", theme="winter orchard project"),
            frame("E3", predicate="used", instrument="notebook", theme="winter orchard project"),
            frame("E4", predicate="used", instrument="tripod", theme="seaside restoration project"),
        ])
        proof, diagnostics = execute_frame_operation_proof(
            context, question, plan, ledger, effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "2")
        self.assertEqual(set(proof.calculation["distinct_values"]), {"notebook", "ticket"})
        self.assertEqual(diagnostics["rejected_counts"]["scope"], 1)
        self.assertEqual(diagnostics["execution_contract"]["count_scope"]["topic_source"], "question")

    def test_typed_requested_status_applies_without_question_keyword_pattern(self):
        question = "What is the count of confirmed acquisitions in my Orion history?"
        plan = count_plan(question, "Orion history", entities=("Orion",), group="event")
        proof, diagnostics = execute_event_count_proof(
            plan, question, build_evidence_ledger([
                row("I brought home the coral notebook for my Orion setup.", "t1", "s1"),
                row("I brought home the silver ticket for my Orion setup.", "t2", "s2"),
            ]), effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "2")
        self.assertEqual(diagnostics["requested_status"], "completed")

    def test_event_executor_cannot_substitute_project_count_for_instruments(self):
        question = "How many tools did I use for Orion projects?"
        plan = count_plan(question, "Orion tools", entities=("Orion",))
        proof, diagnostics = execute_event_count_proof(
            plan, question,
            build_evidence_ledger([row("I completed the Orion project.", "t1", "s1")]),
            effective_operation="count_distinct",
        )
        self.assertIsNone(proof)
        self.assertFalse(diagnostics["applicable"])
        self.assertEqual(diagnostics["reason"], "specific_role_requires_typed_frames")

    def test_distinct_claims_in_one_turn_survive_canonicalization(self):
        question = "How many tools did I use for Orion projects?"
        plan = count_plan(question, "Orion tools", entities=("Orion",))
        ledger = build_evidence_ledger([row(
            "I used the notebook in a completed Orion project. "
            "I used the ticket in a completed Orion project.",
            "same-turn", "same-session",
        )])
        context = FakeFrameByEvidenceContext([
            frame("E1", predicate="used", instrument="notebook", theme="Orion project"),
            frame("E2", predicate="used", instrument="ticket", theme="Orion project"),
        ])
        proof, diagnostics = execute_frame_operation_proof(
            context, question, plan, ledger, effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "2")
        self.assertEqual(diagnostics["canonical_claim_count"], 2)

    def test_overlapping_copies_of_one_claim_are_still_deduplicated(self):
        question = "How many tools did I use for Orion projects?"
        plan = count_plan(question, "Orion tools", entities=("Orion",))
        ledger = build_evidence_ledger([
            row("I used the notebook in a completed Orion project.", "t1", "s1"),
            row("I used the notebook in a completed Orion project with detailed notes.", "t1", "s1"),
        ])
        context = FakeFrameByEvidenceContext([
            frame("E1", predicate="used", instrument="notebook", theme="Orion project"),
            frame("E2", predicate="used", instrument="notebook", theme="Orion project"),
        ])
        proof, diagnostics = execute_frame_operation_proof(
            context, question, plan, ledger, effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "1")
        self.assertEqual(diagnostics["canonical_claim_count"], 1)

    def test_low_rank_retrieved_sessions_reach_count_compilation(self):
        question = "How many completed acquisitions belong to my Orion setup?"
        plan = count_plan(question, "Orion acquisitions", entities=("Orion",), group="object")
        facts = [row(
            f"I brought home the {item} for my Orion setup.", f"t{index}", f"s{index}",
        ) for index, item in enumerate(("coral notebook", "blue ticket", "silver tripod", "green course"))]
        noise = [row("The recorded number was discussed in another topic.", f"n{i}", f"noise{i}") for i in range(15)]
        expanded, audit = expand_count_compiler_sources(plan, question, facts[:1], noise + facts)
        proof, _ = execute_event_count_proof(
            plan, question, build_evidence_ledger(expanded, event_status_resolver=derive_event_status_hint),
            effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "4")
        self.assertEqual(audit["restored_session_ids"], ["s1", "s2", "s3"])
        self.assertTrue(audit["retrieved_scope_sources_preserved"])
        self.assertFalse(audit["retrieval_exhaustive"])
        self.assertEqual(audit["rejected_counts"]["topic"], 15)

    def test_count_expansion_preserves_status_and_actor_filtering(self):
        question = "How many completed acquisitions belong to my Orion setup?"
        plan = count_plan(question, "Orion acquisitions", entities=("Orion",), group="object")
        candidates = [
            row("I brought home the coral notebook for my Orion setup.", "t1", "s1"),
            row("I plan to purchase a tripod for my Orion setup.", "t2", "s2"),
            row("Mira purchased a ticket for the Orion setup.", "t3", "s3"),
            row("I brought home a course for another setup.", "t4", "s4"),
        ]
        expanded, _ = expand_count_compiler_sources(plan, question, [], candidates)
        proof, _ = execute_event_count_proof(
            plan, question, build_evidence_ledger(expanded, event_status_resolver=derive_event_status_hint),
            effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "1")

    def test_expansion_ignores_answer_and_category_metadata(self):
        question = "How many completed acquisitions belong to my Orion setup?"
        plan = count_plan(question, "Orion acquisitions", entities=("Orion",), group="object")
        source = row("I brought home the coral notebook for my Orion setup.", "t1", "s1")
        ordinary, _ = expand_count_compiler_sources(plan, question, [], [source])
        poisoned, _ = expand_count_compiler_sources(plan, question, [], [{
            **source, "answer": "9999", "answer_session_ids": ["not-a-source"],
            "question_type": "unrelated", "category": 999,
        }])
        self.assertEqual(build_evidence_ledger(ordinary), build_evidence_ledger(poisoned))

    def test_missing_predicate_does_not_turn_a_recipient_into_the_actor(self):
        question = "How many tools did I use for Orion projects?"
        plan = count_plan(question, "Orion tools", entities=("Orion",))
        for text in (
            "Mira completed an Orion purchase of the ticket that should not be counted as mine.",
            "Omar gave the Orion ticket to me.",
            "My colleague used the Orion ticket.",
        ):
            with self.subTest(text=text):
                ledger = build_evidence_ledger([
                    row("I used the notebook in an Orion project.", "t1", "s1"),
                    row(text, "t2", "s2"),
                ])
                context = FakeFrameByEvidenceContext([
                    frame("E1", predicate="used", instrument="notebook"),
                    frame("E2", instrument="ticket"),
                ])
                proof, diagnostics = execute_frame_operation_proof(
                    context, question, plan, ledger, effective_operation="count_distinct",
                )
                self.assertEqual(proof.answer, "1")
                self.assertEqual(diagnostics["rejected_counts"]["actor_unresolved"], 1)

    def test_first_person_claim_survives_a_missing_extracted_predicate(self):
        question = "How many tools did I use for Orion projects?"
        plan = count_plan(question, "Orion tools", entities=("Orion",))
        proof, _ = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([frame("E1", instrument="notebook")]),
            question, plan,
            build_evidence_ledger([row("I used the notebook for Orion.", "t1", "s1")]),
            effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "1")

    def test_count_preserves_named_peer_source_evidence(self):
        question = "How many tools did Asha use for Orion projects?"
        plan = dataclasses.replace(
            count_plan(question, "Orion tools", entities=("Orion",)), actor="Asha",
        )
        proof, _ = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([frame("E1", predicate="used", instrument="notebook")]),
            question, plan,
            build_evidence_ledger([
                row("Asha used the notebook for Orion.", "t1", "s1", speaker="Asha"),
            ]), effective_operation="count_distinct",
        )
        self.assertEqual(proof.answer, "1")

    def test_count_revalidates_actor_on_previously_compiled_frames(self):
        question = "How many tools did I use for Orion projects?"
        plan = count_plan(question, "Orion tools", entities=("Orion",))
        quote = "Mira completed an Orion purchase that should not be counted as mine."
        saved = EvidenceFrame(
            frame_id="F1", source_turn_id="t1", source_session_id="s1",
            source_role="user", source_speaker="user", source_timestamp="2027-04-30",
            source_quote=quote, clause=quote, provenance_identity=("s1", "t1"),
            **frame("E1", instrument="purchase"),
        )
        with patch("PRAGMOS_evidence_compiler.compile_evidence_frames", return_value=(
            [saved], {"applicable": True, "status": "COMPLETE"},
        )):
            proof, diagnostics = execute_frame_operation_proof(
                None, question, plan, [], effective_operation="count_distinct",
            )
        self.assertFalse(proof.executable)
        self.assertEqual(diagnostics["rejected_counts"]["actor_unresolved"], 1)
        self.assertEqual(diagnostics["count_source_validation"][0]["actor_after"], "unknown")

    def test_an_unresolved_truncated_claim_cannot_be_a_completed_count_fact(self):
        question = "How many tools did I use for Orion projects?"
        plan = count_plan(question, "Orion tools", entities=("Orion",))
        proof, diagnostics = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([frame("E1", instrument="notebook")]),
            question, plan,
            build_evidence_ledger([row("I used the notebook for Orion...", "t1", "s1")]),
            effective_operation="count_distinct",
        )
        self.assertFalse(proof.executable)
        self.assertEqual(diagnostics["rejected_counts"]["source_clause_truncated"], 1)

    def test_full_source_recovery_restores_the_negative_qualifier(self):
        question = "How many completed acquisitions belong to my Orion setup?"
        plan = count_plan(question, "Orion acquisitions", entities=("Orion",), group="object")
        text = "I estimated $80 for my Orion notebook, but that amount was never spent."
        turn = SimpleNamespace(session_id="s1", turn_id=0, role="user", speaker="user",
                               timestamp="2027-04-30", text=text)
        preview = row("I estimated $80 for my Orion notebook, but that amount...", "0", "s1")
        restored, audit = restore_count_source_turns([preview], {0: turn})
        self.assertEqual(restored[0]["source_quote"], text)
        self.assertEqual(preview["source_quote"], "I estimated $80 for my Orion notebook, but that amount...")
        self.assertEqual(audit["restored_source_turn_ids"], ["0"])
        proof, diagnostics = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([frame("E1", object="notebook")]),
            question, plan, build_evidence_ledger(restored), effective_operation="count_distinct",
        )
        self.assertFalse(proof.executable)
        self.assertEqual(diagnostics["rejected_counts"]["status"], 1)

    def test_full_source_recovery_keeps_distinct_claims_in_one_turn(self):
        quote = "I used the notebook for Orion. I used the ticket for Orion."
        turn = SimpleNamespace(session_id="s1", turn_id=1, role="user", speaker="user",
                               timestamp="2027-04-30", text=quote)
        rows, diagnostics = restore_count_source_turns([
            row("I used the notebook for Orion...", "1", "s1"),
            row(quote, 1, "s1"),
        ], {1: turn})
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(build_evidence_ledger(rows)), 2)
        self.assertEqual(diagnostics["restored_row_count"], 1)

    def test_source_recovery_expands_only_the_referenced_chunk_turns(self):
        turns = {index: SimpleNamespace(
            session_id="s1", turn_id=index, role="user", speaker="user",
            timestamp="2027-04-30", text=f"Claim {index}.",
        ) for index in (1, 2, 3)}
        source = {"source_turn_ids": [1, 2], "source_session_id": "s1", "source_quote": "preview"}
        rows, diagnostics = restore_count_source_turns([source], turns)
        self.assertEqual([r["source_turn_id"] for r in rows], [1, 2])
        self.assertEqual(diagnostics["restored_source_turn_ids"], ["1", "2"])

    def test_source_recovery_does_not_cross_session_boundaries(self):
        source = row("Unchanged evidence.", "1", "s1")
        turn = SimpleNamespace(session_id="other", turn_id=1, role="user", speaker="user",
                               timestamp="2027-04-30", text="Unrelated source.")
        rows, diagnostics = restore_count_source_turns([source], {1: turn})
        self.assertEqual(rows, [source])
        self.assertEqual(diagnostics["unresolved_row_count"], 1)

    def test_count_actor_validation_does_not_change_collection_routing(self):
        question = "Which Orion tools were listed?"
        count = count_plan(question, "Orion tools", entities=("Orion",))
        plan = dataclasses.replace(count, operator="COLLECT", result_shape="value_set")
        proof, _ = execute_frame_operation_proof(
            FakeFrameByEvidenceContext([frame("E1", instrument="notebook")]),
            question, plan,
            build_evidence_ledger([row("The Orion notebook is mine.", "t1", "s1")]),
            effective_operation="collect_distinct",
        )
        self.assertEqual(proof.answer, "notebook")

    def test_source_expansion_is_inactive_for_lookup_and_arithmetic(self):
        question = "What is the Orion expense?"
        for operator in ("LOOKUP", "SUM"):
            with self.subTest(operator=operator):
                plan = parse_semantic_plan({
                    "operator": operator, "answer_type": "amount", "answer_role": "amount",
                    "actor": "user", "confidence": "high", "target_unit": "money",
                    "operands": [{"id": "O1", "text": "Orion expense", "kind": "quantity"}],
                }, question)
                selected = [row("I paid $20 for Orion.", "t1", "s1")]
                expanded, diagnostics = expand_count_compiler_sources(
                    plan, question, selected, [row("I paid $90 for Orion.", "t2", "s2")],
                )
                self.assertEqual(expanded, selected)
                self.assertFalse(diagnostics["applicable"])


if __name__ == "__main__":
    unittest.main()
