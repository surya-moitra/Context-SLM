"""Binding, extraction and acquisition counterexamples, without benchmark IDs."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_typed_binding import (
    bind_input_datatype, bind_operand_properties, bind_scalar_operands,
    compatible_references, contains_phrase, group_reference, independent_operands,
    property_group_key, refine_request, resolve_caption_addresses, select_relation_claims,
)
from PRAGMOS_typed_claims import bind_typed_claim, source_windows
from PRAGMOS_typed_computation import execute_typed_plan, run_typed_computation, working_windows
from test_PRAGMOS_typed_computation import bind, request, selection


class PropertyBindingTests(unittest.TestCase):
    def prices(self, missing=False):
        texts = ["The first listed price was $150.", "The first payment was $120.",
                 "The second listed price was $200.",
                 "The second payment is unrecorded." if missing else "The second payment was $170."]
        specs = []
        for group, prop, side in (("first", "listed price", "left"), ("first", "payment", "left"),
                                  ("second", "listed price", "right"), ("second", "payment", "right")):
            spec = selection(actor=None, binding="event_property", event=group, entity=group, key=group,
                             attribute=prop, group_side=side, numbers=[0], quantity_kind="measure")
            if missing and len(specs) == 3:
                spec.update(numbers=[], quantity_kind="missing", missing_span="payment is unrecorded")
            specs.append(spec)
        return bind(texts, specs)

    def query(self):
        return replace(request("relative_difference", grouped=True, comparison="greater"),
                       target_actor="user", property_binding=True,
                       left_operand="baseline quoted value", right_operand="actual paid value",
                       group_refs=(("left", "first"), ("right", "second")))

    def test_source_properties_are_bound_once_across_groups(self):
        claims, windows, sources = self.prices()
        choose = Mock(return_value={"left": "listed price", "right": "payment"})
        bound, audit = bind_operand_properties(choose, "Compare relative reductions.", self.query(), claims, windows, sources)
        self.assertEqual([c.operand for c in bound], ["left", "right", "left", "right"])
        self.assertEqual(choose.call_count, 1)
        self.assertEqual(execute_typed_plan(self.query(), bound, windows, sources)["answer"], "Yes")
        self.assertEqual(len(audit["claim_id_map"]), 4)

    def test_absolute_and_relative_comparisons_do_not_conflate(self):
        claims, windows, sources = self.prices()
        query = self.query()
        bound, _ = bind_operand_properties(Mock(return_value={"left": "listed price", "right": "payment"}),
                                           "Compare.", query, claims, windows, sources)
        self.assertEqual(execute_typed_plan(replace(query, operation="subtract"), bound, windows, sources)["answer"], "No")
        self.assertEqual(execute_typed_plan(replace(query, comparison="less"), bound, windows, sources)["answer"], "No")

    def test_missing_property_blocks_partial_calculation(self):
        claims, windows, sources = self.prices(missing=True)
        bound, _ = bind_operand_properties(Mock(return_value={"left": "listed price", "right": "payment"}),
                                           "Compare.", self.query(), claims, windows, sources)
        proof = execute_typed_plan(self.query(), bound, windows, sources)
        self.assertFalse(proof["executable"])
        self.assertEqual(proof["coverage"]["status"], "EXPLICIT_MISSING_INPUT")

    def test_same_property_cannot_supply_both_roles(self):
        claims, windows, sources = self.prices()
        with self.assertRaisesRegex(ContractError, "not_independent"):
            bind_operand_properties(Mock(return_value={"left": "payment", "right": "payment"}),
                                    "Compare.", self.query(), claims, windows, sources)

    def test_invented_property_is_rejected(self):
        claims, windows, sources = self.prices()
        with self.assertRaisesRegex(ContractError, "not_source_grounded"):
            bind_operand_properties(Mock(return_value={"left": "listed price", "right": "tax"}),
                                    "Compare.", self.query(), claims, windows, sources)

    def test_unassigned_required_property_is_not_dropped(self):
        claims, windows, sources = self.prices()
        with self.assertRaisesRegex(ContractError, "alias_role_unresolved"):
            # Source properties cannot be silently ignored just because another
            # property happens to have the requested physical unit.
            bind_operand_properties(Mock(return_value={"left": "listed price", "right": "payment"}),
                                    "Compare.", self.query(), [*claims, replace(claims[0], attribute="another property")], windows, sources)

    def test_different_source_labels_can_share_an_operand_role(self):
        claims, windows, sources = bind(["The first listed price was $150.", "The first payment was $120.",
                                        "The second sticker price was $200.", "The second amount paid was $170."], [
            selection(actor=None, binding="event_property", event=group, entity=group, key=group, attribute=prop,
                      group_side=side, numbers=[0], quantity_kind="measure")
            for group, prop, side in (("first", "listed price", "left"), ("first", "payment", "left"),
                                     ("second", "sticker price", "right"), ("second", "amount paid", "right"))])
        choose = Mock(return_value={"P1": "left", "P2": "right", "P3": "left", "P4": "right"})
        bound, _ = bind_operand_properties(choose, "Compare relative reductions.", self.query(), claims, windows, sources)
        self.assertEqual(execute_typed_plan(self.query(), bound, windows, sources)["answer"], "Yes")
        self.assertEqual(choose.call_count, 1)

    def test_related_owners_cannot_complete_each_others_inputs(self):
        claims, windows, sources = bind(["My brother's Cedar balance was $70.", "My sister's Cedar payment was $20."], [
            selection(actor="My brother", binding="requested_relation", relation_anchor="My brother", entity="Cedar", key="Cedar",
                      attribute="balance", numbers=[0], quantity_kind="measure"),
            selection(actor="My sister", binding="requested_relation", relation_anchor="My sister", entity="Cedar", key="Cedar",
                      attribute="payment", numbers=[0], quantity_kind="measure")])
        query = replace(request("subtract", grouped=True), property_binding=True, actor_scope="related", target_actor="user")
        bound, _ = bind_operand_properties(Mock(return_value={"left": "balance", "right": "payment"}),
                                           "Compare.", query, claims, windows, sources)
        proof = execute_typed_plan(query, bound, windows, sources)
        self.assertFalse(proof["executable"])
        self.assertEqual(len(proof["coverage"]["missing_group_operands"]), 2)


class CaptionAddressTests(unittest.TestCase):
    def captions(self):
        labels = [("early quoted price", "early quoted price", "early quoted price"),
                  ("early payment", "early payment", "early payment"),
                  ("late quoted price", "late", "quoted price"),
                  ("late payment", "late payment", "late payment")]
        return bind([f"My {caption} was ${value}." for (caption, _, _), value in zip(labels, (150, 120, 200, 170))],
                    [selection(actor="My", entity=caption, key=key, attribute=prop,
                               numbers=[0], quantity_kind="measure") for caption, key, prop in labels])

    def test_one_grounded_address_propagates_without_group_name_rules(self):
        claims, windows, sources = self.captions()
        bound, mapping, repairs = resolve_caption_addresses(request(), claims, windows, sources)
        self.assertEqual([(c.key, c.attribute) for c in bound],
                         [("early", "quoted price"), ("early", "payment"),
                          ("late", "quoted price"), ("late", "payment")])
        self.assertEqual(len(mapping), 4)
        self.assertEqual(len(repairs), 3)
        self.assertEqual([c.quantity for c in claims], [c.quantity for c in bound])
        self.assertEqual([c.window_id for c in claims], [c.window_id for c in bound])
        self.assertEqual([c.source_id for c in claims], [c.source_id for c in bound])
        reordered, _, _ = resolve_caption_addresses(request(), list(reversed(claims)), windows, sources)
        self.assertEqual([(c.key, c.attribute) for c in reversed(reordered)], [(c.key, c.attribute) for c in bound])

    def test_resolved_addresses_are_an_exact_noop(self):
        claims, windows, sources = PropertyBindingTests().prices()
        bound, mapping, repairs = resolve_caption_addresses(request(), claims, windows, sources)
        self.assertIs(bound, claims)
        self.assertEqual((mapping, repairs), ({}, []))

    def test_unanchored_caption_is_not_split_by_guessing_common_prefixes(self):
        claims, windows, sources = self.captions()
        with self.assertRaisesRegex(ContractError, "no_grounded_anchor"):
            resolve_caption_addresses(request(), claims[:2], windows, sources)

    def test_wrong_owner_or_disallowed_status_cannot_donate_an_anchor(self):
        claims, windows, sources = self.captions()
        for anchor in (replace(claims[2], actor_binding="other"),
                       replace(claims[2], status="cancelled"),
                       replace(claims[2], actor="Mira")):
            with self.subTest(anchor=anchor):
                with self.assertRaisesRegex(ContractError, "no_grounded_anchor"):
                    resolve_caption_addresses(request(), [claims[0], anchor], windows, sources)

    def test_competing_source_decompositions_are_rejected(self):
        claims, windows, sources = self.captions()
        competing = replace(claims[2], key="early quoted", attribute="price")
        with self.assertRaisesRegex(ContractError, "competing_decompositions"):
            resolve_caption_addresses(request(), [claims[0], claims[2], competing], windows, sources)

    def test_subword_prefix_cannot_become_an_occurrence_key(self):
        claims, windows, sources = bind(["My onerous price was $12.", "My one quote was $20."], [
            selection(actor="My", entity="onerous price", key="onerous price", attribute="onerous price", numbers=[0], quantity_kind="measure"),
            selection(actor="My", entity="one quote", key="one", attribute="quote", numbers=[1], quantity_kind="measure")])
        with self.assertRaisesRegex(ContractError, "no_grounded_anchor"):
            resolve_caption_addresses(request(), claims, windows, sources)


class TemporalAddressTests(unittest.TestCase):
    def observations(self, extra=False):
        values = [("Cedar", "02 May 2022", 100), ("Cedar", "16 May 2022", 130),
                  ("Cedar", "30 May 2022", 180), ("Birch", "02 May 2022", 400),
                  ("Birch", "30 May 2022", 440)]
        if extra:
            values.append(("Cedar", "30 May 2022", 190))
        return bind([f"On {date} my {name} channel had {value} followers." for name, date, value in values],
                    [selection(actor="my", entity=name, key=name, attribute="followers", numbers=[0], quantity_kind="count", time=date)
                     for name, date, _ in values])

    def query(self):
        return replace(request("subtract", grouped=True, rank="max", projection="entity", question="During May 2022", time_scope="May 2022"),
                       target_actor="user", property_binding=True, operand_order=(("left", "latest"), ("right", "earliest")))

    def test_endpoint_selection_is_by_event_date_not_retrieval_order(self):
        claims, windows, sources = self.observations()
        query = self.query()
        bound, _ = bind_operand_properties(Mock(side_effect=AssertionError("No semantic endpoint relabeling")),
                                           "Which change?", query, list(reversed(claims)), windows, sources)
        proof = execute_typed_plan(query, bound, windows, sources)
        self.assertEqual(proof["answer"], "Cedar")
        self.assertEqual(sum(c.operand == "context" for c in bound), 1)

    def test_missing_group_endpoint_remains_a_retrieval_obligation(self):
        claims, windows, sources = self.observations()
        bound, _ = bind_operand_properties(Mock(), "Which change?", self.query(), claims[:-1], windows, sources)
        proof = execute_typed_plan(self.query(), bound, windows, sources)
        self.assertFalse(proof["executable"])
        self.assertEqual(proof["coverage"]["status"], "MISSING_OPERAND")
        self.assertEqual(proof["coverage"]["missing_group_operands"][0]["key"], "birch")

    def test_conflicting_endpoint_is_not_resolved_by_first_row(self):
        claims, windows, sources = self.observations(extra=True)
        with self.assertRaisesRegex(ContractError, "endpoint_not_unique"):
            bind_operand_properties(Mock(), "Which change?", self.query(), claims, windows, sources)


class IdentityAndScopeTests(unittest.TestCase):
    def test_speaker_scope_excludes_relatives_but_related_scope_keeps_them(self):
        claims, windows, sources = bind(["I paid $11.", "My cousin paid $24."], [
            selection(entity="paid", numbers=[0], quantity_kind="measure"),
            selection(actor="My cousin", binding="requested_relation", relation_anchor="My cousin", entity="paid", numbers=[0], quantity_kind="measure")])
        query = replace(request(), target_actor="user")
        self.assertEqual(execute_typed_plan(query, claims, windows, sources)["answer"], "$11")
        self.assertEqual(execute_typed_plan(replace(query, actor_scope="related"), claims, windows, sources)["answer"], "$24")

    def test_possessive_anchor_cannot_belong_to_a_different_actor(self):
        with self.assertRaisesRegex(ContractError, "possessive_source_link"):
            bind(["My cousin spoke to Mira."], [selection(actor="Mira", binding="requested_relation", relation_anchor="My cousin", entity="Mira")])

    def test_distinct_occurrence_properties_cannot_link_by_generic_substring(self):
        with self.assertRaisesRegex(ContractError, "no_source_link"):
            bind(["The first original price was $20.", "The second original price was $40."], [
                selection(actor=None, binding="event_property", event="first original price", entity="original price", key="first original price", numbers=[0], quantity_kind="measure"),
                selection(actor=None, binding="linked_event", entity="second original price", key="second original price", numbers=[0], quantity_kind="measure", link_index=0)])

    def test_repeat_cannot_change_a_measurement(self):
        with self.assertRaisesRegex(ContractError, "changes_the_original"):
            bind(["I paid $20 at Cedar.", "I paid $40 at Cedar."], [
                selection(entity="Cedar", numbers=[0], quantity_kind="measure"),
                selection(entity="Cedar", kind="reference", numbers=[0], quantity_kind="measure", link_index=0)])

    def test_separate_transactions_at_same_entity_still_both_count(self):
        claims, windows, sources = bind(["I paid $20 at Cedar.", "I paid $40 at Cedar."], [
            selection(entity="Cedar", numbers=[0], quantity_kind="measure"),
            selection(entity="Cedar", numbers=[0], quantity_kind="measure")])
        self.assertEqual(execute_typed_plan(request(), claims, windows, sources)["answer"], "$60")

    def test_subject_binding_does_not_admit_a_one_word_topic_collision(self):
        sources = {"A": Source("A", "s", "1", "user", "user", None, "I presented the orchard service poster."),
                   "B": Source("B", "s", "2", "user", "user", None, "Write a poem about orchards.")}
        selected = working_windows("At which university?", {"bound_subject": "orchard service poster", "scope_tokens": ["university"]}, source_windows(sources), {})
        self.assertEqual([w.source_id for w in selected], ["A"])

    def test_definite_reference_admits_an_operand_but_does_not_merge_occurrences(self):
        claims, windows, _ = bind(["My earlier studio trial took 18 seconds.", "The later trial took 13 seconds."], [
            selection(actor="My", entity="earlier studio trial", numbers=[0], quantity_kind="measure"),
            selection(actor=None, binding="event_property", event="later trial", entity="later trial", numbers=[0], quantity_kind="measure")])
        chosen = working_windows("What reduction?", {"scope_tokens": ["studio"], "ordered_occurrences": True},
                                 list(windows.values()), {claims[0].claim_id: claims[0]})
        self.assertEqual([w.source_id for w in chosen], ["S1", "S2"])
        self.assertIsNone(claims[1].link)
        self.assertEqual(compatible_references({claims[0].claim_id: claims[0]}, list(windows.values())[1], "later trial", None), {})

    def test_definite_type_head_does_not_admit_an_unrelated_named_place(self):
        claims, windows, sources = bind(["I visited Cedar University.", "The Pine University hosted a separate reception."], [
            selection(entity="Cedar University", key="Cedar University"),
            selection(actor=None, binding="event_property", event="reception", entity="Pine University", key="Pine University")])
        chosen = working_windows("Where was my poster shown?", {"scope_tokens": ["poster"]},
                                 list(windows.values()), {claims[0].claim_id: claims[0]})
        self.assertNotIn("S2", [w.source_id for w in chosen])

    def test_relation_head_requires_a_demonstrative_not_a_relative_clause(self):
        claims, _, sources = bind(["I visited Cedar University."], [selection(entity="Cedar University", key="Cedar University")])
        sources["S2"] = Source("S2", "s", "2", "user", "user", None,
                               "The sentence that states another institute's affiliation mentions Pine University.")
        sources["S3"] = Source("S3", "s", "3", "user", "user", None, "That university hosted the session.")
        windows, known = source_windows(sources), {claims[0].claim_id: claims[0]}
        contract = {"scope_tokens": ["poster"]}
        original = working_windows("Where was my poster shown?", contract, windows, known)
        strict = working_windows("Where was my poster shown?", {**contract, "literal_relation_heads": True}, windows, known)
        self.assertEqual([w.source_id for w in original], ["S1", "S2", "S3"])
        self.assertEqual([w.source_id for w in strict], ["S1", "S3"])

    def test_group_references_require_one_whole_identity(self):
        groups = (("left", "first"), ("right", "second"))
        self.assertEqual(group_reference("The first original price was $20.", groups), ("left", "first"))
        self.assertIsNone(group_reference("The first and second orders.", groups))
        self.assertFalse(contains_phrase("firstly", "first"))

    def test_property_suffix_is_not_part_of_the_group_identity(self):
        source = Source("S", "s", "t", "user", "user", None, "The second original price was $120.")
        window = source_windows({"S": source})[0]
        self.assertEqual(property_group_key("second original price", "120", "original price", window, [0]), "second")

    def test_property_only_name_cannot_be_an_invented_group(self):
        source = Source("S", "s", "t", "user", "user", None, "The original price was $120.")
        window = source_windows({"S": source})[0]
        with self.assertRaisesRegex(ContractError, "no_distinct_entity"):
            property_group_key("original price", "original price", "original price", window, [0])


class AcquisitionTests(unittest.TestCase):
    def run_missing(self, expanded=None):
        sources = {"A": Source("A", "s", "1", "user", "user", None, "I recorded 41 pages."),
                   "B": Source("B", "t", "2", "user", "user", None, "I read 7 pages.")}
        windows = source_windows(sources)
        query = replace(request("subtract"), target_actor="user", left_operand="recorded pages", right_operand="read pages")
        fetch = Mock(return_value={"old-turn-under-different-id": sources["B"]} if expanded is None else expanded)
        rows = [selection(entity="pages", operand="left", numbers=[0], quantity_kind="measure"),
                selection(entity="pages", operand="right", numbers=[0], quantity_kind="measure")]
        report = {"calls": {"planner_calls": 0, "extractor_calls": 0}}
        with patch("PRAGMOS_typed_computation.plan_small_request", return_value=query), patch(
                "PRAGMOS_typed_computation.working_windows", return_value=windows[:1]), patch(
                "PRAGMOS_typed_computation.select_small_claims", side_effect=[{"decision": "include", "facts": [row]} for row in rows]):
            result = run_typed_computation(SimpleNamespace(count_tokens=lambda _: 10), "What remains?", sources, report,
                n_ctx=2048, max_answer_tokens=64, contract={"extended_contracts": True}, retrieve_missing=fetch)
        return result, fetch

    def test_existing_unreviewed_turn_is_not_discarded_as_duplicate(self):
        result, fetch = self.run_missing()
        self.assertEqual(result["answer"], "34 pages")
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(result["targeted_retrieval"]["source_ids"], ["B"])

    def test_missing_operand_does_not_become_zero_or_supported_abstention(self):
        result, _ = self.run_missing(expanded={})
        self.assertEqual(result["status"], "NOT_PROVEN")
        self.assertIsNone(result["answer"])
        self.assertFalse(result["proof"]["abstention_supported"])

    def test_returned_turn_cannot_mutate_existing_provenance(self):
        changed = Source("different-key", "t", "2", "user", "user", None, "I read 17 pages.")
        with self.assertRaisesRegex(ContractError, "changed_an_existing_raw_turn"):
            self.run_missing(expanded={"X": changed})


class RelationExtractionTests(unittest.TestCase):
    def test_linked_objects_do_not_promote_planned_events_to_completed(self):
        query = replace(request("lookup", projection="entity", key_role="location", statuses=["completed"]),
                        source_addressed=True, key_description="university", subject="studio poster")
        for sentence, status in (("I plan to display the studio poster at Cedar University.", "planned"),):
            with self.subTest(sentence=sentence):
                source = Source("S", "s", "1", "user", "user", None, sentence)
                window = source_windows({"S": source})[0]
                choose = Mock(side_effect=[{"subject": "I", "object": "studio poster"},
                                           {"subject": "studio poster", "object": "Cedar University"}])
                payload = select_relation_claims(choose, "At which university did I display the studio poster?",
                                                 query, window, source, {})
                claims, known = [], {}
                for spec in payload["facts"]:
                    claim = bind_typed_claim(window, source, spec, known)
                    claims.append(claim)
                    known[claim.claim_id] = claim
                self.assertEqual({c.status for c in claims}, {status})
                self.assertFalse(execute_typed_plan(query, claims, {window.window_id: window}, {"S": source})["executable"])

    def test_negated_relationship_retains_existing_rejection(self):
        query = replace(request("lookup", projection="entity"), source_addressed=True)
        source = Source("S", "s", "1", "user", "user", None, "I did not display the studio poster at Cedar University.")
        window = source_windows({"S": source})[0]
        with self.assertRaisesRegex(ContractError, "negated_relation"):
            select_relation_claims(Mock(return_value={"subject": "I", "object": "studio poster"}),
                                   "Where did I display the studio poster?", query, window, source, {})

    def test_a_source_question_cannot_become_a_relationship_assertion(self):
        query = replace(request("lookup", projection="entity"), source_addressed=True)
        source = Source("S", "s", "1", "user", "user", None, "Did I display the studio poster at Cedar University?")
        window = source_windows({"S": source})[0]
        with self.assertRaisesRegex(ContractError, "not_a_relationship_assertion"):
            select_relation_claims(Mock(return_value={"subject": "I", "object": "Cedar University"}),
                                   "Where did I display the studio poster?", query, window, source, {})

    def test_literal_requested_head_attests_answer_without_a_semantic_guess(self):
        query = replace(request("lookup", projection="entity", key_role="location"),
                        source_addressed=True, key_description="university", subject="studio poster")
        sources = {"A": Source("A", "s", "1", "user", "user", None, "I displayed the studio poster at Lantern Forum."),
                   "B": Source("B", "s", "2", "user", "user", None, "Lantern Forum was held at Cedar University; the reception was at Pine College.")}
        windows = source_windows(sources)
        root = select_relation_claims(Mock(return_value={"subject": "I", "object": "Lantern Forum"}),
                                     "At which university?", query, windows[0], sources["A"], {})
        parent = bind_typed_claim(windows[0], sources["A"], root["facts"][0], {})
        choose = Mock(return_value={"subject": "Lantern Forum", "object": "Cedar University"})
        result = select_relation_claims(choose, "At which university?", query, windows[1], sources["B"], {parent.claim_id: parent})
        self.assertEqual(result["facts"][0]["operand"], "member")
        self.assertEqual(choose.call_count, 1)

    def test_main_object_does_not_hide_a_following_venue(self):
        query = replace(request("lookup", projection="entity", key_role="location"), source_addressed=True,
                        key_description="university")
        source = Source("A", "s", "1", "user", "user", None, "I displayed the studio poster at Lantern Forum.")
        window = source_windows({"A": source})[0]
        choose = Mock(side_effect=[{"subject": "I", "object": "studio poster"},
                                   {"subject": "studio poster", "object": "Lantern Forum"}])
        payload = select_relation_claims(choose, "At which university did I display the studio poster?", query, window, source, {})
        self.assertEqual([row["key"] for row in payload["facts"]], ["studio poster", "Lantern Forum"])
        self.assertTrue(all(row["kind"] == "context" for row in payload["facts"]))

    def test_new_context_bridge_alone_cannot_veto_a_correct_direct_answer(self):
        claims, windows, sources = bind(["I presented at Lantern Forum."], [
            selection(entity="Lantern Forum", key="Lantern Forum", kind="context", operand="context")])
        query = replace(request("lookup", projection="entity", key_role="location"), source_addressed=True)
        proof = execute_typed_plan(query, claims, windows, sources, baseline_answer="Lantern Forum")
        self.assertFalse(proof["abstention_supported"])

    def test_source_relative_relationship_uses_host_not_adjacent_venue(self):
        query = replace(request("lookup", projection="entity", key_role="location"), source_addressed=True, subject="studio poster")
        sources = {"A": Source("A", "s", "1", "user", "user", None, "I presented the studio poster at Lantern Symposium."),
                   "B": Source("B", "s", "2", "user", "user", None, "Lantern Symposium was held at Cedar University; the reception was at Pine College.")}
        windows = source_windows(sources)
        root = select_relation_claims(Mock(return_value={"subject": "I", "object": "Lantern Symposium"}),
                                     "At which university?", query, windows[0], sources["A"], {})
        parent = bind_typed_claim(windows[0], sources["A"], root["facts"][0], {})
        choose = Mock(side_effect=[{"subject": "Lantern Symposium", "object": "Cedar University"}, {"role": "answer"}])
        payload = select_relation_claims(choose, "At which university?", query, windows[1], sources["B"], {parent.claim_id: parent})
        child = bind_typed_claim(windows[1], sources["B"], payload["facts"][0], {parent.claim_id: parent})
        proof = execute_typed_plan(query, [parent, child], {w.window_id: w for w in windows}, sources)
        self.assertEqual(proof["answer"], "Cedar University")
        with self.assertRaisesRegex(ContractError, "one_source_assertion|argument_order"):
            select_relation_claims(Mock(return_value={"subject": "Lantern Symposium", "object": "Pine College"}),
                                   "At which university?", query, windows[1], sources["B"], {parent.claim_id: parent})

    def test_unbound_relationship_cannot_guess_an_owner(self):
        query = replace(request("lookup", projection="entity"), source_addressed=True, subject="studio poster")
        source = Source("A", "s", "1", "user", "user", None, "The reception was at Pine College.")
        with self.assertRaisesRegex(ContractError, "no_bound_subject"):
            select_relation_claims(Mock(), "Where?", query, source_windows({"A": source})[0], source, {})


class QueryRefinementTests(unittest.TestCase):
    def test_changed_binding_module_invalidates_typed_resume_only(self):
        import sys
        import tempfile
        from pathlib import Path
        import PRAGMOS_benchmark_LongMemEval as lme
        import PRAGMOS_benchmark_LoCoMo as locomo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, model = root / "data.json", root / "model.gguf"
            data.write_text("[]\n")
            model.write_bytes(b"test fixture")
            for adapter in (lme, locomo):
                for mode in ("off", "shadow"):
                    with self.subTest(adapter=adapter.__name__, mode=mode), patch.object(sys, "argv", [
                            "test", "--data-file", str(data), "--model-path", str(model), "--mode", "pragmos_context",
                            "--pragmos-composable-mode", mode, "--pragmos-composable-protocol", "typed_v4"]):
                        options = adapter.parse_args()
                    build = lambda: adapter.build_run_manifest(options, str(data), []) if adapter is lme else adapter.build_manifest(options, [])
                    before = build()
                    original = adapter.local_file_identity
                    def changed(path):
                        value = original(path)
                        if Path(path).name == "PRAGMOS_typed_binding.py":
                            value = {**value, "sha256": "0" * 64}
                        return value
                    with patch.object(adapter, "local_file_identity", side_effect=changed):
                        after = build()
                    self.assertEqual("source_identities" in lme.manifest_mismatch_fields(before, after), mode != "off")

    def test_calendar_inputs_refine_subtraction_without_date_invention(self):
        base = replace(request("subtract", question="How many months?", output_unit="months"),
                       left_operand="placement date", right_operand="receipt date")
        planned = refine_request(Mock(side_effect=AssertionError("Input types come from raw facts")),
                                 "How many months elapsed for the order?", base, [])
        claims, windows, sources = bind(["I placed the order on 4 May 2022.", "I received the order on 4 September 2022."], [
            selection(entity="order", event="placed", time="4 May 2022"),
            selection(entity="order", event="received", time="4 September 2022")])
        planned = bind_input_datatype(planned, claims)
        self.assertEqual(planned.operation, "calendar_difference")
        self.assertEqual(planned.output_unit, "months")
        self.assertEqual(planned.left_operand, "placement date")
        bound, _ = bind_scalar_operands(Mock(return_value={"expression": "A to B"}),
                                        "How many months?", planned, claims, windows, sources)
        self.assertEqual(execute_typed_plan(planned, bound, windows, sources)["answer"], "4 months")

    def test_dates_attached_to_counts_are_not_calendar_arithmetic(self):
        claims, _, _ = TemporalAddressTests().observations()
        query = replace(request("subtract"), source_addressed=True)
        self.assertEqual(bind_input_datatype(query, claims).operation, "subtract")

    def test_duplicate_operands_fail_before_extraction(self):
        with self.assertRaisesRegex(ContractError, "not_independent"):
            independent_operands("May balance", "May balance")

    def test_refined_slot_must_be_a_literal_of_the_question(self):
        with self.assertRaisesRegex(ContractError, "not_question_grounded"):
            refine_request(Mock(side_effect=[{"count_output": "number"}, {"owner_span": "invented owner"}]),
                           "How many objects?", request("count", projection="entity"), [])

    def test_valid_scalar_remainder_contract_is_not_reclassified(self):
        planned = replace(request("subtract"), left_operand="capacity", right_operand="consumption")
        choose = Mock(side_effect=AssertionError("No redundant output/owner classification"))
        refined = refine_request(choose, "What remains?", planned, [{"text": "I read 7 pages."}])
        self.assertEqual((refined.projection, refined.actor_scope), (planned.projection, planned.actor_scope))
        self.assertEqual((refined.left_operand, refined.right_operand), ("capacity", "consumption"))


class ScalarSourceBindingTests(unittest.TestCase):
    def test_duration_reduction_uses_source_ids_not_hallucinated_operand_phrases(self):
        claims, windows, sources = bind(["My earlier studio trial took 3 minutes and 10 seconds.",
                                        "The later trial took 2 minutes and 40 seconds."], [
            selection(actor="My", entity="earlier studio trial", numbers=[0, 1], quantity_kind="measure"),
            selection(actor=None, binding="event_property", event="later trial", entity="later trial", numbers=[0, 1], quantity_kind="measure")])
        query = replace(request("subtract", question="In seconds?", output_unit="seconds"), value_binding=True)
        bound, audit = bind_scalar_operands(Mock(side_effect=[{"chronology": "A before B"}, {"expression": "INITIAL - FINAL"}]),
                                            "By how many seconds was the later trial faster?", query, claims, windows, sources)
        self.assertEqual(execute_typed_plan(query, bound, windows, sources)["answer"], "30 seconds")
        self.assertEqual(audit["expression"], "A minus B")

    def test_signed_reduction_is_not_forced_positive(self):
        claims, windows, sources = bind(["My first trial took 12 seconds.", "My later trial took 17 seconds."], [
            selection(actor="My", entity="first trial", numbers=[0], quantity_kind="measure"),
            selection(actor="My", entity="later trial", numbers=[0], quantity_kind="measure")])
        query = replace(request("subtract"), value_binding=True)
        bound, _ = bind_scalar_operands(Mock(side_effect=[{"chronology": "A before B"}, {"expression": "INITIAL - FINAL"}]),
                                        "What reduction?", query, claims, windows, sources)
        self.assertEqual(execute_typed_plan(query, bound, windows, sources)["answer"], "-5 seconds")

    def test_extra_required_input_is_not_dropped(self):
        claims, windows, sources = bind(["I recorded 3 seconds.", "I recorded 5 seconds.", "I recorded 8 seconds."], [
            selection(entity="recorded", numbers=[0], quantity_kind="measure")] * 3)
        with self.assertRaisesRegex(ContractError, "extra_unresolved_members"):
            bind_scalar_operands(Mock(), "What difference?", replace(request("subtract"), value_binding=True), claims, windows, sources)


if __name__ == "__main__":
    unittest.main()
