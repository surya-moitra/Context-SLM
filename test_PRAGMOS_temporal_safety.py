import copy
import json
import unittest

import PRAGMOS_temporal_algebra as algebra
from PRAGMOS_benchmark_LongMemEval import resolve_temporal_expression
from PRAGMOS_temporal_contracts import promotable_temporal_result, run_temporal_candidate, supported_temporal_abstention
from test_PRAGMOS_temporal_algebra import operand, plan, source, step
from test_PRAGMOS_temporal_contracts import FakeLocalModel, annotation, constraint_review_payload, contract, legacy_plan


def inventory(*constraints):
    return dict(status="reviewed", constraints=list(constraints))


def constraint(kind, span, target=None, anchor=None, value=None):
    return dict(kind=kind, span=span, target=target, anchor=anchor, value=value)


class QuestionConstraintSafetyTests(unittest.TestCase):
    def setUp(self):
        self.question = "Where did I attend the workshop first in June 2028?"
        self.request = plan([operand("A", "workshop")],
                            [step("a", "scan", operand="A"),
                             step("w", "window", "a", span="June 2028"),
                             step("e", "earliest", "w"),
                             step("p", "project", "e", field="location")], kind="location")
        self.requirements = inventory(constraint("window", "June 2028", "workshop"),
                                      constraint("earliest", "first", "workshop"),
                                      constraint("projection", "Where", value="location"))

    def verify(self, request=None, requirements=None, question=None):
        return algebra.verify_question_constraints(
            request or self.request, requirements or self.requirements, question or self.question,
            temporal_resolver=resolve_temporal_expression, reference_timestamp="2028-06-20")

    def test_requested_window_and_selection_are_preserved(self):
        before = copy.deepcopy((self.request, self.requirements))
        self.assertEqual(self.verify()["status"], "verified")
        self.assertEqual((self.request, self.requirements), before)

    def test_scalar_first_entails_chronology_not_a_full_sequence_output(self):
        requirements = inventory(*self.requirements["constraints"], constraint("order", "first", "workshop"))
        self.assertEqual(self.verify(requirements=requirements)["status"], "verified")

    def test_list_order_cannot_be_replaced_with_one_earliest_item(self):
        request = copy.deepcopy(self.request)
        request["answer_kind"] = "list"
        requirements = inventory(*self.requirements["constraints"], constraint("order", "first", "workshop"))
        with self.assertRaisesRegex(ValueError, "constraint_not_represented:order"):
            self.verify(request, requirements)

    def test_earliest_without_requested_window_is_rejected(self):
        request = copy.deepcopy(self.request)
        request["steps"].pop(1)
        request["steps"][1]["args"] = ["a"]
        with self.assertRaisesRegex(ValueError, "constraint_not_represented:window"):
            self.verify(request)

    def test_filter_after_earliest_is_not_equivalent(self):
        request = copy.deepcopy(self.request)
        request["steps"][1:3] = [step("e", "earliest", "a"), step("w", "window", "e", span="June 2028")]
        request["steps"][-1]["args"] = ["w"]
        with self.assertRaisesRegex(ValueError, "reduced_before_filter"):
            self.verify(request)

    def test_unrequested_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unrequested_temporal_constraint"):
            self.verify(requirements=inventory(self.requirements["constraints"][0]))

    def test_unknown_review_cannot_authorize_candidate(self):
        with self.assertRaisesRegex(ValueError, "not_reviewed"):
            self.verify(requirements=dict(status="uncertain", constraints=[]))

    def test_invented_constraint_span_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ungrounded_question_constraint"):
            self.verify(requirements=inventory(constraint("window", "July 2028", "workshop")))

    def test_companion_cannot_be_replaced_by_performer(self):
        request = copy.deepcopy(self.request)
        request["answer_kind"] = "actor"
        request["steps"][-1]["field"] = "actor"
        requirements = copy.deepcopy(self.requirements)
        requirements["constraints"][-1]["value"] = "person"
        with self.assertRaisesRegex(ValueError, "constraint_not_represented:projection"):
            self.verify(request, requirements)

    def test_wrong_actor_status_or_phase_cannot_certify(self):
        for kind, value in [("actor", "Nila"), ("status", "planned"), ("phase", "end")]:
            q = self.question + " Nila planned to finish."
            requirements = inventory(*self.requirements["constraints"],
                                     constraint(kind, "Nila planned to finish", "workshop", value=value))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "constraint_not_represented:" + kind):
                self.verify(requirements=requirements, question=q)

    def test_scoped_window_does_not_constrain_the_anchor(self):
        q = "Which activities happened after induction in June 2028?"
        request = plan([operand("A", "activities", True), operand("B", "induction")],
                       [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                        step("w", "window", "a", span="June 2028"), step("f", "after", "w", "b"),
                        step("p", "project", "f", field="event")], kind="event")
        requirements = inventory(constraint("window", "June 2028", "activities"),
                                 constraint("after", "after", "activities", "induction"))
        self.assertEqual(self.verify(request, requirements, q)["status"], "verified")
        wrong = copy.deepcopy(request)
        wrong["steps"][2]["args"] = ["b"]
        with self.assertRaisesRegex(ValueError, "constraint_not_represented:window"):
            self.verify(wrong, requirements, q)

    def test_before_cannot_satisfy_an_after_constraint(self):
        q = "Which activities happened after induction?"
        request = plan([operand("A", "activities", True), operand("B", "induction")],
                       [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                        step("f", "before", "a", "b"), step("p", "project", "f", field="event")], kind="event")
        with self.assertRaisesRegex(ValueError, "constraint_not_represented:after"):
            self.verify(request, inventory(constraint("after", "after", "activities", "induction")), q)

    def test_reducing_only_the_comparison_anchor_remains_valid(self):
        q = "Which activities in June 2028 followed the first induction?"
        request = plan([operand("A", "activities", True), operand("B", "induction")],
                       [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                        step("e", "earliest", "b"), step("f", "after", "a", "e"),
                        step("w", "window", "f", span="June 2028"), step("p", "project", "w", field="event")], kind="event")
        requirements = inventory(constraint("window", "June 2028", "activities"),
                                 constraint("after", "followed", "activities", "induction"),
                                 constraint("earliest", "first", "induction"))
        self.assertEqual(self.verify(request, requirements, q)["status"], "verified")

    def test_question_clock_and_unit_are_mandatory(self):
        q = "How many weeks ago did I attend the workshop?"
        request = plan([operand("A", "workshop")],
                       [step("a", "scan", operand="A"), step("d", "difference", "a", unit="week")])
        requirements = inventory(constraint("unit", "weeks", value="week"), constraint("question_time", "ago"))
        self.assertEqual(self.verify(request, requirements, q)["status"], "verified")
        request["steps"][-1]["unit"] = "day"
        with self.assertRaisesRegex(ValueError, "constraint_not_represented:unit"):
            self.verify(request, requirements, q)
        with self.assertRaisesRegex(ValueError, "duration_unit_not_reviewed"):
            self.verify(request, inventory(constraint("question_time", "ago")), q)
        with self.assertRaisesRegex(ValueError, "time_reference_not_reviewed"):
            self.verify(request, inventory(constraint("unit", "weeks", value="week")), q)

    def test_duration_unit_does_not_turn_a_filter_anchor_into_an_endpoint(self):
        q = "How many days ago was the workshop after induction?"
        request = plan([operand("A", "workshop"), operand("B", "induction")],
                       [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                        step("f", "after", "a", "b"), step("d", "difference", "f", unit="day")])
        requirements = inventory(constraint("after", "after", "workshop", "induction"),
                                 constraint("unit", "days", value="day"), constraint("question_time", "ago"))
        self.assertEqual(self.verify(request, requirements, q)["status"], "verified")

    def test_distinct_count_cannot_degrade_to_occurrence_count(self):
        q = "How many distinct venues did I visit in June 2028?"
        request = plan([operand("A", "visit", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("p", "project", "w", field="location"), step("d", "distinct", "p"),
                        step("n", "count", "d")], kind="count")
        requirements = inventory(constraint("window", "June 2028", "visit"),
                                 constraint("distinct", "distinct", "visit"), constraint("count", "How many", "visit"))
        self.assertEqual(self.verify(request, requirements, q)["status"], "verified")
        request["steps"] = request["steps"][:2] + [step("n", "count", "w")]
        with self.assertRaisesRegex(ValueError, "constraint_not_represented:distinct"):
            self.verify(request, requirements, q)

    def test_unscoped_multi_operand_window_is_not_guessed(self):
        request = copy.deepcopy(self.request)
        request["operands"].append(operand("B", "induction"))
        with self.assertRaisesRegex(ValueError, "ambiguous_question_window_scope"):
            self.verify(request, inventory(constraint("window", "June 2028")))

    def test_requested_projection_cannot_authorize_unrequested_frequency(self):
        request = plan([operand("A", "workshop", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("g", "group_count", "w", field="location"),
                        step("m", "argmax", "g", field="location")], kind="location")
        requirements = inventory(constraint("window", "June 2028", "workshop"),
                                 constraint("projection", "Where", value="location"))
        with self.assertRaisesRegex(ValueError, "unrequested_temporal_constraint"):
            self.verify(request, requirements)

    def test_frequency_remains_valid_when_explicitly_requested(self):
        q = "Which location did I visit most often in June 2028?"
        request = plan([operand("A", "visit", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("g", "group_count", "w", field="location"),
                        step("m", "argmax", "g", field="location")], kind="location")
        requirements = inventory(constraint("window", "June 2028", "visit"),
                                 constraint("frequency_max", "most often", "visit"),
                                 constraint("projection", "location", value="location"))
        self.assertEqual(self.verify(request, requirements, q)["status"], "verified")


class RuntimeEventLinkSafetyTests(unittest.TestCase):
    def run_review(self, following, relation):
        primary = "I attended the workshop on 3 June 2028."
        text = primary + " " + following
        rows = algebra.prepare_source_rows([source(text)])
        request = plan([operand("A", "workshop")],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("p", "project", "w", field="location")], kind="location")
        calls = []
        def call(context, prompt, schema, key, diagnostics, **kwargs):
            calls.append(key)
            if key == "temporal_event_episode_links":
                return dict(same_occurrence={"elaboration": "yes", "separate_event": "no", "uncertain": "uncertain"}[relation])
            evidence = json.loads(prompt.split("\nEvidence: ")[1].split("\n<|end|>")[0])
            if key == "temporal_composed_assertions":
                return dict(assertions={r["id"]: dict(
                    actor_span="I", predicate_span="attended" if r["clause"] == primary else "visited" if "visited" in r["clause"] else "arrived",
                    argument_span=None, phase="occurrence", factuality="actual",
                    matches=["A"] if r["clause"] == primary else []) for r in evidence})
            if key == "temporal_event_attributes":
                venue = next((r for r in evidence if "Birch Hall" in r["text"]), None)
                return dict(location=dict(value="Birch Hall" if venue else None,
                                          evidence_id=venue["id"] if venue else None))
            raise AssertionError(key)
        diagnostics = dict(calls={}, retrieval={}, reviewed_source_ids=[])
        events = algebra._review_sources(None, rows, request, call, diagnostics, resolve_temporal_expression)
        return events, diagnostics, calls

    def test_another_activitys_venue_is_not_available_for_projection(self):
        events, _, calls = self.run_review("I visited the gallery on 3 June 2028 at Birch Hall.", "separate_event")
        self.assertNotIn("location", events[0]["attributes"])
        self.assertIn("temporal_event_episode_links", calls)

    def test_conflicting_event_dates_override_an_erroneous_link_review(self):
        events, diagnostics, _ = self.run_review("I visited the gallery on 9 June 2028 at Birch Hall.", "elaboration")
        self.assertNotIn("location", events[0]["attributes"])
        self.assertTrue(diagnostics["event_episode_links"][0]["date_boundary_ids"])

    def test_same_day_does_not_merge_distinct_activities(self):
        events, _, _ = self.run_review("I visited the gallery on 3 June 2028 at Birch Hall.", "separate_event")
        self.assertNotIn("location", events[0]["attributes"])

    def test_same_event_adjacent_attribute_remains_available(self):
        events, _, _ = self.run_review("I arrived at Birch Hall with Nila.", "elaboration")
        self.assertEqual(events[0]["attributes"]["location"], ["Birch Hall"])

    def test_uncertain_link_cannot_become_an_absence_certificate(self):
        events, diagnostics, _ = self.run_review("I arrived at Birch Hall with Nila.", "uncertain")
        self.assertNotIn("location", events[0]["attributes"])
        self.assertTrue(diagnostics["retrieval"]["event_link_review_incomplete"])


class IntervalRefinementSafetyTests(unittest.TestCase):
    def events(self, interval=("2028-06-01", "2028-06-30"), anchor="2028-05-31"):
        base = dict(status="completed", actor="user", attributes={}, attribute_provenance=[],
                    source={}, source_span=[], event="lamp")
        first = dict(base, id="a", matches=["A"],
                     time=dict(interval_start=interval[0], interval_end=interval[1]),
                     relation=dict(direction="after", amount=7, unit="day"), reference_operand="B")
        second = dict(base, id="b", event="blinds", matches=["B"], time=dict(value=anchor),
                      relation=None, reference_operand=None)
        return [first, second]

    def test_exact_offset_inside_month_refines_without_mutating_sources(self):
        events = self.events()
        before = copy.deepcopy(events)
        resolved = algebra._resolve_event_times(events)
        self.assertEqual(algebra._interval(resolved[0])[0].isoformat(), "2028-06-07")
        self.assertEqual(algebra._interval(resolved[0])[0], algebra._interval(resolved[0])[1])
        self.assertEqual(events, before)

    def test_offset_outside_stated_interval_remains_a_conflict(self):
        with self.assertRaises(algebra.Incomplete):
            algebra._resolve_event_times(self.events(anchor="2028-04-30"))

    def test_conflicting_exact_dates_remain_a_conflict(self):
        with self.assertRaises(algebra.Incomplete):
            algebra._resolve_event_times(self.events(interval=("2028-06-08", "2028-06-08")))

    def test_matching_exact_date_does_not_change_precision(self):
        resolved = algebra._resolve_event_times(self.events(interval=("2028-06-07", "2028-06-07")))
        self.assertEqual(algebra._interval(resolved[0])[0].isoformat(), "2028-06-07")

    def test_ambiguous_anchor_cannot_invent_an_exact_date(self):
        events = self.events()
        events.append(dict(events[1], id="c", time=dict(value="2028-06-01")))
        resolved = algebra._resolve_event_times(events)
        self.assertEqual(algebra._interval(resolved[0]), algebra._interval(events[0]))


class TemporalPromotionSafetyTests(unittest.TestCase):
    def test_event_name_misclassified_as_date_window_gets_one_typed_repair(self):
        q = "How many days passed between the workshop and gallery?"
        calls = []
        def call(context, prompt, schema, key, diagnostics, **kwargs):
            calls.append(key)
            if key == "temporal_constraint_presence":
                return {family: family in {"window", "duration"} for family in algebra.QUESTION_FAMILIES if family != "projection"}
            if key == "temporal_question_constraints_window":
                return dict(spans=["workshop"])
            if key == "temporal_question_constraints_window_repair":
                return dict(spans=[])
            if key == "temporal_question_constraints_duration":
                return dict(unit="day", reference="events")
            if key == "temporal_question_constraints_projection":
                return dict(field="time")
            raise AssertionError(key)
        result = algebra.review_question_constraints(None, q, call, {},
            operands=[operand("A", "workshop"), operand("B", "gallery")], temporal_resolver=resolve_temporal_expression)
        self.assertFalse(any(c["kind"] == "window" for c in result["constraints"]))
        self.assertEqual(calls.count("temporal_question_constraints_window_repair"), 1)

    def test_repeated_bad_window_is_uncertainty_not_silent_constraint_removal(self):
        calls = []
        def call(context, prompt, schema, key, diagnostics, **kwargs):
            calls.append(key)
            if key == "temporal_constraint_presence":
                return {family: family == "window" for family in algebra.QUESTION_FAMILIES if family != "projection"}
            return dict(spans=["workshop"])
        with self.assertRaisesRegex(ValueError, "not_temporally_typed"):
            algebra.review_question_constraints(None, "When was the workshop?", call, {},
                                                temporal_resolver=resolve_temporal_expression)
        self.assertEqual(calls.count("temporal_question_constraints_window_repair"), 1)

    def test_complete_simple_candidate_cannot_skip_window_validation(self):
        q = "On what date was my first harbor workshop in June 2028?"
        requirements = inventory(constraint("window", "June 2028", "harbor workshop"),
                                 constraint("earliest", "first", "harbor workshop"),
                                 constraint("projection", "date", value="time"))
        class IndependentReview(FakeLocalModel):
            def llm(self, prompt, **kwargs):
                if "Inventory EVERY" in prompt or "Constraint family:" in prompt:
                    self.calls.append((prompt, kwargs))
                    payload = constraint_review_payload(prompt, requirements["constraints"])
                    return {"choices": [{"text": json.dumps(payload)}]}
                return super().llm(prompt, **kwargs)
        missing_window = contract("harbor workshop", answer_kind="date", projection="time",
                                  reference="none", occurrence="earliest", selection_span="first")
        model = IndependentReview(missing_window, {})
        result = run_temporal_candidate(model, q, legacy_plan(), [], temporal_resolver=resolve_temporal_expression)
        self.assertEqual(result["fallback_action"], "preserve")
        self.assertIsNone(result["proof"])
        self.assertIsNone(result["contract"])
        self.assertIn("constraint_not_represented:window", result["reason"])
        self.assertFalse(promotable_temporal_result(result))
        self.assertNotIn("temporal_assertions", result["calls"])

    def test_valid_simple_window_keeps_its_supported_answer(self):
        q = "On what date was my first harbor workshop in June 2028?"
        request = contract("harbor workshop", answer_kind="date", projection="time", reference="none",
                           occurrence="earliest", selection_span="first", window="June 2028")
        may, june = "I completed the harbor workshop on 2028-05-03.", "I completed the harbor workshop on 2028-06-03."
        rows = [source(may, "1"), source(june, "2")]
        before = copy.deepcopy(rows)
        model = FakeLocalModel(request, {may: annotation(may), june: annotation(june)})
        result = run_temporal_candidate(model, q, legacy_plan(), rows, temporal_resolver=resolve_temporal_expression,
                                        baseline_answer="June 3, 2028", reference_timestamp="2028-06-20")
        self.assertEqual(result["proof"]["answer"], "2028-06-03")
        self.assertTrue(result["proof"]["certificate"]["question_constraints_checked"])
        self.assertTrue(promotable_temporal_result(result))
        self.assertNotIn("composition", result)
        self.assertEqual(rows, before)

    def test_compound_year_month_result_remains_supported(self):
        q = "How many years and months passed between my harbor workshop and gallery visit?"
        request = contract("harbor workshop", "gallery visit", unit="year_month")
        left, right = "I completed the harbor workshop on 2027-04-02.", "I completed the gallery visit on 2028-06-03."
        rows = [source(left, "1"), source(right, "2")]
        model = FakeLocalModel(request, {left: annotation(left), right: annotation(right, matches=("O2",))})
        result = run_temporal_candidate(model, q, legacy_plan(), rows, temporal_resolver=resolve_temporal_expression,
                                        baseline_answer="1 year and 2 months")
        self.assertEqual(result["proof"]["answer"], "1 year and 2 months")
        self.assertTrue(promotable_temporal_result(result))

    def test_implicit_birth_keeps_existing_age_path(self):
        q = "How old was I at the harbor workshop?"
        request = contract("birth", "harbor workshop", unit="year")
        birth, workshop = "I was born on 1990-05-10.", "I completed the harbor workshop on 2028-05-09."
        rows = [source(birth, "1"), source(workshop, "2")]
        model = FakeLocalModel(request, {birth: annotation(birth, predicate="born"),
                                         workshop: annotation(workshop, matches=("O2",))})
        # The existing retriever uses local embeddings to propose birth/born
        # paraphrases; the copied assertion still has to bind the right event.
        model.encode_normalized_texts = lambda texts: [[1.0] for _ in texts]
        result = run_temporal_candidate(model, q, legacy_plan(), rows, temporal_resolver=resolve_temporal_expression,
                                        baseline_answer="37 years")
        self.assertEqual(result["proof"]["answer"], "37 years")
        self.assertTrue(promotable_temporal_result(result))

    def test_supplied_composed_plan_still_requires_question_coverage(self):
        q = "Where was my first workshop in June 2028?"
        request = plan([operand("A", "workshop")],
                       [step("a", "scan", operand="A"), step("e", "earliest", "a"),
                        step("p", "project", "e", field="location")], kind="location", temporal_span=q)
        requirements = inventory(constraint("window", "June 2028", "workshop"),
                                 constraint("earliest", "first", "workshop"))
        def no_calls(*args, **kwargs):
            raise AssertionError("An invalid plan must not review sources or veto the baseline")
        result = algebra.run_composed_candidate(None, q, [], temporal_resolver=resolve_temporal_expression,
            call=no_calls, validated_plan=request, question_constraints=requirements,
            baseline_answer="Supported previous answer", baseline_evidence_complete=True)
        self.assertEqual(result["fallback_action"], "preserve")
        self.assertIsNone(result["proof"])
        self.assertFalse(supported_temporal_abstention(dict(mode="validated", composition=result)))

    def test_source_only_certificate_cannot_replace_an_answer(self):
        certificate = dict(output_typed=True, all_operands_bound=True, actor_and_status_bound=True,
                           source_spans_grounded=True, event_times_not_report_times=True)
        candidate = dict(mode="validated", proof=dict(status="COMPLETE", executable=True, answer="5 days",
                                                       bindings=[{}], certificate=certificate))
        self.assertFalse(promotable_temporal_result(candidate))
        certificate["question_constraints_checked"] = True
        self.assertTrue(promotable_temporal_result(candidate))

    def test_inventory_prompt_contains_no_evidence_or_proposed_answer(self):
        captured = []
        def call(context, prompt, schema, key, diagnostics, **kwargs):
            captured.append(prompt)
            if key == "temporal_constraint_presence":
                return {k: False for k in algebra.QUESTION_FAMILIES if k != "projection"}
            return dict(field="time")
        algebra.review_question_constraints(None, "When was the workshop?", call, {})
        self.assertEqual(len(captured), 2)
        for prompt in captured:
            self.assertNotIn("Evidence:", prompt)
            self.assertNotIn("Proposed answer:", prompt)


if __name__ == "__main__":
    unittest.main()
