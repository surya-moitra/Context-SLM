import copy
import json
import unittest
from unittest.mock import patch

from PRAGMOS_benchmark_LongMemEval import completed_operation_in_trace, resolve_temporal_expression
from PRAGMOS_temporal_algebra import (
    execute_program, prepare_source_rows, run_composed_candidate, validate_plan, validate_review,
)
from PRAGMOS_temporal_contracts import (
    eligible_temporal_candidate, promotable_temporal_result, run_temporal_candidate,
    supported_temporal_abstention,
    lower_temporal_contract,
)
from test_PRAGMOS_temporal_contracts import constraint_review_payload, legacy_plan


def operand(key, text, collection=False, actor="user", **extra):
    return dict(id=key, text=text, actor=actor, collection=collection,
                entities=[], phase="any", status="completed", **extra)


def step(key, op, *args, **extra):
    return dict(id=key, op=op, args=list(args), operand=extra.get("operand"),
                field=extra.get("field"), span=extra.get("span"), unit=extra.get("unit"))


def plan(operands, steps, kind="duration", temporal_span="between"):
    return dict(applicable=True, temporal_span=temporal_span, answer_kind=kind,
                operands=operands, steps=steps, result=steps[-1]["id"] if steps else None)


def difference_plan():
    return plan([operand("A", "installing the light"), operand("B", "ordering the blinds")],
                [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                 step("d", "difference", "a", "b", unit="day")])


QUESTION = "How many days passed between installing the light and ordering the blinds?"


def source(text, key="1", speaker="user", role="user", timestamp="2028-06-20"):
    return dict(source_turn_id=key, source_session_id="session-" + key,
                source_quote=text, source_role=role, source_speaker=speaker,
                source_timestamp=timestamp)


def annotation(text, matches, actor="I", predicate="installed", **extra):
    result = dict(event_span=text, actor_span=actor, predicate_span=predicate,
                  factuality="actual", phase="occurrence", matches=list(matches), attributes=[],
                  reference_operand=None, relation_span=None, relative_direction="none",
                  reference_factuality="unknown", reference_actor_span=None, recount_span=None)
    result.update(extra)
    return result


def event(text, key, matches, **extra):
    row = prepare_source_rows([source(text, key)])[0]
    row["id"] = "S" + key
    payload = annotation(text, matches, **extra)
    return row, payload


def frames(rows, annotations, request):
    result = []
    for row, review in zip(rows, annotations):
        result.extend(validate_review(row, {"status": "reviewed", "events": review}, request,
                                      resolve_temporal_expression))
    return result


def execute(request, evidence, **kwargs):
    return execute_program(request, evidence, temporal_resolver=resolve_temporal_expression, **kwargs)


class FakeModel:
    context_length = 2048

    def __init__(self, request, annotations=None, witnesses=None, uncertain=False):
        self.request = request
        self.annotations = annotations or {}
        self.witnesses = witnesses or {}
        self.uncertain = uncertain
        self.calls = []
        self.operand_names = {o["id"]: "E" + str(i) for i, o in enumerate(request.get("operands", []), 1)}
        self.operand_names.update({key + ":" + field: value + ":" + field
                                   for key, value in list(self.operand_names.items())
                                   for field in ("person", "location", "object")})

    def count_tokens(self, text):
        return len(text.split())

    def call(self, context, prompt, schema, key, diagnostics, **kwargs):
        self.calls.append(key)
        diagnostics["calls"][key] = diagnostics["calls"].get(key, 0) + 1
        if key == "temporal_composition_gate":
            return {"required": bool(self.request.get("applicable"))}
        if key == "temporal_composition_signature":
            return dict(kind=self.request.get("answer_kind", "event"), unit="day", reference="events")
        if key == "temporal_constraint_presence" or key.startswith("temporal_question_constraints_") or key in {"temporal_question_selection_scope", "temporal_question_window_scope"}:
            question = json.loads(prompt.split("\nQuestion: ")[1].split("\n<|end|>")[0])
            specs = {o["id"]: o for o in self.request["operands"]}
            lineage, constraints = {}, []
            for node in self.request["steps"]:
                args, op = node["args"], node["op"]
                lineage[node["id"]] = ({node["operand"]} if op == "scan" else lineage[args[0]]
                                        if op in {"before", "after"} else set().union(*(lineage[a] for a in args)))
                target = specs[next(iter(lineage[node["id"]]))]["text"] if len(lineage[node["id"]]) == 1 else None
                kind = {"argmax": "frequency_max", "argmin": "frequency_min", "difference": "unit"}.get(op, op)
                if kind in {"window", "earliest", "latest", "order", "before", "after", "count", "distinct", "frequency_max", "frequency_min", "unit"}:
                    anchor = specs[next(iter(lineage[args[1]]))]["text"] if op in {"before", "after"} else None
                    constraints.append(dict(kind=kind, span=node.get("span") or question,
                                            target=target if kind != "unit" else None, anchor=anchor,
                                            value=node.get("unit") if kind == "unit" else None))
            root = self.request["steps"][-1]
            projection = "time" if root["op"] == "difference" else root.get("field") or self.request["answer_kind"]
            constraints.append(dict(kind="projection", span=question, target=None, anchor=None,
                                    value=projection))
            return constraint_review_payload(prompt, constraints)
        if key == "temporal_event_selectors":
            return {"selectors": [{k: v for k, v in o.items() if k not in {"id", "phase"}}
                                  for o in self.request.get("operands", [])]}
        if key in {"temporal_composition_plan", "temporal_composition_repair"}:
            nodes = {n["id"]: n for n in self.request.get("steps", [])}
            def expression(key):
                node = nodes[key]
                if node["op"] == "scan":
                    return dict(op="scan", operand=self.operand_names[node["operand"]])
                output = {k: v for k, v in node.items() if k in {"op", "field", "span", "unit"} and v is not None}
                keys = ["input"] if len(node["args"]) == 1 else ["left", "right"]
                output.update({k: expression(a) for k, a in zip(keys, node["args"])})
                return output
            return dict(expression=expression(self.request["result"]))
        if key == "temporal_event_episode_links":
            return dict(same_occurrence="yes")
        rows = json.loads(prompt.split("\nEvidence: ")[1].split("\n<|end|>")[0])
        if key == "temporal_composed_assertions":
            reviews = {}
            for row in rows:
                events = copy.deepcopy(self.annotations.get(row["clause"], []))
                e = events[0] if events else {}
                reviews[row["id"]] = dict(actor_span=e.get("actor_span"), predicate_span=e.get("predicate_span"),
                                           argument_span=None, phase=e.get("phase", "unknown"),
                                           factuality=e.get("factuality", "unknown"),
                                           matches=[self.operand_names.get(m, m) for m in e.get("matches", [])])
            return {"assertions": reviews}
        if key == "temporal_event_attributes":
            return {node["field"]: {"value": None, "evidence_id": None}
                    for node in self.request["steps"] if node.get("field") in {"person", "location", "object"}}
        if key == "temporal_baseline_support_review":
            spec = json.loads(prompt.split("\nOperand: ")[1].split("\nEvidence: ")[0])
            witnesses = [w for w in self.witnesses.get(rows[0]["text"], [])
                         if self.operand_names.get(w["operand"], w["operand"]) == spec["id"]]
            return dict(support="uncertain" if self.uncertain else "event" if witnesses else "none",
                        quote=witnesses[0]["span"] if witnesses else None)
        raise AssertionError("Unexpected stage: " + key)

    def run(self, rows, question=QUESTION, **kwargs):
        return run_composed_candidate(self, question, rows, temporal_resolver=resolve_temporal_expression,
                                      call=self.call, **kwargs)


class TemporalProgramValidationTests(unittest.TestCase):
    def test_valid_program_preserves_all_operands(self):
        self.assertEqual(validate_plan(difference_plan(), QUESTION)["answer_kind"], "duration")

    def test_ignored_required_operand_is_rejected(self):
        request = difference_plan()
        request["steps"] = [step("a", "scan", operand="A"), step("d", "difference", "a", unit="day")]
        with self.assertRaisesRegex(ValueError, "drops_required"):
            validate_plan(request, QUESTION)

    def test_dead_temporal_branch_cannot_authorize_non_temporal_answer(self):
        request = difference_plan()
        request["steps"].append(step("out", "project", "a", field="event"))
        request.update(result="out", answer_kind="event")
        with self.assertRaises(ValueError):
            validate_plan(request, QUESTION)

    def test_forward_references_are_rejected(self):
        request = difference_plan()
        request["steps"][0]["args"] = ["d"]
        with self.assertRaisesRegex(ValueError, "cyclic"):
            validate_plan(request, QUESTION)

    def test_output_cannot_change_from_duration_to_count(self):
        request = difference_plan()
        request["answer_kind"] = "count"
        with self.assertRaisesRegex(ValueError, "output_type"):
            validate_plan(request, QUESTION)

    def test_grouped_venue_counts_cannot_project_a_person(self):
        request = plan([operand("A", "activities", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("g", "group_count", "w", field="location"), step("m", "argmax", "g", field="person")],
                       kind="person", temporal_span="June 2028")
        with self.assertRaisesRegex(ValueError, "projection"):
            validate_plan(request, "Which activities did I attend in June 2028?")

    def test_non_temporal_arithmetic_does_not_activate(self):
        request = dict(applicable=False)
        model = FakeModel(request)
        result = model.run([])
        self.assertFalse(result["applicable"])
        self.assertEqual(result["fallback_action"], "preserve")
        self.assertEqual(model.calls, ["temporal_composition_gate"])

    def test_a_set_cannot_be_its_own_before_anchor(self):
        request = plan([operand("A", "activities", True)],
                       [step("a", "scan", operand="A"), step("f", "before", "a", "a"),
                        step("n", "count", "f")], kind="count", temporal_span="before")
        with self.assertRaisesRegex(ValueError, "own_temporal_anchor"):
            validate_plan(request, "How many activities happened before the anchor?")


class TemporalSourceBindingTests(unittest.TestCase):
    def test_companion_and_venue_bound_to_correct_episode(self):
        text = "I attended the workshop on 2028-06-03. Nila came with me to Cedar Hall. I visited the gallery on 2028-06-09. Omar came with me to Birch Hall."
        request = plan([operand("A", "activities", True)], [step("a", "scan", operand="A")], kind="event")
        row = prepare_source_rows([source(text)])[0]
        reviews = [annotation("I attended the workshop on 2028-06-03.", ["A"], predicate="attended",
                              attributes=[dict(role="person", span="Nila", link_span="Nila came with me to Cedar Hall."),
                                          dict(role="location", span="Cedar Hall", link_span="Nila came with me to Cedar Hall.")]),
                   annotation("I visited the gallery on 2028-06-09.", ["A"], predicate="visited",
                              attributes=[dict(role="person", span="Omar", link_span="Omar came with me to Birch Hall.")])]
        evidence = frames([row], [reviews], request)
        self.assertEqual(evidence[0]["attributes"]["person"], ["Nila"])
        self.assertEqual(evidence[1]["attributes"]["person"], ["Omar"])

    def test_attribute_cannot_cross_another_primary_event(self):
        text = "I attended the workshop on 2028-06-03. I visited the gallery on 2028-06-09. Omar came with me."
        row = prepare_source_rows([source(text)])[0]
        request = plan([operand("A", "activities", True)], [], kind="event")
        reviews = [annotation("I attended the workshop on 2028-06-03.", ["A"], predicate="attended",
                              attributes=[dict(role="person", span="Omar", link_span="Omar came with me.")]),
                   annotation("I visited the gallery on 2028-06-09.", ["A"], predicate="visited")]
        with self.assertRaisesRegex(ValueError, "crosses"):
            frames([row], [reviews], request)

    def test_wrong_speaker_and_planned_activity_do_not_bind(self):
        request = difference_plan()
        text = "Nila installed the light on 2028-06-01."
        row = prepare_source_rows([source(text)])[0]
        evidence = frames([row], [[annotation(text, ["A"], actor="Nila")]], request)
        self.assertEqual(evidence[0]["matches"], [])
        text = "I planned installing the light on 2028-06-01."
        row = prepare_source_rows([source(text)])[0]
        evidence = frames([row], [[annotation(text, ["A"], predicate="planned", factuality="planned")]], request)
        self.assertEqual(evidence[0]["matches"], [])

    def test_assistant_example_is_not_user_event(self):
        text = "I installed the light on 2028-06-01."
        row = prepare_source_rows([source(text, speaker="assistant", role="assistant")])[0]
        evidence = frames([row], [[annotation(text, ["A"])]], difference_plan())
        self.assertEqual(evidence[0]["matches"], [])

    def test_entity_labels_are_not_fuzzy_merged(self):
        request = difference_plan()
        request["operands"][0]["entities"] = ["red light"]
        text = "I installed the blue light on 2028-06-01."
        row = prepare_source_rows([source(text)])[0]
        evidence = frames([row], [[annotation(text, ["A"])]], request)
        self.assertEqual(evidence[0]["matches"], [])

    def test_finish_request_does_not_accept_start(self):
        request = difference_plan()
        request["operands"][0]["phase"] = "end"
        text = "I started installing the light on 2028-06-01."
        row = prepare_source_rows([source(text)])[0]
        evidence = frames([row], [[annotation(text, ["A"], predicate="started", phase="start")]], request)
        self.assertEqual(evidence[0]["matches"], [])

    def test_same_primary_assertion_cannot_cover_two_named_events(self):
        text = "I installed the light and ordered the blinds on 2028-06-01."
        row = prepare_source_rows([source(text)])[0]
        with self.assertRaisesRegex(ValueError, "distinct_named"):
            frames([row], [[annotation(text, ["A", "B"])]], difference_plan())


class TemporalCompositionExecutionTests(unittest.TestCase):
    def test_relative_interval_without_absolute_dates(self):
        text = "I ordered the blinds three days after installing the light."
        row = prepare_source_rows([source(text, timestamp=None)])[0]
        review = annotation(text, ["B"], predicate="ordered", reference_operand="A",
                            relation_span="three days after installing the light", relative_direction="after",
                            reference_factuality="actual")
        evidence = frames([row], [[review]], difference_plan())
        result = execute(difference_plan(), evidence)
        self.assertEqual(result["answer"], "3 days")
        self.assertTrue(result["relative_witness_ids"])

    def test_relative_order_without_absolute_dates(self):
        request = plan([operand("A", "installing the light"), operand("B", "ordering the blinds")],
                       [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                        step("u", "union", "a", "b"), step("e", "earliest", "u"),
                        step("p", "project", "e", field="event")], kind="event")
        text = "I ordered the blinds after installing the light."
        row = prepare_source_rows([source(text, timestamp=None)])[0]
        review = annotation(text, ["B"], predicate="ordered", reference_operand="A",
                            relation_span="after installing the light", relative_direction="after", reference_factuality="actual")
        result = execute(request, frames([row], [[review]], request))
        self.assertEqual(result["answer"], "installing the light")

    def test_unknown_reference_status_does_not_prove_an_endpoint(self):
        text = "I ordered the blinds three days after installing the light."
        row = prepare_source_rows([source(text)])[0]
        review = annotation(text, ["B"], predicate="ordered", reference_operand="A",
                            relation_span="three days after installing the light", relative_direction="after")
        with self.assertRaisesRegex(ValueError, "not_asserted"):
            frames([row], [[review]], difference_plan())

    def test_relative_chain_across_sessions(self):
        request = difference_plan()
        request["operands"].append(operand("C", "attending the workshop"))
        request["steps"] = [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                            step("c", "scan", operand="C"), step("u", "union", "b", "c"),
                            step("f", "earliest", "u"), step("d", "difference", "a", "f", unit="day")]
        row1, a1 = event("I attended the workshop two days after installing the light.", "1", ["C"], predicate="attended",
                         reference_operand="A", relation_span="two days after installing the light", relative_direction="after", reference_factuality="actual")
        row2, a2 = event("I ordered the blinds four days after attending the workshop.", "2", ["B"], predicate="ordered",
                         reference_operand="C", relation_span="four days after attending the workshop", relative_direction="after", reference_factuality="actual")
        result = execute(request, frames([row1, row2], [[a1], [a2]], request))
        self.assertEqual(result["answer"], "2 days")

    def test_window_does_not_filter_an_outside_anchor(self):
        request = plan([operand("A", "activities", True), operand("B", "buying the lens")],
                       [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                        step("w", "window", "a", span="June 2028"), step("f", "after", "w", "b"),
                        step("e", "earliest", "f"), step("p", "project", "e", field="event")],
                       kind="event", temporal_span="June 2028")
        row1, a1 = event("I bought the lens on 2028-05-20.", "1", ["B"], predicate="bought")
        row2, a2 = event("I attended the workshop on 2028-06-03.", "2", ["A"], predicate="attended")
        row3, a3 = event("I visited the gallery on 2028-06-09.", "3", ["A"], predicate="visited")
        result = execute(request, frames([row1, row2, row3], [[a1], [a2], [a3]], request), collection_complete=True)
        self.assertIn("workshop", result["answer"])

    def test_missing_companion_never_substitutes_performer(self):
        request = plan([operand("A", "activities", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("p", "project", "w", field="person")], kind="person", temporal_span="June 2028")
        row, a = event("I attended the workshop on 2028-06-03.", "1", ["A"], predicate="attended")
        result = execute(request, frames([row], [[a]], request), collection_complete=True)
        self.assertEqual(result["evidence_state"], "attribute_not_grounded")
        self.assertIsNone(result["answer"])

    def test_count_before_anchor_counts_other_occurrences(self):
        request = plan([operand("A", "activities", True), operand("B", "attending the induction")],
                       [step("a", "scan", operand="A"), step("b", "scan", operand="B"),
                        step("f", "before", "a", "b"), step("n", "count", "f")], kind="count")
        rows, annotations = [], []
        for index, date in enumerate(["2028-06-01", "2028-06-04", "2028-06-08", "2028-06-10"]):
            text = f"I attended the {'induction' if index == 3 else 'workshop'} on {date}."
            row, a = event(text, str(index), ["A", "B"] if index == 3 else ["A"], predicate="attended")
            rows.append(row)
            annotations.append([a])
        result = execute(request, frames(rows, annotations, request), collection_complete=True)
        self.assertEqual(result["answer"], "3")

    def test_incomplete_collection_is_not_a_certified_total(self):
        request = plan([operand("A", "activities", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("n", "count", "w")], kind="count", temporal_span="June 2028")
        row, a = event("I attended the workshop on 2028-06-03.", "1", ["A"], predicate="attended")
        result = execute(request, frames([row], [[a]], request))
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["partial_value"], "1")

    def test_zero_is_valid_only_with_complete_collection_coverage(self):
        request = plan([operand("A", "activities", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("n", "count", "w")], kind="count", temporal_span="June 2028")
        self.assertEqual(execute(request, [], collection_complete=True)["answer"], "0")
        self.assertEqual(execute(request, [])["status"], "PARTIAL")

    def test_frequency_counts_visits_not_distinct_venues(self):
        request = plan([operand("A", "activities", True)],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("g", "group_count", "w", field="location"), step("m", "argmax", "g", field="location")],
                       kind="location", temporal_span="June 2028")
        rows, annotations = [], []
        for index, venue in enumerate(["Cedar Hall", "Birch Hall", "Cedar Hall", "Birch Hall", "Cedar Hall"], 1):
            text = f"I attended the workshop at {venue} on 2028-06-{index:02d}."
            row, a = event(text, str(index), ["A"], predicate="attended",
                           attributes=[dict(role="location", span=venue, link_span=text)])
            rows.append(row)
            annotations.append([a])
        result = execute(request, frames(rows, annotations, request), collection_complete=True)
        self.assertEqual(result["answer"], "Cedar Hall")

    def test_unknown_date_is_resolution_failure_not_missing_event(self):
        row1, a1 = event("I installed the light some time ago.", "1", ["A"])
        row2, a2 = event("I ordered the blinds on 2028-06-05.", "2", ["B"], predicate="ordered")
        result = execute(difference_plan(), frames([row1, row2], [[a1], [a2]], difference_plan()))
        self.assertEqual(result["evidence_state"], "temporal_resolution_failed")
        self.assertEqual(result["obligations"]["A"]["event_identity"], "supported")

    def test_duplicate_mentions_do_not_bind_a_different_operand(self):
        rows, annotations = [], []
        for key in ["1", "2"]:
            row, a = event("I installed the light on 2028-06-01.", key, ["A"])
            rows.append(row)
            annotations.append([a])
        result = execute(difference_plan(), frames(rows, annotations, difference_plan()))
        self.assertEqual(result["obligations"]["B"]["event_identity"], "not_found_in_reviewed_evidence")


class TemporalFallbackSafetyTests(unittest.TestCase):
    def setUp(self):
        self.text = "I installed the light on 2028-06-01."
        self.sources = [source(self.text)]
        self.annotations = {self.text: [annotation(self.text, ["A"])]}

    def test_missing_event_requires_independent_raw_review_and_retrieval(self):
        model = FakeModel(difference_plan(), self.annotations,
                          {self.text: [dict(operand="A", span=self.text)]})
        result = model.run(self.sources, baseline_answer="4 days", baseline_evidence_complete=True,
                           retrieve_missing=lambda _: [])
        self.assertEqual(result["fallback_action"], "insufficient_evidence")
        wrapped = dict(mode="validated", fallback_action=result["fallback_action"], composition=result)
        self.assertTrue(supported_temporal_abstention(wrapped))
        wrapped["mode"] = "shadow"
        self.assertFalse(supported_temporal_abstention(wrapped))

    def test_independent_witness_rescues_an_extraction_miss(self):
        text = "I installed the light on 2028-06-01 and ordered the blinds on 2028-06-05."
        model = FakeModel(difference_plan(), {text: [annotation(text, [])]}, {text: [dict(operand="A", span="installed the light"),
                                                       dict(operand="B", span="ordered the blinds")]})
        result = model.run([source(text)], baseline_answer="4 days", baseline_evidence_complete=True,
                           retrieve_missing=lambda _: [])
        self.assertEqual(result["fallback_action"], "preserve")
        self.assertEqual(result["baseline_support_review"]["missing_operands"], [])

    def test_unknown_source_review_cannot_certify_absence(self):
        model = FakeModel(difference_plan(), {})
        result = model.run(self.sources, baseline_answer="existing answer", baseline_evidence_complete=True,
                           retrieve_missing=lambda _: [])
        self.assertFalse(result["source_reviews_complete"])
        self.assertEqual(result["fallback_action"], "preserve")

    def test_missing_attribute_gets_targeted_search_and_independent_review(self):
        q = "Who accompanied me to the workshop in June 2028?"
        request = plan([operand("A", "workshop")],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("p", "project", "w", field="person")], kind="person", temporal_span="June 2028")
        text = "I attended the workshop on 2028-06-03."
        model = FakeModel(request, {text: [annotation(text, ["A"], predicate="attended")]})
        queries = []
        result = model.run([source(text)], question=q, baseline_answer="the performer",
                           baseline_evidence_complete=True, retrieve_missing=lambda query: queries.append(query) or [])
        self.assertEqual(queries, ["workshop person"])
        self.assertEqual(result["fallback_action"], "insufficient_evidence")
        self.assertEqual(result["proof"]["mandatory_gaps"][0]["required_field"], "person")

    def test_raw_attribute_witness_rescues_attribute_extraction_failure(self):
        q = "Who accompanied me to the workshop in June 2028?"
        request = plan([operand("A", "workshop")],
                       [step("a", "scan", operand="A"), step("w", "window", "a", span="June 2028"),
                        step("p", "project", "w", field="person")], kind="person", temporal_span="June 2028")
        first, second = "I attended the workshop on 2028-06-03.", "Nila came with me."
        text = first + " " + second
        model = FakeModel(request, {first: [annotation(first, ["A"], predicate="attended")],
                                    second: [annotation(second, [], actor="Nila", predicate="came")]},
                          {text: [dict(operand="A:person", span=second)]})
        result = model.run([source(text)], question=q, baseline_answer="Nila", baseline_evidence_complete=True,
                           retrieve_missing=lambda _: [])
        self.assertEqual(result["fallback_action"], "preserve")
        self.assertEqual(result["baseline_support_review"]["missing_operands"], [])

    def test_compound_or_subday_units_never_silently_become_days_or_years(self):
        for unit in ["year_month", "hour", "minute"]:
            previous = dict(schema_version=2, mode="validated", applicable=True, status="PARTIAL",
                            calls={}, answer_contract=dict(kind="duration", unit=unit, reference="events"), proof=None)
            with self.subTest(unit=unit), patch("PRAGMOS_temporal_contracts._run_simple_temporal_candidate", return_value=previous), \
                 patch("PRAGMOS_temporal_algebra.run_composed_candidate", side_effect=AssertionError("precision changed")):
                result = run_temporal_candidate(object(), QUESTION, legacy_plan(), self.sources,
                                                temporal_resolver=resolve_temporal_expression, baseline_answer="existing")
                self.assertEqual(result["fallback_action"], "preserve")

    def test_lowering_preserves_existing_query_and_input_contract(self):
        from test_PRAGMOS_temporal_contracts import contract
        original = contract("installing the light", "ordering the blinds")
        before = copy.deepcopy(original)
        lowered = lower_temporal_contract(original, QUESTION)
        self.assertIsNotNone(lowered)
        self.assertEqual(lowered["steps"][-1]["op"], "difference")
        self.assertEqual(original, before)

    def test_zero_promotion_needs_complete_reviewed_source_coverage(self):
        certificate = dict(output_typed=True, all_operands_bound=True, actor_and_status_bound=True,
                           source_spans_grounded=True, event_times_not_report_times=True,
                           question_constraints_checked=True)
        candidate = dict(mode="validated", proof=dict(status="COMPLETE", executable=True, answer="0",
                                                       bindings=[], certificate=certificate))
        self.assertFalse(promotable_temporal_result(candidate))
        candidate["proof"]["search_coverage"] = dict(corpus_complete=True, reviewed_turn_count=1,
                                                    sources=[dict(session_id="s", turn_id="1")])
        self.assertTrue(promotable_temporal_result(candidate))

    def test_blocked_legacy_proof_is_not_reported_as_a_completed_operation(self):
        row = dict(operation_result=dict(status="complete"),
                   evidence_compiler_proof=dict(status="COMPLETE", executable=True),
                   temporal_contract_result=dict(abstention_applied=True))
        self.assertFalse(completed_operation_in_trace(row))
        row["temporal_contract_result"] = dict(promoted=True)
        self.assertTrue(completed_operation_in_trace(row))

    def test_uncertain_independent_review_preserves_baseline(self):
        model = FakeModel(difference_plan(), self.annotations, uncertain=True)
        result = model.run(self.sources, baseline_answer="4 days", baseline_evidence_complete=True,
                           retrieve_missing=lambda _: [])
        self.assertEqual(result["fallback_action"], "preserve")

    def test_missing_baseline_evidence_cannot_authorize_abstention(self):
        model = FakeModel(difference_plan(), self.annotations)
        result = model.run(self.sources, baseline_answer="4 days", retrieve_missing=lambda _: [])
        self.assertEqual(result["fallback_action"], "preserve")
        self.assertNotIn("temporal_baseline_support_review", model.calls)

    def test_retrieval_cap_cannot_become_an_absence_proof(self):
        rows = self.sources + [source("We discussed unrelated plans.", str(i)) for i in range(2, 28)]
        model = FakeModel(difference_plan(), self.annotations)
        result = model.run(rows, baseline_answer="4 days", baseline_evidence_complete=True,
                           retrieve_missing=lambda _: [])
        self.assertTrue(result["retrieval"]["truncated"])
        self.assertEqual(result["fallback_action"], "preserve")

    def test_targeted_retrieval_recovers_an_operand(self):
        other = "I ordered the blinds on 2028-06-05."
        annotations = {**self.annotations, other: [annotation(other, ["B"], predicate="ordered")]}
        model = FakeModel(difference_plan(), annotations)
        result = model.run(self.sources, baseline_answer="unknown", baseline_evidence_complete=True,
                           retrieve_missing=lambda _: [source(other, "2")])
        self.assertEqual(result["proof"]["answer"], "4 days")
        self.assertEqual(result["fallback_action"], "preserve")

    def test_parser_failure_preserves_baseline(self):
        model = FakeModel({"applicable": True})
        result = model.run(self.sources, baseline_answer="correct existing answer")
        self.assertEqual(result["status"], "EXTRACTION_FAILED")
        self.assertEqual(result["fallback_action"], "preserve")

    def test_source_views_are_not_mutated(self):
        before = copy.deepcopy(self.sources)
        model = FakeModel(difference_plan(), self.annotations)
        model.run(self.sources, baseline_answer="answer", retrieve_missing=lambda _: [])
        self.assertEqual(self.sources, before)

    def test_protected_routes_and_off_mode_do_not_call_model(self):
        class NoCalls:
            def llm(self, *args, **kwargs):
                raise AssertionError("unexpected model call")
        request = legacy_plan("COUNT_DISTINCT", role="count")
        self.assertTrue(eligible_temporal_candidate(request))
        for kwargs in [dict(mode="off"), dict(protected=True),
                       dict(query_intent={"intent": "assistant_memory"}),
                       dict(query_intent={"intent": "preference_recommendation"}),
                       dict(state_selector={"deterministic_enabled": True, "requested_attribute": "employer"})]:
            with self.subTest(kwargs=kwargs):
                result = run_temporal_candidate(NoCalls(), QUESTION, request, self.sources,
                                                temporal_resolver=resolve_temporal_expression,
                                                baseline_answer="existing", **kwargs)
                self.assertEqual(result["calls"], {})
                self.assertFalse(promotable_temporal_result(result))
                self.assertFalse(supported_temporal_abstention(result))


if __name__ == "__main__":
    unittest.main()
