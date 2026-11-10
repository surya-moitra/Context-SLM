import copy
import dataclasses
import datetime
import json
import unittest

from PRAGMOS_benchmark_LongMemEval import resolve_temporal_expression
from PRAGMOS_evidence_compiler import build_evidence_ledger, parse_semantic_plan
from PRAGMOS_temporal_contracts import (
    compile_program, eligible_request, execute_contract, parse_contract,
    run_temporal_candidate, select_candidate_atoms, validate_assertion,
    promotable_temporal_result,
)


def contract(*events, **kwargs):
    result = dict(task="derive", answer_kind="duration", projection="time", unit="day",
                  reference="events", relation="none", occurrence="unique", window=None,
                  status="completed", events=[dict(id=f"O{i + 1}", text=text, actor="user",
                                                   predicate=None, phase="any", entities=[])
                                             for i, text in enumerate(events)])
    result.update(kwargs)
    return result


def source(text, turn="1", session="s1", speaker="user", role="user", timestamp="2027-04-20"):
    return dict(source_quote=text, source_turn_id=turn, source_session_id=session,
                source_speaker=speaker, source_role=role, source_timestamp=timestamp)


def assertion(atom, *, actor="I", predicate="completed", argument=None,
              matches=("O1",), status="completed", event_span=None):
    return validate_assertion(dict(id=atom.atom_id, event_span=event_span or atom.clause,
                                   actor_span=actor, predicate_span=predicate,
                                   argument_span=argument, status=status, matches=list(matches),
                                   phase="occurrence",
                                   polarity="negated" if status=="negated" else "affirmed" if status!="unknown" else "uncertain",
                                   modality="actual" if status=="completed" else "planned" if status=="planned" else "unknown"),
                              atom, contract("harbor workshop", "gallery visit"),
                              resolve_temporal_expression)


def constraint_review_payload(prompt, constraints):
    from PRAGMOS_temporal_algebra import QUESTION_FAMILIES
    if "constraint flags" in prompt:
        return {family: any(c["kind"] in kinds for c in constraints)
                for family, kinds in QUESTION_FAMILIES.items() if family != "projection"}
    family = prompt.split("Constraint family: ")[1].split(".")[0]
    if family == "window_scope":
        span = json.loads(prompt.split("\nCopied date filter: ")[1].split("\nQuestion event names:")[0])
        return dict(target=next(c["target"] for c in constraints if c["kind"] == "window" and c["span"] == span))
    if family == "selection_scope":
        op = json.loads(prompt.split("\nRequested selection: ")[1].split("\nQuestion event names:")[0])
        return dict(target=next(c["target"] for c in constraints if c["kind"] == op))
    selected = [c for c in constraints if c["kind"] in QUESTION_FAMILIES[family]]
    if family == "window":
        return dict(spans=[c["span"] for c in selected])
    if family == "selection":
        return {op: any(c["kind"] == op for c in selected) for op in QUESTION_FAMILIES["selection"]}
    if family == "comparison":
        return dict(relations=[dict(direction=c["kind"], target=c["target"], anchor=c["anchor"]) for c in selected])
    if family == "actor":
        return dict(bindings=[dict(actor=c["value"], target=c["target"]) for c in selected])
    if family == "aggregation":
        kinds = {c["kind"] for c in selected}
        return dict(operator="count_distinct" if {"count", "distinct"} <= kinds else selected[0]["kind"] if selected else "none",
                    target=selected[0]["target"] if selected else None)
    if family == "duration":
        return dict(unit=next((c["value"] for c in selected if c["kind"] == "unit"), "unknown"),
                    reference="question_time" if any(c["kind"] == "question_time" for c in selected) else "events")
    if family == "projection":
        return dict(field=selected[0]["value"])
    return dict(value=selected[0]["value"] if selected else "any", target=selected[0]["target"] if selected else None)


class FakeLocalModel:
    context_length = 2048

    def __init__(self, request, annotations):
        self.request, self.annotations, self.calls = request, annotations, []

    def llm(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if "Describe ONLY the type" in prompt:
            payload = dict(kind=self.request.get("answer_kind", "unknown"),
                           unit=self.request.get("unit", "unknown"),
                           reference=self.request.get("reference", "none"))
        elif "temporal REQUEST" in prompt:
            payload = self.request
        elif "Inventory EVERY" in prompt or "Constraint family:" in prompt:
            question = json.loads(prompt.split("\nQuestion: ")[1].split("\n<|end|>")[0])
            def item(kind, span=None, target=None, value=None):
                return dict(kind=kind, span=span or question, target=target, anchor=None, value=value)
            constraints = []
            if self.request.get("window"):
                constraints.extend(item("window", self.request["window"], e["text"]) for e in self.request["events"])
            if self.request.get("occurrence", "unique") != "unique":
                constraints.extend(item(self.request["occurrence"], target=e["text"]) for e in self.request["events"])
            if self.request.get("relation") in {"earliest", "latest", "order"}:
                constraints.append(item(self.request["relation"]))
            if self.request.get("answer_kind") == "duration":
                constraints.append(item("unit", value=self.request["unit"]))
                if self.request.get("reference") == "question_time":
                    constraints.append(item("question_time"))
            projection = "person" if self.request.get("projection") == "argument" else self.request.get("projection")
            constraints.append(item("projection", value=projection))
            payload = constraint_review_payload(prompt, constraints)
        else:
            evidence = json.loads(prompt.split("\nEvidence: ")[1].split("\n<|end|>")[0])
            payload = {"assertions": [dict(id=row["id"], **self.annotations[row["clause"]]) for row in evidence]}
        return {"choices": [{"text": json.dumps(payload)}]}


def annotation(text, actor="I", predicate="completed", argument=None, matches=("O1",), status="completed"):
    return dict(event_span=text, actor_span=actor, predicate_span=predicate,
                argument_span=argument, matches=list(matches), status=status,
                phase="occurrence",
                polarity="negated" if status=="negated" else "affirmed" if status!="unknown" else "uncertain",
                modality="actual" if status=="completed" else "planned" if status=="planned" else "unknown")


def legacy_plan(operator="DATE_LOOKUP", role="date", kinds=("event",)):
    return parse_semantic_plan(dict(operator=operator, answer_type="scalar", target_unit=None,
                                   actor="user", answer_role=role, group_role="event",
                                   requested_status="completed", confidence="high",
                                   operands=[dict(id=f"O{i}", text="harbor workshop", kind=kind,
                                                  actor="user", hard_entities=[])
                                             for i, kind in enumerate(kinds)]),
                               "harbor workshop")


class TemporalRequestContractTests(unittest.TestCase):
    def test_duration_requires_a_clock_and_one_event_or_two_events(self):
        c = contract("harbor workshop")
        with self.assertRaisesRegex(ValueError, "endpoints"):
            parse_contract(c, "How long since my harbor workshop?")
        c["reference"] = "question_time"
        self.assertEqual(parse_contract(c, "How long since my harbor workshop?"), c)

    def test_output_and_projection_must_agree(self):
        c = contract("harbor workshop", reference="question_time", projection="event")
        with self.assertRaisesRegex(ValueError, "projection"):
            parse_contract(c, "How long since my harbor workshop?")

    def test_invented_event_or_actor_is_rejected(self):
        c = contract("harbor workshop", "gallery visit")
        with self.assertRaisesRegex(ValueError, "event_not"):
            parse_contract(c, "How long since my harbor workshop?")
        c["events"][0]["actor"] = "Unknown Person"
        with self.assertRaisesRegex(ValueError, "actor_not"):
            parse_contract(c, "How long between my harbor workshop and gallery visit?")

    def test_duplicate_event_ids_cannot_cover_two_operands(self):
        c = contract("harbor workshop", "gallery visit")
        c["events"][1]["id"] = "O1"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_contract(c, "harbor workshop and gallery visit")

    def test_age_can_request_implicit_birth_but_not_an_invented_date(self):
        c = contract("birth", "harbor workshop", unit="year")
        self.assertEqual(parse_contract(c, "How old was I at the harbor workshop?"), c)

    def test_recall_is_not_a_derived_event_computation(self):
        c = contract("harbor workshop", task="recall")
        with self.assertRaisesRegex(ValueError, "not_a_derived"):
            parse_contract(c, "How long was my harbor workshop?")

    def test_request_window_must_come_from_the_question(self):
        c = contract("harbor workshop", answer_kind="date", projection="time", reference="none", window="July 2027")
        with self.assertRaisesRegex(ValueError, "window_not"):
            parse_contract(c, "When was my harbor workshop?")

    def test_program_preserves_duration_type_and_reference_clock(self):
        p = compile_program(contract("harbor workshop", reference="question_time"))
        self.assertEqual(p["op"], "calendar_difference")
        self.assertEqual(p["returns"], "duration")
        self.assertEqual(p["args"][-1]["op"], "question_time")

    def test_order_requires_multiple_events(self):
        c = contract("harbor workshop", answer_kind="ordered_events", projection="event", relation="order")
        with self.assertRaisesRegex(ValueError, "multiple"):
            parse_contract(c, "harbor workshop")


class TemporalAssertionContractTests(unittest.TestCase):
    def test_reporter_is_not_the_performer(self):
        atom = build_evidence_ledger([source("Mira completed the harbor workshop on 2027-04-02.")])[0]
        a = assertion(atom, actor="Mira")
        self.assertEqual(a["actor"], "Mira")
        self.assertEqual(a["matches"], [])

    def test_user_role_does_not_merge_named_peers(self):
        atom = build_evidence_ledger([source("I completed the harbor workshop on 2027-04-02.", speaker="Mira")])[0]
        a = assertion(atom)
        self.assertEqual(a["actor"], "Mira")
        self.assertEqual(a["matches"], [])

    def test_assistant_example_cannot_become_a_user_event(self):
        atom = build_evidence_ledger([source("user completed the harbor workshop on 2027-04-02.",
                                              speaker="assistant", role="assistant")])[0]
        a = assertion(atom, actor="user")
        self.assertEqual(a["matches"], [])

    def test_late_first_person_cannot_own_another_predicate(self):
        atom = build_evidence_ledger([source("Mira was planning a harbor workshop, not an event I completed.")])[0]
        with self.assertRaisesRegex(ValueError, "performer_not"):
            assertion(atom, actor="I", predicate="planning")

    def test_all_roles_must_share_one_grounded_assertion(self):
        atom = build_evidence_ledger([source("Mira completed the harbor workshop while I watched.")])[0]
        with self.assertRaisesRegex(ValueError, "same_assertion"):
            assertion(atom, event_span="Mira completed the harbor workshop", actor="I")

    def test_hallucinated_actor_or_predicate_cannot_certify(self):
        atom = build_evidence_ledger([source("I completed the harbor workshop on 2027-04-02.")])[0]
        for kwargs in (dict(actor="Mira"), dict(predicate="attended")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                assertion(atom, **kwargs)

    def test_hallucinated_optional_argument_cannot_be_projected(self):
        atom = build_evidence_ledger([source("I completed the harbor workshop on 2027-04-02.")])[0]
        with self.assertRaisesRegex(ValueError, "argument_span"):
            assertion(atom, argument="invented participant")

    def test_named_arguments_remain_distinct(self):
        c = contract("The Silver Orchard", reference="question_time")
        c["events"][0]["entities"] = ["The Silver Orchard"]
        atom = build_evidence_ledger([source("I finished The Glass Orchard yesterday.")])[0]
        raw = annotation(atom.clause, predicate="finished")
        raw["id"] = atom.atom_id
        a = validate_assertion(raw, atom, c, resolve_temporal_expression)
        self.assertEqual(a["matches"], [])

    def test_time_reference_mention_cannot_bind_primary_event(self):
        c = contract("harbor workshop", "gallery visit")
        c["events"][0]["entities"] = ["harbor workshop"]
        atom = build_evidence_ledger([source("I completed the gallery visit three days after the harbor workshop.")])[0]
        raw = dict(id=atom.atom_id, **annotation(atom.clause))
        a = validate_assertion(raw, atom, c, resolve_temporal_expression)
        self.assertEqual(a["matches"], ["O2"])

    def test_raw_provenance_is_not_rewritten(self):
        atom = build_evidence_ledger([source("I completed the harbor workshop on 2027-04-02.")])[0]
        before = atom.as_dict()
        a = assertion(atom)
        self.assertEqual(atom.as_dict(), before)
        self.assertEqual(a["atom"].source_quote, atom.source_quote)

    def test_completed_flag_cannot_override_planned_modality(self):
        atom = build_evidence_ledger([source("I planned to complete the harbor workshop.")])[0]
        raw = dict(id=atom.atom_id, **annotation(atom.clause, predicate="complete"))
        raw["modality"] = "planned"
        with self.assertRaisesRegex(ValueError, "modality"):
            validate_assertion(raw, atom, contract("harbor workshop"), resolve_temporal_expression)

    def test_completion_of_an_activity_cannot_bind_its_start(self):
        c = contract("starting the harbor workshop", reference="question_time")
        c["events"][0].update(phase="start", entities=["harbor workshop"])
        atom = build_evidence_ledger([source("I finished the harbor workshop on 2027-04-02.")])[0]
        raw = dict(id=atom.atom_id, **annotation(atom.clause, predicate="finished"))
        raw["phase"] = "end"
        a = validate_assertion(raw, atom, c, resolve_temporal_expression)
        self.assertEqual(a["matches"], [])

    def test_one_clause_cannot_give_two_distinct_events_the_same_unbound_date(self):
        c = contract("harbor workshop", "gallery visit")
        atom = build_evidence_ledger([source("I completed the harbor workshop on 2027-04-02 and the gallery visit on 2027-04-07.")])[0]
        a = validate_assertion(dict(id=atom.atom_id, **annotation(atom.clause)), atom, c, resolve_temporal_expression)
        self.assertEqual(a["matches"], [])
        self.assertEqual(a["rejection_reason"], "multiple_events_need_separate_time_bindings")


class TemporalExecutionContractTests(unittest.TestCase):
    def events(self, left="2027-04-02", right="2027-04-07", **source_kwargs):
        atoms = build_evidence_ledger([
            source(f"I completed the harbor workshop on {left}.", **source_kwargs),
            source(f"I completed the gallery visit on {right}.", turn="2", session="s2", **source_kwargs),
        ])
        return [assertion(atoms[0]), assertion(atoms[1], matches=("O2",))]

    def execute(self, c, assertions, **kwargs):
        return execute_contract(c, assertions, temporal_resolver=resolve_temporal_expression, **kwargs)

    def test_subtracts_event_times_not_report_times(self):
        p = self.execute(contract("harbor workshop", "gallery visit"), self.events(timestamp="2027-06-01"))
        self.assertEqual(p["answer"], "5 days")
        self.assertTrue(p["executable"])

    def test_ago_uses_question_clock_and_preserves_units(self):
        p = self.execute(contract("harbor workshop", reference="question_time", unit="week"),
                         self.events()[:1], reference_timestamp="2027-04-23")
        self.assertEqual(p["answer"], "3 weeks")

    def test_missing_question_clock_never_uses_machine_clock(self):
        p = self.execute(contract("harbor workshop", reference="question_time"), self.events()[:1])
        self.assertFalse(p["executable"])
        self.assertEqual(p["reason"], "question_reference_time_missing")

    def test_missing_operand_cannot_be_replaced_by_duplicate_mentions(self):
        events = self.events()[:1]
        p = self.execute(contract("harbor workshop", "gallery visit"), events + events)
        self.assertFalse(p["executable"])
        self.assertIn("O2", p["reason"])

    def test_unknown_planned_and_negated_are_not_completed_events(self):
        for status in ("planned", "negated", "unknown"):
            events = self.events()
            events[0] = {**events[0], "status": status}
            with self.subTest(status=status):
                self.assertFalse(self.execute(contract("harbor workshop", "gallery visit"), events)["executable"])

    def test_explicit_planned_query_can_use_planned_events(self):
        events = [{**a, "status": "planned"} for a in self.events()]
        self.assertEqual(self.execute(contract("harbor workshop", "gallery visit", status="planned"), events)["answer"], "5 days")

    def test_repeated_reports_do_not_create_false_conflict(self):
        events = self.events()
        self.assertEqual(self.execute(contract("harbor workshop", "gallery visit"), events + events)["answer"], "5 days")

    def test_distinct_occurrences_remain_ambiguous_without_selector(self):
        events = self.events() + self.events(left="2027-04-04")[:1]
        p = self.execute(contract("harbor workshop", "gallery visit"), events)
        self.assertEqual(p["status"], "CONFLICTING")

    def test_first_occurrence_is_selected_within_a_window(self):
        events = self.events()[:1] + self.events(left="2027-04-04")[:1]
        c = contract("harbor workshop", answer_kind="date", occurrence="earliest", window="April 2027", reference="none")
        self.assertEqual(self.execute(c, events)["answer"], "2027-04-02")

    def test_calendar_months_and_compound_units(self):
        events = self.events(left="2022-12-12", right="2027-08-12")
        c = contract("harbor workshop", "gallery visit", unit="year_month")
        self.assertEqual(self.execute(c, events)["answer"], "4 years and 8 months")

    def test_age_respects_birthday_not_just_year_number(self):
        events = self.events(left="1981-07-22", right="2027-07-01")
        self.assertEqual(self.execute(contract("harbor workshop", "gallery visit", unit="year"), events)["answer"], "45 years")

    def test_equal_dates_do_not_invent_event_order(self):
        c = contract("harbor workshop", "gallery visit", answer_kind="event", projection="event", relation="earliest")
        self.assertFalse(self.execute(c, self.events(right="2027-04-02"))["executable"])

    def test_order_returns_event_names_not_dates(self):
        c = contract("harbor workshop", "gallery visit", answer_kind="ordered_events", projection="event", relation="order")
        self.assertEqual(self.execute(c, self.events(left="2027-04-09"))["answer"], "gallery visit; harbor workshop")

    def test_who_met_projects_participant_not_first_person_actor(self):
        events = self.events()
        events[0]["argument"] = "Mira"
        events[1]["argument"] = "Omar"
        c = contract("harbor workshop", "gallery visit", answer_kind="actor", projection="argument", relation="earliest")
        self.assertEqual(self.execute(c, events)["answer"], "Mira")

    def test_unknown_participant_cannot_be_substituted_by_actor(self):
        c = contract("harbor workshop", "gallery visit", answer_kind="actor", projection="argument", relation="earliest")
        self.assertFalse(self.execute(c, self.events())["executable"])

    def test_date_only_evidence_cannot_claim_hour_precision(self):
        p = self.execute(contract("harbor workshop", "gallery visit", unit="hour"), self.events())
        self.assertFalse(p["executable"])

    def test_same_actor_chain_can_cross_reporters_without_changing_provenance(self):
        c = contract("harbor workshop", "gallery visit")
        for event in c["events"]:
            event["actor"] = "Mira"
        rows = [source("Mira completed the harbor workshop on 2027-04-02.", speaker="Omar"),
                source("Mira completed the gallery visit three days after the harbor workshop.",
                       speaker="Nila", turn="2", session="s2")]
        atoms = build_evidence_ledger(rows)
        assertions = [validate_assertion(dict(id=a.atom_id, **annotation(a.clause, actor="Mira")),
                                         a, c, resolve_temporal_expression) for a in atoms]
        p = self.execute(c, assertions)
        self.assertEqual(p["answer"], "3 days")
        self.assertEqual([b["source"]["source_speaker"] for b in p["bindings"]], ["Omar", "Nila"])


class TemporalIsolationTests(unittest.TestCase):
    def test_multisession_arithmetic_and_collections_do_not_activate(self):
        for operator in ("COUNT", "COUNT_DISTINCT", "SUM", "AVERAGE", "COLLECT", "UNION", "INTERSECTION", "ARGMAX", "ARGMIN", "DIFFERENCE"):
            plan = legacy_plan(operator, role="count", kinds=("quantity", "quantity"))
            with self.subTest(operator=operator):
                self.assertFalse(eligible_request(plan))

    def test_assistant_preference_and_inference_routes_remain_protected(self):
        for intent in ("assistant_memory", "preference_recommendation", "evidence_synthesis"):
            self.assertFalse(eligible_request(legacy_plan(), query_intent={"intent": intent}))

    def test_unary_measured_venue_selector_is_not_an_event_time_comparison(self):
        self.assertFalse(eligible_request(legacy_plan("ARGMAX", role="location", kinds=("event",))))
        self.assertTrue(eligible_request(legacy_plan("ARGMAX", role="location", kinds=("event", "event"))))

    def test_state_history_and_existing_typed_answers_remain_protected(self):
        self.assertFalse(eligible_request(legacy_plan(), protected=True))
        self.assertFalse(eligible_request(legacy_plan(), state_selector={"deterministic_enabled": True, "requested_attribute": "employer"}))

    def test_off_mode_performs_no_model_calls(self):
        ctx = FakeLocalModel({}, {})
        result = run_temporal_candidate(ctx, "question", legacy_plan(), [], mode="off", temporal_resolver=resolve_temporal_expression)
        self.assertFalse(result["applicable"])
        self.assertEqual(ctx.calls, [])

    def test_request_to_execution_is_separate_from_legacy_plan_and_evidence(self):
        text = "I completed the harbor workshop on 2027-04-02."
        rows = [source(text)]
        before = copy.deepcopy(rows)
        c = contract("harbor workshop", reference="question_time")
        ctx = FakeLocalModel(c, {text: annotation(text)})
        plan = legacy_plan()
        p = run_temporal_candidate(ctx, "How long since my harbor workshop?", plan, rows,
                                   temporal_resolver=resolve_temporal_expression, reference_timestamp="2027-04-15")
        self.assertEqual(p["proof"]["answer"], "13 days")
        self.assertEqual(rows, before)
        self.assertEqual(plan.operator, "DATE_LOOKUP")
        self.assertEqual(p["calls"], {"temporal_answer_contract": 1, "temporal_contract": 1,
                                     "temporal_constraint_presence": 1, "temporal_question_constraints_duration": 1,
                                     "temporal_question_constraints_projection": 1, "temporal_assertions": 1})
        json.dumps(p)

    def test_recall_preserves_existing_answer_without_assertion_calls(self):
        ctx = FakeLocalModel(contract("harbor workshop", task="recall"), {})
        p = run_temporal_candidate(ctx, "How long was my harbor workshop?", legacy_plan(), [], temporal_resolver=resolve_temporal_expression)
        self.assertEqual(p["status"], "preserved_recall_or_unknown")
        self.assertEqual(len(ctx.calls), 2)

    def test_bad_local_output_falls_back_with_bounded_calls(self):
        ctx = FakeLocalModel({}, {})
        p = run_temporal_candidate(ctx, "How long?", legacy_plan(), [], temporal_resolver=resolve_temporal_expression)
        self.assertEqual(p["status"], "preserved_recall_or_unknown")
        self.assertEqual(len(ctx.calls), 1)

    def test_shadow_or_incomplete_certificate_cannot_replace_a_baseline(self):
        p = {"mode": "validated", "proof": {"status": "COMPLETE", "executable": True,
             "answer": "5 days", "bindings": [{}], "certificate": {}}}
        self.assertFalse(promotable_temporal_result(p))
        keys = ("output_typed", "all_operands_bound", "actor_and_status_bound",
                "source_spans_grounded", "event_times_not_report_times", "question_constraints_checked")
        p["proof"]["certificate"] = dict.fromkeys(keys, True)
        self.assertTrue(promotable_temporal_result(p))
        p["mode"] = "shadow"
        self.assertFalse(promotable_temporal_result(p))

    def test_empty_invented_event_time_does_not_become_report_time(self):
        atom = build_evidence_ledger([source("I completed the harbor workshop some time earlier.")])[0]
        a = assertion(atom)
        p = execute_contract(contract("harbor workshop", reference="question_time"), [a], reference_timestamp="2027-04-25")
        self.assertFalse(p["executable"])

    def test_project_labels_and_names_are_not_fuzzy_merged(self):
        c = contract("project 1", reference="question_time")
        c["events"][0]["entities"] = ["project 1"]
        atom = build_evidence_ledger([source("I finished project 2 on 2027-04-02.")])[0]
        a = validate_assertion(dict(id=atom.atom_id, **annotation(atom.clause, predicate="finished")), atom, c, resolve_temporal_expression)
        self.assertEqual(a["matches"], [])

    def test_candidate_selection_is_bounded_and_covers_each_operand(self):
        ledger = build_evidence_ledger([source("I completed a harbor workshop.", turn=str(i), session=str(i)) for i in range(40)]
                                       + [source("I completed a gallery visit.", turn="g", session="g")])
        atoms = select_candidate_atoms(ledger, contract("harbor workshop", "gallery visit"))
        self.assertEqual(len(atoms), 24)
        self.assertIn("g", [a.source_turn_id for a in atoms])


if __name__ == "__main__":
    unittest.main()
