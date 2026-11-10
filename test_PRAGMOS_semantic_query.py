import copy
import json
import unittest
import sys
from types import SimpleNamespace
from unittest import mock

from PRAGMOS_benchmark_LongMemEval import resolve_temporal_expression
from PRAGMOS_semantic_query import (
    _bind_relative_events, _node, _review_recounts, _support_events, candidate_allowed, discover_windows,
    enumerate_queries, lower_query, matches_answer_contract, run_query_candidate, run_temporal_query, select_query,
    shortlist_queries,
)
from PRAGMOS_temporal_contracts import promotable_temporal_result
from PRAGMOS_evidence_compiler import build_evidence_ledger


def selector(key, text, actor="user", entities=()):
    return dict(id=key, text=text, actor=actor, entities=list(entities),
                phase="any", status="completed", collection=False)


def source(text, key="1", speaker="user", role="user", timestamp="2031-06-20"):
    return dict(source_quote=text, source_turn_id=key, source_session_id="s" + key,
                source_speaker=speaker, source_role=role, source_timestamp=timestamp)


def scan(key):
    return _node("scan", operand=key)


class QueryEnumerationTests(unittest.TestCase):
    def test_six_event_order_is_representable_without_tree_depth_limit(self):
        q = "Order the kiln, loom, lathe, easel, press and oven activities during June 2031."
        specs = [selector("E" + str(i), word) for i, word in enumerate(
            ["kiln", "loom", "lathe", "easel", "press", "oven"], 1)]
        expression = _node("project", _node("order", _node("window", _node("union", *[
            scan(s["id"]) for s in specs]), span="June 2031")), field="event")
        plan = lower_query(expression, specs, q)
        self.assertEqual(plan["answer_kind"], "list")
        self.assertEqual(len([n for n in plan["steps"] if n["op"] == "scan"]), 6)
        self.assertLess(len(plan["steps"]), 32)

    def test_named_operands_cannot_be_dropped(self):
        specs = [selector("E1", "kiln"), selector("E2", "loom")]
        with self.assertRaisesRegex(ValueError, "drops_required"):
            lower_query(_node("project", _node("earliest", scan("E1")), field="event"),
                        specs, "Which came first, kiln or loom?")

    def test_unary_duration_uses_question_clock_not_date_projection(self):
        specs = [selector("E1", "kiln")]
        plans = enumerate_queries(specs, "How many months ago did I finish the kiln?")
        durations = [p for p in plans if p["answer_kind"] == "duration"
                     and not any(s.get("intrinsic") for s in p["operands"])]
        self.assertTrue(durations)
        self.assertTrue(all(p["steps"][-1]["unit"] == "month" for p in durations))
        self.assertTrue(all(len(p["steps"][-1]["args"]) == 1 for p in durations))
        self.assertTrue(all("question is asked" in p["readback"] for p in durations))

    def test_before_and_after_both_operand_directions_are_proposals(self):
        specs = [selector("E1", "activities"), selector("E2", "induction")]
        plans = enumerate_queries(specs, "Which activities happened after induction?")
        filters = [n for p in plans for n in p["steps"] if n["op"] in {"before", "after"}]
        self.assertEqual({n["op"] for n in filters}, {"before", "after"})

    def test_counts_frequency_and_companion_share_generic_compositions(self):
        specs = [selector("E1", "visit")]
        plans = enumerate_queries(specs, "Which visit in June 2031?", [("June 2031", {})])
        roots = {p["steps"][-1]["op"] for p in plans}
        self.assertTrue({"count", "argmax", "project"} <= roots)
        self.assertTrue(any(p["steps"][-1].get("field") == "person" for p in plans))

    def test_windows_only_come_from_resolvable_calendar_evidence(self):
        question = "How many days passed between installing the lamp and ordering the shades?"
        self.assertEqual(discover_windows(question, resolve_temporal_expression, "2031-06-20"), [])

    def test_range_endpoints_get_one_interval_proposal(self):
        q = "Which venue did I visit between 3 June 2031 and 20 June 2031?"
        windows = discover_windows(q, resolve_temporal_expression, "2031-06-30")
        self.assertTrue(any(v["interval_start"] == "2031-06-03" and
                            v["interval_end"] == "2031-06-20" for _, v in windows))

    def test_readback_does_not_duplicate_entities_or_include_answers(self):
        specs = [selector("E1", "repair kiln", entities=["kiln"])]
        p = lower_query(_node("difference", scan("E1"), unit="day"), specs,
                        "How many days ago did I repair kiln?")
        self.assertNotIn("kiln kiln", p["readback"])
        self.assertNotIn("answer", p)

    def test_duration_can_follow_a_filtered_occurrence_selection(self):
        specs = [selector("E1", "class"), selector("E2", "orientation")]
        plans = enumerate_queries(specs, "How many days ago was my first class after orientation?")
        target = lower_query(_node("difference", _node("earliest", _node("after", scan("E1"), scan("E2"))),
                                   unit="day"), specs,
                             "How many days ago was my first class after orientation?")
        self.assertTrue(any(p["steps"] == target["steps"] for p in plans))

    def test_plain_factual_lookup_cannot_become_executable_temporal_work(self):
        with self.assertRaisesRegex(ValueError, "time_constraint"):
            lower_query(_node("project", scan("E1"), field="object"),
                        [selector("E1", "kiln")], "What kind of kiln do I own?")

    def test_candidate_pool_keeps_full_order_and_first_last_distinct(self):
        q = "Order kiln, loom and lathe."
        plans = enumerate_queries([selector("E1", "kiln"), selector("E2", "loom"), selector("E3", "lathe")], q)
        events = [p for p in plans if p["steps"][-1].get("field") == "event"]
        selected = shortlist_queries(None, q, events, limit=24)
        for op in ("earliest", "latest", "order"):
            self.assertTrue(any(any(n["op"] == op for n in p["steps"])
                and not any(n["op"] in {"before", "after", "window"} for n in p["steps"])
                for p in selected), op)

    def test_readback_preserves_named_and_first_person_actor(self):
        for actor in ("user", "Mina"):
            q = "How old was " + ("I" if actor == "user" else actor) + " at graduation?"
            p = next(p for p in enumerate_queries([selector("E1", "graduation", actor=actor)], q)
                     if any(s.get("intrinsic") for s in p["operands"]))
            self.assertIn("How old was " + ("I" if actor == "user" else actor), p["readback"])


class ControlledModel:
    def __init__(self, specs, prefer, support, answer_form="elapsed time or age"):
        self.specs, self.prefer, self.support = specs, prefer, support
        self.answer_form = answer_form

    def __call__(self, context, prompt, schema, key, diagnostics, **kwargs):
        diagnostics["calls"][key] = diagnostics["calls"].get(key, 0) + 1
        data = json.loads(prompt.split("<|user|>\n", 1)[1].split("<|end|>", 1)[0])
        if key == "semantic_answer_contract":
            return dict(request="copied request meaning", answer_form=self.answer_form)
        if key == "semantic_query_selectors":
            return dict(events=[{k: s[k] for k in ("text", "actor", "entities")} for s in self.specs])
        if key == "semantic_query_equivalence":
            return dict(reason="Same requested information", equivalent=True)
        if key.startswith("semantic_query_"):
            matches = [v for v in data["alternatives"] if self.prefer(v)]
            return dict(request="copied request meaning", meaning=matches[0] if matches else "No alternative asks for the same information.")
        if key == "semantic_event_support":
            supported = self.support(data)
            spec = next((s for s in self.specs if s["text"] == data["target"]["activity"]
                        and s["actor"] == data["target"]["performer"]), dict(id="BIRTH"))
            status = supported[spec["id"]]
            return dict(choice={"actual": "A", "denied": "B"}.get(status, "C"))
        raise AssertionError(key)


class CandidateIsolationTests(unittest.TestCase):
    def test_both_adapters_require_explicit_opt_in(self):
        from PRAGMOS_benchmark_LongMemEval import parse_args as lme_args
        from PRAGMOS_benchmark_LoCoMo import parse_args as locomo_args
        for parse in (lme_args, locomo_args):
            with self.subTest(adapter=parse.__module__):
                with mock.patch.object(sys, "argv", ["benchmark", "--data-file", "unused.json"]):
                    args = parse()
                self.assertEqual(args.pragmos_temporal_engine, "contracts")
                self.assertEqual(args.pragmos_temporal_contract_mode, "off")
                with mock.patch.object(sys, "argv", ["benchmark", "--data-file", "unused.json",
                        "--pragmos-temporal-engine", "semantic_query",
                        "--pragmos-temporal-contract-mode", "shadow"]):
                    args = parse()
                self.assertEqual(args.pragmos_temporal_engine, "semantic_query")
                self.assertEqual(args.pragmos_temporal_contract_mode, "shadow")

    def test_protected_contracts_do_not_call_model(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("unexpected model call")
        for config in [dict(mode="off"), dict(protected=True),
                       dict(query_intent=dict(intent="assistant_memory")),
                       dict(query_intent=dict(intent="preference_recommendation")),
                       dict(query_intent=dict(intent="evidence_synthesis")),
                       dict(state_selector=dict(deterministic_enabled=True, requested_attribute="employer"))]:
            with self.subTest(config=config):
                r = run_query_candidate(None, "question", [], temporal_resolver=resolve_temporal_expression,
                                        call=forbidden, **config)
                self.assertEqual(r["calls"], {})
                self.assertFalse(promotable_temporal_result(r))

    def test_bad_selector_is_not_silently_repaired_with_invented_entities(self):
        def bad(context, prompt, schema, key, diagnostics, **kwargs):
            return dict(events=[dict(text="unmentioned", actor="user", entities=[])])
        r = run_query_candidate(None, "When did I repair the kiln?", [],
                                temporal_resolver=resolve_temporal_expression, call=bad)
        self.assertEqual(r["status"], "EXTRACTION_FAILED")
        self.assertEqual(r["fallback_action"], "preserve")

    def test_selection_order_disagreement_cannot_promote(self):
        calls = []
        def disagree(context, prompt, schema, key, diagnostics, **kwargs):
            calls.append(key)
            data = json.loads(prompt.split("<|user|>\n", 1)[1].split("<|end|>", 1)[0])
            return dict(request="meaning", meaning=data["alternatives"][0])
        with self.assertRaisesRegex(ValueError, "uncertain"):
            select_query(None, "Which happened first?", [dict(readback="earliest"), dict(readback="latest")],
                         disagree, dict(calls={}))
        self.assertEqual(calls, ["semantic_query_select", "semantic_query_permutation_check"])

    def test_copied_sources_and_baseline_objects_are_not_mutated(self):
        rows = [source("I repaired the kiln on 3 June 2031.")]
        before = copy.deepcopy(rows)
        fake = ControlledModel([selector("E1", "repaired the kiln", entities=["kiln"])],
                               lambda text: "number of days" in text and "happened" not in text,
                               lambda data: dict(E1="actual"))
        result = run_query_candidate(None, "How many days ago did I repair the kiln?", rows,
                                     temporal_resolver=resolve_temporal_expression,
                                     reference_timestamp="2031-06-20", call=fake)
        self.assertEqual(rows, before)
        # The fake selector uses source tense instead of a copied question span.
        self.assertEqual(result["status"], "EXTRACTION_FAILED")

    def test_elapsed_answer_uses_supported_event_time_not_reporting_date(self):
        specs = [selector("E1", "repair the kiln", entities=["kiln"])]
        raw = [source("I repaired the kiln on 3 June 2031.", timestamp="2031-06-10")]
        before = copy.deepcopy(raw)
        fake = ControlledModel(specs, lambda text: text.startswith("How many days have passed since 'repair")
                               and "one that" not in text, lambda data: dict(E1="actual"))
        r = run_query_candidate(None, "How many days ago did I repair the kiln?", raw,
                                temporal_resolver=resolve_temporal_expression,
                                reference_timestamp="2031-06-20", call=fake)
        self.assertEqual((r["proof"] or {}).get("answer"), "17 days")
        self.assertTrue(promotable_temporal_result(r))
        self.assertEqual(raw, before)
        json.dumps(r)

    def test_another_performers_plan_cannot_supply_the_event_date(self):
        specs = [selector("E1", "repair the kiln", entities=["kiln"])]
        raw = [source("Nora considered repairing the kiln, but I did not do that.", "1"),
               source("I repaired the kiln on 3 June 2031.", "2")]
        fake = ControlledModel(specs, lambda text: text.startswith("How many days have passed since 'repair")
                               and "one that" not in text,
                               lambda data: dict(E1="unmentioned" if "Nora" in data["source"] else "actual"))
        r = run_query_candidate(None, "How many days ago did I repair the kiln?", raw,
                                temporal_resolver=resolve_temporal_expression,
                                reference_timestamp="2031-06-20", call=fake)
        self.assertEqual(r["proof"]["answer"], "17 days")
        self.assertEqual(len(r["proof"]["bindings"]), 1)

    def test_missing_evidence_does_not_become_an_absence_certificate(self):
        fake = ControlledModel([selector("E1", "repair the kiln")],
                               lambda text: text.startswith("How many days have passed since 'repair")
                               and "one that" not in text, lambda data: dict(E1="unmentioned"))
        r = run_query_candidate(None, "How many days ago did I repair the kiln?", [source("I bought a mug.")],
                                temporal_resolver=resolve_temporal_expression,
                                reference_timestamp="2031-06-20", call=fake)
        self.assertEqual(r["status"], "PARTIAL")
        self.assertEqual(r["fallback_action"], "preserve")
        self.assertFalse(promotable_temporal_result(r))

    def test_undated_event_is_not_assigned_its_reporting_date(self):
        fake = ControlledModel([selector("E1", "repair the kiln")],
                               lambda text: text.startswith("How many days have passed since 'repair")
                               and "one that" not in text, lambda data: dict(E1="actual"))
        r = run_query_candidate(None, "How many days ago did I repair the kiln?", [source("I repaired the kiln.")],
                                temporal_resolver=resolve_temporal_expression,
                                reference_timestamp="2031-06-20", call=fake)
        self.assertNotEqual(r["status"], "COMPLETE")
        self.assertFalse(promotable_temporal_result(r))

    def test_parser_failure_does_not_manufacture_abstention(self):
        def broken(*args, **kwargs):
            raise ValueError("model failed")
        r = run_query_candidate(None, "When did I repair the kiln?", [],
                                temporal_resolver=resolve_temporal_expression, call=broken)
        self.assertEqual(r["fallback_action"], "preserve")
        self.assertIsNone(r["proof"])

    def test_output_type_is_checked_independently_of_executable_arithmetic(self):
        specs = [selector("E1", "repair the kiln")]
        p = lower_query(_node("difference", scan("E1"), unit="day"), specs,
                        "How many days ago did I repair the kiln?")
        self.assertTrue(matches_answer_contract(p, dict(field="duration")))
        self.assertFalse(matches_answer_contract(p, dict(field="time")))
        self.assertFalse(matches_answer_contract(p, dict(field="event")))

    def test_old_misclassified_plan_is_not_an_eligibility_gate(self):
        self.assertTrue(candidate_allowed(dict(operator="NONE")))
        self.assertTrue(candidate_allowed(None))

    def test_unrestored_sources_do_not_call_model(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("must not inspect unresolved sources")
        result = run_temporal_query(None, "When did I repair the kiln?", None, [],
            temporal_resolver=resolve_temporal_expression, baseline_evidence_complete=False, call=forbidden)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["calls"], {})
        self.assertFalse(promotable_temporal_result(result))

    def test_shadow_proof_cannot_promote(self):
        fake = ControlledModel([selector("E1", "repair the kiln")],
            lambda text: text.startswith("How many days have passed since 'repair") and "one that" not in text,
            lambda data: dict(E1="actual"))
        r = run_query_candidate(None, "How many days ago did I repair the kiln?",
            [source("I repaired the kiln on 3 June 2031.")], mode="shadow",
            temporal_resolver=resolve_temporal_expression, reference_timestamp="2031-06-20", call=fake)
        self.assertEqual(r["proof"]["answer"], "17 days")
        self.assertFalse(promotable_temporal_result(r))

    def test_population_count_needs_complete_evidence(self):
        fake = ControlledModel([selector("E1", "visit", entities=["kiln"])],
            lambda text: text.startswith("How many 'visit") and "distinct" not in text,
            lambda data: dict(E1="actual"), answer_form="number of occurrences or distinct values")
        r = run_query_candidate(None, "How many times did I visit the kiln during June 2031?",
            [source("I visited the kiln on 3 June 2031.")],
            temporal_resolver=resolve_temporal_expression, reference_timestamp="2031-06-20", call=fake)
        self.assertEqual(r["reason"], "scoped_population_not_complete")
        self.assertFalse(promotable_temporal_result(r))

    def test_model_budget_exhaustion_retains_baseline_not_zero_or_abstention(self):
        def exhaust(context, prompt, schema, key, diagnostics, **kwargs):
            diagnostics["calls"][key] = 64
            return dict(answer_form="elapsed time or age", request="elapsed time")
        r = run_query_candidate(None, "How many days ago did I repair the kiln?", [],
            temporal_resolver=resolve_temporal_expression, call=exhaust)
        self.assertEqual(r["reason"], "semantic_query_call_budget_exhausted")
        self.assertEqual(r["fallback_action"], "preserve")
        self.assertFalse(promotable_temporal_result(r))

    def test_non_equivalent_query_is_rejected_before_source_review(self):
        base = ControlledModel([selector("E1", "repair the kiln")],
            lambda text: text.startswith("How many days have passed since 'repair") and "one that" not in text,
            lambda data: self.fail("source must not be inspected"))
        def checked(context, prompt, schema, key, diagnostics, **kwargs):
            if key == "semantic_query_equivalence":
                return dict(reason="Different requested information", equivalent=False)
            return base(context, prompt, schema, key, diagnostics, **kwargs)
        r = run_query_candidate(None, "How many days ago did I repair the kiln?", [],
            temporal_resolver=resolve_temporal_expression, call=checked)
        self.assertEqual(r["reason"], "selected_query_not_semantically_equivalent")
        self.assertFalse(promotable_temporal_result(r))

    def test_entity_substring_is_not_actor_or_entity_support(self):
        atoms = build_evidence_ledger([source("I met Annette on 3 June 2031.")])
        def forbidden(*args, **kwargs):
            self.fail("An unrelated entity should not be promoted to a semantic match")
        result = _support_events(None, atoms, [selector("E1", "meet", entities=["Ann"])],
                                 forbidden, dict(calls={}, resolver=resolve_temporal_expression))
        self.assertEqual(result, [])

    def test_birth_after_the_event_cannot_become_a_positive_age(self):
        fake = ControlledModel([selector("E1", "completed the degree", entities=["degree"])],
            lambda text: text.startswith("How old was"),
            lambda data: dict(E1="actual" if "degree" in data["source"] else "unmentioned",
                              BIRTH="actual" if "born" in data["source"] else "unmentioned"))
        # Isolate the directional arithmetic guard from lexical/dense ranking.
        with mock.patch("PRAGMOS_semantic_query.select_candidate_atoms", side_effect=lambda ledger, *a, **kw: ledger):
            r = run_query_candidate(None, "How old was I when I completed the degree?",
                [source("I completed the degree on 3 June 2031."), source("I was born on 1 January 2032.", "2")],
                temporal_resolver=resolve_temporal_expression, reference_timestamp="2033-01-01", call=fake)
        self.assertEqual(r["reason"], "birth_reference_after_event")
        self.assertFalse(promotable_temporal_result(r))

    def test_no_evidence_of_birth_cannot_be_invented_for_age(self):
        fake = ControlledModel([selector("E1", "completed the degree", entities=["degree"])],
            lambda text: text.startswith("How old was"),
            lambda data: dict(E1="actual", BIRTH="unmentioned"))
        r = run_query_candidate(None, "How old was I when I completed the degree?",
            [source("I completed the degree on 3 June 2031.")],
            temporal_resolver=resolve_temporal_expression, reference_timestamp="2033-01-01", call=fake)
        self.assertFalse(promotable_temporal_result(r))
        self.assertEqual(r["proof"]["obligations"]["BIRTH"]["event_identity"], "not_found_in_reviewed_evidence")

    def test_recount_requires_grounded_link_and_keeps_provenance(self):
        def events():
            return [dict(actor="user", matches=["E1"], time=dict(value="2031-06-03"),
                         event="visited kiln", atom=SimpleNamespace(clause="I visited the kiln.")),
                    dict(actor="user", matches=["E1"], time=dict(value="2031-06-03"),
                         event="attended kiln", atom=SimpleNamespace(clause="That same visit was enjoyable."))]
        def review(same, quote):
            return lambda *a, **kw: dict(same=same, source="B", link_quote=quote)
        for same, quote in [(False, None), (True, "invented link")]:
            rows = events()
            _review_recounts(None, rows, review(same, quote), dict(calls={}))
            self.assertNotEqual(rows[0]["event"], rows[1]["event"])
            self.assertNotIn("recount", rows[1])
        rows = events()
        _review_recounts(None, rows, review(True, "That same visit"), dict(calls={}))
        self.assertEqual(rows[0]["event"], rows[1]["event"])
        self.assertEqual(rows[1]["recount_provenance"]["quote"], "That same visit was enjoyable.")


if __name__ == "__main__":
    unittest.main()
