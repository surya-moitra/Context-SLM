import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PRAGMOS_benchmark_LongMemEval import resolve_temporal_expression
from PRAGMOS_semantic_checker import semantic_model_identity
from PRAGMOS_semantic_query import run_query_candidate
from PRAGMOS_temporal_contracts import promotable_temporal_result
from test_PRAGMOS_semantic_query import ControlledModel, selector, source


class Checker:
    identity = dict(kind="test_checker_not_actual_model")

    def __init__(self, matches):
        self.matches = matches

    def rank(self, question, candidates):
        results = [dict(plan=p, score=.95 if self.matches(p) else .01,
                        forward_entailment=.95, reverse_entailment=.95) for p in candidates]
        return sorted(results, key=lambda r: -r["score"])


def elapsed(p):
    return p["readback"].startswith("How many days have passed since 'repair") and "one that" not in p["readback"]


class SemanticCheckerIsolationTests(unittest.TestCase):
    def model(self):
        # Deliberately wrong type proposal: it must not discard a valid query.
        return ControlledModel([selector("E1", "repair the kiln", entities=["kiln"])],
            lambda text: False, lambda data: dict(E1="actual"), answer_form="calendar date")

    def run_candidate(self, checker, **options):
        return run_query_candidate(None, "How many days ago did I repair the kiln?",
            [source("I repaired the kiln on 3 June 2031.")],
            temporal_resolver=resolve_temporal_expression, reference_timestamp="2031-06-20",
            semantic_checker=checker, call=self.model(), **options)

    def test_checker_replaces_self_judging_not_arithmetic(self):
        r = self.run_candidate(Checker(elapsed))
        self.assertEqual(r["proof"]["answer"], "17 days")
        self.assertTrue(promotable_temporal_result(r))
        self.assertNotIn("semantic_answer_contract", r["calls"])
        self.assertNotIn("semantic_query_equivalence", r["calls"])
        self.assertEqual(r["proof"]["semantic_validation_basis"], "bidirectional_local_nli")

    def test_semantic_score_alone_cannot_prove_missing_source_time(self):
        fake = self.model()
        r = run_query_candidate(None, "How many days ago did I repair the kiln?",
            [source("I repaired the kiln.")], temporal_resolver=resolve_temporal_expression,
            reference_timestamp="2031-06-20", semantic_checker=Checker(elapsed), call=fake)
        self.assertFalse(promotable_temporal_result(r))

    def test_disagreeing_typed_answers_keep_baseline(self):
        checker = Checker(lambda p: elapsed(p) or (p["answer_kind"] == "date"
            and not any(n["op"] == "latest" for n in p["steps"])))
        r = self.run_candidate(checker)
        self.assertEqual(r["reason"], "plausible_interpretations_disagree")
        self.assertFalse(promotable_temporal_result(r))

    def test_unresolved_interpretation_cannot_be_dropped_for_easier_answer(self):
        r = self.run_candidate(Checker(lambda p: elapsed(p) or any(s.get("intrinsic") for s in p["operands"])))
        self.assertEqual(r["reason"], "plausible_interpretation_unresolved")
        self.assertFalse(promotable_temporal_result(r))

    def test_shadow_does_not_promote_nli_result(self):
        r = self.run_candidate(Checker(elapsed), mode="shadow")
        self.assertEqual(r["proof"]["answer"], "17 days")
        self.assertFalse(promotable_temporal_result(r))

    def test_low_agreement_is_not_an_absence_certificate(self):
        r = self.run_candidate(Checker(lambda p: False))
        self.assertEqual(r["reason"], "semantic_agreement_below_threshold")
        self.assertEqual(r["fallback_action"], "preserve")
        self.assertFalse(promotable_temporal_result(r))

    def test_model_is_never_loaded_on_protected_or_off_path(self):
        with mock.patch("PRAGMOS_semantic_query.load_semantic_checker", side_effect=AssertionError("load")):
            for options in (dict(mode="off"), dict(protected=True),
                            dict(query_intent=dict(intent="assistant_memory")),
                            dict(query_intent=dict(intent="preference_recommendation"))):
                r = run_query_candidate(None, "question", [], temporal_resolver=resolve_temporal_expression,
                                        semantic_checker_path="unused", **options)
                self.assertEqual(r["calls"], {})


class ModelIdentityTests(unittest.TestCase):
    def test_missing_local_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "requires_local"):
                semantic_model_identity(folder)

    def test_tokenizer_and_weight_changes_change_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (root / name).write_text("test fixture, not model weights")
            first = semantic_model_identity(folder)
            (root / "tokenizer.json").write_text("changed test fixture")
            second = semantic_model_identity(folder)
            self.assertNotEqual(first["fingerprint"], second["fingerprint"])
            self.assertFalse(first["trust_remote_code"])
            self.assertTrue(first["local_files_only"])
            self.assertEqual(first["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
