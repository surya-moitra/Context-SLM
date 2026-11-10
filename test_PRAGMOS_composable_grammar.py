import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PRAGMOS_composable_candidate import (
    _call, _grammar, _validate_native_grammar, extraction_schema, run_candidate,
    source_span_choices,
)
from PRAGMOS_composable_execution import ContractError, Source
from test_PRAGMOS_composable_candidate import context
from test_PRAGMOS_composable_execution import data


SPECIAL_SPANS = (
    'land. "When', 'a "quoted" label', r'a\b', 'two\\slashes',
    'line one\nline two', 'one\ttwo', 'one\rtwo', 'control\x01text',
    'caf\u00e9', '\u6771\u4eac', '\U0001f642', "owner's item", '"\\"',
)


def actor_rule(grammar):
    return next(line.split(" ::= ", 1)[1] for line in grammar._grammar.splitlines()
                if line.startswith("facts-item-actor-span ::= "))


class ComposableGrammarTests(unittest.TestCase):
    def test_every_special_span_retains_its_exact_json_value(self):
        for span in SPECIAL_SPANS:
            with self.subTest(span=span):
                grammar = _grammar("extractor", source_spans=(span,))
                json_literal = json.loads(actor_rule(grammar))
                self.assertEqual(json_literal, json.dumps(span))
                self.assertEqual(json.loads(json_literal), span)

    def test_generated_source_choices_retain_embedded_quotes(self):
        text = 'I said "the label"; the path is C:\\shared. I paid $23.'
        source = Source("S1", "s1", "t1", "user", "user", None, text)
        spans = source_span_choices([source])
        self.assertTrue(any('"' in span for span in spans))
        self.assertTrue(any('\\' in span for span in spans))
        grammar = _grammar("extractor", ("S1",), ("O1",), spans, ("$23",))
        self.assertEqual(source.text, text)
        self.assertIn(json.dumps(json.dumps('said "the label')), actor_rule(grammar))

    def test_simple_ascii_schema_is_byte_identical_to_existing_converter(self):
        from llama_cpp import LlamaGrammar
        arguments = (("CS1",), ("O1",), ("I", "paid", "ordinary words"), ("$23",))
        expected = LlamaGrammar.from_json_schema(json.dumps(extraction_schema(*arguments)), verbose=False)
        self.assertEqual(_grammar("extractor", *arguments)._grammar, expected._grammar)

    def test_does_not_patch_the_shared_library_converter(self):
        from llama_cpp.llama_grammar import SchemaConverter
        original = SchemaConverter._format_literal
        _grammar("extractor", source_spans=('a new "quoted" span',))
        self.assertIs(SchemaConverter._format_literal, original)

    def test_conversion_failure_is_not_unconstrained_generation(self):
        _grammar.cache_clear()
        sources, _ = data([17])
        ctx = context([])
        with patch("llama_cpp.llama_grammar.SchemaConverter.resolve_refs", side_effect=ValueError("bad schema")):
            result = run_candidate(ctx, "What total?", sources, n_ctx=2048)
        self.assertEqual(result["reason"], "planner_grammar_build_failed:ValueError")
        self.assertEqual(ctx.llm.calls, [])
        self.assertTrue(ctx.llm.restored)

    def test_native_rejection_prevents_any_generation_and_restores_state(self):
        from llama_cpp import LlamaGrammar
        sources, _ = data([17])
        ctx = context([])
        ctx.llm._model = SimpleNamespace(vocab=object())
        with patch("PRAGMOS_composable_candidate._grammar", return_value=LlamaGrammar.from_string('root ::= "x"')), \
             patch("llama_cpp.llama_cpp.llama_sampler_init_grammar", return_value=None), \
             patch("llama_cpp.llama_cpp.llama_sampler_free") as free:
            result = run_candidate(ctx, "What total?", sources, n_ctx=2048)
        self.assertEqual(result["reason"], "planner_native_grammar_rejected")
        self.assertEqual(result["calls"]["planner_calls"], 0)
        self.assertEqual(result["calls"]["grammar_checks"], [{"stage": "planner", "accepted": False}])
        self.assertEqual(ctx.llm.calls, [])
        self.assertTrue(ctx.llm.restored)
        free.assert_not_called()

    def test_native_validation_sampler_is_freed_before_generation(self):
        from llama_cpp import LlamaGrammar
        ctx = context([{"ok": True}])
        ctx.llm._model = SimpleNamespace(vocab=object())
        token = object()
        audit = {"planner_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        with patch("PRAGMOS_composable_candidate._grammar", return_value=LlamaGrammar.from_string('root ::= "x"')), \
             patch("llama_cpp.llama_cpp.llama_sampler_init_grammar", return_value=token), \
             patch("llama_cpp.llama_cpp.llama_sampler_free") as free:
            value = _call(ctx, "fixture", "planner", 2048, audit)
        self.assertEqual(value, {"ok": True})
        free.assert_called_once_with(token)
        self.assertEqual(audit["planner_calls"], 1)
        self.assertTrue(audit["grammar_checks"][0]["accepted"])

    def test_native_validation_cannot_be_assumed_without_model_vocabulary(self):
        with self.assertRaisesRegex(ContractError, "native_grammar_validation_unavailable"):
            _validate_native_grammar(context([]), None, "extractor", {})


class ComposableNativeGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("PRAGMOS_GRAMMAR_TEST_MODEL")
        if not path:
            raise unittest.SkipTest("Set PRAGMOS_GRAMMAR_TEST_MODEL for native vocabulary-only grammar checks")
        if not Path(path).is_file():
            raise AssertionError("Native grammar fixture model is missing")
        from llama_cpp import llama_cpp
        from llama_cpp._internals import LlamaModel
        params = llama_cpp.llama_model_default_params()
        params.vocab_only = True
        params.n_gpu_layers = 0
        cls.model = LlamaModel(path_model=path, params=params, verbose=False)
        cls.addClassCleanup(cls.model.close)

    def accept(self, grammar):
        audit = {}
        ctx = SimpleNamespace(llm=SimpleNamespace(_model=self.model))
        _validate_native_grammar(ctx, grammar, "fixture", audit)
        self.assertEqual(audit["grammar_checks"], [{"stage": "fixture", "accepted": True}])

    def test_native_parser_accepts_all_escaped_source_literals(self):
        for span in SPECIAL_SPANS:
            with self.subTest(span=span):
                self.accept(_grammar("extractor", ("CS1",), ("O1",), (span,)))

    def test_native_parser_accepts_generated_source_choices(self):
        text = 'I used "a label" at C:\\shared.\nThen I paid $23; I did not spend $45.'
        source = Source("CS1", "s1", "t1", "user", "user", None, text)
        self.accept(_grammar("extractor", ("CS1",), ("O1",), source_span_choices([source]), ("$23", "$45")))

    def test_native_parser_accepts_deterministic_mixed_character_corpus(self):
        rng = random.Random(923)
        alphabet = 'abcXYZ01 "\\\n\t\r/<>|()[]{}' + '\u00e9\u6771\U0001f642'
        spans = tuple(''.join(rng.choice(alphabet) for _ in range(24)) for _ in range(64))
        self.accept(_grammar("extractor", ("CS1",), ("O1",), spans))

    def test_native_parser_accepts_planner_output_contract(self):
        self.accept(_grammar("planner", answer_fields=("value", "left.value", "right.value")))


if __name__ == "__main__":
    unittest.main()
