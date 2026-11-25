import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PRAGMOS_factual_read import locate_read_value, proof_scope_admission, read_operation_obligations, read_sources, run_factual_read, validate_read
from PRAGMOS_evidence_compiler import parse_semantic_plan


def source(text, **extra):
    return {"source_id": "R1", "text": text, "role": "user", "speaker": "user",
            "session_id": "s", "turn_id": 1, "timestamp": "2030-04-12", **extra}


def payload(answer, **extra):
    return {"mode": "read", "source_id": "R1", "answer": answer, **extra}


def validate(question, text, answer, *, anchors=lambda text: True):
    return validate_read(payload(answer), question, [source(text)], anchors_supported=anchors,
                         count_tokens=lambda s: len(s.split()))


class LiteralReadTests(unittest.TestCase):
    def test_explicit_cardinality_is_a_value_not_a_count_of_assertions(self):
        r = validate("How many workshops did I complete?", "I completed 17 workshops.", "17")
        self.assertEqual(r["answer"], "17")
        self.assertFalse(r["certificate"]["semantic_certification"])

    def test_measurement_unit_is_preserved_as_original_source_span(self):
        r = validate("How long did the dough rise?", "The dough rose for 87 minutes.", "87")
        self.assertEqual(r["answer"], "87 minutes")

    def test_memory_property_does_not_sum_other_measurements(self):
        r = validate("How much memory does my desktop have?", "My desktop has 32 GB RAM and 2 TB storage.", "32 GB")
        self.assertEqual(r["answer"], "32 GB")

    def test_quantity_projection_discards_only_requested_descriptor(self):
        q = "How many sessions did I complete?"
        s = [source("I completed 17 separate workshops.")]
        self.assertEqual(locate_read_value(q, "17 sessions", s, lambda t: True)[0][1], "17")
        self.assertEqual(locate_read_value(q, "17 unmentioned", s, lambda t: True), [])

    def test_quantity_projection_does_not_change_units(self):
        self.assertEqual(locate_read_value("What was the duration?", "1 hour",
                                          [source("It lasted 60 minutes.")], lambda t: True), [])

    def test_surname_value_and_descriptor_are_not_interchangeable(self):
        q, s = "What was my previous surname?", "My previous surname was Valdez."
        self.assertEqual(validate(q, s, "Valdez")["status"], "COMPLETE")
        self.assertNotEqual(validate(q, s, "previous surname")["status"], "COMPLETE")

    def test_computed_answer_not_present_in_source_cannot_be_read(self):
        self.assertNotEqual(validate("What was my total spending?", "I spent $13 and $21.", "$34")["status"], "COMPLETE")

    def test_derive_and_uncertain_do_not_certify_absence(self):
        for mode in ("derive", "unresolved"):
            r = validate_read(payload("", mode=mode, source_id=""), "What is my total?", [],
                              anchors_supported=lambda s: True, count_tokens=len)
            self.assertEqual(r["status"], "NOT_PROVEN")
            self.assertIsNone(r["answer"])

    def test_wrong_owner_and_wrong_object_fail_local_scope_obligations(self):
        for q, s, a, owner in [
            ("What did my uncle give me?", "My sister gave me a mixer.", "mixer", "uncle"),
            ("What is my pet's name?", "The firm's name is Northstar.", "Northstar", "pet"),
        ]:
            with self.subTest(q=q):
                self.assertNotEqual(validate(q, s, a, anchors=lambda t: owner in t)["status"], "COMPLETE")

    def test_answer_cannot_be_part_of_a_larger_word(self):
        self.assertNotEqual(validate("What was the title?", "The title was Seventeen.", "Seven")["status"], "COMPLETE")

    def test_repeated_value_is_not_given_invented_offsets(self):
        self.assertNotEqual(validate("What is the title?", "The title Pearl replaced Pearl.", "Pearl")["status"], "COMPLETE")

    def test_measurement_cannot_be_converted_by_reader(self):
        self.assertNotEqual(validate("How long was the journey?", "The journey took one hour.", "60 minutes")["status"], "COMPLETE")

    def test_newline_cannot_certify_a_prefix_of_a_multiline_value(self):
        self.assertNotEqual(validate('What is my boat named?', 'My boat is named Solstice\nDawn.', 'Solstice')['status'], 'COMPLETE')

    def test_certificate_offsets_address_original_statement(self):
        r = validate("What is my boat named?", "My boat is named Solstice.", "Solstice")
        c = r["certificate"]
        a, b = c["value"]["span"]
        self.assertEqual(c["assertion"][a:b], r["answer"])

    def test_derived_scope_can_span_multiple_sources_without_numeric_members(self):
        p = {"provenance": [{"source": {"text": "I attend ceramics on Tuesday."}},
                             {"source": {"text": "I attend ceramics on Friday."}}]}
        result = proof_scope_admission(p, [{"text": "ceramics"}], lambda g, s: g["text"] in s)
        self.assertTrue(result["accepted"])

    def test_scope_admission_handles_frame_and_typed_provenance(self):
        group = [{"text": "uncle"}]
        for p in ({"calculation": {"provenance": [{"source_quote": "My sister gave me a mixer."}]}},
                  {"provenance": [{"source": {"text": "My sister gave me a mixer."}}]}):
            self.assertFalse(proof_scope_admission(p, group, lambda g, s: g["text"] in s)["accepted"])

    def test_source_identity_and_role_are_checked_before_restoration(self):
        turn = SimpleNamespace(text="My boat is named Solstice.", role="user", speaker="user",
                               session_id="s", timestamp="2030-04-12", turn_id=1)
        row = {"source_turn_ids": [1], "source_session_id": "s", "source_quote": turn.text,
               "role": "user", "speaker": "user", "timestamp": turn.timestamp}
        self.assertEqual(read_sources([row], {1: turn}, ["user"])[0]["text"], turn.text)
        self.assertEqual(read_sources([row], {1: turn}, ["assistant"]), [])
        self.assertEqual(read_sources([{**row, "source_session_id": "other"}], {1: turn}, ["user"]), [])

    def test_a_read_cannot_bypass_a_missing_mandatory_cost(self):
        q = 'What is the total price of the pass and the shuttle?'
        plan = parse_semantic_plan({'operator': 'SUM', 'answer_type': 'amount', 'answer_role': 'amount',
            'actor': 'user', 'confidence': 'high', 'operands': [
                {'id': 'A', 'text': 'pass', 'kind': 'quantity', 'hard_entities': []},
                {'id': 'B', 'text': 'shuttle', 'kind': 'quantity', 'hard_entities': []}]}, q)
        statement = 'The pass cost $23.'
        r = validate(q, statement, '$23')
        turn = SimpleNamespace(session_id='s', text=statement)
        self.assertFalse(read_operation_obligations(r, plan, {1: turn})['accepted'])

    def test_a_read_retains_an_authoritative_cancellation(self):
        q = 'How much did I pay for the pass?'
        plan = parse_semantic_plan({'operator': 'SUM', 'answer_type': 'amount', 'answer_role': 'amount',
            'actor': 'user', 'requested_status': 'completed', 'confidence': 'high', 'operands': [
                {'id': 'A', 'text': 'pass', 'kind': 'quantity', 'hard_entities': []}]}, q)
        statement = 'The pass cost $23, but that payment was never made.'
        r = validate(q, statement, '$23')
        turn = SimpleNamespace(session_id='s', text=statement)
        self.assertFalse(read_operation_obligations(r, plan, {1: turn})['accepted'])

    def test_a_winning_entity_cannot_be_replaced_by_its_measurement(self):
        q = 'At which shop did I spend the most?'
        plan = parse_semantic_plan({'operator': 'ARGMAX', 'answer_type': 'scalar', 'answer_role': 'location',
            'actor': 'user', 'confidence': 'high', 'operands': [
                {'id': 'A', 'text': 'shop spending', 'kind': 'quantity', 'hard_entities': []}]}, q)
        r = validate(q, 'I spent $58 at a shop.', '$58')
        self.assertFalse(read_operation_obligations(r, plan, {})['accepted'])


class AggregateReadAdmissionTests(unittest.TestCase):
    def admission(self, question, statement, answer, *, operator='SUM', role='amount', target_unit=None):
        plan = parse_semantic_plan({'operator': operator, 'answer_type': 'scalar', 'answer_role': role,
            'actor': 'user', 'requested_status': 'completed', 'confidence': 'high',
            'target_unit': target_unit, 'operands': [
                {'id': 'A', 'text': question, 'kind': 'quantity', 'hard_entities': []}]}, question)
        report = validate(question, statement, answer)
        self.assertEqual(report['status'], 'COMPLETE', report)
        before = copy.deepcopy(report), copy.deepcopy(plan)
        result = read_operation_obligations(report, plan, {1: SimpleNamespace(session_id='s', text=statement)})
        self.assertEqual(before, (report, plan))
        return result

    def test_quantity_sentence_and_quantity_value_cannot_answer_earnings(self):
        for answer in ('I sold 9 tickets at the theatre.', '9'):
            with self.subTest(answer=answer):
                result = self.admission('How much did I earn from the theatre sales?',
                                        'I sold 9 tickets at the theatre.', answer)
                self.assertFalse(result['accepted'])

    def test_uniform_unit_price_is_not_a_total(self):
        for statement, answer in [('Every ticket sold for $6.', '$6'),
                                   ('The total income per ticket was $6.', '$6'),
                                   ('The total income for each ticket was $6.', '$6'),
                                   ('The total income was $6 each.', '$6')]:
            with self.subTest(statement=statement):
                result = self.admission('What was my total income from ticket sales?', statement, answer)
                self.assertFalse(result['accepted'])

    def test_explicit_total_can_still_be_read_in_multiple_currencies(self):
        for value in ('$54', '54 euros', 'fifty four dollars', '54'):
            with self.subTest(value=value):
                result = self.admission('What was my total income from the theatre sales?',
                                        f'My total income from the theatre sales was {value}.', value)
                self.assertTrue(result['accepted'], result)
                self.assertEqual(result['result_binding']['reason'], 'aggregate_read_property_bound')

    def test_explicit_total_does_not_require_a_single_number_in_the_statement(self):
        result = self.admission('What was my total income?',
            'My total income was $54 after selling 9 tickets for $6 each.', '$54')
        self.assertTrue(result['accepted'], result)

    def test_a_separate_distributive_clause_does_not_reject_a_stated_total(self):
        result = self.admission('What was my total income?',
            'My total income was $54, and every ticket sold for $6.', '$54')
        self.assertTrue(result['accepted'], result)

    def test_matching_currency_does_not_replace_the_requested_property(self):
        result = self.admission('What was my total income?', 'My total expenditure was $54.', '$54')
        self.assertFalse(result['accepted'])

    def test_partial_measurement_does_not_borrow_a_total_label_from_another_clause(self):
        result = self.admission('What was my total income?',
            'My total income was unrecorded, but my ticket price was $6.', '$6')
        self.assertFalse(result['accepted'])

    def test_net_and_gross_modifiers_are_not_dropped(self):
        result = self.admission('What was my net income?', 'My gross income was $54.', '$54')
        self.assertFalse(result['accepted'])

    def test_target_units_are_checked_without_converting_the_answer(self):
        result = self.admission('What was my total income in euros?',
            'My total income was $54.', '$54', target_unit='eur')
        self.assertFalse(result['accepted'])
        result = self.admission('What was my total duration?',
            'My total duration was 90 minutes.', '90 minutes', role='duration', target_unit='hours')
        self.assertTrue(result['accepted'], result)

    def test_amount_is_not_assumed_to_mean_money(self):
        result = self.admission('What was my total discount?',
            'My total discount was 18 percent.', '18 percent')
        self.assertTrue(result['accepted'], result)

    def test_explicitly_requested_denominator_is_preserved(self):
        result = self.admission('What was my price per ticket?',
            'My price per ticket was $6.', '$6')
        self.assertTrue(result['accepted'], result)

    def test_existing_nonaggregate_numeric_and_label_contracts_do_not_load_the_parser(self):
        with patch('PRAGMOS_factual_scope._parser', side_effect=AssertionError('unrelated read changed')):
            for question, statement, answer, op, role in [
                ('How many workshops did I complete?', 'I completed 17 workshops.', '17', 'COUNT', 'count'),
                ('How long was the journey?', 'The journey took 90 minutes each way.', '90 minutes each way', 'LOOKUP', 'duration'),
                ('What is my hourly rate?', 'My hourly rate is $20 per hour.', '$20 per hour', 'LOOKUP', 'amount'),
                ('At what time do I stop answering?', 'I stop answering at 20:45.', '20:45', 'DATE_LOOKUP', 'date'),
                ('What is my project named?', 'My project is named Grove X4.', 'Grove X4', 'LOOKUP', 'label'),
                ('What ratio do I use?', 'I use one part syrup to five parts water.', 'one part syrup to five parts water', 'LOOKUP', 'object'),
            ]:
                with self.subTest(question=question):
                    self.assertTrue(self.admission(question, statement, answer, operator=op, role=role)['accepted'])

    def test_binding_resources_unavailable_declines_only_the_shortcut(self):
        with patch('PRAGMOS_factual_scope._parser', side_effect=OSError('pipeline unavailable')):
            result = self.admission('What was my total income?', 'My total income was $54.', '$54')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'aggregate_read_binding_unavailable')


class FakeLLM:
    cache = None
    def __init__(self, text, fail=False):
        self.text, self.fail, self.state = text, fail, {"sequence": [1, 2]}
    def n_ctx(self):
        return 2048
    def save_state(self):
        return copy.deepcopy(self.state)
    def load_state(self, state):
        self.state = copy.deepcopy(state)
    def __call__(self, prompt, **kwargs):
        self.kwargs = kwargs
        self.state["sequence"].append(99)
        if self.fail:
            raise RuntimeError("generation failed")
        text = self.text
        for stop in kwargs.get('stop', []):
            text = text.split(stop)[0]
        return {"choices": [{"text": text, "finish_reason": "stop"}]}


class ReaderIsolationTests(unittest.TestCase):
    def run_reader(self, llm):
        turn = SimpleNamespace(text="My boat is named Solstice.", role="user", speaker="user",
                               session_id="s", timestamp="2030-04-12", turn_id=1)
        row = {"source_turn_ids": [1], "source_session_id": "s", "source_quote": turn.text, "role": "user"}
        context = SimpleNamespace(llm=llm, count_tokens=lambda s: len(s.split()))
        return run_factual_read(context, "What is my boat named?", [row], {1: turn}, roles=["user"],
                                anchors_supported=lambda text: "boat" in text)

    def test_successful_read_restores_model_state(self):
        llm = FakeLLM("Solstice")
        before = copy.deepcopy(llm.state)
        r = self.run_reader(llm)
        self.assertEqual(r["answer"], "Solstice")
        self.assertEqual(llm.state, before)
        self.assertEqual(r["model_calls"], 1)

    def test_failure_is_local_and_restores_model_state(self):
        llm = FakeLLM("", fail=True)
        before = copy.deepcopy(llm.state)
        r = self.run_reader(llm)
        self.assertEqual(r["status"], "NOT_PROVEN")
        self.assertEqual(llm.state, before)

    def test_invalid_shape_fails_closed(self):
        self.assertEqual(self.run_reader(FakeLLM('{}'))["status"], "NOT_PROVEN")

    def test_single_value_boundary_stops_metadata(self):
        llm = FakeLLM('Solstice\n===\nSource: unrelated metadata')
        r = self.run_reader(llm)
        self.assertEqual(r['answer'], 'Solstice')
        self.assertIn('\n', llm.kwargs['stop'])


if __name__ == '__main__':
    unittest.main()
