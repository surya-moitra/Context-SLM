import copy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import PRAGMOS_benchmark_LongMemEval as adapter
from PRAGMOS_evidence_compiler import OperationProof
from PRAGMOS_factual_read import read_operation_obligations, validate_read
from test_PRAGMOS_composable_integration import arguments, fixtures, make_context, semantic_plan, semantic_trace
from test_PRAGMOS_assistance_integration import ContractModel, layer_fixture


class FactualIntegrationTests(unittest.TestCase):
    def run_case(self, name, question, operator, sessions, *, enabled, read=None):
        args, context = arguments(), make_context()
        args.pragmos_factual_mode = 'guarded' if enabled else 'off'
        record = {'question_date': '2027-02-01', 'haystack_sessions': sessions,
                  'haystack_dates': [f'2027-01-0{i+3}' for i in range(len(sessions))],
                  'haystack_session_ids': [f's{i}' for i in range(len(sessions))]}
        with patch.object(adapter, 'plan_query_with_local_slm', return_value=semantic_plan(question, operator)), \
             patch('PRAGMOS_factual_read.run_factual_read', return_value=read or {
                 'status': 'NOT_PROVEN', 'answer': None, 'model_calls': 0, 'reason': 'no_source_read'}):
            result = adapter.run_pragmos_record(context, record, question, args, name)
        return result, context

    def test_inconclusive_reader_preserves_existing_six_family_answers(self):
        for case in fixtures():
            with self.subTest(family=case[0]):
                old, _ = self.run_case(*case, enabled=False)
                new, _ = self.run_case(*case, enabled=True)
                for field in ('text', 'answer_decision', 'operation_result', 'state_history_result',
                              'ordinal_list_result', 'typed_answer_result', 'final_memories'):
                    self.assertEqual(old[field], new[field], field)

    def test_base_profile_is_not_mutated_by_computation_in_guarded_mode(self):
        case = next(c for c in fixtures() if c[0] == 'multisession')
        result, _ = self.run_case(*case, enabled=True)
        self.assertFalse(result['base_query_profile']['allow_distributed_actor_evidence'])
        self.assertIsNot(result['base_query_profile'], result['query_profile'])

    def test_disabled_semantic_planner_keeps_the_existing_answer_path(self):
        args = arguments()
        args.pragmos_factual_mode = 'guarded'
        eligible = adapter.factual_read_eligible(
            args, {'intent': adapter.USER_MEMORY_INTENT}, {}, {'operation': 'none'},
            {'selector': adapter.STATE_SELECTOR_NONE}, 'value', None)
        self.assertFalse(eligible)

    def test_positive_source_read_does_not_run_a_competing_calculation(self):
        read = {'status': 'COMPLETE', 'answer': '17', 'model_calls': 1,
                'prompt_tokens': 211, 'source_tokens': 8, 'sources': [], 'certificate': {
                    'kind': 'property_read', 'assertion': 'I completed 17 workshops.',
                    'source': {'turn_id': 1, 'session_id': 's0'}}}
        question = 'How many workshops did I complete?'
        with patch.object(adapter, 'extract_candidates_from_selected_evidence', side_effect=AssertionError('duplicate extraction')), \
             patch.object(adapter, 'execute_operation_plan', side_effect=AssertionError('unneeded calculation')):
            r, _ = self.run_case('arbitrary', question, 'COUNT', [[{'role': 'user', 'content': 'I completed 17 workshops.'}]],
                                 enabled=True, read=read)
        self.assertEqual(r['text'], '17')
        self.assertEqual(r['answer_decision'], 'source_bound_factual_read')
        self.assertEqual(r['llm_call_counts']['factual_reader'], 1)
        self.assertEqual(r['llm_call_counts']['candidate_extraction'], 0)
        self.assertIsNone(r['baseline_prediction'])
        self.assertIsNone(r['composable_execution_result'])
        self.assertEqual(r['operation_plan']['operation'], 'none')
        self.assertFalse(r['context_packing_diagnostics']['final_answer_prompt_used'])

    def test_named_peer_contract_is_not_replaced_by_quote_only_anchor_checks(self):
        args = arguments()
        args.pragmos_factual_mode = 'guarded'
        result = adapter.factual_proof_admission({}, args, {'actor_binding_enabled': True})
        self.assertTrue(result['accepted'])
        self.assertEqual(result['reason'], 'existing_actor_bound_contract')

    def test_resolved_computation_scope_is_not_reopened_as_a_lexical_filter(self):
        args = arguments()
        args.pragmos_factual_mode = 'guarded'
        profile = {'required_anchor_groups': [{'text': 'earlier', 'tokens': ['earlier']}]}
        result = adapter.factual_proof_admission({}, args, profile, unresolved_scope=[])
        self.assertTrue(result['accepted'])

    def test_unresolved_subject_cannot_be_overridden_by_an_unrelated_proof(self):
        args = arguments()
        args.pragmos_factual_mode = 'guarded'
        profile = {'required_anchor_groups': [{'text': 'uncle', 'tokens': ['uncle']}]}
        proof = {'provenance': [{'source': {'text': 'My sister gave me a mixer.'}}]}
        result = adapter.factual_proof_admission(proof, args, profile, unresolved_scope=['uncle'])
        self.assertFalse(result['accepted'])

    def test_rejected_wrong_owner_frame_cannot_reenter_through_fallback(self):
        wrong = OperationProof('COMPLETE', 'COLLECT', 'a mixer', executable=True,
                               calculation={'provenance': [{'source_quote': 'My sister gave me a mixer.'}]})
        with patch.object(adapter, 'execute_frame_operation_proof', return_value=(wrong, {
                'applicable': True, 'status': 'COMPLETE'})) as frames, \
             patch.object(adapter, 'execute_grounded_candidate_operation_proof', return_value=(None, {
                'applicable': False})):
            result, _ = self.run_case('unrelated-owner', 'What did my uncle give me?', 'COLLECT',
                                      [[{'role': 'user', 'content': 'My sister gave me a mixer.'}]], enabled=True)
        self.assertTrue(frames.called)
        self.assertEqual(result['text'], 'I do not know.')
        self.assertFalse(result['evidence_compiler_proof']['executable'])
        self.assertEqual(result['abstention_reason'], 'requested_scope_not_in_positive_proof')


class ImmediateCheckpointParityTests(unittest.TestCase):
    def test_off_preserves_immediate_checkpoint_in_all_six_families(self):
        try:
            code = subprocess.check_output(['git', 'show', '06a966c:PRAGMOS_benchmark_LongMemEval.py'])
        except (OSError, subprocess.CalledProcessError):
            self.skipTest('Immediate pre-factual checkpoint is unavailable')
        old = ModuleType('_pragmos_pre_factual_checkpoint')
        old.__file__ = adapter.__file__
        sys.modules[old.__name__] = old
        try:
            exec(compile(code, '<pre-factual-checkpoint>', 'exec'), old.__dict__)
            for name, q, op, sessions in fixtures():
                with self.subTest(family=name):
                    record = {'question_date': '2027-02-01', 'haystack_sessions': sessions,
                              'haystack_dates': [f'2027-01-0{i+3}' for i in range(len(sessions))],
                              'haystack_session_ids': [f's{i}' for i in range(len(sessions))]}
                    results, contexts = [], []
                    for module in (old, adapter):
                        context, args = make_context(), arguments()
                        args.pragmos_factual_mode = 'off'
                        with patch.object(module, 'plan_query_with_local_slm', return_value=semantic_plan(q, op)):
                            results.append(module.run_pragmos_record(context, record, q, args, name))
                        contexts.append(context)
                    self.assertEqual(semantic_trace(results[0]), semantic_trace(results[1]))
                    self.assertEqual(contexts[0].llm.calls, contexts[1].llm.calls)
                    self.assertEqual(contexts[0].llm.state, contexts[1].llm.state)
                    self.assertEqual(contexts[0].vector_memory, contexts[1].vector_memory)
                    self.assertEqual(list(contexts[0].G.edges(data=True, keys=True)),
                                     list(contexts[1].G.edges(data=True, keys=True)))
        finally:
            sys.modules.pop(old.__name__, None)


class BothGuardedModesTests(unittest.TestCase):
    def run_case(self, case, *, factual, action='retrieve_record', read=None):
        name, question, operator, sessions = case
        context = layer_fixture(ContractModel(action=action))
        args = arguments()
        args.pragmos_assistance_mode = 'guarded'
        args.pragmos_factual_mode = 'guarded' if factual else 'off'
        record = {'question_date': '2027-02-01', 'haystack_sessions': sessions,
                  'haystack_dates': [f'2027-01-0{i+3}' for i in range(len(sessions))],
                  'haystack_session_ids': [f's{i}' for i in range(len(sessions))]}
        with patch('PRAGMOS_assistance_contracts._grammar', return_value=object()), \
             patch.object(adapter, 'plan_query_with_local_slm', return_value=semantic_plan(question, operator)), \
             patch('PRAGMOS_factual_read.run_factual_read', return_value=read or {
                 'status': 'NOT_PROVEN', 'answer': None, 'model_calls': 0, 'reason': 'no_supported_read'}) as reader:
            result = adapter.run_pragmos_record(context, record, question, args, name)
        return result, context, reader.call_count

    def test_inconclusive_factual_read_preserves_all_six_families_with_advice_guard_on(self):
        for case in fixtures():
            with self.subTest(family=case[0]):
                old, a, _ = self.run_case(case, factual=False)
                new, b, _ = self.run_case(case, factual=True)
                for field in ('text', 'answer_decision', 'operation_result', 'state_history_result',
                              'ordinal_list_result', 'typed_answer_result', 'final_memories'):
                    self.assertEqual(old[field], new[field], field)
                self.assertEqual(a.llm.state, b.llm.state)
                self.assertEqual(a.vector_memory, b.vector_memory)

    def test_accepted_advice_never_enters_factual_reader(self):
        case = ('advice', 'Suggest an activity.', 'LOOKUP', [[{'role': 'user', 'content': 'I enjoy drawing.'}]])
        old, a, _ = self.run_case(case, factual=False, action='create_advice')
        new, b, calls = self.run_case(case, factual=True, action='create_advice')
        self.assertEqual(calls, 0)
        self.assertEqual(new['answer_decision'], 'contextual_assistance')
        self.assertEqual(semantic_trace(old), semantic_trace(new))
        self.assertEqual(a.llm.calls, b.llm.calls)

    def test_both_guards_can_return_an_explicit_count_without_computing(self):
        read = {'status': 'COMPLETE', 'answer': '17', 'model_calls': 1, 'prompt_tokens': 211,
                'source_tokens': 8, 'sources': [], 'certificate': {'kind': 'property_read',
                    'assertion': 'I completed 17 workshops.', 'source': {'turn_id': 1, 'session_id': 's0'}}}
        case = ('explicit-count', 'How many workshops did I complete?', 'COUNT',
                [[{'role': 'user', 'content': 'I completed 17 workshops.'}]])
        result, _, calls = self.run_case(case, factual=True, read=read)
        self.assertEqual(result['text'], '17')
        self.assertEqual(calls, 1)
        self.assertEqual(result['llm_call_counts']['assistance_request'], 1)
        self.assertEqual(result['llm_call_counts']['factual_reader'], 1)


class AggregateReadFallbackTests(unittest.TestCase):
    def test_rejected_read_reaches_the_working_executor_with_both_guards_on(self):
        from PRAGMOS_typed_claims import bind_typed_claim, source_windows
        from PRAGMOS_typed_computation import execute_typed_plan
        from test_PRAGMOS_typed_computation import request, selection

        question = 'How much did I earn from the theatre sales?'
        count_text = 'I sold 9 tickets at the theatre.'
        price_text = 'Every ticket sold for $6.'
        record = {'question_date': '2027-02-01', 'haystack_sessions': [
            [{'role': 'user', 'content': count_text}], [{'role': 'user', 'content': price_text}]],
            'haystack_dates': ['2027-01-03', '2027-01-04'], 'haystack_session_ids': ['s0', 's1']}
        read = validate_read({'mode': 'read', 'source_id': 'R1', 'answer': count_text}, question,
            [{'source_id': 'R1', 'text': count_text, 'turn_id': 1, 'session_id': 's0'}],
            anchors_supported=lambda _: True, count_tokens=lambda s: len(s.split()))
        self.assertEqual(read['status'], 'COMPLETE')
        read.update(model_calls=1, prompt_tokens=200, source_tokens=10, sources=[])

        def compute(context, q, sources, **kwargs):
            windows = source_windows(sources)
            count = next(w for w in windows if w.text == count_text)
            price = next(w for w in windows if w.text == price_text)
            first = bind_typed_claim(count, sources[count.source_id], selection(entity='9 tickets',
                key='9 tickets', attribute='tickets', numbers=[0], quantity_kind='count', operand='left'), {})
            second = bind_typed_claim(price, sources[price.source_id], selection(actor=None,
                binding='linked_event', link=first.claim_id, entity='ticket', key='ticket', attribute='ticket',
                numbers=[0], quantity_kind='rate', operand='right'), {first.claim_id: first})
            proof = execute_typed_plan(request('multiply'), [first, second],
                {count.window_id: count, price.window_id: price}, sources)
            return {'status': proof['status'], 'answer': proof['answer'], 'proof': proof,
                'extraction_complete': True, 'isolation': 'llama_state_restored_no_external_cache',
                'selected_engine': 'typed_v4', 'calls': {'planner_calls': 1, 'extractor_calls': 2}}

        results, contexts = [], []
        for enabled in (False, True):
            args = arguments()
            args.pragmos_assistance_mode = 'guarded'
            args.pragmos_factual_mode = 'guarded' if enabled else 'off'
            args.pragmos_proof_first = True
            args.pragmos_composable_mode = 'candidate'
            args.pragmos_composable_protocol = 'typed_v4'
            context = layer_fixture(ContractModel(action='derive_fact'))
            with patch('PRAGMOS_assistance_contracts._grammar', return_value=object()), \
                 patch.object(adapter, 'plan_query_with_local_slm', return_value=semantic_plan(question, 'SUM')), \
                 patch('PRAGMOS_factual_read.run_factual_read', return_value=copy.deepcopy(read)), \
                 patch('PRAGMOS_composable_candidate.run_typed_candidate', side_effect=compute) as executor, \
                 patch.object(adapter, 'execute_operation_plan', side_effect=AssertionError('duplicate fallback')):
                result = adapter.run_pragmos_record(context, record, question, args, 'neutral-revenue')
            self.assertEqual(executor.call_count, 1)
            self.assertEqual(result['text'], '$54')
            self.assertEqual(result['answer_decision'], 'composable_evidence_execution')
            self.assertFalse(result['safe_abstention'])
            results.append(result)
            contexts.append(context)
        self.assertEqual(results[1]['execution_diagnostics']['factual_read']['status'], 'NOT_PROVEN')
        self.assertEqual(results[0]['operation_result'], results[1]['operation_result'])
        self.assertEqual(results[0]['final_memories'], results[1]['final_memories'])
        self.assertEqual(contexts[0].vector_memory, contexts[1].vector_memory)
        self.assertEqual(list(contexts[0].G.edges(data=True, keys=True)),
                         list(contexts[1].G.edges(data=True, keys=True)))

    def test_explicit_aggregate_property_still_shortcuts_with_both_guards_on(self):
        question = 'What was my total income from the theatre sales?'
        statement = 'My total income from the theatre sales was $54.'
        read = validate_read({'mode': 'read', 'source_id': 'R1', 'answer': '$54'}, question,
            [{'source_id': 'R1', 'text': statement, 'turn_id': 1, 'session_id': 's0'}],
            anchors_supported=lambda _: True, count_tokens=lambda s: len(s.split()))
        read.update(model_calls=1, prompt_tokens=200, source_tokens=10, sources=[])
        case = ('stated-total', question, 'SUM', [[{'role': 'user', 'content': statement}]])
        with patch.object(adapter, 'extract_candidates_from_selected_evidence', side_effect=AssertionError('unneeded extraction')):
            result, _, _ = BothGuardedModesTests().run_case(case, factual=True, read=read)
        self.assertEqual(result['text'], '$54')
        self.assertEqual(result['answer_decision'], 'source_bound_factual_read')


class SavedReadAdmissionTests(unittest.TestCase):
    root = Path('/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2')

    def test_saved_user_and_multi_reader_promotions_keep_all_other_admissions(self):
        paths = [self.root / 'single_session_user_35_odd_v1/runs/factual_read_guarded_v2',
                 self.root / 'execution_latency_v2/runs/factual_read_guarded_v2']
        files = [next(path.glob('*_trace.jsonl'), None) for path in paths]
        if any(path is None for path in files):
            self.skipTest('Local source-bound read artifacts unavailable')
        from PRAGMOS_evidence_compiler import parse_semantic_plan
        checked = 0
        for path in files:
            for row in map(json.loads, path.read_text().splitlines()):
                read = (row.get('execution_diagnostics') or {}).get('factual_read') or {}
                if read.get('status') != 'COMPLETE':
                    continue
                raw = copy.deepcopy(row['semantic_query_plan'])
                raw['operands'] = [{**o, 'id': o['operand_id']} for o in raw.get('operands', [])]
                plan = parse_semantic_plan(raw, row['question'])
                source = read['certificate']['source']
                checks = (read.get('source_obligations', {}).get('numeric_source_validation') or {}).get('source_checks') or []
                original = (checks[0].get('source') or {}).get('source_quote') if checks else None
                turn = SimpleNamespace(session_id=source['session_id'], text=original or read['certificate']['assertion'])
                admission = read_operation_obligations(read, plan, {source['turn_id']: turn})
                with self.subTest(question=row['question_id']):
                    self.assertEqual(admission['accepted'], row['question_id'] != 'synthetic_v2s_r2_0223')
                checked += 1
        self.assertEqual(checked, 26)


if __name__ == '__main__':
    unittest.main()
