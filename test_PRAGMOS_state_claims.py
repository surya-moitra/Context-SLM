"""Counterexamples for atomic state claims; fixtures are not model accuracy."""

import unittest
from dataclasses import replace
from unittest.mock import patch
import os

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_state_binding import extract_history, membership_history, quantity_literal_choices, apply_corrections, select_history, predicate_history, choose_executable_query
from PRAGMOS_state_claims import relevant_windows, review_value, membership_changes, legacy_contract_review, action_assertion, correction_links, qualification_kind, read_time_selector, requested_output_projection
from PRAGMOS_state_views import accepted_state_candidate
from test_PRAGMOS_state_binding import fact
from test_PRAGMOS_composable_candidate import context
from PRAGMOS_state_views import run_state_candidate
from test_PRAGMOS_state_contracts import contract_payloads


def source(text, day=1):
    return Source('S' + str(day), 'session' + str(day), str(day), 'user', 'user', f'2025-01-{day:02d}', text)


def responses(*items):
    queue = iter(items)
    return lambda *args: next(queue)


def meaning(owner='speaker', match='requested', status='actual', assertion='0'):
    option = ('6' if 'unresolved' in (owner, match, status) else '4' if owner == 'other' else
              '5' if match == 'different' else {'reference': '2', 'planned': '3'}.get(status, '1'))
    return {'assertion': assertion, 'meaning': option}


def report():
    return {'observations': [], 'reviewed_windows': [], 'source_view_complete': False}


class ClaimBindingTests(unittest.TestCase):
    def test_number_owned_by_neighbour_is_not_assigned_to_narrator(self):
        for relation in ('neighbour', 'roommate', 'cousin', 'landlord'):
            with self.subTest(relation=relation):
                s = source(f'My {relation} owns two instruments. I own a flute.')
                rows = extract_history(responses({'values': ['two']}, meaning(owner='other')), '', 'instruments',
                                       s, 0, s.text, 1, 'quantity')
                self.assertEqual(rows, [])

    def test_entity_led_claim_can_belong_to_speaker_without_first_person_in_same_sentence(self):
        for entity in ('HarborView', 'QuietNotes', 'SilverTide'):
            with self.subTest(entity=entity):
                s = source(f'{entity} now has 907 subscribers. I own that account.')
                rows = extract_history(responses({'values': ['907']}, meaning()), '', 'subscribers', s, 0, s.text, 1, 'quantity')
                self.assertEqual(rows[0].value, '907')
                self.assertEqual(rows[0].claim_assertion, f'{entity} now has 907 subscribers.')
                self.assertEqual(rows[0].claim_review['owner'], 'speaker')

    def test_qualifications_are_preserved_as_source_claim_status(self):
        s = source('I have 11 tokens. I kept an old 6-token receipt. I want 20 tokens.')
        rows = extract_history(responses({'values': ['6', '11', '20']},
                               meaning(status='reference', assertion='1'), meaning(),
                               meaning(status='planned', assertion='2')),
                               '', 'tokens', s, 0, s.text, 1, 'quantity')
        self.assertEqual([(r.value, r.kind, r.modality) for r in rows],
                         [('6', 'reference', 'actual'), ('11', 'snapshot', 'actual'), ('20', 'snapshot', 'planned')])
        self.assertIsNone(rows[-1].previous)

    def test_corrected_amounts_are_not_confused_with_another_financial_property(self):
        s = source('My authorization is $225, correcting $150. I spent $190.')
        rows = extract_history(responses({'values': ['$150', '$225', '$190']},
                               meaning(status='reference'), meaning(), meaning(match='different', assertion='1')),
                               '', 'authorization', s, 0, s.text, 1, 'quantity')
        self.assertEqual([r.value for r in rows if r.kind == 'snapshot'], ['$225'])

    def test_spelled_out_frequency_is_not_dropped_for_lacking_digit_candidates(self):
        s = source('I practice once a fortnight.')
        rows = extract_history(responses({'values': ['once a fortnight']}, meaning()), '', 'practice',
                               s, 0, s.text, 1, 'quantity')
        self.assertEqual(rows[0].value, 'once a fortnight')

    def test_unresolved_binding_cannot_be_silently_dropped_from_a_complete_view(self):
        s = source('I have 9 tokens.')
        with self.assertRaisesRegex(ContractError, 'binding_unresolved'):
            extract_history(responses({'values': ['9']}, meaning(owner='unresolved')), '', 'tokens', s, 0, s.text, 1, 'quantity')

    def test_fabricated_value_is_rejected_before_owner_review(self):
        s = source('I have 9 tokens.')
        with self.assertRaisesRegex(ContractError, 'literal_not_in_source'):
            extract_history(responses({'values': ['99']}), '', 'tokens', s, 0, s.text, 1, 'quantity')

    def test_selected_assertion_must_contain_the_value(self):
        with self.assertRaisesRegex(ContractError, 'assertion_address_invalid'):
            review_value(responses(meaning(assertion='1')), 'tokens', [], '9', 'I have 9 tokens. I rest.',
                         ['I have 9 tokens.', 'I rest.'])

    def test_attribute_cannot_be_borrowed_from_another_assertion(self):
        payload = {'attribute': 'tall', 'assertion': '0', 'meaning': '1'}
        with self.assertRaisesRegex(ContractError, 'attribute_not_in_assertion'):
            review_value(responses(payload), 'tokens', [], '9', 'I have 9 tokens. I am tall.',
                         ['I have 9 tokens.', 'I am tall.'])

    def test_literal_catalog_cannot_expand_part_of_a_larger_numeral(self):
        options = quantity_literal_choices('I have 116 stamps and 16 coins.')
        self.assertNotIn('16 stamps', options)
        self.assertIn('16 coins', options)

    def test_literal_catalog_retains_frequency_period(self):
        self.assertIn('four times a month', quantity_literal_choices('I attend four times a month.'))


class MembershipTests(unittest.TestCase):
    def test_initial_set_plus_acquisition_minus_disposal_has_provenance(self):
        a = source('I own a flute, a cello and a piano.')
        b = source('I acquired a viola. I sold the cello.', 2)
        ask = responses({'baseline': ['flute', 'cello', 'piano'], 'add': [], 'remove': [], 'complete_baseline': True},
                        {'coverage': 'full'},
                        {'assertion': '0'}, meaning(), {'assertion': '0'}, meaning(), {'assertion': '0'}, meaning(),
                        {'baseline': [], 'add': ['viola'], 'remove': ['cello'], 'complete_baseline': False},
                        {'assertion': '0'}, meaning(), {'assertion': '1'}, meaning(), {'operation': 'count'})
        audit = report()
        result = membership_history(ask, 'How many instruments do I own?', 'instruments', [],
                                    [(b, 0, b.text), (a, 0, a.text)], audit)
        self.assertEqual(result['answer'], '3')
        self.assertEqual([r['value'] for r in result['current_members']], ['flute', 'piano', 'viola'])
        self.assertTrue(audit['source_view_complete'])
        self.assertTrue(result['complete_initial_set'])
        self.assertEqual(result['selected_observations'][0][-1]['action'], 'remove')

    def test_additions_without_initial_set_cannot_certify_current_count(self):
        s = source('I bought a flute.')
        with self.assertRaisesRegex(ContractError, 'initial_set_not_established'):
            membership_history(responses({'baseline': [], 'add': ['flute'], 'remove': [], 'complete_baseline': False},
                               {'assertion': '0'}, meaning()),
                               'How many?', 'instruments', [], [(s, 0, s.text)], report())

    def test_unknown_disposal_does_not_decrement_an_arbitrary_member(self):
        a = source('I own a flute.')
        b = source('I sold a piano.', 2)
        with self.assertRaisesRegex(ContractError, 'disposal_identity_unresolved'):
            membership_history(responses({'baseline': ['flute'], 'add': [], 'remove': [], 'complete_baseline': True},
                               {'coverage': 'full'}, {'assertion': '0'}, meaning(),
                               {'baseline': [], 'add': [], 'remove': ['piano'], 'complete_baseline': False}, {'assertion': '0'}, meaning()),
                               'How many?', 'instruments', [], [(a, 0, a.text), (b, 0, b.text)], report())

    def test_other_owner_cannot_enter_a_membership_reduction(self):
        s = source('My neighbour owns a flute.')
        with self.assertRaisesRegex(ContractError, 'action_not_bound'):
            membership_history(responses({'baseline': ['flute'], 'add': [], 'remove': [], 'complete_baseline': True},
                               {'coverage': 'full'}, {'assertion': '0'}, meaning(owner='other')),
                               'How many?', 'instruments', [], [(s, 0, s.text)], report())

    def test_same_turn_reacquisition_requires_explicit_order_not_field_order(self):
        with self.assertRaisesRegex(ContractError, 'order_binding'):
            membership_changes(responses({'baseline': [], 'add': ['flute'], 'remove': ['flute'], 'complete_baseline': False}),
                               'instruments', [], 'I sold a flute and later bought that flute again.')

    def test_fabricated_member_cannot_enter_a_set(self):
        with self.assertRaisesRegex(ContractError, 'item_not_literal'):
            membership_changes(responses({'baseline': ['piano'], 'add': [], 'remove': [], 'complete_baseline': True}),
                               'instruments', [], 'I own a flute.')

    def test_partial_confirmation_cannot_reset_membership(self):
        result = membership_changes(responses(
            {'baseline': ['bell'], 'add': ['flute'], 'remove': [], 'complete_baseline': True},
            {'coverage': 'partial'}), 'instruments', [], 'The bell is still here. I bought a flute.')
        self.assertEqual(result['baseline'], [])
        self.assertFalse(result['complete_baseline'])
        self.assertEqual(result['add'], ['flute'])

    def test_disposal_is_bound_to_transaction_not_a_later_photo(self):
        text = 'I sold my drum. I kept an old drum photo.'
        selected = action_assertion(responses({'assertion': '0'}), 'remove', 'drum', text,
                                    ['I sold my drum.', 'I kept an old drum photo.'])
        self.assertEqual(selected, 'I sold my drum.')

    def test_unsupported_event_has_no_executable_membership_witness(self):
        with self.assertRaisesRegex(ContractError, 'event_assertion_unbound'):
            action_assertion(responses({'assertion': 'unsupported'}), 'remove', 'drum',
                             'I kept a drum photo.', ['I kept a drum photo.'])


class CorrectionTests(unittest.TestCase):
    def test_ordinary_changes_need_no_correction_operator(self):
        s = source('I had 12 and later reached 18.')
        self.assertEqual(correction_links(responses({'qualification': None}), [fact('12'), fact('18', 2)], s, s.text), [])

    def test_explicit_error_qualification_binds_old_and_new_in_one_assertion(self):
        s = source('My corrected amount is 18, not the mistaken 12.')
        audit = []
        links = correction_links(responses({'qualification': 'corrected'}, {'description': 'The report was wrong.', 'kind': 'report_error'},
                                 {'corrections': [{'old': '12', 'new': '18'}]}),
                                 [fact('12'), fact('18', 2)], s, s.text, audit=audit)
        self.assertEqual(links[0]['qualification'], 'corrected')
        self.assertEqual(audit[0]['status'], 'accepted')

    def test_a_qualification_cannot_join_two_separate_observations(self):
        s = source('My earlier amount was 12. My new observation is 18.')
        audit = []
        links = correction_links(responses({'qualification': 'new observation'}, {'description': 'A proposed report correction.', 'kind': 'report_error'},
                                 {'corrections': [{'old': '12', 'new': '18'}]}),
                                 [fact('12'), fact('18', 2)], s, s.text, audit=audit)
        self.assertEqual(links, [])
        self.assertEqual(audit[0]['reason'], 'state_correction_arguments_not_in_one_assertion')

    def test_conflicting_order_in_a_claimed_correction_cannot_certify_an_answer(self):
        s = source('My corrected amount is 18, not 12.')
        with self.assertRaisesRegex(ContractError, 'contradicts_history_order'):
            correction_links(responses({'qualification': 'corrected'}, {'description': 'A report correction.', 'kind': 'report_error'},
                             {'corrections': [{'old': '18', 'new': '12'}]}),
                             [fact('12'), fact('18', 2)], s, s.text)

    def test_real_change_qualification_cannot_retract_a_true_state(self):
        s = source('My grade changed from Amber to Jade.')
        audit = []
        links = correction_links(responses({'qualification': 'changed from Amber to Jade'},
                                 {'description': 'The actual grade changed.', 'kind': 'real_change'}),
                                 [fact('Amber'), fact('Jade', 2)], s, s.text, audit=audit)
        self.assertEqual(links, [])
        self.assertEqual(audit[0]['status'], 'qualification_not_a_report_error')

    def link(self, old='12', new='18', source_id='S2'):
        return {'old': old, 'new': new, 'source_id': source_id, 'quote': f'My corrected value is {new}, not {old}.'}

    def test_mistaken_report_is_retracted_not_treated_as_a_real_earliest_state(self):
        rows = [fact('12'), fact('12', 2), fact('18', 3, source_id='S2', timestamp='2025-01-02')]
        corrected = apply_corrections(rows, [self.link()])
        self.assertEqual([r.kind for r in corrected], ['reference', 'reference', 'correction'])
        self.assertEqual(select_history(corrected, 'earliest')[0].value, '18')
        self.assertEqual(select_history(corrected, 'latest')[0].value, '18')
        self.assertIn('invalidated_by', corrected[0].claim_review)
        self.assertEqual(rows[0].kind, 'snapshot')

    def test_real_state_changes_are_preserved_without_an_explicit_correction(self):
        rows = [fact('12'), fact('18', 2)]
        self.assertEqual(apply_corrections(rows, []), rows)

    def test_a_repeated_value_in_a_different_earlier_occurrence_is_not_erased(self):
        rows = [fact('12'), fact('15', 2), fact('12', 3), fact('12', 4),
                fact('18', 5, source_id='S4', timestamp='2025-01-04')]
        corrected = apply_corrections(rows, [self.link(source_id='S4')])
        self.assertEqual(corrected[0].kind, 'snapshot')
        self.assertEqual(corrected[2].kind, 'reference')

    def test_correction_cannot_retract_another_owner(self):
        rows = [fact('12', actor='Nora', actor_binding='named'), fact('12', 2),
                fact('18', 3, source_id='S2', timestamp='2025-01-02')]
        self.assertEqual(apply_corrections(rows, [self.link()])[0].kind, 'snapshot')

    def test_correction_cycles_are_rejected(self):
        with self.assertRaisesRegex(ContractError, 'correction_cycle'):
            apply_corrections([], [self.link(), self.link('18', '12')])


class PredicateBindingTests(unittest.TestCase):
    def plan(self):
        return {'operation': 'exists', 'form': 'boolean', 'whole_units': False, 'slots': [
            {'label': 'completion', 'subject': None, 'property': 'completion', 'anchors': [],
             'selector': 'latest', 'marker': None, 'value_kind': 'text', 'status': 'actual'}]}

    def test_explicit_current_denial_overrides_old_positive_state(self):
        a, b = source('I have completed the assignment.'), source('I have not completed the revised assignment.', 2)
        ask = responses({'answer': 'Yes', 'quote': a.text, 'actor': 'I'}, {'answer': 'No', 'quote': b.text, 'actor': 'I'})
        with patch('PRAGMOS_state_views.plan_state_contract', return_value=self.plan()):
            result = predicate_history(ask, 'Did I complete the assignment?', [(a, 0, a.text), (b, 0, b.text)], {})
        self.assertEqual(result['answer'], 'No')

    def test_missing_assertion_is_not_an_explicit_negative(self):
        a = source('I plan to complete the assignment.')
        with patch('PRAGMOS_state_views.plan_state_contract', return_value=self.plan()):
            with self.assertRaisesRegex(ContractError, 'operand_unresolved'):
                predicate_history(responses({'answer': 'not established', 'quote': None, 'actor': None}),
                                  'Did I complete the assignment?', [(a, 0, a.text)], {})

    def test_a_different_object_cannot_overwrite_an_explicit_named_predicate(self):
        a = source('I finished The Marble Atlas. My essay remains unfinished.')
        prompts = []
        def ask(prompt, *args):
            prompts.append(prompt)
            return {'answer': 'Yes', 'quote': 'I finished The Marble Atlas.', 'actor': 'I'}
        with patch('PRAGMOS_state_views.plan_state_contract', return_value=self.plan()):
            result = predicate_history(ask, 'Did I finish The Marble Atlas?', [(a, 0, a.text)], {})
        self.assertEqual(result['answer'], 'Yes')
        self.assertNotIn('essay remains', prompts[0])


class RoutingAndSupportTests(unittest.TestCase):
    def test_unsupported_named_owner_does_not_enter_a_first_person_state_view(self):
        a = source('Leena has 7 reports.')
        ctx = context([])
        with patch('PRAGMOS_composable_candidate.schema_grammar', return_value=None):
            result = run_state_candidate(ctx, 'How many reports does Leena have?', {'S1': a}, n_ctx=2048, protocol="bound_v3")
        self.assertEqual(result['status'], 'NOT_PROVEN')
        self.assertFalse(result['observations'])
        self.assertFalse(accepted_state_candidate(result, 'candidate'))
        self.assertTrue(ctx.llm.restored)

    def test_two_source_narrators_cannot_be_merged_as_one_implicit_question_owner(self):
        from test_PRAGMOS_state_contracts import planner_responses
        a, b = replace(source('I have 4 tokens.'), speaker='Asha'), replace(source('I have 9 tokens.', 2), speaker='Miles')
        ctx = context(planner_responses(contract_payloads('How many tokens do I have?', 'tokens', projection='cardinality', dimension='count')))
        with patch('PRAGMOS_composable_candidate.schema_grammar', return_value=None):
            result = run_state_candidate(ctx, 'How many tokens do I have?', {'S1': a, 'S2': b}, n_ctx=2048, protocol="bound_v3")
        self.assertEqual(result['status'], 'NOT_PROVEN')
        self.assertIn('multiple_source_speakers', result['reason'])
        self.assertTrue(ctx.llm.restored)

    def test_typed_date_and_clock_operands_cannot_be_ignored_by_a_model_selector(self):
        def forbidden(*args):
            self.fail('A typed temporal operand must bind before model inference')
        for question in ('What was the reading on 12 March?', 'What was the reading on September 23?',
                         'What was the reading on 2024-07-19?', 'What was the reading at 14:30?'):
            with self.subTest(question=question):
                self.assertEqual(read_time_selector(forbidden, question), 'at_time')

    def test_an_ordinary_numeric_operand_does_not_become_a_date(self):
        self.assertEqual(read_time_selector(responses({'selector': 'latest'}), 'How many of my 12 cables remain?'), 'latest')

    def test_current_text_state_cannot_select_a_previous_answer_from_a_menu(self):
        def no_model_choice(*args):
            self.fail('The query-time contract already fixes the only executable read')
        plan = choose_executable_query(no_model_choice, 'Where is my current location?',
                                       [fact('Northwood'), fact('Southmere', 2)], read_selector='latest')
        self.assertEqual(plan['operation'], 'read')
        self.assertEqual(plan['selectors'], ['latest'])

    def test_historical_membership_is_not_answered_with_the_current_set(self):
        from test_PRAGMOS_state_program import contract, program, select
        from PRAGMOS_state_program import validate_program
        c = contract('How many instruments did I own initially?', 'instruments', projection='cardinality',
                     dimension='count', family='collect', selector='earliest')
        current = program(c.question, [select(selector='latest'), {'id': 'N2', 'op': 'members', 'input': 'N1'},
                                      {'id': 'N3', 'op': 'count', 'input': 'N2'}])
        with self.assertRaisesRegex(ContractError, 'requested_state_selector'):
            validate_program(current, c, [], {'speaker:user'})

    def test_independent_events_do_not_become_an_overwritten_state(self):
        a, b = source('I paid $12 for one delivery.'), source('I paid $18 for another delivery.', 2)
        ctx = context([*contract_payloads('What is the total cost?', 'cost', family='calculate', selector='all'),
                       {'sources': ['0', '1']},
                       {'view': 'independent_events'}])
        with patch('PRAGMOS_composable_candidate.schema_grammar', return_value=None):
            result = run_state_candidate(ctx, 'What is the total cost?', {'S1': a, 'S2': b}, n_ctx=2048, protocol="bound_v3")
        self.assertEqual(result['status'], 'NOT_PROVEN')
        self.assertIsNone(result['answer'])
        self.assertTrue(ctx.llm.restored)

    def test_existing_dense_encoder_can_admit_alias_without_admitting_unrelated_context(self):
        a, b = source('I bought a new bike.'), source('I asked about fruit.', 2)
        audit = {'source_semantic_scores': {'S1:0': .71, 'S2:0': .09}, 'source_min_score': .35}
        selected = relevant_windows(responses(), 'How many bicycles?', [(a, 0, a.text), (b, 0, b.text)], audit)
        self.assertEqual([item[0].source_id for item in selected], ['S1'])

    def test_named_entity_scope_excludes_a_different_named_topic(self):
        a, b = source('I finished The Marble Atlas.'), source('I did not finish The Birch Index.', 2)
        audit = {'identity_constraints': ['The Marble Atlas'],
                 'source_semantic_scores': {'S1:0': .76, 'S2:0': .72}, 'source_min_score': .35}
        selected = relevant_windows(responses(), 'Did I finish The Marble Atlas?', [(a, 0, a.text), (b, 0, b.text)], audit)
        self.assertEqual([item[0].source_id for item in selected], ['S1'])

    def test_registered_alias_preserves_named_entity_scope(self):
        a, b = source('Morgan now works at Fern.'), source('Rowan now works at Cedar.', 2)
        audit = {'identity_constraints': ['Moe'], 'identity_aliases': {'Moe': ['Morgan']},
                 'source_semantic_scores': {'S1:0': .76, 'S2:0': .72}, 'source_min_score': .35}
        selected = relevant_windows(responses(), 'Where does Moe work?', [(a, 0, a.text), (b, 0, b.text)], audit)
        self.assertEqual([item[0].source_id for item in selected], ['S1'])

    def test_selected_synonym_continuation_is_not_deleted_by_a_literal_hint(self):
        a, b = source('I own a bicycle.'), source('I bought another bike.', 2)
        audit = {}
        selected = relevant_windows(responses({'sources': ['0', '1']}), 'How many bicycles?',
                                    [(a, 0, a.text), (b, 0, b.text)], audit)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(audit['source_selection']), 2)

    def test_duplicate_source_selection_is_not_accepted(self):
        a = source('I own a bicycle.')
        with self.assertRaisesRegex(ContractError, 'selection_invalid'):
            relevant_windows(responses({'sources': ['0', '0']}), 'How many?', [(a, 0, a.text)], {})

    def test_count_of_statements_is_incompatible_with_scalar_state_view(self):
        result = {'source_representation': 'scalar_history', 'contract': {'operation': 'read'}, 'answer': 'four times a month',
                  'requested_output_kind': 'frequency', 'proof': {'selected_observations': [[{'value': 'four times a month'}]]}}
        review = legacy_contract_review({'operator': 'COUNT_DISTINCT', 'answer': '2'}, result)
        self.assertEqual(review['decision'], 'replace_incompatible_view')

    def test_a_valid_sum_cannot_be_displaced_by_a_state_snapshot(self):
        result = {'source_representation': 'scalar_history', 'source_value_kind': 'quantity',
                  'contract': {'operation': 'read'}, 'answer': '$19'}
        self.assertEqual(legacy_contract_review({'operator': 'SUM', 'answer': '$57'}, result)['decision'],
                         'preserve_compatible_legacy_completion')

    def test_a_valid_cardinality_cannot_be_displaced_by_a_scalar_measurement(self):
        result = {'source_representation': 'scalar_history', 'contract': {'operation': 'read'},
                  'answer': '7', 'requested_output_kind': 'cardinality'}
        self.assertEqual(legacy_contract_review({'operator': 'COUNT_DISTINCT', 'answer': '3'}, result)['decision'],
                         'preserve_compatible_legacy_completion')

    def test_rate_projection_needs_a_source_period_not_just_a_model_label(self):
        result = {'source_representation': 'scalar_history', 'contract': {'operation': 'read'},
                  'answer': '7', 'requested_output_kind': 'frequency', 'proof': {'selected_observations': [[{'value': '7'}]]}}
        self.assertEqual(legacy_contract_review({'operator': 'COUNT', 'answer': '3'}, result)['decision'],
                         'preserve_compatible_legacy_completion')

    def test_duration_is_not_a_recurrence_rate(self):
        result = {'source_representation': 'scalar_history', 'contract': {'operation': 'read'},
                  'answer': '7 months', 'requested_output_kind': 'frequency', 'proof': {'selected_observations': [[{'value': '7 months'}]]}}
        self.assertEqual(legacy_contract_review({'operator': 'COUNT', 'answer': '3'}, result)['decision'],
                         'preserve_compatible_legacy_completion')

    def test_entity_matching_does_not_bind_a_name_inside_an_unrelated_word(self):
        a, b = source('Ana has the records.'), source('I ordered bananas.', 2)
        audit = {'identity_constraints': ['Ana'], 'identity_aliases': {'Ana': ['I']},
                 'source_semantic_scores': {'S1:0': .71, 'S2:0': .72}, 'source_min_score': .35}
        selected = relevant_windows(responses(), 'Where is Ana?', [(a, 0, a.text), (b, 0, b.text)], audit)
        self.assertEqual([item[0].source_id for item in selected], ['S1'])

    def test_item_label_cannot_satisfy_a_requested_numeric_measurement(self):
        result = {'source_value_kind': 'quantity', 'contract': {'operation': 'maximum'}, 'answer': '153 points',
                  'requested_projection': 'numeric_value'}
        review = legacy_contract_review({'operator': 'ARGMAX', 'answer': 'the latest round'}, result)
        self.assertEqual(review['decision'], 'replace_incompatible_projection')

    def test_requested_winning_name_is_not_replaced_by_its_numeric_measurement(self):
        result = {'source_value_kind': 'quantity', 'contract': {'operation': 'maximum'}, 'answer': '$800',
                  'requested_projection': 'entity_or_text'}
        self.assertEqual(legacy_contract_review({'operator': 'ARGMAX', 'answer': 'Project Fern'}, result)['decision'],
                         'preserve_compatible_legacy_completion')

    def test_requested_projection_requires_a_literal_query_operand(self):
        with self.assertRaisesRegex(ContractError, 'projection_not_query_bound'):
            requested_output_projection(responses({'request': 'invented quantity', 'description': 'A number.',
                                        'projection': 'numeric_value'}), 'Which task had the largest budget?')

    def test_ambiguous_extremum_projection_preserves_completed_result(self):
        result = {'contract': {'operation': 'minimum'}, 'answer': '18 seconds',
                  'requested_projection': 'unresolved'}
        self.assertEqual(legacy_contract_review({'operator': 'ARGMIN', 'answer': 'the morning attempt'}, result)['decision'],
                         'preserve_compatible_legacy_completion')

    def test_source_absence_does_not_veto_a_completed_legacy_proof(self):
        result = {'proof': {'kind': 'abstention', 'required_identity_unbound': ['nickname']}, 'answer': 'I do not know.'}
        review = legacy_contract_review({'operator': 'SUM', 'answer': '$57'}, result)
        self.assertEqual(review['decision'], 'preserve_compatible_legacy_completion')
        self.assertEqual(review['reason'], 'absence_is_not_contradiction')

    def test_compatible_completed_operation_is_preserved(self):
        result = {'source_representation': 'scalar_history', 'source_value_kind': 'quantity',
                  'contract': {'operation': 'maximum'}, 'answer': '153 points',
                  'requested_projection': 'numeric_value'}
        self.assertEqual(legacy_contract_review({'operator': 'ARGMAX', 'answer': '162 points'}, result)['decision'],
                         'preserve_compatible_legacy_completion')

    def test_abstention_is_not_a_positive_binding_certificate(self):
        result = {'version': 2, 'status': 'COMPLETE', 'answer': 'I do not know.', 'source_view_complete': True,
                  'isolation': 'llama_state_restored_no_external_cache', 'proof': {'kind': 'abstention',
                  'required_identity_unbound': ['Design Director'], 'reviewed_source_ids': ['S1'],
                  'source_witnesses': [{'quote': 'I am a Design Associate.'}]}}
        self.assertFalse(accepted_state_candidate(result, 'candidate'))
        self.assertFalse(accepted_state_candidate(result, 'shadow'))
        result['answer'] = '4'
        self.assertFalse(accepted_state_candidate(result, 'candidate'))


class NativeQualificationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('PRAGMOS_LIVE_STATE_MODEL'), 'Explicit local-model integration test')
    def test_real_changes_and_error_corrections_have_distinct_operator_gates(self):
        import hashlib
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from llama_cpp import Llama
        from PRAGMOS_composable_candidate import _call
        cases = [
            ('progressed from Coral to Jade', 'My badge progressed from Coral to Jade last month, and now it is Onyx.', False),
            ('a temporary transfer increased the team', 'I began with 4 reports, then a temporary transfer increased the team to 7.', False),
            ('the earlier entry was wrong', 'The earlier entry was wrong; the original limit was 240, not 180.', True),
            ('correcting the old note', 'The final certificate lists 320, correcting the old note that said 290.', True),
            ('forwarded the old receipt', 'I forwarded the old receipt showing 11 units; my current stock is 18.', False),
            ('no error was found', 'An audit confirmed both recorded figures; no error was found.', False),
            ('I had 5 and later acquired 3', 'I had 5 and later acquired 3 more, leaving 8.', False),
            ('the original date was a typo', 'The original date was a typo: it should read March 9, not March 7.', True),
        ]
        audit = {'planner_calls': 0, 'extractor_calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        llm = Llama(model_path=os.environ['PRAGMOS_LIVE_STATE_MODEL'], n_ctx=2048, n_threads=8,
                    n_gpu_layers=40, seed=42, verbose=False)
        ctx = SimpleNamespace(llm=llm, count_tokens=lambda s: len(llm.tokenize(s.encode(), add_bos=True, special=True)))
        rows = []
        time_rows = []
        output_rows = []
        try:
            def ask(task, schema, reserve, stage):
                return _call(ctx, '<|user|>\n' + task + '\n<|end|>\n<|assistant|>\n', stage, 2048,
                             audit, schema=schema, max_tokens=reserve)
            for phrase, text, expected in cases:
                result = qualification_kind(ask, phrase, text)
                rows.append({'phrase': phrase, 'source': text, 'expected_retraction_gate': expected,
                             'actual_retraction_gate': result['kind'] == 'report_error', 'result': result})
            for question, expected in [
                ('What day is my ceramics course?', 'latest'),
                ('Where did we spend our most recent weekend getaway?', 'latest'),
                ('Where did I originally store my spare cables?', 'earliest'),
                ('Which colour preceded my present badge?', 'previous'),
                ('What was the reading on 12 March?', 'at_time'),
            ]:
                actual = read_time_selector(ask, question)
                time_rows.append({'question': question, 'expected': expected, 'actual': actual})
            from PRAGMOS_state_claims import requested_output_projection
            for question, expected in [
                ('Which of my projects had the largest budget?', 'entity_or_text'),
                ('What was the largest budget across my projects?', 'numeric_value'),
                ('What is my top score in the puzzle league?', 'numeric_value'),
                ('Who had the most points in the puzzle league?', 'entity_or_text'),
                ('What was my longest recorded distance?', 'numeric_value'),
                ('Which runner recorded the fastest time?', 'entity_or_text'),
                ('What was my smallest monthly expense?', 'numeric_value'),
                ('Which product had the lowest price?', 'entity_or_text'),
            ]:
                output_rows.append({'question': question, 'expected': expected, 'actual': requested_output_projection(ask, question)})
        finally:
            llm.close()
        target = os.environ.get('PRAGMOS_LIVE_STATE_REPORT')
        if target:
            path = Path(target)
            if path.exists():
                self.fail('Use a new native probe report path')
            path.parent.mkdir(parents=True, exist_ok=True)
            code = Path(__file__).with_name('PRAGMOS_state_claims.py')
            path.write_text(json.dumps({'scope': 'local_source_operator_probe_not_benchmark',
                            'source_sha256': hashlib.sha256(code.read_bytes()).hexdigest(),
                            'configuration': {'n_ctx': 2048, 'seed': 42, 'n_threads': 8, 'n_gpu_layers': 40},
                            'rows': rows, 'time_rows': time_rows, 'output_rows': output_rows, 'calls': audit}, indent=2) + '\n')
        self.assertEqual(len(rows), len(cases))
        for row in rows:
            self.assertEqual(row['actual_retraction_gate'], row['expected_retraction_gate'], row)
        for row in time_rows:
            self.assertEqual(row['actual'], row['expected'], row)
        for row in output_rows:
            self.assertEqual(row['actual'], row['expected'], row)


if __name__ == '__main__':
    unittest.main()
