import unittest
from unittest.mock import patch

from PRAGMOS_benchmark_LongMemEval import anchor_group_is_supported, build_query_profile
from PRAGMOS_factual_scope import FactualScopeRepair, resource_identity


@unittest.skipUnless(resource_identity()['available'], 'Optional syntax pipeline/WordNet not installed')
class FactualScopeTests(unittest.TestCase):
    def evaluate(self, question, text):
        profile = build_query_profile(question)
        repair = FactualScopeRepair(question, profile['required_anchor_groups'], anchor_group_is_supported)
        return repair(text), repair.decisions[-1]

    def test_functional_heading_and_completed_are_not_names(self):
        accepted, audit = self.evaluate('Under what title did I record my completed river survey work?',
            'I finished the river survey project and gave it the title Silent Reef.')
        self.assertTrue(accepted, audit)
        self.assertFalse(audit['semantic_certification'])

    def test_nominal_event_keeps_its_underlying_object(self):
        q = 'Which person participated in my selection of the camping stove?'
        self.assertTrue(self.evaluate(q, 'I compared camping stove options; Nadia helped me make the final choice.')[0])
        self.assertFalse(self.evaluate(q, 'I compared tent options; Nadia helped me make the final choice.')[0])

    def test_capitalized_discourse_verb_is_not_an_owner(self):
        accepted, audit = self.evaluate('Consulting the records, how many weaving workshops did I complete?',
                                       'I completed 11 separate weaving workshops.')
        self.assertTrue(accepted, audit)

    def test_real_company_named_consulting_is_not_discarded(self):
        self.assertFalse(self.evaluate('What is my completed project for Consulting named?',
                                      'I finished the project for Solstice and named it Ember.')[0])

    def test_real_name_is_not_a_synonym(self):
        self.assertFalse(self.evaluate('Under what title did I record my completed Aster study?',
                                      'I finished the Boreal study and titled it Ember.')[0])

    def test_related_owner_and_entity_polysemy_are_not_aliases(self):
        q = 'Under what title did my uncle record his completed river survey?'
        for s in ['My sister finished her river survey and titled it Ember.',
                  'My pawnbroker finished his river survey and titled it Ember.']:
            with self.subTest(source=s):
                self.assertFalse(self.evaluate(q, s)[0])

    def test_self_is_not_a_different_named_owner(self):
        self.assertFalse(self.evaluate('Which person helped with my selection of the camping stove?',
                                      'Nadia compared camping stove options and helped Arlo make his choice.')[0])

    def test_negated_or_prospective_event_is_not_promoted(self):
        q = 'Under what title did I record my completed river survey work?'
        for s in ['I never finished the river survey project; its title was Ember.',
                  'I plan to finish the river survey project and give it the title Ember.']:
            with self.subTest(source=s):
                self.assertFalse(self.evaluate(q, s)[0])

    def test_habitual_qualifier_is_not_deleted(self):
        for s in ['My unusual journey to Aster Hall took two hours.',
                  'I once took two hours to reach Aster Hall.']:
            self.assertFalse(self.evaluate('How long does my usual journey to Aster Hall take?', s)[0])

    def test_historical_qualifier_is_not_a_discourse_word(self):
        self.assertFalse(self.evaluate('What was my earlier address?', 'My current address is Elm Lane.')[0])

    def test_detached_temporal_event_remains_required(self):
        self.assertFalse(self.evaluate('After kayaking, where did I have my completed lesson?',
                                      'I finished my lesson at Birch Lodge before cycling.')[0])

    def test_detached_creation_event_is_not_treated_as_a_retrieval_preamble(self):
        self.assertFalse(self.evaluate('Painting the mural, where was my completed project shown?',
                                      'I finished my sculpture project and showed it at Birch Lodge.')[0])

    def test_adposition_misparse_outside_question_operator_does_not_remove_an_entity(self):
        self.assertFalse(self.evaluate('What did my completed project for Under earn?',
                                      'I finished my project for Solstice and earned $50.')[0])

    def test_no_independent_target_means_no_repair(self):
        self.assertFalse(self.evaluate('Under what title?', 'Its title is Ember.')[0])

    def test_missing_resources_fail_closed_without_downloading(self):
        with patch('PRAGMOS_factual_scope.resource_identity', return_value={'available': False}):
            self.assertFalse(self.evaluate('Under what title did I record my completed survey?',
                                          'I finished my survey and titled it Ember.')[0])


if __name__ == '__main__':
    unittest.main()
