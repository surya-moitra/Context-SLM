import copy
import unittest

from PRAGMOS_replay_temporal_queries import replay_sources


class ReplaySourceTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(question_id="q", question="When did I repair the kiln?",
                           haystack_session_ids=["s1", "s2"],
                           haystack_dates=["2031-06-01", "2031-06-20"],
                           haystack_sessions=[[dict(role="user", content="I bought a mug.")],
                                              [dict(role="user", content="I repaired the kiln on 3 June.")]],
                           answer="3 June", answer_session_ids=["s2"])
        self.trace = dict(question=self.record["question"], evidence_compiler_ledger=[
            dict(source_turn_id=1, source_session_id="s1", source_quote="I bought a mug.")])

    def test_replay_never_selects_gold_session(self):
        before = copy.deepcopy(self.record)
        rows, complete = replay_sources(self.trace, self.record)
        self.assertEqual([row["source_session_id"] for row in rows], ["s1"])
        self.assertFalse(complete)
        self.assertEqual(self.record, before)

    def test_changed_question_is_rejected(self):
        self.record["question"] = "What did I buy?"
        with self.assertRaisesRegex(ValueError, "question_changed"):
            replay_sources(self.trace, self.record)

    def test_changed_source_is_rejected(self):
        self.record["haystack_sessions"][0][0]["content"] = "I bought a bowl."
        with self.assertRaisesRegex(ValueError, "source_content_changed"):
            replay_sources(self.trace, self.record)

    def test_foreign_session_cannot_reuse_a_valid_turn_id(self):
        self.trace["evidence_compiler_ledger"][0]["source_session_id"] = "s2"
        with self.assertRaisesRegex(ValueError, "identity_invalid"):
            replay_sources(self.trace, self.record)

    def test_empty_retrieval_does_not_fall_back_to_gold(self):
        self.trace["evidence_compiler_ledger"] = []
        rows, complete = replay_sources(self.trace, self.record)
        self.assertEqual(rows, [])
        self.assertFalse(complete)


if __name__ == "__main__":
    unittest.main()
