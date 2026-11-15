import copy
import json
import tempfile
import unittest
from pathlib import Path

from build_PRAGMOS_knowledge_update_pack import (
    DATA_NAME, FIELDS, build_artifacts, build_record, messages, payload, reference_answer, write_artifacts,
)
from PRAGMOS_knowledge_update_pack_cases import CASES, PROFILE_GROUPS


def source_fixture(*, single=False):
    sessions = [[{"role": "user", "content": "Old state."},
                 {"role": "assistant", "content": "Old acknowledgement."},
                 {"role": "user", "content": "Unrelated background."}]
                for _ in range(3)]
    source = {
        "question_id": "old", "question_type": "knowledge-update", "question": "old?",
        "answer": "old", "question_date": "2028/02/01 (Tue) 12:00",
        "haystack_session_ids": ["later", "noise", "earlier"],
        "haystack_dates": ["2028/01/20 (Thu) 12:00", "2028/01/10 (Mon) 12:00", "2028/01/01 (Sat) 12:00"],
        "haystack_sessions": sessions, "answer_session_ids": ["earlier", "later"],
    }
    audit = {"evidence": [{"session_id": "later", "turn_index": 0, "role": "user"}]}
    if not single:
        audit["evidence"].append({"session_id": "earlier", "turn_index": 0, "role": "user"})
    else:
        source["answer_session_ids"] = ["later"]
    return source, audit


class KnowledgeUpdatePackTests(unittest.TestCase):
    def test_complete_unique_profile_inventory(self):
        indices = [index for group in PROFILE_GROUPS.values() for index in group]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(set(indices), set(range(366, 444)))
        selected = [case["source_profile_index"] for case in CASES]
        self.assertEqual(len(selected), 40)
        self.assertEqual(len(set(selected)), 40)
        for name, group in PROFILE_GROUPS.items():
            with self.subTest(profile=name):
                self.assertTrue(set(group).intersection(selected))

    def test_all_authored_cases_have_bound_proofs(self):
        for case in CASES:
            with self.subTest(index=case["source_profile_index"]):
                source, audit = source_fixture()
                original = copy.deepcopy(source)
                record, reference = build_record(source, [audit], case)
                self.assertEqual(source, original)
                self.assertEqual(set(record), FIELDS)
                self.assertEqual(record["answer"], reference_answer(reference["facts"], case["proof"]))
                self.assertEqual(record["haystack_dates"], source["haystack_dates"])
                self.assertEqual(record["haystack_sessions"][1], source["haystack_sessions"][1])
                self.assertEqual([len(s) for s in record["haystack_sessions"]], [3, 3, 3])
                self.assertEqual(reference["episodes"][0]["source_session_index"], 2)
                self.assertEqual(reference["episodes"][1]["source_session_index"], 0)
                self.assertNotIn("answer", " ".join(record["haystack_session_ids"]))
                self.assertTrue(set(record["answer_session_ids"]).issubset(record["haystack_session_ids"]))
                for fact in reference["facts"].values():
                    turn = messages(record["haystack_sessions"][fact["source_session_index"]])[fact["source_turn_index"]]
                    self.assertEqual(turn["content"][fact["quote_start"]:fact["quote_end"]], fact["text"])
                self.assertEqual(record["haystack_sessions"][0][2], source["haystack_sessions"][0][2])

    def test_single_source_turn_is_preserved_with_valid_spans(self):
        for fixture in CASES:
            source, audit = source_fixture(single=True)
            record, reference = build_record(source, [audit], fixture)
            self.assertEqual(len(record["answer_session_ids"]), 1)
            for fact in reference["facts"].values():
                self.assertEqual(record["haystack_sessions"][0][0]["content"][fact["quote_start"]:fact["quote_end"]], fact["text"])

    def test_retires_old_support_without_mutating_source(self):
        source, audit = source_fixture()
        older = copy.deepcopy(audit)
        older["evidence"].append({"session_id": "earlier", "turn_index": 2, "role": "user"})
        result, reference = build_record(source, [older, audit], CASES[0])
        self.assertEqual(len(reference["retired_source_slots"]), 1)
        self.assertNotEqual(result["haystack_sessions"][2][2]["content"], "Unrelated background.")
        self.assertEqual(source["haystack_sessions"][2][2]["content"], "Unrelated background.")

    def test_newest_audit_multiple_turns_in_one_session(self):
        source, audit = source_fixture()
        audit["evidence"].append({"session_id": "earlier", "turn_index": 2, "role": "user"})
        _, reference = build_record(source, [audit], CASES[0])
        self.assertEqual(len(reference["retired_source_slots"]), 1)
        self.assertEqual(reference["episodes"][0]["source_turn_index"], 0)

    def test_restores_a_missing_history_session_without_adding_turns(self):
        source, audit = source_fixture()
        audit["evidence"] = [audit["evidence"][0]]
        record, reference = build_record(source, [audit], CASES[0])
        self.assertEqual(len(reference["restored_support_slots"]), 1)
        self.assertEqual(len(record["answer_session_ids"]), 2)
        self.assertEqual([len(s) for s in record["haystack_sessions"]], [3, 3, 3])

    def test_authored_gold_calculations(self):
        expected = {366: "26 minutes 15 seconds", 367: "4", 373: "3", 379: "Initially 6; now 8",
                    389: "less", 406: "142 points", 421: "2", 422: "6 kg", 424: "5-2", 432: "2"}
        for fixture in CASES:
            index = fixture["source_profile_index"]
            if index in expected:
                source, audit = source_fixture()
                record, _ = build_record(source, [audit], fixture)
                self.assertEqual(record["answer"], expected[index])

    def test_six_distinct_unsupported_scopes(self):
        absent = [case for case in CASES if case["proof"]["op"] == "absent"]
        self.assertEqual(len(absent), 6)
        self.assertEqual(len({case["proof"]["missing_anchor"] for case in absent}), 6)

    def test_rejects_wrong_role(self):
        source, audit = source_fixture()
        source["haystack_sessions"][0][0]["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "not a user turn"):
            build_record(source, [audit], CASES[0])

    def test_rejects_future_source(self):
        source, audit = source_fixture()
        source["question_date"] = "2027/01/01 (Fri) 12:00"
        with self.assertRaisesRegex(ValueError, "after"):
            build_record(source, [audit], CASES[0])

    def test_rejects_unknown_reference_and_duplicate_fact(self):
        source, audit = source_fixture()
        fixture = copy.deepcopy(CASES[0])
        fixture["proof"]["ids"] = ["MISSING"]
        with self.assertRaisesRegex(ValueError, "Unbound"):
            build_record(source, [audit], fixture)
        fixture = copy.deepcopy(CASES[0])
        fixture["episodes"][1].append(fixture["episodes"][0][1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_record(source, [audit], fixture)

    def test_inference_schema_rejects_privileged_metadata(self):
        source, audit = source_fixture()
        source["reference"] = {"op": "pick"}
        with self.assertRaisesRegex(ValueError, "schema"):
            build_record(source, [audit], CASES[0])

    def test_frozen_outputs_are_idempotent_and_preflighted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {"data.json": payload([1]), "evaluation_only/audit.json": payload({"valid": True})}
            write_artifacts(root, artifacts)
            write_artifacts(root, artifacts, check_only=True)
            changed = {"new.json": payload([2]), "data.json": payload([3])}
            with self.assertRaisesRegex(ValueError, "Frozen"):
                write_artifacts(root, changed)
            self.assertFalse((root / "new.json").exists())
            self.assertEqual(json.loads((root / "data.json").read_text()), [1])

    def test_check_only_never_creates_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing"):
                write_artifacts(Path(directory), {"new.json": b"[]"}, check_only=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_real_source_inventory_and_history_preservation(self):
        project = Path("/Users/moitra/surya/dev_project/context")
        source_path = project / "benchmark/longMemEval/synthetic_v2/semantic_revision_2/pragmos_synthetic_longmemeval_500_v2_semantic_r2.json"
        if not source_path.exists():
            self.skipTest("Local source dataset is not installed")
        source = json.loads(source_path.read_text())
        artifacts, manifest = build_artifacts(project)
        records = json.loads(artifacts[DATA_NAME])
        audits = json.loads(artifacts["evaluation_only/case_audit.json"])
        inventory = json.loads(artifacts["evaluation_only/coverage_inventory.json"])
        self.assertEqual(len(records), 40)
        self.assertEqual(manifest["historical_success_profile_count"], 21)
        self.assertEqual(sum(r["historical_official_judge_correct"] for r in inventory), 35)
        success_types = {r["primary_profile"] for r in inventory if r["historical_official_judge_correct"]}
        controls = {r["primary_profile"] for r in inventory if r["selected"] and r["historical_official_judge_correct"]}
        self.assertEqual(success_types, controls)
        for record, audit in zip(records, audits):
            with self.subTest(question=record["question_id"]):
                original = source[audit["source_profile_index"]]
                self.assertEqual(len(record["answer_session_ids"]), 2)
                changed = {(e["source_session_index"], e["source_turn_index"]) for e in audit["episodes"]}
                for e in audit["episodes"]:
                    if e["neutralized_assistant_turn"] is not None:
                        changed.add((e["source_session_index"], e["neutralized_assistant_turn"]))
                for e in audit["retired_source_slots"]:
                    si, ti = e["source_session_index"], e["source_turn_index"]
                    changed.add((si, ti))
                    old_turns = messages(original["haystack_sessions"][si])
                    if ti + 1 < len(old_turns) and old_turns[ti + 1]["role"] == "assistant":
                        changed.add((si, ti + 1))
                self.assertEqual(record["haystack_dates"], original["haystack_dates"])
                self.assertEqual(len(record["haystack_sessions"]), len(original["haystack_sessions"]))
                for si, (before, after) in enumerate(zip(original["haystack_sessions"], record["haystack_sessions"])):
                    self.assertEqual(len(messages(before)), len(messages(after)))
                    for ti, (old, new) in enumerate(zip(messages(before), messages(after))):
                        self.assertEqual(old["role"], new["role"])
                        if (si, ti) not in changed:
                            self.assertEqual(old, new)
                for fact in audit["facts"].values():
                    turn = messages(record["haystack_sessions"][fact["source_session_index"]])[fact["source_turn_index"]]
                    self.assertEqual(turn["content"][fact["quote_start"]:fact["quote_end"]], fact["text"])


if __name__ == "__main__":
    unittest.main()
