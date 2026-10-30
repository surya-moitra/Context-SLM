import json
import re
import unittest
from collections import Counter
from pathlib import Path

from analyze_PRAGMOS_synthetic_LoCoMo_v2 import sha256_path
from generate_PRAGMOS_synthetic_LoCoMo_v2 import validate_dataset


ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT / "benchmark" / "locomo" / "synthetic_v2"
DATA_PATH = V2_DIR / "pragmos_synthetic_locomo_1986_v2.json"
MANIFEST_PATH = V2_DIR / "pragmos_synthetic_locomo_1986_v2_manifest.json"
REPORT_PATH = V2_DIR / "pragmos_synthetic_locomo_1986_v2_semantic_report.json"
SESSION_RE = re.compile(r"session_\d+$")


class SyntheticLoCoMoV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_structure_and_evidence_are_valid(self):
        validation = validate_dataset(self.dataset)
        self.assertEqual(validation["conversation_count"], 10)
        self.assertEqual(validation["question_count"], 1986)
        self.assertEqual(
            validation["category_counts"],
            {1: 282, 2: 321, 3: 96, 4: 841, 5: 446},
        )

    def test_no_generator_internal_fields_remain(self):
        serialized = json.dumps(self.dataset)
        self.assertNotIn('"_facts"', serialized)
        self.assertNotIn('"_topics"', serialized)
        self.assertNotIn('"_target_words"', serialized)

    def test_semantic_difficulty_gate_passes(self):
        self.assertTrue(self.report["comparison"]["passed"])
        self.assertTrue(self.manifest["difficulty_comparison_passed"])

    def test_official_content_overlap_is_zero(self):
        leakage = self.manifest["leakage_checks"]
        self.assertEqual(leakage["exact_official_question_overlap_count"], 0)
        self.assertEqual(leakage["exact_official_long_answer_overlap_count"], 0)
        self.assertEqual(
            leakage["exact_official_turn_overlap_count_min_5_words"], 0
        )

    def test_manifest_hashes_match_generated_artifacts(self):
        self.assertEqual(
            self.manifest["dataset_sha256"], sha256_path(DATA_PATH)
        )
        self.assertEqual(
            self.manifest["semantic_report_sha256"], sha256_path(REPORT_PATH)
        )

    def test_question_families_are_broad(self):
        families = {
            category: Counter(
                qa["synthetic_family"]
                for sample in self.dataset
                for qa in sample["qa"]
                if int(qa["category"]) == category
            )
            for category in range(1, 6)
        }
        self.assertGreaterEqual(len(families[1]), 12)
        self.assertGreaterEqual(len(families[2]), 12)
        self.assertGreaterEqual(len(families[3]), 14)
        self.assertGreaterEqual(len(families[4]), 20)
        self.assertGreaterEqual(len(families[5]), 20)

    def test_turns_and_questions_are_unique(self):
        questions = [
            qa["question"] for sample in self.dataset for qa in sample["qa"]
        ]
        turns = [
            turn["text"]
            for sample in self.dataset
            for key, session in sample["conversation"].items()
            if SESSION_RE.fullmatch(key) and isinstance(session, list)
            for turn in session
        ]
        self.assertEqual(len(questions), len(set(questions)))
        self.assertGreaterEqual(len(set(turns)) / len(turns), 0.99)


if __name__ == "__main__":
    unittest.main()
