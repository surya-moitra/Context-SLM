import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import generate_PRAGMOS_synthetic_LoCoMo as generator


class SyntheticLoCoMoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generator.generate_dataset(seed=generator.DEFAULT_SEED)
        cls.validation = generator.validate_dataset(cls.dataset)

    def test_generation_is_deterministic(self):
        first = generator.generate_dataset(seed=91)
        second = generator.generate_dataset(seed=91)

        self.assertEqual(first, second)

    def test_official_conversation_and_category_profile_is_preserved(self):
        self.assertEqual(len(self.dataset), 10)
        self.assertEqual(self.validation["question_count"], 1986)
        self.assertEqual(
            self.validation["category_counts"],
            generator.EXPECTED_CATEGORY_COUNTS,
        )
        for sample, profile in zip(self.dataset, generator.CONVERSATION_PROFILES):
            counts = Counter(qa["category"] for qa in sample["qa"])
            turn_count = sum(
                len(turns)
                for key, turns in sample["conversation"].items()
                if key.startswith("session_")
                and not key.endswith("_date_time")
            )
            self.assertEqual(turn_count, profile.target_turn_count)
            self.assertEqual(
                {
                    category: counts.get(category, 0)
                    for category in generator.CATEGORY_NAMES
                },
                profile.category_counts,
            )

    def test_dialog_ids_and_evidence_are_valid(self):
        for sample in self.dataset:
            conversation = sample["conversation"]
            dialog_ids = {
                turn["dia_id"]
                for key, turns in conversation.items()
                if key.startswith("session_")
                and not key.endswith("_date_time")
                for turn in turns
            }
            for qa in sample["qa"]:
                self.assertTrue(qa["evidence"])
                self.assertTrue(set(qa["evidence"]).issubset(dialog_ids))

    def test_multi_hop_is_cross_session_and_single_hop_is_singular(self):
        for sample in self.dataset:
            for qa in sample["qa"]:
                evidence_sessions = {
                    evidence_id.split(":", 1)[0] for evidence_id in qa["evidence"]
                }
                if qa["category"] == 1:
                    self.assertGreaterEqual(len(qa["evidence"]), 2)
                    self.assertGreaterEqual(len(evidence_sessions), 2)
                elif qa["category"] == 4:
                    self.assertEqual(len(qa["evidence"]), 1)

    def test_adversarial_answers_are_null_and_other_answers_are_present(self):
        for sample in self.dataset:
            for qa in sample["qa"]:
                if qa["category"] == 5:
                    self.assertIsNone(qa["answer"])
                    self.assertIn(
                        qa["synthetic_family"],
                        {
                            "adversarial-wrong-attribute",
                            "adversarial-wrong-speaker",
                        },
                    )
                else:
                    self.assertNotIn(qa["answer"], {None, ""})

    def test_caption_grounded_and_failure_pressure_families_exist(self):
        families = Counter(
            qa["synthetic_family"]
            for sample in self.dataset
            for qa in sample["qa"]
        )

        self.assertGreater(families["visual-caption-single-hop"], 0)
        self.assertGreater(families["cross-session-list-union"], 0)
        self.assertGreater(families["cross-session-count-distinct"], 0)
        self.assertGreater(families["speaker-set-intersection"], 0)
        self.assertGreater(families["long-history-list-union"], 0)
        self.assertGreater(families["cross-session-relative-date-join"], 0)
        self.assertGreater(families["multi-premise-behavioral-inference"], 0)
        self.assertGreater(families["relative-weekday"], 0)
        self.assertGreater(families["geographic-country-inference"], 0)
        self.assertGreater(families["adversarial-wrong-attribute"], 0)
        self.assertGreater(families["adversarial-wrong-speaker"], 0)

        rows = [qa for sample in self.dataset for qa in sample["qa"]]
        self.assertEqual(
            max(
                len(qa["evidence"])
                for qa in rows
                if qa["synthetic_family"] == "long-history-list-union"
            ),
            19,
        )
        self.assertEqual(
            max(
                len(qa["evidence"])
                for qa in rows
                if qa["synthetic_family"]
                == "multi-premise-behavioral-inference"
            ),
            17,
        )
        self.assertEqual(
            max(
                len(qa["evidence"])
                for qa in rows
                if qa["synthetic_family"]
                == "cross-session-relative-date-join"
            ),
            4,
        )

    def test_exact_official_question_reuse_is_rejected(self):
        synthetic_question = self.dataset[0]["qa"][0]["question"]
        official = [{"qa": [{"question": synthetic_question}]}]

        with self.assertRaisesRegex(ValueError, "official questions were reused"):
            generator.validate_dataset(
                self.dataset,
                official_dataset=official,
            )

    def test_manifest_records_hashes_and_development_only_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "synthetic.json"
            manifest_path = root / "manifest.json"
            manifest = generator.write_artifacts(
                self.dataset,
                self.validation,
                output_path=output,
                manifest_path=manifest_path,
                seed=generator.DEFAULT_SEED,
            )

            self.assertFalse(manifest["official_locomo_score"])
            self.assertEqual(
                manifest["purpose"],
                "development_and_regression_testing_only",
            )
            self.assertEqual(
                manifest["dataset_sha256"],
                generator.sha256_path(output),
            )
            self.assertFalse(
                manifest["official_question_reuse_check"]["performed"]
            )
            self.assertIsNone(
                manifest["official_question_reuse_check"][
                    "official_data_sha256"
                ]
            )
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )

    def test_manifest_records_official_reuse_check_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            official_path = root / "official.json"
            official_path.write_text("[]\n", encoding="utf-8")
            manifest = generator.write_artifacts(
                self.dataset,
                self.validation,
                output_path=root / "synthetic.json",
                manifest_path=root / "manifest.json",
                seed=generator.DEFAULT_SEED,
                official_data_path=official_path,
            )

            reuse_check = manifest["official_question_reuse_check"]
            self.assertTrue(reuse_check["performed"])
            self.assertEqual(
                reuse_check["official_data_sha256"],
                generator.sha256_path(official_path),
            )
            self.assertEqual(reuse_check["matched_question_count"], 0)


if __name__ == "__main__":
    unittest.main()
