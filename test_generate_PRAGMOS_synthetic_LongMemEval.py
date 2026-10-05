import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import generate_PRAGMOS_synthetic_LongMemEval as generator


class SyntheticLongMemEvalTests(unittest.TestCase):
    def sample_profile(self):
        rows = []
        for question_type in sorted(generator.QUESTION_TYPES):
            rows.append(
                {
                    "source_index": len(rows),
                    "question_type": question_type,
                    "abstention": question_type
                    in {
                        "knowledge-update",
                        "multi-session",
                        "single-session-user",
                        "temporal-reasoning",
                    },
                }
            )
            rows.append(
                {
                    "source_index": len(rows),
                    "question_type": question_type,
                    "abstention": False,
                }
            )
        return rows

    def test_generation_is_deterministic(self):
        profile = self.sample_profile()
        first, first_families = generator.generate_records(
            profile,
            seed=73,
            session_min=10,
            session_max=12,
        )
        second, second_families = generator.generate_records(
            profile,
            seed=73,
            session_min=10,
            session_max=12,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_families, second_families)

    def test_profile_type_and_abstention_shape_is_preserved(self):
        profile = self.sample_profile()
        records, _ = generator.generate_records(
            profile,
            seed=19,
            session_min=10,
            session_max=12,
        )
        validation = generator.validate_records(
            records,
            profile,
            official_records=[],
            session_min=10,
            session_max=12,
        )
        expected_types = Counter(row["question_type"] for row in profile)
        expected_abs = Counter(
            row["question_type"] for row in profile if row["abstention"]
        )
        self.assertEqual(validation["question_type_counts"], dict(sorted(expected_types.items())))
        self.assertEqual(validation["abstention_counts"], dict(sorted(expected_abs.items())))

    def test_answer_sessions_are_real_sessions_with_structured_messages(self):
        profile = self.sample_profile()
        records, _ = generator.generate_records(
            profile,
            seed=31,
            session_min=10,
            session_max=12,
        )
        for record in records:
            self.assertTrue(record["answer_session_ids"])
            self.assertTrue(
                set(record["answer_session_ids"]).issubset(record["haystack_session_ids"])
            )
            self.assertEqual(len(record["haystack_sessions"]), len(record["haystack_dates"]))
            for session in record["haystack_sessions"]:
                self.assertEqual([message["role"] for message in session], ["user", "assistant"])

    def test_no_exact_official_question_is_reused(self):
        profile = self.sample_profile()
        records, _ = generator.generate_records(
            profile,
            seed=47,
            session_min=10,
            session_max=12,
        )
        official = [{"question": "What is a deliberately unrelated official question?"}]
        validation = generator.validate_records(
            records,
            profile,
            official_records=official,
            session_min=10,
            session_max=12,
        )
        self.assertEqual(validation["exact_official_question_reuse_count"], 0)

    def test_manifest_records_hashes_and_development_only_status(self):
        profile = self.sample_profile()
        records, families = generator.generate_records(
            profile,
            seed=59,
            session_min=10,
            session_max=12,
        )
        validation = generator.validate_records(
            records,
            profile,
            official_records=[],
            session_min=10,
            session_max=12,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            profile_path = temp / "profile.json"
            output_path = temp / "synthetic.json"
            manifest_path = temp / "manifest.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            manifest = generator.write_artifacts(
                records,
                families,
                validation,
                output_path=output_path,
                manifest_path=manifest_path,
                profile_path=profile_path,
                seed=59,
                session_min=10,
                session_max=12,
            )
            self.assertFalse(manifest["official_longmemeval_score"])
            self.assertEqual(manifest["purpose"], "development_and_regression_testing_only")
            self.assertEqual(manifest["dataset_sha256"], generator.sha256_path(output_path))


if __name__ == "__main__":
    unittest.main()
