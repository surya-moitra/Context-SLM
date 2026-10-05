import json
import tempfile
import unittest
from pathlib import Path

import build_PRAGMOS_LongMemEval_regression_suite as builder


class LongMemEvalRegressionSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(
            builder.DEFAULT_SOURCE.read_text(encoding="utf-8")
        )
        cls.selected = builder.select_regression_rows(cls.records)
        cls.validation = builder.validate_selection(cls.records, cls.selected)

    def test_selection_is_deterministic(self):
        self.assertEqual(
            [row["source_index"] for row in self.selected],
            [
                row["source_index"]
                for row in builder.select_regression_rows(self.records)
            ],
        )

    def test_exact_size_and_question_type_targets(self):
        self.assertEqual(len(self.selected), 200)
        self.assertEqual(
            self.validation["question_type_counts"],
            dict(sorted(builder.TYPE_TARGET_COUNTS.items())),
        )

    def test_all_abstentions_and_regression_anchors_are_present(self):
        selected_indices = {row["source_index"] for row in self.selected}
        annotated = builder.annotate_records(self.records)
        abstention_indices = {
            row["source_index"] for row in annotated if row["abstention"]
        }
        required = (
            builder.HISTORICAL_MULTI_FAILURE_INDICES
            | builder.HISTORICAL_UPDATE_FAILURE_INDICES
            | builder.FIX_VALIDATION_ANCHOR_INDICES
        )
        self.assertTrue(abstention_indices.issubset(selected_indices))
        self.assertTrue(required.issubset(selected_indices))
        self.assertEqual(self.validation["abstention_count"], 30)

    def test_every_positive_subfamily_is_covered(self):
        annotated = builder.annotate_records(self.records)
        for question_type in builder.TYPE_TARGET_COUNTS:
            expected = {
                row["family"]
                for row in annotated
                if row["question_type"] == question_type
                and not row["abstention"]
            }
            actual = {
                row["family"]
                for row in self.selected
                if row["question_type"] == question_type
                and not row["abstention"]
            }
            self.assertEqual(actual, expected)

    def test_subset_records_are_unmodified_source_records(self):
        for row in self.selected:
            self.assertEqual(
                row["record"],
                self.records[row["source_index"]],
            )

    def test_manifest_records_hashes_and_source_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "suite.json"
            manifest_path = root / "manifest.json"
            manifest = builder.write_artifacts(
                self.records,
                self.selected,
                self.validation,
                source_path=builder.DEFAULT_SOURCE,
                output_path=output,
                manifest_path=manifest_path,
                seed=builder.DEFAULT_SEED,
            )

            self.assertEqual(
                manifest["output_sha256"],
                builder.sha256_path(output),
            )
            self.assertEqual(
                manifest["source_sha256"],
                builder.sha256_path(builder.DEFAULT_SOURCE),
            )
            self.assertEqual(len(manifest["records"]), 200)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
