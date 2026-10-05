import json
import tempfile
import unittest
from pathlib import Path

import reorder_PRAGMOS_synthetic_LongMemEval as reorder


class ReorderSyntheticLongMemEvalTests(unittest.TestCase):
    def sample_records(self):
        question_types = [
            "single-session-user",
            "multi-session",
            "single-session-preference",
            "multi-session",
            "temporal-reasoning",
        ]
        return [
            {
                "question_id": f"q{index}",
                "question_type": question_type,
                "question": f"Question {index}?",
                "answer": f"Answer {index}",
            }
            for index, question_type in enumerate(question_types)
        ]

    def test_groups_selected_type_stably_without_content_changes(self):
        source = self.sample_records()
        grouped = reorder.stable_question_type_block(
            source,
            question_type="multi-session",
            start_index=1,
        )
        self.assertEqual(
            [record["question_id"] for record in grouped],
            ["q0", "q1", "q3", "q2", "q4"],
        )
        validation = reorder.validate_reorder(
            source,
            grouped,
            question_type="multi-session",
            start_index=1,
        )
        self.assertTrue(validation["content_unchanged"])
        self.assertEqual(validation["block"]["count"], 2)
        self.assertEqual(validation["block"]["start_index_0_based"], 1)
        self.assertEqual(validation["block"]["end_index_inclusive_0_based"], 2)
        self.assertEqual(
            reorder.record_set_sha256(source),
            reorder.record_set_sha256(grouped),
        )

    def test_rejects_a_content_change(self):
        source = self.sample_records()
        grouped = reorder.stable_question_type_block(
            source,
            question_type="multi-session",
            start_index=1,
        )
        grouped[1] = {**grouped[1], "answer": "Changed"}
        with self.assertRaisesRegex(ValueError, "changed record content"):
            reorder.validate_reorder(
                source,
                grouped,
                question_type="multi-session",
                start_index=1,
            )

    def test_manifest_links_source_and_output_hashes(self):
        source = self.sample_records()
        grouped = reorder.stable_question_type_block(
            source,
            question_type="multi-session",
            start_index=1,
        )
        validation = reorder.validate_reorder(
            source,
            grouped,
            question_type="multi-session",
            start_index=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "source.json"
            output_path = temp / "output.json"
            manifest_path = temp / "manifest.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            manifest = reorder.write_artifacts(
                source,
                grouped,
                validation,
                source_path=source_path,
                output_path=output_path,
                manifest_path=manifest_path,
            )
            self.assertEqual(
                manifest["source_dataset_sha256"],
                reorder.sha256_path(source_path),
            )
            self.assertEqual(
                manifest["dataset_sha256"],
                reorder.sha256_path(output_path),
            )
            self.assertEqual(
                manifest["source_record_set_sha256"],
                manifest["validation"]["record_set_sha256"],
            )

    def test_rejects_out_of_range_start_index(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            reorder.stable_question_type_block(
                self.sample_records(),
                question_type="multi-session",
                start_index=4,
            )


if __name__ == "__main__":
    unittest.main()
