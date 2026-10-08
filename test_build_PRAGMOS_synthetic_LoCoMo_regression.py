import json
import unittest
from collections import Counter
from pathlib import Path

import build_PRAGMOS_synthetic_LoCoMo_regression as builder


class SyntheticLoCoMoRegressionSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(
            Path(builder.DEFAULT_SOURCE).read_text(encoding="utf-8")
        )
        cls.first = builder.build_regression_suite(cls.source)
        cls.validation = builder.validate_regression_suite(
            cls.source,
            cls.first,
        )

    def test_selection_is_deterministic(self):
        self.assertEqual(
            self.first,
            builder.build_regression_suite(self.source),
        )

    def test_question_and_category_targets_are_exact(self):
        self.assertEqual(self.validation["question_count"], 600)
        self.assertEqual(
            self.validation["category_counts"],
            builder.EXPECTED_CATEGORY_COUNTS,
        )
        self.assertEqual(
            self.validation["family_counts"],
            builder.FAMILY_TARGETS,
        )

    def test_all_conversations_are_retained_unchanged(self):
        self.assertEqual(len(self.first), len(self.source))
        for source_sample, subset_sample in zip(self.source, self.first):
            self.assertEqual(source_sample["sample_id"], subset_sample["sample_id"])
            self.assertEqual(source_sample["conversation"], subset_sample["conversation"])
            self.assertTrue(subset_sample["qa"])

    def test_selected_questions_are_unmodified_source_records(self):
        source_by_id = {
            qa["question_id"]: qa
            for sample in self.source
            for qa in sample["qa"]
        }
        selected = [qa for sample in self.first for qa in sample["qa"]]
        self.assertEqual(len({qa["question_id"] for qa in selected}), 600)
        for qa in selected:
            self.assertEqual(qa, source_by_id[qa["question_id"]])

    def test_each_family_spans_available_conversations(self):
        source_samples = {}
        selected_samples = {}
        for dataset, target in (
            (self.source, source_samples),
            (self.first, selected_samples),
        ):
            by_family = {}
            for sample in dataset:
                for qa in sample["qa"]:
                    by_family.setdefault(qa["synthetic_family"], set()).add(
                        sample["sample_id"]
                    )
            target.update(by_family)

        for family, target_count in builder.FAMILY_TARGETS.items():
            expected_coverage = min(target_count, len(source_samples[family]))
            self.assertEqual(
                len(selected_samples[family]),
                expected_coverage,
                family,
            )

    def test_category_three_includes_every_available_behavioral_case(self):
        selected = [
            qa
            for sample in self.first
            for qa in sample["qa"]
            if qa["category"] == 3
        ]
        families = Counter(qa["synthetic_family"] for qa in selected)
        self.assertEqual(sum(families.values()), 80)
        self.assertEqual(families["multi-premise-behavioral-inference"], 7)

    def test_high_complexity_extrema_are_preserved(self):
        source_complexity = builder.evidence_complexity(self.source)
        subset_complexity = builder.evidence_complexity(self.first)
        for family in (
            "long-history-list-union",
            "cross-session-relative-date-join",
            "multi-premise-behavioral-inference",
        ):
            self.assertEqual(subset_complexity[family], source_complexity[family])


if __name__ == "__main__":
    unittest.main()
