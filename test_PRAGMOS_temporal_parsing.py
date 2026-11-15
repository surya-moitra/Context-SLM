import unittest
from unittest import mock

from PRAGMOS_benchmark_LongMemEval import resolve_temporal_expression
from PRAGMOS_evidence_compiler import build_evidence_ledger


class TemporalParsingBoundsTests(unittest.TestCase):
    def test_bce_relative_years_are_unannotated_not_clamped(self):
        for years in (3000, 4000, 4500, 450000):
            with self.subTest(years=years):
                self.assertIsNone(resolve_temporal_expression(
                    f"The formation occurred {years:,} years ago.", "2028-02-11",
                ))

    def test_large_offsets_in_all_supported_relative_units(self):
        for unit in ("days", "weeks", "months", "years"):
            for quantity in (1000000000, "9" * 400):
                with self.subTest(unit=unit, quantity=str(quantity)[:12]):
                    self.assertIsNone(resolve_temporal_expression(
                        f"The event occurred {quantity} {unit} ago.", "2028-02-11",
                    ))

    def test_relative_expressions_near_calendar_lower_bound(self):
        for expression in ("yesterday", "last week", "last month", "last year",
                           "one day ago", "one week ago", "one month ago", "one year ago",
                           "last Monday"):
            with self.subTest(expression=expression):
                self.assertIsNone(resolve_temporal_expression(expression, "0001-01-01"))

    def test_relative_expressions_near_calendar_upper_bound(self):
        for expression in ("tomorrow", "next week", "next month", "next year"):
            with self.subTest(expression=expression):
                self.assertIsNone(resolve_temporal_expression(expression, "9999-12-31"))

    def test_invalid_explicit_calendar_years_do_not_crash(self):
        for expression in ("in 0000", "since 0000", "January 0000"):
            with self.subTest(expression=expression):
                self.assertIsNone(resolve_temporal_expression(expression, "2028-02-11"))

    def test_ordinary_relative_results_are_unchanged(self):
        cases = (
            ("today", "day", "2028-02-11", "2028-02-11", "2028-02-11"),
            ("yesterday", "day", "2028-02-10", "2028-02-10", "2028-02-10"),
            ("tomorrow", "day", "2028-02-12", "2028-02-12", "2028-02-12"),
            ("three days ago", "day", "2028-02-08", "2028-02-08", "2028-02-08"),
            ("two weeks ago", "week", "2028-W04", "2028-01-24", "2028-01-30"),
            ("two months ago", "month", "2027-12", "2027-12-01", "2027-12-31"),
            ("two years ago", "year", "2026", "2026-01-01", "2026-12-31"),
            ("last week", "week", "2028-W05", "2028-01-31", "2028-02-06"),
            ("next month", "month", "2028-03", "2028-03-01", "2028-03-31"),
            ("last year", "year", "2027", "2027-01-01", "2027-12-31"),
        )
        for text, granularity, value, start, end in cases:
            with self.subTest(text=text):
                result = resolve_temporal_expression(text, "2028-02-11")
                self.assertEqual(result["granularity"], granularity)
                self.assertEqual(result["value"], value)
                self.assertEqual(result["interval_start"], start)
                self.assertEqual(result["interval_end"], end)

    def test_ordinary_explicit_results_and_leap_months_are_preserved(self):
        for text, expected in (("on 2027-04-03", "2027-04-03"),
                               ("on 3 January 2027", "2027-01-03"),
                               ("in February 2028", "2028-02"),
                               ("in 2026", "2026")):
            with self.subTest(text=text):
                self.assertEqual(resolve_temporal_expression(text, "2028-02-11")["value"], expected)
        result = resolve_temporal_expression("next month", "2028-01-31")
        self.assertEqual(result["interval_end"], "2028-02-29")

    def test_valid_early_years_are_not_rejected_by_an_arbitrary_age_cap(self):
        result = resolve_temporal_expression("two thousand years ago", "2028-02-11")
        self.assertEqual(result["value"], "28")
        self.assertEqual(result["interval_start"], "0028-01-01")
        result = resolve_temporal_expression("one year ago", "0002-01-01")
        self.assertEqual(result["interval_start"], "0001-01-01")

    def test_non_integral_offsets_and_missing_timestamps_stay_unresolved(self):
        self.assertIsNone(resolve_temporal_expression("1.5 years ago", "2028-02-11"))
        self.assertIsNone(resolve_temporal_expression("two years ago", None))
        self.assertIsNone(resolve_temporal_expression("two years ago", "not a timestamp"))

    def test_ledger_retains_unsupported_text_and_other_valid_temporal_atoms(self):
        text = "The glacier formed 450,000 years ago; I finished the workshop yesterday."
        ledger = build_evidence_ledger([{
            "source_quote": text, "source_timestamp": "2028-02-11",
            "source_turn_id": "t1", "source_session_id": "s1", "source_role": "user",
        }], temporal_resolver=resolve_temporal_expression)
        self.assertEqual(len(ledger), 2)
        self.assertIsNone(ledger[0].temporal_expression)
        self.assertEqual(ledger[0].source_quote, text)
        self.assertEqual(ledger[0].clause, "The glacier formed 450,000 years ago")
        self.assertEqual(ledger[1].temporal_expression["value"], "2028-02-10")
        self.assertEqual(ledger[1].source_turn_id, "t1")

    def test_non_parse_programming_errors_are_not_suppressed(self):
        for error in (TypeError("unexpected type"), RuntimeError("unexpected bug")):
            with self.subTest(error=type(error).__name__):
                with mock.patch("PRAGMOS_benchmark_LongMemEval._resolve_temporal_expression", side_effect=error):
                    with self.assertRaises(type(error)):
                        resolve_temporal_expression("yesterday", "2028-02-11")


if __name__ == "__main__":
    unittest.main()
