"""Tests for the scoring metrics and their CLI registry."""

from __future__ import annotations

from prompt_eval.metrics import REGISTRY, contains, exact_match


class TestExactMatch:
    def test_identical_strings_score_one(self) -> None:
        assert exact_match("12345", "12345") == 1.0

    def test_is_case_insensitive_and_trims(self) -> None:
        assert exact_match("  AB-9876 \n", "ab-9876") == 1.0

    def test_substring_alone_is_not_a_match(self) -> None:
        # This is the whole point of having a separate `contains` metric.
        assert exact_match("the invoice number is 12345", "12345") == 0.0

    def test_mismatch_scores_zero(self) -> None:
        assert exact_match("7733", "88812") == 0.0


class TestContains:
    def test_expected_substring_anywhere_scores_one(self) -> None:
        assert contains("the invoice number is 12345", "12345") == 1.0

    def test_is_case_insensitive(self) -> None:
        assert contains("Ref AB-9876 follows", "ab-9876") == 1.0

    def test_absent_substring_scores_zero(self) -> None:
        assert contains("no invoice number on this page", "12345") == 0.0

    def test_empty_expected_matches_everything(self) -> None:
        # Documents a sharp edge: an empty `expected` column would score 1.0
        # for every row and silently inflate results.
        assert contains("anything", "") == 1.0


class TestRegistry:
    def test_exposes_both_metrics_by_cli_name(self) -> None:
        assert set(REGISTRY) == {"exact_match", "contains"}

    def test_registry_entries_are_the_metric_callables(self) -> None:
        assert REGISTRY["exact_match"] is exact_match
        assert REGISTRY["contains"] is contains

    def test_all_metrics_return_scores_within_unit_interval(self) -> None:
        cases = [("a", "a"), ("a", "b"), ("abc", "b"), ("", "x")]
        for name, metric in REGISTRY.items():
            for output, expected in cases:
                score = metric(output, expected)
                assert 0.0 <= score <= 1.0, f"{name} returned {score}"
