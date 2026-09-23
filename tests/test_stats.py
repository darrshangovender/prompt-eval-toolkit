"""Tests for the paired bootstrap comparison.

The bootstrap seeds its RNG with a fixed value, so every assertion here is
exactly reproducible — no tolerance-tuning or flaky thresholds.
"""

from __future__ import annotations

import pytest

from prompt_eval.stats import compare


class TestGuards:
    def test_unpaired_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="paired observations"):
            compare([1.0, 0.0], [1.0])

    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty dataset"):
            compare([], [])


class TestMeansAndDelta:
    def test_reports_means_and_signed_delta(self) -> None:
        result = compare([1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0])
        assert result.baseline_mean == 0.25
        assert result.candidate_mean == 0.75
        assert result.delta == pytest.approx(0.5)

    def test_regression_produces_negative_delta(self) -> None:
        result = compare([1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0])
        assert result.delta == pytest.approx(-0.75)

    def test_relative_pct_is_delta_over_baseline(self) -> None:
        result = compare([0.5, 0.5], [1.0, 1.0])
        assert result.relative_pct == pytest.approx(100.0)

    def test_relative_pct_is_infinite_when_baseline_scores_zero(self) -> None:
        # Guard against a ZeroDivisionError on the common "baseline never got
        # a single row right" case.
        result = compare([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
        assert result.relative_pct == float("inf")


class TestSignificance:
    def test_identical_scores_are_maximally_inconclusive(self) -> None:
        result = compare([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])
        assert result.delta == 0.0
        assert result.p_value == 1.0
        assert result.ci_low == 0.0 and result.ci_high == 0.0

    def test_unanimous_improvement_is_significant(self) -> None:
        scores_n = 20
        result = compare([0.0] * scores_n, [1.0] * scores_n)
        assert result.p_value == 0.0
        assert result.ci_low == 1.0 and result.ci_high == 1.0

    def test_tiny_noisy_sample_is_not_significant(self) -> None:
        # Two rows of coin-flip data must not clear p <= 0.05; if it does,
        # the CLI would tell someone to promote a prompt on no evidence.
        result = compare([1.0, 0.0], [0.0, 1.0])
        assert result.p_value > 0.05

    def test_confidence_interval_brackets_the_observed_delta(self) -> None:
        baseline = [1.0, 0.0] * 15
        candidate = [1.0, 1.0] * 15
        result = compare(baseline, candidate)
        assert result.ci_low <= result.delta <= result.ci_high

    def test_p_value_stays_within_unit_interval(self) -> None:
        result = compare([1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0])
        assert 0.0 <= result.p_value <= 1.0


class TestDeterminism:
    def test_repeated_runs_give_identical_results(self) -> None:
        # The seed is fixed on purpose: a CI gate that flips verdicts between
        # runs on the same data is worse than no gate.
        args = ([1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0])
        first = compare(*args)
        second = compare(*args)
        assert first == second

    def test_more_bootstrap_resamples_still_agree_on_direction(self) -> None:
        args = ([0.0] * 10 + [1.0] * 2, [1.0] * 10 + [0.0] * 2)
        small = compare(*args, n_bootstrap=500)
        large = compare(*args, n_bootstrap=5000)
        assert small.delta == large.delta
        assert (small.p_value <= 0.05) == (large.p_value <= 0.05)


class TestSampleSizeEvidence:
    """The bootstrap has zero variance when every paired difference is identical:
    each resample reproduces the same delta, so p collapses to exactly 0.0 at any
    n. `compare([0.0, 0.0], [1.0, 1.0])` reported p=0.000 and the CLI printed
    'Candidate wins. Promote.' on two rows — the exact failure the tool exists to
    prevent."""

    def test_two_unanimous_rows_are_not_enough_evidence(self) -> None:
        result = compare([0.0, 0.0], [1.0, 1.0])
        assert not result.significant
        assert result.sign_p_value == 0.5

    def test_five_unanimous_rows_are_still_not_enough(self) -> None:
        # Exact two-sided sign test: 2 * (1/2)^5 = 0.0625, just short of 0.05.
        result = compare([0.0] * 5, [1.0] * 5)
        assert not result.significant
        assert result.sign_p_value == pytest.approx(0.0625)

    def test_six_unanimous_rows_clear_the_bar(self) -> None:
        result = compare([0.0] * 6, [1.0] * 6)
        assert result.significant
        assert result.sign_p_value == pytest.approx(0.03125)

    def test_large_unanimous_sample_remains_significant(self) -> None:
        result = compare([0.0] * 20, [1.0] * 20)
        assert result.significant

    def test_identical_scores_are_never_significant(self) -> None:
        result = compare([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])
        assert not result.significant
        assert result.sign_p_value == 1.0

    def test_noisy_large_sample_direction_still_detected(self) -> None:
        # 18 wins, 2 losses over 20 pairs — a genuine effect with real evidence.
        baseline = [0.0] * 18 + [1.0] * 2
        candidate = [1.0] * 18 + [0.0] * 2
        assert compare(baseline, candidate).significant
