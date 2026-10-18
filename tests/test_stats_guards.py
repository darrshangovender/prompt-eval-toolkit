"""Input validation and the reporting fields of `compare`."""

from __future__ import annotations

import math

import pytest

from prompt_eval.stats import MIN_BOOTSTRAP, compare


class TestInputValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores_are_rejected_with_their_position(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"candidate_scores\[1\] is not finite"):
            compare([1.0, 0.0], [1.0, bad])
        with pytest.raises(ValueError, match=r"baseline_scores\[0\] is not finite"):
            compare([bad, 0.0], [1.0, 1.0])

    def test_non_numeric_scores_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a number"):
            compare([1.0, "high"], [1.0, 0.0])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="not a number"):
            compare([1.0, None], [1.0, 0.0])  # type: ignore[list-item]

    @pytest.mark.parametrize("b", [0, 1, MIN_BOOTSTRAP - 1, -5, 2.5, True, None])
    def test_n_bootstrap_must_be_an_integer_of_at_least_the_minimum(self, b) -> None:
        with pytest.raises(ValueError, match="n_bootstrap"):
            compare([1.0, 0.0, 1.0], [0.0, 1.0, 1.0], n_bootstrap=b)

    def test_the_minimum_n_bootstrap_is_accepted(self) -> None:
        assert compare([1.0, 0.0, 1.0], [0.0, 1.0, 1.0], n_bootstrap=MIN_BOOTSTRAP).n_bootstrap == 100

    def test_integer_and_bool_scores_are_accepted(self) -> None:
        r = compare([True, False, False], [True, True, True], n_bootstrap=MIN_BOOTSTRAP)
        assert r.delta == pytest.approx(2 / 3)

    def test_tuples_and_other_sequences_are_accepted(self) -> None:
        r = compare((0.0, 1.0, 1.0), (1.0, 1.0, 1.0), n_bootstrap=MIN_BOOTSTRAP)
        assert r.wins == 1 and r.ties == 2


class TestReportedFields:
    def test_wins_losses_and_ties_partition_the_pairs(self) -> None:
        r = compare([0, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 0], n_bootstrap=MIN_BOOTSTRAP)
        assert (r.wins, r.losses, r.ties, r.n) == (2, 1, 3, 6)

    def test_settings_used_are_echoed_so_a_result_can_be_reproduced(self) -> None:
        r = compare([0, 1, 0, 1], [1, 1, 1, 0], n_bootstrap=250, seed=7)
        assert (r.n_bootstrap, r.seed) == (250, 7)
        assert compare([0, 1, 0, 1], [1, 1, 1, 0], n_bootstrap=250, seed=7) == r

    def test_relative_pct_is_zero_not_infinite_when_both_means_are_zero(self) -> None:
        r = compare([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert r.relative_pct == 0.0

    def test_relative_pct_is_signed_infinity_for_a_gain_from_zero(self) -> None:
        assert compare([0.0, 0.0], [1.0, 0.0]).relative_pct == math.inf

    def test_relative_pct_for_a_loss_is_negative(self) -> None:
        r = compare([1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0])
        assert r.relative_pct == pytest.approx(-75.0)

    def test_the_p_value_is_never_exactly_zero_for_a_non_degenerate_sample(self) -> None:
        # 30 wins and 1 loss: no resample is as extreme as the data, but a finite number of
        # resamples cannot prove the null impossible, so the add-one estimate is 1/(B+1).
        r = compare([0.0] * 30 + [1.0], [1.0] * 30 + [0.0], n_bootstrap=999)
        assert (r.wins, r.losses) == (30, 1)
        assert r.p_value == pytest.approx(1 / 1000)

    def test_different_seeds_move_the_p_value_by_monte_carlo_error_only(self) -> None:
        base = [0.0] * 14 + [1.0] * 6 + [0.0] * 20
        cand = [1.0] * 14 + [0.0] * 6 + [0.0] * 20
        ps = [compare(base, cand, n_bootstrap=5000, seed=s).p_value for s in range(8)]
        assert max(ps) - min(ps) < 0.02
        assert len(set(ps)) > 1, "the seed must actually change the resampling"
        verdicts = {compare(base, cand, seed=s).significant for s in range(8)}
        assert len(verdicts) == 1

    def test_a_large_improvement_on_a_sizeable_set_is_significant(self) -> None:
        r = compare([0.0] * 20 + [1.0] * 20, [1.0] * 19 + [0.0] + [1.0] * 20)
        assert r.significant and r.delta == pytest.approx(19 / 40)
        assert r.wins == 19 and r.losses == 0

    def test_a_large_regression_is_significant_with_a_negative_delta(self) -> None:
        r = compare([1.0] * 40, [0.0] * 20 + [1.0] * 20)
        assert r.significant and r.delta == pytest.approx(-0.5)

    def test_an_effect_with_few_discordant_pairs_is_not_called_significant(self) -> None:
        # 5 wins, 0 losses among 60 pairs: the bootstrap alone is tempted (p well under 0.05)
        # but the exact test says 2 * (1/2)^5 = 0.0625.
        r = compare([0.0] * 5 + [1.0] * 55, [1.0] * 5 + [1.0] * 55)
        assert r.p_value <= 0.05 < r.sign_p_value
        assert not r.significant
