"""The statistics, checked against answers that do not come from this code.

Three kinds of reference:

* closed forms (an exact binomial tail worked by hand, the unanimous case, and the *exact* bootstrap
  distribution of a five-row example by enumerating all 3,125 resamples);
* scipy, an independent implementation: the same estimator through `scipy.stats.bootstrap`, the
  paired t-test, a sign-flip permutation test, the exact binomial test;
* calibration: simulate datasets where the two prompts are truly equal and count false alarms.

Monte Carlo tolerances are stated as multiples of the standard error, not tuned to pass: everything
is seeded, so a failure here is a real disagreement, not a flake.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from scipy import stats as sps

from benchmarks.verify_stats import exact_bootstrap_p, simulate
from prompt_eval.stats import ALPHA, compare, sign_test_p


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-4) / n)


class TestKnownAnswers:
    def test_sign_test_ten_wins_two_losses_is_158_over_4096(self) -> None:
        # 2 * (C(12,10) + C(12,11) + C(12,12)) / 2**12 = 2 * 79 / 4096, worked by hand.
        assert sign_test_p(10, 2) == pytest.approx(158 / 4096, abs=1e-15)
        assert sign_test_p(10, 2) == pytest.approx(sps.binomtest(10, 12, 0.5).pvalue, abs=1e-12)

    @pytest.mark.parametrize("wins,losses", [(0, 0), (1, 0), (0, 5), (3, 3), (7, 2), (30, 11), (2, 40)])
    def test_sign_test_matches_scipy_binomtest(self, wins: int, losses: int) -> None:
        expected = 1.0 if wins + losses == 0 else sps.binomtest(max(wins, losses), wins + losses).pvalue
        assert sign_test_p(wins, losses) == pytest.approx(expected, abs=1e-12)

    def test_sign_test_is_symmetric_and_capped_at_one(self) -> None:
        assert sign_test_p(4, 9) == sign_test_p(9, 4)
        assert sign_test_p(5, 5) == 1.0

    @pytest.mark.parametrize("n", [1, 2, 3, 6, 10, 20])
    def test_unanimous_wins_give_the_exact_sign_flip_p_value(self, n: int) -> None:
        # Only the all-plus and all-minus sign patterns are as extreme: 2 / 2**n.
        assert compare([0.0] * n, [1.0] * n).p_value == pytest.approx(min(1.0, 2.0 ** (1 - n)))

    @pytest.mark.parametrize(
        "base,cand",
        [
            ([0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]),
            ([0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0]),
            ([0.2, 0.9, 0.4, 0.5, 0.7], [0.5, 0.95, 0.9, 0.55, 0.6]),
        ],
    )
    def test_bootstrap_p_matches_the_exact_bootstrap_distribution(self, base, cand) -> None:
        """For n = 5 all 5**5 resamples can be listed, giving the exact bootstrap p-value."""
        diffs = [c - b for b, c in zip(base, cand, strict=True)]
        exact = exact_bootstrap_p(diffs)
        b = 100_000
        got = compare(base, cand, n_bootstrap=b).p_value
        # (1 + extreme) / (B + 1) estimates the exact value to within 4 Monte Carlo SEs plus 1/(B+1).
        assert abs(got - exact) < 4 * _se(exact, b) + 1 / (b + 1)

    def test_two_row_bootstrap_distribution_is_known_in_closed_form(self) -> None:
        """d = [1, 0]: centred rows are +-0.5, so a resampled mean is +0.5, 0, -0.5 with
        probabilities 1/4, 1/2, 1/4 and P(|mean| >= 0.5) = 1/2. The sign floor then refuses it."""
        r = compare([0.0, 1.0], [1.0, 1.0], n_bootstrap=100_000)
        assert r.p_value == pytest.approx(0.5, abs=4 * _se(0.5, 100_000))
        assert not r.significant


def _scipy_p(diffs: np.ndarray, resamples: int, seed: int) -> float:
    """The same estimator through scipy's own bootstrap machinery."""
    delta = float(diffs.mean())
    res = sps.bootstrap((diffs - delta,), np.mean, n_resamples=resamples, method="percentile",
                        random_state=np.random.default_rng(seed))
    dist = res.bootstrap_distribution
    return (1 + float(np.sum(np.abs(dist) >= abs(delta) * (1 - 1e-12)))) / (len(dist) + 1)


def _datasets() -> dict[str, tuple[list[float], list[float]]]:
    rng = np.random.default_rng(2024)
    out = {}
    for n in (12, 30, 60):
        b = (rng.random(n) < 0.55).astype(float)
        keep = rng.random(n) < 0.75
        c = np.where(keep, b, (rng.random(n) < 0.8).astype(float))
        out[f"0/1 n={n}"] = (list(b), list(c))
        b = rng.normal(0.6, 0.2, n).clip(0, 1)
        c = (b + rng.normal(0.05, 0.15, n)).clip(0, 1)
        out[f"continuous n={n}"] = (list(b), list(c))
    return out


class TestAgainstScipy:
    @pytest.mark.parametrize("name", list(_datasets()))
    def test_p_value_matches_scipys_bootstrap_of_the_same_estimator(self, name: str) -> None:
        base, cand = _datasets()[name]
        diffs = np.asarray(cand) - np.asarray(base)
        if not diffs.any() or len(set(diffs.tolist())) == 1:
            pytest.skip("degenerate differences use the exact sign-flip value, covered elsewhere")
        ours = compare(base, cand, n_bootstrap=20_000).p_value
        theirs = _scipy_p(diffs, 100_000, seed=7)
        # SE of a difference of two independent Monte Carlo estimates, at the worst case p = 0.5.
        sd = math.sqrt(0.25 / 20_000 + 0.25 / 100_000)
        assert abs(ours - theirs) < 4 * sd

    @pytest.mark.parametrize("name", list(_datasets()))
    def test_confidence_interval_matches_scipys_percentile_bootstrap(self, name: str) -> None:
        base, cand = _datasets()[name]
        diffs = np.asarray(cand) - np.asarray(base)
        ours = compare(base, cand, n_bootstrap=20_000)
        theirs = sps.bootstrap((diffs,), np.mean, n_resamples=100_000, confidence_level=0.95,
                               method="percentile", random_state=np.random.default_rng(8)
                               ).confidence_interval
        # Quantile Monte Carlo error is about SE(mean)/sqrt(B) times a tail factor; for these data
        # sets SE(mean) <= 0.15, so 0.015 is several standard errors, and below any grid step that
        # matters for a delta.
        assert ours.ci_low == pytest.approx(theirs.low, abs=0.015)
        assert ours.ci_high == pytest.approx(theirs.high, abs=0.015)

    def test_a_large_continuous_sample_agrees_with_ttest_and_permutation(self) -> None:
        """Three different tests of the same hypothesis converge as n grows."""
        rng = np.random.default_rng(99)
        n = 200
        base = rng.normal(0.6, 0.2, n).clip(0, 1)
        cand = (base + rng.normal(0.02, 0.15, n)).clip(0, 1)
        ours = compare(list(base), list(cand), n_bootstrap=20_000).p_value
        t = sps.ttest_rel(cand, base).pvalue
        perm = sps.permutation_test((base, cand), lambda x, y: np.mean(y - x),
                                    permutation_type="samples", n_resamples=20_000,
                                    random_state=np.random.default_rng(5)).pvalue
        assert ours == pytest.approx(t, abs=0.03)
        assert ours == pytest.approx(perm, abs=0.03)

    def test_scores_that_differ_by_a_constant_agree_with_ttest_direction_and_scale(self) -> None:
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, 80)
        cand = base + 0.5 + rng.normal(0, 0.2, 80)
        r = compare(list(base), list(cand), n_bootstrap=5000)
        assert r.delta == pytest.approx(float((cand - base).mean()), abs=1e-12)
        assert r.p_value < 0.001 and sps.ttest_rel(cand, base).pvalue < 0.001


class TestCalibration:
    """When the prompts are truly equal, p <= 0.05 must happen at most about 5% of the time."""

    TRIALS = 600
    BOOT = 399

    def test_01_scores_hold_the_level_with_and_without_the_sign_floor(self) -> None:
        rates = simulate("ternary", 40, 0.0, self.TRIALS, self.BOOT, seed=17)
        bound = ALPHA + 3 * _se(ALPHA, self.TRIALS)
        assert rates["bootstrap"] <= bound, rates
        assert rates["bootstrap+sign (significant)"] <= bound, rates
        assert rates["bootstrap+sign (significant)"] <= rates["bootstrap"]

    def test_continuous_scores_hold_the_level_through_the_gate(self) -> None:
        rates = simulate("normal", 30, 0.0, self.TRIALS, self.BOOT, seed=18)
        assert rates["bootstrap+sign (significant)"] <= ALPHA + 3 * _se(ALPHA, self.TRIALS), rates

    def test_the_gate_is_what_protects_tiny_samples(self) -> None:
        """At n = 10 the bootstrap alone over-rejects on continuous scores; the gate fixes it."""
        rates = simulate("normal", 10, 0.0, self.TRIALS, self.BOOT, seed=19)
        bound = ALPHA + 3 * _se(ALPHA, self.TRIALS)
        assert rates["bootstrap"] > ALPHA, "if this stops holding, the README claim must change"
        assert rates["bootstrap+sign (significant)"] <= bound, rates

    def test_a_strong_effect_is_detected(self) -> None:
        # P(win) = 0.60, P(loss) = 0.10 on 40 pairs: far beyond the noise.
        rates = simulate("ternary", 40, 0.50, 200, self.BOOT, seed=20)
        assert rates["bootstrap+sign (significant)"] >= 0.95, rates

    def test_a_one_sided_p_value_would_fail_the_level_check(self) -> None:
        """Guards the test itself: halving the p-value (a one-sided test reported as two-sided)
        must push the false-alarm rate past the bound, otherwise the bound could not catch it."""
        rng = random.Random(21)
        n, trials, boot = 40, self.TRIALS, self.BOOT
        halved = 0
        for _ in range(trials):
            diffs = [1.0 if (u := rng.random()) < 0.1 else -1.0 if u < 0.2 else 0.0
                     for _ in range(n)]
            r = compare([0.0] * n, diffs, n_bootstrap=boot, seed=rng.randrange(2**31))
            halved += r.p_value / 2 <= ALPHA
        assert halved / trials > ALPHA + 3 * _se(ALPHA, trials)
