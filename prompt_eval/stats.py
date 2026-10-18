"""Paired bootstrap comparison of two prompts scored on the same examples.

Pure standard library (no numpy/scipy at run time; scipy is a *dev* dependency, used only to check
this module against an independent implementation, see `tests/test_stats_scipy.py` and
`benchmarks/verify_stats.py`).

What is computed, from the paired differences ``d_i = candidate_i - baseline_i``:

* **Confidence interval**: the percentile bootstrap interval of ``mean(d)`` (resample the *pairs*
  with replacement, take the 2.5th and 97.5th percentiles of the resampled means).
* **p-value**: a bootstrap *hypothesis test* of "the mean difference is zero". The differences are
  centred (``d_i - mean(d)``) so the resampling distribution obeys the null, and
  ``p = (1 + #{|mean(d*) - mean(d)| >= |mean(d)|}) / (B + 1)``. The ``+1`` terms (Davison and
  Hinkley, 1997, sec. 4.4) make the p-value a valid, never-zero estimate. Both come from the same
  resamples: the mean of a centred resample is the mean of the raw resample minus ``mean(d)``.
* **Sign test**: the exact two-sided sign test (the exact McNemar test when scores are 0/1) on the
  pairs that differ. `CompareResult.significant` needs this *and* the bootstrap to clear ALPHA.

Why both. The bootstrap is an approximation that leans on the sample being large enough to look
like its population. With a handful of pairs it is not. In the extreme, when every difference is
identical (the candidate fixes every case the baseline failed, on two rows) every centred resample
is exactly zero and a naive bootstrap reports p = 0. We special-case that to the exact sign-flip
p-value ``2^(1-n)`` and keep the sign test as a floor on how much the bootstrap may claim. The price
is power at small and medium n (see the README table); the benefit is a gate that cannot promote a
prompt on two rows.

Why a bootstrap and not a t-test. Scores here are usually 0/1, so the differences take only the
values -1, 0 and +1 and are far from normal. In the simulations in the README the null-centred
bootstrap and the paired t-test behave alike (similar false-alarm and detection rates once n is
40 or more); the bootstrap's advantage is that it needs no normality assumption and works unchanged
for any metric that returns a number, not that it beats the t-test. Below about n = 20, and for
skewed continuous scores, neither holds the 5% level on its own: see the README tables.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

#: Conventional significance level; `CompareResult.significant` is the gate the CLI acts on.
ALPHA = 0.05

#: Fewest resamples accepted. Fewer than this cannot resolve the 2.5% tails of the interval.
MIN_BOOTSTRAP = 100

#: Relative tolerance for "at least as extreme" comparisons, so float rounding in a resampled mean
#: (0.30000000000000004 vs 0.3) cannot turn a tie into a miss.
_EPS = 1e-12


@dataclass
class CompareResult:
    baseline_mean: float
    candidate_mean: float
    delta: float
    #: ``delta / baseline_mean * 100``. ``inf`` (signed) when the baseline mean is 0 and there is a
    #: delta, ``0.0`` when both are 0.
    relative_pct: float
    #: Two-sided, null-centred paired bootstrap p-value; always in ``[1/(B+1), 1]``.
    p_value: float
    ci_low: float
    ci_high: float
    n: int = 0
    #: Exact two-sided sign-test p-value on the pairs that differ. See `sign_test_p`.
    sign_p_value: float = 1.0
    #: Pairs where the candidate scored higher / lower / the same.
    wins: int = 0
    losses: int = 0
    ties: int = 0
    n_bootstrap: int = 0
    seed: int = 0

    @property
    def significant(self) -> bool:
        """Whether there is enough evidence to act on the observed delta.

        Requires *both* the bootstrap and the exact sign test to clear ALPHA.
        """
        return self.p_value <= ALPHA and self.sign_p_value <= ALPHA


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test: ``2 * P(X >= max(wins, losses))`` for ``X ~ Binomial(w+l, 1/2)``.

    Ties (pairs that did not differ) carry no directional information and are dropped. With 0/1
    scores this is the exact McNemar test. No wins and no losses gives 1.0.
    """
    k = wins + losses
    if k == 0:
        return 1.0
    m = max(wins, losses)
    tail = sum(math.comb(k, i) for i in range(m, k + 1)) / 2**k
    return min(1.0, 2.0 * tail)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), ``q`` in [0, 1], input sorted."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _validate(scores: Sequence[float], name: str) -> list[float]:
    out = []
    for i, s in enumerate(scores):
        try:
            v = float(s)
        except (TypeError, ValueError):
            raise ValueError(f"{name}[{i}] is not a number: {s!r}") from None
        if not math.isfinite(v):
            raise ValueError(f"{name}[{i}] is not finite: {s!r}")
        out.append(v)
    return out


def compare(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> CompareResult:
    """Paired bootstrap over two equal-length score lists (same examples, same order).

    Raises ``ValueError`` if the lists differ in length (there is no unpaired path), are empty,
    contain a non-finite score, or ``n_bootstrap`` is below `MIN_BOOTSTRAP`. The result is fully
    determined by the inputs and ``seed``.
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("Bootstrap currently expects paired observations on the same examples.")
    n = len(baseline_scores)
    if n == 0:
        raise ValueError("Empty dataset — nothing to compare.")
    if (
        not isinstance(n_bootstrap, int)
        or isinstance(n_bootstrap, bool)
        or n_bootstrap < MIN_BOOTSTRAP
    ):
        raise ValueError(f"n_bootstrap must be an integer >= {MIN_BOOTSTRAP}")
    base = _validate(baseline_scores, "baseline_scores")
    cand = _validate(candidate_scores, "candidate_scores")

    baseline_mean = math.fsum(base) / n
    candidate_mean = math.fsum(cand) / n
    diffs = [c - b for b, c in zip(base, cand, strict=True)]
    delta = math.fsum(diffs) / n
    if baseline_mean:
        rel = delta / baseline_mean * 100.0
    else:
        rel = math.copysign(math.inf, delta) if delta else 0.0

    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = n - wins - losses
    sign_p = sign_test_p(wins, losses)

    rng = random.Random(seed)
    deltas = sorted(math.fsum(rng.choices(diffs, k=n)) / n for _ in range(n_bootstrap))
    ci_low = _percentile(deltas, 0.025)
    ci_high = _percentile(deltas, 0.975)

    if delta == 0.0:
        p_value = 1.0
    elif all(d == diffs[0] for d in diffs):
        # Every difference identical: each centred resample is exactly zero, so the bootstrap has
        # no spread and would say p = 1/(B+1) at any n. The exact sign-flip p-value is 2^(1-n).
        p_value = min(1.0, 2.0 ** (1 - n))
    else:
        threshold = abs(delta) * (1.0 - _EPS)
        extreme = sum(1 for d in deltas if abs(d - delta) >= threshold)
        p_value = (1 + extreme) / (n_bootstrap + 1)

    return CompareResult(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        relative_pct=rel,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        sign_p_value=sign_p,
        wins=wins,
        losses=losses,
        ties=ties,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
