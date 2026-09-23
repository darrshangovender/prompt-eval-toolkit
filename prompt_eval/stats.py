"""Bootstrap confidence intervals + p-value for two-sample comparison.

Pure-stdlib (no scipy dependency). For 40-example datasets this is fine; for
10k examples you'd want a smarter implementation but you're also not using
this tool at that scale.

Why bootstrap and not a t-test:
  - Our metric scores are 0..1 and often bimodal (mostly 0s and 1s for
    exact_match). t-test assumptions are violated; bootstrap doesn't care
    about distribution shape.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

#: Conventional significance level; `CompareResult.significant` is the gate the
#: CLI acts on.
ALPHA = 0.05


@dataclass
class CompareResult:
    baseline_mean: float
    candidate_mean: float
    delta: float
    relative_pct: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int = 0
    #: Exact two-sided sign-test p-value on the paired differences. See
    #: :func:`_sign_test_p` for why the bootstrap alone is not enough.
    sign_p_value: float = 1.0

    @property
    def significant(self) -> bool:
        """Whether there is enough evidence to act on the observed delta.

        Requires *both* the bootstrap and the sign test to clear ALPHA. The
        bootstrap alone reported p=0.000 on two examples (see `_sign_test_p`),
        which the CLI turned into "Candidate wins. Promote."
        """
        return self.p_value <= ALPHA and self.sign_p_value <= ALPHA


def _sign_test_p(baseline_scores: list[float], candidate_scores: list[float]) -> float:
    """Exact two-sided sign test on the paired differences.

    The paired bootstrap has *zero variance* whenever every paired difference is
    identical: each resample reproduces exactly the same delta, so no resample
    ever crosses the null, and the p-value collapses to exactly 0.0 — at any
    sample size. `compare([0.0, 0.0], [1.0, 1.0])` returned p=0.000 with a
    [1.0, 1.0] CI, and the CLI printed "Candidate wins. Promote." on two rows.
    That is not an exotic input: "the candidate fixes every case the baseline
    failed" is the ordinary success shape on a small curated set.

    The sign test does respect sample size — two unanimous wins is p=0.5, and you
    need six before unanimity alone clears 0.05 — so it serves as a floor on how
    much the bootstrap is allowed to claim.
    """
    wins = sum(1 for b, c in zip(baseline_scores, candidate_scores) if c > b)
    losses = sum(1 for b, c in zip(baseline_scores, candidate_scores) if c < b)
    n_eff = wins + losses
    if n_eff == 0:
        return 1.0  # every pair tied: no directional evidence at all
    k = max(wins, losses)
    tail = sum(math.comb(n_eff, i) for i in range(k, n_eff + 1)) / 2**n_eff
    return min(1.0, 2.0 * tail)


def compare(baseline_scores: list[float], candidate_scores: list[float], *, n_bootstrap: int = 5000) -> CompareResult:
    """Paired bootstrap over two equal-length score lists.

    Raises if the lists differ in length — there is no unpaired path.
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("Bootstrap currently expects paired observations on the same examples.")
    n = len(baseline_scores)
    if n == 0:
        raise ValueError("Empty dataset — nothing to compare.")

    baseline_mean = sum(baseline_scores) / n
    candidate_mean = sum(candidate_scores) / n
    delta = candidate_mean - baseline_mean
    rel = (delta / baseline_mean * 100.0) if baseline_mean else float("inf")

    # Paired bootstrap: for each resample, resample (i) with replacement and
    # compute the delta on the resample. The distribution of those deltas
    # gives us a CI for the true delta; p-value is fraction of resamples
    # where the sign flipped (one-sided).
    rng = random.Random(42)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        b = sum(baseline_scores[i] for i in indices) / n
        c = sum(candidate_scores[i] for i in indices) / n
        deltas.append(c - b)
    deltas.sort()
    ci_low = deltas[int(0.025 * n_bootstrap)]
    ci_high = deltas[int(0.975 * n_bootstrap)]
    # Two-sided p: fraction of bootstrap deltas with sign opposite to observed,
    # times 2.
    if delta >= 0:
        p_one_sided = sum(1 for d in deltas if d <= 0) / n_bootstrap
    else:
        p_one_sided = sum(1 for d in deltas if d >= 0) / n_bootstrap
    p_value = min(1.0, 2.0 * p_one_sided)

    return CompareResult(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        relative_pct=rel,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        sign_p_value=_sign_test_p(baseline_scores, candidate_scores),
    )
