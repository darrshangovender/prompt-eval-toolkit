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

import random
from dataclasses import dataclass


@dataclass
class CompareResult:
    baseline_mean: float
    candidate_mean: float
    delta: float
    relative_pct: float
    p_value: float
    ci_low: float
    ci_high: float


def compare(baseline_scores: list[float], candidate_scores: list[float], *, n_bootstrap: int = 5000) -> CompareResult:
    """Paired bootstrap if both lists same length, otherwise unpaired."""
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
    )
