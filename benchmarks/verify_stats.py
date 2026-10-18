"""Checks `prompt_eval.stats.compare` against known answers and against scipy, and measures how often
it is wrong.

    python benchmarks/verify_stats.py            # all four sections, about 1-2 minutes
    python benchmarks/verify_stats.py --quick    # fewer simulated trials (a smoke run, noisier)

Needs scipy and numpy (the `dev` extra). Everything is seeded; reruns print identical tables.

1. Known cases: closed-form answers (an exact binomial tail, the unanimous case, and the exact
   bootstrap distribution of a five-row example by enumerating all 3,125 resamples).
2. The bundled SYNTHETIC example, against four scipy references: `scipy.stats.bootstrap` (the same
   estimator, independently implemented), the paired t-test, a sign-flip permutation test and the
   exact binomial (McNemar) test.
3. Calibration: when the two prompts are in truth the same, how often does `p <= 0.05`? It should be
   at most 5%.
4. Power: when the candidate is truly better, how often is it detected, with and without the exact
   sign-test floor that `significant` applies.

The simulated data in 3 and 4 is SYNTHETIC: random score differences with stated parameters.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompt_eval.metrics import REGISTRY  # noqa: E402
from prompt_eval.runner import load_dataset, load_outputs  # noqa: E402
from prompt_eval.stats import ALPHA, compare, sign_test_p  # noqa: E402

ALPHA_STR = f"{ALPHA:.2f}"


# ---------------------------------------------------------------- 1. known cases


def exact_bootstrap_p(diffs: list[float]) -> float:
    """The exact null-centred bootstrap p-value by enumerating every resample (n^n of them)."""
    n = len(diffs)
    delta = math.fsum(diffs) / n
    centred = [d - delta for d in diffs]
    hits = 0
    for combo in itertools.product(range(n), repeat=n):
        m = math.fsum(centred[i] for i in combo) / n
        hits += abs(m) >= abs(delta) * (1 - 1e-12)
    return hits / n**n


def known_cases() -> list[dict]:
    rows = []
    # 10 wins, 2 losses: 2 * (C(12,10) + C(12,11) + C(12,12)) / 2**12 = 158 / 4096 by hand.
    ours = sign_test_p(10, 2)
    rows.append({
        "case": "sign test, 10 wins / 2 losses", "expected": 158 / 4096,
        "expected_from": "2 x (66+12+1) / 4096, by hand", "ours": ours,
        "scipy": float(sps.binomtest(10, 12, 0.5).pvalue),
    })
    # Two unanimous pairs: only 2 of the 4 sign patterns are as extreme, so p = 0.5, not 0.
    r = compare([0.0, 0.0], [1.0, 1.0])
    rows.append({"case": "2 unanimous wins", "expected": 0.5, "expected_from": "2^(1-2)",
                 "ours": r.p_value, "scipy": float(sps.binomtest(2, 2, 0.5).pvalue)})
    # n = 20 unanimous: 2^(1-20).
    r = compare([0.0] * 20, [1.0] * 20)
    rows.append({"case": "20 unanimous wins", "expected": 2.0**-19, "expected_from": "2^(1-20)",
                 "ours": r.p_value, "scipy": float(sps.binomtest(20, 20, 0.5).pvalue)})
    # A five-row example whose bootstrap distribution can be enumerated exactly.
    base = [0.0, 0.0, 1.0, 0.0, 1.0]
    cand = [1.0, 1.0, 1.0, 0.0, 1.0]
    diffs = [c - b for b, c in zip(base, cand, strict=True)]
    exact = exact_bootstrap_p(diffs)
    r = compare(base, cand, n_bootstrap=200_000)
    rows.append({"case": "5 rows, exact bootstrap by enumeration (3,125 resamples)",
                 "expected": exact, "expected_from": "enumerated", "ours": r.p_value,
                 "scipy": float("nan")})
    return rows


# ---------------------------------------------------------------- 2. bundled example vs scipy


def scipy_references(base: list[float], cand: list[float]) -> dict:
    b, c = np.asarray(base), np.asarray(cand)
    d = c - b
    delta = float(d.mean())
    # Same estimator, scipy's machinery: bootstrap the *centred* differences and count extremes.
    res = sps.bootstrap((d - delta,), np.mean, n_resamples=100_000, method="percentile",
                        random_state=np.random.default_rng(11))
    dist = res.bootstrap_distribution
    boot_p = (1 + float(np.sum(np.abs(dist) >= abs(delta) * (1 - 1e-12)))) / (len(dist) + 1)
    ci = sps.bootstrap((d,), np.mean, n_resamples=100_000, confidence_level=0.95,
                       method="percentile", random_state=np.random.default_rng(12)).confidence_interval
    perm = sps.permutation_test((b, c), lambda x, y: np.mean(y - x), permutation_type="samples",
                                n_resamples=100_000, alternative="two-sided",
                                random_state=np.random.default_rng(13))
    w, lo = int((d > 0).sum()), int((d < 0).sum())
    return {
        "scipy_bootstrap_p": boot_p,
        "ttest_rel_p": float(sps.ttest_rel(c, b).pvalue),
        "permutation_p": float(perm.pvalue),
        "binomtest_p": float(sps.binomtest(max(w, lo), w + lo, 0.5).pvalue) if w + lo else 1.0,
        "scipy_ci": (float(ci.low), float(ci.high)),
    }


def bundled_example() -> list[dict]:
    data = ROOT / "data"
    examples = load_dataset(data / "example_dataset.yml")
    base_out = load_outputs(data / "example_outputs_baseline.json", expected_rows=len(examples))
    cand_out = load_outputs(data / "example_outputs_candidate.json", expected_rows=len(examples))
    rows = []
    for name, metric in REGISTRY.items():
        base = [metric(o, e.expected) for o, e in zip(base_out, examples, strict=True)]
        cand = [metric(o, e.expected) for o, e in zip(cand_out, examples, strict=True)]
        r = compare(base, cand, n_bootstrap=20_000)
        rows.append({"metric": name, "n": r.n, "baseline": r.baseline_mean,
                     "candidate": r.candidate_mean, "wins": r.wins, "losses": r.losses,
                     "ours_p": r.p_value, "ours_sign_p": r.sign_p_value,
                     "ours_ci": (r.ci_low, r.ci_high), "significant": r.significant,
                     **scipy_references(base, cand)})
    return rows


# ---------------------------------------------------------------- 3 and 4. simulation


def _ternary(rng: random.Random, n: int, p_win: float, p_loss: float) -> list[float]:
    out = []
    for _ in range(n):
        u = rng.random()
        out.append(1.0 if u < p_win else -1.0 if u < p_win + p_loss else 0.0)
    return out


def _normal(rng: random.Random, n: int, mean: float) -> list[float]:
    return [rng.gauss(mean, 1.0) for _ in range(n)]


def _skewed(rng: random.Random, n: int, mean: float) -> list[float]:
    # Exponential(1) - 1 has mean 0 and a long right tail; shift for the alternative.
    return [rng.expovariate(1.0) - 1.0 + mean for _ in range(n)]


FAMILIES = {
    "0/1 scores (differences -1, 0, +1)": ("ternary", None),
    "continuous, symmetric (normal)": ("normal", _normal),
    "continuous, skewed (exponential)": ("skewed", _skewed),
}


def simulate(family: str, n: int, effect: float, trials: int, n_bootstrap: int, seed: int) -> dict:
    """Rejection rates at ALPHA over ``trials`` simulated paired datasets.

    ``effect`` is 0 for the null. For the 0/1 family it is the win-rate surplus: P(win) is
    ``0.10 + effect`` and P(loss) is ``0.10``. For the others it is the mean difference in units of
    the (unit) standard deviation of the differences.
    """
    rng = random.Random(seed)
    rej = {"bootstrap": 0, "bootstrap+sign (significant)": 0, "paired t-test": 0, "sign test": 0}
    for _ in range(trials):
        if family == "ternary":
            diffs = _ternary(rng, n, 0.10 + effect, 0.10)
        elif family == "normal":
            diffs = _normal(rng, n, effect)
        else:
            diffs = _skewed(rng, n, effect)
        r = compare([0.0] * n, diffs, n_bootstrap=n_bootstrap, seed=rng.randrange(2**31))
        rej["bootstrap"] += r.p_value <= ALPHA
        rej["bootstrap+sign (significant)"] += r.significant
        rej["sign test"] += r.sign_p_value <= ALPHA
        if len(set(diffs)) > 1:
            rej["paired t-test"] += float(sps.ttest_1samp(diffs, 0.0).pvalue) <= ALPHA
    return {k: v / trials for k, v in rej.items()}


# ---------------------------------------------------------------- printing


def _f(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.2e}" if 0 < x < 1e-3 else f"{x:.4f}"


def known_lines(known: list[dict]) -> list[str]:
    lines = ["| case | expected | where from | `prompt_eval` | scipy |", "|---|---|---|---|---|"]
    for k in known:
        lines.append(f"| {k['case']} | {_f(k['expected'])} | {k['expected_from']} | {_f(k['ours'])} "
                     f"| {_f(k['scipy'])} |")
    return lines


def example_lines(example: list[dict]) -> list[str]:
    lines = ["| metric | n | baseline | candidate | wins/losses | p (ours) | p (scipy bootstrap) "
             "| p (paired t) | p (sign-flip permutation) | p (exact binomial) | sign p (ours) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in example:
        lines.append(
            f"| {e['metric']} | {e['n']} | {e['baseline']:.3f} | {e['candidate']:.3f} "
            f"| {e['wins']}/{e['losses']} | {_f(e['ours_p'])} | {_f(e['scipy_bootstrap_p'])} "
            f"| {_f(e['ttest_rel_p'])} | {_f(e['permutation_p'])} | {_f(e['binomtest_p'])} "
            f"| {_f(e['ours_sign_p'])} |")
    lines += ["", "| metric | 95% CI (ours) | 95% CI (scipy percentile bootstrap) |", "|---|---|---|"]
    for e in example:
        lines.append(f"| {e['metric']} | {e['ours_ci'][0]:+.3f} .. {e['ours_ci'][1]:+.3f} "
                     f"| {e['scipy_ci'][0]:+.3f} .. {e['scipy_ci'][1]:+.3f} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="200 trials per cell instead of 1,000")
    ap.add_argument("--json", action="store_true", help="print the raw numbers as JSON instead")
    args = ap.parse_args()
    trials = 200 if args.quick else 1000
    n_boot = 499
    se = math.sqrt(ALPHA * (1 - ALPHA) / trials)

    known = known_cases()
    example = bundled_example()
    sizes = (10, 20, 40, 100)
    calibration = {fam: {n: simulate(key, n, 0.0, trials, n_boot, seed=1000 + n)
                         for n in sizes} for fam, (key, _) in FAMILIES.items()}
    power_cells = {"0/1 scores (differences -1, 0, +1)": (0.10, "win rate 20% vs loss rate 10%"),
                   "continuous, symmetric (normal)": (0.4, "mean difference 0.4 SD")}
    power = {fam: {n: simulate(FAMILIES[fam][0], n, eff, trials, n_boot, seed=2000 + n)
                   for n in (20, 40, 100)} for fam, (eff, _) in power_cells.items()}

    if args.json:
        print(json.dumps({"known": known, "example": example, "calibration": calibration,
                          "power": power}, indent=1, default=str))
        return 0

    print("### 1. Known cases\n")
    print("\n".join(known_lines(known)))

    print("\n### 2. Bundled SYNTHETIC example against scipy (20,000 resamples ours, 100,000 scipy's)\n")
    print("\n".join(example_lines(example)))

    print(f"\n### 3. Calibration: share of {trials:,} simulated no-difference datasets with "
          f"p <= {ALPHA_STR} (should be <= {ALPHA_STR}; Monte Carlo SE about {se:.3f})\n")
    cols = list(next(iter(next(iter(calibration.values())).values())))
    print("| differences | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for fam, by_n in calibration.items():
        for n, rates in by_n.items():
            print(f"| {fam} | {n} | " + " | ".join(f"{rates[c]:.3f}" for c in cols) + " |")

    print(f"\n### 4. Power: share of {trials:,} simulated datasets with a true improvement "
          f"called significant (p <= {ALPHA_STR})\n")
    print("| true difference | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for fam, by_n in power.items():
        for n, rates in by_n.items():
            print(f"| {power_cells[fam][1]} | {n} | " + " | ".join(f"{rates[c]:.3f}" for c in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
