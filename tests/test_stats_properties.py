"""Properties of `compare` that must hold for any input, not just the examples someone thought of."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_eval.stats import MIN_BOOTSTRAP, compare

# Multiples of 1/8 are exact in binary floating point, so shifting and negating them is lossless and
# the invariance checks below can demand exact equality instead of a tolerance.
score = st.sampled_from([i / 8 for i in range(9)])
paired = st.integers(1, 25).flatmap(
    lambda n: st.tuples(st.lists(score, min_size=n, max_size=n), st.lists(score, min_size=n, max_size=n))
)
B = MIN_BOOTSTRAP  # keep each example fast; the properties do not depend on B
FAST = settings(max_examples=80, deadline=None)


@FAST
@given(paired)
def test_p_value_and_interval_are_well_formed(pair) -> None:
    base, cand = pair
    r = compare(base, cand, n_bootstrap=B)
    assert 0.0 < r.p_value <= 1.0
    assert 0.0 < r.sign_p_value <= 1.0
    assert r.ci_low <= r.ci_high
    diffs = [c - b for b, c in zip(base, cand, strict=True)]
    assert min(diffs) - 1e-12 <= r.ci_low and r.ci_high <= max(diffs) + 1e-12
    assert r.wins + r.losses + r.ties == r.n == len(base)


@FAST
@given(paired)
def test_swapping_the_prompts_negates_delta_and_mirrors_the_interval(pair) -> None:
    base, cand = pair
    a = compare(base, cand, n_bootstrap=B)
    b = compare(cand, base, n_bootstrap=B)
    assert b.delta == -a.delta
    assert b.p_value == a.p_value
    assert b.sign_p_value == a.sign_p_value
    assert b.ci_low == pytest.approx(-a.ci_high, abs=1e-12)
    assert b.ci_high == pytest.approx(-a.ci_low, abs=1e-12)
    assert (b.wins, b.losses) == (a.losses, a.wins)


@FAST
@given(paired, st.sampled_from([-2.0, -0.5, 0.25, 1.0, 4.0]))
def test_adding_the_same_constant_to_both_prompts_changes_nothing(pair, shift) -> None:
    base, cand = pair
    a = compare(base, cand, n_bootstrap=B)
    b = compare([x + shift for x in base], [x + shift for x in cand], n_bootstrap=B)
    assert (b.delta, b.p_value, b.sign_p_value, b.ci_low, b.ci_high) == (
        a.delta, a.p_value, a.sign_p_value, a.ci_low, a.ci_high)


@FAST
@given(paired, st.sampled_from([0.5, 2.0, 8.0]))
def test_rescaling_both_prompts_scales_delta_and_interval_but_not_the_p_values(pair, k) -> None:
    base, cand = pair
    a = compare(base, cand, n_bootstrap=B)
    b = compare([x * k for x in base], [x * k for x in cand], n_bootstrap=B)
    assert b.delta == pytest.approx(a.delta * k, abs=1e-12)
    assert b.ci_low == pytest.approx(a.ci_low * k, abs=1e-12)
    assert b.ci_high == pytest.approx(a.ci_high * k, abs=1e-12)
    assert b.p_value == a.p_value and b.sign_p_value == a.sign_p_value


@FAST
@given(paired, st.integers(0, 2**31))
def test_same_seed_same_answer(pair, seed) -> None:
    base, cand = pair
    assert compare(base, cand, n_bootstrap=B, seed=seed) == compare(base, cand, n_bootstrap=B, seed=seed)


@FAST
@given(paired)
def test_the_sign_floor_depends_only_on_the_win_and_loss_counts(pair) -> None:
    base, cand = pair
    r = compare(base, cand, n_bootstrap=B)
    k = r.wins + r.losses
    m = max(r.wins, r.losses)
    expected = 1.0 if k == 0 else min(
        1.0, 2 * sum(math.comb(k, i) for i in range(m, k + 1)) / 2**k)
    assert r.sign_p_value == pytest.approx(expected, abs=1e-15)


@FAST
@given(st.integers(1, 40), st.floats(0.05, 1.0), st.booleans())
def test_unanimous_differences_get_the_exact_sign_flip_p_value(n, step, up) -> None:
    base = [0.0] * n
    cand = [step if up else -step] * n
    r = compare(base, cand, n_bootstrap=B)
    assert r.p_value == pytest.approx(min(1.0, 2.0 ** (1 - n)))
    assert r.sign_p_value == pytest.approx(min(1.0, 2.0 ** (1 - n)))


@FAST
@given(st.lists(score, min_size=1, max_size=25))
def test_identical_prompts_are_never_a_difference(scores) -> None:
    r = compare(scores, list(scores), n_bootstrap=B)
    assert (r.delta, r.p_value, r.sign_p_value, r.ci_low, r.ci_high) == (0.0, 1.0, 1.0, 0.0, 0.0)
    assert not r.significant


@FAST
@given(paired)
def test_significant_requires_both_tests(pair) -> None:
    base, cand = pair
    r = compare(base, cand, n_bootstrap=B)
    assert r.significant == (r.p_value <= 0.05 and r.sign_p_value <= 0.05)
