"""Stress-strength interference: closed forms, symmetry, monotonicity,
quadrature against simulation, uncertainty and refusals."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate, stats

from repyability import (
    StressStrengthCurve,
    StressStrengthResult,
    stress_strength,
    stress_strength_over_time,
)

N, W, E, L = (
    surv.Normal.from_params,
    surv.Weibull.from_params,
    surv.Exponential.from_params,
    surv.LogNormal.from_params,
)


def test_both_normal_is_the_normal_of_the_margin():
    r = stress_strength(N([10, 1]), N([7, 1.5]))
    z = 3 / math.hypot(1, 1.5)
    assert r.method == "exact"
    assert r.reliability == pytest.approx(stats.norm.cdf(z), rel=1e-14)
    assert r.failure_probability == pytest.approx(stats.norm.cdf(-z))
    assert r.reliability_index == pytest.approx(z)


def test_both_lognormal_uses_the_logs():
    r = stress_strength(L([5, 0.2]), L([4.5, 0.3]))
    z = 0.5 / math.hypot(0.2, 0.3)
    assert r.reliability == pytest.approx(stats.norm.cdf(z))


def test_both_exponential_is_the_stress_rate_over_the_total():
    # P(Y > X) = lam_X / (lam_X + lam_Y)
    r = stress_strength(E([0.5]), E([2.0]))
    assert r.reliability == pytest.approx(2.0 / 2.5)


def test_weibull_with_a_common_shape():
    r = stress_strength(W([120, 3]), W([80, 3]))
    assert r.method == "exact"
    assert r.reliability == pytest.approx(120**3 / (120**3 + 80**3))


def test_weibull_with_different_shapes_is_numerical():
    r = stress_strength(W([120, 3]), W([80, 2]))
    assert r.method == "numerical"
    brute, _ = integrate.quad(
        lambda x: W([80, 2]).df(x) * W([120, 3]).sf(x), 0, np.inf
    )
    assert r.reliability == pytest.approx(brute, rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 7])
@pytest.mark.parametrize(
    "strength, stress",
    [
        (E([0.5]), E([2.0])),
        (W([120, 3]), W([80, 3])),
    ],
)
def test_closed_forms_match_quadrature(strength, stress, n):
    exact = stress_strength(strength, stress, n_loads=n)
    numerical = stress_strength(
        strength, stress, n_loads=n, method="numerical"
    )
    assert exact.method == "exact" and numerical.method == "numerical"
    assert numerical.reliability == pytest.approx(exact.reliability, rel=1e-8)
    assert numerical.failure_probability == pytest.approx(
        exact.failure_probability, rel=1e-8
    )


def test_normal_quadrature_matches_closed_form():
    exact = stress_strength(N([10, 1]), N([7, 1.5]))
    numerical = stress_strength(N([10, 1]), N([7, 1.5]), method="numerical")
    assert numerical.reliability == pytest.approx(exact.reliability, abs=1e-12)


@pytest.mark.parametrize(
    "model",
    [N([5, 2]), W([10, 1.7]), L([1, 0.5]), E([0.3])],
)
def test_identical_continuous_models_give_one_half(model):
    r = stress_strength(model, model, method="numerical")
    assert r.reliability == pytest.approx(0.5, abs=1e-10)


def test_repeated_loads_of_identical_models():
    # P(Y > max of n iid copies) = 1 / (n + 1)
    for n in (1, 4, 9):
        r = stress_strength(W([10, 1.7]), W([10, 1.7]), n_loads=n)
        assert r.reliability == pytest.approx(1 / (n + 1), rel=1e-12)
        r = stress_strength(
            W([10, 1.7]), W([10, 1.7]), n_loads=n, method="numerical"
        )
        assert r.reliability == pytest.approx(1 / (n + 1), rel=1e-8)


def test_monotone_in_strength_stress_and_loads():
    stress = N([7, 1.5])
    by_mean = [
        stress_strength(N([m, 1]), stress).reliability for m in (6, 8, 10)
    ]
    assert by_mean == sorted(by_mean)
    by_spread = [
        stress_strength(W([20, 4]), W([a, 3]), method="numerical").reliability
        for a in (10, 14, 18)
    ]
    assert by_spread == sorted(by_spread, reverse=True)
    by_loads = [
        stress_strength(W([20, 4]), L([2, 0.3]), n_loads=n).reliability
        for n in (1, 2, 5, 50)
    ]
    assert by_loads == sorted(by_loads, reverse=True)


def test_the_normal_strength_may_be_negative_and_far_apart():
    r = stress_strength(N([-5, 1]), N([-9, 1]), method="numerical")
    assert r.reliability == pytest.approx(stats.norm.cdf(4 / math.sqrt(2)))
    far = stress_strength(N([10, 1]), N([0, 1]), method="numerical")
    assert far.failure_probability == pytest.approx(
        stats.norm.cdf(-10 / math.sqrt(2)), abs=1e-12
    )


def test_an_offset_model_is_not_given_a_closed_form():
    shifted = surv.Weibull.from_params([100, 2], gamma=20)
    r = stress_strength(shifted, W([60, 2]))
    assert r.method == "numerical"
    brute, _ = integrate.quad(
        lambda x: W([60, 2]).df(x) * shifted.sf(x), 20, np.inf
    )
    # Below the offset the strength cannot fail.
    assert r.reliability == pytest.approx(
        brute + W([60, 2]).ff(20.0), rel=1e-8
    )


@pytest.mark.parametrize("n", [1, 4])
def test_quadrature_agrees_with_simulation_within_4_se(n):
    strength, stress = W([120, 3.5]), L([4.2, 0.2])
    exact = stress_strength(strength, stress, n_loads=n)
    sim = stress_strength(
        strength,
        stress,
        n_loads=n,
        method="simulate",
        mc_samples=200_000,
        seed=7,
    )
    assert sim.method == "simulated" and sim.n_samples == 200_000
    assert abs(sim.reliability - exact.reliability) < 4 * sim.standard_error


def test_simulation_is_seeded():
    a = stress_strength(N([5, 1]), N([4, 1]), method="simulate", seed=3)
    b = stress_strength(N([5, 1]), N([4, 1]), method="simulate", seed=3)
    assert a.reliability == b.reliability


def test_uncertainty_brackets_the_point_value():
    rng = np.random.default_rng(0)
    strength = surv.Normal.fit(rng.normal(50, 5, 80))
    stress = surv.Weibull.fit(40 * rng.weibull(3.0, 60))
    r = stress_strength(strength, stress, uncertainty="fit", seed=1)
    assert r.lower < r.reliability < r.upper
    assert r.confidence == 0.95 and r.n_draws == 1000
    again = stress_strength(strength, stress, uncertainty="fit", seed=1)
    assert (again.lower, again.upper) == (r.lower, r.upper)
    wide = stress_strength(
        strength, stress, uncertainty="fit", seed=1, confidence=0.99
    )
    assert wide.lower <= r.lower and wide.upper >= r.upper


def test_uncertainty_is_refused_for_a_model_without_a_covariance():
    with pytest.raises(ValueError, match="covariance"):
        stress_strength(N([5, 1]), N([4, 1]), uncertainty="fit")
    with pytest.raises(ValueError, match="simulate"):
        stress_strength(
            N([5, 1]), N([4, 1]), uncertainty="fit", method="simulate"
        )


def test_strength_that_degrades_with_age():
    stress = N([60, 8])

    def strength_at(t):
        return N([100 - 2 * t, 10])

    ages = [0, 5, 10, 20]
    curve = stress_strength_over_time(strength_at, stress, ages)
    assert isinstance(curve, StressStrengthCurve)
    assert curve.method == "exact"
    assert list(curve.times) == ages
    assert np.all(np.diff(curve.reliability) < 0)
    for t, r in zip(ages, curve.reliability):
        assert r == pytest.approx(
            stress_strength(strength_at(t), stress).reliability
        )
    assert np.allclose(curve.reliability + curve.failure_probability, 1)


def test_results_go_through_json():
    import json

    r = stress_strength(N([5, 1]), N([4, 1]))
    assert isinstance(r, StressStrengthResult)
    assert json.loads(json.dumps(r.to_dict()))["method"] == "exact"
    curve = stress_strength_over_time(
        lambda t: N([10 - t, 1]), N([5, 1]), [0, 1]
    )
    assert len(json.loads(json.dumps(curve.to_dict()))["reliability"]) == 2


@pytest.mark.parametrize(
    "call, error, text",
    [
        (
            lambda: stress_strength(surv.Weibull, W([1, 2])),
            TypeError,
            "itself",
        ),
        (lambda: stress_strength(W([1, 2]), surv.Normal), TypeError, "itself"),
        (lambda: stress_strength(None, W([1, 2])), TypeError, "strength"),
        (lambda: stress_strength(W([1, 2]), 3.0), TypeError, "stress"),
        (
            lambda: stress_strength(W([1, 2]), W([1, 2]), n_loads=0),
            ValueError,
            "n_loads",
        ),
        (
            lambda: stress_strength(W([1, 2]), W([1, 2]), n_loads=1.5),
            ValueError,
            "n_loads",
        ),
        (
            lambda: stress_strength(W([1, 2]), W([1, 2]), method="best"),
            ValueError,
            "method",
        ),
        (
            lambda: stress_strength(W([1, 2]), W([1, 2]), seed=-1),
            ValueError,
            "seed",
        ),
        (
            lambda: stress_strength(
                N([1, 1]), N([1, 1]), uncertainty="fit", confidence=1.0
            ),
            ValueError,
            "confidence",
        ),
        (
            lambda: stress_strength(
                W([1, 2]),
                surv.Weibull.from_params([1, 2], lfp_p=0.5),
            ),
            ValueError,
            "never occurs",
        ),
        (
            lambda: stress_strength(
                surv.Bernoulli.from_params([0.1]), W([1, 2])
            ),
            ValueError,
            "fixed",
        ),
        (
            lambda: stress_strength_over_time(3, W([1, 2]), [1]),
            TypeError,
            "strength_at",
        ),
        (
            lambda: stress_strength_over_time(
                lambda t: W([1, 2]), W([1, 2]), []
            ),
            ValueError,
            "times",
        ),
        (
            lambda: stress_strength_over_time(
                lambda t: W([1, 2]), W([1, 2]), [math.nan]
            ),
            ValueError,
            "times",
        ),
    ],
)
def test_refusals(call, error, text):
    with pytest.raises(error, match=text):
        call()
