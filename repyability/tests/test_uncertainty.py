"""Tests for epistemic (parameter) uncertainty:
``NonRepairableRBD.sf_uncertainty`` and ``UncertaintyResult``.

Where the draws have a known distribution, the samples are checked against
its percentiles: the fraction of samples below an analytic percentile must
be that percentile, up to binomial sampling error. Where the draws come from
a short list of models, every sample must be one of the system values those
models give, worked out by building the diagram with them.
"""

import itertools
import math

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import (
    BetaFactor,
    CCFGroup,
    NonRepairableRBD,
    UncertaintyResult,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params


def one_node(model):
    return NonRepairableRBD([("s", "c"), ("c", "t")], {"c": model})


def assert_percentile(samples, value, q, sigmas=4.5):
    """The fraction of samples at or below ``value`` is ``q``, up to
    binomial sampling error."""
    n = len(samples)
    fraction = np.mean(samples <= value)
    assert abs(fraction - q) < sigmas * math.sqrt(q * (1 - q) / n)


# -- the draws, against their known distributions -----------------------------


def test_fit_draws_follow_the_fits_normal_approximation():
    # An exponential fit: the rate is drawn log-normally, with the fit's
    # variance carried to the log scale (delta method), so the reliability
    # exp(-rate * t) has known percentiles.
    times = np.linspace(20, 400, 40)
    model = surv.Exponential.fit(times)
    rate = float(np.ravel(model.params)[0])
    s = math.sqrt(float(np.ravel(model.hess_inv)[0])) / rate
    result = one_node(model).sf_uncertainty(
        100.0, {"c": "fit"}, n_draws=40_000, seed=0
    )
    assert result.nominal == pytest.approx(math.exp(-rate * 100.0))
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        # A high rate is a low reliability.
        value = math.exp(-100.0 * rate * math.exp(st.norm.ppf(1 - q) * s))
        assert_percentile(result.samples, value, q)


def test_fit_draws_agree_with_surpyvals_confidence_bounds():
    model = surv.Weibull.fit(np.linspace(200, 1800, 50))
    result = one_node(model).sf_uncertainty(
        500.0, {"c": "fit"}, n_draws=50_000, seed=1
    )
    lower, upper = result.interval(0.9)
    bounds = np.ravel(model.cb(500.0, alpha_ci=0.1))
    assert lower == pytest.approx(bounds[0], abs=0.01)
    assert upper == pytest.approx(bounds[1], abs=0.01)


def test_fit_draws_keep_an_offset():
    model = surv.Weibull.fit(np.linspace(200, 1800, 50) + 100, offset=True)
    rbd = one_node(model)
    # Before the offset nothing can fail, in every draw.
    with np.errstate(invalid="ignore"):
        early = rbd.sf_uncertainty(
            0.5 * model.gamma, {"c": "fit"}, n_draws=200
        )
    np.testing.assert_array_equal(early.samples, 1.0)
    later = rbd.sf_uncertainty(1000.0, {"c": "fit"}, n_draws=200, seed=2)
    assert later.std > 0


def test_parameter_distributions():
    # The rate is uniform between 1 and 3 per 1000 h.
    result = one_node(E([0.002])).sf_uncertainty(
        300.0,
        {"c": {"failure_rate": st.uniform(0.001, 0.002)}},
        n_draws=40_000,
        seed=3,
    )
    for q in (0.05, 0.5, 0.95):
        rate = 0.001 + (1 - q) * 0.002
        assert_percentile(result.samples, math.exp(-300.0 * rate), q)
    assert result.nominal == pytest.approx(math.exp(-0.6))


def test_a_weibull_shape_can_be_uncertain_alone():
    # The scale stays at 100; the shape is drawn from a surpyval
    # distribution (anything with qf will do).
    shape = surv.Normal.from_params([2.0, 0.1])
    result = one_node(W([100.0, 2.0])).sf_uncertainty(
        50.0, {"c": {"beta": shape}}, n_draws=40_000, seed=4
    )
    for q in (0.1, 0.9):
        # (1/2)**beta falls as beta rises, so the reliability rises.
        beta = 2.0 + 0.1 * st.norm.ppf(q)
        assert_percentile(result.samples, math.exp(-(0.5**beta)), q)


def test_a_fixed_probability_can_be_uncertain():
    result = one_node(FixedEventProbability.from_params(0.1)).sf_uncertainty(
        uncertainty={"c": {"p": st.beta(2, 18)}}, n_draws=40_000, seed=5
    )
    for q in (0.05, 0.5, 0.95):
        assert_percentile(result.samples, 1 - st.beta(2, 18).ppf(1 - q), q)


# -- the system, draw by draw -------------------------------------------------


BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("d", "t"),
    ("e", "t"),
]


def bridge(models):
    return NonRepairableRBD(BRIDGE, models)


def test_every_draw_is_the_system_with_the_drawn_models():
    base = {n: W([100.0 + 20 * i, 1.5]) for i, n in enumerate("abcde")}
    choices = {
        "a": [W([80.0, 1.5]), W([140.0, 2.5])],
        "d": [W([60.0, 1.2]), W([200.0, 3.0]), W([100.0, 2.0])],
    }
    x = np.array([30.0, 90.0])
    result = bridge(base).sf_uncertainty(x, choices, n_draws=400, seed=6)
    assert result.samples.shape == (400, 2)
    possible = []
    for a, d in itertools.product(choices["a"], choices["d"]):
        possible.append(bridge({**base, "a": a, "d": d}).sf(x))
    possible = np.array(possible)
    for row in result.samples:
        assert np.any(np.all(np.abs(possible - row) < 1e-14, axis=1))
    # Every combination turns up.
    hits = {
        int(np.argmin(np.abs(possible - row).sum(axis=1)))
        for row in result.samples
    }
    assert hits == set(range(len(possible)))
    np.testing.assert_allclose(result.nominal, bridge(base).sf(x))


def test_shared_uncertainty_draws_one_model_for_all():
    unit = E([0.01])
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        {"p1": unit, "p2": E([0.01])},  # equal models, separate objects
    )
    choices = [E([0.005]), E([0.02])]
    s = [math.exp(-0.005 * 50), math.exp(-0.02 * 50)]
    shared = rbd.sf_uncertainty(50.0, {("p1", "p2"): choices}, n_draws=500)
    allowed = {1 - (1 - v) ** 2 for v in s}
    assert all(
        any(abs(sample - a) < 1e-14 for a in allowed)
        for sample in shared.samples
    )
    independent = rbd.sf_uncertainty(
        50.0, {"p1": choices, "p2": choices}, n_draws=500, seed=7
    )
    mixed = 1 - (1 - s[0]) * (1 - s[1])
    assert np.any(np.abs(independent.samples - mixed) < 1e-14)
    # Shared uncertainty spreads the system's more.
    assert shared.std > independent.std


def test_draws_do_not_depend_on_the_times():
    rbd = one_node(E([0.01]))
    spec = {"c": {"failure_rate": st.uniform(0.005, 0.01)}}
    both = rbd.sf_uncertainty(
        np.array([10.0, 50.0]), spec, n_draws=100, seed=8
    )
    one = rbd.sf_uncertainty(50.0, spec, n_draws=100, seed=8)
    np.testing.assert_array_equal(both.samples[:, 1], one.samples)


def test_seeds_make_the_draws_reproducible():
    rbd = one_node(E([0.01]))
    spec = {"c": {"failure_rate": st.uniform(0.005, 0.01)}}
    first = rbd.sf_uncertainty(50.0, spec, n_draws=50, seed=9)
    again = rbd.sf_uncertainty(50.0, spec, n_draws=50, seed=9)
    other = rbd.sf_uncertainty(50.0, spec, n_draws=50, seed=10)
    np.testing.assert_array_equal(first.samples, again.samples)
    assert not np.array_equal(first.samples, other.samples)


# -- the result ---------------------------------------------------------------


def test_the_result_summaries():
    samples = np.array([0.1, 0.5, 0.2, 0.9, 0.4])
    result = UncertaintyResult(samples=samples, nominal=0.4, n_draws=5)
    assert result.mean == pytest.approx(0.42)
    assert result.median == 0.4
    assert result.std == pytest.approx(np.std(samples, ddof=1))
    assert result.percentile(50) == 0.4
    assert result.interval(0.5) == (
        np.percentile(samples, 25),
        np.percentile(samples, 75),
    )
    assert isinstance(result.mean, float)
    assert set(result) == {"samples", "nominal", "n_draws"}
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="level"):
            result.interval(bad)
    rows = UncertaintyResult(
        samples=np.array([[0.1, 0.2], [0.3, 0.6]]),
        nominal=np.array([0.2, 0.4]),
        n_draws=2,
    )
    np.testing.assert_allclose(rows.mean, [0.2, 0.4])
    lower, upper = rows.interval(0.9)
    assert lower.shape == upper.shape == (2,)


# -- invalid requests ---------------------------------------------------------


def test_invalid_requests_are_rejected():
    rbd = NonRepairableRBD(
        [
            ("s", "a"),
            ("a", "b"),
            ("b", "t"),
            ("s", "a2"),
            ("a2", "t"),
        ],
        {"a": W([100.0, 2.0]), "b": E([0.01]), "a2": "a"},
    )
    cases = [
        ({}, "Give the uncertain nodes"),
        ({"z": "fit"}, "not a component"),
        ({"s": "fit"}, "not a component"),
        ({"a2": [E([0.01])]}, "repeat of"),
        ({"a": [W([90.0, 2.0])], ("a", "b"): [E([0.01])]}, "twice"),
        ({("a", "b"): [E([0.01])]}, "same model"),
        ({"a": "bootstrap"}, "unknown uncertainty"),
        ({"a": {}}, "no parameter distributions"),
        ({"a": []}, "empty"),
        ({"a": [object()]}, "needs sf"),
        ({"a": {"gamma": st.uniform(0, 1)}}, "not parameters"),
        ({"a": {"alpha": 5.0}}, "quantile function"),
        ({"b": {"failure_rate": st.norm(0.01, 0.01)}}, "outside its range"),
        ({"a": "fit"}, "covariance"),
        ({"a": 3}, "must be 'fit'"),
    ]
    for spec, match in cases:
        with pytest.raises(ValueError, match=match):
            rbd.sf_uncertainty(10.0, spec, n_draws=20, seed=0)
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="n_draws"):
            rbd.sf_uncertainty(10.0, {"b": [E([0.02])]}, n_draws=bad)
    with pytest.raises(ValueError, match="x is required"):
        rbd.sf_uncertainty(uncertainty={"b": [E([0.02])]})
    with pytest.raises(ValueError, match="no parameters to draw"):
        # A nested diagram is a node model with no parameters to draw.
        one_node(rbd).sf_uncertainty(
            10.0, {"c": {"rate": st.uniform()}}, n_draws=5
        )


def test_common_cause_groups_are_not_supported():
    unit = FixedEventProbability.from_params(0.1)
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError):
        rbd.sf_uncertainty(uncertainty={"a": [unit]})


def test_a_model_on_the_edge_of_its_range_has_no_normal_approximation():
    model = surv.Exponential.fit(np.linspace(20, 400, 40))
    edge = surv.Exponential.fit(np.linspace(20, 400, 40))
    edge.params = np.array([0.0])
    with pytest.raises(ValueError, match="edge of its range"):
        one_node(edge).sf_uncertainty(10.0, {"c": "fit"}, n_draws=5)
    broken = surv.Exponential.fit(np.linspace(20, 400, 40))
    broken.hess_inv = np.array([[np.nan]])
    with pytest.raises(ValueError, match="finite"):
        one_node(broken).sf_uncertainty(10.0, {"c": "fit"}, n_draws=5)
    assert (
        one_node(model)
        .sf_uncertainty(10.0, {"c": "fit"}, n_draws=5, seed=0)
        .n_draws
        == 5
    )
