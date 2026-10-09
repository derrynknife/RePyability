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
    s = math.sqrt(float(np.ravel(model.covariance())[0])) / rate
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
    # Before the offset nothing can fail, in every draw. surpyval's
    # covariance leaves the offset out (SurPyval#830), so it is held, and
    # the draws say so.
    with np.errstate(invalid="ignore"), pytest.warns(
        UserWarning, match="offset.*SurPyval#830"
    ):
        early = rbd.sf_uncertainty(
            0.5 * model.gamma, {"c": "fit"}, n_draws=200
        )
    np.testing.assert_array_equal(early.samples, 1.0)
    with pytest.warns(UserWarning, match="offset"):
        later = rbd.sf_uncertainty(1000.0, {"c": "fit"}, n_draws=200, seed=2)
    assert later.std > 0


def shares_fit(lfp=True, zi=False):
    """A Weibull fitted to 60 failures, with 40 units that never failed
    (``lfp``) and 10 dead on arrival (``zi``)."""
    x = [np.random.default_rng(5).weibull(2, 60) * 100]
    c = [np.zeros(60)]
    if lfp:
        x, c = x + [np.full(40, 400.0)], c + [np.ones(40)]
    if zi:
        x, c = x + [np.zeros(10)], c + [np.zeros(10)]
    x, c = np.concatenate(x), np.concatenate(c).astype(int)
    return surv.Weibull.fit(x, c=c, lfp=lfp, zi=zi)


@pytest.mark.parametrize("share", ["lfp_p", "f0"])
def test_fit_draws_vary_the_shares_that_never_fail_and_are_dead(share):
    # Long after every unit that can fail has failed, the reliability is
    # the share that never fails, 1 - lfp_p; just after 0 it is the share
    # not dead on arrival, 1 - f0 (#267). Each share is drawn on the logit
    # scale, its variance from surpyval's covariance() (the delta method),
    # so the reliability there has known percentiles; before #267 it was
    # held at its fitted value.
    model = shares_fit(lfp=share == "lfp_p", zi=share == "f0")
    p = float(getattr(model, share))
    sd = math.sqrt(float(model.covariance()[-1, -1])) / (p * (1 - p))
    at = 1e4 if share == "lfp_p" else 1e-9
    result = one_node(model).sf_uncertainty(
        at, {"c": "fit"}, n_draws=40_000, seed=4
    )
    assert result.nominal == pytest.approx(1 - p)
    logit = math.log(p / (1 - p))
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        # A high share is a low reliability.
        drawn = 1 / (1 + math.exp(-(logit + st.norm.ppf(1 - q) * sd)))
        assert_percentile(result.samples, 1 - drawn, q)


def test_fit_draws_every_parameter_with_the_fits_covariance():
    # A fit of both shares: the draws of all four parameters have the
    # fit's covariance, to the skew of the log and logit scales over the
    # draws: each variance within 15%, each correlation within 0.05.
    from repyability.rbd._model_utils import parametric_spec
    from repyability.rbd.uncertainty import draw_models

    model = shares_fit(lfp=True, zi=True)
    drawn = draw_models(model, "fit", 20_000, np.random.default_rng(6), "c")
    params = np.array([parametric_spec(m).params for m in drawn])
    assert parametric_spec(model).names == ["alpha", "beta", "lfp_p", "f0"]
    drawn_cov, fitted = np.cov(params.T), model.covariance()
    np.testing.assert_allclose(np.diag(drawn_cov), np.diag(fitted), rtol=0.15)
    sd = np.sqrt(np.diag(fitted))
    np.testing.assert_allclose(
        np.corrcoef(params.T), fitted / np.outer(sd, sd), atol=0.05
    )
    # Centred on the fit on the log and logit scales: the median is it.
    np.testing.assert_allclose(
        np.median(params, axis=0), parametric_spec(model).params, rtol=0.01
    )


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


def test_a_common_cause_groups_member_is_not_drawn_alone():
    # Its members carry one model (see test_ccf_analyses.py).
    unit = FixedEventProbability.from_params(0.1)
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(ValueError, match="together"):
        rbd.sf_uncertainty(uncertainty={"a": [unit]})


def test_a_model_on_the_edge_of_its_range_has_no_normal_approximation():
    model = surv.Exponential.fit(np.linspace(20, 400, 40))
    edge = surv.Exponential.fit(np.linspace(20, 400, 40))
    edge.params = np.array([0.0])
    with pytest.raises(ValueError, match="edge of its range"):
        one_node(edge).sf_uncertainty(10.0, {"c": "fit"}, n_draws=5)
    broken = surv.Exponential.fit(np.linspace(20, 400, 40))
    broken.covariance = lambda: np.array([[np.nan]])
    with pytest.raises(ValueError, match="finite"):
        one_node(broken).sf_uncertainty(10.0, {"c": "fit"}, n_draws=5)
    assert (
        one_node(model)
        .sf_uncertainty(10.0, {"c": "fit"}, n_draws=5, seed=0)
        .n_draws
        == 5
    )


# -- the MTTF, B-life and time to a reliability (#133) ------------------------


def pump_fit():
    times = W([100.0, 2.0]).qf(np.linspace(0.025, 0.975, 20))
    return surv.Weibull.fit(times)


def test_each_draws_mttf_and_times_are_its_models():
    fit = pump_fit()
    rbd = one_node(fit)
    drawn = rbd._uncertain_draws({"c": "fit"}, 300, 0)[0]["c"]
    mean = rbd.mean_uncertainty({"c": "fit"}, n_draws=300, seed=0)
    np.testing.assert_allclose(
        mean.samples, [m.mean() for m in drawn], rtol=1e-10
    )
    assert mean.nominal == pytest.approx(fit.mean(), rel=1e-10)
    t80 = rbd.time_to_reliability_uncertainty(
        0.8, {"c": "fit"}, n_draws=300, seed=0
    )
    np.testing.assert_allclose(
        t80.samples, [m.qf(0.2) for m in drawn], rtol=1e-9
    )
    assert t80.nominal == pytest.approx(fit.qf(0.2), rel=1e-9)
    b20 = rbd.bx_life_uncertainty(20, {"c": "fit"}, n_draws=300, seed=0)
    np.testing.assert_array_equal(b20.samples, t80.samples)
    assert b20.nominal == t80.nominal
    # The same draws as the reliability's.
    sf = rbd.sf_uncertainty(40.0, {"c": "fit"}, n_draws=300, seed=0)
    np.testing.assert_allclose(
        sf.samples, [m.sf(40.0) for m in drawn], rtol=1e-12
    )


def test_a_shared_rate_gives_the_mttfs_distribution():
    # Two pumps of one type in parallel, rate uniform on [0.001, 0.003]:
    # the MTTF is 1.5 / rate, so its percentiles are the rate's, inverted.
    pump = E([0.002])
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        {"p1": pump, "p2": pump},
    )
    rate = st.uniform(0.001, 0.002)
    result = rbd.mean_uncertainty(
        {("p1", "p2"): {"failure_rate": rate}}, n_draws=1500, seed=3
    )
    assert result.nominal == pytest.approx(750.0, rel=1e-10)
    for q in (0.05, 0.5, 0.95):
        assert_percentile(result.samples, 1.5 / rate.ppf(1 - q), q)
    # The system's B10: 1 - (1 - e^(-rate t))^2 = 0.9.
    b10 = rbd.bx_life_uncertainty(
        10, {("p1", "p2"): {"failure_rate": rate}}, n_draws=1500, seed=3
    )
    unit = 1 - math.sqrt(0.1)  # each pump's reliability at the B10
    for q in (0.05, 0.5, 0.95):
        assert_percentile(b10.samples, -math.log(unit) / rate.ppf(1 - q), q)


def test_a_system_draw_is_the_system_with_the_drawn_models():
    base = {n: W([100.0 + 20 * i, 1.5]) for i, n in enumerate("abcde")}
    choices = {"a": [W([80.0, 1.5]), W([140.0, 2.5])]}
    mean = bridge(base).mean_uncertainty(choices, n_draws=40, seed=6)
    t90 = bridge(base).time_to_reliability_uncertainty(
        0.9, choices, n_draws=40, seed=6
    )
    built = [bridge({**base, "a": a}) for a in choices["a"]]
    means = [b.mean() for b in built]
    times = [b.time_to_reliability(0.9) for b in built]
    for m, t in zip(mean.samples, t90.samples):
        assert min(abs(m - v) for v in means) < 1e-8 * m
        assert min(abs(t - v) for v in times) < 1e-8 * t
    assert {round(m, 6) for m in mean.samples} == {round(m, 6) for m in means}


def test_the_lifetime_uncertainties_refuse_what_they_cannot_do():
    fixed = one_node(FixedEventProbability.from_params(0.1))
    spec = {"c": [FixedEventProbability.from_params(0.2)]}
    with pytest.raises(ValueError, match="no lifetimes"):
        fixed.mean_uncertainty(spec)
    with pytest.raises(ValueError, match="does not vary with time"):
        fixed.time_to_reliability_uncertainty(0.9, spec)
    rbd = one_node(pump_fit())
    for target in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"in \(0, 1\)"):
            rbd.time_to_reliability_uncertainty(target, {"c": "fit"})
    with pytest.raises(ValueError, match="percentage"):
        rbd.bx_life_uncertainty(100, {"c": "fit"})
    with pytest.raises(ValueError, match="Give the uncertain nodes"):
        rbd.mean_uncertainty({})
    # A common-cause group splitting a probability has no exact MTTF.
    unit = E([0.01])
    grouped = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="split a failure"):
        grouped.mean_uncertainty({("a", "b"): [unit]})
    routes = grouped.analysis_routes()
    assert routes["mean_uncertainty"].route == "refused"
    assert routes["bx_life_uncertainty"].route == "simulated"
    assert rbd.analysis_routes()["bx_life_uncertainty"].route == "simulated"
