"""A surpyval Bayesian fit as a life: its posterior predictive
(``PosteriorPredictive``), ``S(t) = sum_i w_i S(t | theta_i)`` over the
posterior's points.

The hand-checked case is the conjugate one: exponential lives whose rate
has a Gamma(a, b) posterior (shape a, rate b) have the Lomax predictive

    S(t) = E[exp(-lambda t)] = (b / (b + t)) ** a,

with mean b / (a - 1) and quantile b ((1 - p) ** (-1 / a) - 1).
"""

import importlib.util
import json
import pickle

import numpy as np
import pytest
import surpyval as surv
from surpyval.bayes import BayesianParametric

from repyability import (
    NonRepairable,
    NonRepairableRBD,
    PosteriorPredictive,
    RepairableRBD,
)
from repyability.rbd import routes
from repyability.rbd._sampling import stream_sampler
from repyability.rbd.serialisation import rbd_from_dict

A, B = 5.0, 1250.0
N_DRAWS = 4000
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
REPAIR = surv.Exponential.from_params([0.05])


def lomax_sf(t):
    return (B / (B + np.asarray(t, dtype=float))) ** A


@pytest.fixture(scope="module")
def rates():
    return np.random.default_rng(1).gamma(A, 1.0 / B, N_DRAWS)


@pytest.fixture(scope="module")
def drawn(rates):
    """A fit from known draws of the Gamma(5, 1250) posterior."""
    return BayesianParametric.from_draws(surv.Exponential, rates[:, None])


@pytest.fixture(scope="module")
def weibull_fit():
    """A Weibull fit's posterior, 300 draws about alpha 200, beta 1.8."""
    g = np.random.default_rng(3)
    draws = np.c_[
        200 * g.lognormal(0, 0.15, 300), 1.8 * g.lognormal(0, 0.15, 300)
    ]
    return BayesianParametric.from_draws(surv.Weibull, draws)


def mixture_of(life: PosteriorPredictive):
    """The same predictive as an explicit surpyval mixture."""
    return surv.MixtureModel.from_params(
        surv.Weibull, life.points, life.weights
    )


# -- the life itself, hand-checked ---------------------------------------


def test_the_average_over_the_draws_exactly(drawn, rates):
    life = PosteriorPredictive(drawn)
    t = np.array([0.0, 1.0, 100.0, 1000.0, 10000.0])
    survival = np.exp(-np.outer(t, rates))
    np.testing.assert_allclose(life.sf(t), survival.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(
        life.ff(t), (-np.expm1(-np.outer(t, rates))).mean(axis=1), rtol=1e-12
    )
    np.testing.assert_allclose(
        life.df(t),
        (rates * np.exp(-np.outer(t, rates))).mean(axis=1),
        rtol=1e-12,
    )
    assert life.mean() == pytest.approx(np.mean(1.0 / rates), rel=1e-12)


def test_the_lomax_closed_form_within_the_draws_error(drawn, rates):
    # S(t) is the mean of exp(-lambda t) over 4000 draws: its Monte-Carlo
    # error is their standard deviation over sqrt(4000).
    life = PosteriorPredictive(drawn)
    for t in (10.0, 100.0, 1000.0, 5000.0):
        values = np.exp(-rates * t)
        error = values.std() / np.sqrt(N_DRAWS)
        assert abs(life.sf(t) - lomax_sf(t)) < 4 * error


def test_a_conjugate_fit_is_the_lomax_exactly():
    # Prior Gamma(2, 500) on the rate; 3 failures in 750 hours: the
    # posterior is Gamma(2 + 3, 500 + 750) = Gamma(5, 1250).
    fit = surv.Exponential.fit_bayes(
        [100, 250, 400],
        priors={"failure_rate": surv.Gamma.from_params([2.0, 500.0])},
    )
    life = PosteriorPredictive(fit)
    t = np.array([0.0, 1e-9, 100.0, 1e4])
    np.testing.assert_allclose(life.sf(t), lomax_sf(t), rtol=1e-14)
    # ff to its own precision: 1 - (1 + t / b) ** -a, at t = 1e-9 about
    # a t / b = 4e-12.
    exact = -np.expm1(-A * np.log1p(1e-9 / B))
    assert life.ff(1e-9) == pytest.approx(exact, rel=1e-14)
    assert life.mean() == pytest.approx(B / (A - 1), rel=1e-14)
    assert life.qf(0.5) == pytest.approx(B * (2 ** (1 / A) - 1), rel=1e-14)
    assert life.sf(100.0) == pytest.approx(float(fit.sf(100.0)), rel=1e-14)


def test_each_function_to_its_own_precision(weibull_fit):
    life = PosteriorPredictive(weibull_fit)
    P, w = life.points, life.weights
    t = np.array([1e-6, 1e-3, 1.0, 100.0, 400.0, 2000.0])
    ff = (w * surv.Weibull.ff(t[:, None], P[:, 0], P[:, 1])).sum(axis=1)
    sf = (w * surv.Weibull.sf(t[:, None], P[:, 0], P[:, 1])).sum(axis=1)
    df = (w * surv.Weibull.df(t[:, None], P[:, 0], P[:, 1])).sum(axis=1)
    np.testing.assert_allclose(life.ff(t), ff, rtol=1e-12)
    np.testing.assert_allclose(life.sf(t), sf, rtol=1e-12)
    # The hazard is the predictive's, df / sf, not the points' averaged.
    np.testing.assert_allclose(life.hf(t), df / sf, rtol=1e-12)
    # -log sf, from ff while it is small (ff about 1e-11 at 1e-6).
    np.testing.assert_allclose(life.Hf(t[:2]), -np.log1p(-ff[:2]), rtol=1e-12)
    assert life.cs(50.0, 100.0) == pytest.approx(
        life.sf(150.0) / life.sf(100.0), rel=1e-15
    )


def test_its_quantile_inverts_it_in_both_tails(weibull_fit):
    life = PosteriorPredictive(weibull_fit)
    p = np.array([1e-12, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-6, 1 - 1e-10])
    q = life.qf(p)
    low = p < 0.5
    np.testing.assert_allclose(life.ff(q[low]), p[low], rtol=1e-12)
    np.testing.assert_allclose(life.sf(q[~low]), 1 - p[~low], rtol=1e-9)
    t = np.array([1.0, 50.0, 200.0, 800.0])
    np.testing.assert_allclose(life.qf(life.ff(t)), t, rtol=1e-10)


def test_its_mean_is_the_area_under_its_survival(weibull_fit):
    from scipy.integrate import quad

    life = PosteriorPredictive(weibull_fit)
    area = quad(lambda t: float(life.sf(t)), 0, np.inf, limit=200)[0]
    assert life.mean() == pytest.approx(area, rel=1e-8)


def test_its_draws_have_its_distribution(weibull_fit):
    life = PosteriorPredictive(weibull_fit)
    np.random.seed(4)
    lives = life.random(20000)
    for t in (50.0, 200.0, 500.0):
        p = float(life.sf(t))
        assert abs(np.mean(lives > t) - p) < 4 * np.sqrt(p * (1 - p) / 2e4)
    # A repairable diagram's streams draw it from one uniform a life.
    u = np.random.default_rng(5).random(1000)
    assert np.array_equal(stream_sampler(life)(u), life._draw(u))
    assert np.all(np.isfinite(life._draw(np.array([0.0, 1 - 2**-53]))))


def test_no_result_depends_on_the_unit_of_time(weibull_fit):
    life = PosteriorPredictive(weibull_fit)
    for k in (1e-6, 1e6):
        draws = weibull_fit.posterior.points * [k, 1.0]
        scaled = PosteriorPredictive(
            BayesianParametric.from_draws(surv.Weibull, draws)
        )
        assert scaled.sf(100.0 * k) == pytest.approx(life.sf(100.0), 1e-12)
        assert scaled.mean() == pytest.approx(k * life.mean(), rel=1e-12)
        assert scaled.qf(0.3) == pytest.approx(k * life.qf(0.3), rel=1e-12)


# -- in a diagram ------------------------------------------------------------


def test_a_block_in_series_and_in_parallel(drawn, weibull_fit):
    other = surv.Weibull.from_params([300.0, 2.0])
    life = PosteriorPredictive(drawn)
    t = np.array([10.0, 100.0, 500.0])
    series = NonRepairableRBD(SERIES, {"a": drawn, "b": other})
    np.testing.assert_allclose(
        series.sf(t), life.sf(t) * other.sf(t), rtol=1e-14
    )
    parallel = NonRepairableRBD(PARALLEL, {"a": drawn, "b": other})
    np.testing.assert_allclose(
        parallel.ff(t), life.ff(t) * other.ff(t), rtol=1e-14
    )
    assert isinstance(series.reliabilities["a"], PosteriorPredictive)
    assert "PosteriorPredictive" in repr(series.reliabilities["a"])
    # The MTTF of the series: the area under its reliability.
    from scipy.integrate import quad

    area = quad(lambda x: float(series.sf(x)), 0, np.inf, limit=200)[0]
    assert series.mean() == pytest.approx(area, rel=1e-7)
    route = series.analysis_routes()["sf"]
    assert route.route == routes.EXACT


def test_a_repairable_component(weibull_fit):
    rbd = RepairableRBD(
        PARALLEL,
        {
            "a": {"reliability": weibull_fit, "repairability": REPAIR},
            "b": {
                "reliability": surv.Weibull.from_params([150.0, 2.0]),
                "repairability": REPAIR,
            },
        },
    )
    assert isinstance(rbd.components["a"].reliability, PosteriorPredictive)
    simulated = rbd.availability(
        t_simulation=2000.0,
        mc_samples=400,
        seed=11,
        control_variate=False,
        conditional=False,
    )
    exact = rbd.mission_availability(2000.0)
    error = simulated.mean_availability_interval().standard_error
    assert abs(simulated.mean_availability - exact) < 4 * error
    # Its long-run availability is MTTF / (MTTF + MTTR).
    single = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": weibull_fit, "repairability": REPAIR}},
    )
    life = PosteriorPredictive(weibull_fit)
    assert single.mean_availability() == pytest.approx(
        life.mean() / (life.mean() + 20.0), rel=1e-9
    )
    engine = rbd.analysis_routes()["availability"].engine
    if importlib.util.find_spec("numba") is None:
        return
    assert engine == "numba"
    options = dict(
        t_simulation=500.0, mc_samples=50, seed=5, control_variate=False
    )
    python = rbd.availability(engine="python", **options)
    numba = rbd.availability(engine="numba", **options)
    assert python.mean_availability == numba.mean_availability


def test_a_part_and_its_maintenance_as_the_explicit_mixture(weibull_fit):
    small = BayesianParametric.from_draws(
        surv.Weibull, weibull_fit.posterior.points[:30]
    )
    life = PosteriorPredictive(small)
    part = NonRepairable(small, planned_cost=1.0, unplanned_cost=5.0)
    same = NonRepairable(
        mixture_of(life), planned_cost=1.0, unplanned_cost=5.0
    )
    assert isinstance(part.reliability, PosteriorPredictive)
    assert part.find_optimal_replacement() == pytest.approx(
        same.find_optimal_replacement(), rel=1e-6
    )
    policy, other = (
        part.optimal_replacement_policy(),
        same.optimal_replacement_policy(),
    )
    assert policy.cost_rate == pytest.approx(other.cost_rate, rel=1e-9)
    ours, theirs = part.failures(300.0, age=50.0), same.failures(300.0, 50.0)
    assert ours.expected_failures == pytest.approx(
        theirs.expected_failures, rel=1e-9
    )
    assert ours.probability == pytest.approx(theirs.probability, rel=1e-12)


# -- saving, refusals ------------------------------------------------------


def test_saved_whole_and_pickled(weibull_fit):
    rbd = NonRepairableRBD(SERIES, {"a": weibull_fit, "b": REPAIR})
    saved = json.loads(json.dumps(rbd.to_dict()))
    back = rbd_from_dict(saved)
    assert isinstance(back.reliabilities["a"], PosteriorPredictive)
    assert back.sf(150.0) == rbd.sf(150.0)
    again = pickle.loads(pickle.dumps(rbd))
    assert again.sf(150.0) == rbd.sf(150.0)
    life = PosteriorPredictive(weibull_fit)
    restored = PosteriorPredictive.from_dict(
        json.loads(json.dumps(life.to_dict()))
    )
    assert np.array_equal(restored.points, life.points)
    repairable = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": weibull_fit, "repairability": REPAIR}},
    )
    loaded = RepairableRBD.from_dict(
        json.loads(json.dumps(repairable.to_dict()))
    )
    assert loaded.mean_availability() == repairable.mean_availability()


def test_parameter_uncertainty_draws_its_posterior(weibull_fit):
    # "fit" draws the posterior itself, by default too (its credible band:
    # test_bayesian_bands.py); its posterior's models also go in as a list.
    rbd = NonRepairableRBD(SERIES, {"a": weibull_fit, "b": REPAIR})
    assert rbd.sf_uncertainty(100.0, n_draws=20, seed=1).band == "credible"
    models = weibull_fit.draw_models(20, random_state=1)
    result = rbd.sf_uncertainty(100.0, {"a": models}, n_draws=20, seed=1)
    assert result is not None


def test_what_it_does_not_take():
    with pytest.raises(TypeError, match="Bayesian fit"):
        PosteriorPredictive(surv.Weibull.from_params([1.0, 2.0]))
    counts = BayesianParametric.from_draws(
        surv.Poisson, np.linspace(1.0, 2.0, 5)[:, None]
    )
    with pytest.raises(ValueError, match="count, not a life"):
        NonRepairableRBD(SERIES, {"a": counts, "b": REPAIR})


# -- an offset, a limited failure population, zero inflation ---------------

EXTRAS = {"gamma": 20.0, "lfp_p": 0.8, "f0": 0.05}


def one_point(**extras):
    """A posterior that is one point (five equal draws), with ``extras``:
    its predictive is the plain model at that point."""
    names = ["alpha", "beta", *extras]
    row = [200.0, 1.8, *extras.values()]
    fit = BayesianParametric.from_draws(
        surv.Weibull, np.tile(row, (5, 1)), names
    )
    return fit, surv.Weibull.from_params([200.0, 1.8], **extras)


@pytest.fixture(scope="module")
def shares_fit():
    """400 draws, each with its own offset, share that ever fails and
    share dead on arrival."""
    g = np.random.default_rng(5)
    draws = np.c_[
        200 * g.lognormal(0, 0.15, 400),
        1.8 * g.lognormal(0, 0.15, 400),
        g.uniform(10.0, 30.0, 400),
        g.uniform(0.7, 0.9, 400),
        g.uniform(0.01, 0.05, 400),
    ]
    return BayesianParametric.from_draws(
        surv.Weibull, draws, ["alpha", "beta", "gamma", "lfp_p", "f0"]
    )


@pytest.mark.parametrize(
    "extras",
    [{"gamma": 20.0}, {"lfp_p": 0.8}, {"f0": 0.05}, EXTRAS],
    ids=["offset", "lfp", "zi", "all three"],
)
def test_one_point_with_extras_is_the_plain_model(extras):
    fit, plain = one_point(**extras)
    life = PosteriorPredictive(fit)
    t = np.array([5.0, 25.0, 100.0, 300.0, 3000.0])
    for name in ("sf", "ff", "Hf"):
        assert getattr(life, name)(t) == pytest.approx(
            getattr(plain, name)(t), rel=1e-12
        )
    p = np.array([0.001, 0.1, 0.5, 0.75])
    assert life.qf(p) == pytest.approx(plain.qf(p), rel=1e-10)
    assert life.never_fails == pytest.approx(1 - extras.get("lfp_p", 1.0))
    # In each kind of diagram it is the plain model.
    ours = NonRepairableRBD(SERIES, {"a": fit, "b": REPAIR})
    theirs = NonRepairableRBD(SERIES, {"a": plain, "b": REPAIR})
    assert ours.sf(t) == pytest.approx(theirs.sf(t), rel=1e-12)
    part = NonRepairable(fit, planned_cost=1.0, unplanned_cost=5.0)
    same = NonRepairable(plain, planned_cost=1.0, unplanned_cost=5.0)
    assert part.find_optimal_replacement() == pytest.approx(
        same.find_optimal_replacement(), rel=1e-6
    )
    assert part.failures(300.0, age=50.0).expected_failures == pytest.approx(
        same.failures(300.0, age=50.0).expected_failures, rel=1e-9
    )
    for life_model, other in ((fit, plain),):
        a = RepairableRBD(
            [("s", "a"), ("a", "t")],
            {"a": {"reliability": life_model, "repairability": REPAIR}},
        )
        b = RepairableRBD(
            [("s", "a"), ("a", "t")],
            {"a": {"reliability": other, "repairability": REPAIR}},
        )
        assert a.mission_availability(500.0) == pytest.approx(
            b.mission_availability(500.0), rel=1e-9
        )


def test_with_extras_its_functions_are_the_fits(shares_fit):
    life = PosteriorPredictive(shares_fit)
    t = np.array([0.0, 5.0, 25.0, 100.0, 300.0, 1e5])
    for name in ("sf", "ff", "df", "hf", "Hf"):
        assert np.array_equal(
            getattr(life, name)(t), getattr(shares_fit, name)(t)
        )
    p = life.extras["lfp_p"]
    assert life.never_fails == pytest.approx(1 - np.mean(p), rel=1e-12)
    assert life.sf(1e9) == pytest.approx(life.never_fails, rel=1e-9)
    assert life.mean() == np.inf


def test_with_extras_its_draws_have_its_distribution(shares_fit):
    """Uniforms on a fine grid drawn into lives: the share at or below t
    is the predictive's F(t), to the grid's step (each point's share of
    [0, 1) is cut into equal steps)."""
    life = PosteriorPredictive(shares_fit)
    n = 400 * 2000
    lives = life._draw((np.arange(n) + 0.5) / n)
    for t in (0.0, 15.0, 40.0, 150.0, 400.0, 1e6):
        assert np.mean(lives <= t) == pytest.approx(
            float(life.ff(t)), abs=2e-3
        )
    # Units dead on arrival live 0; those that never fail, forever.
    assert np.mean(lives == 0.0) == pytest.approx(
        np.mean(life.extras["f0"]), abs=1e-3
    )
    assert np.mean(np.isinf(lives)) == pytest.approx(
        life.never_fails, abs=1e-3
    )


def test_with_extras_its_band_is_the_posterior_quantiles(shares_fit):
    """The credible band of R(t): the quantiles of each draw's R(t) at the
    posterior's points, picked by their probabilities."""
    t = 150.0
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": shares_fit})
    result = rbd.sf_uncertainty(t, n_draws=4000, seed=3)
    assert result.band == "credible"
    names = shares_fit.posterior.parameter_names
    values = np.array(
        [
            surv.Weibull.from_params(
                [row[names.index("alpha")], row[names.index("beta")]],
                **{k: row[names.index(k)] for k in ("gamma", "lfp_p", "f0")},
            ).sf(t)
            for row in shares_fit.posterior.points
        ]
    )
    lower, upper = result.interval(0.9)
    assert lower == pytest.approx(np.quantile(values, 0.05), rel=0.02)
    assert upper == pytest.approx(np.quantile(values, 0.95), rel=0.02)
    assert result.nominal == pytest.approx(float(shares_fit.sf(t)), 1e-12)


def test_with_extras_a_repairable_availability_against_simulation(
    shares_fit,
):
    rbd = RepairableRBD(
        PARALLEL,
        {
            "a": {"reliability": shares_fit, "repairability": REPAIR},
            "b": {
                "reliability": surv.Weibull.from_params([150.0, 2.0]),
                "repairability": REPAIR,
            },
        },
    )
    options = dict(t_simulation=1500.0, mc_samples=400, control_variate=False)
    simulated = rbd.availability(seed=11, conditional=False, **options)
    exact = rbd.mission_availability(1500.0)
    error = simulated.mean_availability_interval().standard_error
    assert abs(simulated.mean_availability - exact) < 4 * error
    again = rbd.availability(seed=11, conditional=False, **options)
    assert again.mean_availability == simulated.mean_availability
    # Some units never fail: in the long run the system is up for good.
    assert rbd.mean_availability() == 1.0


def test_with_extras_saved_whole(shares_fit):
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": shares_fit, "repairability": REPAIR}},
    )
    loaded = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    assert loaded.mission_availability(400.0) == rbd.mission_availability(
        400.0
    )
