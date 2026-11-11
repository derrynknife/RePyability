"""A Bayesian block's credible band, carried to the system's result.

The parameter-uncertainty methods draw a Bayesian block from its posterior:
each draw a posterior point picked by its probability (from the exact
family of a conjugate posterior), the block in that draw the plain
distribution at it. The checks are closed forms of the posterior: with an
exponential life whose rate ``lam`` has a Gamma(a, b) posterior (shape a,
rate b), a system value that is monotone in ``lam`` has its band at the
value of the Gamma's quantiles.

Monte-Carlo error: a band's end is an empirical quantile of ``n`` draws.
Rather than the value, its *probability level* is checked: the exact
distribution's probability below the band's end is the level aimed at,
``p``, within four standard errors of an empirical quantile's level,
``4 sqrt(p (1 - p) / n)`` (about 0.0062 for p = 0.05 and n = 20000).
"""

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv
from surpyval.bayes import BayesianParametric

from repyability import NonRepairableRBD, RepairableRBD
from repyability.rbd import routes

E = surv.Exponential.from_params

#: A conjugate exponential fit: prior Gamma(2, 500) on the rate, and four
#: failures in 1550 h, so the posterior is Gamma(6, 2050).
DATA = [100.0, 250.0, 400.0, 800.0]
A, B = 6.0, 2050.0
#: Draws of each band, and its ends' probability levels' tolerance.
N = 20_000
P = 0.05
TOL = 4 * math.sqrt(P * (1 - P) / N)


def conjugate(data=DATA, prior=(2.0, 500.0)):
    return surv.Exponential.fit_bayes(
        data, priors={"failure_rate": surv.Gamma.from_params(list(prior))}
    )


def gamma(a=A, b=B):
    """The posterior of the rate, Gamma(a, b) (shape a, rate b)."""
    return st.gamma(a, scale=1.0 / b)


def one_block(fit):
    return NonRepairableRBD([("s", "p"), ("p", "t")], {"p": fit})


def test_the_fit_is_the_gamma_posterior_this_file_takes():
    assert conjugate().posterior.exact == {
        "name": "failure_rate",
        "family": "gamma",
        "params": [A, B],
    }


def test_one_block_its_band_the_posterior_quantiles_of_its_reliability():
    """R(t) = exp(-lam t), falling in lam: the band's lower end is
    exp(-t q_0.95), its upper exp(-t q_0.05), q the Gamma(6, 2050)
    quantiles. Checked by level: P(exp(-lam t) <= lower) = P(lam >= -log
    lower / t) = 0.05, within 4 standard errors. The point value is the
    Lomax predictive, (b / (b + t)) ** a."""
    t = 300.0
    result = one_block(conjugate()).sf_uncertainty(t, n_draws=N, seed=1)
    lower, upper = result.interval(0.9)
    assert gamma().sf(-math.log(lower) / t) == pytest.approx(P, abs=TOL)
    assert gamma().cdf(-math.log(upper) / t) == pytest.approx(P, abs=TOL)
    # Against the closed form itself, to the error that level implies.
    q05, q95 = gamma().ppf([0.05, 0.95])
    assert lower == pytest.approx(math.exp(-t * q95), rel=0.03)
    assert upper == pytest.approx(math.exp(-t * q05), rel=0.01)
    assert result.nominal == pytest.approx((B / (B + t)) ** A, rel=1e-12)
    # The predictive is the posterior mean of exp(-lam t): the draws'
    # mean is within 4 standard errors of it.
    se = result.std / math.sqrt(N)
    assert abs(result.mean - result.nominal) < 4 * se
    assert result.band == "credible"


def test_a_fit_from_fixed_draws_resamples_them():
    """A fit from another sampler's draws (equal weights): each band end's
    level among the draws themselves is 0.05 (or 0.95), within 4
    standard errors."""
    rates = np.random.default_rng(3).gamma(A, 1.0 / B, 4000)
    fit = BayesianParametric.from_draws(surv.Exponential, rates[:, None])
    t = 300.0
    result = one_block(fit).sf_uncertainty(t, n_draws=N, seed=2)
    lower, upper = result.interval(0.9)
    values = np.exp(-rates * t)
    assert np.mean(values <= lower) == pytest.approx(P, abs=TOL)
    assert np.mean(values >= upper) == pytest.approx(P, abs=TOL)
    # Every drawn model is one of the draws' distributions.
    drawn = one_block(fit)._uncertain_draws(None, 50, 0)[0]["p"]
    assert {float(m.params[0]) for m in drawn} <= set(rates.tolist())


def test_a_grid_posterior_picks_its_points_by_their_weights():
    """A grid's points have unequal weights (some none): the band's ends
    have the weighted levels of R(t | theta_i) over the points, and no
    point of no weight is drawn."""
    fit = surv.Weibull.fit_bayes(
        [2100, 3400, 4050, 5000, 5000],
        c=[0, 0, 0, 1, 1],
        priors={"beta": surv.prior.around(2, between=(1.5, 3))},
    )
    weights = fit.posterior.weights
    assert weights.min() == 0.0 and weights.max() > 10 / len(weights)
    rbd = one_block(fit)
    life = rbd.reliabilities["p"]
    t = 3000.0
    result = rbd.sf_uncertainty(t, n_draws=N, seed=4)
    lower, upper = result.interval(0.9)
    values = surv.Weibull.sf(t, life.points[:, 0], life.points[:, 1])
    assert weights @ (values <= lower) == pytest.approx(P, abs=TOL)
    assert weights @ (values >= upper) == pytest.approx(P, abs=TOL)
    drawn = rbd._uncertain_draws(None, 2000, 5)[0]["p"]
    held = {tuple(row) for row in life.points[life.weights > 0]}
    assert all(tuple(m.params) in held for m in drawn)


def test_a_normal_mean_with_its_spread_known_from_its_exact_family():
    """A Normal life with its spread known (sigma = 1), its mean's
    posterior Normal(m, s) exactly: R(t) = 1 - Phi((t - mu) / sigma) rises
    with mu, so the band's lower end is at mu's 5th percentile (by level).
    Each drawn model keeps the known spread."""
    fit = surv.Normal.fit_bayes(
        [10.0, 11.0, 12.0, 9.0],
        fixed={"sigma": 1.0},
        priors={"mu": surv.Normal.from_params([10.0, 3.0])},
    )
    assert fit.posterior.exact["family"] == "normal"
    m, s = fit.posterior.exact["params"]
    rbd = one_block(fit)
    t = 9.5
    result = rbd.sf_uncertainty(t, n_draws=N, seed=14)
    lower, upper = result.interval(0.9)
    mean = st.norm(m, s)
    assert mean.cdf(t - st.norm.ppf(1 - lower)) == pytest.approx(P, abs=TOL)
    assert mean.sf(t - st.norm.ppf(1 - upper)) == pytest.approx(P, abs=TOL)
    drawn = rbd._uncertain_draws(None, 20, 0)[0]["p"]
    assert all(float(model.params[1]) == 1.0 for model in drawn)


def test_two_blocks_in_series_the_sum_of_their_rates():
    """R = exp(-(lam1 + lam2) t). With one Gamma prior, data of one total
    time and different counts, the posteriors share their rate b: the sum
    is Gamma(a1 + a2, b), and the band is at its quantiles."""
    first = conjugate([100.0, 250.0, 400.0, 800.0])  # Gamma(6, 2050)
    second = conjugate([1550.0])  # Gamma(3, 2050)
    rbd = NonRepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")], {"p": first, "q": second}
    )
    t = 200.0
    result = rbd.sf_uncertainty(t, n_draws=N, seed=5)
    lower, upper = result.interval(0.9)
    total = gamma(9.0, B)
    assert total.sf(-math.log(lower) / t) == pytest.approx(P, abs=TOL)
    assert total.cdf(-math.log(upper) / t) == pytest.approx(P, abs=TOL)
    # The point value: the product of the two Lomax predictives.
    assert result.nominal == pytest.approx(
        (B / (B + t)) ** 6 * (B / (B + t)) ** 3, rel=1e-12
    )


def test_two_blocks_in_series_of_unequal_scales_against_a_monte_carlo():
    """Posteriors Gamma(6, 2050) and Gamma(3, 800): the sum has no simple
    closed form, so the band's levels are checked against 10**6 draws of
    our own (their error, 4 sqrt(0.05 0.95 / 10**6) = 0.0009, is within
    the tolerance's)."""
    first = conjugate()
    second = conjugate([300.0], prior=(2.0, 500.0))  # Gamma(3, 800)
    assert second.posterior.exact["params"] == [3.0, 800.0]
    rbd = NonRepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")], {"p": first, "q": second}
    )
    t = 200.0
    result = rbd.sf_uncertainty(t, n_draws=N, seed=6)
    lower, upper = result.interval(0.9)
    g = np.random.default_rng(0)
    own = np.exp(
        -(g.gamma(6.0, 1 / 2050.0, 10**6) + g.gamma(3.0, 1 / 800.0, 10**6)) * t
    )
    assert np.mean(own <= lower) == pytest.approx(P, abs=TOL + 0.0009)
    assert np.mean(own >= upper) == pytest.approx(P, abs=TOL + 0.0009)


def test_a_parallel_pair_of_one_fit_shares_one_draw():
    """Two blocks given the same fit object are one population: one draw
    of lam for both (as one ordinary fit object shared by nodes is). So
    R = 1 - (1 - exp(-lam t)) ** 2, falling in lam, and the band is at the
    Gamma's quantiles: lower = R(q_0.95)."""
    fit = conjugate()
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        {"p1": fit, "p2": fit},
    )
    assert rbd.reliabilities["p1"] is rbd.reliabilities["p2"]
    assert rbd._fitted_uncertainty() == {("p1", "p2"): "fit"}
    t = 300.0
    result = rbd.sf_uncertainty(t, n_draws=N, seed=7)
    lower, upper = result.interval(0.9)

    def rate_at(r):
        # R = 1 - (1 - s)**2 inverted for s = exp(-lam t), then lam.
        return -math.log(1.0 - math.sqrt(1.0 - r)) / t

    assert gamma().sf(rate_at(lower)) == pytest.approx(P, abs=TOL)
    assert gamma().cdf(rate_at(upper)) == pytest.approx(P, abs=TOL)
    # The point value is the diagram's own: each block's life the Lomax
    # predictive L, the two lives apart, 1 - (1 - L)^2. The draws' mean
    # is the system's reliability averaged over the one shared lam,
    # E[1 - (1 - e^(-lam t))^2] = 2 L(t) - L(2t), lower (by Jensen: the
    # blocks' failures go together when they share lam).
    lomax = (B / (B + t)) ** A
    assert result.nominal == pytest.approx(1 - (1 - lomax) ** 2, rel=1e-12)
    shared = 2 * lomax - (B / (B + 2 * t)) ** A
    assert shared < result.nominal
    assert abs(result.mean - shared) < 4 * result.std / math.sqrt(N)


def test_a_parallel_pair_of_two_fits_draws_them_apart():
    """Two fit objects (even of the same data) are drawn apart, as two
    ordinary fits are: R = 1 - (1 - e^(-lam1 t))(1 - e^(-lam2 t)) with
    lam1, lam2 independent, checked against 10**6 draws of our own."""
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        {"p1": conjugate(), "p2": conjugate()},
    )
    assert set(rbd._fitted_uncertainty()) == {"p1", "p2"}
    t = 300.0
    result = rbd.sf_uncertainty(t, n_draws=N, seed=8)
    lower, upper = result.interval(0.9)
    g = np.random.default_rng(1)
    lam = g.gamma(A, 1 / B, (2, 10**6))
    own = 1 - np.prod(1 - np.exp(-lam * t), axis=0)
    assert np.mean(own <= lower) == pytest.approx(P, abs=TOL + 0.0009)
    assert np.mean(own >= upper) == pytest.approx(P, abs=TOL + 0.0009)


def test_one_of_a_shared_fits_nodes_drawn_alone_is_warned_about():
    """As for an ordinary fit object: drawing one of the nodes that hold
    it, the other kept at its predictive, understates the band."""
    fit = conjugate()
    rbd = NonRepairableRBD(
        [("s", "p1"), ("p1", "p2"), ("p2", "t")], {"p1": fit, "p2": fit}
    )
    with pytest.warns(UserWarning, match="drawn without 'p2'"):
        rbd.sf_uncertainty(100.0, {"p1": "fit"}, n_draws=10, seed=0)


def test_the_mttf_band_the_quantiles_of_one_over_the_rate():
    """MTTF = 1 / lam: lower = 1 / q_0.95, upper = 1 / q_0.05 (by level).
    The point value is the predictive's mean, b / (a - 1)."""
    result = one_block(conjugate()).mean_uncertainty(n_draws=N, seed=9)
    lower, upper = result.interval(0.9)
    assert gamma().sf(1.0 / lower) == pytest.approx(P, abs=TOL)
    assert gamma().cdf(1.0 / upper) == pytest.approx(P, abs=TOL)
    assert result.nominal == pytest.approx(B / (A - 1.0), rel=1e-9)
    assert result.band == "credible"


def test_the_time_to_a_reliability_and_a_b_life():
    """t(R) = -log R / lam, falling in lam: the B10 band's lower end is
    -log 0.9 / q_0.95 (by level)."""
    rbd = one_block(conjugate())
    result = rbd.bx_life_uncertainty(10, n_draws=4000, seed=10)
    lower, upper = result.interval(0.9)
    tol = 4 * math.sqrt(P * (1 - P) / 4000)
    assert gamma().sf(-math.log(0.9) / lower) == pytest.approx(P, abs=tol)
    assert gamma().cdf(-math.log(0.9) / upper) == pytest.approx(P, abs=tol)
    assert result.band == "credible"


def test_a_repairable_availability_band():
    """A component with the Bayesian life and an exponential repair at
    rate mu: A = mu / (lam + mu), falling in lam, so lower = mu / (q_0.95
    + mu) (by level, 4000 draws). The point value is the predictive's:
    MTTF / (MTTF + MTTR), MTTF = b / (a - 1)."""
    mu = 0.01
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": conjugate(), "repairability": E([mu])}},
    )
    n = 4000
    tol = 4 * math.sqrt(P * (1 - P) / n)
    result = rbd.mean_availability_uncertainty(n_draws=n, seed=11)
    lower, upper = result.interval(0.9)
    assert gamma().sf(mu / lower - mu) == pytest.approx(P, abs=tol)
    assert gamma().cdf(mu / upper - mu) == pytest.approx(P, abs=tol)
    mttf = B / (A - 1.0)
    assert result.nominal == pytest.approx(mttf / (mttf + 1.0 / mu), rel=1e-9)
    assert result.band == "credible"


def test_a_bayesian_repair_is_drawn_too():
    """A Bayesian fit as a repair time is drawn from its posterior: with
    the life exponential at lam, A = mu / (lam + mu) rises in mu, so the
    band's upper end is at q_0.95 of mu's posterior (by level)."""
    lam = 0.002
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": E([lam]), "repairability": conjugate()}},
    )
    n = 4000
    tol = 4 * math.sqrt(P * (1 - P) / n)
    result = rbd.mean_availability_uncertainty(n_draws=n, seed=12)
    lower, upper = result.interval(0.9)

    def rate_at(a):
        return a * lam / (1.0 - a)

    assert gamma().cdf(rate_at(lower)) == pytest.approx(P, abs=tol)
    assert gamma().sf(rate_at(upper)) == pytest.approx(P, abs=tol)


def _weibull_draws(fit, n, g):
    """Weibull parameters from the normal approximation of a fit, on the
    log scale (both are positive): our own draws, for the mixed band."""
    params = np.asarray(fit.params, float)
    cov = np.asarray(fit.covariance(), float) / np.outer(params, params)
    z = g.multivariate_normal(np.log(params), cov, n)
    return np.exp(z)


def test_a_mixed_band_over_both_kinds_of_uncertainty():
    """A Weibull fitted by maximum likelihood (its covariance) in series
    with the Bayesian block. The band over both is wider than with either
    held (the other drawn alone), and its levels agree with 10**6 draws
    of our own of both (Weibull on the log scale, the rate from its
    Gamma posterior), within the tolerance and their error (0.0009)."""
    data = surv.Weibull.from_params([1000.0, 2.0]).qf(
        np.linspace(0.05, 0.95, 15)
    )
    weibull = surv.Weibull.fit(data)
    rbd = NonRepairableRBD(
        [("s", "w"), ("w", "e"), ("e", "t")], {"w": weibull, "e": conjugate()}
    )
    t = 300.0
    both = rbd.sf_uncertainty(t, n_draws=N, seed=13)
    alone_w = rbd.sf_uncertainty(t, {"w": "fit"}, n_draws=N, seed=13)
    alone_e = rbd.sf_uncertainty(t, {"e": "fit"}, n_draws=N, seed=13)
    assert (both.band, alone_w.band, alone_e.band) == (
        "mixed",
        "confidence",
        "credible",
    )

    def width(r):
        lower, upper = r.interval(0.9)
        return upper - lower

    assert width(both) > max(width(alone_w), width(alone_e))
    g = np.random.default_rng(2)
    alpha, beta = _weibull_draws(weibull, 10**6, g).T
    own = np.exp(-((t / alpha) ** beta)) * np.exp(
        -g.gamma(A, 1 / B, 10**6) * t
    )
    lower, upper = both.interval(0.9)
    assert np.mean(own <= lower) == pytest.approx(P, abs=TOL + 0.0009)
    assert np.mean(own >= upper) == pytest.approx(P, abs=TOL + 0.0009)
    # The point value stays the fit's and the predictive's.
    assert both.nominal == pytest.approx(
        float(weibull.sf(t)) * (B / (B + t)) ** A, rel=1e-12
    )


def test_given_distributions_and_lists_say_so():
    rbd = NonRepairableRBD(
        [("s", "w"), ("w", "e"), ("e", "t")],
        {"w": E([0.001]), "e": conjugate()},
    )
    given = {"failure_rate": st.uniform(0.0005, 0.001)}
    assert rbd.sf_uncertainty(100.0, {"w": given}, n_draws=5, seed=0).band == (
        "given"
    )
    mixed = rbd.sf_uncertainty(
        100.0, {"w": given, "e": "fit"}, n_draws=5, seed=0
    )
    assert mixed.band == "mixed"
    # A list of models drawn from the fit is still taken.
    models = conjugate().draw_models(20, random_state=1)
    assert (
        rbd.sf_uncertainty(100.0, {"e": models}, n_draws=5, seed=0).band
        == "given"
    )


def test_seeded_draws_are_reproducible_and_sobol_takes_them():
    rbd = one_block(conjugate())
    first = rbd.sf_uncertainty([100.0, 300.0], n_draws=500, seed=21)
    again = rbd.sf_uncertainty([100.0, 300.0], n_draws=500, seed=21)
    other = rbd.sf_uncertainty([100.0, 300.0], n_draws=500, seed=22)
    assert np.array_equal(first.samples, again.samples)
    assert not np.array_equal(first.samples, other.samples)
    # Sobol points: one dimension for the block's draw, the band by level
    # (a quasi-random sample is at least as precise as a random one).
    t = 300.0
    sobol = rbd.sf_uncertainty(t, n_draws=2**14, seed=3, sampling="sobol")
    lower, upper = sobol.interval(0.9)
    tol = 4 * math.sqrt(P * (1 - P) / 2**14)
    assert gamma().sf(-math.log(lower) / t) == pytest.approx(P, abs=tol)
    assert gamma().cdf(-math.log(upper) / t) == pytest.approx(P, abs=tol)
    again = rbd.sf_uncertainty(t, n_draws=2**14, seed=3, sampling="sobol")
    assert np.array_equal(sobol.samples, again.samples)


def test_uncertainty_importance_takes_it_by_sobol_and_refuses_the_delta():
    """The Sobol indices draw from the posterior. The delta method moves
    parameters about one value, which a posterior has not: it refuses in
    plain words, pointing to method='sobol'."""
    rbd = NonRepairableRBD(
        [("s", "w"), ("w", "e"), ("e", "t")],
        {"w": E([0.001]), "e": conjugate()},
    )
    spec = {"w": {"failure_rate": st.uniform(0.0005, 0.001)}, "e": "fit"}
    parts = rbd.uncertainty_importance(
        100.0, spec, method="sobol", n_draws=2000, seed=0
    )
    assert set(parts.first_order) == {"w", "e"}
    with pytest.raises(ValueError, match="Bayesian fit.*method='sobol'"):
        rbd.uncertainty_importance(100.0, spec)
    repairable = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": conjugate(), "repairability": E([0.01])}},
    )
    with pytest.raises(ValueError, match="Bayesian fit.*method='sobol'"):
        repairable.uncertainty_importance()
    sobol = repairable.uncertainty_importance(
        method="sobol", n_draws=200, seed=0
    )
    assert list(sobol.first_order) == ["a"]


def test_parameters_given_by_name_are_refused_in_plain_words():
    rbd = one_block(conjugate())
    with pytest.raises(ValueError, match="its posterior.*give 'fit'"):
        rbd.sf_uncertainty(
            100.0, {"p": {"failure_rate": st.uniform(0.001, 0.002)}}
        )


def test_the_routes_say_how_it_is_drawn_and_agree():
    rbd = one_block(conjugate())
    report = rbd.analysis_routes()
    for name in (
        "sf_uncertainty",
        "mean_uncertainty",
        "time_to_reliability_uncertainty",
        "bx_life_uncertainty",
    ):
        assert report[name].route == routes.SIMULATED
        assert "posterior" in report[name].reason
    assert report["uncertainty_importance"].route == routes.NUMERICAL
    assert "method='sobol' takes it" in report["uncertainty_importance"].reason
    # Each simulated route runs, on the default draws.
    rbd.sf_uncertainty(100.0, n_draws=5, seed=0)
    rbd.mean_uncertainty(n_draws=5, seed=0)
    rbd.time_to_reliability_uncertainty(0.5, n_draws=5, seed=0)
    rbd.bx_life_uncertainty(10, n_draws=5, seed=0)
    repairable = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": conjugate(),
                "repairability": E([0.01]),
                "repair_cost": 5.0,
            }
        },
    )
    report = repairable.analysis_routes()
    for name in (
        "mean_availability_uncertainty",
        "point_availability_uncertainty",
        "mission_availability_uncertainty",
        "expected_cost_rate_uncertainty",
    ):
        assert report[name].route != routes.REFUSED
        assert "posterior" in report[name].reason
    assert "method='sobol' takes it" in report["uncertainty_importance"].reason
    repairable.mean_availability_uncertainty(n_draws=3, seed=0)
    repairable.point_availability_uncertainty([10.0], n_draws=3, seed=0)
    repairable.mission_availability_uncertainty(10.0, n_draws=3, seed=0)
    assert (
        repairable.expected_cost_rate_uncertainty(n_draws=3, seed=0).band
        == "credible"
    )


def test_the_result_through_json():
    result = one_block(conjugate()).sf_uncertainty(
        [100.0, 300.0], n_draws=50, seed=0
    )
    back = json.loads(json.dumps(result.to_dict()))
    assert back["band"] == "credible"
    assert np.allclose(back["samples"], result.samples)
    assert "band='credible'" in repr(result)


def _digest(values) -> str:
    data = np.ascontiguousarray(np.asarray(values, dtype=float)).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


UNCHANGED_PATH = Path(__file__).with_name("bayesian_bands_unchanged.json")

# The record, every draw, was taken on macOS (arm64) on dev before Bayesian
# blocks were drawn (#394); these are the digests of its draws' bytes,
# which the stored record must still match to the last bit. (The bands,
# a result's ``band``, are #394's names for where the draws came from.)
UNCHANGED_DIGESTS = {
    "sf": "e4c2edb103399415",
    "sf_both_sobol": "4dc9c35c36d153ec",
    "mean": "01ffc81a09647456",
    "bx_life_both": "710ead24ae5b9452",
    "mean_availability": "ca68d502f8edfc9b",
    "point_availability": "9714884fcc483b3e",
}

# What another platform changes. A fitted node's draws pass through its
# fit's parameter covariance, which surpyval works out numerically (an
# optimiser and a Hessian): its last bits depend on the platform's libm and
# BLAS, and so differ between macOS and Linux and between Linux CPUs (CI's
# jobs gave three different digests of the same draws on one commit). The
# seeded normals are the same everywhere; the drawn values are not. Measured
# against the record (Linux x86-64 and arm64, numpy 2.5): every draw and
# interval end within 1.1e-15 of its value, a first-order index (a ratio of
# variances, so a difference of close numbers) within 1.9e-11. The checks
# allow about a hundred times that, far below what any change to the draws
# (a stream, a scale, a parameter) moves.
DRAW_RTOL = 1e-13
INDEX_RTOL = 1e-9


def _unchanged_results() -> dict:
    """The guide's fitted examples (reliability.md and repairable.md):
    each result's draws (``samples``), its 90% interval's lower end
    (``lower``) or a first-order index (``index``), and its band."""
    out = {}

    def keep(name, result):
        out[name] = {
            "samples": np.asarray(result.samples).tolist(),
            "lower": np.asarray(result.interval(0.9)[0]).tolist(),
            "band": result.band,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = surv.Weibull.from_params([100, 2]).qf(
            np.linspace(0.025, 0.975, 20)
        )
        pump_fit = surv.Weibull.fit(data)
        fitted = NonRepairableRBD(
            [
                ("s", "pump1"),
                ("s", "pump2"),
                ("pump1", "valve"),
                ("pump2", "valve"),
                ("valve", "t"),
            ],
            {
                "pump1": pump_fit,
                "pump2": pump_fit,
                "valve": surv.Weibull.from_params([200, 1.5]),
            },
        )
        both = {
            ("pump1", "pump2"): "fit",
            "valve": {"alpha": st.uniform(150, 100)},
        }
        keep("sf", fitted.sf_uncertainty([25.0, 50.0], n_draws=500, seed=0))
        keep(
            "sf_both_sobol",
            fitted.sf_uncertainty(
                50, both, n_draws=256, seed=3, sampling="sobol"
            ),
        )
        keep("mean", fitted.mean_uncertainty(n_draws=200, seed=0))
        keep(
            "bx_life_both",
            fitted.bx_life_uncertainty(10, both, n_draws=200, seed=0),
        )
        parts = fitted.uncertainty_importance(50, both)
        out["importance_pumps"] = {
            "index": parts.first_order[("pump1", "pump2")]
        }
        parts = fitted.uncertainty_importance(
            50, both, method="sobol", n_draws=200, seed=0
        )
        out["importance_valve_sobol"] = {"index": parts.first_order["valve"]}

        pump_repair = surv.LogNormal.fit(np.exp(np.linspace(-0.8, 0.8, 20)))
        valve_life = surv.Weibull.fit(
            surv.Weibull.from_params([55.0, 1.5]).qf(
                np.linspace(0.05, 0.95, 15)
            )
        )
        plant = RepairableRBD(
            [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")],
            {
                "A": {"reliability": E([0.1]), "repairability": pump_repair},
                "B": {"reliability": E([0.1]), "repairability": pump_repair},
                "C": {"reliability": valve_life, "repairability": E([0.5])},
            },
        )
        keep(
            "mean_availability",
            plant.mean_availability_uncertainty(n_draws=100, seed=0),
        )
        keep(
            "point_availability",
            plant.point_availability_uncertainty(
                [5.0, 20.0], n_draws=30, seed=1
            ),
        )
        parts = plant.uncertainty_importance()
        out["importance_plant"] = {"index": parts.first_order[("A", "B")]}
    return out


def record_unchanged_results():
    """Re-record ``bayesian_bands_unchanged.json``: only after a deliberate
    change of these draws, with ``UNCHANGED_DIGESTS`` updated to match."""
    with open(UNCHANGED_PATH, "w") as file:
        json.dump(_unchanged_results(), file, indent=1)
        file.write("\n")


def test_the_unchanged_record_is_the_one_taken_before_bayesian_blocks():
    with open(UNCHANGED_PATH) as file:
        recorded = json.load(file)
    for name, digest in UNCHANGED_DIGESTS.items():
        assert _digest(recorded[name]["samples"]) == digest, name


def test_diagrams_with_no_bayesian_block_are_unchanged():
    """The guide's fitted examples give the draws recorded before Bayesian
    blocks were drawn: every draw, to the platform's last bits (see
    ``DRAW_RTOL``)."""
    with open(UNCHANGED_PATH) as file:
        recorded = json.load(file)
    results = _unchanged_results()
    assert results.keys() == recorded.keys()
    for name, want in recorded.items():
        got = results[name]
        if "index" in want:
            np.testing.assert_allclose(
                got["index"], want["index"], rtol=INDEX_RTOL, err_msg=name
            )
            continue
        assert got["band"] == want["band"], name
        assert np.shape(got["samples"]) == np.shape(want["samples"]), name
        np.testing.assert_allclose(
            got["samples"], want["samples"], rtol=DRAW_RTOL, err_msg=name
        )
        np.testing.assert_allclose(
            got["lower"], want["lower"], rtol=DRAW_RTOL, err_msg=name
        )
