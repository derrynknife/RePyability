"""``"fit"`` uncertainty for every fitted kind of block, not only a plain
distribution: a regression node, a degradation process, a mixture, an
offset fit and the fitted repairable-unit processes (Crow-AMSAA, a
generalised renewal process). Each is drawn from its fit's covariance
and rebuilt by surpyval's ``with_params`` (an offset fit, from surpyval's
refits to resampled data), so its band is checked against surpyval's
own where surpyval gives one, and its draws against what they must be.
"""

import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, RegressionNode, RepairableRBD
from repyability.rbd import _repairable_uncertainty as repairable
from repyability.rbd.uncertainty import draw_models, is_fit

SINGLE = [("s", "c"), ("c", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]


def one_node(model):
    return NonRepairableRBD(SINGLE, {"c": model})


def regression():
    """A Weibull proportional-hazards fit to 80 lives at two loads."""
    rng = np.random.default_rng(1)
    Z = rng.binomial(1, 0.5, 80).reshape(-1, 1).astype(float)
    x = surv.Weibull.random(80, 10, 2) * np.exp(-0.5 * Z[:, 0])
    return surv.WeibullPH.fit(x=x, Z=Z)


def wiener():
    """A Wiener degradation process fitted to five units' paths."""
    rng = np.random.default_rng(1)
    t = np.tile(np.arange(0, 110, 10.0), 5)
    i = np.repeat(np.arange(5), 11)
    steps = rng.normal(5, np.sqrt(10), size=(5, 10))
    y = np.hstack([np.r_[0.0, np.cumsum(s)] for s in steps])
    return surv.WienerProcess.fit(t, y, i, threshold=100)


def generalised_renewal():
    """A Kijima I process fitted to ten units' simulated histories (its
    restoration factor about 0.5, well inside its range)."""
    np.random.seed(3)
    truth = surv.GeneralizedRenewal.fit_from_parameters(
        [10.0, 2.0], 0.5, kijima="i"
    )
    data = truth.time_terminated_simulation_data(T=100, items=10)
    return surv.GeneralizedRenewal.fit(data.x, i=data.i, c=data.c, kijima="i")


def crow_amsaa():
    gaps = np.random.default_rng(1).exponential(10, 30)
    return surv.CrowAMSAA.fit(np.cumsum(gaps))


# -- a band against surpyval's own -------------------------------------------


def test_a_regression_nodes_band_is_surpyvals_wald_band():
    # Drawn from the regression's covariance, the node's 90% band on its
    # reliability is surpyval's own delta-method (Wald) band at the same
    # covariates, to the skew of the normal approximation over the draws.
    model = regression()
    node = RegressionNode(model, covariates=[0.0])
    result = one_node(node).sf_uncertainty(
        5.0, {"c": "fit"}, n_draws=10_000, seed=0
    )
    lower, upper = result.interval(0.9)
    bounds = np.ravel(model.cb(5.0, [[0.0]], alpha_ci=0.1))
    assert result.nominal == pytest.approx(
        float(np.ravel(model.sf(5.0, [[0.0]]))[0])
    )
    assert result.band == "confidence"
    assert lower == pytest.approx(bounds[0], abs=0.01)
    assert upper == pytest.approx(bounds[1], abs=0.01)


def test_a_degradation_process_band_is_surpyvals_wald_band():
    model = wiener()
    result = one_node(model).sf_uncertainty(
        200.0, {"c": "fit"}, n_draws=10_000, seed=0
    )
    lower, upper = result.interval(0.9)
    bounds = np.ravel(model.cb(200.0, alpha_ci=0.1))
    assert result.nominal == pytest.approx(float(np.ravel(model.sf(200.0))[0]))
    # surpyval forms its band on the logit of F: about as wide.
    assert lower == pytest.approx(bounds[0], abs=0.05)
    assert upper == pytest.approx(bounds[1], abs=0.05)


def test_an_offset_fit_is_drawn_from_surpyvals_refits():
    # The draws are surpyval's refits to resampled data, each with its own
    # offset (which the covariance leaves out), so before the smallest
    # refit's offset nothing fails, and after the fitted one some draws
    # have failed less: the band is the bootstrap band surpyval gives.
    data = np.linspace(200, 1800, 50) + 100
    model = surv.Weibull.fit(data, offset=True)
    drawn = draw_models(model, "fit", 400, np.random.default_rng(0), "c")
    offsets = np.array([m.gamma for m in drawn])
    assert offsets.std() > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = one_node(model).sf_uncertainty(
            800.0, {"c": "fit"}, n_draws=4_000, seed=1
        )
    lower, upper = result.interval(0.9)
    boot = np.ravel(
        model.cb(800.0, alpha_ci=0.1, method="bootstrap", random_state=0)
    )
    assert lower == pytest.approx(boot[0], abs=0.02)
    assert upper == pytest.approx(boot[1], abs=0.02)


def test_an_offset_fit_surpyval_cannot_resample_holds_its_offset():
    # Interval-censored data cannot be resampled: the offset is held where
    # it was fitted, and the draws say so.
    lives = np.linspace(200, 1800, 30) + 100
    xl, xr = lives - 20, lives + 20
    model = surv.Weibull.fit(
        xl=np.r_[lives[:20], xl[20:]],
        xr=np.r_[lives[:20], xr[20:]],
        offset=True,
    )
    with pytest.warns(UserWarning, match="offset is held where it was"):
        drawn = draw_models(model, "fit", 50, np.random.default_rng(0), "c")
    assert {m.gamma for m in drawn} == {model.gamma}


def test_a_mixtures_draws_are_its_fits_with_weights_summing_to_one():
    rng = np.random.default_rng(2)
    x = np.r_[rng.weibull(3, 60) * 10, rng.weibull(3, 60) * 50]
    model = surv.MixtureModel.fit(x, dist=surv.Weibull, m=2)
    assert is_fit(model)
    drawn = draw_models(model, "fit", 500, np.random.default_rng(0), "c")
    weights = np.array([m.w for m in drawn])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.all(weights >= 0) and weights[:, 0].std() > 0
    # The draws' parameters centre on the fit's.
    params = np.array([np.ravel(m.params) for m in drawn])
    np.testing.assert_allclose(
        params.mean(axis=0), np.ravel(model.params), rtol=0.05
    )


# -- nodes of one fit ---------------------------------------------------------


def test_regression_nodes_of_one_fit_share_its_draws():
    # Two pumps at different loads, from one regression fit: drawn once,
    # each at its own covariates. Drawn apart they would lose that the
    # fit's error moves both together.
    model = regression()
    a = RegressionNode(model, covariates=[0.0])
    b = RegressionNode(model, covariates=[1.0])
    rbd = NonRepairableRBD(SERIES, {"a": a, "b": b})
    assert rbd._fitted_uncertainty() == {("a", "b"): "fit"}
    drawn, _ = rbd._uncertain_draws(None, 20, 0)
    for one, other in zip(drawn["a"], drawn["b"]):
        assert one.model is other.model
        assert list(one.covariates) == [0.0]
        assert list(other.covariates) == [1.0]
    together = rbd.sf_uncertainty(5.0, n_draws=20, seed=0)
    expected = [
        float(one.sf(5.0) * other.sf(5.0))
        for one, other in zip(drawn["a"], drawn["b"])
    ]
    np.testing.assert_allclose(together.samples, expected, rtol=1e-12)
    # Given alone, the other node keeps the fit as it is, and is named.
    with pytest.warns(UserWarning, match="hold the same model"):
        rbd.sf_uncertainty(5.0, {"a": "fit"}, n_draws=10, seed=0)


# -- repairable-unit processes ------------------------------------------------


def test_a_fitted_crow_amsaa_life_is_drawn_from_its_fit():
    # Given as a component's life, a Crow-AMSAA fit is minimal repair of a
    # Weibull: each draw is the process at drawn parameters, made that
    # Weibull and minimal repair again. Its scale's standard error is
    # two-thirds of it, so some draws on its own scale are negative:
    # surpyval refuses them, and they are drawn again and counted.
    process = crow_amsaa()
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": process,
                "repairability": surv.Exponential.from_params([1.0]),
            }
        },
    )
    assert repairable.fitted(rbd) == {"a": {"reliability": "fit"}}
    inputs, groups = repairable.sources(rbd, None)
    with pytest.warns(UserWarning, match="3 of the parameter vectors"):
        drawn, drawn_groups = repairable.draws(rbd, inputs, groups, 30, 0)
    alphas = []
    for i in range(30):
        built = repairable.build(rbd, inputs, drawn, drawn_groups, i)
        spec = built._init_args["components"]["a"]
        process_i = drawn[0]["reliability"][i]
        np.testing.assert_allclose(
            np.ravel(spec["reliability"].params), np.ravel(process_i.params)
        )
        assert spec["repair"] == {"model": "kijima1", "q": 1.0}
        alphas.append(float(np.ravel(process_i.params)[0]))
    assert np.std(alphas) > 0
    with pytest.warns(UserWarning, match="drawn again"):
        result = rbd.point_availability_uncertainty(50.0, n_draws=40, seed=0)
        again = rbd.point_availability_uncertainty(50.0, n_draws=40, seed=0)
    assert result.nominal == pytest.approx(rbd.point_availability(50.0))
    assert result.std > 0
    np.testing.assert_array_equal(result.samples, again.samples)


def test_a_generalised_renewal_life_draws_its_restoration_factor():
    process = generalised_renewal()
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": process,
                "repairability": surv.Exponential.from_params([1.0]),
            }
        },
    )
    inputs, groups = repairable.sources(rbd, None)
    drawn, drawn_groups = repairable.draws(rbd, inputs, groups, 200, 0)
    qs = []
    for i in range(200):
        built = repairable.build(rbd, inputs, drawn, drawn_groups, i)
        repair = built._init_args["components"]["a"]["repair"]
        assert repair["model"] == "kijima1"
        assert repair["q"] == pytest.approx(drawn[0]["reliability"][i].q)
        qs.append(repair["q"])
    # Drawn from the fit's covariance: about its standard error.
    sd = float(np.sqrt(process.covariance()[0, 0]))
    assert np.std(qs) == pytest.approx(sd, rel=0.2)
    assert np.mean(qs) == pytest.approx(process.restoration, abs=0.5 * sd)


def test_parameter_distributions_of_a_process_life_still_vary_its_weibull():
    # Distributions over parameters keep working on the life the process
    # is (a Weibull), as before.
    import scipy.stats as st

    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": crow_amsaa(),
                "repairability": surv.Exponential.from_params([1.0]),
            }
        },
    )
    result = rbd.point_availability_uncertainty(
        50.0,
        {"a": {"reliability": {"alpha": st.uniform(8.0, 4.0)}}},
        n_draws=10,
        seed=0,
    )
    assert result.std > 0


# -- refusals, said plainly ---------------------------------------------------


def test_a_model_without_a_covariance_is_refused_plainly():
    built = regression().with_params(regression().params)
    node = RegressionNode(built, covariates=[0.0])
    assert not is_fit(node)
    with pytest.raises(ValueError, match="covariance.*list of models"):
        one_node(node).sf_uncertainty(5.0, {"c": "fit"}, n_draws=5)


def test_a_joint_models_prediction_is_refused_plainly(monkeypatch):
    # surpyval's joint models have a covariance but no with_params.
    from repyability.rbd import joint_node
    from repyability.rbd.uncertainty import _redrawable

    fake = object()
    monkeypatch.setattr(joint_node, "is_joint", lambda m: m is fake)
    with pytest.raises(ValueError, match="joint model.*with_params"):
        _redrawable(fake, "Node 'c'")


def test_draws_surpyval_refuses_are_drawn_again_and_counted():
    model = regression()
    node = RegressionNode(model, covariates=[0.0])
    # A covariance wide enough that some drawn scales are negative.
    alpha = float(model.params[0])
    wide = np.diag([(0.5 * alpha) ** 2, 1e-4, 1e-4])
    model.covariance = lambda: wide
    with pytest.warns(UserWarning, match=r"\d+ of the parameter vectors"):
        drawn = draw_models(node, "fit", 2_000, np.random.default_rng(0), "c")
    assert all(float(m.model.params[0]) > 0 for m in drawn)
    # Too wide: more refused than asked for, refused.
    # Both the scale and the shape must be positive: about three in four
    # draws are not.
    beta = float(model.params[1])
    model.covariance = lambda: np.diag(
        [(50 * alpha) ** 2, (50 * beta) ** 2, 1e-4]
    )
    with pytest.raises(ValueError, match="normal approximation of this fit"):
        draw_models(node, "fit", 200, np.random.default_rng(0), "c")


def test_the_delta_method_varies_a_regressions_parameters():
    # The delta method's variance of the reliability, from the regression's
    # covariance and the derivatives in its parameters, is the variance of
    # the draws, to the curvature over them; with two nodes of one fit, the
    # fit is one input.
    model = regression()
    node = RegressionNode(model, covariates=[0.0])
    rbd = one_node(node)
    parts = rbd.uncertainty_importance(5.0, {"c": "fit"})
    drawn = rbd.sf_uncertainty(5.0, {"c": "fit"}, n_draws=20_000, seed=0)
    assert parts.variance == pytest.approx(drawn.std**2, rel=0.1)
    pair = NonRepairableRBD(
        SERIES,
        {"a": node, "b": RegressionNode(model, covariates=[1.0])},
    )
    shares = pair.uncertainty_importance(5.0)
    assert shares.first_order == {("a", "b"): pytest.approx(1.0)}


def test_a_repairable_diagrams_delta_method_sends_a_process_to_sobol():
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": crow_amsaa(),
                "repairability": surv.Exponential.from_params([1.0]),
            }
        },
    )
    assert (
        "method='sobol' takes them"
        in (rbd.analysis_routes()["uncertainty_importance"].reason)
        or rbd.analysis_routes()["uncertainty_importance"].route == "refused"
    )
    with pytest.raises(ValueError, match="method='sobol'"):
        repairable.delta(
            rbd,
            "point_availability",
            50.0,
            None,
            repairable.sources(rbd, None)[0],
            {},
            1e-4,
        )


# -- seeded and quasi-random draws --------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: RegressionNode(regression(), covariates=[1.0]),
        wiener,
        lambda: surv.Weibull.fit(
            np.linspace(200, 1800, 50) + 100, offset=True
        ),
    ],
    ids=["regression", "degradation", "offset"],
)
def test_seeded_draws_repeat_and_sobol_points_draw_them(make):
    rbd = one_node(make())
    at = (
        400.0
        if not isinstance(rbd.reliabilities["c"], RegressionNode)
        else 5.0
    )
    one = rbd.sf_uncertainty(at, n_draws=64, seed=3)
    two = rbd.sf_uncertainty(at, n_draws=64, seed=3)
    np.testing.assert_array_equal(one.samples, two.samples)
    sobol = rbd.sf_uncertainty(at, n_draws=64, seed=3, sampling="sobol")
    again = rbd.sf_uncertainty(at, n_draws=64, seed=3, sampling="sobol")
    np.testing.assert_array_equal(sobol.samples, again.samples)
    assert sobol.std > 0
