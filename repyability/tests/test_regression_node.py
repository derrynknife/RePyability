"""Tests for :class:`RegressionNode` (issue #37): an RBD node backed by a
fitted surpyval regression model at a fixed set of covariates.

The node is deliberately family-agnostic -- it works for accelerated-failure-
time, proportional-hazards (Cox) and proportional-odds models alike, because
its reliability is just ``model.sf(x, Z)`` at the stored covariates ``Z`` and
everything else (system reliability, age conditioning, MTTF, serialisation)
follows from that single univariate survival curve.
"""

import json

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, NonRepairableRBD, RegressionNode

# The time-varying-covariate (schedule) mode needs a surpyval build exposing
# sf_tvc / StepSchedule; skip those tests on older builds so CI stays green.
try:
    from surpyval.univariate.regression import StepSchedule

    _HAS_TVC = True
except ImportError:  # pragma: no cover
    StepSchedule = None
    _HAS_TVC = False

needs_tvc = pytest.mark.skipif(
    not _HAS_TVC, reason="surpyval build lacks sf_tvc / StepSchedule"
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 1))
    x = rng.weibull(1.8, size=400) * 80.0 * np.exp(-0.4 * Z[:, 0]) + 1e-3
    return x, Z


@pytest.fixture(scope="module")
def models(data):
    x, Z = data
    return {
        "aft": surv.WeibullAFT.fit(x, Z=Z),
        "ph": surv.CoxPH.fit(x, Z=Z),
        "po": surv.ProportionalOddsFitter(surv.Weibull).fit(x, Z=Z),
    }


def _Z(cov, n):
    return np.repeat(np.atleast_2d(cov), n, axis=0)


# -- reliability matches the model at the stored covariates ----------------


@pytest.mark.parametrize("family", ["aft", "ph", "po"])
def test_sf_matches_model_at_covariates(models, family):
    model = models[family]
    node = RegressionNode(model, covariates=[0.3])
    xt = np.array([5.0, 20.0, 60.0])
    assert np.allclose(node.sf(xt), model.sf(xt, _Z([0.3], len(xt))))
    assert np.allclose(node.ff(xt), 1.0 - node.sf(xt))


@pytest.mark.parametrize("family", ["aft", "ph", "po"])
def test_covariates_change_reliability(models, family):
    model = models[family]
    xt = np.array([30.0])
    # In this data a higher covariate shortens life, so it lowers reliability.
    hi = RegressionNode(model, covariates=[0.6]).sf(xt)[0]
    lo = RegressionNode(model, covariates=[-0.6]).sf(xt)[0]
    assert hi < lo


# -- mean / random (simulation paths) -------------------------------------


def test_mean_and_random(models):
    node = RegressionNode(models["aft"], covariates=[0.2])
    assert node.mean() > 0.0
    np.random.seed(0)
    a = node.random(20000)
    np.random.seed(0)
    b = node.random(20000)
    assert np.allclose(a, b)  # reproducible under a fixed global seed
    assert (a > 0).all()
    # sample mean tracks the integrated mean
    assert a.mean() == pytest.approx(node.mean(), rel=0.05)


def test_semiparametric_mean_random_raise_clearly(models):
    # A semiparametric Cox baseline has no proper tail, so MTTF is undefined;
    # mean/random must raise rather than return a wrong number; sf works.
    node = RegressionNode(models["ph"], covariates=[0.2])
    assert 0.0 < node.sf(np.array([20.0]))[0] < 1.0
    with pytest.raises(ValueError, match="proper|Cox|undefined"):
        node.mean()
    with pytest.raises(ValueError, match="proper|Cox|undefined"):
        node.random(10)


# -- age conditioning is the ordinary, uniform mechanism ------------------


@pytest.mark.parametrize("family", ["aft", "ph", "po"])
def test_age_conditioning_is_plain_conditional_survival(models, family):
    # A single regression component in series: system reliability given the
    # component's age is exactly sf(age + x) / sf(age) -- no special handling.
    node = RegressionNode(models[family], covariates=[0.4])
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": node})
    a, x = 25.0, 15.0
    got = float(rbd.sf_given_state(x, {"a": NodeState(age=a)}))
    expected = node.sf(np.array([a + x]))[0] / node.sf(np.array([a]))[0]
    assert got == pytest.approx(expected)


# -- inside a larger RBD --------------------------------------------------


@pytest.fixture(scope="module")
def rbd(models):
    return NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            # A parametric regression model has a proper lifetime, so the
            # simulation paths (node_mttf / mean) are well-defined.
            "a": RegressionNode(models["aft"], covariates=[0.5]),
            "b": surv.Weibull.from_params([120.0, 2.5]),
        },
    )


def test_rbd_is_time_varying_and_evaluates(rbd):
    assert rbd.is_time_varying
    assert 0.0 < float(rbd.sf(30.0)) < 1.0


def test_rbd_condition_and_rul(rbd):
    st = {"a": NodeState(age=40.0)}
    worn = float(rbd.sf_given_state(30.0, st))
    fresh = float(rbd.sf_given_state(30.0, {}))
    assert worn < fresh  # a worn component lowers go-forward reliability
    assert rbd.remaining_life(0.8, st) > 0.0


def test_dead_regression_node_contributes_zero(rbd):
    r = float(rbd.sf_given_state(10.0, {"a": NodeState(alive=False)}))
    assert r == pytest.approx(0.0)


def test_importances_and_mttf_with_regression_node(rbd):
    imp = rbd.importances_given_state(30.0, {"a": NodeState(age=40.0)})
    assert set(imp) == {"birnbaum", "criticality"}
    assert "a" in imp["birnbaum"] and "b" in imp["birnbaum"]
    mttf = rbd.node_mttf(mc_samples=1500, seed=1)
    assert mttf["a"] > 0.0 and mttf["b"] > 0.0
    assert rbd.mean(2000, seed=1) > 0.0


# -- serialisation --------------------------------------------------------


@pytest.mark.parametrize("family", ["aft", "ph", "po"])
def test_node_dict_roundtrip(models, family):
    node = RegressionNode(models[family], covariates=[0.3])
    node2 = RegressionNode.from_dict(json.loads(json.dumps(node.to_dict())))
    xt = np.array([5.0, 25.0, 70.0])
    assert np.allclose(node.sf(xt), node2.sf(xt))
    assert np.allclose(node2.covariates, node.covariates)


def test_rbd_with_regression_node_json_roundtrip(rbd):
    restored = NonRepairableRBD.from_json(rbd.to_json())
    st = {"a": NodeState(age=40.0)}
    assert np.isclose(
        float(rbd.sf_given_state(30.0, st)),
        float(restored.sf_given_state(30.0, st)),
    )


# -- construction validation ----------------------------------------------


def test_wrong_covariate_width_rejected(models):
    # The model was fitted with one covariate; two is a mismatch.
    with pytest.raises(ValueError, match="regression model|covariates"):
        RegressionNode(models["aft"], covariates=[0.1, 0.2])


def test_non_regression_model_rejected():
    # A plain univariate distribution has no sf(x, Z) covariate interface.
    with pytest.raises(ValueError, match="regression model|covariates"):
        RegressionNode(
            surv.Weibull.from_params([100.0, 2.0]), covariates=[0.1]
        )


def test_requires_exactly_one_of_covariates_or_schedule(models):
    with pytest.raises(ValueError, match="exactly one"):
        RegressionNode(models["aft"])  # neither


# -- time-varying covariate (schedule) mode, issue #37 ---------------------


@needs_tvc
@pytest.mark.parametrize("family", ["aft", "ph"])
def test_schedule_sf_matches_sf_tvc(models, family):
    sched = StepSchedule.from_changepoints([0, 50], [[0.0], [0.8]])
    node = RegressionNode(models[family], schedule=sched)
    xt = np.array([30.0, 80.0, 150.0])
    assert np.allclose(node.sf(xt), np.ravel(models[family].sf_tvc(xt, sched)))


@needs_tvc
def test_schedule_condition_based_is_given(models):
    # The condition-based `age` path must equal surpyval's sf_tvc(given=age):
    # forward reliability from the component's current life under the schedule.
    sched = StepSchedule.from_changepoints([0, 50], [[0.0], [0.8]])
    node = RegressionNode(models["aft"], schedule=sched)
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": node})
    age = 60.0
    for x in (20.0, 40.0):
        got = float(rbd.sf_given_state(x, {"c": NodeState(age=age)}))
        want = float(
            np.ravel(
                models["aft"].sf_tvc(np.array([age + x]), sched, given=age)
            )[0]
        )
        assert got == pytest.approx(want)


@needs_tvc
def test_schedule_more_load_lowers_reliability(models):
    # A schedule ramping to a higher load is less reliable than a benign one.
    low = RegressionNode(
        models["aft"],
        schedule=StepSchedule.from_changepoints([0, 50], [[0.0], [0.0]]),
    )
    high = RegressionNode(
        models["aft"],
        schedule=StepSchedule.from_changepoints([0, 50], [[0.0], [1.0]]),
    )
    xt = np.array([120.0])
    assert high.sf(xt)[0] < low.sf(xt)[0]


@needs_tvc
def test_po_schedule_rejected(models):
    # Proportional odds does not compose along a time-varying covariate.
    with pytest.raises(ValueError, match="proportional-odds|sf_tvc"):
        RegressionNode(
            models["po"],
            schedule=StepSchedule.from_changepoints([0, 50], [[0.0], [0.8]]),
        )


@needs_tvc
def test_schedule_serialisation_roundtrip(models):
    sched = StepSchedule.from_changepoints([0, 50, 120], [[0.0], [0.8], [0.3]])
    node = RegressionNode(models["aft"], schedule=sched)
    node2 = RegressionNode.from_dict(json.loads(json.dumps(node.to_dict())))
    xt = np.array([30.0, 90.0, 160.0])
    assert np.allclose(node.sf(xt), node2.sf(xt))


@needs_tvc
def test_rbd_with_schedule_node_json_roundtrip(models):
    sched = StepSchedule.from_changepoints([0, 50], [[0.0], [0.8]])
    rbd = NonRepairableRBD(
        [("s", "c"), ("c", "t")],
        {"c": RegressionNode(models["aft"], schedule=sched)},
    )
    restored = NonRepairableRBD.from_json(rbd.to_json())
    assert np.isclose(float(rbd.sf(80.0)), float(restored.sf(80.0)))


@needs_tvc
def test_schedule_mean_and_random(models):
    node = RegressionNode(
        models["aft"],
        schedule=StepSchedule.from_changepoints([0, 50], [[0.0], [0.5]]),
    )
    assert node.mean() > 0.0
    np.random.seed(0)
    a = node.random(2000)
    np.random.seed(0)
    b = node.random(2000)
    assert np.allclose(a, b) and (a > 0).all()
