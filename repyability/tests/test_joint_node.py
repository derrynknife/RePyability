"""A unit of a surpyval joint model (readings and failures) as a block of
a non-repairable diagram: ``JointNode``.

The node is the unit's dynamic prediction moved to now: its reliability at
mission time ``t`` is ``pred.sf(age + t)`` (surpyval has conditioned it on
the unit reaching ``age``), its lifetimes ``pred.qf(u) - age``. The checks
are against those values worked out by hand, and the simulations against
them within their sampling error."""

import json

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    JointNode,
    NodeState,
    NonRepairable,
    NonRepairableRBD,
    RepairableRBD,
)
from repyability.rbd import routes
from repyability.rbd.serialisation import exactly, serialise_model
from repyability.tests.catalogue import PUMP, joint_model, joint_unit

SERIES = [("in", "pump"), ("pump", "valve"), ("valve", "out")]
PARALLEL = [("in", "pump"), ("in", "valve"), ("pump", "out"), ("valve", "out")]
VALVE = surv.Weibull.from_params([80.0, 1.5])
T = np.array([5.0, 20.0, 45.0, 90.0])


@pytest.fixture(scope="module")
def pred():
    """The pump's prediction: 40 hours old, at load 1, readings rising."""
    model = joint_model()
    return model.predict(PUMP["Z"], PUMP["markers"], given=PUMP["age"])


def test_the_node_is_the_prediction_from_now(pred):
    node = JointNode(pred)
    assert node.age == 40.0
    np.testing.assert_array_equal(node.sf(T), pred.sf(40.0 + T))
    np.testing.assert_array_equal(node.ff(T), pred.ff(40.0 + T))
    np.testing.assert_array_equal(node.Hf(T), pred.Hf(40.0 + T))
    np.testing.assert_array_equal(node.df(T), pred.df(40.0 + T))
    np.testing.assert_array_equal(node.hf(T), pred.hf(40.0 + T))
    # Now it is surely running: the prediction is conditioned on its age.
    assert node.sf(0.0) == 1.0
    assert np.ndim(node.sf(20.0)) == 0
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(node.qf(p), pred.qf(p) - 40.0)
    np.testing.assert_allclose(node.sf(node.qf(p)), 1.0 - p, rtol=1e-8)


def test_a_model_with_the_inputs_is_its_prediction(pred):
    node = joint_unit()
    np.testing.assert_array_equal(node.sf(T), JointNode(pred).sf(T))
    assert node.prediction.markers == pred.markers


def test_series_and_parallel_are_the_products(pred):
    pump = pred.sf(40.0 + T)
    valve = VALVE.sf(T)
    series = NonRepairableRBD(SERIES, {"pump": pred, "valve": VALVE})
    parallel = NonRepairableRBD(PARALLEL, {"pump": pred, "valve": VALVE})
    # A prediction given as a node is its unit's JointNode.
    assert isinstance(series.reliabilities["pump"], JointNode)
    np.testing.assert_allclose(series.sf(T), pump * valve, rtol=1e-12)
    np.testing.assert_allclose(
        parallel.ff(T), (1.0 - pump) * (1.0 - valve), rtol=1e-12
    )
    # Birnbaum: the system's reliability with the pump working less with
    # it failed, in series the valve's reliability.
    np.testing.assert_allclose(
        series.birnbaum_importance(20.0)["pump"], VALVE.sf(20.0), rtol=1e-12
    )


def test_its_draws_are_its_life_left(pred):
    node = JointNode(pred)
    np.random.seed(11)
    drawn = node.random(500)
    np.random.seed(11)
    u = np.random.uniform(size=500)
    np.testing.assert_array_equal(drawn, pred.qf(u) - 40.0)
    assert np.all(drawn >= 0.0)


def test_the_simulations_agree_with_the_exact_values(pred):
    rbd = NonRepairableRBD(SERIES, {"pump": pred, "valve": VALVE})
    n = 2000
    lives = rbd.random(n, seed=3)
    exact = rbd.sf(T)
    simulated = (lives[:, None] > T[None, :]).mean(axis=0)
    error = np.sqrt(exact * (1.0 - exact) / n)
    assert np.all(np.abs(simulated - exact) <= 4.0 * error + 1e-12)
    # The MTTF from now: exact against simulated.
    mean = rbd.mean()
    assert np.isfinite(mean)
    assert abs(lives.mean() - mean) <= 4.0 * lives.std() / np.sqrt(n)


def test_its_mean_is_the_mean_life_left(pred):
    # A straight marker path whose slope varies between units leaves the
    # unit a chance of never failing (a path that falls away), however
    # small: its mean life left is then infinite, as the model says.
    node = JointNode(pred)
    assert pred.sf(1e6) > 1e-14
    assert node.mean() == np.inf
    # In series with a valve, whose life ends, the system's is finite: the
    # area under the products.
    rbd = NonRepairableRBD(SERIES, {"pump": node, "valve": VALVE})
    from scipy.integrate import simpson

    t = np.linspace(0.0, 800.0, 8001)
    area = simpson(rbd.sf(t), x=t)
    assert rbd.mean() == pytest.approx(area, rel=1e-7)


def test_a_state_takes_its_age_from_the_prediction(pred):
    rbd = NonRepairableRBD(SERIES, {"pump": pred, "valve": VALVE})
    valve = VALVE.sf(30.0 + T) / VALVE.sf(30.0)
    expected = pred.sf(40.0 + T) * valve
    left_out = rbd.sf_given_state(T, {"valve": NodeState(age=30.0)})
    np.testing.assert_allclose(left_out, expected, rtol=1e-12)
    # Its own age again is the same, not conditioned twice.
    same = rbd.sf_given_state(
        T, {"pump": NodeState(age=40.0), "valve": NodeState(age=30.0)}
    )
    np.testing.assert_allclose(same, expected, rtol=1e-12)
    failed = rbd.sf_given_state(T, {"pump": NodeState(alive=False)})
    np.testing.assert_array_equal(failed, 0.0)
    for age in (0.0, 10.0, 60.0):
        with pytest.raises(ValueError, match="condition it again"):
            rbd.sf_given_state(T, {"pump": NodeState(age=age)})
    with pytest.raises(ValueError, match="condition it again"):
        rbd.importances_given_state(20.0, {"pump": NodeState(age=5.0)})


def test_a_repairable_component_takes_no_joint_model(pred):
    repairs = surv.Exponential.from_params([1.0])
    for life in (pred, JointNode(pred)):
        with pytest.raises(ValueError, match="takes no joint model"):
            NonRepairable(life, repairs)
        with pytest.raises(ValueError, match="takes no joint model"):
            RepairableRBD(
                SERIES,
                {
                    "pump": {"reliability": life, "repairability": repairs},
                    "valve": {"reliability": VALVE, "repairability": repairs},
                },
            )
        with pytest.raises(TypeError, match="takes no joint model"):
            RepairableRBD(
                SERIES,
                {
                    "pump": life,
                    "valve": {"reliability": VALVE, "repairability": repairs},
                },
            )


def test_a_diagram_with_a_joint_node_round_trips(pred):
    for given in (pred, JointNode(pred)):
        rbd = NonRepairableRBD(SERIES, {"pump": given, "valve": VALVE})
        text = rbd.to_json()
        back = NonRepairableRBD.from_json(text)
        node = back.reliabilities["pump"]
        assert isinstance(node, JointNode)
        assert node.age == 40.0
        assert node.prediction.markers == pred.markers
        np.testing.assert_allclose(back.sf(T), rbd.sf(T), rtol=1e-12)
        # The saved form is plain JSON, and loads as the class it is.
        with exactly():
            saved = serialise_model(rbd.reliabilities["pump"])
        assert saved["kind"] == "joint_node"
        assert json.loads(json.dumps(saved)) == saved
        assert saved["covariates"] == [1.0]


def test_its_inputs_are_checked(pred):
    model = joint_model()
    with pytest.raises(TypeError, match="leave out Z"):
        JointNode(pred, Z={"load": 1.0})
    with pytest.raises(ValueError, match="age now"):
        JointNode(model, Z={"load": 1.0}, markers={0: 2.0})
    with pytest.raises(ValueError, match="missing"):
        JointNode(model, Z={"load": np.nan})
    with pytest.raises(TypeError, match="joint model's prediction"):
        JointNode(VALVE)
    with pytest.raises(TypeError, match="JointNode"):
        NonRepairableRBD(SERIES, {"pump": model, "valve": VALVE})


def test_a_new_unit_is_the_population_life():
    # No readings and age 0: a new unit at its load, the model's own
    # survival over the population's marker paths.
    model = joint_model()
    node = JointNode(model, Z={"load": 1.0})
    assert node.age == 0.0
    np.testing.assert_allclose(
        node.sf(T), model.sf(T, Z={"load": 1.0}), rtol=1e-12
    )


def test_the_routes_say_it_is_numerical(pred):
    route, how = routes.model_route(JointNode(pred))
    assert route == routes.NUMERICAL
    assert "quadrature" in how
    report = NonRepairableRBD(
        SERIES, {"pump": pred, "valve": VALVE}
    ).analysis_routes()
    assert report["sf"].route == routes.NUMERICAL
    assert "pump" in report["sf"].reason
