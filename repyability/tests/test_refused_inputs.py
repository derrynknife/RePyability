"""Inputs refused where they are given (#233): numbers given as text or as
True/False, surpyval's distributions given in place of models of them, a
replacement given as text, simulation options given to an exact mean,
percentiles on the wrong scale, and one component's spares on two shelves.
"""

import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    DegradingNode,
    FaultTree,
    LoadSharingModel,
    NonRepairable,
    NonRepairableRBD,
    RepairableRBD,
    StandbyModel,
)
from repyability.rbd.results import UncertaintyResult

W, E = surv.Weibull.from_params, surv.Exponential.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def unit(**extra):
    return {"reliability": W([100, 2]), "repairability": E([0.5]), **extra}


def pair(**extra):
    return RepairableRBD(PAIR, {"a": unit(**extra), "b": unit()})


# -- numbers given as text or as True/False ----------------------------------


@pytest.mark.parametrize(
    "key",
    ["repair_cost", "replace_cost", "acquisition_cost", "downtime_cost"],
)
@pytest.mark.parametrize("value", ["800", True], ids=repr)
def test_a_cost_is_a_number(key, value):
    with pytest.raises(ValueError, match=f"{key} must be a number"):
        pair(**{key: value})


def test_a_cost_given_as_a_number_still_counts():
    assert pair(repair_cost=np.float32(800.0)).costs["a"]["repair_cost"] == 800


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            {"preventive": {"interval": 50.0, "cost": "800"}},
            "preventive_cost must be a number",
        ),
        (
            {"preventive": {"interval": "50"}},
            "preventive interval must be a positive number",
        ),
        (
            {"inspection": {"interval": "8760"}},
            "inspection interval must be a positive, finite number, "
            "got '8760'",
        ),
        (
            {"inspection": {"interval": True}},
            "inspection interval must be a positive, finite number, got True",
        ),
        (
            {"inspection": {"interval": 100.0, "cost": "5"}},
            "inspection_cost must be a number",
        ),
    ],
    ids=[
        "a preventive cost",
        "a preventive interval",
        "an inspection interval",
        "an inspection interval of True",
        "an inspection cost",
    ],
)
def test_maintenance_is_given_in_numbers(spec, message):
    with pytest.raises(ValueError, match=message):
        pair(**spec)


@pytest.mark.parametrize(
    "times", ["8760", True, [1.0, "2"], np.array(["1"])], ids=repr
)
def test_a_time_is_a_number(times):
    rbd = NonRepairableRBD(PAIR, {"a": W([100, 2]), "b": W([90, 2])})
    with pytest.raises(TypeError, match="x must be numbers"):
        rbd.sf(times)
    repairable = pair()
    for method in (
        repairable.point_availability,
        repairable.mission_availability,
        repairable.point_unavailability,
    ):
        with pytest.raises(TypeError, match="must be numbers"):
            method(times)


def test_times_given_as_numbers_still_count():
    rbd = NonRepairableRBD(PAIR, {"a": W([100, 2]), "b": W([90, 2])})
    assert rbd.sf(np.int64(50)) == rbd.sf(50.0)
    assert np.array_equal(rbd.sf([10, 50]), rbd.sf(np.array([10.0, 50.0])))
    assert rbd.sf(np.float32(50.0)) == pytest.approx(rbd.sf(50.0))


# -- surpyval's distribution, not a model of it -------------------------------


def test_a_distribution_class_is_no_model():
    fit = "fit it to data \\(surv.Weibull.fit\\(times\\)\\)"
    with pytest.raises(
        TypeError, match=f"node 'a' is surpyval's Weibull.*{fit}"
    ):
        NonRepairableRBD(PAIR, {"a": surv.Weibull, "b": W([90, 2])})
    with pytest.raises(
        TypeError, match=f"The reliability of component 'a' is.*{fit}"
    ):
        RepairableRBD(
            PAIR,
            {"a": unit(reliability=surv.Weibull), "b": unit()},
        )
    with pytest.raises(
        TypeError,
        match="The repairability of component 'a' is surpyval's Exponential",
    ):
        RepairableRBD(
            PAIR,
            {"a": unit(repairability=surv.Exponential), "b": unit()},
        )
    with pytest.raises(
        TypeError, match=f"Event 'a' is surpyval's Weibull.*{fit}"
    ):
        FaultTree({"T": ("and", ["a", "b"])}, {"a": surv.Weibull, "b": 0.01})
    with pytest.raises(TypeError, match=f"life is surpyval's Weibull.*{fit}"):
        NonRepairable(surv.Weibull)
    with pytest.raises(TypeError, match="time_to_replace is surpyval's"):
        NonRepairable(W([100, 2]), surv.Exponential)


def test_a_repairable_component_is_told_it_needs_its_repairs():
    with pytest.raises(TypeError, match="needs its repairs too"):
        RepairableRBD(PAIR, {"a": W([100, 2]), "b": unit()})
    with pytest.raises(
        TypeError, match="surv.Weibull.fit\\(times\\).*repairs"
    ):
        RepairableRBD(PAIR, {"a": surv.Weibull, "b": unit()})


# -- NonRepairable(life, "instant") -------------------------------------------


def test_a_nonrepairable_takes_instant_replacement_as_a_spec_does():
    instant = NonRepairable(W([100, 2]), "instant")
    assert instant.time_to_replace.mean() == 0.0
    as_none = NonRepairable(W([100, 2]))
    assert as_none.time_to_replace.mean() == 0.0
    given = RepairableRBD(PAIR, {"a": instant, "b": as_none})
    spec = RepairableRBD(
        PAIR,
        {
            "a": {"reliability": W([100, 2]), "repairability": "instant"},
            "b": {"reliability": W([100, 2]), "repairability": "instant"},
        },
    )
    assert given.mean_availability() == spec.mean_availability() == 1.0


@pytest.mark.parametrize(
    "replacement, error, message",
    [
        ("quick", ValueError, "'quick' is no model.*'instant'"),
        (5.0, TypeError, "a model of the time to replace.*got 5.0"),
    ],
)
def test_a_replacement_is_a_model(replacement, error, message):
    with pytest.raises(error, match=message):
        NonRepairable(W([100, 2]), replacement)


# -- the simulation options of an exact mean ----------------------------------


def standby():
    return StandbyModel([E([0.01]), E([0.01])])


def load_sharing():
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    pump = surv.ExponentialAFT.fit(x, Z=load)
    return LoadSharingModel([pump, pump], load=2.0, k=1)


def degrading():
    return DegradingNode([(100, W([1000, 2])), (50, E([1 / 500]))])


MODELS = [standby, load_sharing, degrading]


@pytest.mark.parametrize("model", MODELS)
def test_an_exact_mean_warns_of_options_it_ignores(model):
    node = model()
    exact = node.mean()
    with pytest.warns(FutureWarning, match="ignores mc_samples, seed.*0.14"):
        assert node.mean(mc_samples=10, seed=1) == exact
    with pytest.raises(TypeError, match="only with method='simulate'"):
        node.mean(method="exact", mc_samples=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert node.mean(method="exact") == exact


@pytest.mark.parametrize("model", MODELS)
def test_an_exact_mean_can_be_checked_by_simulation(model):
    node = model()
    simulated = node.mean(method="simulate", mc_samples=20_000, seed=1)
    assert simulated == node.mean(method="simulate", mc_samples=20_000, seed=1)
    assert simulated == pytest.approx(node.mean(), rel=0.03)
    assert simulated != node.mean()


def test_a_mean_method_is_one_of_two():
    with pytest.raises(ValueError, match="'exact' or 'simulate'"):
        standby().mean(method="sim")


# -- percentiles on the 0 to 100 scale ----------------------------------------


def test_a_percentile_below_one_warns_of_its_scale():
    lives = np.random.default_rng(2).weibull(2, 30) * 100
    rbd = NonRepairableRBD(
        PAIR, {"a": surv.Weibull.fit(lives), "b": W([90, 2])}
    )
    result = rbd.mean_uncertainty(n_draws=200, seed=1)
    assert isinstance(result, UncertaintyResult)
    with pytest.warns(UserWarning, match="0 to 100"):
        low = result.percentile(0.05)
    assert low == pytest.approx(np.percentile(result.samples, 0.05))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result.percentile(5)
        result.percentile([0, 50, 100])
        result.interval(0.9)


def test_a_cost_percentile_below_one_warns_of_its_scale():
    rbd = pair(repair_cost=10.0)
    cost = rbd.cost(1000.0, mc_samples=200, seed=1)
    with pytest.warns(UserWarning, match="0 to 100"):
        cost.percentile(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost.percentile(50)


# -- one component's spares on one shelf --------------------------------------


def test_a_component_is_stocked_on_one_shelf():
    rbd = pair()
    with pytest.raises(ValueError, match="Node 'a' is in nodes and in part"):
        rbd.spares_demand(1000.0, nodes=["a"], parts={"p": ["a", "b"]})
    with pytest.raises(ValueError, match="lists 'a' more than once"):
        rbd.spares_demand(1000.0, parts={"p": ["a", "a"]})
    with pytest.raises(ValueError, match="Node 'a' is in nodes and in part"):
        rbd.spares_stock(
            10.0, fill_rate=0.9, nodes=["a"], parts={"p": ["a", "b"]}
        )
    pooled = rbd.spares_demand(300.0, parts={"p": ["a", "b"]})["p"]
    assert pooled.members == ("a", "b")
