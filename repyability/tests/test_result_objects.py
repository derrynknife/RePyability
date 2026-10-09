"""Results that serialise and print predictably (#235): ``to_dict()`` on
every result, ready for ``json.dumps``; short reprs; the capacity's mean a
property, as the other results' values are; and no warning from a risk
reduction worth that is infinite."""

import dataclasses
import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    ConditionalRun,
    ControlVariate,
    FaultTree,
    NonRepairableRBD,
    RepairableRBD,
    SparesDemand,
)
from repyability.rbd.results import _ResultMapping

W, E = surv.Weibull.from_params, surv.Exponential.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
# Tuple node names, which JSON cannot hold as keys.
TUPLES = [
    ("s", ("pump", 1)),
    ("s", ("pump", 2)),
    (("pump", 1), "t"),
    (("pump", 2), "t"),
]


def unit(**extra):
    return {
        "reliability": W([100, 2]),
        "repairability": E([0.5]),
        "repair_cost": 10.0,
        **extra,
    }


def as_json(result):
    """The result's ``to_dict()`` through JSON and back."""
    return json.loads(json.dumps(result.to_dict()))


def fields_of(result):
    return {field.name for field in dataclasses.fields(result)}


@pytest.fixture(scope="module")
def repairable():
    return RepairableRBD(PAIR, {"a": unit(), "b": unit()})


@pytest.fixture(scope="module")
def availability(repairable):
    return repairable.availability(1000.0, mc_samples=1000, seed=1)


def test_an_availability_result_goes_through_json(availability):
    out = as_json(availability)
    assert set(out) == fields_of(availability)
    assert out["n_simulations"] == 1000
    assert out["criticalities"]["iou"]["up"].keys() == {"a", "b"}
    assert len(out["timeline"]) == len(availability.timeline)
    assert out["cost"]["n_simulations"] == 1000
    assert isinstance(out["control_variate"], (dict, type(None)))


def test_the_results_of_the_exact_methods_go_through_json(repairable):
    results = [
        repairable.expected_events(1000.0),
        repairable.expected_cost(1000.0),
        repairable.availability_rate(100.0),
        repairable.spares_demand(300.0)["a"],
        repairable.spares_demand(300.0, parts={"p": ["a", "b"]})["p"],
        repairable.spares_stock(10.0, fill_rate=0.9)["a"],
    ]
    for result in results:
        assert isinstance(result, _ResultMapping)
        assert set(as_json(result)) == fields_of(result)
    demand = as_json(results[3])
    assert demand["probabilities"] == pytest.approx(
        results[3].probabilities.tolist()
    )
    assert as_json(results[4])["members"] == ["a", "b"]


def test_a_simulated_cost_goes_through_json(repairable):
    cost = repairable.cost(1000.0, mc_samples=200, seed=1)
    out = as_json(cost)
    assert set(out) == fields_of(cost)
    assert len(out["samples"]) == 200


def test_a_capacity_distribution_goes_through_json():
    rbd = NonRepairableRBD(
        PAIR, {"a": W([100, 2]), "b": W([90, 2])}, capacity={"a": 1, "b": 2}
    )
    capacity = rbd.capacity_distribution(50.0)
    out = as_json(capacity)
    assert out["levels"] == capacity.levels.tolist()
    assert out["probabilities"] == pytest.approx(
        capacity.probabilities.tolist()
    )


def test_an_interval_goes_through_json():
    rbd = NonRepairableRBD(PAIR, {"a": W([100, 2]), "b": W([90, 2])})
    interval = rbd.mean_time_to_failure_interval(mc_samples=500, seed=1)
    out = as_json(interval)
    assert set(out) == fields_of(interval)
    assert out["lower"] <= out["estimate"] <= out["upper"]


def test_an_uncertainty_result_goes_through_json():
    lives = np.random.default_rng(2).weibull(2, 30) * 100
    rbd = NonRepairableRBD(
        PAIR, {"a": surv.Weibull.fit(lives), "b": W([90, 2])}
    )
    result = rbd.mean_uncertainty(n_draws=100, seed=1)
    assert len(as_json(result)["samples"]) == 100
    importance = rbd.uncertainty_importance(50.0, n_draws=100, seed=1)
    assert set(as_json(importance)) == fields_of(importance)


def test_node_names_that_are_tuples_become_text():
    rbd = RepairableRBD(TUPLES, {("pump", 1): unit(), ("pump", 2): unit()})
    result = rbd.availability(100.0, mc_samples=50, seed=1)
    up = as_json(result)["criticalities"]["iou"]["up"]
    assert set(up) == {"('pump', 1)", "('pump', 2)"}
    pairs = NonRepairableRBD(
        TUPLES, {("pump", 1): W([100, 2]), ("pump", 2): W([90, 2])}
    ).joint_importance(50.0)
    out = json.loads(json.dumps(pairs.to_dict()))
    first = out["('pump', 1)"]
    assert first["('pump', 2)"] == pytest.approx(
        float(pairs[("pump", 1), ("pump", 2)])
    )


def test_a_route_goes_through_json(repairable):
    routes = repairable.analysis_routes()
    out = json.loads(json.dumps({k: v.to_dict() for k, v in routes.items()}))
    assert (
        out["mean_availability"]["route"] == routes["mean_availability"].route
    )
    assert set(out["availability"]) == set(routes["availability"].to_dict())


def test_timelines_go_through_json(repairable):
    run = repairable.simulate_timelines(500.0, mc_samples=5, seed=1)
    out = as_json(run)
    assert set(out) == fields_of(run)
    assert len(out["system"]["timelines"]) == 5
    first = json.loads(json.dumps(run.system[0].to_dict()))
    assert first["end"] == 500.0
    assert first["changes"] == run.system[0].changes.tolist()
    assert first["changes"]  # the first history changes state


# -- short reprs --------------------------------------------------------------


def test_the_reprs_are_summaries(repairable, availability):
    assert len(repr(availability)) < 400
    cost = repairable.cost(1000.0, mc_samples=1000, seed=1)
    assert len(repr(cost)) < 300
    demand = SparesDemand(np.full(500, 1 / 500), 1e4, 1, "exact")
    assert len(repr(demand)) < 200
    assert "values>" in repr(demand)
    assert len(repr(repairable.availability)) < 200
    assert len(repr(repairable)) < 200


def test_a_control_variate_and_a_conditional_run_print_short():
    twin = np.linspace(0.0, 1.0, 10_000)
    control = ControlVariate(twin, exact=0.9, coefficient=1.0, correlation=1)
    assert len(repr(control)) < 200
    assert repr(control).endswith("n_simulations=10000)")
    run = ConditionalRun(
        modules=("m",),
        states=2,
        availability_square=0.5,
        whole=False,
        uptimes=np.ones((1000, 2)),
        costs=None,
    )
    assert len(repr(run)) < 200
    assert "n_simulations=1000" in repr(run)


# -- the capacity's mean, a property ------------------------------------------


def test_the_capacity_mean_is_a_property():
    rbd = NonRepairableRBD(
        PAIR, {"a": W([100, 2]), "b": W([90, 2])}, capacity={"a": 1, "b": 2}
    )
    capacity = rbd.capacity_distribution(50.0)
    expected = float(capacity.levels @ capacity.probabilities)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert capacity.mean == pytest.approx(expected)
        assert float(capacity.mean) == pytest.approx(expected)
    with pytest.warns(FutureWarning, match="0.14"):
        assert capacity.mean() == pytest.approx(expected)


def test_spares_mean_is_still_a_property():
    demand = SparesDemand(np.array([0.5, 0.5]), 1.0, 1, "exact")
    assert demand.mean == 0.5


# -- a risk reduction worth that is infinite ----------------------------------


def test_an_infinite_risk_reduction_worth_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rbd = NonRepairableRBD(PAIR, {"a": W([100, 2]), "b": W([90, 2])})
        worth = rbd.risk_reduction_worth(50.0)
        repairable = RepairableRBD(PAIR, {"a": unit(), "b": unit()})
        repairable_worth = repairable.risk_reduction_worth()
        tree = FaultTree(
            {"T": ("and", ["a", "b"])}, {"a": 0.01, "b": 0.02}
        ).risk_reduction_worth()
    for values in (worth, repairable_worth, tree):
        assert all(math.isinf(float(v)) for v in values.values())
