"""Choosing maintenance intervals for a system target.

``RepairableRBD.optimal_replacement_intervals`` chooses each component's
age-replacement interval for the lowest long-run cost rate, the lowest that
keeps the system's availability to a target, or the highest availability
within a cost rate. The references: the single-component optimum
(``NonRepairable.find_optimal_replacement``) and brute force over a grid of
intervals.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, RepairableRBD

W = surv.Weibull.from_params
E = surv.Exponential.from_params
LN = surv.LogNormal.from_params


def pump(interval=1000.0, **extra):
    return {
        "reliability": W([1000, 2.5]),
        "repairability": LN([3.0, 0.5]),
        "replace_cost": 5000.0,
        "preventive": {
            "interval": interval,
            "duration": W([8, 3]),
            "cost": 1000.0,
        },
        **extra,
    }


@pytest.fixture(scope="module")
def plant():
    # A pump in series with a pair of pumps in parallel.
    return RepairableRBD(
        [("s", "a"), ("a", "b1"), ("a", "b2"), ("b1", "t"), ("b2", "t")],
        {"a": pump(), "b1": pump(), "b2": pump()},
        downtime_cost_rate=500.0,
    )


def grid_best(plant, feasible=lambda rbd: True):
    """The cheapest intervals on a grid (the pair given the same), and the
    cost rate."""
    best = (math.inf, None)
    for a, b in itertools.product(
        np.arange(500.0, 700.0, 10.0), np.arange(420.0, 700.0, 10.0)
    ):
        rbd = plant._with_intervals(preventive={"a": a, "b1": b, "b2": b})
        if feasible(rbd):
            best = min(best, (rbd.expected_cost_rate(), (a, b)))
    return best


def test_one_component_is_find_optimal_replacement():
    # Instant repair and replacement, no downtime cost: the component's own
    # cost rate, (c_p R(T) + c_u F(T)) / integral_0^T R.
    life = W([1000, 2.5])
    unit = NonRepairable(life)
    unit.set_costs_planned_and_unplanned(cp=1000.0, cu=5000.0)
    rbd = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {
            "c": {
                "reliability": life,
                "repairability": "instant",
                "replace_cost": 5000.0,
                "preventive": {"interval": 800.0, "cost": 1000.0},
            }
        },
    )
    plan = rbd.optimal_replacement_intervals()
    assert plan.intervals["c"] == pytest.approx(
        unit.find_optimal_replacement(), rel=1e-3
    )
    assert plan.cost_rate == pytest.approx(
        unit.cost_rate(unit.find_optimal_replacement()), rel=1e-8
    )
    assert plan.availability == 1.0


def test_the_cheapest_intervals(plant):
    plan = plant.optimal_replacement_intervals()
    grid_cost, (a, b) = grid_best(plant)
    assert plan.cost_rate <= grid_cost + 1e-9
    assert plan.intervals["a"] == pytest.approx(a, abs=15.0)
    assert plan.intervals["b1"] == pytest.approx(b, abs=15.0)
    # The pump alone in the line is replaced later than a pump with a
    # standby: its replacements stop the plant too.
    assert plan.intervals["a"] > plan.intervals["b1"] + 50.0
    assert plan.intervals["b1"] == pytest.approx(plan.intervals["b2"])
    again = plant._with_intervals(preventive=plan.intervals)
    assert again.expected_cost_rate() == pytest.approx(plan.cost_rate)
    assert again.mean_availability() == pytest.approx(plan.availability)


def test_an_availability_target(plant):
    free = plant.optimal_replacement_intervals()
    target = 0.98035
    plan = plant.optimal_replacement_intervals(min_availability=target)
    assert plan.availability >= target
    assert plan.cost_rate > free.cost_rate
    grid_cost, _ = grid_best(
        plant, lambda rbd: rbd.mean_availability() >= target
    )
    assert plan.cost_rate <= grid_cost + 1e-9


def test_a_target_out_of_reach(plant):
    with pytest.raises(
        ValueError, match=r"the most any intervals give is 0\.98"
    ):
        plant.optimal_replacement_intervals(min_availability=0.99)
    with pytest.raises(ValueError, match="the least any intervals cost is 20"):
        plant.optimal_replacement_intervals(max_cost_rate=10.0)


def test_a_cost_cap(plant):
    free = plant.optimal_replacement_intervals()
    cap = free.cost_rate * 1.001
    plan = plant.optimal_replacement_intervals(max_cost_rate=cap)
    assert plan.cost_rate <= cap
    assert plan.availability >= free.availability
    best = max(
        rbd.mean_availability()
        for rbd in (
            plant._with_intervals(preventive={"a": a, "b1": b, "b2": b})
            for a, b in itertools.product(
                np.arange(560.0, 660.0, 10.0), np.arange(460.0, 660.0, 10.0)
            )
        )
        if rbd.expected_cost_rate() <= cap
    )
    assert plan.availability >= best - 1e-9


def test_a_memoryless_unit_is_never_replaced():
    rbd = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {
            "c": {
                "reliability": E([1e-3]),
                "repairability": E([0.1]),
                "replace_cost": 500.0,
                "preventive": {"interval": 500.0, "cost": 100.0},
            }
        },
        downtime_cost_rate=10.0,
    )
    plan = rbd.optimal_replacement_intervals()
    assert plan.intervals["c"] == math.inf


def test_only_the_named_components_change(plant):
    plan = plant.optimal_replacement_intervals(nodes=["a"])
    assert set(plan.intervals) == {"a"}
    fixed = plant._with_intervals(preventive={"a": plan.intervals["a"]})
    assert fixed.expected_cost_rate() == pytest.approx(plan.cost_rate)


def test_what_it_refuses(plant):
    with pytest.raises(ValueError, match="at most one"):
        plant.optimal_replacement_intervals(
            min_availability=0.9, max_cost_rate=30.0
        )
    with pytest.raises(ValueError, match="not a component under age"):
        plant.optimal_replacement_intervals(nodes=["s"])
    with pytest.raises(ValueError, match=r"min_availability must be"):
        plant.optimal_replacement_intervals(min_availability=1.5)
    block = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {"c": pump(preventive={"interval": 500.0, "policy": "block"})},
        downtime_cost_rate=500.0,
    )
    with pytest.raises(ValueError, match="No component is under age"):
        block.optimal_replacement_intervals()
    unpriced = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {
            "c": {
                "reliability": W([1000, 2.5]),
                "repairability": LN([3.0, 0.5]),
                "preventive": {"interval": 500.0},
            }
        },
    )
    with pytest.raises(ValueError, match="Nothing is priced"):
        unpriced.optimal_replacement_intervals()
