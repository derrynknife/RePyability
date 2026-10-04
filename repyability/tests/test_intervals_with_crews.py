"""Maintenance and test intervals with limited repair crews (#184), and
``with_intervals``, which applies a plan to a diagram.

With limited crews a component can wait for one, and the exact long-run
values the interval choices search do not hold: the choices refuse, unless
asked to choose as if every repair started at once, which gives the plan
of the same diagram with unlimited crews.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD

EDGES = [("s", "a"), ("a", "b1"), ("a", "b2"), ("b1", "t"), ("b2", "t")]


def pump(interval=1000.0):
    return {
        "reliability": surv.Weibull.from_params([1000, 2.5]),
        "repairability": surv.LogNormal.from_params([3.0, 0.5]),
        "replace_cost": 5000.0,
        "preventive": {
            "interval": interval,
            "duration": surv.Weibull.from_params([8, 3]),
            "cost": 1000.0,
        },
    }


def plant(crews=None, **intervals):
    return RepairableRBD(
        EDGES,
        {n: pump(intervals.get(n, 1000.0)) for n in ("a", "b1", "b2")},
        downtime_cost_rate=500.0,
        repair_crews=crews,
    )


def valve(interval=8760.0, offset=0.0):
    return {
        "reliability": surv.Exponential.from_params([2e-6]),
        "repairability": "instant",
        "inspection": {"interval": interval, "cost": 500.0, "offset": offset},
    }


def valves(crews=None):
    return RepairableRBD(
        [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        {"v1": valve(), "v2": valve()},
        downtime_cost_rate=100.0,
        repair_crews=crews,
    )


def test_limited_crews_refuse_and_say_what_to_do():
    crewed = plant(crews=1)
    with pytest.raises(NotImplementedError, match="assume_unlimited_crews"):
        crewed.optimal_replacement_intervals()
    with pytest.raises(NotImplementedError, match="with_intervals"):
        valves(crews=1).optimal_inspection_intervals(
            allowed=[4380.0, 8760.0], min_availability=0.999
        )
    for rbd, method in (
        (crewed, "optimal_replacement_intervals"),
        (valves(crews=1), "optimal_inspection_intervals"),
    ):
        route = rbd.analysis_routes()[method]
        assert route.route == "refused"
        assert "assume_unlimited_crews=True" in route.reason


def test_the_plan_without_waiting_is_the_unlimited_crews_plan():
    assumed = plant(crews=1).optimal_replacement_intervals(
        assume_unlimited_crews=True
    )
    unlimited = plant().optimal_replacement_intervals()
    assert assumed == unlimited
    tests = valves(crews=1).optimal_inspection_intervals(
        allowed=[2190.0, 4380.0, 8760.0],
        min_availability=1 - 1e-5,
        offsets="stagger",
        assume_unlimited_crews=True,
    )
    assert tests == valves().optimal_inspection_intervals(
        allowed=[2190.0, 4380.0, 8760.0],
        min_availability=1 - 1e-5,
        offsets="stagger",
    )
    # Enough crews for every job: nothing waits, and nothing is refused.
    assert plant(crews=3).optimal_replacement_intervals() == unlimited


def test_a_plan_is_applied_to_the_diagram():
    crewed = plant(crews=1)
    plan = crewed.optimal_replacement_intervals(assume_unlimited_crews=True)
    applied = crewed.with_intervals(plan)
    assert applied.repair_crews == 1
    for node, interval in plan.intervals.items():
        assert applied._preventive[node].interval == interval
    # Without the crews, it is the plan's own values.
    free = plant().with_intervals(plan)
    assert free.expected_cost_rate() == pytest.approx(plan.cost_rate, 1e-12)
    assert free.mean_availability() == pytest.approx(
        plan.availability, rel=1e-12
    )
    # With them, simulated: waiting for the crew costs more downtime.
    waiting = applied.cost(200_000.0, mc_samples=40, seed=3)
    unwaiting = free.cost(200_000.0, mc_samples=40, seed=3)
    assert waiting.mean > unwaiting.mean
    assert unwaiting.mean / 200_000.0 == pytest.approx(plan.cost_rate, 0.05)
    # The diagram itself is unchanged.
    assert crewed._preventive["a"].interval == 1000.0


def test_intervals_and_offsets_by_hand():
    rbd = valves()
    moved = rbd.with_intervals({"v1": 4380.0, "v2": 4380.0}, {"v2": 2190.0})
    assert moved._inspection["v1"].interval == 4380.0
    assert moved._inspection["v2"].offset == 2190.0
    plan = rbd.optimal_inspection_intervals(
        allowed=[4380.0, 8760.0], min_availability=0.999, offsets="stagger"
    )
    applied = rbd.with_intervals(plan)
    assert {
        n: applied._inspection[n].offset for n in ("v1", "v2")
    } == plan.offsets
    assert applied.mean_availability() == pytest.approx(
        plan.availability, rel=1e-12
    )
    # An offset keeps its share of a changed interval, as by default.
    halfway = RepairableRBD(
        [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        {"v1": valve(), "v2": valve(offset=4380.0)},
    )
    assert halfway.with_intervals({"v2": 2190.0})._inspection[
        "v2"
    ].offset == pytest.approx(1095.0)
    # Only an offset.
    assert (
        rbd.with_intervals({}, {"v2": 100.0})._inspection["v2"].offset == 100.0
    )


def test_never_replacing_drops_the_schedule():
    rbd = plant()
    never = rbd.with_intervals({"a": math.inf})
    assert "a" not in never._preventive
    assert set(never._preventive) == {"b1", "b2"}
    built = RepairableRBD(
        EDGES,
        {
            "a": {k: v for k, v in pump().items() if k != "preventive"},
            "b1": pump(),
            "b2": pump(),
        },
        downtime_cost_rate=500.0,
    )
    assert never.expected_cost_rate() == built.expected_cost_rate()


def test_with_intervals_is_checked():
    rbd = plant()
    with pytest.raises(ValueError, match="no maintenance or test schedule"):
        rbd.with_intervals({"s": 100.0})
    with pytest.raises(ValueError, match="no maintenance or test schedule"):
        RepairableRBD(
            [("s", "x"), ("x", "t")],
            {
                "x": {
                    "reliability": surv.Exponential.from_params([1e-3]),
                    "repairability": "instant",
                }
            },
        ).with_intervals({"x": 10.0})
    with pytest.raises(ValueError, match="no test schedule"):
        rbd.with_intervals({}, {"a": 10.0})
    with pytest.raises(ValueError):
        rbd.with_intervals({"a": -5.0})
    with pytest.raises(ValueError):
        valves().with_intervals({"v1": 100.0}, {"v1": 150.0})
    block = RepairableRBD(
        [("s", "x"), ("x", "t")],
        {
            "x": {
                **pump(),
                "preventive": {"interval": 500.0, "policy": "block"},
            }
        },
    )
    with pytest.raises(ValueError, match="only age replacement"):
        block.with_intervals({"x": np.inf})
