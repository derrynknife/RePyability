"""The test intervals chosen for an equivalent annual cost or a net
present value (``optimal_inspection_intervals(objective=...)``): a tested
unit's discounted cycle against its closed form and against
``expected_cost``, the cost rate at a rate of 0, and a PFDavg limit that
binds under each objective."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _discounted_long_run

RATE, TEST, FOUND, DOWN = 2e-5, 500.0, 2000.0, 3.0


def valve(life=None, interval=8760.0, offset=0.0, **inspection):
    return {
        "reliability": life or surv.Exponential.from_params([RATE]),
        "repairability": "instant",
        "replace_cost": FOUND,
        "downtime_cost": DOWN,
        "inspection": {
            "interval": interval,
            "cost": TEST,
            "offset": offset,
            **inspection,
        },
    }


def one(downtime_cost_rate=20.0, **kwargs):
    return RepairableRBD(
        [("s", "v"), ("v", "t")],
        {"v": valve(**kwargs)},
        downtime_cost_rate=downtime_cost_rate,
    )


def closed_form(interval, rate):
    """An exponential unit's equivalent annual cost, its hidden failures
    found at tests every ``interval`` and repaired at once: with ``q``
    the chance of surviving an interval and ``e`` an interval's
    discount, a cycle ends at test ``k`` with chance ``q^(k-1) (1 - q)``,
    so its discount is ``phi = e (1 - q) / (1 - e q)``; its tests cost
    ``TEST e / (1 - e q)``, the failure ``FOUND phi`` when found, and
    the unit's downtime, from the failure (discounted, ``lam / (lam +
    r)``) to the test, ``DOWN (lam / (lam + r) - phi) / r``."""
    q = math.exp(-RATE * interval)
    e = math.exp(-rate * interval)
    phi = e * (1.0 - q) / (1.0 - e * q)
    cycle = (
        TEST * e / (1.0 - e * q)
        + FOUND * phi
        + DOWN * (RATE / (RATE + rate) - phi) / rate
    )
    return rate * cycle / (1.0 - phi)


@pytest.mark.parametrize("rate", [1e-6, 1e-5, 1e-4, 1e-3])
@pytest.mark.parametrize("interval", [100.0, 1500.0, 8760.0, 90000.0])
def test_an_exponential_units_cycle_is_its_closed_form(rate, interval):
    system = one(interval=interval)
    assert _discounted_long_run.node_eac(system, "v", rate) == pytest.approx(
        closed_form(interval, rate), rel=1e-10
    )


def test_the_closed_form_tends_to_the_cost_rate():
    system = one(interval=1500.0, downtime_cost_rate=0.0)
    assert closed_form(1500.0, 1e-10) == pytest.approx(
        system.expected_cost_rate(), rel=1e-6
    )


@pytest.mark.parametrize("offset", [0.0, 100.0])
@pytest.mark.parametrize("weibull", [False, True])
def test_a_tested_units_cost_is_its_discounted_costs_for_ever(offset, weibull):
    # r times the present value of every cost from new, over a horizon
    # so long that what is left is discounted to exp(-40).
    rate = 1e-3
    life = surv.Weibull.from_params([1500, 2.0]) if weibull else None
    system = one(
        life=life, interval=300.0, offset=offset, downtime_cost_rate=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        present = system.expected_cost(40.0 / rate, discount_rate=rate).mean
    eac = _discounted_long_run.node_eac(system, "v", rate)
    assert eac == pytest.approx(rate * float(present), rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"min_availability": 0.99},
        {"allowed": [730.0, 2190.0, 4380.0, 8760.0]},
        {"allowed": [730.0, 2190.0, 4380.0], "offset_shares": "stagger"},
    ],
)
def test_at_rate_0_the_plan_is_the_cost_rates(kwargs):
    system = pair() if "allowed" in kwargs else one()
    by_cost = system.optimal_inspection_intervals(**kwargs)
    by_eac = system.optimal_inspection_intervals(objective="eac", **kwargs)
    assert by_eac.intervals == by_cost.intervals
    assert by_eac.offsets == by_cost.offsets
    assert by_eac.cost_rate == by_cost.cost_rate
    assert by_eac.availability == by_cost.availability
    assert by_eac.objective_value == by_cost.cost_rate


def pair(**kwargs):
    return RepairableRBD(
        [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        {"v1": valve(), "v2": valve()},
        downtime_cost_rate=2000.0,
        **kwargs,
    )


def test_one_units_interval_is_the_closed_forms_best():
    rate = 1e-4
    system = one(downtime_cost_rate=0.0)
    plan = system.optimal_inspection_intervals(
        objective="eac", discount_rate=rate
    )
    grid = np.geomspace(300.0, 20000.0, 400)
    costs = [closed_form(t, rate) for t in grid]
    best = int(np.argmin(costs))
    step = grid[1] / grid[0]
    assert grid[best] / step < plan.intervals["v"] < grid[best] * step
    assert plan.objective_value <= costs[best] * (1 + 1e-12)
    assert plan.objective_value == pytest.approx(
        closed_form(plan.intervals["v"], rate), rel=1e-10
    )
    # Discounted, the tests are a little further apart here (observed
    # on this case).
    plain = system.optimal_inspection_intervals()
    assert plan.intervals["v"] > plain.intervals["v"]


@pytest.mark.parametrize(
    "objective, extra",
    [("eac", {}), ("npv", {"horizon": 20000.0})],
)
def test_a_pfd_limit_binds_under_each_objective(objective, extra):
    system = one()
    kwargs = dict(objective=objective, discount_rate=1e-4, **extra)
    free = system.optimal_inspection_intervals(**kwargs)
    limit = 0.5 * (1.0 - free.availability)  # half its PFDavg
    held = system.optimal_inspection_intervals(
        min_availability=1.0 - limit, **kwargs
    )
    assert 1.0 - held.availability == pytest.approx(limit, rel=1e-4)
    assert 1.0 - held.availability <= limit * (1 + 1e-9)
    assert held.intervals["v"] < free.intervals["v"]
    worse = held.objective_value - free.objective_value
    assert worse > 0 if objective == "eac" else worse < 0
    assert held.objective == objective


def test_npv_from_allowed_intervals_is_the_best_expected_npv():
    system = pair()
    calendar = [2190.0, 4380.0]
    kwargs = dict(discount_rate=1e-4, value_rate=10.0)
    plan = system.optimal_inspection_intervals(
        objective="npv", horizon=20000.0, allowed=calendar, **kwargs
    )
    values = {}
    for a in calendar:
        for b in calendar:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values[(a, b)] = (
                    system.with_intervals({"v1": a, "v2": b})
                    .expected_npv(20000.0, **kwargs)
                    .npv
                )
    best = max(values, key=values.get)
    assert (plan.intervals["v1"], plan.intervals["v2"]) == best
    assert plan.objective_value == pytest.approx(values[best], rel=1e-9)
    assert plan.offsets == {"v1": 0.0, "v2": 0.0}


def test_offsets_are_chosen_with_the_objective():
    system = pair()
    plan = system.optimal_inspection_intervals(
        objective="eac",
        discount_rate=1e-4,
        allowed=[730.0, 2190.0, 4380.0],
        offset_shares="stagger",
    )
    assert plan.objective == "eac"
    again = system.with_intervals(plan)
    rate = 1e-4
    own = sum(
        _discounted_long_run.node_eac(again, node, rate)
        for node in ("v1", "v2")
    )
    assert plan.objective_value == pytest.approx(
        own + 2000.0 * (1.0 - again.mean_availability()), rel=1e-10
    )


@pytest.mark.parametrize(
    "inspection, repair",
    [
        ({"coverage": 0.9, "full_test": 8760.0 * 4}, "instant"),
        ({}, surv.Exponential.from_params([0.1])),
        ({"duration": surv.Exponential.from_params([1.0])}, "instant"),
    ],
)
def test_eac_refuses_tests_with_no_cycle_worked_out(inspection, repair):
    spec = valve(**inspection)
    spec["repairability"] = repair
    system = RepairableRBD([("s", "v"), ("v", "t")], {"v": spec})
    with pytest.raises(NotImplementedError, match="tested") as refused:
        system.optimal_inspection_intervals(
            objective="eac", discount_rate=1e-4
        )
    assert 'objective="npv"' in str(refused.value)
    plain = system.optimal_inspection_intervals()
    by_eac = system.optimal_inspection_intervals(objective="eac")
    assert by_eac.intervals == plain.intervals


def test_the_arguments_are_checked():
    choose = one().optimal_inspection_intervals
    with pytest.raises(ValueError, match="objective must be"):
        choose(objective="profit")
    with pytest.raises(ValueError, match="discount_rate needs an objective"):
        choose(discount_rate=1e-4)
    with pytest.raises(ValueError, match="give horizon"):
        choose(objective="npv")
    with pytest.raises(ValueError, match="no horizon"):
        choose(objective="eac", horizon=1000.0)


def test_crews_salaries_are_in_the_value():
    rate = 1e-4
    crewed = RepairableRBD(
        [("s", "v"), ("v", "p"), ("p", "t")],
        {
            "v": valve(),
            "p": {
                "reliability": surv.Exponential.from_params([1e-3]),
                "repairability": surv.Exponential.from_params([0.1]),
            },
        },
        repair_crews=1,
        crew_cost_rate=7.0,
    )
    free = RepairableRBD(
        **{**crewed._init_args, "repair_crews": None, "crew_cost_rate": 0.0}
    )
    kwargs = dict(objective="eac", discount_rate=rate)
    plan = crewed.optimal_inspection_intervals(
        assume_unlimited_crews=True, **kwargs
    )
    base = free.optimal_inspection_intervals(**kwargs)
    assert plan.intervals == base.intervals
    assert plan.objective_value == pytest.approx(
        base.objective_value + 7.0, rel=1e-12
    )
