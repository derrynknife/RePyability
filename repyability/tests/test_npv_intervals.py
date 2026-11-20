"""The replacement intervals chosen for an equivalent annual cost or a net
present value (``optimal_replacement_intervals(objective=...)``): against
one unit's discounted policy, the cost rate at a rate of 0, the textbook
discounted renewal cycle on a grid, and ``expected_npv`` itself."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import NonRepairable, RepairableRBD
from repyability.rbd import _discounted_long_run

A_YEAR = math.log(1.10) / 8760


def pump(interval=600.0, duration=True):
    spec = {
        "reliability": surv.Weibull.from_params([1000, 2.5]),
        "repairability": surv.LogNormal.from_params([3.0, 0.5]),
        "replace_cost": 5000.0,
        "downtime_cost": 30.0,
        "preventive": {"interval": interval, "cost": 1000.0},
    }
    if duration:
        spec["preventive"]["duration"] = surv.Weibull.from_params([8, 3])
    return spec


def plant(**kwargs):
    return RepairableRBD(
        [("s", "a"), ("a", "b1"), ("a", "b2"), ("b1", "t"), ("b2", "t")],
        {"a": pump(), "b1": pump(), "b2": pump()},
        downtime_cost_rate=500.0,
        **kwargs,
    )


def one_unit(life, cp=1.0, cu=5.0, interval=500.0):
    return RepairableRBD(
        [("s", "unit"), ("unit", "t")],
        {
            "unit": {
                "reliability": life,
                "repairability": "instant",
                "replace_cost": cu,
                "preventive": {"interval": interval, "cost": cp},
            }
        },
    )


@pytest.mark.parametrize("rate", [A_YEAR, 1e-4, 1e-3])
def test_one_unit_is_the_units_own_discounted_policy(rate):
    life = surv.Weibull.from_params([1000, 2.5])
    unit = NonRepairable(life)
    unit.set_costs_planned_and_unplanned(cp=1, cu=5)
    policy = unit.optimal_replacement_policy(discount_rate=rate)
    plan = one_unit(life).optimal_replacement_intervals(
        objective="eac", discount_rate=rate
    )
    assert plan.intervals["unit"] == pytest.approx(policy.interval, rel=1e-5)
    assert plan.objective_value == pytest.approx(policy.eac, rel=1e-10)
    assert plan.objective == "eac" and plan.discount_rate == rate
    assert "Fox" in plan.basis and "Approximate" not in plan.basis


@pytest.mark.parametrize(
    "targets",
    [
        {},
        {"min_availability": 0.98032},
        {"allowed": [400.0, 500.0, 600.0, math.inf]},
    ],
)
def test_at_rate_0_the_plan_is_the_cost_rates(targets):
    system = plant()
    by_cost = system.optimal_replacement_intervals(**targets)
    by_eac = system.optimal_replacement_intervals(objective="eac", **targets)
    assert by_eac.intervals == by_cost.intervals
    assert by_eac.cost_rate == by_cost.cost_rate
    assert by_eac.availability == by_cost.availability
    assert by_eac.objective_value == by_cost.cost_rate
    assert by_cost.objective == "cost" and by_cost.objective_value is None


@pytest.mark.parametrize("duration", [True, False])
def test_a_components_cost_is_its_discounted_costs_for_ever(duration):
    # r times the present value of every cost from new, the horizon so
    # long that what is left is discounted to exp(-40).
    rate = 1e-3
    system = RepairableRBD(
        [("s", "a"), ("a", "t")], {"a": pump(duration=duration)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        present = system.expected_cost(40.0 / rate, discount_rate=rate).mean
    eac = _discounted_long_run.node_eac(system, "a", rate)
    assert eac == pytest.approx(rate * float(present), rel=1e-6)
    assert _discounted_long_run.node_eac(system, "a", 0.0) == pytest.approx(
        system.expected_cost_rate(), rel=1e-9
    )


SERIES = {
    "a": dict(alpha=1000.0, beta=2.5, mu=0.05, cu=5000.0, cp=1000.0),
    "b": dict(alpha=700.0, beta=3.5, mu=0.1, cu=3000.0, cp=400.0),
}
DOWN, VALUE = 800.0, 200.0


def series(interval=500.0):
    return RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            node: {
                "reliability": surv.Weibull.from_params(
                    [c["alpha"], c["beta"]]
                ),
                "repairability": surv.Exponential.from_params([c["mu"]]),
                "replace_cost": c["cu"],
                "preventive": {"interval": interval, "cost": c["cp"]},
            }
            for node, c in SERIES.items()
        },
        downtime_cost_rate=DOWN,
    )


def textbook(node, age, rate):
    """A component's equivalent annual cost and long-run availability at
    a replacement ``age``, written out: the failure density and the
    survival function integrated by scipy, an exponential repair's
    discounted mean time ``1 / (mu + r)``."""
    c = SERIES[node]

    def sf(x):
        return math.exp(-((x / c["alpha"]) ** c["beta"]))

    def pdf(x):
        return c["beta"] / x * (x / c["alpha"]) ** c["beta"] * sf(x)

    up = quad(lambda x: math.exp(-rate * x) * sf(x), 0, age)[0]
    failed = quad(lambda x: math.exp(-rate * x) * pdf(x), 0, age)[0]
    kept = math.exp(-rate * age) * sf(age)
    repair = 1.0 / (c["mu"] + rate)
    eac = (c["cu"] * failed + c["cp"] * kept) / (up + failed * repair)
    mean_up = quad(sf, 0, age)[0]
    availability = mean_up / (mean_up + (1.0 - sf(age)) / c["mu"])
    return eac, availability


def test_a_series_system_against_every_plan_of_a_grid():
    rate = 5e-4
    ages = np.geomspace(150.0, 1500.0, 47)
    own = {node: [textbook(node, t, rate) for t in ages] for node in SERIES}
    # The equivalent annual cost of every pair of ages, written out.
    grid = np.array(
        [
            [
                ea + eb + DOWN * (1.0 - aa * ab) - VALUE * aa * ab
                for eb, ab in own["b"]
            ]
            for ea, aa in own["a"]
        ]
    )
    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    system = series()
    kwargs = dict(objective="eac", discount_rate=rate, value_rate=VALUE)
    # From the grid's ages, every combination valued: the grid's best.
    allowed = system.optimal_replacement_intervals(
        allowed=[float(t) for t in ages[::2]], **kwargs
    )
    coarse = grid[::2, ::2]
    ci, cj = np.unravel_index(int(np.argmin(coarse)), coarse.shape)
    assert allowed.intervals == {
        "a": float(ages[::2][ci]),
        "b": float(ages[::2][cj]),
    }
    assert allowed.objective_value == pytest.approx(coarse[ci, cj], rel=1e-8)
    # Searched: no plan of the grid is better, and it is by the grid's.
    plan = system.optimal_replacement_intervals(**kwargs)
    assert plan.objective_value <= grid.min() + 1e-9 * abs(grid.min())
    step = ages[1] / ages[0]
    assert ages[i] / step < plan.intervals["a"] < ages[i] * step
    assert ages[j] / step < plan.intervals["b"] < ages[j] * step
    assert "Approximate" in plan.basis
    assert "downtime" in plan.basis and "value" in plan.basis


def test_a_higher_rate_does_not_shorten_these_intervals():
    # As observed on these cases; not a theorem, and asserted on them
    # only.
    life = surv.Weibull.from_params([1000, 2.5])
    unit = one_unit(life)
    system = series()
    rates = [0.0, A_YEAR, 1e-4, 5e-4, 1e-3]
    lone, both = [], []
    for rate in rates:
        kwargs = dict(objective="eac", discount_rate=rate)
        lone.append(unit.optimal_replacement_intervals(**kwargs).intervals)
        both.append(system.optimal_replacement_intervals(**kwargs).intervals)
    for before, after in zip(lone, lone[1:]):
        assert after["unit"] >= before["unit"] * (1 - 1e-6)
    for before, after in zip(both, both[1:]):
        for node in SERIES:
            assert after[node] >= before[node] * (1 - 1e-6)
    assert lone[-1]["unit"] > 1.05 * lone[0]["unit"]


def test_the_targets_are_kept():
    system = series()
    kwargs = dict(objective="eac", discount_rate=5e-4)
    free = system.optimal_replacement_intervals(**kwargs)
    floor = free.availability + 0.0005
    held = system.optimal_replacement_intervals(
        min_availability=floor, **kwargs
    )
    assert held.availability >= floor - 1e-9
    assert held.objective_value > free.objective_value
    # A cap on the long-run cost rate that the free plan does not meet:
    # worth 5000 a unit time up, the best plan is the most available.
    options = [150.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    kwargs["value_rate"] = 5000.0
    loose = system.optimal_replacement_intervals(allowed=options, **kwargs)
    cheapest = system.optimal_replacement_intervals(allowed=options)
    assert loose.cost_rate > cheapest.cost_rate
    cap = 0.5 * (loose.cost_rate + cheapest.cost_rate)
    capped = system.optimal_replacement_intervals(
        max_cost_rate=cap, allowed=options, **kwargs
    )
    assert capped.cost_rate <= cap < loose.cost_rate
    assert capped.objective_value > loose.objective_value
    assert capped.objective == "eac"


def npv_of(system, intervals, horizon, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            system.with_intervals(intervals)
            .expected_npv(horizon, **kwargs)
            .npv
        )


def test_npv_from_allowed_intervals_is_the_best_expected_npv():
    life = surv.Weibull.from_params([1000, 2.5])
    unit = one_unit(life)
    options = [350.0, 450.0, 525.0, 600.0, 800.0, math.inf]
    kwargs = dict(discount_rate=1e-3)
    values = [npv_of(unit, {"unit": t}, 3000.0, **kwargs) for t in options]
    plan = unit.optimal_replacement_intervals(
        objective="npv", horizon=3000.0, allowed=options, **kwargs
    )
    best = int(np.argmax(values))
    assert plan.intervals == {"unit": options[best]}
    assert plan.objective_value == pytest.approx(values[best], rel=1e-9)
    assert plan.objective == "npv" and plan.horizon == 3000.0
    assert "expected_npv" in plan.basis


def test_npv_is_searched_from_the_long_runs_plan():
    life = surv.Weibull.from_params([1000, 2.5])
    unit = one_unit(life)
    kwargs = dict(discount_rate=1e-3, value_rate=0.01)
    long_run = unit.optimal_replacement_intervals(objective="eac", **kwargs)
    plan = unit.optimal_replacement_intervals(
        objective="npv", horizon=3000.0, **kwargs
    )
    achieved = npv_of(unit, plan.intervals, 3000.0, **kwargs)
    assert plan.objective_value == pytest.approx(achieved, rel=1e-9)
    start = npv_of(unit, long_run.intervals, 3000.0, **kwargs)
    assert plan.objective_value >= start - 1e-9 * abs(start)
    assert plan.value_rate == 0.01


def test_like_units_can_be_worth_replacing_at_different_ages_from_new():
    # From new, a pair replaced at one age are down together; the long
    # run does not see it, the net present value does.
    pair = RepairableRBD(
        [("s", "b1"), ("s", "b2"), ("b1", "t"), ("b2", "t")],
        {"b1": pump(), "b2": pump()},
        downtime_cost_rate=5000.0,
    )
    options = [400.0, 600.0]
    kwargs = dict(discount_rate=1e-4)
    plan = pair.optimal_replacement_intervals(
        objective="npv", horizon=3000.0, allowed=options, **kwargs
    )
    values = {
        (a, b): npv_of(pair, {"b1": a, "b2": b}, 3000.0, **kwargs)
        for a in options
        for b in options
    }
    best = max(values, key=values.get)
    assert (plan.intervals["b1"], plan.intervals["b2"]) == best
    assert plan.objective_value == pytest.approx(values[best], rel=1e-9)


def test_crews_salaries_are_in_the_value():
    crewed = plant(repair_crews=1, crew_cost_rate=7.0)
    free = plant()
    rate = 1e-4
    for objective, extra, paid in (
        ("eac", {}, 7.0),
        ("npv", {"horizon": 2000.0}, 7.0 * -math.expm1(-rate * 2000) / rate),
    ):
        kwargs = dict(
            objective=objective,
            discount_rate=rate,
            allowed={"a": [600.0], "b1": [500.0], "b2": [500.0]},
            **extra,
        )
        with pytest.raises(NotImplementedError, match="crew"):
            crewed.optimal_replacement_intervals(**kwargs)
        plan = crewed.optimal_replacement_intervals(
            assume_unlimited_crews=True, **kwargs
        )
        base = free.optimal_replacement_intervals(**kwargs)
        sign = -1.0 if objective == "npv" else 1.0
        assert plan.objective_value == pytest.approx(
            base.objective_value + sign * paid, rel=1e-12
        )
        assert plan.cost_rate == pytest.approx(base.cost_rate + 7.0)


def with_valve(**schedule):
    return RepairableRBD(
        [("s", "a"), ("a", "v"), ("v", "t")],
        {
            "a": pump(),
            "v": {
                "reliability": surv.Exponential.from_params([1e-4]),
                "repairability": surv.Exponential.from_params([0.1]),
                "replace_cost": 100.0,
                **schedule,
            },
        },
    )


@pytest.mark.parametrize(
    "schedule, what",
    [
        ({"inspection": {"interval": 500.0, "cost": 10.0}}, "tested"),
        (
            {
                "preventive": {
                    "interval": 2000.0,
                    "cost": 10.0,
                    "policy": "block",
                }
            },
            "block",
        ),
    ],
)
def test_eac_refuses_what_has_no_cycle_of_its_own(schedule, what):
    system = with_valve(**schedule)
    with pytest.raises(NotImplementedError, match=what) as refused:
        system.optimal_replacement_intervals(
            ["a"], objective="eac", discount_rate=1e-4
        )
    assert 'objective="npv"' in str(refused.value)
    # Undiscounted, it is the cost rate, which has it.
    plain = system.optimal_replacement_intervals(["a"])
    by_eac = system.optimal_replacement_intervals(["a"], objective="eac")
    assert by_eac.intervals == plain.intervals


def test_the_arguments_are_checked():
    system = plant()
    choose = system.optimal_replacement_intervals
    with pytest.raises(ValueError, match="objective must be"):
        choose(objective="profit")
    with pytest.raises(ValueError, match="discount_rate needs an objective"):
        choose(discount_rate=1e-4)
    with pytest.raises(ValueError, match="value_rate and horizon need"):
        choose(value_rate=3.0, horizon=100.0)
    with pytest.raises(ValueError, match="give horizon"):
        choose(objective="npv", discount_rate=1e-4)
    with pytest.raises(ValueError, match="no horizon"):
        choose(objective="eac", horizon=1000.0)
    with pytest.raises(ValueError, match="value_rate must be"):
        choose(objective="eac", value_rate=-1.0)
    with pytest.raises(ValueError, match="discount_rate must be"):
        choose(objective="eac", discount_rate=-0.1)


def test_the_plan_goes_through_json():
    plan = plant().optimal_replacement_intervals(
        objective="eac", discount_rate=1e-4, value_rate=100.0
    )
    again = json.loads(json.dumps(plan.to_dict()))
    assert again["objective"] == "eac"
    assert again["objective_value"] == plan.objective_value
    assert again["basis"] == plan.basis
