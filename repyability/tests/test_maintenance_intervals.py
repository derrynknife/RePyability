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


# -- replacement on a calendar (#230) -----------------------------------------

CALENDAR = [250.0, 500.0, 750.0, math.inf]


def calendar_best(plant, allowed, feasible=lambda rbd: True):
    """The cheapest combination of ``allowed`` intervals, by brute force."""
    best = (math.inf, None)
    for combination in itertools.product(allowed, repeat=3):
        intervals = dict(zip(["a", "b1", "b2"], combination))
        rbd = plant._with_intervals(preventive=intervals)
        if feasible(rbd):
            best = min(best, (rbd.expected_cost_rate(), combination))
    return best


def test_replacement_intervals_from_a_calendar(plant):
    plan = plant.optimal_replacement_intervals(allowed=CALENDAR)
    cost, combination = calendar_best(plant, CALENDAR)
    assert tuple(plan.intervals.values()) == combination
    assert plan.cost_rate == pytest.approx(cost, rel=1e-12)
    # The calendar's best costs no less than the continuous search's.
    assert plan.cost_rate >= plant.optimal_replacement_intervals().cost_rate


def test_replacement_intervals_from_a_calendar_per_node(plant):
    allowed = {"a": [750.0], "b1": CALENDAR, "b2": [500.0, math.inf]}
    plan = plant.optimal_replacement_intervals(allowed=allowed)
    assert plan.intervals["a"] == 750.0
    assert plan.intervals["b2"] in (500.0, math.inf)
    target = plant.optimal_replacement_intervals(
        allowed=CALENDAR, min_availability=0.98
    )
    _, combination = calendar_best(
        plant, CALENDAR, lambda rbd: rbd.mean_availability() >= 0.98
    )
    assert target.availability >= 0.98
    assert tuple(target.intervals.values()) == combination


@pytest.mark.parametrize(
    "allowed, match",
    [
        ([0.0, 500.0], "positive numbers \\(inf for never\\)"),
        ([-1.0], "positive numbers"),
        ({"a": [500.0]}, "no intervals for node 'b1'"),
        ({"a": [500.0], "b1": [], "b2": [500.0]}, "no intervals"),
    ],
)
def test_what_a_calendar_refuses(plant, allowed, match):
    with pytest.raises(ValueError, match=match):
        plant.optimal_replacement_intervals(allowed=allowed)


def test_integer_node_names():
    rbd = RepairableRBD(
        [("s", 1), (1, "t")], {1: pump()}, downtime_cost_rate=500.0
    )
    plan = rbd.optimal_replacement_intervals()
    assert set(plan.intervals) == {1}


# -- Proof-test intervals -----------------------------------------------------

CALENDAR = [730.0, 2190.0, 4380.0, 8760.0, 17520.0]  # 1, 3, 6, 12, 24 months


def valve(interval=8760.0, rate=2e-6, cost=500.0, **extra):
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval, "cost": cost},
        **extra,
    }


def one_valve(**kwargs):
    return RepairableRBD([("s", "v"), ("v", "t")], {"v": valve(**kwargs)})


def two_valves(**kwargs):
    return RepairableRBD(
        [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        {"v1": valve(**kwargs), "v2": valve(**kwargs)},
    )


def test_one_inspected_component_by_formula():
    # Testing costs c_i per test; each hour the unit lies failed costs c_d:
    # the cost rate c_i / tau + c_d U(tau) is least near
    # sqrt(2 c_i / (lambda c_d)).
    lam, c_i, c_d = 1e-4, 2000.0, 500.0
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": valve(interval=100.0, rate=lam, cost=c_i, downtime_cost=c_d)},
    )
    plan = rbd.optimal_inspection_intervals()

    def rate(tau):
        down = 1.0 + math.expm1(-lam * tau) / (lam * tau)
        return c_i / tau + c_d * down

    taus = np.linspace(200.0, 400.0, 20001)
    brute = taus[np.argmin([rate(tau) for tau in taus])]
    assert plan.intervals["p"] == pytest.approx(brute, rel=1e-3)
    assert plan.cost_rate == pytest.approx(rate(brute), rel=1e-9)
    assert plan.intervals["p"] == pytest.approx(
        math.sqrt(2 * c_i / (lam * c_d)), rel=0.05
    )


def test_redundancy_lets_the_tests_be_rarer():
    # PFDavg at most 1e-3: about lambda tau / 2 for one valve, (lambda tau)^2
    # / 3 for two tested together.
    target = 1.0 - 1e-3
    one = one_valve().optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=target
    )
    two = two_valves().optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=target
    )
    assert one.intervals == {"v": 730.0}
    assert two.intervals == {"v1": 17520.0, "v2": 17520.0}
    assert one.availability >= target and two.availability >= target
    assert 1.0 - two.availability == pytest.approx(
        (2e-6 * 17520.0) ** 2 / 3, rel=0.03
    )


def test_every_combination_is_tried():
    # Two valves with different costs: the method against enumeration.
    rbd = RepairableRBD(
        [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        {"v1": valve(cost=500.0), "v2": valve(cost=2000.0, rate=5e-6)},
        downtime_cost_rate=5000.0,
    )
    target = 1.0 - 2e-4
    plan = rbd.optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=target
    )
    best = (math.inf, None)
    for a, b in itertools.product(CALENDAR, CALENDAR):
        design = rbd._with_intervals(inspection={"v1": a, "v2": b})
        if design.mean_availability() >= target:
            best = min(best, (design.expected_cost_rate(), (a, b)))
    assert (plan.intervals["v1"], plan.intervals["v2"]) == best[1]
    assert plan.cost_rate == pytest.approx(best[0])


def test_the_local_search_on_many_combinations(monkeypatch):
    # Force the local search on a problem small enough to enumerate.
    import repyability.rbd.repairable_rbd as module

    rbd = two_valves()
    exhaustive = rbd.optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=1.0 - 1e-4
    )
    monkeypatch.setattr(module, "_MAX_COMBINATIONS", 1)
    local = rbd.optimal_inspection_intervals(
        allowed=CALENDAR, min_availability=1.0 - 1e-4
    )
    # The valves are alike, so the intervals may come out swapped.
    assert sorted(local.intervals.values()) == sorted(
        exhaustive.intervals.values()
    )
    assert local.cost_rate == pytest.approx(exhaustive.cost_rate)
    assert local.availability >= 1.0 - 1e-4


def test_a_cost_cap_on_the_tests():
    # Within 0.1 per hour of tests, the most available two-valve schedule.
    plan = two_valves().optimal_inspection_intervals(
        allowed=CALENDAR, max_cost_rate=0.1
    )
    assert plan.cost_rate <= 0.1
    for a, b in itertools.product(CALENDAR, CALENDAR):
        design = two_valves()._with_intervals(inspection={"v1": a, "v2": b})
        if design.expected_cost_rate() <= 0.1:
            assert design.mean_availability() <= plan.availability + 1e-15


def test_what_the_inspection_choice_refuses():
    with pytest.raises(ValueError, match="More than one component"):
        two_valves().optimal_inspection_intervals()
    with pytest.raises(ValueError, match="not a component of the RBD"):
        two_valves().optimal_inspection_intervals(
            nodes=["s"], allowed=CALENDAR
        )
    with pytest.raises(ValueError, match="positive, finite"):
        two_valves().optimal_inspection_intervals(allowed=[730.0, -1.0])
    with pytest.raises(ValueError, match="no intervals for node 'v2'"):
        two_valves().optimal_inspection_intervals(allowed={"v1": CALENDAR})
    with pytest.raises(
        ValueError, match="the most the allowed intervals give"
    ):
        one_valve().optimal_inspection_intervals(
            allowed=CALENDAR, min_availability=1.0 - 1e-5
        )
    with pytest.raises(ValueError, match="No component has hidden failures"):
        RepairableRBD(
            [("s", "c"), ("c", "t")], {"c": pump()}
        ).optimal_inspection_intervals()
