"""``allocate_redundancy`` by net present value or equivalent annual cost
(``objective=``, ``value_rate=``): the total cost's design with no value,
every design of a small diagram drawn out and valued, and a design that
changes with the discount rate."""

import itertools
import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD

HORIZON = 87600.0
PARTS = {
    "a": dict(life=1e-3, repair=0.1, fix=500.0, buy=20000.0),
    "b": dict(life=4e-4, repair=0.05, fix=900.0, buy=35000.0),
    "c": dict(life=2e-3, repair=0.5, fix=100.0, buy=6000.0),
}
DOWNTIME = 40.0


def part(name):
    p = PARTS[name]
    return {
        "reliability": surv.Exponential.from_params([p["life"]]),
        "repairability": surv.Exponential.from_params([p["repair"]]),
        "repair_cost": p["fix"],
        "acquisition_cost": p["buy"],
    }


def drawn(units):
    """The line with each part's copies drawn in parallel: every copy of
    one part feeds every copy of the next."""
    stages = [
        [(name, i) for i in range(units[name])] for name in ("a", "b", "c")
    ]
    edges = [("s", node) for node in stages[0]]
    for before, after in zip(stages, stages[1:]):
        edges += list(itertools.product(before, after))
    edges += [(node, "t") for node in stages[-1]]
    return RepairableRBD(
        edges,
        {node: part(node[0]) for stage in stages for node in stage},
        downtime_cost_rate=DOWNTIME,
    )


def line():
    return RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
        {name: part(name) for name in PARTS},
        downtime_cost_rate=DOWNTIME,
    )


def present(rate, horizon=HORIZON):
    return -math.expm1(-rate * horizon) / rate if rate else horizon


def valued(units, rate, value_rate):
    """A design's net present value, from the diagram drawn out: what it
    earns at its long-run availability, less its total cost."""
    design = drawn(units)
    value = value_rate * present(rate) * design.mean_availability()
    return value - design.total_cost(HORIZON, discount_rate=rate)


@pytest.mark.parametrize("rate", [0.0, 2e-5])
@pytest.mark.parametrize("method", ["exact", "greedy"])
def test_with_no_value_it_is_the_total_costs_design(rate, method):
    kwargs = dict(discount_rate=rate, method=method)
    by_cost = line().allocate_redundancy(HORIZON, **kwargs)
    for objective in ("npv", "eac"):
        best = line().allocate_redundancy(
            HORIZON, objective=objective, **kwargs
        )
        assert best.units == by_cost.units
        assert best.total_cost == by_cost.total_cost
        assert best.acquisition_cost == by_cost.acquisition_cost
        assert best.cost_rate == by_cost.cost_rate
        assert best.availability == by_cost.availability
        assert best.npv == -by_cost.total_cost and best.value == 0.0
        assert best.eac == pytest.approx(
            by_cost.total_cost / present(rate), rel=1e-12
        )
    assert by_cost.objective == "cost" and by_cost.npv is None


@pytest.mark.parametrize("rate", [0.0, 1e-5, 1e-4])
@pytest.mark.parametrize("value_rate", [0.0, 150.0])
def test_against_every_design_drawn_out(rate, value_rate):
    counts = range(1, 5)
    values = {
        n: valued(dict(zip("abc", n)), rate, value_rate)
        for n in itertools.product(counts, repeat=3)
    }
    best_units = max(values, key=values.get)
    assert max(best_units) < 4  # the best is inside what was tried
    best = line().allocate_redundancy(
        HORIZON, objective="npv", discount_rate=rate, value_rate=value_rate
    )
    assert tuple(best.units[name] for name in "abc") == best_units
    assert best.npv == pytest.approx(values[best_units], rel=1e-9)
    assert best.npv == pytest.approx(
        best.value - best.acquisition_cost - best.running_cost, rel=1e-12
    )
    assert best.eac == pytest.approx(-best.npv / present(rate), rel=1e-12)
    by_eac = line().allocate_redundancy(
        HORIZON, objective="eac", discount_rate=rate, value_rate=value_rate
    )
    assert by_eac.units == best.units and by_eac.npv == best.npv
    assert "long-run" in best.basis and "expected_npv" in best.basis


@pytest.mark.parametrize("seed", range(8))
def test_the_searchs_bounds_hold_on_random_small_cases(seed):
    # The exact search stops at its bounds (a copy that cannot pay for
    # itself, the copies a design no worse can afford): with a value and
    # a discount rate it must still find the best of every design.
    rng = np.random.default_rng(seed)
    parts = {
        name: {
            "reliability": surv.Exponential.from_params(
                [float(10 ** rng.uniform(-3.5, -2.5))]
            ),
            "repairability": surv.Exponential.from_params(
                [float(10 ** rng.uniform(-1.5, -0.5))]
            ),
            "repair_cost": float(rng.uniform(100.0, 2000.0)),
            "acquisition_cost": float(rng.uniform(5000.0, 50000.0)),
        }
        for name in ("a", "b")
    }
    downtime = float(rng.uniform(0.0, 100.0))
    value_rate = float(rng.uniform(0.0, 400.0))
    rate = float(10 ** rng.uniform(-6.0, -4.0))

    def design(units):
        stages = [[(n, i) for i in range(units[n])] for n in ("a", "b")]
        edges = [("s", node) for node in stages[0]]
        edges += list(itertools.product(*stages))
        edges += [(node, "t") for node in stages[1]]
        return RepairableRBD(
            edges,
            {node: parts[node[0]] for stage in stages for node in stage},
            downtime_cost_rate=downtime,
        )

    values = {}
    for n in itertools.product(range(1, 7), repeat=2):
        drawn_out = design(dict(zip("ab", n)))
        values[n] = value_rate * present(
            rate
        ) * drawn_out.mean_availability() - drawn_out.total_cost(
            HORIZON, discount_rate=rate
        )
    best_units = max(values, key=values.get)
    assert max(best_units) < 6
    best = design({"a": 1, "b": 1}).allocate_redundancy(
        HORIZON, objective="npv", discount_rate=rate, value_rate=value_rate
    )
    found = (best.units[("a", 0)], best.units[("b", 0)])
    assert found == best_units
    assert best.npv == pytest.approx(values[best_units], rel=1e-9)


def test_the_best_design_changes_with_the_discount_rate():
    # Discounted hard, the downtime a copy saves in later years is worth
    # little now, and the copy is bought at the start.
    kwargs = dict(objective="npv", value_rate=150.0)
    plain = line().allocate_redundancy(HORIZON, **kwargs)
    hard = line().allocate_redundancy(HORIZON, discount_rate=1e-4, **kwargs)
    assert plain.units != hard.units
    assert all(hard.units[n] <= plain.units[n] for n in PARTS)
    assert sum(hard.units.values()) < sum(plain.units.values())


def test_a_value_buys_more_copies_than_the_cost_alone():
    by_cost = line().allocate_redundancy(HORIZON)
    worth = line().allocate_redundancy(
        HORIZON, objective="npv", value_rate=2000.0
    )
    assert sum(worth.units.values()) > sum(by_cost.units.values())
    assert worth.availability > by_cost.availability
    assert worth.value == pytest.approx(
        2000.0 * HORIZON * worth.availability, rel=1e-12
    )


def test_the_constraints_are_kept():
    kwargs = dict(objective="npv", discount_rate=1e-5, value_rate=150.0)
    free = line().allocate_redundancy(HORIZON, **kwargs)
    floor = 1.0 - 0.1 * (1.0 - free.availability)
    held = line().allocate_redundancy(
        HORIZON, min_availability=floor, **kwargs
    )
    assert held.availability >= floor and held.npv < free.npv
    capped = line().allocate_redundancy(HORIZON, max_units=1, **kwargs)
    assert set(capped.units.values()) == {1}
    assert capped.npv <= free.npv


def test_for_ever_at_a_discount_rate():
    rate = 1e-5
    best = line().allocate_redundancy(
        math.inf, objective="eac", discount_rate=rate, value_rate=150.0
    )
    assert best.eac == pytest.approx(-best.npv * rate, rel=1e-12)


def test_the_arguments_are_checked():
    allocate = line().allocate_redundancy
    with pytest.raises(ValueError, match="objective must be"):
        allocate(HORIZON, objective="profit")
    with pytest.raises(ValueError, match="value_rate needs an objective"):
        allocate(HORIZON, value_rate=10.0)
    with pytest.raises(ValueError, match="value_rate must be"):
        allocate(HORIZON, objective="npv", value_rate=-1.0)


def test_crews_keep_their_refusal():
    crewed = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": part("a"), "b": part("b")},
        downtime_cost_rate=DOWNTIME,
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="crew"):
        crewed.allocate_redundancy(HORIZON, objective="npv", value_rate=5.0)


def test_the_result_goes_through_json():
    best = line().allocate_redundancy(
        HORIZON, objective="npv", discount_rate=1e-5, value_rate=150.0
    )
    again = json.loads(json.dumps(best.to_dict()))
    assert again["objective"] == "npv" and again["npv"] == best.npv
    assert again["value_rate"] == 150.0 and again["basis"] == best.basis
