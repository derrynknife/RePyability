"""Tests for the cost-reliability trade-off of redundancy allocation
(``NonRepairableRBD.redundancy_front``, issue #80).

The front is checked against the non-dominated set of every allocation,
found by brute force, on a series system (from the dynamic program) and on
a bridge network (by enumeration), and against ``allocate_redundancy`` at
every point.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import ComponentOption, NonRepairableRBD
from repyability.rbd import redundancy_allocation


def fixed(reliability):
    return FixedEventProbability.from_params(1.0 - reliability)


def series(reliabilities):
    names = [f"n{i}" for i in range(len(reliabilities))]
    edges = [("s", names[0])] + list(zip(names, names[1:]))
    rbd = NonRepairableRBD(
        edges + [(names[-1], "t")],
        {n: fixed(p) for n, p in zip(names, reliabilities)},
    )
    return names, rbd


BRIDGE_EDGES = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("d", "t"),
    ("e", "t"),
]
BRIDGE_COSTS = {"a": 3.0, "b": 2.0, "c": 4.0, "d": 1.5, "e": 2.5}
MISSION = 400.0


@pytest.fixture
def bridge():
    scales = {"a": 900, "b": 700, "c": 1200, "d": 500, "e": 800}
    return NonRepairableRBD(
        BRIDGE_EDGES,
        {n: surv.Weibull.from_params([a, 1.5]) for n, a in scales.items()},
    )


def naive_front(points):
    """The non-dominated (use, reliability) pairs, one per duplicate."""
    unique = set(points)
    return sorted(
        (use, r)
        for use, r in unique
        if not any(
            all(a <= b for a, b in zip(u, use))
            and s >= r
            and (u, s) != (use, r)
            for u, s in unique
        )
    )


def as_points(front, resources):
    return sorted(
        (tuple(d.resources[r] for r in resources), d.reliability)
        for d in front
    )


def assert_same_front(front, expected, resources):
    got = as_points(front, resources)
    assert len(got) == len(expected)
    for (use, r), (e_use, e_r) in zip(got, expected):
        assert use == pytest.approx(e_use)
        assert r == pytest.approx(e_r, abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_series_front_matches_brute_force(seed):
    # From the dynamic program, with one resource and with two.
    rng = np.random.default_rng(seed)
    reliabilities = rng.uniform(0.5, 0.95, 3)
    names, rbd = series(reliabilities)
    cost = dict(zip(names, map(float, rng.integers(1, 4, 3))))
    weight = dict(zip(names, map(float, rng.integers(1, 4, 3))))
    every = [
        (
            tuple(
                float(sum(u * c[n] for u, n in zip(units, names)))
                for c in (cost, weight)
            ),
            math.prod(1 - (1 - p) ** u for p, u in zip(reliabilities, units)),
        )
        for units in itertools.product(range(1, 5), repeat=3)
    ]
    limit = sum(cost.values()) + 5.0
    front = rbd.redundancy_front(cost, budget=limit, max_units=4)
    expected = naive_front(
        [((use[0],), r) for use, r in every if use[0] <= limit + 1e-9]
    )
    assert_same_front(front, expected, ["cost"])
    # One resource: the reliability rises with the cost.
    assert [d.cost for d in front] == sorted(d.cost for d in front)
    assert all(a.reliability < b.reliability for a, b in zip(front, front[1:]))
    both = {n: {"cost": cost[n], "weight": weight[n]} for n in names}
    front = rbd.redundancy_front(both, max_units=4)
    assert_same_front(front, naive_front(every), ["cost", "weight"])


def test_bridge_front_matches_brute_force(bridge):
    # Not a series of the costed nodes: every combination is evaluated.
    base = bridge._base_node_probabilities(
        np.atleast_1d(MISSION), set(), set()
    )
    every = []
    for units in itertools.product(range(1, 4), repeat=5):
        probabilities = dict(base)
        for node, n in zip(BRIDGE_COSTS, units):
            probabilities[node] = 1.0 - (1.0 - np.asarray(base[node])) ** n
        reliability = float(
            np.ravel(bridge.system_probability(probabilities))[0]
        )
        cost = sum(n * c for n, c in zip(units, BRIDGE_COSTS.values()))
        if cost <= 24.0 + 1e-9:
            every.append(((float(cost),), reliability))
    front = bridge.redundancy_front(
        BRIDGE_COSTS, budget=24.0, t=MISSION, max_units=3
    )
    assert_same_front(front, naive_front(every), ["cost"])


def test_every_point_is_an_optimum(bridge):
    # The best design within each point's cost is that point, and so is the
    # cheapest design reaching its reliability.
    front = bridge.redundancy_front(BRIDGE_COSTS, budget=22.0, t=MISSION)
    for point in front:
        best = bridge.allocate_redundancy(
            BRIDGE_COSTS, budget=point.cost, t=MISSION
        )
        assert best.reliability == pytest.approx(point.reliability, abs=1e-12)
        cheapest = bridge.allocate_redundancy(
            BRIDGE_COSTS, target=point.reliability, t=MISSION
        )
        assert cheapest.cost == pytest.approx(point.cost)


def test_front_with_types_and_strategies():
    # Options with mixing, a node that needs two copies, and a choice of
    # strategy: every point is the budget optimum at its cost.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": surv.Exponential.from_params([1 / 2000]),
            "b": surv.Weibull.from_params([1500, 1.5]),
        },
    )
    costs = {
        "a": 1.0,
        "b": [
            ComponentOption(
                "cheap", surv.Weibull.from_params([1500, 1.5]), 1.0
            ),
            ComponentOption(
                "good", surv.Weibull.from_params([3000, 2.0]), 2.5
            ),
        ],
    }
    kwargs = dict(t=1000.0, required={"a": 2}, strategy={"a": "choose"})
    front = rbd.redundancy_front(costs, budget=9.0, **kwargs)
    assert front
    for point in front:
        best = rbd.allocate_redundancy(costs, budget=point.cost, **kwargs)
        assert best.reliability == pytest.approx(point.reliability, abs=1e-12)
        assert point.units["a"] >= 2
        assert sum(point.mix["b"].values()) == point.units["b"]


def test_caps_alone_bound_the_front():
    names, rbd = series([0.9, 0.8])
    front = rbd.redundancy_front({"n0": 1.0, "n1": 1.0}, max_units=2)
    assert [(d.cost, d.units) for d in front] == [
        (2.0, {"n0": 1, "n1": 1}),
        (3.0, {"n0": 1, "n1": 2}),
        (4.0, {"n0": 2, "n1": 2}),
    ]


def test_front_needs_bounds(bridge, monkeypatch):
    names, rbd = series([0.9, 0.8])
    with pytest.raises(ValueError, match="without limit"):
        rbd.redundancy_front({"n0": 1.0, "n1": 1.0})
    monkeypatch.setattr(redundancy_allocation, "EXACT_SEARCH_LIMIT", 5)
    with pytest.raises(ValueError, match="examined more than 5"):
        bridge.redundancy_front(BRIDGE_COSTS, budget=24.0, t=MISSION)


def test_series_front_does_not_enumerate(monkeypatch):
    # The dynamic program, not enumeration: no limit on combinations.
    monkeypatch.setattr(redundancy_allocation, "EXACT_SEARCH_LIMIT", 1)
    names, rbd = series([0.9, 0.8, 0.7])
    front = rbd.redundancy_front(
        {"n0": 1.0, "n1": 1.0, "n2": 1.0}, budget=9, max_units=4
    )
    assert front[0].cost == 3.0
    assert front[-1].cost == 9.0


def test_bridge_front_with_types_and_two_resources():
    # A light, costly type and a heavy, cheap one on a bridge node: every
    # combination within both limits, against brute force.
    rbd = NonRepairableRBD(
        BRIDGE_EDGES,
        {n: fixed(p) for n, p in zip("abcde", (0.8, 0.7, 0.9, 0.6, 0.75))},
    )
    d_types = [
        ComponentOption("light", 0.6, {"cost": 3.0, "weight": 1.0}),
        ComponentOption("heavy", 0.6, {"cost": 1.0, "weight": 3.0}),
    ]
    costs = {
        "a": {"cost": 2.0, "weight": 2.0},
        "b": {"cost": 1.0, "weight": 1.0},
        "d": d_types,
    }
    limits = {"cost": 9.0, "weight": 9.0}
    base = rbd._base_node_probabilities(np.atleast_1d(1.0), set(), set())
    every = []
    for a, b in itertools.product(range(1, 4), repeat=2):
        for d in itertools.product(range(4), repeat=2):
            if not 1 <= sum(d) <= 3:
                continue
            cost = 2 * a + b + 3 * d[0] + d[1]
            weight = 2 * a + b + d[0] + 3 * d[1]
            if cost > 9 or weight > 9:
                continue
            probabilities = dict(base)
            probabilities["a"] = np.array([1 - 0.2**a])
            probabilities["b"] = np.array([1 - 0.3**b])
            probabilities["d"] = np.array([1 - 0.4 ** sum(d)])
            r = float(np.ravel(rbd.system_probability(probabilities))[0])
            every.append(((float(cost), float(weight)), r))
    front = rbd.redundancy_front(costs, budget=limits, max_units=3)
    assert_same_front(front, naive_front(every), ["cost", "weight"])
