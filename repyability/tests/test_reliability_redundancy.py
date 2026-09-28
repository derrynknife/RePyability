"""Tests for reliability-redundancy allocation
(``NonRepairableRBD.allocate_reliability_redundancy``, issue #79).

The solver is checked against the best published solutions of the four
classic benchmarks (series, series-parallel, bridge and overspeed
protection systems), against plain redundancy allocation when the
component reliabilities are fixed, and against a closed form for one node.
"""

import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import BetaFactor, CCFGroup, NonRepairableRBD

MISSION = 1000.0


def fixed(reliability):
    return FixedEventProbability.from_params(1.0 - reliability)


def benchmark_uses(alpha, v, w, beta=1.5):
    # Volume v n^2, cost alpha (-T / ln r)^beta (n + e^(n/4)) and weight
    # w n e^(n/4): the resource model of the classic benchmarks.
    def make(i):
        def use(r, n):
            return {
                "volume": v[i] * n * n,
                "cost": alpha[i]
                * (-MISSION / math.log(r)) ** beta
                * (n + math.exp(n / 4)),
                "weight": w[i] * n * math.exp(n / 4),
            }

        return use

    return [make(i) for i in range(len(v))]


NODES = ["x1", "x2", "x3", "x4", "x5"]
SERIES = [(a, b) for a, b in zip(["s"] + NODES, NODES + ["t"])]
BRIDGE = [
    ("s", "x1"),
    ("s", "x3"),
    ("x1", "x2"),
    ("x3", "x4"),
    ("x1", "x5"),
    ("x5", "x4"),
    ("x3", "x5"),
    ("x5", "x2"),
    ("x2", "t"),
    ("x4", "t"),
]
SERIES_PARALLEL = [
    ("s", "x1"),
    ("x1", "x2"),
    ("x2", "t"),
    ("s", "x3"),
    ("s", "x4"),
    ("x3", "x5"),
    ("x4", "x5"),
    ("x5", "t"),
]
OVERSPEED = [(a, b) for a, b in zip(["s"] + NODES[:4], NODES[:4] + ["t"])]
ALPHA = [2.33e-5, 1.45e-5, 0.541e-5, 8.05e-5, 1.95e-5]
LIMITS = {"volume": 110, "cost": 175, "weight": 200}
BOUNDS = (0.5, 1 - 1e-6)


def solve(edges, nodes, uses, budget):
    rbd = NonRepairableRBD(edges, {n: fixed(0.9) for n in nodes})
    best = rbd.allocate_reliability_redundancy(
        dict(zip(nodes, uses)), budget=budget, bounds=BOUNDS, max_units=10
    )
    # Within the limits and bounds, and the reliability is the system's at
    # the solution.
    for resource, limit in budget.items():
        assert best.resources[resource] <= limit * (1 + 1e-9)
    for r in best.component_reliability.values():
        assert BOUNDS[0] <= r <= BOUNDS[1]
    probabilities = {
        n: 1 - (1 - best.component_reliability[n]) ** best.units[n]
        for n in nodes
    }
    assert best.reliability == pytest.approx(
        float(rbd.system_probability(probabilities)[0]), abs=1e-15
    )
    return best


# -- the classic benchmarks (Tillman, Hwang & Kuo, 1977; Hikita et al.) ----


def test_series_benchmark():
    best = solve(
        SERIES,
        NODES,
        benchmark_uses(ALPHA, [1, 2, 3, 4, 2], [7, 8, 8, 6, 9]),
        LIMITS,
    )
    assert round(best.reliability, 6) == 0.931682
    assert tuple(best.units.values()) == (3, 2, 2, 3, 3)
    assert list(best.component_reliability.values()) == pytest.approx(
        [0.779427, 0.871837, 0.902885, 0.711403, 0.787800], abs=1e-4
    )


def test_bridge_benchmark():
    best = solve(
        BRIDGE,
        NODES,
        benchmark_uses(ALPHA, [1, 2, 3, 4, 2], [7, 8, 8, 6, 9]),
        LIMITS,
    )
    assert round(best.reliability, 8) == 0.99988964
    assert tuple(best.units.values()) == (3, 3, 2, 4, 1)


def test_series_parallel_benchmark():
    best = solve(
        SERIES_PARALLEL,
        NODES,
        benchmark_uses(
            [2.5e-5, 1.45e-5, 0.541e-5, 0.541e-5, 2.1e-5],
            [2, 4, 5, 8, 4],
            [3.5, 4, 4, 3.5, 4.5],
        ),
        {"volume": 180, "cost": 175, "weight": 100},
    )
    assert round(best.reliability, 8) == 0.99997665
    assert tuple(best.units.values()) == (2, 2, 2, 2, 4)


def test_overspeed_benchmark():
    best = solve(
        OVERSPEED,
        NODES[:4],
        benchmark_uses(
            [1.0e-5, 2.3e-5, 0.3e-5, 2.3e-5], [1, 2, 3, 2], [6, 6, 8, 7]
        ),
        {"volume": 250, "cost": 400, "weight": 500},
    )
    # Quoted as 0.99995467 or 0.99995468; (5, 6, 4, 5) ties it to 1e-15.
    assert best.reliability == pytest.approx(0.9999546747, abs=1e-9)
    assert tuple(best.units.values()) in {(5, 5, 4, 6), (5, 6, 4, 5)}


# -- special cases ---------------------------------------------------------


def test_fixed_reliabilities_are_redundancy_allocation():
    # With each component's reliability fixed, only the copies are chosen:
    # the exact redundancy allocation.
    scales = {"a": 900, "b": 700, "c": 1200, "d": 500, "e": 800}
    rbd = NonRepairableRBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("a", "c"),
            ("b", "c"),
            ("a", "d"),
            ("c", "d"),
            ("b", "e"),
            ("d", "t"),
            ("e", "t"),
        ],
        {n: surv.Weibull.from_params([a, 1.5]) for n, a in scales.items()},
    )
    costs = {"a": 3.0, "b": 2.0, "c": 4.0, "d": 1.5, "e": 2.5}
    t = 400.0
    p = {n: float(rbd.reliabilities[n].sf(t)) for n in costs}
    redundancy = rbd.allocate_redundancy(costs, budget=22.0, t=t)
    both = rbd.allocate_reliability_redundancy(
        {n: (lambda c: lambda r, k: c * k)(c) for n, c in costs.items()},
        budget=22.0,
        bounds={n: (p[n], p[n]) for n in costs},
        t=t,
    )
    assert both.reliability == pytest.approx(redundancy.reliability, abs=1e-14)
    assert both.cost <= 22.0 + 1e-9


def test_copies_bounded_by_the_budget_alone():
    # No max_units: the budget bounds the copies (and uses that would
    # overflow for thousands of copies are never evaluated there).
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": fixed(0.9)})

    def use(r, n):
        return {"cost": n * math.exp(n / 4) * (1 + r), "volume": n * n}

    best = rbd.allocate_reliability_redundancy(
        {"a": use}, budget={"cost": 40, "volume": 50}, bounds=(0.5, 0.95)
    )
    assert best.resources["cost"] <= 40 + 1e-9
    assert best.units["a"] >= 2


def test_one_node_closed_form():
    # One node: for n copies the budget binds, 10 n + n g(r) = B with
    # g(r) = (-1/ln r)^1.5, so r = exp(-(B/n - 10)^(-2/3)); the best n is
    # found by trying each.
    budget = 70.0
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": fixed(0.9)})

    def cost(r, n):
        return n * (10 + (-1 / math.log(r)) ** 1.5)

    best = rbd.allocate_reliability_redundancy(
        {"a": cost}, budget=budget, bounds=(0.5, 0.999)
    )
    candidates = []
    for n in range(1, 7):
        r = math.exp(-((budget / n - 10) ** (-2 / 3)))
        r = min(max(r, 0.5), 0.999)
        if cost(r, n) <= budget * (1 + 1e-12):
            candidates.append((1 - (1 - r) ** n, n, r))
    reliability, n, r = max(candidates)
    assert best.units == {"a": n}
    assert best.component_reliability["a"] == pytest.approx(r, rel=1e-6)
    assert best.reliability == pytest.approx(reliability, rel=1e-9)


def test_other_nodes_are_evaluated_at_the_mission_time():
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": fixed(0.9), "b": surv.Weibull.from_params([2000, 2])},
    )

    def cost(r, n):
        return n * (10 + (-1 / math.log(r)) ** 1.5)

    best = rbd.allocate_reliability_redundancy(
        {"a": cost}, budget=50, bounds=(0.5, 0.999), t=MISSION
    )
    node = 1 - (1 - best.component_reliability["a"]) ** best.units["a"]
    assert best.reliability == pytest.approx(
        node * float(rbd.reliabilities["b"].sf(MISSION)), rel=1e-12
    )
    with pytest.raises(ValueError, match="mission time"):
        rbd.allocate_reliability_redundancy(
            {"a": cost}, budget=50, bounds=(0.5, 0.999)
        )


def test_answer_fits_the_budget_when_the_solver_does_not():
    # A cost that jumps from 10 to 100 a copy at r = 0.8: the continuous
    # solver sees no gradient in the cost and runs past the jump, outside
    # the budget; the answer falls back to what fits.
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": fixed(0.9)})
    best = rbd.allocate_reliability_redundancy(
        {"a": lambda r, n: n * (10.0 if r < 0.8 else 100.0)},
        budget=30,
        bounds=(0.5, 0.99),
    )
    assert best.cost <= 30
    assert best.component_reliability["a"] < 0.8


# -- validation ------------------------------------------------------------


def cost(r, n):
    return n * (10 + (-1 / math.log(r)) ** 1.5)


@pytest.mark.parametrize(
    "uses, kwargs, match",
    [
        ({}, {}, "at least one node"),
        ({"zz": cost}, {}, "not a component node"),
        ({"a": 5.0}, {}, "function of"),
        ({"a": cost}, {"bounds": (0.9, 0.5)}, "0 <= low <= high <= 1"),
        ({"a": cost}, {"bounds": (0.5, 1.5)}, "0 <= low <= high <= 1"),
        ({"a": cost}, {"bounds": 0.5}, "a pair"),
        ({"a": cost}, {"bounds": {"b": (0.5, 0.9)}}, "exactly the nodes"),
        ({"a": cost}, {"budget": 5}, "cannot afford"),
        ({"a": lambda r, n: 1.0 + r}, {}, "without limit"),
        ({"a": lambda r, n: -1.0 * n}, {}, "non-negative"),
        ({"a": lambda r, n: math.nan}, {}, "finite"),
        (
            {"a": cost, "b": lambda r, n: {"cost": n}},
            {},
            "number or a dict",
        ),
        ({"a": cost}, {"budget": {"volume": 50}}, "not use"),
        ({"a": cost}, {"max_units": 0}, "at least 1"),
    ],
)
def test_validation(uses, kwargs, match):
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": fixed(0.9), "b": fixed(0.8)},
    )
    arguments = {"budget": 50, "bounds": (0.5, 0.999)}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        rbd.allocate_reliability_redundancy(uses, **arguments)


def test_ccf_not_supported():
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": fixed(0.9), "b": fixed(0.9)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        rbd.allocate_reliability_redundancy(
            {"a": cost}, budget=50, bounds=(0.5, 0.999)
        )


def test_result_is_a_mapping():
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": fixed(0.9)})
    best = rbd.allocate_reliability_redundancy(
        {"a": cost}, budget=50, bounds=(0.5, 0.999)
    )
    assert set(best) == {
        "units",
        "component_reliability",
        "reliability",
        "cost",
        "resources",
    }
    assert best["units"] == best.units
    assert np.isclose(best.resources["cost"], best.cost)
