"""Tests for redundancy allocation (``NonRepairableRBD.allocate_redundancy``,
issue #40).

Expectations are hand-computed or come from an independent brute-force
search over every allocation, and the modelling identity at the heart of the
method -- ``n`` active copies of a node behave as ``1 - (1 - p) ** n`` -- is
checked against an RBD with the copies drawn out explicitly in parallel.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import BetaFactor, CCFGroup, NonRepairableRBD
from repyability.rbd import redundancy_allocation


def fixed(reliability):
    return FixedEventProbability.from_params(1.0 - reliability)


@pytest.fixture
def series_ab():
    # a: 90% reliable, b: 80% reliable, in series.
    return NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": fixed(0.9), "b": fixed(0.8)},
    )


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
    # A non-series-parallel structure with time-varying (Weibull) nodes.
    scales = {"a": 900, "b": 700, "c": 1200, "d": 500, "e": 800}
    return NonRepairableRBD(
        BRIDGE_EDGES,
        {n: surv.Weibull.from_params([a, 1.5]) for n, a in scales.items()},
    )


def reliability_of(rbd, costs, units, t):
    # Independent of the implementation: substitute 1 - (1 - p) ** n and
    # evaluate the exact system reliability directly.
    base = rbd._base_node_probabilities(np.atleast_1d(t), set(), set())
    probs = dict(base)
    for node, n in zip(costs, units):
        probs[node] = 1.0 - (1.0 - np.asarray(base[node])) ** n
    return float(np.ravel(rbd.system_probability(probs))[0])


# -- hand-computed optima (series, one cost unit each) ---------------------
#
# Reliabilities: (1,1) 0.72; (2,1) 0.792; (1,2) 0.864; (3,1) 0.7992;
# (2,2) 0.9504; (1,3) 0.8928.


@pytest.mark.parametrize("method", ["exact", "greedy"])
@pytest.mark.parametrize(
    "budget, units, reliability",
    [
        (2, {"a": 1, "b": 1}, 0.72),
        (3, {"a": 1, "b": 2}, 0.864),
        (4, {"a": 2, "b": 2}, 0.9504),
    ],
)
def test_budget_form_hand_computed(
    series_ab, method, budget, units, reliability
):
    result = series_ab.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, budget=budget, method=method
    )
    assert result.units == units
    assert result.reliability == pytest.approx(reliability)
    assert result.cost == pytest.approx(sum(units.values()))
    assert result.method == method


@pytest.mark.parametrize("method", ["exact", "greedy"])
@pytest.mark.parametrize(
    "target, units",
    [
        (0.70, {"a": 1, "b": 1}),
        (0.85, {"a": 1, "b": 2}),
        (0.90, {"a": 2, "b": 2}),
    ],
)
def test_target_form_hand_computed(series_ab, method, target, units):
    result = series_ab.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, target=target, method=method
    )
    assert result.units == units
    assert result.reliability >= target


# -- the modelling identity ------------------------------------------------


def test_copies_equal_explicit_parallel_units(bridge):
    # The allocation's reliability must equal an RBD where every chosen copy
    # is drawn out as a separate node wired in parallel (same predecessors
    # and successors as the original).
    result = bridge.allocate_redundancy(BRIDGE_COSTS, budget=22.0, t=MISSION)

    def copies(node):
        if node not in result.units:
            return [node]
        return [f"{node}{i}" for i in range(result.units[node])]

    edges = [
        (a, b) for u, v in BRIDGE_EDGES for a in copies(u) for b in copies(v)
    ]
    models = {
        copy: bridge.reliabilities[node]
        for node in result.units
        for copy in copies(node)
    }
    expanded = NonRepairableRBD(edges, models)
    assert float(expanded.sf(MISSION)) == pytest.approx(result.reliability)


# -- exact is optimal: independent brute force -----------------------------


def test_exact_budget_matches_brute_force(bridge):
    budget = 22.0
    spare = budget - sum(BRIDGE_COSTS.values())
    ranges = [range(1, 2 + int(spare // c)) for c in BRIDGE_COSTS.values()]
    best = max(
        (
            reliability_of(bridge, BRIDGE_COSTS, units, MISSION)
            for units in itertools.product(*ranges)
            if sum(u * c for u, c in zip(units, BRIDGE_COSTS.values()))
            <= budget + 1e-9
        )
    )
    result = bridge.allocate_redundancy(BRIDGE_COSTS, budget=budget, t=MISSION)
    assert result.reliability == pytest.approx(best, abs=1e-12)
    assert result.cost <= budget


def test_exact_target_matches_brute_force(bridge):
    target = 0.97
    cheapest = min(
        sum(u * c for u, c in zip(units, BRIDGE_COSTS.values()))
        for units in itertools.product(*(range(1, 6) for _ in BRIDGE_COSTS))
        if reliability_of(bridge, BRIDGE_COSTS, units, MISSION) >= target
    )
    result = bridge.allocate_redundancy(BRIDGE_COSTS, target=target, t=MISSION)
    assert result.cost == pytest.approx(cheapest)
    assert result.reliability >= target


def test_greedy_is_feasible_but_not_always_optimal(bridge):
    exact = bridge.allocate_redundancy(BRIDGE_COSTS, budget=22.0, t=MISSION)
    greedy = bridge.allocate_redundancy(
        BRIDGE_COSTS, budget=22.0, t=MISSION, method="greedy"
    )
    assert greedy.cost <= 22.0
    assert greedy.reliability <= exact.reliability
    # On this bridge the heuristic genuinely falls short of the optimum,
    # which is why "exact" is the default.
    assert greedy.reliability < exact.reliability - 1e-6


def test_greedy_target_is_feasible_and_never_cheaper_than_exact(bridge):
    exact = bridge.allocate_redundancy(BRIDGE_COSTS, target=0.97, t=MISSION)
    greedy = bridge.allocate_redundancy(
        BRIDGE_COSTS, target=0.97, t=MISSION, method="greedy"
    )
    assert greedy.reliability >= 0.97
    assert greedy.cost >= exact.cost - 1e-9


# -- max_units and the search guard ----------------------------------------


def test_max_units_caps_copies(series_ab):
    capped = series_ab.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, budget=10, max_units=2
    )
    assert max(capped.units.values()) <= 2
    per_node = series_ab.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, budget=10, max_units={"b": 1}
    )
    assert per_node.units["b"] == 1


def test_unreachable_target_within_max_units(series_ab):
    with pytest.raises(ValueError, match="unreachable.*max_units"):
        series_ab.allocate_redundancy(
            {"a": 1.0, "b": 1.0}, target=0.99, max_units=2
        )


def test_unreachable_target_even_unlimited():
    # c is not costed and caps the system at 50%.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "c"), ("c", "t")],
        {"a": fixed(0.9), "c": fixed(0.5)},
    )
    with pytest.raises(ValueError, match="unreachable.*unlimited"):
        rbd.allocate_redundancy({"a": 1.0}, target=0.6)


def test_oversized_exact_search_fails_fast(series_ab, monkeypatch):
    monkeypatch.setattr(redundancy_allocation, "EXACT_SEARCH_LIMIT", 5)
    with pytest.raises(ValueError, match="greedy"):
        series_ab.allocate_redundancy({"a": 1.0, "b": 1.0}, budget=30)
    # The heuristic still answers.
    result = series_ab.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, budget=30, method="greedy"
    )
    assert result.cost <= 30


def test_no_budget_is_spent_on_useless_copies():
    # With a perfectly reliable node p, the only allocation that exhausts a
    # budget of 6 (a: cost 2, p: cost 1) is a=2, p=2 -- but p's second copy
    # adds nothing. The reported allocation must drop it: a=2, p=1, cost 5.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "p"), ("p", "t")],
        {"a": fixed(0.8), "p": fixed(1.0)},
    )
    result = rbd.allocate_redundancy({"a": 2.0, "p": 1.0}, budget=6)
    assert result.units == {"a": 2, "p": 1}
    assert result.cost == pytest.approx(5.0)
    assert result.reliability == pytest.approx(0.96)


# -- validation ------------------------------------------------------------


def test_exactly_one_objective(series_ab):
    with pytest.raises(ValueError, match="exactly one"):
        series_ab.allocate_redundancy({"a": 1.0})
    with pytest.raises(ValueError, match="exactly one"):
        series_ab.allocate_redundancy({"a": 1.0}, budget=3, target=0.9)


@pytest.mark.parametrize(
    "costs, match",
    [
        ({}, "at least one"),
        ({"zzz": 1.0}, "not a component"),
        ({"s": 1.0}, "not a component"),
        ({"a": 0.0}, "finite and positive"),
        ({"a": -1.0}, "finite and positive"),
        ({"a": math.inf}, "finite and positive"),
    ],
)
def test_bad_costs_rejected(series_ab, costs, match):
    with pytest.raises(ValueError, match=match):
        series_ab.allocate_redundancy(costs, budget=10)


def test_budget_must_afford_one_of_each(series_ab):
    with pytest.raises(ValueError, match="at least 2"):
        series_ab.allocate_redundancy({"a": 1.0, "b": 1.0}, budget=1.5)


def test_bad_target_and_method_rejected(series_ab):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        series_ab.allocate_redundancy({"a": 1.0}, target=1.0)
    with pytest.raises(ValueError, match="method"):
        series_ab.allocate_redundancy({"a": 1.0}, budget=3, method="milp")
    with pytest.raises(ValueError, match="max_units"):
        series_ab.allocate_redundancy({"a": 1.0}, budget=3, max_units=0)
    with pytest.raises(ValueError, match="not in costs"):
        series_ab.allocate_redundancy({"a": 1.0}, budget=3, max_units={"b": 2})


def test_time_varying_rbd_needs_mission_time(bridge):
    with pytest.raises(ValueError, match="mission time"):
        bridge.allocate_redundancy(BRIDGE_COSTS, budget=22.0)


def test_ccf_rbd_not_supported():
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": fixed(0.9), "b": fixed(0.9)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        rbd.allocate_redundancy({"a": 1.0}, budget=3)


def test_result_is_a_mapping(series_ab):
    result = series_ab.allocate_redundancy({"a": 1.0, "b": 1.0}, budget=3)
    assert result["units"] == result.units
    assert set(result) == {"units", "reliability", "cost", "method"}
