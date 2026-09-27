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


def test_oversized_exact_search_fails_fast(bridge, monkeypatch):
    # The bridge is not a series of its costed nodes, so its exact answer
    # comes from the exhaustive search, which gives up with guidance.
    monkeypatch.setattr(redundancy_allocation, "EXACT_SEARCH_LIMIT", 5)
    with pytest.raises(ValueError, match="greedy"):
        bridge.allocate_redundancy(BRIDGE_COSTS, budget=30.0, t=MISSION)
    # The heuristic still answers.
    result = bridge.allocate_redundancy(
        BRIDGE_COSTS, budget=30.0, t=MISSION, method="greedy"
    )
    assert result.cost <= 30


def test_oversized_dynamic_program_fails_fast(series_ab, monkeypatch):
    monkeypatch.setattr(redundancy_allocation, "SERIES_STATE_LIMIT", 3)
    with pytest.raises(ValueError, match="greedy"):
        series_ab.allocate_redundancy({"a": 1.0, "b": 1.0}, budget=30)


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


def test_budget_or_target_required(series_ab):
    with pytest.raises(ValueError, match="a budget, a target, or both"):
        series_ab.allocate_redundancy({"a": 1.0})


@pytest.mark.parametrize("method", ["exact", "greedy"])
def test_target_within_a_budget(series_ab, method):
    # The cheapest design meeting 90% is (2, 2) at a cost of 4: a budget of
    # 4 allows it, a budget of 3.5 does not.
    result = series_ab.allocate_redundancy(
        {"a": 1.0, "b": 1.0}, target=0.9, budget=4, method=method
    )
    assert result.units == {"a": 2, "b": 2}
    with pytest.raises(ValueError, match="within the budget"):
        series_ab.allocate_redundancy(
            {"a": 1.0, "b": 1.0}, target=0.9, budget=3.5, method=method
        )


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
    assert set(result) == {
        "units",
        "reliability",
        "cost",
        "method",
        "resources",
    }
    assert result.resources == {"cost": 3.0}


# -- several resources (issue #76) -----------------------------------------

BRIDGE_WEIGHTS = {"a": 4.0, "b": 1.0, "c": 3.0, "d": 5.0, "e": 2.0}
BRIDGE_AMOUNTS = [list(BRIDGE_COSTS.values()), list(BRIDGE_WEIGHTS.values())]


def two_resources(costs, weights):
    return {n: {"cost": costs[n], "weight": weights[n]} for n in costs}


def use_of(units, amounts):
    return math.fsum(u * a for u, a in zip(units, amounts))


def designs(amounts, limits):
    # Every allocation within the limits (one list of amounts per
    # resource, math.inf for a resource that is not limited): the
    # independent brute force the exact answers are checked against.
    ranges = []
    for i in range(len(amounts[0])):
        most = min(
            math.floor((limit - sum(amounts[r])) / amounts[r][i] + 1e-9)
            for r, limit in enumerate(limits)
            if math.isfinite(limit) and amounts[r][i] > 0.0
        )
        ranges.append(range(1, most + 2))
    for units in itertools.product(*ranges):
        if all(
            use_of(units, amounts[r]) <= limit + 1e-9
            for r, limit in enumerate(limits)
        ):
            yield units


@pytest.mark.parametrize("method", ["exact", "greedy"])
def test_two_resources_budget_matches_brute_force(bridge, method):
    limits = (22.0, 26.0)
    best = max(
        reliability_of(bridge, BRIDGE_COSTS, units, MISSION)
        for units in designs(BRIDGE_AMOUNTS, limits)
    )
    result = bridge.allocate_redundancy(
        two_resources(BRIDGE_COSTS, BRIDGE_WEIGHTS),
        budget={"cost": limits[0], "weight": limits[1]},
        t=MISSION,
        method=method,
    )
    units = tuple(result.units.values())
    assert result.resources == {
        "cost": pytest.approx(use_of(units, BRIDGE_AMOUNTS[0])),
        "weight": pytest.approx(use_of(units, BRIDGE_AMOUNTS[1])),
    }
    assert result.cost == result.resources["cost"]
    assert result.resources["cost"] <= limits[0]
    assert result.resources["weight"] <= limits[1]
    assert result.reliability == pytest.approx(
        reliability_of(bridge, BRIDGE_COSTS, units, MISSION), abs=1e-15
    )
    if method == "exact":
        assert result.reliability == pytest.approx(best, abs=1e-12)
    else:
        assert result.reliability <= best + 1e-12


@pytest.mark.parametrize("method", ["exact", "greedy"])
def test_one_limit_of_two_resources(bridge, method):
    # Only the weight is limited: the cost is recorded but free.
    costs = two_resources(BRIDGE_COSTS, BRIDGE_WEIGHTS)
    best = max(
        reliability_of(bridge, BRIDGE_COSTS, units, MISSION)
        for units in designs(BRIDGE_AMOUNTS, (math.inf, 24.0))
    )
    result = bridge.allocate_redundancy(
        costs, budget={"weight": 24.0}, t=MISSION, method=method
    )
    assert result.resources["weight"] <= 24.0
    if method == "exact":
        assert result.reliability == pytest.approx(best, abs=1e-12)
    else:
        assert result.reliability <= best + 1e-12


def test_two_resources_target_matches_brute_force(bridge):
    costs = two_resources(BRIDGE_COSTS, BRIDGE_WEIGHTS)
    # The cheapest design meeting the target with a weight of at most 21
    # (the cheapest without the limit weighs 23).
    cheapest = min(
        use_of(units, BRIDGE_AMOUNTS[0])
        for units in designs(BRIDGE_AMOUNTS, (math.inf, 21.0))
        if reliability_of(bridge, BRIDGE_COSTS, units, MISSION) >= 0.92
    )
    result = bridge.allocate_redundancy(
        costs, target=0.92, budget={"weight": 21.0}, t=MISSION
    )
    assert result.cost == pytest.approx(cheapest)
    assert result.resources["weight"] <= 21.0
    assert result.reliability >= 0.92
    # minimise names the resource to minimise: the lightest design meeting
    # the target. Every design lighter than the lightest one found within
    # a weight of 30 is among those enumerated, so this is exhaustive.
    meeting = [
        units
        for units in designs(BRIDGE_AMOUNTS, (math.inf, 30.0))
        if reliability_of(bridge, BRIDGE_COSTS, units, MISSION) >= 0.92
    ]
    assert meeting
    lightest = min(use_of(units, BRIDGE_AMOUNTS[1]) for units in meeting)
    result = bridge.allocate_redundancy(
        costs, target=0.92, minimise="weight", t=MISSION
    )
    assert result.cost == pytest.approx(lightest)
    assert result.cost == result.resources["weight"]
    assert result.reliability >= 0.92


def series_rbd(reliabilities):
    names = [f"n{i}" for i in range(len(reliabilities))]
    edges = [("s", names[0])] + list(zip(names, names[1:]))
    rbd = NonRepairableRBD(
        edges + [(names[-1], "t")],
        {n: fixed(p) for n, p in zip(names, reliabilities)},
    )
    return names, rbd


@pytest.mark.parametrize("seed", range(8))
def test_dynamic_program_matches_brute_force(seed):
    # A series of costed nodes is solved by dynamic programming: check it
    # against every allocation, with one resource and with two, in both
    # forms.
    rng = np.random.default_rng(seed)
    reliabilities = rng.uniform(0.5, 0.95, 4)
    names, rbd = series_rbd(reliabilities)
    cost = rng.integers(1, 5, 4).astype(float)
    weight = rng.uniform(0.5, 3.0, 4)
    costs = two_resources(dict(zip(names, cost)), dict(zip(names, weight)))
    limits = (cost.sum() + 6.0, weight.sum() + 4.0)
    every = list(itertools.product(range(1, 6), repeat=4))

    def reliability(units):
        return math.prod(
            1.0 - (1.0 - p) ** n for p, n in zip(reliabilities, units)
        )

    def fits(units, cost_limit, weight_limit):
        return (
            use_of(units, cost) <= cost_limit + 1e-9
            and use_of(units, weight) <= weight_limit + 1e-9
        )

    result = rbd.allocate_redundancy(
        dict(zip(names, cost)), budget=limits[0], max_units=5
    )
    assert result.reliability == pytest.approx(
        max(reliability(u) for u in every if fits(u, limits[0], math.inf)),
        abs=1e-12,
    )
    best = max(reliability(u) for u in every if fits(u, *limits))
    result = rbd.allocate_redundancy(
        costs,
        budget={"cost": limits[0], "weight": limits[1]},
        max_units=5,
    )
    assert result.reliability == pytest.approx(best, abs=1e-12)
    assert fits(tuple(result.units.values()), *limits)
    target = 0.9 * best
    result = rbd.allocate_redundancy(
        costs, target=target, budget={"weight": limits[1]}, max_units=5
    )
    assert result.reliability >= target
    assert result.cost == pytest.approx(
        min(
            use_of(u, cost)
            for u in every
            if fits(u, math.inf, limits[1]) and reliability(u) >= target
        )
    )


def test_dynamic_program_solves_large_series_problems():
    # Fourteen subsystems with cost and weight limits: far beyond the
    # exhaustive search, but quick for the dynamic program.
    rng = np.random.default_rng(1)
    names, rbd = series_rbd(rng.uniform(0.7, 0.95, 14))
    costs = {
        n: {"cost": float(c), "weight": float(w)}
        for n, c, w in zip(
            names, rng.integers(1, 6, 14), rng.integers(3, 10, 14)
        )
    }
    budget = {"cost": 60.0, "weight": 150.0}
    exact = rbd.allocate_redundancy(costs, budget=budget)
    greedy = rbd.allocate_redundancy(costs, budget=budget, method="greedy")
    assert exact.resources["cost"] <= 60.0
    assert exact.resources["weight"] <= 150.0
    assert exact.reliability >= greedy.reliability


@pytest.mark.parametrize(
    "costs, kwargs, match",
    [
        ({"a": 1.0, "b": {"cost": 1.0}}, {"budget": 3}, "not a mixture"),
        (
            {"a": {"cost": 1.0, "weight": 1.0}, "b": {"cost": 1.0}},
            {"budget": {"cost": 3}},
            "same resources",
        ),
        ({"a": {}}, {"budget": 3}, "at least one resource"),
        ({"a": {"cost": -1.0}}, {"budget": {"cost": 3}}, "non-negative"),
        (
            {"a": {"cost": math.nan}},
            {"budget": {"cost": 3}},
            "non-negative",
        ),
        (
            {"a": {"cost": 1.0, "weight": 1.0}},
            {"budget": 3},
            "dict of limits",
        ),
        ({"a": {"cost": 1.0}}, {"budget": {"volume": 3}}, "not use"),
        ({"a": {"cost": 1.0}}, {"budget": {}}, "at least one resource"),
        (
            {"a": {"cost": 1.0, "weight": 1.0}},
            {"budget": {"cost": 3, "weight": 0.5}},
            r"budget\['weight'\] must be finite and at least 1",
        ),
        (
            {"a": {"weight": 1.0, "volume": 1.0}},
            {"target": 0.95},
            "minimise must name",
        ),
        ({"a": 1.0}, {"budget": 3, "minimise": "cost"}, "give a target"),
        ({"a": 1.0}, {"target": 0.95, "minimise": "mass"}, "one of the"),
        (
            {"a": {"cost": 1.0, "weight": 0.0}},
            {"budget": {"weight": 3}},
            "without limit",
        ),
    ],
)
def test_resource_validation(series_ab, costs, kwargs, match):
    with pytest.raises(ValueError, match=match):
        series_ab.allocate_redundancy(costs, **kwargs)


def test_zero_use_is_fine_when_bounded(series_ab):
    # b uses no weight, but its cost is limited, so the search is bounded.
    result = series_ab.allocate_redundancy(
        {"a": {"cost": 1.0, "weight": 2.0}, "b": {"cost": 1.0, "weight": 0.0}},
        budget={"cost": 4, "weight": 2},
    )
    assert result.units == {"a": 1, "b": 3}
    assert result.resources == {"cost": 4.0, "weight": 2.0}


def test_resources_named_without_cost(series_ab):
    # With a single resource of any name, it is the one minimised.
    result = series_ab.allocate_redundancy(
        {"a": {"mass": 1.0}, "b": {"mass": 1.0}}, target=0.9
    )
    assert result.units == {"a": 2, "b": 2}
    assert result.cost == 4.0
    assert result.resources == {"mass": 4.0}


@pytest.mark.parametrize("method", ["exact", "greedy"])
def test_copies_that_use_nothing(series_ab, method):
    # Copies of b cost nothing, up to its max_units: both methods take them
    # all (and the greedy heuristic takes them first).
    costs = {"a": {"cost": 1.0}, "b": {"cost": 0.0}}
    result = series_ab.allocate_redundancy(
        costs, budget={"cost": 3}, max_units={"b": 2}, method=method
    )
    assert result.units == {"a": 3, "b": 2}
    # Target form: (2, 2) and (2, 3) both cost 2; the tie goes to the more
    # reliable.
    result = series_ab.allocate_redundancy(
        costs, target=0.9, max_units={"b": 3}, method=method
    )
    assert result.units == {"a": 2, "b": 3}
    assert result.cost == 2.0


def dominates(t, s):
    return all(a <= b for a, b in zip(t[0], s[0])) and t[1] >= s[1]


def naive_front(states):
    # One representative of every (use, value) that nothing else beats.
    unique = {(s[0], s[1]): s for s in states}
    return {
        key
        for key, s in unique.items()
        if not any(dominates(t, s) and t != key for t in unique)
    }


@pytest.mark.parametrize("m", [1, 2, 3])
def test_pareto_front_matches_naive(m):
    # Value rising with use gives a large front; integer amounts and values
    # in exact steps of 1/256 give near-ties, exact ties and duplicates.
    rng = np.random.default_rng(m)
    high = {1: 30, 2: 12, 3: 6}[m]
    states = []
    for i in range(400):
        use = tuple(rng.integers(0, high, m).astype(float))
        states.append((use, (2 * sum(use) + rng.integers(3)) / 256, i))
    kept = redundancy_allocation._pareto_front(list(states))
    assert len(kept) > 10
    assert len({(s[0], s[1]) for s in kept}) == len(kept)
    assert {(s[0], s[1]) for s in kept} == naive_front(states)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_series_front_is_the_front_of_every_design(m, seed):
    # The dynamic program's final front is exactly the non-dominated set of
    # all complete designs within the budget (and the bound).
    rng = np.random.default_rng(10 * m + seed)
    choices = []
    for q in rng.uniform(0.05, 0.5, 4):
        per_copy = rng.integers(1, 4, m).astype(float)
        choices.append(
            [(tuple(n * per_copy), math.log1p(-(q**n))) for n in range(1, 5)]
        )
    budget = tuple(
        sum(node[0][0][r] for node in choices) + rng.uniform(3.0, 8.0)
        for r in range(m)
    )
    bound = budget[0] - 1.0
    designs_ = []
    for picks in itertools.product(range(4), repeat=4):
        use, value = (0.0,) * m, 0.0
        for node, pick in zip(choices, picks):
            use = tuple(u + e for u, e in zip(use, node[pick][0]))
            value = value + node[pick][1]
        if all(u <= b for u, b in zip(use, budget)) and use[0] <= bound:
            designs_.append((use, value, picks))
    front = redundancy_allocation.series_front(
        choices, budget=budget, bound=bound
    )
    assert {(u, v) for u, v, _ in front} == naive_front(designs_)
    for use, value, picks in front:
        assert (use, value, picks) in designs_
    assert [f[0][0] for f in front] == sorted(f[0][0] for f in front)


@pytest.mark.parametrize("seed", range(4))
def test_three_resources_match_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    reliabilities = rng.uniform(0.5, 0.95, 3)
    names, rbd = series_rbd(reliabilities)
    resources = ("cost", "weight", "volume")
    amounts = rng.uniform(0.5, 3.0, (3, 3))  # [resource][node]
    costs = {
        n: {r: amounts[k][i] for k, r in enumerate(resources)}
        for i, n in enumerate(names)
    }
    limits = amounts.sum(axis=1) + rng.uniform(2.0, 5.0, 3)
    every = list(itertools.product(range(1, 6), repeat=3))

    def reliability(units):
        return math.prod(
            1.0 - (1.0 - p) ** n for p, n in zip(reliabilities, units)
        )

    def fits(units, limits):
        return all(
            use_of(units, amounts[k]) <= limit + 1e-9
            for k, limit in enumerate(limits)
        )

    best = max(reliability(u) for u in every if fits(u, limits))
    result = rbd.allocate_redundancy(
        costs, budget=dict(zip(resources, limits)), max_units=5
    )
    assert result.reliability == pytest.approx(best, abs=1e-12)
    assert fits(tuple(result.units.values()), limits)
    target = 0.95 * best
    result = rbd.allocate_redundancy(
        costs,
        target=target,
        budget={"weight": limits[1], "volume": limits[2]},
        max_units=5,
    )
    free = (math.inf, limits[1], limits[2])
    assert result.cost == pytest.approx(
        min(
            use_of(u, amounts[0])
            for u in every
            if fits(u, free) and reliability(u) >= target
        )
    )


def test_target_equal_to_an_achievable_reliability(series_ab, bridge):
    # A target exactly equal to a design's reliability is met by it: the
    # cheapest design reaching the budget optimum's reliability costs no
    # more than that optimum (dynamic program and exhaustive search).
    for rbd, costs, budget, t in [
        (series_ab, {"a": 1.0, "b": 1.0}, 4.0, None),
        (bridge, BRIDGE_COSTS, 22.0, MISSION),
    ]:
        best = rbd.allocate_redundancy(costs, budget=budget, t=t)
        cheapest = rbd.allocate_redundancy(costs, target=best.reliability, t=t)
        assert cheapest.reliability >= best.reliability
        assert cheapest.cost <= best.cost + 1e-9


def test_large_budget_still_spent_on_small_gains(series_ab):
    # With a budget of 14 the last copies add little reliability, but they
    # still add some, and the optimum is the brute-force one.
    best = max(
        (1 - 0.1**a) * (1 - 0.2**b)
        for a in range(1, 14)
        for b in range(1, 15 - a)
    )
    result = series_ab.allocate_redundancy({"a": 1.0, "b": 1.0}, budget=14)
    assert result.cost == 14
    assert result.reliability == pytest.approx(best, abs=1e-15)


def test_greedy_measures_copies_by_the_limited_resources():
    # Only weight is limited, so copies of the light node b are the cheap
    # ones, whatever their cost.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": fixed(0.8), "b": fixed(0.8)},
    )
    costs = {
        "a": {"cost": 1.0, "weight": 5.0},
        "b": {"cost": 5.0, "weight": 1.0},
    }
    for method in ("exact", "greedy"):
        result = rbd.allocate_redundancy(
            costs, budget={"weight": 11}, method=method
        )
        assert result.units == {"a": 1, "b": 6}
