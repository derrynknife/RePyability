"""Tests for redundancy allocation with a choice of component types, with
and without mixing (``ComponentOption``, issue #77).

The exact answers are checked against an independent brute force over every
combination of copies of every type, on a series system (solved by dynamic
programming) and on a bridge network (solved by exhaustive search), and
against the published optima of the classic 14-subsystem benchmark of
Fyffe, Hines & Lee (1968) in the mixing form of Coit & Smith (1996).
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import ComponentOption, NonRepairableRBD
from repyability.rbd import redundancy_allocation
from repyability.rbd.redundancy_allocation import node_designs


def fixed(reliability):
    return FixedEventProbability.from_params(1.0 - reliability)


def series(n):
    names = [f"n{i}" for i in range(n)]
    edges = [("s", names[0])] + list(zip(names, names[1:]))
    rbd = NonRepairableRBD(
        edges + [(names[-1], "t")], {name: fixed(0.5) for name in names}
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


@pytest.fixture
def bridge():
    scales = {"a": 900, "b": 700, "c": 1200, "d": 500, "e": 800}
    return NonRepairableRBD(
        BRIDGE_EDGES,
        {n: surv.Weibull.from_params([a, 1.5]) for n, a in scales.items()},
    )


def as_vector(cost):
    return tuple(cost.values()) if isinstance(cost, dict) else (cost,)


def node_options(costs, node, rbd, t):
    """(reliability, use vector) of each type a node may use."""
    entry = costs[node]
    if isinstance(entry, list):
        out = []
        for option in entry:
            model = option.reliability
            p = model if isinstance(model, float) else model.sf(t).item()
            out.append((p, as_vector(option.cost)))
        return out
    return [(rbd.reliabilities[node].sf(t).item(), as_vector(entry))]


def count_vectors(kinds, cap, mixing):
    """Every design of a node, independently of node_designs."""
    size = len(kinds)
    for counts in itertools.product(range(cap + 1), repeat=size):
        total = sum(counts)
        if not 1 <= total <= cap:
            continue
        if not mixing and sum(1 for k in counts if k) > 1:
            continue
        yield counts


def brute_force(rbd, costs, cap, limits, t, mixing):
    """Every allocation within the limits: (reliability, use, counts)."""
    nodes = list(costs)
    kinds = [node_options(costs, node, rbd, t) for node in nodes]
    x = np.atleast_1d(1.0 if t is None else float(t))
    base = rbd._base_node_probabilities(x, set(), set())
    per_node = []
    for node_kinds in kinds:
        designs = []
        for counts in count_vectors(node_kinds, cap, mixing):
            q = math.prod(
                (1 - p) ** k for (p, _), k in zip(node_kinds, counts)
            )
            use = tuple(
                sum(k * a[r] for (_, a), k in zip(node_kinds, counts))
                for r in range(len(limits))
            )
            designs.append((1.0 - q, use, counts))
        per_node.append(designs)
    for combo in itertools.product(*per_node):
        use = tuple(sum(d[1][r] for d in combo) for r in range(len(limits)))
        if any(u > limit + 1e-9 for u, limit in zip(use, limits)):
            continue
        probabilities = dict(base)
        for node, design in zip(nodes, combo):
            probabilities[node] = np.array([design[0]])
        reliability = float(np.ravel(rbd.system_probability(probabilities))[0])
        yield reliability, use, tuple(d[2] for d in combo)


def check_budget(rbd, costs, cap, limits, t, mixing):
    best = max(
        r for r, _, _ in brute_force(rbd, costs, cap, limits, t, mixing)
    )
    budget = (
        limits[0]
        if len(limits) == 1
        else dict(zip(("cost", "weight"), limits))
    )
    result = rbd.allocate_redundancy(
        costs, budget=budget, t=t, max_units=cap, mixing=mixing
    )
    assert result.reliability == pytest.approx(best, abs=1e-12)
    for used, limit in zip(result.resources.values(), limits):
        assert used <= limit + 1e-9
    return result


def check_target(rbd, costs, cap, target, t, mixing, limits):
    # limits: one per resource, math.inf where it is not limited.
    cheapest = min(
        use[0]
        for r, use, _ in brute_force(rbd, costs, cap, limits, t, mixing)
        if r >= target
    )
    budget = {
        name: limit
        for name, limit in zip(("cost", "weight"), limits)
        if math.isfinite(limit)
    }
    result = rbd.allocate_redundancy(
        costs,
        target=target,
        budget=budget or None,
        t=t,
        max_units=cap,
        mixing=mixing,
    )
    assert result.cost == pytest.approx(cheapest)
    assert result.reliability >= target
    return result


def random_options(rng, n, resources):
    out = []
    for j in range(n):
        amounts = rng.integers(1, 5, len(resources)).astype(float)
        cost = (
            float(amounts[0])
            if len(resources) == 1
            else dict(zip(resources, map(float, amounts)))
        )
        out.append(
            ComponentOption(f"type{j}", float(rng.uniform(0.5, 0.97)), cost)
        )
    return out


# -- exact is optimal: independent brute force -----------------------------


@pytest.mark.parametrize("mixing", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test_series_matches_brute_force(seed, mixing):
    # A series of nodes with two or three types each: dynamic programming.
    rng = np.random.default_rng(seed)
    names, rbd = series(3)
    costs = {
        n: random_options(rng, 2 + (i == 0), ("cost",))
        for i, n in enumerate(names)
    }
    total = sum(min(o.cost for o in costs[n]) for n in names)
    check_budget(rbd, costs, 3, (total + 5.0,), None, mixing)
    best = max(
        r for r, _, _ in brute_force(rbd, costs, 3, (math.inf,), None, mixing)
    )
    check_target(rbd, costs, 3, 0.9 * best, None, mixing, (math.inf,))


@pytest.mark.parametrize("mixing", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_series_two_resources_match_brute_force(seed, mixing):
    rng = np.random.default_rng(10 + seed)
    names, rbd = series(3)
    costs = {n: random_options(rng, 2, ("cost", "weight")) for n in names}
    least = [
        sum(min(o.cost[r] for o in costs[n]) for n in names)
        for r in ("cost", "weight")
    ]
    limits = (least[0] + 5.0, least[1] + 4.0)
    check_budget(rbd, costs, 3, limits, None, mixing)
    best = max(
        r for r, _, _ in brute_force(rbd, costs, 3, limits, None, mixing)
    )
    check_target(
        rbd, costs, 3, 0.95 * best, None, mixing, limits=(math.inf, limits[1])
    )


@pytest.mark.parametrize("mixing", [True, False])
def test_bridge_matches_brute_force(bridge, mixing):
    # Not a series of the costed nodes: exhaustive search. Two nodes have a
    # choice of types (one time-varying, one fixed), the others do not.
    costs = {
        "a": [
            ComponentOption("own", bridge.reliabilities["a"], 3.0),
            ComponentOption(
                "rugged", surv.Weibull.from_params([2000, 2]), 5.0
            ),
        ],
        "b": 2.0,
        "c": 4.0,
        "d": [
            ComponentOption("cheap", 0.6, 1.0),
            ComponentOption("good", 0.9, 2.5),
        ],
        "e": 2.5,
    }
    result = check_budget(bridge, costs, 3, (21.0,), 400.0, mixing)
    assert set(result.mix) == {"a", "d"}
    for node, used in result.mix.items():
        assert sum(used.values()) == result.units[node]
        assert all(k > 0 for k in used.values())
        if not mixing:
            assert len(used) == 1
    check_target(bridge, costs, 3, 0.97, 400.0, mixing, (math.inf,))


def test_mixing_is_never_worse():
    rng = np.random.default_rng(7)
    for _ in range(5):
        names, rbd = series(3)
        costs = {n: random_options(rng, 3, ("cost",)) for n in names}
        budget = sum(min(o.cost for o in costs[n]) for n in names) + 6.0
        mixed = rbd.allocate_redundancy(costs, budget=budget, max_units=4)
        single = rbd.allocate_redundancy(
            costs, budget=budget, max_units=4, mixing=False
        )
        assert mixed.reliability >= single.reliability - 1e-15


@pytest.mark.parametrize("method", ["exact", "greedy"])
@pytest.mark.parametrize("form", ["budget", "target"])
def test_one_option_is_the_plain_node(bridge, method, form):
    # A node whose only option is its own model and cost is the plain node.
    plain = {"a": 3.0, "b": 2.0, "c": 4.0, "d": 1.5, "e": 2.5}
    optioned = dict(plain)
    for node in ("a", "d"):
        optioned[node] = [
            ComponentOption("own", bridge.reliabilities[node], plain[node])
        ]
    kwargs = {"budget": 22.0} if form == "budget" else {"target": 0.97}
    one = bridge.allocate_redundancy(plain, t=400.0, method=method, **kwargs)
    two = bridge.allocate_redundancy(
        optioned, t=400.0, method=method, **kwargs
    )
    assert two.units == one.units
    assert two.reliability == one.reliability
    assert two.cost == one.cost
    assert two.mix == {node: {"own": one.units[node]} for node in ("a", "d")}


@pytest.mark.parametrize("mixing", [True, False])
def test_greedy_is_feasible_and_never_better(mixing):
    rng = np.random.default_rng(3)
    for _ in range(5):
        names, rbd = series(4)
        costs = {n: random_options(rng, 2, ("cost",)) for n in names}
        budget = sum(min(o.cost for o in costs[n]) for n in names) + 8.0
        exact = rbd.allocate_redundancy(
            costs, budget=budget, max_units=5, mixing=mixing
        )
        greedy = rbd.allocate_redundancy(
            costs, budget=budget, max_units=5, mixing=mixing, method="greedy"
        )
        assert greedy.cost <= budget + 1e-9
        assert greedy.reliability <= exact.reliability + 1e-15
        for node, used in greedy.mix.items():
            assert sum(used.values()) == greedy.units[node] <= 5
            assert mixing or len(used) == 1


def test_options_are_evaluated_at_the_mission_time():
    # A Weibull option at t = 500 is its survival probability there.
    names, rbd = series(2)
    model = surv.Weibull.from_params([1000, 2])
    p = float(model.sf(500))
    timed = {
        "n0": [ComponentOption("w", model, 1.0)],
        "n1": [ComponentOption("f", 0.8, 1.0)],
    }
    fixed_equivalent = {
        "n0": [ComponentOption("w", p, 1.0)],
        "n1": [ComponentOption("f", 0.8, 1.0)],
    }
    one = rbd.allocate_redundancy(timed, budget=5, t=500)
    two = rbd.allocate_redundancy(fixed_equivalent, budget=5)
    assert one.units == two.units
    assert one.reliability == pytest.approx(two.reliability, rel=1e-12)
    with pytest.raises(ValueError, match="'w' of node 'n0' is time-varying"):
        rbd.allocate_redundancy(timed, budget=5)


# -- the designs of one node -----------------------------------------------


@pytest.mark.parametrize("mixing", [True, False])
@pytest.mark.parametrize("m", [1, 2])
def test_node_designs_are_the_front_of_every_design(m, mixing):
    rng = np.random.default_rng(m + 2 * mixing)
    kinds = [
        (float(p), tuple(rng.integers(1, 4, m).astype(float)))
        for p in rng.uniform(0.5, 0.95, 3)
    ]
    spare = tuple(sum(a[r] for _, a in kinds) + 3.0 for r in range(m))
    every = []
    for counts in count_vectors(kinds, 4, mixing):
        use = tuple(
            float(sum(k * a[r] for (_, a), k in zip(kinds, counts)))
            for r in range(m)
        )
        if any(u > s for u, s in zip(use, spare)):
            continue
        q = 1.0
        for (p, _), k in zip(kinds, counts):
            if k:
                q *= (1.0 - p) ** k
        every.append((use, 1.0 - q, counts))

    def beaten(s):
        return any(
            all(a <= b for a, b in zip(t[0], s[0]))
            and t[1] >= s[1]
            and (t[0], t[1]) != (s[0], s[1])
            for t in every
        )

    expected = {(s[0], s[1]) for s in every if not beaten(s)}
    designs = node_designs(kinds, 4, spare, mixing)
    assert {(d.use, d.reliability) for d in designs} == expected
    assert len(designs) == len(expected)
    assert designs == sorted(designs, key=lambda d: (d.use[0], -d.reliability))
    for d in designs:
        assert 1 <= sum(d.counts) <= 4
        assert d.unreliability == pytest.approx(1.0 - d.reliability)


def test_node_designs_stop_where_copies_add_nothing():
    # 1 - 0.5 ** n is exactly 1 in double precision for n >= 54.
    designs = node_designs([(0.5, 1.0)], math.inf, spare=1000.0)
    assert designs[-1].reliability == 1.0
    assert sum(designs[-1].counts) <= 61
    assert [d.reliability for d in designs] == sorted(
        {d.reliability for d in designs}
    )


# -- the classic benchmark -------------------------------------------------

# Fyffe, Hines & Lee (1968): 14 subsystems in series, each with three or
# four component types (reliability, cost, weight), at most 8 components per
# subsystem, cost at most 130, and the weight limit varied (Nakagawa &
# Miyazaki, 1981). Coit & Smith (1996) allowed mixing types.
BENCHMARK = [
    [(0.90, 1, 3), (0.93, 1, 4), (0.91, 2, 2), (0.95, 2, 5)],
    [(0.95, 2, 8), (0.94, 1, 10), (0.93, 1, 9)],
    [(0.85, 2, 7), (0.90, 3, 5), (0.87, 1, 6), (0.92, 4, 4)],
    [(0.83, 3, 5), (0.87, 4, 6), (0.85, 5, 4)],
    [(0.94, 2, 4), (0.93, 2, 3), (0.95, 3, 5)],
    [(0.99, 3, 5), (0.98, 3, 4), (0.97, 2, 5), (0.96, 2, 4)],
    [(0.91, 4, 7), (0.92, 4, 8), (0.94, 5, 9)],
    [(0.81, 3, 4), (0.90, 5, 7), (0.91, 6, 6)],
    [(0.97, 2, 8), (0.99, 3, 9), (0.96, 4, 7), (0.91, 3, 8)],
    [(0.83, 4, 6), (0.85, 4, 5), (0.90, 5, 6)],
    [(0.94, 3, 5), (0.95, 4, 6), (0.96, 5, 6)],
    [(0.79, 2, 4), (0.82, 3, 5), (0.85, 4, 6), (0.90, 5, 7)],
    [(0.98, 2, 5), (0.99, 3, 5), (0.97, 2, 6)],
    [(0.90, 4, 6), (0.92, 4, 7), (0.95, 5, 6), (0.99, 6, 9)],
]


@pytest.mark.parametrize(
    "weight, reliability",
    [(191, 0.986811), (190, 0.986416), (189, 0.985922), (159, 0.954565)],
)
def test_benchmark_with_mixing(weight, reliability):
    # The best reliabilities reported in the literature for these instances
    # (to the six decimals they are quoted at).
    names, rbd = series(14)
    costs = {
        name: [
            ComponentOption(j, r, {"cost": c, "weight": w})
            for j, (r, c, w) in enumerate(types)
        ]
        for name, types in zip(names, BENCHMARK)
    }
    best = rbd.allocate_redundancy(
        costs, budget={"cost": 130, "weight": weight}, max_units=8
    )
    assert round(best.reliability, 6) == reliability
    assert best.resources["cost"] <= 130
    assert best.resources["weight"] <= weight
    assert all(1 <= n <= 8 for n in best.units.values())
    # Evaluated independently: the product of the subsystems.
    independent = math.prod(
        1.0
        - math.prod(
            (1.0 - types[j][0]) ** k for j, k in best.mix[name].items()
        )
        for name, types in zip(names, BENCHMARK)
    )
    assert independent == pytest.approx(best.reliability, rel=1e-12)


# -- validation ------------------------------------------------------------


def two_types():
    return [
        ComponentOption("x", 0.9, 1.0),
        ComponentOption("y", 0.95, 2.0),
    ]


@pytest.mark.parametrize(
    "costs, kwargs, match",
    [
        ({"n0": []}, {"budget": 3}, "empty list of options"),
        ({"n0": [0.9]}, {"budget": 3}, "ComponentOption instances"),
        (
            {
                "n0": [
                    ComponentOption("x", 0.9, 1.0),
                    ComponentOption("x", 0.95, 2.0),
                ]
            },
            {"budget": 3},
            "distinct names",
        ),
        (
            {"n0": [ComponentOption("x", "high", 1.0)]},
            {"budget": 3},
            "a model with an sf method",
        ),
        (
            {"n0": [ComponentOption("x", 1.5, 1.0)]},
            {"budget": 3},
            r"in \[0, 1\]",
        ),
        (
            {"n0": [ComponentOption("x", math.nan, 1.0)]},
            {"budget": 3},
            r"in \[0, 1\]",
        ),
        (
            {"n0": [ComponentOption("x", 0.9, {"cost": 1.0})], "n1": 1.0},
            {"budget": 3},
            "not a mixture",
        ),
        (
            {
                "n0": [
                    ComponentOption("x", 0.9, {"cost": 1.0, "weight": 1.0}),
                    ComponentOption("y", 0.9, {"cost": 1.0}),
                ]
            },
            {"budget": {"cost": 3}},
            "option 'y' of node 'n0'",
        ),
        (
            {"n0": [ComponentOption("x", 0.9, 0.0)]},
            {"budget": 3},
            "finite and positive",
        ),
        (
            {
                "n0": [
                    ComponentOption("x", 0.9, {"cost": 1.0, "weight": 0.0}),
                    ComponentOption("y", 0.9, {"cost": 1.0, "weight": 1.0}),
                ]
            },
            {"budget": {"weight": 3}},
            "without limit",
        ),
        ({"n0": two_types()}, {"budget": 3, "mixing": "yes"}, "mixing"),
    ],
)
def test_option_validation(costs, kwargs, match):
    _, rbd = series(2)
    with pytest.raises(ValueError, match=match):
        rbd.allocate_redundancy(costs, **kwargs)


def test_too_many_designs_fails_fast(monkeypatch):
    monkeypatch.setattr(redundancy_allocation, "DESIGN_LIMIT", 5)
    _, rbd = series(2)
    with pytest.raises(ValueError, match="node 'n0'.*max_units"):
        rbd.allocate_redundancy({"n0": two_types()}, budget=10)


def test_no_design_fits_the_budget():
    # One copy of either type fits each limit alone, but neither fits both.
    _, rbd = series(2)
    options = [
        ComponentOption("light", 0.9, {"cost": 5.0, "weight": 1.0}),
        ComponentOption("cheap", 0.9, {"cost": 1.0, "weight": 5.0}),
    ]
    with pytest.raises(ValueError, match="No design of node 'n0' fits"):
        rbd.allocate_redundancy(
            {"n0": options}, budget={"cost": 2, "weight": 2}
        )


# -- details of the searches -----------------------------------------------


def test_no_budget_is_spent_on_an_irrelevant_node():
    # x only duplicates the direct path a -> t, so it adds nothing: the
    # search fills the budget, then moves x back to one copy.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t"), ("a", "x"), ("x", "t")],
        {"a": fixed(0.8), "x": fixed(0.7)},
    )
    result = rbd.allocate_redundancy({"a": 2.0, "x": 1.0}, budget=6)
    assert result.units == {"a": 2, "x": 1}
    assert result.cost == 5.0
    assert result.reliability == pytest.approx(0.96)


def test_cost_ties_go_to_the_more_reliable_design():
    # a in series with b and c in parallel (not a series of the costed
    # nodes, so the exhaustive search): copies of b are free, so every
    # cheapest design takes all three.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("a", "c"), ("b", "t"), ("c", "t")],
        {"a": fixed(0.9), "b": fixed(0.6), "c": fixed(0.7)},
    )
    costs = {"a": {"cost": 1.0}, "b": {"cost": 0.0}, "c": {"cost": 1.0}}
    # Both (a, c) = (1, 2) and (2, 1) meet 89% at a cost of 3; the search
    # meets (1, 2) first, but (2, 1) is more reliable.
    result = rbd.allocate_redundancy(costs, target=0.89, max_units={"b": 3})
    designs = [
        (a + c, (1 - 0.1**a) * (1 - 0.4**b * 0.3**c), (a, b, c))
        for a in range(1, 5)
        for b in range(1, 4)
        for c in range(1, 5)
    ]
    cheapest = min(cost for cost, r, _ in designs if r >= 0.89)
    best = max(r for cost, r, _ in designs if r >= 0.89 and cost == cheapest)
    assert result.cost == cheapest
    assert result.reliability == pytest.approx(best)
    assert result.units["b"] == 3


def test_target_ceiling_uses_the_best_type():
    # Only the good type can reach 95%, one copy of it is enough.
    _, rbd = series(1)
    options = [
        ComponentOption("poor", 0.5, 1.0),
        ComponentOption("good", 0.99, 5.0),
    ]
    result = rbd.allocate_redundancy({"n0": options}, target=0.95, max_units=2)
    assert result.mix == {"n0": {"good": 1}}
    with pytest.raises(ValueError, match="unreachable"):
        rbd.allocate_redundancy({"n0": options[:1]}, target=0.95, max_units=2)


def test_greedy_changes_the_type_of_a_copy():
    # From one standard copy, swapping it for a premium one (same spend) is
    # better than adding a second standard copy.
    _, rbd = series(1)
    options = [
        ComponentOption("standard", 0.8, 1.0),
        ComponentOption("premium", 0.99, 2.0),
    ]
    for mixing in (True, False):
        result = rbd.allocate_redundancy(
            {"n0": options}, budget=2, method="greedy", mixing=mixing
        )
        assert result.mix == {"n0": {"premium": 1}}


def test_greedy_starts_from_the_cheapest_types():
    # A budget that affords exactly one of each node's cheapest type.
    _, rbd = series(2)
    options = [
        ComponentOption("cheap", 0.8, 1.0),
        ComponentOption("dear", 0.9, 5.0),
    ]
    result = rbd.allocate_redundancy(
        {"n0": options, "n1": options}, budget=2, method="greedy"
    )
    assert result.mix == {"n0": {"cheap": 1}, "n1": {"cheap": 1}}


@pytest.mark.parametrize("method", ["exact", "greedy"])
def test_target_start_must_fit_every_limit(method):
    # The cheapest type meets the target but is too heavy: the answer must
    # still respect the weight limit.
    _, rbd = series(1)
    options = [
        ComponentOption("heavy", 0.95, {"cost": 1.0, "weight": 5.0}),
        ComponentOption("light", 0.95, {"cost": 2.0, "weight": 1.0}),
    ]
    if method == "greedy":
        with pytest.raises(ValueError, match="within the budget"):
            rbd.allocate_redundancy(
                {"n0": options},
                target=0.9,
                budget={"weight": 2},
                method=method,
                max_units=3,
            )
        return
    result = rbd.allocate_redundancy(
        {"n0": options}, target=0.9, budget={"weight": 2}, max_units=3
    )
    assert result.mix == {"n0": {"light": 1}}
    assert result.resources == {"cost": 2.0, "weight": 1.0}
