"""The capacity analysis (#97): the exact distribution of how much a system
can deliver, from its nodes' capacities.

The reference values are independent of the engine: every combination of
the components' states is enumerated, and the most that can flow through
the ones reached (k-out-of-n nodes need k of their inputs reached) is found
by networkx's maximum flow, with each node split into an arc that carries
its capacity. Closed forms check k identical units of ``1 / k`` capacity,
and the capacity is positive exactly when the system works.
"""

import itertools
import math
import pickle
import random

import networkx as nx
import numpy as np
import pytest
import surpyval as surv

from repyability import (
    RBD,
    BetaFactor,
    CapacityDistribution,
    CCFGroup,
    DegradingNode,
    NonRepairableRBD,
    RepairableRBD,
)
from repyability.rbd import capacity as engine

FEP = surv.FixedEventProbability
E = surv.Exponential.from_params
W = surv.Weibull.from_params

BRIDGE = [
    ("s", 1),
    ("s", 2),
    (1, 3),
    (2, 3),
    (1, 4),
    (3, 4),
    (2, 5),
    (3, 5),
    (4, "t"),
    (5, "t"),
]


def enumerated(edges, k, capacity, works, repeats=None):
    """The capacity distribution by enumerating the components' states and
    finding each one's maximum flow with networkx."""
    repeats = repeats or {}
    G = nx.DiGraph(edges)
    (source,) = [n for n in G if G.in_degree(n) == 0]
    (sink,) = [n for n in G if G.out_degree(n) == 0]
    components = sorted(
        {repeats.get(n, n) for n in G if n not in (source, sink)}, key=str
    )
    order = list(nx.topological_sort(G))
    out: dict = {}
    for states in itertools.product([False, True], repeat=len(components)):
        up = dict(zip(components, states))
        chance = math.prod(
            works[c] if up[c] else 1.0 - works[c] for c in components
        )
        reached = {source: True}
        for v in order[1:]:
            fed = sum(reached[u] for u in G.predecessors(v)) >= k.get(v, 1)
            reached[v] = fed and (v == sink or up[repeats.get(v, v)])
        level = 0.0
        if reached[sink]:
            flow = nx.DiGraph()
            for v in G:
                if reached[v]:
                    limit = capacity.get(repeats.get(v, v), math.inf)
                    arc = {} if math.isinf(limit) else {"capacity": limit}
                    flow.add_edge((v, 0), (v, 1), **arc)
            for a, b in G.edges:
                if reached[a] and reached[b]:
                    flow.add_edge((a, 1), (b, 0))
            try:
                level = float(
                    nx.maximum_flow_value(flow, (source, 0), (sink, 1))
                )
            except nx.NetworkXUnbounded:
                level = math.inf
        level = engine.tidy(level)
        out[level] = out.get(level, 0.0) + chance
    levels = sorted(level for level, p in out.items() if p > 0)
    return np.array(levels), np.array([out[level] for level in levels])


def exact(edges, k, capacity, works, repeats=None):
    """The engine's distribution, through a fixed-probability RBD."""
    models = {c: FEP.from_params(1.0 - p) for c, p in works.items()}
    if repeats:
        models.update(repeats)
    rbd = NonRepairableRBD(edges, models, k=k, capacity=capacity)
    return rbd.capacity_distribution()


def assert_same(got: CapacityDistribution, want) -> None:
    levels, probabilities = want
    np.testing.assert_allclose(got.levels, levels)
    np.testing.assert_allclose(got.probabilities, probabilities, rtol=1e-12)


# --- Against enumeration -----------------------------------------------------


def test_bridge_matches_enumeration():
    capacity = {1: 10, 2: 20, 3: 5, 4: 15, 5: 10}
    works = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.95}
    got = exact(BRIDGE, {}, capacity, works)
    assert_same(got, enumerated(BRIDGE, {}, capacity, works))
    # All working: 25 = min(10 + 20, 15 + 10, 10 + 5 + 10, 20 + 5 + 15).
    assert got.levels[-1] == 25.0


def test_bridge_without_capacity_on_the_bridge():
    # A bridge with no capacity limits nothing: the cuts through it are
    # never the least.
    capacity = {1: 10, 2: 20, 4: 15, 5: 10}
    works = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.95}
    got = exact(BRIDGE, {}, capacity, works)
    assert_same(got, enumerated(BRIDGE, {}, capacity, works))


def test_koon_nodes_match_enumeration():
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("s", "c"),
        ("a", "v"),
        ("b", "v"),
        ("c", "v"),
        ("a", "w"),
        ("w", "t"),
        ("v", "t"),
    ]
    capacity = {"a": 3, "b": 5, "c": 7, "v": 12, "w": 2}
    works = {"a": 0.9, "b": 0.8, "c": 0.7, "v": 0.95, "w": 0.6}
    for k in ({"v": 2}, {"v": 3}, {"t": 2}, {"v": 2, "t": 2}):
        got = exact(edges, k, capacity, works)
        assert_same(got, enumerated(edges, k, capacity, works))


def test_repeated_nodes_match_enumeration():
    # One power supply drawn in both branches, carrying each branch's flow.
    edges = [
        ("s", "a"),
        ("a", "psu"),
        ("psu", "t"),
        ("s", "b"),
        ("b", "psu2"),
        ("psu2", "t"),
        ("s", "c"),
        ("c", "t"),
    ]
    repeats = {"psu2": "psu"}
    works = {"a": 0.9, "b": 0.8, "c": 0.7, "psu": 0.6}
    for capacity in ({"a": 1, "b": 2, "c": 4}, {"a": 1, "b": 2, "psu": 1.5}):
        got = exact(edges, {}, capacity, works, repeats)
        assert_same(got, enumerated(edges, {}, capacity, works, repeats))


@pytest.mark.parametrize("seed", range(4))
def test_random_diagrams_match_enumeration(seed):
    rng = random.Random(seed)
    checked = 0
    while checked < 15:
        size = rng.randint(2, 7)
        edges = set()
        for i in range(size):
            for j in range(i + 1, size):
                if rng.random() < 0.35:
                    edges.add((i, j))
        for i in range(size):
            if not any(b == i for _, b in edges) or rng.random() < 0.2:
                edges.add(("s", i))
            if not any(a == i for a, _ in edges) or rng.random() < 0.2:
                edges.add((i, "t"))
        graph = nx.DiGraph(sorted(edges, key=str))
        k = {
            v: rng.randint(2, graph.in_degree(v))
            for v in list(range(size)) + ["t"]
            if graph.in_degree(v) >= 2 and rng.random() < 0.3
        }
        try:
            RBD(graph.edges, k=k)
        except ValueError:
            continue
        levels = [1, 2, 3, 0.1, 0.2, 7.5, math.inf]
        capacity = {v: rng.choice(levels) for v in range(size)}
        works = {v: rng.uniform(0.05, 0.95) for v in range(size)}
        got = exact(list(graph.edges), k, capacity, works)
        assert_same(got, enumerated(graph.edges, k, capacity, works))
        checked += 1


# --- Closed forms ------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 10])
def test_k_identical_units_of_one_kth_capacity(k):
    # k units of 1/k in parallel: the capacity is j/k with the binomial
    # probability of j working, however the sums of 1/k round.
    p = 0.85
    units = [f"u{i}" for i in range(k)]
    rbd = NonRepairableRBD(
        [("s", u) for u in units] + [(u, "t") for u in units],
        {u: FEP.from_params(1 - p) for u in units},
        capacity={u: 1 / k for u in units},
    )
    capacity = rbd.capacity_distribution()
    np.testing.assert_allclose(capacity.levels, np.arange(k + 1) / k)
    binomial = [
        math.comb(k, j) * p**j * (1 - p) ** (k - j) for j in range(k + 1)
    ]
    np.testing.assert_allclose(capacity.probabilities, binomial, rtol=1e-12)
    assert capacity.meets(1.0) == pytest.approx(p**k)
    assert capacity.meets(1 / k) == pytest.approx(1 - (1 - p) ** k)
    assert capacity.mean() == pytest.approx(p)
    # Against a demand of 1 the fraction delivered is the expected
    # capacity: no state carries more than 1.
    assert capacity.delivered_fraction(1.0) == pytest.approx(p)


def test_n_units_of_one_kth_meet_the_demand_as_k_out_of_n():
    # Five units of 1/3: meeting a demand of 1 needs three working.
    p = 0.9
    units = [f"u{i}" for i in range(5)]
    rbd = NonRepairableRBD(
        [("s", u) for u in units] + [(u, "t") for u in units],
        {u: FEP.from_params(1 - p) for u in units},
        capacity={u: 1 / 3 for u in units},
    )
    three_of_five = sum(
        math.comb(5, j) * p**j * (1 - p) ** (5 - j) for j in range(3, 6)
    )
    assert rbd.capacity_distribution().meets(1.0) == pytest.approx(
        three_of_five
    )


def test_series_takes_the_least_and_parallel_the_sum():
    rbd = RBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        capacity={"a": 30, "b": 50, "c": 60},
    )
    capacity = rbd.system_capacity({"a": 0.9, "b": 0.8, "c": 0.7})
    assert capacity.levels.tolist() == [0.0, 30.0, 50.0, 60.0]
    np.testing.assert_allclose(
        capacity.probabilities,
        [
            1 - 0.7 * (1 - 0.1 * 0.2),
            0.7 * 0.9 * 0.2,
            0.7 * 0.1 * 0.8,
            0.7 * 0.9 * 0.8,
        ],
    )


def test_three_half_capacity_pumps():
    # The issue's example: one failure costs nothing, two cost half.
    rbd = RBD(
        [("s", u) for u in "abc"] + [(u, "t") for u in "abc"],
        capacity={u: 50 for u in "abc"},
    )
    capacity = rbd.system_capacity({u: 0.9 for u in "abc"})
    assert capacity.levels.tolist() == [0.0, 50.0, 100.0, 150.0]
    assert capacity.meets(100) == pytest.approx(0.9**3 + 3 * 0.9**2 * 0.1)
    assert capacity.delivered_fraction(100) == pytest.approx(
        0.9**3 + 3 * 0.9**2 * 0.1 + 0.5 * 3 * 0.9 * 0.1**2
    )


# --- k-out-of-n nodes --------------------------------------------------------


def test_a_voting_output_passes_nothing_below_k():
    # Three pumps voting 2-out-of-3 at the output: one pump alone is a
    # failed system, so it delivers nothing.
    rbd = RBD(
        [("s", u) for u in "abc"] + [(u, "t") for u in "abc"],
        k={"t": 2},
        capacity={u: 50 for u in "abc"},
    )
    capacity = rbd.system_capacity({u: 0.9 for u in "abc"})
    assert capacity.levels.tolist() == [0.0, 100.0, 150.0]
    assert capacity.probabilities[0] == pytest.approx(
        0.1**3 + 3 * 0.9 * 0.1**2
    )


def test_needing_all_members_still_adds_their_capacities():
    # A node needing every one of its inputs works as a series chain would,
    # but carries their sum.
    rbd = RBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        k={"t": 2},
        capacity={"a": 1, "b": 2},
    )
    capacity = rbd.system_capacity({"a": 0.9, "b": 0.8})
    assert capacity.levels.tolist() == [0.0, 3.0]
    assert capacity.probabilities[1] == pytest.approx(0.72)


def test_a_component_that_never_decides_can_still_carry_flow():
    # "w" needs both "a" and "v", while "a" also feeds the output directly:
    # whether the system works depends on "a" alone, but "v" and "w" add a
    # second route while they work.
    edges = [
        ("s", "a"),
        ("s", "v"),
        ("a", "w"),
        ("v", "w"),
        ("a", "t"),
        ("w", "t"),
    ]
    k = {"w": 2}
    capacity = {"a": 10, "v": 10, "w": 10}
    works = {"a": 0.9, "v": 0.8, "w": 0.7}
    got = exact(edges, k, capacity, works)
    assert_same(got, enumerated(edges, k, capacity, works))
    assert got.levels.tolist() == [0.0, 10.0, 20.0]


# --- The capacity is positive exactly when the system works ------------------


def test_positive_capacity_is_the_reliability_over_time():
    rbd = NonRepairableRBD(
        BRIDGE,
        {
            1: W([1000, 2]),
            2: W([800, 1.5]),
            3: E([1e-3]),
            4: W([1200, 3]),
            5: W([900, 2]),
        },
        capacity={1: 10, 2: 20, 3: 5, 4: 15, 5: 10},
    )
    t = np.array([0.0, 100.0, 500.0, 2000.0])
    capacity = rbd.capacity_distribution(t)
    assert capacity.probabilities.shape == (len(capacity.levels), 4)
    np.testing.assert_allclose(capacity.probabilities.sum(axis=0), 1.0)
    np.testing.assert_allclose(capacity.meets(1e-9), rbd.sf(t))
    # A scalar time gives one probability per level, and floats.
    one = rbd.capacity_distribution(500.0)
    np.testing.assert_allclose(one.probabilities, capacity.probabilities[:, 2])
    assert isinstance(one.meets(20), float)
    assert isinstance(one.mean(), float)


@pytest.mark.filterwarnings("ignore:Common-cause group:UserWarning")
def test_common_cause_groups_are_honoured():
    rbd = NonRepairableRBD(
        [("in", "a"), ("in", "b"), ("a", "out"), ("b", "out")],
        {"a": W([100, 2]), "b": W([100, 2])},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        capacity={"a": 1, "b": 2},
    )
    capacity = rbd.capacity_distribution(50)
    assert capacity.meets(0.5) == pytest.approx(rbd.sf(50))
    # The shared cause fails both together, so both up is more likely than
    # with independent failures.
    independent = NonRepairableRBD(
        [("in", "a"), ("in", "b"), ("a", "out"), ("b", "out")],
        {"a": W([100, 2]), "b": W([100, 2])},
        capacity={"a": 1, "b": 2},
    ).capacity_distribution(50)
    assert capacity.meets(3) > independent.meets(3)
    assert capacity.probabilities.sum() == pytest.approx(1.0)


def test_forced_nodes():
    rbd = RBD(
        [("s", u) for u in "abc"] + [(u, "t") for u in "abc"],
        capacity={u: 50 for u in "abc"},
    )
    models = {u: FEP.from_params(0.1) for u in "abc"}
    plant = NonRepairableRBD(rbd.G.edges, models, capacity=rbd.capacity)
    broken = plant.capacity_distribution(broken_nodes=["a"])
    assert broken.levels.tolist() == [0.0, 50.0, 100.0]
    working = plant.capacity_distribution(working_nodes=["a", "b", "c"])
    assert working.levels.tolist() == [150.0]
    assert working.probabilities.tolist() == [1.0]
    with pytest.raises(ValueError):
        plant.capacity_distribution(broken_nodes=["nope"])


def test_repairable_long_run():
    pump = {
        "reliability": E([0.1]),
        "repairability": E([1.0]),
    }
    plant = RepairableRBD(
        [("s", u) for u in "abc"] + [(u, "t") for u in "abc"],
        {u: pump for u in "abc"},
        capacity={u: 50 for u in "abc"},
    )
    capacity = plant.capacity_distribution()
    a = 10 / 11
    np.testing.assert_allclose(
        capacity.probabilities,
        [(1 - a) ** 3, 3 * a * (1 - a) ** 2, 3 * a**2 * (1 - a), a**3],
    )
    assert capacity.meets(1) == pytest.approx(plant.mean_availability())
    assert capacity.mean() == pytest.approx(150 * a)
    down = plant.capacity_distribution(broken_nodes=["a"])
    assert down.levels.tolist() == [0.0, 50.0, 100.0]


def test_repairable_calendar_and_nested_rbd():
    # Two units inspected together are down together more often than
    # independent ones: the long-run distribution averages over the
    # inspection schedule, as mean_availability does.
    inspected = {
        "reliability": E([0.01]),
        "repairability": "instant",
        "inspection": {"interval": 10.0},
    }
    pair = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": inspected, "b": inspected},
        capacity={"a": 1, "b": 1},
    )
    capacity = pair.capacity_distribution()
    assert capacity.meets(0.5) == pytest.approx(pair.mean_availability())
    q = 1 - (1 - math.exp(-0.1)) / 0.1  # each unit's unavailability
    assert capacity.probabilities[0] > q**2
    # A nested RBD is one node: up while its own system is up.
    unit = {"reliability": E([0.1]), "repairability": E([1.0])}
    inner = RepairableRBD([("s", "x"), ("x", "t")], {"x": unit})
    outer = RepairableRBD(
        [("s", "inner"), ("s", "y"), ("inner", "t"), ("y", "t")],
        {"inner": inner, "y": unit},
        capacity={"inner": 2, "y": 1},
    )
    a = 10 / 11
    np.testing.assert_allclose(
        outer.capacity_distribution().probabilities,
        [(1 - a) ** 2, a * (1 - a), a * (1 - a), a**2],
    )


# --- The result --------------------------------------------------------------


def test_meeting_a_demand_allows_for_rounding():
    capacity = CapacityDistribution(
        np.array([0.0, 0.9999999999999999, 2.0]), np.array([0.2, 0.3, 0.5])
    )
    assert capacity.meets(1.0) == pytest.approx(0.8)
    assert capacity.meets(0.0) == pytest.approx(1.0)
    assert capacity.meets(-5.0) == pytest.approx(1.0)
    assert capacity.meets(math.inf) == 0.0
    with pytest.raises(ValueError, match="NaN"):
        capacity.meets(math.nan)


def test_unlimited_capacity():
    # With no capacity on a branch, the system's capacity is unlimited
    # while that branch works.
    rbd = RBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        capacity={"a": 5},
    )
    capacity = rbd.system_capacity({"a": 0.9, "b": 0.8})
    assert capacity.levels.tolist() == [0.0, 5.0, math.inf]
    assert capacity.meets(math.inf) == pytest.approx(0.8)
    assert capacity.mean() == math.inf
    assert capacity.delivered_fraction(10) == pytest.approx(
        0.8 + 0.9 * 0.2 * 0.5
    )
    # An edge from the input to the output carries anything.
    direct = RBD(
        [("s", "a"), ("a", "t"), ("s", "t")], capacity={"a": 5}
    ).system_capacity({"a": 0.5})
    assert direct.levels.tolist() == [math.inf]


def test_mean_ignores_unreachable_infinite_levels():
    capacity = CapacityDistribution(
        np.array([0.0, 1.0, math.inf]),
        np.array([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]),
    )
    mean = capacity.mean()
    assert mean[0] == pytest.approx(0.5)
    assert mean[1] == math.inf


@pytest.mark.parametrize("demand", [0.0, -1.0, math.inf, math.nan])
def test_delivered_fraction_needs_a_positive_finite_demand(demand):
    capacity = CapacityDistribution(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="positive, finite"):
        capacity.delivered_fraction(demand)


def test_the_result_is_a_mapping():
    capacity = CapacityDistribution(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    assert set(capacity) == {"levels", "probabilities"}
    assert capacity["levels"] is capacity.levels


# --- Construction, saving and errors -----------------------------------------


@pytest.mark.parametrize(
    "capacity, message",
    [
        ({"s": 1}, "input or output"),
        ({"t": 1}, "input or output"),
        ({"zz": 1}, "Unknown node"),
        ({"a": 0}, "positive"),
        ({"a": -2.5}, "positive"),
        ({"a": math.nan}, "positive"),
        ({"a": "10"}, "number"),
        ({"a": True}, "number"),
        ({"a": None}, "number"),
    ],
)
def test_invalid_capacities(capacity, message):
    with pytest.raises(ValueError, match=message):
        RBD([("s", "a"), ("a", "t")], capacity=capacity)


def test_a_repeated_node_takes_the_capacity_of_the_one_it_repeats():
    with pytest.raises(ValueError, match="repeats node"):
        NonRepairableRBD(
            [("s", "a"), ("a", "t"), ("s", "a2"), ("a2", "t")],
            {"a": FEP.from_params(0.1), "a2": "a"},
            capacity={"a2": 3},
        )


def test_no_capacity_no_analysis():
    rbd = RBD([("s", "a"), ("a", "t")])
    assert rbd.capacity == {}
    with pytest.raises(ValueError, match="No node has a capacity"):
        rbd.system_capacity({"a": 0.5})


def test_an_invalid_diagram_has_no_capacity_analysis():
    with pytest.warns(UserWarning):
        rbd = RBD(
            [("s", "a"), ("a", "t"), ("a", "b"), ("b", "a")],
            on_infeasible_rbd="warn",
            capacity={"a": 1},
        )
    with pytest.raises(ValueError, match="valid diagram"):
        rbd.system_capacity({"a": 0.5, "b": 0.5})


def test_capacities_are_saved():
    nodes = {("pump", 1): W([100, 2]), ("pump", 2): W([100, 2])}
    rbd = NonRepairableRBD(
        [("s", n) for n in nodes] + [(n, "t") for n in nodes],
        nodes,
        capacity={("pump", 1): 2.5, ("pump", 2): math.inf},
    )
    back = NonRepairableRBD.from_json(rbd.to_json())
    assert back.capacity == rbd.capacity
    np.testing.assert_allclose(
        back.capacity_distribution(50).probabilities,
        rbd.capacity_distribution(50).probabilities,
    )
    unit = {"reliability": E([0.1]), "repairability": E([1.0])}
    plant = RepairableRBD(
        [("s", "a"), ("a", "t")], {"a": unit}, capacity={"a": 7}
    )
    assert RepairableRBD.from_dict(plant.to_dict()).capacity == {"a": 7.0}
    # Without capacities nothing is written, and older files still load.
    plain = RepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
    saved = plain.to_dict()
    assert saved["capacity"] is None
    del saved["capacity"]
    assert RepairableRBD.from_dict(saved).capacity == {}


def test_the_reduced_diagram_pickles():
    rbd = RBD(BRIDGE, capacity={1: 1, 2: 1})
    rbd.system_capacity({v: 0.5 for v in range(1, 6)})
    copy = pickle.loads(pickle.dumps(rbd))
    np.testing.assert_allclose(
        copy.system_capacity({v: 0.5 for v in range(1, 6)}).probabilities,
        rbd.system_capacity({v: 0.5 for v in range(1, 6)}).probabilities,
    )


# --- Multi-state components (#98) --------------------------------------------


def enumerated_levels(edges, k, outcomes):
    """The capacity distribution by enumerating every component's levels:
    ``outcomes`` maps each component to its ``(level, probability)``
    pairs, 0 being failed."""
    G = nx.DiGraph(edges)
    (source,) = [n for n in G if G.in_degree(n) == 0]
    (sink,) = [n for n in G if G.out_degree(n) == 0]
    components = sorted(outcomes, key=str)
    order = list(nx.topological_sort(G))
    out: dict = {}
    for states in itertools.product(*(outcomes[c] for c in components)):
        level_of = {c: level for c, (level, _) in zip(components, states)}
        chance = math.prod(p for _, p in states)
        reached = {source: True}
        for v in order[1:]:
            fed = sum(reached[u] for u in G.predecessors(v)) >= k.get(v, 1)
            reached[v] = fed and (v == sink or level_of[v] > 0)
        level = 0.0
        if reached[sink]:
            flow = nx.DiGraph()
            for v in G:
                if reached[v]:
                    limit = level_of.get(v, math.inf)
                    arc = {} if math.isinf(limit) else {"capacity": limit}
                    flow.add_edge((v, 0), (v, 1), **arc)
            for a, b in G.edges:
                if reached[a] and reached[b]:
                    flow.add_edge((a, 1), (b, 0))
            try:
                level = float(
                    nx.maximum_flow_value(flow, (source, 0), (sink, 1))
                )
            except nx.NetworkXUnbounded:
                level = math.inf
        level = engine.tidy(level)
        out[level] = out.get(level, 0.0) + chance
    levels = sorted(level for level, p in out.items() if p > 0)
    return np.array(levels), np.array([out[level] for level in levels])


def test_multi_level_components_match_enumeration():
    # The bridge with pumps that work at full or reduced output.
    capacity = {
        1: {10: 0.7, 5: 0.3},
        2: {20: 0.5, 12: 0.25, 6: 0.25},
        3: 5,
        4: {15: 0.9, 7.5: 0.1},
        5: 10,
    }
    works = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.95}
    outcomes = {}
    for c, spec in capacity.items():
        levels = spec if isinstance(spec, dict) else {spec: 1.0}
        outcomes[c] = [(0.0, 1 - works[c])] + [
            (float(level), works[c] * share) for level, share in levels.items()
        ]
    got = exact(BRIDGE, {}, capacity, works)
    assert_same(got, enumerated_levels(BRIDGE, {}, outcomes))
    # And with a node voting 2-out-of-2 at the output.
    assert_same(
        exact(BRIDGE, {"t": 2}, capacity, works),
        enumerated_levels(BRIDGE, {"t": 2}, outcomes),
    )


@pytest.mark.parametrize("seed", range(3))
def test_random_multi_level_diagrams_match_enumeration(seed):
    rng = random.Random(100 + seed)
    checked = 0
    while checked < 10:
        size = rng.randint(2, 6)
        edges = set()
        for i in range(size):
            for j in range(i + 1, size):
                if rng.random() < 0.4:
                    edges.add((i, j))
        for i in range(size):
            if not any(b == i for _, b in edges) or rng.random() < 0.2:
                edges.add(("s", i))
            if not any(a == i for a, _ in edges) or rng.random() < 0.2:
                edges.add((i, "t"))
        graph = nx.DiGraph(sorted(edges, key=str))
        k = {
            v: rng.randint(2, graph.in_degree(v))
            for v in list(range(size)) + ["t"]
            if graph.in_degree(v) >= 2 and rng.random() < 0.3
        }
        try:
            RBD(graph.edges, k=k)
        except ValueError:
            continue
        capacity: dict = {}
        outcomes = {}
        works = {v: rng.uniform(0.1, 0.9) for v in range(size)}
        for v in range(size):
            count = rng.randint(1, 3)
            levels = rng.sample([1, 2, 3, 0.5, 4.5], count)
            shares = [rng.random() + 0.1 for _ in levels]
            total = sum(shares)
            spec = {
                level: share / total for level, share in zip(levels, shares)
            }
            capacity[v] = spec
            outcomes[v] = [(0.0, 1 - works[v])] + [
                (float(level), works[v] * share)
                for level, share in spec.items()
            ]
        got = exact(list(graph.edges), k, capacity, works)
        assert_same(got, enumerated_levels(graph.edges, k, outcomes))
        checked += 1


def test_a_single_level_is_the_binary_capacity():
    works = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.95}
    binary = {1: 10, 2: 20, 3: 5, 4: 15, 5: 10}
    single = {node: {level: 1.0} for node, level in binary.items()}
    a = exact(BRIDGE, {}, binary, works)
    b = exact(BRIDGE, {}, single, works)
    np.testing.assert_array_equal(a.levels, b.levels)
    np.testing.assert_allclose(a.probabilities, b.probabilities, rtol=1e-14)


def test_levels_scale_with_the_reliability_over_time():
    pump = W([1000, 2])
    rbd = NonRepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": pump},
        capacity={"p": {100: 0.75, 60: 0.25}},
    )
    t = np.array([100.0, 1000.0])
    capacity = rbd.capacity_distribution(t)
    R = pump.sf(t)
    np.testing.assert_allclose(
        capacity.probabilities, [1 - R, 0.25 * R, 0.75 * R]
    )
    assert capacity.levels.tolist() == [0.0, 60.0, 100.0]


def test_degrading_node_stage_probabilities():
    # Identical exponential stages: the stage at t is Poisson(rate * t).
    rate = 1 / 300
    node = DegradingNode([(100, E([rate])), (50, E([rate])), (20, E([rate]))])
    t = np.array([0.0, 100.0, 500.0, 1500.0])
    stages = node.stage_probabilities(t)
    for j in range(3):
        poisson = (rate * t) ** j * np.exp(-rate * t) / math.factorial(j)
        np.testing.assert_allclose(stages[j], poisson, atol=1e-12)
    np.testing.assert_allclose(stages.sum(axis=0), node.sf(t), atol=1e-12)
    # Distinct rates: the hypoexponential, to the convolution's accuracy.
    l1, l2 = 1 / 300, 1 / 200
    node = DegradingNode([(100, E([l1])), (50, E([l2]))])
    stages = node.stage_probabilities(t)
    np.testing.assert_allclose(stages[0], np.exp(-l1 * t), atol=1e-12)
    second = l1 / (l2 - l1) * (np.exp(-l1 * t) - np.exp(-l2 * t))
    np.testing.assert_allclose(stages[1], second, atol=1e-6)
    # A scalar time gives one value per stage.
    assert node.stage_probabilities(500.0).shape == (2,)
    # Before time 0 it is new: in its first stage.
    np.testing.assert_allclose(node.stage_probabilities(-5.0), [1.0, 0.0])


def test_a_one_stage_degrading_node_is_a_binary_component():
    model = W([100, 2])
    node = DegradingNode([(10, model)])
    assert node.sf(50) == pytest.approx(float(model.sf(50)), abs=1e-15)
    assert node.ff(50) == pytest.approx(float(model.ff(50)), abs=1e-15)
    assert node.mean() == pytest.approx(float(model.mean()))
    staged = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": node, "b": W([80, 1.5])},
        capacity={"b": 4},
    ).capacity_distribution(60)
    plain = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": model, "b": W([80, 1.5])},
        capacity={"a": 10, "b": 4},
    ).capacity_distribution(60)
    np.testing.assert_array_equal(staged.levels, plain.levels)
    np.testing.assert_allclose(staged.probabilities, plain.probabilities)


def test_a_degrading_node_in_a_system():
    pump = DegradingNode([(100, W([1000, 2])), (50, E([1 / 500]))])
    rbd = NonRepairableRBD(
        [("s", "pump"), ("pump", "pipe"), ("pipe", "t")],
        {"pump": pump, "pipe": E([1e-5])},
        capacity={"pipe": 80},
    )
    t = np.array([200.0, 1000.0])
    capacity = rbd.capacity_distribution(t)
    stages = pump.stage_probabilities(t)
    pipe = np.exp(-1e-5 * t)
    assert capacity.levels.tolist() == [0.0, 50.0, 80.0]
    np.testing.assert_allclose(capacity.probabilities[1], stages[1] * pipe)
    np.testing.assert_allclose(capacity.probabilities[2], stages[0] * pipe)
    np.testing.assert_allclose(capacity.meets(1e-9), rbd.sf(t))
    # Forced working, it is in one of its stages; forced broken, down.
    working = rbd.capacity_distribution(1000.0, working_nodes=["pump"])
    share = stages[:, 1] / stages[:, 1].sum()
    np.testing.assert_allclose(
        working.probabilities,
        [1 - pipe[1], share[1] * pipe[1], share[0] * pipe[1]],
    )
    broken = rbd.capacity_distribution(1000.0, broken_nodes=["pump"])
    assert broken.levels.tolist() == [0.0]
    # A capacity given for the node takes the place of its stages'.
    flat = NonRepairableRBD(
        [("s", "pump"), ("pump", "t")], {"pump": pump}, capacity={"pump": 7}
    ).capacity_distribution(1000.0)
    assert flat.levels.tolist() == [0.0, 7.0]


def test_degrading_node_checks_its_stages():
    with pytest.raises(ValueError, match="at least one stage"):
        DegradingNode([])
    with pytest.raises(ValueError, match="pair"):
        DegradingNode([W([100, 2])])
    with pytest.raises(ValueError, match="positive"):
        DegradingNode([(0, W([100, 2]))])
    with pytest.raises(ValueError, match="number"):
        DegradingNode([("full", W([100, 2]))])


def test_a_nested_rbd_brings_its_own_capacity():
    # A pump skid of two pumps, as a node in series with a pipe, has the
    # distribution of the same diagram drawn flat.
    pump = W([1000, 2])
    skid = NonRepairableRBD(
        [("in", "p1"), ("in", "p2"), ("p1", "out"), ("p2", "out")],
        {"p1": pump, "p2": pump},
        capacity={"p1": 40, "p2": 60},
    )
    nested = NonRepairableRBD(
        [("s", "skid"), ("skid", "pipe"), ("pipe", "t")],
        {"skid": skid, "pipe": E([1e-4])},
        capacity={"pipe": 80},
    )
    flat = NonRepairableRBD(
        [
            ("s", "p1"),
            ("s", "p2"),
            ("p1", "pipe"),
            ("p2", "pipe"),
            ("pipe", "t"),
        ],
        {"p1": pump, "p2": pump, "pipe": E([1e-4])},
        capacity={"p1": 40, "p2": 60, "pipe": 80},
    )
    t = np.array([300.0, 900.0])
    a = nested.capacity_distribution(t)
    b = flat.capacity_distribution(t)
    np.testing.assert_array_equal(a.levels, b.levels)
    np.testing.assert_allclose(a.probabilities, b.probabilities)
    # Given a capacity of its own in the outer RBD, the nested one is a
    # single component.
    single = NonRepairableRBD(
        [("s", "skid"), ("skid", "t")], {"skid": skid}, capacity={"skid": 5}
    ).capacity_distribution(300.0)
    assert single.levels.tolist() == [0.0, 5.0]
    # A nested RBD with no capacities limits nothing.
    plain = NonRepairableRBD([("in", "x"), ("x", "out")], {"x": pump})
    outer = NonRepairableRBD(
        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
        {"x": plain, "y": pump},
        capacity={"y": 3},
    ).capacity_distribution(300.0)
    assert outer.levels.tolist() == [0.0, 3.0, math.inf]


def test_repairable_multi_state_nodes_in_the_long_run():
    unit = {"reliability": E([0.1]), "repairability": E([1.0])}
    a = 10 / 11
    # Levels while it is up.
    plant = RepairableRBD(
        [("s", "x"), ("x", "t")],
        {"x": unit},
        capacity={"x": {100: 0.8, 50: 0.2}},
    )
    np.testing.assert_allclose(
        plant.capacity_distribution().probabilities,
        [1 - a, 0.2 * a, 0.8 * a],
    )
    # A nested RBD with capacities, against the flat diagram.
    inner = RepairableRBD(
        [("s", "p"), ("s", "q"), ("p", "t"), ("q", "t")],
        {"p": unit, "q": unit},
        capacity={"p": 1, "q": 2},
    )
    nested = RepairableRBD(
        [("s", "inner"), ("inner", "v"), ("v", "t")],
        {"inner": inner, "v": unit},
        capacity={"v": 2.5},
    )
    flat = RepairableRBD(
        [("s", "p"), ("s", "q"), ("p", "v"), ("q", "v"), ("v", "t")],
        {"p": unit, "q": unit, "v": unit},
        capacity={"p": 1, "q": 2, "v": 2.5},
    )
    x, y = nested.capacity_distribution(), flat.capacity_distribution()
    np.testing.assert_array_equal(x.levels, y.levels)
    np.testing.assert_allclose(x.probabilities, y.probabilities)
    # A degrading component: down 1 - A of the time, and up in each stage
    # in proportion to its mean time.
    stages = DegradingNode([(100, W([100, 2])), (50, E([1 / 50]))])
    degrading = RepairableRBD(
        [("s", "x"), ("x", "t")],
        {"x": {"reliability": stages, "repairability": E([0.1])}},
    )
    first = 100 * math.gamma(1.5)
    cycle = first + 50 + 10
    np.testing.assert_allclose(
        degrading.capacity_distribution().probabilities,
        [10 / cycle, 50 / cycle, first / cycle],
    )
    assert degrading.mean_availability() == pytest.approx(1 - 10 / cycle)
    # On a schedule, its time in each stage has no exact value here.
    scheduled = RepairableRBD(
        [("s", "x"), ("x", "t")],
        {
            "x": {
                "reliability": stages,
                "repairability": E([0.1]),
                "preventive": {"interval": 100.0},
            }
        },
    )
    with pytest.raises(NotImplementedError, match="stages"):
        scheduled.capacity_distribution()


def test_a_stage_that_may_never_end():
    # Units whose second stage never ends (30%) are caught in it for good;
    # the rest fail, are renewed, and sooner or later are caught too.
    forever = surv.Weibull.from_params([100, 2], p=0.7)
    stages = DegradingNode([(100, E([0.01])), (50, forever)])
    np.testing.assert_allclose(stages.stage_fractions(), [0.0, 1.0])
    assert stages.mean() == math.inf
    rbd = RepairableRBD(
        [("s", "x"), ("x", "t")],
        {"x": {"reliability": stages, "repairability": E([0.5])}},
    )
    assert rbd.mean_availability() == pytest.approx(1.0)
    capacity = rbd.capacity_distribution()
    assert capacity.levels.tolist() == [50.0]
    # And a cold standby with such a unit is up for good in the long run.
    assert rbd.node_availability()["x"] == pytest.approx(1.0)


def test_common_cause_members_need_given_capacities():
    stages = DegradingNode([(2, W([100, 2])), (1, W([100, 2]))])
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": stages, "b": stages},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        rbd.capacity_distribution(50)


@pytest.mark.parametrize(
    "levels, message",
    [
        ({}, "empty"),
        ({10: 0.5}, "add up to 1"),
        ({10: 0.6, 5: 0.6}, "add up to 1"),
        ({0: 1.0}, "positive"),
        ({10: 0.0, 5: 1.0}, r"\(0, 1\]"),
        ({10: "half", 5: 0.5}, "number"),
    ],
)
def test_invalid_capacity_levels(levels, message):
    with pytest.raises(ValueError, match=message):
        RBD([("s", "a"), ("a", "t")], capacity={"a": levels})


def test_capacity_levels_are_kept_in_order_and_saved():
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": W([100, 2])},
        capacity={"a": {100: 0.75, 40: 0.25}},
    )
    assert list(rbd.capacity["a"]) == [40.0, 100.0]
    back = NonRepairableRBD.from_json(rbd.to_json())
    assert back.capacity == rbd.capacity
    stages = DegradingNode([(100, W([1000, 2])), (50, E([1 / 500]))])
    staged = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": stages})
    again = NonRepairableRBD.from_json(staged.to_json())
    assert isinstance(again.reliabilities["a"], DegradingNode)
    assert again.reliabilities["a"].capacities == (100.0, 50.0)
    np.testing.assert_allclose(
        again.capacity_distribution(700.0).probabilities,
        staged.capacity_distribution(700.0).probabilities,
    )


def test_system_capacity_needs_the_capacity_of_every_node():
    stages = DegradingNode([(2, W([100, 2])), (1, W([100, 2]))])
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": stages})
    with pytest.raises(ValueError, match="capacity_distribution"):
        rbd.system_capacity({"a": 0.5})
    # With a capacity given for the node, its probability describes it.
    given = NonRepairableRBD(
        [("s", "a"), ("a", "t")], {"a": stages}, capacity={"a": 3}
    )
    assert given.system_capacity({"a": 0.5}).levels.tolist() == [0.0, 3.0]
