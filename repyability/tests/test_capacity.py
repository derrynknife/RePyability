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
