"""Meshed structures (#171, #172): a core's decision diagram built from
counts of reached predecessors, Fussell-Vesely from the diagram rather
than the cut sets, the cut and path sets listed by lookups, a fault tree's
core built from its gates, and a core too meshed to work out at all left to
the simulations, which follow the graph itself."""

import itertools
import time

import numpy as np
import pytest
import surpyval as surv

from repyability import FaultTree, NonRepairableRBD, RepairableRBD
from repyability.rbd import bdd, modular
from repyability.rbd.rbd_graph import RBDGraph
from repyability.tests.catalogue import BRIDGE, too_meshed
from repyability.tests.test_rbd_modular import enumerate_states, random_diagram

W = surv.Weibull.from_params
E = surv.Exponential.from_params


def meshed_dag(n, seed=1, p=0.2):
    """A random acyclic diagram of ``n`` nodes, each joined to a later one
    with probability ``p`` (as in #172)."""
    rng = np.random.default_rng(seed)
    edges = {
        (i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p
    }
    for i in range(n):
        if not any(a == i for a, _ in edges):
            edges.add((i, "t"))
        if not any(b == i for _, b in edges):
            edges.add(("s", i))
    return sorted(edges, key=str)


def ladder(rungs):
    """Two rails with a bridge across them at each rung (#172)."""
    edges, top, bottom = [], "s", "s"
    for i in range(rungs):
        a, b, c, d, e = (f"{x}{i}" for x in "abcde")
        edges += [(top, a), (bottom, b), (a, c), (b, c)]
        edges += [(a, d), (b, e), (c, d), (c, e)]
        top, bottom = d, e
    return sorted(set(edges + [(top, "t"), (bottom, "t")]), key=str)


def graph_of(edges, k=None):
    graph = RBDGraph(edges)
    for node in graph.nodes:
        graph.nodes[node]["k"] = (k or {}).get(node, 1)
    return graph


def test_a_meshed_diagram_needs_few_states(monkeypatch):
    # 35 nodes and 129 edges: the states were sets of reached vertices,
    # millions of them (44 seconds); with counted predecessors, the
    # diagram takes a few hundred thousand steps.
    monkeypatch.setattr(bdd, "STEP_LIMIT", 1_000_000)
    edges = meshed_dag(35)
    rbd = NonRepairableRBD(edges, {i: W([100, 2]) for i in range(35)})
    assert not rbd.structure_check["is_too_meshed"]
    assert 0.99 < rbd.sf(50.0) < 1.0


@pytest.mark.parametrize("seed", range(3))
def test_the_counted_states_decide_the_structure(seed):
    rng = np.random.default_rng(seed)
    for _ in range(60):
        n = int(rng.integers(3, 11))
        edges, k = random_diagram(rng, n)
        try:
            decomposition = modular.decompose(
                graph_of(edges, k), "s", "t", core="bdd"
            )
        except ValueError:
            continue
        nodes = [v for v in range(n) if any(v in e for e in edges)]
        states, works = enumerate_states(edges, k, nodes)
        for row, expected in zip(states, works):
            status = dict(zip(nodes, row))
            assert decomposition.works(status, "p") == expected


def test_fussell_vesely_is_its_definition_on_a_ladder():
    # A minimal cut set holding the node has failed, over the system
    # failing, by every state of a three-rung ladder.
    edges = ladder(3)
    nodes = sorted({v for e in edges for v in e} - {"s", "t"})
    life = W([100, 2])
    rbd = NonRepairableRBD(edges, {v: life for v in nodes})
    fv = rbd.fussell_vesely(60.0)
    cuts = [frozenset(c) for c in rbd.get_min_cut_sets()]
    q = float(life.ff(60.0))
    numerator = dict.fromkeys(nodes, 0.0)
    failing = 0.0
    for row in itertools.product([False, True], repeat=len(nodes)):
        failed = {v for v, f in zip(nodes, row) if f}
        weight = q ** len(failed) * (1 - q) ** (len(nodes) - len(failed))
        if any(c <= failed for c in cuts):
            failing += weight
        for v in nodes:
            if any(v in c and c <= failed for c in cuts):
                numerator[v] += weight
    for v in nodes:
        assert float(fv[v]) == pytest.approx(numerator[v] / failing, rel=1e-12)


def test_fussell_vesely_on_a_long_ladder_is_quick():
    # Thirty rungs took 19 seconds through the union of each node's cut
    # sets; on the core's decision diagram a fraction of one.
    edges = ladder(30)
    nodes = {v for e in edges for v in e} - {"s", "t"}
    rbd = NonRepairableRBD(edges, {v: W([100, 2]) for v in nodes})
    start = time.perf_counter()
    fv = rbd.fussell_vesely(50.0)
    assert time.perf_counter() - start < 10.0
    assert set(fv) == nodes and all(0.0 < float(x) <= 1.0 for x in fv.values())
    # The cut sets themselves are listed quickly too, thousands of them
    # (a hundred rungs' took three minutes).
    start = time.perf_counter()
    assert len(rbd.get_min_cut_sets()) > 1000
    assert time.perf_counter() - start < 10.0


def shared_events(ands, events, seed=0):
    """An OR of ``ands`` ANDs of three events each, drawn from ``events``
    shared ones (#171)."""
    rng = np.random.default_rng(seed)
    names = [f"e{j}" for j in range(events)]
    gates = {
        f"a{i}": ("and", list(rng.choice(names, 3, replace=False)))
        for i in range(ands)
    }
    gates["top"] = ("or", list(gates))
    return gates, names


def test_a_fault_tree_of_shared_events_is_built_quickly():
    # An OR of fifty ANDs over twenty-five shared events: 18 seconds to
    # list its path sets on construction (#171), and no listing now.
    gates, names = shared_events(50, 25)
    start = time.perf_counter()
    tree = FaultTree(gates, {e: 0.05 for e in names}, top="top")
    q = tree.ff()
    tree.birnbaum_importance()
    assert time.perf_counter() - start < 10.0
    ands = {frozenset(g[1]) for g in gates.values() if g[0] == "and"}
    minimal = {s for s in ands if not any(o < s for o in ands)}
    assert set(tree.minimal_cut_sets()) == minimal
    assert 0.0 < q < 1.0


def test_a_fault_tree_of_shared_events_is_exact():
    # Every state of ten shared events, weighed.
    gates, names = shared_events(14, 10, seed=3)
    rng = np.random.default_rng(5)
    q = {e: float(rng.uniform(0.01, 0.3)) for e in names}
    tree = FaultTree(gates, q, top="top")
    ands = [set(g[1]) for g in gates.values() if g[0] == "and"]
    expected = 0.0
    for row in itertools.product([False, True], repeat=len(names)):
        occurred = {e for e, x in zip(names, row) if x}
        if any(a <= occurred for a in ands):
            weight = 1.0
            for e in names:
                weight *= q[e] if e in occurred else 1.0 - q[e]
            expected += weight
    assert tree.ff() == pytest.approx(expected, rel=1e-12)


def _graph_and_decomposition(rng):
    n = int(rng.integers(3, 11))
    edges, k = random_diagram(rng, n)
    graph = graph_of(edges, k)
    nodes = [v for v in range(n) if v in graph.nodes]
    aliases = {}
    if rng.random() < 0.3 and len(nodes) > 3:
        a, b = rng.choice(nodes, 2, replace=False)
        aliases = {int(a): int(b)}
    # (A repeated node is no junction: a RepairableRBD, whose junctions are
    # folded, has no repeats.)
    shared = set(aliases) | set(aliases.values())
    perfect = {v for v in nodes if v not in shared and rng.random() < 0.2}
    whole = modular.decompose(graph, "s", "t", aliases=aliases)
    exact = modular.fold(whole, perfect)
    stand_in = modular.GraphStructure(graph, "s", "t", aliases, perfect, "")
    return nodes, perfect, aliases, exact, stand_in


def test_the_graph_stand_in_works_and_lasts_as_the_structure():
    rng = np.random.default_rng(4)
    checked = 0
    while checked < 200:
        try:
            nodes, perfect, aliases, exact, stand_in = (
                _graph_and_decomposition(rng)
            )
        except ValueError:
            continue
        components = sorted(
            {aliases.get(v, v) for v in nodes if v not in perfect}
        )
        for row in itertools.product([False, True], repeat=len(components)):
            status = dict(zip(components, row))
            assert stand_in.works(status) == exact.works(status, "p")
        size = 50
        lifetimes = {c: rng.exponential(1.0, size) for c in components}
        np.testing.assert_array_equal(
            stand_in.lifetime(lifetimes, size),
            exact.lifetime(lifetimes, size),
        )
        checked += 1


def test_a_core_too_meshed_is_simulated_as_one_worked_out():
    life = W([100, 2])
    models = {"a": life, "b": life, "c": E([0.002]), "d": life, "e": life}
    meshed = too_meshed(lambda: NonRepairableRBD(BRIDGE, models))
    worked = NonRepairableRBD(BRIDGE, models)
    np.testing.assert_array_equal(
        meshed.random(5000, seed=3), worked.random(5000, seed=3)
    )
    assert meshed.mean(method="simulate", seed=2) == worked.mean(
        method="simulate", seed=2
    )
    with pytest.raises(NotImplementedError, match="too meshed .* random"):
        meshed.sf(50.0)
    with pytest.raises(NotImplementedError, match="too meshed"):
        meshed.find_irrelevant_components()
    spec = {"reliability": life, "repairability": E([0.5])}
    meshed = too_meshed(
        lambda: RepairableRBD(BRIDGE, {n: dict(spec) for n in "abcde"})
    )
    worked = RepairableRBD(BRIDGE, {n: dict(spec) for n in "abcde"})
    run = meshed.availability(300.0, mc_samples=200, seed=5, engine="python")
    same = worked.availability(300.0, mc_samples=200, seed=5, engine="python")
    np.testing.assert_array_equal(run.uptimes, same.uptimes)
    # The compiled engine runs only a structure worked out.
    assert meshed.analysis_routes()["mean_availability"].route == "refused"
    with pytest.raises(NotImplementedError, match="availability, cost"):
        meshed.mean_availability()


def test_a_core_beyond_the_budget_is_given_up_on(monkeypatch):
    monkeypatch.setattr(bdd, "STEP_LIMIT", 50)
    edges = meshed_dag(30)
    with pytest.raises(bdd.TooLarge, match="more than 50 steps"):
        modular.decompose(graph_of(edges), "s", "t", core="bdd")
    rbd = NonRepairableRBD(edges, {i: W([100, 2]) for i in range(30)})
    assert rbd.structure_check["is_too_meshed"]
    lives = rbd.random(1000, seed=1)
    assert np.all(lives > 0)
