"""The core decided by a binary decision diagram built from its graph (#102),
held to the path-set route on random diagrams and through the public
methods, and run where the path sets are too many to list."""

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import RBD, NonRepairableRBD, RepairableRBD
from repyability.rbd import bdd, modular
from repyability.rbd.modular import decompose
from repyability.tests.test_rbd_modular import random_diagram

W, E = surv.Weibull.from_params, surv.Exponential.from_params


def both(graph, aliases=None):
    """The path-set and graph decompositions of ``graph``."""
    return (
        decompose(graph, "s", "t", aliases=aliases, core="paths"),
        decompose(graph, "s", "t", aliases=aliases, core="bdd"),
    )


def assert_same(paths, graph, nodes, rng):
    assert graph.from_graph and not paths.from_graph
    assert graph.nodes == paths.nodes
    assert graph.path_sets() == paths.path_sets()
    assert graph.cut_sets() == paths.cut_sets()
    p = {n: rng.uniform(0, 1, 3) for n in nodes}
    for a, b in zip(
        paths.probabilities(p, shape=3), graph.probabilities(p, shape=3)
    ):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-15)
    p = {n: float(rng.uniform()) for n in nodes}
    q = {n: 1.0 - v for n, v in p.items()}
    works, fails, gradient = paths.value_and_gradient(p, q)
    g_works, g_fails, g_gradient = graph.value_and_gradient(p, q)
    assert g_works == pytest.approx(works, rel=1e-12)
    assert g_fails == pytest.approx(fails, rel=1e-12, abs=1e-15)
    for n in set(gradient) | set(g_gradient):
        assert g_gradient.get(n, 0.0) == pytest.approx(
            gradient.get(n, 0.0), rel=1e-9, abs=1e-14
        )
    for _ in range(25):
        status = {n: bool(rng.random() < 0.6) for n in nodes}
        for method in ("p", "c"):
            assert graph.works(status, method) == paths.works(status)
    lives = {n: rng.exponential(1.0, 5) for n in nodes}
    np.testing.assert_array_equal(
        graph.lifetime(lives, 5), paths.lifetime(lives, 5)
    )


def test_random_diagrams_match_the_path_set_route():
    cores = 0
    for seed in range(250):
        rng = np.random.default_rng(seed)
        edges, k = random_diagram(rng, int(rng.integers(3, 12)))
        rbd = RBD(edges, k=k, on_infeasible_rbd="ignore")
        if not rbd.structure_check["is_valid"]:
            continue
        try:
            paths, graph = both(rbd.G)
        except ValueError:
            continue
        cores += paths.core is not None
        if paths.core is None:
            # Nothing left to decide: the reduction did it all.
            assert graph.core is None and not graph.from_graph
            continue
        assert_same(paths, graph, sorted(rbd.nodes, key=str), rng)
    assert cores > 150


def test_components_drawn_in_several_places():
    checked = 0
    for seed in range(150):
        rng = np.random.default_rng(10_000 + seed)
        edges, k = random_diagram(rng, int(rng.integers(4, 10)))
        names = sorted({n for e in edges for n in e} - {"s", "t"}, key=str)
        models = {
            n: FixedEventProbability.from_params(float(rng.uniform(0.05, 0.5)))
            for n in names
        }
        for _ in range(int(rng.integers(1, 3))):
            a, b = rng.choice(len(names), 2, replace=False)
            if not isinstance(models[names[a]], str):
                models[names[b]] = names[a]
        if any(
            isinstance(models[models[n]], str)
            for n in names
            if isinstance(models[n], str)
        ):
            continue
        try:
            rbd = NonRepairableRBD(edges, models, k=k)
            paths, graph = both(rbd.G, rbd._component_aliases())
        except ValueError:
            continue
        if paths.core is None:
            continue
        components = sorted(
            {rbd._component_aliases().get(n, n) for n in names}, key=str
        )
        assert_same(paths, graph, components, rng)
        checked += 1
    assert checked > 40


def bridges(n):
    """``n`` bridges in series: 4 ** n minimal path sets."""
    edges, before = [], "s"
    for b in range(n):
        a, b_, c, d, e = (f"{x}{b}" for x in "abcde")
        after = f"j{b}" if b < n - 1 else "t"
        edges += [
            (before, a),
            (before, b_),
            (a, c),
            (b_, c),
            (a, d),
            (b_, e),
            (c, d),
            (c, e),
            (d, after),
            (e, after),
        ]
        before = after
    return edges


def test_a_chain_of_bridges_too_meshed_to_list_its_paths(monkeypatch):
    # Forty bridges in series have 4 ** 40 minimal path sets; the decision
    # diagram grows with them one by one.
    p = 0.9
    bridge = 2 * p**2 + 2 * p**3 - 5 * p**4 + 2 * p**5
    monkeypatch.setattr(modular, "CORE_METHOD", "bdd")
    edges = bridges(40)
    names = {n for e in edges for n in e} - {"s", "t"}
    joints = {n for n in names if n.startswith("j")}
    rbd = NonRepairableRBD(
        edges,
        {
            n: FixedEventProbability.from_params(0.0 if n in joints else 0.1)
            for n in names
        },
    )
    steps, _ = rbd._decomposition().core_plan()
    assert len(steps) < 40 * 20
    assert rbd.sf() == pytest.approx(bridge**40, rel=1e-12)
    assert rbd.ff() == pytest.approx(1.0 - bridge**40, rel=1e-12)


def test_the_public_methods_agree(monkeypatch):
    # A bridge of bridges, with a 2-out-of-3 vote and a component in two
    # places, through the methods a user calls.
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "d"),
        ("b", "e"),
        ("c", "d"),
        ("c", "e"),
        ("d", "v"),
        ("e", "v"),
        ("s", "f"),
        ("f", "v"),
        ("v", "g"),
        ("g", "t"),
        ("s", "a2"),
        ("a2", "t"),
    ]
    models = {
        "a": W([100.0, 2.0]),
        "b": W([120.0, 1.5]),
        "c": E([0.01]),
        "d": W([90.0, 3.0]),
        "e": W([150.0, 2.5]),
        "f": E([0.02]),
        "v": FixedEventProbability.from_params(0.0),
        "g": W([300.0, 2.0]),
        "a2": "a",
    }

    def build(core):
        monkeypatch.setattr(modular, "CORE_METHOD", core)
        return NonRepairableRBD(edges, models, k={"v": 2})

    paths, graph = build("paths"), build("bdd")
    assert graph._decomposition().from_graph
    t = np.array([10.0, 50.0, 100.0])
    np.testing.assert_allclose(graph.sf(t), paths.sf(t), rtol=1e-12)
    for method in ("birnbaum_importance", "criticality_importance"):
        a, b = getattr(paths, method)(50.0), getattr(graph, method)(50.0)
        for node in a:
            assert b[node] == pytest.approx(a[node], rel=1e-9, abs=1e-15)
    assert graph.mean() == pytest.approx(paths.mean(), rel=1e-9)
    assert graph.get_min_path_sets() == paths.get_min_path_sets()
    assert graph.get_min_cut_sets() == paths.get_min_cut_sets()
    np.testing.assert_array_equal(
        graph.random(500, seed=3), paths.random(500, seed=3)
    )
    repairable = {
        n: (
            m
            if isinstance(m, str)
            else {"reliability": m, "repairability": E([0.5])}
        )
        for n, m in models.items()
        if n != "a2"
    }

    def repair(core):
        monkeypatch.setattr(modular, "CORE_METHOD", core)
        return RepairableRBD(
            [e for e in edges if "a2" not in e], repairable, k={"v": 2}
        )

    a, b = repair("paths"), repair("bdd")
    assert b.mean_availability() == pytest.approx(
        a.mean_availability(), rel=1e-12
    )
    run_a = a.availability(500.0, mc_samples=20, seed=1, engine="python")
    run_b = b.availability(500.0, mc_samples=20, seed=1, engine="python")
    np.testing.assert_array_equal(run_b.availability, run_a.availability)


def test_the_diagram_is_reduced():
    # No decision whose outcomes agree, no two equal decisions, and no
    # variable decided twice on the way from the root to an outcome.
    for seed in range(60):
        edges, k = random_diagram(np.random.default_rng(seed), 9)
        rbd = RBD(edges, k=k, on_infeasible_rbd="ignore")
        if not rbd.structure_check["is_valid"]:
            continue
        try:
            graph = decompose(rbd.G, "s", "t", core="bdd")
        except ValueError:
            continue
        if not graph.from_graph:
            continue
        steps, root = graph.core_plan()
        assert all(a != i for _, a, i in steps)
        assert len(set(steps)) == len(steps)
        stack = [(root, frozenset())]
        while stack:
            slot, seen = stack.pop()
            if slot <= bdd.WORK:
                continue
            pivot, active, inactive = steps[slot - 2]
            assert pivot not in seen
            stack += [(active, seen | {pivot}), (inactive, seen | {pivot})]


def test_the_option_is_checked():
    rbd = RBD([("s", "a"), ("a", "t")])
    with pytest.raises(ValueError, match="'paths' or 'bdd'"):
        decompose(rbd.G, "s", "t", core="zdd")


def test_the_orders_are_topological():
    for seed in range(40):
        edges, k = random_diagram(np.random.default_rng(seed), 10)
        rbd = RBD(edges, k=k, on_infeasible_rbd="ignore")
        if not rbd.structure_check["is_valid"]:
            continue
        nodes = [n for n in rbd.G.nodes if n not in ("s", "t")]
        pred = {n: set(rbd.G.predecessors(n)) for n in rbd.G.nodes}
        succ = {n: set(rbd.G.successors(n)) for n in rbd.G.nodes}
        index = {n: i for i, n in enumerate(sorted(nodes, key=str))}
        number = {**index, "s": -1, "t": -2}
        p = {number[n]: {number[u] for u in us} for n, us in pred.items()}
        s = {number[n]: {number[u] for u in us} for n, us in succ.items()}
        for sequence in (
            bdd.order(index.values(), p, s, -1, -2),
            bdd._breadth_first(sorted(index.values()), p, s, -1),
            bdd._greedy(sorted(index.values()), p, s, -1, -2),
        ):
            assert sorted(sequence) == sorted(index.values())
            position = {v: i for i, v in enumerate(sequence)}
            for v in sequence:
                assert all(u == -1 or position[u] < position[v] for u in p[v])
