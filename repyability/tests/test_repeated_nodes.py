"""Tests for repeated nodes: one component drawn in several places.

A repeated node stays where it is drawn, and every appearance is the one
component: it works, or has failed, everywhere at once. The reference is the
structure function enumerated over the components' states, with each
repeated node taking the state of the component it repeats, and reachability
worked out on the diagram exactly as drawn.
"""

import itertools

import networkx as nx
import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import FaultTree, NonRepairableRBD, PerfectReliability
from repyability.tests.test_rbd_modular import random_diagram, unreduced

fail = FixedEventProbability.from_params


def enumerated(edges, k, components, aliases):
    """Every combination of the components' states, and whether the system
    works in each: a node is reached when its component works and at least
    k of its inputs are reached."""
    states = np.array(
        list(itertools.product([False, True], repeat=len(components))),
        dtype=bool,
    ).reshape(-1, len(components))
    column = {c: states[:, i] for i, c in enumerate(components)}
    for copy, original in aliases.items():
        column[copy] = column[original]
    graph = nx.DiGraph(edges)
    reached: dict = {}
    for v in nx.topological_sort(graph):
        inputs = list(graph.predecessors(v))
        works = column.get(v, np.ones(len(states), dtype=bool))
        if not inputs:
            reached[v] = works
            continue
        count = np.sum([reached[u] for u in inputs], axis=0)
        reached[v] = works & (count >= k.get(v, 1))
    return states, reached["t"]


def minimal(sets):
    return {s for s in sets if not any(other < s for other in sets)}


def probability(states, works, p, components, fixed=None):
    """The probability that the system works (``fixed``: components held
    working (True) or failed (False))."""
    fixed = fixed or {}
    weight = np.ones(len(states))
    keep = np.ones(len(states), dtype=bool)
    for i, c in enumerate(components):
        if c in fixed:
            keep &= states[:, i] == fixed[c]
            continue
        weight *= np.where(states[:, i], p[c], 1 - p[c])
    return float(np.sum(weight[keep & works]))


def random_repeats(rng, n):
    """A random diagram over nodes 0..n-1 in which one or two nodes are
    drawn again elsewhere: each chosen node is a repeat of another."""
    edges, k = random_diagram(rng, n)
    nodes = list(range(n))
    copies = rng.choice(
        n, size=int(rng.integers(1, min(3, n - 1) + 1)), replace=False
    )
    originals = [v for v in nodes if v not in copies]
    aliases = {int(c): int(rng.choice(originals)) for c in copies}
    return edges, k, originals, aliases


# -- the counterexample -----------------------------------------------------


def test_a_repeated_node_adds_no_paths():
    # X is drawn twice: before A, and after Y on the way to B. Joining the
    # two drawings into one node would add the path {X, B}, leaving Y out.
    rbd = NonRepairableRBD(
        [
            ("s", "X"),
            ("X", "A"),
            ("A", "t"),
            ("s", "Y"),
            ("Y", "X2"),
            ("X2", "B"),
            ("B", "t"),
        ],
        {
            "X": fail(0.1),
            "X2": "X",
            "A": fail(0.1),
            "Y": fail(0.1),
            "B": fail(0.1),
        },
    )
    assert rbd.get_min_path_sets(include_in_out_nodes=False) == {
        frozenset({"X", "A"}),
        frozenset({"X", "Y", "B"}),
    }
    # X works, and A does or both Y and B do.
    assert rbd.sf() == pytest.approx(0.9 * (1 - 0.1 * (1 - 0.81)), rel=1e-15)
    assert rbd.get_min_cut_sets() == {
        frozenset({"X"}),
        frozenset({"A", "Y"}),
        frozenset({"A", "B"}),
    }
    assert rbd.nodes == ["X", "A", "Y", "B"]
    assert rbd.repeated == {"X2": "X"}
    assert rbd.is_system_working(
        {"X": True, "A": False, "Y": True, "B": True}, "p"
    )
    assert not rbd.is_system_working(
        {"X": True, "A": False, "Y": False, "B": True}, "c"
    )


# -- random diagrams against the definition ---------------------------------


@pytest.mark.parametrize("seed", range(200))
def test_random_diagrams_with_repeats_match_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    edges, k, components, aliases = random_repeats(rng, n)
    p = {c: float(rng.uniform(0.3, 0.95)) for c in components}
    models = {c: fail(1 - p[c]) for c in components}
    models.update(aliases)
    rbd = NonRepairableRBD(edges, models, k=k or None)
    states, works = enumerated(edges, k, components, aliases)

    exact = probability(states, works, p, components)
    assert rbd.sf() == pytest.approx(exact, rel=1e-12, abs=1e-15)
    assert rbd.system_probability({c: p[c] for c in components}, method="c")[
        0
    ] == pytest.approx(exact, rel=1e-12, abs=1e-15)
    assert unreduced(
        NonRepairableRBD(edges, models, k=k or None)
    ).sf() == pytest.approx(exact, rel=1e-12, abs=1e-15)

    working = [
        frozenset(c for c, s in zip(components, row) if s) for row in states
    ]
    paths = minimal({w for w, ok in zip(working, works) if ok})
    everyone = frozenset(components)
    cuts = minimal({everyone - w for w, ok in zip(working, works) if not ok})
    if exact == 1.0 and paths == {frozenset()}:
        assert rbd.get_min_cut_sets() == set()
    else:
        assert rbd.get_min_path_sets(include_in_out_nodes=False) == paths
        assert rbd.get_min_cut_sets() == cuts
    for row, ok in zip(states, works):
        status = dict(zip(components, map(bool, row)))
        assert rbd.is_system_working(status, "p") == bool(ok)
        assert rbd.is_system_working(status, "c") == bool(ok)

    # Birnbaum importance, from the definition.
    birnbaum = rbd.birnbaum_importance()
    for c in components:
        up = probability(states, works, p, components, {c: True})
        down = probability(states, works, p, components, {c: False})
        assert birnbaum[c] == pytest.approx(up - down, abs=1e-12)

    # A component none of whose appearances matter is irrelevant.
    relevant = set().union(*paths) if paths else set()
    assert rbd.find_irrelevant_components() == set(components) - relevant


@pytest.mark.parametrize("seed", range(20))
def test_simulated_lifetimes_share_the_components(seed):
    # A repeated node fails when its component does: the simulated
    # lifetimes follow the exact reliability.
    rng = np.random.default_rng(900 + seed)
    edges, k, components, aliases = random_repeats(
        rng, int(rng.integers(3, 7))
    )
    models = {
        c: surv.Weibull.from_params([float(rng.uniform(50, 150)), 2.0])
        for c in components
    }
    models.update(aliases)
    rbd = NonRepairableRBD(edges, models, k=k or None)
    lifetimes = rbd.random(20000, seed=seed)
    finite = lifetimes[np.isfinite(lifetimes)]
    for t in (40.0, 80.0):
        exact = rbd.sf(t)
        simulated = np.mean(lifetimes > t)
        se = np.sqrt(exact * (1 - exact) / len(lifetimes)) + 1e-9
        assert abs(simulated - exact) < 5 * se
    assert len(finite) == len(lifetimes) or rbd.sf(1e9) > 0


def test_the_event_loop_agrees(monkeypatch):
    # With a model the batched sampler cannot replay, random() steps
    # through the failures one by one, over the components.
    edges = [
        ("s", "X"),
        ("X", "A"),
        ("A", "t"),
        ("s", "Y"),
        ("Y", "X2"),
        ("X2", "B"),
        ("B", "t"),
    ]
    unit = surv.Weibull.from_params([100.0, 2.0])
    models = {"X": unit, "X2": "X", "A": unit, "Y": unit, "B": unit}
    rbd = NonRepairableRBD(edges, models)
    monkeypatch.setattr(rbd, "_row_sampler", lambda: None)
    lifetimes = rbd.random(20000, seed=3)
    exact = rbd.sf(60.0)
    se = np.sqrt(exact * (1 - exact) / len(lifetimes))
    assert abs(np.mean(lifetimes > 60.0) - exact) < 5 * se


# -- what used to be rejected ---------------------------------------------


def test_only_a_repeat_is_left():
    # "X" feeds only the output that "X2" (the same component) feeds
    # directly: that appearance can never matter, and the system is the
    # component alone.
    rbd = NonRepairableRBD(
        [("s", "X2"), ("X2", "t"), ("X2", "X"), ("X", "t")],
        {"X": fail(0.25), "X2": "X"},
    )
    assert rbd.sf() == pytest.approx(0.75, rel=1e-15)
    assert rbd.get_min_path_sets(include_in_out_nodes=False) == {
        frozenset({"X"})
    }
    assert rbd.find_irrelevant_components() == set()


def test_a_component_drawn_twice_in_a_row():
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "a2"), ("a2", "t")],
        {"a": fail(0.1), "b": fail(0.2), "a2": "a"},
    )
    assert rbd.sf() == pytest.approx(0.9 * 0.8, rel=1e-15)


def test_forcing_goes_through_the_component():
    rbd = NonRepairableRBD(
        [
            ("s", "X"),
            ("X", "A"),
            ("A", "t"),
            ("s", "Y"),
            ("Y", "X2"),
            ("X2", "B"),
            ("B", "t"),
        ],
        {
            "X": fail(0.1),
            "X2": "X",
            "A": fail(0.1),
            "Y": fail(0.1),
            "B": fail(0.1),
        },
    )
    assert rbd.sf(broken_nodes=["X"]) == 0.0
    assert rbd.sf(working_nodes=["X"]) == pytest.approx(1 - 0.1 * 0.19)
    with pytest.raises(ValueError, match="repeat of node"):
        rbd.sf(working_nodes=["X2"])


def test_saving_keeps_the_repeat():
    rbd = NonRepairableRBD(
        [
            ("s", "X"),
            ("X", "A"),
            ("A", "t"),
            ("s", "Y"),
            ("Y", "X2"),
            ("X2", "B"),
            ("B", "t"),
        ],
        {
            "X": fail(0.1),
            "X2": "X",
            "A": fail(0.1),
            "Y": fail(0.1),
            "B": fail(0.1),
        },
    )
    restored = NonRepairableRBD.from_json(rbd.to_json())
    assert restored.repeated == {"X2": "X"}
    assert restored.sf() == rbd.sf()


def test_a_junction_repeated_in_a_fault_tree_diagram():
    # A fault tree whose shared gate becomes repeated nodes and junctions.
    tree = FaultTree(
        {
            "top": ("and", ["g1", "g2"]),
            "g1": ("or", ["a", "shared"]),
            "g2": ("or", ["b", "shared"]),
            "shared": ("vote", 2, ["x", "y", "z"]),
        },
        {"a": 0.1, "b": 0.2, "x": 0.3, "y": 0.3, "z": 0.3},
    )
    rbd = tree.to_rbd()
    assert set(rbd.repeated.values()) == {"x", "y", "z"}
    assert any(m is PerfectReliability for m in rbd.reliabilities.values())
    assert 1 - rbd.sf() == pytest.approx(tree.top_event_probability())
