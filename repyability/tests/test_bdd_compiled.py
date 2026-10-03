"""The decision diagram's construction and replay compiled with numba
(``_bdd_kernel``): the same plan, step for step, as ``bdd.build`` makes in
Python, and the same values, to the last bit."""

import numpy as np
import pytest
from surpyval import FixedEventProbability

from repyability import RBD, NonRepairableRBD
from repyability.rbd import _compiled, bdd, modular
from repyability.rbd.modular import decompose
from repyability.tests.test_bdd_core import bridges
from repyability.tests.test_rbd_modular import random_diagram

pytestmark = pytest.mark.skipif(
    not _compiled.available(), reason="numba is not installed"
)


def grid(rows, cols):
    """A meshed grid, from the input to the output, with links downward."""

    def name(r, c):
        return f"n{r}_{c}"

    edges = [("s", name(r, 0)) for r in range(rows)]
    edges += [(name(r, cols - 1), "t") for r in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((name(r, c), name(r, c + 1)))
            if r + 1 < rows:
                edges.append((name(r, c), name(r + 1, c)))
    return edges


@pytest.fixture
def builds(monkeypatch):
    """Each core ``bdd.build`` is given, and its plan in Python."""
    seen = []
    python = bdd._build

    def build(*args):
        plan = python(*args)
        seen.append((args, plan))
        return plan

    monkeypatch.setattr(bdd, "COMPILED", False)
    monkeypatch.setattr(bdd, "_build", build)
    return seen


def _check(seen):
    for args, plan in seen:
        compiled = bdd._compiled_build(*args)
        assert compiled is not None
        assert compiled == plan


def test_random_diagrams(builds):
    for seed in range(250):
        rng = np.random.default_rng(seed)
        edges, k = random_diagram(rng, int(rng.integers(3, 12)))
        rbd = RBD(edges, k=k, on_infeasible_rbd="ignore")
        if not rbd.structure_check["is_valid"]:
            continue
        try:
            decompose(rbd.G, "s", "t", core="bdd")
        except ValueError:
            continue
    assert len(builds) > 150
    _check(builds)


def test_components_drawn_in_several_places(builds):
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
            decompose(rbd.G, "s", "t", rbd._component_aliases(), core="bdd")
        except ValueError:
            continue
    assert any(
        len({v for v in args[6].values()}) < len(args[6]) for args, _ in builds
    ), "a component drawn twice"
    _check(builds)


@pytest.mark.parametrize("edges", [bridges(12), grid(4, 9), grid(7, 7)])
def test_meshed_cores(builds, edges):
    decompose(RBD(edges).G, "s", "t", core="bdd")
    assert builds and len(builds[-1][1][0]) > 50
    _check(builds)


def test_a_frontier_too_wide_is_built_in_python(monkeypatch):
    monkeypatch.setattr(bdd, "COMPILED_WIDTH", 2)
    monkeypatch.setattr(bdd, "COMPILED", True)
    rbd = RBD(grid(4, 5))
    plan = decompose(rbd.G, "s", "t", core="bdd").core_plan()
    monkeypatch.setattr(bdd, "COMPILED", False)
    assert decompose(rbd.G, "s", "t", core="bdd").core_plan() == plan


def test_auto_compiles_only_large_cores(monkeypatch):
    compiled = []
    real = bdd._compiled_build

    def spy(*args):
        compiled.append(len(args[0]))
        return real(*args)

    monkeypatch.setattr(bdd, "_compiled_build", spy)
    decompose(RBD(grid(3, 4)).G, "s", "t", core="bdd")
    assert not compiled
    decompose(RBD(grid(9, 18)).G, "s", "t", core="bdd")
    assert compiled


def _bits_equal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert np.array_equal(a.view(np.int64), b.view(np.int64))


@pytest.mark.parametrize("edges", [bridges(6), grid(5, 6)])
@pytest.mark.parametrize("shape", [None, (7,), (2, 3), (40,)])
def test_the_compiled_replay_is_pythons_to_the_bit(monkeypatch, edges, shape):
    rbd = RBD(edges)
    graph = decompose(rbd.G, "s", "t", core="bdd")
    assert graph.from_graph
    rng = np.random.default_rng(len(edges))
    nodes = sorted(rbd.nodes, key=str)

    def draw():
        if shape is None:
            return float(rng.uniform(0.0, 1.0))
        return rng.uniform(0.0, 1.0, shape)

    p = {n: draw() for n in nodes}
    p[nodes[0]] = 1.0 if shape is None else np.ones(shape)  # a sure node
    q = {n: 1.0 - v for n, v in p.items()}
    monkeypatch.setattr(modular, "COMPILED_STEPS", 10**9)
    python = (
        graph.probabilities(p, q, shape=shape),
        graph.value_and_gradient(p, q, shape=shape),
    )
    monkeypatch.setattr(modular, "COMPILED_STEPS", 1)
    compiled = (
        graph.probabilities(p, q, shape=shape),
        graph.value_and_gradient(p, q, shape=shape),
    )
    assert graph._plan_arrays is not None
    for a, b in zip(python[0], compiled[0]):
        _bits_equal(a, b)
    works, fails, gradient = python[1]
    c_works, c_fails, c_gradient = compiled[1]
    _bits_equal(works, c_works)
    _bits_equal(fails, c_fails)
    assert set(gradient) == set(c_gradient)
    for n in gradient:
        _bits_equal(gradient[n], c_gradient[n])


def test_probabilities_not_numbers_are_replayed_in_python(monkeypatch):
    # Values autograd traces (or anything but numbers and arrays of them)
    # keep the Python replay.
    graph = decompose(RBD(grid(4, 4)).G, "s", "t", core="bdd")
    monkeypatch.setattr(modular, "COMPILED_STEPS", 1)
    terms = len(graph.terms)
    assert graph._compiled_plan([0.5] * terms, [0.5] * terms, None)
    traced = [object()] * terms
    assert graph._compiled_plan(traced, traced, None) is None
