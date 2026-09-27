"""The exact engine's reduction to modules (issue #83).

Before the Shannon decomposition, the series chains, parallel groups and
k-out-of-n groups of a diagram are reduced to closed forms, and nodes that
a direct edge bypasses are dropped; only what is left (the "core") is
decomposed from its minimal path sets. These tests hold the reduced engine
to brute-force enumeration of every component state on random diagrams, to
the unreduced engine (``decompose(..., reduce=False)``, which is the engine
as it was before) on nested compositions too large to enumerate and through
every public method, and check that a series-parallel diagram never lists
its path sets, however many it has.
"""

import itertools
import pickle
import warnings
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import RBD, BetaFactor, CCFGroup, NonRepairableRBD
from repyability.rbd import modular
from repyability.rbd.modular import KOON, NODE, PARALLEL, SERIES, decompose
from repyability.tests.test_performance_equivalence import (
    assert_same,
    rbds,
    repairable_rbds,
)

W = surv.Weibull.from_params


def unreduced(rbd):
    """``rbd``, switched to the engine with no reduction."""
    rbd._modules = decompose(
        rbd.G, rbd.input_node, rbd.output_node, reduce=False
    )
    for cached in ("_min_path_sets", "_min_cut_sets"):
        rbd.__dict__.pop(cached, None)
    return rbd


# -- random diagrams, against enumeration -------------------------------------


def random_diagram(rng, n):
    """A random RBD over nodes 0..n-1 (in topological order), with random
    k-out-of-n values and the odd direct edge from the input to the output:
    bridges, shared nodes and bypasses as well as series-parallel parts."""
    edges = set()
    for i in range(n):
        earlier = ["s"] + list(range(i))
        for j in rng.choice(
            len(earlier), rng.integers(1, min(3, len(earlier)) + 1), False
        ):
            edges.add((earlier[j], i))
    feeding = {a for a, _ in edges}
    for i in range(n):
        if i not in feeding or rng.random() < 0.2:
            edges.add((i, "t"))
    if rng.random() < 0.1:
        edges.add(("s", "t"))
    inputs: dict = {}
    for a, b in edges:
        inputs.setdefault(b, []).append(a)
    k = {
        v: int(rng.integers(2, len(ins) + 1))
        for v, ins in inputs.items()
        if len(ins) > 1 and rng.random() < 0.3
    }
    return sorted(edges, key=str), k


def enumerate_states(edges, k, nodes):
    """Every combination of the nodes' states (the rows of a boolean
    matrix, one column per node), and whether the system works in each, by
    the definition of the structure function: a node is reached when it
    works and at least k of its inputs are reached."""
    states = np.array(
        list(itertools.product([False, True], repeat=len(nodes))), dtype=bool
    ).reshape(-1, len(nodes))
    column = {n: states[:, i] for i, n in enumerate(nodes)}
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


@pytest.mark.parametrize("chunk", range(4))
def test_random_diagrams_match_enumeration(chunk):
    kinds = set()
    for seed in range(chunk * 150, (chunk + 1) * 150):
        rng = np.random.default_rng(seed)
        edges, k = random_diagram(rng, 2 + seed % 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                rbd = RBD(edges, k=k)
            except ValueError:
                continue
        nodes = rbd.nodes
        states, works = enumerate_states(edges, k, nodes)

        def members(row, state):
            return frozenset(n for n, b in zip(nodes, row) if b == state)

        path_sets = minimal({members(row, True) for row in states[works]})
        cut_sets = minimal({members(row, False) for row in states[~works]})
        structure = rbd._decomposition()
        kinds |= {term[0] for term in structure.terms}
        assert rbd.get_min_path_sets(False) == path_sets, seed
        assert rbd.get_min_cut_sets() == cut_sets, seed
        assert rbd.get_min_cut_sets(True) == cut_sets | {
            frozenset(["s"]),
            frozenset(["t"]),
        }
        relevant = set().union(*path_sets)
        assert rbd.find_irrelevant_components() == set(nodes) - relevant
        for row, expected in zip(states, works):
            status = dict(zip(nodes, row.tolist()))
            assert rbd.is_system_working(status, "p") is bool(expected), seed
            assert rbd.is_system_working(status, "c") is bool(expected), seed

        # The system probability: the sum over the working states.
        p = rng.uniform(0, 1, (len(nodes), 3))

        def probability(p, working=True):
            weights = np.where(states[:, :, None], p, 1 - p).prod(axis=1)
            return weights[works == working].sum(axis=0)

        exact = probability(p)
        named = dict(zip(nodes, p))
        for method in ("p", "c"):
            np.testing.assert_allclose(
                rbd.system_probability(named, method),
                exact,
                rtol=1e-12,
                atol=1e-15,
            )
        # Each node's Birnbaum importance: R(i works) - R(i failed).
        first = {n: float(v[0]) for n, v in named.items()}
        R, Q, gradient = structure.value_and_gradient(
            first, {n: 1 - v for n, v in first.items()}
        )
        assert R == pytest.approx(exact[0], rel=1e-12, abs=1e-15)
        failing = probability(p, working=False)[0]
        assert Q == pytest.approx(failing, rel=1e-12, abs=1e-15)
        for i, n in enumerate(nodes):
            up, down = p[:, :1].copy(), p[:, :1].copy()
            up[i], down[i] = 1.0, 0.0
            birnbaum = probability(up)[0] - probability(down)[0]
            assert gradient.get(n, 0.0) == pytest.approx(
                birnbaum, rel=1e-12, abs=1e-15
            ), (seed, n)
        # The lifetime: the longest-lived minimal path set's shortest life.
        lives = {n: rng.exponential(1.0, 20) for n in rbd.G.nodes}
        expected = np.full(20, -np.inf)
        for path_set in path_sets:
            life = np.full(20, np.inf)
            for n in path_set:
                life = np.minimum(life, lives[n])
            expected = np.maximum(expected, life)
        assert np.array_equal(structure.lifetime(lives, 20), expected), seed
    assert {NODE, SERIES, PARALLEL} <= kinds


# -- nested compositions, against the unreduced engine ------------------------


class Composition:
    """A random nesting of series, parallel and k-out-of-n groups and
    bridges, drawn as an RBD from ``"s"`` to ``"t"``."""

    def __init__(self, rng, depth):
        self.rng, self.edges, self.k, self.count = rng, [], {}, 0
        exits = self.build(depth, ["s"])
        self.edges += [(e, "t") for e in exits]

    def node(self, inputs):
        self.count += 1
        v = f"n{self.count}"
        self.edges += [(u, v) for u in inputs]
        return v

    def build(self, depth, inputs):
        r = self.rng.random()
        if depth == 0 or r < 0.25:
            return [self.node(inputs)]
        if r < 0.45:
            for _ in range(self.rng.integers(2, 4)):
                inputs = self.build(depth - 1, inputs)
            return inputs
        if r < 0.7:
            return [
                e
                for _ in range(self.rng.integers(2, 4))
                for e in self.build(depth - 1, inputs)
            ]
        if r < 0.9:
            # k of n groups, each joined into one node, feed a voter.
            n = int(self.rng.integers(3, 5))
            members = []
            for _ in range(n):
                exits = self.build(depth - 1, inputs)
                members.append(
                    exits[0] if len(exits) == 1 else self.node(exits)
                )
            voter = self.node(members)
            self.k[voter] = int(self.rng.integers(2, n + 1))
            return [voter]
        a, b = self.node(inputs), self.node(inputs)
        c = self.node([a, b])
        return [self.node([a, c]), self.node([b, c])]


def test_nested_compositions_match_the_unreduced_engine():
    kinds, cores = set(), 0
    for seed in range(150):
        rng = np.random.default_rng(seed)
        composition = Composition(rng, int(rng.integers(1, 5)))
        rbd = RBD(composition.edges, k=composition.k)
        if len(rbd.nodes) > 30:
            # The unreduced engine's path set search gets slow.
            continue
        reference = decompose(rbd.G, "s", "t", reduce=False)
        structure = rbd._decomposition()
        kinds |= {term[0] for term in structure.terms}
        cores += structure.core is not None
        path_sets = reference.path_sets()
        assert rbd.get_min_path_sets(False) == path_sets, seed
        assert rbd.get_min_cut_sets() == reference.cut_sets(), seed
        for _ in range(40):
            status = {n: bool(rng.random() < 0.7) for n in rbd.nodes}
            expected = any(all(status[n] for n in ps) for ps in path_sets)
            assert rbd.is_system_working(status, "p") is expected, seed
            assert rbd.is_system_working(status, "c") is expected, seed
        p = {n: rng.uniform(0, 1, 4) for n in rbd.nodes}
        exact = reference.probabilities(p, shape=4)[0]
        for method in ("p", "c"):
            np.testing.assert_allclose(
                rbd.system_probability(p, method),
                exact,
                rtol=1e-12,
                atol=1e-15,
            )
        first = {n: float(v[0]) for n, v in p.items()}
        complements = {n: 1 - v for n, v in first.items()}
        works, _, gradient = structure.value_and_gradient(first, complements)
        _, _, expected = reference.value_and_gradient(first, complements)
        assert works == pytest.approx(exact[0], rel=1e-12, abs=1e-15)
        for n in rbd.nodes:
            assert gradient.get(n, 0.0) == pytest.approx(
                expected.get(n, 0.0), rel=1e-10, abs=1e-14
            ), (seed, n)
        lives = {n: rng.exponential(1.0, 10) for n in rbd.G.nodes}
        assert np.array_equal(
            structure.lifetime(lives, 10), reference.lifetime(lives, 10)
        ), seed
    assert kinds == {NODE, SERIES, PARALLEL, KOON}
    assert cores > 10


# -- every public method, reduced or not --------------------------------------


@pytest.mark.parametrize("name", sorted(rbds()))
def test_non_repairable_methods_match_the_unreduced_engine(name):
    rbd, reference = rbds()[name], unreduced(rbds()[name])
    x = np.array([50.0, 400.0, 900.0])
    for method in ("p", "c"):
        np.testing.assert_allclose(
            rbd.sf(x, method=method),
            reference.sf(x, method=method),
            rtol=1e-12,
        )
    assert rbd.get_min_path_sets() == reference.get_min_path_sets()
    assert rbd.get_min_cut_sets() == reference.get_min_cut_sets()
    assert rbd.find_irrelevant_components() == (
        reference.find_irrelevant_components()
    )
    assert np.array_equal(
        rbd.random(300, seed=3), reference.random(300, seed=3)
    )
    if rbd.ccf_groups:
        return
    measures = [
        lambda r: r.birnbaum_importance(x),
        lambda r: r.improvement_potential(x),
        lambda r: r.risk_achievement_worth(x),
        lambda r: r.risk_reduction_worth(x),
        lambda r: r.criticality_importance(x),
        lambda r: r.criticality_importance(x, kind="success"),
        lambda r: r.fussell_vesely(x),
        lambda r: r.fussell_vesely(x, fv_type="p"),
        lambda r: r.structural_importance(),
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        for measure in measures:
            got, expected = measure(rbd), measure(reference)
            assert set(got) == set(expected)
            for node in got:
                np.testing.assert_allclose(
                    got[node], expected[node], rtol=1e-10, atol=1e-14
                )


@pytest.mark.parametrize("name", sorted(repairable_rbds()))
def test_repairable_methods_match_the_unreduced_engine(name):
    rbd, reference = repairable_rbds()[name], unreduced(
        repairable_rbds()[name]
    )
    for method in (
        "mean_availability",
        "system_failure_frequency",
        "mean_up_time",
        "mean_down_time",
    ):
        try:
            expected = getattr(reference, method)()
        except NotImplementedError:
            continue
        assert getattr(rbd, method)() == pytest.approx(expected, rel=1e-12)
    for measure in (
        "birnbaum_importance",
        "risk_achievement_worth",
        "criticality_importance",
    ):
        try:
            expected = getattr(reference, measure)()
        except NotImplementedError:
            continue
        got = getattr(rbd, measure)()
        for node in expected:
            assert got[node] == pytest.approx(expected[node], rel=1e-10)
    # The structure function is the same, so the simulation is too.
    for method in ("p", "c"):
        assert_same(
            rbd.availability(200.0, N=20, seed=5, method=method),
            reference.availability(200.0, N=20, seed=5, method=method),
        )


# -- what reduces, and what does not ------------------------------------------


def staged(stages, unit):
    """``stages`` duplicated pairs in series, each pair feeding the next."""
    edges, previous = [], ["s"]
    for i in range(stages):
        pair = [f"a{i}", f"b{i}"]
        edges += [(p, u) for p in previous for u in pair]
        previous = pair
    edges += [(p, "t") for p in previous]
    nodes = [n for i in range(stages) for n in (f"a{i}", f"b{i}")]
    return edges, {n: unit(i) for i, n in enumerate(nodes)}


def test_a_series_parallel_diagram_never_lists_its_path_sets(monkeypatch):
    # Thirty stages have 2 ** 30 minimal path sets.
    def no_path_sets(*args, **kwargs):
        raise AssertionError("the path sets were listed")

    monkeypatch.setattr(modular, "find_min_path_sets", no_path_sets)
    edges, models = staged(30, lambda i: W([100 + 3 * i, 1.5 + i / 30]))
    rbd = NonRepairableRBD(edges, models)
    structure = rbd._decomposition()
    root = structure.terms[structure.root]
    assert structure.core is None and root[0] == SERIES and len(root[1]) == 30
    assert all(structure.terms[c][0] == PARALLEL for c in root[1])

    x = np.linspace(1, 300, 50)
    expected = np.ones_like(x)
    for i in range(30):
        q_a, q_b = models[f"a{i}"].ff(x), models[f"b{i}"].ff(x)
        expected = expected * (1 - q_a * q_b)
    np.testing.assert_allclose(rbd.sf(x), expected, rtol=1e-12)
    # 1 - (the probability of failing): exact to rounding in 1.
    np.testing.assert_allclose(
        rbd.sf(x, method="c"), expected, rtol=1e-12, atol=1e-15
    )
    # A unit's Birnbaum importance: its partner has failed, and every other
    # stage works.
    birnbaum = rbd.birnbaum_importance(x)
    stage = 1 - models["a0"].ff(x) * models["b0"].ff(x)
    np.testing.assert_allclose(
        birnbaum["a0"], models["b0"].ff(x) * expected / stage, rtol=1e-10
    )
    for measure in (
        rbd.improvement_potential,
        rbd.risk_achievement_worth,
        rbd.risk_reduction_worth,
        rbd.criticality_importance,
        rbd.fussell_vesely,
    ):
        assert len(measure(x)) == 60
    assert rbd.get_min_cut_sets() == {
        frozenset([f"a{i}", f"b{i}"]) for i in range(30)
    }
    lifetimes = rbd.random(1000, seed=1)
    assert lifetimes.shape == (1000,) and np.isfinite(lifetimes).all()
    status = {n: True for n in rbd.nodes}
    assert rbd.is_system_working(status, "p")
    status["a7"] = status["b7"] = False
    assert not rbd.is_system_working(status, "c")


def test_small_unreliabilities_keep_their_precision():
    # Each unit fails with probability 1e-9, so each stage with 1e-18 and the
    # system with about 3e-17: far below the rounding of 1 - R.
    q = Fraction(1, 10**9)
    edges, models = staged(
        30, lambda i: FixedEventProbability.from_params(1e-9)
    )
    rbd = NonRepairableRBD(edges, models)
    stage = q * q
    system = 1 - (1 - stage) ** 30
    p = {n: 1 - 1e-9 for n in rbd.nodes}
    fails = rbd._decomposition().probabilities(
        p, {n: 1e-9 for n in rbd.nodes}
    )[1]
    assert fails == pytest.approx(float(system), rel=1e-12)
    # Each unit accounts for its stage's share of the system failures.
    share = float(stage * (1 - stage) ** 29 / system)
    for value in rbd.criticality_importance().values():
        assert value == pytest.approx(share, rel=1e-6)


def test_a_k_out_of_n_group_of_chains_reduces():
    rbd = RBD(
        [
            ("s", "a1"),
            ("a1", "a2"),
            ("a2", "v"),
            ("s", "b"),
            ("b", "v"),
            ("s", "c"),
            ("c", "v"),
            ("v", "t"),
        ],
        k={"v": 2},
    )
    structure = rbd._decomposition()
    assert structure.core is None
    assert sorted(term[0] for term in structure.terms if term[0] != NODE) == [
        SERIES,
        SERIES,
        KOON,
    ]
    p = {"a1": 0.9, "a2": 0.8, "b": 0.7, "c": 0.6, "v": 0.95}
    a = 0.9 * 0.8
    two_of_three = a * 0.7 + a * 0.6 + 0.7 * 0.6 - 2 * a * 0.7 * 0.6
    assert rbd.system_probability(p)[0] == pytest.approx(0.95 * two_of_three)
    # Needing all three is plain series.
    edges = [(u, w) for u, w in rbd.G.edges]
    all_three = RBD(edges, k={"v": 3})
    kinds = {term[0] for term in all_three._decomposition().terms}
    assert kinds == {NODE, SERIES}
    assert all_three.system_probability(p)[0] == pytest.approx(
        0.95 * a * 0.7 * 0.6
    )


def test_the_output_node_can_vote():
    rbd = RBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("s", "c"),
            ("a", "t"),
            ("b", "t"),
            ("c", "t"),
        ],
        k={"t": 2},
    )
    structure = rbd._decomposition()
    assert structure.terms[structure.root][0] == KOON
    p = {"a": 0.9, "b": 0.8, "c": 0.7}
    exact = 0.9 * 0.8 + 0.9 * 0.7 + 0.8 * 0.7 - 2 * 0.9 * 0.8 * 0.7
    assert rbd.system_probability(p)[0] == pytest.approx(exact)
    assert rbd.system_probability(p, "c")[0] == pytest.approx(exact)


def test_a_bridge_is_left_as_it_is():
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "d"),
        ("c", "d"),
        ("c", "e"),
        ("b", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    rbd = RBD(edges)
    structure = rbd._decomposition()
    reference = decompose(rbd.G, "s", "t", reduce=False)
    assert all(term[0] == NODE for term in structure.terms)
    assert structure.path_sets() == reference.path_sets()
    p = {n: np.linspace(0.1, 0.9, 5) for n in "abcde"}
    np.testing.assert_allclose(
        rbd.system_probability(p), reference.probabilities(p, shape=5)[0]
    )


def test_a_bridge_inside_a_series_parallel_diagram():
    # The bridge is the core, over reduced groups; the rest reduces around it.
    edges = [
        ("s", "x1"),
        ("s", "x2"),
        ("x1", "a1"),
        ("x2", "a1"),
        ("a1", "a2"),
        ("x1", "b"),
        ("x2", "b"),
        ("a2", "c"),
        ("b", "c"),
        ("a2", "d"),
        ("c", "d"),
        ("c", "e"),
        ("b", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    rbd = RBD(edges)
    structure = rbd._decomposition()
    assert structure.core is not None
    kinds = sorted(term[0] for term in structure.terms if term[0] != NODE)
    assert kinds == [SERIES, PARALLEL]
    reference = decompose(rbd.G, "s", "t", reduce=False)
    p = {n: np.linspace(0.2, 0.95, 4) for n in rbd.nodes}
    np.testing.assert_allclose(
        rbd.system_probability(p), reference.probabilities(p, shape=4)[0]
    )
    assert rbd.get_min_cut_sets() == reference.cut_sets()


def test_a_bypassed_node_is_left_out():
    rbd = RBD([("s", "a"), ("a", "t"), ("a", "b"), ("b", "t")])
    structure = rbd._decomposition()
    assert structure.nodes == {"a"}
    assert rbd.find_irrelevant_components() == {"b"}
    assert rbd.structure_check["irrelevant_nodes"] == {"b"}
    importance = rbd.structural_importance()
    assert importance == {"a": 1.0, "b": 0.0}


def test_a_direct_edge_makes_the_system_always_work():
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t"), ("s", "t")],
        {"a": W([10, 2]), "b": W([20, 2])},
    )
    structure = rbd._decomposition()
    assert structure.always_works
    assert structure.probabilities({}) == (1.0, 0.0)
    assert structure.value_and_gradient({}, {}) == (1.0, 0.0, {})
    assert rbd.sf(100.0) == 1.0
    assert rbd.get_min_path_sets(False) == {frozenset()}
    assert rbd.get_min_cut_sets() == set()
    assert rbd.find_irrelevant_components() == {"a", "b"}
    assert rbd.is_system_working({"a": False, "b": False}, "p")
    assert np.isinf(rbd.random(5, seed=1)).all()


def test_a_repeated_node_matches_the_unreduced_engine():
    # "psu2" is the power supply "psu" drawn a second time.
    edges = [
        ("s", "psu"),
        ("psu", "p1"),
        ("s", "psu2"),
        ("psu2", "p2"),
        ("p1", "v"),
        ("p2", "v"),
        ("s", "backup"),
        ("backup", "v"),
        ("v", "t"),
    ]
    models = {
        "psu": W([300, 1.2]),
        "psu2": "psu",
        "p1": W([100, 2]),
        "p2": W([120, 2]),
        "backup": W([60, 3]),
        "v": W([500, 1.5]),
    }
    rbd = NonRepairableRBD(edges, models)
    reference = unreduced(NonRepairableRBD(edges, models))
    x = np.array([10.0, 50.0, 150.0])
    np.testing.assert_allclose(rbd.sf(x), reference.sf(x), rtol=1e-12)
    bi, expected = rbd.birnbaum_importance(x), reference.birnbaum_importance(x)
    for node in expected:
        np.testing.assert_allclose(bi[node], expected[node], rtol=1e-10)


def test_common_cause_groups_are_conditioned_before_reducing():
    # The members of a beta-factor group sit in parallel: the group is
    # conditioned on its shared cause first, and each case reduces.
    unit = FixedEventProbability.from_params(0.1)
    edges = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
    models = {
        "a": unit,
        "b": unit,
        "c": FixedEventProbability.from_params(0.05),
    }
    groups = [CCFGroup(["a", "b"], BetaFactor(0.1))]
    rbd = NonRepairableRBD(edges, models, ccf_groups=groups)
    reference = unreduced(NonRepairableRBD(edges, models, ccf_groups=groups))
    # The shared cause fails both with probability 0.1 * 0.1; otherwise each
    # fails alone with probability 0.9 * 0.1.
    pair = (1 - 0.1 * 0.1) * (1 - (0.9 * 0.1) ** 2)
    assert rbd.sf() == pytest.approx(reference.sf(), rel=1e-12)
    assert rbd.sf() == pytest.approx(pair * 0.95, rel=1e-12)


def test_a_long_chain_needs_no_recursion():
    n = 3000
    edges = [("s", "c0")] + [(f"c{i}", f"c{i + 1}") for i in range(n - 1)]
    rbd = RBD(edges + [(f"c{n - 1}", "t")])
    assert rbd.system_probability({f"c{i}": 0.9999 for i in range(n)})[
        0
    ] == pytest.approx(0.9999**n)
    assert rbd.get_min_path_sets(False) == {frozenset(rbd.nodes)}
    assert len(rbd.get_min_cut_sets()) == n


def test_a_large_core_is_checked_by_looping_over_its_sets(monkeypatch):
    # Past a size, the compiled structure function loops over the core's
    # sets instead of writing them out; both agree.
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "d"),
        ("c", "d"),
        ("c", "e"),
        ("b", "e"),
        ("d", "x1"),
        ("d", "x2"),
        ("e", "x1"),
        ("e", "x2"),
        ("x1", "t"),
        ("x2", "t"),
    ]
    written, looped = RBD(edges), RBD(edges)
    everything = {n: True for n in written.nodes}
    for method in ("p", "c"):
        assert written.is_system_working(everything, method)
    monkeypatch.setattr(modular, "_WRITTEN_OUT", 0)
    rng = np.random.default_rng(4)
    for _ in range(200):
        status = {n: bool(rng.random() < 0.6) for n in written.nodes}
        for method in ("p", "c"):
            assert written.is_system_working(status, method) is (
                looped.is_system_working(status, method)
            )


def test_an_rbd_pickles_after_its_structure_function_is_compiled():
    rbd = RBD([("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")])
    status = {"a": False, "b": True, "c": True}
    assert rbd.is_system_working(status, "p")
    copy = pickle.loads(pickle.dumps(rbd))
    assert copy.is_system_working(status, "p")
    assert not copy.is_system_working({**status, "c": False}, "c")


@pytest.mark.parametrize("rare", ["failure", "success"])
def test_importance_near_certainty_keeps_its_precision(rare):
    # A bridge of units failing (or working) with probability 1e-8: the
    # system fails (or works) with probability about 2e-16 and each Birnbaum
    # importance is about 1e-8, far below the rounding of the probabilities
    # near 1 that it is a difference of, so both are taken from the other
    # end.
    rbd = RBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("a", "c"),
            ("b", "c"),
            ("a", "d"),
            ("c", "d"),
            ("c", "e"),
            ("b", "e"),
            ("d", "t"),
            ("e", "t"),
        ]
    )
    small = Fraction(1, 10**8)
    p = 1 - small if rare == "failure" else small

    def works(fixed=None):
        total = Fraction(0)
        for bits in itertools.product([False, True], repeat=5):
            status = dict(zip(rbd.nodes, bits))
            if fixed and status[fixed[0]] != fixed[1]:
                continue
            if rbd.is_system_working(status, "p"):
                term = Fraction(1)
                for n, b in status.items():
                    if not fixed or n != fixed[0]:
                        term *= p if b else 1 - p
                total += term
        return total

    R, Q, gradient = rbd._decomposition().value_and_gradient(
        {n: float(p) for n in rbd.nodes}, {n: float(1 - p) for n in rbd.nodes}
    )
    assert R == pytest.approx(float(works()), rel=1e-12, abs=0)
    assert Q == pytest.approx(float(1 - works()), rel=1e-12, abs=0)
    for n in rbd.nodes:
        birnbaum = works((n, True)) - works((n, False))
        assert gradient[n] == pytest.approx(float(birnbaum), rel=1e-12, abs=0)


def test_a_member_certain_to_work_leaves_its_partners_no_importance():
    # With "b" certain to work, whether "a" works makes no difference now,
    # and neither does anything else in the parallel group.
    rbd = RBD([("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")])
    p = {"a": 0.3, "b": 1.0, "c": 0.9}
    q = {n: 1 - v for n, v in p.items()}
    R, Q, gradient = rbd._decomposition().value_and_gradient(p, q)
    assert (R, Q) == (pytest.approx(0.9), pytest.approx(0.1))
    assert gradient.get("a", 0.0) == 0.0
    assert gradient["c"] == pytest.approx(1.0)
