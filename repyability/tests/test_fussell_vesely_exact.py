"""The exact Fussell-Vesely importance (#137): the probability that some
minimal cut set (or path set) containing a node has failed, every node in
it, over the system's unreliability, against an enumeration of every state
of the components."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from surpyval import Exponential, FixedEventProbability, Weibull

from repyability import FaultTree, NonRepairableRBD, RepairableRBD
from repyability.rbd import modular
from repyability.rbd.helper_classes import PerfectReliability

F = FixedEventProbability.from_params

# The bridge: "c" joins the two paths.
BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("c", "e"),
    ("d", "t"),
    ("e", "t"),
]


def enumerated(rbd, q, sets):
    """{node: P(every node of some set containing it has failed)} / Q, over
    every state of the nodes, which fail with probabilities ``q`` (the
    junctions, which never fail, at 0)."""
    nodes = list(q)
    cuts = rbd.get_min_cut_sets()
    share = dict.fromkeys(nodes, 0.0)
    system = 0.0
    for state in itertools.product([False, True], repeat=len(nodes)):
        failed = {n for n, down in zip(nodes, state) if down}
        probability = math.prod(
            q[n] if down else 1 - q[n] for n, down in zip(nodes, state)
        )
        if any(cut <= failed for cut in cuts):
            system += probability
        for n in nodes:
            if any(n in s and s <= failed for s in sets):
                share[n] += probability
    return {n: share[n] / system for n in nodes}


def bridges(m):
    """``m`` bridges in series, joined by junctions."""
    edges, q, before = [], {}, "s"
    for i in range(m):
        a, b, c, d, e = (f"{x}{i}" for x in "abcde")
        after = "t" if i == m - 1 else f"j{i}"
        edges += [
            (before, a),
            (before, b),
            (a, c),
            (b, c),
            (a, d),
            (c, d),
            (b, e),
            (c, e),
            (d, after),
            (e, after),
        ]
        q.update({a: 0.3, b: 0.4, c: 0.5, d: 0.2, e: 0.6})
        if after != "t":
            q[after] = 0.0
        before = after
    return edges, q


def structures():
    """``(edges, q, k, repeated)``: diagrams with overlapping cut sets."""
    yield "bridge", BRIDGE, dict(a=0.3, b=0.4, c=0.5, d=0.2, e=0.6), {}, {}
    yield "bridge, likely failures", BRIDGE, dict(
        a=0.9, b=0.8, c=0.7, d=0.95, e=0.85
    ), {}, {}
    # A bridge of modules: a series pair for "a", a parallel pair for "e".
    yield "bridge of modules", [
        ("s", "x1"),
        ("x1", "x2"),
        ("x2", "c"),
        ("x2", "d"),
        ("s", "b"),
        ("b", "c"),
        ("b", "y1"),
        ("b", "y2"),
        ("c", "d"),
        ("c", "y1"),
        ("c", "y2"),
        ("d", "t"),
        ("y1", "t"),
        ("y2", "t"),
    ], dict(x1=0.3, x2=0.2, b=0.4, c=0.5, d=0.2, y1=0.6, y2=0.5), {}, {}
    # A 2-out-of-3 vote of series pairs, then a parallel pair.
    yield "2-out-of-3 vote", [
        ("s", "a1"),
        ("a1", "a2"),
        ("s", "b1"),
        ("b1", "b2"),
        ("s", "c1"),
        ("c1", "c2"),
        ("a2", "v"),
        ("b2", "v"),
        ("c2", "v"),
        ("v", "p"),
        ("v", "q"),
        ("p", "t"),
        ("q", "t"),
    ], dict(
        a1=0.3, a2=0.2, b1=0.4, b2=0.1, c1=0.5, c2=0.25, p=0.6, q=0.7, v=0.0
    ), {
        "v": 2
    }, {}
    yield "repeated node", [
        ("s", "a"),
        ("a", "b"),
        ("b", "t"),
        ("s", "c"),
        ("c", "a2"),
        ("a2", "t"),
        ("s", "d"),
        ("d", "t"),
    ], dict(a=0.3, b=0.4, c=0.5, d=0.6), {}, {"a2": "a"}
    edges, q = bridges(2)
    yield "two bridges in series", edges, q, {}, {}


def build(edges, q, k, repeated):
    reliabilities = {
        n: (F(v) if v > 0 else PerfectReliability) for n, v in q.items()
    }
    reliabilities.update(repeated)
    return NonRepairableRBD(edges, reliabilities, k=k or None)


@pytest.mark.parametrize("core", ["paths", "bdd"])
@pytest.mark.parametrize("fv_type", ["c", "p"])
@pytest.mark.parametrize(
    "edges, q, k, repeated",
    [s[1:] for s in structures()],
    ids=[s[0] for s in structures()],
)
def test_the_exact_measure_is_the_union(
    monkeypatch, edges, q, k, repeated, fv_type, core
):
    # The core decided by its path sets or by a decision diagram.
    monkeypatch.setattr(modular, "CORE_METHOD", core)
    rbd = build(edges, q, k, repeated)
    if fv_type == "c":
        sets = rbd.get_min_cut_sets()
    else:
        sets = {
            frozenset(s)
            for s in rbd.get_min_path_sets(include_in_out_nodes=False)
        }
    want = enumerated(rbd, q, sets)
    got = rbd.fussell_vesely(fv_type=fv_type)
    rare = rbd.fussell_vesely(fv_type=fv_type, method="rare_event")
    assert set(got) == set(rare)
    for node, value in got.items():
        assert value == pytest.approx(want[node], rel=1e-12, abs=1e-15)
        # The sum is never less than the union.
        assert rare[node] >= value * (1 - 1e-12)
        if fv_type == "c":
            assert value <= 1 + 1e-12


def test_the_union_stays_at_most_one_where_the_sum_passes_it():
    # Weibull lives on the bridge: once the system is likely down, the
    # rare-event sum nears 2, the share no more than 1.
    rbd = NonRepairableRBD(
        BRIDGE,
        {
            "a": Weibull.from_params([1000, 2]),
            "b": Weibull.from_params([1500, 1.5]),
            "c": Weibull.from_params([2000, 3]),
            "d": Weibull.from_params([1200, 2.5]),
            "e": Weibull.from_params([1800, 1.2]),
        },
    )
    t = np.array([900.0, 2000.0, 3000.0, 5000.0])
    exact = rbd.fussell_vesely(t)
    rare = rbd.fussell_vesely(t, method="rare_event")
    assert max(v.max() for v in exact.values()) <= 1 + 1e-12
    assert max(v[-1] for v in rare.values()) > 1.9
    # One time at a time, the same.
    for i, x in enumerate(t):
        at = rbd.fussell_vesely(x)
        for node in exact:
            assert at[node] == pytest.approx(exact[node][i], rel=1e-14)


def test_cut_sets_that_share_no_node_give_the_sum():
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")],
        {"p1": F(0.1), "p2": F(0.1), "v": F(0.05)},
    )
    exact = rbd.fussell_vesely()
    rare = rbd.fussell_vesely(method="rare_event")
    for node in exact:
        assert exact[node] == pytest.approx(rare[node], rel=1e-14)


@pytest.mark.parametrize("q", [0.3, 1e-4, 1e-12])
def test_small_probabilities_keep_their_precision(q):
    # A bridge of identical nodes: it fails with probability
    # 2q^2 + 2q^3 - 5q^4 + 2q^5 (it is self-dual), "a" is in the cut sets
    # {a, b} and {a, c, e}, and "c" in {a, c, e} and {b, c, d}.
    rbd = NonRepairableRBD(BRIDGE, {n: F(q) for n in "abcde"})
    Q = Fraction(q)
    system = 2 * Q**2 + 2 * Q**3 - 5 * Q**4 + 2 * Q**5
    want = {
        "a": Q * (Q + (1 - Q) * Q**2) / system,
        "c": Q * (2 * Q**2 - Q**4) / system,
    }
    got = rbd.fussell_vesely()
    for node, value in want.items():
        assert got[node] == pytest.approx(float(value), rel=1e-13)


def test_the_method_is_checked():
    rbd = NonRepairableRBD(BRIDGE, {n: F(0.1) for n in "abcde"})
    with pytest.raises(ValueError, match="method must be"):
        rbd.fussell_vesely(method="union")
    tree = FaultTree({"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2})
    with pytest.raises(ValueError, match="method must be"):
        tree.fussell_vesely(method="union")


def test_a_fault_tree_with_repeated_events():
    # The bridge as a fault tree: the top event is any of its cut sets.
    tree = FaultTree(
        {
            "top": ("or", ["ab", "de", "ace", "bcd"]),
            "ab": ("and", ["a", "b"]),
            "de": ("and", ["d", "e"]),
            "ace": ("and", ["a", "c", "e"]),
            "bcd": ("and", ["b", "c", "d"]),
        },
        {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.95, "e": 0.85},
    )
    rbd = NonRepairableRBD(
        BRIDGE,
        {
            n: F(v)
            for n, v in dict(a=0.9, b=0.8, c=0.7, d=0.95, e=0.85).items()
        },
    )
    exact = rbd.fussell_vesely()
    rare = rbd.fussell_vesely(method="rare_event")
    for event, value in tree.fussell_vesely().items():
        assert value == pytest.approx(exact[event], rel=1e-12)
    for event, value in tree.fussell_vesely(method="rare_event").items():
        assert value == pytest.approx(rare[event], rel=1e-12)


def repairable(life, repair=1.0):
    return {
        "reliability": Exponential.from_params([life]),
        "repairability": Exponential.from_params([repair]),
    }


def test_a_repairable_diagram_at_its_long_run_unavailabilities():
    rates = dict(a=0.3, b=0.4, c=0.5, d=0.2, e=0.6)
    rbd = RepairableRBD(BRIDGE, {n: repairable(r) for n, r in rates.items()})
    down = {n: 1 - a for n, a in rbd.node_availability().items()}
    fixed = NonRepairableRBD(BRIDGE, {n: F(q) for n, q in down.items()})
    for method in ("exact", "rare_event"):
        got = rbd.fussell_vesely(method=method)
        want = fixed.fussell_vesely(method=method)
        for node in rates:
            assert got[node] == pytest.approx(want[node], rel=1e-12)
    assert max(rbd.fussell_vesely().values()) <= 1


def test_a_repairable_diagram_averages_the_union_over_its_inspections():
    # A hidden failure found by inspections: its unavailability changes
    # over the interval, and the measures average over it (the union at
    # each time, then the average, over the system's average).
    rbd = RepairableRBD(
        BRIDGE,
        {
            "a": {
                "reliability": Exponential.from_params([0.01]),
                "repairability": "instant",
                "inspection": {"interval": 50.0},
            },
            **{n: repairable(0.2) for n in "bcde"},
        },
    )
    p, q, weights = rbd._importance_probabilities(None, None)
    size = len(weights)
    cuts = rbd.get_min_cut_sets()
    nodes = list("abcde")
    share = {n: np.zeros(size) for n in nodes}
    system = np.zeros(size)
    for state in itertools.product([False, True], repeat=len(nodes)):
        failed = {n for n, down in zip(nodes, state) if down}
        probability = np.ones(size)
        for n, down in zip(nodes, state):
            probability = probability * (
                np.broadcast_to(q[n], size) if down else p[n]
            )
        if any(cut <= failed for cut in cuts):
            system += probability
        for n in nodes:
            if any(n in c and c <= failed for c in cuts):
                share[n] += probability
    got = rbd.fussell_vesely()
    for n in nodes:
        assert got[n] == pytest.approx(
            weights @ share[n] / (weights @ system), rel=1e-11
        )


def test_large_diagrams_stay_fast():
    # Forty bridges in series: a meshed core, decided by a decision
    # diagram, and the cut sets through each node worked out in one plan.
    edges, q = bridges(40)
    rbd = build(edges, q, {}, {})
    fv = rbd.fussell_vesely()
    assert len(fv) == 200
    # Each bridge on its own: its share is its own nodes' over the system.
    alone, _ = bridges(1)
    one = NonRepairableRBD(
        alone,
        {
            n: F(v)
            for n, v in dict(a0=0.3, b0=0.4, c0=0.5, d0=0.2, e0=0.6).items()
        },
    )
    one_fv = one.fussell_vesely()
    scale = one.ff() / rbd.ff()
    for node in "abcde":
        assert fv[f"{node}7"] == pytest.approx(
            one_fv[f"{node}0"] * scale, rel=1e-10
        )
