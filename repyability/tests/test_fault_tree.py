"""Tests for :class:`FaultTree`: static fault trees and their conversion to
and from reliability block diagrams.

The reference for every exact quantity is the definition, enumerated: the
top event occurs in a state when it occurs by the gates' logic, worked out
recursively and independently of the library, and its probability is the
sum over every state in which it occurs. Random trees mix OR, AND and VOTE
gates, repeated events and shared gates.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import FaultTree, NonRepairableRBD, PerfectReliability
from repyability.rbd.ccf import shock_outcomes
from repyability.tests.test_rbd_modular import random_diagram

W = surv.Weibull.from_params


# -- the definition, enumerated ----------------------------------------------


def occurs_by_hand(gates, x, occurred) -> bool:
    if x not in gates:
        return x in occurred
    kind, *rest = gates[x]
    inputs = rest[-1]
    k = {"or": 1, "and": len(inputs)}.get(kind.lower(), rest[0])
    return sum(occurs_by_hand(gates, c, occurred) for c in inputs) >= k


def minimal(sets):
    return {s for s in sets if not any(other < s for other in sets)}


def enumerated(gates, probabilities, top, fixed=None):
    """The top event probability, its minimal cut and path sets, and the
    occurring states, from every state of the events (``fixed`` holds
    events forced to occur (True) or not (False))."""
    fixed = fixed or {}
    names = list(probabilities)
    occurring, total = [], []
    for state in itertools.product((False, True), repeat=len(names)):
        occurred = frozenset(n for n, s in zip(names, state) if s)
        if any((n in occurred) != v for n, v in fixed.items()):
            continue
        if occurs_by_hand(gates, top, occurred):
            occurring.append(occurred)
            weight = 1.0
            for n, s in zip(names, state):
                if n in fixed:
                    continue
                weight *= probabilities[n] if s else 1.0 - probabilities[n]
            total.append(weight)
    everything = frozenset(names)
    working = [
        everything - s
        for s in (
            frozenset(n for n, v in zip(names, state) if v)
            for state in itertools.product((False, True), repeat=len(names))
        )
        if not occurs_by_hand(gates, top, s)
    ]
    return math.fsum(total), minimal(set(occurring)), minimal(set(working))


def random_tree(rng, n_events=7, n_gates=6):
    """A random tree over events e0.. and gates g0.. (g{n-1} the top):
    each gate takes two to four inputs among the events and earlier gates,
    so events and gates are often repeated; the top also takes whatever no
    gate has used, so that everything is below it."""
    events = [f"e{i}" for i in range(n_events)]
    gates = {}
    used = set()
    for g in range(n_gates):
        pool = events + [f"g{j}" for j in range(g)]
        size = int(rng.integers(2, min(4, len(pool)) + 1))
        inputs = [pool[i] for i in rng.choice(len(pool), size, replace=False)]
        if g == n_gates - 1:
            inputs += [x for x in pool if x not in used and x not in inputs]
        used.update(inputs)
        kind = str(rng.choice(["or", "and", "vote"]))
        if kind == "vote":
            gates[f"g{g}"] = (
                "vote",
                int(rng.integers(1, len(inputs) + 1)),
                inputs,
            )
        else:
            gates[f"g{g}"] = (kind, inputs)
    probabilities = {e: float(rng.uniform(0.02, 0.6)) for e in events}
    return gates, probabilities, f"g{n_gates - 1}"


# -- a small tree by hand -----------------------------------------------------


def cooling():
    return FaultTree(
        {
            "no cooling": ("or", ["no flow", "valve"]),
            "no flow": ("and", ["pump 1", "pump 2"]),
        },
        {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
    )


def test_a_small_tree_by_hand():
    tree = cooling()
    assert tree.top == "no cooling"
    assert tree.gates == {
        "no cooling": ("or", ["no flow", "valve"]),
        "no flow": ("and", ["pump 1", "pump 2"]),
    }
    assert tree.repeated_events == frozenset()
    assert tree.is_fixed
    q = 1 - (1 - 0.1**2) * (1 - 0.05)
    assert tree.top_event_probability() == pytest.approx(q, rel=1e-15)
    assert tree.minimal_cut_sets() == [
        frozenset({"valve"}),
        frozenset({"pump 1", "pump 2"}),
    ]
    assert tree.minimal_path_sets() == [
        frozenset({"pump 1", "valve"}),
        frozenset({"pump 2", "valve"}),
    ]
    assert tree.occurs({"valve"}) and tree.occurs({"pump 1", "pump 2"})
    assert not tree.occurs({"pump 1"}) and not tree.occurs(set())
    # Birnbaum: P(top | e) - P(top | not e).
    birnbaum = tree.birnbaum_importance()
    assert birnbaum["valve"] == pytest.approx(1 - 0.01, rel=1e-15)
    assert birnbaum["pump 1"] == pytest.approx(0.1 * 0.95, rel=1e-15)
    assert tree.risk_achievement_worth()["valve"] == pytest.approx(1 / q)
    assert tree.risk_reduction_worth()["valve"] == pytest.approx(q / 0.01)
    assert tree.criticality_importance()["valve"] == pytest.approx(
        0.99 * 0.05 / q
    )
    assert tree.fussell_vesely()["pump 1"] == pytest.approx(0.01 / q)
    ranked = tree.ranked_cut_sets()
    assert ranked[0] == (frozenset({"valve"}), 0.05)
    assert ranked[1][1] == pytest.approx(0.01)
    assert "3 events" in repr(tree)


def test_a_shared_supply_is_counted_once():
    # Two channels, either of which will do, share a power supply.
    tree = FaultTree(
        {
            "no output": ("and", ["channel A", "channel B"]),
            "channel A": ("or", ["supply", "pump A"]),
            "channel B": ("or", ["supply", "pump B"]),
        },
        {"supply": 0.01, "pump A": 0.1, "pump B": 0.1},
    )
    assert tree.repeated_events == frozenset({"supply"})
    exact = 0.01 + 0.99 * 0.1**2
    assert tree.top_event_probability() == pytest.approx(exact, rel=1e-15)
    # Treating the supply's two appearances as independent understates it:
    # one supply failure takes out both channels at once.
    independent = (1 - 0.99 * 0.9) ** 2
    assert independent == pytest.approx(0.011881)
    assert exact > 1.6 * independent
    assert tree.minimal_cut_sets() == [
        frozenset({"supply"}),
        frozenset({"pump A", "pump B"}),
    ]


def test_vote_gates():
    tree = FaultTree(
        {"top": ("vote", 2, ["a", "b", "c"])}, {"a": 0.1, "b": 0.2, "c": 0.3}
    )
    exact = 0.1 * 0.2 + 0.1 * 0.3 + 0.2 * 0.3 - 2 * 0.1 * 0.2 * 0.3
    assert tree.top_event_probability() == pytest.approx(exact, rel=1e-15)
    # A vote of 1 is an OR gate; of all its inputs, an AND gate.
    for k, reference in [(1, 1 - 0.9 * 0.8 * 0.7), (3, 0.1 * 0.2 * 0.3)]:
        tree = FaultTree(
            {"top": ("VOTE", k, ["a", "b", "c"])},
            {"a": 0.1, "b": 0.2, "c": 0.3},
        )
        assert tree.top_event_probability() == pytest.approx(
            reference, rel=1e-15
        )


def test_lifetime_models_give_probabilities_over_time():
    unit = W([100.0, 2.0])
    tree = FaultTree(
        {"top": ("or", ["a", "pair"]), "pair": ("and", ["b", "c"])},
        {"a": unit, "b": unit, "c": FixedEventProbability.from_params(0.2)},
    )
    assert not tree.is_fixed
    t = np.array([10.0, 50.0, 200.0])
    F = unit.ff(t)
    expected = 1 - (1 - F) * (1 - F * 0.2)
    np.testing.assert_allclose(tree.top_event_probability(t), expected)
    assert isinstance(tree.top_event_probability(50.0), float)
    assert tree.top_event_probability(50.0) == pytest.approx(expected[1])
    with pytest.raises(ValueError, match="t is required"):
        tree.top_event_probability()
    with pytest.raises(ValueError, match="single number"):
        tree.ranked_cut_sets(t)
    birnbaum = tree.birnbaum_importance(t)
    np.testing.assert_allclose(birnbaum["a"], 1 - F * 0.2)
    assert isinstance(tree.birnbaum_importance(50.0)["a"], float)


def test_single_input_gates_and_events_feeding_the_top_directly():
    tree = FaultTree(
        {"top": ("or", ["only"]), "only": ("and", ["a"])}, {"a": 0.25}
    )
    assert tree.top_event_probability() == 0.25
    assert tree.minimal_cut_sets() == [frozenset({"a"})]


# -- random trees against the definition --------------------------------------


def cut_set_union(probabilities, cut_sets, event) -> float:
    """The probability that every event of some minimal cut set containing
    ``event`` has occurred, over every combination of the events."""
    events = list(probabilities)
    total = 0.0
    for state in itertools.product([False, True], repeat=len(events)):
        occurred = {x for x, o in zip(events, state) if o}
        if any(event in c and c <= occurred for c in cut_sets):
            total += math.prod(
                probabilities[x] if o else 1 - probabilities[x]
                for x, o in zip(events, state)
            )
    return total


@pytest.mark.parametrize("seed", range(150))
def test_random_trees_match_the_definition(seed):
    rng = np.random.default_rng(seed)
    gates, probabilities, top = random_tree(
        rng, int(rng.integers(3, 9)), int(rng.integers(1, 7))
    )
    tree = FaultTree(gates, probabilities)
    assert tree.top == top
    probability, cut_sets, path_sets = enumerated(gates, probabilities, top)
    assert tree.top_event_probability() == pytest.approx(
        probability, rel=1e-12, abs=1e-15
    )
    assert set(tree.minimal_cut_sets()) == cut_sets
    assert set(tree.minimal_path_sets()) == path_sets
    for size in range(len(probabilities) + 1):
        for occurred in itertools.combinations(probabilities, size):
            assert tree.occurs(occurred) == occurs_by_hand(
                gates, top, set(occurred)
            )

    # Importance measures, from conditional probabilities enumerated.
    birnbaum = tree.birnbaum_importance()
    criticality = tree.criticality_importance()
    raw = tree.risk_achievement_worth()
    rrw = tree.risk_reduction_worth()
    fv = tree.fussell_vesely()
    rare = tree.fussell_vesely(method="rare_event")
    for e, q in probabilities.items():
        given, _, _ = enumerated(gates, probabilities, top, {e: True})
        given_not, _, _ = enumerated(gates, probabilities, top, {e: False})
        assert birnbaum[e] == pytest.approx(given - given_not, abs=1e-13)
        if probability > 0:
            assert criticality[e] == pytest.approx(
                (given - given_not) * q / probability, abs=1e-12
            )
            assert raw[e] == pytest.approx(given / probability, rel=1e-11)
            share = sum(
                math.prod(probabilities[x] for x in c)
                for c in cut_sets
                if e in c
            )
            assert rare[e] == pytest.approx(share / probability, rel=1e-11)
            union = cut_set_union(probabilities, cut_sets, e)
            assert fv[e] == pytest.approx(
                union / probability, rel=1e-11, abs=1e-15
            )
        if given_not > 0:
            assert rrw[e] == pytest.approx(probability / given_not, rel=1e-11)

    # Saved and restored.
    restored = FaultTree.from_json(tree.to_json())
    assert restored.gates == tree.gates
    assert restored.top == tree.top
    assert restored.top_event_probability() == tree.top_event_probability()


@pytest.mark.parametrize("seed", range(150))
def test_random_trees_convert_to_diagrams(seed):
    # Repeated events and gates included: a diagram's repeated node is the
    # one component wherever it is drawn.
    rng = np.random.default_rng(1000 + seed)
    gates, probabilities, top = random_tree(
        rng, int(rng.integers(3, 8)), int(rng.integers(1, 6))
    )
    tree = FaultTree(gates, probabilities)
    probability = tree.top_event_probability()
    rbd = tree.to_rbd()
    assert 1 - rbd.sf() == pytest.approx(probability, rel=1e-12, abs=1e-15)
    junctions = {
        n for n, m in rbd.reliabilities.items() if m is PerfectReliability
    }
    assert {c for c in rbd.get_min_cut_sets() if not c & junctions} == set(
        tree.minimal_cut_sets()
    )
    back = FaultTree.from_rbd(rbd)
    assert back.top_event_probability() == pytest.approx(
        probability, rel=1e-12, abs=1e-15
    )
    assert set(back.minimal_cut_sets()) == set(tree.minimal_cut_sets())


def test_events_the_logic_absorbs_leave_no_gate_behind():
    # #170: given e4 and e6, the vote needs e5, so e1 and e2 cannot affect
    # the top event. The diagram's module of them made a gate that nothing
    # used, which the tree refused.
    gates = {
        "g0": ("and", ["e6", "e4", "g2"]),
        "g2": ("vote", 3, ["e4", "g4", "g3", "e5"]),
        "g4": ("vote", 2, ["e5", "e6"]),
        "g3": ("and", ["e1", "e2"]),
    }
    tree = FaultTree(gates, {e: 0.3 for e in ["e1", "e2", "e4", "e5", "e6"]})
    back = FaultTree.from_rbd(tree.to_rbd())
    assert list(back.gates) == ["TOP"]
    kind, inputs = back.gates["TOP"]
    assert kind == "and" and sorted(inputs) == ["e4", "e5", "e6"]
    assert set(back.events) == {"e4", "e5", "e6"}
    assert back.top_event_probability() == pytest.approx(0.3**3, rel=1e-14)


@pytest.mark.parametrize("seed", [335, 749])
def test_larger_random_trees_convert_back(seed):
    # The two of a thousand random trees of seven events and six gates
    # that #170's gate left behind; the gates kept are numbered in turn.
    gates, probabilities, top = random_tree(np.random.default_rng(seed), 7, 6)
    tree = FaultTree(gates, probabilities)
    back = FaultTree.from_rbd(tree.to_rbd())
    assert back.top_event_probability() == pytest.approx(
        tree.top_event_probability(), rel=1e-12
    )
    assert set(back.minimal_cut_sets()) == set(tree.minimal_cut_sets())
    names = [g for g in back.gates if g != "TOP"]
    assert names == [f"G{i}" for i in range(1, len(names) + 1)]


def test_trees_without_repeats_always_convert():
    for seed in range(200):
        rng = np.random.default_rng(5000 + seed)
        # A tree: every gate and event feeds one gate.
        events = [f"e{i}" for i in range(int(rng.integers(2, 9)))]
        pool = list(events)
        gates = {}
        g = 0
        while len(pool) > 1:
            size = int(rng.integers(2, min(4, len(pool)) + 1))
            chosen = [
                pool.pop(int(rng.integers(len(pool)))) for _ in range(size)
            ]
            kind = str(rng.choice(["or", "and", "vote"]))
            name = f"g{g}"
            g += 1
            if kind == "vote":
                gates[name] = ("vote", int(rng.integers(1, size + 1)), chosen)
            else:
                gates[name] = (kind, chosen)
            pool.append(name)
        if not gates:
            gates["g0"] = ("or", pool)
        probabilities = {e: float(rng.uniform(0.05, 0.5)) for e in events}
        tree = FaultTree(gates, probabilities)
        rbd = tree.to_rbd()
        assert 1 - rbd.sf() == pytest.approx(
            tree.top_event_probability(), rel=1e-12
        )


def test_a_shared_gate_converts_exactly():
    # "shared" feeds an OR and an AND below a vote: drawn in both places,
    # its events are repeated nodes, the one component each.
    tree = FaultTree(
        {
            "top": ("vote", 2, ["g1", "g2", "c"]),
            "g1": ("or", ["a", "shared"]),
            "g2": ("and", ["b", "shared"]),
            "shared": ("or", ["x", "y"]),
        },
        {"a": 0.1, "b": 0.2, "c": 0.3, "x": 0.05, "y": 0.02},
    )
    probability, cut_sets, _ = enumerated(tree.gates, dict(tree.events), "top")
    assert tree.top_event_probability() == pytest.approx(
        probability, rel=1e-13
    )
    assert set(tree.minimal_cut_sets()) == cut_sets
    rbd = tree.to_rbd()
    assert rbd.repeated == {"x (2)": "x", "y (2)": "y"}
    assert 1 - rbd.sf() == pytest.approx(probability, rel=1e-13)


def test_a_bridge_round_trips():
    # The tree of a bridge repeats every event across its cut sets; it
    # converts back to a diagram with the same logic.
    fail = FixedEventProbability.from_params
    bridge = NonRepairableRBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("a", "c"),
            ("b", "c"),
            ("a", "d"),
            ("c", "d"),
            ("b", "e"),
            ("d", "t"),
            ("e", "t"),
        ],
        {n: fail(0.1 + 0.05 * i) for i, n in enumerate("abcde")},
    )
    back = FaultTree.from_rbd(bridge).to_rbd()
    assert back.sf() == pytest.approx(bridge.sf(), rel=1e-13)
    assert back.get_min_cut_sets() == bridge.get_min_cut_sets()


def test_a_repeated_event_a_diagram_can_draw_converts():
    tree = FaultTree(
        {
            "no output": ("and", ["channel A", "channel B"]),
            "channel A": ("or", ["supply", "pump A"]),
            "channel B": ("or", ["supply", "pump B"]),
        },
        {"supply": 0.01, "pump A": 0.1, "pump B": 0.1},
    )
    rbd = tree.to_rbd()
    assert rbd.repeated  # the supply's second appearance
    assert 1 - rbd.sf() == pytest.approx(tree.top_event_probability())


def test_to_rbd_names_avoid_the_events_names():
    tree = FaultTree(
        {"top": ("vote", 2, ["input", "output", "top (vote)"])},
        {"input": 0.1, "output": 0.2, "top (vote)": 0.3},
    )
    rbd = tree.to_rbd()
    assert rbd.input_node not in tree.events
    assert rbd.output_node not in tree.events
    assert 1 - rbd.sf() == pytest.approx(tree.top_event_probability())


# -- diagrams to trees --------------------------------------------------------


@pytest.mark.parametrize("seed", range(120))
def test_random_diagrams_become_trees_with_the_same_logic(seed):
    rng = np.random.default_rng(7000 + seed)
    n = int(rng.integers(2, 9))
    edges, k = random_diagram(rng, n)
    models = {
        i: FixedEventProbability.from_params(float(rng.uniform(0.02, 0.5)))
        for i in range(n)
    }
    rbd = NonRepairableRBD(edges, models, k=k or None)
    if rbd._decomposition().always_works:
        with pytest.raises(ValueError, match="cannot fail"):
            FaultTree.from_rbd(rbd)
        return
    tree = FaultTree.from_rbd(rbd)
    assert tree.top == "TOP"
    assert tree.top_event_probability() == pytest.approx(
        1 - rbd.sf(), rel=1e-11, abs=1e-15
    )
    assert set(tree.minimal_cut_sets()) == rbd.get_min_cut_sets()
    # Irrelevant nodes are left out; every other is an event.
    assert set(tree.events) == set(rbd._decomposition().nodes)
    for measure in (
        "birnbaum_importance",
        "criticality_importance",
        "risk_achievement_worth",
        "risk_reduction_worth",
        "fussell_vesely",
    ):
        ours = getattr(tree, measure)()
        with np.errstate(divide="ignore", invalid="ignore"):
            theirs = getattr(rbd, measure)()
        for e in tree.events:
            assert ours[e] == pytest.approx(theirs[e], rel=1e-8, abs=1e-12), (
                measure,
                e,
            )


def test_a_diagram_over_time():
    unit = W([100.0, 2.0])
    rbd = NonRepairableRBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("s", "c"),
            ("a", "v"),
            ("b", "v"),
            ("c", "v"),
            ("v", "d"),
            ("d", "t"),
        ],
        {"a": unit, "b": unit, "c": unit, "v": PerfectReliability, "d": unit},
        k={"v": 2},
    )
    tree = FaultTree.from_rbd(rbd)
    # The perfect junction drops out: it can never fail.
    assert tree.gates == {
        "G1": ("vote", 2, ["a", "b", "c"]),
        "TOP": ("or", ["G1", "d"]),
    }
    t = np.array([20.0, 80.0])
    np.testing.assert_allclose(tree.top_event_probability(t), 1 - rbd.sf(t))


def test_a_bridge_becomes_an_or_of_its_cut_sets():
    p = FixedEventProbability.from_params
    rbd = NonRepairableRBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("a", "c"),
            ("b", "c"),
            ("a", "d"),
            ("c", "d"),
            ("b", "e"),
            ("d", "t"),
            ("e", "t"),
        ],
        {n: p(0.1) for n in "abcde"},
    )
    tree = FaultTree.from_rbd(rbd)
    top_inputs = tree.gates["TOP"][1]
    assert tree.gates["TOP"][0] == "or"
    assert len(top_inputs) == len(rbd.get_min_cut_sets())
    assert tree.top_event_probability() == pytest.approx(1 - rbd.sf())


def test_one_node_diagrams_and_names_that_clash():
    rbd = NonRepairableRBD(
        [("s", "TOP"), ("TOP", "t")],
        {"TOP": FixedEventProbability.from_params(0.2)},
    )
    tree = FaultTree.from_rbd(rbd)
    assert tree.top != "TOP" and tree.gates[tree.top] == ("or", ["TOP"])
    assert tree.top_event_probability() == pytest.approx(0.2)


def test_diagrams_that_cannot_fail_or_be_converted():
    p = FixedEventProbability.from_params
    with pytest.raises(ValueError, match="cannot fail"):
        FaultTree.from_rbd(
            NonRepairableRBD(
                [("s", "a"), ("a", "t"), ("s", "t")], {"a": p(0.1)}
            )
        )
    with pytest.raises(ValueError, match="cannot fail"):
        FaultTree.from_rbd(
            NonRepairableRBD(
                [("s", "a"), ("a", "t")], {"a": PerfectReliability}
            )
        )
    with pytest.raises(TypeError, match="NonRepairableRBD"):
        FaultTree.from_rbd("not a diagram")


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "gates, events, match",
    [
        ({}, {"a": 0.1}, "at least one gate"),
        ({"g": ("xor", ["a", "b"])}, {"a": 0.1, "b": 0.1}, "must be"),
        ({"g": "or"}, {"a": 0.1}, "must be"),
        ({"g": ("or",)}, {"a": 0.1}, "must be"),
        ({"g": ("or", [])}, {"a": 0.1}, "non-empty"),
        ({"g": ("or", "a")}, {"a": 0.1}, "non-empty list"),
        ({"g": ("or", ["a", "a"])}, {"a": 0.1}, "more than once"),
        ({"g": ("vote", 4, ["a", "b"])}, {"a": 0.1, "b": 0.1}, "between 1"),
        ({"g": ("vote", 0, ["a", "b"])}, {"a": 0.1, "b": 0.1}, "between 1"),
        ({"g": ("vote", 1.5, ["a", "b"])}, {"a": 0.1, "b": 0.1}, "integer"),
        ({"g": ("vote", ["a", "b"])}, {"a": 0.1, "b": 0.1}, "must be"),
        ({"g": ("or", ["a", "z"])}, {"a": 0.1}, "neither a gate nor an event"),
        ({"g": ("or", ["a"])}, {"a": 1.5}, r"in \[0, 1\]"),
        ({"g": ("or", ["a"])}, {"a": "x"}, "probability or a lifetime"),
        ({"g": ("or", ["a"])}, {"a": 0.1, "g": 0.2}, "both gates and events"),
        ({"g": ("or", ["a"])}, {"a": 0.1, "b": 0.2}, "not below the top"),
        (
            {"g": ("or", ["a"]), "h": ("or", ["b"])},
            {"a": 0.1, "b": 0.1},
            "Give the top event",
        ),
        (
            {"g": ("or", ["h"]), "h": ("or", ["g"])},
            {},
            "gates form a loop through",
        ),
    ],
)
def test_invalid_trees_are_rejected(gates, events, match):
    with pytest.raises(ValueError, match=match):
        FaultTree(gates, events)


def test_loops_and_bad_top_events_are_rejected():
    with pytest.raises(ValueError, match="loop"):
        FaultTree(
            {
                "top": ("or", ["g", "a"]),
                "g": ("and", ["h", "b"]),
                "h": ("or", ["g"]),
            },
            {"a": 0.1, "b": 0.1},
            top="top",
        )
    with pytest.raises(ValueError, match="must be a gate"):
        FaultTree({"g": ("or", ["a"])}, {"a": 0.1}, top="a")
    with pytest.raises(ValueError, match="Unknown event"):
        cooling().occurs({"pump 3"})
    with pytest.raises(ValueError, match="Not a fault tree"):
        FaultTree.from_dict({"type": "NonRepairableRBD"})


def test_models_and_names_survive_saving():
    unit = W([100.0, 2.0])
    tree = FaultTree(
        {("top", 1): ("vote", 2, ["a", "b", (3, "c")])},
        {
            "a": unit,
            "b": 0.2,
            (3, "c"): FixedEventProbability.from_params(0.1),
        },
    )
    restored = FaultTree.from_json(tree.to_json())
    assert restored.top == ("top", 1)
    assert set(restored.events) == {"a", "b", (3, "c")}
    t = np.array([10.0, 60.0])
    np.testing.assert_array_equal(
        restored.top_event_probability(t), tree.top_event_probability(t)
    )


def test_importance_where_the_top_event_cannot_occur():
    tree = FaultTree({"top": ("and", ["a", "b"])}, {"a": 0.0, "b": 0.5})
    assert tree.top_event_probability() == 0.0
    assert math.isnan(tree.criticality_importance()["b"])
    assert math.isnan(tree.fussell_vesely()["b"])
    assert tree.birnbaum_importance()["b"] == 0.0
    # Preventing "b" makes the top event impossible, as it already was.
    assert math.isnan(tree.risk_reduction_worth()["b"])


# -- common-cause groups (#184) -----------------------------------------------


def cooling_with_ccf(model, q_pump=0.05):
    from repyability import CCFGroup

    return FaultTree(
        {
            "no cooling": ("or", ["no flow", "valve"]),
            "no flow": ("and", ["pump 1", "pump 2"]),
        },
        {"pump 1": q_pump, "pump 2": q_pump, "valve": 0.01},
        ccf_groups=[CCFGroup(["pump 1", "pump 2"], model)],
    )


def test_a_beta_factor_pair_by_hand():
    from repyability import BetaFactor

    tree = cooling_with_ccf(BetaFactor(0.1))
    # The shared cause (beta Q) fails both pumps; without it each fails on
    # its own with (1 - beta) Q.
    Q, beta = 0.05, 0.1
    both = beta * Q + (1 - beta * Q) * ((1 - beta) * Q) ** 2
    assert tree.top_event_probability() == pytest.approx(
        1 - (1 - both) * (1 - 0.01), rel=1e-14
    )
    assert tree.sf() == pytest.approx((1 - both) * (1 - 0.01), rel=1e-14)
    # The pumps' cut set is the pair failing together, the shared cause
    # included: nearly three times the independent pair's 0.0025.
    ranked = dict(tree.ranked_cut_sets())
    assert ranked[frozenset({"pump 1", "pump 2"})] == pytest.approx(
        both, rel=1e-14
    )
    assert ranked[frozenset({"valve"})] == 0.01
    # The logic is the tree's: the cut sets are of basic events.
    assert tree.minimal_cut_sets() == [
        frozenset({"valve"}),
        frozenset({"pump 1", "pump 2"}),
    ]
    # A pump's occurrence says the other has likely occurred too.
    one = beta * Q + (1 - beta * Q) * (1 - beta) * Q
    given = both / one
    raw = tree.risk_achievement_worth()["pump 1"]
    assert raw == pytest.approx(
        (1 - (1 - given) * 0.99) / tree.top_event_probability(), rel=1e-12
    )


def same_groups(a, b) -> bool:
    return [(list(g.members), g.model) for g in a] == [
        (list(g.members), g.model) for g in b
    ]


GROUP_MODELS = [
    "beta",
    "beta-rate",
    "mgl",
    "mgl-independent",
    "mgl-rate",
]


def group_model(kind, size):
    from repyability import MGL, BetaFactor

    letters = (0.15, 0.4, 0.5)[: size - 1]
    return {
        "beta": lambda: BetaFactor(0.1),
        "beta-rate": lambda: BetaFactor(0.2, basis="rate"),
        "mgl": lambda: MGL(*letters),
        "mgl-independent": lambda: MGL(*letters, shocks="independent"),
        "mgl-rate": lambda: MGL(*letters, basis="rate"),
    }[kind]()


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("kind", GROUP_MODELS)
def test_random_trees_with_groups_agree_with_their_diagrams(seed, kind):
    from repyability import CCFGroup

    rng = np.random.default_rng(9000 + seed)
    gates, probabilities, top = random_tree(
        rng, int(rng.integers(4, 8)), int(rng.integers(1, 6))
    )
    events = list(probabilities)
    size = int(rng.integers(2, min(4, len(events)) + 1))
    members = [events[i] for i in rng.choice(len(events), size, False)]
    groups = [CCFGroup(members, group_model(kind, size))]
    q = float(rng.uniform(0.005, 0.1))
    probabilities.update({m: q for m in members})
    rest = [e for e in events if e not in members]
    if len(rest) >= 2 and rng.random() < 0.5:
        # A second group, of another size.
        second = rest[:2]
        groups.append(CCFGroup(second, group_model(kind, 2)))
        probabilities.update({m: q / 2 for m in second})
    tree = FaultTree(gates, probabilities, ccf_groups=groups)
    rbd = tree.to_rbd()
    assert rbd.ccf_groups == groups
    probability = tree.top_event_probability()
    assert probability == pytest.approx(rbd.ff(), rel=1e-12, abs=1e-16)
    assert tree.sf() == pytest.approx(rbd.sf(), rel=1e-12, abs=1e-16)
    for measure in (
        "birnbaum_importance",
        "criticality_importance",
        "risk_achievement_worth",
        "risk_reduction_worth",
        "fussell_vesely",
    ):
        ours = getattr(tree, measure)()
        with np.errstate(divide="ignore", invalid="ignore"):
            theirs = getattr(rbd, measure)()
        for e in tree.events:
            assert ours[e] == pytest.approx(
                theirs[e], rel=1e-9, abs=1e-14, nan_ok=True
            ), (measure, e)
    ours = tree.fussell_vesely(method="rare_event")
    theirs = rbd.fussell_vesely(method="rare_event")
    for e in tree.events:
        assert ours[e] == pytest.approx(theirs[e], rel=1e-9, abs=1e-14)

    # A cut set's probability is its events' occurring together, enumerated
    # over the groups' outcomes.
    p, qs = tree._event_probabilities(np.array([1.0]))
    outcomes = list(shock_outcomes(tree.ccf_groups, p, qs))
    for cut, value in tree.ranked_cut_sets():
        together = sum(
            float(np.ravel(w)[0]) * math.prod(f[e][0] for e in cut)
            for w, _, f in outcomes
        )
        assert value == pytest.approx(together, rel=1e-12)

    # Back to a tree, and saved and restored. An event the logic absorbs
    # is left out of the diagram's tree, and so is a group of only such
    # events; one with some keeps them, as events the logic absorbs (#237).
    relevant = set().union(*tree.minimal_cut_sets())
    some = [g for g in groups if relevant & set(g.members)]
    back = FaultTree.from_rbd(rbd)
    assert same_groups(back.ccf_groups, some)
    assert set(back.events) == relevant.union(*(g.members for g in some))
    assert back.top_event_probability() == pytest.approx(
        probability, rel=1e-12, abs=1e-16
    )
    restored = FaultTree.from_json(tree.to_json())
    assert same_groups(restored.ccf_groups, groups)
    assert restored.top_event_probability() == probability


def test_a_group_is_its_cause_drawn_as_a_repeated_event():
    from repyability import BetaFactor, CCFGroup

    # Without groups, an AND of the pair is their product; a group makes it
    # the shared cause's probability and more.
    gates = {"top": ("and", ["a", "b"])}
    plain = FaultTree(gates, {"a": 0.01, "b": 0.01})
    grouped = FaultTree(
        gates,
        {"a": 0.01, "b": 0.01},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    assert plain.top_event_probability() == pytest.approx(1e-4)
    assert grouped.top_event_probability() == pytest.approx(
        0.001 + 0.999 * 0.009**2, rel=1e-14
    )
    # The same as the cause drawn as a repeated event of its own:
    # OR(a's own, cause) and OR(b's own, cause), the cause occurring with
    # beta Q and each pump on its own with (1 - beta) Q.
    explicit = FaultTree(
        {
            "top": ("and", ["A", "B"]),
            "A": ("or", ["a alone", "cause"]),
            "B": ("or", ["b alone", "cause"]),
        },
        {"a alone": 0.009, "b alone": 0.009, "cause": 0.001},
    )
    assert grouped.top_event_probability() == pytest.approx(
        explicit.top_event_probability(), rel=1e-13
    )


def test_groups_over_time():
    from repyability import MGL, CCFGroup

    unit = W([1000.0, 1.5])
    tree = FaultTree(
        {"top": ("or", ["trains", "d"]), "trains": ("vote", 2, list("abc"))},
        {"a": unit, "b": unit, "c": unit, "d": W([5000.0, 1.2])},
        ccf_groups=[CCFGroup(list("abc"), MGL(0.1, 0.3, basis="rate"))],
    )
    t = np.array([50.0, 200.0, 800.0])
    rbd = tree.to_rbd()
    np.testing.assert_allclose(
        tree.top_event_probability(t), rbd.ff(t), rtol=1e-12
    )
    for e, v in tree.birnbaum_importance(t).items():
        np.testing.assert_allclose(v, rbd.birnbaum_importance(t)[e], rtol=1e-9)
    # Each time's value is the scalar one.
    assert tree.top_event_probability(200.0) == pytest.approx(
        tree.top_event_probability(t)[1], rel=1e-15
    )
    # More often than the trains failing independently while failures are
    # rare (by 800 h, when half the units have failed, the shared causes'
    # bunching of failures no longer adds to a vote's).
    independent = FaultTree(tree.gates, tree.events)
    assert np.all(
        tree.top_event_probability(t[:2])
        > independent.top_event_probability(t[:2])
    )


def test_a_repeated_grouped_event_converts():
    from repyability import BetaFactor, CCFGroup

    # Event "a" feeds two gates: the diagram draws it twice, the group
    # takes its first place.
    tree = FaultTree(
        {
            "top": ("and", ["g1", "g2"]),
            "g1": ("or", ["a", "b"]),
            "g2": ("or", ["a", "c"]),
        },
        {"a": 0.02, "b": 0.02, "c": 0.05},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
    )
    rbd = tree.to_rbd()
    assert rbd.repeated
    assert tree.top_event_probability() == pytest.approx(rbd.ff(), rel=1e-13)
    # By hand: a's occurring decides the top event; otherwise b and c must.
    Q, beta = 0.02, 0.2
    own = (1 - beta) * Q
    shock = beta * Q
    a = shock + (1 - shock) * own
    b_not_a = (1 - shock) * (1 - own) * own
    assert tree.top_event_probability() == pytest.approx(
        a + b_not_a * 0.05, rel=1e-13
    )


def test_groups_are_checked():
    from repyability import BetaFactor, CCFGroup

    gates = {"top": ("or", ["a", "b", "c"])}
    events = {"a": 0.01, "b": 0.01, "c": 0.02}
    with pytest.raises(ValueError, match="not a basic event"):
        FaultTree(
            gates, events, ccf_groups=[CCFGroup(["a", "z"], BetaFactor(0.1))]
        )
    with pytest.raises(ValueError, match="more than one CCF group"):
        FaultTree(
            gates,
            {**events, "c": 0.01},
            ccf_groups=[
                CCFGroup(["a", "b"], BetaFactor(0.1)),
                CCFGroup(["b", "c"], BetaFactor(0.1)),
            ],
        )
    with pytest.raises(ValueError, match="not symmetric"):
        FaultTree(
            gates, events, ccf_groups=[CCFGroup(["a", "c"], BetaFactor(0.1))]
        )
    with pytest.raises(ValueError, match="not symmetric"):
        FaultTree(
            gates,
            {"a": W([10.0, 2.0]), "b": W([10.0, 3.0]), "c": 0.1},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
    with pytest.raises(ValueError, match="CCFGroup instances"):
        FaultTree(gates, events, ccf_groups=[("a", "b")])
    # A gate is no basic event.
    with pytest.raises(ValueError, match="not a basic event"):
        FaultTree(
            {"top": ("or", ["g", "c"]), "g": ("and", ["a", "b"])},
            {"a": 0.01, "b": 0.01, "c": 0.01},
            ccf_groups=[CCFGroup(["g", "c"], BetaFactor(0.1))],
        )
    # Keyword only.
    with pytest.raises(TypeError):
        FaultTree(gates, events, "top", [])


def test_a_probability_split_warns_beyond_its_range_once():
    import warnings

    from repyability import BetaFactor

    tree = cooling_with_ccf(BetaFactor(0.1), q_pump=0.3)
    with pytest.warns(UserWarning, match="Q = 0.3"):
        tree.top_event_probability()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tree.top_event_probability()
        tree.fussell_vesely()
        cooling_with_ccf(BetaFactor(0.1, basis="rate"), 0.3).ff()


def test_a_diagrams_groups_become_the_trees():
    from repyability import MGL, CCFGroup

    p = FixedEventProbability.from_params
    group = CCFGroup(["p1", "p2", "p3"], MGL(0.1, 0.3))
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("s", "p3")]
        + [("p1", "v"), ("p2", "v"), ("p3", "v"), ("v", "t")],
        {"p1": p(0.05), "p2": p(0.05), "p3": p(0.05), "v": p(0.001)},
        ccf_groups=[group],
    )
    tree = FaultTree.from_rbd(rbd)
    assert tree.ccf_groups == [group]
    assert tree.top_event_probability() == pytest.approx(rbd.ff(), rel=1e-13)
    for e, v in tree.criticality_importance().items():
        assert v == pytest.approx(rbd.criticality_importance()[e], rel=1e-9)

    # A member that cannot affect the system stays an event of the tree,
    # one its logic absorbs, for the group's shocks to strike it with the
    # others (#237).
    irrelevant = NonRepairableRBD(
        [("s", "x"), ("s", "j"), ("x", "m"), ("j", "m"), ("m", "a")]
        + [("a", "t")],
        {"x": p(0.05), "j": PerfectReliability, "m": PerfectReliability}
        | {"a": p(0.05)},
        ccf_groups=[CCFGroup(["a", "x"], MGL(0.1))],
    )
    kept = FaultTree.from_rbd(irrelevant)
    assert set(kept.events) == {"a", "x"}
    assert kept.minimal_cut_sets() == [frozenset({"a"})] or set(
        map(frozenset, kept.minimal_cut_sets())
    ) == {frozenset({"a"})}
    assert kept.top_event_probability() == pytest.approx(
        irrelevant.ff(), rel=1e-13
    )
    # A group none of whose members can is left out with them.
    p2 = {n: p(0.05) for n in ("x", "y")}
    unaffected = NonRepairableRBD(
        [("s", "x"), ("x", "m"), ("s", "y"), ("y", "m"), ("s", "j")]
        + [("j", "m"), ("m", "a"), ("a", "t")],
        {**p2, "j": PerfectReliability, "m": PerfectReliability}
        | {"a": p(0.05)},
        ccf_groups=[CCFGroup(["x", "y"], MGL(0.1))],
    )
    tree = FaultTree.from_rbd(unaffected)
    assert tree.ccf_groups == [] and set(tree.events) == {"a"}
    assert tree.top_event_probability() == pytest.approx(unaffected.ff())
