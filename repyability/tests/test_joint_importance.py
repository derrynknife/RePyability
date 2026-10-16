"""The joint (second-order) importance of pairs of nodes (#194): whether
improving two together is worth more than improving each.

Checked against the system worked out with each pair held working and
failed, against its meaning (the system's change when both improve, less
the changes when each does, is the joint importance times the two
improvements, exactly, as the structure is multilinear), and across the
classes: a fault tree's equals its diagram's. With common-cause groups
(#317) each pair is conditioned on its states: checked against
conditioning on every combination of every group's outcomes, and with
the shared causes at zero against the diagram without groups."""

import itertools

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    FaultTree,
    NonRepairableRBD,
    RepairableRBD,
)
from repyability.rbd.ccf import shock_outcomes

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = FixedEventProbability.from_params

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
Q = {"a": 0.1, "b": 0.2, "c": 0.05, "d": 0.3, "e": 0.15}


def bridge(q=Q):
    return NonRepairableRBD(BRIDGE, {n: F(v) for n, v in q.items()})


def system(rbd, values):
    return float(np.ravel(rbd.system_probability(values))[0])


def test_each_pair_held_working_and_failed():
    rbd = bridge()
    joint = rbd.joint_importance()
    assert list(joint) == list(itertools.combinations("abcde", 2))
    base = {n: 1 - v for n, v in Q.items()}
    for (i, j), value in joint.items():
        held = {
            (a, b): system(rbd, {**base, i: a, j: b})
            for a in (0.0, 1.0)
            for b in (0.0, 1.0)
        }
        expected = held[1, 1] - held[1, 0] - held[0, 1] + held[0, 0]
        assert value == pytest.approx(expected, abs=1e-14)


def test_the_change_of_both_is_the_sum_of_each_and_the_joint_part():
    # R multilinear: R(p + u e_i + v e_j) - R(p + u e_i) - R(p + v e_j) +
    # R(p) = JRI(i, j) u v, exactly.
    rbd = bridge()
    joint = rbd.joint_importance()
    base = {n: 1 - v for n, v in Q.items()}
    u, v = 0.05, 0.07
    for (i, j), value in joint.items():
        both = system(rbd, {**base, i: base[i] + u, j: base[j] + v})
        one = system(rbd, {**base, i: base[i] + u})
        other = system(rbd, {**base, j: base[j] + v})
        assert both - one - other + system(rbd, base) == pytest.approx(
            value * u * v, abs=1e-14
        )


def test_series_complements_and_parallel_substitutes():
    series = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": F(0.1), "b": F(0.2)}
    )
    assert series.joint_importance()[("a", "b")] == pytest.approx(1.0)
    parallel = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": F(0.1), "b": F(0.2)},
    )
    assert parallel.joint_importance()[("a", "b")] == pytest.approx(-1.0)


def test_over_time_and_with_nodes_held():
    rbd = NonRepairableRBD(
        BRIDGE, {n: W([60 + 15 * k, 1.5]) for k, n in enumerate("abcde")}
    )
    x = np.array([10.0, 50.0])
    joint = rbd.joint_importance(x)
    assert joint[("a", "b")].shape == (2,)
    for k, t in enumerate(x):
        at = rbd.joint_importance(t)
        for pair, value in at.items():
            assert value == pytest.approx(joint[pair][k], abs=1e-14)
    held = rbd.joint_importance(50.0, working_nodes=["c"])
    assert held[("a", "c")] == 0.0 and held[("c", "d")] == 0.0
    # With c working the bridge is a or b, then d or e: the joint importance
    # of a and d is q_b q_e.
    sf = {n: float(m.sf(50.0)) for n, m in rbd.reliabilities.items()}
    assert held[("a", "d")] == pytest.approx(
        (1 - sf["b"]) * (1 - sf["e"]), rel=1e-12
    )


def test_a_repeated_node_is_one_component():
    shared = NonRepairableRBD(
        [
            ("in", "a"),
            ("a", "psu"),
            ("psu", "out"),
            ("in", "b"),
            ("b", "psu2"),
            ("psu2", "out"),
        ],
        {"a": F(0.1), "b": F(0.2), "psu": F(0.05), "psu2": "psu"},
    )
    joint = shared.joint_importance()
    # Its pairs are the components', once each: psu2 is psu.
    pairs = {frozenset(pair): value for pair, value in joint.items()}
    assert set(pairs) == {
        frozenset({"a", "b"}),
        frozenset({"a", "psu"}),
        frozenset({"b", "psu"}),
    }
    # R = R_psu * (1 - q_a q_b): psu is a complement of each unit.
    assert pairs[frozenset({"a", "psu"})] == pytest.approx(0.2)
    assert pairs[frozenset({"a", "b"})] == pytest.approx(-0.95)


def conditioned_pairs(rbd, x, held=()):
    """The joint importance of each pair by conditioning on every
    combination of every group's outcomes: the system given each of the
    pair working and failed (a member's state through the outcomes, a
    node outside the groups held), ``R(1, 1) - R(1, 0) - R(0, 1) + R(0,
    0)``; 0 for a pair with a node held."""
    p = {n: np.atleast_1d(rbd.reliabilities[n].sf(x)) for n in rbd.nodes}
    q = {n: np.atleast_1d(rbd.reliabilities[n].ff(x)) for n in rbd.nodes}
    for node in held:  # held working
        p[node], q[node] = np.ones_like(p[node]), np.zeros_like(q[node])
    grouped = {m for g in rbd.ccf_groups for m in g.members}
    out = {}
    for i, j in itertools.combinations(rbd.nodes, 2):
        if i in held or j in held:
            out[i, j] = np.zeros_like(p[i])
            continue
        sums = {}
        for w, given, failing in shock_outcomes(rbd.ccf_groups, p, q):
            for a, b in itertools.product((True, False), repeat=2):
                chance = w
                hold_p, hold_q = dict(given), dict(failing)
                for node, up in ((i, a), (j, b)):
                    if node in grouped:
                        chance = chance * (given if up else failing)[node]
                    hold_p[node] = (1.0 if up else 0.0) * np.ones_like(p[i])
                    hold_q[node] = 1.0 - hold_p[node]
                up_, _ = rbd._system_probabilities(hold_p, hold_q)
                was = sums.get((a, b), (0.0, 0.0))
                sums[a, b] = (was[0] + chance, was[1] + chance * up_)
        R = {combo: up_ / chance for combo, (chance, up_) in sums.items()}
        out[i, j] = (
            R[True, True] - R[True, False] - R[False, True] + R[False, False]
        )
    return out


GROUPED_BRIDGE = {
    "a": E([0.01]),
    "b": E([0.01]),
    "c": W([80, 1.5]),
    "d": E([0.01]),
    "e": E([0.01]),
}


@pytest.mark.parametrize(
    "groups",
    [
        [CCFGroup(["a", "b"], BetaFactor(0.2))],
        [CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
        [
            CCFGroup(["a", "b"], BetaFactor(0.2)),
            CCFGroup(["d", "e"], BetaFactor(0.3, basis="rate")),
        ],
        [CCFGroup(["a", "d", "e"], MGL(0.1, 0.3))],
        [CCFGroup(["a", "d", "e"], MGL(0.1, 0.3, shocks="independent"))],
    ],
)
def test_with_common_cause_groups_each_pair_is_conditioned_on(groups):
    # Pairs within a group, across groups, of a member and a node outside
    # the groups, and of two outside, at times and at one; with nodes
    # held, their pairs are 0 and the rest conditioned on them.
    rbd = NonRepairableRBD(BRIDGE, GROUPED_BRIDGE, ccf_groups=groups)
    x = np.array([5.0, 40.0, 150.0])
    joint = rbd.joint_importance(x)
    expected = conditioned_pairs(rbd, x)
    assert set(joint) == set(expected)
    for pair, value in expected.items():
        np.testing.assert_allclose(joint[pair], value, rtol=1e-9, atol=1e-14)
    at = rbd.joint_importance(40.0)
    for pair, value in at.items():
        assert value == pytest.approx(joint[pair][1], abs=1e-14)
    held = rbd.joint_importance(x, working_nodes=["c"])
    for pair, value in conditioned_pairs(rbd, x, held={"c"}).items():
        np.testing.assert_allclose(held[pair], value, rtol=1e-9, atol=1e-14)
    assert np.all(held[("a", "c")] == 0.0)
    # A member cannot be held, as for the other measures.
    with pytest.raises(NotImplementedError, match="CCF group member"):
        rbd.joint_importance(x, broken_nodes=["a"])
    # A fault tree of the diagram has the same pairs.
    tree = FaultTree.from_rbd(rbd)
    theirs = tree.joint_importance(x)
    for pair, value in joint.items():
        np.testing.assert_allclose(theirs[pair], value, rtol=1e-9, atol=1e-14)


def test_shared_causes_at_zero_give_the_pairs_without_groups():
    x = np.array([5.0, 40.0, 150.0])
    plain = NonRepairableRBD(BRIDGE, GROUPED_BRIDGE).joint_importance(x)
    for model in (BetaFactor(0.0), BetaFactor(0.0, basis="rate")):
        grouped = NonRepairableRBD(
            BRIDGE,
            GROUPED_BRIDGE,
            ccf_groups=[
                CCFGroup(["a", "b"], model),
                CCFGroup(["d", "e"], model),
            ],
        ).joint_importance(x)
        for pair, value in plain.items():
            np.testing.assert_allclose(
                grouped[pair], value, rtol=1e-10, atol=1e-14
            )


def unit(rate, repair=1.0):
    return {"reliability": E([rate]), "repairability": E([repair])}


PAIR_THEN_C = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]


def held_pairs(value, i, j):
    """``value(working, broken)`` with i and j held each way, combined."""
    up_up = value([i, j], [])
    up_down = value([i], [j])
    down_up = value([j], [i])
    down_down = value([], [i, j])
    return up_up - up_down - down_up + down_down


def test_a_repairable_diagram_in_the_long_run_and_over_time():
    rbd = RepairableRBD(
        PAIR_THEN_C,
        {
            "a": {"reliability": W([10, 2.5]), "repairability": E([1.0])},
            "b": unit(0.1),
            "c": unit(0.02, 0.5),
        },
    )
    long_run = rbd.joint_importance()
    for (i, j), value in long_run.items():
        expected = held_pairs(lambda w, b: rbd.mean_availability(w, b), i, j)
        assert value == pytest.approx(expected, abs=1e-12)
    x = np.array([2.0, 20.0])
    at = rbd.joint_importance(x=x)
    for (i, j), value in at.items():
        expected = held_pairs(
            lambda w, b: rbd.point_availability(x, w, b), i, j
        )
        np.testing.assert_allclose(value, expected, atol=1e-9)
    window = rbd.joint_importance(window=10.0)
    for (i, j), value in window.items():
        expected = held_pairs(
            lambda w, b: rbd.mission_availability(10.0, w, b), i, j
        )
        assert value == pytest.approx(expected, abs=1e-8)


def test_with_crews_each_pair_is_held_in_the_chain():
    rbd = RepairableRBD(
        PAIR_THEN_C,
        {"a": unit(0.1), "b": unit(0.1), "c": unit(0.02)},
        repair_crews=1,
    )
    joint = rbd.joint_importance()
    for (i, j), value in joint.items():
        expected = held_pairs(lambda w, b: rbd.mean_availability(w, b), i, j)
        assert value == pytest.approx(expected, abs=1e-12)


def test_a_repairable_diagram_with_groups():
    # With the shared cause at zero, the pairs without the group, in the
    # long run, at times and over a window (test_ccf_repairable_modules
    # checks a shared cause against conditioning on every combination).
    units = {"a": unit(0.1), "b": unit(0.1), "c": unit(0.02, 0.5)}
    plain = RepairableRBD(PAIR_THEN_C, units)
    none = RepairableRBD(
        PAIR_THEN_C,
        units,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.0))],
    )
    x = np.array([2.0, 20.0])
    for kwargs, rel in (
        ({}, 1e-9),
        ({"x": x}, 1e-5),
        ({"window": 10.0}, 1e-5),
    ):
        # Over time the groups' chains are followed numerically.
        ours = none.joint_importance(**kwargs)
        theirs = plain.joint_importance(**kwargs)
        assert set(ours) == set(theirs)
        for pair, value in theirs.items():
            np.testing.assert_allclose(ours[pair], value, rtol=rel)
    # A shared cause: with a down, b is likelier down too, so c is worth
    # less then, and more of a complement of a; the pair a, b is as
    # before, c alone remaining given their states.
    shared = RepairableRBD(
        PAIR_THEN_C,
        units,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    joint, plain_joint = shared.joint_importance(), plain.joint_importance()
    assert joint[("a", "c")] > plain_joint[("a", "c")] > 0
    assert joint[("a", "b")] == pytest.approx(plain_joint[("a", "b")])
    assert joint[("a", "b")] == pytest.approx(
        -float(shared.node_availability()["c"]), rel=1e-9
    )
    held = shared.joint_importance(working_nodes=["c"])
    assert held[("a", "c")] == 0.0 and held[("a", "b")] == pytest.approx(-1.0)
    with pytest.raises(NotImplementedError, match="common-cause group"):
        shared.joint_importance(broken_nodes=["a"])
    # With limited repair crews too, the groups refuse as the other
    # measures do.
    crews = RepairableRBD(
        PAIR_THEN_C,
        units,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="repair crews"):
        crews.joint_importance()


def test_a_fault_tree_s_is_its_diagram_s():
    tree = FaultTree(
        {
            "top": ("or", ["valve", "flow"]),
            "flow": ("and", ["pump 1", "pump 2"]),
        },
        {"pump 1": 0.1, "pump 2": 0.2, "valve": 0.05},
    )
    joint = tree.joint_importance()
    # Held occurring (1) or not (0), in the top event's terms.
    q = {"pump 1": 0.1, "pump 2": 0.2, "valve": 0.05}

    def top(values):
        return FaultTree(
            {
                "top": ("or", ["valve", "flow"]),
                "flow": ("and", ["pump 1", "pump 2"]),
            },
            values,
        ).top_event_probability()

    for (e, f), value in joint.items():
        held = {
            (a, b): top({**q, e: a, f: b})
            for a in (0.0, 1.0)
            for b in (0.0, 1.0)
        }
        second = held[1, 1] - held[1, 0] - held[0, 1] + held[0, 0]
        assert value == pytest.approx(-second, abs=1e-14)
    assert joint[("pump 1", "pump 2")] < 0 < joint[("pump 1", "valve")]
    rbd = tree.to_rbd()
    as_diagram = rbd.joint_importance()
    for (e, f), value in joint.items():
        found = as_diagram.get((e, f), as_diagram.get((f, e)))
        assert value == pytest.approx(found, abs=1e-14)


def test_a_fault_tree_over_time_and_with_groups():
    tree = FaultTree(
        {"top": ("and", ["x", "y", "z"])},
        {"x": W([100, 2]), "y": E([0.01]), "z": E([0.02])},
    )
    joint = tree.joint_importance([10.0, 100.0])
    assert joint[("x", "y")].shape == (2,)
    # An AND of three: -q_z for the pair x, y.
    np.testing.assert_allclose(
        joint[("x", "y")], -tree.events["z"].ff([10.0, 100.0]), rtol=1e-12
    )
    # With a group, as its diagram's, and from the top event with each
    # pair's states conditioned on: an AND of the pair, so the top event
    # given both occurring is 1, and else 0.
    grouped = FaultTree(
        {"top": ("and", ["x", "y", "z"])},
        {"x": 0.1, "y": 0.1, "z": 0.3},
        ccf_groups=[CCFGroup(["x", "y"], BetaFactor(0.1))],
    )
    joint = grouped.joint_importance()
    assert joint[("x", "y")] == pytest.approx(-0.3)
    # Given x occurred, so has y with (0.01 + 0.99 * 0.09 ** 2) / 0.0991.
    assert joint[("x", "z")] == pytest.approx(-0.018019 / 0.0991, rel=1e-9)
    as_diagram = grouped.to_rbd().joint_importance()
    for pair, value in joint.items():
        assert value == pytest.approx(as_diagram[pair], abs=1e-14)
    none = FaultTree(
        {"top": ("and", ["x", "y", "z"])},
        {"x": 0.1, "y": 0.1, "z": 0.3},
        ccf_groups=[CCFGroup(["x", "y"], BetaFactor(0.0))],
    ).joint_importance()
    plain = FaultTree(
        {"top": ("and", ["x", "y", "z"])}, {"x": 0.1, "y": 0.1, "z": 0.3}
    ).joint_importance()
    for pair, value in plain.items():
        assert none[pair] == pytest.approx(value, abs=1e-14)


def test_pairs_are_found_either_way_round():
    rbd = bridge()
    joint = rbd.joint_importance()
    assert joint[("b", "a")] == joint[("a", "b")]
    assert ("e", "d") in joint and ("d", "e") in joint
    assert joint.get(("e", "a")) == joint[("a", "e")]
    assert joint.get(("a", "z"), "none") == "none"
    with pytest.raises(KeyError):
        joint[("a", "z")]
    # The same pairs, however the nodes were given.
    backwards = NonRepairableRBD(
        BRIDGE, {n: F(Q[n]) for n in reversed(list(Q))}
    )
    assert backwards.joint_importance() == pytest.approx(dict(joint))
    assert set(backwards.joint_importance()) == set(joint)
