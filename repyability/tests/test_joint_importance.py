"""The joint (second-order) importance of pairs of nodes (#194): whether
improving two together is worth more than improving each.

Checked against the system worked out with each pair held working and
failed, against its meaning (the system's change when both improve, less
the changes when each does, is the joint importance times the two
improvements, exactly, as the structure is multilinear), and across the
classes: a fault tree's equals its diagram's."""

import itertools

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import (
    BetaFactor,
    CCFGroup,
    FaultTree,
    NonRepairableRBD,
    RepairableRBD,
)

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


def test_common_cause_groups_are_refused():
    grouped = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": F(0.1), "b": F(0.1)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        grouped.joint_importance()


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
    grouped = RepairableRBD(
        PAIR_THEN_C,
        {n: unit(0.1) for n in "abc"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        grouped.joint_importance()


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
    grouped = FaultTree(
        {"top": ("and", ["x", "y"])},
        {"x": 0.1, "y": 0.1},
        ccf_groups=[CCFGroup(["x", "y"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        grouped.joint_importance()


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
