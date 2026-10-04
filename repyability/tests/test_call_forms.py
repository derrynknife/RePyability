"""The forms a call may take (#224, #225).

- A repairable diagram's measures take their times first, as a
  non-repairable diagram's do: ``birnbaum_importance(5.0)`` is
  ``birnbaum_importance(x=5.0)``, where the time was read as node names.
- A bare string given where node names go is one node's name, not its
  characters, on every method that takes node names (and on
  ``FaultTree.occurs``); the nodes refused are named in order.
"""

import inspect

import numpy as np
import pytest
import surpyval as surv

from repyability import FaultTree, NonRepairableRBD, RepairableRBD
from repyability.rbd.rbd import RBD
from repyability.utils.wrappers import NODE_ARGUMENTS

E, W = surv.Exponential.from_params, surv.Weibull.from_params

#: The measures that take their times (``x``) first.
TIMES_FIRST = [
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "fussell_vesely",
    "joint_importance",
    "differential_importance",
    "parameter_sensitivity",
]


def same(a, b):
    """Whether two results hold the same values."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.fixture(scope="module")
def repairable():
    unit = {"reliability": W([100.0, 1.5]), "repairability": E([0.5])}
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {n: dict(unit) for n in "abc"},
    )


# -- #224: times first ------------------------------------------------------


@pytest.mark.parametrize("method", TIMES_FIRST)
@pytest.mark.parametrize("x", [5.0, 40, [5.0, 40.0], np.array([5.0, 40.0])])
def test_times_given_first_are_x(repairable, method, x):
    measure = getattr(repairable, method)
    assert same(measure(x), measure(x=x))


def test_a_window_given_first_is_the_window(repairable):
    assert same(
        repairable.barlow_proschan_importance(100.0),
        repairable.barlow_proschan_importance(window=100.0),
    )


def test_times_given_twice_are_refused(repairable):
    with pytest.raises(TypeError, match="give them once, as x="):
        repairable.birnbaum_importance(5.0, x=5.0)


def test_node_names_first_are_still_node_names(repairable):
    assert same(
        repairable.birnbaum_importance(["a"]),
        repairable.birnbaum_importance(working_nodes=["a"]),
    )


def test_numbers_that_are_nodes_are_nodes():
    unit = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(
        [(0, 1), (0, 2), (1, 3), (2, 3)], {1: dict(unit), 2: dict(unit)}
    )
    held = rbd.birnbaum_importance([1])
    assert same(held, rbd.birnbaum_importance(working_nodes=[1]))
    assert not same(held, rbd.birnbaum_importance())


# -- #225: a node's name given alone -----------------------------------------


@pytest.fixture(scope="module")
def nonrepairable():
    return NonRepairableRBD(
        [("s", "belt"), ("belt", "seal1"), ("seal1", "t")],
        {"belt": W([1000.0, 2.0]), "seal1": E([1e-3])},
    )


def test_a_string_is_one_node(nonrepairable, repairable):
    assert nonrepairable.sf(100.0, working_nodes="belt") == pytest.approx(
        nonrepairable.sf(100.0, working_nodes=["belt"])
    )
    assert nonrepairable.sf(100.0, broken_nodes="seal1") == 0.0
    assert repairable.mean_availability(working_nodes="a") == pytest.approx(
        repairable.mean_availability(working_nodes=["a"])
    )
    assert same(
        repairable.birnbaum_importance("a"),
        repairable.birnbaum_importance(["a"]),
    )


def test_one_letter_names_are_not_taken_from_a_word(repairable):
    # "ab" names no node: its letters were each taken as one.
    with pytest.raises(ValueError, match="Unknown node 'ab'"):
        repairable.mean_availability(working_nodes="ab")


def test_unknown_nodes_are_named_in_order(nonrepairable):
    with pytest.raises(ValueError) as caught:
        nonrepairable.sf(1.0, working_nodes=["zz", "aa", "mm"])
    assert "Unknown nodes ['aa', 'mm', 'zz']" in str(caught.value)


def test_the_same_refusal_every_run(nonrepairable):
    messages = set()
    for _ in range(5):
        with pytest.raises(ValueError) as caught:
            nonrepairable.sf(1.0, working_nodes={"q", "w", "e", "r"})
        messages.add(str(caught.value))
    assert len(messages) == 1


@pytest.mark.parametrize("cls", [RBD, NonRepairableRBD, RepairableRBD])
def test_every_method_taking_node_names_takes_one(cls):
    missing = [
        name
        for name in dir(cls)
        if not name.startswith("_")
        and inspect.isfunction(func := inspect.getattr_static(cls, name))
        and any(
            p in inspect.signature(func).parameters for p in NODE_ARGUMENTS
        )
        and not getattr(func, "node_names", False)
    ]
    assert missing == []


def test_a_fault_tree_event_given_alone():
    tree = FaultTree({"top": ("and", ["ab", "c"])}, {"ab": 0.1, "c": 0.2})
    assert not tree.occurs("ab")  # "ab" alone, not "a" and "b"
    assert tree.occurs(["ab", "c"])
    with pytest.raises(ValueError, match="'a'"):
        tree.occurs("a")
