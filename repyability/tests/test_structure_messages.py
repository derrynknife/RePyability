"""What an infeasible diagram says is wrong with it (#131): the
``ValueError`` (or, with ``on_infeasible_rbd="warn"``, the warning) lists
each of ``structure_check``'s findings on a line of its own, names a model
whose key is in no edge (with the node it was likely meant for), and does
not report the input and output nodes as missing models; and
``on_infeasible_rbd="ignore"`` builds a diagram with k-out-of-n errors, for
``structure_check`` to be read.
"""

import warnings

import pytest
import surpyval as surv

from repyability import (
    RBD,
    NonRepairableRBD,
    PerfectReliability,
    RepairableRBD,
)

W = surv.Weibull.from_params([100.0, 2.0])
PAIR = [("s", "a"), ("s", "b"), ("a", "v"), ("b", "v"), ("v", "t")]


def problems(build) -> list:
    """The finding lines of the error ``build()`` raises."""
    with pytest.raises(ValueError) as error:
        build()
    lines = str(error.value).splitlines()
    assert lines[0] == "RBD not correctly structured:"
    return [line.removeprefix("  - ") for line in lines[1:]]


@pytest.mark.parametrize(
    "build, expected",
    [
        (
            lambda: NonRepairableRBD(
                [("s", "a"), ("a", "b"), ("b", "t")], {"a": W}
            ),
            ["node 'b' (in the edges) has no model"],
        ),
        (
            lambda: NonRepairableRBD(
                [("s", "pump"), ("pump", "t")], {"pmup": W}
            ),
            [
                "node 'pump' (in the edges) has no model",
                "model 'pmup' is not a node in the edges; did you mean "
                "'pump'?",
            ],
        ),
        (
            lambda: NonRepairableRBD(
                [("s", "a"), ("a", "t")], {"a": W, "b": W}
            ),
            ["model 'b' is not a node in the edges"],
        ),
        (
            lambda: NonRepairableRBD(
                [("s", "a"), ("a", "b"), ("b", "a"), ("b", "t")],
                {"a": W, "b": W},
            ),
            ["there is a cycle through 'a', 'b'"],
        ),
        (
            lambda: NonRepairableRBD(
                [("s", "a"), ("a", "t"), ("s", "b")], {"a": W, "b": W}
            ),
            [
                "more than one node has no outgoing edges, so the output "
                "node is not clear: 'b', 't' (every node but the output "
                "needs an edge onward)"
            ],
        ),
        (
            lambda: NonRepairableRBD(
                [("s", "a"), ("a", "t"), ("b", "t")], {"a": W, "b": W}
            ),
            [
                "more than one node has no incoming edges, so the input "
                "node is not clear: 'b', 's' (only the input node has none)"
            ],
        ),
        (
            lambda: NonRepairableRBD(
                PAIR,
                {"a": W, "b": W, "v": PerfectReliability},
                k={"v": 3},
            ),
            ["node 'v' needs 3 of its inputs working (k = 3) but has only 2"],
        ),
        (
            lambda: NonRepairableRBD(
                PAIR,
                {"a": W, "b": W, "v": PerfectReliability},
                k={"v": 0},
            ),
            [
                "node 'v' has k = 0: k is how many of its inputs must work, "
                "a whole number from 1"
            ],
        ),
        (
            lambda: NonRepairableRBD(
                [("s", "a"), ("a", "t")], {"a": W}, k={"z": 2}
            ),
            ["k is given for 'z', which is not a node of the diagram"],
        ),
    ],
    ids=[
        "no model",
        "typo",
        "unused model",
        "cycle",
        "dangling branch",
        "second source",
        "k above the inputs",
        "k of zero",
        "k for no node",
    ],
)
def test_each_finding_has_a_line(build, expected):
    assert problems(build) == expected


def test_a_repairable_diagram_says_the_same():
    unit = {"reliability": W, "repairability": W}
    assert problems(
        lambda: RepairableRBD([("s", "pump"), ("pump", "t")], {"pmup": unit})
    ) == [
        "node 'pump' (in the edges) has no model",
        "model 'pmup' is not a node in the edges; did you mean 'pump'?",
    ]
    assert problems(
        lambda: RepairableRBD(
            [("s", "a"), ("a", "b"), ("b", "t")], {"a": unit}
        )
    ) == ["node 'b' (in the edges) has no model"]


def test_the_findings_stay_in_the_structure_check():
    rbd = NonRepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {"pmup": W},
        on_infeasible_rbd="ignore",
    )
    check = rbd.structure_check
    assert check["is_valid"] is False
    assert check["nodes_in_no_edge"] == ["pmup"]
    assert check["nodes_with_no_model"] == ["pump"]
    assert check["nodes_with_no_reliability_distribution"] == ["pump"]
    # The input and output are found, and need no models.
    assert (rbd.input_node, rbd.output_node) == ("s", "t")
    # A model in no edge is no part of the diagram.
    assert "pmup" not in rbd.G
    assert set(rbd.reliabilities) == {"s", "t"}


def test_warn_says_the_same_and_builds():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rbd = NonRepairableRBD(
            [("s", "a"), ("a", "t")],
            {"a": W, "b": W},
            on_infeasible_rbd="warn",
        )
    messages = [str(w.message) for w in caught]
    assert any(
        m.startswith("RBD not correctly structured:\n")
        and "model 'b' is not a node in the edges" in m
        for m in messages
    )
    # Otherwise sound: it works without the unused model.
    assert rbd.sf(50.0) == pytest.approx(W.sf(50.0))


def test_ignore_reaches_the_structure_check_with_k_errors():
    rbd = NonRepairableRBD(
        PAIR,
        {"a": W, "b": W, "v": PerfectReliability},
        k={"v": 3},
        on_infeasible_rbd="ignore",
    )
    assert rbd.structure_check["is_valid"] is False
    assert rbd.structure_check["koon_errors"] == [
        "node 'v' needs 3 of its inputs working (k = 3) but has only 2"
    ]


def test_a_structure_alone_takes_any_nodes():
    # The base class with no models: nothing is missing.
    structure = RBD([("s", "a"), ("a", "t")])
    assert structure.structure_check["is_valid"] is True
    assert "nodes_with_no_model" not in structure.structure_check
