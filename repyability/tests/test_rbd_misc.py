"""
Tests miscellaneous cases for the NonRepairableRBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import numpy as np
import pytest
import surpyval as surv
from surpyval import Exponential, FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD


def test_rbd_mean_time_to_failure_series():
    # Two exponential components in series fail at the first failure, so the
    # system MTTF is 1 / (lambda_1 + lambda_2). Exercises the Monte-Carlo
    # random()/mean() path.
    rbd = NonRepairableRBD(
        [("input", "a"), ("a", "b"), ("b", "output")],
        {
            "a": Exponential.from_params([0.01]),
            "b": Exponential.from_params([0.02]),
        },
    )
    np.random.seed(0)
    assert rbd.mean(mc_samples=20_000) == pytest.approx(1 / 0.03, rel=5e-2)


# Check components are correct lengths
def test_rbd_components(rbd1: NonRepairableRBD, rbd2: NonRepairableRBD):
    assert len(rbd1.nodes) == 3
    assert len(rbd2.nodes) == 6


# Test get_all_path_sets()
def test_rbd_get_all_path_sets(rbd1: NonRepairableRBD, rbd2: NonRepairableRBD):
    assert list(rbd1.get_all_path_sets()) == [
        ["source", "pump1", "valve", "sink"],
        ["source", "pump2", "valve", "sink"],
    ]
    assert list(rbd2.get_all_path_sets()) == [
        [1, 2, 3, 5, 6, 7, 8],
        [1, 2, 4, 7, 8],
    ]


# Check ValueError's


def test_rbd_node_not_in_edge_list():
    with pytest.raises(ValueError):
        edges = [(1, 3)]
        reliabilities = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, reliabilities)


def test_rbd_component_not_in_reliability_dict():
    with pytest.raises(ValueError):
        reliabilities = {2: FixedEventProbability.from_params(1 - 0.8)}
        edges = [(1, 2), (2, 3), (3, 4)]
        NonRepairableRBD(edges, reliabilities)


def test_rbd_node_with_no_output():
    with pytest.raises(ValueError):
        edges = [(1, 2), (2, 3), (2, 4)]
        reliabilities = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, reliabilities)


def test_rbd_node_with_no_input():
    with pytest.raises(ValueError):
        edges = [(1, 2), (2, 3), (4, 2)]
        reliabilities = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, reliabilities)


def test_rbd_circular_dependency():
    with pytest.raises(ValueError):
        edges = [("s", 2), (2, 3), (3, 4), (4, 2), (4, "t")]
        reliabilities = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, reliabilities)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"input_node": "a"}, "incoming edges"),
        ({"output_node": "b"}, "outgoing edges"),
    ],
)
def test_named_input_and_output_must_be_the_source_and_sink(kwargs, message):
    # Naming an inner node used to be accepted, silently analysing a
    # different system (the nodes beyond it dropped out).
    unit = surv.Weibull.from_params([100, 2])
    edges = [("s", "a"), ("a", "b"), ("b", "t")]
    models = {"s": unit, "a": unit, "b": unit, "t": unit}
    with pytest.raises(ValueError, match=message):
        NonRepairableRBD(edges, models, **kwargs)
    unit_spec = {
        "reliability": surv.Exponential.from_params([0.1]),
        "repairability": surv.Exponential.from_params([1.0]),
    }
    with pytest.raises(ValueError, match=message):
        RepairableRBD(edges, {n: unit_spec for n in "sabt"}, **kwargs)
    named = NonRepairableRBD(
        edges, {"a": unit, "b": unit}, input_node="s", output_node="t"
    )
    assert named.sf(50) == pytest.approx(unit.sf(50).item() ** 2)


@pytest.mark.parametrize("cls", ["RBD", "NonRepairableRBD", "RepairableRBD"])
def test_an_invalid_on_infeasible_rbd_is_rejected_on_a_valid_diagram(cls):
    # The base RBD (and so RepairableRBD) checked the value only when the
    # diagram was invalid, so a typo passed unnoticed on a valid one.
    from repyability import RBD

    edges = [("s", "a"), ("a", "t")]
    unit = surv.Exponential.from_params([0.1])
    build = {
        "RBD": lambda: RBD(edges, on_infeasible_rbd="warm"),
        "NonRepairableRBD": lambda: NonRepairableRBD(
            edges, {"a": unit}, on_infeasible_rbd="warm"
        ),
        "RepairableRBD": lambda: RepairableRBD(
            edges,
            {"a": {"reliability": unit, "repairability": unit}},
            on_infeasible_rbd="warm",
        ),
    }[cls]
    with pytest.raises(ValueError, match="must be one of"):
        build()


def test_structure_warning():
    with pytest.warns(UserWarning, match="^Structural Errors in RBD"):
        NonRepairableRBD(
            [("s", "a"), ("a", "t"), ("b", "t")],
            {
                "a": Exponential.from_params([0.1]),
                "b": Exponential.from_params([0.1]),
            },
            on_infeasible_rbd="warn",
        )


def test_system_probability_rejects_an_unknown_method():
    # Anything but "p" used to be taken silently as cut sets.
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {n: Exponential.from_params([0.1]) for n in "ab"},
    )
    probabilities = {"a": 0.9, "b": 0.8}
    by_paths = rbd.system_probability(probabilities, method="p")
    by_cuts = rbd.system_probability(probabilities, method="c")
    assert by_paths == pytest.approx([0.98])
    assert by_cuts == pytest.approx([0.98])
    with pytest.raises(ValueError, match="'p' or 'c'"):
        rbd.system_probability(probabilities, method="cut sets")
    with pytest.raises(ValueError, match="'p' or 'c'"):
        rbd.sf(1.0, method="x")
