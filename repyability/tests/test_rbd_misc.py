"""
Tests miscellaneous cases for the NonRepairableRBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import numpy as np
import pytest
from surpyval import Exponential, FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD


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
