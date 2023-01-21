"""
Tests miscellaneous cases for the NonRepairableRBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest
from surpyval import FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD


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
        nodes = {1: "input_node", 2: 2, 3: "output_node"}
        edges = [(1, 3)]
        components = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, nodes, components)


def test_rbd_component_not_in_reliability_dict():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 2: 2, 3: "output_node"}
        edges = [(1, 2), (2, 3)]
        NonRepairableRBD(edges, nodes)


def test_rbd_node_not_in_node_list():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 3: "output_node"}
        edges = [(1, 2), (2, 3)]
        components = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, nodes, components)


def test_rbd_node_with_no_output():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 2: 2, 4: 2, 3: "output_node"}
        edges = [(1, 2), (2, 3), (2, 4)]
        components = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, nodes, components)


def test_rbd_node_with_no_input():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 2: 2, 4: 2, 3: "output_node"}
        edges = [(1, 2), (2, 3), (4, 2)]
        components = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, nodes, components)


def test_rbd_circular_dependency():
    with pytest.raises(ValueError):
        nodes = {"s": "input_node", 2: 2, 3: 2, 4: 2, "t": "output_node"}
        edges = [("s", 2), (2, 3), (3, 4), (4, 2), (4, "t")]
        components = {2: FixedEventProbability.from_params(1 - 0.8)}
        NonRepairableRBD(edges, nodes, components)
