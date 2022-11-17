"""
Tests miscellaneous cases for the RBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest

from repyability.rbd.rbd import RBD
from repyability.tests.fixed_probability import FixedProbabilityFitter


# Check components are correct lengths
def test_rbd_components(rbd1: RBD, rbd2: RBD):
    assert len(rbd1.reliability) == 5
    assert len(rbd2.reliability) == 8


# Test get_all_path_sets()
def test_rbd_get_all_path_sets(rbd1: RBD, rbd2: RBD):
    assert list(rbd1.get_all_path_sets()) == [
        ["source", "pump1", "valve", "sink"],
        ["source", "pump2", "valve", "sink"],
    ]
    assert list(rbd2.get_all_path_sets()) == [
        [1, 2, 3, 5, 6, 7, 8],
        [1, 2, 4, 7, 8],
    ]


def test_rbd_node_type_dict(rbd1: RBD):
    assert rbd1.G.nodes["source"]["type"] == "input_node"
    assert rbd1.G.nodes["pump1"]["type"] == "node"
    assert rbd1.G.nodes["pump2"]["type"] == "node"
    assert rbd1.G.nodes["sink"]["type"] == "output_node"


# Check ValueError's


def test_rbd_node_not_in_edge_list():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 2: 2, 3: "output_node"}
        edges = [(1, 3)]
        components = {2: FixedProbabilityFitter.from_params(0.8)}
        RBD(nodes, components, edges)


def test_rbd_component_not_in_reliability_dict():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 2: 2, 3: "output_node"}
        edges = [(1, 2), (2, 3)]
        components = {}
        RBD(nodes, components, edges)


def test_rbd_node_not_in_node_list():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 3: "output_node"}
        edges = [(1, 2), (2, 3)]
        components = {2: FixedProbabilityFitter.from_params(0.8)}
        RBD(nodes, components, edges)


def test_rbd_node_with_no_output():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 2: 2, 4: 2, 3: "output_node"}
        edges = [(1, 2), (2, 3), (2, 4)]
        components = {2: FixedProbabilityFitter.from_params(0.8)}
        RBD(nodes, components, edges)


def test_rbd_node_with_no_input():
    with pytest.raises(ValueError):
        nodes = {1: "input_node", 2: 2, 4: 2, 3: "output_node"}
        edges = [(1, 2), (2, 3), (4, 2)]
        components = {2: FixedProbabilityFitter.from_params(0.8)}
        RBD(nodes, components, edges)


def test_rbd_circular_dependency():
    with pytest.raises(ValueError):
        nodes = {"s": "input_node", 2: 2, 3: 2, 4: 2, "t": "output_node"}
        edges = [("s", 2), (2, 3), (3, 4), (4, 2), (4, "t")]
        components = {2: FixedProbabilityFitter.from_params(0.8)}
        RBD(nodes, components, edges)
