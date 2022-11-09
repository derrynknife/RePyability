"""
Tests miscellaneous cases for the RBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

from repyability.rbd.rbd import RBD


# Check components are correct lengths
def test_rbd_components(rbd1: RBD, rbd2: RBD):
    assert len(rbd1.components) == 5
    assert len(rbd2.components) == 8


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
