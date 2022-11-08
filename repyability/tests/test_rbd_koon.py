"""
Tests many of the RBD class methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

from repyability.rbd.rbd import RBD


def test_rbd_koon_default_k(rbd_series: RBD):
    assert rbd_series.G.nodes[1]["k"] == 1
    assert rbd_series.G.nodes[2]["k"] == 1
    assert rbd_series.G.nodes[3]["k"] == 1
    assert rbd_series.G.nodes[4]["k"] == 1
    assert rbd_series.G.nodes[5]["k"] == 1


def test_rbd_koon_k_given(rbd1_koon: RBD):
    assert rbd1_koon.G.nodes["source"]["k"] == 1
    assert rbd1_koon.G.nodes["pump1"]["k"] == 1
    assert rbd1_koon.G.nodes["pump2"]["k"] == 1
    assert rbd1_koon.G.nodes["valve"]["k"] == 2
    assert rbd1_koon.G.nodes["sink"]["k"] == 1


def test_rbd_get_min_path_sets_rbd_series_koon(rbd_series_koon: RBD):
    assert {} == rbd_series_koon.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1_koon(rbd1_koon: RBD):
    assert {
        frozenset(["source", "pump1", "pump2", "valve", "sink"])
    } == rbd1_koon.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1_exclude_in_out_node_koon(rbd1_koon: RBD):
    assert {
        frozenset(["pump1", "pump2", "valve"])
    } == rbd1_koon.get_min_path_sets(include_in_out_nodes=False)


def test_rbd_get_min_path_sets_rbd2_koon(rbd2_koon: RBD):
    assert {
        frozenset(frozenset([1, 2, 3, 4, 5, 6, 7, 8])),
    } == rbd2_koon.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3_koon1(rbd3_koon1: RBD):
    assert {
        frozenset([0, 1, 2, 6]),
        frozenset([0, 1, 4, 6]),
    } == rbd3_koon1.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3_koon2(rbd3_koon2: RBD):
    assert {frozenset([0, 1, 2, 5, 6])} == rbd3_koon2.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3_koon3(rbd3_koon3: RBD):
    assert {frozenset([0, 1, 2, 3, 5, 6])} == rbd3_koon3.get_min_path_sets()
