"""
Tests many of the RBD class methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

from repyability.rbd.rbd import RBD


def test_rbd_get_min_path_sets_rbd_series(rbd_series: RBD):
    assert {frozenset([1, 2, 3, 4, 5])} == rbd_series.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_parallel(rbd_parallel: RBD):
    assert {
        frozenset([1, 2, 5]),
        frozenset([1, 3, 5]),
        frozenset([1, 4, 5]),
    } == rbd_parallel.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1(rbd1: RBD):
    assert {
        frozenset(["source", "pump1", "valve", "sink"]),
        frozenset(["source", "pump2", "valve", "sink"]),
    } == rbd1.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd1_exclude_in_out_node(rbd1: RBD):
    assert {
        frozenset(["pump1", "valve"]),
        frozenset(["pump2", "valve"]),
    } == rbd1.get_min_path_sets(include_in_out_nodes=False)


def test_rbd_get_min_path_sets_rbd2(rbd2: RBD):
    assert {
        frozenset([1, 2, 3, 5, 6, 7, 8]),
        frozenset([1, 2, 4, 7, 8]),
    } == rbd2.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd3(rbd3: RBD):
    assert {
        frozenset([0, 1, 2, 6]),
        frozenset([0, 1, 5, 4, 6]),
        frozenset([0, 3, 4, 6]),
        frozenset([0, 3, 5, 2, 6]),
    } == rbd3.get_min_path_sets()


def test_rbd_get_min_path_sets_rbd_double_parallel(
    rbd_double_parallel_args: dict,
):
    rbd = RBD(**rbd_double_parallel_args)
    assert {
        frozenset(["s", 1, 3, 4, "t"]),
        frozenset(["s", 2, 3, 4, "t"]),
        frozenset(["s", 1, 3, 5, "t"]),
        frozenset(["s", 2, 3, 5, "t"]),
    } == rbd.get_min_path_sets()
