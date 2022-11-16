"""
Tests RBD's get_min_cut_sets() method.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

from repyability.rbd.rbd import RBD


def test_rbd_get_min_cut_sets_rbd_series(rbd_series: RBD):
    assert {
        frozenset([2]),
        frozenset([3]),
        frozenset([4]),
    } == rbd_series.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd_parallel(rbd_parallel: RBD):
    assert {frozenset([2, 3, 4])} == rbd_parallel.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd1(rbd1: RBD):
    assert {
        frozenset(["pump1", "pump2"]),
        frozenset(["valve"]),
    } == rbd1.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd2(rbd2: RBD):
    assert {
        frozenset([2]),
        frozenset([3, 4]),
        frozenset([4, 5]),
        frozenset([4, 6]),
        frozenset([7]),
    } == rbd2.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd3(rbd3: RBD):
    assert {
        frozenset([1, 3]),
        frozenset([2, 4]),
        frozenset([1, 4, 5]),
        frozenset([2, 3, 5]),
    } == rbd3.get_min_cut_sets()
