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


# KooN Tests


def test_rbd_get_min_cut_sets_rbd_series_koon(rbd_series_koon: RBD):
    # It's a series RBD with the middle node k=2 so the system doesn't work
    # at all
    assert {} == rbd_series_koon.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd1_koon(rbd1_koon: RBD):
    assert {
        frozenset(["pump1"]),
        frozenset(["pump2"]),
        frozenset(["valve"]),
    } == rbd1_koon.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd2_koon(rbd2_koon: RBD):
    assert {
        frozenset([2]),
        frozenset([3]),
        frozenset([4]),
        frozenset([5]),
        frozenset([6]),
        frozenset([7]),
    } == rbd2_koon.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd3_koon1(rbd3_koon1: RBD):
    assert {
        frozenset([2, 4]),
        frozenset([1, 4]),
        frozenset([2, 3]),
    } == rbd3_koon1.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd3_koon2(rbd3_koon2: RBD):
    assert {
        frozenset([1, 3]),
        frozenset([2, 4]),
        frozenset([1, 4]),
        frozenset([5, 4]),
        frozenset([1, 3]),
        frozenset([5, 3]),
        frozenset([2, 3]),
    } == rbd3_koon2.get_min_cut_sets()


def test_rbd_get_min_cut_sets_rbd3_koon3(rbd3_koon3: RBD):
    assert {
        frozenset([2, 4]),
        frozenset([2, 1]),
        frozenset([3]),
        frozenset([1, 4]),
        frozenset([4, 5]),
        frozenset([4, 1]),
    } == rbd3_koon3.get_min_cut_sets()
