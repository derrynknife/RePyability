"""
Tests Berge's algorithm for the minimal cut sets
(minimal_cut_sets_from_path_sets / RBD.get_min_cut_sets) and the new default
path-set method for sf().

Berge's output is cross-checked against the previous implementation (full
Cartesian product of the path sets, then minimalised) across every fixture,
including the k-out-of-n ones, plus a few hand-computed examples.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import itertools

import pytest

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.rbd import minimal_cut_sets_from_path_sets


def _reference_min_cut_sets(path_sets):
    """The previous implementation: full Cartesian product then minimalise.

    Kept here purely as an independent reference to verify Berge against.
    """
    path_sets = [set(p) for p in path_sets]
    prods = itertools.product(*path_sets)
    cut_sets = [frozenset(prod) for prod in prods if prod]
    minimal: list = []
    for cut_set in cut_sets:
        is_minimal = True
        for other in minimal.copy():
            if cut_set.issuperset(other):
                is_minimal = False
                break
            if cut_set.issubset(other):
                minimal.remove(other)
        if is_minimal:
            minimal.append(cut_set)
    return set(minimal)


# --- Hand-computed examples ------------------------------------------------


def test_berge_series():
    assert minimal_cut_sets_from_path_sets([{2, 3, 4}]) == {
        frozenset([2]),
        frozenset([3]),
        frozenset([4]),
    }


def test_berge_parallel():
    assert minimal_cut_sets_from_path_sets([{2}, {3}, {4}]) == {
        frozenset([2, 3, 4])
    }


def test_berge_bridge_example():
    # Path sets of a small bridge: cut sets are the minimal hitting sets.
    path_sets = [frozenset({1, 2}), frozenset({3, 4}), frozenset({1, 4})]
    assert minimal_cut_sets_from_path_sets(path_sets) == {
        frozenset({1, 3}),
        frozenset({1, 4}),
        frozenset({2, 4}),
    }


# --- Cross-check Berge against the reference across all fixtures ------------


FIXTURE_NAMES = [
    "rbd_series",
    "rbd_parallel",
    "rbd1",
    "rbd2",
    "rbd3",
    "rbd_repeated_component_parallel",
    "rbd1_koon",
    "rbd2_koon",
    "rbd3_koon1",
    "rbd3_koon2",
    "rbd3_koon3",
    "rbd_koon_simple",
]


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_berge_matches_reference(fixture_name, request):
    rbd = request.getfixturevalue(fixture_name)
    path_sets = rbd.get_min_path_sets(include_in_out_nodes=False)
    expected = _reference_min_cut_sets(path_sets)
    assert minimal_cut_sets_from_path_sets(path_sets) == expected
    # And the method itself returns the same.
    assert rbd.get_min_cut_sets() == expected


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_cut_sets_are_minimal_transversals(fixture_name, request):
    rbd = request.getfixturevalue(fixture_name)
    path_sets = [
        set(p) for p in rbd.get_min_path_sets(include_in_out_nodes=False)
    ]
    cut_sets = rbd.get_min_cut_sets()

    # Every cut set must hit every path set (it is a transversal).
    for cut_set in cut_sets:
        for path_set in path_sets:
            assert cut_set & path_set

    # No cut set is a proper subset of another (they are all minimal).
    for a in cut_sets:
        for b in cut_sets:
            if a != b:
                assert not a < b


# --- New default: path-set method ------------------------------------------


def test_sf_default_method_is_path_set(rbd3: NonRepairableRBD):
    # The default (method=None) must match the explicit path-set method, and
    # both methods agree.
    assert rbd3.sf(1) == pytest.approx(rbd3.sf(1, method="p"))
    assert rbd3.sf(1) == pytest.approx(rbd3.sf(1, method="c"))


def test_sf_default_with_approx_does_not_raise(rbd1: NonRepairableRBD):
    # approx=True with the default method must route to cut sets, not raise.
    value = rbd1.sf(1, approx=True)
    assert value == pytest.approx(rbd1.sf(1, method="c"))


def test_sf_explicit_path_with_approx_raises(rbd1: NonRepairableRBD):
    with pytest.raises(ValueError):
        rbd1.sf(1, method="p", approx=True)
