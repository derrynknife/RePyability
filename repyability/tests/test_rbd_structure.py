"""
Tests structure cases for the RBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest

from repyability.rbd.rbd import RBD


# Check components are correct lengths
def test_input_errors():
    two_inputs_edges = [(0, 1), (2, 1), (1, 3), (3, 4)]

    # Test warning raised when asked
    with pytest.warns():
        RBD(two_inputs_edges, on_infeasible_rbd="warn")

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        RBD(two_inputs_edges)

    # Test ignore error when asked
    RBD(two_inputs_edges, on_infeasible_rbd="ignore")


def test_output_errors():
    two_outputs_edges = [(0, 1), (1, 2), (2, 3), (2, 4)]

    # Test warning raised when asked
    with pytest.warns():
        RBD(two_outputs_edges, on_infeasible_rbd="warn")

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        RBD(two_outputs_edges)

    # Test ignore error when asked
    RBD(two_outputs_edges, on_infeasible_rbd="ignore")


def test_cycle_errors():
    edges_with_cycle = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 1),
        (3, 4),
    ]

    # Test warning raised when asked
    with pytest.warns():
        RBD(edges_with_cycle, on_infeasible_rbd="warn")

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        RBD(edges_with_cycle)

    # Test ignore error when asked
    RBD(edges_with_cycle, on_infeasible_rbd="ignore")


def test_series_edges():
    series_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    # Test no exception raised
    RBD(series_edges)


def test_parallel_edges():
    parallel_edges = [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (2, 4),
    ]
    # Test no exception raised
    RBD(parallel_edges)


def test_bridge_edges():
    bridge_structure = [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 6),
        (2, 3),
        (2, 5),
        (5, 6),
        (0, 4),
        (4, 2),
        (4, 5),
    ]
    # Test no exception raised
    RBD(bridge_structure)


def test_irrelevant_components_simple():
    rbd = RBD(
        edges=[(1, 2), (2, 3), (1, 2), (2, 4), (4, 3)],
    )
    assert rbd.find_irrelevant_components() == {4}


def test_irrelevant_components_series():
    rbd = RBD(
        edges=[(1, 2), (2, 3), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 3)],
    )
    assert rbd.find_irrelevant_components() == {4, 5, 6, 7}


def test_irrelevant_components_parallel():
    rbd = RBD(
        edges=[(1, 2), (2, 3), (1, 2), (2, 4), (4, 3), (1, 2), (2, 5), (5, 3)],
    )
    assert rbd.find_irrelevant_components() == {4, 5}


def test_irrelevant_components_multiple_path_sets():
    rbd = RBD(
        edges=[
            (1, 2),
            (2, 3),
            (1, 2),
            (2, 4),
            (4, 3),
            (1, 2),
            (2, 5),
            (5, 3),
            (1, 6),
            (6, 3),
            (1, 7),
            (7, 3),
            (1, 8),
            (8, 3),
        ],
    )
    assert rbd.find_irrelevant_components() == {4, 5}
