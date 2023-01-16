"""
Tests structure cases for the RBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest
from repyability.rbd.rbd import RBD


# Check components are correct lengths
def test_input_errors():
    two_inputs_edges = [
        (0, 1),
        (2, 1),
        (1, 3),
        (3, 4)
    ]
    with pytest.raises(ValueError):
        rbd = RBD(two_inputs_edges)

def test_output_errors():
    two_outputs_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4)
    ]
    with pytest.raises(ValueError):
        rbd = RBD(two_outputs_edges)

def test_cycle_errors():
    edges_with_cycle = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 1),
        (3, 4),
    ]
    with pytest.raises(ValueError):
        rbd = RBD(edges_with_cycle)

def test_series_edges():
    series_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    # Test no exception raised
    rbd = RBD(series_edges)

def test_parallel_edges():
    parallel_edges = [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (2, 4),
    ]
    # Test no exception raised
    rbd = RBD(parallel_edges)

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
        (4, 5)
    ]
    # Test no exception raised
    rbd = RBD(bridge_structure)


