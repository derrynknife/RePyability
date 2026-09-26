"""
Tests allocation cases for the RBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import pytest
from numpy.random import rand

from repyability.rbd.rbd import RBD


# Test that allocation to a series system works.
def test_series_allocation():
    two_inputs_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
    ]
    rbd = RBD(two_inputs_edges)
    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        achieved = rbd.system_probability(allocated_probs)
        assert pytest.approx(achieved, rel=1e-3) == target

    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        assert pytest.approx(allocated_probs[1], rel=1e-3) == target ** (1 / 6)


# Test that allocation to a parallel system works.
def test_parallel_allocation():
    two_inputs_edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    rbd = RBD(two_inputs_edges)
    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        achieved = rbd.system_probability(allocated_probs)
        assert pytest.approx(achieved, rel=1e-3) == target

    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        assert pytest.approx(allocated_probs[1], rel=1e-3) == 1 - (
            1 - target
        ) ** (1 / 6)


def _series():
    return RBD([("s", 1), (1, 2), (2, 3), (3, "t")])


def test_improvement_allocation_below_the_current_reliability_stays_valid():
    # The current system probability is 0.9 * 0.8 * 0.7 = 0.504. A lower
    # target lowers the node probabilities, each capped at a failure
    # probability of 1; the old solver returned negative probabilities.
    rbd = _series()
    new = rbd.improvement_allocation(0.3, {1: 0.9, 2: 0.8, 3: 0.7})
    assert all(0.0 <= p <= 1.0 for p in new.values())
    assert rbd.system_probability(new).item() == pytest.approx(0.3)
    factors = {(1 - new[n]) / (1 - p) for n, p in {1: 0.9, 2: 0.8}.items()}
    assert max(factors) == pytest.approx(min(factors))


def test_improvement_allocation_raises_for_an_unreachable_target():
    # With nodes 1 and 2 fixed the system cannot beat 0.9 * 0.8 = 0.72.
    with pytest.raises(ValueError, match="cannot be reached.*0.72"):
        _series().improvement_allocation(
            0.9, {1: 0.9, 2: 0.8, 3: 0.7}, fixed=[1, 2]
        )


def test_allocation_to_a_target_of_one_makes_the_nodes_perfect():
    rbd = _series()
    assert rbd.equal_allocation(1.0) == {1: 1.0, 2: 1.0, 3: 1.0}
    assert rbd.res.x[0] == float("inf")
    assert rbd.equal_allocation(0.0) == {1: 0.0, 2: 0.0, 3: 0.0}


@pytest.mark.parametrize("value", [[0.9, 0.8], 1.2, -0.1])
def test_improvement_allocation_rejects_invalid_node_probabilities(value):
    with pytest.raises(ValueError, match="node_probabilities"):
        _series().improvement_allocation(0.9, {1: value})
