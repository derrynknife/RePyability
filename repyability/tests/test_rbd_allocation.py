"""
Tests allocation cases for the RBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
# import pytest
# from numpy.random import rand

# from repyability.rbd.rbd import RBD


# Test that allocation to a series system works.
# def test_series_allocation():
#     two_inputs_edges = [
#         (0, 1),
#         (1, 2),
#         (2, 3),
#         (3, 4),
#         (4, 5),
#         (5, 6),
#         (6, 7),
#     ]
#     rbd = RBD(two_inputs_edges)
#     for i in range(10):
#         target = rand()
#         allocated_probs = rbd.allocate_probability(target)
#         achieved = rbd.system_probability(allocated_probs)
#         assert pytest.approx(achieved, rel=1e-3) == target

#     for i in range(10):
#         target = rand()
#         allocated_probs = rbd.allocate_probability(target)
#       assert pytest.approx(allocated_probs[1], rel=1e-3) == target ** (1 / 6)


# Test that allocation to a parallel system works.
# def test_parallel_allocation():
#     two_inputs_edges = [
#         (0, 1),
#         (0, 2),
#         (0, 3),
#         (0, 4),
#         (0, 5),
#         (0, 6),
#         (1, 7),
#         (2, 7),
#         (3, 7),
#         (4, 7),
#         (5, 7),
#         (6, 7),
#     ]
#     rbd = RBD(two_inputs_edges)
#     for i in range(10):
#         target = rand()
#         allocated_probs = rbd.allocate_probability(target)
#         achieved = rbd.system_probability(allocated_probs)
#         assert pytest.approx(achieved, rel=1e-3) == target

#     for i in range(10):
#         target = rand()
#         allocated_probs = rbd.allocate_probability(target)
#         assert pytest.approx(allocated_probs[1], rel=1e-3) == 1 - (
#             1 - target
#         ) ** (1 / 6)
