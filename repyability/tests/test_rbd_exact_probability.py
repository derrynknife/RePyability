"""
Tests the exact reliability engine (probability_any_set_satisfied, used by
system_probability) and the BDD-free is_system_working() evaluation.

The exact engine is cross-checked against a brute-force enumeration over every
component up/down combination, on a non-series-parallel (bridge) network where
the cut-set and path-set methods must both equal the true value.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import itertools

import pytest
from surpyval import FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD


def _brute_force_reliability(rbd: NonRepairableRBD, rel: dict) -> float:
    """Reliability by summing P(state) over every state in which the system
    works, using the path-set structure function directly."""
    nodes = list(rbd.nodes)
    path_sets = [
        set(p) for p in rbd.get_min_path_sets(include_in_out_nodes=False)
    ]
    total = 0.0
    for bits in itertools.product([0, 1], repeat=len(nodes)):
        status = {n: bool(b) for n, b in zip(nodes, bits)}
        works = any(all(status[c] for c in ps) for ps in path_sets)
        if not works:
            continue
        p = 1.0
        for n, b in zip(nodes, bits):
            p *= rel[n] if b else (1 - rel[n])
        total += p
    return total


def _bridge_rbd(rel: dict) -> NonRepairableRBD:
    # rbd3 topology: a bridge network that is not reducible to series/parallel.
    edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (3, 4),
        (1, 5),
        (3, 5),
        (5, 2),
        (5, 4),
        (2, 6),
        (4, 6),
    ]
    reliabilities = {
        n: FixedEventProbability.from_params(1 - r) for n, r in rel.items()
    }
    return NonRepairableRBD(edges, reliabilities)


def test_exact_matches_brute_force_bridge():
    rel = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.85, 5: 0.6}
    rbd = _bridge_rbd(rel)
    expected = _brute_force_reliability(rbd, rel)
    assert rbd.sf(1, method="p")[0] == pytest.approx(expected)
    assert rbd.sf(1, method="c")[0] == pytest.approx(expected)


def test_cut_and_path_methods_agree(
    rbd3: NonRepairableRBD,
    rbd1: NonRepairableRBD,
    rbd_parallel: NonRepairableRBD,
):
    for rbd in (rbd3, rbd1, rbd_parallel):
        assert rbd.sf(1, method="p") == pytest.approx(rbd.sf(1, method="c"))


def test_is_system_working_series(rbd_series: NonRepairableRBD):
    status = {n: True for n in rbd_series.G.nodes}
    assert rbd_series.is_system_working(status, "p")
    assert rbd_series.is_system_working(status, "c")

    # Any single failure breaks a series system.
    status[3] = False
    assert not rbd_series.is_system_working(status, "p")
    assert not rbd_series.is_system_working(status, "c")


def test_is_system_working_parallel(rbd_parallel: NonRepairableRBD):
    status = {n: True for n in rbd_parallel.G.nodes}
    # All redundant branches down -> system down.
    status[2] = status[3] = status[4] = False
    assert not rbd_parallel.is_system_working(status, "p")
    assert not rbd_parallel.is_system_working(status, "c")

    # A single branch back up -> system up.
    status[3] = True
    assert rbd_parallel.is_system_working(status, "p")
    assert rbd_parallel.is_system_working(status, "c")


def test_is_system_working_bad_method(rbd_series: NonRepairableRBD):
    status = {n: True for n in rbd_series.G.nodes}
    with pytest.raises(ValueError):
        rbd_series.is_system_working(status, "x")
