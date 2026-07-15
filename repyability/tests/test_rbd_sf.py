"""
Tests NonRepairableRBD's sf() method (and the briefly the ff() method),
using both the cut set and path set methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import pytest
from surpyval import FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD


def test_rbd_sf_ff_by_node(rbd_series: NonRepairableRBD):
    t = 5
    node_sf = rbd_series.sf_by_node(t)
    node_ff = rbd_series.ff_by_node(t)
    for k in node_sf.keys():
        # surpyval >= 0.11 computes ff(t) directly (more precisely than
        # 1 - sf(t)), so the two can differ by a floating-point ULP.
        assert node_ff[k] == pytest.approx(1 - node_sf[k])


def test_rel_unrel(rbd_series: NonRepairableRBD):
    t = 5
    rel = rbd_series.reliability(t)
    unrel = rbd_series.unreliability(t)
    assert unrel == 1 - rel


# Test sf() w/ simple series NonRepairableRBD
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_series(rbd_series: NonRepairableRBD, method: str):
    t = 5
    assert (
        pytest.approx(
            rbd_series.reliabilities[2].sf(t)
            * rbd_series.reliabilities[3].sf(t)
            * rbd_series.reliabilities[4].sf(t)
        )
        == rbd_series.sf(t, method=method)[0]
    )


# Test sf() w/ simple series NonRepairableRBD
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_parallel(rbd_parallel: NonRepairableRBD, method: str):
    t = 2
    assert pytest.approx(
        (1 - rbd_parallel.reliabilities[2].sf(t))
        * (1 - rbd_parallel.reliabilities[3].sf(t))
        * (1 - rbd_parallel.reliabilities[4].sf(t))
    ) == 1 - rbd_parallel.sf(t, method=method)


# Test sf() w/ a simple NonRepairableRBD with both parallel and
# series components
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_composite(rbd1: NonRepairableRBD, method: str):
    """Tests with an NonRepairableRBD with both parallel and series
    components."""
    t = 2
    assert pytest.approx(
        (
            1
            - (1 - rbd1.reliabilities["pump1"].sf(t))
            * (1 - rbd1.reliabilities["pump2"].sf(t))
        )
        * rbd1.reliabilities["valve"].sf(t)
    ) == rbd1.sf(t, method=method)


# Test sf() w/ an NonRepairableRBD that cannot be reduced to parallel or series
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_complex(rbd3: NonRepairableRBD, method: str):
    assert pytest.approx(0.994780625) == rbd3.sf(1000, method=method)


# Test sf() w/ repeated component
# i.e. Two nodes correspond to one node
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_repeated_component(
    rbd_repeated_component_parallel: NonRepairableRBD, method: str
):
    t = 2
    assert pytest.approx(
        (1 - rbd_repeated_component_parallel.reliabilities[2].sf(t))
        * (1 - rbd_repeated_component_parallel.reliabilities[3].sf(t))
        * (1 - rbd_repeated_component_parallel.reliabilities[4].sf(t))
        # Not 'component 5' as nodes 2 and 5 are both component 2
    ) == 1 - rbd_repeated_component_parallel.sf(t, method=method)


# Test sf() w/ broken node
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_broken_node(rbd_parallel: NonRepairableRBD, method: str):
    t = 2
    assert pytest.approx(
        (1 - rbd_parallel.reliabilities[2].sf(t))
        * (1 - rbd_parallel.reliabilities[4].sf(t))
    ) == 1 - rbd_parallel.sf(t, broken_nodes=[3], method=method)


# Test sf() w/ working node
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_working_node(rbd_series: NonRepairableRBD, method: str):
    t = 2
    assert pytest.approx(
        rbd_series.reliabilities[2].sf(t) * rbd_series.reliabilities[4].sf(t)
    ) == rbd_series.sf(t, working_nodes=[3], method=method)


# Test ff(), can just test parallel as ff() just calls 1 - sf()
def test_rbd_ff(rbd_parallel: NonRepairableRBD):
    t = 5
    assert pytest.approx(
        (1 - rbd_parallel.reliabilities[2].sf(t))
        * (1 - rbd_parallel.reliabilities[3].sf(t))
        * (1 - rbd_parallel.reliabilities[4].sf(t))
    ) == rbd_parallel.ff(t)


# Test sf() w/ an NonRepairableRBD as a node
@pytest.mark.parametrize("method", ["c", "p"])
def test_rbd_sf_NonRepairableRBD_as_node(rbd1: NonRepairableRBD, method: str):
    """rbd_parallel but node 3 is rbd1."""
    edges = [
        (1, 2),
        (1, "NonRepairableRBD"),
        (1, 4),
        (2, 5),
        ("NonRepairableRBD", 5),
        (4, 5),
    ]
    reliabilities = {
        2: FixedEventProbability.from_params(1 - 0.8),
        "NonRepairableRBD": rbd1,
        4: FixedEventProbability.from_params(1 - 0.85),
    }
    rbd = NonRepairableRBD(edges, reliabilities)
    t = 2
    assert pytest.approx(rbd1.sf(t)) == rbd.reliabilities[
        "NonRepairableRBD"
    ].sf(t, method=method)
    assert pytest.approx(
        rbd.reliabilities[2].ff(t)
        * rbd.reliabilities["NonRepairableRBD"].ff(t)
        * rbd.reliabilities[4].ff(t)
    ) == rbd.ff(t, method=method)


def test_rbd_conditional_survival():
    """cs(x, X) returns sf(X + x) / sf(X), clipped to [0, 1]."""
    import numpy as np
    from surpyval import Exponential, Weibull

    rbd = NonRepairableRBD(
        [("input", "c1"), ("c1", "c2"), ("c2", "output")],
        {
            "c1": Weibull.from_params([100, 2]),
            "c2": Exponential.from_params([0.01]),
        },
    )
    x = np.array([10.0, 20.0, 50.0])
    X = 30.0
    expected = rbd.sf(x + X) / rbd.sf(X)
    assert np.allclose(rbd.cs(x, X), expected)
    # Surviving zero further time is certain.
    assert rbd.cs(0.0, X)[0] == 1.0
