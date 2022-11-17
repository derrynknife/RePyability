"""
Tests RBD's sf() method (and the briefly the ff() method).

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest

from repyability.rbd.rbd import RBD
from repyability.tests.fixed_probability import FixedProbabilityFitter


# Test sf() w/ simple series RBD
def test_rbd_sf_series(rbd_series: RBD):
    t = 5
    assert (
        pytest.approx(
            rbd_series.reliability[2].sf(t)
            * rbd_series.reliability[3].sf(t)
            * rbd_series.reliability[4].sf(t)
        )
        == rbd_series.sf(t)[0]
    )


# Test sf() w/ simple series RBD
def test_rbd_sf_parallel(rbd_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_parallel.reliability[2].sf(t))
            * (1 - rbd_parallel.reliability[3].sf(t))
            * (1 - rbd_parallel.reliability[4].sf(t))
        )
        == 1 - rbd_parallel.sf(t)[0]
    )


# Test sf() w/ a simple RBD with both parallel and series components
def test_rbd_sf_composite(rbd1: RBD):
    """Tests with an RBD with both parallel and series components."""
    t = 2
    assert (
        pytest.approx(
            (
                1
                - (1 - rbd1.reliability["pump1"].sf(t))
                * (1 - rbd1.reliability["pump2"].sf(t))
            )
            * rbd1.reliability["valve"].sf(t)
        )
        == rbd1.sf(t)[0]
    )


# Test sf() w/ an RBD that cannot be reduced to parallel or series
def test_rbd_sf_complex(rbd3: RBD):
    assert pytest.approx(0.994780625) == rbd3.sf(1000)[0]


# Test sf() w/ repeated component
# i.e. Two nodes correspond to one node
def test_rbd_sf_repeated_component(rbd_repeated_component_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_repeated_component_parallel.reliability[2].sf(t))
            * (1 - rbd_repeated_component_parallel.reliability[3].sf(t))
            * (1 - rbd_repeated_component_parallel.reliability[4].sf(t))
            # Not 'component 5' as nodes 2 and 5 are both component 2
        )
        == 1 - rbd_repeated_component_parallel.sf(t)[0]
    )


# Test sf() w/ broken node
def test_rbd_sf_broken_node(rbd_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_parallel.reliability[2].sf(t))
            * (1 - rbd_parallel.reliability[4].sf(t))
        )
        == 1 - rbd_parallel.sf(t, broken_nodes=[3])[0]
    )


# Test sf() w/ broken component
def test_rbd_sf_broken_component(rbd_repeated_component_parallel: RBD):
    t = 2
    assert (
        pytest.approx(
            (1 - rbd_repeated_component_parallel.reliability[2].sf(t))
            * (1 - rbd_repeated_component_parallel.reliability[4].sf(t))
        )
        == 1 - rbd_repeated_component_parallel.sf(t, broken_components=[3])[0]
    )
    assert (
        pytest.approx(
            (1 - rbd_repeated_component_parallel.reliability[3].sf(t))
            * (1 - rbd_repeated_component_parallel.reliability[4].sf(t))
        )
        == 1 - rbd_repeated_component_parallel.sf(t, broken_components=[2])[0]
    )


# Test sf() w/ working node
def test_rbd_sf_working_node(rbd_series: RBD):
    t = 2
    assert (
        pytest.approx(
            rbd_series.reliability[2].sf(t) * rbd_series.reliability[4].sf(t)
        )
        == rbd_series.sf(t, working_nodes=[3])[0]
    )


# Test sf() w/ working component
def test_rbd_sf_working_component(rbd_repeated_component_parallel: RBD):
    t = 2
    assert (
        pytest.approx(1)
        == rbd_repeated_component_parallel.sf(t, working_components=[2])[0]
    )
    assert (
        pytest.approx(1)
        == rbd_repeated_component_parallel.sf(t, working_components=[3])[0]
    )


# Test sf() w/ working node with repeated component
def test_rbd_sf_working_node_repeated_component(
    rbd_repeated_component_composite: RBD,
):
    rbd = rbd_repeated_component_composite
    t = 2
    assert (
        pytest.approx(1 - rbd.reliability[2].ff(t) * rbd.reliability[3].ff(t))
        == rbd.sf(t, working_nodes=[3])[0]
    )


# Test ff(), can just test parallel as ff() just calls 1 - sf()
def test_rbd_ff(rbd_parallel: RBD):
    t = 5
    assert (
        pytest.approx(
            (1 - rbd_parallel.reliability[2].sf(t))
            * (1 - rbd_parallel.reliability[3].sf(t))
            * (1 - rbd_parallel.reliability[4].sf(t))
        )
        == rbd_parallel.ff(t)[0]
    )


# Test ValueError


def test_rbd_sf_arg_inconsistency(rbd1: RBD):
    with pytest.raises(ValueError):
        rbd1.sf(2, working_nodes=["pump1"], broken_components=["pump1"])


# Test sf() w/ an RBD as a node
def test_rbd_sf_RBD_as_node(rbd1: RBD):
    """rbd_parallel but node 3 is rbd1."""
    nodes = {1: "input_node", 2: 2, 3: "RBD", 4: 4, 5: "output_node"}
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]
    components = {
        2: FixedProbabilityFitter.from_params(0.8),
        "RBD": rbd1,
        4: FixedProbabilityFitter.from_params(0.85),
    }
    rbd = RBD(nodes, components, edges)
    t = 2
    assert pytest.approx(rbd1.sf(t)) == rbd.reliability["RBD"].sf(t)
    assert (
        pytest.approx(
            rbd.reliability[2].ff(t)
            * rbd.reliability["RBD"].ff(t)
            * rbd.reliability[4].ff(t)
        )
        == rbd.ff(t)[0]
    )
