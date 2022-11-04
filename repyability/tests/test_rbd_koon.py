import pytest

from repyability.rbd.rbd import RBD
from repyability.tests.fixed_probability import FixedProbabilityFitter


# koon RBD pytest fixtures
@pytest.fixture
def rbd_koon1() -> RBD:
    qp = 0.03
    qv = 0.01
    nodes = {
        "source": "input_node",
        "pump1": "pump1",
        "pump2": "pump2",
        "valve": "valve",
        "sink": "output_node",
    }
    edges = [
        ("source", "pump1"),
        ("source", "pump2"),
        ("pump1", "valve"),
        ("pump2", "valve"),
        ("valve", "sink"),
    ]
    components = {
        "pump1": FixedProbabilityFitter.from_params(1 - qp),
        "pump2": FixedProbabilityFitter.from_params(1 - qp),
        "valve": FixedProbabilityFitter.from_params(1 - qv),
    }
    k = {"valve": 2}
    return RBD(nodes, components, edges, k)


def test_rbd_koon_default_k(rbd_series: RBD):
    assert rbd_series.G.nodes[1]["k"] == 1
    assert rbd_series.G.nodes[2]["k"] == 1
    assert rbd_series.G.nodes[3]["k"] == 1
    assert rbd_series.G.nodes[4]["k"] == 1
    assert rbd_series.G.nodes[5]["k"] == 1


def test_rbd_koon_k_given(rbd_koon1: RBD):
    assert rbd_koon1.G.nodes["source"]["k"] == 1
    assert rbd_koon1.G.nodes["pump1"]["k"] == 1
    assert rbd_koon1.G.nodes["pump2"]["k"] == 1
    assert rbd_koon1.G.nodes["valve"]["k"] == 2
    assert rbd_koon1.G.nodes["sink"]["k"] == 1
