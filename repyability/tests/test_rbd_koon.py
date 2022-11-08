"""
Tests many of the RBD class methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

from repyability.rbd.rbd import RBD


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
