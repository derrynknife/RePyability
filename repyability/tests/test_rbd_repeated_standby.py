"""
Tests Repeated Nodes.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest
from surpyval import Exponential, Gamma, Weibull

from repyability.rbd.repeated_standby_node import RepeatedStandbyNode

exp_model = Exponential.from_params([1])


def test_repated_once():
    model = Weibull.from_params([10, 2])
    repeated_with_1 = RepeatedStandbyNode(model, 1, N=20_000)
    assert pytest.approx(model.sf(10), rel=1e-1) == repeated_with_1.sf(10)
    assert pytest.approx(model.ff(10), rel=1e-1) == repeated_with_1.ff(10)
    assert pytest.approx(model.mean(), rel=1e-1) == repeated_with_1.mean()


def test_random():
    node = RepeatedStandbyNode(exp_model, 3, N=20_000)
    size = 1000
    randoms = node.random(size)
    assert randoms.shape == (size,)


def test_mean():
    node = RepeatedStandbyNode(exp_model, 3, N=20_000)
    mean = node.mean()
    assert pytest.approx(mean, rel=1e-1) == 3 * 3


def test_sf():
    node = RepeatedStandbyNode(exp_model, 3, N=50_000)
    assert pytest.approx(node.sf(10), rel=1e-1) == Gamma.from_params(
        [3, 1.0]
    ).sf(10)
    assert pytest.approx(node.sf(1), rel=1e-1) == Gamma.from_params(
        [3, 1.0]
    ).sf(1)


def test_ff():
    node = RepeatedStandbyNode(exp_model, 3)
    assert pytest.approx(node.ff(10), rel=1e-1) == Gamma.from_params(
        [3, 1.0]
    ).ff(10)
    assert pytest.approx(node.ff(1), rel=1e-1) == Gamma.from_params(
        [3, 1.0]
    ).ff(1)
