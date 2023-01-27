"""
Tests Repeated Nodes.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest
from surpyval import Weibull

from repyability.rbd.repeated_node import RepeatedNode


def test_repated_once_sf_correct_series():
    model = Weibull.from_params([10, 2])
    repeated_with_1 = RepeatedNode(model, 1, "series")
    assert model.sf(10) == repeated_with_1.sf(10)
    assert model.ff(10) == repeated_with_1.ff(10)
    assert pytest.approx(model.mean(), rel=1e-2) == repeated_with_1.mean()


def test_repated_once_sf_correct_parallel():
    model = Weibull.from_params([10, 2])
    repeated_with_1 = RepeatedNode(model, 1, "parallel")
    assert model.sf(10) == repeated_with_1.sf(10)
    assert model.ff(10) == repeated_with_1.ff(10)
    assert pytest.approx(model.mean(), rel=1e-2) == repeated_with_1.mean()


def test_repated_twice_sf_correct_series():
    model = Weibull.from_params([10, 2])
    repeated_twice = RepeatedNode(model, 2, "series")
    assert model.sf(10) ** 2 == repeated_twice.sf(10)
    assert 1 - ((1 - model.ff(10)) ** 2) == repeated_twice.ff(10)


def test_repated_50_sf_correct_series():
    model = Weibull.from_params([10, 2])
    repeated_twice = RepeatedNode(model, 50, "series")
    assert model.sf(10) ** 50 == repeated_twice.sf(10)
    assert 1 - ((1 - model.ff(10)) ** 50) == repeated_twice.ff(10)


def test_repated_twice_sf_correct_parallel():
    model = Weibull.from_params([10, 2])
    repeated_twice = RepeatedNode(model, 2, "parallel")
    assert 1 - ((1 - model.sf(10)) ** 2) == repeated_twice.sf(10)
    assert model.ff(10) ** 2 == repeated_twice.ff(10)


def test_repated_50_sf_correct_parallel():
    model = Weibull.from_params([10, 2])
    repeated_twice = RepeatedNode(model, 50, "parallel")
    assert 1 - ((1 - model.sf(10)) ** 50) == repeated_twice.sf(10)
    assert model.ff(10) ** 50 == repeated_twice.ff(10)
