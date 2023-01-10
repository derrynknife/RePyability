"""
Tests Standby Nodes.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import numpy as np
import pytest
from surpyval import Normal, Weibull

from repyability.non_repairable import NonRepairable
from repyability.rbd.standby_node import StandbyModel


def test_incorrect_args():
    with pytest.raises(ValueError):
        StandbyModel(
            reliabilities=[
                Weibull.from_params([10, 2]),
                Weibull.from_params([10, 2]),
                Weibull.from_params([10, 2]),
            ],
            k=4,
        )


def test_mean():
    stdby = StandbyModel(
        reliabilities=[
            Normal.from_params([10, 2]),
            Normal.from_params([10, 2]),
            Normal.from_params([10, 2]),
        ],
        k=1,
    )
    assert pytest.approx(stdby.mean(), abs=1e-1) == 30.0


def test_mean_k2():
    model = Normal.from_params([10, 2])
    stdby = StandbyModel(reliabilities=[model, model, model], k=2)

    expected = (
        np.vstack([model.random(10_000), model.random(10_000)])
        .max(axis=0)
        .mean()
    )
    assert pytest.approx(stdby.mean(), abs=1e-1) == expected


def test_in_non_repairable():
    stdby = StandbyModel(
        reliabilities=[
            Normal.from_params([10, 2]),
            Normal.from_params([10, 2]),
            Normal.from_params([10, 2]),
        ],
        k=2,
    )
    replace = Weibull.from_params([10, 2])
    non_repairable = NonRepairable(stdby, replace)
    assert (
        pytest.approx(non_repairable.mean_availability(), abs=1e-3) == 0.5568
    )
