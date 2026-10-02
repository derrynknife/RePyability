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


def test_a_simulated_arrangement_has_one_mean():
    # Its mean is that of the lifetimes its Kaplan-Meier fit is made from:
    # the same at every call, the area under its sf, and made without
    # touching numpy's global RNG.
    w = Weibull.from_params([100, 2])
    sim = StandbyModel(
        [w, w, w], k=2, dormancy_factor=0.5, mc_samples=2000, seed=1
    )
    assert sim.model is not None
    before = np.random.get_state()[1].copy()
    assert sim.mean() == sim.mean() == sim.random(2000, seed=1).mean()
    assert np.array_equal(np.random.get_state()[1], before)
    t = np.linspace(0.0, 1000.0, 200_001)
    area = np.trapezoid(np.ravel(sim.sf(t)), t)
    assert area == pytest.approx(sim.mean(), rel=1e-4)
    # A fresh estimate, from new draws, when asked for.
    assert sim.mean(mc_samples=2000, seed=2) != sim.mean()


def test_a_repairable_rbds_exact_values_repeat_with_a_simulated_node():
    from surpyval import Exponential

    from repyability import RepairableRBD

    w = Weibull.from_params([100, 2])
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {
                "reliability": StandbyModel([w, w, w], k=2),
                "repairability": Exponential.from_params([0.5]),
            },
            "b": {
                "reliability": Weibull.from_params([200, 1.5]),
                "repairability": Exponential.from_params([1.0]),
            },
        },
    )
    assert rbd.mean_availability() == rbd.mean_availability()
    assert rbd.system_failure_frequency() == rbd.system_failure_frequency()
    assert rbd.birnbaum_importance() == rbd.birnbaum_importance()
