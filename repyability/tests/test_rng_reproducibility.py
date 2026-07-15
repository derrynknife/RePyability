"""The Monte-Carlo entry points accept a ``seed`` for reproducible results.

surpyval's ``.random()`` draws from numpy's global RNG (it has no seed
argument), so reproducibility is achieved by seeding that global RNG for the
duration of a simulation and restoring it afterwards. A seed must therefore
(a) make results repeatable and (b) leave the caller's global RNG stream
undisturbed. These tests check both.
"""

import numpy as np
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD
from repyability.rbd.repeated_node import RepeatedNode
from repyability.rbd.standby_node import StandbyModel


def _series_rbd():
    edges = [("s", 1), (1, 2), (2, "t")]
    reliabilities = {
        1: surv.Weibull.from_params([20, 2]),
        2: surv.Weibull.from_params([100, 3]),
    }
    return NonRepairableRBD(edges, reliabilities)


def test_nonrepairable_random_reproducible_with_seed():
    rbd = _series_rbd()
    a = rbd.random(50, seed=42)
    b = rbd.random(50, seed=42)
    np.testing.assert_array_equal(a, b)
    # A different seed (very probably) gives a different draw.
    assert not np.array_equal(a, rbd.random(50, seed=7))


def test_nonrepairable_mean_reproducible_with_seed():
    rbd = _series_rbd()
    assert rbd.mean(2000, seed=1) == rbd.mean(2000, seed=1)


def test_seed_does_not_disturb_global_rng():
    rbd = _series_rbd()
    np.random.seed(123)
    before = np.random.random(5)
    np.random.seed(123)
    rbd.random(20, seed=42)  # seeds + restores the global RNG internally
    after = np.random.random(5)
    np.testing.assert_array_equal(before, after)


def test_no_seed_is_nondeterministic_but_valid():
    rbd = _series_rbd()
    out = rbd.random(10)  # seed=None -> no error, valid output
    assert out.shape == (10,)
    assert np.all(out >= 0)


def test_standby_random_reproducible_with_seed():
    models = [surv.Weibull.from_params([5, 1.1]) for _ in range(3)]
    standby = StandbyModel(models)
    np.testing.assert_array_equal(
        standby.random(100, seed=3), standby.random(100, seed=3)
    )


def test_repeated_node_random_reproducible_with_seed():
    node = RepeatedNode(
        surv.Weibull.from_params([10, 2]), repeats=3, kind="parallel"
    )
    np.testing.assert_array_equal(
        node.random(100, seed=9), node.random(100, seed=9)
    )


def test_repairable_availability_reproducible_with_seed():
    rbd = RepairableRBD(
        edges=[("s", "c"), ("c", "t")],
        components={
            "c": {
                "reliability": surv.Exponential.from_params([0.25]),
                "repairability": surv.Exponential.from_params([0.5]),
            }
        },
    )
    r1 = rbd.availability(t_simulation=10.0, N=200, seed=5)
    r2 = rbd.availability(t_simulation=10.0, N=200, seed=5)
    np.testing.assert_array_equal(r1["availability"], r2["availability"])
    assert r1["system_uptime"] == r2["system_uptime"]
