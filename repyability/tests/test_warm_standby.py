"""Tests for warm/hot standby (``StandbyModel(dormancy_factor=...)``).

``dormancy_factor`` is the dormant-to-operating aging ratio: 0 is the
existing cold standby, 1 is hot (an ordinary k-of-n parallel arrangement).
Identical Exponential units make everything exactly computable — with ``j``
units alive the stage rate is ``lam * (k + (j - k) * kappa)`` and the
lifetime is hypoexponential over the stages — so most expectations here are
closed-form arithmetic, and the Monte-Carlo path is checked against them.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, StandbyModel

LAM = 0.1
EXP = surv.Exponential.from_params([LAM])


# -- exact closed forms (identical Exponential units) ----------------------


def test_hot_standby_is_exactly_parallel():
    # kappa = 1: dormant spares age like operating units, so 1-of-2 hot
    # standby is the ordinary two-unit parallel arrangement.
    hot = StandbyModel([EXP, EXP], dormancy_factor=1.0)
    x = np.array([5.0, 15.0, 40.0])
    parallel = 2.0 * np.exp(-LAM * x) - np.exp(-2.0 * LAM * x)
    assert np.allclose(np.ravel(hot.sf(x)), parallel)
    assert hot.mean() == pytest.approx(1.5 / LAM)


@pytest.mark.parametrize("kappa", [0.3, 0.7])
def test_warm_mean_matches_stage_arithmetic(kappa):
    # 1-of-2: stage rates are lam*(1 + kappa) (both alive) then lam.
    warm = StandbyModel([EXP, EXP], dormancy_factor=kappa)
    expected = 1.0 / (LAM * (1.0 + kappa)) + 1.0 / LAM
    assert warm.mean() == pytest.approx(expected)


def test_warm_k2_of_3_mean():
    # 2-of-3: stages are (2 operating + 1 dormant) then (2 operating).
    warm = StandbyModel([EXP, EXP, EXP], k=2, dormancy_factor=0.5)
    expected = 1.0 / (LAM * (2.0 + 0.5)) + 1.0 / (2.0 * LAM)
    assert warm.mean() == pytest.approx(expected)


def test_dormancy_orders_the_survival_curves():
    # More dormant aging can only hurt: cold >= warm >= hotter warm >= hot.
    x = 12.0
    sfs = [
        float(np.ravel(StandbyModel([EXP, EXP], dormancy_factor=k).sf([x]))[0])
        for k in (0.0, 0.3, 0.7, 1.0)
    ]
    assert all(a >= b for a, b in zip(sfs, sfs[1:]))
    assert sfs[0] > sfs[-1]  # cold is strictly better than hot


def test_cold_default_is_unchanged():
    # dormancy_factor defaults to 0 and reproduces the cold result (the sum
    # of two exponential lifetimes: mean 2 / lam).
    cold = StandbyModel([EXP, EXP])
    assert cold.dormancy_factor == 0.0
    assert cold.mean() == pytest.approx(2.0 / LAM)


# -- the Monte-Carlo path agrees with the closed form ----------------------


def test_warm_simulation_matches_closed_form():
    warm = StandbyModel([EXP, EXP], dormancy_factor=0.5)
    draws = warm.random(40_000, seed=2)
    for x in (5.0, 15.0, 40.0):
        closed = float(np.ravel(warm.sf([x]))[0])
        assert np.mean(draws > x) == pytest.approx(closed, abs=0.01)


def test_weibull_warm_sits_between_cold_and_hot():
    # Non-exponential units take the simulated (Kaplan-Meier) path; warm must
    # land between the cold sum and the hot parallel arrangement.
    W = surv.Weibull.from_params([50.0, 2.0])
    cold = StandbyModel([W, W]).mean()
    warm = StandbyModel([W, W], dormancy_factor=0.4, n_sims=4000, seed=7)
    hot = StandbyModel([W, W], dormancy_factor=1.0, n_sims=4000, seed=7)
    assert warm.model is not None  # simulated, not closed form
    assert float(hot.mean()) < float(warm.mean()) < float(cold)


def test_warm_random_is_reproducible():
    warm = StandbyModel([EXP, EXP], dormancy_factor=0.5)
    a = warm.random(500, seed=11)
    b = warm.random(500, seed=11)
    assert np.allclose(a, b)


# -- inside an RBD, and serialisation --------------------------------------


def test_warm_standby_node_in_rbd_round_trips():
    rbd = NonRepairableRBD(
        [("s", "g"), ("g", "t")],
        {"g": StandbyModel([EXP, EXP], dormancy_factor=0.5)},
    )
    restored = NonRepairableRBD.from_json(rbd.to_json())
    node = restored.reliabilities["g"]
    assert node.dormancy_factor == 0.5
    assert float(restored.sf(12.0)) == pytest.approx(float(rbd.sf(12.0)))


# -- validation ------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_dormancy_factor_out_of_range_rejected(bad):
    with pytest.raises(ValueError, match="dormancy_factor"):
        StandbyModel([EXP, EXP], dormancy_factor=bad)


def test_warm_with_imperfect_switching_rejected():
    with pytest.raises(NotImplementedError, match="switching"):
        StandbyModel(
            [EXP, EXP], dormancy_factor=0.5, switching_probability=0.9
        )
