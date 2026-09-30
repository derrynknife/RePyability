"""Tests for the load-sharing dynamic node (issue #38): coupled units that
share a load and age faster as siblings fail.

The exact anchor is the ``phi == 1`` limit (no load effect), where the group
lifetime is the ``(N-k+1)``-th order statistic of ``N`` i.i.d. Exponentials --
the ordinary k-out-of-n parallel result -- which the hypoexponential closed
form must reproduce exactly.
"""

import numpy as np
import pytest
import surpyval as surv
from scipy.special import comb

from repyability import LoadSharingModel, NonRepairableRBD


@pytest.fixture(scope="module")
def exp_aft():
    """An Exponential-AFT unit fitted with load as the covariate (higher load
    -> shorter life), so the identical-unit closed form applies."""
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=800)
    x = rng.exponential(scale=100.0 / np.exp(0.6 * (load - 1.0)), size=800)
    return surv.ExponentialAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))


def _k_of_n_sf(t, n, k, lam):
    """P(>= k of n i.i.d. Exp(lam) units still alive at t)."""
    p = np.exp(-lam * t)
    return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


# -- exact validation: phi == 1 reproduces k-out-of-n --------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_phi1_hypoexponential_is_exact_k_of_n(k):
    # With no load effect the stage rates are s*lambda (s = N..k); the
    # hypoexponential of those rates is exactly the (N-k+1)-th order statistic
    # of N i.i.d. Exponentials. This pins the surpyval contract the closed
    # form relies on: Hypoexponential.from_params(rates) is the sum of
    # independent Exponentials with those rates.
    lam, n = 0.02, 4
    rates = np.array([s * lam for s in range(n, k - 1, -1)])
    hypo = surv.Hypoexponential.from_params(rates)
    t = np.array([10.0, 40.0, 90.0, 180.0])
    expected = np.array([_k_of_n_sf(ti, n, k, lam) for ti in t])
    assert np.allclose(np.ravel(hypo.sf(t)), expected)
    assert float(np.ravel(hypo.sf(np.array([0.0])))[0]) == pytest.approx(1.0)
    # mean of the hypoexponential = sum 1/rate
    assert float(np.ravel(hypo.mean())[0]) == pytest.approx(
        np.sum(1.0 / rates)
    )


# -- closed form <-> Monte-Carlo agreement --------------------------------


def test_closed_form_matches_monte_carlo(exp_aft):
    ls = LoadSharingModel([exp_aft] * 3, load=3.0, k=1)
    assert ls.is_simulated is False  # identical Exp baseline -> closed form
    mc = surv.KaplanMeier.fit(ls.random(15000, seed=1))
    t = np.array([40.0, 120.0, 300.0])
    assert np.allclose(ls.sf(t), mc.sf(t), atol=0.025)


@pytest.fixture(scope="module")
def weibull_aft():
    """A Weibull-AFT unit: its baseline forces the Monte-Carlo
    (Kaplan-Meier) path."""
    rng = np.random.default_rng(1)
    load = rng.uniform(0.5, 2.0, size=400)
    x = rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1)) + 1e-3
    return surv.WeibullAFT.fit(x, Z=load.reshape(-1, 1))


def test_reproducible_with_seed(weibull_aft):
    waft = weibull_aft
    a = LoadSharingModel([waft, waft], load=2.0, k=1, n_sims=500, seed=7)
    b = LoadSharingModel([waft, waft], load=2.0, k=1, n_sims=500, seed=7)
    assert a.is_simulated is True
    t = np.array([30.0, 90.0])
    assert np.allclose(a.sf(t), b.sf(t))


def test_a_simulated_group_has_one_mean(weibull_aft):
    # The mean of the lifetimes its fit is made from, the same at every
    # call, and made without touching numpy's global RNG.
    group = LoadSharingModel(
        [weibull_aft] * 2, load=2.0, k=1, n_sims=500, seed=7
    )
    assert group.is_simulated
    before = np.random.get_state()[1].copy()
    assert group.mean() == group.mean() == group.random(500, seed=7).mean()
    assert np.array_equal(np.random.get_state()[1], before)
    assert group.mean(N=500, seed=8) != group.mean()


# -- dependent failure: sharing shortens life -----------------------------


def test_more_shared_load_shortens_life(exp_aft):
    light = LoadSharingModel([exp_aft] * 3, load=1.5, k=1).mean()
    heavy = LoadSharingModel([exp_aft] * 3, load=6.0, k=1).mean()
    assert heavy < light


def test_higher_k_shortens_life(exp_aft):
    # Needing more survivors (higher k) fails the group sooner.
    k1 = LoadSharingModel([exp_aft] * 3, load=3.0, k=1).mean()
    k3 = LoadSharingModel([exp_aft] * 3, load=3.0, k=3).mean()
    assert k3 < k1


def test_is_simulated_flag(exp_aft):
    # Identical Exponential baseline -> exact closed form.
    assert LoadSharingModel([exp_aft] * 2, load=2.0).is_simulated is False


# -- inside an RBD --------------------------------------------------------


@pytest.fixture(scope="module")
def rbd(exp_aft):
    grp = LoadSharingModel([exp_aft] * 3, load=3.0, k=2)
    return NonRepairableRBD(
        [("s", "g"), ("g", "c"), ("c", "t")],
        {"g": grp, "c": surv.Weibull.from_params([300.0, 2.0])},
    )


def test_rbd_is_time_varying_and_non_analytic(rbd):
    assert rbd.is_time_varying
    assert rbd.is_analytically_solvable() is False  # a simulation node


def test_rbd_evaluates_and_mttf(rbd):
    assert 0.0 < float(rbd.sf(60.0)) < 1.0
    assert rbd.mean(2000, seed=1) > 0.0
    mttf = rbd.node_mttf(mc_samples=1500, seed=1)
    assert mttf["g"] > 0.0 and mttf["c"] > 0.0


# -- serialisation --------------------------------------------------------


def test_rbd_with_load_sharing_json_roundtrip(rbd):
    restored = NonRepairableRBD.from_json(rbd.to_json())
    g = restored.reliabilities["g"]
    assert isinstance(g, LoadSharingModel)
    assert g.load == 3.0 and g.k == 2 and g.N == 3
    t = np.array([40.0, 120.0])
    np.testing.assert_allclose(
        restored.reliabilities["g"].sf(t),
        rbd.reliabilities["g"].sf(t),
    )


# -- construction validation ----------------------------------------------


def test_non_aft_unit_rejected():
    with pytest.raises(ValueError, match="accelerated-failure-time|AFT"):
        LoadSharingModel([surv.Weibull.from_params([100.0, 2.0])], load=1.0)


def test_k_bounds(exp_aft):
    with pytest.raises(ValueError, match="k"):
        LoadSharingModel([exp_aft, exp_aft], load=1.0, k=3)
    with pytest.raises(ValueError, match="k"):
        LoadSharingModel([exp_aft], load=1.0, k=0)


def test_empty_units_rejected():
    with pytest.raises(ValueError, match="at least one"):
        LoadSharingModel([], load=1.0)


@pytest.mark.parametrize(
    "x, shape",
    [
        (60.0, ()),
        ([60.0], (1,)),
        ([30.0, 90.0], (2,)),
        ([[30.0], [90.0]], (2, 1)),
    ],
)
def test_simulated_group_gives_the_shape_it_is_given(x, shape):
    """A simulated group (a Kaplan-Meier fit) answers in the shape of its
    query, a float for one time, as the closed forms do (surpyval#381)."""
    rng = np.random.default_rng(1)
    load = rng.uniform(0.5, 2.0, size=400)
    lives = rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1))
    waft = surv.WeibullAFT.fit(lives + 1e-3, Z=load.reshape(-1, 1))
    group = LoadSharingModel([waft, waft], load=2.0, k=1, n_sims=500, seed=7)
    assert group.is_simulated
    for function in (group.sf, group.ff):
        assert np.shape(function(x)) == shape
    assert isinstance(group.sf(60.0), float)
    one_by_one = [float(group.sf(t)) for t in np.ravel(x)]
    assert np.ravel(group.sf(x)) == pytest.approx(one_by_one)
