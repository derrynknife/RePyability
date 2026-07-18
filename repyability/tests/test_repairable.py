"""Tests for Repairable: minimal-repair ("as bad as old") components and
the optimal-overhaul-interval (Barlow-Hunter) policy.

For a power-law cumulative intensity Lambda(t) = (t/alpha)^beta (surpyval
Crow-AMSAA), the cost rate g(t) = (cr*Lambda(t) + co)/t has the closed-form
minimiser

    t* = alpha * (co / (cr * (beta - 1)))^(1/beta),    beta > 1

with g(t*) = co * beta / ((beta - 1) * t*). The numeric optimiser is held
to this closed form; beta <= 1 (no wear-out) must give an infinite
interval.
Reference: https://reliawiki.org/index.php/Repairable_Systems_Analysis
"""

import numpy as np
import pytest
import surpyval as surv

from repyability import MaintenancePolicy, Repairable


def closed_form_interval(alpha, beta, cr, co):
    return alpha * (co / (cr * (beta - 1.0))) ** (1.0 / beta)


def closed_form_cost_rate(alpha, beta, cr, co):
    t_star = closed_form_interval(alpha, beta, cr, co)
    return co * beta / ((beta - 1.0) * t_star)


def test_cost_and_cost_rate_values():
    alpha, beta = 100.0, 1.5
    rep = Repairable(surv.CrowAMSAA.from_params([alpha, beta]))
    rep.set_repair_and_overhaul_costs(10.0, 1000.0)

    t = 250.0
    lam = (t / alpha) ** beta
    assert rep.cost(t) == pytest.approx(10.0 * lam + 1000.0)
    assert rep.cost_rate(t) == pytest.approx((10.0 * lam + 1000.0) / t)

    # Numpy-style return contract: scalar in -> float, array in -> array.
    assert isinstance(rep.cost_rate(t), float)
    arr = rep.cost_rate(np.array([100.0, 250.0]))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)


def test_optimal_interval_matches_closed_form():
    # The second case is the Crow-AMSAA equivalent of the ReliaWiki
    # repairable-systems example (lambda = 2.1211e-5, beta = 1.4738, in
    # the Lambda(t) = lambda*t^beta parameterisation, i.e.
    # alpha = lambda^(-1/beta) ~ 1481.6).
    cases = [
        (100.0, 1.5, 10.0, 1000.0),
        (1481.6, 1.4738, 100.0, 5000.0),
        (50.0, 3.0, 1.0, 40.0),
        (2000.0, 2.2, 250.0, 60_000.0),
    ]
    for alpha, beta, cr, co in cases:
        rep = Repairable(surv.CrowAMSAA.from_params([alpha, beta]))
        rep.set_repair_and_overhaul_costs(cr, co)
        t_star = rep.find_optimal_overhaul_interval()
        assert t_star == pytest.approx(
            closed_form_interval(alpha, beta, cr, co), rel=1e-4
        )
        # And it is a genuine minimum: cheaper than nearby intervals.
        assert rep.cost_rate(t_star) <= rep.cost_rate(0.8 * t_star)
        assert rep.cost_rate(t_star) <= rep.cost_rate(1.25 * t_star)


def test_duane_model_closed_form():
    # Duane cif: Lambda(t) = b * t^a (surpyval params [a, b]), giving
    # t* = (co / (cr*b*(a-1)))^(1/a) for a > 1.
    a, b, cr, co = 1.8, 0.01, 20.0, 4000.0
    rep = Repairable(surv.Duane.from_params([a, b]))
    rep.set_repair_and_overhaul_costs(cr, co)
    expected = (co / (cr * b * (a - 1.0))) ** (1.0 / a)
    assert rep.find_optimal_overhaul_interval() == pytest.approx(
        expected, rel=1e-4
    )


def test_no_wear_out_never_overhauls():
    # beta = 1 is an HPP (constant failure intensity): an overhaul buys
    # nothing. beta < 1 (reliability growth) makes overhauls strictly
    # worse. Both must return an infinite interval.
    for beta in (1.0, 0.8):
        rep = Repairable(surv.CrowAMSAA.from_params([500.0, beta]))
        rep.set_repair_and_overhaul_costs(10.0, 1000.0)
        assert rep.find_optimal_overhaul_interval() == np.inf


def test_optimal_overhaul_policy():
    alpha, beta, cr, co = 100.0, 1.5, 10.0, 1000.0
    rep = Repairable(surv.CrowAMSAA.from_params([alpha, beta]))
    rep.set_repair_and_overhaul_costs(cr, co)
    policy = rep.optimal_overhaul_policy()
    assert isinstance(policy, MaintenancePolicy)
    assert policy.interval == pytest.approx(
        closed_form_interval(alpha, beta, cr, co), rel=1e-4
    )
    assert policy.cost_rate == pytest.approx(
        closed_form_cost_rate(alpha, beta, cr, co), rel=1e-4
    )


def test_policy_when_never_overhauling():
    # HPP-like unit: the limiting cost rate is minimal repairs at the
    # long-run rate of occurrence, cr * Lambda(t)/t = cr/alpha for
    # beta = 1.
    alpha, cr, co = 500.0, 10.0, 1000.0
    rep = Repairable(surv.CrowAMSAA.from_params([alpha, 1.0]))
    rep.set_repair_and_overhaul_costs(cr, co)
    policy = rep.optimal_overhaul_policy()
    assert policy.interval == np.inf
    assert policy.cost_rate == pytest.approx(cr / alpha, rel=1e-3)


def test_cost_validation():
    rep = Repairable(surv.CrowAMSAA.from_params([100.0, 1.5]))
    with pytest.raises(ValueError, match="less than overhaul"):
        rep.set_repair_and_overhaul_costs(10.0, 10.0)
    with pytest.raises(ValueError, match="positive"):
        rep.set_repair_and_overhaul_costs(0.0, 10.0)
    with pytest.raises(ValueError, match="costs not set"):
        rep.find_optimal_overhaul_interval()
    with pytest.raises(ValueError, match="costs not set"):
        rep.cost_rate(10.0)


def test_model_without_cif_raises():
    with pytest.raises(ValueError, match="cif"):
        Repairable(object())
