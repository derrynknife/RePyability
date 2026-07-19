"""
Tests the imperfect-repair (generalized-renewal / Kijima) support in the
``Repairable`` component: sourcing E[N(t)] from a simulation-backed
``GeneralizedRenewal`` model rather than an analytic ``cif``.

The exact anchors use the ``q = 1`` (minimal-repair) limit, where the
generalized-renewal process reduces to the baseline's cumulative hazard and the
optimal overhaul interval has a closed form.
"""

import numpy as np
import pytest
import surpyval as surv
from surpyval.recurrent import CrowAMSAA, GeneralizedRenewal

from repyability.maintenance import FailureLimitPolicy
from repyability.repairable import (
    Repairable,
    minimal_repair_time_to_nth_failure,
)


def _gr(q, alpha=100.0, beta=2.0, kijima="ii"):
    """A generalized-renewal model, Weibull baseline, restoration factor q."""
    return GeneralizedRenewal.fit_from_parameters(
        [alpha, beta], q, kijima=kijima, dist=surv.Weibull
    )


def test_is_simulated_flag():
    # A generalized-renewal (mcf-only) model is simulation-backed...
    assert Repairable(_gr(0.5)).is_simulated is True
    # ...while an analytic cumulative-intensity model (cif) is not.
    analytic = Repairable(CrowAMSAA.from_params([0.5, 1.6]))
    assert analytic.is_simulated is False


def test_requires_cif_or_mcf():
    class NoCounting:
        pass

    with pytest.raises(ValueError, match="cif|mcf"):
        Repairable(NoCounting())


def test_q1_expected_failures_matches_cumulative_hazard():
    """At q=1 the process is minimal repair, so E[N(t)] equals the baseline
    Weibull cumulative hazard ``(t/alpha)**beta``."""
    rep = Repairable(_gr(1.0, alpha=100.0, beta=2.0))
    t = np.array([50.0, 100.0, 150.0])
    enf = rep._expected_failures(t, seed=3, n_simulations=8000)
    expected = (t / 100.0) ** 2.0
    np.testing.assert_allclose(enf, expected, rtol=0.05)


def test_cost_and_cost_rate_contract():
    rep = Repairable(_gr(0.5))
    rep.set_repair_and_overhaul_costs(1.0, 5.0)
    assert isinstance(rep.cost(100.0, seed=1, n_simulations=300), float)
    assert isinstance(rep.cost_rate(100.0, seed=1, n_simulations=300), float)
    arr = rep.cost_rate(np.array([100.0, 200.0]), seed=1, n_simulations=300)
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)


def test_optimal_interval_matches_analytic_at_q1():
    """For minimal repair (q=1) with a Weibull baseline the optimal overhaul
    interval has the closed form ``alpha * (co/(cr*(beta-1))) ** (1/beta)``;
    the simulated optimum must land near it."""
    alpha, beta, cr, co = 100.0, 2.0, 1.0, 5.0
    analytic = alpha * (co / (cr * (beta - 1.0))) ** (1.0 / beta)
    rep = Repairable(_gr(1.0, alpha=alpha, beta=beta))
    rep.set_repair_and_overhaul_costs(cr, co)
    interval = rep.find_optimal_overhaul_interval(
        seed=2, n_simulations=1200, max_interval=600.0
    )
    assert interval == pytest.approx(analytic, rel=0.12)


def test_reproducible_with_seed():
    rep = Repairable(_gr(0.5))
    rep.set_repair_and_overhaul_costs(1.0, 5.0)
    p1 = rep.optimal_overhaul_policy(
        seed=7, n_simulations=400, max_interval=600.0
    )
    p2 = rep.optimal_overhaul_policy(
        seed=7, n_simulations=400, max_interval=600.0
    )
    assert p1.interval == p2.interval
    assert p1.cost_rate == p2.cost_rate


def test_costs_required_and_ordered():
    rep = Repairable(_gr(0.5))
    with pytest.raises(ValueError, match="costs not set"):
        rep.find_optimal_overhaul_interval(seed=1, n_simulations=200)
    with pytest.raises(ValueError, match="less than"):
        rep.set_repair_and_overhaul_costs(5.0, 1.0)


# -- Replace-at-N-th-failure policy ---------------------------------------


def test_minimal_repair_time_to_nth_failure_closed_form():
    # T_1 is just the baseline mean: alpha * Gamma(1 + 1/beta).
    from scipy.special import gamma

    assert minimal_repair_time_to_nth_failure(100.0, 2.0, 1) == pytest.approx(
        100.0 * gamma(1.5)
    )
    # A monotone increasing sequence.
    ts = [
        minimal_repair_time_to_nth_failure(100.0, 2.0, n) for n in range(1, 6)
    ]
    assert all(a < b for a, b in zip(ts, ts[1:]))
    with pytest.raises(ValueError):
        minimal_repair_time_to_nth_failure(100.0, 2.0, 0)


def test_expected_time_to_nth_failure_matches_closed_form_at_q1():
    rep = Repairable(_gr(1.0, alpha=100.0, beta=2.0))
    for n in (3, 6):
        sim = rep.expected_time_to_nth_failure(n, seed=1, n_simulations=5000)
        exact = minimal_repair_time_to_nth_failure(100.0, 2.0, n)
        assert sim == pytest.approx(exact, rel=0.05)


def test_expected_time_to_nth_failure_analytic_raises():
    analytic = Repairable(CrowAMSAA.from_params([0.5, 1.6]))
    with pytest.raises(ValueError, match="minimal_repair_time_to_nth_failure"):
        analytic.expected_time_to_nth_failure(3)


def test_expected_time_to_nth_failure_validation():
    rep = Repairable(_gr(0.5))
    with pytest.raises(ValueError, match="positive"):
        rep.expected_time_to_nth_failure(0, seed=1, n_simulations=200)


def test_optimal_failure_limit_policy():
    rep = Repairable(_gr(0.4, kijima="i"))
    rep.set_repair_and_overhaul_costs(1.0, 5.0)
    policy = rep.optimal_failure_limit_policy(
        seed=3, n_simulations=1200, max_failures=25
    )
    assert isinstance(policy, FailureLimitPolicy)
    assert policy.failure_count >= 1
    assert policy.cost_rate > 0
    # Reproducible for a fixed seed.
    again = rep.optimal_failure_limit_policy(
        seed=3, n_simulations=1200, max_failures=25
    )
    assert policy == again


def test_failure_limit_costs_required():
    rep = Repairable(_gr(0.5))
    with pytest.raises(ValueError, match="costs not set"):
        rep.find_optimal_replacement_failure_count(seed=1, n_simulations=200)
