"""The repair, overhaul and replace policy with cycles
(``Repairable.overhaul_replace_cost_rate`` and
``optimal_overhaul_replace_policy``): checked against closed forms, the
periodic replacement it reduces to, and a seeded simulation of failure
times that does not use the formula's expected counts."""

import numpy as np
import pytest
import surpyval as surv
from surpyval.recurrent import CrowAMSAA, GeneralizedRenewal

from repyability import OverhaulReplacePolicy, Repairable


def unit(alpha=100.0, beta=2.0, cr=1.0, co=5.0):
    out = Repairable(CrowAMSAA.from_params([alpha, beta]))
    out.set_repair_and_overhaul_costs(cr=cr, co=co)
    return out


def test_with_a_constant_failure_rate_ages_do_not_matter():
    # cr / alpha + ((K - 1) co + cp) / (K T), by hand: 0.01 + 50 / 100.
    rate = unit(100.0, 1.0, 1.0, 10.0).overhaul_replace_cost_rate(
        50.0, 2, 0.5, 40.0
    )
    assert rate == pytest.approx(0.51, rel=1e-12)


def test_one_cycle_is_periodic_replacement_at_the_replacement_cost():
    part = unit(co=40.0)
    policy = part.optimal_overhaul_replace_policy(0.5, 40.0, max_cycles=1)
    classic = part.optimal_overhaul_policy()
    assert policy.cycles == 1
    assert policy.interval == pytest.approx(classic.interval, rel=1e-6)
    assert policy.cost_rate == pytest.approx(classic.cost_rate, rel=1e-9)
    # By hand for a power law: T* = alpha (cp / (cr (b - 1)))^(1/b).
    assert classic.interval == pytest.approx(
        100.0 * (40.0 / 1.0) ** 0.5, rel=1e-6
    )


def test_a_full_renewal_overhaul_repeats_the_same_period():
    # With factor 0 every period starts new: K Lambda(T) repairs.
    part = unit()
    t, k = 80.0, 3
    by_hand = (k * (t / 100.0) ** 2 + (k - 1) * 5.0 + 30.0) / (k * t)
    assert part.overhaul_replace_cost_rate(t, k, 0.0, 30.0) == (
        pytest.approx(by_hand, rel=1e-12)
    )


def test_a_hand_worked_two_period_cycle():
    # Lambda = (a / 100)^2; f = 0.5, T = 100: v1 = 0.5 * 100 = 50, so
    # period 1 expects 1 failure, period 2 expects (150/100)^2 - 0.25 = 2.
    part = unit()
    rate = part.overhaul_replace_cost_rate(100.0, 2, 0.5, 20.0)
    assert rate == pytest.approx((3.0 + 5.0 + 20.0) / 200.0, rel=1e-12)


@pytest.mark.parametrize(
    "beta, f, k, t",
    [(2.0, 0.5, 3, 90.0), (1.5, 0.8, 4, 60.0), (3.0, 0.0, 2, 120.0)],
)
def test_the_rate_agrees_with_a_simulation_of_failure_times(beta, f, k, t):
    alpha, cr, co, cp = 100.0, 1.0, 5.0, 20.0
    part = unit(alpha, beta, cr, co)
    rng = np.random.default_rng(12345)
    cycles = 100_000
    repairs = np.zeros(cycles)
    v = 0.0
    for _ in range(k):
        # Failure ages are the inverse of Lambda at unit-rate arrivals
        # from Lambda(v): counted by walking the arrivals.
        start = (v / alpha) ** beta
        end = ((v + t) / alpha) ** beta
        y = start + np.cumsum(rng.exponential(size=(cycles, 60)), axis=1)
        repairs += (y < end).sum(axis=1)
        v = f * (v + t)
    cost = cr * repairs + (k - 1) * co + cp
    estimate = cost.mean() / (k * t)
    error = cost.std(ddof=1) / np.sqrt(cycles) / (k * t)
    exact = part.overhaul_replace_cost_rate(t, k, f, cp)
    assert abs(estimate - exact) < 4 * error


def test_the_best_policy_beats_every_other_choice():
    part = unit(100.0, 2.5, 1.0, 5.0)
    f, cp = 0.6, 20.0
    best = part.optimal_overhaul_replace_policy(f, cp, max_cycles=12)
    assert isinstance(best, OverhaulReplacePolicy)
    grid = np.linspace(20.0, 600.0, 400)
    for k in range(1, 13):
        assert best.cost_rate <= part.overhaul_replace_cost_rate(
            grid, k, f, cp
        ).min() * (1 + 1e-9)
    assert best.cost_rate == pytest.approx(
        part.overhaul_replace_cost_rate(best.interval, best.cycles, f, cp),
        rel=1e-12,
    )


def test_a_unit_that_does_not_wear_out_is_never_overhauled():
    policy = unit(100.0, 1.0, 1.0, 5.0).optimal_overhaul_replace_policy(
        0.5, 20.0
    )
    assert policy.interval == np.inf and policy.cycles == 1
    assert policy.cost_rate == pytest.approx(0.01, rel=1e-3)


def test_a_cheap_overhaul_that_renews_is_never_followed_by_a_replacement():
    # Factor 0 and an overhaul cheaper than a replacement: replacing never
    # pays, and the cycles run to the limit.
    policy = unit().optimal_overhaul_replace_policy(0.0, 40.0, max_cycles=15)
    assert policy.cycles == 15
    periodic = unit().optimal_overhaul_policy()
    assert policy.cost_rate > periodic.cost_rate
    # More periods spread the replacement thinner, towards the periodic
    # overhaul's rate from above.
    longer = unit().optimal_overhaul_replace_policy(0.0, 40.0, max_cycles=60)
    assert periodic.cost_rate < longer.cost_rate < policy.cost_rate


def test_the_result_is_plain_data():
    policy = unit().optimal_overhaul_replace_policy(0.5, 40.0)
    assert set(policy.to_dict()) == {"interval", "cycles", "cost_rate"}


def test_inputs_are_checked():
    part = unit()
    with pytest.raises(ValueError, match="overhaul_factor"):
        part.overhaul_replace_cost_rate(10.0, 2, 1.0, 40.0)
    with pytest.raises(ValueError, match="cycles"):
        part.overhaul_replace_cost_rate(10.0, 0, 0.5, 40.0)
    with pytest.raises(ValueError, match="interval"):
        part.overhaul_replace_cost_rate(-1.0, 2, 0.5, 40.0)
    with pytest.raises(ValueError, match="replacement_cost"):
        part.optimal_overhaul_replace_policy(0.5, -3.0)
    with pytest.raises(ValueError, match="max_cycles"):
        part.optimal_overhaul_replace_policy(0.5, 3.0, max_cycles=0)
    bare = Repairable(CrowAMSAA.from_params([100.0, 2.0]))
    with pytest.raises(ValueError, match="costs not set"):
        bare.optimal_overhaul_replace_policy(0.5, 3.0)


def test_a_simulation_backed_model_is_refused():
    grp = GeneralizedRenewal.fit_from_parameters(
        [100.0, 2.0], 0.5, kijima="i", dist=surv.Weibull
    )
    part = Repairable(grp)
    part.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
    with pytest.raises(NotImplementedError, match="analytic"):
        part.optimal_overhaul_replace_policy(0.5, 20.0)
