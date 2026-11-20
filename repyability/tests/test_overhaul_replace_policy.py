"""The repair, overhaul and replace policy with cycles
(``Repairable.overhaul_replace_cost_rate`` and
``optimal_overhaul_replace_policy``): checked against closed forms, the
periodic replacement it reduces to, a seeded simulation of failure
times that does not use the formula's expected counts, and the plans it
is compared with (overhaul only, replace only, a failure limit), each
worked out independently."""

import itertools
import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.special import gammaln
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
    replace = policy.alternatives["replace only"]
    assert "overhaul and replace" not in policy.alternatives
    assert replace["cycles"] == 1
    assert replace["interval"] == pytest.approx(classic.interval, rel=1e-6)
    assert replace["cost_rate"] == pytest.approx(classic.cost_rate, rel=1e-9)
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
    found = best.alternatives["overhaul and replace"]
    assert found["cost_rate"] == pytest.approx(
        part.overhaul_replace_cost_rate(
            found["interval"], found["cycles"], f, cp
        ),
        rel=1e-12,
    )


def test_a_unit_that_does_not_wear_out_is_never_overhauled():
    policy = unit(100.0, 1.0, 1.0, 5.0).optimal_overhaul_replace_policy(
        0.5, 20.0
    )
    assert policy.interval == np.inf and policy.cycles == 1
    assert policy.plan == "run to failure"
    assert policy.cost_rate == pytest.approx(0.01, rel=1e-3)


def test_an_overhaul_that_renews_is_the_as_new_overhaul_policy():
    # Factor 0 and an overhaul cheaper than a replacement: replacing never
    # pays, and overhauling only is the periodic overhaul policy at co.
    policy = unit().optimal_overhaul_replace_policy(0.0, 40.0, max_cycles=15)
    periodic = unit().optimal_overhaul_policy()
    assert policy.plan == "overhaul only" and policy.cycles == np.inf
    assert policy.interval == pytest.approx(periodic.interval, rel=1e-6)
    assert policy.cost_rate == pytest.approx(periodic.cost_rate, rel=1e-9)
    # The cycles of overhaul and replace run to the limit, and more
    # periods spread the replacement thinner, towards it from above.
    k_cycle = policy.alternatives["overhaul and replace"]
    assert k_cycle["cycles"] == 15 and policy.cycles_limit_reached
    longer = unit().optimal_overhaul_replace_policy(0.0, 40.0, max_cycles=60)
    assert (
        periodic.cost_rate
        < longer.alternatives["overhaul and replace"]["cost_rate"]
        < k_cycle["cost_rate"]
    )
    assert policy.saving == pytest.approx(
        min(
            v["cost_rate"]
            for k, v in policy.alternatives.items()
            if k != "overhaul only"
        )
        - policy.cost_rate,
        rel=1e-12,
    )


def test_many_periods_tend_to_never_replacing():
    # g(T, K) falls towards g_inf(T) = (cr [L(v* + T) - L(v*)] + co) / T,
    # v* = f T / (1 - f), as K grows: by hand for L(a) = (a / 100)^2.
    part = unit()
    f, t, cp = 0.6, 80.0, 20.0
    v = f * t / (1 - f)
    g_inf = (((v + t) / 100.0) ** 2 - (v / 100.0) ** 2 + 5.0) / t
    rates = [part.overhaul_replace_cost_rate(t, k, f, cp) for k in (10, 100)]
    assert rates[0] > rates[1] > g_inf
    assert part.overhaul_replace_cost_rate(t, 100_000, f, cp) == (
        pytest.approx(g_inf, rel=1e-4)
    )
    policy = part.optimal_overhaul_replace_policy(f, cp)
    best = policy.alternatives["overhaul only"]
    v = f * best["interval"] / (1 - f)
    assert best["cost_rate"] == pytest.approx(
        (((v + best["interval"]) / 100.0) ** 2 - (v / 100.0) ** 2 + 5.0)
        / best["interval"],
        rel=1e-12,
    )


def test_overhauling_for_ever_beats_the_old_answer():
    # The case the search over K alone got wrong: its best (overhaul every
    # 177.7, replace every second period, 0.1172) loses to overhauling
    # every 73.8 and never replacing (0.1129).
    part = unit(100.0, 2.5, 1.0, 5.0)
    policy = part.optimal_overhaul_replace_policy(0.6, 20.0, max_cycles=5)
    old = policy.alternatives["overhaul and replace"]
    assert (old["cycles"], round(old["interval"], 1)) == (2, 177.7)
    assert round(old["cost_rate"], 4) == 0.1172
    assert policy.plan == "overhaul only" and policy.cycles == np.inf
    assert policy.interval == pytest.approx(73.79, abs=0.01)
    assert policy.cost_rate == pytest.approx(0.11293, abs=1e-5)
    assert policy.failure_count is None
    assert not policy.cycles_limit_reached


def test_a_small_restoration_makes_replacing_at_a_failure_count_pay():
    part = unit(100.0, 2.5, 1.0, 5.0)
    policy = part.optimal_overhaul_replace_policy(0.9, 20.0, max_cycles=5)
    assert policy.plan == "failure limit"
    assert policy.failure_count == 13
    assert policy.interval == np.inf and policy.cycles == 1
    # By hand: (12 cr + cp) / E[T_13], E[T_n] = alpha Gamma(n + 1/b) /
    # Gamma(n) for a power law.
    e_t13 = 100.0 * math.gamma(13 + 1 / 2.5) / math.gamma(13)
    assert policy.cost_rate == pytest.approx(32.0 / e_t13, rel=1e-9)


def _independent_best(alpha, beta, cr, co, f, cp, max_cycles, max_failures):
    """The four plans' cost rates, each worked out without the optimiser:
    closed forms where there are, a fine grid of intervals otherwise."""
    t = np.logspace(0.0, 4.0, 40_001)

    def lam(a):
        return (a / alpha) ** beta

    # Periodic replacement of a power law: T* = alpha (cp / (cr (b-1)))^(1/b).
    t_star = alpha * (cp / (cr * (beta - 1.0))) ** (1.0 / beta)
    replace = (cr * lam(t_star) + cp) / t_star
    v = f * t / (1.0 - f)
    never = ((cr * (lam(v + t) - lam(v)) + co) / t).min()
    k_cycle = math.inf
    for k in range(2, max_cycles + 1):
        v, repairs = np.zeros_like(t), np.zeros_like(t)
        for _ in range(k):
            repairs += lam(v + t) - lam(v)
            v = f * (v + t)
        rate = (cr * repairs + (k - 1) * co + cp) / (k * t)
        k_cycle = min(k_cycle, rate.min())
    n = np.arange(1, max_failures + 1)
    e_tn = alpha * np.exp(gammaln(n + 1.0 / beta) - gammaln(n))
    limit = ((cr * (n - 1) + cp) / e_tn).min()
    return {
        "replace only": replace,
        "overhaul only": never,
        "overhaul and replace": k_cycle,
        "failure limit": limit,
    }


GRID = list(
    itertools.product(
        (1.5, 2.5, 3.5),
        (0.2, 0.5, 0.8),
        (10.0, 20.0, 50.0, 100.0),
        (2.0, 5.0, 10.0),
    )
)


def test_the_returned_plan_is_never_beaten_over_a_grid_of_cases():
    # 108 cases: shape 1.5-3.5, overhauls keeping 20-80% of the age, and a
    # range of overhaul and replacement costs.
    assert len(GRID) == 108
    plans = set()
    for beta, f, cp, co in GRID:
        part = unit(100.0, beta, 1.0, co)
        policy = part.optimal_overhaul_replace_policy(f, cp, max_cycles=6)
        others = _independent_best(100.0, beta, 1.0, co, f, cp, 6, 30)
        best = min(others.values())
        # Never beaten by any of the four (a grid can only overstate a
        # minimum), and no lower than the true one by more than the grid's
        # resolution.
        assert policy.cost_rate <= best * (1 + 1e-9)
        assert policy.cost_rate >= best * (1 - 1e-5)
        for plan, rate in others.items():
            assert policy.alternatives[plan]["cost_rate"] <= rate * (1 + 1e-9)
        plans.add(policy.plan)
    # Overhauling for ever and replacing at a failure count both win in
    # some cases, so the comparison matters.
    assert {"overhaul only", "failure limit"} <= plans


def test_the_result_is_plain_data():
    policy = unit().optimal_overhaul_replace_policy(0.5, 40.0)
    out = policy.to_dict()
    assert set(out) == {
        "interval",
        "cycles",
        "cost_rate",
        "plan",
        "failure_count",
        "alternatives",
        "saving",
        "cycles_limit_reached",
    }
    assert out["alternatives"]["overhaul only"]["cycles"] == math.inf
    json.dumps(out)


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
    with pytest.raises(ValueError, match="max_failures"):
        part.optimal_overhaul_replace_policy(0.5, 3.0, max_failures=0)
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
