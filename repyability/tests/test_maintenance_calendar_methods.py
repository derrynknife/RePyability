"""The calendar policy's other methods (#319, UC-13): integration between
opportunities in continuous time (phase 2, the brief's section 4),
sampling for a long cycle (phase 3, section 5), and how ``method="auto"``
chooses."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import Calendar, NonRepairable, Opportunity
from repyability.maintenance import _replacement

LIFE = surv.Weibull.from_params([1000.0, 2.5])
BEARING = NonRepairable(LIFE)

# Shuts only (no replacement on any day): quarterly at 3, annual at 1, a
# failure at 20.
SHUTS = Calendar(
    [
        Opportunity("quarterly", 3.0, period=91.0),
        Opportunity("annual", 1.0, period=364.0),
    ],
    failure=20.0,
    anytime=None,
)

# Weekly stops at 5 too, and a failure at 40: every kind is used.
NESTED = Calendar(
    [
        Opportunity("weekly", 5.0, period=7.0),
        Opportunity("quarterly", 3.0, period=91.0),
        Opportunity("annual", 1.0, period=364.0),
    ],
    failure=40.0,
    anytime=None,
)

# Co-prime periods: the calendar repeats only after 338,780,400 days.
COPRIME = Calendar(
    [
        Opportunity("weekly", 6.0, period=7.0, offset=3.0),
        Opportunity("p93", 4.0, period=93.0, offset=10.0),
        Opportunity("p400", 2.0, period=400.0, offset=50.0),
        Opportunity("p1301", 1.0, period=1301.0, offset=200.0),
    ],
    failure=40.0,
    anytime=None,
)


@pytest.fixture(scope="module")
def nested_exact():
    return BEARING.optimal_calendar_policy(NESTED, step=1.0)


# Phase 2, test 1: it agrees with the grid, ever closer as the grid's step
# shrinks.


def test_integration_agrees_with_the_grid():
    integrated = BEARING.optimal_calendar_policy(NESTED, method="integration")
    assert integrated.method == "integration" and integrated.step is None
    gaps = {}
    for step in (7.0, 1.0):
        grid = BEARING.optimal_calendar_policy(NESTED, step=step)
        gaps[step] = abs(grid.cost_rate - integrated.cost_rate)
        assert grid.cost_rate == pytest.approx(integrated.cost_rate, rel=1e-3)
    assert gaps[1.0] < gaps[7.0]
    grid = BEARING.optimal_calendar_policy(NESTED, step=1.0)
    for time, kind, age in integrated.schedule(kinds=("quarterly", "annual")):
        assert abs(grid.threshold(time) - age) <= 2.0, (time, kind)


# Test 2: opportunities every day at the planned price are age replacement.


def test_daily_opportunities_are_age_replacement():
    unit = NonRepairable(LIFE, planned_cost=1.0, unplanned_cost=5.0)
    classical = unit.optimal_replacement_policy()
    daily = Calendar([Opportunity("daily", 1.0, period=1.0)], anytime=None)
    policy = unit.optimal_calendar_policy(daily, method="integration")
    # Deciding once a day costs a hair more than at any moment.
    assert policy.cost_rate == pytest.approx(classical.cost_rate, rel=1e-6)
    assert policy.cost_rate >= classical.cost_rate * (1 - 1e-9)
    assert abs(policy.threshold(0.0) - classical.interval) <= 1.0


# Test 3: it refuses a replacement on any day.


def test_integration_refuses_a_replacement_on_any_day():
    calendar = Calendar(SHUTS.opportunities, failure=20.0, anytime=4.0)
    with pytest.raises(NotImplementedError, match="method='dp'"):
        BEARING.optimal_calendar_policy(calendar, method="integration")


# Test 4: long gaps, an annual shut and a life of about three years.


def test_long_gaps_need_few_nodes():
    unit = NonRepairable(surv.Weibull.from_params([1100.0, 3.0]))
    annual = Calendar([Opportunity("annual", 1.0, period=365.0)], 10.0, None)
    integrated = unit.optimal_calendar_policy(annual, method="integration")
    grid = unit.optimal_calendar_policy(annual, step=1.0)
    assert integrated.settings["nodes"] < 365
    assert integrated.cost_rate == pytest.approx(grid.cost_rate, rel=1e-4)
    assert abs(integrated.threshold(0.0) - grid.threshold(0.0)) <= 1.0


def test_integration_repeats_a_plan():
    plan = Calendar(
        [Opportunity("t", 1.0, dates=[364.0, 728.0, 1092.0])], 20.0, None
    )
    annual = Calendar([Opportunity("t", 1.0, period=364.0)], 20.0, None)
    dated = BEARING.optimal_calendar_policy(plan, method="integration")
    periodic = BEARING.optimal_calendar_policy(annual, method="integration")
    assert dated.cost_rate == pytest.approx(periodic.cost_rate, rel=1e-9)
    assert dated.threshold(0.0) == math.inf
    assert dated.threshold(364.0) == pytest.approx(periodic.threshold(0.0))
    with pytest.raises(NotImplementedError, match="method='dp'"):
        BEARING.optimal_calendar_policy(
            plan, method="integration", tail="anytime"
        )


def test_integration_with_no_opportunities_runs_to_failure():
    policy = BEARING.optimal_calendar_policy(
        Calendar(failure=20.0, anytime=None), method="integration"
    )
    assert policy.cost_rate == pytest.approx(20.0 / LIFE.mean(), rel=1e-12)


# Phase 3, test 1: sampling reproduces the exact rate.


def test_sampling_reproduces_the_exact_rate(nested_exact):
    sampled = BEARING.optimal_calendar_policy(
        NESTED, step=1.0, method="sampling", mc_samples=32, seed=1
    )
    assert sampled.basis == "sampled"
    assert abs(sampled.cost_rate - nested_exact.cost_rate) < (
        2 * sampled.cost_rate_se
    )
    fewer = BEARING.optimal_calendar_policy(
        NESTED, step=1.0, method="sampling", mc_samples=8, seed=1
    )
    # The error falls as one over the root of the samples.
    assert 1.2 < fewer.cost_rate_se / sampled.cost_rate_se < 3.5


# Test 2: a co-prime calendar, in seconds a sample.


def test_a_coprime_calendar_is_sampled(monkeypatch):
    policy = BEARING.optimal_calendar_policy(
        COPRIME, step=1.0, mc_samples=8, seed=3
    )
    assert policy.method == "sampling"
    assert np.isfinite(policy.cost_rate) and policy.cost_rate_se > 0.0
    assert policy.period is None
    # A window twice as long changes the rate by less than its error.
    monkeypatch.setattr(
        _replacement, "SETTLE_SPANS", 2 * _replacement.SETTLE_SPANS
    )
    longer = BEARING.optimal_calendar_policy(
        COPRIME, step=1.0, mc_samples=8, seed=3
    )
    assert abs(longer.cost_rate - policy.cost_rate) < policy.cost_rate_se


# Test 3: the policy itself is never sampled.


def test_the_sampled_policy_is_deterministic():
    one = BEARING.optimal_calendar_policy(
        COPRIME, step=1.0, mc_samples=2, seed=1
    )
    other = BEARING.optimal_calendar_policy(
        COPRIME, step=1.0, mc_samples=2, seed=2
    )
    times = np.array([10.0, 50.0, 200.0, 1450.0])
    assert np.array_equal(one.threshold(times), other.threshold(times))
    assert np.array_equal(one.threshold(times), one.threshold(times))
    shuts = one.schedule(until=500.0, kinds=("p400", "p1301"))
    assert [(t, k) for t, k, _ in shuts] == [
        (50.0, "p400"),
        (200.0, "p1301"),
        (450.0, "p400"),
    ]
    for time, _, age in shuts:
        assert one.threshold(time) == age
    with pytest.raises(ValueError, match="until"):
        one.schedule()


# Test 4: which method "auto" takes.


def test_auto_chooses_the_method(nested_exact):
    assert nested_exact.method == "dp" and nested_exact.basis == "exact"
    sampled = BEARING.optimal_calendar_policy(
        COPRIME, step=1.0, mc_samples=2, seed=0
    )
    assert sampled.method == "sampling"
    plan = Calendar([Opportunity("t", 1.0, dates=[400.0, 1130.0])], 20.0, None)
    assert BEARING.optimal_calendar_policy(plan, step=10.0).method == "dp"
    # Never integration, which is asked for.
    assert all(
        BEARING.optimal_calendar_policy(c, step=step, mc_samples=2).method
        != "integration"
        for c, step in ((SHUTS, 7.0), (plan, 10.0))
    )
    with pytest.raises(NotImplementedError, match="method='dp'"):
        BEARING.optimal_calendar_policy(plan, method="sampling")


def test_dp_on_a_long_cycle_differences_a_long_horizon():
    policy = BEARING.optimal_calendar_policy(COPRIME, step=1.0, method="dp")
    assert policy.basis == "differenced"
    sampled = BEARING.optimal_calendar_policy(
        COPRIME, step=1.0, mc_samples=16, seed=5
    )
    assert abs(policy.cost_rate - sampled.cost_rate) < 4 * sampled.cost_rate_se
    times = np.array([10.0, 200.0, 450.0])
    assert np.array_equal(policy.threshold(times), sampled.threshold(times))
