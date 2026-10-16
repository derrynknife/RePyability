"""The calendar policy's other methods (#319, UC-13): integration between
opportunities in continuous time (phase 2, the brief's section 4),
sampling for a long cycle (phase 3, section 5), and how ``method="auto"``
chooses."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import Calendar, NonRepairable, Opportunity
from repyability.maintenance import _replacement


def solve(part, calendar, **options):
    """The calendar policy, on a grid of a given step where a check needs
    one (the public method sizes its own)."""
    return _replacement.optimal_calendar_policy(part, calendar, **options)


def price(part, calendar, rule, **options):
    """A rule's cost rate, likewise."""
    return _replacement.calendar_policy_cost(part, calendar, rule, **options)


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
    return solve(BEARING, NESTED, step=1.0)


# Phase 2, test 1: it agrees with the grid, ever closer as the grid's step
# shrinks.


def test_integration_agrees_with_the_grid():
    integrated = solve(BEARING, NESTED, method="integration")
    assert integrated.method == "integration" and integrated.step is None
    gaps = {}
    for step in (7.0, 1.0):
        grid = solve(BEARING, NESTED, step=step)
        gaps[step] = abs(grid.cost_rate - integrated.cost_rate)
        assert grid.cost_rate == pytest.approx(integrated.cost_rate, rel=1e-3)
    assert gaps[1.0] < gaps[7.0]
    grid = solve(BEARING, NESTED, step=1.0)
    for time, kind, age, _ in integrated.schedule(
        kinds=("quarterly", "annual")
    ):
        assert abs(grid._threshold(time) - age) <= 2.0, (time, kind)


# Test 2: opportunities every day at the planned price are age replacement.


def test_daily_opportunities_are_age_replacement():
    unit = NonRepairable(LIFE, planned_cost=1.0, unplanned_cost=5.0)
    classical = unit.optimal_replacement_policy()
    daily = Calendar([Opportunity("daily", 1.0, period=1.0)], anytime=None)
    policy = solve(unit, daily, method="integration")
    # Deciding once a day costs a hair more than at any moment.
    assert policy.cost_rate == pytest.approx(classical.cost_rate, rel=1e-6)
    assert policy.cost_rate >= classical.cost_rate * (1 - 1e-9)
    assert abs(policy._threshold(0.0) - classical.interval) <= 1.0


# Test 3: it refuses a replacement on any day.


def test_integration_refuses_a_replacement_on_any_day():
    calendar = Calendar(SHUTS.opportunities, failure=20.0, anytime=4.0)
    with pytest.raises(NotImplementedError, match="method='dp'"):
        solve(BEARING, calendar, method="integration")


# Test 4: long gaps, an annual shut and a life of about three years.


def test_long_gaps_need_few_nodes():
    unit = NonRepairable(surv.Weibull.from_params([1100.0, 3.0]))
    annual = Calendar([Opportunity("annual", 1.0, period=365.0)], 10.0, None)
    integrated = solve(unit, annual, method="integration")
    grid = solve(unit, annual, step=1.0)
    assert integrated.settings["nodes"] < 365
    assert integrated.cost_rate == pytest.approx(grid.cost_rate, rel=1e-4)
    assert abs(integrated._threshold(0.0) - grid._threshold(0.0)) <= 1.0


def test_integration_repeats_a_plan():
    plan = Calendar(
        [Opportunity("t", 1.0, dates=[364.0, 728.0, 1092.0])], 20.0, None
    )
    annual = Calendar([Opportunity("t", 1.0, period=364.0)], 20.0, None)
    dated = solve(BEARING, plan, method="integration")
    periodic = solve(BEARING, annual, method="integration")
    assert dated.cost_rate == pytest.approx(periodic.cost_rate, rel=1e-9)
    assert dated._threshold(0.0) == math.inf
    assert dated._threshold(364.0) == pytest.approx(periodic._threshold(0.0))
    with pytest.raises(NotImplementedError, match="method='dp'"):
        solve(BEARING, plan, method="integration", tail="anytime")


def test_integration_refuses_a_calendar_with_no_planned_replacement():
    # No opportunities and none on any day: nothing to find but running
    # to failure, refused as the other methods refuse it.
    with pytest.raises(ValueError, match="no planned replacement"):
        solve(
            BEARING, Calendar(failure=20.0, anytime=None), method="integration"
        )


# Phase 3, test 1: sampling reproduces the exact rate.


def test_sampling_reproduces_the_exact_rate(nested_exact):
    sampled = solve(
        BEARING, NESTED, step=1.0, method="sampling", mc_samples=32, seed=1
    )
    assert sampled.basis == "sampled"
    assert abs(sampled.cost_rate - nested_exact.cost_rate) < (
        2 * sampled.cost_rate_se
    )
    fewer = solve(
        BEARING, NESTED, step=1.0, method="sampling", mc_samples=8, seed=1
    )
    # The error falls as one over the root of the samples.
    assert 1.2 < fewer.cost_rate_se / sampled.cost_rate_se < 3.5


# Test 2: a co-prime calendar, in seconds a sample.


def test_a_coprime_calendar_is_sampled(monkeypatch):
    policy = solve(BEARING, COPRIME, step=1.0, mc_samples=8, seed=3)
    assert policy.method == "sampling"
    assert np.isfinite(policy.cost_rate) and policy.cost_rate_se > 0.0
    assert policy.period is None
    # A window twice as long changes the rate by less than its error.
    monkeypatch.setattr(
        _replacement, "SETTLE_SPANS", 2 * _replacement.SETTLE_SPANS
    )
    longer = solve(BEARING, COPRIME, step=1.0, mc_samples=8, seed=3)
    assert abs(longer.cost_rate - policy.cost_rate) < policy.cost_rate_se


# Test 3: the policy itself is never sampled.


def test_the_sampled_policy_is_deterministic():
    one = solve(BEARING, COPRIME, step=1.0, mc_samples=2, seed=1)
    other = solve(BEARING, COPRIME, step=1.0, mc_samples=2, seed=2)
    times = np.array([10.0, 50.0, 200.0, 1450.0])
    assert np.array_equal(one._threshold(times), other._threshold(times))
    assert np.array_equal(one._threshold(times), one._threshold(times))
    shuts = one.schedule(until=500.0, kinds=("p400", "p1301"))
    assert [(t, k) for t, k, _, _ in shuts] == [
        (50.0, "p400"),
        (200.0, "p1301"),
        (450.0, "p400"),
    ]
    for time, _, age, _ in shuts:
        assert one._threshold(time) == age
    with pytest.raises(ValueError, match="until"):
        one.schedule()


# Test 4: which method "auto" takes.


def test_auto_chooses_the_method(nested_exact):
    assert nested_exact.method == "dp" and nested_exact.basis == "exact"
    # Falling to sampling is said: the cycle is too long, and mc_samples
    # sets the cost.
    with pytest.warns(UserWarning, match="mc_samples") as record:
        sampled = solve(BEARING, COPRIME, step=1.0, mc_samples=2, seed=0)
    assert sampled.method == "sampling"
    assert f"more than {_replacement.MAX_CYCLE:,}" in str(record[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asked = solve(
            BEARING, COPRIME, step=1.0, method="sampling", mc_samples=2, seed=0
        )
    assert asked.cost_rate == sampled.cost_rate
    plan = Calendar([Opportunity("t", 1.0, dates=[400.0, 1130.0])], 20.0, None)
    assert solve(BEARING, plan, step=10.0).method == "dp"
    # Never integration, which is asked for.
    assert all(
        solve(BEARING, c, step=step, mc_samples=2).method != "integration"
        for c, step in ((SHUTS, 7.0), (plan, 10.0))
    )
    with pytest.raises(NotImplementedError, match="method='dp'"):
        solve(BEARING, plan, method="sampling")


def test_dp_on_a_long_cycle_differences_a_long_horizon():
    policy = solve(BEARING, COPRIME, step=1.0, method="dp")
    assert policy.basis == "differenced"
    sampled = solve(BEARING, COPRIME, step=1.0, mc_samples=16, seed=5)
    assert abs(policy.cost_rate - sampled.cost_rate) < 4 * sampled.cost_rate_se
    times = np.array([10.0, 200.0, 450.0])
    assert np.array_equal(policy._threshold(times), sampled._threshold(times))
