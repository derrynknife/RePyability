"""The long-run optimal replacement of a part over a calendar of
opportunities (``NonRepairable.optimal_calendar_policy``; #319, UC-13 of
``repyability/maintenance/README.md``): the calendar, and phase 1, dynamic
programming on a grid (the brief's section 3)."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import Calendar, CalendarPolicy, NonRepairable, Opportunity
from repyability.maintenance import _replacement
from repyability.maintenance._calendar import _Grid

LIFE = surv.Weibull.from_params([1000.0, 2.5])
BEARING = NonRepairable(LIFE)

# The plant: weekly stops at 3, quarterly shuts at 2, an annual one at 1,
# a planned replacement on any other day at 4, a failure at 20.
PLANT = Calendar(
    [
        Opportunity("weekly", 3.0, period=7.0),
        Opportunity("quarterly", 2.0, period=91.0),
        Opportunity("annual", 1.0, period=364.0),
    ],
    failure=20.0,
    anytime=4.0,
)


@pytest.fixture(scope="module")
def plant():
    return BEARING.optimal_calendar_policy(PLANT, step=1.0)


# The calendar.


def test_an_opportunity_on_a_period_or_dates():
    weekly = Opportunity("weekly", 3, period=7, offset=3)
    assert weekly.cost == 3.0 and weekly.period == 7.0
    assert weekly.times(0.0, 20.0).tolist() == [3.0, 10.0, 17.0]
    # Both ways from the offset.
    assert weekly.times(-10.0, 0.0).tolist() == [-4.0]
    plan = Opportunity("turnaround", 1.0, dates=[1130.0, 400.0, 400.0])
    assert plan.dates == (400.0, 1130.0)
    assert not plan.periodic


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"name": "", "cost": 1.0, "period": 7.0}, TypeError, "name"),
        ({"name": "anytime", "cost": 1.0, "period": 7.0}, ValueError, "any"),
        ({"name": "a", "cost": -1.0, "period": 7.0}, ValueError, "cost"),
        ({"name": "a", "cost": True, "period": 7.0}, ValueError, "cost"),
        ({"name": "a", "cost": 1.0}, ValueError, "one of"),
        (
            {"name": "a", "cost": 1.0, "period": 7.0, "dates": [1.0]},
            ValueError,
            "one of",
        ),
        ({"name": "a", "cost": 1.0, "period": 0.0}, ValueError, "period"),
        ({"name": "a", "cost": 1.0, "period": math.inf}, ValueError, "period"),
        (
            {"name": "a", "cost": 1.0, "period": 7.0, "offset": math.nan},
            ValueError,
            "offset",
        ),
        (
            {"name": "a", "cost": 1.0, "dates": [1.0], "offset": 2.0},
            ValueError,
            "offset",
        ),
        ({"name": "a", "cost": 1.0, "dates": []}, ValueError, "dates"),
        ({"name": "a", "cost": 1.0, "dates": [math.inf]}, ValueError, "dates"),
    ],
)
def test_an_opportunity_is_checked(kwargs, error, match):
    with pytest.raises(error, match=match):
        Opportunity(**kwargs)


def test_the_calendar_period_and_prices():
    assert PLANT.periodic and PLANT.period == 364.0
    assert PLANT.planned_cost([0.0, 1.0, 7.0, 91.0]).tolist() == [
        1.0,
        4.0,
        3.0,
        2.0,
    ]
    # Laid on a grid, an opportunity counts at the grid time nearest it.
    assert PLANT.planned_cost([7.4], step=1.0).tolist() == [3.0]
    assert PLANT.planned_cost([7.4]).tolist() == [4.0]
    leap = Calendar([Opportunity("a", 1.0, period=365.25)], 9.0, None)
    assert leap.period == 365.25
    mixed = Calendar(
        [
            Opportunity("w", 1.0, period=7.0),
            Opportunity("a", 1.0, period=365.25),
        ],
        9.0,
        None,
    )
    assert mixed.period == 10227.0
    plan = Calendar([Opportunity("t", 1.0, dates=[400.0, 1130.0])], 9.0)
    assert not plan.periodic and plan.period is None
    assert plan.plan_end == 1130.0
    assert Calendar(failure=9.0, anytime=None).planned_cost([5.0])[0] == (
        math.inf
    )


def test_the_calendar_is_checked():
    with pytest.raises(ValueError, match="differ"):
        Calendar(
            [
                Opportunity("a", 1.0, period=7.0),
                Opportunity("a", 2.0, period=9.0),
            ]
        )
    with pytest.raises(TypeError, match="Opportunity"):
        Calendar([("a", 1.0)])
    with pytest.raises(ValueError, match="failure"):
        Calendar(failure=-1.0)
    with pytest.raises(ValueError, match="anytime"):
        Calendar(anytime="cheap")
    # The prices taken from the part must be there.
    with pytest.raises(ValueError, match="failure price"):
        BEARING.optimal_calendar_policy(Calendar(), step=10.0)


def test_the_part_gives_the_failure_and_anytime_prices():
    unit = NonRepairable(LIFE, planned_cost=4.0, unplanned_cost=20.0)
    by_part = unit.optimal_calendar_policy(
        Calendar([Opportunity("annual", 1.0, period=364.0)]), step=7.0
    )
    given = BEARING.optimal_calendar_policy(
        Calendar([Opportunity("annual", 1.0, period=364.0)], 20.0, 4.0),
        step=7.0,
    )
    assert by_part.cost_rate == given.cost_rate


def test_dates_off_the_grid_are_moved_with_a_warning():
    calendar = Calendar([Opportunity("a", 1.0, period=365.25)], 20.0, None)
    with pytest.warns(UserWarning, match="nearest"):
        grid = _Grid(calendar, None, 1.0)
    # Each date to its nearest grid time: no drift over the cycle.
    costs, _ = grid.prices(0, 4 * 1461)
    assert np.flatnonzero(np.isfinite(costs))[:5].tolist() == [
        0,
        365,
        730,
        1096,
        1461,
    ]


def test_the_default_step_puts_the_opportunities_on_the_grid():
    step = _replacement._default_step(PLANT, 887.0)
    assert 7.0 / step == round(7.0 / step)
    assert step <= 887.0 / _replacement.STEPS_PER_LIFE * 1.0001


# Phase 1, test 1: with no opportunities and a planned replacement on any
# day at the planned price, it is age replacement.


def test_it_collapses_to_age_replacement():
    unit = NonRepairable(
        surv.Weibull.from_params([1000, 2.5]),
        planned_cost=1.0,
        unplanned_cost=5.0,
    )
    classical = unit.optimal_replacement_policy()
    policy = unit.optimal_calendar_policy(Calendar(), step=1.0)
    assert policy.method == "dp" and policy.basis == "exact"
    assert abs(policy._threshold(10.0) - classical.interval) <= 1.0
    assert policy.cost_rate == pytest.approx(classical.cost_rate, rel=1e-5)


# Test 2: relative value iteration agrees with differencing a long finite
# horizon over whole cycles.


def test_the_cycle_rate_is_the_long_horizons():
    ages = _replacement._Ages(BEARING, 1.0, None)
    grid = _Grid(PLANT, 4.0, 1.0)
    costs, _ = grid.prices(0, grid.cycle)
    cycle = _replacement._relative_values(costs, ages, 20.0)
    record, _, _, _ = _replacement._window(grid, 0, 40 * 364, ages, 20.0)
    differenced = _replacement._differenced(record, 364, 11 * 364, 1.0)
    assert differenced == pytest.approx(cycle.rate, rel=1e-6)
    assert cycle.low <= cycle.rate <= cycle.high


# Test 3: the policy simulated plainly costs g a unit of time.


def simulated(policy, calendar, step, years, n=4000, seed=0):
    """The mean cost a day of ``policy`` over ``years`` cycles of the
    plant, after five to settle, each unit's life drawn whole and a failed
    unit replaced at once; and its standard error."""
    rng = np.random.default_rng(seed)
    days = int(364 * (years + 5) / step)
    burn = int(364 * 5 / step)
    times = np.arange(days) * step
    prices = calendar.planned_cost(times, step=step)
    limits = policy._threshold(times)
    scale, shape = 1000.0, 2.5
    age = np.zeros(n)
    life = scale * rng.weibull(shape, n)
    cost = np.zeros(n)
    for d in range(days):
        counted = d >= burn
        # A planned replacement at the start of the step.
        if np.isfinite(prices[d]):
            replaced = age >= limits[d] - 1e-9
            if counted:
                cost[replaced] += prices[d]
            age[replaced] = 0.0
            life[replaced] = scale * rng.weibull(shape, replaced.sum())
        # Failures within the step, each replaced at once.
        end = age + step
        failing = life < end
        while failing.any():
            if counted:
                cost[failing] += calendar.failure
            age[failing] = age[failing] - life[failing]
            life[failing] = scale * rng.weibull(shape, failing.sum())
            failing = life < end
            end = age + step
            failing &= life < end
        age += step
    rate = cost / (364 * years)
    return rate.mean(), rate.std() / math.sqrt(n)


def test_the_simulated_policy_costs_its_rate(plant):
    mean, error = simulated(plant, PLANT, 1.0, 6)
    assert abs(mean - plant.cost_rate) < 2 * error


# What the policy does, by kind: its rates and costs.


def test_the_costs_by_kind_sum_to_the_rate(plant):
    assert sum(plant.cost_rates.values()) == pytest.approx(
        plant.cost_rate, rel=1e-9
    )
    for kind, rate in plant.replacement_rates.items():
        price = {"failure": 20.0, "anytime": 4.0}.get(kind)
        if price is None:
            price = {o.name: o.cost for o in PLANT.opportunities}[kind]
        assert plant.cost_rates[kind] == pytest.approx(rate * price)
    # At these prices the annual shut does all the planned work.
    year = {k: v * 364 for k, v in plant.replacement_rates.items()}
    assert year["annual"] == pytest.approx(0.9619, abs=1e-4)
    assert year["failure"] == pytest.approx(0.0783, abs=1e-4)
    assert year["weekly"] == year["quarterly"] == year["anytime"] == 0.0


def counted(policy, calendar, years=20, n=4000, seed=1):
    """Each kind's replacements a day, simulated plainly as ``simulated``
    does, after five years to settle."""
    rng = np.random.default_rng(seed)
    times = np.arange(364 * (years + 5), dtype=float)
    burn = 364 * 5
    prices = calendar.planned_cost(times, step=1.0)
    limits = policy._threshold(times)
    names = {o.cost: o.name for o in calendar.opportunities}
    names[calendar.anytime] = "anytime"
    age, life = np.zeros(n), 1000.0 * rng.weibull(2.5, n)
    count = dict.fromkeys(list(names.values()) + ["failure"], 0.0)
    for d, t in enumerate(times):
        replaced = age >= limits[d] - 1e-9
        if d >= burn and np.isfinite(prices[d]):
            count[names[prices[d]]] += replaced.sum()
        age[replaced] = 0.0
        life[replaced] = 1000.0 * rng.weibull(2.5, replaced.sum())
        failing = life < age + 1.0
        while failing.any():
            if d >= burn:
                count["failure"] += failing.sum()
            age[failing] -= life[failing]
            life[failing] = 1000.0 * rng.weibull(2.5, failing.sum())
            failing = life < age + 1.0
        age += 1.0
    return {k: v / (n * 364 * years) for k, v in count.items()}


def test_the_rates_by_kind_are_simulated():
    calendar = Calendar(
        [
            Opportunity("quarterly", 2.0, period=91.0),
            Opportunity("annual", 1.0, period=364.0),
        ],
        failure=60.0,
        anytime=4.0,
    )
    policy = BEARING.optimal_calendar_policy(calendar, step=1.0)
    simulated_rates = counted(policy, calendar)
    for kind, rate in policy.replacement_rates.items():
        assert simulated_rates[kind] == pytest.approx(rate, rel=0.03, abs=2e-6)
    # Dearer failures: the quarterly shuts are used too.
    assert policy.replacement_rates["quarterly"] > 0.0


def test_the_schedule_says_which_shuts_are_used(plant):
    by_kind = {}
    for _, kind, _, chance in plant.schedule():
        by_kind.setdefault(kind, []).append(chance)
    # The long-run chance of a replacement at each shut: the annual
    # shut's is its rate over the year; the others are never used.
    assert by_kind["annual"] == [
        pytest.approx(plant.replacement_rates["annual"] * 364, rel=1e-9)
    ]
    assert max(by_kind["quarterly"] + by_kind["weekly"]) == 0.0


def test_the_next_replacement_is_checked(plant):
    with pytest.raises(ValueError, match="age"):
        plant.next_replacement(-1.0)
    with pytest.raises(ValueError, match="age"):
        plant.next_replacement(True)
    with pytest.raises(ValueError, match="now"):
        plant.next_replacement(10.0, now=math.inf)


# Test 4: the phase effect, and the dearer day never used.


def test_a_shut_shortly_before_the_annual_waits_longer(plant):
    quarterly = {t: a for t, _, a, _ in plant.schedule(kinds=("quarterly",))}
    # The quarterly shut a quarter before the annual waits for it longer
    # than the one three quarters before it.
    assert quarterly[273.0] > quarterly[91.0]
    weekly = [a for t, _, a, _ in plant.schedule(kinds=("weekly",))]
    assert weekly[-1] > weekly[0]


def test_the_dearer_day_is_never_used(plant):
    # Any day at 4, a weekly stop at 3 within a week: no unit the policy
    # keeps reaches the age the dearer day would replace it at. The
    # oldest is one the annual shut kept, a year on.
    annual = plant._threshold(0.0)
    oldest = annual + 364.0
    days = np.arange(364.0)
    kinds = PLANT.planned_cost(days, step=1.0)
    anytime = days[kinds == 4.0]
    assert np.all(plant._threshold(anytime) > oldest)


# Test 5: skipping the annual shut for a young unit.


def test_a_young_unit_skips_the_annual_shut(plant):
    # 60 days old at the annual shut: kept to the next; 300: replaced.
    kept = plant.next_replacement(age=60.0, now=364.0)
    assert (kept.time, kept.kind, kept.price) == (728.0, "annual", 1.0)
    replaced = plant.next_replacement(age=300.0, now=364.0)
    assert (replaced.time, replaced.kind) == (364.0, "annual")
    # Just put in at the shut: not replaced again there.
    assert plant.next_replacement(age=0.0, now=364.0).time == 728.0
    # Put in on day 200 after a failure: 164 days old at the next shut.
    assert plant.next_replacement(age=100.0, now=300.0).time == 364.0


# Test 6: the policy does not depend on the horizon, but near its end.


def test_the_policy_does_not_depend_on_the_horizon():
    ages = _replacement._Ages(BEARING, 1.0, None)
    grid = _Grid(PLANT, 4.0, 1.0)
    third = slice(2 * 364, 3 * 364)

    def year_three(years):
        _, first, _, _ = _replacement._window(grid, 0, years * 364, ages, 20.0)
        return first[third]

    long = year_three(20)
    assert np.array_equal(year_three(10), long)
    assert np.array_equal(year_three(40), long)
    # Five years: the end reaches back into year three, by a few days.
    short = year_three(5)
    finite = (short >= 0) & (long >= 0)
    assert np.abs(short[finite] - long[finite]).max() <= 10


# Test 7: it beats the simple policies, over ten years of the plant.

TEN_YEARS = 3640.0


def test_it_beats_the_simple_policies(plant):
    def ten_years(rule):
        return BEARING.calendar_policy_cost(PLANT, rule, step=1.0) * TEN_YEARS

    age = NonRepairable(
        LIFE, planned_cost=4.0, unplanned_cost=20.0
    ).optimal_replacement_policy()
    optimal = plant.cost_rate * TEN_YEARS
    every_annual = ten_years({"annual": 0.0})
    any_day = ten_years({"anytime": age.interval})
    run_to_failure = ten_years({})
    assert optimal == pytest.approx(25.27, abs=0.01)
    assert every_annual == pytest.approx(25.48, abs=0.01)
    assert any_day == pytest.approx(50.41, abs=0.01)
    assert run_to_failure == pytest.approx(20.0 / 887.3 * TEN_YEARS, rel=1e-3)
    assert optimal < every_annual < any_day < run_to_failure
    # The optimal policy itself, as a rule, costs its rate.
    rule = {"annual": plant._threshold(0.0)}
    assert ten_years(rule) >= optimal


def test_running_to_failure_costs_the_failures():
    rate = BEARING.calendar_policy_cost(PLANT, {}, step=1.0)
    assert rate == pytest.approx(20.0 / LIFE.mean(), rel=1e-5)


# Test 8: a life whose replacing ages are not all those from the youngest.


def test_a_set_that_is_no_threshold_is_flagged():
    unit = NonRepairable(surv.LogNormal.from_params([6.0, 1.0]))
    calendar = Calendar([Opportunity("monthly", 1.0, period=30.0)], 20.0)
    with pytest.warns(UserWarning, match="youngest"):
        policy = unit.optimal_calendar_policy(
            calendar, step=1.0, max_age=20000.0
        )
    assert not policy.threshold_only
    assert np.isfinite(policy._threshold(0.0))


def test_a_threshold_is_not_flagged(plant):
    assert plant.threshold_only


# Plans of given dates.

PLAN = Calendar(
    [
        Opportunity("weekly", 5.0, period=7.0),
        Opportunity("turnaround", 1.0, dates=[400.0, 1130.0, 1860.0]),
    ],
    failure=40.0,
    anytime=None,
)


def test_a_repeated_plan_is_the_periodic_calendar():
    plan = Calendar(
        [Opportunity("t", 1.0, dates=[364.0, 728.0, 1092.0])], 20.0, None
    )
    annual = Calendar([Opportunity("t", 1.0, period=364.0)], 20.0, None)
    dated = BEARING.optimal_calendar_policy(plan, step=1.0)
    periodic = BEARING.optimal_calendar_policy(annual, step=1.0)
    assert dated.cost_rate == pytest.approx(periodic.cost_rate, rel=1e-12)
    assert dated.settings["repeat"] == 1092.0
    # Repeated forward only: nothing at 0, before the plan.
    assert dated._threshold(0.0) == math.inf
    assert dated._threshold(364.0) == periodic._threshold(364.0)


def test_a_plan_says_until_when_its_tail_does_not_matter():
    repeat = BEARING.optimal_calendar_policy(PLAN, step=1.0)
    anytime = BEARING.optimal_calendar_policy(PLAN, step=1.0, tail="anytime")
    assert repeat.determined_until == anytime.determined_until
    until = repeat.determined_until
    assert 0.0 < until <= PLAN.plan_end
    before = np.arange(0.0, until)
    assert np.array_equal(
        repeat._threshold(before), anytime._threshold(before)
    )
    assert repeat._threshold(until) != anytime._threshold(until)
    assert repeat.cost_rate == anytime.cost_rate
    assert [t for t, _, _, _ in repeat.schedule(kinds=("turnaround",))] == [
        400.0,
        1130.0,
        1860.0,
    ]


# The policy result.


def test_the_policy_reads_and_goes_through_json(plant):
    assert isinstance(plant, CalendarPolicy)
    assert plant["cost_rate"] == plant.cost_rate
    assert plant.period == 364.0
    with pytest.raises(AttributeError):
        plant.cost_rate = 0.0
    plain = json.loads(json.dumps(plant.to_dict()))
    assert plain["method"] == "dp" and plain["basis"] == "exact"
    thresholds = plant._threshold(np.array([0.0, 7.0, 3.0]))
    assert thresholds.shape == (3,)
    assert thresholds[0] == plant._threshold(0.0)
    # The thresholds repeat with the calendar.
    assert plant._threshold(364.0 * 7 + 91.0) == plant._threshold(91.0)


def test_the_schedule(plant):
    shuts = plant.schedule()
    assert len(shuts) == 52
    assert shuts[0][:3] == (0.0, "annual", plant._threshold(0.0))
    days = plant.schedule(kinds=("anytime",))
    assert len(days) == 364 - 52
    with pytest.raises(ValueError, match="Unknown"):
        plant.schedule(kinds=("monthly",))
    with pytest.raises(ValueError, match="until"):
        plant.schedule(until=-1.0)


def test_the_answer_is_the_same_in_hours():
    # The default step is sized from the life and the calendar, never in
    # absolute time.
    days = BEARING.optimal_calendar_policy(PLANT)
    hourly = NonRepairable(surv.Weibull.from_params([24000.0, 2.5]))
    hours = hourly.optimal_calendar_policy(
        Calendar(
            [
                Opportunity("weekly", 3.0, period=168.0),
                Opportunity("quarterly", 2.0, period=91 * 24.0),
                Opportunity("annual", 1.0, period=364 * 24.0),
            ],
            failure=20.0,
            anytime=4.0,
        )
    )
    assert hours.step == pytest.approx(days.step * 24, rel=1e-12)
    assert hours.cost_rate == pytest.approx(days.cost_rate / 24, rel=1e-9)
    assert hours._threshold(91 * 24.0) == pytest.approx(
        days._threshold(91.0) * 24, rel=1e-9
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"method": "fast"}, ValueError, "method"),
        ({"tail": "loop"}, ValueError, "tail"),
        ({"step": 0.0}, ValueError, "step"),
        ({"step": "1"}, ValueError, "step"),
        ({"max_age": -1.0}, ValueError, "max_age"),
        ({"method": "sampling", "mc_samples": 1}, ValueError, "mc_samples"),
    ],
)
def test_the_arguments_are_checked(kwargs, error, match):
    with pytest.raises(error, match=match):
        BEARING.optimal_calendar_policy(PLANT, **kwargs)
    with pytest.raises(TypeError, match="Calendar"):
        BEARING.optimal_calendar_policy([Opportunity("a", 1.0, period=7.0)])


def test_a_rule_is_checked():
    with pytest.raises(ValueError, match="Unknown kind"):
        BEARING.calendar_policy_cost(PLANT, {"monthly": 0.0}, step=7.0)
    with pytest.raises(ValueError, match="at least 0"):
        BEARING.calendar_policy_cost(PLANT, {"annual": -1.0}, step=7.0)
    with pytest.raises(TypeError, match="dict"):
        BEARING.calendar_policy_cost(PLANT, [0.0], step=7.0)
    shut_only = Calendar(PLANT.opportunities, 20.0, None)
    with pytest.raises(ValueError, match="any day"):
        BEARING.calendar_policy_cost(shut_only, {"anytime": 1.0}, step=7.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert BEARING.calendar_policy_cost(
            shut_only, {"annual": math.inf}, step=7.0
        ) == pytest.approx(20.0 / LIFE.mean(), rel=1e-4)
