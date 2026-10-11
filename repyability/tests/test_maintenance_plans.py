"""The NPV of skipping the next planned replacement right after a failure
(#319, UC-6 of ``repyability/maintenance/README.md``)."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import ReplacementPlan, SkipValue

LIFE = surv.Weibull.from_params([365.0, 2.0])
YEAR = 365.25


def per_day(rate):
    return math.log1p(rate) / YEAR


def weibull_lives(rng, size):
    return 365.0 * np.sqrt(-np.log(rng.random(size)))


def simulated_calendar(days, every, planned, unplanned, rate, n, seed):
    """The NPV of skipping on a calendar plan, simulated plainly: done, a
    new unit at ``days`` and its failures to ``days + every``; skipped,
    the failures of the unit in now (and its replacements) from ``days``
    to ``days + every``. The rest is the same both ways."""
    rng, r = np.random.default_rng(seed), per_day(rate)

    def failures(start, keep_from):
        time, value = np.full(n, start), np.zeros(n)
        while True:
            time = time + weibull_lives(rng, n)
            hit = time < days + every
            if not hit.any():
                return value
            counted = hit & (time >= keep_from)
            value[counted] += unplanned * np.exp(-r * time[counted])
            time = np.where(hit, time, np.inf)

    done = planned * math.exp(-r * days) + failures(days, days)
    skipped = failures(0.0, days)
    difference = done - skipped
    return difference.mean(), difference.std() / math.sqrt(n)


def simulated_resetting(every, planned, unplanned, rate, n, seed, years):
    """The NPV of skipping on a plan that resets, simulated plainly: the
    present value of each choice over ``years``, the first cycle cut off
    at ``every`` (done) or ``2 every`` (skipped)."""
    rng, r = np.random.default_rng(seed), per_day(rate)

    def value(first):
        time, total = np.zeros(n), np.zeros(n)
        cut = np.full(n, first)
        live = np.ones(n, dtype=bool)
        while live.any():
            life = weibull_lives(rng, live.sum())
            failed = life < cut[live]
            time[live] += np.minimum(life, cut[live])
            costs = np.where(failed, unplanned, planned)
            total[live] += costs * np.exp(-r * time[live])
            cut[live] = every
            live = time < years * YEAR
        return total

    done, skipped = value(every), value(2.0 * every)
    error = math.sqrt((done.var() + skipped.var()) / n)
    return (done - skipped).mean(), error


def test_the_bearing_on_a_plan_that_resets():
    # UC-6: every 180 days, 150 planned, 400 at a failure, 8% a year.
    plan = ReplacementPlan(LIFE, 180, 150, 400, resets=True)
    value = plan.skip_next(discount_rate=0.08)
    assert isinstance(value, SkipValue)
    assert value.days_to_next == 180.0
    assert value.npv == pytest.approx(26.558, abs=1e-3)
    mean, error = simulated_resetting(
        180.0, 150.0, 400.0, 0.08, 40_000, 3, 300
    )
    assert abs(value.npv - mean) < 4 * error


@pytest.mark.parametrize("days", [1.0, 30.0, 120.0, 180.0])
def test_the_bearing_on_a_calendar_plan(days):
    plan = ReplacementPlan(LIFE, 180, 150, 400, resets=False)
    value = plan.skip_next(0.08, days_to_next=days)
    mean, error = simulated_calendar(
        days, 180.0, 150.0, 400.0, 0.08, 300_000, int(days)
    )
    assert abs(value.npv - mean) < 4 * error


def test_the_calendar_answer_by_days_to_the_next_date():
    # On a 180-day plan, too frequent for the bearing (the best age is
    # near 298 days), skipping always pays, the more the sooner the next
    # date; on a 365-day plan it stops paying some 200 to 300 days out.
    short = ReplacementPlan(LIFE, 180, 150, 400, resets=False)
    values = [short.skip_next(0.08, d).npv for d in (1, 30, 90, 180)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(149.06, abs=0.01)
    assert values[-1] == pytest.approx(35.26, abs=0.01)
    yearly = ReplacementPlan(LIFE, 365, 150, 400, resets=False)
    assert yearly.skip_next(0.08, 10).npv > 0 > yearly.skip_next(0.08, 300).npv


def exponential_closed_forms(rate, every, planned, unplanned, r, days):
    """The two plans' NPVs for an exponential life, written out: its
    failures come at ``rate`` whatever its age."""
    total = r + rate

    def length(cut):
        return -math.expm1(-total * cut) / total

    def cost(cut):
        failed = rate * length(cut)
        return unplanned * failed + planned * math.exp(-total * cut)

    once, twice = length(every), length(2 * every)
    resetting = (
        cost(every) - cost(2 * every) + cost(every) * (twice - once) / once
    )

    def failures(t):
        # A new unit and its replacements fail at ``rate`` throughout.
        return unplanned * rate * (-math.expm1(-r * t) / r if r else t)

    later = math.exp(-r * days)
    calendar = (
        planned * later
        + later * failures(every)
        - (failures(days + every) - failures(days))
    )
    return resetting, calendar


@pytest.mark.parametrize("yearly", [0.0, 0.08, 0.5])
def test_an_exponential_life_against_its_closed_forms(yearly):
    rate, every, days = 1 / 200.0, 90.0, 40.0
    life = surv.Exponential.from_params([rate])
    resetting, calendar = exponential_closed_forms(
        rate, every, 150.0, 400.0, per_day(yearly), days
    )
    plan = ReplacementPlan(life, every, 150, 400, resets=True)
    assert plan.skip_next(yearly).npv == pytest.approx(resetting, rel=1e-8)
    plan = ReplacementPlan(life, every, 150, 400, resets=False)
    assert plan.skip_next(yearly, days).npv == pytest.approx(
        calendar, rel=1e-6
    )
    # With no ageing, a planned replacement only costs: skipping saves.
    assert resetting > 0 and calendar > 0


def test_with_no_discounting_a_resetting_plan_is_the_classic_comparison():
    # Undiscounted, skipping lengthens the first cycle: it saves its cost
    # difference, less the long-run cost rate over the time it adds.
    every, planned, unplanned = 180.0, 150.0, 400.0

    def mean_length(cut):
        return quad(lambda x: float(LIFE.sf(x)), 0, cut, epsrel=1e-13)[0]

    def cycle_cost(cut):
        kept = float(LIFE.sf(cut))
        return unplanned * (1 - kept) + planned * kept

    rate_of_cost = cycle_cost(every) / mean_length(every)
    classic = (
        cycle_cost(every)
        - cycle_cost(2 * every)
        + rate_of_cost * (mean_length(2 * every) - mean_length(every))
    )
    plan = ReplacementPlan(LIFE, every, planned, unplanned, resets=True)
    assert plan.skip_next(0.0).npv == pytest.approx(classic, rel=1e-9)
    # And a tiny rate is next to it.
    assert plan.skip_next(1e-9).npv == pytest.approx(classic, rel=1e-6)


def test_the_value_goes_through_json():
    plan = ReplacementPlan(LIFE, 180, 150, 400, resets=False)
    plain = json.loads(json.dumps(plan.skip_next(0.08, 30).to_dict()))
    assert plain["resets"] is False
    assert plain["days_to_next"] == 30.0
    assert plain["npv"] == pytest.approx(123.08, abs=0.01)


def test_days_to_the_next_date_go_with_a_calendar_plan_only():
    resetting = ReplacementPlan(LIFE, 180, 150, 400, resets=True)
    with pytest.raises(ValueError, match="give no days_to_next"):
        resetting.skip_next(0.08, days_to_next=30)
    calendar = ReplacementPlan(LIFE, 180, 150, 400, resets=False)
    with pytest.raises(ValueError, match="needs days_to_next"):
        calendar.skip_next(0.08)
    with pytest.raises(ValueError, match="at most every"):
        calendar.skip_next(0.08, days_to_next=181)
    for days in (0, -5, math.inf, True, "30"):
        with pytest.raises(ValueError, match="days_to_next"):
            calendar.skip_next(0.08, days_to_next=days)


@pytest.mark.parametrize("rate", [-0.01, math.inf, math.nan, True, "8%"])
def test_a_discount_rate_is_a_rate_a_year_at_least_0(rate):
    plan = ReplacementPlan(LIFE, 180, 150, 400, resets=True)
    with pytest.raises(ValueError, match="discount_rate"):
        plan.skip_next(rate)


@pytest.mark.parametrize(
    "arguments, error, match",
    [
        ((LIFE, 0, 150, 400), ValueError, "every"),
        ((LIFE, -1, 150, 400), ValueError, "every"),
        ((LIFE, 180, -1, 400), ValueError, "planned_cost"),
        ((LIFE, 180, 150, math.inf), ValueError, "unplanned_cost"),
        ((LIFE, 180, 150, "400"), ValueError, "unplanned_cost"),
        ((surv.Weibull, 180, 150, 400), TypeError, "life"),
        ((None, 180, 150, 400), TypeError, "life"),
    ],
)
def test_a_plan_is_checked(arguments, error, match):
    with pytest.raises(error, match=match):
        ReplacementPlan(*arguments, resets=True)


def test_resets_is_true_or_false():
    with pytest.raises(TypeError, match="resets"):
        ReplacementPlan(LIFE, 180, 150, 400, resets=1)
