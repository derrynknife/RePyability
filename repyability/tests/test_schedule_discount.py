"""Present values on the calendar: ``discount=`` on ``Task.due``,
``work_by_period``, ``failures_by_period`` and ``events_by_period``, and
``Task.long_run_value``.

The worked examples are UC-47 to UC-49 of
``repyability/schedule/README.md``, each checked by hand in the comments,
against closed forms and against sums and simulations written out
plainly here.
"""

import json
import math
from datetime import date, timedelta

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, RepairableRBD
from repyability.schedule import (
    DutyPlan,
    Recurrence,
    Shifts,
    Task,
    Usage,
    UsageForecast,
    _discount,
    events_by_period,
    failures_by_period,
    work_by_period,
)
from repyability.schedule._forecast import _period_edges
from repyability.schedule._recurrence import _midnight

TODAY = "2026-11-20"
RATE = math.log(1.07)  # 7% a year, as a continuous rate a year


def grease(**options):
    """$250 every 30 days from 20 November 2026."""
    return Task(
        "grease bearings",
        every=Recurrence.every(30, "days", TODAY),
        cost=250.0,
        **options,
    )


def first_tuesdays(**options):
    return Task(
        "inspect",
        every=Recurrence.monthly(weekday="Tue", nth=1),
        cost=250.0,
        **options,
    )


@pytest.fixture
def pump():
    """Pump P-101: Mon-Fri 06:00-22:00 from Monday 16 Nov 2026."""
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    return plan.forecast(
        "2026-11-16", "2027-12-01", readings={"hours": 12_340}
    )


def ring(**costs):
    return NonRepairable(surv.Weibull.from_params([9000.0, 2.5]), **costs)


# ------------------------------------------------------------ the helpers


def test_the_discount_convention():
    # 7% a year: a year on is worth 1/1.07, a year being 365.25 days
    assert _discount.factor(0.07, 1.0) == pytest.approx(1 / 1.07, rel=1e-15)
    assert _discount.continuous(0.07) == math.log1p(0.07)
    assert _discount.factor_on(
        0.07, date(2026, 11, 20), date(2027, 11, 20)
    ) == pytest.approx(1.07 ** (-365 / 365.25), rel=1e-15)
    # per hour of a model on the calendar
    assert _discount.per_unit(0.07, 1.0) == pytest.approx(RATE / 8766.0)
    assert _discount.per_unit(0.07, 24.0) == pytest.approx(RATE / 365.25)
    # the equivalent annual cost: npv / years at no discount
    assert _discount.equivalent_annual(300.0, 0.0, 2.0) == 150.0
    assert _discount.equivalent_annual(100.0, 0.07, math.inf) == (
        pytest.approx(100.0 * RATE)
    )
    # 1 at 0, 1, 2, ... years for ever at 7%: 1.07 / 0.07
    assert _discount.perpetuity(0.07, 0.0, 1.0) == pytest.approx(1.07 / 0.07)
    assert _discount.words(0.07, date(2026, 11, 20)) == (
        "Discounted to 2026-11-20 at 7% a year."
    )
    for bad in (-0.01, "7%", True, 10.0, math.nan):
        with pytest.raises(ValueError, match="yearly rate such as 0.07"):
            _discount.yearly(bad)
    assert _discount.yearly(None) is None


# ------------------------------------------------------------ UC-47


def test_uc47_thirty_day_task_this_year_and_for_ever():
    task = grease()
    year = task.due(TODAY, "2027-11-19", discount=0.07)
    # days 0, 30, ..., 360: 13 tasks, $3,250; each 250 / 1.07^(days/365.25)
    assert year.count == 13 and year.cost == 3250.0
    by_hand = sum(250.0 / 1.07 ** (30 * k / 365.25) for k in range(13))
    assert year.npv == pytest.approx(by_hand, rel=1e-14)
    assert round(year.npv, 2) == 3144.10
    assert year.discount == 0.07
    assert year.notes[-1] == "Discounted to 2026-11-20 at 7% a year."
    # the steady cost a year over the 365 days worth it
    years = 365 / 365.25
    assert year.eac == pytest.approx(
        RATE * year.npv / (1 - math.exp(-RATE * years))
    )
    # for ever: 250 / (1 - 1.07^(-30/365.25))
    value = task.long_run_value(TODAY, 0.07)
    assert value.method == "closed form" and value.precision == 0.0
    assert value.npv == pytest.approx(
        250.0 / (1 - 1.07 ** (-30 / 365.25)), rel=1e-13
    )
    assert round(value.npv) == 45112
    assert value.eac == pytest.approx(value.npv * RATE)
    assert round(value.eac) == 3052
    json.dumps(value.to_dict())
    json.dumps(year.to_dict())


def test_uc47_by_quarter():
    work = work_by_period(
        [grease()], "quarter", TODAY, "2027-11-19", discount=0.07
    )
    assert work.count == (2, 3, 3, 3, 2)
    assert [round(x, 2) for x in work.npv] == [
        498.61,
        737.61,
        725.41,
        713.42,
        469.05,
    ]
    assert round(work.total_npv, 2) == 3144.10
    assert work.notes == ("Discounted to 2026-11-20 at 7% a year.",)
    json.dumps(work.to_dict())


def test_long_run_against_100000_dates():
    # 100,000 dates 30 days apart reach past where dates go (8,200
    # years): summed as day counts, smallest first.
    days = 30.0 * np.arange(100_000)[::-1]
    brute = math.fsum(250.0 * np.exp(-RATE * days / 365.25))
    assert grease().long_run_value(TODAY, 0.07).npv == pytest.approx(
        brute, rel=1e-12
    )
    # weekly, from an anchor three days on, at 3%
    weekly = Task(
        "check", every=Recurrence.every(2, "weeks", "2026-11-23"), cost=80.0
    )
    days = 3.0 + 14.0 * np.arange(100_000)[::-1]
    brute = math.fsum(80.0 * np.exp(-math.log(1.03) * days / 365.25))
    assert weekly.long_run_value(TODAY, 0.03).npv == pytest.approx(
        brute, rel=1e-12
    )


def test_a_pattern_for_ever_against_2000_years_of_dates():
    # The first Tuesday of each month for 2,000 years, found from the
    # calendar itself: the first day of the month moved on to a Tuesday.
    start = date(2026, 11, 20)
    brute = []
    for index in range(2026 * 12 + 10, 4026 * 12 + 10):
        first = date(index // 12, index % 12 + 1, 1)
        tuesday = first + timedelta(days=(1 - first.weekday()) % 7)
        if tuesday >= start:
            brute.append(250.0 * 1.07 ** (-(tuesday - start).days / 365.25))
    value = first_tuesdays().long_run_value(start, 0.07)
    assert value.method == "summed"
    assert value.npv == pytest.approx(math.fsum(brute[::-1]), rel=1e-9)
    assert round(value.npv, 2) == 44350.51
    # what is left out is under the precision asked, and is what a task
    # every day after the last date summed would add
    assert 0.0 < value.precision <= 1e-9 * value.npv
    days = (value.summed_to - start).days
    v = 1.07 ** (-1 / 365.25)
    assert value.precision == pytest.approx(250.0 * v ** (days + 1) / (1 - v))
    assert value.count == len(
        first_tuesdays().due(start, value.summed_to).dates
    )
    # a looser precision stops sooner, within its own bound of the value
    loose = first_tuesdays().long_run_value(start, 0.07, precision=1e-3)
    assert loose.summed_to < value.summed_to
    assert 0.0 <= value.npv - loose.npv <= loose.precision


def test_long_run_with_visits_and_overdue():
    start = date(2026, 11, 20)
    # Overdue (last done 1 September), the interval counting from when it
    # is done: done today, then every 30 days.
    overdue = grease(last_done="2026-09-01")
    assert overdue.long_run_value(start, 0.07).npv == pytest.approx(
        grease().long_run_value(start, 0.07).npv
    )
    # Overdue with dates that stay fixed: done today, then on the rule's
    # own dates (20 December on).
    fixed = Task(
        "fixed",
        every=Recurrence.every(30, "days", "2026-09-21"),
        last_done="2026-09-21",
        resets=False,
        cost=250.0,
    )
    value = fixed.long_run_value(start, 0.07)
    done = fixed.due(start, "2027-03-01").done
    assert done[:3] == (start, date(2026, 12, 20), date(2027, 1, 19))
    assert value.npv == pytest.approx(
        250.0 + 250.0 * 1.07 ** (-30 / 365.25) / (1 - 1.07 ** (-30 / 365.25))
    )
    # A task that waits for Tuesday's visit is summed, and is the dates'
    # own present value over a long period, to the bound.
    waits = grease(visits=Recurrence.weekly("Tue"))
    value = waits.long_run_value(start, 0.07, precision=1e-6)
    assert value.method == "summed"
    direct = waits.due(start, value.summed_to, discount=0.07)
    assert value.npv == pytest.approx(direct.npv, rel=1e-14)
    longer = waits.due(start, "2500-01-01", discount=0.07).npv
    assert 0.0 < longer - value.npv <= value.precision
    # nothing each time is nothing for ever
    assert grease().cost == 250.0
    free = Task("look", every=Recurrence.weekly("Mon"), cost=0.0)
    assert free.long_run_value(start, 0.07).npv == 0.0


def test_what_has_no_long_run_value_is_refused(pump):
    with pytest.raises(ValueError, match="costs without end.*due\\(start"):
        grease().long_run_value(TODAY, 0.0)
    with pytest.raises(ValueError, match="has no cost to value: give cost="):
        Task("x", every=Recurrence.weekly("Mon")).long_run_value(TODAY, 0.07)
    given = Task(
        "x", every=Recurrence.on(["2027-01-05", "2027-06-01"]), cost=9.0
    )
    with pytest.raises(
        ValueError,
        match="given dates, which end on 2027-06-01: "
        r"due\(2026-11-20, 2027-06-01, discount=...\)",
    ):
        given.long_run_value(TODAY, 0.07)
    with pytest.raises(ValueError, match="nothing is left of it from then"):
        given.long_run_value("2028-01-01", 0.07)
    at_visits = grease(visits=Recurrence.on(["2027-01-05", "2027-06-01"]))
    with pytest.raises(ValueError, match="visits on given dates"):
        at_visits.long_run_value(TODAY, 0.07)
    by_use = Task("service", every=Usage(500, "hours"), cost=300.0)
    with pytest.raises(ValueError, match="limit by use.*usage=..., discount"):
        by_use.long_run_value(TODAY, 0.07)
    # given dates beside a rule that goes on are not an end
    both = Task(
        "x",
        every=Recurrence.on(["2027-01-05"]),
        also=Recurrence.yearly(month=7, day=1),
        cost=9.0,
    )
    assert both.long_run_value(TODAY, 0.07).method == "summed"
    # a rate too low to settle before dates run out
    with pytest.raises(ValueError, match="At 0.2% a year .* year 9999"):
        first_tuesdays().long_run_value(TODAY, 0.002)
    # ... also where only the sum itself shows it (a task once a year)
    yearly = Task("x", every=Recurrence.yearly(month=7, day=1), cost=9.0)
    with pytest.raises(ValueError, match=r"looser precision=.*due\(start"):
        yearly.long_run_value(TODAY, 0.002, precision=5e-7)
    loose = yearly.long_run_value(TODAY, 0.002, precision=1e-4)
    # 9 a year for ever at 0.2%: about 9 / ln(1.002)
    assert loose.npv == pytest.approx(9.0 / math.log(1.002), rel=2e-3)
    with pytest.raises(ValueError, match="precision must be a share"):
        grease().long_run_value(TODAY, 0.07, precision=0)
    with pytest.raises(ValueError, match="yearly rate such as 0.07"):
        grease().long_run_value(TODAY, None)


def test_no_discount_leaves_results_as_they_were():
    new = {"discount", "npv", "eac", "total_npv", "notes"}
    task = grease()
    plain = task.due(TODAY, "2027-11-19").to_dict()
    assert plain["discount"] is None and plain["npv"] is None
    assert plain["eac"] is None and plain["notes"] == []
    priced = task.due(TODAY, "2027-11-19", discount=0.07).to_dict()
    assert {k: v for k, v in plain.items() if k not in new} == {
        k: v for k, v in priced.items() if k not in new
    }
    tasks = [task, Task("walk round", every=Recurrence.weekly("Mon"))]
    plain = work_by_period(tasks, "month", TODAY, "2027-11-19").to_dict()
    assert plain["npv"] is None and plain["total_npv"] is None
    assert plain["eac"] is None and plain["notes"] == []
    priced = work_by_period(
        tasks, "month", TODAY, "2027-11-19", discount=0.07
    ).to_dict()
    assert {k: v for k, v in plain.items() if k not in new} == {
        k: v for k, v in priced.items() if k not in new
    }
    # the task with no cost is named, in no_cost and in the notes
    assert priced["no_cost"] == ["walk round"]
    assert priced["notes"][-1] == (
        "The present values leave out the tasks with no cost: walk round."
    )
    no_cost = tasks[1].due(TODAY, "2027-11-19", discount=0.07)
    assert no_cost.npv is None and no_cost.eac is None
    assert no_cost.notes[-1] == (
        "No present value: the task has no cost (give cost=)."
    )


def test_discount_of_zero_is_the_plain_total():
    year = grease().due(TODAY, "2027-11-19", discount=0)
    assert year.npv == year.cost == 3250.0
    assert year.eac == pytest.approx(3250.0 / (365 / 365.25))
    work = work_by_period(
        [grease(), first_tuesdays()], "month", TODAY, "2027-11-19", discount=0
    )
    assert work.npv == work.cost
    assert work.total_npv == sum(work.cost)


def test_periods_present_values_add_up():
    tasks = [
        grease(visits=Recurrence.weekly("Tue")),
        first_tuesdays(),
        Task(
            "service",
            every=Recurrence.every(3, "months", "2026-12-20"),
            cost=480.0,
        ),
    ]
    whole = sum(t.due(TODAY, "2029-11-19", discount=0.05).npv for t in tasks)
    for period in ("day", "week", "month", "quarter", "year"):
        work = work_by_period(
            tasks, period, TODAY, "2029-11-19", discount=0.05
        )
        assert work.total_npv == pytest.approx(whole, rel=1e-13)
        assert sum(work.npv) == pytest.approx(whole, rel=1e-13)
        assert all(0.0 <= a <= b for a, b in zip(work.npv, work.cost))
    # each task is discounted from the date it is done (after its visit)
    waits = tasks[0].due(TODAY, "2026-12-31", discount=0.05)
    assert waits.done[0] == date(2026, 11, 24)
    assert waits.npv == pytest.approx(
        sum(
            250.0 * 1.05 ** (-(d - date(2026, 11, 20)).days / 365.25)
            for d in waits.done
        )
    )


def test_far_dates_are_the_anchors_own():
    # A rule's dates far from its anchor, against dateutil's walked from
    # the anchor itself.
    rules = [
        Recurrence.every(30, "days", "2024-02-29"),
        Recurrence.every(2, "weeks", "2026-01-31"),
        Recurrence.every(5, "months", "2026-01-31"),
        Recurrence.every(3, "months", "2026-01-31", short_months="skip"),
        Recurrence.every(3, "years", "2024-02-29"),
        Recurrence.weekly(["Mon", "Thu"], every=3, anchor="2026-11-14"),
        Recurrence.monthly(
            weekday="Fri", nth=-1, every=7, anchor="2025-03-01"
        ),
        Recurrence.monthly(day=31, every=2, anchor="2026-11-14"),
        Recurrence.yearly(
            month=3, weekday="Tue", nth=1, every=2, anchor="2026-11-14"
        ),
        Recurrence.yearly(month=2, day=29, every=5, anchor="2027-12-31"),
    ]
    for rule in rules:
        walked = rule._rrule(rule.anchor)
        for days in (0, 1, 27, 59, 365, 366, 1500, 25_000, 147_000):
            lo = rule.anchor + timedelta(days=days)
            want = []
            for moment in walked.xafter(_midnight(lo), inc=True):
                want.append(moment.date())
                if len(want) == 4:
                    break
            got = rule.dates(lo, want[-1])
            assert list(got) == want, (rule.describe(), lo)


# ------------------------------------------------------------ UC-48


def test_uc48_wear_ring_present_value_by_month(pump):
    part = ring(unplanned_cost=4800.0)
    plain = failures_by_period(part, pump, "month", age=5200.0)
    months = failures_by_period(part, pump, "month", age=5200.0, discount=0.07)
    # every field there was is as it was
    for name in plain:
        if name in ("notes", "total"):
            continue
        a, b = plain[name], months[name]
        if name in ("discount", "expected_npv", "npv", "npv_precision", "eac"):
            assert a is None and b is not None
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b
    assert months.notes[:-1] == plain.notes
    assert {
        k: v for k, v in months.total.items() if "npv" not in k
    } == plain.total
    # A month's cost is worth at most its value from the month's start,
    # at least from its end: $2,874.75 to $2,890.71 over the 13 months.
    _, seconds = _period_edges(pump, "month")
    ends = seconds / 86400.0 / 365.25
    starts = np.r_[0.0, ends[:-1]]
    high = months.expected_cost * 1.07**-starts
    low = months.expected_cost * 1.07**-ends
    assert round(high.sum(), 2) == 2890.71 and round(low.sum(), 2) == 2874.75
    assert np.all(months.expected_npv <= high + 1e-9)
    assert np.all(months.expected_npv >= low - 1e-9)
    # Each failure's cost discounted from when it is expected to fall:
    # $2,882.742, to within $0.0165 (the 36 windows' bracket of the
    # method before gave $2,882.77 give or take $2.47).
    assert months.npv == pytest.approx(2882.742, abs=5e-4)
    assert months.npv_precision == pytest.approx(0.01646, abs=5e-5)
    assert months.npv_precision <= 1e-3 * months.npv
    finer = failures_by_period(
        part, pump, "month", age=5200.0, discount=0.07, npv_tolerance=1e-5
    )
    assert abs(finer.npv - months.npv) <= months.npv_precision
    assert finer.npv_precision <= 1e-5 * finer.npv
    assert months.expected_npv.sum() == pytest.approx(months.npv, rel=1e-13)
    assert months.total["expected_npv"] == months.npv
    assert months.total["npv_precision"] == months.npv_precision
    assert [round(float(x), 2) for x in months.expected_npv] == [
        104.40,
        226.20,
        214.53,
        210.05,
        246.83,
        239.74,
        230.71,
        242.24,
        241.48,
        239.51,
        236.52,
        222.18,
        228.36,
    ]
    years = ends[-1]
    assert months.eac == pytest.approx(
        RATE * months.npv / (1 - math.exp(-RATE * years))
    )
    assert round(months.eac, 2) == 2869.51
    assert months.notes[-1] == (
        "Discounted to 2026-11-16 at 7% a year. The present values "
        "(expected_npv, npv) discount the expected cost from when it is "
        "expected to fall, to within 0.0163 in all; expected_cost and the "
        "ranges are not discounted."
    )
    json.dumps(months.to_dict())
    # no discount is the expected cost itself
    zero = failures_by_period(part, pump, "month", age=5200.0, discount=0)
    assert np.array_equal(zero.expected_npv, zero.expected_cost)
    assert zero.npv == zero.total["expected_cost"]
    assert zero.npv_precision == 0.0


def test_constant_rate_in_calendar_time_against_the_closed_form():
    # A failure per 2,000 h, $1,000 each, replaced at once, in calendar
    # time over 2027 (365 days): rate x cost x the integral of the
    # discount factor, 0.5 (1 - exp(-r T)) / r with r an hour.
    part = NonRepairable(
        surv.Exponential.from_params([1 / 2000.0]), unplanned_cost=1000.0
    )
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    r = RATE / 8766.0
    hours = 365 * 24.0
    exact = 0.5 * -math.expm1(-r * hours) / r
    last = None
    for tolerance in (1e-2, 1e-3, 1e-4):
        quarters = failures_by_period(
            part,
            usage,
            "quarter",
            unit="hours",
            discount=0.07,
            npv_tolerance=tolerance,
        )
        assert quarters.npv_precision <= tolerance * quarters.npv
        assert abs(quarters.npv - exact) <= quarters.npv_precision
        if last is not None:
            assert quarters.npv_precision <= last
        last = quarters.npv_precision
    # each quarter against its own closed form
    _, seconds = _period_edges(usage, "quarter")
    t = np.r_[0.0, seconds / 3600.0]
    by_quarter = 0.5 * np.diff(-np.expm1(-r * t)) / r
    assert quarters.expected_npv == pytest.approx(by_quarter, rel=1e-4)


def calendar_years(operating):
    """When the pump of the fixture reaches each operating hour, in years
    from Monday 16 November 2026 00:00: 16 hours a weekday from 06:00."""
    week, rest = np.divmod(operating, 80.0)
    day, hour = np.divmod(rest, 16.0)
    return (week * 168.0 + day * 24.0 + 6.0 + hour) / 8766.0


def test_the_present_value_holds_a_simulation(pump):
    # 400,000 rings renewed at each failure over the forecast's 4,352
    # operating hours from age 5,200, each failure's $4,800 discounted
    # from when the pump reaches that hour on its shifts.
    rng = np.random.default_rng(48)
    n, span = 400_000, 4352.0
    h0 = (5200.0 / 9000.0) ** 2.5
    t = 9000.0 * (h0 + rng.exponential(size=n)) ** 0.4 - 5200.0
    worth = np.zeros(n)
    while True:
        alive = t < span
        if not alive.any():
            break
        worth[alive] += 4800.0 * np.exp(-RATE * calendar_years(t[alive]))
        t[alive] += 9000.0 * rng.weibull(2.5, alive.sum())
    se = worth.std() / np.sqrt(n)
    part = ring(unplanned_cost=4800.0)
    last = None
    for tolerance in (5e-3, 1e-3, 1e-4):
        months = failures_by_period(
            part,
            pump,
            "month",
            age=5200.0,
            discount=0.07,
            npv_tolerance=tolerance,
        )
        assert months.npv_precision <= tolerance * months.npv
        assert abs(worth.mean() - months.npv) < (4 * se + months.npv_precision)
        if last is not None:
            # the finer value is within the coarser one's precision
            assert months.npv_precision <= last[1]
            assert abs(months.npv - last[0]) <= last[1]
        last = (months.npv, months.npv_precision)
    assert abs(worth.mean() - months.npv) < 4 * se


def test_nothing_accrues_while_idle(pump):
    days = failures_by_period(
        ring(unplanned_cost=4800.0), pump, "day", age=5200.0, discount=0.07
    )
    # Sat 21 and Sun 22 Nov: no hours, no cost, no present value
    assert days.operating[5:7].tolist() == [0.0, 0.0]
    assert days.expected_npv[5:7].tolist() == [0.0, 0.0]
    assert np.all(days.expected_npv[days.operating > 0] > 0.0)
    assert np.all(days.expected_npv <= days.expected_cost)
    months = failures_by_period(
        ring(unplanned_cost=4800.0), pump, "month", age=5200.0, discount=0.07
    )
    # cut by day or by month, the same value, within the precisions
    assert abs(days.npv - months.npv) <= (
        days.npv_precision + months.npv_precision
    )


def test_a_policy_as_a_present_value(pump):
    part = ring(unplanned_cost=4800.0, planned_cost=900.0)
    for resets in (True, False):
        months = failures_by_period(
            part,
            pump,
            "month",
            age=1000.0,
            every=3000.0,
            resets=resets,
            discount=0.07,
        )
        _, seconds = _period_edges(pump, "month")
        ends = seconds / 86400.0 / 365.25
        high = months.expected_cost * 1.07 ** -np.r_[0.0, ends[:-1]]
        low = months.expected_cost * 1.07**-ends
        assert np.all(months.expected_npv <= high + 1e-4)
        assert np.all(months.expected_npv >= low - 1e-4)
        assert months.npv_precision <= 1e-3 * months.npv
        assert months.npv < months.total["expected_cost"]
    # Block replacement: the planned $900 at 2,000 h on (3,000 less the
    # 1,000 since the last) falls at one instant, in May 2027; that
    # month's value holds it discounted from then.
    at = float(calendar_years(np.array([2000.0]))[0])
    may = [s.strftime("%Y-%m") for s in months.starts].index("2027-05")
    assert months.expected_planned[may] == pytest.approx(1.0)
    assert months.expected_npv[may] > 900.0 * 1.07**-at


def test_a_part_with_no_cost_has_no_present_value(pump):
    months = failures_by_period(
        ring(), pump, "month", age=5200.0, discount=0.07
    )
    assert months.expected_npv is None and months.npv is None
    assert months.eac is None and months.discount == 0.07
    assert months.notes[-1] == (
        "No present value: the part has no cost (give unplanned_cost=)."
    )
    # A replacement that takes time is run to failure with failures'
    # exact counts: its cost is discounted the same way.
    slow = NonRepairable(
        surv.Weibull.from_params([9000.0, 2.5]),
        surv.Exponential.from_params([1 / 8.0]),
        unplanned_cost=4800.0,
    )
    months = failures_by_period(slow, pump, "month", age=5200.0, discount=0.07)
    assert months.method == "exact" and months.cost_low is None
    _, seconds = _period_edges(pump, "month")
    ends = seconds / 86400.0 / 365.25
    high = months.expected_cost * 1.07 ** -np.r_[0.0, ends[:-1]]
    low = months.expected_cost * 1.07**-ends
    assert np.all(months.expected_npv <= high + 1e-9)
    assert np.all(months.expected_npv >= low - 1e-9)
    assert months.npv == pytest.approx(months.expected_npv.sum())
    assert 0.0 < months.npv_precision <= 1e-3 * months.npv
    assert months.notes[-1].endswith(
        f"to within {months.npv_precision:.3g} in all; expected_cost is "
        "not discounted."
    )
    with pytest.raises(ValueError, match="npv_tolerance must be a share"):
        failures_by_period(
            slow, pump, "month", discount=0.07, npv_tolerance=0.0
        )
    with pytest.raises(ValueError, match="yearly rate such as 0.07"):
        failures_by_period(slow, pump, "month", discount=-0.07)


def test_a_forecast_short_of_the_tolerance_says_so(pump, monkeypatch):
    # With two rounds of halving the most, 1e-7 is out of reach: the
    # notes say how far it got.
    from repyability.rbd import _common

    monkeypatch.setattr(_common, "_DISCOUNT_ROUNDS", 2)
    months = failures_by_period(
        ring(unplanned_cost=4800.0),
        pump,
        "month",
        age=5200.0,
        discount=0.07,
        npv_tolerance=1e-7,
    )
    assert 1e-7 * months.npv < months.npv_precision < 1e-3 * months.npv
    assert any(
        "short of npv_tolerance=1e-07: 2 rounds of halving is the most "
        "worked out" in note
        for note in months.notes
    )
    assert abs(months.npv - 2882.741) <= months.npv_precision + 5e-4


# ------------------------------------------------------------ UC-49


def costed_pump():
    return RepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {
            "pump": {
                "reliability": surv.Exponential.from_params([1 / 2000.0]),
                "repairability": "instant",
                "repair_cost": 1000.0,
            }
        },
    )


def test_uc49_constant_rate_by_month_as_present_values():
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    plain = events_by_period(
        costed_pump(), usage, unit="hours", cost=True, seed=1, mc_samples=200
    )
    months = events_by_period(
        costed_pump(),
        usage,
        unit="hours",
        cost=True,
        seed=1,
        mc_samples=200,
        discount=0.07,
    )
    assert plain.discount is None and plain.expected_npv is None
    assert plain.npv is None and plain.eac is None
    assert np.array_equal(plain.cost.mean, months.cost.mean)
    assert np.array_equal(plain.model_ends, months.model_ends)
    # $0.50 an hour: 0.5 (1 - exp(-r T)) / r, r = ln(1.07) / 8,766 an hour
    r = RATE / 8766.0
    to = [0.5 * -math.expm1(-r * hours) / r for hours in (744, 1416, 8760)]
    assert [round(x, 2) for x in to] == [370.93, 704.15, 4235.21]
    assert months.expected_npv[0] == pytest.approx(to[0], rel=1e-6)
    assert months.expected_npv[1] == pytest.approx(to[1] - to[0], rel=1e-6)
    assert months.npv == pytest.approx(to[2], rel=1e-7)
    assert months.expected_npv.sum() == pytest.approx(months.npv)
    # costs at a constant rate: the equivalent annual cost is the rate
    assert months.eac == pytest.approx(0.5 * 8766.0, rel=1e-7)
    assert months.notes[1].startswith(
        "Discounted to 2027-01-01 at 7% a year. The present values "
        "(expected_npv, npv) discount each cost from when it falls (exact); "
        "cost (its mean and range) is not discounted"
    )
    json.dumps(months.to_dict())
    # at no discount, the periods' means
    zero = events_by_period(
        costed_pump(),
        usage,
        unit="hours",
        cost=True,
        seed=1,
        mc_samples=200,
        discount=0,
    )
    assert zero.expected_npv == pytest.approx(zero.cost.mean, rel=1e-9)


def test_events_present_value_is_expected_npvs_cost():
    # A unit that wears, takes time to repair, is replaced on a block
    # schedule and costs while the system is down, with the diagram in
    # days: the forecast's value is expected_npv's cost over the horizon.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": surv.Weibull.from_params([200.0, 2.0]),
                "repairability": surv.Exponential.from_params([1 / 2.0]),
                "repair_cost": 700.0,
                "preventive": {
                    "interval": 150.0,
                    "policy": "block",
                    "duration": surv.ExactEventTime.from_params([0.5]),
                    "cost": 120.0,
                },
            }
        },
        downtime_cost_rate=40.0,
    )
    usage = UsageForecast.calendar("2027-01-01", "2029-01-01")
    quarters = events_by_period(
        rbd,
        usage,
        "quarter",
        unit="days",
        cost=True,
        seed=2,
        mc_samples=200,
        discount=0.07,
    )
    npv = rbd.expected_npv(731.0, discount_rate=RATE / 365.25)
    assert quarters.npv == pytest.approx(npv.cost, rel=1e-7)
    assert quarters.npv < float(np.sum(quarters.cost.mean))
    assert np.all(quarters.expected_npv <= quarters.cost.mean + 1e-9)
    assert quarters.eac == pytest.approx(npv.eac * 365.25, rel=1e-7)


def test_events_discount_needs_cost():
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    with pytest.raises(ValueError, match="with cost=True"):
        events_by_period(costed_pump(), usage, unit="hours", discount=0.07)
    with pytest.raises(ValueError, match="yearly rate such as 0.07"):
        events_by_period(
            costed_pump(), usage, unit="hours", cost=True, discount="7%"
        )
