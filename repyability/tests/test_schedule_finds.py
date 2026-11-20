"""What a test of a hidden failure is worth: ``task_value`` with
``Finds`` (UC-54 of ``repyability/schedule/README.md``).

Checked against an exponential device's closed forms on the device's
clock, ``failure_finding_window`` in model time, and sums that must hold.
"""

import json
import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, failure_finding_window
from repyability.schedule import (
    DutyPlan,
    Finds,
    Recurrence,
    Replaces,
    Shifts,
    Task,
    UsageForecast,
    task_value,
)
from repyability.tests.test_failure_finding_window import closed_form

DAYS = 365.25
RATE = math.log(1.07)  # 7% a year, continuously, per year


def trip():
    """A trip failing hidden at 0.1 a year."""
    return NonRepairable(surv.Exponential.from_params([0.1]))


def trip_test(**options):
    """Tested every 6 months from 1 January 2027, $400 a test."""
    options.setdefault("cost", 400.0)
    return Task(
        "trip test",
        every=Recurrence.every(6, "months", "2027-01-01"),
        **options,
    )


def finds(**options):
    options.setdefault("demand_rate", 0.5)
    options.setdefault("consequence_cost", 250_000.0)
    options.setdefault("repair_cost", 3_000.0)
    return Finds(trip(), **options)


@pytest.fixture(scope="module")
def three_years():
    return UsageForecast.calendar("2026-11-16", "2029-11-16")


def by_hand(tests, window, reveal, discount, price=400.0):
    """Each side's cost from the closed form, in years: (with, without,
    found, failed time with, failed time without)."""
    demand, consequence, repair = 0.5, 250_000.0, 3_000.0
    each = consequence + (repair if reveal else 0.0)
    shown = demand if reveal else 0.0
    q, failed, discounted = closed_form(0.1, tests, window, shown, discount)
    factors = np.exp(-discount * np.array(tests))
    with_task = (
        price * factors.sum()
        + repair * factors @ q
        + each * demand * discounted
    )
    _, never, never_discounted = closed_form(0.1, [], window, shown, discount)
    return with_task, each * demand * never_discounted, q, failed, never


def test_uc54_six_monthly_trip_test(three_years):
    tests = [d / DAYS for d in (46, 227, 411, 593, 777, 958)]
    window = 1096 / DAYS
    plain = task_value(trip_test(), finds(), three_years, unit="years")
    assert [str(d) for d in plain.dates] == [
        "2027-01-01",
        "2027-07-01",
        "2028-01-01",
        "2028-07-01",
        "2029-01-01",
        "2029-07-01",
    ]
    np.testing.assert_allclose(plain.model_times, tests, rtol=1e-12)
    with_task, without, q, failed, never = by_hand(tests, window, True, 0.0)
    total = plain.total
    assert total["count"] == 6
    assert total["found"] == pytest.approx(q.sum(), rel=1e-9)
    assert total["failures_with"] == pytest.approx(0.5 * failed, rel=1e-9)
    assert total["failures_without"] == pytest.approx(0.5 * never, rel=1e-9)
    assert total["unavailability_with"] == pytest.approx(
        failed / window, rel=1e-9
    )
    assert total["unavailability_without"] == pytest.approx(
        never / window, rel=1e-9
    )
    assert abs(total["with_task"] - with_task) <= plain.precision
    assert abs(total["without_task"] - without) <= plain.precision
    assert plain.precision < 1e-6 * without
    # The README's numbers.
    assert total["found"] == pytest.approx(0.2279, abs=5e-5)
    assert total["failures_with"] == pytest.approx(0.0320, abs=5e-5)
    assert total["failures_without"] == pytest.approx(0.1341, abs=5e-5)
    assert total["with_task"] == pytest.approx(11_174.90, abs=0.01)
    assert total["without_task"] == pytest.approx(33_931.58, abs=0.01)
    assert total["value"] == pytest.approx(22_756.68, abs=0.01)
    assert total["task_cost"] == 2400.0 and plain.eac is None
    assert plain.total["with_task"] == plain.total["cost_with"]

    worth = task_value(
        trip_test(), finds(), three_years, unit="years", discount=0.07
    )
    with_pv, without_pv, _, _, _ = by_hand(tests, window, True, RATE)
    total = worth.total
    assert total["task_cost"] == pytest.approx(
        400.0 * sum(1.07**-t for t in tests), rel=1e-12
    )
    assert abs(total["with_task"] - with_pv) <= worth.precision
    assert abs(total["without_task"] - without_pv) <= worth.precision
    assert abs(total["value"] - (without_pv - with_pv)) <= worth.precision
    assert worth.precision < 2e-3 * without_pv
    assert with_pv == pytest.approx(10_087.23, abs=0.01)
    assert without_pv == pytest.approx(29_936.76, abs=0.01)
    assert total["cost_with"] == plain.total["cost_with"]
    assert set(worth.eac) == {"with_task", "without_task", "value"}
    assert "never tested" in worth.without_words
    assert "tests the device" in worth.does
    json.dumps(worth.to_dict())


def test_only_tests_find_a_failure(three_years):
    """``demands_reveal=False``: without the task a failure is for good."""
    tests = [d / DAYS for d in (46, 227, 411, 593, 777, 958)]
    window = 1096 / DAYS
    found = task_value(
        trip_test(),
        finds(demands_reveal=False),
        three_years,
        unit="years",
        discount=0.07,
    )
    with_pv, without_pv, q, failed, never = by_hand(tests, window, False, RATE)
    assert found.total["found"] == pytest.approx(q.sum(), rel=1e-9)
    assert found.total["failures_without"] == pytest.approx(
        0.5 * never, rel=1e-9
    )
    assert never == pytest.approx(
        window - (1 - math.exp(-0.1 * window)) / 0.1, rel=1e-12
    )
    assert abs(found.total["with_task"] - with_pv) <= found.precision
    assert abs(found.total["without_task"] - without_pv) <= found.precision
    assert "never found" in found.without_words


def test_matches_the_model_time_calculation():
    """Every 30 days in calendar time is every 720 h on a Weibull
    device's clock, 1,000 h old: ``failure_finding_window``'s numbers,
    its discount rate a year's as an hour's."""
    device = NonRepairable(surv.Weibull.from_params([6000.0, 1.8]))
    task = Task(
        "test", every=Recurrence.every(30, "days", "2027-01-11"), cost=50.0
    )
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    does = Finds(
        device, demand_rate=1e-3, consequence_cost=9_000.0, repair_cost=700.0
    )
    times = 240.0 + 720.0 * np.arange(12)
    for discount, rate in ((None, 0.0), (0.07, RATE / (24.0 * DAYS))):
        found = task_value(task, does, usage, age=1000.0, discount=discount)
        prices = dict(
            age=1000.0,
            repair_cost=700.0,
            demand_rate=1e-3,
            consequence_cost=9_000.0,
            discount_rate=rate,
        )
        tested = failure_finding_window(
            device, times, 8760.0, test_cost=50.0, **prices
        )
        never = failure_finding_window(device, [], 8760.0, **prices)
        np.testing.assert_allclose(found.model_times, times)
        slack = found.precision + tested.precision + never.precision
        assert found.total["found"] == pytest.approx(
            tested.failures_found, rel=1e-9
        )
        assert found.total["cost_with"] == pytest.approx(tested.cost, rel=1e-9)
        assert abs(found.total["with_task"] - tested.present_value) <= slack
        assert abs(found.total["without_task"] - never.present_value) <= slack
        if discount is None:
            assert found.total["with_task"] == found.total["cost_with"]
            assert found.total["without_task"] == pytest.approx(
                never.cost, rel=1e-9
            )


def test_uneven_dates_under_a_duty_plan():
    """The first Tuesday of each month on a pump that runs Mon-Fri 06:00
    to 22:00: the tests are at midnight, at uneven times on the pump's
    clock, and demands come only while it runs. Against the closed form
    on that clock, discounted by hand over each calendar day."""
    usage = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2026-11-16", "2027-12-01"
    )
    task = Task(
        "start test",
        every=Recurrence.monthly(weekday="Tue", nth=1),
        cost=120.0,
    )
    rate, demand, consequence, repair = 2e-4, 1e-3, 20_000.0, 900.0
    device = NonRepairable(surv.Exponential.from_params([rate]))
    does = Finds(
        device,
        demand_rate=demand,
        consequence_cost=consequence,
        repair_cost=repair,
    )
    plain = task_value(task, does, usage)
    assert plain.dates[:3] == (
        date(2026, 12, 1),
        date(2027, 1, 5),
        date(2027, 2, 2),
    )
    assert len(plain.dates) == 12
    hours = [
        float(usage.model_time(datetime.combine(d, datetime.min.time())))
        for d in plain.dates
    ]
    np.testing.assert_allclose(plain.model_times, hours)
    assert len(set(np.round(np.diff(hours)))) > 1  # 320 h and 400 h apart
    window = float(usage.model_time(usage.end))
    q, failed, _ = closed_form(rate, hours, window, demand)
    _, never, _ = closed_form(rate, [], window, demand)
    assert plain.total["found"] == pytest.approx(q.sum(), rel=1e-9)
    assert plain.total["failures_with"] == pytest.approx(
        demand * failed, rel=1e-9
    )
    assert plain.total["failures_without"] == pytest.approx(
        demand * never, rel=1e-9
    )

    worth = task_value(task, does, usage, discount=0.07)
    # By hand: each day's expected demands on a failed device (the closed
    # form at the day's ends on the pump's clock) discounted at the day's
    # start and at its end bracket the true value.
    start = datetime(2026, 11, 16)
    days = (datetime(2027, 12, 1) - start).days
    stamps = [start + timedelta(days=d) for d in range(days + 1)]
    clock = np.array([float(usage.model_time(s)) for s in stamps])
    k = rate + demand

    def failed_to(t, tests):
        edges = np.array([0.0, *tests])
        since = t - edges[np.searchsorted(edges, t, side="right") - 1]
        done = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= t:
                done += rate / k * ((b - a) + math.expm1(-k * (b - a)) / k)
        return done + rate / k * (since + math.expm1(-k * since) / k)

    low = high = 0.0
    never_low = never_high = 0.0
    each = (consequence + repair) * demand
    for day in range(days):
        early, late = 1.07 ** (-day / DAYS), 1.07 ** (-(day + 1) / DAYS)
        amount = failed_to(clock[day + 1], hours) - failed_to(
            clock[day], hours
        )
        low, high = low + each * amount * late, high + each * amount * early
        amount = failed_to(clock[day + 1], []) - failed_to(clock[day], [])
        never_low += each * amount * late
        never_high += each * amount * early
    lumps = sum(
        (120.0 + repair * found)
        * 1.07 ** (-(d - date(2026, 11, 16)).days / DAYS)
        for d, found in zip(plain.dates, q)
    )
    assert low + lumps <= worth.total["with_task"] <= high + lumps
    assert never_low <= worth.total["without_task"] <= never_high
    assert worth.total["task_cost"] == pytest.approx(
        sum(
            120.0 * 1.07 ** (-(d - date(2026, 11, 16)).days / DAYS)
            for d in plain.dates
        ),
        rel=1e-12,
    )


def test_tests_while_the_device_does_not_run(three_years):
    """Two tests with no running between them are one test of the
    device: both cost, and the second finds nothing."""
    usage = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2026-11-16", "2027-03-01"
    )
    # Saturday and Sunday midnights: the pump last ran on Friday.
    weekend = Task(
        "start test",
        every=Recurrence.weekly(["Sat", "Sun"]),
        cost=10.0,
    )
    saturdays = Task("start test", every=Recurrence.weekly(["Sat"]), cost=10.0)
    device = NonRepairable(surv.Exponential.from_params([1e-3]))
    does = Finds(device, demand_rate=1e-3, consequence_cost=5_000.0)
    both = task_value(weekend, does, usage)
    one = task_value(saturdays, does, usage)
    assert both.total["count"] == 2 * one.total["count"]
    assert both.total["found"] == pytest.approx(one.total["found"], rel=1e-12)
    assert both.total["cost_with"] == pytest.approx(
        one.total["cost_with"] + 10.0 * one.total["count"], rel=1e-12
    )


def test_the_value_follows_the_prices(three_years):
    """Dearer consequences make the tests worth more, dearer tests less;
    with nothing at stake the value is minus the tests' cost, and it
    changes sign at the test price that equals the saving."""

    def value(**options):
        cost = options.pop("cost", 400.0)
        return task_value(
            trip_test(cost=cost),
            finds(**options),
            three_years,
            unit="years",
            discount=0.07,
        )

    values = [
        value(consequence_cost=c).total["value"]
        for c in (0.0, 1e3, 1e4, 1e5, 1e6)
    ]
    assert all(b > a for a, b in zip(values, values[1:]))
    nothing = value(consequence_cost=0.0, repair_cost=0.0)
    assert nothing.total["value"] == -nothing.total["task_cost"] < 0.0
    assert nothing.precision == 0.0
    free = value(cost=0.0)
    dear = value(cost=4000.0)
    assert free.total["value"] > value().total["value"] > dear.total["value"]
    factors = value(cost=1.0).total["task_cost"]
    break_even = free.total["value"] / factors
    assert free.total["value"] > 0.0
    slack = 2.0 * free.precision
    assert abs(value(cost=break_even).total["value"]) <= slack
    assert value(cost=1.01 * break_even).total["value"] < 0.0
    assert value(cost=0.99 * break_even).total["value"] > 0.0
    # No demands: nothing to save.
    idle = value(demand_rate=0.0)
    assert idle.total["failures_without"] == 0.0
    assert idle.total["value"] < 0.0


def test_periods_add_up(three_years):
    months = task_value(
        trip_test(), finds(), three_years, unit="years", discount=0.07
    )
    years = task_value(
        trip_test(),
        finds(),
        three_years,
        unit="years",
        discount=0.07,
        period="year",
    )
    for found in (months, years):
        for name in (
            "with_task",
            "without_task",
            "value",
            "task_cost",
            "cost_with",
            "cost_without",
            "failures_with",
            "failures_without",
            "failures_avoided",
            "found",
        ):
            assert float(np.sum(getattr(found, name))) == pytest.approx(
                found.total[name], rel=1e-12, abs=1e-12
            )
        np.testing.assert_allclose(
            found.value, found.without_task - found.with_task
        )
        assert np.all(found.unavailability_with <= 0.1)
        # Each side is worked out on its own grid, to 1e-6 of itself
        # (before the first test the two are the same curve).
        assert np.all(
            found.unavailability_with
            <= found.unavailability_without * (1.0 + 1e-6)
        )
    for name in ("cost_with", "cost_without", "found"):
        assert months.total[name] == pytest.approx(years.total[name], rel=1e-9)
    for name in ("with_task", "without_task", "value"):
        assert months.total[name] == pytest.approx(
            years.total[name], abs=months.precision + years.precision
        )
    # Untested, the trip's unavailability climbs year on year.
    assert np.all(np.diff(years.unavailability_without) > 0.0)


def test_a_replacement_has_no_hidden_failures(three_years):
    part = NonRepairable(
        surv.Weibull.from_params([3.0, 2.0]), unplanned_cost=100.0
    )
    found = task_value(trip_test(), Replaces(part), three_years, unit="years")
    assert found.found is None and found.unavailability_with is None
    assert "found" not in found.total


def test_refusals(three_years):
    with pytest.raises(ValueError, match="what one test costs"):
        task_value(trip_test(cost=None), finds(), three_years)
    with pytest.raises(TypeError, match="NonRepairable"):
        Finds(
            surv.Exponential.from_params([0.1]),
            demand_rate=1.0,
            consequence_cost=1.0,
        )
    with pytest.raises(TypeError, match="demand_rate"):
        Finds(trip(), consequence_cost=1.0)
    with pytest.raises(ValueError, match="demand_rate must be a number"):
        Finds(trip(), demand_rate=-1.0, consequence_cost=1.0)
    with pytest.raises(ValueError, match="demands_reveal"):
        Finds(trip(), demand_rate=1.0, consequence_cost=1.0, demands_reveal=1)
    with pytest.raises(TypeError, match="Finds\\(device"):
        task_value(trip_test(), trip(), three_years)
