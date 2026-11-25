"""Forecasts through a duty in a unit of use (km, cycles, starts).

A life counted in km is read from the duty plan's km meter
(``UsageForecast.reading``), so a km model on a km duty is forecast on
dates as an hours model is on its hours: a constant rate against the same
life in days at continuous running, a weekday duty against a hand count
of its days, units of one kind converted (miles on a km plan), and a unit
of another kind refused.
"""

import datetime as dt

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable
from repyability.schedule import (
    DutyPlan,
    Rate,
    Recurrence,
    Replaces,
    Services,
    Shifts,
    Task,
    UsageForecast,
    failures_by_period,
    task_value,
)

KM = surv.Weibull.from_params([60_000.0, 2.0])


@pytest.fixture(scope="module")
def truck():
    """400 km a day, every day of 2027."""
    return DutyPlan(Rate(400, per="day", meter="km")).forecast(
        "2027-01-01", "2028-01-01"
    )


def test_a_km_life_on_a_km_duty_is_the_life_in_days(truck):
    # 60,000 km at 400 km a day is 150 days: the same Weibull, by month.
    by_km = failures_by_period(
        NonRepairable(KM, unplanned_cost=900.0),
        truck,
        unit="km",
        discount=0.07,
    )
    by_days = failures_by_period(
        NonRepairable(
            surv.Weibull.from_params([150.0, 2.0]), unplanned_cost=900.0
        ),
        UsageForecast.calendar("2027-01-01", "2028-01-01"),
        unit="days",
        discount=0.07,
    )
    assert by_km.unit == "km"
    np.testing.assert_allclose(
        by_km.expected_failures, by_days.expected_failures, rtol=1e-12
    )
    np.testing.assert_allclose(by_km.model_ends[:1], [31 * 400.0])
    assert by_km.npv == pytest.approx(by_days.npv, rel=1e-12)
    assert any("'km' meter" in note for note in by_km.notes)


def test_a_weekday_duty_against_a_hand_count():
    # 500 km on each weekday, none at weekends: by each month's end the
    # truck has run 500 km times the weekdays since 1 January.
    usage = DutyPlan(
        Shifts("Mon-Fri", "06:00", "22:00"), per_hour={"km": 500.0 / 16.0}
    ).forecast("2027-01-01", "2027-07-01")
    part = NonRepairable(KM)
    months = failures_by_period(part, usage, unit="km")
    cumulative, day, weekdays = [], dt.date(2027, 1, 1), 0
    for month in range(2, 8):
        while day < dt.date(2027, month, 1):
            weekdays += day.weekday() < 5
            day += dt.timedelta(days=1)
        cumulative.append(500.0 * weekdays)
    assert cumulative[0] == 500.0 * 21  # January 2027 has 21 weekdays
    np.testing.assert_allclose(months.model_ends, cumulative, rtol=1e-12)
    counts = part.failures(np.array(cumulative)).expected_failures
    np.testing.assert_allclose(
        months.expected_failures,
        np.diff(np.asarray(counts, float), prepend=0.0),
        rtol=0.0,
        atol=1e-12,
    )


def test_units_of_one_kind_convert(truck):
    assert truck.reading("km") == ("km", 1.0)
    assert truck.reading("kilometres") == ("km", 1.0)
    assert truck.reading("miles") == ("km", 1.609344)
    assert truck.reading("days") == ("hours", 24.0)
    miles = NonRepairable(surv.Weibull.from_params([60_000.0 / 1.609344, 2.0]))
    in_miles = failures_by_period(miles, truck, unit="miles")
    in_km = failures_by_period(NonRepairable(KM), truck, unit="km")
    np.testing.assert_allclose(
        in_miles.model_ends * 1.609344, in_km.model_ends, rtol=1e-12
    )
    np.testing.assert_allclose(
        in_miles.expected_failures, in_km.expected_failures, rtol=1e-6
    )
    assert str(truck.moment(800.0, unit="km")) == "2027-01-03 00:00:00"


def test_a_unit_of_another_kind_is_refused():
    hours_only = UsageForecast.calendar("2027-01-01", "2028-01-01")
    with pytest.raises(ValueError, match="a distance cannot be read"):
        failures_by_period(NonRepairable(KM), hours_only, unit="km")
    with pytest.raises(ValueError, match="count of cycles"):
        hours_only.model_time("2027-02-01", unit="cycles")
    with pytest.raises(ValueError, match="unit of time"):
        hours_only.reading("hrs")


def test_starts_are_a_meter_of_every_run():
    usage = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2027-01-04", "2027-01-11"
    )
    assert usage.model_time("2027-01-11", unit="starts") == 5.0


def test_a_replacement_task_by_km(truck):
    task = Task("new tyre", every=Recurrence.every(3, "months", "2027-01-01"))
    part = NonRepairable(KM, planned_cost=500.0, unplanned_cost=4000.0)
    found = task_value(task, Replaces(part), truck, unit="km", discount=0)
    # 1 January (the start), 1 April, 1 July and 1 October: 0, 90, 181
    # and 273 days of 400 km.
    np.testing.assert_allclose(
        found.model_times, [0.0, 90 * 400.0, 181 * 400.0, 273 * 400.0]
    )


def test_a_service_by_km_with_no_effect_is_its_cost(truck):
    from repyability.maintenance import UnitPlan

    task = Task(
        "service",
        every=Recurrence.every(3, "months", "2027-01-01"),
        cost=100.0,
    )
    plan = UnitPlan(KM, repair="instant")
    # On the truck's km (test_duty_rates.py values one with an effect).
    worth = task_value(task, Services(plan), truck, unit="km")
    assert worth.total["value"] == -worth.total["task_cost"]
