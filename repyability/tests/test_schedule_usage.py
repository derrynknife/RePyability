"""The map between moments and meters (``UsageForecast``), limits by use
(``Usage``) and tasks that fall due by use.

The worked examples are the use cases' (``repyability/schedule/README.md``),
each checked by hand in the comments.
"""

import json
from datetime import date, datetime

import numpy as np
import pytest

from repyability.schedule import (
    DutyPlan,
    Rate,
    Recurrence,
    Runs,
    Shifts,
    Task,
    Usage,
    UsageForecast,
    work_by_period,
)


@pytest.fixture
def pump():
    """Pump P-101: Mon-Fri 06:00-22:00 (16 h a run, 80 h a week), 12,340 h
    at the start of Monday 16 Nov 2026."""
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    return plan.forecast(
        "2026-11-16", "2027-12-01", readings={"hours": 12_340}
    )


def test_pump_hours_by_date_and_month(pump):
    # to 1 Jan 2027: 11 weekdays in Nov (from the 16th) + 23 in Dec, 34 x
    # 16 = 544 h
    assert pump.at("2027-01-01")["hours"] == 12_884.0
    months = pump.by_period("month")
    # weekdays per month x 16 h
    assert months.usage["hours"] == (
        176.0,
        368.0,
        336.0,
        320.0,
        368.0,
        352.0,
        336.0,
        352.0,
        352.0,
        352.0,
        352.0,
        336.0,
        352.0,
    )
    assert months.at_end["hours"][-1] == 16_692.0
    assert str(months.starts[0]) == "2026-11-16 00:00:00"
    assert str(months.starts[1]) == "2026-12-01 00:00:00"
    # Mondays to Fridays start one run each: 11 in Nov
    assert months.usage["starts"][0] == 11.0


def test_pump_service_due_mid_run(pump):
    # 14,000 h is 1,660 h on: 20 weeks (1,600 h) to Mon 5 Apr 2027, then
    # 16 + 16 + 16 h to Wed 22:00 and 12 h into Thursday's run: 18:00.
    c = pump.when(hours=14_000)
    assert c.moment == datetime(2027, 4, 8, 18, 0)
    assert c.next_stop_after == datetime(2027, 4, 8, 22, 0)
    assert c.last_stop_before == datetime(2027, 4, 7, 22, 0)
    assert c.exact and c.layer == "background"
    assert c.earliest == c.latest == c.moment
    # 16,000 h: 25 weeks later, Thu 30 Sep 2027 18:00
    due = pump.due(Usage(2000, "hours"), last_done_reading=12_000)
    assert due.moments == (datetime(2027, 4, 8, 18), datetime(2027, 9, 30, 18))
    assert due.readings == (14_000.0, 16_000.0)
    assert due.count == 2 and not due.overdue
    # at a flat 80 h a week it would show as Saturday 10 April, a day the
    # pump doesn't run: the plan matters
    flat = DutyPlan(Rate(80, per="week")).forecast(
        "2026-11-16", "2027-12-01", readings={"hours": 12_340}
    )
    assert flat.when(hours=14_000).moment.date() == date(2027, 4, 10)


def test_map_both_ways(pump):
    # model time is operating time since the start
    assert pump.model_time("2026-11-23") == 80.0
    assert pump.model_time("2026-11-23", unit="days") == pytest.approx(80 / 24)
    assert pump.model_time("2026-11-23", unit="minutes") == 4800.0
    moments = [
        datetime(2027, 1, 4, 7, 15),
        datetime(2027, 6, 3, 21, 59),
        datetime(2027, 9, 30, 18),
    ]
    times = pump.model_time(moments)
    assert isinstance(times, np.ndarray)
    for moment, t in zip(moments, times):
        assert pump.moment(t) == moment  # running then: the map inverts
    assert pump.moment(0.0) == datetime(2026, 11, 16)
    assert pump.moment(10**7) is None
    # in a stop, the first moment the hours were reached: when the run ended
    sat = pump.model_time("2026-11-21 12:00")
    assert pump.moment(sat) == datetime(2026, 11, 20, 22)


def test_duty_rate_schedule(pump):
    week = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2026-11-16", "2026-11-23"
    )
    assert week.duty() == pytest.approx(80 / 168)
    assert week.rate("starts", per="hours") == 1 / 16
    s = week.schedule()
    # off to 06:00, on to 22:00, off to 06:00 the next day, ...
    assert s.times[:4] == (0.0, 6.0, 22.0, 30.0)
    assert s.shares[:4] == (0.0, 1.0, 0.0, 1.0)
    assert week.schedule(unit="days").times[1] == 0.25


def test_generator_concrete_run():
    # On 12:31 Sat 14 Nov, off 03:27 Sun 15 Nov: 14 h 56 min. From 486.2 h,
    # 500 h is 13.8 h = 13 h 48 min on: 02:19, in the run; it reads 501.13
    # h after it.
    plan = DutyPlan(Runs([("2026-11-14 12:31", "2026-11-15 03:27")]))
    f = plan.forecast("2026-11-14", "2026-11-16", readings={"hours": 486.2})
    c = f.when(hours=500)
    assert c.moment == datetime(2026, 11, 15, 2, 19)
    assert c.next_stop_after == datetime(2026, 11, 15, 3, 27)
    assert c.last_stop_before is None
    assert f.at("2026-11-16")["hours"] == pytest.approx(486.2 + 14 + 56 / 60)
    assert round(f.at("2026-11-16")["hours"], 2) == 501.13


def test_conveyor_general_rate_and_range():
    # 3 h a day from 1,180 h at the start of 1 Dec 2026; due at 1,250 h:
    # 70/3 = 23 1/3 days on, 24 Dec 08:00; 1,500 h: 320/3 = 106 2/3 days,
    # 17 Mar 2027 16:00; then every 250/3 = 83 1/3 days.
    plan = DutyPlan(Rate(3, per="day"))
    f = plan.forecast("2026-12-01", "2027-12-01", readings={"hours": 1_180})
    due = f.due(Usage(250, "hours"), last_done_reading=1_000)
    assert due.moments[:2] == (
        datetime(2026, 12, 24, 8),
        datetime(2027, 3, 17, 16),
    )
    assert (due.moments[2] - due.moments[1]).total_seconds() == pytest.approx(
        250 / 3 * 86400
    )
    assert due.exact[0] is False  # a general rate is an assumption
    assert due.next_stops_after[0] is None
    # 2 to 4 h a day: 70/4 = 17.5 days (18 Dec 12:00) to 70/2 = 35 days
    # (5 Jan 00:00); the middle at 3 h a day
    ranged = DutyPlan(Rate((2, 4), per="day")).forecast(
        "2026-12-01", "2027-12-01", readings={"hours": 1_180}
    )
    c = ranged.when(hours=1_250)
    assert c.moment == datetime(2026, 12, 24, 8)
    assert c.earliest == datetime(2026, 12, 18, 12)
    assert c.latest == datetime(2027, 1, 5)
    assert not c.exact
    r = ranged.at("2026-12-11")
    assert (r.low["hours"], r["hours"], r.high["hours"]) == pytest.approx(
        (1200.0, 1210.0, 1220.0)
    )
    months = ranged.by_period("month")
    assert months.low["hours"][0] == pytest.approx(62.0)
    assert months.high["hours"][0] == pytest.approx(124.0)


def test_layered_forecast_exact_then_a_range():
    # Exact runs to 16 Nov 03:30, then 2 to 4 h a day: a due in the runs is
    # exact, one in the rate a range.
    plan = DutyPlan(
        background=Rate((2, 4), per="day"),
        specific=[
            Runs(
                [
                    ("2026-11-14 12:31", "2026-11-15 03:27"),
                    ("2026-11-15 12:30", "2026-11-16 03:30"),
                ],
                from_="2026-11-14",
            )
        ],
    )
    f = plan.forecast("2026-11-14", "2027-11-14", readings={"hours": 4_820})
    in_runs = f.when(hours=4_830)
    assert in_runs.exact and in_runs.layer == "specific"
    assert in_runs.moment == datetime(2026, 11, 14, 22, 31)
    later = f.when(hours=4_900)
    assert later.layer == "background" and not later.exact
    assert later.earliest < later.moment < later.latest


def test_readings_anchor_and_meters_not_read():
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    # read at the end of Monday's run, the forecast from Monday's start
    f = plan.forecast(
        "2026-11-16",
        "2026-11-30",
        readings={"hours": 1_016},
        read_at="2026-11-16 22:00",
    )
    assert f.at("2026-11-16")["hours"] == 1_000.0
    assert f.at("2026-11-16 06:00")["starts"] == 1.0  # Monday's run starts
    assert any("'starts' was not read" in n for n in f.notes)
    # read after the forecast's start, before it: counted back
    f = plan.forecast(
        "2026-11-17",
        "2026-11-30",
        readings={"hours": 1_000},
        read_at="2026-11-16",
    )
    assert f.at("2026-11-17")["hours"] == 1_016.0
    assert str(f.read_at) == "2026-11-16 00:00:00"


def test_identity_map_with_no_plan():
    f = UsageForecast.calendar(
        "2027-01-01", "2027-02-01", readings={"hours": 10}
    )
    assert f.at("2027-01-02")["hours"] == 34.0
    assert f.model_time("2027-01-11", unit="days") == 10.0
    assert f.duty() == 1.0
    # with a zone, real hours: Sydney's 4 Oct 2026 is 23 h long
    f = UsageForecast.calendar(
        "2026-10-04", "2026-10-06", time_zone="Australia/Sydney"
    )
    assert f.at("2026-10-05")["hours"] == 23.0


def test_starts_and_stops():
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    f = plan.forecast("2026-11-16", "2026-11-30")
    c = f.when(starts=3)  # the third run starts Wed 06:00
    assert c.moment == datetime(2026, 11, 18, 6)
    assert c.next_stop_after == datetime(2026, 11, 18, 22)
    assert c.last_stop_before == datetime(2026, 11, 17, 22)
    assert f.due(Usage(5, "starts"), 0).moments == (
        datetime(2026, 11, 20, 6),
        datetime(2026, 11, 27, 6),
    )


def test_overdue_and_not_reached():
    f = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2026-11-16", "2026-11-30", readings={"hours": 12_340}
    )
    c = f.when(hours=12_000)
    assert c.already_reached and c.moment is None
    assert f.when(hours=99_999).moment is None
    due = f.due(Usage(500, "hours"), last_done_reading=11_800)
    # due at 12,300 h, before the start: overdue; 12,800 h is not reached
    # by 30 Nov (12,500 h then)
    assert due.overdue and due.moments[0] is None
    assert due.readings == (12_300.0,)


def test_refusals(pump):
    with pytest.raises(ValueError, match="outside the forecast"):
        pump.at("2030-01-01")
    with pytest.raises(ValueError, match="no meter 'km'"):
        pump.when(km=10)
    with pytest.raises(TypeError, match="one meter"):
        pump.when(hours=1, starts=2)
    with pytest.raises(ValueError, match="no meter"):
        DutyPlan(Shifts("Mon", "06:00", "07:00")).forecast(
            "2026-11-16", "2026-11-30", readings={"km": 3}
        )
    with pytest.raises(ValueError, match="after start"):
        DutyPlan(Shifts("Mon", "06:00", "07:00")).forecast(
            "2026-11-16", "2026-11-16"
        )
    with pytest.raises(ValueError, match="period must be"):
        pump.by_period("fortnight")
    with pytest.raises(ValueError, match="unit must be"):
        pump.model_time("2026-11-20", unit="furlongs")


def test_results_are_plain_json(pump):
    for result in (
        pump.when(hours=14_000),
        pump.due(Usage(2000), 12_000),
        pump.by_period("quarter"),
        pump.schedule(),
        pump.stretches[0],
    ):
        data = json.loads(json.dumps(result.to_dict()))
        assert isinstance(data, dict)
    assert json.loads(json.dumps(pump.at("2027-01-01").to_dict()))[
        "readings"
    ] == {
        "hours": 12_884.0,
        "starts": 34.0,
    }


# --- tasks by use ----------------------------------------------------------


def test_task_by_use(pump):
    service = Task(
        "Service",
        every=Usage(2000, "hours"),
        last_done_reading=12_000,
        cost=900.0,
    )
    due = service.due("2026-11-16", "2027-11-30", usage=pump)
    assert due.dates == (date(2027, 4, 8), date(2027, 9, 30))
    assert due.moments == (datetime(2027, 4, 8, 18), datetime(2027, 9, 30, 18))
    assert due.readings == (14_000.0, 16_000.0)
    assert due.stops_after == (
        datetime(2027, 4, 8, 22),
        datetime(2027, 9, 30, 22),
    )
    assert due.limits == ("every", "every") and due.cost == 1800.0
    work = work_by_period(
        [service], "month", "2026-11-16", "2027-11-30", usage=pump
    )
    assert work.count == (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0)
    with pytest.raises(ValueError, match="give usage="):
        service.due("2026-11-16", "2027-11-30")
    with pytest.raises(ValueError, match="last_done_reading"):
        Task("x", every=Usage(10)).due("2026-11-16", "2026-11-20", usage=pump)


def test_task_reading_from_last_done(pump):
    # last done on Mon 23 Nov 2026 00:00, at 12,420 h: next at 14,420 h
    task = Task("Service", every=Usage(2000), last_done="2026-11-23")
    due = task.due("2026-11-16", "2027-11-30", usage=pump)
    assert due.readings[0] == 14_420.0


def test_two_limits_the_calendar_first():
    # A standby generator runs 15 h a month; its oil change is every 250 h
    # or 12 months. 250 h takes 16 2/3 months, so the 12-month limit sets
    # every date; the hours would be reached on 4 Apr 2028 (~4.7 months
    # after the first calendar date).
    usage = DutyPlan(Rate(15, per="month")).forecast(
        "2026-11-14", "2029-01-01", readings={"hours": 100}
    )
    oil = Task(
        "Oil change",
        every=Usage(250),
        also=Recurrence.every(12, "months", "2026-11-14"),
        last_done="2026-11-14",
        last_done_reading=100,
    )
    due = oil.due("2026-11-14", "2028-12-31", usage=usage)
    assert due.dates == (date(2027, 11, 14), date(2028, 11, 14))
    assert due.limits == ("also", "also")
    assert usage.when(hours=350).moment == datetime(2028, 4, 4)


def test_two_limits_the_hours_first(pump):
    # 2,000 h or 12 months on the pump: the hours come first (8 Apr 2027)
    task = Task(
        "Service",
        every=Usage(2000),
        also=Recurrence.every(12, "months", "2026-11-16"),
        last_done="2026-11-16",
        last_done_reading=12_340,
    )
    due = task.due("2026-11-16", "2027-11-30", usage=pump)
    # 14,340 h is 2,000 h on, 25 whole weeks: reached at the end of
    # Friday 7 May's run, 22:00
    assert due.limits[0] == "every"
    assert due.readings[0] == 14_340.0
    assert due.moments[0] == datetime(2027, 5, 7, 22)


def test_task_round_trips_with_usage():
    task = Task(
        "Service",
        every=Usage(2000, "hours"),
        also=Recurrence.every(12, "months", "2026-11-16"),
        last_done_reading=12_000,
    )
    data = json.loads(json.dumps(task.to_dict()))
    assert data["every"] == {
        "kind": "usage",
        "amount": 2000.0,
        "meter": "hours",
    }
    assert Task.from_dict(data) == task
    assert Usage.from_dict(Usage(5, "starts").to_dict()) == Usage(5, "starts")


def test_uc10_pump_every_500_hours():
    # Mon-Fri 06:00-22:00, 12,340 h on Mon 12 Oct 2026, every 500 h from
    # 12,000: 12,500 h is 160 h on, ten runs (to Fri 23 Oct 22:00);
    # 13,000 h is 660 h on, 41 runs and 4 h (Tue 8 Dec 10:00); 13,500 h is
    # 1,160 h on, 72 runs and 8 h (Wed 20 Jan 2027 14:00).
    usage = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2026-10-12", "2027-02-01", readings={"hours": 12_340}
    )
    due = usage.due(Usage(500, "hours"), last_done_reading=12_000)
    assert due.moments == (
        datetime(2026, 10, 23, 22),
        datetime(2026, 12, 8, 10),
        datetime(2027, 1, 20, 14),
    )
