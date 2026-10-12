"""Calendar tasks and their due dates (#319, UC-2 of
``repyability/maintenance/README.md``)."""

import json
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from repyability import Monthly, Task, TaskDates

WEEKLY = Task("inspect conveyor", first="2026-07-15", every=timedelta(weeks=1))


def walked(task, start, end):
    """The task's dates by walking the calendar a day at a time: its
    definition, written out plainly."""
    dates, day = [], min(start, task.first)
    while day <= end:
        on_calendar = (day - task.first).days % task.every.days == 0
        if day >= task.first and day >= start and on_calendar:
            dates.append(day)
        day += timedelta(days=1)
    return tuple(dates)


def test_a_weekly_task_through_the_rest_of_2026():
    # UC-2: first done on Wednesday 15 July 2026; asked on Sunday 11
    # October, through 31 December: every Wednesday from 14 October.
    due = WEEKLY.due("2026-10-11", "2026-12-31")
    assert isinstance(due, TaskDates)
    assert due.count == 12
    assert [d.isoformat() for d in due.dates] == [
        "2026-10-14",
        "2026-10-21",
        "2026-10-28",
        "2026-11-04",
        "2026-11-11",
        "2026-11-18",
        "2026-11-25",
        "2026-12-02",
        "2026-12-09",
        "2026-12-16",
        "2026-12-23",
        "2026-12-30",
    ]
    assert all(d.strftime("%A") == "Wednesday" for d in due.dates)
    # With the 13 done since 15 July, 25 in 2026.
    assert WEEKLY.due("2026-07-15", "2026-10-10").count == 13
    assert WEEKLY.due("2026-01-01", "2026-12-31").count == 25


def test_the_dates_are_the_calendar_walked_a_day_at_a_time():
    rng = np.random.default_rng(0)
    origin = date(2026, 1, 1)
    for _ in range(300):
        task = Task(
            "t",
            first=origin + timedelta(days=int(rng.integers(0, 400))),
            every=int(rng.integers(1, 60)),
        )
        start = origin + timedelta(days=int(rng.integers(0, 500)))
        end = start + timedelta(days=int(rng.integers(0, 300)))
        due = task.due(start, end)
        assert due.dates == walked(task, start, end)
        assert due.count == len(due.dates)


def test_both_ends_of_the_period_are_included():
    due = WEEKLY.due("2026-10-14", "2026-10-28")
    assert due.dates == (
        date(2026, 10, 14),
        date(2026, 10, 21),
        date(2026, 10, 28),
    )
    assert WEEKLY.due("2026-10-14", "2026-10-14").count == 1
    assert WEEKLY.due("2026-10-15", "2026-10-20").count == 0


def test_nothing_falls_due_before_the_first_date():
    assert WEEKLY.due("2026-01-01", "2026-07-14").count == 0
    assert WEEKLY.due("2026-07-01", "2026-07-22").dates == (
        date(2026, 7, 15),
        date(2026, 7, 22),
    )


def test_a_weekly_task_from_new_year_falls_due_53_times_in_2026():
    # 2026 has 365 days: 52 weeks and a day, so a task on 1 January comes
    # round again on 31 December.
    task = Task("t", first=date(2026, 1, 1), every=7)
    assert task.due("2026-01-01", "2026-12-31").count == 53


def test_an_interval_is_a_timedelta_or_a_number_of_days():
    assert Task("t", every=7, first="2026-07-15") == Task(
        "t", every=timedelta(days=7), first="2026-07-15"
    )
    assert Task("t", every=7.0, first="2026-07-15").every == timedelta(weeks=1)


@pytest.mark.parametrize(
    "every, error",
    [
        (0, ValueError),
        (-7, ValueError),
        (1.5, ValueError),
        (timedelta(hours=12), ValueError),
        (timedelta(0), ValueError),
        (True, TypeError),
        ("7", TypeError),
        (None, TypeError),
    ],
)
def test_an_interval_must_be_whole_days(every, error):
    with pytest.raises(error, match="every"):
        Task("t", every=every, first="2026-07-15")


@pytest.mark.parametrize(
    "first, error",
    [
        (datetime(2026, 7, 15, 9, 0), TypeError),
        ("15/07/2026", ValueError),
        (20260715, TypeError),
    ],
)
def test_a_date_is_a_date_or_iso_text(first, error):
    with pytest.raises(error, match="first"):
        Task("t", every=7, first=first)


def test_a_period_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="before it starts"):
        WEEKLY.due("2026-12-31", "2026-10-11")


def test_the_name_is_text():
    with pytest.raises(TypeError, match="name"):
        Task(3, every=7, first="2026-07-15")


def test_the_dates_go_through_json():
    due = WEEKLY.due("2026-10-11", "2026-10-31")
    assert json.loads(json.dumps(due.to_dict())) == {
        "task": "inspect conveyor",
        "start": "2026-10-11",
        "end": "2026-10-31",
        "dates": ["2026-10-14", "2026-10-21", "2026-10-28"],
        "count": 3,
        "cost": None,
    }


def test_the_weekly_task_costs_3000_through_the_rest_of_2026():
    # UC-3: 250 each time, the same every time: 12 times, 3000.
    task = Task(
        "inspect conveyor",
        first="2026-07-15",
        every=timedelta(weeks=1),
        cost=250,
    )
    due = task.due("2026-10-11", "2026-12-31")
    assert due.count == 12
    assert due.cost == 3000.0
    assert json.loads(json.dumps(due.to_dict()))["cost"] == 3000.0


def test_the_cost_is_the_count_times_the_cost_each_time():
    task = Task("t", every=7, first="2026-07-15", cost=12.5)
    for end in ("2026-07-14", "2026-07-15", "2026-08-31", "2027-07-15"):
        due = task.due("2026-07-01", end)
        assert due.cost == due.count * 12.5


def test_a_task_with_no_cost_has_none_and_a_free_one_costs_0():
    assert WEEKLY.cost is None
    assert WEEKLY.due("2026-10-11", "2026-12-31").cost is None
    free = Task("t", every=7, first="2026-07-15", cost=0)
    assert free.due("2026-10-11", "2026-12-31").cost == 0.0


@pytest.mark.parametrize(
    "cost", [-1.0, float("inf"), float("nan"), True, "250", [250]]
)
def test_a_cost_must_be_a_number_at_least_0(cost):
    with pytest.raises(ValueError, match="cost"):
        Task("t", every=7, first="2026-07-15", cost=cost)


# Monthly rules (UC-4).


def month_length(day):
    following = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return (following - date(day.year, day.month, 1)).days


def on_rule(rule, day):
    """Whether ``day`` is one of a monthly rule's dates: the rule's
    definition, written out plainly."""
    if rule.weekday is not None:
        if day.strftime("%A") != rule.weekday:
            return False
        if rule.nth > 0:
            return (day.day - 1) // 7 == rule.nth - 1
        return (month_length(day) - day.day) // 7 == -rule.nth - 1
    if rule.short_months == "skip":
        return day.day == rule.day
    return day.day == min(rule.day, month_length(day))


def walked_monthly(task, start, end):
    dates, day = [], max(start, task.starting or start)
    while day <= end:
        if on_rule(task.on, day):
            dates.append(day)
        day += timedelta(days=1)
    return tuple(dates)


def test_the_first_tuesday_and_the_30th_through_the_rest_of_2026():
    # UC-4: from Sunday 11 October to 31 December 2026.
    tuesdays = Task("t", on=Monthly(weekday="Tuesday", nth=1))
    assert tuesdays.due("2026-10-11", "2026-12-31").dates == (
        date(2026, 11, 3),
        date(2026, 12, 1),
    )
    thirtieths = Task("t", on=Monthly(day=30))
    assert thirtieths.due("2026-10-11", "2026-12-31").dates == (
        date(2026, 10, 30),
        date(2026, 11, 30),
        date(2026, 12, 30),
    )


def test_a_short_month_has_its_last_day_or_is_skipped():
    last = Task("t", on=Monthly(day=30))
    skip = Task("t", on=Monthly(day=30, short_months="skip"))
    assert last.on.short_months == "last"
    february = ("2027-02-01", "2027-02-28")
    assert last.due(*february).dates == (date(2027, 2, 28),)
    assert skip.due(*february).count == 0
    # 2028 is a leap year.
    assert last.due("2028-02-01", "2028-02-29").dates == (date(2028, 2, 29),)
    # The 31st: the last day of every month, or only the long ones.
    longest = Task("t", on=Monthly(day=31))
    assert longest.due("2026-01-01", "2026-12-31").count == 12
    only_long = Task("t", on=Monthly(day=31, short_months="skip"))
    assert only_long.due("2026-01-01", "2026-12-31").count == 7


def test_the_monthly_dates_are_the_calendar_walked_a_day_at_a_time():
    rng = np.random.default_rng(1)
    origin = date(2026, 1, 1)
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    for _ in range(200):
        if rng.random() < 0.5:
            rule = Monthly(
                day=int(rng.integers(1, 32)),
                short_months=str(rng.choice(["last", "skip"])),
            )
        else:
            rule = Monthly(
                weekday=str(rng.choice(weekdays)),
                nth=int(rng.choice([1, 2, 3, 4, -1, -2, -3, -4])),
            )
        # Half the tasks apply from a date, half whenever asked.
        starting = origin + timedelta(days=int(rng.integers(0, 1500)))
        task = Task(
            "t", on=rule, starting=starting if rng.random() < 0.5 else None
        )
        start = origin + timedelta(days=int(rng.integers(0, 1800)))
        end = start + timedelta(days=int(rng.integers(0, 400)))
        assert task.due(start, end).dates == walked_monthly(task, start, end)


def test_the_last_weekday_of_the_month():
    fridays = Task("t", on=Monthly(weekday="friday", nth=-1))
    assert fridays.on.weekday == "Friday"
    assert fridays.due("2026-10-01", "2026-12-31").dates == (
        date(2026, 10, 30),
        date(2026, 11, 27),
        date(2026, 12, 25),
    )


def test_a_monthly_task_costs_its_count_times_its_cost():
    task = Task("t", on=Monthly(day=30), cost=100)
    assert task.due("2026-10-11", "2026-12-31").cost == 300.0


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"day": 30, "weekday": "Tuesday", "nth": 1},
        {"day": 0},
        {"day": 32},
        {"day": 15.5},
        {"day": True},
        {"day": 30, "nth": 1},
        {"day": 30, "short_months": "first"},
        {"weekday": "Tues", "nth": 1},
        {"weekday": 1, "nth": 1},
        {"weekday": "Tuesday"},
        {"weekday": "Tuesday", "nth": 5},
        {"weekday": "Tuesday", "nth": 0},
        {"weekday": "Tuesday", "nth": 1.5},
        {"weekday": "Tuesday", "nth": 1, "short_months": "skip"},
    ],
)
def test_a_monthly_rule_is_checked(arguments):
    with pytest.raises(ValueError):
        Monthly(**arguments)


def test_an_interval_or_a_rule_is_needed():
    with pytest.raises(TypeError, match="every"):
        Task("t", every="monthly", first="2026-07-15")
    with pytest.raises(TypeError, match="one of them"):
        Task("t")
    with pytest.raises(TypeError, match="one of them"):
        Task("t", every=7, first="2026-07-15", on=Monthly(day=30))


def test_a_monthly_rule_goes_in_on_and_names_its_day():
    with pytest.raises(TypeError, match="on=Monthly"):
        Task("t", every=Monthly(day=30), first="2026-07-15")
    with pytest.raises(TypeError):
        Monthly(30)  # the day by name: Monthly(day=30)
    with pytest.raises(TypeError, match="on must be"):
        Task("t", on=7)


def test_a_rule_applies_from_starting_and_an_interval_from_first():
    task = Task("t", on=Monthly(day=30), starting="2026-11-15")
    assert task.due("2026-10-01", "2026-12-31").dates == (
        date(2026, 11, 30),
        date(2026, 12, 30),
    )
    with pytest.raises(ValueError, match="starting="):
        Task("t", on=Monthly(day=30), first="2026-07-15")
    with pytest.raises(ValueError, match="first="):
        Task("t", every=7, first="2026-07-15", starting="2026-07-15")
    with pytest.raises(TypeError, match="first time"):
        Task("t", every=7)


def test_every_argument_but_the_name_is_given_by_name():
    with pytest.raises(TypeError):
        Task("t", "2026-07-15", 7)
