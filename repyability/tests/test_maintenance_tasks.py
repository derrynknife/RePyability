"""Calendar tasks and their due dates (#319, UC-2 of
``repyability/maintenance/README.md``)."""

import json
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from repyability import Task, TaskDates

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
    assert Task("t", "2026-07-15", 7) == Task(
        "t", "2026-07-15", timedelta(days=7)
    )
    assert Task("t", "2026-07-15", 7.0).every == timedelta(weeks=1)


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
        Task("t", "2026-07-15", every)


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
        Task("t", first, 7)


def test_a_period_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="before it starts"):
        WEEKLY.due("2026-12-31", "2026-10-11")


def test_the_name_is_text():
    with pytest.raises(TypeError, match="name"):
        Task(3, "2026-07-15", 7)


def test_the_dates_go_through_json():
    due = WEEKLY.due("2026-10-11", "2026-10-31")
    assert json.loads(json.dumps(due.to_dict())) == {
        "task": "inspect conveyor",
        "start": "2026-10-11",
        "end": "2026-10-31",
        "dates": ["2026-10-14", "2026-10-21", "2026-10-28"],
        "count": 3,
    }
