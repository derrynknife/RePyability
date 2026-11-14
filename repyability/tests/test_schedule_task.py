"""Tasks on dates and work by period (repyability.schedule, UC-2, UC-3,
UC-30 and UC-31 of ``repyability/schedule/README.md``)."""

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from repyability.schedule import (
    Recurrence,
    Task,
    TaskDue,
    WorkByPeriod,
    _task,
    work_by_period,
)

D = date.fromisoformat
WEEKLY = Recurrence.every(1, "weeks", "2026-07-15")


def iso(days):
    return [d.isoformat() for d in days]


def test_the_weekly_task_falls_due_12_times_and_costs_3000():
    # UC-2 and UC-3: $250 a time, from 11 October through 31 December.
    task = Task("inspect conveyor", every=WEEKLY, cost=250)
    due = task.due("2026-10-11", "2026-12-31")
    assert isinstance(due, TaskDue)
    assert due.count == 12 and due.cost == 3000.0
    assert due.dates[0] == D("2026-10-14") and due.dates[-1] == D("2026-12-30")
    assert due.done == due.dates and set(due.limits) == {"every"}
    assert due.late == (0,) * 12 and all(due.on_time)
    assert not due.overdue and due.hours is None


def test_cost_and_hours_are_the_count_times_each():
    task = Task("t", every=WEEKLY, cost=12.5, duration=1.5)
    for end in ("2026-07-14", "2026-07-15", "2026-08-31", "2027-07-15"):
        due = task.due("2026-07-01", end)
        assert due.cost == due.count * 12.5
        assert due.hours == due.count * 1.5
    free = Task("t", every=WEEKLY, cost=0)
    assert free.due("2026-10-11", "2026-12-31").cost == 0.0


def test_a_pattern_task_is_on_its_rule_dates():
    task = Task("check pumps", every=Recurrence.monthly(weekday="Tue", nth=1))
    due = task.due("2026-10-11", "2026-12-31")
    assert iso(due.dates) == ["2026-11-03", "2026-12-01"]
    assert not due.resets


def test_an_interval_counts_from_when_it_was_last_done():
    rule = Recurrence.every(3, "months", "2026-01-15")
    # Last done on 20 September: due 20 December, then every 3 months
    # from there (the anchor's 15th no longer matters).
    task = Task("service", every=rule, last_done="2026-09-20")
    due = task.due("2026-11-01", "2027-09-30")
    assert iso(due.dates) == [
        "2026-12-20",
        "2027-03-20",
        "2027-06-20",
        "2027-09-20",
    ]
    assert due.resets
    # Kept to its own dates, it is on the 15th: 15 October was missed, so
    # it is overdue on 1 November, then back on its dates.
    fixed = Task("service", every=rule, last_done="2026-09-20", resets=False)
    kept = fixed.due("2026-11-01", "2027-09-30")
    assert kept.overdue and not kept.resets
    assert iso(kept.dates) == [
        "2026-10-15",
        "2027-01-15",
        "2027-04-15",
        "2027-07-15",
    ]
    assert iso(kept.done)[0] == "2026-11-01"


def test_month_ends_hold_while_each_is_done_on_its_date():
    # Done on its due date, an interval from 31 January keeps its day: it
    # is not carried to the 28th after February.
    task = Task("t", every=Recurrence.every(1, "months", "2027-01-31"))
    assert iso(task.due("2027-01-01", "2027-05-31").dates) == [
        "2027-01-31",
        "2027-02-28",
        "2027-03-31",
        "2027-04-30",
        "2027-05-31",
    ]


def test_visits_round_each_due_date_up_and_say_how_late():
    # Done at the Tuesday visit on or after each due date; an interval
    # counts again from the visit.
    task = Task(
        "service",
        every=Recurrence.every(3, "months", "2026-01-15"),
        last_done="2026-09-20",
        visits=Recurrence.weekly("Tue"),
        tolerance=3,
    )
    due = task.due("2026-11-01", "2027-12-31")
    assert iso(due.dates) == [
        "2026-12-20",
        "2027-03-22",
        "2027-06-23",
        "2027-09-29",
    ]
    assert iso(due.done) == [
        "2026-12-22",
        "2027-03-23",
        "2027-06-29",
        "2027-10-05",
    ]
    assert due.late == (2, 1, 6, 6)
    assert due.on_time == (True, True, False, False)
    for d, done in zip(due.dates, due.done):
        assert done.weekday() == 1 and 0 <= (done - d).days < 7


def test_a_fixed_rule_waits_for_its_visit_without_drifting():
    task = Task(
        "t",
        every=Recurrence.monthly(day=1),
        visits=Recurrence.weekly("Thu"),
    )
    due = task.due("2027-01-01", "2027-04-30")
    assert iso(due.dates) == [
        "2027-01-01",
        "2027-02-01",
        "2027-03-01",
        "2027-04-01",
    ]
    assert iso(due.done) == [
        "2027-01-07",
        "2027-02-04",
        "2027-03-04",
        "2027-04-01",
    ]


def test_two_limits_whichever_comes_first():
    # Every 12 months or the first Saturday of every 5th month: the
    # pattern always comes first, and the result says which limit set
    # each date.
    task = Task(
        "oil change",
        every=Recurrence.every(12, "months", "2026-01-01"),
        also=Recurrence.monthly(
            weekday="Saturday", nth=1, every=5, anchor="2026-01-01"
        ),
        last_done="2026-06-01",
    )
    due = task.due("2026-06-01", "2028-12-31")
    assert iso(due.dates) == [
        "2026-06-06",
        "2026-11-07",
        "2027-04-03",
        "2027-09-04",
        "2028-02-05",
        "2028-07-01",
        "2028-12-02",
    ]
    assert set(due.limits) == {"also"}
    # Every 6 months or every 12: the 6 always comes first, from last done.
    both = Task(
        "t",
        every=Recurrence.every(12, "months", "2026-01-10"),
        also=Recurrence.every(6, "months", "2026-01-10"),
        last_done="2026-03-01",
    )
    due = both.due("2026-03-01", "2027-12-31")
    assert iso(due.dates) == ["2026-09-01", "2027-03-01", "2027-09-01"]
    assert due.limits == ("also",) * 3
    assert "whichever comes first" in both.describe()


def test_two_limits_the_first_wins_at_each_step():
    # Every 30 days or the 1st of the month: worked by hand from last done
    # on 10 January 2027.
    task = Task(
        "t",
        every=Recurrence.every(30, "days", "2027-01-01"),
        also=Recurrence.monthly(day=1),
        last_done="2027-01-10",
    )
    due = task.due("2027-01-10", "2027-04-30")
    # 1 Feb (the 1st, before 9 Feb); then 1 Mar (before 3 Mar); 31 Mar
    # (30 days from 1 Mar, before 1 Apr); then 1 Apr; 1 May is outside.
    assert iso(due.dates) == [
        "2027-02-01",
        "2027-03-01",
        "2027-03-31",
        "2027-04-01",
    ]
    assert due.limits == ("also", "also", "every", "also")


def test_an_overdue_task_is_done_at_the_start():
    task = Task(
        "t",
        every=Recurrence.every(1, "months", "2026-01-01"),
        last_done="2026-03-10",
    )
    due = task.due("2026-06-01", "2026-08-31")
    assert due.overdue
    assert iso(due.dates) == ["2026-04-10", "2026-07-01", "2026-08-01"]
    assert iso(due.done) == ["2026-06-01", "2026-07-01", "2026-08-01"]
    assert due.late[0] == 52 and not due.on_time[0]
    assert due.notes[0].startswith("Overdue at 2026-06-01")
    assert due.count == 3


def test_the_notes_say_what_the_rule_left_out():
    task = Task("t", every=Recurrence.monthly(weekday="Sat", nth=5))
    due = task.due("2027-01-01", "2027-12-31")
    assert due.count == 4
    assert due.notes[0].startswith("8 months have no fifth Saturday")


def test_the_dates_are_the_rule_walked_when_done_on_time():
    # With no last done and no visits, a task's dates are its rule's.
    rng = np.random.default_rng(2)
    origin = date(2026, 1, 1)
    for _ in range(100):
        anchor = origin + timedelta(days=int(rng.integers(0, 400)))
        rule = Recurrence.every(
            int(rng.integers(1, 5)),
            str(rng.choice(["days", "weeks", "months"])),
            anchor,
        )
        start = origin + timedelta(days=int(rng.integers(0, 600)))
        end = start + timedelta(days=int(rng.integers(0, 400)))
        assert Task("t", every=rule).due(start, end).dates == rule.dates(
            start, end
        )


def test_a_task_goes_through_json_and_back():
    task = Task(
        "service",
        every=Recurrence.every(3, "months", "2026-01-15"),
        also=Recurrence.yearly(month=11, weekday="Tuesday", nth=1),
        last_done="2026-09-20",
        visits=Recurrence.weekly("Tue"),
        tolerance=3,
        cost=480.0,
        duration=2.5,
        resets=True,
    )
    data = json.loads(json.dumps(task.to_dict()))
    assert data["kind"] == "task" and data["every"]["kind"] == "recurrence"
    assert Task.from_dict(data) == task
    due = task.due("2026-11-01", "2027-12-31")
    plain = json.loads(json.dumps(due.to_dict()))
    # The overhaul (3 Nov, a Tuesday visit) comes before 20 December.
    assert plain["dates"][:2] == ["2026-11-03", "2027-02-03"]
    assert plain["limits"][:2] == ["also", "every"]
    assert plain["done"][1] == "2027-02-09"
    with pytest.raises(ValueError, match="later version"):
        Task.from_dict({**data, "version": 99})
    with pytest.raises(ValueError, match="not a saved task"):
        Task.from_dict({"name": "x"})


def test_describe():
    task = Task(
        "Inspect",
        every=Recurrence.weekly("Mon"),
        visits=Recurrence.weekly("Mon-Fri"),
    )
    assert task.describe() == (
        "Inspect: every Monday; done at the visits (every weekday, Monday "
        "to Friday)"
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"every": 7}, TypeError, "every must be a Recurrence"),
        ({"every": WEEKLY, "also": "monthly"}, TypeError, "also must be"),
        ({"every": WEEKLY, "cost": -1}, ValueError, "cost"),
        ({"every": WEEKLY, "cost": "250"}, ValueError, "cost"),
        ({"every": WEEKLY, "cost": float("nan")}, ValueError, "cost"),
        ({"every": WEEKLY, "duration": True}, ValueError, "duration"),
        ({"every": WEEKLY, "tolerance": -1}, ValueError, "tolerance"),
        ({"every": WEEKLY, "tolerance": 1.5}, ValueError, "tolerance"),
        ({"every": WEEKLY, "visits": "Tue"}, TypeError, "visits"),
        ({"every": WEEKLY, "resets": 1}, TypeError, "resets"),
        ({"every": WEEKLY, "last_done": "15/07/2026"}, ValueError, "YYYY"),
    ],
)
def test_task_inputs_are_checked(kwargs, error, match):
    with pytest.raises(error, match=match):
        Task("t", **kwargs)


def test_the_name_is_text_and_the_rest_by_name():
    with pytest.raises(TypeError, match="name"):
        Task(3, every=WEEKLY)
    with pytest.raises(TypeError):
        Task("t", WEEKLY)
    with pytest.raises(ValueError, match="before it starts"):
        Task("t", every=WEEKLY).due("2026-12-31", "2026-10-11")
    with pytest.raises(ValueError, match="after the period"):
        Task("t", every=WEEKLY, last_done="2027-01-05").due(
            "2026-10-11", "2026-12-31"
        )


# ---------------------------------------------------------------- the hook


@dataclass(frozen=True)
class EveryNHours:
    """A stand-in for a limit by use, through the hook alone: due every
    ``amount`` hours of a meter that runs 10 hours a day from midnight,
    read from the context ``Task._due`` is given."""

    amount: float

    def _task_limit(self, key, resets):
        return _Bound(self.amount, key)


@dataclass(frozen=True)
class _Bound:
    amount: float
    key: str
    resets: bool = True

    def describe(self):
        return f"every {self.amount:g} hours"

    def to_dict(self):
        return {"kind": "every-n-hours", "amount": self.amount}

    def bind(self, task, start, end, context):
        return _Clock(self, context["read_at"])

    @classmethod
    def from_dict(cls, data):
        return EveryNHours(data["amount"])


class _Clock:
    def __init__(self, limit, read_at):
        self.limit, self.origin = limit, datetime.combine(read_at, time())

    def _at(self, hours):
        days, rest = divmod(hours, 10.0)
        return self.origin + timedelta(days=days, hours=rest)

    def _hours(self, moment):
        days = (moment - self.origin).days
        into = (moment - self.origin - timedelta(days=days)).seconds / 3600
        return days * 10.0 + min(into, 10.0)

    def first(self, last_done):
        return _task._Due(
            self._at(self.limit.amount), self.limit.key, self.limit.amount
        )

    def after(self, done, due):
        reading = self._hours(done) + self.limit.amount
        return _task._Due(self._at(reading), self.limit.key, reading)


def test_a_limit_by_use_slots_in_through_the_hook():
    task = Task(
        "service",
        every=EveryNHours(25.0),
        also=Recurrence.monthly(day=10),
    )
    due = task._due("2027-01-01", "2027-01-31", {"read_at": D("2027-01-01")})
    # 25 h at 10 h a day: 3 Jan 05:00, then 50 h at the end of 5 Jan's
    # running (this stand-in puts it at midnight, 6 Jan), ...;
    # the 10th of the month comes first once, then the hours restart.
    assert iso(due.dates)[:3] == ["2027-01-03", "2027-01-06", "2027-01-08"]
    assert "also" in due.limits
    _task.LIMIT_KINDS["every-n-hours"] = _Bound
    try:
        back = Task.from_dict(task.to_dict())
    finally:
        del _task.LIMIT_KINDS["every-n-hours"]
    assert back == task


# ---------------------------------------------------------------- periods


def test_work_by_month():
    # UC-31: an inspection every Monday (1.5 h, $120), a service every 3
    # months from 20 December (4 h, $480), and the overhaul on the first
    # Tuesday of November with no cost given.
    tasks = [
        Task(
            "inspect", every=Recurrence.weekly("Mon"), duration=1.5, cost=120
        ),
        Task(
            "service",
            every=Recurrence.every(3, "months", "2026-12-20"),
            duration=4.0,
            cost=480,
        ),
        Task(
            "overhaul",
            every=Recurrence.yearly(month=11, weekday="Tue", nth=1),
            duration=40.0,
        ),
    ]
    work = work_by_period(tasks, "month", "2026-11-16", "2027-11-15")
    assert isinstance(work, WorkByPeriod)
    assert len(work.starts) == 13
    assert work.starts[0] == D("2026-11-16") and work.ends[0] == D(
        "2026-11-30"
    )
    assert work.starts[-1] == D("2027-11-01") and work.ends[-1] == D(
        "2027-11-15"
    )
    # Mondays: 3 in the rest of Nov 2026, then 4, 4, 4, 5, 4, 5, 4, 4, 5,
    # 4, 4, and 3 in 1-15 Nov 2027: 53 in all.
    assert work.by_task["inspect"] == (3, 4, 4, 4, 5, 4, 5, 4, 4, 5, 4, 4, 3)
    assert work.by_task["service"] == (0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)
    assert work.by_task["overhaul"][-1] == 1  # 2 Nov 2027
    assert sum(work.count) == 53 + 4 + 1
    assert work.hours[1] == 4 * 1.5 + 4.0
    assert work.cost[1] == 4 * 120 + 480
    assert work.hours[-1] == 3 * 1.5 + 40.0
    assert work.cost[-1] == 3 * 120  # the overhaul has no cost
    assert work.no_cost == ("overhaul",) and work.no_hours == ()
    assert ("overhaul", D("2027-11-02")) in work.tasks[-1]
    # The periods add up to each task's whole.
    for task in tasks:
        assert (
            sum(work.by_task[task.name])
            == task.due("2026-11-16", "2027-11-15").count
        )
    assert json.loads(json.dumps(work.to_dict()))["starts"][0] == "2026-11-16"


@pytest.mark.parametrize(
    "period, n", [("day", 31), ("week", 5), ("quarter", 1), ("year", 1)]
)
def test_the_periods_cover_the_dates_asked(period, n):
    work = work_by_period(
        [Task("t", every=Recurrence.every(1, "days", "2027-01-01"))],
        period,
        "2027-01-01",
        "2027-01-31",
    )
    assert len(work.starts) == n
    assert work.starts[0] == D("2027-01-01") and work.ends[-1] == D(
        "2027-01-31"
    )
    assert sum(work.count) == 31
    for a, b in zip(work.ends, work.starts[1:]):
        assert b == a + timedelta(days=1)
    if period == "week":
        assert all(s.weekday() == 0 for s in work.starts[1:])


def test_work_by_period_checks_its_inputs():
    t = Task("t", every=WEEKLY)
    with pytest.raises(ValueError, match="period"):
        work_by_period([t], "fortnight", "2027-01-01", "2027-01-31")
    with pytest.raises(ValueError, match="own name"):
        work_by_period([t, t], "month", "2027-01-01", "2027-01-31")
    with pytest.raises(TypeError, match="list of Task"):
        work_by_period(t, "month", "2027-01-01", "2027-01-31")
