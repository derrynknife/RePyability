"""Schedules: which dates a rule gives, and when tasks fall due on them.

RePyability computes; Reliafy (or any caller) records. This module takes
dates, rules and plain data from the caller and returns plain data: it
never reads the clock, stores anything or knows a workspace. Dates are
whole local dates, and every period asked about includes both of its
ends.

- ``Recurrence``: which dates (every so many days, weeks, months or
  years; named weekdays; a day or weekday of the month; a pattern in a
  named month each year; given dates), with a ``Season``, exclusions and
  moves.
- ``Task``: when a task falls due by its rule (and a second limit,
  whichever comes first), at its visits; ``work_by_period`` sums the
  work by calendar period.
- ``DutyPlan``: how equipment runs (``Runs``, ``Shifts``, ``Roster``,
  ``Rate``, ``Continuous``; a background and specific parts on top), in
  the site's time zone; its ``forecast`` is a ``UsageForecast``, the map
  between moments and meter readings both ways, which puts a ``Usage``
  limit ("every 2,000 operating hours") on dates.

``README.md`` in this package records the use cases, the question each
answers and how it is solved.
"""

from repyability.schedule._duty import (
    Continuous,
    DutyPlan,
    Rate,
    Roster,
    Runs,
    Shifts,
)
from repyability.schedule._recurrence import (
    Recurrence,
    RecurrenceNotes,
    Season,
)
from repyability.schedule._task import (
    Task,
    TaskDue,
    WorkByPeriod,
    work_by_period,
)
from repyability.schedule._usage import (
    Crossing,
    DutySchedule,
    PeriodUsage,
    Readings,
    Stretch,
    Usage,
    UsageDue,
    UsageForecast,
)

__all__ = [
    "Continuous",
    "Crossing",
    "DutyPlan",
    "DutySchedule",
    "PeriodUsage",
    "Rate",
    "Readings",
    "Roster",
    "Runs",
    "Shifts",
    "Stretch",
    "Usage",
    "UsageDue",
    "UsageForecast",
    "Recurrence",
    "RecurrenceNotes",
    "Season",
    "Task",
    "TaskDue",
    "WorkByPeriod",
    "work_by_period",
]
