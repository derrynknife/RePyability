"""Tasks on a fixed calendar, and the dates they fall due (#319, UC-2 of
this package's ``README.md``).

A calendar task happens on its dates whatever the equipment does: on its
first date and every whole interval after it. So the dates it falls due in
a period are date arithmetic, exact, with no probability. Dates are whole
days, and a period includes both of its ends, as people say it ("through
31 December").
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Tuple

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks


def _date(value, name: str) -> date:
    """``value`` as a ``date``: a ``datetime.date``, or ISO text such as
    ``"2026-07-15"``; a ``datetime`` is refused, as a task's dates are
    whole days."""
    if isinstance(value, datetime):
        raise TypeError(
            f"{name} must be a date (a datetime.date, or text such as "
            f"'2026-07-15'), without a time of day; got {value!r}."
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"{name} must be a date written as YYYY-MM-DD, such as "
                f"'2026-07-15'; got {value!r}."
            ) from None
    raise TypeError(
        f"{name} must be a date (a datetime.date, or text such as "
        f"'2026-07-15'); got {value!r}."
    )


def _interval(value) -> timedelta:
    """``value`` as a whole number of days, at least one: a
    ``datetime.timedelta``, or a number of days."""
    if isinstance(value, timedelta):
        days = value / timedelta(days=1)
        if not days.is_integer() or days < 1:
            raise ValueError(
                "every must be a whole number of days, at least one; got "
                f"{value!r}."
            )
        return value
    if checks.is_number(value):
        return timedelta(days=checks.whole_number(value, "every"))
    raise TypeError(
        "every must be a datetime.timedelta or a number of days; got "
        f"{value!r}."
    )


@dataclass(frozen=True, init=False)
class Task:
    """A task on a fixed calendar: done on its first date and every
    interval after it, whatever the equipment does.

    Parameters
    ----------
    name : str
        What the task is called.
    first : datetime.date or str
        The date it was first done (or will be), as a date or as text
        such as ``"2026-07-15"``.
    every : datetime.timedelta or int
        The interval between one time and the next: a whole number of
        days, at least one, as a ``timedelta`` (``timedelta(weeks=1)``) or
        a number of days (``7``).

    Examples
    --------
    A weekly task first done on Wednesday 15 July 2026 falls due 12 times
    from Sunday 11 October to the end of 2026:

    >>> from datetime import timedelta
    >>> from repyability.maintenance import Task
    >>> conveyor = Task(
    ...     "inspect conveyor", first="2026-07-15", every=timedelta(weeks=1)
    ... )
    >>> due = conveyor.due("2026-10-11", "2026-12-31")
    >>> due.count
    12
    >>> due.dates[0], due.dates[-1]
    (datetime.date(2026, 10, 14), datetime.date(2026, 12, 30))
    """

    name: str
    first: date
    every: timedelta

    def __init__(self, name: str, first, every) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be text; got {name!r}.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "first", _date(first, "first"))
        object.__setattr__(self, "every", _interval(every))

    def due(self, start, end) -> "TaskDates":
        """The dates the task falls due from ``start`` to ``end``, both
        days included.

        Parameters
        ----------
        start, end : datetime.date or str
            The first and last days of the period, as dates or as text
            such as ``"2026-12-31"``.

        Returns
        -------
        TaskDates
            The task's dates in the period, and how many there are.
        """
        start, end = _date(start, "start"), _date(end, "end")
        if end < start:
            raise ValueError(
                f"The period ends ({end}) before it starts ({start})."
            )
        # The first time on or after the start, and the last on or before
        # the end, as counts of intervals from the first date.
        first = max(0, -((self.first - start) // self.every))
        last = (end - self.first) // self.every
        dates = tuple(
            self.first + k * self.every for k in range(first, last + 1)
        )
        return TaskDates(self.name, start, end, dates, len(dates))


@dataclass
class TaskDates(_ResultMapping):
    """The dates a [`Task`][repyability.Task] falls due in a period:
    returned by its ``due``.

    Attributes
    ----------
    task : str
        The task's name.
    start, end : datetime.date
        The period's first and last days, both included.
    dates : tuple of datetime.date
        The dates the task falls due in the period, in order.
    count : int
        How many there are.
    """

    task: str
    start: date
    end: date
    dates: Tuple[date, ...]
    count: int
