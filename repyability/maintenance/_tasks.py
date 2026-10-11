"""Tasks on a fixed calendar, and the dates they fall due (#319, UC-2 to
UC-4 of this package's ``README.md``).

A calendar task happens on its dates whatever the equipment does: from its
first date, every whole number of days, or on a monthly rule (a day of the
month, or the first Tuesday, say). So the dates it falls due in a period
are date arithmetic, exact, with no probability. Dates are whole days, and
a period includes both of its ends, as people say it ("through 31
December").

Every schedule is a recurrence rule of python-dateutil (``rrule``, the
iCalendar standard's), from the task's first date, and a period's dates
are the rule's between its ends.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple, Union

from dateutil.rrule import DAILY, FR, MO, MONTHLY, SA, SU, TH, TU, WE, rrule

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: The days of the week, by name, as dateutil's weekdays.
_WEEKDAYS = {
    "Monday": MO,
    "Tuesday": TU,
    "Wednesday": WE,
    "Thursday": TH,
    "Friday": FR,
    "Saturday": SA,
    "Sunday": SU,
}

#: The weekdays of a month a monthly rule may name: first to fourth, and
#: last to fourth from last, which every month has.
_NTHS = (1, 2, 3, 4, -1, -2, -3, -4)


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
        "every must be a datetime.timedelta, a number of days or a "
        "Monthly rule; got "
        f"{value!r}."
    )


def _midnight(day: date) -> datetime:
    """``day`` as dateutil takes it, a ``datetime`` at its start."""
    return datetime.combine(day, time())


@dataclass(frozen=True, init=False)
class Monthly:
    """A monthly rule for a [`Task`][repyability.Task]: a day of the
    month (the 30th), or a weekday of it (the first Tuesday).

    Parameters
    ----------
    day : int, optional
        The day of the month, 1 to 31.
    weekday : str, optional
        The day of the week, by name (``"Tuesday"``; any case), with
        ``nth``. Give a ``day`` or a ``weekday``, not both.
    nth : int, optional
        Which of the month's ``weekday`` s: 1 to 4 (the first to the
        fourth), or -1 to -4 (the last to the fourth from last), which
        every month has.
    short_months : {"last", "skip"}, optional
        With a ``day`` past the 28th, what a month too short for it does:
        ``"last"`` (the default) falls due on its last day instead (the
        28th or 29th of February for the 30th); ``"skip"`` does not fall
        due that month.

    Examples
    --------
    On the 30th of every month, on 28 February in 2027:

    >>> from repyability.maintenance import Monthly, Task
    >>> readings = Task("meter reading", "2026-10-01", every=Monthly(30))
    >>> due = readings.due("2027-01-01", "2027-03-31")
    >>> [day.isoformat() for day in due.dates]
    ['2027-01-30', '2027-02-28', '2027-03-30']

    On the first Tuesday:

    >>> first_tuesday = Monthly(weekday="Tuesday", nth=1)
    >>> pumps = Task("check pumps", "2026-10-01", every=first_tuesday)
    >>> pumps.due("2026-10-11", "2026-12-31").count
    2
    """

    day: Optional[int]
    weekday: Optional[str]
    nth: Optional[int]
    short_months: Optional[str]

    def __init__(
        self, day=None, *, weekday=None, nth=None, short_months=None
    ) -> None:
        if (day is None) == (weekday is None):
            raise ValueError(
                "Monthly takes a day of the month (Monthly(30)) or a "
                "weekday (Monthly(weekday='Tuesday', nth=1)), not both or "
                "neither."
            )
        if day is not None:
            day = checks.whole_number(day, "day")
            if day > 31:
                raise ValueError(f"day must be 1 to 31; got {day!r}.")
            if nth is not None:
                raise ValueError(
                    "nth goes with a weekday, which this rule has not: it "
                    f"is on day {day} of the month."
                )
            short_months = "last" if short_months is None else short_months
            checks.one_of("short_months", short_months, ("last", "skip"))
        else:
            name = weekday.capitalize() if isinstance(weekday, str) else None
            if name not in _WEEKDAYS:
                raise ValueError(
                    "weekday must be a day of the week, such as 'Tuesday'; "
                    f"got {weekday!r}."
                )
            weekday = name
            if not checks.is_whole(nth) or int(nth) not in _NTHS:
                raise ValueError(
                    "nth must be 1 to 4 (the first to the fourth) or -1 to "
                    "-4 (the last to the fourth from last), which every "
                    f"month has; got {nth!r}."
                )
            nth = int(nth)
            if short_months is not None:
                raise ValueError(
                    "short_months goes with a day of the month: every month "
                    f"has a {_ordinal(nth)} {weekday}."
                )
        object.__setattr__(self, "day", day)
        object.__setattr__(self, "weekday", weekday)
        object.__setattr__(self, "nth", nth)
        object.__setattr__(self, "short_months", short_months)

    def _rule(self, first: date) -> rrule:
        """The rule's dates from ``first`` on."""
        start = _midnight(first)
        if self.weekday is not None:
            assert self.nth is not None
            weekday = _WEEKDAYS[self.weekday](self.nth)
            return rrule(MONTHLY, dtstart=start, byweekday=weekday)
        assert self.day is not None
        if self.day <= 28 or self.short_months == "skip":
            return rrule(MONTHLY, dtstart=start, bymonthday=self.day)
        # The latest of the 28th up to the day that the month has.
        return rrule(
            MONTHLY,
            dtstart=start,
            bymonthday=tuple(range(28, self.day + 1)),
            bysetpos=-1,
        )


def _ordinal(nth: int) -> str:
    """``nth`` in words: "first", "last", "second from last"."""
    words = {1: "first", 2: "second", 3: "third", 4: "fourth"}
    if nth > 0:
        return words[nth]
    return "last" if nth == -1 else f"{words[-nth]} from last"


@dataclass(frozen=True, init=False)
class Task:
    """A task on a fixed calendar: done from its first date on, every
    whole number of days or on a monthly rule, whatever the equipment does.

    Parameters
    ----------
    name : str
        What the task is called.
    first : datetime.date or str
        The date its schedule starts, as a date or as text such as
        ``"2026-07-15"``. With an interval, it is the first time the task
        is done; with a monthly rule, the task falls due on the rule's
        dates from then on.
    every : datetime.timedelta, int or Monthly
        The interval between one time and the next, a whole number of
        days, at least one, as a ``timedelta`` (``timedelta(weeks=1)``) or
        a number of days (``7``); or a monthly rule
        ([`Monthly`][repyability.Monthly]: the 30th, the first Tuesday).
    cost : float, optional
        What it costs each time it is done, at least 0, in your currency.
        By default None: not given, and the dates it falls due have no
        cost.

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

    At 250 each time, they cost 3000:

    >>> conveyor = Task(
    ...     "inspect conveyor",
    ...     first="2026-07-15",
    ...     every=timedelta(weeks=1),
    ...     cost=250.0,
    ... )
    >>> conveyor.due("2026-10-11", "2026-12-31").cost
    3000.0
    """

    name: str
    first: date
    every: Union[timedelta, Monthly]
    cost: Optional[float]

    def __init__(self, name: str, first, every, cost=None) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be text; got {name!r}.")
        if cost is not None and not (
            checks.is_number(cost) and 0.0 <= float(cost) < float("inf")
        ):
            raise ValueError(
                f"cost must be a number, at least 0 (or None); got {cost!r}."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "first", _date(first, "first"))
        object.__setattr__(
            self,
            "every",
            every if isinstance(every, Monthly) else _interval(every),
        )
        object.__setattr__(self, "cost", None if cost is None else float(cost))

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
            The task's dates in the period, how many there are, and what
            they cost.
        """
        start, end = _date(start, "start"), _date(end, "end")
        if end < start:
            raise ValueError(
                f"The period ends ({end}) before it starts ({start})."
            )
        times = self._rule().between(
            _midnight(start), _midnight(end), inc=True
        )
        dates = tuple(time.date() for time in times)
        cost = None if self.cost is None else len(dates) * self.cost
        return TaskDates(self.name, start, end, dates, len(dates), cost)

    def _rule(self) -> rrule:
        """The task's dates, as a dateutil rule from its first date."""
        if isinstance(self.every, Monthly):
            return self.every._rule(self.first)
        return rrule(
            DAILY, dtstart=_midnight(self.first), interval=self.every.days
        )


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
    cost : float or None
        What they cost: the task's cost each time, times ``count``; None
        if the task was given no cost.
    """

    task: str
    start: date
    end: date
    dates: Tuple[date, ...]
    count: int
    cost: Optional[float]
