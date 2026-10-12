"""Tasks on a fixed calendar, and the dates they fall due (#319, UC-2 to
UC-4 of this package's ``README.md``); and tasks by use, every so many
hours or starts of an asset (UC-10), whose times are worked out in
``_usage``.

A calendar task happens on its dates whatever the equipment does: every
whole number of days from its first date (``every=``), or on the days a
monthly rule names (``on=``: a day of the month, or the first Tuesday,
say). So the dates it falls due in a period
are date arithmetic, exact, with no probability. Dates are whole days, and
a period includes both of its ends, as people say it ("through 31
December").

Every schedule is a recurrence rule of python-dateutil (``rrule``, the
iCalendar standard's): an interval's from the task's first date, a monthly
rule's from the date it applies from (``starting``, or the period's
start), and a period's dates are the rule's between its ends.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple, Union

from dateutil.rrule import DAILY, FR, MO, MONTHLY, SA, SU, TH, TU, WE, rrule

from repyability.maintenance._usage import (
    Asset,
    Hours,
    Starts,
    UsageDue,
    _moment,
    crossings,
)
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
        "every must be a datetime.timedelta or a number of days (a task "
        "by use takes Hours(...) or Starts(...)); got "
        f"{value!r}."
    )


def _midnight(day: date) -> datetime:
    """``day`` as dateutil takes it, a ``datetime`` at its start."""
    return datetime.combine(day, time())


@dataclass(frozen=True, init=False)
class Monthly:
    """A monthly rule naming the days a [`Task`][repyability.Task] falls
    due (its ``on``): a day of the month (the 30th), or a weekday of it
    (the first Tuesday).

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
    >>> readings = Task("meter reading", on=Monthly(day=30))
    >>> due = readings.due("2027-01-01", "2027-03-31")
    >>> [day.isoformat() for day in due.dates]
    ['2027-01-30', '2027-02-28', '2027-03-30']

    On the first Tuesday:

    >>> pumps = Task("check pumps", on=Monthly(weekday="Tuesday", nth=1))
    >>> pumps.due("2026-10-11", "2026-12-31").count
    2
    """

    day: Optional[int]
    weekday: Optional[str]
    nth: Optional[int]
    short_months: Optional[str]

    def __init__(
        self, *, day=None, weekday=None, nth=None, short_months=None
    ) -> None:
        if (day is None) == (weekday is None):
            raise ValueError(
                "Monthly takes a day of the month (Monthly(day=30)) or a "
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
    """A task: on the calendar, every so many days from its first date or
    on the days a monthly rule names, whatever the equipment does; or by
    use, every so many hours or starts of an asset.

    Parameters
    ----------
    name : str
        What the task is called.
    every : datetime.timedelta, int, Hours or Starts, optional
        An interval between one time and the next: on the calendar, a
        whole number of days, at least one, as a ``timedelta``
        (``timedelta(weeks=1)``) or a number of days (``7``); by use, the
        running hours or starts of ``asset`` ([`Hours`][repyability.Hours],
        [`Starts`][repyability.Starts]). Give ``every`` or ``on``.
    on : Monthly, optional
        A rule naming the days the task falls due
        ([`Monthly`][repyability.Monthly]: the 30th, the first Tuesday).
    first : datetime.date or str, optional
        With an interval on the calendar: the first time the task is done,
        as a date or as text such as ``"2026-07-15"``.
    starting : datetime.date or str, optional
        With a rule (``on``): the date from which it applies. By default
        None: the rule's days whenever asked.
    cost : float, optional
        What it costs each time it is done, at least 0, in your currency.
        By default None: not given, and the dates it falls due have no
        cost.
    asset : Asset, optional
        For a task by use: the [`Asset`][repyability.Asset] whose meter it
        goes by.
    last_done : float, optional
        For a task by use: the meter reading at its last time, at most the
        asset's reading, which must not yet have reached the next.

    Examples
    --------
    A weekly task first done on Wednesday 15 July 2026 falls due 12 times
    from Sunday 11 October to the end of 2026, at 250 each time 3000:

    >>> from datetime import timedelta
    >>> from repyability.maintenance import Task
    >>> conveyor = Task(
    ...     "inspect conveyor",
    ...     every=timedelta(weeks=1),
    ...     first="2026-07-15",
    ...     cost=250.0,
    ... )
    >>> due = conveyor.due("2026-10-11", "2026-12-31")
    >>> due.count, due.cost
    (12, 3000.0)
    >>> due.dates[0], due.dates[-1]
    (datetime.date(2026, 10, 14), datetime.date(2026, 12, 30))

    On the 30th of each month (the last day of a shorter one):

    >>> from repyability.maintenance import Monthly
    >>> readings = Task("meter reading", on=Monthly(day=30))
    >>> [day.isoformat() for day in readings.due(
    ...     "2027-01-01", "2027-03-31").dates]
    ['2027-01-30', '2027-02-28', '2027-03-30']

    By use: a pump that runs 06:00 to 22:00 on weekdays, serviced every
    500 running hours, its meter at 12,340 hours on Monday 12 October
    2026, before its start:

    >>> from repyability.maintenance import Asset, Hours, Window
    >>> pump = Asset(
    ...     "pump 1",
    ...     runs=Window("Mon-Fri", "06:00", "22:00"),
    ...     meters={"hours": 12_340},
    ...     read_at="2026-10-12 00:00",
    ... )
    >>> service = Task(
    ...     "pump service", every=Hours(500), asset=pump, last_done=12_000
    ... )
    >>> [str(time) for time in service.due("2026-10-12", "2026-12-31").times]
    ['2026-10-23 22:00:00', '2026-12-08 10:00:00']
    """

    name: str
    every: Union[timedelta, Hours, Starts, None]
    on: Optional[Monthly]
    first: Optional[date]
    starting: Optional[date]
    cost: Optional[float]
    asset: Optional[Asset]
    last_done: Optional[float]

    def __init__(
        self,
        name: str,
        *,
        every=None,
        on=None,
        first=None,
        starting=None,
        cost=None,
        asset=None,
        last_done=None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be text; got {name!r}.")
        if cost is not None and not (
            checks.is_number(cost) and 0.0 <= float(cost) < float("inf")
        ):
            raise ValueError(
                f"cost must be a number, at least 0 (or None); got {cost!r}."
            )
        if isinstance(every, Monthly):
            raise TypeError(
                "A monthly rule names the days a task falls due: give it as "
                "on=Monthly(...), not every=."
            )
        if (every is None) == (on is None):
            raise TypeError(
                "A task needs an interval (every=...) or a rule naming its "
                "days (on=Monthly(...)), one of them."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "cost", None if cost is None else float(cost))
        for field in ("every", "on", "first", "starting", "asset"):
            object.__setattr__(self, field, None)
        object.__setattr__(self, "last_done", None)
        if on is not None:
            self._on_rule(on, first, starting, asset, last_done)
        elif isinstance(every, (Hours, Starts)):
            object.__setattr__(self, "every", every)
            self._by_use(first, starting, asset, last_done)
        else:
            self._interval(every, first, starting, asset, last_done)

    def _on_rule(self, on, first, starting, asset, last_done) -> None:
        """Check and keep a task on the days a rule names."""
        if not isinstance(on, Monthly):
            raise TypeError(
                f"on must be a rule naming days (Monthly(...)); got {on!r}."
            )
        if first is not None:
            raise ValueError(
                "A rule's days are not counted from a first time: give the "
                "date it applies from as starting=, not first=."
            )
        if asset is not None or last_done is not None:
            raise ValueError(
                "asset and last_done go with a task by use (every=Hours(...) "
                "or Starts(...)), not with a rule naming days."
            )
        object.__setattr__(self, "on", on)
        if starting is not None:
            object.__setattr__(self, "starting", _date(starting, "starting"))

    def _interval(self, every, first, starting, asset, last_done) -> None:
        """Check and keep a task every so many days from its first date."""
        if asset is not None or last_done is not None:
            raise ValueError(
                "asset and last_done go with a task by use (every=Hours(...) "
                f"or Starts(...)); this one is every {every!r}."
            )
        if starting is not None:
            raise ValueError(
                "starting goes with a rule naming days (on=); an interval "
                "counts from the first time the task is done (first=)."
            )
        if first is None:
            raise TypeError(
                "A task every so many days needs the first time it is done "
                "(first=...)."
            )
        object.__setattr__(self, "every", _interval(every))
        object.__setattr__(self, "first", _date(first, "first"))

    def _by_use(self, first, starting, asset, last_done) -> None:
        """Check and keep a task by use: its asset, with the meter it goes
        by, and the reading at its last time."""
        every = self.every
        assert isinstance(every, (Hours, Starts))
        if first is not None or starting is not None:
            raise ValueError(
                "A task by use falls due by its asset's meter, not from a "
                "date: give last_done (the meter reading at its last time), "
                "and no first or starting."
            )
        if not isinstance(asset, Asset):
            raise TypeError(
                f"A task by use needs its asset (asset=...); got {asset!r}."
            )
        if every.meter not in asset.meters:
            raise ValueError(
                f"Asset {asset.name!r} has no {every.meter!r} meter, which "
                "this task goes by."
            )
        if not (
            checks.is_number(last_done) and 0.0 <= float(last_done) < math.inf
        ):
            raise ValueError(
                "last_done must be the meter reading at the task's last "
                f"time, a number at least 0; got {last_done!r}."
            )
        reading = asset.meters[every.meter]
        if float(last_done) > reading:
            raise ValueError(
                f"last_done ({float(last_done):g}) is past the asset's "
                f"{every.meter} reading ({reading:g})."
            )
        if reading >= float(last_done) + every.amount:
            raise ValueError(
                f"The task is due already: the asset's {every.meter} reading "
                f"({reading:g}) has reached last_done plus every "
                f"({float(last_done) + every.amount:g})."
            )
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "last_done", float(last_done))

    def due(self, start, end) -> Union["TaskDates", UsageDue]:
        """When the task falls due from ``start`` to ``end``, both days
        included.

        Parameters
        ----------
        start, end : datetime.date or str
            The first and last days of the period, as dates or as text
            such as ``"2026-12-31"``. For a task by use, the period may not
            start before the day its asset's meters were read.

        Returns
        -------
        TaskDates or UsageDue
            A task on the calendar: its dates in the period
            ([`TaskDates`][repyability.TaskDates]); a task by use: the
            readings and times it falls due
            ([`UsageDue`][repyability.UsageDue]); with how many there are,
            and what they cost.
        """
        start, end = _date(start, "start"), _date(end, "end")
        if end < start:
            raise ValueError(
                f"The period ends ({end}) before it starts ({start})."
            )
        if self.asset is not None:
            return self._due_by_use(start, end)
        times = self._rule(start).between(
            _midnight(start), _midnight(end), inc=True
        )
        dates = tuple(time.date() for time in times)
        cost = None if self.cost is None else len(dates) * self.cost
        return TaskDates(self.name, start, end, dates, len(dates), cost)

    def _due_by_use(self, start: date, end: date) -> UsageDue:
        """When a task by use falls due from ``start`` to ``end``."""
        asset, every = self.asset, self.every
        assert asset is not None and isinstance(every, (Hours, Starts))
        assert self.last_done is not None
        if start < asset.read_at.date():
            raise ValueError(
                f"The period starts ({start}) before the meters of asset "
                f"{asset.name!r} were read ({asset.read_at})."
            )
        begins = max(_moment(start, "start"), asset.read_at)
        ends = _moment(end + timedelta(days=1), "end")
        readings, times = crossings(asset, every, self.last_done, begins, ends)
        cost = None if self.cost is None else len(times) * self.cost
        return UsageDue(
            self.name,
            start,
            end,
            every.meter,
            readings,
            times,
            len(times),
            cost,
        )

    def _rule(self, start: date) -> rrule:
        """The task's dates as a dateutil rule: a rule's from the date it
        applies from (or the period's ``start``), an interval's from its
        first date."""
        if self.on is not None:
            applies = start if self.starting is None else self.starting
            return self.on._rule(applies)
        assert self.first is not None and isinstance(self.every, timedelta)
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
