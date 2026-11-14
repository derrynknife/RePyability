"""Tasks on dates: when a ``Task`` falls due in a period (``TaskDue``),
and the work in each calendar period (``work_by_period``).

A task has one or two limits, whichever comes first: ``every=`` and
``also=``. In this release each is a ``Recurrence``; a limit by use (so
many operating hours) plugs into the same loop through ``_Limit`` below.

**When the task is done.** A forecast has to assume it. Here a task is
done on its due date, or, with ``visits=``, at the first visit on or
after it. Then:

- a limit that **resets** (by default an interval, ``Recurrence.every``)
  counts again from when the task was done;
- a **fixed** limit (a weekday pattern, a yearly date, given dates) keeps
  its own dates; its next is the first after the task was done.

``resets=`` overrides the default for every limit, and the result says
which applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from repyability.schedule._recurrence import (
    VERSION,
    Recurrence,
    _date,
    _is_number,
    _is_whole,
    _period,
    _ScheduleResult,
)

#: The periods ``work_by_period`` sums over.
PERIODS = ("day", "week", "month", "quarter", "year")

#: The most due dates one call works out, a guard against a rule that
#: never advances.
_MAX_DUES = 1_000_000


# ------------------------------------------------------------ the hook


@dataclass(frozen=True)
class _Due:
    """One due point a limit gives: when (a ``datetime``; a calendar
    limit's at midnight) and which limit (``"every"`` or ``"also"``).
    ``reading`` is for a limit by use (the meter reading it is due at)."""

    moment: datetime
    limit: str
    reading: Optional[float] = None
    stop_after: Optional[datetime] = None


class _BoundLimit(Protocol):
    """A limit bound to one ``Task.due`` call."""

    def first(self, last_done: Optional[datetime]) -> Optional[_Due]:
        """The first due point once the task was last done at
        ``last_done`` (None: never recorded, so the first from the
        period's start, which ``bind`` was given)."""

    def after(self, done: datetime, due: _Due) -> Optional[_Due]:
        """The next due point once ``due`` (from whichever limit) was
        done at ``done``."""


class _Limit(Protocol):
    """What ``Task(every=...)`` and ``also=`` take: a ``Recurrence``, or
    anything with ``_task_limit(key, resets)`` returning one of these.
    A limit by use (a later release) implements it, and reads what it
    needs (the usage forecast, the reading when last done) from the
    ``context`` that ``Task.due`` passes to ``bind``."""

    def bind(
        self, task: "Task", start: date, end: date, context: Dict[str, Any]
    ) -> _BoundLimit:
        """The limit for one call of ``task.due(start, end, ...)``."""

    def describe(self) -> str:
        """The limit in plain words."""

    def to_dict(self) -> dict:
        """The limit as plain data, with a ``"kind"``."""

    @property
    def resets(self) -> bool:
        """Whether the limit counts again from when the task is done."""


#: Saved limits other than a recurrence, by their ``"kind"``: a later
#: release registers its limits here, so ``Task.from_dict`` reads them.
LIMIT_KINDS: Dict[str, Any] = {}


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time())


@dataclass(frozen=True)
class _CalendarLimit:
    """A ``Recurrence`` as a task's limit."""

    rule: Recurrence
    key: str
    resets: bool

    def bind(self, task, start, end, context) -> "_Running":
        return _Running(self, start)

    def describe(self) -> str:
        return self.rule.describe()

    def to_dict(self) -> dict:
        return self.rule.to_dict()


class _Running:
    """A calendar limit in one ``Task.due`` call, with the anchor it
    counts from now (moved when a resetting limit's task is done off its
    own date)."""

    def __init__(self, limit: _CalendarLimit, start: date) -> None:
        self.limit, self.start, self.rule = limit, start, limit.rule

    def _due(self, day: Optional[date]) -> Optional[_Due]:
        return None if day is None else _Due(_midnight(day), self.limit.key)

    def _restart(self, day: date) -> None:
        if self.limit.resets and self.rule.rule != "on":
            self.rule = self.rule.reanchored(day)

    def first(self, last_done: Optional[datetime]) -> Optional[_Due]:
        if last_done is None:
            return self._due(self.rule.on_or_after(self.start))
        self._restart(last_done.date())
        return self._due(self.rule.after(last_done.date()))

    def after(self, done: datetime, due: _Due) -> Optional[_Due]:
        if not (due.limit == self.limit.key and due.moment == done):
            self._restart(done.date())
        return self._due(self.rule.after(done.date()))


def _as_limit(value, key: str, resets: Optional[bool]):
    """``every=`` or ``also=`` as a limit."""
    if isinstance(value, Recurrence):
        default = value.rule == "every"
        return _CalendarLimit(
            value, key, default if resets is None else bool(resets)
        )
    make = getattr(value, "_task_limit", None)
    if callable(make):
        return make(key, resets)
    raise TypeError(
        f"{key} must be a Recurrence (Recurrence.every(3, 'months', ...), "
        f"Recurrence.monthly(...)); got {value!r}."
    )


# ------------------------------------------------------------ results


@dataclass(frozen=True)
class TaskDue(_ScheduleResult):
    """When a ``Task`` falls due in a period: returned by its ``due``.

    Attributes
    ----------
    task : str
        The task's name.
    start, end : datetime.date
        The period's first and last dates, both included.
    dates : tuple of datetime.date
        The dates the task falls due, in order. The first is before
        ``start`` when the task was already overdue then (see
        ``overdue``).
    limits : tuple of str
        Which limit set each date: ``"every"`` or ``"also"``.
    done : tuple of datetime.date
        When each is assumed done: its date, or the visit it waits for
        (``visits=``); an overdue one at ``start``.
    late : tuple of int
        The days each is done after its date (0 when done on it).
    on_time : tuple of bool
        Whether each is done within the task's ``tolerance`` of its date.
    overdue : bool
        Whether the task was overdue at ``start``: due after it was last
        done and before ``start``. It is then the first entry, done at
        ``start``.
    count : int
        How many times it is done in the period.
    cost : float or None
        The cost each time times ``count``; None if the task has no cost.
    hours : float or None
        The duration each time times ``count``; None if the task has no
        duration.
    resets : bool
        Whether the interval counts again from when the task is done
        (else the limits keep their own dates).
    moments : tuple of datetime.datetime
        When each falls due, to the minute: a calendar limit's at
        midnight, a limit by use's when the meter reaches its reading.
    readings : tuple of float or None
        The meter reading each limit by use falls due at (None for a
        calendar limit's).
    stops_after : tuple of datetime.datetime or None
        For a limit by use due mid-run, when the equipment next stops
        (the task is done then); None otherwise.
    notes : tuple of str
        Plain words on what the rules left out or moved in the period
        (months with no fifth Saturday, excluded dates), and on the
        assumptions made.
    """

    task: str
    start: date
    end: date
    dates: Tuple[date, ...]
    limits: Tuple[str, ...]
    done: Tuple[date, ...]
    late: Tuple[int, ...]
    on_time: Tuple[bool, ...]
    overdue: bool
    count: int
    cost: Optional[float]
    hours: Optional[float]
    resets: bool
    notes: Tuple[str, ...]
    moments: Tuple[datetime, ...] = ()
    readings: Tuple[Optional[float], ...] = ()
    stops_after: Tuple[Optional[datetime], ...] = ()


@dataclass(frozen=True)
class WorkByPeriod(_ScheduleResult):
    """The work tasks need in each calendar period: returned by
    ``work_by_period``.

    Attributes
    ----------
    period : str
        ``"day"``, ``"week"`` (Monday to Sunday), ``"month"``,
        ``"quarter"`` or ``"year"``.
    starts, ends : tuple of datetime.date
        Each period's first and last dates, both included; the first and
        last periods are cut to the dates asked for.
    count : tuple of int
        How many tasks are done in each period (by the date each is
        done).
    hours : tuple of float
        Their durations summed, of the tasks that have one.
    cost : tuple of float
        Their costs summed, of the tasks that have one.
    by_task : dict
        Each task's count in each period, by name.
    tasks : tuple of tuple of (str, datetime.date)
        The tasks done in each period, with the date each is done.
    no_hours, no_cost : tuple of str
        The tasks with no duration (or no cost), left out of ``hours``
        (or ``cost``), so a total is not taken for all the work.
    """

    period: str
    starts: Tuple[date, ...]
    ends: Tuple[date, ...]
    count: Tuple[int, ...]
    hours: Tuple[float, ...]
    cost: Tuple[float, ...]
    by_task: Dict[str, Tuple[int, ...]]
    tasks: Tuple[Tuple[Tuple[str, date], ...], ...]
    no_hours: Tuple[str, ...]
    no_cost: Tuple[str, ...]


# ------------------------------------------------------------ task


def _amount(value, name: str, what: str) -> Optional[float]:
    if value is None:
        return None
    if not (_is_number(value) and float(value) >= 0.0):
        raise ValueError(
            f"{name} must be a number, at least 0 ({what}), or None; got "
            f"{value!r}."
        )
    return float(value)


@dataclass(frozen=True, init=False, eq=True)
class Task:
    """A task that falls due by the calendar: every so many days, weeks,
    months or years, or on a pattern of dates; with an optional second
    limit, whichever comes first.

    Parameters
    ----------
    name : str
        What the task is called.
    every : Recurrence
        When it falls due. An interval (``Recurrence.every``) counts from
        when the task was last done; a pattern (``weekly``, ``monthly``,
        ``yearly``, ``on``) keeps its own dates (see ``resets``).
    also : Recurrence, optional
        A second limit: the task falls due by whichever comes first, and
        both count again from when it is done.
    last_done : datetime.date or str, optional
        When it was last done. Without it, the first due date is the
        rule's first on or after the period's start. With it, a due date
        before the period's start means the task is overdue.
    tolerance : int, optional
        How many days after its due date the task may be done and still
        be on time, by default 0.
    visits : Recurrence, optional
        When the work is done (a weekly crew visit): each due date waits
        for the first visit on or after it.
    cost : float, optional
        What it costs each time, at least 0; None (the default) when not
        given, so a cost never given is not taken as free.
    duration : float, optional
        The hours of work each time, at least 0; None when not given.
    resets : bool, optional
        Whether a limit counts again from when the task is done. By
        default an interval does and a pattern does not (and a limit by
        use, ``Usage``, does).
    last_done_reading : float, optional
        The meter reading when the task was last done, for a limit by use
        (``every=Usage(2000, "hours")``). Without it, the reading at
        ``last_done`` from the usage forecast.

    Examples
    --------
    A weekly task first done on Wednesday 15 July 2026, $250 a time:

    >>> from repyability.schedule import Recurrence, Task
    >>> conveyor = Task(
    ...     "inspect conveyor",
    ...     every=Recurrence.every(1, "weeks", "2026-07-15"),
    ...     cost=250.0,
    ... )
    >>> due = conveyor.due("2026-10-11", "2026-12-31")
    >>> due.count, due.cost
    (12, 3000.0)
    >>> str(due.dates[0]), str(due.dates[-1])
    ('2026-10-14', '2026-12-30')
    """

    name: str
    every: Any
    also: Any
    last_done: Optional[date]
    tolerance: int
    visits: Optional[Recurrence]
    cost: Optional[float]
    duration: Optional[float]
    resets: Optional[bool]
    last_done_reading: Optional[float]

    def __init__(
        self,
        name: str,
        *,
        every,
        also=None,
        last_done=None,
        tolerance=0,
        visits=None,
        cost=None,
        duration=None,
        resets=None,
        last_done_reading=None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"name must be text, not empty; got {name!r}.")
        if resets is not None and not isinstance(resets, bool):
            raise TypeError(
                f"resets must be True, False or None; got {resets!r}."
            )
        if not _is_whole(tolerance) or tolerance < 0:
            raise ValueError(
                "tolerance must be a whole number of days, at least 0; got "
                f"{tolerance!r}."
            )
        if visits is not None and not isinstance(visits, Recurrence):
            raise TypeError(
                "visits must be a Recurrence (when the work is done, such "
                f"as Recurrence.weekly('Tue')); got {visits!r}."
            )
        _as_limit(every, "every", resets)  # checks it
        if also is not None:
            _as_limit(also, "also", resets)
        values = dict(
            name=name,
            every=every,
            also=also,
            last_done=(
                None if last_done is None else _date(last_done, "last_done")
            ),
            tolerance=int(tolerance),
            visits=visits,
            cost=_amount(cost, "cost", "in your currency"),
            duration=_amount(duration, "duration", "hours of work"),
            resets=resets,
            last_done_reading=_amount(
                last_done_reading, "last_done_reading", "a meter reading"
            ),
        )
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def _limits(self) -> List[Any]:
        out = [_as_limit(self.every, "every", self.resets)]
        if self.also is not None:
            out.append(_as_limit(self.also, "also", self.resets))
        return out

    def describe(self) -> str:
        """The task's schedule in plain words.

        Returns
        -------
        str
        """
        limits = self._limits()
        text = f"{self.name}: {limits[0].describe()}"
        if len(limits) > 1:
            text += f", or {limits[1].describe()}, whichever comes first"
        if self.visits is not None:
            text += f"; done at the visits ({self.visits.describe()})"
        return text

    def due(self, start, end, usage=None) -> TaskDue:
        """When the task falls due from ``start`` to ``end``, **both
        included**, assuming each time is done on its due date (or at its
        visit).

        Parameters
        ----------
        start, end : datetime.date or str
            The first and last dates of the period.
        usage : UsageForecast, optional
            The map from dates to meter readings
            ([`DutyPlan.forecast`][repyability.schedule.DutyPlan.forecast]),
            which a limit by use (``Usage``) needs; it should cover the
            period.

        Returns
        -------
        TaskDue
        """
        return self._due(start, end, {"usage": usage})

    def _due(self, start, end, context: Dict[str, Any]) -> TaskDue:
        """``due`` with what the limits need (a later release's usage
        forecast) in ``context``."""
        start, end = _period(start, end)
        kinds = self._limits()
        limits = [limit.bind(self, start, end, context) for limit in kinds]
        last = None if self.last_done is None else _midnight(self.last_done)
        if (
            last is not None
            and self.last_done is not None
            and self.last_done > end
        ):
            raise ValueError(
                f"last_done ({self.last_done}) is after the period ends "
                f"({end})."
            )
        nexts = [limit.first(last) for limit in limits]
        dates, which, done_on, late, on_time = [], [], [], [], []
        moments, readings, stops = [], [], []
        overdue = False
        for _ in range(_MAX_DUES):
            pending = [n for n in nexts if n is not None]
            if not pending:
                break
            due = min(pending, key=lambda n: (n.moment, n.limit != "every"))
            day = due.moment.date()
            if day > end:
                break
            if day < start:
                overdue, done = True, _midnight(start)
            elif self.visits is not None:
                visit = self.visits.on_or_after(day)
                if visit is None:
                    break  # no visit left to do it at
                done = _midnight(visit)
            else:
                done = due.moment
            if done.date() > end:
                break  # done after the period: not this period's work
            days_late = (done.date() - day).days
            dates.append(day)
            moments.append(due.moment)
            readings.append(due.reading)
            stops.append(due.stop_after)
            which.append(due.limit)
            done_on.append(done.date())
            late.append(days_late)
            on_time.append(days_late <= self.tolerance)
            nexts = [limit.after(done, due) for limit in limits]
        else:
            raise RuntimeError(
                f"Task {self.name!r} falls due more than {_MAX_DUES} times "
                "in the period."
            )
        count = len(dates)
        notes = list(self._notes(start, end))
        if overdue:
            notes.insert(
                0,
                f"Overdue at {start}: due on {dates[0]}, after it was last "
                f"done ({self.last_done}); assumed done on {start}.",
            )
        return TaskDue(
            task=self.name,
            start=start,
            end=end,
            dates=tuple(dates),
            limits=tuple(which),
            done=tuple(done_on),
            late=tuple(late),
            on_time=tuple(on_time),
            overdue=overdue,
            count=count,
            cost=None if self.cost is None else count * self.cost,
            hours=None if self.duration is None else count * self.duration,
            resets=any(getattr(kind, "resets", False) for kind in kinds),
            notes=tuple(notes),
            moments=tuple(moments),
            readings=tuple(readings),
            stops_after=tuple(stops),
        )

    def _notes(self, start: date, end: date) -> List[str]:
        out: List[str] = []
        for value in (self.every, self.also, self.visits):
            if isinstance(value, Recurrence):
                out.extend(value.notes(start, end).text)
        return out

    # -- saving --------------------------------------------------------

    def to_dict(self) -> dict:
        """The task as plain data, for JSON: a ``"kind"`` and a
        ``"version"``, its limits and visits as their own ``to_dict()``,
        dates as ISO text, and the options given.

        Returns
        -------
        dict
        """
        out: Dict[str, Any] = {
            "kind": "task",
            "version": VERSION,
            "name": self.name,
            "every": self.every.to_dict(),
        }
        if self.also is not None:
            out["also"] = self.also.to_dict()
        if self.last_done is not None:
            out["last_done"] = self.last_done.isoformat()
        if self.tolerance:
            out["tolerance"] = self.tolerance
        if self.visits is not None:
            out["visits"] = self.visits.to_dict()
        for key in ("cost", "duration", "resets", "last_done_reading"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        return out

    @classmethod
    def from_dict(cls, data) -> "Task":
        """A task from its ``to_dict()``.

        Parameters
        ----------
        data : dict
            The saved task.

        Returns
        -------
        Task
        """
        if not isinstance(data, dict) or data.get("kind") != "task":
            raise ValueError(
                f"This is not a saved task (no kind 'task'); got {data!r}."
            )
        version = data.get("version", VERSION)
        if not _is_whole(version) or version > VERSION:
            raise ValueError(
                f"This task was saved by a later version (format "
                f"{version!r}); upgrade RePyability to read it."
            )
        return cls(
            data["name"],
            every=_limit_from_dict(data["every"]),
            also=(
                None if "also" not in data else _limit_from_dict(data["also"])
            ),
            last_done=data.get("last_done"),
            tolerance=data.get("tolerance", 0),
            visits=(
                None
                if "visits" not in data
                else Recurrence.from_dict(data["visits"])
            ),
            cost=data.get("cost"),
            duration=data.get("duration"),
            resets=data.get("resets"),
            last_done_reading=data.get("last_done_reading"),
        )


def _limit_from_dict(data) -> Any:
    kind = data.get("kind") if isinstance(data, dict) else None
    if kind == "recurrence":
        return Recurrence.from_dict(data)
    if kind in LIMIT_KINDS:
        return LIMIT_KINDS[kind].from_dict(data)
    raise ValueError(f"A saved limit has an unknown kind {kind!r}.")


# ------------------------------------------------------------ periods


def _period_start(day: date, period: str) -> date:
    if period == "day":
        return day
    if period == "week":
        return day - timedelta(days=day.weekday())
    if period == "month":
        return day.replace(day=1)
    if period == "quarter":
        return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
    return date(day.year, 1, 1)


def _next_period(first: date, period: str) -> date:
    if period == "day":
        return first + timedelta(days=1)
    if period == "week":
        return first + timedelta(days=7)
    months = {"month": 1, "quarter": 3, "year": 12}[period]
    index = first.year * 12 + first.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _periods(start: date, end: date, period: str):
    """The periods from ``start`` to ``end``, each ``(first, last)``, cut
    to the dates asked for."""
    out: List[Tuple[date, date]] = []
    first = _period_start(start, period)
    while first <= end:
        following = _next_period(first, period)
        out.append(
            (max(first, start), min(following - timedelta(days=1), end))
        )
        first = following
    return out


def work_by_period(
    tasks: Sequence[Task], period, start, end, usage=None
) -> WorkByPeriod:
    """The work tasks need in each calendar period from ``start`` to
    ``end`` (both included): how many are done, their hours and cost,
    each counted in the period it is done in.

    Parameters
    ----------
    tasks : sequence of Task
        The tasks, with names that differ.
    period : {"day", "week", "month", "quarter", "year"}
        The periods (a week runs Monday to Sunday); the first and last
        are cut to ``start`` and ``end``.
    start, end : datetime.date or str
        The first and last dates.
    usage : UsageForecast, optional
        The map from dates to meter readings, for tasks with a limit by
        use (see ``Task.due``).

    Returns
    -------
    WorkByPeriod

    Examples
    --------
    >>> from repyability.schedule import Recurrence, Task, work_by_period
    >>> inspect = Task("inspect", every=Recurrence.weekly("Mon"),
    ...                duration=1.5, cost=120.0)
    >>> work = work_by_period([inspect], "month", "2026-11-16", "2027-01-31")
    >>> work.count, work.hours
    ((3, 4, 4), (4.5, 6.0, 6.0))
    """
    if period not in PERIODS:
        raise ValueError(
            f"period must be one of {', '.join(map(repr, PERIODS))}; got "
            f"{period!r}."
        )
    if isinstance(tasks, Task) or not all(isinstance(t, Task) for t in tasks):
        raise TypeError("tasks must be a list of Task.")
    names = [t.name for t in tasks]
    if len(set(names)) != len(names):
        raise ValueError(
            "Each task needs its own name, to be told apart in the result."
        )
    start, end = _period(start, end)
    spans = _periods(start, end, period)
    firsts = [a for a, _ in spans]
    n = len(spans)
    count, hours, cost = [0] * n, [0.0] * n, [0.0] * n
    listed: List[List[Tuple[str, date]]] = [[] for _ in range(n)]
    by_task: Dict[str, Tuple[int, ...]] = {}
    for task in tasks:
        due = task.due(start, end, usage=usage)
        mine = [0] * n
        for day in due.done:
            index = max(i for i, first in enumerate(firsts) if first <= day)
            mine[index] += 1
            count[index] += 1
            listed[index].append((task.name, day))
            if task.duration is not None:
                hours[index] += task.duration
            if task.cost is not None:
                cost[index] += task.cost
        by_task[task.name] = tuple(mine)
    return WorkByPeriod(
        period=period,
        starts=tuple(firsts),
        ends=tuple(b for _, b in spans),
        count=tuple(count),
        hours=tuple(hours),
        cost=tuple(cost),
        by_task=by_task,
        tasks=tuple(
            tuple(sorted(x, key=lambda p: (p[1], p[0]))) for x in listed
        ),
        no_hours=tuple(t.name for t in tasks if t.duration is None),
        no_cost=tuple(t.name for t in tasks if t.cost is None),
    )
