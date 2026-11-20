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

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from repyability.schedule import _discount
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


def _unused_limit(kind: Any, due: "_Due", own: Optional["_Due"]) -> str:
    """The note on a limit that set none of the dates: when it would have
    been reached, measured from the first date the other limit set."""
    first = due.moment.date()
    words = f'The limit "{kind.describe()}" set none of the dates'
    if own is None:
        return (
            f"{words}: when the task was first due ({first}, by the other "
            "limit), it would not have been reached within the forecast."
        )
    days = (own.moment - due.moment).total_seconds() / 86400.0
    if days >= 60.0:
        gap = f"about {days / (365.25 / 12):.1f} months"
    else:
        gap = f"{days:.0f} days"
    return (
        f"{words}: when the task was first due ({first}, by the other "
        f"limit), it would not have been reached until {own.moment.date()}, "
        f"{gap} later."
    )


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
    """A task's limit. ``Task(every=...)`` and ``also=`` take a
    ``Recurrence`` (made one by ``_as_limit``), or any object with
    ``_task_limit(key, resets)`` returning one of these: ``key`` is
    ``"every"`` or ``"also"``, ``resets`` the task's ``resets=`` (None
    for the limit's default). A limit by use implements it, and reads
    what it needs (the usage forecast, the reading when last done) from
    the ``context`` that ``Task._due`` passes to ``bind``. Its
    ``to_dict()`` is what ``Task.to_dict`` saves; ``LIMIT_KINDS`` maps its
    ``"kind"`` to a class whose ``from_dict`` gives back the object the
    task was made with."""

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


#: Saved limits other than a recurrence, by their ``"kind"``: a limit
#: by use registers its class here, so ``Task.from_dict`` reads it.
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
    discount : float or None
        The yearly discount rate given (``due(..., discount=0.07)``);
        None when none was.
    npv : float or None
        With a discount and a cost: what the period's costs are worth on
        ``start``, each discounted from the date it is done, ``cost / (1
        + discount) ** years`` with a year of 365.25 days.
    eac : float or None
        The equivalent annual cost: the steady cost a year over the
        period worth ``npv`` (``cost`` over the period's years at a
        discount of 0).
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
    discount: Optional[float] = None
    npv: Optional[float] = None
    eac: Optional[float] = None


@dataclass(frozen=True)
class LongRunValue(_ScheduleResult):
    """What a task repeating for ever is worth now: returned by
    ``Task.long_run_value``.

    Attributes
    ----------
    task : str
        The task's name.
    start : datetime.date
        The date the costs are discounted to; the dates from it count.
    discount : float
        The yearly discount rate.
    npv : float
        The present value on ``start`` of the task's cost every time it
        is done, for ever.
    eac : float
        The equivalent annual cost: the steady cost a year, for ever,
        worth ``npv`` (``npv * ln(1 + discount)``).
    method : str
        ``"closed form"`` (a plain interval of days or weeks: a geometric
        sum, exact) or ``"summed"`` (any other task: its dates to
        ``summed_to``, one by one).
    precision : float
        The most the dates left out can add to ``npv``: 0 for a closed
        form; for a sum, what a task done every day after ``summed_to``
        would add, which no task exceeds.
    summed_to : datetime.date or None
        The last date summed to (None for a closed form).
    count : int or None
        How many times the task is done to ``summed_to`` (None for a
        closed form).
    notes : tuple of str
        What was assumed.
    """

    task: str
    start: date
    discount: float
    npv: float
    eac: float
    method: str
    precision: float
    summed_to: Optional[date]
    count: Optional[int]
    notes: Tuple[str, ...]


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
    discount : float or None
        The yearly discount rate given; None when none was.
    npv : tuple of float or None
        With a discount: what each period's ``cost`` is worth on the
        first date, each task discounted from the date it is done.
    total_npv : float or None
        The periods' ``npv`` summed.
    eac : float or None
        The equivalent annual cost: the steady cost a year over the dates
        asked for worth ``total_npv``.
    notes : tuple of str
        What the present values assumed (empty with no discount).
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
    discount: Optional[float] = None
    npv: Optional[Tuple[float, ...]] = None
    total_npv: Optional[float] = None
    eac: Optional[float] = None
    notes: Tuple[str, ...] = ()


# ------------------------------------------------------------ task

#: The years a long-run sum first runs to, before it is known how far it
#: must go.
_FIRST_YEARS = 25.0


def _years(start: date, end: date) -> float:
    """The length in years of a period that includes both its dates."""
    return ((end - start).days + 1) / _discount.DAYS_A_YEAR


def _present_value(
    cost: float, discount: float, start: date, done: Sequence[date]
) -> float:
    """What ``cost`` on each date of ``done`` is worth on ``start``."""
    return math.fsum(
        cost * _discount.factor_on(discount, start, day) for day in done
    )


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
        A second limit: the task falls due by whichever comes first; once
        it is done, a resetting limit counts again from then (see
        ``resets``).
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

    def due(
        self, start, end, usage=None, *, discount=None, escalation=None
    ) -> TaskDue:
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
        discount : float, optional
            A yearly discount rate (0.07 for 7% a year): the result then
            has ``npv``, the costs' present value on ``start``, each
            discounted from the date it is done, and ``eac``. By default
            None: no present value.
        escalation : float, optional
            With ``discount``, the effective yearly rate every cost
            escalates at from its price today (0.03 for 3% a year): the
            present values (and ``eac``) take each cost grown to when it
            falls, at the net yearly rate ``(1 + discount) / (1 + escalation)
            - 1``. By default None: today's prices.

        Returns
        -------
        TaskDue

        Examples
        --------
        $250 every 30 days from 20 November 2026, at 7% a year, over the
        first year:

        >>> from repyability.schedule import Recurrence, Task
        >>> task = Task(
        ...     "grease bearings",
        ...     every=Recurrence.every(30, "days", "2026-11-20"),
        ...     cost=250.0,
        ... )
        >>> year = task.due("2026-11-20", "2027-11-19", discount=0.07)
        >>> year.count, year.cost, round(year.npv, 2)
        (13, 3250.0, 3144.1)
        """
        return self._due(start, end, {"usage": usage}, discount, escalation)

    def _due(
        self,
        start,
        end,
        context: Dict[str, Any],
        discount=None,
        escalation=None,
    ) -> TaskDue:
        """``due`` with what the limits need (a later release's usage
        forecast) in ``context``."""
        start, end = _period(start, end)
        discount = _discount.yearly(discount)
        given, discount = discount, _discount.net(discount, escalation)
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
        # For each limit, the first time another set the date while it was
        # pending: (the date set, its own due point, or None past the
        # forecast), for the note on a limit that sets none.
        beaten: Dict[str, Tuple[_Due, Optional[_Due]]] = {}
        for _ in range(_MAX_DUES):
            pending = [n for n in nexts if n is not None]
            if not pending:
                break
            due = min(pending, key=lambda n: (n.moment, n.limit != "every"))
            if len(kinds) > 1 and due.moment.date() <= end:
                for key, n in zip(("every", "also"), nexts):
                    if key != due.limit and key not in beaten:
                        beaten[key] = (due, n)
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
        for key, kind in zip(("every", "also"), kinds):
            if key in beaten and key not in which:
                notes.append(_unused_limit(kind, *beaten[key]))
        if overdue:
            notes.insert(
                0,
                f"Overdue at {start}: due on {dates[0]}, after it was last "
                f"done ({self.last_done}); assumed done on {start}.",
            )
        npv = eac = None
        if discount is not None:
            if self.cost is None:
                notes.append(
                    "No present value: the task has no cost (give cost=)."
                )
            else:
                npv = _present_value(self.cost, discount, start, done_on)
                eac = _discount.equivalent_annual(
                    npv, discount, _years(start, end)
                )
                notes.append(_discount.words(given, start, escalation))
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
            discount=given,
            npv=npv,
            eac=eac,
        )

    def long_run_value(
        self, start, discount, *, precision: float = 1e-9, escalation=None
    ) -> LongRunValue:
        """What the task is worth on ``start`` repeating for ever: the
        present value of its cost every time it is done from ``start``
        on, at a yearly discount rate.

        A plain interval of days or weeks (no second limit, visits,
        season or excluded dates) is a geometric sum, in closed form:
        ``cost / (1 - (1 + discount) ** (-days / 365.25))`` from the
        first time it is done. Any other task is summed date by date
        until what is left cannot matter: a task is done once a day at
        most, so the dates after any day are worth less than a task done
        every day from then, and the sum stops where that is under
        ``precision`` of the total.

        Parameters
        ----------
        start : datetime.date or str
            The date the costs are discounted to; the task's dates from
            it count (an overdue task is done on it).
        discount : float
            The yearly discount rate, more than 0 (0.07 for 7% a year).
        escalation : float, optional
            The effective yearly rate the cost escalates at from its price
            today (0.03 for 3% a year), below ``discount``: the value is
            taken at the net yearly rate ``(1 + discount) / (1 +
            escalation) - 1``. By default None: today's price.
        precision : float, optional
            For a task that is summed: the most the dates left out may
            add, as a share of the value; by default 1e-9.

        Returns
        -------
        LongRunValue

        Raises
        ------
        ValueError
            The task has no cost; ``discount`` is 0 (the costs add up
            without end: ``due`` gives a period's); its dates end (given
            dates: ``due`` to the last gives the whole value); a limit is
            by use (``Usage``: it follows a usage forecast, which ends);
            or the rate is too low for the sum to settle to ``precision``
            by the year 9999.

        Examples
        --------
        $250 every 30 days from 20 November 2026, at 7% a year:

        >>> from repyability.schedule import Recurrence, Task
        >>> task = Task(
        ...     "grease bearings",
        ...     every=Recurrence.every(30, "days", "2026-11-20"),
        ...     cost=250.0,
        ... )
        >>> value = task.long_run_value("2026-11-20", 0.07)
        >>> round(value.npv), value.method
        (45112, 'closed form')
        """
        start = _date(start, "start")
        discount = _discount.yearly(discount, required=True)
        assert discount is not None
        given, discount = discount, _discount.net(discount, escalation)
        assert discount is not None
        if escalation is not None and not discount > 0.0:
            raise ValueError(
                f"The cost escalates at {_discount.percent(escalation)} a "
                f"year, no slower than the discount of "
                f"{_discount.percent(given)}: repeating for ever, it is "
                "worth without end. Ask for a period's value with "
                "due(start, end, discount=, escalation=)."
            )
        if not (_is_number(precision) and 0.0 < float(precision) < 1.0):
            raise ValueError(
                "precision must be a share of the value, more than 0 and "
                f"less than 1 (such as 1e-9); got {precision!r}."
            )
        self._check_for_ever(start, discount)
        assert self.cost is not None
        notes = [_discount.words(given, start, escalation)]
        gap = self._plain_gap()
        to: Optional[date] = None
        count: Optional[int] = None
        if self.cost == 0.0:
            npv, method, left = 0.0, "closed form", 0.0
            notes.append("The task costs nothing each time.")
        elif gap is not None:
            # The interval repeats exactly from the second time the task
            # is done (the first may be an overdue one, done on start).
            assert isinstance(self.every, Recurrence)
            assert self.every.anchor is not None
            known = max(start, self.every.anchor, self.last_done or start)
            first, second = self._due(
                start, known + timedelta(days=2 * gap), {}
            ).done[:2]
            npv = _present_value(self.cost, discount, start, [first])
            npv += self.cost * _discount.perpetuity(
                discount,
                _discount.years_from(start, second),
                gap / _discount.DAYS_A_YEAR,
            )
            method, left = "closed form", 0.0
            notes.append(
                f"Done on {first}, then every {gap} days from {second}, "
                "for ever: a geometric sum."
            )
        else:
            npv, left, to, count = self._summed(
                start, discount, float(precision)
            )
            method = "summed"
            notes.append(
                f"Summed date by date to {to} ({count} times); a task done "
                f"every day after it would add {left:.3g}, which this one "
                "cannot exceed."
            )
        return LongRunValue(
            task=self.name,
            start=start,
            discount=given,
            npv=npv,
            eac=_discount.equivalent_annual(npv, discount, math.inf),
            method=method,
            precision=left,
            summed_to=to,
            count=count,
            notes=tuple(notes),
        )

    def _check_for_ever(self, start: date, discount: float) -> None:
        """Refuse what has no long-run value, saying what to run."""
        if self.cost is None:
            raise ValueError(
                f"Task {self.name!r} has no cost to value: give cost=."
            )
        if discount == 0.0:
            raise ValueError(
                "With no discount (discount=0) a task repeating for ever "
                "costs without end: give a yearly rate above 0, or ask for "
                "a period's cost with due(start, end)."
            )
        kinds = self._limits()
        if not all(isinstance(kind, _CalendarLimit) for kind in kinds):
            raise ValueError(
                f"Task {self.name!r} has a limit by use, which follows a "
                "usage forecast, and a forecast ends: ask for the present "
                "value over the forecast with due(start, end, usage=..., "
                "discount=...)."
            )
        ends = [k.rule.given[-1] for k in kinds if k.rule.rule == "on"]
        if self.visits is not None and self.visits.rule == "on":
            what = "is done at visits on given dates"
            last = self.visits.given[-1]
        elif len(ends) == len(kinds):
            what, last = "is on given dates", max(ends)
        else:
            return
        raise ValueError(
            f"Task {self.name!r} {what}, which end on {last}: "
            + (
                "nothing is left of it from then."
                if last < start
                else f"due({start}, {last}, discount=...) gives its whole "
                "present value."
            )
        )

    def _plain_gap(self) -> Optional[int]:
        """The days between a plain interval's dates (every so many days
        or weeks, and nothing else that moves a date); None for any other
        task."""
        rule = self.every
        if (
            self.also is not None
            or self.visits is not None
            or not isinstance(rule, Recurrence)
            or rule.rule != "every"
            or rule.unit not in ("days", "weeks")
            or rule.season is not None
            or rule.excluding
        ):
            return None
        return rule.every_n * (7 if rule.unit == "weeks" else 1)

    def _summed(self, start: date, discount: float, precision: float):
        """The task's present value summed date by date, to where a task
        done every day after could add no more than ``precision`` of it:
        the value, that bound, the last date summed to and the count."""
        assert self.cost is not None
        day = 1.0 / _discount.DAYS_A_YEAR
        rate = _discount.continuous(discount)
        latest = (date.max - start).days
        days = min(int(_FIRST_YEARS * _discount.DAYS_A_YEAR), latest)
        # The whole value is no more than a task every day from start, so
        # the bound cannot be under the precision before this many days.
        soonest = -math.log(precision) / rate * _discount.DAYS_A_YEAR - 1.0
        while soonest <= latest:
            end = start + timedelta(days=days)
            done = self._due(start, end, {}).done
            npv = _present_value(self.cost, discount, start, done)
            # Every day after ``end``, for ever.
            left = self.cost * _discount.perpetuity(
                discount, (days + 1) * day, day
            )
            if npv > 0.0 and left <= precision * npv:
                return npv, left, end, len(done)
            if days == latest:
                break
            # Where the bound falls to the precision of what is summed so
            # far (the whole value is no less); twice as far with nothing
            # summed yet.
            if npv > 0.0:
                more = math.log(left / (precision * npv)) / rate
                days = days + int(more * _discount.DAYS_A_YEAR) + 1
            else:
                days = 2 * days
            days = min(days, latest)
        raise ValueError(
            f"At {_discount.percent(discount)} a year the sum does not "
            f"settle to precision={precision:g} by the year 9999, as far "
            "as dates go: give a looser precision=, or ask for a period's "
            "present value with due(start, end, discount=...)."
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
        }
        for key, limit in zip(("every", "also"), self._limits()):
            out[key] = limit.to_dict()
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
    tasks: Sequence[Task],
    period,
    start,
    end,
    usage=None,
    *,
    discount=None,
    escalation=None,
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
    discount : float, optional
        A yearly discount rate (0.07 for 7% a year): the result then has
        each period's ``npv`` (its costs' present value on ``start``, each
        task discounted from the date it is done), ``total_npv`` and
        ``eac``. By default None: no present values.
    escalation : float, optional
        With ``discount``, the effective yearly rate every cost
        escalates at from its price today (0.03 for 3% a year): the
        present values (and ``eac``) take each cost grown to when it
        falls, at the net yearly rate ``(1 + discount) / (1 + escalation)
        - 1``. By default None: today's prices.

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
    discount = _discount.yearly(discount)
    given, discount = discount, _discount.net(discount, escalation)
    spans = _periods(start, end, period)
    firsts = [a for a, _ in spans]
    n = len(spans)
    count, hours, cost = [0] * n, [0.0] * n, [0.0] * n
    worth: List[List[float]] = [[] for _ in range(n)]
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
                if discount is not None:
                    worth[index].append(
                        task.cost * _discount.factor_on(discount, start, day)
                    )
        by_task[task.name] = tuple(mine)
    npv = total_npv = eac = None
    notes: List[str] = []
    if discount is not None:
        npv = tuple(math.fsum(x) for x in worth)
        total_npv = math.fsum(npv)
        eac = _discount.equivalent_annual(
            total_npv, discount, _years(start, end)
        )
        notes.append(_discount.words(given, start, escalation))
        missing = [t.name for t in tasks if t.cost is None]
        if missing:
            notes.append(
                "The present values leave out the tasks with no cost: "
                f"{', '.join(missing)}."
            )
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
        discount=given,
        npv=npv,
        total_npv=total_npv,
        eac=eac,
        notes=tuple(notes),
    )
