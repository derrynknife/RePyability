"""What a scheduled task is worth: the cost with it against without it,
on real dates (UC-50 of this package's ``README.md``).

``task_value(task, does, usage)`` takes the task (its dates, from
``Task.due``), what it does to a part, as plain data, and the usage
forecast, and returns a ``TaskValue``: what the part costs with the task
and without it, and the difference, by calendar period and in all, as
present values with a ``discount``. What the task does is said by the
caller, not saved on the ``Task`` (that record is the caller's):

- ``Replaces(part)``: the task puts a new part in at every date it is
  done, whatever the part's age (block replacement on dates); without
  it, the part runs to failure.

Between two of the task's dates the part is a renewal process from new,
so its expected failures to any time are a sum of
``NonRepairable.failures`` over the stretches between the dates. Each
side's failure cost is discounted as ``failures_by_period`` discounts it
(``_forecast._accrued_npv``); a planned replacement from its own instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from repyability.utils.checks import is_number

from . import _discount
from ._forecast import (
    _YEAR_SECONDS,
    NPV_TOLERANCE,
    _accrued_npv,
    _model_label,
    _period_edges,
    _resolve_unit,
    _tolerance,
    _usage,
)
from ._task import Task
from ._usage import _Plain


@dataclass(frozen=True)
class Replaces:
    """What a task does: it replaces ``part`` with a new one every time it
    is done, whatever the part's age (and the part is replaced when it
    fails, too). Plain data for
    [`task_value`][repyability.schedule.task_value].

    Parameters
    ----------
    part : NonRepairable
        The part, with its life in the unit of the usage forecast's model
        time and its ``unplanned_cost`` (a failure's). Its
        ``planned_cost`` is the task's price unless the task has a
        ``cost``.
    """

    part: Any

    def __post_init__(self) -> None:
        from repyability.non_repairable import NonRepairable

        if not isinstance(self.part, NonRepairable):
            raise TypeError(
                "Replaces takes the part the task replaces, a "
                f"NonRepairable; got {self.part!r}."
            )

    def describe(self) -> str:
        """What the task does, in words."""
        return "replaces the part with a new one"


@dataclass(frozen=True)
class TaskValue(_Plain):
    """What a scheduled task is worth against not doing it: returned by
    [`task_value`][repyability.schedule.task_value].

    Money is a present value at the forecast's start when a ``discount``
    is given, and the plain expected cost otherwise. Each period's
    numbers are its own and add up to ``total``'s.

    Attributes
    ----------
    task : str
        The task's name.
    does : str
        What the task does, in words.
    with_words, without_words : str
        What "with the task" and "without it" were taken to mean.
    period : str
        ``"day"``, ``"week"``, ``"month"``, ``"quarter"`` or ``"year"``.
    starts, ends : tuple of datetime
        Each period's start and end at the site.
    unit : str
        The part's unit of time.
    age : float
        The part's age at the forecast's start, in ``unit``.
    dates : tuple of datetime.date
        The dates the task is done in the forecast.
    model_times : numpy.ndarray
        The part's clock (operating time since the forecast's start, in
        ``unit``) at each time the task is done.
    discount : float or None
        The yearly discount rate given; None when none was.
    with_task, without_task : numpy.ndarray
        What the part costs in each period with the task (its failures
        and the task's own cost) and without it (its failures): present
        values with a discount.
    value : numpy.ndarray
        ``without_task - with_task``: what the task saves in each period,
        net of its own cost (positive when it pays).
    task_cost : numpy.ndarray
        The task's own cost in each period (in ``with_task``).
    cost_with, cost_without : numpy.ndarray
        The same two costs undiscounted.
    failures_with, failures_without : numpy.ndarray
        The expected failures in each period with and without the task.
    failures_avoided : numpy.ndarray
        ``failures_without - failures_with``.
    total : dict
        The whole forecast's: ``with_task``, ``without_task``, ``value``,
        ``task_cost``, ``cost_with``, ``cost_without``,
        ``failures_with``, ``failures_without``, ``failures_avoided``
        and ``count`` (the times the task is done).
    precision : float
        How far the true total ``value`` can be from the one given: the
        two sides' present values are each within half a bracket of
        theirs (as ``FailuresByPeriod.npv_precision``); 0 undiscounted.
    eac : dict or None
        With a discount, the equivalent annual cost of ``with_task`` and
        ``without_task`` and of ``value`` over the forecast: the steady
        amount a year worth each.
    notes : tuple of str
        What was assumed.
    """

    task: str
    does: str
    with_words: str
    without_words: str
    period: str
    starts: Tuple[datetime, ...]
    ends: Tuple[datetime, ...]
    unit: str
    age: float
    dates: Tuple[date, ...]
    model_times: np.ndarray
    discount: Optional[float]
    with_task: np.ndarray
    without_task: np.ndarray
    value: np.ndarray
    task_cost: np.ndarray
    cost_with: np.ndarray
    cost_without: np.ndarray
    failures_with: np.ndarray
    failures_without: np.ndarray
    failures_avoided: np.ndarray
    total: Dict[str, Any]
    precision: float
    eac: Optional[Dict[str, float]]
    notes: Tuple[str, ...]

    def __repr__(self) -> str:
        money = "present value" if self.discount is not None else "cost"
        return (
            f"TaskValue({self.task!r} over {len(self.starts)} "
            f"{self.period}s from {self.starts[0]:%Y-%m-%d}: {money} with "
            f"{self.total['with_task']:.6g}, without "
            f"{self.total['without_task']:.6g}, value "
            f"{self.total['value']:.6g}; failures avoided "
            f"{self.total['failures_avoided']:.4g})"
        )


def _price(task: Task, part, notes: List[str]) -> float:
    """The planned replacement's one price: the task's cost, else the
    part's planned cost; refused when both are given and differ, or
    neither is."""
    planned = getattr(part, "cp", None)
    if task.cost is not None and planned is not None:
        if not np.isclose(task.cost, planned, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"Task {task.name!r} costs {task.cost:g} and the part's "
                f"planned_cost is {planned:g}: the task is the planned "
                "replacement, so they are one price. Give the same number "
                "to both, or leave one out."
            )
        return float(task.cost)
    if task.cost is not None:
        return float(task.cost)
    if planned is not None:
        notes.append(
            f"The task has no cost: each time it is done costs the part's "
            f"planned_cost, {planned:g}."
        )
        return float(planned)
    raise ValueError(
        f"Task {task.name!r} has no cost and the part has no "
        "planned_cost: give the task's cost= (what a planned replacement "
        "costs), or the part's planned_cost=."
    )


def _done_instants(task: Task, usage, notes: List[str]):
    """The task's dates in the forecast, each with the real seconds from
    the forecast's start to when it is done: midnight of its date, or,
    for a limit by use done on the day it falls due, the moment the meter
    reaches its reading. A date before the forecast starts (an overdue
    task, done at the start) counts at the start."""
    origin, start_s = usage._origin, usage._start_s
    span = usage._end_s - start_s
    last = (usage.end - timedelta(microseconds=1)).date()
    due = task.due(usage.start.date(), last, usage=usage)
    dates: List[date] = []
    seconds: List[float] = []
    for day, moment in zip(due.done, due.moments):
        if task.visits is None and day == moment.date():
            when = moment
        else:
            when = datetime.combine(day, datetime.min.time())
        s = (usage._clock.instant(when, "done") - origin).total_seconds()
        s = max(s - start_s, 0.0)
        if s < span:
            dates.append(day)
            seconds.append(s)
    notes.extend(due.notes)
    return due, dates, np.array(seconds, float)


def _renewals(part, age: float, times: np.ndarray):
    """The part's expected failures from 0 to each of an array of model
    times, as a function: run to failure from ``age`` when ``times`` is
    empty, else renewed at each of ``times`` (from new after each)."""
    cuts = np.unique(times)
    starts = np.concatenate([[0.0], cuts[cuts > 0.0]])
    first_age = 0.0 if cuts.size and cuts[0] <= 0.0 else age
    lengths = np.append(np.diff(starts), np.inf)

    def expected(ends) -> np.ndarray:
        ends = np.asarray(ends, float)
        out = np.zeros(ends.shape)
        for k, (s, length) in enumerate(zip(starts, lengths)):
            ran = np.clip(ends - s, 0.0, length)
            if not np.any(ran > 0.0):
                continue
            out += np.asarray(
                part.failures(
                    ran, age=first_age if k == 0 else 0.0
                ).expected_failures,
                float,
            ).reshape(ends.shape)
        return out

    return expected


def task_value(
    task,
    does,
    usage,
    *,
    age: float = 0.0,
    unit: Optional[str] = None,
    discount: Optional[float] = None,
    period: str = "month",
    npv_tolerance: float = NPV_TOLERANCE,
) -> TaskValue:
    """What a scheduled task is worth against not doing it, by calendar
    period and in all: the part's cost with the task, without it, and the
    difference, as present values with a ``discount``.

    ``Replaces(part)``: the task puts a new part in at every date it is
    done (``Task.due`` over the forecast, with its visits, second limit
    and ``resets``), whatever the part's age, and a failure in between is
    replaced too without moving the dates (block replacement on dates).
    Without the task the part runs to failure from the same age over the
    same dates. Between two of the task's dates the part is a renewal
    process from new, so the expected failures are
    [`NonRepairable.failures`][repyability.NonRepairable.failures]' over
    each stretch (exact); a failure costs the part's ``unplanned_cost``
    and the task its ``cost``. A failure's cost is discounted from when
    it is expected to fall, as
    [`failures_by_period`][repyability.schedule.failures_by_period]
    discounts it (to within ``precision``), and the task's own cost from
    when it is done.

    Parameters
    ----------
    task : Task
        The task, by the calendar or by use (``every=Usage(...)``). A
        calendar task is done at midnight of its date (or of its visit);
        a task by use when the meter reaches its reading.
    does : Replaces
        What the task does: ``Replaces(part)``, the part a
        ``NonRepairable`` with an ``unplanned_cost``. Its
        ``planned_cost`` is the task's price where the task has no
        ``cost``; given both, they must agree.
    usage : UsageForecast
        The map between dates and the part's operating time
        ([`DutyPlan.forecast`][repyability.schedule.DutyPlan.forecast], or
        ``UsageForecast.calendar`` for a life in calendar time); its
        start is "now", and its span the window valued.
    age : float, optional
        The part's age at the forecast's start, in ``unit``; by default 0.
    unit : str, optional
        The unit the part's life is in; by default the one its model's
        label names, else hours (with a note).
    discount : float, optional
        A yearly discount rate (0.07 for 7% a year): money is then a
        present value at the forecast's start. By default None: plain
        expected costs.
    period : str, optional
        ``"day"``, ``"week"``, ``"month"`` (the default), ``"quarter"`` or
        ``"year"``.
    npv_tolerance : float, optional
        How close each side's failure cost is discounted, as for
        ``failures_by_period``; by default 1e-3.

    Returns
    -------
    TaskValue

    Raises
    ------
    TypeError
        ``task`` is not a ``Task``, or ``does`` is not a ``Replaces``.
    ValueError
        The part has no ``unplanned_cost``; the task and the part give
        two prices for the planned replacement, or neither gives one;
        or as ``Task.due`` and ``failures_by_period``.

    Examples
    --------
    A pump's wear ring (Weibull, scale 9,000 operating hours, shape 2.5),
    5,200 h old, $4,800 when it fails; a new one ($600) at every
    6-monthly service; the pump runs Monday to Friday, 06:00 to 22:00:

    >>> import surpyval as surv
    >>> from repyability import NonRepairable
    >>> from repyability.schedule import (
    ...     DutyPlan, Recurrence, Replaces, Shifts, Task, task_value
    ... )
    >>> ring = NonRepairable(
    ...     surv.Weibull.from_params([9000.0, 2.5]), unplanned_cost=4800.0
    ... )
    >>> service = Task(
    ...     "new wear ring",
    ...     every=Recurrence.every(6, "months", "2026-12-01"),
    ...     cost=600.0,
    ... )
    >>> usage = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
    ...     "2026-11-16", "2027-12-01"
    ... )
    >>> worth = task_value(service, Replaces(ring), usage, age=5200.0)
    >>> [str(d) for d in worth.dates]
    ['2026-12-01', '2027-06-01']
    >>> total = worth.total
    >>> round(total["with_task"], 2), round(total["without_task"], 2)
    (1550.82, 2987.56)
    >>> round(total["value"], 2), round(total["failures_avoided"], 3)
    (1436.74, 0.549)
    """
    if not isinstance(task, Task):
        raise TypeError(f"task must be a Task; got {task!r}.")
    if not isinstance(does, Replaces):
        raise TypeError(
            "does must say what the task does: Replaces(part) for a task "
            "that puts a new part in. The value of an inspection or of a "
            "service is not built yet: for a hidden failure's test, "
            "repyability.maintenance.failure_finding gives the long-run "
            "numbers in model time. Got " + repr(does) + "."
        )
    usage = _usage(usage)
    part = does.part
    notes: List[str] = []
    unit = _resolve_unit(
        unit, _model_label(getattr(part, "reliability", None)), "life", notes
    )
    if not (is_number(age) and 0.0 <= float(age) < np.inf):
        raise ValueError(f"age must be a number, at least 0; got {age!r}.")
    age = float(age)
    discount = _discount.yearly(discount)
    npv_tolerance = _tolerance(npv_tolerance)
    failure_cost = getattr(part, "cu", None)
    if failure_cost is None:
        raise ValueError(
            "The part has no unplanned_cost: without a failure's cost the "
            "task saves nothing to set against its own. Give the "
            "NonRepairable unplanned_cost=."
        )
    price = _price(task, part, notes)

    found, seconds = _period_edges(usage, period)
    n = seconds.size
    ends = np.atleast_1d(usage.model_time(list(found.ends), unit=unit))
    ends = np.maximum.accumulate(np.maximum(ends, 0.0))
    due, dates, done_s = _done_instants(task, usage, notes)
    starts_s = usage._start_s
    clock_s = usage._start_s + done_s
    scale = usage._unit(unit)
    base = usage._at_s("hours", starts_s)
    times = (usage._at_s("hours", clock_s) - base) / scale
    times = np.maximum(np.asarray(times, float).reshape(done_s.shape), 0.0)

    with_failures = _renewals(part, age, times)
    without_failures = _renewals(part, age, np.array([]))
    fails_with = np.diff(with_failures(ends), prepend=0.0)
    fails_without = np.diff(without_failures(ends), prepend=0.0)
    in_period = np.minimum(
        np.searchsorted(seconds, done_s, side="right"), n - 1
    )
    count = np.bincount(in_period, minlength=n).astype(float)
    task_cost: np.ndarray = price * count
    cost_with = failure_cost * fails_with + task_cost
    cost_without = failure_cost * fails_without

    precision = 0.0
    eac: Optional[Dict[str, float]] = None
    if discount is None or discount == 0.0 or ends[-1] <= 0.0:
        money_with, money_without = cost_with, cost_without
        money_task = task_cost
    else:

        def side(expected):
            def costs(model_ends: np.ndarray) -> np.ndarray:
                return failure_cost * np.diff(
                    expected(model_ends), prepend=0.0
                )

            by_period, _, half = _accrued_npv(
                usage, seconds, unit, discount, npv_tolerance, costs, notes
            )
            return by_period, half

        failures_with_pv, half_with = side(with_failures)
        failures_without_pv, half_without = side(without_failures)
        planned = price * _discount.factor(discount, done_s / _YEAR_SECONDS)
        money_task = np.bincount(
            in_period, weights=np.atleast_1d(planned), minlength=n
        ).astype(float)
        money_with = failures_with_pv + money_task
        money_without = failures_without_pv
        precision = half_with + half_without
    value = money_without - money_with
    if discount is not None:
        years = float(seconds[-1]) / _YEAR_SECONDS
        eac = {
            key: _discount.equivalent_annual(
                float(np.sum(amounts)), discount, years
            )
            for key, amounts in (
                ("with_task", money_with),
                ("without_task", money_without),
                ("value", value),
            )
        }
    with_words = (
        f"With the task: the part replaced with a new one each time the "
        f"task is done ({len(dates)} times in the forecast), at {price:g} "
        "whatever its age, and when it fails, at "
        f"{failure_cost:g}; a failure does not move the task's dates."
    )
    without_words = (
        "Without the task: the part run to failure from the same age "
        f"({age:g} {unit}) over the same dates, replaced only when it "
        f"fails, at {failure_cost:g}."
    )
    notes.append(
        "Operating time comes from the duty plan "
        f"({usage.describe()}); the part's age is at the forecast's start."
    )
    notes.append(
        "A calendar task is done at midnight of its date (or its visit's), "
        "a task by use when the meter reaches its reading; a planned "
        "replacement takes no time on the part's clock."
    )
    if discount is not None:
        notes.append(
            f"{_discount.words(discount, usage.start)} A failure's expected "
            "cost is discounted from when it is expected to fall, each "
            f"side to within its bracket ({precision:.3g} on the value in "
            "all); the task's cost from when it is done."
        )
    total = {
        "with_task": float(np.sum(money_with)),
        "without_task": float(np.sum(money_without)),
        "value": float(np.sum(value)),
        "task_cost": float(np.sum(money_task)),
        "cost_with": float(np.sum(cost_with)),
        "cost_without": float(np.sum(cost_without)),
        "failures_with": float(np.sum(fails_with)),
        "failures_without": float(np.sum(fails_without)),
        "failures_avoided": float(np.sum(fails_without - fails_with)),
        "count": len(dates),
    }
    return TaskValue(
        task=task.name,
        does=does.describe(),
        with_words=with_words,
        without_words=without_words,
        period=period,
        starts=found.starts,
        ends=found.ends,
        unit=unit,
        age=age,
        dates=tuple(dates),
        model_times=times,
        discount=discount,
        with_task=np.asarray(money_with, float),
        without_task=np.asarray(money_without, float),
        value=np.asarray(value, float),
        task_cost=np.asarray(money_task, float),
        cost_with=cost_with,
        cost_without=cost_without,
        failures_with=fails_with,
        failures_without=fails_without,
        failures_avoided=fails_without - fails_with,
        total=total,
        precision=float(precision),
        eac=eac,
        notes=tuple(notes),
    )
