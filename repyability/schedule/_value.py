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
- ``Finds(device, demand_rate=, consequence_cost=)``: the task tests a
  protective device for a hidden failure (UC-54) and has a failure it
  finds repaired; without it, the device is never tested.

Either way each side is a count of events that accrues with the part's
operating time, each at one cost (a part's failures; the demands that
meet a device failed), and costs at instants (the task's own; the
repairs of what a test finds). With ``Replaces``, between two of the
task's dates the part is a renewal process from new, so its expected
failures to any time are a sum of ``NonRepairable.failures`` over the
stretches between the dates. With ``Finds``, the device's chance of
being failed over time is ``maintenance._finding_window``'s, with the
tests at the task's times on the device's clock. Each side's accrued
cost is discounted as ``failures_by_period`` discounts it
(``_forecast._accrued_npv``); a cost at an instant from that instant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from ._service import Condition, Services, simulated  # noqa: F401
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
class Finds:
    """What a task does: it tests ``device``, a protective device (a
    relief valve, a trip, a standby pump's start), for a failure nobody
    would otherwise see, and a failure it finds is repaired as new.
    Plain data for [`task_value`][repyability.schedule.task_value].

    Parameters
    ----------
    device : NonRepairable
        The device, with its life (a failure is hidden) in the unit of
        the usage forecast's model time. Its own costs are not used.
    demand_rate : float
        How often the protected event comes, per unit of the device's
        time, at least 0.
    consequence_cost : float
        What a demand that meets the device failed costs, at least 0.
    repair_cost : float, optional
        What repairing a failure costs, found by a test or shown by a
        demand; by default 0.
    demands_reveal : bool, optional
        Whether a demand that meets the device failed shows the failure,
        which is repaired as new then (the default), with the task and
        without it. False: only a test finds a failure, as
        [`failure_finding`][repyability.failure_finding] has it.
    """

    device: Any
    demand_rate: float = field(kw_only=True)
    consequence_cost: float = field(kw_only=True)
    repair_cost: float = field(default=0.0, kw_only=True)
    demands_reveal: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        from repyability.maintenance._decisions import (
            _check_protective,
            _number,
        )
        from repyability.non_repairable import NonRepairable

        if not isinstance(self.device, NonRepairable):
            raise TypeError(
                "Finds takes the device the task tests, a NonRepairable; "
                f"got {self.device!r}."
            )
        _check_protective(self.device)
        for name in ("demand_rate", "consequence_cost", "repair_cost"):
            _number(getattr(self, name), name)
        if not isinstance(self.demands_reveal, bool):
            raise ValueError(
                "demands_reveal must be True or False; got "
                f"{self.demands_reveal!r}."
            )

    def describe(self) -> str:
        """What the task does, in words."""
        return "tests the device for a hidden failure"


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
        two sides' present values are each within their integrals'
        error estimates of theirs (as
        ``FailuresByPeriod.npv_precision``); 0 undiscounted.
    eac : dict or None
        With a discount, the equivalent annual cost of ``with_task`` and
        ``without_task`` and of ``value`` over the forecast: the steady
        amount a year worth each.
    notes : tuple of str
        What was assumed.
    found : numpy.ndarray or None
        With ``Finds``: the expected failures the tests find in each
        period (``total["found"]`` in all). None otherwise.
    unavailability_with, unavailability_without : numpy.ndarray or None
        With ``Finds``: the device's mean unavailability in each period
        with the task and without it, the share of the period's
        operating time it is failed (in a period it does not run, its
        chance of being failed then); ``total`` has each over the whole
        forecast. None otherwise.
    method : str
        ``"exact"``, or ``"simulated"`` for a service with an effect.
    value_lower, value_upper : numpy.ndarray or None
        Simulated: each period's value's two-sided interval at
        ``confidence`` (``total`` has ``value_lower`` and ``value_upper``
        over the whole forecast). None when exact.
    confidence, n_simulations, seed : float, int, int or None
        Simulated: the interval's level, the runs and their seed (to
        repeat them). None when exact.

    With ``Services`` and no effect, the unit's own costs are the same
    with and without the service and are left out of both sides:
    ``with_task`` is the service's cost, ``without_task`` 0, and the
    failures are NaN (not worked out), none avoided. With an effect,
    each side is the unit's own costs (its plan's prices) in the
    simulated runs, ``with_task`` adding the service's cost.

    With ``Finds``, "the part" is the device and a "failure" above is a
    demand that meets it failed: ``failures_with``, ``failures_without``
    and ``failures_avoided`` count those, and ``with_task`` is the
    tests' cost, the repairs' and those demands'.
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
    found: Optional[np.ndarray] = None
    unavailability_with: Optional[np.ndarray] = None
    unavailability_without: Optional[np.ndarray] = None
    method: str = "exact"
    value_lower: Optional[np.ndarray] = None
    value_upper: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    n_simulations: Optional[int] = None
    seed: Optional[int] = None

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


@dataclass(frozen=True)
class _Plan:
    """The two sides of a task's value: each a count of events that
    accrues with the part's operating time, at ``event_cost`` each, and,
    with the task, its own cost and ``extra`` at instants."""

    with_events: Callable[[np.ndarray], np.ndarray]
    without_events: Callable[[np.ndarray], np.ndarray]
    event_cost: float
    extra_seconds: np.ndarray  # costs at instants besides the task's own
    extra_amounts: np.ndarray
    error: float  # how far the costs' total can be out, undiscounted
    with_words: str
    without_words: str
    notes: Tuple[str, ...]
    with_hidden: Any = None  # the device's hidden failures (Finds)
    without_hidden: Any = None
    found: Optional[np.ndarray] = None  # at each of extra_seconds


def _replacing(part, price, failure_cost, age, unit, times, count) -> _Plan:
    """``Replaces``: block replacement at ``times`` against running to
    failure."""
    return _Plan(
        with_events=_renewals(part, age, times),
        without_events=_renewals(part, age, np.array([])),
        event_cost=failure_cost,
        extra_seconds=np.array([]),
        extra_amounts=np.array([]),
        error=0.0,
        with_words=(
            f"With the task: the part replaced with a new one each time "
            f"the task is done ({count} times in the forecast), at "
            f"{price:g} whatever its age, and when it fails, at "
            f"{failure_cost:g}; a failure does not move the task's dates."
        ),
        without_words=(
            "Without the task: the part run to failure from the same age "
            f"({age:g} {unit}) over the same dates, replaced only when it "
            f"fails, at {failure_cost:g}."
        ),
        notes=(
            "A calendar task is done at midnight of its date (or its "
            "visit's), a task by use when the meter reaches its reading; "
            "a planned replacement takes no time on the part's clock.",
        ),
    )


def _testing(
    does: Finds, price, age, unit, times, done_s, window, count
) -> _Plan:
    """``Finds``: the device tested at ``times`` against never tested."""
    from repyability.maintenance._finding_window import _hidden

    demand = float(does.demand_rate)
    repair = float(does.repair_cost)
    consequence = float(does.consequence_cost)
    reveal = demand if does.demands_reveal else 0.0
    each = consequence + (repair if reveal > 0.0 else 0.0)
    # Tests at one time on the device's clock (it did not run between
    # them) are one test of it: the first finds what there is to find.
    tests, first = np.unique(np.minimum(times, window), return_index=True)
    first = first[tests > 0.0]  # at 0 the device is working
    tests = tests[tests > 0.0]
    model = does.device.reliability
    tested = _hidden(model, age, tests, window, reveal)
    never = _hidden(model, age, np.array([]), window, reveal)
    if does.demands_reveal:
        shown = (
            f", and shows the failure, repaired then at {repair:g}"
            if demand > 0.0
            else ""
        )
        left = (
            "a failure stays until a demand meets it, which costs "
            f"{consequence:g} and has it repaired at {repair:g}"
        )
    else:
        shown = "; only a test finds a failure"
        left = (
            "a failure is never found, and every demand after it costs "
            f"{consequence:g}"
        )
    return _Plan(
        with_events=lambda ends: demand * tested.failed_time(ends),
        without_events=lambda ends: demand * never.failed_time(ends),
        event_cost=each,
        extra_seconds=done_s[first],
        extra_amounts=repair * tested.found,
        error=(
            each * demand * (tested.time_error + never.time_error)
            + repair * tested.found_error
        ),
        with_words=(
            f"With the task: the device tested each time the task is "
            f"done ({count} times in the forecast), at {price:g} a test; "
            f"a failure a test finds repaired as new at {repair:g}; a "
            f"demand (at a rate of {demand:g}, in {unit}) that meets "
            f"the device failed costs {consequence:g}{shown}."
        ),
        without_words=(
            "Without the task: the device never tested, from the same "
            f"age ({age:g} {unit}) over the same dates: {left}."
        ),
        notes=(
            "The device is taken to be working at the forecast's start. "
            "A calendar task is done at midnight of its date (or its "
            "visit's), a task by use when the meter reaches its reading; "
            "a test and a repair take no time, and a test never misses "
            "a failure.",
            "Demands come at random, at their rate per unit of the "
            f"device's own time ({unit}), so only while it runs, and "
            "independently of the device.",
        ),
        with_hidden=tested,
        without_hidden=never,
        found=tested.found,
    )


def _in_periods(ends: np.ndarray, hidden) -> np.ndarray:
    """A device's mean unavailability in each period of operating time
    ending at ``ends``: its chance of being failed then where the period
    has none."""
    failed = np.diff(hidden.failed_time(ends), prepend=0.0)
    ran = np.diff(ends, prepend=0.0)
    at = hidden.unavailability(ends)
    return np.where(ran > 0.0, failed / np.where(ran > 0.0, ran, 1.0), at)


def _servicing(
    does: Services,
    price: float,
    usage,
    unit: str,
    age: float,
    times: np.ndarray,
    done_s: np.ndarray,
    seconds: np.ndarray,
    discount: Optional[float],
    mc_samples: Optional[int],
    seed,
    confidence: float,
    count: int,
) -> Tuple[_Plan, Any]:
    """``Services``: the words and notes of a service's two sides, and,
    with an effect, its simulated runs (``_service.simulated``); None
    for a service with no effect, which is valued exactly."""
    from repyability.value._rates import per_unit

    from ._usage import UNITS

    words = (
        f"With the task: the unit serviced each time the task is done "
        f"({count} times in the forecast), at {price:g} a time"
    )
    common = (
        "A calendar task is done at midnight of its date (or its visit's), "
        "a task by use when the meter reaches its reading; a service "
        "takes no time."
    )
    if does.no_effect:
        why = (
            "a factor of 1 leaves the unit as old as it was"
            if does.effect is not None
            else "no effect was given"
        )
        return (
            _Plan(
                with_events=np.zeros_like,
                without_events=np.zeros_like,
                event_cost=0.0,
                extra_seconds=np.array([]),
                extra_amounts=np.array([]),
                error=0.0,
                with_words=(
                    f"{words}, with no effect on its life ({why}): the "
                    "service is an accepted cost. The unit's own costs are "
                    "the same with and without it, and are left out of "
                    "both sides."
                ),
                without_words=(
                    "Without the task: the unit as it is; its own costs, "
                    "the same as with the task, left out."
                ),
                notes=(
                    common,
                    "The service's effect on the life was taken as none: "
                    "its value is minus its own cost. State an effect "
                    "(a maintenance.Restoration, or a schedule.Condition) "
                    "to credit it with a benefit.",
                ),
            ),
            None,
        )
    # The diagram's clock is calendar time: in the life's unit, or in
    # hours for a life in a unit of use (its duty a rate of that use).
    use = None if unit in UNITS else unit
    clock_unit = "hours" if use is not None else unit
    hours = UNITS[clock_unit]
    clock = done_s / 3600.0 / hours
    ends = seconds / 3600.0 / hours
    rate = 0.0 if not discount else per_unit(discount, hours)
    sim = simulated(
        does,
        usage,
        clock_unit,
        age,
        clock,
        ends,
        rate,
        mc_samples,
        seed,
        confidence,
        use=use,
    )
    return (
        _Plan(
            with_events=np.zeros_like,
            without_events=np.zeros_like,
            event_cost=0.0,
            extra_seconds=np.array([]),
            extra_amounts=np.array([]),
            error=0.0,
            with_words=(
                f"{words}, {does.describe()[len('services the unit, '):]}; "
                "the unit's failures, repairs and downtime at its plan's "
                "prices."
            ),
            without_words=(
                "Without the task: the same plan from the same age "
                f"({age:g} {unit}) over the same dates, never serviced."
            ),
            notes=(
                common,
                f"Simulated: {sim.n} runs of each side from seed "
                f"{sim.seed}, on the same random numbers (the unit draws "
                "the same failures until a service changes them); the "
                f"value's interval is two-sided at {confidence:g}.",
            ),
        ),
        sim,
    )


def task_value(
    task,
    does,
    usage,
    *,
    age: float = 0.0,
    unit: Optional[str] = None,
    discount: Optional[float] = None,
    escalation: Optional[float] = None,
    period: str = "month",
    npv_tolerance: float = NPV_TOLERANCE,
    mc_samples: Optional[int] = None,
    seed: Optional[int] = None,
    confidence: float = 0.9,
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

    ``Finds(device, demand_rate=, consequence_cost=, repair_cost=)``: the
    task tests a protective device for a hidden failure at every date it
    is done, at the task's ``cost`` a test, and a failure it finds is
    repaired as new. Demands come at ``demand_rate`` per unit of the
    device's time, and one that meets the device failed costs
    ``consequence_cost`` (and, by default, shows the failure, which is
    repaired then). Without the task the device is never tested. The
    device's chance of being failed over time, with the tests at the
    task's times on its clock, is
    [`failure_finding_window`][repyability.failure_finding_window]'s
    (numerical, its error in ``precision``); the demands' cost is
    discounted as it accrues, and a test and the repair of what it finds
    from when the task is done. The result's "failures" are then the
    demands that meet the device failed, and it has the failures
    ``found`` and the device's unavailability with and without the task.

    ``Services(plan, effect=)``: the task services the unit without
    replacing it. With no effect (the default) the service is an accepted
    cost: its value is minus its cost's present value, exactly. With an
    effect (a ``maintenance.Restoration``, or a fitted
    [`Condition`][repyability.schedule.Condition]) the plan's one-node
    diagram is simulated with the service at its dated times and without
    it, from one seed (common random numbers), each run priced with the
    plan's prices by period; the value is the mean difference, with its
    interval (``value_lower``, ``value_upper``).

    Parameters
    ----------
    task : Task
        The task, by the calendar or by use (``every=Usage(...)``). A
        calendar task is done at midnight of its date (or of its visit);
        a task by use when the meter reaches its reading.
    does : Replaces, Finds or Services
        What the task does. ``Replaces(part)``: the part a
        ``NonRepairable`` with an ``unplanned_cost``; its
        ``planned_cost`` is the task's price where the task has no
        ``cost``, and given both, they must agree. ``Finds(device,
        demand_rate=, consequence_cost=)``: the device a
        ``NonRepairable``, the task's ``cost`` a test's.
        ``Services(plan, effect=)``: the task's ``cost`` a service's.
    usage : UsageForecast
        The map between dates and the part's operating time
        ([`DutyPlan.forecast`][repyability.schedule.DutyPlan.forecast], or
        ``UsageForecast.calendar`` for a life in calendar time); its
        start is "now", and its span the window valued.
    age : float, optional
        The part's age at the forecast's start, in ``unit``; by default 0.
        A device that is tested is taken to be working then.
    unit : str, optional
        The unit the part's life is in; by default the one its model's
        label names, else hours (with a note).
    discount : float, optional
        A yearly discount rate (0.07 for 7% a year): money is then a
        present value at the forecast's start. By default None: plain
        expected costs.
    escalation : float, optional
        With ``discount``, the effective yearly rate every cost
        escalates at from its price today (0.03 for 3% a year): the
        present values (and ``eac``) take each cost grown to when it
        falls, at the net yearly rate ``(1 + discount) / (1 + escalation)
        - 1``. By default None: today's prices.
    period : str, optional
        ``"day"``, ``"week"``, ``"month"`` (the default), ``"quarter"`` or
        ``"year"``.
    npv_tolerance : float, optional
        How close each side's failure cost is discounted, as for
        ``failures_by_period``; by default 1e-3.
    mc_samples : int, optional
        For a service with an effect: the runs of each side, by default
        4,000.
    seed : int, optional
        For a service with an effect: seeds the runs; by default one is
        drawn, and kept in the result.
    confidence : float, optional
        For a service with an effect: the level of the value's two-sided
        interval, by default 0.9.

    Returns
    -------
    TaskValue

    Raises
    ------
    TypeError
        ``task`` is not a ``Task``, or ``does`` is neither a
        ``Replaces`` nor a ``Finds``.
    ValueError
        The part has no ``unplanned_cost``; the task and the part give
        two prices for the planned replacement, or neither gives one; a
        task that tests has no ``cost``; or as ``Task.due`` and
        ``failures_by_period``.
    NotImplementedError
        With ``Finds``: more than 2,000 tests in the forecast.

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
    if not isinstance(does, (Replaces, Finds, Services)):
        raise TypeError(
            "does must say what the task does: Replaces(part) for a task "
            "that puts a new part in, Finds(device, demand_rate=, "
            "consequence_cost=) for a test of a hidden failure, or "
            "Services(plan, effect=) for a service. Got " + repr(does) + "."
        )
    usage = _usage(usage)
    notes: List[str] = []
    if isinstance(does, Services):
        life = None if does.plan is None else does.plan.life
        if isinstance(does.effect, Condition):
            life = does.effect.life
        label = _model_label(life)
    else:
        part = does.part if isinstance(does, Replaces) else does.device
        label = _model_label(getattr(part, "reliability", None))
    unit = _resolve_unit(unit, label, "life", notes, use=True)
    meter, scale = usage.reading(unit)
    if not (is_number(age) and 0.0 <= float(age) < np.inf):
        raise ValueError(f"age must be a number, at least 0; got {age!r}.")
    age = float(age)
    discount = _discount.yearly(discount)
    given, discount = discount, _discount.net(discount, escalation)
    npv_tolerance = _tolerance(npv_tolerance)
    if isinstance(does, Services):
        if task.cost is None:
            raise ValueError(
                f"Task {task.name!r} has no cost: give the task's cost=, "
                "what one service costs (0.0 if it is free)."
            )
        price = float(task.cost)
    elif isinstance(does, Replaces):
        failure_cost = getattr(part, "cu", None)
        if failure_cost is None:
            raise ValueError(
                "The part has no unplanned_cost: without a failure's cost "
                "the task saves nothing to set against its own. Give the "
                "NonRepairable unplanned_cost=."
            )
        price = _price(task, part, notes)
    elif task.cost is None:
        raise ValueError(
            f"Task {task.name!r} has no cost: give the task's cost=, what "
            "one test costs (0.0 if it is free)."
        )
    else:
        price = float(task.cost)

    found, seconds = _period_edges(usage, period)
    n = seconds.size
    ends = np.atleast_1d(usage.model_time(list(found.ends), unit=unit))
    ends = np.maximum.accumulate(np.maximum(ends, 0.0))
    due, dates, done_s = _done_instants(task, usage, notes)
    retire_at = getattr(getattr(does, "plan", None), "retire_at", None)
    if isinstance(does, Services) and retire_at is not None:
        # The unit is retired on its plan (on the diagram's clock, the
        # forecast's calendar time in ``unit``): no service after it.
        from ._usage import UNITS

        kept = done_s / 3600.0 / UNITS[unit] <= retire_at
        if not kept.all():
            notes.append(
                f"The plan retires the unit {retire_at:g} {unit} into the "
                f"forecast: the {int((~kept).sum())} service(s) after it "
                "are not done."
            )
            dates = [d for d, k in zip(dates, kept) if k]
            done_s = done_s[kept]
    starts_s = usage._start_s
    clock_s = usage._start_s + done_s
    base = usage._at_s(meter, starts_s)
    times = (usage._at_s(meter, clock_s) - base) / scale
    times = np.maximum(np.asarray(times, float).reshape(done_s.shape), 0.0)

    def by_period(at_seconds, amounts=None) -> np.ndarray:
        index = np.minimum(
            np.searchsorted(seconds, at_seconds, side="right"), n - 1
        )
        return np.bincount(index, weights=amounts, minlength=n).astype(float)

    eac: Optional[Dict[str, float]] = None
    sim: Optional[Any] = None
    task_cost: np.ndarray = price * by_period(done_s)
    if isinstance(does, Services):
        plan, sim = _servicing(
            does,
            price,
            usage,
            unit,
            age,
            times,
            done_s,
            seconds,
            discount,
            mc_samples,
            seed,
            confidence,
            len(dates),
        )
        precision = 0.0
        if discount is None or discount == 0.0:
            money_task = task_cost
        else:
            factors = _discount.factor(discount, done_s / _YEAR_SECONDS)
            money_task = by_period(done_s, np.atleast_1d(price * factors))
        if sim is None:
            fails_with = np.full(n, np.nan)
            fails_without = np.full(n, np.nan)
            cost_with, cost_without = task_cost, np.zeros(n)
            money_with, money_without = money_task, np.zeros(n)
        else:
            fails_with, fails_without = sim.failures_with, sim.failures_without
            cost_with = sim.with_cost + task_cost
            cost_without = sim.without_cost
            money_with = sim.with_money + money_task
            money_without = sim.without_money
    else:
        if isinstance(does, Replaces):
            plan = _replacing(
                part, price, failure_cost, age, unit, times, len(dates)
            )
        else:
            plan = _testing(
                does,
                price,
                age,
                unit,
                times,
                done_s,
                float(ends[-1]),
                len(dates),
            )
        each = plan.event_cost
        fails_with = np.diff(plan.with_events(ends), prepend=0.0)
        fails_without = np.diff(plan.without_events(ends), prepend=0.0)

        cost_with = each * fails_with + task_cost
        if plan.extra_seconds.size:
            cost_with = cost_with + by_period(
                plan.extra_seconds, plan.extra_amounts
            )
        cost_without = each * fails_without

        precision = plan.error
        if discount is None or discount == 0.0 or ends[-1] <= 0.0:
            money_with, money_without = cost_with, cost_without
            money_task = task_cost
        else:

            def side(expected):
                def costs(model_ends: np.ndarray) -> np.ndarray:
                    return each * np.diff(expected(model_ends), prepend=0.0)

                accrued, _, half = _accrued_npv(
                    usage, seconds, unit, discount, npv_tolerance, costs, notes
                )
                return accrued, half

            def at_instants(at_seconds, amounts) -> np.ndarray:
                factors = _discount.factor(
                    discount, at_seconds / _YEAR_SECONDS
                )
                return by_period(at_seconds, np.atleast_1d(amounts * factors))

            failures_with_pv, half_with = side(plan.with_events)
            failures_without_pv, half_without = side(plan.without_events)
            money_task = at_instants(done_s, price)
            money_with = failures_with_pv + money_task
            if plan.extra_seconds.size:
                money_with = money_with + at_instants(
                    plan.extra_seconds, plan.extra_amounts
                )
            money_without = failures_without_pv
            precision = half_with + half_without + plan.error
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
    notes.append(
        "Operating time comes from the duty plan "
        f"({usage.describe()}); the part's age is at the forecast's start."
    )
    notes.extend(plan.notes)
    if plan.error > 0.0:
        notes.append(
            "The device's failed time is worked out on a grid, to within "
            f"{plan.error:.3g} of the costs in all (in the precision)."
        )
    if discount is not None and isinstance(does, Services):
        notes.append(
            f"{_discount.words(given, usage.start, escalation)} The "
            "service's cost is discounted from when it is done"
            + (
                "; each run's costs from when they fell."
                if sim is not None
                else "."
            )
        )
    elif discount is not None:
        notes.append(
            f"{_discount.words(given, usage.start, escalation)} A failure's "
            "expected "
            "cost is discounted from when it is expected to fall, each "
            f"side to within its error estimate ({precision:.3g} on the "
            "value in "
            "all); the task's cost from when it is done."
        )
    avoided = fails_without - fails_with
    if sim is None and isinstance(does, Services):
        avoided = np.zeros(n)  # no effect: none avoided
    total = {
        "with_task": float(np.sum(money_with)),
        "without_task": float(np.sum(money_without)),
        "value": float(np.sum(value)),
        "task_cost": float(np.sum(money_task)),
        "cost_with": float(np.sum(cost_with)),
        "cost_without": float(np.sum(cost_without)),
        "failures_with": float(np.sum(fails_with)),
        "failures_without": float(np.sum(fails_without)),
        "failures_avoided": float(np.sum(avoided)),
        "count": len(dates),
    }
    hidden: Dict[str, Any] = {}
    if sim is not None:
        total["value_lower"] = sim.total_lower - total["task_cost"]
        total["value_upper"] = sim.total_upper - total["task_cost"]
        hidden = {
            "method": "simulated",
            "value_lower": sim.lower - money_task,
            "value_upper": sim.upper - money_task,
            "confidence": float(confidence),
            "n_simulations": sim.n,
            "seed": sim.seed,
        }
    if plan.found is not None:
        whole = np.array([ends[-1]])
        hidden = {
            "found": by_period(plan.extra_seconds, plan.found),
            "unavailability_with": _in_periods(ends, plan.with_hidden),
            "unavailability_without": _in_periods(ends, plan.without_hidden),
        }
        total["found"] = float(np.sum(plan.found))
        total["unavailability_with"] = float(
            _in_periods(whole, plan.with_hidden)[0]
        )
        total["unavailability_without"] = float(
            _in_periods(whole, plan.without_hidden)[0]
        )
    return TaskValue(
        task=task.name,
        does=does.describe(),
        with_words=plan.with_words,
        without_words=plan.without_words,
        period=period,
        starts=found.starts,
        ends=found.ends,
        unit=unit,
        age=age,
        dates=tuple(dates),
        model_times=times,
        discount=given,
        with_task=np.asarray(money_with, float),
        without_task=np.asarray(money_without, float),
        value=np.asarray(value, float),
        task_cost=np.asarray(money_task, float),
        cost_with=cost_with,
        cost_without=cost_without,
        failures_with=fails_with,
        failures_without=fails_without,
        failures_avoided=avoided,
        total=total,
        precision=float(precision),
        eac=eac,
        notes=tuple(notes),
        **hidden,
    )
