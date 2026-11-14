"""Failures, events and shuts on the calendar, through the map between
dates and model time.

RePyability's models answer in model time; this module asks them from
dates and labels their answers with dates. It adds no probability: each
function turns calendar dates into model time (operating time through a
``UsageForecast``, or calendar time since a start), calls an existing
method, and puts the dates back.

- ``failures_by_period``: a ``NonRepairable`` part's expected failures,
  cost and their range in each calendar period
  (``NonRepairable.failures``, ``cost_by_window``,
  ``cost_distribution``).
- ``events_by_period``: a ``RepairableRBD``'s expected events (and,
  on request, cost) in each calendar period
  (``RepairableRBD.expected_events_by_window``, ``cost_by_window``).
- ``shutdown_spec``: shuts on a ``Recurrence`` as a ``RepairableRBD``
  shutdown at given times.
- ``Opportunity.from_recurrence`` (``opportunity_times`` here): dated
  shuts laid on a part's model time for a ``Calendar``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from repyability.utils.checks import is_number

from ._meters import UNITS, Clock, clock, time_unit
from ._recurrence import Recurrence, _period
from ._usage import UsageForecast, _Plain

#: The range every forecast here gives: the 5th and 95th percentiles, a
#: 90% two-sided range.
RANGE_LEVELS = (5.0, 95.0)
RANGE_WORDS = "the 5th to 95th percentile (90% of outcomes, two-sided)"


# ------------------------------------------------------------ helpers


def _model_label(model: Any) -> Optional[str]:
    """The unit label a model carries (surpyval's ``model.unit``, or a
    diagram's ``unit``), or None."""
    label = getattr(model, "unit", None)
    return label if isinstance(label, str) else None


def _resolve_unit(
    unit: Optional[str], label: Optional[str], what: str, notes: List[str]
) -> str:
    """The model-time unit to use: ``unit`` if given, else the one the
    model's label names, else hours (with a note)."""
    if unit is not None:
        if unit not in UNITS:
            raise ValueError(
                f"unit must be one of {', '.join(map(repr, UNITS))}; got "
                f"{unit!r}."
            )
        named = time_unit(label)
        if named is not None and named != unit:
            raise ValueError(
                f"unit={unit!r}, but the {what} is labelled {label!r}: give "
                "the unit the life was fitted in."
            )
        return unit
    if label is None:
        notes.append(
            f"The {what} names no unit: its times are taken as hours "
            "(give unit= otherwise)."
        )
        return "hours"
    named = time_unit(label)
    if named is None:
        raise ValueError(
            f"The {what} is labelled {label!r}, which is not a unit of "
            f"time: give unit=, one of {', '.join(map(repr, UNITS))}, for "
            "how its times are read from dates."
        )
    return named


def _usage(usage) -> UsageForecast:
    if not isinstance(usage, UsageForecast):
        raise TypeError(
            "usage must be a UsageForecast (DutyPlan(...).forecast(...), "
            "or UsageForecast.calendar(start, end) for calendar time); got "
            f"{usage!r}."
        )
    return usage


def _period_edges(usage: UsageForecast, period: str):
    """The forecast's periods: their starts and ends (the site's
    wall-clock times) and the real seconds from the forecast's start to
    each end."""
    found = usage.by_period(period)
    origin = usage._clock.instant(usage.start, "start")
    seconds = np.array(
        [
            (usage._clock.instant(t, "end") - origin).total_seconds()
            for t in found.ends
        ]
    )
    return found, seconds


def _instants(days, clock_: Clock, at) -> List[datetime]:
    """Each date at the clock time ``at`` (midnight by default), as a real
    instant at the site."""
    when = clock("00:00" if at is None else at, "at")
    return [clock_.wall(datetime.combine(day, when)) for day in days]


# ------------------------------------------------------- a part by period


@dataclass(frozen=True)
class FailuresByPeriod(_Plain):
    """A part's failures and their cost in each calendar period: returned
    by ``failures_by_period``.

    Each period's numbers are its own (not a running total) and add up
    over the periods to the whole forecast's (the ranges do not add:
    ``total`` has the whole forecast's own).

    Attributes
    ----------
    period : str
        ``"day"``, ``"week"``, ``"month"``, ``"quarter"`` or ``"year"``.
    starts, ends : tuple of datetime
        Each period's start and end at the site (the first starts at the
        forecast's start, the last ends at its end).
    unit : str
        The model's unit of time.
    operating : numpy.ndarray
        The operating time in each period, in ``unit``.
    model_ends : numpy.ndarray
        The operating time from the forecast's start to each period's
        end: the model-time windows the part was asked about.
    age : float
        The part's age at the forecast's start, in ``unit``.
    policy : str
        ``"run to failure"``, ``"age replacement"``, ``"block
        replacement"`` or ``"calendar"``.
    method : str
        ``"exact"`` (``NonRepairable.failures``) or ``"numerical"``
        (``cost_by_window``, to its ``precision``).
    expected_failures : numpy.ndarray
        The expected failures in each period.
    probability : numpy.ndarray
        The chance of at least one failure from the forecast's start to
        each period's end.
    probability_in_period : numpy.ndarray or None
        The chance of at least one failure in each period.
    expected_planned : numpy.ndarray or None
        The expected planned replacements in each period (None when run
        to failure).
    expected_cost : numpy.ndarray or None
        The expected cost in each period; None for a part with no cost.
    cost_low, cost_high : numpy.ndarray or None
        Each period's cost range: its 5th and 95th percentiles (90% of
        outcomes, two-sided).
    failures_low, failures_high : numpy.ndarray or None
        The same for the number of failures.
    total : dict
        The whole forecast's: ``expected_failures``, ``probability`` (of
        at least one failure), ``expected_cost``, ``cost_low``,
        ``cost_high``, ``failures_low``, ``failures_high`` and
        ``failure_probabilities`` (of 0, 1, 2, ... failures).
    range : str
        What the ranges are, in words.
    notes : tuple of str
        What the forecast assumed.
    """

    period: str
    starts: Tuple[datetime, ...]
    ends: Tuple[datetime, ...]
    unit: str
    operating: np.ndarray
    model_ends: np.ndarray
    age: float
    policy: str
    method: str
    expected_failures: np.ndarray
    probability: np.ndarray
    probability_in_period: Optional[np.ndarray]
    expected_planned: Optional[np.ndarray]
    expected_cost: Optional[np.ndarray]
    cost_low: Optional[np.ndarray]
    cost_high: Optional[np.ndarray]
    failures_low: Optional[np.ndarray]
    failures_high: Optional[np.ndarray]
    total: Dict[str, Any]
    range: str
    notes: Tuple[str, ...]

    def __repr__(self) -> str:
        cost = (
            ""
            if self.total.get("expected_cost") is None
            else f", expected cost {self.total['expected_cost']:.4g}"
        )
        return (
            f"FailuresByPeriod({len(self.starts)} {self.period}s from "
            f"{self.starts[0]:%Y-%m-%d}: expected failures "
            f"{self.total['expected_failures']:.4g}, chance of at least "
            f"one {self.total['probability']:.3g}{cost})"
        )


def _percentile(values, probabilities, q: float) -> float:
    """The least value not exceeded with a chance of at least ``q``%."""
    cumulative = np.cumsum(probabilities)
    k = int(np.searchsorted(cumulative, q / 100.0 - 1e-12))
    return float(values[min(k, len(values) - 1)])


def failures_by_period(
    part,
    usage,
    period: str = "month",
    *,
    age: float = 0.0,
    unit: Optional[str] = None,
    every: Optional[float] = None,
    resets: Optional[bool] = None,
    since_renewal: Optional[float] = None,
    calendar=None,
) -> FailuresByPeriod:
    """A part's expected failures, their cost and its range in each
    calendar period, through the map between dates and operating time.

    Each period's end is turned into operating time since the forecast's
    start (``usage.model_time``), the part is asked about those windows
    with its existing methods, and the answers come back labelled with the
    periods' dates. Each period's expected failures, cost and range, under
    any policy (``every``, ``calendar``), are
    [`NonRepairable.cost_by_window`][repyability.NonRepairable.cost_by_window]'s
    (numerical); run to failure, the chance of at least one failure by
    each period's end is
    [`NonRepairable.failures`][repyability.NonRepairable.failures]'s
    (exact). A part ``cost_by_window`` does not take (no
    ``unplanned_cost``, or a replacement that takes time) is run to
    failure with ``failures``' exact expected failures and no ranges, the
    reason in the notes. Ranges are the 5th to 95th percentile: 90% of
    outcomes, two-sided.

    Parameters
    ----------
    part : NonRepairable
        The part, its life in operating time (hours by default).
    usage : UsageForecast
        The map between dates and operating hours
        ([`DutyPlan.forecast`][repyability.schedule.DutyPlan.forecast]); its
        start is the part's "now". ``UsageForecast.calendar`` for a life in
        calendar time.
    period : str, optional
        ``"day"``, ``"week"``, ``"month"`` (the default), ``"quarter"`` or
        ``"year"``, at the site.
    age : float, optional
        The part's age at the forecast's start, in ``unit``; by default 0.
    unit : str, optional
        The unit the part's life is in (``"hours"``, ``"days"``, ...); by
        default the one its model's label names, else hours (with a note).
    every, resets, since_renewal, calendar : optional
        A replacement policy, as
        [`NonRepairable.cost_by_window`][repyability.NonRepairable.cost_by_window]
        takes it, in ``unit``; by default run to failure. A ``calendar``'s
        times count from the forecast's start
        ([`Opportunity.from_recurrence`][repyability.Opportunity.from_recurrence]
        lays dated shuts there).

    Returns
    -------
    FailuresByPeriod

    Examples
    --------
    A pump's wear ring (Weibull, scale 9,000 operating hours, shape 2.5),
    5,200 h old, on a pump running Monday to Friday 06:00 to 22:00, at
    $4,800 a failure:

    >>> import surpyval as surv
    >>> from repyability import NonRepairable
    >>> from repyability.schedule import DutyPlan, Shifts, failures_by_period
    >>> ring = NonRepairable(
    ...     surv.Weibull.from_params([9000.0, 2.5]), unplanned_cost=4800.0
    ... )
    >>> usage = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
    ...     "2026-11-16", "2027-12-01", readings={"hours": 12_340}
    ... )
    >>> months = failures_by_period(ring, usage, "month", age=5200.0)
    >>> round(float(months.expected_failures.sum()), 3)
    0.622
    >>> round(float(months.probability[-1]), 3)
    0.596
    >>> round(months.total["expected_cost"])
    2988
    """
    usage = _usage(usage)
    notes: List[str] = []
    unit = _resolve_unit(
        unit, _model_label(getattr(part, "reliability", None)), "life", notes
    )
    if not (is_number(age) and 0.0 <= float(age) < np.inf):
        raise ValueError(f"age must be a number, at least 0; got {age!r}.")
    age = float(age)
    found = usage.by_period(period)
    ends = np.atleast_1d(usage.model_time(list(found.ends), unit=unit))
    ends = np.maximum.accumulate(np.maximum(ends, 0.0))
    operating = np.diff(ends, prepend=0.0)
    n = len(ends)
    # cost_by_window takes increasing windows: periods with no operating
    # time (a week the plant is shut) have nothing in them.
    moving = operating > 0.0
    total_end = float(ends[-1]) if n else 0.0
    expected = np.zeros(n)
    planned = cost = low = high = f_low = f_high = in_period = None
    total: Dict[str, Any] = {}
    has_cost = getattr(part, "cu", None) is not None
    windows = None
    if total_end > 0.0:
        try:
            windows = part.cost_by_window(
                ends[moving],
                every,
                resets=resets,
                age=age,
                since_renewal=since_renewal,
                calendar=calendar,
                levels=RANGE_LEVELS,
            )
        except (ValueError, NotImplementedError) as error:
            if every is not None or calendar is not None:
                raise
            # Run to failure needs no cost_by_window: the expected values
            # are NonRepairable.failures', with no ranges.
            notes.append(
                f"No cost or ranges by period: {error} The expected "
                "failures are NonRepairable.failures' (exact; a "
                "replacement taking time runs no hours, and the dates "
                "take it as taking no calendar time)."
            )
    if windows is not None:
        policy = windows.policy
        method = "numerical"
        planned, cost = np.zeros(n), np.zeros(n)
        low, high = np.zeros(n), np.zeros(n)
        f_low, f_high = np.zeros(n), np.zeros(n)
        in_period = np.zeros(n)
        for k, w in zip(np.flatnonzero(moving), windows.windows):
            expected[k] = w.expected_failures
            planned[k] = w.expected_planned
            cost[k] = w.mean
            low[k], high[k] = (float(c) for c in w.percentiles)
            counts = np.arange(len(w.failure_probabilities))
            f_low[k] = _percentile(counts, w.failure_probabilities, 5.0)
            f_high[k] = _percentile(counts, w.failure_probabilities, 95.0)
            in_period[k] = 1.0 - float(w.failure_probabilities[0])
        whole = windows.total
        counts = np.arange(len(whole.failure_probabilities))
        total = {
            "expected_failures": float(whole.expected_failures),
            "expected_cost": float(whole.mean),
            "cost_low": float(whole.percentiles[0]),
            "cost_high": float(whole.percentiles[1]),
            "failures_low": _percentile(
                counts, whole.failure_probabilities, 5.0
            ),
            "failures_high": _percentile(
                counts, whole.failure_probabilities, 95.0
            ),
            "failure_probabilities": [
                float(p) for p in whole.failure_probabilities
            ],
        }
        if policy == "run to failure":
            planned = None
    else:
        policy, method = "run to failure", "exact"
    if policy == "run to failure":
        forecast = part.failures(np.r_[ends, total_end], age=age)
        cumulative = np.asarray(forecast.expected_failures, float)
        probability = np.asarray(forecast.probability, float)[:n]
        if method == "exact":
            expected = np.diff(cumulative[:n], prepend=0.0)
            if has_cost:
                cost = expected * part.cu
            total = {
                "expected_failures": float(cumulative[-1]),
                "expected_cost": (
                    None if not has_cost else float(cumulative[-1] * part.cu)
                ),
            }
        total["probability"] = float(probability[-1]) if n else 0.0
    else:
        # The chance of a failure by each period's end, under the policy:
        # cost_distribution over each span from the start.
        probability = np.zeros(n)
        for i in range(n):
            if ends[i] > 0.0:
                spread = part.cost_distribution(
                    float(ends[i]),
                    every,
                    resets=resets,
                    age=age,
                    since_renewal=since_renewal,
                    calendar=calendar,
                )
                probability[i] = 1.0 - float(spread.failure_probabilities[0])
        total["probability"] = float(probability[-1]) if n else 0.0
    notes.append(
        "Operating time comes from the duty plan "
        f"({usage.describe()}); the part's age is at the forecast's start."
    )
    if low is not None:
        notes.append(f"Ranges are {RANGE_WORDS}.")
    return FailuresByPeriod(
        period=period,
        starts=found.starts,
        ends=found.ends,
        unit=unit,
        operating=operating,
        model_ends=ends,
        age=age,
        policy=policy,
        method=method,
        expected_failures=expected,
        probability=probability,
        probability_in_period=in_period,
        expected_planned=planned,
        expected_cost=cost,
        cost_low=low,
        cost_high=high,
        failures_low=f_low,
        failures_high=f_high,
        total=total,
        range=RANGE_WORDS,
        notes=tuple(notes),
    )


# ------------------------------------------------------ a system by period


@dataclass(frozen=True)
class EventsByPeriod(_Plain):
    """A system's expected events (and, on request, cost) in each calendar
    period: returned by ``events_by_period``.

    Attributes
    ----------
    period : str
    starts, ends : tuple of datetime
        Each period's start and end at the site.
    unit : str
        The diagram's unit of time.
    model_ends : numpy.ndarray
        Each period's end on the diagram's clock: real (calendar) time
        since the forecast's start, in ``unit``.
    events : EventsByWindow
        [`RepairableRBD.expected_events_by_window`][repyability.RepairableRBD.expected_events_by_window]'s
        answer over those windows (exact): failures, repairs,
        replacements, tests, outages and downtime, by period.
    cost : CostByWindow or None
        [`RepairableRBD.cost_by_window`][repyability.RepairableRBD.cost_by_window]'s
        answer, with ``cost=True``: each period's exact mean, and its range
        (5th to 95th percentile) simulated.
    range : str or None
        What the cost range is, in words (None with no cost).
    notes : tuple of str
    """

    period: str
    starts: Tuple[datetime, ...]
    ends: Tuple[datetime, ...]
    unit: str
    model_ends: np.ndarray
    events: Any
    cost: Any
    range: Optional[str]
    notes: Tuple[str, ...]

    def __repr__(self) -> str:
        return (
            f"EventsByPeriod({len(self.starts)} {self.period}s from "
            f"{self.starts[0]:%Y-%m-%d}: expected system failures "
            f"{float(np.sum(self.events.system_failures)):.4g})"
        )


def events_by_period(
    rbd,
    usage,
    period: str = "month",
    *,
    state=None,
    unit: Optional[str] = None,
    cost: bool = False,
    mc_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> EventsByPeriod:
    """A ``RepairableRBD``'s expected events, and on request its cost, in
    each calendar period.

    A diagram's clock is calendar time (its blocks' ``"duty"`` and
    ``"basis"`` put running hours on it), so each period's end is the real
    time from the forecast's start to it, in the diagram's unit; the
    diagram is asked about those windows
    ([`expected_events_by_window`][repyability.RepairableRBD.expected_events_by_window],
    exact) and the answers come back labelled with the periods' dates.
    With ``cost=True``,
    [`cost_by_window`][repyability.RepairableRBD.cost_by_window] adds
    each period's cost: its mean exact, its range (the 5th to 95th
    percentile, 90% of outcomes, two-sided) simulated.

    Parameters
    ----------
    rbd : RepairableRBD
    usage : UsageForecast
        Gives the periods, the site's time zone and "now" (its start);
        ``UsageForecast.calendar(start, end)`` when there is no plan.
        ``usage.duty()`` is the share a block's ``"duty"`` takes.
    period : str, optional
        ``"day"``, ``"week"``, ``"month"`` (the default), ``"quarter"`` or
        ``"year"``.
    state : dict or str, optional
        Where the components are at the forecast's start, as for
        [`expected_events`][repyability.RepairableRBD.expected_events].
    unit : str, optional
        The diagram's unit of time; by default the one its models' labels
        name, else hours (with a note).
    cost : bool, optional
        Also the cost by period (simulated range), by default False.
    mc_samples, seed : optional
        With ``cost=True``, the simulations, as for ``cost_by_window``.

    Returns
    -------
    EventsByPeriod

    Examples
    --------
    A pump whose failures come at a constant rate (one per 2,000 hours on
    average), repaired at once, by month in 2027:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> from repyability.schedule import UsageForecast, events_by_period
    >>> rbd = RepairableRBD(
    ...     [("s", "pump"), ("pump", "t")],
    ...     {"pump": {
    ...         "reliability": surv.Exponential.from_params([1 / 2000.0]),
    ...         "repairability": "instant",
    ...     }},
    ... )
    >>> usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    >>> months = events_by_period(rbd, usage, unit="hours")
    >>> [round(float(x), 3) for x in months.events.system_failures[:2]]
    [0.372, 0.336]
    """
    usage = _usage(usage)
    notes: List[str] = []
    unit = _resolve_unit(unit, _model_label(rbd), "diagram", notes)
    found, seconds = _period_edges(usage, period)
    ends = seconds / 3600.0 / UNITS[unit]
    events = rbd.expected_events_by_window(ends, state=state)
    costs = None
    if cost:
        costs = rbd.cost_by_window(
            ends,
            state=state,
            mc_samples=mc_samples,
            seed=seed,
            levels=RANGE_LEVELS,
        )
        notes.append(
            f"Each period's mean cost is exact; its range ({RANGE_WORDS}) "
            "is simulated."
        )
    notes.append(
        "The diagram's clock is calendar time from the forecast's start; "
        "running hours reach it through the blocks' duty."
    )
    return EventsByPeriod(
        period=period,
        starts=found.starts,
        ends=found.ends,
        unit=unit,
        model_ends=ends,
        events=events,
        cost=costs,
        range=None if costs is None else RANGE_WORDS,
        notes=tuple(notes),
    )


# ---------------------------------------------------------- shuts on dates


def _rule(rule) -> Recurrence:
    if not isinstance(rule, Recurrence):
        raise TypeError(
            "rule must be a Recurrence (Recurrence.yearly(...), "
            f"Recurrence.on([...]), ...); got {rule!r}."
        )
    return rule


def calendar_times(
    rule,
    start,
    end,
    *,
    unit: str = "hours",
    at=None,
    time_zone: Optional[str] = None,
) -> np.ndarray:
    """The rule's dates from ``start`` to ``end`` (both included), each at
    the clock time ``at`` (midnight by default), as real time since
    ``start``'s midnight at the site, in ``unit``: a diagram's clock.

    Examples
    --------
    >>> from repyability.schedule import Recurrence
    >>> from repyability.schedule._forecast import calendar_times
    >>> overhaul = Recurrence.yearly(month=11, weekday="Tuesday", nth=1)
    >>> calendar_times(
    ...     overhaul, "2026-11-14", "2029-12-31", unit="days"
    ... ).tolist()
    [353.0, 724.0, 1088.0]
    """
    first, last = _period(start, end)
    if unit not in UNITS:
        raise ValueError(
            f"unit must be one of {', '.join(map(repr, UNITS))}; got "
            f"{unit!r}."
        )
    clock_ = Clock(time_zone)
    origin = clock_.midnight(first)
    moments = _instants(_rule(rule).dates(first, last), clock_, at)
    return np.array(
        [(m - origin).total_seconds() / 3600.0 / UNITS[unit] for m in moments]
    )


def shutdown_spec(
    rule,
    start,
    end,
    *,
    duration: float,
    unit: str = "hours",
    at=None,
    time_zone: Optional[str] = None,
    cost: Optional[float] = None,
    scope=None,
    idle_ageing: Optional[bool] = None,
) -> dict:
    """A ``RepairableRBD`` shutdown from a rule on dates: its ``"times"``
    are the rule's dates from ``start`` to ``end`` at the clock time
    ``at``, on the diagram's clock (real time since ``start``'s midnight,
    in ``unit``). A rule such as "the first Tuesday of November" has no
    fixed period (364 or 371 days apart), so it goes in as given times.

    Parameters
    ----------
    rule : Recurrence
        When the shuts are.
    start, end : datetime.date or str
        The diagram's time 0 is ``start``'s midnight; the shuts from
        ``start`` to ``end``, both included.
    duration : float
        How long each shut lasts, in ``unit``.
    unit : str, optional
        The diagram's unit, by default hours.
    at : str or datetime.time, optional
        The clock time each shut starts, by default midnight.
    time_zone : str, optional
        The site's IANA zone (real hours across daylight saving); by
        default clock times as they are.
    cost, scope, idle_ageing : optional
        The shutdown's ``"cost"``, ``"scope"`` and ``"idle_ageing"``, as
        ``RepairableRBD(shutdowns=)`` takes them; left out when not given.

    Returns
    -------
    dict
        The spec for ``RepairableRBD(shutdowns={name: spec})``.

    Examples
    --------
    The overhaul on the first Tuesday of November, from 06:00 for 48
    hours, on a diagram whose time 0 is 14 November 2026:

    >>> from repyability.schedule import Recurrence, shutdown_spec
    >>> overhaul = Recurrence.yearly(month=11, weekday="Tuesday", nth=1)
    >>> spec = shutdown_spec(
    ...     overhaul, "2026-11-14", "2029-12-31", duration=48.0, at="06:00"
    ... )
    >>> spec["times"]
    [8478.0, 17382.0, 26118.0]
    """
    times = calendar_times(
        rule, start, end, unit=unit, at=at, time_zone=time_zone
    )
    if len(times) == 0:
        raise ValueError(
            f"The rule ({_rule(rule).describe()}) gives no dates from "
            f"{start} to {end}: no shuts to put on the diagram."
        )
    spec: Dict[str, Any] = {
        "times": [float(t) for t in times],
        "duration": duration,
    }
    for key, value in (
        ("cost", cost),
        ("scope", scope),
        ("idle_ageing", idle_ageing),
    ):
        if value is not None:
            spec[key] = value
    return spec


def opportunity_times(
    rule,
    start,
    end,
    *,
    unit: str = "hours",
    usage=None,
    at=None,
    time_zone: Optional[str] = None,
) -> np.ndarray:
    """``Opportunity.from_recurrence``'s times: the rule's dates from
    ``start`` to ``end`` on a part's model time. With ``usage``, operating
    time since the forecast's start (a life in operating hours); without,
    real time since ``start``'s midnight (a life in calendar time). Dates
    at the same operating time (shuts while the plant is idle) are one
    time."""
    if usage is None:
        return calendar_times(
            rule, start, end, unit=unit, at=at, time_zone=time_zone
        )
    usage = _usage(usage)
    if time_zone is not None and time_zone != usage.time_zone:
        raise ValueError(
            f"time_zone={time_zone!r}, but the usage forecast is in "
            f"{usage.time_zone!r}: the dates are read in the forecast's."
        )
    if unit not in UNITS:
        raise ValueError(
            f"unit must be one of {', '.join(map(repr, UNITS))}; got "
            f"{unit!r}."
        )
    first, last = _period(start, end)
    moments = _instants(_rule(rule).dates(first, last), usage._clock, at)
    times = [usage.model_time(m, unit=unit) for m in moments]
    return np.unique(np.array(times, float))
