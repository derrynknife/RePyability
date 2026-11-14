"""The map between moments and meter readings that a duty plan gives.

``DutyPlan.forecast`` returns a ``UsageForecast``: for each meter, a
running total against time, a straight line between the moments the
plan changes, flat while the equipment is off, and stepping up by a run's
starts. It answers both ways: the readings at a moment (``at``), the
moment a meter reaches a reading (``when``), the amount in each calendar
period (``by_period``), operating time since the start and back
(``model_time``, ``moment``), the share of the time running (``duty``),
one meter per unit of another (``rate``) and the duty as steps
(``schedule``). ``Usage`` is a limit by use ("every 2,000 operating
hours") and ``UsageForecast.due`` puts it on dates.

A plan with a range of rates (2 to 4 h a day) gives three maps: the middle
one, which every method answers from, and the low and high ones beside
it (``low``, ``high``, ``earliest``, ``latest``). Where the plan is exact
(runs, shifts, rosters, all the time) the three agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from repyability.rbd.results import _ResultMapping

from ._duty import Continuous, DutyPlan, Rate, pieces
from ._meters import (
    CALENDAR_PERIODS,
    UNITS,
    Clock,
    format_moment,
    nonnegative,
    positive,
    to_plain,
)

_HOUR_S = 3600.0
_ROWS = ("middle", "low", "high")


class _Plain(_ResultMapping):
    """A result whose ``to_dict`` writes moments as ISO text."""

    def to_dict(self) -> dict:
        return {name: to_plain(getattr(self, name)) for name in self}


@dataclass(frozen=True)
class Stretch(_Plain):
    """A stretch of a forecast with one source for its operating hours.

    Attributes
    ----------
    start, end : datetime
        When the stretch starts and ends.
    layer : str or None
        ``"specific"`` or ``"background"``; None where no part covers
        (nothing runs).
    description : str
        The part in words ("exact runs", "about 3 h a day").
    exact : bool
        Whether the hours are exact (runs, shifts, a roster, all the
        time, or nothing) rather than a general rate.
    """

    start: datetime
    end: datetime
    layer: Optional[str]
    description: str
    exact: bool


class Readings(Mapping):
    """The meters' readings at a moment: read as a mapping of meter to
    reading (``readings["hours"]``), with the moment, the low and high
    readings of a range of rates, and the stretch it falls in.

    Attributes
    ----------
    moment : datetime
    low, high : dict
        Each meter's reading under the low and the high rates (equal to
        the reading where the plan is exact).
    layer : str or None
        The layer of the stretch the moment falls in.
    """

    def __init__(self, moment, values, low, high, layer) -> None:
        self.moment = moment
        self._values = dict(values)
        self.low = dict(low)
        self.high = dict(high)
        self.layer = layer

    def __getitem__(self, meter):
        return self._values[meter]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self) -> str:
        return f"Readings({self.moment}, {self._values})"

    def to_dict(self) -> dict:
        return to_plain(
            {
                "moment": self.moment,
                "readings": self._values,
                "low": self.low,
                "high": self.high,
                "layer": self.layer,
            }
        )


@dataclass(frozen=True)
class Crossing(_Plain):
    """When a meter reaches a reading: returned by ``UsageForecast.when``.

    Attributes
    ----------
    meter : str
    reading : float
    moment : datetime or None
        When the meter reaches the reading, by the middle map; None when
        it is not reached in the forecast (or was reached before it).
    earliest, latest : datetime or None
        The same under the high and the low rates of a range (equal to
        ``moment`` where the plan is exact; None past the forecast).
    last_stop_before : datetime or None
        The last time the equipment stopped before it reaches the reading
        (the last chance to do a task early while it is off).
    next_stop_after : datetime or None
        The first time it stops at or after then (when a task due mid-run
        can be done). None where the stretch is a general rate, which has
        no stops, or where the stop is not known.
    layer : str or None
        The layer of the stretch it falls in.
    exact : bool
        Whether the moment is exact (in a stretch of runs, shifts, a roster
        or all the time, and the same under low and high rates).
    already_reached : bool
        Whether the meter had already reached the reading at the
        forecast's start.
    """

    meter: str
    reading: float
    moment: Optional[datetime]
    earliest: Optional[datetime]
    latest: Optional[datetime]
    last_stop_before: Optional[datetime]
    next_stop_after: Optional[datetime]
    layer: Optional[str]
    exact: bool
    already_reached: bool


@dataclass(frozen=True)
class UsageDue(_Plain):
    """When a limit by use falls due: returned by ``UsageForecast.due``.

    Attributes
    ----------
    meter : str
    every : float
        The limit, in the meter's unit.
    last_done_reading : float
        The meter's reading when the task was last done.
    readings : tuple of float
        The readings at which it falls due from the window's start to its
        end.
    moments, earliest, latest, last_stops_before, next_stops_after, layers
        Each due reading's ``Crossing`` fields, in order.
    exact : tuple of bool
    overdue : bool
        Whether the first due reading had already been reached at the
        window's start (it is then the first of ``readings``, with the
        moment None).
    count : int
        How many fall due in the window.
    """

    meter: str
    every: float
    last_done_reading: float
    readings: Tuple[float, ...]
    moments: Tuple[Optional[datetime], ...]
    earliest: Tuple[Optional[datetime], ...]
    latest: Tuple[Optional[datetime], ...]
    last_stops_before: Tuple[Optional[datetime], ...]
    next_stops_after: Tuple[Optional[datetime], ...]
    layers: Tuple[Optional[str], ...]
    exact: Tuple[bool, ...]
    overdue: bool
    count: int


@dataclass(frozen=True)
class PeriodUsage(_Plain):
    """Each meter's amount in each calendar period: returned by
    ``UsageForecast.by_period``. A period runs from its start up to (not
    including) its end; the first starts at the forecast's start and the
    last ends at its end.

    Attributes
    ----------
    period : str
    starts, ends : tuple of datetime
    usage : dict
        Each meter's amount in each period (by the middle map).
    low, high : dict
        The same under the low and the high rates.
    at_end : dict
        Each meter's reading at each period's end.
    """

    period: str
    starts: Tuple[datetime, ...]
    ends: Tuple[datetime, ...]
    usage: Dict[str, Tuple[float, ...]]
    low: Dict[str, Tuple[float, ...]]
    high: Dict[str, Tuple[float, ...]]
    at_end: Dict[str, Tuple[float, ...]]


@dataclass(frozen=True)
class DutySchedule(_Plain):
    """The duty as steps over calendar time since the forecast's start:
    from ``times[k]`` (in ``unit``) the equipment runs ``shares[k]`` of
    the time (1 while running, 0 while off, the rate's share on a general
    rate). Returned by ``UsageForecast.schedule``.

    Attributes
    ----------
    unit : str
    times : tuple of float
    shares : tuple of float
    """

    unit: str
    times: Tuple[float, ...]
    shares: Tuple[float, ...]


@dataclass(frozen=True, init=False)
class Usage:
    """A limit by use: every so much of a meter, such as ``Usage(2000,
    "hours")`` (every 2,000 operating hours) or ``Usage(500, "starts")``.

    Examples
    --------
    >>> from repyability.schedule import Usage
    >>> Usage(2000, "hours").describe()
    'every 2,000 hours'
    """

    amount: float
    meter: str

    def __init__(self, amount, meter: str = "hours") -> None:
        if not isinstance(meter, str) or not meter:
            raise ValueError(f"meter must name a meter; got {meter!r}.")
        object.__setattr__(self, "amount", positive(amount, "Usage amount"))
        object.__setattr__(self, "meter", meter)

    def describe(self) -> str:
        return f"every {self.amount:,.6g} {self.meter}"

    def _task_limit(self, key: str, resets: Optional[bool]) -> "_UsageLimit":
        """This limit as a ``Task``'s ``every=`` or ``also=``: it counts
        again from the reading when the task is done unless ``resets`` is
        False (then it keeps to multiples from the last done reading)."""
        return _UsageLimit(self, key, True if resets is None else bool(resets))

    def to_dict(self) -> dict:
        return {"kind": "usage", "amount": self.amount, "meter": self.meter}

    @classmethod
    def from_dict(cls, data: dict) -> "Usage":
        if data.get("kind") != "usage":
            raise ValueError(f"Not a usage limit: kind {data.get('kind')!r}.")
        return cls(data["amount"], data["meter"])


@dataclass(frozen=True)
class _UsageLimit:
    """A ``Usage`` as a task's limit (the hook in ``_task.py``)."""

    usage: Usage
    key: str
    resets: bool

    def bind(self, task, start, end, context) -> "_BoundUsage":
        forecast = context.get("usage")
        if not isinstance(forecast, UsageForecast):
            raise ValueError(
                f"Task {task.name!r} falls due by use "
                f"({self.usage.describe()}): give usage=, the forecast of "
                "its meters (DutyPlan(...).forecast(...), or "
                "UsageForecast.calendar(...) for calendar time)."
            )
        if self.usage.meter not in forecast.meters:
            raise ValueError(
                f"Task {task.name!r} goes by meter {self.usage.meter!r}, "
                f"which the usage forecast does not have (it has "
                f"{', '.join(map(repr, forecast.meters))})."
            )
        reading = task.last_done_reading
        if reading is None and task.last_done is not None:
            reading = forecast.at(task.last_done)[self.usage.meter]
        if reading is None:
            raise ValueError(
                f"Task {task.name!r} falls due by use: give "
                "last_done_reading= (the meter reading when it was last "
                "done)."
            )
        return _BoundUsage(self, forecast, float(reading), start)

    def describe(self) -> str:
        return self.usage.describe()

    def to_dict(self) -> dict:
        return self.usage.to_dict()


class _BoundUsage:
    """A limit by use in one ``Task.due`` call."""

    def __init__(self, limit, forecast, last_reading, start) -> None:
        self.limit, self.forecast = limit, forecast
        self.meter, self.amount = limit.usage.meter, limit.usage.amount
        self.last = last_reading
        self.start = start
        self.target = last_reading + self.amount

    def _due(self, target: float):
        from ._task import _Due

        f = self.forecast
        self.target = target
        c = f._crossing(self.meter, target, 0.0)
        if c.moment is None and not c.already_reached:
            return None
        if c.moment is None:
            # Reached before the forecast: overdue, at its first moment.
            moment = min(
                f._moment(0.0).replace(tzinfo=None),
                datetime.combine(self.start, datetime.min.time())
                - timedelta(seconds=1),
            )
        else:
            moment = c.moment.replace(tzinfo=None)
        stop = c.next_stop_after
        return _Due(
            moment,
            self.limit.key,
            reading=target,
            stop_after=None if stop is None else stop.replace(tzinfo=None),
        )

    def first(self, last_done):
        return self._due(self.last + self.amount)

    def after(self, done, due):
        f = self.forecast
        try:
            reading = f.at(done)[self.meter]
        except ValueError:
            return None  # done after the forecast ends
        if self.limit.resets:
            return self._due(reading + self.amount)
        k = np.floor((reading - self.last) / self.amount + 1e-9) + 1
        return self._due(self.last + max(k, 1) * self.amount)


def _knots(lines, jumps, origin: datetime, span: float):
    """A meter's running total as knots: times (seconds from ``origin``)
    and, for the middle, low and high maps, the totals there, each jump a
    pair of knots at one time (before and after)."""

    def s(t: datetime) -> float:
        return (t - origin).total_seconds()

    times = {0.0, span}
    for a, b, *_ in lines:
        times.update((s(a), s(b)))
    for t, _ in jumps:
        times.add(s(t))
    u = np.array(sorted(times))
    dr = np.zeros((3, len(u)))
    ja = np.zeros((3, len(u)))
    for a, b, *rates in lines:
        i, j = np.searchsorted(u, [s(a), s(b)])
        dr[:, i] += rates
        dr[:, j] -= rates
    for t, w in jumps:
        ja[:, np.searchsorted(u, s(t))] += w
    rate = np.cumsum(dr, axis=1)[:, :-1]
    inc = rate * np.diff(u) / _HOUR_S
    before = np.concatenate(
        [np.zeros((3, 1)), np.cumsum(inc + ja[:, :-1], axis=1)], axis=1
    )
    after = before + ja
    values = np.empty((3, 2 * len(u)))
    values[:, 0::2] = before
    values[:, 1::2] = after
    return np.repeat(u, 2), values


def _value(t_knots, row, t, left: bool = False):
    """The running total at ``t`` (seconds; an array), right-continuous,
    or its left limit."""
    t = np.asarray(t, dtype=float)
    side: Literal["left", "right"] = "left" if left else "right"
    k = np.clip(
        np.searchsorted(t_knots, t, side=side) - 1, 0, len(t_knots) - 2
    )
    t0, t1 = t_knots[k], t_knots[k + 1]
    v0, v1 = row[k], row[k + 1]
    width = t1 - t0
    frac = np.where(width > 0, (t - t0) / np.where(width > 0, width, 1.0), 1.0)
    frac = np.clip(frac, 0.0, 1.0)
    out = v0 + (v1 - v0) * frac
    return out


def _cross(t_knots, row, v: float) -> Optional[float]:
    """The first time (seconds) the running total reaches ``v``; None if
    it never does; -inf if it had at the first knot."""
    k = int(np.searchsorted(row, v, side="left"))
    if k >= len(row):
        return None
    if k == 0:
        return -np.inf
    t0, t1, v0, v1 = t_knots[k - 1], t_knots[k], row[k - 1], row[k]
    if t1 == t0:
        return float(t1)
    return float(t0 + (v - v0) * (t1 - t0) / (v1 - v0))


class UsageForecast:
    """The map between moments and meter readings from a duty plan (see the
    module's notes). Made by ``DutyPlan.forecast``, or
    ``UsageForecast.calendar`` with no plan.

    Attributes
    ----------
    start, end, read_at : datetime
        The forecast's span and when its readings were taken, as the
        site's wall-clock times (with the zone, when the plan has one).
    time_zone : str or None
    meters : tuple of str
    readings : dict
        The readings it is anchored on.
    fill : str
        How a background rate filled periods specific parts covered in
        part (``DutyPlan``'s ``fill``).
    stretches : tuple of Stretch
        Which layer each stretch of the forecast came from.
    notes : tuple of str
        What the forecast assumed: meters not read, times that don't
        exist or happen twice, stretches with no plan.
    plan : DutyPlan
    """

    def __init__(
        self,
        plan,
        clock_,
        start,
        end,
        origin,
        span,
        read_s,
        maps,
        runs,
        stretches,
        readings,
        notes,
    ) -> None:
        self.plan = plan
        self._clock = clock_
        self._origin = origin
        self._span = span
        self._start_s = (start - origin).total_seconds()
        self._end_s = (end - origin).total_seconds()
        self._maps = maps  # meter -> (t_knots, values (3, K))
        self._runs = runs
        self._stretches = stretches
        self.start = clock_.local(start)
        self.end = clock_.local(end)
        self.read_at = clock_.local(origin + timedelta(seconds=read_s))
        self.time_zone = clock_.name
        self.meters = tuple(maps)
        self.readings = dict(readings)
        self.fill = plan.fill
        self.notes = tuple(notes)
        self.stretches = tuple(
            Stretch(
                clock_.local(st.begin),
                clock_.local(st.end),
                st.layer,
                (
                    "not running (no plan)"
                    if st.part is None
                    else st.part.describe()
                ),
                st.part is None or st.part.runs_kind,
            )
            for st in stretches
        )

    @classmethod
    def calendar(
        cls,
        start,
        end,
        readings: Optional[Dict[str, float]] = None,
        read_at=None,
        time_zone: Optional[str] = None,
    ) -> "UsageForecast":
        """The identity map, for when there is no duty plan: calendar time
        is operating time (as for a life fitted in calendar days), so
        ``"hours"`` counts every real hour.

        Examples
        --------
        >>> from repyability.schedule import UsageForecast
        >>> usage = UsageForecast.calendar("2027-01-01", "2027-02-01")
        >>> usage.at("2027-01-02")["hours"]
        24.0
        """
        return DutyPlan(Continuous(), time_zone=time_zone).forecast(
            start, end, readings=readings, read_at=read_at
        )

    def __repr__(self) -> str:
        return (
            f"UsageForecast({self.start} to {self.end}, meters "
            f"{list(self.meters)}: {self.describe()})"
        )

    # -- helpers -------------------------------------------------------------

    def _seconds(self, moment, name: str = "moment") -> float:
        t = (self._clock.instant(moment, name) - self._origin).total_seconds()
        if not -1e-6 <= t <= self._span + 1e-6:
            raise ValueError(
                f"{name} {moment} is outside the forecast (from "
                f"{self.start} to {self.end}): make the forecast longer."
            )
        return min(max(t, 0.0), self._span)

    def _moment(self, t: Optional[float]) -> Optional[datetime]:
        if t is None or not np.isfinite(t):
            return None
        return self._clock.local(self._origin + timedelta(seconds=float(t)))

    def _map(self, meter: str):
        if meter not in self._maps:
            raise ValueError(
                f"The plan has no meter {meter!r}; it has "
                f"{', '.join(map(repr, self.meters))}."
            )
        return self._maps[meter]

    def _stretch_at(self, t: float):
        for st in self._stretches:
            b = (st.begin - self._origin).total_seconds()
            e = (st.end - self._origin).total_seconds()
            if b <= t < e:
                return st
        return self._stretches[-1] if self._stretches else None

    def _at_s(self, meter: str, t, row: int = 0, left: bool = False):
        knots, values = self._map(meter)
        return _value(knots, values[row], t, left)

    # -- readings and crossings ----------------------------------------------

    def at(self, moment) -> Readings:
        """Each meter's reading at ``moment`` (a run starting then counts
        its start).

        Examples
        --------
        >>> from repyability.schedule import DutyPlan, Shifts
        >>> pump = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
        >>> usage = pump.forecast(
        ...     "2026-11-16", "2027-12-01", readings={"hours": 12_340}
        ... )
        >>> usage.at("2027-01-01")["hours"]
        12884.0
        """
        t = self._seconds(moment)
        rows = [
            {m: float(self._at_s(m, t, r)) for m in self.meters}
            for r in range(3)
        ]
        st = self._stretch_at(t)
        return Readings(
            self._moment(t),
            rows[0],
            rows[1],
            rows[2],
            st.layer if st else None,
        )

    def _crossing(self, meter: str, reading: float, after: float) -> Crossing:
        knots, values = self._map(meter)
        times = [_cross(knots, values[r], reading) for r in range(3)]
        start_value = float(_value(knots, values[0], after))
        already = start_value >= reading
        t = times[0]
        moment = None if already or t is None else t
        st = self._stretch_at(moment) if moment is not None else None
        before = after_stop = None
        if moment is not None and st is not None and st.part is not None:
            if st.part.runs_kind:
                before, after_stop = self._stops(moment)
        exact = (
            moment is not None
            and (st is None or st.part is None or st.part.runs_kind)
            and times[1] == times[2]
        )

        def bound(x):
            return None if already or x is None else self._moment(x)

        return Crossing(
            meter=meter,
            reading=float(reading),
            moment=self._moment(moment),
            earliest=bound(times[2]),
            latest=bound(times[1]),
            last_stop_before=self._moment(before),
            next_stop_after=self._moment(after_stop),
            layer=None if st is None else st.layer,
            exact=bool(exact),
            already_reached=bool(already),
        )

    def _stops(self, t: float):
        """The last stop before ``t`` and the next stop at or after it, for
        a moment in a run (seconds)."""
        ons = self._runs[0]
        i = int(np.searchsorted(ons, t, side="right")) - 1
        if i < 0 or t > self._runs[1][i]:
            return None, None
        after = self._runs[2][i]
        before = None
        for j in range(i - 1, -1, -1):
            stop = self._runs[2][j]
            if stop is not None and stop <= ons[i]:
                before = stop
                break
        return before, after

    def when(self, **reading) -> Crossing:
        """When a meter reaches a reading, such as ``when(hours=14_000)``,
        searching from the forecast's start.

        A moment in a run comes with the stop before and after it, since a
        task due mid-run is done while the equipment is off.

        Examples
        --------
        >>> from repyability.schedule import DutyPlan, Runs
        >>> run = Runs([("2026-11-14 12:31", "2026-11-15 03:27")])
        >>> usage = DutyPlan(run).forecast(
        ...     "2026-11-14", "2026-11-16", readings={"hours": 486.2}
        ... )
        >>> crossing = usage.when(hours=500)
        >>> str(crossing.moment), str(crossing.next_stop_after)
        ('2026-11-15 02:19:00', '2026-11-15 03:27:00')
        """
        if len(reading) != 1:
            raise TypeError(
                "when takes one meter and its reading, such as "
                f"when(hours=14_000); got {sorted(reading)}."
            )
        ((meter, value),) = reading.items()
        value = nonnegative(value, f"The reading of meter {meter!r}")
        return self._crossing(meter, value, self._start_s)

    def due(
        self, every: Usage, last_done_reading: float, start=None, end=None
    ) -> UsageDue:
        """When a limit by use falls due from ``start`` to ``end`` (the
        forecast's by default): at the readings ``last_done_reading +
        every``, ``+ 2 * every``, ... (each due counted from the one before,
        as when every task is done when it falls due).

        Examples
        --------
        >>> from repyability.schedule import DutyPlan, Shifts, Usage
        >>> pump = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
        >>> usage = pump.forecast(
        ...     "2026-11-16", "2027-12-01", readings={"hours": 12_340}
        ... )
        >>> [str(m) for m in usage.due(Usage(2000, "hours"), 12_000).moments]
        ['2027-04-08 18:00:00', '2027-09-30 18:00:00']
        """
        if not isinstance(every, Usage):
            raise TypeError(
                f"every must be a Usage, such as Usage(2000, 'hours'); got "
                f"{every!r}."
            )
        last = nonnegative(last_done_reading, "last_done_reading")
        a = self._start_s if start is None else self._seconds(start, "start")
        b = self._end_s if end is None else self._seconds(end, "end")
        knots, values = self._map(every.meter)
        top = float(values[0][-1])
        crossings: List[Crossing] = []
        target = last + every.amount
        overdue = False
        while target <= top + 1e-9 or not crossings:
            c = self._crossing(every.meter, target, a)
            if c.already_reached:
                if not crossings and not overdue:
                    overdue = True
                    crossings.append(c)
                target += every.amount
                continue
            if c.moment is None:
                break
            t = (
                self._clock.instant(c.moment, "m") - self._origin
            ).total_seconds()
            if t > b + 1e-6:
                break
            crossings.append(c)
            target += every.amount
        return UsageDue(
            meter=every.meter,
            every=every.amount,
            last_done_reading=last,
            readings=tuple(c.reading for c in crossings),
            moments=tuple(c.moment for c in crossings),
            earliest=tuple(c.earliest for c in crossings),
            latest=tuple(c.latest for c in crossings),
            last_stops_before=tuple(c.last_stop_before for c in crossings),
            next_stops_after=tuple(c.next_stop_after for c in crossings),
            layers=tuple(c.layer for c in crossings),
            exact=tuple(c.exact for c in crossings),
            overdue=overdue,
            count=len(crossings),
        )

    # -- periods, model time, duty -------------------------------------------

    def by_period(self, period: str = "month") -> PeriodUsage:
        """Each meter's amount in each calendar period at the site:
        ``"day"``, ``"week"`` (Monday to Sunday), ``"month"``,
        ``"quarter"`` or ``"year"``. The amounts of a span's periods add up
        to the readings' change over it.

        Examples
        --------
        >>> from repyability.schedule import DutyPlan, Shifts
        >>> pump = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
        >>> usage = pump.forecast(
        ...     "2026-11-16", "2027-12-01", readings={"hours": 12_340}
        ... )
        >>> usage.by_period("month").usage["hours"][:3]
        (176.0, 368.0, 336.0)
        """
        if period not in CALENDAR_PERIODS:
            raise ValueError(
                "period must be one of "
                f"{', '.join(map(repr, CALENDAR_PERIODS))}; got {period!r}."
            )
        start = self._origin + timedelta(seconds=self._start_s)
        end = self._origin + timedelta(seconds=self._end_s)
        bounds = self._clock.period_bounds(start, end, period)
        bounds = [start] + [t for t in bounds if start < t < end] + [end]
        secs = np.array([(t - self._origin).total_seconds() for t in bounds])
        usage, low, high, at_end = {}, {}, {}, {}
        for m in self.meters:
            rows = [self._at_s(m, secs, r, left=True) for r in range(3)]
            usage[m] = tuple(float(x) for x in np.diff(rows[0]))
            low[m] = tuple(float(x) for x in np.diff(rows[1]))
            high[m] = tuple(float(x) for x in np.diff(rows[2]))
            at_end[m] = tuple(float(x) for x in rows[0][1:])
        return PeriodUsage(
            period=period,
            starts=tuple(self._clock.local(t) for t in bounds[:-1]),
            ends=tuple(self._clock.local(t) for t in bounds[1:]),
            usage=usage,
            low=low,
            high=high,
            at_end=at_end,
        )

    def _unit(self, unit: str) -> float:
        if unit not in UNITS:
            raise ValueError(
                f"unit must be one of {', '.join(map(repr, UNITS))}; got "
                f"{unit!r}."
            )
        return UNITS[unit]

    def model_time(self, moments, unit: str = "hours"):
        """Operating time since the forecast's start at each moment, in
        ``unit`` (the unit a model's life was fitted in): how a model in
        operating time is reached from dates. A moment or a sequence (an
        array back).

        Examples
        --------
        >>> from repyability.schedule import DutyPlan, Shifts
        >>> pump = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
        >>> usage = pump.forecast("2026-11-16", "2027-12-01")
        >>> usage.model_time("2026-11-23")
        80.0
        """
        scale = self._unit(unit)
        single = isinstance(moments, (str, datetime)) or not hasattr(
            moments, "__iter__"
        )
        given = [moments] if single else list(moments)
        t = np.array([self._seconds(m) for m in given])
        values = (
            self._at_s("hours", t) - self._at_s("hours", self._start_s)
        ) / scale
        return float(values[0]) if single else values

    def moment(
        self, model_time: float, unit: str = "hours"
    ) -> Optional[datetime]:
        """The first moment operating time since the start reaches
        ``model_time`` (in ``unit``); None past the forecast. The inverse
        of ``model_time`` wherever the equipment is running."""
        mt = nonnegative(model_time, "model_time")
        knots, values = self._map("hours")
        base = float(_value(knots, values[0], self._start_s))
        if mt == 0.0:
            return self._moment(self._start_s)
        t = _cross(knots, values[0], base + mt * self._unit(unit))
        return self._moment(t)

    def duty(self, start=None, end=None) -> float:
        """The share of the time running from ``start`` to ``end`` (the
        forecast's span by default): what a ``RepairableRBD`` block's
        ``"duty"`` takes."""
        a = self._start_s if start is None else self._seconds(start, "start")
        b = self._end_s if end is None else self._seconds(end, "end")
        if b <= a:
            raise ValueError("end must be after start.")
        hours = self._at_s("hours", b, left=True) - self._at_s(
            "hours", a, left=True
        )
        return float(hours / ((b - a) / _HOUR_S))

    def rate(
        self, meter: str, per: str = "hours", start=None, end=None
    ) -> float:
        """One meter per unit of another over a span, such as
        ``rate("starts", per="hours")``: starts per operating hour.

        Examples
        --------
        >>> from repyability.schedule import DutyPlan, Shifts
        >>> plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
        >>> usage = plan.forecast("2026-11-16", "2026-11-23")
        >>> usage.rate("starts", per="hours")
        0.0625
        """
        a = self._start_s if start is None else self._seconds(start, "start")
        b = self._end_s if end is None else self._seconds(end, "end")

        def change(m):
            return float(
                self._at_s(m, b, left=True) - self._at_s(m, a, left=True)
            )

        denominator = change(per)
        if denominator <= 0.0:
            raise ValueError(
                f"Meter {per!r} does not move from {self._moment(a)} to "
                f"{self._moment(b)}: no rate per it."
            )
        return change(meter) / denominator

    def life_at_usage(
        self,
        model,
        meter: str = "starts",
        per: str = "hours",
        start=None,
        end=None,
    ):
        """A life on two clocks taken at this plan's rate: ``model``'s
        ``at_usage`` at ``rate(meter, per=per)`` over the span (the
        forecast's by default), such as starts per operating hour.

        ``model`` is a life fitted on two clocks (surpyval 0.28's
        ``EquivalenceModel``, fitted to hours and starts); the life back
        is on the first clock (``per``), ready to be a block's or a
        ``NonRepairable``'s life.

        Raises
        ------
        TypeError
            If ``model`` has no ``at_usage`` (a life on two clocks needs
            surpyval 0.28 or later).

        Examples
        --------
        >>> from repyability.schedule import DutyPlan, Shifts
        >>> class TwoClocks:  # stands in for surpyval's EquivalenceModel
        ...     def at_usage(self, rate):
        ...         return f"the life at {rate} starts an hour"
        >>> plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
        >>> usage = plan.forecast("2026-11-16", "2026-11-23")
        >>> usage.life_at_usage(TwoClocks())
        'the life at 0.0625 starts an hour'
        """
        at_usage = getattr(model, "at_usage", None)
        if not callable(at_usage):
            raise TypeError(
                "life_at_usage takes a life fitted on two clocks (surpyval's "
                "EquivalenceModel, surpyval 0.28 or later, with at_usage); "
                f"got {model!r}."
            )
        return at_usage(self.rate(meter, per=per, start=start, end=end))

    def schedule(self, unit: str = "hours") -> DutySchedule:
        """The duty as steps over calendar time since the forecast's start,
        in ``unit``: the shape a duty that changes over time takes."""
        scale = self._unit(unit)
        knots, values = self._map("hours")
        row = values[0]
        times: List[float] = []
        shares: List[float] = []
        for k in range(len(knots) - 1):
            t0, t1 = knots[k], knots[k + 1]
            if t1 <= t0 or t1 <= self._start_s or t0 >= self._end_s:
                continue
            share = float((row[k + 1] - row[k]) / ((t1 - t0) / _HOUR_S))
            share = round(share, 12)
            at = (max(t0, self._start_s) - self._start_s) / _HOUR_S / scale
            if shares and shares[-1] == share:
                continue
            times.append(float(at))
            shares.append(share)
        return DutySchedule(unit, tuple(times), tuple(shares))

    # -- words and saving ----------------------------------------------------

    def describe(self) -> str:
        """Where the hours come from, stretch by stretch, in plain words:
        ``"exact runs to 16 Nov 03:30, then about 3 h a day"``."""
        start = self._origin + timedelta(seconds=self._start_s)
        end = self._origin + timedelta(seconds=self._end_s)
        phrases: List[Tuple[str, datetime]] = []
        for st in self._stretches:
            if st.end <= start or st.begin >= end:
                continue
            words = (
                "not running (no plan)"
                if st.part is None
                else st.part.describe()
            )
            if phrases and phrases[-1][0] == words:
                phrases[-1] = (words, st.end)
            else:
                phrases.append((words, st.end))
        year = self._clock.local(start).year
        text = []
        for i, (words, until) in enumerate(phrases):
            if i == len(phrases) - 1:
                text.append(words)
            else:
                local = self._clock.local(until)
                text.append(
                    f"{words} to {format_moment(local, local.year != year)}"
                )
        out = ", then ".join(text)
        if self.fill == "top_up" and any(
            isinstance(st.part, Rate) and st.layer == "background"
            for st in self._stretches
        ):
            out += (
                " (topped up to each period's amount where the specific "
                "parts end)"
            )
        return out

    def to_dict(self) -> dict:
        """The forecast as plain data: its span, readings, the readings at
        its start and end, its stretches, its notes and its plan."""
        start = self.at(self.start)
        end = self.at(self.end)
        return to_plain(
            {
                "start": self.start,
                "end": self.end,
                "read_at": self.read_at,
                "time_zone": self.time_zone,
                "fill": self.fill,
                "readings": self.readings,
                "meters": list(self.meters),
                "at_start": dict(start),
                "at_end": dict(end),
                "at_end_low": end.low,
                "at_end_high": end.high,
                "stretches": [s.to_dict() for s in self.stretches],
                "description": self.describe(),
                "notes": list(self.notes),
                "plan": self.plan.to_dict(),
            }
        )


def _build(plan: DutyPlan, start, end, readings, read_at) -> UsageForecast:
    """``plan.forecast``'s work (see ``DutyPlan.forecast``)."""
    clock_ = Clock(plan.time_zone)
    s = clock_.instant(start, "start")
    e = clock_.instant(end, "end")
    if not e > s:
        raise ValueError(f"end ({end}) must be after start ({start}).")
    r = s if read_at is None else clock_.instant(read_at, "read_at")
    origin, last = min(s, r), max(e, r)
    meters = plan.meters()
    given: Dict[str, float] = {}
    if readings is not None:
        if not isinstance(readings, dict):
            raise TypeError(
                "readings must be a dict of meter readings, such as "
                f"{{'hours': 12340}}; got {readings!r}."
            )
        for meter, value in readings.items():
            if meter not in meters:
                raise ValueError(
                    f"The plan has no meter {meter!r}; it has "
                    f"{', '.join(map(repr, meters))}."
                )
            given[meter] = nonnegative(
                value, f"The reading of meter {meter!r}"
            )
    got = pieces(plan, clock_, origin, last)
    span = (last - origin).total_seconds()
    read_s = (r - origin).total_seconds()
    start_s = (s - origin).total_seconds()
    notes = list(clock_.notes)
    maps = {}
    for meter in meters:
        knots, values = _knots(
            got.lines[meter], got.jumps[meter], origin, span
        )
        if meter in given:
            anchor = _value(knots, values[0], read_s)
            values = values + (given[meter] - anchor)
        else:
            anchor = _value(knots, values[0], start_s)
            values = values - anchor
            notes.append(
                f"Meter {meter!r} was not read: counted from 0 at the start."
            )
        maps[meter] = (knots, values)
    if any(st.part is None for st in got.stretches) and "hours" in meters:
        notes.append(
            "Some of the forecast has no part of the plan: nothing runs then."
        )
    runs = got.runs
    run_arrays = (
        np.array([(a - origin).total_seconds() for a, _, _ in runs]),
        np.array([(b - origin).total_seconds() for _, b, _ in runs]),
        [
            None if stop is None else (stop - origin).total_seconds()
            for *_, stop in runs
        ],
    )
    return UsageForecast(
        plan,
        clock_,
        s,
        e,
        origin,
        span,
        read_s,
        maps,
        run_arrays,
        got.stretches,
        given,
        notes,
    )


def _register() -> None:
    from ._task import LIMIT_KINDS

    LIMIT_KINDS["usage"] = Usage


_register()
