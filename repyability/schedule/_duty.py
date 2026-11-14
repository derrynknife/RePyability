"""Duty plans: how a piece of equipment runs, and the operating hours (and
other meters) that gives it by any moment.

A ``DutyPlan`` is made of parts:

* ``Runs``: concrete runs, each a start and a stop;
* ``Shifts``: a repeating week of clock times (a stop before the start is
  on the next day);
* ``Roster``: a cycle of days on and off, with a run's clock times;
* ``Rate``: a general rate, spread evenly over each calendar period (3 h a
  day; 15 h a month; 1,200 km a week), or a range of rates;
* ``Continuous``: all the time.

**From specific to general.** A plan has a *background* (one part, or
parts that follow each other from dates, the last with no end) and
*specific* parts on top, each covering its own span. Wherever a specific
part covers, it is used; the moment it runs out, the background is. The
forecast says which stretch came from which ("exact runs to 16 Nov 03:30,
then about 3 h a day").

**What is refused.** Two parts of one layer that would count the same
time on the same meter twice (overlapping spans) are refused, naming
both; a background part with no end ends where the next one starts. A
specific part over the background is the layering, not an overlap.

**Meters.** ``"hours"`` counts real running hours. Every run adds one to
``"starts"`` (a run an outage splits is two runs); ``per_run`` and
``per_hour`` add other meters (cycles, tonnes); a ``Rate`` may drive any
meter of its own (``meter="km"``).

Nothing here reads the clock or stores anything: ``DutyPlan.forecast``
takes the start, the end and the readings, and returns a
``UsageForecast`` (``_usage.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

from repyability.utils.checks import is_number

from ._meters import (
    CALENDAR_PERIODS,
    MONTH_NAMES,
    PERIODS,
    Clock,
    clock,
    describe_days,
    format_moment,
    from_iso,
    nonnegative,
    positive,
    wall_time,
    weekdays,
)

#: The version of the saved form every ``to_dict`` writes.
VERSION = 1

_DAY = timedelta(days=1)


# --- options a part and a plan share ---------------------------------------


def _season(value) -> Optional[Tuple[int, ...]]:
    """A season as month numbers: ``None`` (all year), a sequence of
    month numbers (1 to 12), or text naming months and ranges of them
    (``"Apr-Oct"``, ``"Nov-Feb"``, ``"Jan, Jul"``)."""
    if value is None:
        return None
    if isinstance(value, str):
        names = {m.lower(): k + 1 for k, m in enumerate(MONTH_NAMES)}
        found: set = set()
        for part in value.split(","):
            ends = [e.strip().lower()[:3] for e in part.split("-")]
            if len(ends) > 2 or not all(e in names for e in ends):
                raise ValueError(
                    "season must name months, or ranges of them, separated "
                    f"by commas (such as 'Apr-Oct'); got {value!r}."
                )
            first, last = names[ends[0]] - 1, names[ends[-1]] - 1
            found.update(
                (first + k) % 12 + 1 for k in range((last - first) % 12 + 1)
            )
        return tuple(sorted(found))
    months: Tuple[int, ...]
    try:
        months = tuple(sorted(set(value)))
    except TypeError:
        months = ()
    if not months or not all(
        isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= 12
        for m in months
    ):
        raise ValueError(
            "season must be month numbers (1 to 12), or text such as "
            f"'Apr-Oct'; got {value!r}."
        )
    return months


def _utilisation(value) -> Optional[float]:
    if value is None:
        return None
    if not (is_number(value) and 0.0 < float(value) <= 1.0):
        raise ValueError(
            "utilisation must be a share more than 0 and at most 1 (0.9: "
            f"90% of the planned running happens); got {value!r}."
        )
    return float(value)


def _optional_moment(value, name: str) -> Optional[datetime]:
    return None if value is None else wall_time(value, name)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat()


def _meter_amounts(value, name: str) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f"{name} must be a dict of meter names and amounts, such as "
            f"{{'tonnes': 40}}; got {value!r}."
        )
    out = {}
    for meter, amount in value.items():
        if not isinstance(meter, str) or meter in ("hours", "starts"):
            raise ValueError(
                f"{name} names other meters than 'hours' and 'starts' (the "
                f"plan counts those itself); got {meter!r}."
            )
        out[meter] = nonnegative(amount, f"The amount of meter {meter!r}")
    return out


# --- the parts -------------------------------------------------------------


@dataclass(frozen=True, init=False)
class _Part:
    """What every part has: when it applies (``from_``, ``until``), its
    season and its utilisation (each overriding the plan's)."""

    from_: Optional[datetime]
    until: Optional[datetime]
    season: Optional[Tuple[int, ...]]
    utilisation: Optional[float]

    #: Whether the part is made of runs (a start and a stop each).
    runs_kind = True
    kind = ""

    def _common(self, from_, until, season, utilisation) -> None:
        object.__setattr__(self, "from_", _optional_moment(from_, "from_"))
        object.__setattr__(self, "until", _optional_moment(until, "until"))
        object.__setattr__(self, "season", _season(season))
        object.__setattr__(self, "utilisation", _utilisation(utilisation))
        if self.from_ is not None and self.until is not None:
            try:
                ordered = self.from_ < self.until
            except TypeError:
                ordered = True
            if not ordered:
                raise ValueError(
                    f"{self.describe()}: until ({self.until}) must be after "
                    f"from_ ({self.from_})."
                )

    @property
    def meter(self) -> str:
        """The meter the part drives."""
        return "hours"

    def describe(self) -> str:  # pragma: no cover - every part has its own
        raise NotImplementedError

    def _span(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """When the part applies, as given (wall-clock times)."""
        return self.from_, self.until

    def _common_dict(self) -> dict:
        return {
            "from": _iso(self.from_),
            "until": _iso(self.until),
            "season": None if self.season is None else list(self.season),
            "utilisation": self.utilisation,
        }

    @staticmethod
    def _common_kwargs(data: dict) -> dict:
        return {
            "from_": from_iso(data.get("from")),
            "until": from_iso(data.get("until")),
            "season": data.get("season"),
            "utilisation": data.get("utilisation"),
        }

    def _runs(self, clock_: Clock, begin: datetime, end: datetime):
        """The runs (real instants) going on from ``begin`` to ``end``:
        ``(on, off)`` pairs in order."""
        raise NotImplementedError  # pragma: no cover


@dataclass(frozen=True, init=False)
class Runs(_Part):
    """Concrete runs, each a start and a stop: "on at 12:31 on 14 Nov, off
    at 03:27 on 15 Nov".

    Parameters
    ----------
    runs : sequence of (on, off)
        Each run's start and stop: datetimes, or ISO text such as
        ``"2026-11-14 12:31"``. Runs may not overlap.
    from_, until : datetime or str, optional
        The span the runs cover, when it is more than from the first start
        to the last stop (the default): as a specific part, the background
        is not used anywhere in it.
    season, utilisation : optional
        As for ``DutyPlan``, for this part only.

    Examples
    --------
    >>> from repyability.schedule import Runs
    >>> run = Runs([("2026-11-14 12:31", "2026-11-15 03:27")])
    >>> run.hours
    14.933333333333334
    """

    runs: Tuple[Tuple[datetime, datetime], ...]
    kind = "runs"

    def __init__(
        self, runs, *, from_=None, until=None, season=None, utilisation=None
    ) -> None:
        try:
            pairs = [tuple(pair) for pair in runs]
        except TypeError:
            pairs = []
        if not pairs or not all(len(p) == 2 for p in pairs):
            raise TypeError(
                "runs must be a sequence of (on, off) pairs, at least one, "
                "such as [('2026-11-14 12:31', '2026-11-15 03:27')]; got "
                f"{runs!r}."
            )
        parsed = tuple(
            (wall_time(on, "a run's on"), wall_time(off, "a run's off"))
            for on, off in pairs
        )
        try:
            parsed = tuple(sorted(parsed))
        except TypeError:
            raise ValueError(
                "Give every run's times with a zone or every run's without "
                "one, not a mix."
            ) from None
        for on, off in parsed:
            if not on < off:
                raise ValueError(
                    f"A run must stop after it starts: on {on}, off {off}."
                )
        for (_, off), (on, _) in zip(parsed, parsed[1:]):
            if on < off:
                raise ValueError(
                    f"Two runs overlap (one starts at {on}, before another "
                    f"stops, at {off}): each hour may be in one run only."
                )
        object.__setattr__(self, "runs", parsed)
        self._common(from_, until, season, utilisation)

    @property
    def hours(self) -> float:
        """The runs' hours, by the clock (no zone)."""
        return sum((off - on) / timedelta(hours=1) for on, off in self.runs)

    def describe(self) -> str:
        return "exact runs"

    def _span(self):
        return (
            self.runs[0][0] if self.from_ is None else self.from_,
            self.runs[-1][1] if self.until is None else self.until,
        )

    def _runs(self, clock_, begin, end):
        out = []
        for on, off in self.runs:
            a, b = clock_.instant(on, "a run's on"), clock_.instant(off, "off")
            if b > begin and a < end:
                out.append((a, b))
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": VERSION,
            "runs": [
                [on.isoformat(), off.isoformat()] for on, off in self.runs
            ],
            **self._common_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Runs":
        return cls(
            [(from_iso(a), from_iso(b)) for a, b in data["runs"]],
            **cls._common_kwargs(data),
        )


def _window(days, start, stop) -> Tuple[Tuple[int, ...], time, time]:
    start, stop = clock(start, "start"), clock(stop, "stop")
    if start == stop:
        raise ValueError(
            f"A run cannot start and stop at the same time ({start}): give "
            "when it stops (for all day, use Continuous)."
        )
    return weekdays(days), start, stop


def _wall_run(day: date, start: time, stop: time) -> Tuple[datetime, datetime]:
    """The wall-clock run starting on ``day``."""
    on = datetime.combine(day, start)
    off = datetime.combine(day + (_DAY if stop < start else timedelta()), stop)
    return on, off


def _days_runs(clock_, begin, end, runs_on):
    """Real runs from ``runs_on(day)`` (each local day's wall-clock runs)
    that are going on from ``begin`` to ``end``."""
    out = []
    day = clock_.local_naive(begin).date() - _DAY
    last = clock_.local_naive(end).date()
    while day <= last:
        for on, off in runs_on(day):
            a, b = clock_.wall(on), clock_.wall(off)
            if b > begin and a < end and b > a:
                out.append((a, b))
        day += _DAY
    return sorted(out)


@dataclass(frozen=True, init=False)
class Shifts(_Part):
    """A repeating week of clock times: Monday to Friday, 06:00 to 22:00.

    Parameters
    ----------
    days : str or sequence of (days, start, stop)
        The days of the week a run starts on (``"Mon-Fri"``, ``"Sat"``,
        ``"Mon, Wed, Fri"``; a range may wrap round the week), with
        ``start`` and ``stop``; or several such shifts as triples, for two
        shifts with a break. Their runs may not overlap.
    start, stop : datetime.time or str
        The clock times a run starts and stops, such as ``"06:00"`` and
        ``"22:00"``. A stop before the start is on the next day (a run
        overnight).
    from_, until, season, utilisation : optional
        As for ``DutyPlan``, for this part only.

    Examples
    --------
    >>> from repyability.schedule import Shifts
    >>> Shifts("Mon-Fri", "06:00", "22:00").hours_a_week
    80.0
    >>> two = [("Mon-Fri", "06:00", "14:00"), ("Mon-Fri", "14:30", "22:30")]
    >>> Shifts(two).describe()
    'Mon-Fri 06:00-14:00 and Mon-Fri 14:30-22:30'
    """

    windows: Tuple[Tuple[Tuple[int, ...], time, time], ...]
    kind = "shifts"

    def __init__(
        self,
        days,
        start=None,
        stop=None,
        *,
        from_=None,
        until=None,
        season=None,
        utilisation=None,
    ) -> None:
        if start is None and stop is None and not isinstance(days, str):
            try:
                triples = [tuple(w) for w in days]
            except TypeError:
                triples = []
            if not triples or not all(len(w) == 3 for w in triples):
                raise TypeError(
                    "Give Shifts(days, start, stop), or a sequence of "
                    "(days, start, stop) triples; got "
                    f"{days!r}."
                )
            windows = tuple(_window(*w) for w in triples)
        else:
            if start is None or stop is None:
                raise TypeError("Shifts needs days, start and stop.")
            windows = (_window(days, start, stop),)
        object.__setattr__(self, "windows", windows)
        self._common(from_, until, season, utilisation)
        runs = []
        for day in range(15):
            d = date(2001, 1, 1) + timedelta(days=day)  # a Monday
            runs.extend(self._wall_runs(d))
        runs.sort()
        for (_, off), (on, _) in zip(runs, runs[1:]):
            if on < off:
                raise ValueError(
                    f"The runs of {self.describe()} overlap (one starts on "
                    f"{on:%a} at {on:%H:%M}, before another stops, at "
                    f"{off:%H:%M}): each hour may be in one run only."
                )

    def _wall_runs(self, day: date):
        return [
            _wall_run(day, start, stop)
            for days, start, stop in self.windows
            if day.weekday() in days
        ]

    @property
    def hours_a_week(self) -> float:
        """The hours of a week's runs, by the clock (no zone)."""
        total = timedelta()
        for days, start, stop in self.windows:
            on, off = _wall_run(date(2001, 1, 1), start, stop)
            total += (off - on) * len(days)
        return total / timedelta(hours=1)

    def describe(self) -> str:
        return " and ".join(
            f"{describe_days(days)} {start:%H:%M}-{stop:%H:%M}"
            for days, start, stop in self.windows
        )

    def _runs(self, clock_, begin, end):
        return _days_runs(clock_, begin, end, self._wall_runs)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": VERSION,
            "windows": [
                [list(days), f"{start:%H:%M:%S}", f"{stop:%H:%M:%S}"]
                for days, start, stop in self.windows
            ],
            **self._common_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Shifts":
        return cls(
            [tuple(w) for w in data["windows"]], **cls._common_kwargs(data)
        )


@dataclass(frozen=True, init=False)
class Roster(_Part):
    """A cycle of days on and off: four days on, four off, 07:00 to 19:00.

    Parameters
    ----------
    pattern : sequence of int
        Days on and days off, in turn, starting with days on: ``(4, 4)``;
        ``(2, 2, 3, 2, 2, 3)`` for a 2-2-3 roster.
    anchor : date or str
        The first day of a cycle (its first day on).
    start, stop : datetime.time or str
        The clock times each day's run starts and stops (a stop before
        the start is on the next day).
    from_, until, season, utilisation : optional
        As for ``DutyPlan``, for this part only.

    Examples
    --------
    >>> from repyability.schedule import Roster
    >>> Roster((4, 4), "2026-11-02", "07:00", "19:00").describe()
    '4 on, 4 off, 07:00-19:00'
    """

    pattern: Tuple[int, ...]
    anchor: date
    start: time
    stop: time
    kind = "roster"

    def __init__(
        self,
        pattern,
        anchor,
        start,
        stop,
        *,
        from_=None,
        until=None,
        season=None,
        utilisation=None,
    ) -> None:
        try:
            counts = tuple(pattern)
        except TypeError:
            counts = ()
        if (
            not counts
            or len(counts) % 2
            or not all(
                isinstance(c, int) and not isinstance(c, bool) and c >= 1
                for c in counts
            )
        ):
            raise ValueError(
                "pattern must be days on and days off in turn, whole numbers "
                f"of at least 1, starting with days on, such as (4, 4); got "
                f"{pattern!r}."
            )
        _, start, stop = _window("Mon", start, stop)
        object.__setattr__(self, "pattern", counts)
        object.__setattr__(self, "anchor", wall_time(anchor, "anchor").date())
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        self._common(from_, until, season, utilisation)

    def on(self, day: date) -> bool:
        """Whether ``day`` is a day on."""
        k = (day - self.anchor).days % sum(self.pattern)
        for i, count in enumerate(self.pattern):
            if k < count:
                return i % 2 == 0
            k -= count
        return False  # pragma: no cover

    def describe(self) -> str:
        cycle = ", ".join(
            f"{c} {'on' if i % 2 == 0 else 'off'}"
            for i, c in enumerate(self.pattern)
        )
        return f"{cycle}, {self.start:%H:%M}-{self.stop:%H:%M}"

    def _runs(self, clock_, begin, end):
        return _days_runs(
            clock_,
            begin,
            end,
            lambda d: (
                [_wall_run(d, self.start, self.stop)] if self.on(d) else []
            ),
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": VERSION,
            "pattern": list(self.pattern),
            "anchor": self.anchor.isoformat(),
            "start": f"{self.start:%H:%M:%S}",
            "stop": f"{self.stop:%H:%M:%S}",
            **self._common_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Roster":
        return cls(
            data["pattern"],
            data["anchor"],
            data["start"],
            data["stop"],
            **cls._common_kwargs(data),
        )


@dataclass(frozen=True, init=False)
class Continuous(_Part):
    """Running all the time: a 24/7 process pump. A continuous stretch has
    no starts (it is one run with no start of its own in the plan).

    Examples
    --------
    >>> from repyability.schedule import Continuous
    >>> Continuous().describe()
    'all the time'
    """

    kind = "continuous"

    def __init__(
        self, *, from_=None, until=None, season=None, utilisation=None
    ) -> None:
        self._common(from_, until, season, utilisation)

    def describe(self) -> str:
        return "all the time"

    def _runs(self, clock_, begin, end):
        return [(begin, end)]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "version": VERSION, **self._common_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "Continuous":
        return cls(**cls._common_kwargs(data))


def _number_text(x: float) -> str:
    return f"{x:,.6g}"


@dataclass(frozen=True, init=False)
class Rate(_Part):
    """A general rate, spread evenly over each calendar period: 3 h a day;
    15 h a month; 1,200 km a week; or a range, 2 to 4 h a day.

    Parameters
    ----------
    amount : float or (float, float)
        How much of the meter each period: a number more than 0, or a low
        and a high (a range: the forecast then gives the middle, and the
        low and high answers beside it).
    per : str
        ``"hour"``, ``"day"``, ``"week"`` (Monday to Sunday), ``"month"``,
        ``"quarter"`` or ``"year"``. Each calendar period at the site gets
        the amount, spread evenly over its real length (15 h a month is 15
        h in February and in March).
    meter : str, default "hours"
        The meter it drives. For ``"hours"``, the amount cannot be more
        than the period's length.
    from_, until, season, utilisation : optional
        As for ``DutyPlan``, for this part only.

    Examples
    --------
    >>> from repyability.schedule import Rate
    >>> Rate(3, per="day").describe()
    'about 3 h a day'
    >>> Rate((2, 4), per="day").describe()
    'about 2 to 4 h a day'
    """

    amount: float
    low: float
    high: float
    per: str
    meter_name: str
    runs_kind = False
    kind = "rate"

    def __init__(
        self,
        amount,
        per,
        meter: str = "hours",
        *,
        from_=None,
        until=None,
        season=None,
        utilisation=None,
    ) -> None:
        if isinstance(amount, (list, tuple)):
            if len(amount) != 2:
                raise ValueError(
                    f"A range of rates is a low and a high; got {amount!r}."
                )
            low = positive(amount[0], "The low rate")
            high = positive(amount[1], "The high rate")
            if low > high:
                raise ValueError(
                    f"The low rate ({low}) must be at most the high ({high})."
                )
        else:
            low = high = positive(amount, "amount")
        if per not in PERIODS:
            raise ValueError(
                f"per must be one of {', '.join(map(repr, PERIODS))}; got "
                f"{per!r}."
            )
        if not isinstance(meter, str) or not meter or meter == "starts":
            raise ValueError(
                "meter must name the meter the rate drives ('hours', 'km', "
                f"...; starts come from runs); got {meter!r}."
            )
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "amount", (low + high) / 2.0)
        object.__setattr__(self, "per", per)
        object.__setattr__(self, "meter_name", meter)
        self._common(from_, until, season, utilisation)
        if meter == "hours":
            longest = {
                "hour": 1,
                "day": 25,
                "week": 169,
                "month": 745,
                "quarter": 2209,
                "year": 8784,
            }[per]
            if high > longest:
                raise ValueError(
                    f"{self.describe()} is more hours than a {per} has."
                )

    @property
    def meter(self) -> str:
        return self.meter_name

    @property
    def is_range(self) -> bool:
        return self.low != self.high

    def describe(self) -> str:
        unit = "h" if self.meter_name == "hours" else self.meter_name
        amount = (
            f"{_number_text(self.low)} to {_number_text(self.high)}"
            if self.is_range
            else _number_text(self.amount)
        )
        article = "an" if self.per == "hour" else "a"
        return f"about {amount} {unit} {article} {self.per}"

    def _periods(self, clock_: Clock, begin: datetime, end: datetime):
        """``(t0, t1, p0, p1)``: each calendar period ``p0`` to ``p1`` that
        meets ``begin`` to ``end``, and the stretch ``t0`` to ``t1`` of it
        inside them."""
        bounds = clock_.period_bounds(begin, end, self.per)
        out = []
        for a, b in zip(bounds, bounds[1:]):
            lo, hi = max(a, begin), min(b, end)
            if hi > lo:
                out.append((lo, hi, a, b))
        return out

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": VERSION,
            "amount": [self.low, self.high] if self.is_range else self.amount,
            "per": self.per,
            "meter": self.meter_name,
            **self._common_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Rate":
        return cls(
            data["amount"],
            data["per"],
            data.get("meter", "hours"),
            **cls._common_kwargs(data),
        )


Part = Union[Runs, Shifts, Roster, Rate, Continuous]
PARTS = {cls.kind: cls for cls in (Runs, Shifts, Roster, Rate, Continuous)}


def part_from_dict(data: dict) -> Part:
    """A part from its ``to_dict()``."""
    kind = data.get("kind")
    if kind not in PARTS:
        raise ValueError(
            f"Not a duty plan part: kind {kind!r} (one of "
            f"{', '.join(map(repr, PARTS))})."
        )
    return PARTS[kind].from_dict(data)


# --- the plan --------------------------------------------------------------


def _parts(value, name: str) -> Tuple[Part, ...]:
    if value is None:
        return ()
    if isinstance(value, _Part):
        return (cast(Part, value),)
    try:
        parts = tuple(value)
    except TypeError:
        parts = (value,)
    for part in parts:
        if not isinstance(part, _Part):
            raise TypeError(
                f"{name} must be duty plan parts (Runs, Shifts, Roster, Rate "
                f"or Continuous); got {part!r}."
            )
    return parts


@dataclass(frozen=True)
class _Stretch:
    """A stretch of time with one source for a meter: a part of a layer,
    or none."""

    begin: datetime
    end: datetime
    layer: Optional[str]
    part: Optional[Part]


@dataclass(frozen=True, init=False)
class DutyPlan:
    """How a piece of equipment runs: a background, and specific parts on
    top of it, giving its operating hours (and other meters) by any moment.

    Parameters
    ----------
    *parts : Runs, Shifts, Roster, Rate or Continuous
        The background, when it is given positionally: one part, or parts
        that follow each other (each from its ``from_``; one with no
        ``until`` ends where the next starts).
    background : part or sequence of parts, optional
        The same, by name (give one or the other).
    specific : sequence of parts, optional
        Parts on top of the background, each covering its own span
        (``Runs`` from its first start to its last stop, or ``from_`` to
        ``until``): used wherever they cover, the background wherever
        they don't. They may not overlap each other on a meter.
    time_zone : str, optional
        The site's IANA zone (``"Australia/Sydney"``): plans are in its
        wall-clock time and count real hours, so a night shift is an hour
        short or long on the nights the clocks change. None: clock times
        taken as they are, with no daylight saving.
    season : str or sequence of int, optional
        The months it runs in (``"Apr-Oct"``); a run counts by the day it
        starts. A part's own ``season`` takes its place for that part.
    outages : sequence of (start, end), optional
        Moments with no running at all (shutdowns, out of service), in
        every layer.
    utilisation : float, default 1
        The share of the planned running that happens, taken as an
        expected value: each planned hour (and start) counts this much. A
        part's own takes its place for that part.
    per_hour, per_run : dict, optional
        Other meters each operating hour or each run adds:
        ``per_hour={"tonnes": 40}``, ``per_run={"cycles": 1}``.
    fill : {"pro_rata", "top_up"}, default "pro_rata"
        How a background ``Rate`` fills a calendar period that specific
        parts cover in part. ``"pro_rata"``: only the time they leave,
        at the rate spread by time (3 h a day, with the specific runs to
        03:30, gives the rest of the day 3 x 20.5/24 h), so no duty is
        added at the handover. ``"top_up"``, for a rate read as a
        budget (15 h a month): the period gets what the specific parts
        left of its amount (never less than 0), spread over the time they
        leave. A ``Shifts`` or ``Roster`` background simply resumes at its
        next clock time either way, counting only the part of a run that
        no specific part covers.

    Examples
    --------
    >>> from repyability.schedule import DutyPlan, Rate, Runs
    >>> plan = DutyPlan(
    ...     background=Rate(3, per="day"),
    ...     specific=[Runs([("2026-11-14 12:31", "2026-11-15 03:27"),
    ...                     ("2026-11-15 12:30", "2026-11-16 03:30")],
    ...                    from_="2026-11-14")],
    ... )
    >>> plan.forecast("2026-11-14", "2027-11-14").describe()
    'exact runs to 16 Nov 03:30, then about 3 h a day'
    """

    background: Tuple[Part, ...]
    specific: Tuple[Part, ...]
    time_zone: Optional[str]
    season: Optional[Tuple[int, ...]]
    outages: Tuple[Tuple[datetime, datetime], ...]
    utilisation: float
    per_hour: Dict[str, float]
    per_run: Dict[str, float]
    fill: str

    def __init__(
        self,
        *parts,
        background=None,
        specific=None,
        time_zone: Optional[str] = None,
        season=None,
        outages=None,
        utilisation=1.0,
        per_hour=None,
        per_run=None,
        fill: str = "pro_rata",
    ) -> None:
        if fill not in FILLS:
            raise ValueError(
                f"fill must be 'pro_rata' or 'top_up'; got {fill!r}."
            )
        object.__setattr__(self, "fill", fill)
        if parts and background is not None:
            raise TypeError(
                "Give the background positionally or as background=, not both."
            )
        back = _parts(parts if parts else background, "background")
        spec = _parts(specific, "specific")
        if not back and not spec:
            raise ValueError(
                "A duty plan needs at least one part (Runs, Shifts, Roster, "
                "Rate or Continuous). With no plan at all, calendar time is "
                "operating time: use UsageForecast.calendar."
            )
        Clock(time_zone)  # refuses an unknown zone here
        object.__setattr__(self, "background", back)
        object.__setattr__(self, "specific", spec)
        object.__setattr__(self, "time_zone", time_zone)
        object.__setattr__(self, "season", _season(season))
        pairs = []
        for pair in outages or ():
            try:
                a, b = pair
            except (TypeError, ValueError):
                raise TypeError(
                    "outages must be (start, end) pairs; got " f"{pair!r}."
                ) from None
            a, b = wall_time(a, "an outage start"), wall_time(
                b, "an outage end"
            )
            try:
                ordered = a < b
            except TypeError:
                ordered = True
            if not ordered:
                raise ValueError(
                    f"An outage must end after it starts: {a} to {b}."
                )
            pairs.append((a, b))
        object.__setattr__(self, "outages", tuple(pairs))
        u = _utilisation(utilisation)
        object.__setattr__(self, "utilisation", 1.0 if u is None else u)
        object.__setattr__(
            self, "per_hour", _meter_amounts(per_hour, "per_hour")
        )
        object.__setattr__(self, "per_run", _meter_amounts(per_run, "per_run"))
        own = {p.meter for p in back + spec if not p.runs_kind} - {"hours"}
        both = own & (set(self.per_hour) | set(self.per_run))
        if both:
            raise ValueError(
                f"Meter {sorted(both)[0]!r} is given both a Rate of its own "
                "and an amount per hour or per run: give one."
            )
        # Refuse overlaps now, on the spans as given (no zone needed to
        # order them, but where a mix of zones makes them incomparable,
        # the forecast checks again on real instants).
        try:
            self._layers(Clock(time_zone))
        except TypeError:
            pass

    # -- layering ------------------------------------------------------------

    def meters(self) -> Tuple[str, ...]:
        """The meters the plan drives, ``"hours"`` (and ``"starts"``, when
        it has runs) first."""
        parts = self.background + self.specific
        out = []
        if any(p.meter == "hours" for p in parts):
            out.append("hours")
        if any(p.runs_kind for p in parts):
            out.append("starts")
        for name in (
            [p.meter for p in parts if p.meter != "hours"]
            + list(self.per_hour)
            + list(self.per_run)
        ):
            if name not in out:
                out.append(name)
        return tuple(out)

    def _layers(self, clock_: Clock) -> Dict[str, Tuple[list, list]]:
        """For each meter a part drives, its background and specific parts
        as ``(begin, end, part)`` in real instants (None: unbounded),
        refusing overlaps."""
        out: Dict[str, Tuple[list, list]] = {}

        def instant(value, name):
            return None if value is None else clock_.instant(value, name)

        for meter in {p.meter for p in self.background + self.specific}:
            back = []
            for p in self.background:
                if p.meter == meter:
                    a, b = p._span()
                    back.append([instant(a, "from_"), instant(b, "until"), p])
            back.sort(key=lambda s: (s[0] is not None, s[0] or 0))
            for i, span in enumerate(back):
                if span[1] is None:
                    later = [
                        s[0]
                        for s in back[i + 1 :]
                        if s[0] is not None
                        and (span[0] is None or s[0] > span[0])
                    ]
                    if later:
                        span[1] = min(later)
            spec = []
            for p in self.specific:
                if p.meter == meter:
                    a, b = p._span()
                    spec.append([instant(a, "from_"), instant(b, "until"), p])
            for layer, spans in (("background", back), ("specific", spec)):
                for i, (a1, b1, p1) in enumerate(spans):
                    for a2, b2, p2 in spans[i + 1 :]:
                        if _overlap(a1, b1, a2, b2):
                            raise ValueError(
                                f"Two {layer} parts overlap on meter "
                                f"{meter!r}: {p1.describe()} "
                                f"({_span_text(clock_, a1, b1)}) and "
                                f"{p2.describe()} "
                                f"({_span_text(clock_, a2, b2)}). Each "
                                "moment may have one part per layer: end "
                                "one before the other starts (until=, "
                                "from_=), or put one in specific=."
                            )
            out[meter] = (
                [tuple(s) for s in back],
                [tuple(s) for s in spec],
            )
        return out

    def _stretches(
        self, layers, meter: str, begin: datetime, end: datetime
    ) -> List[_Stretch]:
        """The stretches from ``begin`` to ``end`` with one source for
        ``meter``: specific where one covers, else the background."""
        back, spec = layers.get(meter, ([], []))
        cuts = {begin, end}
        for a, b, _ in back + spec:
            for t in (a, b):
                if t is not None and begin < t < end:
                    cuts.add(t)
        bounds = sorted(cuts)
        out: List[_Stretch] = []
        for a, b in zip(bounds, bounds[1:]):
            layer: Optional[str] = None
            part: Optional[Part] = None
            for name, spans in (("specific", spec), ("background", back)):
                for s0, s1, p in spans:
                    if (s0 is None or s0 <= a) and (s1 is None or s1 >= b):
                        layer, part = name, p
                        break
                if part is not None:
                    break
            if out and out[-1].part is part and out[-1].layer == layer:
                out[-1] = _Stretch(out[-1].begin, b, layer, part)
            else:
                out.append(_Stretch(a, b, layer, part))
        return out

    # -- the forecast --------------------------------------------------------

    def forecast(
        self,
        start,
        end,
        readings: Optional[Dict[str, float]] = None,
        read_at=None,
    ):
        """The map between moments and meter readings from ``start`` to
        ``end``, anchored on the readings at ``read_at``.

        Parameters
        ----------
        start, end : datetime, date or str
            The span of the forecast (wall-clock times at the site, unless
            given with a zone).
        readings : dict, optional
            Meter readings at ``read_at``, such as ``{"hours": 12_340}``.
            A meter not read counts from 0 at ``start`` (said in the
            notes).
        read_at : datetime, date or str, optional
            When the meters were read; ``start`` by default. It may be
            before or after ``start``.

        Returns
        -------
        UsageForecast
        """
        from ._usage import _build

        return _build(self, start, end, readings, read_at)

    # -- saving --------------------------------------------------------------

    def describe(self) -> str:
        """The plan in a few words."""
        back = "; ".join(p.describe() for p in self.background)
        if not self.specific:
            return back
        spec = "; ".join(p.describe() for p in self.specific)
        return f"{spec}, over {back}" if back else spec

    def to_dict(self) -> dict:
        """The plan as plain data, ready for ``json.dumps``: every part
        with its ``"kind"``, moments as ISO text, the zone by name."""
        return {
            "kind": "duty_plan",
            "version": VERSION,
            "background": [p.to_dict() for p in self.background],
            "specific": [p.to_dict() for p in self.specific],
            "time_zone": self.time_zone,
            "season": None if self.season is None else list(self.season),
            "outages": [
                [a.isoformat(), b.isoformat()] for a, b in self.outages
            ],
            "utilisation": self.utilisation,
            "per_hour": dict(self.per_hour),
            "per_run": dict(self.per_run),
            "fill": self.fill,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DutyPlan":
        """A plan from its ``to_dict()``."""
        if data.get("kind") != "duty_plan":
            raise ValueError(
                f"Not a duty plan: kind {data.get('kind')!r} ('duty_plan')."
            )
        if data.get("version", VERSION) > VERSION:
            raise ValueError(
                f"This duty plan was saved by a later RePyability (version "
                f"{data['version']}); upgrade to read it."
            )
        return cls(
            background=[part_from_dict(p) for p in data.get("background", [])],
            specific=[part_from_dict(p) for p in data.get("specific", [])],
            time_zone=data.get("time_zone"),
            season=data.get("season"),
            outages=[
                (from_iso(a), from_iso(b)) for a, b in data.get("outages", [])
            ],
            utilisation=data.get("utilisation", 1.0),
            per_hour=data.get("per_hour"),
            per_run=data.get("per_run"),
            fill=data.get("fill", "pro_rata"),
        )


def _overlap(a1, b1, a2, b2) -> bool:
    """Whether two spans (None: unbounded) share any time."""
    starts_before_end = a1 is None or b2 is None or a1 < b2
    ends_after_start = b1 is None or a2 is None or b1 > a2
    return starts_before_end and ends_after_start


def _span_text(clock_: Clock, a, b) -> str:
    first = "the start" if a is None else str(clock_.local(a))
    last = "no end" if b is None else str(clock_.local(b))
    return f"{first} to {last}"


def _subtract(
    pieces: Sequence[Tuple[datetime, datetime]],
    holes: Sequence[Tuple[datetime, datetime]],
) -> List[Tuple[datetime, datetime, bool, bool]]:
    """``pieces`` less ``holes``: ``(a, b, after_hole, before_hole)``, the
    last two saying whether a piece's start is a hole's end and its end a
    hole's start."""
    out = []
    for a, b in pieces:
        parts = [(a, b, False, False)]
        for h0, h1 in holes:
            cut = []
            for p0, p1, s, e in parts:
                if h1 <= p0 or h0 >= p1:
                    cut.append((p0, p1, s, e))
                    continue
                if h0 > p0:
                    cut.append((p0, h0, s, True))
                if h1 < p1:
                    cut.append((h1, p1, True, e))
            parts = cut
        out.extend(parts)
    return out


def _in_season(clock_: Clock, moment: datetime, season) -> bool:
    return season is None or clock_.local_naive(moment).month in season


def _season_spans(clock_: Clock, a: datetime, b: datetime, season):
    """``a`` to ``b`` cut at local month starts, keeping the months in
    ``season``."""
    if season is None:
        return [(a, b)]
    cuts = (
        [a]
        + [t for t in clock_.period_bounds(a, b, "month") if a < t < b]
        + [b]
    )
    return [
        (x, y) for x, y in zip(cuts, cuts[1:]) if _in_season(clock_, x, season)
    ]


_HOUR = timedelta(hours=1)

#: How a background rate fills a calendar period part covered by specific
#: parts: by the time left (``"pro_rata"``), or up to the period's amount
#: (``"top_up"``).
FILLS = ("pro_rata", "top_up")


@dataclass(frozen=True)
class _Pieces:
    """What a plan gives over a span, ready for the map: each meter's
    linear stretches and jumps, the runs (with their real stops) and the
    stretches by source."""

    lines: Dict[str, List[Tuple[datetime, datetime, float, float, float]]]
    jumps: Dict[str, List[Tuple[datetime, float]]]
    runs: List[Tuple[datetime, datetime, Optional[datetime]]]
    stretches: List[_Stretch]


def _part_output(plan, clock_, layers, stretch, holes, meter, end):
    """What the part of ``stretch`` gives over it: the meter's lines
    ``(a, b, mid, low, high)`` (per real hour), its runs ``(a, b, stop)``
    and its starts ``(moment, weight)``."""
    part = stretch.part
    u = plan.utilisation if part.utilisation is None else part.utilisation
    season = plan.season if part.season is None else part.season
    lines: list = []
    runs: list = []
    starts: list = []
    if part.runs_kind:
        continuous = isinstance(part, Continuous)
        if continuous:
            raw = [(stretch.begin, stretch.end)]
        else:
            raw = part._runs(clock_, stretch.begin, stretch.end)
        until = (
            clock_.instant(part.until, "until")
            if continuous and part.until is not None
            else None
        )
        for on, off in raw:
            if continuous:
                spans = _season_spans(clock_, on, off, season)
            elif _in_season(clock_, on, season):
                spans = [(on, off)]
            else:
                continue
            for s_on, s_off in spans:
                a0, b0 = max(s_on, stretch.begin), min(s_off, stretch.end)
                if b0 <= a0:
                    continue
                # A run cut at the start of its stretch (the handover from
                # another part) has no start of its own here; one cut at
                # the end has no known stop, unless that is the forecast's
                # end (it then stops when the run does).
                counted = not continuous and s_on >= stretch.begin
                if continuous:
                    real_stop = (
                        s_off if until is not None and s_off == until else None
                    )
                elif s_off <= stretch.end or stretch.end == end:
                    real_stop = s_off
                else:
                    real_stop = None
                for a, b, after_hole, before_hole in _subtract(
                    [(a0, b0)], holes
                ):
                    runs.append((a, b, b if before_hole else real_stop))
                    lines.append((a, b, u, u, u))
                    if (a == a0 and counted) or after_hole:
                        starts.append((a, u))
        return lines, runs, starts
    for t0, t1, p0, p1 in part._periods(clock_, stretch.begin, stretch.end):
        length = (p1 - p0) / _HOUR
        amounts: Tuple[float, ...] = (part.amount, part.low, part.high)
        room = length
        if plan.fill == "top_up" and stretch.layer == "background":
            given, covered = _specific_in(
                plan, clock_, layers, meter, p0, p1, holes, end
            )
            if covered > 0.0:
                room = length - covered
                amounts = tuple(
                    max(0.0, x - g) for x, g in zip(amounts, given)
                )
        rates = tuple(x / room for x in amounts) if room > 0.0 else (0.0,) * 3
        if meter == "hours" and rates[2] > 1.0 + 1e-12:
            if plan.fill == "top_up" and room < length:
                clock_._note(
                    f"Topping up {part.describe()} in the {part.per} from "
                    f"{format_moment(clock_.local(p0))} needs more hours than "
                    "are left in it: it runs all the rest of it."
                )
                rates = tuple(min(r, 1.0) for r in rates)
            else:
                raise ValueError(
                    f"{part.describe()} is more hours than the {part.per} "
                    f"from {clock_.local(p0)} has."
                )
        for a1, b1 in _season_spans(clock_, t0, t1, season):
            for a, b, _, _ in _subtract([(a1, b1)], holes):
                lines.append((a, b) + tuple(r * u for r in rates))
    return lines, runs, starts


def _specific_in(plan, clock_, layers, meter, p0, p1, holes, end):
    """The amounts (middle, low, high) the specific parts give ``meter``
    from ``p0`` to ``p1``, and the hours they cover."""
    given = [0.0, 0.0, 0.0]
    covered = 0.0
    for st in plan._stretches(layers, meter, p0, p1):
        if st.layer != "specific":
            continue
        covered += (st.end - st.begin) / _HOUR
        lines, _, _ = _part_output(plan, clock_, layers, st, holes, meter, end)
        for a, b, *rates in lines:
            hours = (b - a) / _HOUR
            for k in range(3):
                given[k] += rates[k] * hours
    return given, covered


def pieces(plan: DutyPlan, clock_: Clock, begin: datetime, end: datetime):
    """What ``plan`` gives from ``begin`` to ``end`` (real instants)."""
    layers = plan._layers(clock_)
    holes = sorted(
        (
            clock_.instant(a, "an outage start"),
            clock_.instant(b, "an outage end"),
        )
        for a, b in plan.outages
    )
    meters = plan.meters()
    lines: Dict[str, list] = {m: [] for m in meters}
    jumps: Dict[str, list] = {m: [] for m in meters}
    runs: list = []
    if "hours" in layers:
        stretches = plan._stretches(layers, "hours", begin, end)
    else:
        stretches = [_Stretch(begin, end, None, None)]
    for st in stretches:
        if st.part is None or "hours" not in layers:
            continue
        got, its_runs, starts = _part_output(
            plan, clock_, layers, st, holes, "hours", end
        )
        lines["hours"].extend(got)
        runs.extend(its_runs)
        for a, b, mid, lo, hi in got:
            for m, k in plan.per_hour.items():
                lines[m].append((a, b, k * mid, k * lo, k * hi))
        if "starts" in jumps:
            for t, w in starts:
                jumps["starts"].append((t, w))
                for m, k in plan.per_run.items():
                    jumps[m].append((t, k * w))
    for meter in meters:
        if meter in ("hours", "starts") or meter not in layers:
            continue
        for st in plan._stretches(layers, meter, begin, end):
            if st.part is not None:
                got, _, _ = _part_output(
                    plan, clock_, layers, st, holes, meter, end
                )
                lines[meter].extend(got)
    return _Pieces(lines, jumps, sorted(runs), stretches)


__all__ = [
    "CALENDAR_PERIODS",
    "Continuous",
    "DutyPlan",
    "Rate",
    "Roster",
    "Runs",
    "Shifts",
    "part_from_dict",
]
