"""Assets that run to an operating schedule, their meters, and the times
a task by use falls due (#319, UC-10 of this package's ``README.md``).

An asset runs in windows of clock time: on the days named, from a start
time to a stop time (overnight, into the next day, when the stop is
earlier than the start). Each run adds its hours to the asset's
``"hours"`` meter and one to its ``"starts"`` meter. From the meters as
read at a time, a task done every so many hours (or starts) falls due at
the moment its meter crosses the next multiple after the reading at its
last time: walking the runs from the reading, the run in which the meter
reaches it, and the time within that run. It is exact, with every run
going ahead. Times are clock times, with no daylight saving.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterator, Optional, Set, Tuple, Union

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: The days of the week, by their names and the first three letters.
_DAYS = {
    name: number
    for number, full in enumerate(
        (
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        )
    )
    for name in (full, full[:3])
}

#: The meters an asset may have: what each run adds to them.
METERS = ("hours", "starts")

_HOUR = timedelta(hours=1)


def _weekdays(days) -> Tuple[int, ...]:
    """``days`` as weekday numbers (Monday 0): text naming days and ranges,
    separated by commas, such as ``"Mon-Fri"`` or ``"Mon, Wed, Fri"``; a
    range may wrap round the week (``"Fri-Mon"``)."""
    if not isinstance(days, str):
        raise TypeError(
            "days must be text naming days of the week, such as 'Mon-Fri' "
            f"or 'Mon, Wed, Fri'; got {days!r}."
        )
    numbers: Set[int] = set()
    for part in days.split(","):
        ends = [end.strip().lower() for end in part.split("-")]
        if len(ends) > 2 or not all(end in _DAYS for end in ends):
            raise ValueError(
                "days must name days of the week, or ranges of them, "
                f"separated by commas (such as 'Mon-Fri'); got {days!r}."
            )
        first, last = _DAYS[ends[0]], _DAYS[ends[-1]]
        numbers.update((first + k) % 7 for k in range((last - first) % 7 + 1))
    return tuple(sorted(numbers))


def _clock(value, name: str) -> time:
    """``value`` as a clock time: a ``datetime.time``, or text such as
    ``"06:00"``."""
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        try:
            return time.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"{name} must be a clock time written as HH:MM, such as "
                f"'06:00'; got {value!r}."
            ) from None
    raise TypeError(
        f"{name} must be a clock time (a datetime.time, or text such as "
        f"'06:00'); got {value!r}."
    )


def _moment(value, name: str) -> datetime:
    """``value`` as a date and time: a ``datetime``, a ``date`` (its
    midnight), or ISO text such as ``"2026-10-12 06:00"``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"{name} must be a date and time written as YYYY-MM-DD "
                f"HH:MM, such as '2026-10-12 06:00'; got {value!r}."
            ) from None
    raise TypeError(
        f"{name} must be a date and time (a datetime, or text such as "
        f"'2026-10-12 06:00'); got {value!r}."
    )


@dataclass(frozen=True, init=False)
class Window:
    """When an [`Asset`][repyability.Asset] runs: on the days named, from
    a start time to a stop time.

    Parameters
    ----------
    days : str
        The days of the week a run starts on: names or their first three
        letters, and ranges of them, separated by commas (``"Mon-Fri"``,
        ``"Sat"``, ``"Mon, Wed, Fri"``). A range may wrap round the week.
    start, stop : datetime.time or str
        The clock times a run starts and stops, such as ``"06:00"`` and
        ``"22:00"``. A stop earlier than the start is on the next day (a
        run overnight); the two may not be the same.

    Examples
    --------
    >>> from repyability.maintenance import Window
    >>> shift = Window("Mon-Fri", "06:00", "22:00")
    >>> shift.hours
    16.0
    >>> Window("Fri", "22:00", "06:00").hours
    8.0
    """

    days: Tuple[int, ...]
    start: time
    stop: time

    def __init__(self, days, start, stop) -> None:
        start, stop = _clock(start, "start"), _clock(stop, "stop")
        if start == stop:
            raise ValueError(
                f"A run cannot start and stop at the same time ({start}): "
                "give when it stops."
            )
        object.__setattr__(self, "days", _weekdays(days))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def hours(self) -> float:
        """The hours of each run."""
        day = date(2000, 1, 3)
        begins, ends = self._run(day)
        return (ends - begins) / _HOUR

    def _run(self, day: date) -> Tuple[datetime, datetime]:
        """The run starting on ``day``: its start and stop."""
        begins = datetime.combine(day, self.start)
        ends = datetime.combine(day, self.stop)
        if self.stop < self.start:
            ends += timedelta(days=1)
        return begins, ends


@dataclass(frozen=True, init=False, eq=False)
class Asset:
    """An asset that runs to an operating schedule, with meters read at a
    time: a pump on shifts, an aircraft's flights.

    Parameters
    ----------
    name : str
        What the asset is called.
    runs : Window or sequence of Window
        When it runs. Its windows' runs may not overlap.
    meters : dict
        Its meters' readings: ``"hours"`` (its running hours) and
        ``"starts"`` (how many runs it has started), each at least 0.
    read_at : datetime.datetime or str
        When the meters were read, such as ``"2026-10-12 00:00"``. A run
        in progress then counts from that time on; one starting then
        counts its start.

    Examples
    --------
    >>> from repyability.maintenance import Asset, Window
    >>> pump = Asset(
    ...     "pump 1",
    ...     runs=Window("Mon-Fri", "06:00", "22:00"),
    ...     meters={"hours": 12_340},
    ...     read_at="2026-10-12 00:00",
    ... )
    >>> pump.meters
    {'hours': 12340.0}
    """

    name: str
    runs: Tuple[Window, ...]
    meters: Dict[str, float]
    read_at: datetime

    def __init__(self, name: str, runs, meters, read_at) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be text; got {name!r}.")
        windows = (runs,) if isinstance(runs, Window) else tuple(runs)
        if not windows or not all(isinstance(w, Window) for w in windows):
            raise TypeError(
                "runs must be a Window, or a sequence of them, at least "
                f"one; got {runs!r}."
            )
        if not isinstance(meters, dict) or not meters:
            raise TypeError(
                "meters must be a dict of readings, such as {'hours': "
                f"12340}}; got {meters!r}."
            )
        readings = {}
        for meter, reading in meters.items():
            if meter not in METERS:
                raise ValueError(
                    f"An asset has the meters {', '.join(METERS)}; got "
                    f"{meter!r}."
                )
            if not (
                checks.is_number(reading) and 0.0 <= float(reading) < math.inf
            ):
                raise ValueError(
                    f"The reading of meter {meter!r} must be a number, at "
                    f"least 0; got {reading!r}."
                )
            readings[meter] = float(reading)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "runs", windows)
        object.__setattr__(self, "meters", readings)
        object.__setattr__(self, "read_at", _moment(read_at, "read_at"))
        self._check_runs_apart()

    def _check_runs_apart(self) -> None:
        """Refuse windows whose runs overlap, which would count the same
        hours twice: over a fortnight, every run of every window."""
        start = self.read_at
        runs = list(self._runs(start, start + timedelta(days=15)))
        for (_, ended), (begun, _) in zip(runs, runs[1:]):
            if begun < ended:
                raise ValueError(
                    f"The runs of asset {self.name!r} overlap (one starts "
                    f"at {begun}, before another stops, at {ended}): each "
                    "hour may be in one run only."
                )

    def _runs(
        self, start: datetime, end: datetime
    ) -> Iterator[Tuple[datetime, datetime]]:
        """Every run that is going on or starts from ``start`` to before
        ``end``, in order of their starts."""
        day = start.date() - timedelta(days=1)
        while datetime.combine(day, time()) < end:
            runs = sorted(
                window._run(day)
                for window in self.runs
                if day.weekday() in window.days
            )
            for begins, ends in runs:
                if ends > start and begins < end:
                    yield begins, ends
            day += timedelta(days=1)


@dataclass(frozen=True, init=False)
class Hours:
    """A task's interval in running hours: ``Hours(500)``, a task every
    500 hours of an [`Asset`][repyability.Asset]'s running."""

    amount: float

    def __init__(self, amount) -> None:
        if not (checks.is_number(amount) and 0.0 < float(amount) < math.inf):
            raise ValueError(
                f"Hours must be a number, more than 0; got {amount!r}."
            )
        object.__setattr__(self, "amount", float(amount))

    meter = "hours"


@dataclass(frozen=True, init=False)
class Starts:
    """A task's interval in starts: ``Starts(1000)``, a task every 1,000
    runs an [`Asset`][repyability.Asset] starts (an aircraft's cycles, a
    pump's starts)."""

    amount: float

    def __init__(self, amount) -> None:
        if not (checks.is_whole(amount) and amount >= 1):
            raise ValueError(
                f"Starts must be a whole number, at least 1; got {amount!r}."
            )
        object.__setattr__(self, "amount", float(amount))

    meter = "starts"


Usage = Union[Hours, Starts]


def crossings(
    asset: Asset,
    every: Usage,
    last_done: float,
    start: datetime,
    end: datetime,
) -> Tuple[Tuple[float, ...], Tuple[datetime, ...]]:
    """The readings at which a task every ``every`` of ``asset``, last done
    at ``last_done``, falls due from ``start`` to before ``end``, and the
    times its meter crosses them (see the module's notes)."""
    meter = asset.meters[every.meter]
    target = last_done + every.amount
    readings, times = [], []
    for begins, ends in asset._runs(asset.read_at, end):
        if every.meter == "starts":
            if begins < asset.read_at:
                continue
            meter += 1.0
            while target <= meter:
                if begins >= start:
                    readings.append(target)
                    times.append(begins)
                target += every.amount
        else:
            begun = max(begins, asset.read_at)
            run = (ends - begun) / _HOUR
            while target <= meter + run:
                moment = begun + (target - meter) * _HOUR
                if moment >= end:
                    break
                if moment >= start:
                    readings.append(target)
                    times.append(moment)
                target += every.amount
            meter += run
    return tuple(readings), tuple(times)


@dataclass
class UsageDue(_ResultMapping):
    """When a [`Task`][repyability.Task] by use falls due in a period:
    returned by its ``due``.

    Attributes
    ----------
    task : str
        The task's name.
    start, end : datetime.date
        The period's first and last days, both included.
    meter : str
        The meter the task goes by: ``"hours"`` or ``"starts"``.
    readings : tuple of float
        The meter readings at which it falls due in the period, in order.
    times : tuple of datetime.datetime
        When the meter reaches each of them.
    count : int
        How many fall due in the period.
    cost : float or None
        What they cost: the task's cost each time, times ``count``; None
        if the task was given no cost.
    """

    task: str
    start: date
    end: date
    meter: str
    readings: Tuple[float, ...]
    times: Tuple[datetime, ...]
    count: int
    cost: Optional[float]
