"""Moments, time zones, periods and plain data for ``repyability.schedule``.

Everything here is arithmetic on dates and times; nothing reads the clock.

* **Moments in, moments out.** A moment is given as a ``datetime``, a
  ``date`` (its midnight) or ISO text (``"2026-11-16 06:00"``). Given with
  no zone it is a wall-clock time at the site (the plan's ``time_zone``);
  given with one it is converted to the site's. Answers come back as the
  site's wall-clock times, carrying the zone when there is one, and with
  no zone when the plan has none.
* **Real elapsed time.** Underneath, every moment is a real instant (UTC),
  so a run across the night the clocks go forward is an hour short and
  the night they go back an hour long: the meter counts real hours. With
  no zone, wall-clock times are taken as they are, with no daylight
  saving.
* **Times that don't exist or happen twice.** A wall-clock time in the gap
  when the clocks go forward (02:30 on such a night) moves to the end of
  the gap, the first real moment after it (03:00); a time that happens
  twice when they go back means the first. Each is said in the notes of
  the answer that met it.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from repyability.rbd.results import plain
from repyability.utils.checks import is_number

UTC = timezone.utc

#: The lengths of the periods a rate may be given per, and the periods a
#: forecast is summed by: fixed lengths in hours, or calendar periods.
FIXED_PERIODS = {"hour": 1.0}
CALENDAR_PERIODS = ("day", "week", "month", "quarter", "year")
PERIODS = tuple(FIXED_PERIODS) + CALENDAR_PERIODS

#: Model-time units, in hours. Model time is operating time; a model's
#: unit is the one its life was fitted in.
UNITS = {
    "seconds": 1.0 / 3600.0,
    "minutes": 1.0 / 60.0,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 168.0,
    "years": 24.0 * 365.25,
}

# TODO(units): RePyability's model-unit work reads the models' unit labels
# (surpyval's ``model.unit``) and compares them, but converts nothing, so
# this module keeps the one table of time units above (and the spellings
# below). Replace both with a shared table if the units work brings one.
#: The spellings of a model's time unit label read as each of ``UNITS``.
UNIT_LABELS = {
    label: unit
    for unit, labels in {
        "seconds": ("s", "sec", "secs", "second", "seconds"),
        "minutes": ("min", "mins", "minute", "minutes"),
        "hours": ("h", "hr", "hrs", "hour", "hours"),
        "days": ("d", "day", "days"),
        "weeks": ("w", "wk", "wks", "week", "weeks"),
        "years": ("y", "yr", "yrs", "year", "years"),
    }.items()
    for label in labels
}


def time_unit(label) -> Optional[str]:
    """The unit of ``UNITS`` a model's unit label names (``"h"``,
    ``"hrs"`` and ``"hours"`` are all ``"hours"``); None when it names no
    time unit (``"km"``, ``"cycles"``) or is not text."""
    if not isinstance(label, str):
        return None
    return UNIT_LABELS.get(" ".join(label.lower().split()))


#: The days of the week, by their names and the first three letters.
DAYS = {
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
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def positive(value, name: str) -> float:
    """``value`` as a float more than 0 and finite, or a ``ValueError``
    naming it."""
    if not (is_number(value) and 0.0 < float(value) < math.inf):
        raise ValueError(
            f"{name} must be a number more than 0; got {value!r}."
        )
    return float(value)


def nonnegative(value, name: str) -> float:
    """``value`` as a float at least 0 and finite."""
    if not (is_number(value) and 0.0 <= float(value) < math.inf):
        raise ValueError(
            f"{name} must be a number, at least 0; got {value!r}."
        )
    return float(value)


def weekdays(days) -> Tuple[int, ...]:
    """``days`` as weekday numbers (Monday 0): text naming days and ranges,
    separated by commas, such as ``"Mon-Fri"`` or ``"Mon, Wed, Fri"``; a
    range may wrap round the week (``"Fri-Mon"``). A sequence of weekday
    numbers is taken too."""
    if isinstance(days, (list, tuple)) and all(
        isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6
        for d in days
    ):
        if not days:
            raise ValueError("days must name at least one day of the week.")
        return tuple(sorted(set(days)))
    if not isinstance(days, str):
        raise TypeError(
            "days must be text naming days of the week, such as 'Mon-Fri' "
            f"or 'Mon, Wed, Fri'; got {days!r}."
        )
    numbers: set = set()
    for part in days.split(","):
        ends = [end.strip().lower() for end in part.split("-")]
        if len(ends) > 2 or not all(end in DAYS for end in ends):
            raise ValueError(
                "days must name days of the week, or ranges of them, "
                f"separated by commas (such as 'Mon-Fri'); got {days!r}."
            )
        first, last = DAYS[ends[0]], DAYS[ends[-1]]
        numbers.update((first + k) % 7 for k in range((last - first) % 7 + 1))
    return tuple(sorted(numbers))


def describe_days(days: Tuple[int, ...]) -> str:
    """Weekday numbers as short text: ``(0, 1, 2, 3, 4)`` is
    ``"Mon-Fri"``."""
    if len(days) == 7:
        return "every day"
    runs: List[List[int]] = []
    for d in days:
        if runs and d == runs[-1][-1] + 1:
            runs[-1].append(d)
        else:
            runs.append([d])
    parts = []
    for run in runs:
        if len(run) >= 3:
            parts.append(f"{DAY_NAMES[run[0]]}-{DAY_NAMES[run[-1]]}")
        else:
            parts.extend(DAY_NAMES[d] for d in run)
    return ", ".join(parts)


def clock(value, name: str) -> time:
    """``value`` as a clock time: a ``datetime.time``, or text such as
    ``"06:00"``."""
    if isinstance(value, time):
        return value.replace(tzinfo=None)
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


def wall_time(value, name: str) -> datetime:
    """``value`` as a ``datetime`` as given (a ``date`` as its midnight,
    ISO text parsed); its zone, if any, kept."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time())
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(
                f"{name} must be a date and time written as YYYY-MM-DD "
                f"HH:MM, such as '2026-11-16 06:00'; got {value!r}."
            ) from None
    raise TypeError(
        f"{name} must be a date and time (a datetime, a date, or text such "
        f"as '2026-11-16 06:00'); got {value!r}."
    )


def zone(name) -> Optional[ZoneInfo]:
    """The IANA zone named, or None for clock times with no daylight
    saving."""
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError(
            "time_zone must be an IANA zone name, such as "
            f"'Australia/Sydney', or None; got {name!r}."
        )
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(
            f"time_zone {name!r} is not a time zone this computer knows: "
            "give an IANA name, such as 'Australia/Sydney' or 'UTC' (on "
            "Windows, install the tzdata package)."
        ) from None


def format_moment(moment: datetime, year: bool = True) -> str:
    """A moment in plain words: ``"16 Nov 2026 03:30"``, the time left
    out at midnight and the year when ``year`` is false."""
    text = f"{moment.day} {MONTH_NAMES[moment.month - 1]}"
    if year:
        text += f" {moment.year}"
    if (moment.hour, moment.minute, moment.second) != (0, 0, 0):
        text += f" {moment:%H:%M}"
    return text


class Clock:
    """A site's clock: wall-clock times there and the real instants they
    are, for an IANA zone or none (clock times taken as they are)."""

    def __init__(self, time_zone: Optional[str]) -> None:
        self.name = time_zone
        self.zone = zone(time_zone)
        self.notes: List[str] = []

    def instant(self, value, name: str) -> datetime:
        """``value`` (a moment as given) as a real instant, in UTC."""
        moment = wall_time(value, name)
        if moment.tzinfo is not None:
            if self.zone is None:
                # A plan with no zone keeps clock times: an aware moment's
                # own wall-clock time is taken.
                moment = moment.replace(tzinfo=None)
            else:
                return moment.astimezone(UTC)
        return self.wall(moment)

    def wall(self, moment: datetime) -> datetime:
        """A wall-clock time at the site (naive) as a real instant, moving
        a time that doesn't exist to the end of its gap and taking the
        first of a time that happens twice, with a note."""
        if self.zone is None:
            return moment.replace(tzinfo=UTC)
        first = moment.replace(tzinfo=self.zone, fold=0)
        second = moment.replace(tzinfo=self.zone, fold=1)
        if first.utcoffset() == second.utcoffset():
            return first.astimezone(UTC)
        early, late = sorted((first.astimezone(UTC), second.astimezone(UTC)))
        back = first.astimezone(UTC).astimezone(self.zone).replace(tzinfo=None)
        if back == moment:
            self._note(
                f"{format_moment(moment)} happens twice in {self.name} "
                "(the clocks go back): taken as the first."
            )
            return early
        # The gap: find the instant the offset changes.
        while late - early > timedelta(seconds=1):
            middle = early + (late - early) / 2
            if middle.astimezone(self.zone).utcoffset() == (
                early.astimezone(self.zone).utcoffset()
            ):
                early = middle
            else:
                late = middle
        late = late.replace(microsecond=0)
        self._note(
            f"{format_moment(moment)} does not exist in {self.name} (the "
            "clocks go forward): taken as "
            f"{format_moment(self.local(late))}."
        )
        return late

    def local(self, instant: datetime) -> datetime:
        """A real instant as the site's wall-clock time: carrying the zone,
        or naive when there is none."""
        if self.zone is None:
            return instant.astimezone(UTC).replace(tzinfo=None)
        return instant.astimezone(self.zone)

    def local_naive(self, instant: datetime) -> datetime:
        """A real instant as the site's wall-clock time, with no zone."""
        return self.local(instant).replace(tzinfo=None)

    def midnight(self, day: date) -> datetime:
        """The instant a local day starts."""
        return self.wall(datetime.combine(day, time()))

    def _note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def period_bounds(
        self, begin: datetime, end: datetime, per: str
    ) -> List[datetime]:
        """The instants the local calendar periods ``per`` start, from the
        one holding ``begin`` to the first at or after ``end``."""
        if per in FIXED_PERIODS:
            step = timedelta(hours=FIXED_PERIODS[per])
            out = [begin]
            while out[-1] < end:
                out.append(out[-1] + step)
            return out
        local = self.local_naive(begin)
        day = local.date()
        if per == "day":
            first = day
        elif per == "week":
            first = day - timedelta(days=day.weekday())
        elif per == "month":
            first = day.replace(day=1)
        elif per == "quarter":
            first = date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
        else:
            first = date(day.year, 1, 1)
        bounds = [self.midnight(first)]
        k = 0
        while bounds[-1] < end:
            k += 1
            bounds.append(self.midnight(step_period(first, per, k)))
        return bounds


def step_period(first: date, per: str, k: int) -> date:
    """The start of the ``k``-th calendar period ``per`` after ``first``."""
    if per == "day":
        return first + timedelta(days=k)
    if per == "week":
        return first + timedelta(weeks=k)
    months = {"month": 1, "quarter": 3, "year": 12}[per] * k
    index = first.year * 12 + first.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def to_plain(value) -> Any:
    """``value`` as plain data for JSON: moments as ISO text, the rest as
    ``repyability.rbd.results.plain`` makes it."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_plain(v) for v in value.tolist()]
    return plain(value)


def from_iso(value) -> Optional[datetime]:
    """ISO text back to a ``datetime`` (None stays None)."""
    if value is None:
        return None
    return datetime.fromisoformat(value)
