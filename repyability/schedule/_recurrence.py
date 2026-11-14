"""Recurrence rules: which dates a rule gives (``Recurrence``), and the part
of each year something applies in (``Season``).

Dates are whole local dates (a ``datetime.date``, or ISO text such as
``"2026-11-14"``): a rule such as "the first Saturday" has no time of day,
so no time zone or daylight saving can move it. Every period asked about
includes **both** of its ends. The date arithmetic is python-dateutil's
``rrule`` (the iCalendar standard's, RFC 5545), behind these types, which
are what is saved (``to_dict``); two choices differ from iCalendar on
purpose: a day of the month that a short month lacks (the 30th in
February) falls on that month's last day by default, and months or years
counted from an anchor keep the anchor's day.
"""

from __future__ import annotations

import calendar as _calendar
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from dateutil.rrule import (
    DAILY,
    FR,
    MO,
    MONTHLY,
    SA,
    SU,
    TH,
    TU,
    WE,
    WEEKLY,
    YEARLY,
    rrule,
    rrulestr,
)

from repyability.rbd.results import _ResultMapping

#: The version of the saved shapes (``to_dict``), read back by later
#: releases.
VERSION = 1

#: The days of the week, Monday first.
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_RRULE_DAYS = (MO, TU, WE, TH, FR, SA, SU)
_DAY_NUMBERS = {
    key: number
    for number, name in enumerate(WEEKDAYS)
    for key in (name.lower(), name[:3].lower())
}

MONTHS = tuple(_calendar.month_name[1:])
_MONTH_NUMBERS = {
    key: number
    for number, name in enumerate(MONTHS, start=1)
    for key in (name.lower(), name[:3].lower())
}

#: The units ``Recurrence.every`` counts in.
UNITS = ("days", "weeks", "months", "years")

#: What an excluded date does (``moved=``).
MOVES = ("skip", "before", "after")

_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


# ---------------------------------------------------------------- checks


def _is_whole(value) -> bool:
    """Whether ``value`` is a whole number (``3`` or ``3.0``, not
    ``True``)."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return (
        isinstance(value, (float, np.floating)) and float(value).is_integer()
    )


def _is_number(value) -> bool:
    """Whether ``value`` is a real, finite number (not a bool)."""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(
        value, (int, float, np.integer, np.floating)
    ) and np.isfinite(float(value))


def _whole(value, name: str, low: int = 1, high: Optional[int] = None):
    """``value`` as an ``int`` from ``low`` to ``high``, or a
    ``ValueError`` naming it."""
    if (
        not _is_whole(value)
        or value < low
        or (high is not None and value > high)
    ):
        limit = (
            f"from {low} to {high}" if high is not None else f"at least {low}"
        )
        raise ValueError(
            f"{name} must be a whole number {limit}; got {value!r}."
        )
    return int(value)


def _date(value, name: str) -> date:
    """``value`` as a ``date``: a ``datetime.date`` or ISO text such as
    ``"2026-11-14"``. A ``datetime`` is refused: rules work in whole
    dates."""
    if isinstance(value, datetime):
        raise TypeError(
            f"{name} must be a date (a datetime.date, or text such as "
            f"'2026-11-14') without a time of day; got {value!r}."
        )
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"{name} must be a date written as YYYY-MM-DD, such as "
                f"'2026-11-14'; got {value!r}."
            ) from None
    raise TypeError(
        f"{name} must be a date (a datetime.date, or text such as "
        f"'2026-11-14'); got {value!r}."
    )


def _period(start, end) -> Tuple[date, date]:
    """A period's first and last dates, both included."""
    first, last = _date(start, "start"), _date(end, "end")
    if last < first:
        raise ValueError(
            f"The period ends ({last}) before it starts ({first}): give "
            "the earlier date as start."
        )
    return first, last


def _weekday(value, name: str = "weekday") -> int:
    """A day of the week by name or its first three letters (any case),
    as its number, Monday 0."""
    key = value.strip().lower() if isinstance(value, str) else None
    if key not in _DAY_NUMBERS:
        raise ValueError(
            f"{name} must be a day of the week, such as 'Saturday' or "
            f"'Sat'; got {value!r}."
        )
    return _DAY_NUMBERS[key]


def _weekdays(value) -> Tuple[int, ...]:
    """Days of the week: a name (``"Saturday"``), text naming days and
    ranges separated by commas (``"Mon-Fri"``, ``"Mon, Wed, Fri"``; a
    range may wrap round the week), or a list of names."""
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)) and value:
        parts = list(value)
        if not all(isinstance(p, str) for p in parts):
            raise TypeError(
                "weekdays must be names of days, such as ['Mon', 'Wed']; "
                f"got {value!r}."
            )
    else:
        raise TypeError(
            "weekdays must name days of the week, such as 'Saturday', "
            f"'Mon-Fri' or ['Mon', 'Wed', 'Fri']; got {value!r}."
        )
    numbers: set = set()
    for part in parts:
        ends = part.split("-")
        if len(ends) > 2:
            raise ValueError(
                "weekdays must name days, or ranges of them such as "
                f"'Mon-Fri'; got {value!r}."
            )
        first, last = (_weekday(e, "weekdays") for e in (ends[0], ends[-1]))
        numbers.update((first + k) % 7 for k in range((last - first) % 7 + 1))
    return tuple(sorted(numbers))


def _month(value, name: str = "month") -> int:
    """A month by number (1 to 12) or name (``"November"``, ``"Nov"``)."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _MONTH_NUMBERS:
            return _MONTH_NUMBERS[key]
        raise ValueError(
            f"{name} must be a month, 1 to 12 or a name such as 'November'; "
            f"got {value!r}."
        )
    return _whole(value, name, 1, 12)


def _nth(value) -> int:
    """Which of a month's weekdays: 1 to 5, or -1 to -5 (counted from the
    month's end)."""
    if not _is_whole(value) or int(value) == 0 or not -5 <= int(value) <= 5:
        raise ValueError(
            "nth must be 1 to 5 (the first to the fifth) or -1 to -5 (the "
            f"last to the fifth from last); got {value!r}."
        )
    return int(value)


def _one_of(name: str, value, choices) -> None:
    if value not in choices:
        listed = ", ".join(repr(c) for c in choices)
        raise ValueError(f"{name} must be one of {listed}; got {value!r}.")


# ---------------------------------------------------------------- words


def _day_text(day: date) -> str:
    """A date in words: ``"5 Dec 2026"``."""
    return f"{day.day} {MONTHS[day.month - 1][:3]} {day.year}"


def _ordinal_day(day: int) -> str:
    """``1`` -> ``"1st"``, ``22`` -> ``"22nd"``, ``30`` -> ``"30th"``."""
    suffix = (
        "th"
        if 10 <= day % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return f"{day}{suffix}"


def _nth_text(nth: int) -> str:
    """``1`` -> ``"first"``, ``-1`` -> ``"last"``, ``-2`` ->
    ``"second from last"``."""
    if nth > 0:
        return _ORDINALS[nth]
    return "last" if nth == -1 else f"{_ORDINALS[-nth]} from last"


def _listed(words: List[str]) -> str:
    """``["a", "b", "c"]`` -> ``"a, b and c"``."""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def _every_nth(every: int, what: str) -> str:
    """``(1, "month")`` -> ``"every month"``; ``(3, "month")`` ->
    ``"every 3rd month"``."""
    return (
        f"every {what}"
        if every == 1
        else f"every {_ordinal_day(every)} {what}"
    )


# ---------------------------------------------------------------- results


def _plain(value) -> Any:
    """``value`` as plain data for JSON: dates as ISO text, results and
    rules as their ``to_dict()``, tuples as lists."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(getattr(value, "to_dict", None)) and not isinstance(
        value, type
    ):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value


class _ScheduleResult(_ResultMapping):
    """A schedule result: a read-only mapping of its fields, with
    ``to_dict()`` writing dates as ISO text."""

    def to_dict(self) -> dict:
        """The result as plain data, ready for ``json.dumps``: each field
        by name, dates as ISO text (``"2026-12-05"``), tuples as lists.

        Returns
        -------
        dict
            The fields, as plain data.
        """
        return {name: _plain(getattr(self, name)) for name in self}


@dataclass(frozen=True)
class RecurrenceNotes(_ScheduleResult):
    """What a ``Recurrence`` left out or moved in a period: returned by
    its ``notes(start, end)``.

    Attributes
    ----------
    start, end : datetime.date
        The period's first and last dates, both included.
    missing_months : tuple of str
        The months (``"2027-02"``) on the rule's cycle that have no such
        day, and so no date: a fifth Saturday, or the 31st with
        ``short_months="skip"``.
    out_of_season : tuple of datetime.date
        The rule's dates left out because they are outside its season.
    excluded : tuple of datetime.date
        The rule's dates left out because they are excluded
        (``moved="skip"``).
    moved : tuple of (datetime.date, datetime.date)
        The excluded dates that were moved, each with the date it moved
        to.
    text : tuple of str
        The same, in plain words, one line each.
    """

    start: date
    end: date
    missing_months: Tuple[str, ...]
    out_of_season: Tuple[date, ...]
    excluded: Tuple[date, ...]
    moved: Tuple[Tuple[date, date], ...]
    text: Tuple[str, ...]


# ---------------------------------------------------------------- season


@dataclass(frozen=True, init=False)
class Season:
    """The part of each year something applies in: some months (April to
    October), or from one day of the year to another (1 September to 30
    April, across the new year), both ends included.

    Parameters
    ----------
    months : sequence of int or str, optional
        The months, by number (1 to 12) or name (``"Apr"``).
    start, end : str, optional
        The first and last days of the season, as ``"MM-DD"``. Give
        ``months``, or ``start`` and ``end``.

    Examples
    --------
    >>> from repyability.schedule import Season
    >>> summer = Season(months=[10, 11, 12, 1, 2, 3])
    >>> summer.includes("2027-01-15"), summer.includes("2027-06-01")
    (True, False)
    >>> summer.describe()
    'October to March'
    >>> Season(start="12-01", end="02-28").includes("2027-02-28")
    True
    """

    months: Optional[Tuple[int, ...]]
    start: Optional[str]
    end: Optional[str]

    def __init__(self, *, months=None, start=None, end=None) -> None:
        if (months is None) == (start is None and end is None):
            raise ValueError(
                "A season is some months (months=[4, 5, 6]) or a first and "
                "last day of the year (start='09-01', end='04-30'), one of "
                "them."
            )
        if months is not None:
            if isinstance(months, (str, bytes)) or not hasattr(
                months, "__iter__"
            ):
                raise TypeError(
                    "months must be a list of months, such as [4, 5, 6]; "
                    f"got {months!r}."
                )
            numbers = tuple(sorted({_month(m, "months") for m in months}))
            if not numbers:
                raise ValueError("A season needs at least one month.")
            if len(numbers) == 12:
                raise ValueError(
                    "A season of all twelve months is the whole year: leave "
                    "the season out."
                )
            object.__setattr__(self, "months", numbers)
            object.__setattr__(self, "start", None)
            object.__setattr__(self, "end", None)
            return
        for name, value in (("start", start), ("end", end)):
            ok = isinstance(value, str) and len(value) == 5 and value[2] == "-"
            if ok:
                try:
                    date.fromisoformat(f"2028-{value}")  # 2028: 02-29 is a day
                except ValueError:
                    ok = False
            if not ok:
                raise ValueError(
                    f"{name} is a day of the year as MM-DD, such as '09-01'; "
                    f"got {value!r}."
                )
        if start == end:
            raise ValueError("A season's start and end are different days.")
        object.__setattr__(self, "months", None)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def includes(self, day) -> bool:
        """Whether a date is in the season (both ends included).

        Parameters
        ----------
        day : datetime.date or str
            The date.

        Returns
        -------
        bool
        """
        day = _date(day, "day")
        if self.months is not None:
            return day.month in self.months
        assert self.start is not None and self.end is not None
        md = day.isoformat()[5:]
        if self.start <= self.end:
            return self.start <= md <= self.end
        return md >= self.start or md <= self.end

    def describe(self) -> str:
        """The season in plain words: ``"April to October"``.

        Returns
        -------
        str
        """
        if self.months is None:
            assert self.start is not None and self.end is not None
            a = date.fromisoformat(f"2028-{self.start}")
            b = date.fromisoformat(f"2028-{self.end}")
            return (
                f"{a.day} {MONTHS[a.month - 1]} to {b.day} "
                f"{MONTHS[b.month - 1]}"
            )
        runs = _month_runs(self.months)
        words = [
            MONTHS[a - 1] if a == b else f"{MONTHS[a - 1]} to {MONTHS[b - 1]}"
            for a, b in runs
        ]
        return _listed(words)

    def to_dict(self) -> dict:
        """The season as plain data: ``{"months": [...]}`` or
        ``{"start": "MM-DD", "end": "MM-DD"}``.

        Returns
        -------
        dict
        """
        if self.months is not None:
            return {"months": list(self.months)}
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data) -> "Season":
        """A season from its ``to_dict()``.

        Parameters
        ----------
        data : dict
            ``{"months": [...]}`` or ``{"start": ..., "end": ...}``.

        Returns
        -------
        Season
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"A season is a dict; got {data!r}.")
        return cls(
            months=data.get("months"),
            start=data.get("start"),
            end=data.get("end"),
        )


def _month_runs(months: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """Months grouped into runs of consecutive months, a run allowed to
    wrap round the new year: ``(1, 2, 3, 10, 11, 12)`` -> ``[(10, 3)]``."""
    chosen = set(months)
    starts = [m for m in months if (m - 2) % 12 + 1 not in chosen]
    runs = []
    for first in starts:
        last = first
        while last % 12 + 1 in chosen:
            last = last % 12 + 1
        runs.append((first, last))
    return runs


# ---------------------------------------------------------------- rule


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time())


def _exclusions(value) -> Tuple[Tuple[date, date], ...]:
    """``excluding=`` as sorted ``(first, last)`` date ranges: each item a
    date (or ISO text) or a pair of them, both ends included."""
    if value is None:
        return ()
    if isinstance(value, (str, date)) or not hasattr(value, "__iter__"):
        raise TypeError(
            "excluding must be a list of dates or (first, last) pairs, such "
            f"as ['2026-12-25', ('2027-01-04', '2027-01-08')]; got {value!r}."
        )
    ranges = []
    for item in value:
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise ValueError(
                    "An excluded range is a pair, (first, last); got "
                    f"{item!r}."
                )
            first = _date(item[0], "excluding")
            last = _date(item[1], "excluding")
            if last < first:
                raise ValueError(
                    f"An excluded range ends ({last}) before it starts "
                    f"({first})."
                )
        else:
            first = last = _date(item, "excluding")
        ranges.append((first, last))
    return tuple(sorted(set(ranges)))


@dataclass(frozen=True, init=False, repr=False)
class Recurrence:
    """Which dates something falls on: every so many days, weeks, months
    or years; named weekdays; a day or weekday of the month; a date or
    weekday pattern in a named month each year; or given dates.

    Made with its named constructors (``Recurrence.every``, ``weekly``,
    ``monthly``, ``yearly`` and ``on``), each taking the same options:

    - ``season=``: a ``Season`` (or its dict); dates outside it don't
      count.
    - ``excluding=``: dates, or ``(first, last)`` ranges both ends
      included, that don't count (holidays, a shut).
    - ``moved=``: what an excluded date does: ``"skip"`` (the default),
      ``"before"`` (the nearest day before that is not excluded) or
      ``"after"`` (the nearest day after).
    - ``short_months=``: for a day of the month past the 28th (or a
      yearly 29 February), what a month without it does: ``"last"`` (the
      default) falls on its last day, ``"skip"`` has no date that month.

    Every period asked about includes both of its ends.

    Examples
    --------
    >>> from repyability.schedule import Recurrence
    >>> first_saturday = Recurrence.monthly(weekday="Saturday", nth=1)
    >>> [str(d) for d in first_saturday.dates("2026-11-14", "2027-03-31")]
    ['2026-12-05', '2027-01-02', '2027-02-06', '2027-03-06']
    >>> first_saturday.describe()
    'the first Saturday of every month'
    >>> overhaul = Recurrence.yearly(month=11, weekday="Tuesday", nth=1)
    >>> [str(d) for d in overhaul.dates("2026-11-14", "2029-12-31")]
    ['2027-11-02', '2028-11-07', '2029-11-06']
    """

    rule: str
    every_n: int
    unit: Optional[str]
    anchor: Optional[date]
    weekdays: Tuple[int, ...]
    nth: Optional[int]
    day: Optional[int]
    month: Optional[int]
    given: Tuple[date, ...]
    season: Optional[Season]
    excluding: Tuple[Tuple[date, date], ...]
    moved: str
    short_months: Optional[str]

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "Make a Recurrence with Recurrence.every(...), .weekly(...), "
            ".monthly(...), .yearly(...) or .on(...)."
        )

    # -- construction --------------------------------------------------

    @classmethod
    def _build(
        cls,
        rule: str,
        *,
        every_n: int = 1,
        unit: Optional[str] = None,
        anchor=None,
        weekdays: Tuple[int, ...] = (),
        nth: Optional[int] = None,
        day: Optional[int] = None,
        month: Optional[int] = None,
        given: Tuple[date, ...] = (),
        season=None,
        excluding=None,
        moved=None,
        short_months=None,
    ) -> "Recurrence":
        self = object.__new__(cls)
        anchor = None if anchor is None else _date(anchor, "anchor")
        if season is not None and not isinstance(season, Season):
            season = Season.from_dict(season)
        ranges = _exclusions(excluding)
        if moved is None:
            moved = "skip"
        else:
            _one_of("moved", moved, MOVES)
            if not ranges and moved != "skip":
                raise ValueError(
                    "moved says what an excluded date does: give the "
                    "excluded dates too (excluding=[...])."
                )
        day_based = day is not None or (
            rule == "every" and unit in ("months", "years")
        )
        if day_based:
            short_months = "last" if short_months is None else short_months
            _one_of("short_months", short_months, ("last", "skip"))
        elif short_months is not None:
            raise ValueError(
                "short_months goes with a day of the month (the 30th, or "
                "every so many months from an anchor): this rule has none."
            )
        values = dict(
            rule=rule,
            every_n=every_n,
            unit=unit,
            anchor=anchor,
            weekdays=weekdays,
            nth=nth,
            day=day,
            month=month,
            given=given,
            season=season,
            excluding=ranges,
            moved=moved,
            short_months=short_months,
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self._check_season()
        return self

    def _check_season(self) -> None:
        """Refuse a yearly rule whose month is never in its season."""
        season = self.season
        if season is None or season.months is None:
            return
        month = self.month
        if self.rule == "every" and self.unit == "years":
            assert self.anchor is not None
            month = self.anchor.month
        if month is not None and month not in season.months:
            raise ValueError(
                f"The rule falls in {MONTHS[month - 1]}, which is not in "
                f"its season ({season.describe()}): it would give no dates."
            )

    @classmethod
    def every(
        cls,
        n,
        unit,
        anchor,
        *,
        season=None,
        excluding=None,
        moved=None,
        short_months=None,
    ) -> "Recurrence":
        """Every ``n`` days, weeks, months or years from a date.

        Months and years keep the anchor's day of the month (31 January,
        28 February, 31 March: a shorter month's last day, or no date
        that month with ``short_months="skip"``), each counted from the
        anchor, not from the date before.

        Parameters
        ----------
        n : int
            How many units apart, at least 1.
        unit : {"days", "weeks", "months", "years"}
            The unit (the singular is taken too).
        anchor : datetime.date or str
            A date the rule falls on, its first: no date comes before it.
        season, excluding, moved, short_months
            The options every rule takes (see the class).

        Returns
        -------
        Recurrence

        Examples
        --------
        >>> from repyability.schedule import Recurrence
        >>> weekly = Recurrence.every(1, "week", "2026-07-15")
        >>> len(weekly.dates("2026-10-11", "2026-12-31"))
        12
        >>> monthly = Recurrence.every(1, "months", "2027-01-31")
        >>> [str(d) for d in monthly.dates("2027-01-01", "2027-03-31")]
        ['2027-01-31', '2027-02-28', '2027-03-31']
        """
        n = _whole(n, "n")
        if isinstance(unit, str) and unit + "s" in UNITS:
            unit = unit + "s"
        _one_of("unit", unit, UNITS)
        if anchor is None:
            raise TypeError(
                "Recurrence.every needs an anchor: a date it falls on, such "
                "as anchor='2026-07-15'."
            )
        return cls._build(
            "every",
            every_n=n,
            unit=unit,
            anchor=anchor,
            season=season,
            excluding=excluding,
            moved=moved,
            short_months=short_months,
        )

    @classmethod
    def weekly(
        cls,
        weekdays,
        *,
        every=1,
        anchor=None,
        season=None,
        excluding=None,
        moved=None,
    ) -> "Recurrence":
        """On named days of the week, every week or every ``every``
        weeks.

        Parameters
        ----------
        weekdays : str or sequence of str
            The days: ``"Saturday"``, ``"Mon-Fri"``, ``"Mon, Wed, Fri"``
            or ``["Mon", "Wed"]`` (names or their first three letters,
            any case).
        every : int, optional
            Every how many weeks, by default 1. Above 1, weeks are counted
            from the anchor's week (weeks start on Monday), so it needs an
            ``anchor``.
        anchor : datetime.date or str, optional
            The date the rule starts: no date comes before it.
        season, excluding, moved
            The options every rule takes (see the class).

        Returns
        -------
        Recurrence

        Examples
        --------
        >>> from repyability.schedule import Recurrence
        >>> Recurrence.weekly("Mon, Wed, Fri").describe()
        'every Monday, Wednesday and Friday'
        """
        every = _whole(every, "every")
        if every > 1 and anchor is None:
            raise ValueError(
                f"Every {every} weeks counts from a week: give anchor=, a "
                "date in the first week."
            )
        return cls._build(
            "weekly",
            every_n=every,
            weekdays=_weekdays(weekdays),
            anchor=anchor,
            season=season,
            excluding=excluding,
            moved=moved,
        )

    @classmethod
    def monthly(
        cls,
        *,
        day=None,
        weekday=None,
        nth=None,
        every=1,
        anchor=None,
        season=None,
        excluding=None,
        moved=None,
        short_months=None,
    ) -> "Recurrence":
        """A day of the month (the 30th), or the ``nth`` weekday of it
        (the first Saturday, the last Friday), every month or every
        ``every`` months.

        A fifth weekday (``nth=5`` or ``-5``) is in only some months: the
        others have no date, and ``notes`` lists them.

        Parameters
        ----------
        day : int, optional
            The day of the month, 1 to 31. A month too short for it falls
            on its last day, or is skipped (``short_months``).
        weekday : str, optional
            The day of the week, with ``nth``. Give ``day`` or
            ``weekday``.
        nth : int, optional
            Which of the month's weekdays: 1 to 5, or -1 (the last) to -5.
        every : int, optional
            Every how many months, by default 1. Above 1, months are
            counted from the anchor's month, so it needs an ``anchor``.
        anchor : datetime.date or str, optional
            The date the rule starts: no date comes before it.
        season, excluding, moved, short_months
            The options every rule takes (see the class).

        Returns
        -------
        Recurrence

        Examples
        --------
        >>> from repyability.schedule import Recurrence
        >>> thirtieth = Recurrence.monthly(day=30)
        >>> [str(d) for d in thirtieth.dates("2027-01-01", "2027-03-31")]
        ['2027-01-30', '2027-02-28', '2027-03-30']
        >>> Recurrence.monthly(weekday="Fri", nth=-1, every=3,
        ...                    anchor="2027-01-01").describe()
        'the last Friday of every 3rd month, from 1 Jan 2027'
        """
        every = _whole(every, "every")
        if every > 1 and anchor is None:
            raise ValueError(
                f"Every {every} months counts from a month: give anchor=, a "
                "date in the first month."
            )
        day, weekdays, nth = cls._day_or_weekday(day, weekday, nth)
        return cls._build(
            "monthly",
            every_n=every,
            day=day,
            weekdays=weekdays,
            nth=nth,
            anchor=anchor,
            season=season,
            excluding=excluding,
            moved=moved,
            short_months=short_months,
        )

    @classmethod
    def yearly(
        cls,
        *,
        month,
        day=None,
        weekday=None,
        nth=None,
        every=1,
        anchor=None,
        season=None,
        excluding=None,
        moved=None,
        short_months=None,
    ) -> "Recurrence":
        """A date (1 July) or a weekday pattern (the first Tuesday) in a
        named month, every year or every ``every`` years.

        Parameters
        ----------
        month : int or str
            The month, 1 to 12 or a name (``"November"``, ``"Nov"``).
        day : int, optional
            The day of that month. 29 February falls on the 28th in other
            years, or is skipped (``short_months``).
        weekday : str, optional
            The day of the week, with ``nth``. Give ``day`` or
            ``weekday``.
        nth : int, optional
            Which of the month's weekdays: 1 to 5, or -1 (the last) to -5.
        every : int, optional
            Every how many years, by default 1. Above 1, years are counted
            from the anchor's year, so it needs an ``anchor``.
        anchor : datetime.date or str, optional
            The date the rule starts: no date comes before it.
        season, excluding, moved, short_months
            The options every rule takes (see the class).

        Returns
        -------
        Recurrence

        Examples
        --------
        >>> from repyability.schedule import Recurrence
        >>> Recurrence.yearly(month=11, weekday="Tuesday", nth=1).describe()
        'the first Tuesday of November, every year'
        >>> Recurrence.yearly(month="Jul", day=1).describe()
        '1 July, every year'
        """
        every = _whole(every, "every")
        if every > 1 and anchor is None:
            raise ValueError(
                f"Every {every} years counts from a year: give anchor=, a "
                "date in the first year."
            )
        month = _month(month)
        day, weekdays, nth = cls._day_or_weekday(day, weekday, nth)
        if day is not None and day > _calendar.monthrange(2028, month)[1]:
            raise ValueError(f"{MONTHS[month - 1]} has no day {day}.")
        return cls._build(
            "yearly",
            every_n=every,
            month=month,
            day=day,
            weekdays=weekdays,
            nth=nth,
            anchor=anchor,
            season=season,
            excluding=excluding,
            moved=moved,
            short_months=short_months,
        )

    @staticmethod
    def _day_or_weekday(day, weekday, nth):
        if (day is None) == (weekday is None):
            raise ValueError(
                "Give a day of the month (day=30) or a weekday with nth "
                "(weekday='Saturday', nth=1), one of them."
            )
        if day is not None:
            if nth is not None:
                raise ValueError(
                    "nth goes with a weekday (the first Saturday), not with "
                    "a day of the month."
                )
            return _whole(day, "day", 1, 31), (), None
        if nth is None:
            raise ValueError(
                "A weekday of the month needs nth, which of them: 1 for "
                "the first, -1 for the last."
            )
        return None, (_weekday(weekday),), _nth(nth)

    @classmethod
    def on(
        cls, dates, *, season=None, excluding=None, moved=None
    ) -> "Recurrence":
        """Given dates (turnarounds booked years ahead, say).

        Parameters
        ----------
        dates : sequence of datetime.date or str
            The dates, at least one, in any order.
        season, excluding, moved
            The options every rule takes (see the class).

        Returns
        -------
        Recurrence

        Examples
        --------
        >>> from repyability.schedule import Recurrence
        >>> Recurrence.on(["2028-09-09", "2027-03-03"]).describe()
        'on 3 Mar 2027 and 9 Sep 2028'
        """
        if isinstance(dates, (str, date)) or not hasattr(dates, "__iter__"):
            raise TypeError(
                "dates must be a list of dates, such as ['2027-03-03']; got "
                f"{dates!r}."
            )
        given = tuple(sorted({_date(d, "dates") for d in dates}))
        if not given:
            raise ValueError("Recurrence.on needs at least one date.")
        return cls._build(
            "on",
            given=given,
            season=season,
            excluding=excluding,
            moved=moved,
        )

    def reanchored(self, anchor) -> "Recurrence":
        """The same rule from another anchor (for an interval counted from
        when a task was last done).

        Parameters
        ----------
        anchor : datetime.date or str
            The new anchor.

        Returns
        -------
        Recurrence
        """
        if self.rule == "on":
            raise ValueError("Given dates have no anchor to move.")
        copy = object.__new__(Recurrence)
        for f in dataclasses.fields(self):
            object.__setattr__(copy, f.name, getattr(self, f.name))
        object.__setattr__(copy, "anchor", _date(anchor, "anchor"))
        copy._check_season()
        return copy

    # -- dates ---------------------------------------------------------

    def _rrule(self, dtstart: date) -> rrule:
        """The rule's raw dates (before season and exclusions) as a
        dateutil rule from ``dtstart``."""
        start = _midnight(dtstart)
        if self.rule == "every":
            assert self.unit is not None and self.anchor is not None
            if self.unit in ("days", "weeks"):
                step = self.every_n * (7 if self.unit == "weeks" else 1)
                return rrule(DAILY, dtstart=start, interval=step)
            day = self.anchor.day
            if self.unit == "months":
                return self._by_day(MONTHLY, start, self.every_n, day, None)
            return self._by_day(
                YEARLY, start, self.every_n, day, self.anchor.month
            )
        if self.rule == "weekly":
            days = [_RRULE_DAYS[d] for d in self.weekdays]
            return rrule(
                WEEKLY,
                dtstart=start,
                interval=self.every_n,
                byweekday=days,
                wkst=MO,
            )
        freq = MONTHLY if self.rule == "monthly" else YEARLY
        if self.day is not None:
            return self._by_day(
                freq, start, self.every_n, self.day, self.month
            )
        assert self.nth is not None
        return rrule(
            freq,
            dtstart=start,
            interval=self.every_n,
            bymonth=self.month,
            byweekday=_RRULE_DAYS[self.weekdays[0]](self.nth),
        )

    def _by_day(self, freq, start, interval, day, month) -> rrule:
        """A day of the month: the day itself, or, when a short month
        lacks it and short months fall on their last day, the latest of the
        28th up to it (iCalendar's ``BYSETPOS=-1``)."""
        if day <= 28 or self.short_months == "skip":
            return rrule(
                freq,
                dtstart=start,
                interval=interval,
                bymonth=month,
                bymonthday=day,
            )
        return rrule(
            freq,
            dtstart=start,
            interval=interval,
            bymonth=month,
            bymonthday=tuple(range(28, day + 1)),
            bysetpos=-1,
        )

    def _pad(self) -> timedelta:
        """How far a move can carry a date: the longest run of excluded
        days, and one."""
        if self.moved == "skip" or not self.excluding:
            return timedelta(0)
        longest, run_start, run_end = 0, None, None
        for first, last in self.excluding:
            if run_end is not None and first <= run_end + timedelta(days=1):
                run_end = max(run_end, last)
            else:
                run_start, run_end = first, last
            assert run_start is not None
            longest = max(longest, (run_end - run_start).days + 1)
        return timedelta(days=longest + 1)

    def _excluded(self, day: date) -> bool:
        return any(first <= day <= last for first, last in self.excluding)

    def _raw(self, lo: date) -> Iterator[date]:
        """The rule's raw dates from ``lo`` on, in order."""
        if self.rule == "on":
            yield from (d for d in self.given if d >= lo)
            return
        if self.anchor is not None:
            lo = max(lo, self.anchor)
            rule = self._rrule(self.anchor)
        else:
            rule = self._rrule(lo)
        for moment in rule.xafter(_midnight(lo), inc=True):
            yield moment.date()

    def _walk(self, lo: date) -> Iterator[Tuple[date, Optional[date], str]]:
        """Each raw date from ``lo`` on with what became of it: ``(raw,
        final, why)``, ``final`` None when it does not count, ``why`` one
        of ``"kept"``, ``"season"``, ``"excluded"`` or ``"moved"``. The
        finals never go backwards."""
        step = timedelta(days=-1 if self.moved == "before" else 1)
        for raw in self._raw(lo):
            if self.season is not None and not self.season.includes(raw):
                yield raw, None, "season"
            elif not self._excluded(raw):
                yield raw, raw, "kept"
            elif self.moved == "skip":
                yield raw, None, "excluded"
            else:
                moved = raw
                while self._excluded(moved):
                    moved += step
                yield raw, moved, "moved"

    def _finals(self, start: date) -> Iterator[date]:
        """The rule's dates from ``start`` on, in order, each once."""
        pad = self._pad()
        lo = date.min if start - date.min < pad else start - pad
        last = None
        for _, final, _ in self._walk(lo):
            if final is None or final < start or final == last:
                continue
            last = final
            yield final

    def dates(self, start, end) -> Tuple[date, ...]:
        """The rule's dates from ``start`` to ``end``, **both included**.

        Parameters
        ----------
        start, end : datetime.date or str
            The first and last dates of the period.

        Returns
        -------
        tuple of datetime.date
            In order.
        """
        start, end = _period(start, end)
        out = []
        for day in self._finals(start):
            if day > end:
                break
            out.append(day)
        return tuple(out)

    def on_or_after(self, day) -> Optional[date]:
        """The rule's first date on or after ``day``.

        Parameters
        ----------
        day : datetime.date or str
            The date.

        Returns
        -------
        datetime.date or None
            None when the rule has no date from then on.
        """
        return next(self._finals(_date(day, "day")), None)

    def after(self, day) -> Optional[date]:
        """The rule's first date strictly after ``day``.

        Parameters
        ----------
        day : datetime.date or str
            The date.

        Returns
        -------
        datetime.date or None
            None when the rule has no date after it.
        """
        day = _date(day, "day")
        if day == date.max:
            return None
        return self.on_or_after(day + timedelta(days=1))

    def includes(self, day) -> bool:
        """Whether the rule falls on ``day``.

        Parameters
        ----------
        day : datetime.date or str
            The date.

        Returns
        -------
        bool
        """
        day = _date(day, "day")
        return self.on_or_after(day) == day

    def number(self, day) -> int:
        """How many of the rule's dates there are up to ``day``
        (included), from its anchor: a visit's number, the first being 1
        (0 before it), for work done "every third visit".

        Parameters
        ----------
        day : datetime.date or str
            The date.

        Returns
        -------
        int
        """
        day = _date(day, "day")
        if self.rule != "on" and self.anchor is None:
            raise ValueError(
                "number counts the dates from the rule's anchor, which this "
                "rule has not: give anchor=."
            )
        count = 0
        for final in self._finals(date.min):
            if final > day:
                break
            count += 1
        return count

    def notes(self, start, end) -> RecurrenceNotes:
        """What the rule left out or moved from ``start`` to ``end``
        (both included): months with no such day, dates out of season,
        excluded dates, and moved ones.

        Parameters
        ----------
        start, end : datetime.date or str
            The first and last dates of the period.

        Returns
        -------
        RecurrenceNotes
        """
        start, end = _period(start, end)
        season, excluded, moved = [], [], []
        for raw, final, why in self._walk(start):
            if raw > end:
                break
            if why == "season":
                season.append(raw)
            elif why == "excluded":
                excluded.append(raw)
            elif why == "moved":
                assert final is not None
                moved.append((raw, final))
        missing = self._missing_months(start, end)
        text = []
        if missing:
            names = [f"{MONTHS[int(m[5:]) - 1][:3]} {m[:4]}" for m in missing]
            what = self._missing_what()
            plural = "s" if len(missing) > 1 else ""
            verb = "have" if len(missing) > 1 else "has"
            text.append(
                f"{len(missing)} month{plural} {verb} no {what} and "
                f"{'are' if len(missing) > 1 else 'is'} skipped: "
                f"{_listed(names)}."
            )
        if season:
            text.append(
                f"{len(season)} date{'s' if len(season) > 1 else ''} out of "
                f"season ({self.season.describe() if self.season else ''}) "
                "not counted."
            )
        if excluded:
            text.append(
                "Excluded, not counted: "
                f"{_listed([_day_text(d) for d in excluded])}."
            )
        for raw, final in moved:
            text.append(
                f"{_day_text(raw)} is excluded: moved to {_day_text(final)}."
            )
        return RecurrenceNotes(
            start,
            end,
            tuple(missing),
            tuple(season),
            tuple(excluded),
            tuple(moved),
            tuple(text),
        )

    def _missing_what(self) -> str:
        if self.nth is not None:
            return f"{_nth_text(self.nth)} {WEEKDAYS[self.weekdays[0]]}"
        anchor = self.anchor or date.min
        day = self.day if self.day is not None else anchor.day
        if self.rule == "yearly" or (
            self.rule == "every" and self.unit == "years"
        ):
            return f"{day} {MONTHS[(self.month or anchor.month) - 1]}"
        return _ordinal_day(day)

    def _missing_months(self, start: date, end: date) -> List[str]:
        """The months on the rule's cycle, overlapping the period (and in
        a season of months), that have no such day."""
        if self.rule in ("weekly", "on"):
            return []
        if self.rule == "every" and self.unit in ("days", "weeks"):
            return []
        fifth = self.nth is not None and abs(self.nth) == 5
        day = self.day
        if self.rule == "every":
            assert self.anchor is not None
            day = self.anchor.day
        skips = day is not None and day > 28 and self.short_months == "skip"
        if not (fifth or skips):
            return []
        yearly = self.rule == "yearly" or (
            self.rule == "every" and self.unit == "years"
        )
        month_of_year = self.month
        if self.rule == "every" and yearly:
            assert self.anchor is not None
            month_of_year = self.anchor.month
        step = self.every_n * (12 if yearly else 1)
        origin = self.anchor
        out = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            index = y * 12 + m - 1
            on_cycle = True
            if origin is not None:
                since = index - (origin.year * 12 + origin.month - 1)
                on_cycle = since >= 0 and since % step == 0
                if yearly and month_of_year is not None:
                    on_cycle = on_cycle or False
            if yearly and month_of_year is not None and m != month_of_year:
                on_cycle = False
            in_season = (
                self.season is None
                or self.season.months is None
                or m in self.season.months
            )
            if on_cycle and in_season:
                length = _calendar.monthrange(y, m)[1]
                if fifth:
                    assert self.weekdays
                    first_weekday = date(y, m, 1).weekday()
                    offset = (self.weekdays[0] - first_weekday) % 7
                    exists = offset + 28 < length
                else:
                    assert day is not None
                    exists = day <= length
                if not exists:
                    out.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        return out

    # -- words ---------------------------------------------------------

    def describe(self) -> str:
        """The rule in plain words: ``"the first Tuesday of November,
        every year"``.

        Returns
        -------
        str
        """
        text = self._describe_rule()
        if self.season is not None:
            text += f", in {self.season.describe()} only"
        if self.excluding:
            if len(self.excluding) <= 3:
                items = [
                    (
                        _day_text(a)
                        if a == b
                        else f"{_day_text(a)} to {_day_text(b)}"
                    )
                    for a, b in self.excluding
                ]
                except_text = _listed(items)
            else:
                except_text = f"{len(self.excluding)} excluded dates or ranges"
            how = {
                "skip": "skipped",
                "before": "moved to the day before",
                "after": "moved to the day after",
            }[self.moved]
            text += f", except {except_text} ({how})"
        return text

    def _describe_rule(self) -> str:
        since = (
            "" if self.anchor is None else f", from {_day_text(self.anchor)}"
        )
        if self.rule == "on":
            if len(self.given) <= 4:
                return "on " + _listed([_day_text(d) for d in self.given])
            return (
                f"on {len(self.given)} given dates, from "
                f"{_day_text(self.given[0])} to {_day_text(self.given[-1])}"
            )
        if self.rule == "every":
            assert self.unit is not None and self.anchor is not None
            single = self.unit[:-1]
            head = (
                f"every {single}"
                if self.every_n == 1
                else f"every {self.every_n} {self.unit}"
            )
            text = f"{head} from {_day_text(self.anchor)}"
            if self.unit in ("months", "years") and self.anchor.day > 28:
                text += (
                    " (a shorter month's last day)"
                    if self.short_months == "last"
                    else " (skipping months too short)"
                )
            elif self.unit == "years" and (
                self.anchor.month,
                self.anchor.day,
            ) == (2, 29):
                text += ""
            return text
        if self.rule == "weekly":
            names = [WEEKDAYS[d] for d in self.weekdays]
            if self.weekdays == (0, 1, 2, 3, 4):
                days = "weekday, Monday to Friday"
            elif len(self.weekdays) == 7:
                days = "day"
            else:
                days = _listed(names)
            if self.every_n == 1:
                return f"every {days}{since}"
            assert self.anchor is not None
            return (
                f"every {self.every_n} weeks on {_listed(names)}, counted "
                f"from the week of {_day_text(self.anchor)}"
            )
        if self.day is not None:
            what = _ordinal_day(self.day)
        else:
            assert self.nth is not None
            what = f"{_nth_text(self.nth)} {WEEKDAYS[self.weekdays[0]]}"
        if self.rule == "monthly":
            text = f"the {what} of {_every_nth(self.every_n, 'month')}"
            if self.day is not None and self.day > 28:
                text += (
                    " (a shorter month's last day)"
                    if self.short_months == "last"
                    else " (skipping months too short)"
                )
            elif self.nth is not None and abs(self.nth) == 5:
                text += " (months without one are skipped)"
            return text + since
        assert self.month is not None
        month = MONTHS[self.month - 1]
        if self.day is not None:
            text = f"{self.day} {month}, {_every_nth(self.every_n, 'year')}"
            if (self.month, self.day) == (2, 29):
                text += (
                    " (28 February in other years)"
                    if self.short_months == "last"
                    else " (leap years only)"
                )
        else:
            text = f"the {what} of {month}, {_every_nth(self.every_n, 'year')}"
            if self.nth is not None and abs(self.nth) == 5:
                text += " (years without one are skipped)"
        return text + since

    def __repr__(self) -> str:
        return f"<Recurrence: {self.describe()}>"

    # -- saving --------------------------------------------------------

    def to_dict(self) -> dict:
        """The rule as plain data, for JSON: a ``"kind"`` and a
        ``"version"``, the rule's own fields (dates as ISO text, weekdays
        and months by name), and the options that differ from their
        defaults.

        Returns
        -------
        dict
        """
        out: Dict[str, Any] = {
            "kind": "recurrence",
            "version": VERSION,
            "rule": self.rule,
        }
        if self.rule == "on":
            out["dates"] = [d.isoformat() for d in self.given]
        else:
            out["every"] = self.every_n
        if self.unit is not None:
            out["unit"] = self.unit
        if self.anchor is not None:
            out["anchor"] = self.anchor.isoformat()
        if self.rule == "weekly":
            out["weekdays"] = [WEEKDAYS[d] for d in self.weekdays]
        elif self.weekdays:
            out["weekday"] = WEEKDAYS[self.weekdays[0]]
        for name in ("nth", "day"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.month is not None:
            out["month"] = MONTHS[self.month - 1]
        if self.season is not None:
            out["season"] = self.season.to_dict()
        if self.excluding:
            out["excluding"] = [
                a.isoformat() if a == b else [a.isoformat(), b.isoformat()]
                for a, b in self.excluding
            ]
        if self.moved != "skip":
            out["moved"] = self.moved
        if self.short_months not in (None, "last"):
            out["short_months"] = self.short_months
        return out

    @classmethod
    def from_dict(cls, data) -> "Recurrence":
        """A rule from its ``to_dict()``, or from the shapes stored by
        earlier tools: a work package's cadence (``every``, ``unit``,
        ``anchor`` and an optional ``pattern`` of ``weekday`` with ``nth``,
        or ``day``) and a task's interval (``interval``, ``unit``,
        ``start``).

        Parameters
        ----------
        data : dict
            The saved rule.

        Returns
        -------
        Recurrence
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"A saved rule is a dict; got {data!r}.")
        if data.get("kind") == "recurrence":
            return cls._from_own(data)
        if "kind" in data:
            raise ValueError(
                f"This is a saved {data['kind']!r}, not a recurrence rule."
            )
        if "interval" in data:
            return cls._from_task(data)
        if "unit" in data and "anchor" in data:
            return cls._from_cadence(data)
        raise ValueError(
            "This dict is not a saved rule: it has no 'kind', no package "
            "cadence (every, unit, anchor) and no task interval (interval, "
            f"unit, start); got keys {sorted(data)}."
        )

    @classmethod
    def _from_own(cls, data) -> "Recurrence":
        version = data.get("version", VERSION)
        if not _is_whole(version) or version > VERSION:
            raise ValueError(
                "This rule was saved by a later version (format "
                f"{version!r}); "
                "upgrade RePyability to read it."
            )
        options = {
            "season": data.get("season"),
            "excluding": [
                tuple(x) if isinstance(x, list) else x
                for x in data.get("excluding", [])
            ]
            or None,
            "moved": data.get("moved"),
        }
        rule = data.get("rule")
        if rule == "on":
            return cls.on(data["dates"], **options)
        short = (
            {"short_months": data["short_months"]}
            if "short_months" in data
            else {}
        )
        if rule == "every":
            return cls.every(
                data["every"], data["unit"], data["anchor"], **options, **short
            )
        if rule == "weekly":
            return cls.weekly(
                data["weekdays"],
                every=data.get("every", 1),
                anchor=data.get("anchor"),
                **options,
            )
        if rule == "monthly":
            return cls.monthly(
                day=data.get("day"),
                weekday=data.get("weekday"),
                nth=data.get("nth"),
                every=data.get("every", 1),
                anchor=data.get("anchor"),
                **options,
                **short,
            )
        if rule == "yearly":
            return cls.yearly(
                month=data["month"],
                day=data.get("day"),
                weekday=data.get("weekday"),
                nth=data.get("nth"),
                every=data.get("every", 1),
                anchor=data.get("anchor"),
                **options,
                **short,
            )
        raise ValueError(
            "A saved rule's 'rule' is every, weekly, monthly, yearly or on; "
            f"got {rule!r}."
        )

    @classmethod
    def _from_cadence(cls, data) -> "Recurrence":
        """A work package's cadence: every n units from an anchor, or a
        pattern inside that."""
        every, unit, anchor = (
            data.get("every", 1),
            data["unit"],
            data["anchor"],
        )
        pattern = data.get("pattern") or {}
        season = data.get("season") or None
        weekday, nth, day = (
            pattern.get("weekday"),
            pattern.get("nth"),
            pattern.get("day"),
        )
        if weekday is not None and unit == "weeks":
            return cls.weekly(
                weekday, every=every, anchor=anchor, season=season
            )
        if (weekday is not None or day is not None) and unit == "months":
            return cls.monthly(
                weekday=weekday,
                nth=nth,
                day=day,
                every=every,
                anchor=anchor,
                season=season,
            )
        if weekday is not None or day is not None:
            raise ValueError(
                "A cadence's weekday goes with weeks or months, and its day "
                f"with months; got a pattern {dict(pattern)!r} on {unit!r}."
            )
        return cls.every(every, unit, anchor, season=season)

    @classmethod
    def _from_task(cls, data) -> "Recurrence":
        """A task's interval on the calendar (``interval``, ``unit``,
        ``start``)."""
        interval, unit = data["interval"], data.get("unit") or "days"
        if unit not in UNITS:
            raise ValueError(
                f"A task every {interval} {unit} goes by use, not by the "
                "calendar: it is not a recurrence rule."
            )
        if not _is_whole(interval):
            raise ValueError(
                f"A rule counts whole {unit}; this task is every "
                f"{interval!r} {unit}: give it in a smaller unit."
            )
        if data.get("start") is None:
            raise ValueError(
                "A task's interval needs its start date ('start') to be a "
                "rule."
            )
        return cls.every(interval, unit, data["start"])

    def to_rrule(self, start=None) -> str:
        """The rule as iCalendar (RFC 5545) text: a ``DTSTART`` line, the
        ``RRULE`` (or ``RDATE``) and any ``EXDATE`` lines, which
        python-dateutil's ``rrulestr`` reads back to the same dates.

        Parameters
        ----------
        start : datetime.date or str, optional
            The ``DTSTART`` for a rule with no anchor (the first date it
            may fall on). A rule with an anchor starts there.

        Returns
        -------
        str

        Notes
        -----
        iCalendar cannot say a season of days of the year, nor move an
        excluded date: those rules are refused.
        """
        if self.moved != "skip":
            raise ValueError(
                "iCalendar cannot move an excluded date: only rules whose "
                "excluded dates are skipped can be written as an RRULE."
            )
        season = self.season
        if season is not None and season.months is None:
            raise ValueError(
                "iCalendar cannot say a season of days of the year: only a "
                "season of months can be written as an RRULE."
            )
        if self.rule == "on":
            first = self.given[0]
            lines = [
                f"DTSTART:{_midnight(first):%Y%m%dT%H%M%S}",
                "RDATE:"
                + ",".join(
                    f"{_midnight(d):%Y%m%dT%H%M%S}" for d in self.given
                ),
            ]
            # dateutil's set takes DTSTART as a date too: drop it from
            # the dates unless it is in season and not excluded.
            if not self._counts(first):
                lines.append(f"EXDATE:{_midnight(first):%Y%m%dT%H%M%S}")
            lines.extend(self._exdates(self.given[0], self.given[-1]))
            return "\n".join(lines)
        if self.anchor is not None:
            dtstart = self.anchor
        elif start is not None:
            dtstart = _date(start, "start")
        else:
            raise ValueError(
                "A rule with no anchor needs start=, the date its RRULE "
                "starts from."
            )
        rule = self._rrule(dtstart)
        if season is not None:
            assert season.months is not None
            rule = rule.replace(bymonth=season.months)
        text = str(rule)
        exdates = self._exdates(dtstart, None)
        return "\n".join([text] + exdates)

    def _counts(self, day: date) -> bool:
        in_season = self.season is None or self.season.includes(day)
        return in_season and not self._excluded(day)

    def _exdates(self, lo: date, hi: Optional[date]) -> List[str]:
        """EXDATE lines for the excluded days from ``lo`` (to ``hi``)."""
        days = []
        for first, last in self.excluding:
            if last < lo or (hi is not None and first > hi):
                continue
            day = max(first, lo)
            while day <= last and (hi is None or day <= hi):
                days.append(_midnight(day))
                day += timedelta(days=1)
        if not days:
            return []
        return ["EXDATE:" + ",".join(f"{d:%Y%m%dT%H%M%S}" for d in days)]


def _read_rrule(text: str, start, end) -> Tuple[date, ...]:
    """The dates of iCalendar text from ``start`` to ``end``, both
    included (used by the tests to read ``to_rrule`` back)."""
    first, last = _period(start, end)
    rules = rrulestr(text, forceset=True)
    return tuple(
        m.date()
        for m in rules.between(_midnight(first), _midnight(last), inc=True)
    )
