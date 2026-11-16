"""A site's shutdowns on real dates, as one record (UC-43 of this
package's ``README.md``): each kind of shut (a weekly stop, the annual
shut, one-off turnarounds, a service every 15,000 km), what replacing a
part then costs and how long the shut lasts; the price of a planned
replacement on any other day; and the site's time zone.

The record says *when* and *how much*. How the shuts land on a part's own
clock (calendar time, or a meter through an operating plan) is worked out
here (``_Layout``), and so is whether they truly repeat in that clock,
which decides how a policy can be worked out (``_decide``): exactly, by
sampling a long cycle, averaged over the shuts' usual gaps, or over a
horizon at a discount rate. Nothing is chosen silently: where the maths
does not settle the method, the caller is asked for one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from repyability.maintenance._calendar import ANYTIME, _common_multiple
from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: Hours in a plain period's units: a month is a twelfth of a year of
#: 365.25 days. A plain period (``every="30 days"``) is a fixed length of
#: real time, not a calendar rule.
_PERIOD_HOURS = {
    "minutes": 1.0 / 60.0,
    "hours": 1.0,
    "days": 24.0,
    "weeks": 168.0,
    "months": 24.0 * 365.25 / 12.0,
    "years": 24.0 * 365.25,
}
_PERIOD_WORDS = {
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days",
    "w": "weeks",
    "wk": "weeks",
    "wks": "weeks",
    "week": "weeks",
    "weeks": "weeks",
    "month": "months",
    "months": "months",
    "y": "years",
    "yr": "years",
    "yrs": "years",
    "year": "years",
    "years": "years",
}
_QUANTITY = re.compile(
    r"^\s*([0-9][0-9,_]*(?:\.[0-9]+)?)\s*([A-Za-z][\w ]*)?\s*$"
)

#: The years ahead over which shuts are laid on the part's clock to see
#: whether they repeat, and over which a shut's usual gaps are taken for
#: the averaged route: 28 years, after which weekdays fall on the same
#: dates again (between the century years).
SPAN_YEARS = 28
#: Two model times are the same within this share of the larger.
_SAME = 1e-9
#: Work a laptop does in about this many seconds still counts as a short
#: cycle, worked out exactly without being asked.
SHORT_SECONDS = 60.0
#: A shut kind's gaps are kept as at most this many distinct lengths for
#: the averaged formula (more are grouped into this many bins).
_MAX_GAPS = 64

ROUTES = ("exact", "exact_long_cycle", "sampling", "averaged", "horizon")
METHODS = (None, "exact", "sampling", "averaged")
FAILURE_TIMINGS = ("half_step", "exact")


class SlowSolverWarning(UserWarning):
    """A policy worked out by quadrature (``failure_timing="exact"``) will
    take a while: the warning says about how long, and that
    ``failure_timing="half_step"`` gives the same to about 1e-7 in far
    less."""


# ---------------------------------------------------------------- words


def _money(value: float) -> str:
    """A price in plain words: ``$3``, ``$0.50``, ``$9,000``."""
    if float(value).is_integer():
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def _number(value: float) -> str:
    """A figure to about 4 significant figures, with thousands marked."""
    if not math.isfinite(value):
        return "never" if value > 0 else str(value)
    if value == 0.0:
        return "0"
    digits = max(0, 3 - int(math.floor(math.log10(abs(value)))))
    text = f"{value:,.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _day_text(day: date) -> str:
    return f"{day.day} {day:%b} {day.year}"


def _listed(words: List[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


# --------------------------------------------------------------- parsing


def _period_hours(value, name: str) -> Tuple[float, str]:
    """A plain period written as text (``"30 days"``, ``"8h"``, ``"6
    months"``), as hours, and the words for it."""
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be text such as '30 days' or '8h'; got {value!r}."
        )
    found = _QUANTITY.match(value)
    unit = None
    if found and found.group(2):
        unit = _PERIOD_WORDS.get(found.group(2).strip().lower())
    if not found or unit is None:
        raise ValueError(
            f"{name} must be an amount and a unit of time, such as '30 days', "
            f"'8h', '6 months' or '2 years'; got {value!r}."
        )
    amount = float(found.group(1).replace(",", "").replace("_", ""))
    if not amount > 0.0:
        raise ValueError(f"{name} must be more than 0; got {value!r}.")
    words = f"{_number(amount)} {unit if amount != 1 else unit[:-1]}"
    return amount * _PERIOD_HOURS[unit], words


def _duration(value) -> Optional[float]:
    """A shut's duration in hours: text (``"8h"``, ``"3d"``) or a number
    of hours; None if not given."""
    if value is None:
        return None
    if checks.is_number(value):
        if not 0.0 < float(value) < math.inf:
            raise ValueError(
                f"A shut's duration must be more than 0; got {value!r}."
            )
        return float(value)
    return _period_hours(value, "duration")[0]


def _iso_date(value, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(
                f"{name} must be a date written as YYYY-MM-DD, such as "
                f"'2026-11-16'; got {value!r}."
            ) from None
    raise TypeError(
        f"{name} must be a date (a datetime.date, or text such as "
        f"'2026-11-16'); got {value!r}."
    )


def quantity(value, unit: Optional[str], name: str) -> float:
    """A number, or text giving a number in the part's unit (``"3,100
    h"``, ``"208,000 km"``), as the number."""
    if checks.is_number(value):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a number, or text such as '3,100 h'; got "
            f"{value!r}."
        )
    found = _QUANTITY.match(value)
    if not found:
        raise ValueError(
            f"{name} must be a number, with the part's unit if you like "
            f"('3,100 h'); got {value!r}."
        )
    amount = float(found.group(1).replace(",", "").replace("_", ""))
    given = found.group(2)
    if given:
        from repyability.schedule._meters import time_unit

        word = given.strip()
        named = time_unit(word)
        if unit is None:
            raise ValueError(
                f"{name} {value!r} names a unit, but the part's unit is not "
                "known: give unit= too."
            )
        same = (named is not None and named == time_unit(unit)) or (
            word.lower() == str(unit).lower()
        )
        if not same:
            raise ValueError(
                f"{name} {value!r} is in {word!r}, but the part's life is in "
                f"{unit!r}: give it in {unit!r}."
            )
    return amount


def _gap(value, name: str) -> Tuple[int, str]:
    """A gap such as ``"4 years"`` as a whole number and a unit of
    [`Recurrence.every`][repyability.schedule.Recurrence.every]."""
    from repyability.schedule._recurrence import UNITS as EVERY_UNITS

    words = value.strip().split() if isinstance(value, str) else []
    unit = words[1].lower() if len(words) == 2 else ""
    if unit + "s" in EVERY_UNITS:
        unit += "s"
    if (
        len(words) != 2
        or not words[0].isdigit()
        or int(words[0]) < 1
        or unit not in EVERY_UNITS
    ):
        raise ValueError(
            f"{name} must be a whole number of days, weeks, months or years, "
            f"such as '4 years'; got {value!r}."
        )
    return int(words[0]), unit


def _gap_words(n: int, unit: str) -> str:
    return f"{n} {unit[:-1] if n == 1 else unit}"


# ---------------------------------------------------------------- records


@dataclass(frozen=True)
class Shutdown:
    """One kind of shut at a site: when it happens, what replacing a part
    then costs, and how long it lasts.

    When is one of: a ``rule`` (a
    [`Recurrence`][repyability.schedule.Recurrence]: every Saturday, the
    first Tuesday of November), ``dates`` (fixed dates, such as
    turnarounds; with ``then_every`` they recur at that gap after the
    last), or ``every``, a
    plain period (``"30 days"``, ``"2 years"``: a fixed length of real
    time, a month a twelfth of 365.25 days; a number is that many of the
    part's own units) or a use (``Usage(15_000, "km")``: at each whole
    15,000 km on the meter), with ``also_every`` a calendar limit,
    whichever comes first (then counted again from each shut).

    Parameters
    ----------
    name : str
        The shut's name, as results list it.
    rule : Recurrence, optional
        When it happens, on dates.
    dates : list of date or str, optional
        Its fixed dates, instead of a rule. A policy uses them as dated
        (never averaged): see
        [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
    then_every : str, optional
        With ``dates``: the gap at which the kind recurs after its last
        date (``"4 years"``, ``"18 months"``, ``"6 weeks"``, ``"90
        days"``; whole numbers), counted from that date as
        [`Recurrence.every`][repyability.schedule.Recurrence.every]
        counts. The known dates are used exactly; after them the kind is
        one of the recurring shuts, and results say so.
    every : str, float or Usage, optional
        A plain period or a use, instead of a rule or dates.
    also_every : str, optional
        With ``every=Usage(...)``: a plain period, whichever comes first.
    replace_cost : float
        What replacing the part at this shut costs, at least 0.
    duration : str or float, optional
        How long the shut lasts (``"8h"``, ``"3d"``, or hours): the
        outage a diagram takes. A policy needs only when and how much.
    at : str, optional
        The clock time the shut starts (``"21:00"``), by default
        midnight. Not with a use, which falls when the meter gets there.

    Examples
    --------
    >>> from repyability import Shutdown
    >>> from repyability.schedule import Recurrence
    >>> Shutdown("weekly stop", Recurrence.weekly("Sat"), replace_cost=3.0,
    ...          duration="8h").describe()
    'weekly stop (every Saturday, $3)'
    >>> Shutdown("turnaround", dates=["2027-03-03", "2028-09-09"],
    ...          then_every="4 years", replace_cost=0.5).describe()
    'turnaround (3 Mar 2027, 9 Sep 2028, then every 4 years, $0.50)'
    """

    name: str
    rule: Any = None
    dates: Optional[Tuple[date, ...]] = None
    then_every: Optional[str] = None
    every: Any = None
    also_every: Optional[str] = None
    replace_cost: float = math.nan
    duration: Optional[float] = None
    at: Optional[time] = None

    def __post_init__(self):
        from repyability.schedule import Recurrence, Usage
        from repyability.schedule._meters import clock

        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError(f"A shut's name must be text; got {self.name!r}.")
        if self.name in (ANYTIME, "failure", "any other day"):
            raise ValueError(
                f"{self.name!r} names what is not a shut (a planned "
                "replacement on any other day, or a failure): give the shut "
                "another name."
            )
        given = [
            k
            for k, v in (
                ("rule", self.rule),
                ("dates", self.dates),
                ("every", self.every),
            )
            if v is not None
        ]
        if len(given) != 1:
            raise ValueError(
                f"Shutdown {self.name!r}: give when it happens one way: a "
                "rule (a Recurrence), dates, or every= (a period such as "
                "'30 days', or a Usage)."
            )
        if self.rule is not None and not isinstance(self.rule, Recurrence):
            raise TypeError(
                f"Shutdown {self.name!r}: rule must be a Recurrence "
                f"(Recurrence.weekly('Sat'), ...); got {self.rule!r}."
            )
        if self.dates is not None:
            if isinstance(self.dates, (str, date)) or not hasattr(
                self.dates, "__iter__"
            ):
                raise TypeError(
                    f"Shutdown {self.name!r}: dates must be a list of dates; "
                    f"got {self.dates!r}."
                )
            days = sorted(
                {_iso_date(d, f"A date of {self.name!r}") for d in self.dates}
            )
            if not days:
                raise ValueError(
                    f"Shutdown {self.name!r}: dates must have at least one."
                )
            object.__setattr__(self, "dates", tuple(days))
        if self.then_every is not None:
            if self.dates is None:
                raise ValueError(
                    f"Shutdown {self.name!r}: then_every is the gap after "
                    "the last of its dates: give dates= too."
                )
            _gap(self.then_every, f"Shutdown {self.name!r}: then_every")
        if self.every is not None:
            if isinstance(self.every, Usage):
                pass
            elif checks.is_number(self.every):
                if not 0.0 < float(self.every) < math.inf:
                    raise ValueError(
                        f"Shutdown {self.name!r}: every must be more than 0; "
                        f"got {self.every!r}."
                    )
                object.__setattr__(self, "every", float(self.every))
            else:
                _period_hours(self.every, f"Shutdown {self.name!r}: every")
        if self.also_every is not None:
            if not isinstance(self.every, Usage):
                raise ValueError(
                    f"Shutdown {self.name!r}: also_every is a calendar limit "
                    "on a shut by use (every=Usage(...)), whichever comes "
                    "first."
                )
            _period_hours(
                self.also_every, f"Shutdown {self.name!r}: also_every"
            )
        if not (
            checks.is_number(self.replace_cost)
            and 0.0 <= float(self.replace_cost) < math.inf
        ):
            raise ValueError(
                f"Shutdown {self.name!r}: replace_cost must be a number, at "
                f"least 0; got {self.replace_cost!r}."
            )
        object.__setattr__(self, "replace_cost", float(self.replace_cost))
        object.__setattr__(self, "duration", _duration(self.duration))
        if self.at is not None:
            if isinstance(self.every, Usage):
                raise ValueError(
                    f"Shutdown {self.name!r}: a shut by use falls when the "
                    "meter gets there, so it takes no clock time (at=)."
                )
            object.__setattr__(self, "at", clock(self.at, "at"))

    def __init__(
        self,
        name,
        rule=None,
        *,
        dates=None,
        then_every=None,
        every=None,
        also_every=None,
        replace_cost,
        duration=None,
        at=None,
    ):
        for key, value in (
            ("name", name),
            ("rule", rule),
            ("dates", dates),
            ("then_every", then_every),
            ("every", every),
            ("also_every", also_every),
            ("replace_cost", replace_cost),
            ("duration", duration),
            ("at", at),
        ):
            object.__setattr__(self, key, value)
        self.__post_init__()

    # -- words ---------------------------------------------------------

    @property
    def by_use(self) -> bool:
        """Whether it recurs by use (``every=Usage(...)``)."""
        from repyability.schedule import Usage

        return isinstance(self.every, Usage)

    def when(self) -> str:
        """When it happens, in words."""
        if self.rule is not None:
            return self.rule.describe()
        if self.dates is not None:
            text = ", ".join(_day_text(d) for d in self.dates)
            if self.then_every is not None:
                n, unit = _gap(self.then_every, "then_every")
                text += f", then every {_gap_words(n, unit)}"
            return text
        if self.by_use:
            text = self.every.describe()
            if self.also_every is not None:
                text += (
                    f" or {_period_hours(self.also_every, 'also_every')[1]}"
                )
                text += ", whichever comes first"
            return text
        if isinstance(self.every, float):
            return f"every {_number(self.every)} (the part's units)"
        return f"every {_period_hours(self.every, 'every')[1]}"

    def _after_dates(self) -> Optional["Shutdown"]:
        """With ``then_every``: the kind as it recurs from its last date
        (a [`Recurrence.every`][repyability.schedule.Recurrence.every]
        anchored there, whose first date is that last date), else None."""
        from repyability.schedule import Recurrence

        if self.then_every is None or not self.dates:
            return None
        n, unit = _gap(self.then_every, "then_every")
        return Shutdown(
            self.name,
            Recurrence.every(n, unit, self.dates[-1]),
            replace_cost=self.replace_cost,
            duration=self.duration,
            at=None if self.at is None else self.at.strftime("%H:%M"),
        )

    def describe(self) -> str:
        """The shut in a few words: ``"annual shut (the first Tuesday of
        November, $1)"``."""
        return f"{self.name} ({self.when()}, {_money(self.replace_cost)})"

    # -- saving --------------------------------------------------------

    def to_dict(self) -> dict:
        """The shut as plain data, ready for ``json.dumps``."""
        from repyability.schedule import Usage

        every: Any = self.every
        if isinstance(every, Usage):
            every = every.to_dict()
        return {
            "name": self.name,
            "rule": None if self.rule is None else self.rule.to_dict(),
            "dates": (
                None
                if self.dates is None
                else [d.isoformat() for d in self.dates]
            ),
            "then_every": self.then_every,
            "every": every,
            "also_every": self.also_every,
            "replace_cost": self.replace_cost,
            "duration_hours": self.duration,
            "at": None if self.at is None else self.at.strftime("%H:%M"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Shutdown":
        """A shut from its ``to_dict()``.

        Parameters
        ----------
        data : dict
            What ``to_dict()`` gave.

        Returns
        -------
        Shutdown
        """
        from repyability.schedule import Recurrence, Usage

        every = data.get("every")
        if isinstance(every, dict):
            every = Usage.from_dict(every)
        rule = data.get("rule")
        return cls(
            data["name"],
            None if rule is None else Recurrence.from_dict(rule),
            dates=data.get("dates"),
            then_every=data.get("then_every"),
            every=every,
            also_every=data.get("also_every"),
            replace_cost=data["replace_cost"],
            duration=data.get("duration_hours"),
            at=data.get("at"),
        )


class Shutdowns:
    """All of a site's shutdowns, as one record: each kind of shut, the
    price of a planned replacement on any other day (or none), and the
    site's time zone.

    The same record gives a part's replacement policy its cheap moments
    ([`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy])
    and a diagram its outages (``on_diagram``). A failure is not priced
    here: it costs the part's own ``unplanned_cost`` everywhere.

    Parameters
    ----------
    shutdowns : list of Shutdown
        The kinds of shut, each with its own name.
    any_day : float or None, optional
        What a planned replacement costs on any other day; None (the
        default): none is allowed but at the shuts (and at failures).
    time_zone : str, optional
        The site's IANA zone (``"Australia/Brisbane"``); None: clock times
        as they are, with no daylight saving.

    Examples
    --------
    >>> from repyability import Shutdown, Shutdowns
    >>> from repyability.schedule import Recurrence
    >>> site = Shutdowns(
    ...     [
    ...         Shutdown("weekly stop", Recurrence.weekly("Sat"),
    ...                  replace_cost=3.0, duration="8h"),
    ...         Shutdown("annual shut", Recurrence.yearly(
    ...             month="November", weekday="Tuesday", nth=1),
    ...             replace_cost=1.0, duration="3d"),
    ...     ],
    ...     any_day=4.0,
    ...     time_zone="Australia/Brisbane",
    ... )
    >>> print(site.describe())  # doctest: +NORMALIZE_WHITESPACE
    2 kinds of shut: weekly stop (every Saturday, $3) and annual shut
    (the first Tuesday of November, every year, $1); any other day $4.
    """

    def __init__(self, shutdowns=(), *, any_day=None, time_zone=None):
        from repyability.schedule._meters import zone

        if isinstance(shutdowns, Shutdown):
            shutdowns = [shutdowns]
        shutdowns = tuple(shutdowns)
        for shut in shutdowns:
            if not isinstance(shut, Shutdown):
                raise TypeError(
                    "Shutdowns takes a list of Shutdown, one per kind of "
                    f"shut; got {shut!r}."
                )
        names = [s.name for s in shutdowns]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"Shut names must differ; {repeated} repeat.")
        if any_day is not None and not (
            checks.is_number(any_day) and 0.0 <= float(any_day) < math.inf
        ):
            raise ValueError(
                "any_day must be a price, a number at least 0, or None for "
                f"no planned replacement on other days; got {any_day!r}."
            )
        zone(time_zone)
        self.shutdowns = shutdowns
        self.any_day = None if any_day is None else float(any_day)
        self.time_zone = time_zone

    def __repr__(self) -> str:
        return (
            f"Shutdowns({list(self.shutdowns)!r}, any_day={self.any_day!r}, "
            f"time_zone={self.time_zone!r})"
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Shutdowns) and self.to_dict() == other.to_dict()
        )

    def __hash__(self) -> int:
        return hash(repr(self.to_dict()))

    @property
    def names(self) -> Tuple[str, ...]:
        """The kinds' names, in order."""
        return tuple(s.name for s in self.shutdowns)

    def __getitem__(self, name: str) -> Shutdown:
        for shut in self.shutdowns:
            if shut.name == name:
                return shut
        raise KeyError(
            f"No shut named {name!r}; the site's are {list(self.names)}."
        )

    def describe(self) -> str:
        """The site in a sentence."""
        other = (
            "no planned replacement on other days"
            if self.any_day is None
            else f"any other day {_money(self.any_day)}"
        )
        if not self.shutdowns:
            return f"No shuts; {other}."
        n = len(self.shutdowns)
        kinds = "kind" if n == 1 else "kinds"
        listed = _listed([s.describe() for s in self.shutdowns])
        return f"{n} {kinds} of shut: {listed}; {other}."

    def dates(
        self, start, end, *, plan=None, readings=None
    ) -> Dict[str, Tuple[date, ...]]:
        """Every shut from ``start`` to ``end`` (both included), by kind:
        a rule's dates, the given dates, a plain period's from ``start``
        (one period on, then every period), and with a ``plan``, a use's
        (where the meter reaches each whole amount, from ``readings`` at
        ``start``).

        Parameters
        ----------
        start : date or str
            The first day.
        end : date or str
            The last day (included).
        plan : DutyPlan, optional
            How the equipment runs, for shuts by use.
        readings : dict, optional
            The meters at ``start``, for shuts by use.

        Returns
        -------
        dict
            Each kind's name and its dates, in order.
        """
        first, last = _iso_date(start, "start"), _iso_date(end, "end")
        if last < first:
            raise ValueError("end must be on or after start.")
        layout = _Layout(
            None, self, plan=plan, now=first, readings=readings, unit=None
        )
        span_end = layout.clock.midnight(last + timedelta(days=1))
        out = {}
        for kind in layout.kinds:
            moments = kind.moments_until(span_end)
            out[kind.shut.name] = tuple(
                sorted(
                    {
                        layout.clock.local_naive(m).date()
                        for m in moments
                        if first <= layout.clock.local_naive(m).date() <= last
                    }
                )
            )
        return out

    def route(
        self,
        part,
        *,
        plan=None,
        method=None,
        horizon=None,
        discount=None,
        now=None,
        readings=None,
        unit=None,
        failure_timing="half_step",
        dated_until=None,
    ) -> "ShutdownRoute":
        """How a policy for ``part`` at this site will be worked out,
        before working it out: the route, the solver, why, and about how
        long it takes (as ``analysis_routes()`` does for a diagram).

        Parameters
        ----------
        part : NonRepairable
            The part.
        plan : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        method : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        horizon : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        discount : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        now : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        readings : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        unit : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        failure_timing : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].
        dated_until : optional
            As for
            [`NonRepairable.optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy].

        Returns
        -------
        ShutdownRoute
            ``route``, ``solver``, ``text`` and ``estimate``. Where the
            site needs a choice (a long cycle, or shuts that don't
            repeat), ``route`` is ``"long_cycle"`` or
            ``"horizon_or_averaged"``, the solver None, and the text says
            what to choose.
        """
        layout = _Layout(
            part, self, plan=plan, now=now, readings=readings, unit=unit
        )
        decision = _decide(
            layout,
            method=method,
            horizon=horizon,
            discount=discount,
            failure_timing=failure_timing,
            refuse=False,
            dated_until=dated_until,
        )
        return decision.as_route()

    def on_diagram(
        self,
        start,
        end,
        *,
        unit: str = "hours",
        plan=None,
        readings=None,
        cost: Optional[Dict[str, float]] = None,
        scope: Optional[Dict[str, Any]] = None,
        idle_ageing=None,
    ) -> Dict[str, dict]:
        """The shuts as a diagram's outages: the spec
        [`RepairableRBD(shutdowns=...)`][repyability.RepairableRBD] takes,
        one shutdown per kind, at each shut's dates from ``start`` to
        ``end`` (through the ``plan`` for a shut by use), on the
        diagram's clock (real time since ``start``'s midnight, in
        ``unit``), each lasting its ``duration``.

        The same record gives a part's policy its opportunities, so a
        change to a shut's rule moves both.

        Parameters
        ----------
        start : date or str
            The diagram's time 0 is ``start``'s midnight.
        end : date or str
            The last day shuts are taken to (from ``start``, both
            included).
        unit : str, optional
            The diagram's unit, by default hours.
        plan : DutyPlan, optional
            How the equipment runs, for shuts by use.
        readings : dict, optional
            The meters at ``start``, for shuts by use.
        cost : dict, optional
            Each kind's shutdown ``"cost"``, by name.
        scope : dict, optional
            Each kind's shutdown ``"scope"``, by name, as a diagram's
            shutdown takes it.
        idle_ageing : bool, optional
            The shutdowns' ``"idle_ageing"``.

        Returns
        -------
        dict
            ``{name: {"times": [...], "duration": ...}}``, kinds with no
            shut in the span left out.
        """
        from repyability.schedule._meters import UNITS

        if unit not in UNITS:
            raise ValueError(
                f"unit must be one of {', '.join(map(repr, UNITS))}; got "
                f"{unit!r}."
            )
        first, last = _iso_date(start, "start"), _iso_date(end, "end")
        layout = _Layout(
            None, self, plan=plan, now=first, readings=readings, unit=None
        )
        origin = layout.origin
        span_end = layout.clock.midnight(last + timedelta(days=1))
        out: Dict[str, dict] = {}
        for kind in layout.kinds:
            shut = kind.shut
            if shut.duration is None:
                raise ValueError(
                    f"Shutdown {shut.name!r} has no duration: give it one "
                    "(duration='8h') to put it on a diagram."
                )
            moments = [m for m in kind.moments_until(span_end) if m >= origin]
            if not moments:
                continue
            times = sorted(
                {
                    round(
                        (m - origin).total_seconds() / 3600.0 / UNITS[unit],
                        9,
                    )
                    for m in moments
                }
            )
            spec: Dict[str, Any] = {
                "times": [float(t) for t in times],
                "duration": shut.duration / UNITS[unit],
            }
            for key, value in (
                ("cost", (cost or {}).get(shut.name)),
                ("scope", (scope or {}).get(shut.name)),
                ("idle_ageing", idle_ageing),
            ):
                if value is not None:
                    spec[key] = value
            out[shut.name] = spec
        return out

    def to_dict(self) -> dict:
        """The site as plain data, ready for ``json.dumps`` (``"kind"``
        ``"shutdowns"``, with a ``"version"``)."""
        return {
            "kind": "shutdowns",
            "version": 1,
            "shutdowns": [s.to_dict() for s in self.shutdowns],
            "any_day": self.any_day,
            "time_zone": self.time_zone,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Shutdowns":
        """A site from its ``to_dict()``.

        Parameters
        ----------
        data : dict
            What ``to_dict()`` gave.

        Returns
        -------
        Shutdowns
        """
        if not isinstance(data, dict) or data.get("kind") != "shutdowns":
            raise ValueError(
                "Not a saved Shutdowns: its kind must be 'shutdowns'; got "
                f"{data!r:.80}."
            )
        if data.get("version", 1) > 1:
            raise ValueError(
                f"This Shutdowns was saved by a newer RePyability (version "
                f"{data['version']}): upgrade to read it."
            )
        return cls(
            [Shutdown.from_dict(s) for s in data.get("shutdowns", [])],
            any_day=data.get("any_day"),
            time_zone=data.get("time_zone"),
        )


# ----------------------------------------------------------------- route


@dataclass(frozen=True)
class ShutdownRoute(_ResultMapping):
    """How a policy at a site will be worked out: returned by
    [`Shutdowns.route`][repyability.Shutdowns.route].

    Attributes
    ----------
    route : str
        ``"exact"``, ``"exact_long_cycle"``, ``"sampling"``,
        ``"averaged"``, ``"horizon"`` or ``"dated_then_long_run"`` (a
        dated window to the last fixed date, then the long-run average of
        the recurring shuts); where a choice is needed,
        ``"long_cycle"`` (choose ``method="exact"`` or ``"sampling"``)
        or ``"horizon_or_averaged"`` (choose ``method="averaged"``, or
        ``horizon=`` and ``discount=``).
    text : str
        Why, in plain words, and what to choose where a choice is needed.
    solver : str or None
        ``"closed form"``, ``"quadrature"``, ``"grid"`` or
        ``"sampling"``; None where a choice is needed.
    estimate : str
        About how long it takes.
    seconds : float or None
        The estimate, in seconds (None where a choice is needed).
    long_run_route : str or None
        For ``"dated_then_long_run"``: the route of the long run after
        the dated window (``solver`` is the window's).
    long_run_solver : str or None
        For ``"dated_then_long_run"``: the long run's solver.
    dated_until : str or None
        For ``"dated_then_long_run"``: the window's last day.
    """

    route: str
    text: str
    solver: Optional[str] = None
    estimate: str = ""
    seconds: Optional[float] = None
    long_run_route: Optional[str] = None
    long_run_solver: Optional[str] = None
    dated_until: Optional[str] = None

    def __repr__(self) -> str:
        return f"ShutdownRoute({self.route!r}, {self.text!r})"


def _seconds_text(seconds: float) -> str:
    if seconds < 1.0:
        return "under a second"
    if seconds < 90.0:
        return f"about {max(1, round(seconds))} s"
    if seconds < 5400.0:
        return f"about {round(seconds / 60.0)} min"
    return f"about {seconds / 3600.0:.1f} h"


# ---------------------------------------------------------------- layout


@dataclass
class _Pattern:
    """A kind's shuts repeating in the part's clock: every ``period``
    model units, at ``offsets`` (from model time 0) within one."""

    period: float
    offsets: Tuple[float, ...]


class _Kind:
    """One kind of shut laid on the part's clock."""

    def __init__(self, layout: "_Layout", shut: Shutdown):
        self.layout = layout
        self.shut = shut
        self.name = shut.name
        self.price = shut.replace_cost
        self._times: Optional[np.ndarray] = None
        self._times_end = -math.inf
        self._moments: Dict[float, datetime] = {}

    def nearest(
        self, t: float, within: float
    ) -> Tuple[float, Optional[datetime]]:
        """The shut at model time ``t``, or the nearest within ``within``
        (a grid lays a shut up to half a step away): its own model time
        and real instant."""
        if self._times is not None and self._times.size and within > 0.0:
            i = int(np.argmin(np.abs(self._times - t)))
            if abs(self._times[i] - t) <= within * (1 + 1e-9):
                t = float(self._times[i])
        return t, self.moment_of(t)

    def moment_of(self, t: float) -> Optional[datetime]:
        """The real instant of the shut at model time ``t`` (its own date,
        not the first moment the part's clock reaches ``t``)."""
        found = self._moments.get(float(np.round(t, 9)))
        if found is not None:
            return found
        return self.layout.moment_at([t])[0]

    # -- moments -------------------------------------------------------

    def moments_until(self, end: datetime) -> List[datetime]:
        """The shut's real instants from the layout's first moment to
        ``end`` (instants)."""
        layout = self.layout
        shut = self.shut
        clock_ = layout.clock
        start = layout.first_moment
        if shut.rule is not None or shut.dates is not None:
            first_day = clock_.local_naive(start).date()
            last_day = clock_.local_naive(end).date()
            if shut.rule is not None:
                days = shut.rule.dates(first_day, last_day)
            else:
                given: Tuple[date, ...] = shut.dates or ()
                days = tuple(d for d in given if first_day <= d <= last_day)
                after = shut._after_dates()
                if after is not None:
                    days = tuple(
                        sorted(
                            set(days)
                            | set(after.rule.dates(first_day, last_day))
                        )
                    )
            at = shut.at or time()
            moments = [clock_.wall(datetime.combine(d, at)) for d in days]
            return [m for m in moments if start <= m < end]
        if shut.by_use:
            return layout.use_moments(shut, end)
        # A plain period of real time, from now (at its clock time).
        hours, _ = _period_hours(shut.every, "every")
        base = layout.origin
        if shut.at is not None:
            base = clock_.wall(
                datetime.combine(layout.now, shut.at)  # type: ignore[arg-type]
            )
        count = int(math.floor((end - base).total_seconds() / 3600.0 / hours))
        return [base + timedelta(hours=hours * k) for k in range(1, count + 1)]

    def times(self, end: float) -> np.ndarray:
        """The shut's model times from 0 to ``end`` (both included)."""
        if self._times is None or end > self._times_end:
            self._times = self.layout.kind_times(self, end)
            self._times_end = end
        t = self._times
        return t[(t >= -1e-12) & (t <= end * (1 + 1e-12) + 1e-12)]

    # -- repeating -----------------------------------------------------

    def pattern(self) -> Optional[_Pattern]:
        """The kind's shuts as they repeat in the part's clock, or None
        where they do not (within ``_SAME``) over ``SPAN_YEARS``."""
        layout = self.layout
        shut = self.shut
        if (
            shut.by_use
            and layout.same_clock(shut)
            and not self.calendar_bites()
        ):
            amount = shut.every.amount * layout.scale
            reading = layout.reading(shut.every.meter) * layout.scale
            offset = (
                math.floor(reading / amount + 1e-9) + 1
            ) * amount - reading
            return _Pattern(
                amount, (offset % amount if offset >= amount else offset,)
            )
        if layout.now is None:
            # No dates: a plain period, in the part's own units.
            if isinstance(shut.every, float):
                return _Pattern(shut.every, (shut.every,))
            hours, _ = _period_hours(shut.every, "every")
            period = hours / layout.unit_hours()
            return _Pattern(period, (period,))
        times = self.times(layout.span_model())
        return _repeats(times)

    def calendar_bites(self) -> bool:
        """For a shut by use with a calendar limit: whether the limit ever
        comes first over ``SPAN_YEARS`` (a use alone: False)."""
        shut = self.shut
        if not shut.by_use or shut.also_every is None:
            return False
        limit, _ = _period_hours(shut.also_every, "also_every")
        moments = self.layout.use_moments(
            shut, self.layout.span_end(), limit=None
        )
        stamps = [self.layout.origin] + moments
        gaps = np.diff([m.timestamp() for m in stamps]) / 3600.0
        return bool(gaps.size == 0 or gaps.max() > limit * (1 + 1e-9))

    def gaps(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """The kind's usual gaps in the part's clock, for the averaged
        route: the distinct lengths, how often each comes, and words
        (``"every 364-371 days"``)."""
        layout = self.layout
        shut = self.shut
        pattern = None
        if shut.dates is not None:
            # Fixed dates take the dated window (``_dated``), never this.
            raise AssertionError("fixed dates are never averaged")
        try:
            pattern = self.pattern()
        except ValueError:
            pattern = None
        if pattern is not None and len(pattern.offsets) == 1:
            values = np.array([pattern.period])
            counts = np.array([1.0])
        else:
            times = self.times(layout.span_model())
            diffs = np.diff(np.sort(times))
            diffs = diffs[diffs > _SAME * max(1.0, float(np.max(times)))]
            if diffs.size == 0:
                raise ValueError(
                    f"{shut.name!r} has fewer than two shuts in the part's "
                    "clock over the next 28 years, so no usual gap: give a "
                    "horizon= and discount= instead."
                )
            values, counts = _grouped(diffs)
        return values, counts, self.gap_words()

    def gap_words(self) -> str:
        """The kind's gaps in words, for the averaged route's statement."""
        layout = self.layout
        shut = self.shut
        if shut.by_use:
            stamps = [layout.origin] + layout.use_moments(
                shut, layout.span_end()
            )
            days = np.diff([m.timestamp() for m in stamps])[1:] / 86400.0
            text = f"{shut.name}: {shut.every.describe()}"
            if days.size:
                text += f" = about {_number(round(float(np.mean(days))))} days"
            return text
        if layout.now is None:
            return f"{shut.name}: {shut.when()}"
        moments = self.moments_until(layout.span_end())
        if len(moments) < 2:
            return f"{shut.name}: {shut.when()}"
        days = np.diff([m.timestamp() for m in moments]) / 86400.0
        low, high = float(np.min(days)), float(np.max(days))
        if abs(high - low) < 1e-6:
            if abs(low - 1.0) < 1e-9:
                return f"{shut.name}: every day"
            return f"{shut.name}: every {_number(low)} days"
        distinct = np.unique(np.round(days, 3))
        if distinct.size == 2:
            return (
                f"{shut.name}: every {_number(low)} or {_number(high)} days, "
                "averaged"
            )
        return (
            f"{shut.name}: every {_number(low)}-{_number(high)} days, averaged"
        )

    def why_not(self) -> str:
        """Why the kind's shuts don't repeat in the part's clock, in
        words."""
        layout = self.layout
        shut = self.shut
        if shut.dates is not None:
            return f"the {shut.name} dates are one-offs"
        if shut.by_use and self.calendar_bites():
            return (
                f"the {shut.name}'s calendar limit ({shut.also_every}) "
                "sometimes comes before its use"
            )
        if shut.by_use and not layout.same_clock(shut):
            return (
                f"the {shut.name} ({shut.every.describe()}) doesn't keep step "
                f"with the part's clock ({layout.unit_words()})"
            )
        moments = self.moments_until(layout.span_end())
        if len(moments) >= 2:
            times = layout.model_at(moments)
            gaps = np.diff(times)
            low, high = float(gaps.min()), float(gaps.max())
            unit = layout.unit_words()
            days = np.diff([m.timestamp() for m in moments]) / 86400.0
            distinct = np.unique(np.round(days, 3))
            if distinct.size > 1:
                if distinct.size == 2:
                    a, b = _number(distinct[0]), _number(distinct[1])
                    gap = f"{a} or {b} days"
                else:
                    a, b = _number(distinct.min()), _number(distinct.max())
                    gap = f"{a} to {b} days"
                return f"the {shut.name} falls {gap} apart"
            return (
                f"the {shut.name} falls {_number(low)} to {_number(high)} "
                f"{unit} apart in the part's clock"
            )
        return f"the {shut.name} has too few shuts to repeat"


def _grouped(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distinct lengths and how often each comes (within ``_SAME``),
    binned into ``_MAX_GAPS`` where there are more."""
    scale = max(1.0, float(np.max(values)))
    keys = np.round(values / (scale * 1e-9)).astype(np.int64)
    unique, index, counts = np.unique(
        keys, return_index=True, return_counts=True
    )
    lengths = values[index]
    if lengths.size > _MAX_GAPS:
        edges = np.linspace(values.min(), values.max(), _MAX_GAPS + 1)
        which = np.clip(
            np.searchsorted(edges, values, side="right") - 1, 0, _MAX_GAPS - 1
        )
        sums = np.bincount(which, values, minlength=_MAX_GAPS)
        counts = np.bincount(which, minlength=_MAX_GAPS)
        keep = counts > 0
        return sums[keep] / counts[keep], counts[keep].astype(float)
    return lengths, counts.astype(float)


def _repeats(times: np.ndarray) -> Optional[_Pattern]:
    """The pattern model times ``times`` (from 0, in order) repeat with,
    if they do: the fewest shuts ``p`` such that each is a fixed length
    after the one ``p`` before, seen at least twice over."""
    times = np.unique(np.asarray(times, float))
    n = times.size
    if n < 3:
        return None
    scale = max(1.0, float(times[-1]))
    for p in range(1, n // 3 + 1):
        steps = times[p:] - times[:-p]
        if steps.max() - steps.min() <= _SAME * scale:
            period = float(np.mean(steps))
            if not period > 0.0:
                return None
            return _Pattern(period, tuple(float(t) for t in times[:p]))
    return None


class _Layout:
    """A site's shuts on a part's clock: model time 0 is ``now``'s
    midnight (or an arbitrary start with no ``now``), and model time is
    the part's age clock (calendar time in its unit, or a meter of the
    ``plan``)."""

    def __init__(self, part, site, *, plan, now, readings, unit):
        from repyability.schedule import DutyPlan
        from repyability.schedule._meters import Clock

        if not isinstance(site, Shutdowns):
            raise TypeError(
                f"site must be a Shutdowns (the site's shuts); got {site!r}."
            )
        if plan is not None and not isinstance(plan, DutyPlan):
            raise TypeError(
                "plan must be a DutyPlan (how the equipment runs); got "
                f"{plan!r}."
            )
        zone = site.time_zone
        if plan is not None and plan.time_zone != zone:
            if zone is None:
                zone = plan.time_zone
            elif plan.time_zone is not None:
                raise ValueError(
                    f"The site is in {site.time_zone!r} and the plan in "
                    f"{plan.time_zone!r}: give both the same time zone."
                )
        self.part = part
        self.site = site
        self.plan = plan
        self.given = dict(plan=plan, now=now, readings=readings, unit=unit)
        self.clock = Clock(zone)
        self.time_zone = zone
        self.notes: List[str] = []
        self.readings = dict(readings or {})
        for meter, value in self.readings.items():
            if not (checks.is_number(value) and math.isfinite(float(value))):
                raise ValueError(
                    f"The reading of {meter!r} must be a number; got "
                    f"{value!r}."
                )
        self.now = None if now is None else _iso_date(now, "now")
        needs_now = [
            s.name
            for s in site.shutdowns
            if s.rule is not None or s.dates is not None
        ]
        if self.now is None and (needs_now or plan is not None):
            what = (
                f"the {needs_now[0]} is on dates"
                if needs_now
                else "a plan is on dates"
            )
            raise ValueError(
                f"now= is needed, the day answers count from: {what}."
            )
        if self.now is None and self.readings:
            raise ValueError(
                "readings place shuts on dates: give now= too, the day they "
                "were read."
            )
        self.unit, self.meter, self.scale = self._resolve_unit(part, unit)
        for shut in site.shutdowns:
            if shut.by_use:
                if plan is None and not self.same_clock(shut):
                    raise ValueError(
                        f"The {shut.name} recurs by use "
                        f"({shut.every.describe()}): give plan=, how the "
                        "equipment runs, to place it on the part's clock."
                    )
                if shut.also_every is not None and self.now is None:
                    raise ValueError(
                        f"The {shut.name} has a calendar limit "
                        f"({shut.also_every}): give now= too."
                    )
            elif (
                part is not None
                and not isinstance(shut.every, float)
                and shut.every is not None
                and self.unit_hours_or_none() is None
                and plan is None
            ):
                if self.unit is None:
                    raise ValueError(
                        f"The {shut.name} is every {shut.every}, and the "
                        "part's life names no unit: give unit= (the unit it "
                        "was fitted in, such as 'days' or 'hours')."
                    )
                raise ValueError(
                    f"The {shut.name} is every {shut.every}, but the part's "
                    f"life is in {self.unit_words()}: give plan= to turn "
                    "calendar time into its clock."
                )
            if isinstance(shut.every, float) and self.now is not None:
                raise ValueError(
                    f"The {shut.name} is every {_number(shut.every)} of the "
                    "part's units, which has no dates: give every as text "
                    "('30 days') to use it with now=."
                )
        self._forecast_obj: Any = None
        self._forecast_end: Any = None
        self._first: Any = None
        self.origin: Any = None
        if self.now is not None:
            dated = [d for s in site.shutdowns if s.dates for d in s.dates]
            first_day = min([self.now] + dated)
            self._first = self.clock.midnight(first_day)
            self.origin = self.clock.midnight(self.now)
        else:
            self.origin = None
        self.kinds = [_Kind(self, s) for s in site.shutdowns]

    # -- units ---------------------------------------------------------

    def _resolve_unit(self, part, unit) -> Tuple[Optional[str], str, float]:
        """The part's unit (a time unit, or a meter of the plan), the
        meter its model time is read from, and the model units per unit
        of that meter."""
        from repyability.schedule._meters import UNITS, time_unit

        label = getattr(getattr(part, "reliability", None), "unit", None)
        label = label if isinstance(label, str) else None
        meters = self.plan.meters() if self.plan is not None else ()
        chosen = unit if unit is not None else label
        if unit is not None and label is not None:
            if (time_unit(unit) or unit) != (time_unit(label) or label):
                raise ValueError(
                    f"unit={unit!r}, but the part's life is labelled "
                    f"{label!r}: give the unit the life was fitted in."
                )
        if chosen is None:
            return None, "hours", 1.0
        named = time_unit(chosen)
        if named is not None:
            return named, "hours", 1.0 / UNITS[named]
        if self.plan is not None and chosen not in meters:
            raise ValueError(
                f"The part's life is in {chosen!r}, which the plan does not "
                f"drive (its meters are {', '.join(map(repr, meters))})."
            )
        return chosen, chosen, 1.0

    def unit_hours_or_none(self) -> Optional[float]:
        """Hours in one of the part's units, None for a meter or no unit."""
        from repyability.schedule._meters import UNITS

        if self.unit is None or self.unit not in UNITS:
            return None
        return UNITS[self.unit]

    def unit_hours(self) -> float:
        hours = self.unit_hours_or_none()
        if hours is None:
            raise ValueError(
                "The part's life names no unit of time: give unit= (the unit "
                "it was fitted in, such as 'days' or 'hours'), so that "
                "periods such as '30 days' can be put in it."
            )
        return hours

    def unit_words(self) -> str:
        if self.unit is None:
            return "the part's units"
        if self.plan is not None and self.meter == "hours" and self.unit:
            return f"operating {self.unit}"
        return self.unit

    def same_clock(self, shut: Shutdown) -> bool:
        """Whether a shut by use is on the meter the part's age is read
        from."""
        if not shut.by_use:
            return False
        meter = shut.every.meter
        if self.unit is None:
            return self.plan is None
        if self.meter == "hours":
            return meter == "hours"
        return meter == self.meter

    def reading(self, meter: str) -> float:
        value = self.readings.get(meter)
        return 0.0 if value is None else float(value)

    # -- forecast ------------------------------------------------------

    @property
    def first_moment(self) -> datetime:
        if self._first is None:
            raise ValueError("now= is needed to put shuts on dates.")
        return self._first

    def span_end(self) -> datetime:
        """The end of the span over which repeating is judged."""
        assert self.now is not None
        return self.clock.midnight(
            self.now.replace(year=self.now.year + SPAN_YEARS)
            if not (self.now.month == 2 and self.now.day == 29)
            else self.now.replace(year=self.now.year + SPAN_YEARS, day=28)
        )

    def _forecast(self, end: datetime):
        """The plan's forecast from the first moment to at least ``end``."""
        if self.plan is None:
            return None
        if self._forecast_end is None or end > self._forecast_end:
            target = max(end, self.span_end())
            if self._forecast_end is not None:
                target = max(
                    target,
                    self._forecast_end
                    + (self._forecast_end - self.first_moment),
                )
            readings = {k: float(v) for k, v in self.readings.items()}
            assert self.now is not None
            self._forecast_obj = self.plan.forecast(
                self.clock.local(self.first_moment),
                self.clock.local(target),
                readings=readings or None,
                read_at=self.clock.local(self.origin),
            )
            self._forecast_end = target
        return self._forecast_obj

    def model_at(self, moments) -> np.ndarray:
        """Model time (from now) at real instants."""
        moments = list(moments)
        if not moments:
            return np.zeros(0)
        if self.plan is None:
            hours = np.array(
                [(m - self.origin).total_seconds() / 3600.0 for m in moments]
            )
            return hours * (self.scale if self.unit is not None else 1.0)
        end = max(moments)
        forecast = self._forecast(end + timedelta(days=1))
        seconds = np.array(
            [(m - forecast._origin).total_seconds() for m in moments]
        )
        base = forecast._at_s(
            self.meter, (self.origin - forecast._origin).total_seconds()
        )
        return (forecast._at_s(self.meter, seconds) - base) * self.scale

    def moment_at(self, model_times) -> List[Optional[datetime]]:
        """The first real instant model time reaches each of
        ``model_times`` (None where the forecast never does)."""
        from repyability.schedule._usage import _cross

        values = np.atleast_1d(np.asarray(model_times, float))
        if self.origin is None:
            return [None for _ in values]
        if self.plan is None:
            per = 1.0 / (self.scale if self.unit is not None else 1.0)
            return [
                self.origin + timedelta(hours=float(v) * per) for v in values
            ]
        out: List[Optional[datetime]] = []
        for v in values:
            for _ in range(8):
                forecast = self._forecast(
                    self._forecast_end or self.span_end()
                )
                knots, rows = forecast._map(self.meter)
                base = float(
                    forecast._at_s(
                        self.meter,
                        (self.origin - forecast._origin).total_seconds(),
                    )
                )
                t = _cross(knots, rows[0], base + float(v) / self.scale)
                if t is not None:
                    out.append(
                        forecast._origin
                        + timedelta(seconds=max(float(t), 0.0))
                    )
                    break
                grown = (self._forecast_end or self.span_end()) + (
                    self._forecast_end - self.first_moment
                )
                if grown.year > self.first_moment.year + 400:
                    out.append(None)
                    break
                self._forecast(grown)
            else:
                out.append(None)
        return out

    def span_model(self) -> float:
        """Model time at the end of the span repeating is judged over."""
        return float(self.model_at([self.span_end()])[0])

    def kind_times(self, kind: _Kind, end: float) -> np.ndarray:
        """A kind's model times from 0 to ``end``."""
        shut = kind.shut
        if self.now is None:
            pattern = kind.pattern()
            assert pattern is not None
            count = int(math.floor(end / pattern.period)) + 2
            times = (
                np.asarray(pattern.offsets)[None, :]
                + pattern.period * np.arange(count)[:, None]
            ).ravel()
            return np.sort(times[times <= end * (1 + 1e-12)])
        if shut.by_use and self.same_clock(shut) and shut.also_every is None:
            amount = shut.every.amount * self.scale
            reading = self.reading(shut.every.meter) * self.scale
            first = (
                math.floor(reading / amount + 1e-9) + 1
            ) * amount - reading
            count = int(math.floor((end - first) / amount)) + 1
            return first + amount * np.arange(max(count, 0))
        until = self.moment_at([end])[0]
        if until is None:
            until = self._forecast_end or self.span_end()
        moments = kind.moments_until(until + timedelta(days=1))
        times = np.round(self.model_at(moments), 9)
        kind._moments = {}
        for t, m in zip(times, moments):
            kind._moments.setdefault(float(t), m)
        times = times[times >= -1e-9]
        return np.unique(times)

    def use_moments(
        self, shut: Shutdown, end: datetime, limit="shut"
    ) -> List[datetime]:
        """The instants a shut by use falls, from now to ``end``: where the
        meter reaches each whole amount from its reading now, or, with a
        calendar limit, each amount on from the last shut or the limit,
        whichever first. ``limit=None`` leaves the limit out."""
        usage = shut.every
        meter = usage.meter
        amount = float(usage.amount)
        reading = self.reading(meter)
        cap = None
        if limit == "shut" and shut.also_every is not None:
            cap, _ = _period_hours(shut.also_every, "also_every")
        if self.plan is None:
            # The meter is the part's own calendar hours.
            per = 1.0
            if self.unit is not None and self.meter == "hours":
                per = 1.0
            out: List[datetime] = []
            target = (math.floor(reading / amount + 1e-9) + 1) * amount
            last = self.origin
            while True:
                when = self.origin + timedelta(hours=(target - reading) * per)
                if cap is not None and when > last + timedelta(hours=cap):
                    when = last + timedelta(hours=cap)
                    target = (
                        reading + (when - self.origin).total_seconds() / 3600.0
                    )
                if when >= end:
                    return out
                out.append(when)
                last = when
                target += amount
        forecast = self._forecast(end)
        from repyability.schedule._usage import _cross

        knots, rows = forecast._map(meter)
        origin_s = (self.origin - forecast._origin).total_seconds()
        start_value = float(forecast._at_s(meter, origin_s))
        target = (math.floor(reading / amount + 1e-9) + 1) * amount
        target = start_value + (target - reading)
        out = []
        last_s = origin_s
        end_s = (end - forecast._origin).total_seconds()
        while True:
            t = _cross(knots, rows[0], target)
            t = math.inf if t is None else max(float(t), origin_s)
            if cap is not None and t > last_s + cap * 3600.0:
                t = last_s + cap * 3600.0
                target = float(forecast._at_s(meter, t))
            if not t < end_s:
                return out
            out.append(forecast._origin + timedelta(seconds=t))
            last_s = t
            target += amount

    # -- pace ----------------------------------------------------------

    def pace(self) -> Optional[float]:
        """Model units a year: from the unit for calendar time, or the
        plan's average over ``SPAN_YEARS``; None where neither is known."""
        if self.plan is None:
            hours = self.unit_hours_or_none()
            if hours is None:
                return None
            return 24.0 * 365.25 / hours
        end = self.span_end()
        years = (end - self.origin).total_seconds() / 3600.0 / (24.0 * 365.25)
        return float(self.model_at([end])[0]) / years

    def years_at(self, model_times: np.ndarray) -> np.ndarray:
        """Calendar years from now at which model time reaches each of
        ``model_times`` (for discounting)."""
        from repyability.schedule._usage import _value  # noqa: F401

        model_times = np.asarray(model_times, float)
        if self.plan is None:
            pace = self.pace()
            assert pace is not None
            return model_times / pace
        forecast = self._forecast(self.span_end())
        top = float(np.max(model_times)) if model_times.size else 0.0
        # model_at looks a day past what it is asked, so ask a day short
        # of the forecast's end (else every call would extend it).
        while (
            float(self.model_at([self._forecast_end - timedelta(days=1)])[0])
            < top
        ):
            self._forecast(
                self._forecast_end + (self._forecast_end - self.first_moment)
            )
        forecast = self._forecast_obj
        knots, rows = forecast._map(self.meter)
        origin_s = (self.origin - forecast._origin).total_seconds()
        base = float(forecast._at_s(self.meter, origin_s))
        values = rows[0]
        targets = base + model_times / self.scale
        k = np.clip(
            np.searchsorted(values, targets, side="left"), 1, values.size - 1
        )
        v0, v1 = values[k - 1], values[k]
        t0, t1 = knots[k - 1], knots[k]
        width = np.where(v1 > v0, v1 - v0, 1.0)
        t = np.where(v1 > v0, t0 + (targets - v0) * (t1 - t0) / width, t1)
        t = np.maximum(t, origin_s)
        return (t - origin_s) / 3600.0 / (24.0 * 365.25)


# -------------------------------------------------------------- decision


@dataclass
class _Decision:
    """How a policy will be worked out: the route, solver and words, and
    what the solver needs (the kinds' patterns and the cycle for the
    exact routes)."""

    route: str
    text: str
    solver: Optional[str]
    seconds: Optional[float]
    patterns: Dict[str, _Pattern] = field(default_factory=dict)
    cycle: Optional[float] = None
    horizon: Optional[datetime] = None
    discount: Optional[float] = None
    failure_timing: Optional[str] = None
    why: str = ""
    how: str = ""
    #: The dated window (``"dated_then_long_run"``): its last day, its end
    #: in the part's clock, the long run's decision and layout, and the
    #: fixed dates ``(kind, date)`` in it.
    window_day: Optional[date] = None
    window_end: Optional[float] = None
    long_run: Optional["_Decision"] = None
    recurring: Any = None
    fixed: Tuple[Tuple[str, date], ...] = ()

    def as_route(self) -> ShutdownRoute:
        long_run = self.long_run
        return ShutdownRoute(
            self.route,
            self.text,
            self.solver,
            "" if self.seconds is None else _seconds_text(self.seconds),
            self.seconds,
            None if long_run is None else long_run.route,
            None if long_run is None else long_run.solver,
            None if self.window_day is None else self.window_day.isoformat(),
        )


def _grid_seconds(cycle_steps: float, ages: float) -> float:
    """About how long relative value iteration over a cycle takes: the
    steps swept (at least twice the cycle, at most 50 lives of the age
    grid) at about 12 microseconds a step and 5.5 ns an age."""
    work = max(2.0 * cycle_steps, 50.0 * ages)
    return work * (1.2e-5 + 5.5e-9 * ages)


def _quadrature_seconds(shuts: float, ages: float, sweeps: float) -> float:
    """About how long quadrature over ``shuts`` gaps takes, ``sweeps``
    times: about 50 microseconds and 0.3 microseconds an age, per gap."""
    return shuts * sweeps * (5e-5 + 3e-7 * ages)


def _life_scale(part) -> Tuple[float, float, float]:
    """The part's mean life, the age it outlives with a chance of 1e-9
    (the oldest on a grid) and of 1e-3 (a life's span)."""
    from repyability.maintenance._replacement import (
        FAR,
        SPAN,
        _age_where,
        _mean,
        _survival,
    )

    sf = _survival(part)
    mean = _mean(part, sf)
    return mean, _age_where(sf, FAR, mean), _age_where(sf, SPAN, mean)


def _horizon_moment(layout: _Layout, horizon) -> datetime:
    day = _iso_date(horizon, "horizon")
    assert layout.now is not None
    if day <= layout.now:
        raise ValueError(
            f"horizon must be after now ({layout.now}); got {horizon!r}."
        )
    return layout.clock.midnight(day)


def _decide(
    layout: _Layout,
    *,
    method,
    horizon,
    discount,
    failure_timing,
    refuse: bool = True,
    dated_until=None,
) -> _Decision:
    """The route a policy takes, its solver and the words for both;
    ``refuse``: raise where the caller must choose (else return the
    choice as the route). A site with fixed dates (or ``dated_until``)
    takes the dated window, then the long run of its recurring shuts
    (``_dated``)."""
    from repyability.maintenance._integration import (
        MAX_AGES,
        MAX_OPPORTUNITIES,
        run_ages,
    )

    checks.one_of("failure_timing", failure_timing, FAILURE_TIMINGS)
    if method not in METHODS:
        raise ValueError(
            "method must be None (worked out exactly where the maths allows), "
            f"'exact', 'sampling' or 'averaged'; got {method!r}."
        )
    site = layout.site
    part = layout.part
    long_discount = None
    if horizon is None and discount is not None:
        long_discount = _long_run_discount(layout, discount, method)
    elif horizon is not None:
        if discount is None:
            raise ValueError(
                "horizon= needs discount= too: the yearly rate future costs "
                "are discounted at (0 for the plain total)."
            )
        if method is not None:
            raise ValueError(
                f"A horizon is its own route: give method={method!r} or "
                "horizon= and discount=, not both."
            )
        if not (checks.is_number(discount) and 0.0 <= float(discount) < 10.0):
            raise ValueError(
                "discount must be a yearly rate such as 0.07 (7%), at least "
                f"0; got {discount!r}."
            )
        if layout.now is None:
            raise ValueError(
                "A horizon is on real dates: give now= too, the day costs "
                "count from."
            )
        if dated_until is not None:
            raise ValueError(
                "dated_until= extends the dated window before the long run; "
                "a horizon already takes every shut on its real dates to its "
                "end: give one or the other."
            )
    any_day = site.any_day
    if not site.shutdowns and any_day is None:
        raise ValueError(
            "The site allows no planned replacement (no shuts, and none on "
            "any other day: any_day is None), so there is no policy to find "
            "but running to failure: give a shut or an any_day price."
        )
    if failure_timing == "exact" and any_day is not None:
        raise NotImplementedError(
            "failure_timing='exact' works between the shuts in continuous "
            "time, and this site also allows a planned replacement on any "
            f"other day ({_money(any_day)}), whose best moment inside a gap "
            "the quadrature does not find: use failure_timing='half_step' "
            "(the default), which gives the same to about 1e-7."
        )
    mean, oldest, span = _life_scale(part)
    if horizon is not None:
        end = _horizon_moment(layout, horizon)
        end_model = float(layout.model_at([end])[0])
        rate = float(discount)
        words = f"{_day_text(layout.clock.local_naive(end).date())}"
        if failure_timing == "exact":
            shuts = sum(k.times(end_model).size for k in layout.kinds)
            ages = float(run_ages(mean, oldest))
            seconds = _quadrature_seconds(shuts, ages, 2.0)
            if shuts > MAX_OPPORTUNITIES:
                raise ValueError(
                    f"failure_timing='exact' is refused beyond "
                    f"{MAX_OPPORTUNITIES:,} shuts, and the horizon to {words} "
                    f"has {shuts:,}: use failure_timing='half_step' (the "
                    "default), which gives the same to about 1e-7."
                )
            text = (
                f"Lowest net present cost on the real dates to {words} at "
                f"{rate:.0%} a year, by quadrature: the part's life is "
                f"integrated across each gap between shuts ({shuts:,} shuts, "
                "32 pieces each), failures timed at the instant they happen "
                "(failure_timing='exact'). "
                f"{_seconds_text(seconds).capitalize()}. "
                "The default, failure_timing='half_step', gives the same to "
                "about 1e-7, faster."
            )
            solver = "quadrature"
        else:
            steps = end_model / (mean / 1000.0)
            ages = oldest / (mean / 1000.0)
            seconds = 3.0 * steps * (1.2e-5 + 5.5e-9 * ages)
            text = (
                f"Lowest net present cost on the real dates to {words} at "
                f"{rate:.0%} a year, on a grid of steps sized from the life "
                f"(about {_number(mean / 1000.0)} {layout.unit_words()}), "
                "each "
                "failure's replacement counted half a step old "
                "(failure_timing='half_step'). "
                f"{_seconds_text(seconds).capitalize()}."
            )
            solver = "grid"
        return _Decision(
            "horizon",
            text,
            solver,
            seconds,
            horizon=end,
            discount=rate,
            failure_timing=failure_timing,
        )
    if horizon is None:
        from repyability.maintenance._dated import dated_decision

        dated = dated_decision(
            layout,
            method=method,
            failure_timing=failure_timing,
            refuse=refuse,
            dated_until=dated_until,
            discount=long_discount,
        )
        if dated is not None:
            return dated
    if not site.shutdowns:
        if long_discount is not None:
            return _Decision(
                "exact",
                "No shuts: this is plain age replacement, at the age with "
                "the least equivalent annual cost at "
                f"{_percent(long_discount)} a year (Fox, 1966; one "
                "discounted integral of the part's survival an age).",
                "closed form",
                0.01,
                cycle=None,
                discount=long_discount,
                failure_timing=None,
                why="there are no shuts, so this is plain age replacement",
                how="by Fox's formula (closed form)",
            )
        return _Decision(
            "exact",
            "No shuts: this is plain age replacement (exact long-run "
            "average), "
            "by the renewal-reward formula (closed form; one integral of the "
            "part's survival).",
            "closed form",
            0.01,
            cycle=None,
            failure_timing=None,
            why="there are no shuts, so this is plain age replacement",
            how="by the renewal-reward formula (closed form)",
        )
    if method == "averaged":
        gaps: List[str] = []
        for kind in layout.kinds:
            kind.gaps()  # refuses a kind with no gap
            gaps.append(kind.gap_words())
        return _Decision(
            "averaged",
            "Long-run average by the renewal-reward formula (closed form; one "
            "integral of the part's survival). Assumes shuts fall evenly "
            f"within their usual gaps ({'; '.join(gaps)}). Exact dates not "
            "used for the cost (they are for the dates given back).",
            "closed form",
            0.5,
            failure_timing=None,
            why="; ".join(gaps),
        )
    patterns: Dict[str, _Pattern] = {}
    broken: List[str] = []
    for kind in layout.kinds:
        pattern = kind.pattern()
        if pattern is None:
            broken.append(kind.why_not())
        else:
            patterns[kind.name] = pattern
    if broken:
        choice = (
            "Choose method='averaged' (assumes shuts fall evenly within their "
            "usual gaps) or give horizon= and discount= (lowest net present "
            "cost on the real dates)."
        )
        if long_discount is not None:
            choice = (
                "A discounted long run needs shuts that repeat, so that the "
                "same cycle comes round again: give horizon= with discount= "
                "(lowest net present cost on the real dates), or leave "
                "discount= out for method='averaged'."
            )
        reason = (
            "These shuts don't repeat on a fixed cycle in the part's clock: "
            f"{_listed(broken)}."
        )
        if method in ("exact", "sampling"):
            raise ValueError(
                f"{reason} So there is no repeating cycle to "
                f"{'be exact over' if method == 'exact' else 'sample'}: use "
                "method='averaged' or a horizon (horizon= and discount=)."
            )
        if refuse:
            raise ValueError(f"{reason} {choice}")
        return _Decision(
            "horizon_or_averaged", f"{reason} {choice}", None, None
        )
    cycle = float(_common_multiple(p.period for p in patterns.values()))
    total = sum(len(p.offsets) * cycle / p.period for p in patterns.values())
    step = _exact_step(patterns, cycle, mean)
    cycle_steps = cycle / step
    ages = oldest / step
    if failure_timing == "exact":
        q_ages = min(MAX_AGES, max(801.0, oldest / step))
        sweeps = max(2.0, min(400.0 * oldest / cycle, 50.0 * oldest / cycle))
        seconds = _quadrature_seconds(total, q_ages, sweeps)
        solver = "quadrature"
        if total > MAX_OPPORTUNITIES:
            seconds = math.inf
    else:
        seconds = _grid_seconds(cycle_steps, ages)
        solver = "grid"
    repeat = _repeat_words(layout, patterns, cycle)
    solving = "by quadrature" if failure_timing == "exact" else "on the grid"
    if method == "sampling":
        return _Decision(
            "sampling",
            f"An estimate of the long-run average, from slices of the cycle "
            f"({repeat}), each solved exactly ({solving}), averaged; the "
            "result has its standard error.",
            "sampling",
            None,
            patterns=patterns,
            cycle=cycle,
            failure_timing=failure_timing,
            why=repeat,
            how=(
                "from sampled slices of the cycle, each solved "
                + (
                    "by quadrature"
                    if failure_timing == "exact"
                    else "on the grid"
                )
            ),
        )
    short = seconds <= SHORT_SECONDS
    if failure_timing == "exact" and total > MAX_OPPORTUNITIES:
        raise ValueError(
            f"failure_timing='exact' is refused beyond {MAX_OPPORTUNITIES:,} "
            f"shuts a cycle, and this cycle ({repeat}) has {int(total):,}: "
            "use "
            "failure_timing='half_step' (the default), which gives the same "
            "to about 1e-7."
        )
    how = how_words(failure_timing)
    lead = "Exact long-run average"
    if long_discount is not None:
        lead = (
            "Lowest equivalent annual cost at "
            f"{_percent(long_discount)} a year, discounted over the long run "
            "(the present value of every replacement from now on, forever, "
            "by discounted value iteration over the cycle)"
        )
    if short:
        return _Decision(
            "exact",
            f"{lead}: {repeat}, so nothing about shut timing "
            f"is assumed; worked out {how}.",
            solver,
            seconds,
            patterns=patterns,
            cycle=cycle,
            discount=long_discount,
            failure_timing=failure_timing,
            why=f"{repeat}, so nothing about shut timing is assumed",
            how=how,
        )
    long_text = (
        f"The shuts repeat, but only after a long cycle ({repeat}): working "
        f"it out exactly takes {_seconds_text(seconds)}."
    )
    if method == "exact":
        return _Decision(
            "exact_long_cycle",
            f"{lead} over a long cycle ({repeat}), worked out "
            f"{how}: {_seconds_text(seconds)}.",
            solver,
            seconds,
            patterns=patterns,
            cycle=cycle,
            discount=long_discount,
            failure_timing=failure_timing,
            why=f"over a long cycle: {repeat}",
            how=how,
        )
    choice = (
        f"{long_text} Choose method='exact' (the same calculation, "
        f"{_seconds_text(seconds)}) or method='sampling' (an estimate with "
        "its "
        "standard error, as many samples as you ask for)."
    )
    if long_discount is not None:
        choice = (
            f"{long_text} Discounted, choose method='exact' (the same "
            f"calculation, {_seconds_text(seconds)}), or give horizon= with "
            "discount= (lowest net present cost on the real dates); sampling "
            "gives a long-run average, not a present value."
        )
    if refuse:
        raise ValueError(choice)
    return _Decision(
        "long_cycle",
        choice,
        None,
        None,
        patterns=patterns,
        cycle=cycle,
        discount=long_discount,
    )


def how_words(failure_timing) -> str:
    """How the exact routes work a cycle out, in words."""
    if failure_timing == "exact":
        return (
            "by quadrature (the part's life integrated across each gap "
            "between shuts, failures timed at the instant they happen)"
        )
    return (
        "on a grid of steps sized from the life, each failure's "
        "replacement counted half a step old"
    )


def _percent(rate: float) -> str:
    """A yearly rate in words: ``7%``, ``2.5%``."""
    return f"{100.0 * rate:.4g}%"


def _long_run_discount(layout: _Layout, discount, method) -> Optional[float]:
    """The yearly discount of a discounted long run (``discount=`` with no
    horizon), checked: None for 0 (the long-run average). Refused where
    the long run is not discounted exactly: the averaged and sampled
    routes (long-run averages), a duty plan (a clock that does not keep a
    steady pace with the calendar) and a clock with no pace in time."""
    if not (checks.is_number(discount) and 0.0 <= float(discount) < 10.0):
        raise ValueError(
            "discount must be a yearly rate such as 0.07 (7%), at least "
            f"0; got {discount!r}."
        )
    rate = float(discount)
    if rate == 0.0:
        return None
    if method in ("averaged", "sampling"):
        raise ValueError(
            f"method={method!r} gives a long-run average, which is not "
            "discounted: leave method out (or give method='exact') with "
            "discount=, or give horizon= with discount= (lowest net present "
            "cost on the real dates)."
        )
    if layout.plan is not None:
        raise ValueError(
            "A discounted long run needs the part's clock to keep a steady "
            "pace with the calendar, and a duty plan's does not: give "
            "horizon= with discount= (lowest net present cost on the real "
            "dates, through the plan), or leave discount= out."
        )
    if layout.pace() is None:
        raise ValueError(
            f"The part's clock ({layout.unit}) has no pace in time, so a "
            "yearly discount is no rate on it: give a duty plan with "
            "horizon= and discount=, or leave discount= out."
        )
    return rate


def _exact_step(
    patterns: Dict[str, _Pattern], cycle: float, mean: float
) -> float:
    """The grid step the exact route takes (as ``_default_step`` gives
    it for the calendar of the patterns)."""
    from repyability.maintenance._replacement import (
        STEPS_PER_LIFE,
        _common_divisor,
    )

    base = mean / STEPS_PER_LIFE
    periods = [p.period for p in patterns.values()]
    others = [o for p in patterns.values() for o in p.offsets]
    for common in (
        _common_divisor(periods + others),
        _common_divisor(periods),
    ):
        if common is not None and common >= base / 10.0:
            return common / math.ceil(common / base)
    return base


def _repeat_words(
    layout: _Layout, patterns: Dict[str, _Pattern], cycle: float
) -> str:
    """How the shuts repeat, in words."""
    unit = layout.unit_words()
    if len(patterns) == 1:
        name, pattern = next(iter(patterns.items()))
        shut = layout.site[name]
        kind = next(k for k in layout.kinds if k.name == name)
        if (
            shut.by_use
            and layout.same_clock(shut)
            and not kind.calendar_bites()
        ):
            return (
                f"the {name} falls {shut.every.describe()}, on the part's own "
                f"clock ({unit})"
            )
        if len(pattern.offsets) == 1:
            return f"the {name} repeats every {_number(pattern.period)} {unit}"
        return (
            f"the {name} repeats every {_number(pattern.period)} {unit} "
            f"({len(pattern.offsets)} shuts a cycle)"
        )
    each = ", ".join(
        f"{name} every {_number(p.period)}" for name, p in patterns.items()
    )
    return f"the shuts repeat every {_number(cycle)} {unit} ({each})"
