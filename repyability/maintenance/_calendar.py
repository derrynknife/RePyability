"""The opportunities to replace a part, in model time (#319, UC-13 of this
package's ``README.md``): shuts on a period (weekly, quarterly, annual,
each with its own price and offset) or on given times, a planned
replacement on any day at a dearer price, and a failure at the dearest.

Times are numbers in the life model's unit, from 0; dates are turned
into them at the edge (``repyability.schedule`` gives the dates). A
calendar is laid on a grid of a step for the dynamic programme
(``_Grid``): each date to its nearest grid time, so a date moves by half a
step at most and the dates of a period keep no drift.
"""

import math
import warnings
from dataclasses import dataclass, fields
from datetime import date
from fractions import Fraction
from typing import Any, Optional, Tuple

import numpy as np

from repyability.utils import checks
from repyability.utils.wrappers import outside_level

#: The name of a planned replacement on any day, among the kinds of
#: replacement a schedule lists; no opportunity may take it.
ANYTIME = "anytime"

#: A price taken from the part: its unplanned cost (a failure) or its
#: planned cost (a replacement on any day).
PART = "part"

#: The largest denominator a period is read with, as a fraction: a period
#: of 365.25 is 1461/4, of 30.4375 is 487/16.
_DENOMINATOR = 1_000_000

#: The share of a step a date may move, laid on the grid, before a
#: warning says so.
_MOVED = 0.01


def _fraction(value: float) -> Fraction:
    """``value`` as the fraction it is read as, for periods' common
    multiple."""
    return Fraction(value).limit_denominator(_DENOMINATOR)


def _common_multiple(values) -> Fraction:
    """The least common multiple of positive fractions."""
    fractions = [_fraction(v) for v in values]
    numerator = math.lcm(*(f.numerator for f in fractions))
    denominator = math.gcd(*(f.denominator for f in fractions))
    return Fraction(numerator, denominator)


def _price(value, name: str) -> float:
    """``value`` as a price: a finite number, at least 0."""
    if not (checks.is_number(value) and 0.0 <= float(value) < math.inf):
        raise ValueError(
            f"{name} must be a number, at least 0; got {value!r}."
        )
    return float(value)


#: The names that are no opportunity's: a planned replacement on any day,
#: and a failure (on real dates, the words results use for them).
_NOT_NAMES = (ANYTIME,)
_NOT_NAMES_ON_DATES = (ANYTIME, "failure", "any other day", "any day")


def _price_deprecated(where: str) -> None:
    """Warn that a price was put on an opportunity (``where`` says how)."""
    from repyability.utils.deprecation import price_on_part

    price_on_part(where)


@dataclass(frozen=True, init=False, repr=False)
class Opportunity:
    """A kind of opportunity to replace a part: a stop or shut (weekly,
    quarterly, annual, a turnaround, a service) that says when it happens,
    how long it lasts and what it takes down. It carries no price: an
    opportunity applies to every part, and each part brings its own
    (`NonRepairable`'s ``opportunity_cost`` and ``opportunity_costs``).
    Opportunities are collected in
    [`Opportunities`][repyability.Opportunities].

    It is either in **model time** (numbers in the life's unit, from 0):

    - every ``period`` from ``offset`` (on and on, both ways:
      ``offset + k * period`` for every whole ``k``), or
    - ``dates`` given as numbers;

    or on **real dates**:

    - a ``rule`` (a [`Recurrence`][repyability.schedule.Recurrence]:
      every Saturday, the first Tuesday of November),
    - ``dates`` given as dates (with ``then_every`` they recur at that gap
      after the last), or
    - ``every``, a plain period (``"30 days"``; a number is that many of
      the part's own units) or a use (``Usage(15_000, "km")``), with
      ``also_every`` a calendar limit, whichever comes first.

    Parameters
    ----------
    name : str
        Its name, as results list it (not ``"anytime"``, which names a
        replacement on any day; on real dates not ``"failure"``,
        ``"any day"`` or ``"any other day"`` either).
    rule : Recurrence, optional
        When it happens, on real dates.
    dates : list of float, or of date or str, optional
        Its times: numbers in model time, or dates (``"2027-03-03"``).
    period : float, optional
        In model time: the time from one to the next, more than 0.
    offset : float, optional
        In model time: the time of one of them, by default 0. Only with
        ``period``.
    then_every : str, optional
        On real dates, with ``dates``: the gap at which it recurs after
        its last date (``"4 years"``, ``"18 months"``; whole numbers).
    every : str, float or Usage, optional
        On real dates: a plain period or a use, instead of a rule or
        dates.
    also_every : str, optional
        With ``every=Usage(...)``: a plain period, whichever comes first.
    duration : float or str, optional
        How long it lasts. In model time a number in the life's unit; on
        real dates text (``"8h"``, ``"3d"``) or a number of hours. A part
        whose ``job_duration`` is longer does not use it; a diagram takes
        it as the outage. None (the default): not said, and every part
        may use it.
    at : str, optional
        On real dates: the clock time it starts (``"21:00"``), by default
        midnight. Not with a use.
    takes_down : list of str, optional
        The labels (lines, units, areas) it takes down; by default None,
        the whole site. A policy given ``where=`` uses only the
        opportunities that take that label down. On a diagram
        (``RepairableRBD(shutdowns=Opportunities(...))``) the labels are
        the nodes it holds off line.
    idle_ageing : bool, optional
        On a diagram: whether the blocks it holds off line and does not
        work on keep ageing through it (True, the default) or stand still
        until it ends. A part's policy does not use it.
    opportunistic : float or dict, optional
        On a diagram: the components it renews if they are at least so
        old, as a diagram's shutdown's ``"opportunistic"`` (an age for
        every component in its scope under age replacement, or a dict of
        an age by component). A part's policy does not use it.
    cost : float, optional
        **Deprecated** (also as the second argument, the 0.14 form
        ``Opportunity(name, cost, period=...)``): a price on the
        opportunity, applied to the part being solved as its override
        for this opportunity. Give the part ``opportunity_cost`` or
        ``opportunity_costs`` instead.

    Examples
    --------
    >>> from repyability import Opportunity
    >>> from repyability.schedule import Recurrence
    >>> weekly = Opportunity("weekly", period=7.0)
    >>> annual = Opportunity("annual", period=364.0, offset=182.0)
    >>> turnarounds = Opportunity("turnaround", dates=[400.0, 1130.0])
    >>> weekly.times(0.0, 30.0).tolist()
    [0.0, 7.0, 14.0, 21.0, 28.0]
    >>> stop = Opportunity("weekly stop", Recurrence.weekly("Sat"),
    ...                    duration="8h")
    >>> stop.model_time, stop.describe()
    (False, 'weekly stop (every Saturday, 8 h)')
    >>> Opportunity("turnaround", dates=["2027-03-03", "2028-09-09"],
    ...             duration="21d", takes_down=["Line 2"]).describe()
    'turnaround (3 Mar 2027, 9 Sep 2028, 504 h, takes down Line 2)'
    """

    name: str
    cost: Optional[float] = None
    period: Optional[float] = None
    offset: float = 0.0
    dates: Optional[Tuple[Any, ...]] = None
    rule: Any = None
    then_every: Optional[str] = None
    every: Any = None
    also_every: Optional[str] = None
    duration: Optional[float] = None
    at: Any = None
    takes_down: Optional[Tuple[str, ...]] = None
    idle_ageing: bool = True
    opportunistic: Any = None

    def __init__(
        self,
        name,
        rule=None,
        *legacy,
        dates=None,
        period=None,
        offset=None,
        then_every=None,
        every=None,
        also_every=None,
        duration=None,
        at=None,
        takes_down=None,
        cost=None,
        idle_ageing=True,
        opportunistic=None,
    ):
        if checks.is_number(rule):
            # The 0.14 form, Opportunity(name, cost, period, offset, dates).
            if cost is not None:
                raise TypeError(
                    "Opportunity() got the price twice: as its second "
                    "argument and as cost=."
                )
            if len(legacy) > 3:
                raise TypeError(
                    "Opportunity(name, cost, period, offset, dates) takes "
                    f"at most 5 arguments; got {2 + len(legacy)}."
                )
            cost, rule = rule, None
            given = dict(zip(("period", "offset", "dates"), legacy))
            for key, value in given.items():
                if {"period": period, "offset": offset, "dates": dates}[
                    key
                ] is not None:
                    raise TypeError(f"Opportunity() got {key} twice.")
            period = given.get("period", period)
            offset = given.get("offset", offset)
            dates = given.get("dates", dates)
            _price_deprecated("Opportunity(name, cost, ...)")
        elif legacy:
            raise TypeError(
                "Opportunity() takes the name and a rule as its only "
                "positional arguments; give the rest by name (dates=, "
                "period=, duration=, ...)."
            )
        elif cost is not None:
            _price_deprecated("Opportunity(..., cost=)")
        for key, value in (
            ("name", name),
            ("cost", cost),
            ("period", period),
            ("offset", offset),
            ("dates", dates),
            ("rule", rule),
            ("then_every", then_every),
            ("every", every),
            ("also_every", also_every),
            ("duration", duration),
            ("at", at),
            ("takes_down", takes_down),
            ("idle_ageing", idle_ageing),
            ("opportunistic", opportunistic),
        ):
            object.__setattr__(self, key, value)
        self._check()

    def __repr__(self) -> str:
        shown = [repr(self.name)]
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "name" or value is None:
                continue
            if f.name == "offset" and value == 0.0:
                continue
            if f.name == "idle_ageing" and value is True:
                continue
            shown.append(f"{f.name}={value!r}")
        return f"Opportunity({', '.join(shown)})"

    def _check(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(
                f"An opportunity's name must be text; got {self.name!r}."
            )
        if self.name == ANYTIME:
            raise ValueError(
                f"{ANYTIME!r} names a planned replacement on any day: give "
                "the opportunity another name."
            )
        if self.cost is not None:
            object.__setattr__(
                self, "cost", _price(self.cost, f"The cost of {self.name!r}")
            )
        on_dates = (
            self.rule is not None
            or self.every is not None
            or (self.dates is not None and not _numbers(self.dates))
        )
        in_model = self.period is not None or (
            self.dates is not None and _numbers(self.dates)
        )
        if on_dates and self.period is not None:
            raise ValueError(
                f"Opportunity {self.name!r}: period= is for model time, and "
                "a rule, dates or every= for real dates: give one of the "
                "two."
            )
        if not (on_dates or in_model):
            raise ValueError(
                f"Opportunity {self.name!r}: say when it happens, one of: a "
                "period (with an offset) or dates as numbers, in model "
                "time; or a rule (a Recurrence), dates or every=, on real "
                "dates."
            )
        if in_model:
            self._check_model_time()
        else:
            self._check_dates()
        takes_down = self.takes_down
        if takes_down is not None:
            if isinstance(takes_down, str):
                takes_down = (takes_down,)
            takes_down = tuple(takes_down)
            if not takes_down or not all(
                isinstance(t, str) and t for t in takes_down
            ):
                raise ValueError(
                    f"Opportunity {self.name!r}: takes_down must be a list "
                    "of labels (text), at least one, or None for the whole "
                    f"site; got {self.takes_down!r}."
                )
            object.__setattr__(self, "takes_down", takes_down)
        self._check_diagram_options()

    def _check_diagram_options(self) -> None:
        """``idle_ageing`` and ``opportunistic``, which only a diagram
        uses: checked here for their kind, and by the diagram against its
        nodes."""
        if not isinstance(self.idle_ageing, (bool, np.bool_)):
            raise ValueError(
                f"Opportunity {self.name!r}: idle_ageing must be True (the "
                "blocks not worked on keep ageing) or False (they stand "
                f"still), got {self.idle_ageing!r}."
            )
        object.__setattr__(self, "idle_ageing", bool(self.idle_ageing))
        given = self.opportunistic
        if given is None:
            return
        if isinstance(given, tuple) and all(
            isinstance(pair, tuple) and len(pair) == 2 for pair in given
        ):
            given = dict(given)
        ages = list(given.values()) if isinstance(given, dict) else [given]
        if not ages or not all(
            checks.is_number(a) and float(a) >= 0.0 for a in ages
        ):
            raise ValueError(
                f"Opportunity {self.name!r}: opportunistic must be an age, "
                "0 or more (inf for never), or a dict of an age by "
                f"component, at least one; got {given!r}."
            )
        object.__setattr__(
            self,
            "opportunistic",
            (
                tuple((node, float(a)) for node, a in given.items())
                if isinstance(given, dict)
                else float(given)
            ),
        )

    def _check_model_time(self) -> None:
        for key in ("then_every", "also_every", "at"):
            if getattr(self, key) is not None:
                raise ValueError(
                    f"Opportunity {self.name!r}: {key}= is for real dates; "
                    "this one is in model time."
                )
        if self.period is not None:
            if not (
                checks.is_number(self.period)
                and 0.0 < float(self.period) < math.inf
            ):
                raise ValueError(
                    f"Opportunity {self.name!r}: the period must be a "
                    f"number, more than 0; got {self.period!r}."
                )
            if self.dates is not None:
                raise ValueError(
                    f"Opportunity {self.name!r}: give a period (with an "
                    "offset) or dates, one of the two."
                )
            offset = 0.0 if self.offset is None else self.offset
            if not (checks.is_number(offset) and math.isfinite(offset)):
                raise ValueError(
                    f"Opportunity {self.name!r}: the offset must be a "
                    f"finite number; got {self.offset!r}."
                )
            object.__setattr__(self, "period", float(self.period))
            object.__setattr__(self, "offset", float(offset))
        else:
            if self.offset not in (None, 0.0):
                raise ValueError(
                    f"Opportunity {self.name!r}: an offset applies only to "
                    "a period; dates give the times themselves."
                )
            object.__setattr__(self, "offset", 0.0)
            dates = checks.real_array(self.dates, "dates")
            if dates.ndim != 1 or dates.size == 0:
                raise ValueError(
                    f"Opportunity {self.name!r}: dates must be a list of "
                    f"times, at least one; got {self.dates!r}."
                )
            if not np.all(np.isfinite(dates)):
                raise ValueError(
                    f"Opportunity {self.name!r}: dates must be finite; got "
                    f"{self.dates!r}."
                )
            object.__setattr__(
                self, "dates", tuple(float(d) for d in np.unique(dates))
            )
        if self.duration is not None:
            if not (
                checks.is_number(self.duration)
                and 0.0 < float(self.duration) < math.inf
            ):
                raise ValueError(
                    f"Opportunity {self.name!r}: in model time a duration "
                    "is a number in the life's unit, more than 0; got "
                    f"{self.duration!r}."
                )
            object.__setattr__(self, "duration", float(self.duration))

    def _check_dates(self) -> None:
        from repyability.maintenance._shutdowns import _duration, check_when

        if self.name in _NOT_NAMES_ON_DATES:
            raise ValueError(
                f"{self.name!r} names what is not an opportunity (a planned "
                "replacement on any other day, or a failure): give the "
                "opportunity another name."
            )
        if self.offset not in (None, 0.0):
            raise ValueError(
                f"Opportunity {self.name!r}: an offset is for a period in "
                "model time."
            )
        object.__setattr__(self, "offset", 0.0)
        dates, every, at = check_when(
            f"Opportunity {self.name!r}",
            self.name,
            rule=self.rule,
            dates=self.dates,
            then_every=self.then_every,
            every=self.every,
            also_every=self.also_every,
            at=self.at,
        )
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "every", every)
        object.__setattr__(self, "at", at)
        object.__setattr__(self, "duration", _duration(self.duration))

    # -- made inside the package ---------------------------------------

    def _replaced(self, **changes) -> "Opportunity":
        """A copy with ``changes`` (fields already checked), made without
        the deprecation the public form of a price warns of."""
        new = object.__new__(Opportunity)
        for f in fields(Opportunity):
            object.__setattr__(
                new, f.name, changes.get(f.name, getattr(self, f.name))
            )
        return new

    @classmethod
    def _at_price(
        cls, name, cost, *, period=None, offset=0.0, dates=None
    ) -> "Opportunity":
        """An opportunity in model time with a price, as the solvers take
        it (no deprecation: the price is the part's, worked out here)."""
        return cls(name, period=period, offset=offset, dates=dates)._replaced(
            cost=_price(cost, f"The cost of {name!r}")
        )

    @classmethod
    def _from_shutdown(cls, shut) -> "Opportunity":
        """A shut of the 0.19 form (``Shutdown``) as an opportunity on real
        dates, its ``replace_cost`` kept as the (deprecated) price."""
        return cls(
            shut.name,
            shut.rule,
            dates=shut.dates,
            then_every=shut.then_every,
            every=shut.every,
            also_every=shut.also_every,
            duration=shut.duration,
            at=shut.at,
        )._replaced(cost=float(shut.replace_cost))

    def _shutdown(self, price: float):
        """This opportunity (on real dates) as the site solvers take it: a
        shut at ``price``."""
        from repyability.maintenance._shutdowns import Shutdown

        return Shutdown(
            self.name,
            self.rule,
            dates=self.dates,
            then_every=self.then_every,
            every=self.every,
            also_every=self.also_every,
            replace_cost=price,
            duration=self.duration,
            at=self.at,
        )

    # -- what it is ----------------------------------------------------

    @property
    def model_time(self) -> bool:
        """Whether it is in model time (a period, or dates as numbers);
        False on real dates."""
        return self.period is not None or (
            self.dates is not None and isinstance(self.dates[0], float)
        )

    @property
    def periodic(self) -> bool:
        """Whether its times are on a period in model time (not given
        dates)."""
        return self.period is not None

    @property
    def by_use(self) -> bool:
        """Whether it recurs by use (``every=Usage(...)``)."""
        from repyability.schedule import Usage

        return isinstance(self.every, Usage)

    def when(self) -> str:
        """When it happens, in words."""
        if self.model_time:
            if self.period is not None:
                text = f"every {self.period:g}"
                return text + (f" from {self.offset:g}" if self.offset else "")
            return "at " + ", ".join(f"{d:g}" for d in self.dates or ())
        return self._shutdown(0.0).when()

    def describe(self) -> str:
        """It in a few words: ``"weekly stop (every Saturday, 8 h)"``."""
        from repyability.maintenance._shutdowns import _money, _number

        words = [self.when()]
        if self.duration is not None:
            words.append(
                f"{self.duration:g} long"
                if self.model_time
                else f"{_number(self.duration)} h"
            )
        if self.takes_down is not None:
            words.append("takes down " + ", ".join(self.takes_down))
        if self.cost is not None:
            words.append(_money(self.cost))
        return f"{self.name} ({', '.join(words)})"

    @classmethod
    def from_recurrence(
        cls,
        name: str,
        rule,
        start,
        end=None,
        *legacy,
        unit: str = "hours",
        usage=None,
        at=None,
        time_zone: Optional[str] = None,
        duration=None,
        takes_down=None,
        cost=None,
    ) -> "Opportunity":
        """An opportunity at a rule's dates, laid on the part's model time.

        The rule's dates from ``start`` to ``end`` (both included), each
        at the clock time ``at`` (midnight by default), become the
        opportunity's ``dates``: with ``usage`` (a
        [`UsageForecast`][repyability.schedule.UsageForecast]), the
        operating time since the forecast's start, for a life in operating
        hours; without, the real time since ``start``'s midnight, for a
        life in calendar time. Dates at one operating time (shuts while
        the plant is idle) become one time. (To keep the rule on real
        dates, give it to ``Opportunity`` itself.)

        Parameters
        ----------
        name : str
            The opportunity's name.
        rule : Recurrence
            When the shuts are
            ([`repyability.schedule.Recurrence`][repyability.schedule.Recurrence]).
        start : datetime.date or str
            The first date to take.
        end : datetime.date or str
            The last date to take (included).
        unit : str, optional
            The life's unit, by default ``"hours"``.
        usage : UsageForecast, optional
            The map from dates to operating time; by default none
            (calendar time).
        at : str or datetime.time, optional
            The clock time each shut starts, by default midnight.
        time_zone : str, optional
            The site's IANA zone with no ``usage`` (with one, its zone).
        duration : float, optional
            How long each lasts, in the life's unit.
        takes_down : list of str, optional
            What it takes down, as for ``Opportunity``.
        cost : float, optional
            **Deprecated** (also as the second argument, the 0.14 form
            ``from_recurrence(name, cost, rule, start, end)``): a price on
            the opportunity; give the part ``opportunity_cost`` instead.

        Returns
        -------
        Opportunity

        Examples
        --------
        The first Tuesday of November, for a life in days, from 14
        November 2026:

        >>> from repyability import Opportunity
        >>> from repyability.schedule import Recurrence
        >>> overhaul = Opportunity.from_recurrence(
        ...     "overhaul",
        ...     Recurrence.yearly(month=11, weekday="Tuesday", nth=1),
        ...     "2026-11-14", "2029-12-31", unit="days",
        ... )
        >>> overhaul.dates
        (353.0, 724.0, 1088.0)
        """
        from repyability.schedule._forecast import opportunity_times

        if checks.is_number(rule):
            # The 0.14 form, from_recurrence(name, cost, rule, start, end).
            if cost is not None or len(legacy) != 1:
                raise TypeError(
                    "Opportunity.from_recurrence(name, cost, rule, start, "
                    "end) takes the price once and all five arguments."
                )
            cost, rule, start, end = rule, start, end, legacy[0]
            _price_deprecated("Opportunity.from_recurrence(name, cost, ...)")
        elif legacy:
            raise TypeError(
                "Opportunity.from_recurrence takes name, rule, start and "
                "end; give the rest by name."
            )
        elif cost is not None:
            _price_deprecated("Opportunity.from_recurrence(..., cost=)")
        if end is None:
            raise TypeError(
                "Opportunity.from_recurrence needs the last date: end=."
            )
        times = opportunity_times(
            rule,
            start,
            end,
            unit=unit,
            usage=usage,
            at=at,
            time_zone=time_zone,
        )
        if len(times) == 0:
            raise ValueError(
                f"Opportunity {name!r}: the rule ({rule.describe()}) gives "
                f"no dates from {start} to {end}."
            )
        made = cls(
            name,
            dates=tuple(float(t) for t in times),
            duration=duration,
            takes_down=takes_down,
        )
        return (
            made
            if cost is None
            else made._replaced(cost=_price(cost, f"The cost of {name!r}"))
        )

    def times(self, start: float, end: float) -> np.ndarray:
        """Its times from ``start`` up to ``end``, in model time.

        Parameters
        ----------
        start : float
            The first time to give, if one falls there.
        end : float
            The time to give them up to (not including).

        Returns
        -------
        numpy.ndarray
            Its times, in order.
        """
        if not self.model_time:
            raise ValueError(
                f"Opportunity {self.name!r} is on real dates: its times in "
                "model time come from a part's clock (a policy lays them "
                "there); give dates in model time, or a period."
            )
        if self.period is None:
            dates = np.asarray(self.dates, float)
            return dates[(dates >= start) & (dates < end)]
        first = math.ceil((start - self.offset) / self.period)
        last = math.ceil((end - self.offset) / self.period)
        times = self.offset + np.arange(first, last) * self.period
        return times[(times >= start) & (times < end)]

    # -- saving --------------------------------------------------------

    def to_dict(self) -> dict:
        """The opportunity as plain data, ready for ``json.dumps``
        (``"kind"`` ``"opportunity"``)."""
        from repyability.schedule import Usage

        out: dict = {"kind": "opportunity", "name": self.name}
        if self.model_time:
            out.update(
                time="model",
                period=self.period,
                offset=self.offset,
                dates=None if self.dates is None else list(self.dates),
                duration=self.duration,
            )
        else:
            every: Any = self.every
            if isinstance(every, Usage):
                every = every.to_dict()
            out.update(
                time="dates",
                rule=None if self.rule is None else self.rule.to_dict(),
                dates=(
                    None
                    if self.dates is None
                    else [d.isoformat() for d in self.dates]
                ),
                then_every=self.then_every,
                every=every,
                also_every=self.also_every,
                duration_hours=self.duration,
                at=None if self.at is None else self.at.strftime("%H:%M"),
            )
        out["takes_down"] = (
            None if self.takes_down is None else list(self.takes_down)
        )
        if self.cost is not None:
            out["cost"] = self.cost
        if not self.idle_ageing:
            out["idle_ageing"] = False
        if self.opportunistic is not None:
            out["opportunistic"] = (
                [[node, age] for node, age in self.opportunistic]
                if isinstance(self.opportunistic, tuple)
                else self.opportunistic
            )
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Opportunity":
        """An opportunity from its ``to_dict()``.

        Parameters
        ----------
        data : dict
            What ``to_dict()`` gave.

        Returns
        -------
        Opportunity
        """
        from repyability.schedule import Recurrence, Usage

        if not isinstance(data, dict) or data.get("kind") != "opportunity":
            raise ValueError(
                "Not a saved Opportunity: its kind must be 'opportunity'; "
                f"got {data!r:.80}."
            )
        if data.get("time") == "model":
            made = cls(
                data["name"],
                period=data.get("period"),
                offset=data.get("offset") if data.get("period") else None,
                dates=data.get("dates"),
                duration=data.get("duration"),
                takes_down=data.get("takes_down"),
            )
        else:
            every = data.get("every")
            if isinstance(every, dict):
                every = Usage.from_dict(every)
            rule = data.get("rule")
            made = cls(
                data["name"],
                None if rule is None else Recurrence.from_dict(rule),
                dates=data.get("dates"),
                then_every=data.get("then_every"),
                every=every,
                also_every=data.get("also_every"),
                duration=data.get("duration_hours"),
                at=data.get("at"),
                takes_down=data.get("takes_down"),
            )
        if data.get("cost") is not None:
            made = made._replaced(cost=_price(data["cost"], "cost"))
        opportunistic = data.get("opportunistic")
        if isinstance(opportunistic, list):
            opportunistic = {node: age for node, age in opportunistic}
        if opportunistic is not None or "idle_ageing" in data:
            made = made._replaced(
                idle_ageing=data.get("idle_ageing", True),
                opportunistic=opportunistic,
            )
            made._check_diagram_options()
        return made


def _numbers(values) -> bool:
    """Whether ``values`` (dates, as given) are numbers (model time), not
    dates."""
    if isinstance(values, (str, date)):
        return False
    try:
        listed = list(values)
    except TypeError:
        return checks.is_number(values)
    return all(checks.is_number(v) for v in listed)


class Calendar:
    """The prices of replacing a part over time, as the model-time solvers
    take them: its opportunities, each with the part's price, a planned
    replacement on any day, and a failure. A policy lays a part's prices
    on [`Opportunities`][repyability.Opportunities] as one of these;
    ``repyability.Calendar``, which a caller builds with the prices on the
    opportunities, is deprecated.

    Parameters
    ----------
    opportunities : list of Opportunity
        The opportunities in model time, each with its own name and price.
    failure : float or "part", optional
        What a replacement at a failure costs. By default ``"part"``: the
        part's ``unplanned_cost``.
    anytime : float, "part" or None, optional
        What a planned replacement on any day costs. By default
        ``"part"``: the part's ``planned_cost``, if it has one (else no
        replacement on any day). None: no replacement but at the
        opportunities (and at failures).

    Examples
    --------
    >>> from repyability.maintenance._calendar import Calendar, Opportunity
    >>> calendar = Calendar(
    ...     [
    ...         Opportunity._at_price("weekly", 3.0, period=7.0),
    ...         Opportunity._at_price("annual", 1.0, period=364.0),
    ...     ],
    ...     failure=40.0,
    ...     anytime=4.0,
    ... )
    >>> calendar.periodic, calendar.period
    (True, 364.0)
    >>> calendar.planned_cost([0.0, 3.0, 7.0]).tolist()
    [1.0, 4.0, 3.0]
    """

    def __init__(self, opportunities=(), failure=PART, anytime=PART):
        if isinstance(opportunities, Opportunity):
            opportunities = [opportunities]
        opportunities = tuple(opportunities)
        for opportunity in opportunities:
            if not isinstance(opportunity, Opportunity):
                raise TypeError(
                    "A calendar's opportunities must each be an "
                    f"Opportunity; got {opportunity!r}."
                )
            if not opportunity.model_time:
                raise ValueError(
                    f"Opportunity {opportunity.name!r} is on real dates, "
                    "and a Calendar is in model time: give the "
                    "opportunities to Opportunities."
                )
            if opportunity.cost is None:
                raise ValueError(
                    f"Opportunity {opportunity.name!r} has no price, and a "
                    "Calendar (deprecated) takes each opportunity's own: "
                    "give the opportunities to Opportunities, which takes "
                    "the prices from the part (its opportunity_cost)."
                )
        names = [o.name for o in opportunities]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"Opportunity names must differ; {repeated} repeat."
            )
        self.opportunities = opportunities
        self.failure = (
            PART if _is_part(failure) else _price(failure, "failure")
        )
        if anytime is None or _is_part(anytime):
            self.anytime = anytime
        else:
            self.anytime = _price(anytime, "anytime")

    def __repr__(self) -> str:
        return (
            f"Calendar({list(self.opportunities)!r}, failure="
            f"{self.failure!r}, anytime={self.anytime!r})"
        )

    @property
    def periodic(self) -> bool:
        """Whether every opportunity is on a period (no given dates), so
        that the calendar repeats."""
        return all(o.periodic for o in self.opportunities)

    @property
    def period(self) -> Optional[float]:
        """The time after which a periodic calendar repeats: the least
        common multiple of its periods (each read as a fraction), which
        may be very long; None if it has dates, or no opportunities."""
        if not self.periodic or not self.opportunities:
            return None
        return float(_common_multiple(o.period for o in self.opportunities))

    @property
    def plan_end(self) -> Optional[float]:
        """The last of the given dates; None if it has none."""
        dated = [o.dates[-1] for o in self.opportunities if not o.periodic]
        return max(dated) if dated else None

    def _repeat(self) -> Optional[float]:
        """The period a plan of given dates repeats with past its last
        date: from its first date to its last, and the mean gap between
        them (the date itself, with one); None with no dates."""
        dated = sorted(
            d for o in self.opportunities if not o.periodic for d in o.dates
        )
        if not dated:
            return None
        first, last = dated[0], dated[-1]
        if len(dated) == 1:
            if not last > 0.0:
                raise ValueError(
                    "A plan of one date repeated after it must have it "
                    f"after 0; it is {last:g}."
                )
            return last
        return (last - first) * len(dated) / (len(dated) - 1)

    def planned_cost(self, times, step: Optional[float] = None) -> np.ndarray:
        """The cheapest planned replacement at each of ``times``.

        Parameters
        ----------
        times : float or array_like
            The times, in the life's unit.
        step : float, optional
            A grid's step: an opportunity then counts at the grid time
            nearest it, as the policy lays it. Without one, only at its
            time itself.

        Returns
        -------
        numpy.ndarray
            The price of the cheapest planned replacement open at each
            time (an opportunity there, or the calendar's own ``anytime``
            price), ``inf`` where none is.
        """
        times = checks.real_array(times, "times")
        out = np.full(times.shape, math.inf)
        if checks.is_number(self.anytime):
            out[:] = float(self.anytime)
        flat = np.ravel(times)
        if flat.size == 0:
            return out
        for opportunity in self.opportunities:
            near = opportunity.times(
                float(flat.min()) - (step or 0.0) - 1.0,
                float(flat.max()) + (step or 0.0) + 1.0,
            )
            if near.size == 0:
                continue
            at = np.clip(np.searchsorted(near, flat), 1, max(near.size - 1, 1))
            distance = (
                np.minimum(
                    np.abs(near[at - 1] - flat), np.abs(near[at] - flat)
                )
                if near.size > 1
                else np.abs(near[0] - flat)
            )
            if step is None:
                hit = distance <= 1e-12 * np.maximum(1.0, np.abs(flat))
            else:
                hit = distance <= 0.5 * float(step) * (1.0 + 1e-12)
            cheaper = hit & (opportunity.cost < out.reshape(-1))
            out.reshape(-1)[cheaper] = opportunity.cost
        return out

    def _prices(self, part) -> Tuple[float, Optional[float]]:
        """The failure and anytime prices, with the part's for
        ``"part"``."""
        if _is_part(self.failure):
            if not hasattr(part, "cu"):
                raise ValueError(
                    "The calendar takes the failure price from the part, "
                    "which has none: give the calendar a failure price, or "
                    "the part an unplanned_cost."
                )
            failure = float(part.cu)
        else:
            failure = float(self.failure)
        if _is_part(self.anytime):
            anytime = float(part.cp) if hasattr(part, "cp") else None
        else:
            anytime = self.anytime
        return failure, anytime


def _is_part(value) -> bool:
    return isinstance(value, str) and value == PART


class _Grid:
    """A calendar laid on a grid of ``step`` from time 0: the cheapest
    planned price at each grid time, and which kind of replacement it is.

    ``tail`` extends given dates past the plan: ``"repeat"`` repeats them
    every ``repeat`` (``Calendar._repeat``), forward only; ``"anytime"``
    leaves only the replacement on any day after them; None leaves them as
    they are. ``cycle`` is the number of steps after which the grid
    repeats, from ``cycle_start`` (None if it does not).
    """

    def __init__(self, calendar, anytime, step, tail=None):
        self.calendar, self.anytime, self.step = calendar, anytime, step
        self.names = [o.name for o in calendar.opportunities] + [ANYTIME]
        self.tail = tail
        end = calendar.plan_end
        self.plan_end = end
        self.repeat = calendar._repeat() if tail == "repeat" else None
        # The grid index from which the prices repeat every cycle: the
        # plan's first date, for a repeated plan.
        self.cycle_start = 0
        if self.repeat is not None:
            first = min(
                d
                for o in calendar.opportunities
                if not o.periodic
                for d in o.dates
            )
            self.cycle_start = max(int(np.rint(first / step)), 0)
        periods = [o.period for o in calendar.opportunities if o.periodic]
        if self.repeat is not None:
            periods.append(self.repeat)
        self.cycle: Optional[int]
        if calendar.opportunities and (
            calendar.periodic or self.repeat is not None
        ):
            self.cycle = int(
                _common_multiple([p / step for p in periods] + [1.0])
            )
        elif not calendar.opportunities:
            self.cycle = 1
        else:
            self.cycle = None
        self._warn_moved()

    def _warn_moved(self) -> None:
        """Warn of an opportunity whose times are not on the grid."""
        for o in self.calendar.opportunities:
            if o.periodic:
                times = o.offset + np.arange(64) * o.period
            else:
                times = np.asarray(o.dates, float)
            moved = np.abs(times - np.rint(times / self.step) * self.step)
            if moved.max() > _MOVED * self.step:
                warnings.warn(
                    f"The times of opportunity {o.name!r} are not on the "
                    f"grid of step {self.step:g}: each is taken at the "
                    f"grid time nearest it (moved by up to "
                    f"{moved.max():.3g}).",
                    stacklevel=outside_level(),
                )

    def _dates(self, o, start: float, end: float) -> np.ndarray:
        if o.periodic:
            return o.times(start, end)
        dates = np.asarray(o.dates, float)
        if self.repeat is None:
            return dates[(dates >= start) & (dates < end)]
        # Repeated forward from the plan, not before it.
        first = max(math.floor((start - dates.max()) / self.repeat), 0)
        last = math.ceil((end - dates.min()) / self.repeat)
        shifts = np.arange(first, last + 1) * self.repeat
        times = (dates[None, :] + shifts[:, None]).ravel()
        return times[(times >= start) & (times < end)]

    def prices(self, start: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """The cheapest planned price at grid times ``start`` to
        ``start + n - 1`` (``inf`` where there is none), and the index in
        ``names`` of the kind of replacement it is (-1 where none)."""
        step = self.step
        costs = np.full(n, math.inf)
        kinds = np.full(n, -1, dtype=np.int64)
        anytime = self.anytime
        if anytime is not None:
            costs[:] = anytime
            kinds[:] = len(self.names) - 1
        low, high = (start - 0.5) * step, (start + n - 0.5) * step
        for k, o in enumerate(self.calendar.opportunities):
            times = self._dates(o, low - step, high + step)
            index = np.rint(times / step).astype(np.int64) - start
            index = index[(index >= 0) & (index < n)]
            cheaper = o.cost < costs[index]
            costs[index[cheaper]] = o.cost
            kinds[index[cheaper]] = k
        return costs, kinds
