"""The opportunities to replace a part, in model time (#319, UC-13 of this
package's ``README.md``): shuts on a period (weekly, quarterly, annual,
each with its own price and offset) or on given times, a planned
replacement on any day at a dearer price, and a failure at the dearest.

Times are numbers in the life model's unit, from 0; turning dates into
them is Reliafy's (``CLAUDE.md``: RePyability is about timelines). A
calendar is laid on a grid of a step for the dynamic programme
(``_Grid``): each date to its nearest grid time, so a date moves by half a
step at most and the dates of a period keep no drift.
"""

import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

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


@dataclass(frozen=True)
class Opportunity:
    """A kind of scheduled opportunity to replace a part: a shut (weekly,
    quarterly, annual, ...) at which replacing it costs ``cost``.

    Its times are either every ``period`` from ``offset`` (on and on, both
    ways: ``offset + k * period`` for every whole ``k``), or the given
    ``dates``. Times are in the life model's unit, from 0.

    Parameters
    ----------
    name : str
        The opportunity's name, as the policy's schedule lists it (not
        ``"anytime"``, which names a replacement on any day).
    cost : float
        What replacing the part at it costs, at least 0.
    period : float, optional
        The time from one to the next, more than 0.
    offset : float, optional
        The time of one of them, by default 0. Only with ``period``.
    dates : array_like, optional
        Its times, instead of ``period``: finite numbers, put in order.

    Examples
    --------
    >>> from repyability import Opportunity
    >>> weekly = Opportunity("weekly", 3.0, period=7.0)
    >>> annual = Opportunity("annual", 1.0, period=364.0, offset=182.0)
    >>> turnarounds = Opportunity("turnaround", 0.5, dates=[400.0, 1130.0])
    >>> weekly.times(0.0, 30.0).tolist()
    [0.0, 7.0, 14.0, 21.0, 28.0]
    """

    name: str
    cost: float
    period: Optional[float] = None
    offset: float = 0.0
    dates: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(
                f"An opportunity's name must be text; got {self.name!r}."
            )
        if self.name == ANYTIME:
            raise ValueError(
                f"{ANYTIME!r} names a planned replacement on any day (the "
                "calendar's anytime price): give the opportunity another "
                "name."
            )
        object.__setattr__(
            self, "cost", _price(self.cost, f"The cost of {self.name!r}")
        )
        if (self.period is None) == (self.dates is None):
            raise ValueError(
                f"Opportunity {self.name!r}: give a period (with an "
                "offset) or dates, one of the two."
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
            if not (
                checks.is_number(self.offset)
                and math.isfinite(float(self.offset))
            ):
                raise ValueError(
                    f"Opportunity {self.name!r}: the offset must be a "
                    f"finite number; got {self.offset!r}."
                )
            object.__setattr__(self, "period", float(self.period))
            object.__setattr__(self, "offset", float(self.offset))
        else:
            if self.offset != 0.0:
                raise ValueError(
                    f"Opportunity {self.name!r}: an offset applies only to "
                    "a period; dates give the times themselves."
                )
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

    @property
    def periodic(self) -> bool:
        """Whether its times are on a period (not given dates)."""
        return self.period is not None

    def times(self, start: float, end: float) -> np.ndarray:
        """Its times from ``start`` up to ``end``.

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
        if self.period is None:
            dates = np.asarray(self.dates, float)
            return dates[(dates >= start) & (dates < end)]
        first = math.ceil((start - self.offset) / self.period)
        last = math.ceil((end - self.offset) / self.period)
        times = self.offset + np.arange(first, last) * self.period
        return times[(times >= start) & (times < end)]


class Calendar:
    """The prices of replacing a part over time: its scheduled
    opportunities, a planned replacement on any day, and a failure.

    Parameters
    ----------
    opportunities : list of Opportunity
        The scheduled opportunities, each with its own name.
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
    >>> from repyability import Calendar, Opportunity
    >>> calendar = Calendar(
    ...     [
    ...         Opportunity("weekly", 3.0, period=7.0),
    ...         Opportunity("annual", 1.0, period=364.0),
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
