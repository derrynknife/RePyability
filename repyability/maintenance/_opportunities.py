"""Opportunities to replace a part, and the part's prices at them (UC-45
of this package's ``README.md``).

An [`Opportunity`][repyability.Opportunity] says when a stop or shut
happens, how long it lasts and what it takes down; it carries no price,
as it applies to every part. [`Opportunities`][repyability.Opportunities]
collects a site's (or a calendar's, in model time), and says whether a
planned replacement may be made on any other day. Each part brings its
own prices (`NonRepairable`'s ``planned_cost``, ``unplanned_cost``,
``opportunity_cost``, ``opportunity_costs`` and ``job_duration``), and a
policy lays them on the opportunities here (``resolve``): as the
model-time solvers' ``Calendar``, or the real-date solvers' site, each
opportunity with the part's price. No numerical method is touched.

The 0.14 and 0.19 forms, which put the prices on the opportunities
(``Calendar``, ``Shutdown``, ``Shutdowns``), still work, deprecated: their
prices are taken as the part's overrides for those opportunities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from repyability.maintenance import _calendar, _shutdowns
from repyability.maintenance._calendar import PART, Opportunity, _price
from repyability.rbd.results import _ResultMapping
from repyability.utils import checks
from repyability.utils.deprecation import price_on_part, replaced

#: The kinds of row a price list has.
OPPORTUNITY = "opportunity"
ANY_DAY = "any day"
FAILURE = "failure"

#: Where a price came from.
OVERRIDE = "override"
ON_OPPORTUNITY = "opportunity"
OPPORTUNITY_COST = "opportunity_cost"
PLANNED_COST = "planned_cost"
UNPLANNED_COST = "unplanned_cost"
COLLECTION = "collection"
NOT_USED = "not used"

#: The keyword arguments that only a site on real dates takes.
_SITE_ONLY = ("plan", "now", "readings", "unit", "horizon", "discount")


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class OpportunityPrice(_ResultMapping):
    """One row of [`OpportunityPrices`][repyability.OpportunityPrices]:
    the price a part pays at an opportunity, on any other day or at a
    failure, and where it came from.

    Attributes
    ----------
    name : str
        The opportunity's name, ``"any day"`` or ``"failure"``.
    kind : str
        ``"opportunity"``, ``"any day"`` or ``"failure"``.
    price : float
        The price; ``nan`` where it is not used.
    used : bool
        Whether the policy may replace the part there.
    source : str
        Where the price came from: ``"override"`` (the part's
        ``opportunity_costs``), ``"opportunity_cost"``,
        ``"planned_cost"`` (an opportunity priced as any other day, as the
        part has no ``opportunity_cost``), ``"unplanned_cost"``,
        ``"opportunity"`` or ``"collection"`` (a deprecated price on the
        opportunity or on the collection), ``"not used"``; from
        ``NonRepairable.opportunity_prices``, the formula's terms.
    reason : str
        The same in words, and why a row is not used ("the 6 h job
        doesn't fit in the 4 h weekly stop").
    """

    name: str
    kind: str
    price: float
    used: bool
    source: str
    reason: str


@dataclass(frozen=True)
class OpportunityPrices(_ResultMapping):
    """A part's prices at each opportunity, on any other day and at a
    failure, one row each: what a policy priced the opportunities at (its
    ``prices``), or what
    [`NonRepairable.opportunity_prices`][repyability.NonRepairable.opportunity_prices]
    works out from a replacement's cost and downtime.

    Attributes
    ----------
    rows : tuple of OpportunityPrice
        The opportunities in order, then ``"any day"`` and ``"failure"``.
    planned_cost : float or None
        From ``opportunity_prices``: the part's price on any day. None
        on a policy's prices.
    unplanned_cost : float or None
        From ``opportunity_prices``: the part's price at a failure.
    opportunity_cost : float or None
        From ``opportunity_prices``: the part's price at an opportunity
        the job fits in.
    job_duration : float or str or None
        From ``opportunity_prices``: how long the replacement takes.
    """

    rows: Tuple[OpportunityPrice, ...] = ()
    planned_cost: Optional[float] = None
    unplanned_cost: Optional[float] = None
    opportunity_cost: Optional[float] = None
    job_duration: Any = None

    def __repr__(self) -> str:
        return (
            "OpportunityPrices("
            + "; ".join(f"{r.name}: {_money_or_not(r)}" for r in self.rows)
            + ")"
        )

    def row(self, name: str) -> OpportunityPrice:
        """The row named ``name``.

        Parameters
        ----------
        name : str
            An opportunity's name, ``"any day"`` or ``"failure"``.

        Returns
        -------
        OpportunityPrice
        """
        for r in self.rows:
            if r.name == name:
                return r
        raise KeyError(
            f"No price for {name!r}; the rows are "
            f"{[r.name for r in self.rows]}."
        )

    def price(self, name: str) -> float:
        """The price at ``name``, ``nan`` where it is not used.

        Parameters
        ----------
        name : str
            An opportunity's name, ``"any day"`` or ``"failure"``.

        Returns
        -------
        float
        """
        return self.row(name).price

    @property
    def unused(self) -> Tuple[str, ...]:
        """The opportunities not used, by name (the rows say why)."""
        return tuple(
            r.name for r in self.rows if r.kind == OPPORTUNITY and not r.used
        )

    def describe(self) -> str:
        """The prices, a line each, with where each came from."""
        return "\n".join(
            f"{r.name}: {_money_or_not(r)} ({r.reason})" for r in self.rows
        )

    def apply(self, part):
        """Give ``part`` these prices (from
        [`NonRepairable.opportunity_prices`][repyability.NonRepairable.opportunity_prices]):
        its ``planned_cost`` (any day), ``unplanned_cost`` (a failure),
        ``opportunity_cost`` and ``job_duration``, its per-opportunity
        overrides cleared, so a policy prices every opportunity as these
        rows do. The downtime cost stays here: it is the system's, not
        the part's.

        Parameters
        ----------
        part : NonRepairable
            The part, changed in place (as
            ``set_costs_planned_and_unplanned`` changes it).

        Returns
        -------
        NonRepairable
            ``part``.
        """
        if self.planned_cost is None or self.unplanned_cost is None:
            raise ValueError(
                "These are a policy's prices, already the part's: only "
                "NonRepairable.opportunity_prices(...) gives prices to "
                "apply."
            )
        part.set_costs_planned_and_unplanned(
            self.planned_cost, self.unplanned_cost
        )
        part._set_opportunity_prices(
            opportunity_cost=self.opportunity_cost,
            job_duration=self.job_duration,
            opportunity_costs=None,
        )
        return part


def _money_or_not(row: OpportunityPrice) -> str:
    if not row.used or not math.isfinite(row.price):
        return "not used"
    return _shutdowns._money(row.price)


def with_prices(policy, prices: OpportunityPrices):
    """``policy`` (a ``CalendarPolicy``, ``ShutdownPolicy`` or
    ``RuleCost``) carrying how the opportunities were priced."""
    object.__setattr__(policy, "_prices", prices)
    return policy


# --------------------------------------------------------- the collection


class Opportunities:
    """A site's opportunities to replace parts (or a calendar's, in model
    time), as one record: each kind of stop or shut, whether a planned
    replacement may be made on any other day, and the site's time zone.

    It carries no prices: each part brings its own (`NonRepairable`'s
    ``planned_cost`` on any day, ``unplanned_cost`` at a failure,
    ``opportunity_cost`` at an opportunity, ``opportunity_costs`` for one
    opportunity, and ``job_duration``), so one record serves every part
    ([`NonRepairable.optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy]).
    It is in model time or on real dates, by what its opportunities are;
    a mix is refused. It replaces ``Calendar`` (model time, 0.14) and
    ``Shutdowns`` (real dates, 0.19), both deprecated.

    Parameters
    ----------
    opportunities : list of Opportunity
        The kinds of opportunity, each with its own name.
    any_day : bool, optional
        Whether a planned replacement may be made on any other day, at the
        part's ``planned_cost``; by default True. False: only at the
        opportunities (and at failures). (A price here is deprecated: it
        is taken as every part's price on any other day.)
    time_zone : str, optional
        On real dates: the site's IANA zone (``"Australia/Brisbane"``);
        None: clock times as they are, with no daylight saving.

    Examples
    --------
    >>> from repyability import Opportunities, Opportunity
    >>> from repyability.schedule import Recurrence
    >>> site = Opportunities(
    ...     [
    ...         Opportunity("weekly stop", Recurrence.weekly("Sat"),
    ...                     duration="8h"),
    ...         Opportunity("turnaround", dates=["2027-03-03"],
    ...                     duration="21d", takes_down=["Line 2"]),
    ...     ],
    ...     time_zone="Australia/Brisbane",
    ... )
    >>> site.model_time, site.names
    (False, ('weekly stop', 'turnaround'))
    >>> plant = Opportunities([Opportunity("stop", period=28.0)],
    ...                       any_day=False)
    >>> plant.model_time
    True
    """

    def __init__(self, opportunities=(), *, any_day=True, time_zone=None):
        from repyability.schedule._meters import zone

        if isinstance(opportunities, (Opportunity, _shutdowns.Shutdown)):
            opportunities = [opportunities]
        listed: List[Opportunity] = []
        for o in opportunities:
            if isinstance(o, _shutdowns.Shutdown):
                o = Opportunity._from_shutdown(o)
            if not isinstance(o, Opportunity):
                raise TypeError(
                    "Opportunities takes a list of Opportunity, one per "
                    f"kind of stop or shut; got {o!r}."
                )
            listed.append(o)
        names = [o.name for o in listed]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"Opportunity names must differ; {repeated} repeat."
            )
        model = [o.name for o in listed if o.model_time]
        dated = [o.name for o in listed if not o.model_time]
        if model and dated:
            raise ValueError(
                "Opportunities are all in model time (a period, or dates "
                "as numbers) or all on real dates (a rule, dates, or "
                f"every=); {_listed(model)} "
                f"{'is' if len(model) == 1 else 'are'} in model time and "
                f"{_listed(dated)} on real dates. Keep one collection per "
                "clock."
            )
        if model and time_zone is not None:
            raise ValueError(
                "time_zone is for opportunities on real dates; these are in "
                "model time."
            )
        zone(time_zone)
        self._any_day_price: Optional[float] = None
        if any_day is None:
            any_day = False
        if checks.is_number(any_day):
            price_on_part("Opportunities(any_day=<price>)")
            self._any_day_price = _price(any_day, "any_day")
            any_day = True
        if not isinstance(any_day, bool):
            raise TypeError(
                "any_day must be True (a planned replacement on any other "
                "day, at the part's planned_cost) or False; got "
                f"{any_day!r}."
            )
        self.opportunities: Tuple[Opportunity, ...] = tuple(listed)
        self.any_day: bool = any_day
        self.time_zone = time_zone
        #: A failure's price from a deprecated Calendar(failure=).
        self._failure_price: Optional[float] = None
        #: The clock: "model", "dates", or None (empty, and not said).
        self._clock: Optional[str] = (
            "model"
            if model
            else "dates" if dated or time_zone is not None else None
        )

    @classmethod
    def _legacy(
        cls, opportunities, *, any_day, any_day_price, failure, clock, zone
    ) -> "Opportunities":
        made = cls(opportunities, any_day=any_day, time_zone=zone)
        made._any_day_price = any_day_price
        made._failure_price = failure
        made._clock = clock
        return made

    # -- reading it ----------------------------------------------------

    @property
    def model_time(self) -> Optional[bool]:
        """True in model time, False on real dates; None for an empty
        collection with no time zone, which a policy takes as model time
        unless it is given a site's arguments (``now=``, ``plan=``, ...)."""
        return None if self._clock is None else self._clock == "model"

    @property
    def names(self) -> Tuple[str, ...]:
        """The opportunities' names, in order."""
        return tuple(o.name for o in self.opportunities)

    def __getitem__(self, name: str) -> Opportunity:
        for o in self.opportunities:
            if o.name == name:
                return o
        raise KeyError(
            f"No opportunity named {name!r}; these are {list(self.names)}."
        )

    def __iter__(self):
        return iter(self.opportunities)

    def __len__(self) -> int:
        return len(self.opportunities)

    def __repr__(self) -> str:
        return (
            f"Opportunities({list(self.opportunities)!r}, any_day="
            f"{self.any_day!r}, time_zone={self.time_zone!r})"
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Opportunities)
            and self.to_dict() == other.to_dict()
        )

    def __hash__(self) -> int:
        return hash(repr(self.to_dict()))

    def describe(self) -> str:
        """The collection in a sentence."""
        other = (
            "a planned replacement on any other day"
            if self.any_day
            else "no planned replacement on other days"
        )
        if not self.opportunities:
            return f"No opportunities; {other}."
        n = len(self.opportunities)
        listed = _shutdowns._listed([o.describe() for o in self.opportunities])
        return (
            f"{n} {'opportunity' if n == 1 else 'opportunities'}: {listed}; "
            f"{other}."
        )

    # -- on real dates -------------------------------------------------

    def _site(self, prices: Optional[Dict[str, float]] = None, any_day=None):
        """The opportunities on real dates as the site solvers take them:
        each a shut at its price in ``prices`` (0 where none is given, for
        what needs no price)."""
        if self._clock == "model":
            raise ValueError(
                "These opportunities are in model time; dates, a diagram's "
                "outages and a site's route are for opportunities on real "
                "dates."
            )
        prices = prices or {}
        return _shutdowns.Shutdowns(
            [o._shutdown(prices.get(o.name, 0.0)) for o in self.opportunities],
            any_day=any_day,
            time_zone=self.time_zone,
        )

    def dates(self, start, end, *, plan=None, readings=None):
        """Every opportunity from ``start`` to ``end`` (both included), by
        kind, as ``Shutdowns.dates`` gave them (real dates only).

        Parameters
        ----------
        start : date or str
            The first day.
        end : date or str
            The last day (included).
        plan : DutyPlan, optional
            How the equipment runs, for opportunities by use.
        readings : dict, optional
            The meters at ``start``, for opportunities by use.

        Returns
        -------
        dict
            Each kind's name and its dates, in order.
        """
        return self._site().dates(start, end, plan=plan, readings=readings)

    def on_diagram(self, start, end, **kwargs) -> Dict[str, dict]:
        """The opportunities as a diagram's outages (real dates only): the
        spec [`RepairableRBD(shutdowns=...)`][repyability.RepairableRBD]
        takes, one per kind, each lasting its ``duration``. The keyword
        arguments (``unit``, ``plan``, ``readings``, ``cost``, ``scope``,
        ``idle_ageing``) are as for ``Shutdowns.on_diagram``.

        Parameters
        ----------
        start : date or str
            The diagram's time 0 is ``start``'s midnight.
        end : date or str
            The last day taken (included).
        **kwargs
            As above.

        Returns
        -------
        dict
            ``{name: {"times": [...], "duration": ...}}``.
        """
        return self._site().on_diagram(start, end, **kwargs)

    def route(self, part, *, where=None, **kwargs):
        """How ``part``'s policy at this site will be worked out, before
        working it out (real dates only), as
        ``optimal_opportunity_policy`` would take it: the keyword
        arguments are its site arguments (``plan``, ``method``,
        ``horizon``, ``discount``, ``now``, ``readings``, ``unit``,
        ``failure_timing``, ``dated_until``).

        Parameters
        ----------
        part : NonRepairable
            The part, with its prices.
        where : str, optional
            As for ``optimal_opportunity_policy``.
        **kwargs
            As above.

        Returns
        -------
        ShutdownRoute
        """
        site, _ = resolve(part, self, where=where, on_dates=True)
        return site.route(part, **kwargs)

    # -- saving --------------------------------------------------------

    def to_dict(self) -> dict:
        """The collection as plain data, ready for ``json.dumps``
        (``"kind"`` ``"opportunities"``, with a ``"version"``)."""
        out: Dict[str, Any] = {
            "kind": "opportunities",
            "version": 1,
            "opportunities": [o.to_dict() for o in self.opportunities],
            "any_day": self.any_day,
            "time_zone": self.time_zone,
        }
        if self._any_day_price is not None:
            out["any_day_price"] = self._any_day_price
        if self._failure_price is not None:
            out["failure_price"] = self._failure_price
        if self._clock is not None:
            out["time"] = self._clock
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Opportunities":
        """A collection from its ``to_dict()``, or from a saved
        ``Shutdowns`` (``"kind"`` ``"shutdowns"``, its prices kept as the
        deprecated prices on the opportunities).

        Parameters
        ----------
        data : dict
            What ``to_dict()`` gave.

        Returns
        -------
        Opportunities
        """
        if isinstance(data, dict) and data.get("kind") == "shutdowns":
            return as_opportunities(_shutdowns.Shutdowns.from_dict(data))
        if not isinstance(data, dict) or data.get("kind") != "opportunities":
            raise ValueError(
                "Not saved Opportunities: its kind must be 'opportunities'; "
                f"got {data!r:.80}."
            )
        if data.get("version", 1) > 1:
            raise ValueError(
                "These Opportunities were saved by a newer RePyability "
                f"(version {data['version']}): upgrade to read them."
            )
        return cls._legacy(
            [Opportunity.from_dict(o) for o in data.get("opportunities", [])],
            any_day=bool(data.get("any_day", True)),
            any_day_price=data.get("any_day_price"),
            failure=data.get("failure_price"),
            clock=data.get("time"),
            zone=data.get("time_zone"),
        )


def _listed(names: List[str]) -> str:
    return _shutdowns._listed([repr(n) for n in names])


# ------------------------------------------------- the deprecated forms


class Calendar(_calendar.Calendar):
    """**Deprecated** (0.14's form; removed in 0.21): use
    [`Opportunities`][repyability.Opportunities], with the prices on the
    part. ``Calendar(opportunities, failure=, anytime=)`` still works and
    gives the same numbers: each opportunity's own price (its deprecated
    ``cost``) is the part's price there, ``failure`` the price at a
    failure (by default the part's ``unplanned_cost``) and ``anytime``
    the price on any day (by default the part's ``planned_cost``; None,
    no replacement on any day: ``Opportunities(any_day=False)``).

    Parameters
    ----------
    opportunities : list of Opportunity
        In model time, each with a price.
    failure : float or "part", optional
        The price at a failure.
    anytime : float, "part" or None, optional
        The price on any day.
    """

    def __init__(self, opportunities=(), failure=PART, anytime=PART):
        replaced(
            "Calendar",
            "Opportunities (in model time, with the prices on the part: "
            "planned_cost, unplanned_cost, opportunity_cost)",
            "an opportunity applies to every part, so its price comes with "
            "the part",
        )
        super().__init__(opportunities, failure=failure, anytime=anytime)


class Shutdown(_shutdowns.Shutdown):
    """**Deprecated** (0.19's form; removed in 0.21): use
    [`Opportunity`][repyability.Opportunity] (on real dates, with the same
    ``rule``, ``dates``, ``then_every``, ``every``, ``also_every``,
    ``duration`` and ``at``), with the price on the part.
    ``replace_cost`` is taken as the part's price at this shut.

    Parameters
    ----------
    name : str
        The shut's name.
    rule : Recurrence, optional
        When it happens.
    dates : list of date or str, optional
        Its fixed dates.
    then_every : str, optional
        The gap after the last date.
    every : str, float or Usage, optional
        A plain period or a use.
    also_every : str, optional
        A calendar limit on a use.
    replace_cost : float
        The price of replacing a part at this shut.
    duration : str or float, optional
        How long it lasts.
    at : str, optional
        The clock time it starts.
    """

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
        replaced(
            "Shutdown",
            "Opportunity (on real dates, with the price on the part: "
            "opportunity_cost, or opportunity_costs={name: price})",
            "an opportunity applies to every part, so its price comes with "
            "the part",
        )
        super().__init__(
            name,
            rule,
            dates=dates,
            then_every=then_every,
            every=every,
            also_every=also_every,
            replace_cost=replace_cost,
            duration=duration,
            at=at,
        )


class Shutdowns(_shutdowns.Shutdowns):
    """**Deprecated** (0.19's form; removed in 0.21): use
    [`Opportunities`][repyability.Opportunities] (``any_day=True`` for a
    planned replacement on any other day at the part's ``planned_cost``),
    with the prices on the part. ``any_day=<price>`` is still taken as the
    price on any other day, and each shut's ``replace_cost`` as the price
    at it.

    Parameters
    ----------
    shutdowns : list of Shutdown
        The kinds of shut.
    any_day : float or None, optional
        The price on any other day; None: none allowed.
    time_zone : str, optional
        The site's IANA zone.
    """

    def __init__(self, shutdowns=(), *, any_day=None, time_zone=None):
        replaced(
            "Shutdowns",
            "Opportunities (with the prices on the part: planned_cost on "
            "any day, opportunity_cost at the opportunities)",
            "an opportunity applies to every part, so its price comes with "
            "the part",
        )
        super().__init__(shutdowns, any_day=any_day, time_zone=time_zone)


# -------------------------------------------- the part's prices laid on


def as_opportunities(value) -> Opportunities:
    """``value`` as ``Opportunities``: itself, a list of ``Opportunity``,
    or a deprecated ``Calendar`` or ``Shutdowns`` (its prices kept as the
    deprecated prices on its opportunities and itself)."""
    if isinstance(value, Opportunities):
        return value
    if isinstance(value, _calendar.Calendar):
        anytime = value.anytime
        return Opportunities._legacy(
            list(value.opportunities),
            any_day=anytime is not None,
            any_day_price=(
                None
                if anytime is None or _calendar._is_part(anytime)
                else float(anytime)
            ),
            failure=(
                None
                if _calendar._is_part(value.failure)
                else float(value.failure)
            ),
            clock="model",
            zone=None,
        )
    if isinstance(value, _shutdowns.Shutdowns):
        return Opportunities._legacy(
            [Opportunity._from_shutdown(s) for s in value.shutdowns],
            any_day=value.any_day is not None,
            any_day_price=value.any_day,
            failure=None,
            clock="dates",
            zone=value.time_zone,
        )
    if isinstance(value, Opportunity):
        return Opportunities([value])
    if isinstance(value, (list, tuple)) and all(
        isinstance(o, Opportunity) for o in value
    ):
        return Opportunities(value)
    raise TypeError(
        "Give the opportunities as Opportunities([Opportunity(...), ...]); "
        f"got {value!r:.80}."
    )


def _overrides(part, collection: Opportunities) -> Dict[str, float]:
    """The part's per-opportunity prices by name, each key checked
    against the collection: a name or an ``Opportunity`` not in it is
    refused, with the names it has."""
    given = getattr(part, "opportunity_costs", None) or {}
    names = collection.names
    have = (
        f"these opportunities are {', '.join(map(repr, names))}"
        if names
        else "there are no opportunities"
    )
    out: Dict[str, float] = {}
    for key, price in given.items():
        if isinstance(key, Opportunity):
            if key not in collection.opportunities:
                raise ValueError(
                    f"The part's opportunity_costs prices the opportunity "
                    f"{key.name!r}, which is not one of these: {have}."
                )
            out[key.name] = float(price)
        elif key in names:
            out[key] = float(price)
        else:
            raise ValueError(
                f"The part's opportunity_costs names {key!r}, which is not "
                f"an opportunity here: {have}."
            )
    return out


def _job_length(job, clock: str) -> Optional[float]:
    """A part's ``job_duration`` in the opportunities' terms: hours on real
    dates (text or a number of hours), the life's unit in model time (a
    number)."""
    if job is None:
        return None
    if clock == "model":
        if not checks.is_number(job):
            raise ValueError(
                f"The part's job_duration {job!r} is text, which is for "
                "opportunities on real dates; in model time give it as a "
                "number in the life's unit."
            )
        return float(job)
    return _shutdowns._duration(job)


def _length_words(value: float, clock: str) -> str:
    if clock == "model":
        return f"{value:g}"
    return f"{_shutdowns._number(value)} h"


def _fit(o: Opportunity, job: Optional[float], clock: str) -> Optional[str]:
    """Why a job of ``job`` does not fit in ``o``; None where it fits
    (or either length is not said)."""
    if job is None or o.duration is None or o.duration >= job:
        return None
    words = (
        f"the {_length_words(job, clock)} job doesn't fit in the "
        f"{_length_words(o.duration, clock)} {o.name}"
    )
    if clock == "model":
        words += " (in the life's unit)"
    return words


def resolve(part, value, *, where=None, on_dates=False):
    """The part's prices laid on the opportunities: the solvers' own
    record (a ``Calendar`` in model time, a ``Shutdowns`` site on real
    dates), each opportunity used at the part's price, and the prices
    (``OpportunityPrices``) with where each came from.

    An opportunity's price is the part's override for it
    (``opportunity_costs``), else a deprecated price on the opportunity,
    else the part's ``opportunity_cost``, else its ``planned_cost`` (as
    any other day). One that does not take ``where`` down, or that the
    part's job does not fit in, is left out, and listed as not used.
    ``on_dates`` says an empty collection that names no clock is on real
    dates (the policy was given a site's arguments).
    """
    collection = as_opportunities(value)
    clock = collection._clock or ("dates" if on_dates else "model")
    if where is not None and not isinstance(where, str):
        raise TypeError(f"where must be a label (text); got {where!r}.")
    overrides = _overrides(part, collection)
    job = _job_length(getattr(part, "job_duration", None), clock)
    general = getattr(part, "opportunity_cost", None)
    planned = getattr(part, "cp", None)
    rows: List[OpportunityPrice] = []
    kept: List[Tuple[Opportunity, float]] = []
    for o in collection.opportunities:
        if (
            where is not None
            and o.takes_down is not None
            and (where not in o.takes_down)
        ):
            rows.append(
                OpportunityPrice(
                    o.name,
                    OPPORTUNITY,
                    math.nan,
                    False,
                    NOT_USED,
                    f"it doesn't take {where!r} down (it takes down "
                    f"{', '.join(o.takes_down)})",
                )
            )
            continue
        why_not = _fit(o, job, clock)
        if why_not is not None:
            rows.append(
                OpportunityPrice(
                    o.name, OPPORTUNITY, math.nan, False, NOT_USED, why_not
                )
            )
            continue
        if o.name in overrides:
            price, source = overrides[o.name], OVERRIDE
            reason = f"the part's opportunity_costs[{o.name!r}]"
        elif o.cost is not None:
            price, source = o.cost, ON_OPPORTUNITY
            reason = "the price on the opportunity (deprecated)"
        elif general is not None:
            price, source = float(general), OPPORTUNITY_COST
            reason = "the part's opportunity_cost"
        elif planned is not None:
            price, source = float(planned), PLANNED_COST
            reason = (
                "the part's planned_cost: it has no opportunity_cost, so "
                "an opportunity is priced as any other day"
            )
        else:
            raise ValueError(
                f"Opportunity {o.name!r} has no price for this part: give "
                "the part an opportunity_cost (or a planned_cost, which "
                "prices an opportunity as any other day), or "
                f"opportunity_costs={{{o.name!r}: price}}."
            )
        rows.append(
            OpportunityPrice(o.name, OPPORTUNITY, price, True, source, reason)
        )
        kept.append((o, price))
    any_day: Optional[float]
    if not collection.any_day:
        any_day = None
        rows.append(
            OpportunityPrice(
                ANY_DAY,
                ANY_DAY,
                math.nan,
                False,
                NOT_USED,
                "not allowed: any_day=False",
            )
        )
    elif collection._any_day_price is not None:
        any_day = collection._any_day_price
        rows.append(
            OpportunityPrice(
                ANY_DAY,
                ANY_DAY,
                any_day,
                True,
                COLLECTION,
                "the price on any day given with the opportunities "
                "(deprecated)",
            )
        )
    elif planned is not None:
        any_day = float(planned)
        rows.append(
            OpportunityPrice(
                ANY_DAY,
                ANY_DAY,
                any_day,
                True,
                PLANNED_COST,
                "the part's planned_cost",
            )
        )
    else:
        any_day = None
        rows.append(
            OpportunityPrice(
                ANY_DAY,
                ANY_DAY,
                math.nan,
                False,
                NOT_USED,
                "the part has no planned_cost",
            )
        )
    if collection._failure_price is not None:
        rows.append(
            OpportunityPrice(
                FAILURE,
                FAILURE,
                collection._failure_price,
                True,
                COLLECTION,
                "the failure price given with the opportunities "
                "(deprecated)",
            )
        )
    elif getattr(part, "cu", None) is not None:
        rows.append(
            OpportunityPrice(
                FAILURE,
                FAILURE,
                float(part.cu),
                True,
                UNPLANNED_COST,
                "the part's unplanned_cost",
            )
        )
    else:
        rows.append(
            OpportunityPrice(
                FAILURE,
                FAILURE,
                math.nan,
                False,
                NOT_USED,
                "the part has no unplanned_cost",
            )
        )
    prices = OpportunityPrices(rows=tuple(rows))
    if clock == "model":
        if collection._any_day_price is not None:
            anytime: Any = collection._any_day_price
        else:
            anytime = PART if collection.any_day else None
        failure: Any = (
            PART
            if collection._failure_price is None
            else collection._failure_price
        )
        calendar = _calendar.Calendar(
            [o._replaced(cost=price) for o, price in kept],
            failure=failure,
            anytime=anytime,
        )
        return calendar, prices
    site = _shutdowns.Shutdowns(
        [o._shutdown(price) for o, price in kept],
        any_day=any_day,
        time_zone=collection.time_zone,
    )
    return site, prices


def site_given(**given) -> bool:
    """Whether a policy call gave a site's arguments (so an empty
    collection naming no clock is on real dates)."""
    age = given.pop("age", 0.0)
    timing = given.pop("failure_timing", "half_step")
    return (
        any(v is not None for v in given.values())
        or not (checks.is_number(age) and float(age) == 0.0)
        or timing != "half_step"
    )


# -------------------------------------------------- the part's prices


def check_part_prices(opportunity_cost, job_duration, opportunity_costs):
    """A part's opportunity prices, checked: ``opportunity_cost`` a price
    or None; ``job_duration`` a number more than 0, text such as
    ``"6h"``, or None (kept as given); ``opportunity_costs`` a mapping
    from opportunity names (text) or ``Opportunity`` objects to prices
    (kept as a dict; checked against the opportunities when a policy
    runs)."""
    if opportunity_cost is not None:
        opportunity_cost = _price(opportunity_cost, "opportunity_cost")
    if job_duration is not None:
        if checks.is_number(job_duration):
            if not 0.0 < float(job_duration) < math.inf:
                raise ValueError(
                    "job_duration must be more than 0; got "
                    f"{job_duration!r}."
                )
            job_duration = float(job_duration)
        elif isinstance(job_duration, str):
            _shutdowns._period_hours(job_duration, "job_duration")
        else:
            raise TypeError(
                "job_duration must be a number (the life's unit in model "
                "time, hours on real dates) or text such as '6h'; got "
                f"{job_duration!r}."
            )
    costs: Dict[Any, float] = {}
    if opportunity_costs is not None:
        if not isinstance(opportunity_costs, Mapping):
            raise TypeError(
                "opportunity_costs must be a dict of opportunity (or its "
                f"name) to price; got {opportunity_costs!r}."
            )
        for key, price in opportunity_costs.items():
            if not (
                isinstance(key, Opportunity) or (isinstance(key, str) and key)
            ):
                raise TypeError(
                    "opportunity_costs' keys must be opportunities' names "
                    f"or the Opportunity objects; got {key!r}."
                )
            label = key.name if isinstance(key, Opportunity) else key
            costs[key] = _price(price, f"opportunity_costs[{label!r}]")
    return opportunity_cost, job_duration, costs


def opportunity_prices(
    opportunities,
    *,
    replacement_cost,
    replacement_duration,
    failure_cost,
    failure_duration,
    downtime_cost,
) -> OpportunityPrices:
    """See ``NonRepairable.opportunity_prices``."""
    collection = as_opportunities(opportunities)
    text = isinstance(replacement_duration, str) or isinstance(
        failure_duration, str
    )
    clock = collection._clock or ("dates" if text else "model")
    replacement_cost = _price(replacement_cost, "replacement_cost")
    failure_cost = _price(failure_cost, "failure_cost")
    downtime_cost = _price(downtime_cost, "downtime_cost")
    lengths: Dict[str, float] = {}
    for name, value in (
        ("replacement_duration", replacement_duration),
        ("failure_duration", failure_duration),
    ):
        if isinstance(value, str):
            if clock == "model":
                raise ValueError(
                    f"{name} {value!r} is text, which is for opportunities "
                    "on real dates; in model time give it as a number in "
                    "the life's unit."
                )
            lengths[name] = float(_shutdowns._period_hours(value, name)[0])
        elif checks.is_number(value) and 0.0 <= float(value) < math.inf:
            lengths[name] = float(value)
        else:
            raise ValueError(
                f"{name} must be a number, at least 0, or text such as "
                f"'6h'; got {value!r}."
            )
    job = lengths["replacement_duration"]
    unit = "h" if clock == "dates" else "(the life's unit)"
    rows: List[OpportunityPrice] = []
    for o in collection.opportunities:
        why_not = _fit(o, job if job > 0.0 else None, clock)
        if why_not is None:
            rows.append(
                OpportunityPrice(
                    o.name,
                    OPPORTUNITY,
                    replacement_cost,
                    True,
                    "replacement_cost",
                    "replacement_cost alone: the plant is down anyway",
                )
            )
        else:
            rows.append(
                OpportunityPrice(
                    o.name, OPPORTUNITY, math.nan, False, NOT_USED, why_not
                )
            )
    planned = replacement_cost + downtime_cost * job
    rows.append(
        OpportunityPrice(
            ANY_DAY,
            ANY_DAY,
            planned,
            collection.any_day,
            "replacement_cost + downtime_cost x replacement_duration",
            f"{replacement_cost:g} + {downtime_cost:g} x {job:g} {unit}"
            + ("" if collection.any_day else "; not allowed: any_day=False"),
        )
    )
    unplanned = failure_cost + downtime_cost * lengths["failure_duration"]
    rows.append(
        OpportunityPrice(
            FAILURE,
            FAILURE,
            unplanned,
            True,
            "failure_cost + downtime_cost x failure_duration",
            f"{failure_cost:g} + {downtime_cost:g} x "
            f"{lengths['failure_duration']:g} {unit}",
        )
    )
    return OpportunityPrices(
        rows=tuple(rows),
        planned_cost=planned,
        unplanned_cost=unplanned,
        opportunity_cost=replacement_cost,
        job_duration=None if job == 0.0 else replacement_duration,
    )
