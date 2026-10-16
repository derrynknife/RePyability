"""Planned replacement, and the value of skipping the next planned
replacement right after a failure (#319, UC-6 of this package's
``README.md``).

A unit is replaced when it fails (unplanned, at ``unplanned_cost``) and on
a plan (planned, at ``planned_cost``), at once either way. Right after a
failure a new unit is in, and the question is whether to skip the next
planned replacement. The NPV of skipping is what it saves: the expected
present value of the costs if the replacement is done, less that if it is
skipped. Costs are discounted continuously, at ``r = log(1 + i) / 365.25``
a day for a rate ``i`` a year.

**A plan that resets** (age replacement): the next planned replacement is
``every`` after the last replacement, planned or not. Each replacement
starts the same process afresh, so a cycle's discounted cost ``A(s)`` and
discounted length ``M(s)`` (cut off at age ``s``) give the value of the
plan from new, ``V = A(T) / (r M(T))`` (renewal-reward, discounted), and
skipping makes the first cycle's cut-off ``2 T`` rather than ``T``::

    NPV = A(T) - A(2T) + A(T) (M(2T) - M(T)) / M(T)

with ``M(s)`` the integral of ``exp(-r x) R(x)`` to ``s``, and ``A(s) =
unplanned L(s) + planned exp(-r s) R(s)``, ``L(s) = 1 - exp(-r s) R(s) -
r M(s)`` the discounted chance of a failure first. The discount rate
cancels from the last term, which holds with no discounting too.

**A plan that does not reset** (calendar, block replacement): planned
replacements fall on fixed dates, the next ``d`` days after the failure.
Done, a new unit goes in at ``d``; skipped, the unit in now runs on. Both
replace at the following date, ``d + T``, so the two differ only before
it::

    NPV = planned exp(-r d) + exp(-r d) F(T) - (F(d + T) - F(d))

with ``F(t)`` the expected discounted cost of the failures of a new unit
and its replacements in ``[0, t)``: a
[`RepairableRBD`][repyability.RepairableRBD]'s ``expected_cost`` of a
component replaced in no time (exact, no simulation).
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: Days in a year, for a discount rate a year.
DAYS_A_YEAR = 365.25


def _amount(value, name: str) -> float:
    """``value`` as a cost: a finite number, at least 0."""
    if not (checks.is_number(value) and 0.0 <= float(value) < math.inf):
        raise ValueError(
            f"{name} must be a number, at least 0; got {value!r}."
        )
    return float(value)


def _days(value, name: str) -> float:
    """``value`` as a number of days: finite, more than 0."""
    if not (checks.is_number(value) and 0.0 < float(value) < math.inf):
        raise ValueError(
            f"{name} must be a number of days, more than 0; got {value!r}."
        )
    return float(value)


@dataclass(frozen=True, init=False)
class ReplacementPlan:
    """A unit replaced when it fails and on a plan.

    Parameters
    ----------
    life : surpyval model
        A unit's life, in days: a fitted surpyval model, or one built with
        ``from_params``.
    every : float
        The plan's interval, in days.
    planned_cost : float
        What a planned replacement costs, at least 0, in your currency.
    unplanned_cost : float
        What a replacement at a failure costs, at least 0.
    resets : bool
        Whether the plan resets at each replacement: True, the next
        planned replacement is ``every`` after the last replacement,
        planned or not (age replacement); False, planned replacements fall
        on fixed dates, ``every`` apart, whatever happens (calendar, or
        block, replacement).

    Examples
    --------
    A bearing with a Weibull life (scale 365 days, shape 2), replaced
    every 180 days for 150, and at a failure for 400; right after a
    failure, at 8% a year, skipping the next planned replacement saves:

    >>> import surpyval as surv
    >>> from repyability.maintenance import ReplacementPlan
    >>> life = surv.Weibull.from_params([365.0, 2.0])
    >>> resetting = ReplacementPlan(life, 180.0, 150.0, 400.0, resets=True)
    >>> round(resetting.skip_next(discount_rate=0.08).npv, 2)
    26.56

    On a calendar, it depends on how soon the next planned date is:

    >>> calendar = ReplacementPlan(life, 180.0, 150.0, 400.0, resets=False)
    >>> round(calendar.skip_next(0.08, days_to_next=30.0).npv, 2)
    123.08
    """

    life: Any
    every: float
    planned_cost: float
    unplanned_cost: float
    resets: bool

    def __init__(
        self, life, every, planned_cost, unplanned_cost, *, resets
    ) -> None:
        checks.no_distribution(life, "life")
        if not callable(getattr(life, "sf", None)):
            raise TypeError(
                "life must be a lifetime model with a survival function "
                f"(sf), such as a fitted surpyval model; got {life!r}."
            )
        if not isinstance(resets, bool):
            raise TypeError(
                "resets must be True (the plan resets at each replacement) "
                f"or False (it keeps to fixed dates); got {resets!r}."
            )
        object.__setattr__(self, "life", life)
        object.__setattr__(self, "every", _days(every, "every"))
        object.__setattr__(
            self, "planned_cost", _amount(planned_cost, "planned_cost")
        )
        object.__setattr__(
            self, "unplanned_cost", _amount(unplanned_cost, "unplanned_cost")
        )
        object.__setattr__(self, "resets", resets)

    def skip_next(
        self, discount_rate, days_to_next: Optional[float] = None
    ) -> "SkipValue":
        """The NPV of skipping the next planned replacement, right after a
        failure (a new unit just in): what skipping saves, net of the
        failures it adds. Positive, skipping pays.

        Parameters
        ----------
        discount_rate : float
            The discount rate a year, at least 0: 0.08 for 8% a year.
        days_to_next : float, optional
            For a plan that does not reset: the days from the failure to
            the next planned date, more than 0 and at most ``every``. A
            plan that resets has its next planned replacement ``every``
            after the failure, and takes none.

        Returns
        -------
        SkipValue
            The NPV of skipping, with what it was worked out for.
        """
        if not (
            checks.is_number(discount_rate)
            and 0.0 <= float(discount_rate) < math.inf
        ):
            raise ValueError(
                "discount_rate must be a rate a year, at least 0 (0.08 for "
                f"8% a year); got {discount_rate!r}."
            )
        rate = math.log1p(float(discount_rate)) / DAYS_A_YEAR
        if self.resets:
            if days_to_next is not None:
                raise ValueError(
                    "A plan that resets has its next planned replacement "
                    f"every ({self.every:g} days) after the failure: give "
                    "no days_to_next."
                )
            days = self.every
            npv = self._skip_resetting(rate)
        else:
            if days_to_next is None:
                raise ValueError(
                    "A plan on fixed dates needs days_to_next: the days "
                    "from the failure to the next planned date."
                )
            days = _days(days_to_next, "days_to_next")
            if days > self.every:
                raise ValueError(
                    f"days_to_next ({days:g}) must be at most every "
                    f"({self.every:g} days): the next planned date is "
                    "within one interval."
                )
            npv = self._skip_calendar(rate, days)
        return SkipValue(
            self.resets, self.every, days, float(discount_rate), npv
        )

    def _skip_resetting(self, rate: float) -> float:
        """The NPV of skipping with a plan that resets (see the module's
        notes)."""
        from repyability.rbd._mean_lifetime import _integrate, model_knots

        def discounted(x: np.ndarray) -> np.ndarray:
            survival = np.asarray(self.life.sf(x), dtype=float).ravel()
            return np.exp(-rate * x) * survival

        knots = model_knots(self.life)

        def length(cut: float) -> float:
            inside = knots[(knots > 0.0) & (knots < cut)]
            return _integrate(
                discounted, np.concatenate(([0.0], inside, [cut]))
            )

        def cost(cut: float, held: float) -> float:
            kept = math.exp(-rate * cut) * float(self.life.sf(cut))
            failed = 1.0 - kept - rate * held
            return self.unplanned_cost * failed + self.planned_cost * kept

        once, twice = length(self.every), length(2.0 * self.every)
        done = cost(self.every, once)
        skipped = cost(2.0 * self.every, twice)
        return done - skipped + done * (twice - once) / once

    def _skip_calendar(self, rate: float, days: float) -> float:
        """The NPV of skipping with a plan on fixed dates, the next
        ``days`` after the failure (see the module's notes)."""
        from repyability.rbd.repairable_rbd import RepairableRBD

        unit = RepairableRBD(
            [("s", "unit"), ("unit", "t")],
            {
                "unit": {
                    "reliability": self.life,
                    "repairability": "instant",
                    "replace_cost": self.unplanned_cost,
                }
            },
        )
        windows = np.array([self.every, days, days + self.every])
        interval, before, after = unit.expected_cost(
            windows, discount_rate=rate
        ).mean
        later = math.exp(-rate * days)
        return float(
            self.planned_cost * later + later * interval - (after - before)
        )


@dataclass
class SkipValue(_ResultMapping):
    """The NPV of skipping the next planned replacement of a
    [`ReplacementPlan`][repyability.ReplacementPlan], right after a
    failure: returned by its ``skip_next``.

    Attributes
    ----------
    resets : bool
        Whether the plan resets at each replacement.
    every : float
        The plan's interval, in days.
    days_to_next : float
        The days from the failure to the next planned replacement:
        ``every`` for a plan that resets.
    discount_rate : float
        The discount rate a year.
    npv : float
        What skipping the next planned replacement saves, in present
        value: the planned cost saved, net of the failures it adds.
        Positive, skipping pays.
    """

    resets: bool
    every: float
    days_to_next: float
    discount_rate: float
    npv: float
