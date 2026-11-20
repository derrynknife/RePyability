"""Money on dates at a yearly discount rate: one convention, kept here
for every schedule result that gives a present value.

``discount`` is the **effective yearly rate** (0.07 is 7% a year), as in
the dated replacement policies (``repyability.maintenance``): an amount
``y`` years after the date it is discounted to is worth ``(1 +
discount) ** -y`` of itself, a year being 365.25 days. The continuous
rate a year is ``ln(1 + discount)`` (``continuous``), and a model in its
own unit of time takes it per unit (``per_unit``).

- ``factor``: what an amount so many years on is worth.
- ``factor_on``: the same for a date, from the date discounted to.
- ``equivalent_annual``: the steady cost a year worth a present value
  over a span (the equivalent annual cost).
- ``perpetuity``: an amount at a first time and every gap after it, for
  ever.
- ``bracket``: an expected cost that accrues inside windows, between
  discounting each window's at its start and at its end.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional, Tuple

import numpy as np

from repyability.utils.checks import is_number

#: The days in a year, for a yearly rate on dates.
DAYS_A_YEAR = 365.25
#: The hours in a year.
HOURS_A_YEAR = 24.0 * DAYS_A_YEAR


def yearly(discount, *, required: bool = False) -> Optional[float]:
    """``discount=`` checked: a yearly rate from 0 (None when not given,
    unless ``required``)."""
    if discount is None and not required:
        return None
    if not (is_number(discount) and 0.0 <= float(discount) < 10.0):
        raise ValueError(
            "discount must be a yearly rate such as 0.07 (7%), at least "
            f"0; got {discount!r}."
        )
    return float(discount)


def continuous(discount: float) -> float:
    """The yearly rate as a continuous one a year: ``ln(1 + discount)``."""
    from repyability.maintenance._horizon import _rate

    return _rate(discount)


def per_unit(discount: float, unit_hours: float) -> float:
    """The yearly rate as a continuous one per unit of a model's time on
    the calendar, the unit being ``unit_hours`` long: what a method in
    model time takes as its ``discount_rate``."""
    return continuous(discount) * unit_hours / HOURS_A_YEAR


def factor(discount: float, years):
    """What an amount ``years`` on is worth now: ``(1 + discount) **
    -years``. A number or an array."""
    return np.exp(-continuous(discount) * np.asarray(years, float))


def years_from(start: date, day: date) -> float:
    """The years from ``start`` to ``day``, both whole dates."""
    return (day - start).days / DAYS_A_YEAR


def factor_on(discount: float, start: date, day: date) -> float:
    """What an amount on ``day`` is worth on ``start``."""
    return float(factor(discount, years_from(start, day)))


def equivalent_annual(npv: float, discount: float, years: float) -> float:
    """The steady cost a year over ``years`` (``math.inf`` for ever) worth
    ``npv`` now: the equivalent annual cost (``npv / years`` at no
    discount)."""
    from repyability.maintenance._horizon import _equivalent

    return _equivalent(float(npv), continuous(discount), float(years))


def perpetuity(discount: float, first_years: float, gap_years: float) -> float:
    """What one unit of money at ``first_years`` and every ``gap_years``
    after it, for ever, is worth now: ``v ** first / (1 - v ** gap)``
    with ``v = 1 / (1 + discount)``."""
    rate = continuous(discount)
    return math.exp(-rate * first_years) / -math.expm1(-rate * gap_years)


def bracket(
    discount: float, amounts, starts, ends
) -> Tuple[np.ndarray, np.ndarray]:
    """Present values of expected costs that accrue inside windows.

    ``amounts[i]`` (at least 0) falls somewhere from ``starts[i]`` to
    ``ends[i]`` (years on). Discounted at its window's start it is worth
    the most it can be, and at its end the least, so the true present
    value of each lies between the two. Returns each window's middle of
    the two and half their difference: the value, and how far the true
    one can be from it.
    """
    amounts = np.asarray(amounts, float)
    high = amounts * factor(discount, starts)
    low = amounts * factor(discount, ends)
    return 0.5 * (high + low), 0.5 * (high - low)


def percent(discount: float) -> str:
    """The yearly rate in words: ``7%``, ``2.5%``."""
    from repyability.maintenance._shutdowns import _percent

    return _percent(discount)


def words(discount: float, start) -> str:
    """The note every discounted result carries."""
    day = start.date() if hasattr(start, "date") else start
    return f"Discounted to {day} at {percent(discount)} a year."
