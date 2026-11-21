"""Discounting: one convention for every present value in RePyability.

Money is discounted **continuously** at a rate per unit of the clock it
falls on: an amount ``t`` units on is worth ``exp(-r t)`` of itself. The
diagrams' methods take ``r`` per unit of their models' time
(``discount_rate``). The calendar's methods take an **effective yearly
rate** ``i`` (``discount=0.07`` is 7% a year), which is ``ln(1 + i)`` a
year (``continuous``) and ``ln(1 + i) * unit_hours / HOURS_A_YEAR`` per
unit of a model's time on the calendar (``per_unit``), a year being 365.25
days. The conversion between the two is made here, and nowhere else.

- ``discount_rate``: a continuous rate per unit time, checked.
- ``yearly``: a yearly rate, checked.
- ``continuous``, ``per_unit``: a yearly rate as a continuous one.
- ``factor``, ``factor_on``, ``years_from``: what an amount later (years
  on, or on a date) is worth now.
- ``discounted_span``: a unit amount per unit time over a span, now.
- ``present_horizon``, ``horizons``: a unit cost rate over horizons.
- ``equivalent``, ``equivalent_annual``: the steady cost worth a present
  value; ``perpetuity``: an amount every gap for ever.
- ``bracket``: an expected cost inside windows, between discounting it at
  each window's start and end.
- ``net``: the yearly rate of costs that escalate at a yearly rate.
"""

from __future__ import annotations

import math
import warnings
from datetime import date
from typing import Callable, List, Optional, Tuple

import numpy as np

from repyability.utils.checks import is_number, real_array
from repyability.utils.wrappers import outside_level

#: The days in a year, for a yearly rate on dates.
DAYS_A_YEAR = 365.25


#: The hours in a year.
HOURS_A_YEAR = 24.0 * DAYS_A_YEAR


def discount_rate(discount_rate) -> float:
    """``discount_rate`` as a float: a ValueError unless it is a finite,
    non-negative number (#184)."""
    number = isinstance(
        discount_rate, (int, float, np.integer, np.floating)
    ) and not isinstance(discount_rate, bool)
    rate = float(discount_rate) if number else float("nan")
    if not (np.isfinite(rate) and rate >= 0.0):
        raise ValueError(
            "discount_rate must be a finite, non-negative number (a "
            "continuous rate per unit time of the component models), got "
            f"{discount_rate!r}."
        )
    return rate


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


def net(discount, escalation) -> Optional[float]:
    """The yearly rate the present values are taken at when every cost
    escalates at the effective yearly ``escalation`` (0.03 for 3% a
    year): an amount ``y`` years on is its price today times ``(1 +
    escalation) ** y``, worth ``((1 + escalation) / (1 + discount)) **
    y`` of that price now, which is ``(1 + net) ** -y``. ``discount`` as
    ``yearly`` gave it; with no escalation, it."""
    if escalation is None:
        return discount
    if discount is None:
        raise ValueError(
            "escalation= grows the costs in their present values: give a "
            "discount= too (0 for none)."
        )
    if not (is_number(escalation) and -1.0 < float(escalation) < 10.0):
        raise ValueError(
            "escalation must be a yearly rate such as 0.03 (3% a year), "
            f"more than -1; got {escalation!r}."
        )
    return (1.0 + discount) / (1.0 + float(escalation)) - 1.0


def continuous(discount: float) -> float:
    """The yearly rate as a continuous one a year: ``ln(1 + discount)``."""
    return math.log1p(float(discount))


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


def discounted_span(a, b, rate: float):
    """The integral of ``exp(-rate t)`` from ``a`` to ``b``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not rate:
        return b - a
    return np.exp(-rate * a) * -np.expm1(-rate * (b - a)) / rate


def horizons(horizon, rate: float) -> np.ndarray:
    """``horizon``, a number or an array of them, as an array of
    non-negative floats: finite, unless costs are discounted at ``rate``
    (#231), where an endless horizon is worth ``1 / rate`` of a steady
    cost rate. Else a ValueError."""
    try:
        values = real_array(horizon, "horizon")
    except (TypeError, ValueError):
        values = np.array(np.nan)
    if values.size and np.all(values >= 0.0) and np.all(values < np.inf):
        return values
    if values.size and np.all(values >= 0.0):
        if rate > 0.0:
            return values
        raise ValueError(
            "horizon must be finite without a discount_rate: an undiscounted "
            f"cost over an endless horizon is endless; got {horizon!r}."
        )
    raise ValueError(
        "horizon must be a finite, non-negative number (or an array of "
        f"them; inf with a discount_rate), got {horizon!r}."
    )


#: A discount rate times the shortest of the components' mean lives
#: beyond which their failures' costs are discounted to almost nothing
#: (``exp(-20)``, 2e-9), and times a horizon beyond which the whole time
#: scale is: a rate given per year with the models in hours, say (#231).
#: A long horizon is no mistake in itself.
DISCOUNTED_AWAY = 20.0


HORIZON_AWAY = 1000.0


def present_horizon(
    horizons: np.ndarray,
    rate: float,
    lives: Callable[[], List[float]] = list,
) -> np.ndarray:
    """What a unit cost rate over ``[0, horizon)`` is worth at the start,
    discounted continuously at ``rate`` per unit time (#184): ``(1 -
    exp(-rate * horizon)) / rate``, the horizon itself undiscounted, and
    ``1 / rate`` for an endless one; for each of ``horizons``. Warns where
    the rate discounts the costs away (#231): times the shortest of the
    components' mean ``lives`` more than ``DISCOUNTED_AWAY``, or times a
    horizon more than ``HORIZON_AWAY``."""
    if rate == 0.0:
        return horizons
    known = [life for life in lives() if 0.0 < life < math.inf]
    finite = horizons[np.isfinite(horizons)]
    span, what = 0.0, ""
    if known and rate * min(known) > DISCOUNTED_AWAY:
        span, what = min(known), "the components' shortest mean life"
    elif finite.size and rate * float(finite.max()) > HORIZON_AWAY:
        span, what = float(finite.max()), "the horizon"
    if what:
        warnings.warn(
            f"discount_rate {rate:g} discounts the costs at {what} "
            f"({span:g}) by a factor of exp(-{rate * span:.3g}), to "
            "almost nothing: it is a continuous rate per unit time of the "
            "component models. For an annual rate i with models in hours, "
            "give math.log(1 + i) / 8760 (7% a year: "
            f"{math.log(1.07) / 8760:.3g} an hour).",
            stacklevel=outside_level(),
        )
    return -np.expm1(-rate * horizons) / rate


def equivalent(npc: float, rate: float, length: float) -> float:
    """The steady cost a unit of time over ``length`` worth ``npc`` at the
    continuous ``rate`` (``npc / length`` at 0)."""
    if rate > 0.0 or (rate < 0.0 and math.isfinite(length)):
        # (Below 0, a net rate of costs that grow faster than the
        # discount, over a finite length.)
        return rate * npc / -math.expm1(-rate * length)
    return npc / length


def equivalent_annual(npv: float, discount: float, years: float) -> float:
    """The steady cost a year over ``years`` (``math.inf`` for ever) worth
    ``npv`` now: the equivalent annual cost (``npv / years`` at no
    discount)."""
    return equivalent(float(npv), continuous(discount), float(years))


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
    # (At a net rate below 0, costs escalating faster than the discount,
    # the window's end is worth more than its start.)
    return 0.5 * (high + low), 0.5 * np.abs(high - low)
