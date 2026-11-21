"""Money on dates at a yearly discount rate, for every schedule result that
gives a present value.

``discount`` is the **effective yearly rate** (0.07 is 7% a year), as in
the dated replacement policies (``repyability.maintenance``): an amount
``y`` years after the date it is discounted to is worth ``(1 +
discount) ** -y`` of itself, a year being 365.25 days. The arithmetic
(``factor``, ``per_unit``, ``equivalent_annual``, ``perpetuity``,
``bracket``, ``net`` and the conversion of a yearly rate to a
continuous one) is ``repyability.value``'s, the one convention for every
present value in RePyability; it is imported here under the names the
schedule module has always used. What is the schedule's own is the
wording of a discounted result (``percent``, ``words``).
"""

from __future__ import annotations

from typing import Optional

from repyability.value._rates import (  # noqa: F401
    DAYS_A_YEAR,
    HOURS_A_YEAR,
    bracket,
    continuous,
    equivalent_annual,
    factor,
    factor_on,
    net,
    per_unit,
    perpetuity,
    yearly,
    years_from,
)


def percent(discount: float) -> str:
    """The yearly rate in words: ``7%``, ``2.5%``."""
    from repyability.maintenance._shutdowns import _percent

    return _percent(discount)


def words(discount: Optional[float], start, escalation=None) -> str:
    """The note every discounted result carries (``discount`` given)."""
    assert discount is not None
    day = start.date() if hasattr(start, "date") else start
    out = f"Discounted to {day} at {percent(discount)} a year."
    if escalation is not None:
        overall_rate = net(discount, escalation)
        assert overall_rate is not None
        overall = percent(overall_rate)
        out += (
            f" The costs escalate at {percent(escalation)} a year from "
            "their prices today: each present value is the cost grown to "
            f"when it falls and discounted, a net {overall} a year; the "
            "costs by period are at today's prices."
        )
    return out
