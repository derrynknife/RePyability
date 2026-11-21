"""Escalation: how the amounts of a valuation grow with time.

A diagram's valuation takes a continuous growth rate per unit of its
models' time for each stream it names (a cost category, a component,
``"value"`` or ``"salvage"``), and, in real terms, an inflation added to
the discount rate and to every growth. A calendar's valuation takes one
effective yearly rate for every cost (``_rates.net``).
"""

import math
from typing import Any, Collection, Dict

from repyability.utils import checks
from repyability.value._categories import CATEGORIES

#: The streams ``escalation`` names besides the cost categories and the
#: components.
STREAMS = ("value", "salvage")


def escalation(components: Collection, escalation) -> Dict[Any, float]:
    """``escalation`` checked, for a valuation of ``components``: a dict
    of a growth rate (a continuous rate per unit time, of either sign) by
    cost category, component, ``"value"`` or ``"salvage"``."""
    if escalation is None:
        return {}
    if not isinstance(escalation, dict):
        raise ValueError(
            "escalation must be a dict of a growth rate per unit time by "
            "cost category, component, 'value' or 'salvage' (e.g. "
            f"{{'repair': 0.03 / 8760}}); got {escalation!r}."
        )
    out: Dict[Any, float] = {}
    for key, rate in escalation.items():
        if not (key in components or key in CATEGORIES or key in STREAMS):
            raise ValueError(
                f"escalation names {key!r}, which is no component, cost "
                f"category ({', '.join(CATEGORIES)}), 'value' or "
                "'salvage'."
            )
        number = checks.number_or_nan(rate)
        if not math.isfinite(number):
            raise ValueError(
                f"The escalation of {key!r} must be a finite growth rate "
                "per unit time of the models (continuous; 3% a year with "
                f"models in hours is math.log(1.03) / 8760); got {rate!r}."
            )
        out[key] = number
    return out


def inflation(inflation, rates: str) -> float:
    """The continuous inflation per unit time added to every rate: with
    ``rates="real"``, ``inflation`` (0 when not given); refused with
    ``"nominal"``, whose rates have it already."""
    checks.one_of("rates", rates, ("nominal", "real"))
    if inflation is None:
        return 0.0
    if rates == "nominal":
        raise ValueError(
            'inflation is for rates="real": nominal rates and growths '
            'have it in them already. Give rates="real" with real '
            "rates, or leave inflation out."
        )
    number = checks.number_or_nan(inflation)
    if not math.isfinite(number):
        raise ValueError(
            "inflation must be a finite continuous rate per unit time of "
            f"the models; got {inflation!r}."
        )
    return number


def growth(growths: Dict[Any, float], lift: float, *keys) -> float:
    """The growth of a stream, by the first of ``keys`` in ``growths``
    (as ``escalation`` gave them), with the inflation ``lift`` added."""
    for key in keys:
        if key in growths:
            return growths[key] + lift
    return lift
