"""When to replace a part in service next (#319, UC-12 of this package's
``README.md``): now, at one of the scheduled outages, at a planned
replacement between them, or at its failure.

Each choice is costed from the part's age ``a`` to its replacement: its
expected cost, less the time it serves valued at ``g``, the long-run cost
a unit of time of the part's
[`optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy]
(the baseline),

    cost(u, price) = price S(u) + failure (1 - S(u)) - g ∫₀ᵘ S,

with ``S(u) = R(a + u) / R(a)``, the chance the part is working ``u`` from
now (replacing now costs its price). The time the part serves is time a
new part would otherwise serve, which ``g`` values: this is the relative
value of the renewal-reward process, a new part taken to be worth the same
whenever it is fitted. With no outages, replacing now is the best from the
optimal replacement age exactly.

A planned replacement is costed at its best time, on a grid of ``_GRID``
times up to the time the part outlives with a chance of ``_FAR``, refined
between the best one's neighbours. Each later choice's break-even age (the
youngest at which replacing now costs no more) is found on a grid of
``_AGES`` ages up to the age a new part reaches with a chance of ``_FAR``,
then halved down to its first.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import minimize_scalar

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: The chance of a part outliving the latest time a planned replacement is
#: tried at, and the oldest age a break-even age is looked for at.
_FAR = 1e-3
#: The times a planned replacement is tried at, before the best is refined.
_GRID = 2001
#: The ages a break-even age is looked for at, before it is halved down to.
_AGES = 64
#: Halvings of a break-even age's bracket.
_HALVINGS = 50


@dataclass
class ReplacementChoice(_ResultMapping):
    """One choice of when to replace a part next: an entry of
    [`NextReplacement`][repyability.NextReplacement].

    Attributes
    ----------
    kind : str
        ``"now"``, ``"outage"`` (at one of the scheduled outages),
        ``"planned"`` (a planned replacement at its best time) or
        ``"failure"`` (run to failure).
    time : float
        When, from now: 0 for now, the outage's time, the planned
        replacement's best time (``inf`` if none beats running to
        failure), ``inf`` at failure.
    price : float
        What the replacement costs if the part has not failed by then (at
        failure, the failure price).
    cost : float
        The expected cost until the part is replaced (at the failure price
        if it fails first), less the time it serves at ``cost_rate``: the
        lowest is the best.
    break_even_age : float or None
        The youngest age at which replacing now costs no more than this
        choice (``inf``: no age up to the one a new part reaches with a
        chance of 0.1%). None for replacing now.
    """

    kind: str
    time: float
    price: float
    cost: float
    break_even_age: Optional[float]


@dataclass
class NextReplacement(_ResultMapping):
    """When to replace a part in service next: returned by
    [`NonRepairable.optimal_next_replacement`][repyability.NonRepairable.optimal_next_replacement].

    Attributes
    ----------
    age : float
        The part's age now.
    best : ReplacementChoice
        The cheapest choice (on a tie, the earlier in ``choices``).
    now : bool
        Whether the best is to replace it now.
    choices : list of ReplacementChoice
        Every choice: now, each outage by its time, a planned replacement at
        its best time, and run to failure.
    replace_from_age : float
        The age from which replacing now beats every later choice: the
        largest break-even age.
    cost_rate : float
        The value of a unit of the part's service, the baseline: the
        long-run cost a unit of time of the part's
        [`optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy].
    """

    age: float
    best: ReplacementChoice
    now: bool
    choices: List[ReplacementChoice]
    replace_from_age: float
    cost_rate: float


def _amount(value, name: str) -> float:
    """``value`` as a cost: a finite number, at least 0."""
    if not (checks.is_number(value) and 0.0 <= float(value) < math.inf):
        raise ValueError(
            f"{name} must be a number, at least 0; got {value!r}."
        )
    return float(value)


def _time(value, name: str) -> float:
    """``value`` as a time from now: a finite number, more than 0."""
    if not (checks.is_number(value) and 0.0 < float(value) < math.inf):
        raise ValueError(
            f"{name} must be a time from now, a number more than 0; "
            f"got {value!r}."
        )
    return float(value)


def _coming(outages) -> list:
    """The scheduled outages as (time, price) pairs, by time."""
    if isinstance(outages, (str, bytes)) or not hasattr(outages, "__iter__"):
        raise TypeError(
            "scheduled_outages must be a list of (time from now, price) "
            f"pairs; got {outages!r}."
        )
    coming = []
    for outage in outages:
        pair = () if isinstance(outage, (str, bytes)) else tuple(outage)
        if len(pair) != 2:
            raise TypeError(
                "each scheduled outage must be a (time from now, price) "
                f"pair; got {outage!r}."
            )
        coming.append(
            (
                _time(pair[0], "the time of a scheduled outage"),
                _amount(pair[1], "the price at a scheduled outage"),
            )
        )
    return sorted(coming)


def _beyond(lasting, scale: float) -> float:
    """The time by which ``lasting`` (a chance of still working) falls to
    ``_FAR``: doubled from ``scale`` past it, then halved down to it."""
    low, high = 0.0, scale
    for _ in range(200):
        if lasting(high) <= _FAR:
            break
        low, high = high, 2.0 * high
    else:
        raise ValueError(
            "The life never falls to a chance of 0.1% of working (some units "
            "never fail), so a part may be kept for good."
        )
    for _ in range(40):
        middle = 0.5 * (low + high)
        if lasting(middle) > _FAR:
            low = middle
        else:
            high = middle
    return high


class _Life:
    """A part's life, priced: each choice's cost from an age."""

    def __init__(self, sf, scale: float, failed: float, rate: float):
        self.sf, self.scale = sf, scale
        self.failed, self.rate = failed, rate

    def survival(self, x: float) -> float:
        with np.errstate(invalid="ignore"):
            return float(np.ravel(self.sf(x))[0])

    def keep(self, age: float, u: float, price: float) -> float:
        """The cost of keeping a part ``age`` old to ``u`` from now, then
        replacing it at ``price``."""
        held = self.survival(age)
        lasting = self.survival(age + u) / held
        cost = price * lasting + self.failed * (1.0 - lasting)
        if self.rate > 0.0:
            with np.errstate(invalid="ignore"):
                served = quad(self.survival, age, age + u, limit=200)[0]
            cost -= self.rate * served / held
        return cost

    def planned(self, age: float, price: float) -> tuple:
        """The best time from now for a planned replacement of a part
        ``age`` old at ``price``, and its cost."""
        held = self.survival(age)
        later = _beyond(lambda u: self.survival(age + u) / held, self.scale)
        times = np.linspace(0.0, later, _GRID)
        with np.errstate(invalid="ignore"):
            lasting = np.ravel(np.asarray(self.sf(age + times), float)) / held
        costs = price * lasting + self.failed * (1.0 - lasting)
        costs -= self.rate * cumulative_trapezoid(lasting, times, initial=0.0)
        i = int(np.argmin(costs))
        if i == _GRID - 1:
            # Still falling where the part is all but sure to have failed:
            # no planned replacement beats running it to failure.
            return math.inf, self.keep(age, math.inf, price)
        best_time, best = float(times[i]), self.keep(age, times[i], price)
        refined = minimize_scalar(
            lambda u: self.keep(age, u, price),
            bounds=(times[max(i - 1, 0)], times[min(i + 1, _GRID - 1)]),
            method="bounded",
            options={"xatol": later * 1e-9},
        )
        if refined.success and refined.fun < best:
            best_time, best = float(refined.x), float(refined.fun)
        if not best < price:
            return 0.0, price
        return best_time, best


def _break_even(beaten, oldest: float) -> float:
    """The youngest age at which ``beaten(age)`` holds: on a grid of
    ``_AGES`` ages up to ``oldest``, then halved down to the first."""
    ages = np.linspace(0.0, oldest, _AGES)
    first = next((i for i, at in enumerate(ages) if beaten(float(at))), None)
    if first is None:
        return math.inf
    if first == 0:
        return 0.0
    low, high = float(ages[first - 1]), float(ages[first])
    for _ in range(_HALVINGS):
        middle = 0.5 * (low + high)
        if beaten(middle):
            high = middle
        else:
            low = middle
    return high


def optimal_next_replacement(
    part, age, *, now_cost=None, scheduled_outages=()
) -> NextReplacement:
    """The work of ``NonRepairable.optimal_next_replacement`` (see
    there)."""
    from repyability.rbd._model_utils import model_mean

    part._require_costs()
    if part.model_parameterization not in ("parametric", "degradation"):
        raise NotImplementedError(
            "optimal_next_replacement needs a life with a survival function "
            "in closed form (a surpyval parametric distribution, mixture or "
            "degradation process); this part's is "
            f"{part.model_parameterization}."
        )
    if not (checks.is_number(age) and 0.0 <= float(age) < math.inf):
        raise ValueError(f"age must be a number, at least 0; got {age!r}.")
    age = float(age)
    now = part.cp if now_cost is None else _amount(now_cost, "now_cost")
    coming = _coming(scheduled_outages)
    rate = float(part.optimal_replacement_policy().cost_rate)
    mean = model_mean(part.reliability)
    life = _Life(part.reliability_function, mean, part.cu, rate)
    if not life.survival(age) > 0.0:
        raise ValueError(
            f"The life gives no chance of a unit working at age {age:g}."
        )

    # Each later choice: its kind, time (a planned replacement's is its
    # best) and price.
    later = [("outage", time, price) for time, price in coming]
    later += [("planned", 0.0, part.cp), ("failure", math.inf, part.cu)]

    def chosen(kind: str, time: float, price: float, at: float) -> tuple:
        """A choice's time and cost for a part ``at`` old."""
        if kind == "planned":
            return life.planned(at, price)
        return time, life.keep(at, time, price)

    oldest = _beyond(life.survival, mean)
    tie = 1e-10 * max(now, part.cu)
    choices = [ReplacementChoice("now", 0.0, now, now, None)]
    evens: List[float] = []
    for kind, time, price in later:
        when, cost = chosen(kind, time, price, age)
        even = _break_even(
            lambda at: now <= chosen(kind, time, price, at)[1] + tie, oldest
        )
        evens.append(even)
        choices.append(ReplacementChoice(kind, when, price, cost, even))
    # On a tie, the earlier choice; a planned replacement that never pays
    # is running to failure.
    best = min(
        choices,
        key=lambda c: (c.cost, c.kind == "planned" and c.time == math.inf),
    )
    return NextReplacement(
        age,
        best,
        best.kind == "now",
        choices,
        max(evens),
        rate,
    )
