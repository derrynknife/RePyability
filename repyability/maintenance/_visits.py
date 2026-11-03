"""The best interval and depth for a layered package of visits.

Work on a part is often bundled into visits: a cheap task at every visit
and a deeper one at every ``k``-th. Visits are a shared cost, so the
visit interval ``T`` and the depth ratio ``k`` are chosen together. This is
a layered version of the age-replacement optimiser, for visits that keep to
fixed times (block replacement).

Each layer is a part renewed at its visits and replaced at once when it
fails, at its unplanned cost. A part renewed every ``P`` costs, a unit of
time,

    rate(P) = (planned + unplanned M(P)) / P,

with ``M(P)`` the expected failures in ``P`` of a new part, each replaced
by a new one at once (a renewal count, exact: the part's
[`failures`][repyability.NonRepairable.failures]). The package is the
visits themselves, the light layer renewed every ``T`` and the deep layer
every ``k T``::

    cost rate = visit_cost / T + rate_light(T) + rate_deep(k T).

For each ratio ``k`` the interval ``T`` is found on a grid of intervals up
to a few lives of the longer-lived part, refined between the best one's
neighbours. If no interval beats never visiting (the cost rate keeps
falling as the interval grows, as for a part that does not wear out), the
interval is ``inf``.
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import minimize_scalar

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: The grid of intervals, and the longest interval tried, in lives of the
#: longer-lived part.
_GRID = 600
_LIVES = 4.0
#: The shortest interval tried, as a share of the shorter life.
_SHORTEST = 1e-3


def _part_rate(part, period) -> np.ndarray:
    """The long-run cost a unit of time of renewing ``part`` every
    ``period`` (an array of them)."""
    period = np.asarray(period, dtype=float)
    failures = np.asarray(
        part.failures(period.ravel()).expected_failures, dtype=float
    ).reshape(period.shape)
    return (part.cp + part.cu * failures) / period


def _mean(part) -> float:
    from repyability.rbd._model_utils import model_mean

    return float(model_mean(part.reliability))


def _check_part(part, name: str) -> None:
    from repyability.non_repairable import NonRepairable

    if not isinstance(part, NonRepairable):
        raise TypeError(
            f"{name} must be a NonRepairable (a part with its costs), or "
            f"None for no such layer; got {part!r}."
        )
    part._require_costs()


@dataclass
class RatioChoice(_ResultMapping):
    """The best visit interval for one depth ratio: an entry of
    [`LayeredVisits`][repyability.LayeredVisits].

    Attributes
    ----------
    ratio : int
        The deep layer is renewed at every ``ratio``-th visit.
    interval : float
        The best visit interval for that ratio (``inf``: never visit).
    cost_rate : float
        The long-run cost a unit of time at it.
    """

    ratio: int
    interval: float
    cost_rate: float


@dataclass
class LayeredVisits(_ResultMapping):
    """The best visit interval and depth ratio: returned by
    [`optimal_layered_visits`][repyability.optimal_layered_visits].

    Attributes
    ----------
    interval : float
        The best interval between visits (``inf``: no visits pay).
    ratio : int
        The best depth ratio: the deep layer at every ``ratio``-th visit.
    deep_interval : float
        ``ratio * interval``: the time between deep renewals.
    cost_rate : float
        The package's long-run cost a unit of time.
    visit_cost_rate : float
        Its parts: the visits themselves, the light layer's renewals and
        failures, and the deep layer's.
    light_cost_rate : float
        The companion of ``visit_cost_rate`` above.
    deep_cost_rate : float
        The companion of ``visit_cost_rate`` above.
    by_ratio : list of RatioChoice
        The best interval and cost rate for each ratio tried, from 1.
    """

    interval: float
    ratio: int
    deep_interval: float
    cost_rate: float
    visit_cost_rate: float
    light_cost_rate: float
    deep_cost_rate: float
    by_ratio: List[RatioChoice]


def layered_cost_rate(
    light, deep, interval, ratio, *, visit_cost=0.0
) -> float:
    """The long-run cost a unit of time of a layered package of visits.

    The visits are ``interval`` apart; ``light`` is renewed at every
    visit and ``deep`` at every ``ratio``-th, each replaced at once when
    it fails (see
    [`optimal_layered_visits`][repyability.optimal_layered_visits]).

    Parameters
    ----------
    light : NonRepairable or None
        The parts of the two layers, with their planned and unplanned
        costs; None for no such layer.
    deep : NonRepairable or None
        The companion of ``light`` above.
    interval : float
        The time between visits, more than 0.
    ratio : int
        The deep layer is renewed at every ``ratio``-th visit.
    visit_cost : float, optional
        What each visit costs by itself, by default 0.

    Returns
    -------
    float
        The cost a unit of time.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import NonRepairable, layered_cost_rate
    >>> filter_ = NonRepairable(
    ...     surv.Weibull.from_params([90.0, 3.0]),
    ...     planned_cost=20.0, unplanned_cost=1000.0,
    ... )
    >>> pump = NonRepairable(
    ...     surv.Weibull.from_params([700.0, 2.5]),
    ...     planned_cost=400.0, unplanned_cost=2500.0,
    ... )
    >>> round(layered_cost_rate(filter_, pump, 60.0, 4, visit_cost=50.0), 3)
    7.873
    """
    interval = checks.real_array(interval, "interval")
    if interval.ndim or not (0.0 < float(interval) < math.inf):
        raise ValueError(
            f"interval must be a number more than 0; got {interval!r}."
        )
    ratio = checks.whole_number(ratio, "ratio")
    for part, name in ((light, "light"), (deep, "deep")):
        if part is not None:
            _check_part(part, name)
    return float(sum(_split(light, deep, float(interval), ratio, visit_cost)))


def _split(light, deep, interval, ratio, visit_cost) -> tuple:
    interval = float(interval)
    visits = visit_cost / interval
    light_rate = 0.0 if light is None else float(_part_rate(light, interval))
    deep_rate = (
        0.0 if deep is None else float(_part_rate(deep, ratio * interval))
    )
    return visits, light_rate, deep_rate


def optimal_layered_visits(
    light, deep, *, visit_cost=0.0, max_ratio=12
) -> LayeredVisits:
    """The best visit interval and depth ratio for a layered package of
    visits: a cheap task at every visit, a deeper one at every ``k``-th.

    In plain terms: visits cost something each time, so more visits mean
    fewer failures but more visit cost. The light layer is renewed at every
    visit and the deep layer at every ``k``-th; this finds the interval
    between visits and the ratio ``k`` that cost least in the long run. The
    visits keep to fixed times (block replacement): a part that fails is
    replaced at once, and renewed again at its next visit. Each layer's
    failures are an exact renewal count (see
    [`failures`][repyability.NonRepairable.failures]). With one layer and
    no visit cost it is the optimal block-replacement interval.

    Parameters
    ----------
    light : NonRepairable or None
        The part renewed at every visit, with its planned and unplanned
        costs; None for no light layer.
    deep : NonRepairable or None
        The part renewed at every ``k``-th visit; None for none.
    visit_cost : float, optional
        What each visit costs by itself (access, set-up), by default 0.
    max_ratio : int, optional
        The deepest ratio tried, by default 12. With one layer the ratio
        is 1.

    Returns
    -------
    LayeredVisits
        The best interval and ratio, the cost rate and its parts, and the
        best interval for each ratio tried. A best ratio at ``max_ratio``
        means a larger one may be better.

    Raises
    ------
    ValueError, TypeError
        If neither layer is given, a part has no costs, or an argument is
        out of range.

    Examples
    --------
    A filter that wears out in about 80 days (a failure costs 1,000 in
    damage) and a pump in about 620, a visit costing 50:

    >>> import surpyval as surv
    >>> from repyability import NonRepairable, optimal_layered_visits
    >>> filter_ = NonRepairable(
    ...     surv.Weibull.from_params([90.0, 3.0]),
    ...     planned_cost=20.0, unplanned_cost=1000.0,
    ... )
    >>> pump = NonRepairable(
    ...     surv.Weibull.from_params([700.0, 2.5]),
    ...     planned_cost=400.0, unplanned_cost=2500.0,
    ... )
    >>> plan = optimal_layered_visits(filter_, pump, visit_cost=50.0)
    >>> plan.ratio, round(plan.interval, 1), round(plan.cost_rate, 3)
    (10, 29.9, 5.836)
    """
    if light is None and deep is None:
        raise ValueError("Give at least one layer: light or deep.")
    for part, name in ((light, "light"), (deep, "deep")):
        if part is not None:
            _check_part(part, name)
    if not (
        checks.is_number(visit_cost) and 0.0 <= float(visit_cost) < math.inf
    ):
        raise ValueError(
            f"visit_cost must be a number, at least 0; got {visit_cost!r}."
        )
    visit_cost = float(visit_cost)
    ratios = checks.whole_number(max_ratio, "max_ratio")
    if deep is None or light is None:
        ratios = 1  # one layer: its depth is the visit's
    lives = [_mean(p) for p in (light, deep) if p is not None]
    longest, shortest = max(lives), min(lives)
    top = _LIVES * longest
    choices = []
    for ratio in range(1, ratios + 1):
        interval, rate = _best_interval(
            light, deep, ratio, visit_cost, top, _SHORTEST * shortest
        )
        choices.append(RatioChoice(ratio, interval, rate))
    best = min(choices, key=lambda choice: choice.cost_rate)
    if math.isinf(best.interval):
        return LayeredVisits(
            math.inf,
            best.ratio,
            math.inf,
            best.cost_rate,
            0.0,
            0.0 if light is None else light.cu / _mean(light),
            0.0 if deep is None else deep.cu / _mean(deep),
            choices,
        )
    visits, light_rate, deep_rate = _split(
        light, deep, best.interval, best.ratio, visit_cost
    )
    return LayeredVisits(
        best.interval,
        best.ratio,
        best.ratio * best.interval,
        best.cost_rate,
        visits,
        light_rate,
        deep_rate,
        choices,
    )


def _run_to_failure(light, deep) -> float:
    """The cost a unit of time with no visits: each layer's failures."""
    return float(sum(p.cu / _mean(p) for p in (light, deep) if p is not None))


def _best_interval(light, deep, ratio, visit_cost, top, bottom):
    """The interval minimising the cost rate for one ratio, and the rate
    (``inf`` and the no-visit rate if none beats never visiting)."""
    grid = np.geomspace(bottom, top, _GRID)
    rate = visit_cost / grid
    if light is not None:
        rate = rate + _part_rate(light, grid)
    if deep is not None:
        rate = rate + _part_rate(deep, ratio * grid)
    i = int(np.nanargmin(rate))
    if i == len(grid) - 1:
        return math.inf, _run_to_failure(light, deep)
    low, high = grid[max(i - 1, 0)], grid[i + 1]

    def total(interval: float) -> float:
        return sum(_split(light, deep, interval, ratio, visit_cost))

    refined = minimize_scalar(
        total,
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1e-10 * high},
    )
    if refined.fun <= rate[i]:
        return float(refined.x), float(refined.fun)
    return float(grid[i]), float(rate[i])
