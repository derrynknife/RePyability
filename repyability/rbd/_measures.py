"""The measures and lives that are functions of a system's values (#315).

Each importance measure is a function of the system's probabilities of
working and of failing, with an element held working or failed, and the
elements' own. A diagram's (``rbd.py``), a fault tree's
(``fault_tree.py``) and a network's (``network.py``) differ in how those
values are worked out (a decomposition, the top event conditioned, a
decision diagram replayed with an element held), not in the measures. The
ratios are here, as a network takes them; a diagram's are worked out on its
decomposition's arrays in ``rbd.py``. With them, what a measure over time
or a life needs of the system's reliability alone, which a diagram and a
network share: an element's failure density, the Barlow-Proschan shares
(each element's part of the system's failure density, integrated over
time) and the inversion of the reliability for a B-life.
"""

from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import brentq

from repyability.rbd._mean_lifetime import model_knots
from repyability.rbd._model_utils import (
    MODEL_ERRORS,
    failure_time_scale,
    is_fixed_probability,
)


def improvement_potential(
    birnbaum: Mapping, q: Mapping
) -> Dict[Hashable, Any]:
    """Each element's improvement potential, ``R(i working) - R``: its
    Birnbaum importance times its probability of failing ``q``, so that a
    small one keeps its precision."""
    return {name: value * q[name] for name, value in birnbaum.items()}


def risk_achievement_worth(
    fails_held: Mapping, fails: Any
) -> Dict[Hashable, Any]:
    """Each element's risk achievement worth: the system's probability of
    failing with the element held failed, over its probability of failing
    as it is (inf or nan, with numpy's warning, where that is 0)."""
    return {name: value / fails for name, value in fails_held.items()}


def risk_reduction_worth(
    fails: Any, fails_held: Mapping
) -> Dict[Hashable, Any]:
    """Each element's risk reduction worth: the system's probability of
    failing as it is over that with the element held working. An element
    whose working ends the risk is worth infinitely much, which is no
    cause for a warning."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {name: fails / value for name, value in fails_held.items()}


def criticality_importance(
    birnbaum: Mapping, member: Mapping, system: Any
) -> Dict[Hashable, Any]:
    """Each element's criticality importance: its Birnbaum importance times
    its own probability (``member``: of failing, for the failure-oriented
    form, or of working, for the success-oriented one) over the system's
    (of failing, or of working); nan where the system's is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            name: np.where(system > 0, value * member[name] / system, np.nan)
            for name, value in birnbaum.items()
        }


def fussell_vesely(numerators: Mapping, fails: Any) -> Dict[Hashable, Any]:
    """Each element's Fussell-Vesely importance: the probability that a
    minimal cut set containing it has failed (or the sum of the sets'
    probabilities, the rare-event approximation), over the system's
    probability of failing; nan where that is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {name: value / fails for name, value in numerators.items()}


def model_scale(model) -> float:
    """A typical failure time of an element's model (see
    ``failure_time_scale``), or NaN where it has none: a fixed probability,
    or a model whose mean cannot be worked out."""
    if is_fixed_probability(model):
        return float("nan")
    try:
        return float(failure_time_scale(model))
    except MODEL_ERRORS:
        return float("nan")


def difference_steps(x: np.ndarray, dx: float, scale) -> np.ndarray:
    """The steps of a finite difference at the times ``x``: ``dx`` of each
    time, and at 0, where there is none to be relative to, ``dx`` of
    ``scale()``, a typical failure time (1 if there is none), so that the
    step does not depend on the unit of time."""
    step = dx * np.abs(x)
    if np.any(step == 0.0):
        typical = scale()
        typical = typical if np.isfinite(typical) and typical > 0 else 1.0
        step = np.where(step == 0.0, dx * typical, step)
    return step


def density(model, x: np.ndarray) -> np.ndarray:
    """An element's failure density at the times ``x``: its model's own
    (``df``) where it has one, else by central differences of its
    probability of failing (one-sided near 0)."""
    try:
        with np.errstate(all="ignore"):
            return np.asarray(model.df(x), dtype=float).reshape(x.shape)
    except MODEL_ERRORS:
        pass
    step = difference_steps(x, 1e-6, lambda: model_scale(model))
    hi, lo = x + step, np.maximum(x - step, 0.0)
    if hasattr(model, "ff"):
        change = np.asarray(model.ff(hi), dtype=float) - np.asarray(
            model.ff(lo), dtype=float
        )
    else:
        change = np.asarray(model.sf(lo), dtype=float) - np.asarray(
            model.sf(hi), dtype=float
        )
    return np.clip(change.reshape(x.shape) / (hi - lo), 0.0, None)


def filled_edges(knots: np.ndarray, start: float, end: float) -> np.ndarray:
    """The pieces of an integral from ``start`` to ``end``: at the
    ``knots`` within, and at least eight to a decade from the first."""
    inside = knots[(knots > start) & (knots < end)]
    low = max(start, float(inside[0]) if inside.size else end * 1e-6)
    if low <= 0.0:
        low = end * 1e-6
    decades = max(np.log10(end / low), 1.0)
    filler = np.geomspace(low, end, int(np.ceil(decades * 8)) + 1)
    return np.unique(np.concatenate([[start, end], inside, filler]))


def integrated_shares(
    parts: Callable[[np.ndarray], np.ndarray],
    knots: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Each column's share of the integral of ``parts`` (``parts(t)``: a
    row per time, a column per element, each the density with which that
    element's failure fails the system) from 0 to each of ``ends``: the
    Barlow-Proschan importances by those times, a row each; nan where the
    integrals' total is 0. The integrals are summed in order of the ends,
    by adaptive quadrature on pieces that start at the ``knots`` (see
    ``_rates.integral``)."""
    from repyability.rbd import _rates

    order = np.argsort(ends)
    totals = np.zeros((len(ends), int(np.shape(parts(np.ones(1)))[1])))
    running = np.zeros(totals.shape[1])
    start = 0.0
    for k in order:
        end = float(ends[k])
        if end > start:
            running = running + _rates.integral(
                parts, filled_edges(knots, start, end)
            )
            start = end
        totals[k] = running
    whole = totals.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(whole[:, None] > 0.0, totals / whole[:, None], np.nan)


def life_end(
    sf: Callable[[np.ndarray], np.ndarray], knots: np.ndarray
) -> float:
    """A time by which what is left of a system's failures is negligible:
    its reliability ``sf`` (of an array of times) within ``1e-14`` of
    what it never loses, found by doubling from the last of the
    ``knots``."""
    never = float(np.ravel(sf(np.array([1e300])))[0])
    start = float(knots[-1]) if knots.size else 1.0
    tail = start * 2.0 ** np.arange(int(np.log2(1e300 / start)) + 1)
    values = np.ravel(sf(tail))
    ended = np.flatnonzero(values - never <= 1e-14)
    return float(tail[ended[0]] if ended.size else tail[-1])


def invert_reliability(
    sf: Callable[[float], float], target, upper_bound: Optional[float] = None
):
    """Solve ``sf(t) == target`` for ``t >= 0``: the time a system's
    reliability falls to ``target``; for each target, as an array, if
    ``target`` is one.

    ``sf`` maps a time to the system's reliability, non-increasing, so the
    solution is unique: the crossing is bracketed between 0 and
    ``upper_bound``, found by doubling from 1 unless given, and found with
    Brent's method. Raises ``ValueError`` if ``target`` is not in (0, 1),
    exceeds the reliability at 0 (so it is never reached), or cannot be
    bracketed.
    """
    if np.ndim(target) > 0:
        targets = np.asarray(target, dtype=float)
        return np.array(
            [
                invert_reliability(sf, float(t), upper_bound)
                for t in targets.ravel()
            ]
        ).reshape(targets.shape)
    if not 0.0 < target < 1.0:
        raise ValueError("target reliability must be in (0, 1).")
    r0 = float(sf(0.0))
    if target > r0:
        raise ValueError(
            f"target reliability {target} exceeds the system reliability "
            f"at t=0 ({r0:.6g}); it is never reached."
        )

    def f(t):
        return float(sf(t)) - target

    hi = upper_bound
    if hi is None:
        hi = 1.0
        for _ in range(1000):
            if f(hi) < 0.0:
                break
            hi *= 2.0
        else:
            raise ValueError(
                "Could not bracket the target reliability; pass an "
                "explicit upper_bound."
            )
    return float(brentq(f, 0.0, hi))


def cumulative_hazard(sf, ff) -> np.ndarray:
    """The cumulative hazard ``-ln R`` from a system's probabilities of
    working ``sf`` and of failing ``ff`` (see ``NonRepairableRBD.Hf``):
    ``-log1p(-F)`` while it is more likely to work than not, so that a
    small one keeps its precision, and ``+inf`` where it cannot work."""
    sf, ff = np.asarray(sf, dtype=float), np.asarray(ff, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(ff < 0.5, -np.log1p(-np.minimum(ff, 0.5)), -np.log(sf))


def density_by_differences(at, x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A failure density at the times ``x`` by central differences of
    step ``h`` of ``at(t)``, the probabilities of working and of failing
    at the times ``t`` (see ``NonRepairableRBD.df``): the lower point
    clipped at 0 (the difference is one-sided near 0), the change taken of
    whichever of the two is the smaller (it keeps its precision), and a
    negative result (rounding) clipped to 0."""
    x_hi, x_lo = x + h, np.maximum(x - h, 0.0)
    sf_lo, ff_lo = at(x_lo)
    sf_hi, ff_hi = at(x_hi)
    change = np.where(ff_hi <= 0.5, ff_hi - ff_lo, sf_lo - sf_hi)
    return np.clip(change / (x_hi - x_lo), 0.0, None)


def rate_parts(
    importance: Mapping[Any, Any],
    density_of: Callable[[Any, np.ndarray], np.ndarray],
    x: np.ndarray,
    groups=(),
    inputs: Optional[Callable[[np.ndarray], Tuple[dict, dict]]] = None,
    fails: Optional[Callable[[dict, dict], Any]] = None,
    scale: Optional[Callable[[], float]] = None,
    held: Collection[Any] = (),
) -> Dict[Any, np.ndarray]:
    """Each element's part in the system reliability's rate of change at
    the times ``x`` (see ``NonRepairableRBD.reliability_rate``): its
    Birnbaum ``importance`` times ``density_of(element, x)``, negated; 0
    for an element ``held``, and for one that cannot matter with an
    infinite density (a Weibull life of shape below 1, at 0). And each of
    the common-cause ``groups``' part, under the tuple of its members: the
    system's change with its members' lives moved on and the rest held,
    by central differences of ``fails(p, q)``, the system's probability
    of failing at the elements' probabilities of working and of failing
    that ``inputs(t)`` gives, the step at 0 sized by ``scale()``, a
    typical failure time."""
    members = {m for group in groups for m in group.members}
    parts: Dict[Any, np.ndarray] = {}
    for node, value in importance.items():
        if node in members:
            continue
        value = np.asarray(value, dtype=float).reshape(x.shape)
        if node in held:
            parts[node] = np.zeros(x.shape)
            continue
        with np.errstate(invalid="ignore"):
            part = -value * density_of(node, x)
        parts[node] = np.where(value == 0.0, 0.0, part)
    if groups:
        assert inputs is not None and fails is not None and scale is not None
        step = difference_steps(x, 1e-6, scale)
        hi, lo = x + step, np.maximum(x - step, 0.0)
        p, q = inputs(x)
        ends = [inputs(t) for t in (hi, lo)]
        for group in groups:
            downs = []
            for moved_p, moved_q in ends:
                given, failing = dict(p), dict(q)
                for member in group.members:
                    given[member] = moved_p[member]
                    failing[member] = moved_q[member]
                downs.append(
                    np.asarray(fails(given, failing), dtype=float).reshape(
                        x.shape
                    )
                )
            parts[tuple(group.members)] = -(downs[0] - downs[1]) / (hi - lo)
    return parts


def barlow_proschan(
    parts: Callable[[np.ndarray], Dict[Any, np.ndarray]],
    models: Iterable[Any],
    x,
    end: Callable[[np.ndarray], float],
) -> Dict[Any, Union[float, np.ndarray]]:
    """Each element's (and common-cause group's) Barlow-Proschan
    importance (see ``NonRepairableRBD.barlow_proschan_importance``): its
    part of the system's failure density, ``parts(t)`` (see
    ``rate_parts``), negated and integrated up to each time of ``x``, or
    over the whole life (``x`` None), which ends at ``end(knots)``, over
    the sum of the parts (``integrated_shares``); the integrals split at
    the ``models``' knots (``model_knots``)."""
    keys = list(parts(np.ones(1)))

    def columns(t: np.ndarray) -> np.ndarray:
        found = parts(t)
        return -np.column_stack([found[key] for key in keys])

    knots = np.concatenate([np.empty(0)] + [model_knots(m) for m in models])
    knots = np.unique(knots[np.isfinite(knots) & (knots > 0.0)])
    if x is None:
        ends = np.array([end(knots)])
    else:
        ends = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        if np.any(~np.isfinite(ends)) or np.any(ends < 0.0):
            raise ValueError(
                f"Times must be finite and non-negative, got {x!r}."
            )
    shares = integrated_shares(columns, knots, ends)
    if x is None or np.ndim(x) == 0:
        return {key: float(shares[0, j]) for j, key in enumerate(keys)}
    shape = np.shape(x)
    return {key: shares[:, j].reshape(shape) for j, key in enumerate(keys)}
