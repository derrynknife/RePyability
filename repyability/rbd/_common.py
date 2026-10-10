"""Small helpers the ``RepairableRBD`` modules share: the checks of a
horizon and a discount rate, a model's mean, survival and cumulative
distribution as arrays, a constant failure rate, the common period of
inspection intervals, and the shapes results take.
"""

import functools
import math
import numbers
import warnings
from fractions import Fraction
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
)

import numpy as np

from repyability.rbd._model_utils import (
    MODEL_ERRORS,
    model_mean,
)
from repyability.utils.checks import (
    number_or_nan,
    real_array,
)
from repyability.utils.wrappers import outside_level


def _safe_mean(model) -> float:
    """A model's mean (see ``model_mean``), or NaN if it has none."""
    try:
        return model_mean(model)
    except MODEL_ERRORS:
        return float("nan")


def _constant_rate(model) -> Optional[float]:
    """The failure rate of a model whose rate is constant (an exponential
    life, or a Weibull of shape 1), or None."""
    try:
        t = model_mean(model) * np.array([0.01, 0.5, 1.0, 3.0])
        hazard = np.asarray(model.hf(t), dtype=float).ravel()
        survival = np.asarray(model.sf(t), dtype=float).ravel()
    except MODEL_ERRORS:
        return None
    rate = float(hazard[0])
    if not (np.isfinite(rate) and rate > 0.0):
        return None
    constant = np.allclose(hazard, rate, rtol=1e-9, atol=0.0)
    exponential = np.allclose(survival, np.exp(-rate * t), rtol=1e-9)
    return rate if constant and exponential else None


def _times_like(rbd, value) -> bool:
    """Whether ``value``, given where node names go, can only be times: a
    number, or numbers none of which is a node of ``rbd`` (#224)."""
    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return False
    if isinstance(value, numbers.Real):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf":
            return False
        items = value.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
        if not items or not all(
            isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))
            for v in items
        ):
            return False
    else:
        return False
    nodes = set(rbd.nodes)
    return not any(v in nodes for v in items)


def _times_first(target: str = "x"):
    """Let a repairable diagram's measure take its times first, as a
    non-repairable diagram's does (#224): a number, or numbers none of
    which is a node, given where the first argument goes (node names, or
    Fussell-Vesely's ``fv_type``) is ``target`` (its times ``x``, or a
    window's length)."""

    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if args and _times_like(self, args[0]):
                if kwargs.get(target) is not None:
                    raise TypeError(
                        f"{method.__name__}() was given times both first "
                        f"and as {target}=: give them once, as {target}=."
                    )
                kwargs[target] = args[0]
                args = args[1:]
            return method(self, *args, **kwargs)

        return wrapper

    return decorate


def _fixed_length(model) -> Optional[float]:
    """The length of a time that is always the same (an
    ``ExactEventTime``, say), or None."""
    try:
        mean = float(model_mean(model))
        if not (np.isfinite(mean) and mean > 0.0):
            return None
        below, above = np.asarray(
            model.ff(np.array([mean * (1.0 - 1e-9), mean * (1.0 + 1e-9)])),
            dtype=float,
        ).ravel()
    except MODEL_ERRORS:
        return None
    return mean if below <= 1e-12 and above >= 1.0 - 1e-12 else None


def _repair_rate(model) -> Optional[float]:
    """The rate of an exponential repair time, ``inf`` for an instant one
    (in no time), or None."""
    if _safe_mean(model) == 0.0:
        return math.inf
    return _constant_rate(model)


def _cdf(model) -> Callable[[np.ndarray], np.ndarray]:
    """A model's CDF, for arrays of times (0 where it gives none)."""

    def cdf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            values = np.asarray(model.ff(x), dtype=float).reshape(x.shape)
        return np.nan_to_num(values, nan=0.0)

    return cdf


def _product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The distribution of the sum of two independent counts, with ``a``
    and ``b`` for ``0, 1, 2, ...``, its negligible tail cut."""
    from scipy.signal import fftconvolve

    out = np.maximum(fftconvolve(a, b), 0.0)
    keep = np.nonzero(out > 1e-16)[0]
    return out[: keep[-1] + 1] if keep.size else np.ones(1)


def _fleet(probabilities: np.ndarray, fleet: int) -> np.ndarray:
    """The distribution of the sum of ``fleet`` independent counts, each
    with ``probabilities`` for ``0, 1, 2, ...``."""
    total, power = np.ones(1), np.asarray(probabilities, dtype=float)
    while fleet:
        if fleet & 1:
            total = _product(total, power)
        fleet >>= 1
        if fleet:
            power = _product(power, power)
    return total / total.sum()


def _summed(counts) -> np.ndarray:
    """The distribution of the sum of independent counts, each given by its
    probabilities for ``0, 1, 2, ...``."""
    total = np.ones(1)
    for count in counts:
        total = _product(total, np.asarray(count, dtype=float))
    return total / total.sum()


def _fractions(counts: np.ndarray) -> np.ndarray:
    """The fractions of simulations with each count ``0, 1, 2, ...``."""
    counts = np.asarray(counts, dtype=np.int64)
    return np.bincount(counts) / len(counts)


def _whole(name: str, value, least: int = 1) -> int:
    """``value`` as a whole number of at least ``least``, or a
    ValueError naming it."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or int(value) != value
        or value < least
    ):
        raise ValueError(
            f"{name} must be a whole number, {least} or more; got {value!r}."
        )
    return int(value)


def _curve_points(value) -> Optional[int]:
    """``curve_points`` checked: None, or a whole number of at least 1."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"curve_points must be a whole number of grid steps, got "
            f"{value!r}."
        )
    if value < 1:
        raise ValueError(f"curve_points must be at least 1, got {value!r}.")
    return int(value)


def _horizon(horizon) -> float:
    """``horizon`` as a finite, non-negative float, or a ValueError (text
    that reads as a number too, #233)."""
    value = number_or_nan(horizon)
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(
            f"horizon must be a finite, non-negative number, got {horizon!r}."
        )
    return value


def _discount_rate(discount_rate) -> float:
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


def _horizons(horizon, rate: float) -> np.ndarray:
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
_DISCOUNTED_AWAY = 20.0


_HORIZON_AWAY = 1000.0


def _present_horizon(
    horizons: np.ndarray,
    rate: float,
    lives: Callable[[], List[float]] = list,
) -> np.ndarray:
    """What a unit cost rate over ``[0, horizon)`` is worth at the start,
    discounted continuously at ``rate`` per unit time (#184): ``(1 -
    exp(-rate * horizon)) / rate``, the horizon itself undiscounted, and
    ``1 / rate`` for an endless one; for each of ``horizons``. Warns where
    the rate discounts the costs away (#231): times the shortest of the
    components' mean ``lives`` more than ``_DISCOUNTED_AWAY``, or times a
    horizon more than ``_HORIZON_AWAY``."""
    if rate == 0.0:
        return horizons
    known = [life for life in lives() if 0.0 < life < math.inf]
    finite = horizons[np.isfinite(horizons)]
    span, what = 0.0, ""
    if known and rate * min(known) > _DISCOUNTED_AWAY:
        span, what = min(known), "the components' shortest mean life"
    elif finite.size and rate * float(finite.max()) > _HORIZON_AWAY:
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


#: How closely a discounted expected cost from new is worked out (#231):
#: its integral by parts, to this share of itself.
_DISCOUNT_TOLERANCE = 1e-8


#: The most rounds of halving its pieces: a jump of the expected cost (a
#: scheduled event's) is closed in on by a half each round.
_DISCOUNT_ROUNDS = 60


def _discount_pieces(ends: np.ndarray, horizon: float):
    """The first pieces of a discounted cost's integral (see
    ``RepairableRBD._discounted_cost``): from 0 to ``horizon`` in 32,
    and split at each window's end, as ``(starts, stops)``."""
    edges = np.unique(
        np.concatenate(([0.0], ends, np.linspace(0.0, horizon, 33)))
    )
    return edges[:-1], edges[1:]


def _common_period(intervals: Iterable[float]) -> float:
    """The least common multiple of the inspection intervals: the period
    after which the schedules repeat together."""
    intervals = sorted(set(intervals))
    if len(intervals) == 1:
        return intervals[0]
    fractions = [Fraction(x).limit_denominator(10**6) for x in intervals]
    if any(
        abs(float(f) - x) > 1e-12 * x for f, x in zip(fractions, intervals)
    ):
        raise NotImplementedError(
            f"The inspection intervals {intervals} have no common period, "
            "so the long-run values cannot be averaged over one: estimate "
            "them by simulation, with availability() or cost()."
        )
    numerator = math.lcm(*(f.numerator for f in fractions))
    denominator = math.gcd(*(f.denominator for f in fractions))
    return numerator / denominator


def _sf_values(sf, x: np.ndarray) -> np.ndarray:
    """A survival function's values at ``x``, as floats in ``x``'s shape."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(sf(x), dtype=float).reshape(np.shape(x))


def _matched(times: np.ndarray, at: np.ndarray, values: np.ndarray):
    """``values``, given at the times ``at``, summed at each of ``times``
    (sorted) that they fall on exactly; 0 at the others."""
    index = np.searchsorted(times, at)
    hit = index < len(times)
    hit[hit] = times[index[hit]] == at[hit]
    return np.bincount(index[hit], values[hit], len(times))


def _shaped(values: np.ndarray, t, times: np.ndarray):
    """``values``, one per time, as a float for a scalar ``t``, else in the
    shape of ``times``."""
    values = np.asarray(values, dtype=float)
    if np.ndim(t) == 0:
        return float(values[0])
    return values.reshape(times.shape)


def _squeeze_values(d: dict) -> dict:
    """Convert a dict of 1-element arrays (as produced by the base RBD
    importance helpers) to a dict of plain floats. Steady-state availability
    has no time dimension, so the repairable importances return floats."""
    return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in d.items()}


def _safe_ratio(numerator, denominator):
    """Ratio that is 0 when the denominator is 0.

    Several criticality/importance measures divide by a total (system uptime,
    downtime, failures, restorations, or a union of intervals) that can
    legitimately be 0 -- e.g. when a redundant component is forced working so
    the system never fails, or a component is forced broken. With nothing to
    attribute, the measure is 0 rather than a NaN/inf (or a crash).
    """
    return numerator / denominator if denominator else 0.0
