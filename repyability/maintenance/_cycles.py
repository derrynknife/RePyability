"""A repair, overhaul and replace policy with cycles (the K-cycle policy).

A unit is repaired minimally at each failure (cost ``cr``: it goes on as bad
as it was), overhauled every ``T`` time units (cost ``co``: the overhaul
takes off part of its age), and, after ``K`` such periods, replaced by a new
unit (cost ``cp``). A replacement cycle is therefore ``K`` periods of
length ``T``, and the cycle repeats.

In plain words: the overhaul makes the unit younger but not new, so each
period starts older than the one before, up to the replacement, which puts
the clock back to zero. The cost rate balances the repairs (which get more
frequent as the unit ages), the overhauls and the replacement, against how
long a replacement cycle lasts.

**The model.** Failures follow a non-homogeneous Poisson process (minimal
repair) with cumulative intensity ``Lambda(a)`` in the unit's age ``a``
(the model's ``cif``). An overhaul keeps a fraction ``f`` of the unit's age
(``0 <= f < 1``; ``f = 0``: as new; ``f`` near 1: little effect), so the
age at the start of period ``j`` is ``v_0 = 0`` and ``v_j = f (v_(j-1) +
T)``. With minimal repair the failures of period ``j`` are an NHPP piece,
with expected number ``Lambda(v_(j-1) + T) - Lambda(v_(j-1))``. A
replacement cycle costs ``cr`` times the repairs, ``co`` for each of its
``K - 1`` overhauls and ``cp`` for the replacement, and lasts ``K T``.
Renewal-reward gives the long-run cost rate::

    g(T, K) = (cr * sum_j [Lambda(v_(j-1) + T) - Lambda(v_(j-1))]
               + (K - 1) co + cp) / (K T)

Repairs, overhauls and replacements take no time. With ``K = 1`` it is the
periodic replacement of ``optimal_overhaul_policy``, at the replacement
cost ``cp``. With a constant failure rate, ages do not matter and ``g``
reduces to ``cr / alpha + ((K - 1) co + cp) / (K T)``, which is lowest at
no overhauls and no replacements.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from repyability.maintenance import OverhaulReplacePolicy
from repyability.utils import checks

#: The most cycles searched when the caller gives no limit.
DEFAULT_MAX_CYCLES = 20


def _factor(value) -> float:
    if not (checks.is_number(value) and 0.0 <= float(value) < 1.0):
        raise ValueError(
            "overhaul_factor is the share of its age a unit keeps through an "
            "overhaul: a number from 0 (as new) up to, not including, 1; "
            f"got {value!r}."
        )
    return float(value)


def _cost(unit, value, name: str) -> float:
    if not (checks.is_number(value) and 0.0 <= float(value) < math.inf):
        raise ValueError(f"{name} must be a cost, at least 0; got {value!r}.")
    return float(value)


def _require_cif(unit) -> None:
    if not unit._analytic:
        raise NotImplementedError(
            "The overhaul-and-replace policy needs the expected failures "
            "of a unit repaired minimally from any age, which an analytic "
            "model gives (a cumulative intensity, cif: Crow-AMSAA, Duane "
            "or a homogeneous Poisson process). An imperfect-repair "
            "model's mean cumulative function counts failures from new "
            "only."
        )


def _lambda(unit, age: np.ndarray) -> np.ndarray:
    return np.asarray(unit.model.cif(age), dtype=float)


@dataclass(frozen=True)
class _Args:
    factor: float
    overhaul_cost: float
    replacement_cost: float


def cost_rate(unit, interval, cycles, overhaul_factor, replacement_cost):
    """See ``Repairable.overhaul_replace_cost_rate``."""
    unit._require_costs()
    _require_cif(unit)
    factor = _factor(overhaul_factor)
    cp = _cost(unit, replacement_cost, "replacement_cost")
    if not (checks.is_whole(cycles) and int(cycles) >= 1):
        raise ValueError(
            f"cycles must be a whole number, 1 or more; got {cycles!r}."
        )
    t = checks.real_array(interval, "interval")
    if not np.all(np.isfinite(t) & (t > 0.0)):
        raise ValueError(
            f"interval must be a time, more than 0; got {interval!r}."
        )
    out = _rate(unit, t, int(cycles), factor, cp)
    return float(out) if np.ndim(interval) == 0 else out


def _rate(unit, t: np.ndarray, cycles: int, factor: float, cp: float):
    """``g(T, K)`` for an array of intervals ``t``."""
    v = np.zeros_like(t, dtype=float)
    repairs = np.zeros_like(t, dtype=float)
    for _ in range(cycles):
        repairs += _lambda(unit, v + t) - _lambda(unit, v)
        v = factor * (v + t)
    total = unit.cr * repairs + (cycles - 1) * unit.co + cp
    return total / (cycles * t)


def _scale(unit, cp: float) -> float:
    """An age at which the repair spend is comparable to the replacement
    cost: where the search grid is centred."""
    target = max(cp, unit.co) / unit.cr
    t = 1.0
    if _lambda(unit, np.array([t]))[0] < target:
        for _ in range(300):
            t *= 2.0
            if _lambda(unit, np.array([t]))[0] >= target:
                break
    else:
        for _ in range(300):
            if _lambda(unit, np.array([t / 2.0]))[0] < target:
                break
            t /= 2.0
    return t


def _best_interval(
    unit, cycles: int, factor: float, cp: float, scale: float
) -> Tuple[float, float]:
    """The interval minimising ``g(., K)`` and its cost rate: ``inf`` and the
    rate at the far end of the grid where lengthening it keeps paying."""
    return _minimise(unit, lambda t: _rate(unit, t, cycles, factor, cp), scale)


def _never_replace_rate(unit, t: np.ndarray, factor: float) -> np.ndarray:
    """``g_inf(T)``: the long-run cost rate of overhauling every ``T`` and
    never replacing. The age at the start of a period settles where an
    overhaul takes off what the period added, ``v* = f T / (1 - f)``, so
    each period then expects ``Lambda(v* + T) - Lambda(v*)`` repairs and
    costs one overhaul. ``g(T, K)`` tends to it as ``K`` grows."""
    v = factor * t / (1.0 - factor)
    repairs = _lambda(unit, v + t) - _lambda(unit, v)
    return (unit.cr * repairs + unit.co) / t


def _minimise(unit, rate, scale: float) -> Tuple[float, float]:
    """The interval minimising ``rate`` (a function of an array of
    intervals) by a grid search and a bounded refinement: ``inf`` and the
    rate at the far end of the grid where lengthening it keeps paying."""
    for _ in range(8):
        grid = scale * np.logspace(-3.0, 3.0, 1201)
        with np.errstate(over="ignore", invalid="ignore"):
            # A steep intensity (Cox-Lewis's) overflows far out: no plan.
            rates = np.nan_to_num(rate(grid), nan=math.inf)
        i = int(np.argmin(rates))
        if i == len(grid) - 1:
            big = float(grid[-1])
            lam = _lambda(unit, np.array([big, 2.0 * big]))
            if (
                lam[0] <= 0.0
                or math.log(lam[1] / lam[0]) / math.log(2.0) <= 1.0 + 1e-9
            ):
                return math.inf, float(rates[-1])
            scale *= 1e3
            continue
        if i == 0:
            scale *= 1e-3
            continue

        def f(x: float) -> float:
            return float(rate(np.array([x]))[0])

        res = minimize_scalar(
            f,
            bounds=(float(grid[i - 1]), float(grid[i + 1])),
            method="bounded",
            options={"xatol": 1e-10 * float(grid[i])},
        )
        return float(res.x), f(float(res.x))
    return float(grid[i]), float(rates[i])


def _failure_limit(unit, cp: float, max_failures: int) -> Tuple[int, float]:
    """The best replace-at-the-``n``-th-failure plan at the replacement
    cost ``cp``: ``(cr (n - 1) + cp) / E[T_n]``, with ``E[T_n]`` exact for
    an analytic model. ``(0, inf)`` when ``E[T_n]`` is infinite (an
    intensity that falls away)."""
    try:
        expected = unit._exact_times_to_failures(max_failures)
    except ValueError:
        return 0, math.inf
    n = np.arange(1, len(expected) + 1)
    rates = (unit.cr * (n - 1) + cp) / expected
    i = int(np.nanargmin(rates))
    return int(n[i]), float(rates[i])


#: The plans compared, in the order a tie is settled (the first wins).
PLANS = (
    "replace only",
    "overhaul and replace",
    "overhaul only",
    "failure limit",
)


def optimal_policy(
    unit, overhaul_factor, replacement_cost, max_cycles, max_failures
) -> OverhaulReplacePolicy:
    """See ``Repairable.optimal_overhaul_replace_policy``."""
    unit._require_costs()
    _require_cif(unit)
    factor = _factor(overhaul_factor)
    cp = _cost(unit, replacement_cost, "replacement_cost")
    if not (checks.is_whole(max_cycles) and int(max_cycles) >= 1):
        raise ValueError(
            "max_cycles must be a whole number, 1 or more; got "
            f"{max_cycles!r}."
        )
    if not (checks.is_whole(max_failures) and int(max_failures) >= 1):
        raise ValueError(
            "max_failures must be a whole number, 1 or more; got "
            f"{max_failures!r}."
        )
    scale = _scale(unit, cp)
    interval, rate = _best_interval(unit, 1, factor, cp, scale)
    if math.isinf(interval):
        # No wear-out: neither overhauls nor replacements pay; run the
        # unit to failure, repairing it.
        return OverhaulReplacePolicy(
            interval=interval,
            cycles=1,
            cost_rate=rate,
            plan="run to failure",
            alternatives={"run to failure": {"cost_rate": rate}},
            saving=0.0,
        )
    found = {
        "replace only": {"interval": interval, "cycles": 1, "cost_rate": rate}
    }
    best = None
    for k in range(2, int(max_cycles) + 1):
        interval, rate = _best_interval(unit, k, factor, cp, scale)
        if best is None or rate < best[2] * (1.0 - 1e-12):
            best = (interval, k, rate)
    limit_reached = False
    if best is not None:
        limit_reached = best[1] == int(max_cycles)
        found["overhaul and replace"] = {
            "interval": best[0],
            "cycles": best[1],
            "cost_rate": best[2],
        }
    interval, rate = _minimise(
        unit, lambda t: _never_replace_rate(unit, t, factor), scale
    )
    found["overhaul only"] = {
        "interval": interval,
        "cycles": math.inf,
        "cost_rate": rate,
    }
    count, rate = _failure_limit(unit, cp, int(max_failures))
    if count:
        found["failure limit"] = {"failure_count": count, "cost_rate": rate}

    winner = PLANS[0]
    for plan in PLANS[1:]:
        if plan in found and found[plan]["cost_rate"] < found[winner][
            "cost_rate"
        ] * (1.0 - 1e-12):
            winner = plan
    others = [found[plan]["cost_rate"] for plan in found if plan != winner]
    chosen = found[winner]
    return OverhaulReplacePolicy(
        interval=chosen.get("interval", math.inf),
        cycles=chosen.get("cycles", 1),
        cost_rate=chosen["cost_rate"],
        plan=winner,
        failure_count=count if winner == "failure limit" else None,
        alternatives=found,
        saving=min(others) - chosen["cost_rate"] if others else 0.0,
        cycles_limit_reached=limit_reached,
    )
