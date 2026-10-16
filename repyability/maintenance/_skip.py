"""The value of skipping the next planned replacement right after a
failure (#319, UC-6 of this package's ``README.md``).

A part is replaced when it fails (unplanned, at its ``unplanned_cost``)
and on a plan (planned, at its ``planned_cost``), in no time either way.
Right after a failure a new unit is in, and the question is whether to
skip the next planned replacement. Each choice is costed in present value,
at a continuous discount rate ``r`` a unit of the life's time (0 by
default: undiscounted), and the NPV of skipping is what it saves: the cost
of doing the replacement, less that of skipping it.

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
cancels from the last term, which holds with no discounting too. The two
present values compared are those of the whole future: ``V`` if the
replacement is done, and ``A(2T) + (1 - r M(2T)) V`` if it is skipped
(the longer first cycle, then the plan from new). Undiscounted both are
infinite (``replace`` and ``skip`` are ``inf``), and only their
difference is finite, the classic comparison: each choice's first
cycle's cost less its length at the plan's cost rate ``g = A(T) /
M(T)``, ``A(T) - g M(T)`` (0, the plan's own) against ``A(2T) - g
M(2T)``. Either way the difference is the NPV above. A discount rate
that discounts the costs away (a rate a year with a life in days, #231)
is warned of as a ``RepairableRBD``'s costs warn of it.

**A plan that does not reset** (block replacement, on a calendar):
planned replacements fall on fixed times, the next ``d`` after the
failure. Done, a new unit goes in at ``d``; skipped, the unit in now runs
on. Both replace at the following time, ``d + T``, so the two differ only
before it, and are compared to it::

    NPV = planned exp(-r d) + exp(-r d) F(T) - (F(d + T) - F(d))

with ``F(t)`` the expected discounted cost of the failures of a new unit
and its replacements in ``[0, t)``: a
[`RepairableRBD`][repyability.RepairableRBD]'s ``expected_cost`` of a
component replaced in no time (exact, no simulation).
"""

import math
from dataclasses import dataclass

import numpy as np

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks


def _time(value, name: str) -> float:
    """``value`` as a time: a finite number, more than 0."""
    if not (checks.is_number(value) and 0.0 < float(value) < math.inf):
        raise ValueError(
            f"{name} must be a time, a number more than 0, in the life's "
            f"time unit; got {value!r}."
        )
    return float(value)


@dataclass
class SkipValue(_ResultMapping):
    """Whether to skip the next planned replacement of a part, right after
    a failure: returned by
    [`NonRepairable.skip_next_replacement`][repyability.NonRepairable.skip_next_replacement].

    Attributes
    ----------
    resets : bool
        Whether the plan resets at each replacement.
    every : float
        The plan's interval, in the life's time unit.
    time_to_next : float
        The time from the failure to the next planned replacement:
        ``every`` for a plan that resets.
    discount_rate : float
        The continuous discount rate a unit of the life's time.
    replace : float
        The present-value cost of doing the next planned replacement.
        On a plan that does not reset, the costs to the following planned
        replacement: the replacement, the new unit's failures to then and
        the failures before it. On a plan that resets, the cost of the
        whole future (the first cycle, then the plan from new), which
        with no discounting is endless: ``inf``.
    skip : float
        The present-value cost of skipping it, over the same comparison:
        the failures to the following planned replacement; or, on a plan
        that resets, the whole future from the longer first cycle
        (``inf`` with no discounting).
    npv : float
        What skipping saves, ``replace - skip``: the planned cost saved,
        net of the failures it adds. Positive, skipping pays. Finite
        where both are endless: the first cycles' difference, each
        cycle's cost less its length at the plan's cost rate.
    best : str
        ``"skip"`` where skipping saves, ``"replace"`` otherwise.
    """

    resets: bool
    every: float
    time_to_next: float
    discount_rate: float
    replace: float
    skip: float
    npv: float
    best: str


def skip_next_replacement(
    part, every, *, resets, time_to_next=None, discount_rate=0.0
) -> SkipValue:
    """The work of ``NonRepairable.skip_next_replacement`` (see there)."""
    part._require_costs()
    if part.model_parameterization not in ("parametric", "degradation"):
        raise NotImplementedError(
            "skip_next_replacement needs a life with a survival function in "
            "closed form (a surpyval parametric distribution, mixture or "
            "degradation process); this part's is "
            f"{part.model_parameterization}."
        )
    every = _time(every, "every")
    if not isinstance(resets, bool):
        raise TypeError(
            "resets must be True (the plan resets at each replacement) or "
            f"False (it keeps to fixed times); got {resets!r}."
        )
    if not (
        checks.is_number(discount_rate)
        and 0.0 <= float(discount_rate) < math.inf
    ):
        raise ValueError(
            "discount_rate must be a continuous rate a unit of the life's "
            f"time, at least 0; got {discount_rate!r}."
        )
    rate = float(discount_rate)
    if resets:
        if time_to_next is not None:
            raise ValueError(
                "A plan that resets has its next planned replacement every "
                f"({every:g}) after the failure: give no time_to_next."
            )
        time_to_next = every
        replace, skip, npv = _resetting(part, every, rate)
    else:
        if time_to_next is None:
            raise ValueError(
                "A plan on fixed times needs time_to_next: the time from "
                "the failure to the next planned replacement."
            )
        time_to_next = _time(time_to_next, "time_to_next")
        if time_to_next > every:
            raise ValueError(
                f"time_to_next ({time_to_next:g}) must be at most every "
                f"({every:g}): the next planned replacement is within one "
                "interval."
            )
        replace, skip = _fixed(part, every, rate, time_to_next)
        npv = replace - skip
    return SkipValue(
        resets,
        every,
        time_to_next,
        rate,
        replace,
        skip,
        npv,
        "skip" if npv > 0.0 else "replace",
    )


def _warn_discounted_away(part, rate: float, horizons) -> None:
    """The check a ``RepairableRBD``'s costs make of a discount rate
    (#231): warn where it discounts the costs at the life's mean (or
    over ``horizons``) to almost nothing."""
    from repyability.rbd._common import _present_horizon
    from repyability.rbd._model_utils import MODEL_ERRORS, model_mean

    def lives() -> list:
        try:
            return [float(model_mean(part.reliability))]
        except MODEL_ERRORS:
            return []

    _present_horizon(np.asarray(horizons, float), rate, lives)


def _resetting(part, every: float, rate: float) -> tuple:
    """The present-value costs of doing and of skipping the next planned
    replacement on a plan that resets, and the NPV of skipping (see the
    module's notes)."""
    from repyability.rbd._mean_lifetime import _integrate, model_knots

    _warn_discounted_away(part, rate, [every, 2.0 * every])
    sf = part.reliability_function

    def discounted(x: np.ndarray) -> np.ndarray:
        survival = np.asarray(sf(x), dtype=float).ravel()
        return np.exp(-rate * x) * survival

    knots = model_knots(part.reliability)

    def length(cut: float) -> float:
        inside = knots[(knots > 0.0) & (knots < cut)]
        return _integrate(discounted, np.concatenate(([0.0], inside, [cut])))

    def cost(cut: float, held: float) -> float:
        kept = math.exp(-rate * cut) * float(np.ravel(sf(cut))[0])
        failed = 1.0 - kept - rate * held
        return part.cu * failed + part.cp * kept

    once, twice = length(every), length(2.0 * every)
    done, skipped = cost(every, once), cost(2.0 * every, twice)
    if rate > 0.0:
        # The whole future: the plan from new is worth A(T) / (r M(T)),
        # and skipping puts it after the longer first cycle.
        value = done / (rate * once)
        skip = skipped + (1.0 - rate * twice) * value
        return value, skip, value - skip
    # Undiscounted, each whole future is endless, and the NPV is the
    # first cycles' difference, each cycle's cost less its length at the
    # plan's cost rate (the plan's own cycle is 0 by construction).
    g = done / once
    return math.inf, math.inf, (done - g * once) - (skipped - g * twice)


def _fixed(part, every: float, rate: float, time_to_next: float) -> tuple:
    """The present-value costs of doing and of skipping the next planned
    replacement on a plan on fixed times, the next ``time_to_next`` after
    the failure, to the following one (see the module's notes)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    unit = RepairableRBD(
        [("s", "unit"), ("unit", "t")],
        {
            "unit": {
                "reliability": part.reliability,
                "repairability": "instant",
                "replace_cost": part.cu,
            }
        },
    )
    windows = np.array([every, time_to_next, time_to_next + every])
    interval, before, after = unit.expected_cost(
        windows, discount_rate=rate
    ).mean
    later = math.exp(-rate * time_to_next)
    return float(part.cp * later + later * interval + before), float(after)
