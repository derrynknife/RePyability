"""surpyval's fitted degradation processes as component lives (#271).

A ``WienerProcess`` or ``GammaProcess`` fitted in surpyval models a unit's
degradation level ``Y(t)`` from its starting level ``y0``; the unit fails
when the level first reaches the model's ``threshold``. As a component's
life it is that first-passage time, which surpyval gives from any level:
``sf``, ``qf`` and ``mean`` take the level now as ``y0``.

Replacement on condition by level (a spec's ``"preventive": {"policy":
"condition", "level": ...}``) inspects the level and replaces a unit found
at or past the level; between inspections it needs the level at the next
inspection given that the unit has not failed by then, which surpyval does
not give yet (SurPyval#836). :func:`level_after` works it out from the
fitted parameters, a workaround to drop once surpyval has it:

- a Wiener process, ``Y(t) = y + mu t + sigma B(t)``: the level at ``t``
  of a path that has not reached the threshold ``L`` by then, whose
  density is the free one less its reflection in ``L`` (the method of
  images), ``phi(a) - exp(2 mu (L - y) / sigma^2) phi(b)``;
- a gamma process, whose increment over ``t`` is Gamma(``alpha t``,
  rate ``beta``): monotone, so not having failed by ``t`` is the level
  being below ``L`` at ``t``, a gamma truncated there.
"""

import math

import numpy as np
from scipy.special import gammainc, gammaincinv, log_ndtr, ndtr

#: The bisection's steps for a Wiener level: enough to settle a float.
_STEPS = 200


def _kinds():
    from surpyval.degradation.process_models import (
        GammaProcessModel,
        WienerProcessModel,
    )

    return WienerProcessModel, GammaProcessModel


def is_degradation(model) -> bool:
    """Whether ``model`` is a surpyval Wiener or gamma degradation process
    (fitted, or built), as a component life."""
    return isinstance(model, _kinds())


def check_life(model, label: str) -> None:
    """Raise if a degradation process cannot be a component's life: one
    fitted with stress covariates needs them, which a component does not
    give it."""
    if getattr(model, "is_accelerated", False):
        raise ValueError(
            f"{label}: its degradation process was fitted with stress "
            "covariates, which a component's life does not take; fit it "
            "at the stress the component runs at."
        )


def level_after(model, y: float, t: float, u: float) -> float:
    """The level at time ``t`` from now of a unit at level ``y`` now that
    has not failed (reached ``model.threshold``) by then, from the uniform
    ``u``: the inverse of its distribution function (see the module
    docstring). A workaround for SurPyval#836."""
    threshold = float(model.threshold)
    if t <= 0.0:
        return y
    wiener, _ = _kinds()
    if isinstance(model, wiener):
        return _wiener_level(model, y, t, u, threshold)
    alpha, beta = (float(p) for p in model.params)
    # P(increment < L - y) is the survival; the increment below it is a
    # gamma truncated there.
    survive = gammainc(alpha * t, beta * (threshold - y))
    return y + float(gammaincinv(alpha * t, u * survive)) / beta


def _wiener_level(model, y: float, t: float, u: float, threshold: float):
    """A Wiener level (see :func:`level_after`), by bisection of its
    distribution function on ``(-inf, threshold)``."""
    mu, sigma = (float(p) for p in model.params)
    gap = threshold - y
    spread = sigma * math.sqrt(t)
    # log of exp(2 mu gap / sigma^2), the reflection's weight.
    weight = 2.0 * mu * gap / sigma**2

    def below(z: float) -> float:
        """P(Y(t) <= z and no crossing by t), for z <= threshold."""
        free = ndtr((z - y - mu * t) / spread)
        image = (z - 2.0 * threshold + y - mu * t) / spread
        return free - math.exp(weight + float(log_ndtr(image)))

    survive = below(threshold)
    target = u * survive
    # A bracket: the free path's mean less a wide margin, up to L.
    lo = min(y + mu * t, threshold) - 40.0 * spread
    hi = threshold
    for _ in range(_STEPS):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if below(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def failure_from(model, y: float, u: float) -> float:
    """The time to failure of a unit at level ``y`` now, from the uniform
    ``u``: surpyval's first-passage quantile from ``y``."""
    return float(np.ravel(model.qf(np.array([u]), y0=y))[0])
