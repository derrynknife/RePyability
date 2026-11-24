"""surpyval's fitted degradation processes as component lives (#271).

A ``WienerProcess`` or ``GammaProcess`` fitted in surpyval models a unit's
degradation level ``Y(t)`` from its starting level ``y0``; the unit fails
when the level first reaches the model's ``threshold``. As a component's
life it is that first-passage time, which surpyval gives from any level:
``sf``, ``qf`` and ``mean`` take the level now as ``y0``.

Replacement on condition by level (a spec's ``"preventive": {"policy":
"condition", "level": ...}``) inspects the level and replaces a unit found
at or past the level; between inspections it needs the level at the next
inspection given that the unit has not failed by then, which surpyval
gives from 0.29 (``level_at(..., given_survival=True)``, the method of
images for a Wiener path; a gamma increment truncated at the
threshold): the simulation draws it by its quantile function
(:func:`level_after`).

The numerical values (``_level_replacement``) take differences of
:func:`level_below` over a grid of levels: the probability that a unit at
level ``y`` now is, a time ``t`` later, unfailed and at a level at most
``z``, from the fitted parameters, broadcast over arrays of levels and
times (surpyval's ``ProcessLevel`` takes one starting level and time at a
time). The tests check it against surpyval's.
"""

import sys

import numpy as np
from scipy.special import gammainc, log_ndtr, ndtr


def _kinds():
    from surpyval.degradation import GammaProcessModel, WienerProcessModel

    return WienerProcessModel, GammaProcessModel


def is_degradation(model) -> bool:
    """Whether ``model`` is a surpyval Wiener or gamma degradation process
    (fitted, or built), as a component life.

    No model can be one before ``surpyval.degradation`` is imported, so
    until then it is not imported to check: it loads surpyval's
    regression code with pandas, which took about 2 s of a session's
    first diagram."""
    if "surpyval.degradation" not in sys.modules:
        return False
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


def level_below(model, y, t, z):
    """The probability that a unit at level ``y`` now has not failed (not
    reached ``model.threshold``) a time ``t > 0`` from now, and is then at
    a level at most ``z``: at ``z`` at or past the threshold, its
    survival. Arrays broadcast. surpyval's ``level_at(...,
    given_survival=True)`` is this over its value at the threshold, for
    one ``y`` and ``t``."""
    threshold = float(model.threshold)
    a, b = (float(p) for p in model.params)
    wiener, _ = _kinds()
    z = np.minimum(z, threshold)
    if isinstance(model, wiener):
        return _wiener_below(a, b, threshold, y, t, z)
    return _gamma_below(a, b, y, t, z)


def _wiener_below(mu, sigma, threshold, y, t, z):
    """``level_below`` for a Wiener process, ``z`` at most the threshold:
    the free path's distribution less its reflection in the threshold."""
    spread = sigma * np.sqrt(t)
    # log of exp(2 mu gap / sigma^2), the reflection's weight.
    weight = 2.0 * mu * (threshold - y) / sigma**2
    free = ndtr((z - y - mu * t) / spread)
    image = (z - 2.0 * threshold + y - mu * t) / spread
    return free - np.exp(weight + log_ndtr(image))


def _gamma_below(alpha, beta, y, t, z):
    """``level_below`` for a gamma process, ``z`` at most the threshold:
    monotone, so unfailed at a level at most ``z`` is an increment of at
    most ``z - y``."""
    return gammainc(alpha * t, beta * np.maximum(z - y, 0.0))


def level_after(model, y: float, t: float, u: float) -> float:
    """The level at time ``t`` from now of a unit at level ``y`` now that
    has not failed (reached ``model.threshold``) by then, from the uniform
    ``u``: surpyval's quantile of the level given survival
    (``level_at(..., given_survival=True)``)."""
    if t <= 0.0:
        return y
    level = model.level_at(t, y0=y, t0=0.0, given_survival=True)
    return float(np.ravel(level.qf(u))[0])


def failure_from(model, y: float, u: float) -> float:
    """The time to failure of a unit at level ``y`` now, from the uniform
    ``u``: surpyval's first-passage quantile from ``y``."""
    return float(np.ravel(model.qf(np.array([u]), y0=y))[0])
