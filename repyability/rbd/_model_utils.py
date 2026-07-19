"""Helpers describing the *capabilities* of the reliability models the RBD
classes consume.

Historically the RBD code branched on ``model.dist.name`` string literals
(e.g. ``"FixedEventProbability"``, ``"Exponential"``) scattered across several
modules. That couples behaviour tightly to surpyval's internal naming: a
rename there silently changes results here. Centralising the checks in these
small helpers keeps that coupling in one place (easy to audit and to cover
with a compatibility test) and gives the call sites intention-revealing names.
"""

from typing import Optional

import numpy as np

# surpyval distribution names whose event probability does not vary with time,
# i.e. ``sf(t)`` is constant. These behave as fixed per-demand probabilities
# rather than lifetime distributions.
FIXED_PROBABILITY_DIST_NAMES = frozenset(
    {"FixedEventProbability", "Bernoulli"}
)


def distribution_name(model) -> Optional[str]:
    """Return the surpyval distribution name of ``model``.

    Returns ``None`` for models that do not expose a surpyval distribution
    (e.g. a ``StandbyModel``, ``RepeatedNode``, nested RBD, or the perfect
    reliability/unreliability helpers).
    """
    dist = getattr(model, "dist", None)
    if dist is None:
        return None
    return getattr(dist, "name", None)


def is_fixed_probability(model) -> bool:
    """True if ``model`` is a time-invariant (fixed) event probability."""
    return distribution_name(model) in FIXED_PROBABILITY_DIST_NAMES


def is_exponential(model) -> bool:
    """True if ``model`` is a surpyval Exponential distribution."""
    return distribution_name(model) == "Exponential"


def model_mean(model) -> float:
    """The model's mean as a plain float.

    Works around surpyval 0.11's ExactEventTime, whose ``mean()`` raises
    AttributeError (its underlying dist has no ``mean``); for an exact event
    time the mean is simply its parameter.
    """
    try:
        return float(np.atleast_1d(model.mean())[0])
    except AttributeError:
        if distribution_name(model) == "ExactEventTime":
            return float(np.atleast_1d(model.params)[0])
        raise


def parametric_spec(model):
    """Return ``(surpyval_class, params, param_names)`` for a parametric node
    model, or ``None`` when it has no reconstructable distribution parameters
    (a ``StandbyModel``, ``RepeatedNode``, nested RBD, a repeated node's source
    name, the perfect-reliability helpers, or a fitted non-parametric model).

    Used by parameter-sensitivity analysis to rebuild a distribution with a
    perturbed parameter. It is faithful for exactly the models
    ``serialisation`` round-trips (surpyval parametric distributions), since it
    goes through the same ``dist name`` + ``from_params`` reconstruction.
    """
    import surpyval

    name = distribution_name(model)
    if name is None:
        return None
    cls = getattr(surpyval, name, None)
    if cls is None or not hasattr(cls, "from_params"):
        return None
    params = [float(p) for p in np.atleast_1d(model.params)]
    dist = getattr(model, "dist", None)
    names = getattr(dist, "param_names", None)
    if not names or len(list(names)) != len(params):
        names = [f"param{i}" for i in range(len(params))]
    return cls, params, list(names)
