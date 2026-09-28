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


#: A surpyval parametric model's offset, limited-failure-population and
#: zero-inflation parameters, and the values that mean it has none of them.
_EXTRAS = {"gamma": 0.0, "p": 1.0, "f0": 0.0}


def model_extras(model) -> dict:
    """The offset (``gamma``), limited-failure-population (``p``, the
    fraction of units that ever fail) and zero-inflation (``f0``, the
    fraction dead on arrival) parameters of a surpyval parametric model,
    where it has them: what ``from_params`` needs, besides the parameters,
    to rebuild it. Empty for a plain model (or any other)."""
    out = {}
    for name, none in _EXTRAS.items():
        value = getattr(model, name, none)
        if value is not None and value != none:
            out[name] = float(value)
    return out


def never_fails(model) -> float:
    """The fraction of units that never fail: ``1 - p`` for a surpyval
    limited-failure-population model, 0 for any other."""
    p = getattr(model, "p", None) if distribution_name(model) else None
    if p is None:
        return 0.0
    return max(0.0, 1.0 - float(p))


def is_exponential(model) -> bool:
    """True if ``model`` is a plain surpyval Exponential lifetime: no
    offset, every unit fails and none is dead on arrival. Only then is it
    memoryless from time 0 with rate ``1 / mean``."""
    return distribution_name(model) == "Exponential" and not model_extras(
        model
    )


def model_mean(model) -> float:
    """The model's mean as a plain float.

    Infinite for a limited-failure-population model: some of its units
    never fail. (surpyval's ``mean()`` is then the *defective* mean, the
    failing units' mean weighted by their fraction, which is not the mean
    of a lifetime.) Works around surpyval 0.11's ExactEventTime, whose
    ``mean()`` raises AttributeError (its underlying dist has no ``mean``);
    for an exact event time the mean is simply its parameter.
    """
    if never_fails(model) > 0.0:
        return float("inf")
    try:
        return float(np.atleast_1d(model.mean())[0])
    except AttributeError:
        if distribution_name(model) == "ExactEventTime":
            return float(np.atleast_1d(model.params)[0])
        raise


def parametric_spec(model):
    """Return ``(surpyval_class, params, param_names, extras)`` for a
    parametric node model, or ``None`` when it has no reconstructable
    distribution parameters (a ``StandbyModel``, ``RepeatedNode``, nested
    RBD, a repeated node's source name, the perfect-reliability helpers, or
    a fitted non-parametric model). ``extras`` are its offset,
    limited-failure-population and zero-inflation parameters (see
    ``model_extras``): ``surpyval_class.from_params(params, **extras)``
    rebuilds it.

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
    return cls, params, list(names), model_extras(model)
