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


#: The values of a surpyval parametric model's offset (``gamma``),
#: limited-failure-population (``p``) and zero-inflation (``f0``)
#: parameters that mean it has none of them.
_PLAIN = {"gamma": 0.0, "p": 1.0, "f0": 0.0}


def never_fails(model) -> float:
    """The fraction of units that never fail: ``1 - p`` for a surpyval
    limited-failure-population model; for a standby arrangement whose
    survival function is a convolution (a sum of lifetimes, some of which
    may never end), the probability that it never fails; 0 for any
    other."""
    if distribution_name(model) is None:
        survival = getattr(model, "_sf_model", None)
        return float(getattr(survival, "never_fails", 0.0) or 0.0)
    p = getattr(model, "p", None)
    if p is None:
        return 0.0
    return max(0.0, 1.0 - float(p))


def is_exponential(model) -> bool:
    """True if ``model`` is a plain surpyval Exponential lifetime: no
    offset, every unit fails and none is dead on arrival. Only then is it
    memoryless from time 0 with rate ``1 / mean``."""
    if distribution_name(model) != "Exponential":
        return False
    extras = getattr(model, "extras", {})
    return all(extras.get(name, none) == none for name, none in _PLAIN.items())


def model_mean(model) -> float:
    """The model's mean lifetime as a plain float: infinite for a
    limited-failure-population model, some of whose units never fail."""
    return float(np.ravel(model.mean())[0])


def failure_time_scale(model) -> float:
    """A typical failure time of ``model``, to size grids and searches by.

    Its mean lifetime; or, when some of its units never fail (so that mean
    is infinite), the mean lifetime of the units that do fail: the same
    distribution with its offset but without ``p`` and ``f0``. NaN when
    neither is finite.
    """
    mean = model_mean(model)
    if np.isfinite(mean):
        return mean
    if distribution_name(model) is not None and never_fails(model) > 0.0:
        offset = {k: v for k, v in model.extras.items() if k == "gamma"}
        failing = model.dist.from_params(
            list(np.ravel(model.params)), **offset
        )
        mean = model_mean(failing)
        if np.isfinite(mean):
            return mean
    return float("nan")


def parametric_spec(model):
    """Return ``(surpyval_class, params, param_names, extras)`` for a
    parametric node model, or ``None`` when it has no reconstructable
    distribution parameters (a ``StandbyModel``, ``RepeatedNode``, nested
    RBD, a repeated node's source name, the perfect-reliability helpers, or
    a fitted non-parametric model). ``extras`` are its offset,
    limited-failure-population and zero-inflation parameters (surpyval's
    ``model.extras``): ``surpyval_class.from_params(params, **extras)``
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
    return cls, params, list(names), dict(getattr(model, "extras", {}))
