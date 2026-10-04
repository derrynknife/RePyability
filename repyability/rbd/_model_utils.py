"""Helpers describing the *capabilities* of the reliability models the RBD
classes consume.

Historically the RBD code branched on ``model.dist.name`` string literals
(e.g. ``"FixedEventProbability"``, ``"Exponential"``) scattered across several
modules. That couples behaviour tightly to surpyval's internal naming: a
rename there silently changes results here. Centralising the checks in these
small helpers keeps that coupling in one place (easy to audit and to cover
with a compatibility test) and gives the call sites intention-revealing names.
"""

import functools
from typing import Any, Dict, List, Optional

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
#: limited-failure-population (``lfp_p``, which ``extras`` calls ``p``
#: before surpyval 0.23) and zero-inflation (``f0``) parameters that mean
#: it has none of them.
_PLAIN = {"gamma": 0.0, "lfp_p": 1.0, "p": 1.0, "f0": 0.0}


def lfp_p(model) -> Optional[float]:
    """The share of a surpyval limited-failure-population model's units
    that ever fail (1 for any other surpyval model): ``lfp_p``, its name
    from surpyval 0.23 (SurPyval#608), or ``p`` before. From 0.23 ``p``
    is deprecated, and names the parameter of a distribution that has one
    (Bernoulli's). None for a model with neither."""
    value = getattr(model, "lfp_p", None)
    if value is None:
        value = getattr(model, "p", None)
    return value


@functools.lru_cache(maxsize=None)
def _lfp_keyword() -> str:
    import inspect

    from surpyval import Weibull

    names = inspect.signature(Weibull.from_params).parameters
    return "lfp_p" if "lfp_p" in names else "p"


def lfp_extras(p) -> Dict[str, Any]:
    """The keyword argument of surpyval's ``from_params`` for a
    limited-failure proportion ``p``, by the name the installed surpyval
    takes: ``{"lfp_p": p}`` from surpyval 0.23 (SurPyval#608), ``{"p":
    p}`` before."""
    return {_lfp_keyword(): p}


def never_fails(model) -> float:
    """The fraction of units that never fail: ``1 - lfp_p`` for a surpyval
    limited-failure-population model; for a standby arrangement whose
    survival function is a convolution (a sum of lifetimes, some of which
    may never end), the probability that it never fails; 0 for any
    other."""
    if distribution_name(model) is None:
        survival = getattr(model, "_sf_model", None)
        return float(getattr(survival, "never_fails", 0.0) or 0.0)
    p = lfp_p(model)
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
    """Return ``(surpyval_class, params, parameter_names, extras)`` for a
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
    names = getattr(dist, "parameter_names", None)
    if not names or len(list(names)) != len(params):
        names = [f"param{i}" for i in range(len(params))]
    return cls, params, list(names), dict(getattr(model, "extras", {}))


def nonparametric(model) -> bool:
    """Whether a node's model is, or is built from, a surpyval
    non-parametric fit (a nested RBD refuses one when it is built)."""
    from surpyval import NonParametric

    from repyability.non_repairable import NonRepairable
    from repyability.rbd.degrading_node import DegradingNode
    from repyability.rbd.repeated_node import RepeatedNode
    from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
    from repyability.rbd.standby_node import StandbyModel

    if isinstance(model, NonParametric):
        return True
    if isinstance(model, StandbyModel):
        return any(nonparametric(unit) for unit in model.reliabilities)
    if isinstance(model, (RepeatedNode, RepeatedStandbyNode)):
        return nonparametric(model.model)
    if isinstance(model, DegradingNode):
        return any(nonparametric(stage) for _, stage in model.stages)
    if isinstance(model, NonRepairable):
        return nonparametric(model.reliability) or nonparametric(
            model.time_to_replace
        )
    if isinstance(model, dict):  # a RepairableRBD component spec
        return any(nonparametric(value) for value in model.values())
    if isinstance(model, (list, tuple)):
        return any(nonparametric(value) for value in model)
    return False


def nonparametric_nodes(models: Dict[Any, Any]) -> List[Any]:
    """The nodes whose models (or component specs) hold a surpyval
    non-parametric fit, in order."""
    return [node for node, model in models.items() if nonparametric(model)]


def refuse_nonparametric(models) -> None:
    """Refuse the nodes whose models (or component specs) hold a surpyval
    non-parametric fit (#149)."""
    nodes = nonparametric_nodes(models)
    if nodes:
        raise ValueError(
            f"Node(s) {nodes} use a non-parametric fit (Kaplan-Meier or "
            "similar) as a lifetime or repair time, which a diagram does "
            "not take: its curve ends at the data, so the MTTF, B-lives and "
            "long-run values beyond it would be artefacts, and its draws "
            "cannot be paired or streamed in simulations. Fit a parametric "
            "distribution in surpyval (e.g. surpyval.Weibull.fit(times)) and "
            "use that."
        )
