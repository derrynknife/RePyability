"""Helpers describing the *capabilities* of the reliability models the RBD
classes consume.

Historically the RBD code branched on ``model.dist.name`` string literals
(e.g. ``"FixedEventProbability"``, ``"Exponential"``) scattered across several
modules. That couples behaviour tightly to surpyval's internal naming: a
rename there silently changes results here. Centralising the checks in these
small helpers keeps that coupling in one place (easy to audit and to cover
with a compatibility test) and gives the call sites intention-revealing names.
"""

from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

#: What a model raises when it cannot work out what it is asked (a quantile,
#: a survival, a mean) at the values given: surpyval's models, and anything
#: else given as one, fail in these ways. Code that probes what a model can
#: do catches them, and lets anything else (a bug) through.
MODEL_ERRORS = (
    ArithmeticError,
    AttributeError,
    LookupError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
)

#: What saving a model that cannot be saved raises (``serialise_model``,
#: a model's ``to_dict`` and ``json.dumps``).
SAVE_ERRORS = (AttributeError, NotImplementedError, TypeError, ValueError)


# surpyval distribution names whose event probability does not vary with time,
# i.e. ``sf(t)`` is constant. These behave as fixed per-demand probabilities
# rather than lifetime distributions.
FIXED_PROBABILITY_DIST_NAMES = frozenset(
    {"FixedEventProbability", "Bernoulli"}
)


def is_mixture(model) -> bool:
    """Whether ``model`` is a surpyval ``MixtureModel``: its ``dist`` is
    only its components' distribution, and its parameters one row per
    component, so it is no single distribution of that name."""
    return type(model).__name__ == "MixtureModel"


def distribution_name(model) -> Optional[str]:
    """Return the surpyval distribution name of ``model``.

    Returns ``None`` for models that do not expose a surpyval distribution
    (e.g. a ``StandbyModel``, ``RepeatedNode``, nested RBD, or the perfect
    reliability/unreliability helpers), and for a ``MixtureModel``, whose
    ``dist`` is only its components': a mixture of exponentials is not
    memoryless, nor is it rebuilt by ``from_params``.
    """
    dist = getattr(model, "dist", None)
    if dist is None or is_mixture(model):
        return None
    return getattr(dist, "name", None)


def is_fixed_probability(model) -> bool:
    """True if ``model`` is a time-invariant (fixed) event probability."""
    return distribution_name(model) in FIXED_PROBABILITY_DIST_NAMES


#: The values of a surpyval parametric model's offset (``gamma``),
#: limited-failure-population (``lfp_p``) and zero-inflation (``f0``)
#: parameters that mean it has none of them.
_PLAIN = {"gamma": 0.0, "lfp_p": 1.0, "f0": 0.0}


def lfp_p(model) -> Optional[float]:
    """The share of a surpyval limited-failure-population model's units
    that ever fail (1 for any other surpyval model); None for a model
    without one (not surpyval's)."""
    return getattr(model, "lfp_p", None)


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


#: The shares a model may have after its distribution's parameters, in the
#: order of surpyval's ``covariance()`` (a limited failure population's
#: ``lfp_p``, then a zero-inflated one's ``f0``), each with the flag that
#: says a fit estimated it and the value that means it has none.
_SHARES = (("lfp", "lfp_p", 1.0), ("zi", "f0", 0.0))


class ParametricSpec(NamedTuple):
    """A parametric model's parameters (see ``parametric_spec``)."""

    cls: Any
    params: List[float]
    names: List[str]
    extras: Dict[str, Any]
    bounds: List[tuple]

    def build(self, values) -> Any:
        """The model with its parameters at ``values``, in the order of
        ``params``."""
        shares = {share for _, share, _ in _SHARES}
        own = [float(v) for n, v in zip(self.names, values) if n not in shares]
        given = {
            n: float(v) for n, v in zip(self.names, values) if n in shares
        }
        return self.cls.from_params(own, **{**self.extras, **given})


def parametric_spec(model) -> Optional[ParametricSpec]:
    """The parameters of a parametric node model, or ``None`` when it has no
    reconstructable distribution parameters (a ``StandbyModel``,
    ``RepeatedNode``, nested RBD, a repeated node's source name, the
    perfect-reliability helpers, or a fitted non-parametric model).

    ``params`` and ``names`` are its distribution's, then the share that
    ever fails (``lfp_p``, a limited failure population) and the share dead
    on arrival (``f0``, zero inflation) where it has them: the order of
    surpyval's ``covariance()``. ``extras`` are surpyval's ``model.extras``
    (the offset kept as it is), ``bounds`` each parameter's range as its
    distribution bounds it (None where it does not), and ``build(values)``
    the model with its parameters at ``values``.

    Used by parameter sensitivity and uncertainty to rebuild a model with
    moved or drawn parameters. It is faithful for exactly the models
    ``serialisation`` round-trips (surpyval parametric distributions), since
    it goes through the same ``dist name`` + ``from_params`` reconstruction.
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
    names = list(names)
    bounds = list(getattr(cls, "bounds", None) or [])
    if len(bounds) != len(params):
        bounds = [(None, None)] * len(params)
    extras = dict(getattr(model, "extras", {}))
    for flag, share, none in _SHARES:
        value = extras.get(share, none)
        if getattr(model, flag, False) or value != none:
            params.append(float(value))
            names.append(share)
            bounds.append((0, 1))
    return ParametricSpec(cls, params, names, extras, bounds)


class CovariateSpec(NamedTuple):
    """A ``RegressionNode``'s covariates as levers (#272), with the
    interface of a ``ParametricSpec``: ``params`` the covariates, ``names``
    ``"covariate.<name>"`` (the model's feature names, or their places),
    ``bounds`` none, and ``build(values)`` the node at those covariates."""

    node: Any
    params: List[float]
    names: List[str]
    bounds: List[tuple]

    def build(self, values) -> Any:
        """The node at the covariates ``values``."""
        from repyability.rbd.regression_node import RegressionNode

        covariates = [float(v) for v in values]
        return RegressionNode(self.node.model, covariates=covariates)


def covariate_spec(model) -> Optional[CovariateSpec]:
    """The covariates of a ``RegressionNode`` at fixed covariates, as
    levers, or None for any other model (one along a schedule has no one
    value to move)."""
    from repyability.rbd.regression_node import RegressionNode

    if not isinstance(model, RegressionNode) or model.covariates is None:
        return None
    params = [float(z) for z in model.covariates]
    features = getattr(model.model, "feature_names", None)
    if not features or len(list(features)) != len(params):
        features = [str(i) for i in range(len(params))]
    names = [f"covariate.{name}" for name in features]
    return CovariateSpec(model, params, names, [(None, None)] * len(params))


def lever_spec(model):
    """What parameter sensitivity moves in a node's model: its
    parameters (``parametric_spec``), or a regression node's covariates
    (``covariate_spec``); None for a model with neither."""
    return parametric_spec(model) or covariate_spec(model)


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
