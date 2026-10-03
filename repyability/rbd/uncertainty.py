"""Epistemic (parameter) uncertainty: draws of the node models.

A node's model is estimated from data, so its parameters are uncertain.
This is *epistemic* uncertainty (about what the model is), as opposed to
the *aleatory* variability the model itself describes (when a given unit
fails). It is propagated by drawing plausible models for the uncertain
nodes, evaluating the system exactly for each draw, and summarising the
spread of the results.

RePyability does not fit models: the draws come from what the fit, done
in surpyval, already provides. A node's uncertainty is one of

- ``"fit"``: the parameters are drawn from the normal approximation of the
  fitted model's maximum-likelihood estimate (surpyval's ``hess_inv``, the
  inverse Hessian of the negative log-likelihood), on the log scale for a
  parameter that must be positive and the logit scale for one in (0, 1),
  so that every draw is valid;
- a mapping ``{parameter name: distribution}``: each named parameter is
  drawn independently from its distribution (anything with ``qf`` or
  ``ppf``, such as a surpyval or scipy.stats distribution), the others
  kept as they are;
- a sequence of models (e.g. refits to bootstrap resamples, or posterior
  draws, made in surpyval): drawn with replacement.

A common-cause group's model may be uncertain too (``draw_ccf_models``):
distributions over its parameters, or a sequence of models.
"""

from collections.abc import Mapping, Sequence
from typing import Any, List, Optional

import numpy as np

#: The draws of a node's model that ``"fit"`` asks for.
FIT = "fit"


def _parametric(model, label: str):
    """The model's distribution, or a ValueError for a model whose
    parameters cannot be redrawn."""
    dist = getattr(model, "dist", None)
    if dist is None or not hasattr(dist, "from_params"):
        raise ValueError(
            f"{label}: its model has no parameters to draw (it is not a "
            "surpyval parametric distribution). Give a list of alternative "
            "models instead."
        )
    return dist


class _Scale:
    """A parameter's map to an unbounded scale: log for a parameter bounded
    on one side, logit for one bounded on both, the identity for an
    unbounded one."""

    def __init__(self, lower: Optional[float], upper: Optional[float]):
        self.lower, self.upper = lower, upper

    def forward(self, v):
        a, b = self.lower, self.upper
        if a is None and b is None:
            return v
        if b is None:
            return np.log(v - a)
        if a is None:
            return np.log(b - v)
        return np.log((v - a) / (b - v))

    def slope(self, v) -> float:
        a, b = self.lower, self.upper
        if a is None and b is None:
            return 1.0
        if b is None:
            return 1.0 / (v - a)
        if a is None:
            return -1.0 / (b - v)
        return (b - a) / ((v - a) * (b - v))

    def inverse(self, z):
        a, b = self.lower, self.upper
        if a is None and b is None:
            return z
        if b is None:
            return a + np.exp(z)
        if a is None:
            return b - np.exp(z)
        return a + (b - a) / (1.0 + np.exp(-z))


def _fit_draws(model, n: int, rng: np.random.Generator, label: str) -> list:
    dist = _parametric(model, label)
    covariance = getattr(model, "hess_inv", None)
    params = np.atleast_1d(np.asarray(model.params, dtype=float))
    if covariance is None:
        raise ValueError(
            f"{label}: 'fit' draws from the fitted model's parameter "
            "covariance (surpyval's hess_inv, from a maximum-likelihood "
            "fit), which this model does not have. Fit it with surpyval, or "
            "give distributions over its parameters or a list of models."
        )
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    k = len(params)
    if covariance.shape != (k, k) or not np.all(np.isfinite(covariance)):
        raise ValueError(
            f"{label}: the fitted model's parameter covariance is not a "
            f"finite {k} x {k} matrix; give distributions over its "
            "parameters or a list of models instead."
        )
    bounds = list(getattr(dist, "bounds", [(None, None)] * k))[:k]
    scales = [_Scale(lower, upper) for lower, upper in bounds]
    for (lower, upper), value in zip(bounds, params):
        if (lower is not None and value <= lower) or (
            upper is not None and value >= upper
        ):
            raise ValueError(
                f"{label}: a fitted parameter, {value!r}, is on the edge of "
                f"its range ({lower}, {upper}), so it has no normal "
                "approximation there; give distributions over its "
                "parameters or a list of models instead."
            )
    # The delta method: the covariance on the unbounded scale.
    slopes = np.array([sc.slope(v) for sc, v in zip(scales, params)])
    centre = np.array([sc.forward(v) for sc, v in zip(scales, params)])
    scaled = covariance * np.outer(slopes, slopes)
    scaled = 0.5 * (scaled + scaled.T)
    values, vectors = np.linalg.eigh(scaled)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    z = centre + rng.standard_normal((n, k)) @ root.T
    drawn = np.column_stack(
        [sc.inverse(z[:, j]) for j, sc in enumerate(scales)]
    )
    return [model.with_params(list(row)) for row in drawn]


def is_fit(model) -> bool:
    """Whether ``"fit"`` can draw ``model``: a surpyval parametric fit with
    a finite parameter covariance, its parameters inside their ranges."""
    try:
        _fit_draws(model, 0, np.random.default_rng(0), "")
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def _quantiles(
    prior, n: int, rng: np.random.Generator, label: str, name
) -> np.ndarray:
    """``n`` draws from ``prior``, through its quantile function."""
    u = rng.random(n)
    if hasattr(prior, "qf"):
        values = prior.qf(u)
    elif hasattr(prior, "ppf"):
        values = prior.ppf(u)
    else:
        raise ValueError(
            f"{label}: the distribution for {name!r} needs a quantile "
            "function (qf, as surpyval's, or ppf, as scipy.stats')."
        )
    return np.asarray(values, dtype=float).reshape(-1)


def _parameter_draws(
    model, priors: Mapping, n: int, rng: np.random.Generator, label: str
) -> list:
    dist = _parametric(model, label)
    names = list(getattr(dist, "param_names", []))
    unknown = [p for p in priors if p not in names]
    if unknown:
        raise ValueError(
            f"{label}: {sorted(map(str, unknown))} are not parameters of its "
            f"{getattr(dist, 'name', 'model')} model, whose parameters are "
            f"{names}."
        )
    params = np.atleast_1d(np.asarray(model.params, dtype=float))
    bounds = list(getattr(dist, "bounds", [(None, None)] * len(names)))
    columns = np.tile(params, (n, 1))
    for name, prior in priors.items():
        values = _quantiles(prior, n, rng, label, name)
        j = names.index(name)
        lower, upper = bounds[j] if j < len(bounds) else (None, None)
        if not np.all(np.isfinite(values)) or (
            (lower is not None and np.any(values <= lower))
            or (upper is not None and np.any(values >= upper))
        ):
            raise ValueError(
                f"{label}: the distribution for {name!r} gives values "
                f"outside its range ({lower}, {upper})."
            )
        columns[:, j] = values
    return [model.with_params(list(row)) for row in columns]


def _ensemble_draws(
    models: Sequence, n: int, rng: np.random.Generator, label: str
) -> list:
    if not models:
        raise ValueError(f"{label}: the list of models is empty.")
    for m in models:
        if not hasattr(m, "sf"):
            raise ValueError(
                f"{label}: every alternative model needs sf, got {m!r}."
            )
    return [models[i] for i in rng.integers(len(models), size=n)]


def draw_models(
    model: Any, spec: Any, n: int, rng: np.random.Generator, label: str
) -> List[Any]:
    """``n`` plausible models of a node whose model is ``model``, as
    ``spec`` says (see the module docstring)."""
    if isinstance(spec, str):
        if spec != FIT:
            raise ValueError(
                f"{label}: unknown uncertainty {spec!r}; give 'fit', a dict "
                "of parameter distributions or a list of models."
            )
        return _fit_draws(model, n, rng, label)
    if isinstance(spec, Mapping):
        if not spec:
            raise ValueError(f"{label}: no parameter distributions given.")
        return _parameter_draws(model, spec, n, rng, label)
    if isinstance(spec, Sequence):
        return _ensemble_draws(spec, n, rng, label)
    raise ValueError(
        f"{label}: the uncertainty must be 'fit', a dict of parameter "
        f"distributions or a list of models, got {spec!r}."
    )


def draw_ccf_models(
    group, spec: Any, n: int, rng: np.random.Generator
) -> List[Any]:
    """``n`` plausible common-cause models of a ``CCFGroup``, as ``spec``
    says: ``{parameter name: distribution}`` over its model's parameters
    (``beta``, and an MGL model's ``gamma``, ``delta``, ...; see
    ``ccf.parameters``), each drawn independently and in ``[0, 1]``, the
    others kept; or a list of models (``BetaFactor`` or ``MGL``, for the
    group's size and with its model's basis), drawn with replacement."""
    from .ccf import MGL, BetaFactor, parameters, with_parameters

    model = group.model
    label = f"Common-cause group {list(group.members)!r}"
    names = list(parameters(model))
    if isinstance(spec, Mapping):
        if not spec:
            raise ValueError(f"{label}: no parameter distributions given.")
        unknown = [p for p in spec if p not in names]
        if unknown:
            raise ValueError(
                f"{label}: {sorted(map(str, unknown))} are not parameters "
                f"of its model, {model!r}, whose parameters are {names}."
            )
        columns = {}
        for name, prior in spec.items():
            values = _quantiles(prior, n, rng, label, name)
            if not np.all((values >= 0.0) & (values <= 1.0)):
                raise ValueError(
                    f"{label}: the distribution for {name!r} gives values "
                    "outside [0, 1]."
                )
            columns[name] = values
        return [
            with_parameters(model, {k: v[i] for k, v in columns.items()})
            for i in range(n)
        ]
    if isinstance(spec, Sequence) and not isinstance(spec, str):
        if not spec:
            raise ValueError(f"{label}: the list of models is empty.")
        for m in spec:
            size = (
                m.required_group_size()
                if isinstance(m, (BetaFactor, MGL))
                else None
            )
            if (
                not isinstance(m, (BetaFactor, MGL))
                or size not in (None, len(group.members))
                or m.basis != model.basis
                or m.shocks != model.shocks
            ):
                raise ValueError(
                    f"{label}: every alternative model must be a "
                    f"BetaFactor or an MGL model for {len(group.members)} "
                    f"members, splitting the {model.basis} as its model "
                    f"does, with {model.shocks} shocks; got {m!r}."
                )
        return [spec[i] for i in rng.integers(len(spec), size=n)]
    raise ValueError(
        f"{label}: its uncertainty must be a dict of distributions over its "
        f"model's parameters ({names}) or a list of models, got {spec!r}."
    )
