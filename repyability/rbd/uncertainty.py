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
"""

from collections.abc import Mapping, Sequence
from typing import Any, List, Optional

import numpy as np

#: The draws of a node's model that ``"fit"`` asks for.
FIT = "fit"


#: The offset, limited-failure-population and zero-inflation parameters,
#: and the values that mean the model has none of them.
_EXTRAS = {"gamma": 0.0, "p": 1.0, "f0": 0.0}


def _extras(model) -> dict:
    """The model's offset, limited-failure-population and zero-inflation
    parameters, where it has them: they keep their values in every draw."""
    out = {}
    for name, none in _EXTRAS.items():
        value = getattr(model, name, none)
        if value is not None and value != none:
            out[name] = value
    return out


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
    extras = _extras(model)
    return [dist.from_params(list(row), **extras) for row in drawn]


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
        values = np.asarray(values, dtype=float).reshape(-1)
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
    extras = _extras(model)
    return [dist.from_params(list(row), **extras) for row in columns]


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
