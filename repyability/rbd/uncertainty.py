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
  fitted model's maximum-likelihood estimate (surpyval's ``covariance()``,
  the inverse of the observed information). For a surpyval distribution
  they are drawn on the log scale for a parameter that must be positive
  and the logit scale for one in (0, 1), so that every draw is valid.
  They are every parameter the fit estimated: the distribution's, and a
  limited failure population's share that ever fails (``lfp_p``) and a
  zero-inflated one's share dead on arrival (``f0``) (#267). An offset
  fit's draws are surpyval's refits to resampled data (its
  ``bootstrap()``, 200 refits, picked with replacement), which carry the
  offset's uncertainty too, since its covariance leaves the offset out;
  where it cannot resample (a model without its data, or left- or
  interval-censored data), the offset is held where it was fitted, with
  a warning.
  The other fitted models a diagram takes are drawn from their
  covariance and rebuilt by surpyval's ``with_params`` (in the order of
  ``covariance_names``): a regression (a ``RegressionNode``'s model, its
  covariates kept), a Wiener or gamma degradation process, a mixture, a
  Royston-Parmar model, and a fitted repairable-unit process
  (Crow-AMSAA, Duane, HPP or generalised renewal). Each parameter is
  drawn on its own scale, as a distribution's is: surpyval gives no range
  per parameter for these models, so the range is read from what
  ``with_params`` refuses, the others at their estimates. A parameter it
  refuses just below 0 (a scale, a shape, a rate) is drawn on the log
  scale; one it also refuses just above 1, and a renewal process's
  restoration factor (which a diagram takes in [0, 1]), on the logit
  scale; any other (a regression coefficient, a location) on its
  own. A mixture's weights are drawn as their log-ratios to the last
  (surpyval's ``w_logits``). The covariance is carried to those scales
  by the delta method, so every draw is valid and none is cut off. A
  draw surpyval still refuses (where parameters are tied together, as a
  spline's are) is drawn again and counted, with a warning, and past
  ``MOST_REFUSED`` (2%) of the draws ``"fit"`` refuses.
  Nodes whose ``RegressionNode`` share one regression fit share its
  draws, each at its own covariates.
  For a Bayesian fit (a ``PosteriorPredictive``) the draws are its
  posterior itself: each a posterior point picked by its probability
  (unequal on a grid; no point left out), or from the exact family of a
  conjugate posterior, and the node in that draw the plain distribution
  at it, not the predictive. The band so drawn is a credible band;
  drawn from a covariance, a confidence band (see ``band``);
- a mapping ``{parameter name: distribution}``: each named parameter is
  drawn independently from its distribution (anything with ``qf`` or
  ``ppf``, such as a surpyval or scipy.stats distribution), the others
  kept as they are;
- a sequence of models (e.g. refits to bootstrap resamples, or posterior
  draws, made in surpyval): drawn with replacement.

A common-cause group's model may be uncertain too (``draw_ccf_models``):
distributions over its parameters, or a sequence of models.

The draws take their randomness from a numpy generator, or (``sampling=
"sobol"``, #200) from a scrambled Sobol sequence: each draw is one of its
points, each parameter (or choice from a list) one of its dimensions, in
the order the draws take them. Quasi-random points cover the parameters
more evenly than random ones, which shrinks the error of the summaries
over the draws, and of the Sobol indices, for the same number of draws.
"""

import difflib
import warnings
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd._bayesian_life import PosteriorPredictive
from repyability.rbd._model_utils import (
    MODEL_ERRORS,
    is_mixture,
    parametric_spec,
)
from repyability.utils.checks import seed as check_seed
from repyability.utils.wrappers import outside_level

#: The draws of a node's model that ``"fit"`` asks for.
FIT = "fit"
#: Where the draws' randomness comes from (see the module docstring).
SAMPLINGS = ("random", "sobol")


def check_sampling(sampling) -> str:
    """``sampling``, checked: one of ``SAMPLINGS``."""
    if sampling not in SAMPLINGS:
        raise ValueError(
            f"sampling must be one of {list(SAMPLINGS)}, got {sampling!r}."
        )
    return sampling


class Counter:
    """Stands in for a generator, to count the dimensions a set of draws
    takes (one for each column of normals or uniforms, and each choice
    from a list), with the shapes a numpy generator gives."""

    def __init__(self):
        self.dimensions = 0

    def standard_normal(self, shape) -> np.ndarray:
        n, k = shape
        self.dimensions += k
        return np.zeros((n, k))

    def random(self, n) -> np.ndarray:
        self.dimensions += 1
        return np.full(n, 0.5)

    def integers(self, high, size) -> np.ndarray:
        self.dimensions += 1
        return np.zeros(size, dtype=int)


class SobolPoints:
    """A scrambled Sobol sequence's points (rows of ``table``), handed out
    a dimension at a time as the draws take them, in the shapes a numpy
    generator gives: normals through the normal quantile function, and a
    choice from a list as the uniform's share of it."""

    def __init__(self, table: np.ndarray):
        # Kept off 0 and 1, where a quantile function is infinite.
        self.table = np.clip(table, 2.0**-53, 1.0 - 2.0**-53)
        self.used = 0

    def _take(self, k: int) -> np.ndarray:
        if self.used + k > self.table.shape[1]:
            raise RuntimeError("The draws took more dimensions than counted.")
        out = self.table[:, self.used : self.used + k]  # noqa: E203
        self.used += k
        return out

    def standard_normal(self, shape) -> np.ndarray:
        from scipy.special import ndtri

        n, k = shape
        return ndtri(self._take(k)[:n])

    def random(self, n) -> np.ndarray:
        return self._take(1)[:n, 0].copy()

    def integers(self, high, size) -> np.ndarray:
        u = self._take(1)[:size, 0]
        return np.minimum((u * high).astype(int), int(high) - 1)


def check_draws(n_draws) -> int:
    """``n_draws``, checked: a positive integer."""
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise ValueError(f"n_draws must be an integer, got {n_draws!r}.")
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}.")
    return int(n_draws)


def draw_with(
    draw: Callable[[int, Any], Any],
    n_draws: int,
    seed,
    sampling: str = "random",
    paired: bool = False,
    join: Optional[Callable[[Any, Any], Any]] = None,
) -> Any:
    """``draw(n, source)``'s draws, ``n_draws`` of them, with their
    randomness from ``seed``: ``sampling="random"``, numpy's generator;
    ``"sobol"``, the points of a scrambled Sobol sequence, the dimensions
    the draws take counted first (see the module docstring). ``paired``:
    two independent sets of ``n_draws``, one after the other, for the
    Sobol indices: one draw of twice as many, or, from the Sobol points,
    two draws from dimensions of their own, put together by ``join``."""
    check_sampling(sampling)
    rng = np.random.default_rng(check_seed(seed))
    if sampling == "random":
        return draw(2 * n_draws if paired else n_draws, rng)
    counter = Counter()
    draw(1, counter)
    width = counter.dimensions
    table = sobol_table(n_draws, 2 * width if paired else width, rng)
    first = draw(n_draws, SobolPoints(table[:, :width]))
    if not paired:
        return first
    assert join is not None
    return join(first, draw(n_draws, SobolPoints(table[:, width:])))


def sobol_table(n: int, dimensions: int, rng) -> np.ndarray:
    """``n`` points of a scrambled Sobol sequence in ``dimensions``
    dimensions, scrambled by ``rng`` (numpy's generator). Its balance is
    best for a power of 2 points; others are taken as they come."""
    if dimensions == 0:
        return np.empty((n, 0))
    from scipy.stats import qmc

    engine = qmc.Sobol(dimensions, scramble=True, seed=rng)
    with warnings.catch_warnings():
        # The warning that n is not a power of 2.
        warnings.simplefilter("ignore", UserWarning)
        return engine.random(n)


def _spec(model, label: str):
    """The model's parameters (``parametric_spec``), or a ValueError for a
    model whose parameters cannot be redrawn."""
    if is_mixture(model):
        # Its dist is its components' alone (#227).
        raise ValueError(
            f"{label}: its model is a surpyval MixtureModel, whose "
            "parameters are its components' and their weights, which are "
            "not drawn one by one: give 'fit', which draws them together "
            "from the fit's covariance, or a list of alternative models "
            "(fitted to resampled data, say)."
        )
    if isinstance(model, PosteriorPredictive):
        raise ValueError(
            f"{label}: its model is a Bayesian fit, whose parameters are "
            "its posterior, not values to redraw one by one: give 'fit', "
            "which draws from the posterior itself, or a list of models."
        )
    spec = parametric_spec(model)
    if spec is None:
        raise ValueError(
            f"{label}: its model has no parameters to draw (it is not a "
            "surpyval parametric distribution). Give a list of alternative "
            "models instead."
        )
    return spec


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


def _fit_covariance(model, label: str, warn: bool = False):
    """The fitted model's parameters (``parametric_spec``), their
    covariance (surpyval's ``covariance()``) and their bounds, checked;
    with ``warn``, a warning that an offset is held where it was fitted."""
    spec = _spec(model, label)
    try:
        covariance = model.covariance()
    except (AttributeError, TypeError, ValueError):
        covariance = None
    if covariance is None:
        raise ValueError(
            f"{label}: 'fit' draws from the fitted model's parameter "
            "covariance (surpyval's covariance(), from a maximum-likelihood "
            "fit), which this model does not have. Fit it with surpyval, or "
            "give distributions over its parameters or a list of models."
        )
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    params = np.asarray(spec.params, dtype=float)
    k = len(params)
    if covariance.shape != (k, k) or not np.all(np.isfinite(covariance)):
        raise ValueError(
            f"{label}: the fitted model's parameter covariance is not a "
            f"finite {k} x {k} matrix; give distributions over its "
            "parameters or a list of models instead."
        )
    for (lower, upper), value in zip(spec.bounds, params):
        if (lower is not None and value <= lower) or (
            upper is not None and value >= upper
        ):
            raise ValueError(
                f"{label}: a fitted parameter, {value!r}, is on the edge of "
                f"its range ({lower}, {upper}), so it has no normal "
                "approximation there; give distributions over its "
                "parameters or a list of models instead."
            )
    if warn and getattr(model, "offset", False):
        warnings.warn(
            f"{label}: its model was fitted with an offset, gamma = "
            f"{spec.extras.get('gamma')!r}, which surpyval's parameter "
            "covariance leaves out and which surpyval cannot draw here by "
            "refitting resampled data (a model without its data, or left- "
            "or interval-censored data), so the offset is held where it "
            "was fitted and the spread drawn is too narrow. For the "
            "offset's uncertainty give a list of models (refits to "
            "resampled data, say).",
            stacklevel=outside_level(),
        )
    return spec, params, covariance, list(spec.bounds)


#: The refits an offset fit's draws are picked from (surpyval's
#: ``bootstrap()``, its default), and their seed: fixed, so that the
#: refits are the same at every call (surpyval keeps them on the model).
BOOTSTRAP_REFITS, BOOTSTRAP_SEED = 200, 0


def _is_distribution(model) -> bool:
    """Whether ``model`` is a surpyval distribution's fit (or one built
    from parameters), whose draws are made on each parameter's scale (see
    the module docstring); not a mixture, a Bayesian fit or another
    model."""
    return (
        not is_mixture(model)
        and not isinstance(model, PosteriorPredictive)
        and parametric_spec(model) is not None
    )


def _fit_draws(model, n: int, rng: np.random.Generator, label: str) -> list:
    if not _is_distribution(model):
        return _rebuilt_draws(model, n, rng, label)
    if n > 0 and getattr(model, "offset", False):
        refits = _bootstrap(model)
        if refits:
            return [refits[i] for i in rng.integers(len(refits), size=n)]
    return _covariance_draws(model, n, rng, label)


def _bootstrap(model) -> list:
    """An offset fit's refits to resampled data (surpyval's
    ``bootstrap()``), or none where surpyval cannot resample its data."""
    try:
        refits = model.bootstrap(BOOTSTRAP_REFITS, random_state=BOOTSTRAP_SEED)
    except (ValueError, *MODEL_ERRORS):
        return []
    return list(refits.models)


def _covariance_draws(
    model, n: int, rng: np.random.Generator, label: str
) -> list:
    spec, params, covariance, bounds = _fit_covariance(model, label, n > 0)
    k = len(params)
    scales = [_Scale(lower, upper) for lower, upper in bounds]
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
    return [spec.build(row) for row in drawn]


def fitted_object(model):
    """The fitted object whose parameters a node model's draws vary: a
    ``RegressionNode``'s regression model, or the model itself."""
    from .regression_node import RegressionNode

    return model.model if isinstance(model, RegressionNode) else model


def shares_fit(a, b) -> bool:
    """Whether two node models are one fit: one object, or two
    ``RegressionNode`` of one regression model (at their own covariates),
    which share its draws."""
    return a is b or fitted_object(a) is fitted_object(b)


def carried(drawn: list, first, model) -> list:
    """The draws of ``first``'s model (``drawn``) for a node whose model,
    ``model``, shares its fit (``shares_fit``): the same draws, or, for a
    ``RegressionNode`` at other covariates, each drawn regression at its
    own covariates."""
    from .regression_node import RegressionNode

    if model is first or not isinstance(model, RegressionNode):
        return drawn
    return [
        RegressionNode(
            m.model, covariates=model.covariates, schedule=model.schedule
        )
        for m in drawn
    ]


def _kind(fit) -> str:
    """A surpyval model's kind, named for a message."""
    return type(fit).__name__


#: The largest share of a fit's draws surpyval may refuse (``with_params``)
#: before ``"fit"`` refuses: drawn on each parameter's scale, a draw
#: should never be refused, so more says the scales are wrong for it.
MOST_REFUSED = 0.02


class _Rebuilt:
    """A fitted model drawn through surpyval's ``with_params`` (see the
    module docstring): its parameters' ``names`` (``covariance_names``),
    their fitted values (``params``), their ``covariance``, each one's
    range as surpyval refuses values past it (``bounds``; a mixture's
    weights (0, 1)), and the node model at other values (``rebuild``)
    or at a point of the scales the draws are made on (``unscaled``)."""

    def __init__(self, model, label: str):
        from .joint_node import is_joint

        if is_joint(model):
            raise ValueError(
                f"{label}: its model is a surpyval joint model's "
                "prediction, which surpyval cannot rebuild at other "
                "parameters (it has no with_params), so 'fit' cannot draw "
                "it. Give a list of alternative models instead "
                "(predictions of refits to resampled data, say)."
            )
        if isinstance(model, PosteriorPredictive):
            raise ValueError(
                f"{label}: its model is a Bayesian fit, drawn from its "
                "posterior, not rebuilt from its parameters."
            )
        self.model, self.label = model, label
        fit = self.fit = fitted_object(model)
        names = getattr(fit, "covariance_names", None)
        if callable(names):
            names = names()
        if not callable(getattr(fit, "with_params", None)) or names is None:
            raise ValueError(
                f"{label}: its model has no parameters to draw (it is not "
                "a surpyval parametric distribution or a surpyval model "
                "with a parameter covariance and with_params). Give a list "
                "of alternative models instead."
            )
        self.names = [str(name) for name in names]
        try:
            covariance = fit.covariance()
        except (ValueError, *MODEL_ERRORS):
            covariance = None
        if covariance is None:
            raise ValueError(
                f"{label}: 'fit' draws from the fitted model's parameter "
                "covariance (surpyval's covariance(), from a "
                f"maximum-likelihood fit), which this {_kind(fit)} does not "
                "have (built from parameters, say). Give a list of models "
                "instead (refits to resampled data, say)."
            )
        covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
        params = np.ravel(np.asarray(fit.params, dtype=float))
        # A mixture's last m parameters are its weights.
        self.m = len(np.ravel(fit.w)) if is_mixture(fit) else 0
        if self.m:
            params = np.r_[params, np.ravel(np.asarray(fit.w, dtype=float))]
        k = len(self.names)
        if len(params) != k:
            raise ValueError(
                f"{label}: its {_kind(fit)}'s parameters ({len(params)}) do "
                f"not line up with its covariance's ({self.names}). Give a "
                "list of models instead."
            )
        if covariance.shape != (k, k) or not np.all(np.isfinite(covariance)):
            raise ValueError(
                f"{label}: the fitted model's parameter covariance is not a "
                f"finite {k} x {k} matrix (a parameter on the edge of its "
                "range, say); give a list of models instead (refits to "
                "resampled data, say)."
            )
        self.params, self.covariance = params, covariance
        if _tried(self.rebuild, params) is None:
            raise ValueError(
                f"{label}: surpyval refuses to rebuild its fitted model at "
                "its own parameters; give a list of models instead."
            )
        self.bounds = self._bounds()

    def rebuild(self, values):
        """The node model with the fit's parameters at ``values`` (in the
        order of ``names``; a mixture's last weight is the rest)."""
        values = np.asarray(values, dtype=float)
        if self.m:
            weights = values[-self.m :].copy()  # noqa: E203
            weights[-1] = 1.0 - weights[:-1].sum()
            drawn = self.fit.with_params(values[: -self.m], w=weights)
        else:
            drawn = self.fit.with_params(values)
        return self._node(drawn)

    def _node(self, drawn):
        from .regression_node import RegressionNode

        if isinstance(self.model, RegressionNode):
            return RegressionNode(
                drawn,
                covariates=self.model.covariates,
                schedule=self.model.schedule,
            )
        return drawn

    def _bounds(self) -> list:
        """Each parameter's range, read from what surpyval refuses (it
        gives no range per parameter for these models): bounded below by
        0 where ``with_params`` refuses the parameter just below 0 (the
        others at their estimates), and above by 1 too where it also
        refuses it just above 1 (or it is a renewal process's restoration
        factor, which a diagram takes in [0, 1]); unbounded otherwise (a
        regression coefficient, a location). A parameter the fit holds
        fixed (no variance) is left as it is; a mixture's weights are in
        (0, 1)."""
        k = len(self.params) - self.m
        variances = np.diag(self.covariance)
        out: list = []
        for j in range(k):
            value = self.params[j]
            if variances[j] <= 0.0:
                out.append((None, None))
                continue
            tiny = 1e-9 * max(1.0, abs(value))
            if self._accepts(j, -tiny):
                out.append((None, None))
                continue
            if value <= 0.0:
                raise ValueError(
                    f"{self.label}: its fitted {self.names[j]}, "
                    f"{value!r}, is on the edge of its range (surpyval "
                    "refuses it below 0), so it has no normal "
                    "approximation there; give a list of models instead "
                    "(refits to resampled data, say)."
                )
            restoration = j == 0 and _restores(self.fit)
            upper = (
                1.0
                if restoration
                or (value < 1.0 and not self._accepts(j, 1 + tiny))
                else None
            )
            if upper is not None and value >= upper:
                raise ValueError(
                    f"{self.label}: its fitted {self.names[j]}, "
                    f"{value!r}, is at or past the edge of its range (0, "
                    "1), so it has no normal approximation there; give a "
                    "list of models instead (refits to resampled data, "
                    "say)."
                )
            out.append((0.0, upper))
        for w in self.params[k:]:
            if not 0.0 < w < 1.0:
                raise ValueError(
                    f"{self.label}: a fitted weight, {w!r}, is on the edge "
                    "of its range, so it has no normal approximation there; "
                    "give a list of models instead."
                )
            out.append((0.0, 1.0))
        return out

    def _accepts(self, j: int, value: float) -> bool:
        moved = self.params.copy()
        moved[j] = value
        return _tried(self.rebuild, moved) is not None

    def scaled(self) -> Tuple[np.ndarray, np.ndarray]:
        """The centre and covariance of the draws, on each parameter's
        scale (log for one bounded below by 0, logit for one in (0, 1),
        its own otherwise; a mixture's weights as their m - 1 log-ratios
        to the last, ``w_logits``), the covariance carried there by the
        delta method: ``G Sigma G^T``, ``G`` the scales' slopes."""
        k = len(self.params) - self.m
        scales = [_Scale(*b) for b in self.bounds[:k]]
        centre = [sc.forward(v) for sc, v in zip(scales, self.params)]
        rows = len(self.params) - (1 if self.m else 0)
        slopes = np.zeros((rows, len(self.params)))
        for j, (sc, v) in enumerate(zip(scales, self.params)):
            slopes[j, j] = sc.slope(v)
        if self.m:
            w = self.params[k:]
            centre += list(np.log(w[:-1] / w[-1]))
            for i in range(self.m - 1):
                slopes[k + i, k + i] = 1.0 / w[i]
                slopes[k + i, -1] = -1.0 / w[-1]
        return np.asarray(centre), slopes @ self.covariance @ slopes.T

    def unscaled(self, z):
        """The node model at a point ``z`` of the draws' scales."""
        k = len(self.params) - self.m
        values = np.array(
            [_Scale(*b).inverse(v) for b, v in zip(self.bounds[:k], z[:k])]
        )
        if not self.m:
            return self.rebuild(values)
        return self._node(self.fit.with_params(values, w_logits=z[k:]))


def _restores(fit) -> bool:
    """Whether ``fit`` is a surpyval renewal process, whose first
    parameter is its restoration factor: in [0, 1] as a diagram takes it
    (0 renews, 1 is minimal repair), where surpyval allows a generalised
    renewal process's past 1."""
    from ._processes import _is_renewal

    return _is_renewal(fit)


def _redrawable(model, label: str) -> _Rebuilt:
    """``model`` as ``_Rebuilt``, or a ValueError, saying what to give
    instead, for a model whose parameters cannot be drawn."""
    return _Rebuilt(model, label)


def _tried(rebuild, values):
    """The model ``rebuild`` makes of ``values``, or None where surpyval
    refuses them (a value outside its parameter's range) or they are not
    finite."""
    if not np.all(np.isfinite(values)):
        return None
    try:
        return rebuild(values)
    except (ValueError, *MODEL_ERRORS):
        return None


def _rebuilt_draws(
    model, n: int, rng: np.random.Generator, label: str
) -> list:
    """``n`` draws of a model rebuilt by surpyval's ``with_params`` (see
    ``_Rebuilt``), from the normal approximation on each parameter's scale.
    A draw surpyval still refuses (a joint constraint, say) is drawn again
    (from ``rng`` itself, or, for a Sobol sequence's points, from a
    generator seeded by them) and counted, with a warning; more than
    ``MOST_REFUSED`` of them is refused."""
    rebuilt = _redrawable(model, label)
    if n == 0:
        return []
    centre, covariance = rebuilt.scaled()
    k = len(centre)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    z = centre + rng.standard_normal((n, k)) @ root.T
    out = [_tried(rebuilt.unscaled, row) for row in z]
    missing = [i for i, m in enumerate(out) if m is None]
    if not missing:
        return out
    if isinstance(rng, np.random.Generator):
        spare = rng
    else:
        import hashlib

        digest = hashlib.sha256(np.ascontiguousarray(z).tobytes()).digest()
        spare = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    refused = 0
    allowed = max(1, int(MOST_REFUSED * n))
    while missing:
        refused += len(missing)
        if refused > allowed:
            raise ValueError(
                f"{label}: surpyval refused {refused} of the {n} parameter "
                f"vectors drawn from its {_kind(rebuilt.fit)} fit's "
                f"covariance ({', '.join(rebuilt.names)}), drawn on each "
                "parameter's scale: more than "
                f"{MOST_REFUSED:.0%}, so the normal approximation of this "
                "fit is poor (its parameters are tied by a constraint, "
                "say). Give a list of models instead (refits to resampled "
                "data, say)."
            )
        again = centre + spare.standard_normal((len(missing), k)) @ root.T
        for i, row in zip(missing, again):
            out[i] = _tried(rebuilt.unscaled, row)
        missing = [i for i in missing if out[i] is None]
    warnings.warn(
        f"{label}: surpyval refused {refused} of the {n} parameter vectors "
        "drawn from its fit's covariance, and they were drawn again.",
        UserWarning,
        stacklevel=outside_level(),
    )
    return out


def is_fit(model) -> bool:
    """Whether ``"fit"`` can draw ``model``: a surpyval parametric fit with
    a finite parameter covariance, its parameters inside their ranges, a
    fitted model surpyval rebuilds by ``with_params`` (a regression, a
    degradation or repairable-unit process, a mixture), or a Bayesian fit
    (its posterior)."""
    if isinstance(model, PosteriorPredictive):
        return True
    try:
        _fit_draws(model, 0, np.random.default_rng(0), "")
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def fitted_together(
    models: Mapping, joined: Iterable[Iterable[Hashable]] = ()
) -> Dict[Hashable, str]:
    """The uncertainty drawn when none is given (#184): ``"fit"`` for each
    of ``models`` (by name) that is a surpyval fit with a parameter
    covariance, the names that share one fit (``shares_fit``), or are
    ``joined``
    (a common-cause group's members, say), given together in a tuple, so
    that they share their draws."""
    fitted = [name for name, model in models.items() if is_fit(model)]
    parent = {name: name for name in fitted}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    holder: Dict[int, Hashable] = {}
    for name in fitted:
        # The nodes of one fit share its draws: one model object, or
        # regression nodes of one regression (at their own covariates).
        key = id(fitted_object(models[name]))
        parent[root(name)] = root(holder.setdefault(key, name))
    for names in joined:
        inside = [name for name in names if name in parent]
        for name in inside[1:]:
            parent[root(name)] = root(inside[0])
    together: Dict[Hashable, list] = {}
    for name in fitted:
        together.setdefault(root(name), []).append(name)
    return {
        names[0] if len(names) == 1 else tuple(names): FIT
        for names in together.values()
    }


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


def _closest(unknown: list, names: list) -> str:
    """ " Did you mean ...?" for the first of ``unknown`` close to one of
    ``names`` (#232), or ""."""
    for key in unknown:
        close = difflib.get_close_matches(str(key), [str(n) for n in names], 1)
        if close:
            return f" Did you mean {close[0]!r}?"
    return ""


def _parameter_draws(
    model, priors: Mapping, n: int, rng: np.random.Generator, label: str
) -> list:
    spec = _spec(model, label)
    names = spec.names
    unknown = [p for p in priors if p not in names]
    if unknown:
        raise ValueError(
            f"{label}: {sorted(map(str, unknown))} are not parameters of its "
            f"{getattr(model.dist, 'name', 'model')} model, whose parameters "
            f"are {names}.{_closest(unknown, names)}"
        )
    columns = np.tile(np.asarray(spec.params, dtype=float), (n, 1))
    for name, prior in priors.items():
        values = _quantiles(prior, n, rng, label, name)
        j = names.index(name)
        lower, upper = spec.bounds[j]
        if not np.all(np.isfinite(values)) or (
            (lower is not None and np.any(values <= lower))
            or (upper is not None and np.any(values >= upper))
        ):
            raise ValueError(
                f"{label}: the distribution for {name!r} gives values "
                f"outside its range ({lower}, {upper})."
            )
        columns[:, j] = values
    return [spec.build(row) for row in columns]


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
        if isinstance(model, PosteriorPredictive):
            # The posterior itself: one uniform a draw.
            return model._posterior_models(rng.random(n))
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


#: What a band over draws is, by the kinds of its uncertain inputs (see
#: ``band``).
CONFIDENCE, CREDIBLE, GIVEN, MIXED = "confidence", "credible", "given", "mixed"


def input_band(model: Any, spec: Any) -> str:
    """What drawing ``model`` as ``spec`` makes of its band: ``"credible"``
    for ``"fit"`` of a Bayesian fit (its posterior), ``"confidence"`` for
    ``"fit"`` of a fit with a parameter covariance (its sampling
    uncertainty), ``"given"`` for distributions or a list of models."""
    if isinstance(spec, str) and spec == FIT:
        return (
            CREDIBLE if isinstance(model, PosteriorPredictive) else CONFIDENCE
        )
    return GIVEN


def band(kinds: Iterable[str]) -> str:
    """What the band over draws of inputs of ``kinds`` (``input_band``'s)
    is: their one kind, or ``"mixed"``, a band over more than one kind of
    uncertainty at once (a confidence band's sampling uncertainty and a
    credible band's posterior, say)."""
    found = set(kinds)
    return found.pop() if len(found) == 1 else MIXED


def draw_ccf_models(
    group, spec: Any, n: int, rng: np.random.Generator
) -> List[Any]:
    """``n`` plausible common-cause models of a ``CCFGroup``, as ``spec``
    says: ``{parameter name: distribution}`` over its model's parameters
    (``beta``, an MGL model's ``gamma``, ``delta``, ..., an alpha-factor
    model's ``alpha_2``, ...; see ``ccf.parameters``), each drawn
    independently and in ``[0, 1]``, the others kept (alpha factors
    drawn to sum past 1 are refused, ``alpha_1`` being the remainder);
    or a list of models (``BetaFactor``, ``MGL`` or ``AlphaFactor``, for
    the group's size and with its model's basis), drawn with
    replacement."""
    from .ccf import _Model, parameters, with_parameters

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
            size = m.required_group_size() if isinstance(m, _Model) else None
            if (
                not isinstance(m, _Model)
                or size not in (None, len(group.members))
                or m.basis != model.basis
                or m.shocks != model.shocks
            ):
                raise ValueError(
                    f"{label}: every alternative model must be a "
                    "BetaFactor or an MGL or AlphaFactor model for "
                    f"{len(group.members)} "
                    f"members, splitting the {model.basis} as its model "
                    f"does, with {model.shocks} shocks; got {m!r}."
                )
        return [spec[i] for i in rng.integers(len(spec), size=n)]
    raise ValueError(
        f"{label}: its uncertainty must be a dict of distributions over its "
        f"model's parameters ({names}) or a list of models, got {spec!r}."
    )


def _variance(prior, label: str, name) -> float:
    """The variance of a parameter's distribution: its own (``var``), or
    from its quantile function on a fine grid of probabilities."""
    own = getattr(prior, "var", None)
    if callable(own):
        try:
            value = float(np.ravel(own())[0])
        except (TypeError, ValueError, AttributeError):
            value = float("nan")
        if np.isfinite(value) and value >= 0.0:
            return value
    u = (np.arange(4096) + 0.5) / 4096
    values = _quantiles_at(prior, u, label, name)
    return float(np.var(values))


def _quantiles_at(prior, u: np.ndarray, label: str, name) -> np.ndarray:
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


def rebuilt_by_surpyval(model) -> bool:
    """Whether ``"fit"`` draws ``model`` through surpyval's
    ``with_params`` (a regression node, a degradation or repairable-unit
    process, a mixture: see the module docstring), not as a surpyval
    distribution's parameters."""
    if _is_distribution(model) or isinstance(model, PosteriorPredictive):
        return False
    try:
        _redrawable(model, "")
    except ValueError:
        return False
    return True


def model_values(model) -> np.ndarray:
    """A parametric node model's parameters, in the order
    ``varied_parameters`` gives their positions and ``moved_model``
    takes them."""
    if rebuilt_by_surpyval(model):
        return np.asarray(_redrawable(model, "").params, dtype=float)
    spec = parametric_spec(model)
    assert spec is not None  # a varied model is parametric
    return np.asarray(spec.params, dtype=float)


def moved_model(model, values, label: str):
    """``model`` with its parameters (``model_values``'s) at ``values``,
    for the delta method's steps; a ValueError, saying what to do, where
    surpyval refuses them (a step past a parameter's range, which it does
    not give for these models)."""
    if not rebuilt_by_surpyval(model):
        spec = parametric_spec(model)
        assert spec is not None  # a varied model is parametric
        return spec.build(list(values))
    built = _tried(
        _redrawable(model, label).rebuild, np.asarray(values, float)
    )
    if built is None:
        raise ValueError(
            f"{label}: a step of the delta method took a parameter of its "
            f"{_kind(fitted_object(model))} past its range, which surpyval "
            "refuses (and gives no range for, to step inside it): "
            "method='sobol' draws it from its fit instead."
        )
    return built


def require_levers(model, spec, label: str) -> None:
    """Raise where a repairable diagram's delta method cannot vary a
    model's parameters: one ``"fit"`` draws through surpyval's
    ``with_params``, whose parameters are not levers of
    ``parameter_sensitivity``."""
    if isinstance(spec, str) and spec == FIT and rebuilt_by_surpyval(model):
        raise ValueError(
            f"{label}: its model is a surpyval "
            f"{_kind(fitted_object(model))}, whose parameters are not "
            "levers of parameter_sensitivity, which the delta method "
            "moves: method='sobol' draws it from its fit."
        )


def varied_parameters(model, spec: Any, label: str):
    """For the delta method: the parameters a node's uncertainty
    varies, as their positions among the model's parameters, their values,
    their covariance (a fit's ``covariance()``, or the variances of the
    distributions given) and their bounds. A list of models has no
    parameters to vary."""
    if isinstance(spec, str):
        if spec != FIT:
            raise ValueError(
                f"{label}: unknown uncertainty {spec!r}; give 'fit', a dict "
                "of parameter distributions or a list of models."
            )
        if isinstance(model, PosteriorPredictive):
            raise ValueError(
                f"{label}: its model is a Bayesian fit. The delta method "
                "moves a model's parameters a little about one value and "
                "weighs the change by their covariance; a Bayesian fit has "
                "its posterior instead, which need not be near a normal "
                "about one value: method='sobol' draws from the posterior "
                "itself."
            )
        if rebuilt_by_surpyval(model):
            # Their ranges as surpyval refuses values past them.
            rebuilt = _redrawable(model, label)
            return (
                list(range(len(rebuilt.params))),
                rebuilt.params,
                rebuilt.covariance,
                rebuilt.bounds,
            )
        _, params, covariance, bounds = _fit_covariance(model, label, True)
        return list(range(len(params))), params, covariance, bounds
    if isinstance(spec, Mapping):
        if not spec:
            raise ValueError(f"{label}: no parameter distributions given.")
        parameters = _spec(model, label)
        names = parameters.names
        unknown = [p for p in spec if p not in names]
        if unknown:
            raise ValueError(
                f"{label}: {sorted(map(str, unknown))} are not parameters of "
                f"its {getattr(model.dist, 'name', 'model')} model, whose "
                f"parameters are {names}."
            )
        params = np.asarray(parameters.params, dtype=float)
        bounds = parameters.bounds
        positions = [names.index(name) for name in spec]
        covariance = np.diag(
            [_variance(prior, label, name) for name, prior in spec.items()]
        )
        return (
            positions,
            params[positions],
            covariance,
            [bounds[j] for j in positions],
        )
    if isinstance(spec, Sequence):
        raise ValueError(
            f"{label}: a list of models has no parameters for the delta "
            "method to vary; method='sobol' draws from the list."
        )
    raise ValueError(
        f"{label}: the uncertainty must be 'fit', a dict of parameter "
        f"distributions or a list of models, got {spec!r}."
    )


def split_group_levers(key, spec, groups) -> Tuple[Optional[int], dict, Any]:
    """An uncertainty given under the tuple of a common-cause group's
    members (``groups``), with the group's own parameters named as its
    levers are (``ccf_beta``, ``ccf_alpha_2``, ..., as ``levers()`` and
    ``parameter_sensitivity`` key them): the group's index, its own
    parameters' distributions by their names (``beta``, ...), and what is
    left of ``spec`` for the members' one model. ``(None, {}, spec)`` for
    any other key or spec."""
    if not (isinstance(key, tuple) and isinstance(spec, Mapping)):
        return None, {}, spec
    index = next(
        (i for i, g in enumerate(groups) if set(g.members) == set(key)), None
    )
    if index is None:
        return None, {}, spec
    from .ccf import parameters

    own = {
        name[len("ccf_") :]: prior
        for name, prior in spec.items()
        if isinstance(name, str) and name.startswith("ccf_")
    }
    levers = [f"ccf_{name}" for name in parameters(groups[index].model)]
    unknown = [f"ccf_{name}" for name in own if f"ccf_{name}" not in levers]
    if unknown:
        raise ValueError(
            f"Common-cause group {list(key)!r}: {unknown} are not levers of "
            f"its model, {groups[index].model!r}, whose levers are "
            f"{levers}.{_closest(unknown, levers)}"
        )
    rest = {
        name: prior
        for name, prior in spec.items()
        if not (isinstance(name, str) and name.startswith("ccf_"))
    }
    return index, own, rest


def varied_ccf_parameters(group, spec: Any):
    """``varied_parameters`` for a common-cause group's model: the names
    of the parameters its uncertainty varies, their values, their
    covariance (the distributions' variances) and their bounds, [0, 1]."""
    from .ccf import parameters

    model = group.model
    label = f"Common-cause group {list(group.members)!r}"
    names = list(parameters(model))
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"{label}: a list of models has no parameters for the delta "
            "method to vary; method='sobol' draws from the list."
        )
    if not spec:
        raise ValueError(f"{label}: no parameter distributions given.")
    unknown = [p for p in spec if p not in names]
    if unknown:
        raise ValueError(
            f"{label}: {sorted(map(str, unknown))} are not parameters of its "
            f"model, {model!r}, whose parameters are {names}."
            f"{_closest(unknown, names)}"
        )
    values = parameters(model)
    chosen = list(spec)
    return (
        chosen,
        np.array([float(values[name]) for name in chosen]),
        np.diag(
            [_variance(prior, label, name) for name, prior in spec.items()]
        ),
        [(0.0, 1.0)] * len(chosen),
    )
