"""Stress-strength interference: the chance a strength exceeds a stress.

A part fails when the stress put on it reaches its strength. Both are
random: the strength varies from part to part, and the stress from use to
use. With a strength ``Y`` and a stress ``X`` that are independent, the
reliability of the part is

    ``R = P(Y > X) = integral of f_X(x) S_Y(x) dx``

where ``f_X`` is the stress's density and ``S_Y`` the strength's survival
function (the chance the strength is above a level). If the part has to
withstand ``n`` independent applications of the stress, it survives when
the strength is above the largest of them:

    ``R_n = P(Y > max(X_1..X_n)) = integral of n f_X F_X^(n-1) S_Y dx``

This is pure probability. RePyability takes the two models as given, fitted
in surpyval (or anything with the methods named in ``stress_strength``),
and fits nothing. It says nothing about where a strength comes from (no
fatigue, wear or crack-growth law): to let the strength change with age,
give the model it has at each age (``stress_strength_over_time``).

**The models share one unit.** Strength and stress must be in the same unit
(both in MPa, both in newtons, both in hours of cycles, ...). RePyability
does not check units; a model in the wrong unit gives a wrong answer.

How ``R`` is worked out:

- **Closed forms** where they exist, for models with no offset, no limited
  failure population and no zero inflation: both Normal, both LogNormal, both
  Exponential, and both Weibull with the same shape (and, for ``n`` loads,
  both Exponential or both Weibull with the same shape).
- **Quadrature** otherwise. With ``u = F_X(x)`` the integral is over the
  stress's probability from 0 to 1, of ``n u^(n-1) S_Y(Q_X(u))``, a smooth
  function between 0 and 1, which the Gauss-Kronrod (7, 15) rule of the
  mean-lifetime integral (``rbd/_mean_lifetime.py``) splits at the
  probabilities where the strength's quantiles fall, and halves until its
  error is within ``RTOL``, relative.
- **Simulation** on request (``method="simulate"``).
"""

import math
import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy import special, stats

from repyability.rbd import _mean_lifetime
from repyability.rbd._model_utils import (
    MODEL_ERRORS,
    distribution_name,
    is_fixed_probability,
    never_fails,
)
from repyability.rbd.results import StressStrengthCurve, StressStrengthResult
from repyability.rbd.uncertainty import draw_models
from repyability.utils import checks
from repyability.utils.wrappers import numpy_seed

#: The methods ``stress_strength`` takes.
METHODS = ("auto", "numerical", "simulate")
#: The values of a surpyval model's offset, limited-failure-population
#: share and zero inflation that mean it has none.
_PLAIN = {"gamma": 0.0, "lfp_p": 1.0, "f0": 0.0}
#: The probabilities of the stress and the strength where the integral is
#: split: its tails, and the bulk a tenth at a time.
_PROBABILITIES = np.unique(
    np.concatenate(
        [
            10.0 ** -np.arange(1, 16),
            np.linspace(0.0, 1.0, 21)[1:-1],
            1.0 - 10.0 ** -np.arange(1, 16),
        ]
    )
)
#: The probability the stress's quantile function is asked for, at the
#: least and the most: 0 and 1 are the ends of its support, which can be
#: infinite.
_LEAST = 1e-300
_MOST = 1.0 - 2.0**-53
#: The draws of the models' parameters ``uncertainty`` takes.
DEFAULT_DRAWS = 1000
#: The simulated pairs of draws ``method="simulate"`` takes.
DEFAULT_SAMPLES = 100_000


def _check_model(model: Any, name: str, methods: Tuple[str, ...]) -> None:
    """Refuse a ``model`` that is not a model of a distribution, or lacks
    the ``methods`` the analysis needs of the ``name`` (strength or
    stress)."""
    if model is None or isinstance(model, (str, bytes, int, float, bool)):
        raise TypeError(
            f"{name} must be a model with {', '.join(methods)} (a fitted "
            f"surpyval model, say), got {model!r}."
        )
    checks.no_distribution(model, name)
    missing = [m for m in methods if not callable(getattr(model, m, None))]
    if missing:
        raise TypeError(
            f"{name} must be a model with {', '.join(methods)} (a fitted "
            f"surpyval model, say); {type(model).__name__} has no "
            f"{', '.join(missing)}."
        )
    if is_fixed_probability(model):
        raise ValueError(
            f"{name} is a fixed probability of an event, not a strength or "
            "a stress with a spread of values."
        )


def _check_models(strength: Any, stress: Any, simulate: bool) -> None:
    _check_model(strength, "strength", ("sf",) + (("random",) * simulate))
    _check_model(
        stress, "stress", ("qf",) if not simulate else ("qf", "random")
    )
    if never_fails(stress) > 0.0:
        raise ValueError(
            "stress is a limited failure population: a share of the "
            "stresses never occurs, which the interference does not take. "
            "Give the distribution of the stress that does occur."
        )


def _check_loads(n_loads: Any) -> int:
    if not checks.is_whole(n_loads) or n_loads < 1:
        raise ValueError(
            f"n_loads must be a whole number of at least 1, got {n_loads!r}."
        )
    return int(n_loads)


def _values(model: Any) -> Optional[Tuple[str, Tuple[float, ...]]]:
    """The distribution's name and parameters of a plain surpyval
    ``model`` (no offset, limited failure population or zero inflation),
    else None."""
    name = distribution_name(model)
    if name is None:
        return None
    extras = getattr(model, "extras", {})
    if any(extras.get(k, v) != v for k, v in _PLAIN.items()):
        return None
    params = tuple(float(p) for p in np.atleast_1d(model.params))
    return name, params


def _closed_form(
    strength: Any, stress: Any, n_loads: int
) -> Optional[Tuple[float, float]]:
    """``(R, 1 - R)`` where a closed form gives it, else None.

    Both Normal: ``R = Phi((mu_Y - mu_X) / sqrt(s_X^2 + s_Y^2))``, since
    ``Y - X`` is normal. Both LogNormal: the same of the logs' parameters,
    since ``log Y - log X`` is normal. Both Exponential, or both Weibull
    with one shape: with ``r = (scale_X / scale_Y)^shape`` (``lam_Y /
    lam_X`` for the exponentials), ``R_n = n B(n, 1 + r)``, which for one
    stress is ``1 / (1 + r)``.
    """
    y, x = _values(strength), _values(stress)
    if y is None or x is None or y[0] != x[0]:
        return None
    kind, (py, px) = y[0], (y[1], x[1])
    if kind in ("Normal", "LogNormal") and n_loads == 1:
        z = (py[0] - px[0]) / math.hypot(px[1], py[1])
        return float(stats.norm.cdf(z)), float(stats.norm.cdf(-z))
    if kind == "Exponential":
        r = py[0] / px[0]
    elif kind == "Weibull" and math.isclose(py[1], px[1], rel_tol=1e-12):
        r = (px[0] / py[0]) ** px[1]
    else:
        return None
    if not (math.isfinite(r) and r > 0.0):
        return None
    if n_loads == 1:
        return 1.0 / (1.0 + r), r / (1.0 + r)
    n = float(n_loads)
    reliable = math.exp(math.log(n) + special.betaln(n, 1.0 + r))
    failed = _failure_by_parts(n, r)
    return reliable, failed


def _failure_by_parts(n: float, r: float) -> float:
    """``1 - n B(n, 1 + r)`` without cancelling, for a whole ``n``: it is
    ``1 - n! / ((1 + r)(2 + r)...(n + r))``, one minus a product of ``k /
    (k + r)``, which ``-expm1`` of the sum of logs keeps exact when the
    product is near 1."""
    k = np.arange(1.0, n + 1.0)
    return float(-np.expm1(-np.sum(np.log1p(r / k))))


def _quadrature(
    strength: Any, stress: Any, n_loads: int, failure: bool = False
) -> float:
    """``R_n`` (or ``1 - R_n``, from the strength's ``ff``, if ``failure``)
    by quadrature over the stress's probability ``u``."""
    n = float(n_loads)
    side = strength.ff if failure else strength.sf

    def integrand(u: np.ndarray) -> np.ndarray:
        u = np.clip(u, _LEAST, _MOST)
        with np.errstate(all="ignore"):
            level = np.asarray(stress.qf(u), dtype=float).ravel()
            value = np.asarray(side(level), dtype=float).ravel()
        if np.any(np.isnan(value)):
            raise ValueError(
                "The strength or the stress gave a value that is not a "
                "number, so the reliability cannot be worked out."
            )
        # The edge of a support: a level beyond it is a sure survival or
        # failure of the strength, which an infinite level gives as nan.
        return value * n * u ** (n - 1.0)

    return _mean_lifetime._integrate(integrand, _edges(strength, stress, n))


def _edges(strength: Any, stress: Any, n: float) -> np.ndarray:
    """The probabilities of the stress at which to split its integral:
    a grid through its tails and bulk, and where the strength's quantiles
    fall (the strength's survival function bends there, and starts)."""
    # The largest of n stresses has the quantile Q(p ** (1 / n)).
    edges = [0.0, 1.0, *(_PROBABILITIES ** (1.0 / n))]
    ff = getattr(stress, "ff", None)
    qf = getattr(strength, "qf", None)
    if ff is not None and qf is not None:
        try:
            with np.errstate(all="ignore"):
                where = np.asarray(
                    ff(np.asarray(qf(_PROBABILITIES), dtype=float)),
                    dtype=float,
                ).ravel()
        except MODEL_ERRORS:  # a model whose quantiles cannot be taken
            where = np.empty(0)
        edges.extend(where[np.isfinite(where)])
    return np.unique(np.clip(edges, 0.0, 1.0))


def _reliability(
    strength: Any, stress: Any, n_loads: int, numerical: bool = False
) -> Tuple[float, float, str]:
    """``(R, 1 - R, method)`` by a closed form if there is one (and
    ``numerical`` is not asked), else by quadrature."""
    if not numerical:
        exact = _closed_form(strength, stress, n_loads)
        if exact is not None:
            return exact[0], exact[1], "exact"
    reliable = _quadrature(strength, stress, n_loads)
    if callable(getattr(strength, "ff", None)):
        failed = _quadrature(strength, stress, n_loads, failure=True)
    else:
        failed = 1.0 - reliable
    return _unit(reliable), _unit(failed), "numerical"


def _unit(value: float) -> float:
    """``value`` kept inside [0, 1], where quadrature rounds past it."""
    return min(1.0, max(0.0, float(value)))


def _index(reliable: float, failed: float) -> float:
    """The normal quantile of the reliability, from whichever of ``R`` and
    ``1 - R`` is the smaller, so that it keeps its precision."""
    if reliable <= failed:
        return float(stats.norm.ppf(reliable))
    return float(-stats.norm.ppf(failed))


def _simulate(
    strength: Any,
    stress: Any,
    n_loads: int,
    mc_samples: int,
    seed: Any,
) -> Tuple[float, float]:
    """The share of ``mc_samples`` draws in which the strength exceeds the
    largest of ``n_loads`` stresses, and its standard error."""
    with numpy_seed(seed):
        y = np.asarray(strength.random(mc_samples), dtype=float).ravel()
        x = np.asarray(
            stress.random(mc_samples * n_loads), dtype=float
        ).reshape(mc_samples, n_loads)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError(
            "The strength or the stress drew a value that is not a number."
        )
    p = float(np.mean(y > x.max(axis=1)))
    return p, math.sqrt(p * (1.0 - p) / mc_samples)


def stress_strength(
    strength: Any,
    stress: Any,
    n_loads: int = 1,
    method: str = "auto",
    uncertainty: Optional[str] = None,
    n_draws: int = DEFAULT_DRAWS,
    confidence: float = 0.95,
    mc_samples: int = DEFAULT_SAMPLES,
    seed: Any = None,
) -> StressStrengthResult:
    """The reliability of a part whose strength must exceed its stress:
    ``R = P(strength > stress)``, for independent strength and stress.

    In plain words: the strength varies from part to part, the stress from
    use to use, and the part fails in a use whose stress is above its
    strength. ``R`` is the chance a randomly chosen part survives a randomly
    chosen use. With ``n_loads`` uses in a row (each stress drawn
    independently, the strength fixed for the part), the part survives
    when its strength is above the largest of them. Strength and stress are
    random variables in the same unit, which is not checked. Exponentials
    and normals with the same mean and spread give very different answers
    from the single "factor of safety" of their means: this is the whole
    distribution's answer. See the user guide for formulas and a worked
    example.

    Parameters
    ----------
    strength : model
        The strength: a fitted surpyval model, or anything with ``sf``
        (and, for the failure side to keep its precision, ``ff`` and, for
        better splits, ``qf``; ``random`` if simulating).
    stress : model
        The stress: a fitted surpyval model, or anything with ``qf`` (and
        ``ff`` for better splits; ``random`` if simulating). In the unit of
        ``strength``.
    n_loads : int
        The number of independent applications of the stress the strength
        must survive. The default is 1.
    method : str
        ``"auto"`` for a closed form where there is one and quadrature
        otherwise (both exact to the precision of the models' own
        methods); ``"numerical"`` for quadrature even where a closed form
        exists; ``"simulate"`` for the Monte-Carlo estimate.
    uncertainty : str, optional
        ``"fit"`` to draw both models' parameters from the normal
        approximation of their fitted estimates (surpyval's
        ``covariance()``), independently of each other (they are fitted to
        separate data), and give the bounds of ``R`` over the draws. A
        model that has no parameter covariance (not fitted by maximum
        likelihood, a mixture, a non-parametric fit) is refused. Not with
        ``method="simulate"``.
    n_draws : int
        The number of draws of the parameters for ``uncertainty``.
    confidence : float
        The confidence level of the bounds, strictly between 0 and 1.
    mc_samples : int
        The number of simulated draws of the strength and the stress when
        ``method="simulate"``.
    seed : int, optional
        Seeds the simulation and the draws of the parameters. The same
        seed gives the same results.

    Returns
    -------
    StressStrengthResult
        ``R`` (``reliability``), ``1 - R``, the reliability index, how it
        was found, and the standard error (simulated) or the bounds
        (uncertainty).

    Raises
    ------
    TypeError
        If a model lacks the methods it needs, or is a surpyval
        distribution itself rather than a model of it.
    ValueError
        If an argument is out of range, a model is a fixed probability or
        has a share that never occurs, or the uncertainty cannot be drawn.
    """
    n_loads = _check_loads(n_loads)
    checks.one_of("method", method, METHODS)
    simulate = method == "simulate"
    _check_models(strength, stress, simulate)
    seed = checks.seed(seed)
    if uncertainty is not None:
        checks.one_of("uncertainty", uncertainty, ("fit",))
        if simulate:
            raise ValueError(
                "uncertainty is not drawn with method='simulate': the "
                "simulation's standard_error is its own spread."
            )
        n_draws = _draws(n_draws)
        confidence = _confidence(confidence)
    if simulate:
        mc_samples = _samples(mc_samples)
        p, error = _simulate(strength, stress, n_loads, mc_samples, seed)
        return StressStrengthResult(
            reliability=p,
            failure_probability=1.0 - p,
            reliability_index=_index(p, 1.0 - p),
            n_loads=n_loads,
            method="simulated",
            standard_error=error,
            n_samples=mc_samples,
        )
    reliable, failed, how = _reliability(
        strength, stress, n_loads, numerical=method == "numerical"
    )
    bounds: dict = {}
    if uncertainty is not None:
        bounds = _bounds(
            strength,
            stress,
            n_loads,
            method == "numerical",
            n_draws,
            confidence,
            seed,
        )
    return StressStrengthResult(
        reliability=reliable,
        failure_probability=failed,
        reliability_index=_index(reliable, failed),
        n_loads=n_loads,
        method=how,
        **bounds,
    )


def _draws(n_draws: Any) -> int:
    if not checks.is_whole(n_draws) or n_draws < 2:
        raise ValueError(
            f"n_draws must be a whole number of at least 2, got {n_draws!r}."
        )
    return int(n_draws)


def _samples(mc_samples: Any) -> int:
    if not checks.is_whole(mc_samples) or mc_samples < 2:
        raise ValueError(
            "mc_samples must be a whole number of at least 2, got "
            f"{mc_samples!r}."
        )
    return int(mc_samples)


def _confidence(confidence: Any) -> float:
    if not checks.is_number(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be a number between 0 and 1, got {confidence!r}."
        )
    return float(confidence)


def _bounds(
    strength: Any,
    stress: Any,
    n_loads: int,
    numerical: bool,
    n_draws: int,
    confidence: float,
    seed: Any,
) -> dict:
    """The bounds of ``R`` over ``n_draws`` draws of both models'
    parameters (drawn independently), as the fields of the result."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        # An offset held where it was fitted warns once, not per model.
        warnings.simplefilter("default")
        weak = draw_models(strength, "fit", n_draws, rng, "strength")
        load = draw_models(stress, "fit", n_draws, rng, "stress")
    values = np.array(
        [_reliability(y, x, n_loads, numerical)[0] for y, x in zip(weak, load)]
    )
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "n_draws": n_draws,
    }


def stress_strength_over_time(
    strength_at: Callable[[float], Any],
    stress: Any,
    times: Any,
    n_loads: int = 1,
    method: str = "auto",
) -> StressStrengthCurve:
    """The reliability of a part whose strength changes with age, against a
    stress that does not: ``R(t) = P(strength at age t > stress)``.

    RePyability has no law for how a strength changes (that is the user's
    knowledge of the part, or a model fitted in surpyval): give the model it
    has at each age. The result is the interference reliability of each
    age's model against the stress, worked out as ``stress_strength`` does.
    It is the chance that a part of that age survives one use (or
    ``n_loads`` uses in a row), not that it survives from new to that age.

    Parameters
    ----------
    strength_at : callable
        A function of an age (a float) that returns the strength's model at
        that age, such as ``lambda t: surv.Normal.from_params([100 - 2 * t,
        8])``.
    stress : model
        The stress's model, as in ``stress_strength``, in the strength's
        unit. It does not change with age.
    times : array_like
        The ages, finite numbers, in the unit ``strength_at`` takes.
    n_loads : int
        The number of independent applications of the stress at each age.
    method : str
        ``"auto"`` for a closed form where there is one and quadrature
        otherwise, or ``"numerical"`` for quadrature throughout.

    Returns
    -------
    StressStrengthCurve
        ``R(t)`` and ``1 - R(t)`` at each age.

    Raises
    ------
    TypeError
        If ``strength_at`` is not callable, or gives something that is not
        a model.
    ValueError
        If an argument is out of range.
    """
    if not callable(strength_at):
        raise TypeError(
            "strength_at must be a function of an age that returns the "
            f"strength's model at that age, got {strength_at!r}."
        )
    n_loads = _check_loads(n_loads)
    checks.one_of("method", method, ("auto", "numerical"))
    ages = checks.real_array(times, "times").ravel()
    if ages.size == 0 or not np.all(np.isfinite(ages)):
        raise ValueError("times must be a non-empty list of finite numbers.")
    _check_model(stress, "stress", ("qf",))
    reliable, failed, hows = [], [], set()
    for age in ages:
        model = strength_at(float(age))
        _check_model(model, f"strength at age {age:g}", ("sf",))
        r, q, how = _reliability(
            model, stress, n_loads, numerical=method == "numerical"
        )
        reliable.append(r)
        failed.append(q)
        hows.add(how)
    return StressStrengthCurve(
        times=ages,
        reliability=np.array(reliable),
        failure_probability=np.array(failed),
        n_loads=n_loads,
        method="exact" if hows == {"exact"} else "numerical",
    )
