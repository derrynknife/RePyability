"""Batched random draws that reproduce surpyval's own sampling exactly.

surpyval draws a sample from a plain parametric model as ``qf(u) + gamma``,
taking one uniform ``u`` from numpy's global RNG per sample. Making those
draws one call at a time costs tens of microseconds of scipy/surpyval
overhead each. Taking the *same* uniforms from the global RNG in one block,
in the same order, and applying ``qf`` to the block yields the same numbers
at a fraction of the cost. (``RepairableRBD``'s simulations take their
uniforms from streams of their own instead, see ``_streams``, and turn
them into draws with the same samplers.)

:func:`inverse_sampler` returns ``None`` for any model whose sampling it
cannot reproduce exactly (nested RBDs, standby nodes, fixed-probability
models, ...), and callers then keep their original, draw-at-a-time code
path. :func:`row_sampler` extends the same idea to composite node models
(standby, repeated, load-sharing, regression and nested-RBD nodes), which
draw several uniforms per sample.

A limited-failure-population or zero-inflated surpyval model draws its
lifetimes through its own quantile function, one global uniform each:
infinite for a unit that never fails, 0 for one dead on arrival.
:func:`inverse_sampler` replays that too.
"""

import math
import weakref
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from surpyval import Parametric

from ._degradation import is_degradation
from ._model_utils import (
    MODEL_ERRORS,
    is_fixed_probability,
    is_mixture,
    lfp_p,
)
from .helper_classes import PerfectReliability, PerfectUnreliability

Sampler = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RowSampler:
    """A model's ``random(1)``, replayed for many samples at once.

    One ``random(1)`` call takes ``width`` uniforms from the global RNG, in a
    fixed order. ``draw`` maps an ``(n, width)`` block of uniforms -- one row
    per call, columns in that order -- to the ``n`` values those calls
    return. The block is taken row by row, so drawing it consumes the global
    RNG exactly as ``n`` successive calls would.
    """

    width: int
    draw: Callable[[np.ndarray], np.ndarray]


def column(u: np.ndarray, j: int, sampler: Sampler) -> np.ndarray:
    """``sampler`` applied to column ``j`` of a block of uniforms.

    Copied to a contiguous array first, as a single draw's uniform is, so
    that numpy's vectorised math runs the same code path."""
    return np.asarray(sampler(np.ascontiguousarray(u[:, j])), dtype=float)


def row_sampler(model) -> Optional[RowSampler]:
    """The :class:`RowSampler` for a node model, or ``None`` if its
    ``random(1)`` cannot be replayed exactly."""
    if model is PerfectReliability:
        return RowSampler(0, lambda u: np.full(len(u), np.inf))
    if model is PerfectUnreliability:
        return RowSampler(0, lambda u: np.zeros(len(u)))
    sampler = inverse_sampler(model)
    if sampler is not None:
        return RowSampler(1, lambda u: column(u, 0, sampler))
    # Composite nodes describe their own draws.
    own = getattr(model, "_row_sampler", None)
    return own() if callable(own) else None


def lifetime_sampler(model) -> Optional[RowSampler]:
    """A component's lifetimes as a :class:`RowSampler` (see
    :func:`row_sampler`), or None if they cannot be drawn in a block. A
    fixed probability, which surpyval draws as an event indicator, is a
    unit that fails at the start (a lifetime of 0) or never (``inf``)."""

    if is_fixed_probability(model):
        failure = float(np.ravel(model.ff(1.0))[0])
        return RowSampler(
            1, lambda u: np.where(u[:, 0] < failure, 0.0, np.inf)
        )
    return row_sampler(model)


def inverse_sampler(model) -> Optional[Sampler]:
    """``u -> model.random(len(u))`` for the uniforms ``u`` that call would
    draw, when the model samples by inverse transform with exactly one global
    uniform per draw; otherwise ``None``.

    This mirrors the branch of surpyval's ``Parametric.random`` that such a
    model takes, operation for operation, so the values are identical: the
    distribution's quantile function plus the offset for a plain model, and
    the model's own quantile function for a limited-failure-population or
    zero-inflated one.
    """
    if (
        isinstance(model, Parametric)
        and type(model).random is Parametric.random
        and hasattr(model.dist, "qf")
    ):
        if lfp_p(model) == 1 and model.f0 == 0:
            dist, params, gamma = model.dist, model.params, model.gamma
            return lambda u: dist.qf(u, *params) + gamma
        return lambda u: np.asarray(model.qf(u), dtype=float)
    return None


#: Steps a mixture's life left at an age takes at most: Newton's, or
#: halving its bracket where a Newton step would leave it (about 3 are
#: taken).
_STEPS = 200
#: A Newton step this small, relative to the point, ends the search.
_SETTLED = 4.0 * np.finfo(float).eps


def _components(model) -> Callable[[str, np.ndarray], np.ndarray]:
    """``values(name, x)``: a mixture's components' function ``name``
    (``"ff"``, ``"sf"``, ``"df"``, ``"qf"``, ``"Hf"`` or ``"hf"``) at the
    1-d ``x``, one column
    a component. In one call over every component where the distribution's
    functions broadcast over their parameters (checked once, against a call
    a component), else one call each."""
    dist = model.dist
    rows = np.atleast_2d(np.asarray(model.params, dtype=float))
    columns = [rows[:, j][None, :] for j in range(rows.shape[1])]

    def one_by_one(name: str, x: np.ndarray) -> np.ndarray:
        f = getattr(dist, name)
        return np.stack([np.ravel(f(x, *row)) for row in rows], axis=1)

    def together(name: str, x: np.ndarray) -> np.ndarray:
        return np.asarray(getattr(dist, name)(x[:, None], *columns), float)

    probe = np.array([0.1, 0.5, 0.9])
    try:
        with np.errstate(all="ignore"):
            points = np.ravel(one_by_one("qf", probe))
            ok = all(
                np.array_equal(
                    together(name, at), one_by_one(name, at), equal_nan=True
                )
                for name, at in (
                    ("qf", probe),
                    ("ff", points),
                    ("sf", points),
                    ("df", points),
                    ("Hf", points),
                    ("hf", points),
                )
            )
    except MODEL_ERRORS:
        ok = False
    return together if ok else one_by_one


class MixtureLife:
    """A surpyval ``MixtureModel``'s life left at a virtual age
    (:meth:`aged`, #295), with its cumulative hazard and quantile function
    (the model's own) as surpyval's ``conditional_gaps`` takes a lifetime.
    :meth:`of` keeps one a mixture.

    Kept although surpyval 0.27 draws the same life itself
    (``conditional_gaps``): that inverts the quantile function wherever
    the cumulative hazard to reach is under 20, which loses about
    ``eps * exp(H)`` of the age it finds (some 1e-10 relative at ``u =
    1e-6``), where :meth:`aged` is the root to the last bit or two, in
    a few evaluations (about a seventh of the time)."""

    _made: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

    def __init__(self, model):
        self._model = model
        self.qf = model.qf
        self._values = _components(model)
        self._log_w = np.log(np.ravel(np.asarray(model.w, dtype=float)))
        self._w = np.exp(self._log_w)
        self._start: Optional[tuple] = None

    def Hf(self, x):
        return self._model.Hf(x)

    def _cumulative(self, x: np.ndarray) -> tuple:
        """The mixture's cumulative hazard and hazard at the 1-d ``x``,
        from its components' (``H = -log sum w_i exp(-H_i)``, and ``h``
        the components' hazards weighed by their shares of the survivors,
        ``w_i S_i``): from the share that has failed while it is under a
        half, where the cumulative hazard is small, else in logs, so that
        both tails keep their precision."""
        with np.errstate(all="ignore"):
            each = self._values("Hf", x)
            rates = self._values("hf", x)
            logs = self._log_w - each
            top = np.max(logs, axis=1, keepdims=True)
            alive = np.exp(logs - top)
            total = np.sum(alive, axis=1)
            failed = -np.expm1(-each) @ self._w
            H = np.where(
                failed < 0.5,
                -np.log1p(-failed),
                -(top[:, 0] + np.log(total)),
            )
            h = np.sum(alive * rates, axis=1) / total
        return H, h

    def aged(self, age: float, u: float) -> float:
        """The life left at virtual age ``age``, from the uniform ``u``: the
        ``x`` with ``H(age + x) = H(age) - log(u)``, as surpyval's
        ``conditional_gaps`` draws it (#295), to the last bit or two.
        Newton's steps on ``H``, with its hazard, from a table of ``H``
        made once a model, inside a bracket that a step leaving it halves
        (about 3 evaluations a draw, where ``conditional_gaps`` takes the
        quantile's or some 150 of ``H`` deep in the tail)."""
        if not u > 0.0:
            return math.inf
        H_age = float(self._cumulative(np.array([age]))[0][0]) if age else 0.0
        if math.isinf(H_age):
            return 0.0  # no survival past its age: it fails at once
        target = H_age - math.log(u)
        if not math.isfinite(target):
            return math.inf
        if target <= H_age:
            return 0.0
        lo, hi = age, math.inf
        x = max(self._first_guess(target), age)
        for _ in range(_STEPS):
            H, h = (float(v[0]) for v in self._cumulative(np.array([x])))
            if H < target:
                lo = x
            else:
                hi = x
            step = (H - target) / h if h > 0.0 else math.nan
            if abs(step) <= _SETTLED * x or H == target:
                return max(x - step - age, 0.0)
            new = x - step
            if not lo < new < hi:
                # Out of the bracket: its middle, or further out.
                new = 0.5 * (lo + hi) if math.isfinite(hi) else 2.0 * x + 1.0
            x = new
        return max(x - age, 0.0)

    def _first_guess(self, target: float) -> float:
        """Where ``H`` reaches ``target``, read off a table of ``H`` on a
        log grid over the components' lives (made once a model)."""
        if self._start is None:
            with np.errstate(all="ignore"):
                scale = float(np.max(self._values("qf", np.array([0.5]))))
            grid = scale * np.logspace(-9.0, 9.0, 2048)
            H, _ = self._cumulative(grid)
            kept = np.isfinite(H) & (H > 0.0)
            kept[kept] &= np.diff(np.r_[-np.inf, H[kept]]) > 0.0
            self._start = (np.log(H[kept]), np.log(grid[kept]))
        log_H, log_x = self._start
        if not log_H.size:
            return 0.0
        return math.exp(np.interp(math.log(target), log_H, log_x))

    @classmethod
    def of(cls, model) -> "MixtureLife":
        """The one for ``model``, made again if its parameters or weights
        have changed since."""
        key = (
            model.dist.name,
            np.asarray(model.params, dtype=float).tobytes(),
            np.asarray(model.w, dtype=float).tobytes(),
        )
        try:
            made = cls._made.get(model)
        except TypeError:  # not weakly referenced
            return cls(model)
        if made is None or made[0] != key:
            made = (key, cls(model))
            cls._made[model] = made
        return made[1]


def stream_sampler(model) -> Optional[Sampler]:
    """What a ``RepairableRBD``'s streams turn a model's uniforms into its
    draws with, one uniform a draw: :func:`inverse_sampler`'s, or a
    surpyval ``MixtureModel``'s quantile function (its own, precise in
    both tails from surpyval 0.25), or a Bayesian fit's posterior predictive's
    draw (``PosteriorPredictive._draw``). None for a model that draws its
    own way.
    (A ``NonRepairableRBD``'s draws replay surpyval's own, and a mixture
    draws there as surpyval does.)"""
    from ._bayesian_life import PosteriorPredictive

    sampler = inverse_sampler(model)
    if sampler is None and is_mixture(model):
        return lambda u: np.asarray(model.qf(u), dtype=float)
    if isinstance(model, PosteriorPredictive):
        # A Bayesian fit's posterior predictive: a posterior point and a
        # life from it, both from the one uniform (see its _draw).
        return model._draw
    if sampler is None and is_degradation(model):
        # A degradation process's first-passage time from its starting
        # level (#271), by its own quantile function.
        return lambda u: np.asarray(model.qf(u), dtype=float)
    return sampler
