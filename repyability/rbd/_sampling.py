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


#: Steps a mixture's quantile takes at most: Newton's, or halving its
#: bracket where a Newton step would leave it (about 6 are taken).
_STEPS = 200
#: A Newton step this small, relative to the point, ends the search.
_SETTLED = 4.0 * np.finfo(float).eps


def _components(model) -> Callable[[str, np.ndarray], np.ndarray]:
    """``values(name, x)``: a mixture's components' function ``name``
    (``"ff"``, ``"sf"``, ``"df"`` or ``"qf"``) at the 1-d ``x``, one column
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
                )
            )
    except MODEL_ERRORS:
        ok = False
    return together if ok else one_by_one


def mixture_quantile(model) -> Sampler:
    """A surpyval ``MixtureModel``'s quantile function, which its own
    (from surpyval 0.24) is not yet fit to replace: that loses the upper
    tail, inverting ``F`` there, and takes some 200 times as long (surpyval
    #821). For each ``u``, the ``x`` with ``F(x) = u``, to the last bit or
    two. The mixture's quantile lies between its
    components' (``F`` is their weighted sum), which bracket it; Newton's
    steps, with the density, go from the bracket's middle (geometric while
    its lower end is above 0), and a step that would leave the bracket
    halves it instead. Above ``u = 1/2`` the survival function is compared
    with ``1 - u``, which is exact, for the long lives' precision."""
    weights = np.ravel(np.asarray(model.w, dtype=float))
    values = _components(model)

    def mixed(name: str, x: np.ndarray) -> np.ndarray:
        return np.sum(values(name, x) * weights, axis=1)

    def middle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(a > 0.0, np.sqrt(a) * np.sqrt(b), 0.5 * (a + b))

    def quantile(u):
        u = np.asarray(u, dtype=float)
        shape = u.shape
        u = u.ravel()
        with np.errstate(all="ignore"):
            each = values("qf", u)
        lo, hi = each.min(axis=1), each.max(axis=1)
        x = hi.copy()
        upper = u > 0.5
        left = 1.0 - u
        active = np.flatnonzero(hi > lo)
        with np.errstate(all="ignore"):
            x[active] = middle(lo[active], hi[active])
        for _ in range(_STEPS):
            if not active.size:
                break
            at = x[active]
            with np.errstate(all="ignore"):
                # Positive past the quantile, and rising at the density.
                residual = np.where(
                    upper[active],
                    left[active] - mixed("sf", at),
                    mixed("ff", at) - u[active],
                )
                newton = at - residual / mixed("df", at)
            exact = residual == 0.0
            past = residual > 0.0
            hi[active[past]] = at[past]
            short = ~past & ~exact
            lo[active[short]] = at[short]
            a, b = lo[active], hi[active]
            inside = (newton > a) & (newton < b)
            with np.errstate(all="ignore"):
                step = np.where(inside, newton, middle(a, b))
            # Newton's step, within the bracket, too small to go on with.
            small = (
                (newton >= a)
                & (newton <= b)
                & (np.abs(newton - at) <= _SETTLED * np.abs(at))
            )
            spent = ~((step > a) & (step < b))
            # The root itself, Newton's last step, the bracket's upper end
            # where no float is left inside it, or the next step.
            x[active] = np.where(
                exact | small,
                np.where(exact, at, newton),
                np.where(spent, b, step),
            )
            active = active[~(exact | small | spent)]
        return x.reshape(shape)

    return quantile


class MixtureLife:
    """A surpyval ``MixtureModel`` with :func:`mixture_quantile` for its
    quantile function (surpyval #821), for surpyval's ``conditional_gaps``
    to draw a life given an age with: its cumulative hazard and quantile.
    :meth:`of` keeps one a mixture."""

    _made: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

    def __init__(self, model):
        self._model = model
        self.qf = mixture_quantile(model)

    def Hf(self, x):
        return self._model.Hf(x)

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
    surpyval ``MixtureModel``'s quantile worked out
    (:func:`mixture_quantile`). None for a model that draws its own way.
    (A ``NonRepairableRBD``'s draws replay surpyval's own, and a mixture
    draws there as surpyval does.)"""
    sampler = inverse_sampler(model)
    if sampler is None and is_mixture(model):
        return mixture_quantile(model)
    if sampler is None and is_degradation(model):
        # A degradation process's first-passage time from its starting
        # level (#271), by its own quantile function.
        return lambda u: np.asarray(model.qf(u), dtype=float)
    return sampler
