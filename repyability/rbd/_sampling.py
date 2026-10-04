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

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from surpyval import Parametric

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
    from repyability.rbd._model_utils import is_fixed_probability

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
        if model.p == 1 and model.f0 == 0:
            dist, params, gamma = model.dist, model.params, model.gamma
            return lambda u: dist.qf(u, *params) + gamma
        return lambda u: np.asarray(model.qf(u), dtype=float)
    return None


def draw_rows(samplers: list[Sampler], size: int) -> list[np.ndarray]:
    """``size`` rounds of one draw from each sampler, in order, as a list of
    ``size``-long arrays (one per sampler).

    The uniforms are taken row by row -- round 0's draws first -- which is
    the order a loop over rounds making one ``random(1)`` call per model
    would consume them in.
    """
    u = np.random.random_sample((size, len(samplers)))
    return [column(u, j, sampler) for j, sampler in enumerate(samplers)]
