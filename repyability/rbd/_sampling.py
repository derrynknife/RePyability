"""Batched random draws that reproduce surpyval's own sampling exactly.

surpyval draws a sample from a plain parametric model as ``qf(u) + gamma``,
taking one uniform ``u`` from numpy's global RNG per sample. The simulations
used to make those draws one call at a time, and each call costs tens of
microseconds of scipy/surpyval overhead. Taking the *same* uniforms from the
global RNG in one block, in the same order, and applying ``qf`` to the block
yields the same numbers at a fraction of the cost, so seeded results do not
change.

:func:`inverse_sampler` returns ``None`` for any model whose sampling it
cannot reproduce exactly (nested RBDs, standby nodes, fixed-probability or
limited-failure-population models, ...), and callers then keep their
original, draw-at-a-time code path. :func:`row_sampler` extends the same idea
to composite node models (standby, repeated, load-sharing, regression and
nested-RBD nodes), which draw several uniforms per sample.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from surpyval import NonParametric, Parametric

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
    if (
        isinstance(model, NonParametric)
        and type(model).random is NonParametric.random
    ):
        # surpyval draws these from a fresh, OS-seeded generator on every
        # call, never from numpy's global RNG: they take no global uniforms
        # (and are not reproducible, batched or not).
        return RowSampler(
            0, lambda u: np.asarray(model.random(len(u)), dtype=float)
        )
    # Composite nodes describe their own draws.
    own = getattr(model, "_row_sampler", None)
    return own() if callable(own) else None


def inverse_sampler(model) -> Optional[Sampler]:
    """``u -> model.random(len(u))`` for the uniforms ``u`` that call would
    draw, when the model samples by inverse transform with exactly one global
    uniform per draw; otherwise ``None``.

    This mirrors the branch of surpyval's ``Parametric.random`` that such a
    model takes, operation for operation, so the values are identical.
    """
    if (
        isinstance(model, Parametric)
        and type(model).random is Parametric.random
        and model.p == 1
        and model.f0 == 0
        and hasattr(model.dist, "qf")
    ):
        dist, params, gamma = model.dist, model.params, model.gamma
        return lambda u: dist.qf(u, *params) + gamma
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


class UniformStream:
    """Single draws from many samplers, in any interleaving, from pre-drawn
    blocks of the global RNG's uniforms.

    Each call to :meth:`draw` takes the next uniform in the global stream,
    exactly as a ``random(1)`` call would, but the uniforms are drawn a block
    at a time and each sampler's ``qf`` is applied to the whole block at
    once. :meth:`close` rewinds the global RNG to just after the last uniform
    handed out, so it ends exactly where the single draws would have left it.
    """

    def __init__(self, block_size: int = 1024):
        self.block_size = block_size
        self._state: Any = None  # np.random.get_state() before the block
        self._block = np.empty(0)
        self._pos = 0
        self._values: dict = {}

    def draw(self, sampler: Sampler) -> float:
        if self._pos == len(self._block):
            self._state = np.random.get_state()
            self._block = np.random.random_sample(self.block_size)
            self._pos = 0
            self._values = {}
        values = self._values.get(sampler)
        if values is None:
            values = self._values[sampler] = np.asarray(
                sampler(self._block), dtype=float
            ).tolist()
        value = values[self._pos]
        self._pos += 1
        return value

    def close(self) -> None:
        if self._state is not None:
            np.random.set_state(self._state)
            np.random.random_sample(self._pos)
            self._state = None
            self._block = np.empty(0)
            self._pos = 0
            self._values = {}
