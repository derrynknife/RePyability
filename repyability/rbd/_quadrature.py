"""Integrals over time of functions of the nodes' curves (#164).

The analyses over a window -- the mission availability and capacity, and
the expected events and cost -- integrate from 0 the system's point
availability, its capacity's distribution or its rate of failures, all
functions of its nodes' curves (see ``_point_availability``). A curve is
smooth between its *breaks* (where a scheduled replacement or a dip starts,
say), and between them linear on a grid of 2,000 steps of its typical up
time (``curve_grids``), many more than its shape needs.

The integrals are summed over pieces (``pieces``): the breaks, with the
gaps between them cut so that no piece spans more than ``PIECE_STEPS``
steps of the finest grid still changing there, so that its quadrature
points miss no bend of the curves. Each piece is then halved until 4-point
Gauss-Legendre quadrature on it agrees with the sum on its halves, to
``TOLERANCE`` (``refined``): the integral of a probability (the
availability, the capacity) as far as it takes, and of the expected events
until the piece is a step of that grid long, within which the curves are
linear and the counts no closer. Summing between every point of every
grid, as before #164, took millions of pieces for a system of tens of
components over years, and many times the cost of the curves themselves.
"""

from typing import Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np

from ._point_availability import curve_breaks, curve_grids

#: Steps of the finest grid still changing that a piece spans at most: its
#: 12 quadrature points (on it and on its halves) are then closer together
#: than the narrowest bump a curve linear on that grid can have (two
#: steps), so that the halving sees every one.
PIECE_STEPS = 8
#: How closely a piece's quadrature must agree with the sum on its halves,
#: per unit of time, relative to the integrand's scale (see ``refined``).
TOLERANCE = 1e-8
#: The most halvings of a piece: to about 1e-12 of the window, where the
#: quadrature's differences are rounding.
MAX_ROUNDS = 40
#: Pieces whose integrands are worked out at once.
BLOCK = 25_000

GAUSS, WEIGHTS = np.polynomial.legendre.leggauss(4)


def _lagrange() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the Lagrange polynomials ``L_g`` through the Gauss points on
    ``[-1, 1]``: ``L_g(-1)``, ``L_g(1)``, and ``w_q L_g'(u_q)`` (row ``g``,
    column ``q``), with which ``stieltjes`` weighs a measure's mass in a
    piece by where the points fall."""
    count = len(GAUSS)
    start, end = np.empty(count), np.empty(count)
    slopes = np.empty((count, count))
    for g in range(count):
        others = np.delete(GAUSS, g)
        polynomial = np.poly1d(others, r=True) / np.prod(GAUSS[g] - others)
        start[g], end[g] = polynomial(-1.0), polynomial(1.0)
        slopes[g] = WEIGHTS * polynomial.deriv()(GAUSS)
    return start, end, slopes


_AT_START, _AT_END, _SLOPES = _lagrange()


class TooMany(Exception):
    """An integral would take more than its limit of pieces: ``count``."""

    def __init__(self, count: int):
        super().__init__(count)
        self.count = count


def pieces(
    curves, fixed: np.ndarray, reach: float, limit: int
) -> Tuple[np.ndarray, np.ndarray]:
    """The edges of the pieces to integrate the ``curves`` over from 0 to
    ``reach``: 0, ``reach``, the ``fixed`` times (where the integral is
    wanted, say) and the curves' breaks, with each gap between them cut
    into equal pieces of at most ``PIECE_STEPS`` steps of the finest of
    the curves' grids that still changes there (one piece, if none does);
    and that step, for each piece (inf where there is none). Raise
    ``TooMany`` if that is more than ``limit`` pieces."""
    parts = [np.array([0.0, reach]), np.asarray(fixed, dtype=float).ravel()]
    grids: List[Tuple[float, float]] = []
    for curve in curves:
        parts.append(curve_breaks(curve, 0.0, reach))
        grids += curve_grids(curve)
    usable = [(s, u) for s, u in grids if np.isfinite(s) and s > 0.0]
    # Where a grid stops changing is a break too.
    parts.append(np.array([u for _, u in usable if np.isfinite(u)]))
    edges = np.unique(np.concatenate(parts))
    edges = edges[(edges >= 0.0) & (edges <= reach)]
    if len(edges) < 2:
        return edges, np.empty(0)
    gaps = np.diff(edges)
    finest = np.full(len(gaps), np.inf)
    if usable:
        steps = np.array([s for s, _ in usable])
        until = np.array([u for _, u in usable])
        order = np.argsort(until)
        steps, until = steps[order], until[order]
        # The finest step of the grids that still change after each gap's
        # start: those ending later, a suffix of them by their ends.
        suffix = np.minimum.accumulate(steps[::-1])[::-1]
        first = np.searchsorted(until, edges[:-1], side="right")
        inside = first < len(steps)
        finest[inside] = suffix[first[inside]]
    with np.errstate(divide="ignore", invalid="ignore"):
        count = np.ceil(gaps / (PIECE_STEPS * finest))
    count = np.where(np.isfinite(count) & (count > 1.0), count, 1.0)
    total = float(count.sum())
    if total > limit:
        raise TooMany(int(total))
    count = count.astype(int)
    width = np.repeat(gaps / count, count)
    offset = np.arange(int(total)) - np.repeat(np.cumsum(count) - count, count)
    cut = np.repeat(edges[:-1], count) + offset * width
    return np.append(cut, edges[-1]), np.repeat(finest, count)


def points(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The Gauss points of the pieces from ``a`` to ``b`` (4 a piece, in
    order) and the pieces' half lengths."""
    middle, half = 0.5 * (a + b), 0.5 * (b - a)
    return (middle[:, None] + half[:, None] * GAUSS).ravel(), half


def summed(values: np.ndarray, half: np.ndarray) -> np.ndarray:
    """The quadrature of ``values`` (at ``points``) on each piece."""
    return (np.reshape(values, (-1, len(GAUSS))) @ WEIGHTS) * half


def stieltjes(
    weights: np.ndarray,
    at_start: np.ndarray,
    at_end: np.ndarray,
    at_points: np.ndarray,
) -> np.ndarray:
    """The integral over each piece of a smooth function against a measure
    ``dM``: ``weights`` the function at the Gauss points (4 a piece), and
    ``M`` at the pieces' starts and ends and at the points. The function is
    taken as the cubic through its values there, so that the integral is
    ``sum_g f(s_g) integral of l_g dM``, ``l_g`` the Lagrange polynomials
    through the points, and by parts each of those is ``l_g(b) M(b) -
    l_g(a) M(a) - integral of l_g' M``, the last by the same quadrature:
    exact for a cubic function and a measure whose ``M`` is of degree 5 at
    most. The weights add up to ``M(b) - M(a)``, so a constant function
    takes the measure's mass in the piece exactly."""
    at_points = np.reshape(at_points, (-1, len(GAUSS)))
    share = (
        at_end[:, None] * _AT_END
        - at_start[:, None] * _AT_START
        - at_points @ _SLOPES.T
    )
    return np.sum(np.reshape(weights, share.shape) * share, axis=1)


def in_blocks(estimate: Callable, a: np.ndarray, b: np.ndarray) -> dict:
    """``estimate`` on the pieces from ``a`` to ``b``, ``BLOCK`` at a
    time, its values by key (a key missing from a block is 0 there)."""
    if len(a) <= BLOCK:
        return estimate(a, b)
    blocks = [
        estimate(a[k : k + BLOCK], b[k : k + BLOCK])  # noqa: E203
        for k in range(0, len(a), BLOCK)
    ]
    keys = {key for block in blocks for key in block}
    return {
        key: np.concatenate(
            [
                block.get(key, np.zeros(len(a[k : k + BLOCK])))  # noqa: E203
                for k, block in zip(range(0, len(a), BLOCK), blocks)
            ]
        )
        for key in keys
    }


def refined(
    estimate: Callable[[np.ndarray, np.ndarray], Dict[Hashable, np.ndarray]],
    edges: np.ndarray,
    finest: np.ndarray,
    limit: int,
    relative=(),
) -> Tuple[np.ndarray, Dict[Hashable, np.ndarray]]:
    """The pieces between ``edges`` (from ``pieces``, with the ``finest``
    grid step in each) halved until ``estimate`` (the integrals of some
    quantities on each of the pieces from ``a`` to ``b``, by key) on each
    agrees with the sum on its halves, to ``TOLERANCE`` times its length,
    for every key: in the integrand's own units (a probability); or for the
    keys in ``relative`` (expected events), relative to the piece's own
    integral (or a millionth of their mean over all the pieces, where it is
    next to 0), and only until the piece is that step long. Returns
    the edges of the pieces kept and the integrals on them, by key: the sums
    on their halves, the closer. Raise ``TooMany`` if that is more than
    ``limit`` pieces."""
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 2:
        return edges, {}
    a, b = edges[:-1], edges[1:]
    finest = np.asarray(finest, dtype=float)
    whole = in_blocks(estimate, a, b)
    span = float(edges[-1] - edges[0])
    scale = {}
    for key, values in whole.items():
        mean = float(np.abs(values).sum()) / span if span > 0.0 else 0.0
        scale[key] = max(mean, 1e-300) if key in relative else 1.0
    shortest = span * 2.0**-MAX_ROUNDS
    kept_a: List[np.ndarray] = []
    kept: List[Dict[Hashable, np.ndarray]] = []
    done_count = 0
    for _ in range(MAX_ROUNDS):
        if not len(a):
            break
        n = len(a)
        mid = 0.5 * (a + b)
        both = in_blocks(
            estimate, np.concatenate([a, mid]), np.concatenate([mid, b])
        )
        keys = set(whole) | set(both)
        length = b - a
        # The counts are no closer than the grid within a step of it.
        coarse = length > finest
        failing = np.zeros(n, dtype=bool)
        halves = {}
        for key in keys:
            pair = both.get(key, np.zeros(2 * n))
            halves[key] = (pair[:n], pair[n:])
            together = pair[:n] + pair[n:]
            difference = np.abs(together - whole.get(key, np.zeros(n)))
            if key in relative:
                # Relative to the piece's own count (to a millionth of the
                # mean's, where it has next to none), so that a window's
                # count is as close as the whole's.
                floor = 1e-6 * scale.get(key, 1.0) * length
                bound = TOLERANCE * np.maximum(np.abs(together), floor)
                failing |= (difference > bound) & coarse
            else:
                failing |= difference > TOLERANCE * length
        settled = ~failing | (length <= shortest)
        kept_a.append(a[settled])
        kept.append(
            {
                key: (left + right)[settled]
                for key, (left, right) in halves.items()
            }
        )
        done_count += int(settled.sum())
        split = ~settled
        if done_count + 2 * int(split.sum()) > limit:
            raise TooMany(done_count + 2 * int(split.sum()))
        a = np.concatenate([a[split], mid[split]])
        b = np.concatenate([mid[split], b[split]])
        finest = np.concatenate([finest[split], finest[split]])
        whole = {
            key: np.concatenate([left[split], right[split]])
            for key, (left, right) in halves.items()
        }
    if len(a):  # out of rounds: the finest there is
        kept_a.append(a)
        kept.append(whole)
    starts = np.concatenate(kept_a)
    order = np.argsort(starts, kind="stable")
    keys = {key for part in kept for key in part}
    values = {
        key: np.concatenate(
            [part.get(key, np.zeros(len(s))) for part, s in zip(kept, kept_a)]
        )[order]
        for key in keys
    }
    return np.append(starts[order], edges[-1]), values


def running(values: Optional[np.ndarray], count: int) -> np.ndarray:
    """The running total of ``values`` (one per piece; none: 0) from 0, at
    each of the ``count + 1`` edges."""
    if values is None:
        values = np.zeros(count)
    return np.concatenate([[0.0], np.cumsum(values)])
