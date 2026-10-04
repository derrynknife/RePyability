"""The rate at which a system's availability (or reliability) changes, and
each component's part in it (#195).

With independent components the system's availability is multilinear in
theirs, ``A(t) = h(A_1(t), ..., A_n(t))``, so

``dA/dt = sum_i I_B^i(t) dA_i/dt``

exactly, ``I_B^i`` the Birnbaum importance at the components' availabilities
then: each term is what component ``i`` does to the system, and the terms
add up to the whole. A component's own curve is worked out numerically
(see ``_point_availability``); ``derivative`` takes its rate of change by
differences that keep to its pieces, and the parts the curve keeps off its
grid (its dips: down times that start at a known time, however short)
exactly, each on its own scale (#240).

At a scheduled event (a block replacement or test that takes the unit off
line, a planned outage) a component's availability jumps, and so may the
system's. The jump is split the same way, along the straight path from the
components' values before it to those after: component ``i``'s part is
``dA_i * integral_0^1 I_B^i(A_before + s dA) ds``, which, ``h`` being
multilinear, Gauss-Legendre quadrature with half as many points as there
are components jumping together gives exactly, and which adds up to the
system's jump (``split_jumps``).
"""

from typing import Any, Callable, Dict, Hashable, Tuple

import numpy as np

from ._point_availability import curve_breaks, curve_grids

#: The step of a curve's differences, as a fraction of the times asked
#: for, where it is not linear on a grid.
_RELATIVE_STEP = 1e-6

#: A curve's change at an instant smaller than this is no jump.
JUMP = 1e-12


def derivative(curve, x: np.ndarray, scale: float) -> np.ndarray:
    """``curve``'s rate of change at each time ``x``: after a jump or a
    bend at ``x``, its rate from then on. A curve that knows parts of its
    rate exactly gives it (its ``rate``: the down times it keeps off its
    grid, which may be far shorter than a step, see ``GridCurve``); any
    other is differenced (``differences``)."""
    rate = getattr(curve, "rate", None)
    if rate is not None:
        return rate(np.asarray(x, dtype=float).ravel(), scale)
    return differences(curve, x, scale)


def differences(curve, x: np.ndarray, scale: float) -> np.ndarray:
    """``curve``'s rate of change at each time ``x`` by differences of its
    values (see ``derivative``).

    A curve linear on a grid (see ``curve_grids``) is differenced a grid
    step either side, which is second order between its points as at them;
    another over ``1e-6 * scale``. The differences keep to the curve's
    pieces (``curve_breaks``): central where its next break either side is
    more than a step away, else one-sided, second order, on the side with
    more room, its points short of the next break. A step is never below
    a millionth of the grid's, so that a cluster of breaks just past ``x``
    (the quantiles a down time's quadrature is split at) cannot shrink it
    to rounding."""
    x = np.asarray(x, dtype=float).ravel()
    out = np.zeros(x.shape)
    if not x.size:
        return out
    step = np.full(x.shape, _RELATIVE_STEP * max(scale, 1e-300))
    for grid, until in curve_grids(curve):
        step = np.where(x < until, grid, step)
    reach = float(x.max() + 4.0 * step.max())
    breaks = np.unique(
        np.concatenate([[0.0], curve_breaks(curve, 0.0, reach)])
    )
    after = np.searchsorted(breaks, x, side="right")
    behind = x - breaks[after - 1]
    ahead = np.where(
        after < len(breaks),
        breaks[np.minimum(after, len(breaks) - 1)] - x,
        np.inf,
    )
    floor = 1e-6 * step
    # A difference never reaches the next break, where the curve may have
    # jumped already: its points keep a margin short of it.
    central = (behind >= 1.25 * step) & (ahead >= 1.25 * step)
    forward = ~central & (ahead >= behind)
    backward = ~central & ~forward
    if central.any():
        h = step[central]
        values = curve.at(np.concatenate([x[central] - h, x[central] + h]))
        n = len(h)
        out[central] = (values[n:] - values[:n]) / (2.0 * h)
    for side, room, sign in (
        (forward, ahead, 1.0),
        (backward, behind, -1.0),
    ):
        if not side.any():
            continue
        h = np.maximum(np.minimum(step[side], 0.4 * room[side]), floor[side])
        t = x[side]
        n = len(t)
        values = curve.at(
            np.concatenate([t, t + sign * h, t + 2.0 * sign * h])
        )
        a0, a1, a2 = values[:n], values[n : 2 * n], values[2 * n :]  # noqa
        out[side] = sign * (-3.0 * a0 + 4.0 * a1 - a2) / (2.0 * h)
    return out


def jumps(
    curves: Dict[Hashable, Any], stop: float
) -> Tuple[np.ndarray, Dict[Hashable, np.ndarray], Dict[Hashable, np.ndarray]]:
    """The times in ``(0, stop]`` at which any of ``curves`` jumps, and each
    curve's values just before and at them (each curve's value at a time
    is its value after anything that happens then)."""
    candidates = [np.empty(0)]
    for curve in curves.values():
        candidates.append(curve_breaks(curve, 0.0, stop))
    times = np.unique(np.concatenate(candidates))
    times = times[(times > 0.0) & (times <= stop)]
    before = np.nextafter(times, -np.inf)
    lefts: Dict[Hashable, np.ndarray] = {}
    rights: Dict[Hashable, np.ndarray] = {}
    jumping = np.zeros(times.shape, dtype=bool)
    for node, curve in curves.items():
        if not times.size:
            lefts[node] = rights[node] = np.empty(0)
            continue
        left = np.asarray(curve.at(before), dtype=np.float64)
        right = np.asarray(curve.at(times), dtype=np.float64)
        lefts[node], rights[node] = left, right
        jumping |= np.abs(right - left) > JUMP
    keep = np.flatnonzero(jumping)
    return (
        times[keep],
        {node: values[keep] for node, values in lefts.items()},
        {node: values[keep] for node, values in rights.items()},
    )


def split_jumps(
    importances: Callable[[Dict[Hashable, np.ndarray]], Dict],
    before: Dict[Hashable, np.ndarray],
    after: Dict[Hashable, np.ndarray],
) -> Dict[Hashable, np.ndarray]:
    """Each component's part in the system's jumps (see the module
    docstring): ``importances(values)`` gives every component's Birnbaum
    importance at the components' ``values`` (1-d arrays of one length),
    and ``before`` and ``after`` are the components' values either side
    of each jump."""
    nodes = list(before)
    if not nodes:
        return {}
    count = len(next(iter(before.values())))
    if count == 0:
        return {node: np.empty(0) for node in nodes}
    change = {node: after[node] - before[node] for node in nodes}
    together = np.sum([np.abs(change[node]) > JUMP for node in nodes], axis=0)
    points, weights = np.polynomial.legendre.leggauss(
        max(1, int(np.ceil(together.max() / 2.0)))
    )
    s = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    # Every jump at every point of the path, in one evaluation.
    values = {
        node: (before[node][:, None] + s[None, :] * change[node][:, None])
        .ravel()
        .clip(0.0, 1.0)
        for node in nodes
    }
    importance = importances(values)
    return {
        node: change[node]
        * (
            np.asarray(importance[node], dtype=float).reshape(count, len(s))
            @ weights
        )
        for node in nodes
    }


#: Gauss-Legendre points and weights on [-1, 1], for each piece of an
#: integral over time (see ``integral``).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
#: The relative accuracy ``integral`` refines to.
RTOL = 1e-9
#: The most pieces ``integral`` may split an integral into.
_MAX_PIECES = 1 << 16


def integral(f: Callable[[np.ndarray], np.ndarray], edges) -> np.ndarray:
    """The integrals of the columns of ``f`` (``f(t)``: one row per time)
    from ``edges[0]`` to ``edges[-1]``: each piece between ``edges`` by
    8-point Gauss-Legendre quadrature, halved while its halves' sum
    differs from it, in any column, by more than ``RTOL`` of the piece's
    share of the columns' total."""
    edges = np.unique(np.asarray(edges, dtype=float))
    a, b = edges[:-1], edges[1:]
    if not a.size:
        return np.zeros(np.shape(f(np.zeros(1)))[1])
    span = float(edges[-1] - edges[0])

    def pieces(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        middle, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        t = (middle[:, None] + half[:, None] * _GL_X).ravel()
        values = np.asarray(f(t), dtype=float).reshape(len(lo), len(_GL_X), -1)
        return np.einsum("pgk,g->pk", values, _GL_W) * half[:, None]

    whole = pieces(a, b)
    done = np.zeros(whole.shape[1])
    while True:
        middle = 0.5 * (a + b)
        left, right = pieces(a, middle), pieces(middle, b)
        halves = left + right
        size = float(np.abs(done).sum() + np.abs(halves).sum())
        error = np.abs(halves - whole).max(axis=1)
        allowed = RTOL * (
            np.abs(halves).sum(axis=1) + size * (b - a) / max(span, 1e-300)
        )
        finished = (error <= allowed) | (middle <= a) | (middle >= b)
        done += halves[finished].sum(axis=0)
        if finished.all() or 2 * int((~finished).sum()) > _MAX_PIECES:
            return done + halves[~finished].sum(axis=0)
        keep = ~finished
        a, b, middle = a[keep], b[keep], middle[keep]
        left, right = left[keep], right[keep]
        a, b = np.concatenate((a, middle)), np.concatenate((middle, b))
        whole = np.concatenate((left, right))
