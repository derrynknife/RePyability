"""A lifetime's mean from its survival function, by quadrature.

The mean of a lifetime ``T`` is the area under its survival function:
``E[T] = integral from 0 to infinity of R(t) dt``. An RBD's reliability is
evaluated exactly (or as exactly as its nodes' own reliabilities), so
integrating it gives the system's mean time to failure without simulating
(#122).

``mean_lifetime`` splits ``[0, inf)`` into pieces at ``knots``: the times at
which the curve changes character, such as quantiles of the node models and
the grid points of numerical curves
(``model_knots`` finds them). The integral ends where the survival function
has fallen below ``_ENDED`` and what lies beyond is negligible, which a
heavy tail can put far out; a tail that falls too slowly for that before
``_FAR`` has an infinite mean. Each piece is integrated by Gauss-Legendre
quadrature, and halved while its two halves' estimates disagree, until the
whole is accurate to ``RTOL``, relative.
"""

import warnings
from typing import Callable, List, Set

import numpy as np

from ._point_availability import knots as quantile_knots

#: Gauss-Legendre points and weights on [-1, 1], for each piece.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
#: The relative accuracy the integral is refined to.
RTOL = 1e-10
#: The most pieces the refinement may split the integral into.
_MAX_PIECES = 1 << 17
#: A survival probability at or below this has ended: the integral stops
#: there. Above it at an enormous time (``_FAR``), some lifetimes never end.
_ENDED = 1e-14
_FAR = 1e300
#: Filler points per decade between the smallest knot and the end, so that
#: no piece spans many orders of magnitude.
_PER_DECADE = 8
#: The most knots taken from one numerical curve's grid: a linear
#: interpolation's kinks are slight, and its full grid would make the
#: integral slow for no gain.
_GRID_KNOTS = 1024


def mean_lifetime(sf: Callable, knots) -> float:
    """The mean of a lifetime with survival function ``sf``.

    Parameters
    ----------
    sf : callable
        The survival function: takes a 1-d float array of times and returns
        the probabilities of surviving beyond them.
    knots : array_like
        Times at which to split the integral (see ``model_knots``). Any may
        be given; without them the pieces are found from ``sf`` alone.

    Returns
    -------
    float
        The mean lifetime: ``inf`` if some lifetimes never end (``sf`` stays
        above ``_ENDED`` for ever).

    Raises
    ------
    ValueError
        If ``sf`` gives a value that is not a number.
    """

    def f(t: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            values = np.asarray(sf(t), dtype=float).ravel()
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "The survival function is not a number at some times, so "
                "its mean cannot be found."
            )
        return values

    far = f(np.array([_FAR]))[0]
    if far > _ENDED:
        return float("inf")
    knots = np.asarray(knots, dtype=float).ravel()
    knots = np.unique(knots[np.isfinite(knots) & (knots > 0.0)])
    start = knots[-1] if knots.size else 1.0
    # Double from the last knot until the survival function has fallen
    # below _ENDED.
    tail = start * 2.0 ** np.arange(int(np.log2(_FAR / start)) + 1)
    values = f(tail)
    ended = np.flatnonzero(values <= _ENDED)
    last = ended[0] if ended.size else tail.size - 1
    end = tail[last]
    low = knots[0] if knots.size else end * 1e-12
    decades = max(np.log10(end / low), 1.0)
    filler = np.geomspace(low, end, int(np.ceil(decades * _PER_DECADE)) + 1)
    edges = np.unique(
        np.concatenate(([0.0], knots[knots < end], filler, tail[: last + 1]))
    )
    mean = _integrate(f, edges)
    if not mean > 0.0:
        return mean
    # A heavy tail adds to the mean beyond: go on doubling until what is
    # left is negligible (each doubling adds at most its start times its
    # survival probability there). If it never is, before _FAR, the tail
    # falls too slowly for the mean to be finite.
    rest = np.cumsum((tail * values)[::-1])[::-1]
    small = np.flatnonzero(rest[last:] <= 0.1 * RTOL * mean)
    if not small.size:
        return float("inf")
    beyond = last + small[0]
    if beyond > last:
        mean += _integrate(f, tail[last : beyond + 1])
    return mean


def _pieces(f: Callable, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each piece ``[a, b]``'s integral of ``f``, by Gauss-Legendre."""
    middle, half = 0.5 * (a + b), 0.5 * (b - a)
    x = middle[:, None] + half[:, None] * _GL_X
    return (f(x.ravel()).reshape(x.shape) @ _GL_W) * half


def _integrate(f: Callable, edges: np.ndarray) -> float:
    """The integral of ``f`` from ``edges[0]`` to ``edges[-1]``, refined
    until accurate to ``RTOL``, relative."""
    a, b = edges[:-1], edges[1:]
    span = edges[-1] - edges[0]
    whole = _pieces(f, a, b)
    done = 0.0
    while True:
        middle = 0.5 * (a + b)
        left, right = _pieces(f, a, middle), _pieces(f, middle, b)
        halves = left + right
        error = np.abs(halves - whole)
        estimate = done + float(halves.sum())
        # Each piece's share of the tolerance: relative to its own integral,
        # plus its width's share of the whole (for pieces near zero).
        allowed = RTOL * (np.abs(halves) + abs(estimate) * (b - a) / span)
        finished = (error <= allowed) | (middle <= a) | (middle >= b)
        done += float(halves[finished].sum())
        if finished.all():
            return done
        a, b, middle = a[~finished], b[~finished], middle[~finished]
        left, right = left[~finished], right[~finished]
        if 2 * a.size > _MAX_PIECES:
            warnings.warn(
                "The integral of the survival function did not reach its "
                f"target accuracy ({RTOL:g}, relative) within "
                f"{_MAX_PIECES} pieces; the mean may be less accurate.",
                RuntimeWarning,
                stacklevel=3,
            )
            return done + float((left + right).sum())
        a, b = np.concatenate((a, middle)), np.concatenate((middle, b))
        whole = np.concatenate((left, right))


def model_knots(model) -> np.ndarray:
    """The times at which to split the integral of a node model's survival
    function: the quantiles of each distribution within it and the grid
    points of its numerical curves, through nested RBDs and composite
    nodes."""
    found: List[np.ndarray] = []
    _collect(model, found, set())
    if not found:
        return np.empty(0)
    times = np.concatenate([np.asarray(k, dtype=float).ravel() for k in found])
    return np.unique(times[np.isfinite(times) & (times > 0.0)])


def _collect(model, found: List[np.ndarray], seen: Set[int]) -> None:
    from .load_sharing_node import LoadSharingModel
    from .non_repairable_rbd import NonRepairableRBD
    from .numerical_convolution import ConvolvedSurvival
    from .regression_node import RegressionNode
    from .repeated_node import RepeatedNode
    from .repeated_standby_node import RepeatedStandbyNode
    from .standby_node import StandbyModel

    if model is None or isinstance(model, (str, bytes)) or id(model) in seen:
        return
    seen.add(id(model))
    inner: list = []
    if isinstance(model, ConvolvedSurvival):
        found.append(_thinned(model._t))
        return
    if isinstance(model, NonRepairableRBD):
        inner = list(model.reliabilities.values())
    elif isinstance(model, StandbyModel):
        inner = [*model.reliabilities, model._sf_model]
    elif isinstance(model, LoadSharingModel):
        inner = [*model.models, model._sf_model]
    elif isinstance(model, (RepeatedNode, RepeatedStandbyNode)):
        inner = [model.model, getattr(model, "_sf_model", None)]
    elif isinstance(model, RegressionNode):
        try:
            found.append(_thinned(model._survival_grid()[0]))
        except Exception:  # a model whose grid cannot be built
            pass
        return
    else:
        found.append(quantile_knots(model))
        return
    for part in inner:
        _collect(part, found, seen)


def _thinned(grid) -> np.ndarray:
    """At most ``_GRID_KNOTS`` of a curve's grid points, evenly spread."""
    grid = np.asarray(grid, dtype=float).ravel()
    return grid[:: max(1, -(-grid.size // _GRID_KNOTS))]
