"""A lifetime's mean from its survival function, by quadrature.

The mean of a lifetime ``T`` is the area under its survival function:
``E[T] = integral from 0 to infinity of R(t) dt``. An RBD's reliability is
evaluated exactly (or as exactly as its nodes' own reliabilities), so
integrating it gives the system's mean time to failure without simulating
(#122).

``mean_lifetime`` splits ``[0, inf)`` into pieces, a few a decade (#229):
one up to where the survival function first falls measurably below 1, then
at the node models' quantiles (``model_knots``), thinned to ``_PER_DECADE``
a decade, and at every kink (``model_kinks``: the grid points of numerical
curves, where each model's support starts), which a smooth rule cannot
cross. The integral ends where the survival function has fallen below
``_ENDED`` and what lies beyond is negligible, which a heavy tail can put
far out; a tail that falls too slowly for that before ``_FAR`` has an
infinite mean. Each piece is integrated by the Gauss-Kronrod (7, 15) rule,
whose seven Gauss points within it bound its error at no extra cost, and
halved until the whole is accurate to ``RTOL``, relative: a few hundred
points of a smooth curve, where a survival function as costly as a large
network's made thousands too many.
"""

import warnings
from typing import Callable, List, Set

import numpy as np

from repyability.utils.wrappers import outside_level

from ._model_utils import MODEL_ERRORS
from ._point_availability import knots as quantile_knots

#: The Gauss-Kronrod (7, 15) rule on [-1, 1] (QUADPACK's): its 15 points,
#: their weights, and the weights of the 7-point Gauss rule on the same
#: points (0 at the eight Kronrod points), whose estimate the 15-point
#: rule's is checked against.
_KRONROD = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
    ]
)
_GK_X = np.concatenate([-_KRONROD, [0.0], _KRONROD[::-1]])
_GK_W = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_GK_W = np.concatenate([_GK_W, _GK_W[-2::-1]])
_G7_W = np.zeros(15)
for _point, _weight in (
    (1, 0.129484966168869693270611432679082),
    (3, 0.279705391489276667901467771423780),
    (5, 0.381830050505118944950369775488975),
):
    _G7_W[_point] = _G7_W[14 - _point] = _weight
_G7_W[7] = 0.417959183673469387755102040816327
#: The relative accuracy the integral is refined to.
RTOL = 1e-10
#: The most pieces the refinement may split the integral into.
_MAX_PIECES = 1 << 17
#: A survival probability at or below this has ended: the integral stops
#: there. Above it at an enormous time (``_FAR``), some lifetimes never end.
_ENDED = 1e-14
_FAR = 1e300
#: A survival probability within this of 1 has barely started to fall:
#: the integral takes one piece up to the last time it is (halved, as any
#: piece, until accurate).
_FLAT = 1e-6
#: The doublings past the last knot worked out at a time.
_DOUBLINGS = 8
#: Pieces per decade the integral starts from, between where the survival
#: function starts to fall and where it ends, and the most knots it takes
#: a decade.
_PER_DECADE = 2
#: The most knots taken from one numerical curve's grid: a linear
#: interpolation's kinks are slight, and its full grid would make the
#: integral slow for no gain.
_GRID_KNOTS = 1024


def mean_lifetime(sf: Callable, knots, kinks=()) -> float:
    """The mean of a lifetime with survival function ``sf``.

    Parameters
    ----------
    sf : callable
        The survival function: takes a 1-d float array of times and returns
        the probabilities of surviving beyond them.
    knots : array_like
        Times that mark the curve's scale (see ``model_knots``): the pieces
        start at some of them, at most ``_PER_DECADE`` a decade. Any may be
        given; without them the pieces are found from ``sf`` alone.
    kinks : array_like, optional
        Times at which the curve may bend sharply (see ``model_kinks``),
        every one of which starts a piece.

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
    knots = _times(knots)
    kinks = _times(kinks)
    given = np.union1d(knots, kinks)
    start = given[-1] if given.size else 1.0
    # Double from the last knot until the survival function has fallen
    # below _ENDED (and on, for its tail's weight, below).
    tail, values = _doubled(f, start)
    ended = np.flatnonzero(values <= _ENDED)
    last = ended[0] if ended.size else tail.size - 1
    end = tail[last]
    low = given[0] if given.size else end * 1e-12
    decades = max(np.log10(end / low), 1.0)
    filler = np.geomspace(low, end, int(np.ceil(decades * _PER_DECADE)) + 1)
    # Up to the last of them where the curve is still 1 (to _FLAT), one
    # piece: it is all but flat, and the pieces start where it falls.
    flat = filler[1.0 - f(filler) <= _FLAT]
    first = float(flat[-1]) if flat.size else 0.0
    edges = np.unique(
        np.concatenate(
            (
                [0.0, first],
                _per_decade(knots[(knots > first) & (knots < end)]),
                kinks[(kinks > first) & (kinks < end)],
                filler[filler > first],
                tail[: last + 1],
            )
        )
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


def _doubled(f: Callable, start: float) -> tuple:
    """``start`` doubled up to ``_FAR``, and ``f`` there: a few at a time,
    until ``f`` is 0, where it stays (a survival function never rises), so
    the rest are taken as 0 rather than worked out; a light tail is 0 a few
    doublings past its end."""
    tail = start * 2.0 ** np.arange(int(np.log2(_FAR / start)) + 1)
    values = np.zeros(tail.size)
    for first in range(0, tail.size, _DOUBLINGS):
        last = min(first + _DOUBLINGS, tail.size)
        values[first:last] = f(tail[first:last])
        if values[last - 1] == 0.0:
            break
    return tail, values


def _times(times) -> np.ndarray:
    """The positive, finite ``times`` (an array or a list of arrays),
    sorted and each once."""
    if isinstance(times, (list, tuple)):
        times = (
            np.concatenate([np.asarray(t, dtype=float).ravel() for t in times])
            if len(times)
            else np.empty(0)
        )
    times = np.asarray(times, dtype=float).ravel()
    return np.unique(times[np.isfinite(times) & (times > 0.0)])


def _per_decade(times: np.ndarray) -> np.ndarray:
    """The first of the sorted positive ``times`` in each ``1 /
    _PER_DECADE`` of a decade."""
    if not times.size:
        return times
    slot = np.floor(np.log10(times) * _PER_DECADE)
    return times[np.concatenate(([True], slot[1:] != slot[:-1]))]


def _pieces(f: Callable, a: np.ndarray, b: np.ndarray):
    """Each piece ``[a, b]``'s integral of ``f`` by the Gauss-Kronrod (7,
    15) rule, and the 7-point Gauss rule's within it."""
    middle, half = 0.5 * (a + b), 0.5 * (b - a)
    x = middle[:, None] + half[:, None] * _GK_X
    y = f(x.ravel()).reshape(x.shape)
    return (y @ _GK_W) * half, (y @ _G7_W) * half


def _integrate(f: Callable, edges: np.ndarray) -> float:
    """The integral of ``f`` from ``edges[0]`` to ``edges[-1]``, each piece
    halved until its Gauss-Kronrod and Gauss estimates agree to ``RTOL``,
    relative: to its own integral, plus its width's share of the whole
    (for pieces near zero)."""
    a, b = edges[:-1], edges[1:]
    span = edges[-1] - edges[0]
    done = 0.0
    while True:
        kronrod, gauss = _pieces(f, a, b)
        estimate = done + float(kronrod.sum())
        error = np.abs(kronrod - gauss)
        allowed = RTOL * (np.abs(kronrod) + abs(estimate) * (b - a) / span)
        middle = 0.5 * (a + b)
        finished = (error <= allowed) | (middle <= a) | (middle >= b)
        done += float(kronrod[finished].sum())
        if finished.all():
            return done
        a, b, middle = a[~finished], b[~finished], middle[~finished]
        if 2 * a.size > _MAX_PIECES:
            warnings.warn(
                "The integral of the survival function did not reach its "
                f"target accuracy ({RTOL:g}, relative) within "
                f"{_MAX_PIECES} pieces; the mean may be less accurate.",
                RuntimeWarning,
                stacklevel=outside_level(),
            )
            return done + float(kronrod[~finished].sum())
        a, b = np.concatenate((a, middle)), np.concatenate((middle, b))


def model_knots(model) -> np.ndarray:
    """The times at which to split the integral of a node model's survival
    function: the quantiles of each distribution within it and the grid
    points of its numerical curves, through nested RBDs and composite
    nodes."""
    hints, kinks = _gathered(model)
    return _times(hints + kinks)


def model_kinks(model) -> np.ndarray:
    """Those of ``model_knots`` at which a node model's survival function
    may bend sharply, every one of which the integral must start a piece
    at: the grid points of its numerical curves, and where each
    distribution's support starts (an offset)."""
    return _times(_gathered(model)[1])


def _gathered(model) -> tuple:
    """A node model's knots: ``(hints, kinks)``, lists of arrays."""
    hints: List[np.ndarray] = []
    kinks: List[np.ndarray] = []
    _collect(model, hints, kinks, set())
    return hints, kinks


def _collect(
    model, hints: List[np.ndarray], kinks: List[np.ndarray], seen: Set[int]
) -> None:
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
        kinks.append(_thinned(model._t))
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
            hints.append(model._knots())
        except MODEL_ERRORS:  # a model whose quantiles cannot be taken
            pass
        kinks.append(model._kinks())
        return
    else:
        quantiles = quantile_knots(model)
        hints.append(quantiles)
        # Its support's start: an offset, below which it never fails.
        kinks.append(quantiles[:1])
        return
    for part in inner:
        _collect(part, hints, kinks, seen)


def _thinned(grid) -> np.ndarray:
    """At most ``_GRID_KNOTS`` of a curve's grid points, evenly spread."""
    grid = np.asarray(grid, dtype=float).ravel()
    return grid[:: max(1, -(-grid.size // _GRID_KNOTS))]
