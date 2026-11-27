"""The long-run values of a warm standby pair with general lives and
repairs and one repair at a time: ``units=2``, ``k=1``, a spare that
ages at ``dormancy_factor`` ``d`` (above 0, to 1 for a hot spare) while
it waits and switches in surely, and ``repair_crews`` 1 among the pair's
units. (A spare that does not age has a plain cycle: ``_standby_cycle``.)

**The process.** Look at the pair each time a unit starts its repair,
which leaves the other operating, and call ``x`` the life the operating
unit has used by then (its exposure: its time operating, and ``d`` times
its time waiting). What follows depends on ``x`` alone, so the pair is a
Markov renewal process on it. With ``U`` the operating unit's whole life
(``F``, given that it is more than ``x``), ``R`` the repair (``G``) and
``L`` the repaired unit's new life (``F``):

- ``R >= U - x``: the operating unit fails first, and the pair is down
  from then until ``R`` while the one crew finishes. The repaired unit
  then operates, as new, and the other starts its repair: ``x = 0``.
- ``R < U - x``: from ``R`` the repaired unit waits, using its life at
  ``d``. If it fails in standby before the operating unit does (``L / d
  < U - x - R``) it starts a repair at once, with the operating unit at
  ``x + R + L / d``. Otherwise the operating unit fails at ``U - x``,
  the spare is switched in having used ``d (U - x - R)`` of its life,
  and the failed one starts its repair: the pair is up throughout.

The pair regenerates at ``x = 0``. For a quantity summed along the way
(the units' failures, the down time, the system's failures, the time),
let ``T(x)`` be its expected sum from ``x`` until the pair is next at 0,
and ``W(x) = (1 - F(x)) T(x)``. Then, with ``U``, ``R`` and ``L``
independent (``U`` now with no condition),

    ``W(x) = b(x) + E[W(x + R + L / d)] + E[W(d (U - R - x)); U - R > x]``

where ``b`` is ``1 - F(x)`` times what one step from ``x`` adds: for the
units' failures ``1 - F(x)``; for the system's ``P(x < U <= x + R)``;
for the down time ``E[(R - (U - x))+; U > x]``; and for the time that
down time and the time up, ``E[min(U - x, R + L / d); U > x]``. By the
renewal reward theorem the long-run values are the ratios of the
``W(0)``. With exponential lives and repairs this is the group's Markov
chain with one crew (``_standby_chain``), which checks it.

**On a grid.** ``x`` is followed on a uniform grid out to where a life
is all but surely over (``TAIL``).

- ``b`` is worked out at each point by quadrature over the repair,
  ``int (1 - G(s)) phi(x + s) ds`` with ``phi`` the life's density (the
  system's failures), ``F(x + s) - F(x)`` (the down time) and ``1 - F(x
  + s)`` (the time up before ``R``), on pieces between the repair's
  quantiles, steps of the life's spread and a geometric run-in to 0: a
  repair far shorter than a step of the grid costs nothing. The time up
  after ``R`` is the trapezoid rule on the grid. The grids are nested,
  so a finer one works out only the points between the coarser's.
- The two expectations take ``U``, ``R`` and ``L / d`` each as a measure
  on the grid's points that keeps every cell's mass and mean
  (``_point_availability``), a tie ``U - R = x`` counted half, and ``W``
  between points as linear: sums over the grid, by fast Fourier
  transforms.
- ``W(0)`` is a sum over the points the pair visits before it next
  regenerates, weighted by how often: that measure is found by stepping
  it until it no longer changes, the pair put back at 0 where it
  regenerates. (Summing ``b``, a step of it, and so on reaches the same
  values, but takes as many steps as the pair's are between
  regenerations, which is millions where repairs are short.)

The error falls as the square of the step for smooth models, so the
grid is halved until two solutions agree to ``RTOL``, each extrapolated
from the last three at the order their differences show
(``_standby_both``).
"""

import warnings
import weakref
from typing import Optional, Tuple

import numpy as np

from repyability.rbd import _point_availability as _grid
from repyability.rbd import _standby_both
from repyability.rbd._standby_chain import StandbyLongRun
from repyability.utils.wrappers import outside_level

#: The relative agreement of two solutions that ends the halving.
RTOL = 1e-9
#: The probability of a life longer than the grid, and of a repair
#: longer than the quadrature's range.
TAIL = 1e-12
#: The grid's steps to begin with, and at most.
FIRST_STEPS = 2**7
MAX_STEPS = 2**15
#: The quadrature of ``b``: the Gauss-Legendre points of each piece; the
#: repair's quantiles the pieces are cut at, with ``1 - 10 ** -k`` beyond
#: them; the steps of the life's interquartile range; the pieces of that
#: length at most; and the halvings of the first towards 0.
POINTS = 12
_QUANTILES = (1e-9, 1e-6, 1e-4, 1e-3, 0.01, 0.05, *np.linspace(0.1, 0.9, 9))
_DECADES = 13
SPREAD_STEPS = 4
MAX_SPREAD = 256
RUN_IN = 50
#: The model values worked out in one call, at most.
BATCH = 2**20
#: The measure's steps: until what is left of its changes to come is
#: within ``STEP_ATOL`` (of a mass of 1), judged over ``_RATIOS`` steps,
#: or its changes have been no smaller for ``STALLED`` steps and are
#: within ``ROUNDING`` (the transforms' rounding, on a large grid); and
#: at most.
STEP_ATOL = 1e-14
_RATIOS = 4
STALLED = 50
ROUNDING = 1e-10
MAX_STEPS_TAKEN = 20_000

_GL_X, _GL_W = np.polynomial.legendre.leggauss(POINTS)

#: The values of each life and repair worked out, kept while the models
#: live (as ``_standby_both`` keeps its own).
_KEPT: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _of(model, name: str, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.asarray(getattr(model, name)(x.ravel()), dtype=float).reshape(
        x.shape
    )


def _edges(life, repair, end: float) -> np.ndarray:
    """The ends of the quadrature's pieces over the repair's time (see
    the module's notes), from 0 to where a repair is all but surely
    over."""
    tail = 1.0 - 10.0 ** -np.arange(2, _DECADES + 1)
    with np.errstate(all="ignore"):
        cuts = _of(repair, "qf", np.concatenate([_QUANTILES, tail]))
        low, high = _of(life, "qf", np.array([0.25, 0.75]))
    cuts = cuts[np.isfinite(cuts) & (cuts > 0.0)]
    top = float(cuts.max())
    spread = (high - low) / SPREAD_STEPS
    reach = min(end, top)
    if not np.isfinite(spread) or spread <= 0.0:
        spread = reach
    spread = max(spread, reach / MAX_SPREAD)
    steps = spread * np.arange(1, int(np.ceil(reach / spread)) + 1)
    run_in = spread * 0.5 ** np.arange(1, RUN_IN + 1)
    edges = np.unique(np.concatenate([[0.0], cuts, steps, run_in]))
    return edges[edges <= top]


class _Quadrature:
    """The points and weights of the quadrature of ``int (1 - G(s)) ...
    ds`` over the repair's time."""

    def __init__(self, life, repair, end: float):
        self.life = life
        edges = _edges(life, repair, end)
        middle = 0.5 * (edges[:-1] + edges[1:])
        half = 0.5 * (edges[1:] - edges[:-1])
        self.s = (middle[:, None] + half[:, None] * _GL_X).ravel()
        self.w = (half[:, None] * _GL_W).ravel() * _of(repair, "sf", self.s)

    def at(self, x: np.ndarray) -> np.ndarray:
        """At each ``x``, a row each: ``1 - F(x)``; ``int (1 - G(s))
        (F(x + s) - F(x)) ds``; ``int (1 - G(s)) f(x + s) ds``; and ``int
        (1 - G(s)) (1 - F(x + s)) ds``."""
        life = self.life
        below, above = _of(life, "ff", x), _of(life, "sf", x)
        out = np.zeros((4, len(x)))
        out[0] = above
        small = (below < 0.5)[:, None]
        chunk = max(POINTS, BATCH // max(len(x), 1))
        with np.errstate(all="ignore"):
            for first in range(0, len(self.s), chunk):
                s = self.s[first : first + chunk]
                w = self.w[first : first + chunk]
                at = x[:, None] + s[None, :]
                later = _of(life, "sf", at)
                # F(x + s) - F(x), from whichever of F and 1 - F is the
                # smaller at x.
                change = np.where(
                    small,
                    _of(life, "ff", at) - below[:, None],
                    above[:, None] - later,
                )
                out[1] += np.maximum(change, 0.0) @ w
                out[2] += np.nan_to_num(_of(life, "df", at)) @ w
                out[3] += later @ w
        return out


def _measure(k_step, survival: np.ndarray) -> Optional[np.ndarray]:
    """The measure of the points visited between regenerations, to a
    common factor (of mass 1): stepped with ``k_step`` (its mass
    ``survival`` at each point, what is lost at a step put back at 0)
    until what is left of its changes to come (the last, as the first
    term of a geometric series at the slowest of the last ``_RATIOS``
    steps' rates) is within ``STEP_ATOL``, or its changes have reached
    their rounding (no smaller for ``STALLED`` steps, and within
    ``ROUNDING``); None if it does not settle."""
    measure = np.zeros(len(survival))
    measure[0] = 1.0
    change = least = np.inf
    stalled = 0
    ratios = [np.inf] * _RATIOS
    for _ in range(MAX_STEPS_TAKEN):
        before = measure
        measure = k_step(before)
        measure[0] += 1.0 - float(measure @ survival)
        last, change = change, float(np.abs(measure - before).sum())
        if change == 0.0:
            return measure
        ratios = ratios[1:] + [change / last]
        slowest = max(ratios)
        if slowest < 1.0 and change * slowest / (1.0 - slowest) <= STEP_ATOL:
            return measure
        stalled = 0 if change < least else stalled + 1
        least = min(least, change)
        if stalled >= STALLED and least <= ROUNDING:
            return measure
    return None


def _cubic(at: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic interpolation on a grid of ``size`` points at the positions
    ``at`` (in steps): the first of the four points each takes, and
    their weights (a row each)."""
    first = np.clip(np.floor(at).astype(int) - 1, 0, size - 4)
    u = at - first
    weights = np.stack(
        [
            -(u - 1.0) * (u - 2.0) * (u - 3.0) / 6.0,
            u * (u - 2.0) * (u - 3.0) / 2.0,
            -u * (u - 1.0) * (u - 3.0) / 2.0,
            u * (u - 1.0) * (u - 2.0) / 6.0,
        ]
    )
    return first, weights


class _Grids:
    """The pair on nested grids of its operating unit's exposure, each
    of half the last one's step."""

    def __init__(self, life, repair, dormancy: float):
        self.life, self.repair, self.dormancy = life, repair, dormancy
        self.end = float(_of(life, "qf", np.array([1.0 - TAIL]))[0])
        self.quadrature = _Quadrature(life, repair, self.end)
        self.splits = (_grid.knots(life), _grid.knots(repair))
        self.steps = 0
        # ``b``'s quadratures at the grid's points, and at ``dormancy``
        # times each.
        self.b = self.b_switched = np.empty((4, 0))

    def _refined(self, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``b``'s quadratures at the points ``x`` of a grid: every one,
        or with ``b`` those at the last grid's points, of twice the
        step, only the points between them."""
        if b.shape[1] != (len(x) + 1) // 2:
            return self.quadrature.at(x)
        out = np.empty((4, len(x)))
        out[:, ::2] = b
        out[:, 1::2] = self.quadrature.at(x[1::2])
        return out

    def solution(self, steps: int) -> Optional[np.ndarray]:
        """The units' failures, the down time and the system's failures
        between regenerations, each over the time (see the module's
        notes), on a grid of ``steps`` steps; None where the measure
        does not settle on it."""
        from scipy import fft

        life, repair, d = self.life, self.repair, self.dormancy
        size = steps + 1
        h = self.end / steps
        t = h * np.arange(size)
        self.b = self._refined(self.b, t)
        self.b_switched = self._refined(self.b_switched, d * t)
        length = fft.next_fast_len(2 * size, real=True)

        def transform(x: np.ndarray) -> np.ndarray:
            return fft.rfft(x, length)

        def correlated(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
            """``sum_l a[l] b[i + l]`` for each ``i`` of the grid."""
            return fft.irfft(np.conj(a_hat) * b_hat, length)[:size]

        life_knots, repair_knots = self.splits
        with np.errstate(divide="ignore", invalid="ignore"):
            used = _grid._lattice(
                *_grid._cells(lambda x: _of(life, "ff", x), t, life_knots)
            )
            repaired = _grid._lattice(
                *_grid._cells(lambda x: _of(repair, "ff", x), t, repair_knots)
            )
            waited = _grid._lattice(
                *_grid._cells(
                    lambda x: _of(life, "ff", d * x), t, life_knots / d
                )
            )
        repaired_hat = transform(repaired)
        # R + L / d at each point, and U - R at each point from 0 up.
        later_hat = transform(
            fft.irfft(repaired_hat * transform(waited), length)[:size]
        )
        ahead_hat = transform(
            np.maximum(correlated(repaired_hat, transform(used)), 0.0)
        )
        first, shares = _cubic(d * np.arange(size), size)

        def switched(measure_hat: np.ndarray) -> np.ndarray:
            """How much of the measure a step moves to each ``d j h``
            (a tie, ``j = 0``, half)."""
            out = correlated(measure_hat, ahead_hat)
            out[0] *= 0.5
            return out

        def k_step(measure: np.ndarray) -> np.ndarray:
            """The measure a step on: each point's share moved to ``x +
            R + L / d``, and to ``d (U - R - x)`` where that is above
            0, shared among the grid's points about it."""
            hat = transform(measure)
            out = fft.irfft(hat * later_hat, length)[:size]
            moved = switched(hat)
            for k, share in enumerate(shares):
                out += np.bincount(first + k, moved * share, minlength=size)
            return out

        survival, down, fails, before = self.b
        measure = _measure(k_step, survival)
        if measure is None:
            return None
        # The time up after R, int (1 - F(d s)) P(U - R > x + s) ds: the
        # trapezoid rule on the grid.
        weights = _of(life, "sf", d * t) * h
        weights[0] *= 0.5
        after = correlated(
            transform(weights), transform(np.maximum(survival - fails, 0.0))
        )
        on_grid = np.stack([survival, down, fails, before + down])
        sums = on_grid @ measure
        sums[3] += after @ measure
        # What the step to each ``d j h`` adds there, as itself, for
        # what the grid's points about it made of it.
        survival, down, fails, before = self.b_switched
        between = sum(
            share * on_grid[:, first + k] for k, share in enumerate(shares)
        )
        sums += (
            np.stack([survival, down, fails, before + down]) - between
        ) @ switched(transform(measure))
        return sums[:3] / sums[3]


def long_run(life, repair, dormancy: float) -> StandbyLongRun:
    """The long-run values of a warm standby pair with one repair at a
    time, lives ``life`` and repairs ``repair`` (any models with ``ff``,
    ``sf``, ``qf`` and a density ``df``), whose spare uses its life at
    ``dormancy`` (above 0) while it waits."""
    try:
        return _KEPT[life][repair][dormancy]
    except (KeyError, TypeError):  # not kept yet, or not kept at all
        pass
    values = _worked_out(life, repair, dormancy)
    try:
        kept = _KEPT.setdefault(life, weakref.WeakKeyDictionary())
        kept.setdefault(repair, {})[dormancy] = values
    except TypeError:  # a model that is not weakly referenced
        pass
    return values


def _settled(grids: _Grids) -> Tuple[Optional[np.ndarray], float]:
    return _standby_both.halved(
        grids.solution, FIRST_STEPS, MAX_STEPS, RTOL, twice=True
    )


def _worked_out(life, repair, dormancy: float) -> StandbyLongRun:
    """See ``long_run``."""
    limit, apart = _settled(_Grids(life, repair, dormancy))
    if limit is None:
        raise NotImplementedError(
            "The long-run values of the warm standby pair do not settle "
            f"on a grid of {MAX_STEPS} steps of its operating unit's "
            "life: simulate it with availability() or cost()."
        )
    if apart > RTOL:
        warnings.warn(
            "The long-run values of a warm standby pair with one repair "
            f"at a time did not settle to {RTOL:g} on a grid of "
            f"{MAX_STEPS} steps: the last two solutions agree to "
            f"{apart:.1e} (relative). Check them with a simulation "
            "(availability()).",
            stacklevel=outside_level(),
        )
    units, down, fails = (float(v) for v in limit)
    return StandbyLongRun(
        availability=1.0 - down,
        failure_frequency=fails,
        unit_failure_frequency=units,
        unavailability=down,
    )
