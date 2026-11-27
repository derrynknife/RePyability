"""The long-run values of a warm standby pair with general lives and
repairs and one repair at a time: ``units=2``, ``k=1``, a spare that
ages at ``dormancy_factor`` ``d`` (above 0, to 1 for a hot spare) while
it waits, and ``repair_crews`` 1 among the pair's units. (A spare that
does not age has a plain cycle: ``_standby_cycle``.) The switch is sure
in what follows; one that can fail is taken up after it.

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

**A switch that can fail** (``switching_probability`` ``p`` below 1).
When the operating unit fails with the spare alive, the switch works
with probability ``p``, as above. If it does not, the pair is down
while the failed unit is repaired (``R'``), and the spare waits on,
still ageing: if it fails in that repair it queues for the crew, and
when the repair ends the repaired unit operates as the spare's repair
starts (``x = 0``). Otherwise a unit starts to operate, new, with the
other waiting, having used some ``y`` of its life: a second family of
states, ``S_y``. From ``S_y``, with ``U`` the operating unit's life and
``L`` the spare's (given that it is more than ``y``):

- ``(L - y) / d < U``: the spare fails first, and its repair starts
  with the operating unit at ``x = (L - y) / d``;
- otherwise at ``U`` the switch works (a repair starts, the spare
  operating at ``x = y + d U``) or fails: down for a repair ``R'``,
  then ``x = 0`` if the spare failed in it, and ``S`` at ``y + d U + d
  R'`` if not.

With ``V(y) = (1 - F(y))`` times the expected sum from ``S_y``, the two
families' equations are coupled:

    ``W(x) = b(x) + E[W(x + R + L / d)] + p E[W(e); e > 0]
    + (1 - p) E[V(e + d R'); e > 0]``, ``e = d (U - R - x)``

    ``V(y) = c(y) + E[W((L - y) / d); L > y] + p E[W(y + d U)]
    + (1 - p) E[V(y + d U + d R')]``

``b`` gains, for each switch that fails, a unit's failure, the system's
and a repair down; ``c`` is those for ``P(L > y + d U)``, and the time
``E[min(U, (L - y) / d); L > y]``. On the grid, ``y`` is followed as
``x`` is, and the measure stepped is of both families' points:

- ``c`` is worked out at each point by quadrature over the operating
  unit's life, as ``b`` is over the repair.
- ``d U`` and ``d R'`` are measures on the grid's points that keep each
  cell's mass, mean and second moment (``_three_points``): either may
  be far shorter than a step, where a measure keeping the mean alone
  costs the step times the function's second difference, an error of
  the first order beside the others' second.
- The spare failing first puts the operating unit at ``(L - y) / d``,
  whose density is that of ``L - y`` at ``d x``: the sums over the grid
  give it at the grid's points, and it is read between them (a cubic).
- What a step moves on from ``x`` to a switch is the whole of ``P(U - R
  > x)``, which the quadrature gives: counting half the ties is right
  where the pair's points are spread, but not for the pair at 0, whose
  operating unit and repair start together. (With a sure switch what a
  tie loses is put back at 0, next to where it belongs.) The switches
  that fail are counted from the same probabilities.

With exponential lives and repairs this is the chain with a switch
that can fail, and with ``p = 1`` the sure switch's arithmetic, to the
bit.
"""

import operator
import warnings
import weakref
from typing import Optional, Tuple

import numpy as np

from repyability.rbd import _point_availability as _grid
from repyability.rbd import _standby_both, _standby_cycle
from repyability.rbd._standby_chain import StandbyLongRun
from repyability.utils.vectors import dot
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


class _Waiting:
    """The quadrature over an operating unit's life ``s`` of what a
    spare that has used ``y`` and goes on waiting makes of it."""

    def __init__(self, life, dormancy: float, end: float):
        self.life, self.dormancy = life, dormancy
        edges = _edges(life, life, end)
        middle = 0.5 * (edges[:-1] + edges[1:])
        half = 0.5 * (edges[1:] - edges[:-1])
        self.s = (middle[:, None] + half[:, None] * _GL_X).ravel()
        w = (half[:, None] * _GL_W).ravel()
        with np.errstate(all="ignore"):
            self.fails = w * np.nan_to_num(_of(life, "df", self.s))
        self.lasts = w * _of(life, "sf", self.s)

    def at(self, y: np.ndarray) -> np.ndarray:
        """At each ``y``, a row each: ``int f(s) (1 - F(y + d s)) ds``
        (the spare outlives the operating unit) and ``int (1 - F(s)) (1
        - F(y + d s)) ds`` (the time both last)."""
        out = np.zeros((2, len(y)))
        chunk = max(POINTS, BATCH // max(len(y), 1))
        for first in range(0, len(self.s), chunk):
            s = self.s[first : first + chunk]
            later = _of(self.life, "sf", y[:, None] + self.dormancy * s)
            out[0] += later @ self.fails[first : first + chunk]
            out[1] += later @ self.lasts[first : first + chunk]
        return out


def _measure(
    k_step, survival: np.ndarray, mass=operator.matmul
) -> Optional[np.ndarray]:
    """The measure of the points visited between regenerations, to a
    common factor (of mass 1): stepped with ``k_step`` (its mass
    ``survival`` at each point, what is lost at a step put back at 0)
    until what is left of its changes to come (the last, as the first
    term of a geometric series at the slowest of the last ``_RATIOS``
    steps' rates) is within ``STEP_ATOL``, or its changes have reached
    their rounding (no smaller for ``STALLED`` steps, and within
    ``ROUNDING``); None if it does not settle. ``mass`` is the product
    that gives a measure's mass."""
    measure = np.zeros(len(survival))
    measure[0] = 1.0
    change = least = np.inf
    stalled = 0
    ratios = [np.inf] * _RATIOS
    for _ in range(MAX_STEPS_TAKEN):
        before = measure
        measure = k_step(before)
        measure[0] += 1.0 - float(mass(measure, survival))
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


def _three_points(cdf, t: np.ndarray, splits) -> np.ndarray:
    """A distribution with cumulative distribution function ``cdf`` as a
    measure on the grid's points that keeps each cell's mass, mean and
    second moment: the cell's mass is shared among its ends and the
    point after (the point before, for the last cell) as the parabola
    through them reads a function there. Times far shorter than a step
    then cost what the function's third difference is, not its
    second."""
    h = t[1] - t[0]
    F = cdf(t)

    def from_start(x):
        return (x - np.floor(x / h) * h) * cdf(x)

    mass = np.diff(F)
    # E[u; cell] and E[u^2; cell], u the place in the cell from 0 to 1,
    # by parts.
    first = F[1:] - _grid._cell_integrals(cdf, t, splits) / h
    # int (x - t0) F dx misplaces a point at the cell's end (its place
    # is read as 0): none of the quadrature's points is one.
    second = F[1:] - 2.0 * _grid._cell_integrals(from_start, t, splits) / h**2
    first = np.clip(first, 0.0, mass)
    second = np.clip(second, 0.0, first)
    out = np.zeros(len(t) + 1)
    out[0] = F[0]
    out[:-2] += 0.5 * (second - 3.0 * first + 2.0 * mass)
    out[1:-1] += 2.0 * first - second
    out[2:] += 0.5 * (second - first)
    # The last cell's share past the grid, read from the parabola
    # through the last three points instead.
    over = out[-1]
    out[-2] += 3.0 * over
    out[-3] -= 3.0 * over
    out[-4] += over
    return out[:-1]


class _Grids:
    """The pair on nested grids of its operating unit's exposure, each
    of half the last one's step."""

    def __init__(self, life, repair, dormancy: float, switching: float = 1.0):
        self.life, self.repair, self.dormancy = life, repair, dormancy
        self.switching = switching
        self.end = float(_of(life, "qf", np.array([1.0 - TAIL]))[0])
        self.quadrature = _Quadrature(life, repair, self.end)
        # With a switch that can fail: a spare's waiting on, at the
        # grid's points, and a repair's mean.
        self.waiting = _Waiting(life, dormancy, self.end)
        self.c = np.empty((2, 0))
        self.repair_mean = (
            _standby_cycle.mean_time(repair) if switching < 1.0 else 0.0
        )
        self.splits = (_grid.knots(life), _grid.knots(repair))
        self.steps = 0
        # ``b``'s quadratures at the grid's points, and at ``dormancy``
        # times each.
        self.b = self.b_switched = np.empty((4, 0))

    def _refined(
        self, b: np.ndarray, x: np.ndarray, quadrature=None
    ) -> np.ndarray:
        """``b``'s quadratures at the points ``x`` of a grid: every one,
        or with ``b`` those at the last grid's points, of twice the
        step, only the points between them."""
        if quadrature is None:
            quadrature = self.quadrature
        if b.shape[1] != (len(x) + 1) // 2:
            return quadrature.at(x)
        out = np.empty((len(b), len(x)))
        out[:, ::2] = b
        out[:, 1::2] = quadrature.at(x[1::2])
        return out

    def solution(self, steps: int) -> Optional[np.ndarray]:
        """The units' failures, the down time and the system's failures
        between regenerations, each over the time (see the module's
        notes), on a grid of ``steps`` steps; None where the measure
        does not settle on it."""
        from scipy import fft

        life, repair, d = self.life, self.repair, self.dormancy
        p = self.switching
        size = steps + 1
        h = self.end / steps
        t = h * np.arange(size)
        self.b = self._refined(self.b, t)
        self.b_switched = self._refined(self.b_switched, d * t)
        length = fft.next_fast_len(2 * size, real=True)

        def transform(x: np.ndarray) -> np.ndarray:
            return fft.rfft(x, length)

        def convolved(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
            """``sum_l a[l] b[i - l]`` for each ``i`` of the grid."""
            return fft.irfft(a_hat * b_hat, length)[:size]

        def correlated(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
            """``sum_l a[l] b[i + l]`` for each ``i`` of the grid."""
            return fft.irfft(np.conj(a_hat) * b_hat, length)[:size]

        def on_grid(model, scale: float, knots: np.ndarray) -> np.ndarray:
            """The measure on the grid's points of ``scale`` times a
            time of ``model``, keeping each cell's mass, mean and
            second moment."""
            with np.errstate(divide="ignore", invalid="ignore"):
                return _three_points(
                    lambda x: _of(model, "ff", x / scale), t, knots * scale
                )

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
        used_hat = transform(used)
        # R + L / d at each point, and U - R at each point from 0 up.
        later_hat = transform(
            fft.irfft(repaired_hat * transform(waited), length)[:size]
        )
        ahead_hat = transform(
            np.maximum(correlated(repaired_hat, used_hat), 0.0)
        )
        first, shares = _cubic(d * np.arange(size), size)

        def switched(measure_hat: np.ndarray) -> np.ndarray:
            """How much of the measure a step moves to each ``d j h``
            (a tie, ``j = 0``, half)."""
            out = correlated(measure_hat, ahead_hat)
            out[0] *= 0.5
            return out

        def shared(out: np.ndarray, moved: np.ndarray) -> None:
            """Add to ``out`` what is ``moved`` to each ``d j h``,
            shared among the grid's points about it."""
            for k, share in enumerate(shares):
                out += np.bincount(first + k, moved * share, minlength=size)

        def k_step(measure: np.ndarray) -> np.ndarray:
            """The measure a step on: each point's share moved to ``x +
            R + L / d``, and to ``d (U - R - x)`` where that is above
            0, shared among the grid's points about it."""
            hat = transform(measure)
            out = fft.irfft(hat * later_hat, length)[:size]
            shared(out, switched(hat))
            return out

        survival, down, fails, before = self.b
        every = survival
        if p < 1.0:
            # A switch that fails (see the module's notes): the measure
            # is of both families' points, the repairs' starts and then
            # the starts with a spare waiting on.
            self.c = self._refined(self.c, t, self.waiting)
            slow_life_hat = transform(on_grid(life, d, life_knots))
            slow_repair_hat = transform(on_grid(repair, d, repair_knots))
            slow_both_hat = transform(
                convolved(slow_life_hat, slow_repair_hat)
            )
            every = np.concatenate([survival, survival])
            # P(U - R > x) at each point, which the quadrature gives,
            # and with the spare still alive then.
            outlasts = np.maximum(survival - fails, 0.0)
            reached = np.maximum(
                outlasts - correlated(transform(waited), transform(outlasts)),
                0.0,
            )

            def switches(measure: np.ndarray, hat: np.ndarray) -> np.ndarray:
                """``switched``, of the whole of what outlasts its
                repair: at 0 what the points above it leave of that
                (half the ties, but for the pair at 0 itself, whose
                operating unit and repair start together)."""
                out = switched(hat)
                out[0] = dot(measure, outlasts) - out[1:].sum()
                return out

            def k_step(measure: np.ndarray) -> np.ndarray:  # noqa: F811
                hat = transform(measure[:size])
                waiting_hat = transform(measure[size:])
                moved = switches(measure[:size], hat)
                out = convolved(hat, later_hat)
                shared(out, p * moved)
                # From a start with a spare waiting: the spare fails
                # first, the operating unit at ``(L - y) / d``, the
                # density of ``L - y`` read between the grid's points;
                # or the operating unit does, and the switch works.
                apart = correlated(waiting_hat, used_hat)
                died = sum(
                    share * apart[first + k] for k, share in enumerate(shares)
                )
                died[0] *= 0.5
                out += d * died
                out += p * convolved(waiting_hat, slow_life_hat)
                # A switch that fails: the spare waits a repair on.
                unswitched = np.zeros(size)
                shared(unswitched, moved)
                waits = convolved(transform(unswitched), slow_repair_hat)
                waits += convolved(waiting_hat, slow_both_hat)
                return np.concatenate([out, (1.0 - p) * waits])

        measure = _measure(k_step, every, operator.matmul if p >= 1.0 else dot)
        if measure is None:
            return None
        waiting, measure = measure[size:], measure[:size]
        # The time up after R, int (1 - F(d s)) P(U - R > x + s) ds: the
        # trapezoid rule on the grid.
        weights = _of(life, "sf", d * t) * h
        weights[0] *= 0.5
        after = correlated(
            transform(weights), transform(np.maximum(survival - fails, 0.0))
        )
        on_points = np.stack([survival, down, fails, before + down])
        sums = on_points @ measure
        sums[3] += after @ measure
        # What the step to each ``d j h`` adds there, as itself, for
        # what the grid's points about it made of it.
        moved = switched(transform(measure))
        if p < 1.0:
            moved = switches(measure, transform(measure))
        survival, down, fails, before = self.b_switched
        between = sum(
            share * on_points[:, first + k] for k, share in enumerate(shares)
        )
        there = np.stack([survival, down, fails, before + down]) - between
        if p < 1.0:
            sums += p * (there @ moved)
            # Each switch that fails: a unit's failure and the
            # system's, and a repair down; and from a start with a
            # spare waiting, the time until either unit fails.
            outlives, both_last = self.c
            failed = (1.0 - p) * (reached @ measure + outlives @ waiting)
            sums += failed * np.array(
                [1.0, self.repair_mean, 1.0, self.repair_mean]
            )
            sums[3] += both_last @ waiting
        else:
            sums += there @ moved
        return sums[:3] / sums[3]


def long_run(
    life, repair, dormancy: float, switching: float = 1.0
) -> StandbyLongRun:
    """The long-run values of a warm standby pair with one repair at a
    time, lives ``life`` and repairs ``repair`` (any models with ``ff``,
    ``sf``, ``qf`` and a density ``df``), whose spare uses its life at
    ``dormancy`` (above 0) while it waits, and whose switch works with
    probability ``switching``."""
    try:
        return _KEPT[life][repair][dormancy, switching]
    except (KeyError, TypeError):  # not kept yet, or not kept at all
        pass
    values = _worked_out(life, repair, dormancy, switching)
    try:
        kept = _KEPT.setdefault(life, weakref.WeakKeyDictionary())
        kept.setdefault(repair, {})[dormancy, switching] = values
    except TypeError:  # a model that is not weakly referenced
        pass
    return values


def _settled(grids: _Grids) -> Tuple[Optional[np.ndarray], float]:
    return _standby_both.halved(
        grids.solution, FIRST_STEPS, MAX_STEPS, RTOL, twice=True
    )


def _worked_out(
    life, repair, dormancy: float, switching: float = 1.0
) -> StandbyLongRun:
    """See ``long_run``."""
    limit, apart = _settled(_Grids(life, repair, dormancy, switching))
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
