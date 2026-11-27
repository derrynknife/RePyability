"""The long-run values of a warm standby pair with general lives and
repairs whose units are both repaired at once: ``units=2``, ``k=1``, a
spare that ages at ``dormancy_factor`` ``d`` (above 0) while it waits
and switches in surely, and a crew for each unit. (With one repair at a
time: ``_standby_warm``; a spare that does not age: ``_standby_both``.)

**The process.** Look at the pair each time a unit starts its repair
with the other operating, and each time a unit starts to operate after
the pair was down. What follows depends on one number each time, so the
pair is a Markov renewal process on two families of states:

- ``A(x)``: a repair starts, and the operating unit has used ``x`` of
  its life. With ``U`` that unit's life (given that it is more than
  ``x``), ``R`` the repair and ``L`` the repaired unit's new life: if
  ``R < U - x`` the repaired unit waits from ``R``, and either fails in
  standby first (``A`` at ``x + R + L / d``) or is switched in when the
  operating one fails (``A`` at ``d (U - x - R)``), as with one repair
  at a time. If ``R >= U - x`` the operating unit fails with the other
  still ``w = R - (U - x)`` from the end of its repair, and the pair is
  down.
- Down, the other unit ``w`` from the end of its repair as this one's
  (``R'``) starts: the first back operates, as new, after ``min(w,
  R')``, the other then ``v = |w - R'|`` from the end of its own.
- ``B(v)``: a unit starts to operate, new, with the other ``v`` from
  the end of its repair. With ``X`` its life: if ``X <= v`` it fails
  with the other ``w = v - X`` from its end (down, as above); otherwise
  the other is repaired at ``v`` and waits, new, the operating unit at
  ``v``: it fails in standby first (``A`` at ``v + L / d``) or is
  switched in (``A`` at ``d (X - v)``).

A repair's time left is drawn when it starts, as the simulation draws
it, so ``B`` needs no more than ``v``. The pair has no state it returns
to with a positive chance (a spare switched in has always aged a
little), so the long-run values are the sums over the embedded chain's
long-run measure of what a step adds, over that of its time (the
Markov renewal theorem): a unit's failure at each ``A`` and each time
the pair goes down, which is also the system's failure; the down time
``E[min(w, R')] = int_0^w (1 - G)``; and the time, that and ``E[min(U -
x, R + L / d)]`` from ``A``, ``E[min(X, v)]`` from ``B`` and ``E[min(X
- v, L / d)]`` from the wait that follows it. With exponential lives
and repairs this is the group's Markov chain with a crew each
(``_standby_chain``), which checks it.

**On two grids.** ``x`` is followed on the grid ``_standby_warm`` uses,
of the operating unit's life, with the same quadratures and measures;
``v`` and ``w`` on a finer one of the repair's range, whose step is a
whole share of the other's, so a repair far shorter than a step of the
life's grid is still followed.

- From ``A`` to down: ``U - x`` has, summed over the measure, a density
  that is smooth at the scale of the life's grid; the sums over that
  grid give it at its points, and it is read between them (a cubic) at
  the fine grid's. The chance of it is the quadrature's own, ``P(x < U
  <= x + R)``.
- From ``B`` on to ``A``: what is left of the measure once the other
  unit is repaired is put on the life's grid, each point's mass and
  place kept.
- What a step moves on to a switch is the whole of what outlasts the
  repair (see ``_standby_warm``), and a tie between two repairs'
  ends is counted half.

The measure is stepped, and scaled to a mass of 1, until it no longer
changes. The error falls as the square of the step, so both grids are
halved together until two solutions agree to ``RTOL``, each
extrapolated from the last three (``_standby_both.halved``).
"""

import warnings
import weakref
from typing import Optional, Tuple

import numpy as np

from repyability.rbd import _point_availability as _grid
from repyability.rbd import _standby_both, _standby_warm
from repyability.rbd._standby_chain import StandbyLongRun
from repyability.rbd._standby_warm import _of
from repyability.utils.vectors import dot
from repyability.utils.wrappers import outside_level

#: The relative agreement of two solutions that ends the halving.
RTOL = 1e-9
#: The steps of the life's grid to begin with, and at most.
FIRST_STEPS = 2**7
MAX_STEPS = 2**15
#: The steps of the fine grid over a repair's range to begin with, at
#: least, and its steps at most on the finest of the life's grids.
FINE_FIRST = 2**8
FINE_MOST = 2**22

_KEPT: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _stationary(k_step, weights: np.ndarray) -> Optional[np.ndarray]:
    """The long-run measure of the embedded chain, of mass 1 (``weights``
    give a measure's mass): ``k_step`` taken, and the measure scaled
    back, until what is left of its changes to come is within
    ``_standby_warm.STEP_ATOL`` (see ``_standby_warm._measure``, whose
    rule this is, for a chain put back at one point); None if it does
    not settle."""
    warm = _standby_warm
    measure = np.zeros(len(weights))
    measure[0] = 1.0
    change = least = np.inf
    stalled = 0
    ratios = [np.inf] * warm._RATIOS
    for _ in range(warm.MAX_STEPS_TAKEN):
        before = measure
        measure = k_step(before)
        measure /= dot(measure, weights)
        last, change = change, float(np.abs(measure - before).sum())
        if change == 0.0:
            return measure
        ratios = ratios[1:] + [change / last]
        slowest = max(ratios)
        if slowest < 1.0 and change * slowest / (1.0 - slowest) <= (
            warm.STEP_ATOL
        ):
            return measure
        stalled = 0 if change < least else stalled + 1
        least = min(least, change)
        if stalled >= warm.STALLED and least <= warm.ROUNDING:
            return measure
    return None


def _before(sf, t: np.ndarray, splits) -> np.ndarray:
    """``int_0^t sf`` at each point of the grid ``t``."""
    out = np.zeros(len(t))
    np.cumsum(_grid._cell_integrals(sf, t, splits), out=out[1:])
    return out


class _Grids:
    """The pair on nested grids: of its operating unit's exposure, and
    of a repair's time left, each of half the last one's step."""

    def __init__(self, life, repair, dormancy: float):
        self.life, self.repair, self.dormancy = life, repair, dormancy
        self.end = float(
            _of(life, "qf", np.array([1.0 - _standby_warm.TAIL]))[0]
        )
        self.reach = float(
            _of(repair, "qf", np.array([1.0 - _standby_warm.TAIL]))[0]
        )
        self.quadrature = _standby_warm._Quadrature(life, repair, self.end)
        self.splits = (_grid.knots(life), _grid.knots(repair))
        # The fine grid's steps in one of the life's grid.
        first = self.end / FIRST_STEPS
        self.shares = 2 ** max(
            0, int(np.ceil(np.log2(FINE_FIRST * first / self.reach)))
        )
        self.b = self.b_switched = np.empty((4, 0))

    def fine_steps(self, steps: int) -> int:
        """The fine grid's steps where the life's grid has ``steps``."""
        fine = self.end / steps / self.shares
        return int(np.ceil(self.reach / fine - 1e-9))

    def _refined(self, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """See ``_standby_warm._Grids._refined``."""
        if b.shape[1] != (len(x) + 1) // 2:
            return self.quadrature.at(x)
        out = np.empty((len(b), len(x)))
        out[:, ::2] = b
        out[:, 1::2] = self.quadrature.at(x[1::2])
        return out

    def solution(self, steps: int) -> Optional[np.ndarray]:
        """The units' failures, the down time and the system's failures
        in the long run, each over the time (see the module's notes),
        on a grid of ``steps`` steps of the life; None where the
        measure does not settle on it."""
        from scipy import fft

        life, repair, d = self.life, self.repair, self.dormancy
        size = steps + 1
        h = self.end / steps
        t = h * np.arange(size)
        self.b = self._refined(self.b, t)
        self.b_switched = self._refined(self.b_switched, d * t)
        survival, _, fails, before = self.b
        length = fft.next_fast_len(2 * size, real=True)

        def transform(x: np.ndarray) -> np.ndarray:
            return fft.rfft(x, length)

        def convolved(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
            return fft.irfft(a_hat * b_hat, length)[:size]

        def correlated(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
            """``sum_l a[l] b[i + l]`` for each ``i`` of the grid."""
            return fft.irfft(np.conj(a_hat) * b_hat, length)[:size]

        def on_grid(cdf, points: np.ndarray, knots: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                return _grid._lattice(*_grid._cells(cdf, points, knots))

        life_knots, repair_knots = self.splits
        used_hat = transform(
            on_grid(lambda x: _of(life, "ff", x), t, life_knots)
        )
        repaired_hat = transform(
            on_grid(lambda x: _of(repair, "ff", x), t, repair_knots)
        )
        waited_hat = transform(
            on_grid(lambda x: _of(life, "ff", d * x), t, life_knots / d)
        )
        # R + L / d at each point, and U - R at each point from 0 up.
        later_hat = transform(convolved(repaired_hat, waited_hat))
        ahead_hat = transform(
            np.maximum(correlated(repaired_hat, used_hat), 0.0)
        )
        first, shares = _standby_warm._cubic(d * np.arange(size), size)
        outlasts = np.maximum(survival - fails, 0.0)

        # The fine grid, of a repair's time left.
        fine = h / self.shares
        m = self.fine_steps(steps)
        v = fine * np.arange(m + 1)
        fine_length = fft.next_fast_len(2 * m + 2, real=True)
        ends_hat = fft.rfft(
            on_grid(lambda x: _of(repair, "ff", x), v, repair_knots),
            fine_length,
        )
        lived_hat = fft.rfft(
            on_grid(lambda x: _of(life, "ff", x), v, life_knots),
            fine_length,
        )
        fine_failed = _of(life, "ff", v)
        # Each fine point's place on the life's grid.
        place = np.arange(m + 1) / self.shares
        inside = place <= steps
        lower = np.floor(place[inside]).astype(int)
        upper = place[inside] - lower
        read_first, read_shares = _standby_warm._cubic(place[inside], size)

        def fine_correlated(a: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
            """``sum_l a[l] b[i + l]`` for each ``i`` of the fine grid."""
            return fft.irfft(
                np.conj(fft.rfft(a, fine_length)) * b_hat, fine_length
            )[: m + 1]

        def waits(left: np.ndarray) -> np.ndarray:
            """The starts whose other unit is repaired first, on the
            life's grid: the operating unit has used its time left.
            Each point's mass is shared among three of the grid's as
            the parabola through them reads a function at its place
            (see ``_standby_warm._three_points``)."""
            out = np.zeros(size + 2)
            for k, share in enumerate(
                (
                    0.5 * (upper - 1.0) * (upper - 2.0),
                    upper * (2.0 - upper),
                    0.5 * upper * (upper - 1.0),
                )
            ):
                out += np.bincount(
                    lower + k, left[inside] * share, minlength=size + 2
                )
            return out[:size]

        def downs(on: np.ndarray, hat: np.ndarray, left: np.ndarray):
            """The measure of the other unit's time left as the pair
            goes down: from a repair's start, and from a start to
            operate."""
            apart = correlated(hat, used_hat)
            sooner = np.zeros(m + 1)
            sooner[inside] = sum(
                share * apart[read_first + k]
                for k, share in enumerate(read_shares)
            )
            sooner[0] *= 0.5
            out = fine_correlated(sooner, ends_hat)
            out[0] *= 0.5
            np.maximum(out, 0.0, out=out)
            total = out.sum()
            if total > 0.0:
                out *= dot(on, fails) / total
            # ``sum_x f[x] left[x + w]``: the operating unit's life
            # ends ``w`` before the other's repair.
            new = fft.irfft(
                np.conj(lived_hat) * fft.rfft(left, fine_length), fine_length
            )[: m + 1]
            new[0] *= 0.5
            np.maximum(new, 0.0, out=new)
            total = new.sum()
            if total > 0.0:
                new *= dot(left, fine_failed) / total
            return out + new

        def switches(measure: np.ndarray, hat: np.ndarray, ahead, lasting):
            """How much of a measure a step moves to each ``d j h``:
            the whole of what outlasts (``lasting``), at 0 what the
            points above it leave of that."""
            out = correlated(hat, ahead)
            out[0] = dot(measure, lasting) - out[1:].sum()
            return out

        def moves(on: np.ndarray, hat: np.ndarray, left: np.ndarray):
            """What is switched in, to each ``d j h``, and the starts
            whose other unit is repaired first with their transform."""
            waiting = waits(left)
            waiting_hat = transform(waiting)
            moved = switches(on, hat, ahead_hat, outlasts)
            moved += switches(waiting, waiting_hat, used_hat, survival)
            return moved, waiting, waiting_hat

        def k_step(measure: np.ndarray) -> np.ndarray:
            on, left = measure[:size], measure[size:]
            hat = transform(on)
            moved, _, waiting_hat = moves(on, hat, left)
            out = convolved(hat, later_hat)
            out += convolved(waiting_hat, waited_hat)
            for k, share in enumerate(shares):
                out += np.bincount(first + k, moved * share, minlength=size)
            # Down: the first back operates, the other |w - R'| from
            # its end (both at once counted once).
            down = downs(on, hat, left)
            down_hat = fft.rfft(down, fine_length)
            new = fft.irfft(np.conj(down_hat) * ends_hat, fine_length)
            new += fft.irfft(np.conj(ends_hat) * down_hat, fine_length)
            new = new[: m + 1]
            new[0] *= 0.5
            return np.concatenate([out, new])

        weights = np.concatenate([survival, np.ones(m + 1)])
        measure = _stationary(k_step, weights)
        if measure is None:
            return None
        on, left = measure[:size], measure[size:]
        hat = transform(on)
        moved, waiting, _ = moves(on, hat, left)
        down = downs(on, hat, left)
        # The time up after a repair, int (1 - F(d s)) P(U - R > x + s)
        # ds, and after a wait begins, with U for U - R: the trapezoid
        # rule on the grid.
        spare = _of(life, "sf", d * t) * h
        spare[0] *= 0.5
        spare_hat = transform(spare)
        after = correlated(spare_hat, transform(outlasts))
        after_wait = correlated(spare_hat, transform(survival))
        both_down = _before(lambda x: _of(repair, "sf", x), v, repair_knots)
        alone = _before(lambda x: _of(life, "sf", x), v, life_knots)
        failures = down.sum()
        units = dot(on, survival) + failures
        down_time = dot(down, both_down)
        time = (
            dot(on, before + after)
            + dot(waiting, after_wait)
            + dot(left, alone)
            + down_time
        )
        # What the step to each ``d j h`` adds there, as itself, for
        # what the grid's points about it made of it.
        there, there_before = self.b_switched[0], self.b_switched[3]
        between = sum(
            share * np.stack([survival, before])[:, first + k]
            for k, share in enumerate(shares)
        )
        units += dot(there - between[0], moved)
        time += dot(there_before - between[1], moved)
        return np.array([units, down_time, failures]) / time


def long_run(life, repair, dormancy: float) -> StandbyLongRun:
    """The long-run values of a warm standby pair whose units are both
    repaired at once, lives ``life`` and repairs ``repair`` (any models
    with ``ff``, ``sf``, ``qf`` and a density ``df``), whose spare uses
    its life at ``dormancy`` (above 0) while it waits and switches in
    surely."""
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
    grids = _Grids(life, repair, dormancy)
    limit = None
    if grids.fine_steps(MAX_STEPS) <= FINE_MOST:
        limit, apart = _settled(grids)
    if limit is None:
        raise NotImplementedError(
            "The long-run values of the warm standby pair with both units "
            f"repaired at once do not settle on a grid of {MAX_STEPS} "
            "steps of its operating unit's life (its repairs may be far "
            "longer than its lives): simulate it with availability() or "
            "cost()."
        )
    if apart > RTOL:
        warnings.warn(
            "The long-run values of a warm standby pair with both units "
            f"repaired at once did not settle to {RTOL:g} on a grid of "
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
