"""The long-run values of a cold standby pair with general lives and
repairs whose units are both repaired at once: ``units=2``, ``k=1``, a
spare that does not age (``dormancy_factor`` 0) and switches in surely,
and a crew for each unit (with one repair at a time the pair has a plain
cycle: ``_standby_cycle``).

**The process.** Look at the pair each time a unit starts to operate
while the other is under repair, and call ``a`` how long that repair has
run (0 when the other has just failed and this one was switched in). The
operating unit is new (a cold spare does not age), so what follows
depends on ``a`` alone: a Markov renewal process on it. With ``X`` the
operating unit's life (``F``) and ``V`` what is left of the other's
repair (``G`` given that it has run ``a``):

- ``V < X``: the repaired unit waits; at ``X`` the operating one fails
  with it ready, it is switched in, and the pair is at ``a = 0``, up
  throughout;
- ``V >= X``: the pair is down from ``X``, both units under repair: the
  old one for ``V - X`` more, the one that just failed for ``R`` (``G``).
  The first back operates, as new: after ``V - X`` with the other ``V -
  X`` into its repair, or after ``R`` with the other ``a + X + R`` into
  its own.

The pair regenerates at ``a = 0``. For a quantity summed along the way
(the time, the down time, the system's failures, the units'), let
``T(a)`` be its expected sum from ``a`` until the pair is next at 0, and
``W(a) = (1 - G(a)) T(a)``. Then, with ``X`` and ``R`` independent,

    ``W(a) = b(a) + E[W(a + X + R)] + E[W(R - X - a); R > X + a]``

where ``b`` is ``1 - G(a)`` times what one step from ``a`` adds: for the
units' failures ``1 - G(a)``; for the system's ``P(R > a + X)``; for the
down time ``E[m(a + X)]``, ``m(u) = int (1 - G(u + s)) (1 - G(s)) ds``
(the shorter of what two repairs have left); and for the time ``E[X]``
times the units' plus the down time's. By the renewal reward theorem the
long-run values are the ratios of the ``W(0)``. With exponential lives
and repairs this is the group's Markov chain with a crew each
(``_standby_chain``), which checks it.

**A switch that can fail** (``switching_probability`` below 1): where a
step ends with the other unit ready (``V < X``, probability ``1 - G(a) -
P(R > a + X)`` of ``1 - G(a)``) and the switch fails, the pair is down
for a repair and then one unit with a waiting spare until a switch works
(``_standby_cycle.switched``), which ``b`` gains; the lives in it add to
the units' failures, and so to the time.

**On a grid.** The equation is solved on a uniform grid of ``a``, out to
where a repair is all but surely over (``TAIL``), with ``X`` and ``R``
each rounded to the grid's nearest point (its probability from the
distribution functions, so no density is needed) and a tie ``R = X + a``
counted half: the expectations are then sums over the grid, worked out
by fast Fourier transforms, and the solution the sum of ``b``, what one
step makes of it, and so on. The grid is halved until two solutions
agree to ``RTOL``, each extrapolated from the last three at the order
their differences show (2 for smooth models; less where a density is
unbounded at 0).
"""

import warnings
import weakref
from typing import List, Optional, Tuple

import numpy as np

from repyability.rbd import _standby_cycle
from repyability.rbd._standby_chain import StandbyLongRun
from repyability.utils.wrappers import outside_level

#: The relative agreement of two solutions that ends the halving.
RTOL = 1e-8
#: The probability of a repair longer than the grid.
TAIL = 1e-12
#: The grid's steps to begin with, and at most.
FIRST_STEPS = 2**8
MAX_STEPS = 2**17
#: How nearly (relative to the sum) a step must only scale the terms for
#: the rest to be summed as a geometric series, and the steps taken at
#: most.
SUM_RTOL = 1e-14
MAX_SUM = 100_000
#: A sum whose terms shrink by less than this a step is left: the grid is
#: too coarse for the lives (each rounds to 0, and the pair never
#: regenerates on it).
SLOWEST = 1e-6

#: The values of each life and repair worked out, kept while the models
#: live: a diagram's long-run analyses ask for them many times over.
_KEPT: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _rounded(model, steps: int, h: float) -> np.ndarray:
    """The probability that a time of ``model`` rounds to each point of
    the grid, ``0, h, ..., steps * h``: point ``l`` takes the times
    between ``(l - 1/2) h`` and ``(l + 1/2) h``, each from whichever of
    the distribution and survival functions is the smaller there."""
    edges = np.concatenate(([0.0], (np.arange(steps + 1) + 0.5) * h))
    below = np.ravel(model.ff(edges))
    above = np.ravel(model.sf(edges))
    return np.where(below[1:] < 0.5, np.diff(below), -np.diff(above))


def _solution(
    life, repair, end: float, steps: int, unswitched: np.ndarray
) -> Optional[np.ndarray]:
    """``W(0)`` for the units' failures, the down time and the system's
    failures (see the module's notes), on a grid of ``steps`` steps to
    ``end``; None where the sum does not settle on it. ``unswitched`` is
    what each gains, on average, where a step ends with the other unit
    ready (a switch that fails, and what follows: none for a sure
    switch)."""
    from scipy import fft

    h = end / steps
    size = steps + 1
    f = _rounded(life, steps, h)
    g = _rounded(repair, steps, h)
    survival = np.ravel(repair.sf(np.arange(size) * h))
    length = fft.next_fast_len(2 * size, real=True)

    def transform(x: np.ndarray) -> np.ndarray:
        return fft.rfft(x, length, axis=-1)

    def correlated(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
        """``sum_l a[l] b[i + l]`` for each ``i`` of the grid."""
        return fft.irfft(np.conj(a_hat) * b_hat, length, axis=-1)[..., :size]

    f_hat = transform(f)
    # X + R at each point, and R - X at each point from 0 up.
    total_hat = np.conj(
        transform(fft.irfft(f_hat * transform(g), length)[:size])
    )
    difference = np.maximum(correlated(f_hat, transform(g)), 0.0)
    difference_hat = transform(difference)
    # m(u): the trapezoid rule on the grid.
    weighted = survival * h
    weighted[0] *= 0.5
    shorter = correlated(transform(weighted), transform(survival))
    still = correlated(f_hat, transform(survival))  # P(R > a + X)
    term = np.stack([survival, correlated(f_hat, transform(shorter)), still])
    term += unswitched[:, None] * np.maximum(survival - still, 0.0)
    np.maximum(term, 0.0, out=term)
    solution = term.copy()
    for _ in range(MAX_SUM):
        term_hat = transform(term)
        # E[W(a + X + R)] + E[W(R - X - a); R > X + a], a tie half.
        step = fft.irfft(
            total_hat * term_hat + np.conj(term_hat) * difference_hat,
            length,
            axis=-1,
        )[..., :size]
        step -= 0.5 * term[:, :1] * difference
        np.maximum(step, 0.0, out=step)
        solution += step
        # Once a step only scales the terms, by ``ratio``, the rest of
        # the sum is the geometric series of the last.
        ratio = float(step[0, 0] / max(term[0, 0], 1e-300))
        off = np.abs(step - ratio * term).max(axis=1)
        term = step
        if ratio > 1.0 - SLOWEST:
            break
        if np.all(off <= SUM_RTOL * (1.0 - ratio) * solution[:, 0]):
            return solution[:, 0] + step[:, 0] * ratio / (1.0 - ratio)
    return None


def _extrapolated(last: List[np.ndarray]) -> np.ndarray:
    """The limit of three solutions on halved grids whose differences
    shrink by a constant factor (the last where they do not shrink)."""
    a, b, c = last
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (a - b) / (b - c)
        limit = c + (c - b) / (ratio - 1.0)
    return np.where(np.isfinite(limit) & (ratio > 1.0), limit, c)


def _settled(life, repair, unswitched: np.ndarray) -> Tuple[np.ndarray, float]:
    """``W(0)`` for the units' failures, the down time and the system's
    failures, the grid halved until two extrapolated solutions agree, and
    how far apart (relatively) the last two were (``unswitched``: see
    ``_solution``)."""
    end = float(np.ravel(repair.qf(1.0 - TAIL))[0])
    steps = FIRST_STEPS
    solutions: List[np.ndarray] = []
    limit: Optional[np.ndarray] = None
    apart = np.inf
    while steps <= MAX_STEPS:
        solution = _solution(life, repair, end, steps, unswitched)
        steps *= 2
        if solution is None:
            solutions.clear()
            limit = None
            continue
        solutions.append(solution)
        if len(solutions) < 3:
            continue
        before, limit = limit, _extrapolated(solutions[-3:])
        if before is None:
            continue
        apart = float(np.max(np.abs(limit - before) / np.abs(limit)))
        if apart <= RTOL:
            break
    if limit is None:
        raise NotImplementedError(
            "The standby pair is under repair so much of the time (its "
            "lives so short beside its repairs) that its long-run values "
            f"do not settle on a grid of {MAX_STEPS} steps: simulate it "
            "with availability() or cost()."
        )
    return limit, apart


def long_run(life, repair, switching: float = 1.0) -> StandbyLongRun:
    """The long-run values of a cold standby pair whose units are both
    repaired at once, lives ``life`` and repairs ``repair`` (any models
    with ``ff``, ``sf`` and ``qf``), whose switch works with probability
    ``switching``."""
    try:
        return _KEPT[life][repair][switching]
    except (KeyError, TypeError):  # not kept yet, or not kept at all
        pass
    values = _worked_out(life, repair, switching)
    try:
        kept = _KEPT.setdefault(life, weakref.WeakKeyDictionary())
        kept.setdefault(repair, {})[switching] = values
    except TypeError:  # a model that is not weakly referenced
        pass
    return values


def _worked_out(life, repair, switching: float) -> StandbyLongRun:
    """See ``long_run``."""
    if switching <= 0.0:
        return _standby_cycle.unswitched(life, repair)
    mean_life = _standby_cycle.mean_time(life)
    unswitched = np.zeros(3)
    if switching < 1.0:
        unswitched = (1.0 - switching) * _standby_cycle.switched(
            repair, switching
        )
    (units, down, fails), apart = _settled(life, repair, unswitched)
    if apart > RTOL:
        warnings.warn(
            "The long-run values of a cold standby pair with both units "
            f"repaired at once did not settle to {RTOL:g} on a grid of "
            f"{MAX_STEPS} steps: the last two solutions agree to "
            f"{apart:.1e} (relative). Check them with a simulation "
            "(availability()).",
            stacklevel=outside_level(),
        )
    cycle = mean_life * units + down
    unavailability = down / cycle
    return StandbyLongRun(
        availability=1.0 - unavailability,
        failure_frequency=fails / cycle,
        unit_failure_frequency=units / cycle,
        unavailability=unavailability,
    )
