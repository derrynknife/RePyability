"""Hidden failures whose tests or repairs take time, or whose tests can
miss a failure, for any life (#159).

A unit with hidden failures is tested every ``T``. A test of a working unit
takes it off line for a time ``D`` (a planned outage), during which it does
not age. A unit that has failed stays down, unseen, until a test finds it:
that test takes its time ``D``, then the unit is repaired, in a time ``R``,
and a new unit is put into service; the tests that fall in the repair are
not done. A test finds a failure with probability ``coverage``, decided at
the first test after the failure; one it misses stays hidden until the
next full test (every ``per`` tests), which finds every failure.

The tests that find failures are the unit's regeneration points: from one,
a new unit is put into service a time ``rho = D + R`` later, runs through
its tests until it fails, and its failure is found at a later test. That
cycle is followed test by test, by its lags from the test that starts it,
on a grid of step ``h`` across a test interval (``_walk``):

- a unit put into service ``v`` into an interval is of age ``T - v`` at
  the next test;
- a working unit of age ``a`` at a test is off line for the test's time
  ``d``, and of age ``a + T - d`` at the next one;

so its age at its tests is a random walk, whose law (masses ``alpha`` on
the age grid) is followed lag by lag: the unit is working at a test with
probability ``sum_a alpha(a) R(a)``, and has failed ``s`` after it with
probability ``sum_a alpha(a) sum_{d <= s} g(d) (R(a) - R(a + s - d))``, ``g``
the test's time on the grid. Each part of a distribution goes to the two
nearest grid points, its mean kept, and the error falls as ``h ** 2``. A
unit with an exponential life needs no ages: it is working, failed or in a
repair at each test, and that chain is followed from test to test instead
(``_Chain``).

In the long run the values follow by renewal-reward over the cycle; with
tests that can miss a failure, over the cycles from each place in the full
tests' period, where the test that ends one falls (a Markov renewal
process). From new, or from a state, the tests that find failures are a
renewal process on the tests, ``r_m = h_m + sum_j r_j q_{m - j}`` (``h`` the
first unit's, ``q`` a cycle's), and the unit's values in each interval
follow from them and the cycle's lags.

When a test is over, and when a unit is back in service, are kept out of
the grid: they are times from a test, exactly, so their CDFs are exact at
any time (but for the sum of two spread times, taken on the grid). In each
interval, ``s`` after its test, the unit is up with probability

``tested * G(s) + sum_l coef[l] * restart.cdf(l, s) - failures(s)``,

``tested`` the probability that it was working at the test, ``G`` the
test's CDF, ``coef[l]`` that the test ``l`` before found a failure (whose
unit is put back into service ``l`` intervals on with ``restart.cdf``),
and ``failures(s)``, on the grid, its expected failures in the interval
by ``s`` (see ``_Rows``).
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from ._block_replacement import (
    _MAX_GRID,
    _MAX_VALUES,
    _UNSUPPORTED,
    _check_life,
    _cumulative,
    _Duration,
    _steps,
    _values,
)
from ._model_utils import distribution_name, never_fails
from ._point_availability import _KNOT_PROBABILITIES, Atoms, _events

#: A unit's chance of still working below which a cycle has ended (its
#: later terms are then exact to rounding).
NEGLIGIBLE = 1e-17
#: The ends of the age walk's masses that hold less than this are dropped,
#: and its masses below the convolutions' rounding, relative to its largest
#: (an FFT's is some 1e-16 of the result's norm, which would otherwise
#: spread the masses by the test's time at every test).
_TRIM = 1e-20
_ROUNDING = 1e-13
#: The most survival values a cycle's walk may take (its lags times the
#: ages each spans) before the values refuse: some seconds' work.
_MAX_EVALUATIONS = 2**27
#: The most tests a unit's replacements are followed over, for its spares.
MAX_RENEWAL_TESTS = 200_000
#: How close the values over time must come to the long run's (a bound on
#: the distance) to be taken to repeat them.
SETTLED = 1e-12


def check(
    life, repair, test, interval: float, node=None, instead=_UNSUPPORTED
) -> None:
    """Raise unless a unit with hidden failures has numerical values (see
    ``TestedUnit``): a surpyval parametric life with a density (no unit
    dead on arrival); a repair and a test time each none, fixed or a
    surpyval parametric model, every one ending; and tests over within the
    interval. Cheap: nothing is worked out. ``instead`` ends a refusal,
    with what to do."""
    where = f"Component {node!r} has hidden failures: their numerical values"
    name = distribution_name(life)
    if name is None or name in (
        "ExactEventTime",
        "FixedEventProbability",
        "Bernoulli",
    ):
        raise NotImplementedError(
            f"{where} need a surpyval parametric lifetime with a density, "
            f"and its reliability model is not one. {instead}"
        )
    if float(getattr(life, "f0", 0.0) or 0.0) > 0.0:
        raise NotImplementedError(
            f"{where} do not cover units dead on arrival (f0 > 0). "
            f"{instead}"
        )
    for model, what in ((repair, "repair"), (test, "test")):
        if model is None or distribution_name(model) == "ExactEventTime":
            continue
        if distribution_name(model) is None:
            raise NotImplementedError(
                f"{where} need its {what} time to be a surpyval parametric "
                f"model, which it is not. {instead}"
            )
        if never_fails(model) > 0.0:
            raise NotImplementedError(
                f"{where} need every {what} to end, and some of its never "
                f"do (p < 1). {instead}"
            )
    if test is None:
        return
    duration = _Duration(test, "test", node)
    T = float(interval)
    if not duration.tail() < T:
        chance = 1.0
        if duration.fixed is None:
            chance = float(np.ravel(_values(test.sf, np.array([T])))[0])
        raise NotImplementedError(
            f"{where} need its tests to be over within its test interval "
            "(but for a chance below 1e-12), and they last as long with a "
            f"chance of {chance:.2g}. {instead}"
        )


def _too_long(node, what: str, instead=_UNSUPPORTED) -> NotImplementedError:
    return NotImplementedError(
        f"Component {node!r}: {what}, too long for the numerical values of "
        f"its hidden failures. {instead}"
    )


class _Left:
    """What is left of a repair under way for ``down_for`` (``model`` a
    surpyval model), as a ``_Duration``."""

    fixed = None

    def __init__(self, model, down_for: float):
        self.model = model
        self.down_for = float(down_for)
        self.known = float(
            np.ravel(_values(model.sf, np.array([self.down_for])))[0]
        )

    def sf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.clip(
            _values(self.model.sf, self.down_for + x) / self.known, 0.0, 1.0
        )

    def cdf(self, x) -> np.ndarray:
        return 1.0 - self.sf(x)

    def in_progress(self, edges: np.ndarray) -> np.ndarray:
        return _cumulative(self.sf, edges)

    def tail(self) -> float:
        p = 1.0 - 1e-12 * self.known
        q = float(np.ravel(self.model.qf(np.array([p])))[0])
        return max(q - self.down_for, 0.0)


_GAUSS = np.polynomial.legendre.leggauss(8)


def _shares(sf, cells: int, h: float, shift: float = 0.0):
    """A time ``shift`` plus a duration whose survival function is ``sf``,
    on the grid of step ``h`` from 0: each of the first ``cells`` cells'
    probability split between its two ends so that its mean is kept,
    ``left = S(x_c) - (mean of S over the cell)`` and ``right = (mean of S)
    - S(x_{c + 1})``, ``S`` the time's survival function. Each is a
    difference of values near each other, to rounding (where one from a
    running integral would lose ``h`` against the grid's length); the
    cell the time starts in is split there."""
    points, weights = _GAUSS
    x = h * np.arange(cells + 1)

    def survival(t):
        t = np.asarray(t, dtype=float)
        since = np.maximum(t - shift, 0.0)
        return np.where(t > shift, np.clip(_values(sf, since), 0.0, 1.0), 1.0)

    S = survival(x)
    a = x[:-1]
    half = 0.5 * h
    nodes = (a + half)[:, None] + half * points[None, :]
    mean = 0.5 * (survival(nodes) @ weights)
    first = int(np.floor(shift / h))
    if 0 <= first < cells and shift > first * h:
        # Split at the shift: 1 before it, the duration's survival after.
        lo, hi = shift, (first + 1) * h
        mid, width = 0.5 * (lo + hi), 0.5 * (hi - lo)
        after = width * float(survival(mid + width * points) @ weights)
        mean[first] = ((shift - first * h) + after) / h
    left = np.maximum(S[:-1] - mean, 0.0)
    right = np.maximum(mean - S[1:], 0.0)
    return left, right


class _Restart:
    """When a unit is put into service, on a frame of test intervals from a
    test (or, for a unit in a repair at 0, from the test before 0), each
    ``T`` long on a grid of ``steps`` cells of ``h``; from its
    probability in each cell, split between the cell's two ends so that its
    mean is kept (``left``, ``right``, on the long grid):

    - ``masses[l]``, where in the ``l``-th interval, on its grid points
      (the part at an interval's end is put into service before the test
      there, and so is tested by it);
    - ``pending(l)``, the probability that it is not by the ``l``-th test,
      which, falling in the repair, is not done;
    - ``cdf(l, s)``, the probability that it is in the ``l``-th interval,
      by ``s`` after its start: exactly, or linear between the grid's
      points for the sum of two spread times.

    It is a fixed time (``atom``: one within rounding of a test starts the
    interval after it, whose test is then not done), a duration from
    ``shift`` on, or the sum of two."""

    def __init__(self, setup: "_Setup", left, right):
        self.T, self.steps, self.h = setup.interval, setup.steps, setup.h
        steps = self.steps
        cells = len(left)
        # Every repair ends: what lies past the grid (at most its 1e-12
        # tail) is put back, so that no cycle loses its unit.
        total = float(np.sum(left) + np.sum(right))
        left, right = left / total, right / total
        lags = max(1, -(-cells // steps))
        pad = lags * steps - cells
        left = np.concatenate([left, np.zeros(pad)]).reshape(lags, steps)
        right = np.concatenate([right, np.zeros(pad)]).reshape(lags, steps)
        masses = np.zeros((lags, steps + 1))
        masses[:, :-1] += left
        masses[:, 1:] += right
        # The intervals it can fall in.
        total = masses.sum(axis=1)
        last = np.flatnonzero(total > 0.0)
        count = int(last[-1]) + 1 if last.size else 1
        self.masses = masses[:count]
        share = total[:count]
        self._pending = np.concatenate([np.cumsum(share[::-1])[::-1], [0.0]])
        #: Its CDF at the long grid's points.
        self.P = np.concatenate([[0.0], np.cumsum((left + right).ravel())])[
            : count * steps + 1
        ]
        self.atom: Optional[Tuple[int, float]] = None
        self.duration = None
        self.shift = 0.0
        at = np.searchsorted(self.P, _KNOT_PROBABILITIES * self.P[-1])
        self._knots = np.unique(self.h * np.clip(at, 0, len(self.P) - 1))

    @classmethod
    def at(cls, setup: "_Setup", time: float) -> "_Restart":
        """At ``time`` from the frame's start, exactly."""
        h, steps = setup.h, setup.steps
        position = time / h
        cell = int(np.floor(position + 1e-9))
        share = max(position - cell, 0.0)
        left, right = np.zeros(cell + 1), np.zeros(cell + 1)
        left[cell], right[cell] = 1.0 - share, share
        out = cls(setup, left, right)
        lag = cell // steps
        out.atom = (lag, max(time - lag * setup.interval, 0.0))
        out._knots = np.array([time])
        return out

    @classmethod
    def after(cls, setup: "_Setup", shift: float, duration) -> "_Restart":
        """``duration`` (a ``_Duration`` or ``_Left``) after ``shift``."""
        if duration.fixed is not None:
            return cls.at(setup, shift + duration.fixed)
        cells = setup.cells(shift + duration.tail())
        sf = getattr(duration, "sf", None) or duration.model.sf
        out = cls(setup, *_shares(sf, cells, setup.h, shift))
        out.duration, out.shift = duration, float(shift)
        return out

    @classmethod
    def sum_of(cls, setup: "_Setup", first, second) -> "_Restart":
        """``first``, then ``second``, both spread: ``first`` as masses on
        the grid's points, which move ``second``'s cells whole."""
        from scipy.signal import fftconvolve

        cells = setup.cells(first.tail() + second.tail())
        h = setup.h
        left, right = _shares(first.model.sf, cells, h)
        points = np.concatenate([left, [0.0]]) + np.concatenate([[0.0], right])
        later = _shares(second.model.sf, cells, h)
        moved = [
            np.maximum(fftconvolve(points, part)[:cells], 0.0)
            for part in later
        ]
        return cls(setup, *moved)

    @property
    def lags(self) -> int:
        """The intervals it can fall in."""
        return len(self.masses)

    def pending(self, lag) -> np.ndarray:
        """``P(not put into service before the lag-th test)``."""
        lag = np.asarray(lag)
        return self._pending[np.clip(lag, 0, len(self._pending) - 1)]

    def _frame_cdf(self, x: np.ndarray) -> np.ndarray:
        """``P(put into service by x)``, ``x`` from the frame's start."""
        if self.duration is not None:
            since = np.maximum(x - self.shift, 0.0)
            return np.where(x >= self.shift, self.duration.cdf(since), 0.0)
        return np.interp(x, self.h * np.arange(len(self.P)), self.P)

    def cdf(self, lag: int, s) -> np.ndarray:
        """``P(put into service in the lag-th interval, by s after its
        start)``."""
        s = np.asarray(s, dtype=float)
        if lag < 0 or lag >= self.lags:
            return np.zeros(s.shape)
        if self.atom is not None:
            at_lag, at = self.atom
            return ((lag == at_lag) & (s >= at)).astype(float)
        start = lag * self.T
        base = self._frame_cdf(np.array([start]))[0]
        return np.maximum(self._frame_cdf(start + s) - base, 0.0)

    def integral(self, lag: int) -> float:
        """``integral_0^T cdf(lag, s) ds``: each mass by how long before
        the interval's end it falls (exact, as the masses keep each
        cell's mean)."""
        if lag < 0 or lag >= self.lags:
            return 0.0
        return float(
            self.masses[lag] @ (self.T - self.h * np.arange(self.steps + 1))
        )

    def knots(self, lag: int) -> np.ndarray:
        """Where in the ``lag``-th interval its CDF bends."""
        local = self._knots - lag * self.T
        return local[(local >= 0.0) & (local < self.T)]


class _Setup:
    """What a tested unit is followed on: its test interval ``T`` and the
    grid across it (``steps`` of ``h``); the test's time (``test``, its CDF
    exact) and its masses ``g`` on the grid; the life's CDF ``F`` on the
    grid; and ``restart``, when a unit is put back into service after a
    test that finds it failed: the test's time, then the repair's."""

    def __init__(self, life, repair, test, interval: float, node=None):
        _check_life(life, node)
        self.life, self.node = life, node
        self.fix = _Duration(repair, "repair", node)
        self.test = _Duration(test, "test", node)
        #: Whether a test takes a working unit off line (a planned outage,
        #: if only for no time), as a test with a time model does.
        self.timed = test is not None
        T = self.interval = float(interval)
        if not self.test.tail() < T:
            raise NotImplementedError(
                f"Component {node!r}: its tests can last as long as its "
                "test interval, which the numerical values of its hidden "
                f"failures do not take. {_UNSUPPORTED}"
            )
        self.steps = steps = _steps(life, T)
        self.h = h = T / steps
        self.grid = h * np.arange(steps + 1)
        D, R = self.test, self.fix
        if D.fixed is not None:
            position = D.fixed / h
            cell = int(np.floor(position + 1e-9))
            g = np.zeros(cell + 2)
            g[cell] = 1.0 - max(position - cell, 0.0)
            g[cell + 1] = max(position - cell, 0.0)
        else:
            # Up to where it is all but surely over (its far tail would
            # spread the ages a walk follows for nothing).
            cells = min(steps, int(np.ceil(D.tail() / h)) + 1)
            left, right = _shares(D.model.sf, cells, h)
            g = np.concatenate([left, [0.0]]) + np.concatenate([[0.0], right])
            g = g / g.sum()
        nonzero = np.flatnonzero(g)
        #: The test's time as masses on the grid's points.
        self.g = g[: (int(nonzero[-1]) if nonzero.size else 0) + 1]
        self.F = np.clip(_values(life.ff, self.grid), 0.0, 1.0)
        self.F[0] = 0.0
        if D.fixed is not None:
            self.restart = _Restart.after(self, D.fixed, R)
        elif R.fixed is not None:
            self.restart = _Restart.after(self, R.fixed, D)
        else:
            self.restart = _Restart.sum_of(self, D, R)
        #: Where in an interval the test's CDF bends.
        self.test_knots = np.empty(0)
        if D.fixed is not None and D.fixed > 0.0:
            self.test_knots = np.array([D.fixed], dtype=float)
        elif D.fixed is None and D.model is not None:
            q = np.ravel(_values(D.model.qf, _KNOT_PROBABILITIES))
            self.test_knots = np.unique(q[np.isfinite(q) & (q > 0.0)])

    def cells(self, reach: float) -> int:
        """The grid's cells from 0 past ``reach``, and an interval more."""
        if not np.isfinite(reach):
            raise _too_long(self.node, "its repairs can take for ever")
        cells = int(np.ceil(reach / self.h)) + self.steps + 2
        if cells + self.steps > _MAX_GRID:
            raise _too_long(
                self.node,
                "its repairs can take far longer than its test interval",
            )
        return cells

    def G(self, s) -> np.ndarray:
        """``P(the test is over by s after it starts)``."""
        return self.test.cdf(np.asarray(s, dtype=float))

    @property
    def test_integral(self) -> float:
        """``integral_0^T G(s) ds``."""
        return float(self.interval - self.test.in_progress(self.grid)[-1])


class _Survival:
    """A life's survival function on the age grid ``origin + i h``, worked
    out a block at a time as the ages a walk reaches grow."""

    def __init__(self, life, origin: float, h: float, node):
        self.life, self.origin, self.h, self.node = life, origin, h, node
        self.lo = 0
        self.values = np.empty(0)
        self.count = 0

    def window(self, lo: int, hi: int) -> np.ndarray:
        if lo < self.lo or hi > self.lo + len(self.values):
            stop = lo + 4 * (hi - lo)
            x = self.origin + self.h * np.arange(lo, stop)
            self.values = np.clip(_values(self.life.sf, x), 0.0, 1.0)
            self.lo = lo
            self.count += stop - lo
            if self.count > _MAX_EVALUATIONS:
                raise _too_long(
                    self.node,
                    "its life lasts so long against its test interval that "
                    "a renewal cycle runs over too many tests",
                )
        return self.values[lo - self.lo : hi - self.lo]  # noqa: E203


class _Cycle(NamedTuple):
    """A cycle of a tested unit, lag by lag from the test that starts it
    (see ``_walk``): the probability that the unit is working at the
    ``l``-th test (``p[l]``); its failures in the ``l``-th interval by its
    end (``fail[l]``), and by each point of the grid for the first lags
    (``failures``); and over all its lags, by their remainder on division
    by ``per``, the sums of ``p`` and of the failures (``p_by``,
    ``failures_by``)."""

    p: np.ndarray
    fail: np.ndarray
    failures: np.ndarray
    p_by: np.ndarray
    failures_by: np.ndarray


def _merge(lo_a: int, a: np.ndarray, lo_b: int, b: np.ndarray):
    """Masses on the age grid from ``lo_a`` and ``lo_b``, added."""
    if not a.size:
        return lo_b, b
    if not b.size:
        return lo_a, a
    lo = min(lo_a, lo_b)
    out = np.zeros(max(lo_a + len(a), lo_b + len(b)) - lo)
    out[lo_a - lo : lo_a - lo + len(a)] += a  # noqa: E203
    out[lo_b - lo : lo_b - lo + len(b)] += b  # noqa: E203
    return lo, out


#: A convolution with an operand this short or shorter is worked out
#: directly, as scipy chooses to, and a longer one by FFT: without scipy's
#: choosing each time, which cost as much as the walk's convolutions in an
#: interval search (#229).
_DIRECT = 64


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` convolved with ``b`` (see ``_DIRECT``)."""
    if not (len(a) and len(b)):
        return np.zeros(max(len(a) + len(b) - 1, 0))
    if min(len(a), len(b)) <= _DIRECT:
        return np.convolve(a, b)
    from scipy.signal import fftconvolve

    return fftconvolve(a, b)


def _correlate(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``a`` correlated with the shorter ``v``, where ``v`` lies within
    ``a`` (see ``_DIRECT``)."""
    if min(len(a), len(v)) <= _DIRECT:
        return np.correlate(a, v, mode="valid")
    from scipy.signal import fftconvolve

    return fftconvolve(a, v[::-1], mode="valid")


def _trimmed(lo: int, ages: np.ndarray):
    """Masses on the age grid without those below the rounding of the
    convolutions that made them (relative to the largest), and without
    the ends that hold less than ``_TRIM`` between them."""
    if not ages.size:
        return 0, ages
    ages = np.where(ages > _ROUNDING * ages.max(), ages, 0.0)
    ahead = np.cumsum(ages)
    behind = np.cumsum(ages[::-1])
    first = int(np.searchsorted(ahead, _TRIM, side="right"))
    last = len(ages) - int(np.searchsorted(behind, _TRIM, side="right"))
    if first >= last:
        return 0, np.empty(0)
    return lo + first, ages[first:last]


def _walk(
    setup: _Setup,
    restart: Optional[_Restart] = None,
    alive: Optional[Tuple[float, float]] = None,
    keep: int = 0,
    per: int = 1,
) -> _Cycle:
    """A cycle of a tested unit, lag by lag from the test that starts it
    (see the module docstring): of the units ``restart`` puts into
    service, or ``alive``, ``(age, weight)``, a unit working at the first
    test with probability ``weight`` times its survival to that age (from
    a state: on an age grid through it). It is followed until the unit has
    surely failed; ``failures`` keeps the first ``keep`` lags'."""
    steps = setup.steps
    g = setup.g
    reach = len(g) - 1
    lo, ages = 0, np.empty(0)
    origin = 0.0
    lags = 0 if restart is None else restart.lags
    if alive is not None:
        start = int(np.floor(alive[0] / setup.h))
        origin = alive[0] - start * setup.h
        lags = max(lags, 2)
    survival = _Survival(setup.life, origin, setup.h, setup.node)
    p_list: List[float] = []
    fail_list: List[float] = []
    kept: List[np.ndarray] = []
    p_by = np.zeros(per)
    failures_by = np.zeros((per, steps + 1))
    lag = 0
    while True:
        if lag == 1 and alive is not None:
            lo, ages = _merge(lo, ages, start, np.array([float(alive[1])]))
        failures = np.zeros(steps + 1)
        p = 0.0
        if ages.size:
            window = survival.window(lo, lo + len(ages) + steps)
            Q = _correlate(window, ages)
            p = float(Q[0])
            failures = np.maximum(_convolve(g, p - Q)[: steps + 1], 0.0)
        w = None
        if restart is not None and lag < restart.lags:
            w = restart.masses[lag]
            failures = failures + _convolve(w, setup.F)[: steps + 1]
        p_list.append(p)
        fail_list.append(float(failures[-1]))
        if lag < keep:
            kept.append(failures)
        p_by[lag % per] += p
        failures_by[lag % per] += failures
        if lag >= lags and p < NEGLIGIBLE:
            break
        # The ages at the next test.
        if ages.size:
            lo, ages = lo - reach + steps, _convolve(ages, g[::-1])
        if w is not None:
            lo, ages = _merge(lo, ages, 0, w[::-1])
        ages = np.maximum(ages, 0.0)
        total = float(ages.sum())
        lo, ages = _trimmed(lo, ages)
        if ages.size:
            # What the trimming dropped, put back: the walk keeps its mass.
            ages = ages * (total / float(ages.sum()))
        lag += 1
    failures_kept = np.zeros((keep, steps + 1))
    if kept:
        failures_kept[: len(kept)] = kept
    return _Cycle(
        np.array(p_list),
        np.array(fail_list),
        failures_kept,
        p_by,
        failures_by,
    )


class _Rows(NamedTuple):
    """Test intervals' values, a row each: ``s`` after the test that
    starts one, the unit is up with probability ``tested * G(s) + sum_l
    coef[l] * restart.cdf(l, s) - failures(s)`` and down with ``tested *
    (1 - G(s)) + sum_l coef[l] * (restart.pending(l) - restart.cdf(l, s))
    + failures(s) + missed`` (see the module docstring): ``tested`` the
    probability that it was working at the test (and so off line until the
    test is over), ``coef[l]`` that the test ``l`` before found a failure,
    ``failures`` its expected failures in the interval by each point of the
    grid, ``missed`` that a failure from before the test, which a test
    missed, waits through it; ``inspected`` the probability that the test
    is done (it is not in a repair) and ``found`` that it finds a
    failure."""

    tested: np.ndarray
    coef: np.ndarray
    failures: np.ndarray
    missed: np.ndarray
    inspected: np.ndarray
    found: np.ndarray


def _stacked(*parts: _Rows) -> _Rows:
    parts = tuple(part for part in parts if len(part.tested))
    return _Rows(
        *(np.concatenate([part[i] for part in parts]) for i in range(6))
    )


def _full(per: int, coverage: float, positions: np.ndarray) -> np.ndarray:
    """Whether the tests at ``positions`` in the full tests' period find
    every failure."""
    if coverage >= 1.0 or per == 1:
        return np.ones(np.shape(positions), dtype=bool)
    return np.asarray(positions) % per == 0


def _found(
    fail: np.ndarray, positions: np.ndarray, coverage: float, per: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """The failures in each interval (``fail[l]``, found or missed at the
    test that ends it, at ``positions[l]`` in the full tests' period):
    the expected failures found at each test (by its index, from 1); those
    missed and waiting through each interval; and the expected tests of a
    failed unit (each test from its failure to the one that finds it)."""
    n = len(fail)
    j = np.arange(1, n + 1)
    full = _full(per, coverage, positions)
    chance = np.where(full, 1.0, coverage)
    wait = np.where(full, 0, (-np.asarray(positions)) % per)
    found = np.bincount(j, fail * chance, n + per + 1)
    found += np.bincount(j + wait, fail * (1.0 - chance), n + per + 1)
    edges = np.bincount(j, fail * (1.0 - chance), n + per + 2)
    edges -= np.bincount(j + wait, fail * (1.0 - chance), n + per + 2)
    missed = np.maximum(np.cumsum(edges)[: n + per + 1], 0.0)
    tests = float(fail @ (chance + (1.0 - chance) * (wait + 1)))
    return found, missed, tests


class _Chain:
    """An exponential life's unit from test to test: at each, the
    probability that it is working (``W``), that it failed since the last
    test (``fresh``) or before it, missed by a test (``missed``), and that
    each of the last tests found a failure whose unit is not yet back in
    service (``history``, by how many tests ago). From these, an
    interval's row (see ``_Rows``) and the next test's state follow."""

    def __init__(self, setup: _Setup, coverage: float, per: int, first=None):
        self.setup, self.coverage, self.per = setup, coverage, per
        F = setup.F
        restart = setup.restart
        self.tf = _convolve(setup.g, F)[: setup.steps + 1]
        self.kappa = 1.0 - float(self.tf[-1])
        self.rf = np.array(
            [_convolve(w, F)[: setup.steps + 1] for w in restart.masses]
        )
        share = restart.masses.sum(axis=1)
        self.a = share - self.rf[:, -1]
        self.lags = restart.lags
        self.pending = restart.pending(np.arange(self.lags))
        #: A unit in a repair at 0 (from a state): its restart on the frame
        #: from the test before 0.
        self.first = first
        if first is not None:
            self.rf0 = np.array(
                [_convolve(w, F)[: setup.steps + 1] for w in first.masses]
            )
            self.a0 = first.masses.sum(axis=1) - self.rf0[:, -1]

    def state(self, W: float, fresh: float) -> np.ndarray:
        """A state with nothing missed or in a repair."""
        out = np.zeros(2 + self.lags)
        out[0], out[1] = W, fresh
        return out

    def step(self, x: np.ndarray, position: int, lag: Optional[int] = None):
        """The row of the interval after a test at ``position`` in the full
        tests' period, from the state ``x`` there, and the next test's
        state; ``lag``, the test's index from the test before 0, for the
        first unit's restart."""
        W, fresh, missed = x[0], x[1], x[2]
        history = x[3:]
        full = bool(_full(self.per, self.coverage, np.array([position]))[0])
        chance = 1.0 if full else self.coverage
        found = fresh * chance + (missed if full else 0.0)
        waiting = (0.0 if full else missed) + fresh * (1.0 - chance)
        coef = np.concatenate([[found], history])[: self.lags]
        failures = W * self.tf + coef @ self.rf
        working = W * self.kappa + float(coef @ self.a)
        inspected = 1.0 - float(coef[1:] @ self.pending[1:])
        if self.first is not None and lag is not None:
            if lag < self.first.lags:
                failures = failures + self.rf0[lag]
                working += float(self.a0[lag])
            inspected -= float(self.first.pending(lag))
        row = (W, coef, failures, waiting, inspected, found)
        nxt = np.concatenate([[working, failures[-1], waiting], coef[:-1]])
        return row, nxt

    def rows(self, x: np.ndarray, positions, first_lag=None, settled=None):
        """The rows from the state ``x`` at the first of the tests at
        ``positions``; and with ``settled``, the long run's state at each
        place in the period, the index of the row from which they are the
        long run's (the state at its test the long run's, and the first
        unit's restart over), the rows stopping there (None if they do not
        get there), else the state after the last."""
        out: list = []
        for k, position in enumerate(positions):
            lag = None if first_lag is None else first_lag + k
            if (
                settled is not None
                and (self.first is None or lag >= self.first.lags)
                and np.max(np.abs(x - settled[int(position)])) <= 1e-15
            ):
                return _rows_of(out, self.lags), k
            row, x = self.step(x, int(position), lag)
            out.append(row)
        return _rows_of(out, self.lags), (None if settled is not None else x)

    def long_run(self) -> Tuple[_Rows, list]:
        """The long run's rows, and its states, at each place in the
        period."""
        x = self.stationary()
        states: list = []
        out: list = []
        for position in range(self.per):
            states.append(x)
            row, x = self.step(x, position)
            out.append(row)
        return _rows_of(out, self.lags), states

    def stationary(self) -> np.ndarray:
        """The long-run state at a full test."""
        n = 2 + self.lags
        period = np.eye(n)
        for position in range(self.per):
            columns = [self.step(period[:, i], position)[1] for i in range(n)]
            period = np.array(columns).T
        weights = np.concatenate([[1.0, 1.0, 1.0], self.pending[1:]])
        system = np.vstack([period - np.eye(n), weights])
        rhs = np.concatenate([np.zeros(n), [1.0]])
        x = np.linalg.lstsq(system, rhs, rcond=None)[0]
        for _ in range(20):
            x = period @ x
            x = x / float(weights @ x)
        return np.maximum(x, 0.0)


def _rows_of(rows: list, lags: int) -> _Rows:
    if not rows:
        return _Rows(
            np.empty(0),
            np.empty((0, lags)),
            np.empty((0, 0)),
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )
    tested, coef, failures, missed, inspected, found = zip(*rows)
    return _Rows(
        np.array(tested, dtype=float),
        np.array(coef, dtype=float).reshape(len(rows), lags),
        np.array(failures, dtype=float),
        np.array(missed, dtype=float),
        np.array(inspected, dtype=float),
        np.array(found, dtype=float),
    )


class TestedLongRun:
    """A tested unit in the long run: its values in the interval after a
    test at each place in the full tests' period (``rows``, by position;
    see ``_Rows``), and from them its profile over the period, from a full
    test, its means and its rates of events."""

    def __init__(self, setup: _Setup, rows: _Rows, per: int):
        self.setup, self.rows, self.per = setup, rows, per
        self.interval = setup.interval
        self.period = per * setup.interval
        T = self.interval
        restart = setup.restart
        lags = restart.lags
        G = setup.test_integral
        cdfs = np.array([restart.integral(lag) for lag in range(lags)])
        pend = restart.pending(np.arange(lags))
        grid_integral = setup.h * (
            rows.failures.sum(axis=1)
            - 0.5 * (rows.failures[:, 0] + rows.failures[:, -1])
        )
        up = rows.tested * G + rows.coef @ cdfs - grid_integral
        down = (
            rows.tested * (T - G)
            + rows.coef @ (T * pend - cdfs)
            + grid_integral
            + T * rows.missed
        )
        #: The mean availability and unavailability over the period (the
        #: latter worked out in its own right).
        self.availability = float(min(1.0, max(0.0, up.sum() / self.period)))
        self.unavailability = float(
            min(1.0, max(0.0, down.sum() / self.period))
        )
        #: Failures, tests done, tests that take a working unit off line
        #: (planned outages) and failures found, per unit time.
        self.failures = float(rows.failures[:, -1].sum() / self.period)
        self.inspections = float(rows.inspected.sum() / self.period)
        self.planned = (
            float(rows.tested.sum() / self.period) if setup.timed else 0.0
        )
        self.corrective = float(rows.found.sum() / self.period)

    @property
    def rate(self) -> float:
        """How fast its availability falls between tests: its failures per
        unit of up time."""
        if self.availability <= 0.0:
            return 1.0 / self.interval
        return self.failures / self.availability

    def _place(self, phase) -> Tuple[np.ndarray, np.ndarray]:
        phase = np.asarray(phase, dtype=float)
        position = np.clip(
            np.floor(phase / self.interval).astype(int), 0, self.per - 1
        )
        return position, np.clip(
            phase - position * self.interval, 0.0, self.interval
        )

    def profile(self, phase) -> Tuple[np.ndarray, np.ndarray]:
        """The probabilities of being up and down at each ``phase``, the
        time since the last full test (from 0 to ``period``)."""
        phase = np.asarray(phase, dtype=float)
        position, s = self._place(phase)
        return _values_at(self.setup, self.rows, position, s)

    def intensity(self, phase) -> np.ndarray:
        """The rate of failing at each ``phase``: the slope of its failures
        on the grid cell it falls in."""
        position, s = self._place(phase)
        h = self.setup.h
        j = np.clip(np.floor(s / h).astype(int), 0, self.setup.steps - 1)
        failures = self.rows.failures
        return np.maximum(
            (failures[position, j + 1] - failures[position, j]) / h, 0.0
        )

    def around(self, position: int) -> Tuple[Tuple[float, float], ...]:
        """Up and down just before the test at ``position`` (at the end of
        the interval before), and just after it starts (which takes a
        working unit off line, with a time model), each worked out in its
        own right."""
        index = np.array([(position - 1) % self.per, position])
        s = np.array([self.interval, 0.0])
        up, down = _values_at(self.setup, self.rows, index, s)
        return (float(up[0]), float(down[0])), (float(up[1]), float(down[1]))

    def edges(self) -> np.ndarray:
        """Over the period, the grid's points and where the test's and the
        restart's CDFs bend: the edges of cells on which the profile is
        smooth."""
        within = _within(self.setup)
        return np.unique(
            (
                self.interval * np.arange(self.per)[:, None] + within[None, :]
            ).ravel()
        )


def _within(setup: _Setup) -> np.ndarray:
    """Where across an interval a row may bend: its grid's points, and the
    test's and the restarts' knots."""
    restart = setup.restart
    parts = [setup.grid[:-1], setup.test_knots]
    parts += [restart.knots(lag) for lag in range(restart.lags)]
    within = np.unique(np.concatenate(parts))
    return within[(within >= 0.0) & (within < setup.interval)]


def _values_at(setup: _Setup, rows: _Rows, index, s, first=None, lag=None):
    """Up and down in the rows ``index`` at ``s`` after their tests (with
    ``first``, a first unit's restart, at each one's ``lag`` from it)."""
    index = np.asarray(index, dtype=int)
    s = np.asarray(s, dtype=float)
    restart = setup.restart
    G = setup.G(s)
    tested = rows.tested[index]
    up = tested * G
    down = tested * (1.0 - G) + rows.missed[index]
    for back in range(restart.lags):
        coef = rows.coef[index, back]
        if not np.any(coef):
            continue
        cdf = restart.cdf(back, s)
        up = up + coef * cdf
        down = down + coef * (float(restart.pending(back)) - cdf)
    if first is not None:
        for back in np.unique(lag[lag < first.lags]):
            on = lag == back
            cdf = first.cdf(int(back), s[on])
            up[on] += cdf
            down[on] += float(first.pending(back)) - cdf
    failed = _interpolated(rows.failures, index, s, setup.h)
    return (
        np.clip(up - failed, 0.0, 1.0),
        np.clip(down + failed, 0.0, 1.0),
    )


def _interpolated(table: np.ndarray, index, s, h: float) -> np.ndarray:
    """The rows ``index`` of ``table`` (on the grid of step ``h``) at
    ``s``, linear between its points."""
    position = np.asarray(s, dtype=float) / h
    j = np.clip(np.floor(position).astype(int), 0, table.shape[1] - 2)
    fraction = position - j
    return table[index, j] * (1.0 - fraction) + table[index, j + 1] * fraction


def _detections(cycle: _Cycle, coverage: float, per: int, n: int):
    """The tests that find a cycle's failure, for a cycle from a test at
    each place ``q`` in the full tests' period: their distribution over the
    lags, to lag ``n`` (``found[q]``, made to add up to 1: the unit surely
    fails and is found, but for rounding, which would otherwise leave the
    tests that find failures a renewal process that slowly dies out); the
    failures missed and waiting through each interval (``missed[q]``); and
    the cycle's mean length, in tests (``lengths[q]``)."""
    lags = np.arange(1, len(cycle.fail) + 1)
    found = np.zeros((per, n))
    missed = np.zeros((per, n))
    lengths = np.zeros(per)
    for q in range(per):
        f, m, _ = _found(cycle.fail, (q + lags) % per, coverage, per)
        f = f / f.sum()
        lengths[q] = float(np.arange(len(f)) @ f)
        found[q], missed[q] = _padded(f, n), _padded(m, n)
    return found, missed, lengths


def _cycle_rows(
    setup: _Setup, cycle: _Cycle, coverage: float, per: int
) -> Tuple[_Rows, np.ndarray, np.ndarray]:
    """The long run's rows, by position, from a cycle (a Markov renewal
    process over where its first test falls in the full tests' period):
    and the long-run chance that a test at each position finds a failure,
    and the cycles' expected lengths, in tests, from each."""
    restart = setup.restart
    n = len(cycle.p) + per + 1
    pend = restart.pending(np.arange(n))
    found, waiting, lengths = _detections(cycle, coverage, per, n)
    moves = np.zeros((per, per))
    missed_by = np.zeros((per, per))
    for q in range(per):
        moves[q] = np.bincount((q + np.arange(n)) % per, found[q], per)
        missed_by[q] = np.bincount(np.arange(n) % per, waiting[q], per)
    if per == 1:
        stationary = np.ones(1)
    else:
        moves = moves / moves.sum(axis=1, keepdims=True)
        system = np.vstack([moves.T - np.eye(per), np.ones(per)])
        rhs = np.concatenate([np.zeros(per), [1.0]])
        stationary = np.linalg.lstsq(system, rhs, rcond=None)[0]
        stationary = np.maximum(stationary, 0.0)
        stationary /= stationary.sum()
    rate = per * stationary / float(stationary @ lengths)
    shift = (np.arange(per)[:, None] - np.arange(per)[None, :]) % per
    tested = (rate[None, :] * cycle.p_by[shift]).sum(axis=1)
    failures = np.einsum("q,pqs->ps", rate, cycle.failures_by[shift])
    missed = np.array(
        [
            sum(rate[q] * missed_by[q, (p_ - q) % per] for q in range(per))
            for p_ in range(per)
        ]
    )
    coef = rate[(np.arange(per)[:, None] - np.arange(restart.lags)) % per]
    inspected = 1.0 - coef[:, 1:] @ pend[1 : restart.lags]  # noqa: E203
    rows = _Rows(tested, coef, failures, missed, inspected, rate.copy())
    return rows, rate, lengths


def _chain_rows(chain: _Chain) -> _Rows:
    """An exponential life's long-run rows, by position, from its chain's
    stationary state at a full test."""
    return chain.long_run()[0]


def _lagged(r: np.ndarray, kernel: np.ndarray, count: int) -> np.ndarray:
    """``out[m] = sum_{j = 1}^{m} r[j] kernel[m - j]`` for ``m`` from 1 to
    ``count`` (``r`` from index 1; ``kernel`` by lag, from 0, its rows
    each an array or a number), as ``out[1:]``."""
    from scipy.signal import fftconvolve

    weights = r[1 : count + 1]  # noqa: E203
    kernel = kernel[:count]
    if len(kernel) < count:
        pad = np.zeros((count - len(kernel),) + kernel.shape[1:])
        kernel = np.concatenate([kernel, pad])
    if kernel.ndim == 1:
        return np.concatenate([[0.0], np.convolve(weights, kernel)[:count]])
    out = np.zeros((count + 1,) + kernel.shape[1:])
    for first in range(0, kernel.shape[1], 512):
        part = kernel[:, first : first + 512]  # noqa: E203
        if count <= 64:
            summed = np.array(
                [weights[: m + 1][::-1] @ part[: m + 1] for m in range(count)]
            )
        else:
            summed = fftconvolve(weights[:, None], part, axes=0)[:count]
        out[1:, first : first + 512] = summed  # noqa: E203
    return out


class TestedCurve:
    """A tested unit's point availability and expected events from 0 (see
    ``_point_availability``). Its tests are at ``base + n T`` for ``n >=
    k0``, as the simulation schedules them (``k0`` 0 or 1), the ``k``-th at
    ``n = k + k0 - 1``, which finds every failure if ``n`` is a multiple
    of the full tests' ``per``. Before the first test (at ``first``) the
    unit is as ``head_up`` and ``head_failures`` say (its failures from 0);
    in the interval after the ``k``-th, as its ``k``-th row says (see
    ``_Rows``), the rows from ``settle`` on repeating the long run's, by the
    test's place in the period. ``first_restart`` puts the unit in a repair
    at 0 back into service (on the frame from the test before 0: its lag
    ``k`` is the ``k``-th interval)."""

    def __init__(
        self,
        unit: "TestedUnit",
        rows: _Rows,
        base: float,
        k0: int,
        head_up: Callable,
        head_failures: Callable,
        settled: bool,
        first_restart: Optional[_Restart] = None,
    ):
        setup = unit.setup
        self.setup = setup
        self.interval = T = setup.interval
        self.base, self.k0 = float(base), int(k0)
        self.first = self.base + self.k0 * T
        self.k1, self.per = self.k0 % unit.per, unit.per
        self.head_up, self.head_failures = head_up, head_failures
        self.first_restart = first_restart
        self.timed = setup.timed
        long_run = unit.long_run.rows
        #: Row 0 stands for the head (unused); rows 1 to ``computed``, then
        #: the long run's by position.
        self.computed = len(rows.tested)
        empty = _Rows(
            np.zeros(1),
            np.zeros((1, setup.restart.lags)),
            np.zeros((1, setup.steps + 1)),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
        )
        self.rows = _stacked(empty, rows, long_run)
        self.settled = settled
        if settled:
            self.settle = self.first + self.computed * T
            self.period: Optional[float] = self.per * T
        else:
            self.settle = np.inf
            self.period = None
        self.head_total = float(
            np.ravel(head_failures(np.array([self.first])))[0]
        )

    def _index(self, k: np.ndarray) -> np.ndarray:
        """The row of the ``k``-th interval (``k >= 1``)."""
        k = np.asarray(k, dtype=np.int64)
        if not self.settled:
            return np.minimum(k, max(self.computed, 1))
        position = (self.k1 + k - 1) % self.per
        return np.where(k <= self.computed, k, self.computed + 1 + position)

    def _test_time(self, k) -> np.ndarray:
        """When the ``k``-th test is."""
        k = np.asarray(k)
        return self.base + (k + self.k0 - 1) * self.interval

    def _count(self, x: np.ndarray, strictly: bool) -> np.ndarray:
        """How many tests fall at or before (or strictly before) each
        ``x``, against their times as scheduled."""
        T = self.interval
        n = np.floor((x - self.base) / T)
        time = self.base + n * T
        n = np.where(time >= x if strictly else time > x, n - 1.0, n)
        later = self.base + (n + 1.0) * T
        n = np.where(later < x if strictly else later <= x, n + 1.0, n)
        return np.maximum(n - self.k0 + 1.0, 0.0).astype(np.int64)

    def _place(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """For times from the first test on: the interval each falls in,
        from 1, and the time since its test."""
        k = np.maximum(self._count(x, False), 1)
        s = x - self._test_time(k)
        return k, np.clip(s, 0.0, self.interval)

    def _tests_before(self, x: np.ndarray) -> np.ndarray:
        """How many tests fall strictly before each ``x``."""
        return self._count(x, True)

    def _whole(self, values: np.ndarray, count: np.ndarray) -> np.ndarray:
        """The sum of ``values`` (one per row of ``rows``) over the
        intervals 1 to ``count``."""
        count = np.asarray(count, dtype=np.int64)
        n = self.computed
        running = np.concatenate([[0.0], np.cumsum(values[1 : n + 1])])
        inside = running[np.clip(count, 0, n)]
        if not self.settled:
            return inside
        later = np.maximum(count - n, 0)
        lr = values[n + 1 :]  # noqa: E203
        # The long run's rows by position, from the interval after the
        # last computed one.
        start = (self.k1 + n) % self.per
        order = lr[(start + np.arange(self.per)) % self.per]
        cycle = np.concatenate([[0.0], np.cumsum(order)])
        whole, rest = np.divmod(later, self.per)
        return inside + whole * cycle[-1] + cycle[rest]

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        out = np.empty(flat.shape)
        head = flat < self.first
        if head.any():
            out[head] = self.head_up(flat[head])
        if (~head).any():
            k, s = self._place(flat[~head])
            out[~head] = _values_at(
                self.setup,
                self.rows,
                self._index(k),
                s,
                self.first_restart,
                k,
            )[0]
        return np.clip(out, 0.0, 1.0).reshape(x.shape)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x`` (see
        ``GridCurve.events``): its failures; the tests that take it off
        line, working (planned outages), with a time model; the failures
        the tests find (each repaired, a corrective action); and the tests
        done (none falls in a repair)."""
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        failures = np.asarray(
            self.head_failures(np.minimum(flat, self.first)), dtype=float
        ).copy()
        later = flat >= self.first
        if later.any():
            k, s = self._place(flat[later])
            rows = self.rows
            within = _interpolated(
                rows.failures, self._index(k), s, self.setup.h
            )
            ends = rows.failures[:, -1]
            failures[later] = (
                self.head_total + self._whole(ends, k - 1) + within
            )
        tests = self._tests_before(flat)
        planned = self._whole(self.rows.tested, tests) if self.timed else None
        shape = x.shape
        return _events(
            np.maximum(failures, 0.0).reshape(shape),
            None if planned is None else planned.reshape(shape),
            corrective=self._whole(self.rows.found, tests).reshape(shape),
            inspections=self._whole(self.rows.inspected, tests).reshape(shape),
        )

    def atoms(self, stop: float) -> Atoms:
        """The tests before ``stop`` that take the unit off line (with a
        time model): each a planned outage of the unit working at it."""
        if not self.timed:
            return Atoms.none()
        count = int(self._tests_before(np.array([float(stop)]))[0])
        if count < 1:
            return Atoms.none()
        k = np.arange(1, count + 1)
        times = self._test_time(k)
        planned = self.rows.tested[self._index(k)]
        drop = planned * (1.0 - float(self.setup.G(np.zeros(1))[0]))
        zero = np.zeros(count)
        return Atoms(times, zero, planned, zero, drop)

    def _offsets(self, within: np.ndarray, start: float, stop: float):
        if stop < self.first:
            return np.empty(0)
        lo = max(int(self._count(np.array([start]), False)[0]), 1)
        hi = int(self._count(np.array([stop]), False)[0])
        if hi < lo:
            return np.empty(0)
        tests = self._test_time(np.arange(lo, hi + 1))
        times = (tests[:, None] + within[None, :]).ravel()
        return times[(times >= start) & (times <= stop)]

    def _head_breaks(self, start: float, stop: float) -> np.ndarray:
        parts = [np.array([0.0, self.first])]
        # Closing in on 0, where a new unit's life may not be smooth.
        parts.append(self.first * 2.0 ** -np.arange(40.0, 0.0, -1.0))
        if self.first_restart is not None:
            frame = self.interval - self.first
            parts.append(self.first_restart.knots(0) - frame)
        times = np.concatenate(parts)
        return times[(times >= start) & (times <= stop) & (times >= 0.0)]

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The times in ``[start, stop]`` between which the curve is
        smooth: the tests, the grid's points after each, and where the
        test's and the restarts' CDFs bend."""
        head = np.linspace(0.0, self.first, 65)
        head = head[(head >= start) & (head <= stop)]
        return np.unique(
            np.concatenate(
                [
                    head,
                    self._head_breaks(start, stop),
                    self._offsets(_within(self.setup), start, stop),
                ]
            )
        )

    def breaks(self, start: float, stop: float) -> np.ndarray:
        """The times in ``[start, stop]`` at which the curve bends other
        than at its grid's points (see ``curve_breaks``): the tests, and
        where the test's and the restarts' CDFs bend after each."""
        setup = self.setup
        restart = setup.restart
        parts: List[np.ndarray] = [np.zeros(1), setup.test_knots]
        parts += [restart.knots(lag) for lag in range(restart.lags)]
        within = np.unique(np.concatenate(parts))
        within = within[(within >= 0.0) & (within < self.interval)]
        return np.unique(
            np.concatenate(
                [
                    self._head_breaks(start, stop),
                    self._offsets(within, start, stop),
                ]
            )
        )

    def grids(self) -> list:
        """Its grid, in every interval (see ``curve_grids``)."""
        return [(self.setup.h, np.inf)]


def _padded(values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros((length,) + values.shape[1:])
    n = min(length, len(values))
    out[:n] = values[:n]
    return out


class TestedUnit:
    """A unit with hidden failures, tested every ``interval`` (see the
    module docstring), each test finding a failure with probability
    ``coverage``, but every ``per``-th, which finds every one: its long run
    (``long_run``) and its values over time (``curve``). ``rate`` is its
    failure rate, for an exponential life (its chain needs no ages), or
    None."""

    def __init__(
        self,
        life,
        repair,
        test,
        interval: float,
        coverage: float = 1.0,
        per: int = 1,
        node=None,
        rate: Optional[float] = None,
    ):
        self.setup = _Setup(life, repair, test, interval, node)
        self.coverage = float(coverage)
        self.per = max(1, int(per)) if self.coverage < 1.0 else 1
        self.rate = rate
        self._long_run: Optional[TestedLongRun] = None
        self._cycle: Optional[_Cycle] = None

    def renewals(
        self, first: float, position: int = 0, instead=_UNSUPPORTED
    ) -> Tuple[Callable[[int], np.ndarray], np.ndarray]:
        """Its replacements as the spares count them, one at each test that
        finds a failure, whatever the test and the repair then take (see
        ``_spares.Tested``): the chances that the unit new at 0, first
        tested at ``first``, is found failed at each of the first ``n``
        tests (``found(n)``), and that a unit put back into service after a
        test that finds a failure is found failed ``1, 2, ...`` tests after
        it (``cycle``). With tests that can miss a failure, ``cycle`` has a
        row for each place in the full tests' period of the test that
        found the failure, and ``position`` is the first test's place.
        ``instead`` ends a refusal.

        Raises
        ------
        NotImplementedError
            If a unit can last more than ``MAX_RENEWAL_TESTS`` tests.
        """
        # The failures in each interval: the first unit's, from its first
        # test (those before it found or missed there), and a cycle's from
        # the test that finds a failure.
        if self.rate is not None:
            first_fail, fail = self._exponential_renewals(first)
        else:
            fail = self._regular().fail
            start = _walk(self.setup, None, alive=(float(first), 1.0))
            working = float(start.p[1]) if len(start.p) > 1 else 0.0
            first_fail = np.concatenate([[1.0 - working], start.fail[1:]])
        if max(len(fail), len(first_fail)) + self.per > MAX_RENEWAL_TESTS:
            raise _too_long(
                self.setup.node,
                "its life lasts so long against its test interval that its "
                f"replacements would be followed over more than "
                f"{MAX_RENEWAL_TESTS} tests",
                instead,
            )
        first_fail = np.maximum(first_fail, 0.0)
        fail = np.maximum(fail, 0.0)
        if self.per == 1:
            first_found, cycle = first_fail, fail / fail.sum()
        else:
            per, coverage = self.per, self.coverage
            places = (int(position) + np.arange(len(first_fail))) % per
            first_found = _found(first_fail, places, coverage, per)[0][1:]
            lags = np.arange(1, len(fail) + 1)
            rows = []
            for q in range(per):
                found_q = _found(fail, (q + lags) % per, coverage, per)[0]
                rows.append(found_q[1:] / found_q.sum())
            cycle = np.array(rows)

        def found(n: int) -> np.ndarray:
            return _padded(first_found, int(n))

        return found, cycle

    def _exponential_renewals(self, first: float):
        """The failures in each interval for ``renewals``, for an
        exponential life: from the test that finds a failure, the unit is
        put back into service in each interval with the restart's chance,
        and a working one is working at the next test with ``kappa``, the
        chance of lasting an interval less its test."""
        setup = self.setup
        chain = _Chain(setup, 1.0, 1)
        kappa = chain.kappa
        share = setup.restart.masses.sum(axis=1)
        fail = []
        p = 0.0
        for lag in range(chain.lags):
            fail.append(p * (1.0 - kappa) + share[lag] - chain.a[lag])
            p = p * kappa + chain.a[lag]

        def geometric(start: float) -> np.ndarray:
            # The failures between later tests, kappa less each time.
            if start <= NEGLIGIBLE or kappa <= 0.0:
                return np.full(1, start * (1.0 - kappa))
            if kappa >= 1.0:
                return np.zeros(MAX_RENEWAL_TESTS + 1)
            n = int(np.ceil(np.log(NEGLIGIBLE / start) / np.log(kappa))) + 1
            n = min(n, MAX_RENEWAL_TESTS + 1)
            return start * kappa ** np.arange(n) * (1.0 - kappa)

        tail = geometric(p) if p > 0.0 else np.empty(0)
        regular = np.concatenate([fail, tail])
        working = float(
            np.ravel(_values(setup.life.sf, np.array([float(first)])))[0]
        )
        start = np.concatenate([[1.0 - working], geometric(working)])
        return start, regular

    def _regular(self, keep: int = 0) -> _Cycle:
        """The cycle from a test that finds a failure, with its first
        ``keep`` lags' failures."""
        if self._cycle is None or len(self._cycle.failures) < keep:
            self._cycle = _walk(
                self.setup, self.setup.restart, keep=keep, per=self.per
            )
        return self._cycle

    @property
    def long_run(self) -> TestedLongRun:
        if self._long_run is None:
            if self.rate is not None:
                chain = _Chain(self.setup, self.coverage, self.per)
                rows = _chain_rows(chain)
            else:
                rows = _cycle_rows(
                    self.setup, self._regular(), self.coverage, self.per
                )[0]
            self._long_run = TestedLongRun(self.setup, rows, self.per)
        return self._long_run

    def curve(
        self, horizon: float, base: float, k0: int, start=None
    ) -> TestedCurve:
        """Its point availability and events from 0, over ``[0, horizon]``
        (``inf``: until it repeats the long run), tested at ``base + n T``
        for ``n >= k0`` (see ``TestedCurve``); from ``start``: None, new at
        0; ``("alive", age, since)``, up at 0 of age ``age``, known up
        ``since`` before 0 (at its last test, or when put into service);
        ``("down", down_for)``, in a repair under way for ``down_for``; or
        ``"stationary"``, in its long-run state.

        Raises
        ------
        NotImplementedError
            If it has not repeated the long run within the intervals the
            grid can keep, when the horizon is longer.
        """
        setup = self.setup
        T, h = setup.interval, setup.h
        long_run = self.long_run
        first = float(base) + int(k0) * T
        k1 = int(k0) % self.per
        if start is not None and start[0] == "stationary":
            # Its last test was T - first before 0.
            phase = T - first
            rows = long_run.rows

            def head_up(x):
                s = phase + np.asarray(x, dtype=float)
                return _values_at(setup, rows, np.zeros(len(s), int), s)[0]

            before = float(_interpolated(rows.failures, [0], [phase], h)[0])

            def head_failures(x):
                s = phase + np.asarray(x, dtype=float)
                index = np.zeros(len(s), dtype=int)
                return _interpolated(rows.failures, index, s, h) - before

            empty = _rows_of([], setup.restart.lags)
            return TestedCurve(
                self, empty, base, k0, head_up, head_failures, True
            )
        cap = max(2, _MAX_VALUES // (setup.steps + 1))
        if np.isfinite(horizon):
            count = int(np.floor(max(horizon - first, 0.0) / T)) + 1
        else:
            count = cap
        needed = count
        count = min(count, cap)
        life = setup.life
        first_restart = None
        alive = None
        if start is None:
            alive = (float(first), 1.0, 0.0)
        elif start[0] == "alive":
            age, since = float(start[1]), float(start[2])
            known = float(
                np.ravel(_values(life.sf, np.array([age - since])))[0]
            )
            alive = (age + float(first), 1.0 / known, age)
        else:
            frame = T - float(first)
            repair = setup.fix
            if repair.fixed is not None:
                first_restart = _Restart.at(
                    setup, frame + max(repair.fixed - float(start[1]), 0.0)
                )
            else:
                first_restart = _Restart.after(
                    setup, frame, _Left(repair.model, float(start[1]))
                )
        if alive is not None:
            at_first, weight, age0 = alive

            def head_up(x):
                ages = age0 + np.asarray(x, dtype=float)
                return np.clip(weight * _values(life.sf, ages), 0.0, 1.0)

            failed0 = float(np.ravel(_values(life.ff, np.array([age0])))[0])

            def head_failures(x):
                ages = age0 + np.asarray(x, dtype=float)
                return np.maximum(
                    weight * (_values(life.ff, ages) - failed0), 0.0
                )

            # The first test finds every failure since it was last known
            # up: before 0, and before the test.
            working = weight * float(
                np.ravel(_values(life.sf, np.array([at_first])))[0]
            )
            pre = max(1.0 - working, 0.0)
        else:
            assert first_restart is not None
            frame = T - float(first)
            pre = 0.0
        positions = (k1 + np.arange(count)) % self.per
        if self.rate is not None:
            chain = _Chain(setup, self.coverage, self.per, first_restart)
            if alive is not None:
                x = chain.state(working, pre)
                head_rows = None
            else:
                x = chain.state(float(chain.a0[0]), float(chain.rf0[0][-1]))
                head_rows = chain.rf0[0]
            rows, settle = chain.rows(
                x, positions, first_lag=1, settled=chain.long_run()[1]
            )
        else:
            if alive is not None:
                first_cycle = _walk(
                    setup, None, alive=(at_first, weight), keep=count + 1
                )
            else:
                first_cycle = _walk(setup, first_restart, keep=count + 1)
            head_rows = first_cycle.failures[0]
            rows, bound = self._walk_rows(
                first_cycle, count, k1, pre, first_restart
            )
            # From the first row whose bound, and the next ``per`` rows',
            # is within ``SETTLED`` (once the first unit's repair is over).
            after = 0 if first_restart is None else first_restart.lags
            close = bound <= SETTLED
            run = np.convolve(close, np.ones(self.per), "valid")
            ready = np.flatnonzero(run >= self.per)
            ready = ready[ready >= after]
            settle = int(ready[0]) if ready.size else None
        if alive is None:
            assert head_rows is not None
            failures0 = head_rows[None, :]
            before = float(_interpolated(failures0, [0], [frame], h)[0])

            def head_up(x):
                s = frame + np.asarray(x, dtype=float)
                index = np.zeros(len(s), dtype=int)
                up = first_restart.cdf(0, s)
                return np.clip(
                    up - _interpolated(failures0, index, s, h), 0.0, 1.0
                )

            def head_failures(x):
                s = frame + np.asarray(x, dtype=float)
                index = np.zeros(len(s), dtype=int)
                return np.maximum(
                    _interpolated(failures0, index, s, h) - before, 0.0
                )

        if settle is not None:
            rows = _Rows(*(part[:settle] for part in rows))
        elif needed > count or not np.isfinite(horizon):
            raise _too_long(
                setup.node,
                "its values over time have not settled into their long run "
                f"within {count} test intervals, as far as the grid can "
                "follow them",
            )
        return TestedCurve(
            self,
            rows,
            base,
            k0,
            head_up,
            head_failures,
            settle is not None,
            first_restart,
        )

    def _walk_rows(
        self,
        first_cycle: _Cycle,
        count: int,
        k1: int,
        pre: float,
        first_restart: Optional[_Restart],
    ) -> Tuple[_Rows, np.ndarray]:
        """The rows of the intervals after the first ``count`` tests, from
        the first unit's cycle (on the frame from the test before 0) and the
        cycles the tests that find failures start: those tests are a
        renewal process on the tests (see the module docstring); and for
        each row, a bound on how far it is from the long run's."""
        setup, per, coverage = self.setup, self.per, self.coverage
        restart = setup.restart
        cycle = self._regular(keep=count)
        n = count + 1
        fail0 = _padded(first_cycle.fail, n)
        fail0[0] += pre
        # The test that ends the first unit's l-th interval is the
        # (l + 1)-th, at (k1 + l) in the period.
        first_found, first_missed, _ = _found(
            fail0, (k1 + np.arange(n)) % per, coverage, per
        )
        found_by, missed, _ = _detections(cycle, coverage, per, n)
        position = (k1 + np.arange(n) - 1) % per  # of the m-th test
        r = np.zeros(n)
        for m in range(1, n):
            j = np.arange(1, m)
            r[m] = first_found[m] + float(
                r[1:m] @ found_by[position[1:m], m - j]
            )
        p = cycle.p.copy()
        p[0] = 0.0
        tested = _padded(first_cycle.p, n) + _lagged(r, p, count)
        failures = _padded(first_cycle.failures, n) + _lagged(
            r, cycle.failures, count
        )
        waiting = _padded(first_missed, n)
        for q in range(per):
            rq = np.where(position == q, r, 0.0)
            waiting = waiting + _lagged(rq, missed[q], count)
        pend = restart.pending(np.arange(n)).astype(float)
        pend[0] = 0.0
        inspected = 1.0 - _lagged(r, pend, count)
        if first_restart is not None:
            inspected = inspected - first_restart.pending(np.arange(n))
        coef = np.zeros((n, restart.lags))
        for back in range(min(restart.lags, n - 1)):
            coef[back + 1 :, back] = r[1 : n - back]  # noqa: E203
        rows = _Rows(
            tested[1:],
            coef[1:],
            failures[1:],
            waiting[1:],
            inspected[1:],
            r[1:],
        )
        # How far each row can be from the long run's: what is left of the
        # first unit's cycle, the tests that found failures as far from
        # their long-run chance, each times what is left of its cycle that
        # many tests on, and the long run's tests before 0, which the start
        # leaves out.
        settled = self.long_run.rows.found
        whole = len(cycle.p) + per + 1
        pend_all = restart.pending(np.arange(whole)).astype(float)
        p_all = _padded(cycle.p, whole)
        _, missed_all, _ = _detections(cycle, coverage, per, whole)
        left = pend_all + p_all + missed_all.max(axis=0)
        tail = np.concatenate([np.cumsum(left[::-1])[::-1], [0.0]])
        first_left = _padded(first_cycle.p, n) + _padded(first_missed, n)
        if first_restart is not None:
            first_left = first_left + first_restart.pending(np.arange(n))
        gap = np.abs(r - settled[position])
        gap[0] = 0.0
        bound = (
            first_left
            + _lagged(gap, left, count)
            + settled.max() * tail[np.minimum(np.arange(n), whole)]
        )
        return rows, bound[1:]
