"""Hidden failures with any life (#144): a unit tested every ``interval``,
in no time, and renewed, in no time, at the test that finds it failed.

Renewals then fall on test instants. A unit renewed at a test is up ``m``
intervals later with probability ``R(m T)``, and is found failed at the
first test after it fails (the next, if it fails at once), so a cycle lasts
``S = 1 + sum_{m >= 1} R(m T)`` intervals on average. In the long run, just
after a test, the unit in service was renewed ``m >= 1`` intervals before
with probability ``R(m T) / S``, and at that test with ``1 / S``; at a time
``u`` into an interval it is up with probability

``A(u) = (R(u) + sum_{m >= 1} R(m T + u)) / S``,

down with ``(F(u) + sum_{m >= 1} (R(m T) - R(m T + u))) / S`` (each
difference worked out from the model's ``ff`` while that is the smaller,
so a small one keeps its precision), and fails at the rate ``sum_m f(m T +
u) / S``; it fails, and is renewed, ``1 / (S T)`` times per unit time. A
constant failure rate ``lambda`` gives ``A(u) = exp(-lambda u)``, as the
closed form does.

From new, or from a known state, the renewals at the tests are a discrete
renewal process on the test lattice (``TestedLifeCurve``): the chance of a
renewal at the ``n``-th test is the chance that the unit in service then
failed since the last one, summed over when it was put in service. They
settle to ``1 / S`` a test, and the unit's point availability to ``A``.

The sums run over the intervals a unit can last (while ``R(m T)`` is at
least ``NEGLIGIBLE``). Within an interval, each term but the unit renewed
at the last test's (``m = 0``) is smooth in the time since it wherever the
life's survival function is smooth: those terms are worked out once, at
Chebyshev points across an interval, and the polynomial through them,
exact to rounding, gives them anywhere in it. Each is checked at points
between them, and one that is not smooth there (a life with a threshold,
say) is summed directly instead. So a curve costs little more to evaluate
at many times than at one.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ._point_availability import Atoms, _events
from ._point_availability import knots as _model_knots
from ._point_availability import multiples_before

#: A unit's chance of lasting so many intervals below which the rest of its
#: life is left out of the sums (they are then exact to rounding).
NEGLIGIBLE = 1e-17

#: The most test intervals a life is summed over, or the renewals from new
#: are followed through, before the exact values refuse.
MAX_INTERVALS = 200_000

#: How close the renewals from new must come to the long run (in the
#: unit's point availability, and each renewal relative to its long-run
#: value) for the curve to be taken to repeat it from then on.
SETTLED = 1e-13

# Gauss-Legendre nodes and weights on [0, 1].
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(16)
_NODES, _WEIGHTS = 0.5 * (_NODES + 1.0), 0.5 * _WEIGHTS

# Chebyshev points across an interval (as fractions of it, with its ends),
# the barycentric weights of the polynomial through them, and the points
# halfway between them (in angle) at which it is checked.
_DEGREE = 32
_CHEB = 0.5 * (1.0 - np.cos(np.pi * np.arange(_DEGREE + 1) / _DEGREE))
_BARY = (-1.0) ** np.arange(_DEGREE + 1)
_BARY[[0, -1]] *= 0.5
_CHECK = 0.5 * (1.0 - np.cos(np.pi * (np.arange(_DEGREE) + 0.5) / _DEGREE))


def _clenshaw_curtis(n: int) -> np.ndarray:
    """The weights on [0, 1] of the values at the Chebyshev points that
    integrate the polynomial through them exactly."""
    j = np.arange(n + 1)
    w = np.ones(n + 1)
    for k in range(1, n // 2 + 1):
        b = 1.0 if 2 * k == n else 2.0
        w -= b / (4.0 * k * k - 1.0) * np.cos(2.0 * k * j * np.pi / n)
    c = np.full(n + 1, 2.0)
    c[[0, -1]] = 1.0
    return c * w / (2.0 * n)


_CC = _clenshaw_curtis(_DEGREE)

# Points at a time when a curve is interpolated.
_CHUNK = 65_536


def _values(function: Callable, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        out = np.asarray(function(x), dtype=float).reshape(x.shape)
    return np.nan_to_num(out, nan=0.0)


def _too_long(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"{what}: estimate the exact values by simulation, with "
        "availability() or cost()."
    )


def _interpolation_weights(position: np.ndarray) -> np.ndarray:
    """For each ``position`` (a fraction of an interval), the weights of
    the values at the Chebyshev points that give the polynomial through
    them there."""
    diff = position[:, None] - _CHEB[None, :]
    exact = diff == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        w = _BARY / diff
    hit = exact.any(axis=1)
    if hit.any():
        w[hit] = exact[hit].astype(float)
    return w / w.sum(axis=1, keepdims=True)


def _gauss(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre points and weights on each cell between ``edges``."""
    a, b = edges[:-1], edges[1:]
    x = a[:, None] + (b - a)[:, None] * _NODES[None, :]
    return x.ravel(), ((b - a)[:, None] * _WEIGHTS[None, :]).ravel()


class TestedLife:
    """The long run of a unit of life ``model`` tested every ``interval``
    and renewed at the test that finds it failed (see the module
    docstring).

    Raises
    ------
    NotImplementedError
        If the life lasts so long against the interval that more than
        ``MAX_INTERVALS`` intervals are needed to sum it.
    """

    def __init__(self, model, interval: float):
        self.model = model
        self.interval = float(interval)
        # The first interval is always part of a cycle: its "survival" is 1.
        survive, failed = [np.ones(1)], [np.zeros(1)]
        start, size = 1, 64
        while survive[-1][-1] >= NEGLIGIBLE:
            if start >= MAX_INTERVALS:
                raise _too_long(
                    "The life lasts too long against the test interval for "
                    "the long-run values of its tests to be summed"
                )
            m = self.interval * np.arange(start, start + size)
            survive.append(np.clip(_values(model.sf, m), 0.0, 1.0))
            failed.append(np.clip(_values(model.ff, m), 0.0, 1.0))
            start, size = start + size, 2 * size
        up = np.concatenate(survive)
        keep = int(np.argmax(up < NEGLIGIBLE)) or len(up)
        #: The chance that a cycle lasts beyond ``m`` intervals (``R(m T)``,
        #: 1 for ``m = 0``), and its complement, for the intervals the life
        #: is summed over.
        self.up = up[:keep]
        self.down = np.concatenate(failed)[:keep]
        #: The mean number of intervals in a cycle.
        self.cycle = float(self.up.sum())
        # The shape of the profile across an interval, for quadrature.
        rate = self.rate
        self.pieces = int(
            min(64, max(2, math.ceil(8.0 * rate * self.interval)))
        )
        self._smooth: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._means: Optional[Tuple[float, float]] = None

    @property
    def rate(self) -> float:
        """The profile's mean rate of decay over an interval:
        ``A(T) = 1 - 1 / S``, so ``-log(1 - 1 / S) / T`` (the failure rate,
        for a constant one)."""
        if self.cycle <= 1.0 + 1e-12:
            # It fails within an interval for certain: a fast enough rate
            # for the grid, which then needs no finer pieces.
            return 64.0 / self.interval
        return float(-np.log1p(-1.0 / self.cycle) / self.interval)

    @property
    def failures(self) -> float:
        """Failures (and renewals) per unit time: ``1 / (S T)``."""
        return 1.0 / (self.cycle * self.interval)

    def drops(self, k: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """For the units renewed ``k >= 1`` intervals before a test (rows),
        their chances of failing between it and each ``phase`` after it
        (columns), ``R(k T) - R(k T + u)``: from the model's ``ff`` while
        that is the smaller, so a small one keeps its precision."""
        k = np.asarray(k, dtype=int)
        phase = np.asarray(phase, dtype=float)
        x = self.interval * k[:, None] + phase[None, :]
        out = np.empty(x.shape)
        small = self.down[k] <= 0.5
        if small.any():
            out[small] = (
                _values(self.model.ff, x[small]) - self.down[k[small], None]
            )
        if (~small).any():
            out[~small] = self.up[k[~small], None] - _values(
                self.model.sf, x[~small]
            )
        return np.maximum(out, 0.0)

    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """The units renewed ``k = 1, 2, ...`` intervals before a test: for
        each (a row), its ``drops`` at the Chebyshev points across an
        interval, the row left at 0 for one the polynomial through them
        does not give to rounding between them; and those ``k`` (summed
        directly)."""
        if self._smooth is None:
            rows = np.arange(1, len(self.up))
            table = np.zeros((len(rows), _DEGREE + 1))
            rough = np.zeros(len(rows), dtype=bool)
            check = _interpolation_weights(_CHECK)
            nodes = self.interval * _CHEB
            between = self.interval * _CHECK
            for first in range(0, len(rows), 8192):
                k = rows[first : first + 8192]  # noqa: E203
                values = self.drops(k, nodes)
                error = np.abs(values @ check.T - self.drops(k, between))
                # The rows' scale (their drop over a whole interval), and
                # the rounding in their values (the model's own, in a far
                # tail, as well as the difference's).
                allowed = 1e-13 * (
                    values[:, -1] + np.minimum(self.down[k], self.up[k])
                )
                bad = ~(error.max(axis=1) <= allowed)
                table[first : first + len(k)] = np.where(  # noqa: E203
                    bad[:, None], 0.0, values
                )
                rough[first : first + len(k)] = bad  # noqa: E203
            self._smooth = (table, rows[rough])
        return self._smooth

    def _bends(self, k: int) -> np.ndarray:
        """Where across an interval the unit renewed ``k`` intervals before
        may bend: its life's knots that fall there."""
        local = _model_knots(self.model) - k * self.interval
        return local[(local > 0.0) & (local < self.interval)]

    @property
    def bends(self) -> np.ndarray:
        """Where across an interval the terms summed directly (see
        ``smooth``) may bend, for quadrature to split at: a life with a
        threshold, say."""
        _, rough = self.smooth()
        if not len(rough):
            return np.empty(0)
        return np.unique(np.concatenate([self._bends(int(k)) for k in rough]))

    def _rough_cells(self, k: int) -> np.ndarray:
        """The cells across an interval on which the drop of the unit
        renewed ``k`` intervals before is integrated directly: split where
        its life may bend, and finely."""
        return np.unique(
            np.concatenate(
                [self.interval * np.arange(65) / 64.0, self._bends(k)],
            )
        )

    def drop_sum(self, phase: np.ndarray) -> np.ndarray:
        """``sum_{m >= 1} (R(m T) - R(m T + u))`` at each ``phase`` ``u``
        (in ``[0, T]``)."""
        phase = np.asarray(phase, dtype=float)
        flat = phase.ravel()
        table, rough = self.smooth()
        column = table.sum(axis=0)
        out = np.empty(flat.shape)
        for first in range(0, len(flat), _CHUNK):
            part = flat[first : first + _CHUNK]  # noqa: E203
            weights = _interpolation_weights(part / self.interval)
            out[first : first + _CHUNK] = weights @ column  # noqa: E203
        for k in rough:
            out = out + self.drops(np.array([k]), flat)[0]
        return np.maximum(out, 0.0).reshape(phase.shape)

    def profile(self, phase) -> Tuple[np.ndarray, np.ndarray]:
        """The probabilities of being up and down at each ``phase``, the
        time since the last test, in the long run (the down one worked out
        in its own right)."""
        phase = np.asarray(phase, dtype=float)
        drop = self.drop_sum(phase)
        later = float(self.up[1:].sum())
        up = _values(self.model.sf, phase) + later - drop
        down = _values(self.model.ff, phase) + drop
        return (
            np.clip(up / self.cycle, 0.0, 1.0),
            np.clip(down / self.cycle, 0.0, 1.0),
        )

    def intensity(self, phase) -> np.ndarray:
        """The rate of failing at each ``phase``, in the long run:
        ``sum_m f(m T + u) / S``."""
        phase = np.asarray(phase, dtype=float)
        flat = phase.ravel()
        total = np.zeros(flat.shape)
        rows = max(1, 2_000_000 // max(1, flat.size))
        for first in range(0, len(self.up), rows):
            m = np.arange(first, min(first + rows, len(self.up)))
            x = self.interval * m[:, None] + flat[None, :]
            total += np.maximum(_values(self.model.df, x), 0.0).sum(axis=0)
        return (total / self.cycle).reshape(phase.shape)

    def _integrals(self) -> Tuple[float, float]:
        """The integrals over an interval of the up and down profiles,
        times ``S``. The unit renewed at the last test is integrated on
        cells that close in on the test geometrically (where its life may
        not be smooth: a Weibull's, say) and split at its knots; the rest
        exactly from their Chebyshev points (Clenshaw-Curtis), but for the
        rough ones, integrated directly."""
        T = self.interval
        knots = _model_knots(self.model)
        edges = np.concatenate(
            [
                [0.0],
                T * 2.0 ** -np.arange(64.0, 0.0, -1.0),
                T * np.arange(1, self.pieces + 1) / self.pieces,
                knots[(knots > 0.0) & (knots < T)],
            ]
        )
        x, w = _gauss(np.unique(edges))
        first_up = float(w @ np.clip(_values(self.model.sf, x), 0.0, 1.0))
        first_down = float(w @ np.clip(_values(self.model.ff, x), 0.0, 1.0))
        table, rough = self.smooth()
        drop = T * float(_CC @ table.sum(axis=0))
        for k in rough:
            x, w = _gauss(self._rough_cells(int(k)))
            drop += float(w @ self.drops(np.array([k]), x)[0])
        later = T * float(self.up[1:].sum())
        return first_up + later - drop, first_down + drop

    @property
    def availability(self) -> float:
        """The mean availability over an interval, in the long run (the
        mean life over the mean cycle, ``E[X] / (S T)``)."""
        if self._means is None:
            self._means = self._integrals()
        return min(1.0, self._means[0] / (self.cycle * self.interval))

    @property
    def unavailability(self) -> float:
        """The mean unavailability over an interval, in the long run (the
        PFDavg of a function with this unit alone), worked out in its own
        right."""
        if self._means is None:
            self._means = self._integrals()
        return min(1.0, self._means[1] / (self.cycle * self.interval))

    def knots(self, start: float, stop: float, offset: float) -> np.ndarray:
        """Test instants (at ``offset`` and every interval from it) in
        ``[start, stop]``, with points between them close enough for
        quadrature; closing in on each test geometrically, where the unit
        renewed there may not age smoothly (a Weibull life's survival, say),
        unless it fails so rarely in an interval that it does not matter."""
        first = int(np.floor((start - offset) / self.interval))
        last = int(np.floor((stop - offset) / self.interval))
        within = np.arange(self.pieces) / self.pieces
        if len(self.down) < 2 or self.down[1] > 1e-9:
            within = np.concatenate(
                [within, 2.0 ** -np.arange(24.0, 0.0, -1.0) / self.pieces]
            )
        within = np.concatenate([within, self.bends / self.interval])
        times = (
            offset
            + self.interval
            * (np.arange(first, last + 1)[:, None] + within[None, :])
        ).ravel()
        return times[(times >= start) & (times <= stop)]


class TestedLifeCurve:
    """A unit tested and renewed as in ``TestedLife``, from 0: its point
    availability and expected events, as the over-time analyses take them
    (see ``_point_availability``).

    It is tested first at ``first`` and every interval after. The unit in
    service at 0 is up at ``x`` with probability ``initial(x)`` and has
    failed by ``x`` with ``initial_down(x)`` (worked out in its own right),
    since it was last known up (by default it is new at 0). Each renewal at
    a test starts a unit afresh, so the chance ``r_n`` of a renewal at the
    ``n``-th test is the chance that the first unit failed since the test
    before, plus the chance of a renewal at an earlier test ``j`` times
    that of a unit failing in its ``n - j``-th interval:

    ``r_n = h_n + sum_j r_j (F((n - j) T) - F((n - j - 1) T))``,

    and the unit is up at ``x`` with probability ``initial(x) + sum_j r_j
    R(x - t_j)``, over the tests ``t_j <= x``. The renewals settle to ``1 /
    S`` a test, and the curve to the long-run profile: they are followed
    until the curve is within ``SETTLED`` of it, and from then on
    (``settle``) it repeats the profile every interval.

    Raises
    ------
    NotImplementedError
        If the renewals take more than ``MAX_INTERVALS`` tests to settle.
    """

    def __init__(
        self,
        life: TestedLife,
        first: float,
        initial: Optional[Callable] = None,
        initial_down: Optional[Callable] = None,
    ):
        model = life.model
        self.life = life
        self.interval = life.interval
        self.period = life.interval
        self.first = float(first)
        self.initial = initial or (
            lambda x: np.clip(_values(model.sf, x), 0.0, 1.0)
        )
        self.initial_down = initial_down or (
            lambda x: np.clip(_values(model.ff, x), 0.0, 1.0)
        )
        # The first unit's failures before 0 (since it was last known up)
        # fall outside every window.
        self._down_at_start = float(self.initial_down(np.zeros(1))[0])
        self._renew()
        self._tables()

    def _renew(self) -> None:
        """The renewals at the tests, until they and the curve settle."""
        life, T = self.life, self.interval
        K = len(life.up)
        # The chance that a renewed unit is found failed at its k-th test.
        fails = np.where(
            life.down[:-1] <= 0.5,
            np.diff(life.down),
            life.up[:-1] - life.up[1:],
        )
        reversed_fails = np.maximum(fails, 0.0)[::-1].copy()
        steady = 1.0 / life.cycle
        # tail[m]: the long run's terms a curve lacks when only m tests
        # are behind it.
        tail = np.concatenate([np.cumsum(life.up[::-1])[::-1], [0.0]])
        block = 256
        renewals = np.zeros(1 + 4 * block)
        found = np.empty(0)
        still_up = np.empty(0)
        before = 0.0
        n = 0
        while True:
            if n >= MAX_INTERVALS:
                raise _too_long(
                    "The renewals at the tests settle too slowly for their "
                    "values over time to be followed exactly"
                )
            if n % block == 0:
                # The next block of tests: the first unit's chance of being
                # found failed at each, and of being up then.
                times = self.first + T * np.arange(n, n + block)
                down = np.clip(self.initial_down(times), 0.0, 1.0)
                found = np.maximum(
                    np.diff(np.concatenate([[before], down])), 0.0
                )
                before = float(down[-1])
                still_up = self.initial(times)
                if len(renewals) < n + block + 1:
                    renewals = np.concatenate(
                        [renewals, np.zeros(len(renewals))]
                    )
            i = n % block
            n += 1
            m = min(n - 1, K - 1)
            value = found[i]
            if m:
                value += float(
                    renewals[n - m : n]  # noqa: E203
                    @ reversed_fails[K - 1 - m :]  # noqa: E203
                )
            renewals[n] = value
            if i == block - 1:
                # How far the curve is from the long run in the interval
                # after this test, at most.
                m = min(n, K)
                recent = renewals[n - m + 1 : n + 1][::-1]  # noqa: E203
                deviation = np.abs(recent - steady)
                bound = (
                    float(still_up[i])
                    + float(deviation @ life.up[:m])
                    + steady * float(tail[m])
                )
                if (
                    bound <= SETTLED
                    and deviation[:block].max() <= SETTLED * steady
                ):
                    break
        #: The tests followed, and the chance of a renewal at each (``r_0 =
        #: 0``: none at the start).
        self.tests = n
        self.renewals = renewals[: n + 1].copy()
        #: From here the curve repeats the long-run profile.
        self.settle = self.first + (n - 1) * T

    def _tables(self) -> None:
        """For each test ``n`` (a row), what the units renewed at the tests
        before it contribute in the interval after it: their chance of
        being up at the test, ``sum_k r_{n-k} R(k T)``, of having failed by
        it, and their drops at the Chebyshev points across it. The
        renewals past those followed are taken at their long-run value,
        so from ``tests + K`` on every row is the long run's."""
        from scipy.signal import fftconvolve

        life = self.life
        K = len(life.up)
        steady = 1.0 / life.cycle
        renewed = np.concatenate([self.renewals[1:], np.full(K, steady)])
        rows = len(renewed)
        #: ``sum_{j <= n} r_j``, for ``n`` up to the rows.
        self.running = np.concatenate([[0.0], np.cumsum(renewed)])
        table, self._rough = life.smooth()

        def history(kernel: np.ndarray) -> np.ndarray:
            # Row n (from 1): sum over k = 1 .. min(n - 1, K - 1) of
            # r_{n-k} kernel[k - 1].
            if kernel.shape[0] == 0:
                return np.zeros((rows,) + kernel.shape[1:])
            if kernel.ndim == 1:
                full = fftconvolve(renewed, kernel)
            else:
                full = fftconvolve(renewed[:, None], kernel, axes=0)
            head = np.zeros((1,) + full.shape[1:])
            return np.concatenate([head, full[: rows - 1]])

        self._up = history(life.up[1:])
        self._failed = history(life.down[1:])
        self._drop = history(table)
        self._rows = rows
        self._renewed = renewed

    def _r(self, n: np.ndarray) -> np.ndarray:
        """``r_n`` (0 for ``n < 1``, the long run's past the rows)."""
        inside = (n >= 1) & (n <= self._rows)
        steady = 1.0 / self.life.cycle
        return np.where(
            inside,
            self._renewed[np.clip(n - 1, 0, self._rows - 1)],
            np.where(n >= 1, steady, 0.0),
        )

    def _total(self, n: np.ndarray) -> np.ndarray:
        """``sum_{j <= n} r_j``."""
        steady = 1.0 / self.life.cycle
        return np.where(
            n <= self._rows,
            self.running[np.clip(n, 0, self._rows)],
            self.running[-1] + (n - self._rows) * steady,
        )

    def _position(self, x: np.ndarray, strictly: bool = False):
        """For each time ``x``: the tests at or before it (strictly before,
        with ``strictly``), and the time since the last of them (``x``
        itself with none)."""
        x = np.asarray(x, dtype=float)
        T = self.interval
        with np.errstate(invalid="ignore"):
            n = np.where(
                x >= self.first, np.floor((x - self.first) / T) + 1.0, 0.0
            )
        n = n.astype(np.int64)
        since = np.where(n > 0, x - self.first - (n - 1) * T, x)
        if strictly:
            on = (n > 0) & (since <= 0.0)
            n = np.where(on, n - 1, n)
            since = np.where(on, T, since)
        return n, np.clip(since, 0.0, np.where(n > 0, T, np.inf))

    def _earlier(
        self, n: np.ndarray, since: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each test ``n >= 1`` and time ``since`` after it: the units
        renewed at the tests before it's chances of being up at it and of
        having failed by it, and their drops by ``since`` after it."""
        row = np.minimum(n, self._rows) - 1
        up, failed = self._up[row], self._failed[row]
        drop = np.empty(n.shape)
        for first in range(0, len(n), _CHUNK):
            part = slice(first, first + _CHUNK)
            weights = _interpolation_weights(since[part] / self.interval)
            drop[part] = (weights * self._drop[row[part]]).sum(axis=1)
        for k in self._rough:
            k = int(k)
            if k >= 1:
                drop += self._r(n - k) * self.life.drops(np.array([k]), since)[
                    0
                ].reshape(n.shape)
        # Units renewed K or more tests before have failed.
        failed = failed + self._total(np.maximum(n - len(self.life.up), 0))
        return up, failed, drop

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        out = np.empty(flat.shape)
        late = flat >= self.settle
        if late.any():
            phase = np.mod(flat[late] - self.first, self.interval)
            out[late] = self.life.profile(phase)[0]
        early = np.flatnonzero(~late)
        if early.size:
            values = np.asarray(self.initial(flat[early]), dtype=float)
            n, since = self._position(flat[early])
            tested = n > 0
            if tested.any():
                n, since = n[tested], since[tested]
                up, _, drop = self._earlier(n, since)
                last = self._r(n) * np.clip(
                    _values(self.life.model.sf, since), 0.0, 1.0
                )
                values = values.copy()
                values[tested] += last + up - drop
            out[early] = values
        return np.clip(out, 0.0, 1.0).reshape(x.shape)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x`` (see
        ``GridCurve.events``): its hidden failures (each unit put in
        service fails at most once before the test that renews it); the
        tests; and the failures they find, each a renewal."""
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        n, since = self._position(flat, strictly=True)
        failures = (
            np.clip(self.initial_down(flat), 0.0, 1.0) - self._down_at_start
        )
        tested = n > 0
        if tested.any():
            m, after = n[tested], since[tested]
            _, failed, drop = self._earlier(m, after)
            last = self._r(m) * np.clip(
                _values(self.life.model.ff, after), 0.0, 1.0
            )
            failures = failures.copy()
            failures[tested] += last + failed + drop
        found = self._total(n)
        shape = x.shape
        return _events(
            np.maximum(failures, 0.0).reshape(shape),
            corrective=found.reshape(shape),
            inspections=n.astype(float).reshape(shape),
        )

    def atoms(self, stop: float) -> Atoms:
        """None: a test, in no time, takes nothing down."""
        return Atoms.none()

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The tests in ``[start, stop]`` and points between them (and
        before the first test) close enough for quadrature."""
        times = self.life.knots(start, stop, self.first)
        if start < self.first:
            # Before the first test, the unit in service at 0: closing in on
            # 0, where a new one may not age smoothly.
            pieces = self.life.pieces
            head = np.concatenate(
                [
                    np.arange(pieces) / pieces,
                    2.0 ** -np.arange(24.0, 0.0, -1.0) / pieces,
                ]
            )
            head = self.first * head
            times = np.concatenate([head[head >= start], times])
        return np.unique(times)


class TestedLifeSteady:
    """A unit tested and renewed as in ``TestedLife``, in its long-run state
    from 0, its last test ``phase`` before 0 (tested then, with 0): the
    long-run profile from 0 on, and its events at their long-run rates."""

    settle = 0.0

    def __init__(self, life: TestedLife, phase: float = 0.0):
        self.life = life
        self.phase = float(phase)
        self.period = life.interval

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.life.profile(np.mod(x + self.phase, self.life.interval))[0]

    def _failed(self, position: np.ndarray) -> np.ndarray:
        """The long-run failures from a test to each ``position`` on the
        calendar after it: ``1 / S`` an interval, and within one the down
        profile."""
        T = self.life.interval
        whole = np.floor(position / T)
        return (
            whole / self.life.cycle
            + self.life.profile(position - whole * T)[1]
        )

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Failures at the long-run rate of each phase, and the tests (at
        the multiples of the interval on the calendar), each finding ``1 /
        S`` failures."""
        x = np.asarray(x, dtype=float)
        failures = self._failed(x + self.phase) - self._failed(
            np.array(self.phase)
        )
        before = multiples_before(x + self.phase, self.life.interval)
        if self.phase:
            before = before - multiples_before(
                np.array(self.phase), self.life.interval
            )
        return _events(
            failures,
            corrective=before / self.life.cycle,
            inspections=before,
        )

    def atoms(self, stop: float) -> Atoms:
        """None: a test, in no time, takes nothing down."""
        return Atoms.none()

    def knots(self, start: float, stop: float) -> np.ndarray:
        return self.life.knots(start, stop, -self.phase)
