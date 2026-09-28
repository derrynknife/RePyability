"""Exact long-run values of a component under block replacement.

Under block replacement every ``T``, a unit is replaced, as new, at each
multiple of ``T`` at which it is up. Between those times it fails and is
repaired as usual, and a replacement that falls while it is down (being
repaired, or still being replaced) is skipped. The block times at which the
unit is up are regeneration points, so by the renewal-reward theorem its
long-run availability and failure frequency are its mean up time and mean
number of failures per cycle, from one such replacement to the next, over
the cycle's mean length.

Put into service as new within a block interval, a unit is an alternating
renewal process of lives and repairs up to the next block time. Its
expected number of failures ``N(x)``, probability of being up ``A(x)`` and
expected up time ``U(x)``, a time ``x`` after it was put into service, solve
renewal equations whose renewal distribution ``H`` is that of a life plus a
repair. A repair still going on at a block time carries the unit into the
next interval, where it starts as new once the repair is done. The cycle
ends at the first block time at which the unit is up.

Everything is computed on a grid of step ``h``. An integral against a
distribution uses the distribution's exact increments over each grid cell
(from its CDF, or from the integral of its survival function), so atoms
(instant or fixed-length repairs and replacements) and densities that are
infinite at 0 are handled. The renewal equations are solved by the midpoint
rule, and where the unit is put into service within an interval is kept as
masses on the grid points, each spread over the two nearest so that its
mean is kept. The error falls as ``h ** 2``.
"""

from typing import NamedTuple

import numpy as np
from scipy.signal import fftconvolve

from ._model_utils import distribution_name, never_fails

_GAUSS = np.polynomial.legendre.leggauss(8)

#: A cycle is followed, one block interval after another, until the
#: probability that it has not ended is below this.
_TAIL = 1e-14
#: The most block intervals followed in one cycle.
_MAX_INTERVALS = 100_000
#: Grid steps per block interval: at least, at most, and per interquartile
#: range of the lives.
_MIN_STEPS, _MAX_STEPS, _STEPS_PER_SPREAD = 2_000, 20_000, 40
#: The longest grid, in steps, for the repairs and replacements that carry a
#: unit from one block interval into the next.
_MAX_GRID = 4_000_000
_UNSUPPORTED = "Estimate them by simulation, with availability() or cost()."


class BlockCycle(NamedTuple):
    """A regeneration cycle under block replacement: its mean up time, mean
    length and mean number of failures (one replacement per cycle), and the
    long-run profiles over a block interval, which is what components
    replaced at the same block times share.

    ``phase`` is a grid over one interval, ``[0, T]``; ``availability`` the
    long-run probability that the unit is up in each cell between its
    points (at the cell's middle); ``failure_rate`` its long-run failure
    intensity in each cell; and ``before`` and ``after`` the probabilities
    that it is up just before and just after a block time."""

    up: float
    length: float
    failures: float
    phase: np.ndarray
    availability: np.ndarray
    failure_rate: np.ndarray
    before: float
    after: float


def _values(function, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):  # e.g. log(0)
        values = function(x.ravel())
    return np.asarray(values, dtype=float).reshape(x.shape)


def _cumulative(function, edges: np.ndarray) -> np.ndarray:
    """``integral_0^e function`` at each of ``edges`` (increasing, from 0),
    by 8-point Gauss-Legendre quadrature on each cell."""
    points, weights = _GAUSS
    a, b = edges[:-1], edges[1:]
    half = 0.5 * (b - a)
    nodes = (a + half)[:, None] + half[:, None] * points[None, :]
    cells = half * (_values(function, nodes) @ weights)
    return np.concatenate([[0.0], np.cumsum(cells)])


class _Duration:
    """A repair or replacement time: ``None`` (instant), a fixed time, or a
    surpyval model."""

    def __init__(self, model, what: str, node):
        self.model = model
        self.fixed = None
        if model is None:
            self.fixed = 0.0
        elif distribution_name(model) == "ExactEventTime":
            self.fixed = float(np.ravel(model.params)[0])
        elif distribution_name(model) is None:
            raise NotImplementedError(
                f"Component {node!r}: its {what} model is not a surpyval "
                "parametric model, which the exact block-replacement values "
                f"need. {_UNSUPPORTED}"
            )
        elif never_fails(model) > 0.0:
            raise NotImplementedError(
                f"Component {node!r}: some of its {what}s never end (p < 1), "
                "so it would end up down for good under block replacement. "
                f"{_UNSUPPORTED}"
            )

    def tail(self) -> float:
        """A time by which the duration is all but surely over."""
        if self.fixed is not None:
            return self.fixed
        try:
            return float(np.ravel(self.model.qf(np.array([1 - 1e-12])))[0])
        except Exception:  # pragma: no cover - surpyval models have qf
            return 50.0 * float(np.ravel(self.model.mean())[0])

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """``P(duration <= x)`` for ``x >= 0``."""
        if self.fixed is not None:
            return (x >= self.fixed).astype(float)
        return np.clip(_values(self.model.ff, x), 0.0, 1.0)

    def in_progress(self, edges: np.ndarray) -> np.ndarray:
        """``integral_0^x P(duration > v) dv`` at the grid ``edges``: the
        expected time it has run, of the first ``x``."""
        if self.fixed is not None:
            return np.minimum(edges, self.fixed)
        return _cumulative(
            lambda v: np.clip(_values(self.model.sf, v), 0.0, 1.0), edges
        )


def _check_life(life, node) -> None:
    name = distribution_name(life)
    if name is None or name in (
        "ExactEventTime",
        "FixedEventProbability",
        "Bernoulli",
    ):
        raise NotImplementedError(
            f"Component {node!r}: the exact block-replacement values need a "
            "surpyval parametric lifetime with a density; its reliability "
            f"model is not one. {_UNSUPPORTED}"
        )
    if float(getattr(life, "f0", 0.0) or 0.0) > 0.0:
        raise NotImplementedError(
            f"Component {node!r}: some of its units are dead on arrival "
            "(f0 > 0), which the exact block-replacement values do not "
            f"cover. {_UNSUPPORTED}"
        )


def _steps(life, interval: float) -> int:
    """Grid steps per block interval: enough to resolve the lives."""
    p = 1.0 - never_fails(life)
    quartiles = np.ravel(life.qf(np.array([0.25 * p, 0.75 * p])))
    spread = float(quartiles[1] - quartiles[0])
    if not np.isfinite(spread) or spread <= 0.0:
        return _MAX_STEPS
    wanted = int(np.ceil(_STEPS_PER_SPREAD * interval / spread))
    return int(np.clip(wanted, _MIN_STEPS, _MAX_STEPS))


def _renewal(H: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``X(x) = b(x) + integral_0^x X(x - s) dH(s)`` on the grid for
    each row ``b`` of ``rhs``, by the midpoint rule: ``X`` at the middle of
    each cell of ``s`` is the mean of its values at the cell's ends."""
    steps = len(H) - 1
    # c[j] = dH_j / 2 for the j-th cell (c[0] and c[steps + 1] are 0);
    # X[i - k] weighs e[k - 1] = c[k] + c[k + 1], and X[0] weighs c[i].
    c = np.concatenate([[0.0], 0.5 * np.diff(H), [0.0]])
    e = c[1:-1] + c[2:]
    X = np.empty_like(rhs)
    X[:, 0] = rhs[:, 0]
    denominator = 1.0 - c[1]
    # The rows of X in reverse order, so that X[i - 1], ..., X[1] is a
    # contiguous slice.
    backwards = np.zeros_like(rhs)
    last = steps  # backwards[:, last] is X[:, 0]
    backwards[:, last] = X[:, 0]
    for i in range(1, steps + 1):
        total = rhs[:, i] + c[i] * X[:, 0]
        if i > 1:
            start = last - i + 1  # X[i - 1] ... X[1]
            total = total + backwards[:, start:last] @ e[: i - 1]
        X[:, i] = total / denominator
        backwards[:, last - i] = X[:, i]
    return X


def _masses(P: np.ndarray, C: np.ndarray, steps: int, h: float):
    """Where a unit is put into service in a block interval, as masses on
    the interval's grid points, from ``P``, the CDF of that time, and ``C``,
    its integral, on the grid. Each part of the distribution goes to the two
    nearest grid points, so its mean is kept; at the interval's end only the
    part before it counts (the rest belongs to the next interval)."""
    second = np.diff(C, 2)  # C[j + 1] - 2 C[j] + C[j - 1] at j = 1, 2, ...
    w = np.empty(steps + 1)
    w[0] = C[1] / h
    w[1:steps] = second[: steps - 1] / h
    w[steps] = P[steps] - (C[steps] - C[steps - 1]) / h
    return w


def block_cycle(
    life, repair, duration, interval: float, node=None
) -> BlockCycle:
    """The regeneration cycle of a component under block replacement.

    Parameters
    ----------
    life : surpyval model
        The time to failure: parametric, with a density (no dead-on-arrival
        units), perhaps with units that never fail.
    repair : surpyval model
        The time to repair (an ``ExactEventTime`` for a fixed time, at 0
        for instant repair). Every repair must end.
    duration : surpyval model or None
        The time a replacement takes, or None for none.
    interval : float
        The block interval ``T``.
    node : Hashable, optional
        The component's name, for the error messages.

    Returns
    -------
    BlockCycle
        The cycle's mean up time, mean length and mean number of failures.

    Raises
    ------
    NotImplementedError
        If a model is not one of those, or the repairs or replacements
        take too long, compared with the interval, for the grid.
    """
    _check_life(life, node)
    fix = _Duration(repair, "repair", node)
    replace = _Duration(duration, "replacement", node)
    T = float(interval)
    steps = _steps(life, T)
    h = T / steps
    grid = h * np.arange(steps + 1)

    # A life: its CDF, and the expected up time of a unit put into service
    # as new, up to each grid time.
    F = np.clip(_values(life.ff, grid), 0.0, 1.0)
    F[0] = 0.0
    up_time = _cumulative(
        lambda v: 1.0 - np.clip(_values(life.ff, v), 0.0, 1.0), grid
    )

    # How far the repairs and replacements that carry a unit from one
    # interval into the next can reach.
    reach = max(fix.tail(), replace.tail())
    length = int(np.ceil(reach / h)) + steps + 2
    if length + steps > _MAX_GRID or not np.isfinite(reach):
        raise NotImplementedError(
            f"Component {node!r}: its repairs or replacements can take far "
            "longer than its block interval, too long to follow on the grid "
            f"of the exact block-replacement values. {_UNSUPPORTED}"
        )
    long_grid = h * np.arange(length + steps + 1)

    # A life plus a repair, H = F * G, with each cell's life taken at a
    # constant density and the repair integrated exactly over it.
    repaired = fix.in_progress(long_grid)
    done = long_grid - repaired  # integral_0^x G, G the repair CDF
    H = np.zeros(steps + 1)
    H[1:] = fftconvolve(np.diff(F) / h, np.diff(done[: steps + 1]))[:steps]
    H = np.clip(H, 0.0, 1.0)

    N, A, U = _renewal(H, np.vstack([F, 1.0 - F, up_time]))
    # The repair still going on at a block time, for a failure in each cell
    # before it: its integral over the cell, from the end of the interval.
    in_repair = np.diff(repaired)

    # Where the unit is put into service in the first interval: once the
    # replacement that starts the cycle is done.
    rho = long_grid[: length + 1]
    P = replace.cdf(rho)
    C = rho - replace.in_progress(rho)

    up = cycle = failures = 0.0
    going = 1.0
    # Up in the middle of each cell of an interval, at its end, and failures
    # in each cell, summed over the intervals of a cycle.
    A_middle = 0.5 * (A[1:] + A[:-1])
    up_at = np.zeros(steps)
    up_at_end = 0.0
    failing = np.zeros(steps)
    for block in range(_MAX_INTERVALS):
        w = _masses(P, C, steps, h)
        up += float(w @ U[::-1])
        failures += float(w @ N[::-1])
        cycle += (block + 1) * T * float(w @ A[::-1])
        up_at += fftconvolve(w[:steps], A_middle)[:steps]
        up_at_end += float(w @ A[::-1])
        # Failures in each cell of this interval, from the units put into
        # service in it.
        density = np.diff(fftconvolve(w, N)[: steps + 1]) / h
        failing += density

        # Carried into the next interval: what had not started by this
        # interval's end, and the repairs still going on at it.
        rest_P = P[steps:] - P[steps]
        rest_C = C[steps:] - C[steps] - P[steps] * rho[: length + 1 - steps]
        P = np.concatenate([rest_P, np.full(steps, rest_P[-1])])
        C = np.concatenate(
            [rest_C, rest_C[-1] + rest_P[-1] * h * np.arange(1, steps + 1)]
        )
        if fix.fixed != 0.0:
            late = density[::-1]  # late[m]: the cell m cells before the end
            down = float(late @ in_repair[:steps])
            ongoing = fftconvolve(
                in_repair[: steps + length], late[::-1], mode="valid"
            )[: length + 1]
            carried = down - ongoing  # CDF of when the repair ends
            P = P + carried
            C = C + np.concatenate(
                [[0.0], np.cumsum(0.5 * h * (carried[1:] + carried[:-1]))]
            )
        going = float(P[-1])
        if going < _TAIL:
            break
    else:
        raise NotImplementedError(
            f"Component {node!r}: under block replacement it is so rarely "
            "up at a block time that a cycle could not be followed to its "
            f"end. {_UNSUPPORTED}"
        )
    # Per interval, in the long run: the sums over a cycle's intervals over
    # the mean number of intervals in a cycle.
    intervals = cycle / T
    before = min(1.0, up_at_end / intervals)
    return BlockCycle(
        up,
        cycle,
        failures,
        grid,
        np.clip(up_at / intervals, 0.0, 1.0),
        np.maximum(failing / intervals, 0.0),
        before,
        before * float(replace.cdf(np.zeros(1))[0]),
    )
