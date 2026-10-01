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

from typing import Callable, NamedTuple, Optional

import numpy as np

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
#: The most values ``block_availability`` keeps (intervals times grid
#: points), and the change from one interval to the next below which its
#: availability has settled into repeating.
_MAX_VALUES = 2**22
_SETTLED = 1e-12
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


class _Grid(NamedTuple):
    """What a unit under block replacement is followed on: the grid over a
    block interval (step ``h``) and the ``length + 1`` points ``rho`` after
    its start that a repair or replacement can reach; ``N``, ``A`` and
    ``U`` over the interval for a unit put into service as new at its
    start; ``in_repair``, the integral of the repair's survival function
    over each cell; and the repair and the replacement."""

    interval: float
    steps: int
    h: float
    grid: np.ndarray
    length: int
    rho: np.ndarray
    N: np.ndarray
    A: np.ndarray
    U: np.ndarray
    in_repair: np.ndarray
    fix: _Duration
    replace: _Duration


def _grid(life, repair, duration, interval: float, node) -> _Grid:
    """The grid and renewal functions of a unit under block replacement
    (see ``block_cycle`` for the arguments)."""
    from scipy.signal import fftconvolve

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
    return _Grid(
        T,
        steps,
        h,
        grid,
        length,
        long_grid[: length + 1],
        N,
        A,
        U,
        in_repair,
        fix,
        replace,
    )


def _carry(P: np.ndarray, C: np.ndarray, density: np.ndarray, g: _Grid):
    """Into the next interval: the CDF of when the unit is put into service
    after this interval's end, and its integral, on ``g.rho`` from that end.
    That is what had not happened by the end (from ``P``, that CDF over this
    interval, and ``C``, its integral), and the ends of the repairs still
    going on at it, of the failures in each of this interval's cells
    (``density``, per unit time). Also returns the probability that such a
    repair is going on at the end."""
    from scipy.signal import fftconvolve

    steps, h, length, rho = g.steps, g.h, g.length, g.rho
    rest_P = P[steps:] - P[steps]
    rest_C = C[steps:] - C[steps] - P[steps] * rho[: length + 1 - steps]
    P = np.concatenate([rest_P, np.full(steps, rest_P[-1])])
    C = np.concatenate(
        [rest_C, rest_C[-1] + rest_P[-1] * h * np.arange(1, steps + 1)]
    )
    down = 0.0
    if g.fix.fixed != 0.0:
        late = density[::-1]  # late[m]: the cell m cells before the end
        down = float(late @ g.in_repair[:steps])
        ongoing = fftconvolve(
            g.in_repair[: steps + length], late[::-1], mode="valid"
        )[: length + 1]
        carried = down - ongoing  # CDF of when the repair ends
        P = P + carried
        C = C + np.concatenate(
            [[0.0], np.cumsum(0.5 * h * (carried[1:] + carried[:-1]))]
        )
    return P, C, down


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
    from scipy.signal import fftconvolve

    g = _grid(life, repair, duration, interval, node)
    steps, h, T, rho = g.steps, g.h, g.interval, g.rho

    # Where the unit is put into service in the first interval: once the
    # replacement that starts the cycle is done.
    P = g.replace.cdf(rho)
    C = rho - g.replace.in_progress(rho)

    up = cycle = failures = 0.0
    going = 1.0
    # Up in the middle of each cell of an interval, at its end, and failures
    # in each cell, summed over the intervals of a cycle.
    A_middle = 0.5 * (g.A[1:] + g.A[:-1])
    up_at = np.zeros(steps)
    up_at_end = 0.0
    failing = np.zeros(steps)
    for block in range(_MAX_INTERVALS):
        w = _masses(P, C, steps, h)
        up += float(w @ g.U[::-1])
        failures += float(w @ g.N[::-1])
        cycle += (block + 1) * T * float(w @ g.A[::-1])
        up_at += fftconvolve(w[:steps], A_middle)[:steps]
        up_at_end += float(w @ g.A[::-1])
        # Failures in each cell of this interval, from the units put into
        # service in it.
        density = np.diff(fftconvolve(w, g.N)[: steps + 1]) / h
        failing += density

        # Carried into the next interval: what had not started by this
        # interval's end, and the repairs still going on at it.
        P, C, _ = _carry(P, C, density, g)
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
        g.grid,
        np.clip(up_at / intervals, 0.0, 1.0),
        np.maximum(failing / intervals, 0.0),
        before,
        before * float(g.replace.cdf(np.zeros(1))[0]),
    )


class BlockAvailability(NamedTuple):
    """The point availability of a unit new at 0 under block replacement,
    interval by interval (see ``block_availability``).

    In the ``k``-th interval, a time ``s`` after its start, the unit is up
    with probability ``smooth[k]`` (on ``grid``, linear between its points)
    plus ``replaced[k]`` times the probability that the replacement due at
    its start is done by ``s``: ``replace.cdf(s)``, or 1 in the first
    interval, which the unit starts new. ``replaced[k]`` is the probability
    that it is up just before that start, and so replaced. ``failures[k]``
    is its expected number of failures in the interval by each point of
    ``grid``. If ``settled``, every later interval repeats the last one."""

    interval: float
    grid: np.ndarray
    smooth: np.ndarray
    replaced: np.ndarray
    replace: _Duration
    settled: bool
    failures: np.ndarray
    #: Whether the first interval starts with the unit new (from new), or
    #: with a replacement due, ``replaced[0]``, as the later ones do (after
    #: a ``BlockHead``).
    fresh: bool = True


class BlockHead(NamedTuple):
    """How a unit started from a state reaches its first block time,
    ``length`` after the start (see ``block_availability``): up there with
    probability ``up``; ``failures(u)``, its expected failures before each
    ``u`` up to it; and ``down_sf``, the survival function of what is left
    of the repair or replacement it was in at the start (None if it was
    up), which may carry on past the block time."""

    length: float
    up: float
    failures: Callable[[np.ndarray], np.ndarray]
    down_sf: Optional[Callable[[np.ndarray], np.ndarray]] = None


def _head_start(head: BlockHead, g: _Grid):
    """The first interval's input after ``head`` (see ``block_availability``):
    the CDF of when the unit is put into service after the block time, and
    its integral, on ``g.rho``; what of that is not the block
    replacement's; and the probability that it is replaced there.

    The repairs going on at the block time are those of the failures before
    it, in cells of the grid's step back from it (each taken at a constant
    density, the repair integrated exactly over the cell, as ``_carry``
    takes them), and the first unit's, if it started down and is still."""
    from scipy.signal import fftconvolve

    h, length, rho = g.h, g.length, g.rho
    cells = max(1, int(np.ceil(head.length / h - 1e-9)))
    edges = np.maximum(head.length - h * np.arange(cells + 1), 0.0)
    counts = np.asarray(head.failures(edges), dtype=float)
    late = np.maximum(counts[:-1] - counts[1:], 0.0) / h
    carried = np.zeros(length + 1)
    if g.fix.fixed != 0.0:
        down = float(late @ g.in_repair[:cells])
        ongoing = fftconvolve(
            g.in_repair[: cells + length], late[::-1], mode="valid"
        )[: length + 1]
        carried = down - ongoing
    if head.down_sf is not None:
        left = np.asarray(head.down_sf(head.length + rho), dtype=float)
        carried = carried + (left[0] - left)
    carried = np.clip(carried, 0.0, 1.0)
    C = np.concatenate(
        [[0.0], np.cumsum(0.5 * h * (carried[1:] + carried[:-1]))]
    )
    return carried, C, float(head.up)


def block_availability(
    life,
    repair,
    duration,
    interval: float,
    horizon: float,
    node=None,
    head: Optional[BlockHead] = None,
) -> BlockAvailability:
    """The point availability of a component new at 0 under block
    replacement, over ``[0, horizon]``, or until it repeats from one
    interval to the next.

    The unit is followed interval by interval as in ``block_cycle``, but
    over all its intervals rather than one cycle's: each block time at
    which it is up starts a replacement, and the next interval starts with
    that as well as the repairs and replacements carried into it. In each
    interval, the unit is up at ``s`` if it has been put into service (the
    first time after the interval's start) by ``s`` and is not down after a
    failure since: the CDF of that time, less its convolution with the
    probability ``1 - A`` of being down. The replacement due at the
    interval's start is kept out of the grid: it starts at the block time
    exactly, so its end's CDF is exact at any time however short it is.

    Started from a state rather than new, the unit reaches its first block
    time as ``head`` says, and the intervals are followed from there: the
    first starts with the replacement of the unit if it is up, and the
    repairs going on (see ``_head_start``).

    Parameters
    ----------
    life, repair, duration, interval, node
        As for ``block_cycle``.
    horizon : float
        The last time the availability is needed at (from the first block
        time, after a ``head``); ``inf`` to follow it until it settles (into
        its long-run cycle).
    head : BlockHead, optional
        How the unit reaches its first block time, started from a state;
        by default None: new at 0, a block time.

    Returns
    -------
    BlockAvailability

    Raises
    ------
    NotImplementedError
        As ``block_cycle`` does, or if the availability has not settled
        into repeating within the intervals the grid can keep.
    """
    from scipy.signal import fftconvolve

    g = _grid(life, repair, duration, interval, node)
    steps, h = g.steps, g.h
    most = max(3, _MAX_VALUES // (steps + 1))
    # An infinite horizon: until it settles, for its long-run cycle.
    count = (
        int(np.floor(horizon / g.interval)) + 1
        if np.isfinite(horizon)
        else most + 1
    )
    # A replacement that starts at the interval's start: the CDF of when
    # the unit is back in service, and its integral.
    start_P = g.replace.cdf(g.rho)
    start_C = g.rho - g.replace.in_progress(g.rho)
    down = 1.0 - g.A
    if head is None:
        # The first interval: put into service new at 0.
        P = np.ones(g.length + 1)
        C = g.rho.copy()
        other = np.zeros(g.length + 1)
        replaced = 1.0
    else:
        other, C, replaced = _head_start(head, g)
        P = other + replaced * start_P
        C = C + replaced * start_C
    rows: list = []
    probabilities: list = []
    failing: list = []
    settled = False
    for k in range(count):
        w = _masses(P, C, steps, h)
        row = other[: steps + 1] - fftconvolve(w, down)[: steps + 1]
        # Failures in each cell of this interval, from the units put into
        # service in it, and their running count over the interval.
        density = np.diff(fftconvolve(w, g.N)[: steps + 1]) / h
        failures = np.concatenate([[0.0], np.cumsum(density * h)])
        if (
            k >= 2
            and abs(replaced - probabilities[-1]) < _SETTLED
            and np.max(np.abs(row - rows[-1])) < _SETTLED
            and np.max(np.abs(failures - failing[-1])) < _SETTLED
        ):
            settled = True
            break
        if len(rows) == most:
            raise NotImplementedError(
                f"Component {node!r}: under block replacement its "
                f"availability from new has not settled after {most} "
                "intervals, as far as the grid can follow it. "
                f"{_UNSUPPORTED}"
            )
        rows.append(row)
        probabilities.append(replaced)
        failing.append(failures)
        other, C, in_repair = _carry(P, C, density, g)
        # Up at the next block time: put into service in this interval and
        # not in a repair at its end (so that no probability is lost).
        replaced = float(P[steps]) - in_repair
        P = other + replaced * start_P
        C = C + replaced * start_C
    return BlockAvailability(
        g.interval,
        g.grid,
        np.array(rows),
        np.array(probabilities),
        g.replace,
        settled,
        np.array(failing),
        head is None,
    )
