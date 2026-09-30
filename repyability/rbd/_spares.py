"""How many spares a component uses (#95): its replacements, counted over a
horizon from new, and in a replenishment lead time in the long run.

A component is replaced at each failure and each preventive replacement:
at the end of each up time ``U``, the smaller of its life ``L`` and its
replacement age ``T`` (``inf`` for none). It is then down for ``X``: a
repair time after a failure, or a maintenance time after a preventive
replacement. Then it is back in service as new. Its cycles ``V = U + X``
are independent, so its ``n``-th replacement is at ``V_1 + ... + V_{n-1} +
U_n``. The number by time ``t``, ``N(t)``, has ``P(N(t) >= n) =
P(V^{*(n-1)} * U <= t)``.

In the long run, a lead time ``tau`` from a random time holds ``s`` or more
replacements if the ``s``-th from then falls within ``tau``. From an up
time, the first comes at the end of what is left of it, ``w``, with density
``(F(T) - F(w)) / E[V]`` for a failure and ``(1 - F(T)) / E[V]`` for a
preventive replacement (``w < T``); the next ones follow its down time and
then whole cycles. From a down time, with density ``P(X > r) / E[V]`` for
the ``r`` left of it, they follow ``r`` and whole cycles. And a
replacement finds the ``s`` before it within ``tau`` if ``X + V^{*(s-1)} +
U <= tau``, which decides the fill rate of a stock of ``s`` replenished one
for one.

The distributions are put on a grid with the replacement age on it. Their
atoms at 0 (dead on arrival, work in no time) and at the age are kept apart
and exactly, so a sum of them that falls at the time counted to is counted
in full. The rest is rounded to the grid twice, each value down to a grid
point and up to the next. The mean of the two, from a grid point, is the
probability half a step later to second order, and is read off half a step
earlier than the time wanted. The grid is refined until the probabilities
move less than ``TOLERANCE``.
"""

import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np

#: How closely the counts' probabilities are computed.
TOLERANCE = 1e-6
#: The grid's first and largest number of steps up to the time.
FIRST_STEPS, MAX_STEPS = 2**9, 2**16
#: A count is followed until the probability of more falls below this.
_TAIL = 1e-12
#: The most replacements a count follows.
MAX_COUNT = 2_000
#: Gauss-Legendre nodes and weights for integrals over grid cells.
_GAUSS = np.polynomial.legendre.leggauss(4)

Cdf = Callable[[np.ndarray], np.ndarray]


class Replacements(NamedTuple):
    """What a component's replacements follow: the CDFs of its life, its
    repair time and its maintenance time (None for maintenance in no time),
    its replacement age (``inf`` for none), and the mean length of its
    cycle, ``E[V]`` (for the counts in the long run)."""

    life: Cdf
    repair: Cdf
    maintenance: Optional[Cdf]
    age: float
    mean_cycle: float


class _Dist(NamedTuple):
    """A distribution on the grid, rounded one way: its continuous part as
    masses on the grid points, and its atoms (on grid points), exactly."""

    rest: np.ndarray
    atoms: np.ndarray

    def __add__(self, other):  # type: ignore[override]
        return _Dist(self.rest + other.rest, self.atoms + other.atoms)

    def scaled(self, factor: float) -> "_Dist":
        return _Dist(factor * self.rest, factor * self.atoms)


def _fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    if not a.any() or not b.any():
        return np.zeros(len(a))
    return np.maximum(fftconvolve(a, b)[: len(a)], 0.0)


def _sum(a: _Dist, b: _Dist) -> _Dist:
    """The distribution of the sum of two independent ones: its atoms are
    sums of atoms, the rest continuous."""
    total = _fft(a.rest + a.atoms, b.rest + b.atoms)
    atoms = _fft(a.atoms, b.atoms)
    return _Dist(np.maximum(total - atoms, 0.0), atoms)


class _Grid:
    """Grid points ``0, step, 2 step, ...`` past ``end``, with ``age`` (if
    finite and within reach) on one of them."""

    def __init__(self, end: float, age: float, steps: int):
        step = end / steps
        if math.isfinite(age) and age <= end:
            step = age / max(1, round(age / step))
        self.step = step
        self.end = end
        self.size = int(math.floor(end / step + 1e-9)) + 3
        self.points = step * np.arange(self.size)

    def index(self, x: float) -> Optional[int]:
        """The grid point at ``x``, or None if it is past the grid."""
        at = int(round(x / self.step))
        return at if at < self.size else None

    def unit(self) -> _Dist:
        """The distribution of 0."""
        return _Dist(np.zeros(self.size), np.eye(1, self.size, 0).ravel())

    def from_cdf(self, cdf: Cdf, up: bool, below: float = math.inf) -> _Dist:
        """A distribution with CDF ``cdf``, its mass at 0 an atom and the
        rest rounded (up or down) cell by cell, up to ``below`` (the rest
        is left out)."""
        edges = np.append(self.points, self.points[-1] + self.step)
        values = np.clip(np.asarray(cdf(edges), dtype=float), 0.0, 1.0)
        values = np.maximum.accumulate(values)
        cells = np.diff(values)
        cells[edges[1:] > below * (1.0 + 1e-12)] = 0.0
        rest = np.zeros(self.size)
        if up:
            rest[1:] = cells[:-1]
        else:
            rest[:] = cells
        atoms = np.zeros(self.size)
        atoms[0] = values[0]
        return _Dist(rest, atoms)

    def from_density(self, density: Callable, up: bool) -> _Dist:
        """A continuous distribution (or part of one) with density
        ``density``, rounded (up or down) cell by cell."""
        nodes, weights = _GAUSS
        half = 0.5 * self.step
        x = self.points[:, None] + half * (nodes[None, :] + 1.0)
        values = np.asarray(density(x.ravel()), dtype=float).reshape(x.shape)
        cells = half * (values @ weights)
        rest = np.zeros(self.size)
        if up:
            rest[1:] = cells[:-1]
        else:
            rest[:] = cells
        return _Dist(rest, np.zeros(self.size))

    def at_most(self, up: _Dist, down: _Dist) -> float:
        """``P(X <= end)`` from ``X`` rounded up and down: its atoms up to
        the end, and the mean of its roundings' continuous parts, which
        from a grid point is the probability half a step later, read half
        a step before the end."""
        reach = int(math.floor(self.end / self.step * (1.0 + 1e-12) + 1e-9))
        atoms = float(up.atoms[: reach + 1].sum())
        cumulative = 0.5 * (np.cumsum(up.rest) + np.cumsum(down.rest))
        position = self.end / self.step - 0.5
        if position < 0.0:
            rest = 0.0
        else:
            low = int(math.floor(position))
            share = position - low
            rest = (1.0 - share) * cumulative[low] + share * cumulative[
                low + 1
            ]
        return min(atoms + rest, 1.0)


class _Pieces(NamedTuple):
    """A component's distributions on a grid, rounded one way: its up time,
    ending in a failure or a preventive replacement, the down time after
    each, a whole cycle, and the down time after a replacement at random."""

    fail: _Dist
    pm: _Dist
    repair: _Dist
    maintenance: _Dist
    up_time: _Dist
    cycle: _Dist
    down_time: _Dist


def _pieces(model: Replacements, grid: _Grid, up: bool) -> _Pieces:
    """``model``'s distributions on ``grid``, rounded up or down."""
    age = model.age
    survive = 0.0
    pm = _Dist(np.zeros(grid.size), np.zeros(grid.size))
    if math.isfinite(age):
        survive = 1.0 - float(model.life(np.array([age]))[0])
        at = grid.index(age)
        if at is not None:
            pm.atoms[at] = max(survive, 0.0)
    fail = grid.from_cdf(model.life, up, below=age)
    repair = grid.from_cdf(model.repair, up)
    maintenance = (
        grid.from_cdf(model.maintenance, up)
        if model.maintenance is not None
        else grid.unit()
    )
    cycle = _sum(fail, repair) + _sum(pm, maintenance)
    down = repair.scaled(1.0 - survive) + maintenance.scaled(survive)
    return _Pieces(fail, pm, repair, maintenance, fail + pm, cycle, down)


def _random_start(
    model: Replacements, grid: _Grid, up: bool
) -> Tuple[_Dist, _Dist, _Dist]:
    """From a random time in the long run: the time left of an up time
    ending in a failure, and in a preventive replacement, and of a down
    time (densities, over ``E[V]``; see the module docstring)."""
    life, age, mean = model.life, model.age, model.mean_cycle
    top = float(life(np.array([age]))[0]) if math.isfinite(age) else 1.0

    def before(x):
        return (x < age).astype(float)

    fail = grid.from_density(
        lambda x: before(x) * (top - np.clip(life(x), 0.0, top)) / mean, up
    )
    pm = grid.from_density(lambda x: before(x) * (1.0 - top) / mean, up)

    def left(x):
        out = top * (1.0 - np.clip(model.repair(x), 0.0, 1.0))
        if model.maintenance is not None:
            out = out + (1.0 - top) * (
                1.0 - np.clip(model.maintenance(x), 0.0, 1.0)
            )
        return out / mean

    return fail, pm, grid.from_density(left, up)


def _tails(grid: _Grid, pairs, cycles) -> List[float]:
    """``P(first * V^{*(s-1)} <= end)`` for ``s = 1, 2, ...`` until it is
    below ``_TAIL``: ``pairs`` the first distribution rounded (up, down),
    ``cycles`` the cycle's."""
    tails: List[float] = []
    up, down = pairs
    while len(tails) < MAX_COUNT:
        tail = grid.at_most(up, down)
        tails.append(tail)
        if tail < _TAIL:
            return tails
        up, down = _sum(up, cycles[0]), _sum(down, cycles[1])
    raise NotImplementedError(
        f"More than {MAX_COUNT} replacements are likely in the time: count "
        "them by simulation instead."
    )


def _count_tails(
    model: Replacements, end: float, kind: str, steps: int
) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, on a grid of about ``steps`` steps
    up to ``end``: for ``kind`` ``"new"`` (from new, by ``end``),
    ``"random"`` (in a lead time ``end`` from a random time in the long
    run) or ``"arrival"`` (in the lead time before a replacement)."""
    grid = _Grid(end, model.age, steps)
    pieces = [_pieces(model, grid, up) for up in (True, False)]
    cycles = (pieces[0].cycle, pieces[1].cycle)
    if kind == "new":
        return np.array(
            _tails(grid, (pieces[0].up_time, pieces[1].up_time), cycles)
        )
    if kind == "arrival":
        firsts = tuple(_sum(p.down_time, p.up_time) for p in pieces)
        return np.array(_tails(grid, firsts, cycles))
    starts = [_random_start(model, grid, up) for up in (True, False)]
    ones, afters = [], []
    for (fail, pm, left), p in zip(starts, pieces):
        ones.append(fail + pm + _sum(left, p.up_time))
        after = (
            _sum(fail, p.repair)
            + _sum(pm, p.maintenance)
            + _sum(left, p.cycle)
        )
        afters.append(_sum(after, p.up_time))
    first = grid.at_most(ones[0], ones[1])
    return np.array([first] + _tails(grid, tuple(afters), cycles))


def count(model: Replacements, end: float, kind: str) -> np.ndarray:
    """The distribution of a component's replacements (see
    ``_count_tails`` for ``kind``): their probabilities for ``0, 1, 2,
    ...``, from grids refined until they agree to ``TOLERANCE``.

    Raises
    ------
    NotImplementedError
        If more than ``MAX_COUNT`` replacements are likely.
    """
    if end <= 0.0:
        return np.ones(1)
    steps = FIRST_STEPS
    previous = _count_tails(model, end, kind, steps)
    while steps < MAX_STEPS:
        steps *= 2
        tails = _count_tails(model, end, kind, steps)
        size = max(len(previous), len(tails))
        change = np.abs(
            np.pad(tails, (0, size - len(tails)))
            - np.pad(previous, (0, size - len(previous)))
        )
        previous = tails
        # The error falls as the step squared: a quarter of what it was, so
        # about a third of the change.
        if change.max(initial=0.0) <= 3.0 * TOLERANCE:
            break
    tails = np.minimum.accumulate(np.clip(previous, 0.0, 1.0))
    return -np.diff(np.concatenate(([1.0], tails, [0.0])))
