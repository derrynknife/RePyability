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

The distributions are put on a grid with the replacement age on it. Their atoms
at 0 (dead on arrival, work in no time) and at the age are kept apart and
exactly, so a sum of them that falls at the time counted to is left out in
full: the count is over ``[0, t)``, as the simulation counts, and windows one
after another add up. The rest is rounded to the grid twice, each value down to
a grid point and up to the next. The mean of the two, from a grid point, is the
probability half a step later to second order, and is read off half a step
earlier than the time wanted. The grid is refined until the probabilities move
less than ``TOLERANCE``.

Two kinds of component are counted otherwise (#147):

- **Under block replacement every ``T``** (``Block``), a unit up at a block
  time is replaced there, and one down then (being repaired or replaced)
  is not. So a new unit's next replacement is at its failure, if that comes
  before the next block time, and at the block time otherwise. The
  distribution of each replacement's time follows from the one before,
  block interval by block interval, on the grid with the block times on it;
  a repair or replacement still going on at a block time carries the next
  unit's start past it, as it does in the simulation.
- **With hidden failures, tested and renewed in no time** (``Tested``), a
  unit is replaced at the test that finds it failed, so the replacements
  fall on the tests, and their count is that of a discrete renewal process
  on them: exactly, with no grid. In a lead time from a random time, the
  next replacement is ``j`` tests on with probability ``R((j - 1) T) / S``
  (``S`` the mean cycle, in tests); and before a replacement, the ones
  before it are whole cycles back.
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


class Block(NamedTuple):
    """What a component under block replacement replaces on: the CDFs of
    its life, its repair time and its block replacement's duration (None
    for one in no time), and its block interval."""

    life: Cdf
    repair: Cdf
    maintenance: Optional[Cdf]
    interval: float


class Tested(NamedTuple):
    """What a component with hidden failures, tested and renewed in no
    time, replaces on: its tests, at ``first`` and every ``interval`` after;
    ``found(n)``, the chances that the unit in service at 0 is found failed
    at each of the first ``n`` tests; and ``cycle``, the chances that a unit
    renewed at a test is found failed ``1, 2, ...`` tests later."""

    first: float
    interval: float
    found: Callable[[int], np.ndarray]
    cycle: np.ndarray


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
        """``P(X < end)`` from ``X`` rounded up and down: its atoms before
        the end (one at the end itself falls after it, as in the
        simulation; the mean of the two roundings', which differ where an
        atom's mass depends on what was rounded, as at a block time), and
        the mean of its roundings' continuous parts, which from a grid
        point is the probability half a step later, read half a step
        before the end."""
        reach = int(math.ceil(self.end / self.step * (1.0 - 1e-12) - 1e-9))
        atoms = 0.5 * float(up.atoms[:reach].sum() + down.atoms[:reach].sum())
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


def _block_step(
    start: _Dist,
    before: np.ndarray,
    life: _Dist,
    period: Optional[int],
    up: bool,
) -> Tuple[_Dist, _Dist]:
    """From the distribution of a new unit's start (on the grid), that of
    its replacement: at its failure, if that comes before the next block
    time (every ``period`` grid steps), and otherwise at that block time
    (an atom there). Rounded up, a failure on the block time itself came
    before it; rounded down, after it. ``before`` holds units put into
    service on a block time but just before it (rounded up onto it), which
    are replaced there unless dead on arrival."""
    size = len(start.rest)
    failed = _Dist(np.zeros(size), np.zeros(size))
    blocks = np.zeros(size)
    if period is None or period >= size:
        return _sum(start, life), _Dist(np.zeros(size), blocks)
    width = period + 1
    cut = _Dist(life.rest[:width], life.atoms[:width])
    dead = float(life.atoms[0])
    for lo in range(0, size, period):
        hi = lo + period
        if hi < size and before[hi]:
            failed.atoms[hi] += before[hi] * dead
            blocks[hi] += before[hi] * (1.0 - dead)
        rest, atoms = start.rest[lo:hi], start.atoms[lo:hi]
        if not (rest.any() or atoms.any()):
            continue
        part = _Dist(
            np.pad(rest, (0, width - len(rest))),
            np.pad(atoms, (0, width - len(atoms))),
        )
        out = _sum(part, cut)
        keep = width if up else width - 1
        stop = min(lo + keep, size)
        failed.rest[lo:stop] += out.rest[: stop - lo]
        failed.atoms[lo:stop] += out.atoms[: stop - lo]
        survive = float(rest.sum() + atoms.sum()) - float(
            out.rest[: stop - lo].sum() + out.atoms[: stop - lo].sum()
        )
        if hi < size:
            blocks[hi] += max(survive, 0.0)
    return failed, _Dist(np.zeros(size), blocks)


def _block_starts(
    failed: _Dist,
    blocks: _Dist,
    repair: _Dist,
    maintenance: _Dist,
    period: Optional[int],
    up: bool,
) -> Tuple[_Dist, np.ndarray]:
    """The next units' starts: after each failure's repair and each block
    replacement's. Rounded up, a start on a block time is before it (its
    true time is at most the grid's), unless it is the unit put in there
    by that block's replacement in no time: those are kept apart."""
    start = _sum(failed, repair) + _sum(blocks, maintenance)
    size = len(start.rest)
    before = np.zeros(size)
    if not up or period is None or period >= size:
        return start, before
    on = np.arange(period, size, period)
    total = start.rest[on] + start.atoms[on]
    renewed = blocks.atoms[on] * float(maintenance.atoms[0])
    before[on] = np.maximum(total - renewed, 0.0)
    start.rest[on] = 0.0
    start.atoms[on] = np.minimum(renewed, total)
    return start, before


def _block_tails(model: Block, end: float, steps: int) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, for a component under block
    replacement, from new, by ``end``, on a grid of about ``steps`` steps
    with the block times on it (see the module docstring)."""
    grid = _Grid(end, model.interval, steps)
    period = grid.index(model.interval)
    pieces = []
    for up in (True, False):
        life = grid.from_cdf(model.life, up)
        repair = grid.from_cdf(model.repair, up)
        maintenance = (
            grid.from_cdf(model.maintenance, up)
            if model.maintenance is not None
            else grid.unit()
        )
        pieces.append((life, repair, maintenance, up))
    starts = [(grid.unit(), np.zeros(grid.size)) for _ in range(2)]
    tails: List[float] = []
    while len(tails) < MAX_COUNT:
        steps_out = [
            _block_step(start, before, life, period, up)
            for (start, before), (life, _, _, up) in zip(starts, pieces)
        ]
        tail = grid.at_most(
            steps_out[0][0] + steps_out[0][1],
            steps_out[1][0] + steps_out[1][1],
        )
        tails.append(tail)
        if tail < _TAIL:
            return np.array(tails)
        starts = [
            _block_starts(failed, blocks, repair, maintenance, period, up)
            for (failed, blocks), (_, repair, maintenance, up) in zip(
                steps_out, pieces
            )
        ]
    raise NotImplementedError(
        f"More than {MAX_COUNT} replacements are likely in the time: count "
        "them by simulation instead."
    )


def _lattice_tails(first: np.ndarray, cycle: np.ndarray, last: int) -> list:
    """``P(J_s <= last)``, ``s = 1, 2, ...``, until it is below ``_TAIL``:
    ``J_1`` has the distribution ``first`` (over ``0, 1, 2, ...`` tests),
    and each next one is a ``cycle`` (over ``0, 1, 2, ...``) more."""
    from scipy.signal import fftconvolve

    tails: List[float] = []
    if last < 0:
        return [0.0]
    head = np.asarray(first, dtype=float)[: last + 1]
    step = np.asarray(cycle, dtype=float)[: last + 1]
    while len(tails) < MAX_COUNT:
        tail = float(head.sum())
        tails.append(min(tail, 1.0))
        if tail < _TAIL:
            return tails
        if len(head) * len(step) <= 4_000_000:
            head = np.convolve(head, step)[: last + 1]
        else:
            head = np.maximum(fftconvolve(head, step)[: last + 1], 0.0)
    raise NotImplementedError(
        f"More than {MAX_COUNT} replacements are likely in the time: count "
        "them by simulation instead."
    )


def _tested_tails(model: Tested, end: float, kind: str) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, for a component whose replacements
    fall on its tests (see the module docstring for ``kind``)."""
    T = model.interval
    cycle = np.concatenate([[0.0], model.cycle])
    if kind == "new":
        # The tests strictly before the end.
        tests = 0
        if end > model.first:
            tests = int(math.ceil((end - model.first) / T - 1e-12))
        if tests == 0:
            return np.zeros(1)
        first = np.concatenate([[0.0], model.found(tests)])
        return np.array(_lattice_tails(first, cycle, tests))
    if kind == "arrival":
        # The replacements before one, whole cycles back, within the time.
        last = int(math.ceil(end / T - 1e-12)) - 1
        return np.array(_lattice_tails(cycle, cycle, last))
    # From a random time: the next replacement j tests on with probability
    # P(cycle >= j) / E[cycle], over the tests in the time (one more with
    # the probability of the fraction of an interval over).
    survive = 1.0 - np.concatenate([[0.0], np.cumsum(model.cycle)])
    delay = np.concatenate([[0.0], np.clip(survive[:-1], 0.0, 1.0)])
    delay /= delay.sum()
    whole = int(math.floor(end / T + 1e-12))
    share = end / T - whole
    if share < 1e-12:
        share = 0.0
    fewer = np.array(_lattice_tails(delay, cycle, whole))
    if not share:
        return fewer
    more = np.array(_lattice_tails(delay, cycle, whole + 1))
    size = max(len(fewer), len(more))
    fewer = np.pad(fewer, (0, size - len(fewer)))
    more = np.pad(more, (0, size - len(more)))
    return (1.0 - share) * fewer + share * more


def count(model, end: float, kind: str) -> np.ndarray:
    """The distribution of a component's replacements (see
    ``_count_tails`` for ``kind``): their probabilities for ``0, 1, 2,
    ...``, from grids refined until they agree to ``TOLERANCE`` (a
    ``Block`` from new only; a ``Tested`` exactly, with no grid).

    Raises
    ------
    NotImplementedError
        If more than ``MAX_COUNT`` replacements are likely.
    """
    if end <= 0.0:
        return np.ones(1)
    if isinstance(model, Tested):
        tails = np.minimum.accumulate(
            np.clip(_tested_tails(model, end, kind), 0.0, 1.0)
        )
        return -np.diff(np.concatenate(([1.0], tails, [0.0])))

    def tails_on(steps: int) -> np.ndarray:
        if isinstance(model, Block):
            return _block_tails(model, end, steps)
        return _count_tails(model, end, kind, steps)

    steps = FIRST_STEPS
    previous = tails_on(steps)
    while steps < MAX_STEPS:
        steps *= 2
        tails = tails_on(steps)
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
