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
  An end on a block time is read just before it, where the replacements'
  distributions start afresh: each rounding's own continuous part before
  it (rounded up onto it, a value came before it; rounded down onto it,
  after it), not the half-step reading, which would take in part of the
  new unit's first step.

  In a lead time ``tau`` in the long run (#160), with repairs and block
  replacements in no time, every block interval starts with a new unit,
  independent of the others, so the demand repeats every interval, from
  a phase ``phi`` uniform on ``[0, T)``. Within an interval, the failures
  are a renewal process from new: with ``F^{*k}`` the ``k``-th life sum's
  CDF, ``p_j = F^{*j} - F^{*(j+1)}`` the chance of exactly ``j`` failures
  by a time and ``M`` the renewal function, the failures in ``[phi, c)``
  number ``k`` or more with probability ``F^{*k}(c) - integral over y in
  [0, phi) of p_{k-1}(c - y) dM(y)`` (by where the unit in service at
  ``phi`` started). A lead time past the interval's end adds the block
  replacement there and the replacements from new over what is left of
  it, ``C(z)`` (counted as from new above). Averaged over the phase, in
  exchanged order, this needs only 1-D sums over where the unit started:
  of the integrals of ``F^{*k}`` within the interval, and of those of the
  from-new count's replacement times past it. A replacement finds ``s``
  before it within ``tau`` as often as one finds ``s`` after it (the
  times between replacements are stationary from one): after a failure
  at ``y`` (rate ``dM``), its new unit's failures up to the interval's
  end, the block replacement there and ``C`` past it; after a block
  replacement, ``C``; over ``M(T) + 1`` replacements an interval. The
  life's sums are kept as the bands of grid points that hold their mass,
  as a block interval may hold many lives.
- **With hidden failures** (``Tested``), a unit is replaced at the test
  that finds it failed, however long the test and the repair then take, so
  the replacements fall on the tests, and their count is that of a
  discrete renewal process on them, from the chances of each cycle's
  length in tests: exactly, tested and renewed in no time, and from the
  cycle followed on a grid otherwise (#159). In a lead time from a random
  time, the next replacement is ``j`` tests on with probability ``P(C >=
  j) / S`` (``C`` a cycle, ``S`` its mean, in tests); and before a
  replacement, the ones before it are whole cycles back.

  With tests that can miss a failure, but for every ``per``-th, which
  finds every one, a cycle depends on where the test that starts it falls
  in the full tests' period (its place, ``q``): the replacements are a
  Markov renewal process over the places, counted the same way, with each
  test's place known from the first's. From a random time, whose next test
  is at each place as often, the next replacement is ``j`` tests on if the
  cycle in progress started ``a`` tests before the next test and lasts ``a
  + j - 1``: with probability the sum over ``a`` of ``r_q P(C_q = a + j -
  1)``, ``r_q`` the long-run chance that a test at the place ``a`` tests
  back finds a failure. Before a replacement, at each place in proportion
  to its share of them, the cycles back are the reversed process's: one of
  ``l`` tests back from a place ``q`` with probability ``pi_q' P(C_q' =
  l) / pi_q``, ``q'`` the place ``l`` tests before and ``pi`` the long-run
  shares of the places.
"""

import math
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from repyability.utils.vectors import dot

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
    """What a component with hidden failures replaces on: its tests, at
    ``first`` and every ``interval`` after; ``found(n)``, the chances that
    the unit in service at 0 is found failed at each of the first ``n``
    tests; and ``cycle``, the chances that a unit renewed at a test is
    found failed ``1, 2, ...`` tests later. With tests that can miss a
    failure, ``cycle`` has a row for each place in the full tests' period
    of the test that renews the unit (see the module docstring), and
    ``position`` is the first test's place."""

    first: float
    interval: float
    found: Callable[[int], np.ndarray]
    cycle: np.ndarray
    position: int = 0


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

    def at_most(self, up: _Dist, down: _Dist, left: bool = False) -> float:
        """``P(X < end)`` from ``X`` rounded up and down: its atoms before
        the end (one at the end itself falls after it, as in the
        simulation; the mean of the two roundings', which differ where an
        atom's mass depends on what was rounded, as at a block time), and
        the mean of its roundings' continuous parts, which from a grid
        point is the probability half a step later, read half a step
        before the end. ``left``, for an end on a block time, where the
        distribution starts afresh: the continuous parts before it, each
        rounding's own (rounded up onto the end, a value came before it;
        rounded down onto it, after it)."""
        reach = int(math.ceil(self.end / self.step * (1.0 - 1e-12) - 1e-9))
        atoms = 0.5 * float(up.atoms[:reach].sum() + down.atoms[:reach].sum())
        if left:
            rest = 0.5 * float(
                up.rest[: reach + 1].sum() + down.rest[:reach].sum()
            )
            return min(atoms + rest, 1.0)
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


def _block_runs(model: Block, end: float, steps: int):
    """The distributions of a component's replacement times under block
    replacement, from new, one after another: ``(grid, up, down, tail)``,
    rounded up and down on a grid of about ``steps`` steps up to ``end``
    with the block times on it, and ``tail`` the chance of each before
    ``end`` (see the module docstring), until it is below ``_TAIL``."""
    grid = _Grid(end, model.interval, steps)
    period = grid.index(model.interval)
    # An end on a block time is read before it, where the replacements'
    # distributions start afresh.
    left = end >= model.interval and (
        abs(end / model.interval - round(end / model.interval)) < 1e-9
    )
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
    for _ in range(MAX_COUNT):
        steps_out = [
            _block_step(start, before, life, period, up)
            for (start, before), (life, _, _, up) in zip(starts, pieces)
        ]
        up_time = steps_out[0][0] + steps_out[0][1]
        down_time = steps_out[1][0] + steps_out[1][1]
        tail = grid.at_most(up_time, down_time, left)
        yield grid, up_time, down_time, tail
        if tail < _TAIL:
            return
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


def _block_tails(model: Block, end: float, steps: int) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, for a component under block
    replacement, from new, by ``end``, on a grid of about ``steps`` steps
    with the block times on it (see the module docstring)."""
    return np.array([tail for *_, tail in _block_runs(model, end, steps)])


class _Read:
    """A distribution on a grid, rounded up and down, read at any times:
    ``P(X < x)``, as ``_Grid.at_most`` reads it at the grid's end, and its
    integral over ``[0, x)``, ``E[(x - X)^+]``. Its masses are those from
    grid point ``offset`` on (none before), with atoms or without."""

    def __init__(
        self,
        step: float,
        offset: int,
        up: np.ndarray,
        down: np.ndarray,
        atoms: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.step, self.offset = step, offset
        self.rest = 0.5 * (np.cumsum(up) + np.cumsum(down))
        mass = 0.5 * (up + down)
        self.atoms = None
        if atoms is not None:
            held = 0.5 * (atoms[0] + atoms[1])
            self.atoms = np.concatenate([[0.0], np.cumsum(held)])
            mass = mass + held
        points = step * (offset + np.arange(len(mass)))
        self.mass = np.cumsum(mass)
        self.moment = np.cumsum(mass * points)
        self.total = float(self.mass[-1]) if len(mass) else 0.0

    def span(self) -> Tuple[float, float]:
        """The times before which ``below`` is 0, and from which it is its
        total (with no atoms)."""
        return (
            (self.offset - 0.5) * self.step,
            (self.offset + len(self.rest) - 0.5) * self.step,
        )

    def below(self, x) -> np.ndarray:
        """``P(X < x)``: its atoms before ``x`` and the mean of its
        roundings' continuous parts half a step earlier."""
        x = np.asarray(x, dtype=float)
        position = x / self.step - 0.5
        floor = np.floor(position)
        share = position - floor
        low = floor.astype(int) - self.offset
        last = len(self.rest) - 1
        if last < 0:
            rest = np.zeros(x.shape)
        else:

            def at(i):
                return np.where(i < 0, 0.0, self.rest[np.clip(i, 0, last)])

            rest = (1.0 - share) * at(low) + share * at(low + 1)
            rest = np.where(position < 0.0, 0.0, rest)
        if self.atoms is not None:
            reach = np.ceil(x / self.step * (1.0 - 1e-12) - 1e-9).astype(int)
            reach = np.clip(reach - self.offset, 0, len(self.atoms) - 1)
            rest = rest + self.atoms[reach]
        return np.minimum(rest, 1.0)

    def integral(self, x) -> np.ndarray:
        """``E[(x - X)^+]``, the integral of ``P(X < z)`` over ``z`` in
        ``[0, x)``: from the roundings' mean, which takes a cell's mass at
        its middle."""
        x = np.asarray(x, dtype=float)
        if not len(self.mass):
            return np.zeros(x.shape)
        at = np.floor(x / self.step * (1.0 + 1e-12) + 1e-9).astype(int)
        at -= self.offset
        index = np.clip(at, 0, len(self.mass) - 1)
        out = np.where(at >= 0, x * self.mass[index] - self.moment[index], 0.0)
        return np.maximum(out, 0.0)


#: The mass left out at each end of a sum of lives kept as a band.
_TRIM = 1e-16


def _band(
    offset: int, up: np.ndarray, down: np.ndarray, size: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Masses from grid point ``offset``, rounded up and down: cut at the
    grid's ``size``, and their negligible ends (``_TRIM``) left out."""
    keep = max(0, min(len(up), size - offset))
    up, down = up[:keep], down[:keep]
    cumulative = np.cumsum(up + down)
    if not keep or cumulative[-1] <= 2.0 * _TRIM:
        return offset, up[:0], down[:0]
    lo = int(np.searchsorted(cumulative, _TRIM, side="right"))
    hi = int(np.searchsorted(cumulative, cumulative[-1] - _TRIM)) + 1
    return offset + lo, up[lo:hi], down[lo:hi]


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    if len(a) * len(b) <= 250_000:
        return np.convolve(a, b)
    return np.maximum(fftconvolve(a, b), 0.0)


def _interval_sums(model: Block, steps: int) -> Tuple[_Grid, List[_Read]]:
    """A block interval from its new unit (#160): the sums of its lives,
    ``S_1, S_2, ...``, on a grid of the interval, each kept as the band
    that holds its mass there, until the chance of the next before the
    interval's end is below ``_TAIL``."""
    T = model.interval
    grid = _Grid(T, T, steps)
    up, down = (
        grid.from_cdf(model.life, rounded) for rounded in (True, False)
    )
    life = _band(0, up.rest, down.rest, grid.size)
    band = life
    reads: List[_Read] = []
    while True:
        read = _Read(grid.step, *band)
        reads.append(read)
        if float(read.below(T)) < _TAIL:
            return grid, reads
        if len(reads) >= MAX_COUNT:
            raise NotImplementedError(
                f"More than {MAX_COUNT} failures are likely in a block "
                "interval: count the spares by simulation instead."
            )
        offset = band[0] + life[0]
        band = _band(
            offset,
            _convolve(band[1], life[1]),
            _convolve(band[2], life[2]),
            grid.size,
        )


def _cells(first: Any, second: _Read, x: np.ndarray) -> Tuple[int, int]:
    """For ``x`` falling: the range of its points outside which
    ``P(first < x) - P(second < x)`` is 0, both being 0 or both their
    totals, with ``first`` the earlier of two sums."""
    lo = first.span()[0]
    hi = second.span()[1]
    start = int(np.searchsorted(-x, -hi, side="left"))
    stop = int(np.searchsorted(-x, -lo, side="right"))
    return max(start - 1, 0), min(stop + 1, len(x))


class _Always:
    """``S_0 = 0``: ``P(S_0 < x)`` is 1 for any ``x > 0``."""

    total = 1.0

    @staticmethod
    def span() -> Tuple[float, float]:
        return 0.0, 0.0

    @staticmethod
    def below(x) -> np.ndarray:
        return (np.asarray(x, dtype=float) > 0.0).astype(float)

    @staticmethod
    def integral(x) -> np.ndarray:
        return np.maximum(np.asarray(x, dtype=float), 0.0)


def _block_lead_tails(
    model: Block, tau: float, kind: str, steps: int
) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, for a component under block
    replacement every ``T``, its repairs and block replacements in no time,
    in a lead time ``tau`` in the long run, from a random time
    (``"random"``) or before a replacement (``"arrival"``), on grids of
    about ``steps`` steps (see the module docstring)."""
    T = model.interval
    grid, sums = _interval_sums(model, steps)
    # The cells of the interval for the integrals over where a unit started
    # (dM, at their middles); before a replacement, cut where a lead time
    # from there first takes in a block time.
    edges = grid.step * np.arange(int(round(T / grid.step)) + 1)
    edges[-1] = T
    cut = T - math.fmod(tau, T)
    if kind == "arrival" and 0.0 < cut < T:
        if np.abs(edges - cut).min() > 1e-9 * grid.step:
            edges = np.sort(np.append(edges, cut))
    # M at the edges: each sum's CDF, 0 before its band and its total after.
    renewals = np.zeros(len(edges))
    after = np.zeros(len(edges) + 1)
    for read in sums:
        lo, hi = read.span()
        start, stop = np.searchsorted(edges, [lo, hi])
        renewals[start:stop] += read.below(edges[start:stop])
        after[stop] += read.total
    renewals += np.cumsum(after)[:-1]
    cells = (np.diff(renewals), 0.5 * (edges[:-1] + edges[1:]))
    if kind == "random":
        return _from_a_random_time(model, tau, steps, sums, *cells)
    return _before_a_replacement(model, tau, steps, sums, *cells)


def _from_a_random_time(
    model: Block,
    tau: float,
    steps: int,
    sums: List[_Read],
    dM: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """From a phase ``phi`` uniform on ``[0, T)``: within the interval
    (``phi < b``), and past its end (``phi >= b``), integrated over the
    phase in exchanged order, over where the unit in service at ``phi``
    started (``dM`` on the cells with middles ``y``)."""
    T = model.interval
    K = len(sums)
    b = max(T - tau, 0.0)
    within = np.zeros(K + 2)
    if b > 0.0:
        # The integrals of dM E[(end - S_k)^+], over the cells' ends: 0
        # before the sum's band, and linear in the end after it.
        ends = T - np.minimum(y, b)
        weights = np.concatenate([[0.0], np.cumsum(dM)])
        moments = np.concatenate([[0.0], np.cumsum(dM * ends)])
        ints = [(T, tau, float(dM @ ends))]
        for read in sums:
            lo, hi = read.span()
            start, stop = np.searchsorted(-ends, [-hi, -lo], side="right")
            inside = dot(dM[start:stop], read.integral(ends[start:stop]))
            past = read.total * moments[start] - read.moment[-1] * (
                weights[start]
            )
            ints.append(
                (
                    float(read.integral(T)),
                    float(read.integral(tau)),
                    float(inside + past),
                )
            )
        ints.append((0.0, 0.0, 0.0))
        mass = float(weights[-1])
        for s in range(1, K + 2):
            (_, tau0, Q0), (T1, tau1, Q1) = ints[s - 1], ints[s]
            within[s] = (T1 - tau1) - (Q0 - Q1) + mass * (tau0 - tau1)
    # B, the failures from phi to the interval's end, and C(z), the
    # replacements from new over what is left past it: their integrals
    # I[k, j] over the phase, from Lambda_j(v), the integral of P(C(z) =
    # j) over phi in [v, T), and the failures' dM p_{k-1}(T - y) by bands.
    left = T - y
    bands = []
    previous: Any = _Always
    for read in sums:
        start, stop = _cells(previous, read, left)
        part = left[start:stop]
        gap = previous.below(part) - read.below(part)
        bands.append((start, stop, dM[start:stop] * gap))
        previous = read
    at_T = np.array([float(read.below(T)) for read in sums])

    def column(at_b: float, at_v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                at_T[k] * at_b - dot(values, at_v[start:stop])
                for k, (start, stop, values) in enumerate(bands)
            ]
        )

    v = np.maximum(y, b)
    before_b, before_v = T - b, T - v
    lambdas = [before_b]
    columns = []
    for run, up, down, _ in _block_runs(model, tau, steps):
        read = _Read(run.step, 0, up.rest, down.rest, (up.atoms, down.atoms))
        whole = float(read.integral(tau))
        at_b = whole - float(read.integral(b + tau - T))
        at_v = whole - read.integral(v + tau - T)
        columns.append(column(before_b - at_b, before_v - at_v))
        lambdas.append(at_b)
        before_b, before_v = at_b, at_v
    columns.append(column(before_b, before_v))

    def alone(s: int) -> float:
        return (within[s] if s < len(within) else 0.0) + (
            lambdas[s - 1] if s - 1 < len(lambdas) else 0.0
        )

    return _assembled(alone, np.array(columns).T, T)


def _before_a_replacement(
    model: Block,
    tau: float,
    steps: int,
    sums: List[_Read],
    dM: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """After a replacement, by the stationarity of the times between them:
    after a failure at ``y`` (``dM``), its new unit's own failures up to
    the interval's end (``B'``), the block replacement there and the
    replacements from new past it (``C``); or after a block replacement,
    ``C``; over the ``M(T) + 1`` replacements of an interval."""
    T = model.interval
    K = len(sums)
    crossing = y > T - tau
    early = float(dM[~crossing].sum())
    w = T - y[crossing]
    z = tau - w
    dMc = dM[crossing]
    at_tau = np.array([float(read.below(tau)) for read in sums])
    # F^{*i}(w) for falling w: its total before ``start``, its band to
    # ``stop``, and 0 after.
    bands = []
    for read in sums:
        lo, hi = read.span()
        start = int(np.searchsorted(-w, -hi, side="right"))
        stop = int(np.searchsorted(-w, -lo, side="right"))
        bands.append((start, stop, read.below(w[start:stop]), read.total))

    def column(chance: np.ndarray) -> np.ndarray:
        weighted = dMc * chance
        sums_to = np.concatenate([[0.0], np.cumsum(weighted)])
        return np.array(
            [
                total * sums_to[start] + dot(values, weighted[start:stop])
                for start, stop, values, total in bands
            ]
        )

    previous = np.ones_like(z)
    reached = [float(dMc.sum())]
    psi = [1.0]
    columns = []
    for run, up, down, tail in _block_runs(model, tau, steps):
        read = _Read(run.step, 0, up.rest, down.rest, (up.atoms, down.atoms))
        cdf = read.below(z)
        columns.append(column(previous - cdf))
        reached.append(dot(dMc, cdf))
        psi.append(tail)
        previous = cdf
    columns.append(column(previous))

    def alone(s: int) -> float:
        total = early * at_tau[s - 1] if (tau < T and s <= K) else 0.0
        total += reached[s - 1] if s - 1 < len(reached) else 0.0
        return total + (psi[s] if s < len(psi) else 0.0)

    return _assembled(alone, np.array(columns).T, float(dM.sum()) + 1.0)


def _assembled(
    alone: Callable[[int], float], joint: np.ndarray, scale: float
) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, until below ``_TAIL``: the terms of
    one part of the count alone, ``alone(s)``, and those of two, the sum
    of ``joint[s - 2 - j, j]`` over ``j`` (rows for ``k = 1, 2, ...``), over
    ``scale``."""
    K, J = joint.shape
    out: List[float] = []
    for s in range(1, K + J + 2):
        total = alone(s)
        for j in range(max(0, s - 1 - K), min(s - 1, J)):
            total += joint[s - 2 - j, j]
        tail = total / scale
        out.append(min(max(tail, 0.0), 1.0))
        if tail < _TAIL:
            break
    return np.array(out)


def _convolved(a: np.ndarray, b: np.ndarray, last: int) -> np.ndarray:
    """The convolution of ``a`` and ``b`` to index ``last``."""
    from scipy.signal import fftconvolve

    if len(a) * len(b) <= 4_000_000:
        return np.convolve(a, b)[: last + 1]
    return np.maximum(fftconvolve(a, b)[: last + 1], 0.0)


def _lattice_tails(
    first: np.ndarray,
    cycle: np.ndarray,
    last: int,
    places: Optional[np.ndarray] = None,
) -> list:
    """``P(J_s <= last)``, ``s = 1, 2, ...``, until it is below ``_TAIL``:
    ``J_1`` has the distribution ``first`` (over ``0, 1, 2, ...`` tests),
    and each next one is a ``cycle`` (over ``0, 1, 2, ...``) more; or, with
    ``places``, the place of each test ``0, 1, ..., last``, a ``cycle[q]``
    more from one at the place ``q``."""
    tails: List[float] = []
    if last < 0:
        return [0.0]
    head = np.asarray(first, dtype=float)[: last + 1]
    step = np.asarray(cycle, dtype=float)[..., : last + 1]
    while len(tails) < MAX_COUNT:
        tail = float(head.sum())
        tails.append(min(tail, 1.0))
        if tail < _TAIL:
            return tails
        if places is None:
            head = _convolved(head, step, last)
            continue
        after = np.zeros(last + 1)
        at = places[: len(head)]
        for q in range(len(step)):
            part = np.where(at == q, head, 0.0)
            if part.any():
                moved = _convolved(part, step[q], last)
                after[: len(moved)] += moved
        head = after
    raise NotImplementedError(
        f"More than {MAX_COUNT} replacements are likely in the time: count "
        "them by simulation instead."
    )


def _cycles(model: Tested) -> np.ndarray:
    """A ``Tested`` model's cycles over ``0, 1, 2, ...`` tests, a row for
    each place (one, with tests that find every failure)."""
    cycle = np.atleast_2d(np.asarray(model.cycle, dtype=float))
    return np.concatenate([np.zeros((len(cycle), 1)), cycle], axis=1)


def _shares(cycles: np.ndarray) -> np.ndarray:
    """The long-run share of the replacements at each place: the
    stationary distribution of the places the cycles move between."""
    per, n = cycles.shape
    if per == 1:
        return np.ones(1)
    moves = np.array(
        [
            np.bincount((q + np.arange(n)) % per, cycles[q], per)
            for q in range(per)
        ]
    )
    moves = moves / moves.sum(axis=1, keepdims=True)
    system = np.vstack([moves.T - np.eye(per), np.ones(per)])
    rhs = np.concatenate([np.zeros(per), [1.0]])
    shares = np.maximum(np.linalg.lstsq(system, rhs, rcond=None)[0], 0.0)
    return shares / shares.sum()


def _places_tails(model: Tested, end: float, kind: str) -> np.ndarray:
    """``_tested_tails`` for tests that can miss a failure: the
    replacements a Markov renewal process over the places of the tests in
    the full tests' period (see the module docstring)."""
    T = model.interval
    cycles = _cycles(model)
    per, n = cycles.shape
    shares = _shares(cycles)
    if kind == "new":
        tests = 0
        if end > model.first:
            tests = int(math.ceil((end - model.first) / T - 1e-12))
        if tests == 0:
            return np.zeros(1)
        first = np.concatenate([[0.0], model.found(tests)])
        places = (model.position + np.arange(tests + 1) - 1) % per
        return np.array(_lattice_tails(first, cycles, tests, places))

    def mixed(parts: list, weights) -> np.ndarray:
        size = max(len(part) for part in parts)
        return sum(
            w * np.pad(part, (0, size - len(part)))
            for w, part in zip(weights, parts)
        )

    if kind == "arrival":
        # Back from a replacement at each place, in proportion to its share
        # of them: the cycles back are the reversed process's, over the
        # tests in the lead time.
        last = int(math.ceil(end / T - 1e-12)) - 1
        if last < 1:
            return np.zeros(1)
        lags = np.arange(min(n, last + 1))
        back = np.zeros((per, len(lags)))
        for q in range(per):
            if shares[q] > 0.0:
                before = (q - lags) % per
                back[q] = shares[before] * cycles[before, lags] / shares[q]
        parts = []
        for q in range(per):
            places = (q - np.arange(last + 1)) % per
            parts.append(np.array(_lattice_tails(back[q], back, last, places)))
        return mixed(parts, shares)
    # From a random time, its next test at each place as often: the next
    # replacement j tests on ends the cycle in progress, started a tests
    # before that test (see the module docstring), j up to the tests in the
    # lead time.
    lengths = cycles @ np.arange(n)
    rates = per * shares / float(shares @ lengths)
    whole = int(math.floor(end / T + 1e-12))
    share = end / T - whole
    if share < 1e-12:
        share = 0.0
    reach = whole + 1
    rows = -(-(n + reach) // per) + 1
    suffix = np.zeros((per, rows * per))
    for q in range(per):
        padded = np.zeros(rows * per)
        padded[:n] = cycles[q]
        # Each lag's sum with those whole periods beyond it.
        folded = padded.reshape(rows, per)[::-1].cumsum(axis=0)[::-1]
        suffix[q] = folded.ravel()
    parts = []
    for p in range(per):
        delay = np.zeros(reach + 1)
        for q in range(per):
            a0 = (p - q) % per or per
            delay[1:] += rates[q] * suffix[q, a0 : a0 + reach]  # noqa: E203
        places = (p + np.arange(reach + 1) - 1) % per
        fewer = np.array(_lattice_tails(delay, cycles, whole, places))
        if share:
            more = np.array(_lattice_tails(delay, cycles, reach, places))
            fewer = mixed([fewer, more], [1.0 - share, share])
        parts.append(fewer)
    return mixed(parts, np.full(per, 1.0 / per))


def _tested_tails(model: Tested, end: float, kind: str) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, for a component whose replacements
    fall on its tests (see the module docstring for ``kind``)."""
    if np.ndim(model.cycle) > 1:
        return _places_tails(model, end, kind)
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


def rate(model) -> float:
    """A component's replacements per unit time in the long run: one over
    its mean cycle (for a ``Tested`` one, its mean cycle in tests times
    the test interval; for a ``Block`` one, with repairs and block
    replacements in no time, the block replacement and the failures in a
    block interval, ``M(T) + 1``, over ``T``)."""
    if isinstance(model, Tested):
        cycles = _cycles(model)
        tests = float(_shares(cycles) @ (cycles @ np.arange(cycles.shape[1])))
        return 1.0 / (model.interval * tests)
    if isinstance(model, Block):
        counts = count(model, model.interval, "new")
        failures = float(np.arange(len(counts)) @ counts)
        return (failures + 1.0) / model.interval
    return 1.0 / model.mean_cycle


def count(model, end: float, kind: str) -> np.ndarray:
    """The distribution of a component's replacements (see
    ``_count_tails`` for ``kind``): their probabilities for ``0, 1, 2,
    ...``, from grids refined until they agree to ``TOLERANCE`` (a
    ``Block`` in a lead time with its repairs and block replacements in no
    time; a ``Tested`` exactly, with no grid).

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
            if kind == "new":
                return _block_tails(model, end, steps)
            return _block_lead_tails(model, end, kind, steps)
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
