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

  With repairs or block replacements that take time (#160), a unit down
  at a block time is not replaced there, and an interval need not start
  with a new unit, so the demand is counted from a typical replacement
  (its Palm distribution): a failure at a phase of the interval, or a
  block replacement. Each is followed by the recursion above, from its
  next unit's start, the rows of all phases at once; their weights are
  the long-run failures at each phase and block replacements an
  interval, followed interval by interval until they settle (the next
  unit's start after a block time is its carry-over). A replacement finds
  ``s`` before it within ``tau`` as often as it finds ``s`` after it,
  ``G_s(tau)``, ``G_s`` the CDF of the ``s``-th one after a typical one;
  and from a random time, ``P(N >= s) = lambda * integral over [0, tau)
  of G_{s-1} - G_s`` (Campbell's formula), ``lambda`` the long-run
  replacements' rate. Only the life is rounded both ways: the repairs and
  block replacements, often far shorter than an interval, are split
  between the grid points either side, keeping their mean (rounded, one
  shorter than a step is dropped or stretched to one, which the mean of
  the two roundings does not undo). A block time measured from a failure
  at a rounded time is rounded the other way (a failure rounded up puts
  it earlier), so it is moved a step, to be rounded as the rest are.
  ``G_s`` jumps at whole intervals (the block times, from a failure at
  any phase, end there), so a lead time of whole intervals falls on a
  jump: each rounding is read on its own there (rounded up, a value is in
  the step before its grid point, and down, in the one after), where the
  half-step reading of their mean blurs the jump over a step. The grids
  are refined by halving the step and extrapolated, their error falling
  as its square.
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

import functools
import math
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from repyability.utils.vectors import dot

#: How closely the counts' probabilities are computed.
TOLERANCE = 1e-6
#: The grid's first and largest number of steps up to the time.
FIRST_STEPS, MAX_STEPS = 2**9, 2**16
#: The same, per block interval, for a block-replaced unit whose repairs or
#: block replacements take time, whose rows cost the steps squared (#160).
PALM_FIRST_STEPS, PALM_MAX_STEPS = 2**7, 2**11
#: A count is followed until the probability of more falls below this.
_TAIL = 1e-12
#: The most replacements a count follows.
MAX_COUNT = 2_000
#: Gauss-Legendre nodes and weights for integrals over grid cells.
_GAUSS = np.polynomial.legendre.leggauss(4)
#: The pieces of a step a split duration is taken on (``_Grid.split``).
_SPLIT = 16

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

    def split(self, cdf: Cdf) -> _Dist:
        """A duration with CDF ``cdf``: its mass at 0 an atom, and the rest
        of each value's shared between the grid points either side of it,
        in proportion to its nearness to each, which keeps its mean (the
        values taken on ``_SPLIT`` pieces of each step). Unlike rounding, it
        neither drops a duration much shorter than a step nor stretches it
        to one (#160)."""
        pieces = self.size * _SPLIT
        edges = (self.step / _SPLIT) * np.arange(pieces + 1)
        values = np.clip(np.asarray(cdf(edges), dtype=float), 0.0, 1.0)
        values = np.maximum.accumulate(values)
        masses = np.diff(values)
        middles = (np.arange(pieces) + 0.5) / _SPLIT
        low = np.floor(middles).astype(int)
        share = middles - low
        rest = np.bincount(
            low, masses * (1.0 - share), minlength=self.size + 1
        )
        rest += np.bincount(low + 1, masses * share, minlength=self.size + 1)
        atoms = np.zeros(self.size)
        atoms[0] = values[0]
        return _Dist(rest[: self.size], atoms)

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
    by that block's replacement in no time (or, split, in under a step):
    those are kept apart."""
    start = _sum(failed, repair) + _sum(blocks, maintenance)
    size = len(start.rest)
    before = np.zeros(size)
    if not up or period is None or period >= size:
        return start, before
    on = np.arange(period, size, period)
    total = start.rest[on] + start.atoms[on]
    renewed = blocks.atoms[on] * float(
        maintenance.atoms[0] + maintenance.rest[0]
    )
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
        self.each = (np.cumsum(up), np.cumsum(down))
        self.rest = 0.5 * (self.each[0] + self.each[1])
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

    def below_each(self, x: float) -> float:
        """``P(X < x)`` as ``below``, but each rounding's continuous part
        read on its own: rounded up, a mass is in the step before its grid
        point, and rounded down, in the step after it, spread evenly. On
        the grid, one rounding is read exactly, and so is a density that
        jumps at a grid point (a block time), which ``below``'s mean of the
        two blurs over a step (#160)."""
        position = x / self.step - self.offset
        rest = 0.0
        for cumulative, start in zip(self.each, (position, position - 1.0)):
            if start <= 0.0 or not len(cumulative):
                continue
            low = int(math.floor(start))
            share = start - low
            last = len(cumulative) - 1
            value = (1.0 - share) * cumulative[min(low, last)]
            if share:
                value += share * cumulative[min(low + 1, last)]
            rest += 0.5 * float(value)
        if self.atoms is not None:
            reach = int(math.ceil(x / self.step * (1.0 - 1e-12) - 1e-9))
            reach = min(max(reach - self.offset, 0), len(self.atoms) - 1)
            rest += float(self.atoms[reach])
        return min(rest, 1.0)

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


def carries_over(model: Block) -> bool:
    """Whether a block-replaced unit's repairs or block replacements take
    time, so that an interval need not start with a new unit."""
    zero = np.zeros(1)
    if float(np.ravel(model.repair(zero))[0]) < 1.0:
        return True
    return model.maintenance is not None and (
        float(np.ravel(model.maintenance(zero))[0]) < 1.0
    )


def _rows_fft(a: np.ndarray, b: np.ndarray, size: int = 0) -> np.ndarray:
    """Each row of ``a`` convolved with ``b`` and cut to ``size`` (the
    rows' length by default), clipped at 0 (as ``_fft``), on every core.
    ``b``'s tail past its last ``_TAIL * 1e-6`` of mass is left out."""
    from scipy import fft

    size = size or a.shape[1]
    tail = np.cumsum(b[::-1])[::-1]
    support = np.flatnonzero(tail > _TAIL * 1e-6)
    if not len(support) or not a.any():
        return np.zeros((a.shape[0], size))
    # Neither's terms past ``size`` reach the terms kept.
    a = a[:, :size]
    kernel = b[: min(support[-1] + 1, size)]
    length = fft.next_fast_len(a.shape[1] + len(kernel) - 1, real=True)
    product = fft.rfft(a, length, axis=1, workers=-1)
    product *= fft.rfft(kernel, length)
    out = fft.irfft(product, length, axis=1, workers=-1)[:, :size]
    if out.shape[1] < size:
        out = np.pad(out, ((0, 0), (0, size - out.shape[1])))
    return np.maximum(out, 0.0, out=out)


def _rows_sum(
    rest: np.ndarray, atoms: np.ndarray, other: _Dist, size: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """``_sum`` of each row's distribution and ``other``, cut to ``size``
    (the rows' length by default); its atoms are None when it has none."""
    total = _rows_fft(rest + atoms, other.rest + other.atoms, size)
    if not (atoms.any() and other.atoms.any()):
        return total, None
    held = _rows_fft(atoms, other.atoms, size)
    return np.maximum(total - held, 0.0), held


def _rows_step(
    rest: np.ndarray,
    atoms: np.ndarray,
    before: np.ndarray,
    life: _Dist,
    period: int,
    up: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_block_step`` row by row, from each row's start (``rest``,
    ``atoms`` and ``before``): its failures' times and its block
    replacements (atoms on the block times)."""
    rows, size = rest.shape
    failed_rest = np.zeros((rows, size))
    failed_atoms = np.zeros((rows, size))
    blocks = np.zeros((rows, size))
    width = period + 1
    cut = _Dist(life.rest[:width], life.atoms[:width])
    dead = float(life.atoms[0])
    for lo in range(0, size, period):
        hi = lo + period
        if hi < size:
            failed_atoms[:, hi] += before[:, hi] * dead
            blocks[:, hi] += before[:, hi] * (1.0 - dead)
        part_rest, part_atoms = rest[:, lo:hi], atoms[:, lo:hi]
        if not (part_rest.any() or part_atoms.any()):
            continue
        keep = width if up else width - 1
        stop = min(lo + keep, size)
        out_rest, out_atoms = _rows_sum(part_rest, part_atoms, cut, stop - lo)
        failed_rest[:, lo:stop] += out_rest
        survive = part_rest.sum(axis=1) + part_atoms.sum(axis=1)
        survive -= out_rest.sum(axis=1)
        if out_atoms is not None:
            failed_atoms[:, lo:stop] += out_atoms
            survive -= out_atoms.sum(axis=1)
        if hi < size:
            blocks[:, hi] += np.maximum(survive, 0.0)
    return failed_rest, failed_atoms, blocks


def _rows_starts(
    failed_rest: np.ndarray,
    failed_atoms: np.ndarray,
    blocks: np.ndarray,
    repair: _Dist,
    maintenance: _Dist,
    period: int,
    up: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_block_starts`` row by row: the next units' starts."""
    start_rest, held = _rows_sum(failed_rest, failed_atoms, repair)
    start_atoms = np.zeros_like(start_rest) if held is None else held
    rows, size = start_rest.shape
    # The block replacements are on the block times alone: the starts after
    # them are their maintenance's, shifted there.
    reach = [
        int(np.flatnonzero(part)[-1]) + 1 if part.any() else 0
        for part in (maintenance.rest, maintenance.atoms)
    ]
    for at in range(0, size, period):
        blocked = blocks[:, at]
        if not blocked.any():
            continue
        for start, part, span in zip(
            (start_rest, start_atoms),
            (maintenance.rest, maintenance.atoms),
            reach,
        ):
            span = min(span, size - at)
            if span:
                start[:, at : at + span] += (
                    blocked[:, None] * part[None, :span]
                )
    before = np.zeros((rows, size))
    if not up or period >= size:
        return start_rest, start_atoms, before
    on = np.arange(period, size, period)
    total = start_rest[:, on] + start_atoms[:, on]
    renewed = blocks[:, on] * float(maintenance.atoms[0] + maintenance.rest[0])
    before[:, on] = np.maximum(total - renewed, 0.0)
    start_rest[:, on] = 0.0
    start_atoms[:, on] = np.minimum(renewed, total)
    return start_rest, start_atoms, before


def _block_pieces(
    model: Block, grid: _Grid, up: bool
) -> Tuple[_Dist, _Dist, _Dist]:
    """A block-replaced unit's life on ``grid``, rounded up or down, and
    its repair's and block replacement's durations, split
    (``_Grid.split``): often much shorter than a block interval, they are
    then right on a grid of the interval (#160)."""
    life = grid.from_cdf(model.life, up)
    repair = grid.split(model.repair)
    maintenance = (
        grid.split(model.maintenance)
        if model.maintenance is not None
        else grid.unit()
    )
    return life, repair, maintenance


def _reach(model: Block, step: float, period: int) -> int:
    """How many steps past a start its repair or block replacement may
    last, but for a negligible chance (at least an interval)."""
    reach = period
    while True:
        left = 1.0 - float(np.ravel(model.repair(np.array([reach * step])))[0])
        if model.maintenance is not None:
            left = max(
                left,
                1.0
                - float(
                    np.ravel(model.maintenance(np.array([reach * step])))[0]
                ),
            )
        if left < _TAIL:
            return reach
        if reach > 64 * period:
            raise NotImplementedError(
                "A repair or block replacement lasting more than 64 block "
                "intervals is likely: count the spares by simulation "
                "instead."
            )
        reach *= 2


@functools.lru_cache(maxsize=64)
def _settled(model: Block, period: int, up: bool) -> Tuple[np.ndarray, float]:
    """The long run at the block times (#160): an interval's expected
    failures at each phase (grid points ``0`` to ``period``, the last
    rounded onto the block time from before it) and its chance of a block
    replacement, followed from new interval by interval, each from the
    starts of next units that the ones before carry over (rounded up onto
    a block time, a start is before it, and replaced there), until they
    settle."""
    step = model.interval / period
    size = period + _reach(model, step, period) + 3
    grid = _Grid((size - 3) * step, model.interval, size - 3)
    life, repair, maintenance = _block_pieces(model, grid, up)
    start, waiting = grid.unit(), np.zeros(size)
    previous: Optional[Tuple[np.ndarray, float]] = None
    for _ in range(MAX_COUNT):
        fails = np.zeros(period + 1)
        block = 0.0
        carry = _Dist(np.zeros(size), np.zeros(size))
        carried = np.zeros(size)

        def pass_on(current: _Dist, before: np.ndarray) -> None:
            """Starts on this interval's end or after it are the next
            intervals', and so are those before a later block time."""
            carry.rest[: size - period] += current.rest[period:]
            carry.atoms[: size - period] += current.atoms[period:]
            current.rest[period:] = 0.0
            current.atoms[period:] = 0.0
            carried[1 : size - period] += before[period + 1 :]
            before[period + 1 :] = 0.0

        current = _Dist(start.rest.copy(), start.atoms.copy())
        before = waiting.copy()
        pass_on(current, before)
        for _ in range(MAX_COUNT):
            failed, blocked = _block_step(current, before, life, period, up)
            fails += failed.rest[: period + 1] + failed.atoms[: period + 1]
            block += float(blocked.atoms[period])
            current, before = _block_starts(
                failed, blocked, repair, maintenance, period, up
            )
            pass_on(current, before)
            left = current.rest.sum() + current.atoms.sum() + before[period]
            if left < _TAIL:
                break
        else:
            raise NotImplementedError(
                f"More than {MAX_COUNT} replacements are likely in a block "
                "interval: count the spares by simulation instead."
            )
        # The next units' starts make a distribution: its mass is kept at
        # one, against the rounding's drift.
        mass = float(carry.rest.sum() + carry.atoms.sum() + carried.sum())
        start, waiting = carry.scaled(1.0 / mass), carried / mass
        if previous is not None:
            moved = max(
                float(np.abs(fails - previous[0]).max()),
                abs(block - previous[1]),
            )
            if moved < _SETTLED:
                return fails, block
        previous = (fails, block)
    raise NotImplementedError(
        "The block intervals did not settle into a long run: count the "
        "spares by simulation instead."
    )


def _palm_mixtures(
    model: Block,
    tau: float,
    period: int,
    up: bool,
    fails: np.ndarray,
    block: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """For ``j = 1, 2, ...``, the ``j``-th replacement after a typical
    one, relative to it, on the grid up to ``tau``: rounded masses and
    atoms, over the typical one's kinds, a failure at each phase (weights
    ``fails``) or a block replacement (``block``), each a row followed by
    the block recursion from the start of its next unit (see the module
    docstring)."""
    step = model.interval / period
    K = int(math.ceil(tau / step)) + 3
    size = period + 1 + K + 3
    grid = _Grid((size - 3) * step, model.interval, size - 3)
    life, repair, maintenance = _block_pieces(model, grid, up)
    phases = np.flatnonzero(fails > 0.0)
    rows = len(phases) + 1
    weights = np.append(fails[phases], block)
    weights = weights / weights.sum()
    # A row's typical replacement in its interval's frame: a failure at its
    # phase, or the block replacement at the interval's start.
    origins = np.append(phases, 0)
    failed_rest = np.zeros((rows, size))
    blocks = np.zeros((rows, size))
    failed_rest[np.arange(rows - 1), phases] = 1.0
    blocks[rows - 1, 0] = 1.0
    start_rest, start_atoms, before = _rows_starts(
        failed_rest,
        np.zeros((rows, size)),
        blocks,
        repair,
        maintenance,
        period,
        up,
    )
    window = origins[:, None] + np.arange(K)[None, :]
    on = np.zeros(size, dtype=bool)
    on[::period] = True
    times = np.flatnonzero(on)
    shifted = times + (1 if up else -1)
    inside = (shifted >= 0) & (shifted < size)
    failure = (np.arange(rows) < rows - 1).astype(float)[:, None]
    past = np.arange(size)[None, :] >= (origins + K)[:, None]
    mixtures: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(MAX_COUNT):
        failed_rest, failed_atoms, blocks = _rows_step(
            start_rest, start_atoms, before, life, period, up
        )
        rest = failed_rest.copy()
        atoms = failed_atoms + blocks
        # Measured from a failure, at a rounded time, a block time is a
        # rounded one, the other way (a failure rounded up puts it
        # earlier): a step on, it is rounded as the rest are.
        moved = atoms[:, on] * failure
        atoms[:, on] -= moved
        rest[:, shifted[inside]] += moved[:, inside]
        mixed_rest = weights @ np.take_along_axis(rest, window, axis=1)
        mixed_atoms = weights @ np.take_along_axis(atoms, window, axis=1)
        mixtures.append((mixed_rest, mixed_atoms))
        if mixed_rest.sum() + mixed_atoms.sum() < _TAIL:
            return mixtures
        start_rest, start_atoms, before = _rows_starts(
            failed_rest, failed_atoms, blocks, repair, maintenance, period, up
        )
        # A start past a row's window cannot reach it.
        start_rest[past] = 0.0
        start_atoms[past] = 0.0
        before[past] = 0.0
    raise NotImplementedError(
        f"More than {MAX_COUNT} replacements are likely in the lead time: "
        "count the spares by simulation instead."
    )


#: How closely an interval's long-run failures settle (``_settled``).
_SETTLED = 1e-12


@functools.lru_cache(maxsize=64)
def _palm(model: Block, tau: float, steps: int) -> Tuple[tuple, tuple, float]:
    """``P(N >= s)``, ``s = 1, 2, ...``, in a lead time ``tau`` in the long
    run, from a random time and before a replacement, for a block-replaced
    unit whose repairs or block replacements take time (#160), on grids of
    ``steps`` steps an interval, and the long-run replacements' rate (see
    the module docstring)."""
    step = model.interval / steps
    rates, parts = [], []
    for up in (True, False):
        fails, block = _settled(model, steps, up)
        rates.append(float(fails.sum() + block) / model.interval)
        parts.append(_palm_mixtures(model, tau, steps, up, fails, block))
    reads = []
    for j in range(max(len(part) for part in parts)):
        (up_rest, up_atoms), (down_rest, down_atoms) = (
            part[j] if j < len(part) else (np.zeros(1), np.zeros(1))
            for part in parts
        )
        size = max(len(up_rest), len(down_rest))

        def padded(a: np.ndarray) -> np.ndarray:
            return np.pad(a, (0, size - len(a)))

        reads.append(
            _Read(
                step,
                0,
                padded(up_rest),
                padded(down_rest),
                (padded(up_atoms), padded(down_atoms)),
            )
        )
    rate = 0.5 * (rates[0] + rates[1])
    arrival = [read.below_each(tau) for read in reads]
    integrals = [tau] + [float(read.integral(tau)) for read in reads]
    random = [
        rate * (integrals[s - 1] - integrals[s])
        for s in range(1, len(integrals))
    ]
    return tuple(_cut(random)), tuple(_cut(arrival)), rate


def _palm_tails(model: Block, tau: float, kind: str) -> np.ndarray:
    """``P(N >= s)``, ``s = 1, 2, ...``, in a lead time ``tau`` (see
    ``_palm``), on grids of an interval refined by halving its step: their
    error falls as the step squared, so each pair is extrapolated (a third
    of their difference past the finer), until two extrapolations agree
    to ``10 * TOLERANCE``, which puts the finer's error at about
    ``TOLERANCE``, or ``PALM_MAX_STEPS``."""
    which = 0 if kind == "random" else 1

    def on(steps: int) -> np.ndarray:
        return np.array(_palm(model, tau, steps)[which])

    def padded(a: np.ndarray, size: int) -> np.ndarray:
        return np.pad(a, (0, size - len(a)))

    steps = PALM_FIRST_STEPS
    coarse = on(steps)
    previous: Optional[np.ndarray] = None
    while True:
        steps *= 2
        fine = on(steps)
        size = max(len(coarse), len(fine))
        guess = (
            padded(fine, size)
            + (padded(fine, size) - padded(coarse, size)) / 3.0
        )
        if previous is not None:
            size = max(size, len(previous))
            change = np.abs(padded(guess, size) - padded(previous, size))
            # What the extrapolations leave falls as the step cubed or
            # faster: the finer's error is at most about a seventh of the
            # change.
            if change.max(initial=0.0) <= 10.0 * TOLERANCE:
                return guess
        if steps >= PALM_MAX_STEPS:
            return guess
        previous, coarse = guess, fine


def _cut(tails: List[float]) -> List[float]:
    """Tails clipped to ``[0, 1]``, to the first below ``_TAIL``."""
    out: List[float] = []
    for tail in tails:
        out.append(min(max(tail, 0.0), 1.0))
        if tail < _TAIL:
            break
    return out


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
    block interval, ``M(T) + 1``, over ``T``, and otherwise those of an
    interval once settled, see ``_settled``)."""
    if isinstance(model, Tested):
        cycles = _cycles(model)
        tests = float(_shares(cycles) @ (cycles @ np.arange(cycles.shape[1])))
        return 1.0 / (model.interval * tests)
    if isinstance(model, Block):
        if carries_over(model):
            return _palm_rate(model)
        counts = count(model, model.interval, "new")
        failures = float(np.arange(len(counts)) @ counts)
        return (failures + 1.0) / model.interval
    return 1.0 / model.mean_cycle


def _palm_rate(model: Block) -> float:
    """A block-replaced unit's replacements per unit time in the long run
    when its repairs or block replacements take time (#160): an interval's
    failures and block replacement once settled, over ``T``, from grids
    refined until they agree to ``TOLERANCE``, relatively."""
    previous = None
    steps = PALM_FIRST_STEPS
    while True:
        rate = 0.0
        for up in (True, False):
            fails, block = _settled(model, steps, up)
            rate += 0.5 * float(fails.sum() + block) / model.interval
        done = previous is not None and abs(rate - previous) <= (
            3.0 * TOLERANCE * rate
        )
        if done or steps >= PALM_MAX_STEPS:
            return rate
        previous, steps = rate, 2 * steps


def count(model, end: float, kind: str) -> np.ndarray:
    """The distribution of a component's replacements (see
    ``_count_tails`` for ``kind``): their probabilities for ``0, 1, 2,
    ...``, from grids refined until they agree to ``TOLERANCE`` (a
    ``Block`` in a lead time with repairs or block replacements that take
    time, see ``_palm_tails``; a ``Tested`` exactly, with no grid).

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

    if isinstance(model, Block) and kind != "new" and carries_over(model):
        tails = _palm_tails(model, end, kind)
        tails = np.minimum.accumulate(np.clip(tails, 0.0, 1.0))
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
