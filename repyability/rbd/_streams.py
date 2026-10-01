"""Keyed streams of random draws for the availability simulations.

Each random quantity a ``RepairableRBD`` simulation draws comes from a stream
of its own: a component's times to failure, its repair times, its
maintenance or test times, and the amounts charged for it when a cost is a
distribution. A stream is named by the component's place in the diagram
(its node name, and the node names down through nested RBDs) and the kind of
quantity, and its ``k``-th draw in simulation ``r`` depends on nothing but
the run's entropy (its seed), the stream's name, ``r`` and ``k``:

- the simulations are taken in blocks of the stream's ``width``: many
  simulations to a block for a component that fails a few times in the
  window, a single one for a component that fails thousands of times. Each
  block of each stream has a generator of its own, seeded from the entropy,
  the stream's name and the block's position;
- a block's uniforms are laid out one row per draw and one column per
  simulation (row ``k`` holds every simulation's ``k``-th uniform), a chunk
  of rows at a time, so a block extended by more rows keeps the rows it
  had; and the stream's quantile function turns them into draws, a whole
  chunk at a time.

So a simulation draws the same numbers however many simulations the run
has, in however many processes or threads, in whatever order, and whichever
engine runs it: the event loop in Python and the compiled one read the same
values. A component in the same place in another system draws the same
uniforms (common random numbers, for ``compare``). With antithetic pairs, a
column serves a pair: the second simulation draws ``1 - u`` for each
uniform ``u`` of the first.

A stream's width follows from how many draws a simulation is expected to
take from it (``first_rows``, ``block_width``), which depends on its models
and the window. It decides which uniform goes to which simulation, so it is
part of what the draws are: ``compare`` gives the two systems' streams the
same widths. How many rows a chunk has only decides how the draws are
computed: a simulation that needs more rows than its block has gets another
chunk, and the uniforms are the same however the rows are chunked.
"""

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# The kinds of quantity a stream draws.
FAILURE, REPAIR, DURATION = 0, 1, 2
#: The uniforms a standby group's switches are decided by.
SWITCH = 7
#: The uniforms an imperfectly repaired unit's lives are drawn from, given
#: its virtual age (a unit as new draws from its ``FAILURE`` stream).
AGED = 8
#: The kind of each cost that can be a distribution, by its cost key.
COST_KINDS = {
    "repair_cost": 3,
    "replace_cost": 4,
    "preventive_cost": 5,
    "inspection_cost": 6,
}

#: The draws a block of a stream aims for: its width times the rows each
#: simulation is expected to use. Seeding a block's generator and calling
#: the quantile function cost some microseconds, spread over this many
#: draws.
BLOCK_DRAWS = 65536
#: The widest block, in simulations (antithetic pairs, with pairs).
MAX_WIDTH = 1024
#: The most rows a chunk has.
MAX_ROWS = 2**22

Name = Tuple[tuple, int]


def path_key(path: tuple) -> int:
    """A place's 64-bit name: a hash of its ``repr``, so the same in every
    process for node names that print the same way (strings, numbers and
    tuples of them)."""
    digest = hashlib.blake2b(repr(path).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def first_rows(expected: float) -> int:
    """The rows of a stream's first chunk, for a stream a simulation is
    expected to draw from ``expected`` times (16 if that is not known):
    that many and four standard deviations of a Poisson count more, and a
    few spare, so that almost every simulation finds all its draws
    there."""
    if not np.isfinite(expected) or expected < 0.0:
        expected = 16.0
    rows = np.ceil(expected + 4.0 * np.sqrt(expected) + 4.0)
    return int(min(rows, MAX_ROWS))


def block_width(rows: int) -> int:
    """The simulations in a block of a stream whose first chunk has
    ``rows`` rows: the largest power of 2 up to ``MAX_WIDTH`` with at most
    ``BLOCK_DRAWS`` draws in the chunk (at least 1)."""
    width = 1
    while width < MAX_WIDTH and 2 * width * rows <= BLOCK_DRAWS:
        width *= 2
    return width


def cost_sampler(cost) -> Callable[[np.ndarray], np.ndarray]:
    """The draws of a cost given as a distribution, from uniforms: its
    quantiles, clipped at 0 (validation has made a negative one
    negligible)."""

    def draw(u: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(cost.qf(u), dtype=float), 0.0)

    return draw


@dataclass(frozen=True)
class Spec:
    """One stream of a run: its name (the place and the kind of quantity),
    the function from uniforms to draws, the rows of its first chunk and
    the width of its blocks."""

    path: tuple
    kind: int
    sampler: Callable[[np.ndarray], np.ndarray]
    rows: int
    width: int

    @property
    def name(self) -> Name:
        return self.path, self.kind

    def chunk_rows(self, chunk: int) -> int:
        """The rows of chunk ``chunk`` of a block: the first as planned,
        and each further one twice the one before."""
        if chunk == 0:
            return self.rows
        return min(max(self.rows, 8) << min(chunk - 1, 32), MAX_ROWS)


def _values(sampler, u: np.ndarray) -> np.ndarray:
    """``sampler`` applied to the 1-d uniforms ``u``, as floats of the
    same shape."""
    values = np.asarray(sampler(u), dtype=float)
    if values.shape != u.shape:
        values = np.broadcast_to(values.ravel(), u.shape)
    return values


class Block:
    """One block of one stream: every draw of ``width`` simulations (or
    antithetic pairs), extended a chunk of rows at a time. ``values`` holds
    a row per simulation, ``values[j, k]`` being simulation ``j``'s
    ``k``-th draw (with pairs, a pair's first simulation and then its
    second, in turn); the uniforms are laid out the other way round (see
    the module docstring)."""

    __slots__ = ("_generator", "_spec", "_antithetic", "chunks", "values")

    def __init__(self, entropy, spec: Spec, index: int, antithetic: bool):
        key = path_key(spec.path)
        seeds = np.random.SeedSequence(
            entropy,
            spawn_key=(key & 0xFFFFFFFF, key >> 32, spec.kind, index),
        )
        self._generator = np.random.Generator(np.random.PCG64(seeds))
        self._spec = spec
        self._antithetic = antithetic
        self.chunks = 0
        self.values = np.empty((0, 0))
        self.extend()

    def extend(self) -> None:
        """Add the block's next chunk of draws: ``chunk_rows`` more for
        each simulation."""
        spec = self._spec
        rows, width = spec.chunk_rows(self.chunks), spec.width
        u = self._generator.random(rows * width)
        draws = _values(spec.sampler, u).reshape(rows, width).T
        if self._antithetic:
            chunk = np.empty((2 * width, rows))
            chunk[0::2] = draws
            chunk[1::2] = _values(spec.sampler, 1.0 - u).reshape(rows, width).T
        else:
            chunk = draws
        if self.chunks:
            chunk = np.concatenate((self.values, chunk), axis=1)
        self.values = np.ascontiguousarray(chunk)
        self.chunks += 1


class Plan:
    """A run's streams, by name, and the entropy their blocks are seeded
    from. ``columns(spec)`` is how many simulations a block of ``spec``
    holds: its width, or twice that with antithetic pairs."""

    def __init__(self, entropy, antithetic: bool, specs: Dict[Name, Spec]):
        self.entropy = entropy
        self.antithetic = antithetic
        self.specs = specs

    def columns(self, spec: Spec) -> int:
        return spec.width * (2 if self.antithetic else 1)

    def block(self, spec: Spec, index: int) -> Block:
        return Block(self.entropy, spec, index, self.antithetic)


def replication_seed(entropy, replication: int) -> np.ndarray:
    """The seed numpy's global RNG is given at the start of simulation
    ``replication``, for the draws that do not come from a stream."""
    seeds = np.random.SeedSequence(entropy, spawn_key=(replication,))
    return seeds.generate_state(4)


def entropy_of(seed) -> int:
    """A run's entropy: a number drawn from numpy's global RNG, or, with a
    seed, from that RNG as the seed would seed it (without touching it).
    So a run with ``seed=s`` is the run ``np.random.seed(s)`` makes
    reproducible, and takes the same seeds (0 to ``2**32 - 1``)."""
    if seed is None:
        return int(np.random.randint(0, 2**62, dtype=np.int64))
    return int(np.random.RandomState(seed).randint(0, 2**62, dtype=np.int64))


class Run:
    """The Python event loop's side of a run: the simulation it is on, the
    streams the components draw from, and whether numpy's global RNG is
    seeded afresh for each simulation (for the components whose draws do
    not come from a stream)."""

    def __init__(self, plan: Plan, reseed: bool):
        self.plan = plan
        self.replication = -1
        self.reseed = reseed
        self._streams: Dict[Name, "Stream"] = {}

    def stream(self, path: tuple, kind: int) -> Optional["Stream"]:
        """The stream named ``(path, kind)``, or None if the run has none
        (the quantity is drawn some other way)."""
        spec = self.plan.specs.get((path, kind))
        if spec is None:
            return None
        stream = self._streams.get(spec.name)
        if stream is None:
            stream = self._streams[spec.name] = Stream(self, spec)
        return stream

    def begin(self, replication: int) -> None:
        """Start simulation ``replication``."""
        self.replication = replication
        if self.reseed:
            np.random.seed(replication_seed(self.plan.entropy, replication))


class Stream:
    """One stream's draws for the Python event loop, one at a time: the
    ``k``-th call in a simulation returns its ``k``-th draw."""

    __slots__ = (
        "_run",
        "_spec",
        "_columns",
        "_block",
        "_index",
        "_column",
        "_j",
        "_k",
        "_replication",
    )

    def __init__(self, run: Run, spec: Spec):
        self._run = run
        self._spec = spec
        self._columns = run.plan.columns(spec)
        self._block: Optional[Block] = None
        self._index = -1
        self._column: list = []
        self._j = 0
        self._k = 0
        self._replication = -1

    def draw(self) -> float:
        if self._replication != self._run.replication:
            self._start()
        k = self._k
        if k == len(self._column):
            block = self._block
            block.extend()  # type: ignore[union-attr]
            self._column = block.values[self._j].tolist()  # type: ignore
        self._k = k + 1
        return self._column[k]

    def _start(self) -> None:
        replication = self._replication = self._run.replication
        index, self._j = divmod(replication, self._columns)
        if index != self._index:
            self._block = self._run.plan.block(self._spec, index)
            self._index = index
        self._column = self._block.values[self._j].tolist()  # type: ignore
        self._k = 0
