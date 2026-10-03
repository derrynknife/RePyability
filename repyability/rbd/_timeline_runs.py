"""``RepairableRBD.simulate_timelines`` (#157): a run's simulations as
timelines, each component's history and the system's.

The simulations are those ``availability`` runs with the same seed, and
their histories are the event loop's, whichever way they are made:

- **Recorded** by the event loop as it runs, on the engine ``engine``
  chooses: the Python loop (``RepairableRBD._replicate`` with
  ``_Context.history``) or numba's (``_kernel._simulate``, recording), in
  however many processes or threads. Each records each component's
  changes and each of the system's, with the component whose change made
  it (see ``Records``).
- **Drawn** from the streams, on the Python engine, when the components
  are independent (plain units with streamed models, and nested RBDs of
  them, held working or broken or not): each unit's lives and repairs, the
  draws the event loop reads (see ``_streams``), added up as it adds them,
  a batch of simulations at once, so the times are the event loop's to the
  last bit; and the system's history merged up the diagram (see
  ``repyability.timelines``). A simulation in which changes of different
  components fall at the same time, which the merge might take in another
  order than the event loop, is run in the loop instead.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Dict, Hashable, List, NamedTuple, Optional, Tuple

import numpy as np

from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _streams
from repyability.timelines import (
    Timelines,
    _constant,
    _Data,
    _leaf,
    _positions,
    _stacked,
    _system_merge,
)

#: The most simulations whose draws are laid out at once.
_BATCH = 4096
#: The most draws laid out at once, for memory: fewer simulations at once
#: for a component that changes state more often.
_BATCH_DRAWS = 2**21
#: How many times its planned draws (see ``_streams.first_rows``) a
#: simulation may take before its component is taken to change state
#: without end.
_MOST = 64
#: The simulations drawn and merged at once (a multiple of the widest
#: stream block), on one thread each: a smaller merge sorts its changes
#: faster, each.
_PART = 2048


class Block(NamedTuple):
    """Simulations' histories, as the event loops record them: each
    component's state at 0 (``start``, a row a simulation) and its changes
    (``bounds[c]`` to ``bounds[c + 1]`` of ``times`` and ``planned``, the
    time and whether it is a planned change down; simulation ``s``'s from
    ``offsets[c, s]`` to ``offsets[c, s + 1]`` of those, in the order they
    happened); the system's state at 0 (``up``); and each simulation's
    changes of the system's (``system_offsets``: the time, the component
    whose change made it, which of that component's changes it was, and
    whether it was planned)."""

    start: np.ndarray
    bounds: np.ndarray
    offsets: np.ndarray
    times: np.ndarray
    planned: np.ndarray
    up: np.ndarray
    system_offsets: np.ndarray
    system_times: np.ndarray
    causes: np.ndarray
    nth: np.ndarray
    system_planned: np.ndarray


class Records:
    """A run's histories, in the order of its simulations: blocks of them
    from numba's loop or from worker processes, and the Python loop's one
    at a time (laid out as a block when compact)."""

    def __init__(self, m: int):
        #: The system's components.
        self.m = m
        self.blocks: List[Block] = []
        self._pending: list = []

    def add(self, history: tuple) -> None:
        """The next simulation's, as ``_replicate`` records it."""
        self._pending.append(history)

    def add_block(self, block: Block) -> None:
        """The next simulations', as a block."""
        self.compact()
        self.blocks.append(block)

    def merge(self, other: "Records") -> None:
        """``other``'s simulations, which come after these."""
        self.compact()
        self.blocks.extend(other.compact().blocks)

    def compact(self) -> "Records":
        """These records with the Python loop's laid out as a block."""
        pending = self._pending
        if pending:
            self._pending = []
            self.blocks.append(_block(pending, self.m))
        return self

    def data(self, t_end: float) -> Tuple[List[_Data], _Data]:
        """Each component's histories, in the order of the components, and
        the system's (its causes the components' places in that order)."""
        self.compact()
        blocks = self.blocks or [_block([], self.m)]
        start = np.concatenate([b.start for b in blocks])
        size = start.shape[0]
        components = []
        for c in range(self.m):
            pieces = [b.bounds[c : c + 2] for b in blocks]  # noqa: E203
            offsets = np.zeros(size + 1, np.int64)
            at, total = 0, 0
            for b, (lo, hi) in zip(blocks, pieces):
                n = b.up.size
                offsets[at + 1 : at + n + 1] = (  # noqa: E203
                    b.offsets[c, 1:] + total
                )
                at += n
                total += hi - lo
            data = _Data(
                np.ascontiguousarray(start[:, c]),
                offsets,
                np.concatenate(
                    [b.times[lo:hi] for b, (lo, hi) in zip(blocks, pieces)]
                ),
                np.zeros(total, np.int64),
                np.zeros(total, np.int64),
                np.concatenate(
                    [b.planned[lo:hi] for b, (lo, hi) in zip(blocks, pieces)]
                ),
                None,
                t_end,
            )
            _, j = _positions(data)
            components.append(replace(data, index=j))
        offsets = np.zeros(size + 1, np.int64)
        np.cumsum(
            np.concatenate([np.diff(b.system_offsets) for b in blocks]),
            out=offsets[1:],
        )
        system = _Data(
            np.concatenate([b.up for b in blocks]),
            offsets,
            np.concatenate([b.system_times for b in blocks]),
            np.concatenate([b.causes for b in blocks]),
            np.concatenate([b.nth for b in blocks]),
            np.concatenate([b.system_planned for b in blocks]),
            None,
            t_end,
        )
        return components, system


def _block(histories: list, m: int) -> Block:
    """The Python loop's records of simulations (see ``_replicate``) as a
    block."""
    n = len(histories)

    def offsets(counts) -> np.ndarray:
        out = np.zeros(n + 1, np.int64)
        np.cumsum(np.asarray(counts, np.int64), out=out[1:])
        return out

    changes = [change for h in histories for change in h[1]]
    system = [change for h in histories for change in h[3]]
    which = np.array([change[0] for change in changes], np.int64)
    row = np.repeat(np.arange(n), [len(h[1]) for h in histories])
    # Each component's changes together, in the order of the simulations
    # and, in each, the order they happened.
    order = np.argsort(which, kind="stable")
    per = np.bincount(which * n + row, minlength=m * n).reshape(m, n)
    grouped = np.zeros((m, n + 1), np.int64)
    np.cumsum(per, axis=1, out=grouped[:, 1:])
    bounds = np.zeros(m + 1, np.int64)
    np.cumsum(grouped[:, n], out=bounds[1:])
    return Block(
        np.array([h[0] for h in histories], np.int8).reshape(n, m),
        bounds,
        grouped,
        np.array([change[1] for change in changes], float)[order],
        np.array([change[2] for change in changes], bool)[order],
        np.array([h[2] for h in histories], np.int8),
        offsets([len(h[3]) for h in histories]),
        np.array([change[0] for change in system], float),
        np.array([change[1] for change in system], np.int64),
        np.array([change[2] for change in system], np.int64),
        np.array([change[3] for change in system], bool),
    )


def simulate(
    rbd,
    t_simulation: float,
    N: int,
    seed,
    working_nodes,
    broken_nodes,
    antithetic: bool,
    engine: str,
    n_jobs,
    start: int,
):
    """``simulate_timelines``: validated, run, and returned as a
    ``TimelineSimulation``."""
    from repyability.rbd.repairable_rbd import _UNSTREAMED
    from repyability.rbd.results import TimelineSimulation

    rbd._require_no_ccf("the simulation", nested=True)
    if (
        isinstance(t_simulation, bool)
        or not isinstance(t_simulation, (int, float, np.integer, np.floating))
        or not (math.isfinite(t_simulation) and t_simulation > 0.0)
    ):
        raise ValueError(
            f"t_simulation must be a positive, finite time, got "
            f"{t_simulation!r}."
        )
    t_simulation = float(t_simulation)
    montecarlo.check_count(N, antithetic, "mc_samples")
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, np.integer))
        or start < 0
    ):
        raise ValueError(
            f"start must be a whole number, 0 or more, got {start!r}."
        )
    start = int(start)
    if antithetic and start % 2:
        raise ValueError(
            f"Antithetic simulations come in pairs: start must be even, got "
            f"{start}."
        )
    if start and seed is None:
        raise ValueError(
            "Simulations from start are part of a seeded run: give its seed."
        )
    if engine not in ("auto", "python", "numba"):
        raise ValueError(
            f"engine must be 'auto', 'python' or 'numba' (the engines that "
            f"record histories), got {engine!r}."
        )
    jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
    working = set() if working_nodes is None else set(working_nodes)
    broken = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working, broken)
    state = np.random.get_state()
    after = state
    try:
        entropy = _streams.entropy_of(seed)
        if seed is None:
            after = np.random.get_state()
        plan, complete = rbd._stream_plan(t_simulation, entropy, antithetic)
        if antithetic and not complete:
            raise NotImplementedError(_UNSTREAMED)
        chosen = _engine(rbd, plan, engine, N)
        if chosen == "python" and independent(rbd, plan):
            method = "streams"
            data, system = _streamed(
                rbd,
                plan,
                (t_simulation, working, broken, entropy, antithetic),
                start,
                N,
                jobs,
            )
        else:
            method = "event loop"
            tally = rbd._run(
                t_simulation,
                working,
                broken,
                "p",
                N,
                False,
                seed,
                antithetic,
                jobs=jobs,
                engine=chosen,
                entropy=entropy,
                first=start,
                histories=True,
            )
            parts, recorded = tally.histories.data(t_simulation)
            data = dict(zip(rbd.components, parts))
            system = _named(rbd, recorded)
    finally:
        np.random.set_state(after)
    return TimelineSimulation(
        system=Timelines._from_data(system),
        components={
            node: Timelines._from_data(d, node) for node, d in data.items()
        },
        time_simulated_to=t_simulation,
        n_simulations=int(N),
        antithetic=bool(antithetic),
        engine=chosen,
        method=method,
        start=start,
    )


def _engine(rbd, plan: _streams.Plan, engine: str, N: int) -> str:
    """The engine that records the run: ``"python"`` or ``"numba"``, as
    ``availability`` would choose between them (another package's engine
    records no histories)."""
    from repyability.rbd import _compiled

    if engine == "python":
        return engine
    if engine == "numba":
        return rbd._simulation_engine("numba", plan, None, N)
    if (
        _compiled.available()
        and _compiled.unsupported(rbd, plan, None, numba=True) is None
        and _compiled.worthwhile(plan, N, "numba")
    ):
        return _compiled.ready("numba", auto=True)
    return "python"


def engine_choice(rbd, plan: _streams.Plan) -> Tuple[str, str]:
    """The engine ``engine="auto"`` records a long run on, and why (see
    ``RepairableRBD.analysis_routes``)."""
    from repyability.rbd import _compiled

    reason = _compiled.unsupported(rbd, plan, None, numba=True)
    if reason is not None:
        return "python", f"the compiled engine does not simulate {reason}"
    if not _compiled.available():
        return (
            "python",
            "numba is not installed; pip install 'repyability[fast]' for "
            "the compiled engine",
        )
    return (
        "numba",
        "compiled for a run long enough to repay loading it; a short one "
        "runs in Python",
    )


def _named(rbd, data: _Data) -> _Data:
    """The system's ``data`` with its causes, the components' places in
    ``rbd.components``, made their places in ``rbd.nodes`` (as
    ``RBD.system_timeline`` gives them)."""
    nodes = list(rbd.nodes)
    place = np.array(
        [nodes.index(node) for node in rbd.components], dtype=np.int64
    )
    return replace(data, causes=place[data.causes], leaves=tuple(nodes))


def independent(rbd, plan: _streams.Plan, prefix: tuple = ()) -> bool:
    """Whether each component's history follows from its own draws alone:
    plain units with streamed lives and repairs, and nested RBDs of them,
    with no crew a job can wait for and nothing scheduled; and a structure
    worked out (one too meshed is followed in the loop)."""
    from repyability.non_repairable import NonRepairable
    from repyability.rbd.repairable_rbd import RepairableRBD

    if (
        rbd._too_meshed() is not None
        or rbd._preventive
        or rbd._inspection
        or rbd._standby
        or rbd._imperfect
        or rbd._maintenance
        or rbd._crews_limited()
    ):
        return False
    for name, component in rbd.components.items():
        path = prefix + (name,)
        if type(component) is RepairableRBD:
            if not independent(component, plan, path):
                return False
        elif not (
            type(component) is NonRepairable
            and (path, _streams.FAILURE) in plan.specs
            and (path, _streams.REPAIR) in plan.specs
        ):
            return False
    return True


def _streamed(
    rbd,
    plan: _streams.Plan,
    run: tuple,
    first: int,
    N: int,
    jobs: Optional[int],
) -> Tuple[Dict[Hashable, _Data], _Data]:
    """Simulations ``first`` to ``first + N - 1`` of the run (its window,
    held nodes, entropy and pairing), drawn from the streams: each
    component's histories and the system's. Those in which changes of
    different components fall at the same time are run in the event loop
    instead. With ``jobs``, the simulations are drawn in parts on that
    many threads."""
    t_end, working, broken = run[0], run[1], run[2]
    threads = 1 if jobs is None else jobs
    width = max(
        (plan.columns(spec) for spec in plan.specs.values()), default=1
    )
    size = -(-_PART // width) * width
    cuts = list(range(0, N, size)) + [N]
    spans = [(first + a, b - a) for a, b in zip(cuts, cuts[1:])]
    if threads > 1 and len(spans) > 1:
        with ThreadPoolExecutor(min(threads, len(spans))) as pool:
            parts = list(
                pool.map(
                    lambda span: _part(
                        rbd, plan, t_end, working, broken, *span
                    ),
                    spans,
                )
            )
    else:
        parts = [
            _part(rbd, plan, t_end, working, broken, *span) for span in spans
        ]
    if len(parts) == 1:
        data, system, tied = parts[0]
    else:
        data = {
            node: _stacked([part[0][node] for part in parts], None, t_end)
            for node in rbd.components
        }
        system = _stacked(
            [part[1] for part in parts], parts[0][1].leaves, t_end
        )
        tied = np.concatenate(
            [part[2] + (span[0] - first) for part, span in zip(parts, spans)]
        )
    if tied.size:
        data, system = _looped(rbd, run, first, tied, data, system)
    return data, system


def _part(
    rbd,
    plan: _streams.Plan,
    t_end: float,
    working: set,
    broken: set,
    first: int,
    N: int,
) -> Tuple[Dict[Hashable, _Data], _Data, np.ndarray]:
    """Simulations ``first`` to ``first + N - 1``, drawn: each component's
    histories, the system's, and the simulations (from 0, here) in which
    changes of different components fall at the same time."""
    data, tied = _drawn(rbd, plan, (), t_end, first, N, working, broken)
    relevant = rbd._decomposition().nodes
    system, more = _system_merge(
        rbd, {n: d for n, d in data.items() if n in relevant}, N, t_end
    )
    return data, system, np.union1d(tied, more)


def _looped(
    rbd,
    run: tuple,
    first: int,
    tied: np.ndarray,
    data: Dict[Hashable, _Data],
    system: _Data,
) -> Tuple[Dict[Hashable, _Data], _Data]:
    """``data`` and ``system`` with simulations ``tied`` (from ``first``)
    run in the event loop, which takes changes at the same time in its own
    order."""
    t_end, working, broken, entropy, antithetic = run
    ctx = rbd._context(
        t_end, working, broken, "p", None, entropy, antithetic, history=True
    )
    records = Records(len(rbd.components))
    try:
        for r in tied.tolist():
            records.add(rbd._replicate(ctx, first + r).history)
    finally:
        rbd._forget_run()
    parts, recorded = records.data(t_end)
    data = {
        node: _spliced(data[node], tied, part)
        for node, part in zip(rbd.components, parts)
    }
    return data, _spliced(system, tied, _named(rbd, recorded))


def _spliced(data: _Data, at: np.ndarray, other: _Data) -> _Data:
    """``data`` with its histories ``at`` (in order) those of ``other``,
    which holds one for each."""
    pieces = []
    before = 0
    for k, s in enumerate(at.tolist()):
        pieces.append((data, before, s))
        pieces.append((other, k, k + 1))
        before = s + 1
    pieces.append((data, before, data.size))

    def joined(field: str) -> np.ndarray:
        return np.concatenate(
            [
                getattr(d, field)[d.offsets[a] : d.offsets[b]]  # noqa: E203
                for d, a, b in pieces
            ]
        )

    start = data.start.copy()
    start[at] = other.start
    counts = np.concatenate(
        [np.diff(d.offsets[a : b + 1]) for d, a, b in pieces]
    )
    offsets = np.zeros(data.size + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    return replace(
        data,
        start=start,
        offsets=offsets,
        times=joined("times"),
        causes=joined("causes"),
        index=joined("index"),
        planned=joined("planned"),
    )


def _drawn(
    rbd,
    plan: _streams.Plan,
    prefix: tuple,
    t_end: float,
    first: int,
    N: int,
    working: set,
    broken: set,
) -> Tuple[Dict[Hashable, _Data], np.ndarray]:
    """Each component's histories in simulations ``first`` to ``first + N
    - 1``, drawn from its streams (a nested RBD's merged from its own
    components', its causes not kept, as the event loop does not keep
    them); and the simulations (from 0) in which changes of different
    components of a nested RBD fall at the same time."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    out: Dict[Hashable, _Data] = {}
    tied = [np.zeros(0, np.int64)]
    for name, component in rbd.components.items():
        path = prefix + (name,)
        if name in broken or name in working:
            start = np.full(N, name in working, np.int8)
            out[name] = _constant(start, t_end, None)
        elif type(component) is RepairableRBD:
            inner, inside = _drawn(
                component, plan, path, t_end, first, N, set(), set()
            )
            relevant = component._decomposition().nodes
            nested, more = _system_merge(
                component,
                {n: d for n, d in inner.items() if n in relevant},
                N,
                t_end,
            )
            out[name] = _leaf(nested, 0)
            tied += [inside, more]
        else:
            out[name] = _unit(plan, path, t_end, first, N)
    return out, np.unique(np.concatenate(tied))


def _unit(
    plan: _streams.Plan, path: tuple, t_end: float, first: int, N: int
) -> _Data:
    """A plain unit's histories in simulations ``first`` to ``first + N -
    1``: up at 0, then its lives and repairs in turn, from its streams,
    added up as the event loop adds them; its changes before ``t_end``."""
    specs = (
        plan.specs[(path, _streams.FAILURE)],
        plan.specs[(path, _streams.REPAIR)],
    )
    pairs = max(specs[0].rows, specs[1].rows, 1)
    size = max(1, min(_BATCH, _BATCH_DRAWS // (2 * pairs)))
    blocks: Dict[tuple, _streams.Block] = {}
    counts = []
    times = []
    for lo in range(first, first + N, size):
        hi = min(lo + size, first + N)
        changes = _changes(plan, specs, blocks, lo, hi, pairs)
        inside = changes < t_end
        count = inside.sum(axis=1)
        flat = changes[inside]
        short = np.flatnonzero(changes[:, -1] < t_end)
        if short.size:
            # These simulations change state again in the window.
            parts = np.split(flat, np.cumsum(count)[:-1])
            for i in short.tolist():
                parts[i] = _longer(
                    plan, specs, blocks, lo + i, t_end, pairs, path
                )
            count = np.array([part.size for part in parts])
            flat = np.concatenate(parts)
        counts.append(count)
        times.append(flat)
        for key in [k for k in blocks if (k[1] + 1) * k[2] <= hi]:
            del blocks[key]  # passed: every simulation of it is done
    count = np.concatenate(counts)
    offsets = np.zeros(N + 1, np.int64)
    np.cumsum(count, out=offsets[1:])
    flat = np.concatenate(times)
    data = _Data(
        np.ones(N, np.int8),
        offsets,
        flat,
        np.zeros(flat.size, np.int64),
        np.zeros(flat.size, np.int64),
        np.zeros(flat.size, bool),
        None,
        t_end,
    )
    _, j = _positions(data)
    return replace(data, index=j)


def _changes(
    plan: _streams.Plan,
    specs: tuple,
    blocks: dict,
    first: int,
    stop: int,
    pairs: int,
) -> np.ndarray:
    """The times of the first ``pairs`` failures and repairs of
    simulations ``first`` to ``stop - 1``, one row each: their lives and
    repairs in turn, added up one after another as the event loop adds
    them (so to the same bits)."""
    steps = np.empty((stop - first, 2 * pairs))
    steps[:, 0::2] = _draws(plan, specs[0], blocks, first, stop, pairs)
    steps[:, 1::2] = _draws(plan, specs[1], blocks, first, stop, pairs)
    return np.cumsum(steps, axis=1)


def _longer(
    plan: _streams.Plan,
    specs: tuple,
    blocks: dict,
    r: int,
    t_end: float,
    pairs: int,
    path: tuple,
) -> np.ndarray:
    """Simulation ``r``'s changes before ``t_end``, for one with more
    than ``pairs`` failures and repairs in the window: twice as many drawn
    each time, up to a limit."""
    limit = min(_MOST * pairs, _streams.MAX_ROWS)
    while pairs < limit:
        pairs = min(2 * pairs, limit)
        changes = _changes(plan, specs, blocks, r, r + 1, pairs)[0]
        if changes[-1] >= t_end:
            return changes[changes < t_end]
    raise ValueError(
        f"Component {path[-1]!r} changes state more than {2 * pairs:,} "
        f"times in a simulation of [0, {t_end:g}]: too many to keep as "
        "timelines. Are its lives and repairs all of length 0?"
    )


def _draws(
    plan: _streams.Plan,
    spec: _streams.Spec,
    blocks: dict,
    first: int,
    stop: int,
    rows: int,
) -> np.ndarray:
    """The first ``rows`` draws of ``spec``'s stream for simulations
    ``first`` to ``stop - 1``, one row each, as the event loop reads them
    (simulation ``r``, row ``r % columns`` of block ``r // columns``)."""
    columns = plan.columns(spec)
    parts = []
    for index in range(first // columns, (stop - 1) // columns + 1):
        key = (spec.name, index, columns)
        block = blocks.get(key)
        if block is None:
            block = blocks[key] = plan.block(spec, index)
        while block.values.shape[1] < rows:
            block.extend()
        lo = max(first, index * columns) - index * columns
        hi = min(stop, (index + 1) * columns) - index * columns
        parts.append(block.values[lo:hi, :rows])
    return np.concatenate(parts)
