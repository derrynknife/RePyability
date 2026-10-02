"""``RepairableRBD.simulate_timelines`` (#157): a run's simulations as
timelines, each component's history and the system's.

When the components are independent (plain units with streamed models, and
nested RBDs of them, held working or broken or not), each component's
history is drawn straight from its streams: its lives and repairs, the
draws the event loop reads (see ``_streams``), added up in turn, a whole
batch of simulations at once. The times are the event loop's, to the last
bit, since it adds the same draws in the same order. The system's history
is then merged from the components' up the diagram (see
``repyability.timelines``).

Otherwise (repair crews a job can wait for, standby groups, maintenance,
tests, imperfect repair, maintenance groups, models whose draws cannot be
streamed) the components depend on each other, or on more than their own
draws, and their histories are recorded from the event loop's simulations
as it runs them.

Either way the simulations are those ``availability`` runs with the same
seed. The system's history is the event loop's but for changes of
different components at the same time, which the merge takes in the order
of the components and the event loop in the order of its heap.
"""

import math
from typing import Dict, Hashable

import numpy as np

from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _streams
from repyability.timelines import Timelines, _constant, _Data, _raw

#: The most simulations whose draws are laid out at once.
_BATCH = 4096
#: The most draws laid out at once, for memory: fewer simulations at once
#: for a component that changes state more often.
_BATCH_DRAWS = 2**21
#: How many times its planned draws (see ``_streams.first_rows``) a
#: simulation may take before its component is taken to change state
#: without end.
_MOST = 64


def simulate(
    rbd,
    t_simulation: float,
    N: int,
    seed,
    working_nodes,
    broken_nodes,
    antithetic: bool,
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
        if independent(rbd, plan):
            method = "streams"
            data = _drawn(rbd, plan, (), t_simulation, N, working, broken)
        else:
            method = "event loop"
            data = _recorded(
                rbd, t_simulation, N, working, broken, entropy, antithetic
            )
    finally:
        np.random.set_state(after)
    components = {
        node: Timelines._from_data(d, node) for node, d in data.items()
    }
    return TimelineSimulation(
        system=rbd.system_timeline(components),
        components=components,
        time_simulated_to=t_simulation,
        n_simulations=int(N),
        antithetic=bool(antithetic),
        method=method,
    )


def independent(rbd, plan: _streams.Plan, prefix: tuple = ()) -> bool:
    """Whether each component's history follows from its own draws alone:
    plain units with streamed lives and repairs, and nested RBDs of them,
    with no crew a job can wait for and nothing scheduled."""
    from repyability.non_repairable import NonRepairable
    from repyability.rbd.repairable_rbd import RepairableRBD

    if (
        rbd._preventive
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


def _drawn(
    rbd,
    plan: _streams.Plan,
    prefix: tuple,
    t_end: float,
    N: int,
    working: set,
    broken: set,
) -> Dict[Hashable, _Data]:
    """Each component's histories, drawn from its streams (a nested RBD's,
    merged from its own components')."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    out: Dict[Hashable, _Data] = {}
    for name, component in rbd.components.items():
        path = prefix + (name,)
        if name in broken or name in working:
            start = np.full(N, name in working, np.int8)
            out[name] = _constant(start, t_end, None)
        elif type(component) is RepairableRBD:
            inner = _drawn(component, plan, path, t_end, N, set(), set())
            nested = component.system_timeline(
                {
                    node: Timelines._from_data(d, node)
                    for node, d in inner.items()
                }
            )
            out[name] = nested._data
        else:
            out[name] = _unit(plan, path, t_end, N)
    return out


def _unit(plan: _streams.Plan, path: tuple, t_end: float, N: int) -> _Data:
    """A plain unit's histories: up at 0, then its lives and repairs in
    turn, from its streams, added up as the event loop adds them; its
    changes before ``t_end``."""
    specs = (
        plan.specs[(path, _streams.FAILURE)],
        plan.specs[(path, _streams.REPAIR)],
    )
    pairs = max(specs[0].rows, specs[1].rows, 1)
    size = max(1, min(_BATCH, _BATCH_DRAWS // (2 * pairs)))
    blocks: Dict[tuple, _streams.Block] = {}
    counts = []
    times = []
    for first in range(0, N, size):
        stop = min(first + size, N)
        changes = _changes(plan, specs, blocks, first, stop, pairs)
        inside = changes < t_end
        count = inside.sum(axis=1)
        flat = changes[inside]
        short = np.flatnonzero(changes[:, -1] < t_end)
        if short.size:
            # These simulations change state again in the window.
            parts = np.split(flat, np.cumsum(count)[:-1])
            for i in short:
                parts[i] = _longer(
                    plan, specs, blocks, first + int(i), t_end, pairs, path
                )
            count = np.array([part.size for part in parts])
            flat = np.concatenate(parts)
        counts.append(count)
        times.append(flat)
        for key in [k for k in blocks if (k[1] + 1) * k[2] <= stop]:
            del blocks[key]  # passed: every simulation of it is done
    count = np.concatenate(counts)
    offsets = np.zeros(N + 1, np.int64)
    np.cumsum(count, out=offsets[1:])
    flat = np.concatenate(times)
    _, j = _positions_of(offsets, flat.size)
    return _Data(
        np.ones(N, np.int8),
        offsets,
        flat,
        np.zeros(flat.size, np.int64),
        j,
        np.zeros(flat.size, bool),
        None,
        t_end,
    )


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


def _positions_of(offsets: np.ndarray, size: int):
    """Each change's history and position in it, from ``offsets``."""
    counts = np.diff(offsets)
    history = np.repeat(np.arange(counts.size), counts)
    return history, np.arange(size) - offsets[:-1][history]


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


def _recorded(
    rbd,
    t_end: float,
    N: int,
    working: set,
    broken: set,
    entropy,
    antithetic: bool,
) -> Dict[Hashable, _Data]:
    """Each component's histories, recorded from the event loop's
    simulations (see ``RepairableRBD._replicate``)."""
    ctx = rbd._context(
        t_end, working, broken, "p", None, entropy, antithetic
    )._replace(history=True)
    nodes = list(rbd.components)
    starts: list = [[] for _ in nodes]
    changes: list = [[] for _ in nodes]
    planned: list = [[] for _ in nodes]
    for replication in range(N):
        at_start, record = rbd._replicate(ctx, replication).history
        times: list = [[] for _ in nodes]
        flags: list = [[] for _ in nodes]
        for c, t, flag in record:
            times[c].append(t)
            flags[c].append(flag)
        for c in range(len(nodes)):
            starts[c].append(at_start[c])
            changes[c].append(times[c])
            planned[c].append(flags[c])
    return {
        node: _raw(starts[c], changes[c], planned[c], t_end)
        for c, node in enumerate(nodes)
    }
