"""``RepairableRBD.simulate_timelines`` (#157): a run's simulations as
timelines, each component's history and the system's.

The simulations are those ``availability`` runs with the same seed, and
their histories are the event loop's, recorded as it runs on the engine
``engine`` chooses: the Python loop (``RepairableRBD._replicate`` with
``_Context.history``) or numba's (``_kernel._simulate``, recording), in
however many processes or threads. Each records each component's changes
and each of the system's, with the component whose change made it (see
``Records``).
"""

import itertools
import math
from dataclasses import replace
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from repyability.rbd import _ccf_groups, _curves
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _streams
from repyability.timelines import Timelines, _Data


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
                None,
                np.concatenate(
                    [b.planned[lo:hi] for b, (lo, hi) in zip(blocks, pieces)]
                ),
                None,
                t_end,
            )
            components.append(data)
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
    block: each component's changes already together, in the order of the
    simulations and, in each, the order they happened."""
    n = len(histories)
    chain = itertools.chain.from_iterable

    def offsets(counts) -> np.ndarray:
        out = np.zeros(n + 1, np.int64)
        np.cumsum(np.asarray(counts, np.int64), out=out[1:])
        return out

    counts = np.array(
        [[len(h[1][c]) for h in histories] for c in range(m)], np.int64
    ).reshape(m, n)
    grouped = np.zeros((m, n + 1), np.int64)
    np.cumsum(counts, axis=1, out=grouped[:, 1:])
    bounds = np.zeros(m + 1, np.int64)
    np.cumsum(grouped[:, n], out=bounds[1:])
    times = np.fromiter(
        chain(h[1][c] for c in range(m) for h in histories),
        float,
        count=int(bounds[m]),
    )
    # The planned changes, few, by their places among each one's changes.
    planned = np.zeros(times.size, bool)
    for s, h in enumerate(histories):
        for c, places in enumerate(h[2]):
            if places:
                planned[bounds[c] + grouped[c, s] + np.asarray(places)] = True
    system = offsets([len(h[4][0]) for h in histories])

    def system_changes(field: int, dtype) -> np.ndarray:
        lists = (h[4][field] for h in histories)
        return np.fromiter(chain(lists), dtype, count=int(system[n]))

    return Block(
        np.array([h[0] for h in histories], np.int8).reshape(n, m),
        bounds,
        grouped,
        times,
        planned,
        np.array([h[3] for h in histories], np.int8),
        system,
        system_changes(0, float),
        system_changes(1, np.int64),
        system_changes(2, np.int64),
        system_changes(3, bool),
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
    state=None,
):
    """``simulate_timelines``: validated, run, and returned as a
    ``TimelineSimulation``."""
    from repyability.rbd.repairable_rbd import _UNSTREAMED
    from repyability.rbd.results import TimelineSimulation

    _ccf_groups._require_groups_simulated(rbd)
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
    # The components' states at 0, as availability takes them (#163).
    states = _curves._simulation_states(rbd, state, working | broken)
    rng = np.random.get_state()
    after = rng
    try:
        entropy = _streams.entropy_of(seed)
        if seed is None:
            after = np.random.get_state()
        plan, complete = rbd._stream_plan(
            t_simulation, entropy, antithetic, states=states
        )
        if antithetic and not complete:
            raise NotImplementedError(_UNSTREAMED)
        chosen = _engine(rbd, plan, engine, N, states)
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
            states=states,
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
        method="event loop",
        start=start,
    )


def with_costs(
    rbd,
    t_simulation: float,
    N: int,
    seed: Optional[int],
    antithetic: bool,
    engine: str,
    n_jobs,
    start: int,
    states: Optional[dict] = None,
    entropy: Optional[int] = None,
    common: bool = False,
):
    """Simulations ``start`` to ``start + N - 1`` of the run ``seed``
    seeds (or of the run of ``entropy``, a shard's), from new or from the
    components' checked ``states``, in the event loop: each component's
    histories (as ``simulate`` has them) and the run's tally, which keeps
    each simulation's cost beside them (for a conditional run's modules,
    see ``RepairableRBD._conditional_run``). With ``common``, every draw
    must come from a stream, for common random numbers with another
    system's run."""
    from repyability.rbd.repairable_rbd import _UNSTREAMED

    if engine not in ("auto", "python", "numba"):
        raise ValueError(
            f"engine must be 'auto', 'python' or 'numba' (the engines that "
            f"record histories) for a conditional run, got {engine!r}."
        )
    jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
    if entropy is None:
        entropy = _streams.entropy_of(seed)
    states = states or {}
    plan, complete = rbd._stream_plan(
        t_simulation, entropy, antithetic, states
    )
    if antithetic and not complete:
        raise NotImplementedError(_UNSTREAMED)
    tally = rbd._run(
        t_simulation,
        set(),
        set(),
        "p",
        N,
        False,
        seed,
        antithetic,
        jobs=jobs,
        engine=_engine(rbd, plan, engine, N, states=states),
        entropy=entropy,
        common=common,
        first=start,
        states=states,
        histories=True,
    )
    parts, _ = tally.histories.data(t_simulation)
    return dict(zip(rbd.components, parts)), tally


def _engine(rbd, plan: _streams.Plan, engine: str, N: int, states=None) -> str:
    """The engine that records the run: ``"python"`` or ``"numba"``, as
    ``availability`` would choose between them for a run from the
    components' ``states`` (another package's engine records no
    histories)."""
    from repyability.rbd import _compiled

    if engine == "python":
        return engine
    if engine == "numba":
        return rbd._simulation_engine("numba", plan, None, N, states=states)
    if (
        _compiled.available()
        and _compiled.unsupported(rbd, plan, None, numba=True, states=states)
        is None
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
