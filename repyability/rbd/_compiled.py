"""The compiled engine of ``RepairableRBD`` simulations: numba, an optional
dependency (``pip install "repyability[fast]"``).

It runs the simulations of a system whose components are plain
``NonRepairable`` units -- every model a surpyval parametric one, so that
its draws come from its own streams (see ``_streams``) -- in any structure,
with components held working or broken, antithetic pairs, common random
numbers and costs. Anything else (preventive maintenance, inspections,
nested RBDs, capacities, models whose draws cannot be streamed) runs in
Python, which ``engine="auto"`` chooses by itself.

The two engines give the same results to the last bit: the compiled loop
(``_kernel``) is the Python one over arrays, reading the same draws, and
this module adds its simulations' values to the tally's totals, which keep
them exactly, so that the order they come in does not matter (see
``_Tally``).

Only ``_kernel`` imports numba, so checking what the engine can run costs
nothing when numba is not installed.

Other packages can add compiled engines of their own (see ``engines``):
``engine="auto"`` runs the one of highest priority, numba's being 0, on
what ``unsupported`` allows.
"""

import importlib.util
import sys
import warnings
from typing import Any, Optional

import numpy as np

from repyability.rbd import _streams, engines

# The kinds of term of the structure (see ``modular``).
NODE_TERM, SERIES_TERM, PARALLEL_TERM = 0, 1, 2

#: The draws a run is expected to take (a little more than its events)
#: before ``engine="auto"`` runs it compiled, when the compiled loop is not
#: loaded in the process yet: about half a second of the Python loop, as
#: loading the compiled loop from numba's cache takes about a third of a
#: second (compiling it, the first time ever, some seconds).
AUTO_DRAWS = 1_000_000
#: About the most memory a batch of simulations takes: its draws, and room
#: for its systems' changes of state.
BATCH_BYTES = 64 * 2**20
#: The most simulations in a batch.
MAX_BATCH = 65536
#: The most components for which the compiled loop looks the system's state
#: up in a table of every state (``2**n`` bytes) rather than keeping it up
#: to date as components change (see ``_kept``).
MAX_TABLED = 20


def available() -> bool:
    """Whether numba is installed (without importing it)."""
    return importlib.util.find_spec("numba") is not None


def require() -> None:
    """Raise an ImportError unless numba is installed."""
    if not available():
        raise ImportError(
            "engine='numba' needs numba, an optional dependency: install it "
            "with pip install 'repyability[fast]'."
        )


def unsupported(rbd, plan: _streams.Plan, capacity) -> Optional[str]:
    """What in a run the compiled engine cannot simulate, or None."""
    from repyability.non_repairable import NonRepairable
    from repyability.rbd.repairable_rbd import RepairableRBD

    if capacity is not None:
        return "capacities"
    if any(kind == _streams.START for _, kind in plan.specs):
        return "components started from a state"
    if rbd._crews_limited():
        return "repair crews"
    if rbd._standby:
        return "standby groups"
    if rbd._maintenance:
        return "maintenance groups"
    if rbd._imperfect:
        return "imperfect repair"
    if rbd._preventive:
        return "scheduled preventive maintenance"
    if rbd._inspection:
        return "inspections"
    for name, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            return "nested RBDs"
        if type(component) is not NonRepairable:
            return f"node {name!r}'s {type(component).__name__}"
        for kind in (_streams.FAILURE, _streams.REPAIR):
            if ((name,), kind) not in plan.specs:
                return (
                    f"node {name!r}'s models (their draws cannot be "
                    "streamed)"
                )
    return None


def compiled() -> bool:
    """Whether the compiled loop is ready in this process."""
    kernel = sys.modules.get("repyability.rbd._kernel")
    return kernel is not None and kernel.used()


def preferred() -> Optional[str]:
    """The compiled engine ``engine="auto"`` runs: the usable engine of the
    highest priority among those other packages add (see ``engines``) and
    numba (priority 0), or None."""
    for engine in engines.by_priority(minimum=1):
        return engine.name
    if available():
        return "numba"
    for engine in engines.by_priority():
        return engine.name
    return None


def worthwhile(plan: _streams.Plan, N: int) -> bool:
    """Whether ``engine="auto"`` runs ``N`` simulations of ``plan``
    compiled: when a compiled engine is installed and its loop is ready, or
    the run is long enough to pay for loading it."""
    name = preferred()
    if name is None:
        return False
    draws = N * sum(
        spec.rows
        for spec in plan.specs.values()
        if spec.kind in (_streams.FAILURE, _streams.REPAIR)
    )
    if name != "numba":
        return bool(engines.registered()[name].worthwhile(draws))
    if compiled():
        return True
    return draws >= AUTO_DRAWS


def ready(name: str, auto: bool) -> str:
    """The engine to run, once loaded: ``name`` itself or, when
    ``engine="auto"`` chose an engine another package adds and it cannot
    load, the next one (with a warning). Asked for outright, its error is
    raised."""
    engine = engines.get(name)
    if engine is None:
        return name
    try:
        engines.load(engine)
    except ImportError as error:
        if not auto:
            raise
        fallback = preferred() or "python"
        warnings.warn(
            f"The {name!r} simulation engine could not be loaded, so the "
            f"simulations run on {fallback!r}:\n{error}",
            RuntimeWarning,
            stacklevel=4,
        )
        return ready(fallback, auto)
    return name


def _continued(total, values: np.ndarray):
    """``total`` with each of ``values`` added: exactly, for an
    ``ExactSum``, as the tally keeps its totals (#151); for a float, in
    turn, rounding after each, as ``+=`` in a loop would."""
    from repyability.rbd._exact import ExactSum

    if isinstance(total, ExactSum):
        return total.add(values)
    return float(np.cumsum(np.concatenate(([total], values)))[-1])


def _continued_rows(totals: list, values: np.ndarray) -> list:
    """``_continued`` for each column of ``values`` (one row per
    simulation) and its total in ``totals``."""
    from repyability.rbd._exact import ExactSum, add_columns

    if totals and isinstance(totals[0], ExactSum):
        add_columns(totals, values)
        return totals
    stacked = np.vstack((np.asarray(totals, dtype=float)[None, :], values))
    return np.cumsum(stacked, axis=0)[-1].tolist()


class _System:
    """A run's system as arrays for the compiled loop: the components'
    streams and initial states, the charges, and the structure function."""

    def __init__(self, rbd, plan, working, broken, method: str, kernel):
        from repyability.rbd.repairable_rbd import _CATEGORIES

        nodes = list(rbd.components)
        index = {node: c for c, node in enumerate(nodes)}
        n = len(nodes)
        #: The streams the loop reads, in the order of its stream numbers.
        self.specs: list = []
        numbers: dict = {}

        def stream(name) -> int:
            if name not in numbers:
                numbers[name] = len(self.specs)
                self.specs.append(plan.specs[name])
            return numbers[name]

        working, broken = set(working), set(broken)
        start = np.ones(n, np.int8)
        active = np.zeros(n, np.int8)
        fail = np.zeros(n, np.int64)
        repair = np.zeros(n, np.int64)
        for c, node in enumerate(nodes):
            if node in broken:
                start[c] = 0
            elif node not in working:
                active[c] = 1
                fail[c] = stream(((node,), _streams.FAILURE))
                repair[c] = stream(((node,), _streams.REPAIR))

        # What each failure charges: a node's charge slots, in the order the
        # Python loop pays them.
        self.costed = list(rbd.costs)
        cost_index = np.zeros(n, np.int64)
        slot_start = np.zeros(n, np.int64)
        slot_end = np.zeros(n, np.int64)
        categories: list = []
        streams: list = []
        amounts: list = []
        has_rate = np.zeros(n, np.int8)
        rate = np.zeros(n)
        for c, node in enumerate(nodes):
            node_costs = rbd.costs.get(node)
            slot_start[c] = len(categories)
            if node_costs is not None:
                cost_index[c] = self.costed.index(node)
                for key in rbd.PER_FAILURE_COST_KEYS:
                    if key not in node_costs:
                        continue
                    cost = node_costs[key]
                    categories.append(
                        _CATEGORIES.index(key.removesuffix("_cost"))
                    )
                    if isinstance(cost, float):
                        streams.append(-1)
                        amounts.append(cost)
                    else:
                        kind = _streams.COST_KINDS[key]
                        streams.append(stream(((node,), kind)))
                        amounts.append(0.0)
                if "downtime_cost" in node_costs:
                    has_rate[c] = 1
                    rate[c] = node_costs["downtime_cost"]
            slot_end[c] = len(categories)
        self.has_costs = bool(rbd.has_costs)
        initial_up = rbd.is_system_working(
            {node: bool(start[c]) for c, node in enumerate(nodes)}, method
        )
        self.structure = _structure(rbd, index)
        tabled = n <= MAX_TABLED
        table = (
            kernel.truth_table(n, self.structure)
            if tabled
            else np.zeros(0, np.int8)
        )
        #: The structure laid out to be kept up to date, without a table.
        self.kept = _kept(self.structure, None if tabled else start)
        self.system = (
            start,
            active,
            fail,
            repair,
            slot_start,
            slot_end,
            np.array(categories, np.int64),
            np.array(streams, np.int64),
            np.array(amounts, float),
            cost_index,
            has_rate,
            rate,
            int(self.has_costs),
            float(rbd.downtime_cost_rate) if self.has_costs else 0.0,
            int(bool(initial_up)),
            table,
        )
        self.n = n
        # Room for a simulation's system changes: two for each failure it
        # is expected to have (a failure and the restoration after it).
        self.room = (
            2
            * sum(
                int(plan.specs[((node,), _streams.FAILURE)].rows)
                for c, node in enumerate(nodes)
                if active[c]
            )
            + 2
        )


def _structure(rbd, index: dict) -> tuple:
    """The structure function as arrays: the decomposition's terms (see
    ``modular``), each module's members, and the core's path sets."""
    from repyability.rbd.modular import KOON, NODE

    decomposition = rbd._decomposition()
    terms = decomposition.terms
    child_start, child_end, children = _csr(
        [() if term[0] == NODE else term[1] for term in terms]
    )
    core_start, core_end, core_members = _csr(
        [] if decomposition.root is not None else decomposition.core or []
    )
    return (
        np.array([term[0] for term in terms], np.int64),
        np.array(
            [term[2] if term[0] == KOON else 0 for term in terms], np.int64
        ),
        np.array(
            [index[term[1]] if term[0] == NODE else -1 for term in terms],
            np.int64,
        ),
        child_start,
        child_end,
        children,
        -1 if decomposition.root is None else int(decomposition.root),
        int(bool(decomposition.always_works)),
        core_start,
        core_end,
        core_members,
    )


def _kept(structure: tuple, start: Optional[np.ndarray]) -> tuple:
    """The structure laid out for the compiled loop to keep whether the
    system works up to date as components change, rather than work it out
    at each event: each term's kind, what it needs of its members (all of
    them in series, ``k`` in a vote) and its parent; the terms each
    component stands for (a repeated node, more than one); the core's path
    sets each top term is in; and, with the components as they ``start``,
    each term's state, how many of its members work, how many members of
    each path set are down, and how many path sets work. Without ``start``
    (when the loop looks states up in a table), empty."""
    (
        kind,
        k,
        node,
        child_start,
        child_end,
        children,
        _,
        _,
        core_start,
        core_end,
        core_members,
    ) = structure
    terms = kind.size if start is not None else 0
    n = 0 if start is None else start.size
    paths = core_start.size if start is not None else 0
    parent = np.full(terms, -1, np.int32)
    need = np.zeros(terms, np.int32)
    value = np.zeros(terms, np.int8)
    count = np.zeros(terms, np.int32)
    owners: list = [[] for _ in range(n)]
    for i in range(terms):
        members = children[child_start[i] : child_end[i]]
        parent[members] = i
        if kind[i] == NODE_TERM:
            owners[node[i]].append(i)
            value[i] = start[node[i]]  # type: ignore[index]
            continue
        need[i] = members.size if kind[i] == SERIES_TERM else k[i]
        count[i] = int(np.count_nonzero(value[members]))
        if kind[i] == PARALLEL_TERM:
            value[i] = count[i] > 0
        else:
            value[i] = count[i] >= need[i]
    belongs: list = [[] for _ in range(terms)]
    for p in range(paths):
        for j in range(core_start[p], core_end[p]):
            belongs[core_members[j]].append(p)
    down = np.array(
        [
            np.count_nonzero(
                value[core_members[core_start[p] : core_end[p]]] == 0
            )
            for p in range(paths)
        ],
        np.int32,
    )
    node_term_start, node_term_end, node_terms = _csr(owners)
    term_path_start, term_path_end, term_paths = _csr(belongs)
    return (
        kind[:terms].astype(np.int8),
        need,
        parent,
        node_term_start.astype(np.int32),
        node_term_end.astype(np.int32),
        node_terms.astype(np.int32),
        term_path_start.astype(np.int32),
        term_path_end.astype(np.int32),
        term_paths.astype(np.int32),
        value,
        count,
        down,
        int(np.count_nonzero(down == 0)),
    )


def _csr(groups) -> tuple:
    """Groups of numbers as starts, ends and the numbers one after
    another."""
    sizes = np.array([len(group) for group in groups], np.int64)
    ends = np.cumsum(sizes)
    members = [member for group in groups for member in group]
    return ends - sizes, ends, np.array(members, np.int64)


class _Store:
    """The blocks of a run's streams (see ``_streams.Block``), made as
    batches of simulations reach them and dropped once passed."""

    def __init__(self, plan: _streams.Plan, specs: list, pool=None):
        self._plan = plan
        self._specs = specs
        self.columns = np.array([plan.columns(s) for s in specs], np.int64)
        self._blocks: dict = {}
        # Threads to make blocks on (numpy's maths runs without the GIL):
        # each block depends only on its stream and position, so the
        # order they are made in does not matter.
        self._pool = pool

    def arrays(self, first: int, stop: int) -> tuple:
        """Every stream's draws for simulations ``first`` to ``stop``: the
        blocks' values (to be laid one after another), each block's start
        among them and its rows, each stream's first block, and its
        blocks' columns."""
        if not self._specs:
            return (
                [],
                np.zeros((0, 1), np.int64),
                np.zeros((0, 1), np.int64),
                np.zeros(0, np.int64),
                self.columns,
            )
        first_block = first // self.columns
        last_block = (stop - 1) // self.columns
        width = int(np.max(last_block - first_block)) + 1
        offsets = np.zeros((len(self._specs), width), np.int64)
        rows = np.zeros((len(self._specs), width), np.int64)
        missing = [
            (s, index)
            for s in range(len(self._specs))
            for index in range(int(first_block[s]), int(last_block[s]) + 1)
            if (s, index) not in self._blocks
        ]
        made = (
            map(self._make, missing)
            if self._pool is None or len(missing) < 2
            else self._pool.map(self._make, missing)
        )
        self._blocks.update(zip(missing, made))
        parts, position = [], 0
        for s in range(len(self._specs)):
            for j, index in enumerate(
                range(int(first_block[s]), int(last_block[s]) + 1)
            ):
                block = self._blocks[(s, index)]
                offsets[s, j] = position
                rows[s, j] = block.values.shape[1]
                parts.append(block.values.ravel())
                position += block.values.size
        return (
            parts,
            offsets,
            rows,
            first_block.astype(np.int64),
            self.columns,
        )

    def _make(self, key: tuple) -> _streams.Block:
        s, index = key
        return self._plan.block(self._specs[s], index)

    def extend(self, s: int, index: int) -> None:
        """Another chunk of rows for block ``index`` of stream ``s``."""
        self._blocks[(s, index)].extend()

    def drop(self, stop: int) -> None:
        """Forget the blocks only simulations before ``stop`` draw from."""
        for key in [
            key
            for key in self._blocks
            if (key[1] + 1) * self.columns[key[0]] <= stop
        ]:
            del self._blocks[key]


class Runner:
    """Runs a run's simulations compiled, a batch at a time, and adds them
    to the tally in order (see ``RepairableRBD._run``)."""

    def __init__(
        self, rbd, plan, tally, progress, working, broken, method, jobs
    ):
        from repyability.rbd import _kernel

        self._kernel = _kernel
        self._tally = tally
        self._progress = progress
        self._t = float(tally.t_simulation)
        self._model = _System(rbd, plan, working, broken, method, _kernel)
        self._threads = _kernel.threads(jobs)
        self._pool: Any = None
        # numba's thread count is the calling thread's, and stays set: it is
        # put back once the run is over.
        self._restore: Optional[int] = None
        if self._threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(self._threads)
            self._restore = _kernel.get_threads()
        self._store = _Store(plan, self._model.specs, self._pool)
        rows = sum(spec.rows for spec in self._model.specs)
        size = BATCH_BYTES // (8 * rows + 9 * self._model.room)
        widest = int(np.max(self._store.columns, initial=1))
        if size >= widest:
            size -= size % widest
        self._batch = int(max(1, min(size, MAX_BATCH)))
        # Buffers reused from batch to batch: fresh memory costs page faults.
        self._out: Optional[tuple] = None
        self._flat = np.empty(0)

    def __call__(self, start: int, stop: int) -> None:
        """Simulations ``start`` to ``stop``."""
        for first in range(start, stop, self._batch):
            self._run_batch(first, min(first + self._batch, stop))

    def _outputs(self, size: int) -> tuple:
        """The output arrays of a batch of ``size`` simulations (each
        simulation's row is written whole by the loop)."""
        from repyability.rbd.repairable_rbd import _CATEGORIES

        model = self._model
        out = self._out
        if out is None or out[5].shape[1] != model.room:
            batch, n, room = self._batch, model.n, model.room
            out = self._out = (
                np.empty(batch),
                np.empty((batch, n, 3)),
                np.empty((batch, 4, n), np.int64),
                np.empty((batch, 3), np.int64),
                np.empty(batch, np.int64),
                np.empty((batch, room)),
                np.empty((batch, room), np.int8),
                np.empty(batch),
                np.empty((batch, len(_CATEGORIES))),
                np.empty((batch, max(len(model.costed), 1))),
                np.empty(batch, np.int64),
            )
        return tuple(array[:size] for array in out)

    def _run_batch(self, first: int, stop: int) -> None:
        size = stop - first
        todo = np.arange(first, stop, dtype=np.int64)
        while True:
            out = self._outputs(size)
            if self._simulate(todo, first, stop, out) is None:
                break
            # A simulation outgrew its room for system changes: make more,
            # and run the whole batch again (it is rare).
            self._model.room *= 2
        self._add(out, size)
        self._store.drop(stop)
        self._progress.update(size)

    def _simulate(self, todo, first: int, stop: int, out) -> Optional[str]:
        """Run ``todo`` into ``out``, giving streams more rows until every
        simulation has its draws; ``"room"`` if one ran out of room for its
        changes of state, else None."""
        kernel = self._kernel
        status = out[-1]
        while todo.size:
            draws = self._draws(first, stop)
            if self._threads > 1:
                kernel.set_threads(self._threads)
                kernel.run_parallel(
                    todo,
                    first,
                    self._t,
                    self._model.system,
                    self._model.structure,
                    self._model.kept,
                    draws,
                    out,
                    min(todo.size, 4 * self._threads),
                )
            else:
                kernel.run_serial(
                    todo,
                    first,
                    self._t,
                    self._model.system,
                    self._model.structure,
                    self._model.kept,
                    draws,
                    out,
                )
            codes = status[todo - first]
            if np.any(codes < 0):
                return "room"
            short = todo[codes > 0]
            columns = self._store.columns
            for s, index in {
                (int(code) - 1, int(r) // int(columns[code - 1]))
                for r, code in zip(short, codes[codes > 0])
            }:
                self._store.extend(s, index)
            todo = short
        return None

    def _draws(self, first: int, stop: int) -> tuple:
        """The store's arrays for simulations ``first`` to ``stop``, the
        draws in a buffer kept from batch to batch."""
        parts, offsets, rows, first_block, columns = self._store.arrays(
            first, stop
        )
        total = sum(part.size for part in parts)
        if self._flat.size < total:
            self._flat = np.empty(max(total, int(1.25 * self._flat.size)))
        flat = self._flat[: max(total, 1)]
        if parts:
            np.concatenate(parts, out=flat[:total])
        return flat, offsets, rows, first_block, columns

    def _add(self, out, size: int) -> None:
        """Add a batch's simulations to the tally, in order: as
        ``_Tally.add`` would, one at a time (its totals exactly)."""
        (
            uptime,
            node,
            counts,
            system,
            change_count,
            change_times,
            change_deltas,
            cost,
            by_category,
            by_node,
            _,
        ) = out
        tally = self._tally
        tally.n += size
        tally.uptimes.extend(uptime.tolist())
        failures, restorations, planned = system.sum(axis=0).tolist()
        tally.system_failures += failures
        tally.system_restorations += restorations
        tally.system_planned_outages += planned
        tally.fold_columns(uptime, node[:, :, 0], node[:, :, 1], node[:, :, 2])
        for i, added in enumerate(counts.sum(axis=0).tolist()):
            tally.counts[i] = [a + b for a, b in zip(tally.counts[i], added)]
        kept = np.arange(change_times.shape[1]) < change_count[:, None]
        tally.change_arrays.append(
            (change_times[kept], change_deltas[kept].astype(np.int64))
        )
        if self._model.has_costs:
            tally.cost_samples.extend(cost.tolist())
            costed = len(self._model.costed)
            tally.fold_costs(np.hstack([by_category, by_node[:, :costed]]))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
        if self._restore is not None:
            self._kernel.set_threads(self._restore)
