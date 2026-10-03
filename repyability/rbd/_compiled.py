"""The compiled engine of ``RepairableRBD`` simulations: numba, an optional
dependency (``pip install "repyability[fast]"``).

It runs the simulations of a system whose components are plain
``NonRepairable`` units -- every model a surpyval parametric one, so that
its draws come from its own streams (see ``_streams``) -- in any structure,
with components held working or broken, antithetic pairs, common random
numbers and costs, under age and block replacement, with hidden failures
found by periodic tests, with fewer repair crews than components, with
standby groups, with nested RBDs of up to ``MAX_TABLED`` components, and
following the capacity of a system of up to ``MAX_TRACED`` components
(#155). Anything else (replacement on condition, maintenance groups,
imperfect repair, models whose draws cannot be streamed) runs in Python,
which ``engine="auto"`` chooses by itself.

The two engines give the same results to the last bit: the compiled loop
(``_kernel``) is the Python one over arrays, reading the same draws, and
this module adds its simulations' values to the tally's totals, which keep
them exactly, so that the order they come in does not matter (see
``_Tally``).

Only ``_kernel`` imports numba, so checking what the engine can run costs
nothing when numba is not installed.

Other packages can add compiled engines of their own (see ``engines``):
``engine="auto"`` runs the one of highest priority, numba's being 0, on
what ``unsupported`` allows: plain components. Age and block
replacement, inspections, repair crews, standby groups, nested RBDs and
capacities are numba's own loop's (``unsupported(..., numba=True)``), so a
system with them runs on numba, not on an engine of the interface's
version.
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
#: second (compiling it, the first time ever, a minute or two).
AUTO_DRAWS = 1_000_000
#: About the most memory a batch of simulations takes: its draws, and room
#: for its systems' changes of state.
BATCH_BYTES = 64 * 2**20
#: The most components whose capacity numba's own loop follows (the
#: components up after each change, as the bits of an integer).
MAX_TRACED = 63
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


def unsupported(
    rbd,
    plan: _streams.Plan,
    capacity,
    numba: bool = False,
    states: Optional[dict] = None,
) -> Optional[str]:
    """What in a run a compiled engine cannot simulate, or None: an engine
    of the interface's version (see engines) simulates plain
    components; with numba, numba's own loop also simulates age and
    block replacement, inspections of hidden failures, repair crews,
    standby groups, nested RBDs of up to MAX_TABLED components and the
    capacity of a system of up to MAX_TRACED components (#155), from new
    for maintenance, inspections and nested RBDs (a run from the
    components' states shifts their calendars)."""
    if capacity is not None:
        if not numba:
            return "capacities"
        if len(rbd.components) > MAX_TRACED:
            return (
                f"capacities of more than {MAX_TRACED} components (the "
                "states it follows are bits of 64)"
            )
    if any(kind == _streams.START for _, kind in plan.specs):
        return "components started from a state"
    if (
        numba
        and states
        and (rbd._preventive or rbd._inspection or _nested(rbd))
    ):
        return "components started from a state"
    return _unsupported_level(rbd, plan, numba, ())


def _nested(rbd) -> bool:
    """Whether a node of rbd is a nested RBD."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return any(
        isinstance(component, RepairableRBD)
        for component in rbd.components.values()
    )


def _unsupported_level(
    rbd, plan: _streams.Plan, numba: bool, prefix: tuple
) -> Optional[str]:
    """What in rbd, the system or a nested RBD at prefix, a compiled
    engine cannot simulate (see unsupported)."""
    from repyability.non_repairable import NonRepairable
    from repyability.rbd.repairable_rbd import RepairableRBD

    if rbd._too_meshed() is not None:
        # Its structure is the graph itself (modular.GraphStructure).
        return "a structure too meshed to work out"
    if rbd.ccf_groups:
        return "common-cause groups"
    if rbd._crews_limited() and not numba:
        return "repair crews"
    if rbd._standby and not numba:
        return "standby groups"
    if rbd._maintenance:
        return "maintenance groups"
    if rbd._imperfect:
        return "imperfect repair"
    if rbd._preventive:
        if not numba:
            return "scheduled preventive maintenance"
        reason = _unsupported_maintenance(rbd, plan, prefix)
        if reason is not None:
            return reason
    if rbd._inspection:
        if not numba:
            return "inspections"
        reason = _unsupported_inspections(rbd, plan, prefix)
        if reason is not None:
            return reason
    for name, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            if not numba:
                return "nested RBDs"
            if type(component) is not RepairableRBD:
                return f"node {name!r}'s {type(component).__name__}"
            if len(component.components) > MAX_TABLED:
                return (
                    f"node {name!r}'s {len(component.components)} "
                    f"components (a nested RBD of more than {MAX_TABLED})"
                )
            reason = _unsupported_level(
                component, plan, numba, prefix + (name,)
            )
            if reason is not None:
                return reason
            continue
        if type(component) is not NonRepairable:
            return f"node {name!r}'s {type(component).__name__}"
        if not _streamed(prefix + (name,), plan, rbd._standby.get(name)):
            return f"node {name!r}'s models (their draws cannot be streamed)"
    return None


def _unsupported_maintenance(
    rbd, plan: _streams.Plan, prefix: tuple = ()
) -> Optional[str]:
    """What in a run's preventive maintenance numba's loop cannot simulate:
    replacement on condition, and a maintenance time whose draws cannot be
    streamed. (A component whose own draws cannot be is refused for that,
    and every drawn cost is streamed.)"""
    for node, schedule in rbd._preventive.items():
        if schedule.policy == "condition":
            return "replacement on condition"
        path = prefix + (node,)
        if (
            schedule.duration is not None
            and _streamed(path, plan)
            and (path, _streams.DURATION) not in plan.specs
        ):
            return (
                f"node {node!r}'s maintenance time (its draws cannot be "
                "streamed)"
            )
    return None


def _unsupported_inspections(
    rbd, plan: _streams.Plan, prefix: tuple = ()
) -> Optional[str]:
    """What in a run's inspections numba's loop cannot simulate: a test
    time whose draws cannot be streamed."""
    for node, inspection in rbd._inspection.items():
        path = prefix + (node,)
        if (
            inspection.duration is not None
            and _streamed(path, plan)
            and (path, _streams.DURATION) not in plan.specs
        ):
            return f"node {node!r}'s test time (its draws cannot be streamed)"
    return None


def _streamed(path: tuple, plan: _streams.Plan, standby=None) -> bool:
    """Whether the component at path has its lives and repairs
    streamed: a standby group's, its units' (standby, its
    arrangement)."""
    paths: list = (
        [path]
        if standby is None
        else [path + (unit,) for unit in range(standby.units)]
    )
    return all(
        (place, kind) in plan.specs
        for place in paths
        for kind in (_streams.FAILURE, _streams.REPAIR)
    )


def compiled() -> bool:
    """Whether the compiled loop is ready in this process."""
    kernel = sys.modules.get("repyability.rbd._kernel")
    return kernel is not None and kernel.used()


def preferred(numba_only: bool = False) -> Optional[str]:
    """The compiled engine ``engine="auto"`` runs: the usable engine of the
    highest priority among those other packages add (see ``engines``) and
    numba (priority 0), or None. With ``numba_only``, numba if installed
    (for a run only its own loop simulates; see ``unsupported``)."""
    if not numba_only:
        for engine in engines.by_priority(minimum=1):
            return engine.name
    if available():
        return "numba"
    if not numba_only:
        for engine in engines.by_priority():
            return engine.name
    return None


def choice(
    rbd, plan: _streams.Plan, capacity, states: Optional[dict] = None
) -> tuple:
    """The compiled engine ``engine="auto"`` would run a run on (see
    ``preferred``), or None, and what keeps it in Python (None if
    nothing does): an engine of the interface's version where it can
    simulate the run, else numba's own loop where it can."""
    reason = unsupported(rbd, plan, capacity)
    if reason is None:
        return preferred(), None
    reason = unsupported(rbd, plan, capacity, numba=True, states=states)
    if reason is None:
        return preferred(numba_only=True), None
    return None, reason


def worthwhile(
    plan: _streams.Plan, N: int, name: Optional[str] = None
) -> bool:
    """Whether ``engine="auto"`` runs ``N`` simulations of ``plan``
    compiled, on ``name`` (by default the preferred engine): when it is
    installed and its loop is ready, or the run is long enough to pay for
    loading it."""
    name = preferred() if name is None else name
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
        from repyability.rbd.repairable_rbd import _CATEGORIES, RepairableRBD

        nodes = list(rbd.components)
        index = {node: c for c, node in enumerate(nodes)}
        n = len(nodes)
        working, broken = set(working), set(broken)
        #: The streams the loop reads, in the order of its stream numbers.
        self.specs: list = []
        numbers: dict = {}

        def stream(name) -> int:
            if name not in numbers:
                numbers[name] = len(self.specs)
                self.specs.append(plan.specs[name])
            return numbers[name]

        # Every level's components (#155): the system's own first, then
        # each nested RBD's, a level each, in the order they are found
        # (outer before inner); with each component's RBD, name, path (its
        # streams' place) and level, and each nested node's level.
        level_rbds = [rbd]
        level_paths: list = [()]
        everything: list = []
        child_of: dict = {}
        level = 0
        while level < len(level_rbds):
            here, prefix = level_rbds[level], level_paths[level]
            for name, component in here.components.items():
                c = len(everything)
                everything.append((here, name, prefix + (name,), level))
                held = level == 0 and (name in working or name in broken)
                if isinstance(component, RepairableRBD) and not held:
                    child_of[c] = len(level_rbds)
                    level_rbds.append(component)
                    level_paths.append(prefix + (name,))
            level += 1
        n_all = len(everything)
        levels = len(level_rbds)
        level_start = np.zeros(levels, np.int64)
        level_stop = np.zeros(levels, np.int64)
        level_of = np.array([entry[3] for entry in everything], np.int64)
        for c in range(n_all - 1, -1, -1):
            level_start[level_of[c]] = c
        for c in range(n_all):
            level_stop[level_of[c]] = c + 1
        child_level = np.full(n_all, -1, np.int64)
        for c, child in child_of.items():
            child_level[c] = child

        start = np.ones(n_all, np.int8)
        active = np.ones(n_all, np.int8)
        fail = np.zeros(n_all, np.int64)
        repair = np.zeros(n_all, np.int64)
        for c, (here, name, path, _) in enumerate(everything):
            if c < n and name in broken:
                start[c] = 0
                active[c] = 0
            elif c < n and name in working:
                active[c] = 0
            elif c not in child_of and name not in here._standby:
                fail[c] = stream((path, _streams.FAILURE))
                repair[c] = stream((path, _streams.REPAIR))
        #: Each standby group (#155): its node, its units (from the first's
        #: number, in all the groups' units), how many operate, the dormant
        #: rate and the switches' chance and stream (-1 for none drawn);
        #: and each unit's streams of lives and repairs, and its group.
        group_of = np.full(n_all, -1, np.int64)
        groups: list = []
        units: list = []
        for c, (here, name, path, _) in enumerate(everything):
            arrangement = here._standby.get(name)
            if arrangement is None or not active[c]:
                continue
            group_of[c] = len(groups)
            p = float(arrangement.switching_probability)
            groups.append(
                (
                    c,
                    len(units),
                    arrangement.units,
                    arrangement.k,
                    float(arrangement.dormancy_factor),
                    p,
                    (stream((path, _streams.SWITCH)) if 0.0 < p < 1.0 else -1),
                )
            )
            for unit in range(arrangement.units):
                units.append(
                    (
                        stream((path + (unit,), _streams.FAILURE)),
                        stream((path + (unit,), _streams.REPAIR)),
                        len(groups) - 1,
                    )
                )
        group_node, group_first, group_count, group_k = (
            np.array([group[i] for group in groups], np.int64)
            for i in range(4)
        )
        group_dormancy, group_switching = (
            np.array([group[i] for group in groups], float) for i in (4, 5)
        )
        group_switch = np.array([group[6] for group in groups], np.int64)
        unit_fail, unit_repair, unit_group = (
            np.array([unit[i] for unit in units], np.int64) for i in range(3)
        )

        # What each failure charges: a node's charge slots, in the order the
        # Python loop pays them (a nested RBD's own costs are not counted).
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

        def charge_of(c, key):
            """A charge (-2 for none, -1 for a fixed amount, else its
            stream) and its fixed amount: the system's own components'
            only."""
            cost = rbd.costs.get(nodes[c], {}).get(key) if c < n else None
            if cost is None:
                return -2, 0.0
            if isinstance(cost, float):
                return -1, cost
            return stream(((nodes[c],), _streams.COST_KINDS[key])), 0.0

        #: Each component's preventive maintenance (#155), for numba's own
        #: loop: its policy (0 none, 1 age, 2 block), its interval, the
        #: stream of its maintenance times (-1 for maintenance in zero
        #: time), and its preventive charge (the stream of its amounts, -1
        #: for a fixed amount, -2 for none) and fixed amount. An engine of
        #: the interface's version is given no system with any (see
        #: ``unsupported``).
        policy = np.zeros(n_all, np.int8)
        interval = np.zeros(n_all)
        duration = np.full(n_all, -1, np.int64)
        charge = np.full(n_all, -2, np.int64)
        amount = np.zeros(n_all)
        for c, (here, name, path, _) in enumerate(everything):
            schedule = here._preventive.get(name)
            if schedule is None or not active[c]:
                continue
            policy[c] = 1 if schedule.policy == "age" else 2
            interval[c] = float(schedule.interval)
            if schedule.duration is not None:
                duration[c] = stream((path, _streams.DURATION))
            charge[c], amount[c] = charge_of(c, "preventive_cost")
        #: Each component's inspections of its hidden failures (#155):
        #: their interval (0 for none), offset, coverage, whether a test can
        #: miss a failure and the tests from one full test to the next, the
        #: streams of the test times (-1 for tests in zero time) and of the
        #: uniforms that decide whether a test finds a failure, and the
        #: inspection charge (as the preventive one).
        tested = np.zeros(n_all)
        offset = np.zeros(n_all)
        coverage = np.ones(n_all)
        partial = np.zeros(n_all, np.int8)
        per_full = np.ones(n_all, np.int64)
        test_time = np.full(n_all, -1, np.int64)
        test_draw = np.full(n_all, -1, np.int64)
        test_charge = np.full(n_all, -2, np.int64)
        test_amount = np.zeros(n_all)
        for c, (here, name, path, _) in enumerate(everything):
            inspection = here._inspection.get(name)
            if inspection is None or not active[c]:
                continue
            tested[c] = float(inspection.interval)
            offset[c] = float(inspection.offset)
            coverage[c] = float(inspection.coverage)
            partial[c] = int(inspection.partial)
            per_full[c] = inspection.per_full_test
            if inspection.duration is not None:
                test_time[c] = stream((path, _streams.DURATION))
            if inspection.partial:
                test_draw[c] = stream((path, _streams.TEST))
            test_charge[c], test_amount[c] = charge_of(c, "inspection_cost")
        #: Each level's repair crews when there are fewer than its
        #: components (-1 otherwise: no job waits), and each component's
        #: (and standby unit's) rank in their queue (its priority, negated:
        #: the lowest rank first).
        level_crews = np.array(
            [
                here.repair_crews if here._crews_limited() else -1
                for here in level_rbds
            ],
            np.int64,
        )
        rank = np.array(
            [-here._priority.get(name, 0.0) for here, name, _, _ in everything]
            + [
                -everything[group_node[g]][0]._priority.get(
                    everything[group_node[g]][1], 0.0
                )
                for g in unit_group
            ]
        )
        #: Each nested RBD's table of every state of its components (the
        #: system's own is ``table`` below), and the nested RBDs innermost
        #: first, the order they start in.
        tables = [np.zeros(0, np.int8)]
        for level in range(1, levels):
            here = level_rbds[level]
            local = {name: c for c, name in enumerate(here.components)}
            tables.append(
                kernel.truth_table(len(local), _structure(here, local))
            )
        level_table_start = np.cumsum([0] + [t.size for t in tables[:-1]])
        level_order = np.arange(levels - 1, 0, -1, dtype=np.int64)
        #: Everything numba's own loop simulates besides plain components
        #: (see ``_kernel._simulate``).
        self.upkeep = (
            policy,
            interval,
            duration,
            charge,
            amount,
            tested,
            offset,
            coverage,
            partial,
            per_full,
            test_time,
            test_draw,
            test_charge,
            test_amount,
            level_crews,
            rank,
            group_of,
            group_node,
            group_first,
            group_count,
            group_k,
            group_dormancy,
            group_switching,
            group_switch,
            unit_fail,
            unit_repair,
            unit_group,
            child_level,
            level_of,
            level_start,
            level_stop,
            level_table_start.astype(np.int64),
            np.concatenate(tables).astype(np.int8),
            level_order,
        )
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
        self.kept = _kept(self.structure, None if tabled else start[:n])
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
        # is expected to have (a failure and the restoration after it), and
        # for each maintenance or test that takes time.
        self.room = (
            2
            * sum(
                (
                    int(plan.specs[(path, _streams.FAILURE)].rows)
                    if group_of[c] < 0
                    else sum(
                        int(
                            plan.specs[(path + (unit,), _streams.FAILURE)].rows
                        )
                        for unit in range(here._standby[name].units)
                    )
                )
                + (
                    int(plan.specs[(path, _streams.DURATION)].rows)
                    if duration[c] >= 0 or test_time[c] >= 0
                    else 0
                )
                for c, (here, name, path, _) in enumerate(everything)
                if active[c] and c not in child_of
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
    to the tally in order (see ``RepairableRBD._run``). With ``capacity``
    (the run's ``_CapacityRecorder``; numba's own loop only), it follows
    the system's capacity too (see ``_capacity_into``). For a tally that
    keeps histories (numba's own loop only), the loop records each
    simulation's, and the tally keeps them in place of the totals (see
    ``_timeline_runs.Records``)."""

    def __init__(
        self,
        rbd,
        plan,
        tally,
        progress,
        working,
        broken,
        method,
        jobs,
        capacity=None,
    ):
        from repyability.rbd import _kernel

        self._kernel = _kernel
        self._capacity = capacity
        self._nodes = list(rbd.components)
        #: The capacity states met so far, by the components up in them as
        #: bits: their indices into ``_table`` (see ``_kernel.
        #: trace_capacity``), and the table's columns; and the capacity
        #: levels met, by their indices in the table.
        self._states: dict = {}
        self._columns: list = [[], [], [], [], [], [], []]
        self._table: Optional[tuple] = None
        self._levels: dict = {}
        self._tally = tally
        self._progress = progress
        self._t = float(tally.t_simulation)
        self._model = _System(rbd, plan, working, broken, method, _kernel)
        #: Room for each simulation's components' changes, recording them
        #: (see ``_kernel._simulate``): at first, as many as the system's
        #: room expects (two for each failure), and twice as many each time
        #: a simulation runs out.
        self._recording = tally.histories is not None
        self._records = self._model.room if self._recording else 0
        self._threads = _kernel.threads(jobs)
        # The threads that draw the numbers and run the simulations (the
        # compiled loop releases the GIL).
        self._pool: Any = None
        if self._threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(self._threads)
        self._store = _Store(plan, self._model.specs, self._pool)
        rows = sum(spec.rows for spec in self._model.specs)
        # The draws, the system's changes and, following the capacity, each
        # component's change; recording, each component's change (its
        # time, which and whether planned), the cause of each of the
        # system's, and the components' states at 0.
        room = (9 + (16 if capacity is not None else 0)) * self._model.room
        if self._recording:
            room += 13 * (self._records + self._model.room) + self._model.n
        size = BATCH_BYTES // (8 * rows + room)
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
        records = self._records
        if (
            out is None
            or out[5].shape[1] != model.room
            or out[17].shape[1] != records
        ):
            batch, n, room = self._batch, model.n, model.room
            traced = room if self._capacity is not None else 0
            caused = room if self._recording else 0
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
                np.empty(batch, np.int64),
                np.empty(batch, np.int64),
                np.empty((batch, traced)),
                np.empty((batch, traced), np.int64),
                np.empty((batch, n if self._recording else 0), np.int8),
                np.empty(batch, np.int64),
                np.empty((batch, records)),
                np.empty((batch, records), np.int32),
                np.empty((batch, records), np.int8),
                np.empty((batch, caused), np.int32),
                np.empty((batch, caused), np.int64),
                np.empty((batch, caused), np.int8),
            )
        return tuple(array[:size] for array in out)

    def _run_batch(self, first: int, stop: int) -> None:
        size = stop - first
        todo = np.arange(first, stop, dtype=np.int64)
        while True:
            out = self._outputs(size)
            short = self._simulate(todo, first, stop, out)
            if not short:
                break
            # A simulation outgrew its room for system changes (or, recording,
            # for its components'): make more, and run the whole batch again
            # (it is rare).
            if "room" in short:
                self._model.room *= 2
            if "records" in short:
                self._records *= 2
        self._add(out, size)
        self._store.drop(stop)
        self._progress.update(size)

    def _simulate(self, todo, first: int, stop: int, out) -> set:
        """Run ``todo`` into ``out``, giving streams more rows until every
        simulation has its draws: what a simulation ran out of room for,
        ``"room"`` for the system's changes of state and ``"records"`` for
        the components' (recording), or nothing."""
        kernel = self._kernel
        status = out[10]
        while todo.size:
            draws = self._draws(first, stop)
            if self._threads > 1:
                kernel.run_parallel(
                    self._pool,
                    todo,
                    first,
                    self._t,
                    self._model.system,
                    self._model.structure,
                    self._model.kept,
                    draws,
                    out,
                    min(todo.size, 4 * self._threads),
                    self._model.upkeep,
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
                    self._model.upkeep,
                )
            codes = status[todo - first]
            if np.any(codes < 0):
                short = set()
                if np.any(codes == -1):
                    short.add("room")
                if np.any(codes == -2):
                    short.add("records")
                return short
            short = todo[codes > 0]
            columns = self._store.columns
            for s, index in {
                (int(code) - 1, int(r) // int(columns[code - 1]))
                for r, code in zip(short, codes[codes > 0])
            }:
                self._store.extend(s, index)
            todo = short
        return set()

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
            cap_start,
            cap_count,
            cap_times,
            cap_masks,
            rec_start,
            rec_count,
            rec_times,
            rec_which,
            rec_planned,
            change_cause,
            change_index,
            change_planned,
        ) = out
        tally = self._tally
        tally.n += size
        if self._recording:
            from repyability.rbd._timeline_runs import Block

            bounds, offsets, times, planned = self._kernel.group_records(
                rec_count, rec_times, rec_which, rec_planned, self._model.n
            )
            changed = np.zeros(size + 1, np.int64)
            np.cumsum(change_count, out=changed[1:])
            made = np.arange(change_times.shape[1]) < change_count[:, None]
            tally.histories.add_block(
                Block(
                    rec_start.copy(),
                    bounds,
                    offsets,
                    times,
                    planned,
                    np.full(size, self._model.system[14], np.int8),
                    changed,
                    change_times[made],
                    change_cause[made].astype(np.int64),
                    change_index[made],
                    change_planned[made].astype(bool),
                )
            )
            return
        tally.uptimes.extend(uptime.tolist())
        failures, restorations, planned = system.sum(axis=0).tolist()
        tally.system_failures += failures
        tally.system_restorations += restorations
        tally.system_planned_outages += planned
        tally.fold_columns(uptime, node[:, :, 0], node[:, :, 1], node[:, :, 2])
        for i, added in enumerate(counts.sum(axis=0).tolist()):
            tally.counts[i] = [a + b for a, b in zip(tally.counts[i], added)]
        kept = np.arange(change_times.shape[1]) < change_count[:, None]
        tally.add_changes(
            change_times[kept], change_deltas[kept].astype(np.int64)
        )
        if self._model.has_costs:
            tally.cost_samples.extend(cost.tolist())
            costed = len(self._model.costed)
            tally.fold_costs(np.hstack([by_category, by_node[:, :costed]]))
        if self._capacity is not None:
            self._capacity_into(
                tally, cap_start, cap_count, cap_times, cap_masks
            )

    def _states_of(self, masks: np.ndarray) -> np.ndarray:
        """The indices of the capacity states with the components up that
        ``masks``' bits say (``_CapacityRecorder``'s, each worked out once,
        those not met yet together)."""
        known = self._states
        new = [mask for mask in masks.tolist() if mask not in known]
        if new:
            bits = np.arange(len(self._nodes))
            up = (np.array(new, dtype=np.int64)[:, None] >> bits) & 1
            columns, levels = self._columns, self._levels
            for mask, state in zip(
                new, self._capacity.evaluate(self._nodes, up)
            ):
                known[mask] = len(columns[0])
                columns[0].append(state.finite_mean)
                columns[1].append(state.unlimited)
                columns[2].append(state.delivered)
                columns[3].append(len(columns[5]))
                columns[4].append(len(state.levels))
                columns[5].extend(
                    levels.setdefault(level, len(levels))
                    for level in state.levels
                )
                columns[6].extend(state.probabilities)
            self._table = None
        return np.array([known[mask] for mask in masks.tolist()], np.int64)

    def _capacity_into(self, tally, starts, counts, times, masks) -> None:
        """Follow the capacity of a batch's simulations from what the loop
        recorded (see ``_kernel.trace_capacity``), and add it to the tally
        as ``_Tally.add`` adds each simulation's: its exact totals the
        same in any order."""
        from repyability.rbd._exact import ExactSum
        from repyability.rbd.repairable_rbd import _add_at

        kept = np.arange(times.shape[1]) < counts[:, None]
        visited = np.unique(np.concatenate([starts, masks[kept]]))
        indices = self._states_of(visited)
        if self._table is None:
            columns = self._columns
            self._table = (
                np.array(columns[0], float),
                np.array(columns[1], np.int64),
                np.array(columns[2], float),
                np.array(columns[3], np.int64),
                np.array(columns[4], np.int64),
                np.array(columns[5], np.int64),
                np.array(columns[6], float),
            )
        (
            change_times,
            change_steps,
            free_times,
            free_steps,
            spent,
            offsets,
            share,
        ) = self._kernel.trace_capacity(
            starts,
            counts,
            times,
            masks,
            visited,
            indices,
            self._t,
            self._table,
        )
        tally.add_capacity(
            (change_times, change_steps, free_times, free_steps)
        )
        levels = list(self._levels)
        for k in np.flatnonzero(np.diff(offsets)).tolist():
            a, b = offsets[k], offsets[k + 1]
            _add_at(
                tally.capacity_time,
                levels[k],
                float(spent[a]) if b - a == 1 else ExactSum(spent[a:b]),
            )
        if self._capacity.demand is not None:
            tally.delivered.extend(share.tolist())

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
