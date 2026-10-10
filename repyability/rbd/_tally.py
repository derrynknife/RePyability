"""A ``RepairableRBD`` run's totals: what its simulations add up to.

``_Tally`` sums the simulations' results (``_Replication``) into a run's
availability, cost, counts and timelines, exactly however the run is split
into chunks or shards; ``_ModuleRun`` does the same for a conditional run
(#189), whose simulations are its modules' alone; the ``_Capacity*``
classes record the capacity a run delivers.
"""

import json
import math
import warnings
from collections import Counter
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd import (
    _conditional,
    _timeline_runs,
)
from repyability.rbd._exact import (
    ExactSum,
    add_columns,
    binned_parts,
    compact_parts,
)
from repyability.rbd._point_availability import (
    curve_breaks,
)
from repyability.rbd.results import (
    AvailabilityResult,
    ConditionalRun,
    ControlVariate,
    CostResult,
)

if TYPE_CHECKING:
    from repyability.rbd.repairable_rbd import RepairableRBD
from repyability.rbd._time_order import (
    _add_at,
    _by_time,
    _exact_total,
    _group_partials,
    _partials,
    _zeroed,
)
from repyability.utils.checks import (
    simulation_window,
)
from repyability.utils.wrappers import outside_level


def _check_shard_map(shard_map, shard_size, n_jobs) -> None:
    """Check ``availability``'s ``shard_map`` and ``shard_size``."""
    if shard_map is None:
        if shard_size is not None:
            raise ValueError(
                "shard_size is the size of the shards shard_map runs: give "
                "shard_map too."
            )
        return
    if not callable(shard_map):
        raise ValueError(
            "shard_map must be a map: called as shard_map(run_shard, "
            f"shards), like map; got {shard_map!r}."
        )
    if n_jobs is not None:
        raise ValueError(
            "shard_map runs the shards wherever it sends them: give its "
            "workers the CPUs, and leave out n_jobs."
        )


def _module_shards(
    shard_map, template: dict, step: int, first: int, count: int, modules: int
):
    """A conditional run's simulations ``first`` to ``first + count - 1``
    of its modules (#189) as shards, cut at the multiples of ``step``,
    which ``shard_map(run_shard, shards)`` runs wherever it sends them:
    the ``modules`` histories joined in order, each simulation's own cost,
    the costs by category, and by module (None for a module not costed)."""
    from repyability.rbd.shards import run_shard

    stop = first + count
    cuts = [first, *range((first // step + 1) * step, stop, step), stop]
    ranges = list(zip(cuts, cuts[1:]))
    parts = [
        _conditional.from_partial(saved, modules)
        for saved in shard_map(
            run_shard, [_shard_bytes(template, a, b) for a, b in ranges]
        )
    ]
    parts.sort(key=lambda part: part[0])
    if [part[0] for part in parts] != ranges:
        raise ValueError(
            "shard_map gave back other partials than its shards': it must "
            "return run_shard's result for each shard it is given."
        )
    categories: Dict[str, float] = {}
    for part in parts:
        for key, value in part[3].items():
            categories[key] = categories.get(key, 0.0) + value
    by_module = np.sum([part[4] for part in parts], axis=0)
    costed = np.any([part[5] for part in parts], axis=0)
    return (
        _conditional.joined([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
        categories,
        [float(v) if c else None for v, c in zip(by_module, costed)],
    )


#: A window's expected cost split by category and by component (see
#: ``CostResult.by_category`` and ``by_component``).
_Breakdown = Tuple[Dict[str, float], Dict[Hashable, float]]


class _Exacts(NamedTuple):
    """A system's expected values over a window, worked out exactly (see
    ``RepairableRBD._twin_exact``): its mean availability and, if it is
    priced, its expected cost, with that cost's split (see
    ``ExpectedCost``)."""

    availability: float
    cost: Optional[float] = None
    breakdown: Optional[_Breakdown] = None


class _ModuleRun:
    """A system's dependent modules simulated alone, and every other node
    taken exactly given their joint states (#189, see ``_conditional``):
    the machinery of a conditional run (``RepairableRBD._conditional_run``)
    and of a plain run's means taken given its modules
    (``RepairableRBD._conditioned_run``).

    The modules (by default ``_conditional_modules``) are simulated as a
    diagram of their own (``_modules_rbd``), from their states, whose
    streams are named as in the system, so that its simulations are the
    system's modules'. ``prepare`` works out the other nodes' own expected
    values over the window; ``simulate`` runs a range of simulations,
    working out the system given each joint state its modules meet once,
    on a grid; and ``result`` and ``means`` give what they make. The
    components' curves are built once for the whole run (see
    ``_curves``)."""

    def __init__(
        self,
        rbd: "RepairableRBD",
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        entropy: int,
        *,
        antithetic: bool,
        n_jobs: Optional[int],
        engine: str,
        curve_points: Optional[int],
        state=None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        demand: Optional[float] = None,
        capacities: bool = False,
        control_variate: bool = False,
        modules: Optional[list] = None,
    ) -> None:
        T = simulation_window(t_simulation)
        self.rbd = rbd
        self.T = T
        self.working, self.broken, self.method = working, broken, method
        self.antithetic, self.n_jobs = antithetic, n_jobs
        states = rbd._simulation_states(state, working | broken)
        if modules is None:
            modules = rbd._conditional_modules(working, broken)
        self.modules = modules
        self.module_states = {
            node: states[node] for node in modules if node in states
        }
        self.entropy = entropy
        self.engine = engine
        self.sub = rbd._modules_rbd(modules) if modules else None
        self.recorder = _CapacityRecorder(rbd, demand) if capacities else None
        self.capacity_given: Dict[int, _conditional.CapacityGiven] = {}
        self.capacity_parts: List[_conditional.CapacityValues] = []
        self.shards = None
        if self.sub is not None and shard_map is not None:
            plan, _ = self.sub._stream_plan(
                T, entropy, antithetic, self.module_states
            )
            template = {
                **self.sub._shard_template(
                    T,
                    entropy,
                    set(),
                    set(),
                    "p",
                    antithetic,
                    None,
                    self.engine,
                    self.module_states,
                    None,
                ),
                "histories": True,
            }
            self.shards = (
                shard_map,
                template,
                self.sub._shard_size(plan, shard_size),
            )
        # The exact twin's stand-ins for the modules, with common random
        # numbers, and its exact expected values.
        self.twin_sub: Optional["RepairableRBD"] = None
        self.twin_module_states: dict = {}
        self.exacts = _Exacts(0.0)
        if control_variate and self.sub is not None:
            if self.shards is not None:
                raise ValueError(
                    "A run with control_variate simulates the system's twin "
                    "alongside it, here: leave out shard_map."
                )
            twin, _ = rbd._twin()
            self.exacts = rbd._twin_exact(
                twin, T, working, broken, method, state
            )
            self.twin_sub = twin._modules_rbd(modules)
            twin_states = twin._simulation_states(state, working | broken)
            self.twin_module_states = {
                node: twin_states[node]
                for node in modules
                if node in twin_states
            }
        self.twin_uptimes: List[np.ndarray] = []
        self.twin_costs: List[np.ndarray] = []
        self.curve_x = _conditional.steps(curve_points, T)
        self.given: Dict[int, _conditional.Given] = {}
        self.parts: List[Tuple[int, _conditional.Values]] = []
        self.module_up = np.zeros(len(modules))
        self.priced = rbd.has_costs
        self.own_costs: List[np.ndarray] = []
        self.own_categories = dict.fromkeys(_CATEGORIES, 0.0)
        self.own_components: Dict[Hashable, float] = {}
        self.fixed = set(modules)
        # The other nodes' states, as given (the modules are held).
        self.other_states = (
            None
            if state is None
            else {
                node: value
                for node, value in state.items()
                if node not in self.fixed
            }
        )
        # The rest given the modules, every node the crews serve among them.
        self.exact = rbd._crew_free()
        self._memo: dict = {}
        self.n = 0

    @contextmanager
    def _curves(self):
        """Within it, the exact part's curves are shared, as within
        ``RepairableRBD._sharing_curves``, and kept from one round of
        simulations to the next; out of it, the system holds none, so that
        a plain run of it between rounds can go to other processes."""
        exact = self.exact
        if exact._curve_memo is not None:
            yield
            return
        exact._curve_memo = self._memo
        try:
            yield
        finally:
            exact._curve_memo = None

    def prepare(self) -> None:
        """The other nodes' own expected down times and costs over the
        window, which do not depend on the modules, and the grid the
        system given each joint state is worked out on."""
        exact, T = self.exact, self.T
        with self._curves():
            _, _, curves, self.totals, _, _ = exact._window(
                np.array([T]),
                self.working | self.fixed,
                self.broken,
                self.method,
                nodes=True,
                state=self.other_states,
            )
            self.exact_cost = (
                exact.expected_cost(
                    T,
                    self.working | self.fixed,
                    self.broken,
                    self.method,
                    state=self.other_states,
                )
                if self.priced
                else None
            )
        self.others = (
            0.0
            if self.exact_cost is None
            else math.fsum(
                float(np.ravel(value)[0])
                for key, value in self.exact_cost.by_category.items()
                if key != "system_downtime"
            )
        )
        self.x = _conditional.grid(
            T, [curve_breaks(curve, 0.0, T) for curve in curves.values()]
        )

    def _given(self, paths: "_conditional.Paths") -> None:
        """The system given each joint state ``paths`` meet, not worked
        out yet."""
        new = [
            state for state in paths.states.tolist() if state not in self.given
        ]
        if not new:
            return
        with self._curves():
            for state in new:
                self.given[state] = self.exact._conditional_given(
                    self.modules,
                    state,
                    self.working,
                    self.broken,
                    self.method,
                    self.x,
                    self.other_states,
                )
                if self.recorder is not None:
                    self.capacity_given[state] = self.exact._capacity_given(
                        self.modules,
                        state,
                        self.working,
                        self.broken,
                        self.x,
                        self.other_states,
                        self.recorder.demand,
                    )

    def simulate(self, first: int, count: int) -> None:
        """Simulations ``first`` to ``first + count - 1``: the modules'
        histories and own costs, and each simulation's expected values
        given them (and its twin's, if controlled)."""
        T, modules = self.T, self.modules
        histories: List[Any] = []
        simulated = np.zeros(count)
        if self.shards is not None:
            histories, simulated, spent_on, by_module = _module_shards(
                *self.shards, first, count, len(modules)
            )
            for key, spent in spent_on.items():
                self.own_categories[key] += spent
            for node, spent in zip(modules, by_module):
                if spent is not None:
                    self.own_components[node] = (
                        self.own_components.get(node, 0.0) + spent
                    )
        elif self.sub is not None:
            # The modules' histories, and their own costs in the same
            # simulations.
            data, tally = _timeline_runs.with_costs(
                self.sub,
                T,
                count,
                None,
                self.antithetic,
                self.engine,
                self.n_jobs,
                first,
                self.module_states,
                entropy=self.entropy,
                common=self.twin_sub is not None,
            )
            histories = [data[node] for node in modules]
            if self.sub.has_costs:
                tally._fold()
                simulated = np.asarray(tally.cost_samples, dtype=float)
                for key, spent in tally.cost_by_category.items():
                    self.own_categories[key] += float(spent)
                for node, spent in tally.cost_by_component.items():
                    self.own_components[node] = self.own_components.get(
                        node, 0.0
                    ) + float(spent)
        paths = _conditional.paths(histories, count, T)
        self._given(paths)
        self.parts.append(
            (count, _conditional.values(paths, self.given, self.curve_x))
        )
        if self.recorder is not None:
            self.capacity_parts.append(
                _conditional.capacity_values(
                    paths, self.capacity_given, self.curve_x
                )
            )
        if self.twin_sub is not None:
            # The twin's modules in the same simulations: the other nodes
            # are the same, and so is the system given them.
            data, tally = _timeline_runs.with_costs(
                self.twin_sub,
                T,
                count,
                None,
                self.antithetic,
                self.engine,
                self.n_jobs,
                first,
                self.twin_module_states,
                entropy=self.entropy,
                common=self.twin_sub is not None,
            )
            twin_paths = _conditional.paths(
                [data[node] for node in modules], count, T
            )
            self._given(twin_paths)
            twin_uptime = _conditional.values(
                twin_paths, self.given, self.curve_x
            ).uptime
            self.twin_uptimes.append(twin_uptime)
            spent = np.zeros(count)
            if self.twin_sub.has_costs:
                tally._fold()
                spent = np.asarray(tally.cost_samples, dtype=float)
            self.twin_costs.append(
                spent
                + self.others
                + self.rbd.downtime_cost_rate * (T - twin_uptime)
            )
        self.module_up += _conditional.module_totals(histories, count, T)[0]
        self.own_costs.append(simulated)
        self.n = first + count

    def uptimes(self) -> np.ndarray:
        """Each simulation's expected up time given its modules."""
        return np.concatenate([values.uptime for _, values in self.parts])

    def costs(self) -> np.ndarray:
        """Each simulation's expected cost given its modules: their own
        costs, the other nodes' expected costs, and the system's expected
        down time's."""
        return (
            np.concatenate(self.own_costs)
            + self.others
            + self.rbd.downtime_cost_rate * (self.T - self.uptimes())
        )

    def breakdown(self) -> _Breakdown:
        """The expected cost of a window given the modules' histories (the
        mean of ``costs``), by category and by component: the modules' own
        costs in the simulations, the other nodes' expected costs, and the
        system's expected down time's."""
        n, exact_cost = self.n, self.exact_cost
        assert exact_cost is not None
        by_category = {
            key: self.own_categories[key] / n
            + (
                0.0
                if key == "system_downtime"
                else float(np.ravel(exact_cost.by_category.get(key, 0.0))[0])
            )
            for key in _CATEGORIES
        }
        by_category["system_downtime"] = float(
            np.mean(self.rbd.downtime_cost_rate * (self.T - self.uptimes()))
        )
        by_component = {
            node: float(np.ravel(value)[0])
            for node, value in exact_cost.by_component.items()
            if node not in self.fixed
        }
        for node, amount in self.own_components.items():
            by_component[node] = amount / n
        return by_category, by_component

    def controls(
        self,
    ) -> Tuple[Optional[ControlVariate], Optional[ControlVariate]]:
        """The controls of the fractions up and of the costs, if the run is
        controlled (the latter if priced)."""
        if self.twin_sub is None:
            return None, None
        control = ControlVariate.of(
            self.uptimes() / self.T,
            np.concatenate(self.twin_uptimes) / self.T,
            self.exacts[0],
            self.antithetic,
        )
        cost_control = None
        if self.priced and self.exacts[1] is not None:
            cost_control = ControlVariate.of(
                self.costs(),
                np.concatenate(self.twin_costs),
                self.exacts[1],
                self.antithetic,
            )
        return control, cost_control

    def judged(self, target: str) -> np.ndarray:
        """The values a run to a tolerance judges: of the ``target``'s mean
        (see ``_stopping_rule``), controlled if the run is."""
        control, cost_control = self.controls()
        if target == "cost":
            values = self.costs()
            return (
                values
                if cost_control is None
                else cost_control.controlled(values)
            )
        values = self.uptimes() / self.T
        return values if control is None else control.controlled(values)

    def unjudged(self) -> Optional[str]:
        """Why the spread of the simulations' expected values given the
        modules does not show the error (#215): they never changed state,
        so their outages were not sampled, and every simulation's values
        are the same. None once they have, or with no modules, when the
        values are exact."""
        if len(self.given) > 1 or not self.modules:
            return None
        return (
            f"The modules {list(self.modules)} never changed state in the "
            f"{self.n} simulations, so their outages were not sampled: run "
            "more simulations to sample them."
        )

    def _warn_unjudged(self, instead: str) -> None:
        reason = self.unjudged()
        if reason is not None:
            warnings.warn(
                f"{reason} {instead}",
                RuntimeWarning,
                stacklevel=outside_level(),
            )

    def split(self, record: ConditionalRun) -> Optional[_Breakdown]:
        """The expected cost's split a plain run takes with ``record``, its
        means given the modules (see ``means``): ``breakdown``, where the
        cost's mean is taken given them (priced, and the modules changed
        state, see ``ConditionalRun.informative``); else None, and the
        split is the simulations' own."""
        if not self.priced or not record.informative:
            return None
        return self.breakdown()

    def means(self) -> ConditionalRun:
        """The record of a plain run whose means are taken given its
        modules (see ``RepairableRBD._conditioned_run``)."""
        self._warn_unjudged(
            "The means and their intervals are the simulations' own "
            "(method='simulated')."
        )
        return ConditionalRun(
            tuple(self.modules),
            len(self.given),
            whole=True,
            uptimes=self.uptimes(),
            costs=self.costs() if self.priced else None,
        )

    def result(self) -> AvailabilityResult:
        """The conditional run's result (see
        ``RepairableRBD._conditional_run``)."""
        self._warn_unjudged(
            "The mean intervals have no error to give (nan): a plain run "
            "(conditional=False) gives one."
        )
        rbd, T, n = self.rbd, self.T, self.n
        parts, curve_x = self.parts, self.curve_x
        uptimes = self.uptimes()
        failures = np.concatenate([v.failures for _, v in parts])
        planned = np.concatenate([v.planned for _, v in parts])
        restorations = np.concatenate([v.restorations for _, v in parts])
        curve = np.zeros(curve_x.size)
        square = np.zeros(curve_x.size)
        for size, part in parts:
            curve += size * part.curve / n
            square += size * part.curve_square / n
        down = self.totals["downtime"]
        node_downtime: Dict[Hashable, float] = {}
        for node in rbd.components:
            if node in self.working:
                node_downtime[node] = 0.0
            elif node in self.broken:
                node_downtime[node] = n * T
            elif node in self.fixed:
                j = self.modules.index(node)
                node_downtime[node] = n * T - float(self.module_up[j])
            else:
                node_downtime[node] = n * float(np.ravel(down[node])[0])
        node_uptime = {
            node: n * T - value for node, value in node_downtime.items()
        }
        record = ConditionalRun(tuple(self.modules), len(self.given), square)
        control, cost_control = self.controls()
        cost_result = _acquisition_only(rbd, n, T, self.antithetic)
        if self.priced:
            by_category, by_component = self.breakdown()
            cost_result = CostResult(
                samples=self.costs(),
                t_simulation=T,
                n_simulations=n,
                acquisition_cost=rbd.acquisition_cost,
                by_category=by_category,
                by_component=by_component,
                antithetic=self.antithetic,
                conditional=record,
                control_variate=cost_control,
            )
        capacity_fields: dict = {}
        if self.recorder is not None:
            time_at: Dict[float, float] = {}
            for piece in self.capacity_parts:
                for level, spent in piece.time_at.items():
                    time_at[level] = time_at.get(level, 0.0) + spent
            mean = np.sum(
                [piece.curve for piece in self.capacity_parts], axis=0
            )
            unlimited = np.sum(
                [piece.unlimited for piece in self.capacity_parts], axis=0
            )
            demand = self.recorder.demand
            capacity_fields = dict(
                capacity_timeline=curve_x.copy(),
                capacity=np.where(unlimited > 0, np.inf, mean / n),
                capacity_time={
                    level: time_at[level] for level in sorted(time_at)
                },
                demand=demand,
                delivered=(
                    None
                    if demand is None
                    else np.concatenate(
                        [piece.delivered for piece in self.capacity_parts]
                    )
                    / (demand * T)
                ),
            )
        return AvailabilityResult(
            timeline=curve_x,
            availability=curve,
            system_uptime=float(math.fsum(uptimes)),
            time_simulated_to=T,
            criticalities=None,
            node_uptime=node_uptime,
            node_downtime=node_downtime,
            system_downtime=float(n * T - math.fsum(uptimes)),
            system_failures=float(math.fsum(failures)),
            system_restorations=float(math.fsum(restorations)),
            n_simulations=n,
            cost=cost_result,
            system_planned_outages=float(math.fsum(planned)),
            uptimes=uptimes,
            antithetic=self.antithetic,
            conditional=record,
            control_variate=control,
            **capacity_fields,
        )


def _exact_controls(
    tally: "_Tally",
    t_simulation: float,
    exacts: _Exacts,
    antithetic: bool,
) -> Tuple[ControlVariate, Optional[ControlVariate]]:
    """The controls of a run's fractions up and costs by the system itself,
    whose expected values the exact methods work out (#187): ``exacts``,
    its mean availability over the window and its expected cost (None
    unpriced). Its intervals are then those values, with no error (see
    ``ControlVariate.itself``)."""
    exact, exact_cost = exacts.availability, exacts.cost
    fractions = np.asarray(tally.uptimes, dtype=float) / t_simulation
    control = ControlVariate.of(
        fractions, fractions, exact, antithetic, itself=True
    )
    cost_control = None
    if exact_cost is not None:
        costs = np.asarray(tally.cost_samples, dtype=float)
        cost_control = ControlVariate.of(
            costs, costs, exact_cost, antithetic, itself=True
        )
    return control, cost_control


def _chunk_means(control_variate, conditional) -> Dict[str, Optional[bool]]:
    """A chunk's (or a merge's) ``control_variate`` and ``conditional``,
    checked (#236): None, the means ``availability`` takes by default, or
    False, the simulations' own; a chunk holds the system's simulations
    alone, not a twin's or its modules' alone, which True would take."""
    out: Dict[str, Optional[bool]] = {}
    for name, value, alone in (
        ("control_variate", control_variate, "its exact twin's"),
        ("conditional", conditional, "its modules' alone"),
    ):
        if isinstance(value, (bool, np.bool_)) and not value:
            value = False
        elif value is not None:
            raise ValueError(
                f"{name} must be None or False for chunks, got {value!r}: a "
                f"chunk holds the system's simulations, not {alone}, which "
                f"{name}=True simulates. Run the whole run with "
                "availability() for that."
            )
        out[name] = value
    return out


def _acquisition_only(
    rbd: "RepairableRBD", N: int, t_simulation: float, antithetic: bool
) -> Optional[CostResult]:
    """The cost result of ``N`` simulations of ``rbd`` when nothing but its
    ``acquisition_cost`` is priced (#234): no running cost, exactly (its
    mean interval ``"exact"``), with the acquisition beside it; None when
    nothing at all is."""
    if not rbd.acquisition_cost:
        return None
    zeros = np.zeros(N)
    return CostResult(
        samples=zeros,
        t_simulation=t_simulation,
        n_simulations=N,
        acquisition_cost=rbd.acquisition_cost,
        by_category={category: 0.0 for category in _CATEGORIES},
        by_component={},
        antithetic=antithetic,
        control_variate=ControlVariate(
            twin=zeros,
            exact=0.0,
            coefficient=0.0,
            correlation=0.0,
            itself=True,
        ),
    )


def _shard_bytes(template: dict, start: int, stop: int) -> bytes:
    """A shard (see ``RepairableRBD.shards``): the run's ``template`` with
    the range of its simulations, as JSON."""
    return json.dumps({**template, "start": start, "stop": stop}).encode()


#: The cost categories, in the order ``CostResult.by_category`` lists them.
_CATEGORIES = (
    "repair",
    "replace",
    "preventive",
    "inspection",
    "component_downtime",
    "system_downtime",
    "setup",
)


# The category of a repair's own cost (all an imperfect repair is charged).
_REPAIR = _CATEGORIES.index("repair")


class _Replication:
    """One simulation's results, which a :class:`_Tally` adds to its totals
    in the order of the simulations.

    ``node_up``, ``both_up`` and ``both_down`` hold each component's up
    time, and the time it and the system were both up and both down;
    ``counts`` each component's failures, the system failures they caused,
    its restorations and the system restorations they caused; and
    ``changes`` the times the system changed state, with ``deltas`` +1 for
    a restoration and -1 for a failure or planned outage. ``cost`` is the
    simulation's total cost (None when nothing is priced), ``by_category``
    and ``by_node`` its breakdown, and the ``capacity`` fields what the
    capacity trace recorded (see ``_CapacityTrace``). ``replacements``
    counts each component's replacements (its failures, found ones for a
    component with hidden failures, its preventive replacements, and a
    standby group's units' failures): the spares it used.
    """

    __slots__ = (
        "uptime",
        "node_up",
        "both_up",
        "both_down",
        "counts",
        "failures",
        "restorations",
        "planned",
        "changes",
        "deltas",
        "cost",
        "by_category",
        "by_node",
        "capacity_changes",
        "capacity_time",
        "delivered",
        "replacements",
        "opportunistic",
        "history",
    )
    uptime: float
    node_up: List[float]
    both_up: List[float]
    both_down: List[float]
    counts: Tuple[List[int], List[int], List[int], List[int]]
    failures: int
    restorations: int
    planned: int
    changes: List[float]
    deltas: List[int]
    cost: Optional[float]
    by_category: List[float]
    by_node: Dict[Any, float]
    capacity_changes: Optional[list]
    capacity_time: Optional[list]
    delivered: Optional[float]
    replacements: List[int]
    opportunistic: List[int]
    history: Optional[tuple]


class _Tally:
    """The running totals of an availability simulation.

    Every total that is a sum of floats is kept exactly, as an ``ExactSum``
    (see ``_exact``), and rounded once when the result is built (#151): so
    the totals are the same to the last bit however the simulations ran,
    one after another in this process, in blocks in other processes,
    compiled, or in chunks merged later. Per-node totals are lists in the
    order of ``nodes``; the per-simulation values (``uptimes``,
    ``cost_samples``, ``delivered``) are kept in the simulations' order.
    """

    #: Simulations whose per-node values wait to be folded into the
    #: per-node totals (see ``_fold``).
    _ROWS = 256

    def __init__(
        self,
        nodes: list,
        costs: dict,
        t_simulation: float,
        curve_points: Optional[int] = None,
    ):
        n = len(nodes)
        self.nodes = nodes
        self.t_simulation = t_simulation
        self.n = 0
        # The times the simulated systems changed state, and +1 or -1; and
        # the same, as arrays, from the compiled engine.
        self.changes: list = []
        self.deltas: list = []
        self.change_arrays: list = []
        # With curve_points (#153), the changes are counted on a grid of
        # times instead of kept: the net change in how many systems work
        # in each bin up to each grid time (from just after the one
        # before), the grid's first time 0.
        self.curve_points = curve_points
        self.edges: Optional[np.ndarray] = None
        self.binned: Optional[np.ndarray] = None
        if curve_points is not None:
            steps = np.arange(curve_points + 1)
            self.edges = t_simulation * steps / curve_points
            self.binned = np.zeros(curve_points + 1, dtype=np.int64)
        # Per node: its failures, the system failures they caused, its
        # restorations and the system restorations they caused.
        self.counts = [[0] * n for _ in range(4)]
        self.system_restorations = 0
        self.system_failures = 0
        self.system_planned_outages = 0
        self.system_uptime = ExactSum()
        self.system_downtime = ExactSum()
        self.node_uptime = [ExactSum() for _ in range(n)]
        self.node_downtime = [ExactSum() for _ in range(n)]
        self.intersection_uptime = [ExactSum() for _ in range(n)]
        self.intersection_downtime = [ExactSum() for _ in range(n)]
        self.union_uptime = [ExactSum() for _ in range(n)]
        self.union_downtime = [ExactSum() for _ in range(n)]
        # Each simulation's up time, its nodes' up times and times up and
        # down with the system, and its costs, until they are folded into
        # the totals (see _fold).
        self._rows: list = []
        self._cost_rows: list = []
        # One system uptime (and, when priced, one total cost) per
        # replication, in order.
        self.uptimes: List[float] = []
        self.cost_samples: List[float] = []
        self.cost_by_category = {key: ExactSum() for key in _CATEGORIES}
        self.cost_by_component = {node: ExactSum() for node in costs}
        # With capacities (see _CapacityRecorder): the times a simulated
        # system's expected capacity changed and by how much, and the times
        # whether it can carry an unlimited amount did (+1 or -1), as lists
        # from the Python engine and arrays from the compiled one (see
        # capacity_records); the time spent at each capacity (exactly, see
        # _add_at); and each replication's delivered fraction of the
        # demand, in order.
        self.capacity_times: list = []
        self.capacity_steps: list = []
        self.unlimited_times: list = []
        self.unlimited_steps: list = []
        self.capacity_arrays: list = []
        # With curve_points (#190), the changes of the expected capacity are
        # counted on the grid instead, as the changes of state are: in each
        # bin, exactly (arrays whose exact sum, bin by bin, is the bin's
        # total change), and the net change in how many can carry an
        # unlimited amount.
        self.capacity_binned: Optional[list] = None
        self.unlimited_binned: Optional[np.ndarray] = None
        if self.edges is not None:
            self.capacity_binned = []
            self.unlimited_binned = np.zeros(len(self.edges), dtype=np.int64)
        self.capacity_time: dict = {}
        self.delivered: List[float] = []
        # Each replication's replacements of each node, when asked for
        # (see RepairableRBD.spares_demand).
        self.replacements: Optional[List[List[int]]] = None
        # Per node: its early renewals at stops of its maintenance group.
        self.opportunistic = [0] * n
        # The simulations' histories, when asked for (see
        # RepairableRBD.simulate_timelines): kept, in order, in place of
        # every total.
        self.histories: Optional[_timeline_runs.Records] = None

    def add(self, rec: _Replication) -> None:
        """Add the next simulation's results."""
        if self.histories is not None:
            assert rec.history is not None  # recorded (see _replicate)
            self.histories.add(rec.history)
            self.n += 1
            # Each simulation's cost is kept beside its histories (for a
            # conditional run's modules, see _conditional_run).
            if rec.cost is not None:
                self.cost_samples.append(rec.cost)
                self._cost_rows.append(
                    [*rec.by_category, *rec.by_node.values()]
                )
            return
        self.n += 1
        self.uptimes.append(rec.uptime)
        self.system_failures += rec.failures
        self.system_restorations += rec.restorations
        self.system_planned_outages += rec.planned
        self.changes.extend(rec.changes)
        self.deltas.extend(rec.deltas)
        self._rows.append(
            [rec.uptime, *rec.node_up, *rec.both_up, *rec.both_down]
        )
        for totals, counts in zip(self.counts, rec.counts):
            for c, count in enumerate(counts):
                if count:
                    totals[c] += count
        if rec.cost is not None:
            self.cost_samples.append(rec.cost)
            self._cost_rows.append([*rec.by_category, *rec.by_node.values()])
        if len(self._rows) >= self._ROWS:
            self._fold()
        if rec.capacity_changes is not None:
            for time, mean, limitless in rec.capacity_changes:
                if mean:
                    self.capacity_times.append(time)
                    self.capacity_steps.append(mean)
                if limitless:
                    self.unlimited_times.append(time)
                    self.unlimited_steps.append(limitless)
            spent = self.capacity_time
            for level, time in rec.capacity_time or ():
                _add_at(spent, level, time)
            if rec.delivered is not None:
                self.delivered.append(rec.delivered)
        if self.replacements is not None:
            self.replacements.append(rec.replacements)
        for c, count in enumerate(rec.opportunistic):
            if count:
                self.opportunistic[c] += count

    def add_changes(self, times: np.ndarray, deltas: np.ndarray) -> None:
        """Add simulations' changes of state (times, and +1 or -1), as an
        engine gives them: kept, or counted on the grid."""
        self.change_arrays.append((times, deltas))
        if self.binned is not None:
            self._fold_changes()

    def add_capacity(self, arrays: tuple) -> None:
        """Add simulations' changes of capacity, as an engine gives them
        (see ``capacity_records``): kept, or counted on the grid."""
        self.capacity_arrays.append(arrays)
        if self.capacity_binned is not None:
            self._fold_capacity()

    def _grid_bins(self, times: np.ndarray) -> np.ndarray:
        """The grid's bin of each of ``times``: that of the first grid time
        at or after it (``-0.0`` taken as ``0.0``)."""
        assert self.edges is not None
        return np.minimum(
            np.searchsorted(self.edges, times + 0.0, side="left"),
            len(self.edges) - 1,
        )

    def _fold_capacity(self) -> None:
        """Count the changes of capacity kept so far in the grid's bins,
        exactly, and drop them."""
        assert self.capacity_binned is not None
        assert self.unlimited_binned is not None
        size = len(self.unlimited_binned)
        for times, steps, free_times, free_steps in self.capacity_arrays:
            self.capacity_binned.extend(
                binned_parts(self._grid_bins(times), steps, size)
            )
            if len(free_times):
                self.unlimited_binned += np.bincount(
                    self._grid_bins(free_times),
                    weights=free_steps,
                    minlength=size,
                ).astype(np.int64)
        self.capacity_arrays = []
        if len(self.capacity_binned) > 64:
            self.capacity_binned = compact_parts(self.capacity_binned, size)

    def _fold_changes(self) -> None:
        """Count the changes kept so far in the grid's bins (each time in
        the bin of the first grid time at or after it), and drop them."""
        assert self.binned is not None and self.edges is not None
        if not (self.changes or self.change_arrays):
            return
        times, deltas = self.state_changes()
        self.changes, self.deltas, self.change_arrays = [], [], []
        bins = np.minimum(
            np.searchsorted(self.edges, times, side="left"),
            len(self.edges) - 1,
        )
        counted = np.bincount(bins, weights=deltas, minlength=len(self.edges))
        self.binned += counted.astype(np.int64)

    def _fold(self) -> None:
        """Fold the values of the simulations added since the last fold
        into the totals, all at once (see ``fold_columns``), and the
        changes into the grid's bins, with ``curve_points``."""
        if self.binned is not None:
            self._fold_changes()
        if self._rows:
            table = np.asarray(self._rows, dtype=float)
            self._rows = []
            n = len(self.nodes)
            self.fold_columns(
                table[:, 0],
                table[:, 1 : 1 + n],  # noqa: E203
                table[:, 1 + n : 1 + 2 * n],  # noqa: E203
                table[:, 1 + 2 * n :],  # noqa: E203
            )
        if self._cost_rows:
            self.fold_costs(np.asarray(self._cost_rows, dtype=float))
            self._cost_rows = []
        if self.capacity_times or self.unlimited_times:
            self.capacity_arrays.append(
                (
                    np.asarray(self.capacity_times, dtype=float),
                    np.asarray(self.capacity_steps, dtype=float),
                    np.asarray(self.unlimited_times, dtype=float),
                    np.asarray(self.unlimited_steps, dtype=np.int64),
                )
            )
            self.capacity_times, self.capacity_steps = [], []
            self.unlimited_times, self.unlimited_steps = [], []
        if self.capacity_binned is not None and self.capacity_arrays:
            self._fold_capacity()

    def compact(self) -> "_Tally":
        """This tally with its simulations' values folded into the totals
        and its changes of state (and of capacity) in arrays: small to send
        from a worker process to the parent (see ``_simulate_block``)."""
        if self.histories is not None:
            self.histories.compact()
            if self._cost_rows:
                self.fold_costs(np.asarray(self._cost_rows, dtype=float))
                self._cost_rows = []
            return self
        self._fold()
        if self.changes:
            times, deltas = self.state_changes()
            self.changes, self.deltas = [], []
            self.change_arrays = [(times, deltas)]
        return self

    def fold_columns(self, uptime, up, both_up, both_down) -> None:
        """Add simulations' values to the totals, exactly (one row per
        simulation): the system's up time (and so its down time), each
        node's up time (and down time), and its times up and down with the
        system (and so either up or either down)."""
        t_end = self.t_simulation
        uptime = np.asarray(uptime, dtype=float).reshape(-1, 1)
        add_columns(
            [
                self.system_uptime,
                self.system_downtime,
                *self.node_uptime,
                *self.node_downtime,
                *self.intersection_uptime,
                *self.intersection_downtime,
                *self.union_uptime,
                *self.union_downtime,
            ],
            np.hstack(
                [
                    uptime,
                    t_end - uptime,
                    up,
                    t_end - up,
                    both_up,
                    both_down,
                    t_end - both_down,
                    t_end - both_up,
                ]
            ),
        )

    def fold_costs(self, table) -> None:
        """Add simulations' costs to the totals, exactly: one row per
        simulation, its costs by category and then by component (in the
        order of ``cost_by_component``)."""
        add_columns(
            [
                *self.cost_by_category.values(),
                *self.cost_by_component.values(),
            ],
            table,
        )

    def state_changes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Every time a simulated system changed state (``-0.0`` made
        ``0.0``), and +1 for a restoration or -1 for a failure or planned
        outage: new arrays, which the caller may overwrite."""
        times = [np.asarray(self.changes, dtype=float)]
        deltas = [np.asarray(self.deltas, dtype=np.int64)]
        for chunk_times, chunk_deltas in self.change_arrays:
            times.append(chunk_times)
            deltas.append(chunk_deltas)
        return _zeroed(np.concatenate(times)), np.concatenate(deltas)

    def capacity_records(self) -> Tuple[np.ndarray, ...]:
        """Every change of a simulated system's expected capacity, as
        times (``-0.0`` made ``0.0``) and changes, whose exact sum at a
        time is the total change then; and every change of whether it can
        carry an unlimited amount, as times and changes in how many systems
        can (see ``capacity_changes``): new arrays, which the caller may
        overwrite."""
        times = [np.asarray(self.capacity_times, dtype=float)]
        steps = [np.asarray(self.capacity_steps, dtype=float)]
        free_times = [np.asarray(self.unlimited_times, dtype=float)]
        free_steps = [np.asarray(self.unlimited_steps, dtype=np.int64)]
        for arrays in self.capacity_arrays:
            for kept, array in zip(
                (times, steps, free_times, free_steps), arrays
            ):
                kept.append(array)
        return (
            _zeroed(np.concatenate(times)),
            np.concatenate(steps),
            _zeroed(np.concatenate(free_times)),
            np.concatenate(free_steps),
        )

    def capacity_changes(self) -> Tuple[np.ndarray, ...]:
        """The times a simulated system's expected capacity changed, in
        order, the changes then, in the order of their times, and where
        each time's changes start (their exact sum is the total change
        then, see ``_group_totals``); and the times whether it can carry an
        unlimited amount changed, in order, and the net change then in how
        many can."""
        times, steps, free_times, free_steps = self.capacity_records()
        at, steps, starts = _by_time(times, steps)
        # Whole numbers, each time's added exactly (as floats, below 2**53).
        free_at, free_steps, firsts = _by_time(free_times, free_steps)
        counts = (
            np.add.reduceat(free_steps, firsts) if firsts.size else free_steps
        )
        return at, steps, starts, free_at, counts.astype(np.int64)

    # The totals that are lists of floats, one per node, and the counts.
    _NODE_SUMS = (
        "node_uptime",
        "node_downtime",
        "intersection_uptime",
        "intersection_downtime",
        "union_uptime",
        "union_downtime",
    )
    _COUNTS = (
        "system_restorations",
        "system_failures",
        "system_planned_outages",
    )

    def merge(self, other: "_Tally") -> None:
        """Add ``other``'s simulations, which come after this tally's (see
        ``SimulationChunk``). The per-simulation values follow on in order,
        and the totals, kept exactly, are those of one run of them all, to
        the last bit."""
        if self.histories is not None:
            assert other.histories is not None  # blocks of one run
            self.histories.merge(other.histories)
            self.n += other.n
            self.cost_samples.extend(other.cost_samples)
            for rows in (self, other):
                if rows._cost_rows:
                    rows.fold_costs(np.asarray(rows._cost_rows, dtype=float))
                    rows._cost_rows = []
            for key, amount in other.cost_by_category.items():
                self.cost_by_category[key].add(amount)
            for node, amount in other.cost_by_component.items():
                self.cost_by_component[node].add(amount)
            return
        self._fold()
        other._fold()
        self.n += other.n
        self.system_uptime.add(other.system_uptime)
        self.system_downtime.add(other.system_downtime)
        for name in self._COUNTS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in self._NODE_SUMS:
            for mine, theirs in zip(getattr(self, name), getattr(other, name)):
                mine.add(theirs)
        for totals, counts in zip(self.counts, other.counts):
            for c, count in enumerate(counts):
                totals[c] += count
        for c, count in enumerate(other.opportunistic):
            self.opportunistic[c] += count
        if self.binned is not None:
            assert other.binned is not None  # chunks of one run
            self.binned += other.binned
        else:
            times, deltas = other.state_changes()
            self.change_arrays.append((times, deltas))
        self.uptimes.extend(other.uptimes)
        self.cost_samples.extend(other.cost_samples)
        for key, amount in other.cost_by_category.items():
            self.cost_by_category[key].add(amount)
        for node, amount in other.cost_by_component.items():
            self.cost_by_component[node].add(amount)
        if self.capacity_binned is not None:
            # Chunks of one run, on one grid.
            assert other.capacity_binned is not None
            assert self.unlimited_binned is not None
            self.capacity_binned.extend(other.capacity_binned)
            self.unlimited_binned += other.unlimited_binned
        else:
            self.capacity_arrays.append(other.capacity_records())
        for level, time in other.capacity_time.items():
            _add_at(self.capacity_time, level, time)
        self.delivered.extend(other.delivered)
        if self.replacements is not None:
            assert other.replacements is not None  # blocks of one run
            self.replacements.extend(other.replacements)

    def to_dict(self) -> dict:
        """The totals, as JSON data (see ``SimulationChunk.to_dict``): each
        exact total as the floats whose exact sum it is."""
        self._fold()
        times, deltas = self.state_changes()
        at, steps, starts, free_at, counts = self.capacity_changes()
        return {
            "nodes": list(self.nodes),
            "t_simulation": self.t_simulation,
            "n": self.n,
            "changes": times.tolist(),
            "deltas": deltas.tolist(),
            "curve_points": self.curve_points,
            "binned": None if self.binned is None else self.binned.tolist(),
            "counts": [list(counts) for counts in self.counts],
            **{name: getattr(self, name) for name in self._COUNTS},
            "system_uptime": self.system_uptime.partials,
            "system_downtime": self.system_downtime.partials,
            **{
                name: [total.partials for total in getattr(self, name)]
                for name in self._NODE_SUMS
            },
            "uptimes": list(self.uptimes),
            "cost_samples": list(self.cost_samples),
            "cost_by_category": {
                key: amount.partials
                for key, amount in self.cost_by_category.items()
            },
            "cost_by_component": [
                [node, amount.partials]
                for node, amount in self.cost_by_component.items()
            ],
            "capacity_changes": list(
                zip(at.tolist(), _group_partials(steps, starts))
            ),
            "unlimited_changes": list(zip(free_at.tolist(), counts.tolist())),
            "capacity_binned": (
                None
                if self.capacity_binned is None
                or self.unlimited_binned is None
                else [
                    part.tolist()
                    for part in compact_parts(
                        self.capacity_binned, len(self.unlimited_binned)
                    )
                ]
            ),
            "unlimited_binned": (
                None
                if self.unlimited_binned is None
                else self.unlimited_binned.tolist()
            ),
            "capacity_time": sorted(
                (level, _partials(time))
                for level, time in self.capacity_time.items()
            ),
            "delivered": list(self.delivered),
            "opportunistic": list(self.opportunistic),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_Tally":
        """The totals ``to_dict`` gave (node names as JSON gives them
        back)."""
        from repyability.rbd.serialisation import _node_name

        nodes = [_node_name(node) for node in data["nodes"]]
        components = [
            _node_name(node) for node, _ in data["cost_by_component"]
        ]
        tally = cls(
            nodes,
            dict.fromkeys(components),
            data["t_simulation"],
            data.get("curve_points"),
        )
        tally.n = int(data["n"])
        tally.changes = [float(t) for t in data["changes"]]
        tally.deltas = [int(d) for d in data["deltas"]]
        if data.get("binned") is not None:
            tally.binned = np.array(data["binned"], dtype=np.int64)
        tally.counts = [[int(c) for c in counts] for counts in data["counts"]]
        for name in cls._COUNTS:
            setattr(tally, name, int(data[name]))
        tally.system_uptime = _exact_total(data["system_uptime"])
        tally.system_downtime = _exact_total(data["system_downtime"])
        for name in cls._NODE_SUMS:
            setattr(tally, name, [_exact_total(v) for v in data[name]])
        tally.uptimes = [float(v) for v in data["uptimes"]]
        tally.cost_samples = [float(v) for v in data["cost_samples"]]
        tally.cost_by_category.update(
            {
                key: _exact_total(v)
                for key, v in data["cost_by_category"].items()
            }
        )
        tally.cost_by_component = {
            _node_name(node): _exact_total(amount)
            for node, amount in data["cost_by_component"]
        }
        for level, value in data["capacity_time"]:
            tally.capacity_time[float(level)] = _exact_total(value)
        for time, value in data["capacity_changes"]:
            # A total change of 0 still marks its time (as one 0.0).
            for part in _exact_total(value).partials or [0.0]:
                tally.capacity_times.append(float(time))
                tally.capacity_steps.append(part)
        for time, change in data["unlimited_changes"]:
            tally.unlimited_times.append(float(time))
            tally.unlimited_steps.append(int(change))
        if data.get("capacity_binned") is not None:
            tally.capacity_binned = [
                np.array(part, dtype=float) for part in data["capacity_binned"]
            ]
        if data.get("unlimited_binned") is not None:
            tally.unlimited_binned = np.array(
                data["unlimited_binned"], dtype=np.int64
            )
        tally.delivered = [float(v) for v in data["delivered"]]
        tally.opportunistic = [int(c) for c in data["opportunistic"]]
        return tally

    def criticality_counts(self) -> Tuple[dict, dict]:
        """The failure and restoration counts behind the criticality
        indices: node -> Counter, for each node that failed (FCI) or was
        restored (RCI), in the order of the nodes."""
        failed, caused_down, restored, caused_up = self.counts
        FCI = {
            node: Counter(
                component_failures=failed[c], system_failures=caused_down[c]
            )
            for c, node in enumerate(self.nodes)
            if failed[c]
        }
        RCI = {
            node: Counter(
                component_restorations=restored[c],
                system_restorations=caused_up[c],
            )
            for c, node in enumerate(self.nodes)
            if restored[c]
        }
        return FCI, RCI


class _CapacityState(NamedTuple):
    """The system's capacity in one state of the simulation (see
    ``_CapacityRecorder``)."""

    levels: tuple
    probabilities: tuple
    # The expected capacity when it is finite (0 otherwise), and whether it
    # can be unlimited.
    finite_mean: float
    unlimited: int
    # The expected fraction of the demand met, if there is a demand.
    delivered: float


class _CapacityRecorder:
    """The system's capacity in each state of an availability simulation.

    Given which components are down, the capacity has an exact
    distribution: one level, unless some nodes work at several (a node's
    capacity levels, given as a dict), which the simulation, drawing only
    whether each component is up, then weighs by their probabilities. It is
    worked out once per state and kept."""

    def __init__(self, rbd: "RepairableRBD", demand: Optional[float]):
        rbd._require_capacities_given()
        self._rbd = rbd
        self._nodes = list(rbd.nodes)
        self._states: Dict[frozenset, _CapacityState] = {}
        # Whether each node works at one level: a state's capacity is then
        # one level for sure (see evaluate).
        self._single = all(
            not isinstance(levels, dict) or len(levels) == 1
            for levels in rbd.capacity.values()
        )
        if demand is None:
            # Everything up at its highest level: the design capacity.
            top = float(self._distribution(frozenset())[0][-1])
            self.demand: Optional[float] = top if np.isfinite(top) else None
        else:
            demand = float(demand)
            if not (np.isfinite(demand) and demand > 0.0):
                raise ValueError(
                    f"demand must be a positive, finite number, got "
                    f"{demand!r}."
                )
            self.demand = demand

    def _distribution(self, down: frozenset) -> Tuple[np.ndarray, np.ndarray]:
        arrays = {
            node: np.zeros(1) if node in down else np.ones(1)
            for node in self._nodes
        }
        levels, rows = self._rbd._capacity_arrays(arrays, 1)
        return levels, rows[:, 0]

    def __call__(self, down: frozenset) -> _CapacityState:
        state = self._states.get(down)
        if state is None:
            state = self._states[down] = self._state(*self._distribution(down))
        return state

    def _state(
        self, levels: np.ndarray, probabilities: np.ndarray
    ) -> _CapacityState:
        """The state whose capacity has ``levels`` with ``probabilities``
        (those that can be reached)."""
        finite = np.isfinite(levels)
        unlimited = int(np.any(probabilities[~finite] > 0.0))
        delivered = 0.0
        if self.demand is not None:
            delivered = (
                float(np.minimum(levels, self.demand) @ probabilities)
                / self.demand
            )
        return _CapacityState(
            tuple(levels.tolist()),
            tuple(probabilities.tolist()),
            (
                0.0
                if unlimited
                else float(levels[finite] @ probabilities[finite])
            ),
            unlimited,
            delivered,
        )

    def evaluate(self, components: list, up: np.ndarray) -> list:
        """The states with ``components`` up as each row of ``up`` (1 or 0
        for each) says, the others up. When each node works at one level,
        they are worked out together, one column of probabilities each:
        each probability of each is then a product of 0s and 1s, or a sum
        with one 1 at most, so exact, and the same as in the state's own
        distribution (``__call__``), whatever else is worked out with it. A
        state that does not come out as one level for sure is worked out
        on its own all the same."""
        states: list = [None] * len(up)
        if self._single and len(up) > 1:
            ones = np.ones(len(up))
            given = dict(zip(components, np.asarray(up, dtype=float).T))
            levels, rows = self._rbd._capacity_arrays(
                {node: given.get(node, ones) for node in self._nodes},
                len(up),
            )
            top = rows.argmax(axis=0)
            sure = (np.count_nonzero(rows, axis=0) == 1) & (
                rows[top, np.arange(len(up))] == 1.0
            )
            # A state of one level for sure is its level's.
            alone: dict = {}
            for i in np.flatnonzero(sure).tolist():
                level = int(top[i])
                state = alone.get(level)
                if state is None:
                    state = alone[level] = self._state(
                        levels[level : level + 1], np.ones(1)  # noqa: E203
                    )
                states[i] = state
        for i, state in enumerate(states):
            if state is None:
                states[i] = self(
                    frozenset(
                        c for c, works in zip(components, up[i]) if not works
                    )
                )
        return states

    def states(self, downs: list) -> list:
        """The state of each of ``downs`` (the components down), those not
        met yet worked out together (see ``evaluate``)."""
        known = self._states
        new = [down for down in dict.fromkeys(downs) if down not in known]
        if len(new) > 1 and self._single:
            up = [[node not in down for node in self._nodes] for down in new]
            for down, state in zip(
                new, self.evaluate(self._nodes, np.array(up))
            ):
                known[down] = state
        return [self(down) for down in downs]

    def trace(self, status: dict) -> "_CapacityTrace":
        """The capacity of one simulation, which starts with ``status``."""
        return _CapacityTrace(self, status)


class _CapacityTrace:
    """One simulation's capacity over time (for the tally to add up in
    order): each change of the expected capacity, and the time spent at
    each level and the fraction of the demand delivered. The components
    down after each change are recorded as it goes, and the states they
    are in worked out at the end, those met first together (see
    ``_CapacityRecorder.states``)."""

    def __init__(self, recorder: _CapacityRecorder, status):
        self.recorder = recorder
        self.down = {node for node, up in status.items() if not up}
        # The time of each change and the components down after it, from
        # the start.
        self.records: list = [(0.0, frozenset(self.down))]

    def change(self, time: float, node, up: bool) -> None:
        """Component ``node`` went up (or down) at ``time``."""
        if up:
            self.down.discard(node)
        else:
            self.down.add(node)
        self.records.append((time, frozenset(self.down)))

    def finish(self, t_simulation: float, rec: _Replication) -> None:
        """The trace, closed at the end of the window, into ``rec``: (time,
        change of the expected capacity, change of whether it can be
        unlimited), from the start's, and (level, time spent at it), in
        order."""
        records = self.records + [(t_simulation, None)]
        states = self.recorder.states([down for _, down in self.records])
        state = states[0]
        since = delivered = 0.0
        changes = [(0.0, state.finite_mean, state.unlimited)]
        spent = []
        for (time, _), new in zip(records[1:], states[1:] + [None]):
            span = time - since
            if span > 0.0:
                for level, p in zip(state.levels, state.probabilities):
                    spent.append((level, p * span))
                delivered += state.delivered * span
            since = time
            if new is not None:
                mean = new.finite_mean - state.finite_mean
                unlimited = new.unlimited - state.unlimited
                if mean or unlimited:
                    changes.append((time, mean, unlimited))
                state = new
        rec.capacity_changes = changes
        rec.capacity_time = spent
        rec.delivered = (
            None if self.recorder.demand is None else delivered / t_simulation
        )
