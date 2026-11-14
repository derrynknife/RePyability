"""Planned against unplanned downtime, in a simulation.

The system is down *unplanned* at an instant if it would still be down
with every planned task, test and shut in progress finished: if its
structure, with every component down for planned work counted as up, is
down. Planned downtime is the rest of its downtime. A component that has
failed (in a repair, waiting for a crew, or failed unseen until a test
finds it) is down unplanned; one down for preventive maintenance that
takes time, for a test of a working unit that takes time, or held off line
by a shut (whatever its own state underneath, which may be failed) is
down for planned work; a nested RBD is down unplanned when its own
structure is in this sense.

``_Sound`` follows, for one level of a run (the system, or a nested RBD),
which of its components have *not* failed and whether the level is down
unplanned, as the Python loop runs; the compiled loop does the same over
arrays (``_kernel._simulate`` with ``_compiled._System.split``), and the
two agree to the last bit. Only a diagram that can take planned outages
(see ``takes_planned_outages``) is followed: in any other every outage is
unplanned. A nested RBD's own failures come to its parent as it runs ahead
of it (see ``_kernel``): when it goes down, the parent learns whether that
is unplanned, and while it is down, each change of that (its *trail*) is
queued with its time on the parent's side, and taken in time order before
each of the parent's events (``catch_up``).
"""

import dataclasses
import heapq
import math
from functools import partial
from typing import Callable, Dict, Hashable, List, Tuple

import numpy as np

from repyability.rbd._events import Event


def takes_planned_outages(rbd) -> bool:
    """Whether a diagram (or one of its nested RBDs) can be down for
    planned work: preventive maintenance or a test that takes time, or a
    shut that takes time."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if any(s.duration is not None for s in rbd._preventive.values()):
        return True
    if any(i.duration is not None for i in rbd._inspection.values()):
        return True
    if any(shut.duration > 0.0 for shut in rbd._shutdowns.values()):
        return True
    return any(
        takes_planned_outages(component)
        for component in rbd.components.values()
        if isinstance(component, RepairableRBD)
    )


class _Sound:
    """One level of a run (see the module's docstring): ``sound``, each
    component that has not failed; ``down``, whether the level is down
    unplanned; ``side``, a heap of nested RBDs' changes still to take (as
    ``(time, Event)`` pairs, the event's status whether the nested RBD is
    sound, ordered by time alone as the event queue's are); and at the top
    level (``top``) the time down unplanned so far (``unplanned``, and
    ``since`` it last went down), or for a nested RBD the changes of
    ``down`` since its parent last took them (``trail``)."""

    __slots__ = (
        "sound",
        "works",
        "top",
        "down",
        "side",
        "trail",
        "since",
        "unplanned",
    )

    def __init__(
        self,
        sound: Dict[Hashable, bool],
        works: Callable,
        top: bool,
        up: bool,
    ):
        self.sound = sound
        self.works = works
        self.top = top
        self.down = not up and not works(sound)
        self.side: list = []
        self.trail: List[Tuple[float, bool]] = []
        self.since = 0.0
        self.unplanned = 0.0

    def change(self, node, failed: bool, t: float, up: bool) -> None:
        """Component ``node`` has ``failed`` (or not) at ``t``, the level
        ``up`` (as its components show it) after the change."""
        sound = self.sound
        if sound[node] != failed:
            return  # (sound is not failed already)
        sound[node] = not failed
        # Up as shown, it is up with every planned outage over too.
        down = False if up else not self.works(sound)
        if down == self.down:
            return
        self.down = down
        if not self.top:
            self.trail.append((t, down))
        elif down:
            self.since = t
        else:
            self.unplanned += t - self.since

    def take(self, node, source) -> None:
        """Queue the changes of nested RBD ``node`` (``source``, its
        ``_StreamedRBD``) since it last gave them."""
        for t, down in source.take_trail():
            heapq.heappush(self.side, (t, Event(t, node, not down)))

    def catch_up(self, t: float, status: dict, up: bool) -> None:
        """Take the nested RBDs' changes at or before ``t``: each, of a
        nested RBD that is down (``status``), as a change of whether it
        has failed."""
        side = self.side
        while side and side[0][0] <= t:
            change = heapq.heappop(side)[1]
            if not status[change.component]:
                self.change(
                    change.component, not change.status, change.time, up
                )

    def close(self, t_end: float, status: dict, up: bool) -> float:
        """The time the level was down unplanned in ``[0, t_end]``."""
        self.catch_up(math.inf, status, up)
        if self.down:
            self.unplanned += t_end - self.since
        return self.unplanned


# -- exact --------------------------------------------------------------------
#
# With independent blocks, the system is down unplanned at ``t`` with
# probability ``1 - phi(P(block not failed at t))``: its structure over
# each block's probability of not having failed. Where a block's curve
# tells its planned work apart (``failed_at``: preventive maintenance or a
# test of a working unit that takes time; a shut's mask is planned), the
# split follows exactly from the curves the other exact values use.


def require_split(rbd) -> None:
    """Raise (``NotImplementedError``, saying why) if the exact methods
    cannot tell a diagram's planned downtime from its unplanned: with
    planned outages, its limited repair crews' chain or its common-cause
    groups' chains, which do not tell a block in planned work from one
    in a repair (nor do a nested RBD's)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if not takes_planned_outages(rbd):
        return
    tie = None
    if rbd._crews_couple():
        tie = "limited repair crews, whose Markov chain does"
    elif rbd.ccf_groups:
        tie = "common-cause groups, whose Markov chains do"
    if tie is not None:
        raise NotImplementedError(
            "This diagram takes planned outages (maintenance, tests or "
            f"shuts that take time) and has {tie} not tell a block in "
            "planned work from one in a repair: its planned and unplanned "
            "downtime have no exact split here. Estimate them by "
            "simulation, with availability() or maintenance_effect(t, "
            "method='simulate')."
        )
    for component in rbd.components.values():
        if isinstance(component, RepairableRBD):
            require_split(component)


def node_failed_at(rbd, node, curve, x: np.ndarray, states=None):
    """Component ``node``'s probability of having failed (see the module's
    docstring) at each time ``x``, from its ``curve``: in closed form
    where it has one (and ``states`` are given, see
    ``_curves._closed_form_down``), from the curve's own split where it
    has planned work, else all its downtime. A node a shut holds off
    line has failed as its own process says, at its own time."""
    from repyability.rbd import _curves, _shutdowns
    from repyability.rbd._point_availability import failed_at

    if isinstance(curve, _shutdowns.MaskedCurve):
        return node_failed_at(rbd, node, curve.curve, curve.tau(x), states)
    if states is not None:
        down = _curves._closed_form_down(rbd, node, x, states)
        if down is not None:
            return down
    return np.clip(failed_at(curve, x), 0.0, 1.0)


def system_failed_at(
    rbd, curves: dict, x: np.ndarray, working, broken, states=None
) -> np.ndarray:
    """The system's probability of being down unplanned at each time
    ``x``, from its nodes' ``curves`` (the nodes held ``working`` never
    fail, and those held ``broken`` have failed): its unreliability over
    its nodes' probabilities of having failed, worked out in its own
    right (#237)."""
    failures = {
        node: node_failed_at(rbd, node, curve, x, states)
        for node, curve in curves.items()
    }
    probabilities = {node: 1.0 - q for node, q in failures.items()}
    for node in list(rbd.in_or_out) + list(set(working) | set(broken)):
        probabilities[node] = np.ones(len(x))
        failures[node] = np.zeros(len(x))
    probabilities = rbd._probabilities_with_overrides(
        probabilities, working, broken
    )
    failures = rbd._failures_with_overrides(failures, working, broken)
    return np.clip(
        np.asarray(
            rbd._system_unreliability(probabilities, failures), dtype=float
        ),
        0.0,
        1.0,
    )


def window_split(
    rbd, curves: dict, ends: np.ndarray, working, broken, states
) -> Tuple[np.ndarray, Dict[Hashable, np.ndarray]]:
    """The system's expected time down unplanned from 0 to each of
    ``ends``, and each component's expected time failed, from the
    nodes' ``curves`` (see ``_windows._window``); each the integral of
    its probability, on the pieces of ``_quadrature``. Without planned
    outages, all the downtime; refused (``require_split``) where the
    curves do not tell them apart. Returns None for the nodes' when the
    caller has their downtime as it is."""
    from repyability.rbd import _ccf_groups, _curves, _windows

    require_split(rbd)
    horizon = float(ends.max()) if ends.size else 0.0
    rates = list(_curves._closed_form_rates(rbd, states).values())
    rates += _ccf_groups._timing_rates(rbd)
    scale = (
        min(
            [horizon or 1.0]
            + [1.0 / rate for rate in rates if 0.0 < rate < math.inf]
        )
        / 4.0
    )

    def system(x):
        return system_failed_at(rbd, curves, x, working, broken, states)

    totals = _windows._integrated(
        rbd, system, list(curves.values()), ends, horizon, scale
    )
    nodes: Dict[Hashable, np.ndarray] = {}
    for node, curve in curves.items():
        if not _planned_work(rbd, node):
            continue

        def own(x, node=node, curve=curve):
            return node_failed_at(rbd, node, curve, x, states)

        nodes[node] = _windows._integrated(rbd, own, [curve], ends, horizon)
    return np.maximum(totals, 0.0), nodes


def _planned_work(rbd, node) -> bool:
    """Whether a component can be down for planned work: maintenance or
    tests that take time, a shut that holds it off line, or a nested RBD
    that can."""
    from repyability.rbd import _shutdowns
    from repyability.rbd.repairable_rbd import RepairableRBD

    component = rbd.components[node]
    if isinstance(component, RepairableRBD):
        return takes_planned_outages(component)
    schedule = rbd._preventive.get(node)
    inspection = rbd._inspection.get(node)
    return (
        (schedule is not None and schedule.duration is not None)
        or (inspection is not None and inspection.duration is not None)
        or any(
            shut.duration > 0.0 for shut, _ in _shutdowns.holding(rbd, node)
        )
    )


def long_run_split(rbd, working, broken) -> Tuple[float, float]:
    """The system's long-run unavailability, and the part of it that is
    unplanned: over the long run's grid (see ``_long_run._long_run_grid``),
    its structure over its nodes' probabilities of having failed (see
    ``_long_run._failed_at``). Refused as ``mean_unavailability`` is, and
    where the planned outages cannot be told apart (``require_split``)."""
    from repyability.rbd import _long_run

    total = _long_run.mean_unavailability(rbd, working, broken, "p")
    if not takes_planned_outages(rbd):
        return total, total
    require_split(rbd)
    working, broken = rbd._held_nodes(working, broken)
    times, weights = _long_run._long_run_grid(rbd)
    failures = _long_run._failed_at(rbd, times)
    probabilities = {node: 1.0 - q for node, q in failures.items()}
    probabilities = rbd._probabilities_with_overrides(
        probabilities, working, broken
    )
    failures = rbd._failures_with_overrides(failures, working, broken)
    unplanned = float(
        weights @ rbd._system_unreliability(probabilities, failures)
    )
    return total, min(unplanned, total)


# -- what the plan does -------------------------------------------------------

#: The ways ``maintenance_effect`` takes.
METHODS = (None, "exact", "simulate")


def without_planned_work(rbd):
    """The diagram run to failure: built as ``rbd`` was, but with no
    ``"preventive"`` task, shutdown or maintenance group (no member in
    one; and so a nested RBD's own), its tests of hidden failures kept
    (a test bound to a shutdown on its calendar, its time the
    shut's)."""
    from repyability.rbd import _shutdowns
    from repyability.rbd.repairable_rbd import RepairableRBD

    args = dict(rbd._init_args)
    components = {}
    for node, spec in args["components"].items():
        if isinstance(spec, RepairableRBD):
            spec = without_planned_work(spec)
        elif isinstance(spec, dict):
            spec = {
                k: v
                for k, v in spec.items()
                if k not in ("preventive", "group")
            }
            test = spec.get("inspection")
            if isinstance(test, dict) and test.get("shutdown") is not None:
                view, bound = _shutdowns.bound_inspection(
                    node, test, rbd._shutdowns
                )
                if bound["times"] is not None:
                    raise NotImplementedError(
                        f"Component {node!r} is tested at shutdown "
                        f"{bound['shutdown']!r}, at given times, which a "
                        "test without the shutdown cannot keep: give the "
                        "shutdown a period."
                    )
                spec["inspection"] = view
        components[node] = spec
    args.update(components=components, shutdowns=None, maintenance_groups=None)
    return RepairableRBD(**args)


def _twin_states(rbd, state):
    """``state`` as the diagram run to failure takes it: a component in
    its maintenance now is up, as new (the task would have renewed it),
    and no calendar is placed but a test's; a nested RBD's in turn."""
    from repyability.rbd.node_state import NodeState
    from repyability.rbd.repairable_rbd import RepairableRBD

    if not state:
        return state
    out = {}
    for node, start in state.items():
        component = rbd.components.get(node)
        if isinstance(component, RepairableRBD) and isinstance(start, dict):
            out[node] = _twin_states(component, start)
        elif isinstance(start, NodeState):
            if start.maintenance:
                start = NodeState()
            elif node not in rbd._inspection and start.since_scheduled:
                start = dataclasses.replace(start, since_scheduled=None)
            out[node] = start
        else:
            out[node] = start
    return out


def maintenance_effect(
    rbd,
    t,
    *,
    state,
    working_nodes,
    broken_nodes,
    method,
    mc_samples,
    seed,
    engine,
):
    """See ``RepairableRBD.maintenance_effect``."""
    from repyability.utils.checks import is_number, one_of
    from repyability.utils.checks import seed as checked

    one_of("method", method, METHODS)
    checked(seed)
    working, broken = rbd._held_nodes(working_nodes, broken_nodes)
    twin = without_planned_work(rbd) if has_plan(rbd) else rbd
    if t is None:
        if method == "simulate":
            raise ValueError(
                "The long run (t None) is not simulated: give a window t "
                "to simulate it over."
            )
        if state:
            raise ValueError(
                "The long run (t None) does not depend on the components' "
                "states: give a window t to start it from them."
            )
        return _long_run_effect(rbd, twin, working, broken)
    if not is_number(t):
        raise TypeError(f"t must be a number (or None); got {t!r}.")
    t = float(t)
    if not (t > 0.0 and math.isfinite(t)):
        raise ValueError(f"t must be a positive, finite time; got {t!r}.")
    if method != "simulate":
        try:
            return _window_effect(rbd, twin, t, state, working, broken)
        except NotImplementedError:
            if method == "exact":
                raise
    return _simulated_effect(
        rbd, twin, t, state, working, broken, mc_samples, seed, engine
    )


def _costs(rbd, node_down, system_down):
    """The lost production of ``system_down`` and the components' own
    downtime costs over ``node_down``."""
    own = sum(
        (
            float(costs["downtime_cost"]) * float(node_down.get(node, 0.0))
            for node, costs in rbd.costs.items()
            if "downtime_cost" in costs
        ),
        0.0,
    )
    return rbd.downtime_cost_rate * system_down, own


def _effect(rbd, window, method, plan: dict, without: dict, **extra):
    """The ``MaintenanceEffect`` of the values with the plan and without
    it (each a dict of ``downtime``, ``failures``, ``node_failures`` and
    ``cost``; the plan's with its split)."""
    from repyability.rbd.results import MaintenanceEffect

    planned, unplanned = plan["planned"], plan["unplanned"]
    node_planned, node_unplanned = plan.get("nodes", (None, None))
    own_planned = own_unplanned = None
    if node_planned is not None:
        _, own_planned = _costs(rbd, node_planned, 0.0)
        _, own_unplanned = _costs(rbd, node_unplanned, 0.0)
    return MaintenanceEffect(
        window=window,
        method=method,
        system_downtime=plan["downtime"],
        system_planned_downtime=planned,
        system_unplanned_downtime=unplanned,
        lost_production=rbd.downtime_cost_rate * plan["downtime"],
        lost_production_planned=rbd.downtime_cost_rate * planned,
        lost_production_unplanned=rbd.downtime_cost_rate * unplanned,
        component_downtime_cost_planned=own_planned,
        component_downtime_cost_unplanned=own_unplanned,
        node_planned_downtime=node_planned,
        node_unplanned_downtime=node_unplanned,
        system_failures=plan["failures"],
        system_failures_without=without["failures"],
        system_failures_avoided=without["failures"] - plan["failures"],
        node_failures=plan["node_failures"],
        node_failures_without=without["node_failures"],
        node_failures_avoided={
            node: without["node_failures"][node] - value
            for node, value in plan["node_failures"].items()
        },
        system_downtime_without=without["downtime"],
        downtime_saved=without["downtime"] - plan["downtime"],
        cost=plan["cost"],
        cost_without=without["cost"],
        cost_saved=without["cost"] - plan["cost"],
        **extra,
    )


def _window_effect(rbd, twin, t, state, working, broken):
    """``maintenance_effect`` over a window, exactly: from each diagram's
    ``expected_events`` and ``expected_cost``; refused where either is,
    or the split is (see ``require_split``)."""
    require_split(rbd)
    values = []
    for diagram, start in ((rbd, state), (twin, _twin_states(rbd, state))):
        events = diagram.expected_events(
            t, working_nodes=working, broken_nodes=broken, state=start
        )
        cost = (
            float(
                diagram.expected_cost(
                    t, working_nodes=working, broken_nodes=broken, state=start
                ).mean
            )
            if diagram.has_costs
            else 0.0
        )
        out = {
            "downtime": float(events.system_downtime),
            "failures": float(events.system_failures),
            "node_failures": {
                node: float(value)
                for node, value in events.node_failures.items()
            },
            "cost": cost,
        }
        if diagram is rbd:
            assert events.system_unplanned_downtime is not None
            out["planned"] = float(events.system_planned_downtime)
            out["unplanned"] = float(events.system_unplanned_downtime)
            out["nodes"] = tuple(
                {node: float(value) for node, value in table.items()}
                for table in (
                    events.node_planned_downtime,
                    events.node_unplanned_downtime,
                )
            )
        values.append(out)
    return _effect(rbd, t, "exact", values[0], values[1])


def _long_run_effect(rbd, twin, working, broken):
    """``maintenance_effect`` in the long run, exactly: each diagram's
    long-run unavailability, failure frequencies and cost rate, and the
    plan's split (``long_run_split``)."""
    values = []
    for diagram in (rbd, twin):
        total, unplanned = long_run_split(diagram, working, broken)
        actions = diagram.node_action_rates(working, broken)
        out = {
            "downtime": total,
            "failures": float(
                diagram.system_failure_frequency(working, broken)
            ),
            "node_failures": {
                node: float(rates.corrective)
                for node, rates in actions.items()
            },
            "cost": (
                float(diagram.expected_cost_rate(working, broken))
                if diagram.has_costs
                else 0.0
            ),
        }
        if diagram is rbd:
            out["planned"] = max(total - unplanned, 0.0)
            out["unplanned"] = unplanned
            down = {
                node: 1.0 - float(value)
                for node, value in rbd._probabilities_with_overrides(
                    rbd.node_availability(), *rbd._held_nodes(working, broken)
                ).items()
                if node in rbd.components
            }
            failed = _long_run_failed(rbd, working, broken, down)
            out["nodes"] = (
                {
                    node: max(down[node] - failed[node], 0.0)
                    for node in rbd.components
                },
                failed,
            )
        values.append(out)
    return _effect(rbd, None, "exact", values[0], values[1])


def _long_run_failed(rbd, working, broken, down: dict) -> dict:
    """Each component's long-run probability of having failed: its
    downtime without planned outages, else its profile over the long
    run's grid (see ``_long_run._failed_at``)."""
    from repyability.rbd import _long_run

    planned = [n for n in rbd.components if _planned_work(rbd, n)]
    out = {node: down[node] for node in rbd.components}
    if not planned:
        return out
    times, weights = _long_run._long_run_grid(rbd)
    failures = rbd._failures_with_overrides(
        _long_run._failed_at(rbd, times), *rbd._held_nodes(working, broken)
    )
    for node in planned:
        out[node] = min(
            float(weights @ np.asarray(failures[node])), down[node]
        )
    return out


def _simulated_effect(
    rbd, twin, t, state, working, broken, mc_samples, seed, engine
):
    """``maintenance_effect`` over a window, simulated: each diagram run
    for ``mc_samples`` simulations from one entropy (so on the same
    streams, each simulation of the plan paired with one without it), the
    plan's split its own (see ``AvailabilityResult``)."""
    from repyability.rbd import _curves, _event_loop, _streams

    n = 10_000 if mc_samples is None else int(mc_samples)
    for diagram in (rbd, twin):
        _require_simulated(diagram)
    entropy = _streams.entropy_of(seed)
    tallies = []
    for diagram, start in ((rbd, state), (twin, _twin_states(rbd, state))):
        tallies.append(
            _event_loop._run(
                diagram,
                t,
                working,
                broken,
                "p",
                n,
                False,
                None,
                engine=engine,
                entropy=entropy,
                states=_curves._simulation_states(
                    diagram, start, working | broken
                ),
                curve_points=1,
            )
        )
    values: List[dict] = []
    for diagram, tally in zip((rbd, twin), tallies):
        tally._fold()
        values.append(
            {
                "downtime": float(tally.system_downtime) / n,
                "failures": tally.system_failures / n,
                "node_failures": {
                    node: count / n
                    for node, count in zip(tally.nodes, tally.counts[0])
                },
                "cost": (
                    float(np.mean(tally.cost_samples))
                    if diagram.has_costs
                    else 0.0
                ),
            }
        )
    unplanned = float(tallies[0].unplanned_downtime()) / n
    values[0]["unplanned"] = unplanned
    values[0]["planned"] = values[0]["downtime"] - unplanned
    up = [np.asarray(tally.uptimes, dtype=float) for tally in tallies]
    errors = {"downtime_saved": _paired_se(up[0] - up[1])}
    if rbd.has_costs or twin.has_costs:
        samples = [
            (
                np.asarray(tally.cost_samples, dtype=float)
                if diagram.has_costs
                else np.zeros(n)
            )
            for diagram, tally in zip((rbd, twin), tallies)
        ]
        errors["cost_saved"] = _paired_se(samples[1] - samples[0])
    return _effect(
        rbd,
        t,
        "simulated",
        values[0],
        values[1],
        mc_samples=n,
        standard_errors=errors,
    )


def _paired_se(differences: np.ndarray) -> float:
    """The standard error of the mean of paired ``differences``."""
    n = len(differences)
    if n < 2:
        return math.nan
    return float(np.std(differences, ddof=1) / math.sqrt(n))


def has_plan(rbd) -> bool:
    """Whether a diagram has planned work that running to failure leaves
    out: a ``"preventive"`` task, a shutdown or a maintenance group (or a
    nested RBD with one)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return bool(rbd._preventive or rbd._shutdowns or rbd._maintenance) or any(
        has_plan(component)
        for component in rbd.components.values()
        if isinstance(component, RepairableRBD)
    )


def _require_simulated(rbd) -> None:
    """Raise unless a run of the diagram simulates (as ``availability``
    refuses)."""
    from repyability.rbd import _ccf_groups

    _ccf_groups._require_groups_simulated(rbd)


def effect_route(rbd, out: dict):
    """How ``maintenance_effect`` over a window is worked out (see
    ``analysis_routes``), given the diagram's other routes ``out``:
    exactly where both diagrams' ``expected_events`` (and, priced,
    ``expected_cost``) are and the split is (``require_split``); else
    simulated, or refused as the simulation is. The long run's route is
    in its reason."""
    from repyability.rbd import routes as r

    twin_made = r.refusal(partial(without_planned_work, rbd))
    if twin_made:
        return r.refused(twin_made)
    twin = without_planned_work(rbd) if has_plan(rbd) else rbd
    # (Run to failure, the twin has no plan, and so no twin of its own.)
    twin_out = out if twin is rbd or has_plan(twin) else twin.analysis_routes()
    names = ("expected_events",) + (
        ("expected_cost",) if rbd.has_costs else ()
    )
    window = [out[name] for name in names] + [twin_out[name] for name in names]
    split = r.refusal(partial(require_split, rbd))
    exact = not split and all(
        route.route in (r.EXACT, r.NUMERICAL) for route in window
    )
    long_run = out["mean_unavailability"]
    if long_run.route == r.REFUSED:
        later = f"In the long run (t None) it refuses: {long_run.reason}"
    elif split:
        later = f"In the long run (t None) it refuses: {split}"
    else:
        later = (
            "In the long run (t None): each diagram's long-run values and "
            "the plan's split over the long run's grid, "
            f"{twin_out['mean_unavailability'].route}."
        )
        if twin_out["mean_unavailability"].route == r.REFUSED:
            later = (
                "In the long run (t None) it refuses: "
                + twin_out["mean_unavailability"].reason
            )
    if exact:
        how = (
            r.NUMERICAL
            if any(route.route == r.NUMERICAL for route in window)
            else r.EXACT
        )
        return r.AnalysisRoute(
            how,
            "Over a window: the plan's and running to failure's "
            "expected_events (and expected_cost), the split from each "
            "block's probability of having failed over time, integrated "
            f"on its curve's pieces. {later}",
        )
    for diagram in (rbd, twin):
        refusal = r.refusal(partial(_require_simulated, diagram))
        if refusal:
            return r.refused(refusal)
    why = (
        split
        or next(route.reason for route in window if route.route == r.REFUSED)
        if (split or any(route.route == r.REFUSED for route in window))
        else ""
    )
    return r.AnalysisRoute(
        r.SIMULATED,
        "Over a window: the plan and running to failure each simulated "
        "from one seed (the same streams), the split followed path by "
        "path"
        + (f", as the exact methods refuse: {why}" if why else "")
        + f". {later}",
    )
