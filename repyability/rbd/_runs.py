"""A ``RepairableRBD``'s simulated runs: ``availability`` and ``cost``,
their result, the engine that runs them, the variance reductions (the
exact twin as a control variate, conditional runs over the dependent
modules), a run in chunks or shards and its merge
(``simulate_chunk``, ``shards``, ``availability_from_chunks``), the
timelines (``simulate_timelines``) and ``compare``. The event loop
itself (``_run``, ``_replicate``) stays with the diagram. The methods of
``RepairableRBD`` of those names call these.
"""

import dataclasses
import hashlib
import json
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    List,
    Optional,
    Tuple,
)

import numpy as np

from repyability._version import __version__
from repyability.rbd import (
    _conditional,
)
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _streams, _timeline_runs
from repyability.rbd._exact import (
    bin_totals,
)
from repyability.rbd._model_utils import (
    SAVE_ERRORS,
)
from repyability.rbd.node_state import NodeState
from repyability.rbd.results import (
    AvailabilityResult,
    ConditionalRun,
    ConfidenceInterval,
    ControlVariate,
    CostResult,
    Criticalities,
    FailureCriticalityIndex,
    RestorationCriticalityIndex,
    TimelineSimulation,
    UpDownImportance,
)

if TYPE_CHECKING:
    from repyability.rbd.chunks import SimulationChunk

from repyability.rbd import (
    _ccf_groups,
    _crews,
    _curves,
    _long_run,
    _requirements,
    _windows,
)
from repyability.rbd._common import (
    _curve_points,
    _safe_ratio,
)
from repyability.rbd._tally import (
    _acquisition_only,
    _Breakdown,
    _CapacityRecorder,
    _check_shard_map,
    _chunk_means,
    _exact_controls,
    _Exacts,
    _ModuleRun,
    _shard_bytes,
    _Tally,
)
from repyability.rbd._time_order import (
    _capacity_totals,
    _group_totals,
    _working_over_time,
)
from repyability.utils.checks import (
    one_of,
    simulation_window,
)

if TYPE_CHECKING:
    from repyability.rbd.repairable_rbd import RepairableRBD


def _state_key(states: dict) -> Optional[str]:
    """The components' ``states`` at the start of a run (checked; see
    ``RepairableRBD._simulation_states``) as JSON, the same in every
    process, for a chunk's settings (see ``simulate_chunk``); None for
    every component new."""

    def plain(states: dict) -> list:
        out = []
        for node, start in states.items():
            if isinstance(start, NodeState):
                if start.new:
                    continue
                value: Any = dataclasses.asdict(start)
            else:
                value = plain(start)
                if not value:
                    continue
            out.append([repr(node), value])
        return sorted(out, key=lambda item: item[0])

    entries = plain(states)
    return json.dumps(entries, sort_keys=True) if entries else None


def _hot_units(spec: dict) -> "RepairableRBD":
    """A standby group's twin (see ``RepairableRBD._twin``): its units
    operating together, each failing and repaired on its own, ``k`` of them
    needed, as a nested RBD whose components are named ``0, 1, ...``, as
    the group's units are, so that their streams are the units'."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    standby = spec["standby"]
    units, k = int(standby.get("units", 2)), int(standby.get("k", 1))
    keys = ("reliability", "repairability") + RepairableRBD.COST_KEYS
    unit = {key: spec[key] for key in keys if key in spec}
    return RepairableRBD(
        [("s", u) for u in range(units)] + [(u, "t") for u in range(units)],
        {u: dict(unit) for u in range(units)},
        k={"t": k} if k > 1 else None,
    )


def _states_from_key(rbd, key: Optional[str]) -> dict:
    """The components' states ``_state_key`` saved, for ``rbd`` (a copy of
    the system the key was made for, as a shard rebuilds it): each node by
    its ``repr``, a nested RBD's own states in turn."""

    def states(rbd, entries: list) -> dict:
        nodes = {repr(node): node for node in rbd.components}
        out: dict = {}
        for name, value in entries:
            node = nodes[name]
            if isinstance(value, dict):
                out[node] = NodeState(**value)
            else:
                out[node] = states(rbd.components[node], value)
        return out

    return {} if key is None else states(rbd, json.loads(key))


def _stopping_rule(
    N: int,
    tolerance: Optional[float],
    confidence: float,
    max_N: Optional[int],
    antithetic: bool,
    target: str,
    t_simulation: float,
) -> Optional[Callable[..., int]]:
    """``None`` for a fixed number of replications; otherwise a function
    of the tally so far giving how many more replications to run (0 once
    the confidence interval of the mean availability over the window, or of
    the mean cost, is at most ``tolerance`` either side, or ``max_N`` have
    run: then with a warning). Given the simulations' ``values`` too (a
    controlled run's, see ``_controlled_run``), it judges those; given why
    they cannot be judged yet (``unjudged``), it runs on to the limit."""
    montecarlo.check_confidence(confidence)
    limit = montecarlo.sample_limit(
        N, tolerance, max_N, antithetic, ("mc_samples", "max_samples")
    )
    if limit is None:
        return None

    def stop(
        tally: "_Tally", values=None, unjudged: Optional[str] = None
    ) -> int:
        if values is not None:
            # The controlled values of a run with an exact twin.
            values = np.asarray(values, dtype=float)
        elif target == "cost":
            values = np.asarray(tally.cost_samples, dtype=float)
        else:
            values = np.asarray(tally.uptimes, dtype=float) / t_simulation
        return montecarlo.more_samples(
            values,
            N,
            tolerance,  # type: ignore[arg-type]
            confidence,
            limit,
            antithetic,
            target,
            "max_samples",
            unjudged,
        )

    return stop


#: Why antithetic pairs and common random numbers (``compare``) are refused
#: when some component's draws do not come from a stream.
_UNSTREAMED = (
    "Antithetic and common random numbers need every component's draws to "
    "be replayable (surpyval parametric distributions and the composite "
    "models built from them)."
)


def failure_criticality_index_per_system_failures(FCI, system_failures):
    """Share of all system failures that each node caused.

    A node causes a system failure when its own failure takes the system
    from up to down. For each node this is the number of system failures
    it caused divided by ``system_failures``; every node gets 0 when there
    were no system failures.

    Parameters
    ----------
    FCI : dict
        Node -> counts, with the number of system failures the node caused
        under ``"system_failures"`` (as accumulated by
        ``RepairableRBD.availability``).
    system_failures : int
        The total number of system failures.

    Returns
    -------
    dict
        Node -> share of the system failures, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     failure_criticality_index_per_system_failures as per_system,
    ... )
    >>> counts = {"a": {"system_failures": 3}, "b": {"system_failures": 1}}
    >>> per_system(counts, 4)
    {'a': 0.75, 'b': 0.25}
    """
    fci = {}
    for node in FCI.keys():
        if system_failures == 0:
            # No system failures occurred (e.g. a redundant node was forced
            # working, or the system was highly reliable over the simulated
            # window), so no node can be credited with causing one.
            fci[node] = 0
        else:
            fci[node] = FCI[node]["system_failures"] / system_failures
    return fci


def failure_criticality_index_per_component_failures(FCI):
    """Share of each node's own failures that caused a system failure.

    For each node: the number of system failures it caused (its failures
    that took the system from up to down) divided by its number of
    failures, or 0 if it never failed.

    Parameters
    ----------
    FCI : dict
        Node -> counts, with the system failures the node caused under
        ``"system_failures"`` and its own failures under
        ``"component_failures"`` (as accumulated by
        ``RepairableRBD.availability``).

    Returns
    -------
    dict
        Node -> share of the node's failures, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     failure_criticality_index_per_component_failures as per_component,
    ... )
    >>> per_component({"a": {"system_failures": 3, "component_failures": 4}})
    {'a': 0.75}
    """
    fci = {}
    for node in FCI.keys():
        try:
            fci[node] = (
                FCI[node]["system_failures"] / FCI[node]["component_failures"]
            )
        except ZeroDivisionError:
            # If there were no component failures then there were no
            # system failures caused by that node
            fci[node] = 0
    return fci


def restoration_criticality_index_by_system(RCI, system_restorations):
    """Share of all system restorations that each node caused.

    A node causes a system restoration when its own restoration takes the
    system from down to up. For each node this is the number of system
    restorations it caused divided by ``system_restorations``; every node
    gets 0 when there were no system restorations.

    Parameters
    ----------
    RCI : dict
        Node -> counts, with the number of system restorations the node
        caused under ``"system_restorations"`` (as accumulated by
        ``RepairableRBD.availability``).
    system_restorations : int
        The total number of system restorations.

    Returns
    -------
    dict
        Node -> share of the system restorations, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     restoration_criticality_index_by_system as by_system,
    ... )
    >>> by_system({"a": {"system_restorations": 1}}, 2)
    {'a': 0.5}
    """
    rci = {}
    for node in RCI.keys():
        if system_restorations == 0:
            # No system restorations occurred, so no node can be credited with
            # causing one.
            rci[node] = 0
        else:
            rci[node] = RCI[node]["system_restorations"] / system_restorations
    return rci


def restoration_criticality_index_by_component(RCI):
    """Share of each node's own restorations that restored the system.

    For each node: the number of system restorations it caused (its
    restorations that took the system from down to up) divided by its
    number of restorations, or 0 if it was never restored.

    Parameters
    ----------
    RCI : dict
        Node -> counts, with the system restorations the node caused under
        ``"system_restorations"`` and its own restorations under
        ``"component_restorations"`` (as accumulated by
        ``RepairableRBD.availability``).

    Returns
    -------
    dict
        Node -> share of the node's restorations, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     restoration_criticality_index_by_component as by_component,
    ... )
    >>> counts = {"system_restorations": 1, "component_restorations": 4}
    >>> by_component({"a": counts})
    {'a': 0.25}
    """
    rci = {}
    for node in RCI.keys():
        try:
            rci[node] = (
                RCI[node]["system_restorations"]
                / RCI[node]["component_restorations"]
            )
        except ZeroDivisionError:
            # If there were no component restorations then there were no
            # times the system was restored by restoring this node.
            rci[node] = 0
    return rci


def _engine_choice(diagram, capacity: bool) -> Tuple[str, str]:
    """The engine ``engine="auto"`` runs a long simulation on, and why
    (see ``_simulation_engine``)."""
    from repyability.rbd import _compiled

    plan = diagram._stream_plan(1.0, 0, False)[0]
    engine, reason = _compiled.choice(
        diagram, plan, object() if capacity else None
    )
    if reason is not None:
        return "python", f"the compiled engine does not simulate {reason}"
    if engine is None:
        return (
            "python",
            "numba is not installed; pip install 'repyability[fast]' "
            "for the compiled engine",
        )
    return (
        engine,
        "compiled for a run long enough to repay loading it; a short "
        "one runs in Python",
    )


def availability(
    diagram,
    t_simulation: float,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    mc_samples: Optional[int],
    verbose: bool,
    seed: Optional[int],
    *,
    tolerance: Optional[float],
    confidence: float,
    max_samples: Optional[int],
    antithetic: bool,
    n_jobs: Optional[int],
    demand: Optional[float],
    engine: str,
    state,
    curve_points: Optional[int],
    shard_map: Optional[Callable],
    shard_size: Optional[int],
    control_variate: Optional[bool],
    conditional: Optional[bool],
) -> AvailabilityResult:
    """See ``RepairableRBD.availability``."""
    N = 10_000 if mc_samples is None else mc_samples

    return _simulated(
        diagram,
        t_simulation,
        working_nodes,
        broken_nodes,
        method,
        N,
        verbose,
        seed,
        tolerance=tolerance,
        confidence=confidence,
        max_N=max_samples,
        antithetic=antithetic,
        n_jobs=n_jobs,
        target="availability",
        demand=demand,
        engine=engine,
        state=state,
        curve_points=_curve_points(curve_points),
        shard_map=shard_map,
        shard_size=shard_size,
        control_variate=control_variate,
        conditional=conditional,
    )


def simulate_timelines(
    diagram,
    t_simulation: float,
    mc_samples: Optional[int],
    seed: Optional[int],
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    antithetic: bool,
    engine: str,
    n_jobs: Optional[int],
    start: int,
    state,
) -> "TimelineSimulation":
    """See ``RepairableRBD.simulate_timelines``."""
    return _timeline_runs.simulate(
        diagram,
        t_simulation,
        1000 if mc_samples is None else mc_samples,
        seed,
        working_nodes,
        broken_nodes,
        antithetic,
        engine,
        n_jobs,
        start,
        state,
    )


def simulate_chunk(
    diagram,
    t_simulation: float,
    start: int,
    stop: int,
    *,
    seed: int,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    antithetic: bool,
    demand: Optional[float],
    engine: str,
    n_jobs: Optional[int],
    verbose: bool,
    state,
    curve_points: Optional[int],
    control_variate: Optional[bool],
    conditional: Optional[bool],
) -> "SimulationChunk":
    """See ``RepairableRBD.simulate_chunk``."""
    if seed is None:
        raise ValueError(
            "A chunk needs the run's seed: the chunks of a run share it."
        )
    for name, value in (("start", start), ("stop", stop)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a whole number, got {value!r}.")
    if not 0 <= start < stop:
        raise ValueError(
            f"The chunk must be simulations start to stop - 1, with 0 <= "
            f"start < stop; got start={start!r}, stop={stop!r}."
        )
    if antithetic and (start % 2 or stop % 2):
        raise ValueError(
            "With antithetic pairs, start and stop must be even, so that "
            "the chunk holds whole pairs."
        )
    means = _chunk_means(control_variate, conditional)
    working = set() if working_nodes is None else set(working_nodes)
    broken = set() if broken_nodes is None else set(broken_nodes)
    diagram._validate_node_overrides(working, broken)
    states = _curves._simulation_states(diagram, state, working | broken)
    chunk = _chunk(
        diagram,
        t_simulation,
        int(start),
        int(stop),
        _streams.entropy_of(seed),
        working,
        broken,
        method,
        antithetic,
        demand,
        engine,
        None if n_jobs is None else montecarlo.jobs(n_jobs),
        verbose,
        states,
        _curve_points(curve_points),
    )
    # Kept only where they differ from availability's defaults, so that
    # a chunk is saved as it was before them.
    chunk.settings.update(
        {name: value for name, value in means.items() if value is False}
    )
    return chunk


def _chunk(
    diagram,
    t_simulation: float,
    start: int,
    stop: int,
    entropy: int,
    working: set,
    broken: set,
    method: str,
    antithetic: bool,
    demand: Optional[float],
    engine: str,
    jobs: Optional[int],
    verbose: bool,
    states: dict,
    curve_points: Optional[int],
) -> "SimulationChunk":
    """Simulations ``start`` to ``stop - 1`` of the run of ``entropy``
    (see ``simulate_chunk``, whose arguments, checked, these are), as
    a chunk with the run's settings."""
    from repyability.rbd.chunks import SimulationChunk

    capacity = _chunk_capacity(diagram, broken, method, demand)
    tally = diagram._run(
        t_simulation,
        working,
        broken,
        method,
        stop - start,
        verbose,
        None,
        antithetic,
        capacity=capacity,
        jobs=jobs,
        engine=engine,
        entropy=entropy,
        first=start,
        states=states,
        curve_points=curve_points,
    )
    settings = {
        "t_simulation": float(t_simulation),
        "entropy": entropy,
        "method": method,
        "working_nodes": sorted(working, key=repr),
        "broken_nodes": sorted(broken, key=repr),
        "antithetic": bool(antithetic),
        "demand": None if demand is None else float(demand),
        "fingerprint": _fingerprint(diagram),
        "state": _state_key(states),
        "curve_points": tally.curve_points,
    }
    return SimulationChunk([(start, stop)], settings, tally)


def _chunk_capacity(
    diagram, broken: set, method: str, demand: Optional[float]
) -> Optional[_CapacityRecorder]:
    """Check a chunk's (or shard's) ``method`` and ``demand``, and give
    the recorder of the capacities it follows, if the system has
    them."""
    diagram.is_system_working(
        {c: c not in broken for c in diagram.components}, method
    )
    if diagram._has_capacity():
        return _CapacityRecorder(diagram, demand)
    if demand is not None:
        raise ValueError(
            "A demand is measured against capacities, and no node has "
            "one: give them with capacity={node: capacity}."
        )
    return None


def shards(
    diagram,
    t_simulation: float,
    mc_samples: Optional[int],
    *,
    seed: Optional[int],
    size: Optional[int],
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    antithetic: bool,
    demand: Optional[float],
    engine: str,
    state,
    curve_points: Optional[int],
) -> List[bytes]:
    """See ``RepairableRBD.shards``."""
    N = 10_000 if mc_samples is None else mc_samples
    montecarlo.check_count(N, antithetic, "mc_samples")
    working = set() if working_nodes is None else set(working_nodes)
    broken = set() if broken_nodes is None else set(broken_nodes)
    diagram._validate_node_overrides(working, broken)
    states = _curves._simulation_states(diagram, state, working | broken)
    capacity = _chunk_capacity(diagram, broken, method, demand)
    _ccf_groups._require_free_members(diagram, working, broken)
    _ccf_groups._require_groups_simulated(diagram, states)
    t_simulation = simulation_window(t_simulation)
    entropy = _streams.entropy_of(seed)
    template = _shard_template(
        diagram,
        t_simulation,
        entropy,
        working,
        broken,
        method,
        antithetic,
        demand,
        engine,
        states,
        _curve_points(curve_points),
    )
    # What a run checks before it simulates (see _run).
    plan, complete = diagram._stream_plan(
        t_simulation, entropy, antithetic, states
    )
    if antithetic and not complete:
        raise NotImplementedError(_UNSTREAMED)
    _simulation_engine(
        diagram, engine, plan, capacity, N, here=False, states=states
    )
    step = _shard_size(plan, size)
    return [
        _shard_bytes(template, first, min(first + step, N))
        for first in range(0, N, step)
    ]


def _shard_template(
    diagram,
    t_simulation: float,
    entropy: int,
    working: set,
    broken: set,
    method: str,
    antithetic: bool,
    demand: Optional[float],
    engine: str,
    states: dict,
    curve_points: Optional[int],
) -> dict:
    """What every shard of a run holds (see ``shards``): all but its
    range."""
    from repyability.rbd.shards import KIND

    return {
        "kind": KIND,
        "repyability_version": __version__,
        "system": _shard_system(diagram),
        "t_simulation": float(t_simulation),
        "entropy": int(entropy),
        "method": method,
        "working_nodes": sorted(working, key=repr),
        "broken_nodes": sorted(broken, key=repr),
        "antithetic": bool(antithetic),
        "demand": None if demand is None else float(demand),
        "engine": engine,
        "state": _state_key(states),
        "curve_points": curve_points,
    }


def _shard_system(diagram) -> dict:
    """This system as a shard carries it: ``to_dict``'s JSON data, with
    every model of a class that loads back as itself (see
    ``serialisation.exactly``), so that a worker simulates this
    system."""
    from repyability.rbd.repairable_rbd import RepairableRBD
    from repyability.rbd.serialisation import exactly

    try:
        with exactly():
            system = diagram.to_dict()
        json.dumps(system)
        if type(diagram) is not RepairableRBD:
            raise NotImplementedError(
                f"it is a {type(diagram).__name__}, which loads as a "
                "RepairableRBD"
            )
    except SAVE_ERRORS as error:
        raise NotImplementedError(
            "A shard carries the system as JSON, and this one cannot be "
            f"saved: {str(error).rstrip('.')}. Run it with "
            "availability(n_jobs=...) instead."
        ) from None
    return system


def _shard_size(plan: _streams.Plan, size: Optional[int]) -> int:
    """Simulations to a shard of the run of ``plan``: ``size`` (by
    default 1024), rounded up to a whole number of its widest block of
    draws, so that no two shards draw the same block."""
    wanted = 1024 if size is None else size
    if (
        isinstance(wanted, bool)
        or not isinstance(wanted, (int, np.integer))
        or wanted < 1
    ):
        raise ValueError(
            "The size of a shard must be a whole number of simulations, "
            f"at least 1; got {size!r}."
        )
    widest = max(
        [plan.columns(spec) for spec in plan.specs.values()]
        + [2 if plan.antithetic else 1]
    )
    return int(-(-int(wanted) // widest) * widest)


def availability_from_chunks(
    diagram,
    chunks,
    mc_samples: Optional[int],
    *,
    allow_gaps: bool,
    control_variate: Optional[bool],
    conditional: Optional[bool],
) -> AvailabilityResult:
    """See ``RepairableRBD.availability_from_chunks``."""
    from repyability.rbd.chunks import SimulationChunk

    if isinstance(chunks, (SimulationChunk, dict, bytes, bytearray)):
        chunks = [chunks]
    chunk = SimulationChunk.merge(chunks)
    if mc_samples is not None and chunk.ranges != [(0, int(mc_samples))]:
        raise ValueError(
            f"The chunks hold simulations {chunk.ranges} (as "
            "(start, stop) ranges), not all of simulations 0 to "
            f"{int(mc_samples) - 1}: some are missing."
        )
    if not allow_gaps and chunk.ranges != [(0, chunk.n_simulations)]:
        held = ", ".join(f"{a} to {b - 1}" for a, b in chunk.ranges)
        raise ValueError(
            f"The chunks hold simulations {held}: those between or "
            "before them are missing (a chunk that never came back?). "
            "Give them all, or pass allow_gaps=True for the result of "
            "the simulations held."
        )
    settings = chunk.settings
    means = _chunk_means(control_variate, conditional)
    for name in ("control_variate", "conditional"):
        if settings.get(name) is False:
            means[name] = False
    fingerprint = _fingerprint(diagram)
    if (
        settings["fingerprint"] is not None
        and fingerprint is not None
        and settings["fingerprint"] != fingerprint
    ):
        raise ValueError(
            "The chunks were simulated with another system (or with "
            "this one saved by another RePyability version)."
        )
    working = set(settings["working_nodes"])
    broken = set(settings["broken_nodes"])
    capacity = None
    if diagram._has_capacity():
        capacity = _CapacityRecorder(diagram, settings["demand"])
    initial_up = bool(
        diagram.is_system_working(
            {c: c not in broken for c in diagram.components},
            settings["method"],
        )
    )
    # The means the run's result takes by default (#187, #189).
    controls, record, breakdown = _default_means(
        diagram,
        chunk._tally,
        float(settings["t_simulation"]),
        working,
        broken,
        settings["method"],
        int(settings["entropy"]),
        bool(settings["antithetic"]),
        _states_from_key(diagram, settings.get("state")),
        chunk.ranges,
        **means,
    )
    return _availability_result(
        diagram,
        chunk._tally,
        settings["t_simulation"],
        initial_up,
        settings["antithetic"],
        capacity,
        controls,
        conditional=record,
        breakdown=breakdown,
    )


def _default_means(
    diagram,
    tally: "_Tally",
    t_simulation: float,
    working: set,
    broken: set,
    method: str,
    entropy: int,
    antithetic: bool,
    state,
    ranges: List[Tuple[int, int]],
    control_variate: Optional[bool] = None,
    conditional: Optional[bool] = None,
) -> Tuple[
    Tuple[Optional[ControlVariate], ...],
    Optional[ConditionalRun],
    Optional[_Breakdown],
]:
    """The means the result of a plain run takes by default, for its
    simulations ``ranges`` in ``tally`` (chunks merged, see
    ``availability_from_chunks``), as ``availability`` takes them: the
    exact methods', where they work them out (#187, see
    ``_exact_controls``), or given the histories of its modules,
    simulated again from the run's ``entropy`` (#189, see
    ``_conditioned_run``), with the expected cost's split found so;
    else none. ``control_variate=False`` takes neither, and
    ``conditional=False`` not the second (#236), as ``availability``
    does."""
    T = t_simulation
    if control_variate is False:
        return (None, None), None, None
    exact_means = _exact_means(diagram, T, working, broken, method, state)
    if exact_means is not None:
        return (
            _exact_controls(tally, T, exact_means, antithetic),
            None,
            exact_means.breakdown,
        )
    if conditional is False:
        return (None, None), None, None
    modules = _conditional_applies(diagram, working, broken, state)
    if not modules:
        return (None, None), None, None
    run = _ModuleRun(
        diagram,
        T,
        working,
        broken,
        method,
        entropy,
        antithetic=antithetic,
        n_jobs=None,
        engine="auto",
        curve_points=1,
        state=state,
        modules=modules,
    )
    try:
        run.prepare()
    except NotImplementedError:
        return (None, None), None, None
    for start, stop in ranges:
        run.simulate(start, stop - start)
    record = run.means()
    return (None, None), record, run.split(record)


def _fingerprint(diagram) -> Optional[str]:
    """A hash of this system saved as JSON (with the RePyability
    version), which chunks of its runs carry; None if it cannot be
    saved."""
    try:
        text = json.dumps(diagram.to_dict(), sort_keys=True)
    except SAVE_ERRORS:
        return None
    return hashlib.sha256(text.encode()).hexdigest()


def compare(
    diagram,
    other: "RepairableRBD",
    t_simulation: float,
    mc_samples: Optional[int],
    seed: Optional[int],
    *,
    quantity: str,
    confidence: float,
    n_jobs: Optional[int],
    engine: str,
    state,
    control_variate: Optional[bool],
) -> ConfidenceInterval:
    """See ``RepairableRBD.compare``."""
    N = 10_000 if mc_samples is None else mc_samples
    t_simulation = simulation_window(t_simulation)

    one_of("quantity", quantity, ("availability", "cost"))
    montecarlo.check_confidence(confidence)
    montecarlo.check_count(N, False, "mc_samples")
    if control_variate not in (None, False):
        raise ValueError(
            "compare takes control_variate=None (the exact difference "
            "where the exact methods work out both systems' expected "
            "values) or False (the simulated difference), got "
            f"{control_variate!r}: its common random numbers are its "
            "own variance reduction."
        )
    if quantity == "cost":
        for rbd in (diagram, other):
            if not (rbd.has_costs or rbd.acquisition_cost):
                raise ValueError(
                    "Both systems must be priced to compare their costs."
                )
    if control_variate is None:
        exact = _exact_difference(
            diagram, other, t_simulation, quantity, state
        )
        if exact is not None:
            return ConfidenceInterval(
                estimate=exact,
                lower=exact,
                upper=exact,
                confidence=confidence,
                standard_error=0.0,
                n_samples=0,
                method="exact",
            )
    jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
    states = [
        _curves._simulation_states(rbd, state) for rbd in (diagram, other)
    ]
    entropy = _streams.entropy_of(seed)
    values = []
    for rbd, start in zip((diagram, other), states):
        tally = rbd._run(
            t_simulation,
            set(),
            set(),
            "p",
            N,
            False,
            None,
            jobs=jobs,
            engine=engine,
            entropy=entropy,
            common=True,
            states=start,
        )
        if quantity == "cost":
            running = (
                np.asarray(tally.cost_samples, dtype=float)
                if rbd.has_costs
                else np.zeros(N)
            )
            values.append(running + rbd.acquisition_cost)
        else:
            values.append(np.asarray(tally.uptimes) / t_simulation)
    differences = values[0] - values[1]
    estimate = float(np.mean(differences))
    standard_error = montecarlo.standard_error(differences, False)
    z = montecarlo.z_value(confidence)
    return ConfidenceInterval(
        estimate=estimate,
        lower=estimate - z * standard_error,
        upper=estimate + z * standard_error,
        confidence=confidence,
        standard_error=standard_error,
        n_samples=N,
        method="simulated",
    )


def _exact_difference(
    diagram,
    other: "RepairableRBD",
    t_simulation: float,
    quantity: str,
    state,
) -> Optional[float]:
    """``compare``'s difference worked out exactly (#236): this
    system's expected value of ``quantity`` over the window less
    ``other``'s, where the exact methods work out both (see
    ``_exact_means``); else None."""
    values = []
    for rbd in (diagram, other):
        exacts = _exact_means(rbd, t_simulation, set(), set(), "p", state)
        if exacts is None:
            return None
        if quantity == "availability":
            values.append(exacts.availability)
            continue
        if exacts.cost is not None:
            running = exacts.cost
        elif rbd.has_costs:
            return None
        else:
            running = 0.0
        values.append(running + rbd.acquisition_cost)
    return float(values[0] - values[1])


def _twin(diagram) -> Tuple["RepairableRBD", List[str]]:
    """This system's exact twin (#154), and what it changes, in words.

    The twin has this system's diagram and components, with their
    models, failing and repaired independently: without the limit on
    repair crews or the maintenance groups, which tie components
    together, and, component by component, without what the exact
    methods over time do not take (see ``_twin_simpler``). Its expected
    values over a window are then exact (``mission_availability``,
    ``expected_cost``), and each of its streams is named as this
    system's is (a standby group's units as the twin's nested units),
    so that, simulated with common random numbers, it follows this
    system closely: see ``availability``'s ``control_variate``.

    Raises
    ------
    NotImplementedError
        If no such twin has exact values over time: a component's model
        is a probability, or a life is simulated.
    """
    from repyability.rbd import routes as r
    from repyability.rbd.repairable_rbd import RepairableRBD

    args = diagram._init_args
    changes = []
    if _crews._crews_limited(diagram):
        changes.append("the limit on repair crews")
    if diagram._maintenance:
        changes.append("the maintenance groups")
    specs: Dict[Any, Any] = {}
    for node, value in args["components"].items():
        if isinstance(value, RepairableRBD):
            nested, inner = _twin(value)
            specs[node] = nested
            changes.extend(f"{what} in {node!r}" for what in inner)
        elif isinstance(value, dict):
            spec = {
                key: item
                for key, item in value.items()
                if key not in ("priority", "group")
            }
            preventive = value.get("preventive")
            if preventive and "opportunity" in preventive:
                # Opportunities come from a group's stops.
                spec["preventive"] = {
                    key: item
                    for key, item in preventive.items()
                    if key != "opportunity"
                }
            specs[node] = spec
        else:
            specs[node] = value
    twin = _twin_from(diagram, specs)
    simpler = {}
    for node, spec in specs.items():
        if isinstance(spec, dict):
            simpler[node], change = _twin_simpler(twin, node, spec)
            if change:
                changes.append(change)
    if any(simpler[node] is not specs[node] for node in simpler):
        twin = _twin_from(diagram, {**specs, **simpler})
    if twin.ccf_groups and r.refusal(
        partial(_ccf_groups._require_groups_over_time, twin, {})
    ):
        # Its groups' chains cannot follow them over time (#158): its
        # members fail independently.
        twin = _twin_from(diagram, {**specs, **simpler}, groups=False)
        changes.append("the common-cause groups")
    route = _curves._over_time(twin)
    if route.route == r.REFUSED:
        raise NotImplementedError(
            "This system has no exact twin to control its simulation "
            f"by: {route.reason}"
        )
    if route.route == r.SIMULATED:
        raise NotImplementedError(
            "This system has no exact twin to control its simulation "
            "by: its components' values over time are simulated "
            f"({route.reason})"
        )
    return twin, changes


def _twin_report(diagram) -> str:
    """What ``analysis_routes`` says of the exact twin a run with
    ``control_variate`` is controlled by (see ``AnalysisRoute.twin``):
    what it leaves out of this system, or why there is none."""
    try:
        _, changes = _twin(diagram)
    except NotImplementedError as error:
        return f"none. {error}"
    _, complete = diagram._stream_specs(1.0)
    if not complete:
        return f"none. {_UNSTREAMED}"
    if not changes:
        return (
            "the system itself, whose values over the window are exact "
            "(mission_availability, expected_cost)."
        )
    return f"the system without {', '.join(changes)}."


def _twin_from(
    diagram, components: dict, groups: bool = True
) -> "RepairableRBD":
    """A system of this one's diagram with ``components`` (see
    ``_twin``), and its common-cause groups unless not ``groups``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    args = diagram._init_args
    return RepairableRBD(
        args["edges"],
        components,
        k=args["k"],
        input_node=args["input_node"],
        output_node=args["output_node"],
        on_infeasible_rbd=args["on_infeasible_rbd"],
        downtime_cost_rate=args["downtime_cost_rate"],
        ccf_groups=args.get("ccf_groups") if groups else None,
    )


def _twin_simpler(diagram, node, spec: dict) -> Tuple[Any, str]:
    """A component's spec in an exact twin (see ``_twin``), this system
    being the twin so far, and what it changes (or ""): a standby group
    whose own chain cannot follow it over time becomes its units
    operating together; imperfect repair becomes perfect (but for
    minimal repair in no time, whose values over time are exact);
    replacement on condition, inspections and block replacement that
    the exact methods do not take go."""
    from repyability.rbd import routes as r

    if node in diagram._standby:
        if r.refusal(partial(_long_run._standby_rates, diagram, node)):
            return (
                _hot_units(spec),
                f"the switching of standby group {node!r} (its units "
                "operate together)",
            )
        return spec, ""
    drop: Dict[str, str] = {}
    if node in diagram._imperfect and r.refusal(
        partial(
            _requirements._require_minimal_repair,
            diagram,
            node,
        )
    ):
        drop["repair"] = drop["replace_after"] = "imperfect repair"
    if node in diagram._inspection and r.refusal(
        partial(
            _requirements._require_tested_exact,
            diagram,
            node,
        )
    ):
        drop["inspection"] = "inspections"
    schedule = diagram._preventive.get(node)
    if (
        schedule is not None
        and schedule.policy in ("block", "condition")
        and r.refusal(
            partial(
                _requirements._require_block_models,
                diagram,
                node,
            )
        )
    ):
        drop["preventive"] = (
            "block replacement"
            if schedule.policy == "block"
            else "replacement on condition"
        )
    if not drop:
        return spec, ""
    what = list(dict.fromkeys(drop[key] for key in drop if key in spec))
    note = " (its failures are revealed)" if "inspection" in drop else ""
    return (
        {key: value for key, value in spec.items() if key not in drop},
        f"the {' and '.join(what)} of {node!r}{note}",
    )


def _simulated(
    diagram,
    t_simulation: float,
    working_nodes,
    broken_nodes,
    method: str,
    N: int,
    verbose: bool,
    seed,
    *,
    tolerance: Optional[float],
    confidence: float,
    max_N: Optional[int],
    antithetic: bool,
    n_jobs: Optional[int],
    target: str,
    demand: Optional[float] = None,
    engine: str = "auto",
    state=None,
    curve_points: Optional[int] = None,
    shard_map: Optional[Callable] = None,
    shard_size: Optional[int] = None,
    control_variate: Optional[bool] = None,
    conditional: Optional[bool] = None,
) -> AvailabilityResult:
    """``availability`` (and ``cost``): validate, run the replications
    (serially, in parallel, or as shards through ``shard_map``, until
    converged if asked; with ``control_variate``, alongside the system's
    exact twin; with ``conditional``, of the dependent modules alone,
    see ``_conditional_run``) and build the result, its curve on a grid
    of ``curve_points`` steps if given. By default its expected values
    are exact where the system is its own exact twin (#187), and taken
    given the modules' histories where a conditional run applies
    (#189, see ``_conditioned_run``)."""
    t_simulation = simulation_window(t_simulation)
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    diagram._validate_node_overrides(working_nodes, broken_nodes)
    states = _curves._simulation_states(
        diagram, state, working_nodes | broken_nodes
    )
    # The initial system state with every component up but those forced
    # broken, which can make the system start down; a simulation that
    # starts down from the components' states records the change at 0
    # (see _replicate).
    initial_status = {c: c not in broken_nodes for c in diagram.components}
    initial_up = bool(diagram.is_system_working(initial_status, method))
    montecarlo.check_count(N, antithetic, "mc_samples")
    stop = _stopping_rule(
        N, tolerance, confidence, max_N, antithetic, target, t_simulation
    )
    _check_shard_map(shard_map, shard_size, n_jobs)
    if control_variate is not None and not isinstance(
        control_variate, (bool, np.bool_)
    ):
        raise ValueError(
            f"control_variate must be True or False, got "
            f"{control_variate!r}."
        )
    if conditional is not None and not isinstance(
        conditional, (bool, np.bool_)
    ):
        raise ValueError(
            f"conditional must be True, False or None, got "
            f"{conditional!r}."
        )
    if conditional:
        if demand is not None and not diagram._has_capacity():
            raise ValueError(
                "A demand is measured against capacities, and no node "
                "has one: give them with capacity={node: capacity}."
            )
        return _conditional_run(
            diagram,
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            seed,
            stop=stop,
            antithetic=antithetic,
            n_jobs=n_jobs,
            engine=engine,
            curve_points=curve_points if target == "availability" else 1,
            target=target,
            state=state,
            shard_map=shard_map,
            shard_size=shard_size,
            demand=demand,
            capacities=target == "availability" and diagram._has_capacity(),
            control_variate=bool(control_variate),
        )
    # By default the expected values are the exact methods', where they
    # work them out (#187): a run to a tolerance stops at once.
    auto = control_variate is None
    exact_means = (
        _exact_means(
            diagram, t_simulation, working_nodes, broken_nodes, method, state
        )
        if auto
        else None
    )
    twin = None
    changes: List[str] = []
    exacts: Optional[_Exacts] = None
    if control_variate:
        twin, changes = _twin(diagram)
        if shard_map is not None and changes:
            raise ValueError(
                "A run with control_variate simulates the system's twin "
                "alongside it, here: leave out shard_map."
            )
        exacts = _twin_exact(
            diagram,
            twin,
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            state,
        )
    # Otherwise, by default, its means are taken given the histories of
    # its modules, where a conditional run applies (#189).
    modules = (
        _conditional_applies(diagram, working_nodes, broken_nodes, state)
        if auto and exact_means is None and conditional is None
        else []
    )
    capacity = None
    # Shards follow the capacities whatever the target, as chunks do.
    if (
        target == "availability" or shard_map is not None
    ) and diagram._has_capacity():
        capacity = _CapacityRecorder(diagram, demand)
    elif demand is not None:
        raise ValueError(
            "A demand is measured against capacities, and no node has "
            "one: give them with capacity={node: capacity}."
        )
    entropy = sharded = None
    if shard_map is not None:
        _ccf_groups._require_groups_simulated(diagram)
        entropy = _streams.entropy_of(seed)
        template = _shard_template(
            diagram,
            t_simulation,
            entropy,
            working_nodes,
            broken_nodes,
            method,
            antithetic,
            demand,
            engine,
            states,
            curve_points,
        )
        plan, _ = diagram._stream_plan(
            t_simulation, entropy, antithetic, states
        )
        step = _shard_size(plan, shard_size)
        sharded = (shard_map, template, step)
    jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
    controls: Tuple[Optional[ControlVariate], ...] = (None, None)
    # The expected cost's split, where the means are exact (#223).
    breakdown: Optional[_Breakdown] = None
    if modules:
        done = _conditioned_run(
            diagram,
            modules,
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            verbose,
            seed,
            antithetic,
            stop,
            capacity=capacity,
            jobs=jobs,
            n_jobs=n_jobs,
            engine=engine,
            state=state,
            states=states,
            curve_points=curve_points,
            target=target,
            sharded=sharded,
            shard_map=shard_map,
            shard_size=shard_size,
        )
        if done is not None:
            tally, record, breakdown = done
            return _availability_result(
                diagram,
                tally,
                t_simulation,
                initial_up,
                antithetic,
                capacity,
                conditional=record,
                breakdown=breakdown,
            )
    if twin is not None:
        assert exacts is not None
        tally, controls = _controlled_run(
            diagram,
            twin,
            exacts,
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            verbose,
            seed,
            antithetic,
            stop,
            capacity=capacity,
            jobs=jobs,
            engine=engine,
            state=state,
            states=states,
            curve_points=curve_points,
            target=target,
            itself=not changes,
            sharded=sharded,
        )
        if not changes:
            # Its means are the system's exact ones (#223).
            breakdown = exacts.breakdown
    else:
        tally = diagram._run(
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            verbose,
            seed,
            antithetic,
            # Exact means need no more simulations.
            None if exact_means is not None else stop,
            capacity=capacity,
            jobs=jobs,
            engine=engine,
            entropy=entropy,
            states=states,
            curve_points=curve_points,
            sharded=sharded,
        )
        if exact_means is not None:
            controls = _exact_controls(
                tally, t_simulation, exact_means, antithetic
            )
            breakdown = exact_means.breakdown
    return _availability_result(
        diagram,
        tally,
        t_simulation,
        initial_up,
        antithetic,
        capacity,
        controls,
        breakdown=breakdown,
    )


def _exact_means(
    diagram,
    t_simulation: float,
    working: set,
    broken: set,
    method: str,
    state,
) -> Optional[_Exacts]:
    """The system's own expected values over the window, where the
    exact methods work them out (#187): its mean availability and, if
    it is priced, its expected cost (see ``_twin_exact``); else None."""
    try:
        return _twin_exact(
            diagram, diagram, t_simulation, working, broken, method, state
        )
    except NotImplementedError:
        return None


def _conditional_applies(
    diagram, working: Collection = (), broken: Collection = (), state=None
) -> list:
    """The modules a conditional run would simulate (see
    ``_conditional_modules``) with the nodes ``working`` and ``broken``
    held and from the components' ``state``, or none if it would
    refuse: the others must be taken exactly given them, which a
    structure too meshed for its decision diagram, or common-cause
    groups the exact methods over time refuse, prevent."""
    from repyability.rbd import routes as r

    if diagram._too_meshed() is not None:
        return []
    if diagram.ccf_groups and r.refusal(
        partial(
            _ccf_groups._require_groups_over_time,
            diagram,
            state or {},
        )
    ):
        return []
    try:
        return diagram._conditional_modules(set(working), set(broken))
    except NotImplementedError:
        return []


def _modules_rbd(diagram, modules: list) -> "RepairableRBD":
    """The ``modules`` of a conditional run (see
    ``_conditional_modules``) as a diagram of their own, each between
    the input and output nodes with its own spec: their streams are
    named as in this system, so its simulations are this system's
    modules'. It has this system's repair crews (whose jobs are the
    modules' alone: crews that tie the components together make every
    node they serve a module), the maintenance groups of its members,
    and no system downtime cost."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    args = diagram._init_args
    options = args["maintenance_groups"] or {}
    groups = {
        name: spec
        for name, spec in options.items()
        if set(diagram._maintenance[name].members) <= set(modules)
    }
    return RepairableRBD(
        [(diagram.input_node, node) for node in modules]
        + [(node, diagram.output_node) for node in modules],
        {node: args["components"][node] for node in modules},
        input_node=diagram.input_node,
        output_node=diagram.output_node,
        repair_crews=args["repair_crews"],
        maintenance_groups=groups or None,
    )


def _capacity_given(
    diagram,
    modules: list,
    state: int,
    working: set,
    broken: set,
    x: np.ndarray,
    others,
    demand: Optional[float],
) -> "_conditional.CapacityGiven":
    """The system's capacity given the modules in one joint ``state``
    (see ``_conditional_given``), on the grid ``x``: from its exact
    distribution over time with the modules held so (a module up at its
    levels by their probabilities, as the simulation weighs them), and
    the other nodes from their states ``others``, against ``demand``
    (see ``_conditional.capacity_given``)."""
    up = {node for j, node in enumerate(modules) if state >> j & 1}
    down = set(modules) - up
    distribution = diagram.point_capacity(
        x, working | up, broken | down, state=others
    )
    return _conditional.capacity_given(
        x, distribution.levels, distribution.probabilities, demand
    )


def _up_at_start(diagram, node, start) -> bool:
    """Whether ``node`` is up just before 0 from its checked ``start``
    (see ``_states``; None, new): a component unless it starts down, a
    nested RBD as its system, from its components'."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if start is None:
        return True
    component = diagram.components[node]
    if isinstance(component, RepairableRBD):
        status = {
            inner: _up_at_start(component, inner, start.get(inner))
            for inner in component.components
        }
        return bool(component.is_system_working(status, "p"))
    return bool(start.alive)


def _crew_free(diagram) -> "RepairableRBD":
    """This system without its limit on repair crews: where every node
    the crews serve is held (a conditional run's modules), they tie
    nothing together, and the rest is worked out as if they were
    unlimited (a nested RBD keeps its own)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if not _crews._crews_limited(diagram):
        return diagram
    return RepairableRBD(**{**diagram._init_args, "repair_crews": None})


def _conditional_given(
    diagram,
    modules: list,
    state: int,
    working: set,
    broken: set,
    method: str,
    x: np.ndarray,
    others=None,
) -> "_conditional.Given":
    """The system given the modules in one joint ``state`` (bit ``j``
    set while module ``j`` is up), on the grid ``x``: with the modules
    held so, and the other nodes from their states ``others`` (new by
    default), its expected up time, failures and planned outages before
    each time, and its point availability (see
    ``_conditional.Given``)."""
    up = {node for j, node in enumerate(modules) if state >> j & 1}
    down = set(modules) - up
    _, _, _, counts, _, _ = _windows._window(
        diagram, x, working | up, broken | down, method, state=others
    )
    available = diagram.point_availability(
        x, working | up, broken | down, method, state=others
    )
    # Just before 0, every other node is up, but one started down.
    started = _curves._states(diagram, others, working | broken | up | down)
    status = {
        node: node not in broken | down
        and _up_at_start(diagram, node, started.get(node))
        for node in diagram.components
    }
    return _conditional.Given(
        x,
        np.asarray(counts["uptime"], dtype=float),
        np.asarray(counts["failures"], dtype=float),
        np.asarray(counts["planned"], dtype=float),
        np.asarray(available, dtype=float),
        float(bool(diagram.is_system_working(status, method))),
    )


def _conditional_run(
    diagram,
    t_simulation: float,
    working: set,
    broken: set,
    method: str,
    N: int,
    seed,
    *,
    stop: Optional[Callable[..., int]],
    antithetic: bool,
    n_jobs: Optional[int],
    engine: str,
    curve_points: Optional[int],
    target: str,
    state=None,
    shard_map: Optional[Callable] = None,
    shard_size: Optional[int] = None,
    demand: Optional[float] = None,
    capacities: bool = False,
    control_variate: bool = False,
) -> AvailabilityResult:
    """``availability`` (and ``cost``) with ``conditional=True`` (#189,
    see ``_conditional`` and ``_ModuleRun``): simulate the modules
    alone, in rounds while ``stop`` asks for more, and take every other
    node exactly given their joint states. From the components'
    ``state``, the modules start from theirs and the others from
    theirs. With ``shard_map``, the modules' simulations run as shards
    (see ``shards``) wherever it sends them. With ``capacities``, the
    system's capacity given the modules' states too, against
    ``demand``. With ``control_variate``, the exact twin's stand-ins for
    the modules (see ``_twin``) are simulated alongside them, with
    common random numbers, and each simulation's expected values given
    theirs, whose mean is the twin's exact one, control the run's (see
    ``ControlVariate``)."""
    run = _ModuleRun(
        diagram,
        t_simulation,
        working,
        broken,
        method,
        _streams.entropy_of(seed),
        antithetic=antithetic,
        n_jobs=n_jobs,
        engine=engine,
        curve_points=curve_points,
        state=state,
        shard_map=shard_map,
        shard_size=shard_size,
        demand=demand,
        capacities=capacities,
        control_variate=control_variate,
    )
    run.prepare()
    first, count = 0, N
    while True:
        run.simulate(first, count)
        first += count
        if stop is None:
            break
        # A run to a tolerance judges the mean asked for (see
        # _stopping_rule), controlled if it is: not while the modules
        # have not changed state (#215).
        count = stop(None, run.judged(target), run.unjudged())
        if not count:
            break
    return run.result()


def _conditioned_run(
    diagram,
    modules: list,
    t_simulation: float,
    working: set,
    broken: set,
    method: str,
    N: int,
    verbose: bool,
    seed,
    antithetic: bool,
    stop: Optional[Callable[..., int]],
    *,
    capacity: Optional[_CapacityRecorder],
    jobs: Optional[int],
    n_jobs: Optional[int],
    engine: str,
    state,
    states: dict,
    curve_points: Optional[int],
    target: str,
    sharded: Optional[tuple],
    shard_map: Optional[Callable],
    shard_size: Optional[int],
) -> Optional[Tuple["_Tally", ConditionalRun, Optional[_Breakdown]]]:
    """A plain run whose means are taken given its ``modules``'
    histories (#189), by default where a conditional run applies: the
    simulations, in rounds while ``stop`` asks for more (judged by
    those means), and of the same simulations each one's expected up
    time and cost given the histories of its modules, worked out as a
    conditional run works them out (see ``_ModuleRun``): the modules'
    simulations are the plain run's, drawn from the same streams. The
    simulations' totals, a record holding those values, and the
    expected cost's split given the modules (None unpriced, or while
    they have not changed state); or None, before simulating anything,
    if the rest cannot be worked out exactly given the modules."""
    entropy = _streams.entropy_of(seed)
    run = _ModuleRun(
        diagram,
        t_simulation,
        working,
        broken,
        method,
        entropy,
        antithetic=antithetic,
        n_jobs=n_jobs,
        # The modules' histories are the event loop's, on every engine.
        engine=engine if engine in ("python", "numba") else "auto",
        curve_points=1,
        state=state,
        shard_map=shard_map,
        shard_size=shard_size,
        modules=modules,
    )
    try:
        run.prepare()
    except NotImplementedError:
        return None
    tally: Optional[_Tally] = None
    first, count = 0, N
    while True:
        part = diagram._run(
            t_simulation,
            working,
            broken,
            method,
            count,
            verbose,
            None,
            antithetic,
            capacity=capacity,
            jobs=jobs,
            engine=engine,
            entropy=entropy,
            first=first,
            states=states,
            curve_points=curve_points,
            sharded=sharded,
        )
        if tally is None:
            tally = part
        else:
            tally.merge(part)
        run.simulate(first, count)
        first += count
        if stop is None:
            break
        # Its means given the modules, or the simulations' own while
        # the modules have not changed state (#215).
        more = stop(tally, None if run.unjudged() else run.judged(target))
        if not more:
            break
        count = more
    assert tally is not None
    record = run.means()
    return tally, record, run.split(record)


def _twin_exact(
    diagram,
    twin: "RepairableRBD",
    t_simulation: float,
    working: set,
    broken: set,
    method: str,
    state,
) -> _Exacts:
    """The exact ``twin``'s (see ``_twin``) mean availability over the
    window and, if both systems are priced, its expected cost, by
    category and by component (else None), building each component's
    curve once for both (#185)."""
    priced = diagram.has_costs and twin.has_costs
    # The cost's curves, which count events, serve the availability too.
    with twin._sharing_curves():
        expected = (
            twin.expected_cost(
                t_simulation, working, broken, method, state=state
            )
            if priced
            else None
        )
        exact = float(
            np.ravel(
                twin.mission_availability(
                    t_simulation, working, broken, method, state=state
                )
            )[0]
        )
    if expected is None:
        return _Exacts(exact)

    def first(value) -> float:
        return float(np.ravel(value)[0])

    return _Exacts(
        exact,
        first(expected.mean),
        (
            {key: first(v) for key, v in expected.by_category.items()},
            {node: first(v) for node, v in expected.by_component.items()},
        ),
    )


def _controlled_run(
    diagram,
    twin: "RepairableRBD",
    exacts: _Exacts,
    t_simulation: float,
    working: set,
    broken: set,
    method: str,
    N: int,
    verbose: bool,
    seed,
    antithetic: bool,
    stop: Optional[Callable[..., int]],
    *,
    capacity: Optional[_CapacityRecorder],
    jobs: Optional[int],
    engine: str,
    state,
    states: dict,
    curve_points: Optional[int],
    target: str,
    itself: bool = False,
    sharded: Optional[tuple] = None,
) -> Tuple["_Tally", Tuple[Optional[ControlVariate], ...]]:
    """Run this system and its exact ``twin`` (see ``_twin``) with
    common random numbers, as ``compare`` does, in rounds while
    ``stop`` asks for more, judged by the controlled values; return
    this system's totals and the controls of its fractions up and of
    its costs (None without costs) by the twin's (see
    ``ControlVariate``), ``itself`` if the twin is this system: by its
    exact values ``exacts``, its mean availability and its expected
    cost (None unpriced; see ``_twin_exact``). A twin that is this
    system is not simulated again, so its run can be ``sharded`` (see
    ``_run``)."""
    assert sharded is None or itself
    twin_states = _curves._simulation_states(twin, state, working | broken)
    exact, exact_cost = exacts.availability, exacts.cost
    entropy = _streams.entropy_of(seed)
    tally: Optional[_Tally] = None
    twin_tally: Optional[_Tally] = None
    first, count = 0, N
    while True:
        part = diagram._run(
            t_simulation,
            working,
            broken,
            method,
            count,
            verbose,
            None,
            antithetic,
            capacity=capacity,
            jobs=jobs,
            engine=engine,
            entropy=entropy,
            common=True,
            first=first,
            states=states,
            curve_points=curve_points,
            sharded=sharded,
        )
        # A twin that is this system, drawn from the same streams, runs
        # as this system does, to the last bit: its run is this one.
        twin_part = (
            None
            if itself
            else twin._run(
                t_simulation,
                working,
                broken,
                method,
                count,
                False,
                None,
                antithetic,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                common=True,
                first=first,
                states=twin_states,
                curve_points=1,
            )
        )
        if tally is None:
            tally, twin_tally = part, twin_part
        else:
            tally.merge(part)
            if twin_tally is not None and twin_part is not None:
                twin_tally.merge(twin_part)
        twins = tally if twin_tally is None else twin_tally
        fractions = np.asarray(tally.uptimes, dtype=float) / t_simulation
        control = ControlVariate.of(
            fractions,
            np.asarray(twins.uptimes, dtype=float) / t_simulation,
            exact,
            antithetic,
            itself=itself,
        )
        cost_control = None
        if exact_cost is not None:
            cost_control = ControlVariate.of(
                tally.cost_samples,
                twins.cost_samples,
                exact_cost,
                antithetic,
                itself=itself,
            )
        if stop is None:
            break
        if target == "cost":
            values = (
                tally.cost_samples
                if cost_control is None
                else cost_control.controlled(tally.cost_samples)
            )
        else:
            values = control.controlled(fractions)
        more = stop(tally, values)
        if not more:
            break
        first, count = first + count, more
    return tally, (control, cost_control)


def _simulation_engine(
    diagram,
    engine: str,
    plan: _streams.Plan,
    capacity: Optional[_CapacityRecorder],
    N: int,
    here: bool = True,
    states: Optional[dict] = None,
) -> str:
    """The engine that runs a simulation: ``"python"``, ``"numba"`` or
    one another package adds (see ``availability``'s ``engine``), for a
    run from the components' ``states``. Not ``here`` (the simulations
    run as shards elsewhere), only whether it can simulate the system
    is checked, and ``engine`` is kept, for each shard's worker to
    choose by."""
    from repyability.rbd import _compiled, engines

    added = engines.registered()
    if engine not in ("auto", "python", "numba") and engine not in added:
        names = ", ".join(
            repr(name) for name in ["auto", "python", "numba", *added]
        )
        raise ValueError(f"engine must be one of {names}, got {engine!r}.")
    if engine == "python":
        return engine
    if engine != "auto":
        # What the engine does not simulate is refused first: installing
        # it would not help.
        reason = _compiled.unsupported(
            diagram, plan, capacity, numba=engine == "numba", states=states
        )
        if reason is not None:
            raise NotImplementedError(
                f"The compiled engine does not simulate {reason}: use "
                "engine='python', or 'auto', which chooses the engine "
                "that can."
            )
        if here and engine == "numba":
            _compiled.require()
        elif here and not added[engine].available():
            raise ImportError(
                f"The {engine!r} simulation engine cannot run here."
            )
        return _compiled.ready(engine, auto=False) if here else engine
    if not here:
        return engine
    name, _ = _compiled.choice(diagram, plan, capacity, states)
    if name is not None and _compiled.worthwhile(plan, N, name):
        return _compiled.ready(name, auto=True)
    return "python"


def _availability_result(
    diagram,
    tally: "_Tally",
    t_simulation: float,
    initial_up: bool,
    antithetic: bool,
    capacity: Optional[_CapacityRecorder] = None,
    controls: Tuple[Optional[ControlVariate], ...] = (None, None),
    conditional: Optional[ConditionalRun] = None,
    breakdown: Optional[_Breakdown] = None,
) -> AvailabilityResult:
    """The ``AvailabilityResult`` of the replications in ``tally``: its
    exact totals rounded, once; with ``controls``, the controls of its
    fractions up and its costs by an exact twin (see
    ``_controlled_run``); with ``conditional``, the record of its means
    taken given its modules' histories (see ``_conditioned_run``); with
    ``breakdown``, the split of the expected cost its mean is found
    with, exact or given the modules (#223), in place of the
    simulations' own."""
    N = tally.n
    nodes = tally.nodes
    tally._fold()

    def rounded(totals) -> list:
        return [float(total) for total in totals]

    system_uptime = float(tally.system_uptime)
    system_downtime = float(tally.system_downtime)
    both_up = rounded(tally.intersection_uptime)
    both_down = rounded(tally.intersection_downtime)
    # Collect Importance/Criticality measures from the simulation
    # reference: https://www.weibull.com/pubs/2004rm_05B_02.pdf
    # Operational Criticality Index
    oci_down = {
        k: _safe_ratio(v, system_downtime) for k, v in zip(nodes, both_down)
    }
    oci_up = {k: _safe_ratio(v, system_uptime) for k, v in zip(nodes, both_up)}
    # Intersection Over Union Importance
    iou_up = {
        k: _safe_ratio(both, either)
        for k, both, either in zip(nodes, both_up, rounded(tally.union_uptime))
    }
    iou_down = {
        k: _safe_ratio(both, either)
        for k, both, either in zip(
            nodes, both_down, rounded(tally.union_downtime)
        )
    }
    FCI, RCI = tally.criticality_counts()
    # Failure Criticality Index Importance
    fci_sys = failure_criticality_index_per_system_failures(
        FCI, tally.system_failures
    )
    fci_comp = failure_criticality_index_per_component_failures(FCI)
    # Restoration Criticality Index Importance
    rci_sys = restoration_criticality_index_by_system(
        RCI, tally.system_restorations
    )
    rci_comp = restoration_criticality_index_by_component(RCI)
    criticalities = Criticalities(
        operational_criticality_index=UpDownImportance(
            up=oci_up, down=oci_down
        ),
        iou=UpDownImportance(up=iou_up, down=iou_down),
        failure_criticality_index=FailureCriticalityIndex(
            per_system_failure=fci_sys, per_component_failure=fci_comp
        ),
        restoration_criticality_index=RestorationCriticalityIndex(
            by_system=rci_sys, by_component=rci_comp
        ),
    )

    # The availability from t=0..t_simulation: how many of the N
    # simulated systems work after each time at which one changed state
    # (and at 0 and t_simulation whether or not any did), over N.
    if tally.binned is not None:
        # On the grid (#153): how many work at each of its times.
        assert tally.edges is not None
        time = tally.edges.copy()
        working = (N if initial_up else 0) + np.cumsum(tally.binned)
    else:
        changed_at, deltas = tally.state_changes()
        time, working = _working_over_time(
            changed_at,
            deltas,
            t_simulation,
            N if initial_up else 0,
        )
    system_availability = working / N

    cost_result = _acquisition_only(diagram, N, t_simulation, antithetic)
    if diagram.has_costs:
        # Per-component means cover each node's repair, replace,
        # preventive and own downtime cost. (System downtime is a
        # system-level quantity and is not attributed to components.)
        by_category, by_component = (
            breakdown
            if breakdown is not None
            else (
                {k: float(v) / N for k, v in tally.cost_by_category.items()},
                {k: float(v) / N for k, v in tally.cost_by_component.items()},
            )
        )
        cost_result = CostResult(
            samples=np.asarray(tally.cost_samples, dtype=float),
            t_simulation=t_simulation,
            n_simulations=N,
            acquisition_cost=diagram.acquisition_cost,
            by_category=by_category,
            by_component=by_component,
            antithetic=antithetic,
            control_variate=controls[1],
            conditional=conditional,
        )

    capacity_fields: dict = {}
    if capacity is not None:
        # The mean capacity from t=0..t_simulation: the simulated
        # systems' total expected capacity after each time at which one
        # changed (unlimited while one can carry an unlimited amount).
        if tally.capacity_binned is not None:
            # On the grid (#190): the total after each of its times.
            assert tally.edges is not None
            assert tally.unlimited_binned is not None
            times = tally.edges.copy()
            totals = np.cumsum(bin_totals(tally.capacity_binned, len(times)))
            unlimited = np.cumsum(tally.unlimited_binned) > 0
        else:
            at, steps, starts, free_at, counts = tally.capacity_changes()
            change = _group_totals(steps, starts)
            del steps, starts  # (their memory serves what follows)
            times, totals, unlimited = _capacity_totals(
                at, change, free_at, counts, t_simulation
            )
            del at, change
        # The mean over the systems, never below 0, and unlimited while
        # one can carry an unlimited amount: np.where(unlimited, inf,
        # np.maximum(totals / N, 0.0)), worked out in place.
        curve = np.divide(totals, N, out=totals)
        np.maximum(curve, 0.0, out=curve)
        curve[unlimited] = np.inf
        capacity_fields = dict(
            capacity_timeline=times,
            capacity=curve,
            capacity_time={
                level: float(tally.capacity_time[level])
                for level in sorted(tally.capacity_time)
            },
            demand=capacity.demand,
            delivered=(
                None
                if capacity.demand is None
                else np.asarray(tally.delivered, dtype=float)
            ),
        )

    return AvailabilityResult(
        timeline=time,
        availability=system_availability,
        system_uptime=system_uptime,
        time_simulated_to=t_simulation,
        criticalities=criticalities,
        node_uptime=dict(zip(nodes, rounded(tally.node_uptime))),
        node_downtime=dict(zip(nodes, rounded(tally.node_downtime))),
        system_downtime=system_downtime,
        system_failures=tally.system_failures,
        system_restorations=tally.system_restorations,
        n_simulations=N,
        cost=cost_result,
        system_planned_outages=tally.system_planned_outages,
        uptimes=np.asarray(tally.uptimes, dtype=float),
        antithetic=antithetic,
        opportunistic_renewals=(
            dict(zip(nodes, tally.opportunistic))
            if diagram._maintenance
            else None
        ),
        control_variate=controls[0],
        conditional=conditional,
        **capacity_fields,
    )


def cost(
    diagram,
    t_simulation: float,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    mc_samples: Optional[int],
    verbose: bool,
    seed: Optional[int],
    *,
    tolerance: Optional[float],
    confidence: float,
    max_samples: Optional[int],
    antithetic: bool,
    n_jobs: Optional[int],
    engine: str,
    state,
    shard_map: Optional[Callable],
    shard_size: Optional[int],
    control_variate: Optional[bool],
    conditional: Optional[bool],
) -> Optional[CostResult]:
    """See ``RepairableRBD.cost``."""
    N = 10_000 if mc_samples is None else mc_samples
    t_simulation = simulation_window(t_simulation)

    if not diagram.has_costs:
        if diagram.acquisition_cost:
            montecarlo.check_count(N, antithetic, "mc_samples")
        return _acquisition_only(diagram, N, t_simulation, antithetic)
    return _simulated(
        diagram,
        t_simulation,
        working_nodes,
        broken_nodes,
        method,
        N,
        verbose,
        seed,
        tolerance=tolerance,
        confidence=confidence,
        max_N=max_samples,
        antithetic=antithetic,
        n_jobs=n_jobs,
        target="cost",
        engine=engine,
        state=state,
        # The cost result has no curve: count the changes, not keep them.
        curve_points=1,
        shard_map=shard_map,
        shard_size=shard_size,
        control_variate=control_variate,
        conditional=conditional,
    ).cost
