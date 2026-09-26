"""Repairable reliability block diagrams: the ``RepairableRBD`` class.

Besides the class, the module holds the helpers its availability
simulation uses: the ``Event`` records of its event queue, the timeline
arithmetic behind the up/down criticality measures (``combined_timeline``,
``intersection``, ``union``, ``time_at_status``) and the failure and
restoration criticality index ratios.
"""

import heapq
import pprint
import warnings
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from typing import (
    Any,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np
from surpyval import ExactEventTime
from tqdm import tqdm

from repyability.non_repairable import NonRepairable
from repyability.rbd._sampling import UniformStream, inverse_sampler
from repyability.rbd.rbd import RBD
from repyability.rbd.results import (
    AvailabilityResult,
    CostResult,
    Criticalities,
    FailureCriticalityIndex,
    RestorationCriticalityIndex,
    UpDownImportance,
)


class _StreamedRBD:
    """Stands in for a nested :class:`RepairableRBD` component during
    ``availability()``: runs the nested RBD's own simulation, with its
    components' draws replayed from the shared :class:`UniformStream` by
    their own stand-ins (see ``RepairableRBD._streamed_components``)."""

    def __init__(self, rbd: "RepairableRBD", sources: dict):
        self._rbd = rbd
        self._sources = sources

    def initialize_event_queue(self, t_simulation):
        self._rbd.initialize_event_queue(t_simulation, sources=self._sources)

    def next_event(self):
        return self._rbd.next_event(sources=self._sources)


class _EventQueue:
    """The simulation's event queue.

    ``queue.PriorityQueue`` is this same heap (``heappush``/``heappop``) plus
    thread locking on every operation, which a single-threaded simulation
    only pays for; events come out in exactly the same order.
    """

    __slots__ = ("_heap",)

    def __init__(self):
        self._heap: list = []

    def put(self, event) -> None:
        heapq.heappush(self._heap, event)

    def get(self):
        return heapq.heappop(self._heap)

    def empty(self) -> bool:
        return not self._heap

    def qsize(self) -> int:
        return len(self._heap)


class _StreamedComponent:
    """Stands in for a :class:`NonRepairable` component during
    ``availability()``: alternates failure and repair draws exactly as
    ``NonRepairable.next_event`` does, but takes them from a shared
    :class:`UniformStream`, which reproduces the same numbers."""

    def __init__(self, failure, repair, stream: UniformStream):
        self._failure = failure
        self._repair = repair
        self._stream = stream
        self._fails_next = True

    def reset(self):
        self._fails_next = True

    def next_event(self):
        if self._fails_next:
            self._fails_next = False
            return self._stream.draw(self._failure), False
        self._fails_next = True
        return self._stream.draw(self._repair), True


def _stand_in(component, stream: UniformStream, made: dict):
    """A component's stand-in for ``RepairableRBD._streamed_components``, or
    ``None`` if its draws cannot be streamed. (A subclass may draw its events
    its own way, so only the classes themselves are streamed.)"""
    if type(component) is RepairableRBD:
        nested = component._streamed_components(stream, made)
        return None if nested is None else _StreamedRBD(component, nested)
    if type(component) is not NonRepairable:
        return None
    failure = inverse_sampler(component.reliability)
    repair = inverse_sampler(component.time_to_replace)
    if failure is None or repair is None:
        return None
    return _StreamedComponent(failure, repair, stream)


@dataclass(order=True)
class Event:
    """A component's change of state in the availability simulation.

    Events compare by ``time`` alone (``component`` and ``status`` take no
    part in comparisons), so the simulation's event queue releases them in
    time order.

    Attributes
    ----------
    time : float
        When the change happens, on the simulation clock.
    component : Hashable
        The node whose state changes.
    status : bool
        The node's state after the event: False for a failure, True for a
        restoration.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import Event
    >>> Event(12.0, "pump", True) > Event(10.0, "valve", False)
    True
    """

    time: float
    component: Hashable = field(compare=False)
    status: bool = field(compare=False)


def combined_timeline(
    timeline_1: List[Tuple[float, int]], timeline_2: List[Tuple[float, int]]
):
    """Merge two up/down timelines and count how many are up over time.

    Each timeline is a list of ``(time, change)`` pairs, as the
    availability simulation records them for a node and for the system: a
    first entry ``(0.0, 1)`` if it starts up or ``(0.0, 0)`` if it starts
    down, then ``(t, 1)`` for each restoration and ``(t, -1)`` for each
    failure, and a last entry ``(t_end, 0)`` closing the window. The
    changes of the two timelines are summed at equal times and accumulated
    in time order.

    Parameters
    ----------
    timeline_1 : list[tuple[float, int]]
        The first timeline, e.g. a node's.
    timeline_2 : list[tuple[float, int]]
        The second timeline, e.g. the system's.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The distinct times in increasing order, and at each time the number
        of the two (0, 1 or 2) that are up from that time until the next.

    Examples
    --------
    A node down on ``[2, 3)`` and a system down on ``[2, 5)``, over a
    window of length 10:

    >>> from repyability.rbd.repairable_rbd import combined_timeline
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> times, n_up = combined_timeline(node, system)
    >>> times.tolist(), n_up.tolist()
    ([0.0, 2.0, 3.0, 5.0, 10.0], [2, 0, 1, 2, 2])
    """
    joint_timeline: defaultdict = defaultdict(lambda: 0)
    for t, e in timeline_1 + timeline_2:
        joint_timeline[t] += e
    events: np.ndarray = np.fromiter(joint_timeline.values(), dtype=np.int8)
    timeline: np.ndarray = np.fromiter(joint_timeline.keys(), dtype=np.float64)
    idx = np.argsort(timeline)
    timeline = timeline[idx]
    events = events[idx]
    events = events.cumsum()
    return timeline, events


def intersection(timeline, event_cumsum):
    """Total time during which both of two timelines are up.

    Sums the lengths of the intervals ``[timeline[j], timeline[j + 1])``
    on which ``event_cumsum[j] == 2``. Pass ``2 - event_cumsum`` instead to
    get the total time during which both are down.

    Parameters
    ----------
    timeline : numpy.ndarray
        Increasing times, as returned by ``combined_timeline``.
    event_cumsum : numpy.ndarray
        How many of the two are up from each time, as returned by
        ``combined_timeline``.

    Returns
    -------
    float
        The total length of those intervals.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     combined_timeline,
    ...     intersection,
    ... )
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> times, n_up = combined_timeline(node, system)
    >>> float(intersection(times, n_up))  # both up
    7.0
    >>> float(intersection(times, 2 - n_up))  # both down
    1.0
    """
    from_idx = np.where(event_cumsum[:-1] == 2)[0]
    to_idx = from_idx + 1
    intersection = timeline[to_idx] - timeline[from_idx]
    return intersection.sum()


def union(timeline, event_cumsum):
    """Total time during which at least one of two timelines is up.

    Sums the lengths of the intervals ``[timeline[j], timeline[j + 1])``
    on which ``event_cumsum[j] > 0``. Pass ``2 - event_cumsum`` instead to
    get the total time during which at least one is down.

    Parameters
    ----------
    timeline : numpy.ndarray
        Increasing times, as returned by ``combined_timeline``.
    event_cumsum : numpy.ndarray
        How many of the two are up from each time, as returned by
        ``combined_timeline``.

    Returns
    -------
    float
        The total length of those intervals.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import combined_timeline, union
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> times, n_up = combined_timeline(node, system)
    >>> float(union(times, n_up))  # either up
    9.0
    >>> float(union(times, 2 - n_up))  # either down
    3.0
    """
    from_idx = np.where(event_cumsum[:-1] > 0)[0]
    to_idx = from_idx + 1
    union = timeline[to_idx] - timeline[from_idx]
    return union.sum()


def intersection_over_union(
    node_timeline: List[Tuple[float, int]],
    system_timeline: List[Tuple[float, int]],
):
    """Intersection over union of a node's and the system's up time.

    The time both are up divided by the time at least one is up, for one
    pair of timelines. (``RepairableRBD.availability`` computes its ``iou``
    measure from totals over all its simulations instead.) If neither is
    ever up the result is nan, with a numpy warning.

    Parameters
    ----------
    node_timeline : list[tuple[float, int]]
        The node's timeline, in the format described in
        ``combined_timeline``.
    system_timeline : list[tuple[float, int]]
        The system's timeline, in the same format.

    Returns
    -------
    float
        The ratio, in ``[0, 1]``: 1 when the two are up at exactly the same
        times.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import intersection_over_union
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> round(float(intersection_over_union(node, system)), 4)  # 7 / 9
    0.7778
    """
    timeline, event_cumsum = combined_timeline(node_timeline, system_timeline)
    return intersection(timeline, event_cumsum) / union(timeline, event_cumsum)


def time_at_status(timeline, status):
    """Sum the gaps that follow a timeline's entries of a given value.

    Sums ``t[j + 1] - t[j]`` over every entry ``(t[j], value)`` of
    ``timeline`` whose value equals ``status``. For a single node's or the
    system's timeline, in the format described in ``combined_timeline``,
    ``status=1`` gives its total up time: every up period starts at the
    initial ``(0.0, 1)`` entry or at a restoration. (``status=-1`` would
    miss a down period at the start, whose entry is ``(0.0, 0)``; the
    simulation only uses ``status=1``.)

    Parameters
    ----------
    timeline : list[tuple[float, int]]
        One node's or the system's timeline.
    status : int
        The entry value whose following gaps are summed.

    Returns
    -------
    float
        The total time.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import time_at_status
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> float(time_at_status(node, 1))
    9.0
    """
    t = np.array([a for a, _ in timeline])
    events = np.array([b for _, b in timeline])
    from_idx = np.where(events[:-1] == status)[0]
    to_idx = from_idx + 1
    union = t[to_idx] - t[from_idx]
    return union.sum()


def _squeeze_values(d: dict) -> dict:
    """Convert a dict of 1-element arrays (as produced by the base RBD
    importance helpers) to a dict of plain floats. Steady-state availability
    has no time dimension, so the repairable importances return floats."""
    return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in d.items()}


def _safe_ratio(numerator, denominator):
    """Ratio that is 0 when the denominator is 0.

    Several criticality/importance measures divide by a total (system uptime,
    downtime, failures, restorations, or a union of intervals) that can
    legitimately be 0 -- e.g. when a redundant component is forced working so
    the system never fails, or a component is forced broken. With nothing to
    attribute, the measure is 0 rather than a NaN/inf (or a crash).
    """
    return numerator / denominator if denominator else 0.0


def _mean_cost(cost) -> float:
    """A declared cost's expected value: the number itself, or the mean of a
    cost distribution."""
    if isinstance(cost, float):
        return cost
    return float(np.ravel(cost.mean())[0])


def _charges(cost, rng, batch: int = 1024) -> Iterator[float]:
    """The endless stream of amounts charged for one per-failure cost: the
    number itself every time, or a fresh draw from its distribution at each
    failure (drawn ``batch`` at a time, since ``qf`` is vectorised). Draws are
    clipped at 0; validation has already made a negative one negligible."""
    while True:
        if isinstance(cost, float):
            yield cost
        else:
            draws = np.ravel(cost.qf(rng.random(batch)))
            yield from np.maximum(draws, 0.0).tolist()


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


class RepairableRBD(RBD):
    """A reliability block diagram of repairable components.

    Each component alternates between working and failed: it fails after a
    time drawn from its reliability model, and is then repaired, as good
    as new, after a time drawn from its repairability model. Components fail
    and are repaired independently of one another and of the system state:
    there is no shared repair crew, and a component keeps running (and can
    fail) while the system is down. The system is up whenever its working
    components connect the input node to the output node.

    Two kinds of analysis are offered:

    - Exact long-run (steady-state) metrics, built from each component's
      availability ``MTTF / (MTTF + MTTR)`` and failure frequency
      ``1 / (MTTF + MTTR)``: ``mean_availability``, ``node_availability``,
      ``system_failure_frequency``, ``mean_up_time``, ``mean_down_time``,
      ``mean_time_between_failures``, ``expected_cost_rate`` and the
      importance measures.
    - Monte-Carlo simulation of a finite window ``[0, t_simulation]`` that
      starts with every component working: ``availability`` (availability
      over time, and criticality measures) and ``cost`` (the distribution
      of the window's cost).

    Times are in the time unit of the component models, and costs in the
    currency they are given in.

    Parameters
    ----------
    edges : Iterable[tuple[Hashable, Hashable]]
        The directed edges of the diagram, e.g.
        ``[("s", "a"), ("a", "b"), ("b", "t")]`` for ``"a"`` and ``"b"`` in
        series between the input node ``"s"`` and the output node ``"t"``.
    components : dict
        One entry per node other than the input and output nodes, keyed by
        node name. Each is one of:

        - A spec dict with ``"reliability"`` (a time-to-failure model, such
          as a fitted surpyval distribution) and ``"repairability"`` (a
          time-to-repair model, or ``"instant"`` for repair in zero time:
          the component still fails, and any repair or replace cost is
          charged, but it is never down), plus optional costs.
          ``"repair_cost"`` and ``"replace_cost"`` are charged at every
          failure of the component: each is a number, or a distribution of
          the cost (anything with ``qf`` and ``mean``, such as a fitted
          surpyval model) drawn afresh at each failure. ``"downtime_cost"``
          is a number charged per unit time the component is down, whether
          or not the system is. A cost left out, None or 0 prices nothing.
        - A [`NonRepairable`][repyability.NonRepairable], pairing a
          reliability model with a time-to-replace model. Each node gets
          its own copy, so one object can be given for several identical
          nodes.
        - A nested ``RepairableRBD``, which acts as one node that is up
          while its own system is up. In a simulation it runs its own
          simulation, on the same clock. It is not copied, so give each
          node its own object: nodes sharing one would share its
          simulation state, and ``availability`` then typically raises a
          ValueError. Its own costs are not counted in this RBD's costs.
    k : dict[Hashable, int], optional
        k-out-of-n nodes, as ``{node: k}``: the node passes only if at
        least ``k`` of the branches entering it are working (and the node
        itself is working). By default None, meaning every k is 1.
    input_node : Hashable, optional
        The input (source) node. By default None: the only node with no
        incoming edge. If given, it must be that node, or a ValueError is
        raised.
    output_node : Hashable, optional
        The output (sink) node. By default None: the only node with no
        outgoing edge. If given, it must be that node, or a ValueError is
        raised.
    on_infeasible_rbd : str, optional
        ``"raise"`` (the default), ``"warn"`` or ``"ignore"``: what to do if
        the diagram is invalid (it has a cycle, a node other than the input
        or output node with no incoming or no outgoing edge, an unusable
        ``k``, or a node with no entry in ``components``): raise a
        ValueError, warn and build the RBD anyway, or build it silently.
        The problems found are recorded in ``structure_check``.
    downtime_cost_rate : float, optional
        Cost per unit time the whole system is down (e.g. lost
        production), by default 0.0 (not priced).

    Attributes
    ----------
    components : dict
        Node name -> the model the node is simulated with: a
        ``NonRepairable`` (built from the spec dict, or a copy of the one
        given) or a nested ``RepairableRBD``.
    repairability : dict
        Node name -> its time-to-repair model (an ``ExactEventTime`` at 0
        for ``"instant"``), or None for a nested ``RepairableRBD``.
    costs : dict
        Node name -> ``{cost key: number or distribution}``, for the nodes
        that declare at least one non-zero cost.
    downtime_cost_rate : float
        The system downtime cost rate.
    input_node : Hashable
        The input node.
    output_node : Hashable
        The output node.
    nodes : list
        The names of the nodes other than the input and output nodes.
    structure_check : dict
        The results of the structural checks made at construction.
    COST_KEYS : tuple[str, ...]
        The optional cost keys of a spec dict: ``"repair_cost"``,
        ``"replace_cost"`` and ``"downtime_cost"``.
    PER_FAILURE_COST_KEYS : tuple[str, ...]
        The cost keys charged per failure, which may be distributions:
        ``"repair_cost"`` and ``"replace_cost"``.
    COMPONENT_SPEC_KEYS : tuple[str, ...]
        Every key a spec dict may carry.

    Raises
    ------
    ValueError
        If a spec dict has an unknown key, or a ``"repairability"`` string
        other than ``"instant"``; if a cost is not a finite, non-negative
        number, a ``"downtime_cost"`` or ``downtime_cost_rate`` is a
        distribution, or a cost distribution has no finite mean or puts
        appreciable probability on a negative cost (its 1e-12 quantile is
        below 0); if a reliability model is not a surpyval parametric or
        non-parametric model or a ``StandbyModel``; if ``input_node`` or
        ``output_node`` is not in the diagram; or if the diagram is invalid
        and ``on_infeasible_rbd`` is ``"raise"``.
    KeyError
        If a spec dict has no ``"reliability"`` or no ``"repairability"``.

    Examples
    --------
    Two pumps in parallel, each failing on average every 10 hours and
    taking 1 hour on average to repair:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> pump = {
    ...     "reliability": surv.Exponential.from_params([0.1]),
    ...     "repairability": surv.Exponential.from_params([1.0]),
    ... }
    >>> pumps = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": pump, "b": pump},
    ... )
    >>> round(pumps.mean_availability(), 4)  # 1 - (1 / 11) ** 2
    0.9917
    >>> result = pumps.availability(t_simulation=50, N=200, seed=0)
    >>> window = result.n_simulations * result.time_simulated_to
    >>> round(float(result.system_uptime) / window, 4)  # simulated
    0.9927

    The pair as one node of a larger system, in series with a valve that
    is replaced instantly at a cost of 250 per failure, while lost
    production costs 1000 per hour of system downtime:

    >>> plant = RepairableRBD(
    ...     [("s", "pumps"), ("pumps", "valve"), ("valve", "t")],
    ...     {
    ...         "pumps": pumps,
    ...         "valve": {
    ...             "reliability": surv.Weibull.from_params([500.0, 2.0]),
    ...             "repairability": "instant",
    ...             "replace_cost": 250.0,
    ...         },
    ...     },
    ...     downtime_cost_rate=1000.0,
    ... )
    >>> round(plant.mean_availability(), 4)  # the valve is never down
    0.9917
    >>> round(plant.expected_cost_rate(), 2)  # cost per hour, long run
    8.83

    A ``NonRepairable`` can stand in for a spec dict, and one object can
    serve several nodes, since each node gets its own copy:

    >>> from repyability import NonRepairable
    >>> unit = NonRepairable(
    ...     surv.Exponential.from_params([0.1]),
    ...     surv.Exponential.from_params([1.0]),
    ... )
    >>> same = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": unit, "b": unit},
    ... )
    >>> round(same.mean_availability(), 4)
    0.9917
    >>> same.components["a"] is same.components["b"]
    False
    """

    #: Optional per-component cost fields accepted in a component spec dict.
    COST_KEYS = ("repair_cost", "replace_cost", "downtime_cost")
    #: The costs charged per failure, which may also be given as a
    #: distribution of the cost (drawn afresh at each failure).
    PER_FAILURE_COST_KEYS = ("repair_cost", "replace_cost")
    #: Every key a component spec dict may carry.
    COMPONENT_SPEC_KEYS = ("reliability", "repairability") + COST_KEYS

    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        components: dict[Any, Any],
        k: Optional[dict[Any, int]] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
        downtime_cost_rate: float = 0.0,
    ):
        # Capture the constructor inputs verbatim (before any mutation) so the
        # RBD can be faithfully serialised via to_dict()/to_json().
        edges = list(edges)
        self._init_args = {
            "edges": [tuple(e) for e in edges],
            "components": dict(components),
            "k": dict(k) if k else None,
            "input_node": input_node,
            "output_node": output_node,
            "on_infeasible_rbd": on_infeasible_rbd,
            "downtime_cost_rate": downtime_cost_rate,
        }
        self.downtime_cost_rate = self._validate_cost(
            "<system>", "downtime_cost_rate", downtime_cost_rate
        )
        # Per-node cost fields, pulled out of the component specs: a number,
        # or (per-failure costs only) a distribution. Only nodes that declare
        # at least one non-zero cost appear here.
        self.costs: dict[Any, dict[str, Any]] = {}
        components = copy(components)
        reliability = {}
        repairability = {}
        for name, component in components.items():
            if isinstance(component, dict):
                self._validate_component_spec(name, component)
                node_costs = {}
                for key in self.COST_KEYS:
                    if component.get(key) is None:
                        continue
                    cost = self._validate_component_cost(
                        name, key, component[key]
                    )
                    # A cost of 0 prices nothing, so it is left out.
                    if not (isinstance(cost, float) and cost == 0.0):
                        node_costs[key] = cost
                if node_costs:
                    self.costs[name] = node_costs
                repair_model = component["repairability"]
                if isinstance(repair_model, str):
                    # "instant": repaired in zero time. The component still
                    # *fails* (failure events fire, and any repair/replace
                    # cost is charged), but each outage has zero length, so
                    # it contributes no downtime and its availability is 1.
                    # The convenient assumption when repairs are much faster
                    # than the timescale being studied, or when no
                    # repair-time data exists.
                    if repair_model != "instant":
                        raise ValueError(
                            f"Component {name!r}: unknown repairability "
                            f"{repair_model!r}. Pass a fitted time-to-repair "
                            "model, or the string 'instant' for repair in "
                            "zero time."
                        )
                    repair_model = ExactEventTime.from_params(0)
                components[name] = NonRepairable(
                    component["reliability"], repair_model
                )
                reliability[name] = component["reliability"]
                repairability[name] = repair_model
            elif isinstance(component, RepairableRBD):
                reliability[name] = component
                repairability[name] = None
            elif isinstance(component, NonRepairable):
                # A NonRepairable carries its simulation state (whether it
                # fails or is repaired next), so each node gets its own copy:
                # one object can then be given for several identical parts,
                # here or in a nested RBD.
                components[name] = copy(component)
                reliability[name] = component.reliability
                repairability[name] = component.time_to_replace

        super().__init__(
            edges,
            set(components.keys()),
            k,
            input_node,
            output_node,
            on_infeasible_rbd,
        )

        # Every intermediate graph node needs a component definition (the
        # input/output nodes do not). Surface missing ones now with a clear
        # error rather than a KeyError mid-simulation.
        missing = [
            n
            for n in self.G.nodes
            if n not in components and n not in self.in_or_out
        ]
        self.structure_check["is_missing_components"] = bool(missing)
        self.structure_check["nodes_with_no_component"] = missing
        if missing:
            self.structure_check["is_valid"] = False
            if on_infeasible_rbd == "raise":
                raise ValueError(
                    f"Node(s) {sorted(missing, key=str)} have no entry in "
                    "the components dict."
                )
            elif on_infeasible_rbd == "warn":
                warnings.warn(
                    "Nodes with no component definition: "
                    + pprint.pformat(missing),
                    stacklevel=2,
                )

        self.components = components
        self.repairability = copy(repairability)

    @classmethod
    def _validate_component_spec(cls, node, spec: dict) -> None:
        """Reject unknown keys in a component spec.

        A mistyped cost key (``repair_costs``) would otherwise be silently
        ignored and priced at zero, which is a quiet way to get the money
        wrong; surface it at construction instead.
        """
        unknown = set(spec) - set(cls.COMPONENT_SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"Component {node!r} has unknown key(s) "
                f"{sorted(map(str, unknown))}. A component spec takes "
                f"{', '.join(cls.COMPONENT_SPEC_KEYS)}."
            )

    @classmethod
    def _validate_cost(cls, node, key: str, value) -> float:
        """Coerce a cost to a finite, non-negative float."""
        if hasattr(value, "qf"):
            allowed = ", ".join(cls.PER_FAILURE_COST_KEYS)
            raise ValueError(
                f"{node!r}: {key} must be a number, not a distribution. Only "
                f"the per-failure costs ({allowed}) may be distributions; a "
                "downtime cost is a rate, and the outage durations already "
                "make it random."
            )
        try:
            cost = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{node!r}: {key} must be a number, got {value!r}."
            ) from None
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError(
                f"{node!r}: {key} must be finite and non-negative, got "
                f"{value!r}."
            )
        return cost

    @classmethod
    def _validate_component_cost(cls, node, key: str, value):
        """Validate a per-component cost: a number (coerced to float) or, for
        the per-failure costs, a distribution of the cost -- anything with
        ``qf`` and ``mean``, such as a fitted surpyval model.

        A distribution must have a finite mean and put no more than 1e-12
        probability on a negative cost.
        """
        if key not in cls.PER_FAILURE_COST_KEYS or not hasattr(value, "qf"):
            return cls._validate_cost(node, key, value)
        mean = _mean_cost(value)
        if not np.isfinite(mean):
            raise ValueError(
                f"{node!r}: the {key} distribution must have a finite mean, "
                f"got {mean:g}."
            )
        lowest = float(np.ravel(value.qf(np.array([1e-12])))[0])
        if not lowest >= 0.0:
            raise ValueError(
                f"{node!r}: the {key} distribution puts appreciable "
                f"probability on a negative cost (its 1e-12 quantile is "
                f"{lowest:g}). Use one on [0, inf), such as a LogNormal, "
                "Gamma or Weibull."
            )
        return value

    @property
    def has_costs(self) -> bool:
        """Whether any cost has been declared (per-component or system-wide).

        True if some component declares a non-zero ``"repair_cost"``,
        ``"replace_cost"`` or ``"downtime_cost"`` (a cost distribution
        always counts), or ``downtime_cost_rate`` is non-zero. Costs of 0
        price nothing, and costs declared inside a nested ``RepairableRBD``
        do not count. When nothing is priced there is no cost model to
        evaluate, so the cost methods short-circuit rather than doing the
        work: ``expected_cost_rate`` returns 0.0, ``cost`` returns None, and
        ``availability`` skips the cost accounting (its result's ``cost`` is
        None).

        Returns
        -------
        bool
            True if anything is priced.
        """
        return bool(self.costs) or bool(self.downtime_cost_rate)

    def _streamed_components(
        self, stream: UniformStream, made: Optional[dict] = None
    ) -> Optional[dict[Any, Any]]:
        """Stand-ins that replay every component's failure/repair draws from
        ``stream``, nested RBDs' components included, or ``None`` if any
        component's draws cannot be reproduced exactly that way (a
        non-parametric model, ...), in which case the simulation draws from
        the components themselves. The stand-ins are called at exactly the
        points the components would be, so the draws come in the same order.

        A component object used for several nodes has one simulation state,
        so it gets one stand-in (``made``, by object). Since each node has
        its own copy of a ``NonRepairable``, only a nested RBD object can be.
        """
        made = {} if made is None else made
        streamed: dict[Any, Any] = {}
        for name, component in self.components.items():
            if id(component) not in made:
                made[id(component)] = _stand_in(component, stream, made)
            if made[id(component)] is None:
                return None
            streamed[name] = made[id(component)]
        return streamed

    def _failure_charges(self) -> dict[Any, list[tuple[str, Iterator[float]]]]:
        """For each costed node, its ``("repair" | "replace", charges)``
        pairs: the stream of amounts charged at the node's successive
        failures in a simulation (see :func:`_charges`).

        A cost distribution draws from its own generator, seeded from -- but
        not consuming -- numpy's global RNG: a seeded run stays reproducible,
        and the failure/repair draws (so every availability output) are the
        same whether or not any cost is random.
        """
        state = np.random.get_state()
        rng = np.random.default_rng(np.random.randint(2**31 - 1))
        np.random.set_state(state)
        return {
            node: [
                (key.removesuffix("_cost"), _charges(node_costs[key], rng))
                for key in self.PER_FAILURE_COST_KEYS
                if key in node_costs
            ]
            for node, node_costs in self.costs.items()
        }

    def expected_cost_rate(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the long-run expected cost per unit time.

        Exact (no simulation), from the steady-state quantities:

        ```text
        rate = downtime_cost_rate * (1 - A_sys)
               + sum_i omega_i * (repair_cost_i + replace_cost_i)
               + sum_i (1 - A_i) * downtime_cost_i
        ```

        where ``A_sys`` is the system's long-run availability
        (``mean_availability``), ``A_i`` is node i's, and
        ``omega_i = 1 / (MTTF_i + MTTR_i)`` is node i's long-run failure
        frequency. So ``repair_cost`` and ``replace_cost`` are charged per
        corrective action (failure), while the downtime rates are charged
        per unit time down. A per-failure cost given as a distribution
        enters through its mean. The result is in cost per unit of the
        component models' time.

        Every cost is optional and defaults to 0, so any subset can be
        priced; with nothing priced (see ``has_costs``) this is 0.0. Costs
        are corrective-only and undiscounted, and only this RBD's own costs
        count: costs declared inside a nested ``RepairableRBD`` are left
        out. The simulated counterpart is ``cost``, whose ``cost_rate``
        converges to this value as the window grows.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working: they never fail, so
            they incur no corrective or downtime cost, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed: they incur no corrective
            cost (they never change state) but are down for all time, by
            default None.

        Returns
        -------
        float
            Expected cost per unit time.

        Raises
        ------
        ValueError
            If something is priced and a working/broken node is unknown, is
            the input or output node, or is in both sets, or a component
            has a non-parametric reliability model. (With nothing priced
            this returns 0.0 without any checks.)

        Examples
        --------
        One component with MTTF 10 and MTTR 1 is up 10/11 of the time and
        fails 1/11 times per unit time, so at 100 per repair and 50 per unit
        time of outage the cost rate is ``(100 + 50) / 11``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...             "repair_cost": 100.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=50.0,
        ... )
        >>> round(rbd.expected_cost_rate(), 4)
        13.6364
        """
        if not self.has_costs:
            return 0.0

        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        forced = working_nodes | broken_nodes

        rate = 0.0

        # Production lost while the *system* is down.
        if self.downtime_cost_rate:
            unavailability = 1.0 - self.mean_availability(
                working_nodes, broken_nodes
            )
            rate += self.downtime_cost_rate * unavailability

        if not self.costs:
            return rate

        node_availability = _squeeze_values(
            self._probabilities_with_overrides(
                self.node_availability(), working_nodes, broken_nodes
            )
        )
        for node, node_costs in self.costs.items():
            # Corrective actions, charged per failure. A forced node never
            # changes state, so it never incurs one.
            per_action = sum(
                _mean_cost(node_costs[key])
                for key in self.PER_FAILURE_COST_KEYS
                if key in node_costs
            )
            if per_action and node not in forced:
                rate += per_action * self._node_failure_frequency(
                    self.components[node]
                )
            # Optional cost of *this component* being down, whether or not
            # the system as a whole is.
            downtime_cost = node_costs.get("downtime_cost", 0.0)
            if downtime_cost:
                rate += downtime_cost * (1.0 - node_availability[node])
        return rate

    def initialize_event_queue(
        self,
        t_simulation,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        sources: Optional[dict] = None,
    ):
        """Start one simulation of the system over ``[0, t_simulation)``.

        Advanced, event-stepping API: ``availability`` runs its simulations
        through it, and a parent RBD uses it to drive a nested
        ``RepairableRBD``, which therefore has the same event API as a
        ``NonRepairable`` component. Every component starts working except
        the ``broken_nodes``, which stay down for the whole window; the
        ``working_nodes`` never fail. The first failure of each other
        component is queued (a nested RBD starts its own simulation and
        queues its first state change), and events at or after
        ``t_simulation`` are dropped. Then call ``next_event`` repeatedly to
        step through the system's state changes; see it for an example.

        Unlike ``availability``, this takes no seed and does not check the
        working/broken nodes: the draws come from numpy's global RNG, so
        seed that (``np.random.seed``) for a reproducible run. The state is
        kept on the RBD itself (``system_state``, ``component_status``,
        ``t_simulation``), and ``availability`` uses and then deletes the
        same state, so do not interleave the two.

        Parameters
        ----------
        t_simulation : float
            The end of the simulated window.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window, by default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); both give the
            same state.
        sources : dict, optional
            Internal: node name -> the object each component's events are
            drawn from, which ``availability`` passes to replay the draws
            in batches. By default None: the components themselves.

        Raises
        ------
        ValueError
            If ``method`` is not ``"p"`` or ``"c"``.
        """
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        # What the components draw their events from: themselves, or
        # stand-ins replaying the same draws (see _streamed_components).
        sources = self.components if sources is None else sources

        # Keep record of component status', initially they're all working
        component_status: dict[Any, bool] = {
            component: True for component in self.components.keys()
        }

        for component in broken_nodes:
            component_status[component] = False

        # The queue supplies failure/repair events in chronological order
        event_queue = _EventQueue()

        # For each component add in the initial failure
        for component_id in self.components.keys():
            component = self.components[component_id]
            if component_id in working_nodes:
                continue
            elif component_id in broken_nodes:
                continue
            # If status not known, then continue
            if isinstance(component, RepairableRBD):
                source = sources[component_id]
                source.initialize_event_queue(t_simulation)
                t_event, event = source.next_event()
            elif isinstance(component, NonRepairable):
                source = sources[component_id]
                source.reset()
                t_event, event = source.next_event()

            # Only consider it if it occurs within the simulation window
            if t_event < t_simulation:
                # Event status is False => event is a component failure
                event_queue.put(Event(t_event, component_id, event))
        self._event_queue = event_queue
        # The initial system state must reflect any forced-broken components
        # (e.g. a broken component in series starts the system down), rather
        # than assuming everything is up.
        self.system_state = self.is_system_working(component_status, method)
        self.t_simulation = t_simulation
        self.component_status = component_status

    def mean_unavailability(self, *args, **kwargs) -> float:
        """Returns the system's long-run (steady-state) unavailability.

        Exactly ``1 - mean_availability(*args, **kwargs)``: the long-run
        fraction of time the system is down.

        Parameters
        ----------
        *args
            Positional arguments of ``mean_availability``
            (``working_nodes``, ``broken_nodes``, ``method``).
        **kwargs
            Keyword arguments of ``mean_availability``.

        Returns
        -------
        float
            Long-run unavailability of the system, in ``[0, 1]``.

        Raises
        ------
        ValueError
            As for ``mean_availability``.
        """
        return 1 - self.mean_availability(*args, **kwargs)

    def mean_availability(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> float:
        """Returns the system's long-run (steady-state) availability.

        Exact, with no simulation. Each component's long-run availability
        is ``MTTF / (MTTF + MTTR)`` (1 if it is repaired instantly; a nested
        ``RepairableRBD``'s is its own ``mean_availability``). Since the
        components fail and are repaired independently, the system's is
        the RBD's structure function evaluated exactly at those
        availabilities. It is the long-run fraction of time the system is
        up; for availability over time, from a start with everything
        working, use ``availability``.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (availability 1), by
            default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (availability 0), by
            default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"``, the default) or the minimal cut sets (``"c"``); both
            give the same exact result.

        Returns
        -------
        float
            Long-run availability of the system, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model, whose MTTF this cannot
            compute.

        Examples
        --------
        A single component with mean time to failure 10 and mean time to
        repair 1 has long-run availability ``10 / (10 + 1)``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> round(rbd.mean_availability(), 4)
        0.9091
        """
        # Good reference on the Availability of a system
        # https://www.diva-portal.org/smash/get/diva2:986067/FULLTEXT01.pdf
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)

        # Cache all component availabilities for efficiency
        component_availability: dict[Hashable, float] = {}
        for comp in self.components:
            if comp in working_nodes:
                component_availability[comp] = 1.0
            elif comp in broken_nodes:
                component_availability[comp] = 0.0
            else:
                component_availability[comp] = float(
                    np.atleast_1d(self.components[comp].mean_availability())[0]
                )

        for comp in self.in_or_out:
            component_availability[comp] = 1.0

        mean_availability = self.system_probability(
            component_availability, method=method
        )
        return mean_availability.item()

    def _next_event_time(self, event: Event, t: float) -> float:
        """When the component of ``event`` next changes state, given ``t``
        from its ``next_event()``.

        A component's ``next_event()`` gives the time *to* its next event,
        measured from ``event``. A nested RBD's gives the time *of* its next
        state change: its simulation runs on the same clock, from 0.
        """
        if isinstance(self.components[event.component], RepairableRBD):
            return t
        return event.time + t

    def next_event(self, method="p", sources: Optional[dict] = None):
        """Advance the current simulation to the system's next state change.

        Advanced, event-stepping API, after ``initialize_event_queue``. It
        takes the queued component events in time order, updating each
        component's state and queueing its next event (if before
        ``t_simulation``), until the system changes state, and returns
        that change. It gives a ``RepairableRBD`` the same event API as a
        ``NonRepairable``, so it can be a node of another RBD. Unlike
        ``NonRepairable.next_event``, which returns the time *to* the
        component's next event, this returns the time *of* the change, on
        the simulation's clock.

        When no further change happens before ``t_simulation``, it returns
        ``(t_simulation, current state)`` and discards the queue, so the
        next call raises a ValueError until ``initialize_event_queue`` is
        called again.

        Parameters
        ----------
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); both give the
            same state.
        sources : dict, optional
            Internal: the same ``sources`` given to
            ``initialize_event_queue``. By default None: the components
            themselves.

        Returns
        -------
        tuple[float, bool]
            The time of the change, and the system's new state (True for
            working, False for failed).

        Raises
        ------
        ValueError
            If the event queue has not been initialised, or was used up by
            an earlier call; or if ``method`` is not ``"p"`` or ``"c"``.

        Examples
        --------
        A component that fails exactly 10 hours after each repair, and
        takes exactly 2 hours to repair:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.ExactEventTime.from_params(10),
        ...             "repairability": surv.ExactEventTime.from_params(2),
        ...         }
        ...     },
        ... )
        >>> rbd.initialize_event_queue(30.0)
        >>> changes = [rbd.next_event()]
        >>> while changes[-1][0] < 30.0:
        ...     changes.append(rbd.next_event())
        >>> changes[:4]
        [(10.0, False), (12.0, True), (22.0, False), (24.0, True)]

        The next failure, at 34, is outside the window, so the last call
        returns the end of the window and the state the system is left in:

        >>> changes[4]
        (30.0, True)
        """
        if not hasattr(self, "_event_queue"):
            raise ValueError("Need to initialize the event queue")
        # The components' draws come from the same sources the queue was
        # initialised with (see initialize_event_queue).
        sources = self.components if sources is None else sources
        new_system_state = copy(self.system_state)

        # Use a while loop to find the next time/event at which the system
        # status changes.
        while new_system_state == self.system_state:
            if self._event_queue.qsize() == 0:
                del self._event_queue
                return self.t_simulation, self.system_state

            event = self._event_queue.get()
            self.component_status[event.component] = event.status
            new_system_state = self.is_system_working(
                self.component_status, method
            )

            next_event_t, next_event_type = sources[
                event.component
            ].next_event()
            next_event = Event(
                self._next_event_time(event, next_event_t),
                event.component,
                next_event_type,
            )
            # But only queue up the event if it occurs before the end
            # of the simulation
            if next_event.time < self.t_simulation:
                self._event_queue.put(next_event)

        self.system_state = new_system_state

        return event.time, self.system_state

    def availability(
        self,
        t_simulation: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        N: int = 10_000,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> AvailabilityResult:
        """Simulate the system's availability over ``[0, t_simulation]``.

        Runs ``N`` independent Monte-Carlo (discrete-event) simulations of
        the system from time 0, each starting with every component working
        (except ``broken_nodes``). Each component alternates failure and
        repair independently of the others and of the system state: it
        fails after a time drawn from its reliability model and is
        restored, as good as new, after a time drawn from its repairability
        model. A nested ``RepairableRBD`` runs its own simulation on the
        same clock, and is down while its system is down. The system state
        is re-evaluated at every component event.

        The result holds the estimated availability over time, the system's
        and every node's total up and down time, the number of system
        failures and restorations, and criticality measures that attribute
        the system's up and down time, failures and restorations to the
        nodes (see [`Criticalities`][repyability.Criticalities]). When
        something is priced (see ``has_costs``), the same simulations also
        accumulate costs, and the result's ``cost`` is a
        [`CostResult`][repyability.CostResult]; otherwise it is None. For
        exact long-run values, use ``mean_availability`` and the other
        steady-state methods.

        Parameters
        ----------
        t_simulation : float
            Length of the window each simulation covers, in the time unit
            of the component models. Events at or after it are not
            simulated.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window: they never fail. By
            default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window: they are down from time
            0 and never repaired. By default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); the results are
            identical.
        N : int, optional
            Number of simulations, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None. The simulations
            draw from numpy's global RNG (as surpyval's ``random`` does).
            With a seed, that RNG is seeded for the run and the caller's
            RNG state is restored afterwards. With None, the run draws from
            (and advances) the global RNG as it stands, so seeding it with
            ``np.random.seed`` beforehand also makes the run reproducible.
            Models that do not draw from the global RNG, such as surpyval's
            non-parametric ones, are not reproducible either way. Costs
            given as distributions draw from a generator of their own,
            seeded from the global RNG without consuming it, so pricing
            never changes the simulated failures and repairs.

        Returns
        -------
        AvailabilityResult
            The availability over time (``timeline``, ``availability``),
            the up and down totals summed over the ``N`` simulations, the
            system failure and restoration counts, the ``criticalities``
            and the ``cost``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets, or if ``method`` is not ``"p"`` or
            ``"c"``.

        Examples
        --------
        Two components in series, each with MTTF 10 and MTTR 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> unit = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")], {"a": unit, "b": unit}
        ... )
        >>> result = rbd.availability(t_simulation=50, N=200, seed=0)
        >>> float(result.timeline[0]), float(result.availability[0])
        (0.0, 1.0)
        >>> float(result.timeline[-1])
        50.0

        The fraction of the window the system was up is close to the
        long-run availability, ``(10 / 11) ** 2`` (over a short window it is
        a little above it on average, as every simulation starts up):

        >>> window = result.n_simulations * result.time_simulated_to
        >>> round(float(result.system_uptime) / window, 4)
        0.8272
        >>> round(rbd.mean_availability(), 4)
        0.8264

        In series the system is up only while every node is up:

        >>> oci = result.criticalities.operational_criticality_index
        >>> {node: round(float(v), 4) for node, v in oci.up.items()}
        {'a': 1.0, 'b': 1.0}
        """
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)

        # aggregate_timeline keeps track of how many of the simulated systems
        # turn on and off at time t.
        # e.g. aggregate_timeline[t] = +2 means two out of the N simulated
        # systems began working again at time t, while
        # aggregate_timeline[t] = -1 means one out of the N simulated systems
        # stopped working at time t.
        # There is a very strong expectation that due to the random sampling
        # that the generall aggregate_timeline[t] would be only -1 or +1.
        aggregate_timeline: dict[float, int] = defaultdict(lambda: 0)
        # The below two assigments ensure the results have data at 0 and time
        # t_simulation regardless of whether it was sampled at these times.
        # Set the end of the timeline to be 0 (i.e. unchanged if no event
        # falls) exactly at time t_simulation.
        aggregate_timeline[t_simulation] = 0
        # The initial system state is the same for every simulation (the forced
        # working/broken sets are fixed): all components start working except
        # those forced broken, which can make the system start down (e.g. a
        # broken component in series). Seed time 0 accordingly.
        initial_status = {c: c not in broken_nodes for c in self.components}
        initial_system_up = bool(
            self.is_system_working(initial_status, method)
        )
        aggregate_timeline[0] = N if initial_system_up else 0

        # Restoration Criticality Index
        RCI: defaultdict = defaultdict(lambda: defaultdict(lambda: 0))
        system_restorations = 0
        system_downtime = 0

        # Failure Criticality Index
        FCI: defaultdict = defaultdict(lambda: defaultdict(lambda: 0))
        system_failures = 0
        system_uptime = 0

        node_downtime: defaultdict = defaultdict(lambda: 0)
        node_uptime: defaultdict = defaultdict(lambda: 0)
        intersection_uptime: defaultdict = defaultdict(lambda: 0)
        intersection_downtime: defaultdict = defaultdict(lambda: 0)
        union_uptime: defaultdict = defaultdict(lambda: 0)
        union_downtime: defaultdict = defaultdict(lambda: 0)

        # Cost accumulation rides along on the same replications, but only
        # when some cost has been declared -- an unpriced RBD does none of
        # this work. Each replication yields one total-cost sample (the
        # distribution); the running totals produce the mean breakdowns.
        has_costs = self.has_costs
        downtime_cost_rates = {
            node: c["downtime_cost"]
            for node, c in self.costs.items()
            if "downtime_cost" in c
        }
        cost_samples: List[float] = []
        cost_by_category = {
            "repair": 0.0,
            "replace": 0.0,
            "component_downtime": 0.0,
            "system_downtime": 0.0,
        }
        cost_by_component = {node: 0.0 for node in self.costs}

        # Perform N simulations. surpyval's ``.random`` draws from numpy's
        # global RNG, so seed it here (if a seed was given) to make the run
        # reproducible, restoring the caller's RNG state once the randomised
        # simulations below have finished.
        _rng_state = None
        if seed is not None:
            _rng_state = np.random.get_state()
            np.random.seed(seed)
        failure_charges = self._failure_charges() if has_costs else {}
        stream = UniformStream()
        sources = self._streamed_components(stream) or self.components

        for _ in tqdm(
            range(N), disable=not verbose, desc="Running simulations"
        ):
            # Initialize the event queue and the system/component statuses
            self.initialize_event_queue(
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                sources,
            )

            # Seed each timeline from the actual initial status so that
            # forced-broken components (and a system they start down) are
            # accounted as down from t=0, not assumed up.
            component_timelines: dict = {
                comp: [(0.0, 1 if self.component_status[comp] else 0)]
                for comp in self.components
            }
            system_timeline = [(0.0, 1 if self.system_state else 0)]
            rep_cost = 0.0

            # Implemented ensure that no events that occur after the
            # end-time of the simulation are added to the queue; so we just
            # need to keep going through the queue until it's empty
            while not self._event_queue.empty():
                # Get the next event and update the component's status

                event = self._event_queue.get()
                # Update the component's status
                self.component_status[event.component] = event.status
                if event.status:
                    RCI[event.component]["component_restorations"] += 1
                else:
                    FCI[event.component]["component_failures"] += 1
                    # Repair and replace are charged per corrective action,
                    # at the failure that triggers it.
                    for category, charges in failure_charges.get(
                        event.component, ()
                    ):
                        charge = next(charges)
                        rep_cost += charge
                        cost_by_category[category] += charge
                        cost_by_component[event.component] += charge

                status = 1 if event.status else -1
                component_timelines[event.component].append(
                    (event.time, status)
                )

                # Record new system state, it could still be the same as
                # system_state in which case we don't bother changing
                # aggregate_timeline, but if it is different, we need to +/-1
                # to aggregate_timeline if the system has gone on/off-line
                new_system_state = self.is_system_working(
                    self.component_status, method
                )
                if new_system_state != self.system_state:
                    status = 1 if new_system_state else -1
                    system_timeline.append((event.time, status))
                    if new_system_state:
                        # System restored
                        aggregate_timeline[event.time] += 1
                        system_restorations += 1
                        RCI[event.component]["system_restorations"] += 1
                    else:
                        aggregate_timeline[event.time] -= 1
                        system_failures += 1
                        FCI[event.component]["system_failures"] += 1

                    # Set the system_state to the new state
                    self.system_state = new_system_state

                # Now we need to get the component's next event
                # If the component just got repaired then we need it's next
                # failure event, otherwise it just broke and we need it's
                # repair event
                next_event_t, next_event_type = sources[
                    event.component
                ].next_event()

                next_event = Event(
                    self._next_event_time(event, next_event_t),
                    event.component,
                    next_event_type,
                )
                # But only queue up the event if it occurs before the end
                # of the simulation
                if next_event.time < t_simulation:
                    self._event_queue.put(next_event)

                # Then move on to the next event... until there's no more
                # events in the queue

            system_timeline.append((t_simulation, 0))

            for component in self.components.keys():
                component_timelines[component].append((t_simulation, 0))
                # This simulation's uptime for the component; the downtime is
                # the remainder of the window. (Use the per-simulation value,
                # not the running cumulative node_uptime[component].)
                component_ut = time_at_status(
                    component_timelines[component], 1
                )
                node_uptime[component] += component_ut
                node_downtime[component] += t_simulation - component_ut
                if component in downtime_cost_rates:
                    charge = downtime_cost_rates[component] * (
                        t_simulation - component_ut
                    )
                    rep_cost += charge
                    cost_by_category["component_downtime"] += charge
                    cost_by_component[component] += charge
                joint_t, joint_events = combined_timeline(
                    component_timelines[component], system_timeline
                )
                intersection_uptime[component] += intersection(
                    joint_t, joint_events
                )
                intersection_downtime[component] += intersection(
                    joint_t, 2 - joint_events
                )
                union_uptime[component] += union(joint_t, joint_events)
                union_downtime[component] += union(joint_t, 2 - joint_events)

            simulation_system_ut = time_at_status(system_timeline, 1)
            system_uptime += simulation_system_ut
            system_downtime += t_simulation - simulation_system_ut

            if has_costs:
                charge = self.downtime_cost_rate * (
                    t_simulation - simulation_system_ut
                )
                rep_cost += charge
                cost_by_category["system_downtime"] += charge
                cost_samples.append(rep_cost)

        # Randomised simulations are done. Leave the global RNG where the
        # draw-at-a-time simulation would have, then restore the caller's
        # state if a seed was given. (If a simulation raises, neither
        # happens: the RNG is left up to a block of uniforms further on.)
        stream.close()
        if _rng_state is not None:
            np.random.set_state(_rng_state)

        # Collect Importance/Criticality measures from the simulation
        # reference: https://www.weibull.com/pubs/2004rm_05B_02.pdf
        # Operational Criticality Index
        oci_down = {
            k: _safe_ratio(v, system_downtime)
            for k, v in dict(intersection_downtime).items()
        }
        oci_up = {
            k: _safe_ratio(v, system_uptime)
            for k, v in dict(intersection_uptime).items()
        }
        # Intersection Over Union Importance
        iou_up = {
            k: _safe_ratio(intersection_uptime[k], union_uptime[k])
            for k in dict(intersection_uptime).keys()
        }
        iou_down = {
            k: _safe_ratio(intersection_downtime[k], union_downtime[k])
            for k in dict(intersection_downtime).keys()
        }
        # Failure Criticality Index Importance
        fci_sys = failure_criticality_index_per_system_failures(
            FCI, system_failures
        )
        fci_comp = failure_criticality_index_per_component_failures(FCI)
        # Restoration Criticality Index Importance
        rci_sys = restoration_criticality_index_by_system(
            RCI, system_restorations
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

        # Now we need to return the system availability from t=0..t_simulation
        # Using numpy arrays for efficiency
        timeline_arr: np.ndarray = np.array(list(aggregate_timeline.items()))

        # Sort the array by event time
        timeline_arr = timeline_arr[timeline_arr[:, 0].argsort()]
        time = timeline_arr[:, 0]

        # Take the cumulative sum, this is basically calculating for each
        # t just how many systems are working, and divide by N to get
        # availability the as a percentage
        system_availability = timeline_arr[:, 1].cumsum() / N

        # Clean up the interim variables of the simulation
        del self._event_queue
        del self.system_state
        del self.t_simulation
        del self.component_status

        cost_result = None
        if has_costs:
            # Per-component means cover each node's repair, replace and own
            # downtime cost. (System downtime is a system-level quantity and
            # is not attributed to components.)
            cost_result = CostResult(
                samples=np.asarray(cost_samples, dtype=float),
                t_simulation=t_simulation,
                n_simulations=N,
                by_category={
                    k: float(v) / N for k, v in cost_by_category.items()
                },
                by_component={
                    k: float(v) / N for k, v in cost_by_component.items()
                },
            )

        simulation_results = AvailabilityResult(
            timeline=time,
            availability=system_availability,
            system_uptime=system_uptime,
            time_simulated_to=t_simulation,
            criticalities=criticalities,
            node_uptime=dict(node_uptime),
            node_downtime=dict(node_downtime),
            system_downtime=system_downtime,
            system_failures=system_failures,
            system_restorations=system_restorations,
            n_simulations=N,
            cost=cost_result,
        )

        return simulation_results

    def cost(
        self,
        t_simulation: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        N: int = 10_000,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> Optional[CostResult]:
        """Simulate the cost of running the system for ``t_simulation``.

        Runs the ``availability`` simulation with cost accumulation and
        returns its [`CostResult`][repyability.CostResult]: the distribution
        of the window's total cost (``samples``, ``mean``,
        ``percentile(q)``), a confidence interval for the mean
        (``mean_interval()``), and per-category and per-component
        breakdowns. In each simulation ``repair_cost`` and ``replace_cost``
        are charged at every failure of their component (a fresh draw for a
        cost distribution), ``downtime_cost`` per unit time its component is
        down, and ``downtime_cost_rate`` per unit time the system is down.
        Only this RBD's own costs count: costs declared inside a nested
        ``RepairableRBD`` are left out. ``result.cost_rate`` converges to
        the exact ``expected_cost_rate`` as the window grows, which is the
        cross-check to reach for.

        Returns None when nothing is priced (see ``has_costs``): there is
        no cost model to evaluate. (The same result is available as
        ``availability(...).cost`` if you also want the availability outputs
        from the same simulations.)

        Parameters
        ----------
        t_simulation : float
            Length of the window each simulation covers, in the time unit
            of the component models.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window, by default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); the results are
            identical.
        N : int, optional
            Number of simulations, each giving one sample of the window's
            total cost, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None. As in
            ``availability``: the global RNG is seeded for the run and the
            caller's state restored afterwards.

        Returns
        -------
        CostResult or None
            The simulated costs, or None if nothing is priced.

        Raises
        ------
        ValueError
            As for ``availability``, when something is priced. (With nothing
            priced this returns None without any checks.)

        Examples
        --------
        One component with MTTF 10 and MTTR 1, at 100 per repair, and 50 per
        hour of system downtime:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...             "repair_cost": 100.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=50.0,
        ... )
        >>> result = rbd.cost(t_simulation=100.0, N=200, seed=0)
        >>> round(result.mean, 2)  # mean cost of a 100-hour window
        1351.33
        >>> round(result.percentile(90), 2)  # 9 windows in 10 cost less
        1915.24
        >>> round(result.cost_rate, 2), round(rbd.expected_cost_rate(), 2)
        (13.51, 13.64)
        """
        if not self.has_costs:
            return None
        return self.availability(
            t_simulation,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            N=N,
            verbose=verbose,
            seed=seed,
        ).cost

    def node_availability(self) -> dict[Hashable, float]:
        """Returns each node's long-run (steady-state) availability.

        Exact: ``MTTF / (MTTF + MTTR)`` for each component (1.0 if it is
        repaired instantly), a nested ``RepairableRBD``'s own
        ``mean_availability``, and 1.0 for the input and output nodes. These
        are the availabilities the importance measures are evaluated at.

        Returns
        -------
        dict[Hashable, float]
            Node name -> long-run availability, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If a component has a non-parametric reliability model, whose
            MTTF this cannot compute.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": "instant"},
        ...     },
        ... )
        >>> availability = rbd.node_availability()
        >>> {node: round(a, 4) for node, a in availability.items()}
        {'a': 0.8333, 'b': 1.0, 's': 1.0, 't': 1.0}
        """
        node_av: dict[Hashable, float] = {}
        for node_name, component in self.components.items():
            node_av[node_name] = float(
                np.atleast_1d(component.mean_availability())[0]
            )

        for node_name in self.in_or_out:
            node_av[node_name] = 1.0

        return node_av

    def _node_failure_frequency(self, component) -> float:
        """A component's long-run failure frequency (failures per unit time):
        recursive for nested RepairableRBDs, ``1 / (MTTF + MTTR)`` for
        NonRepairable components."""
        if isinstance(component, RepairableRBD):
            return component.system_failure_frequency()
        return component.failure_frequency()

    def system_failure_frequency(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run failure frequency (failures per unit
        time), by the Birnbaum/Vesely formula.

        In steady state the system failure frequency is

        ```text
        omega_sys = sum_i I_B(i) * omega_i
        ```

        where ``I_B(i)`` is node i's Birnbaum importance evaluated at the
        nodes' long-run availabilities (see ``birnbaum_importance``) and
        ``omega_i`` is node i's long-run failure frequency: ``1 / (MTTF_i +
        MTTR_i)`` for a component, and a nested ``RepairableRBD``'s own
        ``system_failure_frequency``. Exact for independent repairable
        nodes, with no simulation. Every system failure counts, including
        the zero-length outages an instantly repaired component causes.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (they contribute no
            failures), by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (they contribute no
            failures), by default None.

        Returns
        -------
        float
            Expected number of system failures per unit time, in the long
            run.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        Two components in series, with MTTFs 5 and 2 and MTTRs of 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.system_failure_frequency(), 4)  # 2/3 * 1/6 + 5/6 * 1/3
        0.3889
        >>> round(rbd.system_failure_frequency(working_nodes=["a"]), 4)
        0.3333
        """
        availability = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        forced = (set() if working_nodes is None else set(working_nodes)) | (
            set() if broken_nodes is None else set(broken_nodes)
        )
        birnbaum = super()._birnbaum_importance(availability)
        omega = 0.0
        for node, component in self.components.items():
            if node in forced:
                # A forced node never changes state, so it contributes no
                # system failures.
                continue
            omega += np.asarray(birnbaum[node]).item() * (
                self._node_failure_frequency(component)
            )
        return omega

    def mean_time_between_failures(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Time Between Failures.

        Exact: ``MTBF = 1 / system_failure_frequency`` — the mean length of
        one full up-down cycle, i.e. ``MTBF = MUT + MDT``. Infinite if the
        system never fails (e.g. a node held working keeps it up, or one
        held broken keeps it down).

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean time between system failures, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        Two components in series, with MTTFs 5 and 2 and MTTRs of 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_up_time(), 4)  # 1 / (0.2 + 0.5)
        1.4286
        >>> round(rbd.mean_down_time(), 4)
        1.1429
        >>> round(rbd.mean_time_between_failures(), 4)  # MUT + MDT
        2.5714
        """
        omega = self.system_failure_frequency(working_nodes, broken_nodes)
        return 1.0 / omega if omega > 0.0 else float("inf")

    def mean_up_time(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Up Time.

        Exact: the mean duration of an uninterrupted working period of the
        system (sometimes called the repairable-system MTTF),
        ``MUT = mean_availability / system_failure_frequency``. If the
        system never fails it is infinite when the system is up and 0.0
        when it is always down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean up time, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        For exponential failures in series, MUT is one over the sum of the
        failure rates:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_up_time(), 4)  # 1 / (0.2 + 0.5)
        1.4286
        >>> round(rbd.mean_up_time(working_nodes=["a"]), 4)  # b alone
        2.0
        """
        availability = self.mean_availability(working_nodes, broken_nodes)
        omega = self.system_failure_frequency(working_nodes, broken_nodes)
        if omega > 0.0:
            return float(availability) / omega
        return float("inf") if availability > 0.0 else 0.0

    def mean_down_time(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Down Time.

        Exact: the mean duration of a system outage (the repairable-system
        MTTR), ``MDT = mean_unavailability / system_failure_frequency``. If
        the system never fails it is 0.0 when the system is always up and
        infinite when it is down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean down time, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        A parallel pair is down only while both are, and the outage ends
        when either is repaired:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_down_time(), 4)  # 1 / (1.0 + 1.0)
        0.5
        >>> rbd.mean_down_time(working_nodes=["a"])  # never down
        0.0
        """
        availability = self.mean_availability(working_nodes, broken_nodes)
        omega = self.system_failure_frequency(working_nodes, broken_nodes)
        if omega > 0.0:
            return (1.0 - float(availability)) / omega
        return 0.0 if availability >= 1.0 else float("inf")

    def birnbaum_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the Birnbaum measure of importance for all nodes,
        evaluated at the nodes' long-run availabilities.

        Exact, with no simulation: ``I_B(i) = A_sys(A_i = 1) -
        A_sys(A_i = 0)``, the system's long-run availability with node i
        always working minus that with node i always failed. It is the
        probability that node i is critical (the system works if and only
        if node i does), and how much the system availability changes per
        unit change in node i's availability. The node availabilities are
        those of ``node_availability``, with ``working_nodes`` and
        ``broken_nodes`` held at 1 and 0.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Birnbaum importances as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        In series, a node is critical exactly when the others work:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> importance = rbd.birnbaum_importance()
        >>> {node: round(i, 4) for node, i in importance.items()}
        {'a': 0.6667, 'b': 0.8333}
        >>> round(rbd.birnbaum_importance(working_nodes=["a"])["b"], 4)
        1.0
        """
        node_probabilities = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._birnbaum_importance(node_probabilities)
        )

    def improvement_potential(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the improvement potential of all nodes, evaluated at the
        nodes' long-run availabilities.

        Exact, with no simulation: ``A_sys(A_i = 1) - A_sys``, the gain in
        the system's long-run availability if node i never failed. The node
        availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and improvement potentials as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> potential = rbd.improvement_potential()
        >>> {node: round(p, 4) for node, p in potential.items()}
        {'a': 0.1111, 'b': 0.2778}
        """
        node_probabilities = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._improvement_potential(node_probabilities)
        )

    def risk_achievement_worth(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the RAW importance per Modarres & Kaminskiy, evaluated at
        the nodes' long-run availabilities. That is RAW_i =
        (unavailability of system given i failed) /
        (nominal system unavailability).

        Exact, with no simulation: ``RAW_i = U_sys(A_i = 0) / U_sys``,
        where ``U = 1 - A`` is long-run unavailability: the factor by which
        the system's unavailability grows if node i is always failed. The
        node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0. Where the
        nominal unavailability is 0 (e.g. every component is repaired
        instantly), the ratio is ``inf`` or ``nan``, with a numpy warning.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RAW importances as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        In series the system is down whenever a node is, so failing any
        node multiplies the unavailability by ``1 / U_sys``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> raw = rbd.risk_achievement_worth()
        >>> {node: round(r, 4) for node, r in raw.items()}
        {'a': 2.25, 'b': 2.25}
        """
        node_probabilities = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._risk_achievement_worth(node_probabilities)
        )

    def risk_reduction_worth(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the RRW importance per Modarres & Kaminskiy, evaluated at
        the nodes' long-run availabilities. That is RRW_i =
        (nominal unavailability of system) /
        (unavailability of system given i is working).

        Exact, with no simulation: ``RRW_i = U_sys / U_sys(A_i = 1)``,
        where ``U = 1 - A`` is long-run unavailability: the factor by which
        the system's unavailability would fall if node i never failed. The
        node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0. It is
        ``inf`` (with a numpy warning) where making node i perfect would
        make the system perfect, as for either node of a parallel pair, and
        ``nan`` if the system is already never down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RRW importances as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> rrw = rbd.risk_reduction_worth()
        >>> {node: round(r, 4) for node, r in rrw.items()}
        {'a': 1.3333, 'b': 2.6667}
        """
        node_probabilities = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._risk_reduction_worth(node_probabilities)
        )

    def criticality_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the criticality importance of all nodes, evaluated at the
        nodes' long-run availabilities.

        Exact, with no simulation: ``I_B(i) * A_i / A_sys``, the Birnbaum
        importance weighted by node i's availability over the system's: the
        probability that node i is working and critical, given the system
        is working. (This is the success form of the measure; the failure
        form, ``I_B(i) * (1 - A_i) / (1 - A_sys)``, is not what is
        returned.) The node availabilities are those of
        ``node_availability``, with ``working_nodes`` and ``broken_nodes``
        held at 1 and 0. If the system is never up, the ratio is ``nan``,
        with a numpy warning.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and criticality importances
            as values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.

        Examples
        --------
        In series every node is working and critical whenever the system
        works:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> criticality = rbd.criticality_importance()
        >>> {node: round(c, 4) for node, c in criticality.items()}
        {'a': 1.0, 'b': 1.0}
        """
        node_probabilities = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._criticality_importance(node_probabilities)
        )

    def fussell_vesely(
        self,
        fv_type: str = "c",
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Calculate Fussell-Vesely importance of all nodes, evaluated at the
        nodes' long-run availabilities.

        Briefly, the Fussell-Vesely importance measure for node i =
        (sum of probabilities of cut-sets including node i occurring, i.e.
        all their nodes failed) / (the probability of the system failing).
        Here a node's probability of having failed is its long-run
        unavailability, ``1 - A_i``, from ``node_availability`` (with
        ``working_nodes`` and ``broken_nodes`` held at availability 1 and
        0), and the system's is ``1 - A_sys``; both are exact, with no
        simulation. The sum over cut sets is the usual rare-event
        approximation of the probability that some cut set containing node
        i has occurred, so with large unavailabilities the measure can
        exceed 1. If the system never fails, the ratio is ``nan`` or
        ``inf``, with a numpy warning.

        Typically this measure is implemented using cut-sets as mentioned
        above, although it can be implemented using path-sets. Both are
        implemented here, selected by ``fv_type``: ``"c"`` sums over the
        minimal cut sets containing node i, ``"p"`` over the minimal path
        sets containing it. Either way each set contributes the probability
        that all of its nodes are failed, and the sum is divided by the
        system's unavailability.

        Parameters
        ----------
        fv_type : str, optional
            Dictates the method of calculation, "c" = cut-set and
            "p" = path-set, by default "c".
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set); if a
            working/broken node is unknown, is the input or output node, or
            is in both sets; or if a component has a non-parametric
            reliability model.

        Examples
        --------
        In series each node is a cut set on its own, so its importance is
        its unavailability over the system's:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> fv = rbd.fussell_vesely()
        >>> {node: round(v, 4) for node, v in fv.items()}
        {'a': 0.375, 'b': 0.75}
        """
        node_probabilities = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._fussell_vesely(node_probabilities, fv_type)
        )

    def fussel_vesely(self, fv_type: str = "c") -> dict[Any, float]:
        """Deprecated alias for ``fussell_vesely`` (corrected spelling).

        Deprecated: use ``fussell_vesely`` instead; this alias will be
        removed in a future release. It returns ``fussell_vesely(fv_type)``
        and, unlike it, takes no ``working_nodes`` or ``broken_nodes``.

        Parameters
        ----------
        fv_type : str, optional
            "c" = cut-set and "p" = path-set, by default "c".

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values.

        Raises
        ------
        ValueError
            As for ``fussell_vesely``.

        Warns
        -----
        DeprecationWarning
            On every call.
        """
        warnings.warn(
            "fussel_vesely() is deprecated; use fussell_vesely() "
            "(Fussell-Vesely). This alias will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fussell_vesely(fv_type)
