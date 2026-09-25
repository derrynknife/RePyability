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
    ``None`` if its draws cannot be streamed."""
    if isinstance(component, RepairableRBD):
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
    """Dataclass to hold an event's information. Comparisons are performed
    by time. status=False means the event is a component failure."""

    time: float
    component: Hashable = field(compare=False)
    status: bool = field(compare=False)


def combined_timeline(
    timeline_1: List[Tuple[float, int]], timeline_2: List[Tuple[float, int]]
):
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
    from_idx = np.where(event_cumsum[:-1] == 2)[0]
    to_idx = from_idx + 1
    intersection = timeline[to_idx] - timeline[from_idx]
    return intersection.sum()


def union(timeline, event_cumsum):
    from_idx = np.where(event_cumsum[:-1] > 0)[0]
    to_idx = from_idx + 1
    union = timeline[to_idx] - timeline[from_idx]
    return union.sum()


def intersection_over_union(
    node_timeline: List[Tuple[float, int]],
    system_timeline: List[Tuple[float, int]],
):
    timeline, event_cumsum = combined_timeline(node_timeline, system_timeline)
    return intersection(timeline, event_cumsum) / union(timeline, event_cumsum)


def time_at_status(timeline, status):
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

        When nothing is priced there is no cost model to evaluate, so the cost
        methods short-circuit rather than doing the work.
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

        A component object used for several nodes, at any level, has one
        event state, so it gets one stand-in (``made``, by object).
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

        .. math::
            E[\\text{cost rate}] =
              c_{\\text{sys}} \\, (1 - A_{\\text{sys}})
              + \\sum_i \\omega_i \\, (c^{\\text{repair}}_i
                                      + c^{\\text{replace}}_i)
              + \\sum_i (1 - A_i) \\, c^{\\text{down}}_i

        where :math:`A` is availability and :math:`\\omega_i =
        1/(MTTF_i + MTTR_i)` is node i's long-run failure frequency — so
        ``repair_cost``/``replace_cost`` are charged per corrective action,
        while the downtime rates are charged per unit time down. A
        per-failure cost given as a distribution enters through its mean.

        Every cost is optional and defaults to 0, so any subset can be
        priced; with nothing priced this is 0. Costs are corrective-only and
        undiscounted.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working: they never fail, so they
            incur no corrective or downtime cost, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed: they incur no corrective
            cost (they never change state) but are down for all time, by
            default None

        Returns
        -------
        float
            Expected cost per unit time

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
        """Returns the system long run UNavailability

        Parameters
        ----------
        *args, **kwargs :
            Any mean_availability() arguments

        Returns
        -------
        float
            Long run unavailability of the system
        """
        return 1 - self.mean_availability(*args, **kwargs)

    def mean_availability(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> float:
        """Returns the system long run availability

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Marks these nodes as always available, by default None
        broken_nodes : Collection[Hashable], optional
            Marks these nodes as failed, by default None
        method : str, optional
            Input either "c" or "p" for the function to use cut sets or path
            sets respectively. Defaults to path sets.

        Returns
        -------
        float
            Long run availability of the system

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input/output node, or
            is in both sets.

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

    def next_event(self, method="p", sources: Optional[dict] = None):
        # This method allows a user to extract the next system status
        # changing event. The intent of this is so that it has the same api
        # as the NonRepairable class so that a RepairableRBD can be used in
        # a RepairableRBD/
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
                # Current time (event.time) + time to next failure
                event.time + next_event_t,
                event.component,
                next_event_type,  # This is a component failure event
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
        """Returns the times, and availability for those times, as numpy
        arrays

        Parameters
        ----------
        t_simulation : float
            Units of time to run each simulation for
        N : int, optional
            Number of simulations, by default 10_000
        verbose : bool, optional
            If True, displays progress bar of simulations, by default False

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            times, availabilities
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
                    # The next event time is the current time [event.time]
                    # plus the time to next event
                    event.time + next_event_t,
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

        Runs the availability simulation with cost accumulation and returns
        its :class:`CostResult` — the distribution of the window's total cost
        (``samples``, ``mean``, ``percentile(q)``), a confidence interval for
        the mean (``mean_interval()``), and per-category and per-component
        breakdowns. ``result.cost_rate`` converges to the exact
        :meth:`expected_cost_rate` as the window grows, which is the
        cross-check to reach for.

        Returns ``None`` when no cost has been declared anywhere — with
        nothing priced there is no cost model to evaluate. (The same result
        is available as ``availability(...).cost`` if you also want the
        availability outputs from the same replications.)
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
        """Returns each node's long-run availability (a dict keyed by node
        name); the input/output nodes are 1.0."""
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

        .. math::
            \\omega = \\sum_i I_B^i \\cdot \\omega_i

        where :math:`I_B^i` is node i's Birnbaum importance evaluated at the
        nodes' availabilities and :math:`\\omega_i = 1/(MTTF_i + MTTR_i)` is
        node i's failure frequency. Exact for independent repairable nodes.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (they contribute no
            failures), by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (they contribute no
            failures), by default None
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
        """Returns the system's long-run Mean Time Between Failures,
        ``MTBF = 1 / failure frequency`` — the mean length of one full
        up-down cycle, i.e. ``MTBF = MUT + MDT``. Infinite if the system
        never fails."""
        omega = self.system_failure_frequency(working_nodes, broken_nodes)
        return 1.0 / omega if omega > 0.0 else float("inf")

    def mean_up_time(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Up Time (mean duration of an
        uninterrupted working period; sometimes called the repairable-system
        MTTF), ``MUT = availability / failure frequency``."""
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
        """Returns the system's long-run Mean Down Time (mean duration of an
        outage; the repairable-system MTTR),
        ``MDT = unavailability / failure frequency``."""
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

        Note: Birnbaum's measure of importance assumes all nodes are
        independent.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Birnbaum importances as
            values
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

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and improvement potentials as
            values
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

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RAW importances as values
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

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RRW importances as values
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

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and criticality importances as
            values
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
        (sum of probabilities of cut-sets including node i occuring/failing) /
        (the probability of the system failing).

        Typically this measure is implemented using cut-sets as mentioned
        above, although the measure can be implemented using path-sets. Both
        are implemented here.

        fv_type dictates the method:
            "c" - cut-set
            "p" - path-set

        Parameters
        ----------
        fv_type : str, optional
            Dictates the method of calculation, 'c' = cut-set and
            'p' = path-set, by default "c"
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set).
        """
        node_probabilities = self._probabilities_with_overrides(
            self.node_availability(), working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._fussell_vesely(node_probabilities, fv_type)
        )

    def fussel_vesely(self, fv_type: str = "c") -> dict[Any, float]:
        """Deprecated alias for :meth:`fussell_vesely` (corrected spelling)."""
        warnings.warn(
            "fussel_vesely() is deprecated; use fussell_vesely() "
            "(Fussell-Vesely). This alias will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fussell_vesely(fv_type)
