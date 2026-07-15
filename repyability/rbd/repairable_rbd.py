import warnings
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Collection, Hashable, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from repyability.non_repairable import NonRepairable
from repyability.rbd.rbd import RBD


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


def _safe_ratio(numerator, denominator):
    """Ratio that is 0 when the denominator is 0.

    Several criticality/importance measures divide by a total (system uptime,
    downtime, failures, restorations, or a union of intervals) that can
    legitimately be 0 -- e.g. when a redundant component is forced working so
    the system never fails, or a component is forced broken. With nothing to
    attribute, the measure is 0 rather than a NaN/inf (or a crash).
    """
    return numerator / denominator if denominator else 0.0


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
    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        components: dict[Any, Any],
        k: Optional[dict[Any, int]] = None,
    ):
        components = copy(components)
        reliability = {}
        repairability = {}
        for name, component in components.items():
            if isinstance(component, dict):
                components[name] = NonRepairable(
                    component["reliability"], component["repairability"]
                )
                reliability[name] = component["reliability"]
                repairability[name] = component["repairability"]
            elif isinstance(component, RepairableRBD):
                reliability[name] = component
                repairability[name] = None
            elif isinstance(component, NonRepairable):
                reliability[name] = component.reliability
                repairability[name] = component.time_to_replace

        super().__init__(edges, k)

        self.components = components
        self.repairability = copy(repairability)

    def initialize_event_queue(
        self,
        t_simulation,
        working_components: Optional[Collection[Hashable]] = None,
        broken_components: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ):
        working_components = (
            set() if working_components is None else set(working_components)
        )
        broken_components = (
            set() if broken_components is None else set(broken_components)
        )

        # Keep record of component status', initially they're all working
        component_status: dict[Any, bool] = {
            component: True for component in self.components.keys()
        }

        for component in broken_components:
            component_status[component] = False

        # PriorityQueue supplies failure/repair events in chronological
        event_queue: PriorityQueue[Event] = PriorityQueue()

        # For each component add in the initial failure
        for component_id in self.components.keys():
            component = self.components[component_id]
            if component_id in working_components:
                continue
            elif component_id in broken_components:
                continue
            # If status not known, then continue
            if isinstance(component, RepairableRBD):
                component.initialize_event_queue(t_simulation)
                t_event, event = component.next_event()
            elif isinstance(component, NonRepairable):
                component.reset()
                t_event, event = component.next_event()

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

    def mean_unavailability(self, *args, **kwargs) -> np.float64:
        """Returns the system long run UNavailability

        Parameters
        ----------
        *args, **kwargs :
            Any mean_availability() arguments

        Returns
        -------
        np.float64
            Long run unavailability of the system
        """
        return 1 - self.mean_availability(*args, **kwargs)

    def mean_availability(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> np.float64:
        """Returns the system long run availability

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Marks these components as perfectly reliable, by default []
        broken_nodes : Collection[Hashable], optional
            Marks these components as perfectly unreliable, by default []
        using: str, optional
            Input either "c" or "p" for the function to use cut sets or path
            sets respectively. Defaults to path sets.

        Returns
        -------
        np.float64
            Long run availability of the system

        Raises
        ------
        ValueError
        """
        # Good reference on the Availability of a system
        # https://www.diva-portal.org/smash/get/diva2:986067/FULLTEXT01.pdf
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)

        # Cache all component reliabilities for efficiency
        component_availability: dict[Hashable, np.float64] = {}
        for comp in self.components:
            if comp in working_nodes:
                component_availability[comp] = np.float64(1.0)
            elif comp in broken_nodes:
                component_availability[comp] = np.float64(0.0)
            else:
                component_availability[comp] = self.components[
                    comp
                ].mean_availability()

        for comp in self.in_or_out:
            component_availability[comp] = np.float64(1.0)

        mean_availability = self.system_probability(
            component_availability, method=method
        )
        return mean_availability.item()

    def next_event(self, method="p"):
        # This method allows a user to extract the next system status
        # changing event. The intent of this is so that it has the same api
        # as the NonRepairable class so that a RepairableRBD can be used in
        # a RepairableRBD/
        if not hasattr(self, "_event_queue"):
            raise ValueError("Need to initialize the event queue")
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

            next_event_t, next_event_type = self.components[
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
    ) -> dict:
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

        components_downtime: defaultdict = defaultdict(lambda: 0)
        components_uptime: defaultdict = defaultdict(lambda: 0)
        intersection_uptime: defaultdict = defaultdict(lambda: 0)
        intersection_downtime: defaultdict = defaultdict(lambda: 0)
        union_uptime: defaultdict = defaultdict(lambda: 0)
        union_downtime: defaultdict = defaultdict(lambda: 0)

        # Perform N simulations. surpyval's ``.random`` draws from numpy's
        # global RNG, so seed it here (if a seed was given) to make the run
        # reproducible, restoring the caller's RNG state once the randomised
        # simulations below have finished.
        _rng_state = None
        if seed is not None:
            _rng_state = np.random.get_state()
            np.random.seed(seed)

        for _ in tqdm(
            range(N), disable=not verbose, desc="Running simulations"
        ):
            # Initialize the event queue and the system/component statuses
            self.initialize_event_queue(
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
            )

            # Seed each timeline from the actual initial status so that
            # forced-broken components (and a system they start down) are
            # accounted as down from t=0, not assumed up.
            component_timelines: dict = {
                comp: [(0.0, 1 if self.component_status[comp] else 0)]
                for comp in self.components
            }
            system_timeline = [(0.0, 1 if self.system_state else 0)]

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
                next_event_t, next_event_type = self.components[
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
                # not the running cumulative components_uptime[component].)
                component_ut = time_at_status(
                    component_timelines[component], 1
                )
                components_uptime[component] += component_ut
                components_downtime[component] += t_simulation - component_ut
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

        # Randomised simulations are done; restore the caller's RNG state.
        if _rng_state is not None:
            np.random.set_state(_rng_state)

        # Collect Importance/Criticality measures from the simulation
        # reference: https://www.weibull.com/pubs/2004rm_05B_02.pdf
        criticalities = {}
        # Operational Criticality Index
        oci_down = {
            k: _safe_ratio(v, system_downtime)
            for k, v in dict(intersection_downtime).items()
        }
        oci_up = {
            k: _safe_ratio(v, system_uptime)
            for k, v in dict(intersection_uptime).items()
        }
        criticalities["operational_criticality_index"] = {
            "up": oci_up,
            "down": oci_down,
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
        criticalities["iou"] = {"up": iou_up, "down": iou_down}
        # Failure Criticality Index Importance
        fci_sys = failure_criticality_index_per_system_failures(
            FCI, system_failures
        )
        fci_comp = failure_criticality_index_per_component_failures(FCI)
        criticalities["failure_criticality_index"] = {
            "per_system_failure": fci_sys,
            "per_component_failure": fci_comp,
        }
        # Restoration Criticality Index Importance
        rci_sys = restoration_criticality_index_by_system(
            RCI, system_restorations
        )
        rci_comp = restoration_criticality_index_by_component(RCI)
        criticalities["restoration_criticality_index"] = {
            "by_system": rci_sys,
            "by_component": rci_comp,
        }

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

        simulation_results = {
            "timeline": time,
            "availability": system_availability,
            "system_uptime": system_uptime,
            "time_simulated_to": t_simulation,
            "criticalities": criticalities,
            "components_uptime": dict(components_uptime),
            "components_downtime": dict(components_downtime),
        }

        return simulation_results

    def node_availability(self):
        node_av: dict = {}
        for node_name, component in self.components.items():
            node_av[node_name] = component.mean_availability()

        for node_name in self.in_or_out:
            node_av[node_name] = np.float64(1.0)

        return node_av

    def birnbaum_importance(self) -> dict[Any, np.ndarray]:
        """Returns the Birnbaum measure of importance for all nodes.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent. If the RBD called on has two or more nodes associated
        with the same component then a UserWarning is raised.

        Returns
        -------
        dict[Any, np.ndarray]
            Dictionary with node names as keys and Birnbaum importances as
            values
        """
        node_probabilities = self.node_availability()
        return super()._birnbaum_importance(node_probabilities)

    def improvement_potential(self) -> dict[Any, np.ndarray]:
        """Returns the improvement potential of all nodes.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and improvement potentials as
            values
        """
        node_probabilities = self.node_availability()
        return super()._improvement_potential(node_probabilities)

    def risk_achievement_worth(self) -> dict[Any, np.ndarray]:
        """Returns the RAW importance per Modarres & Kaminskiy. That is RAW_i =
        (unreliability of system given i failed) /
        (nominal system unreliability).

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RAW importances as values
        """
        node_probabilities = self.node_availability()
        return super()._risk_achievement_worth(node_probabilities)

    def risk_reduction_worth(self) -> dict[Any, np.ndarray]:
        """Returns the RRW importance per Modarres & Kaminskiy. That is RRW_i =
        (nominal unreliability of system) /
        (unreliability of system given i is working).

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RRW importances as values
        """
        node_probabilities = self.node_availability()
        return super()._risk_reduction_worth(node_probabilities)

    def criticality_importance(self) -> dict[Any, np.ndarray]:
        """Returns the criticality importance of all nodes at time/s x.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and criticality importances as
            values
        """
        node_probabilities = self.node_availability()
        return super()._criticality_importance(node_probabilities)

    def fussell_vesely(self, fv_type: str = "c") -> dict[Any, np.ndarray]:
        """Calculate Fussell-Vesely importance of all components.

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

        Returns
        -------
        dict[Any, np.ndarray]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set).
        """
        node_probabilities = self.node_availability()
        return super()._fussell_vesely(node_probabilities, fv_type)

    def fussel_vesely(self, fv_type: str = "c") -> dict[Any, np.ndarray]:
        """Deprecated alias for :meth:`fussell_vesely` (corrected spelling)."""
        warnings.warn(
            "fussel_vesely() is deprecated; use fussell_vesely() "
            "(Fussell-Vesely). This alias will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fussell_vesely(fv_type)
