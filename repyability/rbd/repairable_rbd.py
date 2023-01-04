from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Hashable, Iterable

import numpy as np
from dd import autoref as _bdd
from tqdm import tqdm

from repyability.non_repairable import NonRepairable
from repyability.rbd.rbd import RBD
from repyability.rbd.rbd_args_check import check_rbd_node_args_complete


# Event class for simulation
@dataclass(order=True)
class Event:
    """Dataclass to hold an event's information. Comparisons are performed
    by time. status=False means the event is a component failure."""

    time: float
    component: Hashable = field(compare=False)
    status: bool = field(compare=False)


class RepairableRBD(RBD):
    def __init__(
        self,
        nodes: dict[Any, Any],
        components: dict[Any, Any],
        edges: Iterable[tuple[Hashable, Hashable]],
        k: dict[Any, int] = {},
        mc_samples: int = 10_000,
    ):
        components = copy(components)
        # reliability: dict[Any, Any]
        # repairability: dict[Any, Any]

        # super().__init__(nodes, reliability, edges, k, mc_samples)

        reliability = {}
        repairability = {}
        for node, node_type in nodes.items():
            if (node_type != "input_node") and (node_type != "output_node"):
                component = components[node]
                if isinstance(component, dict):
                    components[node] = NonRepairable(
                        component["reliability"], component["repairability"]
                    )
                    reliability[node] = component["reliability"]
                    repairability[node] = component["repairability"]
                elif isinstance(component, RepairableRBD):
                    reliability[node] = component
                    repairability[node] = None

        check_rbd_node_args_complete(nodes, reliability, edges, repairability)
        super().__init__(nodes, reliability, edges, k, mc_samples)

        self.components = components
        self.repairability = copy(repairability)

        # Compile BDD, sets self.bdd and self.bdd_system_ref. See compile_bdd()
        # docstring for more info.
        self.compile_bdd()

    def is_system_working(self, component_status: dict[Any, bool]) -> bool:
        """Returns a boolean as to whether the system is working given the
        status of the components

        Parameters
        ----------
        component_status : dict[Any, bool]
            Dictionary with all components where
            component_status[component] = True only if the component is
            working, and = False if not working.

        Returns
        -------
        bool
            True if the system is working, otherwise False.
        """
        # This function uses a Binary Decision Diagram (BDD) to enable
        # efficient component_status -> is_system_working lookup. Something
        # very much required given the rate at which this function is called
        # in the N simulations in availability().

        # Now evaluate the BDD given the component status'
        system_given_component_status = self.bdd.let(
            component_status, self.bdd_system_ref
        )

        system_state = self.bdd.to_expr(system_given_component_status)

        return system_state == "TRUE"

    def initialize_event_queue(self, t_simulation):
        # Keep record of component status', initially they're all working
        component_status: dict[Any, bool] = {
            component: True for component in self.components.keys()
        }

        # PriorityQueue supplies failure/repair events in chronological
        event_queue: PriorityQueue[Event] = PriorityQueue()

        # For each component add in the initial failure
        for component_id in self.components.keys():
            component = self.components[component_id]
            if isinstance(component, RepairableRBD):
                component.initialize_event_queue(t_simulation)
            t_event, event = component.next_event()

            # Only consider it if it occurs within the simulation window
            if t_event < t_simulation:
                # Event status is False => event is a component failure
                event_queue.put(Event(t_event, component_id, event))
        self._event_queue = event_queue
        self.system_state = True
        self.t_simulation = t_simulation
        self.component_status = component_status

    def next_event(self):
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
            new_system_state = self.is_system_working(self.component_status)

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
        self, t_simulation: float, N: int = 10_000, verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
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
        # Set all systems working at time 0
        aggregate_timeline[0] = N

        # Perform N simulations
        for _ in tqdm(
            range(N), disable=not verbose, desc="Running simulations"
        ):
            # Initialize the event queue and the system/component statuses
            self.initialize_event_queue(t_simulation)

            # Implemented ensure that no events that occur after the
            # end-time of the simulation are added to the queue; so we just
            # need to keep going through the queue until it's empty
            while not self._event_queue.empty():
                # Get the next event and update the component's status
                event = self._event_queue.get()
                # Update the component's status
                self.component_status[event.component] = event.status

                # Record new system state, it could still be the same as
                # system_state in which case we don't bother changing
                # aggregate_timeline, but if it is different, we need to +/-1
                # to aggregate_timeline if the system has gone on/off-line
                new_system_state = self.is_system_working(
                    self.component_status
                )
                if new_system_state != self.system_state:
                    if new_system_state:
                        aggregate_timeline[event.time] += 1
                    else:
                        aggregate_timeline[event.time] -= 1

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

        # Clean up the interim results of the simulation
        del self._event_queue
        del self.system_state
        del self.t_simulation
        del self.component_status

        return time, system_availability

    def compile_bdd(self):
        """
        Compiles the BDD, setting self.bdd to the BDD manager, and
        self.bdd_system_ref to the BDD variable reference for the system state.
        """
        # Instantiate the manager
        bdd = _bdd.BDD()

        # Enable Dynamic Reordering (Rudell's Sifting Algorithm)
        bdd.configure(reordering=True)

        # For all components, declare the BDD variable,
        # and get the BDD variable reference
        bdd_vars = {}
        for component in self.get_component_names():
            bdd.declare(component)
            bdd_vars[component] = bdd.var(component)

        # Make path expressions
        path_expressions: list[_bdd.Function] = []
        for min_path_set in self.get_min_path_sets(include_in_out_nodes=False):
            min_path_set_as_list = list(min_path_set)
            path_function = bdd_vars[min_path_set_as_list[0]]
            for component in min_path_set_as_list[1:]:
                path_function &= bdd_vars[component]
            path_expressions.append(path_function)

        system: _bdd.Function = path_expressions[0]
        for path_expression in path_expressions[1:]:
            system |= path_expression

        self.bdd = bdd
        self.bdd_system_ref = system

    # Debugging Functions - these help to understand the BDD structure
    def bdd_to_string(self, filename: str) -> str:
        """Returns the BDD as a string. Simply wraps bdd.to_expr()"""
        return self.bdd.to_expr(self.bdd_system_ref)

    def bdd_to_file(self, filename: str):
        """
        Wraps bdd.dump(). The file type is inferred from the extension
        (case insensitive).

        Note: bdd.dump() depends on the python library `pydot` being installed
        (via pip) and the graphviz' `dot` program being in your PATH.
        Install pydot just with pip: `pip install pydot`,
        and install graphviz, at least on Mac, with `brew install graphviz`.
        Installing with brew will at least make sure `dot` is added to your
        PATH.

        Supported extensions:
        '.p' for Pickle
        '.pdf' for PDF
        '.png' for PNG
        '.svg' for SVG
        '.json' for JSON
        """
        self.bdd.dump(filename, roots=[self.bdd_system_ref])
