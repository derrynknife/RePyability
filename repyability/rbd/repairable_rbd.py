from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Hashable, Iterable

import numpy as np

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
        reliability: dict[Any, Any],
        repairability: dict[Any, Any],
        edges: Iterable[tuple[Hashable, Hashable]],
        k: dict[Any, int] = {},
        mc_samples: int = 10_000,
    ):
        check_rbd_node_args_complete(nodes, reliability, edges, repairability)
        super().__init__(nodes, reliability, edges, k, mc_samples)
        self.repairability = copy(repairability)

        # Will initialise on first self.availability() call
        self.min_cut_sets: set[frozenset] | None = None

    # def compile_bdd(self) -> dict:
    #     """Returns a BDD dict for quick ~O(log(n)) lookup of if the system is
    #     working given the working/broken nodes."""
    #     # Need to create the bdd from min path sets

    #     # Format for BDD
    #     # Find first node using:
    #     # - Most common node?
    #     # - Least common node?
    #     # - most "bisecting" node?...
    #     """
    #     bdd = {
    #         1: {1: 2, 0: False},
    #         2: {1: True, 0: False}
    #     }
    #     """

    #     # Need to set the first node of the BDD
    #     self.first_bdd_node = 0
    #     # Pretend output
    #     self.bdd = {0: True}
    #     return {}

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

        # If the minimal cut sets have not yet been calculated
        if self.min_cut_sets is None:
            self.min_cut_sets = super().get_min_cut_sets()

        # Make a set out of the failed components
        failed_components = {
            comp
            for comp, comp_state in component_status.items()
            if not comp_state
        }

        # See if current set of failed components is a superset of any minimal
        # cut-sets
        for cut_set in self.min_cut_sets:
            if failed_components >= cut_set:
                # failed_components is a superset of a minimal cut-set, thereby
                # causing the system to fail
                return False

        # If we've gotten to here then the system is still working
        return True

    def availability(
        self, t_simulation: float, N: int = 10_000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the times, and availability for those times, as numpy
        arrays."""

        # aggregate_timeline keeps track of how many systems turn on and off
        # at time t.
        # e.g. aggregate_timeline[t] = +2 means two out of the N systems began
        # working again at time t, while aggregate_timeline[t] = -1 means one
        # out of the N systems stopped working at time t.
        # One might expect that due to the majority of distributions being
        # CDFs that generally aggregate_timeline[t] would be only -1 or +1.
        aggregate_timeline: dict[float, int] = defaultdict(lambda: 0)

        # Perform N simulations
        for _ in range(N):
            # 'Turn on' this simulation's system at time t=0
            aggregate_timeline[0] += 1
            curr_system_state = True

            # Keep record of component status', initially they're all working
            component_status: dict[Any, bool] = {
                component: True for component in self.reliability.keys()
            }

            # PriorityQueue supplies failure/repair events in chronological
            # order
            event_queue: PriorityQueue[Event] = PriorityQueue()

            # Get the first failure event for each component
            for component in self.reliability.keys():
                t_event = self.reliability[component].random(1)[0]

                # Only consider it if it occurs within the simulation window
                if t_event < t_simulation:
                    # Event status is False => event is a component failure
                    event_queue.put(Event(t_event, component, False))

            # It is implemented such that no events that occur after the
            # end-time of the simulation are added to the queue, so we just
            # need to keep going through the queue until it's empty
            while not event_queue.empty():
                # Get the next event and update the component's status
                event = event_queue.get()
                component_status[event.component] = event.status

                # Record new system state, it could still be the same as
                # curr_system_state in which case we don't bother changing
                # aggregate_timeline, but if it is different, we need to +/-1
                # if the system has gone on/off-line
                new_system_state = self.is_system_working(component_status)
                if new_system_state != curr_system_state:
                    if new_system_state:
                        aggregate_timeline[event.time] += 1
                    else:
                        aggregate_timeline[event.time] -= 1

                # Set the curr_system_state
                curr_system_state = new_system_state

                # Now we need to get the component's next event
                # If the component just got repaired then we need it's next
                # failure event, otherwise it just broke and we need it's
                # repair event
                if event.status:
                    # Component just got repaired, need next failure event
                    next_event = Event(
                        # Current time (event.time) + time to next failure
                        event.time
                        + self.reliability[event.component].random(1)[0],
                        event.component,
                        False,  # This is a component failure event
                    )
                else:
                    # Component just failed, need its repair event
                    next_event = Event(
                        # Current time (event.time) + time to repair
                        event.time
                        + self.repairability[event.component].random(1)[0],
                        event.component,
                        True,  # This is a component repair event
                    )

                # But only queue up the event if it occurs before the end
                # of the simulation
                if next_event.time < t_simulation:
                    event_queue.put(next_event)

                # Then move on to the next event... until there's no more
                # events before t_simulation

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

        return time, system_availability
