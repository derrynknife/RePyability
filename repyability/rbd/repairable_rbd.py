from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Hashable, Iterable

import numpy as np

from repyability.rbd.rbd import RBD


# Event class for simulation
@dataclass(order=True)
class Event:
    """Dataclass to hold an event's information. Comparisons are performed
    by time."""

    time: float
    node: Hashable = field(compare=False)
    status: int = field(compare=False)


class RepairableRBD(RBD):
    def __init__(
        self,
        nodes: dict[Any, Any],
        reliability: dict[Any, Any],
        repair: dict[Any, Any],
        edges: Iterable[tuple[Hashable, Hashable]],
        mc_samples: int = 10_000,
    ):
        super().__init__(nodes, reliability, edges, mc_samples)
        self.repair = copy(repair)

    def compile_bdd(self) -> dict:
        # Need to create the bdd from min path sets

        # Format for BDD
        # Find first node using:
        # - Most common node?
        # - Least common node?
        # - most "bisecting" node?...
        """
        bdd = {
            1: {1: 2, 0: False},
            2: {1: True, 0: False}
        }
        """

        # Need to set the first node of the BDD
        self.first_bdd_node = 0
        # Pretend output
        self.bdd = {0: True}
        return {}

    def is_system_working(self, status: dict[Any, int]) -> bool:
        """Returns a boolean as to whether the system is working given the
        status of the nodes given by `status`

        Parameters
        ----------
        status : dict[Any, int]
            Dictionary of node designator with an int value of it's status.
              0 for broken, 1 for working.

        Returns
        -------
        bool
            boolean of whether the system is functioning or not.
        """
        return self.is_working_from_node(self.first_bdd_node, status)

    def is_working_from_node(self, node, status):
        val = self.bdd[node][status[node]]
        if type(val) == bool:
            return val
        else:
            return self.is_working_from_node(val, status)

    def availability(self, T, N=10_000) -> tuple:

        agg_timeline: dict[float, int] = defaultdict(lambda: 0)
        for i in range(N):
            pq: PriorityQueue = PriorityQueue()
            t = 0
            node_status = {}
            # Set system condition at zero to working
            agg_timeline[0] += 1
            system_status = 1

            for k in self.reliability.keys():
                node_status[k] = 0
                node = Event(self.reliability[k].random(1).item(), k, 0)
                pq.put(node)

            while not pq.empty():
                event = pq.get()
                new_t = event.time

                if new_t > T:
                    # Add zero since nothing will have changed since last event
                    agg_timeline[T] += 0
                    break
                else:
                    # Replace with BDD solution commented out below
                    new_system_status = 1 - system_status
                    # Uncomment the below when working
                    # new_system_status = self.working(node_status)
                    if new_system_status != system_status:
                        if new_system_status == 0:
                            agg_timeline[new_t] -= 1
                        else:
                            agg_timeline[new_t] += 1

                        system_status = new_system_status

                    # set_node to new status
                    node_status[event.node] = event.status

                    if event.status == 0:
                        dist = self.repair[event.node]
                    elif event.status == 1:
                        dist = self.reliability[event.node]

                    # Create a new event an put it into the queue
                    new_event = Event(
                        new_t + dist.random(1).item(),
                        event.node,
                        1 - event.status,
                    )
                    pq.put(new_event)
                    # Update simulation time
                    t = new_t

        tl: np.ndarray = np.array(list(agg_timeline.items()))
        tl = tl[tl[:, 0].argsort()]
        tl[:, 1] = tl[:, 1].cumsum()
        tl[:, 1] = tl[:, 1] / tl[0, 1]

        x, a = tl[:, 0], tl[:, 1]

        return x, a
