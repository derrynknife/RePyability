from queue import PriorityQueue

import numpy as np
from surpyval import KaplanMeier


class StandbyModel:
    def __init__(self, reliabilities, k=1, n_sims=10_000):
        self.reliabilities = reliabilities
        self.k = k
        self.N = len(reliabilities)
        self.n_sims = n_sims

        if self.k == 1:
            # If k is only one for the standby node the
            # reliability can be estimated from the sum
            # of each of the components in the node.
            # i.e. it will fail after all of them fail.
            x_random = 0
            for model in self.reliabilities:
                x_random += model.random(n_sims)

        else:
            # If k are required to continue then the sim needs
            # a little more complexity. An individual run instance
            # can be simulated by getting failures for the first k
            # components. Then each next random failure is then
            # added to the current smallest of the k 'active'.
            x_random = np.zeros(n_sims)
            for i in range(n_sims):
                pq = PriorityQueue()
                # start k streams:
                for node in self.reliabilities[:k]:
                    pq.put(node.random(1).item())

                # Add the next event time to the lowest value in the queue
                for node in self.reliabilities[k:]:
                    next_t = node.random(1).item()
                    current_lowest = pq.get()
                    pq.put(current_lowest + next_t)

                x_random[i] = pq.get()

        # Finish by creating the approximation of the
        # standby arrangement
        self.model = KaplanMeier.fit(x_random)

    def sf(self, x):
        return self.model.sf(x)

    def ff(self, x):
        return self.model.ff(x)
