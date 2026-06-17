from queue import PriorityQueue

import numpy as np
from surpyval import KaplanMeier

from .numerical_convolution import ConvolvedSurvival


class StandbyModel:
    def __init__(self, reliabilities, k=1, n_sims=10_000, lower=-np.inf):
        if k > len(reliabilities):
            raise ValueError(
                "Must be more nodes in the standby arrangement"
                + " than are required (k)"
            )
        self.reliabilities = reliabilities
        self.k = k
        self.N = len(reliabilities)
        self.n_sims = n_sims

        if k == 1:
            # Cold standby (k=1): the lifetime is the sum of the components'
            # lifetimes, whose survival function is their convolution. Compute
            # it deterministically by numerical convolution rather than
            # fitting a Kaplan-Meier to Monte-Carlo samples.
            self._sf_model = ConvolvedSurvival(reliabilities)
            self.model = None
        else:
            # For k >= 2 the lifetime is not a simple sum (it depends on the
            # order in which components fail), so fall back to the
            # Monte-Carlo + Kaplan-Meier approximation.
            x_random = self.random(n_sims)
            self.model = KaplanMeier.fit(x_random, set_lower_limit=lower)
            self._sf_model = None

    def random(self, size):
        if self.k == 1:
            # If k is only one for the standby node the
            # reliability can be estimated from the sum
            # of each of the components in the node.
            # i.e. it will fail after all of them fail.
            x_random = np.zeros(size)
            for model in self.reliabilities:
                x_random += model.random(size)

        else:
            # If k are required to continue then a random draw needs
            # a little more complexity. An individual run instance
            # can be simulated by getting failures for the first k
            # components. The simulation then tracks the k active
            # components. This is done by adding the next random
            # instance to the lowest of the k active components.
            # This is because the next standby component will start
            # working once the next of the k fails. This means the
            # lowest of the k active components will be the next
            # to fail. By simply adding the standby failures to the
            # lowest of the k active, at the end of the simulation
            # the lowest value in the queue will be the standby nodes
            # failure time. This simulation is repeated size
            # number of times and then the model is approximated
            # with a non parametric estimate.
            x_random = np.zeros(size)
            for i in range(size):
                pq: PriorityQueue = PriorityQueue()
                # start k streams:
                for node in self.reliabilities[: self.k]:
                    pq.put(node.random(1).item())

                # Add the next event time to the lowest value in the queue
                for node in self.reliabilities[self.k :]:  # noqa: E203
                    next_t = node.random(1).item()
                    current_lowest = pq.get()
                    pq.put(current_lowest + next_t)

                x_random[i] = pq.get()
        return x_random

    def mean(self, N=10_000):
        return self.random(N).mean()

    def sf(self, *args, **kwargs):
        if self._sf_model is not None:
            return self._sf_model.sf(*args, **kwargs)
        return self.model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        if self._sf_model is not None:
            return self._sf_model.ff(*args, **kwargs)
        return self.model.ff(*args, **kwargs)
