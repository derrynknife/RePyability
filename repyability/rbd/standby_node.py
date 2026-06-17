from queue import PriorityQueue

import numpy as np
from scipy.stats import gamma as _gamma
from surpyval import KaplanMeier

from .numerical_convolution import (
    ConvolvedSurvival,
    is_perfect_switching,
    switch_success_probs,
)


def _identical_exponential_rate(models):
    """If every model is an Exponential with the same rate, return that rate;
    otherwise return None. The rate is taken as 1 / mean."""
    rate = None
    for model in models:
        dist = getattr(model, "dist", None)
        if dist is None or getattr(dist, "name", "") != "Exponential":
            return None
        mean = float(model.mean())
        if mean <= 0.0:
            return None
        this_rate = 1.0 / mean
        if rate is None:
            rate = this_rate
        elif not np.isclose(this_rate, rate):
            return None
    return rate


class _ExponentialStandbySurvival:
    """Exact survival function of an identical-exponential k-out-of-n cold
    standby arrangement.

    With N identical units of rate ``rate``, k operating at a time, the system
    fails at the (N-k+1)-th failure. While k units operate the failure rate is
    ``k*rate``, and by the memorylessness of the exponential the inter-failure
    times are i.i.d. Exponential(k*rate). Hence the lifetime is exactly
    Erlang(N-k+1, k*rate) -- deterministic, for any k.
    """

    def __init__(self, rate, n_units, k):
        self.shape = n_units - k + 1  # number of inter-failure gaps
        self.rate = k * rate  # failure rate while k units operate

    def sf(self, x):
        return _gamma.sf(x, a=self.shape, scale=1.0 / self.rate)

    def ff(self, x):
        return _gamma.cdf(x, a=self.shape, scale=1.0 / self.rate)

    def mean(self, *args, **kwargs):
        return self.shape / self.rate


class StandbyModel:
    def __init__(
        self,
        reliabilities,
        k=1,
        n_sims=10_000,
        lower=-np.inf,
        switching_probability=1.0,
    ):
        if k > len(reliabilities):
            raise ValueError(
                "Must be more nodes in the standby arrangement"
                + " than are required (k)"
            )
        self.reliabilities = reliabilities
        self.k = k
        self.N = len(reliabilities)
        self.n_sims = n_sims
        self.switching_probability = switching_probability

        rate = _identical_exponential_rate(reliabilities)
        if rate is not None and is_perfect_switching(switching_probability):
            # Identical exponential units: the cold standby lifetime is exactly
            # Erlang(N-k+1, k*rate) for any k, by the memorylessness of the
            # exponential. Use that closed form directly.
            self._sf_model = _ExponentialStandbySurvival(rate, self.N, k)
            self.model = None
        elif k == 1:
            # Cold standby (k=1): the lifetime is the sum of the components'
            # lifetimes (or a mixture of partial sums under imperfect
            # switching), whose survival function is computed deterministically
            # by numerical convolution rather than from Monte-Carlo samples.
            self._sf_model = ConvolvedSurvival(
                reliabilities, switching_probability=switching_probability
            )
            self.model = None
        else:
            # For k >= 2 with general (non-exponential) lifetimes the lifetime
            # is not a simple sum (it depends on the order in which components
            # fail), so fall back to the Monte-Carlo + Kaplan-Meier
            # approximation. Imperfect switching is not modelled in that case
            # yet.
            if not is_perfect_switching(switching_probability):
                raise NotImplementedError(
                    "switching_probability is only supported for k=1 cold"
                    " standby; for k>=2 leave it at 1.0 (perfect switching)."
                )
            x_random = self.random(n_sims)
            self.model = KaplanMeier.fit(x_random, set_lower_limit=lower)
            self._sf_model = None

    def random(self, size):
        if self.k == 1:
            # If k is only one for the standby node the reliability can be
            # estimated from the sum of each of the components in the node,
            # i.e. it will fail after all of them fail.
            x_random = np.asarray(
                self.reliabilities[0].random(size), dtype=float
            )
            if is_perfect_switching(self.switching_probability):
                for model in self.reliabilities[1:]:
                    x_random = x_random + model.random(size)
            else:
                # Under imperfect switching a spare only contributes if every
                # switch up to and including its own has succeeded.
                probs = switch_success_probs(
                    self.switching_probability, self.N
                )
                running = np.ones(size, dtype=bool)
                for model, p in zip(self.reliabilities[1:], probs):
                    running = running & (np.random.random(size) < p)
                    x_random = x_random + np.where(
                        running, model.random(size), 0.0
                    )

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
        # Use the exact/deterministic mean when an analytic survival model is
        # available (exponential closed form or convolution); otherwise fall
        # back to the Monte-Carlo estimate.
        if self._sf_model is not None:
            return self._sf_model.mean()
        return self.random(N).mean()

    def sf(self, *args, **kwargs):
        if self._sf_model is not None:
            return self._sf_model.sf(*args, **kwargs)
        return self.model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        if self._sf_model is not None:
            return self._sf_model.ff(*args, **kwargs)
        return self.model.ff(*args, **kwargs)
