import numpy as np
from surpyval import KaplanMeier


class RepeatedStandbyNode:
    def __init__(self, model, repeats, N=10_000, lower=-np.inf):
        self.model = model
        self.repeats = repeats
        x_random = self.random(N)

        # Create an approximation of the standby arrangement with
        # a Kaplan-Meier estimation.
        self.model = KaplanMeier.fit(x_random, set_lower_limit=lower)

    def random(self, size):
        randoms = self.model.random((self.repeats, size))
        return randoms.sum(axis=0)

    def mean(self, N=1000):
        return self.random(N).mean()

    def sf(self, *args, **kwargs):
        return self.model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        return self.model.ff(*args, **kwargs)
