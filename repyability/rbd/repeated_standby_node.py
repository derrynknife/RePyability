import numpy as np
from surpyval import KaplanMeier

from .numerical_convolution import ConvolvedSurvival


class RepeatedStandbyNode:
    def __init__(
        self,
        model,
        repeats,
        N=10_000,
        lower=-np.inf,
        switching_probability=1.0,
    ):
        self.model = model
        self.repeats = repeats
        self.switching_probability = switching_probability

        # Repeated cold standby: the lifetime is the sum of `repeats`
        # independent copies of `model` (a mixture of partial sums under
        # imperfect switching). Its survival function is computed
        # deterministically here. Note: random()/mean() below still reflect
        # perfect switching.
        self._sf_model = ConvolvedSurvival(
            [model] * repeats, switching_probability=switching_probability
        )

        # A Kaplan-Meier fit is retained only for random()/mean().
        x_random = self.random(N)
        self.model = KaplanMeier.fit(x_random, set_lower_limit=lower)

    def random(self, size):
        randoms = self.model.random((self.repeats, size))
        return randoms.sum(axis=0)

    def mean(self, N=1000):
        return self.random(N).mean()

    def sf(self, *args, **kwargs):
        return self._sf_model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        return self._sf_model.ff(*args, **kwargs)
