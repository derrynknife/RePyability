import numpy as np


class FixedProbability:
    def sf(self, x):
        return np.ones_like(x) * self.p

    def ff(self, x):
        return 1 - (np.ones_like(x) * self.p)


class FixedProbabilityFitter:
    @classmethod
    def from_params(cls, p):
        out = FixedProbability()
        out.p = p
        return out
