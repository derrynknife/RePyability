import numpy as np


class PerfectReliability:
    @classmethod
    def sf(cls, x):
        return np.ones_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def random(cls, size):
        return np.ones(size) * np.inf


class PerfectUnreliability:
    @classmethod
    def sf(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.ones_like(x).astype(float)

    def random(self, size):
        return np.zeros_like(size)


class ExactFailureTimeModel:
    def sf(self, x):
        x = np.atleast_1d(x)
        return (x < self.T).astype(float)

    def ff(self, x):
        x = np.atleast_1d(x)
        return (x >= self.T).astype(float)

    def random(self, size):
        return np.ones_like(size) * self.T


class ExactFailureTime:
    @classmethod
    def from_params(cls, T):
        out = ExactFailureTimeModel()
        out.T = T
        return out
