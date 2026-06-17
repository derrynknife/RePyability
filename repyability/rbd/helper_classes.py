import numpy as np


class PerfectReliability:
    @classmethod
    def sf(cls, x):
        return np.ones_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def cs(cls, x, X):
        # Always working -> conditional survival is always 1.
        return np.ones_like(np.atleast_1d(x)).astype(float)

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

    @classmethod
    def cs(cls, x, X):
        # Never working -> conditional survival is always 0.
        return np.zeros_like(np.atleast_1d(x)).astype(float)

    @classmethod
    def random(self, size):
        return np.zeros(size)
