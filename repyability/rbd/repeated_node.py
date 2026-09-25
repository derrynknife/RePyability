import numpy as np

from repyability.utils.wrappers import numpy_seed

from ._sampling import RowSampler, inverse_sampler

REPEATED_NODE_TYPES = {"parallel", "series"}
PARALLEL = 1
SERIES = 0


class RepeatedNode:
    def __init__(self, model, repeats, kind):
        if kind not in REPEATED_NODE_TYPES:
            raise ValueError("'kind' must be either 'parallel' or 'series'")

        self.model = model
        if kind == "parallel":
            self.kind = PARALLEL
        else:
            self.kind = SERIES
        self.repeats = repeats

    def random(self, size, seed=None):
        with numpy_seed(seed):
            randoms = self.model.random((size, self.repeats))
        if self.kind == SERIES:
            # If repetition is in series, then a random event will be the
            # smallest of all the events in series. i.e. when the first item
            # fails.
            return randoms.min(axis=1)
        else:
            # If repetition is in parallel, then a random event will be the
            # largest of all the events in series. i.e. when the last item
            # fails.
            return randoms.max(axis=1)

    def _row_sampler(self):
        """``random(1)`` as a :class:`~._sampling.RowSampler` (``repeats``
        draws of the model, in order), so an RBD with this node batches its
        draws; ``None`` unless the model's draws can be replayed."""
        sampler = inverse_sampler(self.model)
        if sampler is None:
            return None
        reduce = np.min if self.kind == SERIES else np.max

        def draw(u):
            draws = sampler(np.ascontiguousarray(u))
            return reduce(np.asarray(draws, dtype=float), axis=1)

        return RowSampler(self.repeats, draw)

    def mean(self, N=1_000_000, seed=None):
        return self.random(N, seed=seed).mean()

    def sf(self, x):
        if self.kind == SERIES:
            sf = self.model.sf(x) ** self.repeats
        else:
            sf = 1 - (self.model.ff(x) ** self.repeats)
        return sf

    def ff(self, x):
        if self.kind == SERIES:
            ff = 1 - (self.model.sf(x) ** self.repeats)
        else:
            ff = self.model.ff(x) ** self.repeats
        return ff

    def cs(self, x, X):
        """Conditional survival ``R(x | X) = sf(X + x) / sf(X)``."""
        from repyability.utils.wrappers import conditional_survival

        return conditional_survival(self, x, X)
