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

    def random(self, size):
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

    def mean(self, N=1_000_000):
        return self.random(N).mean()

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
