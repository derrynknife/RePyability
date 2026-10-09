"""The draws a ``RepairableRBD`` simulation takes, worked out from the
definition of its streams (see ``repyability.rbd._streams``) rather than
by the code that makes them: uniform ``k`` of simulation ``r`` is the
``k``-th of numpy's Philox generator keyed from the run's entropy and the
stream's name, with the counter ``(0, r, 0, 0)`` (``r`` an antithetic
pair's number, with pairs)."""

from collections import defaultdict

import numpy as np

from repyability.rbd import _streams


def reference_uniform(entropy, spec, r: int, k: int) -> float:
    """Uniform ``k`` of simulation (or pair) ``r`` of the stream ``spec``."""
    path = _streams.path_key(spec.path)
    seeds = np.random.SeedSequence(
        entropy, spawn_key=(path & 0xFFFFFFFF, path >> 32, spec.kind)
    )
    key = seeds.generate_state(2, np.uint64)
    counter = np.array([0, r, 0, 0], dtype=np.uint64)
    philox = np.random.Philox(key=key, counter=counter)
    return float(np.random.Generator(philox).random(k + 1)[k])


def reference_draw(entropy, spec, antithetic, replication, k) -> float:
    """Draw ``k`` of simulation ``replication`` from the stream ``spec``."""
    r = replication // 2 if antithetic else replication
    u = reference_uniform(entropy, spec, r, k)
    if antithetic and replication % 2:
        u = 1.0 - u
    return float(np.asarray(spec.sampler(np.array([u])), dtype=float)[0])


class KeyedDraws:
    """The successive draws of each stream in each simulation of a run of
    ``rbd`` over ``t_simulation`` seeded with ``seed`` (or after
    ``np.random.seed(seed)``)."""

    def __init__(self, rbd, t_simulation, seed, antithetic=False):
        self.entropy = _streams.entropy_of(seed)
        self.plan, _ = rbd._stream_plan(t_simulation, self.entropy, antithetic)
        self.antithetic = antithetic
        self.taken: dict = defaultdict(int)

    def next(self, path: tuple, kind: int, replication: int = 0) -> float:
        """The next draw of stream ``(path, kind)`` in simulation
        ``replication``."""
        spec = self.plan.specs[(path, kind)]
        k = self.taken[(path, kind, replication)]
        self.taken[(path, kind, replication)] = k + 1
        return reference_draw(
            self.entropy, spec, self.antithetic, replication, k
        )
