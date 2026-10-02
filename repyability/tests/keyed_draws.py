"""The draws a ``RepairableRBD`` simulation takes, worked out from the
definition of its streams (see ``repyability.rbd._streams``) rather than
by the code that makes them: a generator for each block of each stream,
seeded from the run's entropy, the stream's name and the block's position,
whose uniforms are laid out one row per draw and one column per simulation
(or antithetic pair)."""

from collections import defaultdict

import numpy as np

from repyability.rbd import _streams


def reference_draw(entropy, spec, antithetic, replication, k) -> float:
    """Draw ``k`` of simulation ``replication`` from the stream ``spec``."""
    column = replication // 2 if antithetic else replication
    block, j = divmod(column, spec.width)
    key = _streams.path_key(spec.path)
    seeds = np.random.SeedSequence(
        entropy, spawn_key=(key & 0xFFFFFFFF, key >> 32, spec.kind, block)
    )
    uniforms = np.random.Generator(np.random.PCG64(seeds)).random(
        (k + 1) * spec.width
    )
    u = uniforms[k * spec.width + j]
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
