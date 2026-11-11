"""A phased mission's simulation (see ``PhasedMission``): each block's
life drawn once per mission, then the missions played through the phases
together, the take-overs of blocks on standby one mission at a time.

A block's state in a mission is the age (on its life's clock, see
``_mission_settings``) at which it will fail if it runs on, ``end``, and
the mission's time less its age, ``offset``, so that, running, it fails at
the mission's time ``end + offset``. A block that runs through a phase
keeps both; a guaranteed one spends no hazard, so its ``end`` moves on by
the hazard of the stretch; an idle one does not age, so its ``offset``
grows by the time it is idle. A mission with no block on standby or
guaranteed, every life drawn as before, so plays out as the plain
simulation did, to the last bit.
"""

from typing import Dict, Hashable, List, NamedTuple, Optional, Tuple

import numpy as np

from repyability.rbd._mission_settings import behaviour
from repyability.rbd._sampling import lifetime_sampler


class Runs(NamedTuple):
    """Simulated missions."""

    #: The phase each fails in (its number), ``len(phases)`` for none.
    phase: np.ndarray
    #: When each fails, from the mission's start (``inf`` for never).
    time: np.ndarray
    #: For each phase and block on standby in it, by ``(phase number,
    #: block)``: whether each mission started it in the phase.
    started: Dict[Tuple[int, Hashable], np.ndarray]


def joined(parts: List[Runs]) -> Runs:
    """Batches of simulated missions as one."""
    return Runs(
        np.concatenate([part.phase for part in parts]),
        np.concatenate([part.time for part in parts]),
        {
            key: np.concatenate([part.started[key] for part in parts])
            for key in parts[0].started
        },
    )


def draw(mission, size: int, antithetic: bool) -> Dict[Hashable, np.ndarray]:
    """Each component's life, ``size`` draws of the age at which it fails
    (on its first phase's life), from numpy's global RNG as it stands."""
    models = {node: life.at(0) for node, life in mission._lives.items()}
    samplers = {
        node: lifetime_sampler(model) for node, model in models.items()
    }
    if antithetic and any(s is None for s in samplers.values()):
        raise NotImplementedError(
            "Antithetic sampling needs every component's lives to be "
            "drawn from uniforms (surpyval parametric distributions and "
            "the composite nodes built from them)."
        )
    width = sum(s.width for s in samplers.values() if s is not None)
    rows = size // 2 if antithetic else size
    u = np.random.random_sample((rows, width))
    lives: Dict[Hashable, np.ndarray] = {}
    start = 0
    for node, sampler in samplers.items():
        if sampler is None:
            lives[node] = np.asarray(models[node].random(size), dtype=float)
            continue
        block = u[:, start : start + sampler.width]
        start += sampler.width
        if antithetic:
            both = np.empty(size)
            both[0::2] = sampler.draw(block)
            both[1::2] = sampler.draw(1.0 - block)
            lives[node] = both
        else:
            lives[node] = np.asarray(sampler.draw(block), dtype=float)
    return lives


def play(
    mission,
    lives: Dict[Hashable, np.ndarray],
    size: int,
    held: Optional[Hashable] = None,
) -> Runs:
    """The ``size`` missions with the components' ``lives`` (see
    ``draw``), the component ``held`` (if any) never failing."""
    count = len(mission.phases)
    end = {
        node: (
            np.full(size, np.inf)
            if node == held
            else np.array(life, dtype=float, copy=True)
        )
        for node, life in lives.items()
    }
    offset = {node: np.zeros(size) for node in lives}
    # When each block failed (inf while it has not).
    dead = {node: np.full(size, np.inf) for node in lives}
    phase_of = np.full(size, count)
    time_of = np.full(size, np.inf)
    started: Dict[Tuple[int, Hashable], np.ndarray] = {}
    start = 0.0
    for j, (phase, setting) in enumerate(
        zip(mission.phases, mission._settings)
    ):
        stop = phase.end
        # Each block's failure as the phase's structure reads it: when it
        # failed or will fail if it runs on (-inf while idle on standby).
        at: Dict[Hashable, np.ndarray] = {}
        idle: List[Hashable] = []
        for node in lives:
            life = mission._lives[node]
            how = behaviour(setting.setting(node))
            alive = dead[node] > start
            if life.per_phase is not None and j > 0 and alive.any():
                age, ends = life.carry(
                    j, start - offset[node][alive], end[node][alive]
                )
                end[node][alive] = ends
                offset[node][alive] = start - age
            if how.standby:
                idle.append(node)
                at[node] = np.where(alive, -np.inf, dead[node])
                continue
            if not how.spends:
                _guaranteed(
                    life, j, end[node], offset[node], alive, start, stop
                )
            failing = np.where(alive, end[node] + offset[node], dead[node])
            at[node] = failing
            dead[node] = np.where(failing <= stop, failing, np.inf)
        rbd = phase.rbd
        structure = rbd._set_structure()
        nodes = {
            n: at.get(n, np.full(size, np.inf)) for n in rbd._components()
        }
        lifetime = np.asarray(structure.lifetime(nodes, size), dtype=float)
        going = phase_of == count
        failed = going & ~(lifetime > stop)
        fails_at = np.maximum(lifetime, start)
        if idle:
            for node in idle:
                started[(j, node)] = np.zeros(size, dtype=bool)
            works = structure.structure_function()
            for m in np.flatnonzero(failed):
                ready = {
                    node: (
                        end[node][m] - (start - offset[node][m])
                        if dead[node][m] > start
                        else None
                    )
                    for node in idle
                }
                fail, starts = _take_over(
                    works,
                    {
                        n: float(v[m])
                        for n, v in nodes.items()
                        if n not in idle
                    },
                    ready,
                    start,
                    stop,
                )
                for node, (when, until) in starts.items():
                    started[(j, node)][m] = True
                    offset[node][m] += when - start
                    if until <= stop:
                        dead[node][m] = until
                if fail is None:
                    failed[m] = False
                else:
                    fails_at[m] = fail
            for node in idle:
                # The time idle: the whole phase, unless started.
                offset[node] += np.where(
                    started[(j, node)], 0.0, phase.duration
                )
        phase_of[failed] = j
        time_of[failed] = fails_at[failed]
        start = stop
    return Runs(phase_of, time_of, started)


def _guaranteed(life, j, end, offset, alive, start, stop) -> None:
    """Move on, in place, the age at which a block guaranteed in phase
    ``j`` fails, by the hazard it does not spend running through it."""
    live = np.flatnonzero(alive & np.isfinite(end))
    if not live.size:
        return
    first = start - offset[live]
    last = stop - offset[live]
    skipped = life.hazard(j, last) - life.hazard(j, first)
    moved = skipped > 0.0
    if moved.any():
        ages = end[live[moved]]
        target = life.hazard(j, ages) + skipped[moved]
        # Not before the phase's end: rounding cannot fail it in the phase.
        end[live[moved]] = np.maximum(life.inverse(j, target), last[moved])


def _take_over(
    works,
    running: Dict[Hashable, float],
    ready: Dict[Hashable, Optional[float]],
    start: float,
    stop: float,
) -> Tuple[Optional[float], Dict[Hashable, Tuple[float, float]]]:
    """One mission through a phase with blocks on standby: when it fails
    (None if it gets through) and when each block that was started was.

    ``running`` holds when each other block fails (or failed), ``ready``
    each standby block's running time left (None if it has failed), in the
    order they are started. Whenever the structure (``works``) does not
    work, at the phase's start or at a failure, idle blocks are started in
    order until it does; then each just started (latest first) whose
    removal still leaves it working goes back to idle. If it cannot be made
    to work, the mission fails then. Each block started is given with
    when it started and when it fails (or will).
    """
    # Each started block's start and failure.
    starts: Dict[Hashable, Tuple[float, float]] = {}

    def status(t: float) -> Dict[Hashable, bool]:
        state = {node: failing > t for node, failing in running.items()}
        for node in ready:
            state[node] = node in starts and starts[node][1] > t
        return state

    t = start
    while True:
        if not works(status(t)):
            fresh = []
            for node, left in ready.items():
                if node in starts or left is None:
                    continue
                starts[node] = (t, t + left)
                fresh.append(node)
                if works(status(t)):
                    break
            else:
                # It cannot be made to work: none of them is started.
                for node in fresh:
                    del starts[node]
                return t, starts
            for node in reversed(fresh):
                kept = starts.pop(node)
                if not works(status(t)):
                    starts[node] = kept
        upcoming = [
            failing
            for failing in [*running.values()]
            + [until for _, until in starts.values()]
            if t < failing <= stop
        ]
        if not upcoming:
            return None, starts
        t = min(upcoming)
