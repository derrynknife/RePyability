"""A phased mission's simulation (see ``PhasedMission``): each block's
life drawn once per mission, then the missions played through the phases
together, the take-overs of blocks on standby included.

A block's state in a mission is the age (on its life's clock, see
``_mission_settings``) at which it will fail if it runs on, ``end``, and
the mission's time less its age, ``offset``, so that, running, it fails at
the mission's time ``end + offset``. A block that runs through a phase
keeps both; a guaranteed one spends no hazard, so its ``end`` moves on by
the hazard of the stretch; an idle one ages at its ``dormancy_factor``
``f`` (0, cold, by default), so its ``offset`` grows by ``1 - f`` of the
time it is idle, and, warm, it fails idle once its age reaches ``end``. A
block called on starts if it is alive and starts on demand, one draw per
block on standby, phase and mission (``draw_switches``, made only where a
switch can fail, after the lives); else it has failed. A mission with no
block on standby or guaranteed, every life drawn as before, so plays out
as the plain simulation did, to the last bit.
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


def draw_switches(
    mission, size: int, antithetic: bool = False
) -> Dict[Tuple[int, Hashable], np.ndarray]:
    """For each phase and block on standby in it whose switch can fail,
    by ``(phase number, block)``: whether it starts if called on in each
    of ``size`` missions, from numpy's global RNG as it stands (drawn
    after the lives, in phase order, then the phase's order of calling
    on its blocks; none where every switch is perfect)."""
    out: Dict[Tuple[int, Hashable], np.ndarray] = {}
    for j, setting in enumerate(mission._settings):
        for node in setting.standby_order(mission.components):
            chance = behaviour(setting.setting(node)).switching
            if chance == 1.0:
                continue
            if antithetic:
                half = np.random.random_sample(size // 2)
                u = np.empty(size)
                u[0::2], u[1::2] = half, 1.0 - half
            else:
                u = np.random.random_sample(size)
            out[(j, node)] = u < chance
    return out


def play(
    mission,
    lives: Dict[Hashable, np.ndarray],
    size: int,
    held: Optional[Hashable] = None,
    switches: Optional[Dict[Tuple[int, Hashable], np.ndarray]] = None,
) -> Runs:
    """The ``size`` missions with the components' ``lives`` (see
    ``draw``) and whether their switches work (``switches``, see
    ``draw_switches``; perfect where not given), the component ``held``
    (if any) never failing (nor its switch)."""
    switches = switches or {}
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
                at[node] = np.where(alive, -np.inf, dead[node])
                continue
            if not how.spends:
                _guaranteed(
                    life, j, end[node], offset[node], alive, start, stop
                )
            failing = np.where(alive, end[node] + offset[node], dead[node])
            at[node] = failing
            dead[node] = np.where(failing <= stop, failing, np.inf)
        idle = setting.standby_order(lives)
        # Each idle block's running time left at the phase's start, its
        # dormancy, and whether it starts if called on.
        left = {
            node: np.where(
                dead[node] > start, end[node] - (start - offset[node]), np.nan
            )
            for node in idle
        }
        rates = {
            node: behaviour(setting.setting(node)).dormancy_factor
            for node in idle
        }
        passes = {
            node: (
                np.ones(size, dtype=bool)
                if node == held or (j, node) not in switches
                else switches[(j, node)]
            )
            for node in idle
        }
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
            m = np.flatnonzero(failed)
            fail, when, lost = _take_over(
                structure,
                {n: v[m] for n, v in nodes.items()},
                {node: left[node][m] for node in idle},
                start,
                stop,
                rates,
                {node: passes[node][m] for node in idle},
            )
            failed[m] = fail <= stop
            fails_at[m] = fail
            for node in idle:
                _after_standby(
                    node,
                    j,
                    m,
                    when[node],
                    lost[node],
                    left[node],
                    rates[node],
                    (start, stop, phase.duration),
                    started,
                    offset[node],
                    dead[node],
                )
        phase_of[failed] = j
        time_of[failed] = fails_at[failed]
        start = stop
    return Runs(phase_of, time_of, started)


def _after_standby(
    node, j, m, when, lost, left, rate, span, started, offset, dead
) -> None:
    """Bring up to date, in place, block ``node``'s state (its ``offset``
    and when it failed, ``dead``) after phase ``j``, which spans ``span``,
    on standby in it with dormancy ``rate``: in the missions ``m`` whose
    take-overs were worked out, started at ``when`` (``nan`` if not), or
    found failed when called on at ``lost`` (``nan`` if not), its running
    time ``left`` at the phase's start; idle through the phase where
    neither."""
    start, stop, duration = span
    begun = ~np.isnan(when)
    ms = m[begun]
    started[(j, node)][ms] = True
    # Idle until started, ageing at the dormancy, then running.
    idle_for = np.full(left.shape, duration)
    idle_for[ms] = when[begun] - start
    offset += (1.0 - rate) * idle_for
    until = when[begun] + (left[ms] - rate * idle_for[ms])
    dead[ms] = np.where(until <= stop, until, dead[ms])
    found = ~np.isnan(lost)
    dead[m[found]] = lost[found]
    if rate > 0.0:
        # Warm, and never started or found failed: failed idle if its age
        # reached its end within the phase.
        quiet = np.ones(left.shape, dtype=bool)
        quiet[ms] = False
        quiet[m[found]] = False
        quiet &= ~np.isnan(left)
        with np.errstate(divide="ignore", invalid="ignore"):
            idle_end = start + left / rate
        gone = quiet & (idle_end <= stop)
        dead[gone] = idle_end[gone]


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
    structure,
    nodes: Dict[Hashable, np.ndarray],
    ready: Dict[Hashable, np.ndarray],
    start: float,
    stop: float,
    dormancy: Optional[Dict[Hashable, float]] = None,
    passes: Optional[Dict[Hashable, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[Hashable, np.ndarray], Dict[Hashable, np.ndarray]]:
    """Missions through a phase with blocks on standby, all at once: when
    each fails (``inf`` if it gets through), when each block on standby
    was started in it (``nan`` if not), and when each was found failed
    when called on (``nan`` if not): failed idle, at the time it failed,
    or failing to start, at the call.

    ``nodes`` holds when each block fails (or failed), ``ready`` each
    standby block's running time left at the phase's start (``nan`` if it
    has failed), in the order they are called on, ``dormancy_factor`` how fast
    each ages idle (0 if not given: it does not) and ``passes`` whether
    each starts if called on (surely if not given). Whenever the
    structure does not work, at the phase's start or at a failure, idle
    blocks are called on in order until it does: one alive (its time left
    less its dormant ageing still positive) that passes its switch
    starts, any other has failed; then each just started (latest first)
    whose removal still leaves it working goes back to idle. If it cannot
    be made to work, the mission fails then.

    The structure works at ``t`` when its lifetime, from each block's
    failure (a block on standby's when started and running, ``-inf``
    idle), is past ``t``; and nothing changes until it stops working, so
    each round jumps to then.
    """
    size = len(next(iter(nodes.values())))
    dormancy = dormancy or {}
    passes = passes or {}
    ready = {node: np.array(v, dtype=float) for node, v in ready.items()}
    times = {n: np.array(v, dtype=float) for n, v in nodes.items()}
    for node in ready:
        times[node] = np.full(size, -np.inf)
    when = {node: np.full(size, np.nan) for node in ready}
    lost = {node: np.full(size, np.nan) for node in ready}
    fail = np.full(size, np.inf)
    t = np.full(size, float(start))

    def lasts(m: np.ndarray) -> np.ndarray:
        """Until when the structure works, in missions ``m``."""
        return structure.lifetime({n: v[m] for n, v in times.items()}, m.size)

    def put_back(node: Hashable, m: np.ndarray) -> None:
        when[node][m] = np.nan
        times[node][m] = -np.inf

    going = np.arange(size)
    while going.size:
        down = lasts(going)
        going = going[down <= stop]
        t[going] = np.maximum(t[going], down[down <= stop])
        # Start idle blocks in order while it does not work.
        broken = going
        fresh = {}
        for node in ready:
            idle = np.isnan(when[node][broken]) & ~np.isnan(
                ready[node][broken]
            )
            called = broken[idle]
            rate = dormancy.get(node, 0.0)
            left = ready[node][called] - rate * (t[called] - start)
            ok = left > 0.0
            if node in passes:
                ok &= passes[node][called]
            gone = called[~ok]
            if gone.size:
                # Failed idle (warm) by now, or failing to start.
                with np.errstate(divide="ignore", invalid="ignore"):
                    idle_end = start + ready[node][gone] / rate
                lost[node][gone] = np.where(left[~ok] > 0.0, t[gone], idle_end)
                ready[node][gone] = np.nan
            begin = called[ok]
            when[node][begin] = t[begin]
            times[node][begin] = t[begin] + left[ok]
            fresh[node] = begin
            broken = broken[~(lasts(broken) > t[broken])]
        # It cannot be made to work: none of them is started.
        for node, begun in fresh.items():
            put_back(node, np.intersect1d(begun, broken))
        fail[broken] = t[broken]
        going = np.setdiff1d(going, broken)
        # Those just started that are not needed go back to idle.
        for node in reversed(list(ready)):
            back = np.intersect1d(fresh[node], going)
            kept = times[node][back]
            put_back(node, back)
            needed = ~(lasts(back) > t[back])
            when[node][back[needed]] = t[back[needed]]
            times[node][back[needed]] = kept[needed]
    return fail, when, lost
