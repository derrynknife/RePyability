"""Repair during a phased mission's phases (see ``PhasedMission``): the
simulation of a mission whose blocks are repaired while it runs, and the
checks of its ``repairability``.

A block with a repair model is under repair from the moment it fails, for
a time drawn from that model, and is then as good as new: it runs again
on a new life, drawn from its life model, whatever phase the mission is
in by then (a repair goes on through the phases' boundaries). Every block
is repaired on its own; there are no crews to wait for. The other blocks
fail once, as in any mission.

With repair a phase's diagram can go down and come back, so the mission
fails at the first instant the diagram of the phase under way is down,
and at a phase's start if its diagram is down then (a mission does not
wait for a repair). Events at one instant are taken in this order:
failures, then a phase's start (its diagram read as the blocks then
stand), then restorations. So a block back exactly when the structure
would fail, or exactly at a boundary where the next phase needs it, is
back too late.

The simulation plays all the missions through a phase together. Each
repaired block holds when it next fails (or failed, while under repair)
and when it is back from that failure; the structure's lifetime from
those failures, the blocks never coming back, is when the mission fails
unless a block is back first (the structure is coherent: a restoration
never fails it), so each round takes the missions' earliest restoration
that comes before that failure and the phase's end, and gives its block
a new life and the repair that will follow it. The Markov chain of
exponential lives and repairs (``_mission_chain``) follows the same
rules: a change to one goes into the other.
"""

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional

import numpy as np

from repyability.rbd import _mission_runs
from repyability.rbd._mission_settings import behaviour
from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd._sampling import lifetime_sampler
from repyability.utils.checks import no_distribution


def checked(repairability, mission) -> Dict[Hashable, Any]:
    """The mission's ``repairability``, each block's model of its repair
    times by name, checked against the mission.

    Raises
    ------
    ValueError
        If it is not a mapping, names a block that is not a component of
        the mission, is a member of a common-cause group, or has a
        probability of failing for a life, or a block is on standby in a
        mission with repairs, or a repaired block is set by a phase.
    TypeError
        If a repair model is not a model of a time (one with ``sf`` and
        ``random``), or is a surpyval distribution itself.
    """
    if repairability is None:
        return {}
    if not isinstance(repairability, Mapping):
        raise ValueError(
            "repairability must be a dict of each repaired block's model "
            f"of its repair times, by the block's name, got {repairability!r}."
        )
    members = {m for group in mission.ccf_groups for m in group.members}
    out: Dict[Hashable, Any] = {}
    for node, model in repairability.items():
        if node in members:
            raise ValueError(
                f"Block {node!r} is a member of a common-cause group, whose "
                "causes fail their members for the rest of the mission: a "
                "repaired block is in no group."
            )
        if node not in mission._given:
            raise ValueError(
                f"repairability names {node!r}, which is not a component "
                "of the mission: a node with a life model in one of its "
                "phases' diagrams."
            )
        if is_fixed_probability(mission.components[node]):
            raise ValueError(
                f"The model of block {node!r} is a probability of failing, "
                "which has no life to start again after a repair: give a "
                "repaired block a distribution of its life."
            )
        no_distribution(model, f"The repairability of block {node!r}")
        if not (
            callable(getattr(model, "sf", None))
            and callable(getattr(model, "random", None))
        ):
            raise TypeError(
                f"The repairability of block {node!r} must be a model of "
                "its repair times (a surpyval distribution's model, such "
                f"as surv.Exponential.from_params([2.0])), got {model!r}."
            )
        out[node] = model
    return out


def check_settings(repairability, phases) -> None:
    """Refuse the settings (``phases``, a mission drawn as one diagram's)
    that repairs do not go with: a block on standby, and a repaired block
    that is not working or not needed on its node's model."""
    if not repairability:
        return
    for phase in phases:
        for node, setting in phase.blocks.items():
            how = behaviour(setting)
            if how.standby:
                raise ValueError(
                    f"Phase {phase.name!r} has block {node!r} on standby, "
                    "and the mission has repairs: a mission with repairs "
                    "during its phases takes no block on standby."
                )
            if node in repairability and (
                not how.spends
                or how.model is not None
                or how.covariates is not None
            ):
                raise ValueError(
                    f"Phase {phase.name!r} sets block {node!r}, which is "
                    "repaired: a repaired block is working or not needed "
                    "in a phase, on its node's model, which each new life "
                    "is drawn from."
                )


def _draws(
    model,
    size: int,
    antithetic: bool,
    uniforms: Optional[Callable[[int], np.ndarray]],
) -> np.ndarray:
    """``size`` draws of ``model``: from numpy's global RNG as it stands
    (in antithetic pairs if asked), or from ``uniforms(j)``, its ``j``-th
    column of uniforms."""
    sampler = lifetime_sampler(model)
    if sampler is None:
        if antithetic or uniforms is not None:
            raise NotImplementedError(
                "Antithetic sampling and common random numbers need every "
                "block's draws, its repairs' too, to be replayable from "
                "uniforms (surpyval parametric distributions and the "
                "composite nodes built from them)."
            )
        return np.asarray(model.random(size), dtype=float)
    if uniforms is not None:
        u = np.column_stack([uniforms(j) for j in range(sampler.width)])
        return np.asarray(sampler.draw(u), dtype=float)
    if antithetic:
        half = np.random.random_sample((size // 2, sampler.width))
        both = np.empty(size)
        both[0::2] = sampler.draw(half)
        both[1::2] = sampler.draw(1.0 - half)
        return both
    u = np.random.random_sample((size, sampler.width))
    return np.asarray(sampler.draw(u), dtype=float)


class Source:
    """Where the missions' repaired blocks take their later lives and
    their repair times from: each block's ``k``-th, for every mission at
    once, drawn when first asked for and kept, so that the same missions
    played again (with a block held) meet the same draws.

    From numpy's global RNG as it stands (in antithetic pairs if asked),
    in the order the play asks; or, with ``key``, each from a stream of
    its own (common random numbers, see ``_paired``).
    """

    def __init__(
        self,
        mission,
        size: int,
        antithetic: bool = False,
        key: Optional[int] = None,
    ):
        self.mission = mission
        self.size = size
        self.antithetic = antithetic
        self.key = key
        self._kept: Dict[tuple, np.ndarray] = {}

    def _drawn(self, what: str, node: Hashable, k: int, model) -> np.ndarray:
        found = self._kept.get((what, node, k))
        if found is None:
            uniforms = None
            if self.key is not None:
                from repyability.rbd import _paired

                key, label = self.key, f"{what} {k} {node!r}"

                def uniforms(j: int) -> np.ndarray:
                    return _paired.uniforms(key, label, j, self.size)

            found = _draws(model, self.size, self.antithetic, uniforms)
            self._kept[(what, node, k)] = found
        return found

    def life(self, node: Hashable, k: int) -> np.ndarray:
        """Block ``node``'s ``k``-th life after its first, in every
        mission."""
        model = self.mission._lives[node].at(0)
        return self._drawn("life", node, k, model)

    def repair(self, node: Hashable, k: int) -> np.ndarray:
        """How long block ``node``'s repair after its ``k``-th life takes
        (its first life is its 0-th), in every mission."""
        return self._drawn("repair", node, k, self.mission.repairability[node])


def replayable(mission) -> None:
    """Raise unless every repair's draws can be replayed from uniforms,
    as common random numbers need."""
    for node, model in mission.repairability.items():
        if lifetime_sampler(model) is None:
            raise NotImplementedError(
                "Common random numbers need every block's draws to be "
                "replayable from uniforms (surpyval parametric "
                "distributions and the composite nodes built from them); "
                f"those of the repairs of block {node!r} are not."
            )


def play(
    mission,
    lives: Dict[Hashable, np.ndarray],
    size: int,
    source: Source,
    held: Optional[Hashable] = None,
) -> "_mission_runs.Runs":
    """The ``size`` missions with the components' first ``lives`` (see
    ``_mission_runs.draw``) and the repaired blocks' later lives and
    repairs from ``source``, the component ``held`` (if any) never
    failing."""
    count = len(mission.phases)
    repaired: List[Hashable] = [
        node for node in lives if node in mission.repairability
    ]
    once = [node for node in lives if node not in mission.repairability]
    state = _mission_runs.started(lives, once, size, held)
    # When each repaired block next fails (or failed, while under repair),
    # when it is back from that failure, and the lives it has had since
    # its first.
    fails = {
        node: (
            np.full(size, np.inf)
            if node == held
            else np.array(lives[node], dtype=float, copy=True)
        )
        for node in repaired
    }
    back = {node: fails[node] + source.repair(node, 0) for node in repaired}
    used = {node: np.zeros(size, dtype=int) for node in repaired}
    phase_of = np.full(size, count)
    time_of = np.full(size, np.inf)
    start = 0.0
    for j, (phase, setting) in enumerate(
        zip(mission.phases, mission._settings)
    ):
        stop = phase.end
        at = _mission_runs.read(mission, j, setting, state, start, stop)
        structure = phase.rbd._set_structure()
        components = phase.rbd._components()
        going = np.flatnonzero(phase_of == count)
        while going.size:
            nodes = {
                n: (
                    fails[n][going]
                    if n in fails
                    else (
                        at[n][going]
                        if n in at
                        else np.full(going.size, np.inf)
                    )
                )
                for n in components
            }
            failing = np.asarray(
                structure.lifetime(nodes, going.size), dtype=float
            )
            if repaired:
                backs = np.stack([back[n][going] for n in repaired])
                first = backs.argmin(axis=0)
                restored = backs.min(axis=0)
            else:
                first = np.zeros(going.size, dtype=int)
                restored = np.full(going.size, np.inf)
            # A failure, and the phase's end, come before a restoration
            # at the same instant.
            restore = (restored < failing) & (restored < stop)
            settled, when = going[~restore], failing[~restore]
            failed = ~(when > stop)
            phase_of[settled[failed]] = j
            time_of[settled[failed]] = np.maximum(when[failed], start)
            for i, node in enumerate(repaired):
                m = going[restore & (first == i)]
                if not m.size:
                    continue
                used[node][m] += 1
                for k in np.unique(used[node][m]):
                    mk = m[used[node][m] == k]
                    fails[node][mk] = (
                        back[node][mk] + source.life(node, int(k))[mk]
                    )
                    back[node][mk] = (
                        fails[node][mk] + source.repair(node, int(k))[mk]
                    )
            going = going[restore]
        start = stop
    return _mission_runs.Runs(phase_of, time_of, {})
