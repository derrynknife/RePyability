"""Exact values of a phased mission with blocks on standby (see
``PhasedMission.from_diagram``), where every block's life has a constant
failure rate in every phase: a continuous-time Markov chain over the
blocks' states, run through the phases.

With constant rates a block's age does not matter, only whether it is
running, idle or failed, so the mission is a Markov chain over the
blocks' joint states. Within a phase a running block fails at its rate in
the phase (none when guaranteed), an idle one at its dormancy times that
rate; when the phase's structure stops working, the blocks on standby are
called on by the same rule as the simulation's (``_mission_runs``): in
order, each starting with its switching probability (else failed, on
demand), until the structure works, then each just started that is not
needed stood down (latest first); if none makes it work the mission
fails. A block's switch has one outcome per phase, as a simulated
mission's (``draw_switches``): a block stood down after starting is idle
*and passed*, and starts surely if called on again in the phase. At a
phase's start every failed block stays failed, every live block on
standby in the phase goes idle and every other runs, and the structure
is checked as at a failure.

A failed mission keeps which blocks on standby it had started in the
phase (for the summary's take-overs) until the phase ends. The states
reachable in each phase are listed from its starting states, up to
``MAX_STATES``, and the chain carried through the phase by
``scipy.sparse.linalg.expm_multiply``, accurate to about 1e-15 in each
probability (the failure probabilities are summed from the failed
states' own, so a small one keeps its precision).
"""

from typing import Any, Callable, Dict, Hashable, List, NamedTuple, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from repyability.rbd._common import _constant_rate
from repyability.rbd._mission_settings import PhasedLife, _hazard, behaviour
from repyability.rbd._model_utils import MODEL_ERRORS, is_fixed_probability

#: The most states the chain may reach in one phase before the exact
#: values refuse, pointing to the simulation.
MAX_STATES = 20_000

# A block's state in a phase.
FAILED, RUNNING, IDLE, PASSED, FAILED_STARTED = 0, 1, 2, 3, 4
_UP = (RUNNING,)
_IDLE = (IDLE, PASSED)


class ChainValues(NamedTuple):
    """A mission's values from its chain."""

    #: The probability of failing in each phase.
    failing: List[float]
    #: The probability of getting through each phase.
    through: List[float]
    #: Per phase, each block on standby in it: the probability that it
    #: was started in the phase.
    takeovers: List[Dict[Hashable, float]]
    #: The reliability at each of the times asked for (or None).
    curve: Optional[np.ndarray]


def _rate(model) -> Optional[float]:
    """The constant failure rate of the life ``model``, or None: from
    surpyval's hazard where it has one, else from its cumulative hazard
    (a regression block at fixed covariates, say), ``H(t) = l t`` at
    times either side of its mean life."""
    if is_fixed_probability(model):
        return None
    rate = _constant_rate(model)
    if rate is not None:
        return rate
    if not callable(getattr(model, "sf", None)):
        return None
    try:
        from repyability.rbd._mission_settings import _inverse

        scale = float(_inverse(model, np.array([1.0]))[0])
        t = scale * np.array([0.01, 0.5, 1.0, 2.0, 5.0])
        hazard = np.asarray(_hazard(model, t), dtype=float)
    except MODEL_ERRORS:
        return None
    if not (np.isfinite(scale) and scale > 0 and np.isfinite(hazard).all()):
        return None
    rates = hazard / t
    if np.allclose(rates, rates[2], rtol=1e-9, atol=0.0) and rates[2] > 0:
        return float(rates[2])
    return None


def _phase_models(life, count: int) -> List[Any]:
    """A block's life model in each of ``count`` phases."""
    if life.per_phase is not None:
        return list(life.per_phase)
    if isinstance(life.model, PhasedLife):
        return list(life.model.models)
    return [life.model] * count


def rates(mission) -> Dict[Hashable, List[float]]:
    """Each block's failure rate in each phase, running (0 where it is
    guaranteed).

    Raises
    ------
    NotImplementedError
        If a block's life has no constant failure rate in a phase, with
        the block and phase named.
    """
    count = len(mission.phases)
    out: Dict[Hashable, List[float]] = {}
    for node, life in mission._lives.items():
        row = []
        for j, model in enumerate(_phase_models(life, count)):
            setting = mission._settings[j]
            if not behaviour(setting.setting(node)).spends:
                row.append(0.0)
                continue
            rate = _rate(model)
            if rate is None:
                raise NotImplementedError(
                    f"Blocks on standby start when the phase needs them, "
                    "which depends on the other blocks' failures; the "
                    "mission is exact (a Markov chain) only where every "
                    "block's life has a constant failure rate, and block "
                    f"{node!r} has none in phase {setting.name!r}, so it is "
                    "simulated: method='simulate', or reliability_interval."
                )
            row.append(rate)
        out[node] = row
    return out


class _Phase:
    """One phase's chain: its states, generator and starting rule."""

    def __init__(self, mission, j: int, rate, held: Optional[Hashable]):
        self.blocks = list(mission.components)
        self.index = {node: i for i, node in enumerate(self.blocks)}
        setting = mission._settings[j]
        phase = mission.phases[j]
        self.order = [
            self.index[n] for n in setting.standby_order(self.blocks)
        ]
        self.standby = frozenset(self.order)
        how = [behaviour(setting.setting(node)) for node in self.blocks]
        self.switching = [
            1.0 if node == held else h.switching
            for node, h in zip(self.blocks, how)
        ]
        self.running = [
            0.0 if node == held else rate[node][j] for node in self.blocks
        ]
        self.dormant = [r * h.dormancy for r, h in zip(self.running, how)]
        self._works = _structure(mission, phase, self.blocks)
        self.duration = phase.duration

    def works(self, codes: tuple) -> bool:
        return self._works(
            frozenset(i for i, c in enumerate(codes) if c in _UP)
        )

    def entered(self, codes: tuple) -> tuple:
        """A live mission's state at this phase's start, from its state
        at the last one's end: failed blocks failed, live ones idle if on
        standby here, else running."""
        return tuple(
            (
                FAILED
                if c in (FAILED, FAILED_STARTED)
                else (IDLE if i in self.standby else RUNNING)
            )
            for i, c in enumerate(codes)
        )

    def resolve(self, codes: tuple) -> List[tuple]:
        """Where the state ``codes`` goes at once, with the probabilities:
        itself if the structure works, else the blocks on standby called
        on in order (see the module docstring); a failed mission is
        ``("failed", started)``, the blocks it had started in the
        phase."""
        if self.works(codes):
            return [(1.0, codes)]
        before = frozenset(
            i for i in self.standby if codes[i] in (RUNNING, FAILED_STARTED)
        )
        out: List[tuple] = []

        def go(codes: tuple, at: int, fresh: list, chance: float) -> None:
            for position in range(at, len(self.order)):
                i = self.order[position]
                if codes[i] not in _IDLE:
                    continue
                p = 1.0 if codes[i] == PASSED else self.switching[i]
                if p < 1.0:
                    go(
                        _put(codes, i, FAILED),
                        position + 1,
                        fresh,
                        chance * (1 - p),
                    )
                    if p == 0.0:
                        return
                chance *= p
                codes = _put(codes, i, RUNNING)
                fresh = fresh + [i]
                if self.works(codes):
                    for k in reversed(fresh):
                        back = PASSED if self.switching[k] < 1.0 else IDLE
                        trial = _put(codes, k, back)
                        if self.works(trial):
                            codes = trial
                    out.append((chance, codes))
                    return
            out.append((chance, ("failed", before)))

        go(codes, 0, [], 1.0)
        return out

    def moves(self, codes: tuple) -> List[tuple]:
        """The transitions out of the live state ``codes``: ``(rate,
        probability, state)``."""
        out = []
        for i, c in enumerate(codes):
            if c in _UP and self.running[i] > 0.0:
                after = FAILED_STARTED if i in self.standby else FAILED
                for p, state in self.resolve(_put(codes, i, after)):
                    out.append((self.running[i], p, state))
            elif c in _IDLE and self.dormant[i] > 0.0:
                out.append((self.dormant[i], 1.0, _put(codes, i, FAILED)))
        return out


def _put(codes: tuple, i: int, code: int) -> tuple:
    return codes[:i] + (code,) + codes[i + 1 :]


def _structure(mission, phase, blocks) -> Callable[[frozenset], bool]:
    """Whether the phase's structure works with the blocks numbered in a
    set running and the others not, as the simulation asks it (each
    worked out once)."""
    structure = phase.rbd._set_structure()
    components = list(phase.rbd._components())
    index = {node: i for i, node in enumerate(blocks)}
    known: Dict[frozenset, bool] = {}

    def works(up: frozenset) -> bool:
        if up not in known:
            nodes = {
                n: np.array(
                    [np.inf if n not in index or index[n] in up else -np.inf]
                )
                for n in components
            }
            known[up] = bool(structure.lifetime(nodes, 1)[0] > 0.0)
        return known[up]

    return works


def run(
    mission,
    held: Optional[Hashable] = None,
    times: Optional[np.ndarray] = None,
) -> ChainValues:
    """The mission's values from its chain, the block ``held`` (if any)
    never failing (nor its switch), and its reliability at ``times``
    (from 0 to the duration; a time at a phase's boundary belongs to the
    phase starting then).

    Raises
    ------
    NotImplementedError
        If a block's life has no constant rate in a phase, or a phase's
        chain reaches more than ``MAX_STATES`` states.
    """
    rate = rates(mission)
    count = len(mission.phases)
    starts = [phase.end - phase.duration for phase in mission.phases]
    if times is not None:
        flat = np.ravel(np.asarray(times, dtype=float))
        within = np.searchsorted(starts, flat, side="right") - 1
        curve = np.empty(flat.shape)
    live: Dict[tuple, float] = {
        tuple(RUNNING for _ in mission.components): 1.0
    }
    failing: List[float] = []
    through: List[float] = []
    takeovers: List[Dict[Hashable, float]] = []
    for j in range(count):
        chain = _Phase(mission, j, rate, held)
        start: Dict[Any, float] = {}
        for codes, p in live.items():
            for q, state in chain.resolve(chain.entered(codes)):
                if p * q > 0.0:
                    start[state] = start.get(state, 0.0) + p * q
        states, generator = _listed(chain, list(start), mission, j)
        number = {state: k for k, state in enumerate(states)}
        vector = np.zeros(len(states))
        for state, p in start.items():
            vector[number[state]] = p
        if times is not None:
            here = np.flatnonzero(within == j)
            order = here[np.argsort(flat[here])]
            at, now = starts[j], vector
            for k in order:
                step = flat[k] - at
                if step > 0.0:
                    now = _carry(generator, now, step)
                    at = flat[k]
                curve[k] = _alive(states, now)
        end = _carry(generator, vector, chain.duration)
        failed = [k for k, s in enumerate(states) if s[0] == "failed"]
        failing.append(float(np.clip(end[failed], 0, None).sum()))
        live = {
            s: float(end[k])
            for k, s in enumerate(states)
            if s[0] != "failed" and end[k] > 0.0
        }
        through.append(_alive(states, end))
        started: Dict[Hashable, float] = {}
        for i in chain.order:
            node = chain.blocks[i]
            total = 0.0
            for k, s in enumerate(states):
                if s[0] == "failed":
                    total += end[k] if i in s[1] else 0.0
                elif s[i] in (RUNNING, FAILED_STARTED):
                    total += end[k]
            started[node] = float(min(max(total, 0.0), 1.0))
        takeovers.append(started)
    out = None
    if times is not None:
        out = np.reshape(curve, np.shape(times))
    return ChainValues(failing, through, takeovers, out)


def _alive(states: List[Any], vector: np.ndarray) -> float:
    """The probability of the live states."""
    total = sum(vector[k] for k, s in enumerate(states) if s[0] != "failed")
    return float(min(max(total, 0.0), 1.0))


def _carry(generator, vector: np.ndarray, duration: float) -> np.ndarray:
    """The chain's distribution ``duration`` after ``vector``."""
    if duration == 0.0 or generator.nnz == 0:
        return vector.copy()
    return np.clip(expm_multiply(generator * duration, vector), 0.0, None)


def _listed(chain: _Phase, first: List[Any], mission, j: int):
    """The states reachable in the phase from ``first``, and the
    transpose of its generator over them (a failed mission's states are
    absorbing).

    Raises
    ------
    NotImplementedError
        If they are more than ``MAX_STATES``.
    """
    states = list(first)
    number = {state: k for k, state in enumerate(states)}
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    k = 0
    while k < len(states):
        state = states[k]
        if state[0] != "failed":
            for rate, p, after in chain.moves(state):
                if after not in number:
                    if len(states) >= MAX_STATES:
                        raise NotImplementedError(
                            "Blocks on standby start when the phase needs "
                            "them; the mission is exact (a Markov chain over "
                            "the blocks' states) up to MAX_STATES = "
                            f"{MAX_STATES} states in a phase, and phase "
                            f"{mission.phases[j].name!r} reaches more, so "
                            "it is simulated: method='simulate', or "
                            "reliability_interval."
                        )
                    number[after] = len(states)
                    states.append(after)
                flow = rate * p
                rows += [number[after], k]
                cols += [k, k]
                values += [flow, -flow]
        k += 1
    size = len(states)
    generator = sparse.csr_matrix((values, (rows, cols)), shape=(size, size))
    return states, generator
