"""The exact long-run values of a repairable standby group (#91) whose units
have exponential lives and repair times.

A group has ``units`` identical units, ``k`` of which must operate for it to
be up; the others wait as spares. An operating unit fails at rate ``lam``,
and a spare at ``dormancy * lam`` (0 for cold standby, 1 for hot), found at
once. A failed unit is repaired at rate ``mu``, ``crews`` at a time (all at
once by default), and then operates if a position is empty, or joins the
spares. When an
operating unit fails and a spare is ready, the switch onto it succeeds with
probability ``switching``; a failed switch leaves the position empty until a
repaired unit fills it.

With exponential units the group is a Markov chain on how many units
operate, how many are spares and how many are under repair. Its states are
few (at most ``(k + 1) * (units + 1)``), and its balance equations are
solved directly. From new (every unit ready) it is followed over time by
``_chain_transient`` (#146).

A group with ``rates`` (a load-sharing group, or a block of identical
units: see ``_events._Standby``) has no spares: every unit up operates,
and the group is up while at least ``k`` do. While ``s`` operate, each
fails at ``rates[s - 1] * lam`` (an exponential unit's life used up at
that rate), so the chain is a birth-death chain on how many operate.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

#: A state: the units operating, the spares, and the units under repair.
State = Tuple[int, int, int]


class StandbyChain(NamedTuple):
    """A group's Markov chain: its states (see ``State``), the first with
    every unit ready (``k`` operating, the rest spares); its generator, the
    rate from each state (row) to each other (column); its long-run
    distribution; and, in each state, whether the group is up, the rate of
    its failures (up to down) and that of its units' failures (the repairs
    it pays for)."""

    states: List[State]
    generator: np.ndarray
    probabilities: np.ndarray
    up: np.ndarray
    failures: np.ndarray
    unit_failures: np.ndarray


class StandbyLongRun(NamedTuple):
    """A group's long-run values: the fraction of time it is up, its failures
    (up to down) per unit time, its units' failures per unit time (the
    repairs it pays for), and the fraction of time it is down (summed over
    the states it is down in, so that a small one keeps its precision)."""

    availability: float
    failure_frequency: float
    unit_failure_frequency: float
    unavailability: float


def _rate(rates: Optional[Sequence[float]], operating: int) -> float:
    """The rate at which each of ``operating`` units uses up its life: 1
    in a standby group."""
    return 1.0 if rates is None else float(rates[operating - 1])


def _state_rates(
    operating: int,
    spares: int,
    k: int,
    lam: float,
    dormancy: float,
    switching: float,
    rates: Optional[Sequence[float]],
) -> Tuple[float, float, float]:
    """In a state: the rate of the units' failures, that of the group's
    (up to down: 0 where it is down) and whether it is up (1 or 0)."""
    if rates is not None:
        each = lam * _rate(rates, operating) if operating else 0.0
        if operating < k:
            return operating * each, 0.0, 0.0
        # Down at a failure only from exactly k operating.
        return operating * each, (k * each if operating == k else 0.0), 1.0
    units_failing = (operating + spares * dormancy) * lam
    if operating != k:
        return units_failing, 0.0, 0.0
    # An operating unit fails, and no spare takes over.
    takeover = switching if spares else 0.0
    return units_failing, k * lam * (1.0 - takeover), 1.0


def _transitions(
    units: int,
    k: int,
    lam: float,
    mu: float,
    dormancy: float,
    switching: float,
    crews: Optional[int],
    rates: Optional[Sequence[float]] = None,
) -> Tuple[List[State], Dict[Tuple[State, State], float]]:
    """The states reached from the one with ``k`` units operating and the
    rest spares (every unit operating, with ``rates``), and the rates
    between them."""
    operate = k if rates is None else units
    start: State = (operate, units - operate, 0)
    states = [start]
    seen = {start}
    moves: Dict[Tuple[State, State], float] = {}

    def add(source: State, target: State, rate: float) -> None:
        if rate <= 0.0 or target == source:
            return
        moves[(source, target)] = moves.get((source, target), 0.0) + rate
        if target not in seen:
            seen.add(target)
            states.append(target)

    for state in states:
        operating, spares, repairing = state
        # An operating unit fails: a spare takes over if the switch works.
        if operating:
            fails = operating * lam * _rate(rates, operating)
            if spares:
                add(
                    state,
                    (operating, spares - 1, repairing + 1),
                    fails * switching,
                )
                add(
                    state,
                    (operating - 1, spares, repairing + 1),
                    fails * (1.0 - switching),
                )
            else:
                add(state, (operating - 1, spares, repairing + 1), fails)
        # A spare fails in standby, and is repaired.
        if spares:
            add(
                state,
                (operating, spares - 1, repairing + 1),
                spares * dormancy * lam,
            )
        # A repaired unit fills an empty position, or joins the spares.
        if repairing:
            target = (
                (operating + 1, spares, repairing - 1)
                if operating < operate
                else (operating, spares + 1, repairing - 1)
            )
            busy = repairing if crews is None else min(repairing, crews)
            add(state, target, busy * mu)
    return states, moves


def chain(
    units: int,
    k: int,
    lam: float,
    mu: float,
    dormancy: float = 0.0,
    switching: float = 1.0,
    crews: Optional[int] = None,
    rates: Optional[Sequence[float]] = None,
) -> StandbyChain:
    """The Markov chain of a group of ``units`` exponential units, ``k``
    needed, and its long-run distribution (see ``long_run`` for the
    parameters).

    Returns
    -------
    StandbyChain
        Its states, generator and long-run distribution, and in each state
        whether the group is up, and the rates of its failures and its
        units'.
    """
    states, moves = _transitions(
        units, k, lam, mu, dormancy, switching, crews, rates
    )
    index = {state: i for i, state in enumerate(states)}
    size = len(states)
    generator = np.zeros((size, size))
    for (source, target), rate in moves.items():
        generator[index[source], index[target]] += rate
    generator -= np.diag(generator.sum(axis=1))
    # The balance equations of the other states, for their probabilities
    # relative to the first's (every unit ready), which keeps small ones
    # precise (see _crew_chain).
    balance = generator.T
    relative = np.linalg.solve(balance[1:, 1:], -balance[1:, 0])
    probabilities = np.append(1.0, np.maximum(relative, 0.0))
    probabilities /= probabilities.sum()
    up = np.zeros(size)
    failures = np.zeros(size)
    unit_failures = np.zeros(size)
    for i, (operating, spares, _) in enumerate(states):
        unit_failures[i], failures[i], up[i] = _state_rates(
            operating, spares, k, lam, dormancy, switching, rates
        )
    return StandbyChain(
        states, generator, probabilities, up, failures, unit_failures
    )


def long_run(
    units: int,
    k: int,
    lam: float,
    mu: float,
    dormancy: float = 0.0,
    switching: float = 1.0,
    crews: Optional[int] = None,
    rates: Optional[Sequence[float]] = None,
) -> StandbyLongRun:
    """The long-run values of a group of ``units`` exponential units, ``k``
    needed (see the module docstring).

    Parameters
    ----------
    units, k : int
        The group's units, and how many must operate.
    lam, mu : float
        A unit's failure rate while it operates, and its repair rate.
    dormancy : float, optional
        A spare's failure rate as a fraction of ``lam``, by default 0.0.
    switching : float, optional
        The probability that switching onto a spare succeeds, by default
        1.0.
    crews : int, optional
        How many units can be repaired at once, by default None: all.
    rates : sequence of float, optional
        For a group with no spares (see the module docstring): the rate
        at which each operating unit uses up its life while ``s``
        operate, ``rates[s - 1]``. By default None: a standby group.

    Returns
    -------
    StandbyLongRun
        The group's availability, failure frequency, unit failure frequency
        and unavailability.
    """
    group = chain(units, k, lam, mu, dormancy, switching, crews, rates)
    availability = unavailability = failures = unit_failures = 0.0
    for state, p in zip(group.states, group.probabilities):
        operating, spares, _ = state
        units_failing, failing, up = _state_rates(
            operating, spares, k, lam, dormancy, switching, rates
        )
        unit_failures += p * units_failing
        if up:
            availability += p
            failures += p * failing
        else:
            unavailability += p
    return StandbyLongRun(
        availability, failures, unit_failures, unavailability
    )
