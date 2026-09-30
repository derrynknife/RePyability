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
solved directly.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

#: A state: the units operating, the spares, and the units under repair.
State = Tuple[int, int, int]


class StandbyLongRun(NamedTuple):
    """A group's long-run values: the fraction of time it is up, its failures
    (up to down) per unit time, and its units' failures per unit time (the
    repairs it pays for)."""

    availability: float
    failure_frequency: float
    unit_failure_frequency: float


def _transitions(
    units: int,
    k: int,
    lam: float,
    mu: float,
    dormancy: float,
    switching: float,
    crews: Optional[int],
) -> Tuple[List[State], Dict[Tuple[State, State], float]]:
    """The states reached from the one with ``k`` units operating and the
    rest spares, and the rates between them."""
    start: State = (k, units - k, 0)
    states = [start]
    seen = {start}
    rates: Dict[Tuple[State, State], float] = {}

    def add(source: State, target: State, rate: float) -> None:
        if rate <= 0.0 or target == source:
            return
        rates[(source, target)] = rates.get((source, target), 0.0) + rate
        if target not in seen:
            seen.add(target)
            states.append(target)

    for state in states:
        operating, spares, repairing = state
        # An operating unit fails: a spare takes over if the switch works.
        if operating:
            fails = operating * lam
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
                if operating < k
                else (operating, spares + 1, repairing - 1)
            )
            busy = repairing if crews is None else min(repairing, crews)
            add(state, target, busy * mu)
    return states, rates


def long_run(
    units: int,
    k: int,
    lam: float,
    mu: float,
    dormancy: float = 0.0,
    switching: float = 1.0,
    crews: Optional[int] = None,
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

    Returns
    -------
    StandbyLongRun
        The group's availability, failure frequency and unit failure
        frequency.
    """
    states, rates = _transitions(units, k, lam, mu, dormancy, switching, crews)
    index = {state: i for i, state in enumerate(states)}
    size = len(states)
    generator = np.zeros((size, size))
    for (source, target), rate in rates.items():
        generator[index[source], index[target]] += rate
    generator -= np.diag(generator.sum(axis=1))
    # The balance equations of the other states, for their probabilities
    # relative to the first's (every unit ready), which keeps small ones
    # precise (see _crew_chain).
    balance = generator.T
    relative = np.linalg.solve(balance[1:, 1:], -balance[1:, 0])
    probabilities = np.append(1.0, np.maximum(relative, 0.0))
    probabilities /= probabilities.sum()
    availability = failures = unit_failures = 0.0
    for state, p in zip(states, probabilities):
        operating, spares, _ = state
        unit_failures += p * (operating + spares * dormancy) * lam
        if operating == k:
            availability += p
            # Up to down: an operating unit fails, and no spare takes over.
            takeover = switching if spares else 0.0
            failures += p * k * lam * (1.0 - takeover)
    return StandbyLongRun(availability, failures, unit_failures)
