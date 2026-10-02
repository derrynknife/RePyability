"""Common-cause groups in a ``RepairableRBD`` (#136): which of a group's
members are down together, in the long run.

A group's members are identical units with a constant failure rate. Its
model (a ``BetaFactor`` or ``MGL``, see ``ccf.py``) splits that rate
between causes: each member's own, and shared ones, each a set of members
that one cause fails at once, with a share of the rate (``_causes``). Each
cause strikes as a Poisson process, and fails the members it names that
are up (those already down stay down). So each member on its own still
fails at its rate while it is up, and its own long-run values are as
without the group: what the group changes is which members are down
together.

- Members whose failures are revealed, each repaired at a constant rate,
  form a continuous-time Markov chain over which of them are down. Its
  long-run distribution is found by the GTH algorithm (Grassmann, Taksar
  and Heyman), which only adds, multiplies and divides positive numbers,
  so a small probability keeps its precision.
- Members whose failures are hidden are found by their tests, at their
  own offsets and intervals: a test finds a failure with the group's
  coverage (all the failures one cause makes are found alike, or missed
  alike), and a full test finds every one. A member is up, down until
  its next test, or down until its next full test (missed). Between tests
  the members only go down, and at a test the member tested is repaired
  if its failure is found; the state at any time is worked out by
  uniformization, a sum of positive terms again. Once every member has
  had a full test, the state no longer depends on where it started (each
  member is up after its full test, and from then on its state depends
  only on the causes and its own tests), so one period of the tests from
  all up reaches the long run, and a second gives the state at each time.
"""

from itertools import product
from typing import Hashable, List, NamedTuple, Sequence, Tuple

import numpy as np

#: The most states a group's chain may have.
MAX_STATES = 20_000

UP, FOUND, MISSED = 0, 1, 2


class GroupStates(NamedTuple):
    """Which of a group's ``members`` are down in each combination
    (``down``, one row per combination, every combination of the members
    up or down), and each combination's probability at each time
    (``probabilities``, one row per time, adding up to 1)."""

    members: Tuple[Hashable, ...]
    down: np.ndarray
    probabilities: np.ndarray


def causes(model, members: Sequence[Hashable], rate: float) -> List[tuple]:
    """The group's causes: ``(member positions, rate)``, each member's own
    first, then the shared ones (see ``ccf._Model._causes``)."""
    own, shared = model._causes(tuple(members))
    position = {member: i for i, member in enumerate(members)}
    out: List[Tuple[Tuple[int, ...], float]] = [
        ((i,), own * rate) for i in range(len(members))
    ]
    out += [
        (tuple(sorted(position[m] for m in struck)), fraction * rate)
        for struck, fraction in shared
        if fraction > 0.0
    ]
    return out


def _states(n: int, levels: int) -> np.ndarray:
    """Every state of ``n`` members with ``levels`` levels each, one row
    per state (its index read in base ``levels``, the first member
    last)."""
    return np.array(list(product(range(levels), repeat=n)), dtype=np.int8)[
        :, ::-1
    ]


def _index(states: np.ndarray, levels: int) -> np.ndarray:
    """Each state's index (see ``_states``)."""
    return states.astype(np.int64) @ (levels ** np.arange(states.shape[1]))


def _strikes(
    states: np.ndarray, levels: int, struck: Tuple[int, ...], level: int
) -> np.ndarray:
    """The state each state goes to when a cause fails the members
    ``struck`` that are up, to ``level``."""
    after = states.copy()
    for i in struck:
        after[:, i] = np.where(after[:, i] == UP, level, after[:, i])
    return _index(after, levels)


def _combinations(states: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Every combination of ``n`` members up or down (``down``, in the
    order of ``_states(n, 2)``), and the combination of each state."""
    down = _states(n, 2).astype(bool)
    return down, _index((states != UP).astype(np.int8), 2)


def revealed(
    model, members: Sequence[Hashable], rate: float, repair: float
) -> GroupStates:
    """The long-run states of a group whose members' failures are
    revealed, each repaired at rate ``repair`` (independently: as many
    repairs at once as are needed)."""
    n = len(members)
    states = _states(n, 2)
    size = len(states)
    _check_size(size, members)
    flow = np.zeros((size, size))
    source = np.arange(size)
    for struck, cause in causes(model, members, rate):
        target = _strikes(states, 2, struck, FOUND)
        moves = target != source
        np.add.at(flow, (source[moves], target[moves]), cause)
    for i in range(n):
        down = states[:, i] == FOUND
        fixed = states.copy()
        fixed[:, i] = UP
        np.add.at(flow, (source[down], _index(fixed, 2)[down]), repair)
    stationary = gth(flow)
    combinations, _ = _combinations(states, n)
    return GroupStates(tuple(members), combinations, stationary[None, :])


def gth(flow: np.ndarray) -> np.ndarray:
    """The long-run distribution of the continuous-time Markov chain with
    the rates ``flow[i, j]`` from state ``i`` to state ``j`` (its diagonal
    ignored), by the GTH algorithm: states are folded into those before
    them, one at a time from the last, and the distribution unfolded."""
    rates = np.array(flow, dtype=float)
    np.fill_diagonal(rates, 0.0)
    size = len(rates)
    for last in range(size - 1, 0, -1):
        out = rates[last, :last].sum()
        if out <= 0.0:
            raise ValueError("The chain does not leave one of its states.")
        rates[:last, last] /= out
        rates[:last, :last] += np.outer(rates[:last, last], rates[last, :last])
    weights = np.zeros(size)
    weights[0] = 1.0
    for j in range(1, size):
        weights[j] = weights[:j] @ rates[:j, j]
    return weights / weights.sum()


class ProofTest(NamedTuple):
    """A member's test: when, which member, and whether it is a full test
    (one that finds a failure it would otherwise miss)."""

    time: float
    member: int
    full: bool


def hidden(
    model,
    members: Sequence[Hashable],
    rate: float,
    coverage: float,
    tests: Sequence[ProofTest],
    period: float,
    times: np.ndarray,
) -> GroupStates:
    """The long-run states, at each of ``times`` (in ``[0, period)``), of a
    group whose members' failures are hidden, found by ``tests`` (each
    member's, in ``(0, period]``, the schedule repeating every
    ``period``), with ``coverage`` the chance that a test finds a cause's
    failures (a full test finds all)."""
    n = len(members)
    partial = coverage < 1.0
    levels = 3 if partial else 2
    states = _states(n, levels)
    size = len(states)
    _check_size(size, members)
    # The uniformized chain: a step of it is the chance of each move in a
    # time 1 / pace.
    source = np.arange(size)
    moves: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for struck, cause in causes(model, members, rate):
        for level, share in ((FOUND, coverage), (MISSED, 1.0 - coverage)):
            if share <= 0.0 or (level == MISSED and not partial):
                continue
            target = _strikes(states, levels, struck, level)
            moved = target != source
            moves.append((source[moved], target[moved], cause * share))
    outflow = np.zeros(size)
    for start, _, value in moves:
        np.add.at(outflow, start, value)
    pace = float(outflow.max()) if len(moves) else 0.0
    step = np.zeros((size, size))
    if pace > 0.0:
        for start, end, value in moves:
            np.add.at(step, (start, end), value / pace)
    np.fill_diagonal(step, 1.0 - outflow / pace if pace > 0.0 else 1.0)
    # Each member's tests: where each state goes.
    found = []
    for i in range(n):
        after = {}
        for full in (False, True):
            fixed = states.copy()
            mask = fixed[:, i] == FOUND
            if full:
                mask |= fixed[:, i] == MISSED
            fixed[mask, i] = UP
            after[full] = _index(fixed, levels)
        found.append(after)

    def evolve(v: np.ndarray, dt: float) -> np.ndarray:
        # v exp(G dt), by uniformization, in steps short enough for exp(-x)
        # not to underflow.
        while dt > 0.0 and pace > 0.0:
            piece = min(dt, 8.0 / pace)
            dt -= piece
            x = pace * piece
            weight = np.exp(-x)
            term = v
            out = weight * term
            k = 0
            while True:
                k += 1
                term = term @ step
                weight *= x / k
                out = out + weight * term
                if k > x and weight < 1e-34:
                    break
            v = out
        return v

    def tested_at(v: np.ndarray, test: ProofTest) -> np.ndarray:
        target = found[test.member][test.full or not partial]
        return np.bincount(target, weights=v, minlength=size)

    order = sorted(tests)
    # From all up, one period reaches the long run (every member has had a
    # full test by its end).
    v = np.zeros(size)
    v[0] = 1.0
    now = 0.0
    for test in order:
        v = tested_at(evolve(v, test.time - now), test)
        now = test.time
    v = evolve(v, period - now)
    # A second period, to each time (tests at a time come before it).
    out = np.empty((len(times), size))
    sequence = np.argsort(times, kind="stable")
    now, upcoming = 0.0, 0
    for row in sequence:
        t = float(times[row])
        while upcoming < len(order) and order[upcoming].time <= t:
            test = order[upcoming]
            v = tested_at(evolve(v, test.time - now), test)
            now = test.time
            upcoming += 1
        v = evolve(v, t - now)
        now = t
        out[row] = v
    combinations, combination = _combinations(states, n)
    probabilities = np.zeros((len(times), len(combinations)))
    for j in range(len(combinations)):
        probabilities[:, j] = out[:, combination == j].sum(axis=1)
    return GroupStates(tuple(members), combinations, probabilities)


def _check_size(size: int, members) -> None:
    if size > MAX_STATES:
        raise NotImplementedError(
            f"The common-cause group of {list(members)} has {size} joint "
            f"states, more than the {MAX_STATES} worked out exactly: split "
            "it, or leave it out."
        )
