"""The Markov chain of components that wait for repair crews (#90).

With fewer ``repair_crews`` than components, a ``RepairableRBD``'s
components no longer fail and recover independently: a component that fails
while every crew is busy waits for one (see ``_Crews``). With exponential
lives and repairs the system is then a continuous-time Markov chain. Its
state is which components are under repair and which are waiting, in the
order the crews will take them: by priority, then first come first served.
The rest are up. Each state says which components are up, so the long-run
values are the system's, state by state, averaged over the chain's
long-run distribution.

A repair that is ``"instant"`` takes no time once a crew starts it. A
component that fails while a crew is free is never down, and one that waits
comes back up the moment a crew reaches it. That crew then takes the next
job.

The chain is enumerated from the state with every component up, and its
balance equations are solved directly: sparse equations for each other
state's probability relative to that one's. They are eliminated from the
states with the most components down to those with the fewest, pivoting on
the diagonal, which is stable as the equations are diagonally dominant by
columns. That order fills in far less than general-purpose orderings, and
every probability keeps its relative precision however small it is: a
first-come-first-served chain of seven components and one crew, 13,700
states, is solved in about a second, to 1e-13 relative error when its
repairs are 1e4 times as fast as its failures.
"""

import math
from typing import Any, Dict, Hashable, List, NamedTuple, Sequence, Tuple

import numpy as np

#: The most states the chain is solved for.
MAX_STATES = 15_000

#: A state: the components under repair (sorted), and those waiting, in the
#: order the crews will take them.
State = Tuple[Tuple[int, ...], Tuple[int, ...]]


class CrewChain(NamedTuple):
    """The states of a crew chain and their long-run probabilities.

    ``up[s, i]`` says whether ``nodes[i]`` is up in state ``s``, and
    ``probabilities[s]`` is the long-run fraction of time in state ``s``.
    ``states`` are the states themselves (the first with every component
    up), and ``generator`` the rates between them (a sparse matrix, ``[s,
    t]`` the rate from ``s`` to ``t``), for the chain over time (see
    ``_chain_transient``). ``transitions`` are its transitions one by one,
    ``(sources, targets, rates, components)``: each a component's failure
    or the end of its repair (with the jobs the crew then takes that are
    done at once), labelled with that component's position (#199).
    """

    nodes: Tuple[Hashable, ...]
    up: np.ndarray
    probabilities: np.ndarray
    states: Tuple[State, ...] = ()
    generator: Any = None
    transitions: Any = None

    def availability(self, node) -> float:
        """The long-run probability that ``node`` is up."""
        column = self.up[:, self.nodes.index(node)]
        return float(self.probabilities @ column)

    def split(self, vector: np.ndarray) -> np.ndarray:
        """``Q_i v`` for each component ``i`` (columns), ``Q_i`` the part of
        the generator made of its transitions: ``Q = sum_i Q_i``, so that
        ``p(t) Q_i v`` is the part of ``d/dt p(t) v`` the component's
        failures and repairs make (#199)."""
        sources, targets, rates, components = self.transitions
        out = np.zeros((len(self.probabilities), len(self.nodes)))
        vector = np.asarray(vector, dtype=float)
        np.add.at(
            out,
            (sources, components),
            rates * (vector[targets] - vector[sources]),
        )
        return out


def _queues(r: int) -> int:
    """How many queues ``r`` components can form: the ordered selections of
    any number of them."""
    return sum(math.perm(r, w) for w in range(r + 1))


def state_count(
    priorities: Sequence[float], instant: Sequence[bool], crews: int
) -> int:
    """The number of states of the chain, counted without building it.

    In a state either nothing waits, and at most ``crews`` of the components
    whose repairs take time are under repair; or every crew is busy with
    such a repair, and some of the other components wait, in any order
    within each priority.

    Parameters
    ----------
    priorities : Sequence[float]
        Each component's priority.
    instant : Sequence[bool]
        Whether each component's repair is instant.
    crews : int
        The number of crews.

    Returns
    -------
    int
        The number of states.
    """
    timed = sum(1 for now in instant if not now)
    count = sum(math.comb(timed, k) for k in range(min(crews, timed) + 1))
    if timed < crews:
        return count
    classes: Dict[float, List[int]] = {}
    for priority, now in zip(priorities, instant):
        sizes = classes.setdefault(priority, [0, 0])
        sizes[0] += not now
        sizes[1] += 1
    # ways[b]: how the classes so far can keep b crews busy, with the rest
    # of each class up or waiting in any order.
    ways = {0: 1}
    for timed_here, size in classes.values():
        grown: Dict[int, int] = {}
        for busy, total in ways.items():
            for k in range(min(timed_here, crews - busy) + 1):
                grown[busy + k] = grown.get(busy + k, 0) + (
                    total * math.comb(timed_here, k) * _queues(size - k)
                )
        ways = grown
    # Every crew busy with nothing waiting was counted above.
    return count + ways.get(crews, 0) - math.comb(timed, crews)


def _enumerate(
    failure_rates: Sequence[float],
    repair_rates: Sequence[float],
    priorities: Sequence[float],
    crews: int,
) -> Tuple[List[State], List[int], List[int], List[float], List[int]]:
    """The states reached from the one with every component up, and the
    transitions between them: sources, targets, rates, and the component
    whose failure or repair each is."""
    instant = [math.isinf(rate) for rate in repair_rates]
    start: State = ((), ())
    index = {start: 0}
    states = [start]
    sources: List[int] = []
    targets: List[int] = []
    rates: List[float] = []
    components: List[int] = []

    def reach(source: int, state: State, rate: float, component: int) -> None:
        target = index.get(state)
        if target is None:
            target = index[state] = len(states)
            states.append(state)
        sources.append(source)
        targets.append(target)
        rates.append(rate)
        components.append(component)

    source = 0
    while source < len(states):
        serving, queue = states[source]
        down = set(serving).union(queue)
        for i, rate in enumerate(failure_rates):
            if i in down:
                continue
            if len(serving) < crews:
                if instant[i]:
                    continue  # repaired at once: nothing changes
                reach(source, (tuple(sorted(serving + (i,))), queue), rate, i)
                continue
            # Behind every waiting job of the same priority or higher.
            place = len(queue)
            while place and priorities[queue[place - 1]] < priorities[i]:
                place -= 1
            reach(
                source,
                (serving, queue[:place] + (i,) + queue[place:]),
                rate,
                i,
            )
        for i in serving:
            rest = tuple(j for j in serving if j != i)
            waiting = queue
            # The crew takes the next job, and an instant one is done at
            # once, so it goes on to the one after.
            while waiting:
                head, waiting = waiting[0], waiting[1:]
                if not instant[head]:
                    rest = tuple(sorted(rest + (head,)))
                    break
            reach(source, (rest, waiting), repair_rates[i], i)
        source += 1
    return states, sources, targets, rates, components


def solve(
    nodes: Sequence[Hashable],
    failure_rates: Sequence[float],
    repair_rates: Sequence[float],
    priorities: Sequence[float],
    crews: int,
) -> CrewChain:
    """The chain of ``nodes`` with these failure and repair rates (``inf``
    for an instant repair) and priorities (higher first), sharing ``crews``
    crews, and its long-run distribution (see the module docstring).

    Parameters
    ----------
    nodes : Sequence[Hashable]
        The components.
    failure_rates, repair_rates : Sequence[float]
        Each component's failure rate, and repair rate.
    priorities : Sequence[float]
        Each component's priority for a crew.
    crews : int
        The number of crews.

    Returns
    -------
    CrewChain
        The states, which components are up in each, and their long-run
        probabilities.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    states, sources, targets, rates, components = _enumerate(
        failure_rates, repair_rates, priorities, crews
    )
    size = len(states)
    up = np.ones((size, len(nodes)), dtype=bool)
    for s, (serving, queue) in enumerate(states):
        up[s, list(serving + queue)] = False
    generator = sparse.csr_matrix(
        (np.array(rates, dtype=float), (sources, targets)), shape=(size, size)
    )
    generator = (
        generator - sparse.diags(np.asarray(generator.sum(axis=1)).ravel())
    ).tocsr()
    transitions = (
        np.array(sources, dtype=np.intp),
        np.array(targets, dtype=np.intp),
        np.array(rates, dtype=float),
        np.array(components, dtype=np.intp),
    )
    if size == 1:
        return CrewChain(
            tuple(nodes),
            up,
            np.ones(1),
            tuple(states),
            generator,
            transitions,
        )
    # The most components down first; within a level, by the queue. The
    # state with every component up, the only one with none down, is last.
    order = sorted(
        range(size),
        key=lambda s: (
            -len(states[s][0]) - len(states[s][1]),
            states[s][1],
            states[s][0],
        ),
    )
    position = np.empty(size, dtype=np.intp)
    position[order] = np.arange(size)
    source = position[np.array(sources, dtype=np.intp)]
    target = position[np.array(targets, dtype=np.intp)]
    rate = np.array(rates, dtype=float)
    outflow = np.bincount(source, weights=rate, minlength=size)
    # Row s: the flow into state s, less the flow out of it.
    balance = sparse.csc_matrix(
        (rate, (target, source)), shape=(size, size)
    ) - sparse.diags(outflow)
    last = size - 1
    relative = splu(
        sparse.csc_matrix(balance[:last, :last]),
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
    ).solve(-balance[:last, last].toarray().ravel())
    weights = np.append(np.maximum(relative, 0.0), 1.0)
    probabilities = np.empty(size)
    probabilities[order] = weights / weights.sum()
    return CrewChain(
        tuple(nodes), up, probabilities, tuple(states), generator, transitions
    )
