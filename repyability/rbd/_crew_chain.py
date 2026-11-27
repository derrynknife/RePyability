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

With a call-out crew (``call_when``, a ``RepairableRBD``'s ``call_out``),
a job that makes that many wait brings one more crew, which takes the
first waiting job at once; while it is engaged there is a crew more, and
it leaves at the first end of a job that leaves a crew with none to take
(the job it has then, if any, is finished by the crew that came free:
the crews are alike). So it is engaged exactly in the states with a job
more under repair than there are crews, and the chain needs no more of a
state. ``calls`` is the rate at which it is called in each state and
``engaged`` those states, for its fee and its rate while engaged.

With common-cause groups (``shared``, #251), a shared cause strikes at its
rate and fails the components it names that are up, at once, as the
simulation's does (``_event_loop._strike``): they join the queue one after
another in the diagram's order, each as a failure of its own would, which
is the order the simulation's event queue takes failures at one instant
in (``_events._EventQueue``), however many the cause strikes. A
component of a group fails on its own at its own cause's rate, which is
what ``failure_rates`` then holds.

A component may stand for several alike copies (``copies``, for the
redundancy allocation's designs): a state then says how many of them are
under repair and waiting (the component as many times among those under
repair and in the queue), not which, as the copies are alike in rates and
priority. Each copy up fails at the component's rate and each under
repair is repaired at its rate, so the chain is that of the copies told
apart, with the states that differ only by which copy is where taken
together.

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
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability.rbd._phase_type import Phases, exponential

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
    done at once), labelled with that component's position, or a shared
    cause's strike, labelled with ``len(nodes)`` plus its group's place
    (``groups`` of them). ``calls`` is
    the rate at which a call-out crew is called in each state, and
    ``engaged`` whether it is engaged there (see the module docstring):
    zeros without one.
    """

    nodes: Tuple[Hashable, ...]
    up: np.ndarray
    probabilities: Any
    states: Tuple[State, ...] = ()
    generator: Any = None
    transitions: Any = None
    calls: Any = None
    engaged: Any = None
    groups: int = 0
    #: How many of each component's copies are up in each state (1 or 0
    #: without copies): ``up`` says whether any is.
    working: Any = None

    def availability(self, node) -> float:
        """The long-run probability that ``node`` is up."""
        column = self.up[:, self.nodes.index(node)]
        return float(self.probabilities @ column)

    def split(self, vector: np.ndarray) -> np.ndarray:
        """``Q_i v`` for each component ``i`` (columns), ``Q_i`` the part of
        the generator made of its transitions: ``Q = sum_i Q_i``, so that
        ``p(t) Q_i v`` is the part of ``d/dt p(t) v`` the component's
        failures and repairs make; then a column for each common-cause
        group, of its shared causes' strikes."""
        sources, targets, rates, components = self.transitions
        out = np.zeros(
            (len(self.probabilities), len(self.nodes) + self.groups)
        )
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
    priorities: Sequence[float],
    instant: Sequence[bool],
    crews: int,
    call_when: int = 0,
    copies: Optional[Sequence[int]] = None,
) -> int:
    """The number of states of the chain, counted without building it.

    In a state either nothing waits, and at most ``crews`` of the components
    whose repairs take time are under repair; or every crew is busy with
    such a repair, and some of the other components wait, in any order
    within each priority. With a call-out crew (``call_when``) or copies
    of components (``copies``) the states are counted as they are
    reached, up to ``MAX_STATES + 1``: more than that many are not told
    apart.

    Parameters
    ----------
    priorities : Sequence[float]
        Each component's priority.
    instant : Sequence[bool]
        Whether each component's repair is instant.
    crews : int
        The number of crews.
    call_when : int, optional
        How many waiting jobs call the call-out crew (0: there is none).
    copies : Sequence[int], optional
        How many alike copies each component stands for (one each, by
        default).

    Returns
    -------
    int
        The number of states.
    """
    if call_when or copies is not None:
        return len(
            _enumerate(
                [1.0] * len(instant),
                [math.inf if now else 1.0 for now in instant],
                priorities,
                crews,
                call_when,
                MAX_STATES + 1,
                (),
                copies,
            )[0]
        )
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


def _phased(rate) -> Phases:
    """A rate as the one phase of its exponential time; phases as given."""
    return rate if isinstance(rate, Phases) else exponential(rate)


def _enumerate(
    failure_rates: Sequence[Any],
    repair_rates: Sequence[Any],
    priorities: Sequence[float],
    crews: int,
    call_when: int = 0,
    limit: int = 0,
    shared: Sequence[Tuple[Tuple[int, ...], float, int]] = (),
    copies: Optional[Sequence[int]] = None,
    start: State = ((), ()),
) -> Tuple[
    List[State],
    List[int],
    List[int],
    List[float],
    List[int],
    List[int],
    List[Tuple[int, ...]],
]:
    """The states reached from ``start`` (by default the one with every
    component up), and the
    transitions between them: sources, targets, rates, the component
    whose failure or repair each is (or, for a strike of one of the
    ``shared`` causes, each ``(components struck, rate, group)``, the
    number of components plus its group), how many times it calls
    the call-out crew (``call_when`` waiting jobs do; see the module
    docstring), and each state's phases. With a ``limit``, it stops once
    it has reached that many states. With ``copies``, each component
    stands for that many alike ones (see the module docstring). A rate
    is that of an exponential time, or a time's ``Phases``."""
    lives = [_phased(rate) for rate in failure_rates]
    repairs = [_phased(rate) for rate in repair_rates]
    instant = [math.isinf(repair.rates[0]) for repair in repairs]
    count = len(lives)
    # A state here is (under repair, waiting, each component's phase).
    first = (start[0], start[1], (0,) * count)
    index = {first: 0}
    states = [first]
    sources: List[int] = []
    targets: List[int] = []
    rates: List[float] = []
    components: List[int] = []
    calls: List[int] = []

    def reach(
        source: int,
        state: tuple,
        rate: float,
        component: int,
        call: int = 0,
    ) -> None:
        target = index.get(state)
        if target is None:
            target = index[state] = len(states)
            states.append(state)
        sources.append(source)
        targets.append(target)
        rates.append(rate)
        components.append(component)
        calls.append(call)

    def at(phases: tuple, i: int, phase: int) -> tuple:
        return phases[:i] + (phase,) + phases[i + 1 :]

    def taken(serving: tuple, waiting: tuple, phases: tuple) -> tuple:
        """A crew takes the next waiting job, and an instant one is done
        at once, so it goes on to the one after."""
        while waiting:
            head, waiting = waiting[0], waiting[1:]
            if not instant[head]:
                return (
                    tuple(sorted(serving + (head,))),
                    waiting,
                    at(phases, head, repairs[head].fresh),
                )
            phases = at(phases, head, lives[head].fresh)
        return serving, waiting, phases

    def failed(state: tuple, i: int) -> Tuple[tuple, int]:
        """The state once component ``i``, up in ``state``, has failed,
        and whether that called the call-out crew."""
        serving, queue, phases = state
        if len(serving) < crews:
            if instant[i]:
                # Repaired at once: a new life starts.
                return (serving, queue, at(phases, i, lives[i].fresh)), 0
            return (
                tuple(sorted(serving + (i,))),
                queue,
                at(phases, i, repairs[i].fresh),
            ), 0
        # Behind every waiting job of the same priority or higher.
        place = len(queue)
        while place and priorities[queue[place - 1]] < priorities[i]:
            place -= 1
        waiting = queue[:place] + (i,) + queue[place:]
        phases = at(phases, i, 0)
        if call_when and len(serving) == crews and len(waiting) >= call_when:
            # The call-out crew comes, and takes the first of them.
            return taken(serving, waiting, phases), 1
        return (serving, waiting, phases), 0

    alike = [1] * count if copies is None else list(copies)
    source = 0
    while source < len(states) and not (limit and len(states) >= limit):
        state = states[source]
        serving, queue, phases = state
        # How many of each component's copies are up.
        up = list(alike)
        for i in serving + queue:
            up[i] -= 1
        for i, life in enumerate(lives):
            if not up[i]:
                continue
            phase = phases[i]
            rate, ends = life.rates[phase], life.ends[phase]
            after, call = failed(state, i)
            if ends and (after != state or call):
                reach(source, after, up[i] * rate * ends, i, call)
            if ends < 1.0 and not life.last(phase):
                reach(
                    source,
                    (serving, queue, at(phases, i, phase + 1)),
                    rate * (1.0 - ends),
                    i,
                )
        for struck, rate, group in shared:
            # Those up fail one after another, in the diagram's order.
            after, called = state, 0
            for i in struck:
                if up[i]:
                    after, call = failed(after, i)
                    called += call
            if after != state or called:
                reach(source, after, rate, count + group, called)
        for i in sorted(set(serving)):
            # One of its copies under repair is done.
            place = serving.index(i)
            rest = serving[:place] + serving[place + 1 :]
            phase = phases[i]
            rate, ends = repairs[i].rates[phase], repairs[i].ends[phase]
            # The crew takes the next job; with none, the call-out crew
            # (engaged with a job more than the crews under repair) goes.
            if ends:
                reach(
                    source,
                    taken(rest, queue, at(phases, i, lives[i].fresh)),
                    serving.count(i) * rate * ends,
                    i,
                )
            if ends < 1.0 and not repairs[i].last(phase):
                reach(
                    source,
                    (serving, queue, at(phases, i, phase + 1)),
                    rate * (1.0 - ends),
                    i,
                )
        source += 1
    return (
        [state[:2] for state in states],
        sources,
        targets,
        rates,
        components,
        calls,
        [state[2] for state in states],
    )


def solve(
    nodes: Sequence[Hashable],
    failure_rates: Sequence[Any],
    repair_rates: Sequence[Any],
    priorities: Sequence[float],
    crews: int,
    call_when: int = 0,
    shared: Sequence[Tuple[Tuple[int, ...], float, int]] = (),
    copies: Optional[Sequence[int]] = None,
    start: Optional[State] = None,
    limit: int = 0,
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
    call_when : int, optional
        How many waiting jobs call a call-out crew (0, the default: there
        is none).
    shared : sequence, optional
        The common-cause groups' shared causes, each ``(positions of the
        components it strikes, in the order they fail in; its rate; its group's
        place)``; ``failure_rates`` are then the components' own causes'.
    copies : Sequence[int], optional
        How many alike copies each component stands for (one each, by
        default): ``working`` then counts those up, and ``up`` says
        whether any is. Not with ``shared`` causes.
    start : State, optional
        The state the chain is followed from, for a chain over time
        alone: its long-run distribution is then not worked out
        (``probabilities`` is None), and ``states[0]`` is ``start`` with
        every component in the phase its time at the start begins in. A
        rate may then be a time's ``Phases`` (see ``_phase_type``).
    limit : int, optional
        With ``start``: stop once this many states are reached (0: no
        limit); the chain is then not whole.

    Returns
    -------
    CrewChain
        The states, which components are up in each, and their long-run
        probabilities.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    states, sources, targets, rates, components, called, _ = _enumerate(
        failure_rates,
        repair_rates,
        priorities,
        crews,
        call_when,
        limit,
        shared,
        copies,
        ((), ()) if start is None else start,
    )
    size = len(states)
    groups = 1 + max((group for _, _, group in shared), default=-1)
    calls = np.bincount(
        np.array(sources, dtype=np.intp),
        weights=np.array(rates, dtype=float) * np.array(called, dtype=float),
        minlength=size,
    )
    engaged = np.array([len(serving) > crews for serving, _ in states])
    working = np.tile(
        (
            np.ones(len(nodes), dtype=np.intp)
            if copies is None
            else np.asarray(copies, dtype=np.intp)
        ),
        (size, 1),
    )
    for s, (serving, queue) in enumerate(states):
        np.subtract.at(working[s], list(serving + queue), 1)
    up = working > 0
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
    if size == 1 or start is not None:
        return CrewChain(
            tuple(nodes),
            up,
            np.ones(1) if start is None else None,
            tuple(states),
            generator,
            transitions,
            calls,
            engaged,
            groups,
            working,
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
        tuple(nodes),
        up,
        probabilities,
        tuple(states),
        generator,
        transitions,
        calls,
        engaged,
        groups,
        working,
    )
