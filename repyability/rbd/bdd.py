"""A binary decision diagram of a diagram's core, built from its graph
(#102).

The exact engine reduces the series-parallel parts of a diagram to modules
and works out what is left, the *core*, by a Shannon decomposition of its
minimal path sets (``shannon.py``). In a meshed core (a chain of bridges, a
grid) the path sets multiply. This module builds the same kind of plan, a
decision diagram over the core's vertices, from the graph itself, without
listing them.

The core is acyclic, so its vertices are decided in a topological order. A
vertex is *reached* when it works and at least ``k`` of its predecessors
are reached (the input always is), and the system works when the output's
``k`` predecessors are. After the first ``i`` vertices of the order, the
rest of the decision depends only on which of the decided vertices that
still feed undecided ones (the *frontier*) are reached, and on the value of
any component drawn in several places (a repeated node) whose other
appearances are still to come. So each such state is solved once: a vertex
is branched on only when enough of its predecessors are reached for it to
matter, a branch whose two outcomes agree is dropped, and equal branches
are shared. The size of the diagram grows with the frontier's width, not
with the number of paths, so the order is chosen to keep the frontier
narrow (``order``).

The plan has the format of ``shannon._shannon_plan``'s: step ``i`` fills
value slot ``i + 2`` from its pivot's active and inactive branches, slots 0
and 1 being the system failing and working. Everything that replays a plan
(the probability, its complement, the gradient, the cut sets) works on it
unchanged.
"""

from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
)

import numpy as np

# Value slots 0 and 1 of a plan: the system fails, the system works.
FAIL, WORK = 0, 1


def order(
    vertices: Iterable[int],
    pred: Mapping[int, Iterable[int]],
    succ: Mapping[int, Iterable[int]],
    source: int,
    sink: int,
) -> List[int]:
    """A topological order of ``vertices`` that keeps the frontier narrow.

    Two orders are tried: a greedy one, which at each step takes the ready
    vertex (every predecessor decided) that leaves the smallest frontier,
    and the breadth-first one. The one whose frontiers would give the
    smaller diagram (the sum over the steps of two to the frontier's size)
    is kept."""
    vertices = sorted(vertices)
    candidates = [_greedy(vertices, pred, succ, source, sink)]
    candidates.append(_breadth_first(vertices, pred, succ, source))
    return min(candidates, key=lambda o: _cost(o, pred, succ, source, sink))


def _frontier_steps(
    sequence: Sequence[int], succ: Mapping, source: int, sink: int
) -> Iterable[int]:
    """The size of the frontier after each vertex of ``sequence``."""
    position = {v: i for i, v in enumerate(sequence)}
    end = len(sequence)
    last: Dict[int, int] = {}
    for u in [source, *sequence]:
        ends = [end if w == sink else position[w] for w in succ.get(u, ())]
        last[u] = max(ends, default=-1)
    closing: Dict[int, int] = {}
    for u, i in last.items():
        closing[i] = closing.get(i, 0) + 1
    size = 1 if last[source] >= 0 else 0
    for i, v in enumerate(sequence):
        if last[v] > i:
            size += 1
        size -= closing.get(i, 0)
        yield size


def _cost(sequence, pred, succ, source, sink) -> float:
    return float(
        sum(
            2.0 ** min(s, 60)
            for s in _frontier_steps(sequence, succ, source, sink)
        )
    )


def _breadth_first(vertices, pred, succ, source) -> List[int]:
    waiting = {v: sum(1 for u in pred[v] if u != source) for v in vertices}
    ready = [v for v in vertices if waiting[v] == 0]
    out: List[int] = []
    head = 0
    while head < len(ready):
        v = ready[head]
        head += 1
        out.append(v)
        for w in sorted(succ.get(v, ())):
            if w in waiting:
                waiting[w] -= 1
                if waiting[w] == 0:
                    ready.append(w)
    return out


def _greedy(vertices, pred, succ, source, sink) -> List[int]:
    waiting = {v: sum(1 for u in pred[v] if u != source) for v in vertices}
    # How many successors each vertex still feeds that are undecided (the
    # output stays undecided to the end).
    feeding = {
        u: sum(1 for w in succ.get(u, ()) if w != sink)
        for u in [source, *vertices]
    }
    to_sink = {u for u in [source, *vertices] if sink in succ.get(u, ())}
    ready = {v for v in vertices if waiting[v] == 0}
    out: List[int] = []
    while ready:

        def score(v: int) -> tuple:
            # The frontier's change: v joins it (it feeds something), and
            # each predecessor whose last undecided successor v is leaves.
            closes = sum(
                1 for u in pred[v] if feeding[u] == 1 and u not in to_sink
            )
            return (1 - closes, v)

        v = min(ready, key=score)
        ready.discard(v)
        out.append(v)
        for u in pred[v]:
            feeding[u] -= 1
        for w in succ.get(v, ()):
            if w in waiting:
                waiting[w] -= 1
                if waiting[w] == 0:
                    ready.add(w)
    return out


def build(
    sequence: Sequence[int],
    pred: Mapping[int, Iterable[int]],
    succ: Mapping[int, Iterable[int]],
    k: Mapping[int, int],
    source: int,
    sink: int,
    variable: Mapping[int, Hashable],
) -> tuple:
    """The decision diagram of the core whose vertices are ``sequence``, a
    topological order, as a plan (see the module docstring). ``variable``
    names the random variable each vertex stands for: two vertices with the
    same variable are one component, drawn in two places.

    Returns
    -------
    tuple
        ``(steps, root)``, each step ``(variable, active slot, inactive
        slot)``; ``root`` is 0 or 1 if the system never or always works.
    """
    n = len(sequence)
    position = {v: i for i, v in enumerate(sequence)}
    last: Dict[int, int] = {}
    for u in [source, *sequence]:
        ends = [n if w == sink else position[w] for w in succ.get(u, ())]
        last[u] = max(ends, default=-1)
    # Where each variable appears first, and last: a variable drawn in
    # several places is decided at its first appearance, so that every
    # variable is decided at the same point on every branch (the diagram
    # is then ordered, and reduced it is canonical), and kept until its
    # last.
    first: Dict[Hashable, int] = {}
    final: Dict[Hashable, int] = {}
    for i, v in enumerate(sequence):
        first.setdefault(variable[v], i)
        final[variable[v]] = i
    preds = [tuple(pred[v]) for v in sequence]
    needs = [k[v] for v in sequence]
    sink_preds, sink_k = tuple(pred[sink]), k[sink]

    def settle(i: int, reached: frozenset, decided: frozenset):
        """Decide vertices from the ``i``-th while no branch is needed: the
        terminal reached (FAIL or WORK), or the state at which the next
        vertex must be branched on, as ``(i, reached, decided)``."""
        while True:
            if sum(1 for u in sink_preds if u in reached) >= sink_k:
                return WORK
            if not reached or i == n:
                return FAIL
            v = sequence[i]
            name = variable[v]
            if first[name] == i and final[name] > i:
                return (i, reached, decided)  # decided here, used later
            reaching = sum(1 for u in preds[i] if u in reached)
            if reaching >= needs[i]:
                value = _decided(decided, name)
                if value is None:
                    return (i, reached, decided)
                works = value
            else:
                works = False
            reached, decided = _step(i, v, works, reached, decided)
            i += 1

    def _step(i, v, works, reached, decided):
        kept = {u for u in reached if last[u] > i}
        if works and last[v] > i:
            kept.add(v)
        if decided:
            decided = frozenset(
                (name, value) for name, value in decided if final[name] > i
            )
        return frozenset(kept), decided

    def branches(state) -> list:
        """The states after the ``i``-th vertex's variable works, and
        fails (the vertex is reached only if enough predecessors are)."""
        i, reached, decided = state
        v = sequence[i]
        name = variable[v]
        enough = sum(1 for u in preds[i] if u in reached) >= needs[i]
        out = []
        for value in (True, False):
            remembered = decided
            if final[name] > i:
                remembered = decided | {(name, value)}
            after, remembered = _step(
                i, v, value and enough, reached, remembered
            )
            out.append(settle(i + 1, after, remembered))
        return out

    slots: Dict[Any, int] = {}
    unique: Dict[tuple, int] = {}
    steps: List[tuple] = []

    def known(state) -> Optional[int]:
        if state == FAIL or state == WORK:
            return state
        return slots.get(state)

    start = settle(
        0,
        frozenset([source]) if last[source] >= 0 else frozenset(),
        frozenset(),
    )
    root = known(start)
    stack: list = []
    if root is None:
        stack.append([start, branches(start), []])
    while stack:
        state, children, solved = stack[-1]
        if len(solved) < 2:
            child = children[len(solved)]
            slot = known(child)
            if slot is None:
                stack.append([child, branches(child), []])
            else:
                solved.append(slot)
            continue
        stack.pop()
        name = variable[sequence[state[0]]]
        active, inactive = solved
        if active == inactive:
            slot = active  # the vertex cannot change the outcome here
        else:
            key = (name, active, inactive)
            slot = unique.get(key)
            if slot is None:
                steps.append(key)
                slot = unique[key] = len(steps) + 1
        slots[state] = slot
        if stack:
            stack[-1][2].append(slot)
        else:
            root = slot
    assert root is not None
    return steps, root


def _decided(decided: frozenset, name) -> Optional[bool]:
    for known_name, value in decided:
        if known_name == name:
            return value
    return None


def pivots(plan: tuple) -> List[Hashable]:
    """The variables a plan branches on, in the order they first appear."""
    return list(dict.fromkeys(pivot for pivot, _, _ in plan[0]))


def path_sets(plan: tuple) -> set:
    """The minimal path sets of the structure a plan decides: the sets of
    variables whose working makes it work, with no smaller one. With the
    pivot working the structure is as its active branch, failed as its
    inactive one; the inactive branch's sets work either way, and the
    active branch's, with the pivot, unless they hold one of those."""
    steps, root = plan
    variables = pivots(plan)
    # Sets as bitmasks (a bit per variable): a union or a subset test is
    # one integer operation.
    bit = {v: 1 << i for i, v in enumerate(variables)}
    sets: List[List[int]] = [[], [0]]
    for pivot, active, inactive in steps:
        without = sets[inactive]
        sets.append(
            without
            + [
                s | bit[pivot]
                for s in sets[active]
                if not any(other & s == other for other in without)
            ]
        )
    return {
        frozenset(v for v in variables if mask & bit[v]) for mask in sets[root]
    }


def lifetime(plan: tuple, lifetimes: Sequence[Any], size: int) -> np.ndarray:
    """The system's lifetime in each of ``size`` samples, from each
    variable's lifetime in ``lifetimes`` (by variable): a structure that
    works with a variable working up to its lifetime, and as without it
    after, lasts ``max(min(variable's, with it), without it)``."""
    steps, root = plan
    values: List[Any] = [np.full(size, -np.inf), np.full(size, np.inf)]
    for pivot, active, inactive in steps:
        values.append(
            np.maximum(
                np.minimum(lifetimes[pivot], values[active]), values[inactive]
            )
        )
    return np.asarray(values[root], dtype=float)


def walk(plan: tuple, values: Sequence[Any]) -> bool:
    """Whether the structure works, given whether each variable works (by
    variable): one branch per decision, from the root to an outcome."""
    steps, slot = plan
    while slot > WORK:
        pivot, active, inactive = steps[slot - 2]
        slot = active if values[pivot] else inactive
    return slot == WORK
