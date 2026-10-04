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
rest of the decision depends only on how many reached predecessors each
undecided vertex (and the output) already has, up to its ``k``, and on the
value of any component drawn in several places (a repeated node) whose
other appearances are still to come. So each such state is solved once: a
vertex is branched on only when enough of its predecessors are reached for
it to matter, a branch whose two outcomes agree is dropped, and equal
branches are shared. Counting reached predecessors, rather than listing
which of the decided vertices that still feed undecided ones (the
*frontier*) are reached, makes the decided vertices that feed the same
ones one state (#172): a meshed diagram of 35 nodes and 129 edges took 44
seconds rather than a fraction of one. The states still grow with the
frontier's width, not with the number of paths, so the order is chosen to
keep the frontier narrow (``order``).

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

from repyability.rbd import _compiled

# Value slots 0 and 1 of a plan: the system fails, the system works.
FAIL, WORK = 0, 1

#: The most work ``build`` does before it gives up (#172), in steps: each
#: vertex decided and each count it changes, and each state's counts and
#: decided variables copied, a step, a fifth of a microsecond or so, so
#: about five seconds (a meshed diagram of 60 nodes and 345 edges takes
#: ten million). A core that needs more is too meshed to work out
#: exactly, and is simulated (see ``modular.GraphStructure``). Raise it to
#: try harder.
STEP_LIMIT = 25_000_000


class TooLarge(NotImplementedError):
    """A core whose decision diagram needs more than ``STEP_LIMIT``
    steps."""


#: Whether ``build`` runs compiled (``_bdd_kernel``, with numba installed):
#: ``"auto"`` (the default) for a core whose diagram may be large (its
#: order's ``_cost`` at least ``COMPILED_COST``), True for every core, False
#: for none. The plan is the same either way, step for step.
COMPILED: Any = "auto"
#: The ``_cost`` from which ``"auto"`` compiles: below it the search takes a
#: few hundredths of a second in Python, less than loading numba does. (A
#: 10 by 20 grid's, ``2**17.4``, is built, with its first probabilities, in
#: 0.17 s in Python and 0.07 s compiled; a 12 by 24 grid's, ``2**19.9``, in
#: 0.9 s and 0.19 s.)
COMPILED_COST = 2.0**15
#: The bits of each of the two integers a compiled state is: the counts'
#: fields, and the values of the variables drawn in several places still to
#: come. A core whose states need more is built in Python.
COMPILED_WIDTH = 62


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
    same variable are one component, drawn in two places. Compiled when
    numba is installed and the diagram may be large (see ``COMPILED``),
    with the same plan.

    Returns
    -------
    tuple
        ``(steps, root)``, each step ``(variable, active slot, inactive
        slot)``; ``root`` is 0 or 1 if the system never or always works.
    """
    if COMPILED and _compiled.available():
        if COMPILED is True or (
            _cost(sequence, pred, succ, source, sink) >= COMPILED_COST
        ):
            plan = _compiled_build(
                sequence, pred, succ, k, source, sink, variable
            )
            if plan is not None:
                return plan
    return _build(sequence, pred, succ, k, source, sink, variable)


def _compiled_build(
    sequence: Sequence[int],
    pred: Mapping[int, Iterable[int]],
    succ: Mapping[int, Iterable[int]],
    k: Mapping[int, int],
    source: int,
    sink: int,
    variable: Mapping[int, Hashable],
) -> Optional[tuple]:
    """``_build``, compiled (``_bdd_kernel.build``): the same plan, step for
    step, and the same steps taken (``TooLarge`` past ``STEP_LIMIT``); None
    for a core whose states do not fit in two integers (``COMPILED_WIDTH``
    bits each).

    A state's counts are an integer's fields: each position (the output's
    is ``n``) has a field wide enough for its ``k``, from the step after its
    first predecessor is decided (from the start for the input's
    successors) until it is decided itself, and a position decided gives
    its field up to those that come in after it. A variable drawn in several
    places has a bit of a second integer, its value, from the step after
    its first appearance until its last. At each step every state lays its
    fields out alike, so equal states are equal integers."""
    from repyability.rbd import _bdd_kernel

    n = len(sequence)
    position = {v: i for i, v in enumerate(sequence)}
    position[sink] = n
    needs = [k[v] for v in sequence] + [k[sink]]
    if min(needs) < 1:
        return None
    later = [[position[w] for w in succ.get(v, ())] for v in sequence]
    from_input = [position[w] for w in succ.get(source, ())]
    # The step from which each position has a field (n + 1 for none).
    alive_from = [n + 1] * (n + 1)
    for w in from_input:
        alive_from[w] = 0
    for u, ends in enumerate(later):
        for w in ends:
            alive_from[w] = min(alive_from[w], u + 1)
    bits = [need.bit_length() for need in needs]
    names = list(dict.fromkeys(variable[v] for v in sequence))
    index = {name: j for j, name in enumerate(names)}
    variables = [index[variable[v]] for v in sequence]
    first = [n] * len(names)
    final = [-1] * len(names)
    for i, x in enumerate(variables):
        first[x] = min(first[x], i)
        final[x] = i
    # Lay the fields and the bits out, step by step, first fit: a field is
    # given up after its position is decided, and a bit after its
    # variable's last appearance.
    entering: List[List[int]] = [[] for _ in range(n + 2)]
    for w in range(n + 1):
        if alive_from[w] <= n:
            entering[alive_from[w]].append(w)
    starting: List[List[int]] = [[] for _ in range(n + 2)]
    ending: List[List[int]] = [[] for _ in range(n + 2)]
    for x in range(len(names)):
        if first[x] < final[x]:
            starting[first[x] + 1].append(x)
            ending[final[x] + 1].append(x)
    offset = [0] * (n + 1)
    name_bit = [0] * len(names)
    used = names_used = 0
    alive: List[List[int]] = []
    pending = [0] * (n + 1)
    current: List[int] = []
    held = 0
    for i in range(n + 1):
        if i and alive_from[i - 1] <= i - 1:
            used &= ~(((1 << bits[i - 1]) - 1) << offset[i - 1])
            current.remove(i - 1)
        for w in entering[i]:
            mask = (1 << bits[w]) - 1
            at = next(
                (
                    o
                    for o in range(COMPILED_WIDTH - bits[w] + 1)
                    if not used & (mask << o)
                ),
                None,
            )
            if at is None:
                return None
            offset[w] = at
            used |= mask << at
            current.append(w)
        alive.append(sorted(current))
        for x in ending[i]:
            names_used &= ~(1 << name_bit[x])
            held -= 1
        for x in starting[i]:
            at = next(
                (o for o in range(COMPILED_WIDTH) if not names_used >> o & 1),
                None,
            )
            if at is None:
                return None
            name_bit[x] = at
            names_used |= 1 << at
            held += 1
        pending[i] = held
    expiring: List[List[int]] = [[] for _ in range(n)]
    for x in range(len(names)):
        if first[x] < final[x]:
            expiring[final[x]].append(x)

    def packed(lists: List[List[int]]) -> tuple:
        pointer = np.zeros(len(lists) + 1, np.int64)
        pointer[1:] = np.cumsum([len(items) for items in lists])
        ids = np.array([u for items in lists for u in items], np.int64)
        return pointer, ids

    succ_ptr, succ_ids = packed(later)
    alive_ptr, alive_ids = packed(alive)
    expire_ptr, expire_ids = packed(expiring)
    out_var, out_active, out_inactive, root, too_large = _bdd_kernel.build(
        n,
        np.array(variables, np.int64),
        np.array(first, np.int64),
        np.array(final, np.int64),
        np.array(needs, np.int64),
        succ_ptr,
        succ_ids,
        np.array(from_input, np.int64),
        np.array(alive_from, np.int64),
        np.array(offset, np.int64),
        np.array(bits, np.int64),
        np.array(name_bit, np.int64),
        expire_ptr,
        expire_ids,
        alive_ptr,
        alive_ids,
        np.array(pending, np.int64),
        STEP_LIMIT,
    )
    if too_large:
        raise TooLarge(
            "The diagram is too meshed to work out exactly: its decision "
            f"diagram takes more than {STEP_LIMIT:,} steps "
            "(repyability.rbd.bdd.STEP_LIMIT)."
        )
    steps = [
        (names[x], a, b)
        for x, a, b in zip(
            out_var.tolist(), out_active.tolist(), out_inactive.tolist()
        )
    ]
    return steps, int(root)


def _build(
    sequence: Sequence[int],
    pred: Mapping[int, Iterable[int]],
    succ: Mapping[int, Iterable[int]],
    k: Mapping[int, int],
    source: int,
    sink: int,
    variable: Mapping[int, Hashable],
) -> tuple:
    """``build`` in Python."""
    n = len(sequence)
    position = {v: i for i, v in enumerate(sequence)}
    position[sink] = n
    # Each vertex's successors, as positions (the output's is n), and the k
    # of each position's vertex.
    later = {
        u: tuple(position[w] for w in succ.get(u, ()))
        for u in [source, *sequence]
    }
    needs = [k[v] for v in sequence] + [k[sink]]
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
    # The variables drawn in several places whose last appearance each
    # position is: only those are forgotten there.
    expiring: Dict[int, set] = {}
    for name, i in final.items():
        if first[name] < i:
            expiring.setdefault(i, set()).add(name)

    # A state's ``counts``: for each undecided position with a reached
    # predecessor, how many, up to its k, as sorted (position, count)
    # pairs. Each settling works on counts of its own, changed in place.
    def reach(u, counts: dict) -> None:
        """Count vertex ``u`` reached, in ``counts``."""
        taken[0] += len(later[u])
        for w in later[u]:
            counts[w] = min(counts.get(w, 0) + 1, needs[w])

    # The steps taken (see STEP_LIMIT).
    taken = [0]

    def settle(i: int, counts: dict, decided: frozenset):
        """Decide vertices from the ``i``-th while no branch is needed: the
        terminal reached (FAIL or WORK), or the state at which the next
        vertex must be branched on, as ``(i, counts, decided)``. ``counts``
        is changed."""
        while True:
            taken[0] += 1
            if counts.get(n, 0) >= needs[n]:
                return WORK
            if not counts or i == n:
                return FAIL
            v = sequence[i]
            name = variable[v]
            if first[name] == i and final[name] > i:
                # Decided here, used later.
                return (i, tuple(sorted(counts.items())), decided)
            if counts.get(i, 0) >= needs[i]:
                if (name, True) in decided:
                    works = True
                elif (name, False) in decided:
                    works = False
                else:
                    return (i, tuple(sorted(counts.items())), decided)
            else:
                works = False
            decided = _step(i, v, works, counts, decided)
            i += 1

    def _step(i, v, works, counts, decided) -> frozenset:
        """Decide the ``i``-th vertex, ``v``: ``counts`` changed, and the
        variables drawn in several places still to come kept."""
        if works:
            reach(v, counts)
        counts.pop(i, None)
        gone = expiring.get(i)
        if decided and gone:
            taken[0] += len(decided)
            decided = frozenset(
                (name, value) for name, value in decided if name not in gone
            )
        return decided

    def branches(state) -> list:
        """The states after the ``i``-th vertex's variable works, and
        fails (the vertex is reached only if enough predecessors are)."""
        i, pairs, decided = state
        taken[0] += 3 * len(pairs) + 3 * len(decided)
        v = sequence[i]
        name = variable[v]
        enough = dict(pairs).get(i, 0) >= needs[i]
        out = []
        for value in (True, False):
            counts = dict(pairs)
            remembered = decided
            if final[name] > i:
                remembered = decided | {(name, value)}
            remembered = _step(i, v, value and enough, counts, remembered)
            out.append(settle(i + 1, counts, remembered))
        return out

    slots: Dict[Any, int] = {}
    unique: Dict[tuple, int] = {}
    steps: List[tuple] = []

    def known(state) -> Optional[int]:
        if state == FAIL or state == WORK:
            return state
        return slots.get(state)

    counts: Dict[int, int] = {}
    reach(source, counts)
    start = settle(0, counts, frozenset())
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
        if taken[0] > STEP_LIMIT:
            raise TooLarge(
                "The diagram is too meshed to work out exactly: its "
                f"decision diagram takes more than {STEP_LIMIT:,} steps "
                "(repyability.rbd.bdd.STEP_LIMIT)."
            )
        if stack:
            stack[-1][2].append(slot)
        else:
            root = slot
    assert root is not None
    return steps, root


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
        # The structure is coherent, so a path set with the pivot failed is
        # one with it working: a minimal one with it working holds one with
        # it failed only by being it, which a lookup finds (#172).
        without = sets[inactive]
        known = set(without)
        sets.append(
            without + [s | bit[pivot] for s in sets[active] if s not in known]
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
