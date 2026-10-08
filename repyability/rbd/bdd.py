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


#: Whether ``build``'s search runs compiled (``_bdd_kernel``, with numba
#: installed): ``"auto"`` (the default) for a core whose diagram may be
#: large (its order's ``_cost`` at least ``COMPILED_COST``), True for every
#: core, False for none. It is one search either way, so one plan.
COMPILED: Any = "auto"
#: The ``_cost`` from which ``"auto"`` compiles: below it the search takes
#: less time in Python than loading numba's compiled search does.
COMPILED_COST = 2.0**15


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
    same variable are one component, drawn in two places. The search
    (``search``) runs compiled when numba is installed and the diagram may
    be large (see ``COMPILED``), and as Python otherwise: the same search,
    so the same plan.

    Returns
    -------
    tuple
        ``(steps, root)``, each step ``(variable, active slot, inactive
        slot)``; ``root`` is 0 or 1 if the system never or always works.
    """
    compiled = bool(COMPILED) and _compiled.available()
    if COMPILED is not True and compiled:
        compiled = _cost(sequence, pred, succ, source, sink) >= COMPILED_COST
    core = _layout(
        sequence, succ, k, source, sink, variable, WORD if compiled else None
    )
    if core is None:
        return [], FAIL
    names, arguments = core
    if compiled:
        from repyability.rbd import _bdd_kernel

        made = _bdd_kernel.run_search(arguments, STEP_LIMIT)
    else:
        made = _run_search(arguments, STEP_LIMIT)
    out_var, out_active, out_inactive, root, too_large = made
    if too_large:
        raise TooLarge(
            "The diagram is too meshed to work out exactly: its decision "
            f"diagram takes more than {STEP_LIMIT:,} steps "
            "(repyability.rbd.bdd.STEP_LIMIT)."
        )
    steps = [
        (names[int(x)], int(a), int(b))
        for x, a, b in zip(out_var, out_active, out_inactive)
    ]
    return steps, int(root)


def _layout(
    sequence: Sequence[int],
    succ: Mapping[int, Iterable[int]],
    k: Mapping[int, int],
    source: int,
    sink: int,
    variable: Mapping[int, Hashable],
    width: Optional[int],
) -> Optional[tuple]:
    """The core as ``search`` takes it: its variables' names, and the
    lists describing it. None for a core whose output needs no reached
    predecessor (which ``modular`` never builds).

    A state is the counts of reached predecessors (each up to its ``k``) of
    the positions still to be decided, and the values of the variables
    drawn in several places still to come, as fields of words of ``width``
    bits (one word of any width for ``None``): each position (the output's
    is ``n``) has a field wide enough for its ``k``, from the step after
    its first predecessor is decided (from the start for the input's
    successors) until it is decided itself; a variable drawn in several
    places has a bit, from the step after its first appearance until its
    last. A field is given up when it ends, to those that come in after it.
    At each step every state lays its fields out alike (first fit), so
    equal states are equal words."""
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
    # Each variable, numbered by its first appearance: decided there, so
    # that every variable is decided at the same point on every branch (the
    # diagram is then ordered, and reduced it is canonical), and kept until
    # its last.
    names = list(dict.fromkeys(variable[v] for v in sequence))
    index = {name: j for j, name in enumerate(names)}
    variables = [index[variable[v]] for v in sequence]
    first = [n] * len(names)
    final = [-1] * len(names)
    for i, x in enumerate(variables):
        first[x] = min(first[x], i)
        final[x] = i
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
    word = [0] * (n + 1)
    offset = [0] * (n + 1)
    name_word = [0] * len(names)
    name_bit = [0] * len(names)
    used: List[int] = []
    alive: List[List[int]] = []
    pending = [0] * (n + 1)
    current: List[int] = []
    held = 0
    for i in range(n + 1):
        if i and alive_from[i - 1] <= i - 1:
            used[word[i - 1]] &= ~(((1 << bits[i - 1]) - 1) << offset[i - 1])
            current.remove(i - 1)
        for x in ending[i]:
            used[name_word[x]] &= ~(1 << name_bit[x])
            held -= 1
        for w in entering[i]:
            word[w], offset[w] = _first_fit(used, bits[w], width)
            current.append(w)
        for x in starting[i]:
            name_word[x], name_bit[x] = _first_fit(used, 1, width)
            held += 1
        alive.append(sorted(current))
        pending[i] = held
    words = max(1, len(used))
    # Each step's count fields, as masks of each word: whether any count is
    # left (the values held are not counts).
    counted = [0] * ((n + 1) * words)
    for i, fields in enumerate(alive):
        for w in fields:
            counted[i * words + word[w]] |= ((1 << bits[w]) - 1) << offset[w]
    expiring: List[List[int]] = [[] for _ in range(n)]
    for x in range(len(names)):
        if first[x] < final[x]:
            expiring[final[x]].append(x)

    def packed(lists: List[List[int]]) -> tuple:
        pointer = [0]
        for items in lists:
            pointer.append(pointer[-1] + len(items))
        return pointer, [u for items in lists for u in items]

    succ_ptr, succ_ids = packed(later)
    alive_ptr, alive_ids = packed(alive)
    expire_ptr, expire_ids = packed(expiring)
    return names, (
        n,
        variables,
        first,
        final,
        needs,
        succ_ptr,
        succ_ids,
        from_input,
        alive_from,
        word,
        offset,
        bits,
        name_word,
        name_bit,
        expire_ptr,
        expire_ids,
        alive_ptr,
        alive_ids,
        pending,
        counted,
        words,
    )


def _first_fit(used: List[int], size: int, width: Optional[int]) -> tuple:
    """The first free place for a field of ``size`` bits in the words
    ``used`` (their taken bits), ``(word, offset)``, taken; a new word if
    none has room (one word of any width for ``width`` None)."""
    mask = (1 << size) - 1
    for w, taken in enumerate(used):
        at = 0
        while width is None or at + size <= width:
            if not taken & (mask << at):
                used[w] |= mask << at
                return w, at
            at += 1
    used.append(mask)
    return len(used) - 1, 0


#: The bits of a word of a compiled search's state (see ``_layout``): a
#: field never straddles two, and a word stays a non-negative int64.
WORD = 62

# What a settling comes to (see ``search``): a state to branch on, or an
# outcome.
_STATE, _FAILS, _WORKS = 0, 1, 2


def _run_search(arguments: tuple, limit: int) -> tuple:
    """``search`` in Python, its working memory as lists, a dict and
    arrays of Python's integers."""
    n, W = arguments[0], arguments[-1]
    rows = n + 3
    memory: tuple = (
        {},
        np.full(64, -1, dtype=np.int64),
        {},
        [],
        [],
        [],
        [0] * (2 * rows),
        [0] * (2 * rows),
        [0] * (2 * rows),
        [0] * (2 * rows * W),
        [0] * rows,
        [0] * rows,
        [0] * (rows * W),
        [0] * (2 * rows),
        [0] * rows,
        [0] * W,
    )
    return search(*arguments, *(limit, *memory))


def search(
    n,
    variable,
    first,
    final,
    needs,
    succ_ptr,
    succ_ids,
    from_input,
    alive_from,
    word,
    offset,
    bits,
    name_word,
    name_bit,
    expire_ptr,
    expire_ids,
    alive_ptr,
    alive_ids,
    pending,
    counted,
    W,
    limit,
    interned,
    slots,
    unique,
    out_var,
    out_active,
    out_inactive,
    kid_kind,
    kid_i,
    kid_id,
    kid_words,
    state_i,
    state_id,
    state_words,
    solved,
    count,
    words,
):
    """The core's decision diagram (see ``_layout`` for the core's
    arguments, the module docstring for the search): its steps into
    ``out_var`` (each variable's number), ``out_active`` and
    ``out_inactive``; its root; and whether it took more than ``limit``
    steps (it then stops short).

    The vertices are decided in their order. From a state, those that need
    no branch are decided at once (*settled*): a vertex too few of whose
    predecessors are reached is decided as not working, and a variable
    drawn in several places takes its value from its first appearance.
    Settling stops at an outcome (the output reached, or no count left to
    reach it) or at a vertex to branch on: one whose variable appears there
    first and again later, or a reached one whose variable's value is still
    to choose. Each state is solved once: it is numbered by its step and
    its words (``interned``, a word at a time), and its slot kept by that
    number (``slots``, grown as needed). Each step is made once
    (``unique``), and a branch whose two outcomes agree is dropped.

    The steps taken (``limit``) count each vertex settled, each successor
    a working vertex reaches and each held value when a variable is
    forgotten; and when a state is branched on, three for each of its
    counts that is not 0 and for each value it holds. This one function
    runs as Python (``_run_search``: one word, a Python integer of any
    width) and compiled (``_bdd_kernel``: words of ``WORD`` bits), with
    the same states and steps, so the same plan.

    The working memory, for each state on the stack (``n + 2`` at most,
    over a base row whose one branch is the start): its two branches
    (``kid_*``: an outcome, or a state's step, number and words), its step,
    number and words, the slots its branches have come to and how many;
    and the state being settled (``words``)."""
    made = 0
    taken = len(from_input)
    next_id = 0
    # The base row: its one branch is the start, the input reached (each
    # of its successors' counts 1; each needs 1 at least), settled from
    # position 0.
    for j in range(W):
        words[j] = 0
    for w in from_input:
        words[word[w]] |= 1 << offset[w]
    count[0] = 0
    top = 1
    row = 0
    target = 0
    start = 0
    forced = -1
    second = False
    while True:
        # Settle the state in ``words`` from position ``start``: for a
        # branch, first deciding it with value ``forced``.
        i = start
        kind = _STATE
        while True:
            x = variable[i] if i < n else 0
            if forced < 0:
                taken += 1
                if (
                    alive_from[n] <= i
                    and ((words[word[n]] >> offset[n]) & ((1 << bits[n]) - 1))
                    >= needs[n]
                ):
                    kind = _WORKS
                    break
                left = False
                for j in range(W):
                    if words[j] & counted[i * W + j]:
                        left = True
                if not left or i == n:
                    kind = _FAILS
                    break
                if first[x] == i and final[x] > i:
                    break
                works = False
                if (
                    alive_from[i] <= i
                    and ((words[word[i]] >> offset[i]) & ((1 << bits[i]) - 1))
                    >= needs[i]
                ):
                    if first[x] == i:
                        break
                    works = (words[name_word[x]] >> name_bit[x]) & 1 == 1
            else:
                works = forced == 1 and (
                    alive_from[i] <= i
                    and ((words[word[i]] >> offset[i]) & ((1 << bits[i]) - 1))
                    >= needs[i]
                )
            # Decide position i: its field goes, and each value forgotten
            # here (the fields coming in may take their bits); if it works,
            # each successor's count goes up (to its k at most); and a
            # variable whose first appearance it is and works is held.
            if alive_from[i] <= i:
                words[word[i]] &= ~(((1 << bits[i]) - 1) << offset[i])
            if expire_ptr[i + 1] > expire_ptr[i]:
                taken += pending[i]
                for e in range(expire_ptr[i], expire_ptr[i + 1]):
                    y = expire_ids[e]
                    words[name_word[y]] &= ~(1 << name_bit[y])
            if works:
                taken += succ_ptr[i + 1] - succ_ptr[i]
                for s in range(succ_ptr[i], succ_ptr[i + 1]):
                    w = succ_ids[s]
                    mask = (1 << bits[w]) - 1
                    c = (words[word[w]] >> offset[w]) & mask
                    if c < needs[w]:
                        c += 1
                    words[word[w]] = (
                        words[word[w]] & ~(mask << offset[w])
                    ) | (c << offset[w])
            if forced == 1 and final[x] > i:
                words[name_word[x]] |= 1 << name_bit[x]
            forced = -1
            i += 1
        # What it settled to: an outcome, or a state, numbered (by its
        # step, then each of its words, in turn).
        kid_kind[target] = kind
        kid_i[target] = i
        if kind == _STATE:
            here = -1 - i
            for j in range(W):
                key = (here, words[j])
                found = interned.get(key, -1)
                if found < 0:
                    found = next_id
                    next_id += 1
                    interned[key] = found
                here = found
            kid_id[target] = here
            for j in range(W):
                kid_words[target * W + j] = words[j]
        if second:
            # The state pushed: its second branch, its variable failing.
            second = False
            target += 1
            for j in range(W):
                words[j] = state_words[row * W + j]
            start = state_i[row]
            forced = 0
            continue
        # Work down the stack until a state must be pushed (its branches
        # to settle) or the start is solved.
        while True:
            t = top - 1
            if count[t] < (1 if t == 0 else 2):
                c = 2 * t + count[t]
                if kid_kind[c] == _FAILS:
                    slot = FAIL
                elif kid_kind[c] == _WORKS:
                    slot = WORK
                elif kid_id[c] < slots.size:
                    slot = slots[kid_id[c]]
                else:
                    slot = -1
                if slot >= 0:
                    solved[c] = slot
                    count[t] += 1
                    continue
                # Push the state, and settle its first branch, its
                # variable working (its second follows).
                row = top
                top += 1
                p = kid_i[c]
                state_i[row] = p
                state_id[row] = kid_id[c]
                nonzero = 0
                for j in range(W):
                    words[j] = kid_words[c * W + j]
                    state_words[row * W + j] = words[j]
                for a in range(alive_ptr[p], alive_ptr[p + 1]):
                    w = alive_ids[a]
                    if (words[word[w]] >> offset[w]) & ((1 << bits[w]) - 1):
                        nonzero += 1
                taken += 3 * nonzero + 3 * pending[p]
                count[row] = 0
                target = 2 * row
                start = p
                forced = 1
                second = True
                break
            if t == 0:
                return out_var, out_active, out_inactive, solved[0], False
            # Both branches solved: the state's slot.
            x = variable[state_i[t]]
            active, inactive = solved[2 * t], solved[2 * t + 1]
            if active == inactive:
                slot = active  # the vertex cannot change the outcome here
            else:
                key3 = (x, active, inactive)
                slot = unique.get(key3, -1)
                if slot < 0:
                    out_var.append(x)
                    out_active.append(active)
                    out_inactive.append(inactive)
                    made += 1
                    slot = made + 1
                    unique[key3] = slot
            number = state_id[t]
            if number >= slots.size:
                grown = np.full(max(2 * slots.size, number + 1), -1, np.int64)
                grown[: slots.size] = slots
                slots = grown
            slots[number] = slot
            if taken > limit:
                return out_var, out_active, out_inactive, -1, True
            top -= 1
            u = top - 1
            solved[2 * u + count[u]] = slot
            count[u] += 1


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
