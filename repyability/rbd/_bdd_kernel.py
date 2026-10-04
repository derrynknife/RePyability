"""The decision diagram of a core (``bdd.py``), compiled (numba): its
construction, and the replay of a plan for probabilities.

``bdd._build`` decides a core's vertices in a topological order, solving
each state (how many reached predecessors each undecided vertex has, up to
its ``k``, and the values of the repeated components still to come) once
(#172). ``build`` runs the same search over integers: a state's counts are
the fields of one, laid out alike for every state at a step (a position's
field, wide enough for its ``k``, from the step after its first
predecessor is decided until it is decided itself; ``bdd._compiled_build``
lays them out), and the values of the repeated components pending the bits
of another. It decides in the same order and counts the same steps
(``bdd.STEP_LIMIT``), so the plan is the same, step for step, and so is
whether the core is too meshed.

``replay`` and ``value_and_gradient`` are ``modular.Decomposition._core_value``
and ``shannon._shannon_value_and_gradient`` for arrays of probabilities: the
same products and sums, in the same order, so the same values to the last
bit.

Importing this module imports numba, which compiles them on first use (or
loads them from numba's cache).
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict

# What ``_settle`` comes to: a state to branch on, or an outcome.
STATE, FAIL, WORK = 0, 1, 2

_KEY = types.UniTuple(types.int64, 3)


@njit(cache=True, inline="always")
def _count(key, w, i, alive_from, offset, bits):
    """Position ``w``'s count in the counts ``key`` of step ``i`` (0 before
    it has a field)."""
    if i < alive_from[w]:
        return 0
    return (key >> offset[w]) & ((1 << bits[w]) - 1)


@njit(cache=True)
def _decide(
    i,
    works,
    key,
    d,
    taken,
    needs,
    succ_ptr,
    succ_ids,
    alive_from,
    offset,
    bits,
    name_bit,
    expire_ptr,
    expire_ids,
    pending,
):
    """``bdd._build``'s ``_step``: the counts and values after position
    ``i`` is decided (it works or not), and the steps taken. (A variable
    expires only at its own last appearance, so ``branches`` never adds
    one at a position where another expires.)"""
    # The position's field goes first: one coming in may take its bits.
    if alive_from[i] <= i:
        key &= ~(((1 << bits[i]) - 1) << offset[i])
    if works:
        taken += succ_ptr[i + 1] - succ_ptr[i]
        for s in range(succ_ptr[i], succ_ptr[i + 1]):
            w = succ_ids[s]
            mask = (1 << bits[w]) - 1
            c = (key >> offset[w]) & mask
            if c < needs[w]:
                c += 1
            key = (key & ~(mask << offset[w])) | (c << offset[w])
    if expire_ptr[i + 1] > expire_ptr[i]:
        taken += pending[i]
        for e in range(expire_ptr[i], expire_ptr[i + 1]):
            d &= ~(1 << name_bit[expire_ids[e]])
    return key, d, taken


@njit(cache=True)
def _settle(
    i,
    key,
    d,
    taken,
    n,
    variable,
    first,
    final,
    needs,
    succ_ptr,
    succ_ids,
    alive_from,
    offset,
    bits,
    name_bit,
    expire_ptr,
    expire_ids,
    pending,
):
    """``bdd._build``'s ``settle``: decide positions from the ``i``-th while
    no branch is needed; ``(kind, i, key, d, taken)``."""
    while True:
        taken += 1
        if _count(key, n, i, alive_from, offset, bits) >= needs[n]:
            return WORK, 0, 0, 0, taken
        if key == 0 or i == n:
            return FAIL, 0, 0, 0, taken
        x = variable[i]
        if first[x] == i and final[x] > i:
            return STATE, i, key, d, taken
        works = 0
        if _count(key, i, i, alive_from, offset, bits) >= needs[i]:
            if first[x] < i:
                works = (d >> name_bit[x]) & 1
            else:
                return STATE, i, key, d, taken
        key, d, taken = _decide(
            i,
            works,
            key,
            d,
            taken,
            needs,
            succ_ptr,
            succ_ids,
            alive_from,
            offset,
            bits,
            name_bit,
            expire_ptr,
            expire_ids,
            pending,
        )
        i += 1


@njit(cache=True)
def _branches(
    i,
    key,
    d,
    taken,
    kids,
    row,
    n,
    variable,
    first,
    final,
    needs,
    succ_ptr,
    succ_ids,
    alive_from,
    offset,
    bits,
    name_bit,
    expire_ptr,
    expire_ids,
    pending,
    alive_ptr,
    alive_ids,
):
    """``bdd._build``'s ``branches``: the states after position ``i``'s
    variable works and fails, into ``kids[row]``; the steps taken."""
    nonzero = 0
    for a in range(alive_ptr[i], alive_ptr[i + 1]):
        w = alive_ids[a]
        if (key >> offset[w]) & ((1 << bits[w]) - 1):
            nonzero += 1
    taken += 3 * nonzero + 3 * pending[i]
    x = variable[i]
    enough = _count(key, i, i, alive_from, offset, bits) >= needs[i]
    for b in range(2):
        value = 1 - b
        key2, d2, taken = _decide(
            i,
            value == 1 and enough,
            key,
            d,
            taken,
            needs,
            succ_ptr,
            succ_ids,
            alive_from,
            offset,
            bits,
            name_bit,
            expire_ptr,
            expire_ids,
            pending,
        )
        if value == 1 and final[x] > i:
            d2 |= 1 << name_bit[x]
        kind, i2, key3, d3, taken = _settle(
            i + 1,
            key2,
            d2,
            taken,
            n,
            variable,
            first,
            final,
            needs,
            succ_ptr,
            succ_ids,
            alive_from,
            offset,
            bits,
            name_bit,
            expire_ptr,
            expire_ids,
            pending,
        )
        kids[row, b, 0], kids[row, b, 1] = kind, i2
        kids[row, b, 2], kids[row, b, 3] = key3, d3
    return taken


@njit(cache=True)
def build(
    n,
    variable,
    first,
    final,
    needs,
    succ_ptr,
    succ_ids,
    from_input,
    alive_from,
    offset,
    bits,
    name_bit,
    expire_ptr,
    expire_ids,
    alive_ptr,
    alive_ids,
    pending,
    limit,
):
    """``bdd._build`` over the arrays ``bdd._compiled_build`` makes: the
    plan's steps (each variable's number, and its active and inactive
    slots), its root, and whether it took more than ``limit`` steps (the
    plan is then cut short)."""
    slots = Dict.empty(key_type=_KEY, value_type=types.int64)
    unique = Dict.empty(key_type=_KEY, value_type=types.int64)
    capacity = 1024
    out_var = np.empty(capacity, np.int64)
    out_active = np.empty(capacity, np.int64)
    out_inactive = np.empty(capacity, np.int64)
    made = 0
    # The stack: each state (its step and two integers), its two branches
    # (kind, step, integers) and the slots solved so far.
    depth = n + 2
    st = np.empty((depth, 3), np.int64)
    kids = np.empty((depth, 2, 4), np.int64)
    solved = np.empty((depth, 2), np.int64)
    count = np.zeros(depth, np.int64)
    top = 0
    root = -1

    # The input reached: each of its successors' counts 1 (each needs 1
    # at least).
    taken = from_input.size
    key = 0
    for w in from_input:
        key |= 1 << offset[w]
    kind, i, key, d, taken = _settle(
        0,
        key,
        0,
        taken,
        n,
        variable,
        first,
        final,
        needs,
        succ_ptr,
        succ_ids,
        alive_from,
        offset,
        bits,
        name_bit,
        expire_ptr,
        expire_ids,
        pending,
    )
    if kind == FAIL:
        root = 0
    elif kind == WORK:
        root = 1
    else:
        st[0, 0], st[0, 1], st[0, 2] = i, key, d
        count[0] = 0
        top = 1
        taken = _branches(
            i,
            key,
            d,
            taken,
            kids,
            0,
            n,
            variable,
            first,
            final,
            needs,
            succ_ptr,
            succ_ids,
            alive_from,
            offset,
            bits,
            name_bit,
            expire_ptr,
            expire_ids,
            pending,
            alive_ptr,
            alive_ids,
        )

    while top > 0:
        t = top - 1
        if count[t] < 2:
            b = count[t]
            ck, ci, cr, cd = (
                kids[t, b, 0],
                kids[t, b, 1],
                kids[t, b, 2],
                kids[t, b, 3],
            )
            if ck == FAIL:
                slot = 0
            elif ck == WORK:
                slot = 1
            else:
                slot = slots.get((ci, cr, cd), -1)
            if slot >= 0:
                solved[t, b] = slot
                count[t] = b + 1
                continue
            # Push the child, with its branches.
            st[top, 0], st[top, 1], st[top, 2] = ci, cr, cd
            count[top] = 0
            taken = _branches(
                ci,
                cr,
                cd,
                taken,
                kids,
                top,
                n,
                variable,
                first,
                final,
                needs,
                succ_ptr,
                succ_ids,
                alive_from,
                offset,
                bits,
                name_bit,
                expire_ptr,
                expire_ids,
                pending,
                alive_ptr,
                alive_ids,
            )
            top += 1
            continue
        # Both branches solved: the state's slot.
        si, sr, sd = st[t, 0], st[t, 1], st[t, 2]
        x = variable[si]
        active, inactive = solved[t, 0], solved[t, 1]
        if active == inactive:
            slot = active
        else:
            slot = unique.get((x, active, inactive), -1)
            if slot < 0:
                if made == capacity:
                    capacity *= 2
                    grown = np.empty(capacity, np.int64)
                    grown[:made] = out_var[:made]
                    out_var = grown
                    grown = np.empty(capacity, np.int64)
                    grown[:made] = out_active[:made]
                    out_active = grown
                    grown = np.empty(capacity, np.int64)
                    grown[:made] = out_inactive[:made]
                    out_inactive = grown
                out_var[made] = x
                out_active[made] = active
                out_inactive[made] = inactive
                made += 1
                slot = made + 1
                unique[(x, active, inactive)] = slot
        slots[(si, sr, sd)] = slot
        if taken > limit:
            return (
                out_var[:made],
                out_active[:made],
                out_inactive[:made],
                -1,
                True,
            )
        top -= 1
        if top > 0:
            u = top - 1
            solved[u, count[u]] = slot
            count[u] += 1
        else:
            root = slot
    return out_var[:made], out_active[:made], out_inactive[:made], root, False


#: The columns replayed at a time: the values of every slot for these
#: columns are kept (in a buffer the caller keeps, so that repeated calls
#: reuse its memory rather than touch fresh memory each time).
COLUMNS = 16


@njit(cache=True)
def replay(pivots, actives, inactives, root, works, fails, complement, values):
    """``Decomposition._core_value``: the plan's value for each column of
    the terms' probabilities of working (``works``, a row per term) and
    failing (``fails``), or with ``complement`` its probability of
    failing. ``values`` has a row per slot and ``COLUMNS`` columns."""
    steps = pivots.size
    m = works.shape[1]
    out = np.empty(m)
    for c0 in range(0, m, COLUMNS):
        c1 = min(c0 + COLUMNS, m)
        w = c1 - c0
        for j in range(w):
            values[0, j] = 1.0 if complement else 0.0
            values[1, j] = 0.0 if complement else 1.0
        for s in range(steps):
            p, a, b = pivots[s], actives[s], inactives[s]
            for j in range(w):
                values[s + 2, j] = (
                    works[p, c0 + j] * values[a, j]
                    + fails[p, c0 + j] * values[b, j]
                )
        for j in range(w):
            out[c0 + j] = values[root, j]
    return out


@njit(cache=True)
def value_and_gradient(
    pivots,
    actives,
    inactives,
    root,
    works,
    fails,
    terminal_fail,
    terminal_work,
    terms,
    values,
    adjoints,
):
    """``shannon._shannon_value_and_gradient``: the plan's value for each
    column, and its derivative with respect to each term's probability
    (a row per term, with whether the term has one: a term the reverse
    pass never reaches has none, as the original's dictionary leaves it
    out). Each adjoint and derivative starts at its first share, as the
    original's do (not at 0, which would turn a -0.0 into 0.0). ``values``
    and ``adjoints`` have a row per slot and ``COLUMNS`` columns."""
    steps = pivots.size
    m = works.shape[1]
    out = np.empty(m)
    gradient = np.zeros((terms, m))
    has_gradient = np.zeros(terms, np.bool_)
    has_adjoint = np.zeros(steps + 2, np.bool_)
    for c0 in range(0, m, COLUMNS):
        c1 = min(c0 + COLUMNS, m)
        w = c1 - c0
        for j in range(w):
            values[0, j] = terminal_fail
            values[1, j] = terminal_work
        for s in range(steps):
            p, a, b = pivots[s], actives[s], inactives[s]
            for j in range(w):
                values[s + 2, j] = (
                    works[p, c0 + j] * values[a, j]
                    + fails[p, c0 + j] * values[b, j]
                )
        for j in range(w):
            out[c0 + j] = values[root, j]
        has_adjoint[:] = False
        has_adjoint[root] = True
        for j in range(w):
            adjoints[root, j] = 1.0
        reached = np.zeros(terms, np.bool_)
        for s in range(steps - 1, -1, -1):
            if not has_adjoint[s + 2]:
                continue
            p, a, b = pivots[s], actives[s], inactives[s]
            for j in range(w):
                change = adjoints[s + 2, j] * (values[a, j] - values[b, j])
                if reached[p]:
                    gradient[p, c0 + j] = gradient[p, c0 + j] + change
                else:
                    gradient[p, c0 + j] = change
            reached[p] = True
            if a > 1:
                for j in range(w):
                    share = adjoints[s + 2, j] * works[p, c0 + j]
                    if has_adjoint[a]:
                        adjoints[a, j] = adjoints[a, j] + share
                    else:
                        adjoints[a, j] = share
                has_adjoint[a] = True
            if b > 1:
                for j in range(w):
                    share = adjoints[s + 2, j] * fails[p, c0 + j]
                    if has_adjoint[b]:
                        adjoints[b, j] = adjoints[b, j] + share
                    else:
                        adjoints[b, j] = share
                has_adjoint[b] = True
        for t in range(terms):
            if reached[t]:
                has_gradient[t] = True
    return out, gradient, has_gradient
