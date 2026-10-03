"""The decision diagram of a core (``bdd.py``), compiled (numba): its
construction, and the replay of a plan for probabilities.

``bdd.build`` decides a core's vertices in a topological order, solving
each state (which of the frontier's vertices are reached, and the values of
the repeated components still to come) once. At each step of the order the
vertices that can be in the frontier are the same on every branch, and so
are the repeated components already decided, so a state is three integers:
the step, a bit for each of those vertices that is reached, and a bit for
each of those components that works. ``build`` runs ``bdd.build``'s search
on them, in the same order, so the plan is the same, step for step.

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


@njit(cache=True)
def _reached(i, r, ids, pos_f):
    """How many of ``ids`` are reached in the state ``r`` of step ``i``."""
    count = 0
    for u in ids:
        p = pos_f[i, u]
        if p >= 0 and (r >> p) & 1:
            count += 1
    return count


@njit(cache=True)
def _step(
    i,
    works,
    r,
    d,
    last,
    final,
    pos_f,
    members_f,
    width_f,
    pos_r,
    members_r,
    width_r,
):
    """``bdd.build``'s ``_step``: the state after vertex ``i`` (which works
    or not), as step ``i + 1`` holds it (a repeated component decided here
    is added by the caller)."""
    kept = 0
    for j in range(width_f[i]):
        if (r >> j) & 1:
            u = members_f[i, j]
            if last[u] > i:
                kept |= 1 << pos_f[i + 1, u]
    if works and last[i] > i:
        kept |= 1 << pos_f[i + 1, i]
    decided = 0
    for j in range(width_r[i]):
        if (d >> j) & 1:
            x = members_r[i, j]
            if final[x] > i:
                decided |= 1 << pos_r[i + 1, x]
    return kept, decided


@njit(cache=True)
def _settle(
    i,
    r,
    d,
    n,
    variable,
    first,
    final,
    needs,
    pred_ptr,
    pred_ids,
    sink_preds,
    sink_k,
    last,
    pos_f,
    members_f,
    width_f,
    pos_r,
    members_r,
    width_r,
):
    """``bdd.build``'s ``settle``: decide vertices from the ``i``-th while
    no branch is needed; ``(kind, i, r, d)``."""
    while True:
        if _reached(i, r, sink_preds, pos_f) >= sink_k:
            return WORK, 0, 0, 0
        if r == 0 or i == n:
            return FAIL, 0, 0, 0
        x = variable[i]
        if first[x] == i and final[x] > i:
            return STATE, i, r, d
        reaching = _reached(
            i, r, pred_ids[pred_ptr[i] : pred_ptr[i + 1]], pos_f
        )
        works = 0
        if reaching >= needs[i]:
            p = pos_r[i, x]
            if p < 0:
                return STATE, i, r, d
            works = (d >> p) & 1
        r, d = _step(
            i,
            works,
            r,
            d,
            last,
            final,
            pos_f,
            members_f,
            width_f,
            pos_r,
            members_r,
            width_r,
        )
        i += 1


@njit(cache=True)
def build(
    n,
    variable,
    first,
    final,
    needs,
    pred_ptr,
    pred_ids,
    sink_preds,
    sink_k,
    last,
    pos_f,
    members_f,
    width_f,
    pos_r,
    members_r,
    width_r,
    start_r,
):
    """``bdd.build`` over the arrays ``bdd._compiled_build`` makes: the
    plan's steps (each variable's number, and its active and inactive
    slots) and its root."""
    slots = Dict.empty(key_type=_KEY, value_type=types.int64)
    unique = Dict.empty(key_type=_KEY, value_type=types.int64)
    capacity = 1024
    out_var = np.empty(capacity, np.int64)
    out_active = np.empty(capacity, np.int64)
    out_inactive = np.empty(capacity, np.int64)
    made = 0
    # The stack: each state (its step and two masks), its two branches
    # (kind, step, masks) and the slots solved so far.
    depth = n + 2
    st = np.empty((depth, 3), np.int64)
    kids = np.empty((depth, 2, 4), np.int64)
    solved = np.empty((depth, 2), np.int64)
    count = np.zeros(depth, np.int64)
    top = 0
    root = -1

    kind, i, r, d = _settle(
        0,
        start_r,
        0,
        n,
        variable,
        first,
        final,
        needs,
        pred_ptr,
        pred_ids,
        sink_preds,
        sink_k,
        last,
        pos_f,
        members_f,
        width_f,
        pos_r,
        members_r,
        width_r,
    )
    if kind == FAIL:
        root = 0
    elif kind == WORK:
        root = 1
    else:
        st[0, 0], st[0, 1], st[0, 2] = i, r, d
        count[0] = 0
        top = 1
        # Its branches (as for every state pushed below).
        for b in range(2):
            value = 1 - b
            x = variable[i]
            enough = (
                _reached(i, r, pred_ids[pred_ptr[i] : pred_ptr[i + 1]], pos_f)
                >= needs[i]
            )
            r2, d2 = _step(
                i,
                value and enough,
                r,
                d,
                last,
                final,
                pos_f,
                members_f,
                width_f,
                pos_r,
                members_r,
                width_r,
            )
            if value == 1 and final[x] > i:
                d2 |= 1 << pos_r[i + 1, x]
            k2, i2, r3, d3 = _settle(
                i + 1,
                r2,
                d2,
                n,
                variable,
                first,
                final,
                needs,
                pred_ptr,
                pred_ids,
                sink_preds,
                sink_k,
                last,
                pos_f,
                members_f,
                width_f,
                pos_r,
                members_r,
                width_r,
            )
            kids[0, b, 0], kids[0, b, 1] = k2, i2
            kids[0, b, 2], kids[0, b, 3] = r3, d3

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
            x = variable[ci]
            enough = (
                _reached(
                    ci, cr, pred_ids[pred_ptr[ci] : pred_ptr[ci + 1]], pos_f
                )
                >= needs[ci]
            )
            for bb in range(2):
                value = 1 - bb
                r2, d2 = _step(
                    ci,
                    value and enough,
                    cr,
                    cd,
                    last,
                    final,
                    pos_f,
                    members_f,
                    width_f,
                    pos_r,
                    members_r,
                    width_r,
                )
                if value == 1 and final[x] > ci:
                    d2 |= 1 << pos_r[ci + 1, x]
                k2, i2, r3, d3 = _settle(
                    ci + 1,
                    r2,
                    d2,
                    n,
                    variable,
                    first,
                    final,
                    needs,
                    pred_ptr,
                    pred_ids,
                    sink_preds,
                    sink_k,
                    last,
                    pos_f,
                    members_f,
                    width_f,
                    pos_r,
                    members_r,
                    width_r,
                )
                kids[top, bb, 0], kids[top, bb, 1] = k2, i2
                kids[top, bb, 2], kids[top, bb, 3] = r3, d3
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
        top -= 1
        if top > 0:
            u = top - 1
            solved[u, count[u]] = slot
            count[u] += 1
        else:
            root = slot
    return out_var[:made], out_active[:made], out_inactive[:made], root


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
