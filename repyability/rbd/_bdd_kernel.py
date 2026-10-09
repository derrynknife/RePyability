"""The decision diagram's search and replay, compiled (numba, #202, #207):
``bdd.search`` and ``shannon.replay`` and ``replay_gradient``, compiled as
they are written, and what calls them with the arrays and containers
numba takes. The search's words are ``bdd.WORD`` bits (``bdd._layout``
lays the states out in them), where in Python one word of any width
holds them; the replay's values are rows of columns. The functions being
the same, so are the plans, step for step, and the values, to the last
bit.

Importing this module imports numba, which compiles them on first use (or
loads them from numba's cache).
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from repyability.rbd import bdd, shannon

_search = njit(cache=True)(bdd.search)


@njit(cache=True)
def _array(values):
    """A typed list of integers as an array (which Python reads at once,
    where it reads a typed list an element at a time)."""
    out = np.empty(len(values), np.int64)
    for j in range(len(values)):
        out[j] = values[j]
    return out


_replay = njit(cache=True)(shannon.replay)
_replay_gradient = njit(cache=True)(shannon.replay_gradient)


def run_search(arguments: tuple, limit: int) -> tuple:
    """``bdd.search`` compiled, for a core ``bdd._layout`` laid out in
    words of ``bdd.WORD`` bits, its working memory as arrays and typed
    dicts and lists."""
    n, W = arguments[0], arguments[-1]
    core = [np.asarray(a, np.int64) for a in arguments[1:-1]]
    rows = n + 3
    pair = types.UniTuple(types.int64, 2)
    triple = types.UniTuple(types.int64, 3)

    def zeros(size: int) -> np.ndarray:
        return np.zeros(size, np.int64)

    out_var, out_active, out_inactive, root, too_large = _search(
        n,
        *core,
        W,
        limit,
        Dict.empty(key_type=pair, value_type=types.int64),
        np.full(64, -1, np.int64),
        Dict.empty(key_type=triple, value_type=types.int64),
        List.empty_list(types.int64),
        List.empty_list(types.int64),
        List.empty_list(types.int64),
        zeros(2 * rows),
        zeros(2 * rows),
        zeros(2 * rows),
        zeros(2 * rows * W),
        zeros(rows),
        zeros(rows),
        zeros(rows * W),
        zeros(2 * rows),
        zeros(rows),
        zeros(W),
    )
    return (
        _array(out_var).tolist(),
        _array(out_active).tolist(),
        _array(out_inactive).tolist(),
        int(root),
        bool(too_large),
    )


@njit(cache=True)
def _columns(pivots, actives, inactives, root, works, fails, fail, work):
    """``shannon.replay`` for each row of ``works`` and ``fails`` (the
    terms' probabilities of working and failing for one value), in one
    array of slots."""
    out = np.empty(works.shape[0])
    values = np.empty(len(pivots) + 2)
    values[0], values[1] = fail, work
    for c in range(works.shape[0]):
        out[c] = _replay(
            pivots, actives, inactives, root, works[c], fails[c], values
        )
    return out


@njit(cache=True)
def _gradient_columns(
    pivots, actives, inactives, root, works, fails, fail, work
):
    """``shannon.replay`` and ``replay_gradient`` for each row (see
    ``_columns``): the values, the derivatives (a column per term) and
    whether each term has one."""
    columns, rows = works.shape
    slots = len(pivots) + 2
    out = np.empty(columns)
    gradient = np.zeros((columns, rows))
    has = np.zeros(rows, np.bool_)
    values = np.empty(slots)
    values[0], values[1] = fail, work
    adjoints = np.empty(slots)
    has_adjoint = np.empty(slots, np.bool_)
    parts = np.empty(rows)
    reached = np.empty(rows, np.bool_)
    for c in range(columns):
        out[c] = _replay(
            pivots, actives, inactives, root, works[c], fails[c], values
        )
        has_adjoint[:] = False
        reached[:] = False
        _replay_gradient(
            pivots,
            actives,
            inactives,
            root,
            works[c],
            fails[c],
            values,
            1.0,
            adjoints,
            has_adjoint,
            parts,
            reached,
        )
        for j in range(rows):
            if reached[j]:
                gradient[c, j] = parts[j]
                has[j] = True
    return out, gradient, has


def plan(pivots, actives, inactives) -> tuple:
    """A plan's steps (as ``shannon.replay`` takes them) as arrays, its
    pivots as rows of the terms it uses; and those terms, in order."""
    used = sorted(set(pivots))
    row = np.full(max(used) + 1, -1, np.int64)
    row[used] = np.arange(len(used))
    arrays = (
        row[np.asarray(pivots, np.int64)],
        np.asarray(actives, np.int64),
        np.asarray(inactives, np.int64),
    )
    return arrays, used


def replay(plan, root, works, fails, complement):
    """``shannon.replay`` compiled, for each row of the terms'
    probabilities of working (``works``, a column per term ``plan`` uses)
    and failing: the core's probability of working, or with ``complement``
    of failing. A row at a time, so it holds a value per slot of the
    plan, whatever the rows."""
    return _columns(
        *plan,
        root,
        works,
        fails,
        1.0 if complement else 0.0,
        0.0 if complement else 1.0,
    )


def replay_gradient(plan, root, works, fails, terminals):
    """``shannon.replay_gradient`` compiled, for each row (see
    ``replay``): the value, the derivative with respect to each used
    term's probability (a row per term), and whether each term has one."""
    out, gradient, has = _gradient_columns(
        *plan,
        root,
        works,
        fails,
        float(terminals[0]),
        float(terminals[1]),
    )
    return out, gradient.T, has
