"""A run's changes put in time order, compiled (numba, #201).

A simulation run records each simulated system's changes (of its expected
capacity, of whether it can carry an unlimited amount, of its state)
simulation by simulation, so they come as one sorted run per simulation,
one after another. The result needs them in time order: their distinct
times, and what changed at each. A stable ``np.argsort`` and the gathers
after it, and a merge by ``np.unique`` and ``searchsorted``, took most of
a capacity run's time, after the simulations were done. Here the changes
are sorted by a stable least-significant-digit radix sort of the times'
bits (a non-negative float's bits, read as an unsigned integer, sort as
the float does), eleven bits a pass, in blocks on numba's threads,
skipping the digits every time shares; and the capacity's times, totals
and limits are merged in one pass.

The results are the numpy path's to the last bit: a stable sort's order,
the groups of equal times, and the running totals added in the order
``np.cumsum`` adds them, the first being the first change itself.
``repairable_rbd`` uses them where numba is installed and the changes are
many (``_COMPILED_ORDER``), and leaves times that are negative, ``-0.0``
or undefined to numpy.

Importing this module imports numba.
"""

from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

# Eleven bits a pass: six passes cover the 64 bits of a time.
_BITS = 11
_RADIX = 1 << _BITS
_DIGITS = 6
_MASK = np.uint64(_RADIX - 1)
# The bits of +inf. A key above it is a negative time, -0.0 or NaN, whose
# bits do not sort as the time does.
_INF = np.uint64(0x7FF0000000000000)
# The parallel passes take the keys in blocks, one block a task: at least
# this many keys a block, and at most this many blocks (the order is the
# same however many there are).
_BLOCK = 1 << 16
_MAX_BLOCKS = 64


@njit(cache=True)
def _digit_counts(keys):
    """How many of ``keys`` have each value of each digit, and whether the
    keys are in order already; or ``ok`` False if one is not a time whose
    bits sort as it does."""
    counts = np.zeros((_DIGITS, _RADIX), np.int64)
    in_order = True
    for i in range(keys.size):
        k = keys[i]
        if k > _INF:
            return counts, False, False
        if i and k < keys[i - 1]:
            in_order = False
        for d in range(_DIGITS):
            counts[d, np.intp((k >> np.uint64(d * _BITS)) & _MASK)] += 1
    return counts, in_order, True


@njit(cache=True, parallel=True)
def _block_counts(keys, shift, blocks):
    """For each of ``blocks`` consecutive blocks of ``keys``, how many have
    each value of the digit at ``shift``."""
    n = keys.size
    size = (n + blocks - 1) // blocks
    counts = np.zeros((blocks, _RADIX), np.int64)
    for j in prange(blocks):
        for i in range(j * size, min(n, (j + 1) * size)):
            counts[j, np.intp((keys[i] >> shift) & _MASK)] += 1
    return counts


@njit(cache=True, parallel=True)
def _scatter(keys, values, to_keys, to_values, offsets, shift, blocks):
    """One pass of the radix sort: each key and its value moved to the next
    place of its digit's bucket, block by block (``offsets``, where each
    block's keys of each digit start), so that keys of one digit keep their
    order."""
    n = keys.size
    size = (n + blocks - 1) // blocks
    for j in prange(blocks):
        place = offsets[j].copy()
        for i in range(j * size, min(n, (j + 1) * size)):
            k = keys[i]
            b = np.intp((k >> shift) & _MASK)
            p = place[b]
            place[b] = p + 1
            to_keys[p] = k
            to_values[p] = values[i]


@njit(cache=True)
def _groups(keys, distinct, starts):
    """The distinct ``keys`` (in order already) into ``distinct``, and where
    each starts into ``starts``; how many there are."""
    m = 0
    for i in range(keys.size):
        if i == 0 or keys[i] != keys[i - 1]:
            starts[m] = i
            distinct[m] = keys[i]
            m += 1
    return m


def _sorted(times: np.ndarray, values: np.ndarray, reuse: bool):
    """The bits of ``times`` and ``values``, in the order of the times
    (see ``sort_by_time``), and the arrays the sort left free (or None);
    None if a time is negative, ``-0.0`` or undefined."""
    times = np.ascontiguousarray(times, dtype=np.float64)
    values = np.ascontiguousarray(values)
    keys = times.view(np.uint64)
    counts, in_order, ok = _digit_counts(keys)
    if not ok:
        return None
    if in_order:
        return keys, values, None
    if not reuse:
        keys, values = keys.copy(), values.copy()
    n = keys.size
    blocks = max(1, min(_MAX_BLOCKS, n // _BLOCK))
    source = (keys, values)
    spare = (np.empty_like(keys), np.empty_like(values))
    for d in range(_DIGITS):
        if counts[d].max() == n:  # every key has this digit
            continue
        shift = np.uint64(d * _BITS)
        per_block = _block_counts(source[0], shift, blocks)
        starts = np.zeros(_RADIX, np.int64)
        np.cumsum(per_block.sum(axis=0)[:-1], out=starts[1:])
        offsets = starts + np.cumsum(per_block, axis=0) - per_block
        _scatter(*source, *spare, offsets, shift, blocks)
        source, spare = spare, source
    return source[0], source[1], spare


def sort_by_time(
    times: np.ndarray, values: np.ndarray, reuse: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``times`` and ``values`` in the order of the times, stably (as
    ``np.argsort(times, kind="stable")`` orders them); None if a time is
    negative, ``-0.0`` or undefined. With ``reuse``, the arrays given may be
    overwritten (they serve as scratch)."""
    ordered = _sorted(times, values, reuse)
    if ordered is None:
        return None
    keys, values, _ = ordered
    return keys.view(np.float64), values


def by_time(
    times: np.ndarray, values: np.ndarray, reuse: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``values`` grouped by their ``times``, as ``_by_time`` groups them:
    the distinct times, in order; the values, in the order of their times;
    and where each time's values start. None if a time is negative,
    ``-0.0`` or undefined. The groups are written into the arrays the sort
    left free, where they are 8 bytes a value (they are views of them)."""
    ordered = _sorted(times, values, reuse)
    if ordered is None:
        return None
    keys, values, spare = ordered
    if spare is not None and spare[1].dtype.itemsize == 8:
        distinct, starts = spare[0], spare[1].view(np.int64)
    else:
        distinct = np.empty(keys.size, np.uint64)
        starts = np.empty(keys.size, np.int64)
    m = _groups(keys, distinct, starts)
    return distinct[:m].view(np.float64), values, starts[:m]


@njit(cache=True)
def _group_firsts(values, starts):
    """See ``group_firsts``."""
    m = starts.size
    firsts = np.empty(m, values.dtype)
    several = 0
    for g in range(m):
        firsts[g] = values[starts[g]]
        stop = starts[g + 1] if g + 1 < m else values.size
        if stop - starts[g] > 1:
            several += 1
    which = np.empty(several, np.int64)
    several = 0
    for g in range(m):
        stop = starts[g + 1] if g + 1 < m else values.size
        if stop - starts[g] > 1:
            which[several] = g
            several += 1
    return firsts, which


def group_firsts(
    values: np.ndarray, starts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """The first of each group of ``values`` (from each of ``starts`` to the
    next), and which groups have more than one value."""
    return _group_firsts(
        np.ascontiguousarray(values), np.ascontiguousarray(starts)
    )


@njit(cache=True)
def _merged_totals(at, change, free_at, counts, t_end):
    """See ``capacity_totals``; ``ok`` False for an undefined time."""
    na, nf = at.size, free_at.size
    size = na + nf + 2
    times = np.empty(size)
    totals = np.empty(size)
    unlimited = np.empty(size, np.bool_)
    if t_end != t_end:
        return times, totals, unlimited, False
    i = j = m = 0
    zero, end = True, True  # 0 and t_end are still to come
    total = 0.0
    free = 0
    while True:
        # The next time: the least of what is left of each.
        left = False
        t = 0.0
        if i < na:
            t = at[i]
            if t != t:
                return times, totals, unlimited, False
            left = True
        if j < nf:
            u = free_at[j]
            if u != u:
                return times, totals, unlimited, False
            if not left or u < t:
                t = u
            left = True
        if zero and (not left or 0.0 < t):
            t = 0.0
            left = True
        if end and (not left or t_end < t):
            t = t_end
            left = True
        if not left:
            break
        step = 0.0
        if i < na and at[i] == t:
            step = change[i]
            i += 1
        if j < nf and free_at[j] == t:
            free += counts[j]
            j += 1
        if zero and t == 0.0:
            zero = False
        if end and t == t_end:
            end = False
        # np.cumsum's: the first total is the first step itself (-0.0
        # stays -0.0), then each step added in turn.
        total = step if m == 0 else total + step
        times[m] = t
        totals[m] = total
        unlimited[m] = free > 0
        m += 1
    return times[:m], totals[:m], unlimited[:m], True


def capacity_totals(
    at: np.ndarray,
    change: np.ndarray,
    free_at: np.ndarray,
    counts: np.ndarray,
    t_end: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The times at which a simulated system's expected capacity
    (``change`` at ``at``) or whether it can carry an unlimited amount (the
    net ``counts`` of systems that can, at ``free_at``) changed, and 0 and
    ``t_end``, merged in one pass: the times, in order and distinct (as
    ``np.unique`` gives them); the running total of the changes, as
    ``np.cumsum`` adds them; and whether any system can carry an unlimited
    amount after each. ``at`` and ``free_at`` are in order, each time once.
    None if a time is undefined."""
    times, totals, unlimited, ok = _merged_totals(
        np.ascontiguousarray(at, dtype=np.float64),
        np.ascontiguousarray(change, dtype=np.float64),
        np.ascontiguousarray(free_at, dtype=np.float64),
        np.ascontiguousarray(counts, dtype=np.int64),
        float(t_end),
    )
    return (times, totals, unlimited) if ok else None
