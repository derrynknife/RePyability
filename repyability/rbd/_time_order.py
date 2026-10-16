"""A run's changes put in time order, by numpy alone (#201, #208).

``_net_by_time`` nets the +1 and -1 changes of state with one sort of
integers, each time's bits above the change's sign (a time is never
negative); ``_by_time`` sorts the capacity's (time, change) pairs, and the
changes in how many systems can carry an unlimited amount, as complex
numbers; ``_capacity_totals`` merges the two the first way. What is made
of a time's changes is exact (whole counts, exact sums), so their order
within a time does not matter. ``test_time_order.py`` checks each against
its definition written out plainly.
"""

from typing import (
    TYPE_CHECKING,
    Tuple,
)

import numpy as np

from repyability.rbd._exact import (
    ExactSum,
)

if TYPE_CHECKING:
    pass


def _net_by_time(
    times: np.ndarray, steps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Changes of +1 or -1 (``steps``) at ``times`` (none negative), netted
    by time: the distinct times, in order, and the net change at each.

    A time is never negative, so its sign bit is free: each change is one
    integer, its time's bits above whether it is +1, and one sort of the
    integers puts the changes in time order (``-0.0`` sorts as ``0.0``)."""
    keys = times.view(np.uint64) << np.uint64(1)
    keys |= steps > 0
    keys.sort()
    if not keys.size:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    at = keys >> np.uint64(1)
    new = np.empty(keys.size, dtype=bool)
    new[0] = True
    np.not_equal(at[1:], at[:-1], out=new[1:])
    firsts = np.flatnonzero(new)
    ups = np.add.reduceat((keys & np.uint64(1)).astype(np.int64), firsts)
    sizes = np.diff(np.append(firsts, keys.size))
    return at[firsts].view(np.float64), 2 * ups - sizes


def _working_over_time(
    changed_at: np.ndarray,
    deltas: np.ndarray,
    t_end: float,
    start: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """How many simulated systems work after each time at which one changed
    state (and at 0 and ``t_end`` whether or not any did): the times, in
    order, and the counts, ``start`` working at 0. Each change is +1 or -1
    (``deltas``); the counts are whole numbers added exactly."""
    times, net = _net_by_time(changed_at, deltas)
    time, weights = _with_ends(times, net.astype(float), t_end)
    weights[0] += start
    return time, weights.cumsum()


def _with_ends(
    times: np.ndarray, values: np.ndarray, t_end: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``times`` (in order, each once) with 0 and ``t_end`` among them, and
    ``values`` at their times (0 at an end that had none)."""
    for end in (0.0, t_end):
        i = int(np.searchsorted(times, end))
        if i == times.size or times[i] != end:
            times = np.insert(times, i, end)
            values = np.insert(values, i, 0)
    return times, values


def _add_at(totals: dict, key, value) -> None:
    """Add ``value`` (a float or an ``ExactSum``) to ``totals[key]``,
    exactly: a float while only one value has come, an ``ExactSum`` once
    another does."""
    held = totals.get(key)
    if held is None:
        totals[key] = ExactSum(value) if isinstance(value, ExactSum) else value
    elif isinstance(held, ExactSum):
        held.add(value)
    else:
        totals[key] = ExactSum((held,)).add(value)


def _partials(value) -> list:
    """The floats whose exact sum is ``value`` (see ``_add_at``)."""
    return value.partials if isinstance(value, ExactSum) else [value]


def _by_time(
    times: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``values`` grouped by their ``times``: the distinct times, in order;
    the values, in the order of their times; and where each time's values
    start. A time's values are in no particular order: what is made of
    them (``_group_totals``, ``_group_partials``) is exact.

    Each (time, value) pair is a complex number, which numpy sorts by its
    real part, then its imaginary one: one sort of the pairs."""
    pairs = np.empty(times.size, dtype=complex)
    pairs.real = times
    pairs.imag = values
    pairs.sort()
    times, values = pairs.real, pairs.imag
    new = np.ones(times.size, dtype=bool)
    new[1:] = times[1:] != times[:-1]
    starts = np.flatnonzero(new)
    return times[starts], values, starts


def _zeroed(times: np.ndarray) -> np.ndarray:
    """``times``, a new array, with ``-0.0`` made ``0.0`` (``+ 0.0``, in
    place)."""
    return np.add(times, 0.0, out=times)


def _capacity_totals(
    at: np.ndarray,
    change: np.ndarray,
    free_at: np.ndarray,
    counts: np.ndarray,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The times at which a simulated system's expected capacity (by
    ``change``, at the times ``at``) or whether it can carry an unlimited
    amount (by the net ``counts`` of systems that can, at ``free_at``)
    changed, and 0 and ``t_end``: in order; the systems' total expected
    capacity after each (the running total of the changes, as
    ``np.cumsum`` adds them); and whether any can carry an unlimited amount
    then. ``at`` and ``free_at`` are in order, each time once (as
    ``_by_time`` gives them).

    The two are merged as ``_net_by_time`` merges its changes, each time's
    bits above whether it is ``free_at``'s: the merged times of each keep
    their order, so their changes follow them in turn."""
    keys = np.concatenate(
        (
            at.view(np.uint64) << np.uint64(1),
            (free_at.view(np.uint64) << np.uint64(1)) | np.uint64(1),
        )
    )
    keys.sort()
    bits = keys >> np.uint64(1)
    new = np.ones(keys.size, dtype=bool)
    new[1:] = bits[1:] != bits[:-1]
    group = np.cumsum(new) - 1
    free = (keys & np.uint64(1)).astype(bool)
    times = bits[new].view(np.float64)
    step = np.zeros(times.size)
    step[group[~free]] = change
    freed = np.zeros(times.size, dtype=np.int64)
    freed[group[free]] = counts
    times, step = _with_ends(times, step, t_end)
    _, freed = _with_ends(bits[new].view(np.float64), freed, t_end)
    return times, np.cumsum(step), np.cumsum(freed) > 0


def _group_stops(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Where each group of ``values`` that starts at one of ``starts``
    stops."""
    stops = np.empty_like(starts)
    stops[:-1] = starts[1:]
    stops[-1:] = values.size
    return stops


def _group_totals(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """The exact sum of each group of ``values`` (from each of ``starts``
    to the next), rounded once: a group of one value, the value."""
    totals = values[starts]
    several = np.flatnonzero(_group_stops(values, starts) - starts > 1)
    for i in several.tolist():
        stop = starts[i + 1] if i + 1 < starts.size else values.size
        totals[i] = float(ExactSum(values[starts[i] : stop]))  # noqa: E203
    return totals


def _group_partials(values: np.ndarray, starts: np.ndarray) -> list:
    """For each group of ``values`` (from each of ``starts`` to the next),
    the floats whose exact sum is the group's (``ExactSum.partials``: none
    for a sum of 0)."""
    stops = _group_stops(values, starts)
    out = []
    for a, b, first in zip(
        starts.tolist(), stops.tolist(), values[starts].tolist()
    ):
        if b - a == 1:
            out.append([first] if first else [])
        else:
            out.append(ExactSum(values[a:b]).partials)
    return out


def _exact_total(data) -> ExactSum:
    """An exact total from ``_Tally.to_dict``'s data: the floats whose
    exact sum it is (or a number)."""
    if isinstance(data, (int, float)):
        return ExactSum(float(data))
    return ExactSum([float(v) for v in data])
