"""Up/down histories over a window, and a system's from its components'.

A *timeline* is a unit's state over a window ``[0, end]``: whether it is up
at 0, and the times it changes state, alternately going down and coming
back up. A [`Timeline`][repyability.Timeline] holds one;
[`Timelines`][repyability.Timelines] several of the same unit over the same
window (one per simulation, say), and works out each measure for all of
them at once.

Timelines combine as a diagram's structure does:
[`series`][repyability.timelines.series] is up while all its inputs are,
[`parallel`][repyability.timelines.parallel] while any is, and
[`k_out_of_n`][repyability.timelines.k_out_of_n] while at least ``k`` are
(``a & b`` and ``a | b`` for short); and
[`RBD.system_timeline`][repyability.RBD.system_timeline] merges components'
timelines up a diagram, through its modules and its core. A merge is one
sweep over the inputs' changes in order of time, a whole set of histories
at once.

Each change of a merged timeline keeps its *cause*, the input that made it
(a diagram's component, for a system's), and whether it was planned
(maintenance) rather than a failure. So a system's timeline says which
component took it down each time, and its failures and planned outages are
counted apart.

Changes at the same time are taken one after another, in a fixed order:
by their causes, in the order the inputs (or a diagram's components) come,
and each cause's changes in their own order. So a unit that fails and is
repaired at once (an instant repair) takes the system down and back up at
that instant, as the simulation does. A unit's changes are counted in the
window ``[0, end)``: one at ``end`` or after it is outside it.
"""

import math
from dataclasses import dataclass, replace
from typing import (
    Dict,
    Hashable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability.utils.checks import number_or_nan

#: The most members the core's path sets may hold in all for the core to be
#: merged path set by path set (a series merge of each, and a parallel
#: merge of those); a larger core is decided at each of its members'
#: changes instead.
_PATH_MEMBERS = 2000


@dataclass(frozen=True)
class _Data:
    """Histories of one unit as arrays. History ``s`` starts up if
    ``start[s]`` is 1, and its changes are ``times[offsets[s]:offsets[s +
    1]]``, in order. Each change has a cause, an index into ``leaves`` (or,
    with ``leaves`` None, the unit itself), its position among its cause's
    changes in its history (``index``, which orders a cause's changes at
    the same time), and whether it is a planned change down."""

    start: np.ndarray
    offsets: np.ndarray
    times: np.ndarray
    causes: np.ndarray
    index: np.ndarray
    planned: np.ndarray
    leaves: Optional[tuple]
    end: float

    @property
    def size(self) -> int:
        return int(self.start.size)


def _positions(data: _Data) -> Tuple[np.ndarray, np.ndarray]:
    """Each change's history, and its position in it."""
    counts = np.diff(data.offsets)
    history = np.repeat(np.arange(data.size), counts)
    return history, np.arange(data.times.size) - data.offsets[:-1][history]


def _after(data: _Data, history: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Whether the unit is up after each change (1) or down (0)."""
    return data.start[history].astype(np.int64) ^ ((j + 1) & 1)


def _constant(start: np.ndarray, end: float, leaves) -> _Data:
    """Histories with no change, starting as ``start``."""
    size = start.size
    empty = np.zeros(0, np.int64)
    return _Data(
        start.astype(np.int8),
        np.zeros(size + 1, np.int64),
        np.zeros(0),
        empty,
        empty,
        np.zeros(0, bool),
        leaves,
        end,
    )


def _check_end(end) -> float:
    if isinstance(end, bool) or not isinstance(
        end, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"end must be a number, got {end!r}.")
    end = float(end)
    if not (math.isfinite(end) and end > 0.0):
        raise ValueError(f"end must be a positive, finite time, got {end!r}.")
    return end


def _checked_changes(changes, end: float, planned) -> Tuple[list, list]:
    """A history's change times (and planned flags), checked: numbers in
    ``[0, end]``, in order; those at ``end`` dropped."""
    times = np.asarray(changes, dtype=float).ravel()
    if planned is None:
        flags = np.zeros(times.size, bool)
    else:
        flags = np.asarray(planned, dtype=bool).ravel()
        if flags.size != times.size:
            raise ValueError(
                f"planned has {flags.size} flags for {times.size} changes: "
                "give one for each change."
            )
    if times.size:
        if not np.all(np.isfinite(times)):
            raise ValueError("The change times must be finite numbers.")
        if times[0] < 0.0 or times[-1] > end:
            raise ValueError(
                f"The change times must be in [0, end], with end {end}: got "
                f"{times.min()} to {times.max()}."
            )
        if np.any(np.diff(times) < 0.0):
            raise ValueError("The change times must be in increasing order.")
    inside = times < end
    return times[inside].tolist(), flags[inside].tolist()


def _raw(
    starts: Sequence[bool],
    changes: Sequence[Sequence[float]],
    planned: Sequence[Sequence[bool]],
    end: float,
) -> _Data:
    """Histories from their start states, change times and planned flags
    (already checked)."""
    counts = np.array([len(c) for c in changes], np.int64)
    offsets = np.zeros(counts.size + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    times = np.array([t for c in changes for t in c], dtype=float)
    flags = np.array([f for p in planned for f in p], dtype=bool)
    data = _Data(
        np.array([bool(s) for s in starts], np.int8),
        offsets,
        times,
        np.zeros(times.size, np.int64),
        np.zeros(times.size, np.int64),
        flags,
        None,
        end,
    )
    history, j = _positions(data)
    up = _after(data, history, j) == 1
    if np.any(flags & up):
        raise ValueError(
            "Only a change down can be planned (maintenance taking the unit "
            "off line): a change back up has planned False."
        )
    return replace(data, index=j)


def _leaf(data: _Data, rank: int) -> _Data:
    """``data`` as a cause of its own, ``rank``: each change its own."""
    _, j = _positions(data)
    return replace(
        data,
        causes=np.full(data.times.size, rank, np.int64),
        index=j,
        leaves=None,
    )


def _tiled(data: _Data, size: int) -> _Data:
    """A single history repeated ``size`` times."""
    if data.size == size:
        return data
    n = data.times.size
    offsets = np.arange(size + 1, dtype=np.int64) * n
    return replace(
        data,
        start=np.repeat(data.start, size),
        offsets=offsets,
        times=np.tile(data.times, size),
        causes=np.tile(data.causes, size),
        index=np.tile(data.index, size),
        planned=np.tile(data.planned, size),
    )


def _history(data: _Data, s: int) -> _Data:
    """History ``s`` alone."""
    a, b = int(data.offsets[s]), int(data.offsets[s + 1])
    return replace(
        data,
        start=data.start[s : s + 1],  # noqa: E203
        offsets=np.array([0, b - a], np.int64),
        times=data.times[a:b],
        causes=data.causes[a:b],
        index=data.index[a:b],
        planned=data.planned[a:b],
    )


@dataclass(frozen=True)
class _Table:
    """The changes a merge takes in (or a diagram's merges, its
    components'), in the order it takes them (see the module docstring):
    each one's history, time, cause, index among its cause's changes, and
    whether it is planned. A change a merge makes goes the way the change
    that made it went (each merge is monotone in its inputs), so it is
    planned when that one is."""

    history: np.ndarray
    times: np.ndarray
    causes: np.ndarray
    index: np.ndarray
    planned: np.ndarray


class _Run(NamedTuple):
    """A unit's histories in a merge: whether each starts up, and its
    changes as places in the merge's ``_Table``, history by history (and in
    each, in its own order)."""

    start: np.ndarray
    key: np.ndarray


def _small(values: np.ndarray) -> np.ndarray:
    """Non-negative whole numbers in the smallest type that holds them,
    which numpy sorts fastest."""
    top = int(values.max()) if values.size else 0
    for kind in (np.uint16, np.uint32):
        if top <= np.iinfo(kind).max:
            return values.astype(kind)
    return values


def _table(datas: Sequence[_Data]) -> Tuple[_Table, List[_Run]]:
    """The changes of ``datas`` (all with the same histories, their causes
    indices into one list) in the order a merge takes them, and each of
    ``datas`` as a ``_Run`` of places in it."""
    empty = (
        np.zeros(0, np.int64),
        np.zeros(0),
        np.zeros(0, np.int64),
        np.zeros(0, np.int64),
        np.zeros(0, bool),
    )
    parts = [
        (
            _positions(data)[0],
            data.times,
            data.causes,
            data.index,
            data.planned,
        )
        for data in datas
    ]
    history, times, causes, index, planned = (
        np.concatenate(column) for column in zip(empty, *parts)
    )
    order = np.lexsort((_small(index), _small(causes), times, _small(history)))
    key = np.empty(order.size, np.int64)
    key[order] = np.arange(order.size)
    table = _Table(
        history[order],
        times[order],
        causes[order],
        index[order],
        planned[order],
    )
    runs = []
    at = 0
    for data in datas:
        stop = at + data.times.size
        runs.append(_Run(data.start, key[at:stop]))
        at = stop
    return table, runs


def _ups(run: _Run, table: _Table) -> np.ndarray:
    """Whether the unit is up after each of its changes (1) or down (0)."""
    history = table.history[run.key]
    firsts = np.searchsorted(history, np.arange(run.start.size))
    j = np.arange(history.size) - firsts[history]
    return run.start[history].astype(np.int64) ^ ((j + 1) & 1)


def _merged(runs: Sequence[_Run], need: int, table: _Table) -> _Run:
    """The histories of a unit that is up while at least ``need`` of
    ``runs`` are: one sweep over their changes in the table's order, every
    history at once."""
    size = runs[0].start.size
    count = np.zeros(size, np.int64)
    keys = []
    deltas = []
    for run in runs:
        count += run.start
        if run.key.size:
            keys.append(run.key)
            deltas.append(2 * _ups(run, table) - 1)
    start = (count >= need).astype(np.int8)
    if not keys:
        return _Run(start, np.zeros(0, np.int64))
    # A change of a diagram's component that reaches several inputs (a
    # repeated node, or a member of several path sets) is one place in the
    # table, and moves them all one way: in whatever order they take it,
    # it takes the count across ``need`` once.
    key = np.concatenate(keys)
    order = np.argsort(key, kind="stable")
    key = key[order]
    delta = np.concatenate(deltas)[order]
    history = table.history[key]
    # How many inputs are up after each change: the history's count at 0
    # and its changes so far.
    total = np.cumsum(delta)
    firsts = np.searchsorted(history, np.arange(size))
    before = np.zeros(size, np.int64)
    later = firsts > 0
    before[later] = total[firsts[later] - 1]
    up = count[history] + total - before[history] >= need
    was = np.empty_like(up)
    was[1:] = up[:-1]
    opens = np.ones(history.size, bool)
    opens[1:] = history[1:] != history[:-1]
    was[opens] = start[history[opens]].astype(bool)
    return _Run(start, key[up != was])


def _unrun(run: _Run, table: _Table, leaves, end: float) -> _Data:
    """``run``'s histories, their changes' times, causes and flags from
    ``table``."""
    key = run.key
    size = run.start.size
    offsets = np.zeros(size + 1, np.int64)
    np.cumsum(np.bincount(table.history[key], minlength=size), out=offsets[1:])
    return _Data(
        run.start,
        offsets,
        table.times[key],
        table.causes[key],
        table.index[key],
        table.planned[key],
        leaves,
        end,
    )


def _complement(data: _Data) -> _Data:
    """Up while ``data`` is down: the same changes, the other way."""
    return replace(
        data,
        start=(1 - data.start).astype(np.int8),
        planned=np.zeros(data.times.size, bool),
    )


def _spans(data: _Data) -> Tuple[np.ndarray, np.ndarray]:
    """Each history's time up and time down."""
    size, end = data.size, data.end
    history, j = _positions(data)
    times = data.times
    previous = np.zeros(times.size)
    later = j > 0
    previous[later] = times[np.flatnonzero(later) - 1]
    lengths = times - previous
    up_before = (data.start[history].astype(np.int64) ^ (j & 1)) == 1
    counts = np.diff(data.offsets)
    last = np.zeros(size)
    some = counts > 0
    last[some] = times[data.offsets[1:][some] - 1]
    final = end - last
    final_up = (data.start.astype(np.int64) ^ (counts & 1)) == 1
    uptime = np.bincount(
        history, weights=np.where(up_before, lengths, 0.0), minlength=size
    ) + np.where(final_up, final, 0.0)
    downtime = np.bincount(
        history, weights=np.where(up_before, 0.0, lengths), minlength=size
    ) + np.where(final_up, 0.0, final)
    return uptime, downtime


def _counts(data: _Data) -> Dict[str, np.ndarray]:
    """Each history's failures (unplanned changes down), planned outages
    and restorations."""
    size = data.size
    history, j = _positions(data)
    up = _after(data, history, j) == 1
    down = ~up
    return {
        "failures": np.bincount(history[down & ~data.planned], minlength=size),
        "planned": np.bincount(history[down & data.planned], minlength=size),
        "restorations": np.bincount(history[up], minlength=size),
    }


def _first_failures(data: _Data) -> np.ndarray:
    """Each history's first failure (an unplanned change down), or inf."""
    history, j = _positions(data)
    failed = (_after(data, history, j) == 0) & ~data.planned
    out = np.full(data.size, np.inf)
    which, at = np.unique(history[failed], return_index=True)
    out[which] = data.times[np.flatnonzero(failed)[at]]
    return out


def _by_cause(data: _Data, name, up: bool) -> Dict[Hashable, np.ndarray]:
    """For each cause, how many failures (or, ``up``, restorations) it made
    in each history."""
    size = data.size
    history, j = _positions(data)
    after = _after(data, history, j) == 1
    chosen = after if up else (~after & ~data.planned)
    leaves = (name,) if data.leaves is None else data.leaves
    out: Dict[Hashable, np.ndarray] = {}
    for rank, leaf in enumerate(leaves):
        mask = chosen & (data.causes == rank)
        out[leaf] = out.get(leaf, 0) + np.bincount(
            history[mask], minlength=size
        )
    return out


def _intervals(data: _Data, up: bool) -> np.ndarray:
    """A single history's spans up (or down), as ``(start, stop)`` rows."""
    times = data.times
    edges = np.concatenate(([0.0], times, [data.end]))
    states = int(data.start[0]) ^ (np.arange(times.size + 1) & 1)
    chosen = states == (1 if up else 0)
    return np.column_stack((edges[:-1][chosen], edges[1:][chosen]))


def _curve(data: _Data) -> Tuple[np.ndarray, np.ndarray]:
    """The fraction of histories up after each time at which one changed
    (and at 0 and the end): the times, in order, and the fractions."""
    from repyability.rbd.repairable_rbd import _working_over_time

    history, j = _positions(data)
    delta = 2 * _after(data, history, j) - 1
    time, working = _working_over_time(
        data.times, delta, data.end, int(data.start.sum())
    )
    return time, working / data.size


def _unified(
    inputs: Sequence[Tuple[_Data, Hashable, bool]],
) -> Tuple[List[_Data], tuple]:
    """``inputs`` (each its data, the label a cause of its own takes, and
    whether it is one whatever its own causes are) with their causes indices
    into one tuple of leaves, in order of first appearance."""
    rank: Dict[Hashable, int] = {}
    out = []
    for data, label, alone in inputs:
        if alone or data.leaves is None:
            data = _leaf(data, rank.setdefault(label, len(rank)))
        else:
            ranks = np.array(
                [rank.setdefault(leaf, len(rank)) for leaf in data.leaves],
                np.int64,
            )
            data = replace(data, causes=ranks[data.causes], leaves=None)
        out.append(data)
    return out, tuple(rank)


def _shape(items) -> Tuple[int, float, bool]:
    """The histories and window of timelines merged together: Timeline and
    Timelines objects, the Timelines all with as many histories, all with
    one window. Whether any is a Timelines."""
    sizes = set()
    ends = set()
    many = False
    for item in items:
        if isinstance(item, Timelines):
            many = True
            sizes.add(item._data.size)
        elif not isinstance(item, Timeline):
            raise TypeError(
                f"Expected Timeline or Timelines objects, got "
                f"{type(item).__name__}."
            )
        ends.add(item._data.end)
    if len(ends) > 1:
        raise ValueError(
            f"Timelines merged together need one window: got ends "
            f"{sorted(ends)}."
        )
    if len(sizes) > 1:
        raise ValueError(
            f"Timelines merged together need as many histories each: got "
            f"{sorted(sizes)}."
        )
    if not ends:
        raise ValueError("Give at least one timeline.")
    return (sizes.pop() if sizes else 1), ends.pop(), many


def _wrap(data: _Data, many: bool, name=None):
    """A Timelines (or, for one history, a Timeline) holding ``data``."""
    if many:
        return Timelines._from_data(data, name)
    return Timeline._from_data(data, name)


def _combine(timelines, need, name, what: str):
    """``timelines`` (positional Timeline or Timelines objects, or one
    mapping of labels to them) merged: up while at least ``need`` of them
    are (``need`` a function of how many there are)."""
    if len(timelines) == 1 and isinstance(timelines[0], Mapping):
        labelled = list(timelines[0].items())
        alone = True
    else:
        labelled = [(getattr(item, "name", None), item) for item in timelines]
        alone = False
    if not labelled:
        raise ValueError(f"{what} needs at least one timeline.")
    items = [item for _, item in labelled]
    size, end, many = _shape(items)
    k = need(len(items))
    data, leaves = _unified(
        [(_tiled(item._data, size), label, alone) for label, item in labelled]
    )
    table, runs = _table(data)
    return _wrap(
        _unrun(_merged(runs, k, table), table, leaves, end), many, name
    )


def series(*timelines, name: Optional[Hashable] = None):
    """The timeline of a unit that is up while all of ``timelines`` are (as
    ``a & b``).

    Parameters
    ----------
    *timelines : Timeline or Timelines, or a mapping of them
        The inputs, all over one window: [`Timeline`][repyability.Timeline]
        objects, or [`Timelines`][repyability.Timelines] with as many
        histories each (a Timeline among them stands for each history).
        Given as one mapping, ``{label: timeline}``, each input is a cause
        of its own, labelled by its key; given one by one, each input's
        changes keep their causes (an unmerged timeline's, its ``name``).
    name : hashable, optional
        The result's name. By default None.

    Returns
    -------
    Timeline or Timelines
        A Timelines if any input is one, else a Timeline. Each change's
        cause is the input change that made it (see the module docstring
        of ``repyability.timelines``).

    Examples
    --------
    >>> from repyability import Timeline
    >>> from repyability.timelines import series
    >>> pump = Timeline([10.0, 12.0], end=100.0)
    >>> valve = Timeline([11.0, 15.0], end=100.0)
    >>> series({"pump": pump, "valve": valve}).down_intervals.tolist()
    [[10.0, 15.0]]
    """
    return _combine(timelines, lambda n: n, name, "series")


def parallel(*timelines, name: Optional[Hashable] = None):
    """The timeline of a unit that is up while any of ``timelines`` is (as
    ``a | b``).

    Parameters
    ----------
    *timelines : Timeline or Timelines, or a mapping of them
        The inputs, as for [`series`][repyability.timelines.series].
    name : hashable, optional
        The result's name. By default None.

    Returns
    -------
    Timeline or Timelines
        A Timelines if any input is one, else a Timeline.

    Examples
    --------
    >>> from repyability import Timeline
    >>> from repyability.timelines import parallel
    >>> a = Timeline([10.0, 12.0], end=100.0)
    >>> b = Timeline([11.0, 15.0], end=100.0)
    >>> parallel(a, b).down_intervals.tolist()
    [[11.0, 12.0]]
    """
    return _combine(timelines, lambda n: 1, name, "parallel")


def k_out_of_n(k: int, *timelines, name: Optional[Hashable] = None):
    """The timeline of a unit that is up while at least ``k`` of
    ``timelines`` are.

    Parameters
    ----------
    k : int
        How many inputs must be up, from 1 (parallel) to their number
        (series).
    *timelines : Timeline or Timelines, or a mapping of them
        The inputs, as for [`series`][repyability.timelines.series].
    name : hashable, optional
        The result's name. By default None.

    Returns
    -------
    Timeline or Timelines
        A Timelines if any input is one, else a Timeline.

    Raises
    ------
    ValueError
        If ``k`` is not a whole number from 1 to the number of inputs.

    Examples
    --------
    >>> from repyability import Timeline
    >>> from repyability.timelines import k_out_of_n
    >>> units = [Timeline(c, end=100.0) for c in ([5, 9], [7, 12], [8, 20])]
    >>> k_out_of_n(2, *units).down_intervals.tolist()
    [[7.0, 12.0]]
    """

    def need(n: int) -> int:
        if (
            isinstance(k, bool)
            or not isinstance(k, (int, np.integer))
            or not 1 <= k <= n
        ):
            raise ValueError(
                f"k must be a whole number from 1 to the {n} inputs, got "
                f"{k!r}."
            )
        return int(k)

    return _combine(timelines, need, name, "k_out_of_n")


def _checked_outage(outage, end: float) -> Tuple[float, float]:
    """An outage log's record, ``(start, stop)``, as floats in the window
    ``[0, end]`` (a stop of None, or past ``end``, at ``end``), or a
    ValueError saying what is wrong with it (#232)."""
    try:
        start, stop = outage
    except (TypeError, ValueError):
        raise ValueError(
            f"Each outage is a (start, end) pair, got {outage!r}."
        ) from None
    first = number_or_nan(start)
    if math.isnan(first):
        raise ValueError(
            f"Outage ({start}, {stop}) starts at {start!r}, which is not a "
            f"time: give its start as a number from 0 to the window's end "
            f"({end:g})."
        )
    if first < 0.0:
        raise ValueError(
            f"Outage ({start}, {stop}) starts before the window's start, "
            "0: give the times from the window's start."
        )
    if first > end:
        raise ValueError(
            f"Outage ({first}, {stop}) starts after the window's end "
            f"({end})."
        )
    if stop is None:
        return first, end
    last = number_or_nan(stop)
    if math.isnan(last):
        raise ValueError(
            f"Outage ({start}, {stop}) ends at {stop!r}, which is not a "
            "time: give its end as a number, or None for an outage that "
            "runs to the window's end."
        )
    if last < first:
        raise ValueError(f"Outage ({start}, {stop}) ends before it starts.")
    return first, min(last, end)


def _merged_outages(outages: list, flags: list, end: float):
    """``outages`` (with their ``planned`` flags) sorted by start, each run
    that overlaps or touches joined into one: planned only if all of it
    was. An end of None runs to ``end``."""
    order = sorted(range(len(outages)), key=lambda i: float(outages[i][0]))
    merged: list = []
    marks: list = []
    for i in order:
        start, stop = outages[i]
        start = float(start)
        stop = end if stop is None else float(stop)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
            marks[-1] = marks[-1] and flags[i]
        else:
            merged.append([start, stop])
            marks.append(flags[i])
    return [tuple(pair) for pair in merged], marks


class Timeline:
    """A unit's up/down history over a window ``[0, end]``: whether it is
    up at 0, and the times it changes state, alternately going down and
    coming back up (see ``repyability.timelines``).

    Timelines merge as a diagram's structure does (``a & b`` is up while
    both are, ``a | b`` while either is; see
    [`series`][repyability.timelines.series],
    [`parallel`][repyability.timelines.parallel],
    [`k_out_of_n`][repyability.timelines.k_out_of_n] and
    [`RBD.system_timeline`][repyability.RBD.system_timeline]), and each
    change of a merged one keeps its cause.

    Parameters
    ----------
    changes : array-like of float
        The times the unit changes state, in increasing order, in
        ``[0, end]`` (one at ``end`` changes nothing in the window and is
        dropped). Two at one time are a change and its undoing at once (an
        instant repair).
    end : float
        The window's end, positive.
    up : bool, optional
        Whether the unit is up at 0. By default True.
    planned : array-like of bool, optional
        For each change, whether it is planned (maintenance taking the unit
        off line) rather than a failure. Only a change down can be. By
        default none is.
    name : hashable, optional
        The timeline's name: the cause its changes have when it is merged
        with others one by one. By default None.

    Raises
    ------
    ValueError
        If a change time is not finite, is outside ``[0, end]`` or out of
        order, if ``end`` is not positive and finite, or if ``planned``
        does not have one flag per change or flags a change back up.

    Examples
    --------
    >>> from repyability import Timeline
    >>> pump = Timeline([100.0, 104.0, 900.0, 930.0], end=8760.0)
    >>> pump.uptime, pump.failures
    (8726.0, 2)
    >>> pump.down_intervals.tolist()
    [[100.0, 104.0], [900.0, 930.0]]
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        changes,
        end: float,
        up: bool = True,
        *,
        planned=None,
        name: Optional[Hashable] = None,
    ):
        end = _check_end(end)
        times, flags = _checked_changes(changes, end, planned)
        self._data = _raw([bool(up)], [times], [flags], end)
        self._name = name

    @classmethod
    def _from_data(cls, data: _Data, name=None) -> "Timeline":
        out = cls.__new__(cls)
        out._data = data
        out._name = name
        return out

    @classmethod
    def from_outages(
        cls,
        outages,
        end: float,
        *,
        planned=None,
        name: Optional[Hashable] = None,
        merge: bool = False,
    ) -> "Timeline":
        """The timeline of a unit that is up but for ``outages``: an
        outage log.

        Two outages that touch, one ending as the next starts, are two
        changes at one time: the unit is down throughout, but restored and
        failed again at that instant, so they count as two failures. Real
        logs often hold one outage as several records, overlapping or
        touching (two work orders on one outage): ``merge=True`` makes each
        run of them one outage (#179).

        Parameters
        ----------
        outages : iterable of (float, float or None)
            Each outage's start and end, in order of start; one ends no
            later than the next starts (in any order, and overlapping, with
            ``merge``). An end of None, or at or after the window's end,
            runs to the end.
        end : float
            The window's end, positive.
        planned : iterable of bool, optional
            For each outage, whether it was planned (maintenance) rather
            than a failure. By default none was. Outages merged into one
            are planned only if all of them were.
        name : hashable, optional
            The timeline's name. By default None.
        merge : bool, optional
            Whether to join outages that overlap or touch into one, after
            sorting them by start. By default False: they must not overlap.

        Returns
        -------
        Timeline
            Up at 0 (down, if an outage starts at 0, from a change at 0).

        Raises
        ------
        ValueError
            If an outage ends before it starts, starts before the one before
            it ends (without ``merge``), or starts outside ``[0, end]``, or
            ``planned`` does not have one flag per outage.

        Examples
        --------
        >>> from repyability import Timeline
        >>> log = Timeline.from_outages([(100, 104), (900, None)], end=1000)
        >>> log.downtime, log.restorations
        (104.0, 1)

        Two work orders on one outage, and two records of another:

        >>> records = [(100, 104), (102, 110), (500, 520), (520, 530)]
        >>> Timeline.from_outages(records, end=1000)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Outage (102.0, 110.0) is out of order: ... merge=True ...
        >>> joined = Timeline.from_outages(records, end=1000, merge=True)
        >>> joined.failures, joined.downtime
        (2, 40.0)
        """
        end = _check_end(end)
        outages = [_checked_outage(outage, end) for outage in outages]
        flags = (
            [False] * len(outages)
            if planned is None
            else [bool(p) for p in planned]
        )
        if len(flags) != len(outages):
            raise ValueError(
                f"planned has {len(flags)} flags for {len(outages)} "
                "outages: give one for each outage."
            )
        if merge:
            outages, flags = _merged_outages(outages, flags, end)
        changes: list = []
        marks: list = []
        last = 0.0
        for (start, stop), flag in zip(outages, flags):
            if start < last:
                raise ValueError(
                    f"Outage ({start}, {stop}) is out of order: each starts "
                    "no earlier than the one before ends. Give merge=True to "
                    "join outages that overlap or touch."
                )
            changes += [start, stop]
            marks += [flag, False]
            last = stop
        return cls(changes, end, planned=marks, name=name)

    @classmethod
    def from_durations(
        cls,
        up_times,
        down_times,
        end: float,
        *,
        up: bool = True,
        name: Optional[Hashable] = None,
    ) -> "Timeline":
        """The timeline of a unit that is up and down for given durations
        in turn: up for ``up_times[0]``, down for ``down_times[0]``, up for
        ``up_times[1]``, and so on (down first if ``up`` is False), as a
        simulation draws lives and repairs.

        Parameters
        ----------
        up_times : array-like of float
            The durations up, in order: non-negative (inf for ever).
        down_times : array-like of float
            The durations down, in order.
        end : float
            The window's end, positive.
        up : bool, optional
            Whether the unit is up at 0. By default True.
        name : hashable, optional
            The timeline's name. By default None.

        Returns
        -------
        Timeline
            The unit's changes before ``end``. Once the durations run out
            it stays as it is.

        Raises
        ------
        ValueError
            If a duration is negative or not a number.

        Examples
        --------
        >>> from repyability import Timeline
        >>> unit = Timeline.from_durations([10, 20, 30], [1, 2], end=50)
        >>> unit.changes.tolist()
        [10.0, 11.0, 31.0, 33.0]
        """
        end = _check_end(end)
        ups = np.asarray(up_times, dtype=float).ravel()
        downs = np.asarray(down_times, dtype=float).ravel()
        for values in (ups, downs):
            if np.any(np.isnan(values)) or np.any(values < 0.0):
                raise ValueError(
                    "The durations must be non-negative numbers (inf for "
                    "ever)."
                )
        first, second = (ups, downs) if up else (downs, ups)
        n = min(first.size, second.size + 1)
        durations = np.empty(n + min(second.size, n))
        durations[0::2] = first[:n]
        durations[1::2] = second[: durations.size - n]
        times = np.cumsum(durations)
        return cls(times[times < end], end, up=up, name=name)

    @property
    def end(self) -> float:
        """The window's end."""
        return self._data.end

    @property
    def up(self) -> bool:
        """Whether the unit is up at 0."""
        return bool(self._data.start[0])

    @property
    def name(self) -> Optional[Hashable]:
        """The timeline's name (see ``Timeline``)."""
        return self._name

    @property
    def changes(self) -> np.ndarray:
        """The times the unit changes state, in order."""
        return self._data.times.copy()

    @property
    def causes(self) -> list:
        """Each change's cause: the input whose change made it (for an
        unmerged timeline, its own ``name``)."""
        data = self._data
        leaves = (self._name,) if data.leaves is None else data.leaves
        return [leaves[c] for c in data.causes.tolist()]

    @property
    def planned(self) -> np.ndarray:
        """For each change, whether it is a planned change down."""
        return self._data.planned.copy()

    @property
    def uptime(self) -> float:
        """The time up in the window."""
        return float(_spans(self._data)[0][0])

    @property
    def downtime(self) -> float:
        """The time down in the window."""
        return float(_spans(self._data)[1][0])

    @property
    def availability(self) -> float:
        """The fraction of the window up."""
        return self.uptime / self.end

    @property
    def failures(self) -> int:
        """The unplanned changes down."""
        return int(_counts(self._data)["failures"][0])

    @property
    def planned_outages(self) -> int:
        """The planned changes down."""
        return int(_counts(self._data)["planned"][0])

    @property
    def restorations(self) -> int:
        """The changes back up."""
        return int(_counts(self._data)["restorations"][0])

    @property
    def first_failure(self) -> float:
        """The time of the first failure (unplanned change down), or inf if
        there is none in the window."""
        return float(_first_failures(self._data)[0])

    @property
    def up_intervals(self) -> np.ndarray:
        """The spans up, as ``(start, stop)`` rows, in order (a change and
        its undoing at once leave a span of length 0)."""
        return _intervals(self._data, True)

    @property
    def down_intervals(self) -> np.ndarray:
        """The spans down, as ``(start, stop)`` rows, in order."""
        return _intervals(self._data, False)

    def state(self, t):
        """Whether the unit is up at ``t``.

        Parameters
        ----------
        t : float or array-like of float
            Times in the window. At a change, the state after it.

        Returns
        -------
        bool or numpy.ndarray
            True where the unit is up.
        """
        times = np.asarray(t, dtype=float)
        counts = np.searchsorted(self._data.times, times, side="right")
        out = (int(self._data.start[0]) ^ (counts & 1)) == 1
        return bool(out) if out.ndim == 0 else out

    def failures_by_cause(self) -> Dict[Hashable, int]:
        """How many failures (unplanned changes down) each cause made.

        Inputs that fail at one instant are taken one after another, in
        the order the inputs come (see the module docstring): of two
        series inputs failing together, the first takes the timeline down
        and is the cause, ``(x & y)`` giving it to ``x`` and ``(y & x)`` to
        ``y``.

        Returns
        -------
        dict
            Cause -> count, for every cause, in order.
        """
        return {
            k: int(v[0])
            for k, v in _by_cause(self._data, self._name, False).items()
        }

    def restorations_by_cause(self) -> Dict[Hashable, int]:
        """How many changes back up each cause made.

        Returns
        -------
        dict
            Cause -> count, for every cause, in order.
        """
        return {
            k: int(v[0])
            for k, v in _by_cause(self._data, self._name, True).items()
        }

    def __len__(self) -> int:
        return int(self._data.times.size)

    def __and__(self, other):
        return series(self, other)

    def __or__(self, other):
        return parallel(self, other)

    def __invert__(self) -> "Timeline":
        return Timeline._from_data(_complement(self._data), self._name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Timeline):
            return NotImplemented
        return _same(self, other)

    def to_dict(self) -> dict:
        """The history as plain data, ready for ``json.dumps`` (#235).

        Returns
        -------
        dict
            ``end``, ``up`` (at 0), ``name``, ``changes`` (the times it
            changes state), ``causes`` (each change's cause) and
            ``planned`` (whether each is a planned change down).
        """
        from repyability.rbd.results import plain

        return {
            "end": self.end,
            "up": bool(self.up),
            "name": plain(self.name),
            "changes": self.changes.tolist(),
            "causes": plain(self.causes),
            "planned": [bool(p) for p in self.planned],
        }

    def __repr__(self) -> str:
        shown = ", ".join(f"{t:g}" for t in self._data.times[:6])
        if len(self) > 6:
            shown += ", ..."
        label = "" if self._name is None else f", name={self._name!r}"
        return f"Timeline([{shown}], end={self.end:g}, up={self.up}{label})"


class Timelines:
    """Up/down histories of one unit over one window ``[0, end]``: one per
    simulation, say (see ``repyability.timelines``). Each measure is worked
    out for all of them at once, as an array with one value per history.

    Indexing gives a history's [`Timeline`][repyability.Timeline] (a slice,
    a Timelines), and Timelines merge as Timeline objects do, history by
    history (``a & b``, ``a | b``, and the functions of
    ``repyability.timelines``).

    Parameters
    ----------
    timelines : sequence of Timeline or Timelines
        The histories, all over one window: a Timelines among them gives
        each of its own, in order (so Timelines of the same unit join).
    name : hashable, optional
        The name: the cause the unit's own changes have when it is merged.
        By default the histories' own name, if they share one.

    Raises
    ------
    ValueError
        If there are no histories, or they do not share a window.

    Examples
    --------
    >>> from repyability import Timeline, Timelines
    >>> runs = Timelines([Timeline([2.0, 3.0], 10.0), Timeline([], 10.0)])
    >>> runs.uptime.tolist(), runs.failures.tolist()
    ([9.0, 10.0], [1, 0])
    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, timelines, name: Optional[Hashable] = None):
        items = list(timelines)
        if not items or not all(
            isinstance(t, (Timeline, Timelines)) for t in items
        ):
            raise ValueError(
                "Give a sequence of one or more Timeline (or Timelines) "
                "objects."
            )
        ends = sorted({t.end for t in items})
        if len(ends) > 1:
            raise ValueError(
                f"Timelines of one unit need one window: got ends {ends}."
            )
        names = {t.name for t in items}
        if name is None and len(names) == 1:
            name = names.pop()
        if all(t._data.leaves is None for t in items):
            data = _stacked([t._data for t in items], None, ends[0])
        else:
            parts, leaves = _unified([(t._data, t.name, False) for t in items])
            data = _stacked(parts, leaves, ends[0])
        self._data = data
        self._name = name

    @classmethod
    def _from_data(cls, data: _Data, name=None) -> "Timelines":
        out = cls.__new__(cls)
        out._data = data
        out._name = name
        return out

    @property
    def end(self) -> float:
        """The window's end."""
        return self._data.end

    @property
    def name(self) -> Optional[Hashable]:
        """The name (see ``Timelines``)."""
        return self._name

    @property
    def up(self) -> np.ndarray:
        """Whether each history starts up."""
        return self._data.start.astype(bool)

    @property
    def uptime(self) -> np.ndarray:
        """Each history's time up."""
        return _spans(self._data)[0]

    @property
    def downtime(self) -> np.ndarray:
        """Each history's time down."""
        return _spans(self._data)[1]

    @property
    def availability(self) -> np.ndarray:
        """Each history's fraction of the window up."""
        return self.uptime / self.end

    @property
    def failures(self) -> np.ndarray:
        """Each history's failures (unplanned changes down)."""
        return _counts(self._data)["failures"]

    @property
    def planned_outages(self) -> np.ndarray:
        """Each history's planned changes down."""
        return _counts(self._data)["planned"]

    @property
    def restorations(self) -> np.ndarray:
        """Each history's changes back up."""
        return _counts(self._data)["restorations"]

    @property
    def first_failure(self) -> np.ndarray:
        """Each history's first failure (unplanned change down), or inf."""
        return _first_failures(self._data)

    def state(self, t) -> np.ndarray:
        """Whether each history is up at ``t``.

        Parameters
        ----------
        t : float or array-like of float
            Times in the window. At a change, the state after it.

        Returns
        -------
        numpy.ndarray
            One row per history (one value per history for a single
            ``t``): True where it is up.
        """
        data = self._data
        times = np.atleast_1d(np.asarray(t, dtype=float))
        history, _ = _positions(data)
        out = np.empty((data.size, times.size), bool)
        for i, time in enumerate(times):
            passed = np.bincount(
                history[data.times <= time], minlength=data.size
            )
            out[:, i] = (data.start.astype(np.int64) ^ (passed & 1)) == 1
        return out[:, 0] if np.ndim(t) == 0 else out

    def availability_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """The fraction of the histories up over time: after each time at
        which one changed (and at 0 and the end), as
        [`AvailabilityResult`][repyability.AvailabilityResult]'s
        ``timeline`` and ``availability`` give it.

        Returns
        -------
        tuple of numpy.ndarray
            The times, in order, and the fraction up from each until the
            next.
        """
        return _curve(self._data)

    def point_availability(self, t):
        """The fraction of the histories up at ``t``.

        Parameters
        ----------
        t : float or array-like of float
            Times in the window. At a change, the state after it.

        Returns
        -------
        float or numpy.ndarray
            The fraction up at each time.
        """
        time, fraction = _curve(self._data)
        at = np.searchsorted(time, np.asarray(t, dtype=float), side="right")
        out = fraction[np.maximum(at - 1, 0)]
        return float(out) if np.ndim(out) == 0 else out

    def failures_by_cause(self) -> Dict[Hashable, np.ndarray]:
        """How many failures (unplanned changes down) each cause made in
        each history.

        Returns
        -------
        dict
            Cause -> an array with one count per history, for every cause,
            in order.
        """
        return _by_cause(self._data, self._name, False)

    def restorations_by_cause(self) -> Dict[Hashable, np.ndarray]:
        """How many changes back up each cause made in each history.

        Returns
        -------
        dict
            Cause -> an array with one count per history, for every cause,
            in order.
        """
        return _by_cause(self._data, self._name, True)

    def __len__(self) -> int:
        return self._data.size

    def __getitem__(self, i):
        size = self._data.size
        if isinstance(i, slice):
            picked = range(size)[i]
            return Timelines._from_data(
                _stacked(
                    [_history(self._data, s) for s in picked],
                    self._data.leaves,
                    self._data.end,
                ),
                self._name,
            )
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise TypeError(f"Index with a whole number, got {i!r}.")
        if not -size <= i < size:
            raise IndexError(f"History {i} of {size}.")
        return Timeline._from_data(
            _history(self._data, int(i) % size), self._name
        )

    def __iter__(self) -> Iterator[Timeline]:
        for s in range(len(self)):
            yield self[s]

    def __and__(self, other):
        return series(self, other)

    def __or__(self, other):
        return parallel(self, other)

    def __invert__(self) -> "Timelines":
        return Timelines._from_data(_complement(self._data), self._name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Timelines):
            return NotImplemented
        return _same(self, other)

    def to_dict(self) -> dict:
        """The histories as plain data, ready for ``json.dumps`` (#235).

        Returns
        -------
        dict
            ``end``, ``name`` and ``timelines``, each history's
            ``Timeline.to_dict()``, in order.
        """
        from repyability.rbd.results import plain

        return {
            "end": self.end,
            "name": plain(self.name),
            "timelines": [history.to_dict() for history in self],
        }

    def __repr__(self) -> str:
        label = "" if self._name is None else f", name={self._name!r}"
        return (
            f"Timelines({len(self)} histories, end={self.end:g}, "
            f"{self._data.times.size} changes{label})"
        )


def _stacked(parts: Sequence[_Data], leaves, end=None) -> _Data:
    """Histories one after another (their causes already in ``leaves``)."""
    if not parts:
        raise ValueError("No histories.")
    end = parts[0].end if end is None else end
    counts = np.concatenate([np.diff(p.offsets) for p in parts])
    offsets = np.zeros(counts.size + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    return _Data(
        np.concatenate([p.start for p in parts]).astype(np.int8),
        offsets,
        np.concatenate([p.times for p in parts]),
        np.concatenate([p.causes for p in parts]),
        np.concatenate([p.index for p in parts]),
        np.concatenate([p.planned for p in parts]),
        leaves,
        end,
    )


def _same(a, b) -> bool:
    """Whether two timelines hold the same histories, causes and names."""
    x, y = a._data, b._data
    return (
        a.name == b.name
        and x.end == y.end
        and np.array_equal(x.start, y.start)
        and np.array_equal(x.offsets, y.offsets)
        and np.array_equal(x.times, y.times)
        and np.array_equal(x.planned, y.planned)
        and _labels(a) == _labels(b)
    )


def _labels(timeline) -> list:
    """Each change's cause, by label."""
    data = timeline._data
    leaves = (timeline.name,) if data.leaves is None else data.leaves
    return [leaves[c] for c in data.causes.tolist()]


def _system_timeline(rbd, timelines: Mapping):
    """``RBD.system_timeline``: the system's timeline, merged up the
    diagram's decomposition (see ``modular``) from its components'."""
    if not isinstance(timelines, Mapping):
        raise TypeError(
            "Give the components' timelines as a mapping, {node: timeline}."
        )
    nodes = list(rbd.nodes)
    known = set(nodes)
    aliases = rbd._component_aliases()
    for node in timelines:
        if node in known:
            continue
        if node in aliases:
            raise ValueError(
                f"Node {node!r} repeats node {aliases[node]!r}: give the "
                f"timeline of {aliases[node]!r}, which it has wherever it is "
                "drawn."
            )
        if node in rbd.in_or_out:
            raise ValueError(
                f"The input or output node {node!r} passes whatever reaches "
                "it: it takes no timeline."
            )
        raise ValueError(f"Unknown node {node!r}: it is not in the diagram.")
    decomposition = rbd._decomposition()
    relevant = decomposition.nodes
    junctions = rbd._junctions()
    missing = [
        node
        for node in nodes
        if node in relevant and node not in junctions and node not in timelines
    ]
    if missing:
        raise ValueError(
            f"No timeline for node(s) {missing}: every component the system "
            "depends on needs one."
        )
    size, end, many = _shape(list(timelines.values()))
    given = {
        node: _tiled(item._data, size)
        for node, item in timelines.items()
        if node in relevant
    }
    system, _ = _system_merge(rbd, given, size, end)
    return _wrap(system, many)


def _system_merge(
    rbd, given: Mapping, size: int, end: float
) -> Tuple[_Data, np.ndarray]:
    """The system's histories, merged up the diagram's decomposition (see
    ``modular``) from ``given``, ``{node: data}``, each component's that
    the system depends on (all with ``size`` histories over ``[0, end]``;
    a junction's may be left out), each change a cause of its own: their
    causes the components' places in ``rbd.nodes``. And the histories in
    which changes of different components fall at the same time, which
    the merge takes in the order of the components."""
    from repyability.rbd.modular import KOON, NODE, PARALLEL, SERIES

    nodes = list(rbd.nodes)
    decomposition = rbd._decomposition()
    rank = {node: i for i, node in enumerate(nodes)}
    given = {node: _leaf(data, rank[node]) for node, data in given.items()}
    table, runs = _table(list(given.values()))
    same = (
        (table.history[1:] == table.history[:-1])
        & (table.times[1:] == table.times[:-1])
        & (table.causes[1:] != table.causes[:-1])
    )
    tied = np.unique(table.history[1:][same])
    leaf = dict(zip(given, runs))
    up = _Run(np.ones(size, np.int8), np.zeros(0, np.int64))
    value: Dict[int, _Run] = {}
    for i, term in enumerate(decomposition.terms):
        kind = term[0]
        if kind == NODE:
            value[i] = leaf.get(term[1], up)
        elif kind == SERIES:
            value[i] = _merged(
                [value[c] for c in term[1]], len(term[1]), table
            )
        elif kind == PARALLEL:
            value[i] = _merged([value[c] for c in term[1]], 1, table)
        else:
            assert kind == KOON
            value[i] = _merged([value[c] for c in term[1]], term[2], table)
    if decomposition.always_works:
        system = up
    elif decomposition.root is not None:
        system = value[decomposition.root]
    else:
        system = _core(decomposition, value, table, size)
    return _unrun(system, table, tuple(nodes), end), tied


def _core(decomposition, value: Dict[int, _Run], table: _Table, size: int):
    """The core's histories from its members' (terms of the decomposition):
    merged path set by path set when they are few enough, else decided at
    each of its members' changes."""
    if not decomposition.from_graph:
        paths = decomposition.core or []
        if not paths:
            return _Run(np.zeros(size, np.int8), np.zeros(0, np.int64))
        if sum(map(len, paths)) <= _PATH_MEMBERS:
            routes = [
                _merged([value[c] for c in path], len(path), table)
                for path in paths
            ]
            return _merged(routes, 1, table)
    return _decided(decomposition, value, table, size)


def _decided(decomposition, value: Dict[int, _Run], table: _Table, size: int):
    """The core's histories, deciding it at each of its members' changes,
    in the table's order (for a core with too many path sets to merge)."""
    from repyability.rbd import bdd

    if decomposition.from_graph:
        steps, root = decomposition.core_plan()
        members = sorted({pivot for pivot, _, _ in steps})
        position = {c: j for j, c in enumerate(members)}
        plan = ([(position[p], a, i) for p, a, i in steps], root)

        def decide(v) -> bool:
            return bdd.walk(plan, v)

    else:
        sets = decomposition.core
        members = sorted(set().union(*sets))
        position = {c: j for j, c in enumerate(members)}
        groups = [tuple(position[c] for c in group) for group in sets]

        def decide(v) -> bool:
            return any(all(v[i] for i in group) for group in groups)

    runs = [value[c] for c in members]
    starts = np.array([run.start for run in runs], np.int8).reshape(
        len(runs), size
    )
    start = np.array(
        [decide(starts[:, s].tolist()) for s in range(size)], np.int8
    )
    parts = [
        (run.key, _ups(run, table), np.full(run.key.size, member, np.int64))
        for member, run in enumerate(runs)
        if run.key.size
    ]
    if not parts:
        return _Run(start, np.zeros(0, np.int64))
    key, after, member = (np.concatenate(column) for column in zip(*parts))
    order = np.argsort(key, kind="stable")
    key = key[order]
    history = table.history[key].tolist()
    after = after[order].tolist()
    member = member[order].tolist()
    kept: List[int] = []
    states: list = []
    current = -1
    up = False
    for e in range(len(history)):
        s = history[e]
        if s != current:
            current = s
            states = starts[:, s].tolist()
            up = bool(start[s])
        states[member[e]] = after[e]
        now = decide(states)
        if now != up:
            kept.append(e)
            up = now
    return _Run(start, key[np.array(kept, np.int64)])


__all__ = ["Timeline", "Timelines", "series", "parallel", "k_out_of_n"]
