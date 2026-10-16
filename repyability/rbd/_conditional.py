"""Conditional Monte Carlo over a repairable diagram's dependent modules
(#189): simulate only them, and take the rest exactly given their states.

A system with a dependency (a standby group of other lives, a nested RBD
that needs simulating, imperfect repair, a maintenance group) is mostly
independent components, whose point availabilities and expected events are
worked out exactly. The structure function is multilinear in each node's
state, and the independent components are independent of the modules, so
given the modules' states ``m`` at ``t`` the system is up with probability
``a_m(t)``: its availability with the modules held in ``m`` and every other
node at its point availability then. A simulation of the modules alone
then contributes its expected values given their histories (Rao-Blackwell):
its up time is the sum, over the stretches between the modules' changes,
of ``a_m``'s integral there; its failures, those the independent
components cause in each stretch (the time-dependent Birnbaum formula with
the modules held) and, at each change of a module down, the chance it takes
the system down, ``a_before - a_after``; and so on. Each estimate is
unbiased, and varies less than a plain simulation's (by the law of total
variance, the independent components' share is gone), but each
simulation's value is its expected value given the modules, whose spread
is less than the window's own.

Changes at one instant are taken as the event loop takes them: the
modules' failures, then their planned outages (a maintenance group's stop
follows the failure that opens it), then their restorations; and a
module's change before the other nodes' at that instant (a scheduled
replacement they share, a unit dead on arrival at 0), which the stretch
from it counts. So the chance a module going down takes the system down is
read just before the instant (``Given.before``), and at 0 from every other
node new.

Each joint state the modules meet has its integrals worked out once, on a
grid (``_STEPS`` steps, finer near 0, and the curves' breaks with the
instants just before and after them), and read off it: the up time by the
cubic through its values and slopes (the point availability) at the ends of
the step a time falls in, the counts linearly.
"""

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

#: The steps of a conditional run's curve without ``curve_points``.
CURVE = 1000

#: The grid the integrals given each joint state are worked out on: its
#: even steps, and the steps of its finer start (geometric, from
#: ``_FIRST`` of the window), where the curves fall from new.
_STEPS = 1024
_START_STEPS = 256
_FIRST = 1e-7


@dataclass
class Paths:
    """The modules' joint histories in a run's simulations (see
    ``paths``): each change of a module (``sim``, ``time``, ``module``,
    whether it went ``down``, and whether that was ``planned``), in order,
    with the joint state ``before`` and ``after`` it (a bitmask, bit ``j``
    set while module ``j`` is up); each simulation's state at 0
    (``start``); and the stretches between changes (``segment_sim``,
    ``segment_start``, ``segment_end``, ``segment_state``, and ``last``,
    whether it ends the window)."""

    n: int
    end: float
    sim: np.ndarray
    time: np.ndarray
    module: np.ndarray
    down: np.ndarray
    planned: np.ndarray
    before: np.ndarray
    after: np.ndarray
    start: np.ndarray
    segment_sim: np.ndarray
    segment_start: np.ndarray
    segment_end: np.ndarray
    segment_state: np.ndarray
    last: np.ndarray

    @property
    def states(self) -> np.ndarray:
        """The joint states met: in a stretch, or before a change."""
        return np.unique(np.concatenate([self.segment_state, self.before]))


def paths(histories: List[Any], n: int, end: float) -> Paths:
    """The modules' joint histories from each module's (``_Data``, one
    history per simulation, see ``repyability.timelines``). Changes of
    several modules at one instant are taken as the event loop takes them:
    failures first, then planned outages (a maintenance group's stop
    follows the failure that opens it), then restorations, a module's own
    changes at one instant together and in their order."""
    sims, times, modules, downs, planned, ranks = [], [], [], [], [], []
    start = np.zeros(n, dtype=np.int64)
    for j, data in enumerate(histories):
        counts = np.diff(data.offsets)
        sim = np.repeat(np.arange(n), counts)
        place = np.arange(data.times.size) - np.repeat(
            data.offsets[:-1], counts
        )
        up = np.repeat(data.start.astype(bool), counts)
        # A history's changes alternate, from its state at 0.
        down = np.where(up, place % 2 == 0, place % 2 == 1)
        plan = np.asarray(data.planned, dtype=bool)
        at = np.asarray(data.times, dtype=float)
        # Its changes at one instant take the rank of the first of them.
        rank = np.where(down, np.where(plan, 1, 0), 2)
        first = np.ones(at.size, dtype=bool)
        first[1:] = (sim[1:] != sim[:-1]) | (at[1:] != at[:-1])
        rank = rank[np.flatnonzero(first)][np.cumsum(first) - 1]
        sims.append(sim)
        times.append(at)
        modules.append(np.full(at.size, j, dtype=np.int64))
        downs.append(down)
        planned.append(plan)
        ranks.append(rank)
        start |= data.start.astype(np.int64) << j
    sim = np.concatenate(sims) if sims else np.empty(0, np.int64)
    time = np.concatenate(times) if times else np.empty(0)
    module = np.concatenate(modules) if modules else np.empty(0, np.int64)
    down = np.concatenate(downs) if downs else np.empty(0, bool)
    plan = np.concatenate(planned) if planned else np.empty(0, bool)
    rank = np.concatenate(ranks) if ranks else np.empty(0, np.int64)
    order = _in_order(sim, time, rank, float(end))
    sim, time, module, down, plan = (
        sim[order],
        time[order],
        module[order],
        down[order],
        plan[order],
    )
    bit = np.left_shift(np.int64(1), module)
    # The modules each simulation has flipped by each change: a running xor
    # over all changes, less the simulation's own start in it.
    running = np.bitwise_xor.accumulate(bit) if bit.size else bit
    first = np.searchsorted(sim, np.arange(n), side="left")
    before_first = np.zeros(n, dtype=np.int64)
    if bit.size:
        before_first = np.where(
            first > 0, running[np.maximum(first - 1, 0)], 0
        ).astype(np.int64)
    after = start[sim] ^ running ^ before_first[sim]
    before = after ^ bit
    # The stretches: from 0, and from each change, to the next or the end,
    # in each simulation from 0. A simulation's start goes before its
    # first change, the changes already in order (#248).
    opens = first + np.arange(n)
    follows = np.arange(sim.size) + sim + 1
    seg_sim = np.empty(n + sim.size, np.int64)
    seg_start = np.empty(n + sim.size)
    seg_state = np.empty(n + sim.size, np.int64)
    seg_sim[opens], seg_sim[follows] = np.arange(n), sim
    seg_start[opens], seg_start[follows] = 0.0, time
    seg_state[opens], seg_state[follows] = start, after
    last = np.ones(seg_sim.size, dtype=bool)
    last[:-1] = seg_sim[1:] != seg_sim[:-1]
    seg_end = np.full(seg_sim.size, float(end))
    seg_end[:-1] = np.where(last[:-1], float(end), seg_start[1:])
    return Paths(
        n,
        float(end),
        sim,
        time,
        module,
        down,
        plan,
        before,
        after,
        start,
        seg_sim,
        seg_start,
        seg_end,
        seg_state,
        last,
    )


def _in_order(
    sim: np.ndarray, time: np.ndarray, rank: np.ndarray, end: float
) -> np.ndarray:
    """The order of the modules' changes (given module after module, each
    module's in order): by simulation and time, and at one instant of one
    simulation by rank, then as given (a module's changes at one instant
    stay together, in their order).

    One sort (#248), of a key that grows with the simulation and the time:
    each simulation's times shifted past the one before's, by a power of
    two beyond the window, so the shift is exact and rounding can make
    nearby keys equal but never put two out of order. Each run of equal
    keys (changes at one instant, or within rounding) is then put in
    order exactly, so the order is the same whatever the sort does with
    them."""
    span = 2.0 ** (math.floor(math.log2(end)) + 1)
    key = sim * span + time
    order = np.argsort(key)
    ranked = key[order]
    tied = ranked[1:] == ranked[:-1]
    if tied.any():
        group = np.cumsum(np.concatenate(([True], ~tied)))
        together = np.zeros(order.size, bool)
        together[1:] |= tied
        together[:-1] |= tied
        places = np.flatnonzero(together)
        rows = order[places]
        exact = np.lexsort(
            (rows, rank[rows], time[rows], sim[rows], group[places])
        )
        order[places] = rows[exact]
    return order


def _members(keys: np.ndarray):
    """``members(m)``: the places of ``keys`` equal to ``m``, in order (one
    stable sort, rather than comparing every key with every ``m``)."""
    order = np.argsort(keys, kind="stable")
    ranked = keys[order]

    def members(m) -> np.ndarray:
        lo = np.searchsorted(ranked, m, side="left")
        hi = np.searchsorted(ranked, m, side="right")
        return order[lo:hi]

    return members


def grid(end: float, breaks: List[np.ndarray]) -> np.ndarray:
    """The times the integrals are worked out at (see the module's
    notes): even steps over ``[0, end]``, geometric ones near 0, and each
    of the ``breaks`` with the instants just before and after it (so that
    a step's slopes are the curve's on it), and the instant before the
    end (a break there too, at a multiple of a scheduled interval)."""
    marks = [
        np.linspace(0.0, end, _STEPS + 1),
        end * np.geomspace(_FIRST, 1.0 / 16.0, _START_STEPS),
        np.array([np.nextafter(end, -math.inf)]),
    ]
    for times in breaks:
        times = np.asarray(times, dtype=float)
        times = times[(times > 0.0) & (times < end)]
        marks += [
            np.nextafter(times, -math.inf),
            times,
            np.nextafter(times, math.inf),
        ]
    return np.unique(np.concatenate(marks))


@dataclass
class Given:
    """The system given one joint state of the modules, on the grid
    ``x``: its expected up time, failures and planned outages before each
    time (``uptime``, ``failures``, ``planned``), its point availability
    (``up``), and its state just before 0, every other node new
    (``start``)."""

    x: np.ndarray
    uptime: np.ndarray
    failures: np.ndarray
    planned: np.ndarray
    up: np.ndarray
    start: float

    def before(self, t) -> np.ndarray:
        """The point availability just before the times ``t``: before the
        other nodes' changes at them (a scheduled replacement, a unit dead
        on arrival), which the grid holds the instant before of."""
        t = np.asarray(t, dtype=float)
        just = np.interp(np.nextafter(t, -math.inf), self.x, self.up)
        return np.where(t > 0.0, just, self.start)

    def at(self, name: str, t) -> np.ndarray:
        """``name`` at the times ``t``: the up time by the cubic through
        its values and slopes at the ends of their steps (its slope the
        point availability, from just after each time), the others
        linearly."""
        t = np.asarray(t, dtype=float)
        if name != "uptime":
            return np.interp(t, self.x, getattr(self, name))
        x = self.x
        i = np.clip(np.searchsorted(x, t, side="right") - 1, 0, x.size - 2)
        h = x[i + 1] - x[i]
        s = (t - x[i]) / np.where(h > 0.0, h, 1.0)
        s2, s3 = s * s, s * s * s
        return (
            (2.0 * s3 - 3.0 * s2 + 1.0) * self.uptime[i]
            + (s3 - 2.0 * s2 + s) * h * self.up[i]
            + (3.0 * s2 - 2.0 * s3) * self.uptime[i + 1]
            + (s3 - s2) * h * self.up[i + 1]
        )


@dataclass
class Values:
    """Each simulation's expected values given its modules' histories:
    its up time, system failures, planned outages and restorations; and
    the curve on ``curve_x``, its mean and mean square over the
    simulations."""

    uptime: np.ndarray
    failures: np.ndarray
    planned: np.ndarray
    restorations: np.ndarray
    curve: np.ndarray
    curve_square: np.ndarray


def values(p: Paths, given: Dict[int, Given], curve_x: np.ndarray) -> Values:
    """Each simulation's expected values (see ``Values``), from the system
    ``given`` each joint state its modules meet."""
    n = p.n
    uptime, failures = np.zeros(n), np.zeros(n)
    planned, ending = np.zeros(n), np.zeros(n)
    curve = np.zeros(curve_x.size)
    square = np.zeros(curve_x.size)
    # Each state's stretches, and the changes down from and into it, found
    # once (#248).
    stretches = _members(p.segment_state)
    downs = np.flatnonzero(p.down)
    leaving, entering = _members(p.before[downs]), _members(p.after[downs])
    for m, state in given.items():
        inside = stretches(m)
        sims = p.segment_sim[inside]
        a, b = p.segment_start[inside], p.segment_end[inside]
        for name, total in (
            ("uptime", uptime),
            ("failures", failures),
            ("planned", planned),
        ):
            total += np.bincount(
                sims,
                weights=state.at(name, b) - state.at(name, a),
                minlength=n,
            )
        # A module going down takes the system down with the chance it was
        # up and is not now: a failure, or a planned outage. It is taken
        # before the other nodes' changes at that instant, which the
        # stretch from it counts.
        for rows, sign in (
            (downs[leaving(m)], 1.0),
            (downs[entering(m)], -1.0),
        ):
            jump = sign * state.before(p.time[rows])
            unplanned = ~p.planned[rows]
            failures += np.bincount(
                p.sim[rows][unplanned], weights=jump[unplanned], minlength=n
            )
            planned += np.bincount(
                p.sim[rows][~unplanned], weights=jump[~unplanned], minlength=n
            )
        # Down at the end: an outage not restored in the window (whose
        # changes at its end are past it).
        closing = inside[p.last[inside]]
        ending[p.segment_sim[closing]] = 1.0 - state.before(p.end)
        # How many simulations are in the state at each of the curve's
        # times, each up with ``up``.
        count = _occupancy(p, inside, curve_x)
        up = state.at("up", curve_x)
        curve += count * up
        square += count * up * up
    restorations = failures + planned - ending
    return Values(
        uptime, failures, planned, restorations, curve / n, square / n
    )


def _occupancy(
    p: Paths, inside: np.ndarray, curve_x: np.ndarray
) -> np.ndarray:
    """How many simulations are in the stretches ``inside`` (their places)
    at each of the curve's times (from just after a change there)."""
    points = curve_x.size
    low = np.searchsorted(curve_x, p.segment_start[inside], side="left")
    high = np.where(
        p.last[inside],
        points,
        np.searchsorted(curve_x, p.segment_end[inside], side="left"),
    )
    occupied = np.bincount(low, minlength=points + 1) - np.bincount(
        high, minlength=points + 1
    )
    return np.cumsum(occupied)[:-1].astype(float)


@dataclass
class CapacityGiven:
    """The system's capacity given one joint state of the modules, on the
    grid ``x`` (see ``capacity_given``): its ``levels``, the time at each
    before each time (one row per level), the amount delivered before each
    time (of ``min(capacity, demand)``, None without a demand), the mean
    of its finite levels at each time and the chance it is unlimited."""

    x: np.ndarray
    levels: np.ndarray
    time: np.ndarray
    delivered: Optional[np.ndarray]
    mean: np.ndarray
    unlimited: np.ndarray


def capacity_given(
    x: np.ndarray,
    levels: np.ndarray,
    probabilities: np.ndarray,
    demand: Optional[float],
) -> CapacityGiven:
    """The system's capacity given a joint state of the modules (see
    ``CapacityGiven``), from its distribution at each time of the grid
    ``x`` (``levels``, and one row of ``probabilities`` per level): the
    integrals by the trapezium rule between the grid's times, which hold
    the instants either side of each break."""
    levels = np.asarray(levels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    step = np.diff(x)

    def cumulative(rate: np.ndarray) -> np.ndarray:
        pieces = 0.5 * (rate[..., 1:] + rate[..., :-1]) * step
        return np.concatenate(
            [np.zeros(rate.shape[:-1] + (1,)), np.cumsum(pieces, axis=-1)],
            axis=-1,
        )

    finite = np.isfinite(levels)
    delivered = None
    if demand is not None:
        served = np.where(finite, np.minimum(levels, demand), demand)
        delivered = cumulative(served @ probabilities)
    finite_levels = np.where(finite, levels, 0.0)
    return CapacityGiven(
        x,
        levels,
        cumulative(probabilities),
        delivered,
        finite_levels @ probabilities,
        probabilities[~finite].sum(axis=0),
    )


@dataclass
class CapacityValues:
    """A round's capacities (see ``capacity_values``): the time at each
    level summed over its simulations, each simulation's delivered amount
    (None without a demand), and on the curve's times the sum over the
    simulations of each one's expected capacity and how many may carry an
    unlimited amount."""

    time_at: Dict[float, float]
    delivered: Optional[np.ndarray]
    curve: np.ndarray
    unlimited: np.ndarray


def capacity_values(
    p: Paths, given: Dict[int, CapacityGiven], curve_x: np.ndarray
) -> CapacityValues:
    """Each simulation's expected capacities given its modules' histories
    (see ``CapacityValues``), from the system's capacity ``given`` each
    joint state its modules meet."""
    n = p.n
    time_at: Dict[float, float] = {}
    delivered: Optional[np.ndarray] = None
    curve = np.zeros(curve_x.size)
    unlimited = np.zeros(curve_x.size)
    stretches = _members(p.segment_state)
    for m, state in given.items():
        inside = stretches(m)
        sims = p.segment_sim[inside]
        a, b = p.segment_start[inside], p.segment_end[inside]
        for row, level in zip(state.time, state.levels):
            spent = np.interp(b, state.x, row) - np.interp(a, state.x, row)
            time_at[float(level)] = time_at.get(float(level), 0.0) + math.fsum(
                spent
            )
        if state.delivered is not None:
            if delivered is None:
                delivered = np.zeros(n)
            delivered += np.bincount(
                sims,
                weights=np.interp(b, state.x, state.delivered)
                - np.interp(a, state.x, state.delivered),
                minlength=n,
            )
        count = _occupancy(p, inside, curve_x)
        curve += count * np.interp(curve_x, state.x, state.mean)
        unlimited += count * (
            np.interp(curve_x, state.x, state.unlimited) > 0.0
        )
    return CapacityValues(time_at, delivered, curve, unlimited)


def module_totals(
    histories: List[Any], n: int, end: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Each module's up time, summed over the simulations, and its down
    time."""
    up = []
    for data in histories:
        counts = np.diff(data.offsets)
        place = np.arange(data.times.size) - np.repeat(
            data.offsets[:-1], counts
        )
        was_up = np.repeat(data.start.astype(bool), counts)
        down = np.where(was_up, place % 2 == 0, place % 2 == 1)
        # Up time: from 0 (if up) and each change up, to each change down
        # and the end.
        total = float(end) * float(np.sum(data.start))
        signed = np.where(down, -(end - data.times), end - data.times)
        total += float(np.sum(signed))
        up.append(total)
    uptime = np.array(up, dtype=float)
    return uptime, n * float(end) - uptime


def steps(curve_points: Optional[int], end: float) -> np.ndarray:
    """The curve's times: ``curve_points`` steps (by default ``CURVE``)."""
    points = CURVE if curve_points is None else int(curve_points)
    return end * np.arange(points + 1) / points


class History(NamedTuple):
    """A module's histories in a shard's simulations (what ``paths`` and
    ``module_totals`` read of a ``repyability.timelines._Data``): whether
    each starts up, where its changes start, their times, and which are
    planned."""

    start: np.ndarray
    offsets: np.ndarray
    times: np.ndarray
    planned: np.ndarray


#: What a module shard's partial says it is (see ``partial_bytes``).
_PARTIAL = "RePyabilityModulePartial"


def partial_bytes(
    histories: List[Any],
    tally,
    start: int,
    stop: int,
    nodes: List[Any],
    priced: bool,
) -> bytes:
    """A module shard's partial: the modules' histories in its
    simulations ``start`` to ``stop - 1``, each simulation's own cost (if
    ``priced``), and the costs by category and by module (in the order of
    ``nodes``), as the bytes of a NumPy ``.npz`` file, read without
    pickle."""
    arrays: Dict[str, Any] = {
        "kind": np.array([_PARTIAL]),
        "range": np.array([start, stop], dtype=np.int64),
    }
    for j, data in enumerate(histories):
        arrays[f"start_{j}"] = np.asarray(data.start, dtype=np.int8)
        arrays[f"offsets_{j}"] = np.asarray(data.offsets, dtype=np.int64)
        arrays[f"times_{j}"] = np.asarray(data.times, dtype=float)
        arrays[f"planned_{j}"] = np.asarray(data.planned, dtype=bool)
    costs = np.zeros(stop - start)
    names: List[str] = []
    spent: List[float] = []
    by_node = np.zeros(len(nodes))
    costed = np.zeros(len(nodes), dtype=bool)
    if priced:
        tally._fold()
        costs = np.asarray(tally.cost_samples, dtype=float)
        for key, value in tally.cost_by_category.items():
            names.append(str(key))
            spent.append(float(value))
        for j, node in enumerate(nodes):
            if node in tally.cost_by_component:
                by_node[j] = float(tally.cost_by_component[node])
                costed[j] = True
    arrays["costs"] = costs
    arrays["category_names"] = np.array(names, dtype=str)
    arrays["category_costs"] = np.array(spent, dtype=float)
    arrays["module_costs"] = by_node
    arrays["module_costed"] = costed
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def from_partial(saved: bytes, modules: int):
    """A module shard's partial (see ``partial_bytes``): its range, the
    ``modules`` histories, each simulation's own cost, the costs by
    category and by module, and which modules the costs by module hold."""
    with np.load(io.BytesIO(saved), allow_pickle=False) as data:
        if str(data["kind"][0]) != _PARTIAL:
            raise ValueError(
                "This is not a module shard's partial: shard_map must give "
                "back what run_shard returns."
            )
        start, stop = (int(v) for v in data["range"])
        histories = [
            History(
                data[f"start_{j}"].copy(),
                data[f"offsets_{j}"].copy(),
                data[f"times_{j}"].copy(),
                data[f"planned_{j}"].copy(),
            )
            for j in range(modules)
        ]
        categories = dict(
            zip(
                (str(name) for name in data["category_names"]),
                (float(value) for value in data["category_costs"]),
            )
        )
        return (
            (start, stop),
            histories,
            data["costs"].copy(),
            categories,
            data["module_costs"].copy(),
            data["module_costed"].copy(),
        )


def joined(parts: List[List[History]]) -> List[History]:
    """Each module's histories over consecutive shards, in order."""
    if len(parts) == 1:
        return parts[0]
    out = []
    for j in range(len(parts[0])):
        pieces = [part[j] for part in parts]
        offsets = [np.zeros(1, np.int64)]
        base = 0
        for piece in pieces:
            offsets.append(piece.offsets[1:] + base)
            base += int(piece.offsets[-1])
        out.append(
            History(
                np.concatenate([piece.start for piece in pieces]),
                np.concatenate(offsets),
                np.concatenate([piece.times for piece in pieces]),
                np.concatenate([piece.planned for piece in pieces]),
            )
        )
    return out
