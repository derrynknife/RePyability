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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    # In each simulation, by time, then by rank (the sort is stable, so a
    # module's changes at one instant stay in their order).
    order = np.lexsort((module, rank, time, sim))
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
    before_first = np.zeros(n, dtype=np.int64)
    if bit.size:
        first = np.searchsorted(sim, np.arange(n), side="left")
        before_first = np.where(
            first > 0, running[np.maximum(first - 1, 0)], 0
        ).astype(np.int64)
    after = start[sim] ^ running ^ before_first[sim]
    before = after ^ bit
    # The stretches: from 0, and from each change, to the next or the end.
    seg_sim = np.concatenate([np.arange(n), sim])
    seg_start = np.concatenate([np.zeros(n), time])
    seg_state = np.concatenate([start, after])
    rank = np.concatenate(
        [np.zeros(n, np.int64), 1 + np.arange(sim.size, dtype=np.int64)]
    )
    order = np.lexsort((rank, seg_sim))
    seg_sim, seg_start, seg_state = (
        seg_sim[order],
        seg_start[order],
        seg_state[order],
    )
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
    points = curve_x.size
    for m, state in given.items():
        inside = p.segment_state == m
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
            ((p.before == m) & p.down, 1.0),
            ((p.after == m) & p.down, -1.0),
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
        closing = inside & p.last
        ending[p.segment_sim[closing]] = 1.0 - state.before(p.end)
        # How many simulations are in the state at each of the curve's
        # times (from just after a change there), each up with ``up``.
        low = np.searchsorted(curve_x, a, side="left")
        high = np.where(
            p.last[inside],
            points,
            np.searchsorted(curve_x, b, side="left"),
        )
        occupied = np.bincount(low, minlength=points + 1) - np.bincount(
            high, minlength=points + 1
        )
        count = np.cumsum(occupied)[:-1].astype(float)
        up = state.at("up", curve_x)
        curve += count * up
        square += count * up * up
    restorations = failures + planned - ending
    return Values(
        uptime, failures, planned, restorations, curve / n, square / n
    )


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


def curve_se(curve: np.ndarray, square: np.ndarray, n: int) -> np.ndarray:
    """The pointwise standard error of a conditional curve: of the mean
    of the simulations' values, from their mean square."""
    if n < 2:
        return np.zeros_like(curve)
    variance = np.maximum(square - curve * curve, 0.0) * n / (n - 1)
    return np.sqrt(variance / n)


def steps(curve_points: Optional[int], end: float) -> np.ndarray:
    """The curve's times: ``curve_points`` steps (by default ``CURVE``)."""
    points = CURVE if curve_points is None else int(curve_points)
    return end * np.arange(points + 1) / points
