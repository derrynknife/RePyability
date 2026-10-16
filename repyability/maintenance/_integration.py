"""Phase 2 of the calendar policy (#319, UC-13): the same problem as the
dynamic programme on a grid (``_replacement``), with continuous time
between opportunities and no replacement on any day (a free boundary
inside a gap, left to the grid).

At opportunity ``k`` (time ``t_k``, price ``c_k``, gap ``Δ`` to the next),
with ``V_{k+1}`` the cost to go at the next by the part's age:

    G(s)   = ∫_s^{t_{k+1}} f(u - s) [c_f + G(u)] du
             + R(t_{k+1} - s) V_{k+1}(t_{k+1} - s)
    W_k(a) = ∫_0^Δ f(a + τ) / R(a) [c_f + G(t_k + τ)] dτ
             + R(a + Δ) / R(a) V_{k+1}(a + Δ)
    V_k(a) = min(W_k(a), c_k + W_k(0))

``G(s)`` is the cost to go of a new unit put in at ``s`` in the gap (after
a failure), a Volterra equation solved backward on ``NODES + 1`` nodes:
each integral by product integration, the distribution's increase over a
piece times the mean of the integrand's ends (exact in the density, so a
density unbounded at 0 is integrated as well as any). ``V_k`` lives on an
age grid of ``AGES`` points, linearly interpolated. A gap's matrices
depend on its length alone, so gaps of one length share them.

Over a periodic calendar, relative value iteration across one cycle (as
``_replacement._relative_values``); a plan of given dates is repeated
past its last date. The threshold at an opportunity is where
``W_k(a) = c_k + W_k(0)``, by bisection.
"""

import math
from fractions import Fraction
from typing import Dict

import numpy as np

from repyability.maintenance._calendar import (
    ANYTIME,
    _common_multiple,
    _fraction,
)
from repyability.maintenance._replacement import (
    FAR,
    MAX_LIVES,
    TOLERANCE,
    CalendarPolicy,
    _age_where,
    _mean,
    _survival,
)

#: Quadrature pieces in each gap between opportunities.
NODES = 32
#: Points of the age grid the cost to go is kept on, at least (more where
#: the shortest gap needs them: the spacing is a whole fraction of it).
AGES = 801
#: The most points the age grid may have.
MAX_AGES = 20_000
#: The most opportunities a cycle may have.
MAX_OPPORTUNITIES = 20_000
#: Bisection steps for a threshold.
HALVINGS = 60


def _cycle(calendar, repeat) -> tuple:
    """The opportunities over one cycle: its length, and their times in
    ``[0, length)``, prices and kinds (a time with several takes the
    cheapest)."""
    periods = [o.period for o in calendar.opportunities if o.periodic]
    if repeat is not None:
        periods.append(repeat)
    length = float(_common_multiple(periods))
    times, prices, kinds = [], [], []
    for k, o in enumerate(calendar.opportunities):
        if o.periodic:
            count = Fraction(_fraction(length)) / _fraction(o.period)
            if count > MAX_OPPORTUNITIES:
                raise NotImplementedError(
                    "method='integration' needs a cycle of at most "
                    f"{MAX_OPPORTUNITIES} opportunities; this one has more: "
                    "use method='dp' or 'sampling'."
                )
            local = o.offset % o.period + np.arange(int(count)) * o.period
        else:
            dates = np.asarray(o.dates, float) % repeat
            copies = int(round(length / repeat))
            local = (
                dates[None, :] + repeat * np.arange(copies)[:, None]
            ).ravel()
        local = np.mod(local, length)
        times.append(local)
        prices.append(np.full(local.size, o.cost))
        kinds.append(np.full(local.size, k))
    t = np.concatenate(times)
    c = np.concatenate(prices)
    kind = np.concatenate(kinds)
    order = np.lexsort((c, t))
    t, c, kind = t[order], c[order], kind[order]
    tolerance = 1e-9 * length
    keep = np.concatenate([[True], np.diff(t) > tolerance])
    if t.size > 1 and length - t[-1] + t[0] <= tolerance:
        keep[-1] = False
    names = [o.name for o in calendar.opportunities] + [ANYTIME]
    return length, t[keep], c[keep], kind[keep], names


class _Gap:
    """A gap's quadrature, by its length: the matrices that give ``G`` at
    its nodes and ``W`` on the age grid from the cost to go at its end."""

    def __init__(self, length, sf, ages, failure):
        self.length = length
        x = np.linspace(0.0, length, NODES + 1)
        self.x = x
        m = NODES
        # G at node i: a new unit at x_i, to the end at x_m.
        left = x[None, :-1] - x[:, None]
        right = x[None, 1:] - x[:, None]
        weights = np.where(
            left >= 0.0,
            sf(np.clip(left, 0.0, None)) - sf(np.clip(right, 0.0, None)),
            0.0,
        )
        a = np.eye(m + 1)
        a[:m, :m] -= 0.5 * weights[:m]
        a[:m, 1:] -= 0.5 * weights[:m]
        self.a_inverse = np.linalg.inv(a)
        self.failed_new = failure * weights[:m].sum(axis=1)
        self.remaining = length - x[:m]
        self.lasting_new = sf(self.remaining)
        # W on the age grid: a unit of each age kept through the gap.
        self.ages = ages
        self.failure = failure
        self.sf = sf
        self.q, self.lasting, self.later = self._kept(ages)
        # Forward, the same operators transposed: a new unit at each node
        # is, at the gap's end, of each age with ``ends`` (how G takes the
        # cost to go there), after ``renewals`` failures on the way.
        reach = np.zeros((m + 1, ages.size))
        reach[:m] = self.lasting_new[:, None] * _spread(self.remaining, ages)
        reach[m, 0] = 1.0
        self.ends = self.a_inverse @ reach
        once = np.zeros(m + 1)
        once[:m] = weights[:m].sum(axis=1)
        renewals = self.a_inverse @ once
        self.fails = self.q @ (1.0 + 0.5 * (renewals[:-1] + renewals[1:]))
        self.kept_to = _nearest(self.later, ages)

    def _kept(self, ages):
        held = self.sf(ages)
        reached = self.sf(ages[:, None] + self.x[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(
                held[:, None] > 0.0, reached / held[:, None], 0.0
            )
        q = scaled[:, :-1] - scaled[:, 1:]
        dead = held <= 0.0
        q[dead] = 0.0
        q[dead, 0] = 1.0
        return q, scaled[:, -1], ages + self.length

    def new_unit(self, after: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """``G`` at the gap's nodes, given the cost to go ``after`` (on
        the age grid ``grid``) at its end."""
        rhs = np.empty(NODES + 1)
        rhs[:NODES] = self.failed_new + self.lasting_new * np.interp(
            self.remaining, grid, after
        )
        rhs[NODES] = after[0]
        return self.a_inverse @ rhs

    def kept(self, g_nodes, after, grid, ages=None) -> np.ndarray:
        """``W`` at ``ages`` (the age grid by default)."""
        if ages is None:
            q, lasting, later = self.q, self.lasting, self.later
        else:
            q, lasting, later = self._kept(ages)
        middle = self.failure + 0.5 * (g_nodes[:-1] + g_nodes[1:])
        return q @ middle + lasting * np.interp(later, grid, after)

    def forward(self, share: np.ndarray):
        """The chance of the unit in place being of each age at the gap's
        end, from ``share`` at its start, and the expected failures in
        it."""
        pieces = share @ self.q
        nodes = np.zeros(NODES + 1)
        nodes[:-1] += 0.5 * pieces
        nodes[1:] += 0.5 * pieces
        kept = share * self.lasting
        j, part = self.kept_to
        size = share.size
        later = nodes @ self.ends
        later += np.bincount(j, kept * (1.0 - part), minlength=size)
        later += np.bincount(j + 1, kept * part, minlength=size)
        return later, float(share @ self.fails)


def _nearest(points: np.ndarray, grid: np.ndarray):
    """Where linear interpolation on ``grid`` takes each of ``points``
    from, as ``np.interp`` does (a point past the grid's end from its last
    age): the index below it, and its share of the way to the next."""
    x = np.clip(points, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    return j, (x - grid[j]) / (grid[j + 1] - grid[j])


def _spread(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The linear interpolation's weights of ``points`` on ``grid``, as a
    matrix (each row sums to 1)."""
    j, part = _nearest(points, grid)
    out = np.zeros((points.size, grid.size))
    rows = np.arange(points.size)
    out[rows, j] = 1.0 - part
    out[rows, j + 1] += part
    return out


class _OpportunityTable:
    """The thresholds at the opportunities of a cycle, from ``origin``
    (a repeated plan's first date: none before it)."""

    def __init__(
        self,
        length,
        times,
        thresholds,
        kinds,
        names,
        origin=0.0,
        prices=None,
        chance=None,
    ):
        self.length, self.times = length, times
        self.thresholds, self.kinds, self.names = thresholds, kinds, names
        self.origin = origin
        self.prices = (
            np.zeros(times.size) if prices is None else np.asarray(prices)
        )
        self.chance = (
            np.zeros(times.size) if chance is None else np.asarray(chance)
        )

    def _index(self, t) -> np.ndarray:
        """The index in ``times`` of the opportunity at each of ``t``
        (within a tolerance), -1 where none falls, or before
        ``origin``."""
        t = np.asarray(t, float)
        index = np.full(t.shape, -1, np.int64)
        if self.times.size == 0:
            return index
        local = np.mod(t, self.length)
        where = np.searchsorted(self.times, local)
        tolerance = 1e-9 * self.length
        for shift in (0, -1):
            i = np.clip(where + shift, 0, self.times.size - 1)
            hit = np.abs(self.times[i] - local) <= tolerance
            hit |= np.abs(self.times[i] - local - self.length) <= tolerance
            hit |= np.abs(self.times[i] - local + self.length) <= tolerance
            index = np.where(hit, i, index)
        before = t < self.origin - tolerance
        return np.where(before, -1, index)

    def at(self, t):
        index = self._index(t)
        if self.times.size == 0:
            return np.full(index.shape, math.inf), index
        found = index >= 0
        i = np.maximum(index, 0)
        out = np.where(found, self.thresholds[i], math.inf)
        kinds = np.where(found, self.kinds[i], -1)
        return out, kinds

    def price_at(self, t: float) -> float:
        """The price of the opportunity at ``t``, ``inf`` if none falls
        there (this table prices no replacement on any day)."""
        index = int(self._index(t))
        return math.inf if index < 0 else float(self.prices[index])

    def next_planned(self, installed, after, oldest):
        """The first opportunity at or after ``after`` (strictly after, if
        the unit was put in then) at which a unit put in at ``installed``
        is replaced, its price and kind; ``(inf, 0.0, None)`` if none
        does. A unit past ``oldest`` (the oldest age worked out) is past
        every threshold, so the opportunities are searched to where the
        unit reaches that age, or a cycle ahead if it already has."""
        if self.times.size == 0:
            return math.inf, 0.0, None
        start = max(after, self.origin)
        first = math.floor((start - self.times[-1]) / self.length)
        end = max(installed + oldest, start) + self.length
        last = math.ceil((end - self.times[0]) / self.length)
        cycles = np.arange(first, last + 1) * self.length
        times = (cycles[:, None] + self.times[None, :]).ravel()
        thresholds = np.tile(self.thresholds, cycles.size)
        prices = np.tile(self.prices, cycles.size)
        kinds = np.tile(self.kinds, cycles.size)
        tolerance = 1e-9 * self.length
        later = (
            (times > after + tolerance)
            if after <= installed
            else (times >= after - tolerance)
        )
        due = later & (times >= self.origin - tolerance)
        due &= times - installed >= thresholds - tolerance
        if not due.any():
            return math.inf, 0.0, None
        i = int(np.argmax(due))
        return float(times[i]), float(prices[i]), self.names[kinds[i]]

    def schedule(self, until, wanted):
        out = []
        cycles = int(math.ceil(until / self.length))
        for c in range(cycles):
            for t, a, k, p in zip(
                self.times, self.thresholds, self.kinds, self.chance
            ):
                time = c * self.length + t
                name = self.names[k]
                if (
                    time >= until
                    or time < self.origin - 1e-9 * self.length
                    or (wanted is not None and name not in wanted)
                ):
                    continue
                out.append((float(time), name, float(a), float(p)))
        return out


def integrated_policy(
    part, calendar, failure, *, max_age=None, tail="repeat", life=None
):
    """The calendar policy between opportunities in continuous time (see
    the module's docstring)."""
    sf = _survival(part)
    mean = _mean(part, sf)
    if not calendar.opportunities:
        policy = CalendarPolicy(
            cost_rate=failure / mean,
            cost_rate_se=None,
            replacement_rates={"failure": 1.0 / mean},
            cost_rates={"failure": failure / mean},
            method="integration",
            basis="exact",
            step=None,
            period=None,
            determined_until=None,
            threshold_only=True,
            settings={"opportunities": 0},
            life=life,
        )
        table = _OpportunityTable(
            1.0, np.array([]), np.array([]), np.array([], np.int64), [ANYTIME]
        )
        return policy._bind(table, [ANYTIME], None, 0.0)
    repeat = None
    if not calendar.periodic:
        if tail != "repeat":
            raise NotImplementedError(
                "method='integration' repeats a plan of given dates past "
                "its last date; for tail='anytime' use method='dp'."
            )
        repeat = calendar._repeat()
    length, times, prices, kinds, names = _cycle(calendar, repeat)
    oldest = (
        float(max_age) if max_age is not None else _age_where(sf, FAR, mean)
    )
    gaps = np.diff(np.concatenate([times, [times[0] + length]]))
    # The age grid's spacing a whole fraction of the shortest gap, so that
    # a gap of a whole number of them lands a kept unit on the grid.
    shortest = float(gaps.min())
    spacing = shortest / math.ceil(shortest / (oldest / (AGES - 1)))
    points = int(math.ceil(oldest / spacing)) + 1
    if points > MAX_AGES:
        raise NotImplementedError(
            "method='integration' needs an age grid of at most "
            f"{MAX_AGES} points; gaps of {shortest:g} over ages to "
            f"{oldest:g} need {points}: use method='dp'."
        )
    grid = np.arange(points) * spacing
    shapes: Dict[float, _Gap] = {}

    def gap(width):
        key = round(width, 9)
        if key not in shapes:
            shapes[key] = _Gap(width, sf, grid, failure)
        return shapes[key]

    def sweep(h, keep_values=False):
        after = h
        kept: list = [None] * times.size
        for k in range(times.size - 1, -1, -1):
            shape = gap(gaps[k])
            g_nodes = shape.new_unit(after, grid)
            w = shape.kept(g_nodes, after, grid)
            if keep_values:
                kept[k] = (shape, g_nodes, after, w)
            after = np.minimum(w, prices[k] + w[0])
        return after, kept

    h = np.zeros(grid.size)
    limit = max(1, int(MAX_LIVES * oldest / length))
    low = high = math.nan
    converged = False
    sweeps = 0
    for sweeps in range(1, limit + 1):
        v, _ = sweep(h)
        change = v - h
        low, high = float(change.min()), float(change.max())
        h = v - v[0]
        if high - low <= TOLERANCE * max(abs(high), abs(low), 1e-300):
            converged = True
            break
    _, kept = sweep(h, keep_values=True)
    thresholds = np.empty(times.size)
    upper = True
    for k, (shape, g_nodes, after, w) in enumerate(kept):
        replace = prices[k] + w[0]
        act = replace < w
        if not act.any():
            thresholds[k] = math.inf
            continue
        i = int(np.argmax(act))
        if not act[i:].all():
            upper = False
        if i == 0:
            thresholds[k] = 0.0
            continue
        lo, hi = grid[i - 1], grid[i]
        for _ in range(HALVINGS):
            middle = 0.5 * (lo + hi)
            value = shape.kept(g_nodes, after, grid, np.array([middle]))[0]
            if replace < value:
                hi = middle
            else:
                lo = middle
        thresholds[k] = hi
    replaced, failures = _stationary(kept, prices, grid, length, oldest)
    rates = {name: 0.0 for name in names if name != ANYTIME}
    spent = dict(rates)
    for k in range(times.size):
        rates[names[kinds[k]]] += float(replaced[k] / length)
        spent[names[kinds[k]]] += float(replaced[k] * prices[k] / length)
    rates["failure"] = float(failures / length)
    spent["failure"] = float(failures * failure / length)
    policy = CalendarPolicy(
        cost_rate=0.5 * (low + high) / length,
        cost_rate_se=None,
        replacement_rates=rates,
        cost_rates=spent,
        method="integration",
        basis="exact",
        step=None,
        period=length,
        determined_until=None,
        threshold_only=upper,
        settings={
            "sweeps": sweeps,
            "converged": converged,
            "cost_rate_bounds": (low / length, high / length),
            "nodes": NODES,
            "ages": int(grid.size),
            "max_age": oldest,
            "opportunities": int(times.size),
            "tail": tail if repeat is not None else None,
            "repeat": repeat,
        },
        life=life,
    )
    origin = 0.0
    if repeat is not None:
        origin = min(
            d
            for o in calendar.opportunities
            if not o.periodic
            for d in o.dates
        )
    table = _OpportunityTable(
        length, times, thresholds, kinds, names, origin, prices, replaced
    )
    return policy._bind(table, names, None, oldest)


def _stationary(kept, prices, grid, length, oldest):
    """The long-run chance of a replacement at each opportunity of the
    cycle, and the expected failures over it: forward across the cycle
    (each opportunity replacing the ages its last sweep did, then its
    gap) from a new unit, until the ages at its start settle."""
    share = np.zeros(grid.size)
    share[0] = 1.0

    def across(share):
        replaced = np.zeros(len(kept))
        failures = 0.0
        for k, (shape, _, _, w) in enumerate(kept):
            act = prices[k] + w[0] < w
            moved = float(share[act].sum())
            share = np.where(act, 0.0, share)
            share[0] += moved
            replaced[k] = moved
            share, failed = shape.forward(share)
            failures += failed
        return share, replaced, failures

    limit = max(1, int(MAX_LIVES * oldest / length))
    for _ in range(limit):
        later, _, _ = across(share)
        settled = float(np.abs(later - share).sum()) <= TOLERANCE
        share = later
        if settled:
            break
    _, replaced, failures = across(share)
    return replaced, failures
