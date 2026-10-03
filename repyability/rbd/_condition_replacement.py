"""Exact long-run values of a component replaced on condition (#145).

Replaced on condition, a unit is inspected at every multiple of ``T`` while
it works, and replaced, as new, at an inspection at which it is more likely
than the threshold to fail before the next, given its age ``a``: ``1 - R(a
+ T) / R(a) > threshold``. Its failures are revealed and repaired as usual.
The inspections that replace it are regeneration points, so its long-run
values are its mean up time, failures and so on over a cycle, from one such
replacement to the next, over the cycle's mean length (renewal-reward); and
its availability over an inspection interval, which is what components
inspected at the same times share, is the cycle's over the mean number of
intervals in it.

A cycle is followed one inspection interval at a time, as under block
replacement (see ``_block_replacement``): the units put into service in an
interval are an alternating renewal process of lives and repairs from
their start, and the repairs still going on at its end carry into the
next. What differs is the inspection that ends the interval. Only a unit
old enough is replaced there; a younger one carries on into the next
interval as old as it is. So each interval also starts with the units
carried on, by their ages: one of age ``a`` is up a time ``t`` later with
probability ``R(a + t) / R(a)``, and its failure starts a repair, and a new
unit after it. A unit put into service at ``v`` in the interval is up at
its end, of age ``T - v``, with probability ``R(T - v)``, out of the units
put into service there: by the interval's input and after a repair since
(the renewal density of the units put into service).

Over time (#161), ``condition_availability`` follows the same recursion
from new over all the intervals, rather than one cycle's: each interval
starts with what the inspection at its start replaced, the repairs and
replacements carried into it and the units the inspection kept, by age.
In each, the unit is up at ``s`` if a unit put into service in it is (the
renewal functions, as under block replacement) or a kept unit of age
``a`` is (``R(a + s) / R(a)``), so the curve is a ``BlockCurve`` with the
inspections counted too. After some intervals it repeats from one to the
next, which is its long-run cycle. Started from a state, the unit reaches
its first inspection by its own curve; the unit in service at the start
is followed at its exact age, so that the inspections decide on it at the
age it has, and the units put into service since by their ages on the
grid.
"""

import math
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from repyability.utils.vectors import dot

from ._block_replacement import (
    _MAX_INTERVALS,
    _MAX_VALUES,
    _SETTLED,
    _TAIL,
    _UNSUPPORTED,
    BlockAvailability,
    _carry,
    _cumulative,
    _grid,
    _head_start,
    _masses,
    _values,
)
from ._model_utils import model_mean


class ConditionCycle(NamedTuple):
    """A regeneration cycle under replacement on condition: the fields of a
    ``BlockCycle`` (over an inspection interval, ``before`` and ``after``
    an inspection), and ``replaced``, the mean number of replacements in a
    cycle: 1, or 0 for a unit that no inspection would replace, whose cycle
    is then a life and a repair."""

    up: float
    length: float
    failures: float
    phase: np.ndarray
    availability: np.ndarray
    failure_rate: np.ndarray
    before: float
    after: float
    replaced: float


class _Ages:
    """A life on a grid of ages ``0, h, 2h, ...``, extended as needed: its
    survival and CDF, its integrated survival, and whether an inspection
    replaces a unit of each age."""

    def __init__(self, life, h: float, interval: float, threshold: float):
        self.life, self.h = life, h
        self.interval, self.threshold = interval, threshold
        self.size = 0
        self.R = self.F = self.up = np.empty(0)
        self.replace = np.empty(0, dtype=bool)

    def likely(self, ages: np.ndarray) -> np.ndarray:
        """The chance that a unit of each age fails before the next
        inspection, ``1 - R(a + T) / R(a)``: from the cumulative hazard
        where the model gives one, which keeps a small one precise."""
        later = ages + self.interval
        Hf = getattr(self.life, "Hf", None)
        out = np.full(ages.shape, np.nan)
        if Hf is not None:
            gained = _values(Hf, later) - _values(Hf, ages)
            out = -np.expm1(-np.maximum(gained, 0.0))
        missing = np.isnan(out)
        if missing.any():
            now = np.clip(_values(self.life.sf, ages[missing]), 0.0, 1.0)
            then = np.clip(_values(self.life.sf, later[missing]), 0.0, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                fails = np.where(now > 0.0, 1.0 - then / now, 1.0)
            out[missing] = fails
        return np.clip(np.nan_to_num(out, nan=1.0), 0.0, 1.0)

    def cover(self, last: int) -> None:
        """Extend the grid to cover the ages up to index ``last``."""
        if last < self.size:
            return
        size = max(last + 1, 2 * self.size, 1024)
        x = self.h * np.arange(size)
        self.R = np.clip(_values(self.life.sf, x), 0.0, 1.0)
        self.F = np.clip(_values(self.life.ff, x), 0.0, 1.0)
        self.up = _cumulative(
            lambda v: np.clip(_values(self.life.sf, v), 0.0, 1.0), x
        )
        self.replace = self.likely(x) > self.threshold
        self.size = size


def _never_replaced(ages: _Ages, steps: int) -> bool:
    """Whether no inspection would ever replace a unit: none at an age the
    unit has more than a negligible chance of reaching."""
    last = steps
    while True:
        ages.cover(last)
        reach = ages.R[: last + 1] > 1e-14
        if ages.replace[: last + 1][reach].any():
            return False
        if not reach[-1]:
            return True
        if last > 64 * _MAX_INTERVALS:
            return False
        last *= 2


def _merge(lo_a: int, a: np.ndarray, lo_b: int, b: np.ndarray):
    """Two arrays of masses on the age grid, from ages ``lo_a`` and
    ``lo_b``, added: from the smaller, as one array."""
    if not a.size:
        return lo_b, b
    if not b.size:
        return lo_a, a
    lo = min(lo_a, lo_b)
    hi = max(lo_a + len(a), lo_b + len(b))
    out = np.zeros(hi - lo)
    out[lo_a - lo : lo_a - lo + len(a)] += a  # noqa: E203
    out[lo_b - lo : lo_b - lo + len(b)] += b  # noqa: E203
    return lo, out


def condition_cycle(
    life, repair, duration, interval: float, threshold: float, node=None
) -> ConditionCycle:
    """The regeneration cycle of a component replaced on condition.

    Parameters
    ----------
    life, repair, duration, interval, node
        As for ``block_replacement.block_cycle``: the inspections are every
        ``interval``, and ``duration`` is the time a replacement takes.
    threshold : float
        The chance of failing before the next inspection above which an
        inspection replaces the unit.

    Returns
    -------
    ConditionCycle

    Raises
    ------
    NotImplementedError
        As ``block_cycle`` does, or if the unit is replaced so rarely, for
        its failures and repairs, that a cycle could not be followed to its
        end.
    """
    from scipy.signal import fftconvolve

    g = _grid(life, repair, duration, interval, node, starts=True)
    assert g.renewals is not None and g.done is not None  # starts=True
    steps, h, T = g.steps, g.h, g.interval
    ages = _Ages(life, h, T, threshold)
    if _never_replaced(ages, steps):
        # Then a plain alternating renewal process of lives and repairs.
        mttf = float(model_mean(life))
        mttr = (
            float(g.fix.fixed)
            if g.fix.fixed is not None
            else float(model_mean(repair))
        )
        length = mttf + mttr
        if not (0.0 < length < math.inf):
            raise NotImplementedError(
                f"Component {node!r} is never replaced on condition, and its "
                f"life has no finite mean. {_UNSUPPORTED}"
            )
        available = mttf / length
        return ConditionCycle(
            mttf,
            length,
            1.0,
            g.grid,
            np.full(steps, available),
            np.full(steps, 1.0 / length),
            available,
            available,
            0.0,
        )

    # The units put into service: the one put in, and one after each repair
    # (half of each cell's renewals at each of its ends).
    renewals = np.diff(g.renewals)
    put_in = np.zeros(steps + 1)
    put_in[0] = 1.0
    put_in[:-1] += 0.5 * renewals
    put_in[1:] += 0.5 * renewals
    # A repair's end, from a failure in each cell (integrated over it).
    ending = np.diff(g.done)

    P = g.replace.cdf(g.rho)
    C = g.rho - g.replace.in_progress(g.rho)
    A_middle = 0.5 * (g.A[1:] + g.A[:-1])
    lo, aged = 0, np.empty(0)
    up = cycle = failures = replaced = 0.0
    up_at = np.zeros(steps)
    failing = np.zeros(steps)
    up_at_end = 0.0
    ages.cover(steps)
    # The share of the cycle's units still going from one inspection to the
    # next, over the last few intervals: once the units' state at the
    # inspections keeps its shape, everything after falls by that share an
    # interval, and the rest of the cycle is a geometric series.
    ratios: list = []
    going = 1.0
    for block in range(_MAX_INTERVALS):
        before_sums = (up, failures, up_at.copy(), failing.copy(), up_at_end)
        if aged.size:
            # The units carried on from earlier inspections, by age.
            n = len(aged)
            ages.cover(lo + n + steps)
            R_now = ages.R[lo : lo + n]  # noqa: E203
            weight = np.where(R_now > 0.0, aged / np.maximum(R_now, 1e-300), 0)
            backwards = weight[::-1]
            up_ends = fftconvolve(
                ages.R[lo : lo + n + steps], backwards, mode="valid"
            )
            failed = fftconvolve(
                ages.F[lo : lo + n + steps], backwards, mode="valid"
            )
            fail_cells = np.maximum(np.diff(failed), 0.0)
            up += dot(
                weight,
                ages.up[lo + steps : lo + n + steps]  # noqa: E203
                - ages.up[lo : lo + n],  # noqa: E203
            )
            failures += float(fail_cells.sum())
            up_at += 0.5 * (up_ends[1:] + up_ends[:-1])
            failing += fail_cells / h
            survivors = aged * np.where(
                R_now > 0.0,
                ages.R[lo + steps : lo + n + steps]  # noqa: E203
                / np.maximum(R_now, 1e-300),
                0.0,
            )
            # Their failures' repairs put new units into service.
            started = np.zeros(g.length + 1)
            started[1:] = fftconvolve(fail_cells / h, ending)[: g.length]
            started = np.clip(started, 0.0, None)
            P = P + started
            C = C + np.concatenate(
                [[0.0], np.cumsum(0.5 * h * (started[1:] + started[:-1]))]
            )
        else:
            survivors = np.empty(0)

        # The units put into service in this interval.
        w = _masses(P, C, steps, h)
        up += float(w @ g.U[::-1])
        failures += float(w @ g.N[::-1])
        up_at += fftconvolve(w[:steps], A_middle)[:steps]
        density = np.diff(fftconvolve(w, g.N)[: steps + 1]) / h
        failing += density

        # The units up at the inspection that ends the interval, by age:
        # those put into service in it, as old as the time since...
        fresh = fftconvolve(w, put_in)[: steps + 1] * ages.R[steps::-1]
        total, wanted = float(fresh.sum()), float(w @ g.A[::-1])
        if total > 0.0:
            fresh *= wanted / total
        # ... and those carried on, an interval older.
        lo_up, by_age = _merge(0, fresh[::-1], lo + steps, survivors)
        up_at_end += float(by_age.sum())
        ages.cover(lo_up + len(by_age))
        old = ages.replace[lo_up : lo_up + len(by_age)]  # noqa: E203
        replaced_here = float(by_age[old].sum())
        replaced += replaced_here
        cycle += (block + 1) * T * replaced_here
        kept = np.where(old, 0.0, by_age)
        nonzero = np.flatnonzero(kept > 0.0)
        if nonzero.size:
            lo, aged = lo_up + nonzero[0], kept[nonzero[0] : nonzero[-1] + 1]
        else:
            lo, aged = 0, np.empty(0)

        # Carried into the next interval: what had not started by the
        # inspection, and the repairs still going on at it.
        P, C, _ = _carry(P, C, density, g)
        left = float(P[-1]) + float(aged.sum())
        if left < _TAIL:
            break
        ratios.append(left / going)
        going = left
        if (
            len(ratios) > 20
            and max(ratios[-6:]) - min(ratios[-6:]) < 1e-12
            and ratios[-1] < 1.0
        ):
            # The rest of the cycle: this interval's share, falling by the
            # ratio an interval (the cycle's length weighs each by its end).
            rho = ratios[-1]
            more = rho / (1.0 - rho)
            up += (up - before_sums[0]) * more
            failures += (failures - before_sums[1]) * more
            up_at += (up_at - before_sums[2]) * more
            failing += (failing - before_sums[3]) * more
            up_at_end += (up_at_end - before_sums[4]) * more
            cycle += (
                T
                * replaced_here
                * ((block + 1) * more + rho / (1.0 - rho) ** 2)
            )
            replaced += replaced_here * more
            break
    else:
        raise NotImplementedError(
            f"Component {node!r}: replaced on condition, it fails and is "
            "repaired so many times before an inspection replaces it that "
            f"a cycle could not be followed to its end. {_UNSUPPORTED}"
        )
    # Per interval, in the long run: the sums over a cycle's intervals over
    # the mean number of intervals in a cycle (each cycle ends with one
    # replacement; the sums cover all but a negligible tail of them).
    cycle /= replaced
    up /= replaced
    failures /= replaced
    intervals = cycle / T
    before = min(1.0, up_at_end / replaced / intervals)
    out_for = 1.0 - float(g.replace.cdf(np.zeros(1))[0])
    return ConditionCycle(
        up,
        cycle,
        failures,
        g.grid,
        np.clip(up_at / replaced / intervals, 0.0, 1.0),
        np.maximum(failing / replaced / intervals, 0.0),
        before,
        max(0.0, before - out_for / intervals),
        1.0,
    )


class ConditionHead(NamedTuple):
    """How a unit replaced on condition and started from a state reaches its
    first inspection, ``length`` after the start (see ``BlockHead``): up
    there with probability ``up``; ``failures(u)``, its expected failures
    before each ``u`` up to it; ``down_sf``, the survival function of what
    is left of the repair or replacement it was in at the start (None if it
    was up); and ``first``, if it was up, the probability that the unit in
    service at the start is still, unfailed, at the inspection, and its age
    there."""

    length: float
    up: float
    failures: Callable[[np.ndarray], np.ndarray]
    down_sf: Optional[Callable[[np.ndarray], np.ndarray]] = None
    first: Optional[Tuple[float, float]] = None


def _head_ages(head: ConditionHead, g, ages: _Ages) -> np.ndarray:
    """The units put into service since the start that are up at the first
    inspection, by age on the grid (``0, h, 2h, ...``): one starts after
    each repair (of the failures before the inspection, in cells of the
    grid's step back from it, each repair integrated exactly over its
    cell), and after the repair or replacement going on at the start; it is
    up at the inspection, of age ``a``, if it has not failed since,
    ``R(a)``. Half of each cell's starts go to each of its ends, and the
    total is the probability that such a unit is up there."""
    from scipy.signal import fftconvolve

    h = g.h
    cells = max(1, int(np.ceil(head.length / h - 1e-9)))
    edges = np.maximum(head.length - h * np.arange(cells + 1), 0.0)
    counts = np.asarray(head.failures(edges), dtype=float)
    late = np.maximum(counts[:-1] - counts[1:], 0.0) / h
    ending = np.diff(g.done)[:cells]
    # The starts by each edge back from the inspection: the failures before
    # it whose repairs are over by then.
    started = np.zeros(cells + 1)
    started[:cells] = fftconvolve(late, ending[::-1])[
        cells - 1 : 2 * cells - 1
    ]
    if head.down_sf is not None:
        started += 1.0 - np.asarray(head.down_sf(edges), dtype=float)
    within = np.maximum(started[:-1] - started[1:], 0.0)
    masses = np.zeros(cells + 1)
    masses[:-1] += 0.5 * within
    masses[1:] += 0.5 * within
    ages.cover(cells)
    up = masses * ages.R[: cells + 1]
    wanted = float(head.up) - (head.first[0] if head.first else 0.0)
    total = float(up.sum())
    if wanted <= 0.0 or total <= 0.0:
        return np.zeros(cells + 1)
    return up * (wanted / total)


def condition_availability(
    life,
    repair,
    duration,
    interval: float,
    threshold: float,
    horizon: float,
    node=None,
    head: Optional[ConditionHead] = None,
) -> BlockAvailability:
    """The point availability of a component replaced on condition, new at
    0 (or after ``head``), over ``[0, horizon]`` or until it repeats from
    one inspection interval to the next (see the module docstring).

    Parameters
    ----------
    life, repair, duration, interval, threshold, node
        As for ``condition_cycle``.
    horizon : float
        The last time the availability is needed at (from the first
        inspection, after a ``head``); ``inf`` to follow it until it
        settles (into its long-run cycle).
    head : ConditionHead, optional
        How the unit reaches its first inspection, started from a state;
        by default None: new at 0.

    Returns
    -------
    BlockAvailability
        With ``inspected``, the probability that the unit is up at each
        interval's starting inspection; ``replaced``, that the inspection
        replaces it.

    Raises
    ------
    NotImplementedError
        As ``condition_cycle`` does, or if the availability has not settled
        into repeating within the intervals the grid can keep.
    """
    from scipy.signal import fftconvolve

    g = _grid(life, repair, duration, interval, node, starts=True)
    assert g.renewals is not None and g.done is not None  # starts=True
    steps, h, T = g.steps, g.h, g.interval
    ages = _Ages(life, h, T, threshold)
    ages.cover(steps)
    most = max(3, _MAX_VALUES // (steps + 1))
    count = (
        int(np.floor(horizon / T)) + 1 if np.isfinite(horizon) else most + 1
    )
    renewals = np.diff(g.renewals)
    put_in = np.zeros(steps + 1)
    put_in[0] = 1.0
    put_in[:-1] += 0.5 * renewals
    put_in[1:] += 0.5 * renewals
    ending = np.diff(g.done)
    start_P = g.replace.cdf(g.rho)
    start_C = g.rho - g.replace.in_progress(g.rho)
    down = 1.0 - g.A
    sf = life.sf

    def inspect(lo_up: int, by_age: np.ndarray, first):
        """The inspection: the units up at it by age on the grid (from
        ``lo_up``), and the first unit at its exact age; those old enough
        are replaced. Returns the probability replaced, inspected, the
        units kept by age, and the first unit if kept."""
        ages.cover(lo_up + len(by_age))
        old = ages.replace[lo_up : lo_up + len(by_age)]  # noqa: E203
        inspected = float(by_age.sum())
        replaced = float(by_age[old].sum())
        kept = np.where(old, 0.0, by_age)
        nonzero = np.flatnonzero(kept > 0.0)
        lo, aged = 0, np.empty(0)
        if nonzero.size:
            lo = lo_up + int(nonzero[0])
            aged = kept[nonzero[0] : nonzero[-1] + 1]  # noqa: E203
        if first is not None:
            chance, age = first
            inspected += chance
            if ages.likely(np.array([age]))[0] > threshold:
                replaced += chance
                first = None
        return replaced, inspected, lo, aged, first

    def starts_after(cells: np.ndarray) -> np.ndarray:
        """The CDF, on ``g.rho``, of the units put into service after the
        repairs of the failures in each cell of an interval."""
        started = np.zeros(g.length + 1)
        started[1:] = fftconvolve(cells / h, ending)[: g.length]
        return np.clip(started, 0.0, None)

    if head is None:
        # The first interval: put into service new at 0, no inspection.
        P = np.ones(g.length + 1)
        C = g.rho.copy()
        other = np.zeros(g.length + 1)
        replaced, inspected = 1.0, 0.0
        lo, aged, first = 0, np.empty(0), None
    else:
        other, C, _ = _head_start(head, g)  # type: ignore[arg-type]
        replaced, inspected, lo, aged, first = inspect(
            0, _head_ages(head, g, ages), head.first
        )
        P = other + replaced * start_P
        C = C + replaced * start_C
    rows: list = []
    probabilities: list = []
    failing: list = []
    looked: list = []
    settled = False
    for k in range(count):
        kept_up = np.zeros(steps + 1)
        kept_cells = np.zeros(steps)
        survivors = np.empty(0)
        started = np.zeros(g.length + 1)
        if aged.size:
            # The units the inspection kept, by age.
            n = len(aged)
            ages.cover(lo + n + steps)
            R_now = ages.R[lo : lo + n]  # noqa: E203
            weight = np.where(R_now > 0.0, aged / np.maximum(R_now, 1e-300), 0)
            backwards = weight[::-1]
            kept_up += fftconvolve(
                ages.R[lo : lo + n + steps], backwards, mode="valid"  # noqa
            )
            failed = fftconvolve(
                ages.F[lo : lo + n + steps], backwards, mode="valid"  # noqa
            )
            cells = np.maximum(np.diff(failed), 0.0)
            kept_cells += cells
            started += starts_after(cells)
            survivors = aged * np.where(
                R_now > 0.0,
                ages.R[lo + steps : lo + n + steps]  # noqa: E203
                / np.maximum(R_now, 1e-300),
                0.0,
            )
        after = None
        if first is not None:
            # The unit in service at the start, at its exact age.
            chance, age = first
            R_age = np.clip(_values(sf, age + g.grid), 0.0, 1.0)
            share = chance / max(
                float(_values(sf, np.array([age]))[0]), 1e-300
            )
            kept_up += share * R_age
            cells = share * np.maximum(R_age[:-1] - R_age[1:], 0.0)
            kept_cells += cells
            started += starts_after(cells)
            after = (share * float(R_age[-1]), age + T)
        if started.any():
            P = P + started
            other = other + started
            C = C + np.concatenate(
                [[0.0], np.cumsum(0.5 * h * (started[1:] + started[:-1]))]
            )
        w = _masses(P, C, steps, h)
        row = other[: steps + 1] - fftconvolve(w, down)[: steps + 1] + kept_up
        density = np.diff(fftconvolve(w, g.N)[: steps + 1]) / h
        failures = np.concatenate([[0.0], np.cumsum(density * h + kept_cells)])
        if (
            k >= 2
            and abs(replaced - probabilities[-1]) < _SETTLED
            and abs(inspected - looked[-1]) < _SETTLED
            and np.max(np.abs(row - rows[-1])) < _SETTLED
            and np.max(np.abs(failures - failing[-1])) < _SETTLED
        ):
            settled = True
            break
        if len(rows) == most:
            raise NotImplementedError(
                f"Component {node!r}: replaced on condition, its "
                f"availability has not settled after {most} inspection "
                "intervals, as far as the grid can follow it. "
                f"{_UNSUPPORTED}"
            )
        rows.append(row)
        probabilities.append(replaced)
        failing.append(failures)
        looked.append(inspected)
        # The units up at the inspection that ends the interval, by age:
        # those put into service in it, as old as the time since, and those
        # kept, an interval older.
        fresh = fftconvolve(w, put_in)[: steps + 1] * ages.R[steps::-1]
        total, wanted = float(fresh.sum()), float(w @ g.A[::-1])
        if total > 0.0:
            fresh *= wanted / total
        lo_up, by_age = _merge(0, fresh[::-1], lo + steps, survivors)
        other, C, _ = _carry(P, C, density, g)
        replaced, inspected, lo, aged, first = inspect(lo_up, by_age, after)
        P = other + replaced * start_P
        C = C + replaced * start_C
    return BlockAvailability(
        T,
        g.grid,
        np.array(rows),
        np.array(probabilities),
        g.replace,
        settled,
        np.array(failing),
        head is None,
        np.array(looked),
    )
