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
"""

import math
from typing import NamedTuple

import numpy as np

from ._block_replacement import (
    _MAX_INTERVALS,
    _TAIL,
    _UNSUPPORTED,
    _carry,
    _cumulative,
    _grid,
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
            up += float(
                weight
                @ (
                    ages.up[lo + steps : lo + n + steps]  # noqa: E203
                    - ages.up[lo : lo + n]  # noqa: E203
                )
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
