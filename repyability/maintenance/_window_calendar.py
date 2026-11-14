"""A part's cost over a window from its state now under a calendar policy,
with its spread (UC-8 and UC-9 of this package's ``README.md``).

The policy is a [`CalendarPolicy`][repyability.CalendarPolicy]: at each
opportunity of its calendar it replaces the part if the part is at least
that time's threshold age. So a unit put in new at time ``s`` is next
replaced as planned at a time ``pi(s)`` the policy fixes, at that
opportunity's price, unless it fails first. Unlike age replacement, the
cycles differ by when they start, so the counts are worked out by
following the renewals in time rather than by convolving one cycle's
distribution.

**On a grid.** Time is laid on a grid of step ``h``, the policy's own
step or a whole multiple of it, so the planned replacements fall on it. A
failure in a cell ``(t_(c-1), t_c]`` renews the part half at each end of
the cell (the trapezoidal rule in the measure, as ``_window_cost``'s
convolutions), and counts in the window by the cell's share in it. A
renewal at a grid time stands for those in the two cells beside it: half
takes the planned time of a unit put in in the middle of the cell
before, half of one in the cell after, so a threshold that falls on a
grid time splits its renewals rather than giving them all to one side.

**Before the window** nothing is counted: the renewals at each grid time
before the window's start (the expected number there, the renewal
measure) are followed forward in time, each passing on its failures (the
half back to its own grid time by a geometric sum) and its planned
replacement. What they pass into the window starts the counts.

**In the window** the expected renewals at each grid time are kept by
the counts so far: the failures ``i`` and the planned replacements by
kind ``J``. The renewals of ``(i, J)`` pass a planned replacement to
``(i, J + 1 of its kind)`` and their failures to ``(i + 1, J)``, so each
count's renewals are known once those with fewer events are: the counts
are taken in that order, and none needs a sweep in time. A count's
failures, by cell, are a convolution of its renewals with a new unit's
chance of failing in each cell, cut at each renewal's planned
replacement: ``pi(s)`` keeps one value over runs of ``s``, so one
convolution (FFT) for each value. What a count does not pass on in the
window (an event at or after its end) is its final chance. Counts whose
renewals add up to less than ``_TAIL`` are dropped, into the precision.

**Precision.** Worked out again on a grid twice as coarse: the largest
change of any probability, plus the mass left out, is ``precision``, as
for the other policies.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve

#: Counts whose renewals add up to less than this are dropped (into the
#: precision).
_TAIL = 1e-13

Counts = Tuple[int, Tuple[int, ...]]
NONE: Counts = (0, ())


def check_policy(part, policy) -> None:
    """The policy, checked: a ``CalendarPolicy`` worked out for this
    part's life."""
    from repyability.maintenance._replacement import (
        CalendarPolicy,
        _saved_life,
    )

    if not isinstance(policy, CalendarPolicy):
        raise TypeError(
            "calendar must be the CalendarPolicy of "
            f"NonRepairable.optimal_calendar_policy; got {policy!r}."
        )
    if policy.life is not None:
        saved = _saved_life(part)
        if saved is not None and saved != policy.life:
            raise ValueError(
                "The calendar policy was worked out for another life, not "
                "the life of this part: work it out for this part, "
                "part.optimal_calendar_policy(calendar)."
            )


def grid_step(policy) -> float:
    """The numerical route's grid step: the policy's own."""
    if policy.step is None:
        raise NotImplementedError(
            "A calendar policy worked out by integration has no grid for "
            "the numerical route: work it out with method='dp', or use "
            "method='simulate'."
        )
    return float(policy.step)


class Kinds:
    """The kinds of planned replacement met, in order, with their prices."""

    def __init__(self):
        self.names: List[str] = []
        self.prices: List[float] = []

    def kind(self, name: str, price: float) -> int:
        if name not in self.names:
            self.names.append(name)
            self.prices.append(price)
        return self.names.index(name)

    def full(self, J: Tuple[int, ...]) -> Tuple[int, ...]:
        """``J`` with a place for every kind."""
        return J + (0,) * (len(self.names) - len(J))


def _plus(J: Tuple[int, ...], k: int) -> Tuple[int, ...]:
    """The counts by kind ``J`` with one more of kind ``k``."""
    out = list(J) + [0] * max(k + 1 - len(J), 0)
    out[k] += 1
    return tuple(out)


class _Planned:
    """When the renewals at each grid time are next replaced as planned:
    ``pi`` of a unit put in at the middle of the cell before and of the
    cell after, each with half the weight (one entry where they agree).
    Kept by grid time, for every window on one grid; ``known`` keeps
    ``pi`` by the time a unit is put in, for every grid."""

    def __init__(self, policy, h: float, kinds: Kinds, known: dict):
        self.policy, self.h, self.kinds = policy, h, kinds
        self.known = known
        self.done = 0
        self.source: List[int] = []
        self.weight: List[float] = []
        self.planned: List[int] = []  # a grid time, or -1 for none
        self.kind: List[int] = []

    def _one(self, put_in: float) -> Tuple[int, int]:
        if put_in not in self.known:
            time, price, name = self.policy._next_planned(put_in, put_in)
            if name is None or not math.isfinite(time):
                self.known[put_in] = (math.inf, -1)
            else:
                k = self.kinds.kind(name, float(price))
                self.known[put_in] = (time, k)
        time, k = self.known[put_in]
        if k < 0:
            return -1, -1
        return int(round(time / self.h)), k

    def entries(self, n: int):
        """``(source, weight, planned, kind)`` arrays for the grid times 0
        to ``n``."""
        for s in range(self.done, n + 1):
            before = self._one((s - 0.5) * self.h)
            after = self._one((s + 0.5) * self.h)
            halves = [before] if before == after else [before, after]
            for at, k in halves:
                self.source.append(s)
                self.weight.append(1.0 / len(halves))
                self.planned.append(at)
                self.kind.append(k)
        self.done = max(self.done, n + 1)
        source = np.array(self.source, dtype=np.int64)
        keep = source <= n
        return (
            source[keep],
            np.array(self.weight)[keep],
            np.array(self.planned, dtype=np.int64)[keep],
            np.array(self.kind, dtype=np.int64)[keep],
        )


def _shares(h: float, n: int, edge: float) -> np.ndarray:
    """Each cell ``(t_(c-1), t_c]``'s share before ``edge``, for ``c = 0
    .. n`` (cell 0 is empty)."""
    c = np.arange(n + 1)
    return np.clip((edge - (c - 1) * h) / h, 0.0, 1.0)


class _Window:
    """The counts in ``[start, end)`` on a grid of step ``h``."""

    def __init__(self, life, planned: _Planned, start: float, end: float):
        h = planned.h
        self.life, self.h, self.kinds = life, h, planned.kinds
        self.start, self.end = start, end
        self.tol = 1e-9 * h
        # Grid times 0 .. n; cells 1 .. n + 1 (the last wholly after end).
        self.n = n = int(math.ceil(end / h - 1e-9)) + 1
        self.R = life.R(h * np.arange(n + 2))  # a new unit's, by cell
        self.d = np.concatenate([[0.0], self.R[:-1] - self.R[1:]])
        self.before_start = _shares(h, n + 1, start)
        self.before_end = _shares(h, n + 1, end)
        self.source, self.weight, self.at, self.kind = planned.entries(n)
        reach = np.maximum(self.at - self.source, 0)
        self.chance = np.where(self.at >= 0, self._R(reach), 0.0)
        # The grid times before the window's start: nothing counted yet.
        self.uncounted = int(math.ceil(start / h - 1e-9))
        self.deposits: Dict[Counts, np.ndarray] = {}
        self.final: Dict[Counts, float] = {}
        self.dropped = 0.0

    def _R(self, m: np.ndarray) -> np.ndarray:
        """A new unit's survival to ``m`` cells."""
        m = np.asarray(m, dtype=np.int64)
        out = np.empty(m.shape)
        near = m < len(self.R)
        out[near] = self.R[m[near]]
        if (~near).any():
            out[~near] = self.life.R(self.h * m[~near])
        return out

    def _deposit(self, key: Counts, at, amount) -> None:
        if key not in self.deposits:
            self.deposits[key] = np.zeros(self.n + 1)
        np.add.at(self.deposits[key], at, amount)

    def _end(self, key: Counts, amount: float) -> None:
        self.final[key] = self.final.get(key, 0.0) + amount

    def _planned_at(self, at: int, k: int, amount: float) -> None:
        """An uncounted renewal's planned replacement at grid time ``at``:
        before the window, in it, or after it."""
        t = at * self.h
        if t >= self.end - self.tol:
            self._end(NONE, amount)
        elif t < self.start - self.tol:
            self._deposit(NONE, at, amount)
        else:
            self._deposit((0, (0,) * k + (1,)), at, amount)

    # -- before the window: nothing counted, followed in time ------------

    def _push(self, amount, s: int, surv, cap: int, reach, back: bool):
        """The uncounted renewals ``amount`` at ``s``, whose cycle survives
        by cell as ``surv``, pass on their failures up to a planned
        replacement ``cap`` cells on (-1: none), which they reach with
        the chance ``reach``; the half cell back to ``s`` only with
        ``back``. Failures past the window are final."""
        n = self.n
        last = n + 1 - s if cap < 0 else min(cap, n + 1 - s)
        if last < 1:
            return
        d = surv[:last] - surv[1 : last + 1]
        cells = slice(s + 1, s + last + 1)
        pre = self.before_start[cells]
        counted = self.before_end[cells] - pre
        top = min(last, n - s)  # the cells whose right end is on the grid
        for share, key in ((pre, NONE), (counted, (1, ()))):
            half = 0.5 * amount * d * share
            if not half.any():
                continue
            if key not in self.deposits:
                self.deposits[key] = np.zeros(n + 1)
            into = self.deposits[key]
            into[s + 1 : s + top + 1] += half[:top]
            if back or key != NONE:
                into[s] += half[0]
            into[s + 1 : s + last] += half[1:last]
        after = float((d * (1.0 - self.before_end[cells])).sum())
        # And the failures past the grid's last cell, before the cap.
        after += float(surv[last]) - reach
        self._end(NONE, amount * after)

    def _before(self, first) -> None:
        """The part in now, then the renewals before the window."""
        surv1, cap, k, chance = first
        if cap != 0:
            self._push(1.0, 0, surv1, cap, chance, back=True)
        if cap >= 0:
            self._planned_at(cap, k, chance)
        a = 0.5 * self.d[1]
        for s in range(self.uncounted):
            m = self.deposits.get(NONE, np.zeros(1 + s))[s]
            if m == 0.0:
                continue
            self.deposits[NONE][s] = 0.0
            cell = s + 1
            x = m / (1.0 - a * self.before_start[cell])
            inside = self.before_end[cell] - self.before_start[cell]
            if inside:
                self._deposit((1, ()), s, a * inside * x)
            self._end(NONE, a * (1.0 - self.before_end[cell]) * x)
            for e in np.flatnonzero(self.source == s):
                share = x * self.weight[e]
                at = int(self.at[e])
                cap = at - s if at >= 0 else -1
                self._push(share, s, self.R, cap, self.chance[e], False)
                if at >= 0:
                    amount = share * self.chance[e]
                    self._planned_at(at, int(self.kind[e]), amount)

    # -- in the window: by counts ---------------------------------------

    def _count(self, key: Counts) -> None:
        """Pass on the renewals of ``key``: its planned replacements, its
        failures, and what it does not pass on in the window."""
        m = self.deposits.pop(key)
        total = float(m.sum())
        if total < _TAIL:
            self.dropped += total
            return
        mass = m[self.source] * self.weight
        counted = 0.0
        planned = self.at >= 0
        for k in np.unique(self.kind[planned]):
            these = planned & (self.kind == k)
            amount = np.bincount(
                self.at[these], weights=mass[these] * self.chance[these]
            )
            inside = np.flatnonzero(amount)
            inside = inside[inside * self.h < self.end - self.tol]
            if len(inside):
                target = (key[0], _plus(key[1], int(k)))
                self._deposit(target, inside, amount[inside])
                counted += float(amount[inside].sum())
        # Failures by cell: one convolution for each planned time.
        cells = np.zeros(self.n + 2)
        for at in np.unique(self.at):
            these = (self.at == at) & (mass > 0.0)
            if not these.any():
                continue
            low = int(self.source[these].min())
            high = self.n + 1 if at < 0 else min(int(at), self.n + 1)
            if high <= low:
                continue
            sources = np.zeros(high - low)
            np.add.at(sources, self.source[these] - low, mass[these])
            part = fftconvolve(sources, self.d[: high - low + 1])
            cells[low : high + 1] += np.maximum(part[: high + 1 - low], 0.0)
        failed = cells * self.before_end
        counted += float(failed.sum())
        half = 0.5 * failed
        self._deposit(
            (key[0] + 1, key[1]), np.arange(self.n + 1), half[1:] + half[:-1]
        )
        self._end(key, max(total - counted, 0.0))

    def run(self, first) -> None:
        self._before(first)
        while self.deposits:
            key = min(self.deposits, key=lambda c: (c[0], sum(c[1]), c[1]))
            self._count(key)


def _first(life, policy, h: float, kinds: Kinds, n: int):
    """The part in now: its survival by cell from 0, the cells to its
    planned replacement (-1: none), its kind and its chance of reaching
    it."""
    time, price, name = policy._next_planned(-life.age, 0.0)
    surv = life.S(h * np.arange(n + 2))
    if name is None or not math.isfinite(time):
        return surv, -1, -1, 0.0
    cap = int(round(time / h))
    return surv, cap, kinds.kind(name, float(price)), float(life.S(time))


def window_counts(life, policy, start: float, end: float, h: float, cache):
    """The chance of each count in ``[start, end)``, on a grid of step
    ``h`` (``cache`` keeps the planned times between windows): ``(P,
    by_vector, dropped)``, ``P[i, j]`` the chance of ``i`` failures and
    ``j`` planned replacements, ``by_vector`` each count of failures'
    chance for each vector of planned replacements by kind (in the order
    of ``cache["kinds"].names``), and the chance dropped."""
    kinds = cache.setdefault("kinds", Kinds())
    planned = cache.get(h)
    if planned is None:
        known = cache.setdefault("known", {})
        planned = cache[h] = _Planned(policy, h, kinds, known)
    window = _Window(life, planned, start, end)
    window.run(_first(life, policy, h, kinds, window.n))
    final = window.final
    rows = max(i for i, _ in final) + 1
    vectors = sorted({kinds.full(J) for _, J in final})
    by_vector = {J: np.zeros(rows) for J in vectors}
    for (i, J), p in final.items():
        by_vector[kinds.full(J)][i] += p
    P = np.zeros((rows, max(sum(J) for J in vectors) + 1))
    for J, column in by_vector.items():
        P[:, sum(J)] += column
    return P, by_vector, window.dropped
