"""Exact sums (#151): a simulation run's totals, the same however the run is
split.

Floating-point addition depends on order and grouping, so a total added up
simulation by simulation comes out differently from the same simulations
added in pieces (a run cut into chunks on several machines, see
``chunks``) in its last bits. An ``ExactSum`` keeps its sum *exactly*
instead: as floats whose exact (unrounded) sum is the sum of every value
added, and rounds it once, correctly (``math.fsum``), when it is read. Sums
put together in any order and any grouping then give the same total, to the
last bit, and a slightly more accurate one than adding in order.

The floats it keeps (an expansion) come from repeated correctly rounded
sums: the first is the sum rounded, the next what is left over rounded, and
so on until nothing is left, each about 2**53 smaller than the last, so
there are rarely more than two or three. Single values wait in a list and
are folded in a block at a time. An array is summed exactly at once, a
column at a time, by error-free extraction (Rump, Ogita and Oishi's
AccSum): with ``sigma`` a power of two at least ``n + 2`` times the largest
value, ``q = (sigma + p) - sigma`` rounds each value to a multiple of
``sigma``'s last bit exactly, the ``q`` add up exactly in any order, and
``p - q`` is exact too, and about ``2**53 / n`` times smaller: a few passes
leave nothing over.
"""

import math
from typing import Iterable, List, Sequence

import numpy as np

#: Values waiting to be folded in when an ``ExactSum`` folds them.
_FOLD = 4096
#: Passes of the extraction before what is left is summed by ``fsum``.
_PASSES = 64


def expansion(values: Iterable[float]) -> List[float]:
    """Floats whose exact sum is the exact sum of ``values``, largest first
    (none for a sum of 0). Infinite or undefined sums are kept as their
    (rounded) value."""
    values = [float(v) for v in values]
    out: List[float] = []
    while values:
        try:
            total = math.fsum(values)
        except (OverflowError, ValueError):  # inf - inf, or past the range
            with np.errstate(invalid="ignore", over="ignore"):
                return [float(np.sum(values))]
        if total == 0.0:
            break
        if not math.isfinite(total):
            return [total]
        out.append(total)
        values.append(-total)
    return out


def column_parts(table: np.ndarray) -> List[np.ndarray]:
    """For each column of ``table`` (one row per value), floats whose exact
    sum is the column's exact sum: one array per pass of the extraction
    (see the module docstring), each the exact sum of what that pass took
    from every column. A column that is not finite is summed as it is."""
    p = np.array(table, dtype=float, copy=True)
    if p.ndim == 1:
        p = p[:, None]
    rows = p.shape[0]
    if rows == 0:
        return []
    # sigma >= (rows + 2) * the column's largest value, a power of two.
    lift = math.ceil(math.log2(rows + 2))
    out: List[np.ndarray] = []
    with np.errstate(invalid="ignore", over="ignore"):
        finite = np.all(np.isfinite(p), axis=0)
        if not finite.all():
            out.append(np.where(finite, 0.0, p.sum(axis=0)))
            p[:, ~finite] = 0.0
    for _ in range(_PASSES):
        top = np.max(np.abs(p), axis=0)
        if not top.any():
            return out
        _, exponent = np.frexp(top)
        sigma = np.where(top > 0.0, np.ldexp(1.0, exponent + lift), 0.0)
        if not np.all(np.isfinite(sigma)):  # past 2**1023 / rows: rare
            break
        q = (sigma + p) - sigma
        out.append(q.sum(axis=0))
        p = p - q
    # What is left (none in practice): summed exactly, column by column.
    out.append(np.array([math.fsum(column) for column in p.T]))
    return out


class ExactSum:
    """A sum of floats kept exactly (see the module docstring): add floats,
    arrays of them or other ``ExactSum`` s with ``+=`` (or ``add``), and read
    the correctly rounded total with ``float``. ``partials`` are the floats
    it keeps, whose exact sum is the total, to save and put back."""

    __slots__ = ("_partials", "_pending")

    def __init__(self, values=()):
        self._partials: List[float] = []
        self._pending: List[float] = []
        self.add(values)

    def add(self, values) -> "ExactSum":
        """Add a float, an array or list of floats, or another
        ``ExactSum``."""
        if isinstance(values, float):
            self._pending.append(values)
        elif isinstance(values, ExactSum):
            self._pending.extend(values.partials)
        elif isinstance(values, (int, np.floating, np.integer)):
            self._pending.append(float(values))
        else:
            values = np.asarray(values, dtype=float).ravel()
            if values.size > 8:
                self._pending.extend(
                    float(part[0]) for part in column_parts(values)
                )
            else:
                self._pending.extend(values.tolist())
        if len(self._pending) > _FOLD:
            self._fold()
        return self

    def __iadd__(self, values) -> "ExactSum":
        return self.add(values)

    def _fold(self) -> None:
        if self._pending:
            self._partials = expansion(self._partials + self._pending)
            self._pending = []

    @property
    def partials(self) -> List[float]:
        """The floats kept, whose exact sum is the total."""
        self._fold()
        return list(self._partials)

    def __float__(self) -> float:
        self._fold()
        return math.fsum(self._partials) if self._partials else 0.0

    def __repr__(self) -> str:
        return f"ExactSum({float(self)!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, ExactSum):
            return self.partials == other.partials
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


def add_columns(totals: Sequence[ExactSum], table: np.ndarray) -> None:
    """Add each column of ``table`` (one row per value) to the total of the
    same position in ``totals``, exactly."""
    parts = column_parts(
        np.asarray(table, dtype=float).reshape(-1, len(totals))
    )
    for c, total in enumerate(totals):
        total._pending.extend(float(part[c]) for part in parts)
        if len(total._pending) > _FOLD:
            total._fold()


def binned_parts(bins: np.ndarray, values: np.ndarray, size: int) -> list:
    """For each of ``size`` bins, floats whose exact sum is the exact sum of
    the ``values`` in it (``bins`` the bin of each): one array of ``size``
    per pass of the extraction (see the module docstring), each the exact
    sum, bin by bin, of what that pass took. ``sigma`` is the same for
    every bin, so a pass's values add up exactly however they fall into
    the bins, with no sort: a bin of much smaller values than another's
    is taken by a later pass. Values that are not finite are summed as
    they are."""
    bins = np.asarray(bins, dtype=np.int64).ravel()
    p = np.array(values, dtype=float, copy=True).ravel()
    out: list = []
    if p.size == 0:
        return out
    lift = math.ceil(math.log2(p.size + 2))
    with np.errstate(invalid="ignore", over="ignore"):
        finite = np.isfinite(p)
        if not finite.all():
            out.append(
                np.bincount(bins[~finite], weights=p[~finite], minlength=size)
            )
            p[~finite] = 0.0
    for _ in range(_PASSES):
        top = float(np.max(np.abs(p)))
        if top == 0.0:
            return out
        _, exponent = math.frexp(top)
        sigma = math.ldexp(1.0, exponent + lift)
        if not math.isfinite(sigma):  # past 2**1023 / values: rare
            break
        q = (sigma + p) - sigma
        out.append(np.bincount(bins, weights=q, minlength=size))
        p = p - q
    # What is left (none in practice): summed exactly, bin by bin.
    left = np.zeros(size)
    for b in np.unique(bins[p != 0.0]).tolist():
        left[b] = math.fsum(p[bins == b])
    out.append(left)
    return out


def compact_parts(parts: list, size: int) -> list:
    """Arrays of ``size`` whose exact sum, bin by bin, is that of
    ``parts``': as few as the extraction needs (see ``column_parts``)."""
    if not parts:
        return []
    return column_parts(np.vstack(parts).reshape(-1, size))


def bin_totals(parts: list, size: int) -> np.ndarray:
    """The exact sum, bin by bin, of ``parts`` (arrays of ``size``),
    rounded once."""
    if not parts:
        return np.zeros(size)
    table = np.vstack(compact_parts(parts, size))
    return np.array([math.fsum(table[:, j]) for j in range(size)])
