"""The cash-flow table: amounts priced at their times (events) or over
spans, discounted continuously, by period, category and component, for
one history or many."""

from typing import Any, Dict, List, Tuple

import numpy as np

from repyability.value._categories import CATEGORIES
from repyability.value._rates import discounted_span

ORDER = {
    name: i
    for i, name in enumerate(("acquisition",) + CATEGORIES + ("value",))
}


class Ledger:
    """Amounts by (period, category, component), present and
    undiscounted, as costs (positive) or value, for each of ``size``
    histories."""

    def __init__(self, ends: np.ndarray, rate: float, size: int = 1):
        self.ends = np.asarray(ends, float)
        self.starts = np.concatenate(([0.0], self.ends[:-1]))
        self.rate = rate
        self.size = size
        self.cells: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}

    def _add(self, k, category, component, history, present, nominal):
        cell = self.cells.setdefault(
            (int(k), category, component),
            (np.zeros(self.size), np.zeros(self.size)),
        )
        cell[0][:] += np.bincount(history, present, minlength=self.size)
        cell[1][:] += np.bincount(history, nominal, minlength=self.size)

    def events(self, history, times, price: float, category, component):
        """``price`` charged at each of ``times``, in its history."""
        times = np.asarray(times, float)
        if not price or not times.size:
            return
        history = np.asarray(history, np.int64)
        periods = np.searchsorted(self.ends, times, side="right")
        periods = np.minimum(periods, len(self.ends) - 1)
        present = price * np.exp(-self.rate * times)
        nominal = np.full(times.size, float(price))
        for k in np.unique(periods):
            mask = periods == k
            self._add(
                k,
                category,
                component,
                history[mask],
                present[mask],
                nominal[mask],
            )

    def spans(
        self, history, starts, stops, per_time: float, category, component
    ):
        """``per_time`` over each span from ``starts`` to ``stops``, in its
        history."""
        starts = np.asarray(starts, float)
        if not per_time or not starts.size:
            return
        history = np.asarray(history, np.int64)
        stops = np.asarray(stops, float)
        for k, (lo, hi) in enumerate(zip(self.starts, self.ends)):
            a = np.maximum(starts, lo)
            b = np.minimum(stops, hi)
            inside = b > a
            if not np.any(inside):
                continue
            a, b = a[inside], b[inside]
            self._add(
                k,
                category,
                component,
                history[inside],
                per_time * discounted_span(a, b, self.rate),
                per_time * (b - a),
            )

    def totals(self) -> Dict[tuple, Tuple[np.ndarray, np.ndarray]]:
        """``{(category, component): (present, undiscounted)}`` over every
        period, an amount per history, as costs (positive) or value."""
        out: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        for (_, category, component), (present, nominal) in self.cells.items():
            cell = out.setdefault(
                (category, component),
                (np.zeros(self.size), np.zeros(self.size)),
            )
            cell[0][:] += present
            cell[1][:] += nominal
        return out

    def rows(self, i: int = 0) -> List[Dict[str, Any]]:
        """History ``i``'s cash flows: a row per period, category and
        component, signed (costs negative)."""
        out = []
        for (k, category, component), (present, nominal) in self.cells.items():
            sign = 1.0 if category == "value" else -1.0
            out.append(
                row(
                    k + 1,
                    self.starts[k],
                    self.ends[k],
                    category,
                    component,
                    sign * present[i],
                    sign * nominal[i],
                )
            )
        return sort_rows(out)


def row(period, start, end, category, component, present, nominal) -> dict:
    return {
        "period": int(period),
        "start": float(start),
        "end": float(end),
        "category": category,
        "component": component,
        "present_value": float(present),
        "undiscounted": float(nominal),
    }


def sort_rows(rows: List[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda r: (
            r["period"],
            ORDER.get(r["category"], len(ORDER)),
            str(r["component"]),
        ),
    )
