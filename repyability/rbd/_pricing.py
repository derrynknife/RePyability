"""Up/down histories priced with a diagram's own costs: each failure its
repair and replace costs, each planned outage its preventive cost, each
component's and the system's time down their downtime costs and the
system's time up its value, each discounted at the time it fell (a rate
over a span exactly), by period, category and component, and history by
history.

``plan_vs_actual`` prices one history (an outage log, ``_plan_actual``)
and ``npv_distribution`` every simulated one (``_npv_runs``); what a
history does not determine (a cost charged at an event it does not
record) is each caller's to leave out or refuse.
"""

from typing import Any, Dict, Hashable, List, Mapping, Optional, Tuple

import numpy as np

from repyability.rbd._spec import _mean_cost
from repyability.rbd._tally import _CATEGORIES

ORDER = {
    name: i
    for i, name in enumerate(("acquisition",) + _CATEGORIES + ("value",))
}


#: Why a cost the log does not determine is left out of the comparison.
_UNITS = (
    "A standby group's log is the group's: its units' failures, which "
    "are charged, are not in it."
)
_HIDDEN = (
    "A hidden failure is charged when a test finds it, which the log "
    "does not record."
)
_TESTS = "Tests that find nothing leave no outage in the log."
_INSPECTIONS = "Inspections that replace nothing leave no outage in the log."
_INSTANT = (
    "Its planned replacements take no time, so they leave no outage in "
    "the log."
)
_CALENDAR = (
    "Its calendar policy prices each planned replacement at its "
    "opportunity, which the log does not record."
)
_REPLACED = (
    "It is replaced, rather than repaired, at failures counted from its "
    "last renewal, which the log does not mark."
)
_SETUP = (
    "A set-up cost is charged at each stop of a maintenance group, which "
    "includes renewals the log does not mark."
)
_BOUGHT = "Paid at the start: the log does not record purchases."


def unrecorded(rbd) -> List[Tuple[str, Optional[Hashable], str]]:
    """The priced cells, ``(category, component, reason)``, that an
    up/down history does not determine (see ``_plan_actual``'s notes), by
    the diagram's kinds of component alone."""
    out: List[Tuple[str, Optional[Hashable], str]] = []
    for node, costs in rbd.costs.items():
        failure = [k for k in ("repair_cost", "replace_cost") if k in costs]
        if node in rbd._standby:
            out += [(k[:-5], node, _UNITS) for k in failure]
        elif node in rbd._inspection:
            out += [(k[:-5], node, _HIDDEN) for k in failure]
        else:
            imperfect = rbd._imperfect.get(node)
            if (
                imperfect is not None
                and imperfect.replace_after is not None
                and "replace_cost" in costs
            ):
                out.append(("replace", node, _REPLACED))
        if node in rbd._inspection and "inspection_cost" in costs:
            out.append(("inspection", node, _TESTS))
        schedule = rbd._preventive.get(node)
        if schedule is not None:
            if schedule.policy == "calendar":
                out.append(("preventive", node, _CALENDAR))
            elif schedule.duration is None and "preventive_cost" in costs:
                out.append(("preventive", node, _INSTANT))
            if "inspection_cost" in costs:
                out.append(("inspection", node, _INSPECTIONS))
    if any(spec.setup_cost for spec in rbd._maintenance.values()):
        out.append(("setup", None, _SETUP))
    for node, price in rbd.acquisition_costs.items():
        if price:
            out.append(("acquisition", node, _BOUGHT))
    return out


def discounted_span(a, b, rate: float):
    """The integral of ``exp(-rate t)`` from ``a`` to ``b``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not rate:
        return b - a
    return np.exp(-rate * a) * -np.expm1(-rate * (b - a)) / rate


def changes(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each change of the histories ``data`` (a ``timelines._Data``): its
    history, its time, whether it is a change down, and whether planned."""
    history, j = data.positions
    up_after = (data.start[history].astype(np.int64) ^ ((j + 1) & 1)) == 1
    return history, data.times, ~up_after, data.planned


def spans(data, up: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The spans up (or down) of the histories ``data``: each span's
    history, start and stop."""
    history, j = data.positions
    times = data.times
    previous = np.zeros(times.size)
    later = j > 0
    previous[later] = times[np.flatnonzero(later) - 1]
    up_before = (data.start[history].astype(np.int64) ^ (j & 1)) == 1
    counts = np.diff(data.offsets)
    last = np.zeros(data.size)
    some = counts > 0
    last[some] = times[data.offsets[1:][some] - 1]
    final_up = (data.start.astype(np.int64) ^ (counts & 1)) == 1
    chosen = up_before == up
    final = final_up == up
    return (
        np.concatenate((history[chosen], np.flatnonzero(final))),
        np.concatenate((previous[chosen], last[final])),
        np.concatenate((times[chosen], np.full(int(final.sum()), data.end))),
    )


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


def priced(
    rbd,
    logs: Mapping[Any, Any],
    system: Optional[Any],
    ends: np.ndarray,
    rate: float,
    value_rate: float,
    skip,
    size: int = 1,
) -> Ledger:
    """The cash flows the histories imply over ``[0, ends[-1]]``, by
    period, category and component, history by history, priced with the
    diagram's costs (a cost drawn from a distribution at its mean) at the
    times they fell, but for the cells in ``skip``, ``{(category,
    component)}``. ``logs`` are each component's histories and ``system``
    the system's (``timelines._Data``); ``system`` is needed only for a
    system downtime cost or a value."""
    ledger = Ledger(ends, rate, size)
    for node, costs in rbd.costs.items():
        data = logs[node]
        history, times, down, planned = changes(data)
        failures = down & ~planned
        outages = down & planned
        charges = [("repair_cost", "repair", failures)]
        if node not in rbd._imperfect:
            charges.append(("replace_cost", "replace", failures))
        charges.append(("preventive_cost", "preventive", outages))
        for key, category, mask in charges:
            if key in costs and (category, node) not in skip:
                ledger.events(
                    history[mask],
                    times[mask],
                    _mean_cost(costs[key]),
                    category,
                    node,
                )
        rate_down = costs.get("downtime_cost", 0.0)
        if rate_down:
            ledger.spans(
                *spans(data, up=False), rate_down, "component_downtime", node
            )
    if rbd.downtime_cost_rate or value_rate:
        ledger.spans(
            *spans(system, up=False),
            rbd.downtime_cost_rate,
            "system_downtime",
            None,
        )
        ledger.spans(*spans(system, up=True), value_rate, "value", None)
    return ledger
