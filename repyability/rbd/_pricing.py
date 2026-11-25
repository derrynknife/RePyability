"""Up/down histories priced with a diagram's own costs: each failure its
repair and replace costs, each planned outage its preventive cost, each
component's and the system's time down their downtime costs, the repair
crews' salaries over the whole time and the system's time up its value
(a rate, or each step's of a ``ValueSchedule``), each grown at its
stream's escalation (``price * exp(g t)``) and discounted at the time it
fell (a rate over a span exactly), by period, category and component,
and history by history.

``plan_vs_actual`` prices one history (an outage log, ``_plan_actual``)
and ``npv_distribution`` every simulated one (``_npv_runs``); what a
history does not determine (a cost charged at an event it does not
record) is each caller's to leave out or refuse.
"""

from typing import Any, Hashable, List, Mapping, Optional, Tuple

import numpy as np

from repyability.rbd._common import _crew_salaries
from repyability.rbd._spec import _mean_cost
from repyability.value._flows import (  # noqa: F401
    ORDER,
    Ledger,
    row,
    sort_rows,
)
from repyability.value._rates import discounted_span  # noqa: F401

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
_SALVAGE = (
    "Credited at the plan's end (its retirement or the horizon): the log "
    "does not record a sale."
)


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
    from repyability.rbd import _nested_costs

    for node, price in _nested_costs.acquisitions(rbd).items():
        if price:
            out.append(("acquisition", node, _BOUGHT))
    for node, own in getattr(rbd, "_valuation", {}).items():
        if "salvage" in own:
            out.append(("salvage", node, _SALVAGE))
    return out


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


def priced(
    rbd,
    logs: Mapping[Any, Any],
    system: Optional[Any],
    ends: np.ndarray,
    money,
    skip,
    size: int = 1,
    hidden: Optional[Mapping[Any, Any]] = None,
    ledger: Optional[Ledger] = None,
    where: tuple = (),
) -> Ledger:
    """The cash flows the histories imply over ``[0, ends[-1]]``, by
    period, category and component, history by history, priced with the
    diagram's costs (a cost drawn from a distribution at its mean) at the
    times they fell, but for the cells in ``skip``, ``{(category,
    component)}``. ``money`` (an ``_npv.Money``) gives the rate, the
    value earned (a rate or steps) and each stream's growth: an amount at
    ``t`` is its price times ``exp(g t)``. ``logs`` are each component's
    histories and ``system`` the system's (``timelines._Data``);
    ``system`` is needed only for a system downtime cost or a value.
    ``hidden`` gives, under shutdowns, each component's holds and the
    changes down they hid
    (``_timeline_runs.Hidden``): a hold is no task, and a hidden change
    is charged as a recorded one would be.

    With ``where``, ``rbd`` is the nested RBD at that path of a system,
    its histories its own (see ``_nested_costs``), priced into the
    system's ``ledger``: each of its components labelled with its path,
    and what it pays itself (its downtime, as its node's, and its crews)
    with its node's label; it earns no value, and each stream grows as
    the system's node it is in."""
    if ledger is None:
        ledger = Ledger(ends, money.rate, size)
    head = where[:1]

    def label(node):
        return where + (node,) if where else node

    def growth(node, category):
        return money.growth(*(head or (node,)), category)

    own = (where[0] if len(where) == 1 else where) if where else None
    for node, costs in rbd.costs.items():
        data = logs[node]
        history, times, down, planned = changes(data)
        tasks = down & planned
        left_out = None if hidden is None else hidden[node]
        if left_out is not None:
            tasks &= ~left_out.held
            history = np.concatenate((history, left_out.history))
            times = np.concatenate((times, left_out.times))
            down = np.concatenate((down, np.ones(left_out.times.size, bool)))
            planned = np.concatenate((planned, left_out.planned))
            tasks = np.concatenate((tasks, left_out.planned))
        failures = down & ~planned
        charges = [("repair_cost", "repair", failures)]
        if node not in rbd._imperfect:
            charges.append(("replace_cost", "replace", failures))
        charges.append(("preventive_cost", "preventive", tasks))
        for key, category, mask in charges:
            if key in costs and (category, node) not in skip:
                ledger.events(
                    history[mask],
                    times[mask],
                    _mean_cost(costs[key]),
                    category,
                    label(node),
                    growth(node, category),
                )
        rate_down = costs.get("downtime_cost", 0.0)
        if rate_down:
            ledger.spans(
                *spans(data, up=False),
                rate_down,
                "component_downtime",
                label(node),
                growth(node, "component_downtime"),
            )
    if where:
        if rbd.downtime_cost_rate:
            # The nested system down: its node down.
            ledger.spans(
                *spans(system, up=False),
                rbd.downtime_cost_rate,
                "component_downtime",
                own,
                money.growth(*head, "component_downtime"),
            )
    elif rbd.downtime_cost_rate or money.earns:
        ledger.spans(
            *spans(system, up=False),
            rbd.downtime_cost_rate,
            "system_downtime",
            None,
            money.growth("system_downtime"),
        )
        history, a, b = spans(system, up=True)
        for lo, hi, worth in zip(*money.steps(ledger.ends[-1])):
            # Each step of the value earned, over the spans up within it.
            inside = (b > lo) & (a < hi)
            ledger.spans(
                history[inside],
                np.maximum(a[inside], lo),
                np.minimum(b[inside], hi),
                worth,
                "value",
                None,
                money.growth("value"),
            )
    for node, upgrade in getattr(rbd, "_upgrades", {}).items():
        # An upgrade's capital cost, at its time, in every history.
        if upgrade.cost and ("capital", node) not in skip:
            at = np.full(size, upgrade.at)
            keep = at < ledger.ends[-1]
            ledger.events(
                np.arange(size)[keep],
                at[keep],
                upgrade.cost,
                "capital",
                label(node),
                growth(node, "capital"),
            )
    # The repair crews' salaries, the same in every history.
    ledger.spans(
        np.arange(size),
        np.zeros(size),
        np.full(size, ledger.ends[-1]),
        _crew_salaries(rbd),
        "crews",
        own,
        money.growth(*head, "crews"),
    )
    return ledger
