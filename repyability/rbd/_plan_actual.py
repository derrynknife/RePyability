"""A plan against what happened (``RepairableRBD.plan_vs_actual``): the
plan's expected cash flows over ``[0, now]`` (``expected_npv``'s table)
beside those an outage log implies, priced with the diagram's own costs
and discounted at the times they fell; and the plan re-valued from the
components' states now, which the log gives where the diagram's rules
let it.

What the log determines, by cost category:

- a failure (an unplanned change down) is charged its repair and replace
  costs at its instant, but for a standby group (whose log is the
  group's, not its units'), a tested unit (charged when a test finds the
  failure, which the log does not record) and an imperfect repair
  replaced at some failures (the replace cost, which falls at failures
  counted from a renewal the log does not mark);
- a planned outage of a unit with preventive replacements that take time
  is charged the preventive cost at its start; replacements that take no
  time leave no outage, and a calendar policy prices each at its
  opportunity: neither is compared;
- inspections and tests that find nothing leave no outage: the
  inspection costs are not compared; nor are a maintenance group's
  set-up costs, charged at stops the log does not all mark, nor the
  acquisition costs, which the log does not record (both carried as
  planned in the re-valued plan);
- the time each component is down (``downtime_cost``), the time the
  system is down (``downtime_cost_rate``, its timeline merged from the
  components' through the structure) and the value earned while it is up
  are integrated exactly over each span, discounted.
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

from repyability.rbd import _npv, _pricing, _requirements
from repyability.rbd._common import _discount_rate
from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd.node_state import NodeState
from repyability.rbd.results import ExpectedNPV, PlanVsActual


def _log_state_refusal(rbd, node) -> Optional[str]:
    """Why the log of component ``node`` does not give its state now
    under the diagram's rules, or None if it does: its age since its last
    restoration, how long it has been down and why, and where it is on a
    block or inspection calendar."""
    from repyability.rbd.degrading_node import DegradingNode
    from repyability.rbd.repairable_rbd import RepairableRBD

    from . import _crews

    component = rbd.components[node]
    ask = " Give its current state in state=."
    if isinstance(component, RepairableRBD):
        return (
            f"Component {node!r} is a nested RBD: its state is its own "
            "components' states, which its log (the nested system's) does "
            "not give." + ask
        )
    if node in rbd._standby:
        return (
            f"Component {node!r} is a standby group: its log does not say "
            "which of its units are up, nor their ages." + ask
        )
    if node in rbd._imperfect:
        return (
            f"Component {node!r} is repaired imperfectly: its log does not "
            "give its virtual age." + ask
        )
    if node in rbd._restoring:
        return (
            f"Component {node!r} is restored only partly by its "
            "maintenance: its log does not give its virtual age." + ask
        )
    if node in rbd._inspection:
        return (
            f"Component {node!r} has hidden failures: its log does not say "
            "when each was found, nor whether a failure is waiting for a "
            "test." + ask
        )
    schedule = rbd._preventive.get(node)
    if schedule is not None:
        if schedule.level is not None:
            return (
                f"Component {node!r} is replaced on condition by its "
                "measured degradation level, which its log does not give."
                + ask
            )
        if schedule.policy == "calendar":
            return (
                f"Component {node!r} is replaced on a calendar policy, "
                "whose state its log does not give." + ask
            )
        if schedule.duration is None:
            return (
                f"Component {node!r} has planned replacements that take no "
                "time, which leave no mark in its log: its age is not "
                "known." + ask
            )
    if isinstance(component.reliability, DegradingNode):
        return (
            f"Component {node!r} degrades through stages, and its log "
            "does not say which it is in." + ask
        )
    for what, model in (
        ("reliability", component.reliability),
        ("repairability", component.time_to_replace),
    ):
        if is_fixed_probability(model):
            return (
                f"Component {node!r}: its {what} model is a probability, "
                "not a distribution of times, so the values from a state "
                "part way through a life or repair are not worked out." + ask
            )
    if any(node in group.members for group in rbd.ccf_groups):
        return (
            f"Component {node!r} is in a common-cause group, whose values "
            "from a current state are not worked out." + ask
        )
    if _crews._crews_limited(rbd) and node in _crews._crew_served(rbd):
        return (
            f"Component {node!r} is repaired by a limited number of crews: "
            "its log does not say whether, while down, it was being "
            "repaired or waiting for a crew." + ask
        )
    return None


def _require_log_states(rbd, given=()) -> None:
    """Raise ``NotImplementedError`` for the first component, not in
    ``given``, whose log does not give its state now."""
    for node in rbd.components:
        if node in given:
            continue
        reason = _log_state_refusal(rbd, node)
        if reason is not None:
            raise NotImplementedError(reason)


def _require_no_plans(rbd) -> None:
    """Raise if a component is replaced at the times of a plan, which
    are counted from 0: the rest from today is valued on the same
    diagram from today, where they would be counted again."""
    if rbd._upgrades:
        raise NotImplementedError(
            f"{_requirements._upgraded(rbd)}, counted from 0, which the "
            "rest from today is not valued with yet: value the plan with "
            "expected_npv, and the rest on a diagram whose upgrade is at "
            "the time still to come, counted from today (or, once it is "
            "past, with the new life), with expected_npv(state=)."
        )
    planned = _requirements._planned(rbd)
    if planned:
        raise NotImplementedError(
            f"Component {planned[0]!r} is replaced at the times of a plan "
            "(replace_at), counted from 0, which the rest from today is "
            "not valued with yet: value the plan with expected_npv, and "
            "the rest on a diagram whose replace_at holds the times still "
            "to come, counted from today, with expected_npv(state=)."
        )


def route(rbd, npv_route):
    """``plan_vs_actual``'s route: ``expected_npv``'s (it values the plan
    and the rest from today with it), unless a component's state now is
    not given by its log."""
    from repyability.rbd import routes as r

    if npv_route.route == r.REFUSED:
        return npv_route
    planned = r.refusal(lambda: _require_no_plans(rbd))
    if planned:
        return r.refused(planned)
    for node in rbd.components:
        reason = _log_state_refusal(rbd, node)
        if reason is not None:
            return r.refused(reason, (node,))
    return npv_route


def _checked_logs(rbd, timelines, now: float) -> Dict[Hashable, Any]:
    """``timelines`` checked: a log (a ``Timeline``) for every component,
    each ending at ``now``."""
    from repyability.timelines import Timeline

    if not isinstance(timelines, Mapping):
        raise TypeError(
            "Give the outage logs as a mapping, {component: Timeline}."
        )
    for node, log in timelines.items():
        if node not in rbd.components:
            raise _requirements._not_a_component(rbd, node, "timelines")
        if not isinstance(log, Timeline):
            raise TypeError(
                f"The log of component {node!r} must be a Timeline (one "
                f"history), got {type(log).__name__}."
            )
        if not math.isclose(log.end, now, rel_tol=1e-12):
            raise ValueError(
                f"The log of component {node!r} ends at {log.end:g}, not at "
                f"now ({now:g}): every log runs from 0 to now."
            )
    missing = [node for node in rbd.components if node not in timelines]
    if missing:
        raise ValueError(
            f"No log for component(s) {missing}: a component left out is "
            "not taken to have run without failing. Give each its log "
            "(Timeline([], end=now) for one that never went down)."
        )
    return dict(timelines)


def _check_planned(rbd, node, log) -> None:
    """Refuse planned outages in the log of a component with nothing
    planned that takes it off line."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if not np.any(log.planned):
        return
    if isinstance(rbd.components[node], RepairableRBD):
        return
    schedule = rbd._preventive.get(node)
    inspection = rbd._inspection.get(node)
    if (schedule is not None and schedule.duration is not None) or (
        inspection is not None and inspection.duration is not None
    ):
        return
    raise ValueError(
        f"The log of component {node!r} has planned outages, but it has no "
        "maintenance or test that takes it off line: a change down not "
        "planned is a failure (planned=False)."
    )


def _start_of(start_state, node) -> NodeState:
    if start_state is None:
        return NodeState()
    value = start_state.get(node)
    return value if isinstance(value, NodeState) else NodeState()


def _derived_state(rbd, node, log, now: float, start: NodeState) -> NodeState:
    """The state of component ``node`` now from its log, for a kind
    ``_log_state_refusal`` passes: up, its age since its last restoration
    (or since the start); down, how long, and whether for maintenance
    (the last change's planned flag); and, on a block or inspection
    calendar, the time since its last scheduled time."""
    changes = log.changes
    count = changes.size
    up = log.up if count % 2 == 0 else not log.up
    since = None
    schedule = rbd._preventive.get(node)
    if schedule is not None and schedule.policy in ("block", "condition"):
        interval = schedule.interval
        since = math.fmod((start.since_scheduled or 0.0) + now, interval)
        if since >= interval or math.isclose(since, interval, rel_tol=1e-12):
            since = 0.0
    if up:
        age = now - changes[-1] if count else start.age + now
        return NodeState(age=float(age), since_scheduled=since)
    if count:
        down_for = now - changes[-1]
        maintenance = bool(log.planned[-1])
    else:
        down_for = start.down_for + now
        maintenance = start.maintenance
    return NodeState(
        alive=False,
        down_for=float(down_for),
        maintenance=maintenance,
        since_scheduled=since,
    )


def _states_now(rbd, logs, now, given, start_state) -> Tuple[dict, list]:
    """The components' states now: those ``given``, and the rest from
    their logs (refusing where a log does not give one); and the
    components whose states came from their logs."""
    _require_log_states(rbd, given)
    states = dict(given)
    derived = []
    for node in rbd.components:
        if node in given:
            continue
        start = _start_of(start_state, node)
        log = logs[node]
        if log.up != start.alive:
            raise ValueError(
                f"The log of component {node!r} starts "
                f"{'up' if log.up else 'down'}, but the plan starts it "
                f"{'up' if start.alive else 'down'} (start_state)."
            )
        states[node] = _derived_state(rbd, node, log, now, start)
        derived.append(node)
    return states, derived


# -- pricing a log ----------------------------------------------------------


def _priced_log(
    rbd, logs, ends: np.ndarray, rate: float, value_rate: float, skip
) -> List[dict]:
    """The cash flows the logs imply over ``[0, ends[-1]]``, by period,
    category and component, priced with the diagram's costs (a cost drawn
    from a distribution at its mean) at the times they fell, but for the
    cells in ``skip``, ``{(category, component)}`` (see ``_pricing``)."""
    from repyability.timelines import _system_timeline

    system = (
        _system_timeline(rbd, logs)._data
        if rbd.downtime_cost_rate or value_rate
        else None
    )
    data = {node: logs[node]._data for node in rbd.costs}
    return _pricing.priced(
        rbd, data, system, ends, rate, value_rate, skip
    ).rows()


# -- the analysis -----------------------------------------------------------


def _cut_at(ends: np.ndarray, now: float) -> np.ndarray:
    """The periods' ends with one at ``now``: a period ``now`` falls in
    is split there."""
    near = np.isclose(ends, now, rtol=1e-12, atol=0.0)
    if np.any(near):
        ends = ends.copy()
        ends[np.argmax(near)] = now
        return ends
    return np.sort(np.append(ends, now))


def _without_acquisition(result: ExpectedNPV) -> ExpectedNPV:
    """``result`` with no acquisition cost: the rest of a plan whose
    components are bought."""
    rows = [r for r in result.cash_flows if r["category"] != "acquisition"]
    bought = sum(
        -r["present_value"]
        for r in result.cash_flows
        if r["category"] == "acquisition"
    )
    if not bought:
        return result
    by_component = dict(result.cost_by_component)
    for node, price in _acquired(result.cash_flows).items():
        by_component[node] = by_component.get(node, 0.0) - price
    cost = result.cost - bought
    out = dataclasses.replace(
        result,
        npv=result.npv + bought,
        cost=cost,
        undiscounted_npv=result.undiscounted_npv + bought,
        undiscounted_cost=result.undiscounted_cost - bought,
        cost_by_category={
            k: v
            for k, v in result.cost_by_category.items()
            if k != "acquisition"
        },
        cost_by_component=by_component,
        cash_flows=rows,
    )
    out.eac = cost / out.annuity_factor
    return out


def _acquired(rows) -> Dict[Hashable, float]:
    out: Dict[Hashable, float] = {}
    for r in rows:
        if r["category"] == "acquisition":
            out[r["component"]] = (
                out.get(r["component"], 0.0) - r["present_value"]
            )
    return out


def _variance(actual: List[dict], expected: List[dict]) -> List[dict]:
    """Actual less expected, cell by cell (signed amounts: positive is in
    the plan's favour)."""
    cells: Dict[tuple, dict] = {}
    for rows, sign in ((actual, 1.0), (expected, -1.0)):
        for r in rows:
            key = (r["period"], r["category"], r["component"])
            cell = cells.setdefault(
                key,
                _pricing.row(
                    r["period"],
                    r["start"],
                    r["end"],
                    r["category"],
                    r["component"],
                    0.0,
                    0.0,
                ),
            )
            cell["present_value"] += sign * r["present_value"]
            cell["undiscounted"] += sign * r["undiscounted"]
    return _pricing.sort_rows(list(cells.values()))


def _total(rows, key: str = "present_value") -> float:
    return float(sum(r[key] for r in rows))


def plan_vs_actual(
    rbd,
    timelines,
    now,
    horizon,
    *,
    discount_rate: float,
    value_rate: float,
    periods,
    method: str,
    state,
    start_state,
) -> PlanVsActual:
    """See ``RepairableRBD.plan_vs_actual``."""
    rate = _discount_rate(discount_rate)
    now = _npv._number(now, "now", strict=True)
    horizon = _npv._number(horizon, "horizon", strict=True)
    if math.isclose(now, horizon, rel_tol=1e-12):
        now = horizon
    if now > horizon:
        raise ValueError(
            f"now ({now:g}) is past the horizon ({horizon:g}): the plan "
            "ends before today."
        )
    value_rate = _npv._number(value_rate, "value_rate", strict=False)
    for name, given in (("state", state), ("start_state", start_state)):
        if given is not None and not isinstance(given, Mapping):
            raise TypeError(
                f"{name} must be a dict of NodeStates by component, got "
                f"{given!r}."
            )
    ends = _cut_at(_npv._period_ends(periods, horizon), now)
    logs = _checked_logs(rbd, timelines, now)
    for node, log in logs.items():
        _check_planned(rbd, node, log)

    with rbd._sharing_curves():
        plan = _npv.expected_npv(
            rbd,
            horizon,
            discount_rate=rate,
            value_rate=value_rate,
            periods=ends,
            working_nodes=None,
            broken_nodes=None,
            method=method,
            state=start_state,
        )
    _require_no_plans(rbd)
    given = dict(state or {})
    states, derived = _states_now(rbd, logs, now, given, start_state)
    so_far = int(np.searchsorted(ends, now, side="right"))
    rest_ends = ends[so_far:] - now
    rest = None
    if rest_ends.size:
        rest_ends[-1] = horizon - now
        with rbd._sharing_curves():
            rest = _without_acquisition(
                _npv.expected_npv(
                    rbd,
                    horizon - now,
                    discount_rate=rate,
                    value_rate=value_rate,
                    periods=rest_ends,
                    working_nodes=None,
                    broken_nodes=None,
                    method=method,
                    state=states,
                )
            )

    left_out = _pricing.unrecorded(rbd)
    skip = {(category, node) for category, node, _ in left_out}
    before = [r for r in plan.cash_flows if r["period"] <= so_far]
    expected = [
        r for r in before if (r["category"], r["component"]) not in skip
    ]
    as_planned = [r for r in before if (r["category"], r["component"]) in skip]
    actual = _priced_log(rbd, logs, ends[:so_far], rate, value_rate, skip)
    variance = _variance(actual, expected)

    back = math.exp(-rate * now)
    rest_rows = []
    if rest is not None:
        for r in rest.cash_flows:
            rest_rows.append(
                _pricing.row(
                    r["period"] + so_far,
                    r["start"] + now,
                    r["end"] + now,
                    r["category"],
                    r["component"],
                    r["present_value"] * back,
                    r["undiscounted"],
                )
            )
    by_category: Dict[str, float] = {}
    by_component: Dict[Hashable, float] = {}
    for r in variance:
        by_category[r["category"]] = (
            by_category.get(r["category"], 0.0) + r["present_value"]
        )
        if r["component"] is not None:
            by_component[r["component"]] = (
                by_component.get(r["component"], 0.0) + r["present_value"]
            )

    actual_so_far = _total(actual)
    as_planned_so_far = _total(as_planned)
    rest_npv = _total(rest_rows)
    revalued = actual_so_far + as_planned_so_far + rest_npv
    return PlanVsActual(
        now=now,
        horizon=horizon,
        discount_rate=rate,
        value_rate=value_rate,
        plan_npv=plan.npv,
        revalued_npv=revalued,
        change=revalued - plan.npv,
        expected_so_far=_total(expected),
        actual_so_far=actual_so_far,
        variance_so_far=_total(variance),
        undiscounted_expected_so_far=_total(expected, "undiscounted"),
        undiscounted_actual_so_far=_total(actual, "undiscounted"),
        as_planned_so_far=as_planned_so_far,
        plan_rest_npv=_total(
            [r for r in plan.cash_flows if r["period"] > so_far]
        ),
        rest_npv=rest_npv,
        variance_by_category=by_category,
        variance_by_component=by_component,
        period_ends=[float(x) for x in ends],
        expected_cash_flows=expected,
        actual_cash_flows=actual,
        variance_cash_flows=variance,
        rest_cash_flows=rest_rows,
        not_compared=[
            {"category": c, "component": n, "reason": why}
            for c, n, why in left_out
        ],
        states=states,
        derived_states=derived,
        plan=plan,
        rest=rest,
    )
