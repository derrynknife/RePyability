"""The expected net present value of running a ``RepairableRBD`` over a
horizon (``RepairableRBD.expected_npv``): the costs of
``_windows.expected_cost``, discounted at their exact times, less the
present value of a steady value earned while the system works, the
integral of ``exp(-r t) A(t)`` from its point availability ``A``.
"""

import math
from functools import partial
from typing import Any, Collection, Dict, Hashable, List, Optional

import numpy as np

from repyability.rbd import _curves, _requirements, _windows
from repyability.rbd._common import (
    _discount_rate,
    _present_horizon,
)
from repyability.rbd._tally import _CATEGORIES
from repyability.rbd.results import ExpectedNPV
from repyability.utils import checks


def _number(value, name: str, strict: bool) -> float:
    number = checks.number_or_nan(value)
    low = number > 0.0 if strict else number >= 0.0
    if not (math.isfinite(number) and low):
        what = "more than 0" if strict else "at least 0"
        raise ValueError(
            f"{name} must be a finite number, {what}; got {value!r}."
        )
    return number


def _period_ends(periods, horizon: float) -> np.ndarray:
    """The ends of the periods of the cash-flow table: ``periods`` equal
    ones if a whole number, else the increasing ends given, the last the
    horizon."""
    if periods is None:
        return np.array([horizon])
    if checks.is_number(periods):
        count = float(periods)
        if count != int(count) or count < 1:
            raise ValueError(
                "periods must be a whole number of periods, at least 1, "
                f"or the ends of the periods; got {periods!r}."
            )
        return np.linspace(0.0, horizon, int(count) + 1)[1:]
    ends = checks.real_array(periods, "periods").ravel().astype(float)
    ok = (
        ends.size > 0
        and bool(np.all(np.isfinite(ends)))
        and ends[0] > 0.0
        and bool(np.all(np.diff(ends) > 0.0))
        and math.isclose(ends[-1], horizon, rel_tol=1e-12)
    )
    if not ok:
        raise ValueError(
            "periods, given as the ends of the periods, must be increasing "
            f"and positive, the last the horizon ({horizon:g}); got "
            f"{periods!r}."
        )
    ends[-1] = horizon
    return ends


def _uptime(rbd, ends, rate, working, broken, method, state) -> np.ndarray:
    """The integral from 0 to each of ``ends`` of ``exp(-rate t)`` times
    the system's point availability."""
    states = _curves._states(rbd, state, working | broken)
    horizon = float(ends.max())
    system_at, pieces, crew = _windows._system_at(
        rbd, horizon, working, broken, method, states
    )
    if crew is not None and not crew.nested and not rate:
        return np.asarray(crew.integral(ends), dtype=float)
    return _windows._integrated(
        rbd, system_at, pieces, ends, horizon, rate=rate
    )


def expected_npv(
    rbd,
    horizon,
    *,
    discount_rate: float,
    value_rate: float,
    periods,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
) -> ExpectedNPV:
    """See ``RepairableRBD.expected_npv``."""
    rate = _discount_rate(discount_rate)
    horizon = _number(horizon, "horizon", strict=True)
    value_rate = _number(value_rate, "value_rate", strict=False)
    ends = _period_ends(periods, horizon)
    if not value_rate:
        _requirements._require_costs(rbd)
    method = checks.structure_method(method)
    working, broken = rbd._held_nodes(working_nodes, broken_nodes)
    starts = np.concatenate(([0.0], ends[:-1]))

    present_cells: Dict[Any, Any] = {}
    nominal_cells: Dict[Any, Any] = {}
    systems: Dict[str, Any] = {}
    with rbd._sharing_curves():
        if value_rate:
            up_present = value_rate * _uptime(
                rbd, ends, rate, working, broken, method, state
            )
            up_nominal = (
                value_rate
                * _uptime(rbd, ends, 0.0, working, broken, method, state)
                if rate
                else up_present
            )
        else:
            up_present = up_nominal = np.zeros(len(ends))
        if rbd.has_costs:
            # The costs, cumulative at each period's end, by component
            # and category; discounted at their exact times, and not.
            got = _windows.expected_cost(
                rbd,
                ends,
                working,
                broken,
                method,
                state,
                discount_rate=rate,
                cells=present_cells,
            )
            if rate:
                plain = _windows.expected_cost(
                    rbd,
                    ends,
                    working,
                    broken,
                    method,
                    state,
                    discount_rate=0.0,
                    cells=nominal_cells,
                )
            else:
                plain, nominal_cells = got, present_cells
            # The system's downtime, a group's set-up and a shutdown's
            # costs are no component's.
            for category in ("system_downtime", "setup", "shutdown"):
                a = np.atleast_1d(got.by_category[category])
                if np.any(a):
                    systems[category] = (
                        a,
                        np.atleast_1d(plain.by_category[category]),
                    )
        elif rate:
            _present_horizon(
                np.array([horizon]),
                rate,
                partial(_requirements._mean_lives, rbd),
            )

    def increments(cumulative) -> np.ndarray:
        cumulative = np.broadcast_to(
            np.asarray(cumulative, dtype=float), (len(ends),)
        )
        return np.diff(np.concatenate(([0.0], cumulative)))

    rows: List[Dict[str, Any]] = []
    cost_by_category: Dict[str, float] = {}
    cost_by_component: Dict[Hashable, float] = {}

    def add(category, component, present, nominal) -> None:
        present, nominal = increments(present), increments(nominal)
        sign = 1.0 if category == "value" else -1.0
        for k in range(len(ends)):
            if present[k] == 0.0 and nominal[k] == 0.0:
                continue
            rows.append(
                {
                    "period": k + 1,
                    "start": float(starts[k]),
                    "end": float(ends[k]),
                    "category": category,
                    "component": component,
                    "present_value": sign * float(present[k]),
                    "undiscounted": sign * float(nominal[k]),
                }
            )
        if category != "value":
            total = float(present.sum())
            cost_by_category[category] = (
                cost_by_category.get(category, 0.0) + total
            )
            if component is not None:
                cost_by_component[component] = (
                    cost_by_component.get(component, 0.0) + total
                )

    for node, price in rbd.acquisition_costs.items():
        if price:
            # Paid at the start: all of it in the first period.
            paid = np.full(len(ends), float(price))
            add("acquisition", node, paid, paid)
    for (node, category), values in present_cells.items():
        add(category, node, values, nominal_cells[(node, category)])
    for category, (present, nominal) in systems.items():
        add(category, None, present, nominal)
    add("value", None, up_present, up_nominal)
    order = {name: i for i, name in enumerate(("acquisition",) + _CATEGORIES)}
    rows.sort(
        key=lambda r: (
            r["period"],
            order.get(r["category"], len(order)),
            str(r["component"]),
        )
    )

    value = float(up_present[-1])
    cost = float(sum(cost_by_category.values()))
    undiscounted_value = float(up_nominal[-1])
    undiscounted_cost = float(
        -sum(r["undiscounted"] for r in rows if r["category"] != "value")
    )
    result = ExpectedNPV(
        horizon=horizon,
        discount_rate=rate,
        value_rate=value_rate,
        npv=value - cost,
        value=value,
        cost=cost,
        eac=0.0,
        undiscounted_npv=undiscounted_value - undiscounted_cost,
        undiscounted_value=undiscounted_value,
        undiscounted_cost=undiscounted_cost,
        cost_by_category=cost_by_category,
        cost_by_component=cost_by_component,
        period_ends=[float(x) for x in ends],
        cash_flows=rows,
    )
    result.eac = cost / result.annuity_factor
    return result
