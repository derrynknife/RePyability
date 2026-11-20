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

from repyability.rbd import _curves, _requirements, _shutdowns, _windows
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


def _retirement(retire_at, horizon: float) -> float:
    """When the plan ends: the horizon, or ``retire_at`` (checked: more
    than 0 and no later than the horizon), the time the system is
    retired at."""
    if retire_at is None:
        return horizon
    stop = _number(retire_at, "retire_at", strict=True)
    if math.isclose(stop, horizon, rel_tol=1e-12):
        return horizon
    if stop > horizon:
        raise ValueError(
            f"retire_at ({stop:g}) is past the horizon ({horizon:g}), "
            "where the valuation ends anyway: leave it out, or value a "
            "longer horizon."
        )
    return stop


def _salvage(rbd, salvage, at: float, state, forced) -> Dict[Any, float]:
    """The expected salvage credited at ``at`` (the retirement, or the
    horizon), undiscounted, by component (None: the system's):
    ``salvage`` checked, a number (the system's) or a dict of a number,
    or a function of the age of the unit in service (see
    ``_salvage_by_age``), for each component, and for None the
    system's."""
    if salvage is None:
        return {}
    given = salvage if isinstance(salvage, dict) else {None: salvage}
    out: Dict[Any, float] = {}
    for component, worth in given.items():
        if component is not None and component not in rbd.components:
            raise _requirements._not_a_component(rbd, component, "salvage")
        whose = "the system" if component is None else f"{component!r}"
        if callable(worth):
            if component is None:
                raise ValueError(
                    "The system's salvage must be a number: a salvage by "
                    "age is a component's, of the age of its unit in "
                    "service ({component: function})."
                )
            out[component] = _salvage_by_age(
                rbd, component, worth, at, state, forced
            )
            continue
        number = checks.number_or_nan(worth)
        if not math.isfinite(number):
            raise ValueError(
                f"The salvage of {whose} must be a finite number (what it "
                "is worth at the end; negative for a cost of disposal), or "
                "for a component a function of the age of its unit in "
                f"service; got {worth!r}."
            )
        out[component] = number
    return out


#: The cells the time to the end is cut into for a salvage by age.
_AGE_CELLS = 4000


def _salvage_by_age(rbd, node, worth, at: float, state, forced) -> float:
    """The expected salvage of a component at ``at``, ``worth(age)`` of
    the unit in service then if it is up (a unit down then, in a repair,
    is credited nothing): from the age of the unit in service, which is
    the time since it was put in, at 0 (never failed since: ``R(at)``) or
    at the end of a repair (the expected units put in over each cell of
    the time, from the component's expected failures and its repair
    time, each still up with probability ``R(age)``).

    That is a plain unit's, from new: one with a schedule, tests,
    imperfect repair, a group, crews to wait for or a state at 0 has no
    such age here, and is refused, asking for a number."""
    from repyability.rbd._block_replacement import _Duration
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD  # noqa

    states = _curves._states(rbd, state, forced)
    plain = (
        _curves._plain_unit(rbd, node)
        and type(rbd.components[node]).__name__ == "NonRepairable"
        and node not in forced
        and node not in (states or {})
        and not rbd._crews_couple()
        and not any(node in group.members for group in rbd.ccf_groups)
        and not _shutdowns.holding(rbd, node)
    )
    if not plain:
        raise NotImplementedError(
            f"Component {node!r}: a salvage by age needs the age of the "
            "unit in service at the end, which is worked out for a plain "
            "component from new (no maintenance, upgrade, tests, "
            "imperfect repair, standby, group, shutdown, crews to wait "
            "for or state at 0): give its salvage as a number."
        )
    component = rbd.components[node]
    curve = _curves._unit_curve(rbd, node, at, counts=True)
    edges = np.linspace(0.0, at, _AGE_CELLS + 1)
    middle = 0.5 * (edges[1:] + edges[:-1])
    failing = np.diff(curve.events(edges)["failures"])
    # The units put into service by each edge: those failed in each cell
    # (at its middle) whose repair is over by then.
    repair = _Duration(component.time_to_replace, "repair", node)
    since = edges[:, None] - middle[None, :]
    done = np.where(since > 0.0, repair.cdf(np.maximum(since, 0.0)), 0.0)
    put_in = np.diff(done @ failing)
    ages = np.concatenate(([at], at - middle))
    weights = np.concatenate(([1.0], put_in))
    alive = np.clip(
        np.asarray(component.reliability.sf(ages), float), 0.0, 1.0
    )
    values = np.asarray(worth(ages), dtype=float)
    if values.shape != ages.shape or not np.all(np.isfinite(values)):
        raise ValueError(
            f"The salvage of {node!r} by age must take an array of ages "
            "and return a finite value for each."
        )
    return float(np.sum(values * alive * weights))


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
    retire_at=None,
    salvage=None,
) -> ExpectedNPV:
    """See ``RepairableRBD.expected_npv``."""
    rate = _discount_rate(discount_rate)
    horizon = _number(horizon, "horizon", strict=True)
    value_rate = _number(value_rate, "value_rate", strict=False)
    table_ends = _period_ends(periods, horizon)
    # Retired before the horizon, nothing is earned or spent after: every
    # value is worked out to the retirement, and stays there.
    stop = _retirement(retire_at, horizon)
    ends = np.minimum(table_ends, stop)
    if not value_rate:
        _requirements._require_costs(rbd)
    method = checks.structure_method(method)
    working, broken = rbd._held_nodes(working_nodes, broken_nodes)
    starts = np.concatenate(([0.0], table_ends[:-1]))
    credits = _salvage(rbd, salvage, stop, state, working | broken)

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
            # The system's downtime, a group's set-up, a shutdown's and
            # the repair crews' costs are no component's.
            for category in ("system_downtime", "setup", "shutdown", "crews"):
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
        sign = 1.0 if category in ("value", "salvage") else -1.0
        for k in range(len(ends)):
            if present[k] == 0.0 and nominal[k] == 0.0:
                continue
            rows.append(
                {
                    "period": k + 1,
                    "start": float(starts[k]),
                    "end": float(table_ends[k]),
                    "category": category,
                    "component": component,
                    "present_value": sign * float(present[k]),
                    "undiscounted": sign * float(nominal[k]),
                }
            )
        if category not in ("value", "salvage"):
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
    # The salvage, credited at the retirement (or the horizon): in the
    # period it falls in.
    paid = table_ends >= stop
    worth_now = math.exp(-rate * stop)
    for component, amount in credits.items():
        if amount:
            add("salvage", component, worth_now * amount * paid, amount * paid)
    order = {
        name: i
        for i, name in enumerate(("acquisition",) + _CATEGORIES + ("salvage",))
    }
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
        -sum(
            r["undiscounted"]
            for r in rows
            if r["category"] not in ("value", "salvage")
        )
    )
    undiscounted_salvage = float(sum(credits.values()))
    salvaged = worth_now * undiscounted_salvage
    result = ExpectedNPV(
        horizon=horizon,
        discount_rate=rate,
        value_rate=value_rate,
        npv=value - cost + salvaged,
        value=value,
        cost=cost,
        eac=0.0,
        undiscounted_npv=(
            undiscounted_value - undiscounted_cost + undiscounted_salvage
        ),
        undiscounted_value=undiscounted_value,
        undiscounted_cost=undiscounted_cost,
        cost_by_category=cost_by_category,
        cost_by_component=cost_by_component,
        period_ends=[float(x) for x in table_ends],
        cash_flows=rows,
        salvage=salvaged,
        undiscounted_salvage=undiscounted_salvage,
        retire_at=None if retire_at is None else stop,
    )
    result.eac = cost / result.annuity_factor
    return result
