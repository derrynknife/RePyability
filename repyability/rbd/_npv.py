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
from repyability.value import _escalation as _money
from repyability.value._escalation import STREAMS as _STREAMS  # noqa: F401
from repyability.value._escalation import inflation as _inflation
from repyability.value.value_schedule import ValueSchedule, value_of


def _escalation(rbd, escalation) -> Dict[Any, float]:
    """``escalation`` checked for ``rbd``'s components."""
    return _money.escalation(rbd.components, escalation)


def _number(value, name: str, strict: bool) -> float:
    number = checks.number_or_nan(value)
    low = number > 0.0 if strict else number >= 0.0
    if not (math.isfinite(number) and low):
        what = "more than 0" if strict else "at least 0"
        raise ValueError(
            f"{name} must be a finite number, {what}; got {value!r}."
        )
    return number


def steady_value(value_rate, method: str) -> float:
    """A ``value_rate`` that must be one number (``method`` takes no
    schedule yet), checked."""
    if isinstance(value_rate, ValueSchedule):
        raise ValueError(
            f"{method} takes the value earned as one rate per unit time; "
            "a ValueSchedule of steps is valued by expected_npv. Give "
            "value_rate a number."
        )
    return _number(value_rate, "value_rate", strict=False)


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
    rate_sensitivity: bool = False,
    escalation=None,
    rates: str = "nominal",
    inflation=None,
) -> ExpectedNPV:
    """See ``RepairableRBD.expected_npv``."""
    if not isinstance(rate_sensitivity, (bool, np.bool_)):
        raise TypeError(
            "rate_sensitivity must be True or False (whether to work out "
            f"the NPV's derivative in the discount rate); got "
            f"{rate_sensitivity!r}."
        )
    given_rate = _discount_rate(discount_rate)
    horizon = _number(horizon, "horizon", strict=True)
    value = value_of(value_rate)
    schedule = value if isinstance(value, ValueSchedule) else None
    growths = _escalation(rbd, escalation)
    lift = _inflation(inflation, rates)
    # In real terms with inflation, every amount is made nominal: the
    # rate and every stream's growth take the inflation too, which
    # leaves the present values as they are.
    rate = given_rate + lift

    growth = partial(_money.growth, growths, lift)

    earns = bool(value) if schedule is None else any(schedule.rates)
    table_ends = _period_ends(periods, horizon)
    # Retired before the horizon, nothing is earned or spent after: every
    # value is worked out to the retirement, and stays there.
    stop = _retirement(retire_at, horizon)
    ends = np.minimum(table_ends, stop)
    if not earns:
        _requirements._require_costs(rbd)
    method = checks.structure_method(method)
    working, broken = rbd._held_nodes(working_nodes, broken_nodes)
    starts = np.concatenate(([0.0], table_ends[:-1]))
    credits = _salvage(rbd, salvage, stop, state, working | broken)

    present_cells: Dict[Any, Any] = {}
    nominal_cells: Dict[Any, Any] = {}
    systems: Dict[str, Any] = {}
    worth = 0.0 if schedule is not None else float(value)
    value_growth = growth("value")
    with rbd._sharing_curves():

        def earned(effective: float) -> np.ndarray:
            """The value earned to each end, each amount at ``t`` times
            ``exp(-effective t)``: a steady rate's, or each step's."""
            if schedule is None:
                return worth * _uptime(
                    rbd, ends, effective, working, broken, method, state
                )
            starts, stops, step_rates = schedule.steps(stop)
            knots = np.unique(np.concatenate([ends, starts[1:]]))
            found = dict(
                zip(
                    knots.tolist(),
                    _uptime(
                        rbd, knots, effective, working, broken, method, state
                    ).tolist(),
                )
            )
            found[0.0] = 0.0
            out = np.zeros(len(ends))
            for a, b, step in zip(starts, stops, step_rates):
                if step:
                    upper = np.minimum(ends, b)
                    lower = np.minimum(ends, a)
                    out += step * np.array(
                        [found[u] - found[w] for u, w in zip(upper, lower)]
                    )
            return out

        if earns:
            up_present = earned(rate - value_growth)
            up_nominal = (
                earned(-value_growth) if rate or value_growth else up_present
            )
        else:
            up_present = up_nominal = np.zeros(len(ends))
        if rbd.has_costs:
            # The costs, cumulative at each period's end, by component
            # and category; discounted at their exact times, and not:
            # each at its stream's growth, worked out once for each rate
            # less a growth (a growth above the rate is a rate below 0,
            # valid over a horizon).
            found_at: Dict[float, Any] = {}

            def costs_at(effective: float):
                if effective not in found_at:
                    cells: Dict[Any, Any] = {}
                    if effective >= 0.0:
                        got = _windows.expected_cost(
                            rbd,
                            ends,
                            working,
                            broken,
                            method,
                            state,
                            discount_rate=effective,
                            cells=cells,
                        )
                    else:
                        got = _windows._discounted_cost(
                            rbd,
                            ends,
                            effective,
                            working,
                            broken,
                            method,
                            state,
                            cells,
                        )
                    found_at[effective] = (got, cells)
                return found_at[effective]

            for node, category in costs_at(rate)[1]:
                g = growth(node, category)
                present_cells[(node, category)] = costs_at(rate - g)[1][
                    (node, category)
                ]
                nominal_cells[(node, category)] = costs_at(-g)[1][
                    (node, category)
                ]
            # The system's downtime, a group's set-up, a shutdown's and
            # the repair crews' costs are no component's.
            for category in ("system_downtime", "setup", "shutdown", "crews"):
                g = growth(category)
                a = np.atleast_1d(costs_at(rate - g)[0].by_category[category])
                if np.any(a):
                    systems[category] = (
                        a,
                        np.atleast_1d(costs_at(-g)[0].by_category[category]),
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
    salvage_growth = growth("salvage")
    worth_now = math.exp(-(rate - salvage_growth) * stop)
    grown = math.exp(salvage_growth * stop) if salvage_growth else 1.0
    for component, amount in credits.items():
        if amount:
            add(
                "salvage",
                component,
                worth_now * amount * paid,
                grown * amount * paid,
            )
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
    undiscounted_salvage = grown * float(sum(credits.values()))
    salvaged = worth_now * float(sum(credits.values()))
    result = ExpectedNPV(
        horizon=horizon,
        discount_rate=given_rate,
        value_rate=worth if schedule is None else math.nan,
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
        rate_basis=rates,
        inflation=None if inflation is None else lift,
        escalation=dict(growths),
        value_schedule=schedule,
    )
    result.eac = cost / result.annuity_factor
    if rate_sensitivity:

        def npv_at(r: float) -> float:
            return expected_npv(
                rbd,
                horizon,
                discount_rate=r,
                value_rate=value_rate,
                periods=None,
                working_nodes=working_nodes,
                broken_nodes=broken_nodes,
                method=method,
                state=state,
                retire_at=retire_at,
                salvage=salvage,
                escalation=escalation,
                rates=rates,
                inflation=inflation,
            ).npv

        result.rate_sensitivity, result.rate_sensitivity_error = (
            _rate_derivative(npv_at, given_rate, result.npv, stop)
        )
    return result


#: The step of the discount rate the NPV's derivative is differenced on,
#: times the time valued: the rate's effect on an amount at ``t`` is
#: through ``r t``, so the step is a share of ``1 / stop`` whatever the
#: unit of time.
_RATE_STEP = 1e-2


def _rate_derivative(npv_at, rate: float, base: float, stop: float):
    """The derivative of the NPV in the discount rate at ``rate``, and an
    estimate of its error: Richardson's extrapolation of differences on
    steps ``h`` and ``2 h`` (``h = _RATE_STEP / stop``), central where the
    rate stays at least 0 and one-sided (second order) from ``rate``
    otherwise; the error is a third of the two steps' difference, the
    size of the smaller step's own error, which the extrapolation
    removes the leading term of."""
    h = _RATE_STEP / stop
    if rate >= 2.0 * h:
        up1, down1 = npv_at(rate + h), npv_at(rate - h)
        up2, down2 = npv_at(rate + 2.0 * h), npv_at(rate - 2.0 * h)
        fine = (up1 - down1) / (2.0 * h)
        coarse = (up2 - down2) / (4.0 * h)
    else:
        f1, f2, f4 = (npv_at(rate + k * h) for k in (1.0, 2.0, 4.0))
        fine = (-3.0 * base + 4.0 * f1 - f2) / (2.0 * h)
        coarse = (-3.0 * base + 4.0 * f2 - f4) / (4.0 * h)
    return (4.0 * fine - coarse) / 3.0, abs(fine - coarse) / 3.0
