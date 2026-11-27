"""A ``RepairableRBD``'s equivalent annual cost in the long run: the steady
cost per unit time worth the same, at a discount rate, as every cost of
running the diagram for ever from new.

Each component's own costs are discounted over its renewal cycle (Fox,
1966): a unit run to failure or replaced at an age starts afresh after
each repair or replacement, so the present value of all its costs is its
first cycle's over ``1 - E[exp(-r L)]``, ``L`` the cycle's length. With
``M`` the discounted time up, ``failed`` and ``kept`` the chances of a
cycle ending at a failure or at the age, each discounted to then, and
``m`` the discounted mean time of what follows (a repair, or the
replacement at the age), the rate cancels::

    eac = (failed (c_f + d m_f) + kept (c_p + d m_p))
          / (M + failed m_f + kept m_p)

which at rate 0 is the long-run cost rate (renewal reward).

A component with hidden failures, tested and repaired in no time by tests
that find every failure, starts afresh at the test that finds it failed:
its cycle is the tests up to that one (``_tested_eac``), the first cycle
from new with its tests from their offset. The system's
own terms (the cost of its downtime, a value earned while it works) are
taken at its long-run availability: an approximation, as a system new at
0 is up more often at first, which ``basis`` says.
"""

import math
from typing import Hashable

import numpy as np

from repyability.rbd import _requirements
from repyability.rbd._common import _crew_salaries
from repyability.rbd._mean_lifetime import _integrate, model_knots
from repyability.rbd._model_utils import MODEL_ERRORS, model_mean
from repyability.rbd._spec import _mean_cost

#: Where a discounted time stops: where an amount is discounted to
#: ``exp(-REACH)``, or the time has all but surely ended.
REACH = 40.0
ENDED = 1e-12

#: What a refusal names to run instead.
INSTEAD = (
    'Use objective="npv" with a horizon, which discounts the expected '
    "costs over time (expected_npv), or simulate the plans "
    "(npv_distribution)."
)


def _discounted_time(model, rate: float) -> float:
    """The integral of ``exp(-r t) S(t)`` over all ``t``, ``S`` the
    survival function of a time's ``model``: its discounted mean (its
    mean at rate 0), and 0 for a time that is none."""
    if model is None:
        return 0.0

    def sf(t) -> np.ndarray:
        return np.clip(np.ravel(np.asarray(model.sf(t), dtype=float)), 0, 1)

    if not float(sf(np.zeros(1))[0]) > 0.0:
        return 0.0
    if rate == 0.0:
        return float(model_mean(model))
    try:
        upper = float(model_mean(model))
    except MODEL_ERRORS:
        upper = math.nan
    if not (math.isfinite(upper) and upper > 0.0):
        upper = 1.0
    for _ in range(400):
        if rate * upper >= REACH:
            upper = REACH / rate
            break
        if float(sf(np.array([upper]))[0]) <= ENDED:
            break
        upper *= 2.0
    knots = model_knots(model)
    inside = knots[(knots > 0.0) & (knots < upper)]
    return float(
        _integrate(
            lambda t: np.exp(-rate * t) * sf(t),
            np.concatenate(([0.0], inside, [upper])),
        )
    )


def require_discounted(rbd) -> None:
    """Raise unless every priced part of the diagram has a discounted
    long run worked out here: its priced components plain units or under
    age replacement, with no repair crews to wait for, common-cause
    groups, maintenance groups or shutdowns."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    def refuse(what: str) -> None:
        raise NotImplementedError(
            f"The equivalent annual cost at a discount rate is worked out "
            f"from each component's own renewal cycle, which {what} does "
            f"not have. {INSTEAD}"
        )

    if rbd._crews_couple():
        refuse("a component that can wait for a repair crew")
    if rbd.ccf_groups:
        refuse("a member of a common-cause group")
    if rbd._shutdowns:
        refuse("a diagram with shutdowns")
    if any(group.setup_cost for group in rbd._maintenance.values()):
        refuse("a maintenance group with a set-up cost")
    for node in rbd.costs:
        component = rbd.components[node]
        schedule = rbd._preventive.get(node)
        if isinstance(component, RepairableRBD):
            refuse(f"the nested RBD {node!r}")
        if node in rbd._standby and rbd._standby[node].key == "markov":
            refuse(f"the Markov component {node!r}")
        if node in rbd._standby:
            refuse(f"the standby group {node!r}")
        if node in rbd._inspection and not _tested_cycle(rbd, node):
            refuse(
                f"the tested component {node!r}, whose tests or repairs "
                "take time or can miss a failure,"
            )
        if node in rbd._imperfect:
            refuse(f"component {node!r}, whose repairs are not perfect")
        if node in rbd._early_members:
            refuse(f"component {node!r}, renewed early at its group's stops")
        if schedule is not None and schedule.policy != "age":
            refuse(f"component {node!r}, under {schedule.policy} replacement,")
        _requirements._require_full_restoration(rbd, node)
        _requirements._require_times(rbd, node)


def _tested_cycle(rbd, node) -> bool:
    """Whether a component with hidden failures has the cycle
    ``_tested_eac`` discounts: tested and repaired in no time, by tests
    at an interval that find every failure."""
    inspection = rbd._inspection[node]
    return (
        _requirements._tested_kind(rbd, node) in ("closed", "life")
        and not inspection.partial
        and inspection.times is None
    )


def _tested_eac(rbd, node: Hashable, rate: float) -> float:
    """The equivalent annual cost at ``rate`` (more than 0) of a
    component with hidden failures (see ``_tested_cycle``): ``rate``
    times the present value of its costs for ever from new.

    A cycle from a new unit is the tests at ``t_1 < t_2 < ...`` up to the
    first after its failure at ``X``, which finds it and renews it:
    every test of the cycle is paid for, the failure when it is found,
    and the unit's own downtime from ``X`` to then. With ``p_k`` the
    chance that test ``k`` finds it, a cycle's discount is ``phi = sum
    exp(-r t_k) p_k`` and its cost ``K``; the first cycle's tests are
    from the offset, and every later one's an interval apart from the
    test that began it, so the present value is ``K_1 + phi_1 K / (1 -
    phi)``."""
    from repyability.maintenance._options import _discounted_length, _reach

    costs = rbd.costs.get(node, {})
    component = rbd.components[node]
    inspection = rbd._inspection[node]
    interval = float(inspection.interval)
    survival = component.reliability_function
    reach = _reach(component, rate)
    # A new unit's failure, discounted: 1 - r M, M its discounted life.
    failure = 1.0 - rate * _discounted_length(component, 0.0, reach, rate)
    per_failure = sum(
        _mean_cost(costs[key])
        for key in rbd.PER_FAILURE_COST_KEYS
        if key in costs
    )
    per_test = _mean_cost(costs.get("inspection_cost", 0.0))
    down = costs.get("downtime_cost", 0.0)

    def cycle(first: float):
        """A cycle's cost, its discount and one less its discount, its
        tests from ``first``."""
        count = max(int(math.ceil((reach - first) / interval)), 0) + 1
        tests = first + interval * np.arange(count)
        alive = np.clip(
            np.ravel(
                np.asarray(
                    survival(np.concatenate(([0.0], tests))), dtype=float
                )
            ),
            0.0,
            1.0,
        )
        alive[0] = 1.0
        found = alive[:-1] - alive[1:]
        factor = np.exp(-rate * tests)
        discount = float(factor @ found)
        cost = (
            per_test * float(factor @ alive[:-1])
            + per_failure * discount
            + down * (failure - discount) / rate
        )
        # What is left past the reach is discounted to nothing.
        less = float(-np.expm1(-rate * tests) @ found) + float(alive[-1])
        return cost, discount, less

    cost, discount, less = cycle(interval)
    if inspection.offset:
        first_cost, first_discount, _ = cycle(float(inspection.offset))
    else:
        first_cost, first_discount = cost, discount
    return rate * (first_cost + first_discount * cost / less)


def node_eac(rbd, node: Hashable, rate: float) -> float:
    """A component's own equivalent annual cost at ``rate`` (see the
    module docstring): its failures', its replacements' at an age, its
    tests' and its own downtime's."""
    from repyability.maintenance._options import cycle_eac

    if node in rbd._inspection:
        return _tested_eac(rbd, node, rate)
    costs = rbd.costs.get(node, {})
    component = rbd.components[node]
    schedule = rbd._preventive.get(node)
    age = math.inf if schedule is None else float(schedule.interval)
    return cycle_eac(
        component,
        age,
        rate,
        failure_cost=sum(
            _mean_cost(costs[key])
            for key in rbd.PER_FAILURE_COST_KEYS
            if key in costs
        ),
        planned_cost=_mean_cost(costs.get("preventive_cost", 0.0)),
        downtime=costs.get("downtime_cost", 0.0),
        repair=_discounted_time(component.time_to_replace, rate),
        planned=(
            0.0
            if schedule is None
            else _discounted_time(schedule.duration, rate)
        ),
    )


def system_terms(rbd, availability: float, value_rate: float) -> float:
    """The diagram's own steady terms of an equivalent annual cost, at
    its long-run ``availability``: the repair crews' salaries and the
    cost of its downtime, less the value it earns while it works."""
    return (
        _crew_salaries(rbd)
        + rbd.downtime_cost_rate * (1.0 - availability)
        - value_rate * availability
    )


def basis(rbd, rate: float, value_rate: float) -> str:
    """What a plan's equivalent annual cost is made of, in words: said on
    the result, with what is approximate in it."""
    if rate == 0.0:
        return "Not discounted: the long-run cost rate" + (
            ", less the value rate times the long-run availability."
            if value_rate
            else "."
        )
    own = (
        "Each component's own costs are discounted over its renewal "
        "cycle, exactly (Fox, 1966)."
    )
    system = [
        name
        for name, given in (
            ("the cost of the system's downtime", rbd.downtime_cost_rate),
            ("the value earned while it works", value_rate),
        )
        if given
    ]
    if not system:
        return own
    return (
        f"{own} Approximate: {' and '.join(system)} "
        f"{'are' if len(system) > 1 else 'is'} taken at the long-run "
        "availability, though a system new at 0 is up more often at "
        'first; objective="npv" with a horizon discounts the system\'s '
        "availability over time."
    )
