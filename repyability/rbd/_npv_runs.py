"""The net present value of each simulated run of a ``RepairableRBD``
(``RepairableRBD.npv_distribution``), and their spread.

Each run is a history of ``simulate_timelines`` (the event loop's, on
every engine), priced with the diagram's own costs and discounted at the
times they fell (``_pricing``): each failure its repair and replace
costs, each planned outage its preventive cost, each component's and the
system's time down their downtime costs, and the system's time up its
value, the integral of ``value_rate * exp(-r t)`` over its spans up. The
acquisition costs are paid at 0 in every run. A cost charged at an event
the histories do not record (``_pricing.unrecorded``: a standby group's
units' failures, a test, a replacement that takes no time, a set-up) is
not left out: the method refuses the diagram, naming it. Only the runs'
present values are kept, not their histories.
"""

import math
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

from repyability.rbd import _npv, _pricing, _requirements
from repyability.rbd._common import _discount_rate
from repyability.rbd.results import NPVDistribution, _order_statistic
from repyability.utils import checks

#: Simulations priced at a time: only their present values are kept.
_CHUNK = 2000

_INSTEAD = (
    " expected_npv gives the expected value with it, and cost() the "
    "spread of the undiscounted cost; to simulate the rest, leave the "
    "cost out."
)


def refusal(rbd) -> Optional[Tuple[str, Tuple[Hashable, ...]]]:
    """Why the runs of ``rbd`` cannot be priced from their histories, and
    the components named, or None if they can."""
    if rbd._shutdowns:
        return (
            "The diagram has shutdowns, which hold components off line and "
            "are charged at times the histories do not record: the runs "
            "cannot be priced from them." + _INSTEAD,
            (),
        )
    for category, node, why in _pricing.unrecorded(rbd):
        if category == "acquisition":
            continue
        whose = (
            "A maintenance group's set-up cost"
            if node is None
            else f"The {category} cost of component {node!r}"
        )
        extra = (
            " Give its planned replacements a duration, and each leaves "
            "an outage to price."
            if why == _pricing._INSTANT
            else ""
        )
        return (
            f"{whose} is charged at events the simulated histories do not "
            f"record: {why}{extra}" + _INSTEAD,
            (node,) if node is not None else (),
        )
    for node in rbd._restoring:
        if "replacement_cost" in rbd.costs.get(node, {}):
            return (
                f"The replacement cost of component {node!r} is charged at "
                "every so many of its tasks, which the histories do not "
                "count." + _INSTEAD,
                (node,),
            )
    for node, inspection in rbd._inspection.items():
        if inspection.duration is not None and "preventive_cost" in (
            rbd.costs.get(node, {})
        ):
            return (
                f"Component {node!r} has tests that take it off line and a "
                "preventive cost: its histories do not tell the two "
                "planned outages apart." + _INSTEAD,
                (node,),
            )
    return None


def _require_priceable(rbd) -> None:
    found = refusal(rbd)
    if found is not None:
        raise NotImplementedError(found[0])


def route(rbd, timelines_route):
    """``npv_distribution``'s route: ``simulate_timelines``' (whose
    histories it prices), unless a cost is not in them."""
    from repyability.rbd import routes as r

    if timelines_route.route == r.REFUSED:
        return timelines_route
    found = refusal(rbd)
    if found is not None:
        return r.refused(*found)
    return r.AnalysisRoute(
        r.SIMULATED,
        "The simulations simulate_timelines makes, each history priced "
        "with the diagram's costs and discounted at the times they fell.",
        engine=timelines_route.engine,
        engine_reason=timelines_route.engine_reason,
    )


def _share(value, name: str) -> float:
    number = checks.number_or_nan(value)
    if not (0.0 < number <= 1.0):
        raise ValueError(
            f"{name} must be a share of the runs, more than 0 and at most "
            f"1; got {value!r}."
        )
    return number


def npv_distribution(
    rbd,
    horizon,
    *,
    discount_rate: float,
    value_rate: float,
    mc_samples: Optional[int],
    seed,
    hurdle,
    tail: float,
    engine: str,
    state,
) -> NPVDistribution:
    """See ``RepairableRBD.npv_distribution``."""
    rate = _discount_rate(discount_rate)
    horizon = _npv._number(horizon, "horizon", strict=True)
    value_rate = _npv._number(value_rate, "value_rate", strict=False)
    if not value_rate:
        _requirements._require_costs(rbd)
    if hurdle is not None:
        hurdle = checks.number_or_nan(hurdle)
        if not math.isfinite(hurdle):
            raise ValueError(
                f"hurdle must be a finite number; got {hurdle!r}."
            )
    tail = _share(tail, "tail")
    n = checks.whole_number(
        10_000 if mc_samples is None else mc_samples, "mc_samples"
    )
    checks.one_of("engine", engine, ("auto", "python", "numba"))
    seed = checks.seed(seed)
    if seed is None:
        # The run is made in parts, which share a seed: drawn here, and
        # kept in the result to repeat the run.
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    _require_priceable(rbd)

    ends = np.array([horizon])
    present: Dict[Tuple[str, Any], np.ndarray] = {}
    nominal: Dict[Tuple[str, Any], np.ndarray] = {}
    used = None
    for first in range(0, n, _CHUNK):
        size = min(_CHUNK, n - first)
        runs = rbd.simulate_timelines(
            horizon,
            mc_samples=size,
            seed=seed,
            engine=engine,
            start=first,
            state=state,
        )
        used = used or runs.engine
        logs = {node: runs.components[node]._data for node in rbd.costs}
        ledger = _pricing.priced(
            rbd, logs, runs.system._data, ends, rate, value_rate, (), size
        )
        for key, (pv, plain) in ledger.totals().items():
            present.setdefault(key, np.zeros(n))[first : first + size] = pv
            nominal.setdefault(key, np.zeros(n))[first : first + size] = plain

    zero = np.zeros(n)
    values = present.pop(("value", None), zero)
    plain_value = nominal.pop(("value", None), zero)
    costs = np.zeros(n)
    plain_cost = 0.0
    by_category: Dict[str, float] = {}
    by_component: Dict[Hashable, float] = {}

    def add(category: str, node, amount: float) -> None:
        by_category[category] = by_category.get(category, 0.0) + amount
        if node is not None:
            by_component[node] = by_component.get(node, 0.0) + amount

    for node, price in rbd.acquisition_costs.items():
        if price:
            costs += float(price)
            plain_cost += float(price)
            add("acquisition", node, float(price))
    for (category, node), amounts in present.items():
        costs += amounts
        plain_cost += float(nominal[(category, node)].mean())
        add(category, node, float(amounts.mean()))

    samples = values - costs
    ordered = np.sort(samples)
    worst = max(1, math.ceil(round(tail * n, 9)))
    priced_at_mean: List[Hashable] = [
        node
        for node, node_costs in rbd.costs.items()
        if any(not isinstance(c, float) for c in node_costs.values())
    ]
    plain_value_mean = float(plain_value.mean())
    return NPVDistribution(
        horizon=horizon,
        discount_rate=rate,
        value_rate=value_rate,
        samples=samples,
        value_samples=values,
        cost_samples=costs,
        mean=float(samples.mean()),
        value=float(values.mean()),
        cost=float(costs.mean()),
        undiscounted_npv=plain_value_mean - plain_cost,
        undiscounted_value=plain_value_mean,
        undiscounted_cost=plain_cost,
        cost_by_category=by_category,
        cost_by_component=by_component,
        p10=_order_statistic(ordered, 0.1),
        p50=_order_statistic(ordered, 0.5),
        p90=_order_statistic(ordered, 0.9),
        hurdle=hurdle,
        probability_below_hurdle=(
            None if hurdle is None else float(np.mean(samples < hurdle))
        ),
        tail=tail,
        cvar=float(ordered[:worst].mean()),
        n_simulations=n,
        seed=seed,
        engine=used or engine,
        priced_at_mean=priced_at_mean,
    )
