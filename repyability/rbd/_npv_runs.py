"""The net present value of each simulated run of a ``RepairableRBD``
(``RepairableRBD.npv_distribution``), and their spread.

Each run is a history of ``simulate_timelines`` (the event loop's, on
every engine), priced with the diagram's own costs and discounted at the
times they fell (``_pricing``): each failure its repair and replace
costs, each planned outage its preventive cost, each component's and the
system's time down their downtime costs, and the system's time up its
value, the integral of ``value_rate * exp(-r t)`` over its spans up (each
step's rate over its part of them, for a ``ValueSchedule``), every amount
grown at its stream's escalation (``price * exp(g t)``, see
``_npv.Money``). The
acquisition costs are paid at 0 in every run, and each shutdown's cost at
each of its starts (its calendar's, the same in every run). A component a
shut holds off line shows a planned change in its history that is no
task: the Python loop marks those holds, and records the changes down a
hold hid (a failure in the shut, a renewal made in it), which are charged
as recorded ones are (``_timeline_runs.Hidden``). A cost charged at an event
the histories do not record (``_pricing.unrecorded``: a standby group's
units' failures, a test, a replacement that takes no time, a set-up) is
not left out: the method refuses the diagram, naming it. Only the runs'
present values are kept, not their histories.

An ``AvailabilityContract`` adds, run by run, each period's penalty or
bonus, from the share of the period the system was up in that run (its
spans up, as for the value), paid at the period's end. ``compare_npv``
prices two plans' runs from one seed: a component in the same place in
both draws the same random numbers in both (the streams, as ``compare``
has them), so each run's difference comes from how the plans differ.
"""

import math
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

from repyability.rbd import (
    _crews,
)
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _npv, _pricing, _requirements, _streams
from repyability.rbd.contracts import AvailabilityContract
from repyability.rbd.results import (
    ConfidenceInterval,
    NPVComparison,
    NPVDistribution,
    _order_statistic,
)
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
    if _crews._call_out_priced(rbd):
        return (
            "The call-out crew's fee and rate are charged when it is "
            "called and while it is engaged, which the simulated "
            "histories do not record." + _INSTEAD,
            (),
        )
    for category, node, why in _pricing.unrecorded(rbd):
        if category in ("acquisition", "salvage"):
            # (Paid at the start, and credited at the end, by the runs.)
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
    """Refuse a diagram whose runs cannot be priced (each valuation of
    runs credits the components' own salvage and retirement, through
    ``_npv.own_end``)."""
    found = refusal(rbd)
    if found is not None:
        raise NotImplementedError(found[0])


def route(rbd, timelines_route):
    """``npv_distribution``'s route: ``simulate_timelines``' (whose
    histories it prices), unless a cost is not in them."""
    from repyability.rbd import routes as r

    if timelines_route.route == r.REFUSED:
        return timelines_route
    own = r.refusal(
        lambda: _npv.own_end(rbd, math.inf, _npv.Money(rbd, 0.0, 0.0))
    )
    if own is not None:
        return r.refused(own, tuple(rbd._valuation))
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


def compare_route(npv_route, rbd):
    """``compare_npv``'s route, for this diagram against one like it:
    ``npv_distribution``'s, the runs matched, unless its draws cannot
    be."""
    from repyability.rbd import routes as r

    if npv_route.route == r.REFUSED:
        return npv_route
    why = r.refusal(lambda: _require_matched(rbd, "this plan", 1.0, None))
    if why is not None:
        return r.refused(why, tuple(_unmatched(rbd, 1.0, None) or ()))
    return r.AnalysisRoute(
        r.SIMULATED,
        "Both plans' npv_distribution runs from one seed, a component in "
        "the same place in both drawing the same random numbers (as for "
        "compare), each run's difference priced.",
        engine=npv_route.engine,
        engine_reason=npv_route.engine_reason,
    )


def _share(value, name: str) -> float:
    number = checks.number_or_nan(value)
    if not (0.0 < number <= 1.0):
        raise ValueError(
            f"{name} must be a share of the runs, more than 0 and at most "
            f"1; got {value!r}."
        )
    return number


def _contract(contract) -> Optional[AvailabilityContract]:
    if contract is None or isinstance(contract, AvailabilityContract):
        return contract
    if isinstance(contract, dict):
        return AvailabilityContract.from_dict(contract)
    raise TypeError(
        "contract must be an AvailabilityContract (or its to_dict()), or "
        f"None; got {type(contract).__name__}."
    )


def _settings(rbd, horizon, money, contract, mc_samples):
    """The checked settings both methods share."""
    horizon = _npv._number(horizon, "horizon", strict=True)
    contract = _contract(contract)
    ends = (
        None if contract is None else np.array(contract.period_ends(horizon))
    )
    if not money.earns and contract is None:
        _requirements._require_costs(rbd)
    n = checks.whole_number(
        10_000 if mc_samples is None else mc_samples, "mc_samples"
    )
    return horizon, contract, ends, n


def _seed(seed):
    # The run is made in parts, which share a seed: drawn here (from
    # numpy's global RNG) if not given, and kept in the result to repeat
    # the run.
    return _streams.shared_seed(checks.seed(seed))


def priced_chunk(
    rbd,
    horizon: float,
    ends: np.ndarray,
    money,
    size: int,
    seed: int,
    engine: str,
    first: int,
    state,
):
    """Simulations ``first`` to ``first + size - 1`` of the run ``seed``
    seeds, over ``horizon``, priced (``_pricing.priced``, by the periods
    ending at ``ends``, valued by ``money``, an ``_npv.Money``) with each
    shutdown's cost at its starts: the
    ledger, the runs (a ``TimelineSimulation``)."""
    from repyability.rbd import _timeline_runs

    runs, hidden = _timeline_runs.simulate(
        rbd,
        horizon,
        size,
        seed,
        None,
        None,
        False,
        engine,
        None,
        first,
        state,
        hidden=True,
    )
    logs = {node: runs.components[node]._data for node in rbd.costs}
    ledger = _pricing.priced(
        rbd,
        logs,
        runs.system._data,
        ends,
        money,
        (),
        size,
        hidden=hidden if rbd._shutdowns else None,
    )
    for shut in rbd._shutdowns.values():
        # Its cost at each start, in every run (as the loop charges it).
        starts = shut.starts(horizon)
        ledger.events(
            np.repeat(np.arange(size), starts.size),
            np.tile(starts, size),
            shut.cost,
            "shutdown",
            None,
            money.growth("shutdown"),
        )
    return ledger, runs


def _up_by_period(system, ends: np.ndarray, size: int) -> np.ndarray:
    """Each history's availability in each period ending at ``ends``: the
    share of the period it was up, ``(size, periods)``."""
    history, a, b = _pricing.spans(system, up=True)
    starts = np.concatenate(([0.0], ends[:-1]))
    out = np.zeros((size, ends.size))
    for k, (lo, hi) in enumerate(zip(starts, ends)):
        inside = np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None)
        out[:, k] = np.bincount(history, inside, minlength=size)
    return out / (ends - starts)


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
    contract=None,
    escalation=None,
    rates: str = "nominal",
    inflation=None,
) -> NPVDistribution:
    """See ``RepairableRBD.npv_distribution``."""
    escalation = _npv.own_escalation(rbd, escalation)
    money = _npv.Money(
        rbd, discount_rate, value_rate, escalation, rates, inflation
    )
    horizon, contract, ends, n = _settings(
        rbd, horizon, money, contract, mc_samples
    )
    # The runs end at a retirement, and every run credits the salvage.
    end = _npv.own_end(rbd, horizon, money)
    if end[0] < horizon:
        horizon, contract, ends, n = _settings(
            rbd, end[0], money, contract, mc_samples
        )
    if hurdle is not None:
        hurdle = checks.number_or_nan(hurdle)
        if not math.isfinite(hurdle):
            raise ValueError(
                f"hurdle must be a finite number; got {hurdle!r}."
            )
    tail = _share(tail, "tail")
    checks.one_of("engine", engine, ("auto", "python", "numba"))
    seed = _seed(seed)
    _require_priceable(rbd)
    return _priced_runs(
        rbd,
        horizon,
        money,
        contract,
        ends,
        n,
        seed,
        engine,
        state,
        hurdle,
        tail,
        end,
    )


def _priced_runs(
    rbd,
    horizon: float,
    money,
    contract: Optional[AvailabilityContract],
    contract_ends: Optional[np.ndarray],
    n: int,
    seed: int,
    engine: str,
    state,
    hurdle: Optional[float],
    tail: float,
    end=None,
) -> NPVDistribution:
    """The ``n`` runs of ``seed``, priced (settings checked); ``end``,
    ``_npv.own_end``'s, credits each run the salvage."""
    salvage, plain_salvage, retired = (
        (0.0, 0.0, False) if end is None else (end[1], end[2], end[3])
    )
    ends = np.array([horizon])
    present: Dict[Tuple[str, Any], np.ndarray] = {}
    nominal: Dict[Tuple[str, Any], np.ndarray] = {}
    available = (
        None if contract_ends is None else np.zeros((n, contract_ends.size))
    )
    used = None
    for first in range(0, n, _CHUNK):
        size = min(_CHUNK, n - first)
        ledger, runs = priced_chunk(
            rbd,
            horizon,
            ends,
            money,
            size,
            seed,
            engine,
            first,
            state,
        )
        used = used or runs.engine
        for key, (pv, plain) in ledger.totals().items():
            present.setdefault(key, np.zeros(n))[first : first + size] = pv
            nominal.setdefault(key, np.zeros(n))[first : first + size] = plain
        if available is not None:
            assert contract_ends is not None
            available[first : first + size] = _up_by_period(
                runs.system._data, contract_ends, size
            )

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

    paid = zero
    plain_paid = 0.0
    if contract is not None:
        assert available is not None and contract_ends is not None
        payments = contract.payoff(available)
        # Its payments grow only with the inflation of real rates.
        lift = money.growth()
        paid = payments @ np.exp(-(money.rate - lift) * contract_ends)
        plain_paid = float(
            (payments @ np.exp(lift * contract_ends)).mean()
            if lift
            else payments.sum(axis=1).mean()
        )

    samples = values - costs + paid + salvage
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
        discount_rate=money.given_rate,
        value_rate=money.value_rate,
        samples=samples,
        value_samples=values,
        cost_samples=costs,
        mean=float(samples.mean()),
        value=float(values.mean()),
        cost=float(costs.mean()),
        undiscounted_npv=(
            plain_value_mean - plain_cost + plain_paid + plain_salvage
        ),
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
        contract=None if contract is None else contract.to_dict(),
        contract_samples=None if contract is None else paid,
        contract_payoff=float(paid.mean()) if contract is not None else 0.0,
        undiscounted_contract_payoff=plain_paid,
        period_ends=(
            None
            if contract_ends is None
            else [float(e) for e in contract_ends]
        ),
        period_availability=available,
        salvage=salvage,
        undiscounted_salvage=plain_salvage,
        retire_at=horizon if retired else None,
        **money.stated(),
    )


def _unmatched(rbd, horizon: float, state) -> Optional[List[Hashable]]:
    """The components of ``rbd`` whose draws do not all come from a
    stream of their own (so cannot be matched with another plan's), or
    None if every draw does."""
    from repyability.rbd import _curves, _event_loop
    from repyability.rbd.repairable_rbd import RepairableRBD

    states = _curves._simulation_states(rbd, state)
    specs, complete = _event_loop._stream_specs(rbd, horizon, states=states)
    if complete:
        return None
    named = []
    for name, component in rbd.components.items():
        if type(component) is RepairableRBD:
            if not _event_loop._stream_specs(
                component, horizon, (name,), {}, states.get(name)
            )[1]:
                named.append(name)
            continue
        own = [path for path, _ in specs if path[:1] == (name,)]
        duration = _event_loop._duration_model(rbd, name)
        if not own or (
            duration is not None and ((name,), _streams.DURATION) not in specs
        ):
            named.append(name)
    return named


def _require_matched(rbd, which: str, horizon: float, state) -> None:
    from repyability.rbd._runs import _UNSTREAMED

    named = _unmatched(rbd, horizon, state)
    if named is None:
        return
    whose = f" (components {', '.join(map(repr, named))})" if named else ""
    raise NotImplementedError(
        f"The runs of {which} cannot be matched with the other plan's"
        f"{whose}: {_UNSTREAMED} Compare their npv_distribution results "
        "run independently instead, or model those components with "
        "surpyval parametric distributions."
    )


def _own(escalation, rbd, other):
    """``escalation`` without the components only ``other`` has."""
    if not isinstance(escalation, dict):
        return escalation
    return {
        key: rate
        for key, rate in escalation.items()
        if key in rbd.components or key not in other.components
    }


def compare_npv(
    rbd,
    other,
    horizon,
    *,
    discount_rate: float,
    value_rate: float,
    contract,
    mc_samples: Optional[int],
    seed,
    confidence: float,
    engine: str,
    state,
    escalation=None,
    rates: str = "nominal",
    inflation=None,
) -> NPVComparison:
    """See ``RepairableRBD.compare_npv``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if not isinstance(other, RepairableRBD):
        raise TypeError(
            "other must be a RepairableRBD, the plan to compare with; got "
            f"{type(other).__name__}."
        )
    # Each plan's own: an escalation may name a component of either plan,
    # which the other leaves out.
    monies = [
        _npv.Money(
            d,
            discount_rate,
            value_rate,
            _npv.own_escalation(d, _own(escalation, d, o)),
            rates,
            inflation,
        )
        for d, o in ((rbd, other), (other, rbd))
    ]
    settings = [
        _settings(d, horizon, money, contract, mc_samples)
        for d, money in zip((rbd, other), monies)
    ]
    # Each plan's runs end at its own retirement, crediting its salvage.
    ends_of = [
        _npv.own_end(d, s[0], money)
        for d, s, money in zip((rbd, other), settings, monies)
    ]
    settings = [
        _settings(d, end[0], money, contract, mc_samples) if end[3] else s
        for d, s, money, end in zip((rbd, other), settings, monies, ends_of)
    ]
    n = settings[0][3]
    montecarlo.check_confidence(confidence)
    checks.one_of("engine", engine, ("auto", "python", "numba"))
    seed = _seed(seed)
    for diagram, which, s in zip(
        (rbd, other), ("this plan", "the other plan"), settings
    ):
        _require_priceable(diagram)
        _require_matched(diagram, which, s[0], state)
    this, that = (
        _priced_runs(
            diagram,
            s[0],
            money,
            s[1],
            s[2],
            n,
            seed,
            engine,
            state,
            None,
            0.1,
            end,
        )
        for diagram, money, s, end in zip(
            (rbd, other), monies, settings, ends_of
        )
    )
    differences = this.samples - that.samples
    estimate = float(differences.mean())
    se = montecarlo.standard_error(differences, False)
    apart = math.sqrt(
        montecarlo.standard_error(this.samples, False) ** 2
        + montecarlo.standard_error(that.samples, False) ** 2
    )
    z = montecarlo.z_value(confidence)
    return NPVComparison(
        differences=differences,
        mean=estimate,
        interval=ConfidenceInterval(
            estimate=estimate,
            lower=estimate - z * se,
            upper=estimate + z * se,
            confidence=confidence,
            standard_error=se,
            n_samples=n,
            method="simulated",
        ),
        probability_other_better=float(np.mean(differences < 0.0)),
        independent_standard_error=apart,
        this=this,
        other=that,
        n_simulations=n,
        seed=seed,
    )
