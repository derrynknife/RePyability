"""The least spend that keeps a bad run's downtime under a limit
(``RepairableRBD.least_spend_for_downtime``, UC-68 of the maintenance
README).

The target is the system's downtime over a horizon, in hours or as its
cost (each hour down at the diagram's ``downtime_cost_rate`` and any
``value_rate`` lost), counting all of it (``downtime="all"``) or the
unplanned part alone. A design meets the limit when an upper confidence
bound on the downtime's risk measure (``repyability.value._risk``: the
quantile at ``level``, an order statistic, or the tail mean) is at or
under it.

The levers are each at their own price, and spend has no bound but an
optional ``budget``:

- ``redundancy``: copies of a node in parallel with it, at a price each
  (drawn as ``allocate_redundancy`` draws a train of one node: fed as the
  node is, feeding the node it feeds, whose ``k`` is kept);
- ``intervals``: a node's preventive task's interval, through the
  candidates given, its spend the expected preventive cost over the
  horizon (from the same runs);
- ``inspections``: a node's test interval (hidden failures), through
  the candidates given, its spend the expected inspection cost;
- ``upgrades``: a better life for a node's units from the start (a
  better model bought now), through the offers given, at its capital
  price per unit (a copy of an upgraded node is an upgraded unit too);
- ``crews``: repair crews more than the diagram's ``repair_crews``, at
  a price each and their salaries at the ``crew_cost_rate``;
- ``call_out``: a call-out crew's contract, through the contracts given,
  its spend its expected fees and time engaged;
- ``spares``: spares more on a node's shelf, at a holding price each
  (over the horizon). Every design is run with the shelf (see
  ``_shelves``): from its stock as given (by default none), each spare
  used reordered at once and arriving the lead time later, a repair that
  finds the shelf empty waiting for one. A node's copies draw on its
  shelf (one part), and a group's units on the group's.

Besides the levers' prices, a design's spend is the change in its
expected cost of preventive tasks, inspections and crews over the
horizon, from the same runs (a copy's tasks cost too).

A group of units (a standby, load-sharing or identical block) is one
node: a copy of it is a second group in parallel, and its units'
repairs are jobs for the crews as any other.

The search is greedy marginal analysis, as the allocations' searches
are: from the diagram as it is, each step takes the move that lowers the
risk estimate most per extra spend, every design run on the same seed
(common random numbers), until a design meets the limit or no move
lowers the risk. Its designs are the frontier of spend against the bad
run's downtime. When the levers stop short, the result says which floor
stopped them: the budget, the levers' caps, or downtime no lever left
lowers (the planned task downtime, when it is planned).

With ``uncertainty``, the models are not taken as known: each design
is run over draws of the fitted models (as ``expected_npv_uncertainty``
draws them), the same draws and seeds for every design, its runs shared
among the draws and pooled, so the bad run's downtime is that of the
predictive distribution; the result says so (``uncertainty``).

Every design reports both planned and unplanned downtime (each
simulation's, from the engines' own split, ``_unplanned``): their means
and their quantiles at ``level``, in hours and, with a rate, in cost.
"""

import math
from typing import (
    Dict,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability.rbd import _shelves
from repyability.rbd.results import DowntimeDesign, DowntimeRiskResult
from repyability.utils import checks
from repyability.value import _risk

#: Each design's runs by default.
MC_SAMPLES = 2000
#: The most steps the search takes.
MAX_STEPS = 200


def _share(value, name: str) -> float:
    number = checks.number_or_nan(value)
    if not 0.0 < number < 1.0:
        raise ValueError(
            f"{name} must be a share strictly between 0 and 1; got "
            f"{value!r}."
        )
    return float(number)


def _copy_name(rbd, node, index: int):
    name = f"{node} (copy {index})"
    if name in rbd.G.nodes:
        raise ValueError(
            f"A copy of node {node!r} would be named {name!r}, which the "
            "diagram already has: rename that node."
        )
    return name


class _Lever(NamedTuple):
    """One lever of the search, at a step from 0 (the diagram as given):
    a count lever (``"copies"``, ``"crews"``, ``"spares"``) adds one more
    at each step, at ``price`` each, up to ``cap`` (None: no bound); a
    choice lever (``"interval"``, ``"call_out"``) takes
    ``options[step - 1]``, its options in the order to try them. A
    spares lever's ``shelf`` is the shelf as given (its stock, lead time
    and whether repairs take a spare)."""

    kind: str
    node: Optional[Hashable]
    price: float = 0.0
    cap: Optional[int] = None
    options: Tuple = ()
    shelf: Optional[_shelves.Shelf] = None


#: The cost categories the levers' spend is counted in, besides their
#: prices: their change from the diagram as given, from the same runs.
SPEND_CATEGORIES = ("preventive", "inspection", "crews")


def _price(value, what: str, positive: bool = True) -> float:
    cost = checks.number_or_nan(value)
    if not (math.isfinite(cost) and (cost > 0.0 if positive else cost >= 0)):
        more = "more than 0" if positive else "at least 0"
        raise ValueError(f"{what} must be a number {more}; got {value!r}.")
    return float(cost)


def _cap(value, what: str) -> Optional[int]:
    if value is not None and not (
        checks.is_number(value) and int(value) == value and value >= 0
    ):
        raise ValueError(
            f"{what} must be a whole number, at least 0; got {value!r}."
        )
    return None if value is None else int(value)


def _levers(
    rbd,
    redundancy,
    intervals,
    inspections,
    upgrades,
    max_units,
    crews,
    call_out,
    spares=None,
) -> List[_Lever]:
    """The levers, checked, in the order the search tries their moves:
    the redundancy nodes', the interval nodes', the inspection nodes',
    the upgrade nodes', the spares nodes', the crews' and the call-out
    contracts'. Each
    option of a choice
    lever is built once here, so that one the diagram cannot take is
    refused before the search."""
    from repyability.rbd import _repairable_allocation as allocation
    from repyability.rbd import _requirements

    levers: List[_Lever] = []
    for node, price in (redundancy or {}).items():
        _requirements._require_component(rbd, node, "redundancy")
        allocation._allocation_trains(rbd, {("copies", node): [node]}, ())
        if not isinstance(rbd._init_args["components"].get(node), dict):
            raise ValueError(
                f"Node {node!r} is not given as a spec or a UnitPlan, so it "
                "cannot be copied here: give it as one."
            )
        grouped = [
            name
            for name, spec in (
                rbd._init_args.get("maintenance_groups") or {}
            ).items()
            if node in spec.get("members", ())
        ]
        if grouped or any(
            node in getattr(g, "members", ()) for g in rbd.ccf_groups
        ):
            raise NotImplementedError(
                f"Node {node!r} is in a maintenance or common-cause group, "
                "which its copies would have to join: copy nodes outside "
                "groups, or use allocate_redundancy for groups."
            )
        levers.append(
            _Lever(
                "copies",
                node,
                price=_price(price, f"The price of a copy of node {node!r}"),
                cap=_cap(
                    None if max_units is None else max_units.get(node),
                    f"max_units for node {node!r}",
                ),
            )
        )
    for node, values in (intervals or {}).items():
        _requirements._require_component(rbd, node, "intervals")
        spec = rbd._init_args["components"].get(node)
        preventive = spec.get("preventive") if isinstance(spec, dict) else None
        if not isinstance(preventive, dict) or "interval" not in preventive:
            raise ValueError(
                f"Node {node!r} has no preventive task with an interval to "
                "move: give it one (its 'preventive' spec's 'interval')."
            )
        row = checks.real_array(values, f"intervals[{node!r}]").ravel()
        if row.size == 0 or not np.all(np.isfinite(row) & (row > 0.0)):
            raise ValueError(
                f"intervals[{node!r}] must be the intervals to try, in the "
                f"order to try them, each more than 0; got {values!r}."
            )
        levers.append(
            _Lever("interval", node, options=tuple(float(v) for v in row))
        )
    for node, values in (inspections or {}).items():
        _requirements._require_component(rbd, node, "inspections")
        spec = rbd._init_args["components"].get(node)
        test = spec.get("inspection") if isinstance(spec, dict) else None
        if not isinstance(test, dict) or "interval" not in test:
            raise ValueError(
                f"Node {node!r} has no inspection with an interval to "
                "move: give it one (its 'inspection' spec's 'interval')."
            )
        row = checks.real_array(values, f"inspections[{node!r}]").ravel()
        if row.size == 0 or not np.all(np.isfinite(row) & (row > 0.0)):
            raise ValueError(
                f"inspections[{node!r}] must be the test intervals to try, "
                f"in the order to try them, each more than 0; got {values!r}."
            )
        levers.append(
            _Lever("inspection", node, options=tuple(float(v) for v in row))
        )
    for node, offers in (upgrades or {}).items():
        levers.append(_upgrade_lever(rbd, node, offers))
    if spares is not None and not isinstance(spares, Mapping):
        raise ValueError(
            "spares must be a dict of node: its shelf's lever, e.g. "
            "{'pump': {'price': 400, 'lead_time': 720}}; got "
            f"{spares!r}."
        )
    for node, spec in (spares or {}).items():
        levers.append(_spares_lever(rbd, node, spec))
    if crews is not None:
        levers.append(_crews_lever(rbd, crews))
    if call_out is not None:
        levers.append(_call_out_lever(rbd, call_out))
    for i, lever in enumerate(levers):
        for step in range(1, len(lever.options) + 1):
            state = tuple(step if j == i else 0 for j in range(len(levers)))
            try:
                _design(rbd, levers, state)
            except (ValueError, TypeError, NotImplementedError) as error:
                what = (
                    lever.kind
                    if lever.node is None
                    else f"{lever.kind} of node {lever.node!r}"
                )
                raise type(error)(
                    f"The {what}'s option {step} cannot be taken: {error}"
                ) from error
    return levers


def _fixed_spend(levers: Sequence[_Lever], state: Sequence[int]) -> float:
    """The levers' prices: a count lever's at each step, an upgrade's
    for each unit of its node (its copies are upgraded units too)."""
    copies = {v.node: s for v, s in zip(levers, state) if v.kind == "copies"}
    total = 0.0
    for lever, step in zip(levers, state):
        if lever.kind == "upgrade":
            if step:
                total += lever.options[step - 1][1] * (
                    1 + copies.get(lever.node, 0)
                )
        else:
            total += lever.price * step
    return total


def _upgrade_lever(rbd, node, offers) -> _Lever:
    """``upgrades[node]``: better lives for the node's units from the
    start, each at its capital price per unit, tried in order."""
    from repyability.rbd import _requirements

    _requirements._require_component(rbd, node, "upgrades")
    spec = rbd._init_args["components"].get(node)
    if not isinstance(spec, dict):
        raise ValueError(
            f"Node {node!r} is not given as a spec or a UnitPlan, so its "
            "life cannot be upgraded here: give it as one."
        )
    if spec.get("upgrade") is not None:
        raise ValueError(
            f"Node {node!r} already has an upgrade on its spec; an upgrade "
            "lever replaces its life from the start: leave one out."
        )
    listed = [offers] if isinstance(offers, Mapping) else offers
    if (
        isinstance(listed, (str, bytes))
        or not isinstance(listed, Sequence)
        or not listed
    ):
        raise ValueError(
            f"upgrades[{node!r}] must be an upgrade (a dict of "
            "'reliability', the better life, and 'cost', its price per "
            f"unit) or a list of them to try, in order; got {offers!r}."
        )
    options = []
    for offer in listed:
        if not isinstance(offer, Mapping) or set(offer) != {
            "reliability",
            "cost",
        }:
            raise ValueError(
                f"Each of upgrades[{node!r}] must be a dict of "
                "'reliability' (the better life) and 'cost' (its price per "
                f"unit); got {offer!r}."
            )
        options.append(
            (
                offer["reliability"],
                _price(
                    offer["cost"], f"The cost of an upgrade of node {node!r}"
                ),
            )
        )
    return _Lever("upgrade", node, options=tuple(options))


#: The keys of a spares lever.
SPARES_KEYS = ("price", "lead_time", "max", "stock", "repairs")


def _spares_lever(rbd, node, spec) -> _Lever:
    """``spares[node]``: spares more on the node's shelf, each at its
    ``"price"`` (its holding cost over the horizon), up to ``"max"`` more,
    from the shelf's ``"stock"`` as given (by default 0), with its
    ``"lead_time"`` and whether ``"repairs"`` take a spare."""
    if (
        not isinstance(spec, Mapping)
        or set(spec) - set(SPARES_KEYS)
        or not {"price", "lead_time"} <= set(spec)
    ):
        raise ValueError(
            f"spares[{node!r}] must be a dict of 'price' (what each spare "
            "more on the shelf costs to hold over the horizon) and "
            "'lead_time' (the time a spare ordered takes to arrive), and "
            "optionally 'max' (the most spares more), 'stock' (the spares "
            "held now, by default 0) and 'repairs' (whether a failure "
            f"repaired imperfectly takes a spare too); got {spec!r}."
        )
    _shelves.require(rbd, node, "spares")
    shelf = _shelves.shelf(
        spec.get("stock", 0),
        spec["lead_time"],
        spec.get("repairs", False),
        f"the shelf of node {node!r}",
    )
    return _Lever(
        "spares",
        node,
        price=_price(spec["price"], f"The price of a spare of node {node!r}"),
        cap=_cap(spec.get("max"), f"spares[{node!r}]'s 'max'"),
        shelf=shelf,
    )


def _shelved(rbd, diagram, levers: Sequence[_Lever], state: Sequence[int]):
    """``diagram`` (a design of ``rbd``) with each spares lever's shelf at
    its step: its stock as given and as many more, its node's copies
    drawing on it too."""
    shelves, of = {}, {}
    copies = {v.node: s for v, s in zip(levers, state) if v.kind == "copies"}
    for lever, step in zip(levers, state):
        if lever.kind != "spares":
            continue
        assert lever.shelf is not None  # a spares lever has its shelf
        shelves[lever.node] = lever.shelf._replace(
            stock=lever.shelf.stock + step
        )
        of[lever.node] = lever.node
        for index in range(1, copies.get(lever.node, 0) + 1):
            of[_copy_name(rbd, lever.node, index)] = lever.node
    if not shelves:
        return diagram
    return _shelves.shelved(diagram, _shelves.Shelves(shelves, of))


def _crews_lever(rbd, crews) -> _Lever:
    """``crews``: more repair crews than the diagram's ``repair_crews``,
    each at its ``"price"`` (and its salary at the ``crew_cost_rate``,
    from the runs), up to ``"max"`` more."""
    if not isinstance(crews, Mapping) or set(crews) - {"price", "max"}:
        raise ValueError(
            "crews must be a dict of 'price' (what each crew more costs "
            "over the horizon, besides its salary at the crew_cost_rate) "
            f"and 'max' (the most crews more); got {crews!r}."
        )
    if rbd.repair_crews is None:
        raise ValueError(
            "crews= adds repair crews to the diagram's repair_crews, but it "
            "has as many as are needed (repair_crews=None), so no job "
            "waits: give the diagram repair_crews, or leave crews out."
        )
    price = _price(crews.get("price", 0.0), "crews' 'price'", positive=False)
    if price == 0.0 and not rbd.crew_cost_rate > 0.0:
        raise ValueError(
            "A crew more must cost something: give crews a 'price', or the "
            "diagram a crew_cost_rate (each crew's salary)."
        )
    return _Lever(
        "crews", None, price=price, cap=_cap(crews.get("max"), "crews' 'max'")
    )


def _call_out_lever(rbd, call_out) -> _Lever:
    """``call_out``: the call-out contracts to try, in order, each priced
    by its fee and its rate (from the runs)."""
    from repyability.rbd import _spec

    contracts = [call_out] if isinstance(call_out, Mapping) else call_out
    if (
        isinstance(contracts, (str, bytes))
        or not isinstance(contracts, Sequence)
        or not contracts
    ):
        raise ValueError(
            "call_out must be a call-out contract (a dict, as the "
            "diagram's call_out takes) or a list of them to try, in order; "
            f"got {call_out!r}."
        )
    options = []
    for contract in contracts:
        if not isinstance(contract, Mapping):
            raise ValueError(
                f"Each call_out contract must be a dict; got {contract!r}."
            )
        checked = _spec._validate_call_out(dict(contract), rbd.repair_crews)
        if checked is None or not (checked.fee > 0.0 or checked.rate > 0.0):
            raise ValueError(
                "A call-out contract must cost something: give it a 'fee' "
                f"or a 'rate'; got {dict(contract)!r}."
            )
        options.append(dict(contract))
    return _Lever("call_out", None, options=tuple(options))


def _design(rbd, levers: Sequence[_Lever], state: Sequence[int]):
    """The diagram with each lever at its step in ``state`` (as given at
    0): copies made last, of the node as its other levers leave it."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if not any(state):
        return rbd
    args = dict(rbd._init_args)
    components = dict(args["components"])
    edges = [tuple(e) for e in args["edges"]]
    k = dict(args.get("k") or {})
    for lever, step in zip(levers, state):
        if not step:
            continue
        if lever.kind == "interval":
            spec = dict(components[lever.node])
            spec["preventive"] = dict(
                spec["preventive"], interval=lever.options[step - 1]
            )
            components[lever.node] = spec
        elif lever.kind == "inspection":
            spec = dict(components[lever.node])
            spec["inspection"] = dict(
                spec["inspection"], interval=lever.options[step - 1]
            )
            components[lever.node] = spec
        elif lever.kind == "upgrade":
            spec = dict(components[lever.node])
            spec["reliability"] = lever.options[step - 1][0]
            components[lever.node] = spec
        elif lever.kind == "crews":
            args["repair_crews"] = int(args["repair_crews"]) + step
        elif lever.kind == "call_out":
            args["call_out"] = dict(lever.options[step - 1])
    for lever, step in zip(levers, state):
        if lever.kind != "copies":
            continue
        node = lever.node
        feeds = list(rbd.G.predecessors(node))
        exit_node = next(iter(rbd.G.successors(node)))
        for index in range(1, step + 1):
            name = _copy_name(rbd, node, index)
            components[name] = components[node]
            edges += [(u, name) for u in feeds] + [(name, exit_node)]
    args.update(edges=edges, components=components, k=k or None)
    return RepairableRBD(**args)


def _settings(rbd, levers: Sequence[_Lever], state: Sequence[int]) -> dict:
    """A design's levers, as its result gives them."""
    found: dict = {
        "copies": {},
        "intervals": {},
        "inspections": {},
        "upgrades": {},
        "spares": {},
    }
    for lever, step in zip(levers, state):
        if lever.kind == "copies":
            found["copies"][lever.node] = step
        elif lever.kind == "spares":
            assert lever.shelf is not None  # a spares lever has its shelf
            found["spares"][lever.node] = lever.shelf.stock + step
        elif lever.kind == "upgrade":
            found["upgrades"][lever.node] = step or None
        elif lever.kind in ("interval", "inspection"):
            found[lever.kind + "s"][lever.node] = (
                lever.options[step - 1] if step else None
            )
    crews = next((s for v, s in zip(levers, state) if v.kind == "crews"), 0)
    found["crews"] = (
        None if rbd.repair_crews is None else int(rbd.repair_crews) + crews
    )
    contract = next(
        (
            dict(v.options[s - 1])
            for v, s in zip(levers, state)
            if v.kind == "call_out" and s
        ),
        rbd.call_out,
    )
    found["call_out"] = None if contract is None else dict(contract)
    return found


def _evaluate(
    diagrams,
    horizon,
    engine,
    rate,
    downtime,
    measure,
    level,
    confidence,
    objective,
):
    """One design's runs, ``diagrams`` its ``(diagram, runs, seed)``
    (one, or one for each parameter draw, the runs pooled): its risk,
    and its planned and unplanned downtime."""
    downs, unplanneds, spents = [], [], []
    for diagram, n, seed in diagrams:
        run = diagram.availability(
            horizon,
            mc_samples=n,
            seed=seed,
            engine=engine,
            control_variate=False,
            conditional=False,
        )
        down = horizon - np.asarray(run.uptimes, float)
        downs.append(down)
        unplanneds.append(
            down
            if run.unplanned_downtimes is None
            else np.asarray(run.unplanned_downtimes, float)
        )
        spent = 0.0
        if run.cost is not None:
            spent = sum(
                float(run.cost.by_category.get(name, 0.0))
                for name in SPEND_CATEGORIES
            )
        spents.append(spent)
    down = np.concatenate(downs)
    unplanned = np.concatenate(unplanneds)
    planned = np.maximum(down - unplanned, 0.0)
    target = down if downtime == "all" else unplanned
    scale = 1.0 if measure == "hours" else rate
    found = _risk.risk(target * scale, level, objective, confidence)

    def summary(values) -> Dict[str, Optional[float]]:
        ordered = np.sort(values)
        hours = (float(values.mean()), _risk.quantile(ordered, level))
        return {
            "mean_hours": hours[0],
            "bad_hours": hours[1],
            "mean_cost": hours[0] * rate if rate else None,
            "bad_cost": hours[1] * rate if rate else None,
        }

    # Each draw has as many runs: the mean cost is the draws' mean.
    return found, summary(planned), summary(unplanned), float(np.mean(spents))


class _Drawn(NamedTuple):
    """The parameter draws a design is run over: each draw's diagram as
    given (its fitted models drawn), its runs and its seed, the same for
    every design; and what the result says of them."""

    diagrams: List
    runs: int
    seeds: List[int]
    words: str


def _drawn(rbd, uncertainty, n_draws, n, seed) -> Optional[_Drawn]:
    """The diagram over ``n_draws`` plausible models of its fitted
    inputs, as ``expected_npv_uncertainty`` draws them, with ``n`` runs
    shared among the draws; None without ``uncertainty``."""
    from repyability.rbd import _repairable_uncertainty as drawing

    if uncertainty is None or uncertainty is False:
        return None
    draws = checks.whole_number(n_draws, "n_draws")
    if n < draws:
        raise ValueError(
            f"mc_samples ({n}) must be at least n_draws ({draws}): each "
            "draw of the models needs a run of its own."
        )
    inputs, groups = drawing.sources(
        rbd, None if uncertainty is True else uncertainty
    )
    models, group_models = drawing.draws(
        rbd, inputs, groups, draws, seed, "random"
    )
    diagrams = [
        drawing.build(rbd, inputs, models, group_models, i)
        for i in range(draws)
    ]
    seeds = [
        int(v)
        for v in np.random.default_rng(seed).integers(0, 2**31 - 1, size=draws)
    ]
    fixed = drawing.held_fixed(rbd, inputs, groups)
    held = (
        "; held as given: "
        + ", ".join(f"the {h['role']} of {h['node']!r}" for h in fixed)
        if fixed
        else ""
    )
    words = (
        f"Over the models' uncertainty: {draws} draws of the fitted "
        f"models ({drawing.band_of(rbd, inputs, groups)}), "
        f"{n // draws} runs each, pooled{held}. An upgrade's life is "
        "taken as given. Runs that share a draw are not independent, so "
        "the bound is approximate (exact with one run a draw)."
    )
    return _Drawn(diagrams, n // draws, seeds, words)


def least_spend_for_downtime(
    rbd,
    horizon,
    limit,
    *,
    level: float,
    objective: str,
    measure: str,
    downtime: str,
    redundancy: Optional[Mapping],
    intervals: Optional[Mapping],
    inspections: Optional[Mapping],
    upgrades: Optional[Mapping],
    max_units: Optional[Mapping],
    crews: Optional[Mapping],
    call_out,
    budget,
    value_rate: float,
    confidence: float,
    mc_samples: Optional[int],
    seed,
    engine: str,
    uncertainty=None,
    n_draws: int = 100,
    spares: Optional[Mapping] = None,
) -> DowntimeRiskResult:
    """See ``RepairableRBD.least_spend_for_downtime``."""
    from repyability.rbd import _npv_runs

    horizon = checks.number_or_nan(horizon)
    if not (math.isfinite(horizon) and horizon > 0.0):
        raise ValueError(
            f"horizon must be a number more than 0; got {horizon!r}."
        )
    level = _share(level, "level")
    confidence = _share(confidence, "confidence")
    checks.one_of("objective", objective, _risk.MEASURES)
    checks.one_of("measure", measure, ("hours", "cost"))
    checks.one_of("downtime", downtime, ("all", "unplanned"))
    checks.one_of("engine", engine, ("auto", "python", "numba"))
    worth = checks.number_or_nan(value_rate)
    if not (math.isfinite(worth) and worth >= 0.0):
        raise ValueError(
            f"value_rate must be a number, at least 0; got {value_rate!r}."
        )
    rate = float(rbd.downtime_cost_rate) + float(worth)
    if measure == "cost" and rate <= 0.0:
        raise ValueError(
            'measure="cost" needs downtime to cost something: give the '
            "diagram a downtime_cost_rate, or value_rate= (the value lost "
            'per hour down), or measure="hours".'
        )
    bound = checks.number_or_nan(limit)
    if not (math.isfinite(bound) and bound > 0.0):
        raise ValueError(
            "limit must be the downtime a bad run must stay under, a "
            f"number more than 0; got {limit!r}."
        )
    if budget is not None:
        cap = checks.number_or_nan(budget)
        if not (math.isfinite(cap) and cap >= 0.0):
            raise ValueError(
                f"budget must be a number, at least 0; got {budget!r}."
            )
        budget = float(cap)
    n = checks.whole_number(
        MC_SAMPLES if mc_samples is None else mc_samples, "mc_samples"
    )
    seed = _npv_runs._seed(seed)
    levers = _levers(
        rbd,
        redundancy,
        intervals,
        inspections,
        upgrades,
        max_units,
        crews,
        call_out,
        spares,
    )

    drawn = _drawn(rbd, uncertainty, n_draws, n, seed)

    designs: List[DowntimeDesign] = []
    cache: Dict[Tuple[int, ...], DowntimeDesign] = {}
    base_spent: List[float] = []
    used = [engine]
    methods = [""]

    def score(state: Tuple[int, ...]) -> DowntimeDesign:
        found = cache.get(state)
        if found is not None:
            return found
        if drawn is None:
            diagrams = [
                (
                    _shelved(rbd, _design(rbd, levers, state), levers, state),
                    n,
                    seed,
                )
            ]
        else:
            diagrams = [
                (
                    _shelved(
                        rbd, _design(given, levers, state), levers, state
                    ),
                    drawn.runs,
                    s,
                )
                for given, s in zip(drawn.diagrams, drawn.seeds)
            ]
        risk, planned, unplanned, spent = _evaluate(
            diagrams,
            horizon,
            engine,
            rate,
            downtime,
            measure,
            level,
            confidence,
            objective,
        )
        methods[0] = risk.bound_method
        if not base_spent:
            base_spent.append(spent)
        spend = _fixed_spend(levers, state) + (spent - base_spent[0])
        design = DowntimeDesign(
            **_settings(rbd, levers, state),
            spend=float(spend),
            estimate=risk.estimate,
            bound=risk.bound,
            meets=risk.bound <= bound,
            planned=planned,
            unplanned=unplanned,
        )
        cache[state] = design
        return design

    state: Tuple[int, ...] = (0,) * len(levers)
    current = score(state)
    designs.append(current)
    floor = None
    over_budget = False
    for _ in range(MAX_STEPS):
        if current.meets:
            break
        moves = []
        for i, lever in enumerate(levers):
            if lever.options:
                # Any later option: a quantile can stay put over one step.
                for step in range(state[i] + 1, len(lever.options) + 1):
                    moves.append(state[:i] + (step,) + state[i + 1 :])
            elif lever.cap is None or state[i] < lever.cap:
                moves.append(state[:i] + (state[i] + 1,) + state[i + 1 :])
        if not moves:
            floor = "caps"
            break
        best, best_gain = None, 0.0
        over_budget = False
        for move in moves:
            option = score(move)
            drop = current.estimate - option.estimate
            if drop <= 0.0:
                continue
            if budget is not None and option.spend > budget:
                over_budget = True
                continue
            extra = option.spend - current.spend
            gain = math.inf if extra <= 0.0 else drop / extra
            if best is None or gain > best_gain:
                best, best_gain = (move, option), gain
        if best is None:
            floor = "budget" if over_budget else "lowest"
            break
        state, current = best
        designs.append(current)
    else:
        floor = "steps"

    met = [d for d in designs if d.meets]
    answer = min(met, key=lambda d: d.spend) if met else None
    stopped = None
    if answer is None:
        last = designs[-1]
        planned_bad = float(last.planned["bad_hours"] or 0.0) * (
            1.0 if measure == "hours" else rate
        )
        if downtime == "all" and planned_bad >= bound:
            floor = "planned"
        stopped = _floor_words(floor, last, bound, measure, budget)
    return DowntimeRiskResult(
        horizon=float(horizon),
        limit=float(bound),
        level=level,
        objective=objective,
        measure=measure,
        downtime=downtime,
        confidence=confidence,
        bound_method=methods[0],
        budget=budget,
        met=answer is not None,
        design=(
            answer
            if answer is not None
            else min(designs, key=lambda d: d.estimate)
        ),
        frontier=_frontier(list(cache.values())),
        floor=None if answer is not None else floor,
        stopped=stopped,
        n_simulations=n if drawn is None else drawn.runs * len(drawn.seeds),
        seed=seed,
        uncertainty=None if drawn is None else drawn.words,
        engine=used[0],
    )


def _frontier(designs: List[DowntimeDesign]) -> List[DowntimeDesign]:
    """The designs no other beats on both spend and bad-run downtime,
    by spend: more spend never raises the downtime along it."""
    out: List[DowntimeDesign] = []
    for design in sorted(designs, key=lambda d: (d.spend, d.estimate)):
        if not out or design.estimate < out[-1].estimate:
            out.append(design)
    return out


def _floor_words(floor, last, limit, measure, budget) -> str:
    unit = "hours" if measure == "hours" else "in cost"
    lowest = (
        f"The lowest bad-run downtime the levers reached is "
        f"{last.estimate:.6g} {unit} (bound {last.bound:.6g}), against a "
        f"limit of {limit:.6g}"
    )
    if floor == "budget":
        return (
            f"{lowest}: the budget ({budget:.6g}) stopped them; a move "
            "that would lower it further costs more. Raise budget=, or "
            "leave it out."
        )
    if floor == "caps":
        return (
            f"{lowest}: every lever is at its cap (max_units, crews' "
            "or spares' 'max', or the last option given). Raise the caps, "
            "or give "
            "more options."
        )
    if floor == "planned":
        return (
            f"{lowest}: {last.planned['bad_hours']:.6g} hours of a bad "
            "run are planned task downtime, at or over the limit alone; "
            "shorter intervals add to it, and copies with the same tasks "
            "do not remove it. Give tasks that take less time, or "
            'downtime="unplanned".'
        )
    if floor == "steps":
        return f"{lowest}: the search took its most steps ({MAX_STEPS})."
    return (
        f"{lowest}: no lever given lowers it further. Give other levers "
        "(redundancy=, intervals=, inspections=, upgrades=, spares=, "
        "crews=, call_out=)."
    )
