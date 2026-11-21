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
  horizon (from the same runs).

The search is greedy marginal analysis, as the allocations' searches
are: from the diagram as it is, each step takes the move that lowers the
risk estimate most per extra spend, every design run on the same seed
(common random numbers), until a design meets the limit or no move
lowers the risk. Its designs are the frontier of spend against the bad
run's downtime. When the levers stop short, the result says which floor
stopped them: the budget, the levers' caps, or downtime no lever left
lowers (the planned task downtime, when it is planned).

Every design reports both planned and unplanned downtime (each
simulation's, from the engines' own split, ``_unplanned``): their means
and their quantiles at ``level``, in hours and, with a rate, in cost.
"""

import math
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

import numpy as np

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


def _levers(rbd, redundancy, intervals, max_units):
    """The levers, checked: each redundancy node's price and cap, each
    interval node's candidates."""
    from repyability.rbd import _repairable_allocation as allocation
    from repyability.rbd import _requirements

    prices: Dict[Hashable, float] = {}
    caps: Dict[Hashable, Optional[int]] = {}
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
        cost = checks.number_or_nan(price)
        if not (math.isfinite(cost) and cost > 0.0):
            raise ValueError(
                f"The price of a copy of node {node!r} must be a number "
                f"more than 0; got {price!r}."
            )
        prices[node] = float(cost)
        cap = None if max_units is None else max_units.get(node)
        if cap is not None and not (
            checks.is_number(cap) and int(cap) == cap and cap >= 0
        ):
            raise ValueError(
                f"max_units for node {node!r} must be a whole number of "
                f"copies, at least 0; got {cap!r}."
            )
        caps[node] = None if cap is None else int(cap)
    candidates: Dict[Hashable, Tuple[float, ...]] = {}
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
        candidates[node] = tuple(float(v) for v in row)
    return prices, caps, candidates


def _design(rbd, copies: Dict, steps: Dict, candidates: Dict):
    """The diagram with ``copies[node]`` copies of each node and each
    interval node's task at ``candidates[node][steps[node] - 1]`` (as
    given at step 0)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if not any(copies.values()) and not any(steps.values()):
        return rbd
    args = dict(rbd._init_args)
    components = dict(args["components"])
    edges = [tuple(e) for e in args["edges"]]
    k = dict(args.get("k") or {})
    for node, step in steps.items():
        if step:
            spec = dict(components[node])
            spec["preventive"] = dict(
                spec["preventive"], interval=candidates[node][step - 1]
            )
            components[node] = spec
    for node, count in copies.items():
        feeds = list(rbd.G.predecessors(node))
        exit_node = next(iter(rbd.G.successors(node)))
        for index in range(1, count + 1):
            name = _copy_name(rbd, node, index)
            components[name] = components[node]
            edges += [(u, name) for u in feeds] + [(name, exit_node)]
    args.update(edges=edges, components=components, k=k or None)
    return RepairableRBD(**args)


def _evaluate(
    diagram,
    horizon,
    n,
    seed,
    engine,
    rate,
    downtime,
    measure,
    level,
    confidence,
    objective,
):
    """One design's runs: its risk, and its planned and unplanned
    downtime."""
    run = diagram.availability(
        horizon,
        mc_samples=n,
        seed=seed,
        engine=engine,
        control_variate=False,
        conditional=False,
    )
    down = horizon - np.asarray(run.uptimes, float)
    unplanned = (
        down
        if run.unplanned_downtimes is None
        else np.asarray(run.unplanned_downtimes, float)
    )
    planned = np.maximum(down - unplanned, 0.0)
    target = down if downtime == "all" else unplanned
    scale = 1.0 if measure == "hours" else rate
    found = _risk.risk(target * scale, level, objective, confidence)
    preventive = 0.0
    if run.cost is not None:
        preventive = float(run.cost.by_category.get("preventive", 0.0))

    def summary(values) -> Dict[str, Optional[float]]:
        ordered = np.sort(values)
        hours = (float(values.mean()), _risk.quantile(ordered, level))
        return {
            "mean_hours": hours[0],
            "bad_hours": hours[1],
            "mean_cost": hours[0] * rate if rate else None,
            "bad_cost": hours[1] * rate if rate else None,
        }

    return found, summary(planned), summary(unplanned), preventive


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
    max_units: Optional[Mapping],
    budget,
    value_rate: float,
    confidence: float,
    mc_samples: Optional[int],
    seed,
    engine: str,
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
    prices, caps, candidates = _levers(rbd, redundancy, intervals, max_units)

    designs: List[DowntimeDesign] = []
    cache: Dict[tuple, DowntimeDesign] = {}
    base_preventive: List[float] = []
    used = [engine]
    methods = [""]

    def key(copies, steps):
        return tuple(sorted(copies.items(), key=repr)) + tuple(
            sorted(steps.items(), key=repr)
        )

    def score(copies: Dict, steps: Dict) -> DowntimeDesign:
        found = cache.get(key(copies, steps))
        if found is not None:
            return found
        diagram = _design(rbd, copies, steps, candidates)
        risk, planned, unplanned, preventive = _evaluate(
            diagram,
            horizon,
            n,
            seed,
            engine,
            rate,
            downtime,
            measure,
            level,
            confidence,
            objective,
        )
        methods[0] = risk.bound_method
        if not base_preventive:
            base_preventive.append(preventive)
        spend = sum(prices[node] * c for node, c in copies.items()) + (
            preventive - base_preventive[0]
        )
        design = DowntimeDesign(
            copies=dict(copies),
            intervals={
                node: (candidates[node][s - 1] if s else None)
                for node, s in steps.items()
            },
            spend=float(spend),
            estimate=risk.estimate,
            bound=risk.bound,
            meets=risk.bound <= bound,
            planned=planned,
            unplanned=unplanned,
        )
        cache[key(copies, steps)] = design
        return design

    copies = {node: 0 for node in prices}
    steps = {node: 0 for node in candidates}
    current = score(copies, steps)
    designs.append(current)
    floor = None
    over_budget = False
    for _ in range(MAX_STEPS):
        if current.meets:
            break
        moves = []
        for node in prices:
            if caps[node] is None or copies[node] < caps[node]:
                moves.append(({**copies, node: copies[node] + 1}, steps))
        for node in candidates:
            # Any later interval: a quantile can stay put over one step.
            for step in range(steps[node] + 1, len(candidates[node]) + 1):
                moves.append((copies, {**steps, node: step}))
        if not moves:
            floor = "caps"
            break
        best, best_gain = None, 0.0
        over_budget = False
        for move_copies, move_steps in moves:
            option = score(move_copies, move_steps)
            drop = current.estimate - option.estimate
            if drop <= 0.0:
                continue
            if budget is not None and option.spend > budget:
                over_budget = True
                continue
            extra = option.spend - current.spend
            gain = math.inf if extra <= 0.0 else drop / extra
            if best is None or gain > best_gain:
                best, best_gain = (move_copies, move_steps, option), gain
        if best is None:
            floor = "budget" if over_budget else "lowest"
            break
        copies, steps, current = dict(best[0]), dict(best[1]), best[2]
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
        n_simulations=n,
        seed=seed,
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
            f"{lowest}: every lever is at its cap (max_units, or the last "
            "interval given). Raise max_units, or give more intervals."
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
        "(redundancy=, intervals=)."
    )
