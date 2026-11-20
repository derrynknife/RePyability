"""A ``RepairableRBD``'s maintenance intervals: the age-replacement and
proof-test intervals that meet a target at the lowest cost
(``optimal_replacement_intervals``, ``optimal_inspection_intervals``),
chosen from the allowed calendar every combination at a time or, past
``_MAX_COMBINATIONS``, by a local search, and the diagram with given
intervals (``with_intervals``). The methods of ``RepairableRBD`` of those
names call these.
"""

import dataclasses
import itertools
import math
from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Hashable,
    Optional,
    Tuple,
)

import numpy as np
from scipy.optimize import minimize

from repyability.rbd import _crews, _discounted_long_run, _requirements
from repyability.rbd._common import _crew_salaries, _discount_rate
from repyability.rbd._model_utils import (
    failure_time_scale,
)
from repyability.rbd.results import (
    MaintenancePlan,
    PlanRobustness,
)
from repyability.rbd.routes import Refused
from repyability.utils import checks

if TYPE_CHECKING:
    from repyability.rbd.repairable_rbd import RepairableRBD


#: What the interval choices say when a component can wait for a crew
#: (#184): see ``_require_interval_crews``.
_INTERVAL_CREWS = (
    "the exact long-run values the intervals are chosen by assume (the "
    "Markov chain of the repair queue has no place for scheduled "
    "maintenance or tests)",
    "Choose them as if every repair started at once, with "
    "assume_unlimited_crews=True, then simulate the plan with the crews: "
    "with_intervals(plan).cost() or .availability(), or compare() it with "
    "the schedules you have.",
)


def _objective(cost: float, availability: float, max_cost_rate) -> float:
    """What an interval choice minimises: the cost rate, or the
    unavailability when the cost rate is capped."""
    return 1.0 - availability if max_cost_rate is not None else cost


#: What an interval choice can minimise.
OBJECTIVES = ("cost", "eac", "npv")


class _Goal:
    """What an interval choice minimises in place of the long-run cost
    rate (``objective`` ``"eac"`` or ``"npv"``): the equivalent annual
    cost at a discount rate, less the value earned, or less the expected
    net present value over a horizon. A lower ``score`` is better."""

    def __init__(
        self, objective: str, rate: float, value_rate: float, horizon
    ):
        self.objective = objective
        self.rate = rate
        self.value_rate = value_rate
        self.horizon = horizon
        #: Each component's own equivalent annual cost, by its schedule.
        self._own: dict = {}

    def eac(self, plan, cost: float, availability: float) -> float:
        """``plan``'s equivalent annual cost less the value it earns,
        from its long-run ``cost`` rate and ``availability``: at rate 0,
        those themselves."""
        if self.rate == 0.0:
            return cost - self.value_rate * availability
        own = 0.0
        for node in plan.costs:
            key = (
                node,
                plan._preventive.get(node),
                plan._inspection.get(node),
            )
            if key not in self._own:
                self._own[key] = _discounted_long_run.node_eac(
                    plan, node, self.rate
                )
            own += self._own[key]
        return own + _discounted_long_run.system_terms(
            plan, availability, self.value_rate
        )

    def less_npv(self, plan) -> float:
        """Less ``plan``'s expected net present value over the horizon."""
        return -plan.expected_npv(
            self.horizon,
            discount_rate=self.rate,
            value_rate=self.value_rate,
        ).npv

    def achieved(self, score: float, salaries: float = 0.0) -> float:
        """The objective's value at a ``score``: the equivalent annual
        cost, or the net present value; with the repair crews'
        ``salaries`` per unit time, left out of the score, paid."""
        if self.objective == "npv":
            span = (
                -math.expm1(-self.rate * self.horizon) / self.rate
                if self.rate
                else self.horizon
            )
            return -score - salaries * span
        return score + salaries

    def basis(self, rbd) -> str:
        """What the value achieved is made of, and what is approximate
        in it: said on the plan."""
        if self.objective == "npv":
            return (
                "The expected net present value over the horizon, from "
                "new: expected_npv."
            )
        return _discounted_long_run.basis(rbd, self.rate, self.value_rate)


def _goal(rbd, objective, discount_rate, value_rate, horizon):
    """The checked goal of an interval choice (see ``_Goal``); None for
    the long-run cost rate, ``"cost"``."""
    checks.one_of("objective", objective, OBJECTIVES)
    rate = _discount_rate(discount_rate)
    worth = checks.number_or_nan(value_rate)
    if not (math.isfinite(worth) and worth >= 0.0):
        raise ValueError(
            "value_rate must be a finite number, at least 0 (the value "
            f"earned per unit time while the system works); got "
            f"{value_rate!r}."
        )
    if objective == "cost":
        given = [
            name
            for name, value in (
                ("discount_rate", rate),
                ("value_rate", worth),
                ("horizon", horizon),
            )
            if value
        ]
        if given:
            raise ValueError(
                f"{' and '.join(given)} need{'s' if len(given) == 1 else ''}"
                ' an objective that takes them: objective="eac" (the '
                'equivalent annual cost in the long run) or objective="npv"'
                " (the expected net present value over a horizon)."
            )
        return None
    if objective == "npv":
        length = checks.number_or_nan(horizon) if horizon is not None else 0.0
        if not (math.isfinite(length) and length > 0.0):
            raise ValueError(
                'objective="npv" is over a horizon: give horizon, a '
                f"finite time more than 0; got {horizon!r}."
            )
        return _Goal(objective, rate, worth, float(length))
    if horizon is not None:
        raise ValueError(
            'objective="eac" is the long run, with no horizon: leave '
            'horizon out, or give objective="npv" for the net present '
            "value over it."
        )
    if rate:
        _discounted_long_run.require_discounted(rbd)
    return _Goal(objective, rate, worth, None)


def _long_run_goal(rbd, goal: _Goal) -> Optional[_Goal]:
    """The long-run goal nearest to a net present value's: the
    equivalent annual cost at its rates where the diagram has one worked
    out, else the cost rate (None)."""
    try:
        return _goal(rbd, "eac", goal.rate, goal.value_rate, None)
    except NotImplementedError:
        return None


def _scorer(goal: Optional[_Goal], evaluate, planned, over_time=None):
    """What an interval choice minimises for ``goal``, as a function of
    the intervals (None for the cost rate): ``planned(intervals)`` gives
    the diagram with them for the long run, ``over_time(intervals)`` for
    its curves from new (the same by default)."""
    if goal is None:
        return None
    if goal.objective == "npv":
        build = planned if over_time is None else over_time
        return lambda intervals: goal.less_npv(build(intervals))
    return lambda intervals: goal.eac(planned(intervals), *evaluate(intervals))


def _meets(cost, availability, min_availability, max_cost_rate) -> bool:
    """Whether long-run values meet an interval choice's target."""
    if min_availability is not None and availability < min_availability:
        return False
    if max_cost_rate is not None and cost > max_cost_rate:
        return False
    return True


def _choose_intervals(
    nodes: list,
    evaluate,
    starts: list,
    bounds: Tuple[np.ndarray, np.ndarray],
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
    score=None,
    quadrature: bool = False,
) -> dict:
    """The intervals (by node) that minimise the cost rate, or, with a cost
    cap, the unavailability, subject to the target: SLSQP over the
    logarithms of the intervals from each of ``starts``, keeping the best
    that meets the target. ``evaluate(intervals)`` gives their exact
    ``(cost rate, availability)``. With ``score`` (see ``_scorer``), they
    minimise ``score(intervals)`` instead, subject to the same target;
    one worked out by ``quadrature`` (a net present value over a
    horizon, known to about 1e-8 of itself and with a kink wherever a
    replacement falls on the horizon) is searched without its gradient
    (COBYLA), from steps of a tenth of an interval down to 1e-4 of it.
    A ValueError if no intervals in ``bounds`` meet the target, giving
    the best they can do."""
    low, high = bounds
    box = list(zip(low, high))
    options = {"ftol": 1e-10, "maxiter": 200, "eps": 1e-6}
    cache: dict = {}
    scores: dict = {}

    def point(x) -> tuple:
        """``x`` rounded: the intervals evaluated, and returned."""
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    def values(x) -> Tuple[float, float]:
        key = point(x)
        if key not in cache:
            cache[key] = evaluate(
                {node: math.exp(xi) for node, xi in zip(nodes, key)}
            )
        return cache[key]

    # The objective and the constraint, scaled to be about 1.
    cost0, availability0 = values(high)
    down0 = max(1.0 - availability0, 1e-300)
    rate0 = max(abs(cost0), 1e-300)

    def unavailable(x):
        return (1.0 - values(x)[1]) / down0

    def cost(x):
        return values(x)[0] / rate0

    def scored(x) -> float:
        """What is minimised at ``x``, unscaled."""
        if score is None:
            return _objective(*values(x), max_cost_rate)
        key = point(x)
        if key not in scores:
            scores[key] = score(
                {node: math.exp(xi) for node, xi in zip(nodes, key)}
            )
        return scores[key]

    score0 = max(abs(scored(high)), 1e-300) if score is not None else 1.0

    def given(x):
        return scored(x) / score0

    if max_cost_rate is not None:
        objective = unavailable

        def slack(x):
            return (max_cost_rate * (1.0 - 1e-12) - values(x)[0]) / (
                max_cost_rate
            )

    else:
        objective = cost

        def slack(x):
            assert min_availability is not None
            return (values(x)[1] - min_availability - 1e-12) / (
                1.0 - min_availability
            )

    if score is not None:
        objective = given
    constrained = min_availability is not None or max_cost_rate is not None
    if constrained and not any(
        _meets(*values(start), min_availability, max_cost_rate)
        for start in starts
    ):
        # Can the target be met at all? The most available intervals, or
        # the cheapest, from the starts nearest to it first.
        extreme = unavailable if min_availability is not None else cost
        found = []
        for start in sorted(starts, key=extreme):
            reach = minimize(
                extreme, start, method="SLSQP", bounds=box, options=options
            ).x
            found.append(reach)
            if _meets(*values(reach), min_availability, max_cost_rate):
                break
        else:
            best_cost, best_availability = values(
                min(found + list(starts), key=extreme)
            )
            if min_availability is not None:
                raise ValueError(
                    f"min_availability {min_availability} cannot be met: the "
                    f"most any intervals give is {best_availability:.6g}."
                )
            raise ValueError(
                f"max_cost_rate {max_cost_rate} cannot be met: the least any "
                f"intervals cost is {best_cost:.6g} per unit time."
            )
        starts = [reach] + list(starts)
    best, best_value = None, math.inf
    limits = [{"type": "ineq", "fun": slack}] if constrained else []
    for start in starts:
        if quadrature:
            inside = [
                {"type": "ineq", "fun": lambda x, i=i: x[i] - low[i]}
                for i in range(len(nodes))
            ] + [
                {"type": "ineq", "fun": lambda x, i=i: high[i] - x[i]}
                for i in range(len(nodes))
            ]
            found = minimize(
                objective,
                start,
                method="COBYLA",
                constraints=limits + inside,
                options={"rhobeg": 0.1, "tol": 1e-4, "maxiter": 200},
            )
            found.x = np.clip(found.x, low, high)
        else:
            found = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=box,
                constraints=limits,
                options=options,
            )
        for x in (found.x, start):
            rate, availability = values(x)
            if not _meets(rate, availability, min_availability, max_cost_rate):
                continue
            value = scored(x)
            if value < best_value:
                best, best_value = np.asarray(x, dtype=float), value
    assert best is not None  # the start that meets the target is kept
    return {node: math.exp(xi) for node, xi in zip(nodes, point(best))}


#: Allowed intervals are chosen by trying every combination, up to this many;
#: by a local search beyond.
_MAX_COMBINATIONS = 2000


#: Plans whose costs are within this share of each other, or whose
#: availabilities are within this of each other, are as good: which is
#: chosen does not hang on the last bits of their arithmetic (#229), as it
#: did for identical members' intervals in turn, and of plans that cost the
#: same the most available is chosen however their costs' last bits fall.
_AS_GOOD = 1e-10


def _lower(a: tuple, b: tuple) -> bool:
    """Whether merit ``a`` is lower than merit ``b``: their ``(value,
    size)`` pairs compared in turn, values within ``_AS_GOOD`` of their
    size taken as equal."""
    for (x, size_x), (y, size_y) in zip(a, b):
        if x == y:
            continue
        if abs(x - y) > _AS_GOOD * max(size_x, size_y) or math.isinf(x - y):
            return x < y
    return False


def _divides(interval: float, full: float) -> bool:
    """Whether ``interval`` divides ``full`` a whole number of times."""
    count = full / interval
    return round(count) >= 1 and abs(count - round(count)) <= 1e-9 * count


def _require_chosen(per_node, chosen: list, name: str) -> None:
    """Raise if a dict of per-node options names a node not chosen
    (#222): a typo would otherwise be dropped."""
    if isinstance(per_node, dict):
        others = [node for node in per_node if node not in chosen]
        if others:
            raise ValueError(
                f"{name} names {others}, which are not among the components "
                f"whose intervals are chosen: {chosen}."
            )


def _choose_divisor(
    node,
    full: float,
    bounds: Tuple[float, float],
    evaluate,
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
    score=None,
) -> dict:
    """The interval of a component whose tests can miss a failure (#221),
    from those that divide its full tests' interval ``full`` (``full /
    k``) within ``bounds``, as ``_choose_from`` would choose from them all:
    from some 50 spread evenly in their logarithms, then every one between
    the best's neighbours among those."""
    low, high = bounds
    first = max(1, math.ceil(full / high))
    last = max(first, math.floor(full / low))

    def best_of(ks) -> int:
        options = {node: tuple(full / k for k in ks)}
        chosen = _choose_from(
            [node], options, evaluate, min_availability, max_cost_rate, score
        )
        return int(round(full / chosen[node]))

    if last - first < 64:
        return {node: full / best_of(range(first, last + 1))}
    spread = sorted({int(k) for k in np.round(np.geomspace(first, last, 50))})
    k = best_of(spread)
    i = spread.index(k)
    around = range(
        spread[max(i - 1, 0)], spread[min(i + 1, len(spread) - 1)] + 1
    )
    return {node: full / best_of(around)}


def _choose_from(
    nodes: list,
    options: dict,
    evaluate,
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
    score=None,
) -> dict:
    """The intervals, each from its node's ``options``, that minimise the
    cost rate, or with a cost cap the unavailability, subject to the target:
    every combination when there are at most 2000, a local search (one
    interval changed at a time, from several starts) otherwise. Of plans
    that cost the same, the most available is chosen (and of those as
    available, the cheapest): tests' offsets change no cost. Of plans as
    good (``_AS_GOOD``), the first tried is kept. With ``score`` (see
    ``_scorer``), the lowest ``score(intervals)`` is chosen first, of the
    plans that meet the target. A ValueError if none meets the target,
    giving the best any does."""

    def merit(intervals: dict) -> tuple:
        """(How far from the target, the objective, the other one), each
        with the size its differences are measured against: lower is
        better."""
        cost, availability = evaluate(intervals)
        priced = abs(cost)
        if min_availability is not None:
            short = (max(0.0, min_availability - availability), 1.0)
        elif max_cost_rate is not None:
            short = (max(0.0, cost - max_cost_rate), priced)
        else:
            short = (0.0, 1.0)
        down = (1.0 - availability, 1.0)
        if score is not None:
            scored = score(intervals)
            return short, (scored, abs(scored)), (cost, priced), down
        if max_cost_rate is None:
            return short, (cost, priced), down
        return short, down, (cost, priced)

    count = math.prod(len(options[node]) for node in nodes)
    best: Optional[dict] = None
    best_value: tuple = ()
    if count <= _MAX_COMBINATIONS:
        for combination in itertools.product(
            *(options[node] for node in nodes)
        ):
            candidate = dict(zip(nodes, combination))
            candidate_value = merit(candidate)
            if best is None or _lower(candidate_value, best_value):
                best, best_value = candidate, candidate_value
    else:
        starts = [
            {node: options[node][0] for node in nodes},
            {node: options[node][-1] for node in nodes},
            {node: options[node][len(options[node]) // 2] for node in nodes},
        ]
        for current in starts:
            value = merit(current)
            improved = True
            while improved:
                improved = False
                for node in nodes:
                    for option in options[node]:
                        candidate = {**current, node: option}
                        candidate_value = merit(candidate)
                        if _lower(candidate_value, value):
                            current, value = candidate, candidate_value
                            improved = True
            if best is None or _lower(value, best_value):
                best, best_value = current, value
    assert best is not None
    cost, availability = evaluate(best)
    if not _meets(cost, availability, min_availability, max_cost_rate):
        if min_availability is not None:
            raise ValueError(
                f"min_availability {min_availability} cannot be met: the "
                f"most the allowed intervals give is {availability:.6g}."
            )
        raise ValueError(
            f"max_cost_rate {max_cost_rate} cannot be met: the least the "
            f"allowed intervals cost is {cost:.6g} per unit time."
        )
    return best


def _per_given(rbd, node) -> float:
    """How much of the unit a component's task is counted in passes in
    a unit of the diagram's clock: its duty for a task counted in
    operating time, else 1. The interval choices take and give its
    intervals in that unit, as given, and work on the calendar inside."""
    return getattr(rbd, "_operating_tasks", {}).get(node, 1.0)


def _with_intervals(
    rbd, preventive=None, inspection=None, offsets=None
) -> "RepairableRBD":
    """This RBD with the intervals of some of its preventive or
    inspection schedules changed, for the exact long-run values: a
    shallow copy, sharing everything else, including the renewal cycles
    already worked out (kept by interval). ``offsets`` gives some
    inspected components' first tests as shares of their intervals."""
    # The renewal cycles, kept by interval, are shared with the plan;
    # what its own schedules decide is worked out again.
    for name in ("age_cycles", "block_cycles", "tested_units"):
        rbd._cache.kept(name)
    plan = rbd._shallow_copy("ccf_tables")
    if preventive:
        plan._preventive = dict(rbd._preventive)
        for node, interval in preventive.items():
            plan._preventive[node] = rbd._preventive[node]._replace(
                interval=float(interval) / _per_given(rbd, node)
            )
    if inspection:
        plan._inspection = dict(rbd._inspection)
        for node, interval in inspection.items():
            schedule = rbd._inspection[node]
            # An offset keeps its share of the interval (a pair tested
            # half an interval apart stays so).
            share = schedule.offset / schedule.interval
            interval = float(interval) / _per_given(rbd, node)
            plan._inspection[node] = schedule._replace(
                interval=interval, offset=share * interval
            )
    if offsets:
        plan._inspection = dict(plan._inspection)
        for node, share in offsets.items():
            schedule = plan._inspection[node]
            plan._inspection[node] = schedule._replace(
                offset=float(share) * schedule.interval
            )
    return plan


def _interval_targets(rbd, min_availability, max_cost_rate):
    """The checked targets of an interval choice."""
    if min_availability is not None and max_cost_rate is not None:
        raise ValueError(
            "Give at most one of min_availability and max_cost_rate."
        )
    if not rbd.has_costs:
        raise Refused(
            "Nothing is priced, so no interval costs more than another: "
            "give the components costs (or a downtime_cost_rate)."
        )
    if min_availability is not None:
        if not 0.0 < float(min_availability) < 1.0:
            raise ValueError(
                "min_availability must be a number in (0, 1), got "
                f"{min_availability!r}."
            )
        return float(min_availability), None
    if max_cost_rate is not None:
        if not 0.0 < float(max_cost_rate) < math.inf:
            raise ValueError(
                "max_cost_rate must be a positive number, got "
                f"{max_cost_rate!r}."
            )
        return None, float(max_cost_rate)
    return None, None


def optimal_replacement_intervals(
    rbd,
    nodes: Optional[Collection[Hashable]],
    *,
    allowed,
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
    assume_unlimited_crews: bool,
    objective: str,
    discount_rate: float,
    value_rate: float,
    horizon,
    robust=False,
    uncertainty=None,
    n_draws: int = 200,
    seed=None,
) -> MaintenancePlan:
    """See ``RepairableRBD.optimal_replacement_intervals``."""
    robust = _robust_method(robust)
    chosen = _maintained(rbd, nodes)
    min_availability, max_cost_rate = _interval_targets(
        rbd, min_availability, max_cost_rate
    )
    if rbd._crews_couple():
        if not assume_unlimited_crews:
            _require_interval_crews(rbd)
        salaries = _crew_salaries(rbd)
        plan = _unlimited_crews(rbd).optimal_replacement_intervals(
            chosen,
            allowed=allowed,
            min_availability=min_availability,
            max_cost_rate=_less_salaries(max_cost_rate, salaries),
            objective=objective,
            discount_rate=discount_rate,
            value_rate=value_rate,
            horizon=horizon,
            robust=robust,
            uncertainty=uncertainty,
            n_draws=n_draws,
            seed=seed,
        )
        return _with_salaries(plan, salaries)
    goal = _goal(rbd, objective, discount_rate, value_rate, horizon)
    planned, evaluate, score = _replacement_scoring(rbd, goal)
    if allowed is not None:
        best = _choose_from(
            chosen,
            _allowed_intervals(allowed, chosen, never=True),
            evaluate,
            min_availability,
            max_cost_rate,
            score,
        )
    else:
        near = None
        if goal is not None and goal.objective == "npv":
            # A net present value takes its curves over the horizon at
            # each plan: searched from the long run's best plan.
            near = _searched_replacements(
                rbd,
                chosen,
                evaluate,
                min_availability,
                max_cost_rate,
                _scorer(_long_run_goal(rbd, goal), evaluate, planned),
            )
        best = _searched_replacements(
            rbd,
            chosen,
            evaluate,
            min_availability,
            max_cost_rate,
            score,
            near,
        )
    cost, availability = evaluate(best)
    plan = _planned(
        rbd,
        {node: float(best[node]) for node in chosen},
        cost,
        availability,
        goal,
        score,
        best,
    )
    if not robust:
        return plan
    options = (
        None
        if allowed is None
        else _allowed_intervals(allowed, chosen, never=True)
    )

    def again(diagram) -> dict:
        """The draw's own best intervals."""
        return dict(
            optimal_replacement_intervals(
                diagram,
                chosen,
                allowed=allowed,
                min_availability=min_availability,
                max_cost_rate=max_cost_rate,
                assume_unlimited_crews=assume_unlimited_crews,
                objective=objective,
                discount_rate=discount_rate,
                value_rate=value_rate,
                horizon=horizon,
            ).intervals
        )

    return dataclasses.replace(
        plan,
        robustness=_robustness(
            rbd,
            {node: float(best[node]) for node in chosen},
            options,
            (objective, discount_rate, value_rate, horizon),
            (min_availability, max_cost_rate),
            robust,
            again,
            uncertainty,
            n_draws,
            seed,
        ),
    )


def _replacement_scoring(rbd, goal):
    """For a choice of age-replacement intervals on ``rbd``: the plan
    with given intervals (for the long run), its long-run cost rate and
    availability, and the score of ``goal`` (None for the cost rate)."""

    def planned(intervals: dict) -> "RepairableRBD":
        return _with_intervals(rbd, preventive=intervals)

    def over_time(intervals: dict) -> "RepairableRBD":
        """The plan for its curves from new: a component never replaced
        has no schedule there."""
        never = [node for node in intervals if math.isinf(intervals[node])]
        if not never:
            return planned(intervals)
        return _with_intervals(
            with_intervals(rbd, dict.fromkeys(never, math.inf), None),
            preventive={
                node: interval
                for node, interval in intervals.items()
                if node not in never
            },
        )

    def evaluate(intervals: dict) -> Tuple[float, float]:
        plan = planned(intervals)
        return plan.expected_cost_rate(), plan.mean_availability()

    return planned, evaluate, _scorer(goal, evaluate, planned, over_time)


#: How robust a choice is, by ``robust``'s values (see ``_robustness``).
ROBUST = {
    True: "neighbours",
    "neighbours": "neighbours",
    "reoptimise": "reoptimise",
}

#: How far a neighbour of a continuously searched interval is: this many
#: times longer, or shorter.
NEIGHBOUR_STEP = 1.25


def _robust_method(robust):
    """``robust`` checked: False, or the method's name."""
    if robust is False or robust is None:
        return False
    if isinstance(robust, (bool, str)) and robust in ROBUST:
        return ROBUST[robust]
    raise ValueError(
        "robust must be False, True (the choice against its neighbours "
        'under each draw) or "reoptimise" (against each draw\'s own best '
        f"intervals, slower); got {robust!r}."
    )


def _neighbours(best: dict, options: Optional[dict]) -> list:
    """The choice and its neighbours: each interval one step either way,
    the next of its ``options``, or ``NEIGHBOUR_STEP`` times longer or
    shorter (an interval of never has none)."""
    out = [dict(best)]
    for node, interval in best.items():
        if options is not None:
            values = options[node]
            k = values.index(interval)
            moved = [values[j] for j in (k - 1, k + 1) if 0 <= j < len(values)]
        elif math.isfinite(interval):
            moved = [interval / NEIGHBOUR_STEP, interval * NEIGHBOUR_STEP]
        else:
            moved = []
        out.extend({**best, node: value} for value in moved)
    return out


def _robustness(
    rbd,
    best: dict,
    options: Optional[dict],
    objective: tuple,
    targets: tuple,
    method: str,
    again,
    uncertainty,
    n_draws: int,
    seed,
) -> PlanRobustness:
    """How the choice ``best`` fares under draws of the uncertain models
    (see ``PlanRobustness``): every candidate scored under each draw's
    diagram by the choice's own objective and targets, with the same
    draws for every candidate; or, with ``"reoptimise"``, against each
    draw's own best intervals (``again(diagram)``)."""
    from repyability.rbd import _repairable_uncertainty as drawn

    min_availability, max_cost_rate = targets
    inputs, groups = drawn.sources(rbd, uncertainty)
    models, group_models = drawn.draws(rbd, inputs, groups, n_draws, seed)
    candidates = (
        [best] if method == "reoptimise" else _neighbours(best, options)
    )
    choices: Optional[list] = [] if method == "reoptimise" else None
    scores = np.empty((n_draws, len(candidates)))
    meets = np.empty((n_draws, len(candidates)), dtype=bool)
    regret = np.empty(n_draws)
    for i in range(n_draws):
        diagram = drawn.build(rbd, inputs, models, group_models, i)
        _, evaluate, score = _replacement_scoring(
            diagram, _goal(diagram, *objective)
        )

        def scored(intervals: dict) -> Tuple[float, bool]:
            cost, availability = evaluate(intervals)
            ok = _meets(cost, availability, min_availability, max_cost_rate)
            if score is not None:
                return score(intervals), ok
            return _objective(cost, availability, max_cost_rate), ok

        for k, candidate in enumerate(candidates):
            scores[i, k], meets[i, k] = scored(candidate)
        if choices is not None:
            try:
                own = again(diagram)
            except ValueError:
                # No intervals meet the target under this draw.
                own = None
            choices.append(own)
            regret[i] = (
                np.nan if own is None else scores[i, 0] - scored(own)[0]
            )
        else:
            feasible = np.where(meets[i], scores[i], np.inf)
            low = float(feasible.min())
            regret[i] = scores[i, 0] - low if math.isfinite(low) else np.nan
    if choices is not None:
        share_best = (
            None
            if options is None
            else float(np.mean([own == best for own in choices]))
        )
        by_candidate: list = []
    else:
        feasible = np.where(meets, scores, np.inf)
        winner = np.argmin(feasible, axis=1)
        has = np.isfinite(feasible.min(axis=1))
        by_candidate = [
            float(np.mean(has & (winner == k))) for k in range(len(candidates))
        ]
        # The choice is best where nothing that meets the target scores
        # lower: ties count for it.
        share_best = float(
            np.mean(meets[:, 0] & (scores[:, 0] <= feasible.min(axis=1)))
        )
    known = regret[np.isfinite(regret)]
    return PlanRobustness(
        method=method,
        candidates=candidates,
        scores=scores,
        meets_target=meets,
        regret=regret,
        share_best=share_best,
        share_best_by_candidate=by_candidate,
        share_meets_target=float(np.mean(meets[:, 0])),
        expected_regret=float(known.mean()) if known.size else float("nan"),
        draw_choices=choices,
        n_draws=n_draws,
        band=drawn.band_of(rbd, inputs, groups),
        held_fixed=drawn.held_fixed(rbd, inputs, groups),
    )


def _planned(
    rbd, intervals, cost, availability, goal, score, best, offsets=None
) -> MaintenancePlan:
    """The plan of an interval choice, with what its ``goal`` (if any)
    comes to at the choice ``best``."""
    if goal is None:
        return MaintenancePlan(intervals, cost, availability, offsets=offsets)
    return MaintenancePlan(
        intervals,
        cost,
        availability,
        offsets=offsets,
        objective=goal.objective,
        discount_rate=goal.rate,
        value_rate=goal.value_rate,
        horizon=goal.horizon,
        objective_value=goal.achieved(score(best)),
        basis=goal.basis(rbd),
    )


def _searched_replacements(
    rbd,
    chosen: list,
    evaluate: Callable[[dict], Tuple[float, float]],
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
    score=None,
    near: Optional[dict] = None,
) -> dict:
    """The age-replacement intervals of ``chosen`` that
    ``optimal_replacement_intervals`` gives without ``allowed``: by a
    gradient search over their logarithms, each from a thousandth to a
    thousand times its component's mean life, one at the top of its
    range compared with never replacing it. With ``near`` (intervals),
    the search is from those alone, for a ``score`` worked out by
    quadrature."""
    scale = {}
    for node in chosen:
        life = failure_time_scale(rbd.components[node].reliability)
        scale[node] = life if np.isfinite(life) and life > 0.0 else 1.0
        # In the unit its interval is given in.
        scale[node] *= _per_given(rbd, node)
    low = np.array([math.log(1e-3 * scale[n]) for n in chosen])
    high = np.array([math.log(1e3 * scale[n]) for n in chosen])
    current = [
        math.log(
            min(
                rbd._preventive[n].interval * _per_given(rbd, n),
                1e3 * scale[n],
            )
        )
        for n in chosen
    ]
    starts = [
        np.clip(current, low, high),
        np.array([math.log(scale[n]) for n in chosen]),
        np.array([math.log(0.3 * scale[n]) for n in chosen]),
        high.copy(),
    ]
    if near is not None:
        starts = [
            np.clip(
                [
                    math.log(near[n]) if math.isfinite(near[n]) else h
                    for n, h in zip(chosen, high)
                ],
                low,
                high,
            )
        ]
    best = _choose_intervals(
        chosen,
        evaluate,
        starts,
        (low, high),
        min_availability,
        max_cost_rate,
        score,
        quadrature=near is not None,
    )

    def scored(intervals: dict) -> float:
        if score is not None:
            return score(intervals)
        return _objective(*evaluate(intervals), max_cost_rate)

    # An interval at the top of its range may be better never used.
    for i, node in enumerate(chosen):
        if best[node] >= 0.99 * math.exp(high[i]):
            never = {**best, node: math.inf}
            cost, availability = evaluate(never)
            if _meets(
                cost, availability, min_availability, max_cost_rate
            ) and scored(never) <= scored(best):
                best = never
    return best


def optimal_inspection_intervals(
    rbd,
    nodes: Optional[Collection[Hashable]],
    *,
    allowed,
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
    assume_unlimited_crews: bool,
    offset_shares,
    objective: str,
    discount_rate: float,
    value_rate: float,
    horizon,
) -> MaintenancePlan:
    """See ``RepairableRBD.optimal_inspection_intervals``."""
    chosen = _inspected(rbd, nodes)
    min_availability, max_cost_rate = _interval_targets(
        rbd, min_availability, max_cost_rate
    )
    if rbd._crews_couple():
        if not assume_unlimited_crews:
            _require_interval_crews(rbd)
        salaries = _crew_salaries(rbd)
        plan = _unlimited_crews(rbd).optimal_inspection_intervals(
            chosen,
            allowed=allowed,
            min_availability=min_availability,
            max_cost_rate=_less_salaries(max_cost_rate, salaries),
            offset_shares=offset_shares,
            objective=objective,
            discount_rate=discount_rate,
            value_rate=value_rate,
            horizon=horizon,
        )
        return _with_salaries(plan, salaries)
    goal = _goal(rbd, objective, discount_rate, value_rate, horizon)
    # Each failure rate per unit its interval is given in.
    rates = {
        node: _requirements._tested_scale(rbd, node) / _per_given(rbd, node)
        for node in chosen
    }

    def planned(intervals: dict) -> "RepairableRBD":
        return _with_intervals(rbd, inspection=intervals)

    def evaluate(intervals: dict) -> Tuple[float, float]:
        plan = planned(intervals)
        return plan.expected_cost_rate(), plan.mean_availability()

    score = _scorer(goal, evaluate, planned)
    if offset_shares is not None:
        if allowed is None:
            raise ValueError(
                "offset_shares are chosen with intervals from allowed: "
                "give allowed (when a lone component is tested changes "
                "nothing in the long run)."
            )
        return _choose_tests(
            rbd,
            chosen,
            _dividing(rbd, _allowed_intervals(allowed, chosen)),
            _allowed_shares(offset_shares, chosen),
            min_availability,
            max_cost_rate,
            goal,
        )
    if allowed is None and rbd._inspection[chosen[0]].partial:
        _requirements._require_one_inspected(rbd)
        (node,) = chosen
        rate = rates[node]
        best = _choose_divisor(
            node,
            # Its full tests' interval (see _Inspection.period).
            rbd._inspection[node].period * _per_given(rbd, node),
            (1e-4 / rate, 10.0 / rate),
            evaluate,
            min_availability,
            max_cost_rate,
            score,
        )
    elif allowed is None:
        _requirements._require_one_inspected(rbd)
        (node,) = chosen
        rate = rates[node]
        low = np.array([math.log(1e-4 / rate)])
        high = np.array([math.log(10.0 / rate)])
        current = math.log(
            rbd._inspection[node].interval * _per_given(rbd, node)
        )
        starts = [
            np.clip([current], low, high),
            np.array([math.log(0.01 / rate)]),
            np.array([math.log(0.1 / rate)]),
            np.array([math.log(1.0 / rate)]),
        ]
        searched = partial(
            _choose_intervals,
            chosen,
            evaluate,
            bounds=(low, high),
            min_availability=min_availability,
            max_cost_rate=max_cost_rate,
        )
        if goal is not None and goal.objective == "npv":
            # A net present value takes its curves over the horizon at
            # each interval: searched from the long run's best.
            near = searched(
                starts,
                score=_scorer(_long_run_goal(rbd, goal), evaluate, planned),
            )
            best = searched(
                [np.array([math.log(near[node])])],
                score=score,
                quadrature=True,
            )
        else:
            best = searched(starts, score=score)
    else:
        best = _choose_from(
            chosen,
            _dividing(rbd, _allowed_intervals(allowed, chosen)),
            evaluate,
            min_availability,
            max_cost_rate,
            score,
        )
    cost, availability = evaluate(best)
    # The offsets the plan's tests keep: each its share of the interval.
    offsets_kept = {
        node: float(best[node])
        * rbd._inspection[node].offset
        / rbd._inspection[node].interval
        for node in chosen
    }
    return _planned(
        rbd,
        {node: float(best[node]) for node in chosen},
        cost,
        availability,
        goal,
        score,
        best,
        offsets=offsets_kept,
    )


def _dividing(rbd, options: dict) -> dict:
    """``options``, the intervals allowed each chosen component,
    checked to divide its full tests' interval where its tests can
    miss a failure (#221): every full test is a whole number of tests
    on, so an interval that does not divide it cannot be its."""
    for node, intervals in options.items():
        schedule = rbd._inspection[node]
        if not schedule.partial:
            continue
        full = schedule.period * _per_given(rbd, node)
        bad = [i for i in intervals if not _divides(i, full)]
        if bad:
            some = ", ".join(f"{full / k:g}" for k in range(1, 5))
            raise ValueError(
                f"The tests of component {node!r} can miss a failure, so "
                f"its full tests' interval, {full:g}, must be a whole "
                "number of its test intervals: "
                f"{', '.join(map(repr, bad))} "
                f"do{'es' if len(bad) == 1 else ''} not divide it. Allow "
                f"intervals that do ({some}, ...)."
            )
    return options


def _require_interval_crews(rbd) -> None:
    """Raise if a component can wait for a repair crew, for the interval
    choices (see ``_INTERVAL_CREWS``)."""
    _crews._require_unlimited_crews(rbd, *_INTERVAL_CREWS)


def _unlimited_crews(rbd) -> "RepairableRBD":
    """This RBD with as many repair crews as jobs, built as it was, but
    with no crews' salaries (as many crews as are needed are not paid:
    see ``_with_salaries``)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return RepairableRBD(
        **{
            **rbd._init_args,
            "repair_crews": None,
            "crew_cost_rate": 0.0,
            "call_out": None,
        }
    )


def _less_salaries(max_cost_rate, salaries: float):
    """A ``max_cost_rate`` for the plan without the crews' ``salaries``
    (see ``_unlimited_crews``), which every plan pays."""
    if max_cost_rate is None or not salaries:
        return max_cost_rate
    if max_cost_rate <= salaries:
        raise ValueError(
            f"max_cost_rate {max_cost_rate} cannot be met: the repair "
            f"crews' salaries alone cost {salaries:.6g} per unit time."
        )
    return max_cost_rate - salaries


def _with_salaries(plan: MaintenancePlan, salaries: float) -> MaintenancePlan:
    """``plan``, chosen without the crews' ``salaries``, with them."""
    if not salaries:
        return plan
    achieved = plan.objective_value
    if achieved is not None:
        # Its score again, with the salaries paid.
        goal = _Goal(
            plan.objective, plan.discount_rate, plan.value_rate, plan.horizon
        )
        achieved = goal.achieved(
            -achieved if plan.objective == "npv" else achieved, salaries
        )
    return dataclasses.replace(
        plan, cost_rate=plan.cost_rate + salaries, objective_value=achieved
    )


def with_intervals(rbd, intervals, offsets) -> "RepairableRBD":
    """See ``RepairableRBD.with_intervals``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if isinstance(intervals, MaintenancePlan):
        if offsets is None:
            offsets = intervals.offsets
        intervals = intervals.intervals
    offsets = dict(offsets or {})
    components = dict(rbd._init_args["components"])
    for node in list(dict(intervals)) + list(offsets):
        _requirements._require_component(rbd, node, "with_intervals")
    for node, interval in dict(intervals).items():
        spec = components.get(node)
        # A component has one schedule at most (see the constructor).
        kinds = [
            kind
            for kind in ("preventive", "inspection")
            if isinstance(spec, dict) and spec.get(kind) is not None
        ]
        if not kinds:
            raise ValueError(
                f"Node {node!r} has no maintenance or test schedule to "
                "give an interval."
            )
        (kind,) = kinds
        assert isinstance(spec, dict)
        if node in offsets and kind != "inspection":
            raise ValueError(
                f"Node {node!r} has an offset but no test schedule: an "
                "offset is the time of a component's first test."
            )
        old = dict(spec[kind])
        interval = float(interval)
        new = {**old, "interval": interval}
        if kind == "inspection":
            if node in offsets:
                new["offset"] = float(offsets[node])
            elif old.get("offset"):
                share = float(old["offset"]) / float(old["interval"])
                new["offset"] = share * interval
        spec = {**spec, kind: new}
        if kind == "preventive" and math.isinf(interval):
            if old.get("policy", "age") != "age":
                raise ValueError(
                    f"Node {node!r}: only age replacement can be left to "
                    "failures (an interval of inf)."
                )
            spec = {k: v for k, v in spec.items() if k != "preventive"}
        components[node] = spec
    unknown = set(offsets) - set(dict(intervals))
    for node in unknown:
        spec = components.get(node)
        if not (isinstance(spec, dict) and spec.get("inspection")):
            raise ValueError(
                f"Node {node!r} has an offset but no test schedule."
            )
        components[node] = {
            **spec,
            "inspection": {
                **spec["inspection"],
                "offset": float(offsets[node]),
            },
        }
    return RepairableRBD(**{**rbd._init_args, "components": components})


def _choose_tests(
    rbd,
    chosen: list,
    intervals: dict,
    shares: dict,
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
    goal: Optional[_Goal] = None,
) -> MaintenancePlan:
    """``optimal_inspection_intervals`` with the offsets chosen too: each
    node's interval from ``intervals`` and its first test's share of it
    from ``shares``, searched together as ``_choose_from`` searches
    intervals."""
    if set(chosen) == set(rbd._inspection):
        # Shifting every test by one time changes nothing in the long
        # run: the first node's tests are kept from 0.
        shares = {**shares, chosen[0]: (0.0,)}
    options = {("interval", n): intervals[n] for n in chosen}
    options.update({("offset", n): shares[n] for n in chosen})

    def planned(choice: dict) -> "RepairableRBD":
        return _with_intervals(
            rbd,
            inspection={n: choice[("interval", n)] for n in chosen},
            offsets={n: choice[("offset", n)] for n in chosen},
        )

    def evaluate(choice: dict) -> Tuple[float, float]:
        plan = planned(choice)
        return plan.expected_cost_rate(), plan.mean_availability()

    score = _scorer(goal, evaluate, planned)
    best = _choose_from(
        list(options),
        options,
        evaluate,
        min_availability,
        max_cost_rate,
        score,
    )
    cost, availability = evaluate(best)
    return _planned(
        rbd,
        {n: float(best[("interval", n)]) for n in chosen},
        cost,
        availability,
        goal,
        score,
        best,
        offsets={
            n: float(best[("offset", n)] * best[("interval", n)])
            for n in chosen
        },
    )


def _allowed_shares(offsets, chosen: list) -> dict:
    """``offsets`` (see ``optimal_inspection_intervals``) as a sorted
    tuple of shares of the interval per chosen node."""
    if isinstance(offsets, str):
        if offsets != "stagger":
            raise ValueError(
                "offset_shares must be shares of the interval in [0, "
                "1), a dict of them per node, or 'stagger', got "
                f"{offsets!r}."
            )
        n = len(chosen)
        return {node: tuple(k / n for k in range(n)) for node in chosen}
    per_node = (
        offsets
        if isinstance(offsets, dict)
        else {node: offsets for node in chosen}
    )
    _require_chosen(per_node, chosen, "offset_shares")
    out = {}
    for node in chosen:
        if node not in per_node:
            raise ValueError(
                f"offset_shares gives no share of the interval for node "
                f"{node!r}: give one, in [0, 1), for each of {chosen}."
            )
        given = per_node[node]
        if isinstance(given, (str, bytes)) or not isinstance(
            given, Collection
        ):
            given = [given]
        values = []
        for value in given:
            number = isinstance(
                value, (int, float, np.integer, np.floating)
            ) and not isinstance(value, bool)
            share = float(value) if number else float("nan")
            if not 0.0 <= share < 1.0:
                raise ValueError(
                    "offset_shares are shares of the interval, in [0, "
                    f"1), got {value!r} for node {node!r}."
                )
            values.append(share)
        if not values:
            raise ValueError(
                f"offset_shares gives no share of the interval for node "
                f"{node!r}."
            )
        out[node] = tuple(sorted(set(values)))
    return out


def _inspected(rbd, nodes) -> list:
    """The components with hidden failures named by ``nodes`` (all of
    them by default), checked."""
    if nodes is None:
        if not rbd._inspection:
            raise Refused(
                "No component has hidden failures: give the components "
                "an 'inspection' schedule to have its interval chosen."
            )
        chosen = list(rbd._inspection)
    else:
        chosen = list(nodes)
        if not chosen:
            raise ValueError("nodes is empty.")
        for node in chosen:
            _requirements._require_component(rbd, node, "nodes")
            if node not in rbd._inspection:
                raise ValueError(
                    f"Node {node!r} has no hidden failures: give it an "
                    "'inspection' schedule to have its interval chosen."
                )
    return chosen


def _allowed_intervals(allowed, chosen: list, never: bool = False) -> dict:
    """``allowed`` as a sorted tuple of intervals per chosen node; with
    ``never``, ``inf`` among them (never replacing a component)."""
    per_node = (
        allowed
        if isinstance(allowed, dict)
        else {node: allowed for node in chosen}
    )
    _require_chosen(per_node, chosen, "allowed")
    options = {}
    for node in chosen:
        if node not in per_node:
            raise ValueError(f"allowed gives no intervals for node {node!r}.")
        values = []
        for value in per_node[node]:
            try:
                interval = float(value)
            except (TypeError, ValueError):
                interval = float("nan")
            if not (0.0 < interval < math.inf or (never and interval > 0.0)):
                kind = (
                    "positive numbers (inf for never)"
                    if never
                    else "positive, finite numbers"
                )
                raise ValueError(
                    f"allowed intervals must be {kind}, got {value!r} "
                    f"for node {node!r}."
                )
            values.append(interval)
        if not values:
            raise ValueError(f"allowed gives no intervals for node {node!r}.")
        options[node] = tuple(sorted(set(values)))
    return options


def _maintained(rbd, nodes) -> list:
    """The components under age replacement named by ``nodes`` (all of
    them by default), checked."""
    if nodes is None:
        chosen = [
            node
            for node, schedule in rbd._preventive.items()
            if schedule.policy == "age"
        ]
        if not chosen:
            raise Refused(
                "No component is under age replacement: give the "
                "components a 'preventive' schedule to have its "
                "interval chosen."
            )
        return chosen
    chosen = list(nodes)
    if not chosen:
        raise ValueError("nodes is empty.")
    for node in chosen:
        schedule = rbd._preventive.get(node)
        if schedule is None or schedule.policy != "age":
            raise ValueError(
                f"Node {node!r} is not a component under age "
                "replacement: give it a 'preventive' schedule with "
                "'policy': 'age' to have its interval chosen."
            )
    return chosen
