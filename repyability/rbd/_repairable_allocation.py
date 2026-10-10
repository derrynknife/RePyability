"""A ``RepairableRBD``'s allocations: the redundancy that meets a target
or a budget at the lowest cost (``allocate_redundancy``, trains of nodes
copied whole), and the component availabilities, or mean lives and repair
times, that meet a system availability target (``availability_allocation``,
``mttf_mttr_allocation``). The methods of ``RepairableRBD`` of those names
call these.
"""

import math
import warnings
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import OptimizeResult, brentq, minimize
from scipy.special import expit, logit, logsumexp, softmax

from repyability.rbd import (
    _ccf_groups,
    _costs,
    _crews,
    _long_run,
    _requirements,
)
from repyability.rbd._common import (
    _discount_rate,
    _horizons,
    _present_horizon,
)
from repyability.rbd._model_utils import (
    model_mean,
)
from repyability.rbd.redundancy_allocation import (
    lowest_total_cost,
    redundancy_caps,
)
from repyability.rbd.results import (
    AvailabilityAllocation,
    TotalCostAllocation,
)
from repyability.rbd.routes import Refused
from repyability.utils.checks import (
    number_or_nan,
    one_of,
)
from repyability.utils.wrappers import outside_level

if TYPE_CHECKING:
    from repyability.rbd.repairable_rbd import RepairableRBD

_ALLOCATION_CREWS = (
    "the allocations assume",
    "Compare designs by their long-run values, or simulate them with "
    "compare().",
)


class _Train(NamedTuple):
    """A train ``allocate_redundancy`` may copy (see its ``trains``): its
    name, its nodes in order, the node its last one feeds, where its
    copies join, and how many of its inputs that node needs."""

    name: Hashable
    members: Tuple[Hashable, ...]
    exit: Hashable
    k: int


@dataclass(frozen=True)
class _TrainCopy:
    """A node of a copy of a train, drawn by ``allocate_redundancy``: the
    train's ``index``-th copy of its ``node``. Equal only to itself, so
    that no node of the diagram can share its name."""

    train: Hashable
    index: int
    node: Hashable


def _allocation_target(target) -> float:
    """An availability allocation's target, checked to be in [0, 1]."""
    value = number_or_nan(target)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"target must be a number in [0, 1], got {target!r}.")
    return value


def _require_groups_allocated(rbd, chosen, drawn) -> dict:
    """Raise unless the redundancy allocation takes in the common-cause
    groups (#158): their chains cover them, a member given copies is in
    a ``BetaFactor`` group (its copies join the group, and an MGL
    model's letters are for its group's size), and no member is in a
    train. Returns each member given copies' group."""
    from repyability.rbd.ccf import BetaFactor

    _ccf_groups._require_ccf_long_run(rbd)
    group_of = {m: g for g in rbd.ccf_groups for m in g.members}
    for node in chosen:
        group = group_of.get(node)
        if (
            group is not None
            and _ccf_groups._ccf_timings(rbd, group) is not None
        ):
            raise NotImplementedError(
                f"Node {node!r} is in a common-cause group whose "
                "members' tests or repairs take time: its copies would "
                "join the group, which its chain does not take in, as "
                "yet. Leave the group's members out of nodes, or "
                "estimate the design by simulation, with cost()."
            )
        if group is not None and not isinstance(group.model, BetaFactor):
            raise NotImplementedError(
                f"Node {node!r} is in a common-cause group whose model, "
                f"{group.model!r}, is for a group of "
                f"{len(group.members)}: its copies would join the group, "
                "and an MGL model's letters for a larger group are not "
                "known. Give the group a BetaFactor model, whose beta "
                "holds at any size, or leave its members out of nodes."
            )
    for train in drawn:
        inside = [m for m in train.members if m in group_of]
        if inside:
            raise NotImplementedError(
                f"Train {train.name!r} holds {inside}, in a common-cause "
                "group, whose copies the allocation of trains does not "
                "take in, as yet: give the members copies as nodes."
            )
    return {node: group_of[node] for node in chosen if node in group_of}


def allocate_redundancy(
    rbd,
    horizon: float,
    *,
    nodes: Optional[Collection[Hashable]],
    trains: Optional[Mapping],
    min_availability: Optional[float],
    max_units: Union[int, Dict[Hashable, int], None],
    method: str,
    discount_rate: float,
) -> TotalCostAllocation:
    """See ``RepairableRBD.allocate_redundancy``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    rate = _discount_rate(discount_rate)
    if np.ndim(horizon) != 0:
        raise ValueError(
            f"horizon must be a number for allocate_redundancy, got "
            f"{horizon!r}."
        )
    horizon = float(_horizons(horizon, rate))
    # The running costs count over the horizon's present value.
    present = float(
        _present_horizon(
            np.array(horizon), rate, partial(_requirements._mean_lives, rbd)
        )
    )
    one_of("method", method, ("exact", "greedy"))
    if nodes is None and trains is not None:
        chosen: list = []
    elif nodes is None:
        chosen = [n for n in rbd.components if n in rbd.acquisition_costs]
        if not chosen:
            raise ValueError(
                "No component has an acquisition_cost, so there is "
                "nothing to buy copies of: give the components one, or "
                "name the nodes that may be given copies with `nodes`."
            )
    else:
        chosen = list(dict.fromkeys(nodes))
        if not chosen:
            raise ValueError("nodes must name at least one component.")
        for node in chosen:
            _requirements._require_component(rbd, node, "nodes")
            if isinstance(rbd.components[node], RepairableRBD):
                raise ValueError(
                    f"Node {node!r} is a nested RepairableRBD, which "
                    "cannot be given copies here: its costs are not "
                    "this RBD's."
                )
    drawn = _allocation_trains(rbd, trains, chosen)
    items = chosen + [train.name for train in drawn]
    caps = redundancy_caps(
        items, max_units, named="nodes or trains" if drawn else "nodes"
    )
    max_unavailability = None
    if min_availability is not None:
        try:
            target = float(min_availability)
        except (TypeError, ValueError):
            target = float("nan")
        if not 0.0 < target < 1.0:
            raise ValueError(
                "min_availability must be a number in (0, 1), got "
                f"{min_availability!r}."
            )
        max_unavailability = 1.0 - target

    # Adding copies adds jobs for the crews: the search assumes none waits.
    _crews._require_unlimited_crews(rbd, *_ALLOCATION_CREWS)
    grouped = _require_groups_allocated(rbd, chosen, drawn)
    # Every node's availability (and unavailability) over the times the
    # long-run values average over, and each component's own cost.
    times, weights = _long_run._long_run_grid(rbd)
    up = {
        node: np.atleast_1d(np.asarray(a, dtype=float))
        for node, a in _long_run._availabilities_at(rbd, times).items()
    }
    down = {
        node: np.atleast_1d(np.asarray(u, dtype=float))
        for node, u in _long_run._unavailabilities_at(rbd, times).items()
    }
    copy_cost = {}
    for node in rbd.components:
        rate = _costs._node_cost_rate(
            rbd, node, _long_run._node_availability(rbd, node)
        )
        copy_cost[node] = (
            rbd.acquisition_costs.get(node, 0.0),
            rate,
            rbd.acquisition_costs.get(node, 0.0) + present * rate,
        )
    # A train's copy costs its nodes'.
    train_cost = [
        math.fsum(copy_cost[node][2] for node in train.members)
        for train in drawn
    ]
    item_cost = [copy_cost[node][2] for node in chosen] + train_cost
    for index, (name, cost, cap) in enumerate(zip(items, item_cost, caps)):
        if cost <= 0.0 and cap == math.inf:
            what = "train " if index >= len(chosen) else ""
            raise ValueError(
                f"A copy of {what}{name!r} costs nothing over the horizon "
                "(it has no acquisition or running cost), so copies "
                "could be added without end: give it an "
                "acquisition_cost or a max_units."
            )
    size = len(times)
    structures: Dict[tuple, Any] = {(): rbd._decomposition()}

    def structure(extra: tuple):
        # The diagram with extra[i] copies of train i drawn alongside it.
        if not any(extra):
            return structures[()]
        if extra not in structures:
            structures[extra] = rbd._decompose_graph(
                _with_train_copies(rbd, drawn, extra)
            )
        return structures[extra]

    # With common-cause groups (#158), a member's copies join its group:
    # each design's groups' states, kept.
    designs: Dict[tuple, list] = {}

    def group_states(copies: dict) -> list:
        key = tuple(
            copies.get(m, 1) for g in rbd.ccf_groups for m in g.members
        )
        if key not in designs:
            designs[key] = [
                _ccf_groups._group_states(
                    rbd,
                    group,
                    times,
                    [
                        (
                            0
                            if copies.get(m, 1) == math.inf
                            else copies.get(m, 1)
                        )
                        for m in group.members
                    ],
                )
                for group in rbd.ccf_groups
            ]
        return designs[key]

    def unavailability(counts) -> float:
        p, q = dict(up), dict(down)
        joining: dict = {}
        for node, n in zip(chosen, counts):
            if node in grouped:
                joining[node] = n
                if n == math.inf:
                    p[node], q[node] = np.ones(size), np.zeros(size)
            elif n != 1:
                q[node] = down[node] ** n
                p[node] = 1.0 - q[node]
        extra = []
        for train, n in zip(drawn, counts[len(chosen) :]):
            # Made perfect (the search's bound), it is as many perfect
            # copies as its vote needs.
            copies = train.k if n == math.inf else int(n) - 1
            extra.append(copies)
            for index in range(1, copies + 1):
                for node in train.members:
                    name = _TrainCopy(train.name, index, node)
                    if n == math.inf:
                        p[name], q[name] = np.ones(size), np.zeros(size)
                    else:
                        p[name], q[name] = up[node], down[node]
        points, at = size, weights
        if rbd.ccf_groups:
            p, split, at, _ = _ccf_groups._with_ccf_groups(
                rbd,
                times,
                p,
                q,
                weights,
                group_states(joining),
                "The redundancy allocation",
            )
            assert split is not None
            q, points = split, len(at)
        _, fails = structure(tuple(extra)).probabilities(
            p, q, shape=points, works=False, fails=True
        )
        fails = np.broadcast_to(np.asarray(fails, dtype=float), (points,))
        return float(at @ fails)

    copies_down: Dict[tuple, np.ndarray] = {}

    def all_down(node, k: int) -> np.ndarray:
        # The chance that all k copies of a common-cause group member
        # are down, at each time: from its group's chain of them alone,
        # as the others' copies do not change their law.
        if (node, k) not in copies_down:
            group = grouped[node]
            counts = [k if m == node else 0 for m in group.members]
            states = _ccf_groups._group_states(rbd, group, times, counts)
            copies_down[(node, k)] = states.probabilities[:, -1]
        return copies_down[(node, k)]

    def gain(node):
        # The most the k + 1-th copy of the node can lower the system
        # unavailability: the node's own fall in unavailability (a
        # common-cause group member's, all its copies down, from its
        # group's chain).
        if node in grouped:
            return lambda k: float(
                weights @ (all_down(node, k) - all_down(node, k + 1))
            )
        return lambda k: float(weights @ (down[node] ** k * up[node]))

    def in_series(node) -> bool:
        # Down alone, it takes the system down: it lies on every path.
        alone = {n: np.ones(1) for n in rbd.nodes}
        alone[node] = np.zeros(1)
        return float(rbd.system_probability(alone)[0]) == 0.0

    def train_gain(train):
        # The n + 1-th train changes the system only if it brings its
        # vote to k working inputs, so at most k - 1 of the n there
        # work: it lowers the unavailability by at most its own
        # availability times the chance of that.
        from scipy.stats import binom

        own = -np.expm1(
            np.sum([np.log1p(-down[node]) for node in train.members], 0)
        )
        return lambda n: float(
            weights
            @ ((1.0 - own) * binom.sf(n - train.k, n, np.minimum(own, 1)))
        )

    series = None
    varying = set(rbd._inspection) | set(_long_run._block_nodes(rbd))
    if not drawn and all(
        node not in varying and node not in grouped and in_series(node)
        for node in chosen
    ):
        # Each such node's availability is constant, and the system's is
        # theirs times the rest's: the exact search is then a dynamic
        # program over the nodes.
        series = [(lambda n, q=float(down[node][0]): q**n) for node in chosen]
    counts, _, u = lowest_total_cost(
        unavailability,
        item_cost,
        present * rbd.downtime_cost_rate,
        [gain(node) for node in chosen]
        + [train_gain(train) for train in drawn],
        caps,
        max_unavailability,
        method,
        series,
    )
    units = {name: int(n) for name, n in zip(items, counts)}
    # Every component once, its copies as many times more, and each
    # train's copies their nodes' costs as many times.
    extra = {
        node: units[train.name] - 1
        for train in drawn
        for node in train.members
    }
    acquisition = math.fsum(
        (units.get(node, 1) + extra.get(node, 0)) * copy_cost[node][0]
        for node in rbd.components
    )
    cost_rate = (
        math.fsum(
            (units.get(node, 1) + extra.get(node, 0)) * copy_cost[node][1]
            for node in rbd.components
        )
        + rbd.downtime_cost_rate * u
    )
    return TotalCostAllocation(
        units=units,
        total_cost=acquisition + present * cost_rate,
        acquisition_cost=acquisition,
        cost_rate=cost_rate,
        availability=1.0 - u,
        horizon=horizon,
        method=method,
        discount_rate=float(discount_rate),
        trains=(
            {train.name: list(train.members) for train in drawn}
            if drawn
            else None
        ),
    )


def _allocation_trains(rbd, trains, chosen) -> List[_Train]:
    """The trains ``allocate_redundancy`` may copy (see its ``trains``),
    checked: each a chain of components in series ending at a node that
    feeds one node, a node in one train at most and not in ``chosen``
    (the nodes copied alone), and named apart from the nodes."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if trains is None:
        return []
    if not isinstance(trains, Mapping) or not trains:
        raise ValueError(
            "trains must be a non-empty dict {name: [nodes in series]}, "
            f"got {trains!r}."
        )
    graph = rbd.G
    out: List[_Train] = []
    seen: Dict[Hashable, Hashable] = {}
    for name, members in trains.items():
        if name in graph.nodes:
            raise ValueError(
                f"Train {name!r} is named as a node: name a train apart "
                "from the nodes, as units holds both."
            )
        if isinstance(members, (str, bytes)) or not isinstance(
            members, Collection
        ):
            raise ValueError(
                f"Train {name!r} must be a list of nodes, got " f"{members!r}."
            )
        members = list(members)
        if not members:
            raise ValueError(f"Train {name!r} has no nodes.")
        for node in members:
            _requirements._require_component(rbd, node, f"train {name!r}")
            if isinstance(rbd.components[node], RepairableRBD):
                raise ValueError(
                    f"Node {node!r} of train {name!r} is a nested "
                    "RepairableRBD, which cannot be given copies here: "
                    "its costs are not this RBD's."
                )
            if node in seen:
                raise ValueError(
                    f"Node {node!r} is in trains {seen[node]!r} and "
                    f"{name!r}: a node is in one train at most."
                )
            if node in chosen:
                raise ValueError(
                    f"Node {node!r} is in nodes and in train {name!r}: "
                    "copy it on its own or with its train."
                )
            seen[node] = name
        for a, b in zip(members, members[1:]):
            if list(graph.successors(a)) != [b] or list(
                graph.predecessors(b)
            ) != [a]:
                raise ValueError(
                    f"Train {name!r} must be a chain of nodes in series: "
                    f"{a!r} feeding {b!r} alone, and {b!r} fed by {a!r} "
                    "alone."
                )
        exits = list(graph.successors(members[-1]))
        if len(exits) != 1:
            raise ValueError(
                f"Train {name!r} must end at a node feeding one node, "
                f"which its copies join; {members[-1]!r} feeds "
                f"{len(exits)}."
            )
        out.append(
            _Train(
                name,
                tuple(members),
                exits[0],
                int(graph.nodes[exits[0]]["k"]),
            )
        )
    return out


def _with_train_copies(rbd, trains: List[_Train], extra: tuple):
    """The diagram's graph with ``extra[i]`` copies of ``trains[i]``
    drawn alongside it: each fed as its first node is, feeding the node
    its last feeds, whose ``k`` is kept (see ``allocate_redundancy``)."""
    graph = rbd.G.copy()
    for train, copies in zip(trains, extra):
        first = list(rbd.G.predecessors(train.members[0]))
        for index in range(1, copies + 1):
            names = [_TrainCopy(train.name, index, n) for n in train.members]
            for name, node in zip(names, train.members):
                graph.add_node(name, **rbd.G.nodes[node])
            graph.add_edges_from((u, names[0]) for u in first)
            graph.add_edges_from(zip(names, names[1:]))
            graph.add_edge(names[-1], train.exit)
    return graph


def availability_allocation(
    rbd,
    target: float,
    method: str,
    *,
    fixed: Optional[Collection[Hashable]],
    weights: Optional[Dict],
    max_availabilities: Optional[Dict],
    feasibility: Optional[Dict],
) -> AvailabilityAllocation:
    """See ``RepairableRBD.availability_allocation``."""
    methods = ("cost_based", "improvement", "minimum_effort", "equal")
    if method not in methods:
        raise ValueError(
            "method must be one of "
            f"{', '.join(repr(m) for m in methods)}, got {method!r}."
        )
    options = {
        "weights": (weights, "improvement"),
        "max_availabilities": (max_availabilities, "cost_based"),
        "feasibility": (feasibility, "cost_based"),
    }
    for name, (value, owner) in options.items():
        if value is not None and method != owner:
            raise ValueError(f"{name} applies to method={owner!r} only.")
    target = _allocation_target(target)
    current, free, held = _allocatable(rbd, fixed)
    for name, (value, _) in options.items():
        _allocation_option(rbd, name, value, held)
    view = _allocation_view(rbd, held)
    start = {node: current[node] for node in rbd.nodes}
    if method == "cost_based":
        maxima = {node: current[node] for node in held}
        maxima.update(max_availabilities or {})
        allocated = view.cost_based_allocation(
            target,
            start,
            max_probabilities=maxima,
            feasibility=feasibility,
        )
    elif method == "improvement":
        allocated = view.improvement_allocation(
            target, start, fixed=list(held), weights=weights
        )
    elif method == "equal":
        # From 0.5 for every component allocated, scaled alike, so they
        # all end up the same.
        allocated = view.improvement_allocation(
            target,
            {
                node: current[node] if node in held else 0.5
                for node in rbd.nodes
            },
            fixed=list(held),
        )
    else:
        allocated = _minimum_effort_held(view, target, current, held)
    if view is not rbd and method != "minimum_effort":
        rbd.res = view.res
    mttf, mttr = {}, {}
    for node, (up, down) in free.items():
        a = allocated[node]
        mttf[node] = down * a / (1.0 - a) if a < 1.0 else math.inf
        mttr[node] = up * (1.0 - a) / a if a > 0.0 else math.inf
    return _allocation_result(rbd, view, allocated, mttf, mttr)


def mttf_mttr_allocation(
    rbd,
    target: float,
    *,
    levers: str,
    fixed: Optional[Collection[Hashable]],
    max_mttf: Optional[Dict],
    min_mttr: Optional[Dict],
    mttf_feasibility: Optional[Dict],
    mttr_feasibility: Optional[Dict],
) -> AvailabilityAllocation:
    """See ``RepairableRBD.mttf_mttr_allocation``."""
    if levers not in ("both", "mttf", "mttr"):
        raise ValueError(
            f"levers must be 'both', 'mttf' or 'mttr', got {levers!r}."
        )
    target = _allocation_target(target)
    current, free, held = _allocatable(rbd, fixed)
    most = _allocation_option(rbd, "max_mttf", max_mttf, held)
    least = _allocation_option(rbd, "min_mttr", min_mttr, held)
    ease = (
        _allocation_option(rbd, "mttf_feasibility", mttf_feasibility, held),
        _allocation_option(rbd, "mttr_feasibility", mttr_feasibility, held),
    )
    names = ("mttf_feasibility", "mttr_feasibility")
    # Each component's failure rate and repair time, as (start, floor):
    # each falls from its start towards its floor as its lever is used,
    # to floor + (start - floor) * exp(-v) for the lever's v >= 0.
    span: Dict[Hashable, Tuple[Tuple[float, float], ...]] = {}
    variables: List[Tuple[Hashable, int, float]] = []
    for node, (mttf, mttr) in free.items():
        top = number_or_nan(most.get(node, math.inf))
        if not top >= mttf * (1.0 - 1e-12):
            raise ValueError(
                f"max_mttf[{node!r}] must be at least the component's "
                f"MTTF, {mttf:.6g}, got {most[node]!r}."
            )
        bottom = number_or_nan(least.get(node, 0.0))
        if not 0.0 <= bottom <= mttr * (1.0 + 1e-12):
            raise ValueError(
                f"min_mttr[{node!r}] must be between 0 and the "
                f"component's MTTR, {mttr:.6g}, got {least[node]!r}."
            )
        span[node] = (
            (1.0 / mttf, 1.0 / max(top, mttf)),
            (mttr, min(bottom, mttr)),
        )
        for lever, name in enumerate(names):
            f = number_or_nan(ease[lever].get(node, 0.5))
            if not 0.0 <= f < 1.0:
                raise ValueError(
                    f"{name}[{node!r}] must be in [0, 1), got "
                    f"{ease[lever][node]!r}."
                )
            start, floor = span[node][lever]
            if levers in ("both", ("mttf", "mttr")[lever]) and (start > floor):
                variables.append((node, lever, 1.0 - f))
    if not variables:
        raise ValueError(
            "No MTTF or MTTR can change: every lever is held (by levers, "
            "max_mttf or min_mttr)."
        )
    steepness = np.array([s for _, _, s in variables])
    size = len(variables)

    def design(v) -> Dict[Hashable, List[float]]:
        """Each component's failure rate and repair time."""
        values = {
            node: [ends[0][0], ends[1][0]] for node, ends in span.items()
        }
        for (node, lever, _), x in zip(variables, v):
            if x > 0.0:
                start, floor = span[node][lever]
                gap = (start - floor) * math.exp(-x)
                values[node][lever] = floor + gap
        return values

    def probabilities(v) -> Tuple[Dict, Dict]:
        p = {node: current[node] for node in rbd.nodes}
        q = {node: 1.0 - current[node] for node in rbd.nodes}
        for node, (rate, repair) in design(v).items():
            odds = rate * repair  # of being down
            p[node] = 1.0 / (1.0 + odds)
            q[node] = odds / (1.0 + odds)
        return p, q

    def result(v) -> AvailabilityAllocation:
        values = design(v)
        mttf = {}
        for node, (rate, _) in values.items():
            if rate == span[node][0][0]:
                mttf[node] = free[node][0]  # unchanged, exactly
            else:
                mttf[node] = 1.0 / rate if rate > 0.0 else math.inf
        return _allocation_result(
            rbd,
            view,
            probabilities(v)[0],
            mttf,
            {node: repair for node, (_, repair) in values.items()},
        )

    view = _allocation_view(rbd, held)
    goal = float(logit(target))
    now = view._log_odds(*probabilities(np.zeros(size)))[0]
    if goal <= now:
        rbd.res = OptimizeResult(
            x=np.zeros(size),
            fun=float(np.log(size)),
            success=True,
            message="The target is already met.",
        )
        return result(np.zeros(size))
    best = view._log_odds(*probabilities(np.full(size, np.inf)))[0]
    # The limits are only approached (at an ever-growing cost), so a
    # target at them, to within rounding, cannot be met either.
    if goal >= best - 1e-9:
        raise ValueError(
            f"target {target} cannot be reached: from the current "
            f"availability ({expit(now):.6g}) the system can only "
            f"approach {expit(best):.6g}, with every MTTF and MTTR that "
            "may change at its limit."
        )

    def shortfall(v: np.ndarray) -> float:
        return view._log_odds(*probabilities(v))[0] - goal

    def shortfall_gradient(v: np.ndarray) -> np.ndarray:
        p, q = probabilities(v)
        derivative = view._log_odds(p, q)[1]
        values = design(v)
        gradient = np.empty(size)
        for i, ((node, lever, _), x) in enumerate(zip(variables, v)):
            start, floor = span[node][lever]
            # The lever lowers the odds of being down, rate * repair, at
            # the other quantity times its own fall; p = 1 / (1 + odds).
            other = values[node][1 - lever]
            fall = other * (start - floor) * math.exp(-x)
            gradient[i] = derivative[node] * p[node] ** 2 * fall
        return gradient

    def log_total_cost(v: np.ndarray) -> Tuple[float, np.ndarray]:
        costs = steepness * np.expm1(v)
        return float(logsumexp(costs)), softmax(costs) * steepness * np.exp(v)

    def common_shift(start: np.ndarray, along: np.ndarray) -> np.ndarray:
        """The start moved up by the least common v, on the levers
        ``along`` marks, meeting the target (it exists, as the target is
        below what the limits allow)."""
        high = 1.0
        while shortfall(start + high * along) < 0.0 and high < 1e6:
            high *= 2.0
        return start + along * brentq(
            lambda d: shortfall(start + d * along), 0.0, high, xtol=1e-14
        )

    every = np.ones(size)
    res = minimize(
        log_total_cost,
        common_shift(np.zeros(size), every),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, None)] * size,
        constraints=[
            {"type": "ineq", "fun": shortfall, "jac": shortfall_gradient}
        ],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    # A lever the solver left within a hair of its bound is unused.
    v = np.where(res.x < 1e-9, 0.0, res.x)
    if shortfall(v) < 0.0:
        # Close the solver's last sliver of constraint tolerance, with
        # the levers in use.
        used = (v > 0.0).astype(float)
        v = common_shift(v, used if used.any() else every)
    res.x = v
    res.fun = log_total_cost(v)[0]
    rbd.res = res
    if not res.success:
        warnings.warn(
            "the cost minimisation stopped before converging "
            f"({res.message}); the design meets the target but may not "
            "be the cheapest.",
            stacklevel=outside_level(),
        )
    return result(v)


def _allocatable(
    rbd, fixed
) -> Tuple[Dict[Hashable, float], Dict[Hashable, Tuple[float, float]], set]:
    """For an availability allocation: every component's long-run
    availability; the MTTF and MTTR of those allocated one (with
    corrective repair alone: they fail and take time to repair, with no
    preventive or inspection schedule, are not in a common-cause group,
    whose members' MTTF and MTTR are the group's, and are not in
    ``fixed``); and the set of the others, which keep theirs."""
    _crews._require_unlimited_crews(rbd, *_ALLOCATION_CREWS)
    _ccf_groups._require_ccf_long_run(rbd)
    held = set()
    if fixed is not None:
        held = set(fixed)
        unknown = [node for node in held if node not in rbd.components]
        if unknown:
            raise ValueError(
                f"fixed names {sorted(unknown, key=str)}, which are not "
                "components of this RBD."
            )
    current = {
        node: _long_run._node_availability(rbd, node)
        for node in rbd.components
    }
    free = _allocated_levers(rbd, held)
    for node, (mttf, mttr) in free.items():
        current[node] = mttf / (mttf + mttr)
    return current, free, held


def _allocated_levers(rbd, held: set) -> Dict[Hashable, Tuple[float, float]]:
    """The MTTF and MTTR of each component an availability allocation
    allocates one: those with corrective repair alone (they fail and
    take time to repair, with no preventive or inspection schedule),
    not nested, not in a common-cause group (whose members' MTTF and
    MTTR are the group's) and not in ``held``, which gains the others.
    Raises if there are none: ``analysis_routes`` asks it, as it follows
    from the diagram."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    members = {m for group in rbd.ccf_groups for m in group.members}
    free: Dict[Hashable, Tuple[float, float]] = {}
    for node, component in rbd.components.items():
        if (
            node in held
            or isinstance(component, RepairableRBD)
            or node in rbd._preventive
            or node in rbd._inspection
            or node in rbd._standby
            or node in members
        ):
            held.add(node)
            continue
        mttf = model_mean(component.reliability)
        mttr = model_mean(component.time_to_replace)
        if not (0.0 < mttf < math.inf and 0.0 < mttr < math.inf):
            # Never down, down for good in the end, or up for good.
            held.add(node)
            continue
        free[node] = (mttf, mttr)
    if not free:
        raise Refused(
            "No component can be allocated an availability: only "
            "components with corrective repair alone (that fail and take "
            "time to repair, with no preventive or inspection schedule), "
            "not in fixed, can."
        )
    return free


def _allocation_option(rbd, name: str, mapping, held: set) -> dict:
    """A per-component option of an availability allocation, checked to
    name only components allocated an availability."""
    values = rbd._node_overrides(name, mapping)
    kept = [node for node in values if node in held]
    if kept:
        raise ValueError(
            f"{name} names {sorted(kept, key=str)}, which keep their "
            "availability: only components with corrective repair alone, "
            "not in fixed, are allocated one."
        )
    return values


def _allocation_view(rbd, held: set) -> "RepairableRBD":
    """This RBD as an availability allocation scores it. The components
    allocated an availability have a constant one, so when every held
    component does too the system's availability is the structure
    function of theirs, and the RBD itself will do. A held component
    that is inspected or block-replaced is up with a probability that
    varies over its schedule, together with the others on the calendar:
    the system's availability is then averaged over the long-run grid,
    as ``mean_availability`` averages it, by a view of the RBD whose
    ``_allocation_probability`` and ``_log_odds`` do so. With
    common-cause groups, whose members are held, its points are split
    by the groups' joint states (see ``_with_ccf_groups``)."""
    times, weights = _long_run._long_run_grid(rbd)
    if len(times) == 1 and not rbd.ccf_groups:
        return rbd
    profiles = _long_run._availabilities_at(rbd, times)
    if rbd.ccf_groups:
        profiles, _, weights, _ = _ccf_groups._with_ccf_groups(
            rbd,
            times,
            profiles,
            None,
            weights,
            what="The availability allocation",
        )
    view = copy(rbd)
    view.__dict__.pop("_ccf_tables", None)
    view.__dict__["_allocation_calendar"] = (
        weights,
        {node: profiles[node] for node in held},
        [node for node in rbd.nodes if node not in held],
    )
    return view


def _calendar_arrays(rbd, p: Dict, q: Optional[Dict]) -> Tuple[Dict, Dict]:
    """Node probabilities ``p`` and their complements ``q`` (by default
    ``1 - p``) over the allocation calendar's grid, with the held
    components' own availabilities there."""
    weights, profiles, _ = rbd.__dict__["_allocation_calendar"]
    size = len(weights)
    works = {node: np.full(size, float(p[node])) for node in p}
    fails = {
        node: np.full(
            size, 1.0 - float(p[node]) if q is None else float(q[node])
        )
        for node in p
    }
    for node, profile in profiles.items():
        works[node] = profile
        fails[node] = 1.0 - profile
    for node in rbd.in_or_out:
        works[node] = np.ones(size)
        fails[node] = np.zeros(size)
    return works, fails


def _calendar_means(rbd, works: Dict, fails: Dict) -> Tuple[float, float]:
    """The system's probabilities of working and of failing over the
    allocation calendar's grid, averaged over it."""
    weights = rbd.__dict__["_allocation_calendar"][0]
    size = len(weights)
    up, down = rbd._decomposition().probabilities(works, fails, shape=size)
    return (
        float(weights @ np.broadcast_to(up, (size,))),
        float(weights @ np.broadcast_to(down, (size,))),
    )


def _minimum_effort_held(rbd, target: float, current: Dict, held: set):
    """Albert's minimum-effort allocation to the components not
    ``held``: in series, the held components' availabilities (averaged
    over their calendars) are a factor of the system's, and the others
    make up the rest."""
    rbd._require_series()
    limit = rbd._allocation_probability(
        {node: current[node] if node in held else 1.0 for node in rbd.nodes}
    )
    if target > limit:
        raise ValueError(
            f"target {target} cannot be reached: the components that "
            "keep their availability hold the system to at most "
            f"{limit:.6g}."
        )
    raised = rbd.minimum_effort_allocation(
        target / limit if limit > 0.0 else 0.0,
        {node: 1.0 if node in held else current[node] for node in rbd.nodes},
    )
    return {
        node: current[node] if node in held else raised[node]
        for node in rbd.nodes
    }


def _allocation_result(
    rbd, view: "RepairableRBD", allocated: Dict, mttf: Dict, mttr: Dict
) -> AvailabilityAllocation:
    """An availability allocation's result, scored as ``view`` scores
    it."""
    return AvailabilityAllocation(
        availability={node: float(allocated[node]) for node in rbd.components},
        mttf=mttf,
        mttr=mttr,
        system_availability=view._allocation_probability(
            {node: allocated[node] for node in rbd.nodes}
        ),
    )
