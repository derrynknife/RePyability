"""A ``RepairableRBD``'s capacity: its distribution in the long run
(``capacity_distribution``), at points in time (``point_capacity``) and
over a mission (``mission_capacity``), from each component's chance of
working (``rbd.capacity``'s algorithm on the diagram's structure), with
repair crews' and common-cause groups' chains where they couple the
components. The methods of ``RepairableRBD`` of those names call these.
"""

import itertools
from functools import partial
from typing import (
    Collection,
    Dict,
    Hashable,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd import (
    _ccf_chain,
    _ccf_groups,
    _chain_transient,
    _long_run,
    _quadrature,
)
from repyability.rbd import capacity as _capacity
from repyability.rbd.results import (
    CapacityDistribution,
)
from repyability.utils.checks import (
    nonnegative_times,
)

#: The most patterns of nested RBDs' capacity levels the capacity over time
#: with limited repair crews follows the crews' chain for (#162).
_MAX_CREW_LEVELS = 4096


class _CrewCapacity:
    """The capacity over time of a system whose components wait for repair
    crews, around nested RBDs (#162). The nested RBDs have crews of their
    own, so they are independent of the chain and of each other: at a time
    the system is at a level with probability ``sum_c w_c(t) p(t) v_c``,
    over each combination ``c`` of the nested RBDs' levels (up or down, or
    their own capacities), ``w_c(t)`` its probability from their own
    curves, and ``v_c`` whether the system is at the level in each of the
    chain's states with the nested RBDs at ``c``. The chain is followed
    over time (see ``_chain_transient.Uniformized``) for each combination
    as it first has a probability, and kept."""

    def __init__(self, rbd, chain, start, nested: dict, working, broken):
        self.rbd, self.chain, self.start = rbd, chain, start
        self.nested = nested
        size = len(chain.probabilities)
        own = {
            node: chain.up[:, k].astype(float)
            for k, node in enumerate(chain.nodes)
        }
        # The nested RBDs' places, which each combination of their levels
        # takes (see ``_chain_for``).
        own.update({node: np.ones(size) for node in nested})
        self.arrays, _ = rbd._node_arrays(
            rbd._filled(own, size, working, broken)
        )
        self.size = size
        self.followed: Dict[tuple, tuple] = {}
        timing = rbd._uniformized(
            "The repair crews'",
            chain.generator,
            start,
            chain.probabilities,
            np.ones((size, 1)),
        )
        #: For the pieces of the integrals: the chain's and the nested
        #: RBDs' curves.
        self.curves = [_chain_transient.ChainCurve(timing), *nested.values()]

    def _chain_for(self, levels: tuple):
        """The system's levels with the nested RBDs at ``levels``, and the
        chain followed over time with whether it is at each, in each
        state."""
        if levels not in self.followed:
            own = {
                node: (np.array([level]), np.ones((1, self.size)))
                for node, level in zip(self.nested, levels)
            }
            values, rows = self.rbd._capacity_arrays(
                self.arrays, self.size, own
            )
            self.followed[levels] = (
                values,
                self.rbd._uniformized(
                    "The repair crews'",
                    self.chain.generator,
                    self.start,
                    self.chain.probabilities,
                    rows.T,
                ),
            )
        return self.followed[levels]

    def _distributions(self, x: np.ndarray) -> list:
        """Each nested RBD's capacity distribution at the times ``x``: its
        own, if it has capacities, else up (at its capacity here) or down,
        with its point availability."""
        out = []
        models = self.rbd._capacity_models()
        for node, curve in self.nested.items():
            if node in models:
                out.append(_capacity_at(self.rbd.components[node], curve, x))
            else:
                up = np.clip(np.asarray(curve.at(x), dtype=float), 0.0, 1.0)
                out.append(
                    _capacity.node_distribution(
                        self.rbd.capacity.get(node, np.inf), up, 1.0 - up
                    )
                )
        return out

    def rows(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """The system's capacity levels and their probabilities at the times
        ``x`` (one row per level)."""
        x = np.asarray(x, dtype=float).ravel()
        distributions = self._distributions(x)
        count = int(np.prod([len(levels) for levels, _ in distributions]))
        if count > _MAX_CREW_LEVELS:
            raise NotImplementedError(
                f"With {self.rbd.repair_crews} repair crew(s), the capacity "
                "over time is worked out for each combination of the nested "
                f"RBDs' levels: {count} are more than the {_MAX_CREW_LEVELS} "
                "it takes. Simulate it with availability(demand=...)."
            )
        totals: Dict[float, np.ndarray] = {}
        for choice in itertools.product(
            *[range(len(levels)) for levels, _ in distributions]
        ):
            weight = np.ones(len(x))
            for (_, chances), i in zip(distributions, choice):
                weight = weight * chances[i]
            if not weight.any():
                continue
            key = tuple(
                float(levels[i])
                for (levels, _), i in zip(distributions, choice)
            )
            values, chain = self._chain_for(key)
            at = np.clip(chain.values(x), 0.0, 1.0)
            for level, column in zip(values, at.T):
                level = float(level)
                totals[level] = totals.get(level, 0.0) + weight * column
        levels = np.array(sorted(totals), dtype=float)
        rows = np.array([totals[level] for level in levels]).reshape(
            len(levels), len(x)
        )
        return levels, rows


def _capacity_refusal(rbd) -> Optional[Tuple[str, tuple]]:
    """What ``capacity_distribution`` refuses beyond the long-run
    values, in the order it checks: a degrading component on a
    schedule, here or in a nested RBD, then no capacity at all. The
    message and the nodes, or None."""
    from repyability.rbd import routes as r
    from repyability.rbd.repairable_rbd import RepairableRBD

    for node, model in rbd._capacity_models().items():
        if isinstance(model, RepairableRBD):
            inner = _capacity_refusal(model)
            if inner:
                return inner[0], (node,)
            continue
        message = r.refusal(partial(rbd._require_unscheduled_stages, node))
        if message:
            return message, (node,)
    message = r.refusal(rbd._require_capacity)
    return (message, ()) if message else None


def _crew_capacity(
    rbd, working_nodes, broken_nodes, states: dict
) -> Tuple[np.ndarray, "_chain_transient.Uniformized"]:
    """The capacity over time with limited repair crews and no nested
    RBDs: its levels, and the crews' chain followed over time with, as
    its vectors, whether the system is at each level in each state (as
    ``capacity_distribution`` averages them in the long run)."""
    working, broken = set(working_nodes), set(broken_nodes)
    forced = working | broken
    rbd._require_crew_over_time(frozenset(forced), "capacity")
    chain = rbd._crew_chain(frozenset(forced))
    size = len(chain.probabilities)
    own = {
        node: chain.up[:, k].astype(float)
        for k, node in enumerate(chain.nodes)
    }
    arrays, _ = rbd._node_arrays(rbd._filled(own, size, working, broken))
    levels, rows = rbd._capacity_arrays(arrays, size, {})
    uniformized = rbd._uniformized(
        "The repair crews'",
        chain.generator,
        rbd._crew_start(chain, states),
        chain.probabilities,
        rows.T,
    )
    return levels, uniformized


def _crew_capacity_over(
    rbd, horizon: float, working_nodes, broken_nodes, states: dict
) -> "_CrewCapacity":
    """The capacity over time with limited repair crews around nested
    RBDs (#162): see ``_CrewCapacity``."""
    working, broken = set(working_nodes), set(broken_nodes)
    forced = working | broken
    nested = rbd._require_crew_over_time(frozenset(forced))
    chain = rbd._crew_chain(frozenset(forced))
    curves = {
        node: rbd.components[node]._nested_curve(
            horizon,
            stages=node in rbd._capacity_models(),
            start=states.get(node),
        )
        for node in nested
    }
    return _CrewCapacity(
        rbd,
        chain,
        rbd._crew_start(chain, states),
        curves,
        working,
        broken,
    )


def _capacity_at(rbd, curve, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """This RBD's capacity distribution at the times ``x``, as a node of
    another, from its curve over time (see ``_nested_curve``, with
    ``stages``): from its nodes' curves, or with limited repair crews
    from their chain (#162)."""
    capacity = getattr(curve, "capacity", None)
    if capacity is not None:
        return capacity.rows(x)
    if isinstance(curve, _ccf_chain.GroupsCurve):
        return _groups_capacity_rows(
            rbd, curve.curves, curve.system, x, set(), set()
        )
    return _capacity_rows(rbd, curve.curves, x, set(), set())


def capacity_distribution(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
) -> CapacityDistribution:
    """See ``RepairableRBD.capacity_distribution``."""
    probabilities, weights = _long_run._long_run_probabilities(
        rbd, working_nodes, broken_nodes, "The capacity distribution"
    )
    working_nodes = set(working_nodes or ())
    broken_nodes = set(broken_nodes or ())
    arrays, size = rbd._node_arrays(probabilities)
    own = {}
    for node in rbd._capacity_models():
        levels, shares = _long_run_capacity(rbd, node)
        own[node] = rbd._forced(
            (levels, np.repeat(shares[:, None], size, axis=1)),
            node,
            working_nodes,
            broken_nodes,
        )
    levels, rows = rbd._capacity_arrays(arrays, size, own)
    return CapacityDistribution(levels, rows @ weights)


def _require_capacity_models(rbd) -> None:
    """Raise what ``_capacity_refusal`` reports, in the same order: a
    degrading component on a schedule, here or in a nested RBD, then no
    capacity at all."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    for node, model in rbd._capacity_models().items():
        if isinstance(model, RepairableRBD):
            _require_capacity_models(model)
        else:
            rbd._require_unscheduled_stages(node)
    rbd._require_capacity()


def _capacity_curves(
    rbd, horizon: float, working_nodes, broken_nodes, state=None
) -> dict:
    """The nodes' curves for the capacity over time (see
    ``point_capacity``): a degrading component's following its stages,
    and a node that takes its capacity from its model kept even when
    held working (its levels are then those it is up at)."""
    held = set(working_nodes) - set(rbd._capacity_models())
    return rbd._availability_curves(
        horizon,
        held | set(broken_nodes),
        stages=True,
        state=rbd._states(state, set(working_nodes) | set(broken_nodes)),
    )


def _capacity_rows(
    rbd, curves: dict, x: np.ndarray, working_nodes, broken_nodes
) -> Tuple[np.ndarray, np.ndarray]:
    """The distribution of the system's capacity at each of the times
    ``x``, from its nodes' curves (see ``_capacity_curves``): its
    levels, and one row of probabilities per level, one column per
    time. A node with a capacity carries it with its point availability
    there; a degrading component is in each stage with the probability
    its curve gives, and a nested RBD with capacities brings its own
    distribution."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    size = len(x)
    values = {node: curve.at(x) for node, curve in curves.items()}
    arrays, _ = rbd._node_arrays(
        rbd._filled(values, size, working_nodes, broken_nodes)
    )
    own = {}
    for node, model in rbd._capacity_models().items():
        if node in broken_nodes:
            own[node] = (np.zeros(1), np.ones((1, size)))
            continue
        curve = curves[node]
        if isinstance(model, RepairableRBD):
            distribution = _capacity_at(model, curve, x)
        else:
            distribution = _capacity.merged(
                np.array((0.0,) + model.capacities),
                np.vstack([1.0 - values[node], curve.stages_at(x)]),
            )
        own[node] = rbd._forced(
            distribution, node, working_nodes, broken_nodes
        )
    return rbd._capacity_arrays(arrays, size, own)


def _groups_capacity(
    rbd, horizon: float, working_nodes, broken_nodes, state
) -> tuple:
    """With common-cause groups (#158): the curves of the nodes outside
    them for the capacity over time (see ``_capacity_curves``), and the
    groups' system (see ``_ccf_chain.GroupsSystem``)."""
    working, broken = set(working_nodes), set(broken_nodes)
    states = rbd._states(state, working | broken)
    _ccf_groups._require_free_members(rbd, working, broken)
    _ccf_groups._require_groups_over_time(rbd, states)
    members = {m for group in rbd.ccf_groups for m in group.members}
    held = working - set(rbd._capacity_models())
    curves = rbd._availability_curves(
        horizon,
        held | broken | members,
        stages=True,
        state=states,
        groups=True,
    )
    return curves, _ccf_groups._groups_system(rbd, working, broken, "p")


def _groups_capacity_rows(
    rbd, curves: dict, system, x: np.ndarray, working_nodes, broken_nodes
) -> Tuple[np.ndarray, np.ndarray]:
    """``_capacity_rows`` with common-cause groups: worked out at the
    points the groups' combinations split each time into (see
    ``_ccf_chain.GroupsSystem``), and averaged back onto the times with
    the combinations' probabilities."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    size = len(x)
    values = {node: curve.at(x) for node, curve in curves.items()}
    probabilities, weights, index, _ = system._split(values, x)
    points = len(index)
    arrays, _ = rbd._node_arrays(probabilities)
    own = {}
    for node, model in rbd._capacity_models().items():
        if node in broken_nodes:
            own[node] = (np.zeros(1), np.ones((1, points)))
            continue
        curve = curves[node]
        if isinstance(model, RepairableRBD):
            levels, rows = _capacity_at(model, curve, x)
        else:
            levels, rows = _capacity.merged(
                np.array((0.0,) + model.capacities),
                np.vstack([1.0 - values[node], curve.stages_at(x)]),
            )
        own[node] = rbd._forced(
            (levels, np.asarray(rows)[:, index]),
            node,
            working_nodes,
            broken_nodes,
        )
    levels, rows = rbd._capacity_arrays(arrays, points, own)
    out = np.array(
        [np.bincount(index, weights * row, size) for row in rows]
    ).reshape(len(levels), size)
    return levels, np.clip(out, 0.0, 1.0)


def point_capacity(
    rbd,
    x,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    state,
) -> CapacityDistribution:
    """See ``RepairableRBD.point_capacity``."""
    times = nonnegative_times(x)
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    _require_capacity_models(rbd)
    ends = times.ravel()
    horizon = float(ends.max()) if ends.size else 0.0
    forced = working_nodes | broken_nodes
    if rbd._crews_couple() and rbd._crew_nested(forced):
        capacity = _crew_capacity_over(
            rbd,
            horizon,
            working_nodes,
            broken_nodes,
            rbd._states(state, forced),
        )
        levels, rows = capacity.rows(ends)
    elif rbd._crews_couple():
        levels, chain = _crew_capacity(
            rbd,
            working_nodes,
            broken_nodes,
            rbd._states(state, working_nodes | broken_nodes),
        )
        rows = np.clip(chain.values(ends).T, 0.0, 1.0)
    elif rbd.ccf_groups:
        curves, system = _groups_capacity(
            rbd, horizon, working_nodes, broken_nodes, state
        )
        levels, rows = _groups_capacity_rows(
            rbd, curves, system, ends, working_nodes, broken_nodes
        )
    else:
        curves = _capacity_curves(
            rbd, horizon, working_nodes, broken_nodes, state
        )
        levels, rows = _capacity_rows(
            rbd, curves, ends, working_nodes, broken_nodes
        )
    return CapacityDistribution(
        levels, rows[:, 0] if np.ndim(x) == 0 else rows
    )


def mission_capacity(
    rbd,
    t,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    state,
) -> CapacityDistribution:
    """See ``RepairableRBD.mission_capacity``."""
    windows = nonnegative_times(t)
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    _require_capacity_models(rbd)
    ends = windows.ravel()
    horizon = float(ends.max()) if ends.size else 0.0
    forced = working_nodes | broken_nodes
    if rbd._crews_couple() and rbd._crew_nested(forced):
        capacity = _crew_capacity_over(
            rbd,
            horizon,
            working_nodes,
            broken_nodes,
            rbd._states(state, forced),
        )
        levels, rows = _mission_rows(
            rbd, capacity.curves, capacity.rows, ends, horizon
        )
        return CapacityDistribution(
            levels, rows[:, 0] if np.ndim(t) == 0 else rows
        )
    if rbd._crews_couple():
        levels, chain = _crew_capacity(
            rbd,
            working_nodes,
            broken_nodes,
            rbd._states(state, working_nodes | broken_nodes),
        )
        positive = ends > 0.0
        rows = chain.values(ends).T
        rows[:, positive] = chain.integrals(ends[positive]).T / ends[positive]
        rows = np.clip(rows, 0.0, 1.0)
        keep = np.any(rows != 0.0, axis=1)
        levels, rows = levels[keep], rows[keep]
        return CapacityDistribution(
            levels, rows[:, 0] if np.ndim(t) == 0 else rows
        )
    if rbd.ccf_groups:
        grouped, system = _groups_capacity(
            rbd, horizon, working_nodes, broken_nodes, state
        )
        levels, rows = _mission_rows(
            rbd,
            [*grouped.values(), system.curve],
            lambda x: _groups_capacity_rows(
                rbd, grouped, system, x, working_nodes, broken_nodes
            ),
            ends,
            horizon,
        )
        return CapacityDistribution(
            levels, rows[:, 0] if np.ndim(t) == 0 else rows
        )
    curves = _capacity_curves(rbd, horizon, working_nodes, broken_nodes, state)
    levels, rows = _mission_rows(
        rbd,
        list(curves.values()),
        lambda x: _capacity_rows(rbd, curves, x, working_nodes, broken_nodes),
        ends,
        horizon,
    )
    return CapacityDistribution(
        levels, rows[:, 0] if np.ndim(t) == 0 else rows
    )


def _mission_rows(
    rbd, followed: list, rows_at, ends: np.ndarray, horizon: float
) -> Tuple[np.ndarray, np.ndarray]:
    """The capacity distribution averaged over each window ``[0, end)``
    (see ``mission_capacity``): ``rows_at(x)`` its levels and their
    probabilities at the times ``x``, integrated on pieces cut where
    the ``followed`` curves bend, and extended exactly past the time
    they have settled. Returns the levels and one row per level."""
    from repyability.rbd.repairable_rbd import _MISSION_POINTS, _settling

    settle, period = _settling(followed)
    reach = min(horizon, settle if period is None else settle + period)
    beyond = ends > reach
    fixed = [ends[~beyond]]
    cycles = rest = np.empty(0)
    if period is not None and beyond.any():
        cycles = np.floor((ends[beyond] - settle) / period)
        rest = np.clip(ends[beyond] - settle - cycles * period, 0.0, period)
        fixed += [np.array([settle]), settle + rest]

    def estimate(a: np.ndarray, b: np.ndarray) -> dict:
        # The probability of each capacity level, integrated.
        x, half = _quadrature.points(a, b)
        levels, rows = rows_at(x)
        return {
            float(level): _quadrature.summed(row, half)
            for level, row in zip(levels, rows)
        }

    try:
        edges, finest = _quadrature.pieces(
            followed, np.concatenate(fixed), reach, _MISSION_POINTS
        )
        edges, integrals = _quadrature.refined(
            estimate, edges, finest, _MISSION_POINTS
        )
    except _quadrature.TooMany as error:
        raise NotImplementedError(
            f"Integrating the capacity over [0, {reach}] takes "
            f"{error.count} pieces (the components' curves bend that "
            "often), more than the limit: estimate it by simulation, "
            "with availability(demand=...)."
        ) from None
    # The levels, as ``estimate`` keyed them.
    running: Dict[float, np.ndarray] = {
        float(level): _quadrature.running(values, len(edges) - 1)
        for level, values in integrals.items()
        if isinstance(level, float)
    }
    settled: Dict[float, float] = {}
    if period is None and beyond.any():
        levels, rows = rows_at(np.array([horizon]))
        settled = {float(v): float(r) for v, r in zip(levels, rows[:, 0])}
    inside = np.searchsorted(edges, ends[~beyond])
    positive = ends > 0.0
    at_zero: Dict[float, float] = {}
    if not positive.all():
        levels, rows = rows_at(np.zeros(1))
        at_zero = {float(v): float(r) for v, r in zip(levels, rows[:, 0])}
    averages = {}
    for level in sorted(set(running) | set(settled) | set(at_zero)):
        values = running.get(level, np.zeros(len(edges)))
        totals = np.empty(len(ends))
        totals[~beyond] = values[inside]
        if beyond.any():
            if period is None:
                totals[beyond] = values[-1] + (
                    ends[beyond] - reach
                ) * settled.get(level, 0.0)
            else:
                base = values[np.searchsorted(edges, settle)]
                partial = values[np.searchsorted(edges, settle + rest)]
                totals[beyond] = (
                    base + cycles * (values[-1] - base) + (partial - base)
                )
        average = np.empty(len(ends))
        average[positive] = totals[positive] / ends[positive]
        average[~positive] = at_zero.get(level, 0.0)
        averages[level] = average
    levels = np.array(sorted(averages), dtype=float)
    rows = np.array([averages[level] for level in levels]).reshape(
        len(levels), len(ends)
    )
    keep = np.any(rows != 0.0, axis=1)
    return levels[keep], rows[keep]


def _long_run_capacity(rbd, node) -> Tuple[np.ndarray, np.ndarray]:
    """The long-run distribution of the capacity of a node whose model
    has one (see ``_capacity_models``): a nested RBD's own; for a
    degrading component, down a fraction ``1 - A`` of the time and in
    each stage the rest of it in proportion to the stage's share of its
    working time."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    component = rbd.components[node]
    if isinstance(component, RepairableRBD):
        distribution = component.capacity_distribution()
        return distribution.levels, distribution.probabilities
    rbd._require_unscheduled_stages(node)
    stages = component.reliability
    up = _long_run._node_availability(rbd, node)
    shares = np.concatenate([[1.0 - up], up * stages.stage_fractions()])
    levels, rows = _capacity.merged(
        np.array((0.0,) + stages.capacities), shares[:, None]
    )
    return levels, rows[:, 0]
