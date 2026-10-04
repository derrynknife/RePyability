"""A repairable diagram's importance measures over time (#191): at times
from new (or from the components' current states), or over a window,
rather than in the long run.

Each measure is worked out from points at which the nodes are up with
given probabilities, as in the long run (see
``RepairableRBD._long_run_points``): a ratio measure is a ratio of the
points' averages (``RAW_i``, the system's unavailability with node ``i``
down over its unavailability, both averaged), not an average of ratios.
At a time ``t`` the points are the nodes' point availabilities then (one
point); with common-cause groups, with the groups' joint states then
(from their chains, see ``_ccf_chain.OverTime``), each group conditioned
on within its module as in the long run (see
``RepairableRBD._ccf_measure``); with limited repair crews, the states of
the crews' chain, each with its probability at ``t`` (see
``_chain_transient.Uniformized``). Over a window ``[0, T)`` the points are
those at the times of the window's quadrature (see ``_quadrature``), each
weighted by its share of the window, so that a measure is a ratio of the
system's means over the window, as ``mission_availability`` is its mean
availability.

With limited repair crews, the components depend on each other through
the repair queue, so the Birnbaum importance, the improvement potential
and the risk worths hold each node working and failed in the crews' chain,
solved without it (see ``RepairableRBD._crew_held_importance``): each is
the system's availability so held, at each time or over the window. The
criticality and Fussell-Vesely measures are averages over the chain's
states: each is ``p(t) v`` for a vector ``v`` over the states (a node's
being down and critical in each, say), from the chain over time.
"""

import math
from typing import Callable, Dict, List, NamedTuple, Tuple

import numpy as np

from . import _quadrature
from .rbd import RBD

#: The measures, by the name the public methods give them.
BIRNBAUM = "birnbaum"
IMPROVEMENT = "improvement"
RAW = "raw"
RRW = "rrw"
CRITICALITY = "criticality"
FUSSELL_VESELY = "fussell_vesely"

#: The measures the crews' chain holds each node for (the others are
#: averages over its states).
_HELD = (BIRNBAUM, IMPROVEMENT, RAW, RRW)


class Spec(NamedTuple):
    """A measure (one of the names above) and its options."""

    name: str
    kind: str = "failure"
    fv_type: str = "c"
    method: str = "exact"


def measure(
    rbd, spec: Spec, working_nodes, broken_nodes, *, x, window, state
) -> dict:
    """``spec``'s measure of each node of ``rbd`` at the times ``x``, or
    over the windows ``[0, window)``: floats for a single time or window,
    else arrays in its shape."""
    from .repairable_rbd import _check_times

    if x is not None and window is not None:
        raise ValueError(
            "Give the times x or a window's length, not both: x evaluates "
            "the measure at each time, window averages it over [0, window)."
        )
    working = set(working_nodes or ())
    broken = set(broken_nodes or ())
    rbd._validate_node_overrides(working, broken)
    states = rbd._states(state, working | broken)
    if x is not None:
        times = _check_times(x)
        values = _at(rbd, spec, times.ravel(), working, broken, states, state)
        return _shaped(values, x, times.shape)
    ends = _check_times(window)
    if np.any(ends <= 0.0):
        raise ValueError(
            f"A window must be a positive length of time, got {window!r}."
        )
    flat = ends.ravel()
    out: Dict = {}
    for k, end in enumerate(flat):
        values = _over(rbd, spec, float(end), working, broken, states, state)
        for node, value in values.items():
            out.setdefault(node, np.empty(len(flat)))[k] = float(
                np.ravel(value)[0]
            )
    return _shaped(out, window, ends.shape)


def _shaped(values: dict, given, shape) -> dict:
    """The values (arrays over the flattened times or windows) as floats
    for a single one, else in its shape."""
    if np.ndim(given) == 0:
        return {node: float(np.ravel(v)[0]) for node, v in values.items()}
    return {
        node: np.asarray(v, dtype=float).reshape(shape)
        for node, v in values.items()
    }


# -- the measures at given points -------------------------------------------


def _base(rbd, spec: Spec, p: dict, q: dict, weights) -> dict:
    """The measure of every node from the points' probabilities of working
    ``p`` and failing ``q``: each point's own (``weights`` None), or the
    ratio of their averages with ``weights`` (see ``RBD._birnbaum_importance``
    and the rest)."""
    if spec.name == BIRNBAUM:
        return RBD._birnbaum_importance(rbd, p, weights, q)
    if spec.name == IMPROVEMENT:
        return RBD._improvement_potential(rbd, p, weights, q)
    if spec.name == RAW:
        return RBD._risk_achievement_worth(rbd, p, weights, q)
    if spec.name == RRW:
        return RBD._risk_reduction_worth(rbd, p, weights, q)
    if spec.name == CRITICALITY:
        return RBD._criticality_importance(rbd, p, spec.kind, weights, q)
    return RBD._fussell_vesely(rbd, p, spec.fv_type, spec.method, weights, q)


def _grouped(rbd, spec: Spec, groups: list, p: dict, q: dict, x, weights):
    """The measure with common-cause groups (their chains over time,
    ``groups``, see ``_ccf_chain.OverTime``) at the times ``x``: averaged
    with ``weights``, or at each time without (see
    ``RepairableRBD._ccf_measure``)."""
    from . import _ccf_chain

    tables = [
        _ccf_chain.GroupStates(
            tuple(group.members), group.down, group.probabilities(x)
        )
        for group in groups
    ]
    for table in tables:
        for k, member in enumerate(table.members):
            p[member] = table.probabilities @ np.where(
                table.down[:, k], 0.0, 1.0
            )
            q[member] = table.probabilities @ np.where(
                table.down[:, k], 1.0, 0.0
            )
    return rbd._ccf_measure(
        spec.name,
        (p, q, tables),
        weights,
        spec.kind,
        spec.fv_type,
        spec.method,
    )


def _independent(rbd, curves: dict, x, working, broken) -> Tuple[dict, dict]:
    """The nodes' probabilities of working and of failing at the times
    ``x``, from their ``curves`` (the forced nodes held at 1 or 0)."""
    size = len(x)
    p = {node: np.asarray(c.at(x), dtype=float) for node, c in curves.items()}
    p = rbd._filled(p, size, working, broken)
    q = rbd._failures_with_overrides(
        {node: 1.0 - value for node, value in p.items()}, working, broken
    )
    return p, q


# -- at times ---------------------------------------------------------------


def _at(rbd, spec, times, working, broken, states, state) -> dict:
    """The measure of every node at each of ``times``, as arrays."""
    if rbd._crews_couple():
        if spec.name in _HELD:
            return _held(rbd, spec, working, broken, state, times=times)
        return _chain_measure(rbd, spec, working, broken, states, times=times)
    horizon = float(times.max()) if times.size else 0.0
    forced = working | broken
    if rbd.ccf_groups:
        grouped = rbd._groups_curve(horizon, working, broken, "p", states)
        p, q = _independent(rbd, grouped.curves, times, working, broken)
        values = _grouped(rbd, spec, grouped.system.groups, p, q, times, None)
        return {
            node: np.asarray(value, dtype=float)
            for node, value in values.items()
        }
    curves = rbd._availability_curves(horizon, forced, state=states)
    p, q = _independent(rbd, curves, times, working, broken)
    return {
        node: np.asarray(value, dtype=float)
        for node, value in _base(rbd, spec, p, q, None).items()
    }


# -- over a window ----------------------------------------------------------


def window_points(
    curves: list,
    end: float,
    integrands: Callable[[np.ndarray], Dict],
    limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Times and weights with which a function of the ``curves`` (see
    ``_point_availability``) is integrated over ``[0, end)``: the Gauss
    points of the pieces of ``_quadrature``, halved until the
    ``integrands`` (functions of the curves, by key, at given times) agree
    with their halves', each weighted by its share of its piece. As for
    ``mission_availability``, past the time the curves settle into a
    constant or a period they are not followed: the constant takes the
    rest of the window at one point, a period's points its number of
    repeats."""
    from .repairable_rbd import _settling

    settle, period = _settling(curves)
    reach = min(end, settle if period is None else settle + period)
    fixed = [np.array([reach])]
    cycles, rest = 0.0, 0.0
    if period is not None and end > reach:
        cycles = math.floor((end - settle) / period)
        rest = min(max(end - settle - cycles * period, 0.0), period)
        fixed += [np.array([settle, settle + rest])]

    def estimate(a, b):
        x, half = _quadrature.points(a, b)
        return {
            key: _quadrature.summed(values, half)
            for key, values in integrands(x).items()
        }

    try:
        edges, finest = _quadrature.pieces(
            curves, np.concatenate(fixed), reach, limit
        )
        edges, _ = _quadrature.refined(estimate, edges, finest, limit)
    except _quadrature.TooMany as error:
        raise NotImplementedError(
            f"Averaging the importance over [0, {end}] takes {error.count} "
            "pieces (the components' curves bend that often), more than "
            "the limit: evaluate it at times (x=) instead."
        ) from None
    a, b = edges[:-1], edges[1:]
    x, half = _quadrature.points(a, b)
    weights = (half[:, None] * _quadrature.WEIGHTS[None, :]).ravel()
    if end > reach:
        if period is None:
            # Constant from ``settle``: the rest of the window at one point.
            x = np.append(x, reach)
            weights = np.append(weights, end - reach)
        else:
            # Each repeat of the period counts its points once more, and
            # those before ``settle + rest`` once more still.
            middle = np.repeat(0.5 * (a + b), len(_quadrature.GAUSS))
            later = middle >= settle
            weights = np.where(
                later,
                weights * (cycles + (middle < settle + rest)),
                weights,
            )
    return x, weights


def _over(rbd, spec, end, working, broken, states, state) -> dict:
    """The measure of every node over the window ``[0, end)``."""
    from .repairable_rbd import _MISSION_POINTS

    if rbd._crews_couple():
        if spec.name in _HELD:
            return _held(rbd, spec, working, broken, state, window=end)
        return _chain_measure(rbd, spec, working, broken, states, end=end)
    forced = working | broken
    if rbd.ccf_groups:
        grouped = rbd._groups_curve(end, working, broken, "p", states)

        def system(x):
            return {"up": grouped.at(x)}

        x, weights = window_points([grouped], end, system, _MISSION_POINTS)
        p, q = _independent(rbd, grouped.curves, x, working, broken)
        return _grouped(
            rbd, spec, grouped.system.groups, p, q, x, weights / end
        )
    curves = rbd._availability_curves(end, forced, state=states)

    def integrands(x):
        # The system's availability and each node's Birnbaum importance:
        # the quantities every measure is made of.
        p, q = _independent(rbd, curves, x, working, broken)
        importance, works, _, _, _ = rbd._importances(p, q)
        return {"up": works, **{("node", n): v for n, v in importance.items()}}

    x, weights = window_points(
        list(curves.values()), end, integrands, _MISSION_POINTS
    )
    p, q = _independent(rbd, curves, x, working, broken)
    return _base(rbd, spec, p, q, weights / end)


# -- with limited repair crews ----------------------------------------------


def _without(state, node):
    """``state`` (as given to the public method) less ``node``'s own, for
    the node held working or failed."""
    if isinstance(state, dict):
        return {k: v for k, v in state.items() if k != node}
    return state


def _held(rbd, spec, working, broken, state, times=None, window=None):
    """With limited repair crews, the Birnbaum importance, improvement
    potential or risk worths of every node, at the ``times`` or over the
    ``window``, from the system's unavailability with each node held
    working and failed in the crews' chain (as in the long run, see
    ``RepairableRBD._crew_held_importance``)."""

    def unavailability(up, down, given):
        if times is not None:
            value = rbd.point_availability(times, up, down, state=given)
        else:
            value = rbd.mission_availability(window, up, down, state=given)
        return 1.0 - np.atleast_1d(np.asarray(value, dtype=float))

    base = unavailability(working, broken, state)
    out: Dict = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for node in rbd.nodes:
            given = _without(state, node)
            up = unavailability(working | {node}, broken - {node}, given)
            down = unavailability(working - {node}, broken | {node}, given)
            out[node] = {
                BIRNBAUM: down - up,
                IMPROVEMENT: base - up,
                RAW: down / base,
                RRW: base / up,
            }[spec.name]
    return out


def _chain_measure(rbd, spec, working, broken, states, times=None, end=None):
    """With limited repair crews, the criticality or Fussell-Vesely
    measure of every node, at the ``times`` or over ``[0, end)``: its
    numerator and denominator are averages over the crews' chain's states
    (as in the long run), so each is ``p(t) v`` for a vector ``v`` over
    the states, followed over time from the components' ``states``."""
    forced = frozenset(working | broken)
    nested = rbd._require_crew_over_time(forced)
    if nested:
        raise NotImplementedError(
            f"With {rbd.repair_crews} repair crew(s), the criticality and "
            "Fussell-Vesely measures over time are averages over the "
            "crews' chain's states, which take no nested RBD's own curve "
            f"in, as yet: node {nested[0]!r} is one. Their long-run values "
            "are worked out, and the Birnbaum importance, improvement "
            "potential and risk worths over time."
        )
    p, _ = rbd._chain_probabilities(working, broken)
    q = {node: 1.0 - value for node, value in p.items()}
    importance, works, fails, p, q = rbd._importances(p, q)
    size = len(works)
    if spec.name == CRITICALITY:
        member, system = (p, works) if spec.kind == "success" else (q, fails)
        numerators = {
            node: importance[node] * member[node] for node in rbd.nodes
        }
    else:
        numerators = rbd._fv_numerators(p, q, size, spec.fv_type, spec.method)
        system = fails
    nodes: List = list(rbd.nodes)
    columns = np.column_stack([system] + [numerators[n] for n in nodes])
    chain = rbd._crew_chain(forced)
    followed = rbd._uniformized(
        "The repair crews'",
        chain.generator,
        rbd._crew_start(chain, states),
        chain.probabilities,
        columns,
    )
    if times is not None:
        values = followed.values(times)
    else:
        values = followed.integrals(np.array([end])) / end
    total = values[:, 0]
    out: Dict = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, node in enumerate(nodes):
            share = values[:, k + 1] / total
            if spec.name == CRITICALITY:
                share = np.where(total > 0.0, share, np.nan)
            out[node] = share
    return out
