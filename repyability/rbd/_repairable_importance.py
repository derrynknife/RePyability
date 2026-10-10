"""A ``RepairableRBD``'s importance measures: Birnbaum's and the measures
built on it (improvement potential, risk achievement and reduction
worth, criticality, Fussell-Vesely), in the long run or at a time, and
the measures of change (``differential_importance``, ``joint_importance``,
``availability_rate``, ``barlow_proschan_importance``), with repair crews'
and common-cause groups' chains where they couple the components. The
methods of ``RepairableRBD`` of those names call these.
"""

import math
from collections.abc import Mapping
from typing import (
    Any,
    Collection,
    Dict,
    Hashable,
    List,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd import (
    _ccf_chain,
    _ccf_groups,
    _chain_transient,
    _importance_time,
    _rates,
    _sensitivity,
)
from repyability.rbd._common import (
    _squeeze_values,
)
from repyability.rbd.results import (
    RateBreakdown,
)
from repyability.utils.checks import (
    nonnegative_times,
    one_of,
)

#: The most that may jump at one instant with a hidden common-cause
#: group's tests among them (the Shapley value takes every subset).
_MAX_PLAYERS = 12


def _importance_probabilities(
    rbd, working_nodes, broken_nodes
) -> Tuple[dict, dict, np.ndarray, Optional[np.ndarray]]:
    """``_long_run_points`` for the importance measures: with limited
    repair crews the states of the crews' Markov chain, in each of
    which every node the crews work on is up or down for certain
    (#146). (With common-cause groups, see ``_ccf_long_run_measure``.)"""
    return rbd._long_run_points(working_nodes, broken_nodes)


def _importance_over_time(
    rbd,
    spec: "_importance_time.Spec",
    working_nodes,
    broken_nodes,
    x,
    window,
    state,
) -> dict:
    """An importance measure at the times ``x`` or over the window
    ``[0, window)`` (#191, see ``_importance_time``), from new or from
    the components' ``state``."""
    if x is None and window is None:
        raise ValueError(
            "state is where the components start from, for a measure at "
            "times (x) or over a window (window): give one of them. The "
            "long-run measure does not depend on it."
        )
    if spec.name == _importance_time.FUSSELL_VESELY:
        if spec.fv_type not in ("c", "p"):
            raise ValueError(
                "fv_type must be either 'c' (cut-set) or 'p' (path-set), "
                f"fv_type={spec.fv_type!r} was given."
            )
        one_of("method", spec.method, ("exact", "rare_event"))
    if spec.name == _importance_time.CRITICALITY and spec.kind not in (
        "failure",
        "success",
    ):
        raise ValueError(
            f"kind must be 'failure' or 'success', got {spec.kind!r}."
        )
    return _importance_time.measure(
        rbd,
        spec,
        working_nodes,
        broken_nodes,
        x=x,
        window=window,
        state=state,
    )


def _crew_held_importance(
    rbd, measure: str, working_nodes, broken_nodes
) -> dict:
    """With limited repair crews (#146), an importance measure of each
    node from the system's long-run unavailability with the node held
    working, ``U(1_i)``, and held failed, ``U(0_i)``: the crews' Markov
    chain solved without it, as held it needs no crew. ``"birnbaum"``
    is ``U(0_i) - U(1_i)``, ``"improvement"`` ``U - U(1_i)``, ``"raw"``
    ``U(0_i) / U`` and ``"rrw"`` ``U / U(1_i)``. A node already held
    is held the other way for its own measure."""
    working = set(working_nodes or ())
    broken = set(broken_nodes or ())
    rbd._validate_node_overrides(working, broken)
    base = np.float64(rbd.mean_unavailability(working, broken))
    out: dict = {}
    for node in rbd.nodes:
        up = np.float64(
            rbd.mean_unavailability(working | {node}, broken - {node})
        )
        down = np.float64(
            rbd.mean_unavailability(working - {node}, broken | {node})
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out[node] = {
                "birnbaum": down - up,
                "improvement": base - up,
                "raw": down / base,
                "rrw": base / up,
            }[measure]
    return _squeeze_values(out)


def _failure_importance(
    rbd,
    measure,
    name,
    of_probabilities,
    working,
    broken,
    x,
    window,
    state,
) -> dict[Any, Any]:
    """A measure of the nodes' importance to the system's failure
    (Birnbaum, improvement potential, RAW, RRW): over time where ``x``,
    ``window`` or ``state`` is given, through the common-cause groups'
    or the coupled crews' chains where there are any, and otherwise
    from the nodes' long-run availabilities (``of_probabilities``, the
    structure's measure of them)."""
    if x is not None or window is not None or state is not None:
        return _importance_over_time(
            rbd,
            _importance_time.Spec(measure, "failure"),
            working,
            broken,
            x,
            window,
            state,
        )
    if rbd.ccf_groups:
        return _ccf_groups._ccf_long_run_measure(rbd, name, working, broken)
    if rbd._crews_couple():
        return _crew_held_importance(rbd, name, working, broken)
    node_probabilities, node_failures, weights, _ = _importance_probabilities(
        rbd, working, broken
    )
    return _squeeze_values(
        of_probabilities(node_probabilities, weights, node_failures)
    )


def birnbaum_importance(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    x,
    window,
    state,
) -> dict[Any, Any]:
    """See ``RepairableRBD.birnbaum_importance``."""
    return _failure_importance(
        rbd,
        _importance_time.BIRNBAUM,
        "birnbaum",
        rbd._birnbaum_importance,
        working_nodes,
        broken_nodes,
        x,
        window,
        state,
    )


def improvement_potential(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    x,
    window,
    state,
) -> dict[Any, Any]:
    """See ``RepairableRBD.improvement_potential``."""
    return _failure_importance(
        rbd,
        _importance_time.IMPROVEMENT,
        "improvement",
        rbd._improvement_potential,
        working_nodes,
        broken_nodes,
        x,
        window,
        state,
    )


def risk_achievement_worth(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    x,
    window,
    state,
) -> dict[Any, Any]:
    """See ``RepairableRBD.risk_achievement_worth``."""
    return _failure_importance(
        rbd,
        _importance_time.RAW,
        "raw",
        rbd._risk_achievement_worth,
        working_nodes,
        broken_nodes,
        x,
        window,
        state,
    )


def risk_reduction_worth(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    x,
    window,
    state,
) -> dict[Any, Any]:
    """See ``RepairableRBD.risk_reduction_worth``."""
    return _failure_importance(
        rbd,
        _importance_time.RRW,
        "rrw",
        rbd._risk_reduction_worth,
        working_nodes,
        broken_nodes,
        x,
        window,
        state,
    )


def criticality_importance(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    kind: str,
    *,
    x,
    window,
    state,
) -> dict[Any, Any]:
    """See ``RepairableRBD.criticality_importance``."""
    if x is not None or window is not None or state is not None:
        return _importance_over_time(
            rbd,
            _importance_time.Spec(_importance_time.CRITICALITY, kind),
            working_nodes,
            broken_nodes,
            x,
            window,
            state,
        )
    if rbd.ccf_groups:
        return _ccf_groups._ccf_long_run_measure(
            rbd, "criticality", working_nodes, broken_nodes, kind=kind
        )
    node_probabilities, node_failures, weights, _ = _importance_probabilities(
        rbd, working_nodes, broken_nodes
    )
    return _squeeze_values(
        rbd._criticality_importance(
            node_probabilities,
            kind,
            weights=weights,
            node_failures=node_failures,
        )
    )


def fussell_vesely(
    rbd,
    fv_type: str,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    *,
    x,
    window,
    state,
) -> dict[Any, Any]:
    """See ``RepairableRBD.fussell_vesely``."""
    if x is not None or window is not None or state is not None:
        return _importance_over_time(
            rbd,
            _importance_time.Spec(
                _importance_time.FUSSELL_VESELY,
                "failure",
                fv_type=fv_type,
                method=method,
            ),
            working_nodes,
            broken_nodes,
            x,
            window,
            state,
        )
    if rbd.ccf_groups:
        return _ccf_groups._ccf_long_run_measure(
            rbd,
            "fussell_vesely",
            working_nodes,
            broken_nodes,
            fv_type=fv_type,
            method=method,
        )
    node_probabilities, node_failures, weights, _ = _importance_probabilities(
        rbd, working_nodes, broken_nodes
    )
    return _squeeze_values(
        rbd._fussell_vesely(
            node_probabilities,
            fv_type,
            method,
            weights=weights,
            node_failures=node_failures,
        )
    )


def differential_importance(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    x,
    window,
    state,
    over: str,
    change: str,
    kind: str,
    improving: bool,
    groups: Optional[Mapping[Hashable, Collection[Hashable]]],
    rel_step: Optional[float],
) -> dict:
    """See ``RepairableRBD.differential_importance``."""
    from ._differential import (
        check,
        flattened,
        proportional_scales,
        shares,
    )

    check(over, change, kind)
    held = set(working_nodes or ()) | set(broken_nodes or ())
    timed = {
        key: value
        for key, value in (("x", x), ("window", window), ("state", state))
        if value is not None
    }
    if over == "components":
        if change == "uniform":
            values = rbd.birnbaum_importance(
                working_nodes, broken_nodes, **timed
            )
        else:
            values = rbd.criticality_importance(
                working_nodes, broken_nodes, kind, **timed
            )
        contributions = {
            node: (0.0 * np.asarray(v) if node in held else v)
            for node, v in values.items()
        }
    else:
        derivatives = flattened(
            rbd.parameter_sensitivity(
                working_nodes,
                broken_nodes,
                x=x,
                window=window,
                state=state,
                rel_step=rel_step,
            )
        )
        levers = [
            lever for lever in _sensitivity._levers(rbd) if not lever.discrete
        ]
        continuous = {(lever.key, lever.name) for lever in levers}
        kept = [key for key in derivatives if key in continuous]
        scale = (
            proportional_scales(levers, kept)
            if change == "proportional"
            else dict.fromkeys(kept, 1.0)
        )
        contributions = {
            key: np.asarray(derivatives[key], dtype=float) * scale[key]
            for key in kept
        }
    scalar = x is None or np.ndim(x) == 0
    return shares(contributions, groups, scalar, improving)


def _independent_rates(
    rbd, flat, horizon: float, scale: float, working, broken, states
) -> Tuple[dict, np.ndarray, dict]:
    """``availability_rate``'s parts with independent components: each
    component's Birnbaum importance times its own rate, the times the
    components jump, and each one's part in each jump (see
    ``_rates``)."""
    # A little past the last time, for the differences there.
    curves = rbd._availability_curves(
        1.01 * scale, working | broken, state=states
    )

    def importances(values: dict) -> dict:
        size = len(next(iter(values.values()))) if values else len(flat)
        return rbd._importances(rbd._filled(values, size, working, broken))[0]

    importance = importances(
        {node: curve.at(flat) for node, curve in curves.items()}
    )
    parts = {
        node: importance[node] * _rates.derivative(curves[node], flat, scale)
        for node in curves
    }
    jump_times, before, after = _rates.jumps(curves, horizon)
    return (
        parts,
        jump_times,
        _rates.split_jumps(importances, before, after),
    )


def _crew_rates(
    rbd, flat, horizon: float, scale: float, working, broken, states
) -> Tuple[dict, np.ndarray, dict]:
    """``availability_rate``'s parts with limited repair crews (#199).
    The system is up with probability ``sum_m p(t) u_m w_m(t)`` (see
    ``_chain_transient.CrewCurve``), so its rate is ``sum_m p(t) Q u_m
    w_m(t)`` plus ``sum_m p(t) u_m w_m'(t)``. Split by the component
    whose failures and repairs make each transition, ``Q = sum_i Q_i``
    (``_crew_chain.CrewChain.split``), the first is the chain's
    components' parts, each ``p(t) Q_i u_m`` one more vector of the
    crews' uniformized chain, exact; the second, the nested RBDs', each
    one's Birnbaum importance over the chain times its own rate, as for
    independent components, and their jumps are split likewise."""
    nested = rbd._require_crew_over_time(frozenset(working | broken))
    chain = rbd._crew_chain(frozenset(working | broken))
    size, count = len(chain.probabilities), len(chain.nodes)
    patterns = 2 ** len(nested)
    ups = [
        rbd._crew_vectors(
            chain,
            {
                node: np.full(size, float((m >> j) & 1))
                for j, node in enumerate(nested)
            },
            working,
            broken,
            "p",
        )[0]
        for m in range(patterns)
    ]
    uniformized = rbd._uniformized(
        "The repair crews'",
        chain.generator,
        rbd._crew_start(chain, states),
        chain.probabilities,
        np.column_stack(ups + [chain.split(up) for up in ups]),
    )
    curves = {
        node: rbd.components[node]._nested_curve(
            1.01 * scale, start=states.get(node)
        )
        for node in nested
    }

    def importances(values: dict, times: np.ndarray) -> dict:
        columns = uniformized.values(times)[:, :patterns]
        return dict(
            zip(
                nested,
                _chain_transient.pattern_importances(
                    columns, [values[node] for node in nested]
                ),
            )
        )

    columns = uniformized.values(flat)
    available = {
        node: np.asarray(curve.at(flat), dtype=float)
        for node, curve in curves.items()
    }
    weights = _chain_transient.pattern_weights(
        [available[node] for node in nested]
    )
    parts: Dict[Hashable, np.ndarray] = {}
    for k, node in enumerate(chain.nodes):
        own = columns[:, patterns + k + count * np.arange(patterns)]
        parts[node] = np.sum(own * weights, axis=1)
    importance = importances(available, flat)
    for node in nested:
        parts[node] = importance[node] * _rates.derivative(
            curves[node], flat, scale
        )
    jump_times, before, after = _rates.jumps(curves, horizon)

    def along(values: dict) -> dict:
        # Each jump's time, for each point of the path between the
        # values either side of it.
        points = len(next(iter(values.values()))) // len(jump_times)
        return importances(values, np.repeat(jump_times, points))

    return parts, jump_times, _rates.split_jumps(along, before, after)


def _groups_rates(
    rbd, flat, horizon: float, scale: float, working, broken, states
) -> Tuple[dict, np.ndarray, dict]:
    """``availability_rate``'s parts with common-cause groups (#199).
    The nodes outside the groups are independent of the groups and of
    each other, so each one's part is its Birnbaum importance (over the
    groups' joint states, see ``_ccf_chain.GroupsSystem``) times its own
    rate. A group's members' states move by its chain, ``p_g'(t) =
    p_g(t) Q_g``, so its part is ``p_g(t) Q_g a_g(t)``, ``a_g(t)`` the
    system's availability with the group in each of its states then.
    Split by whose each move is (``_ccf_chain.Move``), ``Q_g`` gives
    each member the part of its own cause and its repairs, and the
    group, under the tuple of its members, that of its causes that
    strike more than one. The jumps are split by ``_groups_jumps``."""
    _ccf_groups._require_free_members(rbd, working, broken)
    _ccf_groups._require_groups_over_time(rbd, states)
    curve = _ccf_groups._groups_curve(
        rbd, 1.01 * scale, working, broken, "p", states
    )
    curves, system = curve.curves, curve.system
    values = {
        node: np.asarray(c.at(flat), dtype=float) for node, c in curves.items()
    }
    importance = system.importance(values, flat)
    parts: Dict[Any, np.ndarray] = {
        node: importance[node] * _rates.derivative(c, flat, scale)
        for node, c in curves.items()
    }
    for number, group in enumerate(system.groups):
        # The system's unavailability with the group in each of its
        # combinations, and the group's chain's states.
        down = system.given(values, flat, number)
        chances = group.raw(flat)
        for label, begin, end, rate in group.moves:
            key = (
                group.members[label]
                if label != _ccf_chain.SHARED
                else tuple(group.members)
            )
            fall = (
                down[:, group.combination[end]]
                - down[:, group.combination[begin]]
            )
            part = -np.sum(chances[:, begin] * rate * fall, axis=1)
            parts[key] = parts.get(key, 0.0) + part
    jump_times, split = _groups_jumps(rbd, curves, system, horizon)
    return parts, jump_times, split


def _groups_jumps(
    rbd, curves: dict, system, horizon: float
) -> Tuple[np.ndarray, dict]:
    """The times in ``(0, horizon]`` at which the system with
    common-cause groups jumps (a scheduled event of a node outside the
    groups, a hidden group's tests), and each one's part in each jump.
    Where only nodes outside the groups jump, independent and the
    system multilinear in them, ``_rates.split_jumps`` splits it. Where
    a test is among what jumps, the members' states are no longer
    independent, and the jump is split by the Shapley value of what
    jumps then (the nodes, and each test, under its member's name),
    the system worked out exactly with each subset of them done: for
    independent nodes alone, the same split."""
    out_times, before, after = _rates.jumps(curves, horizon)
    tests = [group.tests(horizon) for group in system.groups]
    tested = sorted({t for by in tests for t in by})
    times = np.unique(np.concatenate([out_times, np.array(tested)]))
    split: Dict[Any, np.ndarray] = {}

    def part(key) -> np.ndarray:
        if key not in split:
            split[key] = np.zeros(len(times))
        return split[key]

    alone = np.array([t not in set(tested) for t in out_times], bool)
    if alone.any():
        at = out_times[alone]

        def importances(values: dict) -> dict:
            points = len(next(iter(values.values()))) // len(at)
            return system.importance(values, np.repeat(at, points))

        shares = _rates.split_jumps(
            importances,
            {node: v[alone] for node, v in before.items()},
            {node: v[alone] for node, v in after.items()},
        )
        rows = np.searchsorted(times, at)
        for node, values in shares.items():
            part(node)[rows] += values
    for time in tested:
        row = int(np.searchsorted(times, time))
        just = np.nextafter(time, -np.inf)
        low = {
            node: float(np.ravel(c.at(np.array([just])))[0])
            for node, c in curves.items()
        }
        high = {
            node: float(np.ravel(c.at(np.array([time])))[0])
            for node, c in curves.items()
        }
        players: List[Tuple[Any, Any]] = [
            (node, None)
            for node in curves
            if abs(high[node] - low[node]) > _rates.JUMP
        ]
        for number, by in enumerate(tests):
            for test in by.get(time, []):
                players.append((number, test))
        for (who, test), value in zip(
            players, _shapley(rbd, system, players, low, high, time)
        ):
            key = (
                who
                if test is None
                else system.groups[who].members[test.member]
            )
            part(key)[row] += value
    return times, split


def _shapley(
    rbd, system, players: list, low: dict, high: dict, time: float
) -> np.ndarray:
    """Each player's Shapley value in the system's jump at ``time``
    (see ``_groups_jumps``): the system's availability with each
    subset of the players done, a node at its value after the jump
    (``high``) rather than before (``low``), and a test applied to its
    group's states just before the time."""
    count = len(players)
    if count > _MAX_PLAYERS:
        raise NotImplementedError(
            f"{count} nodes and tests change at time {time:g}, with a "
            "common-cause group's tests among them: their jump is split "
            f"for at most {_MAX_PLAYERS}, as yet. Simulate it with "
            "availability()."
        )
    subsets = np.arange(2**count)
    values = {}
    for node in low:
        if (node, None) in players:
            j = players.index((node, None))
            values[node] = np.where(
                (subsets >> j) & 1, high[node], low[node]
            ).astype(float)
        else:
            values[node] = np.full(len(subsets), high[node])
    chances = []
    just = np.array([np.nextafter(time, -np.inf)])
    for number, group in enumerate(system.groups):
        mine = [
            (j, test)
            for j, (who, test) in enumerate(players)
            if test is not None and who == number
        ]
        if not mine:
            chances.append(
                np.repeat(
                    group.probabilities(np.array([time])), len(subsets), 0
                )
            )
            continue
        start = group.raw(just)
        table = np.empty((len(subsets), len(group.down)))
        for subset in subsets:
            done = [test for j, test in mine if (subset >> j) & 1]
            table[subset] = group.grouped(group.tested(start, done))[0]
        chances.append(table)
    worth = system.availability(values, chances)
    out = np.zeros(count)
    weight = [
        math.factorial(k)
        * math.factorial(count - k - 1)
        / math.factorial(count)
        for k in range(count)
    ]
    sizes = np.array([bin(int(s)).count("1") for s in subsets])
    for j in range(count):
        without = subsets[(subsets >> j) & 1 == 0]
        out[j] = sum(
            weight[sizes[s]] * (worth[s | (1 << j)] - worth[s])
            for s in without
        )
    return out


def availability_rate(
    rbd,
    x,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    state,
) -> RateBreakdown:
    """See ``RepairableRBD.availability_rate``."""
    times = nonnegative_times(x)
    working = set() if working_nodes is None else set(working_nodes)
    broken = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working, broken)
    forced = working | broken
    states = rbd._states(state, forced)
    flat = times.ravel()
    horizon = float(flat.max()) if flat.size else 0.0
    scale = horizon if horizon > 0.0 else 1.0
    if rbd._crews_couple():
        parts, jump_times, split = _crew_rates(
            rbd, flat, horizon, scale, working, broken, states
        )
    elif rbd.ccf_groups:
        parts, jump_times, split = _groups_rates(
            rbd, flat, horizon, scale, working, broken, states
        )
    else:
        parts, jump_times, split = _independent_rates(
            rbd, flat, horizon, scale, working, broken, states
        )
    keys = list(rbd.components) + [
        key for key in {**parts, **split} if key not in rbd.components
    ]
    node_rate: Dict[Hashable, np.ndarray] = {}
    rate = np.zeros(len(flat))
    node_jumps: Dict[Hashable, np.ndarray] = {}
    jumps = np.zeros(len(jump_times))
    for key in keys:
        part = np.asarray(parts.get(key, np.zeros(len(flat))), dtype=float)
        node_rate[key] = part
        rate = rate + part
        part = np.asarray(
            split.get(key, np.zeros(len(jump_times))), dtype=float
        )
        node_jumps[key] = part
        jumps = jumps + part

    def shaped(values: np.ndarray):
        if np.ndim(x) == 0:
            return float(values[0])
        return values.reshape(times.shape)

    return RateBreakdown(
        x=shaped(flat),
        rate=shaped(rate),
        node_rate={node: shaped(v) for node, v in node_rate.items()},
        jump_times=jump_times,
        jumps=jumps,
        node_jumps=node_jumps,
    )


def barlow_proschan_importance(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    window,
    state,
) -> dict:
    """See ``RepairableRBD.barlow_proschan_importance``."""
    if window is None:
        if state is not None:
            raise ValueError(
                "state is where the components start from, for the "
                "shares over a window: give its length (window). The "
                "long-run shares do not depend on it."
            )
        terms, _ = rbd._outage_terms(working_nodes, broken_nodes)
        parts: Dict[Any, Any] = {node: 0.0 for node in rbd.components}
        for cause, term in terms:
            parts[cause] = parts.get(cause, 0.0) + term
        scalar = True
    else:
        ends = nonnegative_times(window)
        if np.any(ends <= 0.0):
            raise ValueError(
                "A window must be a positive length of time, got "
                f"{window!r}."
            )
        windows, flat, _, counts, _, _ = rbd._window(
            window,
            working_nodes,
            broken_nodes,
            "p",
            state=state,
            causes=True,
        )
        caused = counts["caused"]
        parts = {
            key: np.asarray(caused.get(key, np.zeros(len(flat))), dtype=float)
            for key in [*rbd.components, *caused]
        }
        scalar = np.ndim(window) == 0
    total: Any = 0.0
    for value in parts.values():
        total = total + np.asarray(value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = {
            key: np.where(
                total > 0.0, np.asarray(value, dtype=float) / total, np.nan
            )
            for key, value in parts.items()
        }
    if scalar:
        return {
            key: float(np.ravel(value)[0]) for key, value in shares.items()
        }
    return {
        key: np.asarray(value).reshape(windows.shape)
        for key, value in shares.items()
    }


def joint_importance(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    x,
    window,
    state,
) -> dict:
    """See ``RepairableRBD.joint_importance``."""
    timed = {
        key: value
        for key, value in (("x", x), ("window", window), ("state", state))
        if value is not None
    }
    with rbd._sharing_curves():
        return rbd._joint_pairs(
            lambda working, broken: rbd.birnbaum_importance(
                sorted(working, key=str), sorted(broken, key=str), **timed
            ),
            working_nodes,
            broken_nodes,
        )
