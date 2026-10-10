"""A ``RepairableRBD``'s values over time from its components' curves: the
availability and unavailability at points (``point_availability``,
``point_unavailability``) and over missions (``mission_availability``,
``mission_unavailability``), and the expected failures, events and cost
in a window (``expected_failures``, ``expected_events``,
``expected_cost``), integrated on coarse pieces (``_quadrature``). The
methods of ``RepairableRBD`` of those names call these.
"""

import math
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
)

import numpy as np

from repyability.rbd import (
    _ccf_groups,
    _quadrature,
    _rates,
)
from repyability.rbd._common import (
    _DISCOUNT_ROUNDS,
    _DISCOUNT_TOLERANCE,
    _discount_pieces,
    _discount_rate,
    _horizons,
    _matched,
    _present_horizon,
    _shaped,
)
from repyability.rbd._point_availability import (
    Atoms,
)
from repyability.rbd._spec import _mean_cost
from repyability.rbd._tally import (
    _CATEGORIES,
)
from repyability.rbd.results import (
    ExpectedCost,
    ExpectedEvents,
)
from repyability.utils.checks import (
    nonnegative_times,
    structure_method,
)

from ._mean_lifetime import _G7_W, _GK_W, _GK_X

if TYPE_CHECKING:
    pass


class _Jumps(NamedTuple):
    """What the system does at the times its nodes fail or are taken down
    at exact times (see ``RepairableRBD._atom_groups``): the times, the
    probability at each that the system fails there, and that a planned
    outage takes it down there, and how much its point availability there
    falls with them (``drop``); if asked for, each node's part in the
    failures (``caused``)."""

    times: np.ndarray
    failures: np.ndarray
    planned: np.ndarray
    drop: np.ndarray
    caused: Optional[dict] = None


def point_availability(
    rbd,
    x,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
):
    """See ``RepairableRBD.point_availability``."""
    return _point(rbd, x, working_nodes, broken_nodes, method, state)


def point_unavailability(
    rbd,
    x,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
):
    """See ``RepairableRBD.point_unavailability``."""
    return _point(
        rbd, x, working_nodes, broken_nodes, method, state, down=True
    )


def _point(rbd, x, working_nodes, broken_nodes, method, state, down=False):
    """``point_availability``, or with ``down``, ``point_unavailability``
    (#237)."""
    times = nonnegative_times(x)
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    horizon = float(times.max()) if times.size else 0.0
    states = rbd._states(state, working_nodes | broken_nodes)
    system_at, _, _ = _system_at(
        rbd, horizon, working_nodes, broken_nodes, method, states, down
    )
    values = system_at(times.ravel())
    if np.ndim(x) == 0:
        return float(values[0])
    return values.reshape(times.shape)


def _system_at(
    rbd, horizon, working_nodes, broken_nodes, method, states, down=False
):
    """The system's point availability (with ``down``, unavailability)
    over ``[0, horizon]``, as a function of the times; the curves whose
    bends its integral's pieces follow; and the crews' curve, where a
    component can wait for a crew (else None)."""
    if rbd._crews_couple():
        crew = rbd._crew_curve(
            horizon, working_nodes, broken_nodes, method, states
        )
        if not down:
            return crew.at, [crew], crew

        def crew_down(x):
            return np.clip(1.0 - crew.at(x), 0.0, 1.0)

        return crew_down, [crew], crew
    if rbd.ccf_groups:
        grouped = _ccf_groups._groups_curve(
            rbd, horizon, working_nodes, broken_nodes, method, states
        )
        return (grouped.down_at if down else grouped.at), [grouped], None
    forced = working_nodes | broken_nodes
    curves = rbd._availability_curves(horizon, forced, state=states)

    def system_at(x):
        if down:
            return rbd._curves_down_at(
                curves, x, working_nodes, broken_nodes, states
            )
        return rbd._curves_at(curves, x, working_nodes, broken_nodes, method)

    return system_at, list(curves.values()), None


def mission_availability(
    rbd,
    t,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
):
    """See ``RepairableRBD.mission_availability``."""
    return _mission(rbd, t, working_nodes, broken_nodes, method, state)


def mission_unavailability(
    rbd,
    t,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
):
    """See ``RepairableRBD.mission_unavailability``."""
    return _mission(
        rbd, t, working_nodes, broken_nodes, method, state, down=True
    )


def _mission(rbd, t, working_nodes, broken_nodes, method, state, down=False):
    """``mission_availability``, or with ``down``,
    ``mission_unavailability`` (#237)."""
    windows = nonnegative_times(t)
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    ends = windows.ravel()
    horizon = float(ends.max()) if ends.size else 0.0
    states = rbd._states(state, working_nodes | broken_nodes)
    system_at, pieces, crew = _system_at(
        rbd, horizon, working_nodes, broken_nodes, method, states, down
    )
    if crew is not None and not crew.nested:
        # The crews' chain integrates it exactly.
        totals = crew.integral(ends)
        if down:
            totals = np.maximum(ends - totals, 0.0)
    else:
        scale = None
        if down:
            # To the unavailability's own size, and the time scale of
            # the fastest closed form in it (see _closed_form_down); a
            # unit repaired at once is never down, and sets none.
            scale = (
                min(
                    [horizon or 1.0]
                    + [
                        1.0 / rate
                        for rate in rbd._closed_form_rates(states).values()
                        if 0.0 < rate < math.inf
                    ]
                )
                / 4.0
            )
        totals = _integrated(rbd, system_at, pieces, ends, horizon, scale)
    averages = np.empty(len(ends))
    positive = ends > 0.0
    averages[positive] = totals[positive] / ends[positive]
    if not positive.all():
        averages[~positive] = system_at(np.zeros(1))[0]
    if np.ndim(t) == 0:
        return float(averages[0])
    return averages.reshape(windows.shape)


def _integrated(
    rbd,
    system_at,
    curves: list,
    ends: np.ndarray,
    horizon: float,
    relative: Optional[float] = None,
) -> np.ndarray:
    """The integral from 0 to each of ``ends`` of the system's point
    availability, ``system_at``, from its nodes' ``curves`` (over the
    pieces of ``_quadrature``, and settling into a constant or a period
    it is extended over: see ``mission_availability``). With
    ``relative``, a time scale, to a tolerance relative to the integral
    itself, pieces being halved down to that scale where the grids are
    coarser: for an unavailability, small, and changing in closed form
    faster than the grids (#237)."""
    from repyability.rbd.repairable_rbd import _MISSION_POINTS, _settling

    # After ``settle`` the system's availability is constant, or repeats
    # with ``period``: it is integrated up to ``reach``, and extended.
    settle, period = _settling(curves)
    reach = min(horizon, settle if period is None else settle + period)
    beyond = ends > reach
    fixed = [ends[~beyond]]
    if period is not None and beyond.any():
        cycles = np.floor((ends[beyond] - settle) / period)
        rest = ends[beyond] - settle - cycles * period
        rest = np.clip(rest, 0.0, period)
        fixed += [np.array([settle]), settle + rest]

    def estimate(a, b):
        x, half = _quadrature.points(a, b)
        return {"up": _quadrature.summed(system_at(x), half)}

    try:
        edges, finest = _quadrature.pieces(
            curves, np.concatenate(fixed), reach, _MISSION_POINTS
        )
        keys: tuple = ()
        if relative is not None:
            keys, finest = ("up",), np.minimum(finest, relative)
        edges, integrals = _quadrature.refined(
            estimate, edges, finest, _MISSION_POINTS, keys
        )
    except _quadrature.TooMany as error:
        raise NotImplementedError(
            f"Integrating the availability over [0, {reach}] takes "
            f"{error.count} pieces (the components' curves bend that "
            "often), more than the limit: estimate it by simulation, "
            "with availability()."
        ) from None
    running = _quadrature.running(integrals.get("up"), len(edges) - 1)

    def uptime(x):
        return running[np.searchsorted(edges, x)]

    totals = np.empty(len(ends))
    totals[~beyond] = uptime(ends[~beyond])
    if beyond.any():
        if period is None:
            level = system_at(np.array([horizon]))[0]
            totals[beyond] = uptime(reach) + (ends[beyond] - reach) * level
        else:
            base = uptime(settle)
            totals[beyond] = (
                base
                + cycles * (uptime(reach) - base)
                + (uptime(settle + rest) - base)
            )
    return totals


def _window(
    rbd,
    t,
    working_nodes,
    broken_nodes,
    method: str,
    nodes: bool = False,
    setups: bool = False,
    state=None,
    causes: bool = False,
):
    """The windows ``t`` checked, flat, and the nodes' curves and the
    system's expected events over them (see ``_window_counts``): the
    forced nodes have no curves. With ``setups``, the stops of each
    maintenance group with a set-up cost are counted too; with
    ``causes``, the system failures each node (or common cause)
    causes."""
    method = structure_method(method)
    windows = nonnegative_times(t)
    working = set() if working_nodes is None else set(working_nodes)
    broken = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working, broken)
    forced = working | broken
    ends = windows.ravel()
    horizon = float(ends.max()) if ends.size else 0.0
    states = rbd._states(state, forced)
    groups = None
    if setups:
        groups = {
            name: [node for node in spec.members if node not in forced]
            for name, spec in rbd._maintenance.items()
            if spec.setup_cost
        }
    if rbd._crews_couple():
        curves, counts = rbd._crew_window(
            ends, working, broken, method, states, nodes, groups, causes
        )
        return windows, ends, curves, counts, working, broken
    if rbd.ccf_groups:
        curves, counts = rbd._groups_window(
            ends, working, broken, method, states, nodes, groups, causes
        )
        return windows, ends, curves, counts, working, broken
    curves = rbd._availability_curves(
        horizon, forced, counts=True, state=states
    )
    counts = _window_counts(
        rbd,
        curves,
        ends,
        working,
        broken,
        method,
        nodes=nodes,
        groups=groups,
        causes=causes,
    )
    return windows, ends, curves, counts, working, broken


def expected_failures(
    rbd,
    t,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
):
    """See ``RepairableRBD.expected_failures``."""
    windows, _, _, counts, _, _ = _window(
        rbd, t, working_nodes, broken_nodes, method, state=state
    )
    return _shaped(counts["failures"], t, windows)


def expected_events(
    rbd,
    t,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
) -> ExpectedEvents:
    """See ``RepairableRBD.expected_events``."""
    windows, ends, curves, counts, working, broken = _window(
        rbd, t, working_nodes, broken_nodes, method, nodes=True, state=state
    )
    node_events = {node: curve.events(ends) for node, curve in curves.items()}
    zero = np.zeros(len(ends))

    def shaped(values) -> Any:
        return _shaped(values, t, windows)

    def per_node(key: str) -> dict:
        return {
            node: shaped(node_events[node][key] if node in curves else zero)
            for node in rbd.components
        }

    downtime = {
        node: shaped(
            counts["downtime"][node]
            if node in curves
            else (ends if node in broken else zero)
        )
        for node in rbd.components
    }
    return ExpectedEvents(
        window=shaped(ends),
        system_failures=shaped(counts["failures"]),
        system_planned_outages=shaped(counts["planned"]),
        system_downtime=shaped(np.maximum(ends - counts["uptime"], 0.0)),
        node_failures=per_node("failures"),
        node_corrective=per_node("corrective"),
        node_preventive=per_node("preventive"),
        node_inspections=per_node("inspections"),
        node_downtime=downtime,
    )


def expected_cost(
    rbd,
    t,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    state,
    *,
    discount_rate: float,
) -> ExpectedCost:
    """See ``RepairableRBD.expected_cost``."""
    rate = _discount_rate(discount_rate)
    if rate > 0.0 and rbd.has_costs:
        _present_horizon(
            _horizons(t, 0.0), rate, rbd._mean_lives
        )  # warns of a rate that discounts the costs away
        return _discounted_cost(
            rbd, t, rate, working_nodes, broken_nodes, method, state
        )
    if not rbd.has_costs:
        windows = nonnegative_times(t)
        zero = _shaped(np.zeros(windows.size), t, windows)
        return ExpectedCost(
            window=_shaped(windows.ravel(), t, windows),
            mean=zero,
            by_category={category: zero for category in _CATEGORIES},
            by_component={},
            acquisition_cost=rbd.acquisition_cost,
        )
    windows, ends, curves, counts, working, broken = _window(
        rbd,
        t,
        working_nodes,
        broken_nodes,
        method,
        nodes=True,
        setups=True,
        state=state,
    )
    node_events = {node: curve.events(ends) for node, curve in curves.items()}
    categories = {category: np.zeros(len(ends)) for category in _CATEGORIES}
    by_component: Dict[Hashable, Any] = {}
    for node, node_costs in rbd.costs.items():
        own = np.zeros(len(ends))
        charges = []
        events = node_events.get(node)
        if events is not None:
            for key, category, kind in (
                ("repair_cost", "repair", "corrective"),
                ("replace_cost", "replace", "corrective"),
                ("preventive_cost", "preventive", "preventive"),
                ("inspection_cost", "inspection", "inspections"),
            ):
                if category == "replace" and node in rbd._imperfect:
                    # Minimally repaired (the only imperfect repair
                    # with exact values), it is never replaced: its
                    # repairs are charged only as repairs, as the
                    # simulation charges them.
                    continue
                if key in node_costs:
                    charges.append(
                        (
                            category,
                            _mean_cost(node_costs[key]) * events[kind],
                        )
                    )
        rate = node_costs.get("downtime_cost", 0.0)
        if rate:
            down = (
                counts["downtime"][node]
                if node in curves
                else (ends if node in broken else np.zeros(len(ends)))
            )
            charges.append(("component_downtime", rate * down))
        for category, charge in charges:
            categories[category] = categories[category] + charge
            own = own + charge
        by_component[node] = _shaped(own, t, windows)
    if rbd.downtime_cost_rate:
        categories["system_downtime"] = rbd.downtime_cost_rate * (
            np.maximum(ends - counts["uptime"], 0.0)
        )
    for name, spec in rbd._maintenance.items():
        if not spec.setup_cost:
            continue
        # Each failure or replacement of a member is a stop, but those
        # at one instant are one.
        stops = -counts["overlaps"].get(name, np.zeros(len(ends)))
        for node in spec.members:
            if node in node_events:
                stops = stops + (
                    node_events[node]["failures"]
                    + node_events[node]["preventive"]
                )
        categories["setup"] = categories["setup"] + spec.setup_cost * stops
    mean = np.zeros(len(ends))
    for values in categories.values():
        mean = mean + values
    return ExpectedCost(
        window=_shaped(ends, t, windows),
        mean=_shaped(mean, t, windows),
        by_category={
            category: _shaped(values, t, windows)
            for category, values in categories.items()
        },
        by_component=by_component,
        acquisition_cost=rbd.acquisition_cost,
    )


def _discounted_cost(
    rbd, t, rate: float, working_nodes, broken_nodes, method: str, state
) -> ExpectedCost:
    """``expected_cost`` at ``t`` discounted at ``rate`` (#231): each
    window's ``exp(-r T) C(T) + r * integral_0^T exp(-r s) C(s) ds``,
    ``C`` the expected cost from new, the integral by the Gauss-Kronrod
    (7, 15) rule on pieces from 0, with the windows among their ends.
    The pieces whose two rules differ most are halved until the
    differences add up to ``_DISCOUNT_TOLERANCE`` of the integral:
    ``C`` jumps at scheduled events (tests, block replacements), where
    a piece's error only halves with it."""
    windows = nonnegative_times(t)
    ends = windows.ravel()
    horizon = float(ends.max()) if ends.size else 0.0
    keys: list = []

    def values(times: np.ndarray) -> np.ndarray:
        """``C`` at ``times``: its mean, each category's and each
        component's, as rows. The longest window is worked out too, so
        that every call's curves reach it, and are shared."""
        got = rbd.expected_cost(
            np.append(times, horizon),
            working_nodes,
            broken_nodes,
            method,
            state,
        )
        if not keys:
            keys.extend(got.by_component)
        rows = [got.mean]
        rows += [got.by_category[category] for category in _CATEGORIES]
        rows += [got.by_component[node] for node in keys]
        return np.vstack([np.ravel(row)[:-1] for row in rows])

    def rules(a: np.ndarray, b: np.ndarray):
        """The pieces' integrals of ``exp(-r s) C(s)`` by the Kronrod
        rule, as rows, and the mean's error, against the Gauss rule."""
        half = 0.5 * (b - a)
        x = (0.5 * (a + b))[:, None] + half[:, None] * _GK_X
        g = np.exp(-rate * x) * values(x.ravel()).reshape(-1, *x.shape)
        kronrod = (g @ _GK_W) * half
        return kronrod, np.abs(kronrod[0] - (g[0] @ _G7_W) * half)

    with rbd._sharing_curves():
        at_ends = values(ends)
        a, b = _discount_pieces(ends, horizon)
        pieces, error = rules(a, b)
        for _ in range(_DISCOUNT_ROUNDS):
            budget = _DISCOUNT_TOLERANCE * abs(float(pieces[0].sum()))
            if error.sum() <= budget:
                break
            # Halve the pieces over their share of the error allowed.
            split = error > budget / error.size
            middle = 0.5 * (a + b)[split]
            halves, errors = rules(
                np.concatenate([a[split], middle]),
                np.concatenate([middle, b[split]]),
            )
            a = np.concatenate([a[~split], a[split], middle])
            b = np.concatenate([b[~split], middle, b[split]])
            pieces = np.concatenate([pieces[:, ~split], halves], axis=1)
            error = np.concatenate([error[~split], errors])
    order = np.argsort(b, kind="stable")
    running = np.concatenate(
        [np.zeros((len(at_ends), 1)), np.cumsum(pieces[:, order], axis=1)],
        axis=1,
    )
    # Every window is a piece's end: the pieces up to it are within it.
    upto = np.searchsorted(b[order], ends, side="right")
    present = np.exp(-rate * ends) * at_ends + rate * running[:, upto]
    rows = iter(present)
    mean = next(rows)
    categories = {category: next(rows) for category in _CATEGORIES}
    components = {node: next(rows) for node in keys}
    return ExpectedCost(
        window=_shaped(ends, t, windows),
        mean=_shaped(mean, t, windows),
        by_category={
            category: _shaped(v, t, windows)
            for category, v in categories.items()
        },
        by_component={
            node: _shaped(v, t, windows) for node, v in components.items()
        },
        acquisition_cost=rbd.acquisition_cost,
        discount_rate=rate,
    )


def _filled(
    rbd, probabilities: dict, size: int, working_nodes, broken_nodes
) -> dict:
    """The nodes' ``probabilities`` (arrays of ``size``), with the input
    and output nodes, and the forced nodes held at 1 or 0, added."""
    out = dict(probabilities)
    for node in list(rbd.in_or_out) + list(working_nodes | broken_nodes):
        out[node] = np.ones(size)
    return rbd._probabilities_with_overrides(out, working_nodes, broken_nodes)


def _atom_groups(
    rbd,
    curves: dict,
    atoms: dict,
    working_nodes,
    broken_nodes,
    crew=None,
    causes: bool = False,
) -> _Jumps:
    """The system at the times its nodes fail or are taken down at
    exact times (their ``atoms``): a unit dead on arrival at 0, an exact
    lifetime, a replacement at its age or at a block time.

    Nodes due at one time go down together, so the system fails there
    if it is up just before and down just after: since they only go
    down, that is the fall in its availability, worked out as the rise
    in its unavailability (a sum of products, which keeps a small one's
    precision). A node's value just before is its point availability
    there with its own events' fall (``drop``) added back; just after,
    less the probabilities of its failures, then of its planned outages
    (the failures are taken to come first). The others are at their
    point availability there: tests and restorations due then are
    done. One node alone fails the system with the probability of its
    failure times its Birnbaum importance there. With limited repair
    crews (``crew``, see ``_chain_transient.CrewSystem``), the nodes are
    the nested RBDs, and the system is worked out over the crews' chain
    at those times. With ``causes``, the failures are split among the
    nodes failing together as ``_rates.split_jumps`` splits the fall
    in availability from just before to just after their failures."""
    down = [
        a.times[(a.failure > 0.0) | (a.planned > 0.0)] for a in atoms.values()
    ]
    times = np.unique(np.concatenate(down)) if down else np.empty(0)
    if not times.size:
        empty = np.empty(0)
        return _Jumps(
            empty,
            empty,
            empty,
            empty,
            {node: empty for node in curves} if causes else None,
        )
    at, before, failed, planned = {}, {}, {}, {}
    for node, curve in curves.items():
        value = np.asarray(curve.at(times.copy()), dtype=float)
        a = atoms[node]
        drop = _matched(times, a.times, a.drop)
        failing = _matched(times, a.times, a.failure)
        stopping = _matched(times, a.times, a.planned)
        at[node] = value
        before[node] = np.clip(value + drop, 0.0, 1.0)
        failed[node] = np.clip(before[node] - failing, 0.0, 1.0)
        planned[node] = np.clip(failed[node] - stopping, 0.0, 1.0)

    def unreliability(values: dict) -> np.ndarray:
        if crew is not None:
            return crew.probabilities(values, times)[1]
        return rbd._system_unreliability(
            _filled(rbd, values, len(times), working_nodes, broken_nodes)
        )

    now, first = unreliability(at), unreliability(before)
    after, last = unreliability(failed), unreliability(planned)
    caused = None
    if causes:

        def importances(values: dict) -> dict:
            size = len(next(iter(values.values())))
            if crew is not None:
                # Each jump's time, for each point of its path.
                points = size // len(times)
                return crew.importance(values, np.repeat(times, points))
            return rbd._importances(
                _filled(rbd, values, size, working_nodes, broken_nodes)
            )[0]

        falls = _rates.split_jumps(importances, before, failed)
        caused = {node: -fall for node, fall in falls.items()}
    return _Jumps(
        times,
        np.maximum(after - first, 0.0),
        np.maximum(last - after, 0.0),
        np.maximum(now - first, 0.0),
        caused,
    )


def _system_atoms(rbd, curves: dict, stop: float, crew=None) -> Atoms:
    """This RBD's own failures and planned outages at exact times before
    ``stop``, from its nodes' ``curves`` (and its crews' chain,
    ``crew``), as a node of another (see ``Atoms``)."""
    atoms = {node: curve.atoms(stop) for node, curve in curves.items()}
    jumps = _atom_groups(rbd, curves, atoms, set(), set(), crew)
    return Atoms(
        jumps.times,
        jumps.failures,
        jumps.planned,
        np.zeros(len(jumps.times)),
        jumps.drop,
    )


def _window_counts(
    rbd,
    curves: dict,
    ends,
    working_nodes,
    broken_nodes,
    method: str,
    nodes: bool = False,
    groups: Optional[dict] = None,
    crew=None,
    causes: bool = False,
) -> dict:
    """The system's expected up time, failures and planned outages in
    ``[0, end)`` for each of ``ends``, from its nodes' ``curves`` (which
    count their events); with ``nodes``, each node's expected down time
    too; with ``causes``, the system failures each node causes (its
    terms of the failures' sum, below, under ``"caused"``; with the
    crews' chain or common-cause groups, each of their components' or
    causes' too, from their rates, #199); and for each of ``groups`` (a
    name and members), the
    expected number of its members' failures and replacements that fall
    at one instant with another's (each stop of a maintenance group is
    one).
    With limited repair crews (``crew``, a
    ``_chain_transient.CrewSystem``, #162), the ``curves`` are the nested
    RBDs', independent of the crews' chain: the system is worked out
    over the chain at each time, and the chain's own components'
    failures, at their rate in each state, add to its failures.

    The up time integrates the system's point availability as
    ``mission_availability`` does. A node's failure takes the system
    down if the node is critical then, which, the nodes being
    independent, it is at ``t`` with probability ``I_B(t)``, its
    Birnbaum importance at the nodes' availabilities then: the system's
    failures are ``integral of sum_i I_B^i(t) dM_i(t)``, ``M_i(t)`` the
    node's expected failures by ``t``, and its planned outages likewise.
    It is summed over the pieces of ``_quadrature``, on each with the
    importance taken as the cubic through its values at the points the
    up time is integrated at (see ``_quadrature.stieltjes``); the events
    at exact times are kept apart and taken together (see
    ``_atom_groups``), and start pieces. The ends do not: a total at
    an end within a piece adds the integral from the piece's start,
    by the same quadrature, so that counting at many times (a nested
    RBD's, for the RBD it is in) costs no more pieces. Past the time the
    nodes have all settled, every count is extended exactly: at its
    long-run rate, or a period at a time.
    """
    from repyability.rbd.repairable_rbd import _MISSION_POINTS, _settling

    ends = np.asarray(ends, dtype=float).ravel()
    horizon = float(ends.max()) if ends.size else 0.0
    followed = list(curves.values()) + ([crew.curve] if crew else [])
    settle, period = _settling(followed)
    reach = min(horizon, settle if period is None else settle + period)
    beyond = ends > reach
    atoms = {node: curve.atoms(reach) for node, curve in curves.items()}
    jumps = _atom_groups(
        rbd, curves, atoms, working_nodes, broken_nodes, crew, causes
    )
    # The events at exact times start pieces; the ends need not (see
    # ``totals_at``), so that a nested RBD counted at many times costs
    # no more pieces.
    fixed = [jumps.times] + [own.times for own in atoms.values()]
    cycles = rest = np.empty(0)
    if period is not None and beyond.any():
        cycles = np.floor((ends[beyond] - settle) / period)
        rest = np.clip(ends[beyond] - settle - cycles * period, 0.0, period)
        fixed += [np.array([settle]), settle + rest]

    # Each node's events at exact times, kept apart (taken together,
    # below): their running totals, by time.
    exact = {}
    for node, own in atoms.items():
        order = np.argsort(own.times, kind="stable")
        exact[node] = (
            own.times[order],
            _quadrature.running(own.failure[order], len(order)),
            _quadrature.running(own.planned[order], len(order)),
        )

    def spread(node, x: np.ndarray):
        """The node's expected failures and planned outages before
        each time ``x``, less those at exact times."""
        events = curves[node].events(x)
        times, failures, planned = exact[node]
        before = np.searchsorted(times, x, side="left")
        return (
            events["failures"] - failures[before],
            events["planned"] - planned[before],
        )

    def estimate(a: np.ndarray, b: np.ndarray) -> dict:
        x, half = _quadrature.points(a, b)
        at_points = {node: curve.at(x) for node, curve in curves.items()}
        # The system's availability and every node's importance at each
        # point, in one pass.
        own: dict = {}
        if crew is not None:
            importance, up, _, rate, own = crew.evaluate(at_points, x)
        else:
            importance, works, fails, _, _ = rbd._importances(
                _filled(rbd, at_points, len(x), working_nodes, broken_nodes)
            )
            up = works if structure_method(method) == "p" else 1.0 - fails
        out: Dict[Hashable, np.ndarray] = {
            "uptime": _quadrature.summed(up, half)
        }
        if nodes:
            for node, values in [*at_points.items(), *own.items()]:
                out[("downtime", node)] = _quadrature.summed(
                    1.0 - values, half
                )
        n = len(a)
        failures, planned = np.zeros(n), np.zeros(n)
        if crew is not None:
            # The chain's components' failures, at their rate in each
            # state, where they take the system down.
            failures += _quadrature.summed(rate, half)
            if causes:
                for key, value in crew.caused(at_points, x).items():
                    out[("caused", key)] = _quadrature.summed(value, half)
        for node in curves:
            counted = spread(node, np.concatenate([a, b, x]))
            for total, values, key in zip(
                (failures, planned), counted, ("caused", None)
            ):
                term = _quadrature.stieltjes(
                    importance[node],
                    values[:n],
                    values[n : 2 * n],  # noqa: E203
                    values[2 * n :],  # noqa: E203
                )
                total += term
                if causes and key is not None:
                    out[(key, node)] = term
        out["failures"], out["planned"] = failures, planned
        return out

    try:
        edges, finest = _quadrature.pieces(
            curves.values(), np.concatenate(fixed), reach, _MISSION_POINTS
        )
        edges, integrals = _quadrature.refined(
            estimate,
            edges,
            finest,
            _MISSION_POINTS,
            relative={"failures", "planned"},
        )
    except _quadrature.TooMany as error:
        raise NotImplementedError(
            f"Counting the expected events over [0, {reach}] takes "
            f"{error.count} pieces (the components' curves bend that "
            "often), more than the limit: estimate them by simulation, "
            "with availability() or cost()."
        ) from None
    pieces = len(edges) - 1
    # The events at exact times, at the start of the piece they fall
    # in (each is at an edge).
    starting: Dict[Any, np.ndarray] = {
        "failures": np.zeros(pieces),
        "planned": np.zeros(pieces),
    }
    if causes:
        for node in curves:
            starting[("caused", node)] = np.zeros(pieces)
    if jumps.times.size:
        piece = np.searchsorted(edges, jumps.times, side="right") - 1
        np.add.at(starting["failures"], piece, jumps.failures)
        np.add.at(starting["planned"], piece, jumps.planned)
        if causes:
            assert jumps.caused is not None
            for node, values in jumps.caused.items():
                np.add.at(starting[("caused", node)], piece, values)
    for name, members in (groups or {}).items():
        starting[("overlaps", name)] = _overlaps(atoms, members, edges)
    keys: List[Any] = ["uptime", "failures", "planned"]
    if causes:
        keys += [("caused", node) for node in curves]
        if crew is not None:
            keys += [("caused", key) for key in crew.cause_keys]
    if nodes:
        keys += [("downtime", node) for node in curves]
        if crew is not None:
            keys += [("downtime", node) for node in crew.served]
    keys += [("overlaps", name) for name in groups or {}]
    series: Dict[Any, np.ndarray] = {
        key: _quadrature.running(
            integrals.get(key, np.zeros(pieces))
            + starting.get(key, np.zeros(pieces)),
            pieces,
        )
        for key in keys
    }

    def totals_at(x: np.ndarray) -> Dict[Any, np.ndarray]:
        """Each series before each time ``x`` (within ``[0, reach]``):
        at an edge, its running total; within a piece, the running
        total at its start, the events there, and its integral from
        there, by the quadrature of the piece's start to ``x``."""
        at = np.minimum(np.searchsorted(edges, x), len(edges) - 1)
        on_edge = edges[at] == x
        out = {key: series[key][at] for key in keys}
        if pieces and not on_edge.all():
            inside = ~on_edge
            k = np.searchsorted(edges, x[inside], side="right") - 1
            k = np.clip(k, 0, pieces - 1)
            partial = _quadrature.in_blocks(estimate, edges[k], x[inside])
            for key in keys:
                out[key][inside] = (
                    series[key][k]
                    + starting.get(key, np.zeros(pieces))[k]
                    + partial.get(key, 0.0)
                )
        return out

    # Past ``reach``: constant rates, or a period at a time.
    rates: dict = {}
    if period is None and beyond.any():
        rates = _settled_rates(
            rbd,
            curves,
            reach,
            horizon,
            working_nodes,
            broken_nodes,
            method,
            crew,
        )
    out: dict = {"downtime": {}, "overlaps": {}, "caused": {}}
    inside = totals_at(ends[~beyond])
    for key, values in series.items():
        totals = np.empty(len(ends))
        totals[~beyond] = inside[key]
        if beyond.any():
            if period is None:
                rate = rates.get(key, 0.0)
                totals[beyond] = values[-1] + (ends[beyond] - reach) * rate
            else:
                base = values[np.searchsorted(edges, settle)]
                partial = values[np.searchsorted(edges, settle + rest)]
                totals[beyond] = (
                    base + cycles * (values[-1] - base) + (partial - base)
                )
        if isinstance(key, tuple):
            out[key[0]][key[1]] = totals
        else:
            out[key] = totals
    return out


def _settled_rates(
    rbd,
    curves: dict,
    reach,
    horizon,
    working_nodes,
    broken_nodes,
    method,
    crew=None,
) -> dict:
    """What ``_window_counts`` counts, per unit time, once every node's
    curve is constant: the system's availability, its failures and
    planned outages (each node's rates, from its counts' slope then,
    times its Birnbaum importance), and each node's unavailability;
    with the crews' chain (``crew``), its own components' failures
    too."""
    x = np.array([horizon])
    values = {node: curve.at(x) for node, curve in curves.items()}
    failures = planned = 0.0
    if crew is not None:
        importance, up, _, rate, own = crew.evaluate(values, x)
        rates: dict = {"uptime": float(up[0])}
        failures += float(rate[0])
        for node, value in own.items():
            rates[("downtime", node)] = 1.0 - float(value[0])
        for key, value in crew.caused(values, x).items():
            rates[("caused", key)] = float(np.ravel(value)[0])
    else:
        filled = _filled(rbd, values, 1, working_nodes, broken_nodes)
        rates = {
            "uptime": float(
                np.ravel(rbd.system_probability(filled, method=method))[0]
            )
        }
        importance = rbd._importances(filled)[0]
    span = np.array([reach, reach + max(reach, 1.0)])
    for node, curve in curves.items():
        events = curve.events(span)
        width = span[1] - span[0]
        weight = float(importance[node][0])
        caused = weight * np.diff(events["failures"])[0] / width
        failures += caused
        planned += weight * np.diff(events["planned"])[0] / width
        rates[("downtime", node)] = 1.0 - float(values[node][0])
        rates[("caused", node)] = caused
    rates["failures"], rates["planned"] = failures, planned
    return rates


def _overlaps(atoms: dict, members, edges: np.ndarray) -> np.ndarray:
    """In each piece between ``edges``, the expected number of the
    ``members``' failures and replacements at exact times that fall at
    one instant with another member's: at each such instant the
    members' expected number, less the probability of at least one
    (they are independent). A maintenance group charges one set-up per
    instant."""
    pieces = len(edges) - 1
    own = [atoms[m] for m in members if m in atoms]
    acting = [a.times[(a.failure + a.preventive) > 0.0] for a in own]
    if len(own) < 2 or not any(t.size for t in acting):
        return np.zeros(pieces)
    times = np.unique(np.concatenate(acting))
    expected = np.zeros(len(times))
    none = np.ones(len(times))
    for a in own:
        chance = np.clip(
            _matched(times, a.times, a.failure + a.preventive), 0.0, 1.0
        )
        expected += chance
        none *= 1.0 - chance
    overlap = np.maximum(expected - (1.0 - none), 0.0)
    piece = np.searchsorted(edges, times, side="right") - 1
    keep = (piece >= 0) & (piece < pieces)
    return np.bincount(piece[keep], overlap[keep], pieces)
