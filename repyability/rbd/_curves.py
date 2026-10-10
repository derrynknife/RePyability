"""A ``RepairableRBD``'s components over time: each node's availability
curve (a plain unit's renewal curve, block replacement's, a tested
unit's, a standby group's, a nested RBD's, and the repair crews' and
common-cause groups' chains where they couple the components), the
states a run or an analysis starts from, and when the curves settle to
their long run (``_settling``). The values over time (``_windows``) and
the simulations' stand-ins are worked out from these.
"""

import functools
import math
from collections.abc import Mapping
from functools import partial
from typing import (
    Dict,
    Hashable,
    NoReturn,
    Optional,
    Tuple,
)

import numpy as np

from repyability.non_repairable import NonRepairable
from repyability.rbd import (
    _ccf_groups,
    _chain_transient,
    _crews,
    _long_run,
    _repairable_capacity,
    _requirements,
    _standby_chain,
    _windows,
)
from repyability.rbd._block_replacement import (
    BlockHead,
    block_availability,
)
from repyability.rbd._common import (
    _common_period,
    _constant_rate,
    _repair_rate,
    _sf_values,
)
from repyability.rbd._condition_replacement import (
    ConditionHead,
    condition_availability,
)
from repyability.rbd._hidden_life import (
    TestedLife,
    TestedLifeCurve,
    TestedLifeSteady,
)
from repyability.rbd._hidden_tests import TestedUnit
from repyability.rbd._long_run import (
    _TESTED_CYCLE,
)
from repyability.rbd._model_utils import (
    MODEL_ERRORS,
    failure_time_scale,
    is_fixed_probability,
    model_mean,
)
from repyability.rbd._point_availability import (
    BlockCurve,
    First,
    InspectionCurve,
    MinimalRepairCurve,
    PartialTestCurve,
    ShiftedCurve,
    StartedBlockCurve,
    SteadyCurve,
    SystemCurve,
)
from repyability.rbd._point_availability import knots as point_knots
from repyability.rbd._point_availability import unit_curve
from repyability.rbd._sampling import stream_sampler
from repyability.rbd.node_state import NodeState
from repyability.rbd.routes import AnalysisRoute
from repyability.utils.checks import (
    structure_method,
)


def _draws_start(start) -> bool:
    """Whether a component started from ``start`` (a ``NodeState``, or
    None for new) draws what is left of its life, repair or maintenance:
    when it is down, or up at an age. Up at age 0, it is new, whatever its
    phase."""
    return isinstance(start, NodeState) and (not start.alive or start.age > 0)


#: Why a simulation does not start in the long-run state.
_STATIONARY_SIMULATION = (
    "A simulation starts from given states, not from the long-run one: give "
    "each component's NodeState (its age, or how long it has been down). "
    "The exact methods (point_availability, mission_availability, "
    "expected_failures, expected_events, expected_cost, point_capacity and "
    "mission_capacity) take state='stationary'."
)


#: The most nested RBDs the crews' chain is followed over time with: it
#: is solved for each pattern of them up and down.
_MAX_CREW_NESTED = 10


#: How a system's values over time take in its common-cause groups (#158).
_GROUPS_OVER_TIME = (
    " Each common-cause group's members' joint states come from its Markov "
    "chain followed from every member up at 0 (by uniformization, or through "
    "its tests' first period, after which they repeat the long run's), and "
    "the system is worked out over the combinations of its members up and "
    "down, the other nodes independent of them; the members' failures by "
    "each cause, at its rate where it takes the system down."
)


#: Grid steps over a component's typical up time, for its point
#: availability (see ``_point_availability``): the error falls as the
#: square of the step, to about 4e-8 here in a mission average.
_POINT_STEPS = 1000


#: The most grid points for one component's point availability.
_POINT_MAX = 2**22


#: The most pieces ``mission_availability`` integrates over.
_MISSION_POINTS = 5_000_000


def _settling_end(curve, long_run: Optional[float], cycle: float, end: float):
    """How far to follow a component's curve that has not settled at its
    long-run value ``long_run`` by ``end`` (see ``_unit_curve``). Its
    distance from that value falls by about the same factor each
    ``cycle`` (an up and a down time), so the largest distances in its
    last two cycles say when it will be within 1e-10: the curve is followed
    until that point is three quarters of the way along, as the check
    wants, and a cycle more. Four times as far where they do not say (a
    distance that does not fall, or too few cycles followed)."""
    grow = 4.0 * end
    if long_run is None or not (
        np.isfinite(cycle) and 0.0 < 2.0 * cycle < end
    ):
        return grow
    t = curve.times
    off = np.abs(curve.at(t) - long_run)
    last = t[-1]
    recent = float(off[t > last - cycle].max())
    before = float(off[(t > last - 2.0 * cycle) & (t <= last - cycle)].max())
    if not 0.0 < recent < before:
        return grow
    cycles = max(math.log(1e-10 / recent) / math.log(recent / before), 0.0)
    settled = last + cycles * cycle
    return min(grow, max(1.25 * end, settled / 0.75 + cycle))


def _up_scale(model, age: Optional[float]) -> float:
    """A typical up time for a unit with lifetime ``model`` (replaced at
    ``age``, if given), to size its point-availability grid by: the smaller
    of its median and the width of its middle 80%; else its typical failure
    time (NaN if it has none, e.g. it never fails)."""
    scale = float("nan")
    qf = getattr(model, "qf", None)
    if qf is not None:
        try:
            with np.errstate(all="ignore"):
                q10, q50, q90 = np.ravel(
                    np.asarray(qf(np.array([0.1, 0.5, 0.9])), dtype=float)
                )
            widths = [w for w in (q50, q90 - q10) if np.isfinite(w) and w > 0]
            if widths:
                scale = min(widths)
        except MODEL_ERRORS:  # a model whose qf cannot take these
            pass
    if not np.isfinite(scale):
        try:
            scale = failure_time_scale(model)
        except MODEL_ERRORS:  # no mean either (e.g. never fails)
            scale = float("nan")
    if age is not None:
        scale = age if not np.isfinite(scale) else min(scale, age)
    return float(scale)


def _settling(curves) -> Tuple[float, Optional[float]]:
    """When the point availabilities ``curves`` (see
    ``_point_availability``) all settle: the time after which they are
    constant (a period of None) or repeat together with the period returned
    (a time of inf if they never do, as far as is known)."""
    curves = list(curves)
    settle = max((float(curve.settle) for curve in curves), default=0.0)
    periods = {curve.period for curve in curves if curve.period is not None}
    if not periods:
        return settle, None
    try:
        return settle, _common_period(periods)
    except NotImplementedError:  # no common period
        return np.inf, None


def _node_over_time(rbd, node, kind: str = "availability") -> Tuple[str, str]:
    """How a component's availability over time is found, for ``kind``
    (see ``_over_time``): the route, and a phrase saying how (the
    message it raises, if refused)."""
    from repyability.rbd import routes as r
    from repyability.rbd.repairable_rbd import RepairableRBD

    component = rbd.components[node]
    if isinstance(component, RepairableRBD):
        if (
            kind == "capacity"
            and component._crews_couple()
            and component._has_capacity()
        ):
            return r.REFUSED, str(
                r.refusal(
                    partial(_no_crew_window, component, None, "the capacities")
                )
            )
        inner = _over_time(component, kind=kind)
        if inner.route == r.REFUSED:
            return r.REFUSED, inner.reason
        return inner.route, f"a nested RBD's availability, {inner.route}"
    if node in rbd._standby:
        message = r.refusal(partial(_long_run._standby_rates, rbd, node))
        if message:
            return r.REFUSED, message
        return (
            r.NUMERICAL,
            "its units' Markov chain, followed over time by " "uniformization",
        )
    message = r.refusal(partial(_requirements._require_time_models, rbd, node))
    if not message:
        message = r.refusal(
            partial(_requirements._require_no_opportunities, rbd, node)
        )
    if not message and node in rbd._imperfect:
        message = r.refusal(
            partial(_requirements._require_minimal_repair, rbd, node)
        )
        if message:
            return r.REFUSED, message
        return (
            r.EXACT,
            "minimal repair in no time: up throughout, failing as often "
            "as its life's cumulative hazard",
        )
    if not message and node in rbd._inspection:
        message = r.refusal(
            partial(_requirements._require_tested_exact, rbd, node)
        )
    schedule = rbd._preventive.get(node)
    calendar = schedule is not None and schedule.policy != "age"
    if not message and calendar:
        message = r.refusal(
            partial(_requirements._require_block_models, rbd, node)
        )
    if message:
        return r.REFUSED, message
    if calendar and schedule.policy == "condition":  # type: ignore
        return (
            r.NUMERICAL,
            "replacement on condition: followed from one inspection to "
            "the next on a grid, the units kept by age",
        )
    if (
        node in rbd._inspection
        and _requirements._tested_kind(rbd, node) == "unit"
    ):
        return r.NUMERICAL, _TESTED_CYCLE + " over time"
    return r.NUMERICAL, "its renewal equation, solved on a grid"


def _over_time(
    rbd,
    reason: str = (
        "Each component's renewal equation solved on a grid (to about "
        "1e-7), and the system exactly at its components' availabilities "
        "at each time."
    ),
    kind: str = "availability",
) -> "AnalysisRoute":
    """How the availability over time from new is found (see
    ``analysis_routes``), and what is built on it: the method's
    ``reason``. ``kind`` is what is asked: the ``"availability"``, the
    expected events over a ``"window"``, or the ``"capacity"``, which
    with limited repair crews come from their chain (see
    ``_crew_over_time``)."""
    from repyability.rbd import routes as r

    if rbd._crews_couple():
        return _crew_over_time(rbd, kind)
    if rbd.ccf_groups:
        refusal = r.refusal(
            partial(
                _ccf_groups._require_groups_over_time,
                rbd,
                {},
            )
        )
        if refusal:
            return r.refused(refusal)
        reason = reason + _GROUPS_OVER_TIME
    nodes = {node: _node_over_time(rbd, node, kind) for node in rbd.components}
    refusals = {
        n: how for n, (route, how) in nodes.items() if route == r.REFUSED
    }
    if refusals:
        return r.refused(next(iter(refusals.values())), tuple(refusals))
    return r.with_nodes(r.NUMERICAL, reason, nodes, "availabilities")


def _crew_over_time(rbd, kind: str) -> "AnalysisRoute":
    """How ``kind`` over time (see ``_over_time``) is found with limited
    repair crews: from their chain, followed over time by
    uniformization (see ``_crew_curve``), unless it refuses."""
    from repyability.rbd import routes as r

    refusal = r.refusal(
        partial(
            functools.partial(_require_crew_over_time, rbd), frozenset(), kind
        )
    )
    if refusal:
        return r.refused(refusal)
    chain = (
        "The Markov chain of the components' states and the repair "
        "queue, followed over time from its state at 0 by "
        "uniformization (to about 1e-13): "
    )
    nested = _crew_nested(rbd, frozenset())
    if kind == "window" and nested:
        reason = chain + (
            "the system's expected failures are those of its "
            "components, at their rates in each state, and of each "
            "nested RBD, independent of the chain, at its importance "
            "over the chain and the other nested RBDs' patterns, "
            "integrated by quadrature."
        )
    elif kind == "window":
        reason = chain + (
            "the system's expected failures, and each component's "
            "failures and down time, are the integrals of their rates "
            "over its states."
        )
    elif kind == "rate":
        reason = chain + (
            "the system's rate of change is split by the component "
            "whose failure or repair makes each transition (a crew "
            "taking the next job belongs to the repair that freed it), "
            "each part one more vector of the chain, exact"
            + (
                "; a nested RBD's part is its importance over the chain "
                "and the other nested RBDs' patterns times its own rate"
                if nested
                else ""
            )
            + "."
        )
    elif kind == "capacity" and nested:
        reason = chain + (
            "the capacity distribution is the probability of the states "
            "at each level, for each combination of the nested RBDs' "
            "levels, weighted by their own distributions, and over a "
            "mission its integral, by quadrature."
        )
    elif kind == "capacity":
        reason = chain + (
            "the capacity distribution is the probability of the states "
            "at each level, and over a mission its integral."
        )
    elif nested:
        reason = chain + (
            "the system's availability is the probability of the states "
            "it is up in, worked out for each pattern of the nested RBDs "
            "up and down with their own availabilities, and the mission "
            "availability its integral, by quadrature."
        )
    else:
        reason = chain + (
            "the system's availability is the probability of the states "
            "it is up in, and the mission availability its integral, "
            "exactly."
        )
    inner = "availability" if kind == "rate" else kind
    nodes = {node: _node_over_time(rbd, node, inner) for node in nested}
    refusals = {
        n: how for n, (route, how) in nodes.items() if route == r.REFUSED
    }
    if refusals:
        return r.refused(next(iter(refusals.values())), tuple(refusals))
    return r.with_nodes(r.NUMERICAL, reason, nodes, "availabilities")


def _require_crew_over_time(
    rbd, forced=frozenset(), kind: str = "availability"
) -> list:
    """Raise what the crews' chain over time refuses, in the order the
    methods check it: common-cause groups, what the chain does not
    cover (see ``_require_crew_chain``), and too many nested RBDs (see
    ``_crew_nested``). Return the nested RBDs, less those in
    ``forced``."""
    _ccf_groups._require_ccf_long_run(rbd)
    _crews._require_crew_chain(rbd)
    return _crew_nested(rbd, forced)


def _availability_curves(
    rbd,
    horizon: float,
    skip,
    counts: bool = False,
    stages: bool = False,
    state: Optional[dict] = None,
    groups: bool = False,
) -> dict:
    """Each node's point availability over ``[0, horizon]``, from new,
    as a curve (see ``_point_availability``), except the nodes in
    ``skip`` (held working or failed). With ``counts``, the curves
    count the nodes' expected events too, and follow them until they
    have settled as well; with ``stages``, a degrading component's
    curve follows its stages (see ``_capacity_models``). ``state``
    (checked by ``_states``) starts the nodes it names from their
    states rather than new. With limited repair crews the nodes are not
    independent: their values over time come from the crews' chain
    instead (see ``_crew_curve``). Nor are the members of common-cause
    groups, though each member's own curve is as without its group:
    ``groups`` says the caller takes them in (see ``_groups_curve``)."""
    assert not rbd._crews_couple(), "the crews' chain, not curves"
    assert groups or not rbd.ccf_groups, "the groups' curve, not curves"
    state = state or {}
    curves: dict = {}
    degrading = rbd._capacity_models() if stages else {}
    memo = rbd._curve_memo
    # The plain units whose curves are built, for their twins.
    plain: list = []
    for node, component in rbd.components.items():
        if node in skip:
            continue
        start = state.get(node)
        key: Optional[tuple] = (node, float(horizon), stages, start)
        try:
            hash(key)
        except TypeError:  # a nested RBD's own states, as a dict
            key = None
        if memo is not None and key is not None:
            # A curve that counts its events serves one that need not.
            found = memo.get((key, True))
            if found is None and not counts:
                found = memo.get((key, False))
            if found is not None:
                curves[node] = found
                continue
        twin = _curve_twin(rbd, node, start, node in degrading, plain)
        if twin is not None:
            curves[node] = curves[twin]
        else:
            curves[node] = _node_curve(
                rbd,
                node,
                component,
                horizon,
                counts,
                node in degrading,
                stages,
                start,
            )
            if _plain_unit(rbd, node):
                plain.append((node, start, node in degrading))
        if memo is not None and key is not None:
            memo[(key, counts)] = curves[node]
    return curves


def _plain_unit(rbd, node) -> bool:
    """Whether a component's curve follows from its life and repair
    models alone (and its state at 0): a unit with no schedule, tests,
    imperfect repair or maintenance group, not a nested RBD or a
    standby group. Another with the same models has the same curve."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return not (
        isinstance(rbd.components[node], RepairableRBD)
        or node in rbd._standby
        or node in rbd._inspection
        or node in rbd._preventive
        or node in rbd._imperfect
        or node in rbd._member_group
    )


def _curve_twin(rbd, node, start, degrading: bool, plain: list):
    """A component among ``plain`` (each ``(node, start, degrading)``,
    a plain unit whose curve is built) whose curve is ``node``'s: the
    same life and repair models, the same state at 0 and stages followed
    alike; or None. Identical components (a bank of pumps) then share
    one curve."""
    if not plain or not _plain_unit(rbd, node):
        return None
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD

    same = NonRepairableRBD._same_model
    component = rbd.components[node]
    for other, other_start, other_degrading in plain:
        known = rbd.components[other]
        if (
            other_degrading == degrading
            and other_start == start
            and same(known.reliability, component.reliability)
            and same(known.time_to_replace, component.time_to_replace)
        ):
            return other
    return None


def _node_curve(
    rbd,
    node,
    component,
    horizon: float,
    counts: bool,
    degrading: bool,
    stages: bool,
    start,
):
    """A node's curve for ``_availability_curves``: a nested RBD's, a
    standby group's, a tested component's or any other unit's."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if isinstance(component, RepairableRBD):
        return _nested_curve(component, horizon, counts, stages, start)
    if node in rbd._standby:
        return _standby_curve(rbd, node, start)
    if node not in rbd._inspection:
        return _unit_curve(rbd, node, horizon, counts, degrading, start)
    unit = _requirements._tested_unit(rbd, node)
    if unit is not None:
        return _tested_unit_curve(rbd, node, unit, horizon, start)
    life = _requirements._tested_life(rbd, node)
    if life is not None:
        return _tested_life_curve(rbd, node, life, start)
    rate, interval = _requirements._inspected_rate(rbd, node)
    inspection = rbd._inspection[node]
    if inspection.partial:
        # From new (its state is not taken: see _check_state).
        return PartialTestCurve(
            rate,
            interval,
            inspection.offset,
            inspection.coverage,
            inspection.per_full_test,
        )
    if start is None and inspection.offset:
        # New at 0, first tested at the offset: on a calendar whose last
        # test was interval - offset before 0, last known up at 0.
        return InspectionCurve(
            rate, interval, interval - inspection.offset, 0.0
        )
    if start is not None and not start.alive:
        raise NotImplementedError(
            f"Component {node!r} has hidden failures, repaired at once "
            "when a test finds them, as the exact values need: it cannot "
            "be down. Give the time since its last test as its phase (it "
            "was found working), or simulate it: availability(state=...)."
        )
    phase = 0.0 if start is None else start.phase or 0.0
    # Last known up at its last test, or, put into service since, when
    # it was.
    since = (
        phase if start is None or start.stationary else min(start.age, phase)
    )
    return InspectionCurve(rate, interval, phase, since)


def _tested_unit_curve(rbd, node, unit: TestedUnit, horizon: float, start):
    """A tested component's point availability and events from 0 when
    its tests or repairs take time, or its tests can miss a failure of
    a life that is not exponential (see ``_hidden_tests``): new at 0,
    first tested at its offset (or after an interval); in its long-run
    state; up at the age its state gives, known up at its last test (or
    when put into service since); or in a repair, under way for its
    ``down_for``. From a state its tests are on the calendar its phase
    places (``k T - phase``), as the simulation's."""
    inspection = rbd._inspection[node]
    if start is None or start.new:
        if inspection.offset:
            return unit.curve(horizon, inspection.offset, 0)
        return unit.curve(horizon, 0.0, 1)
    phase = start.phase or 0.0
    if start.stationary:
        return unit.curve(horizon, -phase, 1, ("stationary",))
    if not start.alive:
        return unit.curve(horizon, -phase, 1, ("down", start.down_for))
    since = min(start.age, phase)
    return unit.curve(horizon, -phase, 1, ("alive", start.age, since))


def _tested_life_curve(rbd, node, life: TestedLife, start):
    """A tested component's point availability and events from 0, for
    a life other than exponential (see ``_hidden_life``): new at 0
    (first tested at its offset, or after an interval), in its long-run
    state, or known up at its last test (or since put into service) at
    the age its state gives."""
    inspection = rbd._inspection[node]
    interval = inspection.interval
    if start is None:
        return TestedLifeCurve(life, inspection.offset or interval)
    # Up: its repairs take no time (see _check_state).
    phase = start.phase or 0.0
    if start.stationary:
        return TestedLifeSteady(life, phase)
    model = life.model
    # Last known up at its last test (or when put into service since),
    # at age ``known_age``; at 0 it is ``start.age`` old.
    since = min(start.age, phase)
    known_age = np.array([start.age - since])
    known_up = float(_sf_values(model.sf, known_age)[0])
    known_down = float(np.atleast_1d(model.ff(known_age))[0])

    def initial(x):
        later = start.age + np.asarray(x, dtype=float)
        return _sf_values(model.sf, later) / known_up

    def initial_down(x):
        later = start.age + np.asarray(x, dtype=float)
        if known_down <= 0.5:
            failed = np.asarray(model.ff(later), dtype=float) - known_down
        else:
            failed = known_up - _sf_values(model.sf, later)
        return np.clip(failed / known_up, 0.0, 1.0)

    first = interval - phase if phase else interval
    return TestedLifeCurve(life, first, initial, initial_down)


def _states(rbd, state, forced=frozenset(), simulated=False) -> dict:
    """``state`` checked (see ``point_availability``): each component it
    names, its ``NodeState``, and each nested RBD, its components'
    states, in the same form; for ``"stationary"``, every component in
    its long-run state. Nodes in ``forced`` (held working or broken)
    take none. ``simulated``: for a simulation, which takes an
    imperfectly repaired component's virtual age (#269)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if state is None:
        return {}
    if isinstance(state, str):
        if state != "stationary":
            raise ValueError(
                "state must be a dict of NodeStates by node, or "
                f"'stationary', got {state!r}."
            )
        return {
            node: (
                _states(component, "stationary")
                if isinstance(component, RepairableRBD)
                else NodeState(stationary=True)
            )
            for node, component in rbd.components.items()
            if node not in forced
        }
    if not isinstance(state, Mapping):
        raise TypeError(
            "state must be a dict of NodeStates by node, or "
            f"'stationary', got {state!r}."
        )
    out: dict = {}
    for node, value in state.items():
        if node not in rbd.components:
            raise _requirements._not_a_component(rbd, node, "state")
        if node in forced:
            raise ValueError(
                f"Node {node!r} is held working or broken: give it no "
                "state."
            )
        component = rbd.components[node]
        if isinstance(component, RepairableRBD):
            if isinstance(value, NodeState):
                if not value.stationary:
                    raise ValueError(
                        f"Node {node!r} is a nested RBD: give its "
                        "components' states, as a dict of NodeStates "
                        "(or 'stationary')."
                    )
                value = "stationary"
            out[node] = _states(component, value, simulated=simulated)
            continue
        if value == "stationary":
            value = NodeState(stationary=True)
        if not isinstance(value, NodeState):
            raise TypeError(
                f"The state of component {node!r} must be a NodeState, "
                f"got {value!r}."
            )
        _check_state(rbd, node, value, simulated)
        out[node] = value
    return out


def _check_state(rbd, node, state: NodeState, simulated=False) -> None:
    """Raise if a component cannot be in ``state``: a phase off a
    calendar or past its interval, maintenance it does not have, an age
    it cannot be up at or a repair or maintenance that is always over
    sooner, a virtual age of a component not repaired imperfectly, or a
    state of a kind of component whose state is not taken (an
    imperfectly repaired one's, outside a simulation)."""
    schedule = rbd._preventive.get(node)
    inspection = rbd._inspection.get(node)
    if state.virtual_age is not None and node not in rbd._imperfect:
        raise ValueError(
            f"Component {node!r} is not repaired imperfectly (it has no "
            "'repair'), so it has no virtual age: give its age alone."
        )
    if (
        schedule is not None
        and schedule.level is not None
        and not (state.new and state.phase is None)
    ):
        raise NotImplementedError(
            f"Component {node!r} is replaced on condition by its "
            "measured degradation level, which a state does not give: "
            "leave it out (new)."
        )
    if (
        inspection is not None
        and inspection.partial
        and (state.phase is not None or not state.new)
    ):
        raise NotImplementedError(
            f"The tests of component {node!r} can miss a failure (a "
            "coverage "
            "below 1): its state (how long since its last full test, "
            "and whether a failure its tests missed is waiting) is not "
            "taken: leave it out (new)."
        )
    interval = None
    if inspection is not None:
        interval = inspection.interval
    elif schedule is not None and schedule.policy != "age":
        interval = schedule.interval
    if state.phase is not None:
        if interval is None:
            raise ValueError(
                f"Component {node!r} is not on a calendar (block "
                "replacement, replacement on condition, or tests): give "
                "it no phase."
            )
        if state.phase >= interval:
            raise ValueError(
                f"Component {node!r}: the phase, {state.phase:g}, is "
                "the time since its last scheduled replacement or test, "
                f"less than its interval, {interval:g}."
            )
    if state.maintenance and (schedule is None or schedule.duration is None):
        raise ValueError(
            f"Component {node!r} has no preventive maintenance that "
            "takes time: it cannot be down for one."
        )
    if state.new or (state.stationary and node in rbd._standby):
        # A standby group in its long-run state starts its units'
        # chain from its long-run distribution.
        return
    if node in rbd._standby:
        raise NotImplementedError(
            f"Component {node!r} is a standby group, whose units' "
            "states are not taken as a state: leave it out (new)."
        )
    imperfect = rbd._imperfect.get(node)
    if imperfect is not None:
        if not simulated:
            raise NotImplementedError(
                f"Component {node!r} is repaired imperfectly: its state "
                "(its age and virtual age) is taken by the simulations "
                "(availability, cost, simulate_timelines), not here: "
                "leave it out (new)."
            )
        if (
            imperfect.replace_after is not None
            or schedule is not None
            or inspection is not None
        ):
            raise NotImplementedError(
                f"Component {node!r} is repaired imperfectly and "
                "replaced after some failures, maintained or tested, "
                "which would need its failures or operating time "
                "since it was renewed, which a state does not give: "
                "leave it out (new)."
            )
    if state.stationary or (
        state.alive and not state.age and not state.virtual_age
    ):
        return
    component = rbd.components[node]
    for what, model in [
        ("reliability", component.reliability),
        ("repairability", component.time_to_replace),
    ]:
        if is_fixed_probability(model):
            raise NotImplementedError(
                f"Component {node!r}: its {what} model is a probability, "
                "not a distribution of times, so it cannot start part "
                "way through a life or repair: leave it out (new)."
            )
    if state.alive:
        # A unit with hidden failures was known to be up only at its
        # last test, or when put into service since.
        since = 0.0
        if inspection is not None:
            since = min(state.age, state.phase or 0.0)
        age = state.age - since + (state.virtual_age or 0.0)
        survive = _sf_values(component.reliability.sf, np.array([age]))
        if not survive[0] > 0.0:
            raise ValueError(
                f"Component {node!r} cannot be up at age {age:g}: its "
                "reliability there is 0."
            )
        return
    model = (
        schedule.duration  # type: ignore[union-attr]
        if state.maintenance
        else component.time_to_replace
    )
    left = _sf_values(model.sf, np.array([state.down_for]))
    if not left[0] > 0.0:
        raise ValueError(
            f"Component {node!r} cannot have been down for "
            f"{state.down_for:g}: its "
            f"{'maintenance' if state.maintenance else 'repair'} is "
            "always over by then."
        )


def _simulation_states(rbd, state, forced=frozenset()) -> dict:
    """``state`` checked for a simulation (see ``availability``): as
    ``_states`` checks it, none stationary (a simulation starts from
    given states), and see ``_check_simulated``."""
    if state is None:
        return {}
    states = _states(rbd, state, forced, simulated=True)
    _check_simulated(rbd, states)
    return states


def _check_simulated(rbd, states: dict) -> None:
    """Raise if a simulation cannot start from ``states`` (checked by
    ``_states``): a component in its long-run state, which only the
    exact methods take; one started from a state whose draws are not
    streamed, being its own subclass or having a model other than a
    surpyval parametric one; or more components down in a repair or
    maintenance going on than repair crews to work on them."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    in_hand = 0
    served = (
        set(_crews._crew_served(rbd)) if _crews._crews_limited(rbd) else set()
    )
    for node, start in states.items():
        component = rbd.components[node]
        if isinstance(component, RepairableRBD):
            _check_simulated(component, start)
            continue
        if start.stationary:
            raise NotImplementedError(_STATIONARY_SIMULATION)
        if start.new:
            continue
        if (
            type(component) is not NonRepairable
            or stream_sampler(component.reliability) is None
            or stream_sampler(component.time_to_replace) is None
        ):
            raise NotImplementedError(
                f"Component {node!r} draws its events its own way (it is "
                f"a {type(component).__name__}, or its models are not "
                "surpyval parametric ones), so it cannot start part way "
                "through a life or repair: leave it out (new)."
            )
        if not start.alive and node in served:
            in_hand += 1
    if served and in_hand > rbd.repair_crews:  # type: ignore[operator]
        raise ValueError(
            f"{in_hand} components are down at the start, in repairs "
            "or maintenance going on, which takes more than the "
            f"{rbd.repair_crews} repair crew(s): a state does not say "
            "which jobs wait."
        )


def _curves_at(
    rbd, curves: dict, x: np.ndarray, working_nodes, broken_nodes, method
) -> np.ndarray:
    """The system's point availability at the times ``x``, from its
    nodes' curves (the forced nodes held at 1 or 0)."""
    probabilities = {node: curve.at(x) for node, curve in curves.items()}
    for node in list(rbd.in_or_out) + list(working_nodes | broken_nodes):
        probabilities[node] = np.ones(len(x))
    probabilities = rbd._probabilities_with_overrides(
        probabilities, working_nodes, broken_nodes
    )
    return np.asarray(
        rbd.system_probability(probabilities, method=method), dtype=float
    )


def _curves_down_at(
    rbd, curves: dict, x: np.ndarray, working_nodes, broken_nodes, states
) -> np.ndarray:
    """The system's point unavailability at the times ``x``, from its
    nodes' curves (the forced nodes held at 1 or 0), and the system's
    failing worked out in its own right, so that a small value keeps
    its precision (#237): each node down with one less its
    availability, or in closed form where it has one (see
    ``_closed_form_down``)."""
    failures = {}
    for node, curve in curves.items():
        down = _closed_form_down(rbd, node, x, states)
        failures[node] = 1.0 - curve.at(x) if down is None else down
    probabilities = {node: 1.0 - q for node, q in failures.items()}
    for node in list(rbd.in_or_out) + list(working_nodes | broken_nodes):
        probabilities[node] = np.ones(len(x))
        failures[node] = np.zeros(len(x))
    probabilities = rbd._probabilities_with_overrides(
        probabilities, working_nodes, broken_nodes
    )
    failures = rbd._failures_with_overrides(
        failures, working_nodes, broken_nodes
    )
    return np.clip(
        np.asarray(
            rbd._system_unreliability(probabilities, failures),
            dtype=float,
        ),
        0.0,
        1.0,
    )


def _closed_form_down(rbd, node, x, states) -> Optional[np.ndarray]:
    """A component's probability of being down at the times ``x``, in
    closed form where it has one: new at 0, with an exponential life
    and an exponential or instant repair and nothing else,
    ``lambda / (lambda + mu) (1 - exp(-(lambda + mu) t))``, which keeps
    its precision however small, as the curve's grid does not within
    its first step (#237). Else None: its curve gives it."""
    rates = _closed_form_rates(rbd, states, (node,)).get(node)
    if rates is None:
        return None
    life, total = _closed_form_pair(rbd, node)
    if math.isinf(total):
        return np.zeros(len(x))
    return -(life / total) * np.expm1(-total * np.asarray(x, float))


def _closed_form_pair(rbd, node) -> Tuple[float, float]:
    """A component's failure rate and the sum of its failure and
    repair rates (``inf`` for an instant repair), or ``(nan, nan)``
    where they are not constant."""
    component = rbd.components.get(node)
    if type(component) is not NonRepairable:
        return math.nan, math.nan
    life = _constant_rate(component.reliability)
    repair = _repair_rate(component.time_to_replace)
    if life is None or repair is None:
        return math.nan, math.nan
    return life, life + repair


def _closed_form_rates(rbd, states, nodes=None) -> Dict[Hashable, float]:
    """Of ``nodes`` (by default every component), those whose
    probability of being down over time has a closed form (see
    ``_closed_form_down``), with the rate it settles at: the sum of
    their failure and repair rates (``inf`` for an instant repair)."""
    out: Dict[Hashable, float] = {}
    for node in rbd.components if nodes is None else nodes:
        if (
            (states or {}).get(node) is not None
            or node in rbd._preventive
            or node in rbd._inspection
            or node in rbd._imperfect
            or node in rbd._standby
        ):
            continue
        life, total = _closed_form_pair(rbd, node)
        if not math.isnan(total):
            out[node] = total
    return out


def _uniformized(what: str, generator, start, steady, vectors):
    """A Markov chain followed over time from ``start`` (see
    ``_chain_transient.Uniformized``), or a NotImplementedError saying
    whose (``what``) cannot be."""
    try:
        return _chain_transient.Uniformized(generator, start, steady, vectors)
    except NotImplementedError as error:
        raise NotImplementedError(
            f"{what} Markov chain cannot be followed over time here: "
            f"{error}. Simulate it with availability()."
        ) from None


def _standby_curve(rbd, node, start: Optional[NodeState] = None):
    """A standby group over time (see ``_chain_transient.ChainCurve``):
    its units' Markov chain from every unit ready, or, for a
    ``start`` that is stationary, from its long-run distribution."""
    life, repair = _long_run._standby_rates(rbd, node)
    arrangement = rbd._standby[node]
    group = _standby_chain.chain(
        arrangement.units,
        arrangement.k,
        life,
        repair,
        arrangement.dormancy_factor,
        arrangement.switching_probability,
        rbd.repair_crews if _crews._crews_limited(rbd) else None,
    )
    if start is not None and start.stationary:
        initial = group.probabilities
    else:
        initial = np.zeros(len(group.states))
        initial[0] = 1.0
    chain = _uniformized(
        f"Component {node!r} is a standby group, whose",
        group.generator,
        initial,
        group.probabilities,
        np.column_stack([group.up, group.failures, group.unit_failures]),
    )
    return _chain_transient.ChainCurve(chain)


def _nested_curve(
    rbd,
    horizon: float,
    counts: bool = False,
    stages: bool = False,
    start=None,
):
    """This RBD as a node of another, over time: the structure function
    at its own nodes' curves (a ``SystemCurve``), or, with limited
    repair crews, its crews' chain (see ``_crew_curve``). ``counts``,
    ``stages`` and ``start`` are as for ``_availability_curves``."""
    if rbd.ccf_groups and not rbd._crews_couple():
        return _ccf_groups._groups_curve(
            rbd, horizon, states=start or {}, counts=counts, stages=stages
        )
    if not rbd._crews_couple():
        inner = _availability_curves(
            rbd, horizon, set(), counts, stages, start
        )
        return SystemCurve(rbd, inner, *_settling(inner.values()))
    curve = _crew_curve(rbd, horizon, states=start or {}, counts=counts)
    if stages and rbd._has_capacity():
        curve.capacity = _repairable_capacity._crew_capacity_over(
            rbd, horizon, set(), set(), start or {}
        )
    return curve


def _crew_nested(rbd, forced) -> list:
    """The nested RBDs the crews' chain is followed over time with:
    those not held working or broken. Raise if there are too many."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    nested = [
        node
        for node, component in rbd.components.items()
        if isinstance(component, RepairableRBD) and node not in forced
    ]
    if len(nested) > _MAX_CREW_NESTED:
        raise NotImplementedError(
            f"With limited repair crews, the availability over time is "
            "worked out for each pattern of the nested RBDs up and down: "
            f"{len(nested)} nested RBDs are more than the "
            f"{_MAX_CREW_NESTED} it takes. Simulate it with "
            "availability()."
        )
    return nested


def _crew_vectors(
    rbd, chain, values: dict, working_nodes, broken_nodes, method: str
) -> Tuple[np.ndarray, dict]:
    """Over the states of the crews' ``chain``, with the other nodes'
    probabilities ``values`` (arrays, one entry per state): whether the
    system is up, and each node's Birnbaum importance."""
    size = len(chain.probabilities)
    own = {
        node: chain.up[:, k].astype(float)
        for k, node in enumerate(chain.nodes)
    }
    own.update(values)
    importance, works, fails, _, _ = rbd._importances(
        _windows._filled(rbd, own, size, working_nodes, broken_nodes)
    )
    up = works if structure_method(method) == "p" else 1.0 - fails
    return np.asarray(up, dtype=float), importance


def _crew_failing(rbd, chain, importance: dict) -> np.ndarray:
    """The rate of the system's failures in each state of the crews'
    ``chain``: a component that is up fails at its rate, and takes the
    system down where it is critical (its Birnbaum ``importance``
    there, 1 or 0), as ``system_failure_frequency`` has it."""
    rates = _crews._crew_chain_rates(rbd)
    failing = np.zeros(len(chain.probabilities))
    for k, node in enumerate(chain.nodes):
        failing += importance[node] * chain.up[:, k] * rates[node][0]
    return failing


def _crew_failing_each(rbd, chain, importance: dict) -> np.ndarray:
    """``_crew_failing`` by component: the rate of the system's failures
    by each of the chain's components (columns) in each of its states
    (#199)."""
    rates = _crews._crew_chain_rates(rbd)
    return np.column_stack(
        [
            importance[node] * chain.up[:, k] * rates[node][0]
            for k, node in enumerate(chain.nodes)
        ]
    )


def _crew_each(rbd, chain, nested: list, working, broken, method: str) -> list:
    """For each pattern of the ``nested`` RBDs up and down, the rate of
    the system's failures by each of the crews' chain's components in
    each state (see ``_crew_failing_each``), for ``CrewSystem``'s
    ``causes``."""
    size = len(chain.probabilities)
    out = []
    for pattern in range(2 ** len(nested)):
        _, importance = _crew_vectors(
            rbd,
            chain,
            {
                node: np.full(size, float((pattern >> j) & 1))
                for j, node in enumerate(nested)
            },
            working,
            broken,
            method,
        )
        out.append(_crew_failing_each(rbd, chain, importance))
    return out


def _crew_start(rbd, chain, states: dict) -> np.ndarray:
    """The probabilities of the crews' chain's states at 0, from the
    components' ``states`` (checked by ``_states``): every component up
    (a component's age does not matter, its life being exponential),
    each in its long-run state (all of them, ``"stationary"``), or as
    they say: a component down is in a repair (as its repair is
    exponential, however long it has taken), and none waits, so there
    can be no more down than crews."""
    named = {node: states[node] for node in chain.nodes if node in states}
    stationary = [node for node, s in named.items() if s.stationary]
    if stationary:
        if len(stationary) < len(chain.nodes):
            raise NotImplementedError(
                "With limited repair crews, the components' long-run "
                "states are tied together by the repair queue: start "
                "every component in its long-run state "
                "(state='stationary'), or give each its state."
            )
        return np.array(chain.probabilities, dtype=float)
    down = tuple(
        k
        for k, node in enumerate(chain.nodes)
        if node in named and not named[node].alive
    )
    assert rbd.repair_crews is not None  # limited crews only
    if len(down) > rbd.repair_crews:
        raise ValueError(
            f"{len(down)} components are down at the start, in repairs "
            "going on, which takes more than the "
            f"{rbd.repair_crews} repair crew(s): a state does not say "
            "which jobs wait."
        )
    start = np.zeros(len(chain.probabilities))
    start[chain.states.index((down, ()))] = 1.0
    return start


def _crew_curve(
    rbd,
    horizon: float,
    working_nodes=frozenset(),
    broken_nodes=frozenset(),
    method: str = "p",
    states: Optional[dict] = None,
    counts: bool = False,
) -> "_chain_transient.CrewCurve":
    """The system over time with limited repair crews (see
    ``_chain_transient.CrewCurve``): the crews' chain, without the
    nodes held working or broken, from the components' ``states``
    (see ``_crew_start``), with the nested RBDs' own curves (each from
    its own state in ``states``) to ``horizon``, counting their events
    with ``counts``."""
    states = states or {}
    working, broken = set(working_nodes), set(broken_nodes)
    forced = working | broken
    nested = _require_crew_over_time(rbd, frozenset(forced))
    chain = _crews._crew_chain(rbd, frozenset(forced))
    vectors = _crew_patterns(rbd, chain, nested, working, broken, method)
    uniformized = _uniformized(
        "The repair crews'",
        chain.generator,
        _crew_start(rbd, chain, states),
        chain.probabilities,
        np.column_stack(vectors),
    )
    curves = {
        node: _nested_curve(
            rbd.components[node],
            horizon,
            counts=counts,
            start=states.get(node),
        )
        for node in nested
    }
    return _chain_transient.CrewCurve(rbd, uniformized, curves)


def _crew_patterns(
    rbd, chain, nested: list, working, broken, method: str
) -> list:
    """The crews' chain's vectors for each pattern of the ``nested``
    RBDs up and down (see ``_chain_transient.pattern_weights``): whether
    the system is up in each state, for every pattern, then the rate of
    its failures by the chain's components, for every pattern (see
    ``_chain_transient.CrewSystem``)."""
    size = len(chain.probabilities)
    ups, failing = [], []
    for pattern in range(2 ** len(nested)):
        up, importance = _crew_vectors(
            rbd,
            chain,
            {
                node: np.full(size, float((pattern >> j) & 1))
                for j, node in enumerate(nested)
            },
            working,
            broken,
            method,
        )
        ups.append(up)
        failing.append(_crew_failing(rbd, chain, importance))
    return ups + failing


def _no_crew_window(
    rbd, node=None, what: str = "the expected events"
) -> NoReturn:
    """Raise that with limited repair crews, ``what`` over time (from
    their chain) cannot take in a nested RBD (``node``), or (no
    ``node``) be given to another RBD that has this one nested, as
    yet."""
    where = (
        "which has no place, as yet, for those of a nested RBD (#162): "
        f"node {node!r} is one"
        if node is not None
        else "which gives none to an RBD this one is nested in, as yet "
        "(#162)"
    )
    raise NotImplementedError(
        f"With {rbd.repair_crews} repair crew(s) for "
        f"{len(_crews._crew_served(rbd))} components, {what} over time come "
        f"from the crews' Markov chain, {where}. Simulate them with "
        "availability() or cost(); the availability over time and the "
        "long-run values are exact."
    )


def _groups_window(
    rbd,
    ends: np.ndarray,
    working_nodes,
    broken_nodes,
    method: str,
    states: dict,
    nodes: bool,
    groups: Optional[dict],
    causes: bool = False,
) -> Tuple[dict, dict]:
    """``_window``'s curves and counts with common-cause groups (#158):
    the system's up time, failures and planned outages over its groups'
    joint states (see ``_ccf_chain.GroupsSystem``), and each node's own
    curve, the members' as without their groups, for their own events
    and down time."""
    working, broken = set(working_nodes), set(broken_nodes)
    horizon = float(np.max(ends)) if len(ends) else 0.0
    system_curve = _ccf_groups._groups_curve(
        rbd, horizon, working, broken, method, states, counts=True
    )
    totals = _windows._window_counts(
        rbd,
        system_curve.curves,
        ends,
        working,
        broken,
        method,
        nodes=nodes,
        groups=groups,
        crew=system_curve.system,
        causes=causes,
    )
    curves = dict(system_curve.curves)
    members = [m for group in rbd.ccf_groups for m in group.members]
    own = _availability_curves(
        rbd,
        horizon,
        set(rbd.components) - set(members),
        counts=True,
        groups=True,
    )
    curves.update(own)
    return curves, totals


def _crew_window(
    rbd,
    ends: np.ndarray,
    working_nodes,
    broken_nodes,
    method: str,
    states: dict,
    nodes: bool,
    groups: Optional[dict],
    causes: bool = False,
) -> Tuple[dict, dict]:
    """``_window``'s curves and counts with limited repair crews, from
    their chain over time: the system's up time and failures (the
    integral of the rate of its failures in each state, as
    ``system_failure_frequency`` has it), and each component's down
    time and failures (at its rate while it is up), each a repair. No
    component has scheduled maintenance or tests, and none fails at the
    same instant as another."""
    working, broken = set(working_nodes), set(broken_nodes)
    forced = working | broken
    nested = _require_crew_over_time(rbd, frozenset(forced), "window")
    chain = _crews._crew_chain(rbd, frozenset(forced))
    rates = _crews._crew_chain_rates(rbd)
    if nested:
        # The nested RBDs, independent of the chain, counted as nodes,
        # with the system worked out over the chain (#162).
        each = (
            _crew_each(rbd, chain, nested, working, broken, method)
            if causes
            else []
        )
        uniformized = _uniformized(
            "The repair crews'",
            chain.generator,
            _crew_start(rbd, chain, states),
            chain.probabilities,
            np.column_stack(
                _crew_patterns(rbd, chain, nested, working, broken, method)
                + [chain.up.astype(float)]
                + each
            ),
        )
        system = _chain_transient.CrewSystem(
            uniformized, nested, chain.nodes, causes
        )
        horizon = float(np.max(ends)) if len(ends) else 0.0
        curves = {
            node: _nested_curve(
                rbd.components[node],
                horizon,
                counts=True,
                start=states.get(node),
            )
            for node in nested
        }
        totals = _windows._window_counts(
            rbd,
            curves,
            ends,
            working,
            broken,
            method,
            nodes=nodes,
            groups=groups,
            crew=system,
            causes=causes,
        )
        column = 2 * system.patterns
        for k, node in enumerate(chain.nodes):
            curves[node] = _chain_transient.CrewNodeEvents(
                uniformized, column + k, rates[node][0]
            )
        return curves, totals
    up, importance = _crew_vectors(rbd, chain, {}, working, broken, method)
    failing = _crew_failing(rbd, chain, importance)
    each = [_crew_failing_each(rbd, chain, importance)] if causes else []
    uniformized = _uniformized(
        "The repair crews'",
        chain.generator,
        _crew_start(rbd, chain, states),
        chain.probabilities,
        np.column_stack([up, failing, chain.up.astype(float)] + each),
    )
    totals = uniformized.integrals(ends)
    zero = np.zeros(len(ends))
    counts: dict = {
        "uptime": np.clip(totals[:, 0], 0.0, ends),
        "failures": np.maximum(totals[:, 1], 0.0),
        "planned": zero,
        "downtime": {},
        "overlaps": {name: zero for name in groups or {}},
    }
    if causes:
        base = 2 + len(chain.nodes)
        counts["caused"] = {
            node: np.maximum(totals[:, base + k], 0.0)
            for k, node in enumerate(chain.nodes)
        }
    curves = {}
    for k, node in enumerate(chain.nodes):
        curves[node] = _chain_transient.CrewNodeEvents(
            uniformized, 2 + k, rates[node][0]
        )
        if nodes:
            counts["downtime"][node] = np.clip(
                ends - totals[:, 2 + k], 0.0, ends
            )
    return curves, counts


def _unit_curve(
    rbd,
    node,
    horizon: float,
    counts: bool = False,
    stages: bool = False,
    start: Optional[NodeState] = None,
    unscheduled: bool = False,
):
    """A component's point availability from new over ``[0, horizon]``
    (see ``_point_availability.unit_curve``): on a grid of
    ``_POINT_STEPS`` steps over its typical up time, with its
    replacement age on the grid, and only as long as it takes to settle
    at its long-run availability (to 1e-10), after which the curve holds
    that value. With ``counts`` it counts the component's expected
    events too, until they grow at their long-run rates (to 1e-7), and
    at those rates after. With ``stages``, a degrading component's
    curve follows its stages too, until they have settled at their
    long-run shares of its up time (to 1e-8). Under block replacement,
    see ``_block_curve``. ``start`` starts it from a state rather than
    new (see ``point_availability``). ``unscheduled`` leaves out its
    preventive maintenance, and follows it to the horizon: the head of a
    block-replacement curve, up to its first block time."""
    _requirements._require_time_models(rbd, node)
    _requirements._require_no_opportunities(rbd, node)
    component = rbd.components[node]
    if node in rbd._imperfect:
        _requirements._require_minimal_repair(rbd, node)
        scale = _up_scale(component.reliability, None)
        if not (np.isfinite(scale) and scale > 0.0):
            scale = horizon if horizon > 0.0 else 1.0
        return MinimalRepairCurve(
            component.reliability, scale, _POINT_STEPS, counts
        )
    schedule = None if unscheduled else rbd._preventive.get(node)
    if schedule is not None and schedule.policy in ("block", "condition"):
        return _block_curve(rbd, node, horizon, start)
    age = None if schedule is None else float(schedule.interval)
    duration = None if schedule is None else schedule.duration
    life = component.reliability
    up_sf = life.sf
    up_splits = point_knots(life)
    repair = component.time_to_replace

    def up_cdf(x):
        return 1.0 - _sf_values(up_sf, x)

    def repair_sf(s):
        return _sf_values(repair.sf, s)

    def duration_sf(s):
        assert duration is not None
        return _sf_values(duration.sf, s)

    maintenance_sf = None if duration is None else duration_sf
    try:
        if unscheduled:
            raise ValueError("followed to the horizon")
        long_run: Optional[float] = _long_run._node_availability(rbd, node)
        rates: Optional[Tuple[float, float, float]] = (
            _long_run._node_frequencies(rbd, node) if counts else None
        )
    except (ValueError, NotImplementedError):
        long_run, rates = None, None
    stage_model = None
    if stages:
        _requirements._require_unscheduled_stages(rbd, node)
        stage_model = life
    if start is not None and start.stationary:
        # Long in service: in its long-run state throughout.
        if long_run is None:
            _long_run._node_availability(rbd, node)  # raises why it has none
        failures, maintained, _ = _long_run._node_frequencies(rbd, node)
        return SteadyCurve(
            float(long_run),  # type: ignore[arg-type]
            failures,
            maintained,
            takedown=duration is not None,
            fractions=(
                None if stage_model is None else stage_model.stage_fractions()
            ),
        )
    first = None
    if _draws_start(start):
        first = _first_unit(
            rbd,
            node,
            start,  # type: ignore[arg-type]
            up_sf,
            up_splits,
            age,
            stage_model,
        )
    scale = _up_scale(life, age)
    if schedule is not None:
        up, cycle, _, _ = _long_run._maintenance_cycle(rbd, node, schedule)
    else:
        up = model_mean(life)
        cycle = up + model_mean(repair)
    horizon = max(horizon, 0.0)
    step = horizon / 1024.0 if horizon > 0.0 else np.inf
    if np.isfinite(scale):
        step = min(step, scale / _POINT_STEPS)
    if not np.isfinite(step):
        step = 1.0
    if age is not None and step < age:
        step = age / np.ceil(age / step)
    # Units that each reach their age are maintained at nearly fixed
    # times, which the curve follows until they have all but ended.
    survive = 0.0
    if age is not None and duration is not None:
        survive = float(_sf_values(up_sf, np.array([age]))[0])
    end = max(horizon, 16.0 * step)
    settles = age is None or survive ** (horizon // age) < 1e-10
    if long_run is not None and settles and np.isfinite(cycle) and cycle > 0.0:
        end = min(end, max(8.0 * cycle, 16.0 * step))
    while True:
        n = int(np.ceil(end / step))
        if n > _POINT_MAX:
            step = end / _POINT_MAX
            if age is not None and step < age:
                step = age / np.floor(age / step)
            if np.isfinite(scale) and step > scale / 250.0:
                raise NotImplementedError(
                    f"Component {node!r} would need more than "
                    f"{_POINT_MAX} grid points to follow its "
                    f"availability to {end} (its typical up time is "
                    f"about {scale:.3g}): estimate it by simulation, "
                    "with availability()."
                )
            n = min(int(np.ceil(end / step)), _POINT_MAX)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                curve = unit_curve(
                    up_cdf,
                    up_splits,
                    repair_sf,
                    point_knots(repair),
                    step,
                    n,
                    age=age,
                    maintenance_sf=maintenance_sf,
                    maintenance_splits=(
                        () if duration is None else point_knots(duration)
                    ),
                    counts=counts,
                    stages=(
                        None
                        if stage_model is None
                        else stage_model.stage_probabilities
                    ),
                    first=first,
                )
        except NotImplementedError as error:
            raise NotImplementedError(
                f"Component {node!r}: {error}. Estimate its availability "
                "by simulation, with availability()."
            ) from None
        events = curve.unit_events
        if events is not None and rates is not None:
            events.failure_rate, events.preventive_rate = rates[:2]
        if (
            curve.stages is not None
            and stage_model is not None
            and long_run is not None
        ):
            curve.stages.fractions = stage_model.stage_fractions()
        if end >= horizon:
            break
        tail = curve.times[len(curve.times) - len(curve.times) // 4 :]
        if (
            long_run is not None
            and np.all(np.abs(curve.at(tail) - long_run) < 1e-10)
            and (age is None or survive ** (end // age) < 1e-10)
            and (events is None or events.settled())
            and (curve.stages is None or curve.stages.settled())
        ):
            break
        end = min(horizon, _settling_end(curve, long_run, cycle, end))
    curve.long_run = long_run
    if curve.unit_events is not None and long_run is None:
        # Followed to the horizon: no rates past it.
        curve.unit_events.failure_rate = None
        curve.unit_events.preventive_rate = None
    return curve


def _block_curve(rbd, node, horizon: float, start: Optional[NodeState] = None):
    """A component's point availability from new under block
    replacement, over ``[0, horizon]`` (see
    ``_block_replacement.block_availability``), or replaced on condition
    (#161, see ``_condition_replacement.condition_availability``); in
    its long-run state (a stationary ``start``), its settled cycle from
    its phase on; from another state, its own curve up to its first
    block time or inspection, and the schedule's from there (see
    ``StartedBlockCurve``)."""
    _requirements._refuse_level(rbd, node)
    component = rbd.components[node]
    schedule = rbd._preventive[node]
    duration = schedule.duration
    stationary = start is not None and start.stationary
    knots = np.empty(0) if duration is None else point_knots(duration)
    life = component.reliability
    models = (life, component.time_to_replace, duration, schedule.interval)

    def followed(horizon: float, head=None):
        if schedule.policy == "condition":
            return condition_availability(
                *models, float(schedule.threshold), horizon, node, head
            )
        return block_availability(*models, horizon, node, head)

    if start is not None and not start.new and not stationary:
        length = schedule.interval - (start.phase or 0.0)
        head = _unit_curve(
            rbd, node, length, counts=True, start=start, unscheduled=True
        )
        down_sf = None
        first = None
        if not start.alive:
            down_sf = _first_unit(
                rbd, node, start, None, (), None, None
            ).down_sf
        else:
            # The unit in service now, if unfailed by the first
            # inspection, and its age there.
            ages = float(start.age) + np.array([0.0, length])
            alive = _sf_values(life.sf, ages)
            first = (float(alive[1] / alive[0]), float(ages[1]))

        def failures(u):
            return head.events(np.asarray(u, dtype=float))["failures"]

        up = float(head.at(np.array([length]))[0])
        reached = (
            ConditionHead(length, up, failures, down_sf, first)
            if schedule.policy == "condition"
            else BlockHead(length, up, failures, down_sf)
        )
        result = followed(max(horizon - length, 0.0), reached)
        return StartedBlockCurve(head, BlockCurve(result, knots), length)
    result = followed(np.inf if stationary else max(horizon, 0.0))
    curve = BlockCurve(result, knots)
    if not stationary:
        return curve
    assert start is not None
    return ShiftedCurve(curve, curve.settle + (start.phase or 0.0))


def _first_unit(
    rbd, node, start: NodeState, up_sf, up_splits, age, stage_model
) -> First:
    """A component's unit at 0 in state ``start`` (see
    ``_point_availability.First``): up at its age, with what is left of
    its life (the survival function ``R(a + s) / R(a)``) and its
    replacement due ``age - a`` from now; or down, with what is left of
    its repair or maintenance (``G(r + s) / G(r)``, after ``r`` so
    far). ``_check_state`` has checked that it can be in the state."""
    if start.alive:
        a = float(start.age)
        survive = float(_sf_values(up_sf, np.array([a]))[0])

        def cdf(s):
            return np.clip(1.0 - _sf_values(up_sf, a + s) / survive, 0.0, 1.0)

        splits = np.asarray(up_splits, dtype=float) - a

        def stages(s):
            return stage_model.stage_probabilities(a + s) / survive

        return First(
            cdf,
            splits[splits > 0.0],
            None if age is None else age - a,
            stages=None if stage_model is None else stages,
        )
    if start.maintenance:
        model = rbd._preventive[node].duration
    else:
        model = rbd.components[node].time_to_replace
    r = float(start.down_for)
    left = float(_sf_values(model.sf, np.array([r]))[0])

    def down_sf(s):
        return _sf_values(model.sf, r + s) / left

    splits = point_knots(model) - r
    return First(down_sf=down_sf, down_splits=splits[splits > 0.0])
