"""The checks a ``RepairableRBD``'s analyses refuse through
(``_require_*``), which ``analysis_routes`` calls too so that its
reasons are the methods' own messages, and the tested units' terms
(``_tested_*``: the life between tests, the rate and phase of the
tests) that the checks and the exact methods share.
"""

import math
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np

from repyability.non_repairable import NonRepairable
from repyability.rbd import (
    _long_run,
)
from repyability.rbd._block_replacement import (
    _check_life,
    _Duration,
)
from repyability.rbd._common import (
    _common_period,
    _constant_rate,
    _sf_values,
)
from repyability.rbd._hidden_life import (
    TestedLife,
)
from repyability.rbd._hidden_tests import TestedUnit
from repyability.rbd._hidden_tests import check as check_tested
from repyability.rbd._model_utils import (
    failure_time_scale,
    is_fixed_probability,
    model_mean,
)
from repyability.rbd.load_sharing_node import LoadSharingModel
from repyability.rbd.rbd import _close_name
from repyability.rbd.routes import Refused
from repyability.rbd.standby_node import StandbyModel


def _mean_lives(rbd) -> List[float]:
    """The components' mean lives (their failure-time scales), against
    which an endless horizon's discount rate is judged (#231)."""
    return [
        failure_time_scale(component.reliability)
        for component in rbd.components.values()
        if isinstance(component, NonRepairable)
    ]


def _require_component(rbd, node, given: str) -> None:
    """Raise if ``node``, given in ``given``, is not a component (see
    ``_not_a_component``)."""
    if node not in rbd.components:
        raise _not_a_component(rbd, node, given)


def _not_a_component(rbd, node, given: str) -> ValueError:
    """The error for ``node``, given in ``given``, that is not a
    component: a junction, which never fails, or an unknown name, with
    the closest component's name if one is close (#232), and those
    that are (#222)."""
    if node in rbd._junction_nodes:
        return ValueError(
            f"Node {node!r} given in {given} is a junction "
            "(PerfectReliability): it never fails, so it is no component "
            "to fail, repair, maintain or stock spares for."
        )
    close = _close_name(node, rbd.components)
    hint = "" if close is None else f" Did you mean {close!r}?"
    return ValueError(
        f"Unknown node {node!r} given in {given}; it is not a component "
        f"of the RBD.{hint} Its components are: {list(rbd.components)}."
    )


def _require_one_inspected(rbd) -> None:
    """Raise if more than one component has hidden failures, when no
    intervals are given to choose from (``optimal_inspection_intervals``
    with ``allowed=None``)."""
    if len(rbd._inspection) > 1:
        raise Refused(
            "More than one component has hidden failures: give the "
            "intervals to choose from in allowed (tests are made on "
            "a calendar, and the long-run values depend on how the "
            "schedules line up)."
        )


def _require_capacities_given(rbd) -> None:
    """Raise if a node takes its capacity from its model (a
    ``DegradingNode``'s stages, or a nested RBD's capacities), which
    the availability simulation does not follow."""
    own = rbd._capacity_models()
    if own:
        raise NotImplementedError(
            f"Node(s) {sorted(own, key=str)} take their capacity from "
            "their models (a DegradingNode's stages, or a nested RBD's "
            "capacities), which the simulation does not follow. Give "
            "them a capacity, or use capacity_distribution() for the "
            "long run."
        )


def _require_reliabilities(rbd, node) -> None:
    """Raise, as the model itself does, if a component's life or repair
    model has no exact or numerical reliability: a ``StandbyModel`` or
    ``LoadSharingModel`` that only simulations take (#149)."""
    component = rbd.components[node]
    for model in (
        getattr(component, "reliability", None),
        getattr(component, "time_to_replace", None),
    ):
        if (
            isinstance(model, (StandbyModel, LoadSharingModel))
            and model.is_simulated
        ):
            raise model._no_reliability()


def _require_time_models(rbd, node) -> None:
    """Raise if a component's life or repair model is a probability,
    not a distribution of times: its long-run and time-dependent values
    then have no exact value (its mean is no mean time). (First, if
    one has no reliability: see ``_require_reliabilities``.)"""
    _require_reliabilities(rbd, node)
    _require_times(rbd, node)


def _require_times(rbd, node) -> None:
    """Raise if a component's life or repair model is a probability,
    not a distribution of times: it has no mean time, nor a value over
    time (see ``_require_time_models``)."""
    component = rbd.components[node]
    for what, model in [
        ("reliability", getattr(component, "reliability", None)),
        ("repairability", getattr(component, "time_to_replace", None)),
    ]:
        if is_fixed_probability(model):
            raise NotImplementedError(
                f"Component {node!r}: its {what} model is a probability, "
                "not a distribution of times, so it has no mean time or "
                "availability over time: a unit fails at once with that "
                "probability, or never. Estimate the system by "
                "simulation, with availability()."
            )


def _require_unscheduled_stages(rbd, node) -> None:
    """Raise if a degrading component is maintained or inspected on a
    schedule: its long-run time in each stage has no exact value."""
    if node in rbd._preventive or node in rbd._inspection:
        raise NotImplementedError(
            f"Component {node!r} degrades through stages and is "
            "maintained or inspected on a schedule: its long-run time in "
            "each stage has no exact value here. Give it a capacity "
            "instead."
        )


def _imperfect_phrase(rbd, node) -> str:
    """How a component is repaired imperfectly, in words."""
    imperfect = rbd._imperfect[node]
    kind = "I" if imperfect.kijima == "kijima1" else "II"
    phrase = f"Kijima {kind}, q = {imperfect.q:g}"
    if imperfect.replace_after is not None:
        phrase += (
            f", replaced at failure {imperfect.replace_after} since it "
            "was renewed"
        )
    return phrase


def _require_perfect_repair(rbd, node) -> None:
    """Raise if a component is repaired imperfectly (a spec's
    ``"repair"``): a repair does not renew it, so its long-run values
    are known only by simulation (its values over time, only by
    simulation too, but for minimal repair in no time: see
    ``_require_minimal_repair``)."""
    if node in rbd._imperfect:
        over_time = (
            "its values over a window from new are exact, as it fails "
            "as often as its life's cumulative hazard"
            if _minimal_repair_blocker(rbd, node) is None
            else "nor its values over time"
        )
        raise NotImplementedError(
            f"Component {node!r} is repaired imperfectly "
            f"({_imperfect_phrase(rbd, node)}), so a repair does not "
            f"renew it: its long-run values have no exact value here "
            f"({over_time}). Estimate them by simulation, with "
            "availability() or cost()."
        )


def _minimal_repair_blocker(rbd, node) -> Optional[str]:
    """Why an imperfectly repaired component's values over time have no
    exact value, or None if they have: when it is minimally repaired
    (Kijima, ``q = 1``) in no time, with no ``replace_after``,
    preventive maintenance or tests, and a life that works at 0. Its
    failures are then a non-homogeneous Poisson process whose intensity
    is its life's hazard at its age (its age is its operating time,
    which is all the time), so that it is up throughout and fails
    ``H(t)`` times by ``t`` on average, ``H`` the life's cumulative
    hazard (see ``MinimalRepairCurve``)."""
    imperfect = rbd._imperfect[node]
    component = rbd.components[node]
    if imperfect.q < 1.0:
        return "its repairs take away some of its age (q below 1)"
    if imperfect.replace_after is not None:
        return (
            f"it is replaced at its failure {imperfect.replace_after} "
            "since it was renewed"
        )
    repair = component.time_to_replace
    if float(np.ravel(_sf_values(repair.sf, np.zeros(1)))[0]) > 0.0:
        return "its repairs take time"
    if node in rbd._preventive:
        return "it has preventive maintenance"
    if node in rbd._inspection:
        return "its failures are found only by its tests"
    life = component.reliability
    if float(np.ravel(_sf_values(life.sf, np.zeros(1)))[0]) < 1.0:
        return "its life may end at 0 (dead on arrival)"
    return None


def _require_minimal_repair(rbd, node) -> None:
    """Raise unless an imperfectly repaired component's values over time
    are exact: minimal repair in no time (see
    ``_minimal_repair_blocker``)."""
    blocker = _minimal_repair_blocker(rbd, node)
    if blocker is not None:
        raise NotImplementedError(
            f"Component {node!r} is repaired imperfectly "
            f"({_imperfect_phrase(rbd, node)}), and {blocker}: its "
            "availability over time has no exact value here (it has "
            "for minimal repair, q = 1, in no time, with no "
            "replace_after, maintenance or tests). Estimate it by "
            "simulation, with availability() or cost()."
        )


def _require_no_opportunities(rbd, node) -> None:
    """Raise if a component can be renewed early at the stops of its
    maintenance group (an ``"opportunity"`` below its interval): when
    depends on the other members, so its long-run values and its
    availability over time are known only by simulation."""
    if node in rbd._early_members:
        raise NotImplementedError(
            f"Component {node!r} is renewed early at the stops of its "
            f"maintenance group {rbd._member_group[node]!r}, which "
            "depend on the other members, so its long-run values and "
            "its availability over time have no exact value here. "
            "Estimate them by simulation, with availability() or cost()."
        )


def _require_separate_setups(rbd) -> None:
    """Raise if two members of a maintenance group with a set-up cost
    can be replaced at the same instants, again and again: on block
    schedules, or never failing before an age replacement in zero time.
    Such replacements share one stop, and its set-up, which the exact
    cost rate, charging a set-up for each member's failures and
    replacements, does not count."""
    for group, spec in rbd._maintenance.items():
        if not spec.setup_cost:
            continue
        clocked = [node for node in spec.members if _on_a_clock(rbd, node)]
        if len(clocked) > 1:
            raise NotImplementedError(
                f"Components {clocked} of maintenance group {group!r} "
                "are replaced on a clock (on block schedules, or never "
                "failing before an age replacement in zero time), so "
                "their replacements can fall at the same instants and "
                "share a set-up, which the exact cost rate does not "
                "count: estimate it by simulation, with cost()."
            )


def _on_a_clock(rbd, node) -> bool:
    """Whether a component's replacements fall on a fixed lattice of
    times: under block replacement, or under age replacement in zero
    time without a failure before it."""
    schedule = rbd._preventive.get(node)
    if schedule is None or not math.isfinite(schedule.interval):
        return False
    if schedule.policy == "block":
        return True
    if schedule.policy != "age" or schedule.duration is not None:
        return False
    survives = rbd.components[node].reliability_function(schedule.interval)
    return bool(np.ravel(survives)[0] >= 1.0)


def _require_block_models(rbd, node) -> None:
    """Raise unless the exact block-replacement values cover the
    component's models (the checks ``block_cycle`` makes first)."""

    _refuse_level(rbd, node)
    component = rbd.components[node]
    _check_life(component.reliability, node)
    _Duration(component.time_to_replace, "repair", node)
    _Duration(rbd._preventive[node].duration, "replacement", node)


def _refuse_level(rbd, node) -> None:
    """Raise for a component replaced on condition by its measured
    degradation level (#271): its inspections follow the level, which
    only the simulation draws."""
    schedule = rbd._preventive.get(node)
    if schedule is not None and schedule.level is not None:
        raise NotImplementedError(
            f"Component {node!r} is replaced on condition by its "
            "measured degradation level, which only the simulation "
            "follows: estimate it by simulation, with availability() or "
            "cost()."
        )


def _require_calendars(rbd) -> None:
    """Raise if the exact long-run values cannot average over the
    components' calendars (block replacement and inspection): a nested
    RBD's calendar with another here, intervals with no common period,
    or one repeating too often in it (before any computation)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    nested = [
        node
        for node, c in rbd.components.items()
        if isinstance(c, RepairableRBD) and _long_run._has_calendar(c)
    ]
    blocks = _long_run._block_nodes(rbd)
    if nested and len(nested) + len(rbd._inspection) + len(blocks) > 1:
        raise NotImplementedError(
            f"Node(s) {sorted(nested, key=str)} are RBDs with hidden "
            "failures or block replacement, and other nodes' inspections "
            "or block replacements here fall at the same times: "
            "estimate the long-run values by simulation, with "
            "availability() or cost()."
        )
    intervals = {rbd._preventive[node].interval for node in blocks}
    intervals |= {rbd._inspection[node].interval for node in rbd._inspection}
    if not intervals:
        return
    period = _common_period(intervals)
    if any(round(period / interval) > 100_000 for interval in intervals):
        if blocks:
            raise NotImplementedError(
                f"The block-replacement and inspection intervals "
                f"{sorted(intervals)} repeat together only after too many "
                "intervals to average over: estimate the long-run values "
                "by simulation, with availability() or cost()."
            )
        raise NotImplementedError(
            f"The inspection intervals {sorted(intervals)} repeat "
            "together only after too many inspections to average "
            "over: estimate the long-run values by simulation, with "
            "availability() or cost()."
        )


def _instant_tests(rbd, node) -> bool:
    """Whether a component with hidden failures is tested, and
    repaired, in no time."""
    return (
        rbd._inspection[node].duration is None
        and model_mean(rbd.components[node].time_to_replace) == 0.0
    )


def _tested_kind(rbd, node) -> str:
    """How a component with hidden failures' values are worked out:
    ``"closed"``, closed forms, for a constant failure rate and tests
    and repair in no time; ``"life"``, summed over the test intervals
    (``TestedLife``, #144), for any other life with those and tests
    that find every failure; and ``"unit"``, its renewal cycle followed
    test by test on a grid (``TestedUnit``, #159), for tests or repairs
    that take time, or tests that can miss a failure of a life that is
    not exponential."""
    instant = _instant_tests(rbd, node)
    if instant and _constant_rate(rbd.components[node].reliability):
        return "closed"
    if instant and not rbd._inspection[node].partial:
        return "life"
    return "unit"


def _require_tested_exact(rbd, node) -> None:
    """Raise unless a component with hidden failures has exact or
    numerical values (long-run, or over time): with tests and repair in
    no time, any life (#144); with tests or repairs that take time, or
    tests that can miss a failure, a surpyval parametric life with a
    density and repair and test times that end, the tests within the
    interval (#159)."""
    if _tested_kind(rbd, node) != "unit":
        return
    component = rbd.components[node]
    inspection = rbd._inspection[node]
    check_tested(
        component.reliability,
        component.time_to_replace,
        inspection.duration,
        inspection.interval,
        node,
    )


def _inspected_rate(rbd, node) -> Tuple[float, float]:
    """The constant failure rate and the inspection interval of a
    component with hidden failures, for the closed forms of an
    exponential life tested and repaired in no time (see
    ``_tested_life`` and ``_tested_unit`` for the others)."""
    _require_tested_exact(rbd, node)
    if _tested_kind(rbd, node) != "closed":
        raise NotImplementedError(
            f"Component {node!r} has hidden failures, and a life that is "
            "not exponential or tests or repairs that take time, which "
            "this does not take: estimate it by simulation, with "
            "availability() or cost()."
        )
    rate = _constant_rate(rbd.components[node].reliability)
    assert rate is not None
    return rate, rbd._inspection[node].interval


def _tested_life(rbd, node) -> Optional[TestedLife]:
    """For a component with hidden failures, a life other than
    exponential, and tests and repair in no time that find every
    failure, its long run under its tests (see ``_hidden_life``),
    worked out once; None for any other (see ``_tested_kind``)."""
    _require_tested_exact(rbd, node)
    if _tested_kind(rbd, node) != "life":
        return None
    component = rbd.components[node]
    interval = float(rbd._inspection[node].interval)
    cache = rbd.__dict__.setdefault("_tested_lives", {})
    key = (node, id(component.reliability), interval)
    if key not in cache:
        cache[key] = TestedLife(component.reliability, interval)
    return cache[key]


def _tested_unit(rbd, node, any_kind: bool = False) -> Optional[TestedUnit]:
    """For a component with hidden failures whose tests or repairs
    take time, or whose tests can miss a failure of a life that is not
    exponential, its values under its tests (see ``_hidden_tests``),
    worked out once; None for any other (see ``_tested_kind``), but
    with ``any_kind``, for its spares (whose tests the caller has
    checked)."""
    if not any_kind:
        _require_tested_exact(rbd, node)
        if _tested_kind(rbd, node) != "unit":
            return None
    component = rbd.components[node]
    inspection = rbd._inspection[node]
    cache = rbd.__dict__.setdefault("_tested_units", {})
    key = (
        node,
        id(component.reliability),
        id(component.time_to_replace),
        id(inspection.duration),
        float(inspection.interval),
        float(inspection.coverage),
        inspection.per_full_test,
    )
    if key not in cache:
        cache[key] = TestedUnit(
            component.reliability,
            component.time_to_replace,
            inspection.duration,
            inspection.interval,
            inspection.coverage,
            inspection.per_full_test,
            node,
            _constant_rate(component.reliability),
        )
    return cache[key]


def _unit_phase(rbd, node, times: np.ndarray) -> np.ndarray:
    """The time since a tested component's last full test at each of
    ``times`` on its calendar (its full tests at its offset and every
    full test's interval from it): where its long run's profile is
    read (see ``TestedLongRun``)."""
    inspection = rbd._inspection[node]
    position = times - inspection.offset if inspection.offset else times
    period = inspection.period
    return position - period * np.floor(position / period)


def _tested_rate(rbd, node) -> float:
    """How fast a tested component's long-run availability falls
    between tests: its failure rate, or for any other life the rate of
    its profile's mean decay (see ``TestedLife.rate``), or its failures
    per unit of up time (``TestedLongRun.rate``), for the long-run
    grid's spacing."""
    unit = _tested_unit(rbd, node)
    if unit is not None:
        return unit.long_run.rate
    life = _tested_life(rbd, node)
    return _inspected_rate(rbd, node)[0] if life is None else life.rate


def _tested_scale(rbd, node) -> float:
    """The rate that sets the scale of a tested component's interval:
    its failure rate, or one over the mean of any other life."""
    _require_tested_exact(rbd, node)
    life = rbd.components[node].reliability
    rate = _constant_rate(life)
    return float(rate) if rate else 1.0 / float(model_mean(life))


def _tested_phase(rbd, node, times: np.ndarray) -> np.ndarray:
    """The time since a tested component's last test at each of
    ``times`` on its calendar (its tests at its offset and every
    interval from it)."""
    inspection = rbd._inspection[node]
    position = times - inspection.offset if inspection.offset else times
    interval = inspection.interval
    return position - interval * np.floor(position / interval)


def _tested_intensity(
    rbd, node, times: np.ndarray, availability
) -> np.ndarray:
    """A tested component's long-run rate of failing at each of
    ``times`` (``availability`` its availability there): its constant
    rate while it is up, or for any other life the rate of its profile
    (see ``TestedLife.intensity``)."""
    unit = _tested_unit(rbd, node)
    if unit is not None:
        return unit.long_run.intensity(_unit_phase(rbd, node, times))
    life = _tested_life(rbd, node)
    if life is None:
        rate, _ = _inspected_rate(rbd, node)
        return rate * availability
    return life.intensity(_tested_phase(rbd, node, times))
