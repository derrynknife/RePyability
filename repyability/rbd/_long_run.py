"""A ``RepairableRBD``'s long-run values: each component's availability
and failure frequency in the long run, exact or numerical (renewal
cycles, block replacement's calendar grid, tested units' cycles), and
the system's (``mean_availability``, ``system_failure_frequency``,
``mean_time_between_failures``, ``mean_up_time``, ``mean_down_time``,
``node_availability``). The methods of ``RepairableRBD`` of those names
call these.
"""

import functools
import math
from functools import partial
from typing import (
    Any,
    Collection,
    Dict,
    Hashable,
    List,
    NoReturn,
    Optional,
    Tuple,
)

import numpy as np

from repyability.non_repairable import NonRepairable
from repyability.rbd import (
    _ccf_groups,
    _crews,
    _requirements,
    _standby_chain,
)
from repyability.rbd._block_replacement import (
    block_cycle,
)
from repyability.rbd._common import (
    _common_period,
    _constant_rate,
)
from repyability.rbd._condition_replacement import (
    condition_cycle,
)
from repyability.rbd._events import (
    _Inspection,
    _Preventive,
)
from repyability.rbd._model_utils import (
    model_mean,
)
from repyability.rbd.routes import AnalysisRoute
from repyability.utils.checks import (
    structure_method,
)

#: How the values of a component with hidden failures whose tests or
#: repairs take time, or whose tests can miss a failure of a life that is
#: not exponential, are found (#159).
_TESTED_CYCLE = (
    "hidden failures with tests or repairs that take time, or tests that "
    "miss: the cycle from one test that finds a failure to the next, "
    "followed test by test on a grid"
)


def _failed_by(component: NonRepairable, age: float) -> float:
    """The probability that a component's unit fails before ``age``: from
    its model's own ``ff``, which keeps a small one's precision."""
    return float(np.ravel(component.reliability.ff(age))[0])


def _log_kept(rate: float, inspection: "_Inspection") -> float:
    """``log(rho)``, ``rho = 1 - (1 - c) (1 - exp(-rate * interval))``: the
    chance that a unit with hidden failures, up after a test, is up after
    the next one or had its failure found by it (a coverage ``c``)."""
    missed = (1.0 - inspection.coverage) * -math.expm1(
        -rate * inspection.interval
    )
    return math.log1p(-missed)


def _tests_in(inspection: "_Inspection", period: float) -> list:
    """The times of an inspection's tests in ``[0, period)`` that are not
    multiples of its interval (those of a schedule with an offset), on a
    calendar that repeats every ``period``."""
    if not inspection.offset:
        return []
    count = int(round(period / inspection.interval))
    times = inspection.offset + inspection.interval * np.arange(count)
    return list(np.where(times >= period, times - period, times))


def mean_unavailability(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
) -> float:
    """See ``RepairableRBD.mean_unavailability``."""
    method = structure_method(method)
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    if rbd.ccf_groups:
        # Each group conditioned on within its module (#218).
        _, weights, inputs = _ccf_groups._ccf_long_run(
            rbd, working_nodes, broken_nodes
        )
        return float(
            weights @ _ccf_groups._ccf_tabled(rbd, *inputs).system()[1]
        )
    availability, unavailability, weights = _long_run_unavailabilities(
        rbd, working_nodes, broken_nodes
    )
    system = rbd._system_unreliability(availability, unavailability)
    return float(weights @ system)


def _node_long_run(rbd, node) -> Tuple[str, str]:
    """How a component's long-run values are found: the route, and a
    phrase saying how (the message it raises, if refused)."""
    from repyability.rbd import routes as r
    from repyability.rbd.repairable_rbd import RepairableRBD

    component = rbd.components[node]
    if isinstance(component, RepairableRBD):
        inner = _long_run_route(component)
        if inner.route == r.REFUSED:
            return r.REFUSED, inner.reason
        return (
            inner.route,
            f"a nested RBD's long-run values, {inner.route}",
        )
    for check in (
        partial(_requirements._require_perfect_repair, rbd),
        partial(_requirements._require_times, rbd),
    ):
        message = r.refusal(partial(check, node))
        if message:
            return r.REFUSED, message
    if node in rbd._standby:
        message = r.refusal(
            partial(functools.partial(_standby_rates, rbd), node)
        )
        if message:
            return r.REFUSED, message
        return r.EXACT, "a standby group's Markov chain"
    if node in rbd._inspection:
        message = r.refusal(
            partial(_requirements._require_tested_exact, rbd, node)
        )
        if message:
            return r.REFUSED, message
        kind = _requirements._tested_kind(rbd, node)
        if kind == "unit":
            return r.NUMERICAL, _TESTED_CYCLE
        if kind == "life":
            return (
                r.NUMERICAL,
                "hidden failures, renewed at the tests, summed over the "
                "test intervals",
            )
        return r.EXACT, "hidden failures at a constant rate"
    schedule = rbd._preventive.get(node)
    if schedule is None:
        message = r.refusal(component.mean_availability)
        if message:
            return r.REFUSED, message
        life, how = r.mean_route(component.reliability)
        if life == r.EXACT:
            return r.EXACT, "its mean life and mean repair time"
        return life, f"its mean life, {how}"
    message = r.refusal(
        partial(_requirements._require_no_opportunities, rbd, node)
    )
    if message:
        return r.REFUSED, message
    life, how = r.model_route(component.reliability)
    if schedule.policy in ("block", "condition"):
        message = r.refusal(
            partial(_requirements._require_block_models, rbd, node)
        )
        if message:
            return r.REFUSED, message
        if schedule.policy == "condition":
            return (
                r.NUMERICAL,
                "replacement on condition: its renewal cycle, followed "
                "from one inspection to the next on a grid",
            )
        return (
            r.NUMERICAL,
            "block replacement: its renewal cycle, solved on a grid",
        )
    if life == r.SIMULATED:
        return life, f"age replacement of a life from {how}"
    return r.NUMERICAL, "age replacement: its mean up time, by quadrature"


def _long_run_nodes(rbd) -> Dict[Any, Tuple[str, str]]:
    """Each component's long-run route (see ``_node_long_run``)."""
    return {node: _node_long_run(rbd, node) for node in rbd.components}


def _long_run_route(rbd, groups: bool = True) -> "AnalysisRoute":
    """How the exact long-run values are found (see
    ``analysis_routes``); without ``groups``, the components' own, as
    ``node_availability`` gives them (the common-cause groups change
    only which are down together)."""
    from repyability.rbd import routes as r

    chains = (
        r.refusal(partial(_ccf_groups._require_ccf_long_run, rbd))
        if groups
        else None
    )
    if chains:
        # (Before the crews: the groups' chains do not take them in.)
        return r.refused(chains)
    if rbd._crews_couple():
        return _crew_chain_route(rbd)
    calendars = r.refusal(partial(_requirements._require_calendars, rbd))
    if calendars:
        return r.refused(calendars)
    nodes = _long_run_nodes(rbd)
    refusals = {
        n: how for n, (route, how) in nodes.items() if route == r.REFUSED
    }
    if refusals:
        return r.refused(next(iter(refusals.values())), tuple(refusals))
    return r.with_nodes(
        r.EXACT,
        "The structure function over the components' long-run "
        "availabilities, exactly"
        + (
            ", and over each common-cause group's members' joint states "
            "(a Markov chain)."
            if rbd.ccf_groups and groups
            else "."
        ),
        nodes,
        "long-run values",
    )


def _crew_chain_route(rbd) -> "AnalysisRoute":
    """How the exact long-run values are found with limited repair
    crews: from their Markov chain, and nested RBDs' own values."""
    from repyability.rbd import routes as r
    from repyability.rbd.repairable_rbd import RepairableRBD

    chain = r.refusal(partial(_crews._require_crew_chain, rbd))
    if chain:
        return r.refused(chain)
    nested = {
        node: _node_long_run(rbd, node)
        for node, component in rbd.components.items()
        if isinstance(component, RepairableRBD)
    }
    refusals = {
        n: how for n, (route, how) in nested.items() if route == r.REFUSED
    }
    if refusals:
        return r.refused(next(iter(refusals.values())), tuple(refusals))
    return r.with_nodes(
        r.EXACT,
        "The long-run distribution of the Markov chain of the "
        "components' states and the repair queue, solved exactly, and "
        "the system's values averaged over its states.",
        nested,
        "long-run values",
    )


def mean_availability(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
) -> float:
    """See ``RepairableRBD.mean_availability``."""
    # Good reference on the Availability of a system
    # https://www.diva-portal.org/smash/get/diva2:986067/FULLTEXT01.pdf
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working_nodes, broken_nodes)

    if rbd.ccf_groups:
        # Each group conditioned on within its module (#218).
        method = structure_method(method)
        _, weights, inputs = _ccf_groups._ccf_long_run(
            rbd, working_nodes, broken_nodes
        )
        up, down = _ccf_groups._ccf_tabled(rbd, *inputs).system()
        return float(weights @ (up if method == "p" else 1.0 - down))
    # Over one period of the inspection schedules, if any (see
    # _long_run_grid), or over the states of the repair crews' Markov
    # chain; otherwise at one point, as the node availabilities are
    # constant.
    availability, weights = _long_run_probabilities(
        rbd, working_nodes, broken_nodes
    )
    system = rbd.system_probability(availability, method=method)
    return float(weights @ system)


def node_availability(rbd) -> dict[Hashable, float]:
    """See ``RepairableRBD.node_availability``."""
    node_av: dict[Hashable, float] = {}
    for node_name in rbd.components:
        node_av[node_name] = _node_availability(rbd, node_name)

    for node_name in rbd.in_or_out:
        node_av[node_name] = 1.0

    return node_av


def _node_availability(rbd, node) -> float:
    """A component's long-run availability (see ``node_availability``):
    with limited repair crews, its long-run probability of being up in
    their Markov chain, or a nested RBD's own, as it has crews of its
    own."""
    _requirements._require_perfect_repair(rbd, node)
    _requirements._require_times(rbd, node)
    if rbd._crews_couple():
        chain = _crews._crew_chain(rbd)
        if node in chain.nodes:
            return chain.availability(node)
        return float(rbd.components[node].mean_availability())
    if node in rbd._standby:
        return _standby_long_run(rbd, node).availability
    if node in rbd._inspection:
        unit = _requirements._tested_unit(rbd, node)
        if unit is not None:
            return unit.long_run.availability
        life = _requirements._tested_life(rbd, node)
        if life is not None:
            return life.availability
        rate, interval = _requirements._inspected_rate(rbd, node)
        inspection = rbd._inspection[node]
        if inspection.partial:
            # Up with probability rho ** k * exp(-rate * u) (see
            # _tested_profile): averaged over a full test's cycle,
            # (1 - rho ** m) / ((1 - coverage) * rate * full_test).
            kept = _log_kept(rate, inspection)
            return float(
                -np.expm1(inspection.per_full_test * kept)
                / ((1.0 - inspection.coverage) * rate * inspection.period)
            )
        return float(-np.expm1(-rate * interval) / (rate * interval))
    schedule = rbd._preventive.get(node)
    if schedule is None:
        component = rbd.components[node]
        return float(np.atleast_1d(component.mean_availability())[0])
    up, cycle, _, _ = _maintenance_cycle(rbd, node, schedule)
    return min(1.0, up / cycle)


def _node_unavailability(rbd, node) -> float:
    """A component's long-run unavailability, as ``_node_availability``
    gives its availability, but worked out in its own right (from a
    standby group's down states, or the mean down time over a cycle),
    so that a small one keeps its precision. For a component whose
    value is constant over the long-run grid, and while the crews do
    not couple the components (see ``_long_run_unavailabilities``)."""
    _requirements._require_perfect_repair(rbd, node)
    _requirements._require_times(rbd, node)
    if node in rbd._standby:
        return _standby_long_run(rbd, node).unavailability
    component = rbd.components[node]
    schedule = rbd._preventive.get(node)
    if schedule is None:
        return float(np.atleast_1d(component.mean_unavailability())[0])
    up, cycle, fails, survives = _maintenance_cycle(rbd, node, schedule)
    if schedule.policy == "block":
        return max(0.0, (cycle - up) / cycle)
    # Down for a repair after a failure before the age (its probability
    # from the model's own ff), else for the maintenance.
    maintenance = (
        0.0 if schedule.duration is None else model_mean(schedule.duration)
    )
    down = (
        fails * model_mean(component.time_to_replace) + survives * maintenance
    )
    return min(1.0, down / cycle)


def _block_nodes(rbd) -> list:
    """The components renewed on a calendar, whose long-run values vary
    with it: under block replacement, or replaced on condition at
    inspections (#145)."""
    return [
        node
        for node, schedule in rbd._preventive.items()
        if schedule.policy in ("block", "condition")
    ]


def _block_cycle(rbd, node):
    """The renewal cycle of a component under block replacement (see
    ``_block_replacement``) or replaced on condition (see
    ``_condition_replacement``), computed once and kept."""
    _requirements._refuse_level(rbd, node)
    component = rbd.components[node]
    schedule = rbd._preventive[node]
    cache = rbd._cache.kept("block_cycles")
    key = (
        node,
        id(component.reliability),
        id(component.time_to_replace),
        float(schedule.interval),
        id(schedule.duration),
        schedule.policy,
        schedule.threshold,
    )
    if key not in cache:
        if schedule.policy == "condition":
            cache[key] = condition_cycle(
                component.reliability,
                component.time_to_replace,
                schedule.duration,
                schedule.interval,
                float(schedule.threshold),
                node,
            )
        else:
            cache[key] = block_cycle(
                component.reliability,
                component.time_to_replace,
                schedule.duration,
                schedule.interval,
                node,
            )
    return cache[key]


def _has_calendar(rbd) -> bool:
    """Whether a component here, or in a nested RBD, is inspected or
    replaced on a calendar: its long-run availability then varies with
    the time of the schedule."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return (
        bool(rbd._inspection)
        or bool(_block_nodes(rbd))
        or any(
            isinstance(c, RepairableRBD) and _has_calendar(c)
            for c in rbd.components.values()
        )
    )


def _unit_nodes(rbd) -> list:
    """The components with hidden failures whose values are worked out
    on a grid (see ``_tested_kind``)."""
    return [
        node
        for node in rbd._inspection
        if _requirements._tested_kind(rbd, node) == "unit"
    ]


def _calendar_grid(
    rbd, blocks: list, units: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """``_long_run_grid`` with components under block replacement, or
    tested on a grid (``units``): the middles of cells over one common
    period of the block and inspection intervals. The cells' edges are
    those of every block-replaced component's profile (its long-run
    availability over its interval, cell by cell), of every such tested
    component's (its grid's points across each test interval, and where
    its test's and its repair's CDFs bend), the block and inspection
    times, and enough points in between for an inspected component's
    availability to vary little across a cell; so each cell lies in one
    cell of every profile, and the mean over the cells is as exact as
    the profiles."""
    intervals = {rbd._preventive[node].interval for node in blocks}
    intervals |= {rbd._inspection[node].interval for node in rbd._inspection}
    schedules = list(rbd._inspection.values())
    period = _common_period(intervals | {s.period for s in schedules})

    def too_long() -> NoReturn:
        raise NotImplementedError(
            f"The block-replacement and inspection intervals "
            f"{sorted(intervals)} repeat together only after too long a "
            "time to average over finely enough: estimate the long-run "
            "values by simulation, with availability() or cost()."
        )

    # The grid's size, before it is built: a common period of many
    # repeats of the profiles (near-equal intervals, say) would not fit
    # in memory.
    size = 0
    for node in blocks:
        phase = _block_cycle(rbd, node).phase
        size += int(round(period / phase[-1])) * len(phase)
    for node in units or ():
        unit = _requirements._tested_unit(rbd, node).long_run  # type: ignore
        size += int(round(period / unit.period)) * len(unit.edges())
    for node in rbd._inspection:
        interval = rbd._inspection[node].interval
        per_interval = np.ceil(
            256.0 * _requirements._tested_rate(rbd, node) * interval
        )
        size += int(per_interval) * int(round(period / interval))
    if size > 4_000_000:
        too_long()
    pieces = [np.array([0.0, period])]
    for interval in intervals:
        pieces.append(interval * np.arange(int(round(period / interval))))
    for schedule in schedules:
        pieces.append(np.array(sorted(_tests_in(schedule, period))))
    for node in blocks:
        phase = _block_cycle(rbd, node).phase
        repeats = int(round(period / phase[-1]))
        pieces.append(
            (
                phase[-1] * np.arange(repeats)[:, None] + phase[None, :-1]
            ).ravel()
        )
    for node in units or ():
        tested = _requirements._tested_unit(rbd, node)
        long_run = tested.long_run  # type: ignore
        repeats = int(round(period / long_run.period))
        edges = (
            rbd._inspection[node].offset
            + long_run.period * np.arange(repeats)[:, None]
            + long_run.edges()[None, :]
        ).ravel()
        pieces.append(edges - period * np.floor(edges / period))
    for node in rbd._inspection:
        rate = _requirements._tested_rate(rbd, node)
        interval = rbd._inspection[node].interval
        per_interval = int(np.ceil(256.0 * rate * interval))
        pieces.append(
            np.linspace(
                0.0,
                period,
                1 + per_interval * int(round(period / interval)),
            )
        )
    edges = np.concatenate(pieces)
    if len(edges) > 4_000_000:
        too_long()
    # Edges closer than rounding are one.
    edges = np.unique(np.round(edges / period, 12)) * period
    middle, width = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    if not units:
        return middle, width / period
    # A tested component's test and repair times have exact CDFs in its
    # profile, which may bend sharply within a cell (a test of hours,
    # say): two Gauss-Legendre points a cell, exact for its linear grid
    # values, and to the cell's fourth power for the rest.
    side = 0.5 * width / np.sqrt(3.0)
    times = np.stack([middle - side, middle + side], axis=1).ravel()
    return times, np.repeat(0.5 * width / period, 2)


def _block_profile(rbd, node, times: np.ndarray, rates: bool = False):
    """A block-replaced component's long-run availability (or failure
    intensity) at each of ``times``: the value of its profile's cell
    (over a block interval) that the time falls in."""
    cycle = _block_cycle(rbd, node)
    interval = float(cycle.phase[-1])
    values = cycle.failure_rate if rates else cycle.availability
    phase = times - interval * np.floor(times / interval)
    cell = np.searchsorted(cycle.phase, phase, side="right") - 1
    return values[np.clip(cell, 0, len(values) - 1)]


def _calendar_outages(rbd, working_nodes, broken_nodes) -> float:
    """The system's planned outages per unit time, in the long run,
    at exact times on the calendar: replacements at block times that
    take time, and tests that take time, each taking a unit working
    then off line. At each such time, the probability that the system
    is up just before and down just after, which (as they only take
    units down) is the fall in the system availability: the rise in its
    unavailability, a difference of small values in a reliable system,
    not of values near 1. With common-cause groups, see
    ``_ccf_groups._ccf_calendar_outages``."""
    changes = _calendar_changes(rbd, working_nodes, broken_nodes)
    if changes is None:
        return 0.0
    period, _, before, after = changes
    rise = rbd._system_unreliability(*after) - rbd._system_unreliability(
        *before
    )
    return float(np.sum(rise)) / period


def _calendar_changes(rbd, working_nodes, broken_nodes):
    """Where the system's planned outages at exact times on the calendar
    start (see ``_calendar_outages``): None if none do, else the common
    period, the instants in it, and the nodes' availabilities and
    unavailabilities just before and just after each (the forced nodes
    held at 1 or 0). The nodes due at an instant take their values then
    from their own models; the others are at the instant, before its
    outages start. Units due at the same time go down together; an
    instant inspection due then comes first."""
    blocks = [
        node
        for node in _block_nodes(rbd)
        if rbd._preventive[node].duration is not None
    ]
    units = [
        node
        for node in _unit_nodes(rbd)
        if _requirements._tested_unit(rbd, node).setup.timed  # type: ignore
    ]
    if not blocks and not units:
        return None
    intervals = {rbd._preventive[node].interval for node in blocks}
    intervals |= {rbd._inspection[node].period for node in rbd._inspection}
    period = _common_period(intervals)
    # At each instant (as a share of the period), the nodes due then,
    # with their values up and down just before and just after.
    due: dict = {}
    for node in blocks:
        cycle = _block_cycle(rbd, node)
        values = (
            (cycle.before, 1.0 - cycle.before),
            (cycle.after, 1.0 - cycle.after),
        )
        interval = rbd._preventive[node].interval
        for k in range(int(round(period / interval))):
            instant = round(k * interval / period, 12)
            due.setdefault(instant, []).append((node, values))
    for node in units:
        inspection = rbd._inspection[node]
        tested = _requirements._tested_unit(rbd, node)
        long_run = tested.long_run  # type: ignore
        for k in range(int(round(period / inspection.interval))):
            time = inspection.offset + k * inspection.interval
            time -= period * math.floor(time / period)
            due.setdefault(round(time / period, 12), []).append(
                (node, long_run.around(k % long_run.per))
            )
    instants = np.array(sorted(due)) * period
    # Just after each instant, before its outages start (an
    # inspection due then is done).
    just_after = instants + 1e-9 * period
    up, down = _availabilities_at(rbd, just_after), (
        _unavailabilities_at(rbd, just_after)
    )
    before, after = dict(up), dict(up)
    before_down, after_down = dict(down), dict(down)
    for column, key in enumerate(sorted(due)):
        for node, values in due[key]:
            for works, fails, (value, failed) in zip(
                (before, after), (before_down, after_down), values
            ):
                works[node] = np.array(works[node], dtype=float)
                fails[node] = np.array(fails[node], dtype=float)
                works[node][column] = value
                fails[node][column] = failed
    return (
        period,
        instants,
        (
            rbd._probabilities_with_overrides(
                before, working_nodes, broken_nodes
            ),
            rbd._failures_with_overrides(
                before_down, working_nodes, broken_nodes
            ),
        ),
        (
            rbd._probabilities_with_overrides(
                after, working_nodes, broken_nodes
            ),
            rbd._failures_with_overrides(
                after_down, working_nodes, broken_nodes
            ),
        ),
    )


def _long_run_grid(rbd) -> Tuple[np.ndarray, np.ndarray]:
    """The times, and their weights (which sum to 1), that the exact
    long-run values average over.

    Components with hidden failures are up with a probability that falls
    between inspections and is restored at each one, and components
    inspected at the same times are down together, so the system's
    long-run values are averages over one period of the inspection
    schedules: by Gauss-Legendre quadrature between consecutive
    inspections, on pieces short enough (a failure rate times their
    length at most 1) for it to be exact to rounding. With no such
    components they are constant: one time, of weight 1. A nested RBD
    with hidden failures enters through its own long-run values, which
    is exact only if nothing else varies with the inspections. With a
    component whose profile is on a grid (block-replaced, or tested
    with tests or repairs that take time: see ``_calendar_grid``), the
    middles of the grids' cells.
    """
    _crews._require_unlimited_crews(rbd)
    _requirements._require_calendars(rbd)
    blocks = _block_nodes(rbd)
    units = _unit_nodes(rbd)
    if blocks or units:
        return _calendar_grid(rbd, blocks, units)
    if not rbd._inspection:
        return np.zeros(1), np.ones(1)
    rates = {
        node: (
            _requirements._tested_rate(rbd, node),
            rbd._inspection[node].interval,
        )
        for node in rbd._inspection
    }
    intervals = {interval for _, interval in rates.values()}
    schedules = list(rbd._inspection.values())
    period = _common_period(intervals | {s.period for s in schedules})
    breaks = {0.0, period}
    for interval in intervals:
        count = int(round(period / interval))
        breaks.update(k * interval for k in range(1, count))
    for schedule in schedules:
        breaks.update(_tests_in(schedule, period))
    fastest = max(rate for rate, _ in rates.values())
    # The tests of components whose lives are not exponential: a unit
    # renewed there may not age smoothly from it (a Weibull life's
    # survival, say), so the pieces after it close in on it.
    graded: set = set()
    for node, schedule in rbd._inspection.items():
        life = _requirements._tested_life(rbd, node)
        if life is None:
            continue
        count = int(round(period / schedule.interval))
        starts = schedule.offset + schedule.interval * np.arange(count)
        starts = np.where(starts >= period, starts - period, starts)
        graded.update(round(float(s) / period, 12) for s in starts)
        # Where its life bends within an interval (a threshold, say).
        bends = (starts[:, None] + life.bends[None, :]).ravel()
        breaks.update(float(b) for b in np.mod(bends, period))
    edges = np.array(sorted(breaks))
    points, weights = np.polynomial.legendre.leggauss(16)
    times, masses = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        pieces = np.linspace(a, b, max(1, math.ceil((b - a) * fastest)) + 1)
        if round(float(a) / period, 12) in graded:
            width = pieces[1] - pieces[0]
            pieces = np.concatenate(
                [
                    pieces[:1],
                    a + width * 2.0 ** -np.arange(48.0, 0.0, -1.0),
                    pieces[1:],
                ]
            )
        for lo, hi in zip(pieces[:-1], pieces[1:]):
            half = 0.5 * (hi - lo)
            times.append(lo + half * (points + 1.0))
            masses.append(half * weights)
    return np.concatenate(times), np.concatenate(masses) / period


def _tested_profile(
    rbd, node, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """A component with hidden failures: its probabilities of being up
    and down at each of ``times``, in the long run.

    Tested at ``offset`` and every ``interval`` after it, it is up with
    probability ``exp(-rate * u)``, ``u`` the time since its last test.
    A test that can miss a failure (a ``coverage`` ``c`` below 1) finds
    what it can: up just after the ``k``-th test since a full one with
    probability ``rho ** k``, where ``rho = 1 - (1 - c) * (1 -
    exp(-rate * interval))`` keeps the units that were up or whose
    failure it found; so up with probability ``rho ** k * exp(-rate *
    u)``. Down is worked out in its own right, by ``expm1``. Any other
    life's profile is its ``TestedLife``'s (#144), and with tests or
    repairs that take time, or tests that can miss a failure of a life
    that is not exponential, its ``TestedLongRun``'s (#159)."""
    unit = _requirements._tested_unit(rbd, node)
    if unit is not None:
        return unit.long_run.profile(
            _requirements._unit_phase(rbd, node, times)
        )
    life = _requirements._tested_life(rbd, node)
    if life is not None:
        return life.profile(_requirements._tested_phase(rbd, node, times))
    rate, interval = _requirements._inspected_rate(rbd, node)
    inspection = rbd._inspection[node]
    position = times - inspection.offset if inspection.offset else times
    if not inspection.partial:
        since = position - interval * np.floor(position / interval)
        return np.exp(-rate * since), -np.expm1(-rate * since)
    cycle = inspection.period
    within = position - cycle * np.floor(position / cycle)
    tests = np.clip(
        np.floor(within / interval), 0, inspection.per_full_test - 1
    )
    exponent = tests * _log_kept(rate, inspection) - rate * (
        within - tests * interval
    )
    return np.exp(exponent), -np.expm1(exponent)


def _availabilities_at(rbd, times: np.ndarray) -> dict:
    """Every node's availability at each of ``times`` (see
    ``_long_run_grid``): ``exp(-lambda * u)``, ``u`` the time since the
    last inspection, for a component with hidden failures; its constant
    long-run availability for any other."""
    out: dict = {}
    blocks = set(_block_nodes(rbd))
    for node in rbd.components:
        if node in rbd._inspection:
            out[node] = _tested_profile(rbd, node, times)[0]
        elif node in blocks:
            out[node] = _block_profile(rbd, node, times)
        else:
            out[node] = np.full(len(times), _node_availability(rbd, node))
    for node in rbd.in_or_out:
        out[node] = np.ones(len(times))
    return out


def _unavailabilities_at(rbd, times: np.ndarray) -> dict:
    """Every node's unavailability at each of ``times``, as
    ``_availabilities_at`` gives their availabilities, each worked out
    in its own right so that a small one keeps its precision:
    ``1 - exp(-lambda * u)`` by ``expm1`` at a time ``u`` since a test,
    for a component with hidden failures, and ``_node_unavailability``
    for one whose value is constant.
    (A block-replaced component's profile is numerical, to about 1e-7,
    so one less it loses nothing.)"""
    out: dict = {}
    blocks = set(_block_nodes(rbd))
    for node in rbd.components:
        if node in rbd._inspection:
            out[node] = _tested_profile(rbd, node, times)[1]
        elif node in blocks:
            out[node] = 1.0 - _block_profile(rbd, node, times)
        else:
            out[node] = np.full(len(times), _node_unavailability(rbd, node))
    for node in rbd.in_or_out:
        out[node] = np.zeros(len(times))
    return out


def _long_run_unavailabilities(
    rbd, working_nodes, broken_nodes
) -> Tuple[dict, dict, np.ndarray]:
    """``_long_run_probabilities``, with every node's unavailability
    too (at each time of ``_long_run_grid``, or in each state of the
    repair crews' Markov chain), each worked out in its own right, so
    that the system's unavailability keeps a small one's precision."""
    probabilities, failures, weights, _ = _long_run_points(
        rbd, working_nodes, broken_nodes
    )
    return probabilities, failures, weights


def _long_run_points(
    rbd, working_nodes, broken_nodes
) -> Tuple[dict, dict, np.ndarray, Optional[np.ndarray]]:
    """``_long_run_unavailabilities``, and each point's time's position
    in ``_long_run_grid``, or None in the states of the repair crews'
    chain."""
    if rbd._crews_couple():
        probabilities, weights = _crews._chain_probabilities(
            rbd, working_nodes, broken_nodes
        )
        forced = set(working_nodes or ()) | set(broken_nodes or ())
        chain = _crews._crew_chain(rbd, frozenset(forced))
        # In each state a node in the chain, or held working or broken,
        # is up or down for certain: one less it is exact.
        failures = {node: 1.0 - value for node, value in probabilities.items()}
        for node, component in rbd.components.items():
            if node not in forced and node not in chain.nodes:
                failures[node] = np.full(
                    len(weights), float(component.mean_unavailability())
                )
        return probabilities, failures, weights, None
    times, weights = _long_run_grid(rbd)
    probabilities = rbd._probabilities_with_overrides(
        _availabilities_at(rbd, times), working_nodes, broken_nodes
    )
    failures = rbd._failures_with_overrides(
        _unavailabilities_at(rbd, times), working_nodes, broken_nodes
    )
    # (Common-cause groups are conditioned on module by module, see
    # _ccf_tabled, by every method that would come here with them.)
    assert not rbd.ccf_groups
    return probabilities, failures, weights, np.arange(len(times))


def _long_run_probabilities(
    rbd, working_nodes, broken_nodes, what: str = "This analysis"
) -> Tuple[dict, np.ndarray]:
    """The node availabilities the long-run values are evaluated at
    (with the forced nodes held at 1 or 0), over the times of
    ``_long_run_grid``, and those times' weights; or, with limited
    repair crews, over the states of their Markov chain (see
    ``_chain_probabilities``). Each value is then a ratio of averages
    of system quantities. With common-cause groups, the times are split
    by every combination of their states (see ``_with_ccf_groups``,
    which ``what`` refuses where they are too many): for the capacity
    distribution, which takes the nodes' joint states."""
    if rbd.ccf_groups:
        # The groups' chains refuse the crews (see _ccf_rates).
        _ccf_groups._require_ccf_long_run(rbd)
    if rbd._crews_couple():
        return _crews._chain_probabilities(rbd, working_nodes, broken_nodes)
    times, weights = _long_run_grid(rbd)
    probabilities = rbd._probabilities_with_overrides(
        _availabilities_at(rbd, times), working_nodes, broken_nodes
    )
    if rbd.ccf_groups:
        _ccf_groups._require_free_members(rbd, working_nodes, broken_nodes)
        probabilities, _, weights, _ = _ccf_groups._with_ccf_groups(
            rbd, times, probabilities, None, weights, what=what
        )
    return probabilities, weights


def _standby_rates(rbd, node) -> Tuple[float, float]:
    """A standby group's units' failure and repair rates, for its exact
    long-run values; raise if their lives and repair times are not
    exponential."""
    component = rbd.components[node]
    life = _constant_rate(component.reliability)
    repair = _constant_rate(component.time_to_replace)
    if life is None or repair is None:
        raise NotImplementedError(
            f"Component {node!r} is a standby group, whose exact "
            "long-run values come from a Markov chain of its units, "
            "which needs their lives and repair times exponential: "
            "simulate it with availability() or cost()."
        )
    return life, repair


def _standby_long_run(rbd, node) -> "_standby_chain.StandbyLongRun":
    """A standby group's long-run values, from its Markov chain (see
    ``_standby_chain``). The crews do not tie it to other nodes here (see
    ``_crews_couple``): its units' repairs wait only for each other,
    with ``repair_crews`` crews when they are the crews' only jobs."""
    life, repair = _standby_rates(rbd, node)
    arrangement = rbd._standby[node]
    return _standby_chain.long_run(
        arrangement.units,
        arrangement.k,
        life,
        repair,
        arrangement.dormancy_factor,
        arrangement.switching_probability,
        rbd.repair_crews if _crews._crews_limited(rbd) else None,
    )


def _node_frequencies(rbd, node) -> Tuple[float, float, float]:
    """A component's long-run failures, preventive replacements and
    planned outages, per unit time: recursive for a nested
    RepairableRBD (whose own preventive replacements are not this RBD's
    to count), ``1 / (MTTF + MTTR)`` failures for a NonRepairable, and
    from its renewal cycle for one under age replacement."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    _requirements._require_perfect_repair(rbd, node)
    _requirements._require_times(rbd, node)
    component = rbd.components[node]
    if isinstance(component, RepairableRBD):
        failures, planned = _outage_frequencies(component)
        return failures, 0.0, planned
    if node in rbd._standby:
        return _standby_long_run(rbd, node).failure_frequency, 0.0, 0.0
    if node in rbd._inspection:
        # At most one failure per inspection interval: the unit, down
        # from its failure, is renewed at the inspection that finds it.
        unit = _requirements._tested_unit(rbd, node)
        if unit is not None:
            # Its tests that take it off line, working, are planned
            # outages.
            long_run = unit.long_run
            return long_run.failures, 0.0, long_run.planned
        life = _requirements._tested_life(rbd, node)
        if life is not None:
            return life.failures, 0.0, 0.0
        rate, interval = _requirements._inspected_rate(rbd, node)
        if rbd._inspection[node].partial:
            # It fails at its constant rate while it is up.
            return rate * _node_availability(rbd, node), 0.0, 0.0
        return float(-np.expm1(-rate * interval) / interval), 0.0, 0.0
    schedule = rbd._preventive.get(node)
    if schedule is None:
        return component.failure_frequency(), 0.0, 0.0
    _, cycle, failures, maintenances = _maintenance_cycle(rbd, node, schedule)
    maintained = maintenances / cycle
    planned = 0.0 if schedule.duration is None else maintained
    return failures / cycle, maintained, planned


def _maintenance_cycle(
    rbd, node, schedule: _Preventive
) -> Tuple[float, float, float, float]:
    """A component's renewal cycle under preventive maintenance: its mean
    up time, mean length, mean number of failures and mean number of
    preventive replacements.

    Under age replacement a cycle ends at a failure or at a preventive
    replacement, whichever comes first. Under block replacement it runs
    from one block time at which the unit is up (and replaced) to the
    next, with any failures and repairs in between (see
    ``_block_replacement``); replaced on condition, from one inspection
    that replaces the unit to the next (see ``_condition_replacement``).
    It is computed once and kept."""
    _requirements._require_no_opportunities(rbd, node)
    _requirements._require_perfect_repair(rbd, node)
    component = rbd.components[node]
    if schedule.policy == "block":
        block = _block_cycle(rbd, node)
        return block.up, block.length, block.failures, 1.0
    if schedule.policy == "condition":
        kept = _block_cycle(rbd, node)
        return kept.up, kept.length, kept.failures, kept.replaced
    # Kept by interval, as a search over intervals revisits them.
    cache = rbd._cache.kept("age_cycles")
    key = (
        node,
        id(component.reliability),
        id(component.time_to_replace),
        float(schedule.interval),
        id(schedule.duration),
    )
    if key in cache:
        return cache[key]
    up = float(component.avg_replacement_time(schedule.interval))
    survives = float(
        np.ravel(component.reliability_function(schedule.interval))[0]
    )
    fails = _failed_by(component, schedule.interval)
    maintenance = (
        0.0 if schedule.duration is None else model_mean(schedule.duration)
    )
    cycle = (
        up
        + fails * model_mean(component.time_to_replace)
        + survives * maintenance
    )
    cache[key] = (up, cycle, fails, survives)
    return cache[key]


def system_failure_frequency(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
) -> float:
    """See ``RepairableRBD.system_failure_frequency``."""
    return _outage_frequencies(rbd, working_nodes, broken_nodes)[0]


def _outage_frequencies(
    rbd,
    working_nodes: Optional[Collection[Hashable]] = None,
    broken_nodes: Optional[Collection[Hashable]] = None,
) -> Tuple[float, float]:
    """The system's long-run failures and planned outages per unit time:
    ``sum_i I_B(i) * omega_i`` over the nodes' failures, and the same
    over their planned outages (a node's outage takes the system down
    when the node is critical, which it is with probability
    ``I_B(i)``): the sum of ``_outage_terms``."""
    terms, planned = _outage_terms(rbd, working_nodes, broken_nodes)
    failures = 0.0
    for _, term in terms:
        failures += term
    return failures, planned


def _outage_terms(
    rbd,
    working_nodes: Optional[Collection[Hashable]] = None,
    broken_nodes: Optional[Collection[Hashable]] = None,
) -> Tuple[List[Tuple[Any, float]], float]:
    """The system's long-run failures per unit time, as the terms of
    their sum, each with what causes it (a node, or a common-cause
    group's members together, for its causes that strike more than one
    of them), and its planned outages per unit time (see
    ``_outage_frequencies``). With limited repair crews, see
    ``_chain_outage_terms``; with common-cause groups,
    ``_ccf_outage_terms``."""
    if rbd.ccf_groups:
        return _ccf_groups._ccf_outage_terms(rbd, working_nodes, broken_nodes)
    if rbd._crews_couple():
        return _crews._chain_outage_terms(rbd, working_nodes, broken_nodes)
    times, weights = _long_run_grid(rbd)
    availability = rbd._probabilities_with_overrides(
        _availabilities_at(rbd, times), working_nodes, broken_nodes
    )
    unavailability = rbd._failures_with_overrides(
        _unavailabilities_at(rbd, times), working_nodes, broken_nodes
    )
    forced = (set() if working_nodes is None else set(working_nodes)) | (
        set() if broken_nodes is None else set(broken_nodes)
    )
    # Each node's Birnbaum importance, from the nodes' unavailabilities
    # as well as their availabilities, so that a small one (a node
    # backed up by reliable redundancy) keeps its precision.
    birnbaum = rbd._birnbaum_importance(
        availability, node_failures=unavailability
    )
    terms: List[Tuple[Any, float]] = []
    planned = 0.0
    blocks = set(_block_nodes(rbd))
    for node in rbd.components:
        if node in forced:
            # A forced node never changes state, so it contributes no
            # system failures.
            continue
        importance = np.asarray(birnbaum[node])
        if node in rbd._inspection:
            # It fails at its constant rate whenever it is up (any other
            # life: at its profile's rate).
            node_failures: Any = _requirements._tested_intensity(
                rbd, node, times, availability[node]
            )
            node_planned: Any = 0.0
        elif node in blocks:
            # Its failure intensity varies over its block interval; its
            # replacements fall at block times (counted below).
            node_failures = _block_profile(rbd, node, times, rates=True)
            node_planned = 0.0
        else:
            node_failures, _, node_planned = _node_frequencies(rbd, node)
        terms.append((node, float(weights @ (importance * node_failures))))
        planned += float(weights @ (importance * node_planned))
    planned += _calendar_outages(rbd, working_nodes, broken_nodes)
    return terms, planned


def mean_time_between_failures(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
) -> float:
    """See ``RepairableRBD.mean_time_between_failures``."""
    omega = rbd.system_failure_frequency(working_nodes, broken_nodes)
    return 1.0 / omega if omega > 0.0 else float("inf")


def mean_up_time(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
) -> float:
    """See ``RepairableRBD.mean_up_time``."""
    availability = rbd.mean_availability(working_nodes, broken_nodes)
    omega = sum(_outage_frequencies(rbd, working_nodes, broken_nodes))
    if omega > 0.0:
        return float(availability) / omega
    return float("inf") if availability > 0.0 else 0.0


def mean_down_time(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
) -> float:
    """See ``RepairableRBD.mean_down_time``."""
    unavailability = rbd.mean_unavailability(working_nodes, broken_nodes)
    omega = sum(_outage_frequencies(rbd, working_nodes, broken_nodes))
    if omega > 0.0:
        return unavailability / omega
    return 0.0 if unavailability <= 0.0 else float("inf")
