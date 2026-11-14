"""A ``RepairableRBD``'s common-cause groups: each cause's rate and timing,
the groups' chains (``_ccf_chain``) over the members' states, in the long
run and over time, and the system worked out with them, conditioned on
module by module (``_ccf_modules.Tabled``) or on every combination of the
groups' members (``_with_ccf_groups``); with the checks that refuse what
the chains cannot take. ``RepairableRBD``'s methods call these.
"""

from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd import (
    _ccf_chain,
    _ccf_modules,
    _curves,
    _long_run,
    _requirements,
)
from repyability.rbd._common import (
    _common_period,
    _constant_rate,
    _fixed_length,
    _safe_mean,
    _squeeze_values,
)
from repyability.rbd._events import (
    _Cause,
)
from repyability.rbd._model_utils import distribution_name, never_fails
from repyability.rbd.modular import GraphStructure
from repyability.utils.checks import (
    structure_method,
)

if TYPE_CHECKING:
    pass


def _has_ccf(rbd) -> bool:
    """Whether this RBD, or one nested in it, has common-cause
    groups."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return bool(getattr(rbd, "ccf_groups", ())) or any(
        isinstance(c, RepairableRBD) and _has_ccf(c)
        for c in rbd.components.values()
    )


def _ccf_rates(rbd, group) -> Tuple[float, Optional[float]]:
    """A common-cause group's members' failure rate, and their repair
    rate (None for hidden failures, found by tests), for its chain
    (see ``_ccf_chain``); raise if the chain does not cover them."""
    where = f"Common-cause group {list(group.members)}"
    if rbd._crews_couple():
        raise NotImplementedError(
            f"{where}: with limited repair crews, the crews' Markov "
            "chain does not take common causes in, as yet. Estimate the "
            "values by simulation, with availability() or cost(), which "
            "take both."
        )
    rates = set()
    for member in group.members:
        for kinds, what in (
            (rbd._preventive, "scheduled maintenance"),
            (rbd._imperfect, "imperfect repair"),
            (rbd._member_group, "a maintenance group"),
        ):
            if member in kinds:
                raise NotImplementedError(
                    f"{where}: member {member!r} has {what}, which the "
                    "group's chain does not take in."
                )
        life = _constant_rate(rbd.components[member].reliability)
        if life is None:
            raise NotImplementedError(
                f"{where}: its chain needs exponential lives (a constant "
                f"failure rate), and the life of member {member!r} is "
                "not. The simulations need one too, as the rate is "
                "split between the causes."
            )
        rates.add(life)
    tested = [member in rbd._inspection for member in group.members]
    if any(tested) and not all(tested):
        raise NotImplementedError(
            f"{where}: some members' failures are hidden and others' "
            "revealed; the group's chain takes one kind."
        )
    life = rates.pop()
    if all(tested):
        # Tests and repairs of no time, a fixed one or an exponential
        # one (#220).
        _ccf_timings(rbd, group)
        for member in group.members:
            _requirements._require_tested_exact(rbd, member)
        if len({rbd._inspection[m].coverage for m in group.members}) > 1:
            raise NotImplementedError(
                f"{where}: its members' tests have different coverages; "
                "the group's chain takes one, which a cause's failures "
                "are found by alike."
            )
        return life, None
    component = rbd.components[group.members[0]]
    repair = _constant_rate(component.time_to_replace)
    if repair is None:
        raise NotImplementedError(
            f"{where}: the chain of members whose failures are revealed "
            "needs exponential repairs, and theirs are not. Estimate the "
            "values by simulation, with availability() or cost(), which "
            "take repairs of any length."
        )
    return life, repair


def _ccf_timings(rbd, group) -> Optional[List["_ccf_chain.Timing"]]:
    """How long the tests and repairs of a common-cause group's members
    (whose failures are hidden) take, for its chain (#220): each a
    ``_ccf_chain.Duration`` of no time, a fixed time, an exponential one,
    or (#106) one of any other length, a surpyval parametric model; None
    where none takes time. Raise for what the chain does not take, saying
    what to do: a model that is not a surpyval parametric one or whose
    lengths do not all end, and a test and repair that can run into the
    member's next test (but for a chance below ``_ccf_chain.TAIL``)."""
    if not all(member in rbd._inspection for member in group.members):
        return None  # revealed failures (see _ccf_rates)
    where = f"Common-cause group {list(group.members)}"
    out = []
    for member in group.members:
        inspection = rbd._inspection[member]
        parts = []
        for what, model in (
            ("tests", inspection.duration),
            ("repairs", rbd.components[member].time_to_replace),
        ):
            if model is None or _safe_mean(model) == 0.0:
                parts.append(_ccf_chain.Duration())
                continue
            fixed = _fixed_length(model)
            if fixed is not None:
                parts.append(_ccf_chain.Duration(fixed=fixed))
                continue
            rate = _constant_rate(model)
            if rate is not None:
                parts.append(_ccf_chain.Duration(rate=rate))
                continue
            if distribution_name(model) is None or never_fails(model) > 0.0:
                raise NotImplementedError(
                    f"{where}: its chain takes tests and repairs of a "
                    "fixed length or of a surpyval parametric model's, "
                    f"every one ending, and the {what} of member "
                    f"{member!r} are neither. Estimate the values by "
                    "simulation, with availability() or cost()."
                )
            parts.append(_ccf_chain.Duration(model=model))
        timing = _ccf_chain.Timing(*parts)
        # Fixed lengths end at fixed times, and exponential ones at their
        # rates, whenever they end; one that varies is followed from the
        # test that started it, which must be over by the next.
        varies = any(timing.varies)
        length = (
            timing.test.tail() + timing.repair.tail()
            if varies
            else timing.test.fixed + timing.repair.fixed
        )
        if length >= inspection.interval:
            chance = (
                f" (but for a chance below {_ccf_chain.TAIL:g})"
                if varies
                else ""
            )
            raise NotImplementedError(
                f"{where}: the test and repair of member {member!r} take "
                f"{length:g} together{chance}, as long as its test "
                f"interval ({inspection.interval:g}) or longer, which its "
                "chain does not take. Estimate the values by simulation, "
                "with availability() or cost()."
            )
        out.append(timing)
    if not any(timing.takes_time for timing in out):
        return None
    return out


def _timings_vary(rbd) -> bool:
    """Whether a common-cause group's member has a test or repair that
    ends at a rate varying with the time since its test (#106, see
    ``_ccf_chain.Timing.varies``)."""
    return any(
        any(timing.varies)
        for group in rbd.ccf_groups
        for timing in _ccf_timings(rbd, group) or ()
    )


def _timing_rates(rbd) -> List[float]:
    """The rates at which the common-cause groups' members' tests and
    repairs of an exponential length end (#220), and one over the median
    of those of a general length (#106), the time scales of
    their chains' transients after a test, for the integrals that
    refine to them (see ``_windows._mission``)."""
    out = []
    for group in rbd.ccf_groups:
        for timing in _ccf_timings(rbd, group) or ():
            for duration in (timing.test, timing.repair):
                if duration.random:
                    out.append(1.0 / _ccf_chain._scale(duration))
    return out


def _shared_causes(rbd) -> List[_Cause]:
    """The common-cause groups' shared causes, as the simulation strikes
    them (see ``_Cause``): each a Poisson process at its share of the
    members' failure rate. A member's own share is its own life (see
    ``_own_rate``)."""
    out = []
    for group in rbd.ccf_groups:
        rate = _constant_rate(rbd.components[group.members[0]].reliability)
        if rate is None:
            continue  # refused (see _require_groups_simulated)
        coins = any(
            member in rbd._inspection and rbd._inspection[member].partial
            for member in group.members
        )
        for struck, cause in _ccf_chain.causes(
            group.model, group.members, rate
        ):
            if len(struck) > 1:
                out.append(
                    _Cause(
                        tuple(group.members[i] for i in struck),
                        cause,
                        coins,
                    )
                )
    return out


def _own_rate(rbd, node) -> Optional[float]:
    """A common-cause group member's own cause's rate, at which the
    simulation draws its life, the shared causes striking it too (see
    ``_shared_causes``): 0 if every cause it has is shared; None for
    any other node (and for a member whose life is not exponential,
    which the simulation refuses, see ``_require_groups_simulated``)."""
    for group in rbd.ccf_groups:
        if node in group.members:
            rate = _constant_rate(rbd.components[node].reliability)
            if rate is None:
                return None
            for struck, cause in _ccf_chain.causes(
                group.model, group.members, rate
            ):
                if struck == (group.members.index(node),):
                    return float(cause)
    return None


def _require_groups_simulated(rbd, states: Optional[dict] = None):
    """Raise unless the simulation takes the common-cause groups in
    (#158): each member's life exponential, as its rate is split between
    the causes; no member maintained on a schedule, repaired
    imperfectly, in a maintenance group, or started with a life of its
    own. A member started from a state starts from it (#106): see
    ``_past_strikes``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    for group in rbd.ccf_groups:
        where = f"Common-cause group {list(group.members)}"
        for member in group.members:
            life = rbd.components[member].reliability
            if _constant_rate(life) is None:
                raise NotImplementedError(
                    f"{where}: its members' failure rate is split "
                    "between the causes, which needs exponential lives, "
                    f"and the life of member {member!r} is not."
                )
            for kinds, what in (
                (rbd._preventive, "scheduled maintenance"),
                (rbd._imperfect, "imperfect repair"),
                (rbd._member_group, "a maintenance group"),
            ):
                if member in kinds:
                    raise NotImplementedError(
                        f"{where}: member {member!r} has {what}, which "
                        "the simulation of its group does not take in, "
                        "as yet."
                    )
            start = (states or {}).get(member)
            if start is not None and start.own_life:
                raise NotImplementedError(
                    f"{where}: member {member!r} has a life of its own "
                    "(a measured level or its own model), and a member's "
                    "life is its group's exponential one, split between "
                    "its causes: leave it out."
                )
    for component in rbd.components.values():
        if isinstance(component, RepairableRBD):
            _require_groups_simulated(component)


def _require_ccf_long_run(rbd) -> None:
    """Raise if a common-cause group's chain does not cover it."""
    for group in rbd.ccf_groups:
        _ccf_rates(rbd, group)


def _calendar_period(rbd) -> float:
    """The period of the schedules the long-run values average over
    (see ``_long_run_grid``): of the tests and block replacements."""
    intervals = {
        rbd._preventive[node].interval for node in _long_run._block_nodes(rbd)
    }
    for schedule in rbd._inspection.values():
        intervals |= {schedule.interval, schedule.period}
    return _common_period(intervals) if intervals else 1.0


def _group_states(rbd, group, times: np.ndarray, counts=None):
    """A common-cause group's members' joint states, at each of
    ``times`` (of ``_long_run_grid``), in the long run (see
    ``_ccf_chain``). With ``counts``, each member's number of copies,
    which join the group (#158, a ``BetaFactor`` model's beta holding
    at any size, see ``_ccf_chain._Counted``), tested with it: the
    states of the members' nodes, each down while all its copies
    are."""
    life, repair = _ccf_rates(rbd, group)
    counts = [1] * len(group.members) if counts is None else list(counts)
    # A member with no copies counted (one made perfect, by the
    # allocation's bounds) is left out: the others' states are as with
    # it.
    kept = [i for i, n in enumerate(counts) if n > 0]
    if not kept:
        return _ccf_chain.GroupStates(
            (), np.zeros((1, 0), dtype=bool), np.ones((len(times), 1))
        )
    members = [group.members[i] for i in kept]
    copies = [int(counts[i]) for i in kept]
    if repair is not None:
        states = _ccf_chain.revealed(
            group.model, members, life, repair, copies
        )
        return states._replace(
            probabilities=np.repeat(states.probabilities, len(times), axis=0)
        )
    tests, period = _group_tests(rbd, group)
    place = {old: new for new, old in enumerate(kept)}
    tests = [
        test._replace(member=place[test.member])
        for test in tests
        if test.member in place
    ]
    coverage = rbd._inspection[group.members[0]].coverage
    timings = _ccf_timings(rbd, group)
    if timings is not None and (len(kept) < len(counts) or max(copies) > 1):
        raise NotImplementedError(
            f"Common-cause group {list(group.members)}: its members' "
            "tests or repairs take time, and copies of them, which the "
            "allocation's chains count, are not taken in, as yet. Leave "
            "its members out of the nodes to allocate."
        )
    return _ccf_chain.hidden(
        group.model,
        members,
        life,
        coverage,
        tests,
        period,
        times,
        copies,
        timings,
    )


def _shifted_calendar(rbd, node, start) -> Optional[float]:
    """The time since the last test of a member started from ``start``
    where its state places its calendar (its ``since_scheduled``, as the
    simulation's ``_phases`` take it: tests then at ``k T - phase``, no
    offset); None where its calendar is its own (from its offset)."""
    if start is None or start.since_scheduled is None:
        return None
    if start.since_scheduled or rbd._inspection[node].offset:
        return float(start.since_scheduled)
    return None


def _group_tests(rbd, group, states: Optional[dict] = None):
    """A common-cause group's members' tests in one period of the
    calendar (``(0, period]``, see ``_calendar_period``), and the
    period: each member's on its own calendar, or on the one its state
    in ``states`` places (see ``_shifted_calendar``)."""
    period = _calendar_period(rbd)
    tests = []
    for position, member in enumerate(group.members):
        schedule = rbd._inspection[member]
        count = int(round(period / schedule.interval))
        phase = _shifted_calendar(rbd, member, (states or {}).get(member))
        if phase is not None:
            # The state's calendar: its tests' coverage is 1 (a state of
            # a member whose tests can miss is refused, see _check_state).
            for k in range(1, count + 1):
                time = min(k * schedule.interval - phase, period)
                tests.append(_ccf_chain.ProofTest(time, position, True))
            continue
        first = 0 if schedule.offset else 1
        for k in range(first, first + count):
            # The last, at the period's end, rounded no further.
            time = min(schedule.offset + k * schedule.interval, period)
            tests.append(
                _ccf_chain.ProofTest(time, position, schedule.is_full(time))
            )
    return tests, period


def _known_up(rbd, node, start) -> float:
    """How long before 0 a common-cause group's member started from
    ``start`` was last known to be up (#106): for one whose failures are
    hidden and that is up, when it was last tested and found working, or
    put into service since (``min(age, since_scheduled)``), as the
    simulation draws its life from then; 0 for any other (up now, as its
    failures are revealed, or down)."""
    if (
        start is None
        or start.new
        or start.stationary
        or not start.alive
        or node not in rbd._inspection
    ):
        return 0.0
    return float(min(start.age, start.since_scheduled or 0.0))


def _group_start(rbd, group, states: Optional[dict]):
    """Where a common-cause group's members start, for its chain over
    time (a ``_ccf_chain.Start``, see ``_require_group_states``): None
    for every member up at 0."""
    states = states or {}
    starts = [states.get(member) for member in group.members]
    if any(start is not None and start.stationary for start in starts):
        return _ccf_chain.Start(stationary=True)
    known = tuple(
        _known_up(rbd, member, start)
        for member, start in zip(group.members, starts)
    )
    down: dict = {}
    timings = _ccf_timings(rbd, group)
    for position, (member, start) in enumerate(zip(group.members, starts)):
        if start is None or start.alive:
            continue
        if member not in rbd._inspection:
            down[position] = True  # in a repair at a rate (revealed)
            continue
        assert timings is not None  # see _require_group_states
        down[position] = _ccf_chain.Left(
            timings[position].repair, float(start.down_for)
        )
    if not down and not any(known):
        return None
    return _ccf_chain.Start(known, down)


def _require_group_states(rbd, states: dict) -> None:
    """Raise unless the common-cause groups' chains can start from the
    members' states (#106): every member of a group in its long-run
    state or none (its members' joint long run has no part of its own);
    no member with a life of its own (its group's rate is its life); and
    a member down only in a repair that takes time (where its failures
    are hidden: one found is repaired at once otherwise)."""
    for group in rbd.ccf_groups:
        where = f"Common-cause group {list(group.members)}"
        starts = {m: states.get(m) for m in group.members}
        stationary = [
            m for m, start in starts.items() if start and start.stationary
        ]
        if stationary and len(stationary) < len(group.members):
            raise NotImplementedError(
                f"{where}: members {stationary} start in their long-run "
                "state and the others do not; the group's members' long "
                "run is a joint one: give every member a stationary "
                "state (state='stationary'), or none."
            )
        timings = None
        for member, start in starts.items():
            if start is None:
                continue
            if start.own_life:
                raise NotImplementedError(
                    f"{where}: member {member!r} has a life of its own "
                    "(a measured level or its own model), and a member's "
                    "life is its group's exponential one, split between "
                    "its causes: leave it out."
                )
            if start.alive or member not in rbd._inspection:
                continue
            if timings is None:
                timings = _ccf_timings(rbd, group) or [
                    _ccf_chain.Timing() for _ in group.members
                ]
            timing = timings[group.members.index(member)]
            if not timing.repair.takes_time:
                raise NotImplementedError(
                    f"{where}: member {member!r} is down, and its "
                    "failures are hidden and repaired at once when a test "
                    "finds them, so it cannot be: give the time since its "
                    "last test as its since_scheduled (it was found "
                    "working), and its age."
                )


def _groups_over_time(rbd, states: Optional[dict] = None) -> list:
    """Each common-cause group's members' joint states over time from
    every member up at 0, or from their states in ``states`` (#106, see
    ``_ccf_chain.OverTime``)."""
    out = []
    for group in rbd.ccf_groups:
        life, repair = _ccf_rates(rbd, group)
        start = _group_start(rbd, group, states)
        if repair is not None:
            where = f"Common-cause group {list(group.members)}'s"
            out.append(
                _ccf_chain.OverTime.revealed(
                    group.model,
                    group.members,
                    life,
                    repair,
                    partial(_curves._uniformized, where),
                    start,
                )
            )
            continue
        tests, period = _group_tests(rbd, group, states)
        out.append(
            _ccf_chain.OverTime.hidden(
                group.model,
                group.members,
                life,
                rbd._inspection[group.members[0]].coverage,
                tests,
                period,
                _ccf_timings(rbd, group),
                start,
            )
        )
    return out


def _require_groups_over_time(rbd, states: dict) -> None:
    """Raise unless the common-cause groups can be followed over time
    (see ``_groups_curve``): their chains cover them, from new or from
    the members' states (see ``_require_group_states``)."""
    _require_ccf_long_run(rbd)
    _require_group_states(rbd, states)


def _groups_system(rbd, working, broken, method: str, states=None):
    """The system with its common-cause groups at given times (see
    ``_ccf_chain.GroupsSystem``), the groups from their members' states
    in ``states`` (#106)."""
    causes = [
        _ccf_chain.causes(
            group.model, group.members, _ccf_rates(rbd, group)[0]
        )
        for group in rbd.ccf_groups
    ]
    return _ccf_chain.GroupsSystem(
        rbd,
        _groups_over_time(rbd, states),
        causes,
        working,
        broken,
        structure_method(method),
    )


def _groups_curve(
    rbd,
    horizon: float,
    working_nodes=frozenset(),
    broken_nodes=frozenset(),
    method: str = "p",
    states: Optional[dict] = None,
    counts: bool = False,
    stages: bool = False,
) -> "_ccf_chain.GroupsCurve":
    """The system over time with common-cause groups (#158, see
    ``_ccf_chain.GroupsCurve``): its nodes outside the groups as their
    own curves (each from its state in ``states``) to ``horizon``,
    counting their events with ``counts`` and following their stages
    with ``stages``, and the groups' members' joint states from their
    chains, from every member up at 0 or from their states (#106)."""
    states = states or {}
    working, broken = set(working_nodes), set(broken_nodes)
    _require_free_members(rbd, working, broken)
    _require_groups_over_time(rbd, states)
    members = {m for group in rbd.ccf_groups for m in group.members}
    curves = _curves._availability_curves(
        rbd,
        horizon,
        working | broken | members,
        counts=counts,
        stages=stages,
        state=states,
        groups=True,
    )
    system = _groups_system(rbd, working, broken, method, states)
    return _ccf_chain.GroupsCurve(rbd, curves, system)


def _require_free_members(rbd, working_nodes, broken_nodes) -> None:
    """Raise if a common-cause group's member is held working or
    broken: the cause it shares would still strike the others."""
    held = set(working_nodes or ()) | set(broken_nodes or ())
    for group in rbd.ccf_groups:
        caught = held & set(group.members)
        if caught:
            raise NotImplementedError(
                f"Node(s) {sorted(caught, key=str)} are in a "
                "common-cause group, whose shared causes would still "
                "strike the others: they cannot be held working or "
                "broken."
            )


def _with_ccf_groups(
    rbd,
    times: np.ndarray,
    probabilities: dict,
    failures: Optional[dict],
    weights: np.ndarray,
    states_by_group: Optional[list] = None,
    what: str = "This analysis",
) -> Tuple[dict, Optional[dict], np.ndarray, np.ndarray]:
    """The long-run points (the times of ``_long_run_grid``, and their
    weights) split by the common-cause groups' joint states: each
    time into one point for each combination of each group's members
    up or down, weighted by its probability then, with those members
    up or down for certain and the other nodes as at the time. The
    long-run values are averages over these points as over the times.
    Also each point's time's position in ``times``. ``states_by_group``
    gives the groups' states (see ``_group_states``), in their order,
    when they are not the groups' own (an allocation's design).

    Every combination of every group's states is a point of its own,
    so the points multiply with the groups: ``what`` (the analysis that
    splits them) refuses, before anything is built, where they would
    take more than ``_ccf_chain.SPLIT_VALUES`` values (#218). The
    long-run values, the importance measures and the values over time
    condition on each group within its module instead (see
    ``_ccf_tabled``)."""
    tables = [
        (
            _group_states(rbd, group, times)
            if states_by_group is None
            else states_by_group[number]
        )
        for number, group in enumerate(rbd.ccf_groups)
    ]
    _ccf_chain.check_split(
        len(times),
        [len(states.down) for states in tables],
        len(probabilities) * (1 if failures is None else 2),
        what,
        rbd.ccf_groups,
    )
    index = np.arange(len(times))
    for states in tables:
        combinations = len(states.down)
        mass = (weights[:, None] * states.probabilities[index, :]).ravel()
        keep = mass > 0.0
        weights = mass[keep]

        def spread(values, combinations=combinations, keep=keep):
            points = len(keep) // combinations
            values = np.broadcast_to(
                np.asarray(values, dtype=float), (points,)
            )
            return np.repeat(values, combinations)[keep]

        probabilities = {n: spread(v) for n, v in probabilities.items()}
        if failures is not None:
            failures = {n: spread(v) for n, v in failures.items()}
        for k, member in enumerate(states.members):
            down = np.tile(states.down[:, k], len(index))[keep]
            probabilities[member] = np.where(down, 0.0, 1.0)
            if failures is not None:
                failures[member] = np.where(down, 1.0, 0.0)
        index = np.repeat(index, combinations)[keep]
    return probabilities, failures, weights, index


def _ccf_tabled(
    rbd, p: dict, q: dict, tables: list, structure=None
) -> "_ccf_modules.Tabled":
    """The system with its common-cause groups at given points (#218):
    each node outside the groups up with the probability ``p[node]``
    and down with ``q[node]`` (1-d arrays of one length), and each
    group's members in each combination of their states with the
    probabilities ``tables`` give (each group's
    ``_ccf_chain.GroupStates`` at the points). Given a group's
    combination its members are independent of everything else, so
    each group is conditioned on only within the smallest module
    holding its members (see ``_ccf_modules.Tabled``): groups in
    separate modules cost a sum, not a product. Over ``structure``
    (the dual, for Fussell-Vesely over path sets) if given, else the
    decomposition."""
    structure = rbd._decomposition() if structure is None else structure
    if isinstance(structure, GraphStructure):
        raise NotImplementedError(structure.reason)
    p, q, size = rbd._node_pairs(p, q)
    return _ccf_modules.Tabled(
        rbd._ccf_plan(structure), rbd.ccf_groups, p, q, size, tables
    )


def _ccf_long_run(rbd, working_nodes, broken_nodes) -> tuple:
    """The long-run grid's times and weights (see ``_long_run_grid``),
    and what ``_ccf_tabled`` takes there: the nodes' availabilities
    and unavailabilities at the times (the forced nodes held at 1 or
    0), and the groups' members' joint states then, from their chains
    (see ``_group_states``)."""
    # The groups' chains first, as the routes report them.
    _require_ccf_long_run(rbd)
    _require_free_members(rbd, working_nodes, broken_nodes)
    times, weights = _long_run._long_run_grid(rbd)
    p = rbd._probabilities_with_overrides(
        _long_run._availabilities_at(rbd, times), working_nodes, broken_nodes
    )
    q = rbd._failures_with_overrides(
        _long_run._unavailabilities_at(rbd, times), working_nodes, broken_nodes
    )
    # The groups' chains, kept for the other long-run values (#229):
    # a choice of intervals asks for the cost rate and the
    # availability of each plan.
    tables = rbd._cache.ccf_tables
    if tables is None:
        tables = rbd._cache.ccf_tables = [
            _group_states(rbd, group, times) for group in rbd.ccf_groups
        ]
    return times, weights, (p, q, tables)


def _ccf_long_run_measure(
    rbd,
    measure: str,
    working_nodes,
    broken_nodes,
    kind: str = "failure",
    fv_type: str = "c",
    method: str = "exact",
) -> dict:
    """An importance measure of every node in the long run, with the
    common-cause groups (see ``_ccf_measure``)."""
    working, broken = rbd._held_nodes(working_nodes, broken_nodes)
    _, weights, inputs = _ccf_long_run(rbd, working, broken)
    return _squeeze_values(
        rbd._ccf_measure(
            measure, inputs, weights, kind, fv_type, method, working | broken
        )
    )


def _ccf_fv_shares(
    rbd, evaluation, inputs: tuple, fv_type: str, method: str
) -> dict:
    """The numerators of the Fussell-Vesely importances (see
    ``_fv_numerators``) with the common-cause groups, at each point:
    exactly, each group conditioned on within the smallest module
    holding its members (see ``_ccf_modules.Evaluation
    .failed_cut_sets``); as rare events, each set's probability of
    having failed summed over its groups' combinations (see
    ``_ccf_modules.expected_product``)."""
    size = evaluation.shape
    zero = np.zeros(size)
    if method == "exact":
        structure = rbd._set_structure()
        if structure.always_works:
            failed: dict = {}
        else:
            if fv_type == "p":
                p, q, tables = inputs
                evaluation = _ccf_tabled(
                    rbd, p, q, tables, structure=structure.dual()
                )
            failed = evaluation.failed_cut_sets()
        return {
            node: np.broadcast_to(
                np.asarray(failed.get(node, zero), dtype=float), (size,)
            )
            for node in rbd.nodes
        }
    outcomes = evaluation.outcomes_of_tables()
    if fv_type == "c":
        node_sets = rbd.get_min_cut_sets()
    else:
        node_sets = {
            frozenset(path_set)
            for path_set in rbd.get_min_path_sets(include_in_out_nodes=False)
        }
    out = {node: np.zeros(size) for node in rbd.nodes}
    for node_set in node_sets:
        value = np.broadcast_to(
            np.asarray(
                _ccf_modules.expected_product(
                    node_set, evaluation.q, outcomes
                ),
                dtype=float,
            ),
            (size,),
        )
        for node in node_set:
            out[node] = out[node] + value
    return out


def _require_ccf_frequencies(rbd) -> None:
    """Raise if the failure frequency with common-cause groups is not
    worked out: where their long-run values are not."""
    if rbd.ccf_groups:
        _require_ccf_long_run(rbd)


def _ccf_calendar_outages(rbd, working_nodes, broken_nodes) -> float:
    """``_long_run._calendar_outages`` with common-cause groups (#293):
    the groups' members' joint states just before and just after each
    instant from their chains (a member's test that takes time, starting
    then, is one of their jumps), the other nodes' as without groups."""
    changes = _long_run._calendar_changes(rbd, working_nodes, broken_nodes)
    if changes is None:
        return 0.0
    period, instants, before, after = changes
    # Just after each instant, and just before it (one before 0 is at the
    # period's end).
    shift = 1e-9 * period
    down = []
    for (p, q), times in (
        (after, instants + shift),
        (before, (instants - shift) % period),
    ):
        tables = [_group_states(rbd, group, times) for group in rbd.ccf_groups]
        down.append(_ccf_tabled(rbd, p, q, tables).system()[1])
    return float(np.sum(down[0] - down[1])) / period


def _ccf_outage_terms(
    rbd, working_nodes, broken_nodes
) -> Tuple[List[Tuple[Any, float]], float]:
    """``_outage_terms`` with common-cause groups, over the long-run
    grid's times (see ``_ccf_long_run``): a node outside the groups
    fails at its rate and takes the system down where it is critical
    (its Birnbaum importance then, over the groups' states); each cause
    strikes at its rate and takes the system down by failing the
    members it names that are up (the rise in the system's
    unavailability with them down)."""
    _require_ccf_frequencies(rbd)
    _require_free_members(rbd, working_nodes, broken_nodes)
    times, weights, inputs = _ccf_long_run(rbd, working_nodes, broken_nodes)
    availability = inputs[0]
    evaluation = _ccf_tabled(rbd, *inputs)
    R_t, Q_t = evaluation.system()
    forced = set(working_nodes or ()) | set(broken_nodes or ())
    members = {m for group in rbd.ccf_groups for m in group.members}
    terms: List[Tuple[Any, float]] = []
    planned = 0.0
    blocks = set(_long_run._block_nodes(rbd))
    for node in rbd.components:
        if node in forced or node in members:
            continue
        up_1, down_1 = evaluation.system(hold={node: True})
        up_0, down_0 = evaluation.system(hold={node: False})
        importance = np.where(Q_t <= R_t, down_0 - down_1, up_1 - up_0)
        if node in rbd._inspection:
            node_failures: Any = _requirements._tested_intensity(
                rbd, node, times, availability[node]
            )
            node_planned: Any = 0.0
        elif node in blocks:
            node_failures = _long_run._block_profile(
                rbd, node, times, rates=True
            )
            node_planned = 0.0
        else:
            node_failures, _, node_planned = _long_run._node_frequencies(
                rbd, node
            )
        terms.append((node, float(weights @ (importance * node_failures))))
        planned += float(weights @ (importance * node_planned))
    for group in rbd.ccf_groups:
        life, _ = _ccf_rates(rbd, group)
        for struck, rate in _ccf_chain.causes(
            group.model, group.members, life
        ):
            hit = {group.members[position]: False for position in struck}
            rise = evaluation.system(hold=hit)[1] - Q_t
            cause = (
                group.members[struck[0]]
                if len(struck) == 1
                else tuple(group.members)
            )
            terms.append((cause, rate * float(weights @ rise)))
    planned += _ccf_calendar_outages(rbd, working_nodes, broken_nodes)
    return terms, planned
