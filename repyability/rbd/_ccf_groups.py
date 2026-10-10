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
            rbd._require_tested_exact(member)
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
    ``_ccf_chain.Duration`` of no time, a fixed time or an exponential
    one; None where none takes time. Raise for what the chain does not
    take, saying what to do: another distribution, a test of an
    exponential length followed by a repair of a fixed one (whose end
    is then no fixed time after the test), and fixed lengths that run
    into the member's next test."""
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
            if rate is None:
                raise NotImplementedError(
                    f"{where}: its chain takes tests and repairs that "
                    "take no time, a fixed time or an exponential one, "
                    f"and the {what} of member {member!r} take none of "
                    f"those. Estimate the values by simulation, with "
                    "availability() or cost(), which take any; or, where "
                    f"the {what} are short next to the test interval, "
                    "give them a fixed length."
                )
            parts.append(_ccf_chain.Duration(rate=rate))
        test, repair = parts
        if test.rate is not None and repair.fixed > 0.0:
            raise NotImplementedError(
                f"{where}: the tests of member {member!r} take an "
                "exponential time, and its repairs a fixed one, which "
                "its chain does not take (the repair would end no fixed "
                "time after the test). Give both a fixed length, or both "
                "an exponential one; or estimate the values by "
                "simulation, with availability() or cost()."
            )
        length = test.fixed + repair.fixed
        if length >= inspection.interval:
            raise NotImplementedError(
                f"{where}: the test and repair of member {member!r} take "
                f"{length:g} together, as long as its test interval "
                f"({inspection.interval:g}) or longer, which its chain "
                "does not take. Estimate the values by simulation, with "
                "availability() or cost()."
            )
        out.append(_ccf_chain.Timing(test, repair))
    if not any(timing.takes_time for timing in out):
        return None
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
    imperfectly, in a maintenance group, or started from a state."""
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
            if start is not None and not start.new:
                raise NotImplementedError(
                    f"{where}: member {member!r} starts from a state, "
                    "which the simulation of its group does not take in, "
                    "as yet: its members start new."
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


def _group_tests(rbd, group) -> Tuple[list, float]:
    """A common-cause group's members' tests in one period of the
    calendar (``(0, period]``, see ``_calendar_period``), and the
    period."""
    period = _calendar_period(rbd)
    tests = []
    for position, member in enumerate(group.members):
        schedule = rbd._inspection[member]
        count = int(round(period / schedule.interval))
        first = 0 if schedule.offset else 1
        for k in range(first, first + count):
            time = schedule.offset + k * schedule.interval
            tests.append(
                _ccf_chain.ProofTest(time, position, schedule.is_full(time))
            )
    return tests, period


def _groups_over_time(rbd) -> list:
    """Each common-cause group's members' joint states over time from
    every member up at 0 (see ``_ccf_chain.OverTime``)."""
    out = []
    for group in rbd.ccf_groups:
        life, repair = _ccf_rates(rbd, group)
        if repair is not None:
            where = f"Common-cause group {list(group.members)}'s"
            out.append(
                _ccf_chain.OverTime.revealed(
                    group.model,
                    group.members,
                    life,
                    repair,
                    partial(_curves._uniformized, where),
                )
            )
            continue
        tests, period = _group_tests(rbd, group)
        out.append(
            _ccf_chain.OverTime.hidden(
                group.model,
                group.members,
                life,
                rbd._inspection[group.members[0]].coverage,
                tests,
                period,
                _ccf_timings(rbd, group),
            )
        )
    return out


def _require_groups_over_time(rbd, states: dict) -> None:
    """Raise unless the common-cause groups can be followed over time
    from new (see ``_groups_curve``): their chains cover them, and no
    member starts from a state."""
    _require_ccf_long_run(rbd)
    started = sorted(
        (m for g in rbd.ccf_groups for m in g.members if m in states),
        key=str,
    )
    if started:
        raise NotImplementedError(
            f"Node(s) {started} are in a common-cause group: its "
            "values over time are worked out from every member new at "
            "0, and a member's current state is not taken in, as yet."
        )


def _groups_system(rbd, working, broken, method: str):
    """The system with its common-cause groups at given times (see
    ``_ccf_chain.GroupsSystem``)."""
    causes = [
        _ccf_chain.causes(
            group.model, group.members, _ccf_rates(rbd, group)[0]
        )
        for group in rbd.ccf_groups
    ]
    return _ccf_chain.GroupsSystem(
        rbd,
        _groups_over_time(rbd),
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
    chains, from every member up at 0."""
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
    system = _groups_system(rbd, working, broken, method)
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
    tables = rbd.__dict__.get("_ccf_tables")
    if tables is None:
        tables = rbd.__dict__["_ccf_tables"] = [
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
    working = set(working_nodes or ())
    broken = set(broken_nodes or ())
    rbd._validate_node_overrides(working, broken)
    _, weights, inputs = _ccf_long_run(rbd, working, broken)
    return _squeeze_values(
        rbd._ccf_measure(measure, inputs, weights, kind, fv_type, method)
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
    worked out: with block replacements that take time (planned
    outages at block times, which the groups' states would change)."""
    if not rbd.ccf_groups:
        return
    _require_ccf_long_run(rbd)
    timed = [
        node
        for node in _long_run._block_nodes(rbd)
        if rbd._preventive[node].duration is not None
    ]
    if timed:
        raise NotImplementedError(
            "The system's planned outages at block replacements that "
            f"take time (of {sorted(timed, key=str)}) are not worked "
            "out with common-cause groups, as yet."
        )
    tested = [
        node
        for node in rbd._inspection
        if rbd._inspection[node].duration is not None
    ]
    if tested:
        raise NotImplementedError(
            "The system's planned outages at tests that take time (of "
            f"{sorted(tested, key=str)}) are not worked out with "
            "common-cause groups, as yet."
        )


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
            node_failures: Any = rbd._tested_intensity(
                node, times, availability[node]
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
    return terms, planned
