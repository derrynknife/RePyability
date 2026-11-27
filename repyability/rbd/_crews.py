"""A ``RepairableRBD``'s repair crews in the exact methods: whether the
crews limit or couple the components, and the crews' Markov chain
(``_crew_chain``, which copies the simulation's queue) with its rates,
state probabilities and outage terms in the long run.
``RepairableRBD``'s methods call these.
"""

import math
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

from repyability.rbd import (
    _ccf_chain,
)
from repyability.rbd import _crew_chain as crew_chain
from repyability.rbd import _long_run, _requirements
from repyability.rbd._common import (
    _constant_rate,
    _repair_rate,
)
from repyability.rbd._events import (
    _Unit,
)
from repyability.rbd.ccf import Shocks
from repyability.rbd.results import CrewJob, CrewStates
from repyability.rbd.routes import Refused


def _crew_served(rbd) -> list:
    """The jobs the repair crews work on, by key: this RBD's own
    components, not a nested RBD's (which has crews of its own), and
    each unit of a standby group (``_Unit(node, unit)``)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    served: list = []
    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            continue
        arrangement = rbd._standby.get(node)
        if arrangement is None:
            served.append(node)
        else:
            served.extend(
                _Unit(node, unit) for unit in range(arrangement.units)
            )
    return served


def _crews_limited(rbd) -> bool:
    """Whether there are fewer repair crews than components that may
    need one, so that a job may wait."""
    return rbd.repair_crews is not None and rbd.repair_crews < len(
        _crew_served(rbd)
    )


def _call_when(rbd) -> int:
    """How many waiting jobs call the call-out crew (``call_out``'s
    ``"when_waiting"``); 0 with none, or where no job can wait."""
    call_out = getattr(rbd, "_call_out", None)
    if call_out is None or not _crews_limited(rbd):
        return 0
    return call_out.when_waiting


def _call_out_priced(rbd) -> bool:
    """Whether a call-out crew that can be called costs anything."""
    call_out = getattr(rbd, "_call_out", None)
    return bool(
        call_out is not None
        and (call_out.fee or call_out.rate)
        and _crews_limited(rbd)
    )


def _call_out_charge(rbd, calls, engaged):
    """What the call-out crew costs for ``calls`` call-outs and a time
    ``engaged``: its fee each time it comes, and its rate while engaged."""
    return rbd._call_out.fee * calls + rbd._call_out.rate * engaged


def _call_out_vector(rbd, chain) -> np.ndarray:
    """The rate at which the call-out crew costs in each state of the
    crews' ``chain``: its fee at the rate it is called there, and its
    rate where it is engaged."""
    return _call_out_charge(rbd, chain.calls, chain.engaged.astype(float))


def _call_out_cost_rate(rbd, working_nodes, broken_nodes) -> float:
    """The call-out crew's long-run cost per unit time, from the crews'
    chain without the nodes held (0.0 where it costs nothing)."""
    if not _call_out_priced(rbd):
        return 0.0
    forced = frozenset(working_nodes or ()) | frozenset(broken_nodes or ())
    chain = _crew_chain(rbd, forced)
    return float(chain.probabilities @ _call_out_vector(rbd, chain))


def _require_unlimited_crews(
    rbd,
    assumes: str = "these values assume",
    advice: str = "Simulate the system with availability() or cost().",
) -> None:
    """Raise if a job may wait for a repair crew: components then no
    longer fail and recover independently, which ``assumes`` (what
    assumes it, and ``assume``): do ``advice`` instead."""
    if rbd._crews_couple():
        raise NotImplementedError(
            f"With {rbd.repair_crews} repair crew(s) for "
            f"{len(_crew_served(rbd))} components, a component can "
            "wait for a crew, so the components no longer fail and "
            f"recover independently, which {assumes}. {advice}"
        )


def _no_crew_chain(rbd, why: str) -> NoReturn:
    """Raise that the repair crews' Markov chain does not cover this RBD,
    because ``why``."""
    raise NotImplementedError(
        f"With {rbd.repair_crews} repair crew(s) for "
        f"{len(_crew_served(rbd))} components, a component can wait "
        "for a crew, and the exact long-run values come from a Markov "
        "chain of the components' states and the repair queue: "
        f"{why}. Simulate the system with availability() or cost()."
    )


def _crew_chain_rates(rbd) -> Dict[Any, Tuple[float, float]]:
    """The failure and repair rates (``inf`` for an instant repair) of
    the components the repair crews work on, for their Markov chain
    (see ``_crew_chain.py``). Raise if the chain does not cover them:
    scheduled maintenance or inspection, a life or repair time that is
    not exponential, or more states than it is solved for."""
    assert rbd.repair_crews is not None  # limited crews only
    for node in rbd._standby:
        _no_crew_chain(
            rbd,
            "it has no place for a standby group, which component "
            f"{node!r} is",
        )
    rates: Dict[Any, Tuple[float, float]] = {}
    for node in _crew_served(rbd):
        component = rbd.components[node]
        if node in rbd._preventive:
            _no_crew_chain(
                rbd,
                "it has no place for scheduled maintenance, which "
                f"component {node!r} has",
            )
        if node in rbd._inspection:
            _no_crew_chain(
                rbd,
                "it has no place for inspections, which component "
                f"{node!r} has",
            )
        if node in rbd._imperfect:
            _no_crew_chain(
                rbd,
                "it has no place for imperfect repair, which component "
                f"{node!r} has",
            )
        life = _constant_rate(component.reliability)
        if life is None:
            _no_crew_chain(
                rbd,
                "it needs exponential lives (a constant failure rate), "
                f"and the life of component {node!r} is not one",
            )
        repair = _repair_rate(component.time_to_replace)
        if repair is None:
            _no_crew_chain(
                rbd,
                "it needs exponential repair times (or instant repair), "
                f"and the repair times of component {node!r} are not",
            )
        rates[node] = (life, repair)
    priorities = tuple(rbd._priority.get(node, 0.0) for node in rates)
    instant = tuple(math.isinf(repair) for _, repair in rates.values())
    call_when = _call_when(rbd)
    # (Kept beside the chains, under a key of another shape.)
    counts = rbd._cache.kept("crew_chains")
    key = (priorities, instant, rbd.repair_crews, call_when)
    if key not in counts:
        counts[key] = crew_chain.state_count(*key)
    if counts[key] > crew_chain.MAX_STATES:
        _no_crew_chain(
            rbd,
            (
                "here it has more than the "
                if call_when
                else f"here it has {counts[key]:,} states, more than the "
            )
            + f"{crew_chain.MAX_STATES:,} "
            + ("states " if call_when else "")
            + "it is solved for",
        )
    return rates


def _crew_causes(
    rbd, nodes, rates: Optional[dict] = None
) -> Tuple[Dict[Any, float], List[tuple]]:
    """The causes of failure of the crews' chain's ``nodes`` with
    common-cause groups (#251), as the simulation strikes them (see
    ``_ccf_groups._shared_causes``): each node's own cause's rate (its
    failure rate, outside the groups), and the shared causes, each ``(the
    places in nodes of the members it strikes, in the diagram's order,
    which is the order the simulation's queue takes failures at one
    instant in (see ``_events._EventQueue``); its rate; its group's
    place)``. A member not among ``nodes`` (held working, it never fails,
    and held broken it is never repaired) is struck no more, and the
    causes it shares still strike the others. Raise if the chain does not
    take a group in: a member repaired instantly. ``rates`` are the
    members' failure and repair rates, by default from the crews' chain
    (``_crew_chain_rates``); the first failure's chain gives its own."""
    if rates is None:
        rates = _crew_chain_rates(rbd)
    own = {node: rates[node][0] for node in nodes if node in rates}
    place = {node: k for k, node in enumerate(nodes)}
    order = {node: k for k, node in enumerate(rbd.components)}
    shared: List[tuple] = []
    for g, group in enumerate(rbd.ccf_groups):
        members = list(group.members)
        where = f"common-cause group {members}"
        for member in members:
            if math.isinf(rates[member][1]):
                _no_crew_chain(
                    rbd,
                    f"it takes no member of a common-cause group repaired "
                    f"instantly, as member {member!r} of {where} is",
                )
        # A Shocks group's shocks add to each member's own rate; the
        # other models split the members' one.
        lives = (
            tuple(rates[member][0] for member in members)
            if isinstance(group.model, Shocks)
            else rates[members[0]][0]
        )
        for struck, rate in _ccf_chain.causes(group.model, members, lives):
            if len(struck) == 1:
                if members[struck[0]] in place:
                    own[members[struck[0]]] = rate
                continue
            left = sorted(
                (members[i] for i in struck if members[i] in place),
                key=order.__getitem__,
            )
            if len(left) == 1:
                own[left[0]] += rate
            elif left:
                shared.append((tuple(place[m] for m in left), rate, g))
    return own, shared


def _require_crew_chain(rbd) -> None:
    """Raise if the exact long-run values with limited repair crews
    cannot be computed: the Markov chain does not cover the components
    (see ``_crew_chain_rates``) or their common-cause groups (#251, see
    ``_crew_causes``), or nested RBDs' calendars fall together (see
    ``_require_calendars``), or a component draws on a shelf of spares
    (see ``_require_no_shelves``)."""
    _requirements._require_no_shelves(rbd)
    rates = _crew_chain_rates(rbd)
    if rbd.ccf_groups:
        _crew_causes(rbd, list(rates))
    _requirements._require_calendars(rbd)


def _crew_chain(
    rbd, forced: frozenset = frozenset()
) -> "crew_chain.CrewChain":
    """The Markov chain of the components the repair crews work on, and
    its long-run distribution (see ``_crew_chain.py``), leaving out the
    nodes in ``forced``: held working a node never fails, and held
    broken it is never repaired, so neither needs a crew. Solved once
    for each set of rates and kept."""
    _require_crew_chain(rbd)
    rates = _crew_chain_rates(rbd)
    assert rbd.repair_crews is not None  # limited crews only
    nodes = [node for node in rates if node not in forced]
    priorities = [rbd._priority.get(node, 0.0) for node in nodes]
    call_when = _call_when(rbd)
    own = {node: rates[node][0] for node in nodes}
    shared: List[tuple] = []
    if rbd.ccf_groups:
        own, shared = _crew_causes(rbd, nodes)
    key = (
        rbd.repair_crews,
        call_when,
        tuple(
            zip(
                nodes,
                ((own[node], rates[node][1]) for node in nodes),
                priorities,
            )
        ),
        tuple(shared),
    )
    cache = rbd._cache.kept("crew_chains")
    if key not in cache:
        cache[key] = crew_chain.solve(
            nodes,
            [own[node] for node in nodes],
            [rates[node][1] for node in nodes],
            priorities,
            rbd.repair_crews,
            call_when,
            shared,
        )
    return cache[key]


def _chain_probabilities(
    rbd, working_nodes, broken_nodes
) -> Tuple[dict, np.ndarray]:
    """With limited repair crews, every node's availability in each
    state of their Markov chain (1 or 0 for the components the crews
    work on and those held working or broken, and a nested RBD's own
    long-run availability, as it has crews of its own), and the states'
    long-run probabilities: the long-run values are then averages over
    the states, as over the times of ``_long_run_grid``."""
    working_nodes = set(working_nodes or ())
    broken_nodes = set(broken_nodes or ())
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    chain = _crew_chain(rbd, frozenset(working_nodes | broken_nodes))
    size = len(chain.probabilities)
    out: dict = {
        node: chain.up[:, k].astype(float)
        for k, node in enumerate(chain.nodes)
    }
    for node, component in rbd.components.items():
        if node in working_nodes:
            out[node] = np.ones(size)
        elif node in broken_nodes:
            out[node] = np.zeros(size)
        elif node not in out:
            out[node] = np.full(size, float(component.mean_availability()))
    for node in rbd.in_or_out:
        out[node] = np.ones(size)
    return out, chain.probabilities


def crew_jobs(rbd) -> List[CrewJob]:
    """See ``RepairableRBD.crew_jobs``."""
    return [
        (
            CrewJob(key.node, key.unit)
            if isinstance(key, _Unit)
            else CrewJob(key, None)
        )
        for key in _crew_served(rbd)
    ]


def _require_coupling_crews(rbd) -> None:
    """Raise (``routes.Refused``) if no job can wait for a repair crew
    behind another component's, so that there is no crews' chain: the
    crews are unlimited, or as many as their jobs, or serve one standby
    group's units alone (whose own chain counts them)."""
    if not rbd._crews_couple():
        raise Refused(
            "No job here can wait for a repair crew behind another "
            "component's (the crews are unlimited, as many as their jobs, "
            "or serve one standby group's units alone), so the components "
            "fail and recover independently, with no crews' chain: "
            "node_availability gives each one's long-run availability."
        )


def crew_states(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
) -> CrewStates:
    """See ``RepairableRBD.crew_states``."""
    working, broken = rbd._held_nodes(working_nodes, broken_nodes)
    _require_coupling_crews(rbd)
    availability, probabilities = _chain_probabilities(rbd, working, broken)
    chain = _crew_chain(rbd, frozenset(working) | frozenset(broken))
    return CrewStates(
        availability,
        probabilities,
        np.array(chain.calls, dtype=float),
        chain.engaged.astype(float),
    )


def _chain_outage_terms(
    rbd, working_nodes, broken_nodes
) -> Tuple[List[Tuple[Any, float]], float]:
    """``_outage_terms`` with limited repair crews, over the
    states of their Markov chain: in each, a component that is up fails
    at its constant rate, and takes the system down if it is critical
    there. A nested RBD enters through its own frequencies, as it has
    crews of its own."""
    availability, unavailability, weights = (
        _long_run._long_run_unavailabilities(rbd, working_nodes, broken_nodes)
    )
    forced = set(working_nodes or ()) | set(broken_nodes or ())
    rates = _crew_chain_rates(rbd)
    chain = _crew_chain(rbd, frozenset(forced))
    own = _crew_causes(rbd, chain.nodes)[0] if rbd.ccf_groups else {}
    birnbaum = rbd._birnbaum_importance(
        availability, node_failures=unavailability
    )
    terms: List[Tuple[Any, float]] = []
    planned = 0.0
    for node in rbd.components:
        if node in forced:
            continue
        importance = np.asarray(birnbaum[node])
        if node in rates:
            # (Its own cause's rate, in a common-cause group.)
            life = own.get(node, rates[node][0])
            node_failures, node_planned = life * availability[node], 0.0
        else:
            node_failures, _, node_planned = _long_run._node_frequencies(
                rbd, node
            )
        terms.append((node, float(weights @ (importance * node_failures))))
        planned += float(weights @ (importance * node_planned))
    if rbd.ccf_groups:
        # Each group's shared causes: a strike takes the system down
        # where it is up, and down with the members struck down.
        from repyability.rbd import _curves

        struck = _curves._crew_strikes(rbd, chain, availability)
        for g, group in enumerate(rbd.ccf_groups):
            terms.append((tuple(group.members), float(weights @ struck[:, g])))
    return terms, planned
