"""Reliability and mean time to the first system failure of a
``RepairableRBD``.

The probability that the system runs to ``t`` without a failure, and the
mean time until the first one, differ from a non-repairable diagram's: a
component can fail and be repaired while its partner in a redundant pair
keeps the system up. Two exact routes cover what they can:

- **Series structures** (every component a cut set of one). No repair
  can happen before the first system failure, since the first failure of
  any component is one of the system. So ``R(t) = prod_i R_i(t)``, each
  from the component's state at 0, whatever its life distribution.
- **Exponential lives and repairs** (or instant repairs), with any
  number of repair crews. The components' joint states form a Markov
  chain (the crews' chain of ``_crew_chain``, which has one crew per
  component when the diagram has no limit). Making the states where the
  system is down absorbing, ``R(t)`` is the chain's probability of not
  having been absorbed by ``t``, followed by uniformization
  (``p(t) 1 = E[s_N]``, ``s_k = p(0) P^k 1`` and ``N`` Poisson with mean
  ``L t``, as in ``_chain_transient``), and the mean time to absorption is
  one linear solve over the states where the system is up.

A sub-system (a nested ``RepairableRBD``) with nothing coupling it to
the rest (no repair crews or call-out of its own or around it, no
common-cause, maintenance or shutdown groups in it) is the same, to its
first failure, as its components drawn into the diagram in its place,
each as ``(node, component)``: ``_flat`` draws it so for these two
methods, and the routes above take the drawn diagram.

Everything else (a life or repair that is not exponential in a redundant
structure, scheduled maintenance, tests, standby groups, common-cause
groups, a coupled sub-system) is simulated on request:
``method="simulate"`` reads each history's first failure off
``simulate_timelines``.

Wherever the expected events are worked out, the first failure is also
bracketed (``first_failure_bounds``). With ``F(t)`` the probability that
the system has failed by ``t`` (or was down at the start), ``Q(t)`` its
unavailability and ``W(t)`` its expected failures in ``[0, t)``: a system
down at ``s <= t`` has failed by ``t``, so ``F(t) >= Q(s)``; and the
first failure is one of the failures counted, so ``F(t) <= Q(0) + W(t)``.
The first holds only while every outage is a failure: a diagram that can
be down for planned work (maintenance, tests or shuts that take time) can
be down without having failed, and its upper bound is 1. The mean's lower
bound is the integral of ``max(0, 1 - Q(0) - W(t))``.
"""

import math
import warnings
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np

from repyability.rbd import _crew_chain as crew_chain
from repyability.rbd import (
    _crews,
    _curves,
    _overrides,
    _phase_type,
    _requirements,
    _unplanned,
)
from repyability.rbd._common import _constant_rate, _repair_rate
from repyability.rbd._mean_lifetime import (
    _integrate,
    mean_lifetime,
    model_knots,
)
from repyability.rbd.modular import NODE, SERIES, GraphStructure
from repyability.rbd.results import ConfidenceInterval, FirstFailureBounds
from repyability.utils.checks import (
    is_number,
    nonnegative_times,
    one_of,
    simulation_options,
)
from repyability.utils.vectors import dot

#: The most steps uniformization takes, and the most work (steps times the
#: chain's transitions) it does before the chain is given up on.
MAX_STEPS = 2_000_000
MAX_WORK = 1e10
#: The most states of a chain whose lives or repairs are followed as
#: phases.
MAX_PHASE_STATES = 400_000
#: The least probability of failing that a value's accuracy is sized by:
#: about what uniformization itself keeps.
_FLOOR = 1e-12
#: The share of the largest probability of failing asked for below which
#: one's accuracy is sized by that share, not by itself.
_SMALL = 1e-3
#: How the exact values are worked out (``_require_exact``).
AS_SERIES, AS_CHAIN, AS_PHASES = "series", "chain", "phases"
#: A step's survival probability below this has ended: the rest of the
#: chain's survival is within it.
_ENDED = 1e-18
_SPREAD, _MARGIN = 12.0, 20.0

_SIMULATE = (
    " Estimate it with method='simulate' (with t_simulation and "
    "mc_samples), or bracket it with first_failure_bounds(t) (and its mean "
    "with mean_time_to_first_failure_bound()), from the expected failures."
)


def _refuse(why: str) -> NoReturn:
    raise NotImplementedError(why + "." + _SIMULATE)


def _coupling(rbd) -> Optional[str]:
    """What ties a sub-system's components to each other beyond its
    structure, or None."""
    if _crews._crews_limited(rbd) or _crews._call_when(rbd):
        return "repair crews"
    if rbd.ccf_groups:
        return "common-cause groups"
    if rbd._maintenance:
        return "maintenance groups"
    if rbd._shutdowns:
        return "shutdowns"
    return None


def _flat(rbd, state=None):
    """The diagram with each sub-system drawn in its place (see the
    module's notes), its components named ``(node, component)``, and the
    state as given; the diagram itself if it has none. Raise if a
    sub-system is coupled, or is given a state."""
    from repyability.rbd.helper_classes import PerfectReliability
    from repyability.rbd.repairable_rbd import RepairableRBD

    nested = [
        node
        for node, component in rbd.components.items()
        if isinstance(component, RepairableRBD)
    ]
    if not nested:
        return rbd
    if _crews._crews_limited(rbd):
        # Drawn in, its components' repairs would wait for the crews.
        _refuse(
            "The exact values do not cover a nested diagram (component "
            f"{nested[0]!r}) beside repair crews that jobs wait for, "
            "which its components' repairs, drawn into the diagram, would "
            "wait for too"
        )
    if isinstance(state, dict) and any(node in state for node in nested):
        given = next(node for node in nested if node in state)
        _refuse(
            "The exact values do not cover a nested diagram given a state "
            f"(component {given!r})"
        )
    args = rbd._init_args
    components = {
        node: spec
        for node, spec in args["components"].items()
        if node not in nested
    }
    k = dict(args["k"] or {})
    ports: Dict[Any, Tuple[Any, Any]] = {}
    edges: list = []
    for node in nested:
        coupled = _coupling(rbd.components[node])
        if coupled:
            _refuse(
                "The exact values do not cover a nested diagram "
                f"(component {node!r}) with {coupled}, which tie its "
                "components together"
            )
        inner = _flat(rbd.components[node])
        given = inner._init_args
        ports[node] = ((node, inner.input_node), (node, inner.output_node))
        edges += [((node, u), (node, v)) for u, v in given["edges"]]
        for name in inner.G.nodes:
            components[(node, name)] = given["components"].get(
                name, PerfectReliability
            )
        k.update(
            {(node, name): value for name, value in (given["k"] or {}).items()}
        )
        if node in k:
            k[ports[node][0]] = k.pop(node)
    for u, v in args["edges"]:
        edges.append(
            (
                ports[u][1] if u in ports else u,
                ports[v][0] if v in ports else v,
            )
        )
    # No job waits for a crew (above, and in each sub-system): drawn in,
    # every component has one.
    return RepairableRBD(
        **dict(
            args,
            edges=edges,
            components=components,
            k=k or None,
            repair_crews=None,
            call_out=None,
        )
    )


def _components_are_plain(rbd) -> None:
    """Raise if the first failure's exact values do not cover the
    diagram's kinds of component or maintenance."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if rbd._maintenance:
        _refuse(
            "The exact values do not cover maintenance groups, which "
            "stop their members together"
        )
    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            _refuse(
                "The exact values do not cover a nested diagram "
                f"(component {node!r}), which has repairs of its own"
            )
        for table, what, why in (
            (
                rbd._preventive,
                "scheduled maintenance",
                " (a fixed interval is no time a Markov chain's phases "
                "follow)",
            ),
            (
                rbd._inspection,
                "inspections",
                " (tests at fixed intervals are no times a Markov chain's "
                "phases follow)",
            ),
            (rbd._imperfect, "imperfect repair", ""),
        ):
            if node in table:
                if table is rbd._standby and table[node].key == "markov":
                    what = "a Markov component's own chain"
                _refuse(
                    f"The exact values do not cover {what}, which "
                    f"component {node!r} has{why}"
                )
        _requirements._require_reliabilities(rbd, node)
    _groups_fit(rbd)


def _groups_fit(rbd) -> None:
    """Raise if the chain does not take the diagram's common-cause or
    standby groups in: they are chains of exponential units."""
    for node in rbd._standby:
        what = rbd._standby[node].what
        if rbd._crews_couple():
            _refuse(
                f"The exact values do not cover {what} (component "
                f"{node!r}) whose units wait for repair crews with other "
                "components' jobs"
            )
        from repyability.rbd import _long_run

        try:
            _long_run._standby_rates(rbd, node)
        except NotImplementedError:
            _refuse(
                f"The exact values cover {what} only of units with "
                "exponential lives and repair times, which those of "
                f"component {node!r} are not"
            )
    for group in rbd.ccf_groups:
        for member in group.members:
            component = rbd.components[member]
            if member in rbd._standby:
                _refuse(
                    "The exact values do not cover a standby group in a "
                    f"common-cause group, as component {member!r} is"
                )
            if _constant_rate(component.reliability) is None or not (
                0.0
                < (_repair_rate(component.time_to_replace) or 0.0)
                < math.inf
            ):
                _refuse(
                    "The exact values cover a common-cause group only of "
                    "members with exponential lives and repair times, "
                    f"which those of member {member!r} of group "
                    f"{list(group.members)} are not"
                )


def _member_rates(rbd) -> Dict[Any, tuple]:
    """The failure and repair rates of the common-cause groups' members."""
    return {
        member: (
            _constant_rate(rbd.components[member].reliability),
            _repair_rate(rbd.components[member].time_to_replace),
        )
        for group in rbd.ccf_groups
        for member in group.members
    }


def _causes(rbd, nodes: list) -> Tuple[Dict[Any, float], List[tuple]]:
    """The common-cause groups' causes over the chain's ``nodes`` (see
    ``_crews._crew_causes``): each member's own cause's rate, and the
    shared causes."""
    if not rbd.ccf_groups:
        return {}, []
    return _crews._crew_causes(rbd, nodes, _member_rates(rbd))


def _group(rbd, node) -> tuple:
    """A standby group's Markov chain (or a Markov component's own; see
    ``_long_run._group_chain``) and its states' probabilities at the
    start, as its curve over time takes them
    (``_curves._standby_curve``)."""
    from repyability.rbd import _long_run

    chain = _long_run._group_chain(rbd, node)
    if rbd._standby[node].key == "markov":
        return chain, np.asarray(rbd.components[node]._start, dtype=float)
    first = np.zeros(len(chain.states))
    first[0] = 1.0
    return chain, first


def _in_series(structure, term: int) -> bool:
    """Whether the term is a node or a series of terms like it."""
    kind = structure.terms[term][0]
    if kind == NODE:
        return True
    return kind == SERIES and all(
        _in_series(structure, child) for child in structure.terms[term][1]
    )


def _is_series(rbd) -> bool:
    """Whether every component is a cut set of one: the system works only
    while all its relevant components do."""
    structure = rbd._decomposition()
    if isinstance(structure, GraphStructure):
        return False
    if structure.always_works:
        return True
    if rbd.ccf_groups or rbd._standby:
        # A group's members fail together, and a standby group's units
        # are repaired before it fails.
        return False
    return structure.root is not None and _in_series(structure, structure.root)


def _rates(rbd) -> Optional[Dict[Any, Tuple[float, float]]]:
    """Each component's failure and repair rates (``inf`` for an instant
    repair) if every life and repair is exponential; None otherwise.
    (Not a standby group's, which has a chain of its own.)"""
    rates: Dict[Any, Tuple[float, float]] = {}
    for node, component in rbd.components.items():
        if node in rbd._standby:
            continue  # a chain of its own (``_groups_fit``)
        life = _constant_rate(component.reliability)
        repair = _repair_rate(component.time_to_replace)
        if life is None or repair is None:
            return None
        rates[node] = (life, repair)
    return rates


def _crew_count(rbd) -> int:
    return rbd.repair_crews or max(len(rbd.components), 1)


def _queued(rbd) -> bool:
    """Whether repairs can wait for a crew, which ties every component to
    the others."""
    return bool(rbd._crews_couple())


def _chain_fits(rbd, rates: dict) -> None:
    """Raise if the chain of exponential components is too large."""
    plain = list(rates)
    count = crew_chain.state_count(
        [rbd._priority.get(node, 0.0) for node in plain],
        [math.isinf(rates[node][1]) for node in plain],
        _crew_count(rbd),
        _crews._call_when(rbd),
    )
    for node in rbd._standby:
        count *= len(_group(rbd, node)[0].states)
    if count > crew_chain.MAX_STATES:
        # (With a call-out crew, the states are counted only that far.)
        _refuse(
            "Its Markov chain has more than the "
            f"{crew_chain.MAX_STATES:,} states it is solved for"
        )


def _modules(rbd) -> Tuple[list, list]:
    """``(alone, modules)``: the components each of which fails the
    system alone, and the other series factors of the structure, each the
    list of its components. With repairs that wait for a crew, one module
    of every component, as the queue ties them; a standby group is a
    module, never alone."""
    structure = rbd._decomposition()
    if _queued(rbd):
        return [], [list(rbd.components)]
    if structure.always_works:
        return [], []
    if structure.root is None or rbd.ccf_groups:
        # (A common-cause group ties its members' modules together.)
        return [], [[n for n in rbd.components if n in structure.nodes]]
    alone: list = []
    modules: list = []

    def under(term: int) -> list:
        entry = structure.terms[term]
        if entry[0] == NODE:
            return [entry[1]]
        return [node for child in entry[1] for node in under(child)]

    def walk(term: int) -> None:
        entry = structure.terms[term]
        if entry[0] == NODE and entry[1] in rbd._standby:
            modules.append([entry[1]])
        elif entry[0] == NODE:
            alone.append(entry[1])
        elif entry[0] == SERIES:
            for child in entry[1]:
                walk(child)
        else:
            modules.append(under(term))

    walk(structure.root)
    return alone, modules


def _node_times(
    rbd, node, states, level: int, horizon: float, own: Optional[dict] = None
) -> tuple:
    """The phases of the life and of the repair of ``node`` (the repair
    a rate of ``inf`` if instant), from its state. A common-cause
    group's member fails on its own at its own cause's rate (``own``)."""
    component = rbd.components[node]
    state = states.get(node)
    up = state is None or state.alive
    age = 0.0 if state is None or not up else float(state.age)
    if own and node in own:
        life = _phase_type.exponential(own[node])
    else:
        life = _phase_type.phases(component.reliability, level, horizon, age)
    if _repair_rate(component.time_to_replace) == math.inf:
        return life, math.inf
    gone = 0.0 if up else float(state.down_for)
    return life, _phase_type.phases(
        component.time_to_replace, level, horizon, gone
    )


def _phases_fit(rbd) -> None:
    """Raise if the diagram's lives and repairs cannot be followed as
    phases: a time with no distribution function or scale, or a module
    whose chain, a level past the fewest a value is extrapolated from
    (few values settle in fewer), has too many states."""
    level = _phase_type.LEVELS[_phase_type.FEWEST]
    for module in _modules(rbd)[1]:
        count = 1
        for node in module:
            if node in rbd._standby:
                count *= len(_group(rbd, node)[0].states)
                continue
            component = rbd.components[node]
            for what, model in (
                ("life", component.reliability),
                ("repair times", component.time_to_replace),
            ):
                if not callable(getattr(model, "ff", None)):
                    _refuse(
                        "A structure that is not a series follows its "
                        f"lives and repairs as phases, and the {what} of "
                        f"component {node!r} have no distribution "
                        "function (ff) to take them from"
                    )
            try:
                life, repair = _node_times(rbd, node, {}, level, math.inf)
            except NotImplementedError as error:
                _refuse(f"Component {node!r}: {error}".rstrip("."))
            count *= len(life.rates) + (
                1 if repair == math.inf else len(repair.rates)
            )
        if count > MAX_PHASE_STATES:
            _refuse(
                "A structure that is not a series follows the lives and "
                "repairs that are not exponential as phases, and the "
                f"chain of components {module} would have more than the "
                f"{MAX_PHASE_STATES:,} states it is followed over (about "
                f"{count:,})"
            )


def _require_exact(rbd) -> str:
    """Raise if the first failure's exact values do not cover the diagram;
    otherwise how it is worked out: ``AS_SERIES``, ``AS_CHAIN`` (the
    Markov chain of exponential components) or ``AS_PHASES`` (chains
    with phases). A sub-system is drawn in (``_flat``)."""
    rbd = _flat(rbd)
    structure = rbd._decomposition()
    if isinstance(structure, GraphStructure):
        raise NotImplementedError(structure.reason)
    _components_are_plain(rbd)
    if _is_series(rbd):
        return AS_SERIES
    rates = _rates(rbd)
    if rates is not None:
        _chain_fits(rbd, rates)
        return AS_CHAIN
    _phases_fit(rbd)
    return AS_PHASES


_GROUPS = (
    " Common-cause groups' shared causes strike in the chain as in the "
    "crews' chain, and a standby group's own chain of its units goes "
    "beside it."
)
_ROUTES = {
    AS_SERIES: "A series structure: the product of each component's "
    "survival from its state at 0, as no repair can come before the first "
    "failure.",
    AS_CHAIN: "The Markov chain of the components' states (with the repair "
    "crews' queue), the states the system is down in made absorbing: "
    "the survival probability by uniformization (to about 1e-13) and "
    "the mean time to absorption from one linear solve.",
    AS_PHASES: "The Markov chain of the components' states (with the "
    "repair crews' queue), the states the system is down in made "
    "absorbing, each life and repair that is not exponential followed as "
    "phases (a mixture of Erlangs on a grid of ages; an Erlang's own, "
    "exactly), each series factor of the structure on its own: the grid "
    "is halved, and the values extrapolated, until they agree to rtol "
    "(1e-3 of each probability of failing, by default), and refused "
    "when the values are asked for if they do not within the chain's "
    "size, which depends on the times.",
}


_MAY_REFUSE = {
    AS_CHAIN: "the chain is followed by uniformization, in steps that "
    "grow with the times asked and with its fastest rate, and is refused "
    "when they are too many (or the chain too large) for the times "
    "asked.",
    AS_PHASES: "the values are refused (NotSettled, a NotImplementedError) "
    "when they do not settle to rtol within the chain's size and the "
    "steps it can take, which grow with the times asked: a looser rtol "
    "takes them, method='simulate' estimates them, and "
    "first_failure_bounds brackets them.",
}


def may_refuse(rbd) -> str:
    """When the exact values can still be refused once asked for (the
    route's ``may_refuse``): "" for a series structure."""
    return _MAY_REFUSE.get(_require_exact(rbd), "")


def exact_route(rbd) -> str:
    """How the exact values are worked out, for ``analysis_routes``."""
    how = _ROUTES[_require_exact(rbd)]
    flat = _flat(rbd)
    return how + (_GROUPS if flat.ccf_groups or flat._standby else "")


# -- a series structure ------------------------------------------------------


def _series_parts(rbd, states: dict, nodes=None) -> list:
    """For each component the system needs (or each of ``nodes``):
    ``(sf, age)``, its survival
    function and age at 0; a component down at 0 makes the system's
    reliability 0 (an empty list is never returned for it: see
    ``_down_at_start``)."""
    structure = rbd._decomposition()
    needed = [
        node
        for node in rbd.components
        if node in (structure.nodes if nodes is None else nodes)
    ]
    parts = []
    for node in needed:
        state = states.get(node)
        age = 0.0 if state is None else float(state.age)
        life = rbd.components[node].reliability
        if state is not None and state.alive and state.own_life:
            # A life of its own (see _overrides), from now or at its age.
            life, at = _overrides.life_now(node, rbd.components[node], state)
            age = 0.0 if at is None else at
        parts.append((life, age))
    return parts


def _down_at_start(rbd, states: dict) -> bool:
    structure = rbd._decomposition()
    return any(
        not states[node].alive
        for node in states
        if node in structure.nodes and not structure.always_works
    )


def _series_sf(parts: list, x: np.ndarray) -> np.ndarray:
    out = np.ones(x.shape)
    for model, age in parts:
        if age > 0.0:
            out = out * (
                np.asarray(model.sf(age + x), dtype=float)
                / float(np.asarray(model.sf(np.array([age])))[0])
            )
        else:
            out = out * np.asarray(model.sf(x), dtype=float)
    return out


def _series_mean(parts: list) -> float:
    knots = [np.array([1.0])]
    for model, age in parts:
        shifted = np.asarray(model_knots(model), dtype=float) - age
        knots.append(shifted[shifted > 0.0])
    return mean_lifetime(
        lambda t: _series_sf(parts, np.asarray(t, dtype=float)),
        np.concatenate(knots),
    )


# -- the Markov chain --------------------------------------------------------


_FAR_APART = (
    "Its Markov chain's rates are too far apart to follow over this long"
)


class _Absorbing:
    """The crews' chain of ``nodes`` (every component, by default) with
    the system-down states made absorbing, from the start distribution
    ``start`` over its states; each life and repair as its phases at
    ``level`` (see ``_phase_type``), to the ages a ``horizon`` reaches.
    The other components are taken as up: ``nodes`` are a series factor
    of the structure. Raise ``_phase_type.Unaffordable`` if the chain is
    too large, or too long to follow."""

    def __init__(
        self,
        rbd,
        states: dict,
        nodes=None,
        level: int = _phase_type.LEVELS[0],
        horizon: float = math.inf,
    ):
        from scipy import sparse

        every = list(rbd.components) if nodes is None else list(nodes)
        groups = [node for node in every if node in rbd._standby]
        nodes = [node for node in every if node not in rbd._standby]
        named = {node: states[node] for node in every if node in states}
        _overrides.refuse_in_chain(named)
        own, shared = _causes(rbd, nodes)
        times = [
            _node_times(rbd, node, named, level, horizon, own)
            for node in nodes
        ]
        self.exact = all(
            time == math.inf or time.exact for pair in times for time in pair
        )
        stationary = [node for node, s in named.items() if s.stationary]
        if stationary:
            if len(stationary) < len(every):
                raise NotImplementedError(
                    "The components' long-run states are tied together by "
                    "the repair queue: start every component in its "
                    "long-run state (state='stationary'), or give each its "
                    "state."
                )
            if any(
                time != math.inf and len(time.rates) > 1
                for pair in times
                for time in pair
            ):
                _refuse(
                    "The exact values do not start lives and repairs that "
                    "are not exponential from their long-run state "
                    "(state='stationary'): give each component's state"
                )
            start = None
        else:
            down = tuple(
                k
                for k, node in enumerate(nodes)
                if node in named and not named[node].alive
            )
            crews = rbd.repair_crews or len(every)
            if len(down) > crews:
                raise ValueError(
                    f"{len(down)} components are down at the start, in "
                    f"repairs going on, which takes more than the {crews} "
                    "repair crew(s): a state does not say which jobs wait."
                )
            start = (down, ())
        chain = crew_chain.solve(
            nodes,
            [life for life, _ in times],
            [repair for _, repair in times],
            [rbd._priority.get(node, 0.0) for node in nodes],
            _crew_count(rbd),
            _crews._call_when(rbd),
            shared,
            start=start,
            limit=MAX_PHASE_STATES + 1,
        )
        size = len(chain.states)
        if start is None:
            begin = np.array(chain.probabilities, dtype=float)
        else:
            begin = np.zeros(size)
            begin[0] = 1.0
        generator = sparse.csr_matrix(chain.generator)
        ups = {
            node: chain.up[:, k].astype(float)
            for k, node in enumerate(chain.nodes)
        }
        for node in groups:
            # A standby group's own chain beside the rest: independent of
            # it, so the states are pairs and the generators add.
            group, first = _group(rbd, node)
            count = len(group.states)
            if size * count > MAX_PHASE_STATES:
                size *= count
                break
            begin = np.kron(
                begin, group.probabilities if start is None else first
            )
            generator = sparse.kron(
                generator, sparse.identity(count), format="csr"
            ) + sparse.kron(
                sparse.identity(size),
                sparse.csr_matrix(group.generator),
                format="csr",
            )
            ups = {name: np.repeat(up, count) for name, up in ups.items()}
            ups[node] = np.tile(group.up.astype(float), size)
            size *= count
        if size > MAX_PHASE_STATES:
            raise _phase_type.Unaffordable(
                "Its Markov chain has more than the "
                f"{MAX_PHASE_STATES:,} states it is followed over"
            )
        structure = rbd._decomposition()
        node_up = {node: np.ones(size) for node in structure.nodes}
        node_up.update(
            {node: up for node, up in ups.items() if node in node_up}
        )
        works, _ = structure.probabilities(node_up, shape=(size,))
        self.up = np.asarray(works, dtype=float) > 0.5
        self.start = begin[self.up]
        self.generator = generator[self.up][:, self.up].tocsr()
        # The system down at the start has failed already.
        self.at_start = float(self.start.sum())
        # Uniformization's steps so far (``_followed``).
        self._jump: Any = None
        self._vector = self.start.copy()
        self._survivals: list = []
        self._over = False

    def _followed(self, rate: float, needed: int, to_end: bool) -> np.ndarray:
        """``s_k = p(0) P^k 1`` for the first ``needed`` steps of the
        chain uniformized at ``rate`` (fewer once the chain has ended),
        kept from call to call. With ``to_end``, as many of them as the
        work allows are taken before giving up: the chain may end first."""
        from scipy import sparse

        matrix = self.generator
        size = matrix.shape[0]
        if self._jump is None:
            self._jump = (
                sparse.identity(size, format="csr") + matrix / rate
            ).T
        steps = min(needed, MAX_STEPS)
        afford = int(MAX_WORK / max(matrix.nnz, 1))
        if not self._over and needed > afford:
            if not to_end:
                raise _phase_type.Unaffordable(_FAR_APART)
            steps = min(steps, afford)
        ones = np.ones(size)
        while len(self._survivals) < steps and not self._over:
            value = dot(self._vector, ones)
            self._survivals.append(value)
            if value < _ENDED:
                self._over = True
                break
            self._vector = self._jump @ self._vector
        if not self._over and steps < needed:
            raise _phase_type.Unaffordable(_FAR_APART)
        return np.array(self._survivals[:needed])

    def sf(self, x: np.ndarray, to_end: bool = False) -> np.ndarray:
        """``p(0) exp(T t) 1`` at each time, by uniformization."""
        from scipy.stats import poisson

        matrix = self.generator
        size = matrix.shape[0]
        if size == 0 or self.at_start == 0.0:
            return np.zeros(x.shape)
        top = float(np.max(-matrix.diagonal()))
        if top == 0.0 or not self._leaks().any():
            return np.full(x.shape, self.at_start)
        rate = 1.02 * top
        mean = rate * float(x.max())
        needed = int(mean + _SPREAD * math.sqrt(mean) + _MARGIN) + 1
        s = self._followed(rate, needed, to_end)
        k = np.arange(len(s))
        out = np.empty(x.shape)
        for i, t in enumerate(x.ravel()):
            out.ravel()[i] = float(poisson.pmf(k, rate * t) @ s)
        return np.minimum(out, self.at_start)

    def _leaks(self) -> np.ndarray:
        """The states the system can fail from at once."""
        matrix = self.generator
        return np.asarray(matrix.sum(axis=1)).ravel() < -1e-12 * np.maximum(
            -matrix.diagonal(), 1.0
        )

    def _ends(self) -> bool:
        """Whether the system can fail from every state the chain may start
        in: from a state where nothing fails (every repair instant, say) it
        never does, and the mean time is infinite."""
        matrix = self.generator
        leak = self._leaks()
        reach = matrix.copy()
        reach.setdiag(0.0)
        reach.eliminate_zeros()
        reach.data[:] = 1.0
        can = leak.copy()
        while True:
            more = can | ((reach @ can.astype(float)) > 0.0)
            if (more == can).all():
                break
            can = more
        return bool(np.all(can[self.start > 0.0]))

    def mean(self) -> float:
        """The mean time to absorption from the start distribution."""
        from scipy.sparse.linalg import spsolve

        if self.generator.shape[0] == 0 or self.at_start == 0.0:
            return 0.0
        if not self._ends():
            return math.inf
        time = spsolve(
            (-self.generator).tocsc(), np.ones(self.generator.shape[0])
        )
        return float(self.start @ np.atleast_1d(time))


class _Factored:
    """The first failure of a diagram whose repairs wait for no crew, as
    its structure's series factors: the components that fail the system
    alone, by their survival, and the other factors, independent of each
    other, each by its chain (``_Absorbing``) at ``level``."""

    def __init__(self, rbd, states: dict, level: int, horizon: float):
        alone, modules = _modules(rbd)
        self.failed = any(
            node in states and not states[node].alive for node in alone
        )
        self.parts = _series_parts(rbd, states, alone)
        self.models = [
            rbd.components[node].reliability
            for module in modules
            for node in module
            if node not in rbd._standby
        ]
        self.chains = [
            _Absorbing(rbd, states, module, level, horizon)
            for module in modules
        ]
        self.exact = all(chain.exact for chain in self.chains)

    def sf(self, x: np.ndarray, to_end: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.failed:
            return np.zeros(x.shape)
        out = _series_sf(self.parts, x)
        # A chain is followed only as far as the others leave anything:
        # one too long to follow to its end may not need to be.
        waiting = list(self.chains)
        while waiting:
            later, short = [], None
            for chain in waiting:
                live = out > _ENDED
                if not live.any():
                    continue
                try:
                    out[live] *= chain.sf(x[live], to_end)
                except _phase_type.Unaffordable as error:
                    later.append(chain)
                    short = error
            if short is not None and len(later) == len(waiting):
                raise short
            waiting = later
        return out

    def mean(self) -> float:
        if not self.parts and len(self.chains) == 1:
            return 0.0 if self.failed else self.chains[0].mean()
        if self.failed:
            return 0.0
        if not self.parts and not any(c._ends() for c in self.chains):
            return math.inf
        knots = [np.array([1.0])]
        for model, age in self.parts:
            shifted = np.asarray(model_knots(model), dtype=float) - age
            knots.append(shifted[shifted > 0.0])
        knots += [
            np.asarray(model_knots(model), dtype=float)
            for model in self.models
        ]
        return mean_lifetime(
            lambda t: self.sf(np.asarray(t, dtype=float), True),
            np.concatenate(knots),
        )


def _failing_size(survival: np.ndarray) -> np.ndarray:
    """What the accuracy of each probability of failing is a share of:
    the probability itself, or ``_SMALL`` of the largest of them if it
    is less (and never less than ``_FLOOR``)."""
    failing = 1.0 - survival
    return np.maximum(
        failing, max(_SMALL * float(np.max(failing, initial=0.0)), _FLOOR)
    )


def _check_rtol(rtol) -> float:
    if not is_number(rtol) or not 0.0 < rtol < 1.0:
        raise ValueError(
            f"rtol must be a number above 0 and below 1, got {rtol!r}."
        )
    return float(rtol)


def _by_phases(
    rbd, states: dict, value, horizon, count: int, rtol, size
) -> np.ndarray:
    """``value(_Factored, which)``, the ``count`` values ``which`` of a
    level's chains, at the levels of phases, each extrapolated until it
    agrees to ``rtol * size(values)`` (``_phase_type.converged``); a
    level's chains are followed to ``horizon(which)``, the ages the
    values not yet settled reach."""
    every = np.arange(count)
    first = _Factored(rbd, states, _phase_type.LEVELS[0], horizon(every))
    return _phase_type.converged(
        lambda level, which: value(
            (
                first
                if level == _phase_type.LEVELS[0]
                else _Factored(rbd, states, level, horizon(which))
            ),
            which,
        ),
        rtol,
        size,
        first.exact,
        _SIMULATE,
        count,
    )


# -- the methods -------------------------------------------------------------


def _chain(rbd, states: dict) -> _Absorbing:
    try:
        return _Absorbing(rbd, states)
    except _phase_type.Unaffordable as error:
        _refuse(str(error))


def _exact_sf(rbd, x: np.ndarray, state, rtol: float) -> np.ndarray:
    rbd = _flat(rbd, state)
    states = _curves._states(rbd, state)
    how = _require_exact(rbd)
    stationary = any(s.stationary for s in states.values())
    if how == AS_SERIES and not stationary:
        if _down_at_start(rbd, states):
            return np.zeros(x.shape)
        return _series_sf(_series_parts(rbd, states), x)
    if how == AS_PHASES and not stationary:
        if x.size == 0:
            return np.zeros(x.shape)
        flat = x.ravel()
        survival = _by_phases(
            rbd,
            states,
            lambda factored, which: factored.sf(flat[which]),
            lambda which: float(flat[which].max()),
            flat.size,
            rtol,
            _failing_size,
        )
        return np.clip(survival, 0.0, 1.0).reshape(x.shape)
    try:
        return _chain(rbd, states).sf(x)
    except _phase_type.Unaffordable as error:
        _refuse(str(error))


def _exact_mean(rbd, state, rtol: float) -> float:
    rbd = _flat(rbd, state)
    states = _curves._states(rbd, state)
    how = _require_exact(rbd)
    stationary = any(s.stationary for s in states.values())
    if how == AS_SERIES and not stationary:
        if _down_at_start(rbd, states):
            return 0.0
        return _series_mean(_series_parts(rbd, states))
    if how == AS_PHASES and not stationary:
        return float(
            _by_phases(
                rbd,
                states,
                lambda factored, which: np.array([factored.mean()]),
                lambda which: math.inf,
                1,
                rtol,
                lambda value: np.maximum(np.abs(value), _FLOOR),
            )[0]
        )
    return _chain(rbd, states).mean()


def first_failure_sf(
    rbd,
    x,
    state=None,
    method: str = "exact",
    t_simulation: Optional[float] = None,
    mc_samples: Optional[int] = None,
    seed: Optional[int] = None,
    engine: str = "auto",
    rtol: float = 1e-3,
):
    """See ``RepairableRBD.first_failure_sf``."""
    one_of("method", method, ("exact", "simulate"))
    times = nonnegative_times(x)
    scalar = np.ndim(x) == 0
    rtol = _check_rtol(rtol)
    if method == "exact":
        simulation_options(
            "first_failure_sf()",
            {
                "t_simulation": t_simulation,
                "mc_samples": mc_samples,
                "seed": seed,
                "engine": None if engine == "auto" else engine,
            },
        )
        out = _exact_sf(rbd, times, state, rtol)
    else:
        runs = _simulated(
            rbd,
            float(times.max()) if t_simulation is None else t_simulation,
            state,
            mc_samples,
            seed,
            engine,
        )
        first = runs.system.first_failure
        out = np.array([float(np.mean(first > t)) for t in times])
    return float(out[0]) if scalar else out.reshape(np.shape(x))


def _simulated(rbd, t_simulation, state, mc_samples, seed, engine):
    return rbd.simulate_timelines(
        t_simulation,
        mc_samples=mc_samples,
        seed=seed,
        engine=engine,
        state=state,
    )


def mean_time_to_first_failure(
    rbd,
    state=None,
    method: str = "exact",
    t_simulation: Optional[float] = None,
    mc_samples: Optional[int] = None,
    seed: Optional[int] = None,
    engine: str = "auto",
    confidence: float = 0.95,
    rtol: float = 1e-3,
):
    """See ``RepairableRBD.mean_time_to_first_failure``."""
    one_of("method", method, ("exact", "simulate"))
    rtol = _check_rtol(rtol)
    if method == "exact":
        simulation_options(
            "mean_time_to_first_failure()",
            {
                "t_simulation": t_simulation,
                "mc_samples": mc_samples,
                "seed": seed,
                "engine": None if engine == "auto" else engine,
            },
        )
        return _exact_mean(rbd, state, rtol)
    if t_simulation is None:
        raise ValueError(
            "method='simulate' needs t_simulation, the time each "
            "simulation runs to."
        )
    from scipy.stats import norm

    runs = _simulated(rbd, t_simulation, state, mc_samples, seed, engine)
    first = runs.system.first_failure
    limited = np.minimum(first, float(t_simulation))
    cut = int(np.sum(~np.isfinite(first)))
    if cut:
        warnings.warn(
            f"{cut} of {len(first)} simulations had no system failure by "
            f"t_simulation={t_simulation}: the estimate is their mean "
            "time to the first failure or t_simulation, a lower bound of "
            "the mean. Simulate for longer for the mean itself.",
            stacklevel=2,
        )
    n = len(limited)
    error = float(np.std(limited, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = float(norm.ppf(0.5 + confidence / 2.0))
    estimate = float(np.mean(limited))
    return ConfidenceInterval(
        estimate=estimate,
        lower=max(0.0, estimate - z * error),
        upper=estimate + z * error,
        confidence=confidence,
        standard_error=error,
        n_samples=n,
        method="simulated",
    )


# -- the bracket -------------------------------------------------------------


def _start_down(rbd, state) -> float:
    """``Q(0)``: the probability that the system is down at the start."""
    return float(
        np.asarray(rbd.point_unavailability(np.array([0.0]), state=state))[0]
    )


def _expected_failures(rbd, times: np.ndarray, state) -> np.ndarray:
    return np.asarray(
        rbd.expected_events(times, state=state).system_failures, dtype=float
    ).reshape(times.shape)


def first_failure_bounds(rbd, t, state=None) -> FirstFailureBounds:
    """See ``RepairableRBD.first_failure_bounds``."""
    times = nonnegative_times(t).ravel()
    scalar = np.ndim(t) == 0
    failures = _expected_failures(rbd, times, state)
    down = np.asarray(
        rbd.point_unavailability(np.concatenate(([0.0], times)), state=state),
        dtype=float,
    ).reshape(times.size + 1)
    at_start, unavailability = float(down[0]), down[1:]
    lower = np.clip(1.0 - at_start - failures, 0.0, 1.0)
    planned = _unplanned.takes_planned_outages(rbd)
    if planned:
        upper = np.ones(times.shape)
    else:
        # The first failure's probability never falls: it is at least the
        # unavailability at every time given up to each.
        order = np.argsort(times, kind="stable")
        highest = np.maximum.accumulate(
            np.maximum(unavailability[order], at_start)
        )
        upper = np.empty(times.shape)
        upper[order] = np.clip(1.0 - highest, 0.0, 1.0)

    def shaped(values: np.ndarray):
        return float(values[0]) if scalar else values.reshape(np.shape(t))

    return FirstFailureBounds(
        time=shaped(times),
        lower=shaped(lower),
        upper=shaped(upper),
        expected_failures=shaped(failures),
        unavailability=shaped(unavailability),
        planned_outages=planned,
    )


#: The relative accuracy of the lower bound of the mean: below the
#: expected failures' own (about 1e-7), which it is no more accurate than.
_MEAN_RTOL = 1e-8
#: The times at which the expected failures are worked out at once, in the
#: search for the time the lower bound reaches 0.
_BATCH = 16
#: The relative width that time is found to.
_END_RTOL = 1e-6


def _bound_ends(rbd, state, up: float) -> Tuple[float, float]:
    """``(low, high)``, within ``_END_RTOL`` of each other, with ``W(low) <
    up <= W(high)``: where the lower bound ``up - W(t)`` reaches 0; ``(inf,
    inf)`` if it never does. Searched for from 1 (whatever the unit of
    time): down by powers of 2, ``_BATCH`` at a time; up by doubling one
    at a time, as the expected events are worked out to the furthest time
    asked; then on a geometric grid between the two found."""
    low = high = 1.0
    if _expected_failures(rbd, np.array([1.0]), state)[0] < up:
        while _expected_failures(rbd, np.array([high]), state)[0] < up:
            low, high = high, 2.0 * high
            if high > 1e300:
                return math.inf, math.inf
    else:
        while low == high:
            times = high * 2.0 ** -np.arange(_BATCH, 0, -1.0)
            short = _expected_failures(rbd, times, state) < up
            if not short.any():
                high = times[0]
                low = high
                if high < 1e-300:
                    return 0.0, 0.0
                continue
            k = int(np.sum(short))
            low = times[k - 1]
            high = times[k] if k < times.size else high
    while high - low > _END_RTOL * high:
        times = np.geomspace(low, high, _BATCH + 2)[1:-1]
        short = _expected_failures(rbd, times, state) < up
        k = int(np.sum(short))
        low = times[k - 1] if k else low
        high = times[k] if k < times.size else high
    return low, high


def mean_time_to_first_failure_bound(rbd, state=None) -> float:
    """See ``RepairableRBD.mean_time_to_first_failure_bound``."""
    up = 1.0 - _start_down(rbd, state)
    if up <= 0.0:
        return 0.0
    low, high = _bound_ends(rbd, state, up)
    if not math.isfinite(high):
        return math.inf
    if high == 0.0:
        return 0.0

    def bound(x: np.ndarray) -> np.ndarray:
        # Always with ``high`` among the windows, so that every call sums
        # on the same pieces and the curve is one function of time.
        counted = _expected_failures(rbd, np.append(x, high), state)[:-1]
        return np.maximum(up - counted, 0.0)

    # Pieces an octave long up to ``low``; the rest, to ``high``, is a
    # triangle at most ``_END_RTOL`` of the whole.
    edges = np.concatenate(([0.0], low * 2.0 ** -np.arange(30.0, -1.0, -1.0)))
    mean = _integrate(bound, edges, _MEAN_RTOL)
    return mean + 0.5 * float(bound(np.array([low]))[0]) * (high - low)
