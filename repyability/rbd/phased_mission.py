"""Phased missions (#100): a mission that runs through phases (take-off,
cruise, landing), each with its own duration and its own diagram over the
same non-repairable components.

A component that fails in one phase stays failed in the later ones, so the
phases are not independent, and the mission's reliability is not the
product of theirs. The mission succeeds if each phase's diagram works
throughout the phase; as the components only ever fail, that is its
working at the phase's end, ``T_j``.

The exact values use the classic transformation of Esary and Ziehms (1975):
a component's life through the mission is a chain of independent segments,
one per phase, the ``k``-th survived with probability ``R(T_k) /
R(T_{k-1})``, and the component works at ``T_j`` if it survives its first
``j`` segments. The mission succeeds when every phase's diagram works with
its components so, a condition on independent segments that a binary
decision diagram works out exactly (#142), after Zang, Sun and Trivedi
(1999): each phase's structure is built from its diagram's modules and
core (see ``modular.py``) as an ordered decision diagram over the
segments, a component's segments next to each other in phase order, and
the phases are combined by conjunction (see ``_ordered_bdd.py``). No path
set is listed, so a meshed phase stays as cheap as its diagram.
``METHOD = "paths"`` decomposes the phases' minimal path sets over the
segments instead, as before (``_mission_plan``). The simulation draws each
component's life once per mission instead.

The importance measures (#316) hold a component across every phase: working
throughout (every segment survived) or failed from the start (none), and
replay the plan with its segments so held. The mission's minimal cut sets
are sets of events "component failed by the end of phase j", from the
phases' own cut sets with the usual phased-mission minimisation (a set is
dropped when another is implied by it: fewer components, or later phases);
the exact Fussell-Vesely importance takes the probability that some cut set
naming a component has occurred from a decision diagram of those events
over the segments (``_cut_set_plan``).
"""

import math
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import bdd, modular, routes
from repyability.rbd._ordered_bdd import FALSE, TRUE, OrderedBDD
from repyability.rbd._sampling import lifetime_sampler
from repyability.rbd.results import ConfidenceInterval
from repyability.rbd.shannon import _plan_rows, replay, room
from repyability.utils.checks import one_of
from repyability.utils.deprecation import refuse_removed_names
from repyability.utils.wrappers import numpy_seed

#: How the exact values are worked out: ``"bdd"`` (the default), by the
#: phases' decision diagrams over the segments, or ``"paths"``, by the
#: Shannon decomposition of the phases' minimal path sets over them
#: (slower, but for the smallest missions). Read when a mission's values
#: are first worked out.
METHOD = "bdd"

#: The most nodes of the missions's decision diagram before the exact values
#: refuse, pointing to the simulation (and of the cut sets' diagram before
#: the exact Fussell-Vesely importance refuses, pointing to the rare-event
#: sum).
MAX_NODES = 1_000_000

#: The most distinct sub-problems the decomposition of the path sets
#: (``METHOD = "paths"``) solves before it refuses.
MAX_STATES = 200_000

# Value slots 0 and 1 of a plan hold the mission failing and succeeding.
_FAIL, _SUCCEED = 0, 1


class _Phase(NamedTuple):
    name: Hashable
    duration: float
    rbd: Any
    #: When the phase ends, from the mission's start.
    end: float
    #: The components its diagram uses.
    components: Tuple[Hashable, ...]


def _mission_plan(families: Sequence[Sequence[frozenset]]) -> tuple:
    """The Shannon decomposition of the probability that each of
    ``families`` has a set whose elements are all active (the elements
    independent), as a plan ``_shannon_value_and_gradient`` replays: step
    ``i`` fills value slot ``i + 2`` from its pivot's two branches.

    A state is the families still to be satisfied, each with the elements
    already active taken out of its sets: a family with an empty set is
    satisfied and drops out, and a state with a family that has no set left
    fails. Each distinct state is solved once.

    Raises
    ------
    NotImplementedError
        If more than ``MAX_STATES`` states are needed.
    """
    slots: Dict[frozenset, int] = {}
    steps: List[tuple] = []

    def settle(families) -> Optional[frozenset]:
        """The state of ``families``: None if one cannot be satisfied."""
        state = []
        for family in families:
            if frozenset() in family:
                continue
            if not family:
                return None
            state.append(family)
        return frozenset(state)

    def known(state: Optional[frozenset]) -> Optional[int]:
        if state is None:
            return _FAIL
        if not state:
            return _SUCCEED
        return slots.get(state)

    def split(state: frozenset) -> list:
        # Pivot on the element in the most sets, the lowest on a tie (the
        # elements are numbered, so the plan is the same in every run).
        counts: Dict[int, int] = {}
        for family in state:
            for elements in family:
                for element in elements:
                    counts[element] = counts.get(element, 0) + 1
        pivot = max(counts, key=lambda e: (counts[e], -e))
        active = settle(
            frozenset(s - {pivot} for s in family) for family in state
        )
        inactive = settle(
            frozenset(s for s in family if pivot not in s) for family in state
        )
        return [state, pivot, [active, inactive], []]

    root_state = settle(frozenset(family) for family in families)
    root = known(root_state)
    stack = []
    if root is None:
        assert root_state is not None
        stack.append(split(root_state))
    while stack:
        state, pivot, branches, solved = stack[-1]
        if len(solved) < 2:
            branch = branches[len(solved)]
            slot = known(branch)
            if slot is None:
                if len(slots) + len(stack) > MAX_STATES:
                    raise NotImplementedError(
                        "The mission is too large to solve exactly (more "
                        f"than {MAX_STATES:,} states of its decomposition). "
                        "Simulate it: method='simulate'."
                    )
                stack.append(split(branch))
            else:
                solved.append(slot)
            continue
        steps.append((pivot, solved[0], solved[1]))
        slots[state] = len(steps) + 1
        stack.pop()
        if stack:
            stack[-1][3].append(slots[state])
        else:
            root = slots[state]
    assert root is not None
    return steps, root


def _phase_diagram(
    manager: OrderedBDD, rbd, j: int, rank: Dict[Hashable, int], count: int
) -> int:
    """Phase ``j``'s structure as a decision diagram over the segments:
    built from its diagram's modules and core (see ``modular.py``), each
    component working when its first ``j + 1`` segments are survived.
    Component ``c``'s ``k``-th segment is variable ``rank[c] * count + k``,
    so a component's segments are decided together, in phase order."""
    try:
        decomposition = rbd._decomposition()
    except ValueError:
        return FALSE  # no set of working nodes makes the phase work
    if decomposition.always_works:
        return TRUE
    made: List[int] = []
    for term in decomposition.terms:
        kind = term[0]
        if kind == modular.NODE:
            chain = TRUE
            for k in range(j, -1, -1):
                chain = manager.node(rank[term[1]] * count + k, FALSE, chain)
            made.append(chain)
        elif kind == modular.SERIES:
            made.append(manager.conjunction([made[c] for c in term[1]]))
        elif kind == modular.PARALLEL:
            made.append(manager.disjunction([made[c] for c in term[1]]))
        else:
            made.append(manager.at_least([made[c] for c in term[1]], term[2]))
    if decomposition.root is not None:
        return made[decomposition.root]
    steps, root = decomposition.core_plan()
    slots = [FALSE, TRUE]
    for pivot, active, inactive in steps:
        slots.append(manager.ite(made[pivot], slots[active], slots[inactive]))
    return slots[root]


def _outcomes(
    plan: tuple, p: Dict[int, float], q: Dict[int, float]
) -> Tuple[float, float]:
    """The probabilities that the mission of ``plan`` succeeds and that it
    fails, for the segments' survival probabilities ``p`` and their
    complements ``q`` (by element number), each to its own relative
    precision: the plan replayed with the terminals of failing and
    succeeding at 0 and 1, and again with them swapped."""
    pivots, actives, inactives, names = _plan_rows(plan)
    works = [p[name] for name in names]
    fails = [q[name] for name in names]
    success, failure = (
        replay(
            pivots,
            actives,
            inactives,
            plan[1],
            works,
            fails,
            room(len(pivots), fail, work),
        )
        for fail, work in ((0.0, 1.0), (1.0, 0.0))
    )
    return float(success), float(failure)


def _weaker(one: Dict[Hashable, int], other: Dict[Hashable, int]) -> bool:
    """Whether the cut set ``one`` (each component's phase, by number) is
    implied by ``other`` and is not it: every event of ``one`` follows from
    one of ``other``, the same component failed by that phase or an
    earlier one. Then ``other`` is no minimal cut set."""
    return one != other and all(
        component in other and other[component] <= phase
        for component, phase in one.items()
    )


@refuse_removed_names
class PhasedMission:
    """A mission through phases, each with its own duration and diagram over
    the same non-repairable components.

    A mission (take-off, cruise, landing) runs through its phases in order.
    Each phase is a ``NonRepairableRBD`` over some of the mission's
    components, which it must keep working for the phase's duration. A
    component is the same in every phase that uses it (the same node name,
    with the same model), and its life runs through the whole mission: once
    failed, it stays failed in the later phases, even those whose diagrams
    tolerate it. So the phases are not independent, and the mission's
    reliability is not the product of theirs. The mission succeeds if every
    phase's diagram works to the phase's end.

    ``reliability`` and ``phase_failure_probabilities`` are exact by
    default: each component's life through the mission is a chain of
    independent segments, one per phase (Esary and Ziehms, 1975), and the
    phases' structures over those segments are combined in one binary
    decision diagram, built from their diagrams' modules and cores
    without listing path sets (see the module docstring), up to
    ``MAX_NODES`` nodes. ``method="simulate"`` draws each component's life
    once per mission
    instead, and ``reliability_interval`` gives the simulated reliability
    with a confidence interval, to a tolerance if asked. The importance
    measures (``birnbaum_importance`` and the others) hold a component
    across every phase, working throughout or failed from the start, and
    ``minimal_cut_sets`` gives the events, a component failed by the end
    of a phase, that together fail the mission.

    Parameters
    ----------
    phases : Sequence[tuple[Hashable, float, NonRepairableRBD]]
        The phases in order, each a ``(name, duration, rbd)``: a distinct
        name, a duration (finite, and not negative: a phase of no duration
        needs its diagram working at that instant), and the phase's
        diagram, whose node names are the mission's components.

    Attributes
    ----------
    phases : list
        The phases, each with its ``name``, ``duration``, ``rbd``, ``end``
        (from the mission's start) and the ``components`` it uses.
    components : dict
        Each component's model, by node name, in order of first use.
    duration : float
        The mission's length: the sum of the phases' durations.

    Raises
    ------
    ValueError
        If ``phases`` is empty, a phase is not a ``(name, duration, rbd)``
        with a distinct name, a duration that is finite and not negative
        and a ``NonRepairableRBD``, a phase has common-cause groups, or a
        node name has different models in different phases.

    Examples
    --------
    Two engines, either of which is enough to cruise, but both needed to
    take off, for 0.1 hours then 5:

    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD, PhasedMission
    >>> engine = surv.Exponential.from_params([0.01])
    >>> both = NonRepairableRBD(
    ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
    ...     {"e1": engine, "e2": engine},
    ... )
    >>> either = NonRepairableRBD(
    ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
    ...     {"e1": engine, "e2": engine},
    ... )
    >>> mission = PhasedMission(
    ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
    ... )
    >>> round(mission.reliability(), 6)
    0.995628
    """

    def __init__(self, phases: Sequence[Tuple[Hashable, float, Any]]):
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD

        phases = list(phases)
        if not phases:
            raise ValueError("A mission needs at least one phase.")
        self.components: Dict[Hashable, Any] = {}
        self.phases: List[_Phase] = []
        names: set = set()
        end = 0.0
        for phase in phases:
            if not (isinstance(phase, (tuple, list)) and len(phase) == 3):
                raise ValueError(
                    "Each phase must be a (name, duration, rbd), got "
                    f"{phase!r}."
                )
            name, duration, rbd = phase
            if name in names:
                raise ValueError(f"Two phases are named {name!r}.")
            names.add(name)
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float, np.number))
                or not 0.0 <= float(duration) < np.inf
            ):
                raise ValueError(
                    f"Phase {name!r}: the duration must be a finite number, "
                    f"not negative, got {duration!r}."
                )
            if not isinstance(rbd, NonRepairableRBD):
                raise ValueError(
                    f"Phase {name!r}: the diagram must be a NonRepairableRBD, "
                    f"got {type(rbd).__name__}."
                )
            if rbd.ccf_groups:
                raise ValueError(
                    f"Phase {name!r} has common-cause groups, which a phased "
                    "mission does not model."
                )
            ends = {rbd.input_node, rbd.output_node}
            used = tuple(n for n in rbd._components() if n not in ends)
            for node in used:
                model = rbd.reliabilities[node]
                known = self.components.setdefault(node, model)
                if not NonRepairableRBD._same_model(known, model):
                    raise ValueError(
                        f"Node {node!r} has a different model in phase "
                        f"{name!r} than before: a component keeps one "
                        "lifetime model through the mission."
                    )
            end += float(duration)
            self.phases.append(_Phase(name, float(duration), rbd, end, used))
        self.duration = end
        # The decomposition of the first so many phases, once worked out.
        self._plans: Dict[int, tuple] = {}
        # The minimal cut sets, and the plan of their occurrence by
        # component (see ``_cut_set_plan``), once worked out.
        self._cut_sets: Optional[List[frozenset]] = None
        self._cut_plan: Optional[Tuple[tuple, Dict[Hashable, int]]] = None

    # ------------------------------------------------------------------
    # Exact
    # ------------------------------------------------------------------

    def _at_ends(self) -> Dict[Hashable, Tuple[np.ndarray, np.ndarray]]:
        """Each component's probabilities of working and of having failed
        at the phases' ends, from its model (each in its own right, so a
        small one keeps its precision)."""
        ends = np.array([phase.end for phase in self.phases])
        out: Dict[Hashable, Tuple[np.ndarray, np.ndarray]] = {}
        for node, model in self.components.items():
            with np.errstate(all="ignore"):
                sf = np.clip(np.ravel(model.sf(ends)).astype(float), 0, 1)
                ff = np.clip(np.ravel(model.ff(ends)).astype(float), 0, 1)
            out[node] = (sf, ff)
        return out

    def _segments(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Each component's segments' survival probabilities, and their
        complements, by element number (``component * phases + k``): the
        ``k``-th survived with ``R(T_k) / R(T_{k-1})``, the first from the
        mission's start (so a unit dead on arrival fails in it)."""
        count = len(self.phases)
        p: Dict[int, float] = {}
        q: Dict[int, float] = {}
        for index, (sf, ff) in enumerate(self._at_ends().values()):
            before_sf, before_ff = 1.0, 0.0
            for k in range(count):
                element = index * count + k
                if before_sf > 0.0:
                    p[element] = min(sf[k] / before_sf, 1.0)
                    q[element] = min(
                        max(ff[k] - before_ff, 0.0) / before_sf, 1.0
                    )
                else:
                    p[element], q[element] = 0.0, 1.0
                before_sf, before_ff = sf[k], ff[k]
        return p, q

    def _families(self) -> List[List[frozenset]]:
        """Each phase's minimal path sets over the segments: a path of
        phase ``j`` needs each of its components' first ``j`` segments."""
        index = {node: i for i, node in enumerate(self.components)}
        count = len(self.phases)
        families = []
        for j, phase in enumerate(self.phases):
            try:
                paths = phase.rbd.get_min_path_sets(include_in_out_nodes=False)
            except ValueError:
                paths = set()  # no set of working nodes makes it work
            families.append(
                [
                    frozenset(
                        index[node] * count + k
                        for node in path
                        for k in range(j + 1)
                    )
                    for path in paths
                ]
            )
        return families

    def _order(self) -> List[Hashable]:
        """The components in the order the decision diagram decides them:
        as the phases' structures meet them, the first phase's first (its
        core's terms in the order its own decision diagram decides them,
        each term's components depth first), then any no phase needs."""
        order: Dict[Hashable, None] = {}
        for phase in self.phases:
            try:
                decomposition = phase.rbd._decomposition()
            except ValueError:
                continue  # the phase cannot work: it uses no component
            terms = decomposition.terms
            if decomposition.root is not None:
                tops = [decomposition.root]
            elif decomposition.always_works:
                tops = []
            else:
                tops = bdd.pivots(decomposition.core_plan())
            stack = list(reversed(tops))
            while stack:
                term = terms[stack.pop()]
                if term[0] == modular.NODE:
                    order.setdefault(term[1], None)
                else:
                    stack.extend(reversed(term[1]))
        for component in self.components:
            order.setdefault(component, None)
        return list(order)

    def _diagram_plans(self) -> List[tuple]:
        """For each number of phases, the plan of the probability that the
        mission gets through that many (see ``_mission_plan``), from the
        decision diagram of the phases over the segments.

        Raises
        ------
        NotImplementedError
            If the diagram needs more than ``MAX_NODES`` nodes.
        """
        count = len(self.phases)
        order = self._order()
        rank = {component: i for i, component in enumerate(order)}
        element = {component: i for i, component in enumerate(self.components)}
        manager = OrderedBDD(
            MAX_NODES,
            "The mission is too large to solve exactly (more than "
            f"{MAX_NODES:,} nodes of its decision diagram). Simulate it: "
            "method='simulate'.",
        )
        mission, roots = TRUE, []
        for j, phase in enumerate(self.phases):
            mission = manager.ite(
                mission,
                _phase_diagram(manager, phase.rbd, j, rank, count),
                FALSE,
            )
            roots.append(mission)

        def pivot(variable: int) -> int:
            # A segment's element number, as ``_segments`` numbers it.
            component = order[variable // count]
            return element[component] * count + variable % count

        plans = []
        for root in roots:
            steps, (slot,) = manager.plan([root], pivot)
            plans.append((steps, slot))
        return plans

    def _plan(self, phases: int) -> tuple:
        """The plan of the probability that the mission gets through its
        first ``phases`` phases (see ``_mission_plan``), worked out once.

        Raises
        ------
        NotImplementedError
            If the diagram needs more than ``MAX_NODES`` nodes (with
            ``METHOD = "paths"``, the decomposition more than
            ``MAX_STATES`` states).
        """
        plan = self._plans.get(phases)
        if plan is None:
            if METHOD not in ("bdd", "paths"):
                raise ValueError(
                    f"phased_mission.METHOD must be 'bdd' or 'paths', got "
                    f"{METHOD!r}."
                )
            if METHOD == "paths":
                plan = _mission_plan(self._families()[:phases])
                self._plans[phases] = plan
            else:
                for j, made in enumerate(self._diagram_plans()):
                    self._plans[j + 1] = made
                plan = self._plans[phases]
        return plan

    def _exact(self, phases: int) -> Tuple[float, float]:
        """The probability that the mission gets through its first
        ``phases`` phases, and that it fails in them, each to its own
        relative precision."""
        return _outcomes(self._plan(phases), *self._segments())

    # ------------------------------------------------------------------
    # Importance and cut sets (#316)
    # ------------------------------------------------------------------

    def _held(self) -> Dict[Hashable, Tuple[float, float, float, float]]:
        """For each component, the mission's reliability and unreliability
        with it working throughout (every segment survived), then with it
        failed from the start (none): the plan replayed with its segments
        so held."""
        plan = self._plan(len(self.phases))
        p, q = self._segments()
        count = len(self.phases)
        out: Dict[Hashable, Tuple[float, float, float, float]] = {}
        for index, component in enumerate(self.components):
            segments = range(index * count, (index + 1) * count)
            working = _outcomes(
                plan,
                {**p, **dict.fromkeys(segments, 1.0)},
                {**q, **dict.fromkeys(segments, 0.0)},
            )
            failed = _outcomes(
                plan,
                {**p, **dict.fromkeys(segments, 0.0)},
                {**q, **dict.fromkeys(segments, 1.0)},
            )
            out[component] = (*working, *failed)
        return out

    def _measures(self) -> Dict[Hashable, Tuple[float, float, float]]:
        """Each component's Birnbaum importance, ``R(works) - R(failed)``,
        improvement potential, ``R(works) - R``, and the fall in the
        reliability with it failed from the start, ``R - R(failed)``, each
        taken from the probabilities of succeeding or of failing,
        whichever are the smaller (a difference of values near 1 would
        cancel), as a diagram's are (see
        ``modular.Decomposition.value_and_gradient``): ``R(works) -
        R(failed)`` is ``F(failed) - F(works)``."""
        success, failure = self._exact(len(self.phases))
        out: Dict[Hashable, Tuple[float, float, float]] = {}
        for component, (
            r_works,
            f_works,
            r_failed,
            f_failed,
        ) in self._held().items():
            if success <= failure:
                out[component] = (
                    r_works - r_failed,
                    r_works - success,
                    success - r_failed,
                )
            else:
                out[component] = (
                    f_failed - f_works,
                    failure - f_works,
                    f_failed - failure,
                )
        return out

    def _cut_set_plan(self) -> Tuple[tuple, Dict[Hashable, int]]:
        """One plan (in ``shannon.py``'s format) of, for each component,
        the event that some minimal cut set naming it (in any phase) has
        occurred, with the root slot of each component's: a decision
        diagram over the segments, an event "failed by the end of phase
        ``j``" being the component's first ``j + 1`` segments not all
        survived. Worked out once.

        Raises
        ------
        NotImplementedError
            If the diagram needs more than ``MAX_NODES`` nodes.
        """
        if self._cut_plan is None:
            count = len(self.phases)
            index = {node: i for i, node in enumerate(self.components)}
            phase = {ph.name: j for j, ph in enumerate(self.phases)}
            manager = OrderedBDD(
                MAX_NODES,
                "The mission's cut sets are too many to combine exactly "
                f"(more than {MAX_NODES:,} nodes of their decision "
                "diagram). Sum them instead: method='rare_event'.",
            )
            occurred: Dict[Hashable, List[int]] = {
                node: [] for node in self.components
            }
            for cut in self.minimal_cut_sets():
                events = []
                for node, name in cut:
                    chain = TRUE
                    for k in range(phase[name], -1, -1):
                        chain = manager.node(
                            index[node] * count + k, FALSE, chain
                        )
                    events.append(manager.ite(chain, FALSE, TRUE))
                diagram = manager.conjunction(events)
                for node, _ in cut:
                    occurred[node].append(diagram)
            roots = {
                node: manager.disjunction(diagrams)
                for node, diagrams in occurred.items()
            }
            steps, slots = manager.plan(list(roots.values()), lambda v: v)
            # The plan's own root is no one component's: each's slot is
            # read from the replayed values.
            self._cut_plan = ((steps, _FAIL), dict(zip(roots, slots)))
        return self._cut_plan

    def _fv_numerators(self, method: str) -> Dict[Hashable, float]:
        """For each component, the probability that some minimal cut set
        naming it has occurred (``method="exact"``), or the sum of the cut
        sets' probabilities (``"rare_event"``), each cut set's the product
        of its events', ``F_c(T_j)`` for "``c`` failed by the end of phase
        ``j``"."""
        one_of("method", method, ("exact", "rare_event"))
        if method == "exact":
            plan, slots = self._cut_set_plan()
            pivots, actives, inactives, names = _plan_rows(plan)
            p, q = self._segments()
            values = room(len(pivots), 0.0, 1.0)
            replay(
                pivots,
                actives,
                inactives,
                _FAIL,
                [p[name] for name in names],
                [q[name] for name in names],
                values,
            )
            return {node: float(values[slot]) for node, slot in slots.items()}
        failed = self._at_ends()
        phase = {ph.name: j for j, ph in enumerate(self.phases)}
        out: Dict[Hashable, float] = dict.fromkeys(self.components, 0.0)
        for cut in self.minimal_cut_sets():
            chance = float(
                np.prod([failed[node][1][phase[name]] for node, name in cut])
            )
            for node, _ in cut:
                out[node] += chance
        return out

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _failure_phases(self, size: int, antithetic: bool) -> np.ndarray:
        """The phase in which each of ``size`` simulated missions fails
        (its number, ``len(phases)`` for none), each component's life drawn
        once per mission from numpy's global RNG as it stands."""
        samplers = {
            node: lifetime_sampler(model)
            for node, model in self.components.items()
        }
        if antithetic and any(s is None for s in samplers.values()):
            raise NotImplementedError(
                "Antithetic sampling needs every component's lives to be "
                "drawn from uniforms (surpyval parametric distributions and "
                "the composite nodes built from them)."
            )
        width = sum(s.width for s in samplers.values() if s is not None)
        rows = size // 2 if antithetic else size
        u = np.random.random_sample((rows, width))
        lives: Dict[Hashable, np.ndarray] = {}
        start = 0
        for node, sampler in samplers.items():
            if sampler is None:
                model = self.components[node]
                lives[node] = np.asarray(model.random(size), dtype=float)
                continue
            block = u[:, start : start + sampler.width]
            start += sampler.width
            if antithetic:
                both = np.empty(size)
                both[0::2] = sampler.draw(block)
                both[1::2] = sampler.draw(1.0 - block)
                lives[node] = both
            else:
                lives[node] = np.asarray(sampler.draw(block), dtype=float)
        failed = np.full(size, len(self.phases))
        for j in range(len(self.phases) - 1, -1, -1):
            phase = self.phases[j]
            rbd = phase.rbd
            nodes = {
                n: lives.get(n, np.full(size, np.inf))
                for n in rbd._components()
            }
            lifetime = np.asarray(
                rbd._decomposition().lifetime(nodes, size), dtype=float
            )
            failed[~(lifetime > phase.end)] = j
        return failed

    def _simulated(
        self,
        mc_samples: int,
        seed,
        antithetic: bool = False,
        tolerance=None,
        confidence: float = 0.95,
        max_samples=None,
    ) -> np.ndarray:
        """The failure phases of ``mc_samples`` simulated missions, then
        more while a run to ``tolerance`` needs them."""
        montecarlo.check_count(mc_samples, antithetic, "mc_samples")
        montecarlo.check_confidence(confidence)
        limit = montecarlo.sample_limit(
            mc_samples,
            tolerance,
            max_samples,
            antithetic,
            ("mc_samples", "max_samples"),
        )
        parts: List[np.ndarray] = []
        with numpy_seed(seed):
            batch = mc_samples
            while batch:
                parts.append(self._failure_phases(batch, antithetic))
                batch = 0
                if limit is not None:
                    succeeded = np.concatenate(parts) == len(self.phases)
                    batch = montecarlo.more_samples(
                        succeeded.astype(float),
                        mc_samples,
                        tolerance,
                        confidence,
                        limit,
                        antithetic,
                        "mission reliability",
                        "max_samples",
                    )
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reliability(
        self,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> float:
        """The probability that the mission succeeds: that every phase's
        diagram works to the phase's end.

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default): the Esary-Ziehms decomposition (see
            the class docstring). ``"simulate"``: the fraction of
            ``mc_samples`` simulated missions that succeed.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000
            (``method="simulate"`` only).
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float
            The mission's reliability.

        Raises
        ------
        ValueError
            If ``method`` is unknown, or ``mc_samples`` is given with
            ``method="exact"`` or is not a positive integer.
        NotImplementedError
            If the mission's decision diagram needs more than
            ``MAX_NODES`` nodes (with ``METHOD = "paths"``, the
            decomposition more than ``MAX_STATES`` sub-problems): simulate
            instead.

        Examples
        --------
        Two phases with the same diagram are one phase as long as both:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> pair = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> mission = PhasedMission(
        ...     [("first", 30.0, pair), ("second", 20.0, pair)]
        ... )
        >>> round(mission.reliability(), 10) == round(float(pair.sf(50.0)), 10)
        True
        """
        return self._outcome(method, mc_samples, seed)[0]

    def unreliability(
        self,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> float:
        """The probability that the mission fails, ``1 - reliability()``,
        computed to its own relative precision when it is small (see
        ``reliability`` for the parameters).

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000.
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float
            The mission's unreliability.
        """
        return self._outcome(method, mc_samples, seed)[1]

    #: The mission's reliability, by the name the diagrams give it (#179).
    sf = reliability
    #: The mission's unreliability, by the name the diagrams give it.
    ff = unreliability

    def phase_failure_probabilities(
        self,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> Dict[Hashable, float]:
        """The probability that the mission fails in each phase: that the
        earlier phases get through and this one does not. They add up to
        the mission's unreliability.

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``: the fraction of
            ``mc_samples`` simulated missions that fail in each phase.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000.
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        dict[Hashable, float]
            Per phase, by name, the probability of failing in it.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> failing = mission.phase_failure_probabilities()
        >>> {name: round(chance, 5) for name, chance in failing.items()}
        {'out': 0.09516, 'back': 0.08611}
        """
        method = self._method(method, mc_samples)
        names = [phase.name for phase in self.phases]
        if method == "simulate":
            failed = self._simulated(
                10_000 if mc_samples is None else mc_samples, seed
            )
            counts = np.bincount(failed, minlength=len(names) + 1)
            return {
                name: float(counts[j] / len(failed))
                for j, name in enumerate(names)
            }
        out: Dict[Hashable, float] = {}
        before = 0.0
        for j, name in enumerate(names):
            _, failure = self._exact(j + 1)
            out[name] = max(failure - before, 0.0)
            before = failure
        return out

    def reliability_interval(
        self,
        mc_samples: int = 10_000,
        seed=None,
        *,
        confidence: float = 0.95,
        tolerance: Optional[float] = None,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
    ) -> ConfidenceInterval:
        """The mission's reliability by simulation, with a confidence
        interval: each component's life drawn once per mission.

        Parameters
        ----------
        mc_samples : int, optional
            The number of simulated missions, by default 10_000; with a
            ``tolerance``, the size of each batch.
        seed : int, optional
            Seeds the simulation, by default None.
        confidence : float, optional
            The interval's confidence level, by default 0.95.
        tolerance : float, optional
            Simulate until the interval is at most ``tolerance`` either side
            of the estimate, in batches of ``mc_samples``, or until
            ``max_samples`` missions (then with a RuntimeWarning). By
            default None: ``mc_samples`` missions.
        max_samples : int, optional
            The most missions a run to ``tolerance`` simulates, by default
            ``100 * mc_samples``.
        antithetic : bool, optional
            Draw the missions in antithetic pairs, the second of each from
            ``1 - u`` for the first's uniforms ``u``, by default False
            (``mc_samples`` must then be even).

        Returns
        -------
        ConfidenceInterval
            The estimate, its standard error and a normal interval, clipped
            to ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``mc_samples``, ``confidence``, ``tolerance`` or
            ``max_samples`` is invalid.
        NotImplementedError
            If ``antithetic`` is asked for and a component's lives cannot
            be drawn from uniforms.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> interval = mission.reliability_interval(20_000, seed=1)
        >>> bool(interval.lower < mission.reliability() < interval.upper)
        True
        """
        failed = self._simulated(
            mc_samples, seed, antithetic, tolerance, confidence, max_samples
        )
        succeeded = (failed == len(self.phases)).astype(float)
        estimate = float(succeeded.mean())
        error = montecarlo.standard_error(succeeded, antithetic)
        half = montecarlo.z_value(confidence) * error
        return ConfidenceInterval(
            estimate=estimate,
            lower=max(estimate - half, 0.0),
            upper=min(estimate + half, 1.0),
            confidence=confidence,
            standard_error=error,
            n_samples=len(succeeded),
        )

    def birnbaum_importance(self) -> Dict[Hashable, float]:
        """Birnbaum importance of each component over the mission.

        ``B_i = R(i works) - R(i failed)``: the mission's reliability with
        component ``i`` working throughout (every phase's segment of its
        life survived) less its reliability with ``i`` failed from the
        start (failed in the first phase, so in every phase), the
        probability that ``i`` is critical to the mission. Each is the
        mission's decision diagram replayed with the component's segments
        so held, and the difference is taken from the probabilities of
        succeeding or of failing, whichever are the smaller, so a small
        importance in a reliable mission keeps its precision. Exact, at
        the mission's end; it assumes independent components.

        Returns
        -------
        dict[Hashable, float]
            ``{component: importance}`` over the mission's components.

        Raises
        ------
        NotImplementedError
            If the mission's decision diagram needs more than ``MAX_NODES``
            nodes.

        Examples
        --------
        Two engines, both needed to take off and either to cruise: an
        engine failed from the start fails the take-off, and one working
        throughout carries the mission unless the other fails in the
        take-off, so each is critical with the probability that the other
        lasts the take-off, ``exp(-0.01 * 0.1)``.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> engine = surv.Exponential.from_params([0.01])
        >>> both = NonRepairableRBD(
        ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> either = NonRepairableRBD(
        ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> mission = PhasedMission(
        ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
        ... )
        >>> {k: round(v, 6) for k, v in mission.birnbaum_importance().items()}
        {'e1': 0.999, 'e2': 0.999}
        """
        return {
            component: birnbaum
            for component, (birnbaum, _, _) in self._measures().items()
        }

    def improvement_potential(self) -> Dict[Hashable, float]:
        """Improvement potential of each component over the mission.

        ``IP_i = R(i works) - R``: how much the mission's reliability would
        rise if component ``i`` worked throughout (every phase's segment
        of its life survived). Each is the mission's decision diagram
        replayed with the component's segments so held (see
        [`birnbaum_importance`][repyability.PhasedMission.birnbaum_importance]),
        the difference taken from the probabilities of succeeding or of
        failing, whichever are the smaller. Exact, at the mission's end; it
        assumes independent components.

        Returns
        -------
        dict[Hashable, float]
            ``{component: improvement potential}`` over the mission's
            components.

        Raises
        ------
        NotImplementedError
            If the mission's decision diagram needs more than ``MAX_NODES``
            nodes.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> engine = surv.Exponential.from_params([0.01])
        >>> both = NonRepairableRBD(
        ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> either = NonRepairableRBD(
        ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> mission = PhasedMission(
        ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
        ... )
        >>> potential = mission.improvement_potential()
        >>> {k: round(v, 6) for k, v in potential.items()}
        {'e1': 0.003372, 'e2': 0.003372}
        """
        return {
            component: potential
            for component, (_, potential, _) in self._measures().items()
        }

    def risk_achievement_worth(self) -> Dict[Hashable, float]:
        """Risk achievement worth (RAW) of each component over the mission.

        ``RAW_i = F(i failed) / F``, per Modarres & Kaminskiy: the factor by
        which the mission's unreliability would grow if component ``i``
        were failed from the start (failed in the first phase, so in every
        phase). It is at least 1. Each unreliability is the mission's
        decision diagram replayed with the component's segments so held
        (see
        [`birnbaum_importance`][repyability.PhasedMission.birnbaum_importance]),
        in its own right (not as one less a reliability), so a small one
        keeps its precision; where the mission cannot fail the division
        gives ``inf`` or ``nan``. Exact, at the mission's end; it assumes
        independent components.

        Returns
        -------
        dict[Hashable, float]
            ``{component: RAW}`` over the mission's components.

        Raises
        ------
        NotImplementedError
            If the mission's decision diagram needs more than ``MAX_NODES``
            nodes.

        Examples
        --------
        Either engine failed from the start fails the take-off, so the
        mission: its unreliability becomes 1.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> engine = surv.Exponential.from_params([0.01])
        >>> both = NonRepairableRBD(
        ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> either = NonRepairableRBD(
        ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> mission = PhasedMission(
        ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
        ... )
        >>> raw = mission.risk_achievement_worth()
        >>> {k: round(v, 2) for k, v in raw.items()}
        {'e1': 228.74, 'e2': 228.74}
        >>> round(1 / mission.unreliability(), 2)
        228.74
        """
        _, failure = self._exact(len(self.phases))
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                component: float(np.float64(f_failed) / failure)
                for component, (_, _, _, f_failed) in self._held().items()
            }

    def risk_reduction_worth(self) -> Dict[Hashable, float]:
        """Risk reduction worth (RRW) of each component over the mission.

        ``RRW_i = F / F(i works)``, per Modarres & Kaminskiy: the factor by
        which the mission's unreliability would shrink if component ``i``
        worked throughout (every phase's segment of its life survived). It
        is at least 1, and ``inf`` where the component working throughout
        makes the mission certain. Each unreliability is the mission's
        decision diagram replayed with the component's segments so held
        (see
        [`birnbaum_importance`][repyability.PhasedMission.birnbaum_importance]),
        in its own right, so a small one keeps its precision. Exact, at the
        mission's end; it assumes independent components.

        Returns
        -------
        dict[Hashable, float]
            ``{component: RRW}`` over the mission's components.

        Raises
        ------
        NotImplementedError
            If the mission's decision diagram needs more than ``MAX_NODES``
            nodes.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> engine = surv.Exponential.from_params([0.01])
        >>> both = NonRepairableRBD(
        ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> either = NonRepairableRBD(
        ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> mission = PhasedMission(
        ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
        ... )
        >>> rrw = mission.risk_reduction_worth()
        >>> {k: round(v, 4) for k, v in rrw.items()}
        {'e1': 4.374, 'e2': 4.374}
        """
        _, failure = self._exact(len(self.phases))
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                component: float(np.float64(failure) / f_works)
                for component, (_, f_works, _, _) in self._held().items()
            }

    def criticality_importance(
        self, kind: str = "failure"
    ) -> Dict[Hashable, float]:
        """Criticality importance of each component over the mission.

        With ``R`` and ``F`` the mission's reliability and unreliability,
        and ``R(i works)`` and ``R(i failed)`` its reliability with
        component ``i`` working throughout and failed from the start (see
        [`birnbaum_importance`][repyability.PhasedMission.birnbaum_importance]):

        - ``kind="failure"`` (the default) gives the failure-oriented form
          (Rausand & Høyland), ``CI_i = (R(i works) - R) / F``: the
          probability, given that the mission has failed, that it would
          have succeeded with component ``i`` working throughout; the
          share of the mission's failures that ``i`` accounts for. For a
          diagram this is ``B_i * (1 - R_i) / (1 - R_sys)``, as
          [`criticality_importance`][repyability.NonRepairableRBD.criticality_importance]
          gives it. It is ``nan`` where the mission cannot fail.
        - ``kind="success"`` gives the success-oriented form,
          ``CI_i = (R - R(i failed)) / R``: the probability, given that the
          mission succeeds, that it would have failed with component ``i``
          failed from the start (for a diagram, ``B_i * R_i / R_sys``). It
          is ``nan`` where the mission cannot succeed.

        Each difference is taken from the probabilities of succeeding or
        of failing, whichever are the smaller, so a small share keeps its
        precision. Exact, at the mission's end; it assumes independent
        components.

        Parameters
        ----------
        kind : str, optional
            ``"failure"`` (the default) or ``"success"``.

        Returns
        -------
        dict[Hashable, float]
            ``{component: criticality importance}`` over the mission's
            components.

        Raises
        ------
        ValueError
            If ``kind`` is neither ``"failure"`` nor ``"success"``.
        NotImplementedError
            If the mission's decision diagram needs more than ``MAX_NODES``
            nodes.

        Examples
        --------
        Two engines, both needed to take off and either to cruise: the
        mission's failures without either engine's own failure are the
        other's in the take-off, so each engine accounts for a share
        ``1 - 1 / RRW_i`` of the failures.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> engine = surv.Exponential.from_params([0.01])
        >>> both = NonRepairableRBD(
        ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> either = NonRepairableRBD(
        ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> mission = PhasedMission(
        ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
        ... )
        >>> ci = mission.criticality_importance()
        >>> {k: round(v, 4) for k, v in ci.items()}
        {'e1': 0.7714, 'e2': 0.7714}
        >>> round(1 - 1 / mission.risk_reduction_worth()["e1"], 4)
        0.7714
        """
        one_of("kind", kind, ("failure", "success"))
        success, failure = self._exact(len(self.phases))
        mission, which = (failure, 1) if kind == "failure" else (success, 2)
        return {
            component: (
                float(measures[which] / mission) if mission > 0.0 else math.nan
            )
            for component, measures in self._measures().items()
        }

    def fussell_vesely(self, method: str = "exact") -> Dict[Hashable, float]:
        """Fussell-Vesely importance of each component over the mission:
        the share of the mission's failure probability that involves the
        component, through the mission's minimal cut sets (see
        [`minimal_cut_sets`][repyability.PhasedMission.minimal_cut_sets]),
        sets of events "component ``c`` has failed by the end of phase
        ``j``", each with probability ``F_c(T_j)``, that together fail the
        mission. A cut set has occurred when every one of its events has,
        and a component's are the cut sets naming it in any phase:

            FV_i = P(some minimal cut set naming i has occurred) / F

        with ``F`` the mission's unreliability. ``method="exact"`` (the
        default) works out the probability of that union exactly, from a
        decision diagram of the cut sets' events over the components'
        segments (an event being the component's first ``j + 1`` segments
        not all survived), so it is between 0 and 1.
        ``method="rare_event"`` gives the rare-event approximation that
        many PRA tools report, summing the cut sets' probabilities, each
        the product of its events', instead of taking their union:

            FV_i = sum over minimal cut sets C naming i of
                   prod over (c, j) in C of F_c(T_j), divided by F

        The sum over-estimates the union, so it can exceed 1 when the
        failure probabilities are not small. ``F`` and each event's
        probability are worked out in their own right (the mission's as
        [`unreliability`][repyability.PhasedMission.unreliability] is, a
        component's from its model's ``ff``), and so is the union, so
        small ones keep their precision. Exact; it assumes independent
        components.

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default): the probability of the union of the
            cut sets' occurrences; or ``"rare_event"``: the sum of their
            probabilities.

        Returns
        -------
        dict[Hashable, float]
            ``{component: Fussell-Vesely importance}`` over the mission's
            components.

        Raises
        ------
        ValueError
            If ``method`` is not ``"exact"`` or ``"rare_event"``.
        NotImplementedError
            If the mission's decision diagram, or with ``method="exact"``
            the cut sets', needs more than ``MAX_NODES`` nodes.

        Examples
        --------
        Two engines, both needed to take off and either to cruise: an
        engine's share is the failures through its own loss in the
        take-off or both engines' by the end of the cruise, not through the
        other's loss in the take-off alone. The rare-event sum counts a
        failure that is both at once twice.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> engine = surv.Exponential.from_params([0.01])
        >>> both = NonRepairableRBD(
        ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> either = NonRepairableRBD(
        ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> mission = PhasedMission(
        ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
        ... )
        >>> fv = mission.fussell_vesely()
        >>> {k: round(v, 4) for k, v in fv.items()}
        {'e1': 0.7827, 'e2': 0.7827}
        >>> fv = mission.fussell_vesely(method="rare_event")
        >>> {k: round(v, 4) for k, v in fv.items()}
        {'e1': 0.7941, 'e2': 0.7941}
        """
        _, failure = self._exact(len(self.phases))
        numerators = self._fv_numerators(method)
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                component: float(np.float64(numerator) / failure)
                for component, numerator in numerators.items()
            }

    def minimal_cut_sets(self) -> List[frozenset]:
        """The mission's minimal cut sets: sets of events ``(component,
        phase name)``, "the component has failed by the end of that
        phase", that together fail the mission, smallest first (then by
        name).

        A failure by one phase is one by every later phase, so an event is
        implied by the same component's at an earlier phase. A cut set is
        minimal when no event can be dropped and none replaced by a weaker
        one, the component's failure by a later phase. They are the
        phases' own minimal cut sets (each a phase's diagram's, its
        components' failures dated to that phase), less those another
        implies: a set is dropped when every event of another follows from
        one of its own (the other names no component it lacks, each at
        that phase or a later one), the usual phased-mission minimisation.
        A phase whose diagram can never work has the empty cut set, which
        implies every other. They are worked out on first use and kept.

        Returns
        -------
        list of frozenset
            The minimal cut sets, each a frozenset of ``(component, phase
            name)`` events.

        Raises
        ------
        NotImplementedError
            If a phase's diagram is too meshed for its cut sets.

        Examples
        --------
        Two engines, both needed to take off and either to cruise: an
        engine failed by the end of the take-off fails the mission, as do
        both failed by the end of the cruise.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> engine = surv.Exponential.from_params([0.01])
        >>> both = NonRepairableRBD(
        ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> either = NonRepairableRBD(
        ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
        ...     {"e1": engine, "e2": engine},
        ... )
        >>> mission = PhasedMission(
        ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
        ... )
        >>> for cut in mission.minimal_cut_sets():
        ...     print(sorted(cut))
        [('e1', 'take-off')]
        [('e2', 'take-off')]
        [('e1', 'cruise'), ('e2', 'cruise')]
        """
        if self._cut_sets is None:
            dated: List[Dict[Hashable, int]] = []
            for j, phase in enumerate(self.phases):
                try:
                    cuts = phase.rbd.get_min_cut_sets()
                except ValueError:
                    cuts = {frozenset()}  # no set of working nodes works
                dated.extend(dict.fromkeys(cut, j) for cut in cuts)
            kept = [
                cut
                for cut in dated
                if not any(_weaker(other, cut) for other in dated)
            ]
            self._cut_sets = sorted(
                (
                    frozenset(
                        (node, self.phases[j].name) for node, j in cut.items()
                    )
                    for cut in kept
                ),
                key=lambda cut: (len(cut), sorted(map(str, cut))),
            )
        return list(self._cut_sets)

    def analysis_routes(self) -> Dict[str, routes.AnalysisRoute]:
        """How each analysis of the mission is computed, found without
        running it: exactly, by simulation, or not at all.

        The exact analyses replay the phases' decision diagram over the
        components' segments, and refuse when it would need more than
        ``MAX_NODES`` nodes (the diagram is built, and kept for the
        analyses; the Fussell-Vesely importance's cut sets' diagram too).
        The refusal is the message the method would raise.

        Returns
        -------
        dict[str, AnalysisRoute]
            For each public analysis, by method name: its ``route``
            (``"exact"``, ``"simulated"`` or ``"refused"``) and the
            ``reason`` (see [`AnalysisRoute`][repyability.AnalysisRoute]).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> routes = mission.analysis_routes()
        >>> routes["reliability"].route, routes["reliability_interval"].route
        ('exact', 'simulated')
        """
        r = routes
        refused = r.refusal(lambda: self._plan(len(self.phases)))
        diagram = (
            r.refused(refused)
            if refused
            else r.AnalysisRoute(
                r.EXACT,
                "The phases' decision diagram over the components' "
                "segments (Esary and Ziehms), replayed.",
            )
        )
        out = {
            name: diagram
            for name in (
                "reliability",
                "unreliability",
                "sf",
                "ff",
                "phase_failure_probabilities",
                "birnbaum_importance",
                "improvement_potential",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "criticality_importance",
            )
        }
        cuts = refused or r.refusal(self._cut_set_plan)
        out["fussell_vesely"] = (
            r.refused(cuts)
            if cuts
            else r.AnalysisRoute(
                r.EXACT,
                "The mission's minimal cut sets, their occurrence a "
                "decision diagram over the components' segments.",
            )
        )
        out["reliability_interval"] = r.AnalysisRoute(
            r.SIMULATED, "Each component's life drawn once per mission."
        )
        return out

    def _method(self, method: str, mc_samples) -> str:
        one_of("method", method, ("exact", "simulate"))
        if method == "exact" and mc_samples is not None:
            raise ValueError("mc_samples applies only to method='simulate'.")
        return method

    def _outcome(self, method: str, mc_samples, seed) -> Tuple[float, float]:
        """The mission's reliability and unreliability."""
        method = self._method(method, mc_samples)
        if method == "simulate":
            failed = self._simulated(
                10_000 if mc_samples is None else mc_samples, seed
            )
            success = float(np.mean(failed == len(self.phases)))
            return success, 1.0 - success
        return self._exact(len(self.phases))

    def __repr__(self) -> str:
        phases = ", ".join(
            f"{phase.name!r} ({phase.duration:g})" for phase in self.phases
        )
        return f"PhasedMission({phases})"
