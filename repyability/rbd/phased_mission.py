"""Phased missions (#100): a mission that runs through phases (take-off,
cruise, landing), each with its own duration and its own diagram over the
same non-repairable components.

A component that fails in one phase stays failed in the later ones, so the
phases are not independent, and the mission's reliability is not the
product of theirs. The mission succeeds if each phase's diagram works
throughout the phase; as the components only ever fail, that is its
working at the phase's end, ``T_j``. (Blocks repaired during the phases,
``repairability``, can fail and come back, so the mission then fails at
the first instant a phase's diagram is down: see ``_mission_repairs``.)

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

import dataclasses
import json
import math
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability._version import __version__
from repyability.rbd import (
    _measures,
    _mission_ccf,
    _mission_chain,
    _mission_repairs,
    _mission_runs,
)
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import bdd, modular, routes
from repyability.rbd._differential import check as check_differential
from repyability.rbd._differential import shares
from repyability.rbd._importances import with_importances
from repyability.rbd._mission_results import (
    MissionDrivers,
    MissionPhaseSummary,
    MissionSummary,
    wilson_band,
)
from repyability.rbd._mission_settings import (
    BlockSetting,
    Life,
    MissionPhase,
    behaviour,
    life_of,
    setting_from_dict,
    setting_to_dict,
)
from repyability.rbd._ordered_bdd import FALSE, TRUE, OrderedBDD
from repyability.rbd._units import diagram_unit
from repyability.rbd.ccf import as_groups
from repyability.rbd.rbd import Pairs
from repyability.rbd.results import ConfidenceInterval, ImportanceTable
from repyability.rbd.shannon import (
    _plan_rows,
    replay,
    room,
    value_and_gradient,
)
from repyability.utils.checks import (
    is_number,
    one_of,
    real_array,
    simulation_options,
)
from repyability.utils.deprecation import refuse_removed_names
from repyability.utils.json_io import read_json, write_json
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

if TYPE_CHECKING:
    from repyability.rbd.results import Lever

# The settings a member of a common-cause group may have in a phase.
_PLAIN = (BlockSetting(), BlockSetting(state="not_needed"))

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
    built from its diagram's modules and core (see ``modular.py``, the
    junctions folded in as always working), each component working when
    its first ``j + 1`` segments are survived. Component ``c``'s ``k``-th
    segment is variable ``rank[c] * count + k``, so a component's segments
    are decided together, in phase order."""
    try:
        decomposition = rbd._set_structure()
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
    of a phase, that together fail the mission. There is no structural
    importance: a component's state is one per phase segment, so "the
    share of the state vectors in which it is critical" has no single
    meaning for a mission. Nor is there a ``mean``, as a mission that
    succeeds has no time to failure: ``expected_mission_time`` is the
    mean time it runs, to its failure or its end. ``ccf_groups`` gives the
    mission common-cause groups, each shared cause a component of its own
    that fails its members from the phase it strikes in, and
    ``repairability`` the blocks repaired while the mission runs.
    ``levers()`` lists the
    components' parameters and the phases' durations, which
    ``parameter_sensitivity`` differences and ``with_levers`` moves, and
    ``to_dict`` and ``to_json`` save the mission (``from_dict`` and
    ``from_json`` load it).

    Parameters
    ----------
    phases : Sequence[tuple[Hashable, float, NonRepairableRBD]]
        The phases in order, each a ``(name, duration, rbd)``: a distinct
        name, a duration (finite, and not negative: a phase of no duration
        needs its diagram working at that instant), and the phase's
        diagram, whose node names are the mission's components.
    ccf_groups : CCFGroup or list of CCFGroup, optional
        Common-cause groups over the mission's components: a shared cause
        that strikes in one phase fails its members from then on. A
        group's model has a life for each cause: one that splits the
        failure rate (``BetaFactor(beta, basis="rate")``, an ``MGL`` or
        ``AlphaFactor`` so), whose members carry one surpyval parametric
        model, or ``Shocks`` at a rate of their own, for members of any
        models. Each shared cause is then a component of the mission,
        named ``("cause", members it fails)`` (``"cause 1"``, ``"cause
        2"``, ... with several groups), and a member's own entry is that
        of its own failures. See the [phased missions
        guide](../guide/phased-missions.md).
    repairability : dict, optional
        The blocks repaired during the phases: each one's model of its
        repair times, by name. Such a block is under repair from the
        moment it fails and then runs again, as good as new, whatever
        the phase by then. The mission then fails at the first instant
        the diagram of the phase under way is down, and at a phase's
        start if its diagram is down then. Exact where every life has a
        constant failure rate and every repair is exponential (a Markov
        chain through the phases), and simulated otherwise. A repaired
        block is in no common-cause group. See the [phased missions
        guide](../guide/phased-missions.md).

    Attributes
    ----------
    phases : list
        The phases, each with its ``name``, ``duration``, ``rbd``, ``end``
        (from the mission's start) and the ``components`` it uses.
    components : dict
        Each component's model, by node name, in order of first use: with
        common-cause groups, a member's model of its own failures, and
        each shared cause's.
    ccf_groups : list
        The common-cause groups.
    repairability : dict
        Each repaired block's model of its repair times, by name.
    duration : float
        The mission's length: the sum of the phases' durations.

    Raises
    ------
    ValueError
        If ``phases`` is empty, a phase is not a ``(name, duration, rbd)``
        with a distinct name, a duration that is finite and not negative
        and a ``NonRepairableRBD``, a phase's diagram has common-cause
        groups of its own, a node name has different models in different
        phases, or a common-cause group is not one of the mission's (a
        member that is not a component or is in two groups, members of
        different models, a model that splits a failure probability, or
        members' models that take no hazard ratio), or ``repairability``
        is not a dict of the mission's components' repair models (or
        names a member of a group, or a block whose life is a
        probability of failing).
    TypeError
        If a repair model is not a model of a time, or is a surpyval
        distribution itself.

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

    def __init__(
        self,
        phases: Sequence[Tuple[Hashable, float, Any]],
        ccf_groups: Any = None,
        repairability: Optional[Mapping[Hashable, Any]] = None,
    ):
        phases = list(phases)
        if not phases:
            raise ValueError("A mission needs at least one phase.")
        groups = as_groups(ccf_groups)
        # As given, for ``with_levers`` (see ``_from_args``).
        self._init_args: Dict[str, Any] = {
            "phases": phases,
            "ccf_groups": groups,
            "repairability": repairability,
        }
        self.components, self.phases, end = self._read(phases)
        # The components as given: with common-cause groups, the members
        # with their whole lives, and no cause.
        self._given: Dict[Hashable, Any] = self.components
        self.ccf_groups: list = _mission_ccf.checked(
            groups, self._given, phases
        )
        if self.ccf_groups:
            # Each shared cause a component of its own, after each member
            # it fails in every phase's diagram (see ``_mission_ccf``).
            self.components, self.phases, end = self._read(
                _mission_ccf.written(phases, self.ccf_groups, self._given)
            )
        self.duration = end
        # The blocks repaired during the phases, each with its model of
        # its repair times (see ``_mission_repairs``).
        self.repairability: Dict[Hashable, Any] = _mission_repairs.checked(
            repairability, self
        )
        # The unit the components' models (and so the phases' durations)
        # are in, None if they name none; a warning names the components
        # when they name different units.
        self.unit: Optional[str] = diagram_unit(self.components, "mission")
        # Each phase's settings of its blocks (none here: every block
        # working; see ``from_diagram``), and each component's life on its
        # running clock.
        self._settings: List[MissionPhase] = [
            MissionPhase(phase.name, phase.duration) for phase in self.phases
        ]
        self._lives: Dict[Hashable, Life] = {
            node: Life(model) for node, model in self.components.items()
        }
        # The decomposition of the first so many phases, once worked out.
        self._plans: Dict[int, tuple] = {}
        # The minimal cut sets, and the plan of their occurrence by
        # component (see ``_cut_set_plan``), once worked out.
        self._cut_sets: Optional[List[frozenset]] = None
        self._cut_plan: Optional[Tuple[tuple, Dict[Hashable, int]]] = None
        # The values from the Markov chain of a mission with blocks on
        # standby, once worked out.
        self._chain_kept: Optional[_mission_chain.ChainValues] = None

    @staticmethod
    def _read(
        phases: Sequence[Any],
    ) -> Tuple[Dict[Hashable, Any], List[_Phase], float]:
        """The mission's components (each one's model, by name, in order
        of first use), its phases and its duration, from the ``phases``
        given, each checked."""
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD

        components: Dict[Hashable, Any] = {}
        read: List[_Phase] = []
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
                    f"Phase {name!r} has common-cause groups, which a phase "
                    "cannot hold on its own, as a cause that strikes in one "
                    "phase fails its members in the later ones too: give "
                    "the groups to the mission (ccf_groups=...)."
                )
            # A junction (a PerfectReliability node, such as a vote point)
            # is no component: the diagram folds it out of its structure.
            left_out = {rbd.input_node, rbd.output_node} | rbd._junctions()
            used = tuple(n for n in rbd._components() if n not in left_out)
            for node in used:
                model = rbd.reliabilities[node]
                known = components.setdefault(node, model)
                if not NonRepairableRBD._same_model(known, model):
                    raise ValueError(
                        f"Node {node!r} has a different model in phase "
                        f"{name!r} than before: a component keeps one "
                        "lifetime model through the mission."
                    )
            end += float(duration)
            read.append(_Phase(name, float(duration), rbd, end, used))
        return components, read, end

    @classmethod
    def from_diagram(
        cls,
        rbd: Any,
        phases: Sequence[MissionPhase],
        ccf_groups: Any = None,
        repairability: Optional[Mapping[Hashable, Any]] = None,
    ) -> "PhasedMission":
        """A mission drawn as one diagram, run through phases that each
        set how its blocks behave: one block on standby, another unable to
        fail, a third not needed, a block on a life model of the phase's
        own, a regression block at the phase's covariates, a vote needing
        fewer inputs.

        Each phase is a [`MissionPhase`][repyability.MissionPhase]: a
        name, a duration, and for some blocks a state (``"working"``, the
        default; ``"standby"``; ``"guaranteed"``; ``"not_needed"``) or a
        [`BlockSetting`][repyability.BlockSetting] with the block's life
        model or covariates in the phase, and for some k-out-of-n nodes
        the number of inputs the phase needs. Every block has one running clock
        through the mission and one budget of hazard: it ages while it
        runs, spends hazard while it runs and can fail, and fails once it
        has spent its budget. Working, it runs; guaranteed, it runs but
        spends nothing; on standby it is idle (neither ageing nor
        spending) until the phase's structure needs it, then runs to the
        phase's end; not needed, the structure takes it as working while
        it runs on. Given a model in a phase, it runs on that model
        through the phase, its age carried in and out. The [phased missions
        guide](../guide/phased-missions.md) has the rule for starting
        blocks on standby.

        A mission with no block on standby is exact, as one built from a
        diagram per phase is (each block's segments survived with its
        life's probabilities, 1 in a phase it is guaranteed). One with a
        block on standby is exact too where every block's life has a
        constant failure rate in every phase (a Markov chain over the
        blocks' states), and simulated otherwise. ``analysis_routes()``
        says which.

        Parameters
        ----------
        rbd : NonRepairableRBD
            The diagram, the same in every phase but for the settings.
        phases : Sequence[MissionPhase]
            The phases in order, with distinct names.
        ccf_groups : CCFGroup or list of CCFGroup, optional
            Common-cause groups over the diagram's blocks, as the
            constructor takes them. A member is working or not needed in
            every phase, on its node's model.
        repairability : dict, optional
            The blocks repaired during the phases, as the constructor
            takes them. A repaired block is working or not needed in
            every phase, on its node's model, and a mission with repairs
            takes no block on standby.

        Returns
        -------
        PhasedMission
            The mission, with a phase per ``MissionPhase``, each with the
            diagram as the phase sees it (a block not needed taken as
            working, the phase's vote counts).

        Raises
        ------
        ValueError
            If ``rbd`` is not a ``NonRepairableRBD`` or has common-cause
            groups of its own, a group is not one the mission takes (see
            the class) or a phase sets a member, ``repairability`` is not
            as the class takes it, or comes with a block on standby or a
            repaired block that a phase sets, there is no phase, a
            phase is not a ``MissionPhase``,
            two have one name, a phase names a block the diagram does not
            have (or a repeated node, the input or output node, or a
            junction), a vote count for a node not in the diagram or that
            the diagram refuses, a phase's model the block's node could
            not have been given (or for a block whose model is a
            probability of failing), or covariates for a block whose model
            is not a ``RegressionNode`` with fixed covariates, or of the
            wrong length.
        TypeError
            If a phase's model is not a model the node could have been
            given (as the diagram's own check says).

        Examples
        --------
        Two exponential units in series, the second unable to fail in the
        first phase: ``R = exp(-0.01 * 30) * exp(-0.02 * 20)``.

        >>> import math
        >>> import surpyval as surv
        >>> from repyability import MissionPhase, NonRepairableRBD
        >>> from repyability import PhasedMission
        >>> series = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": surv.Exponential.from_params([0.01]),
        ...         "b": surv.Exponential.from_params([0.02]),
        ...     },
        ... )
        >>> mission = PhasedMission.from_diagram(
        ...     series,
        ...     [
        ...         MissionPhase("first", 10.0, blocks={"b": "guaranteed"}),
        ...         MissionPhase("second", 20.0),
        ...     ],
        ... )
        >>> math.isclose(mission.reliability(), math.exp(-0.3 - 0.4))
        True
        """
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD

        if not isinstance(rbd, NonRepairableRBD):
            raise ValueError(
                "The diagram must be a NonRepairableRBD, got "
                f"{type(rbd).__name__}."
            )
        if rbd.ccf_groups:
            raise ValueError(
                "The diagram has common-cause groups: give them to the "
                "mission (ccf_groups=...), as a cause that strikes in one "
                "phase fails its members in the later ones too."
            )
        phases = list(phases)
        if not phases:
            raise ValueError("A mission needs at least one phase.")
        drawn = set(rbd.G.nodes)
        ends = {rbd.input_node, rbd.output_node}
        blocks = set(rbd._components()) - ends - rbd._junctions()
        for phase in phases:
            if not isinstance(phase, MissionPhase):
                raise ValueError(
                    f"Each phase must be a MissionPhase, got {phase!r}."
                )
            for node in phase.blocks:
                if node in blocks:
                    continue
                if node in rbd.repeated:
                    why = (
                        f"is a repeat of {rbd.repeated[node]!r}: set the "
                        "component it repeats"
                    )
                elif node in ends or node in drawn:
                    why = "is the input or output node, or a junction"
                else:
                    why = "is not a node of the diagram"
                raise ValueError(
                    f"Phase {phase.name!r} sets block {node!r}, which {why}."
                )
            for node in phase.k:
                if node not in drawn:
                    raise ValueError(
                        f"Phase {phase.name!r} sets the vote count of "
                        f"{node!r}, which is not a node of the diagram."
                    )
        groups = as_groups(ccf_groups)
        for group in groups:
            for phase in phases:
                for node in getattr(group, "members", ()):
                    setting = phase.blocks.get(node)
                    if setting is not None and setting not in _PLAIN:
                        raise ValueError(
                            f"Phase {phase.name!r} sets block {node!r}, a "
                            "member of a common-cause group, whose life is "
                            "split between its causes: a member is working "
                            "or not needed in a phase, on its node's model."
                        )
        mission = cls(
            [
                (phase.name, phase.duration, cls._as_seen(rbd, phase))
                for phase in phases
            ],
            ccf_groups=groups,
            repairability=repairability,
        )
        _mission_repairs.check_settings(mission.repairability, phases)
        mission._settings = phases
        mission._init_args = {
            "rbd": rbd,
            "phases": phases,
            "ccf_groups": groups,
            "repairability": repairability,
        }
        starts = [ph.end - ph.duration for ph in mission.phases]
        mission._lives = {
            node: life_of(node, model, phases, starts)
            for node, model in mission.components.items()
        }
        return mission

    @staticmethod
    def _as_seen(rbd: Any, phase: MissionPhase) -> Any:
        """The diagram as ``phase`` sees it: each block it does not need a
        junction (always working), and its vote counts; the diagram itself
        if it changes neither."""
        from repyability.rbd.helper_classes import PerfectReliability
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD

        dropped = [
            node
            for node, setting in phase.blocks.items()
            if not behaviour(setting).in_structure
        ]
        if not dropped and not phase.k:
            return rbd
        given = rbd._init_args
        reliabilities = dict(given["reliabilities"])
        for node in dropped:
            reliabilities[node] = PerfectReliability
        try:
            return NonRepairableRBD(
                given["edges"],
                reliabilities,
                k={**(given["k"] or {}), **phase.k},
                input_node=given["input_node"],
                output_node=given["output_node"],
                on_infeasible_rbd=given["on_infeasible_rbd"],
            )
        except ValueError as error:
            raise ValueError(f"Phase {phase.name!r}: {error}") from error

    # ------------------------------------------------------------------
    # Exact
    # ------------------------------------------------------------------

    def _guaranteed(self, node: Hashable) -> List[bool]:
        """Whether block ``node`` is guaranteed (spends no hazard) in each
        phase."""
        return [not behaviour(p.setting(node)).spends for p in self._settings]

    def _standby(self) -> List[Hashable]:
        """The components on standby in some phase."""
        return [
            node
            for node in self.components
            if any(behaviour(p.setting(node)).standby for p in self._settings)
        ]

    def _chained(self) -> str:
        """What makes the mission's exact values a Markov chain's
        (``_mission_chain``) rather than the decision diagram's, as a
        refusal names it: its blocks on standby, or those repaired during
        its phases; empty if neither."""
        for found, what in (
            (self._standby(), "on standby"),
            (list(self.repairability), "repaired during its phases"),
        ):
            if found:
                names = ", ".join(repr(node) for node in found)
                return f"blocks {what} ({names})"
        return ""

    def _require_exact(self, what: Optional[str] = None) -> None:
        """Refuse an exact analysis of a mission with blocks on standby
        or repaired
        that its Markov chain cannot work out (``_mission_chain``: a life
        with no constant rate, too many states), whose take-overs are
        then simulated; and, whatever the lives, an analysis of such a
        mission that the chain does not give (``what``: the importance
        measures and cut sets, which then name the one to use instead)."""
        chained = self._chained()
        if not chained:
            return
        if what is None:
            self._chain_values()
            return
        raise NotImplementedError(
            f"The {what} of a mission with {chained} is "
            "not worked out: failure_drivers gives each block's part in "
            "the mission's failures (exact where the mission is, else "
            "simulated)."
        )

    def _require_measures(self, what: str) -> None:
        """Refuse the measures ``what`` of a mission with blocks on
        standby whose Markov chain cannot be worked out (they are taken
        from the chain, a block held working or failed in it), naming
        ``failure_drivers``, which simulates each block's part."""
        chained = self._chained()
        if not chained:
            return
        try:
            self._chain_values()
        except _mission_chain.ChainRefused as error:
            raise NotImplementedError(
                f"The {what} of a mission with {chained} "
                "are worked out from its Markov chain, as its reliability "
                f"is: {error.reason}. failure_drivers gives each block's "
                "part in the mission's failures, simulated."
            ) from error

    def _chain_values(
        self,
        held: Optional[Dict[Hashable, bool]] = None,
        times: Optional[np.ndarray] = None,
    ) -> "_mission_chain.ChainValues":
        """The values of a mission with blocks on standby from its Markov
        chain (see ``_mission_chain.run``: each block ``held`` working,
        True, or failed from the start, False), the plain ones kept."""
        if held is None and times is None:
            if self._chain_kept is None:
                self._chain_kept = _mission_chain.run(self)
            return self._chain_kept
        return _mission_chain.run(self, held, times)

    def _raw_ends(
        self, ends: Optional[np.ndarray] = None
    ) -> Dict[Hashable, Tuple[np.ndarray, np.ndarray]]:
        """Each component's life's probabilities of working and of having
        failed at the phases' ends (or at ``ends``, as many as the first
        phases), from its model on its running clock, the mission's (each
        in its own right, so a small one keeps its precision)."""
        if ends is None:
            ends = np.array([phase.end for phase in self.phases])
        out: Dict[Hashable, Tuple[np.ndarray, np.ndarray]] = {}
        for node in self.components:
            model = self._lives[node].model
            with np.errstate(all="ignore"):
                sf = np.clip(np.ravel(model.sf(ends)).astype(float), 0, 1)
                ff = np.clip(np.ravel(model.ff(ends)).astype(float), 0, 1)
            out[node] = (sf, ff)
        return out

    @staticmethod
    def _chain(
        sf: np.ndarray, ff: np.ndarray, guaranteed: Sequence[bool]
    ) -> Tuple[List[float], List[float]]:
        """A component's segments' survival probabilities and their
        complements, from its life's at the phases' ends: the ``k``-th
        survived with ``R(T_k) / R(T_{k-1})``, the first from the mission's
        start (so a unit dead on arrival fails in it), and surely in a
        phase in which it is guaranteed (where it ages but cannot fail)."""
        p: List[float] = []
        q: List[float] = []
        before_sf, before_ff = 1.0, 0.0
        for k in range(len(sf)):
            if guaranteed[k]:
                p.append(1.0)
                q.append(0.0)
            elif before_sf > 0.0:
                p.append(min(sf[k] / before_sf, 1.0))
                q.append(min(max(ff[k] - before_ff, 0.0) / before_sf, 1.0))
            else:
                p.append(0.0)
                q.append(1.0)
            before_sf, before_ff = sf[k], ff[k]
        return p, q

    def _at_ends(self) -> Dict[Hashable, Tuple[np.ndarray, np.ndarray]]:
        """Each component's probabilities of working and of having failed
        at the phases' ends: its life's, or where it is guaranteed in some
        phase, through its segments (``F_j = F_{j-1} + R_{j-1} q_j``, so a
        small one keeps its precision)."""
        out = self._raw_ends()
        for node, (sf, ff) in out.items():
            guaranteed = self._guaranteed(node)
            if any(guaranteed):
                p, q = self._chain(sf, ff, guaranteed)
                works, failed = np.empty(len(p)), np.empty(len(p))
                before = 1.0
                for k in range(len(p)):
                    failed[k] = (failed[k - 1] if k else 0.0) + before * q[k]
                    works[k] = before = before * p[k]
                out[node] = (works, failed)
        return out

    def _segments(
        self,
        ends: Optional[np.ndarray] = None,
        held: Optional[Dict[Hashable, float]] = None,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Each component's segments' survival probabilities, and their
        complements, by element number (``component * phases + k``; see
        ``_chain``), at the phases' ends or at ``ends`` (the first
        ``len(ends)`` phases, the last perhaps cut short). A component in
        ``held`` has every segment survived with that probability (1:
        working throughout, 0: failed from the start)."""
        return self._segments_of(self._raw_ends(ends), held)

    def _segments_of(
        self,
        raw: Dict[Hashable, Tuple[np.ndarray, np.ndarray]],
        held: Optional[Dict[Hashable, float]] = None,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """``_segments`` from each component's life's probabilities of
        working and of having failed at the ends, ``raw`` (see
        ``_raw_ends``)."""
        count = len(self.phases)
        p: Dict[int, float] = {}
        q: Dict[int, float] = {}
        held = held or {}
        for index, (node, (sf, ff)) in enumerate(raw.items()):
            chain = (
                ([held[node]] * len(sf), [1.0 - held[node]] * len(sf))
                if node in held
                else self._chain(sf, ff, self._guaranteed(node))
            )
            for k, (works, fails) in enumerate(zip(*chain)):
                p[index * count + k] = works
                q[index * count + k] = fails
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
                decomposition = phase.rbd._set_structure()
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
        self._require_exact()
        if self._chained():
            chain = self._chain_values()
            return chain.through[phases - 1], float(
                sum(chain.failing[:phases])
            )
        return _outcomes(self._plan(phases), *self._segments())

    # ------------------------------------------------------------------
    # Importance and cut sets (#316)
    # ------------------------------------------------------------------

    def _held_outcomes(
        self, holds: Sequence[Dict[Hashable, bool]]
    ) -> List[Tuple[float, float]]:
        """For each of ``holds``, components held working throughout (True:
        every segment survived) or failed from the start (False: none),
        the mission's reliability and unreliability, each to its own
        precision: the plan replayed with their segments so held, or,
        with blocks on standby, the Markov chain run with them so held."""
        if self._chained():
            out = []
            for held in holds:
                chain = self._chain_values(held=held)
                out.append((chain.through[-1], float(sum(chain.failing))))
            return out
        count = len(self.phases)
        plan = self._plan(count)
        p, q = self._segments()
        index = {node: i for i, node in enumerate(self.components)}
        out = []
        for held in holds:
            given, failing = dict(p), dict(q)
            for node, works in held.items():
                segments = range(
                    index[node] * count, (index[node] + 1) * count
                )
                given.update(dict.fromkeys(segments, 1.0 if works else 0.0))
                failing.update(dict.fromkeys(segments, 0.0 if works else 1.0))
            out.append(_outcomes(plan, given, failing))
        return out

    def _held(self) -> Dict[Hashable, Tuple[float, float, float, float]]:
        """For each component, the mission's reliability and unreliability
        with it working throughout (every segment survived), then with it
        failed from the start (none); see ``_held_outcomes``."""
        components = list(self.components)
        values = self._held_outcomes(
            [{node: works} for node in components for works in (True, False)]
        )
        return {
            node: (*values[2 * k], *values[2 * k + 1])
            for k, node in enumerate(components)
        }

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
        return self._runs(size, antithetic).phase

    def _runs(
        self, size: int, antithetic: bool = False
    ) -> "_mission_runs.Runs":
        """``size`` simulated missions (see ``_mission_runs``)."""
        lives = _mission_runs.draw(self, size, antithetic)
        if self.repairability:
            source = _mission_repairs.Source(self, size, antithetic)
            return _mission_repairs.play(self, lives, size, source)
        switches = _mission_runs.draw_switches(self, size, antithetic)
        return _mission_runs.play(self, lives, size, switches=switches)

    def _simulated(
        self,
        mc_samples: int,
        seed,
        antithetic: bool = False,
        tolerance=None,
        confidence: float = 0.95,
        max_samples=None,
        least: int = 1,
    ) -> np.ndarray:
        """The failure phases of ``mc_samples`` simulated missions (at
        least ``least``), then more while a run to ``tolerance`` needs
        them."""
        return self._simulated_runs(
            mc_samples,
            seed,
            antithetic,
            tolerance,
            confidence,
            max_samples,
            least,
        ).phase

    def _simulated_runs(
        self,
        mc_samples: int,
        seed,
        antithetic: bool = False,
        tolerance=None,
        confidence: float = 0.95,
        max_samples=None,
        least: int = 1,
    ) -> "_mission_runs.Runs":
        """``mc_samples`` simulated missions (at least ``least``), then
        more while a run to ``tolerance`` needs them."""
        montecarlo.check_count(mc_samples, antithetic, "mc_samples", least)
        montecarlo.check_confidence(confidence)
        limit = montecarlo.sample_limit(
            mc_samples,
            tolerance,
            max_samples,
            antithetic,
            ("mc_samples", "max_samples"),
        )
        parts: List[_mission_runs.Runs] = []
        with numpy_seed(seed):
            batch = mc_samples
            while batch:
                parts.append(self._runs(batch, antithetic))
                batch = 0
                if limit is not None:
                    succeeded = _mission_runs.joined(parts).phase == len(
                        self.phases
                    )
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
        return _mission_runs.joined(parts)

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
            If ``method`` is unknown, or ``mc_samples`` or ``seed`` is
            given with ``method="exact"``, or ``mc_samples`` is not a
            positive integer.
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
        return self._outcome(method, mc_samples, seed, "reliability")[0]

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
        return self._outcome(method, mc_samples, seed, "unreliability")[1]

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
        method = self._method(
            method, mc_samples, seed, "phase_failure_probabilities"
        )
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
            If ``mc_samples`` is not a whole number of at least 2 (an
            interval needs two missions; even, with ``antithetic``), or
            ``confidence``, ``tolerance`` or ``max_samples`` is invalid.
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
            mc_samples,
            seed,
            antithetic,
            tolerance,
            confidence,
            max_samples,
            least=2,
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
            method="simulated",
        )

    def importances(self, *, simulate: bool = False) -> ImportanceTable:
        """Every importance measure in one table.

        Calls each importance method of this model in turn (Birnbaum,
        improvement potential, risk achievement and reduction worth,
        criticality, Fussell-Vesely, differential, joint, Barlow-Proschan
        and structural, those the model has) with the arguments given, and
        gathers what they return. Each entry of ``measures`` is exactly
        the method's own result; nothing is computed a second way. A
        measure this model cannot work out is named under ``refused`` with
        the reason its method gives (as
        [`analysis_routes`][repyability.PhasedMission.analysis_routes]
        has it), and a measure only a simulation gives is listed there as
        simulated unless ``simulate=True``.

        Parameters
        ----------
        simulate : bool, optional
            Whether to run the measures that are simulated. By default
            False: they are listed under ``refused`` with ``"simulated:
            pass simulate=True"``.

        Returns
        -------
        ImportanceTable
            [`ImportanceTable`][repyability.ImportanceTable]: ``measures``
            (``{method name: {component: value}}``), ``refused``
            (``{method name: reason}``) and ``rows()``, one plain dict
            per component for a table. The joint importance, which is per
            pair, is in ``measures`` and not in ``rows()``.

        Raises
        ------
        NotImplementedError
            If no measure can be worked out here; the message gives each
            one's reason.
        ValueError
            If ``simulate`` is not a bool.
        """
        from repyability.rbd import _importances

        calls = {
            name: getattr(self, name)
            for name in _importances.MEASURES
            if hasattr(self, name)
        }
        return _importances.table(self, calls, simulate)

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
        importance in a reliable mission keeps its precision. With blocks
        on standby, each is the mission's Markov chain run with the block
        so held (working: it never fails, nor does its switch, though it
        idles until called on; failed: from the start), where the chain
        is worked out (every life at a constant rate), as for
        ``reliability``; the other measures follow. Exact, at the
        mission's end; it assumes independent components.

        Returns
        -------
        dict[Hashable, float]
            ``{component: importance}`` over the mission's components.

        Raises
        ------
        NotImplementedError
            If the mission's decision diagram needs more than ``MAX_NODES``
            nodes, or a block is on standby and the mission's Markov chain
            cannot be worked out (naming ``failure_drivers``).

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
        self._require_measures("importance measures")
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
            nodes, or a block is on standby and the mission's Markov chain
            cannot be worked out (naming ``failure_drivers``).

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
        self._require_measures("importance measures")
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
            nodes, or a block is on standby and the mission's Markov chain
            cannot be worked out (naming ``failure_drivers``).

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
        self._require_measures("importance measures")
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
            nodes, or a block is on standby and the mission's Markov chain
            cannot be worked out (naming ``failure_drivers``).

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
        self._require_measures("importance measures")
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
            nodes, or a block is on standby and the mission's Markov chain
            cannot be worked out (naming ``failure_drivers``).

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
        self._require_measures("importance measures")
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
        self._require_exact("Fussell-Vesely importance")
        _, failure = self._exact(len(self.phases))
        numerators = self._fv_numerators(method)
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                component: float(np.float64(numerator) / failure)
                for component, numerator in numerators.items()
            }

    def differential_importance(
        self,
        *,
        change: str = "uniform",
        kind: str = "failure",
        groups: Optional[Mapping[Hashable, Collection[Hashable]]] = None,
    ) -> Dict[Hashable, float]:
        """Each component's share of the change in the mission's
        reliability when they all change together: the differential
        importance measure (DIM, Borgonovo & Apostolakis, 2001), as
        [`NonRepairableRBD.differential_importance`][repyability.NonRepairableRBD.differential_importance]
        defines it over the components, each held across every phase.

        ``change="uniform"`` moves every component's probability by as
        much, and shares out the Birnbaum importance (see
        [`birnbaum_importance`][repyability.PhasedMission.birnbaum_importance]);
        ``"proportional"`` each by the same fraction of itself, and shares
        out the criticality importance of that ``kind`` (see
        [`criticality_importance`][repyability.PhasedMission.criticality_importance]).
        The shares add up to 1, and a group's share is the sum of its
        members' (``groups``). NaN where there is nothing to share out. At
        the mission's end; with blocks on standby, from the mission's
        Markov chain, as the Birnbaum importance is.

        Parameters
        ----------
        change : str, optional
            ``"uniform"`` (the default) or ``"proportional"``.
        kind : str, optional
            With a proportional change, which of each component's
            probabilities moves in proportion: ``"failure"`` (the
            default), of failing, or ``"success"``, of working.
        groups : dict, optional
            ``{name: components}``: each group's share, the sum of its
            components', instead of each component's.

        Returns
        -------
        dict[Hashable, float]
            The shares, by component or group name.

        Raises
        ------
        ValueError
            For an unknown ``change`` or ``kind``, or a group naming an
            unknown component.
        NotImplementedError
            As the Birnbaum importance refuses.

        Examples
        --------
        One unit through two phases, and a second needed only in the
        last, its life running from the mission's start: ``B_a =
        exp(-0.4)`` (``b`` lasting 20 hours) and ``B_b = exp(-0.2)``.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> E = surv.Exponential.from_params
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": E([0.01])})
        >>> two = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {"a": E([0.01]), "b": E([0.02])},
        ... )
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, two)])
        >>> shares = mission.differential_importance()
        >>> {k: round(v, 4) for k, v in shares.items()}
        {'a': 0.4502, 'b': 0.5498}
        """
        check_differential("components", change, kind)
        if change == "uniform":
            values = self.birnbaum_importance()
        else:
            values = self.criticality_importance(kind)
        return shares(values, groups)

    def joint_importance(self) -> Dict[Tuple[Hashable, Hashable], float]:
        """The joint (second-order) importance of each pair of components
        over the mission: whether improving the two together is worth more
        than improving each (Hong & Lie, 1993), as
        [`NonRepairableRBD.joint_importance`][repyability.NonRepairableRBD.joint_importance]
        defines it, each component held across every phase.

        ``JRI(i, j) = R(1_i, 1_j) - R(1_i, 0_j) - R(0_i, 1_j) + R(0_i,
        0_j)``, with ``1`` a component working throughout and ``0`` failed
        from the start: how much component ``j``'s Birnbaum importance
        rises when ``i`` goes from failed to working. Each value is the
        mission's decision diagram replayed with the two so held (with
        blocks on standby, the mission's Markov chain run so), and the sum
        is taken of the probabilities of succeeding or of failing,
        whichever are the smaller, so a small one keeps its precision.
        Positive, the two are complements (needed together); negative,
        substitutes (either will do); 0 where neither changes what the
        other is worth. Exact, at the mission's end.

        Returns
        -------
        dict
            ``{(i, j): JRI}`` for each pair once, its names in order as
            text, and found either way round.

        Raises
        ------
        NotImplementedError
            As the Birnbaum importance refuses.

        Examples
        --------
        Two engines, both needed to take off and either to cruise: with
        either failed from the start the mission fails, so the two are
        complements, each worth everything with the other working and
        nothing without it.

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
        >>> mission.joint_importance()["e1", "e2"]
        1.0
        """
        self._require_measures("importance measures")
        success, failure = self._exact(len(self.phases))
        names = list(self.components)
        pairs = [(i, j) for k, i in enumerate(names) for j in names[k + 1 :]]
        values = self._held_outcomes(
            [
                {i: first, j: second}
                for i, j in pairs
                for first, second in (
                    (True, True),
                    (True, False),
                    (False, True),
                    (False, False),
                )
            ]
        )
        out = Pairs()
        for n, (i, j) in enumerate(pairs):
            both, one, other, neither = values[4 * n : 4 * n + 4]
            if success <= failure:
                value = both[0] - one[0] - other[0] + neither[0]
            else:
                value = -(both[1] - one[1] - other[1] + neither[1])
            out[Pairs.oriented(i, j)] = float(value)
        return out

    def barlow_proschan_importance(self, x=None) -> Dict[Hashable, Any]:
        """Each component's Barlow-Proschan importance over the mission:
        the probability that the mission's failure is caused by the
        component's, given that it fails during its phases by ``x``, or
        by its end.

        In phase ``j``, component ``i``'s failure at time ``t`` fails the
        mission when the mission has got that far with ``i`` critical to
        phase ``j`` then: the rate of change of the mission's reliability
        ``R(t)`` (see
        [`reliability_over_time`][repyability.PhasedMission.reliability_over_time])
        with the probability that ``i`` survives the phase so far, ``p_ij(t)
        = R_i(t) / R_i(T_{j-1})``, times how fast that falls, ``f_i(t) /
        R_i(T_{j-1})``, its density (its model's ``df``, or differences of
        its ``ff``) over its reliability at the phase's start. The
        derivatives are one reverse pass of the mission's decision
        diagram, and each component's part is integrated by adaptive
        quadrature (to about ``1e-9``, relative) on pieces split at the
        phases' boundaries; its share of the parts' sum is its importance,
        and the shares add up to 1. Each part is a component's term of
        ``-dR/dt`` within the phases. A failure at a phase's start, of a
        phase that needs a component already failed, is caused by no
        failure then and takes no part: the shares are of the failures
        during the phases. With one phase it is the diagram's
        [`barlow_proschan_importance`][repyability.NonRepairableRBD.barlow_proschan_importance]
        at the phase's end.

        Parameters
        ----------
        x : float or array_like, optional
            Mission times, from 0 to the duration, by which the mission has
            failed; by default its end.

        Returns
        -------
        dict
            Each component's share: floats for a scalar ``x`` or the
            mission's end, arrays otherwise; nan where the mission cannot
            have failed during its phases by then.

        Raises
        ------
        ValueError
            If a time is not from 0 to the mission's duration.
        NotImplementedError
            If a block is on standby (its failure fails the mission only
            once it is called on), naming ``failure_drivers``, or the
            mission's decision diagram is too large.

        Examples
        --------
        Two units in series through one phase, with failure rates 1 and 3
        per thousand hours: the second causes three quarters of the
        failures.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> E = surv.Exponential.from_params
        >>> series = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {"a": E([0.001]), "b": E([0.003])},
        ... )
        >>> mission = PhasedMission([("only", 100.0, series)])
        >>> shares = mission.barlow_proschan_importance()
        >>> {k: round(v, 6) for k, v in shares.items()}
        {'a': 0.25, 'b': 0.75}
        """
        self._require_exact("Barlow-Proschan importance")
        times = None if x is None else self._times(x, "x")
        count = len(self.phases)
        plans = [self._plan(j + 1) for j in range(count)]
        rows = [_plan_rows(plan) for plan in plans]
        components = list(self.components)
        starts = self._starts()
        before = {
            node: np.concatenate([[1.0], sf[:-1]])
            for node, (sf, _) in self._raw_ends().items()
        }
        guaranteed = {node: self._guaranteed(node) for node in components}
        lives = {node: self._lives[node].model for node in components}

        def importance(t: np.ndarray) -> Dict[Hashable, np.ndarray]:
            # dR/dp_ij at each time, over R_i(T_{j-1}): the rate of change
            # of the mission's reliability per unit of component i's
            # density, 0 where it cannot fail then.
            out = {node: np.zeros(t.shape) for node in components}
            for k, (j, raw) in enumerate(self._cut_short(t)):
                p, q = self._segments_of(raw)
                pivots, actives, inactives, names = rows[j]
                _, gradient, reached = value_and_gradient(
                    pivots,
                    actives,
                    inactives,
                    plans[j][1],
                    [p[name] for name in names],
                    [q[name] for name in names],
                    1.0,
                    0.0,
                    len(names),
                )
                found = {
                    name: value
                    for name, value, has in zip(names, gradient, reached)
                    if has
                }
                for i, node in enumerate(components):
                    start = before[node][j]
                    if guaranteed[node][j] or start <= 0.0:
                        continue
                    # dR/dp_ij, as -dF/dp_ij (the replay is of failing).
                    out[node][k] = -found.get(i * count + j, 0.0) / start
            return out

        def parts(t: np.ndarray) -> Dict[Hashable, np.ndarray]:
            t = np.ravel(np.asarray(t, dtype=float))
            return _measures.rate_parts(
                importance(t),
                lambda node, at: _measures.density(lives[node], at),
                t,
            )

        models = [m for node in components for m in self._lives[node].models()]
        return _measures.barlow_proschan(
            parts,
            models,
            times if times is None or times.ndim else float(times),
            lambda _: self.duration,
            knots=starts[1:],
        )

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
        self._require_exact("minimal cut sets")
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

    # ------------------------------------------------------------------
    # The whole mission at once
    # ------------------------------------------------------------------

    def _starts(self) -> List[float]:
        """When each phase starts, from the mission's start."""
        return [0.0] + [phase.end for phase in self.phases[:-1]]

    def _choose(self, method: Optional[str], mc_samples, seed, name: str):
        """``"exact"`` or ``"simulate"``, and why it is simulated: by
        default (``method=None``), exact unless a block is on standby and
        the mission's Markov chain cannot be worked out."""
        if method is None:
            reason = routes.refusal(self._require_exact)
            return ("simulate", reason) if reason else ("exact", None)
        method = self._method(method, mc_samples, seed, name)
        return method, (
            "Asked for (method='simulate')." if method == "simulate" else None
        )

    def _times(self, t, name: str = "t") -> np.ndarray:
        """``t`` checked: times from 0 to the mission's duration."""
        times = real_array(t, name)
        if not np.all((times >= 0.0) & (times <= self.duration)):
            raise ValueError(
                f"{name} must be times from 0 to the mission's duration, "
                f"{self.duration:g}, got {t!r}."
            )
        return times

    def _exact_over_time(self, times: np.ndarray) -> np.ndarray:
        """The mission's exact reliability at each of ``times``: the
        phases before the one under way (the latest to start by then)
        whole, and that one cut short."""
        if self._chained():
            self._require_exact()
            curve = self._chain_values(times=times).curve
            return np.asarray(curve, dtype=float)
        out = np.empty(times.size)
        for k, (j, raw) in enumerate(self._cut_short(np.ravel(times))):
            out[k] = _outcomes(self._plan(j + 1), *self._segments_of(raw))[0]
        return out.reshape(times.shape)

    def _cut_short(self, times: np.ndarray):
        """For each of ``times`` (a flat array), the phase under way then
        (the latest to start by then) and each component's life's
        probabilities of working and of having failed at the ends of the
        phases before it and at the time (see ``_raw_ends``): each model
        asked once for every time."""
        starts, ends = self._starts(), [phase.end for phase in self.phases]
        whole = self._raw_ends()
        at = self._raw_ends(times)
        phases = np.searchsorted(starts, times, side="right") - 1
        for k, j in enumerate(phases):
            j = int(j)
            if times[k] == ends[j]:
                yield j, {
                    node: (sf[: j + 1], ff[: j + 1])
                    for node, (sf, ff) in whole.items()
                }
            else:
                yield j, {
                    node: (
                        np.append(sf[:j], at[node][0][k]),
                        np.append(ff[:j], at[node][1][k]),
                    )
                    for node, (sf, ff) in whole.items()
                }

    def reliability_over_time(
        self,
        t,
        method: Optional[str] = None,
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ):
        """The probability that the mission has not failed by each mission
        time ``t``, on one clock across the phases.

        It falls through each phase as the blocks fail, and may step down
        at a phase's start, where the phase needs blocks that the earlier
        ones did not (a time at a boundary belongs to the phase starting
        then, so a failure at its start has happened by it). At the
        mission's duration it is the mission's reliability.

        Parameters
        ----------
        t : float or array_like
            Times from 0 to the mission's duration.
        method : str, optional
            ``"exact"``: each time's phase cut short there, the earlier
            ones whole, through the mission's decision diagram.
            ``"simulate"``: the fraction of simulated missions not failed
            by then. By default exact, unless a block is on standby and
            the mission's Markov chain cannot be worked out.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000 (used only
            when simulated).
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float or numpy.ndarray
            The reliability at each time: a float for a scalar ``t``.

        Raises
        ------
        ValueError
            If a time is not from 0 to the duration, or ``method`` is
            unknown, or ``mc_samples`` or ``seed`` is given with
            ``method="exact"``.
        NotImplementedError
            With ``method="exact"``, if a block is on standby and a life
            has no constant rate (or the chain too many states), or the
            mission's decision diagram is too large.

        Examples
        --------
        One exponential unit through two phases: ``R(t) = exp(-0.01 t)``
        across them.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> over = mission.reliability_over_time([5.0, 15.0])
        >>> [round(float(r), 6) for r in over]
        [0.951229, 0.860708]
        """
        times = self._times(t)
        method, _ = self._choose(
            method, mc_samples, seed, "reliability_over_time"
        )
        if method == "simulate":
            runs = self._simulated_runs(
                10_000 if mc_samples is None else mc_samples, seed
            )
            out = np.mean(runs.time[:, None] > times.ravel()[None, :], axis=0)
            out = out.reshape(times.shape)
        else:
            out = self._exact_over_time(times)
        return float(out) if out.ndim == 0 else out

    def expected_mission_time(
        self,
        method: Optional[str] = None,
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> float:
        """The expected time the mission runs: until it fails, or to its
        end, whichever comes first.

        A mission that succeeds has no time to failure, so it has no mean
        life (``mean``); this is the integral of the reliability over the
        mission's time (see
        [`reliability_over_time`][repyability.PhasedMission.reliability_over_time]),
        from 0 to the duration, ``E[min(T, duration)]``: the duration
        less the time a failed mission loses. Exact, it is integrated by
        adaptive Gauss-Legendre quadrature (to about ``1e-9``, relative),
        on pieces split at the phases' boundaries.

        Parameters
        ----------
        method : str, optional
            ``"exact"``: the reliability over time integrated (numerical).
            ``"simulate"``: the mean over simulated missions of the time
            each runs. By default exact, unless a block is on standby and
            the mission's Markov chain cannot be worked out.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000 (used only
            when simulated).
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float
            The expected time the mission runs, in the components' unit.

        Raises
        ------
        ValueError
            If ``method`` is unknown, or ``mc_samples`` or ``seed`` is
            given with ``method="exact"``.
        NotImplementedError
            With ``method="exact"``, as ``reliability_over_time`` refuses.

        Examples
        --------
        One exponential unit for 20 hours: ``(1 - exp(-0.2)) / 0.01``.

        >>> import math
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> math.isclose(
        ...     mission.expected_mission_time(),
        ...     (1 - math.exp(-0.2)) / 0.01,
        ...     rel_tol=1e-9,
        ... )
        True
        """
        from repyability.rbd import _rates
        from repyability.rbd._mean_lifetime import model_knots

        method, _ = self._choose(
            method, mc_samples, seed, "expected_mission_time"
        )
        if method == "simulate":
            runs = self._simulated_runs(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return float(np.mean(np.minimum(runs.time, self.duration)))
        knots = np.concatenate(
            [np.empty(0)]
            + [
                model_knots(model)
                for life in self._lives.values()
                for model in life.models()
            ]
        )
        inside = knots[(knots > 0.0) & (knots < self.duration)]
        edges = np.unique(
            np.concatenate([[0.0], [p.end for p in self.phases], inside])
        )
        if self.duration == 0.0:
            return 0.0
        return float(
            _rates.integral(
                lambda t: self._exact_over_time(np.asarray(t))[:, None], edges
            )[0]
        )

    def _phase_failures(self, held: Optional[Hashable] = None) -> List[float]:
        """The exact probability of failing in each phase, with the
        component ``held`` (if any) working throughout."""
        if self._chained():
            hold = None if held is None else {held: True}
            return list(self._chain_values(held=hold).failing)
        segments = self._segments(held=None if held is None else {held: 1.0})
        out: List[float] = []
        before = 0.0
        for j in range(len(self.phases)):
            _, failure = _outcomes(self._plan(j + 1), *segments)
            out.append(failure - before)
            before = failure
        return out

    def _drivers(
        self, runs: Optional[Tuple[Any, Dict[Hashable, Any]]]
    ) -> "MissionDrivers":
        """The drivers, exact (``runs`` None) or from simulated missions
        ``runs``: the base run and one with each component held."""
        names = [phase.name for phase in self.phases]
        count = len(names)
        falls: Dict[Hashable, Dict[Hashable, float]] = {}
        totals: Dict[Hashable, float] = {}
        if runs is None:
            self._require_exact()
            base = self._phase_failures()
            for node in self.components:
                with_it = self._phase_failures(node)
                falls[node] = {
                    name: base[j] - with_it[j] for j, name in enumerate(names)
                }
                totals[node] = float(sum(falls[node].values()))
            return MissionDrivers("exact", names, falls, totals)
        plain, held = runs
        errors: Dict[Hashable, float] = {}
        for node, other in held.items():
            falls[node] = {
                name: float(
                    np.mean(plain.phase == j) - np.mean(other.phase == j)
                )
                for j, name in enumerate(names)
            }
            difference = (plain.phase < count).astype(float) - (
                other.phase < count
            ).astype(float)
            totals[node] = float(difference.mean())
            errors[node] = montecarlo.standard_error(difference, False)
        return MissionDrivers(
            "simulated", names, falls, totals, errors, len(plain.phase)
        )

    def _played(
        self, mc_samples: int, seed
    ) -> Tuple[Any, Dict[Hashable, Any]]:
        """``mc_samples`` simulated missions, and the same missions with
        each component never failing (common random numbers)."""
        montecarlo.check_count(mc_samples, False, "mc_samples", 2)
        with numpy_seed(seed):
            lives = _mission_runs.draw(self, mc_samples, False)
            if self.repairability:
                # The later lives and the repairs are drawn as the plays
                # ask for them, and kept for the plays that follow.
                source = _mission_repairs.Source(self, mc_samples)
                return _mission_repairs.play(
                    self, lives, mc_samples, source
                ), {
                    node: _mission_repairs.play(
                        self, lives, mc_samples, source, held=node
                    )
                    for node in self.components
                }
            switches = _mission_runs.draw_switches(self, mc_samples)
        plain = _mission_runs.play(self, lives, mc_samples, switches=switches)
        held = {
            node: _mission_runs.play(
                self, lives, mc_samples, held=node, switches=switches
            )
            for node in self.components
        }
        return plain, held

    def failure_drivers(
        self,
        method: Optional[str] = None,
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> "MissionDrivers":
        """Which blocks drove the mission's failures, phase by phase.

        For each block and each phase, the fall in the phase's failure
        probability if the block never failed: the mission worked out
        again with the block held working throughout (on standby it still
        idles until needed, but never fails). The total over the phases
        is the fall in the mission's unreliability, the block's
        improvement potential. A fall can be negative in a phase, when
        the block never failing carries missions into a later phase that
        then fails.

        Parameters
        ----------
        method : str, optional
            ``"exact"``: the mission's decision diagram replayed with each
            block's segments held. ``"simulate"``: ``mc_samples`` missions
            played again with each block held, on the same draws (common
            random numbers), with each total's standard error. By default
            exact, unless a block is on standby and the mission's Markov
            chain cannot be worked out.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000 (used only
            when simulated).
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        MissionDrivers
            The falls by block and phase, and their totals.

        Raises
        ------
        ValueError
            If ``method`` is unknown, or ``mc_samples`` or ``seed`` is
            given with ``method="exact"``.
        NotImplementedError
            With ``method="exact"``, if a block is on standby and a life
            has no constant rate (or the chain too many states), or the
            mission's decision diagram is too large.

        Examples
        --------
        Two exponential units in series for one phase: the mission fails
        with ``1 - exp(-0.03 t)``, and without unit ``a``'s failures with
        ``1 - exp(-0.02 t)``.

        >>> import math
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> series = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": surv.Exponential.from_params([0.01]),
        ...         "b": surv.Exponential.from_params([0.02]),
        ...     },
        ... )
        >>> mission = PhasedMission([("only", 10.0, series)])
        >>> drivers = mission.failure_drivers()
        >>> math.isclose(drivers.totals["a"], math.exp(-0.2) - math.exp(-0.3))
        True
        """
        method, _ = self._choose(method, mc_samples, seed, "failure_drivers")
        if method == "simulate":
            return self._drivers(
                self._played(
                    10_000 if mc_samples is None else mc_samples, seed
                )
            )
        return self._drivers(None)

    def summary(
        self,
        times=None,
        method: Optional[str] = None,
        *,
        mc_samples: Optional[int] = None,
        seed=None,
        confidence: float = 0.95,
    ) -> "MissionSummary":
        """The whole mission at once, from one run: its reliability, its
        reliability over time, each phase's part in its failures, and the
        blocks that drove them.

        Exact where the mission is (no block on standby, or every life
        with a constant rate), else from one
        simulation of ``mc_samples`` missions, which every value is read
        from (the drivers from the same missions played again with each
        block held, see
        [`failure_drivers`][repyability.PhasedMission.failure_drivers]).

        Parameters
        ----------
        times : array_like, optional
            The times for the reliability over time, from 0 to the
            duration; by default 20 steps through each phase, every phase
            boundary included.
        method : str, optional
            ``"exact"`` or ``"simulate"``; by default exact unless a
            block is on standby and a life has no constant rate.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000 (used only
            when simulated).
        seed : int, optional
            Seeds the simulation, by default None.
        confidence : float, optional
            The confidence level of the simulated reliability's interval
            and of the band about its reliability over time (two-sided),
            by default 0.95.

        Returns
        -------
        MissionSummary
            The mission's values, each phase's
            ([`MissionPhaseSummary`][repyability.MissionPhaseSummary]) and
            the drivers ([`MissionDrivers`][repyability.MissionDrivers]).

        Raises
        ------
        ValueError
            If a time is not from 0 to the duration, ``method`` is
            unknown, ``mc_samples`` or ``seed`` is given with
            ``method="exact"``, or ``mc_samples`` is less than 2.
        NotImplementedError
            With ``method="exact"``, if a block is on standby and a life
            has no constant rate (or the chain too many states), or the
            mission's decision diagram is too large.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> result = mission.summary()
        >>> result.method, result.riskiest_phase
        ('exact', 'out')
        >>> round(result.phases[1].conditional_failure, 6)  # 1 - exp(-0.1)
        0.095163
        """
        starts = self._starts()
        names = [phase.name for phase in self.phases]
        count = len(names)
        if times is None:
            grid = [
                np.linspace(start, phase.end, 21)
                for start, phase in zip(starts, self.phases)
            ]
            grid_times = np.unique(np.concatenate(grid))
        else:
            grid_times = np.ravel(self._times(times, "times"))
        method, reason = self._choose(method, mc_samples, seed, "summary")
        takeovers: List[Dict[Hashable, float]] = [{} for _ in names]
        interval = None
        n_samples = None
        band: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
        if method == "simulate":
            montecarlo.check_confidence(confidence)
            n_samples = 10_000 if mc_samples is None else mc_samples
            runs = self._played(n_samples, seed)
            plain = runs[0]
            succeeded = (plain.phase == count).astype(float)
            success = float(succeeded.mean())
            failure = 1.0 - success
            error = montecarlo.standard_error(succeeded, False)
            half = montecarlo.z_value(confidence) * error
            interval = ConfidenceInterval(
                estimate=success,
                lower=max(success - half, 0.0),
                upper=min(success + half, 1.0),
                confidence=confidence,
                standard_error=error,
                n_samples=n_samples,
                method="simulated",
            )
            failing = [float(np.mean(plain.phase == j)) for j in range(count)]
            through = [float(np.mean(plain.phase > j)) for j in range(count)]
            curve = np.mean(plain.time[:, None] > grid_times[None, :], axis=0)
            band = wilson_band(
                curve, n_samples, montecarlo.z_value(confidence)
            )
            for (j, node), started in plain.started.items():
                takeovers[j][node] = float(started.mean())
            drivers = self._drivers(runs)
        else:
            success, failure = self._exact(count)
            failing = self._phase_failures()
            through = [self._exact(j + 1)[0] for j in range(count)]
            curve = self._exact_over_time(grid_times)
            drivers = self._drivers(None)
            if self._chained():
                takeovers = list(self._chain_values().takeovers)
        phases = []
        reached = 1.0
        for j, (phase, setting) in enumerate(zip(self.phases, self._settings)):
            phases.append(
                MissionPhaseSummary(
                    name=phase.name,
                    start=starts[j],
                    end=phase.end,
                    failure_probability=failing[j],
                    share=failing[j] / failure if failure > 0 else math.nan,
                    conditional_failure=(
                        failing[j] / reached if reached > 0 else math.nan
                    ),
                    reliability_at_end=through[j],
                    settings={
                        node: setting_to_dict(given)
                        for node, given in setting.blocks.items()
                    },
                    takeovers=takeovers[j],
                    top_driver=drivers.top(phase.name),
                )
            )
            reached = through[j]
        riskiest = names[int(np.argmax(failing))] if max(failing) > 0 else None
        return MissionSummary(
            method="simulated" if method == "simulate" else "exact",
            reason=reason,
            duration=self.duration,
            reliability=success,
            unreliability=failure,
            interval=interval,
            times=grid_times,
            reliability_over_time=np.asarray(curve, dtype=float),
            phases=phases,
            drivers=drivers,
            riskiest_phase=riskiest,
            n_samples=n_samples,
            lower=band[0],
            upper=band[1],
        )

    def compare(
        self,
        other: "PhasedMission",
        mc_samples: int = 100_000,
        seed=None,
        *,
        confidence: float = 0.95,
        method: Optional[str] = None,
    ) -> ConfidenceInterval:
        """How much more (or less) likely this mission is to succeed than
        ``other``: exactly where ``reliability`` works out both, and
        otherwise by simulation with common random numbers, as
        [`NonRepairableRBD.compare`][repyability.NonRepairableRBD.compare]
        compares two diagrams.

        Simulated, both missions are played ``mc_samples`` times, a
        component of the same name in both drawing the same random
        numbers for its life (the same quantile of its own model), and a
        block on standby the same for whether it starts in a phase of the
        same number, so the differences come from how the missions
        differ, not from chance. ``reliability_interval`` draws as it did.

        Parameters
        ----------
        other : PhasedMission
            The mission to compare with.
        mc_samples : int, optional
            Simulated, the number of missions of each, by default 100_000.
        seed : int, optional
            Simulated, the seed, by default None.
        confidence : float, optional
            The confidence level of the interval, by default 0.95.
        method : str, optional
            None (the default): exact where both reliabilities are, else
            simulated. ``"exact"``: refused where one is not exact.
            ``"simulate"``: simulated whatever.

        Returns
        -------
        ConfidenceInterval
            This mission's reliability minus ``other``'s: exact
            (``method="exact"``, with no error), or the mean of the paired
            differences of success (``"simulated"``), with its standard
            error and a normal interval within ``[-1, 1]``.

        Raises
        ------
        TypeError
            If ``other`` is not a ``PhasedMission``.
        ValueError
            If ``mc_samples``, ``confidence`` or ``method`` is not valid.
        NotImplementedError
            Simulated, if a block's life cannot be drawn from uniforms;
            with ``method="exact"``, as ``reliability`` refuses.

        Examples
        --------
        The way back cut from 10 hours to 5, for a unit failing at 0.01 an
        hour: ``exp(-0.15) - exp(-0.2)``, about 0.042.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> long = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> short = PhasedMission([("out", 10.0, one), ("back", 5.0, one)])
        >>> gain = short.compare(long)
        >>> round(gain.estimate, 4), gain.method
        (0.042, 'exact')
        """
        from repyability.rbd import _paired

        _paired.check(method, mc_samples, confidence)
        _paired.same_kind(
            self, other, "a mission compares with another mission"
        )
        if method != "simulate":
            try:
                exact = float(self.reliability() - other.reliability())
            except NotImplementedError:
                if method == "exact":
                    raise
            else:
                return _paired.exact(exact, confidence)
        key = _paired.key(seed)
        count = (len(self.phases), len(other.phases))
        ours = _mission_runs.keyed(self, mc_samples, key).phase == count[0]
        theirs = _mission_runs.keyed(other, mc_samples, key).phase == count[1]
        return _paired.simulated(
            ours.astype(float) - theirs.astype(float), confidence, clip=1.0
        )

    # ------------------------------------------------------------------
    # Levers
    # ------------------------------------------------------------------

    @classmethod
    def _from_args(cls, args: Dict[str, Any]) -> "PhasedMission":
        """The mission built from its constructor's arguments (see
        ``_init_args``): ``from_diagram``'s, with the diagram under
        ``"rbd"``, or the constructor's own."""
        groups, repairs = args.get("ccf_groups"), args.get("repairability")
        if "rbd" in args:
            return cls.from_diagram(
                args["rbd"], args["phases"], groups, repairs
            )
        return cls(args["phases"], groups, repairs)

    def parameter_sensitivity(
        self, rel_step: float = 1e-5
    ) -> Dict[Hashable, Dict[str, float]]:
        """The sensitivity of the mission's reliability to each lever: each
        component's model's parameters, by surpyval's names, and each
        phase's duration.

        Each derivative is a central difference of the mission's exact
        reliability, the mission rebuilt with the lever moved either way
        by ``rel_step`` of its value (``rel_step`` itself at 0), as
        [`with_levers`][repyability.PhasedMission.with_levers] builds it:
        a component's model in every phase that uses it, a phase's
        duration moving the later phases with it. It is one-sided where
        one side is not a valid value, and nan where neither is; it is
        differenced on the mission's unreliability where that is the
        smaller, so a small change keeps its precision. A block's model
        of a phase's own (a ``BlockSetting``'s ``model``) is not a lever.
        ``levers()`` lists them, with their values and ranges.

        In the guide's Greeks it is the levers' *deltas* (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        Parameters
        ----------
        rel_step : float, optional
            The relative step of the differences, by default 1e-5.

        Returns
        -------
        dict
            ``{component: {parameter: derivative}}``, and the phases'
            durations under ``None``, as ``{"phase.<name>.duration":
            derivative}``.

        Raises
        ------
        ValueError
            If ``rel_step`` is not a number between 0 and 1.
        NotImplementedError
            As ``reliability`` refuses, or, with blocks on standby whose
            Markov chain cannot be worked out, naming ``failure_drivers``.

        Examples
        --------
        One exponential unit for 10 hours then 10 more: ``R = exp(-20
        rate)``, so ``dR/d rate = -20 exp(-0.2)``, and each phase's
        duration moves it by ``-0.01 exp(-0.2)``.

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> sens = mission.parameter_sensitivity()
        >>> round(sens["a"]["failure_rate"], 4)
        -16.3746
        >>> round(sens[None]["phase.back.duration"], 6)
        -0.008187
        """
        from repyability.rbd import _sensitivity

        self._require_measures("parameter sensitivities")
        if not (is_number(rel_step) and 0.0 < float(rel_step) < 1.0):
            raise ValueError(
                f"rel_step must be a number between 0 and 1, got {rel_step!r}."
            )
        success, failure = self._exact(len(self.phases))
        # Differences of the smaller of the two keep their precision.
        smaller = 1 if failure <= success else 0
        sign = -1.0 if smaller else 1.0
        base = np.array([sign * (failure if smaller else success)])
        out: Dict[Hashable, Dict[str, float]] = {}
        for lever in _sensitivity._mission_levers(self):

            def at(value, lever=lever):
                try:
                    changed = PhasedMission._from_args(lever.build(value))
                except (ValueError, TypeError):
                    return None
                found = changed._exact(len(changed.phases))
                return np.array([sign * found[smaller]])

            change = _sensitivity._difference(at, lever, float(rel_step), base)
            out.setdefault(lever.key, {})[lever.name] = float(change[0])
        return out

    def levers(self) -> List["Lever"]:
        """The levers ``parameter_sensitivity`` moves, in the order it
        reports them: each component's model's parameters, by surpyval's
        names (a regression block's covariates), then each phase's
        duration, under ``None`` as ``"phase.<name>.duration"``; each a
        [`Lever`][repyability.Lever] with its value and range, as a
        diagram's ``levers()`` lists them. Nothing is worked out;
        [`with_levers`][repyability.PhasedMission.with_levers] sets them.

        Returns
        -------
        list of Lever
            The levers, in ``parameter_sensitivity``'s order.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 5.0, one)])
        >>> for lever in mission.levers():
        ...     print(lever.key, lever.name, lever.value)
        a alpha 100.0
        a beta 2.0
        None phase.out.duration 10.0
        None phase.back.duration 5.0
        """
        from repyability.rbd import _sensitivity

        return _sensitivity.public_levers(self)

    def with_levers(self, values) -> "PhasedMission":
        """A copy of this mission with levers at new values: a component's
        model rebuilt with the parameter changed, in every phase's diagram
        that uses it, or a phase's duration. ``levers()`` lists them.

        Parameters
        ----------
        values : dict
            Each lever's new value, by its [`Lever`][repyability.Lever]
            (from ``levers()``) or by its ``(key, name)``, as
            ``parameter_sensitivity`` reports it.

        Returns
        -------
        PhasedMission
            The new mission, built as this one was with the levers moved;
            this one is unchanged.

        Raises
        ------
        ValueError
            For a lever the mission does not have (with the closest name),
            or a value its model or the mission refuses (one outside the
            lever's ``bounds``).
        TypeError
            If ``values`` is not a dict, a lever is neither a ``Lever``
            nor a ``(key, name)``, or a value is not a number.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> shorter = mission.with_levers({(None, "phase.back.duration"): 5.0})
        >>> shorter.duration
        15.0
        """
        from repyability.rbd import _sensitivity

        return _sensitivity.with_levers(self, values)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """The mission as a JSON-friendly dict (see ``from_dict``), as a
        diagram saves itself (see [`RBD.to_dict`][repyability.RBD.to_dict]).

        A mission built from a diagram per phase saves each phase's name,
        duration and diagram (``NonRepairableRBD.to_dict``); one drawn as
        one diagram (``from_diagram``) saves the diagram once, and each
        phase's name, duration, blocks' settings (their states,
        covariates, models, start probabilities and dormancy) and vote
        counts.

        Returns
        -------
        dict
            The ``repyability_version`` that wrote it, the ``type``
            (``"PhasedMission"``), the ``phases``, for a mission drawn
            as one diagram the ``rbd``, and the ``ccf_groups`` and the
            ``repairability`` if it has any.

        Raises
        ------
        NotImplementedError
            If a model cannot be serialised: one that is not surpyval's
            (nor a RePyability node model).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 5.0, one)])
        >>> d = mission.to_dict()
        >>> d["type"], [phase["duration"] for phase in d["phases"]]
        ('PhasedMission', [10.0, 5.0])
        """
        out: Dict[str, Any] = {
            "repyability_version": __version__,
            "type": "PhasedMission",
        }
        from repyability.rbd.serialisation import (
            _ccf_to_list,
            serialise_model,
        )

        if self.ccf_groups:
            out["ccf_groups"] = _ccf_to_list(self.ccf_groups)
        if self.repairability:
            out["repairability"] = [
                {"node": node, "model": serialise_model(model)}
                for node, model in self.repairability.items()
            ]
        if "rbd" not in self._init_args:
            # The diagrams as given: a group's causes are written into
            # them again when the mission is rebuilt.
            out["phases"] = [
                {
                    "name": name,
                    "duration": float(duration),
                    "rbd": rbd.to_dict(),
                }
                for name, duration, rbd in self._init_args["phases"]
            ]
            return out
        out["rbd"] = self._init_args["rbd"].to_dict()
        out["phases"] = [
            {
                "name": setting.name,
                "duration": setting.duration,
                "blocks": [
                    {
                        "node": node,
                        "setting": setting_to_dict(given, saving=True),
                    }
                    for node, given in setting.blocks.items()
                ],
                "k": [
                    {"node": node, "k": count}
                    for node, count in setting.k.items()
                ],
            }
            for setting in self._settings
        ]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "PhasedMission":
        """Rebuild a mission from ``to_dict``'s output, by calling the
        constructor (or ``from_diagram``) again, so it is checked again.
        Tuple names, which JSON turns into lists, are turned back.

        Parameters
        ----------
        d : dict
            As ``to_dict`` returns (possibly after a JSON round trip).

        Returns
        -------
        PhasedMission
            The mission.

        Raises
        ------
        ValueError
            If ``d`` is not a mission's dict, a model kind is unknown, or
            the constructor rejects the stored inputs.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 5.0, one)])
        >>> again = PhasedMission.from_dict(mission.to_dict())
        >>> again.reliability() == mission.reliability()
        True
        """
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD
        from repyability.rbd.serialisation import (
            _ccf_from_list,
            _node_name,
            deserialise_model,
        )

        repairs = {
            _node_name(entry["node"]): deserialise_model(entry["model"])
            for entry in d.get("repairability") or ()
        }

        kind = d.get("type") if isinstance(d, Mapping) else None
        if kind != "PhasedMission":
            raise ValueError(
                "PhasedMission.from_dict takes a mission's to_dict() "
                f'document, whose "type" is "PhasedMission", got {kind!r}.'
            )
        if "rbd" not in d:
            return cls(
                [
                    (
                        _node_name(phase["name"]),
                        phase["duration"],
                        NonRepairableRBD.from_dict(phase["rbd"]),
                    )
                    for phase in d["phases"]
                ],
                ccf_groups=_ccf_from_list(d.get("ccf_groups")),
                repairability=repairs or None,
            )
        return cls.from_diagram(
            NonRepairableRBD.from_dict(d["rbd"]),
            [
                MissionPhase(
                    _node_name(phase["name"]),
                    phase["duration"],
                    blocks={
                        _node_name(entry["node"]): setting_from_dict(
                            entry["setting"]
                        )
                        for entry in phase.get("blocks") or ()
                    },
                    k={
                        _node_name(entry["node"]): entry["k"]
                        for entry in phase.get("k") or ()
                    },
                )
                for phase in d["phases"]
            ],
            ccf_groups=_ccf_from_list(d.get("ccf_groups")),
            repairability=repairs or None,
        )

    def to_json(self, fp=None, **json_kwargs) -> Optional[str]:
        """The mission as a JSON document (see ``to_dict``): returned, or
        written to ``fp`` (as surpyval's models' ``to_json(fp)`` writes
        them).

        Parameters
        ----------
        fp : str, os.PathLike or file, optional
            A path, or a file opened for writing, to write the document to;
            by default None: it is returned.
        **json_kwargs
            Passed to ``json.dumps``, e.g. ``indent=2``.

        Returns
        -------
        str or None
            The JSON document, or None once written to ``fp``.

        Raises
        ------
        NotImplementedError
            If a model cannot be serialised (see ``to_dict``).
        TypeError
            If a name cannot be encoded as JSON.
        """
        return write_json(json.dumps(self.to_dict(), **json_kwargs), fp)

    @classmethod
    def from_json(cls, s) -> "PhasedMission":
        """Rebuild a mission from ``to_json``'s output.

        Parameters
        ----------
        s : str, os.PathLike or file
            The JSON document: its text, a path to a file holding it, or a
            file opened for reading.

        Returns
        -------
        PhasedMission
            The mission.

        Raises
        ------
        ValueError
            If ``s`` is not valid JSON, or for the reasons given in
            ``from_dict``.
        FileNotFoundError
            If ``s`` is neither a JSON document nor the path of a file.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 5.0, one)])
        >>> round(PhasedMission.from_json(mission.to_json()).reliability(), 4)
        0.9778
        """
        return cls.from_dict(json.loads(read_json(s)))

    def analysis_routes(self) -> Dict[str, routes.AnalysisRoute]:
        """How each analysis of the mission is computed, found without
        running it: exactly, by simulation, or not at all.

        The exact analyses replay the phases' decision diagram over the
        components' segments, and refuse when it would need more than
        ``MAX_NODES`` nodes (the diagram is built, and kept for the
        analyses; the Fussell-Vesely importance's cut sets' diagram too).
        They are as exact as the components' reliabilities: a component
        whose reliability is numerical makes them numerical, and one with
        none (a ``StandbyModel`` only simulations take) makes them refuse,
        as the diagrams' routes have it. The refusal is the message the
        method would raise.

        Returns
        -------
        dict[str, AnalysisRoute]
            For each public analysis, by method name: its ``route``
            (``"exact"``, ``"numerical"``, ``"simulated"`` or
            ``"refused"``), the ``reason`` and the components that decide
            it (see [`AnalysisRoute`][repyability.AnalysisRoute]).

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
        # A component whose model has no exact or numerical reliability
        # refuses what is built on it, as the diagrams' routes have it.
        # A block's route is the worst of its models' (one per phase where
        # a phase sets its model or covariates).
        nodes = {
            n: max(
                (r.model_route(m) for m in life.models()),
                key=lambda found: r._RANK[found[0]],
            )
            for n, life in self._lives.items()
        }
        count = len(self.phases)
        repaired = bool(self.repairability)
        standby = bool(self._chained())
        refused = r.refusal(self._require_exact) or (
            None if standby else r.refusal(lambda: self._plan(count))
        )
        how = (
            "A Markov chain over the blocks' states (running or failed), "
            "carried through the phases, the repaired blocks coming back "
            "in it; the mission fails at the first instant a phase's "
            "diagram is down."
            if repaired
            else (
                "A Markov chain over the blocks' states (running, idle, "
                "failed), carried through the phases, the take-overs of the "
                "blocks on standby in it."
                if standby
                else "The phases' decision diagram over the components' "
                "segments (Esary and Ziehms), replayed."
            )
        ) + (
            " Each common-cause group's shared causes are components of "
            "their own, a cause's segments repeated across the members it "
            "fails."
            if self.ccf_groups
            else ""
        )
        diagram = (
            r.refused(refused)
            if refused
            else r.with_nodes(r.EXACT, how, nodes, "reliabilities")
        )
        out = {
            name: diagram
            for name in ("reliability", "unreliability", "sf", "ff")
        }
        out["phase_failure_probabilities"] = diagram
        measures = r.refusal(
            lambda: self._require_measures("importance measures")
        )
        for name in (
            "birnbaum_importance",
            "improvement_potential",
            "risk_achievement_worth",
            "risk_reduction_worth",
            "criticality_importance",
            "differential_importance",
            "joint_importance",
        ):
            out[name] = r.refused(measures) if measures else diagram
        cuts = (
            r.refusal(lambda: self._require_exact("Fussell-Vesely importance"))
            or refused
            or r.refusal(self._cut_set_plan)
        )
        out["fussell_vesely"] = (
            r.refused(cuts)
            if cuts
            else r.with_nodes(
                r.EXACT,
                "The mission's minimal cut sets, their occurrence a "
                "decision diagram over the components' segments.",
                nodes,
                "reliabilities",
            )
        )
        densities = (
            r.refusal(
                lambda: self._require_exact("Barlow-Proschan importance")
            )
            or refused
        )
        out["barlow_proschan_importance"] = (
            r.refused(densities)
            if densities
            else r.with_nodes(
                r.NUMERICAL,
                "Each component's part of the rate at which the mission "
                "fails within its phases (the decision diagram's "
                "derivative by the component's segment, times its "
                "density), integrated over the mission's time by adaptive "
                "Gauss-Legendre quadrature (to about 1e-9).",
                nodes,
                "reliabilities",
            )
        )
        moved = r.refusal(
            lambda: self._require_measures("parameter sensitivities")
        )
        out["parameter_sensitivity"] = (
            r.refused(moved or diagram.reason)
            if moved or diagram.route == r.REFUSED
            else r.with_nodes(
                r.NUMERICAL,
                "Differences of the mission's exact reliability, the "
                "mission rebuilt with each lever moved either way. "
                + diagram.reason,
                nodes,
                "reliabilities",
            )
        )
        simulated = r.AnalysisRoute(
            r.SIMULATED,
            (
                "Each component's life drawn once per mission, and each "
                "repaired block's repairs and later lives as it fails and "
                "comes back."
                if repaired
                else "Each component's life drawn once per mission."
            ),
        )
        out["reliability_interval"] = simulated
        _, reason = self._choose(None, None, None, "")
        whole = (
            r.AnalysisRoute(
                r.SIMULATED,
                f"{reason} Each block's life drawn once per mission, the "
                "missions played through the phases.",
            )
            if reason
            else diagram
        )
        for name in ("reliability_over_time", "failure_drivers", "summary"):
            out[name] = whole
        out["expected_mission_time"] = (
            whole
            if whole.route != r.EXACT
            else r.with_nodes(
                r.NUMERICAL,
                "The reliability over the mission's time integrated by "
                "adaptive Gauss-Legendre quadrature (to about 1e-9). "
                + diagram.reason,
                nodes,
                "reliabilities",
            )
        )
        replay = r.refusal(lambda: _mission_runs.replayable(self))
        if diagram.route != r.REFUSED:
            out["compare"] = dataclasses.replace(
                diagram,
                reason="The difference of the two missions' reliabilities, "
                "where the other's is worked out too; otherwise, or with "
                "method='simulate', missions of both played with common "
                "random numbers. " + diagram.reason,
            )
        else:
            out["compare"] = (
                r.refused(replay)
                if replay
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "Missions of both played, a component of the same name "
                    "drawing the same random numbers in both (common random "
                    "numbers).",
                )
            )
        return with_importances(out)

    def _method(self, method: str, mc_samples, seed, name: str) -> str:
        one_of("method", method, ("exact", "simulate"))
        if method == "exact":
            simulation_options(
                f"{name}()", {"mc_samples": mc_samples, "seed": seed}
            )
        return method

    def _outcome(
        self, method: str, mc_samples, seed, name: str
    ) -> Tuple[float, float]:
        """The mission's reliability and unreliability, as ``name`` asks
        for them."""
        method = self._method(method, mc_samples, seed, name)
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
        unit = "" if self.unit is None else f"; unit {self.unit!r}"
        return f"PhasedMission({phases}{unit})"
