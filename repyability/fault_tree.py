"""Static fault tree analysis.

A fault tree works down from an undesired *top event*, usually the system
failing, to its causes, through *gates*:

- an **OR** gate occurs when any of its inputs occurs;
- an **AND** gate occurs when all of its inputs occur;
- a **VOTE** gate (k-out-of-n) occurs when at least ``k`` of its ``n``
  inputs occur.

Its leaves are the *basic events*: component failures, each with a
probability of having occurred or a lifetime model. An event (or a gate) may
feed several gates: a *repeated* event, such as a power supply that several
subsystems share.

A fault tree and a reliability block diagram describe one system from
opposite sides: the tree says when the system fails, the diagram when it
works. An OR gate is a block of its inputs in series, an AND gate a block in
parallel, and a VOTE gate needing ``k`` of ``n`` failures a block needing
``n - k + 1`` of ``n`` working. So the tree is evaluated by the exact engine
behind the diagrams (see ``rbd/modular.py``): every gate below which nothing
is shared with the rest of the tree is a module with a closed form, and what
the repeated events tie together is left as a core: its binary decision
diagram, built from the gates (#171), so that the minimal path and cut sets,
which multiply with the shared events, are found from it only when asked
for. Nothing is approximated: the top event probability and the importance
measures are exact, repeated events included.

The analyses over a lifetime are the diagram's, on the tree's own
evaluation (#314): the mean time to the top event, the B-lives, the density
and hazard, the Barlow-Proschan importance and the parameter sensitivities
call the functions ``rbd/non_repairable_rbd.py`` shares with
``NonRepairableRBD``, with the tree's probabilities in place of the
diagram's; the uncertainty from fitted event models is drawn by the
diagram of the tree (``to_rbd()``, converted once), whose draws are the
events'.
"""

import functools
import json
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
from numpy.typing import ArrayLike

from repyability.rbd._ordered_bdd import OrderedBDD
from repyability.rbd.modular import (
    KOON,
    NODE,
    PARALLEL,
    SERIES,
    Decomposition,
)
from repyability.rbd.routes import AnalysisRoute, Refused
from repyability.utils.checks import no_distribution, one_of, real_array

if TYPE_CHECKING:  # pragma: no cover
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD
    from repyability.rbd.results import Lever, UncertaintyResult

#: The kinds of gate.
GATE_KINDS = ("or", "and", "vote")

#: Building the core's decision diagram gives up, with guidance, beyond
#: this many nodes.
DIAGRAM_LIMIT = 2_000_000


class _Made(NamedTuple):
    """A gate ``FaultTree.from_rbd`` has made, by its place among them,
    before it is named."""

    place: int


class _Gate(NamedTuple):
    kind: str
    # The gate occurs when at least k of its inputs occur.
    k: int
    inputs: Tuple[Hashable, ...]


def _parse_gate(name, spec) -> _Gate:
    """A gate from ``("or", inputs)``, ``("and", inputs)`` or
    ``("vote", k, inputs)``, or a ValueError naming the gate."""
    usage = (
        f"Gate {name!r} must be ('or', inputs), ('and', inputs) or "
        f"('vote', k, inputs), got {spec!r}."
    )
    if not isinstance(spec, (tuple, list)) or not spec:
        raise ValueError(usage)
    kind = spec[0]
    if not isinstance(kind, str) or kind.lower() not in GATE_KINDS:
        raise ValueError(usage)
    kind = kind.lower()
    if len(spec) != (3 if kind == "vote" else 2):
        raise ValueError(usage)
    inputs = spec[-1]
    if not isinstance(inputs, (tuple, list)) or not inputs:
        raise ValueError(
            f"Gate {name!r} needs a non-empty list of inputs, got "
            f"{inputs!r}."
        )
    inputs = tuple(inputs)
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"Gate {name!r} lists an input more than once.")
    n = len(inputs)
    if kind == "or":
        return _Gate(kind, 1, inputs)
    if kind == "and":
        return _Gate(kind, n, inputs)
    k = spec[1]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(
            f"Vote gate {name!r}: k must be an integer, got {k!r}."
        )
    if not 1 <= k <= n:
        raise ValueError(
            f"Vote gate {name!r}: k must be between 1 and its {n} inputs, "
            f"got {k}."
        )
    return _Gate(kind, int(k), inputs)


def _parse_event(name, model):
    """A basic event's model: a probability (a float) or an object with
    ``sf``."""
    if isinstance(model, (int, float, np.integer, np.floating)) and not (
        isinstance(model, bool)
    ):
        probability = float(model)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Event {name!r}: a probability must be in [0, 1], got "
                f"{model!r}."
            )
        return probability
    no_distribution(model, f"Event {name!r}")
    if not hasattr(model, "sf"):
        raise ValueError(
            f"Event {name!r} must be a probability or a lifetime model with "
            f"sf and ff (e.g. a surpyval distribution), got {model!r}."
        )
    return model


def _sort_key(s: frozenset) -> tuple:
    return (len(s), sorted(map(str, s)))


class FaultTree:
    """A static fault tree: gates over basic events.

    Parameters
    ----------
    gates : dict
        ``{gate name: gate}``, each gate one of ``("or", inputs)`` (occurs
        when any input occurs), ``("and", inputs)`` (when all do) or
        ``("vote", k, inputs)`` (when at least ``k`` of them do, ``1 <= k <=
        len(inputs)``). ``inputs`` is a list of gate and event names. An
        input may feed several gates, but the gates must not form a loop.
    events : dict
        ``{event name: model}`` for the basic events: the probability that
        the event has occurred, a number in [0, 1], or a lifetime model whose
        ``ff(t)`` is the probability that it has occurred by time ``t``
        (a surpyval distribution such as ``Weibull`` or
        ``FixedEventProbability``, or a RePyability model with ``sf``). Gate
        and event names must differ.
    top : Hashable, optional
        The top event, a gate. By default the one gate that is no other
        gate's input.
    ccf_groups : list of CCFGroup, optional
        Common-cause groups over basic events (keyword only), as a
        ``NonRepairableRBD`` takes them: each group's members,
        events with the same model, occur together through its shared
        causes as well as on their own, split by its ``BetaFactor``,
        ``MGL`` or ``AlphaFactor`` model at every ``t``. The top event
        probability and the importance measures sum over the groups'
        shock outcomes, exactly
        as the diagram's do; the cut sets stay sets of basic events (their
        probabilities, in ``ranked_cut_sets``, take the groups in). By
        default none.

    Attributes
    ----------
    top : Hashable
        The top event.
    gates : dict
        ``{gate name: gate}`` as given, the kind in lower case.
    events : dict
        ``{event name: model}``, probabilities as floats.
    repeated_events : frozenset
        The events that feed more than one gate.
    is_fixed : bool
        Whether no event's probability depends on time, so that ``t`` may
        be left out.
    ccf_groups : list of CCFGroup
        The common-cause groups, as given.

    Raises
    ------
    ValueError
        If a gate is malformed, an input is neither a gate nor an event, a
        name is both, the gates form a loop, a gate or event is not below
        the top event, the top event is not a gate (or cannot be inferred),
        an event's model is invalid, or a common-cause group's member is
        not an event, is in two groups, or has a model the others do not.

    Examples
    --------
    Cooling is lost if both pumps fail or the valve fails:

    >>> from repyability import FaultTree
    >>> tree = FaultTree(
    ...     {
    ...         "no cooling": ("or", ["no flow", "valve"]),
    ...         "no flow": ("and", ["pump 1", "pump 2"]),
    ...     },
    ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
    ... )
    >>> tree.top
    'no cooling'
    >>> round(tree.top_event_probability(), 4)  # 1 - (1 - 0.01) * 0.95
    0.0595
    >>> [sorted(c) for c in tree.minimal_cut_sets()]
    [['valve'], ['pump 1', 'pump 2']]
    """

    def __init__(
        self,
        gates: Mapping[Hashable, Sequence[Any]],
        events: Mapping[Hashable, Any],
        top: Optional[Hashable] = None,
        *,
        ccf_groups: Optional[Sequence[Any]] = None,
    ):
        if not gates:
            raise ValueError("A fault tree needs at least one gate.")
        self._gates = {
            name: _parse_gate(name, spec) for name, spec in gates.items()
        }
        self.events = {
            name: _parse_event(name, m) for name, m in events.items()
        }
        both = set(self._gates) & set(self.events)
        if both:
            raise ValueError(
                f"Name(s) {sorted(map(str, both))} are both gates and events."
            )
        parents: Dict[Hashable, List[Hashable]] = {}
        for name, gate in self._gates.items():
            for x in gate.inputs:
                if x not in self._gates and x not in self.events:
                    raise ValueError(
                        f"Gate {name!r} has input {x!r}, which is neither a "
                        "gate nor an event."
                    )
                parents.setdefault(x, []).append(name)
        if top is None:
            roots = [g for g in self._gates if g not in parents]
            if not roots:
                # Each gate is another's input, so going up from any gate
                # comes back round: the gates form a loop (#232).
                seen_gates: set = set()
                place: Hashable = next(iter(self._gates))
                while place not in seen_gates:
                    seen_gates.add(place)
                    place = parents[place][0]
                raise ValueError(f"The gates form a loop through {place!r}.")
            if len(roots) > 1:
                raise ValueError(
                    f"Give the top event: gates {sorted(map(str, roots))} "
                    "are no gate's input."
                )
            top = roots[0]
        elif top not in self._gates:
            raise ValueError(f"The top event {top!r} must be a gate.")
        self.top = top
        self._parents = parents
        order = self._post_order()
        unused = (set(self._gates) | set(self.events)) - set(order)
        if unused:
            raise ValueError(
                f"Gate(s) or event(s) {sorted(map(str, unused))} are not "
                f"below the top event {top!r}."
            )
        self.repeated_events = frozenset(
            e for e in self.events if len(parents.get(e, ())) > 1
        )
        self.is_fixed = all(
            isinstance(m, float) or _model_is_fixed(m)
            for m in self.events.values()
        )
        self._decomposition = self._decompose(order)
        self._cut_sets: Optional[List[frozenset]] = None
        self.ccf_groups = self._validated_ccf(ccf_groups)
        self._ccf_warned: set = set()

    def _validated_ccf(self, groups) -> list:
        """The common-cause groups, checked: each a ``CCFGroup`` of basic
        events, an event in one group at most, and the members of a group
        with one model (a symmetric group, as the models assume)."""
        from repyability.rbd.ccf import as_groups, checked_groups
        from repyability.rbd.serialisation import serialise_model

        groups = as_groups(groups)
        if not groups:
            return []

        def check_member(member, group):
            if member not in self.events:
                raise ValueError(
                    f"CCF group member {member!r} is not a basic event "
                    "of the tree."
                )

        def saved(member):
            # Compared as saved, as the diagram compares its members (and,
            # as there, not at all for a model that cannot be saved).
            event = self.events[member]
            return (
                event if isinstance(event, float) else serialise_model(event)
            )

        return checked_groups(
            groups,
            check_member,
            saved,
            lambda group: (
                f"CCF group {list(group.members)} is not symmetric: its "
                "members must have the same model."
            ),
            noun="Event",
        )

    def _evaluation(
        self,
        p: dict,
        q: dict,
        size: int,
        expand: bool = True,
        groups: Optional[list] = None,
    ):
        """The tree's probabilities at the events' probabilities of not
        having occurred ``p`` and of having occurred ``q``, with the
        common-cause groups (the tree's, or ``groups`` in their place, as
        the sensitivities move a group's model) each conditioned on within
        the smallest module holding its members, or (with ``expand``)
        written out as shock events where those would multiply (#219, see
        ``rbd._ccf_modules``)."""
        from repyability.rbd import _ccf_modules

        if groups is None:
            groups = self.ccf_groups
        plans = self.__dict__.setdefault("_ccf_plans", {})
        key = (expand, _ccf_modules.Plan.key(groups))
        if key not in plans:
            plans[key] = _ccf_modules.Plan(
                self._decomposition,
                groups,
                (
                    functools.partial(self._ccf_expanded, groups=groups)
                    if expand
                    else None
                ),
            )
        return _ccf_modules.Evaluation(
            plans[key], groups, p, q, size, self._ccf_check
        )

    def _ccf_expanded(
        self, chosen: dict, groups: Optional[list] = None
    ) -> Optional[Decomposition]:
        """The tree's structure with the causes of the groups ``chosen``
        (each group's, as the members each fails) written out as events of
        their own (#219): each member's place taken by an OR gate of its
        own failure and the causes that strike it, every cause one event
        (``_ccf_modules.Shock``) under all its members' gates. None if the
        tree is then too entangled to work out."""
        from repyability.rbd._ccf_modules import Failed, Shock

        if groups is None:
            groups = self.ccf_groups
        gates = self.gates
        events: Dict[Hashable, Any] = {e: 0.5 for e in self.events}
        failed: Dict[Hashable, Any] = {}
        for g, sets in chosen.items():
            for member in groups[g].members:
                strikes = [
                    Shock(g, k)
                    for k, struck in enumerate(sets)
                    if member in struck
                ]
                for shock in strikes:
                    events[shock] = 0.5
                failed[member] = Failed(member)
        for name, gate in list(gates.items()):
            *head, inputs = gate
            gates[name] = (*head, [failed.get(x, x) for x in inputs])
        for member, gate in failed.items():
            g = next(
                i for i, group in enumerate(groups) if member in group.members
            )
            gates[gate] = (
                "or",
                [member]
                + [
                    Shock(g, k)
                    for k, struck in enumerate(chosen[g])
                    if member in struck
                ],
            )
        try:
            return FaultTree(gates, events, top=self.top)._decomposition
        except NotImplementedError:
            return None

    def _ccf_check(self, index: int, group, Q: np.ndarray) -> None:
        """Warn, once per group, where a group splitting the probability
        of failing is evaluated past ``VALIDITY`` (as the RBD does)."""
        from repyability.rbd.ccf import VALIDITY, validity_warning

        if group.model.basis != "probability" or index in self._ccf_warned:
            return
        largest = float(np.nanmax(Q, initial=0.0))
        if largest > VALIDITY:
            self._ccf_warned.add(index)
            validity_warning(group, largest)

    def _require_time_varying(self) -> None:
        """Raise if no event's probability depends on time, so that the
        time to a reliability is undefined."""
        if self.is_fixed:
            raise Refused(
                "The top event probability does not vary with time (every "
                "event is a fixed probability); the time to a reliability "
                "is undefined."
            )

    def _require_lifetimes(self) -> None:
        """Raise unless the exact mean time to the top event is defined:
        its probability must vary with time, and hold over whole
        lifetimes, which common-cause groups that split a probability do
        not (as ``NonRepairableRBD.mean`` refuses them)."""
        if self.is_fixed:
            raise Refused(
                "The top event probability does not vary with time (every "
                "event is a fixed probability): the system has no "
                "lifetimes to average."
            )
        if any(g.model.basis == "probability" for g in self.ccf_groups):
            raise NotImplementedError(
                "The exact mean time to the top event does not account for "
                "common-cause (CCF) groups that split a failure "
                "probability (the default basis): their models assume it "
                "is small (see CCFGroup), while over a whole lifetime it "
                "runs to 1. Split the failure rate instead "
                "(BetaFactor(beta, basis='rate'), or MGL(..., "
                "basis='rate')), which holds over the whole life; the "
                "diagram of the tree, to_rbd(), estimates the MTTF with "
                "the common cause left out, mean(method='simulate')."
            )

    def _event_model(self, event: Hashable) -> Any:
        """An event's model: as given, a probability as the
        ``FixedEventProbability`` of it (as ``to_rbd`` draws it), whose
        one parameter, ``p``, is its lever."""
        from surpyval import FixedEventProbability

        model = self.events[event]
        if isinstance(model, float):
            return FixedEventProbability.from_params(model)
        return model

    def _as_event(self, event: Hashable, model: Any) -> Any:
        """A moved ``model`` of ``event`` as the event is given: the
        probability of a ``FixedEventProbability`` for an event given as
        one (see ``with_levers``)."""
        if isinstance(self.events[event], float):
            return float(np.ravel(model.params)[0])
        return model

    def _lifetime_models(self) -> list:
        """The events' lifetime models (a probability has none)."""
        return [m for m in self.events.values() if not isinstance(m, float)]

    def _time_scale(self) -> float:
        """The shortest typical failure time over the events' models (see
        ``NonRepairableRBD._time_scale``), to size a step at time 0 by;
        NaN if none has one."""
        from repyability.rbd._measures import model_scale

        scales = [model_scale(m) for m in self._lifetime_models()]
        scales = [x for x in scales if np.isfinite(x) and x > 0.0]
        return min(scales) if scales else float("nan")

    @property
    def _init_args(self) -> dict:
        """The constructor's arguments, as ``with_levers`` rebuilds the
        tree from them."""
        return {
            "gates": self.gates,
            "events": dict(self.events),
            "top": self.top,
            "ccf_groups": list(self.ccf_groups) or None,
        }

    def _rbd(self) -> "NonRepairableRBD":
        """The diagram of the tree (``to_rbd``), converted once: the
        uncertainty analyses draw the events' models through it, its
        nodes being the events (a repeated event's later places repeats of
        its node) and its common-cause groups the tree's own."""
        rbd = self.__dict__.get("_rbd_cache")
        if rbd is None:
            rbd = self.__dict__["_rbd_cache"] = self.to_rbd()
        return rbd

    @property
    def gates(self) -> Dict[Hashable, tuple]:
        """``{gate name: gate}``: ``("or", inputs)``, ``("and", inputs)`` or
        ``("vote", k, inputs)``."""
        return {
            name: (
                (g.kind, g.k, list(g.inputs))
                if g.kind == "vote"
                else (g.kind, list(g.inputs))
            )
            for name, g in self._gates.items()
        }

    def __repr__(self) -> str:
        return (
            f"FaultTree(top={self.top!r}, {len(self._gates)} gates, "
            f"{len(self.events)} events)"
        )

    # -- structure ---------------------------------------------------------

    def _post_order(self) -> list:
        """The gates and events below the top, each after its inputs; a
        ValueError if the gates form a loop."""
        order: list = []
        state: Dict[Hashable, int] = {}  # 1: being visited, 2: done
        stack: list = [(self.top, False)]
        while stack:
            x, expanded = stack.pop()
            if expanded:
                state[x] = 2
                order.append(x)
                continue
            if state.get(x) == 2:
                continue
            if state.get(x) == 1:
                raise ValueError(f"The gates form a loop through {x!r}.")
            if x in self.events:
                state[x] = 2
                order.append(x)
                continue
            state[x] = 1
            stack.append((x, True))
            for c in reversed(self._gates[x].inputs):
                if state.get(c) != 2:
                    stack.append((c, False))
        return order

    def _decompose(self, order: list) -> Decomposition:
        """The tree as the exact engine's decomposition, over the system
        *working* (no event occurring): each gate below which nothing is
        shared with the rest of the tree is a module; the rest, if any, is
        a core given by its decision diagram, built from the gates over the
        terms they share (see ``_core``)."""
        gates, events = self._gates, self.events
        # A gate is self-contained when everything below it has one parent.
        own: Dict[Hashable, bool] = {}
        for x in order:
            if x in gates:
                own[x] = all(
                    len(self._parents[c]) == 1 and (c in events or own[c])
                    for c in gates[x].inputs
                )
        terms: list = []
        position: Dict[Hashable, int] = {}
        for x in order:
            if x in events:
                position[x] = len(terms)
                terms.append((NODE, x))
                continue
            if not own[x]:
                continue
            children = [position[c] for c in gates[x].inputs]
            n = len(children)
            working = n - gates[x].k + 1  # inputs that must not occur
            if n == 1:
                position[x] = children[0]
                continue
            position[x] = len(terms)
            if working == n:
                terms.append((SERIES, tuple(children)))
            elif working == 1:
                terms.append((PARALLEL, tuple(children)))
            else:
                terms.append((KOON, tuple(children), working))
        # Terms of self-contained gates below a shared one stay unused: drop
        # them, keeping the order (children before parents).
        if own[self.top]:
            return _pruned(terms, [position[self.top]])
        steps, root = self._core(position)
        return _pruned(
            terms, sorted({pivot for pivot, _, _ in steps}), (steps, root)
        )

    def _core(self, position: Dict[Hashable, int]) -> Tuple[list, int]:
        """The decision diagram of the top event not occurring, over the
        terms of the events and self-contained gates the shared gates are
        built from (``position``), as a plan (see ``modular``): a gate does
        not occur while at least ``n - k + 1`` of its ``n`` inputs do not.
        The terms are decided in the order a walk down the tree from the
        top first meets them, which keeps a shared event's gates together.
        Listing the core's minimal path sets instead (before #171) took
        minutes for an OR of fifty ANDs over twenty-five shared events, as
        they multiply with the shared events; the diagram is found in a
        fraction of a second."""
        gates = self._gates
        variable: Dict[int, int] = {}
        stack, seen = [self.top], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if x in position:
                variable.setdefault(position[x], len(variable))
                continue
            stack.extend(reversed(gates[x].inputs))
        term_of = {v: term for term, v in variable.items()}
        diagrams = OrderedBDD(
            limit=DIAGRAM_LIMIT,
            message=(
                "The repeated events tie together too much of the fault tree "
                f"to work it out exactly (its decision diagram has more than "
                f"{DIAGRAM_LIMIT:,} nodes): convert it with to_rbd() and "
                "simulate the diagram, NonRepairableRBD.random or "
                "mean(method='simulate')."
            ),
        )
        works: Dict[Hashable, int] = {}
        for x in self._post_order():
            if x in position:
                if position[x] in variable:
                    works[x] = diagrams.node(variable[position[x]], 0, 1)
                continue
            if x not in seen:
                continue
            gate = gates[x]
            inputs = [works[c] for c in gate.inputs]
            needed = len(inputs) - gate.k + 1
            if needed == len(inputs):
                works[x] = diagrams.conjunction(inputs)
            elif needed == 1:
                works[x] = diagrams.disjunction(inputs)
            else:
                works[x] = diagrams.at_least(inputs, needed)
        steps, (root,) = diagrams.plan([works[self.top]], lambda v: term_of[v])
        return steps, root

    # -- evaluation --------------------------------------------------------

    def _times(self, t) -> Tuple[np.ndarray, bool]:
        if t is None:
            if not self.is_fixed:
                raise ValueError(
                    "t is required: an event's probability depends on time."
                )
            t = 1.0
        scalar = np.ndim(t) == 0
        return np.atleast_1d(real_array(t, "t")), scalar

    def _event_probabilities(self, t: np.ndarray) -> Tuple[dict, dict]:
        """Each event's probability of not having occurred (``p``) and of
        having occurred (``q``) by each of the times ``t``."""
        p, q = {}, {}
        for name, model in self.events.items():
            if isinstance(model, float):
                q[name] = np.full(len(t), model)
                p[name] = 1.0 - q[name]
                continue
            p[name] = np.broadcast_to(
                np.asarray(model.sf(t), dtype=float), t.shape
            ).copy()
            if hasattr(model, "ff"):
                q[name] = np.broadcast_to(
                    np.asarray(model.ff(t), dtype=float), t.shape
                ).copy()
            else:
                q[name] = 1.0 - p[name]
        return p, q

    def _top(self, p: dict, q: dict, size: int) -> np.ndarray:
        """The top event probability from the events' probabilities, with
        the common-cause groups."""
        if not self.ccf_groups:
            return self._independent_top(p, q, size)
        _, total = self._evaluation(p, q, size).system()
        return np.broadcast_to(np.asarray(total, dtype=float), (size,))

    def _independent_top(self, p: dict, q: dict, size: int) -> np.ndarray:
        """The top event probability of independent events."""
        _, fails = self._decomposition.probabilities(
            p, q, shape=size, works=False, fails=True
        )
        return np.broadcast_to(np.asarray(fails, dtype=float), (size,))

    @staticmethod
    def _out(values, scalar: bool):
        if isinstance(values, dict):
            return {
                k: (float(np.asarray(v).reshape(-1)[0]) if scalar else v)
                for k, v in values.items()
            }
        return float(np.asarray(values).reshape(-1)[0]) if scalar else values

    def top_event_probability(self, t: Optional[ArrayLike] = None):
        """The probability that the top event has occurred.

        Exact, repeated events included. With lifetime models it is the
        probability that the top event has occurred by time ``t``: the
        unreliability of the system the tree describes.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        float or numpy.ndarray
            The probability, a float for a number ``t`` and an array for an
            array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on time.

        Examples
        --------
        A power supply shared by two redundant channels (a repeated event).
        Treating its two appearances as independent would give 0.0119, too
        low: one supply failure takes out both channels.

        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {
        ...         "no output": ("and", ["channel A", "channel B"]),
        ...         "channel A": ("or", ["supply", "pump A"]),
        ...         "channel B": ("or", ["supply", "pump B"]),
        ...     },
        ...     {"supply": 0.01, "pump A": 0.1, "pump B": 0.1},
        ... )
        >>> round(tree.top_event_probability(), 4)  # 0.01 + 0.99 * 0.1 ** 2
        0.0199
        """
        t, scalar = self._times(t)
        p, q = self._event_probabilities(t)
        return self._out(self._top(p, q, len(t)), scalar)

    def ff(self, t: Optional[ArrayLike] = None):
        """The probability that the top event has occurred by ``t``: the
        unreliability of the system the tree describes, by the name an
        RBD's ``ff`` has (the same as ``top_event_probability``).

        Parameters
        ----------
        t : array_like, optional
            Time/s, as for ``top_event_probability``.

        Returns
        -------
        float or numpy.ndarray
            The probability.
        """
        return self.top_event_probability(t)

    def sf(self, t: Optional[ArrayLike] = None):
        """The probability that the top event has not occurred by ``t``: the
        reliability of the system the tree describes, as an RBD's ``sf``
        gives it. Worked out in its own right, not as one less ``ff``, so a
        small one keeps its precision.

        Parameters
        ----------
        t : array_like, optional
            Time/s, as for ``top_event_probability``.

        Returns
        -------
        float or numpy.ndarray
            The probability.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2}
        ... )
        >>> round(tree.sf(), 4), round(tree.ff(), 4)
        (0.98, 0.02)
        """
        t, scalar = self._times(t)
        p, q = self._event_probabilities(t)
        size = len(t)
        if self.ccf_groups:
            total, _ = self._evaluation(p, q, size).system()
        else:
            total, _ = self._decomposition.probabilities(
                p, q, shape=size, works=True, fails=False
            )
        values = np.broadcast_to(np.asarray(total, dtype=float), (size,))
        return self._out(values, scalar)

    def _sf_and_ff(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The probabilities that the top event has not occurred and that
        it has by the ``times``, each worked out in its own right (so a
        small one keeps its precision), from one evaluation."""
        p, q = self._event_probabilities(times)
        size = len(times)
        if self.ccf_groups:
            works, fails = self._evaluation(p, q, size).system()
        else:
            works, fails = self._decomposition.probabilities(
                p, q, shape=size, works=True, fails=True
            )
        return (
            np.broadcast_to(np.asarray(works, dtype=float), (size,)),
            np.broadcast_to(np.asarray(fails, dtype=float), (size,)),
        )

    def Hf(self, t: Optional[ArrayLike] = None):
        """The cumulative hazard ``H(t) = -ln R(t)`` of the system the tree
        describes, ``R`` its [`sf`][repyability.FaultTree.sf]: exact (so
        it honours the common-cause groups), ``+inf`` where the top event
        is certain. While the top event is less likely than not it is
        ``-log1p(-F(t))`` from its probability ``F``, so that a small one
        keeps its precision, as ``NonRepairableRBD.Hf`` has it.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        float or numpy.ndarray
            The cumulative hazard: a float for a number ``t``, an array
            for an array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on
            time.

        Examples
        --------
        One exponential event's cumulative hazard is its rate times ``t``:

        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c"])},
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> round(tree.Hf(50), 6)
        0.5
        """
        from repyability.rbd._measures import cumulative_hazard

        times, scalar = self._times(t)
        return self._out(cumulative_hazard(*self._sf_and_ff(times)), scalar)

    def _density(self, times: np.ndarray, dx: float) -> np.ndarray:
        """The density of the time to the top event at ``times``, by
        central differences of the exact probabilities (see ``df``)."""
        from repyability.rbd._measures import (
            density_by_differences,
            difference_steps,
        )

        return density_by_differences(
            self._sf_and_ff,
            times,
            difference_steps(times, dx, self._time_scale),
        )

    def df(self, t: Optional[ArrayLike] = None, dx: float = 1e-6):
        """The density of the time to the top event, ``f(t) = -dR/dt``.

        By a central finite difference of the exact
        [`sf`][repyability.FaultTree.sf] (so it honours the common-cause
        groups), with step ``dx * |t|``, relative to ``t`` (at 0, ``dx``
        of the shortest of the events' typical failure times), as
        ``NonRepairableRBD.df`` takes the diagram's: the difference is of
        the top event probability while it is the smaller, which keeps
        its precision, the lower point is clipped at 0, and a negative
        result (rounding) is clipped to 0. It assumes the probability is
        smooth near ``t``, which does not hold where an event's model has a
        kink or a step; a tree of probabilities alone has a density of 0.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.
        dx : float, optional
            The relative step of the difference, by default 1e-6.

        Returns
        -------
        float or numpy.ndarray
            The density: a float for a number ``t``, an array for an
            array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on
            time.

        Examples
        --------
        One exponential event: ``f(t) = rate * exp(-rate * t)``.

        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c"])},
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> round(tree.df(50), 6)
        0.006065
        """
        times, scalar = self._times(t)
        return self._out(self._density(times, dx), scalar)

    def hf(self, t: Optional[ArrayLike] = None, dx: float = 1e-6):
        """The hazard rate of the top event, ``h(t) = f(t) / R(t)``: the
        numerical density of [`df`][repyability.FaultTree.df] over the
        exact [`sf`][repyability.FaultTree.sf] (both honour the
        common-cause groups), ``+inf`` where the top event is certain.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.
        dx : float, optional
            The relative step of the difference, passed to ``df``, by
            default 1e-6.

        Returns
        -------
        float or numpy.ndarray
            The hazard rate: a float for a number ``t``, an array for an
            array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on
            time.

        Examples
        --------
        One exponential event has a constant hazard rate:

        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c"])},
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> round(tree.hf(50), 6)
        0.01
        """
        times, scalar = self._times(t)
        sf, _ = self._sf_and_ff(times)
        density = self._density(times, dx)
        return self._out(
            np.divide(
                density, sf, out=np.full_like(density, np.inf), where=sf > 0
            ),
            scalar,
        )

    def mean(self) -> float:
        """The mean time to the top event: the MTTF of the system the tree
        describes.

        Exact to about ``1e-10``, relative: the area under the probability
        that the top event has not occurred, ``integral from 0 to infinity
        of R(t) dt`` with ``R`` the exact [`sf`][repyability.FaultTree.sf],
        integrated by adaptive quadrature on pieces that start at the
        events' models' kinks, as ``NonRepairableRBD.mean`` integrates the
        diagram's (it is as exact as the events' models' own
        probabilities: see ``analysis_routes``). ``inf`` if the top event
        may never occur (an event of a limited failure population). An
        event given as a probability has occurred, or not, from the start
        and for good, as ``sf`` takes it. A common-cause group that splits
        the failure rate (``basis="rate"``) is included; one splitting a
        probability is refused, as the diagram refuses it.

        Returns
        -------
        float
            The mean time to the top event (``inf`` if it may never
            occur).

        Raises
        ------
        ValueError
            If every event is a probability, so that the top event has no
            time.
        NotImplementedError
            With a common-cause group that splits a failure probability
            (the default basis): its model assumes it is small (see
            [`CCFGroup`][repyability.CCFGroup]), while over a whole
            lifetime it runs to 1.

        Examples
        --------
        Either of two exponential events, with rates 0.01 and 0.03, is a
        failure rate of 0.04: a mean time of 25.

        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> E = surv.Exponential.from_params
        >>> tree = FaultTree(
        ...     {"top": ("or", ["a", "b"])}, {"a": E([0.01]), "b": E([0.03])}
        ... )
        >>> round(tree.mean(), 8)
        25.0
        """
        from repyability.rbd._mean_lifetime import (
            mean_lifetime,
            model_kinks,
            model_knots,
        )

        self._require_lifetimes()
        models = self._lifetime_models()
        return mean_lifetime(
            lambda t: self.sf(t),
            [model_knots(m) for m in models],
            [model_kinks(m) for m in models],
        )

    def time_to_reliability(self, target, upper_bound: Optional[float] = None):
        """The time at which the probability that the top event has not
        occurred falls to ``target``: the inverse of
        [`sf`][repyability.FaultTree.sf] (so it honours the common-cause
        groups), found by Brent's method between 0 and an upper bound
        found by doubling from 1 unless given, as
        ``NonRepairableRBD.time_to_reliability`` finds the diagram's.

        Parameters
        ----------
        target : float or array_like
            The reliability to solve for, in (0, 1), or several.
        upper_bound : float, optional
            A time at which the reliability is below ``target``; by
            default found by doubling.

        Returns
        -------
        float or numpy.ndarray
            The time at which ``sf(t) == target``: an array of one for
            each target, if several are given.

        Raises
        ------
        ValueError
            If ``target`` is not in (0, 1); if every event is a
            probability (the reliability is constant in time); if
            ``target`` exceeds the reliability at ``t = 0``, so it is
            never reached; if no upper bound is found (after 1000
            doublings); or if the reliability at ``upper_bound`` is still
            above ``target``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c"])},
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> round(tree.time_to_reliability(0.9), 2)
        32.46
        """
        from repyability.rbd._measures import invert_reliability

        self._require_time_varying()
        return invert_reliability(
            lambda t: float(self.sf(t)), target, upper_bound
        )

    def bx_life(self, x, upper_bound: Optional[float] = None):
        """The Bx life: the time by which the top event has occurred in
        ``x`` percent of systems, ``time_to_reliability(1 - x / 100)``.
        ``bx_life(10)`` is the B10 life.

        Parameters
        ----------
        x : float or array_like
            The percentage in which the top event has occurred, in
            (0, 100), or several.
        upper_bound : float, optional
            As for ``time_to_reliability``.

        Returns
        -------
        float or numpy.ndarray
            The time (one for each ``x``, if several are given).

        Raises
        ------
        ValueError
            If ``x`` is not in (0, 100), or as for
            ``time_to_reliability``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c"])},
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> round(tree.bx_life(10), 2)
        32.46
        """
        x = np.asarray(x, dtype=float) if np.ndim(x) else x
        if not np.all((0.0 < x) & (x < 100.0)):
            raise ValueError("x must be a percentage in (0, 100).")
        return self.time_to_reliability(1.0 - x / 100.0, upper_bound)

    def occurs(self, events: Collection[Hashable]) -> bool:
        """Whether the top event occurs when exactly ``events`` have.

        Parameters
        ----------
        events : Collection[Hashable]
            The basic events that have occurred; every other has not.

        Returns
        -------
        bool
            Whether the top event occurs.

        Raises
        ------
        ValueError
            If an event is unknown.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("vote", 2, ["a", "b", "c"])},
        ...     {"a": 0.1, "b": 0.1, "c": 0.1},
        ... )
        >>> tree.occurs({"a"}), tree.occurs({"a", "c"})
        (False, True)
        """
        # A bare string is one event's name, not its characters (#225).
        occurred: set = {events} if isinstance(events, str) else set(events)
        unknown = occurred - set(self.events)
        if unknown:
            raise ValueError(f"Unknown event(s) {sorted(map(str, unknown))}.")
        return not self._decomposition.works(
            {e: e not in occurred for e in self.events}
        )

    # -- cut sets ----------------------------------------------------------

    def minimal_cut_sets(self) -> List[frozenset]:
        """The minimal cut sets: the smallest sets of basic events whose
        joint occurrence makes the top event occur.

        Returns
        -------
        list of frozenset
            The minimal cut sets, smallest first (then by name).

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("vote", 2, ["a", "b", "c"])},
        ...     {"a": 0.1, "b": 0.1, "c": 0.1},
        ... )
        >>> [sorted(c) for c in tree.minimal_cut_sets()]
        [['a', 'b'], ['a', 'c'], ['b', 'c']]
        """
        if self._cut_sets is None:
            self._cut_sets = sorted(
                self._decomposition.cut_sets(), key=_sort_key
            )
        return list(self._cut_sets)

    def get_min_cut_sets(self) -> set:
        """The minimal cut sets as a set, as an RBD's ``get_min_cut_sets``
        gives them (``minimal_cut_sets`` lists them).

        Returns
        -------
        set of frozenset
            The minimal cut sets.
        """
        return set(self.minimal_cut_sets())

    def get_min_path_sets(self) -> set:
        """The minimal path sets as a set, as an RBD's
        ``get_min_path_sets(include_in_out_nodes=False)`` gives them
        (``minimal_path_sets`` lists them).

        Returns
        -------
        set of frozenset
            The minimal path sets.
        """
        return set(self.minimal_path_sets())

    def minimal_path_sets(self) -> List[frozenset]:
        """The minimal path sets: the smallest sets of basic events whose
        not occurring keeps the top event from occurring.

        Returns
        -------
        list of frozenset
            The minimal path sets, smallest first (then by name).

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["a", "both"]), "both": ("and", ["b", "c"])},
        ...     {"a": 0.1, "b": 0.1, "c": 0.1},
        ... )
        >>> [sorted(p) for p in tree.minimal_path_sets()]
        [['a', 'b'], ['a', 'c']]
        """
        return sorted(self._decomposition.path_sets(), key=_sort_key)

    def ranked_cut_sets(
        self, t: Optional[float] = None
    ) -> List[Tuple[frozenset, float]]:
        """The minimal cut sets with their probabilities, most likely first.

        A cut set's probability is that all its events occur: the product
        of theirs, or with common-cause groups, as the groups' shared
        causes add to it (a cut set of two members of a group is often the
        largest). The largest say which combinations of failures dominate
        the top event.

        Parameters
        ----------
        t : float, optional
            The time, a single number. May be left out when no event's
            probability depends on time.

        Returns
        -------
        list of tuple
            ``(cut set, probability)``, in decreasing probability (ties
            smallest first).

        Raises
        ------
        ValueError
            If ``t`` is not a single number, or is left out and an event's
            probability depends on time.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.005},
        ... )
        >>> [(sorted(c), round(p, 4)) for c, p in tree.ranked_cut_sets()]
        [(['pump 1', 'pump 2'], 0.01), (['valve'], 0.005)]
        """
        if t is not None and np.ndim(t) != 0:
            raise ValueError("t must be a single number.")
        from repyability.rbd._ccf_modules import (
            expected_product,
            group_outcomes,
        )

        times, _ = self._times(t)
        p, q = self._event_probabilities(times)
        outcomes = group_outcomes(self.ccf_groups, p, q, self._ccf_check)
        ranked = [
            (
                c,
                float(
                    np.asarray(expected_product(c, q, outcomes)).reshape(-1)[0]
                ),
            )
            for c in self.minimal_cut_sets()
        ]
        order = sorted(range(len(ranked)), key=lambda i: (-ranked[i][1], i))
        return [ranked[i] for i in order]

    # -- importance --------------------------------------------------------

    def _held(self, p: dict, q: dict, size: int) -> Tuple[Any, dict, dict]:
        """The top event probability of independent events, and for each
        event that probability given that it has occurred and given that
        it has not."""
        top = self._independent_top(p, q, size)
        occurred, not_occurred = {}, {}
        for e in self.events:
            p_e, q_e = dict(p), dict(q)
            p_e[e], q_e[e] = np.zeros(size), np.ones(size)
            occurred[e] = self._independent_top(p_e, q_e, size)
            p_e[e], q_e[e] = np.ones(size), np.zeros(size)
            not_occurred[e] = self._independent_top(p_e, q_e, size)
        return top, occurred, not_occurred

    def _conditioned(self, t) -> Tuple[np.ndarray, dict, dict, dict, bool]:
        """The top event probability, and for each event the top event
        probability given that it has occurred and given that it has not,
        with the events' probabilities of occurring."""
        times, scalar = self._times(t)
        p, q = self._event_probabilities(times)
        size = len(times)
        if not self.ccf_groups:
            top, occurred, not_occurred = self._held(p, q, size)
            return top, occurred, not_occurred, q, scalar
        # With common-cause groups, as the diagram's measures: a member's
        # occurrence says something of the others', so the top event's
        # probability given it is a sum over its group's shock outcomes,
        # each weighed by the member's chance of that state in it, the
        # other groups conditioned on within their modules (#219); an event
        # outside the groups is held, as without them.
        evaluation = self._evaluation(p, q, size)
        _, top_sum = evaluation.system()
        occurred, not_occurred, marginal = {}, {}, dict(q)
        for e in self.events:
            given = evaluation.given((e,))
            (_, _, not_occurred[e]), (chance, _, occurred[e]) = (
                given[True,],
                given[False,],
            )
            if e in evaluation.group_of:
                marginal[e] = np.asarray(chance, dtype=float)
        top = np.broadcast_to(np.asarray(top_sum, dtype=float), (size,))
        return top, occurred, not_occurred, marginal, scalar

    def birnbaum_importance(self, t: Optional[ArrayLike] = None) -> dict:
        """Birnbaum importance of each basic event.

        In the guide's Greeks it is *delta*: how far the system moves with each
        basic event (see [Sensitivities: the Greeks](../guide/greeks.md)).

        ``P(top | e occurred) - P(top | e did not)``: how much the top event
        probability depends on the event, the rate at which it rises with
        the event's probability.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        dict
            ``{event: importance}``: floats for a number ``t``, arrays for
            an array.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> {e: round(v, 4) for e, v in tree.birnbaum_importance().items()}
        {'pump 1': 0.095, 'pump 2': 0.095, 'valve': 0.99}
        """
        _, occurred, not_occurred, _, scalar = self._conditioned(t)
        return self._out(
            {e: occurred[e] - not_occurred[e] for e in self.events}, scalar
        )

    def criticality_importance(self, t: Optional[ArrayLike] = None) -> dict:
        """Criticality importance of each basic event.

        ``I_B(e) * P(e) / P(top)``: the probability that the event has
        occurred and is critical, given that the top event has occurred, so
        the share of the top event it accounts for. ``nan`` where the top
        event cannot occur.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        dict
            ``{event: importance}``: floats for a number ``t``, arrays for
            an array.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> {e: round(v, 4) for e, v in tree.criticality_importance().items()}
        {'pump 1': 0.1597, 'pump 2': 0.1597, 'valve': 0.8319}
        """
        top, occurred, not_occurred, q, scalar = self._conditioned(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {
                e: (occurred[e] - not_occurred[e]) * q[e] / top
                for e in self.events
            }
        return self._out(out, scalar)

    def risk_achievement_worth(self, t: Optional[ArrayLike] = None) -> dict:
        """Risk achievement worth of each basic event.

        ``P(top | e occurred) / P(top)``: how many times more likely the top
        event is once the event has occurred.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        dict
            ``{event: importance}``: floats for a number ``t``, arrays for
            an array.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> {e: round(v, 2) for e, v in tree.risk_achievement_worth().items()}
        {'pump 1': 2.44, 'pump 2': 2.44, 'valve': 16.81}
        """
        top, occurred, _, _, scalar = self._conditioned(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {e: occurred[e] / top for e in self.events}
        return self._out(out, scalar)

    def risk_reduction_worth(self, t: Optional[ArrayLike] = None) -> dict:
        """Risk reduction worth of each basic event.

        ``P(top) / P(top | e did not occur)``: the factor by which the top
        event would be less likely if the event could not occur.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        dict
            ``{event: importance}``: floats for a number ``t``, arrays for
            an array (``inf`` where preventing the event would make the top
            event impossible).

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> {e: round(v, 3) for e, v in tree.risk_reduction_worth().items()}
        {'pump 1': 1.19, 'pump 2': 1.19, 'valve': 5.95}
        """
        top, _, not_occurred, _, scalar = self._conditioned(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {e: top / not_occurred[e] for e in self.events}
        return self._out(out, scalar)

    def fussell_vesely(
        self, t: Optional[ArrayLike] = None, method: str = "exact"
    ) -> dict:
        """Fussell-Vesely importance of each basic event.

        The share of the top event probability carried by the minimal cut
        sets that contain the event: the probability that one of them has
        occurred (every event in it), over the top event probability, as
        ``NonRepairableRBD.fussell_vesely`` computes it. It is exact, and
        between 0 and 1, by default; ``method="rare_event"`` sums the cut
        sets' probabilities instead (the rare-event approximation of their
        union, as many PRA tools report it), which can exceed 1 when the
        events are not rare. ``nan`` where the top event cannot occur.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.
        method : str, optional
            ``"exact"`` (the default) or ``"rare_event"``.

        Returns
        -------
        dict
            ``{event: importance}``: floats for a number ``t``, arrays for
            an array.

        Raises
        ------
        ValueError
            If ``method`` is not "exact" or "rare_event", or ``t`` is left
            out and an event's probability depends on time.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> {e: round(v, 4) for e, v in tree.fussell_vesely().items()}
        {'pump 1': 0.1681, 'pump 2': 0.1681, 'valve': 0.8403}
        """
        one_of("method", method, ("exact", "rare_event"))
        from repyability.rbd._ccf_modules import (
            expected_product,
            group_outcomes,
        )

        times, scalar = self._times(t)
        p, q = self._event_probabilities(times)
        size = len(times)
        top = self._top(p, q, size)
        share = {e: np.zeros(size) for e in self.events}
        # The common-cause groups each conditioned on within the smallest
        # module holding its members (#219).
        if method == "exact":
            failed = (
                self._evaluation(p, q, size, expand=False).failed_cut_sets()
                if self.ccf_groups
                else self._decomposition.failed_cut_sets(p, q, shape=size)
            )
            for e, probability in failed.items():
                share[e] = share[e] + probability
        else:
            outcomes = group_outcomes(self.ccf_groups, p, q, self._ccf_check)
            for cut in self.minimal_cut_sets():
                probability = expected_product(cut, q, outcomes)
                for e in cut:
                    share[e] = share[e] + probability
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {e: share[e] / top for e in self.events}
        return self._out(out, scalar)

    def differential_importance(
        self,
        t: Optional[ArrayLike] = None,
        *,
        change: str = "uniform",
        groups: Optional[Mapping[Hashable, Collection[Hashable]]] = None,
    ) -> dict:
        """Each basic event's share of the change in the top event
        probability when they all change together: the differential
        importance measure (DIM, Borgonovo & Apostolakis, 2001).

        In the guide's Greeks it is *DIM*, the shares of a change (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        ``DIM_e = dP/dq_e dq_e / sum_f dP/dq_f dq_f``, with ``q_e`` the
        events' probabilities, so the shares add up to 1, and a group's
        share is the sum of its members' (``groups``): what share of a
        possible gain lies in the pumps, say. ``change="uniform"`` moves
        every probability by as much, which shares out the Birnbaum
        importance; ``"proportional"`` each by the same fraction of
        itself, which shares out the criticality importance. With
        common-cause groups, a member's measures are conditioned through
        the groups' outcomes, as ``birnbaum_importance`` and
        ``criticality_importance`` work them out. NaN where the shares'
        total is 0 (the top event cannot occur, or cannot be helped).

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.
        change : str, optional
            ``"uniform"`` (the default) or ``"proportional"``.
        groups : dict, optional
            ``{name: events}``: each group's share, the sum of its events',
            instead of each event's.

        Returns
        -------
        dict
            ``{event: share}``, or ``{group: share}`` with ``groups``:
            floats for a number ``t``, arrays for an array.

        Raises
        ------
        ValueError
            If ``change`` is not "uniform" or "proportional", a group names
            an unknown event, or ``t`` is left out and an event's
            probability depends on time.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> shares = tree.differential_importance(
        ...     groups={"pumps": ["pump 1", "pump 2"], "valve": ["valve"]}
        ... )
        >>> {g: round(v, 4) for g, v in shares.items()}
        {'pumps': 0.161, 'valve': 0.839}
        """
        from repyability.rbd._differential import CHANGES, shares

        if change not in CHANGES:
            raise ValueError(
                f"change must be 'uniform' (every event's probability moved "
                f"by as much) or 'proportional' (each by the same fraction "
                f"of itself), got {change!r}."
            )
        if change == "uniform":
            values = self.birnbaum_importance(t)
        else:
            values = self.criticality_importance(t)
        return shares(values, groups, scalar=t is None or np.ndim(t) == 0)

    def joint_importance(self, t: Optional[ArrayLike] = None) -> dict:
        """The joint (second-order) importance of each pair of basic events:
        whether preventing the two together is worth more than
        preventing each.

        In the guide's Greeks it is *gamma*: complements or substitutes (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        ``JRI(e, f) = -d2P / dq_e dq_f``, ``P`` the top event probability
        and ``q`` the events' probabilities: how much event ``f``'s
        Birnbaum importance falls when event ``e`` goes from occurring to
        not. It is the joint importance of the diagram the tree describes
        (see ``to_rbd``), in its reliability. Positive, the two are
        complements, as under an OR gate: preventing either makes
        preventing the other worth more. Negative, they are substitutes, as
        under an AND gate: either one prevented keeps the gate from
        occurring. Exact, from the top event probability with both events
        held, as the tree is multilinear in the events. With common-cause
        groups, the top event probability given each of the pair occurring
        and not is conditioned on their states, as ``birnbaum_importance``
        conditions on one (two members of one group on their joint state),
        and an event outside the groups is held.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        dict
            ``{(e, f): JRI}`` for each pair once, its events' names in
            order as text, and found either way round (the measure is
            symmetric): floats for a number ``t``, arrays for an array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on
            time.

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> joint = tree.joint_importance()
        >>> round(joint[("pump 1", "pump 2")], 4)  # substitutes
        -0.95
        >>> round(joint[("pump 1", "valve")], 4)  # complements
        0.1
        """
        from repyability.rbd.rbd import Pairs

        times, scalar = self._times(t)
        p, q = self._event_probabilities(times)
        size = len(times)
        events = list(self.events)
        if self.ccf_groups:
            from repyability.rbd._ccf_modules import joint_importance

            evaluation = self._evaluation(p, q, size)
            pairs = Pairs.of(
                events, lambda e, f: joint_importance(evaluation, e, f)
            )
            return Pairs(self._out(pairs, scalar))
        ones, zeros = np.ones(size), np.zeros(size)

        def importance(p_: dict, q_: dict, of: list) -> dict:
            out = {}
            for f in of:
                p_f, q_f = dict(p_), dict(q_)
                p_f[f], q_f[f] = zeros, ones
                occurred = self._top(p_f, q_f, size)
                p_f[f], q_f[f] = ones, zeros
                out[f] = occurred - self._top(p_f, q_f, size)
            return out

        pairs = Pairs()
        for k, e in enumerate(events):
            rest = events[k + 1 :]  # noqa: E203
            if not rest:
                continue
            p_e, q_e = dict(p), dict(q)
            p_e[e], q_e[e] = zeros, ones
            occurring = importance(p_e, q_e, rest)
            p_e[e], q_e[e] = ones, zeros
            prevented = importance(p_e, q_e, rest)
            for f in rest:
                pairs[Pairs.oriented(e, f)] = prevented[f] - occurring[f]
        return Pairs(self._out(pairs, scalar))

    def structural_importance(self) -> Dict[Hashable, float]:
        """The structural importance of each basic event: the fraction of
        the states of the other events in which it is pivotal, the top
        event occurring when it has and not when it has not.

        It is the Birnbaum importance with every event's probability at
        1/2, so it depends on the gates alone, not on the events' models
        or the common-cause groups: useful at design time, before any
        data. It is the diagram's ``structural_importance`` (see
        ``to_rbd``).

        Returns
        -------
        dict
            ``{event: importance}``, each in [0, 1].

        Examples
        --------
        The valve is pivotal unless both pumps have failed (three states
        of four), and a pump when the other has failed and the valve has
        not (one of four):

        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> tree.structural_importance()
        {'pump 1': 0.25, 'pump 2': 0.25, 'valve': 0.75}
        """
        half = {e: np.full(1, 0.5) for e in self.events}
        _, occurred, not_occurred = self._held(half, half, 1)
        return {
            e: float(np.ravel(occurred[e] - not_occurred[e])[0])
            for e in self.events
        }

    def improvement_potential(self, t: Optional[ArrayLike] = None) -> dict:
        """The improvement potential of each basic event: ``P(top) - P(top
        | e did not occur)``, how far the top event probability would fall
        if the event could not occur. It is ``I_B(e) * q_e``, the Birnbaum
        importance times the event's probability, and is worked out as
        that product, so a small one keeps its precision; with common-cause
        groups, a member's is conditioned through the groups' outcomes, as
        ``birnbaum_importance`` is.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.

        Returns
        -------
        dict
            ``{event: potential}``: floats for a number ``t``, arrays for
            an array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on
            time.

        Examples
        --------
        A valve that could not fail would take the top event from 0.0595
        to the pumps' 0.01:

        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> {e: round(v, 4) for e, v in tree.improvement_potential().items()}
        {'pump 1': 0.0095, 'pump 2': 0.0095, 'valve': 0.0495}
        """
        _, occurred, not_occurred, q, scalar = self._conditioned(t)
        return self._out(
            {e: (occurred[e] - not_occurred[e]) * q[e] for e in self.events},
            scalar,
        )

    def _rate_parts(self, t: np.ndarray) -> Dict[Any, np.ndarray]:
        """Each event's part in the rate of change of the probability that
        the top event has not occurred, at the times ``t``, and each
        common-cause group's under the tuple of its members (see
        ``NonRepairableRBD.reliability_rate`` and ``rate_parts``): a
        probability has no density, so no part."""
        from repyability.rbd._measures import density as model_density
        from repyability.rbd._measures import rate_parts

        _, occurred, not_occurred, _, _ = self._conditioned(t)
        size = len(t)

        def density(event, x: np.ndarray) -> np.ndarray:
            model = self.events[event]
            if isinstance(model, float):
                return np.zeros(x.shape)
            return model_density(model, x)

        return rate_parts(
            {e: occurred[e] - not_occurred[e] for e in self.events},
            density,
            t,
            self.ccf_groups,
            inputs=self._event_probabilities,
            fails=lambda p, q: self._top(p, q, size),
            scale=self._time_scale,
        )

    def barlow_proschan_importance(
        self, t: Optional[ArrayLike] = None
    ) -> dict:
        """Each basic event's Barlow-Proschan importance: the probability
        that the top event is caused by the event's occurring (Barlow &
        Proschan, 1975), given that it has occurred by ``t``, or ever.

        In the guide's Greeks it is *theta*, integrated: who caused the
        failures (see [Sensitivities: the Greeks](../guide/greeks.md)).

        An event's occurring causes the top event when the event is
        critical then, so the probability that the top event has occurred
        by ``t`` through event ``e`` is ``integral_0^t I_B^e(s) f_e(s)
        ds``, its Birnbaum importance times its model's density, and its
        share of the sum over the events is its importance: the shares add
        up to 1. A common-cause group's part is its members' together,
        under the tuple of their names (the top event probability's change
        with the members' lives moved on, by differences). The integrals
        are by adaptive quadrature to about ``1e-9``, as
        ``NonRepairableRBD.barlow_proschan_importance`` works out the
        diagram's. An event given as a probability has no density, so no
        share.

        Parameters
        ----------
        t : array_like, optional
            Time/s by which the top event has occurred, a number or an
            array; by default over the whole life.

        Returns
        -------
        dict
            ``{event: share}`` (and each common-cause group's): floats for
            a number ``t`` or the whole life, arrays for an array. NaN
            where the top event cannot have occurred.

        Raises
        ------
        ValueError
            For a negative time.

        Examples
        --------
        Either of two exponential events, with rates 1 and 3: the second
        causes three quarters of the top events.

        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> E = surv.Exponential.from_params
        >>> tree = FaultTree(
        ...     {"top": ("or", ["a", "b"])}, {"a": E([1.0]), "b": E([3.0])}
        ... )
        >>> shares = tree.barlow_proschan_importance()
        >>> {e: round(v, 6) for e, v in shares.items()}
        {'a': 0.25, 'b': 0.75}
        """
        from repyability.rbd._measures import barlow_proschan, life_end

        def sf(x: np.ndarray) -> np.ndarray:
            return np.asarray(self.sf(np.atleast_1d(x)), dtype=float).ravel()

        return barlow_proschan(
            self._rate_parts,
            self._lifetime_models(),
            t,
            lambda knots: life_end(sf, knots),
        )

    def parameter_sensitivity(
        self, t: Optional[ArrayLike] = None, rel_step: float = 1e-5
    ) -> Dict[Any, Dict[str, Any]]:
        """The sensitivity of the probability that the top event has not
        occurred (``sf``) to each parameter of the events' models.

        In the guide's Greeks it is the levers' *deltas* (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        For event ``e`` with parameter ``theta``, ``d sf / d theta =
        I_B(e) * d sf_e(t; theta) / d theta``: its Birnbaum importance
        times how far its model's reliability moves with the parameter,
        the latter by a central difference with step ``rel_step *
        |theta|`` (``rel_step`` if ``theta`` is 0), the model rebuilt with
        ``from_params`` (one-sided where one side is not a valid value, nan
        where neither is), as ``NonRepairableRBD.parameter_sensitivity``
        works out the diagram's. An event given as a probability has one
        parameter, ``p``, the probability (as a ``FixedEventProbability``
        names it). The members of a common-cause group carry one model,
        so its parameters are reported once, under the tuple of the
        members, each moved for every member at once, and so are the
        group's own (``"ccf_beta"``, and an [`MGL`][repyability.MGL]
        model's ``"ccf_gamma"``, ...), each by a central difference of
        the exact probabilities with the groups. ``levers()`` lists the
        parameters, with their values and ranges, and ``with_levers``
        builds the tree with them moved.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when no event's
            probability depends on time.
        rel_step : float, optional
            The relative step of the differences, by default 1e-5.

        Returns
        -------
        dict
            ``{event: {parameter: sensitivity}}``, by surpyval's parameter
            names, a common-cause group's under the tuple of its members:
            floats for a number ``t``, arrays for an array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on
            time.

        Examples
        --------
        One exponential event: ``R = exp(-rate t)``, so ``dR/d rate = -t
        exp(-rate t)``, about -9.048 at ``t = 10`` with rate 0.01:

        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c"])},
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> sens = tree.parameter_sensitivity(10)
        >>> {name: round(v, 3) for name, v in sens["c"].items()}
        {'failure_rate': -9.048}
        """
        from repyability.rbd._model_utils import lever_spec
        from repyability.rbd.ccf import CCFGroup
        from repyability.rbd.non_repairable_rbd import (
            _dsf_dparam,
            ccf_group_sensitivity,
        )

        times, scalar = self._times(t)
        size = len(times)
        _, occurred, not_occurred, _, _ = self._conditioned(times)
        p, q = self._event_probabilities(times)
        groups = list(self.ccf_groups)
        group_of = {m: g for g in groups for m in g.members}
        out: Dict[Any, Dict[str, Any]] = {}
        for event in self.events:
            spec = lever_spec(self._event_model(event))
            if event not in group_of:
                if spec is None:
                    continue
                birnbaum = occurred[event] - not_occurred[event]
                out[event] = {
                    name: self._out(
                        birnbaum * _dsf_dparam(spec, j, times, rel_step),
                        scalar,
                    )
                    for j, name in enumerate(spec.names)
                }
                continue
            group = group_of[event]
            if event != group.members[0]:
                continue
            index = next(i for i, g in enumerate(groups) if g is group)

            def system(members=None, model=None, group=group, index=index):
                given, failing = p, q
                if members is not None:
                    works, fails = members
                    given = {**p, **{m: works for m in group.members}}
                    failing = {**q, **{m: fails for m in group.members}}
                trial = groups
                if model is not None:
                    trial = list(groups)
                    trial[index] = CCFGroup(group.members, model)
                up, down = self._evaluation(
                    given, failing, size, groups=trial
                ).system()
                return (
                    np.broadcast_to(np.asarray(up, dtype=float), (size,)),
                    np.broadcast_to(np.asarray(down, dtype=float), (size,)),
                )

            values = ccf_group_sensitivity(
                system, group, spec, times, rel_step
            )
            out[tuple(group.members)] = {
                name: self._out(value, scalar)
                for name, value in values.items()
            }
        return out

    def levers(self) -> List["Lever"]:
        """The parameters ``parameter_sensitivity`` moves, in the order it
        reports them: each event's model's, by surpyval's names (``p``
        for an event given as a probability), and each common-cause
        group's under the tuple of its members (the one model its members
        carry, then its own, ``"ccf_beta"``, ...), each a
        [`Lever`][repyability.Lever] with its value and range, as a
        diagram's ``levers()`` lists them. Nothing is worked out;
        [`with_levers`][repyability.FaultTree.with_levers] sets them.

        Returns
        -------
        list of Lever
            The levers, in ``parameter_sensitivity``'s order.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c", "d"])},
        ...     {"c": surv.Weibull.from_params([100, 2]), "d": 0.05},
        ... )
        >>> for lever in tree.levers():
        ...     print(lever.key, lever.name, lever.value, lever.bounds)
        c alpha 100.0 (0.0, inf)
        c beta 2.0 (0.0, inf)
        d p 0.05 (0.0, 1.0)
        """
        from repyability.rbd import _sensitivity

        return _sensitivity.public_levers(self)

    def with_levers(self, values) -> "FaultTree":
        """A copy of this tree with parameters at new values, each moved
        as ``parameter_sensitivity`` moves it: an event's model rebuilt
        with the parameter changed (an event given as a probability keeps
        being one), a common-cause group's members' one model for every
        member, a group's own (``"ccf_beta"``, ...) in its model.
        ``levers()`` lists the parameters.

        Parameters
        ----------
        values : dict
            Each parameter's new value, by its
            [`Lever`][repyability.Lever] (from ``levers()``) or by its
            ``(key, name)``, as ``parameter_sensitivity`` reports it.

        Returns
        -------
        FaultTree
            The new tree, built as this one was with the parameters
            moved; this one is unchanged.

        Raises
        ------
        ValueError
            For a parameter the tree does not have (with the closest
            name), or a value its model refuses (one outside the lever's
            ``bounds``).
        TypeError
            If ``values`` is not a dict, a lever is neither a ``Lever``
            nor a ``(key, name)``, or a value is not a number.

        Examples
        --------
        Twice the Weibull scale: at 100 the top event has not occurred
        with probability ``exp(-(100 / 200) ** 2)``, about 0.7788, where
        it had ``exp(-1)``:

        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["c"])},
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> longer = tree.with_levers({("c", "alpha"): 200.0})
        >>> round(longer.sf(100), 4)
        0.7788
        """
        from repyability.rbd import _sensitivity

        return _sensitivity.with_levers(self, values)

    # -- uncertainty -------------------------------------------------------

    def sf_uncertainty(
        self,
        t: Optional[ArrayLike] = None,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
    ) -> "UncertaintyResult":
        """The probability that the top event has not occurred, over
        plausible event models (parameter uncertainty).

        As ``NonRepairableRBD.sf_uncertainty`` does for a diagram's
        reliability (see there for the kinds of uncertainty): each of the
        ``n_draws`` draws gives every uncertain event a plausible model,
        from the normal approximation of its surpyval fit (``"fit"``),
        from distributions over named parameters, or from a list of
        models, and the probability at ``t`` is worked out exactly for the
        draw, with the common-cause groups (whose models may be drawn too,
        the [`CCFGroup`][repyability.CCFGroup] given as a key). The draws
        are the diagram's of the tree (``to_rbd``, converted once): the
        events are its nodes, so they are keyed as its nodes are, events
        of one population together in a tuple, and a group's members
        together. An event given as a probability has no fit, so it is
        not drawn by default; its ``p`` may be given a distribution.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array. May be left out when every
            event's probability, drawn or not, is fixed.
        uncertainty : dict, optional
            ``{event or tuple of events: uncertainty}``, and ``{CCFGroup:
            uncertainty}``, as for ``NonRepairableRBD.sf_uncertainty``. By
            default every event whose model is a surpyval fit with a
            parameter covariance, as ``"fit"``.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seeds the draws, for reproducible results.
        sampling : str, optional
            ``"random"`` (the default), or ``"sobol"`` for the points of a
            scrambled Sobol sequence.

        Returns
        -------
        UncertaintyResult
            The probability of every draw (``samples``), the nominal value
            with the events' own models, and summaries such as
            ``interval`` (see
            [`UncertaintyResult`][repyability.UncertaintyResult]).

        Raises
        ------
        ValueError
            As for ``NonRepairableRBD.sf_uncertainty``: an unknown event
            or parameter, an uncertainty that cannot be drawn, a group's
            members not given together, or ``t`` left out for lifetime
            models.

        Examples
        --------
        Cooling is lost if both pumps fail or the valve does. The pumps
        are of one type, whose exponential failure rate is known only to
        lie between 1 and 3 per 1000 h, so they share the one draw:

        >>> import scipy.stats as st
        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> pump = surv.Exponential.from_params([0.002])
        >>> tree = FaultTree(
        ...     {"no cooling": ("or", ["no flow", "valve"]),
        ...      "no flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": pump, "pump 2": pump, "valve": 0.01},
        ... )
        >>> rate = {"failure_rate": st.uniform(0.001, 0.002)}
        >>> result = tree.sf_uncertainty(
        ...     500, {("pump 1", "pump 2"): rate}, n_draws=10_000, seed=1
        ... )
        >>> round(result.nominal, 4)
        0.5944
        >>> lower, upper = result.interval(0.9)
        >>> round(lower, 3), round(upper, 3)
        (0.409, 0.813)
        """
        return self._rbd().sf_uncertainty(
            t, uncertainty, n_draws=n_draws, seed=seed, sampling=sampling
        )

    def mean_uncertainty(
        self,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
    ) -> "UncertaintyResult":
        """The mean time to the top event over plausible event models
        (parameter uncertainty).

        As [`sf_uncertainty`][repyability.FaultTree.sf_uncertainty] does
        for the probability: each draw gives every uncertain event a
        plausible model, and the mean time to the top event is worked out
        exactly for the draw, as [`mean`][repyability.FaultTree.mean]
        works it out. The draws are the diagram's of the tree (``to_rbd``,
        converted once), as for ``sf_uncertainty``.

        Parameters
        ----------
        uncertainty : dict, optional
            ``{event or tuple of events: uncertainty}``, and ``{CCFGroup:
            uncertainty}``, as for ``sf_uncertainty``. By default every
            event whose model is a surpyval fit with a parameter
            covariance, as ``"fit"``.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seeds the draws, for reproducible results.
        sampling : str, optional
            ``"random"`` (the default), or ``"sobol"``.

        Returns
        -------
        UncertaintyResult
            The mean time of every draw (``samples``; ``inf`` for a draw
            in which the top event may never occur), the exact mean with
            the events' own models (``nominal``), and summaries such as
            ``interval``.

        Raises
        ------
        ValueError
            As for ``sf_uncertainty``, or if every event is a probability
            (the top event has no time).
        NotImplementedError
            With a common-cause group that splits a failure probability,
            as for ``mean``.

        Examples
        --------
        Two pumps of one type, both of which must fail, whose exponential
        failure rate is known only to lie between 1 and 3 per 1000 h:

        >>> import scipy.stats as st
        >>> import surpyval as surv
        >>> from repyability import FaultTree
        >>> pump = surv.Exponential.from_params([0.002])
        >>> tree = FaultTree(
        ...     {"no flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": pump, "pump 2": pump},
        ... )
        >>> rate = {"failure_rate": st.uniform(0.001, 0.002)}
        >>> result = tree.mean_uncertainty(
        ...     {("pump 1", "pump 2"): rate}, seed=1
        ... )
        >>> round(result.nominal, 6)  # 1.5 / 0.002
        750.0
        >>> lower, upper = result.interval(0.9)
        >>> round(lower), round(upper)
        (515, 1352)
        """
        self._require_lifetimes()
        return self._rbd().mean_uncertainty(
            uncertainty, n_draws=n_draws, seed=seed, sampling=sampling
        )

    # -- how each analysis is computed ---------------------------------------

    def analysis_routes(self) -> Dict[str, AnalysisRoute]:
        """How each analysis of this tree is computed, found without
        running it: exactly, numerically, by simulation, or not at all.

        As a diagram's
        [`analysis_routes`][repyability.NonRepairableRBD.analysis_routes]:
        the gates are always evaluated exactly, but a value is only as
        exact as the events' probabilities it is made of, so an event
        whose model's reliability is numerical makes the analyses built
        on it numerical, and one with no reliability makes them refuse. A
        refusal is found by the check the method itself runs, and its
        reason is the message it would raise.

        Returns
        -------
        dict[str, AnalysisRoute]
            For each public analysis, by method name: its ``route``
            (``"exact"``, ``"numerical"``, ``"simulated"`` or
            ``"refused"``), the ``reason``, and the ``events`` that decide
            it, as ``nodes`` (see
            [`AnalysisRoute`][repyability.AnalysisRoute]).

        Examples
        --------
        A tree of probabilities has no time to the top event:

        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2}
        ... )
        >>> routes = tree.analysis_routes()
        >>> routes["top_event_probability"].route
        'exact'
        >>> routes["mean"].route
        'refused'
        >>> routes["sf_uncertainty"].route
        'simulated'
        """
        from repyability.rbd import routes as r
        from repyability.rbd._model_utils import parametric_spec

        events = {
            e: (
                (r.EXACT, "a probability")
                if isinstance(m, float)
                else (r.model_route(m))
            )
            for e, m in self.events.items()
        }
        out: Dict[str, AnalysisRoute] = {}

        def give(names, route) -> None:
            for name in names:
                out[name] = route

        def built(route, reason):
            return r.with_nodes(route, reason, events, "probabilities")

        grouped = bool(self.ccf_groups)
        summed = (
            ", summed over the common-cause groups' outcomes."
            if grouped
            else "."
        )
        through = (
            " A common-cause group's member is conditioned on through the "
            "groups' outcomes."
            if grouped
            else ""
        )
        give(
            ("top_event_probability", "ff", "sf", "Hf"),
            built(
                r.EXACT,
                "The gates over the events' probabilities, exactly" + summed,
            ),
        )
        give(
            ("df", "hf"),
            built(
                r.NUMERICAL,
                "The exact top event probability, differentiated by central "
                "differences.",
            ),
        )
        fixed = r.refusal(self._require_time_varying)
        give(
            ("time_to_reliability", "bx_life"),
            (
                r.refused(fixed)
                if fixed
                else built(
                    r.NUMERICAL,
                    "The exact top event probability, inverted by "
                    "root-finding.",
                )
            ),
        )
        give(
            (
                "birnbaum_importance",
                "improvement_potential",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "criticality_importance",
            ),
            built(
                r.EXACT,
                "The exact top event probability, with each event occurred "
                "and not." + through,
            ),
        )
        give(
            ("ranked_cut_sets",),
            built(
                r.EXACT,
                "Each minimal cut set's exact probability of its events "
                "occurring together" + summed,
            ),
        )
        give(
            ("fussell_vesely",),
            built(
                r.EXACT,
                "The exact probability that a minimal cut set containing "
                "each event has occurred, over the top event probability "
                "(method='rare_event' sums the cut sets' probabilities "
                "instead)" + summed,
            ),
        )
        give(
            ("joint_importance",),
            built(
                r.EXACT,
                "Each event's exact Birnbaum importance with the other "
                "event held not occurring, less with it held occurring."
                + through,
            ),
        )
        give(
            ("differential_importance",),
            built(
                r.EXACT,
                "The exact Birnbaum importance of each event (or, for a "
                "proportional change, its criticality importance), shared "
                "out." + through,
            ),
        )
        out["structural_importance"] = AnalysisRoute(
            r.EXACT, "From the gates alone."
        )
        differenced = sorted(
            str(e)
            for e, m in self.events.items()
            if not isinstance(m, float) and parametric_spec(m) is None
        )
        out["barlow_proschan_importance"] = built(
            r.NUMERICAL,
            "Each event's Birnbaum importance times its model's failure "
            "density"
            + (
                f" (differences of the probability for {differenced})"
                if differenced
                else ""
            )
            + (
                ", a common-cause group's part by differences of the top "
                "event probability with its members' lives moved on"
                if grouped
                else ""
            )
            + ", integrated over time by adaptive quadrature (to about "
            "1e-9).",
        )
        out["parameter_sensitivity"] = built(
            r.NUMERICAL,
            "The exact Birnbaum importance times each parameter's "
            "derivative, by differences."
            + (
                " A common-cause group's parameters, and its model's, by "
                "differences of the exact top event probability."
                if grouped
                else ""
            ),
        )
        lifetimes = r.refusal(self._require_lifetimes)
        out["mean"] = (
            r.refused(lifetimes)
            if lifetimes
            else built(
                r.NUMERICAL,
                "The exact probability that the top event has not occurred, "
                "integrated over time by adaptive quadrature (to about "
                "1e-10, relative).",
            )
        )
        drawn = (
            " The common-cause groups are in each draw, and their models "
            "may be drawn too."
            if grouped
            else ""
        )
        out["sf_uncertainty"] = built(
            r.SIMULATED,
            "The events' model parameters drawn from their uncertainty, and "
            "the exact top event probability for each draw." + drawn,
        )
        out["mean_uncertainty"] = (
            r.refused(lifetimes)
            if lifetimes
            else built(
                r.SIMULATED,
                "The events' model parameters drawn from their uncertainty, "
                "and the exact mean time to the top event for each draw "
                "(the area under its probability of not having occurred)."
                + drawn,
            )
        )
        return dict(sorted(out.items()))

    # -- conversion --------------------------------------------------------

    def to_rbd(self) -> "NonRepairableRBD":
        """The reliability block diagram of the same system.

        The diagram says when the system works: an OR gate becomes a series
        block of its inputs, an AND gate a parallel block, and a VOTE gate
        needing ``k`` of ``n`` failures a block needing ``n - k + 1`` of its
        ``n`` inputs working, through a perfect junction node. Each event
        becomes a node with the event's model (a probability ``p`` becomes a
        ``FixedEventProbability`` of ``p``); junctions are
        ``PerfectReliability`` nodes named after their gates.

        An event or gate that feeds several gates is drawn once for each
        place: an event's later appearances are repeated nodes (named
        ``"event (2)"``, ...), which the diagram treats as the one
        component. So every tree converts exactly.

        Returns
        -------
        NonRepairableRBD
            The diagram, from input node ``"input"`` to output node
            ``"output"`` (renamed if an event has that name).

        Examples
        --------
        >>> from repyability import FaultTree
        >>> tree = FaultTree(
        ...     {"top": ("or", ["valve", "flow"]),
        ...      "flow": ("and", ["pump 1", "pump 2"])},
        ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
        ... )
        >>> rbd = tree.to_rbd()
        >>> round(rbd.sf(), 4), round(1 - tree.top_event_probability(), 4)
        (0.9405, 0.9405)
        """
        from surpyval import FixedEventProbability

        from repyability.rbd.helper_classes import PerfectReliability
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD

        taken = set(self._gates) | set(self.events)

        def fresh(base) -> Any:
            name = base
            while name in taken:
                name = f"{name}'"
            taken.add(name)
            return name

        edges: list = []
        models: dict = {}
        k: dict = {}
        seen: Dict[Hashable, int] = {}

        def place(event) -> Hashable:
            count = seen.get(event, 0)
            seen[event] = count + 1
            model = self.events[event]
            if isinstance(model, float):
                model = FixedEventProbability.from_params(model)
            if count == 0:
                models[event] = model
                return event
            node = fresh(f"{event} ({count + 1})")
            models[node] = event  # a repeat of the event's node
            return node

        def junction(label) -> Hashable:
            node = fresh(label)
            models[node] = PerfectReliability
            return node

        def join(ends: Tuple[set, set], label) -> Hashable:
            # One node for a block's exits, so that a vote counts blocks.
            entries, exits = ends
            if len(exits) == 1:
                return next(iter(exits))
            node = junction(label)
            edges.extend((x, node) for x in exits)
            return node

        def block(x) -> Tuple[set, set]:
            # The block's entry and exit nodes (drawn afresh at each use).
            if x in self.events:
                node = place(x)
                return {node}, {node}
            gate = self._gates[x]
            parts = [block(c) for c in gate.inputs]
            n = len(parts)
            working = n - gate.k + 1
            if working == n:  # in series
                for (_, before), (after, _) in zip(parts[:-1], parts[1:]):
                    edges.extend((u, v) for u in before for v in after)
                return parts[0][0], parts[-1][1]
            entries = set().union(*(e for e, _ in parts))
            if working == 1:  # in parallel
                return entries, set().union(*(x for _, x in parts))
            vote = junction(f"{x} (vote)")
            k[vote] = working
            for i, part in enumerate(parts):
                edges.append((join(part, f"{x} ({i + 1})"), vote))
            return entries, {vote}

        entries, exits = block(self.top)
        source, sink = fresh("input"), fresh("output")
        edges.extend((source, v) for v in entries)
        edges.extend((u, sink) for u in exits)
        return NonRepairableRBD(
            edges,
            models,
            k=k or None,
            input_node=source,
            output_node=sink,
            ccf_groups=self.ccf_groups or None,
        )

    @classmethod
    def from_rbd(cls, rbd) -> "FaultTree":
        """The fault tree of a reliability block diagram.

        The top event is the system failing. The diagram's modules become
        gates (a series block an OR gate, a parallel block an AND gate, a
        block needing ``k`` of ``n`` members working a VOTE gate on
        ``n - k + 1`` failures), and a part that is not series-parallel
        (e.g. a bridge) an OR gate over its minimal cut sets, each an AND
        gate. The nodes' models become the events' (a node's failure by
        time ``t`` is the event). Gates are named ``"TOP"`` and ``"G1"``,
        ``"G2"``, ... (renamed if a node has that name). Nodes that cannot
        affect the system are left out, but for a member of a common-cause
        group with one that can: the tree keeps it, for the group's causes
        to strike it with the others, as an event its logic makes
        irrelevant (the top event ``OR(G, AND(G, member))``, which is
        ``G``).

        Parameters
        ----------
        rbd : NonRepairableRBD
            The diagram.

        Returns
        -------
        FaultTree
            The tree: its top event probability is the diagram's
            unreliability.

        Raises
        ------
        TypeError
            If ``rbd`` is not a ``NonRepairableRBD``.
        ValueError
            If the diagram cannot fail (a direct input-to-output edge).

        Examples
        --------
        >>> from surpyval import FixedEventProbability
        >>> from repyability import FaultTree, NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": FixedEventProbability.from_params(0.1),
        ...         "p2": FixedEventProbability.from_params(0.1),
        ...         "v": FixedEventProbability.from_params(0.05),
        ...     },
        ... )
        >>> tree = FaultTree.from_rbd(rbd)
        >>> tree.gates
        {'G1': ('and', ['p1', 'p2']), 'TOP': ('or', ['G1', 'v'])}
        >>> round(tree.top_event_probability(), 4), round(1 - rbd.sf(), 4)
        (0.0595, 0.0595)
        """
        from repyability.rbd.helper_classes import PerfectReliability
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD

        if not isinstance(rbd, NonRepairableRBD):
            raise TypeError(
                "from_rbd takes a NonRepairableRBD, got "
                f"{type(rbd).__name__}."
            )
        decomposition = rbd._decomposition()
        if decomposition.always_works:
            raise ValueError(
                "The diagram cannot fail (its input joins its output "
                "directly), so it has no top event."
            )
        taken = set(rbd.reliabilities)

        def fresh(base) -> Any:
            name = base
            while name in taken:
                name = f"{name}'"
            taken.add(name)
            return name

        terms = decomposition.terms
        # The gates as made, each referred to by its _Made until the ones
        # below the top event are named: a module whose members cannot
        # affect the system (absorbed by the logic around it) gets a gate
        # that nothing uses, which is left out (#170).
        made: List[tuple] = []

        def over(inputs: list, needed: int) -> Any:
            # A gate that occurs when ``needed`` of ``inputs`` occur, or
            # None if fewer than that can ever occur.
            if needed > len(inputs):
                return None
            if len(inputs) == 1:
                return inputs[0]
            if needed == 1:
                made.append(("or", inputs))
            elif needed == len(inputs):
                made.append(("and", inputs))
            else:
                made.append(("vote", needed, inputs))
            return _Made(len(made) - 1)

        # Each term's event or gate, or None for one that can never fail,
        # such as a PerfectReliability junction: it drops out of the tree.
        label: Dict[int, Any] = {}
        for i, term in enumerate(terms):
            if term[0] == NODE:
                perfect = rbd.reliabilities[term[1]] is PerfectReliability
                label[i] = None if perfect else term[1]
                continue
            # The failed members that fail the module.
            n = len(term[1])
            if term[0] == SERIES:
                needed = 1
            elif term[0] == PARALLEL:
                needed = n
            else:
                needed = n - term[2] + 1
            members = [label[c] for c in term[1] if label[c] is not None]
            label[i] = over(members, needed)
        if decomposition.root is not None:
            top_label = label[decomposition.root]
        else:
            branches = [
                over([label[c] for c in cut if label[c] is not None], len(cut))
                for cut in decomposition.core_cut_sets()
            ]
            top_label = over([b for b in branches if b is not None], 1)
        if top_label is None:
            raise ValueError(
                "The diagram cannot fail: every way it could needs a node "
                "that never fails."
            )
        # A single node is a top event of its own failure.
        top_gate = (
            made[top_label.place]
            if isinstance(top_label, _Made)
            else ("or", [top_label])
        )
        below = set()
        pending = [x for x in top_gate[-1] if isinstance(x, _Made)]
        while pending:
            ref = pending.pop()
            if ref.place not in below:
                below.add(ref.place)
                pending += [
                    x for x in made[ref.place][-1] if isinstance(x, _Made)
                ]
        names = {i: fresh(f"G{n}") for n, i in enumerate(sorted(below), 1)}
        top = fresh("TOP")

        def named(gate: tuple) -> tuple:
            inputs = [
                names[x.place] if isinstance(x, _Made) else x for x in gate[-1]
            ]
            return (*gate[:-1], inputs)

        gates: Dict[Hashable, tuple] = {
            names[i]: named(made[i]) for i in sorted(below)
        }
        gates[top] = named(top_gate)
        used = {x for g in gates.values() for x in g[-1]}
        events = {
            node: rbd.reliabilities[node] for node in rbd.nodes if node in used
        }
        groups, irrelevant = [], []
        for group in rbd.ccf_groups:
            left_out = [m for m in group.members if m not in events]
            if len(left_out) == len(group.members):
                continue  # none can affect the system, nor can the group
            # A member the logic makes irrelevant (b in a OR (a AND b))
            # stays an event of the tree, for the group's shared causes to
            # strike it with the others (#237).
            irrelevant += [m for m in left_out if m not in irrelevant]
            groups.append(group)
        if irrelevant:
            # TOP = OR(G, AND(G, members)) is G: the members are in the
            # tree without changing it.
            inner, kept = fresh(f"G{len(names) + 1}"), fresh(
                f"G{len(names) + 2}"
            )
            gates[inner] = gates.pop(top)
            gates[kept] = ("and", [inner, *irrelevant])
            gates[top] = ("or", [inner, kept])
            events.update((m, rbd.reliabilities[m]) for m in irrelevant)
        return cls(gates, events, top=top, ccf_groups=groups or None)

    # -- saving ------------------------------------------------------------

    def to_dict(self) -> dict:
        """The tree as a JSON-compatible dict (see ``from_dict``).

        Returns
        -------
        dict
            The gates, the events (probabilities as numbers, models as
            RePyability serialises them), the top event and any
            common-cause groups (as a diagram saves them).
        """
        from repyability._version import __version__
        from repyability.rbd.serialisation import _ccf_to_list, serialise_model

        events: List[Dict[str, Any]] = []
        for name, model in self.events.items():
            if isinstance(model, float):
                events.append({"event": name, "probability": model})
            else:
                events.append({"event": name, "model": serialise_model(model)})
        out = {
            "repyability_version": __version__,
            "type": "FaultTree",
            "top": self.top,
            "gates": [
                {
                    "gate": name,
                    "kind": g.kind,
                    "k": g.k,
                    "inputs": list(g.inputs),
                }
                for name, g in self._gates.items()
            ],
            "events": events,
        }
        if self.ccf_groups:
            out["ccf_groups"] = _ccf_to_list(self.ccf_groups)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "FaultTree":
        """Rebuild a tree from ``to_dict``'s output.

        Parameters
        ----------
        d : dict
            As ``to_dict`` returns.

        Returns
        -------
        FaultTree
            The tree.

        Raises
        ------
        ValueError
            If ``d`` is not a fault tree's dict.
        """
        from repyability.rbd.serialisation import (
            _ccf_from_list,
            _node_name,
            deserialise_model,
        )

        if d.get("type") != "FaultTree":
            raise ValueError(f"Not a fault tree: type {d.get('type')!r}.")
        gates: Dict[Hashable, tuple] = {}
        for g in d["gates"]:
            inputs = [_node_name(x) for x in g["inputs"]]
            gates[_node_name(g["gate"])] = (
                (g["kind"], g["k"], inputs)
                if g["kind"] == "vote"
                else (g["kind"], inputs)
            )
        events = {
            _node_name(e["event"]): (
                e["probability"]
                if "probability" in e
                else deserialise_model(e["model"])
            )
            for e in d["events"]
        }
        return cls(
            gates,
            events,
            top=_node_name(d["top"]),
            ccf_groups=_ccf_from_list(d.get("ccf_groups")),
        )

    def to_json(self, fp=None, **json_kwargs) -> Optional[str]:
        """The tree as a JSON document (see ``to_dict``): returned, or
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
        """
        from repyability.utils.json_io import write_json

        return write_json(json.dumps(self.to_dict(), **json_kwargs), fp)

    @classmethod
    def from_json(cls, s) -> "FaultTree":
        """Rebuild a tree from ``to_json``'s output.

        Parameters
        ----------
        s : str, os.PathLike or file
            The JSON document: its text, a path to a file holding it, or a
            file opened for reading.

        Returns
        -------
        FaultTree
            The tree.
        """
        from repyability.utils.json_io import read_json

        return cls.from_dict(json.loads(read_json(s)))


def _pruned(
    terms: list, roots: List[int], plan: Optional[Tuple[list, int]] = None
) -> Decomposition:
    """The decomposition over the terms under ``roots`` (renumbered, kept in
    order): the one root's, or the core's decision diagram, ``plan``, its
    pivots renumbered too."""
    keep: set = set()
    stack = list(roots)
    while stack:
        v = stack.pop()
        if v in keep:
            continue
        keep.add(v)
        if terms[v][0] != NODE:
            stack.extend(terms[v][1])
    position: Dict[int, int] = {}
    tree: list = []
    for v, term in enumerate(terms):
        if v not in keep:
            continue
        position[v] = len(tree)
        if term[0] == NODE:
            tree.append(term)
        else:
            children = tuple(position[c] for c in term[1])
            tree.append((term[0], children, *term[2:]))
    if plan is None:
        return Decomposition(tree, root=position[roots[0]])
    steps, root = plan
    if root == 1:  # the top never occurs
        return Decomposition([])
    return Decomposition(
        tree, plan=([(position[p], a, b) for p, a, b in steps], root)
    )


def _model_is_fixed(model) -> bool:
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD

    return NonRepairableRBD._model_is_fixed(model)
