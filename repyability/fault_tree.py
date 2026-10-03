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
"""

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
    Union,
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

if TYPE_CHECKING:  # pragma: no cover
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD

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
        ``NonRepairableRBD`` takes them (#184): each group's members,
        events with the same model, occur together through its shared
        causes as well as on their own, split by its ``BetaFactor`` or
        ``MGL`` model at every ``t``. The top event probability and the
        importance measures sum over the groups' shock outcomes, exactly
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
            if len(roots) != 1:
                raise ValueError(
                    "Give the top event: "
                    + (
                        "every gate is another's input."
                        if not roots
                        else f"gates {sorted(map(str, roots))} are no gate's "
                        "input."
                    )
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
        from repyability.rbd.ccf import CCFGroup
        from repyability.rbd.serialisation import serialise_model

        if not groups:
            return []
        seen: set = set()
        for group in groups:
            if not isinstance(group, CCFGroup):
                raise ValueError(
                    "ccf_groups must hold CCFGroup instances, got "
                    f"{type(group).__name__}."
                )
            for member in group.members:
                if member not in self.events:
                    raise ValueError(
                        f"CCF group member {member!r} is not a basic event "
                        "of the tree."
                    )
                if member in seen:
                    raise ValueError(
                        f"Event {member!r} appears in more than one CCF "
                        "group."
                    )
                seen.add(member)
            # Compared as saved, as the diagram compares its members (and,
            # as there, not at all for a model that cannot be saved).
            try:
                specs = [
                    (
                        self.events[m]
                        if isinstance(self.events[m], float)
                        else serialise_model(self.events[m])
                    )
                    for m in group.members
                ]
            except Exception:
                continue
            if any(spec != specs[0] for spec in specs[1:]):
                raise ValueError(
                    f"CCF group {list(group.members)} is not symmetric: its "
                    "members must have the same model."
                )
        return list(groups)

    def _outcomes(self, p: dict, q: dict):
        """For each combination of the common-cause groups' shock outcomes,
        its probability and the events' probabilities given it, under which
        they are independent (one outcome, certain, without groups)."""
        if not self.ccf_groups:
            yield 1.0, p, q
            return
        from repyability.rbd.ccf import shock_outcomes

        yield from shock_outcomes(self.ccf_groups, p, q, self._ccf_check)

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
        return np.atleast_1d(np.asarray(t, dtype=float)), scalar

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
        """The top event probability from the events' probabilities, over
        the common-cause groups' outcomes."""
        total: Any = 0.0
        for weight, given, failing in self._outcomes(p, q):
            total = total + weight * self._independent_top(
                given, failing, size
            )
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
        total: Any = 0.0
        for weight, given, failing in self._outcomes(p, q):
            works, _ = self._decomposition.probabilities(
                given, failing, shape=size, works=True, fails=False
            )
            total = total + weight * np.asarray(works, dtype=float)
        values = np.broadcast_to(np.asarray(total, dtype=float), (size,))
        return self._out(values, scalar)

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
        occurred = set(events)
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
        times, _ = self._times(t)
        p, q = self._event_probabilities(times)
        outcomes = list(self._outcomes(p, q))
        ranked = [
            (
                c,
                float(
                    sum(
                        np.asarray(weight).reshape(-1)[0]
                        * np.prod([failing[e][0] for e in c])
                        for weight, _, failing in outcomes
                    )
                ),
            )
            for c in self.minimal_cut_sets()
        ]
        order = sorted(range(len(ranked)), key=lambda i: (-ranked[i][1], i))
        return [ranked[i] for i in order]

    # -- importance --------------------------------------------------------

    def _conditioned(self, t) -> Tuple[np.ndarray, dict, dict, dict, bool]:
        """The top event probability, and for each event the top event
        probability given that it has occurred and given that it has not,
        with the events' probabilities of occurring."""
        times, scalar = self._times(t)
        p, q = self._event_probabilities(times)
        size = len(times)
        if not self.ccf_groups:
            top = self._top(p, q, size)
            occurred, not_occurred = {}, {}
            for e in self.events:
                p_e, q_e = dict(p), dict(q)
                p_e[e], q_e[e] = np.zeros(size), np.ones(size)
                occurred[e] = self._top(p_e, q_e, size)
                p_e[e], q_e[e] = np.ones(size), np.zeros(size)
                not_occurred[e] = self._top(p_e, q_e, size)
            return top, occurred, not_occurred, q, scalar
        # With common-cause groups, as the diagram's measures: a member's
        # occurrence says something of the others', so the top event's
        # probability given it is a sum over the shock outcomes, each
        # weighed by the member's chance of that state in it; an event
        # outside the groups is held, as without them.
        grouped = {m for group in self.ccf_groups for m in group.members}
        top_sum: Any = 0.0
        joint_in: Dict[Hashable, Any] = {e: 0.0 for e in self.events}
        joint_out: Dict[Hashable, Any] = {e: 0.0 for e in self.events}
        chance_in: Dict[Hashable, Any] = {e: 0.0 for e in grouped}
        chance_out: Dict[Hashable, Any] = {e: 0.0 for e in grouped}
        ones, zeros = np.ones(size), np.zeros(size)
        for weight, given, failing in self._outcomes(p, q):
            top_sum = top_sum + weight * self._independent_top(
                given, failing, size
            )
            for e in self.events:
                yes = self._independent_top(
                    {**given, e: zeros}, {**failing, e: ones}, size
                )
                no = self._independent_top(
                    {**given, e: ones}, {**failing, e: zeros}, size
                )
                if e in grouped:
                    chance_in[e] = chance_in[e] + weight * failing[e]
                    chance_out[e] = chance_out[e] + weight * given[e]
                    joint_in[e] = joint_in[e] + weight * failing[e] * yes
                    joint_out[e] = joint_out[e] + weight * given[e] * no
                else:
                    joint_in[e] = joint_in[e] + weight * yes
                    joint_out[e] = joint_out[e] + weight * no
        occurred, not_occurred, marginal = {}, {}, dict(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            for e in self.events:
                if e in grouped:
                    occurred[e] = joint_in[e] / chance_in[e]
                    not_occurred[e] = joint_out[e] / chance_out[e]
                    marginal[e] = np.asarray(chance_in[e], dtype=float)
                else:
                    occurred[e], not_occurred[e] = joint_in[e], joint_out[e]
        top = np.broadcast_to(np.asarray(top_sum, dtype=float), (size,))
        return top, occurred, not_occurred, marginal, scalar

    def birnbaum_importance(self, t: Optional[ArrayLike] = None) -> dict:
        """Birnbaum importance of each basic event.

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
        if method not in ("exact", "rare_event"):
            raise ValueError(
                f"method must be 'exact' or 'rare_event', got {method!r}."
            )
        times, scalar = self._times(t)
        p, q = self._event_probabilities(times)
        top = self._top(p, q, len(times))
        share = {e: np.zeros(len(times)) for e in self.events}
        # Summed over the common-cause groups' outcomes, the events being
        # independent given each.
        for weight, given, failing in self._outcomes(p, q):
            if method == "exact":
                failed = self._decomposition.failed_cut_sets(
                    given, failing, shape=len(times)
                )
                for e, probability in failed.items():
                    share[e] = share[e] + weight * probability
            else:
                for cut in self.minimal_cut_sets():
                    probability = np.prod([failing[e] for e in cut], axis=0)
                    for e in cut:
                        share[e] = share[e] + weight * probability
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
        """The joint (second-order) importance of each pair of basic events
        (#194): whether preventing the two together is worth more than
        preventing each.

        ``JRI(e, f) = -d2P / dq_e dq_f``, ``P`` the top event probability
        and ``q`` the events' probabilities: how much event ``f``'s
        Birnbaum importance falls when event ``e`` goes from occurring to
        not. It is the joint importance of the diagram the tree describes
        (see ``to_rbd``), in its reliability. Positive, the two are
        complements, as under an OR gate: preventing either makes
        preventing the other worth more. Negative, they are substitutes, as
        under an AND gate: either one prevented keeps the gate from
        occurring. Exact, from the top event probability with both events
        held, as the tree is multilinear in the events.

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
        NotImplementedError
            With common-cause groups, whose members cannot be held.

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
        if self.ccf_groups:
            raise NotImplementedError(
                "The joint importance holds pairs of events occurring and "
                "not, which a common-cause group's members cannot be (the "
                "causes they share would still strike the others): it is "
                "not worked out with common-cause groups, as yet."
            )
        times, scalar = self._times(t)
        p, q = self._event_probabilities(times)
        size = len(times)
        ones, zeros = np.ones(size), np.zeros(size)
        events = list(self.events)

        def importance(p_: dict, q_: dict, of: list) -> dict:
            out = {}
            for f in of:
                p_f, q_f = dict(p_), dict(q_)
                p_f[f], q_f[f] = zeros, ones
                occurred = self._top(p_f, q_f, size)
                p_f[f], q_f[f] = ones, zeros
                out[f] = occurred - self._top(p_f, q_f, size)
            return out

        from repyability.rbd.rbd import Pairs

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
        affect the system are left out.

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
        NotImplementedError
            If a common-cause group has a member that cannot affect the
            system and one that can (the tree's groups are the diagram's,
            less any whose members none can).

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
        groups = []
        for group in rbd.ccf_groups:
            left_out = [m for m in group.members if m not in events]
            if len(left_out) == len(group.members):
                continue  # none can affect the system, nor can the group
            if left_out:
                raise NotImplementedError(
                    f"Common-cause group {list(group.members)}: "
                    f"{left_out} cannot affect the system, so they are no "
                    "events of the tree, but the group's shared causes "
                    "strike them with the others."
                )
            groups.append(group)
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


# A gate as given: ("or", inputs), ("and", inputs) or ("vote", k, inputs).
GateSpec = Union[Tuple[str, Sequence[Hashable]], Tuple[str, int, Sequence]]
