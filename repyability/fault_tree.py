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
the repeated events tie together is left as a core, worked out by the
Shannon decomposition over its minimal path sets. Nothing is approximated:
the top event probability and the importance measures are exact, repeated
events included.
"""

import json
from itertools import combinations
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

#: Working out the core's minimal path sets gives up, with guidance, beyond
#: this many sets for one gate.
PATH_SET_LIMIT = 200_000


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


def _minimal(sets: List[frozenset]) -> List[frozenset]:
    """The sets that contain no other, smallest first."""
    kept: List[frozenset] = []
    for s in sorted(set(sets), key=len):
        if not any(k <= s for k in kept):
            kept.append(s)
    return kept


def _joined(families: Sequence[List[frozenset]]) -> List[frozenset]:
    """One set from each family, joined, in every combination; minimal."""
    out: List[frozenset] = [frozenset()]
    for family in families:
        out = _minimal([a | b for a in out for b in family])
        if len(out) > PATH_SET_LIMIT:
            raise ValueError(
                "The repeated events tie together too much of the tree to "
                f"list its minimal path sets (more than {PATH_SET_LIMIT:,})."
            )
    return out


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

    Raises
    ------
    ValueError
        If a gate is malformed, an input is neither a gate nor an event, a
        name is both, the gates form a loop, a gate or event is not below
        the top event, the top event is not a gate (or cannot be inferred),
        or an event's model is invalid.

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
        a core given by its minimal path sets."""
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
            return _pruned(terms, [position[self.top]], None)
        paths: Dict[Hashable, List[frozenset]] = {}
        for x in order:
            if x in position:
                paths[x] = [frozenset([position[x]])]
                continue
            gate = gates[x]
            families = [paths[c] for c in gate.inputs]
            working = len(families) - gate.k + 1
            if working == len(families):
                paths[x] = _joined(families)
            elif working == 1:
                paths[x] = _minimal([s for f in families for s in f])
            else:
                found: List[frozenset] = []
                for chosen in combinations(families, working):
                    found.extend(_joined(chosen))
                paths[x] = _minimal(found)
        core = paths[self.top]
        return _pruned(terms, sorted(set().union(*core)), core)

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
        """The top event probability from the events' probabilities."""
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
        works, _ = self._decomposition.probabilities(
            p, q, shape=size, works=True, fails=False
        )
        values = np.broadcast_to(np.asarray(works, dtype=float), (size,))
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

        A cut set's probability is the product of its events'. The largest
        say which combinations of failures dominate the top event.

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
        _, q = self._event_probabilities(times)
        ranked = [
            (c, float(np.prod([q[e][0] for e in c])))
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
        top = self._top(p, q, size)
        occurred, not_occurred = {}, {}
        for e in self.events:
            p_e, q_e = dict(p), dict(q)
            p_e[e], q_e[e] = np.zeros(size), np.ones(size)
            occurred[e] = self._top(p_e, q_e, size)
            p_e[e], q_e[e] = np.ones(size), np.zeros(size)
            not_occurred[e] = self._top(p_e, q_e, size)
        return top, occurred, not_occurred, q, scalar

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

    def fussell_vesely(self, t: Optional[ArrayLike] = None) -> dict:
        """Fussell-Vesely importance of each basic event.

        The share of the top event probability carried by the minimal cut
        sets that contain the event: the sum of their probabilities (the
        rare-event approximation of their union) divided by the top event
        probability, as ``NonRepairableRBD.fussell_vesely`` computes it.
        Values can exceed 1 when the events are not rare. ``nan`` where the
        top event cannot occur.

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
        >>> {e: round(v, 4) for e, v in tree.fussell_vesely().items()}
        {'pump 1': 0.1681, 'pump 2': 0.1681, 'valve': 0.8403}
        """
        times, scalar = self._times(t)
        p, q = self._event_probabilities(times)
        top = self._top(p, q, len(times))
        share = {e: np.zeros(len(times)) for e in self.events}
        for cut in self.minimal_cut_sets():
            probability = np.prod([q[e] for e in cut], axis=0)
            for e in cut:
                share[e] = share[e] + probability
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {e: share[e] / top for e in self.events}
        return self._out(out, scalar)

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
            edges, models, k=k or None, input_node=source, output_node=sink
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
            If the diagram has common-cause groups.

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
        if rbd.ccf_groups:
            raise NotImplementedError(
                "A diagram with common-cause groups has no fault tree here."
            )
        decomposition = rbd._decomposition()
        if decomposition.always_works:
            raise ValueError(
                "The diagram cannot fail (its input joins its output "
                "directly), so it has no top event."
            )
        taken = set(rbd.reliabilities)
        count = 0

        def fresh(base) -> Any:
            name = base
            while name in taken:
                name = f"{name}'"
            taken.add(name)
            return name

        def gate_name() -> Any:
            nonlocal count
            count += 1
            return fresh(f"G{count}")

        terms = decomposition.terms
        gates: Dict[Hashable, tuple] = {}

        def over(inputs: list, needed: int) -> Any:
            # A gate that occurs when ``needed`` of ``inputs`` occur, or
            # None if fewer than that can ever occur.
            if needed > len(inputs):
                return None
            if len(inputs) == 1:
                return inputs[0]
            name = gate_name()
            if needed == 1:
                gates[name] = ("or", inputs)
            elif needed == len(inputs):
                gates[name] = ("and", inputs)
            else:
                gates[name] = ("vote", needed, inputs)
            return name

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
        top = fresh("TOP")
        if top_label in gates:
            gates[top] = gates.pop(top_label)
        else:
            # A single node: the top event is its failure.
            gates[top] = ("or", [top_label])
        used = {x for g in gates.values() for x in g[-1]}
        events = {
            node: rbd.reliabilities[node] for node in rbd.nodes if node in used
        }
        return cls(gates, events, top=top)

    # -- saving ------------------------------------------------------------

    def to_dict(self) -> dict:
        """The tree as a JSON-compatible dict (see ``from_dict``).

        Returns
        -------
        dict
            The gates, the events (probabilities as numbers, models as
            RePyability serialises them) and the top event.
        """
        from repyability._version import __version__
        from repyability.rbd.serialisation import serialise_model

        events: List[Dict[str, Any]] = []
        for name, model in self.events.items():
            if isinstance(model, float):
                events.append({"event": name, "probability": model})
            else:
                events.append({"event": name, "model": serialise_model(model)})
        return {
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
        return cls(gates, events, top=_node_name(d["top"]))

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
    terms: list, roots: List[int], core: Optional[List[frozenset]]
) -> Decomposition:
    """The decomposition over the terms under ``roots`` (renumbered, kept in
    order), with the ``core``'s path sets renumbered too."""
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
    if core is None:
        return Decomposition(tree, root=position[roots[0]])
    return Decomposition(
        tree, core=[sorted(position[c] for c in ps) for ps in core]
    )


def _model_is_fixed(model) -> bool:
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD

    return NonRepairableRBD._model_is_fixed(model)


# A gate as given: ("or", inputs), ("and", inputs) or ("vote", k, inputs).
GateSpec = Union[Tuple[str, Sequence[Hashable]], Tuple[str, int, Sequence]]
