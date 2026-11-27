"""Event trees: an initiating event's frequency through its barriers to the
frequency of each outcome.

An event tree starts from an *initiating event*, which occurs at a
frequency, and asks, heading by heading, whether each barrier (a safety
function, a protection layer, an operator action) works or fails on the
demand. A *sequence* is one path through the headings: each heading
succeeded, failed or was not asked (a heading a path does not reach). Its
frequency is the initiator's times the probability of the path, and an
*outcome* collects the sequences that end the same way (NASA's *PRA
Procedures Guide*, IEC 62502, LOPA in IEC 61511-3).

A heading is a fault tree (its top event, the barrier failing) or a
probability (an independent protection layer). The trees may share basic
events (a support system several barriers need), so the headings are not
independent and the probability of a path is not the product of its
branches'. Here it is exact: each sequence is written as one fault tree,
its top the AND of each asked heading's top event or its NOT, over the
headings' gates (named apart by heading) and their events (an event of
one name the same event in every heading), and evaluated by the
non-coherent tree's decision diagram (``_ft_logic.py``). No value is
worked out here: each comes from ``FaultTree``.
"""

import dataclasses
import itertools
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import ArrayLike

from repyability.fault_tree import FaultTree
from repyability.rbd import _measures
from repyability.rbd.routes import AnalysisRoute, AnalysisRoutes
from repyability.utils.checks import is_number, one_of, real_array

if TYPE_CHECKING:  # pragma: no cover
    from repyability.rbd.results import (
        OutcomeFrequencies,
        SequenceFrequencies,
        SequenceProbabilities,
    )

#: A heading's branches on a path: it succeeded, it failed, or the path
#: does not ask it (None).
BRANCHES = ("success", "failure", None)

#: The approximations ``sequence_probabilities`` and the rest take, besides
#: the exact values (None).
APPROXIMATIONS = ("rare_event", "independent")


@dataclasses.dataclass(frozen=True)
class _Gate:
    """A gate a sequence's tree adds: the sequence's top (``heading``
    None), or the NOT of a heading's top event."""

    heading: Hashable = None


Path = Tuple[Optional[str], ...]


class EventTree:
    """An event tree: an initiating event's frequency through the success
    or failure of each heading to the frequency of each outcome, exactly,
    headings that share basic events included.

    See the guide's [event trees](../guide/event-trees.md).

    Parameters
    ----------
    initiator : float or FaultTree
        The initiating event's frequency, a number 0 or more (per unit
        time), or a [`FaultTree`][repyability.FaultTree] of repaired events
        whose long-run frequency of its top event,
        [`system_failure_frequency`][repyability.FaultTree.system_failure_frequency],
        is the initiator's. The headings are taken as probabilities on
        demand given that it has occurred: an initiator tree's events are
        not shared with them.
    headings : list of (name, FaultTree or float)
        The headings in the order they are asked, each a name and either a
        ``FaultTree``, whose top event is the heading failing, or the
        probability that it fails (an independent protection layer, which
        stands as a basic event named for the heading). A dict keeps its
        order too. Events of one name in several trees are one event, and
        must have one model; the trees' gates are named apart, as
        ``(heading, gate)``. A tree with dynamic gates or common-cause
        groups is refused (write a shared cause as a shared event).
    sequences : dict, optional
        ``{path: outcome}``: each path a tuple with an entry per heading,
        ``"success"``, ``"failure"`` or None (the path does not ask the
        heading), and its outcome any name. The paths must partition the
        initiator: every combination of the headings' successes and
        failures on exactly one path. By default every combination is a
        path of its own, its outcome the path, in the usual order (every
        heading succeeding first, the last heading's branch changing
        fastest).

    Attributes
    ----------
    initiator : float or FaultTree
        As given (a number as a float).
    frequency : float
        The initiator's frequency.
    headings : dict
        ``{name: FaultTree or float}``, in order.
    sequences : dict
        ``{path: outcome}``, in order.
    events : dict
        ``{event: model}``: every heading tree's events, and each
        probability heading as an event of its name.
    shared_events : frozenset
        The events of more than one heading: what makes the headings
        dependent.

    Raises
    ------
    ValueError
        If the initiator is not a frequency or a tree, a heading is not
        named once or its model is neither a tree nor a probability in [0,
        1], an event has different models in two headings, a probability
        heading is named as an event, or the paths are malformed or do not
        partition the initiator (naming a combination on none or on two).
    NotImplementedError
        For a heading tree with dynamic gates or common-cause groups, or
        an FDEP gate forcing an event another heading has.

    Examples
    --------
    A loss of cooling (0.1 a year) calls on injection and then on
    recirculation; both pumps need the one power supply:

    >>> from repyability import EventTree, FaultTree
    >>> injection = FaultTree(
    ...     {"no injection": ("or", ["pump A", "power"])},
    ...     {"pump A": 0.05, "power": 0.01},
    ... )
    >>> recirculation = FaultTree(
    ...     {"no recirculation": ("or", ["pump B", "power"])},
    ...     {"pump B": 0.1, "power": 0.01},
    ... )
    >>> tree = EventTree(
    ...     0.1, [("injection", injection), ("recirculation", recirculation)]
    ... )
    >>> frequencies = tree.sequence_frequencies().frequencies
    >>> round(frequencies["failure", "failure"], 8)  # 0.1 * P(both fail)
    0.001495
    >>> naive = tree.sequence_frequencies(approximation="independent")
    >>> round(naive.frequencies["failure", "failure"], 8)  # 0.1 * .0595 * .109
    0.00064855
    """

    def __init__(
        self,
        initiator: Any,
        headings: Any,
        sequences: Optional[Mapping[Sequence, Hashable]] = None,
    ):
        self.initiator, self.frequency = _initiator(initiator)
        self.headings = _headings(headings)
        self._merge()
        self.sequences = _sequences(list(self.headings), sequences)
        self._trees: Dict[Tuple[Path, Optional[str]], Optional[FaultTree]]
        self._trees = {}
        for path in self.sequences:
            self._trees[path, None] = self._tree(path, rare_event=False)

    # -- building --------------------------------------------------------

    def _merge(self) -> None:
        """The headings' events as one set, each event one model; the
        gates of each tree named apart by heading."""
        events: Dict[Hashable, Any] = {}
        owner: Dict[Hashable, Hashable] = {}
        holders: Dict[Hashable, set] = {}
        gates: Dict[Hashable, Dict[Hashable, tuple]] = {}
        tops: Dict[Hashable, Hashable] = {}
        for heading, model in self.headings.items():
            if not isinstance(model, FaultTree):
                continue
            for event, event_model in model.events.items():
                holders.setdefault(event, set()).add(heading)
                if event in events:
                    if not _same(events[event], event_model):
                        raise ValueError(
                            f"Event {event!r} has different models in "
                            f"headings {owner[event]!r} and {heading!r}: an "
                            "event of one name is one event in every "
                            "heading, so give it one model there (or name "
                            "the two events apart)."
                        )
                    continue
                events[event], owner[event] = event_model, heading
            gates[heading] = _named_apart(heading, model)
            tops[heading] = (heading, model.top)
        for heading, model in self.headings.items():
            if isinstance(model, FaultTree):
                continue
            if heading in events:
                raise ValueError(
                    f"Heading {heading!r} is a probability, which stands as "
                    f"a basic event of its name, and {heading!r} is also an "
                    f"event of heading {owner[heading]!r}: rename the "
                    "heading."
                )
            events[heading] = model
            holders[heading] = {heading}
            tops[heading] = heading
        self._forced_apart(holders)
        self.events = events
        self.shared_events = frozenset(
            e for e, held in holders.items() if len(held) > 1
        )
        self._gates, self._tops = gates, tops

    def _forced_apart(self, holders: Dict[Hashable, set]) -> None:
        """Raise if an FDEP gate forces an event another heading has: in a
        sequence's tree it would force it there too, which that heading's
        own tree does not."""
        for heading, model in self.headings.items():
            if not isinstance(model, FaultTree):
                continue
            for gate, spec in model.gates.items():
                if spec[0] != "fdep":
                    continue
                shared = sorted(
                    str(e) for e in spec[2] if len(holders.get(e, ())) > 1
                )
                if shared:
                    raise NotImplementedError(
                        f"FDEP gate {gate!r} of heading {heading!r} forces "
                        f"event(s) {shared}, which another heading has "
                        "too: a sequence would force them there as well. "
                        "Write the dependency as an OR gate of the event "
                        "and its trigger in each heading that has it."
                    )

    def _tree(self, path: Path, rare_event: bool) -> Optional[FaultTree]:
        """The fault tree of a path: its top the AND of each asked
        heading's top event (failure) or its NOT (success; left out with
        ``rare_event``), over the asked headings' gates and events. None
        for a path asking nothing (its probability is 1)."""
        gates: Dict[Hashable, tuple] = {}
        events: Dict[Hashable, Any] = {}
        inputs: List[Hashable] = []
        for heading, branch in zip(self.headings, path):
            if branch is None or (rare_event and branch == "success"):
                continue
            model = self.headings[heading]
            if isinstance(model, FaultTree):
                gates.update(self._gates[heading])
                events.update({e: self.events[e] for e in model.events})
            else:
                events[heading] = model
            top = self._tops[heading]
            if branch == "failure":
                inputs.append(top)
            else:
                gates[_Gate(heading)] = ("not", [top])
                inputs.append(_Gate(heading))
        if not inputs:
            return None
        gates[_Gate()] = ("and", inputs)
        return FaultTree(gates, events, top=_Gate())

    def sequence_tree(
        self, path: Sequence, approximation: Optional[str] = None
    ) -> Optional[FaultTree]:
        """The fault tree of one sequence: its top event the AND of each
        asked heading's top event (where the path fails it) or its NOT
        (where the path succeeds), over the headings' gates, named
        ``(heading, gate)``, and their events, one event to a name.

        Its ``top_event_probability`` is the sequence's probability, its
        [`prime_implicants`][repyability.FaultTree.prime_implicants] the
        sequence's smallest combinations of events occurring and not
        occurring (its minimal cut sets, where it only fails headings).

        Parameters
        ----------
        path : tuple
            One of the tree's paths (see ``sequences``).
        approximation : str, optional
            None (the default) for the sequence as it is, or
            ``"rare_event"`` for the tree with the headings the path
            succeeds left out, which the rare-event values evaluate.

        Returns
        -------
        FaultTree or None
            The tree; None for a path that asks no heading (with
            ``"rare_event"``, one that fails none), whose probability is 1.

        Raises
        ------
        ValueError
            If ``path`` is not one of the tree's paths, or
            ``approximation`` is not None or ``"rare_event"``.

        Examples
        --------
        >>> from repyability import EventTree, FaultTree
        >>> barrier = FaultTree(
        ...     {"fails": ("or", ["valve", "power"])},
        ...     {"valve": 0.05, "power": 0.01},
        ... )
        >>> tree = EventTree(0.1, [("relief", barrier), ("operator", 0.1)])
        >>> sequence = tree.sequence_tree(("success", "failure"))
        >>> round(sequence.top_event_probability(), 6)  # 0.95 * 0.99 * 0.1
        0.09405
        """
        if approximation is not None:
            one_of("approximation", approximation, ("rare_event",))
        path = self._path(path)
        return self._sequence_tree(path, approximation)

    def _sequence_tree(
        self, path: Path, approximation: Optional[str]
    ) -> Optional[FaultTree]:
        key = (path, approximation)
        if key not in self._trees:
            self._trees[key] = self._tree(
                path, rare_event=approximation == "rare_event"
            )
        return self._trees[key]

    def _path(self, path: Sequence) -> Path:
        """``path`` as one of the tree's paths, or a ValueError."""
        try:
            key = tuple(path)
        except TypeError:
            key = None
        if key not in self.sequences:
            raise ValueError(
                f"{path!r} is not one of the event tree's paths: give a "
                "tuple of 'success', 'failure' or None per heading, as "
                "sequences lists them."
            )
        return key  # type: ignore[return-value]

    # -- values ----------------------------------------------------------

    def _times(self, t) -> Tuple[Optional[np.ndarray], tuple]:
        """``t`` checked, and its shape (``()`` for a number or None)."""
        if t is None:
            return None, ()
        times = real_array(t, "t")
        return times, times.shape

    def _probability(
        self, path: Path, t, shape: tuple, approximation: Optional[str]
    ):
        """A path's probability at ``t``: exactly, with the headings it
        succeeds taken as certain (``"rare_event"``), or as the product
        of its branches' own probabilities (``"independent"``)."""
        if approximation == "independent":
            value: Any = np.ones(shape)
            for heading, branch in zip(self.headings, path):
                if branch is None:
                    continue
                q = self._heading_probability(heading, t, shape)
                value = value * (q if branch == "failure" else 1.0 - q)
            return value
        tree = self._sequence_tree(path, approximation)
        if tree is None:
            return np.ones(shape)
        return np.asarray(tree.top_event_probability(t), dtype=float)

    def _heading_probability(self, heading: Hashable, t, shape: tuple):
        model = self.headings[heading]
        if isinstance(model, FaultTree):
            return np.asarray(model.top_event_probability(t), dtype=float)
        return np.full(shape, model)

    def _values(self, t, approximation: Optional[str]) -> Dict[Path, Any]:
        if approximation is not None:
            one_of("approximation", approximation, APPROXIMATIONS)
        times, shape = self._times(t)
        out = {}
        for path in self.sequences:
            value = self._probability(path, times, shape, approximation)
            out[path] = float(value) if shape == () else value.reshape(shape)
        return out

    def sequence_probabilities(
        self,
        t: Optional[ArrayLike] = None,
        approximation: Optional[str] = None,
    ) -> "SequenceProbabilities":
        """Each sequence's probability given the initiating event: the
        probability that its headings fail and succeed as the path says.

        Exact by default: each sequence's tree (``sequence_tree``) by its
        decision diagram, so a basic event two headings share is one event
        and the sequences' probabilities sum to 1.

        Parameters
        ----------
        t : array_like, optional
            Time/s, a number or an array, at which the headings' events are
            taken (their probabilities of having occurred by ``t``, or of
            being down at ``t`` for repaired events). May be left out when
            no event's probability depends on time.
        approximation : str, optional
            None (the default) for the exact values; ``"rare_event"`` to
            leave out the headings each path succeeds (the probability that
            the failed ones fail together, as many tools quantify a
            sequence: it overstates each, and the sequences then sum to
            more than 1); ``"independent"`` for the product of each
            branch's own probability (its heading's alone), which is wrong
            where headings share events.

        Returns
        -------
        SequenceProbabilities
            [`SequenceProbabilities`][repyability.SequenceProbabilities]:
            ``probabilities`` and ``outcomes`` by path, floats for a number
            ``t`` and arrays for an array.

        Raises
        ------
        ValueError
            If ``t`` is left out and an event's probability depends on
            time, or ``approximation`` is unknown.

        Examples
        --------
        >>> from repyability import EventTree
        >>> tree = EventTree(0.5, [("alarm", 0.1), ("relief", 0.01)])
        >>> p = tree.sequence_probabilities().probabilities
        >>> round(p["failure", "success"], 6)  # 0.1 * 0.99
        0.099
        """
        from repyability.rbd.results import SequenceProbabilities

        return SequenceProbabilities(
            probabilities=self._values(t, approximation),
            outcomes=dict(self.sequences),
            approximation=approximation,
        )

    def sequence_frequencies(
        self,
        t: Optional[ArrayLike] = None,
        approximation: Optional[str] = None,
    ) -> "SequenceFrequencies":
        """Each sequence's frequency: the initiator's frequency times the
        sequence's probability (see ``sequence_probabilities``). Exact, the
        frequencies sum to the initiator's.

        Parameters
        ----------
        t : array_like, optional
            Time/s, as for ``sequence_probabilities``.
        approximation : str, optional
            None (the default), ``"rare_event"`` or ``"independent"``, as
            for ``sequence_probabilities``.

        Returns
        -------
        SequenceFrequencies
            [`SequenceFrequencies`][repyability.SequenceFrequencies]:
            ``frequencies`` and ``outcomes`` by path, and the
            ``initiator_frequency``.

        Raises
        ------
        ValueError
            As ``sequence_probabilities``.

        Examples
        --------
        >>> from repyability import EventTree
        >>> tree = EventTree(0.5, [("alarm", 0.1), ("relief", 0.01)])
        >>> f = tree.sequence_frequencies().frequencies
        >>> round(f["failure", "failure"], 6)  # 0.5 * 0.1 * 0.01
        0.0005
        """
        from repyability.rbd.results import SequenceFrequencies

        values = self._values(t, approximation)
        return SequenceFrequencies(
            frequencies={
                path: self.frequency * value for path, value in values.items()
            },
            outcomes=dict(self.sequences),
            initiator_frequency=self.frequency,
            approximation=approximation,
        )

    def outcome_frequencies(
        self,
        t: Optional[ArrayLike] = None,
        approximation: Optional[str] = None,
    ) -> "OutcomeFrequencies":
        """Each outcome's frequency: the sum of its sequences' (see
        ``sequence_frequencies``).

        Parameters
        ----------
        t : array_like, optional
            Time/s, as for ``sequence_probabilities``.
        approximation : str, optional
            None (the default), ``"rare_event"`` or ``"independent"``, as
            for ``sequence_probabilities``.

        Returns
        -------
        OutcomeFrequencies
            [`OutcomeFrequencies`][repyability.OutcomeFrequencies]:
            ``frequencies`` and the ``sequences`` of each outcome, in the
            order the outcomes first appear, and the
            ``initiator_frequency``.

        Raises
        ------
        ValueError
            As ``sequence_probabilities``.

        Examples
        --------
        LOPA: an overpressure (0.2 a year) leads to a release unless the
        alarm or the relief valve stops it:

        >>> from repyability import EventTree
        >>> tree = EventTree(
        ...     0.2,
        ...     [("alarm", 0.1), ("relief", 0.01)],
        ...     {
        ...         ("success", None): "safe",
        ...         ("failure", "success"): "safe",
        ...         ("failure", "failure"): "release",
        ...     },
        ... )
        >>> round(tree.outcome_frequencies().frequencies["release"], 6)
        0.0002
        """
        from repyability.rbd.results import OutcomeFrequencies

        values = self._values(t, approximation)
        frequencies: Dict[Hashable, Any] = {}
        sequences: Dict[Hashable, List[Path]] = {}
        for path, outcome in self.sequences.items():
            value = self.frequency * values[path]
            if outcome in frequencies:
                frequencies[outcome] = frequencies[outcome] + value
            else:
                frequencies[outcome] = value
            sequences.setdefault(outcome, []).append(path)
        return OutcomeFrequencies(
            frequencies=frequencies,
            sequences=sequences,
            initiator_frequency=self.frequency,
            approximation=approximation,
        )

    # -- importance ------------------------------------------------------

    def _outcome_paths(self, outcome: Hashable) -> List[Path]:
        paths = [p for p, o in self.sequences.items() if o == outcome]
        if not paths:
            named = list(dict.fromkeys(self.sequences.values()))
            raise ValueError(
                f"{outcome!r} is not an outcome of the event tree: give one "
                f"of {named}."
            )
        return paths

    def _conditioned(self, outcome: Hashable, t) -> tuple:
        """The outcome's frequency, and with each event held occurred and
        held not occurred, from its sequences' trees conditioned on it (an
        event a sequence's tree does not have leaves it as it is)."""
        paths = self._outcome_paths(outcome)
        times, shape = self._times(t)
        total: Any = np.zeros(shape)
        occurred = {e: np.zeros(shape) for e in self.events}
        not_occurred = {e: np.zeros(shape) for e in self.events}
        for path in paths:
            tree = self._sequence_tree(path, None)
            if tree is None:
                total = total + 1.0
                for e in self.events:
                    occurred[e] = occurred[e] + 1.0
                    not_occurred[e] = not_occurred[e] + 1.0
                continue
            given = tree.conditional_probabilities(times)
            top = np.asarray(given.top, dtype=float)
            total = total + top
            for e in self.events:
                occurred[e] = occurred[e] + np.asarray(
                    given.given_occurred.get(e, top), dtype=float
                )
                not_occurred[e] = not_occurred[e] + np.asarray(
                    given.given_not_occurred.get(e, top), dtype=float
                )
        f = self.frequency
        return (
            f * total,
            {e: f * v for e, v in occurred.items()},
            {e: f * v for e, v in not_occurred.items()},
            shape,
        )

    @staticmethod
    def _out(values: Dict[Hashable, Any], shape: tuple) -> dict:
        return {
            e: float(v) if shape == () else np.asarray(v).reshape(shape)
            for e, v in values.items()
        }

    def birnbaum_importance(
        self, outcome: Hashable, t: Optional[ArrayLike] = None
    ) -> dict:
        """Birnbaum importance of each basic event to an outcome's
        frequency: the outcome's frequency given that the event has
        occurred less that given that it has not, the rate at which the
        frequency rises with the event's probability.

        Exact: each of the outcome's sequences' trees conditioned on the
        event. An event can lower an outcome (a success branch needs it
        not to occur), so the importance can be negative.

        Parameters
        ----------
        outcome : Hashable
            One of the outcomes (see ``sequences``).
        t : array_like, optional
            Time/s, as for ``sequence_probabilities``.

        Returns
        -------
        dict
            ``{event: importance}`` over ``events``, in the outcome's
            frequency's unit: floats for a number ``t``, arrays for an
            array.

        Raises
        ------
        ValueError
            If ``outcome`` is not an outcome of the tree, or ``t`` is left
            out and an event's probability depends on time.

        Examples
        --------
        >>> from repyability import EventTree
        >>> tree = EventTree(0.5, [("alarm", 0.1), ("relief", 0.01)])
        >>> b = tree.birnbaum_importance(("failure", "failure"))
        >>> round(b["relief"], 6)  # 0.5 * 0.1
        0.05
        """
        _, occurred, not_occurred, shape = self._conditioned(outcome, t)
        return self._out(
            {e: occurred[e] - not_occurred[e] for e in self.events}, shape
        )

    def fussell_vesely(
        self, outcome: Hashable, t: Optional[ArrayLike] = None
    ) -> dict:
        """Fussell-Vesely importance of each basic event to an outcome's
        frequency: the share of the frequency that the event's occurring
        takes part in, ``(F - F given e did not occur) / F``.

        This is the form PSA codes report (``1 - 1 / RRW``), exact here on
        the sequences' trees conditioned on the event; under rare events
        it is the share of the outcome's minimal cut sets holding the
        event. Where a success branch needs the event not to occur, it can
        be negative. NaN where the outcome's frequency is 0.

        Parameters
        ----------
        outcome : Hashable
            One of the outcomes (see ``sequences``).
        t : array_like, optional
            Time/s, as for ``sequence_probabilities``.

        Returns
        -------
        dict
            ``{event: importance}`` over ``events``: floats for a number
            ``t``, arrays for an array.

        Raises
        ------
        ValueError
            If ``outcome`` is not an outcome of the tree, or ``t`` is left
            out and an event's probability depends on time.

        Examples
        --------
        >>> from repyability import EventTree
        >>> tree = EventTree(0.5, [("alarm", 0.1), ("relief", 0.01)])
        >>> fv = tree.fussell_vesely(("failure", "failure"))
        >>> round(fv["relief"], 6)
        1.0
        """
        total, _, not_occurred, shape = self._conditioned(outcome, t)
        return self._out(
            _measures.fussell_vesely(
                {e: total - not_occurred[e] for e in self.events}, total
            ),
            shape,
        )

    # -- routes ----------------------------------------------------------

    def analysis_routes(self) -> AnalysisRoutes:
        """How each analysis of this event tree is computed, found without
        running it: exactly, numerically, by simulation, or not at all.

        Each is its sequences' trees' own (see ``sequence_tree``): the
        sequences' probabilities their ``top_event_probability``'s, the
        importances their ``conditional_probabilities``', the least exact
        of them; the frequencies take the initiator tree's
        ``system_failure_frequency`` in too.

        Returns
        -------
        AnalysisRoutes
            An [`AnalysisRoute`][repyability.AnalysisRoute] for each of
            ``sequence_probabilities``, ``sequence_frequencies``,
            ``outcome_frequencies``, ``birnbaum_importance`` and
            ``fussell_vesely``.

        Examples
        --------
        >>> from repyability import EventTree
        >>> tree = EventTree(0.5, [("alarm", 0.1), ("relief", 0.01)])
        >>> tree.analysis_routes()["outcome_frequencies"].route
        'exact'
        """
        from repyability.rbd import routes

        trees = [
            tree
            for (_, approximation), tree in self._trees.items()
            if approximation is None and tree is not None
        ]
        reports = [tree.analysis_routes() for tree in trees]
        given = _worst(
            "Each sequence's tree by its decision diagram, exactly, "
            "shared events one event.",
            [r["top_event_probability"] for r in reports],
        )
        if isinstance(self.initiator, FaultTree):
            initiated = self.initiator.analysis_routes()
            frequency = _worst(
                "The initiator tree's long-run frequency times each "
                "sequence's probability.",
                [initiated["system_failure_frequency"], given],
            )
        else:
            frequency = AnalysisRoute(
                given.route,
                "The initiator's frequency times each sequence's "
                "probability. " + given.reason,
                given.nodes,
            )
        if given.route == routes.REFUSED:
            frequency = given
        importance = _worst(
            "Each of the outcome's sequences' trees conditioned on the "
            "event, exactly.",
            [r["conditional_probabilities"] for r in reports],
        )
        return AnalysisRoutes(
            {
                "sequence_probabilities": given,
                "sequence_frequencies": frequency,
                "outcome_frequencies": frequency,
                "birnbaum_importance": importance,
                "fussell_vesely": importance,
            }
        )

    # -- saving ----------------------------------------------------------

    def to_dict(self) -> dict:
        """The event tree as a JSON-compatible dict (see ``from_dict``).

        Returns
        -------
        dict
            The initiator (a frequency, or its tree's ``to_dict()``), the
            headings in order (each a probability or its tree's
            ``to_dict()``) and the paths with their outcomes.
        """
        from repyability._version import __version__

        def saved(model):
            if isinstance(model, FaultTree):
                return {"tree": model.to_dict()}
            return {"probability": model}

        return {
            "repyability_version": __version__,
            "type": "EventTree",
            "initiator": (
                {"tree": self.initiator.to_dict()}
                if isinstance(self.initiator, FaultTree)
                else {"frequency": self.initiator}
            ),
            "headings": [
                {"heading": name, **saved(model)}
                for name, model in self.headings.items()
            ],
            "sequences": [
                {"path": list(path), "outcome": outcome}
                for path, outcome in self.sequences.items()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventTree":
        """Rebuild an event tree from ``to_dict``'s output.

        Parameters
        ----------
        d : dict
            As ``to_dict`` returns.

        Returns
        -------
        EventTree
            The event tree.

        Raises
        ------
        ValueError
            If ``d`` is not an event tree's dict.
        """
        from repyability.rbd.serialisation import _node_name

        if d.get("type") != "EventTree":
            raise ValueError(f"Not an event tree: type {d.get('type')!r}.")

        def loaded(entry, number):
            if "tree" in entry:
                return FaultTree.from_dict(entry["tree"])
            return entry[number]

        return cls(
            loaded(d["initiator"], "frequency"),
            [
                (_node_name(h["heading"]), loaded(h, "probability"))
                for h in d["headings"]
            ],
            {
                tuple(s["path"]): _node_name(s["outcome"])
                for s in d["sequences"]
            },
        )

    def to_json(self, fp=None, **json_kwargs) -> Optional[str]:
        """The event tree as a JSON document (see ``to_dict``): returned,
        or written to ``fp``.

        Parameters
        ----------
        fp : file-like, optional
            Where to write the document; by default it is returned.
        **json_kwargs
            Passed to ``json.dumps`` (``indent``, say).

        Returns
        -------
        str or None
            The document, or None when written to ``fp``.

        Examples
        --------
        >>> from repyability import EventTree
        >>> tree = EventTree(0.5, [("alarm", 0.1), ("relief", 0.01)])
        >>> EventTree.from_json(tree.to_json()).frequency
        0.5
        """
        text = json.dumps(self.to_dict(), **json_kwargs)
        if fp is None:
            return text
        fp.write(text)
        return None

    @classmethod
    def from_json(cls, s) -> "EventTree":
        """Rebuild an event tree from ``to_json``'s document.

        Parameters
        ----------
        s : str or file-like
            The document, or a file to read it from.

        Returns
        -------
        EventTree
            The event tree.
        """
        text = s if isinstance(s, str) else s.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return (
            f"EventTree(frequency={self.frequency!r}, headings="
            f"{list(self.headings)!r}, {len(self.sequences)} sequences)"
        )


def _initiator(initiator) -> Tuple[Any, float]:
    """The initiator as kept, and its frequency."""
    if isinstance(initiator, FaultTree):
        return initiator, float(initiator.system_failure_frequency())
    if not is_number(initiator) or not float(initiator) >= 0.0:
        raise ValueError(
            "initiator must be the initiating event's frequency, a number "
            "0 or more, or a FaultTree of repaired events whose long-run "
            f"frequency of its top event it is, got {initiator!r}."
        )
    if not np.isfinite(float(initiator)):
        raise ValueError(
            f"initiator must be a finite frequency, got {initiator!r}."
        )
    return float(initiator), float(initiator)


def _headings(headings) -> Dict[Hashable, Any]:
    """The headings as ``{name: FaultTree or probability}``, in order."""
    usage = (
        "headings must be a non-empty list of (name, FaultTree or "
        f"probability) pairs, got {headings!r}."
    )
    pairs = list(headings.items()) if isinstance(headings, Mapping) else None
    if pairs is None:
        try:
            pairs = [tuple(pair) for pair in headings]
        except TypeError:
            raise ValueError(usage) from None
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError(usage)
    out: Dict[Hashable, Any] = {}
    for name, model in pairs:
        try:
            hash(name)
        except TypeError:
            raise ValueError(
                f"A heading's name must be hashable, got {name!r}."
            ) from None
        if name in out:
            raise ValueError(f"Heading {name!r} is given more than once.")
        out[name] = _heading(name, model)
    return out


def _heading(name, model):
    """A heading's model: a tree evaluated exactly, or a probability."""
    if isinstance(model, FaultTree):
        if model.is_dynamic:
            raise NotImplementedError(
                f"Heading {name!r}: a tree with dynamic gates (PAND, SPARE, "
                "SEQ) is not supported, as a sequence takes the NOT of a "
                "heading it succeeds, which has no agreed meaning over an "
                "order of failures. Ask the tree's top_event_probability "
                "and give the heading as that probability, if it shares "
                "no event with the others."
            )
        if model.ccf_groups:
            raise NotImplementedError(
                f"Heading {name!r}: a tree with common-cause groups is not "
                "supported, as a sequence's NOT gates do not take them. "
                "Write each shared cause as a basic event of its own "
                "under the members it fails (a shared event)."
            )
        return model
    if not is_number(model) or not 0.0 <= float(model) <= 1.0:
        raise ValueError(
            f"Heading {name!r} must be a FaultTree or the probability that "
            f"it fails, a number in [0, 1], got {model!r}."
        )
    return float(model)


def _same(a, b) -> bool:
    """Whether two events' models are one: one object, equal
    probabilities, or models saved alike."""
    if a is b:
        return True
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and a == b
    from repyability.rbd._model_utils import SAVE_ERRORS
    from repyability.rbd.serialisation import (
        _serialise_component,
        serialise_model,
    )

    save = _serialise_component if isinstance(a, dict) else serialise_model
    if isinstance(a, dict) != isinstance(b, dict):
        return False
    try:
        return save(a) == save(b)
    except SAVE_ERRORS:
        return False


def _named_apart(heading, tree: FaultTree) -> Dict[Hashable, tuple]:
    """A heading tree's gates as given, each named ``(heading, gate)``."""

    def named(x):
        return (heading, x) if x in tree.gates else x

    out: Dict[Hashable, tuple] = {}
    for gate, spec in tree.gates.items():
        kind, *rest = spec
        if kind == "fdep":
            trigger, dependents = rest
            out[heading, gate] = (kind, named(trigger), list(dependents))
        else:
            *head, inputs = rest
            out[heading, gate] = (kind, *head, [named(x) for x in inputs])
    return out


def _sequences(headings: list, sequences) -> Dict[Path, Hashable]:
    """The paths and their outcomes, checked to partition the initiator;
    by default every combination its own outcome."""
    n = len(headings)
    if sequences is None:
        return {
            path: path
            for path in itertools.product(("success", "failure"), repeat=n)
        }
    if not isinstance(sequences, Mapping) or not sequences:
        raise ValueError(
            "sequences must be a non-empty dict {path: outcome}, each path "
            "a tuple of 'success', 'failure' or None (not asked) per "
            f"heading, got {sequences!r}."
        )
    out: Dict[Path, Hashable] = {}
    for path, outcome in sequences.items():
        if (
            not isinstance(path, tuple)
            or len(path) != n
            or any(branch not in BRANCHES for branch in path)
        ):
            raise ValueError(
                f"Path {path!r} must be a tuple of {n} entries, one per "
                f"heading {headings}, each 'success', 'failure' or None "
                "(the path does not ask the heading)."
            )
        out[path] = outcome
    paths = list(out)
    for a, b in itertools.combinations(paths, 2):
        if all(x is None or y is None or x == y for x, y in zip(a, b)):
            both = tuple(x if x is not None else y for x, y in zip(a, b))
            raise ValueError(
                f"Paths {a!r} and {b!r} overlap: the headings' outcome "
                f"{_filled(both)!r} is on both. Each combination of "
                "successes and failures must be on exactly one path."
            )
    covered = sum(2 ** path.count(None) for path in paths)
    if covered < 2**n:
        missing = _uncovered(paths, n)
        raise ValueError(
            f"The paths cover {covered} of the headings' {2 ** n} "
            f"combinations of successes and failures; "
            f"{missing!r} is on none. Add a path for it, with None where "
            "a heading is not asked."
        )
    return out


def _filled(path: tuple) -> tuple:
    """A combination on a partial path: each heading not asked succeeds."""
    return tuple("success" if x is None else x for x in path)


def _uncovered(paths: List[Path], n: int) -> tuple:
    """A combination of successes and failures on none of ``paths``,
    found by splitting on each heading in turn."""
    prefix: List[str] = []
    live = paths
    for i in range(n):
        for branch in ("success", "failure"):
            narrowed = [p for p in live if p[i] in (None, branch)]
            room = 2 ** (n - i - 1)
            held = sum(2 ** p[i + 1 :].count(None) for p in narrowed)
            if held < room:
                prefix.append(branch)
                live = narrowed
                break
    return tuple(prefix)


def _worst(reason: str, parts: List[AnalysisRoute]) -> AnalysisRoute:
    """The route of a value made of ``parts``: the first refusal among
    them, or the least exact of their routes (exact where there are
    none)."""
    from repyability.rbd import routes

    if not parts:
        return AnalysisRoute(routes.EXACT, reason)
    for part in parts:
        if part.route == routes.REFUSED:
            return part
    order = (routes.EXACT, routes.NUMERICAL, routes.SIMULATED)
    worst = max(parts, key=lambda part: order.index(part.route))
    return AnalysisRoute(worst.route, f"{reason} {worst.reason}", worst.nodes)
