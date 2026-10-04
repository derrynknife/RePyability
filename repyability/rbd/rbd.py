"""The reliability block diagram base class and its exact engine.

``RBD`` holds a diagram's structure and the computations that need only the
structure and per-node probabilities; ``NonRepairableRBD`` and
``RepairableRBD`` build on it. Its exact engine reduces the diagram to
series, parallel and k-out-of-n modules (``modular.py``) and works out what
is left from its minimal path sets by a memoised Shannon decomposition
(``shannon.py``). The module-level functions here are the exact probability
that at least one set of elements fully works, minimal cut sets from minimal
path sets, and the probability scaling used by reliability allocation.
"""

import difflib
import functools
import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import (
    BFGS,
    NonlinearConstraint,
    OptimizeResult,
    brentq,
    minimize,
)
from scipy.sparse import diags
from scipy.special import expit as sigmoid
from scipy.special import logit, logsumexp, softmax

from repyability.rbd import bdd
from repyability.rbd import capacity as _capacity
from repyability.rbd.modular import (
    Decomposition,
    GraphStructure,
    decompose,
    fold,
)
from repyability.rbd.rbd_graph import RBDGraph
from repyability.rbd.results import CapacityDistribution
from repyability.rbd.shannon import (
    _evaluate_shannon_plan,
    _minimal_cut_sets,
    _shannon_plan,
)
from repyability.utils.checks import is_whole, structure_method
from repyability.utils.wrappers import check_probability

_ON_INFEASIBLE_RBD = ("raise", "warn", "ignore")


def _check_on_infeasible_rbd(value: Any) -> None:
    """Raise unless ``value`` is an ``on_infeasible_rbd`` option. It is
    checked on construction whether or not the diagram is valid, so a typo
    cannot pass unnoticed."""
    if value not in _ON_INFEASIBLE_RBD:
        raise ValueError(
            "'on_infeasible_rbd' must be one of {'raise', 'warn', 'ignore'}, "
            f"got {value!r}."
        )


def leaves_out_junctions(method):
    """A measure's or an allocation's per-node results without the
    diagram's perfect junction nodes (see ``RBD._junctions``): drawing
    devices, such as a k-out-of-n vote, that never fail and cannot be
    improved. A dict of such results (by measure) loses them from each."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        junctions = self._junctions()
        if not junctions or not isinstance(result, dict):
            return result

        def strip(values: dict) -> dict:
            return {n: v for n, v in values.items() if n not in junctions}

        if result and all(isinstance(v, dict) for v in result.values()):
            return {key: strip(values) for key, values in result.items()}
        return strip(result)

    return wrapper


def _set_order(nodes: frozenset) -> tuple:
    """Path and cut sets' order in a list: smallest first, then by name (as
    a ``FaultTree`` lists them)."""
    return (len(nodes), sorted(map(str, nodes)))


def _names(nodes) -> str:
    """Node names for a message, in a stable order."""
    return ", ".join(sorted((repr(n) for n in nodes), key=str))


def _close_name(name, candidates) -> Optional[Any]:
    """The one of ``candidates`` whose name is closest to ``name``'s (a
    likely typo), if any is close."""
    by_text = {str(c): c for c in candidates}
    close = difflib.get_close_matches(str(name), list(by_text), n=1)
    return by_text[close[0]] if close else None


def structure_problems(check: dict) -> List[str]:
    """What a diagram's ``structure_check`` found wrong, one sentence each
    (see ``RBD``), in the order a user would fix them."""
    lines = []
    missing = list(check.get("nodes_with_no_model", ()))
    unused = list(check.get("nodes_in_no_edge", ()))
    # A node a misspelt model was meant for needs no hint.
    meant = {_close_name(node, missing) for node in unused}
    for node in missing:
        line = f"node {node!r} (in the edges) has no model"
        if node not in meant:
            line += (
                " (a junction, such as a k-out-of-n vote point, takes "
                "PerfectReliability)"
            )
        lines.append(line)
    for node in unused:
        line = f"model {node!r} is not a node in the edges"
        close = _close_name(node, missing)
        if close is not None:
            line += f"; did you mean {close!r}?"
        lines.append(line)
    for cycle in check.get("cycles", ()):
        lines.append(f"there is a cycle through {_names(cycle)}")
    if check.get("is_empty"):
        return lines + [
            "the diagram has no edges: give them, from the input node to the "
            "output node"
        ]
    sources = check.get("nodes_with_no_predecessors", ())
    if not check.get("has_unique_input_node", True) and not sources:
        lines.append(
            "every node has an incoming edge, so there is no input node (a "
            "node with none)"
        )
    elif not check.get("has_unique_input_node", True):
        lines.append(
            "more than one node has no incoming edges, so the input node is "
            f"not clear: {_names(sources)} (only the input node has none)"
        )
    elif sources:
        lines.append(
            f"node(s) {_names(sources)} have no incoming edges, which only "
            f"the input node, {check['input_node']!r}, may have"
        )
    sinks = check.get("nodes_with_no_successors", ())
    if not check.get("has_unique_output_node", True) and not sinks:
        lines.append(
            "every node has an outgoing edge, so there is no output node (a "
            "node with none)"
        )
    elif not check.get("has_unique_output_node", True):
        lines.append(
            "more than one node has no outgoing edges, so the output node is "
            f"not clear: {_names(sinks)} (every node but the output needs an "
            "edge onward)"
        )
    elif sinks:
        lines.append(
            f"node(s) {_names(sinks)} have no outgoing edges, which only the "
            f"output node, {check['output_node']!r}, may have"
        )
    lines.extend(check.get("koon_errors", ()))
    return lines


def structure_message(check: dict) -> str:
    """The message an infeasible diagram raises or warns with: what its
    ``structure_check`` found, one line each."""
    lines = structure_problems(check) or ["see structure_check"]
    return "RBD not correctly structured:\n" + "\n".join(
        f"  - {line}" for line in lines
    )


def log_linearly_scale_probabilities(p: float, x: float) -> np.ndarray:
    """Scale a probability by shifting the log of its complement.

    Returns ``1 - (1 - p) * exp(-x)``: the complement ``1 - p`` (the
    unreliability, when ``p`` is a reliability) is multiplied by
    ``exp(-x)``, i.e. ``log(1 - p)`` is shifted by ``-x``. A positive ``x``
    moves ``p`` towards 1 and a negative ``x`` moves it down. The result is
    not clipped, so a negative enough ``x`` gives a value below 0. A ``p``
    of exactly 1 is returned unchanged (avoiding ``log(0)``). This is the
    scaling used by ``RBD.improvement_allocation``.

    Parameters
    ----------
    p : float
        A single probability, in [0, 1].
    x : float
        The amount by which ``log(1 - p)`` is decreased.

    Returns
    -------
    np.ndarray
        The scaled probability, as a one-element array.

    Examples
    --------
    ``x = log(2)`` halves the unreliability of a 0.9-reliable node:

    >>> import numpy as np
    >>> from repyability.rbd.rbd import log_linearly_scale_probabilities
    >>> p = log_linearly_scale_probabilities(0.9, np.log(2))
    >>> round(float(p[0]), 4)
    0.95
    """
    if p == 1.0:
        return np.atleast_1d(1.0)
    else:
        return np.atleast_1d(1 - np.exp(-(-np.log(1 - p) + x)))


def scale_probability_dict(
    node_probabilities: Dict[str, float],
    x: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Log-linearly scale every probability in a dict.

    Each entry ``p`` with weight ``w`` becomes ``1 - (1 - p) * exp(-x * w)``
    (see ``log_linearly_scale_probabilities``): its complement is
    multiplied by ``exp(-x * w)``. This is the scaling used by
    ``RBD.improvement_allocation``.

    Parameters
    ----------
    node_probabilities : Dict[str, float]
        The probabilities to scale (single floats), keyed by node name.
    x : float
        The common scale factor.
    weights : Optional[Dict[str, float]], optional
        A weight per node, by default None (a weight of 1.0 for every node).
        If given, it needs an entry for every key of ``node_probabilities``.

    Returns
    -------
    Dict[str, np.ndarray]
        The scaled probabilities, as one-element arrays, with the same keys.

    Raises
    ------
    KeyError
        If ``weights`` is given without an entry for a key of
        ``node_probabilities``.

    Examples
    --------
    With ``x = log(2)``, a weight of 1 halves an unreliability and a weight
    of 2 quarters it:

    >>> import numpy as np
    >>> from repyability.rbd.rbd import scale_probability_dict
    >>> scaled = scale_probability_dict(
    ...     {"a": 0.9, "b": 0.6}, np.log(2), weights={"a": 1.0, "b": 2.0}
    ... )
    >>> {k: round(float(v[0]), 4) for k, v in sorted(scaled.items())}
    {'a': 0.95, 'b': 0.9}
    """
    out = {}
    if weights is None:
        weights = defaultdict(lambda: 1.0)

    # Iterate through the input dictionary of node probabilities
    for k, p in node_probabilities.items():
        # Apply log-linear scaling to the node probability value using the
        # scaling factor and weight for the node
        out[k] = log_linearly_scale_probabilities(p, x * weights[k])

    return out


def probability_any_set_satisfied(
    sets: Iterable[frozenset],
    element_probabilities: Dict[Any, np.ndarray],
    array_shape,
) -> np.ndarray:
    """Exact probability that at least one of the given sets is fully active.

    Each "set" is a collection of elements (e.g. a minimal path set of
    components); the set is "satisfied" when *all* of its elements are active.
    Given the per-element probability of being active, with the elements
    independent, this returns the probability that *at least one* set is
    satisfied.

    With path sets and node reliabilities this is the system reliability; with
    cut sets and node unreliabilities it is the system unreliability.

    The computation is an exact Shannon decomposition of the structure
    function: it repeatedly conditions on a single element being active or
    inactive,

        P(S) = p_e * P(S | e active) + (1 - p_e) * P(S | e inactive),

    and memoises sub-problems. Because shared sub-functions are only solved
    once, this avoids the 2^(#sets) blow-up of the inclusion-exclusion
    principle while returning the identical exact result.

    Parameters
    ----------
    sets : Iterable[frozenset]
        The collection of sets (e.g. minimal path sets or cut sets). No sets
        at all gives probability 0; an empty set is always satisfied, giving
        1.
    element_probabilities : Dict[Any, np.ndarray]
        Maps each element of the sets to its probability of being active (a
        float, or an array of shape ``array_shape``).
    array_shape : int or tuple of int
        The shape of the probability arrays (e.g. their length), used to
        seed the 0/1 base cases.

    Returns
    -------
    np.ndarray
        The probability that at least one set is fully active, an array of
        shape ``array_shape``.

    Raises
    ------
    KeyError
        If an element the result depends on has no entry in
        ``element_probabilities``.

    Examples
    --------
    Two sets sharing ``"c"``, so the answer is ``(1 - 0.1 * 0.2) * 0.95``:

    >>> from repyability.rbd.rbd import probability_any_set_satisfied
    >>> sets = [frozenset({"a", "c"}), frozenset({"b", "c"})]
    >>> p = {"a": 0.9, "b": 0.8, "c": 0.95}
    >>> round(float(probability_any_set_satisfied(sets, p, 1)[0]), 4)
    0.931
    """
    return _evaluate_shannon_plan(
        _shannon_plan(sets), element_probabilities, array_shape
    )


def _averaged(values: Any, weights: Optional[np.ndarray]) -> Any:
    """``values`` averaged over their first axis with ``weights`` (e.g. over
    time), as a one-element array; unchanged without weights."""
    if weights is None:
        return values
    return np.atleast_1d(np.asarray(weights) @ np.asarray(values))


def _probability_value(label: str, value: Any) -> float:
    """``value`` as a single probability in [0, 1], or a ValueError naming
    ``label``."""
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(
            f"{label} must be a single probability, got {array.size} values."
        )
    if not 0.0 <= array.item() <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {array.item()}.")
    return array.item()


def minimal_cut_sets_from_path_sets(
    path_sets: Iterable[frozenset],
) -> set[frozenset]:
    """Return the minimal cut sets given the minimal path sets.

    A minimal cut set is a minimal "transversal" (hitting set) of the path
    sets: a set of components that intersects every path set, with no proper
    subset that also does, so that failing those components breaks every path
    through the system. Because it
    works directly from the path sets, this stays correct for k-out-of-n
    structures, whose k-of-n behaviour is already encoded in the path sets.

    The cut sets are read off the same Shannon decomposition the exact engine
    uses (see ``_shannon_plan`` in ``shannon.py``), which shares every repeated
    sub-problem. At a step pivoting on component ``x``, the system works as
    ``f1`` if ``x`` works and ``f0`` if it has failed, with ``f0 <= f1`` (a
    coherent system never works *better* for a failure). A minimal cut set
    either spares ``x`` -- a minimal cut set of ``f1`` -- or contains it, with
    the rest a minimal cut set of ``f0`` that contains none of ``f1``'s
    (otherwise ``x`` would be redundant). Working up from the constant
    functions (no cut set can stop a system that always works; the empty set
    stops one that never does) gives the root's minimal cut sets. Unlike
    building the transversals one path set at a time (Berge's algorithm),
    whose intermediate families can grow far larger than the answer, this
    only ever holds each sub-problem's own minimal cut sets.

    Parameters
    ----------
    path_sets : Iterable[frozenset]
        The minimal path sets (each a set, or other iterable, of
        components).

    Returns
    -------
    set[frozenset]
        The minimal cut sets. No path sets at all gives ``{frozenset()}``
        (the system never works, so the empty set is a cut set); an empty
        path set gives no cut sets (the system always works).

    Examples
    --------
    Two parallel components ``"a"`` and ``"b"`` in series with ``"c"``:

    >>> from repyability.rbd.rbd import minimal_cut_sets_from_path_sets
    >>> cuts = minimal_cut_sets_from_path_sets([{"a", "c"}, {"b", "c"}])
    >>> sorted(sorted(c) for c in cuts)
    [['a', 'b'], ['c']]
    """
    return _minimal_cut_sets(_shannon_plan(path_sets))


def _capacity_number(node, value: Any, what: str) -> float:
    """A capacity level, checked: a positive number (``inf`` for no limit),
    as a float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"The {what} of node {node!r} must be a number, got {value!r}."
        )
    value = float(value)
    if not value > 0.0:
        raise ValueError(
            f"The {what} of node {node!r} must be positive (inf for no "
            f"limit), got {value!r}."
        )
    return value


def _capacity_levels(node, levels: dict) -> dict:
    """A node's capacity levels while it works, ``{level: probability}``,
    checked: positive levels and probabilities adding up to 1. Returned in
    increasing order of level, scaled to add up to 1 exactly."""
    if not levels:
        raise ValueError(
            f"The capacity levels of node {node!r} are empty: give at least "
            "one {level: probability}."
        )
    out = {}
    for level, probability in levels.items():
        level = _capacity_number(node, level, "capacity level")
        if isinstance(probability, (bool, np.bool_)) or not isinstance(
            probability, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"The probability of capacity {level!r} of node {node!r} "
                f"must be a number, got {probability!r}."
            )
        if not 0.0 < float(probability) <= 1.0:
            raise ValueError(
                f"The probability of capacity {level!r} of node {node!r} "
                f"must be in (0, 1], got {probability!r}."
            )
        out[level] = float(probability)
    total = sum(out.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"The probabilities of the capacity levels of node {node!r} "
            f"must add up to 1 (they are the levels it works at), got "
            f"{total!r}."
        )
    return {level: out[level] / total for level in sorted(out)}


class Pairs(dict):
    """A symmetric measure of pairs (``joint_importance``): each pair once,
    its two names in order as text (so that it does not depend on the order
    the nodes were given in), and found either way round."""

    @staticmethod
    def oriented(i: Hashable, j: Hashable) -> Tuple[Hashable, Hashable]:
        """The pair as it is kept: its names in order as text."""
        return (j, i) if str(j) < str(i) else (i, j)

    def _swapped(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return (key[1], key[0])
        return None

    def __missing__(self, key):
        swapped = self._swapped(key)
        if swapped is not None and dict.__contains__(self, swapped):
            return dict.__getitem__(self, swapped)
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        swapped = self._swapped(key)
        return dict.__contains__(self, key) or (
            swapped is not None and dict.__contains__(self, swapped)
        )

    def get(self, key, default=None):
        return self[key] if key in self else default


class RBD:
    """Reliability block diagram structure: the base of the RBD classes.

    An RBD is a directed acyclic graph from a single input node (the only
    node with no incoming edges) to a single output node (the only node with
    no outgoing edges). Every other node is an intermediate node, i.e. a
    component. The input and output nodes are not components: they are
    perfectly reliable, and are left out of
    [`node_names`][repyability.RBD.node_names], of the node probabilities
    the methods need, and of the nodes that can be forced working or
    broken. The system works when the output node is reached: the input
    node is always reached, and any other node is reached when it works
    and at least ``k`` of its predecessors are reached, where ``k`` is its
    k-out-of-n value (1 unless set with ``k``). Node names can be any
    hashable, e.g. strings or integers.

    This class holds what depends only on the structure, or on the
    structure and given per-node probabilities: minimal path and cut sets,
    the structure function, the exact system probability, structural
    importance, reliability allocation and serialisation. It is usually
    used through [`NonRepairableRBD`][repyability.NonRepairableRBD] or
    [`RepairableRBD`][repyability.RepairableRBD], which add node models
    (their model argument takes the place of ``nodes``). It can also be
    built from ``edges`` alone to analyse a structure with no models.
    Probability calculations assume the nodes are independent.

    Nodes can also be given a ``capacity``, their throughput while they
    work, for the capacity analysis (see
    [`system_capacity`][repyability.RBD.system_capacity]): the system's
    capacity is then the most that can flow from the input to the output.

    The structure is validated on construction. It is infeasible if it has
    a cycle; if it has not exactly one node with no incoming edges, or not
    exactly one with no outgoing edges (e.g. a node in ``nodes`` that is in
    no edge); if a ``k`` is 0 or greater than the node's number of incoming
    edges; or if ``k`` names a node not in the diagram. The subclasses add
    that every node in the edges but the input and output has a model, and
    that every model is for a node in the edges. ``on_infeasible_rbd`` sets
    what then happens: the error or warning lists each finding on a line of
    its own (a model in no edge with the node it was likely meant for), and
    the full report is kept in ``structure_check``. On construction the
    diagram is also reduced to its modules: the series, parallel and
    k-out-of-n parts, each of which has a closed form, leaving only the rest
    (e.g. a bridge) to be worked out from its minimal path sets. This keeps
    large redundant diagrams fast, and a series-parallel diagram never needs
    its path sets, however many there are (see
    [`system_probability`][repyability.RBD.system_probability]). The
    minimal path and cut sets are found on first use and cached, so repeated
    evaluations are cheap.

    Parameters
    ----------
    edges : Iterable[tuple[Hashable, Hashable]]
        The directed edges ``(from_node, to_node)`` of the diagram, e.g.
        ``[("s", "a"), ("a", "t")]`` for the single component ``"a"``
        between input ``"s"`` and output ``"t"``.
    nodes : Iterable, optional
        Node names that must be in the diagram as well as those in
        ``edges``, by default None. A name that is in no edge is an isolated
        node, which makes the structure infeasible. (The subclasses check
        their models' names against the edges instead.)
    k : dict[Any, int], optional
        The k-out-of-n value of nodes, keyed by node name, by default None
        (every node has ``k = 1``).
    input_node : Any, optional
        The input node, by default None: it is found as the node with no
        incoming edges. If given, it must be that node: a name not in the
        diagram, or a node with incoming edges, raises a ValueError. It
        cannot make a diagram with several such nodes feasible.
    output_node : Any, optional
        The output node, by default None: it is found as the node with no
        outgoing edges. The same rules as for ``input_node`` apply.
    on_infeasible_rbd : str, optional
        What to do if the structure is infeasible, by default ``"raise"``:
        ``"raise"`` raises a ValueError, ``"warn"`` issues a UserWarning
        containing ``structure_check`` and builds the RBD anyway, and
        ``"ignore"`` builds it silently. An infeasible RBD may give errors
        or wrong results later.
    capacity : dict[Any, float or dict], optional
        Each node's capacity, keyed by node name, by default None: the
        throughput it passes while it works, a positive number in any unit
        (the same for every node); or, for a node that works at several
        levels, a dict ``{level: probability}`` of the levels and the
        probability of each while it works (adding up to 1). A failed node
        passes nothing, and a node with no capacity given limits nothing:
        it passes whatever reaches it while it works (``inf``). The edges
        carry any amount, so a series chain passes the least of its nodes'
        capacities and a parallel group the sum; a k-out-of-n node passes
        flow only while at least ``k`` of its inputs are reached. Only the
        capacity analysis uses them.

    Attributes
    ----------
    G : RBDGraph
        The diagram, a ``networkx.DiGraph`` subclass; each node's ``"k"``
        attribute is its k-out-of-n value.
    input_node : Hashable
        The input node, or None if it could not be found (only possible
        when an infeasible RBD is built with ``"warn"`` or ``"ignore"``).
    output_node : Hashable
        The output node, or None if it could not be found.
    in_or_out : list
        ``[input_node, output_node]``.
    nodes : list
        The intermediate nodes, in the order they were added to the graph
        (the same list [`node_names`][repyability.RBD.node_names] returns).
    capacity : dict
        The capacities given, keyed by node name: floats, or dicts of
        levels and their probabilities (in increasing order of level).
    structure_check : dict
        The validation report, e.g. ``"is_valid"``, ``"has_cycles"``,
        ``"cycles"``, ``"koon_errors"``, ``"koon_warnings"``,
        ``"irrelevant_nodes"`` and ``"is_too_meshed"``: whether the core
        is too meshed to work out exactly, so that only the simulations run
        (its irrelevant nodes are then not known). The subclasses add their
        own entries.

    Raises
    ------
    ValueError
        If ``input_node`` or ``output_node`` is not a node of the diagram,
        or is not its source or sink (whatever ``on_infeasible_rbd`` is);
        if a capacity is not a positive number or a dict of positive levels
        with probabilities adding up to 1, or is given for the input or
        output node or a node not in the diagram;
        if the structure is infeasible
        and ``on_infeasible_rbd`` is ``"raise"`` (the message lists what is
        wrong, one finding a line); or if ``on_infeasible_rbd`` is not one
        of its three values.

    Examples
    --------
    Two redundant pumps feeding a valve. The subclasses supply the node
    models; the structure methods come from this class:

    >>> from surpyval import FixedEventProbability
    >>> from repyability import NonRepairableRBD
    >>> rbd = NonRepairableRBD(
    ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")],
    ...     {
    ...         "p1": FixedEventProbability.from_params(0.1),
    ...         "p2": FixedEventProbability.from_params(0.1),
    ...         "v": FixedEventProbability.from_params(0.05),
    ...     },
    ... )
    >>> rbd.input_node, rbd.output_node
    ('s', 't')
    >>> rbd.node_names()
    ['p1', 'p2', 'v']
    >>> sorted(sorted(c) for c in rbd.get_min_cut_sets())
    [['p1', 'p2'], ['v']]

    The structure alone, with no node models, is enough for structural
    analysis:

    >>> from repyability import RBD
    >>> structure = RBD(
    ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")]
    ... )
    >>> si = structure.structural_importance()
    >>> {k: round(v, 4) for k, v in sorted(si.items())}
    {'p1': 0.25, 'p2': 0.25, 'v': 0.75}
    """

    # Constructor inputs, captured verbatim by each subclass's ``__init__`` so
    # the RBD can be re-created (see ``serialisation``); declared here so the
    # attribute is visible on the base type.
    _init_args: dict
    # The names a subclass has models for, set before the base constructor
    # runs (None for a structure alone): one in no edge is reported as
    # such, and a node in the edges with none as having no model.
    _models_given: Optional[list] = None
    # Whether the junctions (see ``_junctions``) are folded out of the
    # structure (``modular.fold``): a ``RepairableRBD``'s are no components,
    # while a ``NonRepairableRBD``'s have a model that never fails.
    _FOLDS_JUNCTIONS = False
    # What simulates a diagram too meshed to work out exactly.
    _SIMULATE_INSTEAD = "Simulate it instead."

    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        nodes: Optional[Iterable] = None,
        k: Optional[dict[Any, int]] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
        capacity: Optional[dict] = None,
    ):
        # The constructor is documented in the class docstring (mkdocstrings
        # merges the two). The positional order mirrors the subclasses: the
        # structure -- ``edges`` then ``nodes`` -- comes first, and ``k``
        # (k-out-of-n) is an optional modifier after it.
        _check_on_infeasible_rbd(on_infeasible_rbd)

        # Create RBD graph
        self.G = RBDGraph()
        self.G.add_edges_from(edges)
        if nodes is not None:
            self.G.add_nodes_from(nodes)

        if input_node is not None:
            if input_node not in self.G.nodes:
                raise ValueError("'input_node' not in RBD structure.")
            # Naming any other node would silently analyse a different
            # system (the nodes before it would drop out).
            if self.G.in_degree(input_node) > 0:
                raise ValueError(
                    f"'input_node' {input_node!r} has incoming edges; the "
                    "input node must be the diagram's source (a node with "
                    "no incoming edges)."
                )
            self.input_node = input_node

        if output_node is not None:
            if output_node not in self.G.nodes:
                raise ValueError("'output_node' not in RBD structure.")
            if self.G.out_degree(output_node) > 0:
                raise ValueError(
                    f"'output_node' {output_node!r} has outgoing edges; the "
                    "output node must be the diagram's sink (a node with "
                    "no outgoing edges)."
                )
            self.output_node = output_node

        # Set whether k for KooN nodes
        has_excess_koon_nodes = False
        excess_koon_nodes = []
        valid_rbd = True
        if k is not None:
            for node, k_val in k.items():
                # (Below 1 is refused with the structure, which says why.)
                if not is_whole(k_val):
                    raise ValueError(
                        f"k for node {node!r} must be a whole number (how "
                        f"many of its inputs must work), got {k_val!r}."
                    )
                k_val = int(k_val)
                if node in self.G.nodes:
                    self.G.nodes[node]["k"] = k_val
                else:
                    valid_rbd = False
                    has_excess_koon_nodes = True
                    excess_koon_nodes.append(node)

        # Finally, check valid RBD structure
        structure_check = self.G.is_valid_RBD_structure(
            nodes=nodes, input_node=input_node, output_node=output_node
        )

        structure_check["excess_koon_nodes"] = excess_koon_nodes
        structure_check["has_excess_koon_nodes"] = has_excess_koon_nodes
        if structure_check["is_valid"]:
            structure_check["is_valid"] = valid_rbd

        for node in excess_koon_nodes:
            structure_check["koon_errors"].append(
                f"k is given for {node!r}, which is not a node of the diagram"
            )

        models = self._models_given
        if models is not None:
            # A model for a name in no edge is reported as unused (not added
            # as an isolated node, which would hide the input and output); a
            # node in the edges with no model as such, but for the nodes
            # with no incoming or no outgoing edges, which need none as the
            # input and output (more than one is reported as that).
            ends = {
                n
                for n in self.G.nodes
                if self.G.in_degree(n) == 0 or self.G.out_degree(n) == 0
            }
            given = set(models)
            unused = [n for n in models if n not in self.G]
            missing = [
                n for n in self.G.nodes if n not in given and n not in ends
            ]
            structure_check["nodes_in_no_edge"] = unused
            structure_check["nodes_with_no_model"] = missing
            if unused or missing:
                structure_check["is_valid"] = False

        if not structure_check["is_valid"]:
            if on_infeasible_rbd == "warn":
                warnings.warn(
                    structure_message(structure_check)
                    + "\n(built anyway, as on_infeasible_rbd='warn' asks)",
                    stacklevel=2,
                )
            elif on_infeasible_rbd == "raise":
                raise ValueError(structure_message(structure_check))

        self.structure_check = structure_check
        self.input_node = structure_check["input_node"]
        self.output_node = structure_check["output_node"]
        self.in_or_out = [self.input_node, self.output_node]
        # A repeated node (a subclass's ``_aliases``) is the component it
        # repeats, so it is not a component of its own; nor is a junction
        # folded out of the structure.
        aliases = self._component_aliases()
        folded = self._junctions() if self._FOLDS_JUNCTIONS else frozenset()
        self.nodes = [
            n
            for n in self.G.nodes
            if n not in self.in_or_out and n not in aliases and n not in folded
        ]
        self.capacity = self._validated_capacity(capacity)
        self.structure_check["has_irrelevant_nodes"] = False
        self.structure_check["irrelevant_nodes"] = set()
        self.structure_check["is_too_meshed"] = False

        if (
            not structure_check["has_cycles"]
            and not structure_check["has_nodes_with_no_successor"]
            and not structure_check["has_koon_errors"]
        ):
            # Reduces the diagram, and raises if nothing reaches the output.
            # A core too meshed to work out leaves which nodes are
            # irrelevant unknown.
            if self._too_meshed() is not None:
                self.structure_check["is_too_meshed"] = True
            else:
                irrelevant_nodes = self.find_irrelevant_components()
                if len(irrelevant_nodes) != 0:
                    self.structure_check["has_irrelevant_nodes"] = True
                self.structure_check["irrelevant_nodes"] = irrelevant_nodes

    def __repr__(self) -> str:
        """A short summary: the nodes, the input and output nodes, any
        k-out-of-n nodes, and what else shapes the diagram."""
        nodes = list(getattr(self, "nodes", []))
        shown = ", ".join(repr(n) for n in nodes[:8])
        if len(nodes) > 8:
            shown += f", ... ({len(nodes) - 8} more)"
        parts = [
            f"{len(nodes)} node{'' if len(nodes) == 1 else 's'}: {shown}",
            f"input {getattr(self, 'input_node', None)!r}, "
            f"output {getattr(self, 'output_node', None)!r}",
        ]
        graph = getattr(self, "G", None)
        if graph is not None:
            koon = {
                n: graph.nodes[n]["k"]
                for n in graph.nodes
                if graph.nodes[n]["k"] != 1
            }
            if koon:
                parts.append(
                    "k-out-of-n "
                    + ", ".join(f"{n!r}: {k}" for n, k in koon.items())
                )
        parts.extend(self._repr_details())
        if not getattr(self, "structure_check", {}).get("is_valid", True):
            parts.append("infeasible (see structure_check)")
        return f"{type(self).__name__}({'; '.join(parts)})"

    def _repr_details(self) -> List[str]:
        """What a subclass adds to ``repr``."""
        return []

    def find_irrelevant_components(self) -> set:
        """Return the nodes that cannot affect whether the system works.

        A node is irrelevant when it is in no minimal path set, e.g. a node
        in parallel with a direct edge, which is a connection that never
        fails. Whether such a node works never changes whether the system
        works, so its Birnbaum and structural importance are zero. They are
        found by the reduction to modules, without listing the path sets.
        An irrelevant node is still a node of the RBD (it is in
        [`node_names`][repyability.RBD.node_names]). The same set is found on
        construction and stored in ``structure_check["irrelevant_nodes"]``,
        with ``structure_check["has_irrelevant_nodes"]``.

        Returns
        -------
        set
            The irrelevant node names; empty if every node is relevant.

        Raises
        ------
        ValueError
            If no set of working nodes can reach the output node.

        Examples
        --------
        Node ``"b"`` is bypassed by the direct edge from ``"a"`` to ``"t"``:

        >>> from repyability import RBD
        >>> rbd = RBD([("s", "a"), ("a", "t"), ("a", "b"), ("b", "t")])
        >>> rbd.find_irrelevant_components()
        {'b'}
        """
        decomposition = self._decomposition()
        if isinstance(decomposition, GraphStructure):
            raise NotImplementedError(decomposition.reason)
        return set(self.nodes) - decomposition.nodes

    def get_all_path_sets(self) -> Iterator[list[Hashable]]:
        """Iterate over every path from the input node to the output node.

        A thin wrapper of ``networkx.all_simple_paths``: each path is a list
        of node names in order, from the input node to the output node.
        These are paths of the graph, not reliability path sets: they ignore
        k-out-of-n values (a path through a node with ``k > 1`` does not by
        itself make the system work) and need not be minimal. For those, use
        [`get_min_path_sets`][repyability.RBD.get_min_path_sets]. The number
        of paths can grow exponentially with the size of the diagram, so
        avoid exhausting the iterator on very large RBDs.

        Returns
        -------
        Iterator[list[Hashable]]
            A generator of paths, each a list of node names.

        Examples
        --------
        >>> from repyability import RBD
        >>> rbd = RBD([("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")])
        >>> sorted(rbd.get_all_path_sets())
        [['s', 'a', 't'], ['s', 'b', 't']]
        """
        return nx.all_simple_paths(
            self.G, source=self.input_node, target=self.output_node
        )

    def get_min_path_sets(
        self, include_in_out_nodes=True
    ) -> set[frozenset[Hashable]]:
        """Return the minimal path sets of the RBD.

        A path set is a set of nodes whose working is enough for the system
        to work; it is minimal when no node can be left out. k-out-of-n
        values are accounted for: a minimal path set through a node with
        k-out-of-n value ``k`` combines path sets reaching ``k`` of its
        predecessors. The sets are expanded from the diagram's modules (see
        [`system_probability`][repyability.RBD.system_probability]) on first
        use and cached; each call returns a new set. Their number can grow
        exponentially with redundancy (``n`` stages of duplicated units in
        series have ``2 ** n``), but nothing else needs them: the system
        probability, the importance measures and the cut sets are found
        without listing them. A junction (a node given
        ``PerfectReliability``, such as a k-out-of-n vote point) always
        works, so no path set needs it, and none lists it (#198).

        Parameters
        ----------
        include_in_out_nodes : bool, optional
            Whether each path set includes the input and output nodes, by
            default True. Pass False for the components only. (The default
            differs from
            [`get_min_cut_sets`][repyability.RBD.get_min_cut_sets].)

        Returns
        -------
        set[frozenset[Hashable]]
            The minimal path sets, each a frozenset of node names.

        Raises
        ------
        ValueError
            If there is no path set, i.e. no set of working nodes can reach
            the output node (e.g. a ``k`` greater than the node's number of
            incoming edges).

        Examples
        --------
        A 2-out-of-3 vote ``"v"`` needs two of ``"a"``, ``"b"`` and ``"c"``:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [
        ...         ("s", "a"), ("s", "b"), ("s", "c"),
        ...         ("a", "v"), ("b", "v"), ("c", "v"), ("v", "t"),
        ...     ],
        ...     k={"v": 2},
        ... )
        >>> paths = rbd.get_min_path_sets(include_in_out_nodes=False)
        >>> sorted(sorted(p) for p in paths)
        [['a', 'b', 'v'], ['a', 'c', 'v'], ['b', 'c', 'v']]
        """
        if not hasattr(self, "_min_path_sets"):
            self._min_path_sets: set[frozenset[Hashable]] = (
                self._set_structure().path_sets()
            )
        if not include_in_out_nodes:
            return set(self._min_path_sets)
        ends = {node for node in self.in_or_out if node in self.G}
        return {path_set | ends for path_set in self._min_path_sets}

    def is_system_working(
        self, component_status: dict[Any, bool], method: str
    ) -> bool:
        """Return whether the system works, given which components work.

        This is the structure function. It is evaluated through the
        diagram's modules (see
        [`system_probability`][repyability.RBD.system_probability]): a
        series module works when all of its members do, a parallel one when
        any does, a k-out-of-n one when at least ``k`` do, and what is left
        (e.g. a bridge) through its minimal path sets (``method="p"``: some
        path set has every member working) or cut sets (``method="c"``:
        every cut set has a member working). Both give the same answer;
        ``"p"`` is typically faster as it does not need the cut sets. The
        input and output nodes need no entry. The layout is worked out once,
        so repeated calls (as in the simulations) are cheap.

        Parameters
        ----------
        component_status : dict[Any, bool]
            Whether each component is working (truthy) or failed (falsy),
            keyed by node name. Every node in a minimal path set needs an
            entry; other keys are ignored.
        method : str
            ``"p"`` (path sets) or ``"c"`` (cut sets). There is no default.

        Returns
        -------
        bool
            True if the system is working, otherwise False.

        Raises
        ------
        ValueError
            If ``method`` is not ``"p"`` or ``"c"``.
        KeyError
            If a node needed for the evaluation has no entry in
            ``component_status``.

        Examples
        --------
        Two parallel nodes ``"a"`` and ``"b"`` in series with ``"c"``:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> rbd.is_system_working({"a": False, "b": True, "c": True}, "p")
        True
        >>> rbd.is_system_working({"a": True, "b": True, "c": False}, "c")
        False
        """
        method = structure_method(method)
        return self._decomposition().works(component_status, method)

    def system_timeline(self, timelines: Mapping) -> Any:
        """The system's up/down history from its components': the
        structure function followed over time.

        The components' timelines are merged up the diagram's modules (see
        [`system_probability`][repyability.RBD.system_probability]): a
        series module is up while all its members are, a parallel one while
        any is, a k-out-of-n one while at least ``k`` are, and what is left
        (e.g. a bridge) while one of its minimal path sets is all up. Each
        change of the system's timeline keeps its cause, the component
        whose change made it, and whether that change was planned; changes
        at the same time are taken in the order of the diagram's
        components, each one's in its own order (see
        ``repyability.timelines``).

        It takes any timelines: an outage log, a what-if edit of one, or
        simulated histories (see
        [`RepairableRBD.simulate_timelines`][repyability.RepairableRBD.simulate_timelines]).

        Parameters
        ----------
        timelines : Mapping
            ``{node: timeline}``: a [`Timeline`][repyability.Timeline] or
            [`Timelines`][repyability.Timelines] for each component the
            system depends on (all over one window; Timelines with as many
            histories each, and a Timeline among them standing for each
            history). A drawing junction (a perfectly reliable node) left
            out is up throughout; a component no path set needs is
            ignored.

        Returns
        -------
        Timeline or Timelines
            The system's: a Timelines if any component's is one, else a
            Timeline. Its causes are node names.

        Raises
        ------
        ValueError
            If a component the system depends on has no timeline, a key is
            not a component (the input or output node, a repeated node's
            copy, or an unknown name), or the timelines do not share a
            window or a number of histories.
        TypeError
            If ``timelines`` is not a mapping of Timeline or Timelines
            objects.

        Examples
        --------
        Two pumps in parallel feeding a valve, from their outage logs:

        >>> from repyability import RBD, Timeline
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "v"), ("b", "v"), ("v", "t")]
        ... )
        >>> logs = {
        ...     "a": Timeline.from_outages([(10, 30)], end=100),
        ...     "b": Timeline.from_outages([(20, 25), (60, 70)], end=100),
        ...     "v": Timeline.from_outages([(80, 81)], end=100),
        ... }
        >>> plant = rbd.system_timeline(logs)
        >>> plant.down_intervals.tolist()
        [[20.0, 25.0], [80.0, 81.0]]
        >>> plant.failures_by_cause()
        {'a': 0, 'b': 1, 'v': 1}
        """
        from repyability.timelines import _system_timeline

        return _system_timeline(self, timelines)

    def minimal_path_sets(self) -> List[frozenset]:
        """The minimal path sets of the components (no input or output
        node), as a list, smallest first (then by name): the same as
        [`get_min_path_sets`][repyability.RBD.get_min_path_sets]
        ``(include_in_out_nodes=False)``, in the form a
        [`FaultTree`][repyability.FaultTree]'s ``minimal_path_sets``
        gives, so the same code runs on either.

        Returns
        -------
        list of frozenset
            The minimal path sets.

        Examples
        --------
        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> [sorted(p) for p in rbd.minimal_path_sets()]
        [['a', 'c'], ['b', 'c']]
        """
        return sorted(
            self.get_min_path_sets(include_in_out_nodes=False),
            key=_set_order,
        )

    def minimal_cut_sets(self) -> List[frozenset]:
        """The minimal cut sets of the components, as a list, smallest first
        (then by name): the same as
        [`get_min_cut_sets`][repyability.RBD.get_min_cut_sets]``()``, in
        the form a [`FaultTree`][repyability.FaultTree]'s
        ``minimal_cut_sets`` gives, so the same code runs on either.

        Returns
        -------
        list of frozenset
            The minimal cut sets.

        Examples
        --------
        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> [sorted(c) for c in rbd.minimal_cut_sets()]
        [['c'], ['a', 'b']]
        """
        return sorted(self.get_min_cut_sets(), key=_set_order)

    def get_min_cut_sets(
        self, include_in_out_nodes=False
    ) -> set[frozenset[Hashable]]:
        """Return the minimal cut sets of the RBD.

        A cut set is a set of nodes whose failure is enough for the system
        to fail; it is minimal when no node can be left out. The minimal cut
        sets are the minimal transversals (hitting sets) of the minimal path
        sets, and k-out-of-n values are accounted for. They are built from
        the diagram's modules without listing the path sets: a series
        module's cut sets are its members', a parallel module's combine one
        of each member's, and a k-out-of-n module's combine one of each of
        ``n - k + 1`` members'; the part that is not series-parallel (e.g. a
        bridge) has its own read off the exact engine's Shannon
        decomposition (see ``minimal_cut_sets_from_path_sets`` in this
        module). They are worked out on first use and cached; each call
        returns a new set. A junction (a node given ``PerfectReliability``,
        such as a k-out-of-n vote point) never fails, so a cut set with one
        never happens, and none is listed (#198).

        Parameters
        ----------
        include_in_out_nodes : bool, optional
            Whether to derive the cut sets from the path sets including the
            input and output nodes, by default False (components only). If
            True, the input and output nodes each appear as an extra
            single-node cut set. (The default differs from
            [`get_min_path_sets`][repyability.RBD.get_min_path_sets].)

        Returns
        -------
        set[frozenset[Hashable]]
            The minimal cut sets, each a frozenset of node names.

        Raises
        ------
        ValueError
            If no set of working nodes can reach the output node.

        Examples
        --------
        Two parallel nodes ``"a"`` and ``"b"`` in series with ``"c"``:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> sorted(sorted(c) for c in rbd.get_min_cut_sets())
        [['a', 'b'], ['c']]
        >>> cuts = rbd.get_min_cut_sets(include_in_out_nodes=True)
        >>> sorted(sorted(c) for c in cuts)
        [['a', 'b'], ['c'], ['s'], ['t']]
        """
        # The structure is fixed once built, so the cut sets are worked out
        # once per RBD; each call gets its own copy of the set.
        if not hasattr(self, "_min_cut_sets"):
            self._min_cut_sets: set[frozenset[Hashable]] = (
                self._set_structure().cut_sets()
            )
        if not include_in_out_nodes:
            return set(self._min_cut_sets)
        # Failing the input or the output alone fails the system.
        ends = {frozenset([node]) for node in self.in_or_out if node in self.G}
        return self._min_cut_sets | ends

    def path_set_probabilities(self, node_probabilities):
        """Return the probability that each minimal path set fully works.

        For each minimal path set (components only) this is the product of
        its nodes' probabilities, i.e. the probability that all of them
        work, assuming independent nodes. The path sets share nodes, so these
        values do not add up to the system probability: use
        [`system_probability`][repyability.RBD.system_probability] for that.

        Parameters
        ----------
        node_probabilities : dict
            The probability that each node works, keyed by node name, as
            floats or numpy arrays of equal length (not lists). Every node
            in a minimal path set needs an entry; other keys are ignored.

        Returns
        -------
        np.ndarray
            One value per minimal path set, in no particular order and with
            no labels (to pair values with path sets, take the products over
            ``get_min_path_sets(include_in_out_nodes=False)`` directly).
            The shape is ``(n_path_sets,)`` for float probabilities and
            ``(n_path_sets, n)`` for arrays of length ``n``.

        Raises
        ------
        KeyError
            If a node in a minimal path set has no entry.

        Examples
        --------
        Two parallel nodes ``"a"`` and ``"b"`` in series with ``"c"`` have
        the minimal path sets ``{a, c}`` and ``{b, c}``:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> probs = rbd.path_set_probabilities({"a": 0.9, "b": 0.8, "c": 0.95})
        >>> sorted(round(float(p), 4) for p in probs)
        [0.76, 0.855]
        """
        path_sets = self.get_min_path_sets(include_in_out_nodes=False)
        out = []
        for path in path_sets:
            path_prob = 1
            for node in path:
                path_prob *= node_probabilities[node]
            out.append(path_prob)
        return np.array(out)

    def system_probability(
        self,
        node_probabilities: Dict,
        method: str = "p",
    ) -> np.ndarray:
        """Return the exact system probability from each node's probability.

        Given the probability that each node works -- e.g. its reliability
        at some time or its availability -- this returns the probability
        that the system works, assuming the nodes are independent. It is the
        engine behind the subclasses' reliability, availability and
        importance calculations.

        The result is exact. On construction the diagram is reduced to
        modules, each with a closed form: a series chain works with
        probability ``p1 * p2 * ...``, a parallel group with
        ``1 - (1 - p1) * (1 - p2) * ...``, and a k-out-of-n group by summing
        over how many of its members work. Whatever is not series-parallel
        (e.g. a bridge) is left as a core over modules and single nodes, and
        is expanded by a Shannon (pivotal) decomposition over its minimal
        path sets, with repeated sub-problems solved once. So a
        series-parallel diagram is evaluated without listing its path sets,
        however many there are (``n`` stages of duplicated units in series
        have ``2 ** n``), and only the core pays the combinatorial price.
        Both depend only on the structure, so they are worked out once and
        reused.

        With ``method="p"`` the probability that the system works is
        computed, and with ``method="c"`` the probability that it fails,
        whose complement is returned. Both give the same result (up to
        rounding); ``"p"`` is the default. Every step is a sum of products
        of node probabilities and their complements, so the probability
        computed keeps its full relative precision however close to 0 it
        is.

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works, keyed by node name: floats,
            or 1-d arrays (e.g. one value per time) that all have the same
            length. Every intermediate node needs an entry; entries for the
            input and output nodes, and any other keys, are not used. The
            dict is not modified.
        method : str, optional
            ``"p"`` or ``"paths"`` (the default), or ``"c"`` or
            ``"cuts"``: whether to compute the probability that the system
            works or that it fails.

        Returns
        -------
        np.ndarray
            The system probability, a 1-d array with one value per element
            of the node arrays. It is always an array: float inputs give a
            one-element array.

        Raises
        ------
        ValueError
            If ``method`` is not ``"p"`` or ``"c"``, or the node probability
            arrays are not all the same length.
        KeyError
            If an intermediate node has no entry in ``node_probabilities``.

        Examples
        --------
        Two parallel nodes ``"a"`` and ``"b"`` in series with ``"c"`` work
        with probability ``(1 - 0.1 * 0.2) * 0.95``:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> p = rbd.system_probability({"a": 0.9, "b": 0.8, "c": 0.95})
        >>> round(float(p[0]), 4)
        0.931

        Arrays give one system probability per element:

        >>> import numpy as np
        >>> p = rbd.system_probability(
        ...     {
        ...         "a": np.array([0.9, 0.5]),
        ...         "b": np.array([0.8, 0.5]),
        ...         "c": np.array([0.95, 0.5]),
        ...     }
        ... )
        >>> [round(float(v), 4) for v in p]
        [0.931, 0.375]
        """
        method = structure_method(method)

        arrays, size = self._node_arrays(node_probabilities)
        works, fails = self._decomposition().probabilities(
            arrays, shape=size, works=method == "p", fails=method == "c"
        )
        if method == "p":
            return np.array(works, dtype=float)
        return 1 - np.asarray(fails, dtype=float)

    def _system_unreliability(
        self, node_probabilities: Dict, node_failures: Optional[Dict] = None
    ) -> np.ndarray:
        """The probability that the system fails, worked out as a sum of
        products in its own right rather than as one less the probability
        that it works, so that a small one keeps its precision (``ff``,
        the long-run unavailability, and the risk worths, which divide by
        it). ``node_failures`` gives each node's probability of failing,
        when it is known more precisely than one less its probability of
        working (from a reliable node's ``ff``, say): without it, the
        nodes' are ``1 - p``."""
        return self._system_probabilities(
            node_probabilities, node_failures, works=False
        )[1]

    def _system_probabilities(
        self,
        node_probabilities: Dict,
        node_failures: Optional[Dict] = None,
        works: bool = True,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """The probabilities that the system works (unless ``works`` is
        false: then None) and that it fails, in one pass, each a sum of
        products that keeps a small one's precision (see
        ``_system_unreliability``)."""
        arrays, failures, size = self._node_pairs(
            node_probabilities, node_failures
        )
        up, down = self._decomposition().probabilities(
            arrays, failures, shape=size, works=works, fails=True
        )
        return (
            np.array(up, dtype=float) if works else None,
            np.array(down, dtype=float),
        )

    def _failures_with_overrides(
        self, node_failures: dict, working_nodes, broken_nodes
    ) -> dict:
        """A copy of ``node_failures`` (each node's probability of failing)
        with the nodes held working at 0 and those held broken at 1, as
        ``_probabilities_with_overrides`` holds their probabilities of
        working at 1 and 0."""
        out = dict(node_failures)
        for node in working_nodes or ():
            out[node] = np.zeros_like(
                np.atleast_1d(np.asarray(out[node], dtype=float))
            )
        for node in broken_nodes or ():
            out[node] = np.ones_like(
                np.atleast_1d(np.asarray(out[node], dtype=float))
            )
        return out

    def _node_pairs(
        self, node_probabilities: Dict, node_failures: Optional[Dict] = None
    ) -> tuple[dict, dict, int]:
        """Each intermediate node's probabilities of working and of failing,
        as 1-d arrays of one length, and that length (see ``_node_arrays``):
        the second from ``node_failures`` when it is given (as for
        ``_system_unreliability``), else one less the first."""
        p, size = self._node_arrays(node_probabilities)
        if node_failures is None:
            return p, {node: 1.0 - value for node, value in p.items()}, size
        q, length = self._node_arrays(node_failures)
        if length != size:
            raise ValueError("Probability arrays must be same length")
        return p, q, size

    def _importances(
        self, node_probabilities: Dict, node_failures: Optional[Dict] = None
    ) -> tuple[dict, np.ndarray, np.ndarray, dict, dict]:
        """Every node's Birnbaum importance (the derivative of the system's
        probability of working with respect to the node's), the
        probabilities that the system works and that it fails, and the
        nodes' probabilities of working and of failing (see
        ``_node_pairs``), as 1-d arrays of one length.

        All come from one pass of the decomposition (see
        ``Decomposition.value_and_gradient``), and each is a sum of
        products, so a small one keeps its full relative precision: the
        importance ``R(i works) - R(i failed)`` would be a difference of
        two values near 1 in a reliable system."""
        p, q, size = self._node_pairs(node_probabilities, node_failures)
        works, fails, gradient = self._decomposition().value_and_gradient(
            p, q, shape=size
        )

        def full(value: Any) -> np.ndarray:
            return np.array(
                np.broadcast_to(np.asarray(value, dtype=float), (size,))
            )

        importance = {
            node: full(gradient.get(node, 0.0)) for node in self.nodes
        }
        return importance, full(works), full(fails), p, q

    def _node_arrays(self, node_probabilities: Dict) -> tuple[dict, int]:
        """Each intermediate node's probability as a 1-d array, and their
        common length."""
        arrays: dict = {}
        lengths = set()
        for node in self.nodes:
            arrays[node] = np.atleast_1d(node_probabilities[node])
            lengths.add(len(arrays[node]))
        if len(lengths) > 1:
            raise ValueError("Probability arrays must be same length")
        return arrays, lengths.pop() if lengths else 1

    def _decomposition(self) -> Decomposition:
        """The diagram reduced to modules (see ``modular.py``): the exact
        engine behind the probabilities, the structure function and the
        path and cut sets. It depends only on the structure, so it is built
        once, on construction for a feasible structure. A structure that is
        not a valid RBD is not reduced: its core is the whole diagram, with
        the path sets the memoised search finds."""
        if not hasattr(self, "_modules"):
            if self.structure_check.get("has_cycles"):
                # Built anyway (on_infeasible_rbd), but not to be evaluated:
                # a path around a cycle never ends.
                cycles = self.structure_check.get("cycles", ())
                raise ValueError(
                    "The diagram has a cycle (through "
                    + "; ".join(_names(cycle) for cycle in cycles)
                    + "), so it cannot be evaluated: every path must lead "
                    "from the input node to the output node."
                )
            self._modules = self._decompose_graph(self.G)
        return self._modules

    def _set_structure(self) -> Decomposition:
        """The structure the path and cut sets are read from: the
        decomposition, with the junctions (see ``_junctions``) folded in as
        always working (#198). A junction never fails, so a cut set with
        one never happens, and a path set needs nothing of it; a diagram
        that folds them out of its structure (a ``RepairableRBD``) has
        none left, and one too meshed to work out lists no sets."""
        decomposition = self._decomposition()
        junctions = self._junctions()
        if (
            self._FOLDS_JUNCTIONS
            or not junctions
            or isinstance(decomposition, GraphStructure)
        ):
            return decomposition
        return fold(decomposition, junctions)

    def _decompose_graph(self, graph) -> Decomposition:
        """``graph`` reduced to modules as the diagram's own is (see
        ``_decomposition``): the diagram's graph, or one drawn from it with
        the same input and output nodes, junctions and repeated nodes (an
        allocation's copies of a train)."""
        reducible = self.structure_check["is_valid"] and all(
            graph.nodes[node]["k"] >= 1 for node in graph.nodes
        )
        folded = self._junctions() if self._FOLDS_JUNCTIONS else frozenset()
        try:
            modules = decompose(
                graph,
                self.input_node,
                self.output_node,
                reduce=reducible,
                aliases=self._component_aliases(),
            )
        except bdd.TooLarge as error:
            # Too meshed to work out exactly (#172): the simulations follow
            # the graph itself, and the rest refuses.
            return GraphStructure(
                graph,
                self.input_node,
                self.output_node,
                self._component_aliases(),
                folded,
                f"{error} {self._SIMULATE_INSTEAD}",
            )
        return fold(modules, folded) if folded else modules

    def _meshed_routes(self, out: dict, free: Iterable[str]) -> dict:
        """A route report (see ``analysis_routes``), sorted, with every
        exact or numerical analysis but those in ``free``, which need no
        structure, refused when the core is too meshed to work out (see
        ``_too_meshed``): the simulations alone run."""
        from repyability.rbd import routes as r

        meshed = self._too_meshed()
        if meshed is not None:
            free = set(free)
            for name, route in out.items():
                if route.route in (r.EXACT, r.NUMERICAL) and name not in free:
                    out[name] = r.refused(meshed)
        return dict(sorted(out.items()))

    def _require_no_ccf_pairs(self) -> None:
        """Raise if the diagram has common-cause groups: the joint
        importance holds pairs of nodes working and failed, which a group's
        members cannot be (the causes they share would still strike the
        others)."""
        if getattr(self, "ccf_groups", None):
            raise NotImplementedError(
                "The joint importance holds pairs of nodes working and "
                "failed, which a common-cause group's members cannot be "
                "(the causes they share would still strike the others): it "
                "is not worked out with common-cause groups, as yet."
            )

    def _joint_pairs(self, birnbaum: Callable, working_nodes, broken_nodes):
        """The joint importance of each pair of nodes (see
        ``NonRepairableRBD.joint_importance``) from ``birnbaum(working,
        broken)``, every node's Birnbaum importance with those nodes held:
        node ``j``'s with node ``i`` held working, less with it held
        failed, for each pair once (see ``Pairs``); 0 for a pair with a
        node already held."""
        self._require_no_ccf_pairs()
        working = set(working_nodes or ())
        broken = set(broken_nodes or ())
        base = birnbaum(working, broken)
        nodes = list(base)
        held = working | broken
        up, down = {}, {}
        for node in nodes:
            if node not in held:
                up[node] = birnbaum(working | {node}, broken)
                down[node] = birnbaum(working, broken | {node})
        out = Pairs()
        for k, i in enumerate(nodes):
            for j in nodes[k + 1 :]:  # noqa: E203
                if i in held or j in held:
                    out[Pairs.oriented(i, j)] = 0.0 * base[j]
                else:
                    out[Pairs.oriented(i, j)] = up[i][j] - down[i][j]
        return out

    def _too_meshed(self) -> Optional[str]:
        """Why the diagram cannot be worked out exactly, if its core is too
        meshed (see ``modular.GraphStructure``); None if it can."""
        decomposition = self._decomposition()
        if isinstance(decomposition, GraphStructure):
            return decomposition.reason
        return None

    def _junctions(self) -> frozenset:
        """The nodes that are only drawing devices: perfectly reliable
        junctions (the nodes given ``PerfectReliability``, such as a
        k-out-of-n vote), which never fail and cannot be improved. The
        importance measures and allocations leave them out (and hold them
        at 1); none in a structure alone."""
        return frozenset()

    def _component_aliases(self) -> dict:
        """``{node: component}`` for the nodes that stand for a component
        drawn in more than one place (a ``NonRepairableRBD``'s repeated
        nodes); none in a plain RBD."""
        return getattr(self, "_aliases", {})

    def _validated_capacity(self, capacity: Optional[dict]) -> dict:
        """The node capacities, checked: each a positive number (``inf`` for
        no limit), as a float, or a dict of such levels and their
        probabilities, positive and adding up to 1 (as floats, in
        increasing order of level, and scaled to add up to 1 exactly); and
        each for a node of the diagram that is not the input or output
        node, nor a repeat of another node."""
        if not capacity:
            return {}
        aliases = self._component_aliases()
        out: Dict[Any, Any] = {}
        for node, value in dict(capacity).items():
            if node in self.in_or_out:
                raise ValueError(
                    f"The input or output node {node!r} takes no capacity: "
                    "it passes whatever reaches it."
                )
            if node in aliases:
                raise ValueError(
                    f"Node {node!r} repeats node {aliases[node]!r}: give "
                    f"the capacity of {aliases[node]!r}, which it has "
                    "wherever it is drawn."
                )
            if node not in self.G.nodes:
                raise ValueError(
                    f"Unknown node {node!r} in capacity: it is not in the "
                    "diagram."
                )
            if isinstance(value, dict):
                out[node] = _capacity_levels(node, value)
            else:
                out[node] = _capacity_number(node, value, "capacity")
        return out

    def _capacity_models(self) -> dict:
        """``{node: model}`` for the nodes with no capacity entry whose
        model has a capacity distribution of its own (a nested RBD with
        capacities, or a ``DegradingNode``): none in a plain RBD."""
        return {}

    def _require_capacity(self) -> None:
        """Raise unless some node has a capacity, given or from its
        model."""
        if not self._has_capacity():
            raise ValueError(
                "No node has a capacity: give each node's throughput with "
                "capacity={node: capacity} when building the RBD."
            )

    def _has_capacity(self) -> bool:
        """Whether some node has a capacity, given or from its model."""
        return bool(self.capacity) or bool(self._capacity_models())

    def system_capacity(
        self, node_probabilities: Dict
    ) -> CapacityDistribution:
        """Return the exact distribution of the system's capacity, from each
        node's probability of working.

        Each node carries its ``capacity`` while it works and nothing once
        it has failed (a node given none limits nothing), and the system's
        capacity is the most that can flow from the input to the output:
        the maximum flow of the diagram, which is the least total capacity
        of a cut. A series chain carries the least of its nodes' capacities
        and a parallel group their sum, and a k-out-of-n node passes flow
        only while at least ``k`` of its inputs are reached, so the
        capacity is positive exactly when the system works. With three
        pumps of half the demand each, one failure costs nothing and two
        cost half the output.

        The distribution is exact, over every combination of the nodes'
        states, and assumes they are independent. It is worked out as the
        system probability is: each series, parallel or k-out-of-n module's
        distribution in closed form from its members' (the least of their
        capacities, their sum, their sum while at least ``k`` work), and
        what is left (e.g. a bridge) by conditioning on its parts'
        capacities one at a time. So it costs about what the system
        probability does. The subclasses give it the node probabilities:
        [`NonRepairableRBD.capacity_distribution`][repyability.NonRepairableRBD.capacity_distribution]
        at a time, and
        [`RepairableRBD.capacity_distribution`][repyability.RepairableRBD.capacity_distribution]
        in the long run.

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works, keyed by node name: floats,
            or 1-d arrays (e.g. one value per time) that all have the same
            length. Every intermediate node needs an entry; other keys are
            not used.

        Returns
        -------
        CapacityDistribution
            The capacities the system can have and their probabilities: one
            per level for float probabilities, and one row per level and
            one column per element for arrays.

        Raises
        ------
        ValueError
            If no node has a capacity, the diagram is not a valid RBD, the
            probability arrays are not all the same length, or a node takes
            its capacity from its model (a ``DegradingNode``, or a nested
            RBD with capacities), which a probability of working does not
            describe.
        KeyError
            If an intermediate node has no entry in ``node_probabilities``.

        Examples
        --------
        Two pumps of 60 and 40 in parallel, feeding a pipe that carries 80:

        >>> from repyability import RBD
        >>> plant = RBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "pipe"), ("p2", "pipe"),
        ...      ("pipe", "t")],
        ...     capacity={"p1": 60, "p2": 40, "pipe": 80},
        ... )
        >>> up = {"p1": 0.9, "p2": 0.8, "pipe": 1.0}
        >>> capacity = plant.system_capacity(up)
        >>> capacity.levels.tolist()
        [0.0, 40.0, 60.0, 80.0]
        >>> capacity.probabilities.round(4).tolist()
        [0.02, 0.08, 0.18, 0.72]

        The pipe, not the pumps, limits the plant when both run.
        """
        own = self._capacity_models()
        if own:
            raise ValueError(
                f"Node(s) {sorted(own, key=str)} take their capacity from "
                "their models, which a probability of working does not "
                "describe: use capacity_distribution()."
            )
        scalar = all(np.ndim(node_probabilities[n]) == 0 for n in self.nodes)
        arrays, size = self._node_arrays(node_probabilities)
        levels, probabilities = self._capacity_arrays(arrays, size)
        return CapacityDistribution(
            levels, probabilities[:, 0] if scalar else probabilities
        )

    def _capacity_arrays(
        self, arrays: Dict, size: int, own: Optional[Dict] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The exact distribution of the system's capacity (see
        ``capacity.py``), given each node's probability of working as an
        array of length ``size``: the levels, and one row of probabilities
        per level. ``own`` holds the distributions (levels, and one row per
        level of ``size`` probabilities) of the nodes whose models give
        them."""
        own = own or {}
        self._require_capacity()
        if not self.structure_check["is_valid"]:
            raise ValueError(
                "The capacity analysis needs a valid diagram: this one "
                "failed the structure check (see structure_check)."
            )
        meshed = self._too_meshed()
        if meshed is not None:
            raise NotImplementedError(meshed)
        flow = self._decomposition().flow
        if flow is None:
            raise ValueError(
                "The capacity analysis needs a valid diagram: this one "
                "failed the structure check (see structure_check)."
            )

        junctions = self._junctions()

        def component(name):
            if name in own:
                return own[name]
            if name in junctions and name not in arrays:
                # A junction always works: it passes what reaches it, up to
                # its capacity if it has one.
                works = np.ones(size)
            else:
                works = np.asarray(arrays[name], dtype=float)
            return _capacity.node_distribution(
                self.capacity.get(name, np.inf), works, 1.0 - works
            )

        return _capacity.system_distribution(flow, component, size)

    def _forced(
        self, distribution: tuple, node, working_nodes, broken_nodes
    ) -> tuple:
        """A node's own capacity distribution with the node forced working
        (given that it works) or broken (0), if it is."""
        if node in broken_nodes:
            size = distribution[1].shape[1]
            return np.zeros(1), np.ones((1, size))
        if node in working_nodes:
            return _capacity.working(distribution)
        return distribution

    @leaves_out_junctions
    @check_probability
    def improvement_allocation(
        self,
        target: float,
        node_probabilities: Dict,
        fixed: Optional[list] = None,
        weights=None,
    ):
        """Improve the node probabilities just enough to meet a target.

        Reliability allocation by a common improvement: every node that is
        not ``fixed`` has its unreliability ``1 - p`` multiplied by
        ``exp(-x * w)``, where ``w`` is the node's weight and ``x`` is one
        scale factor shared by all nodes,

            p_new = 1 - (1 - p) * exp(-x * w),

        and ``x`` is solved for (by a bracketed root search) so that
        [`system_probability`][repyability.RBD.system_probability] of the
        new probabilities equals ``target``. With the default equal weights
        every free node's unreliability shrinks by the same factor, keeping
        their ratios. A larger weight changes a node more; a weight of 0
        leaves it unchanged, as does a probability of exactly 1.

        A ``target`` below the current system probability gives ``x < 0``,
        which increases the unreliabilities, each capped at 1 (a node
        probability of 0): the result is then the lowest node probabilities
        that still meet the target. A target the scaling cannot reach
        (e.g. because ``fixed`` nodes limit the system probability) raises a
        ValueError giving the reachable range. A target equal to the best
        reachable probability gives the free nodes probability 1. The
        solver's result is stored on the RBD as ``res``, a scipy
        ``OptimizeResult`` whose ``x`` holds ``x`` (``inf`` when the free
        nodes are made perfect), replacing any earlier one.

        Parameters
        ----------
        target : float
            The system probability to reach, in [0, 1].
        node_probabilities : Dict
            The current probability that each node works, as a single float
            in [0, 1] per node, keyed by node name. An intermediate node
            with no entry starts at 0.5. Any other keys (e.g. the input and
            output nodes) are scaled like the rest and returned.
        fixed : list, optional
            Nodes whose probability is not changed, by default None (every
            node may change).
        weights : dict, optional
            A weight per node, by default None (1.0 for every node). If
            given, it needs an entry for every node that is not ``fixed``.

        Returns
        -------
        dict
            The allocated probability (a float) of every key of
            ``node_probabilities``, and of any intermediate node that was
            missing from it. ``fixed`` nodes keep their probability.

        Raises
        ------
        ValueError
            If ``target`` is above 1 or below 0, if it cannot be reached, or
            if a node probability is not a single value in [0, 1].
        KeyError
            If ``weights`` is given without an entry for a node that is not
            ``fixed``.

        Examples
        --------
        Raising two parallel nodes ``"a"`` and ``"b"`` in series with
        ``"c"`` from a system probability of 0.931 to 0.99:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> current = {"a": 0.9, "b": 0.8, "c": 0.95}
        >>> new = rbd.improvement_allocation(0.99, current)
        >>> {k: round(v, 4) for k, v in sorted(new.items())}
        {'a': 0.9814, 'b': 0.9627, 'c': 0.9907}
        >>> round(float(rbd.system_probability(new)[0]), 4)
        0.99

        Every unreliability was cut by the same factor:

        >>> sorted({round((1 - new[k]) / (1 - current[k]), 4) for k in new})
        [0.1863]
        """
        fixed_nodes = set() if fixed is None else set(fixed)
        probabilities: Dict[Any, float] = {
            node: _probability_value(f"node_probabilities[{node!r}]", value)
            for node, value in node_probabilities.items()
        }
        for node in self.nodes:
            probabilities.setdefault(node, 0.5)
        # A perfect junction stays perfect.
        for node in self._junctions():
            probabilities[node] = 1.0
            fixed_nodes.add(node)

        # Solve for the common multiplier m = exp(-x) of the free nodes'
        # unreliabilities, q -> min(1, q * m ** w): unlike x it has a finite
        # range, from m = 0 (every free node perfect) to the m at which every
        # free node has failed, over which the system probability falls
        # continuously, so the target can be bracketed exactly.
        free = {
            node: (1.0 - p, 1.0 if weights is None else weights[node])
            for node, p in probabilities.items()
            if node not in fixed_nodes
        }

        def allocated(m: float) -> Dict[Any, float]:
            out = dict(probabilities)
            for node, (q, w) in free.items():
                out[node] = 1.0 - min(1.0, q * m**w)
            return out

        def system(m: float) -> float:
            return self._allocation_probability(allocated(m))

        failing = [q ** (-1.0 / w) for q, w in free.values() if q > 0 < w]
        m_max = max(failing, default=1.0)
        best, worst = system(0.0), system(m_max)
        if not worst - 1e-12 <= target <= best + 1e-12:
            raise ValueError(
                f"target {target} cannot be reached: with the fixed nodes "
                "and weights given, the system probability can only range "
                f"from {worst:.6g} to {best:.6g}."
            )
        if target >= best:
            m = 0.0
        elif target <= worst:
            m = m_max
        else:
            m = brentq(lambda m: system(m) - target, 0.0, m_max, xtol=1e-15)
        self.res = OptimizeResult(
            x=np.array([-np.log(m) if m > 0.0 else np.inf]),
            fun=np.array([system(m) - target]),
            success=True,
            message="The target is met.",
        )
        return allocated(m)

    @leaves_out_junctions
    @check_probability
    def equal_allocation(self, target: float):
        """Give every node the same probability, chosen to meet a target.

        Finds the single probability ``p`` that, given to every intermediate
        node, makes [`system_probability`][repyability.RBD.system_probability]
        equal ``target``: e.g. ``target ** (1 / n)`` for ``n`` nodes in
        series and ``1 - (1 - target) ** (1 / n)`` for ``n`` in parallel.
        Any node models are ignored. It runs
        [`improvement_allocation`][repyability.RBD.improvement_allocation]
        from 0.5 for every node, with equal weights and nothing fixed, so the
        solver's result is stored on the RBD as ``res``. A target of 1 gives
        every node probability 1.

        Parameters
        ----------
        target : float
            The system probability to reach, in [0, 1].

        Returns
        -------
        dict
            The allocated probability (a float, the same for every node),
            keyed by intermediate node name.

        Raises
        ------
        ValueError
            If ``target`` is above 1 or below 0.

        Examples
        --------
        Three nodes in series each need ``0.9 ** (1 / 3)``:

        >>> from repyability import RBD
        >>> rbd = RBD([("s", 1), (1, 2), (2, 3), (3, "t")])
        >>> new = rbd.equal_allocation(0.9)
        >>> {k: round(v, 4) for k, v in sorted(new.items())}
        {1: 0.9655, 2: 0.9655, 3: 0.9655}
        """
        node_probabilities = {}
        for node in self.nodes:
            node_probabilities[node] = np.atleast_1d(0.5)

        return self.improvement_allocation(target, node_probabilities)

    @leaves_out_junctions
    @check_probability
    def simple_allocation(
        self,
        target: float,
        weights=None,
    ):
        """Meet a target with the smallest change in the node log-odds.

        Starting from 0.5 for every node, finds the node probabilities that
        meet the target with the least weighted change on the log-odds
        scale. With ``s_i = log(p_i / (1 - p_i))`` a node's log-odds (0 at
        0.5) and ``w_i`` its weight, it minimises ``sum_i s_i ** 2 / w_i``
        subject to [`system_probability`][repyability.RBD.system_probability]
        equalling ``target``. Any node models are ignored.

        At the solution each node's change is proportional to its weight
        times the sensitivity of the system's log-odds to it,
        ``p_i (1 - p_i) I_B(i) / (R (1 - R))`` with ``I_B(i)`` its Birnbaum
        importance: nodes that matter more to the system (e.g. a node in
        series with a redundant pair), and nodes with larger weights, move
        further from 0.5 (at equal sensitivity, twice the weight moves a node
        twice as far), and a node with weight 0 stays at 0.5. With equal
        weights, nodes placed symmetrically (e.g. all in series, or all in
        parallel) get the same value, as
        [`equal_allocation`][repyability.RBD.equal_allocation] gives.

        It is a structural allocation: it uses no component data, only the
        diagram, the target and the weights. With every node at 0.5 the
        Birnbaum importance equals the
        [`structural_importance`][repyability.RBD.structural_importance], so
        a small change moves each node in proportion to its weight times its
        structural importance; for larger changes the importances are
        re-evaluated at the new probabilities. It is not one of the classic
        named methods; see
        [`cost_based_allocation`][repyability.RBD.cost_based_allocation] and
        [`minimum_effort_allocation`][repyability.RBD.minimum_effort_allocation]
        for those.

        It is solved with ``scipy.optimize.minimize`` (``trust-constr``),
        from the point that moves every node in proportion to its weight
        just far enough to meet the target, with exact gradients; the
        system probability is handled on the log-odds scale from both ends
        of the exact engine, so it stays exact at any size. A target of 0
        or 1 is approached to within 1e-12. The solver's result is stored
        on the RBD as ``res``, replacing any earlier one; its ``x`` holds
        the log-odds of the nodes with a positive weight, in node order.

        Parameters
        ----------
        target : float
            The system probability to reach, in [0, 1].
        weights : dict, optional
            A non-negative weight per intermediate node, by default None
            (1.0 for every node). If given, it needs an entry for every
            intermediate node.

        Returns
        -------
        dict
            The allocated probability (a float) of every intermediate node,
            keyed by node name.

        Raises
        ------
        ValueError
            If ``target`` is above 1 or below 0; if it cannot be reached
            because nodes with weight 0 hold the system back (the message
            gives the reachable range); or if a weight is negative or not
            finite.
        KeyError
            If ``weights`` is given without an entry for an intermediate
            node.

        Examples
        --------
        With two parallel nodes ``"a"`` and ``"b"`` in series with ``"c"``,
        the series node ``"c"`` moves furthest:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> new = rbd.simple_allocation(0.99)
        >>> {k: round(v, 4) for k, v in sorted(new.items())}
        {'a': 0.9395, 'b': 0.9395, 'c': 0.9936}
        >>> round(float(rbd.system_probability(new)[0]), 4)
        0.99
        """
        # A perfect junction stays perfect, and takes no weight.
        junctions = self._junctions()
        weight_of = {
            n: 1.0 if weights is None else float(weights[n])
            for n in self.nodes
            if n not in junctions
        }
        for node, value in weight_of.items():
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(
                    f"weights[{node!r}] must be a finite, non-negative "
                    f"number, got {value}."
                )
        free = [n for n in weight_of if weight_of[n] > 0.0]
        weight = np.array([weight_of[n] for n in free])

        def probabilities(log_odds: np.ndarray) -> tuple[Dict, Dict]:
            p = {n: 1.0 if n in junctions else 0.5 for n in self.nodes}
            q = {n: 1.0 - p[n] for n in self.nodes}
            for i, n in enumerate(free):
                p[n] = float(sigmoid(log_odds[i]))
                q[n] = float(sigmoid(-log_odds[i]))
            return p, q

        def system(log_odds: np.ndarray) -> tuple[float, np.ndarray]:
            """The system's log-odds, and its gradient."""
            p, q = probabilities(log_odds)
            value, derivative = self._log_odds(p, q)
            return value, np.array([derivative[n] * p[n] * q[n] for n in free])

        goal = float(logit(np.clip(target, 1e-12, 1.0 - 1e-12)))
        low = system(np.full(len(free), -np.inf))[0]
        high = system(np.full(len(free), np.inf))[0]
        if not low < goal < high:
            raise ValueError(
                f"target {target} cannot be reached: with the weights given "
                "(a node with weight 0 stays at 0.5), the system probability "
                f"can only lie strictly between {sigmoid(low):.6g} and "
                f"{sigmoid(high):.6g}."
            )

        def along_weights(start: np.ndarray) -> np.ndarray:
            """``start`` moved along the weights just far enough to meet
            the target (the system's log-odds is monotone along them)."""
            low_c, high_c = -1.0, 1.0
            while system(start + low_c * weight)[0] > goal:
                low_c *= 2.0
            while system(start + high_c * weight)[0] < goal:
                high_c *= 2.0
            c = brentq(
                lambda c: system(start + c * weight)[0] - goal,
                low_c,
                high_c,
                xtol=1e-14,
            )
            return start + c * weight

        constraint = NonlinearConstraint(
            lambda x: system(x)[0],
            goal,
            goal,
            jac=lambda x: system(x)[1][None, :],
            hess=BFGS(),
        )
        with warnings.catch_warnings():
            # The quasi-Newton update notes a vanishing step once converged.
            warnings.filterwarnings("ignore", message="delta_grad == 0.0")
            res = minimize(
                lambda x: (float(np.sum(x * x / weight)), 2.0 * x / weight),
                along_weights(np.zeros(len(free))),
                jac=True,
                hess=lambda x: diags(2.0 / weight),
                method="trust-constr",
                constraints=[constraint],
                options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 2000},
            )
        log_odds = res.x
        if abs(system(log_odds)[0] - goal) > 1e-10:
            # Close the solver's last sliver of constraint tolerance.
            log_odds = along_weights(log_odds)
            res.x = log_odds
        self.res = res
        return {n: float(v) for n, v in probabilities(log_odds)[0].items()}

    def _allocation_probability(
        self, probabilities: Dict[Any, float]
    ) -> float:
        """The system probability at single node probabilities, as the
        allocation methods score it (with ``_log_odds``)."""
        node_arrays = {k: np.atleast_1d(v) for k, v in probabilities.items()}
        return self.system_probability(node_arrays, method="p").item()

    def _allocation_start(self, node_probabilities: Dict) -> Dict[Any, float]:
        """The current probability of every intermediate node, validated.
        Other keys (e.g. the input and output nodes) are not used."""
        junctions = self._junctions()
        missing = [
            n
            for n in self.nodes
            if n not in node_probabilities and n not in junctions
        ]
        if missing:
            raise ValueError(
                "node_probabilities needs the current probability of every "
                f"intermediate node; missing {missing}."
            )
        return {
            node: (
                1.0
                if node in junctions
                else _probability_value(
                    f"node_probabilities[{node!r}]", node_probabilities[node]
                )
            )
            for node in self.nodes
        }

    def _node_overrides(self, name: str, mapping: Optional[Dict]) -> Dict:
        """A per-node option dict, checked to name intermediate nodes only
        (so a typo cannot fall back to the default unnoticed)."""
        if mapping is None:
            return {}
        nodes = set(self.nodes)
        unknown = [n for n in mapping if n not in nodes]
        if unknown:
            raise ValueError(
                f"{name} has entries for {unknown}, which are not "
                "intermediate nodes."
            )
        return dict(mapping)

    def _log_odds(
        self, p: Dict[Any, float], q: Dict[Any, float]
    ) -> tuple[float, Dict[Any, float]]:
        """The log-odds ``log(R / (1 - R))`` that the system works, and its
        derivative with respect to each node's probability, from the node
        probabilities ``p`` and their complements ``q``. ``R`` and
        ``1 - R`` are each computed as sums of products, so both keep their
        full relative precision, even within 1e-300 of 0 or 1, as does each
        node's Birnbaum importance."""
        R, Q, importance = self._decomposition().value_and_gradient(p, q)
        with np.errstate(divide="ignore"):
            log_odds = float(np.log(R) - np.log(Q))
        scale = R * Q
        return log_odds, {
            n: importance.get(n, 0.0) / scale if scale else 0.0
            for n in self.nodes
        }

    @leaves_out_junctions
    @check_probability
    def minimum_effort_allocation(
        self, target: float, node_probabilities: Dict
    ) -> Dict[Any, float]:
        """Albert's minimum-effort allocation for a series system.

        The minimization-of-effort algorithm (Albert, 1958; MIL-HDBK-338B)
        raises a series system to ``target`` with the least total effort:
        with the current probabilities in ascending order ``R_1 <= ... <=
        R_n``, it raises the ``k`` least reliable nodes to one common level

            R_0 = (target / (R_{k+1} * ... * R_n)) ** (1 / k),

        where ``k`` is the largest ``j`` with ``R_j`` below
        ``(target / (R_{j+1} * ... * R_n)) ** (1 / j)``, and leaves the other
        nodes unchanged. The result holds for any effort function the nodes
        share that meets Albert's conditions (the effort of raising a
        reliability from ``x`` to ``y`` is non-negative, grows with ``y``
        and adds up over successive steps, among others), such as
        ``y - x`` or ``log((1 - x) / (1 - y))``: the answer does not depend
        on which. It needs no solver, so ``res`` is not changed.

        Parameters
        ----------
        target : float
            The system probability to reach, in [0, 1].
        node_probabilities : Dict
            The current probability that each intermediate node works (a
            single value in [0, 1] per node), keyed by node name. Every
            intermediate node needs an entry; other keys are not used.

        Returns
        -------
        dict
            The allocated probability (a float) of every intermediate node,
            keyed by node name. A target the system already meets returns
            the current probabilities.

        Raises
        ------
        ValueError
            If ``target`` is above 1 or below 0; if the diagram is not a
            series system (one path through every intermediate node; use
            [`cost_based_allocation`][repyability.RBD.cost_based_allocation]
            for other structures); or if a node has no probability, or one
            that is not a single value in [0, 1].

        References
        ----------
        A. Albert, "A measure of the effort required to increase
        reliability", Technical Report No. 43, Applied Mathematics and
        Statistics Laboratory, Stanford University, 1958.

        MIL-HDBK-338B, Electronic Reliability Design Handbook, 1998:
        "Minimization of effort algorithm".

        Examples
        --------
        The two weakest of four nodes in series are raised together; the
        others are already good enough:

        >>> from repyability import RBD
        >>> rbd = RBD([("s", "a"), ("a", "b"), ("b", "c"), ("c", "d"),
        ...            ("d", "t")])
        >>> current = {"a": 0.7, "b": 0.8, "c": 0.9, "d": 0.95}
        >>> new = rbd.minimum_effort_allocation(0.6, current)
        >>> {k: round(v, 4) for k, v in new.items()}
        {'a': 0.8377, 'b': 0.8377, 'c': 0.9, 'd': 0.95}
        >>> round(float(rbd.system_probability(new)[0]), 4)
        0.6
        """
        current = self._allocation_start(node_probabilities)
        self._require_series()
        # In logs, so long chains cannot underflow; zeros sort first.
        order = sorted(self.nodes, key=lambda n: current[n])
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log([current[n] for n in order])
            log_target = np.log(target)
            # rest[j]: the log of the product of the nodes after the j-th.
            rest = np.append(np.cumsum(logs[::-1])[::-1][1:], 0.0)
            raised = 0
            for j in range(len(order), 0, -1):
                if logs[j - 1] < (log_target - rest[j - 1]) / j:
                    raised = j
                    break
            allocation = dict(current)
            if raised:
                level = float(np.exp((log_target - rest[raised - 1]) / raised))
                for node in order[:raised]:
                    allocation[node] = level
        return allocation

    def _require_series(self) -> None:
        """A ValueError unless the diagram is a series system, as the
        minimum-effort algorithm needs: in series, every node alone is a cut
        set (and so is in the only path set)."""
        cut_sets = self.get_min_cut_sets()
        junctions = self._junctions()
        if any(
            frozenset([node]) not in cut_sets
            for node in self.nodes
            if node not in junctions
        ):
            raise ValueError(
                "the minimum-effort algorithm applies to a series system (a "
                "single path through every intermediate node); use "
                "cost_based_allocation for other structures."
            )

    @leaves_out_junctions
    @check_probability
    def cost_based_allocation(
        self,
        target: float,
        node_probabilities: Dict,
        max_probabilities: Optional[Dict] = None,
        feasibility: Optional[Dict] = None,
    ) -> Dict[Any, float]:
        """Mettas's cost-based allocation: the cheapest way to a target.

        Reliability allocation as an optimisation (Mettas, 2000): find the
        node probabilities ``R_i`` that meet the system target at the least
        total cost ``sum_i c_i(R_i)``, where each node's cost of improvement
        is

            c_i(R_i) = exp((1 - f_i) * (R_i - R_min_i) / (R_max_i - R_i)),

        ``R_min_i`` is its current probability, ``R_max_i`` the most it can
        reach and ``f_i`` in [0, 1) its feasibility: how easily it can be
        improved relative to the others (the cost rises faster for a lower
        ``f_i``). The cost is 1 at the current probability and grows without
        bound towards the maximum, so every node stays within its bounds and
        none reaches its maximum. It works on any structure: the system
        probability is exact, and the improvement goes where it is cheapest
        per unit of system probability, which depends on each node's
        importance, current value, maximum and feasibility.

        It is solved with ``scipy.optimize.minimize`` (SLSQP), from the
        point where every node has closed the same fraction of its gap to
        its maximum, using exact gradients. The system probability is
        handled on the log-odds scale, computed from both ends of the exact
        engine, and the cost as the log of its sum, so neither loses
        precision nor overflows, however near 0 or 1 the probabilities are.
        The allocation returned meets the target. The solver's result is
        stored on the RBD as ``res``, replacing any earlier one; its ``fun``
        is the log of the total cost of the nodes allowed to change.

        Parameters
        ----------
        target : float
            The system probability to reach, in [0, 1].
        node_probabilities : Dict
            The current probability that each intermediate node works (a
            single value in [0, 1] per node), keyed by node name: the lower
            bound ``R_min``. Every intermediate node needs an entry; other
            keys are not used.
        max_probabilities : dict, optional
            The most each node's probability can be raised to, ``R_max``
            (at least its current probability; equal to it holds the node
            fixed). A node without an entry can approach 1.
        feasibility : dict, optional
            Each node's feasibility ``f`` in [0, 1): higher is easier to
            improve. A node without an entry gets 0.5.

        Returns
        -------
        dict
            The allocated probability (a float) of every intermediate node,
            keyed by node name. A target the system already meets returns
            the current probabilities.

        Raises
        ------
        ValueError
            If ``target`` is above 1 or below 0; if it cannot be reached
            (the message gives the reachable range: the maximum is only
            approached, at an ever-growing cost); if a node has no
            probability, or one that is not a single value in [0, 1]; if a
            maximum is below its node's current probability or a
            feasibility is outside [0, 1); or if ``max_probabilities`` or
            ``feasibility`` names a node that is not an intermediate node.

        Warns
        -----
        UserWarning
            If the solver stops before converging; the allocation returned
            still meets the target, but may not be the cheapest.

        References
        ----------
        A. Mettas, "Reliability allocation and optimization for complex
        systems", Proceedings of the Annual Reliability and Maintainability
        Symposium, 2000, pp. 216-221.

        Examples
        --------
        Two parallel pumps in series with a valve, all at 0.9, to reach
        0.99. The valve is on every path, so it must reach 0.99 itself; the
        cheapest allocation takes it only a little further and raises the
        pumps to supply the rest:

        >>> from repyability import RBD
        >>> rbd = RBD([("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...            ("v", "t")])
        >>> current = {"p1": 0.9, "p2": 0.9, "v": 0.9}
        >>> new = rbd.cost_based_allocation(0.99, current)
        >>> {k: round(v, 4) for k, v in new.items()}
        {'p1': 0.9808, 'p2': 0.9808, 'v': 0.9904}

        A valve that is harder to improve stops even closer to 0.99, and the
        pumps do more:

        >>> new = rbd.cost_based_allocation(
        ...     0.99, current, feasibility={"v": 0.1}
        ... )
        >>> {k: round(v, 4) for k, v in new.items()}
        {'p1': 0.9896, 'p2': 0.9896, 'v': 0.9901}
        """
        current = self._allocation_start(node_probabilities)
        maximum = {node: 1.0 for node in self.nodes}
        for node, value in self._node_overrides(
            "max_probabilities", max_probabilities
        ).items():
            maximum[node] = _probability_value(
                f"max_probabilities[{node!r}]", value
            )
            if maximum[node] < current[node]:
                raise ValueError(
                    f"max_probabilities[{node!r}] ({maximum[node]}) is below "
                    f"the node's current probability ({current[node]})."
                )
        ease = {node: 0.5 for node in self.nodes}
        for node, value in self._node_overrides(
            "feasibility", feasibility
        ).items():
            ease[node] = float(value)
            if not 0.0 <= ease[node] < 1.0:
                raise ValueError(
                    f"feasibility[{node!r}] must be in [0, 1), got {value}."
                )
        free = [n for n in self.nodes if maximum[n] > current[n]]
        steepness = np.array([1.0 - ease[n] for n in free])
        gap = np.array([maximum[n] - current[n] for n in free])

        # Each free node is placed by v >= 0, the log of the factor by which
        # its gap to its maximum has shrunk: R = R_max - gap * exp(-v), so
        # (R - R_min) / (R_max - R) = expm1(v) and its cost is
        # exp((1 - f) * expm1(v)).
        def probabilities(v: np.ndarray) -> tuple[Dict, Dict]:
            p = dict(current)
            q = {n: 1.0 - current[n] for n in self.nodes}
            shrunk = gap * np.exp(-v)
            for i, n in enumerate(free):
                p[n] = maximum[n] - shrunk[i]
                q[n] = (1.0 - maximum[n]) + shrunk[i]
            return p, q

        goal = float(logit(target))
        now = self._log_odds(*probabilities(np.zeros(len(free))))[0]
        if goal <= now:
            self.res = OptimizeResult(
                x=np.zeros(len(free)),
                fun=0.0,
                success=True,
                message="The target is already met.",
            )
            return dict(current)
        best = self._log_odds(*probabilities(np.full(len(free), np.inf)))[0]
        # The maximum is only approached (at an ever-growing cost), so a
        # target at it, to within rounding, cannot be met either.
        if goal >= best - 1e-9:
            raise ValueError(
                f"target {target} cannot be reached: from the current "
                f"probabilities ({sigmoid(now):.6g}) the system can only "
                f"approach {sigmoid(best):.6g}, with every node at its "
                "maximum."
            )

        def shortfall(v: np.ndarray) -> float:
            return self._log_odds(*probabilities(v))[0] - goal

        def shortfall_gradient(v: np.ndarray) -> np.ndarray:
            derivative = self._log_odds(*probabilities(v))[1]
            dp = gap * np.exp(-v)
            return np.array([derivative[n] for n in free]) * dp

        def log_total_cost(v: np.ndarray) -> tuple[float, np.ndarray]:
            costs = steepness * np.expm1(v)
            return float(logsumexp(costs)), softmax(
                costs
            ) * steepness * np.exp(v)

        def common_shift(start: np.ndarray) -> np.ndarray:
            """The start moved up by the least common v meeting the target
            (it exists, as the target is below the reachable maximum)."""
            high = 1.0
            while shortfall(start + high) < 0.0 and high < 1e6:
                high *= 2.0
            return start + brentq(
                lambda d: shortfall(start + d), 0.0, high, xtol=1e-14
            )

        res = minimize(
            log_total_cost,
            common_shift(np.zeros(len(free))),
            jac=True,
            method="SLSQP",
            bounds=[(0.0, None)] * len(free),
            constraints=[
                {"type": "ineq", "fun": shortfall, "jac": shortfall_gradient}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        v = np.maximum(res.x, 0.0)
        if shortfall(v) < 0.0:
            # Close the solver's last sliver of constraint tolerance.
            v = common_shift(v)
        res.x = v
        res.fun = log_total_cost(v)[0]
        self.res = res
        if not res.success:
            warnings.warn(
                "the cost minimisation stopped before converging "
                f"({res.message}); the allocation meets the target but may "
                "not be the cheapest.",
                stacklevel=3,
            )
        return {n: float(v) for n, v in probabilities(v)[0].items()}

    def node_names(self) -> list[Hashable]:
        """Return the names of the intermediate (component) nodes.

        Every node of the diagram except the input and output nodes, in the
        order the nodes were added to the graph (for a feasible RBD, the
        order of first appearance in ``edges``). Irrelevant nodes are
        included. In a ``NonRepairableRBD`` a repeated node is the
        component it repeats, so only the latter is listed.

        Returns
        -------
        list[Hashable]
            The intermediate node names, as a new list.

        Examples
        --------
        >>> from repyability import RBD
        >>> RBD([("s", "a"), ("a", "b"), ("b", "t")]).node_names()
        ['a', 'b']
        """
        return list(self.nodes)

    @leaves_out_junctions
    def structural_importance(
        self,
        working_nodes: Optional[Iterable[Hashable]] = None,
        broken_nodes: Optional[Iterable[Hashable]] = None,
    ) -> dict[Any, float]:
        """Return the structural (probability-free) importance of every node.

        The fraction of the states of the *other* nodes in which the node is
        pivotal -- the system works when the node works and fails when it
        fails, holding the others fixed. A perfect junction node (a
        ``PerfectReliability`` drawing device, such as a k-out-of-n vote) is
        always working, and is left out. It is computed exactly as the
        Birnbaum importance with every node probability at 1/2, so it
        depends only on the RBD's structure and not on any failure model --
        useful at design time, before any life data exists. It is the same
        for a [`NonRepairableRBD`][repyability.NonRepairableRBD] and a
        [`RepairableRBD`][repyability.RepairableRBD] with the same diagram,
        and a common-cause group does not change it.

        Parameters
        ----------
        working_nodes : Iterable[Hashable], optional
            Nodes to condition on as working (probability 1), by default
            None.
        broken_nodes : Iterable[Hashable], optional
            Nodes to condition on as failed (probability 0), by default
            None. With either, the fraction is taken over the states of the
            nodes that are not forced. A forced node's own importance is
            still computed, given the other forced nodes.

        Returns
        -------
        dict[Any, float]
            The structural importance of each intermediate node, in
            ``[0, 1]``, keyed by node name.

        Raises
        ------
        ValueError
            If a node is in both ``working_nodes`` and ``broken_nodes``, is
            the input or output node, or is not an intermediate node of the
            RBD. A ``NonRepairableRBD`` also rejects a repeated node.

        Examples
        --------
        In a two-component parallel system each node is pivotal in half of the
        other node's states, independent of any failure model:

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {
        ...         "a": FixedEventProbability.from_params(0.1),
        ...         "b": FixedEventProbability.from_params(0.4),
        ...     },
        ... )
        >>> si = rbd.structural_importance()
        >>> {k: round(v, 4) for k, v in sorted(si.items())}
        {'a': 0.5, 'b': 0.5}

        Given that ``"a"`` has failed, ``"b"`` is pivotal in every state
        (``"a"``'s own value is unchanged, as it is taken over ``"b"``):

        >>> si = rbd.structural_importance(broken_nodes=["a"])
        >>> {k: round(v, 4) for k, v in sorted(si.items())}
        {'a': 0.5, 'b': 1.0}
        """
        # A perfect junction (see _junctions) is always working: it is no
        # part of the other nodes' states.
        junctions = self._junctions()
        node_probabilities: dict[Any, ArrayLike] = {
            node: np.full(1, 1.0 if node in junctions else 0.5)
            for node in self.nodes
        }
        node_probabilities = self._probabilities_with_overrides(
            node_probabilities, working_nodes, broken_nodes
        )
        importance = self._birnbaum_importance(node_probabilities)
        return {
            node: float(np.asarray(value).reshape(-1)[0])
            for node, value in importance.items()
        }

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the RBD to a JSON-friendly dict.

        The dict holds the RBD's ``type`` and the ``repyability_version``
        that wrote it, plus the constructor inputs as given: the edges, the
        node models (or components), the k-out-of-n values and capacities,
        the input and output nodes, ``on_infeasible_rbd``, and the
        common-cause groups or the downtime cost rate.
        [`from_dict`][repyability.RBD.from_dict] rebuilds the RBD by calling
        its constructor again, so the round trip is faithful even for
        repeated nodes. Node models are serialised structurally: surpyval's
        parametric models in its own format, so an offset, ``p``, ``f0`` and
        a fit's covariance round-trip; the RePyability node models (standby,
        repeated, load-sharing, regression, ``NonRepairable``,
        ``PerfectReliability`` and ``PerfectUnreliability``) and nested RBDs
        recursively. Per-node
        values are stored as lists of entries, so integer and string node
        names both survive JSON. Only a
        [`NonRepairableRBD`][repyability.NonRepairableRBD] or
        [`RepairableRBD`][repyability.RepairableRBD] can be serialised.

        Returns
        -------
        dict
            The serialised RBD.

        Raises
        ------
        NotImplementedError
            If a node model cannot be serialised: one that is neither a
            surpyval model nor a RePyability node model.
        AttributeError
            If called on a bare ``RBD``, which keeps no constructor inputs.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> d = rbd.to_dict()
        >>> d["type"], d["edges"]
        ('NonRepairableRBD', [['s', 'c'], ['c', 't']])
        >>> saved = d["reliabilities"][0]["model"]  # surpyval's own format
        >>> saved["kind"], saved["model"]["distribution"]
        ('surpyval', 'Weibull')
        """
        from repyability.rbd.serialisation import rbd_to_dict

        return rbd_to_dict(self)

    def to_json(self, fp=None, **json_kwargs) -> Optional[str]:
        """Serialise the RBD to JSON: a string, or written to a file.

        Equivalent to ``json.dumps(self.to_dict(), **json_kwargs)``; see
        [`to_dict`][repyability.RBD.to_dict] for what is stored. String,
        integer and tuple node names all survive: JSON turns a tuple into a
        list, and loading turns it back. With ``fp`` it is written there, as
        surpyval's models' ``to_json(fp)`` writes them.

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
            If a node model cannot be serialised (see
            [`to_dict`][repyability.RBD.to_dict]).
        AttributeError
            If called on a bare ``RBD``.
        TypeError
            If a value cannot be encoded as JSON, e.g. a node name that JSON
            has no type for.

        Examples
        --------
        >>> import json
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> json.loads(rbd.to_json(indent=2))["type"]
        'NonRepairableRBD'
        """
        from repyability.rbd.serialisation import rbd_to_json
        from repyability.utils.json_io import write_json

        return write_json(rbd_to_json(self, **json_kwargs), fp)

    @classmethod
    def from_dict(cls, d: dict) -> "RBD":
        """Reconstruct an RBD from the output of ``to_dict``.

        The RBD is rebuilt by calling its constructor with the stored
        inputs, so it is validated again and is equivalent to the original.
        Called on a specific subclass, the document's ``type`` must match;
        called on ``RBD`` it dispatches to whichever type the document names.

        Parameters
        ----------
        d : dict
            A dict made by [`to_dict`][repyability.RBD.to_dict] (possibly
            after a JSON round trip).

        Returns
        -------
        RBD
            The reconstructed ``NonRepairableRBD`` or ``RepairableRBD``.

        Raises
        ------
        ValueError
            If called on a subclass and ``d["type"]`` is not that subclass;
            if the RBD type or a node model kind is unknown; or if the
            constructor rejects the stored inputs.
        KeyError
            If a required entry is missing, e.g. ``"type"`` when called on
            ``RBD``.

        Examples
        --------
        The structure round-trips through a plain dict, so reliability is
        preserved:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> restored = NonRepairableRBD.from_dict(rbd.to_dict())
        >>> round(restored.sf(50), 4) == round(rbd.sf(50), 4)
        True
        """
        from repyability.rbd.serialisation import rbd_from_dict

        if cls is not RBD and cls.__name__ != d.get("type"):
            raise ValueError(
                f"{cls.__name__}.from_dict got a {d.get('type')!r} document; "
                f"use {d.get('type')}.from_dict or RBD.from_dict."
            )
        return rbd_from_dict(d)

    @classmethod
    def from_json(cls, s) -> "RBD":
        """Reconstruct an RBD from a JSON document made by ``to_json``.

        Equivalent to ``cls.from_dict(json.loads(s))``, so the same type
        rules apply (see [`from_dict`][repyability.RBD.from_dict]). ``s``
        can be the document's text or, as surpyval's ``from_json`` takes, a
        path to the file holding it.

        Parameters
        ----------
        s : str, os.PathLike or file
            A JSON document made by [`to_json`][repyability.RBD.to_json]:
            its text, a path to a file holding it, or a file opened for
            reading.

        Returns
        -------
        RBD
            The reconstructed ``NonRepairableRBD`` or ``RepairableRBD``.

        Raises
        ------
        ValueError
            If ``s`` is not valid JSON (``json.JSONDecodeError`` is a
            ValueError), or for the reasons given in ``from_dict``.
        FileNotFoundError
            If ``s`` is neither a JSON document nor the path of a file.
        KeyError
            If a required entry is missing (see ``from_dict``).

        Examples
        --------
        ``RBD.from_json`` returns whichever RBD type the document names:

        >>> import surpyval as surv
        >>> from repyability import RBD, NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> restored = RBD.from_json(rbd.to_json())
        >>> type(restored).__name__
        'NonRepairableRBD'
        >>> round(restored.sf(50), 4)
        0.7788
        """
        import json

        from repyability.utils.json_io import read_json

        return cls.from_dict(json.loads(read_json(s)))

    def _probabilities_with_overrides(
        self,
        node_probabilities: dict,
        working_nodes,
        broken_nodes,
    ) -> dict:
        """Returns a copy of ``node_probabilities`` with any forced nodes set
        to probability 1 (working) or 0 (broken), validating the overrides
        first. Shared by the importance measures and steady-state metrics."""
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        out = dict(node_probabilities)
        for node in working_nodes:
            out[node] = np.ones_like(
                np.atleast_1d(np.asarray(out[node], dtype=float))
            )
        for node in broken_nodes:
            out[node] = np.zeros_like(
                np.atleast_1d(np.asarray(out[node], dtype=float))
            )
        return out

    def _validate_node_overrides(self, working_nodes, broken_nodes) -> None:
        """Validate the working/broken node override sets.

        Raises a ValueError on invalid input rather than silently ignoring it:
        the same node in both sets, the input/output node, or an unknown node
        name (e.g. a typo, which would otherwise silently have no effect and
        return a plausible-but-wrong result).
        """
        working_nodes = set(working_nodes)
        broken_nodes = set(broken_nodes)

        both = working_nodes & broken_nodes
        if both:
            raise ValueError(
                f"Node(s) {sorted(both, key=str)} given as both working and "
                "broken; a node cannot be forced to both states."
            )

        valid = set(self.nodes)
        folded = self._junctions() if self._FOLDS_JUNCTIONS else frozenset()
        for label, nodes in (
            ("working_nodes", working_nodes),
            ("broken_nodes", broken_nodes),
        ):
            for node in nodes:
                if node in self.in_or_out:
                    which = "input" if node == self.input_node else "output"
                    raise ValueError(
                        f"Cannot set the {which} node {node!r} via {label}."
                    )
                if node in folded:
                    raise ValueError(
                        f"Node {node!r} is a junction, which always works: "
                        f"it cannot be set via {label}."
                    )
                if node not in valid:
                    raise ValueError(
                        f"Unknown node {node!r} given to {label}; it is not "
                        "an intermediate node of the RBD. Valid nodes are: "
                        f"{sorted(valid, key=str)}."
                    )

    def _birnbaum_importance(
        self,
        node_probabilities: dict[Any, ArrayLike],
        weights: Optional[np.ndarray] = None,
        node_failures: Optional[dict[Any, ArrayLike]] = None,
    ) -> dict[Any, np.ndarray]:
        """Returns the Birnbaum measure of importance for all nodes:
        ``R(i works) - R(i failed)``, the derivative of the system's
        probability of working with respect to node ``i``'s.

        It is worked out for every node at once, as a sum of products (see
        ``_importances``), so a small one keeps its precision. Note:
        Birnbaum's measure of importance assumes all nodes are independent.

        Parameters
        ----------
        node_probabilities: Dict
            Dictionary containing the probability arrays of the event for every
            node in the RBDGraph. Probability is to be either the reliability
            or the availability (or some other probability that I can't
            conceive).
        weights : np.ndarray, optional
            Weights to average the arrays' elements with (e.g. over time,
            see ``RepairableRBD._long_run_grid``), by default None: one
            value per element. The same for every importance helper here;
            each ratio is then of averages.
        node_failures : Dict, optional
            Each node's probability of failing, when it is known more
            precisely than one less its probability of working (from a
            reliable node's ``ff``, say), by default None. The same for
            every importance helper here (see ``_system_unreliability``).

        Returns
        -------
        dict[Any, ArrayLike]
            Dictionary with node names as keys and Birnbaum importances as
            values
        """
        importance = self._importances(node_probabilities, node_failures)[0]
        return {
            node: _averaged(value, weights)
            for node, value in importance.items()
        }

    def _improvement_potential(
        self,
        node_probabilities: dict[Any, ArrayLike],
        weights: Optional[np.ndarray] = None,
        node_failures: Optional[dict[Any, ArrayLike]] = None,
    ) -> dict[Any, np.ndarray]:
        """Returns the improvement potential of all nodes: ``R(i works) -
        R``, which is ``I_B(i) * (1 - p_i)`` (the system's probability of
        working is linear in each node's), worked out as that product so
        that a small one keeps its precision.

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works (arrays of one length).
        weights, node_failures : optional
            As for ``_birnbaum_importance``.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and improvement potentials as
            values
        """
        importance, _, _, _, q = self._importances(
            node_probabilities, node_failures
        )
        return {
            node: _averaged(importance[node] * q[node], weights)
            for node in self.nodes
        }

    def _risk_achievement_worth(
        self,
        node_probabilities: dict[Any, ArrayLike],
        weights: Optional[np.ndarray] = None,
        node_failures: Optional[dict[Any, ArrayLike]] = None,
    ) -> dict[Any, np.ndarray]:
        """Returns the RAW importance per Modarres & Kaminskiy. That is RAW_i =
        (unreliability of system given i failed) /
        (nominal system unreliability), each unreliability a sum of
        products (see ``_system_unreliability``).

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works (arrays of one length).
        weights, node_failures : optional
            As for ``_birnbaum_importance``.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RAW importances as values
        """
        p, q, _ = self._node_pairs(node_probabilities, node_failures)
        as_is = _averaged(self._system_unreliability(p, q), weights)
        node_importance: dict[Any, np.ndarray] = {}
        for node in self.nodes:
            one, zero = np.ones_like(p[node]), np.zeros_like(p[node])
            when_failed = self._system_unreliability(
                {**p, node: zero}, {**q, node: one}
            )
            node_importance[node] = _averaged(when_failed, weights) / as_is
        return node_importance

    def _risk_reduction_worth(
        self,
        node_probabilities: dict[Any, ArrayLike],
        weights: Optional[np.ndarray] = None,
        node_failures: Optional[dict[Any, ArrayLike]] = None,
    ) -> dict[Any, np.ndarray]:
        """Returns the RRW importance per Modarres & Kaminskiy. That is RRW_i =
        (nominal unreliability of system) /
        (unreliability of system given i is working), each unreliability a
        sum of products (see ``_system_unreliability``).

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works (arrays of one length).
        weights, node_failures : optional
            As for ``_birnbaum_importance``.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RRW importances as values
        """
        p, q, _ = self._node_pairs(node_probabilities, node_failures)
        as_is = _averaged(self._system_unreliability(p, q), weights)
        node_importance: dict[Any, np.ndarray] = {}
        for node in self.nodes:
            one, zero = np.ones_like(p[node]), np.zeros_like(p[node])
            working = self._system_unreliability(
                {**p, node: one}, {**q, node: zero}
            )
            node_importance[node] = as_is / _averaged(working, weights)
        return node_importance

    def _criticality_importance(
        self,
        node_probabilities: dict[Any, ArrayLike],
        kind: str = "failure",
        weights: Optional[np.ndarray] = None,
        node_failures: Optional[dict[Any, ArrayLike]] = None,
    ) -> dict[Any, np.ndarray]:
        """The criticality importance of every node.

        ``kind="failure"`` (the default) gives the failure-oriented form
        (Rausand & Høyland), ``I_B(i) * (1 - p_i) / (1 - P_sys)``: the
        probability that node ``i`` has failed and is critical, given that
        the system has failed, i.e. the share of system failures node ``i``
        accounts for. It is computed from the system unreliability and the
        Birnbaum importance directly (sums of products, like the
        reliability; see ``_importances``), so it is not lost to
        cancellation in ``1 - P_sys`` however reliable the system is; it is
        ``nan`` where the system cannot fail.

        ``kind="success"`` gives the success-oriented form,
        ``I_B(i) * p_i / P_sys``: the probability that node ``i`` is working
        and critical, given that the system works. It is 1 for every node in
        series with the rest, so it cannot rank them; it is ``nan`` where
        the system cannot work.

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works (arrays of one length).
        kind : str, optional
            ``"failure"`` (the default) or ``"success"``.
        weights, node_failures : optional
            As for ``_birnbaum_importance``.

        Returns
        -------
        dict[Any, np.ndarray]
            ``{node: criticality importance}``.

        Raises
        ------
        ValueError
            If ``kind`` is neither ``"failure"`` nor ``"success"``.
        """
        if kind not in ("failure", "success"):
            raise ValueError(
                f"kind must be 'failure' or 'success', got {kind!r}."
            )
        importance, works, fails, p, q = self._importances(
            node_probabilities, node_failures
        )
        # Success-oriented: critical and working, over the system working;
        # failure-oriented: critical and failed, over the system failing
        # (from the system unreliability itself, so that a small one keeps
        # its precision, where 1 - P_sys would cancel).
        system, member = (works, p) if kind == "success" else (fails, q)
        system = _averaged(system, weights)
        node_importance: dict[Any, np.ndarray] = {}
        for node in self.nodes:
            critical = _averaged(importance[node] * member[node], weights)
            with np.errstate(divide="ignore", invalid="ignore"):
                node_importance[node] = np.where(
                    system > 0, critical / system, np.nan
                )
        return node_importance

    def _fussell_vesely(
        self,
        node_probabilities: dict[Any, ArrayLike],
        fv_type: str = "c",
        method: str = "exact",
        weights: Optional[np.ndarray] = None,
        node_failures: Optional[dict[Any, ArrayLike]] = None,
    ) -> dict[Any, np.ndarray]:
        """The Fussell-Vesely importance of every node: the probability
        that some minimal cut set containing it has failed (every node in
        it), over the probability that the system has failed.

        ``method="exact"`` works out that union exactly, from the exact
        engine's decomposition (see ``Decomposition.failed_cut_sets``);
        ``"rare_event"`` sums the probabilities of the cut sets instead,
        the usual rare-event approximation, which over-estimates the union
        and can exceed 1 when failures are not rare.

        ``fv_type="p"`` puts the minimal path sets in place of the cut sets
        (the union, or the sum, of the probabilities that every node of a
        path set containing the node has failed): the exact union is the
        dual structure's (see ``Decomposition.dual``).

        Each set's probability is a product of its nodes' probabilities of
        failing, and the system's probability of failing is a sum of
        products (see ``_system_unreliability``), so a small one keeps its
        precision.

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works (arrays of one length).
        fv_type : str, optional
            ``"c"`` (cut sets, the default) or ``"p"`` (path sets).
        method : str, optional
            ``"exact"`` (the default) or ``"rare_event"``.
        weights, node_failures : optional
            As for ``_birnbaum_importance``.

        Returns
        -------
        dict[Any, np.ndarray]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set), or
            ``method`` not 'exact' or 'rare_event', or if the node
            probability arrays are not all the same length.
        """
        if fv_type not in ("c", "p"):
            raise ValueError(
                "fv_type must be either 'c' (cut-set) or 'p' (path-set), "
                f"fv_type={fv_type!r} was given."
            )
        if method not in ("exact", "rare_event"):
            raise ValueError(
                "method must be 'exact' or 'rare_event', " f"got {method!r}."
            )
        p, q, size = self._node_pairs(node_probabilities, node_failures)

        # The system unreliability, the denominator for every node.
        system_ff = _averaged(self._system_unreliability(p, q), weights)
        numerators = self._fv_numerators(p, q, size, fv_type, method)
        return {
            node: _averaged(numerators[node], weights) / system_ff
            for node in self.nodes
        }

    def _fv_numerators(
        self, p: dict, q: dict, size: int, fv_type: str, method: str
    ) -> dict[Any, np.ndarray]:
        """The numerators of the Fussell-Vesely importances (see
        ``_fussell_vesely``) at the nodes' probabilities of working ``p``
        and of failing ``q`` (1-d arrays of length ``size``): for each node,
        the probability that every node of some minimal cut (or path) set
        containing it has failed, or with ``"rare_event"`` the sum of those
        sets' probabilities."""
        zero = np.zeros(size)
        if method == "exact":
            # The sets are those with the junctions folded in (#198): a
            # path set with one, which never fails, would never fail.
            decomposition = self._set_structure()
            if decomposition.always_works:
                failed: dict = {}
            else:
                if fv_type == "p":
                    decomposition = decomposition.dual()
                failed = decomposition.failed_cut_sets(p, q, shape=size)
            return {
                node: np.broadcast_to(failed.get(node, zero), (size,))
                for node in self.nodes
            }

        # Get node-sets based on what method was requested
        if fv_type == "c":
            node_sets = self.get_min_cut_sets()
        else:
            node_sets = {
                frozenset(path_set)
                for path_set in self.get_min_path_sets(
                    include_in_out_nodes=False
                )
            }
        out: dict[Any, np.ndarray] = {}
        for this_node in self.nodes:
            # The sum of the probabilities of the sets containing the node
            # failing.
            numerator = np.zeros(size)
            for node_set in node_sets:
                if this_node not in node_set:
                    continue
                set_fails = np.ones(size)
                for other_node in node_set:
                    set_fails = set_fails * q[other_node]
                numerator = numerator + set_fails
            out[this_node] = numerator
        return out
