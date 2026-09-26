"""The reliability block diagram base class and its exact engine.

``RBD`` holds a diagram's structure and the computations that need only the
structure and per-node probabilities; ``NonRepairableRBD`` and
``RepairableRBD`` build on it. The module-level functions are the engine it
uses: the exact probability that at least one set of elements fully works
(a memoised Shannon decomposition), minimal cut sets from minimal path sets,
and the probability scaling used by reliability allocation.
"""

import pprint
import warnings
from collections import defaultdict
from copy import copy
from typing import Any, Dict, Hashable, Iterable, Iterator, Optional

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, brentq, minimize
from scipy.special import expit as sigmoid

from repyability.rbd.min_path_sets import min_path_sets as find_min_path_sets
from repyability.rbd.rbd_graph import RBDGraph
from repyability.utils.wrappers import check_probability


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


# Value slots 0 and 1 of a Shannon plan hold the constant 0 and 1 arrays.
_ZERO, _ONE = 0, 1


def _shannon_plan(sets: Iterable[frozenset]) -> tuple[list, int]:
    """Record the Shannon decomposition used by
    :func:`probability_any_set_satisfied` as a replayable plan.

    The decomposition -- which element to pivot on, and which sub-problems
    recur -- depends only on the sets, not on the probabilities, so it can be
    worked out once and replayed for any probabilities. Returns ``(steps,
    root)``: step ``i`` fills value slot ``i + 2`` with
    ``p[pivot] * value[active] + (1 - p[pivot]) * value[inactive]``, and
    ``root`` is the slot holding the answer.

    The decomposition is a depth-first recursion (the active branch, then
    the inactive one, then the step itself), run on an explicit stack so
    that systems with a thousand or more components do not reach Python's
    recursion limit.
    """
    sets = [frozenset(s) for s in sets]
    slots: Dict[frozenset, int] = {}
    steps: list[tuple[Any, int, int]] = []

    def known(state: frozenset) -> Optional[int]:
        # state is a frozenset of frozensets: the sets still to be satisfied,
        # with already-active elements removed.
        if not state:
            # No set can be satisfied any more -> probability 0.
            return _ZERO
        if frozenset() in state:
            # A set has had all its elements satisfied -> probability 1.
            return _ONE
        return slots.get(state)

    def split(state: frozenset) -> list:
        # Pivot on the element appearing in the most sets, which tends to
        # collapse the problem (and the memo table) fastest.
        counts: Dict[Any, int] = {}
        for s in state:
            for element in s:
                counts[element] = counts.get(element, 0) + 1
        pivot = max(counts, key=lambda e: counts[e])

        # Pivot active: it satisfies its requirement, so drop it from every
        # set that contained it (other sets are unaffected).
        state_active = frozenset(s - {pivot} for s in state)
        # Pivot inactive: any set needing it can never be satisfied -> drop it.
        state_inactive = frozenset(s for s in state if pivot not in s)
        # [state, pivot, (branches still to solve), their solved slots]
        return [state, pivot, [state_active, state_inactive], []]

    root_state = frozenset(sets)
    root = known(root_state)
    stack = [] if root is not None else [split(root_state)]
    while stack:
        state, pivot, branches, solved = stack[-1]
        if len(solved) < 2:
            branch = branches[len(solved)]
            slot = known(branch)
            if slot is None:
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


def _evaluate_shannon_plan(
    plan: tuple[list, int],
    element_probabilities: Dict[Any, np.ndarray],
    array_shape,
) -> np.ndarray:
    """Replay a :func:`_shannon_plan` for the given probabilities."""
    steps, root = plan
    values = [np.zeros(array_shape), np.ones(array_shape)]
    for pivot, active, inactive in steps:
        p = element_probabilities[pivot]
        values.append(p * values[active] + (1 - p) * values[inactive])
    return values[root]


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
    uses (see ``_shannon_plan``), which shares every repeated
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


def _minimal_cut_sets(plan: tuple[list, int]) -> set[frozenset]:
    """The minimal cut sets of the structure a :func:`_shannon_plan` was
    built from (see :func:`minimal_cut_sets_from_path_sets`)."""
    steps, root = plan
    # Cut sets as bitmasks (one bit per component), so each union and subset
    # test is a single integer operation.
    components = list(dict.fromkeys(pivot for pivot, _, _ in steps))
    bit = {component: 1 << i for i, component in enumerate(components)}
    # Value slots as in the plan: 0 never works (the empty set is a cut),
    # 1 always works (nothing is a cut), then one per step.
    cuts: list[list[int]] = [[0], []]
    for pivot, active, inactive in steps:
        spare_pivot = cuts[active]
        cuts.append(
            spare_pivot
            + [
                bit[pivot] | rest
                for rest in cuts[inactive]
                if not any(cut & rest == cut for cut in spare_pivot)
            ]
        )
    return {
        frozenset(c for c in components if cut & bit[c]) for cut in cuts[root]
    }


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

    The structure is validated on construction. It is infeasible if it has
    a cycle; if it has not exactly one node with no incoming edges, or not
    exactly one with no outgoing edges (e.g. a node in ``nodes`` that is in
    no edge); if a ``k`` is 0 or greater than the node's number of incoming
    edges; or if ``k`` names a node not in the diagram. ``on_infeasible_rbd``
    sets what then happens, and the full report is kept in
    ``structure_check``. The minimal path sets are found on construction and
    cached, as are the cut sets and the exact engine's decomposition on
    first use, so repeated evaluations are cheap.

    Parameters
    ----------
    edges : Iterable[tuple[Hashable, Hashable]]
        The directed edges ``(from_node, to_node)`` of the diagram, e.g.
        ``[("s", "a"), ("a", "t")]`` for the single component ``"a"``
        between input ``"s"`` and output ``"t"``.
    nodes : Iterable, optional
        Node names that must be in the diagram as well as those in
        ``edges``, by default None. A name that is in no edge is an isolated
        node, which makes the structure infeasible; the subclasses pass
        their component names here so that a component missing from
        ``edges`` is reported.
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
    structure_check : dict
        The validation report, e.g. ``"is_valid"``, ``"has_cycles"``,
        ``"cycles"``, ``"koon_errors"``, ``"koon_warnings"`` and
        ``"irrelevant_nodes"``. The subclasses add their own entries.

    Raises
    ------
    ValueError
        If ``input_node`` or ``output_node`` is not a node of the diagram,
        or is not its source or sink (whatever ``on_infeasible_rbd`` is);
        if the structure is infeasible
        and ``on_infeasible_rbd`` is ``"raise"`` (the message does not
        list the problems: use ``"warn"`` to see them); if
        ``on_infeasible_rbd`` is not one of its three values (the base
        class checks this only for an infeasible structure); or if, with
        ``"warn"`` or ``"ignore"``, no set of working nodes can reach the
        output node (e.g. a ``k`` greater than the node's number of
        incoming edges).

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

    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        nodes: Optional[Iterable] = None,
        k: Optional[dict[Any, int]] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
    ):
        # The constructor is documented in the class docstring (mkdocstrings
        # merges the two). The positional order mirrors the subclasses: the
        # structure -- ``edges`` then ``nodes`` -- comes first, and ``k``
        # (k-out-of-n) is an optional modifier after it.

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

        if has_excess_koon_nodes:
            structure_check["koon_errors"].append(
                "Check if you have repeated KooN nodes"
            )

        if not structure_check["is_valid"]:
            if on_infeasible_rbd == "warn":
                warnings.warn(
                    "Strucutral Errors in RBD:\n"
                    + pprint.pformat(structure_check),
                    stacklevel=2,
                )
            elif on_infeasible_rbd == "raise":
                raise ValueError("RBD not correctly structured")
            elif on_infeasible_rbd == "ignore":
                pass
            else:
                raise ValueError(
                    "'on_infeasible_rbd' must be one of"
                    + " {'raise', 'warn', 'ignore'}"
                )

        self.structure_check = structure_check
        self.input_node = structure_check["input_node"]
        self.output_node = structure_check["output_node"]
        self.in_or_out = [self.input_node, self.output_node]
        self.nodes = [n for n in self.G.nodes if n not in self.in_or_out]
        self.structure_check["has_irrelevant_nodes"] = False
        self.structure_check["irrelevant_nodes"] = set()

        if (
            not structure_check["has_cycles"]
            and not structure_check["has_nodes_with_no_successor"]
        ):
            self.get_min_path_sets()
            irrelevant_nodes = self.find_irrelevant_components()
            if len(irrelevant_nodes) != 0:
                self.structure_check["has_irrelevant_nodes"] = True

            self.structure_check["irrelevant_nodes"] = irrelevant_nodes

    def find_irrelevant_components(self) -> set:
        """Return the nodes that cannot affect whether the system works.

        A node is irrelevant when it is in no minimal path set, e.g. a node
        in parallel with a direct edge, which is a connection that never
        fails. Whether such a node works never changes whether the system
        works, so its Birnbaum and structural importance are zero. It is
        still a node of the RBD (it is in
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
        combined_nodes: set = set().union(*self.get_min_path_sets())
        return set(self.G.nodes).symmetric_difference(combined_nodes)

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
        predecessors. The sets are found by a memoised search back from the
        output node, on construction, and cached; each call returns a new
        set.

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
        # Run min_path_sets() but convert all the inner sets to frozensets
        # and remove the input/output nodes if requested
        if hasattr(self, "_min_path_sets"):
            min_path_sets = self._min_path_sets
        else:
            min_path_sets = find_min_path_sets(
                rbd_graph=self.G,
                curr_node=self.output_node,
                solns={},
            )
            self._min_path_sets: list[set[Hashable]] = min_path_sets

        if min_path_sets == []:
            raise ValueError(
                "RBD has no paths through! Need to re-evaluate the KooN nodes."
            )

        ret_set = set()
        for min_path_set in min_path_sets:
            min_path_set = set(min_path_set)
            if not include_in_out_nodes:
                min_path_set.remove(self.input_node)
                min_path_set.remove(self.output_node)
            ret_set.add(frozenset(min_path_set))

        return ret_set

    def is_system_working(
        self, component_status: dict[Any, bool], method: str
    ) -> bool:
        """Return whether the system works, given which components work.

        This is the structure function. With ``method="p"`` the system works
        when every node of at least one minimal path set works; with
        ``method="c"`` when at least one node of every minimal cut set
        works. Both give the same answer; ``"p"`` is typically faster as it
        does not need the cut sets. The input and output nodes need no
        entry. The sets are cached on first use, so repeated calls (as in
        the simulations) are cheap.

        Parameters
        ----------
        component_status : dict[Any, bool]
            Whether each component is working (truthy) or failed (falsy),
            keyed by node name. Every node in a minimal path (or cut) set
            needs an entry; other keys are ignored.
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
        # The system structure function is evaluated directly from the minimal
        # path/cut sets, which is plenty fast for the rate at which this is
        # called in the simulations (no Binary Decision Diagram required).
        #
        # - path-set ("p"): the system works iff at least one minimal path set
        #   has all of its components working.
        # - cut-set ("c"): the system works iff every minimal cut set has at
        #   least one of its components working.
        #
        # The path/cut sets (excluding the input/output nodes) are cached on
        # first use so repeated calls during a simulation are cheap.
        if method == "p":
            if not hasattr(self, "_eval_path_sets"):
                self._eval_path_sets = [
                    tuple(path_set)
                    for path_set in self.get_min_path_sets(
                        include_in_out_nodes=False
                    )
                ]
            status = component_status.__getitem__
            return any(
                all(map(status, path_set)) for path_set in self._eval_path_sets
            )
        elif method == "c":
            if not hasattr(self, "_eval_cut_sets"):
                self._eval_cut_sets = [
                    tuple(cut_set)
                    for cut_set in self.get_min_cut_sets(
                        include_in_out_nodes=False
                    )
                ]
            status = component_status.__getitem__
            return all(
                any(map(status, cut_set)) for cut_set in self._eval_cut_sets
            )
        else:
            raise ValueError("`method` must be either 'p' or 'c'")

    def get_min_cut_sets(
        self, include_in_out_nodes=False
    ) -> set[frozenset[Hashable]]:
        """Return the minimal cut sets of the RBD.

        A cut set is a set of nodes whose failure is enough for the system
        to fail; it is minimal when no node can be left out. The minimal cut
        sets are the minimal transversals (hitting sets) of the minimal path
        sets, read off the exact engine's Shannon decomposition (see
        ``minimal_cut_sets_from_path_sets`` in this module), so k-out-of-n
        values are accounted for. They are worked out on first use and
        cached for each value of ``include_in_out_nodes``; each call returns
        a new set.

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
        # The structure is fixed once built (the path sets are cached too),
        # so the cut sets are worked out once per RBD; each call gets its own
        # copy of the set.
        if not hasattr(self, "_min_cut_sets"):
            self._min_cut_sets: dict[bool, set[frozenset[Hashable]]] = {}
        key = bool(include_in_out_nodes)
        if key not in self._min_cut_sets:
            if key:
                plan = _shannon_plan(
                    self.get_min_path_sets(include_in_out_nodes=True)
                )
            else:
                # The exact engine's own plan (built on first use).
                plan = self._shannon_plan("p")
            self._min_cut_sets[key] = _minimal_cut_sets(plan)
        return set(self._min_cut_sets[key])

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

        The result is exact. The structure function is expanded by a
        Shannon (pivotal) decomposition over the minimal path sets
        (``method="p"``) or, with the node unreliabilities ``1 - p``, over
        the minimal cut sets (``method="c"``); repeated sub-problems are
        solved once. The decomposition depends only on the structure, so it
        is built on the first call for each method and reused. Both methods
        give the same result; ``"p"`` is the default as it does not need the
        cut sets.

        Parameters
        ----------
        node_probabilities : Dict
            The probability that each node works, keyed by node name: floats,
            or 1-d arrays (e.g. one value per time) that all have the same
            length. Every intermediate node needs an entry; entries for the
            input and output nodes, and any other keys, are not used. The
            dict is not modified.
        method : str, optional
            ``"p"`` (path sets, the default) or ``"c"`` (cut sets).

        Returns
        -------
        np.ndarray
            The system probability, a 1-d array with one value per element
            of the node arrays. It is always an array: float inputs give a
            one-element array.

        Raises
        ------
        ValueError
            If the node probability arrays are not all the same length.
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

        node_probabilities = copy(node_probabilities)
        lengths = np.array([], dtype=np.int64)
        for node in self.nodes:
            node_array = np.atleast_1d(node_probabilities[node])
            node_probabilities[node] = node_array
            lengths = np.append(lengths, len(node_array))

        if np.any(lengths[0] != lengths[1:]):
            raise ValueError("Probability arrays must be same length")
        else:
            # get shape of input array
            array_shape = lengths[0]

        if method == "p":
            # The system reliability is the probability that at least one
            # minimal path set has all of its components working.
            return _evaluate_shannon_plan(
                self._shannon_plan("p"), node_probabilities, array_shape
            )

        # method == "c": work with cut sets and node unreliabilities. The
        # system unreliability is the probability that at least one minimal
        # cut set has all of its components failed.
        node_unreliability = {k: 1 - v for k, v in node_probabilities.items()}
        system_unreliability = _evaluate_shannon_plan(
            self._shannon_plan("c"), node_unreliability, array_shape
        )
        return 1 - system_unreliability

    def _shannon_plan(self, method: str) -> tuple[list, int]:
        """The exact engine's plan over the minimal path sets (``"p"``) or
        cut sets (``"c"``). It depends only on the structure, so it is built
        on first use and reused by every later evaluation (importance
        measures, redundancy allocation and the repairable closed forms call
        the engine many times)."""
        if not hasattr(self, "_shannon_plans"):
            self._shannon_plans: dict[str, tuple[list, int]] = {}
        if method not in self._shannon_plans:
            if method == "p":
                sets = self.get_min_path_sets(include_in_out_nodes=False)
            else:
                sets = self.get_min_cut_sets(include_in_out_nodes=False)
            self._shannon_plans[method] = _shannon_plan(sets)
        return self._shannon_plans[method]

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
        probabilities: Dict[Any, float] = {}
        for node, value in node_probabilities.items():
            value = np.asarray(value, dtype=float)
            if value.size != 1:
                raise ValueError(
                    f"node_probabilities[{node!r}] must be a single "
                    f"probability, got {value.size} values."
                )
            if not 0.0 <= value.item() <= 1.0:
                raise ValueError(
                    f"node_probabilities[{node!r}] must be in [0, 1], got "
                    f"{value.item()}."
                )
            probabilities[node] = value.item()
        for node in self.nodes:
            probabilities.setdefault(node, 0.5)

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
            node_arrays = {
                k: np.atleast_1d(v) for k, v in allocated(m).items()
            }
            return self.system_probability(node_arrays, method="p").item()

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

    @check_probability
    def simple_allocation(
        self,
        target: float,
        weights=None,
    ):
        """Find node probabilities that meet a target, by optimisation.

        Searches for node probabilities ``p_i = sigmoid(w_i * z_i)`` (which
        keeps them in (0, 1)), with ``w_i`` the node's weight, that minimise
        ``(target - R) ** 2``, where ``R`` is
        [`system_probability`][repyability.RBD.system_probability].
        ``scipy.optimize.minimize`` (BFGS) starts from ``z = 0``, i.e. every
        node at 0.5. Any node models are ignored.

        Many allocations meet a target; this returns the one the optimiser
        reaches. The error's gradient for a node scales with its weight and
        its Birnbaum importance, so nodes that matter more to the system
        (e.g. a node in series with a redundant pair) and nodes with larger
        weights move further from 0.5; a weight of 0 keeps a node at 0.5.
        With equal weights, nodes placed symmetrically (e.g. all in series,
        or all in parallel) get the same value.

        The scipy result is stored on the RBD as ``res``, replacing any
        earlier one. Because the tolerance is very tight, ``res.success`` is
        often False ("precision loss") even when the target is met, so check
        the result with ``system_probability`` instead.

        Parameters
        ----------
        target : float
            The system probability to reach, in [0, 1]; 0 and 1 can only be
            approached.
        weights : dict, optional
            A weight per intermediate node, by default None (1.0 for every
            node). If given, it needs an entry for every intermediate node.

        Returns
        -------
        dict
            The allocated probability (a numpy float) of every intermediate
            node, keyed by node name.

        Raises
        ------
        ValueError
            If ``target`` is above 1 or below 0.
        KeyError
            If ``weights`` is given without an entry for an intermediate
            node.

        Examples
        --------
        With two parallel nodes ``"a"`` and ``"b"`` in series with ``"c"``,
        the series node ``"c"`` gets the highest probability:

        >>> from repyability import RBD
        >>> rbd = RBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
        ... )
        >>> new = rbd.simple_allocation(0.99)
        >>> {k: round(float(v), 4) for k, v in sorted(new.items())}
        {'a': 0.9112, 'b': 0.9112, 'c': 0.9979}
        >>> round(float(rbd.system_probability(new)[0]), 4)
        0.99
        """
        node_array_indices = {k: i for i, k in enumerate(self.nodes)}

        if weights is None:
            weights = {n: 1.0 for n in self.nodes}

        def func(node_probabilities_array):
            node_probabilities = {
                node: sigmoid(
                    weights[node]
                    * node_probabilities_array[node_array_indices[node]]
                )
                for node in self.nodes
            }
            system_probability = self.system_probability(node_probabilities)
            loss = target - system_probability
            return loss**2

        res = minimize(func, np.zeros(len(self.nodes)), tol=1e-20)

        node_probabilities = {
            node: sigmoid(weights[node] * res["x"][node_array_indices[node]])
            for node in self.nodes
        }

        self.res = res
        return node_probabilities

    def node_names(self) -> list[Hashable]:
        """Return the names of the intermediate (component) nodes.

        Every node of the diagram except the input and output nodes, in the
        order the nodes were added to the graph (for a feasible RBD, the
        order of first appearance in ``edges``). Irrelevant nodes are
        included. In a ``NonRepairableRBD`` a repeated node is merged into
        the node it repeats, so only the latter is listed.

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

    def structural_importance(
        self,
        working_nodes: Optional[Iterable[Hashable]] = None,
        broken_nodes: Optional[Iterable[Hashable]] = None,
    ) -> dict[Any, float]:
        """Return the structural (probability-free) importance of every node.

        The fraction of the states of the *other* nodes in which the node is
        pivotal -- the system works when the node works and fails when it
        fails, holding the others fixed. It is computed exactly as the
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
        node_probabilities: dict[Any, ArrayLike] = {
            node: np.full(1, 0.5) for node in self.nodes
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
        node models (or components), the k-out-of-n values, the input and
        output nodes, ``on_infeasible_rbd``, and the common-cause groups or
        the downtime cost rate. [`from_dict`][repyability.RBD.from_dict]
        rebuilds the RBD by calling its constructor again, so the round
        trip is faithful even for repeated nodes. Node models are serialised
        structurally: surpyval parametric distributions as their name and
        parameters; the RePyability node models (standby, repeated,
        load-sharing, regression, ``NonRepairable``, ``PerfectReliability``
        and ``PerfectUnreliability``) and nested RBDs recursively. Per-node
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
            If a node model cannot be serialised, e.g. a fitted
            non-parametric model (surpyval has no API to rebuild one).
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
        >>> d["reliabilities"][0]["model"]
        {'kind': 'parametric', 'dist': 'Weibull', 'params': [100.0, 2.0]}
        """
        from repyability.rbd.serialisation import rbd_to_dict

        return rbd_to_dict(self)

    def to_json(self, **json_kwargs) -> str:
        """Serialise the RBD to a JSON string.

        Equivalent to ``json.dumps(self.to_dict(), **json_kwargs)``; see
        [`to_dict`][repyability.RBD.to_dict] for what is stored. String,
        integer and tuple node names all survive: JSON turns a tuple into a
        list, and loading turns it back.

        Parameters
        ----------
        **json_kwargs
            Passed to ``json.dumps``, e.g. ``indent=2``.

        Returns
        -------
        str
            The JSON document.

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

        return rbd_to_json(self, **json_kwargs)

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
    def from_json(cls, s: str) -> "RBD":
        """Reconstruct an RBD from a JSON string made by ``to_json``.

        Equivalent to ``cls.from_dict(json.loads(s))``, so the same type
        rules apply (see [`from_dict`][repyability.RBD.from_dict]).

        Parameters
        ----------
        s : str
            A JSON document made by [`to_json`][repyability.RBD.to_json].

        Returns
        -------
        RBD
            The reconstructed ``NonRepairableRBD`` or ``RepairableRBD``.

        Raises
        ------
        ValueError
            If ``s`` is not valid JSON (``json.JSONDecodeError`` is a
            ValueError), or for the reasons given in ``from_dict``.
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

        return cls.from_dict(json.loads(s))

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
                if node not in valid:
                    raise ValueError(
                        f"Unknown node {node!r} given to {label}; it is not "
                        "an intermediate node of the RBD. Valid nodes are: "
                        f"{sorted(valid, key=str)}."
                    )

    def _birnbaum_importance(
        self, node_probabilities: dict[Any, ArrayLike]
    ) -> dict[Any, np.ndarray]:
        """Returns the Birnbaum measure of importance for all nodes.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent.

        Parameters
        ----------
        node_probabilities: Dict
            Dictionary containing the probability arrays of the event for every
            node in the RBDGraph. Probability is to be either the reliability
            or the availability (or some other probability that I can't
            conceive).

        Returns
        -------
        dict[Any, ArrayLike]
            Dictionary with node names as keys and Birnbaum importances as
            values
        """

        node_importance: dict[Any, np.ndarray] = {}
        for node in self.nodes:
            node_probabilities_i = {
                **node_probabilities,
                **{node: np.ones_like(node_probabilities[node])},
            }
            guaranteed = self.system_probability(node_probabilities_i)
            node_probabilities_i = {
                **node_probabilities,
                **{node: np.zeros_like(node_probabilities[node])},
            }
            guaranteed_not: np.ndarray = self.system_probability(
                node_probabilities_i
            )
            node_importance[node] = guaranteed - guaranteed_not
        return node_importance

    def _improvement_potential(
        self, node_probabilities: dict[Any, ArrayLike]
    ) -> dict[Any, np.ndarray]:
        """Returns the improvement potential of all nodes.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and improvement potentials as
            values
        """
        node_importance: dict[Any, np.ndarray] = {}
        for node in self.nodes:
            node_probabilities_i = {
                **node_probabilities,
                **{node: np.ones_like(node_probabilities[node])},
            }
            when_working = self.system_probability(node_probabilities_i)
            as_is: np.ndarray = self.system_probability(node_probabilities)
            node_importance[node] = when_working - as_is
        return node_importance

    def _risk_achievement_worth(
        self, node_probabilities: dict[Any, ArrayLike]
    ) -> dict[Any, np.ndarray]:
        """Returns the RAW importance per Modarres & Kaminskiy. That is RAW_i =
        (unreliability of system given i failed) /
        (nominal system unreliability).

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RAW importances as values
        """
        node_importance: dict[Any, np.ndarray] = {}
        as_is: np.ndarray = 1 - self.system_probability(node_probabilities)
        for node in self.nodes:
            node_probabilities_i = {
                **node_probabilities,
                **{node: np.zeros_like(node_probabilities[node])},
            }
            when_failed = 1 - self.system_probability(node_probabilities_i)
            node_importance[node] = when_failed / as_is
        return node_importance

    def _risk_reduction_worth(
        self, node_probabilities: dict[Any, ArrayLike]
    ) -> dict[Any, np.ndarray]:
        """Returns the RRW importance per Modarres & Kaminskiy. That is RRW_i =
        (nominal unreliability of system) /
        (unreliability of system given i is working).

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RRW importances as values
        """
        node_importance: dict[Any, np.ndarray] = {}
        as_is: np.ndarray = 1 - self.system_probability(node_probabilities)
        for node in self.nodes:
            node_probabilities_i = {
                **node_probabilities,
                **{node: np.ones_like(node_probabilities[node])},
            }
            working = 1 - self.system_probability(node_probabilities_i)
            node_importance[node] = as_is / working
        return node_importance

    def _criticality_importance(
        self, node_probabilities: dict[Any, ArrayLike]
    ) -> dict[Any, np.ndarray]:
        """Returns the criticality importance of all nodes at time/s x.

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and criticality importances as
            values
        """
        bi: dict[Any, np.ndarray] = self._birnbaum_importance(
            node_probabilities
        )
        node_importance: dict[Any, np.ndarray] = {}
        system_sf: np.ndarray = self.system_probability(node_probabilities)
        for node in self.nodes:
            node_importance[node] = (
                bi[node] * node_probabilities[node] / system_sf
            )
        return node_importance

    def _fussell_vesely(
        self,
        node_probabilities: dict[Any, ArrayLike],
        fv_type: str = "c",
        approx: bool = True,
    ) -> dict[Any, np.ndarray]:
        """Calculate Fussell-Vesely importance of all components at time/s x.

        Briefly, the Fussel-Vesely importance measure for node i =
        (sum of probabilities of cut-sets including node i occuring/failing) /
        (the probability of the system failing).

        Typically this measure is implemented using cut-sets as mentioned
        above, although the measure can be implemented using path-sets. Both
        are implemented here.

        fv_type dictates the method:
            "c" - cut-set
            "p" - path-set

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        fv_type : str, optional
            Dictates the method of calculation, 'c' = cut-set and
            'p' = path-set, by default "c"
        approx: bool, optional
            If True uses the sum of failure probabilities as the approximate
            solution to the (1 - PI(1 - Q)) product, by default True

        Returns
        -------
        dict[Any, np.ndarray]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set), or if the
            node probability arrays are not all the same length.
        """
        node_probabilities_new: dict[Any, np.ndarray] = {}
        lengths = np.array([], dtype=np.int64)
        for k, v in node_probabilities.items():
            node_probabilities_new[k] = np.atleast_1d(v)
            lengths = np.append(lengths, len(node_probabilities_new[k]))

        if np.any(lengths[0] != lengths[1:]):
            raise ValueError("Probability arrays must be same length")
        else:
            # get shape of input array
            array_shape = lengths[0]

        # Get node-sets based on what method was requested
        if fv_type == "c":
            node_sets = self.get_min_cut_sets()
        elif fv_type == "p":
            node_sets = {
                frozenset(path_set)
                for path_set in self.get_min_path_sets(
                    include_in_out_nodes=False
                )
            }
        else:
            raise ValueError(
                f"fv_type must be either 'c' (cut-set) or 'p' (path-set), \
                fv_type={fv_type} was given."
            )

        # Get system unreliability, this will be the denominator for all node
        # importance calcs
        system_probability_complement = np.float64(
            1.0
        ) - self.system_probability(node_probabilities_new)

        # The return dict
        node_importance: dict[Any, np.ndarray] = {}

        # For each node,
        for this_node in self.nodes:
            # Sum up the probabilities of the node_sets containing the node
            # from failing
            node_fv_numerator = (
                np.zeros(array_shape) if approx else np.ones(array_shape)
            )
            for node_set in node_sets:
                node_set_fail_prob = np.ones(array_shape)
                if this_node not in node_set:
                    continue
                else:
                    for other_node in node_set:
                        node_set_fail_prob *= (
                            np.ones(array_shape)
                            - node_probabilities_new[other_node]
                        )
                if approx:
                    node_fv_numerator += node_set_fail_prob
                else:
                    node_fv_numerator *= (
                        np.ones(array_shape) - node_set_fail_prob
                    )

            node_fv_numerator = (
                node_fv_numerator if approx else 1 - node_fv_numerator
            )
            node_importance[this_node] = (
                node_fv_numerator / system_probability_complement
            )
        return node_importance
