import pprint
import warnings
from collections import defaultdict
from copy import copy
from typing import Any, Dict, Hashable, Iterable, Iterator, Optional

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, root
from scipy.special import expit as sigmoid

from repyability.rbd.min_path_sets import min_path_sets as find_min_path_sets
from repyability.rbd.rbd_graph import RBDGraph
from repyability.utils.wrappers import check_probability


def log_linearly_scale_probabilities(p: float, x: float) -> np.ndarray:
    """
    Log-linearly scale probabilities.

    Parameters:
        p (float): Probability value.
        x (float): Input value.

    Returns:
        np.ndarray: Log-linearly scaled probabilities.
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
    """
    Scale a dictionary of node probabilities using log-linear scaling.

    Parameters
    ----------
    node_probabilities : Dict[str, float]
        The dictionary of node probabilities to be scaled.
    x : float
        The scaling factor.
    weights : Optional[Dict[str, float]], optional
        The dictionary of weights for each node, by default None.
        If None, a weight of 1.0 is used for all nodes.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of the scaled probability values.

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
    Given the per-element probability of being active, this returns the
    probability that *at least one* set is satisfied.

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
        The collection of sets (e.g. minimal path sets or cut sets).
    element_probabilities : Dict[Any, np.ndarray]
        Maps each element to its probability array of being active.
    array_shape :
        The shape of the probability arrays, used to seed the 0/1 base cases.

    Returns
    -------
    np.ndarray
        The probability that at least one set is fully active.
    """
    sets = [frozenset(s) for s in sets]
    memo: Dict[frozenset, np.ndarray] = {}

    def recurse(state: frozenset) -> np.ndarray:
        # state is a frozenset of frozensets: the sets still to be satisfied,
        # with already-active elements removed.
        if not state:
            # No set can be satisfied any more -> probability 0.
            return np.zeros(array_shape)
        if frozenset() in state:
            # A set has had all its elements satisfied -> probability 1.
            return np.ones(array_shape)
        if state in memo:
            return memo[state]

        # Pivot on the element appearing in the most sets, which tends to
        # collapse the problem (and the memo table) fastest.
        counts: Dict[Any, int] = {}
        for s in state:
            for element in s:
                counts[element] = counts.get(element, 0) + 1
        pivot = max(counts, key=lambda e: counts[e])
        p = element_probabilities[pivot]

        # Pivot active: it satisfies its requirement, so drop it from every
        # set that contained it (other sets are unaffected).
        state_active = frozenset(s - {pivot} for s in state)
        # Pivot inactive: any set needing it can never be satisfied -> drop it.
        state_inactive = frozenset(s for s in state if pivot not in s)

        result = p * recurse(state_active) + (1 - p) * recurse(state_inactive)
        memo[state] = result
        return result

    return recurse(frozenset(sets))


def _keep_minimal_sets(sets: Iterable[frozenset]) -> list[frozenset]:
    """Return only the inclusion-minimal sets.

    Discards any set that is a superset of another (and de-duplicates).
    """
    minimal: list[frozenset] = []
    # Consider smaller sets first so that a kept set can prune its supersets.
    for candidate in sorted(set(sets), key=len):
        if not any(kept <= candidate for kept in minimal):
            minimal.append(candidate)
    return minimal


def minimal_cut_sets_from_path_sets(
    path_sets: Iterable[frozenset],
) -> set[frozenset]:
    """Return the minimal cut sets given the minimal path sets.

    A minimal cut set is a minimal "transversal" (hitting set) of the path
    sets: a smallest set of components that intersects every path set, so that
    failing those components breaks every path through the system.

    This uses Berge's algorithm: it builds the minimal transversals
    incrementally, one path set at a time, discarding non-minimal candidates at
    every step. Unlike taking the full Cartesian product of the path sets and
    filtering once at the end, it never materialises the (potentially enormous)
    product, which makes it dramatically faster in practice. Because it works
    directly from the path sets, it stays correct for k-out-of-n structures,
    whose k-of-n behaviour is already encoded in the path sets.

    Parameters
    ----------
    path_sets : Iterable[frozenset]
        The minimal path sets (each a set of components).

    Returns
    -------
    set[frozenset]
        The minimal cut sets.
    """
    # Start with the single empty transversal and extend it to hit each path
    # set in turn.
    transversals: list[frozenset] = [frozenset()]
    for path_set in path_sets:
        path_set = frozenset(path_set)
        candidates: list[frozenset] = []
        for transversal in transversals:
            if transversal & path_set:
                # Already hits this path set; keep it unchanged.
                candidates.append(transversal)
            else:
                # Must be extended to hit this path set, by one of its
                # components.
                for component in path_set:
                    candidates.append(transversal | {component})
        # Prune non-minimal candidates now so the working set stays small.
        transversals = _keep_minimal_sets(candidates)
    return set(transversals)


class RBD:
    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        k: Optional[dict[Any, int]] = None,
        nodes: Optional[Iterable] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
    ):
        """Creates and returns a Reliability Block Diagram object.

        Parameters
        ----------
        edges : Iterable[tuple[Hashable, Hashable]]
            The collection of node edges, e.g. [(1, 2), (2, 3)] would
            correspond to the edges 1-2 and 2-3
        k : dict[Any, int]
            A dictionary mapping nodes to k-out-of-n (koon) values, by default
            {}, by default all nodes koon values are 1
        on_infeasible_rbd : {{'raise', 'warn', 'ignore'}}, default 'raise'
            Specifies what to do upon encountering a bad line (a line with too
            many fields). Allowed values are :
                - 'raise', raise an Exception when an infeasible RBD is
                detected.
                - 'warn', raise a warning when an infeasible RBD is detected,
                but return RBD anyway.
                - 'ignore', return the RBD without raising any warnings.

        Raises
        ------
        ValueError
            A node is not in the node list or edge list
        """

        # Create RBD graph
        self.G = RBDGraph()
        self.G.add_edges_from(edges)
        if nodes is not None:
            self.G.add_nodes_from(nodes)

        if input_node is not None:
            if input_node not in self.G.nodes:
                raise ValueError("'input_node' not in RBD structure.")
            else:
                self.input_node = input_node

        if output_node is not None:
            if output_node not in self.G.nodes:
                raise ValueError("'output_node' not in RBD structure.")
            else:
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
        combined_nodes: set = set().union(*self.get_min_path_sets())
        return set(self.G.nodes).symmetric_difference(combined_nodes)

    def get_all_path_sets(self) -> Iterator[list[Hashable]]:
        """Gets all path sets from input_node to output_node

        Really just wraps networkx.all_simple_paths(). This is an expensive
        operation, so be careful using for very large RBDs.

        Returns
        -------
        Iterator[list[Hashable]]
            The iterator of paths
        """
        return nx.all_simple_paths(
            self.G, source=self.input_node, target=self.output_node
        )

    def get_min_path_sets(
        self, include_in_out_nodes=True
    ) -> set[frozenset[Hashable]]:
        """Gets the minimal path-sets of the RBD

        Parameters
        ----------
        include_in_out_nodes : bool, optional
            If false, excludes the input and output nodes
            in the return, by default True

        Returns
        -------
        set[frozenset[Hashable]]
            The set of minimal path-sets
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
        """Returns a boolean as to whether the system is working given the
        status of the components

        Parameters
        ----------
        component_status : dict[Any, bool]
            Dictionary with all components where
            component_status[component] = True only if the component is
            working, and = False if not working.
        method : str
            Either "p" (path-set) or "c" (cut-set). Both return the same
            result; "p" is typically faster as it avoids deriving the cut
            sets.

        Returns
        -------
        bool
            True if the system is working, otherwise False.
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
            return any(
                all(component_status[component] for component in path_set)
                for path_set in self._eval_path_sets
            )
        elif method == "c":
            if not hasattr(self, "_eval_cut_sets"):
                self._eval_cut_sets = [
                    tuple(cut_set)
                    for cut_set in self.get_min_cut_sets(
                        include_in_out_nodes=False
                    )
                ]
            return all(
                any(component_status[component] for component in cut_set)
                for cut_set in self._eval_cut_sets
            )
        else:
            raise ValueError("`method` must be either 'p' or 'c'")

    def get_min_cut_sets(
        self, include_in_out_nodes=False
    ) -> set[frozenset[Hashable]]:
        """
        Returns the set of frozensets of minimal cut sets of the RBD. The outer
        set contains the frozenset of nodes. frozensets were used so the inner
        set elements could be hashable.

        The minimal cut sets are the minimal transversals (hitting sets) of the
        minimal path sets, computed with Berge's algorithm. See
        minimal_cut_sets_from_path_sets() for details.
        """
        path_sets = self.get_min_path_sets(
            include_in_out_nodes=include_in_out_nodes
        )
        return minimal_cut_sets_from_path_sets(path_sets)

    def path_set_probabilities(self, node_probabilities):
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
        """Returns the system probability/ies given the probability of each
        node.

        Parameters
        ----------
        node_probabilities: Dict
            Dictionary containing the probabilities of the event for every node
            in the RBDGraph. Probability is to be either the reliability or the
            availability (or some other probability that I can't conceive).
        method: str, optional
            Input either "c" or "p" for the function to use the cut set or
            path set methods respectively, by default "p". Both methods
            return the same (exact) result; the path set method is the default
            as it avoids deriving the cut sets.

        Returns
        -------
        np.ndarray
            Probability values for all events in nodes_probabilities

        Raises
        ------
        ValueError
            Probability arrays of differing lengths
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
            path_sets = self.get_min_path_sets(include_in_out_nodes=False)
            return probability_any_set_satisfied(
                path_sets, node_probabilities, array_shape
            )

        # method == "c": work with cut sets and node unreliabilities. The
        # system unreliability is the probability that at least one minimal
        # cut set has all of its components failed.
        cut_sets = self.get_min_cut_sets(include_in_out_nodes=False)
        node_unreliability = {k: 1 - v for k, v in node_probabilities.items()}
        system_unreliability = probability_any_set_satisfied(
            cut_sets, node_unreliability, array_shape
        )
        return 1 - system_unreliability

    @check_probability
    def improvement_allocation(
        self,
        target: float,
        node_probabilities: Dict,
        fixed: Optional[list] = None,
        weights=None,
    ):
        if fixed is None:
            fixed = []
        node_probabilities = copy(node_probabilities)

        for n, v in node_probabilities.items():
            node_probabilities[n] = np.atleast_1d(v)

        for n in self.nodes:
            if n not in node_probabilities:
                node_probabilities[n] = np.atleast_1d(0.5)

        # for node in self.in_or_out:
        # node_probabilities[node] = np.atleast_1d(1.0)

        def func(x):
            scaled_probabilities = scale_probability_dict(
                {
                    k: v
                    for k, v in node_probabilities.items()
                    if k not in fixed
                },
                x,
                weights,
            )
            scaled_probabilities = {
                **node_probabilities,
                **scaled_probabilities,
            }

            current = self.system_probability(scaled_probabilities, method="p")
            return current - target

        # Using root
        res = root(func, 1.0, tol=1e-10, method="lm")
        self.res = res
        out = scale_probability_dict(
            {k: v for k, v in node_probabilities.items() if k not in fixed},
            res["x"].item(),
            weights,
        )
        out = {**node_probabilities, **out}
        out = {k: v.item() for k, v in out.items()}
        return out

    @check_probability
    def equal_allocation(self, target: float):
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

    def get_nodes_names(self) -> list[Hashable]:
        """Simply returns the list component names of the RBD."""
        return list(self.nodes)

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
