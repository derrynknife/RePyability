from collections import defaultdict
from copy import copy
from itertools import combinations, product
from typing import Any, Dict, Hashable, Iterable, Iterator, Optional

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, root

from repyability.rbd.min_path_sets import min_path_sets
from repyability.rbd.rbd_graph import RBDGraph


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_inv(x):
    return np.log(x / (1.0 - x))


def check_probability(func):
    """checks probability is between 0 and 1"""

    def wrap(obj, target: float, *args, **kwargs):
        if target > 1:
            raise ValueError("target cannot be above 1.")
        elif target < 0:
            raise ValueError("target cannot be below 0.")
        else:
            return func(obj, target, *args, **kwargs)

    return wrap


def log_linearly_scale_probabilities(p, x):
    if p == 1.0:
        return np.atleast_1d(1.0)
    else:
        return np.atleast_1d(1 - np.exp(-(-np.log(1 - p) + x)))


def scale_probability_dict(node_probabilities, x, weights=None):
    out = {}
    if weights is None:
        weights = defaultdict(lambda: 1.0)
    for k, p in node_probabilities.items():
        out[k] = log_linearly_scale_probabilities(p, x * weights[k])
    return out


class RBD:
    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        k: Optional[dict[Any, int]] = {},
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

        Raises
        ------
        ValueError
            A node is not in the node list or edge list
        """

        # Create RBD graph
        G = RBDGraph()
        G.add_edges_from(edges)
        self.G = G

        # Finally, check valid RBD structure
        if not self.G.is_valid_RBD_structure():
            raise ValueError(
                "RBD not correctly structured. Errors: \n"
                + f"{self.G.rbd_structural_errors}"
            )

        # Set the input and output node references
        for node in self.G.nodes:
            if self.G.out_degree(node) == 0:
                self.output_node = node
            elif self.G.in_degree(node) == 0:
                self.input_node = node

        self.in_or_out = [self.input_node, self.output_node]
        self.nodes = [n for n in self.G.nodes if n not in self.in_or_out]

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
        ret_set = set()

        for min_path_set in min_path_sets(
            rbd_graph=self.G,
            curr_node=self.output_node,
            solns={},
        ):
            if not include_in_out_nodes:
                min_path_set.remove(self.input_node)
                min_path_set.remove(self.output_node)
            ret_set.add(frozenset(min_path_set))

        return ret_set

    def get_min_cut_sets(
        self, include_in_out_nodes=False
    ) -> set[frozenset[Hashable]]:
        """
        Returns the set of frozensets of minimal cut sets of the RBD. The outer
        set contains the frozenset of nodes. frozensets were used so the inner
        set elements could be hashable.
        """
        path_sets = self.get_min_path_sets(
            include_in_out_nodes=include_in_out_nodes
        )

        # Gets the cartesian product across pathsets
        prods = product(*path_sets)

        # We need to remove duplicate nodes in the products to get the cutsets,
        # and discard empty products
        cut_sets = [frozenset(prod) for prod in prods if prod]

        min_cut_sets: list[frozenset] = []

        # Now only insert if minimal, removing any superset (non-minimal)
        # cutsets are encountered
        for cut_set in cut_sets:
            is_minimal_cut_set = True
            for other_cut_set in min_cut_sets.copy():
                if cut_set.issuperset(other_cut_set):
                    is_minimal_cut_set = False
                    break
                if cut_set.issubset(other_cut_set):
                    min_cut_sets.remove(other_cut_set)

            if is_minimal_cut_set:
                min_cut_sets.append(cut_set)

        return set(min_cut_sets)

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
        method: str = "c",
        approx: bool = False,
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
            path set methods respectively, by default "c". Both methods
            ultimately return the same results though.
        approx: bool, optional
            If true, only considers the first-order terms (w.r.t. the
            inclusion-exclusion principle), thereby reducing computation time.
            This approximation is only applicable to the cut set method
            (method="c"), a ValueError exception is raised if method="p" and
            approx=True. This approximation is typically sufficient for most
            use cases where reliabilities are close to 1. By default, False.

        Returns
        -------
        np.ndarray
            Probability values for all events in nodes_probabilities

        Raises
        ------
        ValueError
            - Working/broken node/component inconsistency (a component or node
              is supplied more than once to any of working_nodes, broken_nodes,
              working_components, broken_components)
            - The path set method must not be used with approx=True, see approx
              arg description above
        """

        node_probabilities = copy(node_probabilities)
        lengths = np.array([])
        for node in self.nodes:
            node_array = np.array(node_probabilities[node])
            node_probabilities[node] = node_array
            lengths = np.append(lengths, node_array.shape)
        if np.any(lengths[0] != lengths[1:]):
            raise ValueError("Probability arrays must be same length")
        else:
            # get shape of input array
            array_shape = lengths[0]
        # Return array
        system_prob = np.zeros(array_shape)

        # Check that path set method and approximation are not used together
        # (The approximation is only applicable to the cutset method)
        if method == "p" and approx:
            raise ValueError(
                "The path set method must not be used with \
                approx=True, see approx arg description in docstring."
            )
        # Get all node sets (path sets for method="p" and cut sets for
        # method="c")
        node_sets: set[frozenset]
        if method == "p":
            node_sets = self.get_min_path_sets(include_in_out_nodes=False)
        else:
            # method == "c"
            node_sets = self.get_min_cut_sets(include_in_out_nodes=False)
            node_probabilities = {
                k: 1 - v for k, v in node_probabilities.items()
            }

        num_node_sets = len(node_sets)
        # Perform intersection calculation, which isn't as simple as summating
        # in the case of mutual non-exclusivity
        # i is the 'level' of the intersection calc
        # This is really just applying the inclusion-exclusion principle
        for i in range(1, num_node_sets + 1):
            # Get node set combinations for level i
            node_set_combs = combinations(node_sets, i)

            # Calculate the probability of each level combination
            # Making sure to not multiply a components' probability twice
            level_sum = np.zeros(array_shape)
            for node_set_comb in node_set_combs:
                # Make a set of components out of the node path/tieset
                s = set()
                for path in node_set_comb:
                    for node in path:
                        if node in self.in_or_out:
                            continue
                        else:
                            # Add node to combination list
                            s.add(node)

                # Now calculate the node set probability
                node_set_prob = np.ones(array_shape)
                for comp in s:
                    comp_prob = node_probabilities[comp]
                    node_set_prob = node_set_prob * comp_prob

                # Now add the node set probability to the level sum
                level_sum = level_sum + node_set_prob

            # Finally add/subtract the level sum to/from the system_prob if the
            # level is even/odd
            if i % 2 == 1:
                system_prob = system_prob + level_sum

                # If the approximation is requested, just break from the
                # inclusion-exclusion principle procedure after the first level
                if approx:
                    break
            else:
                system_prob = system_prob - level_sum

        # If cutset method is used, the above returns the unreliability,
        # so we just have to return = 1 - system_prob
        if method == "c":
            return 1 - system_prob

        # Otherwise for the pathset method it's already the reliability
        return system_prob

    @check_probability
    def improvement_allocation(
        self,
        target: float,
        node_probabilities: Dict,
        fixed: list = [],
        weights=None,
    ):
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
                    1 * node_probabilities_array[node_array_indices[node]]
                )
                for node in self.nodes
            }
            system_probability = self.system_probability(node_probabilities)
            system_loss = target - system_probability
            return system_loss**2

        res = minimize(func, np.zeros(len(self.nodes)), tol=1e-20)

        node_probabilities = {
            node: sigmoid(1 * res["x"][node_array_indices[node]])
            for node in self.nodes
        }

        self.res = res
        return node_probabilities

    def get_nodes_names(self) -> list[Hashable]:
        """Simply returns the list component names of the RBD."""
        return list(self.nodes)

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

    def _fussel_vesely(
        self,
        node_probabilities: dict[Any, ArrayLike],
        fv_type: str = "c",
        approx: bool = True,
    ) -> dict[Any, np.ndarray]:
        """Calculate Fussel-Vesely importance of all components at time/s x.

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
            Dictionary with node names as keys and fussel-vessely importances
            as values

        Raises
        ------
        ValueError
            TODO
        NotImplementedError
            TODO
        """
        node_probabilities_new: dict[Any, np.ndarray] = {}
        lengths = np.array([])
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
