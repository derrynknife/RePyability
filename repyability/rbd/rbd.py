from collections import defaultdict
from copy import copy
from itertools import combinations
from typing import Any, Collection, Hashable, Iterable, Iterator
from warnings import warn

import networkx as nx
import numpy as np
import surpyval as surv
from numpy.typing import ArrayLike

from .helper_classes import PerfectReliability, PerfectUnreliability
from .min_cut_sets import min_cut_sets


class RBD:
    # TODO: Implement these
    # Finding cut-sets:
    # https://www.degruyter.com/document/doi/10.1515/9783110725599-007/html?lang=en

    def __init__(
        self,
        nodes: dict[Any, Any],
        reliability: dict[Any, Any],
        edges: Iterable[tuple[Hashable, Hashable]],
        mc_samples: int = 10_000,
    ):
        """Creates and returns a Reliability Block Diagram object.

        Parameters
        ----------
        nodes : dict[Any, Any]
            A dictionary of node names as keys and their component
            names as values (which map to the keys in the reliability
            dict), except for the the input and output nodes which need string
            values `"input_node"` and `"output_node"` respectively
        reliability : dict[Any, Any]
            A dictionary of all non-input-output components names as keys
            with their SurPyval reliability distributions as values
        edges : Iterable[tuple[Hashable, Hashable]]
            The collection of node edges, e.g. [(1, 2), (2, 3)] would
            correspond to the edges 1-2 and 2-3
        mc_samples : int, optional
            TODO, by default 10_000

        Raises
        ------
        ValueError
            A node is not in the node list or edge list
        """

        # Create RBD graph
        G = nx.DiGraph()
        G.add_edges_from(edges)
        self.G = G

        # Copy the components and nodes
        reliability = copy(reliability)
        nodes = copy(nodes)

        # Look through all the nodes.
        visited_nodes = set()
        for node in nodes.keys():
            if not G.has_node(node):
                raise ValueError("Node {} not in edge list".format(node))
            visited_nodes.add(node)
            if nodes[node] == "input_node":
                self.input_node = node
                reliability[node] = PerfectReliability
            elif nodes[node] == "output_node":
                self.output_node = node
                reliability[node] = PerfectReliability

        nodes.pop(self.input_node)
        nodes.pop(self.output_node)
        self.in_or_out = [self.input_node, self.output_node]

        # Create a components to nodes dictionary for efficient sf() lookup
        self.components_to_nodes: dict[Any, set] = defaultdict(set)
        for node, component in nodes.items():
            self.components_to_nodes[component].add(node)

        # Check that all nodes in graph were in the nodes list.
        for n in G.nodes:
            if n not in visited_nodes:
                raise ValueError("Node {} not in nodes list".format(n))

        new_models = {}
        for k, v in reliability.items():
            if type(v) == list:
                sim = 0
                for model in v:
                    sim += model.random(mc_samples)

                new_models[k] = surv.KaplanMeier.fit(sim)

        # This will override the existing list with Non-Parametric
        # models
        reliability = {**reliability, **new_models}

        self.reliability = reliability
        self.nodes = nodes

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
    ) -> set[tuple[Hashable, ...]]:
        """Gets the minimal path-sets of the RBD

        Parameters
        ----------
        include_in_out_nodes : bool, optional
            If false, excludes the input and output nodes
            in the path tuples, by default True

        Returns
        -------
        set[tuple[Hashable, ...]]
            The set of minimal path-sets
        """
        # What differentiates all path sets and minimal path sets is
        # minimal path sets cannot be further reduced by removing components
        # from path. Recall a path set is a set of components (really a path)
        # which if working ensure the system is working.

        # So (brute-force) strategy could be just get all the simple paths from
        # get_all_path_sets() that cannot be reduced.

        # Get all path sets as a list
        all_path_sets = list(self.get_all_path_sets())

        # A path can be reduced when the system is still working after removing
        # any node on that path.
        # Ultimately, every non-minimal path set has a subset that is a minimal
        # path set, so we can just check for every path set if it has a subset,
        # if it doesn't then it is a minimal path-set.

        ret_set: set[tuple[Hashable, ...]] = set()

        # We're not done until all_path_sets is completely empty
        # since every iteration we're either finding a path-set to be minimal,
        # thereby adding it to ret_set and removing it from all_path_sets,
        # OR we're finding the path-set to be non-minimal and removing it
        # from all_path_sets
        while all_path_sets:
            # Get first path-set in all_path_sets
            path_set = all_path_sets[0]

            # Assume it is a minimal path-set
            is_minimal_path_set = True

            # Compare to all other path-sets
            for other_path_set in all_path_sets.copy()[1:]:
                # If path_set is a subset of other_path_set then other_path_set
                # is not a minimal path-set, so we can remove it from
                # all_path_sets and prevent any further consideration of it
                if set(path_set).issubset(set(other_path_set)):
                    all_path_sets.remove(other_path_set)

                # If path_set is a superset of other_path_set then path_set
                # is not a minimal path-set so we can remove it from
                # all_path_sets and move on to the next iteration
                elif set(path_set).issuperset(set(other_path_set)):
                    all_path_sets.remove(path_set)
                    is_minimal_path_set = False
                    break

            # If is_minimal_path_set is still True then we can add path_set
            # to ret_set, and remove it from path_set
            if is_minimal_path_set:
                # If include_in_out_nodes is set to false, remove the input
                # and output nodes
                if not include_in_out_nodes:
                    path_set.remove(self.input_node)
                    path_set.remove(self.output_node)

                # Finally add the path_set as a tuple to the return set
                ret_set.add(tuple(path_set))
                all_path_sets.remove(path_set)

        return ret_set

    def get_min_cut_sets(self) -> set[frozenset[Hashable]]:
        """
        Returns the set of frozensets of minimal cut sets of the RBD. The outer
        set contains the frozenset of nodes. frozensets were used so the inner
        set elements could be hashable.
        """
        return min_cut_sets(self.G, self.input_node, self.output_node)

    def sf(
        self,
        x: ArrayLike,
        working_nodes: Collection[Hashable] = [],
        broken_nodes: Collection[Hashable] = [],
        working_components: Collection[Hashable] = [],
        broken_components: Collection[Hashable] = [],
    ) -> np.ndarray:
        """Returns the system reliability for time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Marks these nodes as perfectly reliable, by default []
        broken_nodes : Collection[Hashable], optional
            Marks these nodes as perfectly unreliable, by default []
        working_components : Collection[Hashable], optional
            Marks these components as perfectly reliable, by default []
        broken_components : Collection[Hashable], optional
            Marks these components as perfectly unreliable, by default []

        Returns
        -------
        np.ndarray
            Reliability values for all nodes at all times x

        Raises
        ------
        ValueError
            - RBD not correctly structured
            - Working/broken node/component inconsistency (a component or node
              is supplied more than once to any of working_nodes, broken_nodes,
              working_components, broken_components)
        """

        if not self.check_rbd_structure():
            raise ValueError(
                "RBD not correctly structured, add edges or nodes \
                to create correct structure."
            )

        # Check for any node/component argument inconsistency
        argument_nodes: set[object] = set()
        argument_nodes.update(working_nodes)
        argument_nodes.update(broken_nodes)
        number_of_arg_comp_nodes = 0
        for comp in working_components:
            argument_nodes.update(self.components_to_nodes[comp])
            number_of_arg_comp_nodes += len(self.components_to_nodes[comp])
        for comp in broken_components:
            argument_nodes.update(self.components_to_nodes[comp])
            number_of_arg_comp_nodes += len(self.components_to_nodes[comp])
        if len(argument_nodes) != (
            len(working_nodes) + len(broken_nodes) + number_of_arg_comp_nodes
        ):
            working_comps_nodes = [
                (comp, self.components_to_nodes[comp])
                for comp in working_components
            ]
            broken_comps_nodes = [
                (comp, self.components_to_nodes[comp])
                for comp in broken_components
            ]

            raise ValueError(
                f"Node/component inconsistency provided. i.e. you have \
                provided sf() with working/broken nodes (or respective \
                components) more than once in the arguments.\n \
                Supplied:\n\
                working_nodes: {working_nodes}\n\
                broken_nodes: {broken_nodes}\n\
                (working_components, their_nodes): {working_comps_nodes}\n\
                (broken_components, their_nodes): {broken_comps_nodes}\n"
            )

        # Turn node iterables into sets for O(1) lookup later
        working_nodes = set(working_nodes)
        broken_nodes = set(broken_nodes)

        # Per Note 3 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS"
        # (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf),
        # you should:
        # - Apply the cut set method with only first order terms if
        #   marginal error is tolerable, and components have high reliability
        #   ?TODO: test whether indeed quicker?
        # - Or if you want to use all terms, then apply either the tie set or
        #   cut set method depending on which has less sets for the system,
        #   and thus less calculation
        #   (typically cut sets)
        #
        # At the moment, only the tie set method is implemented.

        x = np.atleast_1d(x)

        # Get all path sets
        paths = list(self.get_all_path_sets())
        num_paths = len(paths)

        # Cache all component reliabilities for efficiency
        comp_rel_cache_dict: dict[Hashable, np.ndarray] = {}
        for comp in self.reliability:
            if comp in working_components:
                comp_rel_cache_dict[comp] = PerfectReliability().sf(x)
            elif comp in broken_components:
                comp_rel_cache_dict[comp] = PerfectUnreliability().sf(x)
            else:
                comp_rel_cache_dict[comp] = self.reliability[comp].sf(x)
        # We'll just add the two 'perfect component reliabilities' to this dict
        # while we're at it, since they'll be used in lookup later
        comp_rel_cache_dict["PerfectReliability"] = PerfectReliability().sf(x)
        comp_rel_cache_dict[
            "PerfectUnreliability"
        ] = PerfectUnreliability().sf(x)

        # Perform intersection calculation, which isn't as simple as summating
        # in the case of mutual non-exclusivity
        # i is the 'level' of the intersection calc
        system_rel = np.zeros_like(x)  # Return array
        for i in range(1, num_paths + 1):
            # Get tie-set combinations for level i
            tieset_combs = combinations(paths, i)

            # Calculate the reliability of each level combination
            # Making sure to not multiply a components' reliability twice
            level_sum = np.zeros_like(x)
            for tieset_comb in tieset_combs:
                # Make a set of components out of the node path/tieset
                s = set()
                for path in tieset_comb:
                    for node in path:
                        if node in self.in_or_out:
                            continue
                        # Node working/broken takes precedence over
                        # the components reliability
                        if node in working_nodes:
                            s.add("PerfectReliability")
                        elif node in broken_nodes:
                            s.add("PerfectUnreliability")
                        else:
                            s.add(self.nodes[node])  # Add component name

                # Now calculate the tieset reliability
                tieset_rel = np.ones_like(x)
                for comp in s:
                    comp_rel = comp_rel_cache_dict[comp]
                    tieset_rel = tieset_rel * comp_rel

                # Now add the tieset reliability to the level sum
                level_sum = level_sum + tieset_rel

            # Finally add/subtract the level sum to/from the system_rel if the
            # level is even/odd
            if i % 2 == 1:
                system_rel = system_rel + level_sum
            else:
                system_rel = system_rel - level_sum

        return system_rel

    def ff(self, x: ArrayLike, *args, **kwargs) -> np.ndarray:
        """Returns the system unreliability for time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        *args, **kwargs :
            Any sf() arguments

        Returns
        -------
        np.ndarray
            Uneliability values for all nodes at all times x
        """
        return 1 - self.sf(x, *args, **kwargs)

    def check_rbd_structure(self):
        has_circular_dependency = not nx.is_directed_acyclic_graph(self.G)
        node_degrees = defaultdict(lambda: defaultdict(int))

        for edge in self.G.edges:
            source, target = edge
            node_degrees[source]["out"] += 1
            node_degrees[target]["in"] += 1

        input_nodes = [n for k, n in node_degrees.items() if n["in"] == 0]
        output_nodes = [n for k, n in node_degrees.items() if n["out"] == 0]
        has_node_with_no_input = len(input_nodes) != 1
        has_node_with_no_output = len(output_nodes) != 1
        if not any(
            [
                has_circular_dependency,
                has_node_with_no_input,
                has_node_with_no_output,
            ]
        ):

            self.rbd_structural_errors = None
            return True
        else:
            self.rbd_structural_errors = {
                "has_circular_dependency": has_circular_dependency,
                "has_node_with_no_input": has_node_with_no_input,
                "has_node_with_no_output": has_node_with_no_output,
            }
            return False

    # Importance measures
    # https://www.ntnu.edu/documents/624876/1277590549/chapt05.pdf/82cd565f-fa2f-43e4-a81a-095d95d39272
    def birnbaum_importance(self, x: ArrayLike) -> dict[Any, float]:
        """Returns the Birnbaum measure of importance for all nodes.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent. If the RBD called on has two or more nodes associated
        with the same component then a UserWarning is raised.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Birnbaum importances as
            values
        """
        for component, node_set in self.components_to_nodes.items():
            if len(node_set) > 1:
                warn(
                    f"Birnbaum's measure of importance assumes nodes are \
                     dependent, but nodes {node_set} all depend on the same \
                     component '{component}."
                )

        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            failing = self.sf(x, broken_nodes=[node])
            node_importance[node] = working - failing
        return node_importance

    # TODO: update all importance measures to allow for component as well
    def improvement_potential(self, x: ArrayLike) -> dict[Any, float]:
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
        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            as_is = self.sf(x)
            node_importance[node] = working - as_is
        return node_importance

    def risk_achievement_worth(self, x: ArrayLike) -> dict[Any, float]:
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
        node_importance = {}
        system_ff = self.ff(x)
        for node in self.nodes.keys():
            failing = self.ff(x, broken_nodes=[node])
            node_importance[node] = failing / system_ff
        return node_importance

    def risk_reduction_worth(self, x: ArrayLike) -> dict[Any, float]:
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
        node_importance = {}
        system_ff = self.ff(x)
        for node in self.nodes.keys():
            working = self.ff(x, working_nodes=[node])
            node_importance[node] = system_ff / working
        return node_importance

    def criticality_importance(self, x: ArrayLike) -> dict[Any, float]:
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
        bi = self.birnbaum_importance(x)
        node_importance = {}
        system_sf = self.sf(x)
        for node in self.nodes.keys():
            node_sf = self.reliability[self.nodes[node]].sf(x)
            node_importance[node] = bi[node] * node_sf / system_sf
        return node_importance

    def fussel_vesely(
        self, x: ArrayLike, fv_type: str = "c"
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

        # Ensure time is a numpy array
        x = np.atleast_1d(x)

        # Get system unreliability, this will be the denominator for all node
        # importance calcs
        system_unreliability = self.ff(x)

        # The return dict
        node_importance: dict[Any, np.ndarray] = {}

        # Cache the component reliabilities for efficiency
        rel_dict = {}
        for component in self.reliability.keys():
            # TODO: make log
            # Calculating reliability in the log-domain though so the
            # components' reliability can be added avoid possible underflow
            rel_dict[component] = self.reliability[component].ff(x)

        # For each node,
        for node in self.nodes.keys():
            # Sum up the probabilities of the node_sets containing the node
            # from failing
            node_fv_numerator = 0
            for node_set in node_sets:
                if node not in node_set:
                    continue
                node_set_fail_prob = 1
                # Take only the independent components in that node-set, i.e.
                # don't multiply the same component twice in a node-set
                components_in_node_set = {
                    self.nodes[fail_node] for fail_node in node_set
                }
                for component in components_in_node_set:
                    node_set_fail_prob *= rel_dict[component]
                node_fv_numerator += node_set_fail_prob

            node_importance[node] = node_fv_numerator / system_unreliability

        return node_importance
