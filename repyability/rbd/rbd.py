from collections import defaultdict
from copy import copy
from itertools import combinations
from typing import Any, Collection, Hashable, Iterable, Iterator
from warnings import warn

import networkx as nx
import numpy as np
import surpyval as surv
from numpy.typing import ArrayLike

from repyability.rbd.min_path_sets import min_path_sets
from repyability.rbd.rbd_graph import RBDGraph

from .helper_classes import PerfectReliability, PerfectUnreliability
from .min_cut_sets import min_cut_sets
from .rbd_args_check import check_rbd_node_args_complete


class RBD:
    def __init__(
        self,
        nodes: dict[Any, Any],
        reliability: dict[Any, Any],
        edges: Iterable[tuple[Hashable, Hashable]],
        k: dict[Any, int] = {},
        mc_samples: int = 10_000,
    ):
        """Creates and returns a Reliability Block Diagram object.

        Parameters
        ----------
        nodes : dict[Any, Any]
            A dictionary of node names as keys and their respective component
            names as values (which map to the components in the reliability
            dict), except for the the input and output nodes which need string
            values `"input_node"` and `"output_node"` respectively
        reliability : dict[Any, Any]
            A dictionary of all non-input-output components names as keys
            with their SurPyval reliability distributions as values
        edges : Iterable[tuple[Hashable, Hashable]]
            The collection of node edges, e.g. [(1, 2), (2, 3)] would
            correspond to the edges 1-2 and 2-3
        components : dict[Any, Any]
            A dictionary of all non-input-output components names as keys
            with their SurPyval distribution as values
        mc_samples : int, optional
            TODO, by default 10_000

        Raises
        ------
        ValueError
            A node is not in the node list or edge list
        """

        # Check args are complete, will raise ValueError if not
        check_rbd_node_args_complete(nodes, reliability, edges)

        # Create RBD graph
        G = RBDGraph()
        G.add_edges_from(edges)
        self.G = G

        # Set the node k values (k-out-of-n)
        for node, k_val in k.items():
            G.nodes[node]["k"] = k_val

        # Copy the components and nodes
        reliability = copy(reliability)
        nodes = copy(nodes)

        # Look through all the nodes.
        visited_nodes = set()
        for node in nodes.keys():
            visited_nodes.add(node)

            # Set node attribute dict types if input/output
            # (if neither input/output no need to do anything, RBDGraph
            # defaults the type to "node")
            if nodes[node] == "input_node":
                self.input_node = node
                self.G.nodes[node]["type"] = "input_node"
                reliability[node] = PerfectReliability
            elif nodes[node] == "output_node":
                self.output_node = node
                self.G.nodes[node]["type"] = "output_node"
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

        # Finally, check valid RBD structure
        if not self.is_valid_RBD_structure():
            raise ValueError(
                f"RBD not correctly structured, add edges or nodes \
                to create correct structure. See errors: \
                {self.rbd_structural_errors}"
            )

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
        using: str = "p",
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
        using: str, optional
            Input either "c" or "p" for the function to use cut sets or path
            sets respectively. Defaults to path sets.

        Returns
        -------
        np.ndarray
            Reliability values for all nodes at all times x

        Raises
        ------
        ValueError
            - Working/broken node/component inconsistency (a component or node
              is supplied more than once to any of working_nodes, broken_nodes,
              working_components, broken_components)
        """
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
        # TODO: Create cut set logic
        # Get all path sets
        paths = self.get_min_path_sets()
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

    def is_valid_RBD_structure(self) -> bool:
        """Returns False if invalid RBD structure

        Invalid RBD structure includes:
        - having cycles present
        - a non-input/output node having no in/out-nodes

        Returns
        -------
        bool
            True
        """
        has_circular_dependency = not nx.is_directed_acyclic_graph(self.G)
        node_degrees: dict = defaultdict(lambda: defaultdict(int))

        for edge in self.G.edges:
            source, target = edge
            node_degrees[source]["out"] += 1
            node_degrees[target]["in"] += 1

        input_nodes = [n for n in node_degrees.values() if n["in"] == 0]
        output_nodes = [n for n in node_degrees.values() if n["out"] == 0]
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

    def get_component_names(self) -> list[Hashable]:
        """Simply returns the list component names of the RBD."""
        return list(self.reliability.keys())
