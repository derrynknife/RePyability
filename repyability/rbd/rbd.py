from collections import defaultdict
from copy import copy
from itertools import combinations
from typing import Any, Hashable, Iterable
from warnings import warn

import networkx as nx
import numpy as np
import surpyval as surv

from .helper_classes import PerfectReliability, PerfectUnreliability


class RBD:
    # TODO: Implement these
    # Finding cut-sets:
    # https://www.degruyter.com/document/doi/10.1515/9783110725599-007/html?lang=en

    def __init__(
        self,
        nodes: dict[Hashable, Hashable],
        components: dict[Hashable, Any],
        edges: Iterable[tuple[Hashable, Hashable]],
        mc_samples: int = 10_000,
    ):
        """Creates and returns a Reliability Block Diagram object.

        Parameters
        ----------
        nodes : dict[Hashable, Hashable]
            A dictionary of node names as keys and their respective component
            names as values (which map to the components in the components
            dict), except for the the input and output nodes which need string
            values `"input_node"` and `"output_node"` respectively
        components : dict[Hashable, Any]
            A dictionary of all non-input-output components names as keys
            with their SurPyval distribution as values
        edges : Iterable[tuple[Hashable, Hashable]]
            The collection of node edges, e.g. [[1, 2], [2, 3]] would
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
        components = copy(components)
        nodes = copy(nodes)

        # Look through all the nodes.
        visited_nodes = set()
        for node in nodes.keys():
            if not G.has_node(node):
                raise ValueError("Node {} not in edge list".format(node))
            visited_nodes.add(node)
            if nodes[node] in ["input_node", "output_node"]:
                setattr(self, nodes[node], node)
                components[node] = PerfectReliability

        nodes.pop(self.input_node)
        nodes.pop(self.output_node)
        self.in_or_out = [self.input_node, self.output_node]

        # Create a components to nodes dictionary for efficient sf() lookup
        self.components_to_nodes: dict[Hashable, set] = defaultdict(set)
        for node, component in nodes.items():
            self.components_to_nodes[component].add(node)

        # Check that all nodes in graph were in the nodes list.
        for n in G.nodes:
            if n not in visited_nodes:
                raise ValueError("Node {} not in nodes list".format(n))

        new_models = {}
        for k, v in components.items():
            if type(v) == list:
                sim = 0
                for model in v:
                    sim += model.random(mc_samples)

                new_models[k] = surv.KaplanMeier.fit(sim)

        # This will override the existing list with Non-Parametric
        # models
        components = {**components, **new_models}

        self.components = components
        self.nodes = nodes

    def all_path_sets(self):
        # For a very large RBD, this seems expensive; O(n!).....
        # Need to convert to a fault tree using graph algs
        return nx.all_simple_paths(
            self.G, source=self.input_node, target=self.output_node
        )

    def sf(
        self,
        x: int | float | Iterable[int | float],
        working_nodes: Iterable[Hashable] = [],
        broken_nodes: Iterable[Hashable] = [],
        working_components: Iterable[Hashable] = [],
        broken_components: Iterable[Hashable] = [],
    ) -> np.ndarray:
        """Returns the system reliability for time/s x.

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable
        working_nodes : Iterable[Hashable], optional
            Marks these nodes as perfectly reliable, by default []
        broken_nodes : Iterable[Hashable], optional
            Marks these nodes as perfectly unreliable, by default []
        working_components : Iterable[Hashable], optional
            Marks these components as perfectly reliable, by default []
        broken_components : Iterable[Hashable], optional
            Marks these components as perfectly unreliable, by default []

        Returns
        -------
        np.ndarray
            Reliability values for all nodes at all times x

        Raises
        ------
        ValueError
            RBD not correctly structured
        """
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

        if not self.check_rbd_structure():
            raise ValueError(
                "RBD not correctly structured, add edges or nodes \
                to create correct structure."
            )

        x = np.atleast_1d(x)

        # Get all path sets
        paths = list(self.all_path_sets())
        num_paths = len(paths)

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
                        if node not in self.in_or_out:
                            s.add(self.nodes[node])  # Add component name

                # Now calculate the tieset reliability
                tieset_rel = np.ones_like(x)
                for comp in s:
                    # If component is in working_components, or if any of its
                    # nodes are in working_nodes, then make PerfectReliability
                    if (
                        comp in working_components
                        or not self.components_to_nodes[comp].isdisjoint(
                            working_nodes
                        )
                    ):
                        comp_rel = PerfectReliability.sf(x)
                    elif (
                        # If component is in broken_components or any of its
                        # nodes are in broken_nodes, then make
                        # PerfectUnreliability
                        comp in broken_components
                        or not self.components_to_nodes[comp].isdisjoint(
                            broken_nodes
                        )
                    ):
                        comp_rel = PerfectUnreliability.sf(x)
                    else:
                        comp_rel = self.components[comp].sf(x)
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

    def ff(self, x, *args, **kwargs):
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
    def birnbaum_importance(
        self, x: int | float | Iterable[int | float]
    ) -> dict[Hashable, float]:
        """Returns the Birnbaum measure of importance for all nodes.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent. If the RBD called on has two or more nodes associated
        with the same component then a UserWarning is raised.

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable

        Returns
        -------
        dict[Hashable, float]
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
    def improvement_potential(
        self, x: int | float | Iterable[int | float]
    ) -> dict[Hashable, float]:
        """Returns the improvement potential of all nodes.

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable

        Returns
        -------
        dict[Hashable, float]
            Dictionary with node names as keys and improvement potentials as
            values
        """
        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            as_is = self.sf(x)
            node_importance[node] = working - as_is
        return node_importance

    def risk_achievement_worth(
        self, x: int | float | Iterable[int | float]
    ) -> dict[Hashable, float]:
        """Returns the RAW importance per Modarres & Kaminskiy. That is RAW_i =
        (unreliability of system given i failed) /
        (nominal system unreliability).

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable

        Returns
        -------
        dict[Hashable, float]
            Dictionary with node names as keys and RAW importances as values
        """
        node_importance = {}
        system_ff = self.ff(x)
        for node in self.nodes.keys():
            failing = self.ff(x, broken_nodes=[node])
            node_importance[node] = failing / system_ff
        return node_importance

    def risk_reduction_worth(
        self, x: int | float | Iterable[int | float]
    ) -> dict[Hashable, float]:
        """Returns the RRW importance per Modarres & Kaminskiy. That is RRW_i =
        (nominal unreliability of system) /
        (unreliability of system given i is working).

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable

        Returns
        -------
        dict[Hashable, float]
            Dictionary with node names as keys and RRW importances as values
        """
        node_importance = {}
        system_ff = self.ff(x)
        print(f"system_ff = {system_ff}")
        for node in self.nodes.keys():
            working = self.ff(x, working_nodes=[node])
            print(f"node {node} when working has system ff = {working}")
            node_importance[node] = system_ff / working
        return node_importance

    def criticality_importance(
        self, x: int | float | Iterable[int | float]
    ) -> dict[Hashable, float]:
        """Returns the criticality imporatnce of all nodes at time/s x.

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable

        Returns
        -------
        dict[Hashable, float]
            Dictionary with node names as keys and criticality importances as
            values
        """
        bi = self.birnbaum_importance(x)
        node_importance = {}
        system_sf = self.sf(x)
        for node in self.nodes.keys():
            node_sf = self.components[self.nodes[node]].sf(x)
            node_importance[node] = bi[node] * node_sf / system_sf
        return node_importance

    def fussel_vessely(self, x: int | float, fv_type: str = "p"):
        """
        Calculate Fussel-Vesely Importance of all components at time/s x.

        fv_type dictates the method of calculation:
            "p" - path set
            "c" - cut set
        """
        if fv_type not in ["c", "p"]:
            raise ValueError(
                "'fv_type' must be either c (cut set) or p (path set)"
            )

        # TODO: Implement cut set based FV importance.
        if fv_type == "c":
            raise NotImplementedError(
                "cut set type FV importance measure not yet implemented"
            )

        system_reliability = self.sf(x)

        paths = list(self.all_path_sets())
        node_importance = {}

        x = np.atleast_1d(x)
        r_dict = {}
        for component in self.components.keys():
            # Calculating reliability in the log-domain though so the
            # components' reliability can be added avoid possible underflow
            r_dict[component] = np.log(self.components[component].sf(x))

        for node in self.nodes.keys():
            paths_reliability = []
            for path in paths:
                if node in self.in_or_out:
                    continue
                elif node not in path:
                    continue
                path_rel = np.zeros_like(x).astype(float)
                for n in path:
                    if n in self.in_or_out:
                        continue
                    path_rel += r_dict[self.nodes[n]]
                paths_reliability.append(path_rel)

            paths_sf = np.atleast_2d(paths_reliability)
            paths_sf = 1 - np.exp(np.sum(paths_sf, axis=0))
            paths_sf = paths_sf / system_reliability
            node_importance[node] = np.copy(paths_sf)

        return node_importance
