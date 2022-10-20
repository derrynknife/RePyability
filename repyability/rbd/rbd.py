from collections import defaultdict
from copy import copy
from itertools import combinations
from typing import Any, Hashable, Iterable

import networkx as nx
import numpy as np
import surpyval as surv

# TODO: Seek advice from Derryn as to why sf() needs the args it has, may need
# to import PerfectUnreliability again
from .helper_classes import PerfectReliability


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
        """Create a Reliability Block Diagram

        Parameters
        ----------
        nodes : dict[Hashable, Hashable]
            A dictionary of component names as keys, the values don't matter
            except for the input and output nodes which need string values
            `"input_node"` and `"output_node"` respectively
        components : dict[Hashable, Any]
            A dictionary of all non-input-output components names as keys
            with their SurPyval distribution as values
        edges : Iterable[tuple[Hashable, Hashable]]
            The collection of edges, e.g. [[1, 2], [2, 3]] would correspond to
            the edges 1-2 and 2-3
        mc_samples : int, optional
            TODO, by default 10_000

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
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
    ) -> np.ndarray[float]:
        """Returns the system reliability for time/s x

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable
        working_nodes : Iterable[Hashable], optional
            _description_, by default []
        broken_nodes : Iterable[Hashable], optional
            _description_, by default []
        working_components : Iterable[Hashable], optional
            _description_, by default []
        broken_components : Iterable[Hashable], optional
            _description_, by default []

        Returns
        -------
        np.ndarray[float]
            _description_

        Raises
        ------
        ValueError
            _description_
        """

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
                # Make a set of components out of the path/tieset
                s = set()
                for path in tieset_comb:
                    s.update(path)

                # Now calculate the tieset reliability
                tieset_rel = np.ones_like(x)
                for comp in s:
                    tieset_rel = tieset_rel * self.components[comp].sf(x)

                # Now add the tieset reliability to the level sum
                level_sum = level_sum + tieset_rel

            # Finally add/subtract the level sum to/from the system_rel if the
            # level is even/odd
            if i % 2 == 1:
                system_rel = system_rel + level_sum
            else:
                system_rel = system_rel - level_sum

        return system_rel

    def ff(self, x):
        return 1 - self.sf(x)

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
    def birnbaum_importance(self, x):
        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            failing = self.sf(x, broken_nodes=[node])
            node_importance[node] = working - failing
        return node_importance

    # TODO: update all importance measures to allow for component as well
    def improvement_potential(self, x):
        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            as_is = self.sf(x)
            node_importance[node] = working - as_is
        return node_importance

    def risk_achievement_worth(self, x):
        node_importance = {}
        for node in self.nodes.keys():
            failing = self.sf(x, broken_nodes=[node])
            as_is = self.sf(x)
            node_importance[node] = ((1 - failing) / (1 - as_is)) - 1
        return node_importance

    def risk_reduction_worth(self, x):
        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            as_is = self.sf(x)
            node_importance[node] = ((1 - as_is) / (1 - working)) - 1
        return node_importance

    def criticality_importance(self, x):
        I_B = self.birnbaum_importance(x)
        node_importance = {}
        for node in self.nodes.keys():
            as_is = self.sf(x)
            node_ff = self.components[self.nodes[node]].ff(x)
            node_importance[node] = I_B[node] * node_ff / (1 - as_is)
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
