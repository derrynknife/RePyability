import numpy as np
import networkx as nx
from collections import defaultdict
from copy import copy
import surpyval as surv

class PerfectReliability:
    @classmethod
    def sf(cls, x):
        return np.ones_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.zeros_like(x).astype(float)

class PerfectUnreliability:
    @classmethod
    def sf(cls, x):
        return np.zeros_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        return np.ones_like(x).astype(float)

class RBD:
    # TODO: Implement these
    # Finding cut-sets:
    # https://www.degruyter.com/document/doi/10.1515/9783110725599-007/html?lang=en

    def __init__(self, nodes, components, edges, mc_samples=10_000):

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
            if nodes[node] in ['input_node', 'output_node']:
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
        return nx.all_simple_paths(self.G, source=self.input_node, target=self.output_node)

    def sf(self, x, working_nodes=None, broken_nodes=None,
           working_components=None, broken_components=None):
        
        if not self.check_rbd_structure():
            raise ValueError("RBD not correctly structured, add edges or nodes to create correct structure.")
        if working_nodes is None:
            working_nodes = []
        if broken_nodes is None:
            broken_nodes = []
        if working_components is None:
            working_components = []
        if broken_components is None:
            broken_components = []

        x = np.atleast_1d(x)
        r_dict = {}
        for component in self.components.keys():
            if component in working_components:
                r_dict[component] = np.log(PerfectReliability.sf(x))
            elif component in broken_components:
                r_dict[component] = np.log(PerfectUnreliability.sf(x))
            else:
                r_dict[component] = np.log(self.components[component].sf(x))

        paths = self.all_path_sets()
        paths_reliabililty = []
        for path in paths:
            path_rel = np.zeros_like(x).astype(float)
            for node in path:
                if node in self.in_or_out:
                    continue
                if node in working_nodes:
                    path_rel += np.log(PerfectReliability.sf(x))
                elif node in broken_nodes:
                    path_rel += np.log(PerfectUnreliability.sf(x))
                else:
                    path_rel += r_dict[self.nodes[node]]
            paths_reliabililty.append(np.exp(path_rel))
        paths_ff = 1 - np.atleast_2d(paths_reliabililty)
        # return 1 - np.prod(paths_ff, axis=0)
        # Is this really needed?
        return 1 - np.exp(np.sum(np.log(paths_ff), axis=0))

    def ff(self, x):
        return 1 - self.sf(x)

    def check_rbd_structure(self):
        has_circular_dependency = not nx.is_directed_acyclic_graph(self.G)
        node_degrees = defaultdict(lambda: defaultdict(int))

        for edge in self.G.edges:
            source, target = edge
            node_degrees[source]['out'] += 1
            node_degrees[target]['in'] += 1

        input_nodes = [n for k, n in node_degrees.items() if n['in'] == 0]
        output_nodes = [n for k, n in node_degrees.items() if n['out'] == 0]
        has_node_with_no_input = len(input_nodes) != 1
        has_node_with_no_output = len(output_nodes) != 1
        if not any([has_circular_dependency,
                    has_node_with_no_input,
                    has_node_with_no_output]):

            self.rbd_structural_errors = None
            return True
        else:
            self.rbd_structural_errors = {
                "has_circular_dependency": has_circular_dependency,
                "has_node_with_no_input": has_node_with_no_input,
                "has_node_with_no_output": has_node_with_no_output
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

    def fussel_vessely(self, x, fv_type='p'):
        if fv_type not in ['c', 'p']:
            raise ValueError("'fv_type' must be either c (cut set) or p (path set)")

        # TODO: Implement cut set based FV importance.
        if fv_type == 'c':
            raise NotImplementedError("cut set type FV importance measure not yet implemented")

        system_reliability  = self.sf(x)

        paths = list(self.all_path_sets())
        node_importance = {}

        x = np.atleast_1d(x)
        r_dict = {}
        for component in self.components.keys():
            r_dict[component] = np.log(self.components[component].sf(x))

        for node in self.nodes.keys():
            paths_reliabililty = []
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
                paths_reliabililty.append(path_rel)
            
            paths_sf = np.atleast_2d(paths_reliabililty)
            paths_sf = 1 - np.exp(np.sum(paths_sf, axis=0))
            paths_sf = paths_sf / system_reliability
            node_importance[node] = np.copy(paths_sf)
        
        return node_importance
