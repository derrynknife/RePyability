from itertools import combinations
from typing import Hashable

import networkx as nx


def min_cut_sets(
    G: nx.DiGraph, input_node: Hashable, output_node: Hashable
) -> set[frozenset[Hashable]]:
    """Returns the set of minimal cut-sets of a graph

    Note: doesn't consider k-out-of-n nodes

    Parameters
    ----------
    G : nx.DiGraph
        The graph in question
    input_node : Hashable
        The input/source node
    output_node : Hashable
        The output/sink node

    Returns
    -------
    set[frozenset[Hashable]]
        The set of minimal nodal cut-sets, i.e. if any of these node-sets
        fail, then the system fails
    """
    # Get all possible node subsets (excluding input and output nodes)
    intermediate_nodes = set(G.nodes()) - {input_node, output_node}
    node_sets = get_all_node_subsets(intermediate_nodes)

    # For each node_subset, create a copy of graph G with these nodes removed
    # If input_node->output_node has no path then add it to the cut_sets set
    # But if there are any supersets of this cut-set, then those supersets
    # are not minimal cut-sets, so remove them

    # The set of cut-sets (not necessarily minimal cut-sets)
    cut_sets: set[frozenset] = set()

    # Check if the node-set is a cut-set
    for node_set in node_sets:
        # Create a copy of G with node_subset nodes removed
        H = G.copy()
        H.remove_nodes_from(node_set)

        # Check if the node-set is a cut-set
        if not nx.has_path(H, input_node, output_node):
            # If it is, check that there are no supersets of it in cut_sets
            # because if there are they are definitely not minimal cut-sets
            # and should be removed
            for cut_set in cut_sets.copy():
                if node_set.issubset(cut_set):
                    cut_sets.remove(cut_set)

            # Finally add the node-set to the set of cut-sets
            cut_sets.add(node_set)

    # By now all the non-minimal cut-sets have been removed, so its safe to say
    # that cut_sets is the set of all minimal_cut_sets
    minimal_cut_sets = cut_sets

    return minimal_cut_sets


def get_all_node_subsets(nodes: set[Hashable]) -> list[frozenset[Hashable]]:
    node_subsets = []
    for i in range(len(nodes), 0, -1):
        for comb in list(combinations(nodes, i)):
            node_subsets.append(frozenset(comb))
    return node_subsets
