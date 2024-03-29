import copy
import itertools
from typing import Hashable

from repyability.rbd.rbd_graph import RBDGraph


def min_path_sets(
    rbd_graph: RBDGraph,
    curr_node: Hashable,
    solns: dict[Hashable, list[set]],
) -> list[set[Hashable]]:
    """Returns the minimal path sets for an RBDGraph up until curr_node

    This is used as a recursive function in a memoised approach to solving
    the k-out-of-n minimal path-set problem.

    Parameters
    ----------
    rbd_graph : RBDGraph
        The complete RBDGraph
    curr_node : Hashable
        The node to 'go up to', the first call should be the output_node
    solns : dict[Hashable, list[set]]
        The 'memo table' to prevent solving the same subproblem

    Returns
    -------
    list[set[Hashable]]
        The list of minimal path-sets, empty if no path-sets can be made
        (including consideration of the k-out-of-n behaviour)
    """

    # Base case: if curr_node is the input_node then return the input_node
    if rbd_graph.in_degree(curr_node) == 0:
        ret_list = [{curr_node}]
        # Store in the solns dict, and return
        solns[curr_node] = copy.deepcopy(ret_list)
        return ret_list

    # Else, solve the subproblems
    # all_subproblem_solns is like [[subproblem1_sets],[subproblem2_sets],...]
    all_subproblem_solns: list[list[set[Hashable]]] = []
    for in_node in rbd_graph.predecessors(curr_node):
        # Get the subproblem soln (a list of minimal path-sets)
        if in_node in solns:
            # If it's already been solved, just used the memoised result
            # Needs to be deepcopied so it doesn't affect solns[in_node]
            subproblem_soln = copy.deepcopy(solns[in_node])
        else:
            # Otherwise go ahead and solve the subproblem
            subproblem_soln = min_path_sets(
                rbd_graph,
                in_node,
                solns,
            )

        # Add curr_node to each set
        for s in subproblem_soln:
            s.add(curr_node)

        # Collect the subproblem solns, if there are any
        # (subproblem_soln is non-empty)
        # We don't want to add an empty list to all_subproblem_solns
        # as it makes it simpler for the later steps
        if subproblem_soln:
            all_subproblem_solns.append(subproblem_soln)

    # Get the curr_node's k value
    k = rbd_graph.nodes[curr_node]["k"]

    # Merge the subproblem solns to satisfy the curr_node's k requirement
    # e.g. k = 1, just add all the sets to the ret_list
    # e.g. k = 2, add all possible pairs across subproblems
    #   i.e. if there's 2 subproblems, each with 2 minimal path-sets
    #        (aa, ab, ba, bb) then merge the sets (aa+ba, aa+bb, ab+ba, ab+bb)

    # Get all the necessary path-set combinations and discard non-minimal solns
    merged_and_minimal_soln = merge_and_minimalise_subproblem_solns(
        all_subproblem_solns, k
    )

    # Finally, memoise the solution to this curr_node problem, and return
    if rbd_graph.out_degree(curr_node) > 1:
        solns[curr_node] = copy.deepcopy(merged_and_minimal_soln)

    return merged_and_minimal_soln


def get_node_in_combinations(n_in: int, k: int) -> list[tuple[int, ...]]:
    """Returns all the indexed-0 node combinations for a k k-out-of-n node with
    n_in predecessors

    e.g. If the k-out-of-n node has k=2, and has 4 predecessors, this will
    return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)].
    """
    return list(itertools.combinations(range(n_in), k))


def merge_and_minimalise_subproblem_solns(
    all_subproblem_solns: list[list[set[Hashable]]], curr_node_k: int
) -> list[set[Hashable]]:
    """Returns k-wise path-set combinations for the current node"""
    # First get the node-in combinations
    n_in = len(all_subproblem_solns)
    combs = get_node_in_combinations(n_in, curr_node_k)

    # Then form the path-set combinations (or set-additions rather)

    merged_and_minimal_soln: list[set] = []
    for comb in combs:
        # product_candidates are all the subproblem solns we're considering for
        # this combination
        product_candidates: list[list[set]] = [
            all_subproblem_solns[i] for i in comb
        ]

        # itertools.product() performs all the magic for us, basically
        # getting us the '(aa, ab, ba, bb)' we need
        for product_tup in itertools.product(*product_candidates):
            # Now merge the sets, i.e. ({1, 2}, {3, 4}) -> {1, 2, 3, 4}
            s = set(product_tup[0])
            for i in range(1, len(product_tup)):
                s |= product_tup[i]

            # Minimalise, that is go through the merged_and_minimal_soln
            # and discard any supersets of this s, and if s is a superset
            # of any other set then continue (s is not minimal)
            is_path_set_minimal = True  # Assume path-set is minimal
            for other_s in merged_and_minimal_soln.copy():
                if s.issuperset(other_s):
                    is_path_set_minimal = False
                    break
                if s.issubset(other_s):
                    merged_and_minimal_soln.remove(other_s)

            # And add it to the merged_and_minimal_soln list
            if is_path_set_minimal:
                merged_and_minimal_soln.append(s)

    return merged_and_minimal_soln
