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
    if rbd_graph.nodes[curr_node]["type"] == "input_node":
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

    # If there's no way of satisfying the k requirements of curr_node,
    # just skip all the funny business and return the empty list
    if len(all_subproblem_solns) < k:
        solns[curr_node] = []
        return []

    # Merge the subproblem solns to satisfy the curr_node's k requirement
    # e.g. k = 1, just add all the sets to the ret_list
    # e.g. k = 2, add all possible pairs across subproblems
    #   i.e. if there's 2 subproblems, each with 2 minimal path-sets
    #        (aa, ab, ba, bb) then merge the sets (aa+ba, aa+bb, ab+ba, ab+bb)

    # Get all the necessary path-set combinations
    merged_soln = merge_subproblem_solns(all_subproblem_solns, k)

    # Remove all non-minimal path-sets
    merged_and_minimal_soln = minimalise_path_sets(merged_soln)

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


def merge_subproblem_solns(
    all_subproblem_solns: list[list[set[Hashable]]], curr_node_k: int
) -> list[set[Hashable]]:
    """Returns k-wise path-set combinations for the current node"""
    # First get the node-in combinations
    n_in = len(all_subproblem_solns)
    combs = get_node_in_combinations(n_in, curr_node_k)

    # Then form the path-set combinations (or set-additions rather)

    merged_soln: list[set] = []
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

            # And add it to the merged_soln list
            merged_soln.append(s)

    return merged_soln


def minimalise_path_sets(path_sets: list[set]) -> list[set]:
    """Returns only the minimal path-sets

    i.e. Discards all path-sets that are a superset of any of the other
    path-sets.

    Parameters
    ----------
    path_sets : list[set]
        All path-sets to minimalise
    """
    minimal_path_sets = []
    for path_set in path_sets:
        is_path_set_minimal = True  # Assume path-set is minimal
        for other_path_set in path_sets:
            if path_set == other_path_set:
                # Don't consider the same path-set
                continue
            if path_set.issuperset(other_path_set):
                # Discard path_set as it's non-minimal (other_path_set is a
                # subset of this path_set)
                is_path_set_minimal = False
                break
        if is_path_set_minimal:
            minimal_path_sets.append(path_set)
    return minimal_path_sets
