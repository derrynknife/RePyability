import networkx as nx
from networkx import DiGraph


class RBDGraph(DiGraph):
    """A class that wraps networkx.DiGraph for the purposes of RBDs

    RBDGraph differs from DiGraph in only a few ways:
    - RBDGraph utilises DiGraph's ability to have node attributes, this is
      where information like the k in the k-out-of-n sense is stored for each
      node.

    Default node attribute dict:
    {
        "k": 1,
        "type": "node"  # Can be "node"/"input_node"/"output_node"
    }

    """

    def is_valid_RBD_structure(
        self, nodes=None, input_node=None, output_node=None
    ) -> dict:
        """Returns False if invalid RBD structure

        Invalid RBD structure includes:
        - having cycles present
        - a non-input/output node having no in/out-nodes

        Returns
        -------
        bool
            True
        """
        # is_dag = not nx.is_directed_acyclic_graph(self)
        results: dict = {}
        results["is_valid"] = True
        cycles = list(nx.simple_cycles(self))
        if cycles != []:
            results["is_valid"] = False
            results["has_cycles"] = True
        else:
            results["has_cycles"] = False

        cycles_set = {frozenset(cycle) for cycle in cycles}

        results["cycles"] = cycles_set

        input_nodes = [n for n, id in self.in_degree(self.nodes) if (id == 0)]
        output_nodes = [
            n for n, od in self.out_degree(self.nodes) if (od == 0)
        ]
        node_has_no_predecessor = len(input_nodes) != 1
        node_has_no_successor = len(output_nodes) != 1

        results["has_nodes_with_no_predecessor"] = node_has_no_predecessor
        results["has_nodes_with_no_successor"] = node_has_no_successor

        if node_has_no_predecessor:
            results["is_valid"] = False
            if input_node is None:
                results["has_unique_input_node"] = False
                results["input_node"] = None
                results["nodes_with_no_predecessors"] = input_nodes
            else:
                results["has_unique_input_node"] = True
                results["input_node"] = input_node
                results["nodes_with_no_predecessors"] = [
                    n for n in input_nodes if n != input_node
                ]
        else:
            results["has_unique_input_node"] = True
            if input_node is None:
                results["input_node"] = input_nodes[0]
            else:
                results["input_node"] = input_node

        if node_has_no_successor:
            results["is_valid"] = False
            if output_node is None:
                results["has_unique_output_node"] = False
                results["nodes_with_no_successors"] = output_nodes
                results["output_node"] = None
            else:
                results["has_unique_output_node"] = True
                results["nodes_with_no_successors"] = [
                    n for n in output_nodes if n != output_node
                ]
                results["output_node"] = output_node
        else:
            results["has_unique_output_node"] = True
            if output_node is None:
                results["output_node"] = output_nodes[0]
            else:
                results["output_node"] = output_node

        koon_errors = []
        koon_warnings = []
        results["has_koon_errors"] = False
        results["has_koon_warnings"] = False
        for n in self.nodes:
            k = self.nodes[n]["k"]
            if k == 0:
                results["is_valid"] = False
                results["has_koon_errors"] = True
                koon_errors.append(
                    "Node {} has k of zero. Must be positive integer".format(n)
                )
            if k > 1:
                in_degree = self.in_degree(n)
                if in_degree == k:
                    results["has_koon_warnings"] = True
                    koon_warnings.append(
                        (
                            "Node {n} requires {k} working but has {k} inputs."
                            + " Is k correct or should this be a series "
                            + "strucuture instead?"
                        ).format(n=n, k=k)
                    )
                if in_degree < k:
                    results["is_valid"] = False
                    results["has_koon_errors"] = True
                    koon_errors.append(
                        "node {n} requires {k} working but has only"
                        + " {in_degree} paths to it."
                    )

        results["koon_errors"] = koon_errors
        results["koon_warnings"] = koon_warnings

        return results

    # Function that returns the default node attribute dict
    def default_node_attr_dict(self):
        return {"k": 1, "type": "node"}

    # networkx relies on the subclass (RBDGraph) to set the variable
    # `node_attr_dict_factory` to the default node attribute dict function.
    node_attr_dict_factory = default_node_attr_dict
