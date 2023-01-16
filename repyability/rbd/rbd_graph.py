from collections import defaultdict

import networkx as nx


class RBDGraph(nx.DiGraph):
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
        has_circular_dependency = not nx.is_directed_acyclic_graph(self)
        node_degrees: dict = defaultdict(lambda: defaultdict(int))

        for edge in self.edges:
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
            errors = ""
            if has_circular_dependency:
                errors += "- Has circular logic \n"
            if has_node_with_no_input:
                errors += "- Has nodes with no predecessor\n"
            if has_node_with_no_output:
                errors += "- Has nodes with no successor\n"
            self.rbd_structural_errors = errors
            return False

    # Function that returns the default node attribute dict
    def default_node_attr_dict(self):
        return {"k": 1, "type": "node"}

    # networkx relies on the subclass (RBDGraph) to set the variable
    # `node_attr_dict_factory` to the default node attribute dict function.
    node_attr_dict_factory = default_node_attr_dict
