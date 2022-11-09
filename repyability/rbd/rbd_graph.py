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

    # Function that returns the default node attribute dict
    def default_node_attr_dict(self):
        return {"k": 1, "type": "node"}

    # networkx relies on the subclass (RBDGraph) to set the variable
    # `node_attr_dict_factory` to the default node attribute dict function.
    node_attr_dict_factory = default_node_attr_dict
