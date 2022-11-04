import networkx as nx


class RBDGraph(nx.DiGraph):
    def default_node_attr_dict(self):
        return {"k": 1}

    node_attr_dict_factory = default_node_attr_dict
