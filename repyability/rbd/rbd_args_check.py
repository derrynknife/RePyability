"""
Contains function to check an RBD args are complete.
See function docstring for more info.
"""


from typing import Hashable, Iterable


def check_rbd_node_args_complete(
    nodes: dict,
    reliability: dict,
    edges: Iterable[tuple[Hashable, Hashable]],
    repairability: dict | None = None,
) -> None:
    """Checks if the RBD args are complete

    It raises a ValueError if any of the following are discovered:
    - a node is not found in the edges list
    - a node's component is not found in the reliability dict
    - a node's component is not found in the repairability dict (only checked
      if the repairability dict is provided)
    """
    # Check all nodes in edges list
    # Make a set out of the edges list for O(1) lookup
    edges_nodes: set[Hashable] = {node for edge in edges for node in edge}

    for node_name, component_name in nodes.items():
        # Check node is in edges
        if node_name not in edges_nodes:
            raise ValueError(f"Node {node_name} not in edges list.")

        # That's all that's needed for input and output nodes
        if component_name == "input_node" or component_name == "output_node":
            continue

        # Otherwise we need to check that a reliability distribution (or RBD)
        # is provided for that node, in the reliability dict
        if component_name not in reliability:
            raise ValueError(
                f"No reliability value provided for component\
                {component_name} (this component corresponds to node\
                {node_name})."
            )

        # And if repairability is provided, that a repairability distribution
        # (or RBD) is provided for that component
        if repairability is not None and component_name not in repairability:
            raise ValueError(
                f"No repairability value provided for component\
                {component_name} (this component corresponds to node\
                {node_name})."
            )
