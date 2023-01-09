"""
Contains function to check an RBD args are complete.
See function docstring for more info.
"""


from typing import Collection, Hashable, Iterable


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


def check_sf_node_component_args_consistency(
    working_nodes: Collection[Hashable],
    broken_nodes: Collection[Hashable],
    working_components: Collection[Hashable],
    broken_components: Collection[Hashable],
    rbd_components_to_nodes: dict[Hashable, set],
):
    argument_nodes: set[object] = set()
    argument_nodes.update(working_nodes)
    argument_nodes.update(broken_nodes)
    number_of_arg_comp_nodes = 0
    for comp in working_components:
        argument_nodes.update(rbd_components_to_nodes)
        number_of_arg_comp_nodes += len(rbd_components_to_nodes)
    for comp in broken_components:
        argument_nodes.update(rbd_components_to_nodes)
        number_of_arg_comp_nodes += len(rbd_components_to_nodes)
    if len(argument_nodes) != (
        len(working_nodes) + len(broken_nodes) + number_of_arg_comp_nodes
    ):
        working_comps_nodes = [
            (comp, rbd_components_to_nodes) for comp in working_components
        ]
        broken_comps_nodes = [
            (comp, rbd_components_to_nodes) for comp in broken_components
        ]

        raise ValueError(
            f"Node/component inconsistency provided. i.e. you have \
            provided sf() with working/broken nodes (or respective \
            components) more than once in the arguments.\n \
            Supplied:\n\
            working_nodes: {working_nodes}\n\
            broken_nodes: {broken_nodes}\n\
            (working_components, their_nodes): {working_comps_nodes}\n\
            (broken_components, their_nodes): {broken_comps_nodes}\n"
        )
