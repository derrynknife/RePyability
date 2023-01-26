import pytest
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD


def test_excess_nodes():
    """
    Checks that a ValueError is raised when a component isn't provided
    with a repairability distribution.
    """
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]
    reliabilities = {
        2: surv.Weibull.from_params([10, 2]),
        3: surv.Weibull.from_params([10, 2]),
        4: surv.Weibull.from_params([10, 2]),
    }
    # Test warning raised when asked
    with pytest.warns():
        NonRepairableRBD(edges, reliabilities, on_infeasible_rbd="warn")

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        NonRepairableRBD(edges, reliabilities)

    # Test ignore error when asked
    NonRepairableRBD(edges, reliabilities, on_infeasible_rbd="ignore")


def test_excess_reliabilities():
    """
    Checks that a ValueError is raised when a component isn't provided
    with a repairability distribution.
    """
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]
    reliabilities = {
        2: surv.Weibull.from_params([10, 2]),
        3: surv.Weibull.from_params([10, 2]),
        4: surv.Weibull.from_params([10, 2]),
        5: surv.Weibull.from_params([10, 2]),
        7: surv.Weibull.from_params([10, 2]),
    }

    # Test warning raised when asked
    with pytest.warns():
        NonRepairableRBD(edges, reliabilities, on_infeasible_rbd="warn")

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        NonRepairableRBD(edges, reliabilities)

    # Test ignore error when asked
    NonRepairableRBD(edges, reliabilities, on_infeasible_rbd="ignore")


# Check components are correct lengths
def test_input_errors():
    two_inputs_edges = [(0, 1), (2, 1), (1, 3), (3, 4)]
    reliabilities: dict = {}

    # Test warning raised when asked
    with pytest.warns():
        NonRepairableRBD(
            two_inputs_edges, reliabilities, on_infeasible_rbd="warn"
        )

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        NonRepairableRBD(two_inputs_edges, reliabilities)

    # Test ignore error when asked
    NonRepairableRBD(
        two_inputs_edges, reliabilities, on_infeasible_rbd="ignore"
    )


def test_output_errors():
    two_outputs_edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    reliabilities: dict = {}

    # Test warning raised when asked
    with pytest.warns():
        NonRepairableRBD(
            two_outputs_edges, reliabilities, on_infeasible_rbd="warn"
        )

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        NonRepairableRBD(two_outputs_edges, reliabilities)

    # Test ignore error when asked
    NonRepairableRBD(
        two_outputs_edges, reliabilities, on_infeasible_rbd="ignore"
    )


def test_cycle_errors():
    edges_with_cycle = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 1),
        (3, 4),
    ]
    reliabilities: dict = {}

    # Test warning raised when asked
    with pytest.warns():
        NonRepairableRBD(
            edges_with_cycle, reliabilities, on_infeasible_rbd="warn"
        )

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        NonRepairableRBD(edges_with_cycle, reliabilities)

    # Test ignore error when asked
    NonRepairableRBD(
        edges_with_cycle, reliabilities, on_infeasible_rbd="ignore"
    )


# Check components are correct lengths
def test_self_reference():
    two_inputs_edges = [(0, 1), (1, 3)]
    reliabilities = {1: 1}

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        NonRepairableRBD(two_inputs_edges, reliabilities)
