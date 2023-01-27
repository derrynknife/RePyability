import pytest
import surpyval as surv

from repyability.rbd.helper_classes import PerfectReliability as PR
from repyability.rbd.helper_classes import PerfectUnreliability as PU
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


def test_incorrect_infeasible_option():
    edges = [(0, 1), (1, 3)]
    reliabilities = {1: PR}

    with pytest.raises(ValueError):
        NonRepairableRBD(edges, reliabilities, on_infeasible_rbd="foo")


def test_self_reference():
    edges = [(0, 1), (1, 3)]
    reliabilities = {1: 1}

    # Test error is raised when input not provided
    with pytest.raises(ValueError):
        NonRepairableRBD(edges, reliabilities)


def test_repeated_koon_nodes():
    edges = [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
        (0, 4),
        (4, 6),
        (0, 5),
        (5, 6),
        (3, 7),
        (6, 7),
    ]
    reliabilities = {
        1: surv.Weibull.from_params([10, 2]),
        2: surv.Weibull.from_params([10, 2]),
        3: surv.Weibull.from_params([10, 2]),
        4: surv.Weibull.from_params([10, 2]),
        5: surv.Weibull.from_params([10, 2]),
        6: 3,
        7: surv.Weibull.from_params([10, 2]),
    }

    k = {3: 2, 6: 2}

    with pytest.raises(ValueError):
        NonRepairableRBD(edges, reliabilities, k=k)


def test_repeated_nodes():
    edges = [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
    ]
    reliabilities = {
        1: surv.Weibull.from_params([10, 2]),
        2: 1,
    }

    NonRepairableRBD(edges, reliabilities)


def test_perfect_rel_and_unrel():
    edges = [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
    ]
    reliabilities = {
        1: PR,
        2: PU,
    }

    NonRepairableRBD(edges, reliabilities)


def test_nonparametric_node():
    edges = [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
    ]

    reliabilities = {
        1: surv.KaplanMeier.fit([1, 2, 3, 4, 5]),
        2: PU,
    }

    NonRepairableRBD(edges, reliabilities)


def test_repeated_node_in_cycle():
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]

    reliabilities = {
        1: PR,
        2: PR,
        3: 1,
    }

    with pytest.raises(ValueError):
        NonRepairableRBD(edges, reliabilities)
