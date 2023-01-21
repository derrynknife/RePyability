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
    with pytest.raises(ValueError):
        NonRepairableRBD(edges, reliabilities)

    try:
        NonRepairableRBD(edges, reliabilities)
    except Exception as e:
        assert str(e) == "Node 5 not in reliabilities dict"


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

    with pytest.raises(ValueError) as e:
        NonRepairableRBD(edges, reliabilities)

    try:
        NonRepairableRBD(edges, reliabilities)
    except Exception as e:  # noqa
        assert str(e) == "Nodes {7} not in edge list"


def test_matching_inputs():
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
    }

    NonRepairableRBD(edges, reliabilities)
