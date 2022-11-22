import pytest
import surpyval as surv

from repyability.rbd.repairable_rbd import RepairableRBD


# Test RBDs as pytest fixtures
@pytest.fixture
def repairable_rbd1() -> RepairableRBD:
    """A simple RepairableRBD with three intermediate nodes."""
    nodes = {
        "source": "input_node",
        "pump1": "pump1",
        "pump2": "pump2",
        "valve": "valve",
        "sink": "output_node",
    }
    edges = [
        ("source", "pump1"),
        ("source", "pump2"),
        ("pump1", "valve"),
        ("pump2", "valve"),
        ("valve", "sink"),
    ]
    reliability = {
        "pump1": surv.Weibull.from_params([30, 2]),
        "pump2": surv.Weibull.from_params([21, 3.0]),
        "valve": surv.Weibull.from_params([25, 2.7]),
    }

    repairability = {
        "pump1": surv.LogNormal.from_params([0.1, 0.2]),
        "pump2": surv.LogNormal.from_params([0.1, 0.3]),
        "valve": surv.LogNormal.from_params([0.2, 0.2]),
    }
    return RepairableRBD(nodes, reliability, repairability, edges)


def test_repairable_rbd_missing_repairability_component():
    """
    Checks that a ValueError is raised when a component isn't provided
    with a repairability distribution.
    """
    nodes = {
        "input_node": "input_node",
        "no_repairability": "no_repairability",
        "output_node": "output_node",
    }
    reliability = {
        "no_repairability": surv.LogNormal.from_params([0.1, 0.2]),
    }
    edges = [
        ("input_node", "no_repairability"),
        ("no_repairability", "output_node"),
    ]
    repairability = {}
    with pytest.raises(ValueError):
        RepairableRBD(nodes, reliability, repairability, edges)
