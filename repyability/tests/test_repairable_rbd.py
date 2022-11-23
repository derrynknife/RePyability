import numpy as np
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


# TODO: Parameterise the distribution parameters 2/3 times
def test_repairable_rbd_availability_1N():
    """
    The goal here is to, for only one simulation, test if the availability
    is being calculated correctly. We're achieving this by seeding the numpy
    random number generator. NOTE: This of course means that this test is
    coupled to surpyval's .random() implementation!
    """
    # Make a Repairable RBD which has only one non-input/output component
    # This component has an exponential distribution for both it's
    # reliability and repairability
    reliability_dist = surv.Exponential.from_params([0.25])
    repairability_dist = surv.Exponential.from_params([0.5])

    rbd = RepairableRBD(
        nodes={
            "s": "input_node",
            "c": "c",
            "t": "output_node",
        },
        edges=[
            ("s", "c"),
            ("c", "t"),
        ],
        reliability={"c": reliability_dist},
        repairability={"c": repairability_dist},
    )

    # Before anything, set the numpy seed
    np.random.seed(1)

    # Let's run the sim for 50 units of time
    t_simulation = 50

    # Form expected availability
    # At t=0, availability=1 (it starts working)
    exp_t = [0]
    exp_availability = [1]

    # Get first failure event
    t_next_event = reliability_dist.random(1)[0]
    is_comp_working = True

    while t_next_event < t_simulation:
        if is_comp_working:
            # Failure event to be applied
            exp_t.append(t_next_event)
            exp_availability.append(0)
            is_comp_working = False

            # And get the next repair event
            # (the current time + the next event random sample)
            t_next_event += repairability_dist.random(1)[0]

        else:
            # Repair event to be applied
            exp_t.append(t_next_event)
            exp_availability.append(1)
            is_comp_working = True

            # And get the next failure event
            t_next_event += reliability_dist.random(1)[0]

    # Now check if expected == actual, remembering to reset the seed
    np.random.seed(1)
    actual_t, actual_availability = rbd.availability(
        t_simulation=t_simulation, N=1
    )

    assert pytest.approx(exp_t) == actual_t
    assert pytest.approx(exp_availability) == actual_availability
