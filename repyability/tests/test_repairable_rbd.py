from collections import defaultdict

import numpy as np
import pytest
import surpyval as surv

from repyability.rbd.repairable_rbd import RepairableRBD


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


# One component, one simulation, tests simple case
def test_repairable_rbd_availability_one_component_1N():
    """
    The goal here is to, for only one simulation, test if the availability
    is being calculated correctly. We're achieving this by seeding the numpy
    random number generator. NOTE: This of course means that this test is
    coupled to surpyval's .random() implementation!
    """
    # Get a random seed
    seed = np.random.randint(0, 100)

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
    np.random.seed(seed)

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
    np.random.seed(seed)
    actual_t, actual_availability = rbd.availability(
        t_simulation=t_simulation, N=1
    )

    assert pytest.approx(exp_t) == actual_t[:-1]
    assert pytest.approx(exp_availability) == actual_availability[:-1]


# One component, ten simulations, tests the 0:N simulation loop
def test_repairable_rbd_availability_one_component_10N():
    """
    Basically a copy of test_repairable_rbd_availability_one_component_1N but
    performs 10 simulations. NOTE: Again, this is coupled with surpyval's
    .random() implementation.
    (See test_repairable_rbd_availability_one_component_1N()'s docstring.)
    """
    # Get a random seed
    seed = np.random.randint(0, 100)

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
    np.random.seed(seed)

    # Let's run the sim for 50 units of time
    t_simulation = 50

    # Form expected availability
    # At t=0, availability=1 (it starts working)
    exp_timeline = defaultdict(lambda: 0)

    # Perform 10 simulations
    n_simulations = 10
    for i in range(n_simulations):
        # Set initial availability to 1 again
        exp_timeline[0] += 1

        # Get first failure event
        t_next_event = reliability_dist.random(1)[0]
        is_comp_working = True

        while t_next_event < t_simulation:
            if is_comp_working:
                # Failure event to be applied
                exp_timeline[t_next_event] -= 1
                is_comp_working = False

                # And get the next repair event
                # (the current time + the next event random sample)
                t_next_event += repairability_dist.random(1)[0]

            else:
                # Repair event to be applied
                exp_timeline[t_next_event] += 1
                is_comp_working = True

                # And get the next failure event
                t_next_event += reliability_dist.random(1)[0]

    # Make the time and availability lists, sorted by time
    exp_t = sorted(exp_timeline.keys())
    exp_availability = np.array([exp_timeline[t_key] for t_key in exp_t])
    exp_availability = exp_availability.cumsum() / n_simulations

    # Now check if expected == actual, remembering to reset the seed
    np.random.seed(seed)
    actual_t, actual_availability = rbd.availability(
        t_simulation=t_simulation, N=n_simulations
    )

    # Remove last element, which is the same as the second last
    assert pytest.approx(exp_t) == actual_t[:-1]
    assert pytest.approx(exp_availability) == actual_availability[:-1]


# Two parallel components, one simulation, tests the triaging of component
# events
def test_repairable_rbd_availability_two_parallel_components_1N():
    """
    We're gonna make a 2-component parallel RBD, and simulate availability
    once to test the component-event-triaging of availability(). NOTE: Again,
    this is coupled with surpyval's .random() implementation.
    (See test_repairable_rbd_availability_one_component_1N()'s docstring.)
    """

    # Make a two-component parallel RepairableRBD
    # We're just using identical exponential distributions for each for
    # simplicity
    reliability_dist = surv.Exponential.from_params([0.25])
    repairability_dist = surv.Exponential.from_params([0.5])
    rbd = RepairableRBD(
        nodes={
            "s": "input_node",
            "c1": "c1",
            "c2": "c2",
            "t": "output_node",
        },
        edges=[
            ("s", "c1"),
            ("c1", "t"),
            ("s", "c2"),
            ("c2", "t"),
        ],
        reliability={"c1": reliability_dist, "c2": reliability_dist},
        repairability={"c1": repairability_dist, "c2": repairability_dist},
    )

    # Before anything, set the numpy seed randomly
    seed = np.random.randint(0, 100)
    np.random.seed(seed)

    # Get the first failure event for each component
    next_component_events = {
        "c1": {
            "component": "c1",
            "type": "failure",
            "time": reliability_dist.random(1)[0],
        },
        "c2": {
            "component": "c2",
            "type": "failure",
            "time": reliability_dist.random(1)[0],
        },
    }

    # Keep track of component states, they both start working
    component_states = {"c1": True, "c2": True}

    # Helper function to determine from component_states if system is working
    def is_system_working():
        return component_states["c1"] or component_states["c2"]

    # Helper function to get the next event from next_component_events
    def get_next_event():
        if (
            next_component_events["c1"]["time"]
            < next_component_events["c2"]["time"]
        ):
            return next_component_events["c1"]
        else:
            return next_component_events["c2"]

    # Let's run the sim for 50 units of time
    t_simulation = 50

    # Keep track of the system's state, it starts off working
    prev_system_state = True

    # Form expected availability
    # At t=0, availability=1 (it starts working)
    exp_t = [0]
    exp_availability = [1]

    # Get the first failure event
    next_event = get_next_event()

    # While the next event's time is in the simulation
    while next_event["time"] < t_simulation:
        # This event's component
        component = next_event["component"]

        # If the next event is a failure, set the component's state to not
        # working, and if the system was previously working, but is now
        # not working, record this in exp_availability, and add a new
        # repair event to next_component_states
        if next_event["type"] == "failure":
            component_states[component] = False
            if prev_system_state and not is_system_working():
                exp_t.append(next_event["time"])
                exp_availability.append(0)
                prev_system_state = False

            next_component_events[component] = {
                "component": component,
                "type": "repair",
                "time": next_event["time"] + repairability_dist.random(1)[0],
            }

        # Else, the next event is a repair, set the component's state to
        # working, and if the system was previously not working, but is now
        # working, record this in exp_availability, and add a new
        # failure event to next_component_states
        else:
            component_states[component] = True
            if not prev_system_state and is_system_working():
                exp_t.append(next_event["time"])
                exp_availability.append(1)
                prev_system_state = True

            next_component_events[component] = {
                "component": component,
                "type": "failure",
                "time": next_event["time"] + reliability_dist.random(1)[0],
            }

        # Get next event
        next_event = get_next_event()

    # Now check if expected == actual, remembering to reset the seed
    np.random.seed(seed)
    actual_t, actual_availability = rbd.availability(
        t_simulation=t_simulation, N=1
    )

    assert pytest.approx(exp_t) == actual_t[:-1]
    assert pytest.approx(exp_availability) == actual_availability[:-1]


# Two series components, one simulation, tests the triaging of component events
def test_repairable_rbd_availability_two_series_components_1N():
    """
    We're gonna make a 2-component series RBD, and simulate availability
    once to test the component-event-triaging of availability(). NOTE: Again,
    this is coupled with surpyval's .random() implementation.
    (See test_repairable_rbd_availability_one_component_1N()'s docstring.)
    """

    # Make a two-component series RepairableRBD
    # We're just using identical exponential distributions for each for
    # simplicity
    reliability_dist = surv.Exponential.from_params([0.25])
    repairability_dist = surv.Exponential.from_params([0.5])
    rbd = RepairableRBD(
        nodes={
            "s": "input_node",
            "c1": "c1",
            "c2": "c2",
            "t": "output_node",
        },
        edges=[
            ("s", "c1"),
            ("c1", "c2"),
            ("c2", "t"),
        ],
        reliability={"c1": reliability_dist, "c2": reliability_dist},
        repairability={"c1": repairability_dist, "c2": repairability_dist},
    )

    # Before anything, set the numpy seed randomly
    seed = np.random.randint(0, 100)
    np.random.seed(seed)

    # Get the first failure event for each component
    next_component_events = {
        "c1": {
            "component": "c1",
            "type": "failure",
            "time": reliability_dist.random(1)[0],
        },
        "c2": {
            "component": "c2",
            "type": "failure",
            "time": reliability_dist.random(1)[0],
        },
    }

    # Keep track of component states, they both start working
    component_states = {"c1": True, "c2": True}

    # Helper function to determine from component_states if system is working
    def is_system_working():
        return component_states["c1"] and component_states["c2"]

    # Helper function to get the next event from next_component_events
    def get_next_event():
        if (
            next_component_events["c1"]["time"]
            < next_component_events["c2"]["time"]
        ):
            return next_component_events["c1"]
        else:
            return next_component_events["c2"]

    # Let's run the sim for 50 units of time
    t_simulation = 50

    # Keep track of the system's state, it starts off working
    prev_system_state = True

    # Form expected availability
    # At t=0, availability=1 (it starts working)
    exp_t = [0]
    exp_availability = [1]

    # Get the first failure event
    next_event = get_next_event()

    # While the next event's time is in the simulation
    while next_event["time"] < t_simulation:
        # This event's component
        component = next_event["component"]

        # If the next event is a failure, set the component's state to not
        # working, and if the system was previously working, but is now
        # not working, record this in exp_availability, and add a new
        # repair event to next_component_states
        if next_event["type"] == "failure":
            component_states[component] = False
            if prev_system_state and not is_system_working():
                exp_t.append(next_event["time"])
                exp_availability.append(0)
                prev_system_state = False

            next_component_events[component] = {
                "component": component,
                "type": "repair",
                "time": next_event["time"] + repairability_dist.random(1)[0],
            }

        # Else, the next event is a repair, set the component's state to
        # working, and if the system was previously not working, but is now
        # working, record this in exp_availability, and add a new
        # failure event to next_component_states
        else:
            component_states[component] = True
            if not prev_system_state and is_system_working():
                exp_t.append(next_event["time"])
                exp_availability.append(1)
                prev_system_state = True

            next_component_events[component] = {
                "component": component,
                "type": "failure",
                "time": next_event["time"] + reliability_dist.random(1)[0],
            }

        # Get next event
        next_event = get_next_event()

    # Now check if expected == actual, remembering to reset the seed
    np.random.seed(seed)
    actual_t, actual_availability = rbd.availability(
        t_simulation=t_simulation, N=1
    )

    assert pytest.approx(exp_t) == actual_t[:-1]
    assert pytest.approx(exp_availability) == actual_availability[:-1]
