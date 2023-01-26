"""
Contains the pytest fixtures used by many of the tests. They all define
and return NonRepairableRBDs.
"""

import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.standby_node import StandbyModel


@pytest.fixture
def rbd_series() -> NonRepairableRBD:
    """A simple NonRepairableRBD with three intermediate nodes in series."""
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    reliabilities = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
    }
    return NonRepairableRBD(edges, reliabilities)


@pytest.fixture
def rbd_parallel() -> NonRepairableRBD:
    """A simple NonRepairableRBD with three intermediate nodes in parallel."""
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]
    reliabilities = {
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
    }
    return NonRepairableRBD(edges, reliabilities)


@pytest.fixture
def rbd1() -> NonRepairableRBD:
    """Example 6.10 from Modarres & Kaminskiy."""
    qp = 0.03
    qv = 0.01
    edges = [
        ("source", "pump1"),
        ("source", "pump2"),
        ("pump1", "valve"),
        ("pump2", "valve"),
        ("valve", "sink"),
    ]
    reliabilities = {
        "pump1": FixedEventProbability.from_params(qp),
        "pump2": FixedEventProbability.from_params(qp),
        "valve": FixedEventProbability.from_params(qv),
    }

    return NonRepairableRBD(edges, reliabilities)


@pytest.fixture
def rbd2() -> NonRepairableRBD:
    edges = [(1, 2), (2, 3), (2, 4), (4, 7), (3, 5), (5, 6), (6, 7), (7, 8)]
    reliabilities = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
        5: surv.Weibull.from_params([15, 1.2]),
        6: surv.Weibull.from_params([80, 10]),
        7: StandbyModel(
            [
                surv.Weibull.from_params([5, 1.1]),
                surv.Weibull.from_params([5, 1.1]),
                surv.Weibull.from_params([5, 1.1]),
                surv.Weibull.from_params([5, 1.1]),
            ]
        ),
    }
    return NonRepairableRBD(edges, reliabilities)


@pytest.fixture
def rbd3() -> NonRepairableRBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    """
    edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (3, 4),
        (1, 5),
        (3, 5),
        (5, 2),
        (5, 4),
        (2, 6),
        (4, 6),
    ]
    reliabilities = {
        1: FixedEventProbability.from_params(1 - 0.95),
        2: FixedEventProbability.from_params(1 - 0.95),
        3: FixedEventProbability.from_params(1 - 0.95),
        4: FixedEventProbability.from_params(1 - 0.95),
        5: FixedEventProbability.from_params(1 - 0.95),
    }
    return NonRepairableRBD(edges, reliabilities)


@pytest.fixture
def rbd_repeated_component_parallel() -> NonRepairableRBD:
    """Basically rbd_parallel with a repeated component (component 2)."""
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)]
    reliabilities = {
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
        5: 2,
    }
    return NonRepairableRBD(edges, reliabilities)


@pytest.fixture
def rbd1_koon() -> NonRepairableRBD:
    """Example 6.10 from Modarres & Kaminskiy, w/ valve k=2."""
    qp = 0.03
    qv = 0.01
    edges = [
        ("source", "pump1"),
        ("source", "pump2"),
        ("pump1", "valve"),
        ("pump2", "valve"),
        ("valve", "sink"),
    ]
    reliabilities = {
        "pump1": FixedEventProbability.from_params(qp),
        "pump2": FixedEventProbability.from_params(qp),
        "valve": FixedEventProbability.from_params(qv),
    }
    k = {"valve": 2}
    return NonRepairableRBD(edges, reliabilities, k)


@pytest.fixture
def rbd2_koon() -> NonRepairableRBD:
    """rbd2 but k of node 7 is =2."""
    edges = [(1, 2), (2, 3), (2, 4), (4, 7), (3, 5), (5, 6), (6, 7), (7, 8)]
    reliabilities = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
        5: surv.Weibull.from_params([15, 1.2]),
        6: surv.Weibull.from_params([80, 10]),
        7: StandbyModel(
            [
                surv.Weibull.from_params([5, 1.1]),
                surv.Weibull.from_params([5, 1.1]),
                surv.Weibull.from_params([5, 1.1]),
                surv.Weibull.from_params([5, 1.1]),
            ]
        ),
    }
    k = {7: 2}
    return NonRepairableRBD(edges, reliabilities, k)


def rbd3_nodes_edges_components():
    edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (3, 4),
        (1, 5),
        (3, 5),
        (5, 2),
        (5, 4),
        (2, 6),
        (4, 6),
    ]
    reliabilities = {
        1: FixedEventProbability.from_params(1 - 0.95),
        2: FixedEventProbability.from_params(1 - 0.95),
        3: FixedEventProbability.from_params(1 - 0.95),
        4: FixedEventProbability.from_params(1 - 0.95),
        5: FixedEventProbability.from_params(1 - 0.95),
    }
    return edges, reliabilities


@pytest.fixture
def rbd3_koon1() -> NonRepairableRBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(5)=2
    """
    edges, reliabilities = rbd3_nodes_edges_components()
    k = {5: 2}
    return NonRepairableRBD(edges, reliabilities, k)


@pytest.fixture
def rbd3_koon2() -> NonRepairableRBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(2)=2
    """
    edges, reliabilities = rbd3_nodes_edges_components()
    k = {2: 2}
    return NonRepairableRBD(edges, reliabilities, k)


@pytest.fixture
def rbd3_koon3() -> NonRepairableRBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(2)=k(5)=2
    """
    edges, reliabilities = rbd3_nodes_edges_components()
    k = {2: 2, 5: 2}
    return NonRepairableRBD(edges, reliabilities, k)


@pytest.fixture
def rbd_koon_parallel_args() -> dict:
    """
    A s-3-1-t NonRepairableRBD.
    Returns the {nodes, edges, reliabilities} dict, useful so k(4) can be
    changed.
    """
    edges = [
        ("s", 1),
        ("s", 2),
        ("s", 3),
        (1, "v"),
        (2, "v"),
        (3, "v"),
        ("v", "t"),
    ]
    reliabilities = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        "v": FixedEventProbability.from_params(1 - 0.85),
    }
    return {"edges": edges, "reliabilities": reliabilities}


@pytest.fixture
def rbd_koon_composite_args() -> dict:
    """
    A s-2-1-2-1-t NonRepairableRBD.
    Returns the {nodes, edges, reliabilities} dict, useful so k("v1")
    and k("v2") can be changed.
    """
    edges = [
        ("s", "a1"),
        ("s", "a2"),
        ("a1", "v1"),
        ("a2", "v1"),
        ("v1", "b1"),
        ("v1", "b2"),
        ("b1", "v2"),
        ("b2", "v2"),
        ("v2", "t"),
    ]
    reliabilities = {
        "a1": FixedEventProbability.from_params(1 - 0.85),
        "a2": FixedEventProbability.from_params(1 - 0.8),
        "v1": FixedEventProbability.from_params(1 - 0.9),
        "b1": FixedEventProbability.from_params(1 - 0.85),
        "b2": FixedEventProbability.from_params(1 - 0.85),
        "v2": FixedEventProbability.from_params(1 - 0.8),
    }
    return {"edges": edges, "reliabilities": reliabilities}


@pytest.fixture
def rbd_koon_simple() -> NonRepairableRBD:
    edges = [("s", 1), (1, 2), ("s", 2), (2, "t")]
    reliabilities = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
    }
    return NonRepairableRBD(edges, reliabilities, {2: 2})


@pytest.fixture
def rbd_koon_nonminimal_args() -> dict:
    """NonRepairableRBD that can easily trap an algorithm into
    including a non-minimal path-set."""
    edges = [("s", 1), (1, 2), (2, "t"), (1, 3), (3, 4), (4, 5), (5, 2)]
    reliabilities = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
        5: FixedEventProbability.from_params(1 - 0.85),
    }
    return {"edges": edges, "reliabilities": reliabilities}


@pytest.fixture
def rbd_double_parallel_args() -> dict:
    """NonRepairableRBD arguments that make for a 1-2-1-2-1
    NonRepairableRBD."""
    edges = [
        ("s", 1),
        ("s", 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (3, 5),
        (4, "t"),
        (5, "t"),
    ]
    reliabilities = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
        5: FixedEventProbability.from_params(1 - 0.85),
    }
    return {"edges": edges, "reliabilities": reliabilities}
