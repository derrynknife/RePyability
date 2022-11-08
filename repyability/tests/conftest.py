"""
Contains the pytest fixtures used by many of the tests. They all define
and return RBDs.
"""

import pytest
import surpyval as surv

from repyability.rbd.rbd import RBD
from repyability.tests.fixed_probability import FixedProbabilityFitter


@pytest.fixture
def rbd_series() -> RBD:
    """A simple RBD with three intermediate nodes in series."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    components = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd_parallel() -> RBD:
    """A simple RBD with three intermediate nodes in parallel."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]
    components = {
        2: FixedProbabilityFitter.from_params(0.8),
        3: FixedProbabilityFitter.from_params(0.9),
        4: FixedProbabilityFitter.from_params(0.85),
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd1() -> RBD:
    """Example 6.10 from Modarres & Kaminskiy."""
    qp = 0.03
    qv = 0.01
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
    components = {
        "pump1": FixedProbabilityFitter.from_params(1 - qp),
        "pump2": FixedProbabilityFitter.from_params(1 - qp),
        "valve": FixedProbabilityFitter.from_params(1 - qv),
    }

    return RBD(nodes, components, edges)


@pytest.fixture
def rbd2() -> RBD:
    edges = [(1, 2), (2, 3), (2, 4), (4, 7), (3, 5), (5, 6), (6, 7), (7, 8)]
    nodes = {
        1: "input_node",
        8: "output_node",
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
    }
    components = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
        5: surv.Weibull.from_params([15, 1.2]),
        6: surv.Weibull.from_params([80, 10]),
        7: [
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
        ],
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd3() -> RBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    """
    nodes = {0: "input_node", 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: "output_node"}
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
    components = {
        1: FixedProbabilityFitter.from_params(0.95),
        2: FixedProbabilityFitter.from_params(0.95),
        3: FixedProbabilityFitter.from_params(0.95),
        4: FixedProbabilityFitter.from_params(0.95),
        5: FixedProbabilityFitter.from_params(0.95),
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd_repeated_component_parallel() -> RBD:
    """Basically rbd_parallel with a repeated component (component 2)."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: 2, 6: "output_node"}
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)]
    components = {
        2: FixedProbabilityFitter.from_params(0.8),
        3: FixedProbabilityFitter.from_params(0.9),
        4: FixedProbabilityFitter.from_params(0.85),
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd_repeated_component_series() -> RBD:
    """A simple RBD with three intermediate nodes in series."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 2, 5: "output_node"}
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    components = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd_repeated_component_composite() -> RBD:
    """
    An RBD with three intermediate nodes, two of them a repeated component.
    """
    nodes = {1: "input_node", 2: 2, 3: 2, 4: 3, 5: "output_node"}
    edges = [(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)]
    components = {
        2: FixedProbabilityFitter.from_params(0.8),
        3: FixedProbabilityFitter.from_params(0.5),
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd_series_koon() -> RBD:
    """
    Series RBD w/ three intermediate nodes, the middle one with k=2.
    i.e. this system doesn't work at all.
    """
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    components = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
    }
    k = {3: 2}
    return RBD(nodes, components, edges, k)


@pytest.fixture
def rbd1_koon() -> RBD:
    """Example 6.10 from Modarres & Kaminskiy, w/ valve k=2."""
    qp = 0.03
    qv = 0.01
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
    components = {
        "pump1": FixedProbabilityFitter.from_params(1 - qp),
        "pump2": FixedProbabilityFitter.from_params(1 - qp),
        "valve": FixedProbabilityFitter.from_params(1 - qv),
    }
    k = {"valve": 2}
    return RBD(nodes, components, edges, k)


@pytest.fixture
def rbd2_koon() -> RBD:
    """rbd2 but k of node 7 is =2."""
    edges = [(1, 2), (2, 3), (2, 4), (4, 7), (3, 5), (5, 6), (6, 7), (7, 8)]
    nodes = {
        1: "input_node",
        8: "output_node",
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
    }
    components = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
        5: surv.Weibull.from_params([15, 1.2]),
        6: surv.Weibull.from_params([80, 10]),
        7: [
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
        ],
    }
    k = {7: 2}
    return RBD(nodes, components, edges, k)


def rbd3_nodes_edges_components():
    nodes = {0: "input_node", 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: "output_node"}
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
    components = {
        1: FixedProbabilityFitter.from_params(0.95),
        2: FixedProbabilityFitter.from_params(0.95),
        3: FixedProbabilityFitter.from_params(0.95),
        4: FixedProbabilityFitter.from_params(0.95),
        5: FixedProbabilityFitter.from_params(0.95),
    }
    return nodes, edges, components


@pytest.fixture
def rbd3_koon1() -> RBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(5)=2
    """
    nodes, edges, components = rbd3_nodes_edges_components()
    k = {5: 2}
    return RBD(nodes, components, edges, k)


@pytest.fixture
def rbd3_koon2() -> RBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(2)=2
    """
    nodes, edges, components = rbd3_nodes_edges_components()
    k = {2: 2}
    return RBD(nodes, components, edges, k)


@pytest.fixture
def rbd3_koon3() -> RBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(2)=k(5)=2
    """
    nodes, edges, components = rbd3_nodes_edges_components()
    k = {2: 2, 5: 2}
    return RBD(nodes, components, edges, k)
