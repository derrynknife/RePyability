"""
Contains the pytest fixtures used by many of the tests. They all define
and return NonRepairableRBDs.
"""

import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD


@pytest.fixture
def rbd_series() -> NonRepairableRBD:
    """A simple NonRepairableRBD with three intermediate nodes in series."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    reliability = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
    }
    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd_parallel() -> NonRepairableRBD:
    """A simple NonRepairableRBD with three intermediate nodes in parallel."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]
    reliability = {
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
    }
    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd1() -> NonRepairableRBD:
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
    reliability = {
        "pump1": FixedEventProbability.from_params(qp),
        "pump2": FixedEventProbability.from_params(qp),
        "valve": FixedEventProbability.from_params(qv),
    }

    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd2() -> NonRepairableRBD:
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
    reliability = {
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
    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd3() -> NonRepairableRBD:
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
    reliability = {
        1: FixedEventProbability.from_params(1 - 0.95),
        2: FixedEventProbability.from_params(1 - 0.95),
        3: FixedEventProbability.from_params(1 - 0.95),
        4: FixedEventProbability.from_params(1 - 0.95),
        5: FixedEventProbability.from_params(1 - 0.95),
    }
    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd_repeated_component_parallel() -> NonRepairableRBD:
    """Basically rbd_parallel with a repeated component (component 2)."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: 2, 6: "output_node"}
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)]
    reliability = {
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
    }
    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd_repeated_component_series() -> NonRepairableRBD:
    """A simple NonRepairableRBD with three intermediate nodes in series."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 2, 5: "output_node"}
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    reliability = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
    }
    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd_repeated_component_composite() -> NonRepairableRBD:
    """
    An NonRepairableRBD with three intermediate nodes,
    two of them a repeated component.
    """
    nodes = {1: "input_node", 2: 2, 3: 2, 4: 3, 5: "output_node"}
    edges = [(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)]
    reliability = {
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.5),
    }
    return NonRepairableRBD(nodes, reliability, edges)


@pytest.fixture
def rbd_series_koon() -> NonRepairableRBD:
    """
    Series NonRepairableRBD w/ three intermediate nodes, the middle
    one with k=2. i.e. this system doesn't work at all.
    """
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    reliability = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
    }
    k = {3: 2}
    return NonRepairableRBD(nodes, reliability, edges, k)


@pytest.fixture
def rbd1_koon() -> NonRepairableRBD:
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
    reliability = {
        "pump1": FixedEventProbability.from_params(qp),
        "pump2": FixedEventProbability.from_params(qp),
        "valve": FixedEventProbability.from_params(qv),
    }
    k = {"valve": 2}
    return NonRepairableRBD(nodes, reliability, edges, k)


@pytest.fixture
def rbd2_koon() -> NonRepairableRBD:
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
    reliability = {
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
    return NonRepairableRBD(nodes, reliability, edges, k)


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
    reliability = {
        1: FixedEventProbability.from_params(1 - 0.95),
        2: FixedEventProbability.from_params(1 - 0.95),
        3: FixedEventProbability.from_params(1 - 0.95),
        4: FixedEventProbability.from_params(1 - 0.95),
        5: FixedEventProbability.from_params(1 - 0.95),
    }
    return nodes, edges, reliability


@pytest.fixture
def rbd3_koon1() -> NonRepairableRBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(5)=2
    """
    nodes, edges, reliability = rbd3_nodes_edges_components()
    k = {5: 2}
    return NonRepairableRBD(nodes, reliability, edges, k)


@pytest.fixture
def rbd3_koon2() -> NonRepairableRBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(2)=2
    """
    nodes, edges, reliability = rbd3_nodes_edges_components()
    k = {2: 2}
    return NonRepairableRBD(nodes, reliability, edges, k)


@pytest.fixture
def rbd3_koon3() -> NonRepairableRBD:
    """
    Fig. 16.1 from "UNIT 16 RELIABILITY EVALUATION OF COMPLEX SYSTEMS" by
    ignou (https://egyankosh.ac.in/bitstream/123456789/35170/1/Unit-16.pdf).
    w/ k(2)=k(5)=2
    """
    nodes, edges, reliability = rbd3_nodes_edges_components()
    k = {2: 2, 5: 2}
    return NonRepairableRBD(nodes, reliability, edges, k)


@pytest.fixture
def rbd_koon_parallel_args() -> dict:
    """
    A s-3-1-t NonRepairableRBD.
    Returns the {nodes, edges, reliability} dict, useful so k(4) can be
    changed.
    """
    nodes = {"s": "input_node", 1: 1, 2: 2, 3: 3, "v": 4, "t": "output_node"}
    edges = [
        ("s", 1),
        ("s", 2),
        ("s", 3),
        (1, "v"),
        (2, "v"),
        (3, "v"),
        ("v", "t"),
    ]
    reliability = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
    }
    return {"nodes": nodes, "edges": edges, "reliability": reliability}


@pytest.fixture
def rbd_koon_composite_args() -> dict:
    """
    A s-2-1-2-1-t NonRepairableRBD.
    Returns the {nodes, edges, reliability} dict, useful so k("v1") and k("v2")
    can be changed.
    """
    nodes = {
        "s": "input_node",
        "a1": 1,
        "a2": 2,
        "v1": 3,
        "b1": 4,
        "b2": 5,
        "v2": 6,
        "t": "output_node",
    }
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
    reliability = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
        5: FixedEventProbability.from_params(1 - 0.85),
        6: FixedEventProbability.from_params(1 - 0.8),
    }
    return {"nodes": nodes, "edges": edges, "reliability": reliability}


@pytest.fixture
def rbd_koon_simple() -> NonRepairableRBD:
    nodes = {"s": "input_node", 1: 1, 2: 2, "t": "output_node"}
    edges = [("s", 1), (1, 2), ("s", 2), (2, "t")]
    reliability = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
    }
    return NonRepairableRBD(nodes, reliability, edges, {2: 2})


@pytest.fixture
def rbd_koon_nonminimal_args() -> dict:
    """NonRepairableRBD that can easily trap an algorithm into
    including a non-minimal path-set."""
    nodes = {
        "s": "input_node",
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        "t": "output_node",
    }
    edges = [("s", 1), (1, 2), (2, "t"), (1, 3), (3, 4), (4, 5), (5, 2)]
    reliability = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
        5: FixedEventProbability.from_params(1 - 0.85),
    }
    return {"nodes": nodes, "edges": edges, "reliability": reliability}


@pytest.fixture
def rbd_double_parallel_args() -> dict:
    """NonRepairableRBD arguments that make for a 1-2-1-2-1
    NonRepairableRBD."""
    nodes = {
        "s": "input_node",
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        "t": "output_node",
    }
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
    reliability = {
        1: FixedEventProbability.from_params(1 - 0.85),
        2: FixedEventProbability.from_params(1 - 0.8),
        3: FixedEventProbability.from_params(1 - 0.9),
        4: FixedEventProbability.from_params(1 - 0.85),
        5: FixedEventProbability.from_params(1 - 0.85),
    }
    return {"nodes": nodes, "edges": edges, "reliability": reliability}
