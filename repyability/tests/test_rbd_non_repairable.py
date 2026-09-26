import threading

import numpy as np
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


def _within(seconds, func):
    """``func()``'s result, failing (rather than hanging the suite) if it
    does not return within ``seconds``."""
    result: list = []
    worker = threading.Thread(target=lambda: result.append(func()))
    worker.daemon = True
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"did not return within {seconds} s"
    return result[0]


@pytest.mark.parametrize("path", ["batched", "event loop"])
def test_a_system_that_cannot_fail_has_infinite_lifetimes(path):
    # An edge joins the input straight to the output, so the system works
    # even after every node has failed. A zero-inflated node cannot be
    # batched, so it sends random() down the one-at-a-time event loop,
    # which used to wait for ever for a system failure.
    unit = surv.Weibull.from_params([100, 2])
    if path == "event loop":
        unit = surv.Weibull.from_params([100, 2], f0=0.1)
    rbd = NonRepairableRBD([("s", "a"), ("a", "t"), ("s", "t")], {"a": unit})
    assert rbd.sf(50) == 1.0
    lifetimes = _within(20, lambda: rbd.random(5, seed=0))
    assert np.all(np.isinf(lifetimes))
    assert np.isinf(_within(20, lambda: rbd.mean(20, seed=0)))
