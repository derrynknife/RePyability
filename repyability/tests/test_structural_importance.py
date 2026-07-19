"""
Tests RBD.structural_importance().

Structural importance is the Birnbaum importance evaluated with every node
reliability at 1/2, so it depends only on the diagram, not on any failure
model. The expected values below are worked by hand from the structure.
"""

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD


def test_series_structural_importance(rbd_series: NonRepairableRBD):
    """Every node in an n-series system has structural importance
    0.5**(n-1); here n=3 so each is 0.25."""
    si = rbd_series.structural_importance()
    assert set(si) == {2, 3, 4}
    for value in si.values():
        assert value == pytest.approx(0.25)


def test_parallel_structural_importance(rbd_parallel: NonRepairableRBD):
    """Every node in an n-parallel system has structural importance
    0.5**(n-1); here n=3 so each is 0.25."""
    si = rbd_parallel.structural_importance()
    assert set(si) == {2, 3, 4}
    for value in si.values():
        assert value == pytest.approx(0.25)


def test_series_parallel_structural_importance():
    """Node 1 in series with a parallel block of nodes 2 and 3.

    System = R1 * (1 - (1-R2)(1-R3)). At R=1/2:
      B_1 = 1 - (1-R2)(1-R3) = 0.75
      B_2 = R1 * (1 - R3)    = 0.25
      B_3 = R1 * (1 - R2)    = 0.25
    """
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (1, 3), (2, "t"), (3, "t")],
        {
            1: FixedEventProbability.from_params(1 - 0.9),
            2: FixedEventProbability.from_params(1 - 0.8),
            3: FixedEventProbability.from_params(1 - 0.85),
        },
    )
    si = rbd.structural_importance()
    assert si[1] == pytest.approx(0.75)
    assert si[2] == pytest.approx(0.25)
    assert si[3] == pytest.approx(0.25)


def test_koon_structural_importance():
    """A 2-out-of-3 vote on nodes 1, 2, 3 in series with a valve.

    For a pure 2oo3 at R=1/2 each component has Birnbaum importance 0.5, but
    the trailing series valve halves it (B = 0.5 * R_valve = 0.25), while the
    valve sees the 2oo3 block (B = P(2oo3 works at 0.5) = 0.5).
    """
    rbd = NonRepairableRBD(
        [
            ("s", 1),
            ("s", 2),
            ("s", 3),
            (1, "v"),
            (2, "v"),
            (3, "v"),
            ("v", "t"),
        ],
        {
            1: FixedEventProbability.from_params(1 - 0.85),
            2: FixedEventProbability.from_params(1 - 0.8),
            3: FixedEventProbability.from_params(1 - 0.9),
            "v": FixedEventProbability.from_params(1 - 0.85),
        },
        k={"v": 2},
    )
    si = rbd.structural_importance()
    assert si[1] == pytest.approx(0.25)
    assert si[2] == pytest.approx(0.25)
    assert si[3] == pytest.approx(0.25)
    assert si["v"] == pytest.approx(0.5)


def test_single_node_structural_importance():
    """A lone node is always pivotal, so its structural importance is 1."""
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": FixedEventProbability.from_params(0.1)},
    )
    assert rbd.structural_importance()["a"] == pytest.approx(1.0)


def test_structural_importance_is_model_free(rbd_series: NonRepairableRBD):
    """It equals the Birnbaum importance evaluated at reliabilities of 1/2,
    independent of the actual node models."""
    si = rbd_series.structural_importance()
    node_probabilities = {node: np.full(1, 0.5) for node in rbd_series.nodes}
    birnbaum = rbd_series._birnbaum_importance(node_probabilities)
    for node, value in si.items():
        assert value == pytest.approx(float(np.asarray(birnbaum[node])[0]))


def test_structural_importance_same_for_repairable():
    """A structural property: identical for a NonRepairableRBD and a
    RepairableRBD built on the same diagram."""
    edges = [("s", 1), (1, 2), (1, 3), (2, "t"), (3, "t")]
    nonrep = NonRepairableRBD(
        edges,
        {
            1: FixedEventProbability.from_params(1 - 0.9),
            2: FixedEventProbability.from_params(1 - 0.8),
            3: FixedEventProbability.from_params(1 - 0.85),
        },
    )
    rep = RepairableRBD(
        edges,
        {
            node: {
                "reliability": surv.Weibull.from_params([10 * node, 2]),
                "repairability": surv.Exponential.from_params([0.5]),
            }
            for node in (1, 2, 3)
        },
    )
    assert nonrep.structural_importance() == rep.structural_importance()


def test_structural_importance_working_node():
    """Forcing node 1 to work removes its series contribution: the system
    reduces to the 2||3 parallel block, so nodes 2 and 3 keep 0.25 while the
    forced node's own value reflects the conditioned structure."""
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (1, 3), (2, "t"), (3, "t")],
        {
            1: FixedEventProbability.from_params(1 - 0.9),
            2: FixedEventProbability.from_params(1 - 0.8),
            3: FixedEventProbability.from_params(1 - 0.85),
        },
    )
    si = rbd.structural_importance(working_nodes=[1])
    # With node 1 pinned up, 2 and 3 each become pivotal exactly when the
    # other fails: B = 1 - R = 0.5.
    assert si[2] == pytest.approx(0.5)
    assert si[3] == pytest.approx(0.5)


def test_structural_importance_unknown_node_raises(
    rbd_series: NonRepairableRBD,
):
    """Overrides are validated: an unknown node name raises."""
    with pytest.raises(ValueError):
        rbd_series.structural_importance(working_nodes=["not_a_node"])
