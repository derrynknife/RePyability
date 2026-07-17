"""
Tests NonRepairableRBD.is_analytically_solvable() and get_non_analytic_nodes().

An RBD is analytically / BDD solvable iff every node's reliability is available
without Monte-Carlo simulation. Standby nodes (StandbyModel,
RepeatedStandbyNode) are simulation-based (a Kaplan-Meier fit to simulated
samples) and so make an RBD non-analytic.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repeated_node import RepeatedNode

# --- Analytically solvable RBDs --------------------------------------------


def test_analytic_series(rbd_series: NonRepairableRBD):
    assert rbd_series.is_analytically_solvable()
    assert rbd_series.get_non_analytic_nodes() == {}


def test_analytic_parallel_fixed(rbd_parallel: NonRepairableRBD):
    assert rbd_parallel.is_analytically_solvable()
    assert rbd_parallel.get_non_analytic_nodes() == {}


def test_analytic_rbd1(rbd1: NonRepairableRBD):
    assert rbd1.is_analytically_solvable()


def test_repeated_component_still_analytic(
    rbd_repeated_component_parallel: NonRepairableRBD,
):
    # A repeated *component* (shared node) is merged into one node and remains
    # analytically solvable.
    assert rbd_repeated_component_parallel.is_analytically_solvable()


def test_repeated_node_of_parametric_is_analytic():
    # A RepeatedNode wrapping a parametric distribution is analytic.
    edges = [(1, 2), (2, 3)]
    reliabilities = {
        2: RepeatedNode(
            surv.Weibull.from_params([5, 1.1]), repeats=2, kind="series"
        )
    }
    rbd = NonRepairableRBD(edges, reliabilities)
    assert rbd.is_analytically_solvable()


# --- Non-analytic (simulation-based) RBDs ----------------------------------


def test_non_analytic_standby(rbd2: NonRepairableRBD):
    # rbd2 contains a StandbyModel at node 7.
    assert not rbd2.is_analytically_solvable()
    assert rbd2.get_non_analytic_nodes() == {7: "StandbyModel"}


def test_non_analytic_standby_koon(rbd2_koon: NonRepairableRBD):
    # rbd2_koon also contains the StandbyModel at node 7.
    assert not rbd2_koon.is_analytically_solvable()
    assert 7 in rbd2_koon.get_non_analytic_nodes()


# --- structure_check wiring ------------------------------------------------


def test_structure_check_fields(
    rbd_series: NonRepairableRBD, rbd2: NonRepairableRBD
):
    assert rbd_series.structure_check["is_analytically_solvable"] is True
    assert rbd_series.structure_check["non_analytic_nodes"] == {}

    assert rbd2.structure_check["is_analytically_solvable"] is False
    assert rbd2.structure_check["non_analytic_nodes"] == {7: "StandbyModel"}
