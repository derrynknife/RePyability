"""
Tests NonRepairableRBD.is_analytically_solvable() and get_non_analytic_nodes().

An RBD is solvable without simulation iff no node's reliability is
simulated: a Kaplan-Meier fit to simulated lifetimes. A standby or
load-sharing arrangement is simulated only when it has no closed form or
numerical convolution (two units of three needed working, say); a cold
spare for one unit is a convolution, and does not count.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repeated_node import RepeatedNode
from repyability.rbd.standby_node import StandbyModel

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
    # A repeated *component* (one component drawn in several places) is
    # solved exactly, like any other.
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


# --- Standby nodes: only simulated ones count --------------------------------


def test_a_convolved_standby_is_analytic(rbd2: NonRepairableRBD):
    # rbd2's node 7 is a cold spare chain for one unit: a numerical
    # convolution of the units' lives, not a simulation.
    assert rbd2.is_analytically_solvable()
    assert rbd2.get_non_analytic_nodes() == {}


def test_a_convolved_standby_is_analytic_under_k_out_of_n(
    rbd2_koon: NonRepairableRBD,
):
    assert rbd2_koon.is_analytically_solvable()


def simulated_standby_rbd() -> NonRepairableRBD:
    # Two units needed of three, with a cold spare: no closed form or
    # convolution, so the node's reliability is simulated.
    unit = surv.Weibull.from_params([5, 1.1])
    return NonRepairableRBD(
        [(1, 7), (7, 8)],
        {7: StandbyModel([unit] * 3, k=2, n_sims=500, seed=1)},
    )


def test_a_simulated_standby_is_non_analytic():
    rbd = simulated_standby_rbd()
    assert not rbd.is_analytically_solvable()
    assert rbd.get_non_analytic_nodes() == {7: "StandbyModel"}


# --- structure_check wiring ------------------------------------------------


def test_structure_check_fields(rbd_series: NonRepairableRBD):
    assert rbd_series.structure_check["is_analytically_solvable"] is True
    assert rbd_series.structure_check["non_analytic_nodes"] == {}

    rbd = simulated_standby_rbd()
    assert rbd.structure_check["is_analytically_solvable"] is False
    assert rbd.structure_check["non_analytic_nodes"] == {7: "StandbyModel"}
