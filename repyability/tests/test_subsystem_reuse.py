"""One saved sub-system drawn at several nodes of a diagram: each node is
its own copy of it, with a simulation state of its own, so the diagram is
the one with the copies given."""

import copy

import numpy as np
import surpyval as surv

from repyability import RepairableRBD

E = surv.Exponential.from_params


def unit(rate, repair):
    return {
        "reliability": E([rate]),
        "repairability": E([repair]),
        "repair_cost": 100.0,
    }


def pair():
    return RepairableRBD(
        [("i", "p1"), ("i", "p2"), ("p1", "o"), ("p2", "o")],
        {"p1": unit(1e-3, 0.1), "p2": unit(1e-3, 0.1)},
        downtime_cost_rate=5.0,
    )


TWICE = [("s", "A"), ("A", "C"), ("C", "t")]


def test_one_sub_system_at_two_nodes_is_two_copies():
    sub = pair()
    shared = RepairableRBD(TWICE, {"A": sub, "C": sub})
    copied = RepairableRBD(TWICE, {"A": pair(), "C": copy.deepcopy(pair())})
    assert shared.components["A"] is sub
    assert shared.components["C"] is not sub
    # The pair: each pump up 1 / 1.01 of the time; the pair down when
    # both are, and the two pairs in series.
    a = 1.0 / 1.01
    pair_up = 1.0 - (1.0 - a) ** 2
    assert np.isclose(shared.mean_availability(), pair_up**2, rtol=1e-9)
    assert shared.mean_availability() == copied.mean_availability()
    x = shared.availability(2000.0, mc_samples=200, seed=3)
    y = copied.availability(2000.0, mc_samples=200, seed=3)
    np.testing.assert_array_equal(x.availability, y.availability)
    np.testing.assert_array_equal(x.uptimes, y.uptimes)
    cx = shared.cost(2000.0, mc_samples=200, seed=3, control_variate=False)
    cy = copied.cost(2000.0, mc_samples=200, seed=3, control_variate=False)
    assert cx.mean == cy.mean


def test_a_sub_system_within_a_sub_system_is_copied_too():
    inner = pair()
    middle = RepairableRBD([("i", "P"), ("P", "o")], {"P": inner})
    outer = RepairableRBD(TWICE, {"A": middle, "C": middle})
    first = outer.components["A"].components["P"]
    second = outer.components["C"].components["P"]
    assert first is inner and second is not inner
