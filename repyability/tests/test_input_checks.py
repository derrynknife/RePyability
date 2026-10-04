"""Bad inputs are refused with a message that says what to give instead
(#168, #169, #174, #176, #178), rather than a wrong answer or an internal
error."""

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    NonRepairableRBD,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
)

W = surv.Weibull.from_params([100.0, 2.0])
E = surv.Exponential.from_params([0.01])
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


@pytest.mark.parametrize("repeats", [-1, 0, 2.5, True, "2", None])
def test_a_repeated_node_needs_a_whole_number_of_copies(repeats):
    for kind in ("series", "parallel"):
        with pytest.raises(ValueError, match="repeats must be a whole"):
            RepeatedNode(E, repeats, kind)
    with pytest.raises(ValueError, match="repeats must be a whole"):
        RepeatedStandbyNode(E, repeats)


def test_whole_numbers_of_copies_are_taken():
    assert RepeatedNode(E, np.int64(3), "series").sf(100.0) == pytest.approx(
        float(E.sf(100.0)) ** 3
    )
    # A whole float, as numpy arithmetic gives, is a whole number.
    assert RepeatedNode(E, 3.0, "series").repeats == 3
    assert RepeatedStandbyNode(E, np.float64(2.0)).repeats == 2
    assert RepeatedNode(E, 1, "parallel").sf(100.0) == pytest.approx(
        float(E.sf(100.0))
    )


def test_a_repeat_of_a_repeat_names_the_component_to_repeat():
    edges = [("s", x) for x in "abc"] + [(x, "t") for x in "abc"]
    with pytest.raises(ValueError, match="point 'c' at 'a'"):
        NonRepairableRBD(edges, {"a": W, "b": "a", "c": "b"})
    # Repeats of each other name no model at all.
    with pytest.raises(ValueError, match="repeat each other"):
        NonRepairableRBD(
            [("s", "b"), ("s", "c"), ("b", "t"), ("c", "t")],
            {"b": "c", "c": "b"},
        )
    # Repeating the component itself is what works.
    fine = NonRepairableRBD(edges, {"a": W, "b": "a", "c": "a"})
    assert fine.sf(50.0) == pytest.approx(float(W.sf(50.0)))


@pytest.mark.parametrize("k", [1.5, "2", True, None])
def test_k_must_be_a_whole_number(k):
    with pytest.raises(ValueError, match="k for node 't' must be a whole"):
        NonRepairableRBD(PARALLEL, {"a": W, "b": W}, k={"t": k})


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_at_least_one(k):
    with pytest.raises(ValueError, match=f"node 't' has k = {k}"):
        NonRepairableRBD([("s", "a"), ("a", "t")], {"a": W}, k={"t": k})


@pytest.mark.parametrize("k", [np.int64(2), 2.0, np.float64(2.0)])
def test_a_numpy_or_whole_float_k_is_a_whole_number(k):
    two = NonRepairableRBD(PARALLEL, {"a": W, "b": W}, k={"t": k})
    assert two.sf(50.0) == pytest.approx(float(W.sf(50.0)) ** 2)


def test_a_standby_model_needs_one_unit_operating():
    with pytest.raises(ValueError, match="k .* must be a whole number"):
        StandbyModel([E, E], k=0)
    with pytest.raises(ValueError, match="k .* must be a whole number"):
        StandbyModel([E, E], k=1.5)


def test_an_empty_diagram_says_it_has_no_edges():
    with pytest.raises(ValueError, match="the diagram has no edges"):
        NonRepairableRBD([], {})


def test_a_cycle_built_anyway_is_not_evaluated():
    edges = [("s", "a"), ("a", "b"), ("b", "a"), ("b", "t")]
    rbd = NonRepairableRBD(edges, {"a": W, "b": W}, on_infeasible_rbd="ignore")
    with pytest.raises(ValueError, match="cycle .*'a', 'b'"):
        rbd.sf(50.0)


def test_no_input_node_is_said_plainly():
    # Every node on a cycle: none without an incoming edge.
    with pytest.raises(ValueError, match="there is no input node"):
        NonRepairableRBD(
            [("a", "b"), ("b", "a")],
            {"a": W, "b": W},
            on_infeasible_rbd="raise",
        )


def _repairable():
    spec = {
        "reliability": surv.Exponential.from_params([0.01]),
        "repairability": surv.Exponential.from_params([0.5]),
    }
    return RepairableRBD([("s", "a"), ("a", "t")], {"a": spec})


#: Every way to simulate a window of a RepairableRBD.
SIMULATIONS = {
    "availability": lambda rbd, t: rbd.availability(t, mc_samples=4, seed=1),
    "cost": lambda rbd, t: rbd.cost(t, mc_samples=4, seed=1),
    "simulate_timelines": lambda rbd, t: rbd.simulate_timelines(
        t, mc_samples=4, seed=1
    ),
    "simulate_chunk": lambda rbd, t: rbd.simulate_chunk(t, 0, 4, seed=1),
    "shards": lambda rbd, t: rbd.shards(t, 4, seed=1),
    "compare": lambda rbd, t: rbd.compare(rbd, t, mc_samples=4, seed=1),
}


@pytest.mark.parametrize("name", list(SIMULATIONS))
@pytest.mark.parametrize(
    "t", [-5.0, 0.0, 0, np.inf, np.nan, np.float64(-1.0), True, "10", None]
)
def test_a_simulation_needs_a_positive_finite_window(name, t):
    # A negative window gave negative uptimes, and none an availability of
    # nan (#174).
    with pytest.raises(ValueError, match="t_simulation must be a positive"):
        SIMULATIONS[name](_repairable(), t)


@pytest.mark.parametrize("name", list(SIMULATIONS))
def test_any_positive_number_is_a_window(name):
    rbd = _repairable()
    for t in (50, 50.0, np.float64(50.0), np.int64(50)):
        SIMULATIONS[name](rbd, t)


def test_no_spares_are_used_in_no_time_however_it_is_worked_out():
    rbd = _repairable()
    exact = rbd.spares_demand(0.0)["a"]
    simulated = rbd.spares_demand(0.0, method="simulate", mc_samples=4)["a"]
    np.testing.assert_array_equal(exact.probabilities, [1.0])
    np.testing.assert_array_equal(simulated.probabilities, [1.0])
