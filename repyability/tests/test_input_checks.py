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


# -- the 0.14 persona review --------------------------------------------------


def test_counts_of_lifetimes_are_checked():
    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    for bad in (0, -1, 1.5, True, "10"):
        with pytest.raises(ValueError, match="mc_samples must be a positive"):
            rbd.mean(method="simulate", mc_samples=bad, seed=1)
        with pytest.raises(ValueError, match="size must be a positive"):
            rbd.random(bad, seed=1)
    with pytest.raises(ValueError, match="mc_samples must be at least 2"):
        rbd.mean_time_to_failure_interval(mc_samples=1, seed=1)
    with pytest.raises(ValueError, match="mc_samples must be at least 4"):
        rbd.mean_time_to_failure_interval(
            mc_samples=2, seed=1, antithetic=True
        )
    interval = rbd.mean_time_to_failure_interval(mc_samples=2, seed=1)
    assert interval.method == "simulated"
    assert np.isfinite(interval.standard_error)


def test_simulation_options_on_an_exact_call_are_value_errors():
    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    with pytest.raises(ValueError, match="only with method='simulate'"):
        rbd.mean(mc_samples=100)
    with pytest.raises(ValueError, match="seed only with method='simulate'"):
        rbd.mean(seed=1)


def test_a_misspelt_time_keyword_is_a_type_error():
    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    with pytest.raises(TypeError, match="unexpected keyword argument 't'"):
        rbd.sf(t=50)
    with pytest.raises(ValueError, match="x is required: this RBD"):
        rbd.sf()
    assert rbd.sf(x=50.0) == rbd.sf(50.0)


def test_the_held_nodes_must_be_a_collection():
    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    with pytest.raises(TypeError, match="working_nodes takes a collection"):
        rbd.sf(50.0, working_nodes=0)
    with pytest.raises(TypeError, match="broken_nodes takes a collection"):
        rbd.birnbaum_importance(50.0, broken_nodes=1.5)
    repairable = RepairableRBD(
        PARALLEL, {n: {"reliability": W, "repairability": E} for n in "ab"}
    )
    with pytest.raises(TypeError, match="working_nodes takes a collection"):
        repairable.mission_availability(0, 100)
    with pytest.raises(TypeError, match="working_nodes takes a collection"):
        repairable.point_availability(10.0, working_nodes=1.5)
    assert rbd.sf(50.0, working_nodes=["a"]) == 1.0
    assert rbd.sf(50.0, working_nodes={"a"}) == 1.0


def test_bx_life_takes_a_percentage():
    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    with pytest.raises(TypeError, match="x must be numbers, got '10'"):
        rbd.bx_life("10")
    with pytest.raises(TypeError, match="x must be numbers"):
        rbd.bx_life(True)
    with pytest.raises(ValueError, match="percentage in \\(0, 100\\)"):
        rbd.bx_life(0.0)
    assert rbd.sf(rbd.bx_life(10)) == pytest.approx(0.9)
    # A fixed survivor in parallel keeps the reliability at 0.9 for good.
    kept = NonRepairableRBD(
        PARALLEL, {"a": surv.FixedEventProbability.from_params(0.1), "b": W}
    )
    with pytest.raises(ValueError, match="never reached") as error:
        kept.bx_life(20)
    assert "never falls below 0.9" in str(error.value)
    assert kept.bx_life(5) == pytest.approx(kept.time_to_reliability(0.95))


def test_from_dict_needs_a_type():
    from repyability import RBD

    with pytest.raises(ValueError, match='has no "type"'):
        RBD.from_dict({"edges": [["s", "a"], ["a", "t"]]})
    with pytest.raises(ValueError, match='has no "type"'):
        NonRepairableRBD.from_dict({})
    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    assert RBD.from_dict(rbd.to_dict()).sf(50.0) == rbd.sf(50.0)


def test_path_set_probabilities_pair_with_the_path_sets():
    edges = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
    rbd = NonRepairableRBD(edges, {"a": W, "b": W, "c": E})
    probabilities = {"a": 0.9, "b": 0.8, "c": 0.5}
    found = rbd.path_set_probabilities(probabilities)
    expected = [
        np.prod([probabilities[n] for n in path])
        for path in rbd.minimal_path_sets()
    ]
    assert np.allclose(found, expected)
    assert [sorted(p) for p in rbd.minimal_path_sets()] == [
        ["a", "c"],
        ["b", "c"],
    ]
    with pytest.raises(ValueError, match="none given for 'b', 'c'"):
        rbd.path_set_probabilities({"a": 0.9})
    with pytest.raises(TypeError, match="not a time"):
        rbd.path_set_probabilities(50.0)


def test_an_irrelevant_node_behind_a_junction_is_found():
    from repyability import RBD, PerfectReliability

    # x is in parallel with a path through the junction j alone, which
    # never fails: x is in no minimal path set and cannot matter.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "j"), ("a", "x"), ("j", "t"), ("x", "t")],
        {"a": W, "x": E, "j": PerfectReliability},
    )
    assert rbd.find_irrelevant_components() == {"x"}
    assert rbd.structure_check["has_irrelevant_nodes"]
    assert rbd.structure_check["irrelevant_nodes"] == {"x"}
    assert rbd.birnbaum_importance(50.0) == {"a": pytest.approx(1.0), "x": 0.0}
    assert rbd.minimal_path_sets() == [frozenset({"a"})]
    # A junction is no component: never listed. With a vote it is relevant
    # to the structure, and every member is relevant.
    vote = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "j"), ("b", "j"), ("j", "t")],
        {"a": W, "b": W, "j": PerfectReliability},
        k={"j": 2},
    )
    assert vote.find_irrelevant_components() == set()
    assert not vote.structure_check["has_irrelevant_nodes"]
    assert RBD(
        [("s", "a"), ("a", "t"), ("a", "b"), ("b", "t")]
    ).find_irrelevant_components() == {"b"}


def test_a_group_s_levers_name_its_uncertainty():
    import scipy.stats as st

    from repyability import BetaFactor, CCFGroup, FaultTree

    rbd = NonRepairableRBD(
        PARALLEL,
        {"a": W, "b": W},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    )
    group = rbd.ccf_groups[0]
    assert {(lever.key, lever.name) for lever in rbd.levers()} >= {
        (("a", "b"), "ccf_beta")
    }
    by_lever = {
        ("a", "b"): {"alpha": st.uniform(90, 20), "ccf_beta": st.beta(2, 18)}
    }
    by_group = {
        ("a", "b"): {"alpha": st.uniform(90, 20)},
        group: {"beta": st.beta(2, 18)},
    }
    for diagram in (rbd, FaultTree.from_rbd(rbd)):
        ours = diagram.sf_uncertainty(50.0, by_lever, n_draws=40, seed=1)
        theirs = diagram.sf_uncertainty(50.0, by_group, n_draws=40, seed=1)
        assert np.array_equal(ours.samples, theirs.samples)
    own = rbd.sf_uncertainty(
        50.0, {("a", "b"): {"ccf_beta": st.beta(2, 18)}}, n_draws=40, seed=1
    )
    assert np.array_equal(
        own.samples,
        rbd.sf_uncertainty(
            50.0, {group: {"beta": st.beta(2, 18)}}, n_draws=40, seed=1
        ).samples,
    )
    with pytest.raises(ValueError, match="given an uncertainty twice"):
        rbd.sf_uncertainty(
            50.0,
            {
                ("a", "b"): {"ccf_beta": st.beta(2, 18)},
                group: {"beta": st.beta(2, 18)},
            },
            n_draws=4,
            seed=1,
        )
    with pytest.raises(ValueError, match=r"\['ccf_gamma'\] are not levers"):
        rbd.sf_uncertainty(
            50.0,
            {("a", "b"): {"ccf_gamma": st.beta(2, 18)}},
            n_draws=4,
            seed=1,
        )
    repairable = RepairableRBD(
        PARALLEL,
        {n: {"reliability": E, "repairability": E} for n in "ab"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    assert np.array_equal(
        repairable.mean_availability_uncertainty(
            {("a", "b"): {"ccf_beta": st.beta(2, 18)}}, n_draws=10, seed=1
        ).samples,
        repairable.mean_availability_uncertainty(
            {repairable.ccf_groups[0]: {"beta": st.beta(2, 18)}},
            n_draws=10,
            seed=1,
        ).samples,
    )


def test_the_measures_at_zero_are_nan_without_a_warning():
    import warnings

    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in (
            "risk_achievement_worth",
            "fussell_vesely",
            "criticality_importance",
        ):
            found = getattr(rbd, name)(0.0)
            assert all(np.isnan(v) for v in found.values()), name
        assert np.isnan(rbd.fussell_vesely(0.0, method="rare_event")["a"])


def test_with_levers_names_the_kind_of_diagram():
    from repyability import FaultTree

    rbd = NonRepairableRBD(PARALLEL, {"a": W, "b": W})
    with pytest.raises(
        ValueError, match="no node with a model or common-cause group\\. "
    ):
        rbd.with_levers({("zz", "alpha"): 5.0})
    with pytest.raises(
        ValueError, match="no event with a model or common-cause group\\. "
    ):
        FaultTree.from_rbd(rbd).with_levers({("zz", "alpha"): 5.0})
    repairable = RepairableRBD(
        PARALLEL, {n: {"reliability": W, "repairability": E} for n in "ab"}
    )
    with pytest.raises(ValueError, match="or the repair crews \\(None\\)"):
        repairable.with_levers({("zz", "reliability.alpha"): 5.0})


def test_a_non_repairable_prints_its_models_and_costs():
    from repyability import NonRepairable

    unit = NonRepairable(W)
    assert repr(unit) == "NonRepairable(Weibull, replaced in ExactEventTime)"
    priced = NonRepairable(W, E, planned_cost=10.0, unplanned_cost=100.0)
    assert repr(priced) == (
        "NonRepairable(Weibull, replaced in Exponential, planned_cost=10, "
        "unplanned_cost=100)"
    )
