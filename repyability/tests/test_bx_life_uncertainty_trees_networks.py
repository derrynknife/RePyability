"""``time_to_reliability_uncertainty`` and ``bx_life_uncertainty`` of a
fault tree and a network: a B-life with its band, from the draws
``sf_uncertainty`` makes, each draw's time found by inverting its exact
reliability."""

import json

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv

from repyability import FaultTree, Network, NonRepairableRBD

W = surv.Weibull.from_params
E = surv.Exponential.from_params
X = 40.0


def fit(scale=100.0, shape=2.0):
    times = W([scale, shape]).qf(np.linspace(0.025, 0.975, 20))
    return surv.Weibull.fit(times)


def tree(**events):
    pump = fit()
    return FaultTree(
        {
            "top": ("or", ["no flow", "valve"]),
            "no flow": ("and", ["pump 1", "pump 2"]),
        },
        events or {"pump 1": pump, "pump 2": pump, "valve": fit(300.0, 1.5)},
    )


def series(models):
    return Network({"a": (0, 1, models["a"]), "b": (1, 2, models["b"])}, 0, 2)


def series_rbd(models):
    return NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {k: models[k] for k in "ab"}
    )


def same(one, other):
    assert one.n_draws == other.n_draws and one.band == other.band
    np.testing.assert_array_equal(one.samples, other.samples)
    np.testing.assert_array_equal(one.nominal, other.nominal)


# -- a tree ------------------------------------------------------------


def test_a_tree_with_no_uncertain_element_gives_its_b_life_every_draw():
    model = W([100.0, 2.0])
    plain = tree(**{"pump 1": model, "pump 2": model, "valve": W([300, 1.5])})
    given = {("pump 1", "pump 2"): [model], "valve": [W([300, 1.5])]}
    result = plain.bx_life_uncertainty(10, given, n_draws=4, seed=0)
    # The tree's own value and its diagram's (which the draws are of) are
    # two exact evaluations, equal to their rounding.
    np.testing.assert_allclose(
        result.samples, np.full(4, plain.bx_life(10)), rtol=1e-14
    )
    np.testing.assert_allclose(result.nominal, plain.bx_life(10), rtol=1e-14)
    assert np.all(result.samples == result.samples[0])
    assert result.nominal == plain.to_rbd().bx_life(10)


@pytest.mark.parametrize("sampling", ["random", "sobol"])
def test_a_tree_s_result_is_its_diagram_s(sampling):
    fault = tree()
    for given in (None, {("pump 1", "pump 2"): "fit"}):
        ours = fault.time_to_reliability_uncertainty(
            0.9, given, n_draws=16, seed=3, sampling=sampling
        )
        theirs = fault.to_rbd().time_to_reliability_uncertainty(
            0.9, given, n_draws=16, seed=3, sampling=sampling
        )
        same(ours, theirs)
    same(
        fault.bx_life_uncertainty(10, n_draws=8, seed=1),
        fault.to_rbd().bx_life_uncertainty(10, n_draws=8, seed=1),
    )


def test_a_tree_with_a_common_cause_group_is_its_diagram_s():
    from repyability import BetaFactor, CCFGroup

    unit = E([0.01])
    fault = FaultTree(
        {"top": ("and", ["a", "b"])},
        {"a": unit, "b": unit},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    )
    given = {("a", "b"): {"failure_rate": st.uniform(0.008, 0.004)}}
    same(
        fault.bx_life_uncertainty(10, given, n_draws=8, seed=2),
        fault.to_rbd().bx_life_uncertainty(10, given, n_draws=8, seed=2),
    )


def test_several_levels_come_from_the_same_draws_as_each_alone():
    for model, call in (
        (tree(), "bx_life_uncertainty"),
        (series({"a": fit(), "b": fit(90.0, 1.5)}), "bx_life_uncertainty"),
    ):
        several = getattr(model, call)([5, 10, 20], n_draws=12, seed=4)
        assert several.samples.shape == (12, 3)
        assert several.nominal.shape == (3,)
        for j, level in enumerate((5, 10, 20)):
            alone = getattr(model, call)(level, n_draws=12, seed=4)
            np.testing.assert_array_equal(several.samples[:, j], alone.samples)
            assert several.nominal[j] == alone.nominal


def test_the_b_life_is_the_time_to_the_reliability_it_leaves():
    for model in (tree(), series({"a": fit(), "b": fit()})):
        b10 = model.bx_life_uncertainty(10, n_draws=6, seed=1)
        t90 = model.time_to_reliability_uncertainty(0.9, n_draws=6, seed=1)
        same(b10, t90)
        np.testing.assert_allclose(b10.nominal, model.bx_life(10), rtol=1e-14)


def test_a_tree_refuses_what_bx_life_refuses():
    fixed = FaultTree({"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2})
    dynamic = FaultTree(
        {"top": ("pand", ["a", "b"])}, {"a": W([100, 2]), "b": W([100, 2])}
    )
    negated = FaultTree(
        {
            "top": ("or", ["ab", "nc"]),
            "ab": ("and", ["a", "b"]),
            "nc": ("and", ["not a", "c"]),
            "not a": ("not", ["a"]),
        },
        {"a": W([100, 2]), "b": W([100, 2]), "c": W([100, 2])},
    )
    for bad in (fixed, dynamic, negated):
        with pytest.raises((NotImplementedError, ValueError)) as one:
            bad.bx_life(10)
        report = bad.analysis_routes()
        for name, level in (
            ("bx_life_uncertainty", 10),
            ("time_to_reliability_uncertainty", 0.9),
        ):
            with pytest.raises(type(one.value)) as other:
                getattr(bad, name)(level, {"a": "fit"}, n_draws=2)
            # Refused as bx_life is, and the report gives the message.
            assert report[name].route == report["bx_life"].route == "refused"
            assert report[name].reason == str(other.value)


def test_a_tree_checks_its_levels():
    with pytest.raises(ValueError, match="percentage"):
        tree().bx_life_uncertainty(100, n_draws=2)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        tree().time_to_reliability_uncertainty(1.0, n_draws=2)


# -- a network -----------------------------------------------------------


def test_a_network_with_no_uncertain_element_gives_its_b_life_every_draw():
    models = {"a": W([100.0, 2.0]), "b": W([90.0, 1.5])}
    net = series(models)
    result = net.bx_life_uncertainty(
        10, {"a": [models["a"]], "b": [models["b"]]}, n_draws=3, seed=0
    )
    np.testing.assert_array_equal(result.samples, np.full(3, net.bx_life(10)))
    assert result.nominal == net.bx_life(10)


@pytest.mark.parametrize("sampling", ["random", "sobol"])
def test_a_series_network_is_the_series_diagram(sampling):
    models = {"a": fit(), "b": fit(80.0, 1.7)}
    for given in (None, {"a": {"alpha": st.uniform(80, 40)}}):
        same(
            series(models).time_to_reliability_uncertainty(
                0.8, given, n_draws=16, seed=5, sampling=sampling
            ),
            series_rbd(models).time_to_reliability_uncertainty(
                0.8, given, n_draws=16, seed=5, sampling=sampling
            ),
        )
    same(
        series(models).bx_life_uncertainty(10, n_draws=8, seed=1),
        series_rbd(models).bx_life_uncertainty(10, n_draws=8, seed=1),
    )


def test_a_bridge_draw_is_the_bridge_with_the_drawn_models():
    from repyability.tests.test_network import bridge

    base = {n: W([100.0 + 20 * i, 1.5]) for i, n in enumerate("abcde")}
    choices = {"a": [W([80.0, 1.5]), W([140.0, 2.5])]}
    result = bridge(base).bx_life_uncertainty(10, choices, n_draws=20, seed=6)
    built = [bridge({**base, "a": a}).bx_life(10) for a in choices["a"]]
    for value in result.samples:
        assert min(abs(value - b) for b in built) < 1e-9 * value


def test_a_network_refuses_what_bx_life_refuses():
    fixed = Network(
        {"a": (0, 1, surv.FixedEventProbability.from_params(0.1))}, 0, 1
    )
    with pytest.raises(ValueError) as one:
        fixed.bx_life(10)
    for name, level in (
        ("bx_life_uncertainty", 10),
        ("time_to_reliability_uncertainty", 0.9),
    ):
        with pytest.raises(type(one.value)) as other:
            getattr(fixed, name)(level, {"a": "fit"})
        assert str(other.value) == str(one.value)
        assert fixed.analysis_routes()[name].reason == str(one.value)
    with pytest.raises(ValueError, match="percentage"):
        series({"a": fit(), "b": fit()}).bx_life_uncertainty(0, n_draws=2)


# -- both ----------------------------------------------------------------


def test_the_band_says_what_it_is():
    for model, one, other in (
        (tree(), ("pump 1", "pump 2"), "valve"),
        (series({"a": fit(), "b": fit()}), "a", "b"),
    ):
        by_fit = model.bx_life_uncertainty(10, {one: "fit"}, n_draws=4, seed=0)
        assert by_fit.band == "confidence"
    pump = E([0.002])
    exp_tree = FaultTree(
        {"top": ("and", ["p1", "p2"])}, {"p1": pump, "p2": pump}
    )
    given = {("p1", "p2"): {"failure_rate": st.uniform(0.001, 0.002)}}
    assert exp_tree.bx_life_uncertainty(10, given, n_draws=4, seed=0).band == (
        "given"
    )
    mixed_tree = FaultTree(
        {"top": ("and", ["p1", "q"])}, {"p1": pump, "q": fit()}
    )
    mixed = {
        "p1": {"failure_rate": st.uniform(0.001, 0.002)},
        "q": "fit",
    }
    assert mixed_tree.bx_life_uncertainty(
        10, mixed, n_draws=4, seed=0
    ).band == ("mixed")
    net = Network({"a": (0, 1, pump), "b": (1, 2, fit())}, 0, 2)
    assert (
        net.bx_life_uncertainty(
            10,
            (
                mixed | {"a": mixed["p1"]}
                if False
                else {"a": mixed["p1"], "b": "fit"}
            ),
            n_draws=4,
            seed=0,
        ).band
        == "mixed"
    )


def test_the_results_go_through_json():
    for result in (
        tree().bx_life_uncertainty([5, 10], n_draws=4, seed=0),
        series({"a": fit(), "b": fit()}).bx_life_uncertainty(
            10, n_draws=4, seed=0
        ),
    ):
        again = json.loads(json.dumps(result.to_dict()))
        assert again["n_draws"] == 4 and again["band"] == "confidence"


def test_the_reports_route_them_as_simulated():
    for model in (tree(), series({"a": fit(), "b": fit()})):
        report = model.analysis_routes()
        for name in ("bx_life_uncertainty", "time_to_reliability_uncertainty"):
            assert report[name].route == "simulated"
