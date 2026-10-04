"""The exact system MTTF (#122): the area under the exact reliability,
against closed forms, the simulation, and the systems that never fail."""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import (
    BetaFactor,
    CCFGroup,
    NonRepairableRBD,
    PerfectReliability,
    RepeatedNode,
    StandbyModel,
)
from repyability.rbd import routes
from repyability.rbd._mean_lifetime import mean_lifetime, model_knots

E = surv.Exponential.from_params
W = surv.Weibull.from_params


def series(*models):
    names = list(range(len(models)))
    edges = [("s", 0)] + [(i, i + 1) for i in names[:-1]]
    return NonRepairableRBD(
        edges + [(names[-1], "t")], dict(enumerate(models))
    )


def parallel(*models):
    names = range(len(models))
    edges = [("s", i) for i in names] + [(i, "t") for i in names]
    return NonRepairableRBD(edges, dict(enumerate(models)))


def bridge(model):
    edges = [
        ("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"),
        ("a", "d"), ("c", "d"), ("c", "e"), ("b", "e"),
        ("d", "t"), ("e", "t"),
    ]  # fmt: skip
    return NonRepairableRBD(edges, {n: model for n in "abcde"})


CLOSED_FORMS = {
    "exponential": (series(E([0.01])), 100.0),
    "series of exponentials": (
        series(E([0.01]), E([0.02]), E([0.03])),
        1 / 0.06,
    ),
    "parallel pair": (parallel(E([0.01]), E([0.02])), 100 + 50 - 1 / 0.03),
    "2 of 3": (
        NonRepairableRBD(
            [("s", 1), ("s", 2), ("s", 3), (1, "t"), (2, "t"), (3, "t")],
            {n: E([0.01]) for n in (1, 2, 3)},
            k={"t": 2},
        ),
        1 / 0.03 + 1 / 0.02,
    ),
    # R = 2p^2 + 2p^3 - 5p^4 + 2p^5 with p = exp(-t / 100).
    "bridge": (bridge(E([0.01])), 100 * 49 / 60),
    "weibull": (series(W([100, 2])), 100 * math.gamma(1.5)),
    "weibull, shape 0.2": (series(W([100, 0.2])), 100 * math.gamma(6)),
    "offset": (series(W([100, 2], gamma=50)), 50 + 100 * math.gamma(1.5)),
    "dead on arrival": (
        series(W([100, 2], f0=0.1)),
        0.9 * 100 * math.gamma(1.5),
    ),
    "lognormal, a heavy tail": (
        series(surv.LogNormal.from_params([3, 3])),
        math.exp(3 + 4.5),
    ),
    "log-logistic, a heavier tail": (
        series(surv.LogLogistic.from_params([100, 1.2])),
        100 * (math.pi / 1.2) / math.sin(math.pi / 1.2),
    ),
    "gamma, early failures": (series(surv.Gamma.from_params([0.5, 0.01])), 50),
    "a fixed probability in series": (
        series(surv.FixedEventProbability.from_params(0.1), E([0.01])),
        90.0,
    ),
    "repeated in parallel": (
        series(RepeatedNode(E([0.01]), 3, "parallel")),
        100 + 50 + 100 / 3,
    ),
    "series of 60": (
        series(*[W([100, 2])] * 60),
        100 * math.gamma(1.5) / 60**0.5,
    ),
}


@pytest.mark.parametrize("name", CLOSED_FORMS)
def test_the_mttf_matches_closed_forms(name):
    rbd, expected = CLOSED_FORMS[name]
    assert rbd.mean() == pytest.approx(expected, rel=1e-9)
    assert rbd.mean_time_to_failure() == rbd.mean()


def test_a_nested_rbd_brings_its_exact_mttf():
    inner = parallel(E([0.01]), E([0.02]))
    outer = series(inner, E([0.01]))
    expected = integrate.quad(
        lambda t: float(inner.sf(t)) * math.exp(-0.01 * t), 0, np.inf
    )[0]
    assert outer.mean() == pytest.approx(expected, rel=1e-9)
    assert outer.node_mttf()[0] == pytest.approx(inner.mean(), rel=1e-12)


def test_node_models_are_integrated_as_their_reliability_says():
    unit = W([100, 2])
    convolved = StandbyModel([unit, unit])
    assert series(convolved).mean() == pytest.approx(
        2 * 100 * math.gamma(1.5), rel=1e-6
    )
    # A numerical reliability (two of three identical units operating)
    # integrates to the model's own mean.
    renewals = StandbyModel([unit] * 3, k=2)
    assert series(renewals).mean() == pytest.approx(renewals.mean(), rel=1e-6)


@pytest.mark.parametrize(
    "rbd",
    [
        series(W([100, 2], p=0.9)),
        parallel(PerfectReliability, E([0.01])),
        parallel(W([100, 2], p=0.999), E([0.01])),
        # A tail too heavy for a finite mean: shape 1 or less.
        series(surv.LogLogistic.from_params([100, 0.9])),
        series(surv.LogLogistic.from_params([100, 1.0])),
    ],
    ids=["some never fail", "perfect", "in parallel", "shape .9", "shape 1"],
)
def test_a_system_that_may_never_fail_has_an_infinite_mttf(rbd):
    assert rbd.mean() == math.inf


def test_the_simulation_agrees_within_its_error():
    rbd = bridge(W([100, 1.5]))
    interval = rbd.mean_time_to_failure_interval(mc_samples=100_000, seed=4)
    assert abs(interval.estimate - rbd.mean()) < 4 * interval.standard_error
    simulated = rbd.mean(method="simulate", mc_samples=100_000, seed=4)
    assert simulated == interval.estimate


def test_the_exact_mttf_does_not_touch_the_global_rng():
    rbd = bridge(W([100, 1.5]))
    before = np.random.get_state()[1].copy()
    assert rbd.mean() == rbd.mean()
    assert np.array_equal(np.random.get_state()[1], before)


def test_simulation_options_without_simulate_are_refused():
    rbd = series(E([0.01]))
    with pytest.raises(TypeError, match="mc_samples, seed"):
        rbd.mean(1000, seed=1)
    with pytest.raises(TypeError, match="tolerance"):
        rbd.mean_time_to_failure(tolerance=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rbd.mean()
        rbd.mean(method="simulate", mc_samples=100, seed=1)


def test_common_cause_groups_are_refused():
    unit = W([1000, 2])
    grouped = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        grouped.mean()
    # Nested, too; the simulation leaves the common cause out, as before.
    outer = series(grouped, E([0.001]))
    with pytest.raises(NotImplementedError, match="common-cause"):
        outer.mean()
    with pytest.raises(NotImplementedError, match="common-cause"):
        outer.node_mttf()
    report = outer.analysis_routes()
    assert report["mean"].route == routes.REFUSED
    assert report["mean"].nodes == (0,)
    assert report["node_mttf"].route == routes.REFUSED
    assert grouped.mean(method="simulate", mc_samples=1000, seed=1) > 0


def test_a_system_of_fixed_probabilities_has_no_mttf():
    fixed = series(surv.FixedEventProbability.from_params(0.1))
    with pytest.raises(ValueError, match="no lifetimes"):
        fixed.mean()
    with pytest.raises(ValueError, match="method"):
        series(E([0.01])).mean(method="quadrature")


def test_node_mttf_is_exact():
    unit = W([100, 2])
    rbd = NonRepairableRBD(
        [("s", "r"), ("r", "g"), ("g", "t")],
        {
            "r": RepeatedNode(unit, 3, "series"),
            "g": StandbyModel([unit] * 3, k=2),
        },
    )
    mttf = rbd.node_mttf()
    assert mttf["r"] == pytest.approx(100 * math.gamma(1.5) / 3**0.5)
    assert mttf["g"] == rbd.reliabilities["g"].mean()
    with pytest.raises(TypeError):
        rbd.node_mttf(mc_samples=100, seed=1)


def test_the_repeated_node_mean_is_exact_and_can_still_be_simulated():
    node = RepeatedNode(W([100, 2]), 3, "series")
    assert node.mean() == pytest.approx(100 * math.gamma(1.5) / 3**0.5)
    simulated = node.mean(method="simulate", mc_samples=20_000, seed=2)
    assert simulated == node.mean(method="simulate", mc_samples=20_000, seed=2)
    assert simulated == pytest.approx(node.mean(), rel=0.02)


def test_the_report_routes_the_mttf_through_the_nodes():
    assert (
        series(E([0.01])).analysis_routes()["mean"].route == routes.NUMERICAL
    )
    simulated = series(
        StandbyModel([W([100, 2])] * 3, k=2, dormancy_factor=0.5)
    )
    report = simulated.analysis_routes()
    assert report["mean"].route == routes.REFUSED
    assert report["mean"].nodes == (0,)
    assert routes.mean_route(PerfectReliability)[0] == routes.EXACT


def test_the_integral_is_accurate_on_its_own():
    # A survival function with a kink and a jump, given as knots.
    def sf(t):
        return np.where(t < 1.0, 1.0 - 0.5 * t, 0.25 * np.exp(-(t - 1.0)))

    expected = 0.75 + 0.25
    assert mean_lifetime(sf, [1.0]) == pytest.approx(expected, rel=1e-12)
    assert model_knots(W([100, 2])).size > 100
