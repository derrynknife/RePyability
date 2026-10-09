"""The uncertainty methods take what the point methods take (#226).

An array of targets (B-life percentages, reliabilities, times or mission
lengths) gives one row of draws per target, from the same parameter
draws: each element is what that target alone gives with the same seed. A
target out of range is refused naming the argument, and a life some of
whose units never fail has an infinite mean in every draw, with no
warning.
"""

import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, RepairableRBD

DRAWS = dict(n_draws=200, seed=1)


@pytest.fixture(scope="module")
def fitted():
    life = surv.Weibull.fit(np.random.default_rng(0).weibull(2, 30) * 100)
    return NonRepairableRBD([("s", "a"), ("a", "t")], {"a": life})


def each_alone(method, targets):
    return [method(target, **DRAWS) for target in targets]


def agrees(together, alone):
    lower, upper = together.interval(0.9)
    for i, one in enumerate(alone):
        assert together.nominal[i] == pytest.approx(one.nominal, rel=1e-12)
        assert together.median[i] == pytest.approx(one.median, rel=1e-12)
        assert (lower[i], upper[i]) == pytest.approx(
            one.interval(0.9), rel=1e-12
        )


@pytest.mark.parametrize("given", [[1, 10], np.array([1, 10]), (1, 10)])
def test_b_lives(fitted, given):
    agrees(
        fitted.bx_life_uncertainty(given, **DRAWS),
        each_alone(fitted.bx_life_uncertainty, [1, 10]),
    )


def test_times_to_reliabilities(fitted):
    agrees(
        fitted.time_to_reliability_uncertainty([0.99, 0.9], **DRAWS),
        each_alone(fitted.time_to_reliability_uncertainty, [0.99, 0.9]),
    )


def test_survival_keeps_the_times_shape(fitted):
    times = [[10, 50], [60, 70]]
    result = fitted.sf_uncertainty(times, **DRAWS)
    assert np.shape(result.nominal) == (2, 2)
    assert np.shape(result.samples) == (DRAWS["n_draws"], 2, 2)
    np.testing.assert_allclose(result.nominal, fitted.sf(times), rtol=1e-12)
    alone = fitted.sf_uncertainty(60, **DRAWS)
    assert result.median[1, 0] == pytest.approx(alone.median, rel=1e-12)


@pytest.mark.parametrize("given", [[0, 10], [1, 100], [-1]])
def test_percentages_out_of_range_are_named(fitted, given):
    with pytest.raises(
        ValueError, match=r"x must be a percentage in \(0, 100\)"
    ):
        fitted.bx_life_uncertainty(given)


def test_reliabilities_out_of_range_are_named(fitted):
    with pytest.raises(ValueError, match=r"target reliability must be in"):
        fitted.time_to_reliability_uncertainty([0.5, 1.2])


def test_mission_lengths_in_uncertainty_importance():
    repair = surv.Exponential.fit(np.random.default_rng(1).exponential(5, 12))
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": surv.Exponential.from_params([0.01]),
                "repairability": repair,
            }
        },
    )
    together = rbd.uncertainty_importance(
        [10.0, 50.0], of="mission_availability"
    )
    for i, length in enumerate([10.0, 50.0]):
        alone = rbd.uncertainty_importance(length, of="mission_availability")
        assert together.variance[i] == pytest.approx(alone.variance, rel=1e-9)
        for node, value in alone.first_order.items():
            assert together.first_order[node][i] == pytest.approx(value)


def test_a_life_that_may_never_end_has_an_infinite_mean():
    g = np.random.default_rng(3)
    life = surv.Weibull.fit(
        np.r_[g.weibull(2, 40) * 100, np.full(20, 400.0)],
        np.r_[np.zeros(40), np.ones(20)],
        lfp=True,
    )
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": life})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = rbd.mean_uncertainty(**DRAWS)
    assert result.nominal == np.inf
    assert result.interval(0.9) == (np.inf, np.inf)
