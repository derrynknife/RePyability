"""
Tests the numerical-convolution survival function used for standby nodes
(ConvolvedSurvival), and its use in StandbyModel (k=1) and RepeatedStandbyNode.

The numerical sf is checked against two closed forms:
  - sum of n iid Exponential(rate) == Erlang == Gamma(n, scale=1/rate)
  - Exp(rate 1) + Exp(rate 2) == hypoexponential, sf = 2 e^-t - e^-2t
and is verified to be deterministic (no Monte-Carlo), unlike the previous
Kaplan-Meier fit.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from surpyval import Exponential, Gamma, LogNormal, Weibull

from repyability.rbd.numerical_convolution import ConvolvedSurvival
from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
from repyability.rbd.standby_node import StandbyModel

TOL = dict(rel=1e-2, abs=1e-4)


def test_convolution_matches_erlang():
    exp = Exponential.from_params([1])
    conv = ConvolvedSurvival([exp, exp, exp])
    gamma = Gamma.from_params([3.0, 1.0])
    for t in [0.5, 1, 3, 5, 10, 15]:
        assert conv.sf(t) == pytest.approx(gamma.sf(t), **TOL)
        assert conv.ff(t) == pytest.approx(gamma.ff(t), **TOL)
    assert conv.mean() == pytest.approx(3.0, rel=1e-2)


def test_convolution_matches_hypoexponential():
    conv = ConvolvedSurvival(
        [Exponential.from_params([1]), Exponential.from_params([2])]
    )
    for t in [0.5, 1, 2, 4, 8]:
        expected = 2 * np.exp(-t) - np.exp(-2 * t)
        assert conv.sf(t) == pytest.approx(expected, **TOL)


def test_convolution_array_input():
    exp = Exponential.from_params([1])
    conv = ConvolvedSurvival([exp, exp])
    x = np.array([1.0, 2.0, 3.0])
    out = conv.sf(x)
    assert out.shape == x.shape
    gamma = Gamma.from_params([2.0, 1.0])
    assert out == pytest.approx(gamma.sf(x), **TOL)


def test_convolution_single_model_is_identity():
    w = Weibull.from_params([10, 2])
    conv = ConvolvedSurvival([w])
    for t in [1, 5, 10, 20]:
        assert conv.sf(t) == pytest.approx(w.sf(t), **TOL)


def test_standby_model_k1_sf_is_exact_and_deterministic():
    exp = Exponential.from_params([1])
    standby = StandbyModel([exp, exp, exp], k=1)
    standby_again = StandbyModel([exp, exp, exp], k=1)
    gamma = Gamma.from_params([3.0, 1.0])
    for t in [1, 3, 10]:
        assert standby.sf(t) == pytest.approx(gamma.sf(t), **TOL)
        # Deterministic: a second build gives identical values (no sampling).
        assert standby.sf(t) == standby_again.sf(t)


def test_repeated_standby_sf_matches_erlang():
    exp = Exponential.from_params([1])
    node = RepeatedStandbyNode(exp, 3)
    gamma = Gamma.from_params([3.0, 1.0])
    for t in [1, 3, 10]:
        assert node.sf(t) == pytest.approx(gamma.sf(t), **TOL)
        assert node.ff(t) == pytest.approx(gamma.ff(t), **TOL)


def test_convolution_is_second_order_accurate():
    # Adding cell probabilities against the CDF half a cell back leaves an
    # error of order the grid step squared: about 1e-7 here.
    exp1, exp2 = Exponential.from_params([1]), Exponential.from_params([2])
    t = np.array([0.1, 0.5, 1, 2, 4, 8])
    conv = ConvolvedSurvival([exp1, exp2])
    np.testing.assert_allclose(
        conv.sf(t), 2 * np.exp(-t) - np.exp(-2 * t), atol=1e-7
    )
    erlang = ConvolvedSurvival([exp1, exp1, exp1])
    np.testing.assert_allclose(
        erlang.sf(t), Gamma.from_params([3.0, 1.0]).sf(t), atol=1e-7
    )
    assert erlang.mean() == pytest.approx(3.0, rel=1e-6)


def test_convolution_keeps_its_partial_sums_on_request():
    exp = Exponential.from_params([1])
    conv = ConvolvedSurvival([exp, exp, exp], partials=True)
    t = np.array([0.5, 2.0, 5.0])
    for j in (1, 2, 3):
        np.testing.assert_allclose(
            conv.partial_sf(j, t),
            Gamma.from_params([float(j), 1.0]).sf(t),
            atol=1e-7,
        )
    np.testing.assert_allclose(conv.partial_sf(3, t), conv.sf(t))


def _sum_sf(a, b, t):
    """P(A + B > t), by adaptive quadrature."""
    tail = quad(
        lambda s: float(a.df(s)) * float(b.sf(t - s)),
        0,
        t,
        limit=500,
        epsabs=1e-13,
        epsrel=1e-12,
    )[0]
    return float(a.sf(t)) + tail


@pytest.mark.parametrize(
    "units",
    [
        (Weibull.from_params([100, 0.8]), LogNormal.from_params([4.0, 0.5])),
        (Weibull.from_params([100, 0.8]), Weibull.from_params([100, 0.8])),
        (Weibull.from_params([100, 2]), Weibull.from_params([80, 1.5])),
    ],
    ids=["early-life and lognormal", "two early-life", "smooth"],
)
def test_early_life_units_are_summed_accurately(units):
    # A Weibull with shape below 1 has an infinite density at 0. Added up
    # as cumulative probabilities, the sum is as accurate as for smooth
    # units: its density near 0 was lost when densities were convolved,
    # leaving the reliability 1e-3 off and the MTTF 0.3% (two early-life
    # units: 227.31, not 226.60).
    a, b = units
    conv = ConvolvedSurvival(units)
    for t in (10.0, 50.0, 100.0, 150.0, 250.0, 400.0):
        assert float(conv.sf(t)) == pytest.approx(_sum_sf(a, b, t), abs=2e-6)
    assert conv.mean() == pytest.approx(
        float(a.mean()) + float(b.mean()), rel=1e-6
    )


def test_steep_early_life_units_are_summed_to_their_exact_sum():
    # Gamma lifetimes of one rate add up to a gamma of the summed shapes,
    # so two of shape 0.5 (a density rising as t ** -0.5 at 0) are an
    # Exponential, and three of shape 1/3 too: densities convolved were
    # 1e-2 off, and the MTTF of the three 8.5%.
    rate = 0.01
    t = np.array([1.0, 10.0, 50.0, 100.0, 300.0, 800.0])
    for shape, n in ((0.5, 2), (1 / 3, 3)):
        conv = ConvolvedSurvival([Gamma.from_params([shape, rate])] * n)
        np.testing.assert_allclose(conv.sf(t), np.exp(-rate * t), atol=5e-5)
        assert conv.mean() == pytest.approx(1 / rate, rel=2e-5)
    mixed = ConvolvedSurvival(
        [Gamma.from_params([0.5, rate]), Gamma.from_params([1.5, rate])]
    )
    np.testing.assert_allclose(
        mixed.sf(t), Gamma.from_params([2.0, rate]).sf(t), atol=5e-6
    )


def test_a_unit_with_only_a_survival_function_is_summed_the_same():
    class SurvivalOnly:
        def __init__(self, model):
            self.model = model

        def sf(self, x):
            return self.model.sf(x)

        def mean(self):
            return self.model.mean()

    w = Weibull.from_params([100, 0.8])
    plain = ConvolvedSurvival([w, w])
    bare = ConvolvedSurvival([SurvivalOnly(w), SurvivalOnly(w)])
    t = np.array([10.0, 100.0, 400.0])
    np.testing.assert_allclose(bare.sf(t), plain.sf(t), atol=1e-12)
