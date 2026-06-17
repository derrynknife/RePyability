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
from surpyval import Exponential, Gamma, Weibull

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
