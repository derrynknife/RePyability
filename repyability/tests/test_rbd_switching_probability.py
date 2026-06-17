"""
Tests imperfect switching for cold-standby nodes.

Under imperfect switching the lifetime is a mixture of partial sums, weighted
by how many switches succeed. The key checks use the closed form for a single
switch between two Exp(1) components with switch probability p:

    S(t) = (1 - p) e^-t + p * Erlang(2,1).sf(t) = e^-t (1 + p t)

and the corresponding mean for three Exp(1) components, E[T] = 1 + p + p^2.
"""

import numpy as np
import pytest
from surpyval import Exponential

from repyability.rbd.numerical_convolution import ConvolvedSurvival
from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
from repyability.rbd.standby_node import StandbyModel

TOL = dict(rel=1e-2, abs=1e-4)
EXP = Exponential.from_params([1])


def test_single_switch_closed_form():
    for p in [0.0, 0.3, 0.7, 1.0]:
        conv = ConvolvedSurvival([EXP, EXP], switching_probability=p)
        for t in [0.5, 1, 3, 6]:
            expected = np.exp(-t) * (1 + p * t)
            assert conv.sf(t) == pytest.approx(expected, **TOL)


def test_perfect_switching_matches_default():
    a = ConvolvedSurvival([EXP, EXP, EXP], switching_probability=1.0)
    b = ConvolvedSurvival([EXP, EXP, EXP])
    for t in [1, 3, 7]:
        assert a.sf(t) == pytest.approx(b.sf(t), **TOL)


def test_zero_switching_is_primary_only():
    conv = ConvolvedSurvival([EXP, EXP, EXP], switching_probability=0.0)
    for t in [0.5, 1, 3]:
        assert conv.sf(t) == pytest.approx(np.exp(-t), **TOL)


def test_mean_three_components_closed_form():
    # E[T] = (1-p)*1 + p(1-p)*2 + p^2*3 = 1 + p + p^2
    for p in [0.0, 0.4, 0.9, 1.0]:
        conv = ConvolvedSurvival([EXP, EXP, EXP], switching_probability=p)
        assert conv.mean() == pytest.approx(1 + p + p**2, rel=1e-2)


def test_per_switch_sequence():
    # switches [1.0, 0.0]: first always works, second never -> exactly the
    # first two components run -> Erlang(2, 1).
    conv = ConvolvedSurvival([EXP, EXP, EXP], switching_probability=[1.0, 0.0])
    erlang2 = ConvolvedSurvival([EXP, EXP])
    for t in [0.5, 1, 3]:
        assert conv.sf(t) == pytest.approx(erlang2.sf(t), **TOL)


def test_switching_weights_sum_to_one():
    for p in [0.0, 0.5, 1.0]:
        conv = ConvolvedSurvival([EXP, EXP, EXP], switching_probability=p)
        assert sum(conv.switching_weights) == pytest.approx(1.0)


def test_standby_model_switching_sf():
    standby = StandbyModel([EXP, EXP], k=1, switching_probability=0.5)
    for t in [0.5, 1, 3]:
        expected = np.exp(-t) * (1 + 0.5 * t)
        assert standby.sf(t) == pytest.approx(expected, **TOL)


def test_standby_model_switching_random_mean():
    # random() should also reflect imperfect switching: mean ~ 1 + p (n=2).
    standby = StandbyModel([EXP, EXP], k=1, switching_probability=0.5)
    assert standby.random(200_000).mean() == pytest.approx(1.5, abs=5e-2)


def test_repeated_standby_switching_sf():
    node = RepeatedStandbyNode(EXP, 2, switching_probability=0.5)
    for t in [0.5, 1, 3]:
        assert node.sf(t) == pytest.approx(np.exp(-t) * (1 + 0.5 * t), **TOL)


def test_standby_k2_switching_not_implemented():
    with pytest.raises(NotImplementedError):
        StandbyModel([EXP, EXP, EXP], k=2, switching_probability=0.5)


def test_invalid_switching_probability():
    with pytest.raises(ValueError):
        ConvolvedSurvival([EXP, EXP], switching_probability=1.5)
    with pytest.raises(ValueError):
        ConvolvedSurvival([EXP, EXP, EXP], switching_probability=[0.5])
