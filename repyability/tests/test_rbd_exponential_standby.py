"""
Tests the exact closed form used when standby components are identical
Exponentials.

For N identical exponential units of rate lambda with k operating, the cold
standby lifetime is exactly Erlang(N-k+1, k*lambda): while k units run the
failure rate is k*lambda, and by memorylessness the inter-failure times are
i.i.d. Exponential(k*lambda). This is checked against scipy's gamma, against
the surpyval Gamma, and (in distribution) against the Monte-Carlo simulation.
"""

import numpy as np
import pytest
from scipy.stats import gamma
from surpyval import Exponential, Gamma

from repyability.rbd.standby_node import StandbyModel

TOL = dict(rel=1e-2, abs=1e-4)
EXP1 = Exponential.from_params([1])


def test_k1_exponential_is_erlang():
    # N=3, k=1 -> Erlang(3, 1) == Gamma(3, 1)
    standby = StandbyModel([EXP1, EXP1, EXP1], k=1)
    gam = Gamma.from_params([3.0, 1.0])
    for t in [0.5, 1, 3, 10]:
        assert standby.sf(t) == pytest.approx(gam.sf(t), **TOL)
        assert standby.ff(t) == pytest.approx(gam.ff(t), **TOL)
    assert standby.mean() == pytest.approx(3.0, rel=1e-9)


def test_k2_exponential_is_erlang():
    # N=4, k=2 -> fails at the 3rd failure, rate 2 while operating
    # -> Erlang(3, 2) == Gamma(a=3, scale=1/2), mean 3/2
    standby = StandbyModel([EXP1] * 4, k=2)
    for t in [0.5, 1, 2, 5]:
        assert standby.sf(t) == pytest.approx(
            gamma.sf(t, a=3, scale=0.5), **TOL
        )
    assert standby.mean() == pytest.approx(1.5, rel=1e-9)


def test_k_equals_n_is_min_exponential():
    # k=N: all units operate, system fails at the first failure
    # -> Exp(N*rate) == Erlang(1, N*rate)
    standby = StandbyModel([EXP1] * 4, k=4)
    for t in [0.25, 1, 2]:
        assert standby.sf(t) == pytest.approx(np.exp(-4 * t), **TOL)


def test_exponential_closed_form_is_deterministic():
    a = StandbyModel([EXP1] * 5, k=2)
    b = StandbyModel([EXP1] * 5, k=2)
    for t in [1, 2, 5]:
        assert a.sf(t) == b.sf(t)  # exact, no sampling


def test_k2_exponential_matches_simulation():
    # The closed form should agree (in mean) with the priority-queue sim.
    exp = Exponential.from_params([2.0])  # rate 2
    standby = StandbyModel([exp] * 5, k=2)  # Erlang(4, 4), mean 1.0
    assert standby.mean() == pytest.approx(1.0, rel=1e-9)
    sim_mean = standby.random(50_000).mean()
    assert sim_mean == pytest.approx(1.0, rel=3e-2)


def test_non_identical_rates_use_convolution_not_erlang():
    # Different rates -> not the simple Erlang; k=1 still exact via the
    # convolution path: Exp(1) + Exp(2) is hypoexponential, 2 e^-t - e^-2t.
    standby = StandbyModel(
        [Exponential.from_params([1]), Exponential.from_params([2])], k=1
    )
    for t in [0.5, 1, 2]:
        expected = 2 * np.exp(-t) - np.exp(-2 * t)
        assert standby.sf(t) == pytest.approx(expected, **TOL)
