"""Conditional survival at old ages (#268): the chance of surviving a
further ``x`` given survival to ``X`` is worked out from the cumulative
hazard, ``exp(-(H(X + x) - H(X)))``, where the model has one, so it keeps
its precision where ``R(X + x)`` and ``R(X)`` are both too small for a
float, rather than coming out 0 as their ratio did."""

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, NonRepairableRBD
from repyability.utils.wrappers import conditional_survival

W = surv.Weibull.from_params


def weibull_cs(x, X, alpha, beta):
    return np.exp(-(((X + x) / alpha) ** beta - (X / alpha) ** beta))


@pytest.mark.parametrize("X", [50.0, 300.0, 600.0, 1000.0, 2000.0])
def test_a_weibull_far_past_its_life(X):
    # At X = 1000 the Weibull's R(X) = exp(-1000) is below the smallest
    # float; the further 10 hours' survival is exp(-30.3) = 6.9e-14.
    model = W([100, 3])
    expected = weibull_cs(10.0, X, 100, 3)
    assert conditional_survival(model, 10.0, X) == pytest.approx(
        expected, rel=1e-9
    )


def test_a_limited_failure_population_far_past_its_life():
    # Long after the units that fail have failed, the survivors never
    # fail: the conditional survival is 1.
    model = W([100, 3], lfp_p=0.4)
    assert conditional_survival(model, 50.0, 1e4) == pytest.approx(1.0)
    at = 100.0
    f = lambda t: 1 - np.exp(-((t / 100) ** 3))  # noqa: E731
    expected = (1 - 0.4 * f(at + 20)) / (1 - 0.4 * f(at))
    assert conditional_survival(model, 20.0, at) == pytest.approx(
        expected, rel=1e-12
    )


def test_an_age_that_cannot_be_reached_survives_nothing():
    # A model whose life ends: past the end, nothing has survived to X.
    model = surv.Uniform.from_params([0.0, 100.0])
    np.testing.assert_array_equal(
        conditional_survival(model, [10.0, 50.0], 150.0), [0.0, 0.0]
    )


def test_it_agrees_with_the_ratio_where_the_ratio_is_precise():
    model = W([100, 2])
    x = np.linspace(0.0, 200.0, 21)
    for X in (0.0, 10.0, 80.0, 150.0):
        np.testing.assert_allclose(
            conditional_survival(model, x, X),
            model.sf(x + X) / model.sf(X),
            rtol=1e-12,
        )


def test_a_node_state_far_past_its_life():
    # sf_given_state conditions each node on its age through it.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": W([100, 3]), "b": surv.Exponential.from_params([0.001])},
    )
    state = {"a": NodeState(age=1000.0)}
    expected = weibull_cs(10.0, 1000.0, 100, 3) * np.exp(-0.01)
    assert rbd.sf_given_state(10.0, state) == pytest.approx(expected, rel=1e-9)


def test_a_diagrams_own_conditional_survival():
    # The diagram's cs is its own Hf's: two units in series, aged 600.
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": W([100, 3]), "b": W([100, 3])},
    )
    expected = weibull_cs(10.0, 600.0, 100, 3) ** 2
    assert rbd.cs(10.0, 600.0) == pytest.approx(expected, rel=1e-9)
