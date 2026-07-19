"""
Tests NonRepairableRBD.parameter_sensitivity().

The sensitivity of system reliability to a node parameter is the Birnbaum
importance of the node times the derivative of the node's sf with respect to
the parameter. The parameter derivative is checked against closed-form
expressions for Weibull, Exponential and FixedEventProbability.
"""

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.standby_node import StandbyModel


def weibull_dsf_dalpha(t, alpha, beta):
    sf = np.exp(-((t / alpha) ** beta))
    return sf * (beta / alpha) * (t / alpha) ** beta


def weibull_dsf_dbeta(t, alpha, beta):
    sf = np.exp(-((t / alpha) ** beta))
    return sf * (-((t / alpha) ** beta) * np.log(t / alpha))


def exponential_dsf_dfailure_rate(t, failure_rate):
    return -t * np.exp(-failure_rate * t)


def test_weibull_parameter_sensitivity(rbd_series: NonRepairableRBD):
    """Sensitivity == Birnbaum importance * analytic dsf/dparam for a
    Weibull series system."""
    t = 10.0
    sens = rbd_series.parameter_sensitivity(t)
    birnbaum = rbd_series.birnbaum_importance(t)
    params = {2: (20.0, 2.0), 3: (100.0, 3.0), 4: (50.0, 20.0)}
    for node, (alpha, beta) in params.items():
        assert set(sens[node]) == {"alpha", "beta"}
        assert sens[node]["alpha"] == pytest.approx(
            birnbaum[node] * weibull_dsf_dalpha(t, alpha, beta), rel=1e-4
        )
        assert sens[node]["beta"] == pytest.approx(
            birnbaum[node] * weibull_dsf_dbeta(t, alpha, beta), rel=1e-4
        )


def test_exponential_parameter_sensitivity():
    """Sensitivity of a single Exponential node matches the closed form."""
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": surv.Exponential.from_params([0.1]),
            "b": surv.Exponential.from_params([0.2]),
        },
    )
    t = 5.0
    sens = rbd.parameter_sensitivity(t)
    birnbaum = rbd.birnbaum_importance(t)
    for node, rate in {"a": 0.1, "b": 0.2}.items():
        assert set(sens[node]) == {"failure_rate"}
        assert sens[node]["failure_rate"] == pytest.approx(
            birnbaum[node] * exponential_dsf_dfailure_rate(t, rate),
            rel=1e-4,
        )


def test_fixed_probability_parameter_sensitivity(
    rbd_parallel: NonRepairableRBD,
):
    """For a fixed event probability, sf = 1 - p so dsf/dp = -1 and the
    sensitivity is exactly minus the Birnbaum importance."""
    sens = rbd_parallel.parameter_sensitivity(1.0)
    birnbaum = rbd_parallel.birnbaum_importance(1.0)
    for node in (2, 3, 4):
        assert set(sens[node]) == {"p"}
        assert sens[node]["p"] == pytest.approx(-birnbaum[node], rel=1e-4)


def test_scalar_input_returns_floats(rbd_series: NonRepairableRBD):
    sens = rbd_series.parameter_sensitivity(10.0)
    for node_params in sens.values():
        for value in node_params.values():
            assert isinstance(value, float)


def test_array_input_returns_arrays(rbd_series: NonRepairableRBD):
    x = np.array([5.0, 10.0, 15.0])
    sens = rbd_series.parameter_sensitivity(x)
    for node_params in sens.values():
        for value in node_params.values():
            assert isinstance(value, np.ndarray)
            assert value.shape == x.shape


def test_composite_node_omitted():
    """A standby (composite) node has no reconstructable parameters and is
    omitted from the result."""
    rbd = NonRepairableRBD(
        [(1, 2), (2, 3), (3, 4), (4, 5)],
        {
            2: surv.Weibull.from_params([20, 2]),
            3: StandbyModel(
                [
                    surv.Weibull.from_params([5, 1.1]),
                    surv.Weibull.from_params([5, 1.1]),
                ]
            ),
            4: surv.Weibull.from_params([50, 20]),
        },
    )
    sens = rbd.parameter_sensitivity(5.0)
    assert set(sens) == {2, 4}


def test_repeated_node_omitted(
    rbd_repeated_component_parallel: NonRepairableRBD,
):
    """A repeated node references another node's model rather than owning its
    own parameters, so it is omitted."""
    sens = rbd_repeated_component_parallel.parameter_sensitivity(1.0)
    # Node 5 repeats node 2; only the genuine parametric nodes appear.
    assert 5 not in sens
    assert {2, 3, 4}.issubset(set(sens))


def test_forced_node_has_zero_sensitivity(rbd_series: NonRepairableRBD):
    """A node pinned via working_nodes/broken_nodes is independent of its
    parameters, so its sensitivities are zero."""
    sens = rbd_series.parameter_sensitivity(10.0, working_nodes=[2])
    assert sens[2] == {"alpha": 0.0, "beta": 0.0}
    broken = rbd_series.parameter_sensitivity(10.0, broken_nodes=[3])
    assert broken[3] == {"alpha": 0.0, "beta": 0.0}


def test_boundary_parameter_uses_one_sided_difference():
    """A fixed probability pinned near its upper bound cannot be perturbed
    upward (p + h would exceed 1); the finite difference falls back to a
    one-sided estimate instead of raising, and still recovers dsf/dp = -1."""
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": FixedEventProbability.from_params(0.999999)},
    )
    sens = rbd.parameter_sensitivity(1.0)
    # Single series node: Birnbaum importance is 1, dsf/dp = -1.
    assert sens["a"]["p"] == pytest.approx(-1.0, rel=1e-3)


def test_x_required_for_time_varying(rbd_series: NonRepairableRBD):
    with pytest.raises(ValueError):
        rbd_series.parameter_sensitivity()


def test_x_optional_for_fixed(rbd_parallel: NonRepairableRBD):
    """A fixed-probability RBD does not require x."""
    sens = rbd_parallel.parameter_sensitivity()
    assert set(sens) == {2, 3, 4}


def test_unknown_node_override_raises(rbd_series: NonRepairableRBD):
    with pytest.raises(ValueError):
        rbd_series.parameter_sensitivity(10.0, broken_nodes=["nope"])
