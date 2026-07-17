"""Tests for inverse-reliability queries: time_to_reliability and bx_life.

Validated against the closed-form Weibull B_X life,
``bx = alpha * (-ln(1 - x/100)) ** (1/beta)``.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD

FEP = surv.FixedEventProbability


def _weibull_bx(alpha, beta, x):
    return alpha * (-np.log(1.0 - x / 100.0)) ** (1.0 / beta)


def _single_weibull(alpha=100.0, beta=2.0):
    return NonRepairableRBD(
        [("s", 1), (1, "t")], {1: surv.Weibull.from_params([alpha, beta])}
    )


def test_bx_life_matches_weibull_closed_form():
    rbd = _single_weibull(100.0, 2.0)
    for x in (1.0, 10.0, 50.0, 90.0):
        assert rbd.bx_life(x) == pytest.approx(_weibull_bx(100.0, 2.0, x))


def test_time_to_reliability_is_inverse_of_sf():
    rbd = _single_weibull(80.0, 1.5)
    for target in (0.99, 0.9, 0.5, 0.2):
        t = rbd.time_to_reliability(target)
        assert float(rbd.sf(t)) == pytest.approx(target)


def test_bx_life_equals_time_to_reliability():
    rbd = _single_weibull()
    assert rbd.bx_life(10) == pytest.approx(rbd.time_to_reliability(0.9))


def test_series_solution_satisfies_sf():
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (2, "t")],
        {
            1: surv.Weibull.from_params([100, 2]),
            2: surv.Weibull.from_params([50, 1.5]),
        },
    )
    t = rbd.time_to_reliability(0.9)
    assert float(rbd.sf(t)) == pytest.approx(0.9)


def test_conditioning_flows_through():
    # Forcing node 2 perfectly reliable reduces the series system to node 1,
    # so the B10 life equals node 1's own B10 life.
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (2, "t")],
        {
            1: surv.Weibull.from_params([100, 2]),
            2: surv.Weibull.from_params([50, 1.5]),
        },
    )
    assert rbd.bx_life(10, working_nodes=[2]) == pytest.approx(
        _weibull_bx(100, 2, 10)
    )


def test_fixed_rbd_raises():
    rbd = NonRepairableRBD(
        [("s", "A"), ("A", "t")], {"A": FEP.from_params(0.2)}
    )
    with pytest.raises(ValueError, match="does not vary with time"):
        rbd.time_to_reliability(0.9)


def test_invalid_target_and_percentage_raise():
    rbd = _single_weibull()
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="target reliability"):
            rbd.time_to_reliability(bad)
    for bad in (0.0, 100.0, 150.0):
        with pytest.raises(ValueError, match="percentage"):
            rbd.bx_life(bad)


def test_unreachable_target_raises():
    # A fixed node in series caps system reliability at 0.8, so R = 0.9 is
    # never reached.
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (2, "t")],
        {1: FEP.from_params(0.2), 2: surv.Weibull.from_params([100, 2])},
    )
    with pytest.raises(ValueError, match="never reached"):
        rbd.time_to_reliability(0.9)
