"""Tests for the system-level failure density (df), hazard rate (hf) and
cumulative hazard (Hf), validated against closed-form relationships and
surpyval's own distribution functions.
"""

import numpy as np
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD

X = np.array([10.0, 50.0, 100.0, 200.0])


def _single_weibull_rbd():
    # s -> 1 -> t; with a single intermediate node the system reliability
    # equals that node's reliability, giving an exact reference.
    return NonRepairableRBD(
        [("s", 1), (1, "t")], {1: surv.Weibull.from_params([100, 2])}
    )


def test_Hf_matches_negative_log_sf():
    rbd = _single_weibull_rbd()
    np.testing.assert_allclose(rbd.Hf(X), -np.log(rbd.sf(X)))


def test_hf_equals_df_over_sf():
    rbd = _single_weibull_rbd()
    np.testing.assert_allclose(rbd.hf(X), rbd.df(X) / rbd.sf(X), rtol=1e-9)


def test_df_matches_node_density_for_single_node():
    rbd = _single_weibull_rbd()
    w = surv.Weibull.from_params([100, 2])
    np.testing.assert_allclose(rbd.df(X), w.df(X), rtol=1e-3)


def test_hf_matches_node_hazard_for_single_node():
    rbd = _single_weibull_rbd()
    w = surv.Weibull.from_params([100, 2])
    np.testing.assert_allclose(rbd.hf(X), w.hf(X), rtol=1e-3)


def test_series_hazard_is_sum_of_node_hazards():
    # For a pure series system the system hazard rate is the sum of the node
    # hazard rates.
    w1 = surv.Weibull.from_params([100, 2])
    w2 = surv.Weibull.from_params([50, 1.5])
    rbd = NonRepairableRBD([("s", 1), (1, 2), (2, "t")], {1: w1, 2: w2})
    np.testing.assert_allclose(rbd.hf(X), w1.hf(X) + w2.hf(X), rtol=1e-3)


def test_density_is_nonnegative():
    rbd = _single_weibull_rbd()
    assert np.all(rbd.df(X) >= 0.0)
