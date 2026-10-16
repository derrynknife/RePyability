"""Uncertainty importance (#196): each uncertain input's share of a system
quantity's variance, by the delta method and by Sobol indices.

Checked against closed forms (exponential units in series, whose rates
enter every quantity through their sum), against a fit's covariance and
its model's derivatives worked out by hand, between the two methods, and
against the draws of ``sf_uncertainty``."""

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv

from repyability import BetaFactor, CCFGroup, NonRepairableRBD

E, W = surv.Exponential.from_params, surv.Weibull.from_params

SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
RATES = {
    "a": {"failure_rate": st.uniform(0.005, 0.01)},
    "b": {"failure_rate": st.uniform(0.008, 0.004)},
}
# The rates' variances.
VA, VB = 0.01**2 / 12, 0.004**2 / 12


def series():
    return NonRepairableRBD(SERIES, {"a": E([0.01]), "b": E([0.01])})


def fitted(scale, n, shape=2.0):
    u = (np.arange(n) + 0.5) / n
    return surv.Weibull.fit(scale * (-np.log(1 - u)) ** (1 / shape))


@pytest.mark.parametrize(
    "of, x",
    [
        ("sf", 10.0),
        ("mean", None),
        ("bx_life", 10.0),
        ("time_to_reliability", 0.8),
    ],
)
def test_rates_in_series_share_by_their_variances(of, x):
    # Every quantity depends on the rates through their sum alone, so the
    # delta method's parts are in the ratio of the rates' variances.
    parts = series().uncertainty_importance(x, RATES, of=of)
    assert parts.method == "delta"
    assert parts.first_order == pytest.approx(
        {"a": VA / (VA + VB), "b": VB / (VA + VB)}, rel=1e-6
    )
    assert parts.total == parts.first_order


def test_the_variance_in_closed_form():
    # R(10) = exp(-10 (la + lb)): dR/dl = -10 R for each.
    reliability = np.exp(-10 * 0.02)
    parts = series().uncertainty_importance(10.0, RATES)
    assert parts.variance == pytest.approx(
        (10 * reliability) ** 2 * (VA + VB), rel=1e-6
    )
    # The MTTF, 1 / (la + lb): its derivative is -1 / (la + lb)^2.
    mttf = series().uncertainty_importance(None, RATES, of="mean")
    assert mttf.variance == pytest.approx((VA + VB) / 0.02**4, rel=1e-6)


def test_a_fit_s_covariance_and_derivatives():
    pump = fitted(100, 10)
    rbd = NonRepairableRBD([("s", "pump"), ("pump", "t")], {"pump": pump})
    alpha, beta = pump.params
    x = 50.0
    reliability = np.exp(-((x / alpha) ** beta))
    gradient = np.array(
        [
            reliability * beta / alpha * (x / alpha) ** beta,
            -reliability * (x / alpha) ** beta * np.log(x / alpha),
        ]
    )
    parts = rbd.uncertainty_importance(x)
    assert parts.first_order == {"pump": 1.0}
    assert parts.variance == pytest.approx(
        gradient @ np.asarray(pump.covariance()) @ gradient, rel=1e-6
    )


def test_a_population_is_one_input():
    pump = fitted(100, 10)
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")],
        {"p1": pump, "p2": pump, "v": fitted(120, 40)},
    )
    parts = rbd.uncertainty_importance(50.0)
    # The pumps share their fit, and so their uncertainty (by default).
    assert set(parts.first_order) == {("p1", "p2"), "v"}
    assert sum(parts.first_order.values()) == pytest.approx(1.0)
    given = rbd.uncertainty_importance(50.0, {("p1", "p2"): "fit", "v": "fit"})
    assert given.first_order == pytest.approx(parts.first_order)


def test_the_delta_method_and_sobol_agree_for_a_small_uncertainty():
    rbd = series()
    delta = rbd.uncertainty_importance(10.0, RATES)
    sobol = rbd.uncertainty_importance(
        10.0, RATES, method="sobol", n_draws=20_000, seed=3
    )
    assert sobol.method == "sobol"
    for key in "ab":
        assert sobol.first_order[key] == pytest.approx(
            delta.first_order[key], abs=0.02
        )
        assert sobol.total[key] == pytest.approx(
            delta.first_order[key], abs=0.02
        )
    assert sobol.variance == pytest.approx(delta.variance, rel=0.05)


def test_sobol_s_variance_is_the_draws():
    # Its two sets are sf_uncertainty's first 2n draws.
    rbd = NonRepairableRBD(
        [("s", "pump"), ("pump", "valve"), ("valve", "t")],
        {"pump": fitted(100, 10), "valve": fitted(120, 40)},
    )
    sobol = rbd.uncertainty_importance(
        50.0, method="sobol", n_draws=500, seed=7
    )
    draws = rbd.sf_uncertainty(50.0, n_draws=1000, seed=7)
    assert sobol.variance == pytest.approx(np.var(draws.samples, ddof=1))


def test_times_and_lists_of_models():
    rbd = series()
    x = np.array([5.0, 10.0, 40.0])
    parts = rbd.uncertainty_importance(x, RATES)
    assert parts.first_order["a"].shape == (3,)
    for k, t in enumerate(x):
        assert rbd.uncertainty_importance(t, RATES).variance == pytest.approx(
            parts.variance[k]
        )
    models = [E([rate]) for rate in (0.008, 0.01, 0.012)]
    with pytest.raises(ValueError, match="method='sobol'"):
        rbd.uncertainty_importance(10.0, {"a": models})
    sobol = rbd.uncertainty_importance(
        10.0, {"a": models, "b": RATES["b"]}, method="sobol", n_draws=2000
    )
    assert set(sobol.first_order) == {"a", "b"}


def test_a_common_cause_model_s_uncertainty():
    group = CCFGroup(["p1", "p2"], BetaFactor(0.1, basis="rate"))
    rbd = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        {"p1": E([0.002]), "p2": E([0.002])},
        ccf_groups=[group],
    )
    uncertainty = {
        ("p1", "p2"): {"failure_rate": st.uniform(0.001, 0.002)},
        group: {"beta": st.beta(2, 18)},
    }
    parts = rbd.uncertainty_importance(500.0, uncertainty)
    assert set(parts.first_order) == {("p1", "p2"), group}
    assert sum(parts.first_order.values()) == pytest.approx(1.0)
    assert 0 < parts.first_order[group] < 1
    sobol = rbd.uncertainty_importance(
        500.0, uncertainty, method="sobol", n_draws=4000, seed=1
    )
    assert sobol.total[group] == pytest.approx(
        parts.first_order[group], abs=0.05
    )


def test_bad_arguments_are_refused():
    rbd = series()
    with pytest.raises(ValueError, match="of must be"):
        rbd.uncertainty_importance(10.0, RATES, of="median")
    with pytest.raises(ValueError, match="method must be"):
        rbd.uncertainty_importance(10.0, RATES, method="morris")
    with pytest.raises(ValueError, match="not taken"):
        rbd.uncertainty_importance(10.0, RATES, of="mean")
    with pytest.raises(ValueError, match="percentage"):
        rbd.uncertainty_importance(150.0, RATES, of="bx_life")
    with pytest.raises(ValueError, match="x is required"):
        rbd.uncertainty_importance(None, RATES)
    with pytest.raises(ValueError, match="surpyval fit"):
        rbd.uncertainty_importance(10.0)
