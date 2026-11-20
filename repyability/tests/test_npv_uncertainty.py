"""How sure an NPV is: its derivative in the discount rate
(``expected_npv(rate_sensitivity=True)``), its spread over plausible
models of the components (``expected_npv_uncertainty``), and how robust a
choice of replacement intervals is to them
(``optimal_replacement_intervals(robust=...)``)."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import NPVUncertainty, PlanRobustness, RepairableRBD
from repyability.rbd import _repairable_uncertainty

E, W = surv.Exponential.from_params, surv.Weibull.from_params

LAMBDA, MU, PRICE, VALUE = 1e-3, 0.1, 500.0, 10.0
HORIZON = 5 * 8760.0
RATE = math.log(1.08) / 8760


def pump(life=None):
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": E([LAMBDA]) if life is None else life,
                "repairability": E([MU]),
                "repair_cost": PRICE,
                "acquisition_cost": 20000.0,
            }
        },
    )


def availability(t, lam=LAMBDA):
    """An exponential unit's point availability, from new."""
    return MU / (lam + MU) + lam / (lam + MU) * math.exp(-(lam + MU) * t)


def closed_npv(rate, lam=LAMBDA, horizon=HORIZON):
    """The pump's NPV: value while up, less a repair at each failure (at
    the rate ``lam A(t)``), less the price paid at 0."""
    net = VALUE - PRICE * lam
    return (
        integrate.quad(
            lambda t: math.exp(-rate * t) * availability(t, lam) * net,
            0.0,
            horizon,
            limit=200,
            epsabs=1e-9,
            epsrel=1e-12,
        )[0]
        - 20000.0
    )


def closed_slope(rate):
    """``d npv / d r`` of ``closed_npv``: each amount at ``t`` weighed
    by ``-t``."""
    net = VALUE - PRICE * LAMBDA
    return integrate.quad(
        lambda t: -t * math.exp(-rate * t) * availability(t) * net,
        0.0,
        HORIZON,
        limit=200,
        epsabs=1e-6,
        epsrel=1e-12,
    )[0]


@pytest.mark.parametrize("rate", [RATE, 0.0, 1e-9])
def test_the_rate_sensitivity_is_the_closed_form_s(rate):
    # At 0 and just above it the differences are one-sided.
    result = pump().expected_npv(
        HORIZON, discount_rate=rate, value_rate=VALUE, rate_sensitivity=True
    )
    exact = closed_slope(rate)
    assert result.rate_sensitivity == pytest.approx(exact, rel=1e-6)
    # The error it states bounds the one it makes.
    assert (
        abs(result.rate_sensitivity - exact) <= result.rate_sensitivity_error
    )
    assert result.rate_sensitivity_error < 1e-4 * abs(exact)


def test_the_rate_sensitivity_predicts_a_change_of_rate():
    rbd = pump()
    base = rbd.expected_npv(
        HORIZON, discount_rate=RATE, value_rate=VALUE, rate_sensitivity=True
    )
    step = 1e-4 / HORIZON
    moved = rbd.expected_npv(
        HORIZON, discount_rate=RATE + step, value_rate=VALUE
    )
    assert moved.npv - base.npv == pytest.approx(
        base.rate_sensitivity * step, rel=1e-3
    )


def test_the_rate_sensitivity_covers_a_retirement_and_a_salvage():
    # The salvage is discounted from the retirement: its part of the
    # derivative is -stop * exp(-r stop) * salvage.
    rbd = pump()
    kwargs = dict(discount_rate=RATE, value_rate=VALUE, retire_at=20000.0)
    plain = rbd.expected_npv(HORIZON, rate_sensitivity=True, **kwargs)
    sold = rbd.expected_npv(
        HORIZON, salvage=8000.0, rate_sensitivity=True, **kwargs
    )
    assert sold.rate_sensitivity - plain.rate_sensitivity == pytest.approx(
        -20000.0 * math.exp(-RATE * 20000.0) * 8000.0, rel=1e-5
    )


def test_without_asking_there_is_no_rate_sensitivity():
    result = pump().expected_npv(HORIZON, discount_rate=RATE, value_rate=1.0)
    assert result.rate_sensitivity is None
    assert result.rate_sensitivity_error is None
    with pytest.raises(TypeError, match="rate_sensitivity"):
        pump().expected_npv(HORIZON, value_rate=1.0, rate_sensitivity="yes")


# An exponential life fitted to 40 failures: its rate's variance is
# lambda^2 / 40.
FAILURES = np.random.default_rng(0).exponential(1.0 / LAMBDA, 40)


@pytest.fixture(scope="module")
def fitted():
    return pump(surv.Exponential.fit(FAILURES))


@pytest.fixture(scope="module")
def spread(fitted):
    return fitted.expected_npv_uncertainty(
        HORIZON,
        discount_rate=RATE,
        value_rate=VALUE,
        n_draws=400,
        seed=3,
    )


def test_the_delta_method_is_the_closed_form_s(fitted, spread):
    lam = float(fitted.components["p"].reliability.params[0])
    variance = lam**2 / len(FAILURES)
    h = 1e-3 * lam
    slope = (closed_npv(RATE, lam + h) - closed_npv(RATE, lam - h)) / (2 * h)
    covariance = np.asarray(fitted.components["p"].reliability.hess_inv, float)
    assert covariance[0, 0] == pytest.approx(variance, rel=1e-3)
    assert spread.delta_variance == pytest.approx(
        slope**2 * covariance[0, 0], rel=1e-3
    )
    assert spread.variance_shares == {"p": pytest.approx(1.0)}


def test_the_spread_of_the_draws_is_the_delta_method_s(spread):
    # The NPV is nearly linear in the rate over its spread (a few per cent
    # of it): the draws' variance is the delta method's, to the error of a
    # variance from 400 draws (about 7%).
    assert spread.std**2 == pytest.approx(spread.delta_variance, rel=0.25)
    lower, upper = spread.interval(0.9)
    half = 1.6449 * math.sqrt(spread.delta_variance)
    assert lower == pytest.approx(spread.nominal - half, abs=0.3 * half)
    assert upper == pytest.approx(spread.nominal + half, abs=0.3 * half)


def test_each_draw_is_the_npv_of_its_models(fitted, spread):
    inputs, groups = _repairable_uncertainty.sources(fitted, None)
    models, group_models = _repairable_uncertainty.draws(
        fitted, inputs, groups, 400, 3
    )
    for i in (0, 199, 399):
        lam = float(models[0]["reliability"][i].params[0])
        assert spread.samples[i] == pytest.approx(
            closed_npv(RATE, lam), rel=1e-7
        )


def test_the_spread_says_what_it_held_and_goes_through_json(spread):
    assert isinstance(spread, NPVUncertainty)
    assert spread.band == "confidence"
    assert spread.held_fixed == [{"node": "p", "role": "repairability"}]
    out = json.loads(json.dumps(spread.to_dict()))
    assert len(out["samples"]) == 400
    assert out["held_fixed"] == [{"node": "p", "role": "repairability"}]
    assert out["variance_shares"] == {"p": pytest.approx(1.0)}
    assert out["horizon"] == HORIZON
    assert "NPVUncertainty(nominal=" in repr(spread)


def test_a_seed_gives_the_same_draws(fitted):
    def run():
        return fitted.expected_npv_uncertainty(
            HORIZON, value_rate=VALUE, n_draws=5, seed=11, shares=False
        )

    first, second = run(), run()
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.variance_shares == {}
    assert first.delta_variance is None


def test_two_inputs_share_the_variance():
    life = surv.Exponential.fit(FAILURES)
    repairs = np.random.default_rng(1).exponential(1.0 / MU, 15)
    rbd = pump(life)
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": life,
                "repairability": surv.Exponential.fit(repairs),
                "repair_cost": PRICE,
            }
        },
    )
    result = rbd.expected_npv_uncertainty(
        HORIZON,
        {"p": "fit"},
        discount_rate=RATE,
        value_rate=VALUE,
        n_draws=3,
        seed=0,
    )
    assert sum(result.variance_shares.values()) == pytest.approx(1.0)
    assert result.held_fixed == []


# A unit replaced at an age: 1000 costs it nothing extra under the model
# as given, but under a shorter-lived model, 750 is better.
ALLOWED = [500.0, 750.0, 1000.0, 1500.0]
LONG, SHORT = W([1500.0, 3.0]), W([900.0, 3.0])


def replaced(life):
    return RepairableRBD(
        [("s", "u"), ("u", "t")],
        {
            "u": {
                "reliability": life,
                "repairability": E([0.1]),
                "replace_cost": 5000.0,
                "preventive": {
                    "interval": 1000.0,
                    "duration": E([0.5]),
                    "cost": 1000.0,
                },
            }
        },
        downtime_cost_rate=50.0,
    )


def test_the_choice_flips_in_the_draws_whose_model_says_so():
    rbd = replaced(LONG)
    own = rbd.optimal_replacement_intervals(allowed=ALLOWED)
    short = replaced(SHORT).optimal_replacement_intervals(allowed=ALLOWED)
    # The two models choose differently, one step apart.
    i = ALLOWED.index(own.intervals["u"])
    assert abs(ALLOWED.index(short.intervals["u"]) - i) == 1
    uncertainty = {"u": {"reliability": [LONG, SHORT]}}
    plan = rbd.optimal_replacement_intervals(
        allowed=ALLOWED,
        robust=True,
        uncertainty=uncertainty,
        n_draws=60,
        seed=5,
    )
    robustness = plan.robustness
    assert isinstance(robustness, PlanRobustness)
    assert plan.intervals == own.intervals
    inputs, groups = _repairable_uncertainty.sources(rbd, uncertainty)
    models, _ = _repairable_uncertainty.draws(rbd, inputs, groups, 60, 5)
    drew_short = np.array([m is SHORT for m in models[0]["reliability"]])
    assert 0 < drew_short.sum() < 60
    # The choice is the best exactly where the long life was drawn.
    assert robustness.share_best == pytest.approx(1.0 - drew_short.mean())
    # And loses, where it was not, what the short life's own choice
    # saves on its cost rate.
    flipped = replaced(SHORT)
    loss = flipped.with_intervals(own).expected_cost_rate() - (
        flipped.with_intervals(short).expected_cost_rate()
    )
    assert loss > 0.0
    np.testing.assert_allclose(robustness.regret[~drew_short], 0.0, atol=0)
    np.testing.assert_allclose(robustness.regret[drew_short], loss, rtol=1e-9)
    assert robustness.expected_regret == pytest.approx(
        loss * drew_short.mean(), rel=1e-9
    )
    assert robustness.held_fixed == [
        {"node": "u", "role": "repairability"},
        {"node": "u", "role": "preventive.duration"},
    ]
    # Each draw's own choice, chosen again, agrees.
    again = rbd.optimal_replacement_intervals(
        allowed=ALLOWED,
        robust="reoptimise",
        uncertainty=uncertainty,
        n_draws=60,
        seed=5,
    ).robustness
    assert again.share_best == robustness.share_best
    np.testing.assert_allclose(again.regret, robustness.regret, rtol=1e-12)
    out = json.loads(json.dumps(plan.to_dict()))
    assert out["robustness"]["share_best"] == robustness.share_best
    assert len(out["robustness"]["candidates"]) == 3


def test_a_continuous_choice_is_compared_with_its_neighbours():
    rbd = replaced(LONG)
    plan = rbd.optimal_replacement_intervals(
        robust=True,
        uncertainty={"u": {"reliability": [LONG]}},
        n_draws=4,
        seed=0,
    )
    robustness = plan.robustness
    best = plan.intervals["u"]
    assert [c["u"] for c in robustness.candidates] == pytest.approx(
        [best, best / 1.25, best * 1.25]
    )
    # Every draw is the fitted model: the choice is the best in each.
    assert robustness.share_best == 1.0
    assert robustness.share_best_by_candidate == [1.0, 0.0, 0.0]
    assert robustness.expected_regret == 0.0


def test_robust_is_checked():
    with pytest.raises(ValueError, match="robust must be"):
        replaced(LONG).optimal_replacement_intervals(robust="yes")
