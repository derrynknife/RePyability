"""
Tests Non-Repairable Optimal Replacement Time algorithms.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import numpy as np
import pytest
import surpyval as surv
from scipy.special import gamma as gamma_function
from scipy.special import gammainc
from surpyval import KaplanMeier, LogNormal, Weibull

from repyability.non_repairable import NonRepairable
from repyability.utils.wrappers import numpy_seed


def test_optimal_replacement1():
    # https://www.weibull.com/hotwire/issue156/hottopics156.htm
    for beta, alpha, cu, cp, answer in [
        [6, 10000, 30, 5000, 3263.16],
        [2, 80000, 700, 4000, 37509.26],
        [2.5, 80000, 800, 4500, 37170.26],
        [3, 15000, 25, 1500, 3059.266],
        [1.5, 30000, 80, 250, 33029.446],
        [3, 30000, 70, 800, 10920.36],
    ]:
        surv_model = Weibull.from_params((alpha, beta))
        nr_model = NonRepairable(surv_model)
        nr_model.set_costs_planned_and_unplanned(cu, cp)

        assert answer == pytest.approx(
            nr_model.find_optimal_replacement(), rel=1e-2
        )


def test_optimal_replacement_across_scales():
    # If the characteristic life is 10x greater
    # then the optimal replacement time should be 10x
    answer = 0.3263162540188295
    for i in range(1, 10):
        surv_model = Weibull.from_params(((10 ** (i - 1)), 6))
        nr_model = NonRepairable(surv_model)

        nr_model.set_costs_planned_and_unplanned(30, 5000)
        answer_i = answer * (10 ** (i - 1))
        optimal = nr_model.find_optimal_replacement()
        assert answer_i == pytest.approx(optimal, rel=1e-1)


def test_optimal_replacement():
    # https://reliawiki.org/index.php/Optimum_Replacement_Time_Example
    surv_model = Weibull.from_params((1000, 2.5))
    nr_model = NonRepairable(surv_model)
    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert 493 == pytest.approx(nr_model.find_optimal_replacement(), abs=1e-1)


def test_non_parametric_optimal_replacement():
    # https://reliawiki.org/index.php/Optimum_Replacement_Time_Example
    surv_model = Weibull.from_params((1000, 2.5))
    # Seeded: unseeded, about 1 draw in 150 lands the fitted optimum outside
    # the 10% tolerance.
    with numpy_seed(1):
        data = surv_model.random(10000)
    non_p_model = KaplanMeier.fit(data)
    nr_model = NonRepairable(non_p_model)
    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert 493 == pytest.approx(nr_model.find_optimal_replacement(), rel=1e-1)


def test_weibull_no_optimal_replacement():
    surv_model = Weibull.from_params((1000, 0.5))
    nr_model = NonRepairable(surv_model)

    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert nr_model.find_optimal_replacement() == np.inf

    # An offset (3-parameter), zero-inflated or limited-failure-population
    # Weibull with beta <= 1 still does not wear out: the search runs, but
    # finds nothing cheaper than running to failure, so the answer is inf
    # (issue #68: it used to return wherever the search stopped, e.g.
    # 745,244 for the offset model below, whose mean life is 2001). The
    # policy's cost rate is then the run-to-failure rate: cu / MTTF, or 0
    # when some units never fail.
    for model, rate in [
        (Weibull.from_params((1000, 0.5), gamma=1), 5 / 2001),
        (Weibull.from_params((1000, 0.5), f0=0.1), 5 / 1800),
        (Weibull.from_params((1000, 0.5), p=0.9), 0.0),
        (Weibull.from_params((1000, 3.0), p=0.9), 0.0),
        (surv.Gamma.from_params((0.5, 0.001)), 5 / 500),
    ]:
        nr_model = NonRepairable(model)
        nr_model.set_costs_planned_and_unplanned(1, 5)
        assert nr_model.find_optimal_replacement() == np.inf
        policy = nr_model.optimal_replacement_policy()
        assert policy.interval == np.inf
        assert policy.cost_rate == pytest.approx(rate, rel=1e-9)


def weibull_cost_rates(ages, eta, beta, cp, cu, gamma=0.0, f0=0.0):
    """Closed-form age-replacement cost rates of a Weibull with an offset
    ``gamma`` or zero-inflation ``f0``: the cycle length, the integral of R
    up to the age, is a regularised incomplete gamma function."""
    ages = np.asarray(ages, dtype=float)
    shifted = np.clip(ages - gamma, 0.0, None)
    R = (1 - f0) * np.exp(-((shifted / eta) ** beta))
    tail = eta * gamma_function(1 + 1 / beta)
    tail = tail * gammainc(1 / beta, (shifted / eta) ** beta)
    cycle = (1 - f0) * (np.minimum(ages, gamma) + tail)
    return (cp * R + cu * (1 - R)) / cycle


def test_replacing_at_the_end_of_a_failure_free_period():
    # No unit fails before the offset, so up to it the cost rate is cp / t,
    # and here replacing right at the offset beats both running to failure
    # (5 / 1020) and any later age; the search used to stop at 986.
    unit = NonRepairable(Weibull.from_params((10, 0.5), gamma=1000))
    unit.set_costs_planned_and_unplanned(1, 5)
    assert unit.find_optimal_replacement() == 1000
    policy = unit.optimal_replacement_policy()
    assert policy.cost_rate == pytest.approx(1 / 1000)
    ages = np.unique(np.concatenate([np.linspace(1, 5000, 100_000), [1000]]))
    rates = weibull_cost_rates(ages, 10, 0.5, 1, 5, gamma=1000)
    assert ages[np.argmin(rates)] == 1000
    assert policy.cost_rate == pytest.approx(rates.min(), rel=1e-12)


@pytest.mark.parametrize(
    "gamma, f0",
    [(0.0, 0.0), (100.0, 0.0), (0.0, 0.1)],
)
def test_wear_out_optimum_matches_the_closed_form(gamma, f0):
    # With wear-out a finite age pays, with or without an offset or
    # zero-inflation: the search's optimum is that of the closed-form cost
    # rate on a fine grid.
    model = Weibull.from_params(
        (1000, 2.5), gamma=gamma or None, f0=f0 or None
    )
    unit = NonRepairable(model)
    unit.set_costs_planned_and_unplanned(1, 5)
    optimal = unit.find_optimal_replacement()
    ages = np.linspace(1, 3000, 300_000)
    rates = weibull_cost_rates(ages, 1000, 2.5, 1, 5, gamma=gamma, f0=f0)
    assert optimal == pytest.approx(ages[np.argmin(rates)], abs=0.02)
    assert unit.cost_rate(optimal) == pytest.approx(rates.min(), rel=1e-9)
    assert rates.min() < 5 / float(np.ravel(model.mean())[0])


def test_incorrect_args():
    with pytest.raises(ValueError):
        NonRepairable(1)


def test_mean_availability():
    surv_model = Weibull.from_params((1000, 2.5))
    ttr_model = LogNormal.from_params((1.5, 0.1))
    nr_model = NonRepairable(surv_model, ttr_model)
    assert pytest.approx(nr_model.mean_availability()) == 0.9949491865865466
    assert (
        pytest.approx(nr_model.mean_unavailability()) == 1 - 0.9949491865865466
    )

    non_p = KaplanMeier.fit([1, 2, 3, 4, 5, 6])
    model = NonRepairable(non_p, non_p)
    with pytest.raises(ValueError):
        model.mean_availability()


def test_error_when_cp_gt_cu():
    surv_model = Weibull.from_params((1000, 2.5))
    nr_model = NonRepairable(surv_model)
    with pytest.raises(ValueError):
        nr_model.set_costs_planned_and_unplanned(10, 1)


def test_non_parametric_mean():
    surv_model = Weibull.from_params((1000, 2.5))
    x = surv_model.random(10000)
    non_para_model = KaplanMeier.fit(x)
    np_nr_model = NonRepairable(non_para_model)
    p_nr_model = NonRepairable(surv_model)
    assert pytest.approx(
        np_nr_model.avg_replacement_time(1000), rel=1e-1
    ) == p_nr_model.avg_replacement_time(1000)


def test_cost_rates():
    surv_model = Weibull.from_params((1000, 2.5))
    nr_model = NonRepairable(surv_model)
    nr_model.set_costs_planned_and_unplanned(1, 10)
    assert pytest.approx(nr_model._log_cost_rate(1000)) == np.log(
        nr_model.cost_rate(1000)
    )


def test_exponential_never_replaced():
    # Memoryless lifetime: an old unit is statistically as good as new,
    # so preventive replacement can never pay.
    from surpyval import Exponential

    nr_model = NonRepairable(Exponential.from_params([0.001]))
    nr_model.set_costs_planned_and_unplanned(1, 5)
    assert nr_model.find_optimal_replacement() == np.inf

    policy = nr_model.optimal_replacement_policy()
    assert policy.interval == np.inf
    # Run-to-failure cost rate: cu / MTTF = 5 / 1000.
    assert policy.cost_rate == pytest.approx(5 * 0.001)


def test_optimal_replacement_policy():
    surv_model = Weibull.from_params((1000, 2.5))
    nr_model = NonRepairable(surv_model)
    nr_model.set_costs_planned_and_unplanned(1, 5)

    policy = nr_model.optimal_replacement_policy()
    assert policy.interval == pytest.approx(
        nr_model.find_optimal_replacement()
    )
    assert policy.cost_rate == pytest.approx(
        float(nr_model.cost_rate(policy.interval))
    )


def test_policy_requires_costs():
    nr_model = NonRepairable(Weibull.from_params((1000, 2.5)))
    with pytest.raises(ValueError, match="costs not set"):
        nr_model.optimal_replacement_policy()


def _erlang2_sf(rate, t):
    return np.exp(-rate * t) * (1 + rate * t)


@pytest.mark.parametrize(
    "form",
    ["closed form", "convolution", "warm"],
)
def test_a_standby_arrangement_as_the_lifetime(form):
    # Closed-form arrangements (identical exponential units, or cold k = 1)
    # used to fail at construction, and the cost methods failed for every
    # arrangement. Any of them now works through its survival function:
    # exact, a convolution, or warm standby's recursion.
    from repyability import StandbyModel

    if form == "closed form":
        standby = StandbyModel([surv.Exponential.from_params([0.001])] * 2)
    elif form == "convolution":
        standby = StandbyModel([surv.Weibull.from_params([1000, 2.5])] * 2)
    else:
        standby = StandbyModel(
            [surv.Weibull.from_params([1000, 2.5])] * 2,
            dormancy_factor=0.3,
        )
    unit = NonRepairable(standby, surv.Exponential.from_params([1 / 24]))
    unit.set_costs_planned_and_unplanned(1, 5)
    policy = unit.optimal_replacement_policy()
    assert 0 < policy.interval < np.inf
    # The optimum is a minimum of the cost rate.
    for factor in (0.95, 0.99, 1.01, 1.05):
        assert unit.cost_rate(policy.interval * factor) >= policy.cost_rate
    assert 0 < unit.mean_availability() < 1


def test_standby_replacement_matches_the_closed_form():
    # Two identical exponential units in cold standby live an Erlang(2)
    # time, so the age-replacement optimum can be found independently.
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar

    from repyability import StandbyModel

    rate = 0.001

    def cost_rate(t):
        R = _erlang2_sf(rate, t)
        return (1 * R + 5 * (1 - R)) / quad(
            lambda u: _erlang2_sf(rate, u), 0, t
        )[0]

    expected = minimize_scalar(
        cost_rate, bounds=(10, 10_000), method="bounded"
    )
    standby = StandbyModel([surv.Exponential.from_params([rate])] * 2)
    unit = NonRepairable(standby)
    unit.set_costs_planned_and_unplanned(1, 5)
    assert unit.find_optimal_replacement() == pytest.approx(
        expected.x, rel=1e-4
    )
    assert unit.avg_replacement_time(1000) == pytest.approx(
        quad(lambda u: _erlang2_sf(rate, u), 0, 1000)[0]
    )


def test_a_standby_that_may_never_fail_is_never_replaced():
    # With a limited failure population a unit may never fail, so the cost
    # rate keeps falling with the replacement age.
    from repyability import StandbyModel

    never = surv.Weibull.from_params([1000, 2.5], p=0.6)
    unit = NonRepairable(StandbyModel([never, never]))
    unit.set_costs_planned_and_unplanned(1, 5)
    assert unit.find_optimal_replacement() == np.inf


def test_a_closed_form_standby_component_in_a_repairable_rbd():
    from repyability import RepairableRBD, StandbyModel

    standby = StandbyModel([surv.Exponential.from_params([0.001])] * 2)
    rbd = RepairableRBD(
        [("s", "pumps"), ("pumps", "t")],
        {
            "pumps": {
                "reliability": standby,
                "repairability": surv.Exponential.from_params([1 / 24]),
            }
        },
    )
    # MTTF 2000, MTTR 24.
    assert rbd.mean_availability() == pytest.approx(2000 / 2024)


# Kaplan-Meier failures at 10, 20, 30 and 40: linear between them and from
# 1 at age 0, the survival function is exactly R(u) = 1 - u / 40.
_UNIFORM_KM = [10, 20, 30, 40]


def test_non_parametric_cycle_length_is_exact():
    # The cycle length used to start at the first time point and stop at
    # the last one below t (giving 0, 5 and 7.5 here), and the linear
    # extrapolation beyond the data made the survival negative.
    unit = NonRepairable(KaplanMeier.fit(_UNIFORM_KM))
    for t in (5, 25, 40, 60):
        age = min(t, 40)
        assert unit.avg_replacement_time(t) == pytest.approx(age - age**2 / 80)
    assert unit.reliability_function(60) == 0.0
    # With the last time censored, the survival is held beyond it.
    censored = NonRepairable(KaplanMeier.fit(_UNIFORM_KM, c=[0, 0, 0, 1]))
    assert censored.reliability_function(80) == pytest.approx(0.25)
    assert censored.avg_replacement_time(80) == pytest.approx(21.25 + 10)


def test_non_parametric_optimum_is_exact_and_consistent():
    # The cost rate (1 * R + 5 * (1 - R)) / (t - t**2 / 80) is least at
    # t = 20, where it is 0.2. The search used to return 21.6, and the
    # policy's cost rate (0.63) disagreed with the search's.
    unit = NonRepairable(KaplanMeier.fit(_UNIFORM_KM))
    unit.set_costs_planned_and_unplanned(1, 5)
    assert unit.find_optimal_replacement() == pytest.approx(20.0)
    policy = unit.optimal_replacement_policy()
    assert policy.interval == pytest.approx(20.0)
    assert policy.cost_rate == pytest.approx(0.2)


def _lifetimes():
    from repyability import StandbyModel

    return {
        "parametric": Weibull.from_params([1000, 2.5]),
        "non-parametric": KaplanMeier.fit(_UNIFORM_KM),
        "standby": StandbyModel([surv.Exponential.from_params([0.001])] * 2),
    }


@pytest.mark.parametrize("kind", ["parametric", "non-parametric", "standby"])
def test_missing_costs_raise_value_error(kind):
    # find_optimal_replacement() and cost_rate used to raise AttributeError,
    # optimal_replacement_policy() ValueError.
    unit = NonRepairable(_lifetimes()[kind])
    for call in (
        lambda: unit.cost_rate(100.0),
        unit.find_optimal_replacement,
        unit.optimal_replacement_policy,
    ):
        with pytest.raises(ValueError, match="costs not set"):
            call()


def test_never_replacing_needs_no_costs():
    unit = NonRepairable(surv.Exponential.from_params([0.001]))
    assert unit.find_optimal_replacement() == np.inf


@pytest.mark.parametrize(
    "cp, cu",
    [(-1.0, 5.0), (np.nan, 5.0), (1.0, np.inf), (1.0, np.nan)],
    ids=["negative", "nan planned", "infinite unplanned", "nan unplanned"],
)
def test_invalid_costs_are_rejected(cp, cu):
    unit = NonRepairable(Weibull.from_params([1000, 2.5]))
    with pytest.raises(ValueError, match="costs must be"):
        unit.set_costs_planned_and_unplanned(cp, cu)
    unit.set_costs_planned_and_unplanned(0.0, 5.0)  # a free planned swap
