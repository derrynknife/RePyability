"""
Tests Non-Repairable Optimal Replacement Time algorithms.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import numpy as np
import pytest
from surpyval import KaplanMeier, LogNormal, Weibull

from repyability.non_repairable import NonRepairable


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
    non_p_model = KaplanMeier.fit(surv_model.random(10000))
    nr_model = NonRepairable(non_p_model)
    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert 493 == pytest.approx(nr_model.find_optimal_replacement(), rel=1e-1)


def test_weibull_no_optimal_replacement():
    surv_model = Weibull.from_params((1000, 0.5))
    nr_model = NonRepairable(surv_model)

    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert nr_model.find_optimal_replacement() == np.inf

    surv_model = Weibull.from_params((1000, 0.5), gamma=1)
    nr_model = NonRepairable(surv_model)
    nr_model.set_costs_planned_and_unplanned(1, 5)

    with pytest.warns():
        assert nr_model.find_optimal_replacement() != np.inf

    surv_model = Weibull.from_params((1000, 0.5), p=0.9)
    nr_model = NonRepairable(surv_model)
    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert nr_model.find_optimal_replacement() != np.inf

    surv_model = Weibull.from_params((1000, 0.5), f0=0.1)
    nr_model = NonRepairable(surv_model)
    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert nr_model.find_optimal_replacement() != np.inf


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
