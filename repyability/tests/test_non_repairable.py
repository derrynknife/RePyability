"""
Tests Non-Repairable Optimal Replacement Time algorithms.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import numpy as np
import pytest
from surpyval import KaplanMeier, LogNormal, Weibull

from repyability.non_repairable import NonRepairable


def test_optimal_replacement():
    surv_model = Weibull.from_params((1000, 2.5))
    nr_model = NonRepairable(surv_model)

    nr_model.set_costs_planned_and_unplanned(1, 5)
    assert 493 == pytest.approx(nr_model.find_optimal_replacement(), abs=1e-1)


def test_weibull_no_optimal_replacement():
    surv_model = Weibull.from_params((1000, 0.5))
    nr_model = NonRepairable(surv_model)

    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert nr_model.find_optimal_replacement() == np.inf


def test_incorrect_args():
    with pytest.raises(ValueError):
        NonRepairable(1)


def test_mean_availability():
    surv_model = Weibull.from_params((1000, 2.5))
    ttr_model = LogNormal.from_params((1.5, 0.1))
    nr_model = NonRepairable(surv_model, ttr_model)
    assert pytest.approx(nr_model.mean_availability()) == 0.9949491865865466

    non_p = KaplanMeier.fit([1, 2, 3, 4, 5, 6])
    model = NonRepairable(non_p, non_p)
    with pytest.raises(ValueError):
        model.mean_availability()
