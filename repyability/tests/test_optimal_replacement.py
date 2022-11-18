"""
Tests Non-Repairable Optimal Replacement Time algorithms.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import pytest
from surpyval import Weibull

from repyability.non_repairable import NonRepairable


def test_optimal_replacement():
    surv_model = Weibull.from_params((1000, 2.5))
    nr_model = NonRepairable(surv_model)

    nr_model.set_costs_planned_and_unplanned(1, 5)

    assert pytest.approx(nr_model.find_optimal_replacement(), abs=1e-1) == 493
