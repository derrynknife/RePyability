"""What 0.17 deprecated is gone in 0.18: ``repyability.demonstration``'s
nine functions and ``DemonstrationPlan``, as test planning is surpyval's
(``surpyval.test_planning`` has the same functions under the same
names)."""

import importlib

import pytest
from surpyval import test_planning

import repyability

GONE = (
    "demonstration_sample_size",
    "demonstrated_reliability",
    "demonstration_test_multiple",
    "demonstration_pass_probability",
    "demonstration_plan",
    "mtbf_test_time",
    "demonstrated_mtbf",
    "mtbf_pass_probability",
    "mtbf_demonstration_plan",
)


@pytest.mark.parametrize("name", GONE)
def test_the_demonstration_functions_are_surpyvals(name):
    assert not hasattr(repyability, name)
    assert name not in repyability.__all__
    assert callable(getattr(test_planning, name))


def test_the_demonstration_module_and_its_plan_are_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("repyability.demonstration")
    assert not hasattr(repyability, "DemonstrationPlan")
