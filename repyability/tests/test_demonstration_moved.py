"""The demonstration functions are surpyval's (``surpyval.test_planning``):
each of RePyability's warns that it is deprecated, naming surpyval's, and
gives surpyval's answer, ``confidence`` given as ``alpha_ci = 1 -
confidence``. They are removed in 0.19."""

import itertools
import warnings

import numpy as np
import pytest
from surpyval import test_planning

import repyability
from repyability import DemonstrationPlan

NAMES = [
    "demonstration_sample_size",
    "demonstrated_reliability",
    "demonstration_test_multiple",
    "demonstration_pass_probability",
    "mtbf_test_time",
    "demonstrated_mtbf",
    "mtbf_pass_probability",
    "demonstration_plan",
    "mtbf_demonstration_plan",
]

# One call of each: RePyability's arguments, then surpyval's.
CALLS = {
    "demonstration_sample_size": ((0.95, 0.9, 1), (0.95, 1), {}),
    "demonstrated_reliability": ((59, 0.9, 1), (59, 1), {}),
    "demonstration_test_multiple": (
        (0.95, 20, 0.9, 1),
        (0.95, 20, 1),
        {"shape": 2.0},
    ),
    "demonstration_pass_probability": ((0.95, 59, 1), (0.95, 59, 1), None),
    "mtbf_test_time": ((1000.0, 0.9, 2), (1000.0, 2), {}),
    "demonstrated_mtbf": ((5000.0, 0.9, 2), (5000.0, 2), {}),
    "mtbf_pass_probability": ((1000.0, 5000.0, 2), (1000.0, 5000.0, 2), None),
    "demonstration_plan": ((0.9, 0.95, 0.9), (0.9, 0.95), {}),
    "mtbf_demonstration_plan": ((1000.0, 2000.0, 0.9), (1000.0, 2000.0), {}),
}


def _surpyval(name, args, kwargs):
    if kwargs is None:
        return getattr(test_planning, name)(*args)
    return getattr(test_planning, name)(*args, alpha_ci=0.1, **kwargs)


def _fields(plan):
    return [
        plan.n,
        plan.test_multiple,
        plan.test_time,
        plan.failures,
        plan.consumer_risk,
        plan.producer_risk,
    ]


@pytest.mark.parametrize("name", NAMES)
def test_each_warns_naming_surpyvals_function(name):
    ours, theirs, kwargs = CALLS[name]
    function = getattr(repyability, name)
    match = (
        f"repyability.{name} is deprecated: use surpyval.test_planning."
        f"{name},.*removed in 0.19"
    )
    with pytest.warns(FutureWarning, match=match) as caught:
        got = function(*ours, **(kwargs or {}))
    # The warning points at the caller, not inside RePyability.
    assert caught[0].filename == __file__
    expected = _surpyval(name, theirs, kwargs)
    if name.endswith("plan"):
        assert isinstance(got, DemonstrationPlan)
        assert _fields(got) == _fields(expected)
    else:
        assert type(got) is type(expected)
        assert got == pytest.approx(expected, rel=1e-14)


@pytest.mark.filterwarnings("ignore:repyability.*is deprecated:FutureWarning")
def test_the_numbers_are_surpyvals_on_a_grid():
    # The comparison made before delegating (the PR records it): the
    # same counts and plans exactly, the probabilities within the last
    # bits that 1 - (1 - confidence) leaves.
    for r, c, f in itertools.product(
        [0.8, 0.95, 0.999], [0.6, 0.9, 0.975], [0, 1, 5]
    ):
        a = 1.0 - c
        assert repyability.demonstration_sample_size(
            r, c, f, test_multiple=2.0, shape=1.5
        ) == test_planning.demonstration_sample_size(
            r, f, alpha_ci=a, test_multiple=2.0, shape=1.5
        )
        assert repyability.demonstrated_reliability(60, c, f) == pytest.approx(
            test_planning.demonstrated_reliability(60, f, alpha_ci=a),
            rel=1e-14,
        )
        for terminated in (False, True):
            if terminated and f == 0:
                continue
            assert repyability.mtbf_test_time(
                1000.0, c, f, failure_terminated=terminated
            ) == pytest.approx(
                test_planning.mtbf_test_time(
                    1000.0, f, alpha_ci=a, failure_terminated=terminated
                ),
                rel=1e-14,
            )
    plan = repyability.demonstration_plan(0.9, 0.97, 0.9, n=60, shape=2.0)
    assert _fields(plan) == _fields(
        test_planning.demonstration_plan(
            0.9, 0.97, alpha_ci=0.1, n=60, shape=2.0
        )
    )


@pytest.mark.filterwarnings("ignore:repyability.*is deprecated:FutureWarning")
def test_arrays_are_answered_element_by_element():
    together = repyability.mtbf_test_time(
        np.array([1000.0, 2000.0]), np.array([0.9, 0.95]), 2
    )
    assert together.tolist() == [
        repyability.mtbf_test_time(1000.0, 0.9, 2),
        repyability.mtbf_test_time(2000.0, 0.95, 2),
    ]


@pytest.mark.filterwarnings("ignore:repyability.*is deprecated:FutureWarning")
def test_a_confidence_is_checked_as_a_confidence():
    for bad in (0.0, 1.0, 1.5, True, "high", np.nan):
        with pytest.raises(ValueError, match="confidence must be between"):
            repyability.demonstration_sample_size(0.9, bad)
    with pytest.raises(ValueError, match="confidence must be between"):
        repyability.mtbf_test_time(1000.0, np.array([0.9, 1.0]))
    # A low confidence is taken as given: surpyval's warning of an
    # alpha_ci above 0.5 (a confidence passed as alpha_ci) does not apply.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert repyability.demonstrated_reliability(10, 0.3) > 0.9
