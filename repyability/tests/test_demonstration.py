"""Reliability demonstration test planning (#129): each plan against its
forward function, textbook values, and the special cases that reduce to
simpler plans."""

import inspect
import math

import numpy as np
import pytest
import surpyval
from scipy import stats

from repyability import (
    demonstrated_mtbf,
    demonstrated_reliability,
    demonstration_pass_probability,
    demonstration_sample_size,
    demonstration_test_multiple,
    mtbf_pass_probability,
    mtbf_test_time,
)


@pytest.mark.parametrize(
    "reliability, confidence, failures, n",
    [
        (0.95, 0.95, 0, 59),
        (0.95, 0.95, 1, 93),
        (0.90, 0.90, 0, 22),
        (0.99, 0.90, 0, 230),
        (0.999, 0.90, 0, 2302),
        (0.90, 0.90, 2, 52),
    ],
)
def test_sample_sizes_match_textbook_values(
    reliability, confidence, failures, n
):
    assert demonstration_sample_size(reliability, confidence, failures) == n


@pytest.mark.parametrize("reliability", [0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
@pytest.mark.parametrize("confidence", [0.5, 0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("failures", [0, 1, 3])
def test_the_planned_test_is_the_smallest_that_demonstrates(
    reliability, confidence, failures
):
    n = demonstration_sample_size(reliability, confidence, failures)
    assert demonstrated_reliability(n, confidence, failures) >= reliability
    if n - 1 > failures:
        shorter = demonstrated_reliability(n - 1, confidence, failures)
        assert shorter < reliability
    # The planned test passes a design at the target no more often than
    # 1 - confidence, and one unit fewer would pass it more often.
    risk = demonstration_pass_probability(reliability, n, failures)
    assert risk <= 1.0 - confidence + 1e-12
    if n - 1 > failures:
        fewer = demonstration_pass_probability(reliability, n - 1, failures)
        assert fewer > 1.0 - confidence


def _success_run(n, confidence):
    """surpyval's bound after ``n`` successes, by the name the installed
    surpyval takes: ``alpha_ci = 1 - confidence`` from 0.23, which
    deprecates ``confidence`` (surpyval #580), and ``confidence`` before."""
    if "alpha_ci" in inspect.signature(surpyval.success_run).parameters:
        return float(surpyval.success_run(n, alpha_ci=1.0 - confidence))
    return float(surpyval.success_run(n, confidence=confidence))


@pytest.mark.parametrize("n", [1, 5, 59, 1000])
@pytest.mark.parametrize("confidence", [0.6, 0.9, 0.95])
def test_no_failures_is_surpyvals_success_run(n, confidence):
    expected = _success_run(n, confidence)
    assert demonstrated_reliability(n, confidence) == pytest.approx(
        expected, rel=1e-12
    )


def test_the_bound_with_failures_is_clopper_pearson():
    # The chance of so few failures, at the bound, is exactly 1 - C.
    n, failures, confidence = 40, 3, 0.9
    bound = demonstrated_reliability(n, confidence, failures)
    chance = stats.binom.cdf(failures, n, 1.0 - bound)
    assert chance == pytest.approx(1.0 - confidence, rel=1e-9)
    assert demonstrated_reliability(10, 0.9, failures=10) == 0.0


def test_a_one_mission_test_ignores_the_shape():
    assert demonstration_sample_size(
        0.95, 0.9, test_multiple=1.0, shape=3.0
    ) == demonstration_sample_size(0.95, 0.9)
    assert demonstrated_reliability(
        30, 0.9, 1, test_multiple=1.0, shape=0.5
    ) == demonstrated_reliability(30, 0.9, 1)


@pytest.mark.parametrize("k, beta", [(2.0, 2.0), (1.5, 3.0), (0.5, 1.0)])
def test_weibayes_trades_units_for_test_time(k, beta):
    reliability, confidence = 0.95, 0.9
    n = demonstration_sample_size(
        reliability, confidence, test_multiple=k, shape=beta
    )
    closed_form = math.log(1 - confidence) / (k**beta * math.log(reliability))
    assert n == math.ceil(closed_form)
    shown = demonstrated_reliability(
        n, confidence, test_multiple=k, shape=beta
    )
    assert shown == pytest.approx(_success_run(n, confidence) ** (1 / k**beta))
    assert shown >= reliability


@pytest.mark.parametrize("failures", [0, 2])
def test_the_test_multiple_demonstrates_exactly_the_target(failures):
    reliability, n, confidence, beta = 0.97, 25, 0.9, 2.5
    k = demonstration_test_multiple(
        reliability, n, confidence, failures, shape=beta
    )
    shown = demonstrated_reliability(
        n, confidence, failures, test_multiple=k, shape=beta
    )
    assert shown == pytest.approx(reliability, rel=1e-12)
    # Enough units for a one-mission test need less than a mission each.
    many = demonstration_sample_size(reliability, confidence, failures) + 50
    assert (
        demonstration_test_multiple(
            reliability, many, confidence, failures, shape=beta
        )
        < 1.0
    )


@pytest.mark.parametrize(
    "confidence, failures, multiple",
    [(0.9, 0, 2.303), (0.9, 1, 3.890), (0.9, 2, 5.322), (0.8, 0, 1.609)],
)
def test_mtbf_test_times_match_textbook_values(confidence, failures, multiple):
    time = mtbf_test_time(1000.0, confidence, failures)
    assert time == pytest.approx(1000.0 * multiple, abs=0.5)


@pytest.mark.parametrize("failures", [0, 1, 5])
@pytest.mark.parametrize("confidence", [0.6, 0.9, 0.99])
def test_the_mtbf_plan_and_bound_are_inverses(failures, confidence):
    time = mtbf_test_time(250.0, confidence, failures)
    assert demonstrated_mtbf(time, confidence, failures) == pytest.approx(
        250.0, rel=1e-12
    )
    # At the target MTBF the test passes exactly 1 - confidence of the time
    # (the Poisson-chi-squared duality).
    assert mtbf_pass_probability(250.0, time, failures) == pytest.approx(
        1.0 - confidence, rel=1e-9
    )


def test_better_designs_pass_more_often():
    reliabilities = np.linspace(0.9, 0.999, 12)
    chances = [demonstration_pass_probability(r, 40, 1) for r in reliabilities]
    assert np.all(np.diff(chances) > 0)
    mtbfs = np.linspace(100.0, 5000.0, 12)
    chances = [mtbf_pass_probability(m, 3000.0, 2) for m in mtbfs]
    assert np.all(np.diff(chances) > 0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: demonstration_sample_size(1.0),
        lambda: demonstration_sample_size(0.9, 1.2),
        lambda: demonstration_sample_size(0.9, failures=-1),
        lambda: demonstration_sample_size(0.9, failures=1.5),
        lambda: demonstration_sample_size(0.9, test_multiple=2.0),
        lambda: demonstration_sample_size(0.9, test_multiple=0.0, shape=2.0),
        lambda: demonstration_sample_size(0.9, test_multiple=2.0, shape=0.0),
        lambda: demonstrated_reliability(0),
        lambda: demonstrated_reliability(5, failures=6),
        lambda: demonstration_test_multiple(0.9, 3, failures=3, shape=2.0),
        lambda: demonstration_pass_probability(0.0, 10),
        lambda: mtbf_test_time(-1.0),
        lambda: demonstrated_mtbf(0.0),
        lambda: mtbf_pass_probability(100.0, -5.0),
    ],
)
def test_invalid_plans_are_refused(call):
    with pytest.raises(ValueError):
        call()
