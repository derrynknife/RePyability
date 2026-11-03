"""The calculators for each maintenance decision (``run_to_failure``,
``scheduled_replacement``, ``failure_finding`` and its interval,
``on_condition`` and its interval).

Each is checked against a closed form (an exponential life) and against a
seeded simulation of the plan itself, which does not use the formulas the
calculators are built on; the intervals against dense scans."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    NonRepairable,
    failure_finding,
    on_condition,
    optimal_failure_finding_interval,
    optimal_on_condition_interval,
    run_to_failure,
    scheduled_replacement,
)

SCALE, SHAPE = 365.0, 2.0


def bearing():
    return NonRepairable(
        surv.Weibull.from_params([SCALE, SHAPE]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )


def weibull(rng, size, scale=SCALE, shape=SHAPE):
    return scale * rng.weibull(shape, size)


def ratio_and_error(numerator, denominator):
    """The ratio of means of paired samples and its standard error."""
    n = len(numerator)
    ratio = numerator.mean() / denominator.mean()
    resid = numerator - ratio * denominator
    return ratio, resid.std(ddof=1) / math.sqrt(n) / denominator.mean()


# -- run to failure ----------------------------------------------------------


def test_run_to_failure_of_an_exponential_life():
    part = NonRepairable(
        surv.Exponential.from_params([0.002]), unplanned_cost=900.0
    )
    numbers = run_to_failure(part)
    assert numbers.mean_life == pytest.approx(500.0)
    assert numbers.failure_rate == pytest.approx(0.002)
    assert numbers.cost_rate == pytest.approx(1.8)
    assert numbers.availability == 1.0


def test_run_to_failure_counts_the_time_to_replace():
    part = NonRepairable(
        surv.Exponential.from_params([0.002]),
        time_to_replace=surv.Exponential.from_params([0.1]),
        unplanned_cost=900.0,
    )
    numbers = run_to_failure(part)
    assert numbers.mean_replacement_time == pytest.approx(10.0)
    assert numbers.failure_rate == pytest.approx(1 / 510.0)
    assert numbers.availability == pytest.approx(500.0 / 510.0)
    assert numbers.cost_rate == pytest.approx(900.0 / 510.0)


def test_run_to_failure_without_a_cost_has_no_cost_rate():
    part = NonRepairable(surv.Exponential.from_params([0.002]))
    assert run_to_failure(part).cost_rate is None


# -- scheduled replacement ---------------------------------------------------


@pytest.mark.parametrize("interval", [120.0, 250.0, 500.0])
def test_scheduled_replacement_matches_simulated_cycles(interval):
    plan = scheduled_replacement(bearing(), interval)
    life = weibull(np.random.default_rng(4), 500_000)
    failed = life < interval
    cycle = np.minimum(life, interval)
    for exact, numerator in (
        (plan.cost_rate, np.where(failed, 400.0, 150.0)),
        (plan.failure_rate, failed.astype(float)),
        (plan.replacement_rate, (~failed).astype(float)),
    ):
        got, error = ratio_and_error(numerator, cycle)
        assert abs(exact - got) < 4 * error
    assert plan.failure_probability == pytest.approx(failed.mean(), abs=4e-3)
    assert plan.mean_cycle == pytest.approx(cycle.mean(), rel=4e-3)


def test_the_cheapest_schedule_is_found_and_beats_running_to_failure():
    part = bearing()
    best = scheduled_replacement(part)
    assert best.optimal and best.worth_it
    assert best.interval == pytest.approx(
        part.optimal_replacement_policy().interval
    )
    ages = np.arange(50.0, 900.0, 1.0)
    rates = [scheduled_replacement(part, a).cost_rate for a in ages]
    assert best.cost_rate <= min(rates) + 1e-9
    assert best.saving_rate == pytest.approx(
        best.run_to_failure_cost_rate - best.cost_rate
    )


def test_a_schedule_does_not_pay_for_a_part_that_does_not_wear_out():
    part = NonRepairable(
        surv.Exponential.from_params([0.002]),
        planned_cost=100.0,
        unplanned_cost=400.0,
    )
    assert not scheduled_replacement(part, 300.0).worth_it
    with pytest.raises(ValueError, match="never pays"):
        scheduled_replacement(part)


# -- failure finding ---------------------------------------------------------


def valve(rate=0.001):
    return NonRepairable(surv.Exponential.from_params([rate]))


@pytest.mark.parametrize("tau", [0.5, 2.0, 40.0, 900.0])
def test_failure_finding_has_the_exponential_closed_form(tau):
    lam = 0.001
    result = failure_finding(valve(lam), tau, demand_rate=0.02)
    expected = 1.0 - (1.0 - math.exp(-lam * tau)) / (lam * tau)
    assert result.mean_unavailability == pytest.approx(expected, rel=1e-12)
    assert result.multiple_failure_rate == pytest.approx(0.02 * expected)
    assert result.tests_rate == pytest.approx(1.0 / tau)
    if tau < 5.0:  # the textbook approximation, lambda tau / 2
        assert result.mean_unavailability == pytest.approx(
            lam * tau / 2, rel=2e-3
        )


def test_failure_finding_matches_simulated_tests_for_a_weibull_device():
    part = NonRepairable(surv.Weibull.from_params([50.0, 2.5]))
    tau = 12.0
    result = failure_finding(part, tau)
    life = weibull(np.random.default_rng(9), 500_000, 50.0, 2.5)
    tests = np.ceil(life / tau)
    found, error = ratio_and_error(tests * tau - life, tests * tau)
    assert abs(result.mean_unavailability - found) < 4 * error
    rate, rate_error = ratio_and_error(np.ones_like(life), tests * tau)
    assert abs(result.failures_found_rate - rate) < 4 * rate_error


def test_the_interval_for_a_target_meets_it_and_is_the_longest():
    lam, target = 0.001, 0.001
    result = optimal_failure_finding_interval(
        valve(lam), target_unavailability=target
    )
    assert result.basis == "target"
    assert result.mean_unavailability == pytest.approx(target, rel=1e-9)
    # About 2 target / lambda for a small unavailability.
    assert result.interval == pytest.approx(2 * target / lam, rel=2e-3)
    longer = failure_finding(valve(lam), result.interval * 1.05)
    assert longer.mean_unavailability > target


def test_the_cheapest_failure_finding_interval_beats_a_dense_scan():
    part = NonRepairable(surv.Weibull.from_params([20.0, 2.0]))
    prices = dict(
        test_cost=10.0,
        repair_cost=50.0,
        demand_rate=0.1,
        consequence_cost=1e5,
    )
    best = optimal_failure_finding_interval(part, **prices)
    assert best.basis == "cost"
    scan = [
        failure_finding(part, tau, **prices).cost_rate
        for tau in np.concatenate(
            [np.linspace(0.01, 60.0, 600), np.linspace(0.15, 0.25, 801)]
        )
    ]
    assert best.cost_rate <= min(scan) + 1e-9
    assert best.cost_rate == pytest.approx(min(scan), rel=1e-4)


def test_failure_finding_inputs_are_checked():
    with pytest.raises(ValueError, match="demand_rate"):
        failure_finding(valve(), 2.0, consequence_cost=10.0)
    with pytest.raises(ValueError, match="test_interval"):
        failure_finding(valve(), 0.0)
    with pytest.raises(ValueError, match="target_unavailability"):
        optimal_failure_finding_interval(valve())
    with pytest.raises(ValueError, match="probability"):
        optimal_failure_finding_interval(valve(), target_unavailability=2.0)


# -- on condition ------------------------------------------------------------


def simulated_detection(warning, tau, p, n=400_000, seed=6):
    """The share of warnings found: inspections at a uniform phase, each
    finding the warning with probability p."""
    rng = np.random.default_rng(seed)
    length = (
        np.full(n, warning)
        if isinstance(warning, float)
        else warning.qf(rng.random(n))
    )
    first = rng.random(n) * tau
    inspections = np.where(
        length >= first, np.floor((length - first) / tau) + 1, 0
    )
    found = rng.random(n) < 1.0 - (1.0 - p) ** inspections
    return found.mean(), found.std(ddof=1) / math.sqrt(n)


@pytest.mark.parametrize(
    "tau, p", [(20.0, 1.0), (60.0, 1.0), (45.0, 0.5), (12.0, 0.3)]
)
def test_a_fixed_warning_is_found_as_simulated(tau, p):
    result = on_condition(
        30.0, tau, onset_rate=1 / 400, detection_probability=p
    )
    got, error = simulated_detection(30.0, tau, p)
    assert abs(result.detection_probability - got) <= 4 * error + 1e-12


def test_a_fixed_warning_has_the_closed_form_for_a_certain_inspection():
    # Found when an inspection falls in the warning: x / tau of the time.
    assert on_condition(30.0, 60.0, onset_rate=1.0).detection_probability == (
        pytest.approx(0.5)
    )
    assert on_condition(30.0, 30.0, onset_rate=1.0).detection_probability == (
        pytest.approx(1.0)
    )


@pytest.mark.parametrize("tau, p", [(20.0, 1.0), (50.0, 0.8), (15.0, 0.4)])
def test_a_random_warning_is_found_as_simulated(tau, p):
    warning = surv.Weibull.from_params([30.0, 2.0])
    result = on_condition(
        warning, tau, onset_rate=1 / 400, detection_probability=p
    )
    got, error = simulated_detection(warning, tau, p)
    assert abs(result.detection_probability - got) <= 4 * error + 1e-12


def test_on_condition_rates_and_costs_add_up():
    result = on_condition(
        surv.Weibull.from_params([30.0, 2.0]),
        20.0,
        onset_rate=0.01,
        detection_probability=0.9,
        inspection_cost=20.0,
        found_cost=300.0,
        failure_cost=5000.0,
    )
    assert result.found_rate + result.failure_rate == pytest.approx(0.01)
    assert result.cost_rate == pytest.approx(
        20.0 / 20.0 + result.found_rate * 300.0 + result.failure_rate * 5000.0
    )


def test_the_cheapest_inspection_interval_beats_a_dense_scan():
    warning = surv.Weibull.from_params([30.0, 2.0])
    args = dict(
        onset_rate=1 / 400,
        inspection_cost=20.0,
        found_cost=300.0,
        failure_cost=5000.0,
        detection_probability=0.9,
    )
    best = optimal_on_condition_interval(warning, **args)
    scan = [
        on_condition(warning, tau, **args).cost_rate
        for tau in np.linspace(1.0, 150.0, 600)
    ]
    assert best.cost_rate <= min(scan) + 1e-9
    assert best.cost_rate == pytest.approx(min(scan), rel=1e-3)


def test_a_fixed_warning_is_inspected_as_often_as_the_warning_is_long():
    best = optimal_on_condition_interval(
        30.0,
        onset_rate=1 / 400,
        inspection_cost=20.0,
        found_cost=300.0,
        failure_cost=5000.0,
    )
    assert best.interval == pytest.approx(30.0)
    assert best.detection_probability == pytest.approx(1.0)


def test_on_condition_inputs_are_checked():
    with pytest.raises(ValueError, match="detection_probability"):
        on_condition(30.0, 10.0, onset_rate=0.1, detection_probability=0.0)
    with pytest.raises(TypeError, match="pf_interval"):
        on_condition("soon", 10.0, onset_rate=0.1)
    with pytest.raises(ValueError, match="inspection_cost"):
        optimal_on_condition_interval(
            30.0,
            onset_rate=0.1,
            inspection_cost=0.0,
            found_cost=1.0,
            failure_cost=2.0,
        )
