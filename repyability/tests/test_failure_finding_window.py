"""A hidden failure tested at given times over a window:
``failure_finding_window`` (UC-53 of ``repyability/maintenance/README.md``).

Checked against an exponential device's closed forms (a two-state chain
where demands reveal), ``failure_finding``'s long run, and a simulation
of the plan written out plainly here.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    FailureFindingWindow,
    NonRepairable,
    failure_finding,
    failure_finding_window,
)


def exponential(rate):
    return NonRepairable(surv.Exponential.from_params([rate]))


def closed_form(rate, tests, window, reveal=0.0, discount=0.0):
    """An exponential device tested at ``tests``: what each test finds,
    the failed time, and the failed time discounted. From a test (or the
    start) the chance of being failed after ``s`` is ``rate / k (1 -
    exp(-k s))``, ``k = rate + reveal``."""
    k = rate + reveal
    edges = [0.0, *[t for t in tests if 0.0 < t < window], window]
    found, failed, discounted = [], 0.0, 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        length = end - start
        found.append(rate / k * -math.expm1(-k * length))
        failed += rate / k * (length + math.expm1(-k * length) / k)
        if discount > 0.0:
            discounted += (
                rate
                / k
                * math.exp(-discount * start)
                * (
                    -math.expm1(-discount * length) / discount
                    + math.expm1(-(discount + k) * length) / (discount + k)
                )
            )
    if discount == 0.0:
        discounted = failed
    if not (tests and tests[-1] == window):
        found = found[:-1]
    return np.array(found), failed, discounted


def test_uc53_trip_tested_at_uneven_times():
    """The worked example: 0.1 a year, tests at 6 and 12 months and 2
    years, over 3 years; a demand every 2 years."""
    trip = exponential(0.1)
    prices = dict(
        demand_rate=0.5,
        test_cost=200.0,
        repair_cost=1500.0,
        consequence_cost=50_000.0,
    )
    tested = failure_finding_window(trip, [0.5, 1.0, 2.0], 3.0, **prices)
    assert isinstance(tested, FailureFindingWindow)
    np.testing.assert_allclose(
        tested.found_at_tests, [0.04320, 0.04320, 0.07520], atol=5e-6
    )
    assert tested.failures_found == pytest.approx(0.16159, abs=5e-6)
    assert tested.failed_time == pytest.approx(0.10535, abs=5e-6)
    assert tested.mean_unavailability == pytest.approx(0.03512, abs=5e-6)
    assert tested.demands_on_failed == pytest.approx(0.05267, abs=5e-6)
    assert tested.tests_cost == 600.0
    assert tested.repairs_cost == pytest.approx(321.40, abs=0.01)
    assert tested.consequence_cost == pytest.approx(2633.75, abs=0.01)
    assert tested.cost == pytest.approx(3555.15, abs=0.01)
    assert tested.present_value == tested.cost
    never = failure_finding_window(
        trip, [], 3.0, **{**prices, "test_cost": 0.0}
    )
    assert never.failed_time == pytest.approx(0.26814, abs=5e-6)
    assert never.cost == pytest.approx(6904.57, abs=0.01)
    assert never.failures_found == 0.0 and never.tests_cost == 0.0
    json.dumps(tested.to_dict())
    assert "FailureFindingWindow(3 tests over 3" in repr(tested)


@pytest.mark.parametrize("reveal", [True, False])
@pytest.mark.parametrize("discount", [0.0, 0.08])
def test_exponential_closed_form(reveal, discount):
    rate, demand = 0.3, 0.7
    tests = [0.4, 1.0, 1.1, 2.5, 4.0]
    window = 4.0
    found = failure_finding_window(
        exponential(rate),
        tests,
        window,
        demand_rate=demand,
        test_cost=10.0,
        repair_cost=40.0,
        consequence_cost=900.0,
        discount_rate=discount,
        demands_reveal=reveal,
    )
    q, failed, discounted = closed_form(
        rate, tests, window, demand if reveal else 0.0, discount
    )
    np.testing.assert_allclose(found.found_at_tests, q, rtol=1e-9)
    assert found.failed_time == pytest.approx(failed, rel=1e-9)
    assert found.demands_on_failed == pytest.approx(demand * failed, rel=1e-9)
    each = 900.0 + (40.0 if reveal else 0.0)
    cost = 10.0 * 5 + 40.0 * q.sum() + each * demand * failed
    factors = np.exp(-discount * np.array(tests))
    present = (
        10.0 * factors.sum() + 40.0 * factors @ q + each * demand * discounted
    )
    assert abs(found.cost - cost) <= found.precision
    assert abs(found.present_value - present) <= found.precision
    assert found.precision <= 2e-6 * cost
    assert found.present_value == pytest.approx(present, rel=1e-9)
    if discount == 0.0:
        assert found.present_value == found.cost
    # The curve: just before a test at its time, 0 just after.
    at = np.array([0.0, 0.2, 0.4, 0.7, 4.0])
    k = rate + (demand if reveal else 0.0)
    since = np.array([0.0, 0.2, 0.4, 0.3, 1.5])
    curve = failure_finding_window(
        exponential(rate),
        tests,
        window,
        demand_rate=demand,
        times=at,
        demands_reveal=reveal,
    )
    np.testing.assert_allclose(
        curve.unavailability, rate / k * -np.expm1(-k * since), atol=2e-4
    )


def test_even_tests_reproduce_the_long_run():
    """Even tests with no discount, only tests finding a failure: a
    constant rate has ``failure_finding``'s numbers on any whole number
    of intervals, and a Weibull life tends to them over a long window."""
    prices = dict(
        test_cost=5.0,
        repair_cost=30.0,
        demand_rate=0.2,
        consequence_cost=700.0,
    )
    valve = exponential(0.05)
    long_run = failure_finding(valve, 2.0, **prices)
    window = failure_finding_window(
        valve,
        np.arange(1, 26) * 2.0,
        50.0,
        demands_reveal=False,
        **prices,
    )
    assert window.mean_unavailability == pytest.approx(
        long_run.mean_unavailability, rel=1e-9
    )
    assert window.cost / 50.0 == pytest.approx(long_run.cost_rate, rel=1e-9)
    assert window.failures_found / 50.0 == pytest.approx(
        long_run.failures_found_rate, rel=1e-9
    )
    # A Weibull life (mean 8.86) from new, over 400 mean lives: the start
    # is worth about a cycle in 400, so within 1% of the long run.
    wearing = NonRepairable(surv.Weibull.from_params([10.0, 2.0]))
    long_run = failure_finding(wearing, 2.5, **prices)
    span = 3500.0
    window = failure_finding_window(
        wearing,
        np.arange(1, 1401) * 2.5,
        span,
        demands_reveal=False,
        **prices,
    )
    assert window.mean_unavailability == pytest.approx(
        long_run.mean_unavailability, rel=0.01
    )
    assert window.cost / span == pytest.approx(long_run.cost_rate, rel=0.01)
    assert window.failures_found / span == pytest.approx(
        long_run.failures_found_rate, rel=0.01
    )


def _simulated(tests, window, age, demand, reveal, discount, rng, runs):
    """The plan itself, one device at a time: a Weibull(10, 2.2) life from
    ``age``; a failure stays until the next test or, where demands
    reveal, the next demand (an exponential wait: demands are a Poisson
    process), and is then repaired as new. Each run's tests that found a
    failure, demands that met the device failed, and both discounted."""
    scale, shape = 10.0, 2.2
    found = np.zeros(runs)
    met = np.zeros(runs)
    found_pv = np.zeros(runs)
    met_pv = np.zeros(runs)
    tests = np.asarray(tests, float)
    for run in range(runs):
        t, unit_age = 0.0, age
        while True:
            hazard = (unit_age / scale) ** shape + rng.exponential()
            fails = t + scale * hazard ** (1.0 / shape) - unit_age
            if fails >= window:
                break
            later = tests[tests >= fails]
            test = later[0] if later.size else math.inf
            if reveal:
                shown = fails + rng.exponential(1.0 / demand)
                if min(shown, test) > window:
                    break
                if shown < test:
                    met[run] += 1
                    met_pv[run] += math.exp(-discount * shown)
                    t = shown
                else:
                    found[run] += 1
                    found_pv[run] += math.exp(-discount * test)
                    t = test
            else:
                end = min(test, window)
                # Undiscounted only: the demands in the failed time.
                met[run] += rng.poisson(demand * (end - fails))
                if test > window:
                    break
                found[run] += 1
                found_pv[run] += math.exp(-discount * test)
                t = test
            unit_age = 0.0
    return found, met, found_pv, met_pv


@pytest.mark.parametrize("tests", [(3.0, 5.0, 6.0, 9.0, 11.0), ()])
@pytest.mark.parametrize("reveal", [True, False])
def test_weibull_against_a_simulation(tests, reveal):
    """A wearing device, 4 old, tested at uneven times (and never):
    within four standard errors of a seeded simulation of the plan."""
    device = NonRepairable(surv.Weibull.from_params([10.0, 2.2]))
    window, age, demand = 12.0, 4.0, 0.6
    discount = 0.05 if reveal else 0.0
    worked = failure_finding_window(
        device,
        tests,
        window,
        age=age,
        test_cost=0.0,
        repair_cost=1.0,
        demand_rate=demand,
        consequence_cost=1.0,
        discount_rate=discount,
        demands_reveal=reveal,
    )
    rng = np.random.default_rng(20261120)
    runs = 40_000
    found, met, found_pv, met_pv = _simulated(
        tests, window, age, demand, reveal, discount, rng, runs
    )
    repairs = found_pv + (met_pv if reveal else 0.0)
    for value, draws in (
        (worked.failures_found, found),
        (worked.demands_on_failed, met),
    ) + (
        ((worked.present_value, repairs + met_pv),)
        if reveal
        else ((worked.cost, found + met),)
    ):
        se = draws.std(ddof=1) / math.sqrt(runs)
        assert abs(draws.mean() - value) < 4.0 * se + worked.precision


def test_never_tested_without_reveal_is_the_life_itself():
    device = NonRepairable(surv.Weibull.from_params([10.0, 0.8]))
    at = np.linspace(0.0, 15.0, 7)
    found = failure_finding_window(
        device, [], 15.0, age=2.0, demands_reveal=False, times=at
    )
    life = device.reliability
    np.testing.assert_allclose(
        found.unavailability,
        1.0 - life.sf(2.0 + at) / life.sf(2.0),
        atol=1e-12,
    )
    assert found.demands_on_failed is None and found.cost == 0.0


def test_a_test_at_the_start_finds_nothing():
    """The device is working at the start, so a test then only costs."""
    device = NonRepairable(surv.Weibull.from_params([10.0, 1.5]))
    prices = dict(test_cost=7.0, demand_rate=0.3, consequence_cost=50.0)
    with_first = failure_finding_window(device, [0.0, 4.0], 9.0, **prices)
    without = failure_finding_window(device, [4.0], 9.0, **prices)
    assert with_first.found_at_tests[0] == 0.0
    assert with_first.found_at_tests[1] == without.found_at_tests[0]
    assert with_first.cost == pytest.approx(without.cost + 7.0, rel=1e-12)


def test_the_unit_of_time_does_not_matter():
    """Times a thousand, rates a thousandth: the same chances and costs."""
    prices = dict(test_cost=3.0, repair_cost=20.0, consequence_cost=400.0)
    tests = np.array([2.0, 5.0, 6.0])
    one = failure_finding_window(
        NonRepairable(surv.Weibull.from_params([10.0, 1.7])),
        tests,
        9.0,
        age=1.0,
        demand_rate=0.4,
        discount_rate=0.06,
        **prices,
    )
    other = failure_finding_window(
        NonRepairable(surv.Weibull.from_params([10.0e3, 1.7])),
        tests * 1e3,
        9.0e3,
        age=1.0e3,
        demand_rate=0.4e-3,
        discount_rate=0.06e-3,
        **prices,
    )
    np.testing.assert_allclose(
        other.found_at_tests, one.found_at_tests, rtol=1e-9
    )
    assert other.present_value == pytest.approx(one.present_value, rel=1e-9)
    assert other.mean_unavailability == pytest.approx(
        one.mean_unavailability, rel=1e-9
    )


def test_refusals():
    valve = exponential(0.1)
    with pytest.raises(ValueError, match="increasing"):
        failure_finding_window(valve, [2.0, 1.0], 3.0)
    with pytest.raises(ValueError, match="leave out the tests after it"):
        failure_finding_window(valve, [1.0, 4.0], 3.0)
    with pytest.raises(ValueError, match="window must be a number"):
        failure_finding_window(valve, [1.0], 0.0)
    with pytest.raises(ValueError, match="needs the demand_rate"):
        failure_finding_window(valve, [1.0], 3.0, consequence_cost=5.0)
    with pytest.raises(ValueError, match="demands_reveal"):
        failure_finding_window(valve, [1.0], 3.0, demands_reveal="yes")
    with pytest.raises(ValueError, match="tolerance"):
        failure_finding_window(valve, [1.0], 3.0, tolerance=0.0)
    with pytest.raises(ValueError, match="times must be from 0"):
        failure_finding_window(valve, [1.0], 3.0, times=[4.0])
    with pytest.raises(NotImplementedError, match="use failure_finding"):
        failure_finding_window(valve, np.arange(1, 2002) * 1e-3, 3.0)
    bounded = NonRepairable(surv.Uniform.from_params([0.0, 5.0]))
    with pytest.raises(ValueError, match="age of a working device"):
        failure_finding_window(bounded, [1.0], 3.0, age=6.0)
