"""The failures of a part replaced when it fails, over a window from its
current unit's age (``NonRepairable.failures``; #319, UC-5 of
``repyability/maintenance/README.md``), and its costs given to the
constructor."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import FailureForecast, NonRepairable

LIFE = surv.Weibull.from_params([365.0, 2.0])
BEARING = NonRepairable(LIFE, unplanned_cost=400.0)


def simulated_failures(scale, shape, age, days, n=400_000, seed=0, fix=0.0):
    """The failures of a Weibull unit of ``age`` and its replacements in
    ``days``, each replacement taking ``fix``, simulated plainly: the mean
    count and its standard error."""
    rng = np.random.default_rng(seed)
    held = (age / scale) ** shape
    left = scale * (held - np.log(rng.random(n))) ** (1 / shape) - age
    count, time = np.zeros(n), left
    running = time < days
    while running.any():
        count[running] += 1
        fresh = scale * (-np.log(rng.random(running.sum()))) ** (1 / shape)
        time[running] += fix + fresh
        running = time < days
    return count.mean(), count.std() / math.sqrt(n)


def test_the_bearing_over_82_days_from_224():
    # UC-5: installed 1 March 2026, 224 days old on 11 October; 82 days to
    # the end of 31 December.
    forecast = BEARING.failures(82.0, age=224.0)
    assert isinstance(forecast, FailureForecast)
    held, reached = (224 / 365) ** 2, (306 / 365) ** 2
    assert forecast.probability == pytest.approx(
        -math.expm1(held - reached), rel=1e-12
    )
    mean, error = simulated_failures(365.0, 2.0, 224.0, 82.0)
    assert abs(forecast.expected_failures - mean) < 4 * error
    assert forecast.expected_failures == pytest.approx(0.2830, abs=5e-4)
    assert forecast.expected_cost == pytest.approx(
        400.0 * forecast.expected_failures, rel=1e-12
    )
    assert forecast.window == 82.0 and forecast.age == 224.0


def test_an_exponential_life_fails_at_its_rate_whatever_its_age():
    rate = 1 / 200.0
    unit = NonRepairable(surv.Exponential.from_params([rate]))
    for age in (0.0, 180.0):
        forecast = unit.failures(365.0, age=age)
        assert forecast.probability == pytest.approx(
            -math.expm1(-rate * 365.0), rel=1e-10
        )
        assert forecast.expected_failures == pytest.approx(
            rate * 365.0, rel=1e-6
        )


@pytest.mark.parametrize("age, days", [(0, 365), (500, 120), (100, 700)])
def test_the_failures_are_the_renewal_count_simulated(age, days):
    unit = NonRepairable(surv.Weibull.from_params([300.0, 3.0]))
    mean, error = simulated_failures(300.0, 3.0, age, days, seed=age)
    forecast = unit.failures(float(days), age=float(age))
    assert abs(forecast.expected_failures - mean) < 4 * error


def test_a_replacement_that_takes_time_delays_the_next_failures():
    # Ten days to replace each failed unit: fewer failures fit in the
    # window, as simulated, and the chance of the first is unchanged.
    fix = surv.ExactEventTime.from_params([10.0])
    slow = NonRepairable(surv.Weibull.from_params([100.0, 1.5]), fix)
    quick = NonRepairable(surv.Weibull.from_params([100.0, 1.5]))
    mean, error = simulated_failures(100.0, 1.5, 50.0, 400.0, seed=3, fix=10.0)
    slow_forecast = slow.failures(400.0, age=50.0)
    assert abs(slow_forecast.expected_failures - mean) < 4 * error
    assert (
        slow_forecast.expected_failures
        < quick.failures(400.0, 50.0).expected_failures
    )
    assert slow_forecast.probability == quick.failures(400.0, 50.0).probability


def test_windows_at_once():
    windows = np.array([0.0, 30.0, 82.0, 365.0])
    forecast = BEARING.failures(windows, age=224.0)
    assert forecast.expected_failures.shape == (4,)
    assert forecast.expected_failures[0] == 0.0
    assert np.all(np.diff(forecast.expected_failures) > 0)
    for window, failures, chance in zip(
        windows, forecast.expected_failures, forecast.probability
    ):
        one = BEARING.failures(window, age=224.0)
        # The renewal count is worked out to about 1e-7 (expected_events),
        # on a grid that depends on the windows asked for together.
        assert one.expected_failures == pytest.approx(failures, rel=1e-6)
        assert one.probability == pytest.approx(chance, rel=1e-12, abs=1e-15)


def test_a_short_window_has_about_one_failure_at_most():
    # Over a day, a new unit failing as well is negligible: the expected
    # failures are the chance of the first.
    forecast = BEARING.failures(1.0, age=224.0)
    assert forecast.expected_failures == pytest.approx(
        forecast.probability, rel=1e-4
    )


def test_no_unplanned_cost_is_none():
    unit = NonRepairable(LIFE)
    assert unit.failures(82.0, age=224.0).expected_cost is None


def test_the_forecast_goes_through_json():
    forecast = BEARING.failures(82.0, age=224.0)
    plain = json.loads(json.dumps(forecast.to_dict()))
    assert plain["window"] == 82.0
    assert plain["age"] == 224.0
    assert plain["expected_cost"] == pytest.approx(forecast.expected_cost)


def test_a_unit_that_cannot_be_working_at_its_age_is_refused():
    fixed = NonRepairable(surv.ExactEventTime.from_params([10.0]))
    with pytest.raises(ValueError, match="no chance"):
        fixed.failures(30.0, age=20.0)


@pytest.mark.parametrize("t", [-1.0, float("inf"), "82", [1.0, -2.0]])
def test_a_window_is_at_least_0(t):
    with pytest.raises((ValueError, TypeError), match="t must"):
        BEARING.failures(t, age=0.0)


@pytest.mark.parametrize("age", [-1.0, float("inf"), True, "224"])
def test_an_age_is_at_least_0(age):
    with pytest.raises(ValueError, match="age"):
        BEARING.failures(82.0, age=age)


def test_the_costs_given_to_the_constructor():
    both = NonRepairable(LIFE, planned_cost=150, unplanned_cost=400)
    assert (both.cp, both.cu) == (150.0, 400.0)
    set_after = NonRepairable(LIFE)
    set_after.set_costs_planned_and_unplanned(150, 400)
    assert both.find_optimal_replacement() == pytest.approx(
        set_after.find_optimal_replacement(), rel=1e-12
    )
    # One alone: failures need only the unplanned cost; the age
    # replacement needs both.
    assert NonRepairable(LIFE, unplanned_cost=400).cu == 400.0
    with pytest.raises(ValueError, match="costs not set"):
        NonRepairable(LIFE, unplanned_cost=400).find_optimal_replacement()
    with pytest.raises(ValueError, match="costs not set"):
        NonRepairable(LIFE, planned_cost=150).find_optimal_replacement()


@pytest.mark.parametrize(
    "costs, match",
    [
        ({"planned_cost": -1.0}, "planned_cost"),
        ({"unplanned_cost": float("inf")}, "unplanned_cost"),
        ({"unplanned_cost": True}, "unplanned_cost"),
        ({"planned_cost": "150"}, "planned_cost"),
        ({"planned_cost": 400, "unplanned_cost": 150}, "less than"),
    ],
)
def test_the_costs_are_checked(costs, match):
    with pytest.raises(ValueError, match=match):
        NonRepairable(LIFE, **costs)


def test_the_costs_are_given_by_name():
    with pytest.raises(TypeError):
        NonRepairable(LIFE, None, 150, 400)
