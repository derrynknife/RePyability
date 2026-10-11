"""Components replaced when they fail, and their failures over a period of
dates (#319, UC-5 of ``repyability/maintenance/README.md``)."""

import json
import math
from datetime import date, datetime

import numpy as np
import pytest
import surpyval as surv

from repyability import Component, FailureForecast

LIFE = surv.Weibull.from_params([365.0, 2.0])
BEARING = Component("bearing", LIFE, installed="2026-03-01", cost=400.0)


def simulated_failures(scale, shape, age, days, n=400_000, seed=0):
    """The failures of a Weibull unit of ``age`` and its replacements in
    ``days``, simulated plainly: the mean count and its standard error."""
    rng = np.random.default_rng(seed)
    held = (age / scale) ** shape
    left = scale * (held - np.log(rng.random(n))) ** (1 / shape) - age
    count, time = np.zeros(n), left
    running = time < days
    while running.any():
        count[running] += 1
        fresh = scale * (-np.log(rng.random(running.sum()))) ** (1 / shape)
        time[running] += fresh
        running = time < days
    return count.mean(), count.std() / math.sqrt(n)


def test_the_bearing_through_the_rest_of_2026():
    # UC-5: installed 1 March 2026, 224 days old on 11 October; 82 days to
    # 31 December, both included.
    forecast = BEARING.failures("2026-10-11", "2026-12-31")
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


def test_an_exponential_life_fails_at_its_rate_whatever_its_age():
    rate = 1 / 200.0
    unit = Component("u", surv.Exponential.from_params([rate]), "2026-01-01")
    for start in ("2026-01-01", "2026-06-30"):
        forecast = unit.failures(start, "2026-12-31")
        days = (date(2026, 12, 31) - date.fromisoformat(start)).days + 1
        assert forecast.probability == pytest.approx(
            -math.expm1(-rate * days), rel=1e-10
        )
        assert forecast.expected_failures == pytest.approx(
            rate * days, rel=1e-6
        )


@pytest.mark.parametrize("age, days", [(0, 365), (500, 120), (100, 700)])
def test_the_failures_are_the_renewal_count_simulated(age, days):
    installed = date(2026, 1, 1)
    start = date.fromordinal(installed.toordinal() + age)
    end = date.fromordinal(start.toordinal() + days - 1)
    unit = Component("u", surv.Weibull.from_params([300.0, 3.0]), installed)
    mean, error = simulated_failures(300.0, 3.0, age, days, seed=age)
    assert abs(unit.failures(start, end).expected_failures - mean) < 4 * error


def test_a_short_period_has_about_one_failure_at_most():
    # Over a day, a new unit failing as well is negligible: the expected
    # failures are the chance of the first.
    forecast = BEARING.failures("2026-10-11", "2026-10-11")
    assert forecast.expected_failures == pytest.approx(
        forecast.probability, rel=1e-4
    )


def test_a_period_from_the_day_it_was_installed_starts_new():
    forecast = BEARING.failures("2026-03-01", "2026-03-31")
    assert forecast.probability == pytest.approx(
        -math.expm1(-((31 / 365) ** 2)), rel=1e-12
    )


def test_no_cost_given_is_none():
    unit = Component("bearing", LIFE, "2026-03-01")
    assert unit.cost is None
    assert unit.failures("2026-10-11", "2026-12-31").expected_cost is None


def test_the_forecast_goes_through_json():
    forecast = BEARING.failures("2026-10-11", "2026-12-31")
    plain = json.loads(json.dumps(forecast.to_dict()))
    assert plain["component"] == "bearing"
    assert plain["start"] == "2026-10-11"
    assert plain["end"] == "2026-12-31"
    assert plain["expected_cost"] == pytest.approx(forecast.expected_cost)


def test_a_period_is_after_the_installation_and_in_order():
    with pytest.raises(ValueError, match="before the current unit"):
        BEARING.failures("2026-02-01", "2026-12-31")
    with pytest.raises(ValueError, match="before it starts"):
        BEARING.failures("2026-12-31", "2026-10-11")
    with pytest.raises(TypeError, match="start"):
        BEARING.failures(datetime(2026, 10, 11, 9), "2026-12-31")


def test_a_unit_that_cannot_be_working_at_the_start_is_refused():
    fixed = Component(
        "u", surv.ExactEventTime.from_params([10.0]), "2026-01-01"
    )
    with pytest.raises(ValueError, match="no chance"):
        fixed.failures("2026-02-01", "2026-02-28")


@pytest.mark.parametrize(
    "life, error",
    [(surv.Weibull, TypeError), (0.5, TypeError), (None, TypeError)],
)
def test_a_life_is_a_model(life, error):
    with pytest.raises(error, match="life"):
        Component("u", life, "2026-01-01")


@pytest.mark.parametrize("cost", [-1.0, float("inf"), True, "400"])
def test_a_cost_must_be_a_number_at_least_0(cost):
    with pytest.raises(ValueError, match="cost"):
        Component("u", LIFE, "2026-01-01", cost=cost)


def test_the_name_is_text():
    with pytest.raises(TypeError, match="name"):
        Component(1, LIFE, "2026-01-01")
