"""The value of a service on dates (UC-67 of the schedule README):
``task_value(task, Services(plan, effect=), usage)``."""

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable
from repyability.maintenance import (
    AgePolicy,
    ImperfectRepair,
    Prices,
    Restoration,
    UnitPlan,
)
from repyability.schedule import (
    Condition,
    DutyPlan,
    Recurrence,
    Replaces,
    Services,
    Shifts,
    Task,
    UsageForecast,
    task_value,
)

LIFE = surv.Weibull.from_params([8760.0, 3.0])
PLAN = UnitPlan(LIFE, "instant", prices=Prices(repair=12000.0))
SERVICE = Task(
    "gearbox service",
    every=Recurrence.every(3, "months", "2027-04-01"),
    cost=800.0,
)
USAGE = UsageForecast.calendar("2027-01-01", "2028-01-01")
#: The z of the default 90% two-sided interval.
Z90 = 1.6448536269514722


def value(effect=None, plan=PLAN, **kwargs):
    kwargs.setdefault("age", 4000.0)
    kwargs.setdefault("discount", 0.07)
    kwargs.setdefault("period", "quarter")
    return task_value(SERVICE, Services(plan, effect), USAGE, **kwargs)


def standard_error(worth) -> float:
    total = worth.total
    return (total["value_upper"] - total["value_lower"]) / (2.0 * Z90)


def test_uc67_no_effect_is_the_service_cost():
    worth = value()
    assert worth.method == "exact"
    np.testing.assert_allclose(
        worth.value, [0.0, -786.77, -773.62, -760.55], atol=0.005
    )
    assert worth.total["value"] == pytest.approx(-2320.95, abs=0.005)
    assert worth.total["without_task"] == 0.0
    assert worth.total["failures_avoided"] == 0.0
    assert np.isnan(worth.total["failures_with"])
    assert worth.value_lower is None and worth.n_simulations is None
    assert any("taken as none" in note for note in worth.notes)
    plain = value(discount=None)
    assert plain.total["value"] == -2400.0
    # The plan is not needed for no effect.
    assert task_value(
        SERVICE, Services(), USAGE, discount=0.07, period="quarter"
    ).total["value"] == pytest.approx(-2320.95, abs=0.005)


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
def test_a_factor_of_one_is_no_effect_exactly(model):
    worth = value(Restoration(q=1.0, model=model))
    assert worth.method == "exact"
    np.testing.assert_array_equal(worth.value, value().value)


def test_as_new_matches_block_replacement_on_the_dates():
    part = NonRepairable(LIFE, unplanned_cost=12000.0)
    exact = task_value(
        SERVICE,
        Replaces(part),
        USAGE,
        age=4000.0,
        discount=0.07,
        period="quarter",
    )
    # Replaces counts the task as the planned replacement at 800 too.
    worth = value(Restoration(q=0.0), seed=67)
    assert worth.method == "simulated"
    se = standard_error(worth)
    assert abs(worth.total["value"] - exact.total["value"]) < 4 * se
    assert (
        abs(worth.total["failures_with"] - exact.total["failures_with"]) < 0.05
    )
    # Common random numbers: before the first service both sides match.
    assert worth.value[0] == 0.0


def test_as_new_with_minimal_repairs_against_the_overhaul_closed_form():
    minimal = UnitPlan(
        LIFE,
        "instant",
        prices=Prices(repair=12000.0),
        imperfect_repair=ImperfectRepair(model="kijima1", q=1.0),
    )
    worth = value(Restoration(q=0.0), plan=minimal, age=0.0, discount=None)
    hours = np.array([0.0, 2160.0, 4344.0, 6552.0, 8760.0])

    def H(t):
        return (np.asarray(t) / 8760.0) ** 3

    with_service = float(np.sum(H(np.diff(hours))))
    expected = 12000.0 * (H(8760.0) - with_service) - 2400.0
    assert abs(worth.total["value"] - expected) < 4 * standard_error(worth)


def test_seeded_runs_repeat_and_narrow_with_more_runs():
    effect = Restoration(q=0.6)
    one = value(effect, seed=5, mc_samples=1000)
    again = value(effect, seed=5, mc_samples=1000)
    np.testing.assert_array_equal(one.value, again.value)
    assert one.seed == 5 and one.n_simulations == 1000
    more = value(effect, seed=5, mc_samples=4000)
    assert standard_error(more) < standard_error(one)


def test_the_value_falls_as_the_factor_rises():
    worths = [value(Restoration(q=q), seed=11) for q in (0.0, 0.6, 1.2)]
    for better, worse in zip(worths, worths[1:]):
        assert better.total["value_lower"] > worse.total["value_upper"]
    assert worths[-1].total["value"] < -2320.95  # a service that does harm


def test_a_constant_rate_gains_nothing_at_any_factor():
    plan = UnitPlan(
        surv.Exponential.from_params([1 / 8760.0]),
        "instant",
        prices=Prices(repair=12000.0),
    )
    for q in (0.0, 0.5):
        worth = value(Restoration(q=q), plan=plan, seed=3)
        assert abs(worth.total["value"] + 2320.95) < 4 * standard_error(worth)


@pytest.fixture(scope="module")
def serviced_life():
    """A proportional hazards Weibull fitted with a load covariate:
    lives about 6,000 h at 1 and a quarter of that at 2."""
    rng = np.random.default_rng(7)
    load = np.repeat([1.0, 2.0], 40)
    x = 6000.0 * rng.weibull(1.5, load.size) / load**2
    return surv.WeibullPH.fit(x, Z=load[:, None])


def test_a_fitted_condition_follows_its_covariate(serviced_life):
    helps = Condition(serviced_life, fresh=[1.0], stale=[2.0], lasts=2000.0)
    harms = Condition(serviced_life, fresh=[2.0], stale=[1.0], lasts=2000.0)
    good = value(helps, age=0.0, seed=2)
    bad = value(harms, age=0.0, seed=2)
    assert good.method == bad.method == "simulated"
    assert good.total["value_lower"] > 0.0
    assert bad.total["value_upper"] < -2320.95


def test_a_condition_schedule_merges_overlapping_services(serviced_life):
    effect = Condition(serviced_life, fresh=[1.0], stale=[2.0], lasts=10.0)
    schedule = effect.schedule(np.array([0.0, 5.0, 30.0]))
    np.testing.assert_array_equal(schedule.edges[:-1], [0.0, 15.0, 30.0, 40.0])
    np.testing.assert_array_equal(schedule.Z.ravel(), [1.0, 2.0, 1.0, 2.0])


def test_refusals(serviced_life):
    with pytest.raises(TypeError, match="effect must say what a service"):
        Services(PLAN, effect=0.5)
    with pytest.raises(ValueError, match="needs the unit's plan"):
        Services(effect=Restoration(q=0.5))
    with pytest.raises(TypeError, match="UnitPlan"):
        Services({"reliability": LIFE}, Restoration(q=0.5))
    held = UnitPlan(LIFE, "instant", tasks=(AgePolicy(interval=4000.0),))
    with pytest.raises(ValueError, match="without it"):
        Services(held, Restoration(q=0.5))
    free = Task("service", every=Recurrence.every(3, "months", "2027-04-01"))
    with pytest.raises(ValueError, match="what one service costs"):
        task_value(free, Services(), USAGE)
    with pytest.raises(ValueError, match="same covariates"):
        Condition(serviced_life, fresh=[1.0], stale=[2.0, 0.0], lasts=1.0)
    with pytest.raises(ValueError, match="lasts"):
        Condition(serviced_life, fresh=[1.0], stale=[2.0], lasts=0.0)
    shifts = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2027-01-01", "2028-01-01"
    )
    # A fitted condition's covariates on a duty plan are not taken (a
    # virtual age there is: test_a_service_on_a_duty_plan_that_changes).
    condition = Condition(serviced_life, fresh=[1.0], stale=[2.0], lasts=1.0)
    with pytest.raises(ValueError, match="covariate schedule"):
        task_value(SERVICE, Services(PLAN, condition), shifts, seed=1)


# -- a service on a duty plan that changes ---------------------------------

SHIFTS = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
    "2027-01-01", "2028-01-01"
)


def test_a_service_on_a_duty_plan_that_changes():
    # As new (factor 0) on weekday shifts, from 4,000 operating hours:
    # the replacement's exact value through the same shifts.
    part = NonRepairable(LIFE, unplanned_cost=12000.0)
    exact = task_value(
        SERVICE,
        Replaces(part),
        SHIFTS,
        age=4000.0,
        discount=0.07,
        period="quarter",
    )
    worth = task_value(
        SERVICE,
        Services(PLAN, Restoration(q=0.0)),
        SHIFTS,
        age=4000.0,
        discount=0.07,
        period="quarter",
        seed=67,
    )
    assert worth.method == "simulated"
    se = standard_error(worth)
    assert abs(worth.total["value"] - exact.total["value"]) < 4 * se


def test_minimal_repairs_on_shifts_against_the_closed_form():
    # Repaired as bad as old, renewed by each service: the failures in
    # each stretch between services are H of its operating hours, D the
    # shifts' running hours (H(x) = (x / 8,760) ** 3).
    minimal = UnitPlan(
        LIFE,
        "instant",
        prices=Prices(repair=12000.0),
        imperfect_repair=ImperfectRepair(model="kijima1", q=1.0),
    )
    worth = task_value(
        SERVICE,
        Services(minimal, Restoration(q=0.0)),
        SHIFTS,
        discount=None,
        period="quarter",
        seed=5,
    )
    dates = ["2027-01-01", "2027-04-01", "2027-07-01", "2027-10-01"]
    hours = np.append(
        SHIFTS.model_time(dates), SHIFTS.model_time("2028-01-01")
    )

    def H(x):
        return (np.asarray(x) / 8760.0) ** 3

    expected = 12000.0 * (H(hours[-1]) - float(np.sum(H(np.diff(hours)))))
    expected -= 3 * 800.0
    assert abs(worth.total["value"] - expected) < 4 * standard_error(worth)
