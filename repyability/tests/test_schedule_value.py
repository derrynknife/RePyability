"""What a scheduled task is worth: ``task_value`` with ``Replaces``.

The worked example is UC-50 of ``repyability/schedule/README.md``; the
rest check it against ``NonRepairable.renewal_forecast`` (block
replacement in model time), a constant rate's closed form, and a
simulation of both plans written out plainly here.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable
from repyability.schedule import (
    DutyPlan,
    Recurrence,
    Replaces,
    Shifts,
    Task,
    TaskValue,
    Usage,
    UsageForecast,
    failures_by_period,
    task_value,
)

HOURS_A_YEAR = 24.0 * 365.25


def ring(**costs):
    """UC-38's wear ring: Weibull, scale 9,000 h, shape 2.5."""
    costs.setdefault("unplanned_cost", 4800.0)
    return NonRepairable(surv.Weibull.from_params([9000.0, 2.5]), **costs)


def service(**options):
    """A new ring at every 6-monthly service, from 1 December 2026."""
    options.setdefault("cost", 600.0)
    return Task(
        "new wear ring",
        every=Recurrence.every(6, "months", "2026-12-01"),
        **options,
    )


@pytest.fixture(scope="module")
def pump():
    """UC-32's pump: Mon-Fri 06:00-22:00 from Monday 16 Nov 2026."""
    return DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2026-11-16", "2027-12-01", readings={"hours": 12_340.0}
    )


def test_uc50_wear_ring_six_monthly_service(pump):
    plain = task_value(service(), Replaces(ring()), pump, age=5200.0)
    assert isinstance(plain, TaskValue)
    assert [str(d) for d in plain.dates] == ["2026-12-01", "2027-06-01"]
    np.testing.assert_allclose(plain.model_times, [176.0, 2256.0])
    # By hand: 176 h from 5,200 h, then 2,080 h and 2,096 h from new.
    by_hand = (
        ring().failures(176.0, age=5200.0).expected_failures
        + ring().failures(2080.0).expected_failures
        + ring().failures(2096.0).expected_failures
    )
    assert plain.total["failures_with"] == pytest.approx(by_hand, rel=1e-12)
    assert plain.total["failures_with"] == pytest.approx(0.0731, abs=5e-5)
    assert plain.total["with_task"] == pytest.approx(1550.82, abs=0.005)
    assert plain.total["without_task"] == pytest.approx(2987.56, abs=0.005)
    assert plain.total["value"] == pytest.approx(1436.74, abs=0.01)
    assert plain.total["failures_avoided"] == pytest.approx(0.549, abs=5e-4)
    assert plain.total["count"] == 2
    np.testing.assert_allclose(
        plain.failures_with[:8],
        [0.0218, 0.0003, 0.0014, 0.0026, 0.0050, 0.0070, 0.0090, 0.0003],
        atol=5e-5,
    )
    np.testing.assert_array_equal(plain.task_cost[[1, 7]], [600.0, 600.0])
    assert plain.precision == 0.0 and plain.eac is None
    # The same numbers as UC-38's run to failure.
    rtf = failures_by_period(ring(), pump, age=5200.0)
    np.testing.assert_allclose(
        plain.failures_without, rtf.expected_failures, rtol=1e-6, atol=1e-9
    )

    worth = task_value(
        service(), Replaces(ring()), pump, age=5200.0, discount=0.07
    )
    assert worth.total["task_cost"] == pytest.approx(1176.83, abs=0.005)
    # Inside the hand brackets, month by month.
    assert 1516.22 <= worth.total["with_task"] <= 1517.84
    assert 2874.75 <= worth.total["without_task"] <= 2890.71
    assert 1356.92 <= worth.total["value"] <= 1374.49
    assert worth.precision < 1e-3 * worth.total["without_task"]
    assert set(worth.eac) == {"with_task", "without_task", "value"}
    assert "Without the task" in worth.without_words
    json.dumps(worth.to_dict())
    assert "TaskValue('new wear ring'" in repr(worth)


def test_even_dates_match_renewal_forecast():
    """Every 30 days in calendar time is every 720 h: block replacement in
    model time, which renewal_forecast works out from the same state."""
    part = NonRepairable(
        surv.Weibull.from_params([3000.0, 2.5]),
        planned_cost=500.0,
        unplanned_cost=4000.0,
    )
    task = Task("replace", every=Recurrence.every(30, "days", "2027-01-11"))
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    found = task_value(task, Replaces(part), usage, age=1000.0, discount=0)
    assert found.model_times[0] == 240.0
    np.testing.assert_allclose(np.diff(found.model_times), 720.0)
    plan = part.renewal_forecast(
        8760.0, 720.0, resets=False, age=1000.0, since_renewal=480.0
    )
    assert found.total["count"] == plan.expected_renewals == 12
    assert found.total["with_task"] == pytest.approx(
        plan.expected_cost, rel=1e-6
    )
    assert found.total["failures_with"] == pytest.approx(
        plan.expected_failures, rel=1e-6
    )
    assert found.total["without_task"] == pytest.approx(
        4000.0 * plan.expected_failures_without_renewals, rel=1e-9
    )
    assert found.precision == 0.0
    assert any("planned_cost, 500" in note for note in found.notes)


def test_constant_rate_task_only_costs(pump):
    """Replacing a part whose rate is constant changes nothing but the
    bill: the value is minus the task's own present value."""
    part = NonRepairable(
        surv.Exponential.from_params([1 / 5000.0]), unplanned_cost=4800.0
    )
    worth = task_value(
        service(), Replaces(part), pump, age=3000.0, discount=0.07
    )
    # Both sides are the closed form, the rate times the running hours,
    # to NonRepairable.failures' own precision. Its count is the first
    # unit's failures, exact, plus the later units', exact at the points
    # of a grid (to 1e-16) and linear between them, of step h at most a
    # thousandth of the life's median (_curves._POINT_STEPS). The later
    # units' count, rate t - (1 - exp(-rate t)), is convex with second
    # derivative at most rate**2, so a count from 0 is over by at most
    # (rate h)**2 / 8 = (ln 2 / 1000)**2 / 8 = 6.0e-8, whatever the rate
    # (5.97e-8 is found), an absolute error. A period's count is a
    # difference of two such counts on each stretch it overlaps: within
    # `bound` without the task, and 2 * bound with it (a period holds
    # at most one task date, so overlaps two stretches).
    bound = (math.log(2.0) / 1000.0) ** 2 / 8.0
    closed = np.diff(pump.model_time(list(worth.ends)), prepend=0.0) / 5000
    np.testing.assert_allclose(worth.failures_without, closed, atol=bound)
    np.testing.assert_allclose(worth.failures_with, closed, atol=2 * bound)
    # Each period's failure cost on the two sides then differs by at most
    # 4800 * 3 * bound, discounted (a factor at most 1).
    np.testing.assert_allclose(
        worth.value, -worth.task_cost, atol=4800.0 * 3.0 * bound
    )
    assert worth.total["value"] < 0.0
    by_hand = 600.0 * (1.07 ** (-15 / 365.25) + 1.07 ** (-197 / 365.25))
    assert worth.total["task_cost"] == pytest.approx(by_hand, rel=1e-12)


def _simulated(every, first, horizon, age, rng, runs):
    """Each run's failures and their present value, at a continuous rate
    of ln(1.07) a year with time in hours, for a Weibull(2000, 2.5) part
    ``age`` old, renewed at ``first + k every`` (none if ``every`` is
    None) and at each failure."""
    scale, shape = 2000.0, 2.5
    rate = math.log(1.07) / HOURS_A_YEAR
    cuts = [] if every is None else list(np.arange(first, horizon, every))
    bounds = [0.0] + cuts + [horizon]
    count = np.zeros(runs)
    present = np.zeros(runs)
    for k in range(len(bounds) - 1):
        begin, end = bounds[k], bounds[k + 1]
        t = np.full(runs, begin)
        unit_age = np.full(runs, age if k == 0 else 0.0)
        alive = np.ones(runs, bool)
        while alive.any():
            # A life left at the unit's age, by inverting R(a + x) / R(a).
            u = rng.random(alive.sum())
            a = unit_age[alive]
            left = scale * ((a / scale) ** shape - np.log(u)) ** (1 / shape)
            at = t[alive] + left - a
            failed = at < end
            index = np.flatnonzero(alive)
            hit = index[failed]
            count[hit] += 1
            present[hit] += np.exp(-rate * at[failed])
            t[hit] = at[failed]
            unit_age[hit] = 0.0
            alive[index[~failed]] = False
    return count, present


def test_wear_out_against_a_simulation():
    """A wear-out part with a dear failure: the task pays, and a seeded
    simulation of both plans (not using the formulas) agrees."""
    part = NonRepairable(
        surv.Weibull.from_params([2000.0, 2.5]),
        planned_cost=300.0,
        unplanned_cost=5000.0,
    )
    task = Task("replace", every=Recurrence.every(30, "days", "2027-01-11"))
    usage = UsageForecast.calendar("2027-01-01", "2027-07-01")
    horizon = 181 * 24.0
    worth = task_value(task, Replaces(part), usage, age=1500.0, discount=0.07)
    assert worth.total["value"] > 0.0
    rng = np.random.default_rng(20261120)
    runs = 200_000
    with_n, with_pv = _simulated(720.0, 240.0, horizon, 1500.0, rng, runs)
    without_n, without_pv = _simulated(None, 0.0, horizon, 1500.0, rng, runs)
    planned = sum(
        300.0 * math.exp(-math.log(1.07) / HOURS_A_YEAR * t)
        for t in np.arange(240.0, horizon, 720.0)
    )
    for found, draws in (
        (worth.total["failures_with"], with_n),
        (worth.total["failures_without"], without_n),
        (worth.total["with_task"], 5000.0 * with_pv + planned),
        (worth.total["without_task"], 5000.0 * without_pv),
    ):
        se = draws.std(ddof=1) / math.sqrt(runs)
        assert abs(draws.mean() - found) < 4.0 * se + worth.precision
    difference = 5000.0 * (without_pv - with_pv) - planned
    se = difference.std(ddof=1) / math.sqrt(runs)
    assert abs(difference.mean() - worth.total["value"]) < (
        4.0 * se + worth.precision
    )


def test_periods_add_up(pump):
    months = task_value(
        service(), Replaces(ring()), pump, age=5200.0, discount=0.07
    )
    quarters = task_value(
        service(),
        Replaces(ring()),
        pump,
        age=5200.0,
        discount=0.07,
        period="quarter",
    )
    for found in (months, quarters):
        for name in (
            "with_task",
            "without_task",
            "value",
            "task_cost",
            "cost_with",
            "cost_without",
            "failures_with",
            "failures_without",
            "failures_avoided",
        ):
            assert float(np.sum(getattr(found, name))) == pytest.approx(
                found.total[name], rel=1e-12, abs=1e-12
            )
        np.testing.assert_allclose(
            found.value, found.without_task - found.with_task
        )
    for name in ("cost_with", "cost_without", "task_cost"):
        assert months.total[name] == pytest.approx(
            quarters.total[name], rel=1e-9
        )
    for name in ("with_task", "without_task", "value"):
        assert months.total[name] == pytest.approx(
            quarters.total[name],
            abs=months.precision + quarters.precision,
        )


def test_one_price(pump):
    with pytest.raises(ValueError, match="costs 600 and the part's"):
        task_value(service(), Replaces(ring(planned_cost=500.0)), pump)
    with pytest.raises(ValueError, match="give the task's cost="):
        task_value(service(cost=None), Replaces(ring()), pump)
    agreed = task_value(service(), Replaces(ring(planned_cost=600.0)), pump)
    taken = task_value(
        service(cost=None), Replaces(ring(planned_cost=600.0)), pump
    )
    assert agreed.total["with_task"] == taken.total["with_task"]
    with pytest.raises(ValueError, match="unplanned_cost="):
        task_value(
            service(),
            Replaces(NonRepairable(surv.Weibull.from_params([9000.0, 2.5]))),
            pump,
        )


def test_what_is_not_a_replacement_is_refused(pump):
    with pytest.raises(TypeError, match="Replaces"):
        task_value(service(), ring(), pump)
    with pytest.raises(TypeError, match="NonRepairable"):
        Replaces(surv.Weibull.from_params([9000.0, 2.5]))
    with pytest.raises(TypeError, match="Task"):
        task_value("new wear ring", Replaces(ring()), pump)


def test_a_task_by_use_matches_block_replacement_in_model_time(pump):
    """Every 2,000 running hours, last done at 11,840 h (500 h before the
    forecast's 12,340 h): block replacement every 2,000 h in model time,
    500 h into the interval."""
    task = Task(
        "new wear ring",
        every=Usage(2000.0, "hours"),
        last_done_reading=11_840.0,
        cost=600.0,
    )
    found = task_value(task, Replaces(ring()), pump, age=5200.0, discount=0)
    np.testing.assert_allclose(found.model_times, [1500.0, 3500.0])
    plan = ring(planned_cost=600.0).renewal_forecast(
        4352.0, 2000.0, resets=False, age=5200.0, since_renewal=500.0
    )
    assert found.total["with_task"] == pytest.approx(
        plan.expected_cost, rel=1e-6
    )
    assert found.total["failures_with"] == pytest.approx(
        plan.expected_failures, rel=1e-6
    )
    # And by month, as failures_by_period's policy in model time.
    policy = failures_by_period(
        ring(planned_cost=600.0),
        pump,
        age=5200.0,
        every=2000.0,
        resets=False,
        since_renewal=500.0,
        discount=0.07,
    )
    worth = task_value(task, Replaces(ring()), pump, age=5200.0, discount=0.07)
    np.testing.assert_allclose(
        worth.cost_with, policy.expected_cost, rtol=1e-5, atol=1e-6
    )
    assert worth.total["with_task"] == pytest.approx(
        policy.npv, abs=worth.precision + policy.npv_precision
    )
