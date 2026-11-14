"""Failures, events and shuts on the calendar (``failures_by_period``,
``events_by_period``, ``shutdown_spec``, ``Opportunity.from_recurrence``,
``UsageForecast.life_at_usage``).

The worked examples are the use cases' (UC-38 to UC-42 in
``repyability/schedule/README.md``), each checked by hand in the comments
and against the model-time methods they call.
"""

import json
from datetime import datetime

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, NonRepairable, Opportunity, RepairableRBD
from repyability.schedule import (
    DutyPlan,
    Recurrence,
    Shifts,
    UsageForecast,
    events_by_period,
    failures_by_period,
    shutdown_spec,
)

OVERHAUL = Recurrence.yearly(month=11, weekday="Tuesday", nth=1)


@pytest.fixture
def pump():
    """Pump P-101: Mon-Fri 06:00-22:00, 12,340 h at the start of Monday 16
    Nov 2026, forecast to 1 Dec 2027."""
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    return plan.forecast(
        "2026-11-16", "2027-12-01", readings={"hours": 12_340}
    )


def ring(**costs):
    return NonRepairable(surv.Weibull.from_params([9000.0, 2.5]), **costs)


def sf(t):
    return np.exp(-((t / 9000.0) ** 2.5))


# ------------------------------------------------- UC-38: a part by month


def test_uc38_wear_ring_by_month(pump):
    months = failures_by_period(
        ring(unplanned_cost=4800.0), pump, "month", age=5200.0
    )
    # 13 periods: 16-30 Nov 2026, then Dec 2026 to Nov 2027
    assert len(months.starts) == 13
    assert months.starts[0] == datetime(2026, 11, 16)
    assert months.ends[-1] == datetime(2027, 12, 1)
    # operating hours: each month's weekdays x 16
    assert months.operating.tolist() == [
        176,
        368,
        336,
        320,
        368,
        352,
        336,
        352,
        352,
        352,
        352,
        336,
        352,
    ]
    # by hand: the ring fails by 1 Dec 2027 with chance
    # 1 - R(5200 + 4352) / R(5200) = 1 - exp(-(1.16046 - 0.25375)) = 0.596
    assert months.probability[-1] == pytest.approx(
        1 - sf(9552.0) / sf(5200.0), rel=1e-12
    )
    assert months.total["probability"] == pytest.approx(0.5961, abs=1e-4)
    # expected failures: the current ring's and its replacements', 0.622
    # ($2,988 at $4,800)
    assert months.total["expected_failures"] == pytest.approx(0.6224, abs=1e-4)
    assert months.total["expected_cost"] == pytest.approx(2987.6, abs=0.1)
    # each month equals NonRepairable.failures at the operating-hour ends
    exact = ring(unplanned_cost=4800.0).failures(
        np.cumsum(months.operating), age=5200.0
    )
    by_month = np.diff(exact.expected_failures, prepend=0.0)
    np.testing.assert_allclose(months.expected_failures, by_month, atol=1e-6)
    # the plan's table, to the cent it quotes (to the dollar)
    assert np.round(months.expected_cost).tolist() == [
        105,
        227,
        217,
        214,
        252,
        247,
        239,
        252,
        253,
        252,
        250,
        236,
        244,
    ]
    # the months add up to the whole
    assert months.expected_failures.sum() == pytest.approx(
        months.total["expected_failures"], abs=1e-6
    )
    # the range over the forecast: 0 to 1 failure, $0 to $4,800 (90% of
    # outcomes); no failure 40.4%, one 57.0%, two or more 2.6%
    assert (months.total["cost_low"], months.total["cost_high"]) == (
        0.0,
        4800.0,
    )
    p = months.total["failure_probabilities"]
    assert p[0] == pytest.approx(0.4039, abs=1e-4)
    assert p[1] == pytest.approx(0.5701, abs=1e-4)
    assert sum(p[2:]) == pytest.approx(0.0260, abs=1e-4)
    assert "90% of outcomes, two-sided" in months.range
    json.dumps(months.to_dict())


def test_uc38_against_a_plain_simulation(pump):
    # The ring, renewed at each failure, over 4,352 operating hours from
    # age 5,200, simulated plainly: the first life given the age, then new
    # lives.
    rng = np.random.default_rng(38)
    n, span = 200_000, 4352.0
    h0 = (5200.0 / 9000.0) ** 2.5
    first = 9000.0 * (h0 + rng.exponential(size=n)) ** 0.4 - 5200.0
    count = np.zeros(n)
    t = first.copy()
    while True:
        alive = t < span
        if not alive.any():
            break
        count += alive
        t[alive] += 9000.0 * rng.weibull(2.5, alive.sum())
    months = failures_by_period(
        ring(unplanned_cost=4800.0), pump, "month", age=5200.0
    )
    se = count.std() / np.sqrt(n)
    assert abs(count.mean() - months.total["expected_failures"]) < 4 * se
    assert np.mean(count == 0) == pytest.approx(
        1 - months.total["probability"], abs=0.005
    )


def test_periods_with_no_running_have_nothing(pump):
    days = failures_by_period(
        ring(unplanned_cost=4800.0), pump, "day", age=5200.0
    )
    # Sat 21 and Sun 22 Nov: no hours, no failures, no cost
    assert days.operating[5:7].tolist() == [0.0, 0.0]
    assert days.expected_failures[5:7].tolist() == [0.0, 0.0]
    assert days.expected_cost[5:7].tolist() == [0.0, 0.0]
    assert days.probability_in_period[5:7].tolist() == [0.0, 0.0]
    assert days.probability[5] == days.probability[4]
    assert days.expected_failures.sum() == pytest.approx(0.6224, abs=1e-4)


def test_the_unit_is_read_from_the_model_or_given(pump):
    in_days = NonRepairable(
        surv.Weibull.from_params([9000.0 / 24, 2.5]), unplanned_cost=4800.0
    )
    hours = failures_by_period(ring(unplanned_cost=4800.0), pump, age=5200.0)
    days = failures_by_period(in_days, pump, age=5200.0 / 24, unit="days")
    np.testing.assert_allclose(
        days.expected_failures, hours.expected_failures, atol=1e-6
    )
    np.testing.assert_allclose(days.operating * 24, hours.operating)
    assert days.unit == "days"
    # no label: hours, with a note
    assert hours.unit == "hours"
    assert any("names no unit" in note for note in hours.notes)
    # a label in days is read as days
    labelled = NonRepairable(
        surv.Weibull.from_params([9000.0 / 24, 2.5], unit="d"),
        unplanned_cost=4800.0,
    )
    read = failures_by_period(labelled, pump, age=5200.0 / 24)
    assert read.unit == "days"
    np.testing.assert_allclose(
        read.expected_failures, hours.expected_failures, atol=1e-6
    )
    with pytest.raises(ValueError, match="labelled 'd'"):
        failures_by_period(labelled, pump, age=5.0, unit="hours")
    in_km = NonRepairable(
        surv.Weibull.from_params([9000.0, 2.5], unit="km"),
        unplanned_cost=4800.0,
    )
    with pytest.raises(ValueError, match="not a unit of time"):
        failures_by_period(in_km, pump)


def test_a_policy_by_period(pump):
    part = ring(unplanned_cost=4800.0, planned_cost=1200.0)
    months = failures_by_period(
        part, pump, age=5200.0, every=6000.0, resets=True
    )
    assert months.policy == "age replacement"
    assert months.method == "numerical"
    whole = part.cost_distribution(4352.0, 6000.0, resets=True, age=5200.0)
    assert months.expected_failures.sum() == pytest.approx(
        whole.expected_failures, abs=1e-6
    )
    assert months.expected_planned.sum() == pytest.approx(
        whole.expected_planned, abs=1e-6
    )
    assert months.probability[-1] == pytest.approx(
        1 - whole.failure_probabilities[0], abs=1e-9
    )
    # the planned replacement at age 6,000 h, 800 h on: in the 3rd month
    # (cumulative 880 h by the end of January)
    assert months.expected_planned[2] > 0.5


def test_a_part_with_no_cost_or_a_replacement_taking_time(pump):
    plain = failures_by_period(ring(), pump, age=5200.0)
    assert plain.expected_cost is None and plain.cost_high is None
    assert plain.total["expected_cost"] is None
    assert plain.method == "exact"
    assert plain.total["expected_failures"] == pytest.approx(0.6224, abs=1e-4)
    assert any("unplanned_cost" in note for note in plain.notes)
    with pytest.raises(ValueError, match="unplanned_cost"):
        failures_by_period(ring(), pump, every=6000.0, resets=True)
    slow = ring(
        unplanned_cost=4800.0,
        time_to_replace=surv.ExactEventTime.from_params([8.0]),
    )
    timed = failures_by_period(slow, pump, age=5200.0)
    assert timed.cost_high is None
    assert any("takes" in note for note in timed.notes)
    assert timed.expected_cost.sum() == pytest.approx(
        timed.total["expected_cost"]
    )


# ------------------------------------------- UC-39: a system by month


@pytest.fixture
def constant_pump():
    return RepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {
            "pump": {
                "reliability": surv.Exponential.from_params([1 / 2000.0]),
                "repairability": "instant",
                "repair_cost": 1000.0,
            }
        },
    )


def test_uc39_constant_rate_by_calendar_month(constant_pump):
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    months = events_by_period(constant_pump, usage, unit="hours")
    # a constant rate: hours / 2,000 a month; January 744 h -> 0.372,
    # February 672 h -> 0.336; the year 8,760 h -> 4.38
    hours = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24
    np.testing.assert_allclose(months.model_ends, np.cumsum(hours))
    np.testing.assert_allclose(
        months.events.system_failures, hours / 2000.0, atol=1e-6
    )
    assert months.events.system_failures.sum() == pytest.approx(4.38, 1e-6)
    assert months.cost is None
    json.dumps(months.to_dict())


def test_calendar_months_in_a_zone_are_real_hours(constant_pump):
    # Sydney's clocks go forward on 4 Oct 2026: October has 743 hours
    usage = UsageForecast.calendar(
        "2026-10-01", "2026-11-01", time_zone="Australia/Sydney"
    )
    october = events_by_period(constant_pump, usage, unit="hours")
    assert october.model_ends.tolist() == [743.0]
    assert october.events.system_failures[0] == pytest.approx(0.3715, abs=1e-6)


def test_cost_by_month_is_exact_on_average(constant_pump):
    usage = UsageForecast.calendar("2027-01-01", "2027-04-01")
    months = events_by_period(
        constant_pump,
        usage,
        unit="hours",
        cost=True,
        mc_samples=2000,
        seed=1,
    )
    # $1,000 a repair: 372, 336, 372
    np.testing.assert_allclose(
        months.cost.mean, [372.0, 336.0, 372.0], atol=1e-3
    )
    assert months.cost.levels == (5.0, 95.0)
    assert "simulated" in " ".join(months.notes)


def test_uc1_diagram_by_calendar_month_sums_to_the_year():
    fixed = surv.ExactEventTime.from_params
    rbd = RepairableRBD(
        [("s", "pump"), ("pump", "valve"), ("valve", "t")],
        {
            "pump": {
                "reliability": surv.Weibull.from_params([9000.0, 2.5]),
                "repairability": surv.LogNormal.from_params(
                    [np.log(8.0), 0.5]
                ),
                "preventive": {"interval": 6000.0, "duration": fixed([4.0])},
            },
            "valve": {
                "reliability": surv.Weibull.from_params([20000.0, 1.5]),
                "repairability": surv.Exponential.from_params([1 / 12.0]),
                "preventive": {
                    "interval": 8760.0,
                    "policy": "block",
                    "duration": fixed([6.0]),
                },
            },
        },
    )
    state = {
        "pump": NodeState(age=5200.0),
        "valve": NodeState(age=7000.0, since_scheduled=7000.0),
    }
    usage = UsageForecast.calendar("2027-01-01", "2028-01-01")
    months = events_by_period(rbd, usage, state=state, unit="hours")
    year = rbd.expected_events(8760.0, state=state)
    assert months.events.system_failures.sum() == pytest.approx(
        float(year.system_failures), abs=1e-6
    )
    # the valve's block replacement, 1,760 h on, falls in March (1,416 h
    # to the end of February, 2,160 to the end of March)
    # (a hair under 1: the replacement is not done while the valve is in
    # a repair then)
    preventive = months.events.node_preventive["valve"]
    assert preventive[2] > 0.999
    assert np.delete(preventive, 2).max() < 1e-6


# ------------------------------------------- UC-40: shuts on a part's clock


def test_uc40_opportunity_in_calendar_days():
    overhaul = Opportunity.from_recurrence(
        "overhaul", 1.0, OVERHAUL, "2026-11-14", "2029-12-31", unit="days"
    )
    # 14 Nov 2026 -> 2 Nov 2027: 353 days; -> 7 Nov 2028: 371 more (a leap
    # year); -> 6 Nov 2029: 364 more
    assert overhaul.dates == (353.0, 724.0, 1088.0)
    assert not overhaul.periodic


def test_uc40_opportunity_in_operating_hours(pump):
    shut = Opportunity.from_recurrence(
        "overhaul", 500.0, OVERHAUL, "2026-11-16", "2027-11-30", usage=pump
    )
    # to 00:00 on Tue 2 Nov 2027: 4,000 h to the end of October and Monday
    # 1 November's 16 h
    assert shut.dates == (4016.0,)
    # weekends at one operating time: Saturday and Sunday are one time
    weekends = Opportunity.from_recurrence(
        "weekend",
        2.0,
        Recurrence.weekly(["Saturday", "Sunday"]),
        "2026-11-16",
        "2026-11-30",
        usage=pump,
    )
    assert weekends.dates == (80.0, 160.0)
    with pytest.raises(ValueError, match="no dates"):
        Opportunity.from_recurrence(
            "none", 1.0, OVERHAUL, "2026-11-16", "2027-10-31", usage=pump
        )


# ---------------------------------------------- UC-41: shuts on a diagram


def test_uc41_shutdown_spec_on_dates():
    spec = shutdown_spec(
        OVERHAUL,
        "2026-11-14",
        "2029-12-31",
        duration=48.0,
        at="06:00",
        cost=20_000.0,
    )
    # 353, 724 and 1,088 days, each at 06:00: 8,478, 17,382 and 26,118 h
    assert spec == {
        "times": [8478.0, 17382.0, 26118.0],
        "duration": 48.0,
        "cost": 20_000.0,
    }
    # Sydney: both ends in daylight saving, so the same hours
    sydney = shutdown_spec(
        OVERHAUL,
        "2026-11-14",
        "2027-12-31",
        duration=48.0,
        at="06:00",
        time_zone="Australia/Sydney",
    )
    assert sydney["times"] == [8478.0]
    with pytest.raises(ValueError, match="no dates"):
        shutdown_spec(OVERHAUL, "2026-11-14", "2027-10-01", duration=1.0)


def test_uc41_a_diagram_takes_the_dated_shuts(constant_pump):
    spec = shutdown_spec(
        OVERHAUL,
        "2026-11-14",
        "2028-12-31",
        duration=48.0,
        at="06:00",
        cost=20_000.0,
    )
    rbd = RepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {
            "pump": {
                "reliability": surv.Exponential.from_params([1 / 2000.0]),
                "repairability": "instant",
                "repair_cost": 1000.0,
            }
        },
        shutdowns={"overhaul": spec},
    )
    # two shuts in two years, $20,000 each
    spend = rbd.cost(17_520.0, mc_samples=200, seed=1)
    assert spend.by_category["shutdown"] == 40_000.0
    # the exact methods take shuts on a period only, and say so
    usage = UsageForecast.calendar("2026-11-14", "2027-12-01")
    with pytest.raises(NotImplementedError, match="given times"):
        events_by_period(rbd, usage, unit="hours")


# ---------------------------------------------- UC-42: two clocks' rate


class _TwoClocks:
    """Stands in for surpyval's EquivalenceModel (0.28)."""

    def at_usage(self, rate):
        return ("life at", rate)


def test_uc42_rate_for_a_life_on_two_clocks(pump):
    # one start a run, 16 h a run: 0.0625 starts an operating hour
    assert pump.life_at_usage(_TwoClocks()) == ("life at", 0.0625)
    with pytest.raises(TypeError, match="0.28"):
        pump.life_at_usage(surv.Weibull.from_params([1.0, 2.0]))


def test_uc42_with_surpyval_equivalence(pump):
    if not hasattr(getattr(surv, "EquivalenceModel", None), "at_usage"):
        pytest.skip("a life on two clocks needs surpyval 0.28 or later")
    model = surv.WeibullEquivalence.from_params([2000.0, 2.0, 10.0])
    life = pump.life_at_usage(model)
    # T = hours (1 + 10 x 0.0625) = 1.625 hours: Weibull scale 2000 / 1.625
    # = 1,230.8 h, shape 2; mean 1,230.8 x 0.88623 = 1,090.7 h
    assert life.mean() == pytest.approx(1090.7, abs=0.1)
