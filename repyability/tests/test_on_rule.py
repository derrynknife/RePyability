"""A unit plan's task on a recurrence rule (UC-71 of the schedule README):
put on the calendar's dates by ``CalendarContext.diagram``, refused
without a calendar."""

import numpy as np
import pytest
import surpyval as surv

from repyability import OnRule, RepairableRBD, UnitPlan
from repyability.schedule import CalendarContext, Recurrence, UsageForecast

ONE = [("input", "a"), ("a", "output")]
QUARTERLY = Recurrence.monthly(
    weekday="Monday", nth=1, every=3, anchor="2027-01-01"
)
YEAR = CalendarContext(UsageForecast.calendar("2027-01-01", "2028-01-01"))
# 4 January, 5 April, 5 July and 4 October 2027, at 06:00.
HAND = np.array([3, 94, 185, 276]) * 24.0 + 6.0


def plan(life=None, **task):
    return UnitPlan(
        life or surv.Weibull.from_params([3000.0, 2.0]),
        surv.Exponential.from_params([0.125]),
        tasks=(OnRule(QUARTERLY, at="06:00", cost=50.0, **task),),
        prices={"repair": 100.0},
    )


def given_times(unit_plan):
    spec = unit_plan.spec()
    task = {
        k: v for k, v in spec["preventive"].items() if k not in ("rule", "at")
    }
    spec["preventive"] = {**task, "replace_at": tuple(HAND)}
    return RepairableRBD(ONE, {"a": spec})


def test_the_task_falls_on_the_rules_dates():
    rbd = YEAR.diagram(ONE, {"a": plan()})
    np.testing.assert_array_equal(rbd._preventive["a"].times, HAND)
    np.testing.assert_array_equal(YEAR.rule_times(QUARTERLY, "06:00"), HAND)


def test_every_exact_value_is_the_diagram_given_the_times():
    unit = plan(duration=surv.Exponential.from_params([0.5]))
    on_rule = YEAR.diagram(ONE, {"a": unit})
    given = given_times(unit)
    t = np.linspace(0.0, 8760.0, 9)
    np.testing.assert_array_equal(
        on_rule.point_availability(t), given.point_availability(t)
    )
    assert on_rule.expected_cost(8760.0) == given.expected_cost(8760.0)
    a, b = on_rule.expected_events(8760.0), given.expected_events(8760.0)
    assert a.node_preventive == b.node_preventive
    assert a.node_failures == b.node_failures


def test_a_memoryless_unit_by_hand():
    # MTTF 2,000 h, repaired at once at 100: January is the task (50) and
    # 744 / 2,000 failures; February no task.
    unit = UnitPlan(
        surv.Exponential.from_params([1 / 2000.0]),
        "instant",
        tasks=(OnRule(QUARTERLY, at="06:00", cost=50.0),),
        prices={"repair": 100.0},
    )
    months = unit.events_by_period(YEAR, cost=True)
    np.testing.assert_allclose(
        months.events.node_preventive["unit"],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    )
    np.testing.assert_allclose(months.cost.mean[:2], [87.2, 33.6], rtol=1e-7)
    np.testing.assert_allclose(
        unit.expected_cost(8760.0, calendar=YEAR).mean,
        200.0 + 100.0 * 8760.0 / 2000.0,
        rtol=1e-7,
    )


def test_the_plan_keeps_its_rule():
    unit = plan()
    assert UnitPlan.from_spec(unit.spec()) == unit
    back = UnitPlan.from_dict(unit.to_dict())
    assert back.tasks == unit.tasks
    assert unit.to_dict()["spec"]["preventive"]["at"] == "06:00"


def test_without_a_calendar_it_is_refused():
    with pytest.raises(ValueError, match="need a calendar"):
        plan().diagram()
    with pytest.raises(ValueError, match="need a calendar"):
        RepairableRBD(ONE, {"a": plan().spec()})
    with pytest.raises(ValueError, match="need a calendar"):
        plan().expected_npv(8760.0)


def test_a_rule_with_no_date_in_the_calendar_leaves_no_task():
    january = CalendarContext(
        UsageForecast.calendar("2027-01-05", "2027-02-01")
    )
    rbd = january.diagram(ONE, {"a": plan()})
    assert "a" not in rbd._preventive


def test_the_task_is_on_the_calendars_clock():
    spec = plan().spec()
    spec["preventive"]["basis"] = "operating"
    with pytest.raises(ValueError, match="basis cannot be 'operating'"):
        YEAR.diagram(ONE, {"a": spec})


def test_the_rule_must_be_a_recurrence():
    with pytest.raises(TypeError, match="must be a Recurrence"):
        OnRule("quarterly")
    with pytest.raises(ValueError):
        OnRule(QUARTERLY, at="25:00")


def test_diagram_takes_a_diagram_or_its_parts():
    rbd = YEAR.diagram(ONE, {"a": plan()})
    assert YEAR.diagram(rbd) is rbd
    with pytest.raises(TypeError, match="not both"):
        YEAR.diagram(rbd, {"a": plan()})
    with pytest.raises(TypeError, match="options"):
        YEAR.diagram(rbd, repair_crews=1)
