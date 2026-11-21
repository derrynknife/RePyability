"""Duty plans that change over time, and the calendar an analysis runs on
(UC-64 of the schedule README).

A block's ``"duty"`` takes a share of the time or a ``DutySchedule``: on a
plan that changes, its life is drawn in operating hours and put on the
calendar through the plan (``_duty.PlannedLife``), and a task counted in
operating time falls when the plan reaches it. A unit whose repairs and
tasks take no time is then its operating-time process seen through the
plan, ``M(D(t))``, which the exact methods give on the one-node diagram
with no duty.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import _compiled, _event_loop
from repyability.rbd._duty import PlannedLife
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json
from repyability.schedule import (
    CalendarContext,
    DutyPlan,
    DutySchedule,
    Recurrence,
    Shifts,
    UsageForecast,
    events_by_period,
)

ONE = [("s", "a"), ("a", "t")]
W = surv.Weibull.from_params([400.0, 2.0])
PLAN = DutySchedule(
    "hours", (0.0, 1000.0, 2000.0, 3000.0), (1.0, 0.25, 0.0, 0.5)
)


def planned(**more):
    spec = {"reliability": W, "repairability": "instant", "repair_cost": 1.0}
    return RepairableRBD(ONE, {"a": {**spec, **more, "duty": PLAN}})


# -- the plan's clock -------------------------------------------------------


def test_the_operating_time_and_its_inverse():
    life = PlannedLife(W, PLAN, "a")
    t = np.array([0.0, 500.0, 1000.0, 1400.0, 2500.0, 3000.0, 3600.0])
    np.testing.assert_array_equal(
        life.operating(t), [0.0, 500.0, 1000.0, 1100.0, 1250.0, 1250.0, 1550.0]
    )
    # The first time each is reached: a stretch at no duty is passed over.
    assert [life.inverse(h) for h in (0.0, 1100.0, 1250.0, 1550.0)] == [
        0.0,
        1400.0,
        2000.0,
        3600.0,
    ]
    np.testing.assert_allclose(life.sf(t), W.sf(life.operating(t)))


def test_a_life_is_drawn_in_operating_hours_from_when_it_starts():
    life = PlannedLife(W, PLAN, "a")
    u = 0.3
    x = float(W.qf(np.array([u]))[0])
    # Put in at 1500 h (1125 operating hours), it fails when the plan
    # has run x more.
    assert life.life_left(1500.0, 0.0, u) == life.inverse(1125.0 + x) - 1500.0


def test_a_plan_of_one_share_is_that_duty_to_the_bit():
    one = DutySchedule("hours", (0.0,), (0.25,))
    spec = {
        "reliability": W,
        "repairability": surv.Exponential.from_params([0.1]),
    }
    by_plan = RepairableRBD(ONE, {"a": {**spec, "duty": one}})
    by_share = RepairableRBD(ONE, {"a": {**spec, "duty": 0.25}})
    assert by_plan.components["a"].reliability.params.tolist() == (
        by_share.components["a"].reliability.params.tolist()
    )
    a = by_plan.availability(2000.0, mc_samples=200, seed=4)
    b = by_share.availability(2000.0, mc_samples=200, seed=4)
    np.testing.assert_array_equal(a.availability, b.availability)


# -- the simulation against the operating-time process ----------------------


@pytest.mark.parametrize(
    "task",
    [
        None,
        {"policy": "block", "interval": 300.0},
        {"policy": "age", "interval": 300.0},
    ],
    ids=["run to failure", "block", "age"],
)
def test_a_unit_is_its_operating_process_seen_through_its_plan(task):
    more = (
        {} if task is None else {"preventive": {**task, "basis": "operating"}}
    )
    rbd = planned(**more)
    end = 4000.0
    hours = rbd.components["a"].reliability.operating(end)
    spec = {"reliability": W, "repairability": "instant", "repair_cost": 1.0}
    if task is not None:
        spec["preventive"] = dict(task)
    exact = RepairableRBD(ONE, {"a": spec}).expected_events_by_window(
        np.array([hours])
    )
    run = rbd.cost(end, mc_samples=4000, seed=7)
    # Bounds fixed before the run: four standard errors.
    assert abs(run.mean - exact.node_failures["a"][0]) < 4.0 * run.mean_se


def test_a_usage_task_falls_where_the_forecast_reaches_it():
    frame = UsageForecast.calendar("2027-01-01", "2028-01-01")
    shifts = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2027-01-01", "2028-01-01"
    )
    context = CalendarContext(frame, {"a": shifts})
    spec = {
        "reliability": surv.Weibull.from_params([3000.0, 2.0]),
        "repairability": "instant",
        "preventive": {"interval": 4000.0, "basis": "operating"},
    }
    rbd = context.diagram(RepairableRBD(ONE, {"a": spec}))
    due = rbd._preventive["a"].due(0.0)
    expected = context.model_time([shifts.moment(4000.0)])[0]
    assert math.isclose(due, expected, rel_tol=0.0, abs_tol=1e-9)
    assert shifts.moment(4000.0).isoformat().startswith("2027-12-16T22:00")


def test_a_task_on_a_rule_is_replaced_at_its_dates():
    frame = UsageForecast.calendar("2027-01-01", "2028-01-01")
    context = CalendarContext(frame)
    rule = Recurrence.monthly(
        weekday="Monday", nth=1, every=3, anchor="2027-01-01"
    )
    spec = {
        "reliability": W,
        "repairability": "instant",
        "preventive": {"replace_at": context.rule_times(rule, at="06:00")},
    }
    rbd = context.diagram(RepairableRBD(ONE, {"a": spec}))
    # 4 January, 5 April, 5 July and 4 October, at 06:00.
    days = np.array([3, 94, 185, 276]) * 24.0 + 6.0
    np.testing.assert_allclose(rbd._preventive["a"].times, days)


def test_events_by_period_on_a_plan_that_changes():
    frame = UsageForecast.calendar("2027-01-01", "2027-04-01")
    shifts = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2027-01-01", "2027-04-01"
    )
    spec = {
        "reliability": surv.Weibull.from_params([300.0, 2.0]),
        "repairability": surv.Exponential.from_params([0.125]),
        "repair_cost": 100.0,
    }
    rbd = RepairableRBD(ONE, {"a": spec})
    months = events_by_period(
        rbd,
        CalendarContext(frame, {"a": shifts}),
        cost=True,
        discount=0.07,
        mc_samples=300,
        seed=2,
    )
    assert "simulations' histories" in months.notes[0]
    failures = months.events.node_failures["a"]
    np.testing.assert_allclose(months.cost.simulated_mean, failures * 100.0)
    assert np.all(months.expected_npv < months.cost.mean)


# -- saving and what is refused ---------------------------------------------


def test_saving_keeps_the_plan():
    rbd = planned()
    again = rbd_from_json(rbd_to_json(rbd))
    assert again._init_args["components"]["a"]["duty"] == PLAN
    a = rbd.availability(3000.0, mc_samples=100, seed=1).availability
    b = again.availability(3000.0, mc_samples=100, seed=1).availability
    np.testing.assert_array_equal(a, b)


def test_the_compiled_engine_hands_it_to_python():
    rbd = planned()
    plan = _event_loop._stream_plan(rbd, 1.0, 0, False)[0]
    reason = _compiled.unsupported(rbd, plan, None, numba=True)
    assert reason == "the duty plan that changes over time of node 'a'"


@pytest.mark.parametrize(
    "more, words",
    [
        ({"repair": {"model": "kijima1", "q": 0.5}}, "virtual age"),
        (
            {
                "preventive": {
                    "policy": "condition",
                    "interval": 50.0,
                    "threshold": 0.1,
                }
            },
            "on condition",
        ),
        (
            {"inspection": {"interval": 100.0, "basis": "operating"}},
            "tests counted in operating time",
        ),
    ],
)
def test_what_a_plan_that_changes_refuses(more, words):
    with pytest.raises(NotImplementedError, match=words):
        planned(**more)


def test_the_exact_methods_refuse_and_a_state_must_be_new():
    rbd = planned()
    with pytest.raises(NotImplementedError, match="changes over time"):
        rbd.expected_events_by_window(np.array([1000.0]))
    with pytest.raises(NotImplementedError, match="other than new"):
        rbd.availability(
            1000.0, mc_samples=10, seed=1, state={"a": NodeState(age=10.0)}
        )


def test_a_plan_is_checked():
    with pytest.raises(ValueError, match="must start at 0"):
        RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": W,
                    "duty": DutySchedule("hours", (5.0, 9.0), (1.0, 0.5)),
                }
            },
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": W,
                    "duty": DutySchedule("hours", (0.0, 9.0), (1.0, 1.5)),
                }
            },
        )
