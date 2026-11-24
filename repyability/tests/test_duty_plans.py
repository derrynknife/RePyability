"""Duty plans that change over time, and the calendar an analysis runs on
(UC-64 of the schedule README).

A block's ``"duty"`` takes a share of the time or a ``DutySchedule``: on a
plan that changes, its life is drawn in operating hours and put on the
calendar through the plan (``_duty.PlannedLife``), and a task counted in
operating time falls when the plan reaches it. A unit whose repairs and
tasks take no time is then its operating-time process seen through the
plan, ``M(D(t))``: the exact methods follow it so (a
``TimeChangedCurve``), and the simulation is checked against the one-node
diagram with no duty.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, NonRepairable, RepairableRBD
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
    failures_by_period,
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
    # The simulation alone (not the exact means it would take).
    run = rbd.cost(end, mc_samples=4000, seed=7, control_variate=False)
    # Bounds fixed before the run: four standard errors.
    assert abs(run.mean - exact.node_failures["a"][0]) < 4.0 * run.mean_se


# -- exact values through the plan ------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        None,
        {"policy": "block", "interval": 300.0},
        {"policy": "age", "interval": 300.0},
    ],
    ids=["run to failure", "block", "age"],
)
def test_the_exact_methods_follow_the_operating_process_through_the_plan(
    task,
):
    more = (
        {} if task is None else {"preventive": {**task, "basis": "operating"}}
    )
    rbd = planned(**more)
    ends = np.array([500.0, 1500.0, 2000.0, 2500.0, 4000.0])
    hours = rbd.components["a"].reliability.operating(ends)
    spec = {"reliability": W, "repairability": "instant", "repair_cost": 1.0}
    if task is not None:
        spec["preventive"] = dict(task)
    alone = RepairableRBD(ONE, {"a": spec})
    # The plan stands still from 2000 to 3000: 2500 h is 1250 operating
    # hours, as 2000 h is.
    reference = alone.expected_events_by_window(np.unique(hours))
    by_hours = {
        key: np.interp(
            hours,
            np.unique(hours),
            np.cumsum(getattr(reference, key)["a"]),
        )
        for key in ("node_failures", "node_preventive")
    }
    got = rbd.expected_events_by_window(ends)
    for key, want in by_hours.items():
        np.testing.assert_allclose(
            np.cumsum(getattr(got, key)["a"]), want, rtol=0.0, atol=1e-12
        )
    assert got.node_failures["a"][3] == 0.0  # no running, no failures
    assert rbd.analysis_routes()["expected_events"].route == "numerical"


def test_a_constant_rate_through_the_plan_against_the_closed_form():
    # An exponential life in operating hours, repaired at once: it fails
    # at its rate times the share run, so by t it has failed lam D(t)
    # times, and a repair cost c is worth c lam (integral of exp(-r s)
    # times the share at s) at 0.
    lam, c, r = 1.0 / 500.0, 80.0, 2e-4
    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": surv.Exponential.from_params([lam]),
                "repairability": "instant",
                "repair_cost": c,
                "duty": PLAN,
            }
        },
    )
    ends = np.array([700.0, 1600.0, 2600.0, 3900.0])
    events = rbd.expected_events_by_window(ends)
    hours = rbd.components["a"].reliability.operating(ends)
    np.testing.assert_allclose(
        np.cumsum(events.node_failures["a"]), lam * hours, rtol=1e-7
    )
    times = np.array(PLAN.times + (3900.0,))
    shares = np.array(PLAN.shares)
    npv = (
        c
        * lam
        * float(
            np.sum(shares * (np.exp(-r * times[:-1]) - np.exp(-r * times[1:])))
            / r
        )
    )
    got = rbd.expected_cost(3900.0, discount_rate=r).mean
    assert math.isclose(got, npv, rel_tol=1e-7)
    assert math.isclose(
        rbd.expected_cost(3900.0).mean, c * lam * hours[-1], rel_tol=1e-7
    )


def test_a_plan_that_changes_beside_a_unit_on_the_calendar():
    # A series pair, both repaired at once: the system fails whenever
    # either does, so its failures are theirs added up.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {"reliability": W, "repairability": "instant", "duty": PLAN},
            "b": {
                "reliability": surv.Exponential.from_params([1 / 900.0]),
                "repairability": "instant",
            },
        },
    )
    ends = np.array([1000.0, 2500.0, 4000.0])
    got = rbd.expected_events_by_window(ends)
    np.testing.assert_allclose(
        got.system_failures,
        got.node_failures["a"] + got.node_failures["b"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.cumsum(got.node_failures["b"]), ends / 900.0, rtol=1e-7
    )


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


def test_events_by_period_through_a_plan_agrees_with_the_part_alone():
    # Repaired at once, the unit is its operating process seen through
    # the shifts: events_by_period is exact, and gives what the part's
    # own forecast through the same shifts gives.
    frame = UsageForecast.calendar("2027-01-01", "2027-07-01")
    shifts = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2027-01-01", "2027-07-01"
    )
    life = surv.Weibull.from_params([900.0, 2.0])
    spec = {"reliability": life, "repairability": "instant"}
    rbd = RepairableRBD(ONE, {"a": {**spec, "repair_cost": 100.0}})
    months = events_by_period(
        rbd,
        CalendarContext(frame, {"a": shifts}),
        cost=True,
        discount=0.07,
        mc_samples=200,
        seed=2,
    )
    assert not any("histories" in note for note in months.notes)
    part = failures_by_period(
        NonRepairable(life, unplanned_cost=100.0), shifts, discount=0.07
    )
    np.testing.assert_allclose(
        months.events.node_failures["a"], part.expected_failures, rtol=1e-6
    )
    assert part.expected_cost is not None
    np.testing.assert_allclose(months.cost.mean, part.expected_cost, rtol=1e-6)
    assert months.npv is not None and part.npv is not None
    assert part.npv_precision is not None
    assert abs(months.npv - part.npv) <= part.npv_precision


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


@pytest.mark.parametrize(
    "more, words",
    [
        (
            {"repairability": surv.Exponential.from_params([0.1])},
            "its repairs take time",
        ),
        (
            {"preventive": {"interval": 300.0}},
            "counted in calendar time",
        ),
        (
            {
                "preventive": {
                    "interval": 300.0,
                    "basis": "operating",
                    "duration": surv.Exponential.from_params([0.5]),
                }
            },
            "its preventive task takes time",
        ),
    ],
)
def test_the_exact_methods_refuse_what_takes_time(more, words):
    rbd = planned(**more)
    with pytest.raises(NotImplementedError, match=words):
        rbd.expected_events_by_window(np.array([1000.0]))
    route = rbd.analysis_routes()["expected_events"]
    assert route.route == "refused" and words in route.reason


def test_a_state_must_be_new():
    rbd = planned()
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
