"""A life counted in a unit of use (km, cycles) on a diagram: its block's
duty is that use's rate per unit of the diagram's clock (400 km a day),
read from the duty plan's meter (``CalendarContext(use=)``), as a share
of the time is for a life in operating hours (UC-69 of the schedule
README)."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, RepairableRBD
from repyability.maintenance import Prices, Restoration, UnitPlan
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json
from repyability.schedule import (
    CalendarContext,
    DutyPlan,
    DutySchedule,
    Rate,
    Recurrence,
    Replaces,
    Services,
    Shifts,
    Task,
    UsageForecast,
    events_by_period,
    failures_by_period,
    task_value,
)

ONE = [("s", "tyre"), ("tyre", "t")]
KM = surv.Weibull.from_params([60_000.0, 2.0])
FRAME = UsageForecast.calendar("2027-01-01", "2028-01-01")
TRUCK = DutyPlan(Rate(400, per="day", meter="km")).forecast(
    "2027-01-01", "2028-01-01"
)
# 500 km on each weekday, none at weekends.
WEEKDAYS = DutyPlan(
    Shifts("Mon-Fri", "06:00", "22:00"), per_hour={"km": 500.0 / 16.0}
).forecast("2027-01-01", "2028-01-01")
SPEC = {"reliability": KM, "repairability": "instant", "repair_cost": 100.0}


def on(plan, spec=SPEC):
    return CalendarContext(FRAME, {"tyre": plan}, use={"tyre": "km"})


def test_the_duty_is_the_rate_of_use():
    steps = TRUCK.schedule("hours", use="km")
    assert steps.times == (0.0,) and steps.use == "km"
    assert math.isclose(steps.shares[0], 400.0 / 24.0, rel_tol=1e-12)
    miles = TRUCK.schedule("days", use="miles").shares[0]
    assert math.isclose(miles, 400.0 / 1.609344, rel_tol=1e-12)
    with pytest.raises(ValueError, match="unit of time"):
        TRUCK.schedule("hours", use="hours")


def test_a_km_life_at_400_km_a_day_is_its_life_in_days():
    # Weibull(60,000 km, 2) at 400 km a day is Weibull(150 days, 2).
    rbd = RepairableRBD(ONE, {"tyre": SPEC})
    months = events_by_period(rbd, on(TRUCK), cost=True, discount=0.07)
    days = {**SPEC, "reliability": surv.Weibull.from_params([3600.0, 2.0])}
    reference = events_by_period(
        RepairableRBD(ONE, {"tyre": days}),
        FRAME,
        cost=True,
        discount=0.07,
        unit="hours",
    )
    np.testing.assert_allclose(
        months.events.node_failures["tyre"],
        reference.events.node_failures["tyre"],
        rtol=1e-12,
    )
    assert math.isclose(months.npv, reference.npv, rel_tol=1e-12)
    assert any("'km' meter" in note for note in months.notes)


@pytest.mark.parametrize("plan", [TRUCK, WEEKDAYS], ids=["rate", "weekdays"])
def test_the_diagram_agrees_with_the_part_alone(plan):
    rbd = RepairableRBD(ONE, {"tyre": SPEC})
    months = events_by_period(rbd, on(plan))
    part = failures_by_period(
        NonRepairable(KM, unplanned_cost=100.0), plan, unit="km"
    )
    np.testing.assert_allclose(
        np.cumsum(months.events.node_failures["tyre"]),
        np.cumsum(part.expected_failures),
        rtol=1e-6,
    )
    # By hand: 21 weekdays in January 2027, 10,500 km, H = 0.030625; the
    # renewals a little fewer.
    if plan is WEEKDAYS:
        first = months.events.node_failures["tyre"][0]
        assert 0.0300 < first < (10_500 / 60_000) ** 2


def test_a_task_counted_in_km_on_a_varying_duty():
    # Replaced at 40,000 km (or at a failure): exact through the plan,
    # and its simulation within four standard errors (bound fixed first).
    spec = {
        **SPEC,
        "preventive": {
            "policy": "age",
            "interval": 40_000.0,
            "basis": "operating",
            "cost": 30.0,
        },
    }
    context = on(WEEKDAYS)
    rbd = RepairableRBD(ONE, {"tyre": spec})
    months = events_by_period(rbd, context, cost=True)
    exact = float(np.sum(months.cost.mean))
    run = context.diagram(rbd).cost(
        8760.0, mc_samples=4000, seed=1, control_variate=False
    )
    assert abs(run.mean - exact) < 4.0 * run.mean_se
    assert months.events.node_preventive["tyre"].sum() > 0.0


SERVICE = Task(
    "tyre rotation",
    every=Recurrence.every(3, "months", "2027-04-01"),
    cost=800.0,
)


def se(worth) -> float:
    total = worth.total
    return (total["value_upper"] - total["value_lower"]) / (2 * 1.6448536)


@pytest.mark.parametrize("plan", [TRUCK, WEEKDAYS], ids=["rate", "weekdays"])
def test_a_service_by_km_against_the_replacement(plan):
    # As new (factor 0), from 30,000 km: the replacement's exact value.
    life = surv.Weibull.from_params([60_000.0, 3.0])
    part = NonRepairable(life, unplanned_cost=12000.0)
    unit_plan = UnitPlan(life, "instant", prices=Prices(repair=12000.0))
    kwargs = dict(age=30_000.0, unit="km", discount=0.07, period="quarter")
    exact = task_value(SERVICE, Replaces(part), plan, **kwargs)
    worth = task_value(
        SERVICE,
        Services(unit_plan, Restoration(q=0.0)),
        plan,
        seed=67,
        **kwargs,
    )
    assert abs(worth.total["value"] - exact.total["value"]) < 4 * se(worth)


def test_a_state_on_a_share_of_the_time_is_its_operating_age():
    # 12 h a day, from 4,000 operating hours: the service as new equals
    # the replacement's exact value (the age was taken as calendar time).
    half = DutyPlan(Rate(12, per="day")).forecast("2027-01-01", "2028-01-01")
    life = surv.Weibull.from_params([8760.0, 3.0])
    kwargs = dict(age=4000.0, discount=0.07, period="quarter")
    exact = task_value(
        SERVICE,
        Replaces(NonRepairable(life, unplanned_cost=12000.0)),
        half,
        **kwargs,
    )
    worth = task_value(
        SERVICE,
        Services(
            UnitPlan(life, "instant", prices=Prices(repair=12000.0)),
            Restoration(q=0.0),
        ),
        half,
        seed=67,
        **kwargs,
    )
    assert abs(worth.total["value"] - exact.total["value"]) < 4 * se(worth)


def test_saving_keeps_the_rate():
    rbd = on(WEEKDAYS).diagram(RepairableRBD(ONE, {"tyre": SPEC}))
    again = rbd_from_json(rbd_to_json(rbd))
    duty = again._init_args["components"]["tyre"]["duty"]
    assert isinstance(duty, DutySchedule) and duty.use == "km"
    a = rbd.availability(2000.0, mc_samples=50, seed=1).availability
    b = again.availability(2000.0, mc_samples=50, seed=1).availability
    np.testing.assert_array_equal(a, b)
    assert on(TRUCK).to_dict()["use"] == {"tyre": "km"}


def test_what_is_refused():
    with pytest.raises(ValueError, match="give the node its duty plan"):
        CalendarContext(FRAME, use={"tyre": "km"})
    with pytest.raises(ValueError, match="no meter"):
        CalendarContext(FRAME, {"tyre": FRAME}, use={"tyre": "km"})
    with pytest.raises(ValueError, match="rates of km"):
        RepairableRBD(
            ONE,
            {
                "tyre": {
                    **SPEC,
                    "duty": DutySchedule("hours", (0.0,), (-1.0,), "km"),
                }
            },
        )
    # A share of the time above 1 is still refused.
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        RepairableRBD(ONE, {"tyre": {**SPEC, "duty": 16.7}})
