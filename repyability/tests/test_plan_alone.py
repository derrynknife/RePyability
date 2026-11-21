"""A unit's plan valued alone (UC-65 of the maintenance README): every
valuation of a ``UnitPlan`` is its one-node diagram's, so the plan alone
equals a diagram holding it, on every plan kind of the catalogue, with
and without a calendar; and the plan's own salvage, escalation and
retirement are the valuation's arguments for its node."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD, Timeline
from repyability.maintenance import (
    AgePolicy,
    BlockPolicy,
    Inspection,
    Prices,
    UnitPlan,
)
from repyability.rbd import routes
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json
from repyability.schedule import (
    CalendarContext,
    DutyPlan,
    Shifts,
    UsageForecast,
    events_by_period,
)
from repyability.tests import catalogue

W = surv.Weibull.from_params([400.0, 2.0])
REPAIR = surv.LogNormal.from_params([1.0, 0.5])
HORIZON = 300.0
RATE = 1e-3


def holding(plan: UnitPlan) -> RepairableRBD:
    """A diagram holding ``plan`` as its one node, drawn with other end
    nodes than the plan's own diagram (the node keeps its name: a seeded
    run's streams are named by their node)."""
    node = UnitPlan.NODE
    return RepairableRBD([("in", node), (node, "out")], {node: plan})


def _plans():
    """Each distinct plan the catalogue's repairable diagrams hold that
    can stand alone (no maintenance group, no site shutdown, no standby
    group of the diagram's), by a name."""
    out = {}
    for name, rbd in catalogue.repairable_kinds().items():
        for node, plan in rbd.plans.items():
            if plan.group is not None or plan.standby is not None:
                continue
            task = plan.task("preventive")
            if getattr(task, "shutdown", None) is not None:
                continue
            if getattr(plan.task("inspection"), "shutdown", None):
                continue
            if getattr(task, "policy", None) is not None and not isinstance(
                getattr(task, "policy"), str
            ):
                continue  # a calendar policy is built from its site
            key = repr(plan.spec())
            if key not in {repr(p.spec()) for p in out.values()}:
                out[f"{name}: {node}"] = plan
    return out


PLANS = _plans()


def outcome(call):
    """``call()``'s result, or the refusal it raised (its type and
    message)."""
    try:
        return call()
    except (NotImplementedError, ValueError) as error:
        return (type(error), str(error))


def same_npv(a, b) -> None:
    if isinstance(a, tuple) or isinstance(b, tuple):
        assert a == b
        return
    for field in ("npv", "value", "cost", "salvage", "eac"):
        x, y = getattr(a, field), getattr(b, field)
        assert x == pytest.approx(y, rel=1e-12, abs=1e-12), field
    mine = [row["present_value"] for row in a.cash_flows]
    theirs = [row["present_value"] for row in b.cash_flows]
    np.testing.assert_allclose(mine, theirs, rtol=1e-12, atol=1e-12)


def test_the_catalogue_has_plans_of_many_kinds():
    assert len(PLANS) >= 15


@pytest.mark.parametrize("name", sorted(PLANS))
def test_a_plan_alone_is_a_diagram_holding_it(name):
    plan = PLANS[name]
    other = holding(plan)
    options = dict(discount_rate=RATE, value_rate=1.0, periods=3)
    same_npv(
        outcome(lambda: plan.expected_npv(HORIZON, **options)),
        outcome(lambda: other.expected_npv(HORIZON, **options)),
    )
    rate = outcome(plan.expected_cost_rate)
    theirs = outcome(other.expected_cost_rate)
    if isinstance(rate, tuple) or isinstance(theirs, tuple):
        assert rate == theirs
    else:
        assert rate == pytest.approx(theirs, rel=1e-12, abs=1e-12)
    runs = dict(discount_rate=RATE, value_rate=1.0, mc_samples=40, seed=3)
    mine = outcome(lambda: plan.npv_distribution(HORIZON, **runs))
    theirs = outcome(lambda: other.npv_distribution(HORIZON, **runs))
    if isinstance(mine, tuple) or isinstance(theirs, tuple):
        assert mine == theirs
    else:
        np.testing.assert_allclose(
            mine.samples, theirs.samples, rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("name", sorted(PLANS))
def test_a_plan_says_how_it_is_valued_as_its_diagram_does(name):
    plan = PLANS[name]
    mine = plan.analysis_routes()
    theirs = holding(plan).analysis_routes()
    assert mine["expected_npv"].route == theirs["expected_npv"].route
    assert mine["cash_flows"] == mine["expected_npv"]
    assert mine["npv_distribution"].route == theirs["npv_distribution"].route


# -- on a calendar -----------------------------------------------------------

FRAME = UsageForecast.calendar("2027-01-04", "2027-02-01")
SHIFTS = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
    "2027-01-04", "2027-02-01"
)
SHARE = CalendarContext(FRAME, {UnitPlan.NODE: 0.5})
CHANGING = CalendarContext(FRAME, {UnitPlan.NODE: SHIFTS})

ON_CALENDAR = {
    "run to failure": UnitPlan(W, REPAIR, prices=Prices(repair=5.0)),
    "age in operating hours": UnitPlan(
        W,
        REPAIR,
        tasks=(
            AgePolicy(
                interval=250.0, cost=2.0, duration=REPAIR, basis="operating"
            ),
        ),
        prices=Prices(repair=5.0, downtime=1.0),
    ),
    "block": UnitPlan(
        W,
        REPAIR,
        tasks=(BlockPolicy(interval=200.0, cost=2.0, duration=REPAIR),),
        prices=Prices(repair=5.0),
    ),
}


def on_calendar(plan, calendar):
    return calendar.diagram(
        RepairableRBD(
            [("s", UnitPlan.NODE), (UnitPlan.NODE, "t")],
            {UnitPlan.NODE: plan},
        )
    )


@pytest.mark.parametrize("calendar", [SHARE, CHANGING], ids=["share", "plan"])
@pytest.mark.parametrize("name", sorted(ON_CALENDAR))
def test_a_plan_alone_on_a_calendar_is_its_diagram_on_it(name, calendar):
    plan = ON_CALENDAR[name]
    other = on_calendar(plan, calendar)
    runs = dict(discount_rate=RATE, mc_samples=40, seed=11)
    np.testing.assert_allclose(
        plan.npv_distribution(HORIZON, calendar=calendar, **runs).samples,
        other.npv_distribution(HORIZON, **runs).samples,
        rtol=1e-12,
        atol=1e-12,
    )
    same_npv(
        outcome(lambda: plan.expected_npv(HORIZON, calendar=calendar)),
        outcome(lambda: other.expected_npv(HORIZON)),
    )
    events = dict(period="week", mc_samples=40, seed=2)
    mine = plan.failures_by_period(calendar, **events)
    theirs = events_by_period(other, calendar, **events)
    assert json.dumps(mine.to_dict(), default=str) == json.dumps(
        theirs.to_dict(), default=str
    )


def test_a_calendar_must_name_the_plan_node():
    plan = ON_CALENDAR["block"]
    calendar = CalendarContext(FRAME, {"pump": 0.5})
    with pytest.raises(ValueError, match="no node 'pump'"):
        plan.expected_npv(HORIZON, calendar=calendar)
    with pytest.raises(TypeError, match="CalendarContext"):
        plan.expected_npv(HORIZON, calendar=FRAME)


# -- the plan's own money ----------------------------------------------------

TIMED = UnitPlan(
    W,
    REPAIR,
    tasks=(AgePolicy(interval=250.0, cost=2.0, duration=REPAIR),),
    prices=Prices(repair=5.0, downtime=1.0, acquisition=20.0),
)


def with_money(**money) -> UnitPlan:
    from dataclasses import replace

    return replace(TIMED, **money)


def test_its_salvage_escalation_and_retirement_are_the_arguments_for_it():
    options = dict(discount_rate=RATE, value_rate=1.0, periods=4)
    plan = with_money(salvage=7.0, escalation=2e-4, retire_at=240.0)
    same_npv(
        plan.expected_npv(HORIZON, **options),
        TIMED.expected_npv(
            HORIZON,
            salvage={UnitPlan.NODE: 7.0},
            escalation={UnitPlan.NODE: 2e-4},
            retire_at=240.0,
            **options,
        ),
    )


def test_a_salvage_by_age_is_the_argument_for_it():
    def worth(age):
        return 10.0 * np.exp(-np.asarray(age) / 300.0)

    # A salvage by age is worked out for a plain unit from new.
    plain = UnitPlan(W, REPAIR, prices=Prices(repair=5.0))
    from dataclasses import replace

    same_npv(
        replace(plain, salvage=worth).expected_npv(
            HORIZON, discount_rate=RATE
        ),
        plain.expected_npv(
            HORIZON, discount_rate=RATE, salvage={UnitPlan.NODE: worth}
        ),
    )


def test_the_arguments_take_precedence_over_the_plan():
    plan = with_money(salvage=7.0, escalation=2e-4, retire_at=240.0)
    same_npv(
        plan.expected_npv(
            HORIZON,
            discount_rate=RATE,
            salvage={UnitPlan.NODE: 1.0},
            escalation={UnitPlan.NODE: 0.0},
            retire_at=100.0,
        ),
        TIMED.expected_npv(
            HORIZON,
            discount_rate=RATE,
            salvage={UnitPlan.NODE: 1.0},
            retire_at=100.0,
        ),
    )
    # The system's salvage, a number, is added to the unit's own.
    both = plan.expected_npv(HORIZON, discount_rate=RATE, salvage=2.0)
    alone = plan.expected_npv(HORIZON, discount_rate=RATE)
    assert both.salvage - alone.salvage == pytest.approx(
        2.0 * math.exp(-RATE * 240.0), rel=1e-12
    )


def test_a_retirement_after_the_horizon_is_the_horizon():
    late = with_money(retire_at=10 * HORIZON)
    same_npv(
        late.expected_npv(HORIZON, discount_rate=RATE),
        TIMED.expected_npv(HORIZON, discount_rate=RATE),
    )


def test_a_salvage_on_a_diagram_of_several_is_its_node_s():
    unit = with_money(salvage=7.0, escalation=2e-4)
    other = {"reliability": W, "repairability": REPAIR, "repair_cost": 3.0}
    edges = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
    mine = RepairableRBD(edges, {"a": unit, "b": other})
    given = RepairableRBD(edges, {"a": TIMED, "b": other})
    same_npv(
        mine.expected_npv(HORIZON, discount_rate=RATE),
        given.expected_npv(
            HORIZON,
            discount_rate=RATE,
            salvage={"a": 7.0},
            escalation={"a": 2e-4},
        ),
    )


def test_one_unit_retiring_in_a_diagram_of_several_is_refused():
    unit = with_money(retire_at=100.0)
    other = {"reliability": W, "repairability": REPAIR, "repair_cost": 3.0}
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": unit, "b": other}
    )
    with pytest.raises(NotImplementedError, match="retiring one unit"):
        rbd.expected_npv(HORIZON)
    assert rbd.analysis_routes()["expected_npv"].route == routes.REFUSED
    # Retiring the whole system is the valuation's.
    rbd.expected_npv(HORIZON, retire_at=100.0)


def test_the_valuations_of_logs_refuse_what_they_leave_out():
    plan = with_money(salvage=7.0)
    message = "has its own 'salvage'"
    log = Timeline.from_outages([(50.0, 52.0)], 100.0)
    with pytest.raises(NotImplementedError, match=message):
        plan.plan_vs_actual(log, 100.0, HORIZON)
    assert plan.analysis_routes()["plan_vs_actual"].route == routes.REFUSED
    # The runs credit it (test_npv_runs_own_end.py).
    assert plan.analysis_routes()["npv_distribution"].route == (
        routes.SIMULATED
    )
    # Its own escalation the runs take, as the argument.
    grown = with_money(escalation=2e-4)
    runs = dict(discount_rate=RATE, mc_samples=50, seed=4)
    np.testing.assert_array_equal(
        grown.npv_distribution(HORIZON, **runs).samples,
        TIMED.npv_distribution(
            HORIZON, escalation={UnitPlan.NODE: 2e-4}, **runs
        ).samples,
    )


def test_its_own_money_is_checked_and_saved():
    plan = with_money(salvage=7.0, escalation=2e-4, retire_at=240.0)
    assert UnitPlan.from_spec(plan.spec()) == plan
    saved = plan.to_dict()
    assert {"salvage", "escalation", "retire_at"} <= set(saved["spec"])
    assert UnitPlan.from_dict(saved).to_dict() == saved
    loaded = rbd_from_json(rbd_to_json(plan.diagram()))
    assert loaded.plans[UnitPlan.NODE].to_dict() == saved
    for money, match in (
        (dict(salvage=math.inf), "salvage"),
        (dict(escalation="fast"), "escalation"),
        (dict(retire_at=0.0), "retire_at"),
    ):
        with pytest.raises((ValueError, TypeError), match=match):
            with_money(**money).diagram()
    with pytest.raises(NotImplementedError, match="function"):
        with_money(salvage=lambda age: age).to_dict()


# -- the plan's other valuations ----------------------------------------------


def test_cash_flows_and_costs_are_its_diagram_s():
    other = holding(TIMED)
    assert (
        TIMED.cash_flows(HORIZON, periods=3)
        == other.expected_npv(HORIZON, periods=3).cash_flows
    )
    assert TIMED.expected_cost(HORIZON).total == pytest.approx(
        other.expected_cost(HORIZON).total, rel=1e-12
    )
    mine = TIMED.optimal_replacement_interval()
    theirs = other.optimal_replacement_intervals()
    assert mine.intervals[UnitPlan.NODE] == pytest.approx(
        theirs.intervals[UnitPlan.NODE], rel=1e-12
    )


def test_a_plan_against_what_happened_is_its_diagram_s():
    log = Timeline.from_outages([(50.0, 52.0)], 100.0)
    plan = UnitPlan(W, REPAIR, prices=Prices(repair=5.0))
    mine = plan.plan_vs_actual(log, 100.0, HORIZON, discount_rate=RATE)
    theirs = holding(plan).plan_vs_actual(
        {UnitPlan.NODE: log}, 100.0, HORIZON, discount_rate=RATE
    )
    assert mine.actual_so_far == pytest.approx(theirs.actual_so_far)
    assert mine.expected_so_far == pytest.approx(theirs.expected_so_far)


def test_a_tested_unit_s_interval_is_chosen_alone():
    plan = UnitPlan(
        surv.Exponential.from_params([1e-3]),
        "instant",
        tasks=(Inspection(interval=100.0, cost=1.0),),
        prices=Prices(downtime=2.0),
    )
    mine = plan.optimal_inspection_interval()
    theirs = holding(plan).optimal_inspection_intervals()
    assert mine.intervals[UnitPlan.NODE] == pytest.approx(
        theirs.intervals[UnitPlan.NODE], rel=1e-12
    )


def test_the_exact_npv_is_the_mean_of_the_runs():
    # Bounds fixed before running: four standard errors of the runs' mean.
    options = dict(discount_rate=RATE, value_rate=1.0)
    exact = TIMED.expected_npv(2000.0, **options).npv
    runs = TIMED.npv_distribution(
        2000.0, mc_samples=4000, seed=20261121, **options
    )
    error = np.std(runs.samples, ddof=1) / math.sqrt(runs.samples.size)
    assert abs(runs.mean - exact) <= 4.0 * error
