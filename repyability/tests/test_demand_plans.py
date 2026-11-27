"""Demand plans for a probability of failing per demand (UC-73 of the
maintenance README): demands at set times (a spec's ``"demand_times"``),
a life that can end only at them, on the calendar, exact over a window
and simulated; and a rate that changes over time (``"demand_plan"``), an
exponential life in demands put on the diagram's clock as a duty plan
that changes over time is; either counted in the unit's operating time
with ``"demand_basis": "operating"``."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.maintenance import UnitPlan
from repyability.rbd import _compiled, _demands, _event_loop
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json
from repyability.tests.catalogue import E, W

FIXED = surv.FixedEventProbability.from_params
FIXED_TIME = surv.ExactEventTime.from_params
ONE = [("s", "g"), ("g", "t")]


def generator(**more):
    """UC-73's generator: started at 10, 20 and 30, failing to start with
    probability 0.1, a failed start repaired in exactly 15."""
    spec = {
        "reliability": FIXED(0.1),
        "demand_times": [10.0, 20.0, 30.0],
        "repairability": FIXED_TIME([15.0]),
        "repair_cost": 100.0,
        **more,
    }
    return RepairableRBD(ONE, {"g": spec})


# -- demands at set times: the hand-worked case ------------------------------


def test_the_failure_chances_by_hand():
    # f1 = p; at 20 day 10's repair (to 25) is under way; at 30 only day
    # 20's (to 35) is.
    rbd = generator()
    curve = _demands.DemandCurve(
        rbd.components["g"].reliability, rbd.components["g"].time_to_replace
    )
    np.testing.assert_allclose(
        curve.chances, [0.1, 0.09, 0.091], rtol=0.0, atol=1e-15
    )


def test_the_point_availability_by_hand():
    rbd = generator()
    x = np.array([5.0, 10.0, 22.0, 26.0, 31.0, 36.0, 46.0])
    expected = [1.0, 0.9, 1 - 0.19, 1 - 0.09, 1 - 0.181, 1 - 0.091, 1.0]
    np.testing.assert_allclose(
        rbd.point_availability(x), expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        rbd.point_unavailability(x),
        1.0 - np.array(expected),
        rtol=0.0,
        atol=1e-12,
    )


def test_the_window_by_hand():
    # Every repair is over by 45: down for 15 * 0.281 of the 50.
    rbd = generator()
    assert abs(rbd.expected_failures(50.0) - 0.281) < 1e-12
    assert abs(rbd.mission_availability(50.0) - (1 - 15 * 0.281 / 50)) < 1e-12
    # To 22: day 10's failure is down 12 of the 22, day 20's 2.
    assert (
        abs(rbd.mission_availability(22.0) - (1 - (0.1 * 12 + 0.09 * 2) / 22))
        < 1e-12
    )
    # Failures in [0, t): the demand at 20 is not before 20.
    assert abs(rbd.expected_failures(20.0) - 0.1) < 1e-12
    assert abs(rbd.expected_cost(50.0).total - 28.1) < 1e-10


def test_a_simulation_agrees_within_four_standard_errors():
    # Seed and size fixed before the run.
    rbd = generator()
    n = 4000
    result = rbd.availability(
        50.0, mc_samples=n, seed=20261127, control_variate=False
    )
    exact = 1 - 15 * 0.281 / 50
    se = result.mean_availability_interval().standard_error
    assert abs(result.mean_availability - exact) < 4 * se
    # At most 3 failures each, so a failure count's variance is at most
    # 3 times its mean: a bound on its standard error.
    failures = result.system_failures / n
    assert abs(failures - 0.281) < 4 * math.sqrt(3 * 0.281 / n)


def test_a_repair_that_is_no_fixed_time():
    # The chance a repair begun at t_j is under way at t_k is its sf.
    mu = 0.1
    rbd = generator(repairability=E([mu]))
    under = math.exp(-mu * 10.0), math.exp(-mu * 20.0)
    f1 = 0.1
    f2 = 0.1 * (1 - f1 * under[0])
    f3 = 0.1 * (1 - f1 * under[1] - f2 * under[0])
    assert rbd.expected_failures(40.0) == pytest.approx(
        f1 + f2 + f3, abs=1e-14
    )
    x = np.array([25.0])
    down = f1 * math.exp(-mu * 15.0) + f2 * math.exp(-mu * 5.0)
    np.testing.assert_allclose(
        rbd.point_availability(x), [1 - down], rtol=0.0, atol=1e-14
    )
    # The window integrates the exponential tails.
    t = 60.0
    downtime = sum(
        f * (1 - math.exp(-mu * (t - s))) / mu
        for f, s in zip((f1, f2, f3), (10.0, 20.0, 30.0))
    )
    assert rbd.mission_availability(t) == pytest.approx(
        1 - downtime / t, abs=1e-12
    )


def test_a_unit_repaired_at_a_demand_is_not_called_on_by_it():
    # Repaired in exactly 10: day 10's repair ends at 20, and the demand
    # at 20 does not call on it.
    rbd = generator(repairability=FIXED_TIME([10.0]))
    assert rbd.expected_failures(25.0) == pytest.approx(0.1 + 0.09, abs=1e-15)
    life = rbd.components["g"].reliability
    assert life.life_left(20.0, 0.0, 0.05) == 10.0


def test_the_life_from_a_uniform():
    life = _demands.DemandTimesLife(0.1, [10.0, 20.0, 30.0])
    # P(K > k) = 0.9^k: the first demand fails it when u <= 0.1.
    assert life.life_left(0.0, 0.0, 0.1) == 10.0
    assert life.life_left(0.0, 0.0, 0.1 + 1e-9) == 20.0
    assert life.life_left(0.0, 0.0, 0.19 + 1e-9) == 30.0
    assert life.life_left(0.0, 0.0, 0.999) == math.inf
    assert life.life_left(12.0, 0.0, 0.05) == 8.0
    assert life.life_left(30.0, 5.0, 0.05) == math.inf
    np.testing.assert_allclose(life.sf([5.0, 10.0, 31.0]), [1, 0.9, 0.729])


def test_the_long_run_is_refused_naming_the_window():
    rbd = generator()
    with pytest.raises(NotImplementedError, match="mission_availability"):
        rbd.mean_availability()
    routes = rbd.analysis_routes()
    assert routes["mean_availability"].route == "refused"
    assert routes["point_availability"].route == "numerical"
    assert routes["availability"].route == "simulated"
    assert "demand times" in str(routes["availability"])


def test_the_compiled_engine_refuses_it():
    rbd = generator()
    plan = _event_loop._stream_plan(rbd, 1.0, 0, False)[0]
    assert "demand times" in _compiled.unsupported(rbd, plan, None)


def test_a_task_makes_its_values_over_time_simulated():
    rbd = generator(preventive={"interval": 12.0})
    with pytest.raises(NotImplementedError, match="preventive task"):
        rbd.point_availability([1.0])
    result = rbd.availability(50.0, mc_samples=20, seed=1)
    assert 0.0 < result.mean_availability <= 1.0


def test_a_rule_s_times_are_demand_times():
    from repyability.schedule import CalendarContext, Recurrence, UsageForecast

    frame = UsageForecast.calendar("2027-01-01", "2028-01-01")
    quarterly = Recurrence.monthly(
        weekday="Monday", nth=1, every=3, anchor="2027-01-01"
    )
    times = CalendarContext(frame, unit="days").rule_times(quarterly)
    rbd = generator(demand_times=times)
    # Four quarterly starts, each repair over long before the next.
    assert times.size == 4
    assert rbd.expected_failures(365.0) == pytest.approx(0.4, abs=1e-15)


# -- equally spaced demands converge on a constant rate -----------------------


def test_equally_spaced_demands_converge_on_the_rate():
    # Demands every h, each failing with p = lambda h: as h shrinks, the
    # failures are a Poisson process of rate lambda, the demand rate's.
    lam, horizon = 0.02, 200.0
    repair = E([0.2])

    def diagram(**demands):
        return RepairableRBD(
            ONE,
            {
                "g": {
                    "reliability": FIXED(lam * demands.pop("h")),
                    "repairability": repair,
                    **demands,
                }
            },
        )

    errors = []
    for h in (4.0, 1.0, 0.25):
        spaced = diagram(
            h=h, demand_times=np.arange(1, int(horizon / h) + 1) * h
        )
        rate = diagram(h=h, demand_rate=1.0 / h)
        errors.append(
            (
                abs(
                    spaced.mission_availability(horizon)
                    - rate.mission_availability(horizon)
                ),
                abs(
                    spaced.expected_failures(horizon)
                    - rate.expected_failures(horizon)
                ),
            )
        )
    for coarse, fine in zip(errors, errors[1:]):
        # First order in h: a quarter of the spacing, under half the error.
        assert fine[0] < 0.5 * coarse[0]
        assert fine[1] < 0.5 * coarse[1]


# -- a rate that changes over time -------------------------------------------


PLAN = {"times": [0.0, 10.0, 20.0], "rates": [2.0, 0.5, 1.0]}


def pump(**more):
    spec = {
        "reliability": FIXED(0.05),
        "demand_plan": PLAN,
        "repairability": "instant",
        **more,
    }
    return RepairableRBD(ONE, {"g": spec})


def expected_demands(t):
    t = np.asarray(t, dtype=float)
    return (
        2.0 * np.minimum(t, 10.0)
        + 0.5 * np.clip(t - 10.0, 0.0, 10.0)
        + np.maximum(t - 20.0, 0.0)
    )


def test_a_plan_s_failures_are_p_times_the_expected_demands():
    # Repaired at once: its failures are a Poisson process at p lambda(t).
    rbd = pump()
    for t in (5.0, 15.0, 30.0):
        assert rbd.expected_failures(t) == pytest.approx(
            0.05 * expected_demands(t), rel=1e-6
        )


def test_a_plan_s_survival_against_a_simulation():
    # Never repaired within the window: up at t with exp(-p Lambda(t)).
    # Seed, size and points fixed before the run.
    rbd = pump(repairability=FIXED_TIME([1e6]))
    n = 4000
    result = rbd.availability(
        30.0, mc_samples=n, seed=836, control_variate=False, curve_points=6
    )
    exact = np.exp(-0.05 * expected_demands(result.timeline))
    se = np.sqrt(exact * (1 - exact) / n)
    assert np.all(np.abs(result.availability - exact) <= 4 * se + 1e-15)


def test_a_constant_plan_is_the_demand_rate():
    plan = pump(demand_plan={"times": [0.0], "rates": [0.5]})
    rate = RepairableRBD(
        ONE,
        {
            "g": {
                "reliability": FIXED(0.05),
                "demand_rate": 0.5,
                "repairability": "instant",
            }
        },
    )
    assert plan.mean_availability() == rate.mean_availability()
    assert plan.expected_failures(30.0) == rate.expected_failures(30.0)


def test_a_plan_with_repairs_that_take_time_is_simulated():
    rbd = pump(repairability=E([1.0]))
    with pytest.raises(NotImplementedError, match="demand plan"):
        rbd.point_availability([1.0])
    assert rbd.analysis_routes()["availability"].route == "simulated"


# -- the clock: a duty with demands in operating time ------------------------


def test_a_rate_in_operating_time_on_a_duty():
    on_duty = RepairableRBD(
        ONE,
        {
            "g": {
                "reliability": FIXED(0.1),
                "demand_rate": 2.0,
                "duty": 0.25,
                "demand_basis": "operating",
                "repairability": E([1.0]),
            }
        },
    )
    calendar = RepairableRBD(
        ONE,
        {
            "g": {
                "reliability": FIXED(0.1),
                "demand_rate": 0.5,
                "repairability": E([1.0]),
            }
        },
    )
    assert on_duty.mean_availability() == pytest.approx(
        calendar.mean_availability(), rel=1e-15
    )


def test_times_in_operating_time_on_a_duty():
    rbd = generator(
        demand_times=[5.0, 10.0, 15.0], duty=0.5, demand_basis="operating"
    )
    np.testing.assert_array_equal(
        rbd.components["g"].reliability.times, [10.0, 20.0, 30.0]
    )
    assert abs(rbd.expected_failures(50.0) - 0.281) < 1e-12


def test_a_plan_in_operating_time_on_a_duty_plan():
    # D: half the time to 4, then all of it; the plan's rate 2 to an
    # operating time of 10 (calendar 12), then 1.
    rbd = pump(
        demand_plan={"times": [0.0, 10.0], "rates": [2.0, 1.0]},
        duty={"times": [0.0, 4.0], "shares": [0.5, 1.0]},
        demand_basis="operating",
    )
    plan = rbd.components["g"].reliability.plan
    assert plan.times == (0.0, 4.0, 12.0)
    assert plan.shares == (1.0, 2.0, 1.0)
    assert plan.use == "demands"


@pytest.mark.parametrize(
    "more, message",
    [
        ({"duty": 0.5}, "demand_basis"),
        ({"demand_rate": 0.5}, "give one"),
        ({"demand_times": [0.0, 1.0]}, "above 0"),
        ({"demand_times": [1.0, np.inf]}, "above 0"),
        ({"demand_times": "1, 2"}, "list of numbers"),
        ({"demand_basis": "weekly"}, "demand_basis"),
        ({"standby": {"units": 2}}, "not taken together"),
        (
            {"preventive": {"interval": 5.0, "basis": "operating"}},
            "drop its basis",
        ),
        ({"reliability": W([100.0, 2.0])}, "two components in series"),
    ],
)
def test_refusals_say_what_to_do(more, message):
    with pytest.raises((ValueError, TypeError), match=message):
        generator(**more)


@pytest.mark.parametrize(
    "plan, message",
    [
        ([0.5], "dict of 'times' and 'rates'"),
        ({"times": [0.0]}, "dict of 'times' and 'rates'"),
        ({"times": [1.0], "rates": [0.5]}, "start at 0"),
        ({"times": [0.0], "rates": [-0.5]}, "at least 0"),
    ],
)
def test_a_plan_that_is_no_plan_is_refused(plan, message):
    with pytest.raises((ValueError, TypeError), match=message):
        pump(demand_plan=plan)


def test_a_basis_without_demands_is_refused():
    with pytest.raises(ValueError, match="has none"):
        RepairableRBD(
            ONE,
            {
                "g": {
                    "reliability": E([0.1]),
                    "demand_basis": "operating",
                    "repairability": E([1.0]),
                }
            },
        )


# -- what is kept: saving, plans, levers, other nodes' runs -------------------


@pytest.mark.parametrize("build", [generator, pump])
def test_saving_keeps_the_demands(build):
    rbd = build()
    again = rbd_from_json(rbd_to_json(rbd))
    assert json.loads(rbd_to_json(again)) == json.loads(rbd_to_json(rbd))
    assert again.expected_failures(30.0) == rbd.expected_failures(30.0)


def test_a_unit_plan_holds_the_demands():
    rbd = generator(duty=0.5, demand_basis="operating")
    plan = rbd.plans["g"]
    assert plan.demand_times == [10.0, 20.0, 30.0]
    assert plan.demand_basis == "operating"
    assert UnitPlan.from_spec(plan.spec()) == plan
    assert pump().plans["g"].demand_plan == PLAN


def test_the_lever_is_the_probability():
    rbd = generator()
    levers = {lever.name: lever for lever in rbd.levers()}
    assert levers["reliability.p"].value == 0.1
    assert levers["reliability.p"].bounds == (0.0, 1.0)
    halved = rbd.with_levers({("g", "reliability.p"): 0.05})
    assert halved.expected_failures(50.0) == pytest.approx(
        generator(reliability=FIXED(0.05)).expected_failures(50.0), abs=0
    )


def test_a_probability_of_0_never_fails():
    rbd = RepairableRBD(
        ONE + [("s", "b"), ("b", "t")],
        {
            "g": {
                "reliability": FIXED(0.0),
                "demand_times": [1.0],
                "repairability": E([1.0]),
            },
            "b": {"reliability": W([100.0, 2.0]), "repairability": E([1.0])},
        },
    )
    assert "g" not in rbd.components


def test_seeded_runs_of_other_nodes_are_unchanged():
    # Each node draws from its own streams: a demand plan on one leaves
    # the others' runs as they were.
    def diagram(g):
        return RepairableRBD(
            [("s", "a"), ("a", "t"), ("s", "g"), ("g", "t")],
            {
                "a": {
                    "reliability": W([30.0, 2.0]),
                    "repairability": E([0.5]),
                },
                "g": g,
            },
        )

    other = {"reliability": E([0.05]), "repairability": E([0.5])}
    times = {
        "reliability": FIXED(0.2),
        "demand_times": [5.0, 15.0, 40.0],
        "repairability": E([0.5]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = [
            diagram(g).availability(
                60.0, mc_samples=200, seed=7, control_variate=False
            )
            for g in (other, times)
        ]
    assert runs[0].node_downtime["a"] == runs[1].node_downtime["a"]
    assert runs[0].node_uptime["a"] == runs[1].node_uptime["a"]


def test_a_unit_up_at_an_age_is_as_new():
    # It does not age: up at 0, it meets the same demands as a new one.
    from repyability.rbd.node_state import NodeState

    rbd = generator()
    x = np.array([22.0, 31.0])
    np.testing.assert_array_equal(
        rbd.point_availability(x, state={"g": NodeState(age=7.0)}),
        rbd.point_availability(x),
    )
    with pytest.raises(NotImplementedError, match="simulate it"):
        rbd.point_availability(
            x, state={"g": NodeState(alive=False, down_for=1.0)}
        )
    result = rbd.availability(
        50.0, mc_samples=50, seed=2, state={"g": NodeState(age=7.0)}
    )
    assert 0.0 < result.sample_mean_availability <= 1.0
