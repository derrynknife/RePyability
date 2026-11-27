"""A life that wears out with demands at set times, or on a plan of
rates that changes, beside it (a spec's ``"demand"`` with ``"times"`` or
``"plan"``): the demands compete with the life and are renewed with it,
but they are the calendar's, so a unit put into service at ``t0`` at age
``a`` survives ``s`` more with chance ``exp(-G)``, ``G`` the life's ``H(a +
s) - H(a)`` plus ``-log(1 - p)`` for each demand in ``(t0, t0 + s]`` (or
``p`` times the plan's expected demands then)
(``_demands.LifeWithPlannedDemands``). Its values over a window are a
renewal equation on the clock (``_demand_renewal``); the simulations
take it too, and its long run and mean life are refused."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import RepairableRBD
from repyability.maintenance import UnitPlan
from repyability.rbd._demands import (
    DemandTimesLife,
    LifeWithDemands,
    LifeWithPlannedDemands,
)
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json
from repyability.tests.catalogue import E, W

FIXED = surv.FixedEventProbability.from_params
ONE = [("s", "a"), ("a", "t")]
P = 0.1
TIMES = [50.0, 120.0, 200.0, 260.0]
PLAN = {"times": [0.0, 100.0], "rates": [0.05, 0.2]}
LIFE = W([150.0, 2.5])
STEP = -math.log1p(-P)


def spec(demand, life=LIFE, **more):
    return {
        "reliability": life,
        "repairability": E([0.5]),
        "demand": {"probability": P, **demand},
        "repair_cost": 5.0,
        **more,
    }


def built(demand, **more):
    return RepairableRBD(ONE, {"a": spec(demand, **more)})


def exposure(t0, age, s, times=TIMES, p=P, life=LIFE):
    """``G(s)`` written out: the life's hazard from ``age`` over ``s``,
    and ``-log(1 - p)`` for each demand in ``(t0, t0 + s]``."""
    ran = float(life.Hf(age + s) - life.Hf(age)) if s > 0 else 0.0
    met = sum(1 for t in times if t0 < t <= t0 + s)
    return ran - met * math.log1p(-p)


# -- from new ------------------------------------------------------------


def test_from_new_at_set_times_it_is_the_life_and_the_demands_in_series():
    # The life and the set-times demand component as two components in
    # series: R(t) (1 - p)^N(t).
    both = built({"times": TIMES}).components["a"].reliability
    assert isinstance(both, LifeWithPlannedDemands)
    demands = (
        RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": FIXED(P),
                    "repairability": E([0.5]),
                    "demand_times": TIMES,
                }
            },
        )
        .components["a"]
        .reliability
    )
    assert isinstance(demands, DemandTimesLife)
    x = np.array([0.0, 1e-3, 49.9, 50.0, 50.1, 119.0, 120.0, 250.0, 400.0])
    np.testing.assert_allclose(
        both.sf(x), LIFE.sf(x) * demands.sf(x), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        both.ff(x),
        -np.expm1(-(LIFE.Hf(x) + demands.Hf(x))),
        rtol=1e-12,
        atol=0.0,
    )


def test_from_new_on_a_plan_it_is_the_life_and_the_demands_in_series():
    # The demand plan's component is an exponential life in demands on
    # the plan's clock: exp(-p Λ(t)).
    both = built({"plan": PLAN}).components["a"].reliability
    demands = (
        RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": FIXED(P),
                    "repairability": E([0.5]),
                    "demand_plan": PLAN,
                }
            },
        )
        .components["a"]
        .reliability
    )
    x = np.array([0.0, 1.0, 99.0, 100.0, 101.0, 250.0, 600.0])
    np.testing.assert_allclose(
        both.sf(x), LIFE.sf(x) * demands.sf(x), rtol=1e-12, atol=0.0
    )
    expected = np.where(x <= 100.0, 0.05 * x, 5.0 + 0.2 * (x - 100.0))
    np.testing.assert_allclose(
        both.Hf(x), LIFE.Hf(x) + P * expected, rtol=1e-12, atol=0.0
    )


# -- the life left from a start time and age -----------------------------


def test_the_life_left_by_hand():
    # Put back into service at 55 (after a repair), new or at age 30,
    # with demands at 120, 200 and 260 still to come.
    life = built({"times": TIMES}).components["a"].reliability
    h = float(LIFE.Hf(65.0))
    # Less exposure than the life's to 120 and the demand there: the
    # life alone would not get there, the demand at 120 fails it.
    u = math.exp(-(h + 0.5 * STEP))
    assert life.life_left(55.0, 0.0, u) == 65.0
    # The demand at 120 survived, the life ends before 200: H(x) =
    # -log u - step, x = alpha (-log u - step)^(1 / beta).
    target = h + STEP + 0.2
    x = 150.0 * (target - STEP) ** (1.0 / 2.5)
    assert 65.0 < x < 145.0
    assert life.life_left(55.0, 0.0, math.exp(-target)) == pytest.approx(
        x, rel=1e-12
    )
    # At age 30: H(30 + x) - H(30) = -log u - step.
    h30 = float(LIFE.Hf(30.0))
    x = 150.0 * (h30 + target - STEP) ** (1.0 / 2.5) - 30.0
    assert 65.0 < x < 145.0
    assert life.life_left(55.0, 30.0, math.exp(-target)) == pytest.approx(
        x, rel=1e-12
    )
    # Past the last demand, the life alone.
    assert life.life_left(300.0, 0.0, 0.5) == pytest.approx(
        150.0 * math.log(2.0) ** (1.0 / 2.5), rel=1e-12
    )


@pytest.mark.parametrize("t0", [0.0, 55.0, 130.0])
@pytest.mark.parametrize("age", [0.0, 30.0])
def test_the_life_left_inverts_the_summed_hazard(t0, age):
    # Each draw is where G first reaches -log u (a demand's step may take
    # it past), and the draws' survival is exp(-G) from each start.
    life = built({"times": TIMES}).components["a"].reliability
    n = 4000
    u = np.random.default_rng(848).uniform(size=n)
    drawn = np.array([life.life_left(t0, age, v) for v in u])
    for x, v in zip(drawn, u):
        target = -math.log(v)
        reached = exposure(t0, age, x)
        assert reached >= target * (1.0 - 1e-9)
        if any(math.isclose(t0 + x, t) for t in TIMES):
            # At a demand: short of it just before.
            assert exposure(t0, age, x * (1.0 - 1e-12)) < target
        else:
            assert reached == pytest.approx(target, rel=1e-9)
    for s in (10.0, 70.0, 150.0, 250.0):
        chance = math.exp(-exposure(t0, age, s))
        share = float(np.mean(drawn > s))
        error = math.sqrt(chance * (1.0 - chance) / n)
        assert abs(share - chance) <= 4.0 * error + 1.0 / n


def test_on_a_plan_the_life_left_is_the_rate_s_within_a_piece():
    # Within the first piece the plan's demands come at one rate: the
    # draw of a life with that rate beside it.
    life = built({"plan": PLAN}).components["a"].reliability
    rate = LifeWithDemands(LIFE, 0.05, P)
    for u in (0.99, 0.9, 0.8):
        x = life.life_left(10.0, 20.0, u)
        assert 10.0 + x < 100.0
        assert x == pytest.approx(rate.aged(20.0, u), rel=1e-12)
    # Past the plan's change, the hazard written out: the life's, and p
    # times the demands at 0.05 to 100 and 0.2 after.
    u = 0.05
    x = life.life_left(10.0, 20.0, u)
    assert 10.0 + x > 100.0
    g = (
        float(LIFE.Hf(20.0 + x) - LIFE.Hf(20.0))
        + P * 0.05 * 90.0
        + P * 0.2 * (10.0 + x - 100.0)
    )
    assert g == pytest.approx(-math.log(u), rel=1e-9)


# -- with repair ------------------------------------------------------------

#: A unit that wears fast (Weibull(40, 1.5)), called on once at 50 with a
#: chance of 0.3 of failing then, repaired in exactly 35, over 68: at
#: most two failures, the second only after a failure of the life before
#: 33, by a new unit that meets the demand if it is back before 50.
HAND = dict(alpha=40.0, beta=1.5, p=0.3, tau=50.0, r=35.0, horizon=68.0)


def hand_worked_downtime() -> float:
    alpha, beta, p, tau, r, horizon = (
        HAND[k] for k in ("alpha", "beta", "p", "tau", "r", "horizon")
    )

    def reliability(t):
        return math.exp(-((t / alpha) ** beta)) if t > 0.0 else 1.0

    def density(t):
        return (beta / alpha) * (t / alpha) ** (beta - 1.0) * reliability(t)

    def first(t):
        # The first unit's survival from new to t, the demand at tau met.
        if t < 0.0:
            return 1.0
        return reliability(t) * ((1.0 - p) if t >= tau else 1.0)

    # The first failure's downtime within the horizon: down at t if it
    # failed in (t - r, t].
    down = integrate.quad(
        lambda t: first(t - r) - first(t),
        0.0,
        horizon,
        points=[tau, r],
        epsabs=1e-13,
        epsrel=1e-12,
        limit=200,
    )[0]

    def second(x):
        # After the life fails at x, a new unit from x + r, which meets
        # the demand if it is back before it; down from its failure to
        # the horizon (its repair ends past it).
        back = x + r

        def survival(y):
            meets = back < tau and back + y >= tau
            return reliability(y) * ((1.0 - p) if meets else 1.0)

        points = [tau - back] if back < tau else None
        return integrate.quad(
            lambda t: 1.0 - survival(t),
            0.0,
            horizon - back,
            points=points,
            epsabs=1e-13,
            epsrel=1e-12,
            limit=200,
        )[0]

    down += integrate.quad(
        lambda x: density(x) * second(x),
        0.0,
        horizon - r,
        points=[tau - r],
        epsabs=1e-13,
        epsrel=1e-12,
        limit=200,
    )[0]
    return down


def test_a_hand_worked_case_with_repair():
    # Seeded simulation (design fixed before it was run: 4000 runs, seed
    # 848) within 4 standard errors of the downtime worked by hand.
    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": W([HAND["alpha"], HAND["beta"]]),
                "repairability": surv.ExactEventTime.from_params([HAND["r"]]),
                "demand": {"probability": HAND["p"], "times": [HAND["tau"]]},
            }
        },
    )
    horizon = HAND["horizon"]
    exact = 1.0 - hand_worked_downtime() / horizon
    n = 4000
    run = rbd.availability(
        horizon, mc_samples=n, seed=848, control_variate=False
    )
    spread = float(np.std(run.uptimes / horizon)) / math.sqrt(n)
    assert abs(float(run.mean_availability) - exact) < 4.0 * spread
    # The two modes renewed apart (two components in series) differ: a
    # demand there meets a unit whose life has failed.
    series = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {
                "reliability": W([HAND["alpha"], HAND["beta"]]),
                "repairability": surv.ExactEventTime.from_params([HAND["r"]]),
            },
            "b": {
                "reliability": FIXED(HAND["p"]),
                "repairability": surv.ExactEventTime.from_params([HAND["r"]]),
                "demand_times": [HAND["tau"]],
            },
        },
    )
    assert abs(series.mission_availability(horizon) - exact) > 1e-3


def test_a_plan_of_one_rate_is_the_rate():
    # Bit for bit: a plan of one rate is that rate.
    by_plan = built({"plan": {"times": [0.0, 100.0], "rates": [0.05, 0.05]}})
    by_rate = built({"rate": 0.05})
    assert isinstance(by_plan.components["a"].reliability, LifeWithDemands)
    run = dict(mc_samples=200, seed=5, control_variate=False)
    assert (
        by_plan.availability(300.0, **run).mean_availability
        == by_rate.availability(300.0, **run).mean_availability
    )
    assert by_plan.mean_availability() == by_rate.mean_availability()


def test_a_plan_that_changes_only_past_the_horizon_is_the_rate():
    # Simulated on the diagram's clock (4000 runs, seed 849, fixed
    # first), within 4 standard errors of the rate's exact value.
    by_plan = built({"plan": {"times": [0.0, 1000.0], "rates": [0.05, 0.5]}})
    assert isinstance(
        by_plan.components["a"].reliability, LifeWithPlannedDemands
    )
    by_rate = built({"rate": 0.05})
    n = 4000
    run = by_plan.availability(
        400.0, mc_samples=n, seed=849, control_variate=False
    )
    exact = by_rate.mission_availability(400.0)
    spread = float(np.std(run.uptimes / 400.0)) / math.sqrt(n)
    assert abs(float(run.mean_availability) - exact) < 4.0 * spread


def test_set_times_close_in_on_the_constant_rate():
    # Demands every h, each failing with p = c h: as h shrinks, the
    # survival and the draws close in on the rate c's (#835). Between
    # demands the steps lag the rate's hazard by at most c h, and at a
    # demand -log(1 - c h) / h exceeds c by about c^2 h / 2, so the
    # survival is within c h (1 + c t) of the rate's at every time; at
    # the demands themselves the gap is that second term alone.
    c, horizon = 0.005, 600.0
    rate = LifeWithDemands(LIFE, c, 1.0)
    x = np.linspace(0.0, horizon, 3001)
    at_demands, draws = [], []
    for h in (10.0, 1.0, 0.1):
        times = np.arange(1, round(horizon / h) + 1) * h
        life = LifeWithPlannedDemands(LIFE, c * h, times=times)
        gap = np.abs(life.sf(x) - rate.sf(x))
        assert np.all(gap <= c * h * (1.0 + c * x) + 1e-15)
        at_demands.append(
            float(np.max(np.abs(life.sf(times) - rate.sf(times))))
        )
        draws.append(
            max(
                abs(life.life_left(0.0, 0.0, u) - rate.aged(0.0, u))
                for u in (0.9, 0.5, 0.2)
            )
        )
    assert at_demands[1] < 0.2 * at_demands[0]
    assert at_demands[2] < 0.2 * at_demands[1]
    assert draws[1] < 0.2 * draws[0] and draws[2] < 0.2 * draws[1]
    assert draws[2] < 0.1


# -- routes, saving, plans, levers, units, refusals -------------------------


@pytest.mark.parametrize("demand", [{"times": TIMES}, {"plan": PLAN}])
def test_its_window_is_numerical_and_its_long_run_refused(demand):
    rbd = built(demand)
    routes = rbd.analysis_routes()
    assert routes["availability"].route == "simulated"
    assert routes["availability"].engine == "python"
    assert "demands beside the life" in routes["availability"].engine_reason
    for method in (
        "point_availability",
        "mission_availability",
        "expected_failures",
        "expected_cost",
    ):
        assert routes[method].route == "numerical"
        assert "renewal equation" in routes[method].reason
    for method in ("mean_availability", "mean_up_time"):
        assert routes[method].route == "refused"
        assert "wears out and is called on" in routes[method].reason
        assert "values over a window are numerical" in routes[method].reason
    with pytest.raises(NotImplementedError, match="no reliability, mean"):
        rbd.mean_availability()
    assert 0.0 < rbd.mission_availability(300.0) < 1.0


@pytest.mark.parametrize("demand", [{"times": TIMES}, {"plan": PLAN}])
def test_saving_keeps_the_demands(demand):
    rbd = built(demand)
    text = rbd_to_json(rbd)
    again = rbd_from_json(text)
    saved = json.loads(text)
    assert "demand" in json.dumps(saved)
    kept = again._init_args["components"]["a"]["demand"]
    assert kept["probability"] == P
    if "times" in demand:
        assert kept["times"] == TIMES
    else:
        assert kept["plan"]["times"] == PLAN["times"]
        assert kept["plan"]["rates"] == PLAN["rates"]
    run = dict(mc_samples=100, seed=3, control_variate=False)
    assert (
        again.availability(300.0, **run).mean_availability
        == rbd.availability(300.0, **run).mean_availability
    )


@pytest.mark.parametrize("demand", [{"times": TIMES}, {"plan": PLAN}])
def test_a_unit_plan_holds_the_demands(demand):
    rbd = built(demand)
    plan = rbd.plans["a"]
    assert plan.demand == {"probability": P, **demand}
    assert UnitPlan.from_spec(plan.spec()) == plan
    alone = UnitPlan(LIFE, E([0.5]), demand={"probability": P, **demand})
    assert alone.spec()["demand"] == {"probability": P, **demand}
    assert isinstance(
        RepairableRBD(ONE, {"a": alone}).components["a"].reliability,
        LifeWithPlannedDemands,
    )


@pytest.mark.parametrize("demand", [{"times": TIMES}, {"plan": PLAN}])
def test_the_lever_is_the_probability(demand):
    rbd = built(demand)
    levers = {lever.name: lever for lever in rbd.levers() if lever.key == "a"}
    assert levers["demand.probability"].value == P
    assert levers["demand.probability"].bounds == (0.0, 1.0)
    assert "demand.rate" not in levers
    moved = rbd.with_levers({("a", "demand.probability"): 0.3})
    assert moved.components["a"].reliability.p == 0.3
    with pytest.raises(ValueError):
        rbd.with_levers({("a", "demand.probability"): 1.5})


@pytest.mark.parametrize("c", [1e-3, 1e3])
def test_it_scales_with_the_unit(c):
    # Times multiplied by c: the life's scale, the repairs and the
    # demands' times by c (a plan's rates divided by c).
    def rbd(c, demand):
        return RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": W([150.0 * c, 2.5]),
                    "repairability": E([0.5 / c]),
                    "demand": {"probability": P, **demand},
                }
            },
        )

    run = dict(mc_samples=200, seed=9, control_variate=False)
    for demand, scaled in (
        ({"times": TIMES}, {"times": [t * c for t in TIMES]}),
        (
            {"plan": PLAN},
            {
                "plan": {
                    "times": [t * c for t in PLAN["times"]],
                    "rates": [r / c for r in PLAN["rates"]],
                }
            },
        ),
    ):
        one = rbd(1.0, demand).availability(300.0, **run)
        other = rbd(c, scaled).availability(300.0 * c, **run)
        assert float(other.mean_availability) == pytest.approx(
            float(one.mean_availability), rel=1e-9
        )


@pytest.mark.parametrize(
    "demand, more, match",
    [
        ({"rate": 0.05, "times": TIMES}, {}, "give one"),
        ({"times": TIMES, "plan": PLAN}, {}, "give one"),
        ({"times": [0.0, 10.0]}, {}, "its demand's times must be"),
        ({"times": "often"}, {}, "its demand's times must be"),
        ({"plan": [0.1, 0.2]}, {}, "a demand_plan is"),
        ({"times": TIMES}, {"duty": 0.5}, "drop the duty"),
        (
            {"plan": PLAN},
            {"duty": 0.5, "demand_basis": "operating"},
            "drop the duty",
        ),
        ({"times": TIMES}, {"demand_times": TIMES}, "give one"),
        (
            {"times": TIMES},
            {"standby": {"units": 2}},
            "component of its own in series",
        ),
    ],
)
def test_what_does_not_go_with_it_is_refused(demand, more, match):
    with pytest.raises((TypeError, ValueError), match=match):
        built(demand, **more)


def test_a_probability_of_0_leaves_the_life():
    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": LIFE,
                "repairability": E([0.5]),
                "demand": {"probability": 0.0, "times": TIMES},
            }
        },
    )
    assert rbd.components["a"].reliability is LIFE


def test_every_demand_fails_it_with_a_probability_of_1():
    life = LifeWithPlannedDemands(LIFE, 1.0, times=TIMES)
    np.testing.assert_array_equal(
        life.sf(np.array([10.0, 50.0, 60.0])),
        [float(LIFE.sf(10.0)), 0.0, 0.0],
    )
    # A unit put in at 55 that the life does not end first fails at 120
    # (u = 0.5: the life alone would last 150 log(2)^0.4, past 65).
    assert life.life_left(55.0, 0.0, 0.5) == 65.0


# -- values over a window, without simulating ------------------------------
#
# ``_demand_renewal`` solves the renewal equation on a grid whose cells hold
# no demand, each cell's failures and renewals kept by their mass and mean.
# Its error is second order in the step: about ``(h / s)^2 / 8`` of the
# chance moved by each failure and renewal, ``h`` the largest step and
# ``s`` the life's typical up time (``_curves._up_scale``), with a first
# order term local to the cells a fixed repair ends in. Over a window of a
# few up times the bounds below take ``4 (h / s)^2``.


def grid_bound(curve) -> float:
    """``4 (h / s)^2`` for a ``PlannedDemandsCurve`` (see above)."""
    from repyability.rbd._curves import _up_scale

    h = float(np.max(np.diff(curve.t)))
    return 4.0 * (h / _up_scale(curve.life.life, None)) ** 2


def curve_of(rbd, horizon, node="a"):
    from repyability.rbd import _demands

    return _demands.planned_demands_curve(rbd, node, horizon)


@pytest.mark.parametrize(
    "repair",
    [E([0.05]), surv.ExactEventTime.from_params([70.0])],
    ids=["exponential repair", "fixed repair"],
)
def test_a_life_that_never_fails_is_the_set_times_component(repair):
    # A life of rate 1e-300 (survival 1.0 to the bit): the demands alone,
    # whose values are exact (DemandCurve), to 1e-12.
    times = [50.0, 120.0, 120.0, 200.0, 260.0]
    both = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": E([1e-300]),
                "repairability": repair,
                "demand": {"probability": 0.3, "times": times},
                "repair_cost": 5.0,
            }
        },
    )
    assert isinstance(both.components["a"].reliability, LifeWithPlannedDemands)
    alone = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": FIXED(0.3),
                "repairability": repair,
                "demand_times": times,
                "repair_cost": 5.0,
            }
        },
    )
    x = np.linspace(0.0, 300.0, 601)
    np.testing.assert_allclose(
        both.point_availability(x),
        alone.point_availability(x),
        rtol=0.0,
        atol=1e-12,
    )
    for method in ("mission_availability", "expected_failures"):
        assert getattr(both, method)(300.0) == pytest.approx(
            getattr(alone, method)(300.0), rel=0.0, abs=1e-12
        )
    assert both.expected_cost(300.0).mean == pytest.approx(
        alone.expected_cost(300.0).mean, rel=1e-12
    )


@pytest.mark.parametrize("repair", [E([0.5]), E([0.02])])
def test_a_plan_of_one_rate_is_the_rate_s_route(repair):
    # Built on a plan of two equal rates (which a spec turns into the rate
    # itself), the curve equals the rate's own (LifeWithDemands, the
    # renewal equation of a plain unit, about 4e-7 off) within its grid's
    # bound.
    from repyability.rbd._demand_renewal import curve
    from repyability.rbd._demands import USE
    from repyability.schedule import DutySchedule

    plan = DutySchedule("hours", (0.0, 100.0), (0.05, 0.05), USE)
    life = LifeWithPlannedDemands(LIFE, P, plan=plan)
    by_plan = curve(life, repair, 400.0)
    by_rate = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": LIFE,
                "repairability": repair,
                "demand": {"probability": P, "rate": 0.05},
            }
        },
    )
    x = np.linspace(1.0, 399.0, 81)
    bound = grid_bound(by_plan) + 1e-6
    np.testing.assert_allclose(
        by_plan.at(x), by_rate.point_availability(x), rtol=0.0, atol=bound
    )
    failures = by_rate.expected_failures(x[-1])
    assert float(by_plan.events(x[-1:])["failures"][0]) == pytest.approx(
        failures, rel=0.0, abs=bound * failures + 1e-6
    )


def hand_worked_rbd():
    return RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": W([HAND["alpha"], HAND["beta"]]),
                "repairability": surv.ExactEventTime.from_params([HAND["r"]]),
                "demand": {"probability": HAND["p"], "times": [HAND["tau"]]},
            }
        },
    )


def test_the_hand_worked_case_without_simulating():
    # The downtime worked by hand (two nested integrals), within the
    # grid's bound; and the seeded simulation (4000 runs, seed 848, as
    # above) within 4 standard errors of it.
    rbd = hand_worked_rbd()
    horizon = HAND["horizon"]
    exact = 1.0 - hand_worked_downtime() / horizon
    bound = grid_bound(curve_of(rbd, horizon))
    assert rbd.analysis_routes()["mission_availability"].route == "numerical"
    numerical = rbd.mission_availability(horizon)
    assert abs(numerical - exact) <= bound
    n = 4000
    run = rbd.availability(
        horizon, mc_samples=n, seed=848, control_variate=False
    )
    spread = float(np.std(run.uptimes / horizon)) / math.sqrt(n)
    assert abs(float(run.mean_availability) - numerical) < 4.0 * spread


@pytest.mark.parametrize(
    "demand, seed",
    [
        ({"times": [30.0, 60.0, 90.0, 90.0, 150.0, 210.0, 270.0]}, 851),
        ({"plan": {"times": [0.0, 100.0], "rates": [0.02, 0.2]}}, 852),
    ],
    ids=["set times", "plan"],
)
def test_the_values_over_a_window_agree_with_the_simulation(demand, seed):
    # Several demands (two at one time), or a plan whose rate rises, with
    # repairs that take time and a wearing life: the mission availability
    # and the failures against 4000 seeded simulations (design fixed
    # first), within 4 standard errors; a repair costing 1, the cost is
    # the count of failures.
    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": W([80.0, 2.0]),
                "repairability": E([0.1]),
                "demand": {"probability": 0.3, **demand},
                "repair_cost": 1.0,
            }
        },
    )
    horizon, n = 300.0, 4000
    run = rbd.availability(
        horizon, mc_samples=n, seed=seed, control_variate=False
    )
    spread = float(np.std(run.uptimes / horizon)) / math.sqrt(n)
    numerical = rbd.mission_availability(horizon)
    assert abs(float(run.mean_availability) - numerical) < 4.0 * spread
    counts = np.asarray(run.cost.samples, dtype=float)
    spread = float(np.std(counts)) / math.sqrt(n)
    expected = rbd.expected_failures(horizon)
    assert abs(float(np.mean(counts)) - expected) < 4.0 * spread


def test_from_a_unit_up_at_an_age():
    # Up at age 60 at 0 (its life conditioned on that), against 4000
    # seeded simulations from the same state (seed 853), within 4
    # standard errors; and refused from a unit down, or long in service.
    from repyability import NodeState

    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": W([80.0, 2.0]),
                "repairability": E([0.1]),
                "demand": {"probability": 0.3, "times": [40.0, 80.0]},
            }
        },
    )
    state = {"a": NodeState(age=60.0)}
    horizon, n = 120.0, 4000
    numerical = rbd.mission_availability(horizon, state=state)
    assert numerical < rbd.mission_availability(horizon)
    run = rbd.availability(
        horizon, mc_samples=n, seed=853, control_variate=False, state=state
    )
    spread = float(np.std(run.uptimes / horizon)) / math.sqrt(n)
    assert abs(float(run.mean_availability) - numerical) < 4.0 * spread
    with pytest.raises(NotImplementedError, match="simulate it"):
        rbd.point_availability(
            [10.0], state={"a": NodeState(alive=False, down_for=1.0)}
        )
    with pytest.raises(NotImplementedError, match="long-run state"):
        rbd.point_availability([10.0], state="stationary")


def test_an_imperfect_repair_is_refused_over_a_window():
    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                **spec({"times": TIMES}),
                "repair": {"model": "kijima1", "q": 0.5},
            }
        },
    )
    route = rbd.analysis_routes()["point_availability"]
    assert route.route == "refused"
    assert "repaired imperfectly" in route.reason
