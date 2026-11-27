"""A life that wears out with a demand mode beside it (a spec's
``"demand"``): demands a Poisson process of rate ``d``, each failing a
working unit with probability ``p``, compete with the life and are renewed
with it, so the unit's survival is ``R(t) exp(-d p t)``
(``_demands.LifeWithDemands``). With an exponential life of rate ``λ`` it
is ``Exponential(λ + d p)``, on every method; from new, it is the life and
the demand mode as two components in series."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import NonRepairableRBD, RepairableRBD
from repyability.maintenance import UnitPlan
from repyability.rbd._demands import LifeWithDemands
from repyability.rbd._event_loop import _aged_life
from repyability.rbd.serialisation import (
    deserialise_model,
    rbd_from_json,
    rbd_to_json,
    serialise_model,
)
from repyability.tests.catalogue import EDGES, E, W
from repyability.tests.test_analysis_routes import REPAIRABLE_CALLS
from repyability.tests.test_demand_rate import _BY_PARAMETERS
from repyability.tests.test_simulation_engines import identical

FIXED = surv.FixedEventProbability.from_params
ONE = [("s", "a"), ("a", "t")]
P, D = 0.1, 0.05
LAMBDA = 0.004

LIVES = {
    "weibull": W([150.0, 2.5]),
    "lognormal": surv.LogNormal.from_params([4.0, 0.8]),
    "offset weibull": surv.Weibull.from_params([100.0, 1.5], gamma=20.0),
    "some never fail": surv.Weibull.from_params([100.0, 3.0], lfp_p=0.6),
}


def spec(life, p=P, d=D, **more):
    return {
        "reliability": life,
        "repairability": E([0.5]),
        "demand": {"probability": p, "rate": d},
        "repair_cost": 5.0,
        **more,
    }


@pytest.mark.parametrize("name", sorted(LIVES))
def test_its_functions_are_the_life_s_with_the_demands_hazard(name):
    life = LIVES[name]
    both = LifeWithDemands(life, D, P)
    c = D * P
    x = np.array([0.0, 1e-6, 0.5, 10.0, 50.0, 150.0, 400.0, 2000.0])
    np.testing.assert_allclose(
        both.sf(x), life.sf(x) * np.exp(-c * x), rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        both.Hf(x), life.Hf(x) + c * x, rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(both.hf(x), life.hf(x) + c, rtol=1e-15)
    np.testing.assert_allclose(
        both.df(x), (life.hf(x) + c) * both.sf(x), rtol=1e-15, atol=0.0
    )
    # ff keeps its digits where it is small.
    np.testing.assert_allclose(
        both.ff(x), -np.expm1(-(life.Hf(x) + c * x)), rtol=1e-15, atol=0.0
    )
    assert 0.0 < both.ff(np.array([1e-6]))[0] < 1e-6


@pytest.mark.parametrize("name", sorted(LIVES))
def test_its_quantile_inverts_its_distribution(name):
    both = LifeWithDemands(LIVES[name], D, P)
    q = np.array([1e-12, 1e-6, 0.01, 0.3, 0.5, 0.9, 0.999, 1.0 - 1e-9])
    x = both.qf(q)
    assert np.all(np.diff(x) > 0.0)
    # The cumulative hazard is the least x reaching -log(1 - q).
    np.testing.assert_allclose(both.Hf(x), -np.log1p(-q), rtol=1e-12)
    assert both.qf(np.array([0.0]))[0] == 0.0
    assert both.qf(np.array([1.0]))[0] == np.inf


def test_its_draws_follow_its_distribution():
    both = LifeWithDemands(LIVES["weibull"], D, P)
    np.random.seed(20261127)
    draws = np.sort(both.random(20_000))
    empirical = np.arange(1, draws.size + 1) / draws.size
    assert np.max(np.abs(both.ff(draws) - empirical)) < 0.012


@pytest.mark.parametrize("age", [0.0, 30.0, 200.0])
def test_a_draw_at_a_virtual_age_inverts_the_hazard_left(age):
    both = LifeWithDemands(LIVES["weibull"], D, P)
    for u in (1e-9, 0.3, 0.9, 1.0 - 1e-12):
        x = _aged_life(both, age, u)
        reached = both.Hf(np.array([age + x]))[0] - both.Hf(np.array([age]))[0]
        assert reached == pytest.approx(-math.log(u), rel=1e-11)


@pytest.mark.parametrize("name", sorted(LIVES))
def test_its_mean_is_the_integral_of_its_survival(name):
    both = LifeWithDemands(LIVES[name], D, P)
    expected, _ = integrate.quad(
        lambda t: float(both.sf(np.array([t]))[0]),
        0.0,
        np.inf,
        epsabs=0.0,
        epsrel=1e-12,
        limit=1000,
    )
    assert both.mean() == pytest.approx(expected, rel=1e-8)


def test_it_saves_as_its_life_and_its_demands():
    both = LifeWithDemands(LIVES["offset weibull"], D, P, operating=True)
    again = deserialise_model(json.loads(json.dumps(serialise_model(both))))
    assert isinstance(again, LifeWithDemands)
    assert (again.d, again.p, again.operating) == (D, P, True)
    x = np.linspace(0.0, 500.0, 11)
    np.testing.assert_array_equal(again.sf(x), both.sf(x))


def test_from_new_it_is_the_life_and_the_demand_mode_in_series():
    # Two components in series, the life and an exponential life of rate
    # d p: the same reliability and mean.
    life = LIVES["weibull"]
    one = NonRepairableRBD(ONE, {"a": LifeWithDemands(life, D, P)})
    two = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": life, "b": E([D * P])},
    )
    x = np.array([1.0, 20.0, 100.0, 300.0])
    np.testing.assert_allclose(one.sf(x), two.sf(x), rtol=1e-14)
    assert one.mean() == pytest.approx(two.mean(), rel=1e-8)


def pair(**more):
    """The diagram of a unit with an exponential life and a demand mode
    beside two wearing units, and the same with the exponential life of
    the summed rate."""

    def build(a):
        others = {"reliability": W([150.0, 2.0]), "repairability": E([0.2])}
        return RepairableRBD(
            EDGES,
            {"a": a, "b": dict(others), "c": dict(others)},
            downtime_cost_rate=2.0,
        )

    common = {"repairability": E([0.5]), "repair_cost": 5.0, **more}
    return (
        build(
            {
                "reliability": E([LAMBDA]),
                "demand": {"probability": P, "rate": D},
                **common,
            }
        ),
        build({"reliability": E([LAMBDA + D * P]), **common}),
    )


@pytest.mark.parametrize(
    "method", sorted(set(REPAIRABLE_CALLS) - _BY_PARAMETERS)
)
def test_with_an_exponential_life_it_is_the_summed_rate(method):
    # Exact values to the last bit, and seeded simulations bit for bit.
    demand, life = pair()
    call = REPAIRABLE_CALLS[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            expected = call(life)
        except (NotImplementedError, ValueError) as error:
            with pytest.raises(type(error)) as refused:
                call(demand)
            assert str(refused.value) == str(error)
            return
        identical(call(demand), expected)


def test_the_wrapper_on_an_exponential_life_gives_the_summed_rate():
    # The general route (the life taken by its functions) on an
    # exponential life: the exponential of the summed rate, to the
    # accuracy of the numerical methods.
    def build(life):
        return RepairableRBD(
            ONE, {"a": {"reliability": life, "repairability": E([0.5])}}
        )

    wrapped = build(LifeWithDemands(E([LAMBDA]), D, P))
    summed = build(E([LAMBDA + D * P]))
    assert wrapped.mean_availability() == pytest.approx(
        summed.mean_availability(), rel=1e-12
    )
    x = np.array([5.0, 50.0, 300.0])
    np.testing.assert_allclose(
        wrapped.point_availability(x),
        summed.point_availability(x),
        rtol=1e-7,
    )
    assert wrapped.mission_availability(300.0) == pytest.approx(
        summed.mission_availability(300.0), rel=1e-7
    )


def test_the_long_run_availability_of_one_wearing_unit():
    # A unit's long-run availability is its mean life over the mean cycle.
    life = LIVES["weibull"]
    rbd = RepairableRBD(ONE, {"a": spec(life)})
    mttf = LifeWithDemands(life, D, P).mean()
    assert rbd.mean_availability() == pytest.approx(
        mttf / (mttf + 2.0), rel=1e-12
    )


def test_seeded_simulation_agrees_with_the_exact_values():
    rbd = RepairableRBD(EDGES, {k: spec(LIVES["weibull"]) for k in "abc"})
    n = 4000
    run = rbd.availability(400.0, mc_samples=n, seed=11, control_variate=False)
    exact = rbd.mission_availability(400.0)
    simulated = float(run.mean_availability)
    spread = float(np.std(run.uptimes / 400.0)) / math.sqrt(n)
    assert abs(simulated - exact) < 4.0 * spread


def test_the_spec_is_kept_and_the_routes_are_a_life_s():
    rbd = RepairableRBD(ONE, {"a": spec(LIVES["weibull"])})
    assert rbd._init_args["components"]["a"]["demand"] == {
        "probability": P,
        "rate": D,
    }
    assert isinstance(rbd.components["a"].reliability, LifeWithDemands)
    routes = rbd.analysis_routes()
    assert routes["mean_availability"].route == "exact"
    assert routes["point_availability"].route == "numerical"


def test_a_probability_of_0_leaves_the_life():
    life = LIVES["weibull"]
    with_none = RepairableRBD(ONE, {"a": spec(life, p=0.0)})
    alone = RepairableRBD(
        ONE, {"a": {"reliability": life, "repairability": E([0.5])}}
    )
    assert with_none.components["a"].reliability is life
    assert with_none.mean_availability() == alone.mean_availability()


def test_saving_keeps_the_demand_mode():
    rbd = RepairableRBD(ONE, {"a": spec(LIVES["weibull"])})
    saved = json.loads(rbd_to_json(rbd))
    again = rbd_from_json(rbd_to_json(rbd))
    assert again._init_args["components"]["a"]["demand"] == {
        "probability": P,
        "rate": D,
    }
    assert "demand" in json.dumps(saved)
    assert again.mean_availability() == rbd.mean_availability()


def test_a_unit_plan_holds_the_demand_mode():
    rbd = RepairableRBD(ONE, {"a": spec(LIVES["weibull"])})
    plan = rbd.plans["a"]
    assert plan.demand == {"probability": P, "rate": D}
    assert UnitPlan.from_spec(plan.spec()) == plan
    alone = UnitPlan(
        LIVES["weibull"], E([0.5]), demand={"probability": P, "rate": D}
    )
    assert alone.spec()["demand"] == {"probability": P, "rate": D}


def test_the_levers_are_the_probability_and_the_rate():
    demand, life = pair()
    levers = {
        lever.name: lever for lever in demand.levers() if lever.key == "a"
    }
    assert levers["demand.probability"].value == P
    assert levers["demand.probability"].bounds == (0.0, 1.0)
    assert levers["demand.rate"].value == D
    assert levers["demand.rate"].bounds == (0.0, np.inf)
    doubled = demand.with_levers({("a", "demand.rate"): 2 * D})
    _, twice = pair()
    twice = RepairableRBD(
        EDGES,
        {
            **twice._init_args["components"],
            "a": {
                "reliability": E([LAMBDA + 2 * D * P]),
                "repairability": E([0.5]),
                "repair_cost": 5.0,
            },
        },
        downtime_cost_rate=2.0,
    )
    assert doubled.mean_availability() == twice.mean_availability()
    # The summed rate's derivative times the other factor.
    by = demand.parameter_sensitivity()["a"]
    rate = life.parameter_sensitivity()["a"]["reliability.failure_rate"]
    assert by["demand.rate"] == pytest.approx(rate * P, rel=1e-6)
    assert by["demand.probability"] == pytest.approx(rate * D, rel=1e-6)


@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_it_scales_with_the_unit(c):
    # Times multiplied by c: the life's scale by c, the rate divided by c.
    def rbd(c):
        return RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": W([150.0 * c, 2.5]),
                    "repairability": E([0.5 / c]),
                    "demand": {"probability": P, "rate": D / c},
                }
            },
        )

    assert rbd(c).mean_availability() == pytest.approx(
        rbd(1.0).mean_availability(), rel=1e-9
    )


@pytest.mark.parametrize("basis", ["calendar", "operating"])
def test_a_duty_moves_the_demands_with_their_basis(basis):
    life, share = LIVES["weibull"], 0.4
    rbd = RepairableRBD(
        ONE,
        {"a": spec(life, duty=share, demand_basis=basis)},
    )
    moved = rbd.components["a"].reliability
    x = np.array([10.0, 100.0, 400.0])
    rate = D * P * (share if basis == "operating" else 1.0)
    np.testing.assert_allclose(
        moved.sf(x), life.sf(share * x) * np.exp(-rate * x), rtol=1e-13
    )


@pytest.mark.parametrize(
    "demand, match",
    [
        ({"probability": 1.5, "rate": D}, "in \\[0, 1\\]"),
        ({"probability": P, "rate": 0.0}, "finite number above 0"),
        ({"probability": P, "rate": math.inf}, "finite number above 0"),
        ({"probability": P}, "finite number above 0"),
        ({"probability": P, "rate": D, "times": [1.0]}, "give one"),
        ({"probability": P, "rate": D, "every": 1.0}, "a dict of"),
        (0.1, "a dict of"),
    ],
)
def test_a_demand_that_is_not_one_is_refused(demand, match):
    with pytest.raises(ValueError, match=match):
        RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": LIVES["weibull"],
                    "repairability": E([0.5]),
                    "demand": demand,
                }
            },
        )


@pytest.mark.parametrize(
    "more, match",
    [
        ({"reliability": FIXED(P)}, "'demand_rate'"),
        ({"demand_rate": D}, "give one"),
        ({"standby": {"units": 2}}, "component of its own in series"),
        (
            {"demand_basis": "operating"},
            "no duty to put that on the calendar",
        ),
        (
            {"duty": {"times": [0.0, 100.0], "shares": [0.5, 0.2]}},
            "'demand_basis': 'operating'",
        ),
    ],
)
def test_what_does_not_go_with_it_is_refused(more, match):
    with pytest.raises(ValueError, match=match):
        RepairableRBD(ONE, {"a": spec(LIVES["weibull"], **more)})
