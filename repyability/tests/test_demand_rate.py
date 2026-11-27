"""A probability of failing per demand with a demand rate (a spec's
``"demand_rate"``): demands arriving as a Poisson process of rate ``d``,
each failing a working unit with probability ``p``, make its failures a
Poisson process of rate ``d p``, an exponential life. The diagram is then
the diagram with that life, on every method and both engines, while the
spec keeps the probability and the rate."""

import json
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.maintenance import UnitPlan
from repyability.rbd import _compiled, _event_loop
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json
from repyability.tests.catalogue import EDGES, E, W
from repyability.tests.test_analysis_routes import REPAIRABLE_CALLS
from repyability.tests.test_simulation_engines import identical

FIXED = surv.FixedEventProbability.from_params
ONE = [("s", "a"), ("a", "t")]
P, D = 0.1, 0.05

#: The calls whose arguments are made from the given models' parameters
#: (a probability and a rate here, one rate there), which differ.
_BY_PARAMETERS = {
    "parameter_sensitivity",
    "mean_availability_uncertainty",
    "point_availability_uncertainty",
    "mission_availability_uncertainty",
    "expected_cost_rate_uncertainty",
    "expected_npv_uncertainty",
    "uncertainty_importance",
    # A chunk and a shard carry the diagram as saved, its spec as given:
    # their runs are compared through availability_from_chunks.
    "simulate_chunk",
    "shards",
}


def pair(p=P, d=D, **more):
    """The diagram of a pump that may fail to start on demand beside two
    wearing units, and the same with the exponential life its demands
    make."""

    def build(a):
        others = {"reliability": W([150.0, 2.0]), "repairability": E([0.2])}
        return RepairableRBD(
            EDGES,
            {"a": a, "b": dict(others), "c": dict(others)},
            downtime_cost_rate=2.0,
        )

    common = {"repairability": E([0.5]), "repair_cost": 5.0, **more}
    return (
        build({"reliability": FIXED(p), "demand_rate": d, **common}),
        build({"reliability": E([d * p]), **common}),
    )


@pytest.mark.parametrize(
    "method", sorted(set(REPAIRABLE_CALLS) - _BY_PARAMETERS)
)
def test_the_diagram_is_that_with_the_exponential_life(method):
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


def test_the_routes_are_those_of_the_exponential_life():
    demand, life = pair()
    assert demand.analysis_routes() == life.analysis_routes()


def test_sensitivities_take_the_probability_and_the_rate():
    # The rate times the probability is the failure rate: the derivative
    # by each is the rate's times the other.
    demand, life = pair()
    by = demand.parameter_sensitivity()["a"]
    rate = life.parameter_sensitivity()["a"]["reliability.failure_rate"]
    assert by["demand_rate"] == pytest.approx(rate * P, rel=1e-6)
    assert by["reliability.p"] == pytest.approx(rate * D, rel=1e-6)


def test_both_engines_give_the_exponential_life_s_runs():
    pytest.importorskip("numba")
    demand, life = pair()
    for engine in ("python", "numba"):
        identical(
            demand.availability(200.0, mc_samples=50, seed=3, engine=engine),
            life.availability(200.0, mc_samples=50, seed=3, engine=engine),
        )


def test_the_compiled_engine_takes_it():
    # Nothing new for the compiled engine: it sees the exponential life.
    demand, _ = pair()
    plan = _event_loop._stream_plan(demand, 1.0, 0, False)[0]
    assert _compiled.unsupported(demand, plan, None) is None


def test_the_long_run_availability_of_one_unit():
    mu = 0.5
    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": FIXED(P),
                "demand_rate": D,
                "repairability": E([mu]),
            }
        },
    )
    assert rbd.mean_availability() == pytest.approx(
        mu / (mu + D * P), rel=1e-14
    )


def test_a_probability_of_0_never_fails():
    rbd = RepairableRBD(
        ONE + [("s", "b"), ("b", "t")],
        {
            "a": {
                "reliability": FIXED(0.0),
                "demand_rate": D,
                "repairability": E([0.5]),
            },
            "b": {"reliability": W([100.0, 2.0]), "repairability": E([1.0])},
        },
    )
    assert rbd.mean_availability() == 1.0
    assert "a" not in rbd.components


def test_a_probability_of_1_fails_at_every_demand():
    demand, life = pair(p=1.0)
    assert demand.mean_availability() == life.mean_availability()
    assert demand.components["a"].reliability.params[0] == D


@pytest.mark.parametrize(
    "rate", [-1.0, 0.0, np.inf, np.nan, "0.05", True, [0.05]]
)
def test_a_rate_that_is_no_rate_is_refused(rate):
    with pytest.raises(ValueError, match="demand_rate must be a finite"):
        pair(d=rate)


def test_a_rate_on_a_life_says_to_draw_two_components():
    with pytest.raises(ValueError, match="two components in series"):
        RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": W([100.0, 2.0]),
                    "demand_rate": D,
                    "repairability": E([1.0]),
                }
            },
        )


def test_a_bernoulli_is_refused_naming_the_fixed_probability():
    with pytest.raises(ValueError, match="FixedEventProbability.from_params"):
        RepairableRBD(
            ONE,
            {
                "a": {
                    "reliability": surv.Bernoulli.from_params(0.1),
                    "demand_rate": D,
                    "repairability": E([1.0]),
                }
            },
        )


def test_a_bare_probability_is_refused_naming_the_key():
    rbd = RepairableRBD(
        ONE, {"a": {"reliability": FIXED(P), "repairability": E([1.0])}}
    )
    with pytest.raises(
        NotImplementedError, match="give its spec a demand_rate"
    ):
        rbd.mean_availability()
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == "refused" and "demand_rate" in route.reason


def test_saving_keeps_the_probability_and_the_rate():
    demand, _ = pair()
    saved = json.loads(rbd_to_json(demand))
    again = rbd_from_json(rbd_to_json(demand))
    spec = again._init_args["components"]["a"]
    assert spec["demand_rate"] == D
    assert spec["reliability"].params[0] == P
    assert "demand_rate" in json.dumps(saved)
    assert again.mean_availability() == demand.mean_availability()


def test_a_unit_plan_holds_the_rate():
    demand, _ = pair()
    plan = demand.plans["a"]
    assert plan.demand_rate == D
    assert UnitPlan.from_spec(plan.spec()) == plan
    assert plan.spec()["demand_rate"] == D


def test_the_levers_are_the_probability_and_the_rate():
    demand, _ = pair()
    levers = {
        lever.name: lever for lever in demand.levers() if lever.key == "a"
    }
    assert levers["demand_rate"].value == D
    assert levers["demand_rate"].bounds == (0.0, np.inf)
    assert levers["reliability.p"].value == P
    assert levers["reliability.p"].bounds == (0.0, 1.0)
    doubled = demand.with_levers({("a", "demand_rate"): 2 * D})
    _, twice = pair(d=2 * D)
    assert doubled.mean_availability() == twice.mean_availability()


@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_the_rate_scales_inversely_with_the_unit(c):
    # Times multiplied by c: the rate divided by c, the probability kept.
    def rbd(c):
        return RepairableRBD(
            ONE + [("s", "b"), ("b", "t")],
            {
                "a": {
                    "reliability": FIXED(P),
                    "demand_rate": D / c,
                    "repairability": E([0.5 / c]),
                },
                "b": {
                    "reliability": W([100.0 * c, 2.0]),
                    "repairability": E([1.0 / c]),
                },
            },
        )

    base, scaled = rbd(1.0), rbd(c)
    assert scaled.mean_availability() == pytest.approx(
        base.mean_availability(), rel=1e-9
    )
    t = np.array([10.0, 200.0])
    np.testing.assert_allclose(
        scaled.point_availability(t * c),
        base.point_availability(t),
        rtol=1e-6,
    )


def test_a_rate_with_a_duty_is_refused():
    with pytest.raises(ValueError, match="no duty"):
        pair(duty=0.5)
