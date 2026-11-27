"""The first system failure of a repairable diagram with common-cause
groups or standby groups of exponential units (``first_failure_sf``,
``mean_time_to_first_failure``): their chains, made absorbing where the
system fails.

Checked against chains written out by hand, the bracket of
``first_failure_bounds`` and seeded simulations of the diagrams of
``test_ccf_over_time.py`` and ``test_repairable_standby.py``."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import (
    MGL,
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    NodeState,
    RepairableRBD,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params

PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
PAIR_THEN_C = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
THREE_THEN_C = [("s", n) for n in "abd"] + [(n, "c") for n in "abd"]
THREE_THEN_C += [("c", "t")]
TIMES = np.array([10.0, 50.0, 200.0, 500.0])
SAMPLES = 100_000


def unit(lam, mu):
    return {"reliability": E([lam]), "repairability": E([mu])}


OTHER = {"reliability": W([500.0, 2.0]), "repairability": E([0.5])}


def survival(generator, start, times):
    return np.array([(start @ expm(generator * t)).sum() for t in times])


def agrees_with_simulation(rbd, seed, times=TIMES):
    value = rbd.first_failure_sf(times)
    simulated = rbd.first_failure_sf(
        times, method="simulate", mc_samples=SAMPLES, seed=seed
    )
    error = np.sqrt(value * (1.0 - value) / SAMPLES)
    assert np.all(np.abs(simulated - value) <= 4.0 * error), (
        value,
        simulated,
    )
    return value


# -- common-cause groups ---------------------------------------------------


def test_a_beta_factor_pair_is_its_chain_written_out_by_hand():
    lam, mu, beta = 0.01, 0.2, 0.2
    rbd = RepairableRBD(
        PAIR,
        {n: unit(lam, mu) for n in "ab"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    own, both = (1.0 - beta) * lam, beta * lam
    # Both up; one down (either). From both up the shared cause fails the
    # system; with one down, the other's own cause or the shared one does.
    t = np.array(
        [
            [-(2 * own + both), 2 * own],
            [mu, -(mu + own + both)],
        ]
    )
    start = np.array([1.0, 0.0])
    np.testing.assert_allclose(
        rbd.first_failure_sf(TIMES), survival(t, start, TIMES), rtol=1e-10
    )
    assert rbd.mean_time_to_first_failure() == pytest.approx(
        start @ np.linalg.solve(-t, np.ones(2)), rel=1e-10
    )
    # From one member down, in its repair.
    np.testing.assert_allclose(
        rbd.first_failure_sf(TIMES, state={"a": NodeState(alive=False)}),
        survival(t, np.array([0.0, 1.0]), TIMES),
        rtol=1e-10,
    )
    route = rbd.analysis_routes()["first_failure_sf"]
    assert route.route == "numerical" and "shared causes" in route.reason


def ccf_diagrams():
    members = {n: unit(0.01, 0.2) for n in "abd"}
    return {
        "beta factor": RepairableRBD(
            PAIR_THEN_C,
            {"a": members["a"], "b": members["b"], "c": OTHER},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
        ),
        "MGL": RepairableRBD(
            THREE_THEN_C,
            members | {"c": OTHER},
            ccf_groups=[CCFGroup(["a", "b", "d"], MGL(0.2, 0.4))],
        ),
        "alpha factor": RepairableRBD(
            THREE_THEN_C,
            members | {"c": OTHER},
            ccf_groups=[
                CCFGroup(["a", "b", "d"], AlphaFactor([0.8, 0.12, 0.08]))
            ],
        ),
    }


@pytest.mark.parametrize(
    "name, seed", [("beta factor", 21), ("MGL", 22), ("alpha factor", 23)]
)
def test_common_cause_groups_against_the_simulation(name, seed):
    rbd = ccf_diagrams()[name]
    value = agrees_with_simulation(rbd, seed)
    bounds = rbd.first_failure_bounds(TIMES)
    slack = 1e-3 * (1.0 - value) + 4e-6
    assert np.all(bounds.lower <= value + slack)
    assert np.all(value <= bounds.upper + slack)
    # Common causes fail the group sooner than independent members do.
    apart = RepairableRBD(
        rbd._init_args["edges"], rbd._init_args["components"]
    )
    assert np.all(value < apart.first_failure_sf(TIMES))


def test_a_group_of_members_that_are_not_exponential_is_refused():
    wear = {"reliability": W([100.0, 2.0]), "repairability": E([0.2])}
    rbd = RepairableRBD(
        PAIR,
        {"a": wear, "b": wear},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
    )
    with pytest.raises(NotImplementedError, match="common-cause group") as e:
        rbd.first_failure_sf(10.0)
    assert "first_failure_bounds" in str(e.value)
    assert rbd.analysis_routes()["first_failure_sf"].route == "refused"


# -- standby groups ------------------------------------------------------------


def group(lam=0.02, mu=0.25, crews=None, **standby):
    spec = unit(lam, mu) | {"standby": standby}
    return RepairableRBD(
        [("s", "g"), ("g", "t")], {"g": spec}, repair_crews=crews
    )


def test_a_cold_standby_pair_is_its_chain_written_out_by_hand():
    lam, mu = 0.02, 0.25
    rbd = group(lam, mu)
    # One operating with a spare; one operating, the other in repair.
    t = np.array([[-lam, lam], [mu, -(mu + lam)]])
    start = np.array([1.0, 0.0])
    np.testing.assert_allclose(
        rbd.first_failure_sf(TIMES), survival(t, start, TIMES), rtol=1e-10
    )
    assert rbd.mean_time_to_first_failure() == pytest.approx(
        (2 * lam + mu) / lam**2, rel=1e-10
    )
    route = rbd.analysis_routes()["mean_time_to_first_failure"]
    assert route.route == "numerical" and "standby" in route.reason


@pytest.mark.parametrize(
    "crews, standby, seed",
    [
        (None, {}, 31),
        (1, {}, 32),
        (None, {"switching_probability": 0.9}, 33),
        (None, {"dormancy_factor": 0.5}, 34),
        (None, {"units": 3, "k": 2}, 35),
        (2, {"units": 4, "k": 2, "dormancy_factor": 0.2}, 36),
    ],
)
def test_standby_groups_against_the_simulation(crews, standby, seed):
    agrees_with_simulation(group(crews=crews, **standby), seed)


def test_a_standby_group_beside_a_unit_that_wears():
    # The group's chain beside the phases of a Weibull unit, either
    # keeping the system up.
    spec = unit(0.02, 0.25) | {"standby": {}}
    wear = {"reliability": W([100.0, 2.0]), "repairability": E([0.1])}
    rbd = RepairableRBD(
        [("s", "g"), ("s", "w"), ("g", "t"), ("w", "t")],
        {"g": spec, "w": wear},
    )
    value = agrees_with_simulation(rbd, 37)
    alone = group().first_failure_sf(TIMES)
    assert np.all(value > alone)


def test_a_standby_group_from_its_long_run_state():
    lam, mu = 0.02, 0.25
    rbd = group(lam, mu)
    t = np.array([[-lam, lam], [mu, -(mu + lam)]])
    # The long-run distribution of the whole chain (third state: both in
    # repair, the group down).
    q = np.array(
        [
            [-lam, lam, 0.0],
            [mu, -(mu + lam), lam],
            [0.0, 2 * mu, -2 * mu],
        ]
    )
    a = np.vstack([q.T[:-1], np.ones(3)])
    long_run = np.linalg.solve(a, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(
        rbd.first_failure_sf(TIMES, state="stationary"),
        survival(t, long_run[:2], TIMES),
        rtol=1e-9,
    )
    with pytest.raises(NotImplementedError, match="standby group"):
        rbd.first_failure_sf(10.0, state={"g": NodeState(alive=False)})


def test_a_standby_group_of_units_that_wear_is_refused():
    spec = {
        "reliability": W([100.0, 2.0]),
        "repairability": E([0.25]),
        "standby": {},
    }
    rbd = RepairableRBD([("s", "g"), ("g", "t")], {"g": spec})
    with pytest.raises(NotImplementedError, match="exponential lives") as e:
        rbd.mean_time_to_first_failure()
    assert "first_failure_bounds" in str(e.value)


def test_tests_and_maintenance_at_fixed_intervals_stay_refused():
    scheduled = unit(0.01, 0.1) | {"preventive": {"interval": 100.0}}
    tested = {
        "reliability": E([0.01]),
        "repairability": E([0.1]),
        "inspection": {"interval": 50.0},
    }
    for spec, what in ((scheduled, "maintenance"), (tested, "inspections")):
        rbd = RepairableRBD(PAIR, {"a": spec, "b": unit(0.01, 0.1)})
        with pytest.raises(NotImplementedError, match=what) as e:
            rbd.first_failure_sf(10.0)
        assert "fixed interval" in str(e.value)
        assert "first_failure_bounds" in str(e.value)
