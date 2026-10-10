"""Replacement on condition over time (#161): a component inspected every
interval, and replaced at an inspection when it is more likely than a
threshold to fail before the next, followed from new (or from a state) one
inspection interval at a time, with the units each inspection keeps by age.
Checked against block replacement (a threshold of 0 replaces at every
inspection), its long-run values, the inspection that decides on the unit
in service at the start at its exact age, and the simulation."""

import numpy as np
import pytest
import surpyval as surv
from scipy.optimize import brentq

from repyability import NodeState, RepairableRBD
from repyability.rbd import _runs
from repyability.rbd import routes as r

W, E, LN = (
    surv.Weibull.from_params,
    surv.Exponential.from_params,
    surv.LogNormal.from_params,
)
LIFE, REPAIR, INTERVAL = W([1000.0, 2.5]), LN([2.0, 0.5]), 200.0
SHARED = ("replace", "preventive")


def unit(policy="condition", threshold=0.2):
    preventive = {
        "interval": INTERVAL,
        "policy": policy,
        "duration": E([1 / 4.0]),
        "cost": 20.0,
    }
    if policy == "condition":
        preventive.update(threshold=threshold, inspection_cost=3.0)
    return {
        "reliability": LIFE,
        "repairability": REPAIR,
        "replace_cost": 50.0,
        "preventive": preventive,
    }


def single(spec):
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec})


def categories(cost) -> dict:
    return {key: float(np.ravel(v)[0]) for key, v in cost.by_category.items()}


TIMES = np.array([50.0, 199.0, 200.0, 201.0, 650.0, 1000.0, 1999.0, 5000.0])


@pytest.mark.parametrize(
    "state",
    [None, {"c": NodeState(age=30.0, phase=50.0)}],
    ids=["new", "from a state"],
)
def test_a_threshold_of_0_is_block_replacement(state):
    # Every unit up at an inspection is then replaced.
    kw = {} if state is None else {"state": state}
    condition, block = single(unit(threshold=0.0)), single(unit("block"))
    np.testing.assert_allclose(
        condition.point_availability(TIMES, **kw),
        block.point_availability(TIMES, **kw),
        atol=1e-12,
    )
    ours = categories(condition.expected_cost(3000.0, **kw))
    theirs = categories(block.expected_cost(3000.0, **kw))
    for key in SHARED:
        assert ours[key] == pytest.approx(theirs[key], rel=1e-9)
    # Charged at each inspection the unit is up at: each replacement's.
    assert ours["inspection"] / 3.0 == pytest.approx(
        ours["preventive"] / 20.0, rel=1e-9
    )


@pytest.mark.parametrize("phase", [0.0, 70.0])
@pytest.mark.parametrize("threshold", [0.05, 0.2, 0.5])
def test_it_settles_into_its_long_run_cycle(threshold, phase):
    # From its long-run state, every interval is alike: the values over whole
    # intervals are the long-run ones.
    # Over [T, 4T): at phase 0 the inspection at the start is past.
    rbd = single(unit(threshold=threshold))
    state = {"c": NodeState(stationary=True, phase=phase)}
    ends = np.array([INTERVAL, 4 * INTERVAL])
    mission = np.ravel(rbd.mission_availability(ends, state=state))
    up = (ends[1] * mission[1] - ends[0] * mission[0]) / (3 * INTERVAL)
    assert up == pytest.approx(rbd.mean_availability(), abs=2e-6)
    cost = np.ravel(rbd.expected_cost(ends, state=state).total)
    rate = (cost[1] - cost[0]) / (3 * INTERVAL)
    assert rate == pytest.approx(rbd.expected_cost_rate(), rel=2e-6)
    # It repeats from one interval to the next.
    later = rbd.point_availability(TIMES[:4] + 7 * INTERVAL, state=state)
    np.testing.assert_allclose(
        rbd.point_availability(TIMES[:4], state=state), later, atol=1e-10
    )


def threshold_age(threshold: float) -> float:
    """The age from which an inspection replaces the unit."""

    def likely(age):
        return 1.0 - LIFE.sf(age + INTERVAL) / LIFE.sf(age) - threshold

    return brentq(likely, 1.0, 5000.0, xtol=1e-12)


@pytest.mark.parametrize("side", [-1.0, 1.0], ids=["kept", "replaced"])
def test_the_first_inspection_decides_at_the_unit_s_own_age(side):
    # The unit in service at the start reaches its first inspection at its
    # own age, however it falls on the grid: just short of the threshold
    # age it is kept, just past it replaced (if it has not failed).
    phase = 50.0
    first = INTERVAL - phase
    age = threshold_age(0.2) - first + side * 1e-3
    rbd = single(unit(threshold=0.2))
    state = {"c": NodeState(age=age, phase=phase)}
    replaced = categories(rbd.expected_cost(first + 1.0, state=state))
    unfailed = LIFE.sf(age + first) / LIFE.sf(age)
    expected = unfailed if side > 0 else 0.0
    assert replaced["preventive"] / 20.0 == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "state",
    [
        None,
        {"c": NodeState(age=700.0, phase=50.0)},
        {"c": NodeState(alive=False, down_for=1.0, phase=150.0)},
    ],
    ids=["new", "up and kept so far", "in a repair"],
)
def test_against_the_simulation(state):
    rbd = single(unit())
    kw = {} if state is None else {"state": state}
    horizon, n = 1400.0, 8_000
    mission = float(np.ravel(rbd.mission_availability(horizon, **kw))[0])
    failures = float(np.ravel(rbd.expected_failures(horizon, **kw))[0])
    costs = categories(rbd.expected_cost(horizon, **kw))
    simulated = rbd.availability(
        horizon, mc_samples=n, seed=5, control_variate=False, **kw
    )
    up = np.asarray(simulated.uptimes, dtype=float).ravel() / horizon
    assert mission == pytest.approx(up.mean(), abs=4 * up.std() / np.sqrt(n))
    counted = simulated.system_failures / n
    assert failures == pytest.approx(counted, abs=4 * np.sqrt(failures / n))
    cost = simulated.cost
    for key, each in (("replace", 50.0), ("preventive", 20.0)):
        mean = costs[key] / each
        got = float(np.mean(cost.by_category[key])) / each
        assert mean == pytest.approx(got, abs=4 * np.sqrt(mean / n) + 1e-3)
    inspections = float(np.mean(cost.by_category["inspection"])) / 3.0
    assert costs["inspection"] / 3.0 == pytest.approx(inspections, abs=0.02)


def test_the_twin_keeps_it_and_a_run_to_a_tolerance_takes_exact_values():
    rbd = single(unit())
    twin, changes = _runs._twin(rbd)
    assert twin._preventive["c"].policy == "condition" and not changes
    report = rbd.analysis_routes()
    for name in (
        "point_availability",
        "mission_availability",
        "expected_events",
        "expected_cost",
    ):
        assert report[name].route == r.NUMERICAL
    assert "by default a run's means are theirs" in (
        report["availability"].reason
    )
    result = rbd.availability(500.0, tolerance=0.01, mc_samples=200, seed=1)
    exact = float(np.ravel(rbd.mission_availability(500.0))[0])
    estimate = result.mean_availability_interval().estimate
    assert estimate == pytest.approx(exact, abs=1e-12)
