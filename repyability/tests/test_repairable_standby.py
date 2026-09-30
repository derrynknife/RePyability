"""Repairable standby groups (#91): a component spec's ``"standby"`` makes
the node a group of identical units, ``k`` operating and the rest spares,
each repaired on its own and switched in when an operating unit fails.
Checked against the Markov chains of textbook arrangements, timelines
worked by hand with a shared crew, and the simulation."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RBD, RepairableRBD
from repyability.rbd import _compiled
from repyability.rbd import routes as r

E, W, L = (
    surv.Exponential.from_params,
    surv.Weibull.from_params,
    surv.LogNormal.from_params,
)
X = surv.ExactEventTime.from_params


def group(lam=0.01, mu=0.5, crews=None, cost_rate=0.0, costs=None, **standby):
    """A standby group of exponential units, alone in an RBD."""
    spec = {
        "reliability": E([lam]),
        "repairability": E([mu]),
        "standby": standby,
        **(costs or {}),
    }
    return RepairableRBD(
        [("s", "g"), ("g", "t")],
        {"g": spec},
        repair_crews=crews,
        downtime_cost_rate=cost_rate,
    )


def stationary(rates: dict, states: list) -> np.ndarray:
    """The long-run probabilities of a chain given by its ``rates``
    (``{(source, target): rate}``) over ``states``."""
    index = {state: i for i, state in enumerate(states)}
    generator = np.zeros((len(states), len(states)))
    for (source, target), rate in rates.items():
        generator[index[source], index[target]] += rate
    generator -= np.diag(generator.sum(axis=1))
    system = generator.T.copy()
    system[0] = 1.0
    rhs = np.zeros(len(states))
    rhs[0] = 1.0
    return np.linalg.solve(system, rhs)


@pytest.mark.parametrize("crews, both", [(None, 2), (1, 1)])
def test_two_unit_cold_standby_matches_its_markov_chain(crews, both):
    lam, mu = 0.01, 0.5
    # Both up, one under repair, both under repair (by `both` crews).
    weights = np.array([1.0, lam / mu, lam / mu * lam / (both * mu)])
    p = weights / weights.sum()
    rbd = group(lam, mu, crews)
    assert rbd.mean_availability() == pytest.approx(1 - p[2], rel=1e-12)
    assert rbd.system_failure_frequency() == pytest.approx(
        p[1] * lam, rel=1e-12
    )
    # Down, the group waits for the first repair.
    assert rbd.mean_down_time() == pytest.approx(1 / (both * mu), rel=1e-9)
    assert rbd.analysis_routes()["mean_availability"].route == r.EXACT


def test_warm_spares_match_the_textbook_chain():
    # The spare fails at 0.4 of the operating rate, and is repaired too.
    lam, mu, f = 0.01, 0.5, 0.4
    weights = np.array(
        [1.0, (1 + f) * lam / mu, (1 + f) * lam / mu * lam / (2 * mu)]
    )
    p = weights / weights.sum()
    rbd = group(lam, mu, dormancy_factor=f)
    assert rbd.mean_availability() == pytest.approx(1 - p[2], rel=1e-12)
    # Every unit failure is a repair: the operating unit's and the spare's.
    unit_failures = (p[0] * (1 + f) + p[1]) * lam
    priced = group(lam, mu, dormancy_factor=f, costs={"repair_cost": 7.0})
    assert priced.expected_cost_rate() == pytest.approx(
        7.0 * unit_failures, rel=1e-12
    )


def test_a_failed_switch_leaves_the_position_empty():
    lam, mu, p = 0.01, 0.5, 0.9
    ready, repairing, stranded, both = (
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (0, 0, 2),
    )
    rates = {
        (ready, repairing): lam * p,
        (ready, stranded): lam * (1 - p),
        (repairing, ready): mu,
        (repairing, both): lam,
        # The repaired unit fills the empty position; the spare waits on.
        (stranded, ready): mu,
        (both, repairing): 2 * mu,
    }
    pi = stationary(rates, [ready, repairing, stranded, both])
    rbd = group(lam, mu, switching_probability=p)
    assert rbd.mean_availability() == pytest.approx(pi[0] + pi[1], rel=1e-12)
    assert rbd.system_failure_frequency() == pytest.approx(
        pi[0] * lam * (1 - p) + pi[1] * lam, rel=1e-12
    )


def test_a_switch_that_always_fails_leaves_a_single_unit():
    lam, mu = 0.01, 0.5
    rbd = group(lam, mu, switching_probability=0.0)
    assert rbd.mean_availability() == pytest.approx(mu / (lam + mu), rel=1e-12)
    run = rbd.availability(t_simulation=100_000.0, mc_samples=40, seed=2)
    interval = run.mean_availability_interval(confidence=0.999)
    assert interval.lower <= mu / (lam + mu) <= interval.upper


@pytest.mark.parametrize(
    "crews, standby",
    [
        (None, {}),
        (1, {}),
        (None, {"switching_probability": 0.9}),
        (None, {"dormancy_factor": 0.5}),
        (None, {"units": 3, "k": 2}),
        (2, {"units": 4, "k": 2, "dormancy_factor": 0.2}),
    ],
)
def test_the_simulation_agrees_with_the_chain(crews, standby):
    rbd = group(0.02, 0.25, crews, **standby)
    window, samples = 50_000.0, 40
    run = rbd.availability(t_simulation=window, mc_samples=samples, seed=4)
    interval = run.mean_availability_interval(confidence=0.999)
    assert interval.lower <= rbd.mean_availability() <= interval.upper
    spread = math.sqrt(run.system_failures) / (samples * window)
    frequency = rbd.system_failure_frequency()
    assert abs(run.failure_frequency - frequency) < 4 * spread


def pair_with_a_crew(priority=0.0):
    """One crew for c (failing 1 after each repair, repaired in 5) and a
    cold standby pair g (units failing 2 after each repair, repaired in
    1.5)."""
    return RepairableRBD(
        [("s", "c"), ("c", "g"), ("g", "t")],
        {
            "c": {
                "reliability": X(1.0),
                "repairability": X(5.0),
                "priority": priority,
            },
            "g": {
                "reliability": X(2.0),
                "repairability": X(1.5),
                "standby": {},
            },
        },
        repair_crews=1,
    )


def uptimes(rbd, window):
    result = rbd.availability(t_simulation=window, mc_samples=1, seed=0)
    return {node: round(up, 9) for node, up in result.node_uptime.items()}


def test_a_shared_crew_repairs_the_units_in_turn():
    # c holds the crew from 1 to 6. g's duty unit fails at 2 (the spare
    # takes over) and waits; the spare fails at 4 and waits: g is down. At
    # 6 the crew takes unit 0 (due first) until 7.5, when g is up again; c
    # fails at 7 and waits. At 7.5 the crew takes unit 1 (due at 4, before
    # c), until 9, when it becomes the spare; unit 0 fails at 9.5 and the
    # spare takes over.
    assert uptimes(pair_with_a_crew(), 10.0) == {"c": 2.0, "g": 6.5}
    # c ahead in the queue: at 7.5 the crew takes c instead, so unit 1
    # waits, and g is down again when unit 0 fails at 9.5.
    assert uptimes(pair_with_a_crew(priority=1.0), 10.0) == {
        "c": 2.0,
        "g": 6.0,
    }
    # Without a limit on crews, a unit is repaired as it fails, and a
    # spare is always ready.
    free = pair_with_a_crew()
    free.repair_crews = None
    assert uptimes(free, 10.0) == {"c": 2.0, "g": 10.0}
    # The event-stepping API follows the same queue: the system (c and g in
    # series) is down from 1 to the end, with g up again at 10.
    rbd = pair_with_a_crew()
    rbd.initialize_event_queue(10.0)
    changes = [rbd.next_event()]
    while changes[-1][0] < 10.0:
        changes.append(rbd.next_event())
    assert changes == [(1.0, False), (10.0, False)]
    assert rbd.component_status == {"c": False, "g": True}


def test_costs_are_charged_per_unit_repair():
    lam, mu = 0.02, 0.25
    costs = {"repair_cost": 10.0, "downtime_cost": 3.0}
    rbd = group(lam, mu, cost_rate=50.0, costs=costs)
    # Cold: only operating units fail, at lam each while the group is up.
    up = rbd.mean_availability()
    expected = 10.0 * lam * up + (3.0 + 50.0) * (1 - up)
    assert rbd.expected_cost_rate() == pytest.approx(expected, rel=1e-12)
    window = 50_000.0
    cost = rbd.cost(t_simulation=window, mc_samples=100, seed=6)
    interval = cost.mean_interval(confidence=0.999)
    assert interval.lower / window <= expected <= interval.upper / window


def test_what_the_chains_do_not_cover_is_refused():
    spec = {
        "reliability": W([100.0, 1.5]),
        "repairability": E([0.5]),
        "standby": {},
    }
    rbd = RepairableRBD([("s", "g"), ("g", "t")], {"g": spec})
    routes = rbd.analysis_routes()
    with pytest.raises(NotImplementedError, match="standby group") as error:
        rbd.mean_availability()
    assert routes["mean_availability"].reason == str(error.value)
    with pytest.raises(NotImplementedError, match="standby group") as error:
        rbd.point_availability([1.0])
    assert routes["point_availability"].reason == str(error.value)
    rbd.availability(t_simulation=100.0, mc_samples=2, seed=1)
    # A crew shared by the group and another component ties them together.
    shared = RepairableRBD(
        [("s", "g"), ("g", "c"), ("c", "t")],
        {
            "g": {**spec, "reliability": E([0.01])},
            "c": {"reliability": E([0.02]), "repairability": E([1.0])},
        },
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="standby group") as error:
        shared.mean_availability()
    report = shared.analysis_routes()["mean_availability"]
    assert report.route == r.REFUSED and report.reason == str(error.value)
    with pytest.raises(NotImplementedError, match="importance"):
        shared.birnbaum_importance()


def test_the_group_s_values_enter_the_rbd_like_a_component():
    # A group in series with a plain unit; with enough crews for all.
    unit = {"reliability": E([0.05]), "repairability": E([1.0])}
    spec = {"reliability": E([0.01]), "repairability": E([0.5]), "standby": {}}
    rbd = RepairableRBD(
        [("s", "g"), ("g", "u"), ("u", "t")],
        {"g": spec, "u": unit},
        repair_crews=3,
    )
    alone = group(0.01, 0.5)
    assert rbd.node_availability()["g"] == pytest.approx(
        alone.mean_availability(), rel=1e-12
    )
    assert rbd.mean_availability() == pytest.approx(
        alone.mean_availability() / 1.05, rel=1e-12
    )
    importance = rbd.birnbaum_importance()
    assert importance["u"] == pytest.approx(alone.mean_availability())


def test_a_group_inside_a_nested_rbd():
    inner = group(0.01, 0.5)
    outer = RepairableRBD(
        [("s", "x"), ("x", "y"), ("y", "t")],
        {
            "x": inner,
            "y": {"reliability": E([0.02]), "repairability": E([1.0])},
        },
    )
    exact = outer.mean_availability()
    assert exact == pytest.approx(inner.mean_availability() / 1.02, rel=1e-12)
    run = outer.availability(t_simulation=50_000.0, mc_samples=40, seed=5)
    interval = run.mean_availability_interval(confidence=0.999)
    assert interval.lower <= exact <= interval.upper


def test_runs_with_groups_are_reproducible_and_python_only():
    rbd = group(0.02, 0.25, 1, units=3, switching_probability=0.8)
    run = dict(t_simulation=2_000.0, mc_samples=200, seed=9)
    serial = rbd.availability(**run)
    assert rbd.availability(**run).node_uptime == serial.node_uptime
    assert rbd.availability(**run, n_jobs=2).node_uptime == serial.node_uptime
    alone = group(0.02, 0.25, units=3)
    plan = alone._stream_plan(1.0, 0, False)[0]
    assert _compiled.unsupported(alone, plan, None) == "standby groups"
    assert alone.analysis_routes()["availability"].engine == "python"
    gain = group(0.02, 0.25, 2, units=3).compare(
        rbd, 2_000.0, mc_samples=200, seed=9
    )
    assert gain.estimate > 0.0


def test_standby_specs_are_checked_and_saved():
    for bad in (
        {"units": 1},
        {"k": 2},
        {"units": 2.5},
        {"dormancy_factor": 2.0},
        {"switching_probability": -0.1},
        {"spares": 1},
        "two",
    ):
        with pytest.raises(ValueError, match="standby"):
            RepairableRBD(
                [("s", "g"), ("g", "t")],
                {
                    "g": {
                        "reliability": E([0.01]),
                        "repairability": E([0.5]),
                        "standby": bad,
                    }
                },
            )
    for extra in (
        {"preventive": {"interval": 5.0}},
        {"repairability": "instant"},
    ):
        with pytest.raises(ValueError, match="standby group"):
            RepairableRBD(
                [("s", "g"), ("g", "t")],
                {
                    "g": {
                        "reliability": E([0.01]),
                        "repairability": E([0.5]),
                        "standby": {},
                        **extra,
                    }
                },
            )
    rbd = group(0.02, 0.25, 1, units=3, k=2, switching_probability=0.8)
    back = RBD.from_json(rbd.to_json())
    assert back._standby == rbd._standby
    run = dict(t_simulation=500.0, mc_samples=50, seed=7)
    assert (
        back.availability(**run).node_uptime
        == rbd.availability(**run).node_uptime
    )
