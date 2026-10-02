"""Shared repair crews in the RepairableRBD simulation (#89): the machine
repair model's closed form, results unchanged with enough crews, the order
in which crews take waiting jobs, and what the exact methods refuse. The
exact long-run values for exponential components (#90) are tested in
``test_crew_chain.py``."""

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


def machine_repair(n, lam, mu, crews, k=1):
    """The long-run probability that at least ``k`` of ``n`` identical
    exponential units are up, with ``crews`` crews: the number failed is a
    birth-death chain, failing at ``(n - j) * lam`` and repaired at
    ``min(j, crews) * mu``."""
    weights = [1.0]
    for j in range(n):
        weights.append(weights[-1] * (n - j) * lam / (min(j + 1, crews) * mu))
    weights = np.array(weights) / sum(weights)
    return weights[: n - k + 1].sum()


def pumps(n, lam, mu, crews, k=1):
    unit = {"reliability": E([lam]), "repairability": E([mu])}
    edges = [("s", i) for i in range(n)] + [(i, "t") for i in range(n)]
    return RepairableRBD(
        edges,
        {i: dict(unit) for i in range(n)},
        k={"t": k} if k > 1 else None,
        repair_crews=crews,
    )


@pytest.mark.parametrize(
    "n, lam, mu, crews, k",
    [(3, 0.1, 0.5, 1, 1), (3, 0.1, 0.5, 2, 1), (4, 0.2, 0.5, 1, 2)],
)
def test_parallel_units_match_the_machine_repair_model(n, lam, mu, crews, k):
    result = pumps(n, lam, mu, crews, k).availability(
        t_simulation=20_000.0, mc_samples=40, seed=1
    )
    interval = result.mean_availability_interval(confidence=0.999)
    # Over so long a window the start from new hardly counts.
    exact = machine_repair(n, lam, mu, crews, k)
    assert interval.lower - 2e-4 <= exact <= interval.upper + 2e-4


def system(crews=None, **extra):
    def unit(**more):
        spec = {"reliability": W([10, 1.5]), "repairability": L([0, 0.5])}
        spec.update(more)
        return spec

    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": unit(replace_cost=5.0, downtime_cost=2.0),
            "b": unit(preventive={"interval": 8.0, "duration": E([2.0])}),
            "c": unit(**extra),
        },
        downtime_cost_rate=10.0,
        repair_crews=crews,
    )


@pytest.mark.parametrize("crews", [3, 7])
def test_enough_crews_change_nothing(crews):
    run = dict(t_simulation=300.0, mc_samples=200, seed=3)
    free, crewed = system().availability(**run), system(crews).availability(
        **run
    )
    assert crewed.node_uptime == free.node_uptime
    assert crewed.system_failures == free.system_failures
    assert system(crews).cost(**run).mean == system().cost(**run).mean
    assert system(crews).mean_availability() == system().mean_availability()


def timeline(priority_of_a=0.0):
    """One crew; c fails at 1 and takes it until 6; b falls due at 1.5 and a
    at 2, and wait."""
    components = {
        "c": {"reliability": X(1.0), "repairability": X(5.0)},
        "b": {"reliability": X(1.5), "repairability": X(1.0)},
        "a": {
            "reliability": X(2.0),
            "repairability": X(1.0),
            "priority": priority_of_a,
        },
    }
    return RepairableRBD(
        [("s", "c"), ("c", "b"), ("b", "a"), ("a", "t")],
        components,
        repair_crews=1,
    )


def uptimes(rbd, window):
    result = rbd.availability(t_simulation=window, mc_samples=1, seed=0)
    return {node: round(up, 9) for node, up in result.node_uptime.items()}


def test_a_crew_takes_the_waiting_jobs_in_order():
    # First come, first served: b (due at 1.5) 6 -> 7, then a 7 -> 8.
    assert uptimes(timeline(), 7.5) == {"c": 2.0, "b": 2.0, "a": 2.0}
    # a has priority: a 6 -> 7, then b 7 -> 8.
    assert uptimes(timeline(1.0), 7.5) == {"c": 2.0, "b": 1.5, "a": 2.5}


def test_a_test_that_waits_for_a_crew_does_not_age_the_unit():
    # One crew, busy with c from 3.5 to 6.5. h's test, due at 4, waits and
    # runs 6.5 -> 7.5; its next, at 8, runs 8 -> 9. Off-line, h does not
    # age: 4.5 hours up by 9, short of its 5-hour life.
    components = {
        "c": {"reliability": X(3.5), "repairability": X(3.0)},
        "h": {
            "reliability": X(5.0),
            "repairability": "instant",
            "inspection": {"interval": 4.0, "duration": X(1.0)},
        },
    }
    edges = [("s", "c"), ("c", "h"), ("h", "t")]
    crewed = RepairableRBD(edges, components, repair_crews=1)
    assert uptimes(crewed, 9.0) == {"c": 6.0, "h": 4.5}
    # Without a limit, h fails at 6 (5 hours of use) and stays down.
    assert uptimes(RepairableRBD(edges, components), 9.0) == {
        "c": 6.0,
        "h": 5.0,
    }
    # The event-stepping API follows the same queue.
    crewed.initialize_event_queue(9.0)
    changes = [crewed.next_event()]
    while changes[-1][0] < 9.0:
        changes.append(crewed.next_event())
    assert changes == [(3.5, False), (7.5, True), (8.0, False), (9.0, False)]


def test_planned_maintenance_waits_for_a_crew_too():
    # c's repair holds the one crew from 1 to 4; m's maintenance, due at
    # 2, waits, m down, and runs 4 -> 5. c fails again at 5.
    components = {
        "c": {"reliability": X(1.0), "repairability": X(3.0)},
        "m": {
            "reliability": X(100.0),
            "repairability": X(1.0),
            "preventive": {"interval": 2.0, "duration": X(1.0)},
        },
    }
    rbd = RepairableRBD(
        [("s", "c"), ("c", "m"), ("m", "t")], components, repair_crews=1
    )
    assert uptimes(rbd, 5.5) == {"c": 2.0, "m": 2.5}


def test_waiting_is_downtime_and_is_priced():
    run = dict(t_simulation=500.0, mc_samples=300, seed=4)
    free, crewed = system().cost(**run), system(1).cost(**run)
    assert crewed.mean > free.mean
    assert (
        crewed.by_category["system_downtime"]
        > free.by_category["system_downtime"]
    )
    crewed_run = system(1).availability(**run)
    free_run = system().availability(**run)
    assert all(
        crewed_run.node_uptime[n] < free_run.node_uptime[n] for n in "abc"
    )


def test_runs_with_crews_are_reproducible_and_split_alike():
    rbd = system(1)
    run = dict(t_simulation=200.0, mc_samples=600, seed=5)
    serial = rbd.availability(**run)
    assert rbd.availability(**run).node_uptime == serial.node_uptime
    parallel = rbd.availability(**run, n_jobs=2)
    assert parallel.node_uptime == serial.node_uptime
    gain = system(2).compare(rbd, 200.0, mc_samples=300, seed=6)
    assert gain.estimate > 0.0


@pytest.mark.parametrize(
    "method",
    [
        lambda rbd: rbd.mean_availability(),
        lambda rbd: rbd.node_availability(),
        lambda rbd: rbd.system_failure_frequency(),
        lambda rbd: rbd.expected_cost_rate(),
        lambda rbd: rbd.birnbaum_importance(),
        lambda rbd: rbd.point_availability([10.0]),
        lambda rbd: rbd.mission_availability(10.0),
    ],
)
def test_the_exact_methods_refuse_while_a_job_can_wait(method):
    # Weibull lives and lognormal repairs: the crews' Markov chain (#90)
    # does not cover them, and the rest assume independent components.
    with pytest.raises(NotImplementedError, match="repair crew"):
        method(system(1))
    method(system(3))  # enough crews: nothing waits


def test_the_report_and_the_engines_follow_the_crews(monkeypatch):
    report = system(1).analysis_routes()
    assert report["mean_availability"].route == r.REFUSED
    assert "repair crew" in report["mean_availability"].reason
    assert report["availability"].route == r.SIMULATED
    # numba's own loop simulates the crews (#155); an engine of the
    # interface's version is not given them.
    plan = system(1)._stream_plan(1.0, 0, False)[0]
    assert _compiled.unsupported(system(1), plan, None) == "repair crews"
    assert _compiled.unsupported(system(1), plan, None, numba=True) is None
    monkeypatch.setattr(_compiled, "available", lambda: True)
    assert system(1).analysis_routes()["availability"].engine == "numba"
    enough = system(3).analysis_routes()["mean_availability"]
    assert enough.route != r.REFUSED


def test_a_nested_rbd_has_crews_of_its_own():
    unit = {"reliability": W([10, 1.5]), "repairability": L([0, 0.5])}
    inner = RepairableRBD(
        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
        {"x": dict(unit), "y": dict(unit)},
        repair_crews=1,
    )
    # The outer crew works on q only (the nested RBD is not its job), so
    # it never lacks one, but the nested RBD's own crew does.
    outer = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {"p": inner, "q": dict(unit)},
        repair_crews=1,
    )
    assert not outer._crews_limited()
    with pytest.raises(NotImplementedError, match="repair crew"):
        outer.mean_availability()
    result = outer.availability(t_simulation=300.0, mc_samples=50, seed=2)
    assert 0.0 < result.mean_availability_interval().estimate < 1.0


def test_crews_and_priorities_are_checked_and_saved():
    for crews in (0, -1, 1.5, True, "2"):
        with pytest.raises((ValueError, TypeError)):
            system(crews)
    for priority in ("high", np.inf, True):
        with pytest.raises(ValueError, match="priority"):
            system(1, priority=priority)
    rbd = system(1, priority=2)
    back = RBD.from_json(rbd.to_json())
    assert back.repair_crews == 1 and back._priority == {"c": 2.0}
    run = dict(t_simulation=200.0, mc_samples=50, seed=7)
    assert (
        back.availability(**run).node_uptime
        == rbd.availability(**run).node_uptime
    )
    assert "repair_crews" not in system().to_dict()
