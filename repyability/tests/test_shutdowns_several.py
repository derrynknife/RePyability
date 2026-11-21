"""A block renewed at several shutdowns (a list of names in its task's
``"shutdown"``): block replacement at every start of each, each renewal
taking its own shut's length (UC-59).

The references are today's single shutdown (a list of one name is it),
one shutdown at the merged times (two shutdowns of one length are it), a
memoryless unit's two-state chain between renewals of their own lengths
written out here, and the simulation.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import _compiled, _curves, _event_loop

E = surv.Exponential.from_params
W = surv.Weibull.from_params
EDGES = [("s", "a"), ("a", "t")]
#: Standard errors a simulated estimate may be from the exact value.
Z = 4.0


def one(life, shutdowns, names, repair=None, **task):
    """One block renewed at the shutdowns ``names``, each holding it
    alone off line."""
    return RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": life,
                "repairability": repair or E([0.05]),
                "repair_cost": 10.0,
                "preventive": {"shutdown": names, "cost": 4.0, **task},
            }
        },
        shutdowns=shutdowns,
        downtime_cost_rate=2.0,
    )


#: A stop every 100 for 2 and an overhaul every 1000 for 20, from 550.
TWO = {
    "stop": {"period": 100.0, "duration": 2.0},
    "overhaul": {"period": 1000.0, "offset": 550.0, "duration": 20.0},
}


def plain(**more):
    return dict(control_variate=False, conditional=False, **more)


# -- one shutdown -------------------------------------------------------------


@pytest.mark.parametrize(
    "shutdown",
    [
        {"period": 300.0, "offset": 100.0, "duration": 5.0, "cost": 20.0},
        {"times": [120.0, 400.0, 610.0], "duration": 8.0, "cost": 20.0},
    ],
)
def test_a_list_of_one_shutdown_is_that_shutdown(shutdown):
    life = W([300, 2.0])
    named = one(life, {"line": shutdown}, "line")
    listed = one(life, {"line": shutdown}, ["line"])
    t = np.linspace(0.0, 900.0, 181)
    np.testing.assert_allclose(
        listed.point_availability(t), named.point_availability(t), atol=1e-10
    )
    for T in (260.0, 900.0):
        assert listed.mission_availability(T) == pytest.approx(
            named.mission_availability(T), abs=1e-10
        )
        assert listed.expected_cost(T).mean == pytest.approx(
            named.expected_cost(T).mean, rel=1e-10
        )
    mine = listed.expected_npv(900.0, discount_rate=1e-3, value_rate=5.0)
    theirs = named.expected_npv(900.0, discount_rate=1e-3, value_rate=5.0)
    assert mine.npv == pytest.approx(theirs.npv, rel=1e-10)


# -- two shutdowns ------------------------------------------------------------


def test_two_shutdowns_of_one_length_are_one_at_the_merged_times():
    life, d = W([300, 2.0]), 6.0
    shuts = {
        "stop": {"period": 150.0, "duration": d, "cost": 7.0},
        "outage": {"times": [220.0, 500.0], "duration": d, "cost": 7.0},
    }
    two = one(life, shuts, ["stop", "outage"])
    merged = sorted([150.0 * k for k in range(1, 7)] + [220.0, 500.0])
    single = one(
        life, {"all": {"times": merged, "duration": d, "cost": 7.0}}, "all"
    )
    t = np.linspace(0.0, 880.0, 177)
    np.testing.assert_allclose(
        two.point_availability(t), single.point_availability(t), atol=1e-10
    )
    for T in (300.0, 880.0):
        assert two.mission_availability(T) == pytest.approx(
            single.mission_availability(T), abs=1e-10
        )
        mine, theirs = two.expected_events(T), single.expected_events(T)
        for key in ("node_failures", "node_preventive", "node_downtime"):
            assert getattr(mine, key)["a"] == pytest.approx(
                getattr(theirs, key)["a"], abs=1e-10
            )
        assert mine.shutdowns == theirs.shutdowns
    mine = two.expected_npv(880.0, discount_rate=1e-3, value_rate=5.0)
    theirs = single.expected_npv(880.0, discount_rate=1e-3, value_rate=5.0)
    assert mine.npv == pytest.approx(theirs.npv, rel=1e-10)


def memoryless(t, renewals, lam, mu):
    """The point availability of a unit of failure rate ``lam`` and repair
    rate ``mu``, new at 0, renewed at each ``(time, length)`` of
    ``renewals`` if it is up (back, as new, ``length`` later; one in a
    repair carries on), and held off line in each ``[time, time +
    length)``: a two-state chain between them."""
    total = lam + mu

    def from_up(s):
        return mu / total + lam / total * np.exp(-total * s)

    def from_down(s):
        return mu / total * (1.0 - np.exp(-total * s))

    def value(parts, x):
        return sum(
            w * np.where(x >= s, from_up(x - s) if up else from_down(x - s), 0)
            for w, s, up in parts
        )

    parts = [(1.0, 0.0, True)]
    out = np.zeros_like(t)
    held = np.zeros(t.shape, dtype=bool)
    edges = [0.0] + [s for s, _ in renewals] + [math.inf]
    for k, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (t >= start) & (t < end)
        out[inside] = value(parts, t[inside])
        if k == len(renewals):
            break
        length = renewals[k][1]
        up = float(value(parts, np.array([end]))[0])
        assert all(s <= end for _, s, _ in parts), "a renewal past a gap"
        parts = [(up, end + length, True), (1.0 - up, end, False)]
        held |= (t >= end) & (t < end + length)
    return np.where(held, 0.0, out)


def test_an_exponential_block_at_two_shutdowns_against_its_closed_form():
    lam, mu = 1 / 200.0, 1 / 20.0
    rbd = one(E([lam]), TWO, ["stop", "overhaul"], repair=E([mu]))
    renewals = [(100.0 * k, 2.0) for k in range(1, 17)]
    renewals = sorted(renewals + [(550.0, 20.0), (1550.0, 20.0)])
    # (Not within a grid step of the end of a renewal, where the curve
    # bends between two points of its grid.)
    t = np.array([0.0, 50.0, 99.9, 101.0, 103.0, 180.0, 549.9, 551.0, 571.0])
    t = np.concatenate([t, [600.0, 799.0, 1000.0, 1300.0, 1560.0, 1575.0]])
    expected = memoryless(t, renewals, lam, mu)
    np.testing.assert_allclose(rbd.point_availability(t), expected, atol=2e-6)
    # Each renewal, at its time, of a unit that is up just before.
    up = [
        float(memoryless(np.array([s - 1e-9]), renewals, lam, mu)[0])
        for s, _ in renewals
        if s < 1200.0
    ]
    events = rbd.expected_events(1200.0)
    assert events.node_preventive["a"] == pytest.approx(sum(up), abs=2e-5)
    assert events.shutdowns == 12


def test_the_simulation_agrees_from_new_and_from_a_state():
    rbd, T = one(W([300, 2.0]), TWO, ["stop", "overhaul"]), 1500.0
    plan = _event_loop._stream_plan(rbd, T, 0, False)[0]
    assert _compiled.unsupported(rbd, plan, None, numba=True) == "shutdowns"
    run = rbd.availability(T, mc_samples=20_000, seed=3, **plain())
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(T)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error
    assert interval.standard_error < 5e-4
    cost = rbd.expected_cost(T)
    assert abs(run.cost.sample_mean - cost.mean) <= Z * run.cost.sample_se
    state = {"a": NodeState(age=150.0)}
    run = rbd.availability(
        T, mc_samples=20_000, seed=4, state=state, **plain()
    )
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(T, state=state)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error
    cost = rbd.expected_cost(T, state=state)
    assert abs(run.cost.sample_mean - cost.mean) <= Z * run.cost.sample_se


def test_at_given_times_with_a_repair_going_on_agrees_with_the_simulation():
    shuts = {
        "short": {"times": [80.0, 300.0, 520.0], "duration": 3.0},
        "long": {"times": [200.0, 410.0], "duration": 25.0, "cost": 50.0},
    }
    rbd = one(W([250, 2.5]), shuts, ["short", "long"], repair=W([15, 1.5]))
    state = {"a": NodeState(alive=False, down_for=4.0)}
    T = 700.0
    run = rbd.availability(
        T, mc_samples=20_000, seed=11, state=state, **plain()
    )
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(T, state=state)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error
    cost = rbd.expected_cost(T, state=state)
    assert abs(run.cost.sample_mean - cost.mean) <= Z * run.cost.sample_se
    assert run.shutdowns == 5 == rbd.expected_events(T).shutdowns


# -- what is refused, and saving ----------------------------------------------


def test_two_shutdowns_that_start_together_are_refused():
    shuts = {
        "stop": {"period": 100.0, "duration": 2.0},
        "outage": {"times": [250.0, 400.0], "duration": 0.0},
    }
    with pytest.raises(ValueError) as error:
        one(W([300, 2.0]), shuts, ["stop", "outage"])
    assert str(error.value) == (
        "Component 'a' is renewed at shutdowns 'stop' and 'outage', which "
        "both start at 400: which one's renewal is done there is not said. "
        "Move one (its offset or times), or make them one shutdown."
    )


@pytest.mark.parametrize(
    "task, what",
    [
        ({"older_than": 100.0}, "older_than"),
        ({"restore_to": 50.0}, "restore_to"),
        ({"policy": "condition", "threshold": 0.1}, "threshold"),
        ({"policy": "age"}, "policy 'age'"),
    ],
)
def test_a_task_kept_unless_due_or_restoring_is_at_one_shutdown(task, what):
    with pytest.raises(ValueError) as error:
        one(W([300, 2.0]), TWO, ["stop", "overhaul"], **task)
    assert str(error.value) == (
        "Component 'a' is renewed at several shutdowns ('stop', "
        "'overhaul'), at every start of each (the 'block' policy); "
        f"{what} is taken at one shutdown: give one shutdown, or leave it "
        "out."
    )


def test_the_long_run_and_its_state_are_refused():
    rbd = one(W([300, 2.0]), TWO, ["stop", "overhaul"])
    message = (
        "Component 'a' is renewed at several shutdowns ('stop', "
        "'overhaul'), whose renewals the long-run methods do not follow: "
        "ask for its values over a horizon (expected_npv(horizon), "
        "expected_cost(T), mission_availability(T), point_availability(t)), "
        "or estimate the long run by simulation over a long horizon, with "
        "availability() or cost()."
    )
    with pytest.raises(NotImplementedError) as error:
        rbd.mean_availability()
    assert str(error.value) == message
    routes = rbd.analysis_routes()
    assert routes["expected_cost_rate"].reason == message
    assert routes["replacement_cycle"].reason == message
    assert routes["expected_npv"].route == "numerical"
    route, how = _curves._node_over_time(rbd, "a")
    assert route == "numerical" and "several shutdowns" in how
    with pytest.raises(ValueError, match="no long-run state"):
        rbd.mission_availability(
            100.0, state={"a": NodeState(stationary=True)}
        )
    with pytest.raises(ValueError, match="which renewal it is in now"):
        rbd.availability(
            100.0,
            mc_samples=10,
            state={"a": NodeState(alive=False, maintenance=True)},
        )


def test_several_shutdowns_are_saved_and_loaded():
    rbd = one(W([300, 2.0]), TWO, ["stop", "overhaul"])
    back = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    assert back._preventive["a"].shutdown == ("stop", "overhaul")
    assert back.mission_availability(900.0) == rbd.mission_availability(900.0)
