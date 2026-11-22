"""A component replaced at the times of a plan (``"replace_at"``): block
replacement at any list of times, and after the last no more.

The references are block replacement on a period, which equally spaced
times are; closed forms (a memoryless unit between its replacements, a
unit that fails once at most, events at exact times discounted there);
and the simulation.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import _compiled

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params
FIXED = surv.ExactEventTime.from_params
EDGES = [("s", "a"), ("a", "t")]


def one(life, repair, preventive, **more):
    """A system of one component with a preventive task."""
    spec = {
        "reliability": life,
        "repairability": repair,
        "preventive": preventive,
        **more,
    }
    return RepairableRBD(EDGES, {"a": spec})


# -- equally spaced times are block replacement -------------------------------


@pytest.mark.parametrize(
    "repair, duration",
    [
        (LN([3, 0.5]), None),
        (LN([3, 0.5]), W([6, 2])),
        ("instant", FIXED([12.0])),
        (E([0.02]), FIXED([5.0])),
    ],
)
def test_equally_spaced_times_are_block_replacement(repair, duration):
    life = W([1000, 2.5])
    task = {"cost": 50.0}
    if duration is not None:
        task["duration"] = duration
    costs = {"repair_cost": 30.0, "downtime_cost": 2.0}
    block = one(
        life, repair, {"interval": 400.0, "policy": "block", **task}, **costs
    )
    times = [400.0 * k for k in range(1, 6)]
    plan = one(life, repair, {"replace_at": times, **task}, **costs)
    t = np.linspace(0.0, 1900.0, 381)
    np.testing.assert_allclose(
        plan.point_availability(t),
        block.point_availability(t),
        rtol=0,
        atol=1e-10,
    )
    for T in (350.0, 400.0, 1234.5, 1900.0):
        assert plan.mission_availability(T) == pytest.approx(
            block.mission_availability(T), abs=1e-10
        )
        mine, theirs = plan.expected_events(T), block.expected_events(T)
        for key in ("node_failures", "node_preventive", "node_downtime"):
            assert getattr(mine, key)["a"] == pytest.approx(
                getattr(theirs, key)["a"], abs=1e-10
            )
        assert mine.system_planned_outages == pytest.approx(
            theirs.system_planned_outages, abs=1e-10
        )
        assert plan.expected_cost(T).mean == pytest.approx(
            block.expected_cost(T).mean, rel=1e-10
        )
        assert plan.expected_cost(T, discount_rate=1e-3).mean == pytest.approx(
            block.expected_cost(T, discount_rate=1e-3).mean, rel=1e-10
        )
    mine = plan.expected_npv(
        1900.0, discount_rate=1e-3, value_rate=7.0, periods=5
    )
    theirs = block.expected_npv(
        1900.0, discount_rate=1e-3, value_rate=7.0, periods=5
    )
    assert mine.npv == pytest.approx(theirs.npv, rel=1e-10)
    assert mine.cost == pytest.approx(theirs.cost, rel=1e-10)


def test_no_times_before_the_horizon_change_nothing():
    life, repair = W([1000, 2.5]), LN([3, 0.5])
    plain = RepairableRBD(
        EDGES, {"a": {"reliability": life, "repairability": repair}}
    )
    plan = one(life, repair, {"replace_at": [5000.0], "cost": 9.0})
    t = np.linspace(0.0, 3000.0, 61)
    # The plain unit's curve is on a grid of its own: to its precision.
    np.testing.assert_allclose(
        plan.point_availability(t), plain.point_availability(t), atol=2e-6
    )
    events = plan.expected_events(3000.0)
    assert events.node_preventive["a"] == 0.0
    assert events.node_failures["a"] == pytest.approx(
        plain.expected_events(3000.0).node_failures["a"], abs=2e-6
    )


# -- closed forms -------------------------------------------------------------


def memoryless(t, times, lam, mu, d):
    """The point availability of a unit of failure rate ``lam`` and repair
    rate ``mu``, new at 0, replaced in a time ``d`` at each of ``times``
    if it is up: a two-state chain between them, the units replaced back
    as new ``d`` later and those in a repair carrying on."""
    total = lam + mu

    def from_up(s):
        return mu / total + lam / total * np.exp(-total * s)

    def from_down(s):
        return mu / total * (1.0 - np.exp(-total * s))

    # (probability, when it starts, up or down then) of each part.
    parts = [(1.0, 0.0, True)]
    out = np.zeros_like(t)
    edges = list(times) + [math.inf]
    start = 0.0
    for end in edges:
        inside = (t >= start) & (t < end)
        for weight, since, up in parts:
            s = t[inside] - since
            value = from_up(s) if up else from_down(s)
            out[inside] += weight * np.where(s >= 0.0, value, 0.0)
        if not math.isfinite(end):
            break
        up_at = sum(
            w * (from_up(end - s) if up else from_down(end - s))
            for w, s, up in parts
            if end >= s
        )
        waiting = sum(w for w, s, _ in parts if end < s)
        assert waiting == 0.0, "a replacement longer than a gap"
        parts = [(up_at, end + d, True), (1.0 - up_at, end, False)]
        start = end
    return out


@pytest.mark.parametrize("d", [0.0, 7.0])
def test_an_exponential_unit_against_its_closed_form(d):
    lam, mu = 1 / 200.0, 1 / 20.0
    times = [130.0, 310.0, 365.0, 640.0]
    task = {"replace_at": times, "cost": 11.0}
    if d:
        task["duration"] = FIXED([d])
    rbd = one(E([lam]), E([mu]), task)
    # (Not within a grid step of the end of a fixed replacement, where
    # the curve bends between two points of its grid.)
    t = np.array([0.0, 50.0, 129.9, 130.0, 133.0, 138.0, 200.0, 310.0])
    t = np.concatenate([t, [316.0, 364.0, 365.0, 373.0, 500.0, 640.0, 900.0]])
    expected = memoryless(t, times, lam, mu, d)
    np.testing.assert_allclose(rbd.point_availability(t), expected, atol=2e-6)
    # One replacement at T: up to it the chain from new, then the units
    # replaced as new again (none, replaced in no time: it is memoryless).
    single = one(E([lam]), E([mu]), {**task, "replace_at": [130.0]})
    np.testing.assert_allclose(
        single.point_availability(t),
        memoryless(t, [130.0], lam, mu, d),
        atol=2e-6,
    )
    up_at = memoryless(np.array([130.0 - 1e-9]), [], lam, mu, d)[0]
    events = single.expected_events(400.0)
    assert events.node_preventive["a"] == pytest.approx(up_at, abs=2e-6)
    # Its replacement is charged once, at 130, and discounted from there.
    rate = 2e-3
    assert single.expected_cost(400.0, discount_rate=rate).by_category[
        "preventive"
    ] == pytest.approx(11.0 * up_at * math.exp(-rate * 130.0), rel=1e-5)


def test_a_weibull_unit_that_fails_once_against_its_closed_form():
    # Its repairs outlast the horizon, so a unit fails once at most: it is
    # up before T if its first life is, and after T if that life reached T
    # (when it is replaced, in no time) and the second has lasted since.
    life, T = W([300, 2.2]), 180.0
    rbd = one(
        life,
        FIXED([10_000.0]),
        {"replace_at": [T], "cost": 40.0},
        repair_cost=15.0,
    )
    t = np.array([0.0, 60.0, 179.0, 180.0, 181.0, 300.0, 520.0])
    R = life.sf
    expected = np.where(t < T, R(t), R(T) * R(np.maximum(t - T, 0.0)))
    np.testing.assert_allclose(rbd.point_availability(t), expected, atol=2e-6)
    events = rbd.expected_events(520.0)
    failures = float(life.ff(T) + R(T) * life.ff(520.0 - T))
    assert events.node_failures["a"] == pytest.approx(failures, abs=2e-6)
    assert events.node_preventive["a"] == pytest.approx(float(R(T)), abs=2e-6)
    cost = rbd.expected_cost(520.0)
    assert cost.mean == pytest.approx(
        15.0 * failures + 40.0 * float(R(T)), rel=1e-5
    )


def test_replacements_are_discounted_at_their_times():
    # Repaired in no time, the unit is up at every time of the plan, and
    # replaced there; memoryless, it fails at its rate throughout.
    lam, rate, horizon = 1 / 500.0, 1.5e-3, 1000.0
    times = [120.0, 333.0, 480.0, 910.0, 1500.0]
    rbd = one(
        E([lam]),
        "instant",
        {"replace_at": times, "cost": 100.0},
        repair_cost=8.0,
    )
    result = rbd.expected_npv(horizon, discount_rate=rate, value_rate=3.0)
    planned = sum(100.0 * math.exp(-rate * t) for t in times if t < horizon)
    annuity = (1.0 - math.exp(-rate * horizon)) / rate
    # (The unit's chance of being up at each time is on its grid.)
    assert result.cost_by_category["preventive"] == pytest.approx(
        planned, rel=1e-7
    )
    assert result.cost_by_category["repair"] == pytest.approx(
        8.0 * lam * annuity, rel=1e-6
    )
    assert result.npv == pytest.approx(
        3.0 * annuity - planned - 8.0 * lam * annuity, rel=1e-6
    )
    # By period, each replacement falls in its own.
    by_period = rbd.expected_npv(
        horizon, discount_rate=rate, periods=[200.0, 400.0, 1000.0]
    )
    rows = {
        r["period"]: -r["present_value"]
        for r in by_period.cash_flows
        if r["category"] == "preventive"
    }
    assert rows[1] == pytest.approx(100.0 * math.exp(-rate * 120.0), rel=1e-7)
    assert rows[2] == pytest.approx(100.0 * math.exp(-rate * 333.0), rel=1e-7)
    assert rows[3] == pytest.approx(
        100.0 * (math.exp(-rate * 480.0) + math.exp(-rate * 910.0)), rel=1e-7
    )


def test_stretches_of_any_lengths_agree_with_a_finer_grid(monkeypatch):
    # Uneven stretches, each on a grid of its own, with repairs and
    # replacements carried from one onto the next: against the same on
    # grids four times as fine.
    import repyability.rbd._block_replacement as block

    def build():
        return one(
            W([400, 1.8]),
            LN([3.0, 0.6]),
            {
                "replace_at": [90.0, 130.0, 317.0, 700.0, 760.0],
                "duration": W([8, 1.5]),
            },
        )

    t = np.linspace(0.0, 1200.0, 241)
    coarse = build().point_availability(t)
    events = build().expected_events(1200.0)
    monkeypatch.setattr(block, "_MIN_STEPS", 4 * block._MIN_STEPS)
    monkeypatch.setattr(
        block, "_STEPS_PER_SPREAD", 4 * block._STEPS_PER_SPREAD
    )
    fine = build().point_availability(t)
    finer = build().expected_events(1200.0)
    np.testing.assert_allclose(coarse, fine, atol=2e-6)
    for key in ("node_failures", "node_preventive"):
        assert getattr(events, key)["a"] == pytest.approx(
            getattr(finer, key)["a"], abs=5e-6
        )


# -- the simulation -----------------------------------------------------------


def plant():
    """Two units in parallel before a third, each on a plan of its own."""
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": {
                "reliability": W([300, 2.0]),
                "repairability": LN([2.5, 0.6]),
                "preventive": {
                    "replace_at": [150.0, 420.0, 600.0],
                    "duration": W([6, 2]),
                    "cost": 25.0,
                },
                "repair_cost": 60.0,
            },
            "b": {
                "reliability": W([250, 1.5]),
                "repairability": E([0.1]),
                "preventive": {"replace_at": [200.0, 500.0], "cost": 20.0},
                "repair_cost": 40.0,
            },
            "c": {
                "reliability": W([900, 3.0]),
                "repairability": LN([2.0, 0.5]),
                "preventive": {
                    "replace_at": [400.0],
                    "duration": FIXED([10.0]),
                    "cost": 80.0,
                },
                "repair_cost": 200.0,
            },
        },
        downtime_cost_rate=5.0,
    )


def se(run) -> float:
    """The standard error of a run's mean availability."""
    return float(run.mean_availability_interval().standard_error)


def test_the_simulation_agrees_with_the_exact_values():
    rbd, T = plant(), 800.0
    run = rbd.availability(
        T, mc_samples=20_000, seed=11, control_variate=False
    )
    exact = rbd.mission_availability(T)
    assert abs(run.sample_mean_availability - exact) < 4.0 * se(run)
    assert se(run) < 5e-4
    cost = rbd.expected_cost(T)
    assert abs(run.cost.mean - cost.mean) < 4.0 * run.cost.mean_se
    assert run.cost.mean_se < 0.005 * cost.mean
    # By default the run's means are the exact ones.
    default = rbd.availability(T, mc_samples=200, seed=11)
    assert default.mean_availability == pytest.approx(exact, abs=1e-9)


def test_the_simulation_replaces_at_the_times_and_no_more():
    rbd = one(
        W([1e9, 2.0]),
        "instant",
        {"replace_at": [100.0, 250.0], "duration": FIXED([10.0])},
    )
    lines = rbd.simulate_timelines(1000.0, mc_samples=3, seed=2)
    history = lines.components["a"]
    np.testing.assert_array_equal(history.planned_outages, [2, 2, 2])
    np.testing.assert_allclose(history.downtime, 20.0)
    np.testing.assert_allclose(history.point_availability([105.0, 255.0]), 0.0)
    np.testing.assert_allclose(history.point_availability([99.0, 111.0]), 1.0)
    assert rbd.mission_availability(1000.0) == pytest.approx(0.98, abs=1e-9)


def test_a_compiled_engine_leaves_a_plan_to_the_python_loop():
    rbd = plant()
    reason = _compiled._unsupported_maintenance(rbd, None)
    assert reason == "replacement at the times of a plan"
    route = rbd.analysis_routes()["availability"]
    assert route.engine == "python"
    assert "replacement at the times of a plan" in route.engine_reason


# -- from a state -------------------------------------------------------------


def test_from_a_state_the_plan_is_counted_from_now():
    rbd, T = plant(), 700.0
    state = {
        "a": NodeState(age=120.0),
        "b": NodeState(alive=False, down_for=4.0),
        "c": NodeState(age=300.0),
    }
    exact = rbd.mission_availability(T, state=state)
    assert exact != pytest.approx(rbd.mission_availability(T), abs=1e-4)
    run = rbd.availability(
        T, mc_samples=20_000, seed=5, state=state, control_variate=False
    )
    assert abs(run.sample_mean_availability - exact) < 4.0 * se(run)
    cost = rbd.expected_cost(T, state=state)
    assert abs(run.cost.mean - cost.mean) < 4.0 * run.cost.mean_se
    # Memoryless and up, a unit's state changes nothing.
    lam, mu = 1 / 200.0, 1 / 20.0
    task = {"replace_at": [130.0, 310.0], "duration": FIXED([7.0])}
    memory = one(E([lam]), E([mu]), task)
    t = np.array([10.0, 129.0, 131.0, 138.0, 250.0, 311.0, 400.0])
    np.testing.assert_allclose(
        memory.point_availability(t, state={"a": NodeState(age=55.0)}),
        memoryless(t, [130.0, 310.0], lam, mu, 7.0),
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "state, message",
    [
        (
            NodeState(since_scheduled=10.0),
            "Component 'a' is replaced at the times of its plan "
            "(replace_at), counted from now, which place its "
            "replacements: leave its since_scheduled out.",
        ),
        (
            "stationary",
            "Component 'a' is replaced at the times of a plan "
            "(replace_at), which ends, so it has no long-run state: give "
            "its age now (NodeState(age=...)), or leave it out (new).",
        ),
    ],
)
def test_a_plan_takes_no_place_on_a_calendar_and_no_long_run_state(
    state, message
):
    rbd = one(W([300, 2.0]), E([0.1]), {"replace_at": [100.0]})
    with pytest.raises(ValueError) as error:
        rbd.point_availability([50.0], state={"a": state})
    assert str(error.value) == message
    with pytest.raises(ValueError) as error:
        rbd.availability(200.0, mc_samples=10, seed=1, state={"a": state})
    assert str(error.value) == message


# -- refusals -----------------------------------------------------------------

NO_LONG_RUN = (
    "Component 'a' is replaced at the times of a plan (replace_at), which "
    "ends, so the diagram has no long run with it: ask for its values over "
    "a horizon (expected_npv(horizon), expected_cost(T), "
    "mission_availability(T), point_availability(t)), or give the "
    "component an interval, block replacement for ever, for the long run."
)


@pytest.mark.parametrize(
    "call",
    [
        lambda rbd: rbd.mean_availability(),
        lambda rbd: rbd.mean_unavailability(),
        lambda rbd: rbd.node_availability(),
        lambda rbd: rbd.system_failure_frequency(),
        lambda rbd: rbd.expected_cost_rate(),
        lambda rbd: rbd.birnbaum_importance(),
        lambda rbd: rbd.replacement_cycle("a"),
        lambda rbd: rbd.parameter_sensitivity(),
    ],
)
def test_the_long_run_is_refused_saying_what_to_ask_for(call):
    rbd = one(
        W([300, 2.0]), E([0.1]), {"replace_at": [100.0]}, repair_cost=1.0
    )
    with pytest.raises(NotImplementedError) as error:
        call(rbd)
    assert str(error.value) == NO_LONG_RUN
    for name in ("mean_availability", "expected_cost_rate"):
        route = rbd.analysis_routes()[name]
        assert (route.route, route.reason) == ("refused", NO_LONG_RUN)
    for name in ("point_availability", "expected_cost", "expected_npv"):
        assert rbd.analysis_routes()[name].route == "numerical"


def test_spares_and_the_plan_against_the_actual_say_what_to_do():
    rbd = one(
        W([300, 2.0]), E([0.1]), {"replace_at": [100.0]}, repair_cost=1.0
    )
    with pytest.raises(NotImplementedError, match="replace_at.*simulat"):
        rbd.spares_demand(200.0)
    route = rbd.analysis_routes()["plan_vs_actual"]
    assert route.route == "refused"
    assert "counted from today, with expected_npv(state=)" in route.reason


@pytest.mark.parametrize(
    "task, message",
    [
        (
            {"replace_at": [100.0], "interval": 50.0},
            "Component 'a': give its preventive task an interval (replaced "
            "every so often, for ever) or replace_at (replaced at those "
            "times and no more), not both.",
        ),
        (
            {"replace_at": [100.0], "policy": "age"},
            "Component 'a': replace_at replaces the unit at each of its "
            "times if it is up, which is the 'block' policy at given "
            "times, so its policy cannot be 'age': leave the policy out, "
            "or give that policy an interval and no replace_at.",
        ),
        (
            {"replace_at": [100.0, 100.0]},
            "Component 'a': replace_at must be the times at which the unit "
            "is replaced, in the diagram's time from 0: positive, finite "
            "and increasing, got [100.0, 100.0].",
        ),
        (
            {"replace_at": [0.0, 100.0]},
            "Component 'a': replace_at must be the times at which the unit "
            "is replaced, in the diagram's time from 0: positive, finite "
            "and increasing, got [0.0, 100.0].",
        ),
        (
            {"replace_at": []},
            "Component 'a': replace_at must be the times at which the unit "
            "is replaced, in the diagram's time from 0: positive, finite "
            "and increasing, got [].",
        ),
        (
            {"replace_at": ["100"]},
            "Component 'a': replace_at must be the times at which the unit "
            "is replaced, in the diagram's time from 0: positive, finite "
            "and increasing, got ['100'].",
        ),
        (
            {"replace_at": [100.0], "opportunity": 50.0},
            "Component 'a': an opportunity applies only to the 'age' and "
            "'calendar' policies (a unit renewed early at a stop of its "
            "maintenance group).",
        ),
    ],
)
def test_a_plan_s_spec_is_checked(task, message):
    with pytest.raises(ValueError) as error:
        one(W([300, 2.0]), E([0.1]), task)
    assert str(error.value) == message


def test_a_plan_is_not_given_with_a_shutdown():
    with pytest.raises(ValueError) as error:
        RepairableRBD(
            EDGES,
            {
                "a": {
                    "reliability": W([300, 2.0]),
                    "repairability": E([0.1]),
                    "preventive": {"shutdown": "s1", "replace_at": [100.0]},
                }
            },
            shutdowns={"s1": {"period": 100.0, "duration": 1.0}},
        )
    assert str(error.value) == (
        "Component 'a': its preventive task is done at shutdown 's1', "
        "whose times apply, so it takes no replace_at: give the times to "
        "the shutdown."
    )


# -- saving, levers, operating time -------------------------------------------


def test_a_plan_is_saved_with_the_diagram():
    rbd = plant()
    saved = json.loads(json.dumps(rbd.to_dict()))
    again = RepairableRBD.from_dict(saved)
    assert again._preventive["a"].times == (150.0, 420.0, 600.0)
    assert again._preventive["a"].plan == (150.0, 420.0, 600.0)
    t = np.linspace(0.0, 800.0, 33)
    np.testing.assert_allclose(
        again.point_availability(t), rbd.point_availability(t), atol=1e-12
    )
    assert again.expected_cost(800.0).mean == pytest.approx(
        rbd.expected_cost(800.0).mean, rel=1e-12
    )


def test_each_time_of_a_plan_is_a_lever():
    rbd = plant()
    levers = {
        (lever.key, lever.name): lever
        for lever in rbd.levers()
        if "replace_at" in lever.name
    }
    assert sorted(name for key, name in levers if key == "a") == [
        "preventive.replace_at.0",
        "preventive.replace_at.1",
        "preventive.replace_at.2",
    ]
    middle = levers[("a", "preventive.replace_at.1")]
    assert (middle.value, middle.bounds) == (420.0, (150.0, 600.0))
    last = levers[("b", "preventive.replace_at.1")]
    assert (last.value, last.bounds) == (500.0, (200.0, math.inf))
    moved = rbd.with_levers({("a", "preventive.replace_at.1"): 450.0})
    assert moved._preventive["a"].times == (150.0, 450.0, 600.0)
    with pytest.raises(ValueError, match="between 150 and 600"):
        rbd.with_levers({("a", "preventive.replace_at.1"): 600.0})
    # Over a window, a time's sensitivity is the system's own difference.
    window = 800.0
    found = rbd.parameter_sensitivity(window=window)["a"]
    step = 0.5
    name = "preventive.replace_at.1"
    high = rbd.with_levers({("a", name): 420.0 + step})
    low = rbd.with_levers({("a", name): 420.0 - step})
    slope = (
        high.mission_availability(window) - low.mission_availability(window)
    ) / (2.0 * step)
    assert found[name] == pytest.approx(slope, rel=0.02, abs=1e-9)


def test_a_plan_counted_in_operating_time_goes_onto_the_calendar():
    life, repair = W([300, 2.0]), E([0.1])
    operating = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": life,
                "repairability": repair,
                "duty": 0.5,
                "preventive": {
                    "replace_at": [100.0, 250.0],
                    "basis": "operating",
                },
            }
        },
    )
    assert operating._preventive["a"].times == (200.0, 500.0)
