"""Shutdowns: outages planned on a period (or at given times) that last a
fixed time and hold the system, or a named set of its blocks, off line,
with the blocks' work done at them (UC-23, UC-24). The exact values are
checked against closed forms written here (an exponential block in a
system that is shut every period), against the existing block and
inspection machinery the bound tasks are translated into, and against
the simulation; the simulation against the closed forms too.
"""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.rbd import _shutdowns, routes

E = surv.Exponential.from_params
W = surv.Weibull.from_params
FIXED = surv.ExactEventTime.from_params
EDGES = [("s", "a"), ("a", "t")]
LAM, MU = 0.01, 0.5
P, D, OFFSET = 100.0, 10.0, 20.0
#: Standard errors a simulated estimate may be from the exact value.
Z = 4.0


def shut(**more):
    return {"period": P, "offset": OFFSET, "duration": D, **more}


def one(spec=None, shutdown=None, **options):
    """One exponential block (rate LAM, repaired at rate MU), shut every P
    for D from OFFSET."""
    return RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": E([LAM]),
                "repairability": E([MU]),
                **(spec or {}),
            }
        },
        shutdowns={"line": shutdown or shut()},
        **options,
    )


def own(t):
    """The block's own point availability from new."""
    total = LAM + MU
    return MU / total + LAM / total * np.exp(-total * np.asarray(t))


def inside(t, offset=OFFSET):
    t = np.asarray(t, dtype=float)
    return (t >= offset) & ((t - offset) % P < D)


def plain(**more):
    return dict(control_variate=False, conditional=False, **more)


# -- closed forms -------------------------------------------------------------


def test_the_long_run_of_a_shut_system_is_its_share_of_the_time_up():
    rbd = one()
    a = MU / (LAM + MU)
    assert rbd.mean_availability() == pytest.approx((1 - D / P) * a, 1e-12)
    assert rbd.mean_unavailability() == pytest.approx(
        1 - (1 - D / P) * a, 1e-12
    )
    # It fails only while it is on line, and each shut that finds it up is
    # a planned outage.
    assert rbd.system_failure_frequency() == pytest.approx(
        LAM * a * (1 - D / P), 1e-12
    )
    assert rbd.mean_up_time() == pytest.approx(
        (1 - D / P) * a / (LAM * a * (1 - D / P) + a / P), 1e-12
    )
    assert rbd.node_availability()["a"] == pytest.approx((1 - D / P) * a)
    # The shut's cost, every period.
    priced = one(shutdown=shut(cost=5.0))
    assert priced.expected_cost_rate() == pytest.approx(5.0 / P, 1e-12)


def test_the_point_availability_is_held_at_0_inside_a_shut():
    rbd = one()
    x = np.array([5.0, 19.999, 20.0, 25.0, 29.999, 30.0, 70.0, 125.0, 931.0])
    expected = np.where(inside(x), 0.0, own(x))
    np.testing.assert_allclose(rbd.point_availability(x), expected, atol=1e-6)
    np.testing.assert_allclose(
        rbd.point_unavailability(x), 1.0 - expected, atol=1e-6
    )


def test_the_expected_events_count_each_shut_that_finds_the_system_up():
    rbd = one(shutdown=shut(cost=5.0), spec={"repair_cost": 2.0})
    t = 1000.0
    starts = OFFSET + P * np.arange(10)
    events = rbd.expected_events(t)
    # Each start is a planned outage with the chance it finds it up.
    assert events.system_planned_outages == pytest.approx(
        own(starts).sum(), 1e-6
    )
    assert events.shutdowns == 10.0
    # Failures: LAM times its time up on line.
    grid = np.linspace(0.0, t, 2_000_001)
    on_line = np.where(inside(grid), 0.0, own(grid))
    up = np.trapezoid(on_line, grid)
    assert events.system_failures == pytest.approx(LAM * up, 1e-6)
    assert rbd.mission_availability(t) == pytest.approx(up / t, 1e-6)
    # The block is off line in each shut, and down in its repairs.
    down = np.trapezoid(1.0 - on_line, grid)
    assert events.node_downtime["a"] == pytest.approx(down, 1e-6)
    cost = rbd.expected_cost(t)
    assert cost.by_category["shutdown"] == pytest.approx(50.0)
    # Its failures (ageing through the shuts too) are its own repairs.
    total_up = np.trapezoid(own(grid), grid)
    assert cost.by_category["repair"] == pytest.approx(
        2.0 * LAM * total_up, 1e-6
    )
    by_window = rbd.expected_events_by_window(width=250.0, count=4)
    assert by_window.shutdowns.tolist() == [3.0, 2.0, 3.0, 2.0]


def test_the_simulation_agrees_with_the_closed_forms():
    rbd = one(shutdown=shut(cost=5.0), spec={"repair_cost": 2.0})
    t, n = 1000.0, 2000
    run = rbd.availability(t, mc_samples=n, seed=3, engine="python", **plain())
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(t)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error
    starts = OFFSET + P * np.arange(10)
    planned = run.system_planned_outages / n
    assert abs(planned - own(starts).sum()) <= Z * math.sqrt(10 * 0.02 / n)
    events = rbd.expected_events(t)
    failures = run.system_failures / n
    assert abs(failures - events.system_failures) <= Z * math.sqrt(
        events.system_failures / n
    )
    assert run.shutdowns == 10
    cost = run.cost
    assert cost.by_category["shutdown"] == pytest.approx(50.0)


def test_a_block_renewed_at_a_shut_in_no_time_is_block_replacement():
    life = W([150.0, 2.0])

    def system(preventive, shutdowns=None):
        return RepairableRBD(
            EDGES,
            {
                "a": {
                    "reliability": life,
                    "repairability": E([0.5]),
                    "preventive": preventive,
                }
            },
            shutdowns=shutdowns,
        )

    at_shut = system(
        {"shutdown": "s0"},
        {"s0": {"period": 100.0, "offset": 30.0, "duration": 0.0}},
    )
    block = system({"interval": 100.0, "policy": "block"})
    x = np.array([10.0, 50.0, 130.0, 777.0])
    # The first renewal at 30, then every 100: block replacement whose
    # last renewal was 70 before 0, of a unit new at 0.
    state = {"a": NodeState(since_scheduled=70.0)}
    np.testing.assert_allclose(
        at_shut.point_availability(x),
        block.point_availability(x, state=state),
        atol=1e-9,
    )
    assert at_shut.mean_availability() == pytest.approx(
        block.mean_availability(), 1e-12
    )
    np.testing.assert_allclose(
        at_shut.expected_events(x).node_preventive["a"],
        block.expected_events(x, state=state).node_preventive["a"],
        atol=1e-9,
    )


def test_a_block_renewed_at_each_shut_fails_only_between_them():
    # Repaired at once, it is up but in the shut; renewed (down, unaged)
    # in each, it fails at LAM while on line.
    rbd = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": E([LAM]),
                "repairability": "instant",
                "preventive": {"shutdown": "line", "cost": 3.0},
            }
        },
        shutdowns={"line": shut(cost=5.0)},
    )
    assert rbd.mean_availability() == pytest.approx(1 - D / P, 1e-7)
    assert rbd.system_failure_frequency() == pytest.approx(
        LAM * (1 - D / P), 1e-7
    )
    events = rbd.expected_events(1000.0)
    assert events.node_preventive["a"] == pytest.approx(10.0)
    assert events.node_failures["a"] == pytest.approx(
        LAM * (1000.0 - 10 * D), 1e-6
    )
    assert rbd.expected_cost_rate() == pytest.approx((3.0 + 5.0) / P, 1e-7)


def test_a_paused_block_is_its_own_process_shifted_by_the_shuts():
    rbd = one(shutdown=shut(idle_ageing=False))
    a = MU / (LAM + MU)
    assert rbd.mean_availability() == pytest.approx((1 - D / P) * a, 1e-12)
    # Its own clock stands still in each shut: at t, it has run t less
    # the shuts' time before t.
    x = np.array([5.0, 25.0, 35.0, 130.0, 555.0])
    elapsed = np.array([0.0, 5.0, 10.0, 20.0, 60.0])
    expected = np.where(inside(x), 0.0, own(x - elapsed))
    np.testing.assert_allclose(rbd.point_availability(x), expected, atol=1e-6)
    # It fails only while its clock runs.
    priced = one(shutdown=shut(idle_ageing=False), spec={"repair_cost": 1.0})
    assert priced.expected_cost_rate() == pytest.approx(
        LAM * a * (1 - D / P), 1e-9
    )
    t, n = 1000.0, 2000
    run = rbd.availability(t, mc_samples=n, seed=4, engine="python", **plain())
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(t)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error


def test_older_than_at_a_shut_lies_between_renewing_and_leaving_it():
    life = W([300.0, 3.0])

    def system(**preventive):
        return RepairableRBD(
            EDGES,
            {
                "a": {
                    "reliability": life,
                    "repairability": E([0.5]),
                    "preventive": {"shutdown": "line", **preventive},
                }
            },
            shutdowns={"line": shut()},
        )

    every = system()
    young = system(older_than=1.0)
    # Older than 1 at every shut it reaches up: replaced at each.
    assert young.mean_availability() == pytest.approx(
        every.mean_availability(), 1e-6
    )
    middle = system(older_than=150.0)
    assert middle.mean_availability() < every.mean_availability()
    t, n = 2000.0, 1000
    run = middle.availability(t, mc_samples=n, seed=5, **plain())
    interval = run.mean_availability_interval()
    exact = middle.mission_availability(t)
    assert abs(interval.estimate - exact) <= Z * interval.standard_error


def test_a_test_at_each_shut_is_an_inspection_with_its_offset():
    def system(inspection, shutdowns=None):
        return RepairableRBD(
            EDGES,
            {
                "a": {
                    "reliability": E([0.002]),
                    "repairability": E([0.5]),
                    "inspection": inspection,
                }
            },
            shutdowns=shutdowns,
        )

    at_shut = system(
        {"shutdown": "s0"},
        {"s0": {"period": 100.0, "offset": 30.0, "duration": 0.0}},
    )
    tested = system({"interval": 100.0, "offset": 30.0})
    x = np.array([10.0, 50.0, 130.0, 777.0])
    np.testing.assert_allclose(
        at_shut.point_availability(x), tested.point_availability(x)
    )
    assert at_shut.mean_availability() == tested.mean_availability()


def test_the_npv_discounts_each_shut_at_its_time():
    rbd = one(shutdown=shut(cost=5.0), spec={"repair_cost": 2.0})
    npv = rbd.expected_npv(
        1000.0, discount_rate=0.001, value_rate=1.0, periods=2
    )
    starts = OFFSET + P * np.arange(10)
    assert npv.cost_by_category["shutdown"] == pytest.approx(
        5.0 * np.exp(-0.001 * starts).sum(), 1e-6
    )
    rows = [r for r in npv.cash_flows if r["category"] == "shutdown"]
    assert [r["undiscounted"] for r in rows] == [-25.0, -25.0]


# -- routes and engines -------------------------------------------------------


def test_shuts_at_given_times_are_simulated():
    rbd = one(shutdown={"times": [20.0, 250.0], "duration": D, "cost": 1.0})
    report = rbd.analysis_routes()
    assert report["mean_availability"].route == routes.REFUSED
    assert "on a period" in report["mission_availability"].reason
    with pytest.raises(NotImplementedError, match="on a period"):
        rbd.mission_availability(100.0)
    run = rbd.availability(300.0, mc_samples=200, seed=1)
    assert run.shutdowns == 2
    assert run.cost.by_category["shutdown"] == pytest.approx(2.0)
    # The block's own process keeps going: down only for the two shuts and
    # its repairs.
    assert run.mean_availability_interval().estimate < 1.0 - 2 * D / 300.0


def test_what_does_not_take_shutdowns_refuses_them():
    rbd = one()
    report = rbd.analysis_routes()
    for name in _shutdowns.NOT_TAKEN:
        assert report[name].route == routes.REFUSED, name
        assert "shutdowns ('line')" in report[name].reason
    with pytest.raises(NotImplementedError, match="birnbaum_importance"):
        rbd.birnbaum_importance()
    with pytest.raises(NotImplementedError, match="event by event"):
        rbd.initialize_event_queue(100.0)


# -- the spec's checks --------------------------------------------------------


@pytest.mark.parametrize(
    "shutdowns, match",
    [
        ({"x": {"duration": 1.0}}, "a period .* or its times"),
        ({"x": {"period": 10.0, "times": [1.0], "duration": 1.0}}, "one or"),
        ({"x": {"period": 10.0}}, "needs a duration"),
        ({"x": {"period": 10.0, "duration": 10.0}}, "its period"),
        ({"x": {"period": 10.0, "offset": 10.0, "duration": 1.0}}, "offset"),
        ({"x": {"times": [5.0, 3.0], "duration": 1.0}}, "increasing"),
        ({"x": {"times": [1.0, 3.0], "duration": 2.0}}, "gap"),
        ({"x": {"period": 10.0, "duration": 1.0, "cost": -1.0}}, "cost"),
        ({"x": {"period": 10.0, "duration": 1.0, "idle_ageing": 1}}, "idle"),
        ({"x": {"period": 10.0, "duration": 1.0, "every": 2}}, "unknown"),
        (
            {
                "x": {"period": 10.0, "duration": 3.0},
                "y": {"period": 20.0, "offset": 2.0, "duration": 1.0},
            },
            "overlap",
        ),
        ({"x": {"period": 10.0, "duration": 1.0, "scope": ["z"]}}, "'z'"),
        ({"x": {"period": "10", "duration": 1.0}}, "period"),
    ],
)
def test_a_shutdown_is_checked_where_it_is_given(shutdowns, match):
    with pytest.raises(ValueError, match=match):
        RepairableRBD(
            EDGES,
            {"a": {"reliability": E([LAM]), "repairability": E([MU])}},
            shutdowns=shutdowns,
        )


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"preventive": {"shutdown": "nope"}}, "do not have"),
        ({"preventive": {"shutdown": "line", "interval": 5.0}}, "interval"),
        ({"preventive": {"shutdown": "line", "duration": FIXED([1])}}, "dur"),
        ({"preventive": {"shutdown": "line", "policy": "age"}}, "policy"),
        ({"preventive": {"older_than": 5.0, "interval": 9.0}}, "shutdown"),
        (
            {
                "preventive": {
                    "shutdown": "line",
                    "older_than": 5.0,
                    "policy": "condition",
                    "threshold": 0.1,
                }
            },
            "one or the other",
        ),
        ({"preventive": {"shutdown": "line", "older_than": -1}}, "older"),
        ({"inspection": {"shutdown": "line", "offset": 1.0}}, "offset"),
    ],
)
def test_a_task_at_a_shut_is_checked_where_it_is_given(spec, match):
    with pytest.raises(ValueError, match=match):
        one(spec=spec)


def test_a_task_at_a_shut_must_be_in_its_scope():
    with pytest.raises(ValueError, match="must be in its scope"):
        RepairableRBD(
            [("s", "a"), ("a", "b"), ("b", "t")],
            {
                "a": {
                    "reliability": E([LAM]),
                    "repairability": E([MU]),
                    "preventive": {"shutdown": "line"},
                },
                "b": {"reliability": E([LAM]), "repairability": E([MU])},
            },
            shutdowns={"line": shut(scope=["b"])},
        )


def test_standing_a_block_with_a_calendar_of_its_own_still_is_refused():
    with pytest.raises(ValueError, match="calendar of its own"):
        one(
            spec={"preventive": {"interval": 50.0, "policy": "block"}},
            shutdown=shut(idle_ageing=False),
        )
    with pytest.raises(ValueError, match="older_than"):
        one(
            spec={"preventive": {"shutdown": "line", "older_than": 5.0}},
            shutdown=shut(idle_ageing=False),
        )


def test_a_state_leaves_the_calendar_of_a_task_at_a_shut_to_the_shut():
    rbd = one(spec={"preventive": {"shutdown": "line"}})
    with pytest.raises(ValueError, match="since_scheduled"):
        rbd.point_availability(
            10.0, state={"a": NodeState(since_scheduled=5.0)}
        )
    # The shut's calendar is from now: the next shut at its offset.
    value = rbd.point_availability(
        [10.0, 25.0], state={"a": NodeState(age=40.0)}
    )
    assert value[1] == 0.0


def test_a_diagram_with_shutdowns_cannot_be_nested_yet():
    inner = one()
    with pytest.raises(ValueError, match="shutdowns of its own"):
        RepairableRBD([("s", "n"), ("n", "t")], {"n": inner})


# -- saving -------------------------------------------------------------------


def test_shutdowns_are_saved_and_loaded():
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {
                "reliability": W([150.0, 2.0]),
                "repairability": E([0.5]),
                "preventive": {
                    "shutdown": "line",
                    "older_than": 90.0,
                    "cost": 2.0,
                },
            },
            "b": {
                "reliability": E([0.004]),
                "repairability": E([0.5]),
                "inspection": {"shutdown": "outage"},
            },
        },
        shutdowns={
            "line": {
                "period": 100.0,
                "offset": 30.0,
                "duration": 4.0,
                "cost": 9.0,
                "scope": ["a"],
            },
            "outage": {"times": [55.0, 260.0], "duration": 3.0},
        },
    )
    back = RepairableRBD.from_json(rbd.to_json())
    assert back._shutdowns == rbd._shutdowns
    for schedules in ("_preventive", "_inspection"):
        # The same schedules (their fixed durations new models of them).
        assert {
            node: schedule._replace(duration=None)
            for node, schedule in getattr(back, schedules).items()
        } == {
            node: schedule._replace(duration=None)
            for node, schedule in getattr(rbd, schedules).items()
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        left = rbd.cost(400.0, mc_samples=20, seed=1)
        right = back.cost(400.0, mc_samples=20, seed=1)
    np.testing.assert_array_equal(left.samples, right.samples)


# -- levers -------------------------------------------------------------------


def test_a_shutdown_has_levers():
    rbd = one(spec={"preventive": {"shutdown": "line", "cost": 1.0}})
    levers = {lever.name: lever for lever in rbd.levers()}
    assert levers["shutdown.line.period"].bounds == (OFFSET, math.inf)
    assert levers["shutdown.line.duration"].bounds == (0.0, P)
    moved = rbd.with_levers({(None, "shutdown.line.period"): 200.0})
    assert moved._shutdowns["line"].period == 200.0
    # The task at the shut takes its new period.
    assert moved._preventive["a"].interval == 200.0
    assert moved.mean_availability() > rbd.mean_availability()
