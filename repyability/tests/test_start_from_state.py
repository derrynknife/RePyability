"""Repairable analyses from the components' current states, not only from
new (#125): the ``state`` argument of the exact methods
(``point_availability``, ``mission_availability``, ``expected_failures``,
``expected_events``, ``expected_cost``, ``point_capacity`` and
``mission_capacity``) and of the simulation (``availability``, ``cost``,
``compare``, ``simulate_chunk`` and ``initialize_event_queue``).

Checked against closed forms (an exponential unit is memoryless, and one
down comes back as its two-state Markov chain does; a unit that is never
repaired in the window is up with what is left of its life, ``R(a + t) /
R(a)``), against timelines worked out by hand (fixed lives, repairs and
maintenance), against the long run (a stationary start stays in it),
against the simulation from the same states, and the all-new state against
the results from new, to the last bit.
"""

import json

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import binom

from repyability import (
    NodeState,
    NonRepairable,
    RepairableRBD,
    SimulationChunk,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params
L = surv.LogNormal.from_params
X = surv.ExactEventTime.from_params

ONE = [("s", "a"), ("a", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def single(**spec) -> RepairableRBD:
    return RepairableRBD(ONE, {"a": spec})


def maintained_pair() -> RepairableRBD:
    """Two Weibull pumps in parallel, replaced at age 500 (maintenance
    taking time), with costs."""
    pump = {
        "reliability": W([1000.0, 2.5]),
        "repairability": L([3.0, 0.5]),
        "preventive": {
            "interval": 500.0,
            "duration": W([8.0, 3.0]),
            "cost": 100.0,
        },
        "replace_cost": 1000.0,
    }
    return RepairableRBD(
        PARALLEL, {"a": pump, "b": pump}, downtime_cost_rate=500.0
    )


def available_from_down(failure_rate, repair_rate, t):
    """An exponential unit down at 0: its two-state Markov chain from
    down."""
    total = failure_rate + repair_rate
    return repair_rate / total * -np.expm1(-total * t)


def test_new_states_change_nothing():
    rbd = maintained_pair()
    x = np.array([0.0, 10.0, 500.0, 512.0, 1500.0])
    base = rbd.point_availability(x)
    for state in ({}, {"a": NodeState()}, {"a": NodeState(), "b": None}):
        if None in state.values():
            continue
        assert np.array_equal(rbd.point_availability(x, state=state), base)
    events = rbd.expected_events(1000.0)
    same = rbd.expected_events(1000.0, state={"b": NodeState()})
    assert same.system_failures == events.system_failures
    # The simulation draws the same numbers: the same results to the bit.
    run = rbd.availability(800.0, mc_samples=300, seed=4)
    for state in ({}, {"a": NodeState(), "b": NodeState()}):
        again = rbd.availability(800.0, mc_samples=300, seed=4, state=state)
        assert np.array_equal(again.uptimes, run.uptimes)
        assert np.array_equal(again.availability, run.availability)
        assert np.array_equal(again.cost.samples, run.cost.samples)


def test_an_exponential_unit_is_memoryless():
    lam, mu = 0.1, 1.0
    rbd = single(reliability=E([lam]), repairability=E([mu]))
    x = np.array([0.0, 0.5, 2.0, 10.0, 50.0])
    new = rbd.point_availability(x)
    aged = rbd.point_availability(x, state={"a": NodeState(age=250.0)})
    np.testing.assert_allclose(aged, new, atol=1e-9)
    down = {"a": NodeState(alive=False, down_for=3.0)}
    np.testing.assert_allclose(
        rbd.point_availability(x, state=down),
        available_from_down(lam, mu, x),
        atol=3e-7,
    )
    # Its failures from down: lam times its expected up time.
    t = np.array([1.0, 10.0, 100.0])
    total = lam + mu
    uptime = mu / total * (t + np.expm1(-total * t) / total)
    np.testing.assert_allclose(
        rbd.expected_failures(t, state=down), lam * uptime, atol=2e-7
    )
    np.testing.assert_allclose(
        rbd.mission_availability(t, state=down), uptime / t, atol=2e-7
    )


def test_a_unit_up_at_an_age_lives_what_is_left_of_its_life():
    life = W([100.0, 2.5])
    # Not repaired within the window: up while its first life lasts.
    rbd = single(reliability=life, repairability=X([1e6]))
    a = 80.0
    x = np.array([0.0, 5.0, 20.0, 60.0, 150.0])
    left = life.sf(a + x) / life.sf(a)
    state = {"a": NodeState(age=a)}
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state), left, atol=1e-7
    )
    np.testing.assert_allclose(
        rbd.expected_failures(x, state=state), 1.0 - left, atol=1e-7
    )


def test_a_unit_down_comes_back_with_what_is_left_of_its_repair():
    repair = L([1.5, 0.6])
    # Never failing within the window: up once its repair is done.
    rbd = single(reliability=X([1e6]), repairability=repair)
    r = 2.0
    x = np.array([0.0, 0.5, 2.0, 5.0, 20.0])
    done = 1.0 - repair.sf(r + x) / repair.sf(r)
    state = {"a": NodeState(alive=False, down_for=r)}
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state), done, atol=1e-7
    )
    # The simulation: one draw, by the inverse of the remaining time.
    run = rbd.availability(30.0, mc_samples=4000, seed=2, state=state)
    assert run.availability[0] == 0.0
    back = np.sort(30.0 - run.uptimes)
    np.testing.assert_allclose(
        np.searchsorted(back, x[1:], side="right") / 4000,
        done[1:],
        atol=0.03,
    )


def fixed_maintained() -> RepairableRBD:
    """A unit that never fails in the window, replaced at age 100 in 5."""
    return single(
        reliability=X([1e6]),
        repairability=X([1.0]),
        preventive={"interval": 100.0, "duration": X([5.0])},
    )


@pytest.mark.parametrize(
    "age, downs",
    [
        (60.0, [(40.0, 45.0), (145.0, 150.0), (250.0, 255.0)]),
        # Past its age: replaced at once.
        (130.0, [(0.0, 5.0), (105.0, 110.0), (210.0, 215.0)]),
    ],
)
def test_age_replacement_falls_when_the_unit_reaches_its_age(age, downs):
    rbd = fixed_maintained()
    state = {"a": NodeState(age=age)}
    T = 300.0
    down = sum(end - start for start, end in downs)
    assert rbd.mission_availability(T, state=state) == pytest.approx(
        (T - down) / T, abs=1e-9
    )
    middles = np.array([(start + end) / 2.0 for start, end in downs])
    after = np.array([end + 1.0 for _, end in downs])
    np.testing.assert_allclose(
        rbd.point_availability(middles, state=state), 0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        rbd.point_availability(after, state=state), 1.0, atol=1e-9
    )
    events = rbd.expected_events(T, state=state)
    assert float(events.node_preventive["a"]) == pytest.approx(3.0)
    assert float(events.system_planned_outages) == pytest.approx(3.0)
    # The simulation follows the same timeline.
    run = rbd.availability(T, mc_samples=3, seed=0, state=state)
    np.testing.assert_allclose(run.uptimes, T - down)
    assert run.system_planned_outages == 9


def test_a_unit_down_for_maintenance_finishes_it():
    rbd = fixed_maintained()
    state = {"a": NodeState(alive=False, down_for=2.0, maintenance=True)}
    # Back at 3, as new: maintained again at 103 to 108, and 208 to 213.
    assert rbd.mission_availability(300.0, state=state) == pytest.approx(
        (300.0 - 13.0) / 300.0, abs=1e-9
    )
    run = rbd.availability(300.0, mc_samples=2, seed=0, state=state)
    np.testing.assert_allclose(run.uptimes, 287.0)
    # Not a planned outage of the window: it started before it.
    assert run.system_planned_outages == 4


def stationary_plant() -> RepairableRBD:
    pump = {"reliability": E([0.01]), "repairability": L([2.0, 0.5])}
    aged = {
        "reliability": W([800.0, 3.0]),
        "repairability": L([2.5, 0.4]),
        "preventive": {
            "interval": 400.0,
            "duration": W([6.0, 2.0]),
            "cost": 50.0,
        },
        "replace_cost": 400.0,
    }
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {"a": pump, "b": pump, "c": aged},
        downtime_cost_rate=100.0,
    )


def test_a_stationary_start_stays_in_the_long_run():
    rbd = stationary_plant()
    long_run = rbd.mean_availability()
    x = np.array([0.0, 1.0, 37.0, 400.0, 5000.0])
    np.testing.assert_allclose(
        rbd.point_availability(x, state="stationary"), long_run, atol=1e-12
    )
    t = np.array([10.0, 1000.0])
    np.testing.assert_allclose(
        rbd.mission_availability(t, state="stationary"), long_run, atol=1e-12
    )
    np.testing.assert_allclose(
        rbd.expected_failures(t, state="stationary"),
        rbd.system_failure_frequency() * t,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        rbd.expected_cost(t, state="stationary").mean,
        rbd.expected_cost_rate() * t,
        rtol=1e-9,
    )


def test_a_stationary_calendar_starts_at_its_phase():
    unit = {
        "reliability": W([300.0, 2.0]),
        "repairability": L([2.5, 0.7]),
        "preventive": {
            "policy": "block",
            "interval": 100.0,
            "duration": W([5.0, 3.0]),
        },
    }
    rbd = RepairableRBD(ONE, {"a": unit})
    phase = 30.0
    state = {"a": NodeState(stationary=True, phase=phase)}
    x = np.array([0.0, 10.0, 69.0, 71.0, 150.0, 290.0])
    # Long in service: as far from new as it has settled, at the phase.
    far = rbd.point_availability(5000.0 + phase + x)
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state), far, atol=1e-9
    )
    # Over whole intervals: the long-run availability.
    assert rbd.mission_availability(300.0, state=state) == pytest.approx(
        rbd.mean_availability(), abs=1e-9
    )


def test_a_tested_unit_was_last_known_up_at_its_last_test():
    lam = 0.002
    tested = {
        "reliability": E([lam]),
        "repairability": "instant",
        "inspection": {"interval": 100.0},
    }
    rbd = RepairableRBD(ONE, {"a": tested})
    x = np.array([0.0, 30.0, 69.0, 71.0, 150.0])
    since_test = np.where(x < 70.0, 30.0 + x, x - 70.0)
    state = {"a": NodeState(age=1000.0, phase=30.0)}
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state),
        np.exp(-lam * since_test),
        atol=1e-12,
    )
    # Put into service 10 ago, after its last test, 30 ago.
    renewed = {"a": NodeState(age=10.0, phase=30.0)}
    since = np.where(x < 70.0, 10.0 + x, x - 70.0)
    np.testing.assert_allclose(
        rbd.point_availability(x, state=renewed),
        np.exp(-lam * since),
        atol=1e-12,
    )
    events = rbd.expected_events(np.array([50.0, 250.0]), state=renewed)
    found = -np.expm1(-lam * 80.0)
    per_test = -np.expm1(-lam * 100.0)
    np.testing.assert_allclose(
        events.node_corrective["a"], [0.0, found + per_test], rtol=1e-12
    )
    np.testing.assert_allclose(
        events.node_failures["a"],
        [
            np.exp(-lam * 10.0) - np.exp(-lam * 60.0),
            np.exp(-lam * 10.0)
            - np.exp(-lam * 80.0)
            + per_test
            - np.expm1(-lam * 80.0),
        ],
        rtol=1e-12,
    )
    # Over a long mission: the first, shorter interval and then whole ones.
    T = 10_070.0
    whole = -np.expm1(-lam * 100.0) / (lam * 100.0)
    head = np.exp(-lam * 10.0) * -np.expm1(-lam * 70.0) / lam
    assert rbd.mission_availability(T, state=renewed) == pytest.approx(
        (head + (T - 70.0) * whole) / T, rel=1e-12
    )
    # The simulation draws whether it has failed since, unseen.
    run = rbd.availability(
        250.0, mc_samples=4000, seed=5, state=state, control_variate=False
    )
    up = np.exp(-lam * 30.0)
    assert abs(run.availability[0] - up) < 4.0 * np.sqrt(up * (1 - up) / 4000)
    mission = rbd.mission_availability(250.0, state=state)
    interval = run.mean_availability_interval(0.999)
    assert interval.lower <= mission <= interval.upper


def check_against_simulation(rbd, state, T, mc_samples=4000, seed=7):
    """The exact methods from ``state`` within the simulation's
    intervals (and its point availabilities within five standard
    errors)."""
    run = rbd.availability(
        T, mc_samples=mc_samples, seed=seed, state=state, control_variate=False
    )
    mission = rbd.mission_availability(T, state=state)
    interval = run.mean_availability_interval(0.999)
    assert interval.lower <= mission <= interval.upper
    x = np.linspace(0.0, T, 13)[1:-1]
    exact = rbd.point_availability(x, state=state)
    simulated = run.availability[
        np.searchsorted(run.timeline, x, side="right") - 1
    ]
    spread = np.sqrt(np.maximum(exact * (1.0 - exact), 1e-4) / mc_samples)
    assert np.all(np.abs(simulated - exact) <= 5.0 * spread)
    failures = rbd.expected_failures(T, state=state)
    simulated = run.system_failures / mc_samples
    assert abs(simulated - failures) <= 5.0 * np.sqrt(
        max(failures, 1e-3) / mc_samples
    )
    return run


@pytest.mark.parametrize(
    "state",
    [
        {"a": NodeState(age=400.0), "b": NodeState(alive=False, down_for=5)},
        {
            "a": NodeState(age=650.0),
            "b": NodeState(alive=False, down_for=2.0, maintenance=True),
        },
    ],
)
def test_maintained_pumps_are_what_the_simulation_estimates(state):
    rbd = maintained_pair()
    check_against_simulation(rbd, state, 2000.0)
    cost = rbd.cost(
        2000.0, mc_samples=4000, seed=7, state=state, control_variate=False
    )
    interval = cost.mean_interval(0.999)
    exact = rbd.expected_cost(2000.0, state=state).mean
    assert interval.lower <= exact <= interval.upper


@pytest.mark.parametrize(
    "start",
    [
        NodeState(age=40.0, phase=40.0),
        NodeState(age=0.0, phase=30.0),
        NodeState(alive=False, down_for=4.0, phase=95.0),
        NodeState(alive=False, down_for=2.0, maintenance=True, phase=2.0),
    ],
)
def test_block_replacement_from_a_state_is_what_the_simulation_estimates(
    start,
):
    unit = {
        "reliability": W([300.0, 2.0]),
        "repairability": L([2.5, 0.7]),
        "preventive": {
            "policy": "block",
            "interval": 100.0,
            "duration": W([5.0, 3.0]),
        },
    }
    rbd = RepairableRBD(ONE, {"a": unit})
    run = check_against_simulation(rbd, {"a": start}, 450.0)
    planned = rbd.expected_events(450.0, state={"a": start})
    simulated = run.system_planned_outages / run.n_simulations
    assert simulated == pytest.approx(
        float(planned.system_planned_outages), abs=0.03
    )


def test_a_nested_rbd_takes_its_components_states():
    unit = {"reliability": W([500.0, 1.8]), "repairability": L([2.5, 0.6])}
    skid = RepairableRBD(PARALLEL, {"a": unit, "b": unit})
    nested = RepairableRBD(
        [("in", "skid"), ("skid", "c"), ("c", "out")],
        {"skid": skid, "c": unit},
    )
    flat = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {"a": unit, "b": unit, "c": unit},
    )
    inner = {"a": NodeState(alive=False, down_for=1.0), "b": NodeState(300.0)}
    state = {"skid": inner, "c": NodeState(age=50.0)}
    x = np.array([1.0, 10.0, 200.0, 2000.0])
    np.testing.assert_allclose(
        nested.point_availability(x, state=state),
        flat.point_availability(x, state={**inner, "c": state["c"]}),
        atol=1e-12,
    )
    check_against_simulation(nested, state, 600.0)


def test_capacity_from_a_state():
    lam, mu = 0.1, 1.0
    pump = {"reliability": E([lam]), "repairability": E([mu])}
    rbd = RepairableRBD(
        [("in", n) for n in "abc"] + [(n, "out") for n in "abc"],
        {n: pump for n in "abc"},
        capacity={n: 50.0 for n in "abc"},
    )
    state = {"a": NodeState(alive=False, down_for=1.0)}
    x = np.array([0.0, 0.5, 3.0])
    capacity = rbd.point_capacity(x, state=state)
    total = lam + mu
    up = mu / total + lam / total * np.exp(-total * x)
    back = available_from_down(lam, mu, x)
    # The two others binomial, the one down its own way.
    others = np.array([binom.pmf(k, 2, up) for k in range(3)])
    exact = np.zeros((4, len(x)))
    exact[:3] += others * (1.0 - back)
    exact[1:] += others * back
    np.testing.assert_allclose(capacity.probabilities, exact, atol=3e-7)


def test_a_component_down_holds_a_repair_crew():
    # Fixed lives and repairs: a fails at 3; b, down 1 into a repair of 5,
    # is back at 4, when the one crew can start on a, back at 9 (at 8, had
    # b's repair not held the crew).
    rbd = RepairableRBD(
        SERIES,
        {
            "a": {"reliability": X([3.0]), "repairability": X([5.0])},
            "b": {"reliability": X([10.0]), "repairability": X([5.0])},
        },
        repair_crews=1,
    )
    rbd.initialize_event_queue(
        20.0, state={"b": NodeState(alive=False, down_for=1.0)}
    )
    assert rbd.component_status == {"a": True, "b": False}
    assert rbd.system_state is False
    assert rbd.next_event() == (9.0, True)
    with pytest.raises(ValueError, match="more than the 1 repair crew"):
        rbd.availability(
            20.0,
            mc_samples=10,
            state={
                "a": NodeState(alive=False, down_for=1.0),
                "b": NodeState(alive=False, down_for=1.0),
            },
        )


def test_a_system_down_at_the_start_shows_in_the_availability():
    unit = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(SERIES, {"a": unit, "b": unit})
    state = {"a": NodeState(alive=False, down_for=0.5)}
    run = rbd.availability(50.0, mc_samples=500, seed=1, state=state)
    assert run.timeline[0] == 0.0 and run.availability[0] == 0.0
    assert run.system_failures < run.system_restorations


def test_chunks_from_a_state_merge_into_the_run():
    rbd = maintained_pair()
    state = {"a": NodeState(age=300.0)}
    first = rbd.simulate_chunk(900.0, 0, 120, seed=3, state=state)
    rest = rbd.simulate_chunk(900.0, 120, 200, seed=3, state=state)
    merged = rbd.availability_from_chunks(
        [first, SimulationChunk.from_json(rest.to_json())]
    )
    whole = rbd.availability(900.0, mc_samples=200, seed=3, state=state)
    assert np.array_equal(merged.uptimes, whole.uptimes)
    assert json.loads(first.settings["state"])[0][0] == "'a'"
    other = rbd.simulate_chunk(
        900.0, 120, 200, seed=3, state={"a": NodeState(age=301.0)}
    )
    with pytest.raises(ValueError, match="cannot be merged"):
        SimulationChunk.merge([first, other])
    assert rbd.simulate_chunk(900.0, 0, 10, seed=3).settings["state"] is None


def test_compare_from_a_state_draws_the_same_numbers():
    state = {
        "a": NodeState(age=300.0),
        "b": NodeState(alive=False, down_for=1.0),
    }
    difference = maintained_pair().compare(
        maintained_pair(), 500.0, mc_samples=200, seed=2, state=state
    )
    assert difference.estimate == 0.0
    assert difference.standard_error == 0.0


def test_a_run_from_a_state_is_simulated_in_python():
    unit = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(PARALLEL, {"a": unit, "b": unit})
    state = {"a": NodeState(alive=False, down_for=0.5)}
    auto = rbd.availability(50.0, mc_samples=300, seed=1, state=state)
    python = rbd.availability(
        50.0, mc_samples=300, seed=1, state=state, engine="python"
    )
    assert np.array_equal(auto.uptimes, python.uptimes)
    pytest.importorskip("numba")
    with pytest.raises(NotImplementedError, match="started from a state"):
        rbd.availability(
            50.0, mc_samples=300, seed=1, state=state, engine="numba"
        )


def test_what_a_state_cannot_be():
    rbd = maintained_pair()
    with pytest.raises(ValueError, match="not on a calendar"):
        rbd.point_availability(1.0, state={"a": NodeState(phase=1.0)})
    with pytest.raises(ValueError, match="not a component"):
        rbd.point_availability(1.0, state={"z": NodeState()})
    with pytest.raises(ValueError, match="held working or broken"):
        rbd.point_availability(
            1.0, working_nodes=["a"], state={"a": NodeState(age=1.0)}
        )
    with pytest.raises(TypeError, match="must be a NodeState"):
        rbd.point_availability(1.0, state={"a": 3.0})
    with pytest.raises(ValueError, match="dict of NodeStates"):
        rbd.availability(1.0, state="steady")
    block = single(
        reliability=W([300.0, 2.0]),
        repairability=L([2.5, 0.7]),
        preventive={"policy": "block", "interval": 100.0},
    )
    with pytest.raises(ValueError, match="less than its interval"):
        block.availability(1.0, state={"a": NodeState(phase=100.0)})
    with pytest.raises(ValueError, match="no preventive maintenance"):
        block.availability(
            1.0, state={"a": NodeState(alive=False, maintenance=True)}
        )
    fixed = single(reliability=X([5.0]), repairability=X([2.0]))
    with pytest.raises(ValueError, match="cannot be up at age 6"):
        fixed.point_availability(1.0, state={"a": NodeState(age=6.0)})
    with pytest.raises(ValueError, match="cannot have been down for 2"):
        fixed.availability(
            1.0, state={"a": NodeState(alive=False, down_for=2.0)}
        )
    nested = RepairableRBD([("s", "n"), ("n", "t")], {"n": fixed})
    with pytest.raises(ValueError, match="components' states"):
        nested.point_availability(1.0, state={"n": NodeState(age=1.0)})


def test_what_a_state_is_not_taken_for():
    imperfect = single(
        reliability=W([100.0, 2.0]),
        repairability=E([1.0]),
        repair={"model": "kijima1", "q": 0.5},
    )
    with pytest.raises(NotImplementedError, match="virtual age"):
        imperfect.availability(10.0, state={"a": NodeState(age=5.0)})
    standby = single(
        reliability=E([0.01]),
        repairability=E([0.5]),
        standby={"units": 2},
    )
    with pytest.raises(NotImplementedError, match="standby group"):
        standby.availability(10.0, state={"a": NodeState(age=5.0)})
    # A simulation starts from given states, not the long-run one.
    plant = stationary_plant()
    for state in ("stationary", {"a": NodeState(stationary=True)}):
        with pytest.raises(NotImplementedError, match="given states"):
            plant.availability(10.0, mc_samples=10, state=state)
    # A component that draws its events its own way.

    class Own(NonRepairable):
        pass

    own = RepairableRBD(ONE, {"a": Own(E([0.01]), E([0.5]))})
    with pytest.raises(NotImplementedError, match="its own way"):
        own.availability(10.0, mc_samples=10, state={"a": NodeState(3.0)})
    # Hidden failures repaired at once cannot be down in the exact values,
    # but can be simulated.
    tested = single(
        reliability=E([0.002]),
        repairability=E([0.5]),
        inspection={"interval": 100.0},
    )
    down = {"a": NodeState(alive=False, down_for=1.0, phase=10.0)}
    run = tested.availability(50.0, mc_samples=50, seed=0, state=down)
    assert run.availability[0] == 0.0
