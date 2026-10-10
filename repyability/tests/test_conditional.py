"""Conditional runs (#189): ``availability(conditional=True)`` and
``cost(conditional=True)`` simulate only the dependent modules and take
every other node exactly given their states.

Each simulation's values are checked against closed forms worked out from
its module's own history (as ``simulate_timelines`` gives it, the same
simulation), the estimates against plain runs and exact values, and the
runs for reproducibility, their engines and what they refuse."""

import math

import numpy as np
import pytest
from surpyval import Exponential, Weibull

from repyability import NodeState, PerfectReliability, RepairableRBD
from repyability.rbd import routes as r
from repyability.rbd._tally import _ModuleRun

E, W = Exponential.from_params, Weibull.from_params
LIFE, REPAIR = 0.004, 0.25  # y's failure and repair rates


def imperfect(**more):
    """A unit repaired imperfectly (Kijima I, q = 0.5): a module."""
    return {
        "reliability": W([300.0, 2.0]),
        "repairability": E([0.05]),
        "repair": {"model": "kijima1", "q": 0.5},
        **more,
    }


def exponential(rate=LIFE, repair=REPAIR, **more):
    return {"reliability": E([rate]), "repairability": E([repair]), **more}


def x_then_y(life=None, **more):
    """The module x (of ``life``, if given) in series with y, a unit taken
    exactly."""
    x = imperfect(repair_cost=7.0)
    if life is not None:
        x["reliability"] = life
    return RepairableRBD(
        [("s", "x"), ("x", "y"), ("y", "t")],
        {"x": x, "y": exponential(repair_cost=3.0)},
        downtime_cost_rate=10.0,
        **more,
    )


def y_up(t):
    """y's point availability from new."""
    total = LIFE + REPAIR
    return REPAIR / total + LIFE / total * np.exp(-total * np.asarray(t))


def y_uptime(a, b):
    """The integral of ``y_up`` over ``[a, b]``."""
    total = LIFE + REPAIR
    return REPAIR / total * (b - a) + LIFE / total**2 * (
        np.exp(-total * a) - np.exp(-total * b)
    )


@pytest.mark.parametrize(
    "life",
    # Dead on arrival, x can fail at 0, and again as it is repaired (from
    # the age 0 it is repaired to): its changes then fall at one instant.
    [None, W([300.0, 2.0], f0=0.3)],
    ids=["new", "dead on arrival"],
)
def test_each_simulation_is_its_expected_values_given_its_module(life):
    rbd = x_then_y(life)
    T, n = 3000.0, 60
    run = rbd.availability(T, mc_samples=n, seed=5, conditional=True)
    assert run.conditional.modules == ("x",)
    # The module's histories, as the plain run with the seed has them.
    x = rbd.simulate_timelines(T, mc_samples=n, seed=5).components["x"]
    failures = restorations = 0.0
    for k in range(n):
        history = x[k]
        up = history.up_intervals
        uptime = math.fsum(y_uptime(a, b) for a, b in up)
        # To the exact integrals' accuracy, about 1e-8 of the window.
        assert run.uptimes[k] == pytest.approx(uptime, abs=1e-8 * T)
        # y's failures while x is up, and x's failures while y is up.
        downs = history.changes[0::2] if history.up else history.changes[1::2]
        lost = math.fsum(LIFE * y_uptime(a, b) for a, b in up)
        caused = math.fsum(y_up(downs))
        failures += lost + caused
        at_end = float(history.state(T)) * float(y_up(T))
        restorations += lost + caused - (1.0 - at_end)
    assert run.system_failures == pytest.approx(failures, rel=1e-7)
    assert run.system_restorations == pytest.approx(restorations, rel=1e-7)
    assert run.system_planned_outages == 0.0
    # The curve: each simulation up with y's chance while x is.
    states = np.array([x.state(t) for t in run.timeline], dtype=float)
    expected = states.mean(axis=1) * y_up(run.timeline)
    # To the point availability's accuracy (near 0, its grid's steps).
    np.testing.assert_allclose(run.availability, expected, atol=1e-5)
    # x's own time down is its simulated one; y's, its exact one.
    assert run.node_downtime["x"] == pytest.approx(float(np.sum(x.downtime)))
    exact = rbd.expected_events(T, working_nodes=["x"]).node_downtime["y"]
    assert run.node_downtime["y"] == pytest.approx(n * exact)
    # The cost: x's own as simulated, y's exact, the system's down time's.
    cost = rbd.cost(T, mc_samples=n, seed=5, conditional=True)
    own = rbd.cost(T, mc_samples=n, seed=5).by_component["x"]
    assert cost.by_component["x"] == pytest.approx(own)
    y_cost = rbd.expected_cost(T, working_nodes=["x"]).by_component["y"]
    assert cost.by_component["y"] == pytest.approx(y_cost)
    assert cost.by_category["system_downtime"] == pytest.approx(
        10.0 * (T - run.uptimes.mean())
    )
    # Each simulation's: x's repairs in it, y's expected repairs and the
    # system's expected time down given x's history.
    repairs = np.array([len(x[k].changes[0::2]) for k in range(n)])
    np.testing.assert_allclose(
        cost.samples,
        7.0 * repairs + y_cost + 10.0 * (T - run.uptimes),
        rtol=1e-9,
    )


def block():
    """Replaced every 250 time units, taking about 5."""
    return {"interval": 250.0, "policy": "block", "duration": E([0.2])}


@pytest.mark.parametrize("parallel", [False, True], ids=["series", "parallel"])
def test_a_module_changing_with_an_exact_node(parallel):
    # x (never failing) and y are replaced at the same instants, the window
    # ending at one. x's change is taken first, and y's from it.
    x = {**imperfect(), "reliability": W([1e9, 2.0]), "preventive": block()}
    y = {**exponential(), "reliability": W([900.0, 2.0])}
    edges = [("s", "x"), ("x", "y"), ("y", "t")]
    if parallel:
        edges = [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")]
    rbd = RepairableRBD(edges, {"x": x, "y": {**y, "preventive": block()}})
    T, n = 3000.0, 200
    run = rbd.availability(T, mc_samples=n, seed=4, conditional=True)
    plain = rbd.availability(T, mc_samples=n, seed=4)
    assert run.conditional.modules == ("x",)
    # Each replacement before the end takes the system down if it is up
    # just before: always in parallel, if y is in series.
    before = np.nextafter(250.0 * np.arange(1, 13), -math.inf)
    y_before = rbd.point_availability(before, working_nodes=["x"])
    if parallel:
        assert run.system_planned_outages / n == pytest.approx(11.0)
        assert plain.system_planned_outages == 11 * n
        down_at_end = 0.0
    else:
        expected = float(np.sum(y_before[:-1]))
        assert run.system_planned_outages / n == pytest.approx(expected)
        mine = plain.system_planned_outages / n
        assert abs(mine - expected) < 4.0 * math.sqrt(11.0 / 4.0 / n)
        down_at_end = 1.0 - float(y_before[-1])
    # The outages not restored are those down at the end, before the
    # replacements at it.
    restored = run.system_failures + run.system_planned_outages
    assert run.system_restorations / n == pytest.approx(
        restored / n - down_at_end
    )


def test_a_failure_comes_before_the_stop_it_opens():
    # a's failure opens a stop at which b is replaced, and b's, a: in the
    # event loop the failure takes the system down, and the replacement
    # planned with it then takes nothing down.
    def member(scale, interval):
        return {
            **exponential(),
            "reliability": W([scale, 2.0]),
            "group": "g",
            "preventive": {
                "interval": interval,
                "policy": "age",
                "duration": E([0.2]),
                "opportunity": 50.0,
            },
        }

    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "y"), ("y", "t")],
        {"a": member(500.0, 400.0), "b": member(700.0, 600.0)}
        | {"y": exponential()},
    )
    T, n = 3000.0, 100
    run = rbd.availability(T, mc_samples=n, seed=9, conditional=True)
    assert run.conditional.modules == ("a", "b")
    runs = rbd.simulate_timelines(T, mc_samples=n, seed=9)
    failures = planned = 0.0
    stops = checked = 0
    for k in range(n):
        changes = []
        for name in ("a", "b"):
            history = runs.components[name][k]
            downs = np.arange(history.changes.size) % 2 == (1 - history.up)
            for t, down, plan in zip(history.changes, downs, history.planned):
                changes.append((t, 2 if not down else int(plan), name, down))
        # At one instant: failures, then planned outages, then repairs.
        changes.sort(key=lambda change: change[:2])
        up = {"a": True, "b": True}
        for t, rank, name, down in changes:
            if down and up["a"] and up["b"]:
                if rank:
                    planned += float(y_up(t))
                else:
                    failures += float(y_up(t))
            up[name] = not down
        # y's own failures while a and b are up.
        for a0, a1 in runs.components["a"][k].up_intervals:
            for b0, b1 in runs.components["b"][k].up_intervals:
                lo, hi = max(a0, b0), min(a1, b1)
                if hi > lo:
                    failures += LIFE * y_uptime(lo, hi)
        # The stops a failure opens, and how the event loop took them.
        instants = {}
        for t, rank, _, down in changes:
            instants.setdefault(t, set()).add(rank if down else 2)
        opened = [t for t, ranks in instants.items() if {0, 1} <= ranks]
        stops += len(opened)
        system = runs.system[k]
        for t in opened:
            at = np.flatnonzero(system.changes == t)
            if at.size:
                assert not system.planned[at[0]]
                checked += 1
    assert stops > 20 and checked > 10
    assert run.system_failures == pytest.approx(failures, rel=1e-7)
    assert run.system_planned_outages == pytest.approx(planned, rel=1e-7)


def test_modules_and_exact_nodes_dead_on_arrival():
    # In a moment from new, the system fails at 0 if x is dead on arrival
    # (whatever y is), or else with y's chance of it: taken from the state
    # just before 0, every node new.
    rbd = RepairableRBD(
        [("s", "x"), ("x", "y"), ("y", "t")],
        {
            "x": {**imperfect(), "reliability": W([300.0, 2.0], f0=0.3)},
            "y": {**exponential(), "reliability": W([900.0, 2.0], f0=0.2)},
        },
    )
    T, n = 1e-3, 400
    run = rbd.availability(T, mc_samples=n, seed=2, conditional=True)
    x = rbd.simulate_timelines(T, mc_samples=n, seed=2).components["x"]
    dead = np.mean([x[k].changes.size > 0 for k in range(n)])
    assert 0.0 < dead < 1.0
    # (Less the failures of y repaired within the moment, dead again.)
    assert run.system_failures / n == pytest.approx(
        dead + (1.0 - dead) * 0.2, abs=1e-4
    )


def plant(**more):
    """A Weibull standby pair and a 2-out-of-3 group with one repair crew
    (both modules), among independent units."""
    group = RepairableRBD(
        [("s", v) for v in "abc"] + [(v, "t") for v in "abc"],
        {
            v: {
                "reliability": W([400.0, 1.4]),
                "repairability": W([25.0, 2.0]),
                "repair_cost": 5.0,
            }
            for v in "abc"
        },
        k={"t": 2},
        repair_crews=1,
    )
    units = {
        f"u{i}": {
            "reliability": W([900.0 + 40 * i, 1.6]),
            "repairability": E([1.0 / 8.0]),
            "repair_cost": 10.0,
        }
        for i in range(6)
    }
    pair = {
        "reliability": W([500.0, 2.0]),
        "repairability": W([30.0, 1.5]),
        "standby": {"units": 2},
        "repair_cost": 50.0,
    }
    edges = [("s", "u0"), ("u0", "u1"), ("u1", "pair"), ("s", "u2")]
    edges += [("u2", "u3"), ("u3", "j"), ("pair", "j"), ("j", "u4")]
    edges += [("u4", "group"), ("group", "u5"), ("u5", "t")]
    return RepairableRBD(
        edges,
        {**units, "pair": pair, "group": group, "j": PerfectReliability},
        downtime_cost_rate=100.0,
        **more,
    )


def z(a, b, se_a, se_b):
    return (a - b) / math.hypot(se_a, se_b)


def test_it_agrees_with_a_plain_run():
    rbd = plant()
    T, n = 2000.0, 3000
    run = rbd.availability(T, mc_samples=n, seed=11, conditional=True)
    plain = rbd.availability(T, mc_samples=n, seed=12, conditional=False)
    assert run.conditional.modules == ("pair", "group")
    a, b = run.mean_availability_interval(), plain.mean_availability_interval()
    assert (
        abs(z(a.estimate, b.estimate, a.standard_error, b.standard_error)) < 4
    )
    assert a.standard_error < b.standard_error
    c, d = run.cost.mean_interval(), plain.cost.mean_interval()
    assert (
        abs(z(c.estimate, d.estimate, c.standard_error, d.standard_error)) < 4
    )
    # Counts: a count's spread is about its square root.
    for name in ("system_failures", "system_restorations"):
        mine, theirs = getattr(run, name) / n, getattr(plain, name) / n
        assert abs(mine - theirs) < 4.0 * math.sqrt(2.0 * theirs / n)
    # The curve, pointwise, within its error.
    at = np.searchsorted(plain.timeline, run.timeline, side="right") - 1
    gap = run.availability - plain.availability[at]
    spread = np.hypot(run.availability_se, plain.availability_se[at])
    assert np.mean(np.abs(gap) < 4.0 * spread + 1e-9) > 0.99


def test_a_module_held_exactly_is_the_exact_values():
    # The conditional machinery with a node the exact methods take, made a
    # module: its estimates must match the exact values.
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": exponential(0.01, 0.1, repair_cost=4.0),
            "b": exponential(0.02, 0.2),
            "c": {
                "reliability": W([800.0, 2.5]),
                "repairability": E([0.5]),
                "preventive": {"interval": 300.0, "policy": "block"},
                "repair_cost": 2.0,
            },
        },
        downtime_cost_rate=3.0,
    )
    rbd._conditional_modules = lambda working, broken: ["a"]
    T, n = 1500.0, 4000
    run = rbd.availability(T, mc_samples=n, seed=2, conditional=True)
    interval = run.mean_availability_interval()
    exact = rbd.mission_availability(T)
    assert abs(interval.estimate - exact) < 4.0 * interval.standard_error
    failures = rbd.expected_failures(T)
    assert run.system_failures / n == pytest.approx(failures, rel=0.05)
    cost = rbd.cost(T, mc_samples=n, seed=2, conditional=True)
    expected = float(rbd.expected_cost(T).mean)
    assert abs(cost.mean - expected) < 4.0 * cost.mean_se
    points = rbd.point_availability(run.timeline)
    assert (
        np.mean(
            np.abs(run.availability - points) < 4 * run.availability_se + 1e-9
        )
        > 0.99
    )


def test_with_no_module_it_is_exact():
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": exponential(repair_cost=1.0), "b": exponential(0.01, 0.5)},
        downtime_cost_rate=2.0,
    )
    run = rbd.availability(800.0, mc_samples=10, seed=1, conditional=True)
    assert run.conditional.modules == ()
    interval = run.mean_availability_interval()
    assert interval.standard_error < 1e-12  # every simulation alike
    assert interval.estimate == pytest.approx(rbd.mission_availability(800.0))
    assert run.system_failures / 10 == pytest.approx(
        rbd.expected_failures(800.0)
    )
    cost = rbd.cost(800.0, mc_samples=10, seed=1, conditional=True)
    assert cost.mean == pytest.approx(float(rbd.expected_cost(800.0).mean))


def test_runs_are_reproduced_whatever_runs_them():
    rbd = x_then_y()
    first = rbd.availability(2000.0, mc_samples=200, seed=3, conditional=True)
    again = rbd.availability(2000.0, mc_samples=200, seed=3, conditional=True)
    assert np.array_equal(first.uptimes, again.uptimes)
    pair = plant()
    base = pair.cost(1000.0, mc_samples=300, seed=4, conditional=True)
    for options in (
        {"engine": "python"},
        {"n_jobs": 2},
        {"engine": "python", "n_jobs": 2},
    ):
        other = pair.cost(
            1000.0, mc_samples=300, seed=4, conditional=True, **options
        )
        assert np.array_equal(other.samples, base.samples)


def test_a_run_to_a_tolerance():
    rbd = x_then_y()
    run = rbd.availability(
        2000.0,
        mc_samples=100,
        seed=8,
        conditional=True,
        tolerance=0.004,
        max_samples=5000,
    )
    assert 100 < run.n_simulations < 5000
    assert run.n_simulations % 100 == 0
    interval = run.mean_availability_interval()
    assert interval.upper - interval.estimate <= 0.004 + 1e-12
    # Its first simulations are those of the first round, alone.
    alone = rbd.availability(2000.0, mc_samples=100, seed=8, conditional=True)
    assert np.array_equal(run.uptimes[:100], alone.uptimes)


def test_antithetic_pairs():
    rbd = plant()
    run = rbd.availability(
        1000.0, mc_samples=200, seed=6, conditional=True, antithetic=True
    )
    assert run.antithetic
    assert run.mean_availability_interval().standard_error > 0.0


def test_a_conditional_run_has_no_spread():
    cost = x_then_y().cost(1000.0, mc_samples=50, seed=1, conditional=True)
    assert cost.mean_interval().standard_error > 0.0
    for spread in (lambda: cost.percentile(90), lambda: cost.std):
        with pytest.raises(ValueError, match="conditional"):
            spread()


def test_a_plain_run_takes_its_means_given_its_modules():
    # By default the whole system is simulated, and the mean intervals take
    # each simulation's expected values given its modules' histories: a
    # conditional run's with the seed, whose modules draw what they draw in
    # the plain run.
    rbd = plant()
    T, n = 1000.0, 400
    run = rbd.availability(T, mc_samples=n, seed=7)
    plain = rbd.availability(T, mc_samples=n, seed=7, conditional=False)
    alone = rbd.availability(T, mc_samples=n, seed=7, conditional=True)
    assert run.conditional.whole
    assert run.conditional.modules == ("pair", "group")
    assert plain.conditional is None
    # Its simulations, and all but its means, are the plain run's.
    for name in ("uptimes", "timeline", "availability", "availability_se"):
        np.testing.assert_array_equal(getattr(run, name), getattr(plain, name))
    assert run.system_failures == plain.system_failures
    assert run.criticalities == plain.criticalities
    np.testing.assert_array_equal(run.cost.samples, plain.cost.samples)
    assert run.cost.percentile(90) == plain.cost.percentile(90)
    assert run.cost.std == plain.cost.std
    # Its means are the conditional run's.
    np.testing.assert_array_equal(run.conditional.uptimes, alone.uptimes)
    np.testing.assert_array_equal(
        run.cost.conditional.costs, alone.cost.samples
    )
    interval = run.mean_availability_interval()
    assert interval.method == "conditional"
    assert interval == alone.mean_availability_interval()
    assert (
        interval.standard_error
        < plain.mean_availability_interval().standard_error
    )
    cost = run.cost.mean_interval()
    assert cost.method == "conditional"
    assert cost == alone.cost.mean_interval()
    assert rbd.cost(T, mc_samples=n, seed=7).mean_interval() == cost


def test_a_run_to_a_tolerance_judges_its_means_given_its_modules():
    rbd = plant()
    options = dict(mc_samples=100, seed=8, tolerance=0.0015)
    run = rbd.availability(1000.0, max_samples=4000, **options)
    plain = rbd.availability(
        1000.0, max_samples=4000, conditional=False, **options
    )
    assert run.n_simulations < plain.n_simulations
    interval = run.mean_availability_interval()
    assert interval.upper - interval.estimate <= 0.0015 + 1e-12
    # Its first simulations, and their means, are the first round's alone.
    first = rbd.availability(1000.0, mc_samples=100, seed=8)
    np.testing.assert_array_equal(run.uptimes[:100], first.uptimes)
    np.testing.assert_array_equal(
        run.conditional.uptimes[:100], first.conditional.uptimes
    )


def test_what_keeps_a_run_plain(monkeypatch):
    rbd = plant()
    plain = rbd.availability(500.0, mc_samples=50, seed=1, conditional=False)
    run = rbd.availability(500.0, mc_samples=50, seed=1, control_variate=False)
    assert run.conditional is None and run.control_variate is None
    np.testing.assert_array_equal(run.uptimes, plain.uptimes)
    controlled = rbd.availability(
        500.0, mc_samples=50, seed=1, control_variate=True
    )
    assert controlled.conditional is None
    assert not controlled.control_variate.itself
    # Where the rest cannot be worked out exactly, the run is plain.

    def refuse(self):
        raise NotImplementedError("out of reach")

    monkeypatch.setattr(_ModuleRun, "prepare", refuse)
    fallen = rbd.availability(500.0, mc_samples=50, seed=1)
    assert fallen.conditional is None
    np.testing.assert_array_equal(fallen.uptimes, plain.uptimes)
    with pytest.raises(NotImplementedError, match="out of reach"):
        rbd.availability(500.0, mc_samples=50, seed=1, conditional=True)


def test_chunks_and_shards_of_a_run_take_its_means():
    rbd = plant()
    run = rbd.availability(500.0, mc_samples=60, seed=2)
    chunks = [
        rbd.simulate_chunk(500.0, a, b, seed=2) for a, b in ((25, 60), (0, 25))
    ]
    merged = rbd.availability_from_chunks(chunks)
    sharded = rbd.availability(
        500.0, mc_samples=60, seed=2, shard_map=map, shard_size=20
    )
    for other in (merged, sharded):
        np.testing.assert_array_equal(other.uptimes, run.uptimes)
        np.testing.assert_array_equal(
            other.conditional.uptimes, run.conditional.uptimes
        )
        assert (
            other.mean_availability_interval()
            == run.mean_availability_interval()
        )
        assert other.cost.mean_interval() == run.cost.mean_interval()


def test_the_modules():
    assert plant()._conditional_modules(set(), set()) == ["pair", "group"]
    # Held, a module is no module; an exponential standby pair is exact.
    assert plant()._conditional_modules({"pair"}, set()) == ["group"]
    exact = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {**exponential(), "standby": {"units": 2}}},
    )
    assert exact._conditional_modules(set(), set()) == []
    # A maintenance group's members are simulated together.
    grouped = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
        {
            "a": {
                **exponential(),
                "reliability": W([300.0, 2.0]),
                "group": "g",
                "preventive": {"interval": 200.0, "opportunity": 100.0},
            },
            "b": {**exponential(), "group": "g"},
            "c": exponential(),
        },
    )
    assert grouped._conditional_modules(set(), set()) == ["a", "b"]


def test_what_a_conditional_run_refuses():
    rbd = plant()
    with pytest.raises(ValueError, match="True, False or None"):
        rbd.availability(10.0, mc_samples=5, conditional="yes")
    with pytest.raises(ValueError, match="record histories"):
        rbd.availability(10.0, mc_samples=5, conditional=True, engine="other")
    # The twin is simulated here, alongside the modules.
    with pytest.raises(ValueError, match="shard_map"):
        rbd.availability(
            10.0,
            mc_samples=5,
            conditional=True,
            control_variate=True,
            shard_map=map,
        )
    with pytest.raises(ValueError, match="no node has one"):
        rbd.availability(10.0, mc_samples=5, conditional=True, demand=2.0)
    # Crews tie every component together, leaving none to take exactly.
    with pytest.raises(NotImplementedError, match="every component"):
        x_then_y(repair_crews=1).availability(
            10.0, mc_samples=5, conditional=True
        )
    # A group stopping at every outage of the system is tied to it all.
    stopping = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {
                **exponential(),
                "reliability": W([300.0, 2.0]),
                "group": "g",
                "preventive": {"interval": 200.0, "opportunity": 100.0},
            },
            "b": exponential(),
        },
        maintenance_groups={"g": {"system_down": True}},
    )
    with pytest.raises(NotImplementedError, match="system_down"):
        stopping.availability(10.0, mc_samples=5, conditional=True)
    assert "conditional=True" not in (
        stopping.analysis_routes()["availability"].reason
    )


def uptimes_given(timelines, inner_integral, n):
    """Each simulation's expected up time: the integral of the exactly
    taken part's availability over the stretches the modules' structure
    (``timelines``, one per simulation) is up."""
    out = []
    for k in range(n):
        up = timelines[k].up_intervals
        out.append(
            math.fsum(inner_integral(b) - inner_integral(a) for a, b in up)
        )
    return np.array(out)


def test_crews_tie_the_components_into_one_module():
    # One crew for the two pumps: every node it serves is a module, and
    # the nested RBD, with crews of its own, is taken exactly.
    inner = RepairableRBD(
        [("s", "p"), ("s", "q"), ("p", "t"), ("q", "t")],
        {"p": exponential(0.01, 0.5), "q": exponential(0.01, 0.5)},
    )

    def pump():
        return {"reliability": W([100.0, 2.0]), "repairability": W([5.0, 1.5])}

    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "n"), ("b", "n"), ("n", "t")],
        {"a": pump(), "b": pump(), "n": inner},
        repair_crews=1,
    )
    T, n = 500.0, 120
    run = rbd.availability(T, mc_samples=n, seed=3, conditional=True)
    assert run.conditional.modules == ("a", "b")
    # The pumps' histories, as the plain run with the seed has them.
    runs = rbd.simulate_timelines(T, mc_samples=n, seed=3).components
    either = runs["a"] | runs["b"]

    def integral(t):
        return 0.0 if t <= 0.0 else t * inner.mission_availability(t)

    np.testing.assert_allclose(
        run.uptimes, uptimes_given(either, integral, n), atol=1e-6 * T
    )
    plain = rbd.availability(T, mc_samples=4000, seed=4)
    a, b = run.mean_availability_interval(), plain.mean_availability_interval()
    assert (
        abs(z(a.estimate, b.estimate, a.standard_error, b.standard_error)) < 4
    )


def crewed_pair_then_y(**more):
    """A nested pair of Weibull units sharing one crew (a module), in
    series with y, taken exactly."""

    def unit():
        return {
            "reliability": W([150.0, 2.0]),
            "repairability": W([8.0, 1.5]),
            "repair_cost": 7.0,
        }

    inner = RepairableRBD(
        [("s", "p"), ("s", "q"), ("p", "t"), ("q", "t")],
        {"p": unit(), "q": unit()},
        repair_crews=1,
    )
    return RepairableRBD(
        [("s", "n"), ("n", "y"), ("y", "t")],
        {"n": inner, "y": exponential(repair_cost=3.0)},
        downtime_cost_rate=10.0,
        **more,
    )


@pytest.mark.parametrize(
    "state",
    [
        {"n": {"p": NodeState(alive=False)}},
        {"y": NodeState(alive=False)},
        {
            "n": {"p": NodeState(age=100.0), "q": NodeState(alive=False)},
            "y": NodeState(alive=False),
        },
    ],
    ids=["module down", "exact node down", "both"],
)
def test_from_the_components_states(state):
    rbd = crewed_pair_then_y()
    T, n = 600.0, 200
    run = rbd.availability(
        T, mc_samples=n, seed=5, conditional=True, state=state
    )
    # The module's histories, as the plain run from the states has them.
    module = rbd.simulate_timelines(
        T, mc_samples=n, seed=5, state=state
    ).components["n"]
    # y from its state: up, or down in a repair.
    total = LIFE + REPAIR
    start = 0.0 if "y" in state else 1.0
    steady = REPAIR / total

    def integral(t):
        return steady * t + (start - steady) / total * (
            1.0 - math.exp(-total * t)
        )

    np.testing.assert_allclose(
        run.uptimes, uptimes_given(module, integral, n), atol=1e-6 * T
    )
    # By default the whole system is simulated, and the means are these.
    whole = rbd.availability(T, mc_samples=n, seed=5, state=state)
    assert whole.conditional.whole
    np.testing.assert_array_equal(whole.conditional.uptimes, run.uptimes)
    plain = rbd.availability(
        T, mc_samples=4000, seed=11, state=state, conditional=False
    )
    a, b = run.mean_availability_interval(), plain.mean_availability_interval()
    assert (
        abs(z(a.estimate, b.estimate, a.standard_error, b.standard_error)) < 4
    )
    c, d = run.cost.mean_interval(), plain.cost.mean_interval()
    assert (
        abs(z(c.estimate, d.estimate, c.standard_error, d.standard_error)) < 4
    )


@pytest.mark.parametrize(
    "state", [None, {"n": {"p": NodeState(alive=False)}}], ids=["new", "state"]
)
def test_shards_are_the_run(state):
    rbd = crewed_pair_then_y()
    whole = rbd.availability(
        600.0, mc_samples=150, seed=5, conditional=True, state=state
    )
    sharded = rbd.availability(
        600.0,
        mc_samples=150,
        seed=5,
        conditional=True,
        state=state,
        shard_map=map,
        shard_size=64,
    )
    assert np.array_equal(whole.uptimes, sharded.uptimes)
    assert whole.system_failures == sharded.system_failures
    assert np.array_equal(whole.cost.samples, sharded.cost.samples)
    assert whole.cost.by_category == sharded.cost.by_category
    assert whole.cost.by_component == sharded.cost.by_component
    assert whole.node_downtime == sharded.node_downtime


def test_capacities_given_the_modules():
    # x (a module) beside y, then z: the capacity has several levels.
    rbd = RepairableRBD(
        [("s", "x"), ("s", "y"), ("x", "z"), ("y", "z"), ("z", "t")],
        {"x": imperfect(), "y": exponential(), "z": exponential(0.003, 0.2)},
        capacity={"x": 60.0, "y": {40.0: 0.5, 30.0: 0.5}, "z": 100.0},
    )
    T = 2000.0
    run = rbd.availability(
        T, mc_samples=600, seed=4, conditional=True, demand=50.0
    )
    plain = rbd.availability(
        T, mc_samples=6000, seed=9, demand=50.0, conditional=False
    )
    assert sorted(run.capacity_time) == sorted(plain.capacity_time)
    for level, spent in plain.capacity_time.items():
        mine = run.capacity_time[level] / (run.n_simulations * T)
        assert mine == pytest.approx(
            spent / (plain.n_simulations * T), abs=0.005
        )
    # The time at a capacity above 0 is the time up.
    working = math.fsum(v for c, v in run.capacity_time.items() if c > 0)
    assert working == pytest.approx(run.system_uptime, rel=1e-6)
    m, s = run.delivered.mean(), run.delivered.std() / math.sqrt(600)
    pm = plain.delivered.mean()
    ps = plain.delivered.std() / math.sqrt(6000)
    assert abs(z(m, pm, s, ps)) < 4
    assert run.demand == 50.0
    assert run.capacity_timeline.size == run.capacity.size


def test_a_control_variate_on_the_modules():
    rbd = x_then_y()
    T = 3000.0
    run = rbd.availability(
        T, mc_samples=400, seed=5, conditional=True, control_variate=True
    )
    assert run.control_variate is not None
    assert run.control_variate.variance_reduction >= 1.0
    assert run.cost.control_variate is not None
    plain = rbd.availability(T, mc_samples=10000, seed=8, conditional=False)
    a, b = run.mean_availability_interval(), plain.mean_availability_interval()
    assert (
        abs(z(a.estimate, b.estimate, a.standard_error, b.standard_error)) < 4
    )
    c, d = run.cost.mean_interval(), plain.cost.mean_interval()
    assert (
        abs(z(c.estimate, d.estimate, c.standard_error, d.standard_error)) < 4
    )
    # The simulations are the same; only the mean is controlled.
    uncontrolled = rbd.availability(
        T, mc_samples=400, seed=5, conditional=True
    )
    assert uncontrolled.control_variate is None


def test_the_routes_say_when_it_applies():
    report = plant().analysis_routes()
    for name in ("availability", "cost"):
        assert report[name].route == r.SIMULATED
        assert "conditional=True" in report[name].reason
        assert "'pair', 'group'" in report[name].reason
    exact = x_then_y(repair_crews=1)
    assert (
        "conditional=True"
        not in exact.analysis_routes()["availability"].reason
    )


# -- modules that never change state in the run (#215) ------------------------


def standby_and_valve(scale: float = 500.0) -> RepairableRBD:
    """The issue's plant: a pair of pumps on standby (the module) before a
    valve; the pair rarely goes down in a window of 500."""
    from surpyval import LogNormal

    def unit(life):
        return {
            "reliability": W([life, 2.0]),
            "repairability": LogNormal.from_params([2.5, 0.5]),
        }

    return RepairableRBD(
        [("s", "pp"), ("pp", "v"), ("v", "t")],
        {"pp": {**unit(scale), "standby": {"units": 2}}, "v": unit(2000.0)},
        downtime_cost_rate=10000.0,
    )


def test_a_module_that_never_changed_leaves_the_error_to_the_simulations():
    rbd = standby_and_valve()
    with pytest.warns(RuntimeWarning, match="never changed state") as caught:
        cost = rbd.cost(500.0, mc_samples=200, seed=0)
    assert cost.conditional.states == 1
    assert "method='simulated'" in str(caught[0].message)
    interval = cost.mean_interval()
    # The simulations' own mean and error, not the exact value of the rest
    # with no error at all.
    assert interval.method == "simulated"
    assert interval.estimate == cost.mean
    assert interval.standard_error == pytest.approx(cost.mean_se)
    assert interval.standard_error > 0.0
    with pytest.warns(RuntimeWarning, match="never changed state"):
        result = rbd.availability(500.0, mc_samples=200, seed=0)
    interval = result.mean_availability_interval()
    assert interval.method == "simulated"
    assert interval.estimate == pytest.approx(
        np.mean(result.uptimes) / 500.0, abs=1e-15
    )
    assert interval.standard_error > 0.0


def test_a_tolerance_is_not_met_by_a_module_that_never_changed():
    rbd = standby_and_valve()
    with pytest.warns(RuntimeWarning):
        cost = rbd.cost(
            500.0, mc_samples=200, seed=0, tolerance=1.0, max_samples=600
        )
    # It ran on, judged by the simulations' own costs, to the limit.
    assert cost.n_simulations == 600
    with pytest.warns(RuntimeWarning) as caught:
        alone = rbd.cost(
            500.0,
            mc_samples=200,
            seed=0,
            tolerance=1.0,
            max_samples=600,
            conditional=True,
        )
    assert alone.n_simulations == 600
    assert any(
        "did not converge" in str(w.message)
        and "never changed state" in str(w.message)
        for w in caught
    )


def test_a_run_of_modules_that_never_changed_gives_no_error():
    rbd = standby_and_valve()
    with pytest.warns(RuntimeWarning, match="no error to give"):
        result = rbd.availability(
            500.0, mc_samples=200, seed=0, conditional=True
        )
    interval = result.mean_availability_interval()
    assert interval.method == "conditional"
    assert math.isnan(interval.standard_error)
    assert math.isnan(interval.lower) and math.isnan(interval.upper)
    assert np.isnan(result.availability_se).all()
    with pytest.warns(RuntimeWarning, match="no error to give"):
        cost = rbd.cost(500.0, mc_samples=200, seed=0, conditional=True)
    assert math.isnan(cost.mean_interval().standard_error)


def test_modules_that_changed_state_give_the_conditional_error():
    rbd = standby_and_valve(scale=60.0)
    cost = rbd.cost(500.0, mc_samples=200, seed=0)
    assert cost.conditional.states > 1
    interval = cost.mean_interval()
    assert interval.method == "conditional"
    assert 0.0 < interval.standard_error < cost.sample_se
    # The result's own mean and error are the interval's (#223); the
    # simulations' own are beside them.
    assert cost.mean == interval.estimate
    assert cost.mean_se == interval.standard_error
    assert cost.sample_mean == pytest.approx(np.mean(cost.samples))
    assert math.fsum(cost.by_category.values()) == pytest.approx(cost.mean)
