"""Imperfect repair in a RepairableRBD (#109): a component's repairs take
its virtual age forward by Kijima's models instead of renewing it, its
lives are drawn given that age, and it can be replaced at the N-th failure
since it was renewed."""

import math

import numpy as np
import pytest
import surpyval as surv
from surpyval.recurrent.renewal.renewal_model import conditional_gaps

from repyability import RepairableRBD
from repyability.rbd import routes as r
from repyability.rbd.repairable_rbd import _aged_life

E, W, L = (
    surv.Exponential.from_params,
    surv.Weibull.from_params,
    surv.LogNormal.from_params,
)
X = surv.ExactEventTime.from_params

WORN = {
    "reliability": W([100.0, 2.0]),
    "repairability": E([0.5]),
    "repair_cost": 10.0,
    "replace_cost": 100.0,
}


def single(spec, **options):
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec}, **options)


def kijima(model, q, **more):
    return {"repair": {"model": model, "q": q}, **more}


def states(rbd, until):
    rbd.initialize_event_queue(until)
    changes = [rbd.next_event()]
    while changes[-1][0] < until:
        changes.append(rbd.next_event())
    return changes


def same_runs(one, other, t=1000.0):
    a = one.availability(t, mc_samples=200, seed=1)
    b = other.availability(t, mc_samples=200, seed=1)
    return (
        np.array_equal(a.timeline, b.timeline)
        and np.array_equal(a.availability, b.availability)
        and np.array_equal(a.cost.samples, b.cost.samples)
    )


def test_a_timeline_worked_by_hand():
    # Failing at age 30, repaired in 5, Kijima I with q = 0.5: from new it
    # fails at 30 (virtual age 15 after the repair, so 15 left), at 50
    # (age 22.5, 7.5 left) and at 62.5, its third failure, which replaces
    # it: as new at 67.5, it fails at 97.5, and so on.
    rbd = single(
        {
            "reliability": X(30.0),
            "repairability": X(5.0),
            "repair_cost": 10.0,
            "replace_cost": 100.0,
            **kijima("kijima1", 0.5, replace_after=3),
        }
    )
    assert states(rbd, 200.0) == [
        (30.0, False),
        (35.0, True),
        (50.0, False),
        (55.0, True),
        (62.5, False),
        (67.5, True),
        (97.5, False),
        (102.5, True),
        (117.5, False),
        (122.5, True),
        (130.0, False),
        (135.0, True),
        (165.0, False),
        (170.0, True),
        (185.0, False),
        (190.0, True),
        (197.5, False),
        (200.0, False),
    ]
    result = rbd.availability(
        200.0, mc_samples=2, seed=1, control_variate=False
    )
    assert result.mean_availability_interval().estimate == pytest.approx(
        157.5 / 200.0
    )
    # Six repairs and three replacements: each charged its repair, and a
    # replacement its part too, which it takes from the spares.
    assert result.cost.by_category["repair"] == 9 * 10.0
    assert result.cost.by_category["replace"] == 3 * 100.0
    used = rbd.spares_demand(200.0, method="simulate", mc_samples=2, seed=1)
    assert used["c"].probabilities[3] == 1.0


def test_age_replacement_counts_the_operating_time():
    # Replaced at an operating age of 40: from new it fails at 30 and,
    # repaired by 35, is 30 old (virtual age 15, so 15 left), so it is
    # replaced at 45 before failing; and again after each failure.
    rbd = single(
        {
            "reliability": X(30.0),
            "repairability": X(5.0),
            **kijima("kijima1", 0.5),
            "preventive": {"interval": 40.0, "cost": 1.0},
        }
    )
    assert states(rbd, 200.0) == [
        (30.0, False),
        (35.0, True),
        (75.0, False),
        (80.0, True),
        (120.0, False),
        (125.0, True),
        (165.0, False),
        (170.0, True),
        (200.0, True),
    ]
    # Replaced at 45, 90, 135 and 180.
    assert rbd.cost(200.0, mc_samples=2, seed=1).by_category[
        "preventive"
    ] == pytest.approx(4.0)


def test_no_restoration_is_the_component_renewed_at_every_repair():
    assert single(dict(WORN, **kijima("kijima1", 0.0)))._imperfect == {}
    assert same_runs(
        single(dict(WORN, **kijima("kijima2", 0.0))), single(WORN)
    )


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
@pytest.mark.parametrize(
    "more",
    [
        {},
        {"preventive": {"interval": 80.0, "duration": E([1.0]), "cost": 5}},
        {"inspection": {"interval": 20.0}},
    ],
    ids=["plain", "age replacement", "hidden failures"],
)
def test_replacement_at_every_failure_is_renewal(model, more):
    # Replaced at its first failure, the unit is as new after every one:
    # the same draws, costs and spares as the component renewed by its
    # repairs.
    imperfect = single(
        dict(WORN, **kijima(model, 0.6, replace_after=1), **more)
    )
    assert same_runs(imperfect, single(dict(WORN, **more)))


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
def test_minimal_repair_follows_the_cumulative_hazard(model):
    # As bad as old, repaired at once: failures come at the intensity of
    # the life's hazard, E[N(t)] = H(t) = (t / 100) ** 2 (the power law).
    rbd = single(
        {
            "reliability": W([100.0, 2.0]),
            "repairability": "instant",
            **kijima(model, 1.0),
        }
    )
    for t, expected in ((100.0, 1.0), (300.0, 9.0)):
        result = rbd.availability(t, mc_samples=4000, seed=2)
        count = result.system_failures / result.n_simulations
        assert count == pytest.approx(
            expected, abs=4 * math.sqrt(expected / 4000)
        )


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
def test_a_wearing_unit_does_worse_the_less_a_repair_restores(model):
    availability, failures = [], []
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = single(dict(WORN, **kijima(model, q))).availability(
            1000.0, mc_samples=300, seed=4, control_variate=False
        )
        availability.append(result.mean_availability_interval().estimate)
        failures.append(result.system_failures)
    assert all(np.diff(availability) < 0.0)
    assert all(np.diff(failures) > 0)


def test_a_constant_failure_rate_does_not_age():
    # Exponential lives have no memory: however little a repair restores,
    # the unit fails at the same rate.
    unit = dict(WORN, reliability=E([0.02]))
    renewed = single(unit).availability(5000.0, mc_samples=400, seed=5)
    patched = single(dict(unit, **kijima("kijima1", 1.0))).availability(
        5000.0, mc_samples=400, seed=5
    )
    assert patched.system_failures == pytest.approx(
        renewed.system_failures, rel=0.03
    )


def test_hidden_failures_are_repaired_imperfectly():
    tested = dict(WORN, repairability=E([1.0]), inspection={"interval": 20.0})
    renewed = single(tested).availability(2000.0, mc_samples=200, seed=6)
    patched = single(dict(tested, **kijima("kijima2", 0.8))).availability(
        2000.0, mc_samples=200, seed=6
    )
    assert patched.system_failures > 1.2 * renewed.system_failures


def test_seeds_antithetic_pairs_compare_and_processes():
    rbd = single(dict(WORN, **kijima("kijima1", 0.5, replace_after=4)))
    a = rbd.cost(2000.0, mc_samples=100, seed=8)
    assert np.array_equal(
        a.samples, rbd.cost(2000.0, mc_samples=100, seed=8).samples
    )
    paired = rbd.cost(2000.0, mc_samples=100, seed=8, antithetic=True)
    assert paired.mean == pytest.approx(a.mean, rel=0.05)
    same = rbd.compare(rbd, 2000.0, mc_samples=50, seed=8, quantity="cost")
    assert same.estimate == 0.0 and same.standard_error == 0.0
    split = rbd.cost(2000.0, mc_samples=100, seed=8, n_jobs=2)
    assert np.array_equal(np.sort(split.samples), np.sort(a.samples))
    assert rbd.analysis_routes()["availability"].engine == "python"


def test_a_nested_diagram_repairs_imperfectly_too():
    inner = single(
        {
            "reliability": X(30.0),
            "repairability": X(5.0),
            **kijima("kijima1", 0.5, replace_after=3),
        }
    )
    parent = RepairableRBD([("s", "x"), ("x", "t")], {"x": inner})
    result = parent.availability(
        200.0, mc_samples=2, seed=1, control_variate=False
    )
    assert result.mean_availability_interval().estimate == pytest.approx(
        157.5 / 200.0
    )


@pytest.mark.parametrize(
    "model", [W([100.0, 2.0]), W([3.0, 0.7]), E([0.01]), L([3.0, 0.5])]
)
def test_the_aged_life_is_surpyvals(model):
    rng = np.random.default_rng(1)
    for age in (0.0, 0.5, 10.0, 80.0, 300.0):
        for u in rng.uniform(size=5):
            expected = conditional_gaps(model, np.array([age]), np.array([u]))
            assert _aged_life(model, age, u) == pytest.approx(
                expected[0], rel=1e-9, abs=1e-12
            )


def test_the_exact_methods_refuse_imperfect_repair():
    rbd = single(dict(WORN, **kijima("kijima2", 0.5)), downtime_cost_rate=1.0)
    for method in (
        rbd.mean_availability,
        rbd.node_availability,
        rbd.system_failure_frequency,
        rbd.expected_cost_rate,
        lambda: rbd.point_availability([10.0]),
        rbd.birnbaum_importance,
    ):
        with pytest.raises(NotImplementedError, match="repaired imperfectly"):
            method()
    with pytest.raises(NotImplementedError, match="repaired imperfectly"):
        rbd.spares_demand(100.0)
    report = rbd.analysis_routes()
    assert report["mean_availability"].route == r.REFUSED
    assert "Kijima II, q = 0.5" in report["mean_availability"].reason
    assert report["availability"].route == r.SIMULATED
    crewed = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": dict(WORN, reliability=E([0.01]), **kijima("kijima1", 0.5)),
            "b": dict(WORN, reliability=E([0.01])),
        },
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="imperfect repair"):
        crewed.mean_availability()


@pytest.mark.parametrize(
    "more, message",
    [
        ({"repair": "kijima1"}, "must be a dict"),
        ({"repair": {"model": "arai", "q": 0.5}}, "must be a dict"),
        ({"repair": {"model": "kijima1"}}, "must be a dict"),
        ({"repair": {"model": "kijima1", "q": 1.5}}, r"in \[0, 1\]"),
        ({"repair": {"model": "kijima1", "q": True}}, r"in \[0, 1\]"),
        ({"replace_after": 3}, "needs imperfect repair"),
        (kijima("kijima1", 0.0, replace_after=3), "needs imperfect repair"),
        (kijima("kijima1", 0.5, replace_after=0), "at least 1"),
        (kijima("kijima1", 0.5, replace_after=2.5), "at least 1"),
        (kijima("kijima1", 0.5, standby={"units": 2}), "standby group"),
        (
            kijima(
                "kijima1",
                0.5,
                preventive={
                    "interval": 10.0,
                    "policy": "condition",
                    "threshold": 0.1,
                },
            ),
            "on condition",
        ),
    ],
)
def test_the_spec_is_checked(more, message):
    with pytest.raises(ValueError, match=message):
        single(dict(WORN, **more))


def test_imperfect_repair_is_saved():
    rbd = single(dict(WORN, **kijima("kijima2", 0.7, replace_after=4)))
    loaded = RepairableRBD.from_json(rbd.to_json())
    assert loaded._imperfect == rbd._imperfect
    a = rbd.cost(500.0, mc_samples=50, seed=2)
    b = loaded.cost(500.0, mc_samples=50, seed=2)
    assert np.array_equal(a.samples, b.samples)


# -- minimal repair in no time: exact over a window (#179) -------------------


def minimal(life, model="kijima1", **more):
    """A unit minimally repaired (q = 1) in no time."""
    return {
        "reliability": life,
        "repairability": "instant",
        **kijima(model, 1.0),
        **more,
    }


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
@pytest.mark.parametrize(
    "life",
    [
        W([100.0, 2.0]),
        W([100.0, 0.7]),
        W([100.0, 2.0], lfp_p=0.9),
        W([100.0, 2.0], gamma=10.0),
        E([0.01]),
        L([4.0, 0.5]),
    ],
    ids=[
        "wearing out",
        "wearing in",
        "limited population",
        "offset",
        "exponential",
        "lognormal",
    ],
)
def test_minimal_repair_in_no_time_fails_as_its_cumulative_hazard(life, model):
    # Each repair leaves the unit as old as it was and takes no time, so it
    # is up throughout, and its failures are a Poisson process whose rate is
    # its hazard at its age: H(t) of them by t.
    rbd = single(minimal(life, model, repair_cost=2.0, replace_cost=50.0))
    t = np.array([5.0, 60.0, 250.0])
    hazard = np.asarray(life.Hf(t), dtype=float)
    np.testing.assert_allclose(rbd.expected_failures(t), hazard, rtol=1e-9)
    events = rbd.expected_events(t)
    np.testing.assert_allclose(events.node_failures["c"], hazard, rtol=1e-9)
    np.testing.assert_allclose(events.node_corrective["c"], hazard, rtol=1e-9)
    assert np.all(events.node_downtime["c"] == 0.0)
    assert np.all(events.node_preventive["c"] == 0.0)
    assert np.all(rbd.point_availability(t) == 1.0)
    assert rbd.mission_availability(250.0) == pytest.approx(1.0, abs=1e-12)
    # Never replaced, it is charged only its repairs, as in the simulation.
    cost = rbd.expected_cost(t)
    np.testing.assert_allclose(
        cost.by_category["repair"], 2.0 * hazard, rtol=1e-9
    )
    assert np.all(cost.by_category["replace"] == 0.0)


def pump_line(pump, **options):
    """A pump and its spare in parallel, in series with a valve."""
    return RepairableRBD(
        [
            ("s", "pump"),
            ("pump", "valve"),
            ("s", "spare"),
            ("spare", "valve"),
            ("valve", "t"),
        ],
        {
            "pump": pump,
            "spare": {
                "reliability": W([150.0, 1.5]),
                "repairability": E([0.2]),
            },
            "valve": {
                "reliability": W([400.0, 1.2]),
                "repairability": E([0.5]),
            },
        },
        **options,
    )


def test_with_an_exponential_life_minimal_repair_is_renewal():
    # A memoryless unit is as good as new after any repair, so minimal
    # repair and renewal are the same system, worked out two ways: by the
    # cumulative hazard, and by the renewal equation on a grid.
    def line(**repair):
        return pump_line(
            {
                "reliability": E([0.02]),
                "repairability": "instant",
                "repair_cost": 3.0,
                **repair,
            },
            downtime_cost_rate=5.0,
        )

    minimally, renewed = line(**kijima("kijima2", 1.0)), line()
    t = np.array([10.0, 100.0, 400.0])
    np.testing.assert_allclose(
        minimally.expected_failures(t), renewed.expected_failures(t), rtol=1e-6
    )
    np.testing.assert_allclose(
        minimally.mission_availability(t),
        renewed.mission_availability(t),
        rtol=1e-8,
    )
    one, other = minimally.expected_cost(t), renewed.expected_cost(t)
    np.testing.assert_allclose(one.mean, other.mean, rtol=1e-6)

    # And nested, as one node of a larger system.
    def outer(inner):
        return RepairableRBD([("s", "line"), ("line", "t")], {"line": inner})

    np.testing.assert_allclose(
        outer(minimally).expected_failures(t),
        outer(renewed).expected_failures(t),
        rtol=1e-6,
    )


def test_minimal_repair_in_a_system_agrees_with_the_simulation():
    pump = minimal(W([100.0, 2.5]), "kijima2", repair_cost=1.0)
    rbd = pump_line(pump)
    exact = rbd.expected_events(300.0)
    assert exact.node_failures["pump"] == pytest.approx(3.0**2.5, rel=1e-9)
    sim = rbd.availability(300.0, mc_samples=10_000, seed=3)
    n = sim.n_simulations
    # Only the pump's repairs are priced, at 1 each: a run's cost counts
    # its failures.
    error = float(np.std(sim.cost.samples)) / math.sqrt(n)
    assert abs(sim.cost.mean - exact.node_failures["pump"]) < 4.0 * error
    assert sim.system_failures / n == pytest.approx(
        exact.system_failures, rel=0.03
    )
    assert sim.system_uptime / (n * 300.0) == pytest.approx(
        rbd.mission_availability(300.0), abs=1e-3
    )


@pytest.mark.parametrize(
    "spec, blocker",
    [
        (
            minimal(W([100.0, 2.0]), repair={"model": "kijima1", "q": 0.9}),
            "q below 1",
        ),
        (
            minimal(W([100.0, 2.0]), replace_after=3),
            "replaced at its failure 3",
        ),
        (
            dict(minimal(W([100.0, 2.0])), repairability=E([0.5])),
            "repairs take time",
        ),
        (
            minimal(W([100.0, 2.0]), preventive={"interval": 50.0}),
            "preventive maintenance",
        ),
        (minimal(W([100.0, 2.0], f0=0.1)), "dead on arrival"),
    ],
    ids=[
        "q below 1",
        "replaced",
        "timed repairs",
        "maintained",
        "dead on arrival",
    ],
)
def test_other_imperfect_repair_is_simulated_over_time(spec, blocker):
    rbd = single(spec)
    for call in (
        lambda: rbd.expected_failures(50.0),
        lambda: rbd.point_availability([50.0]),
    ):
        with pytest.raises(NotImplementedError, match=blocker):
            call()
    report = rbd.analysis_routes()
    assert report["expected_failures"].route == r.REFUSED
    assert blocker in report["expected_failures"].reason
    assert report["availability"].route == r.SIMULATED


def test_minimal_repair_has_no_long_run():
    # Its hazard grows without end, and so does its rate of failures.
    rbd = single(minimal(W([100.0, 2.0])))
    with pytest.raises(
        NotImplementedError, match="values over a window from new are exact"
    ):
        rbd.mean_availability()
    report = rbd.analysis_routes()
    assert report["mean_availability"].route == r.REFUSED
    assert report["expected_failures"].route == r.NUMERICAL
