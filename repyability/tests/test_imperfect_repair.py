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
    result = rbd.availability(200.0, mc_samples=2, seed=1)
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
            1000.0, mc_samples=300, seed=4
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
    result = parent.availability(200.0, mc_samples=2, seed=1)
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
