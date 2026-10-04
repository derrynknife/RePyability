"""The exact twin as a control variate (#154): ``availability(...,
control_variate=True)`` simulates the system alongside its twin (its
components failing and repaired independently, whose values over the
window are exact) with common random numbers, and controls the mean's
estimate by the twin's error: unbiased, with ``1 - corr**2`` times the
variance."""

import numpy as np
import pytest
import surpyval as surv

from repyability import ControlVariate, NodeState, RepairableRBD
from repyability.rbd import _streams
from repyability.tests.test_performance_equivalence import binomial_first
from repyability.tests.test_simulation_chunks import identical

E, W, L = (
    surv.Exponential.from_params,
    surv.Weibull.from_params,
    surv.LogNormal.from_params,
)
EDGES = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]


def unit(scale=100.0, shape=1.5, **extra):
    return {
        "reliability": W([scale, shape]),
        "repairability": L([1.5, 0.6]),
        "repair_cost": 10.0,
        **extra,
    }


def crewed(scale=100.0, **options):
    return RepairableRBD(
        EDGES,
        {n: unit(scale) for n in "ABC"},
        repair_crews=1,
        downtime_cost_rate=5.0,
        **options,
    )


def within(estimate, reference, sigmas=4.0):
    """Whether two estimates agree to within ``sigmas`` of their joint
    standard error."""
    joint = np.hypot(estimate.standard_error, reference.standard_error)
    return abs(estimate.estimate - reference.estimate) <= sigmas * joint


def test_a_system_that_is_its_own_twin_gets_its_exact_value():
    # Plain components: the twin is the system, simulated with the same
    # draws, so the controlled values are all the exact value.
    rbd = RepairableRBD(EDGES, {n: unit() for n in "ABC"})
    result = rbd.availability(
        300.0, mc_samples=200, seed=3, control_variate=True
    )
    interval = result.mean_availability_interval()
    assert interval.estimate == pytest.approx(
        rbd.mission_availability(300.0), abs=1e-15
    )
    assert interval.standard_error < 1e-15
    control = result.control_variate
    assert control.coefficient == pytest.approx(1.0)
    assert control.correlation == pytest.approx(1.0)
    assert control.variance_reduction > 1e12
    assert rbd.analysis_routes()["availability"].twin.startswith(
        "the system itself"
    )
    # A run to a tolerance then stops at once.
    run = rbd.availability(
        300.0, mc_samples=50, seed=3, tolerance=1e-6, control_variate=True
    )
    assert run.n_simulations == 50


def test_the_controlled_estimate_is_unbiased_and_more_precise():
    rbd = crewed()
    plain = rbd.availability(1000.0, mc_samples=2000, seed=1)
    controlled = rbd.availability(
        1000.0, mc_samples=2000, seed=1, control_variate=True
    )
    reference = rbd.availability(1000.0, mc_samples=30_000, seed=11)
    p, c, r = (
        result.mean_availability_interval()
        for result in (plain, controlled, reference)
    )
    assert within(c, r)
    assert (p.standard_error / c.standard_error) ** 2 > 4.0
    control = controlled.control_variate
    assert control.variance_reduction == pytest.approx(
        (p.standard_error / c.standard_error) ** 2, rel=0.25
    )
    assert control.twin.shape == (2000,)
    assert control.exact == pytest.approx(
        rbd._twin()[0].mission_availability(1000.0)
    )
    fractions = controlled.uptimes / 1000.0
    assert c.estimate == pytest.approx(control.controlled(fractions).mean())
    # Everything else is the simulations' own.
    assert controlled.system_uptime / (2000 * 1000.0) == pytest.approx(
        fractions.mean()
    )
    # The costs are controlled too, by the twin's exact expected cost.
    cost = controlled.cost
    assert cost.control_variate is not None
    assert within(cost.mean_interval(), reference.cost.mean_interval())
    assert (
        plain.cost.mean_interval().standard_error
        > 2 * cost.mean_interval().standard_error
    )
    # Its mean is the controlled estimate (#223); the simulations' own is
    # beside it, and so is their breakdown, a twin's being only the total's.
    assert cost.mean == cost.mean_interval().estimate
    assert cost.sample_mean == pytest.approx(np.mean(cost.samples))
    assert sum(cost.by_category.values()) == pytest.approx(cost.sample_mean)


def test_cost_takes_the_control_too():
    rbd = crewed()
    controlled = rbd.cost(
        1000.0, mc_samples=1000, seed=2, control_variate=True
    )
    through = rbd.availability(
        1000.0, mc_samples=1000, seed=2, control_variate=True
    ).cost
    identical(controlled, through)
    interval = controlled.mean_interval()
    expected = controlled.control_variate.controlled(controlled.samples)
    assert interval.estimate == pytest.approx(expected.mean())


def test_a_run_to_a_tolerance_stops_sooner():
    rbd = crewed()
    run = dict(mc_samples=500, seed=2, tolerance=0.0006)
    plain = rbd.availability(1000.0, **run)
    controlled = rbd.availability(1000.0, control_variate=True, **run)
    assert controlled.n_simulations < plain.n_simulations
    interval = controlled.mean_availability_interval()
    assert interval.upper - interval.estimate <= 0.0006
    assert len(controlled.control_variate.twin) == controlled.n_simulations
    cost = rbd.cost(
        1000.0, mc_samples=300, seed=4, tolerance=15.0, control_variate=True
    )
    assert cost.mean_interval().upper - cost.mean_interval().estimate <= 15.0


def test_antithetic_pairs_and_parallel_runs():
    rbd = crewed()
    run = dict(mc_samples=600, seed=5, control_variate=True)
    paired = rbd.availability(1000.0, antithetic=True, **run)
    assert paired.antithetic and paired.control_variate is not None
    x = paired.uptimes / 1000.0
    pairs = x.reshape(-1, 2).mean(axis=1)
    twin = paired.control_variate.twin.reshape(-1, 2).mean(axis=1)
    covariance = np.cov(pairs, twin)
    assert paired.control_variate.coefficient == pytest.approx(
        covariance[0, 1] / covariance[1, 1]
    )
    identical(
        rbd.availability(1000.0, engine="python", **run),
        rbd.availability(1000.0, engine="python", n_jobs=2, **run),
    )


def test_from_a_state_and_with_held_nodes():
    rbd = crewed()
    state = {
        "A": NodeState(alive=False, down_for=0.5),
        "C": NodeState(alive=True, age=50.0),
    }
    controlled = rbd.availability(
        500.0, mc_samples=1500, seed=6, state=state, control_variate=True
    )
    reference = rbd.availability(
        500.0, mc_samples=20_000, seed=12, state=state
    )
    assert within(
        controlled.mean_availability_interval(),
        reference.mean_availability_interval(),
    )
    held = rbd.availability(
        500.0,
        mc_samples=1500,
        seed=6,
        broken_nodes=["B"],
        control_variate=True,
    )
    exact = rbd._twin()[0].mission_availability(500.0, broken_nodes=["B"])
    assert held.control_variate.exact == pytest.approx(exact)


def test_the_twin_leaves_out_what_ties_components_together():
    standby = {"units": 3, "k": 2}
    rbd = RepairableRBD(
        [
            ("s", "G"),
            ("G", "A"),
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "t"),
        ],
        {
            "G": unit(80.0, 1.8, standby=standby),
            "A": unit(
                group="g", preventive={"interval": 80.0, "opportunity": 40.0}
            ),
            "B": unit(group="g", preventive={"interval": 80.0}),
            "C": unit(repair={"model": "kijima1", "q": 0.5}),
            "D": unit(
                inspection={"interval": 20.0, "duration": E([1.0])},
                priority=2.0,
            ),
        },
        repair_crews=2,
        maintenance_groups={"g": {}},
    )
    twin, changes = rbd._twin()
    assert changes == [
        "the limit on repair crews",
        "the maintenance groups",
        "the switching of standby group 'G' (its units operate together)",
        "the imperfect repair of 'C'",
        "the inspections of 'D' (its failures are revealed)",
    ]
    assert twin.repair_crews is None and not twin._maintenance
    # A and B keep their age replacement, which the exact methods take.
    assert set(twin._preventive) == {"A", "B"}
    # The standby group's twin is its units, as a nested RBD whose streams
    # are the units'.
    assert isinstance(twin.components["G"], RepairableRBD)
    mine, _ = rbd._stream_specs(500.0)
    theirs, _ = twin._stream_specs(500.0)
    for unit_index in range(3):
        for kind in (_streams.FAILURE, _streams.REPAIR):
            assert (("G", unit_index), kind) in mine
            assert (("G", unit_index), kind) in theirs
    assert twin._over_time().route == "numerical"
    route = rbd.analysis_routes()["availability"]
    assert route.twin.startswith("the system without the limit on repair")
    assert "Exact twin: the system without" in str(route)
    controlled = rbd.availability(
        300.0, mc_samples=300, seed=8, control_variate=True
    )
    assert 0.0 < controlled.control_variate.correlation <= 1.0


def test_a_nested_rbd_s_twin():
    pump = unit()
    inner = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": pump, "b": pump},
        repair_crews=1,
    )
    outer = RepairableRBD(
        [("s", "N"), ("N", "c"), ("c", "t")], {"N": inner, "c": pump}
    )
    assert outer.analysis_routes()["availability"].twin == (
        "the system without the limit on repair crews in 'N'."
    )
    plain = outer.availability(
        1000.0, mc_samples=1000, seed=1, conditional=False
    )
    controlled = outer.availability(
        1000.0, mc_samples=1000, seed=1, control_variate=True
    )
    assert (
        plain.mean_availability_interval().standard_error
        > 2 * controlled.mean_availability_interval().standard_error
    )


def test_what_has_no_exact_twin_is_refused():
    fixed = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": surv.FixedEventProbability.from_params(0.1),
                "repairability": E([1.0]),
            }
        },
    )
    with pytest.raises(NotImplementedError, match="no exact twin"):
        fixed.availability(10.0, mc_samples=10, seed=1, control_variate=True)
    assert fixed.analysis_routes()["availability"].twin.startswith("none.")
    unstreamable = RepairableRBD(
        EDGES,
        {
            "A": unit(),
            "B": {
                "reliability": binomial_first([40, 2]),
                "repairability": E([0.5]),
            },
            "C": unit(),
        },
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="replayable"):
        unstreamable.availability(
            10.0, mc_samples=10, seed=1, control_variate=True
        )
    assert "replayable" in unstreamable.analysis_routes()["availability"].twin
    rbd = crewed()
    with pytest.raises(ValueError, match="leave out shard_map"):
        rbd.availability(
            10.0, mc_samples=10, seed=1, control_variate=True, shard_map=map
        )
    with pytest.raises(ValueError, match="True or False"):
        rbd.availability(10.0, mc_samples=10, seed=1, control_variate="yes")


def test_a_control_of_values_that_do_not_vary():
    control = ControlVariate.of([0.9, 0.9, 0.9], [1.0, 1.0, 1.0], exact=0.95)
    assert control.coefficient == 0.0 and control.correlation == 0.0
    np.testing.assert_array_equal(control.controlled([0.9, 0.9, 0.9]), 0.9)
    assert control.variance_reduction == 1.0
    assert set(control) == {
        "twin",
        "exact",
        "coefficient",
        "correlation",
        "itself",
    }


def test_a_system_that_is_its_own_twin_is_simulated_once(monkeypatch):
    # Its twin's run, drawn from the same streams, is its own to the last
    # bit (#186): one run a round, where a twin that differs runs beside it.
    runs = []
    plain = RepairableRBD._run

    def counted(self, *args, **kwargs):
        runs.append(self)
        return plain(self, *args, **kwargs)

    monkeypatch.setattr(RepairableRBD, "_run", counted)
    own = RepairableRBD(EDGES, {n: unit() for n in "ABC"})
    result = own.availability(
        300.0, mc_samples=100, seed=3, control_variate=True
    )
    assert runs == [own]
    np.testing.assert_array_equal(
        result.control_variate.twin, result.uptimes / 300.0
    )
    runs.clear()
    crewed().availability(300.0, mc_samples=100, seed=3, control_variate=True)
    assert len(runs) == 2


def test_minimal_repair_in_no_time_is_kept_by_the_twin():
    # Its values over a window are exact (#179), so the twin keeps it, and
    # a system of it and plain components is its own twin.
    pump = {
        "reliability": surv.Weibull.from_params([100.0, 2.5]),
        "repairability": "instant",
        "repair": {"model": "kijima2", "q": 1.0},
        "repair_cost": 1.0,
    }
    rbd = RepairableRBD(EDGES, {"A": pump, "B": unit(), "C": unit()})
    assert rbd.analysis_routes()["availability"].twin.startswith(
        "the system itself"
    )
    result = rbd.availability(
        300.0, mc_samples=100, seed=1, control_variate=True
    )
    assert result.mean_availability_interval().method == "exact"
    assert result.cost.mean_interval().estimate == pytest.approx(
        rbd.expected_cost(300.0).mean, rel=1e-12
    )
    # Repaired otherwise, it is left out.
    partly = RepairableRBD(
        EDGES,
        {"A": dict(pump, repair={"model": "kijima2", "q": 0.5}), "B": unit()}
        | {"C": unit()},
    )
    assert "imperfect repair" in partly.analysis_routes()["availability"].twin


def test_the_twin_builds_each_curve_once(monkeypatch):
    # Its exact cost and availability share the components' curves (#185),
    # and give what they give apart.
    rbd = RepairableRBD(EDGES, {n: unit() for n in "ABC"})
    twin, _ = rbd._twin()
    apart = (twin.expected_cost(300.0).mean, twin.mission_availability(300.0))
    built = []
    plain = RepairableRBD._unit_curve

    def counted(self, node, *args, **kwargs):
        built.append(node)
        return plain(self, node, *args, **kwargs)

    monkeypatch.setattr(RepairableRBD, "_unit_curve", counted)
    with twin._sharing_curves():
        together = (
            twin.expected_cost(300.0).mean,
            twin.mission_availability(300.0),
        )
    assert sorted(built) == ["A", "B", "C"]
    assert together == apart
    assert twin._curve_memo is None
    built.clear()
    rbd.availability(300.0, mc_samples=20, seed=1, control_variate=True)
    assert sorted(built) == ["A", "B", "C"]


def test_a_run_takes_exact_values_by_default(monkeypatch):
    # #187: a system that is its own twin has exact expected values over the
    # window, which every run takes; one to a tolerance stops at once.
    rbd = RepairableRBD(EDGES, {n: unit() for n in "ABC"})
    run = rbd.availability(300.0, mc_samples=50, seed=3, tolerance=1e-6)
    assert run.n_simulations == 50
    interval = run.mean_availability_interval()
    assert interval.method == "exact"
    assert interval.estimate == rbd.mission_availability(300.0)
    cost = rbd.cost(300.0, mc_samples=50, seed=3, tolerance=1e-6)
    assert cost.mean_interval().method == "exact"
    assert cost.mean_interval().estimate == pytest.approx(
        rbd.expected_cost(300.0).mean, rel=1e-12
    )
    # Without a tolerance too; its simulations are a plain run's, and False
    # simulates to the end.
    fixed = rbd.availability(300.0, mc_samples=50, seed=3)
    assert fixed.control_variate.itself
    assert fixed.mean_availability_interval() == interval
    plain = rbd.availability(
        300.0, mc_samples=50, seed=3, control_variate=False
    )
    assert plain.control_variate is None
    assert plain.mean_availability_interval().method == "simulated"
    for other in (run, fixed):
        np.testing.assert_array_equal(other.uptimes, plain.uptimes)
        np.testing.assert_array_equal(other.availability, plain.availability)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        forced = rbd.availability(
            300.0,
            mc_samples=50,
            seed=3,
            tolerance=1e-6,
            max_samples=100,
            control_variate=False,
        )
    assert forced.n_simulations == 100
    assert forced.control_variate is None
    # As shards, and from chunks of the run.
    sharded = rbd.availability(300.0, mc_samples=50, seed=3, shard_map=map)
    assert sharded.mean_availability_interval() == interval
    chunks = [
        rbd.simulate_chunk(300.0, a, b, seed=3) for a, b in ((0, 20), (20, 50))
    ]
    merged = rbd.availability_from_chunks(chunks)
    assert merged.mean_availability_interval() == interval
    # With a twin that differs, it is as before.
    with pytest.warns(RuntimeWarning, match="did not converge"):
        crewed_run = crewed().availability(
            300.0, mc_samples=50, seed=3, tolerance=1e-6, max_samples=100
        )
    assert crewed_run.control_variate is None

    # Should its exact values be out of reach, it simulates.
    def refuse(*args, **kwargs):
        raise NotImplementedError("too many grid points")

    monkeypatch.setattr(RepairableRBD, "_twin_exact", refuse)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fallen = rbd.availability(
            300.0, mc_samples=50, seed=3, tolerance=1e-6, max_samples=100
        )
    assert fallen.control_variate is None
    assert fallen.n_simulations == 100
    with pytest.raises(NotImplementedError, match="too many grid points"):
        rbd.availability(300.0, mc_samples=50, seed=3, control_variate=True)


def test_a_run_takes_the_exact_values_of_a_system_tied_together():
    # Exponential units sharing a crew are not their own twin, but the
    # crews' chain works out their expected values, which a run takes.
    exponential = {
        "reliability": E([0.01]),
        "repairability": E([0.2]),
        "repair_cost": 10.0,
    }
    rbd = RepairableRBD(
        EDGES,
        {n: exponential for n in "ABC"},
        repair_crews=1,
        downtime_cost_rate=5.0,
    )
    assert rbd.analysis_routes()["availability"].twin.startswith(
        "the system without the limit on repair crews"
    )
    run = rbd.availability(300.0, mc_samples=50, seed=3)
    assert run.control_variate.itself
    interval = run.mean_availability_interval()
    assert interval.method == "exact"
    assert interval.estimate == rbd.mission_availability(300.0)
    assert run.cost.mean_interval().estimate == pytest.approx(
        rbd.expected_cost(300.0).mean, rel=1e-12
    )
    # control_variate=True takes the twin, without the crews' limit.
    twin = rbd.availability(300.0, mc_samples=50, seed=3, control_variate=True)
    assert not twin.control_variate.itself
    np.testing.assert_array_equal(twin.uptimes, run.uptimes)


def test_the_route_says_the_expected_values_need_no_simulation():
    own = RepairableRBD(EDGES, {n: unit() for n in "ABC"}).analysis_routes()
    for name in ("availability", "cost"):
        assert "mission_availability, expected_events" in own[name].reason
        assert "by default a run's means are theirs" in (own[name].reason)
    # Weibull lives sharing a crew have no exact values over a window.
    assert (
        "mission_availability"
        not in crewed().analysis_routes()["availability"].reason
    )
