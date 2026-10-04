"""Exact long-run values of components under block replacement.

Under block replacement every ``T`` a unit is replaced at each multiple of
``T`` at which it is up, and a replacement due while it is down is skipped.
The references are closed forms: memoryless lives, whose block replacement
changes nothing; renewal functions known in closed form (instant repair);
components always up but for their replacements, which components replaced
at the same times take together; and, for the rest, the simulation.
"""

import math

import numpy as np
import pytest
import surpyval as surv

import repyability.rbd._block_replacement as block_replacement
from repyability import RepairableRBD
from repyability.rbd._block_replacement import block_cycle

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params
FIXED = surv.ExactEventTime.from_params
INSTANT = FIXED([0.0])


def component(life, repair, interval, duration=None, **costs):
    preventive = {"interval": interval, "policy": "block"}
    if duration is not None:
        preventive["duration"] = duration
    if "cost" in costs:
        preventive["cost"] = costs.pop("cost")
    return {
        "reliability": life,
        "repairability": repair,
        "preventive": preventive,
        **costs,
    }


def alone(spec, **kwargs):
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec}, **kwargs)


def pair(spec_a, spec_b, parallel=True):
    edges = (
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
        if parallel
        else [("s", "a"), ("a", "b"), ("b", "t")]
    )
    return RepairableRBD(edges, {"a": spec_a, "b": spec_b})


def erlang2_renewals(rate, t):
    """The renewal function of Erlang(2, rate) lives."""
    return rate * t / 2.0 - (1.0 - math.exp(-2.0 * rate * t)) / 4.0


# -- One component: closed forms --------------------------------------------


def test_memoryless_lives_are_not_helped():
    # An exponential unit is as good as new at any age, so replacing it
    # changes nothing: MTTF / (MTTF + MTTR), however the repairs run over
    # the block times.
    lam, mu = 0.01, 0.05
    for repair in (E([mu]), FIXED([1 / mu])):
        rbd = alone(component(E([lam]), repair, 150.0))
        assert rbd.mean_availability() == pytest.approx(
            mu / (lam + mu), abs=1e-7
        )
        assert rbd.system_failure_frequency() == pytest.approx(
            lam * mu / (lam + mu), rel=1e-6
        )


def test_instant_repair_counts_the_renewals():
    # Always up; M(T) failures per interval, M the renewal function, and
    # one replacement: the cost rate is (c_p + c_u * M(T)) / T.
    rate, T = 0.02, 150.0
    rbd = alone(
        component(
            surv.Gamma.from_params([2.0, rate]),
            "instant",
            T,
            replace_cost=10.0,
            cost=3.0,
        )
    )
    renewals = erlang2_renewals(rate, T)
    assert rbd.mean_availability() == pytest.approx(1.0, abs=1e-7)
    assert rbd.system_failure_frequency() == pytest.approx(
        renewals / T, rel=1e-6
    )
    assert rbd.expected_cost_rate() == pytest.approx(
        (3.0 + 10.0 * renewals) / T, rel=1e-6
    )


def test_a_replacement_that_takes_time():
    # Instant repair, replacements of a fixed d: up T - d of every T, with
    # failures only while up.
    lam, T, d = 0.01, 100.0, 4.0
    rbd = alone(component(E([lam]), "instant", T, FIXED([d])))
    assert rbd.mean_availability() == pytest.approx((T - d) / T, abs=1e-7)
    assert rbd.system_failure_frequency() == pytest.approx(
        lam * (T - d) / T, rel=1e-6
    )
    failures, planned = rbd._outage_frequencies()
    assert planned == pytest.approx(1.0 / T, rel=1e-9)


def test_the_error_falls_as_the_square_of_the_step(monkeypatch):
    life, repair, duration = W([100, 2.5]), LN([1.5, 0.5]), LN([0.5, 0.3])
    results = []
    for steps in (1000, 2000, 4000):
        monkeypatch.setattr(block_replacement, "_MIN_STEPS", steps)
        monkeypatch.setattr(block_replacement, "_MAX_STEPS", steps)
        cycle = block_cycle(life, repair, duration, 60.0)
        results.append(np.array([cycle.up, cycle.length, cycle.failures]))
    coarse = np.abs(results[0] - results[2])
    fine = np.abs(results[1] - results[2])
    assert np.all(fine < 1e-6 * results[2])
    assert np.all(coarse / fine > 3.0)


def test_the_profiles_average_to_the_cycle():
    cycle = block_cycle(W([100, 2.5]), LN([1.5, 0.5]), LN([0.5, 0.3]), 60.0)
    assert cycle.availability.mean() == pytest.approx(
        cycle.up / cycle.length, abs=1e-7
    )
    assert cycle.failure_rate.mean() == pytest.approx(
        cycle.failures / cycle.length, rel=1e-7
    )
    # Down for its replacement just after a block time, if it was up.
    assert cycle.after == 0.0
    assert 0.9 < cycle.before < 1.0


def test_one_component_matches_the_simulation():
    spec = component(
        W([100, 2.5]),
        LN([1.5, 0.5]),
        60.0,
        LN([0.5, 0.3]),
        repair_cost=10.0,
        downtime_cost=2.0,
        cost=3.0,
    )
    rbd = alone(spec)
    t = 50_000.0
    result = rbd.availability(
        t_simulation=t, mc_samples=20, seed=3, control_variate=False
    )
    window = result.mean_availability_interval()
    assert abs(rbd.mean_availability() - window.estimate) < 4 * (
        window.standard_error
    )
    cost = rbd.cost(
        t_simulation=t, mc_samples=20, seed=4, control_variate=False
    ).mean_interval()
    assert abs(rbd.expected_cost_rate() * t - cost.estimate) < 4 * (
        cost.standard_error
    )


def test_repairs_longer_than_the_interval():
    # The unit is often down at a block time: cycles run over several
    # intervals. Exact values against the simulation's long-run averages.
    spec = component(W([50, 1.5]), E([1 / 40.0]), 20.0)
    rbd = alone(spec)
    result = rbd.availability(
        t_simulation=50_000.0, mc_samples=20, seed=5, control_variate=False
    )
    window = result.mean_availability_interval()
    assert abs(rbd.mean_availability() - window.estimate) < 4 * (
        window.standard_error
    )


# -- Several components: replaced at the same times ---------------------------


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "series"])
def test_replacements_at_the_same_time_take_the_units_down_together(parallel):
    # Instant repair: each unit is down only for its replacements, at the
    # same times as the other's, so the pair is down for them whether in
    # parallel or in series; averaging each unit's availability first would
    # give 1 - (d / T)^2 or ((T - d) / T)^2.
    T, d = 100.0, 4.0
    spec = component(E([0.01]), "instant", T, FIXED([d]))
    rbd = pair(spec, dict(spec), parallel)
    assert rbd.mean_availability() == pytest.approx((T - d) / T, abs=1e-6)
    failures, planned = rbd._outage_frequencies()
    assert planned == pytest.approx(1.0 / T, rel=1e-6)
    if parallel:
        # One unit failing while the other is up takes nothing down.
        assert failures == pytest.approx(0.0, abs=1e-12)


def test_replacements_every_other_interval():
    # In parallel, replaced every T and every 2T: down together only at
    # the even block times.
    T, d = 100.0, 4.0
    rbd = pair(
        component(E([0.01]), "instant", T, FIXED([d])),
        component(E([0.01]), "instant", 2 * T, FIXED([d])),
    )
    assert rbd.mean_availability() == pytest.approx(
        1.0 - d / (2 * T), abs=1e-6
    )
    _, planned = rbd._outage_frequencies()
    assert planned == pytest.approx(1.0 / (2 * T), rel=1e-6)


def test_a_synchronised_pair_matches_the_simulation():
    spec = component(W([1000, 2.5]), LN([3.0, 0.5]), 580.0, W([8, 3]))
    rbd = pair(spec, dict(spec))
    naive = rbd.node_availability()
    exact = rbd.mean_availability()
    # The pair is down whenever both are replaced: far from 1 - q_a q_b.
    assert exact < 1.0 - (1.0 - naive["a"]) * (1.0 - naive["b"]) - 0.005
    result = rbd.availability(
        t_simulation=100_000.0, mc_samples=20, seed=6, control_variate=False
    )
    window = result.mean_availability_interval()
    assert abs(exact - window.estimate) < 4 * window.standard_error
    t = result.n_simulations * result.time_simulated_to
    _, planned = rbd._outage_frequencies()
    assert result.system_planned_outages / t == pytest.approx(
        planned, rel=0.02
    )


def test_importance_and_allocation_run_on_the_calendar():
    spec = component(
        W([1000, 2.5]),
        LN([3.0, 0.5]),
        580.0,
        W([8, 3]),
        acquisition_cost=100.0,
    )
    rbd = pair(spec, dict(spec), parallel=False)
    birnbaum = rbd.birnbaum_importance()
    assert birnbaum["a"] == pytest.approx(birnbaum["b"], rel=1e-9)
    assert 0.0 < birnbaum["a"] < 1.0
    # Copies of a block-replaced unit are replaced together: a second copy
    # of each helps against failures, not against the replacements.
    design = rbd.allocate_redundancy(8760.0, min_availability=0.97)
    assert design.availability >= 0.97


# -- What the exact values do not cover ---------------------------------------


@pytest.mark.parametrize(
    "life, repair, match",
    [
        (W([100, 2], f0=0.1), E([0.1]), "dead on arrival"),
        (W([100, 2]), E([0.1], lfp_p=0.9), "never end"),
    ],
    ids=["dead on arrival", "endless repairs"],
)
def test_models_it_does_not_cover(life, repair, match):
    rbd = alone(component(life, repair, 50.0))
    with pytest.raises(NotImplementedError, match=match):
        rbd.mean_availability()


def test_a_nested_rbd_on_the_same_calendar():
    inner = alone(component(W([100, 2]), E([0.2]), 50.0))
    outer = RepairableRBD(
        [("s", "n"), ("n", "c"), ("c", "t")],
        {"n": inner, "c": component(W([100, 2]), E([0.2]), 50.0)},
    )
    with pytest.raises(NotImplementedError, match="block replacement"):
        outer.mean_availability()
