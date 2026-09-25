"""Tests for the cost model on :class:`RepairableRBD`: the closed-form cost
rate and the simulated cost distribution.

Every expectation here is a hand-computed reference value rather than a
recorded output. With Exponential reliability and repairability a component
has ``MTTF = 1/lambda``, ``MTTR = 1/mu``, availability ``A = MTTF/(MTTF+MTTR)``
and long-run failure frequency ``omega = 1/(MTTF+MTTR)``, so every cost term
is exact arithmetic; the simulation is checked against the same arithmetic.
"""

import json

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD

# MTTF = 10, MTTR = 1  ->  A = 10/11, omega = 1/11
RELIABILITY = surv.Exponential.from_params([0.1])
REPAIRABILITY = surv.Exponential.from_params([1.0])
A = 10.0 / 11.0
OMEGA = 1.0 / 11.0

SERIES = [("s", "c"), ("c", "t")]


def one_component(spec_extra=None, **kwargs):
    spec = {"reliability": RELIABILITY, "repairability": REPAIRABILITY}
    spec.update(spec_extra or {})
    return RepairableRBD(SERIES, {"c": spec}, **kwargs)


# -- the cost terms, each against exact arithmetic -------------------------


def test_system_downtime_cost():
    rbd = one_component(downtime_cost_rate=50.0)
    assert rbd.expected_cost_rate() == pytest.approx(50.0 * (1.0 - A))


def test_repair_cost_is_charged_per_failure():
    rbd = one_component({"repair_cost": 100.0})
    assert rbd.expected_cost_rate() == pytest.approx(100.0 * OMEGA)


def test_repair_and_replace_costs_add():
    rbd = one_component({"repair_cost": 100.0, "replace_cost": 400.0})
    assert rbd.expected_cost_rate() == pytest.approx(500.0 * OMEGA)


def test_component_downtime_cost_is_charged_per_unit_time():
    rbd = one_component({"downtime_cost": 22.0})
    assert rbd.expected_cost_rate() == pytest.approx(22.0 * (1.0 - A))


def test_terms_sum():
    rbd = one_component(
        {"repair_cost": 100.0, "replace_cost": 400.0, "downtime_cost": 22.0},
        downtime_cost_rate=50.0,
    )
    expected = 500.0 * OMEGA + 22.0 * (1.0 - A) + 50.0 * (1.0 - A)
    assert rbd.expected_cost_rate() == pytest.approx(expected)


# -- everything is optional ------------------------------------------------


def test_no_costs_is_zero_and_reports_unpriced():
    rbd = one_component()
    assert rbd.has_costs is False
    assert rbd.expected_cost_rate() == 0.0


def test_zero_costs_are_not_a_cost_model():
    # Explicit zeros price nothing, so there is still no cost model to run.
    rbd = one_component({"repair_cost": 0.0}, downtime_cost_rate=0.0)
    assert rbd.has_costs is False
    assert rbd.expected_cost_rate() == 0.0


def test_any_single_cost_is_enough():
    for spec, kwargs in [
        ({"repair_cost": 1.0}, {}),
        ({"replace_cost": 1.0}, {}),
        ({"downtime_cost": 1.0}, {}),
        ({}, {"downtime_cost_rate": 1.0}),
    ]:
        rbd = one_component(spec, **kwargs)
        assert rbd.has_costs is True
        assert rbd.expected_cost_rate() > 0.0


# -- conditioning ----------------------------------------------------------


def test_forced_working_node_costs_nothing():
    # A node that never fails incurs no corrective action and no downtime.
    rbd = one_component({"repair_cost": 100.0}, downtime_cost_rate=50.0)
    assert rbd.expected_cost_rate(working_nodes=["c"]) == pytest.approx(0.0)


def test_forced_broken_node_pays_downtime_but_not_repairs():
    # A node held failed never changes state, so no corrective cost; the
    # system is down for all time, so the full downtime rate applies.
    rbd = one_component({"repair_cost": 100.0}, downtime_cost_rate=50.0)
    assert rbd.expected_cost_rate(broken_nodes=["c"]) == pytest.approx(50.0)


# -- structure enters only through availability ----------------------------


def test_corrective_cost_is_topology_independent():
    # Two identical components fail at the same rate whether they are in
    # series or parallel, so the corrective spend is identical; only the
    # system-downtime term sees the redundancy.
    spec = {
        "reliability": RELIABILITY,
        "repairability": REPAIRABILITY,
        "repair_cost": 100.0,
    }
    series = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": spec, "b": spec}
    )
    parallel = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": spec, "b": spec},
    )
    both = pytest.approx(2.0 * 100.0 * OMEGA)
    assert series.expected_cost_rate() == both
    assert parallel.expected_cost_rate() == both


def test_redundancy_lowers_the_downtime_cost():
    spec = {"reliability": RELIABILITY, "repairability": REPAIRABILITY}
    series = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": spec, "b": spec},
        downtime_cost_rate=1000.0,
    )
    parallel = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": spec, "b": spec},
        downtime_cost_rate=1000.0,
    )
    assert parallel.expected_cost_rate() < series.expected_cost_rate()
    # Exact: series is up only if both are, parallel unless both are down.
    assert series.expected_cost_rate() == pytest.approx(1000.0 * (1.0 - A**2))
    assert parallel.expected_cost_rate() == pytest.approx(
        1000.0 * (1.0 - A) ** 2
    )


# -- validation ------------------------------------------------------------


def test_mistyped_cost_key_is_rejected():
    # A silently-ignored cost key would price the component at zero.
    with pytest.raises(ValueError, match="unknown key"):
        one_component({"repair_costs": 10.0})


def test_negative_and_non_numeric_costs_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        one_component({"repair_cost": -5.0})
    with pytest.raises(ValueError, match="must be a number"):
        one_component({"repair_cost": "free"})
    with pytest.raises(ValueError, match="non-negative"):
        one_component(downtime_cost_rate=-1.0)


# -- serialisation ---------------------------------------------------------


def test_costs_survive_a_json_round_trip():
    rbd = one_component(
        {"repair_cost": 100.0, "replace_cost": 400.0, "downtime_cost": 22.0},
        downtime_cost_rate=50.0,
    )
    restored = RepairableRBD.from_json(json.dumps(json.loads(rbd.to_json())))
    assert restored.downtime_cost_rate == 50.0
    assert restored.costs == rbd.costs
    assert restored.expected_cost_rate() == pytest.approx(
        rbd.expected_cost_rate()
    )


def test_uncosted_rbd_still_round_trips():
    rbd = one_component()
    restored = RepairableRBD.from_json(rbd.to_json())
    assert restored.has_costs is False
    assert restored.expected_cost_rate() == 0.0


# -- instant repair (zero repair time) -------------------------------------


def test_instant_repair_availability_is_one():
    rbd = one_component({"repairability": "instant"})
    assert rbd.mean_availability() == pytest.approx(1.0)
    result = rbd.availability(t_simulation=50.0, N=100, seed=1)
    # Every outage has zero length, so availability never dips.
    assert np.all(result.availability == 1.0)
    assert result.system_downtime == pytest.approx(0.0)


def test_instant_repair_still_fails_and_costs():
    # Invisible to availability, but not to money: failures still happen at
    # frequency 1/MTTF and each one is charged.
    rbd = one_component({"repairability": "instant", "repair_cost": 100.0})
    assert rbd.expected_cost_rate() == pytest.approx(100.0 / 10.0)
    result = rbd.cost(t_simulation=500.0, N=400, seed=5)
    assert result.cost_rate == pytest.approx(10.0, rel=0.05)


def test_instant_repair_round_trips():
    rbd = one_component({"repairability": "instant", "repair_cost": 100.0})
    restored = RepairableRBD.from_json(rbd.to_json())
    assert restored.mean_availability() == pytest.approx(1.0)
    assert restored.expected_cost_rate() == pytest.approx(10.0)


def test_unknown_repairability_string_rejected():
    with pytest.raises(ValueError, match="instant"):
        one_component({"repairability": "immediate"})


# -- the simulated cost distribution ---------------------------------------


def full_cost_rbd():
    return one_component(
        {"repair_cost": 100.0, "replace_cost": 400.0, "downtime_cost": 22.0},
        downtime_cost_rate=50.0,
    )


def test_cost_simulation_converges_to_the_closed_form():
    # The whole point of building the exact rate first: the Monte-Carlo mean
    # must converge to it. (A long window drowns the start-up transient.)
    rbd = full_cost_rbd()
    result = rbd.cost(t_simulation=1000.0, N=600, seed=3)
    assert result.cost_rate == pytest.approx(
        rbd.expected_cost_rate(), rel=0.05
    )


def test_cost_breakdowns_are_internally_consistent():
    result = full_cost_rbd().cost(t_simulation=200.0, N=300, seed=4)
    assert set(result.by_category) == {
        "repair",
        "replace",
        "component_downtime",
        "system_downtime",
    }
    # The category means partition the overall mean...
    assert sum(result.by_category.values()) == pytest.approx(result.mean)
    # ...and with one costed component, its attributable share is everything
    # except the (system-level) downtime cost.
    assert result.by_component["c"] == pytest.approx(
        result.mean - result.by_category["system_downtime"]
    )
    assert result.n_simulations == len(result.samples) == 300
    assert result.percentile(100) == pytest.approx(result.samples.max())
    assert result.std > 0.0


def test_repair_and_replace_are_charged_separately_at_each_failure():
    result = full_cost_rbd().availability(t_simulation=200.0, N=100, seed=4)
    # One component in series: every component failure is a system failure.
    failures_per_replication = result.system_failures / 100
    assert result.cost.by_category["repair"] == pytest.approx(
        100.0 * failures_per_replication
    )
    assert result.cost.by_category["replace"] == pytest.approx(
        400.0 * failures_per_replication
    )


# -- the confidence interval on the mean -----------------------------------


def test_mean_interval_covers_the_exact_expected_cost():
    # One Exponential component that starts up: over [0, T] its expected
    # uptime is mu T/(lam+mu) + lam/(lam+mu)^2 (1 - exp(-(lam+mu) T)), it
    # fails at rate lam while up, and it is down the rest of the time. Each
    # failure costs 100 + 400; each unit of downtime costs 22 + 50.
    lam, mu, T = 0.1, 1.0, 100.0
    uptime = mu * T / (lam + mu) + lam / (lam + mu) ** 2 * (
        1.0 - np.exp(-(lam + mu) * T)
    )
    exact = 500.0 * lam * uptime + (22.0 + 50.0) * (T - uptime)

    result = full_cost_rbd().cost(t_simulation=T, N=400, seed=2)
    interval = result.mean_interval(0.99)
    assert interval.lower < exact < interval.upper
    assert interval.estimate == result.mean
    assert interval.standard_error == pytest.approx(result.std / np.sqrt(400))
    assert interval.confidence == 0.99
    assert interval.n_samples == 400


def test_more_replications_sharpen_the_mean_but_not_the_spread():
    few = full_cost_rbd().cost(t_simulation=100.0, N=100, seed=1)
    many = full_cost_rbd().cost(t_simulation=100.0, N=400, seed=1)
    # The mean's standard error shrinks like 1/sqrt(N)...
    assert few.mean_se / many.mean_se == pytest.approx(2.0, rel=0.3)
    # ...while how much a window's cost varies is a property of the system.
    assert many.std == pytest.approx(few.std, rel=0.3)


def test_mean_interval_confidence():
    result = full_cost_rbd().cost(t_simulation=50.0, N=20, seed=0)

    def width(interval):
        return interval.upper - interval.lower

    assert width(result.mean_interval(0.99)) > width(result.mean_interval(0.9))
    for bad in (0.0, 1.0, 95):
        with pytest.raises(ValueError, match="between 0 and 1"):
            result.mean_interval(bad)


def test_cost_is_reproducible_with_a_seed():
    a = full_cost_rbd().cost(t_simulation=100.0, N=50, seed=9)
    b = full_cost_rbd().cost(t_simulation=100.0, N=50, seed=9)
    assert np.allclose(a.samples, b.samples)


def test_cost_rides_along_on_availability():
    result = full_cost_rbd().availability(t_simulation=100.0, N=50, seed=9)
    assert result.cost is not None
    assert result.cost.mean > 0.0


def test_cost_is_none_when_nothing_is_priced():
    rbd = one_component()
    assert rbd.cost(t_simulation=100.0, N=10, seed=0) is None
    assert rbd.availability(t_simulation=100.0, N=10, seed=0).cost is None


def test_forced_working_node_simulates_to_zero_cost():
    rbd = one_component({"repair_cost": 100.0}, downtime_cost_rate=50.0)
    result = rbd.cost(t_simulation=100.0, N=30, seed=2, working_nodes=["c"])
    assert np.allclose(result.samples, 0.0)


# -- per-failure costs drawn from a distribution ---------------------------

# Gamma(alpha=4, beta=0.04): mean alpha/beta = 100, standard deviation 50.
REPAIR_COST = surv.Gamma.from_params([4.0, 0.04])


def test_cost_distribution_enters_the_closed_form_through_its_mean():
    rbd = one_component({"repair_cost": REPAIR_COST})
    assert rbd.expected_cost_rate() == pytest.approx(100.0 * OMEGA)


def test_random_costs_average_to_the_distribution_mean():
    rbd = one_component({"repair_cost": REPAIR_COST})
    result = rbd.availability(t_simulation=200.0, N=100, seed=6)
    # ~1,800 failures, each charged a fresh draw: the average charge per
    # failure is the mean, 100, to within a standard error of about 1.2.
    charged = result.cost.by_category["repair"] * 100
    assert charged / result.system_failures == pytest.approx(100.0, rel=0.05)


def test_pricing_never_changes_the_failure_simulation():
    # Cost draws have their own random stream, so for a given seed the
    # failures, repairs and every availability output are exactly those of
    # the unpriced RBD, whether the costs are fixed or random -- only the
    # money differs.
    def simulate(spec):
        rbd = one_component(spec)
        return rbd.availability(t_simulation=200.0, N=100, seed=6)

    unpriced = simulate({})
    fixed = simulate({"repair_cost": 100.0, "downtime_cost": 22.0})
    drawn = simulate({"repair_cost": REPAIR_COST, "downtime_cost": 22.0})
    for priced in (fixed, drawn):
        assert np.array_equal(priced.availability, unpriced.availability)
        assert priced.system_failures == unpriced.system_failures
        assert priced.node_downtime == unpriced.node_downtime
    assert (
        drawn.cost.by_category["component_downtime"]
        == fixed.cost.by_category["component_downtime"]
    )
    # Random prices add spread on top of the random failures.
    assert drawn.cost.std > fixed.cost.std


def test_random_costs_are_reproducible():
    rbd = one_component({"repair_cost": REPAIR_COST})
    a = rbd.cost(t_simulation=100.0, N=50, seed=9)
    b = rbd.cost(t_simulation=100.0, N=50, seed=9)
    assert np.array_equal(a.samples, b.samples)
    # Seeding numpy's global RNG, rather than passing a seed, works too.
    np.random.seed(9)
    c = rbd.cost(t_simulation=100.0, N=50)
    np.random.seed(9)
    d = rbd.cost(t_simulation=100.0, N=50)
    assert np.array_equal(c.samples, d.samples)


def test_bad_cost_distributions_are_rejected():
    # Normal(50, 50) is negative 16% of the time.
    with pytest.raises(ValueError, match="negative cost"):
        one_component({"repair_cost": surv.Normal.from_params([50.0, 50.0])})
    # A log-logistic with shape < 1 has no finite mean.
    with pytest.raises(ValueError, match="finite mean"):
        one_component(
            {"replace_cost": surv.LogLogistic.from_params([100.0, 0.8])}
        )
    # Downtime costs are rates; the outage durations already randomise them.
    with pytest.raises(ValueError, match="not a distribution"):
        one_component({"downtime_cost": REPAIR_COST})
    with pytest.raises(ValueError, match="not a distribution"):
        one_component(downtime_cost_rate=REPAIR_COST)
    # A Normal ten standard deviations clear of zero is fine in practice.
    one_component({"repair_cost": surv.Normal.from_params([500.0, 50.0])})


def test_cost_distributions_survive_a_json_round_trip():
    rbd = one_component({"repair_cost": REPAIR_COST, "replace_cost": 400.0})
    restored = RepairableRBD.from_json(json.dumps(json.loads(rbd.to_json())))
    assert restored.expected_cost_rate() == pytest.approx(
        rbd.expected_cost_rate()
    )
    a = rbd.cost(t_simulation=100.0, N=20, seed=1)
    b = restored.cost(t_simulation=100.0, N=20, seed=1)
    assert np.array_equal(a.samples, b.samples)
