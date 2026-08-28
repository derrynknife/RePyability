"""Tests for the closed-form cost rate on :class:`RepairableRBD`.

Every expectation here is a hand-computed reference value rather than a
recorded output. With Exponential reliability and repairability a component
has ``MTTF = 1/lambda``, ``MTTR = 1/mu``, availability ``A = MTTF/(MTTF+MTTR)``
and long-run failure frequency ``omega = 1/(MTTF+MTTR)``, so every cost term
is exact arithmetic.
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
    # The category means partition the overall mean...
    assert sum(result.by_category.values()) == pytest.approx(result.mean)
    # ...and with one costed component, its attributable share is everything
    # except the (system-level) downtime cost.
    assert result.by_component["c"] == pytest.approx(
        result.by_category["corrective"]
        + result.by_category["component_downtime"]
    )
    assert result.n_simulations == len(result.samples) == 300
    assert result.percentile(100) == pytest.approx(result.samples.max())
    assert result.std > 0.0


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
