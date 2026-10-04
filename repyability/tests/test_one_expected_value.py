"""One expected value per result (#223).

A cost result's ``mean`` is its run's estimate of the expected cost, the
one ``mean_interval`` gives: exact where the exact methods work it out,
given the modules' histories where those apply, and otherwise the
simulations' own. ``mean_se``, ``cost_rate`` and the breakdowns follow it;
the simulations' own mean is ``sample_mean``. An availability result's
``mean_availability`` is likewise ``mean_availability_interval``'s
estimate, and ``sample_mean_availability`` the simulations' own.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd.results import AvailabilityResult
from repyability.tests.catalogue import repairable_kinds


def unit(failure_rate, repair_rate, **costs):
    return {
        "reliability": surv.Exponential.from_params([failure_rate]),
        "repairability": surv.Exponential.from_params([repair_rate]),
        **costs,
    }


def plant():
    """The costs guide's plant: two pumps in parallel, then a valve."""
    return RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")],
        {
            "A": unit(0.1, 1.0, repair_cost=200.0),
            "B": unit(0.1, 1.0, repair_cost=200.0),
            "C": unit(0.02, 0.5, repair_cost=500.0, replace_cost=1500.0),
        },
        downtime_cost_rate=1000.0,
    )


def agrees(cost):
    """The result's mean, error and rate are its interval's, and its
    breakdown sums to its mean."""
    interval = cost.mean_interval()
    assert cost.mean == interval.estimate
    assert cost.mean_se == interval.standard_error or (
        math.isnan(cost.mean_se) and math.isnan(interval.standard_error)
    )
    assert cost.cost_rate == cost.mean / cost.t_simulation
    assert cost.sample_mean == pytest.approx(float(np.mean(cost.samples)))
    assert math.fsum(cost.by_category.values()) == pytest.approx(
        cost.mean, rel=1e-12
    )
    return interval


def test_an_exact_mean_is_the_expected_cost_and_its_split():
    rbd = plant()
    cost = rbd.cost(1000.0, mc_samples=200, seed=0)
    interval = agrees(cost)
    assert interval.method == "exact" and cost.mean_se == 0.0
    expected = rbd.expected_cost(1000.0)
    assert cost.mean == float(expected.mean)
    for key, value in expected.by_category.items():
        assert cost.by_category[key] == pytest.approx(float(value))
    for node, value in expected.by_component.items():
        assert cost.by_component[node] == pytest.approx(float(value))
    # The simulations' own are beside it.
    assert cost.sample_mean != cost.mean
    assert cost.sample_se > 0.0


def test_the_mean_availability_is_the_interval_s():
    rbd = plant()
    result = rbd.availability(1000.0, mc_samples=200, seed=0)
    interval = result.mean_availability_interval()
    assert interval.method == "exact"
    assert result.mean_availability == interval.estimate
    assert result.mean_availability == pytest.approx(
        float(np.ravel(rbd.mission_availability(1000.0))[0])
    )
    assert result.sample_mean_availability == pytest.approx(
        result.system_uptime / (200 * 1000.0)
    )
    assert result.sample_mean_availability == pytest.approx(
        float(np.mean(result.uptimes)) / 1000.0
    )
    agrees(result.cost)


def test_a_plain_run_s_mean_is_its_own():
    rbd = plant()
    cost = rbd.cost(1000.0, mc_samples=200, seed=0, control_variate=False)
    interval = agrees(cost)
    assert interval.method == "simulated"
    assert cost.mean == cost.sample_mean
    assert cost.mean_se == cost.sample_se
    result = rbd.availability(
        1000.0, mc_samples=200, seed=0, control_variate=False
    )
    assert result.mean_availability == pytest.approx(
        result.sample_mean_availability, rel=1e-14
    )
    # The same simulations as the default run's.
    default = rbd.cost(1000.0, mc_samples=200, seed=0)
    np.testing.assert_array_equal(cost.samples, default.samples)


def test_a_conditional_mean_and_its_split():
    rbd = repairable_kinds()["imperfect repair"]
    cost = rbd.cost(300.0, mc_samples=100, seed=0)
    assert cost.conditional is not None and cost.conditional.whole
    interval = agrees(cost)
    assert interval.method == "conditional"
    assert cost.mean == pytest.approx(float(np.mean(cost.conditional.costs)))
    result = rbd.availability(300.0, mc_samples=100, seed=0)
    assert result.mean_availability == pytest.approx(
        float(np.mean(result.conditional.uptimes)) / 300.0
    )
    alone = rbd.cost(300.0, mc_samples=100, seed=0, conditional=True)
    agrees(alone)
    assert alone.mean == pytest.approx(alone.sample_mean)


def test_a_twin_controls_the_mean_but_not_its_split():
    rbd = repairable_kinds()["one repair crew, exponential"]
    cost = rbd.cost(300.0, mc_samples=100, seed=0, control_variate=True)
    interval = cost.mean_interval()
    assert interval.method == "control_variate"
    assert cost.mean == interval.estimate
    assert cost.mean != cost.sample_mean
    assert math.fsum(cost.by_category.values()) == pytest.approx(
        cost.sample_mean, rel=1e-12
    )


def test_merged_chunks_take_the_same_means():
    rbd = plant()
    chunks = [
        rbd.simulate_chunk(500.0, a, b, seed=4)
        for a, b in [(0, 60), (60, 100)]
    ]
    merged = rbd.availability_from_chunks(chunks).cost
    whole = rbd.cost(500.0, mc_samples=100, seed=4)
    assert merged.mean == whole.mean
    assert merged.by_category == pytest.approx(whole.by_category)
    np.testing.assert_array_equal(merged.samples, whole.samples)


def test_a_result_without_up_times_falls_back_to_its_totals():
    result = AvailabilityResult(
        timeline=np.array([0.0, 10.0]),
        availability=np.array([1.0, 1.0]),
        system_uptime=18.0,
        time_simulated_to=10.0,
        criticalities=None,
        node_uptime={},
        node_downtime={},
        system_downtime=2.0,
        system_failures=1,
        system_restorations=1,
        n_simulations=2,
    )
    assert result.mean_availability == 0.9
    assert result.sample_mean_availability == 0.9
    assert "(mean_availability=" not in repr(result)  # no estimate to give
    assert "sample_mean_availability=0.9" in repr(result)


def test_the_reprs_are_summaries():
    rbd = plant()
    result = rbd.availability(1000.0, mc_samples=2000, seed=0)
    text = repr(result)
    assert len(text) < 600
    assert "method='exact'" in text and "sample_mean=" in text
    assert repr(result.cost).startswith("CostResult(mean=")
    alone = repairable_kinds()["imperfect repair"].cost(
        300.0, mc_samples=50, seed=0, conditional=True
    )
    # A run of the modules alone has no spread to show.
    assert "std=" not in repr(alone)
