"""Delivered capacity over time in the availability simulation (#99).

With capacities of 1, a system's capacity is whether it is up, so the
capacity curve is the availability curve. Over a long window the simulated
averages approach the exact long-run values of ``capacity_distribution``.
A deterministic trace checks every recorded change and total.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import DegradingNode, RepairableRBD

E = surv.Exponential.from_params
X = surv.ExactEventTime.from_params
UNIT = {"reliability": E([0.1]), "repairability": E([1.0])}


def pumps(**kwargs):
    return RepairableRBD(
        [("s", u) for u in "abc"] + [(u, "t") for u in "abc"],
        {u: UNIT for u in "abc"},
        capacity={u: 50 for u in "abc"},
        **kwargs,
    )


def test_capacities_of_one_reproduce_the_availability():
    series = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": UNIT, "b": UNIT},
        capacity={"a": 1, "b": 1},
    )
    result = series.availability(50, mc_samples=300, seed=1)
    np.testing.assert_array_equal(result.capacity_timeline, result.timeline)
    np.testing.assert_allclose(result.capacity, result.availability)
    window = result.n_simulations * result.time_simulated_to
    assert result.mean_capacity == pytest.approx(result.system_uptime / window)
    # The demand is the design capacity, 1: what is delivered is the time up.
    assert result.demand == 1.0
    np.testing.assert_allclose(result.delivered, result.uptimes / 50)
    # In parallel the capacity can be 2, but against a demand of 1 the
    # delivered fraction is still the time up.
    pair = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": UNIT, "b": UNIT},
        capacity={"a": 1, "b": 1},
    )
    result = pair.availability(50, mc_samples=300, seed=2, demand=1)
    np.testing.assert_allclose(result.delivered, result.uptimes / 50)


def test_the_long_run_average_matches_the_exact_capacity():
    plant = pumps()
    exact = plant.capacity_distribution()
    result = plant.availability(2000, mc_samples=200, seed=2, demand=100)
    assert result.mean_capacity == pytest.approx(exact.mean(), rel=2e-3)
    interval = result.delivered_fraction_interval(0.999)
    assert interval.lower <= exact.delivered_fraction(100) <= interval.upper
    assert result.delivered_fraction == interval.estimate
    total = sum(result.capacity_time.values())
    assert total == pytest.approx(200 * 2000)
    assert list(result.capacity_time) == exact.levels.tolist()
    np.testing.assert_allclose(
        [result.capacity_time[level] / total for level in exact.levels],
        exact.probabilities,
        atol=2e-3,
    )


def test_a_deterministic_trace():
    # Two pumps of 1 and 2 in parallel: "a" fails every 10 and takes 2 to
    # repair, "b" every 15 and takes 4. Capacity is lost while the system
    # stays up, and both are down together at 34.
    plant = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {"reliability": X([10]), "repairability": X([2])},
            "b": {"reliability": X([15]), "repairability": X([4])},
        },
        capacity={"a": 1, "b": 2},
    )
    result = plant.availability(40, mc_samples=2, seed=0)
    np.testing.assert_array_equal(
        result.capacity_timeline, [0, 10, 12, 15, 19, 22, 24, 34, 36, 38, 40]
    )
    np.testing.assert_array_equal(
        result.capacity, [3, 2, 3, 1, 3, 2, 3, 0, 1, 3, 3]
    )
    assert result.capacity_time == {0.0: 4.0, 1.0: 12.0, 2.0: 8.0, 3.0: 56.0}
    assert result.mean_capacity == pytest.approx((12 + 16 + 168) / 80)
    # The design capacity, 3, is the demand: 28 of 40 at 3, then shares.
    delivered = (28 * 3 + 4 * 2 + 6 * 1) / (3 * 40)
    np.testing.assert_allclose(result.delivered, [delivered, delivered])


def test_levels_while_up_count_in_proportion():
    plant = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": UNIT},
        capacity={"a": {100: 0.8, 50: 0.2}},
    )
    result = plant.availability(2000, mc_samples=100, seed=3)
    exact = plant.capacity_distribution()
    assert result.demand == 100.0
    assert result.mean_capacity == pytest.approx(exact.mean(), rel=5e-3)
    assert result.delivered_fraction == pytest.approx(
        exact.delivered_fraction(100), rel=5e-3
    )
    # Every moment up is split between the levels, 4 to 1.
    time = result.capacity_time
    assert time[100.0] == pytest.approx(4 * time[50.0])
    assert time[50.0] + time[100.0] == pytest.approx(result.system_uptime)


def test_an_unlimited_capacity():
    # "b" has no capacity, so while it is up the system carries anything.
    plant = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": UNIT, "b": UNIT},
        capacity={"a": 5},
    )
    result = plant.availability(50, mc_samples=50, seed=0)
    assert result.capacity[0] == math.inf
    assert result.mean_capacity == math.inf
    # No finite design capacity, so no delivered fraction without a demand.
    assert result.demand is None and result.delivered is None
    assert result.delivered_fraction is None
    with pytest.raises(ValueError, match="no delivered fractions"):
        result.delivered_fraction_interval()
    result = plant.availability(2000, mc_samples=100, seed=0, demand=5)
    assert result.delivered_fraction == pytest.approx(
        plant.capacity_distribution().delivered_fraction(5), abs=2e-3
    )


def test_parallel_runs_give_the_same_capacity():
    plant = pumps()
    one = plant.availability(100, mc_samples=500, seed=5, n_jobs=1, demand=100)
    two = plant.availability(100, mc_samples=500, seed=5, n_jobs=2, demand=100)
    np.testing.assert_array_equal(one.capacity_timeline, two.capacity_timeline)
    np.testing.assert_array_equal(one.capacity, two.capacity)
    np.testing.assert_array_equal(one.delivered, two.delivered)
    assert one.capacity_time == two.capacity_time
    # Every block's simulations are kept.
    assert len(one.delivered) == 500
    assert sum(one.capacity_time.values()) == pytest.approx(500 * 100)


def test_antithetic_pairs_and_the_interval():
    result = pumps().availability(
        200, mc_samples=100, seed=7, antithetic=True, demand=100
    )
    interval = result.delivered_fraction_interval()
    assert interval.n_samples == 100
    assert interval.lower <= result.delivered_fraction <= interval.upper
    with pytest.raises(ValueError, match="confidence"):
        result.delivered_fraction_interval(1.5)


def test_without_capacities_nothing_is_recorded():
    result = RepairableRBD([("s", "a"), ("a", "t")], {"a": UNIT}).availability(
        10, mc_samples=5, seed=0
    )
    for name in (
        "capacity_timeline",
        "capacity",
        "capacity_time",
        "demand",
        "delivered",
    ):
        assert getattr(result, name) is None
        assert name in result
    assert result.mean_capacity is None
    assert result.delivered_fraction is None
    # The failures and repairs are the same with or without capacities.
    with_capacity = RepairableRBD(
        [("s", "a"), ("a", "t")], {"a": UNIT}, capacity={"a": 2}
    ).availability(10, mc_samples=5, seed=0)
    np.testing.assert_array_equal(with_capacity.uptimes, result.uptimes)


@pytest.mark.parametrize("demand", [0.0, -1.0, math.inf, math.nan])
def test_the_demand_must_be_positive_and_finite(demand):
    with pytest.raises(ValueError, match="positive, finite"):
        pumps().availability(10, mc_samples=2, seed=0, demand=demand)


def test_a_demand_needs_capacities():
    plain = RepairableRBD([("s", "a"), ("a", "t")], {"a": UNIT})
    with pytest.raises(ValueError, match="no node has one"):
        plain.availability(10, mc_samples=2, seed=0, demand=5)


def test_capacities_from_models_are_not_simulated():
    stages = DegradingNode([(100, E([0.1])), (50, E([0.2]))])
    staged = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": stages, "repairability": E([1.0])}},
    )
    with pytest.raises(NotImplementedError, match="DegradingNode"):
        staged.availability(10, mc_samples=2, seed=0)
    # Given a capacity of its own, the node is followed as up or down.
    given = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": stages, "repairability": E([1.0])}},
        capacity={"a": 80},
    )
    result = given.availability(10, mc_samples=2, seed=0)
    assert result.demand == 80.0
    # The cost simulation does not follow the capacity.
    assert staged.cost(10, mc_samples=2, seed=0) is None
