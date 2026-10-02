"""Delivered capacity over time in the availability simulation (#99).

With capacities of 1, a system's capacity is whether it is up, so the
capacity curve is the availability curve. Over a long window the simulated
averages approach the exact long-run values of ``capacity_distribution``.
A deterministic trace checks every recorded change and total.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import DegradingNode, RepairableRBD
from repyability.rbd.repairable_rbd import _CapacityRecorder

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


BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "e"),
    ("b", "e"),
    ("d", "t"),
    ("e", "t"),
]


@pytest.mark.parametrize(
    "edges, capacity, k",
    [
        # A bridge (worked out by conditioning), with capacities whose
        # totals round, and with some nodes unlimited.
        (BRIDGE, {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.7, "e": 1.1}, None),
        (BRIDGE, {"a": 10.0, "c": 0.1, "d": {0.2: 1.0}}, None),
        # A 2-out-of-3 vote after a pair.
        (
            [("s", "a"), ("s", "b"), ("a", "j"), ("b", "j")]
            + [("j", u) for u in "xyz"]
            + [(u, "t") for u in "xyz"],
            {"a": 0.3, "b": 0.6, "j": 1.0, "x": 0.1, "y": 0.2, "z": 0.4},
            {"t": 2},
        ),
    ],
)
def test_states_worked_out_together_are_each_on_its_own(edges, capacity, k):
    # The compiled engine works out the capacity states a batch meets
    # together (#155): with one level per node, exactly as on its own.
    nodes = sorted({u for edge in edges for u in edge} - {"s", "t"})
    rbd = RepairableRBD(
        edges, {u: UNIT for u in nodes}, capacity=capacity, k=k
    )
    ups = np.array(list(itertools.product((1, 0), repeat=len(nodes))))
    for demand in (None, 0.5):
        together = _CapacityRecorder(rbd, demand)
        together._distribution = None  # none of them on its own
        states = together.evaluate(nodes, ups)
        alone = _CapacityRecorder(rbd, demand)
        for up, state in zip(ups, states):
            down = frozenset(u for u, works in zip(nodes, up) if not works)
            assert repr(state) == repr(alone(down))


def test_states_with_levels_are_worked_out_one_at_a_time():
    # A node with several levels: a state's probabilities are then not
    # all 0 or 1, and each is worked out on its own.
    rbd = RepairableRBD(
        [("s", u) for u in "abc"] + [(u, "t") for u in "abc"],
        {u: UNIT for u in "abc"},
        capacity={"a": {100: 0.75, 40: 0.25}, "b": 50, "c": 50},
    )
    ups = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]])
    states = _CapacityRecorder(rbd, None).evaluate(list("abc"), ups)
    alone = _CapacityRecorder(rbd, None)
    for down, state in zip(["", "b", "ab"], states):
        assert repr(state) == repr(alone(frozenset(down)))
    assert states[1].levels == (90.0, 150.0)
