"""Exact long-run values with shared repair crews (#90): the Markov chain of
the components' states and the repair queue, against the machine-repair
model's closed forms, chains worked by hand and the simulation of #89, and
what it refuses."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _crew_chain
from repyability.rbd import routes as r

E, W, L = (
    surv.Exponential.from_params,
    surv.Weibull.from_params,
    surv.LogNormal.from_params,
)


def machine_repair(n, lam, mu, crews):
    """The long-run probabilities that 0, 1, ..., n of ``n`` identical
    exponential units are down, with ``crews`` crews: a birth-death chain,
    failing at ``(n - j) * lam`` and repaired at ``min(j, crews) * mu``."""
    weights = [1.0]
    for j in range(n):
        weights.append(weights[-1] * (n - j) * lam / (min(j + 1, crews) * mu))
    return np.array(weights) / sum(weights)


def units(n, lam, mu, crews, k=1, capacity=None, cost_rate=0.0, **spec):
    """``n`` identical exponential units, ``k`` of them needed."""
    unit = {"reliability": E([lam]), "repairability": E([mu]), **spec}
    edges = [("s", i) for i in range(n)] + [(i, "t") for i in range(n)]
    return RepairableRBD(
        edges,
        {i: dict(unit) for i in range(n)},
        k={"t": k} if k > 1 else None,
        capacity=capacity,
        downtime_cost_rate=cost_rate,
        repair_crews=crews,
    )


@pytest.mark.parametrize(
    "n, lam, mu, crews, k",
    [
        (3, 0.1, 0.5, 1, 1),
        (3, 0.1, 0.5, 2, 1),
        (4, 0.2, 0.5, 1, 2),
        (5, 0.05, 1.0, 2, 3),
        (6, 1e-3, 1.0, 1, 4),
    ],
)
def test_k_of_n_units_match_the_machine_repair_model(n, lam, mu, crews, k):
    p = machine_repair(n, lam, mu, crews)
    up = p[: n - k + 1].sum()
    # It fails when n - k units are down and one of the k up fails.
    frequency = p[n - k] * k * lam
    rbd = units(n, lam, mu, crews, k)
    assert rbd.mean_availability() == pytest.approx(up, rel=1e-12)
    assert rbd.mean_unavailability() == pytest.approx(1 - up, rel=1e-8)
    assert rbd.system_failure_frequency() == pytest.approx(
        frequency, rel=1e-12
    )
    assert rbd.mean_time_between_failures() == pytest.approx(
        1 / frequency, rel=1e-12
    )
    assert rbd.mean_up_time() == pytest.approx(up / frequency, rel=1e-12)
    assert rbd.mean_down_time() == pytest.approx(
        (1 - up) / frequency, rel=1e-8
    )
    unit = sum(p[j] * (n - j) / n for j in range(n + 1))
    expected = {**{i: unit for i in range(n)}, "s": 1.0, "t": 1.0}
    assert rbd.node_availability() == pytest.approx(expected, rel=1e-12)


def test_tiny_probabilities_keep_their_precision():
    # Repairs 1e9 times as fast as failures: with every unit down the
    # probability is about 1e-45, and still exact to rounding.
    n, crews = 5, 2
    chain = _crew_chain.solve(
        range(n), [1e-9] * n, [1.0] * n, [0.0] * n, crews
    )
    down = (~chain.up).sum(axis=1)
    by_count = np.bincount(down, weights=chain.probabilities, minlength=n + 1)
    np.testing.assert_allclose(
        by_count, machine_repair(n, 1e-9, 1.0, crews), rtol=1e-12
    )


def test_the_states_are_counted_before_they_are_built():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(1, 7))
        crews = int(rng.integers(1, 4))
        priorities = [float(p) for p in rng.integers(0, 3, n)]
        repairs = [math.inf if rng.random() < 0.3 else 1.0 for _ in range(n)]
        states = _crew_chain._enumerate([0.1] * n, repairs, priorities, crews)
        instant = [math.isinf(rate) for rate in repairs]
        count = _crew_chain.state_count(priorities, instant, crews)
        assert count == len(states[0])


def test_an_instant_repair_waits_only_for_a_busy_crew():
    # a (failing at 0.1, repaired at 1) and b (failing at 0.2, repaired
    # instantly) in series, with one crew. b, failing while a is under
    # repair, waits until a is done, and is then repaired at once.
    la, mu, lb = 0.1, 1.0, 0.2
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {"reliability": E([la]), "repairability": E([mu])},
            "b": {"reliability": E([lb]), "repairability": "instant"},
        },
        repair_crews=1,
    )
    # All up, a under repair, and a under repair with b waiting.
    weights = np.array([1.0, la / (mu + lb), la / (mu + lb) * lb / mu])
    p = weights / weights.sum()
    assert rbd.mean_availability() == pytest.approx(p[0], rel=1e-12)
    availability = rbd.node_availability()
    assert availability["a"] == pytest.approx(p[0], rel=1e-12)
    assert availability["b"] == pytest.approx(1 - p[2], rel=1e-12)
    # With everything up, b's failures are outages of no length.
    assert rbd.system_failure_frequency() == pytest.approx(
        p[0] * (la + lb), rel=1e-12
    )


def test_priorities_decide_who_waits():
    # One crew for three units, and a's repairs are quick. Put ahead in
    # the queue, a waits less and the others more.
    def trio(priority):
        def unit(mu, **more):
            return {"reliability": E([0.2]), "repairability": E([mu]), **more}

        return RepairableRBD(
            [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
            {
                "a": unit(5.0, priority=priority),
                "b": unit(0.5),
                "c": unit(0.5),
            },
            repair_crews=1,
        )

    first, ahead = trio(0.0).node_availability(), trio(1.0).node_availability()
    assert ahead["a"] > first["a"]
    assert ahead["b"] < first["b"] and ahead["c"] < first["c"]


def test_a_node_held_working_or_broken_needs_no_crew():
    n, lam, mu, crews = 3, 0.1, 0.5, 1
    rbd = units(n, lam, mu, crews, k=2)
    two = machine_repair(2, lam, mu, crews)
    # Held working, 0 leaves 1 and 2 to share the crew, and one of them is
    # enough.
    assert rbd.mean_availability(working_nodes=[0]) == pytest.approx(
        1 - two[2], rel=1e-12
    )
    # Held broken, 0 takes no crew, and both 1 and 2 are needed.
    assert rbd.mean_availability(broken_nodes=[0]) == pytest.approx(
        two[0], rel=1e-12
    )
    assert rbd.system_failure_frequency(broken_nodes=[0]) == pytest.approx(
        two[0] * 2 * lam, rel=1e-12
    )


def test_waiting_for_a_crew_is_priced_as_downtime():
    n, lam, mu, crews = 3, 0.1, 0.5, 1
    rbd = units(
        n,
        lam,
        mu,
        crews,
        k=2,
        cost_rate=20.0,
        repair_cost=4.0,
        downtime_cost=3.0,
    )
    p = machine_repair(n, lam, mu, crews)
    unit = sum(p[j] * (n - j) / n for j in range(n + 1))
    expected = 20.0 * (1 - p[:2].sum()) + n * (
        4.0 * lam * unit + 3.0 * (1 - unit)
    )
    assert rbd.expected_cost_rate() == pytest.approx(expected, rel=1e-12)
    assert rbd.total_cost(100.0) == pytest.approx(100 * expected, rel=1e-12)
    # Held broken, 0 is down for good and never repaired.
    two = machine_repair(2, lam, mu, crews)
    held = sum(two[j] * (2 - j) / 2 for j in range(3))
    expected = (
        20.0 * (1 - two[0]) + 3.0 + 2 * (4.0 * lam * held + 3.0 * (1 - held))
    )
    assert rbd.expected_cost_rate(broken_nodes=[0]) == pytest.approx(
        expected, rel=1e-12
    )


def test_the_capacity_follows_the_crews():
    n, lam, mu, crews = 3, 0.1, 0.5, 1
    rbd = units(n, lam, mu, crews, capacity={i: 50.0 for i in range(n)})
    capacity = rbd.capacity_distribution()
    assert capacity.levels.tolist() == [0.0, 50.0, 100.0, 150.0]
    np.testing.assert_allclose(
        capacity.probabilities, machine_repair(n, lam, mu, crews)[::-1]
    )


def test_a_nested_rbd_brings_its_own_crews():
    lam, mu = 0.1, 0.5
    inner = units(2, lam, mu, 1)
    p = machine_repair(2, lam, mu, 1)
    inner_up, inner_frequency = 1 - p[2], p[1] * lam
    assert inner.mean_availability() == pytest.approx(inner_up, rel=1e-12)
    # Outside, one crew for y and z in parallel, in series with the nested
    # RBD, whose units its own crew repairs: the two halves are
    # independent.
    unit = {"reliability": E([0.05]), "repairability": E([0.25])}
    outer = RepairableRBD(
        [("s", "x"), ("x", "y"), ("x", "z"), ("y", "t"), ("z", "t")],
        {"x": inner, "y": dict(unit), "z": dict(unit)},
        repair_crews=1,
    )
    q = machine_repair(2, 0.05, 0.25, 1)
    pair_up, pair_frequency = 1 - q[2], q[1] * 0.05
    assert outer.mean_availability() == pytest.approx(
        inner_up * pair_up, rel=1e-12
    )
    assert outer.system_failure_frequency() == pytest.approx(
        inner_frequency * pair_up + pair_frequency * inner_up, rel=1e-12
    )
    assert outer.analysis_routes()["mean_availability"].route == r.EXACT


BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("b", "e"),
    ("c", "d"),
    ("c", "e"),
    ("d", "t"),
    ("e", "t"),
]


def bridge(crews, instant=False, **changes):
    """A bridge of five exponential components with priorities and
    costs; ``changes`` replaces some of a component's spec."""
    rates = {
        "a": (0.02, 0.2, 1),
        "b": (0.03, 0.5, 0),
        "c": (0.05, 0.25, 2),
        "d": (0.01, 0.1, 0),
        "e": (0.04, 1.0, 1),
    }
    components = {}
    for node, (lam, mu, priority) in rates.items():
        components[node] = {
            "reliability": E([lam]),
            "repairability": "instant" if instant and node == "e" else E([mu]),
            "priority": priority,
            "repair_cost": 2.0,
            "downtime_cost": 1.0,
            **changes.get(node, {}),
        }
    return RepairableRBD(
        BRIDGE, components, repair_crews=crews, downtime_cost_rate=10.0
    )


@pytest.mark.parametrize("crews, instant", [(1, False), (2, False), (1, True)])
def test_the_exact_values_agree_with_the_simulation(crews, instant):
    rbd = bridge(crews, instant)
    window, samples = 10_000.0, 40
    run = rbd.availability(t_simulation=window, mc_samples=samples, seed=11)
    interval = run.mean_availability_interval(confidence=0.999)
    assert interval.lower <= rbd.mean_availability() <= interval.upper
    spread = math.sqrt(run.system_failures) / (samples * window)
    frequency = rbd.system_failure_frequency()
    assert abs(run.failure_frequency - frequency) < 4 * spread
    for node, up in rbd.node_availability().items():
        if node in rbd.components:
            assert run.node_uptime[node] / (samples * window) == (
                pytest.approx(up, abs=0.01)
            )
    cost = run.cost.mean_interval(confidence=0.999)
    rate = rbd.expected_cost_rate()
    assert cost.lower / window <= rate <= cost.upper / window


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"reliability": W([50, 1.5])}, "exponential lives"),
        ({"repairability": L([0, 0.5])}, "exponential repair times"),
        ({"preventive": {"interval": 5.0}}, "scheduled maintenance"),
        (
            {"inspection": {"interval": 5.0}, "repairability": "instant"},
            "inspections",
        ),
    ],
)
def test_what_the_chain_does_not_cover_is_refused(change, reason):
    rbd = bridge(1, a=change)
    with pytest.raises(NotImplementedError, match=reason) as error:
        rbd.mean_availability()
    route = rbd.analysis_routes()["mean_availability"]
    assert route.route == r.REFUSED and route.reason == str(error.value)
    # The simulation still runs.
    rbd.availability(t_simulation=50.0, mc_samples=2, seed=1)


def test_a_chain_too_large_to_solve_is_refused():
    # Eight units first come, first served, with two crews: the queue's
    # order makes 54,805 states.
    rbd = units(8, 0.1, 1.0, 2)
    with pytest.raises(NotImplementedError, match="54,805 states"):
        rbd.mean_availability()
    assert rbd.analysis_routes()["mean_availability"].route == r.REFUSED
    # With a priority each the order is fixed: 1,801 states.
    ranked = RepairableRBD(
        [("s", i) for i in range(8)] + [(i, "t") for i in range(8)],
        {
            i: {
                "reliability": E([0.1]),
                "repairability": E([1.0]),
                "priority": i,
            }
            for i in range(8)
        },
        repair_crews=2,
    )
    assert 0.99 < ranked.mean_availability() < 1.0


def test_what_assumes_independent_components_is_refused():
    # The allocations; the importance measures and the values over time
    # come from the chain (#146: see test_chains_over_time.py).
    rbd = units(3, 0.1, 0.5, 1, acquisition_cost=1.0)
    report = rbd.analysis_routes()
    for name in (
        "birnbaum_importance",
        "improvement_potential",
        "risk_achievement_worth",
        "risk_reduction_worth",
        "criticality_importance",
        "fussell_vesely",
    ):
        assert report[name].route == r.EXACT
        getattr(rbd, name)()
    for name in ("point_availability", "mission_availability"):
        assert report[name].route == r.NUMERICAL
        getattr(rbd, name)(1.0)
    for name, call in {
        "availability_allocation": lambda: rbd.availability_allocation(0.99),
        "mttf_mttr_allocation": lambda: rbd.mttf_mttr_allocation(0.99),
        "allocate_redundancy": lambda: rbd.allocate_redundancy(100.0),
    }.items():
        with pytest.raises(NotImplementedError, match="allocations") as error:
            call()
        assert report[name].reason == str(error.value)
    assert report["mean_availability"].route == r.EXACT
    assert report["expected_cost_rate"].route == r.EXACT


def test_the_chain_follows_a_change_of_crews():
    rbd = units(3, 0.1, 0.5, 1)
    one = rbd.mean_availability()
    assert one == pytest.approx(1 - machine_repair(3, 0.1, 0.5, 1)[3])
    rbd.repair_crews = 2
    assert rbd.mean_availability() == pytest.approx(
        1 - machine_repair(3, 0.1, 0.5, 2)[3], rel=1e-12
    )
    # As many crews as units: nothing waits, and the units are independent.
    rbd.repair_crews = 3
    assert rbd.mean_availability() == pytest.approx(
        1 - (0.1 / 0.6) ** 3, rel=1e-12
    )
