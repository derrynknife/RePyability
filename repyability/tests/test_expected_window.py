"""Expected failures, outages, downtime and cost over a window from new
(#123): ``RepairableRBD.expected_failures``, ``expected_events`` and
``expected_cost``.

They are checked against closed forms (exponential components, the
time-dependent Birnbaum/Vesely formula integrated by quadrature, tests of
hidden failures), against timelines that can be worked out by hand (units
that never fail before their replacement, replaced together), against the
long-run rates they settle at, and against the simulation's means.
"""

import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import ExpectedCost, ExpectedEvents, RepairableRBD
from repyability.rbd import routes

E = surv.Exponential.from_params
W = surv.Weibull.from_params
L = surv.LogNormal.from_params
X = surv.ExactEventTime.from_params

SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def unit(failure_rate, repair_rate, **more):
    return {
        "reliability": E([failure_rate]),
        "repairability": E([repair_rate]),
        **more,
    }


def available(failure_rate, repair_rate, t):
    """An exponential unit's point availability from new."""
    total = failure_rate + repair_rate
    return repair_rate / total + failure_rate / total * np.exp(-total * t)


def test_one_component_counts_its_alternating_renewals():
    lam, mu = 0.1, 1.0
    rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": unit(lam, mu)})
    t = np.array([0.0, 1.0, 10.0, 100.0, 1e4])
    total = lam + mu
    up = mu / total * t + lam / total**2 * -np.expm1(-total * t)
    failures = lam * up
    np.testing.assert_allclose(rbd.expected_failures(t), failures, rtol=2e-7)
    events = rbd.expected_events(t)
    assert isinstance(events, ExpectedEvents)
    np.testing.assert_allclose(events.window, t)
    np.testing.assert_allclose(events.system_failures, failures, rtol=2e-7)
    np.testing.assert_allclose(events.node_failures["c"], failures, rtol=2e-7)
    np.testing.assert_allclose(
        events.node_corrective["c"], failures, rtol=2e-7
    )
    # The point availability is exact to about 1e-7, so the downtime is to
    # about 1e-7 of the window.
    for downtime in (events.node_downtime["c"], events.system_downtime):
        assert np.all(np.abs(downtime - (t - up)) <= 1e-7 * t)
    assert not np.any(events.node_preventive["c"])
    assert not np.any(events.system_planned_outages)
    # A scalar window gives floats; nothing happens in a window of 0.
    assert isinstance(rbd.expected_failures(10.0), float)
    assert rbd.expected_failures(0.0) == 0.0
    window = rbd.expected_events(10.0)
    assert isinstance(window.system_failures, float)
    assert isinstance(window.node_downtime["c"], float)


@pytest.mark.parametrize("edges", [SERIES, PARALLEL])
def test_the_system_fails_by_the_time_dependent_vesely_formula(edges):
    rates = {"a": (0.1, 1.0), "b": (0.05, 0.5)}
    rbd = RepairableRBD(edges, {n: unit(*r) for n, r in rates.items()})

    def intensity(t):
        a, b = (available(*rates[n], t) for n in "ab")
        if edges is SERIES:
            return 0.1 * a * b + 0.05 * b * a
        return 0.1 * a * (1.0 - b) + 0.05 * b * (1.0 - a)

    for t in (1.0, 10.0, 50.0, 500.0):
        exact = quad(intensity, 0.0, t, limit=200)[0]
        assert rbd.expected_failures(t) == pytest.approx(exact, rel=2e-7)


def test_hidden_failures_are_counted_as_they_happen_and_as_tests_find_them():
    rate, interval = 0.002, 100.0
    tested = {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval},
    }
    rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": tested})
    per_test = -np.expm1(-rate * interval)
    for t, tests in ((1000.0, 9), (1050.0, 10), (1000.5, 10), (99.0, 0)):
        events = rbd.expected_events(t)
        whole, since = divmod(t, interval)
        failures = whole * per_test - np.expm1(-rate * since)
        assert events.node_failures["c"] == pytest.approx(failures, rel=1e-12)
        assert events.system_failures == pytest.approx(failures, rel=1e-9)
        assert events.node_inspections["c"] == tests
        assert events.node_corrective["c"] == pytest.approx(tests * per_test)
        down = whole * (interval - per_test / rate) + (
            since + np.expm1(-rate * since) / rate
        )
        assert events.node_downtime["c"] == pytest.approx(down, rel=1e-9)


def test_replacements_due_together_take_the_system_down_once():
    # Units that never fail before their replacement at 100 hours, which
    # takes 5: replaced at 100, 205, 310, ..., nine times before 1,000
    # hours. Two of them are always replaced together: in series or in
    # parallel, each time is one planned outage of the system, and one stop
    # of their maintenance group.
    def never_failing():
        return {
            "reliability": X([1e6]),
            "repairability": E([1.0]),
            "preventive": {
                "interval": 100.0,
                "duration": X([5.0]),
                "cost": 10.0,
            },
            "group": "g",
        }

    for edges in (SERIES, PARALLEL):
        rbd = RepairableRBD(
            edges,
            {"a": never_failing(), "b": never_failing()},
            maintenance_groups={"g": {"setup_cost": 100.0}},
        )
        events = rbd.expected_events(1000.0)
        assert events.system_planned_outages == pytest.approx(9.0)
        assert events.system_failures == 0.0
        assert events.node_preventive == {"a": 9.0, "b": 9.0}
        assert events.system_downtime == pytest.approx(45.0)
        cost = rbd.expected_cost(1000.0)
        assert cost.by_category["setup"] == pytest.approx(900.0)
        assert cost.by_category["preventive"] == pytest.approx(180.0)
        simulated = rbd.availability(1000.0, mc_samples=20, seed=1)
        assert simulated.system_planned_outages == 9 * 20
        assert simulated.cost.by_category["setup"] == pytest.approx(900.0)


def test_block_replacements_due_together_take_the_system_down_once():
    def block():
        return {
            "reliability": W([1e7, 2.0]),
            "repairability": E([1.0]),
            "preventive": {
                "interval": 100.0,
                "policy": "block",
                "duration": X([5.0]),
            },
            "group": "g",
        }

    rbd = RepairableRBD(
        SERIES,
        {"a": block(), "b": block()},
        maintenance_groups={"g": {"setup_cost": 100.0}},
    )
    events = rbd.expected_events(1000.0)
    assert events.system_planned_outages == pytest.approx(9.0, rel=1e-9)
    assert events.node_preventive["a"] == pytest.approx(9.0, rel=1e-9)
    assert events.system_downtime == pytest.approx(45.0, rel=1e-9)
    assert rbd.expected_cost(1000.0).by_category["setup"] == pytest.approx(
        900.0, rel=1e-9
    )


def test_units_dead_on_arrival_together_fail_the_system_once():
    doa = {
        "reliability": surv.Weibull.from_params([100.0, 1.5], f0=0.1),
        "repairability": E([1.0]),
    }
    rbd = RepairableRBD(SERIES, {"a": doa, "b": doa})
    # At 0 either may be dead: the system fails then with 1 - 0.9 ** 2.
    assert rbd.expected_failures(1e-9) == pytest.approx(0.19, rel=1e-7)
    events = rbd.expected_events(1e-9)
    assert events.node_failures["a"] == pytest.approx(0.1, rel=1e-7)
    assert rbd.expected_failures(0.0) == 0.0


def test_a_replacement_due_at_the_window_s_end_falls_after_it():
    life = W([1000.0, 2.5])
    rbd = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {
            "c": {
                "reliability": life,
                "repairability": E([0.1]),
                "preventive": {"interval": 500.0},
            }
        },
    )
    # The unit new at 0 is replaced at 500 if it survives, and the next at
    # exactly 1000 if it survives too: that one falls after a window of
    # 1000.
    events = rbd.expected_events(np.array([1000.0, 1000.0 + 1e-6]))
    jump = np.diff(events.node_preventive["c"])[0]
    survive = float(np.ravel(life.sf(500.0))[0])
    assert jump == pytest.approx(survive**2, rel=1e-6)
    simulated = rbd._simulated_replacements(1000.0, ["c"], 2000, 3)["c"]
    exact = events.node_failures["c"][0] + events.node_preventive["c"][0]
    assert simulated.mean() == pytest.approx(exact, abs=0.04)
    assert rbd.spares_demand(1000.0)["c"].mean == pytest.approx(
        exact, rel=1e-5
    )


def pump(interval, policy="age", **more):
    return {
        "reliability": W([1000.0, 2.5]),
        "repairability": L([3.0, 0.5]),
        "replace_cost": 5000.0,
        "preventive": {
            "interval": interval,
            "policy": policy,
            "duration": W([8.0, 3.0]),
            "cost": 1000.0,
        },
        **more,
    }


def nested_line():
    inner = RepairableRBD(
        PARALLEL,
        {
            "a": {
                "reliability": W([300.0, 2.0]),
                "repairability": E([0.2]),
                "preventive": {"interval": 200.0, "duration": E([1.0])},
            },
            "b": unit(0.004, 0.5),
        },
    )
    return RepairableRBD(
        [("s", "n"), ("n", "z"), ("z", "t")],
        {
            "n": inner,
            "z": {
                "reliability": W([400.0, 1.2]),
                "repairability": E([1.0]),
                "preventive": {
                    "interval": 200.0,
                    "duration": E([1.0]),
                    "cost": 7.0,
                },
                "replace_cost": 30.0,
            },
        },
        downtime_cost_rate=10.0,
    )


def systems():
    return {
        "age replacement": RepairableRBD(
            PARALLEL,
            {"a": pump(500.0), "b": pump(500.0, downtime_cost=3.0)},
            downtime_cost_rate=500.0,
        ),
        "block replacement": RepairableRBD(
            PARALLEL,
            {"a": pump(500.0, "block"), "b": pump(500.0, "block")},
            downtime_cost_rate=500.0,
        ),
        "tested": RepairableRBD(
            PARALLEL,
            {
                "a": {
                    "reliability": E([0.002]),
                    "repairability": "instant",
                    "inspection": {"interval": 100.0, "cost": 20.0},
                    "repair_cost": 50.0,
                },
                "b": {
                    "reliability": W([500.0, 1.5]),
                    "repairability": E([0.5]),
                    "repair_cost": 10.0,
                },
            },
            downtime_cost_rate=100.0,
        ),
        "nested": nested_line(),
    }


@pytest.mark.parametrize("name", ["block replacement", "tested", "nested"])
def test_the_counts_settle_at_the_long_run_rates(name):
    rbd = systems()[name]
    t = np.array([1e6, 2e6])
    cost = rbd.expected_cost(t)
    events = rbd.expected_events(t)
    failures, planned = rbd._outage_frequencies()

    def slope(values):
        return np.diff(values)[0] / 1e6

    assert slope(events.system_failures) == pytest.approx(failures, rel=1e-7)
    assert slope(events.system_planned_outages) == pytest.approx(
        planned, rel=1e-7, abs=1e-15
    )
    # The block replacements' availability from new settles within about
    # 1e-6 of their long-run profile's (see point_availability).
    tolerance = 1e-4 if name == "block replacement" else 1e-9
    assert slope(cost.mean) == pytest.approx(
        rbd.expected_cost_rate(), rel=tolerance
    )
    assert slope(events.system_downtime) == pytest.approx(
        rbd.mean_unavailability(), rel=tolerance
    )
    if name != "nested":  # (a few seconds more for the same check)
        # The downtime is the mission availability's complement.
        np.testing.assert_allclose(
            events.system_downtime,
            t * (1.0 - rbd.mission_availability(t)),
            rtol=1e-9,
        )


@pytest.mark.parametrize("name", sorted(systems()))
def test_the_expected_cost_is_what_the_simulation_estimates(name):
    rbd = systems()[name]
    t = 3000.0
    exact = rbd.expected_cost(t)
    simulated = rbd.cost(t, mc_samples=3000, seed=11)
    interval = simulated.mean_interval(0.999)
    assert interval.lower <= exact.mean <= interval.upper
    assert sum(exact.by_category.values()) == pytest.approx(exact.mean)
    assert list(exact.by_category) == list(simulated.by_category)
    assert list(exact.by_component) == list(simulated.by_component)
    events = rbd.expected_events(t)
    result = rbd.availability(t, mc_samples=3000, seed=11)
    n = result.n_simulations
    for exact_count, total in (
        (events.system_failures, result.system_failures),
        (events.system_planned_outages, result.system_planned_outages),
    ):
        # Counts of events are near Poisson: within 4 standard errors.
        spread = 4.0 * np.sqrt(max(exact_count, 1e-3) / n) + 1e-9
        assert abs(total / n - exact_count) <= spread
    assert events.system_downtime == pytest.approx(
        result.system_downtime / n, rel=0.05
    )


def test_a_nested_rbd_counts_as_the_flat_one_does():
    # The nested line's inner RBD in series with z is a flat diagram of
    # three components: the counts agree.
    nested = nested_line()
    inner = nested.components["n"]
    flat = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "z"), ("b", "z"), ("z", "t")],
        {
            "a": inner._init_args["components"]["a"],
            "b": inner._init_args["components"]["b"],
            "z": nested._init_args["components"]["z"],
        },
    )
    t = np.array([150.0, 1000.0, 5000.0])
    left, right = nested.expected_events(t), flat.expected_events(t)
    for field in ("system_failures", "system_planned_outages"):
        np.testing.assert_allclose(
            getattr(left, field), getattr(right, field), rtol=1e-6
        )
    np.testing.assert_allclose(
        left.system_downtime, right.system_downtime, rtol=1e-7
    )
    # The nested RBD's failures are its own system's; its replacements are
    # not this RBD's to count.
    np.testing.assert_allclose(
        left.node_failures["n"], inner.expected_failures(t), rtol=1e-12
    )
    assert not np.any(left.node_preventive["n"])


def test_forced_nodes_do_nothing():
    rbd = systems()["age replacement"]
    held = rbd.expected_events(2000.0, working_nodes=["a"])
    assert held.node_failures["a"] == 0.0
    assert held.node_preventive["a"] == 0.0
    assert held.node_downtime["a"] == 0.0
    # With a held working, the parallel pair never fails.
    assert held.system_failures == 0.0
    broken = rbd.expected_events(2000.0, broken_nodes=["a"])
    assert broken.node_downtime["a"] == 2000.0
    alone = RepairableRBD(
        [("s", "b"), ("b", "t")], {"b": pump(500.0, downtime_cost=3.0)}
    )
    assert broken.system_failures == pytest.approx(
        alone.expected_failures(2000.0), rel=1e-9
    )
    # Held broken, a component pays its own downtime cost throughout, but
    # nothing else.
    cost = rbd.expected_cost(2000.0, broken_nodes=["b"])
    assert cost.by_component["b"] == pytest.approx(3.0 * 2000.0)
    with pytest.raises(ValueError, match="both working and broken"):
        rbd.expected_events(10.0, working_nodes=["a"], broken_nodes=["a"])


def test_the_cost_result_adds_up():
    rbd = systems()["age replacement"]
    cost = rbd.expected_cost([1000.0, 2000.0])
    assert isinstance(cost, ExpectedCost)
    assert cost.mean.shape == (2,)
    np.testing.assert_allclose(cost.total, cost.mean + cost.acquisition_cost)
    np.testing.assert_allclose(cost.cost_rate, cost.mean / [1000.0, 2000.0])
    by_component = sum(cost.by_component.values())
    np.testing.assert_allclose(
        by_component,
        cost.mean
        - cost.by_category["system_downtime"]
        - cost.by_category["setup"],
    )
    events = rbd.expected_events([1000.0, 2000.0])
    np.testing.assert_allclose(
        cost.by_category["replace"],
        5000.0 * (events.node_corrective["a"] + events.node_corrective["b"]),
    )
    np.testing.assert_allclose(
        cost.by_category["preventive"],
        1000.0 * (events.node_preventive["a"] + events.node_preventive["b"]),
    )
    np.testing.assert_allclose(
        cost.by_category["system_downtime"], 500.0 * events.system_downtime
    )
    assert dict(cost)["mean"] is cost.mean


def test_nothing_priced_costs_nothing_without_any_checks():
    rbd = RepairableRBD(
        PARALLEL,
        {
            "a": {**unit(0.01, 0.5), "acquisition_cost": 100.0},
            "b": unit(0.01, 0.5),
        },
        repair_crews=1,
    )
    assert not rbd.has_costs
    cost = rbd.expected_cost([10.0, 20.0])
    np.testing.assert_array_equal(cost.mean, [0.0, 0.0])
    assert cost.acquisition_cost == 100.0
    np.testing.assert_array_equal(cost.total, [100.0, 100.0])
    assert rbd.analysis_routes()["expected_cost"].route == routes.EXACT
    # The counts themselves come from the crews' chain (#146; see
    # test_chains_over_time.py).
    route = rbd.analysis_routes()["expected_events"]
    assert route.route == routes.NUMERICAL
    events = rbd.expected_events(10.0)
    assert 0.0 < events.system_failures < events.node_failures["a"]


def test_the_window_and_method_are_checked():
    rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": unit(0.1, 1.0)})
    with pytest.raises(ValueError):
        rbd.expected_failures(-1.0)
    with pytest.raises(ValueError, match=r"'p' \(or 'paths'\) or 'c'"):
        rbd.expected_failures(1.0, method="x")
    assert rbd.expected_failures(10.0, method="c") == pytest.approx(
        rbd.expected_failures(10.0), rel=1e-12
    )
    events = rbd.expected_events([[1.0, 2.0], [3.0, 4.0]])
    assert events.system_failures.shape == (2, 2)
    assert events.node_downtime["c"].shape == (2, 2)


def test_the_routes_report_the_counts():
    plain = RepairableRBD(PARALLEL, {n: unit(0.01, 0.5) for n in "ab"})
    report = plain.analysis_routes()
    for name in ("expected_failures", "expected_events"):
        assert report[name].route == routes.NUMERICAL
        assert "Birnbaum/Vesely" in report[name].reason
    assert report["expected_cost"].route == routes.EXACT
    priced = RepairableRBD(
        PARALLEL,
        {n: unit(0.01, 0.5, repair_cost=1.0) for n in "ab"},
    )
    assert priced.analysis_routes()["expected_cost"].route == (
        routes.NUMERICAL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tested = RepairableRBD(
            [("s", "c"), ("c", "t")],
            {
                "c": {
                    "reliability": W([500.0, 1.5]),
                    "repairability": "instant",
                    "inspection": {"interval": 100.0},
                }
            },
        )
        # Tests that take time are simulated.
        slow = RepairableRBD(
            [("s", "c"), ("c", "t")],
            {
                "c": {
                    "reliability": W([500.0, 1.5]),
                    "repairability": "instant",
                    "inspection": {"interval": 100.0, "duration": E([2.0])},
                }
            },
        )
    assert tested.analysis_routes()["expected_failures"].route == (
        routes.NUMERICAL
    )
    route = slow.analysis_routes()["expected_failures"]
    assert route.route == routes.REFUSED
    with pytest.raises(NotImplementedError) as error:
        slow.expected_failures(100.0)
    assert str(error.value) == route.reason
