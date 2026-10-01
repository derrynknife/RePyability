"""Hidden failures and periodic inspection in RepairableRBD (issue #70).

The event logic is checked on deterministic lifetimes, where every failure,
test and repair can be listed by hand; the exact long-run values against the
closed forms of the issue (and quadrature of the point availabilities); and
the simulation against the exact values.
"""

import json

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import RBD, RepairableRBD
from repyability.rbd.repairable_rbd import Event

X = surv.ExactEventTime.from_params
E = surv.Exponential.from_params
W = surv.Weibull.from_params

PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def single(spec, **kwargs) -> RepairableRBD:
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec}, **kwargs)


def hidden(rate, interval, **extra) -> dict:
    """A component with a constant failure rate and hidden failures,
    tested and repaired in zero time."""
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval},
        **extra,
    }


def per_run(result, value):
    return value / result.n_simulations


# -- the events, on deterministic lifetimes ----------------------------------


def test_a_hidden_failure_is_down_until_the_next_inspection():
    # Fails 12 after each renewal, inspected every 5: fails at 12, found and
    # renewed at 15; fails at 27, found at 30; 42 and 45; 57, found at 60,
    # the end of the window.
    rbd = single(
        {
            "reliability": X(12),
            "repairability": "instant",
            "repair_cost": 100.0,
            "inspection": {"interval": 5, "cost": 1.0},
        }
    )
    result = rbd.availability(60.0, mc_samples=2, seed=1)
    assert per_run(result, result.system_downtime) == 12.0
    assert per_run(result, result.system_failures) == 4
    assert per_run(result, result.system_restorations) == 3
    assert result.system_planned_outages == 0
    # Eleven inspections (5, 10, ..., 55), and three repairs: each charged
    # when a test finds the failure, so the last is not (found at 60).
    assert result.cost.by_category == {
        "repair": 300.0,
        "replace": 0.0,
        "preventive": 0.0,
        "inspection": 11.0,
        "component_downtime": 0.0,
        "system_downtime": 0.0,
        "setup": 0.0,
    }


def test_a_test_takes_the_unit_off_line_and_it_does_not_age():
    # Fails after 12 of running, inspected every 5, each test taking 1 and
    # each repair 2. From 0: tested at 5 and 10 (off-line to 6 and 11, so
    # the failure moves to 14); fails at 14, found at 15, tested to 16,
    # repaired to 18. Renewed at 18 (failure due 30): tested at 20, 25 and
    # 30 (failure now 33); fails at 33, found at 35, back at 38. Renewed
    # (due 50): tested at 40, 45, 50 (due 53); fails at 53, back at 58.
    rbd = single(
        {
            "reliability": X(12),
            "repairability": X(2),
            "inspection": {"interval": 5, "duration": X(1), "cost": 1.0},
            "downtime_cost": 10.0,
        }
    )
    result = rbd.availability(60.0, mc_samples=2, seed=1)
    # Down 8 times for 1 (the tests), and 4 + 5 + 5 for the failures.
    assert per_run(result, result.system_downtime) == 22.0
    assert per_run(result, result.system_planned_outages) == 8
    assert per_run(result, result.system_failures) == 3
    assert per_run(result, result.system_restorations) == 11
    assert result.cost.by_category["inspection"] == 11.0  # per run
    assert result.cost.by_category["component_downtime"] == 220.0
    fci = result.criticalities.failure_criticality_index
    assert fci.per_system_failure == {"c": 1.0}  # tests are not failures


def test_an_inspection_during_a_repair_is_skipped():
    # Fails 3 after each renewal, inspected every 5, repaired in 12: fails
    # at 3, found at 5, back at 17 (the tests at 10 and 15 are skipped);
    # fails at 20, found at 20 (a failure at an inspection is found by it),
    # back at 32; fails at 35, found at 35, back at 47; fails at 50, found
    # at 50.
    rbd = single(
        {
            "reliability": X(3),
            "repairability": X(12),
            "inspection": {"interval": 5, "cost": 1.0},
        }
    )
    result = rbd.availability(55.0, mc_samples=1, seed=1)
    assert result.system_failures == 4
    assert result.cost.by_category["inspection"] == 4.0  # at 5, 20, 35, 50
    # Up 0-3, 17-20, 32-35 and 47-50.
    assert result.system_uptime == 12.0


@pytest.mark.parametrize("interval", [0.7, 0.1, 0.3])
def test_every_inspection_happens_once(interval):
    # Intervals whose multiples round (3 * 0.7 = 2.0999999999999996):
    # nine inspections in ten intervals, the tenth at the end of the window.
    rbd = single(
        {
            "reliability": X(1e9),
            "repairability": "instant",
            "inspection": {"interval": interval, "cost": 1.0},
        }
    )
    result = rbd.availability(10 * interval, mc_samples=1, seed=1)
    assert result.cost.by_category["inspection"] == 9.0


def test_no_inspection_means_failures_are_revealed():
    # The same component without an inspection is repaired at once, at 12,
    # 24, 36 and 48.
    spec = {"reliability": X(12), "repairability": "instant"}
    result = single(spec).availability(60.0, mc_samples=1, seed=1)
    assert result.system_downtime == 0.0
    assert result.system_failures == 4


def test_the_event_api_marks_a_test_outage_as_planned():
    rbd = single(
        {
            "reliability": X(12),
            "repairability": X(2),
            "inspection": {"interval": 5, "duration": X(1)},
        }
    )
    rbd.initialize_event_queue(20.0)
    changes = []
    while True:
        t, state = rbd.next_event()
        if t >= 20.0:
            break
        changes.append((t, state, rbd.last_change_planned))
    assert changes == [
        (5.0, False, True),
        (6.0, True, False),
        (10.0, False, True),
        (11.0, True, False),
        (14.0, False, False),  # the hidden failure
        (18.0, True, False),
    ]


def test_an_event_can_be_an_inspection():
    event = Event(5.0, "c", False, inspection=True)
    assert event.inspection and not event.preventive
    assert Event(4.0, "c", True) < event


# -- exact long-run values ----------------------------------------------------


def test_one_component_is_down_half_an_interval_on_average():
    rate, interval = 0.01, 50.0
    rbd = single(hidden(rate, interval))
    exact = 1 - (1 - np.exp(-rate * interval)) / (rate * interval)
    assert rbd.mean_unavailability() == pytest.approx(exact, rel=1e-12)
    assert 1 - rbd.node_availability()["c"] == pytest.approx(exact, rel=1e-12)
    assert exact == pytest.approx(rate * interval / 2, rel=0.2)
    # At most one failure per interval.
    assert rbd.system_failure_frequency() == pytest.approx(
        (1 - np.exp(-rate * interval)) / interval, rel=1e-12
    )


def test_two_components_tested_together_are_down_together():
    # 1oo2: the average probability of failure on demand of the issue.
    rate, interval = 0.01, 50.0
    rbd = RepairableRBD(
        PAIR, {"a": hidden(rate, interval), "b": hidden(rate, interval)}
    )
    exact = (
        quad(lambda t: (1 - np.exp(-rate * t)) ** 2, 0, interval)[0] / interval
    )
    assert rbd.mean_unavailability() == pytest.approx(exact, rel=1e-12)
    # Not the product of the average unavailabilities: they are down
    # together more often than independent units would be.
    one = 1 - rbd.node_availability()["a"]
    assert rbd.mean_unavailability() > 1.25 * one**2
    # ...and about (lambda * tau) ** 2 / 3 when that is small.
    small = RepairableRBD(PAIR, {"a": hidden(1e-5, 50), "b": hidden(1e-5, 50)})
    assert small.mean_unavailability() == pytest.approx(
        (1e-5 * 50) ** 2 / 3, rel=1e-3
    )
    # The pair fails when the second unit does: once an interval at most.
    assert rbd.system_failure_frequency() == pytest.approx(
        (1 - np.exp(-rate * interval)) ** 2 / interval, rel=1e-12
    )
    omega = rbd.system_failure_frequency()
    assert rbd.mean_up_time() == pytest.approx(
        rbd.mean_availability() / omega, rel=1e-12
    )
    assert rbd.mean_down_time() == pytest.approx(
        rbd.mean_unavailability() / omega, rel=1e-12
    )


def test_two_out_of_three_tested_together():
    rate, interval = 0.004, 100.0
    rbd = RepairableRBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("s", "c"),
            ("a", "v"),
            ("b", "v"),
            ("c", "v"),
            ("v", "t"),
        ],
        {
            **{n: hidden(rate, interval) for n in "abc"},
            "v": {"reliability": E([1e-9]), "repairability": "instant"},
        },
        k={"v": 2},
    )

    def down(t):
        q = 1 - np.exp(-rate * t)
        return 3 * q**2 * (1 - q) + q**3

    exact = quad(down, 0, interval)[0] / interval
    assert rbd.mean_unavailability() == pytest.approx(exact, rel=1e-12)


def test_different_intervals_are_averaged_over_their_common_period():
    # "a" every 20 and "b" every 30, in parallel: they repeat together
    # every 60.
    rate = 0.01
    rbd = RepairableRBD(PAIR, {"a": hidden(rate, 20), "b": hidden(rate, 30)})

    def down(t):
        a = 1 - np.exp(-rate * (t % 20))
        return a * (1 - np.exp(-rate * (t % 30)))

    exact = quad(down, 0, 60, points=[20, 30, 40])[0] / 60
    assert rbd.mean_unavailability() == pytest.approx(exact, rel=1e-12)


def test_revealed_components_mix_with_hidden_ones():
    # A hidden-failure unit in series with an ordinary repairable one.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": hidden(0.01, 50),
            "b": {"reliability": E([0.1]), "repairability": E([1.0])},
        },
    )
    a = (1 - np.exp(-0.5)) / 0.5
    assert rbd.mean_availability() == pytest.approx(a * 10 / 11, rel=1e-12)


def test_importance_is_averaged_over_the_interval():
    rate, interval = 0.01, 50.0
    rbd = RepairableRBD(
        PAIR, {"a": hidden(rate, interval), "b": hidden(rate, interval)}
    )
    one = 1 - rbd.node_availability()["a"]
    both = rbd.mean_unavailability()
    # "a" is critical while "b" is down.
    assert rbd.birnbaum_importance()["a"] == pytest.approx(one, rel=1e-12)
    assert rbd.improvement_potential()["a"] == pytest.approx(both, rel=1e-12)
    # With "a" failed, the system is down while "b" is: a ratio of the
    # time-averaged unavailabilities.
    assert rbd.risk_achievement_worth()["a"] == pytest.approx(
        one / both, rel=1e-12
    )
    with np.errstate(divide="ignore"):
        assert rbd.risk_reduction_worth()["a"] == float("inf")
    # Every system failure has both failed and both critical.
    assert rbd.criticality_importance()["a"] == pytest.approx(1.0)
    assert rbd.fussell_vesely()["a"] == pytest.approx(1.0)
    # Held working, "a" makes the system perfect.
    assert rbd.mean_availability(working_nodes=["a"]) == 1.0


def test_the_cost_rate_trades_tests_against_hidden_downtime():
    rate, test_cost, downtime_cost = 0.01, 100.0, 1000.0

    def cost_rate(interval):
        rbd = single(
            {
                "reliability": E([rate]),
                "repairability": "instant",
                "repair_cost": 40.0,
                "downtime_cost": downtime_cost,
                "inspection": {"interval": interval, "cost": test_cost},
            }
        )
        return rbd.expected_cost_rate()

    interval = 20.0
    unavailable = 1 - (1 - np.exp(-rate * interval)) / (rate * interval)
    failures = (1 - np.exp(-rate * interval)) / interval
    assert cost_rate(interval) == pytest.approx(
        test_cost / interval + downtime_cost * unavailable + 40.0 * failures,
        rel=1e-12,
    )
    # Least near sqrt(2 * c_i / (lambda * c_d)).
    grid = np.linspace(2.0, 10.0, 801)
    best = grid[np.argmin([cost_rate(t) for t in grid])]
    assert best == pytest.approx(
        np.sqrt(2 * test_cost / (rate * downtime_cost)), rel=0.05
    )


def test_a_forced_node_is_not_inspected():
    rbd = single(
        hidden(0.01, 50, repair_cost=10.0)
        | {"inspection": {"interval": 50, "cost": 5.0}}
    )
    assert rbd.expected_cost_rate(working_nodes=["c"]) == 0.0
    result = rbd.availability(200.0, mc_samples=2, seed=1, working_nodes=["c"])
    assert result.cost.by_category["inspection"] == 0.0


def test_exact_values_need_a_constant_rate_and_instant_tests_and_repair():
    for spec in (
        {"reliability": W([100, 2]), "repairability": "instant"},
        {"reliability": E([0.01]), "repairability": E([1.0])},
    ):
        rbd = single({**spec, "inspection": {"interval": 50}})
        with pytest.raises(NotImplementedError, match="hidden failures"):
            rbd.mean_availability()
    timed = single(
        {
            "reliability": E([0.01]),
            "repairability": "instant",
            "inspection": {"interval": 50, "duration": X(1)},
        }
    )
    for method in (
        timed.mean_availability,
        timed.node_availability,
        timed.system_failure_frequency,
        timed.birnbaum_importance,
    ):
        with pytest.raises(NotImplementedError, match="simulation"):
            method()
    # A Weibull of shape 1 has a constant rate.
    shape_one = single(
        {
            "reliability": W([100, 1]),
            "repairability": "instant",
            "inspection": {"interval": 50},
        }
    )
    assert shape_one.mean_unavailability() == pytest.approx(
        single(hidden(0.01, 50)).mean_unavailability(), rel=1e-9
    )


def test_intervals_with_no_common_period_are_simulated():
    rbd = RepairableRBD(
        PAIR, {"a": hidden(0.01, 1.0), "b": hidden(0.01, np.sqrt(2))}
    )
    with pytest.raises(NotImplementedError, match="common period"):
        rbd.mean_availability()
    rbd = RepairableRBD(
        PAIR, {"a": hidden(0.01, 1.0), "b": hidden(0.01, 1.000001)}
    )
    with pytest.raises(NotImplementedError, match="too many"):
        rbd.mean_availability()


def test_a_nested_rbd_with_hidden_failures():
    pair = RepairableRBD(PAIR, {"a": hidden(0.01, 50), "b": hidden(0.01, 50)})
    other = {"reliability": E([0.1]), "repairability": E([1.0])}
    parent = RepairableRBD(
        [("s", "pair"), ("pair", "x"), ("x", "t")],
        {"pair": pair, "x": other},
    )
    # The only node that varies with the inspections: exact, from its own
    # long-run availability.
    assert parent.mean_availability() == pytest.approx(
        pair.mean_availability() * 10 / 11, rel=1e-12
    )
    # With another inspected node alongside, it is not.
    again = RepairableRBD(PAIR, {"a": hidden(0.01, 50), "b": hidden(0.01, 50)})
    mixed = RepairableRBD(
        [("s", "pair"), ("pair", "y"), ("y", "t")],
        {"pair": again, "y": hidden(0.01, 50)},
    )
    with pytest.raises(NotImplementedError, match="simulation"):
        mixed.mean_availability()


# -- the simulation against the exact values ---------------------------------


def test_the_simulation_matches_the_exact_values():
    rate, interval = 0.01, 50.0
    rbd = RepairableRBD(
        PAIR,
        {
            "a": hidden(rate, interval, repair_cost=10.0),
            "b": hidden(rate, interval, repair_cost=10.0),
        },
        downtime_cost_rate=1000.0,
    )
    result = rbd.availability(t_simulation=500.0, mc_samples=4000, seed=7)
    window = result.n_simulations * result.time_simulated_to
    # The window is ten whole intervals, and each interval starts afresh.
    simulated = 1 - float(result.system_uptime) / window
    assert simulated == pytest.approx(rbd.mean_unavailability(), rel=0.03)
    assert float(result.system_failures) / window == pytest.approx(
        rbd.system_failure_frequency(), rel=0.03
    )
    assert result.cost.cost_rate == pytest.approx(
        rbd.expected_cost_rate(), rel=0.03
    )


# -- the spec -----------------------------------------------------------------


@pytest.mark.parametrize(
    "inspection, message",
    [
        (50, "must be a dict"),
        ({"interval": 50, "every": 2}, "unknown inspection key"),
        ({"duration": X(1)}, "needs an interval"),
        ({"interval": 0}, "positive, finite"),
        ({"interval": -5}, "positive, finite"),
        ({"interval": float("inf")}, "positive, finite"),
        ({"interval": "yearly"}, "positive, finite"),
        ({"interval": 50, "duration": 3.0}, "time-to-test model"),
        ({"interval": 50, "cost": -1.0}, "non-negative"),
    ],
)
def test_a_bad_inspection_spec_is_rejected(inspection, message):
    with pytest.raises(ValueError, match=message):
        single({**hidden(0.01, 50), "inspection": inspection})


def test_preventive_maintenance_and_inspection_do_not_mix():
    with pytest.raises(ValueError, match="both"):
        single(
            {
                **hidden(0.01, 50),
                "preventive": {"interval": 100},
            }
        )
    # A preventive interval of inf is no maintenance at all.
    single({**hidden(0.01, 50), "preventive": {"interval": float("inf")}})


def test_an_inspected_rbd_round_trips():
    rbd = RepairableRBD(
        PAIR,
        {
            "a": {
                "reliability": E([0.01]),
                "repairability": X(2),
                "inspection": {
                    "interval": 50,
                    "duration": X(1),
                    "cost": surv.LogNormal.from_params([2.0, 0.3]),
                },
            },
            "b": hidden(0.01, 25),
        },
    )
    for back in (RBD.from_dict(rbd.to_dict()), RBD.from_json(rbd.to_json())):
        assert {n: i.interval for n, i in back._inspection.items()} == {
            "a": 50.0,
            "b": 25.0,
        }
        assert json.dumps(back.to_dict()) == json.dumps(rbd.to_dict())
        first = back.availability(300.0, mc_samples=20, seed=3)
        second = rbd.availability(300.0, mc_samples=20, seed=3)
        assert first.system_uptime == second.system_uptime
        assert first.cost.samples.tolist() == second.cost.samples.tolist()
