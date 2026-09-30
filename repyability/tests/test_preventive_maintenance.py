"""Scheduled preventive maintenance in RepairableRBD (issue #69).

The event logic is checked on deterministic lifetimes, where every failure
and replacement can be listed by hand; the random cases against the exact
renewal-reward values, and against the closed forms of the issue.
"""

import dataclasses
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.special import gamma as gamma_function
from scipy.special import gammainc

from repyability import NonRepairable, RepairableRBD
from repyability.rbd.repairable_rbd import Event

X = surv.ExactEventTime.from_params
E = surv.Exponential.from_params
W = surv.Weibull.from_params
L = surv.LogNormal.from_params


def single(spec, **kwargs) -> RepairableRBD:
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec}, **kwargs)


def per_run(result, value):
    return value / result.n_simulations


# -- the schedule, on deterministic lifetimes --------------------------------


def test_age_replacement_before_the_failure_prevents_every_failure():
    # Fails 10 after each renewal; replaced at age 6: at 6, 12, ..., 54.
    rbd = single(
        {
            "reliability": X(10),
            "repairability": "instant",
            "replace_cost": 5.0,
            "preventive": {"interval": 6, "cost": 1.0},
        }
    )
    result = rbd.availability(60.0, mc_samples=3, seed=1)
    assert result.cost.by_category["preventive"] == 9.0
    assert result.cost.by_category["replace"] == 0.0
    assert result.system_failures == 0
    assert result.system_planned_outages == 0  # maintained in zero time
    assert result.system_uptime == 3 * 60.0


def test_a_failure_at_the_replacement_age_is_a_failure():
    # Age 10 is reached at the failure: the failure comes first, and the
    # repair starts a new unit's clock, so no replacement is ever due.
    rbd = single(
        {
            "reliability": X(10),
            "repairability": "instant",
            "replace_cost": 5.0,
            "preventive": {"interval": 10, "cost": 1.0},
        }
    )
    cost = rbd.cost(60.0, mc_samples=2, seed=1)
    assert cost.by_category["replace"] == 5 * 5.0  # at 10, 20, 30, 40, 50
    assert cost.by_category["preventive"] == 0.0


def test_block_replacement_falls_on_the_calendar():
    # Fails 10 after each renewal; block replacement every 25 whatever the
    # age: failures at 10, 20, 35, 45, 60, 70, 85, 95 and replacements at
    # 25, 50, 75 (the one at 100 is outside the window).
    rbd = single(
        {
            "reliability": X(10),
            "repairability": "instant",
            "replace_cost": 5.0,
            "preventive": {"interval": 25, "policy": "block", "cost": 1.0},
        }
    )
    result = rbd.availability(100.0, mc_samples=3, seed=1)
    assert result.cost.by_category["replace"] == 8 * 5.0
    assert result.cost.by_category["preventive"] == 3 * 1.0
    # Instant repairs are zero-length failures of the system.
    assert per_run(result, result.system_failures) == 8


def test_block_replacement_is_skipped_while_the_unit_is_down():
    # Fails at 10 and is under repair until 30, past the block time 25,
    # which is skipped; each new unit fails 10 after it is put into service,
    # before the next block time, so none is ever replaced.
    rbd = single(
        {
            "reliability": X(10),
            "repairability": X(20),
            "preventive": {"interval": 25, "policy": "block", "cost": 1.0},
        }
    )
    result = rbd.availability(100.0, mc_samples=1, seed=1)
    # 0-10 up, 10-30 down, 30-40 up, 40-60 down, 60-70 up, 70-90 down,
    # 90-100 up: the new units reach no block time alive.
    assert result.cost.by_category["preventive"] == 0.0
    assert result.system_uptime == 40.0
    rbd = single(
        {
            "reliability": X(30),
            "repairability": X(20),
            "preventive": {"interval": 25, "policy": "block", "cost": 1.0},
        }
    )
    result = rbd.availability(100.0, mc_samples=1, seed=1)
    # Replaced (in zero time) at 25, 50 and 75, before it can fail.
    assert result.cost.by_category["preventive"] == 3.0
    assert result.system_uptime == 100.0


def test_maintenance_that_takes_time_is_a_planned_outage():
    # Fails 100 after each renewal; replaced at age 40, which takes 2: up
    # 40, down 2, from 0. Over 420 the replacements start at 40, 82, ...,
    # 418 (ten of them), and the last one ends at the end of the window.
    rbd = single(
        {
            "reliability": X(100),
            "repairability": X(1),
            "downtime_cost": 3.0,
            "preventive": {"interval": 40, "duration": X(2), "cost": 5.0},
        },
        downtime_cost_rate=7.0,
    )
    result = rbd.availability(420.0, mc_samples=2, seed=1)
    assert per_run(result, result.system_uptime) == 400.0
    assert per_run(result, result.system_downtime) == 20.0
    assert per_run(result, result.node_downtime["c"]) == 20.0
    assert per_run(result, result.system_planned_outages) == 10
    assert per_run(result, result.system_restorations) == 9
    assert result.system_failures == 0
    assert result.mean_up_time == 40.0
    assert result.cost.by_category == {
        "repair": 0.0,
        "replace": 0.0,
        "preventive": 50.0,
        "inspection": 0.0,
        "component_downtime": 60.0,
        "system_downtime": 140.0,
    }
    assert result.cost.by_component == {"c": 110.0}
    fci = result.criticalities.failure_criticality_index
    # Planned outages are not failures.
    assert fci.per_system_failure.get("c", 0) == 0
    rci = result.criticalities.restoration_criticality_index
    assert rci.by_system["c"] == 1.0  # c's returns restore the system
    # The exact long-run values: a 42-long cycle, 40 of it up.
    assert rbd.mean_availability() == pytest.approx(40 / 42)
    assert rbd.system_failure_frequency() == 0.0
    assert rbd.mean_up_time() == pytest.approx(40.0)
    assert rbd.mean_down_time() == pytest.approx(2.0)
    assert rbd.mean_time_between_failures() == math.inf
    assert rbd.expected_cost_rate() == pytest.approx(250 / 420)


def test_a_planned_outage_of_a_redundant_unit_does_not_stop_the_system():
    pumps = {
        "a": {
            "reliability": X(100),
            "repairability": X(1),
            "preventive": {"interval": 40, "duration": X(2)},
        },
        "b": {"reliability": X(1000), "repairability": X(1)},
    }
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")], pumps
    )
    result = rbd.availability(420.0, mc_samples=1, seed=1)
    assert result.system_uptime == 420.0
    assert result.system_planned_outages == 0
    assert result.node_downtime["a"] == 20.0


# -- the stepping API and nested RBDs ----------------------------------------


def maintained_unit():
    return single(
        {
            "reliability": X(100),
            "repairability": X(1),
            "preventive": {"interval": 40, "duration": X(2)},
        }
    )


def test_stepping_reports_planned_outages():
    rbd = maintained_unit()
    rbd.initialize_event_queue(100.0)
    steps = []
    for _ in range(4):
        change = rbd.next_event()
        steps.append((*change, rbd.last_change_planned))
    assert steps == [
        (40.0, False, True),
        (42.0, True, False),
        (82.0, False, True),
        (84.0, True, False),
    ]


def test_a_nested_rbds_planned_outage_is_planned_for_its_parent():
    parent = RepairableRBD(
        [("s", "sub"), ("sub", "v"), ("v", "t")],
        {
            "sub": maintained_unit(),
            "v": {"reliability": X(1000), "repairability": X(1)},
        },
    )
    result = parent.availability(420.0, mc_samples=2, seed=1)
    assert per_run(result, result.system_planned_outages) == 10
    assert result.system_failures == 0
    # (In the long run v fails too; hold it working to see sub alone.)
    held = {"working_nodes": ["v"]}
    assert parent.mean_availability(**held) == pytest.approx(40 / 42)
    assert parent.system_failure_frequency(**held) == 0.0
    assert parent.mean_up_time(**held) == pytest.approx(40.0)
    assert parent.mean_down_time(**held) == pytest.approx(2.0)


# -- the closed forms of the issue -------------------------------------------


def weibull_integral(alpha, beta, t):
    """integral_0^t exp(-(u / alpha) ** beta) du."""
    shape = 1.0 / beta
    return (
        alpha
        * gamma_function(1 + shape)
        * gammainc(shape, (t / alpha) ** beta)
    )


def test_age_replacement_rate_is_the_non_repairable_rate():
    # Instant repair and maintenance: the renewal-reward rate
    # (c_p R(T) + c_u F(T)) / integral_0^T R, as NonRepairable computes.
    alpha, beta, T, cp, cu = 100.0, 3.0, 50.0, 1.0, 5.0
    rbd = single(
        {
            "reliability": W([alpha, beta]),
            "repairability": "instant",
            "replace_cost": cu,
            "preventive": {"interval": T, "cost": cp},
        }
    )
    survives = math.exp(-((T / alpha) ** beta))
    exact = (cp * survives + cu * (1 - survives)) / weibull_integral(
        alpha, beta, T
    )
    assert rbd.expected_cost_rate() == pytest.approx(exact, rel=1e-9)
    # Maintained in zero time: never down, so no planned outages; the
    # instant repairs are zero-length failures.
    assert rbd.mean_availability() == 1.0
    assert rbd._outage_frequencies()[1] == 0.0
    assert rbd.mean_up_time() == pytest.approx(
        1 / rbd.system_failure_frequency()
    )
    unit = NonRepairable(W([alpha, beta]))
    unit.set_costs_planned_and_unplanned(cp, cu)
    assert rbd.expected_cost_rate() == pytest.approx(
        float(unit.cost_rate(T)), rel=1e-12
    )
    # ...and the simulation converges to it.
    result = rbd.cost(t_simulation=5000.0, mc_samples=200, seed=2)
    interval = result.mean_interval(0.999)
    assert interval.lower <= exact * 5000.0 <= interval.upper


def test_block_replacement_cannot_help_an_exponential_unit():
    # Memoryless: failures stay a Poisson process at rate lam whatever the
    # replacements, so the rate is exactly lam c_u + c_p / T. Over m blocks
    # the m - 1 replacements inside the window are certain.
    lam, T, cp, cu, m = 0.1, 20.0, 3.0, 10.0, 50
    rbd = single(
        {
            "reliability": E([lam]),
            "repairability": "instant",
            "replace_cost": cu,
            "preventive": {"interval": T, "policy": "block", "cost": cp},
        }
    )
    t = m * T
    result = rbd.cost(t_simulation=t, mc_samples=400, seed=3)
    assert result.by_category["preventive"] == cp * (m - 1)
    failures = result.samples - cp * (m - 1)
    mean, se = failures.mean(), failures.std(ddof=1) / math.sqrt(400)
    assert abs(mean - cu * lam * t) < 4 * se
    assert result.cost_rate == pytest.approx(lam * cu + cp / T, rel=0.02)


# -- the exact long-run values against the simulation ------------------------


def maintained_system():
    """Two maintained pumps in parallel, in series with a maintained valve:
    maintenance takes time, and the repairs too."""
    pump = {
        "reliability": W([100.0, 3.0]),
        "repairability": L([1.0, 0.5]),
        "replace_cost": 40.0,
        "downtime_cost": 2.0,
        "preventive": {
            "interval": 60.0,
            "duration": W([3.0, 2.0]),
            "cost": 10.0,
        },
    }
    valve = {
        "reliability": W([300.0, 2.0]),
        "repairability": E([0.5]),
        "repair_cost": 25.0,
        "preventive": {
            "interval": 150.0,
            "duration": X(1.5),
            "cost": L([2.0, 0.3]),
        },
    }
    return RepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")],
        {"p1": pump, "p2": pump, "v": valve},
        downtime_cost_rate=100.0,
    )


def test_the_renewal_cycle_gives_each_nodes_availability():
    rbd = maintained_system()
    # A pump: up integral_0^60 R, then repaired (mean exp(1.125)) after a
    # failure or maintained (mean 3 Gamma(1.5)) after surviving to 60.
    up = weibull_integral(100.0, 3.0, 60.0)
    survives = math.exp(-(0.6**3))
    cycle = (
        up
        + (1 - survives) * math.exp(1.0 + 0.5**2 / 2)
        + survives * 3.0 * gamma_function(1.5)
    )
    availability = rbd.node_availability()
    assert availability["p1"] == pytest.approx(up / cycle, rel=1e-9)
    assert availability["p2"] == availability["p1"]
    valve_up = weibull_integral(300.0, 2.0, 150.0)
    valve_survives = math.exp(-(0.5**2))
    valve_cycle = valve_up + (1 - valve_survives) * 2.0 + valve_survives * 1.5
    assert availability["v"] == pytest.approx(valve_up / valve_cycle, rel=1e-9)
    pumps_down = (1 - availability["p1"]) ** 2
    assert rbd.mean_availability() == pytest.approx(
        (1 - pumps_down) * availability["v"], rel=1e-12
    )


def test_the_exact_long_run_values_match_a_long_simulation():
    rbd = maintained_system()
    t, n = 20_000.0, 40
    result = rbd.availability(t, mc_samples=n, seed=5)
    window = n * t
    assert result.system_uptime / window == pytest.approx(
        rbd.mean_availability(), abs=0.002
    )
    failures, planned = rbd._outage_frequencies()
    assert failures == rbd.system_failure_frequency()
    assert result.system_failures / window == pytest.approx(failures, rel=0.1)
    assert result.system_planned_outages / window == pytest.approx(
        planned, rel=0.05
    )
    assert result.mean_up_time == pytest.approx(rbd.mean_up_time(), rel=0.05)
    assert result.mean_down_time == pytest.approx(
        rbd.mean_down_time(), rel=0.05
    )
    assert result.cost.cost_rate == pytest.approx(
        rbd.expected_cost_rate(), rel=0.03
    )


def test_block_replacement_has_exact_long_run_values():
    # The details are in test_block_replacement.py; here, the long-run
    # values agree with the simulation's long-window averages.
    rbd = single(
        {
            "reliability": W([100.0, 3.0]),
            "repairability": E([1.0]),
            "replace_cost": 5.0,
            "preventive": {"interval": 50.0, "policy": "block"},
        }
    )
    t = 50_000.0
    result = rbd.availability(t_simulation=t, mc_samples=20, seed=2)
    window = result.mean_availability_interval()
    assert abs(rbd.mean_availability() - window.estimate) < 4 * (
        window.standard_error
    )
    cost = rbd.cost(t_simulation=t, mc_samples=20, seed=3).mean_interval()
    assert abs(rbd.expected_cost_rate() * t - cost.estimate) < 4 * (
        cost.standard_error
    )
    assert rbd.mean_up_time() > 0.0
    assert set(rbd.birnbaum_importance()) >= {"c"}


# -- no maintenance ---------------------------------------------------------


def assert_identical(a, b):
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if dataclasses.is_dataclass(x):
            assert_identical(x, y)
        elif isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert x == y, field.name


@pytest.mark.parametrize("policy", ["age", "block"])
def test_an_infinite_interval_is_no_maintenance(policy):
    def build(preventive):
        spec = {
            "reliability": W([100.0, 3.0]),
            "repairability": L([1.0, 0.5]),
            "replace_cost": L([3.0, 0.4]),
            "downtime_cost": 2.0,
        }
        if preventive:
            spec["preventive"] = {
                "interval": math.inf,
                "policy": policy,
                "duration": W([3.0, 2.0]),
                "cost": 10.0,
            }
        return RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
            {"a": spec, "b": dict(spec)},
            downtime_cost_rate=5.0,
        )

    plain, maintained = build(False), build(True)
    assert_identical(
        plain.availability(800.0, mc_samples=30, seed=7),
        maintained.availability(800.0, mc_samples=30, seed=7),
    )
    assert plain.expected_cost_rate() == maintained.expected_cost_rate()
    assert plain.mean_up_time() == maintained.mean_up_time()


def test_forced_nodes_are_not_maintained():
    rbd = single(
        {
            "reliability": W([100.0, 3.0]),
            "repairability": "instant",
            "preventive": {"interval": 10.0, "cost": 1.0},
        }
    )
    assert (
        rbd.cost(100.0, mc_samples=3, seed=1, working_nodes=["c"]).mean == 0.0
    )
    assert rbd.expected_cost_rate(working_nodes=["c"]) == 0.0
    assert rbd.expected_cost_rate() > 0.0


# -- the spec ---------------------------------------------------------------


@pytest.mark.parametrize(
    "preventive, match",
    [
        ("often", "must be a dict"),
        ({"interval": 10, "every": 2}, "unknown preventive key"),
        ({"policy": "age"}, "needs an interval"),
        ({"interval": 0}, "positive number"),
        ({"interval": -5}, "positive number"),
        ({"interval": float("nan")}, "positive number"),
        ({"interval": "ten"}, "positive number"),
        ({"interval": 10, "policy": "calendar"}, "'age' or 'block'"),
        ({"interval": 10, "duration": "quick"}, "time-to-maintain model"),
        ({"interval": 10, "duration": 2.0}, "time-to-maintain model"),
        ({"interval": 10, "cost": -1.0}, "non-negative"),
        (
            {"interval": 10, "cost": surv.Normal.from_params([1, 5])},
            "negative",
        ),
    ],
)
def test_invalid_preventive_specs_are_rejected(preventive, match):
    with pytest.raises(ValueError, match=match):
        single(
            {
                "reliability": W([100.0, 3.0]),
                "repairability": "instant",
                "preventive": preventive,
            }
        )


def test_a_zero_cost_prices_nothing():
    spec = {
        "reliability": W([100.0, 3.0]),
        "repairability": "instant",
        "preventive": {"interval": 10.0, "cost": 0.0},
    }
    rbd = single(spec)
    assert not rbd.has_costs
    assert rbd.cost(100.0, mc_samples=2, seed=1) is None
    spec["preventive"]["cost"] = 1.0
    assert single(spec).has_costs


def test_a_maintained_rbd_round_trips_through_json():
    rbd = maintained_system()
    again = RepairableRBD.from_json(rbd.to_json())
    assert_identical(
        rbd.availability(500.0, mc_samples=10, seed=8),
        again.availability(500.0, mc_samples=10, seed=8),
    )
    assert again.expected_cost_rate() == rbd.expected_cost_rate()


def test_events_still_compare_by_time_alone():
    assert Event(1.0, "a", False, True) == Event(1.0, "b", True)
    assert Event(1.0, "a", True).preventive is False
