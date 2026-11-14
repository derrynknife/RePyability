"""The work and cost in each of a run of windows from a state
(``expected_events_by_window``, ``cost_by_window``): checked against the
exact counts over the whole span (windows add), hand-worked schedules,
a closed form, and the simulation."""

import numpy as np
import pytest
import surpyval as surv

from repyability import CostByWindow, EventsByWindow, NodeState, RepairableRBD

W = surv.Weibull.from_params
E = surv.Exponential.from_params
FIXED = surv.ExactEventTime.from_params


def plant():
    return RepairableRBD(
        [
            ("s", "pump"),
            ("pump", "valve"),
            ("valve", "sensor"),
            ("sensor", "t"),
        ],
        {
            "pump": {
                "reliability": W([9000.0, 2.5]),
                "repairability": surv.LogNormal.from_params(
                    [np.log(8.0), 0.5]
                ),
                "preventive": {"interval": 6000.0, "duration": FIXED([4.0])},
            },
            "valve": {
                "reliability": W([20000.0, 1.5]),
                "repairability": E([1 / 12.0]),
                "preventive": {
                    "interval": 8760.0,
                    "policy": "block",
                    "duration": FIXED([6.0]),
                },
            },
            "sensor": {
                "reliability": E([1 / 50000.0]),
                "repairability": "instant",
                "inspection": {"interval": 4380.0},
            },
        },
    )


STATE = {
    "pump": NodeState(age=5200.0),
    "valve": NodeState(age=7000.0, since_scheduled=7000.0),
    "sensor": NodeState(age=3000.0, since_scheduled=3000.0),
}


def bearing():
    return RepairableRBD(
        [("s", "b"), ("b", "t")],
        {
            "b": {
                "reliability": W([365.0, 2.0]),
                "repairability": "instant",
                "replace_cost": 400.0,
                "preventive": {
                    "interval": 180.0,
                    "policy": "block",
                    "cost": 100.0,
                },
            }
        },
    )


BEARING_STATE = {"b": NodeState(age=224.0, since_scheduled=159.0)}


def test_the_windows_add_up_to_the_whole_span():
    rbd = plant()
    windows = rbd.expected_events_by_window(width=730.0, count=12, state=STATE)
    whole = rbd.expected_events(730.0 * 12, state=STATE)
    assert isinstance(windows, EventsByWindow)
    for table, total in (
        (windows.node_preventive, whole.node_preventive),
        (windows.node_failures, whole.node_failures),
        (windows.node_inspections, whole.node_inspections),
    ):
        for node in total:
            assert table[node].sum() == pytest.approx(total[node], abs=1e-9)
    assert windows.system_planned_outages.sum() == pytest.approx(
        whole.system_planned_outages, abs=1e-9
    )
    assert min(v.min() for v in windows.node_failures.values()) >= 0.0


def test_the_planners_year_matches_the_hand_worked_table():
    windows = plant().expected_events_by_window(
        width=730.0, count=12, state=STATE
    )
    pump = windows.node_preventive["pump"].round(2)
    assert pump.tolist() == [0, 0.9, 0, 0, 0, 0, 0, 0, 0.05, 0.65, 0, 0.01]
    # The pump's first chance is R(6000) / R(5200), by hand.
    life = W([9000.0, 2.5])
    assert windows.node_preventive["pump"][1] == pytest.approx(
        life.sf(6000.0) / life.sf(5200.0), abs=1e-6
    )
    # Block replacement of the valve at 8760: certain, 1760 h away, in
    # window 3 (1460 to 2190); the sensor's tests at 4380: 1380 h away,
    # in window 2, then 4380 h later, in window 8.
    # (skipped while the valve is down in a repair, so a hair below 1)
    assert windows.node_preventive["valve"].round(2).tolist() == (
        [0, 0, 1] + [0] * 9
    )
    assert windows.node_inspections["sensor"].round(6).tolist() == (
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    )
    # A task and a system outage are different counts.
    assert (
        windows.system_planned_outages[1] < windows.node_preventive["pump"][1]
    )


def test_uneven_windows_and_width_and_count_agree():
    rbd = plant()
    even = rbd.expected_events_by_window(width=500.0, count=4, state=STATE)
    listed = rbd.expected_events_by_window(
        [500.0, 1000.0, 1500.0, 2000.0], state=STATE
    )
    np.testing.assert_allclose(
        even.node_failures["pump"], listed.node_failures["pump"]
    )
    uneven = rbd.expected_events_by_window([300.0, 2000.0], state=STATE)
    assert uneven.starts.tolist() == [0.0, 300.0]
    assert uneven.node_failures["pump"].sum() == pytest.approx(
        even.node_failures["pump"].sum(), abs=1e-9
    )


def test_a_unit_in_its_long_run_state_fails_at_a_steady_rate():
    lam, mu = 0.01, 0.2
    rbd = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {"c": {"reliability": E([lam]), "repairability": E([mu])}},
    )
    windows = rbd.expected_events_by_window(
        width=50.0, count=4, state="stationary"
    )
    # lam * availability * time, in every window.
    expected = lam * mu / (lam + mu) * 50.0
    np.testing.assert_allclose(windows.system_failures, expected, rtol=1e-5)


def test_the_windows_must_be_given_properly():
    rbd = plant()
    for bad in (
        dict(),
        dict(width=10.0),
        dict(ends=[10.0], width=10.0, count=1),
        dict(ends=[0.0, 5.0]),
        dict(ends=[5.0, 5.0]),
        dict(width=-1.0, count=2),
        dict(width=10.0, count=0),
    ):
        with pytest.raises(ValueError):
            rbd.expected_events_by_window(**bad)


def test_costs_by_hand_on_a_nearly_unfailing_bearing():
    # Shuts every 180 days, the next in 21: at 21, 201, 381, ... A bearing
    # that never fails costs exactly $100 a shut.
    rbd = RepairableRBD(
        [("s", "b"), ("b", "t")],
        {
            "b": {
                "reliability": E([1e-12]),
                "repairability": "instant",
                "replace_cost": 400.0,
                "preventive": {
                    "interval": 180.0,
                    "policy": "block",
                    "cost": 100.0,
                },
            }
        },
    )
    out = rbd.cost_by_window(
        [82.0, 447.0], state=BEARING_STATE, mc_samples=50, seed=1
    )
    assert out.mean.round(6).tolist() == [100.0, 200.0]
    assert out.samples.tolist() == [[100.0] * 50, [200.0] * 50]
    assert out.outcomes(1) == [{"cost": 200.0, "share": 1.0}]
    assert out.probability_above(150.0).tolist() == [0.0, 1.0]


def test_cost_means_agree_with_the_simulation_and_add_up():
    rbd = bearing()
    ends = [82.0, 447.0, 813.0, 1086.0]
    out = rbd.cost_by_window(
        ends, state=BEARING_STATE, mc_samples=20_000, seed=1
    )
    assert isinstance(out, CostByWindow)
    whole = rbd.expected_cost(1086.0, state=BEARING_STATE).mean
    assert out.mean.sum() == pytest.approx(whole, rel=1e-12)
    # README's every-shut policy: 1,155 over 1,086 days.
    assert whole == pytest.approx(1155.0, abs=0.5)
    assert np.all(
        np.abs(out.simulated_mean - out.mean) < 4 * out.standard_error
    )
    assert out.samples.min() >= 0.0
    # The same histories cut at each end: the windows add to one run.
    run = rbd.cost(
        1086.0,
        state=BEARING_STATE,
        mc_samples=20_000,
        seed=1,
        control_variate=False,
        conditional=False,
    )
    np.testing.assert_allclose(out.samples.sum(axis=0), run.samples)


def test_outcomes_percentiles_and_budgets():
    rbd = bearing()
    out = rbd.cost_by_window(
        [263.0, 629.0], state=BEARING_STATE, mc_samples=20_000, seed=1
    )
    first = out.outcomes(0)
    assert sum(o["share"] for o in first) == pytest.approx(1.0)
    assert [o["cost"] for o in first] == sorted(o["cost"] for o in first)
    # Costs come in steps of $100 shut and $400 failure.
    assert all(o["cost"] % 100.0 == 0.0 for o in first)
    # README: more than $200 (any failure) in about 29% of the first window.
    assert out.probability_above(200.0)[0] == pytest.approx(0.29, abs=0.02)
    assert out.percentiles.shape == (2, 3)
    assert out.levels == (10.0, 50.0, 90.0)
    np.testing.assert_allclose(out.percentiles[:, 1], out.percentile(50.0))
    # Ranges do not add.
    assert out.total_percentile(90.0) <= out.percentile(90.0).sum()
    assert out.to_dict()["n_simulations"] == 20_000


def test_the_same_seed_gives_the_same_answer():
    a = bearing().cost_by_window(width=100.0, count=3, mc_samples=200, seed=5)
    b = bearing().cost_by_window(width=100.0, count=3, mc_samples=200, seed=5)
    np.testing.assert_array_equal(a.samples, b.samples)
