"""The net present value of each simulated run of a diagram with
shutdowns (``npv_distribution``, ``compare_npv``): each shut's cost at
its starts, a hold not charged as a task, and what a hold hides (a
failure in the shut, a renewal made in it) charged all the same. Each run
at no discounting against the simulation's own cost and uptime, the mean
against the exact ``expected_npv``, and the shuts' costs against their
calendar."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _timeline_runs, routes
from repyability.tests.test_outage_opportunistic import line
from repyability.tests.test_shutdowns import one, outages, shut

E = surv.Exponential.from_params
W = surv.Weibull.from_params


def opportunistic():
    """Outages that hold a and b for 150 and cost 40, renewing a if it is
    500 old: failures in the shuts and renewals made in them, hidden."""
    return line({"a": 500.0}, duration=150.0, cost=40.0)


def moment():
    """A shut in no time that costs 5 at each start, every 100."""
    return one(
        {"repair_cost": 2.0},
        shutdown={"period": 100.0, "duration": 0.0, "cost": 5.0},
    )


def held_unworked():
    """A pump under its own age replacement, held by shuts that do no
    work on it: its holds are no tasks."""
    return one(
        {
            "reliability": W([150.0, 2.0]),
            "repair_cost": 20.0,
            "preventive": {
                "policy": "age",
                "interval": 80.0,
                "duration": W([2.0, 2.0]),
                "cost": 7.0,
            },
        },
        shutdown=shut(cost=3.0),
    )


CASES = {
    "outages at given times": (outages, 800.0),
    "opportunistic renewals in long shuts": (opportunistic, 3000.0),
    "a shut in no time": (moment, 1000.0),
    "holds of a part with its own tasks": (held_unworked, 1000.0),
}


@pytest.mark.parametrize("case", CASES)
def test_undiscounted_each_run_is_its_value_less_its_simulated_cost(case):
    make, horizon = CASES[case]
    rbd, value, n = make(), 2.0, 300
    runs = rbd.npv_distribution(
        horizon, value_rate=value, mc_samples=n, seed=5
    )
    cost = rbd.cost(horizon, mc_samples=n, seed=5, control_variate=False)
    uptime = rbd.simulate_timelines(horizon, mc_samples=n, seed=5).system
    expected = value * uptime.uptime - cost.samples - cost.acquisition_cost
    assert np.allclose(runs.samples, expected, rtol=1e-10, atol=1e-8)
    for category, amount in cost.by_category.items():
        assert runs.cost_by_category.get(category, 0.0) == pytest.approx(
            amount, rel=1e-10, abs=1e-9
        ), category


@pytest.mark.parametrize(
    "case",
    [
        "outages at given times",
        "opportunistic renewals in long shuts",
        "holds of a part with its own tasks",
    ],
)
def test_the_histories_mark_holds_and_what_they_hid(case):
    # The cases above are not priced right by the recorded changes alone.
    make, horizon = CASES[case]
    rbd = make()
    _, hidden = _timeline_runs.simulate(
        rbd,
        horizon,
        300,
        5,
        None,
        None,
        False,
        "auto",
        None,
        0,
        None,
        hidden=True,
    )
    assert sum(int(h.held.sum()) for h in hidden.values()) > 0
    if case != "holds of a part with its own tasks":
        assert sum(h.times.size for h in hidden.values()) > 0


def test_a_shut_in_no_time_is_charged_at_each_start():
    rbd, horizon = moment(), 1000.0
    cost = rbd.cost(horizon, mc_samples=20, seed=1, control_variate=False)
    assert cost.by_category["shutdown"] == 45.0
    assert rbd.expected_cost(horizon).by_category["shutdown"] == 45.0


@pytest.mark.parametrize(
    "make, horizon",
    [
        (outages, 800.0),
        (
            lambda: one(
                {
                    "repair_cost": 2.0,
                    "preventive": {"shutdown": "line", "cost": 4.0},
                },
                shutdown=shut(cost=5.0),
            ),
            1000.0,
        ),
    ],
)
def test_the_mean_is_the_expected_npv(make, horizon):
    rbd, rate, value = make(), 1e-3, 3.0
    runs = rbd.npv_distribution(
        horizon, discount_rate=rate, value_rate=value, mc_samples=4000, seed=3
    )
    exact = rbd.expected_npv(horizon, discount_rate=rate, value_rate=value)
    interval = runs.mean_interval(0.9999)
    assert interval.lower < exact.npv < interval.upper
    # Each shut's cost, the same in every run, discounted at its starts.
    starts = [
        start
        for spec in rbd._shutdowns.values()
        for start in spec.starts(horizon)
        for _ in range(1 if spec.cost else 0)
    ]
    costs = [
        spec.cost
        for spec in rbd._shutdowns.values()
        for _ in spec.starts(horizon)
        if spec.cost
    ]
    shut_cost = sum(c * math.exp(-rate * t) for c, t in zip(costs, starts))
    assert runs.cost_by_category["shutdown"] == pytest.approx(
        shut_cost, rel=1e-12
    )
    assert exact.cost_by_category["shutdown"] == pytest.approx(
        shut_cost, rel=1e-6
    )
    for category, amount in exact.cost_by_category.items():
        assert runs.cost_by_category.get(category, 0.0) == pytest.approx(
            amount, rel=0.1, abs=1e-9
        ), category


def test_two_plans_with_shutdowns_are_compared_run_for_run():
    rbd = opportunistic()
    assert rbd.analysis_routes()["npv_distribution"].route == (
        routes.SIMULATED
    )
    assert rbd.analysis_routes()["compare_npv"].route == routes.SIMULATED
    same = rbd.compare_npv(
        opportunistic(), 3000.0, discount_rate=1e-3, mc_samples=100, seed=2
    )
    assert np.array_equal(same.differences, np.zeros(100))
    # Renewing at an outage only if older than never happens: the plan
    # without the rule, which saves less than the rule's plan.
    never = line({"a": math.inf}, duration=150.0, cost=40.0)
    gain = rbd.compare_npv(
        never, 3000.0, discount_rate=1e-3, mc_samples=400, seed=2
    )
    assert gain.interval.standard_error < gain.independent_standard_error
    assert gain.this.cost_by_category["shutdown"] == pytest.approx(
        gain.other.cost_by_category["shutdown"], rel=1e-15
    )


def test_a_paused_part_held_in_a_shut_is_priced():
    # outages() stands c still in its own shut: its runs priced as above
    # and its own task's cost charged once per shut.
    rbd = outages()
    runs = rbd.npv_distribution(800.0, mc_samples=200, seed=9)
    assert runs.cost_by_component["c"] > 0.0
    cost = rbd.cost(800.0, mc_samples=200, seed=9, control_variate=False)
    assert runs.cost_by_component["c"] == pytest.approx(
        cost.by_component["c"], rel=1e-10
    )


def test_compiled_shuts_are_priced_as_in_python():
    pytest.importorskip("numba")
    rbd = one(
        {"repair_cost": 2.0, "preventive": {"shutdown": "line", "cost": 4.0}},
        shutdown={"period": 100.0, "offset": 20.0, "duration": 0.0},
    )
    python = rbd.npv_distribution(
        1000.0, mc_samples=50, seed=1, engine="python"
    )
    numba = rbd.npv_distribution(1000.0, mc_samples=50, seed=1, engine="numba")
    assert numba.engine == "numba"
    assert np.array_equal(python.samples, numba.samples)


def test_a_diagram_whose_only_costs_are_its_shuts():
    # A diagram with shutdowns and no costs at all but its shuts' is
    # priced: the shuts alone.
    rbd = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": E([1e-3]), "repairability": E([0.5])}},
        shutdowns={"x": {"times": [10.0, 20.0], "duration": 1.0, "cost": 4.0}},
    )
    runs = rbd.npv_distribution(100.0, mc_samples=5, seed=1)
    assert np.array_equal(runs.samples, np.full(5, -8.0))
