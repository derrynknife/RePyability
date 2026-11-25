"""Events counted in the simulations' histories (``events_by_period`` on a
duty plan that changes over time).

Each loop records a component's events as it counts them
(``_events.TALLY_*``: failures, corrective actions, tasks and tests),
those that change no state (a task or test in zero time, a test finding
a hidden failure) included, so a history counts every one. Checked here
against the exact counts where a diagram has them (a duty of one share),
by hand where the plan fixes them (tasks, tests and shuts on dates), and
on a plan that changes against the one-node diagram with no duty run to
``D(end)``. Bounds fixed before the runs: four standard errors.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _events, _timeline_runs
from repyability.schedule import (
    CalendarContext,
    DutyPlan,
    Shifts,
    UsageForecast,
    events_by_period,
)

ONE = [("s", "a"), ("a", "t")]
W = surv.Weibull.from_params([400.0, 2.0])
E = surv.Exponential.from_params
BITS = {
    "node_failures": _events.TALLY_FAILURE,
    "node_corrective": _events.TALLY_CORRECTIVE,
    "node_preventive": _events.TALLY_PREVENTIVE,
    "node_inspections": _events.TALLY_TEST,
}
CASES = {
    "a block task in zero time": {
        "repairability": "instant",
        "preventive": {"policy": "block", "interval": 150.0},
    },
    "an age task in zero time": {
        "repairability": E([0.1]),
        "preventive": {"policy": "age", "interval": 200.0},
    },
    "tests in zero time": {
        "repairability": E([0.2]),
        "inspection": {"interval": 100.0},
    },
    "tests that take time": {
        "repairability": E([0.2]),
        "inspection": {"interval": 100.0, "duration": E([1.0])},
    },
    "tests that can miss": {
        "repairability": E([0.2]),
        "inspection": {"interval": 100.0, "coverage": 0.7, "full_test": 300.0},
    },
}


def per_simulation(events, bit, ends, n):
    """Each simulation's count of the events with ``bit`` in each window."""
    chosen = (events.kinds & bit) != 0
    window = np.searchsorted(ends, events.times[chosen], side="right")
    inside = window < ends.size
    out = np.zeros((n, ends.size))
    np.add.at(out, (events.history[chosen][inside], window[inside]), 1.0)
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_histories_count_what_the_exact_methods_count(name):
    rbd = RepairableRBD(ONE, {"a": {"reliability": W, **CASES[name]}})
    ends = np.array([250.0, 500.0, 1000.0, 2000.0])
    n = 4000
    exact = rbd.expected_events_by_window(ends)
    _, counted = _timeline_runs.simulate(
        rbd, 2000.0, n, 5, None, None, False, "python", None, 0, counted=True
    )
    for field, bit in BITS.items():
        counts = per_simulation(counted["a"], bit, ends, n)
        want = np.asarray(getattr(exact, field)["a"])
        se = counts.std(axis=0, ddof=1) / math.sqrt(n)
        # Tasks and tests on a schedule: the same count in every run (the
        # exact count to its quadrature's digits).
        assert np.all(np.abs(counts.mean(axis=0) - want) <= 4.0 * se + 1e-6)


# -- on a plan that changes -------------------------------------------------

FRAME = UsageForecast.calendar("2027-01-01", "2027-04-01")
SHIFTS = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
    "2027-01-01", "2027-04-01"
)
CONTEXT = CalendarContext(FRAME, {"a": SHIFTS})


def by_month(spec, runs=2000, **options):
    rbd = RepairableRBD(ONE, {"a": {"reliability": W, **spec}}, **options)
    months = events_by_period(rbd, CONTEXT, mc_samples=runs, seed=1)
    assert "simulations' histories" in months.notes[0]
    return months.events


def test_a_task_in_zero_time_is_counted_on_its_dates():
    # Every 168 hours of the calendar: four in each of January (744 h),
    # February (to 1,416 h) and March (to 2,160 h).
    events = by_month(
        {
            "repairability": "instant",
            "preventive": {"policy": "block", "interval": 168.0},
        }
    )
    np.testing.assert_array_equal(events.node_preventive["a"], [4, 4, 4])
    assert np.all(events.node_inspections["a"] == 0.0)


def test_tests_in_operating_time_are_counted_where_the_plan_reaches_them():
    # Every 100 operating hours at 16 a weekday: 336 h in January (21
    # weekdays), 656 h by February's end, 1,024 h by March's: tests at
    # 100-300, 400-600 and 700-1,000 h.
    events = by_month(
        {
            "repairability": E([0.2]),
            "inspection": {"interval": 100.0, "basis": "operating"},
        }
    )
    np.testing.assert_array_equal(events.node_inspections["a"], [3, 3, 4])
    assert np.all(events.node_preventive["a"] == 0.0)


def test_a_shut_is_counted_and_its_holds_are_not_tasks():
    # Shuts at 500, 1,000, 1,500 and 2,000 h: one in January, one in
    # February and two in March, none of them a preventive task.
    events = by_month(
        {"repairability": E([0.2])},
        shutdowns={"sd": {"period": 500.0, "duration": 24.0}},
    )
    np.testing.assert_array_equal(events.shutdowns, [1, 1, 2])
    assert np.all(events.node_preventive["a"] == 0.0)
    assert np.all(events.system_planned_outages <= events.shutdowns)
    assert np.all(events.system_planned_outages > 0.9 * events.shutdowns)


@pytest.mark.parametrize(
    "spec",
    [
        {
            "repairability": "instant",
            "inspection": {"interval": 100.0, "basis": "operating"},
        },
        {
            "repairability": "instant",
            "inspection": {
                "interval": 100.0,
                "basis": "operating",
                "coverage": 0.7,
                "full_test": 300.0,
            },
        },
    ],
    ids=["tests", "tests that can miss"],
)
def test_the_counts_on_the_plan_are_the_operating_processes(spec):
    # Repaired at once and tested in zero time, the unit is its
    # operating-time process seen through the shifts: over the quarter,
    # its counts are the one-node diagram's with no duty over the
    # quarter's operating hours, exact.
    runs = 4000
    events = by_month(spec, runs=runs)
    alone = RepairableRBD(ONE, {"a": {"reliability": W, **spec}})
    life = CONTEXT.diagram(alone).components["a"].reliability
    hours = float(life.operating(2160.0))
    exact = alone.expected_events_by_window(np.array([hours]))
    for field in ("node_failures", "node_corrective", "node_inspections"):
        got = float(np.sum(getattr(events, field)["a"]))
        want = float(getattr(exact, field)["a"][0])
        # A count's variance is at most its mean squared and its mean,
        # here below 2.5: a bound wider than its own standard error.
        se = math.sqrt(want + want**2) / math.sqrt(runs)
        assert abs(got - want) < 4.0 * se, (field, got, want)
