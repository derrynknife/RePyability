"""Replacing a part at planned stops only if it is older than a threshold,
by calendar period (UC-70 of the schedule README).

The stops are a site's shutdowns on dates (``shutdown_spec`` from a
rule), each part's task ``{"shutdown": name, "older_than": X}``, and the
forecast ``events_by_period``, each part from its own age: an
exponential life against its closed form (no ageing, so a part is
replaced at a stop exactly when it has not failed for ``X``), and six
seals of different ages against a plain simulation of the same diagram.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD
from repyability.schedule import (
    Recurrence,
    UsageForecast,
    events_by_period,
    shutdown_spec,
)

START, END = "2027-01-01", "2028-01-01"
QUARTERLY = Recurrence.monthly(day=1, every=3, anchor=START)


def stops(duration: float = 12.0) -> dict:
    """A 12-hour stop on the first of each quarter: 1 January (the
    start), 1 April, 1 July and 1 October, and 1 January 2028 (the end)."""
    return shutdown_spec(QUARTERLY, START, END, duration=duration)


def test_the_stops_on_the_diagram_clock():
    assert stops()["times"] == [0.0, 2160.0, 4344.0, 6552.0, 8760.0]


@pytest.mark.parametrize("age", [0.0, 1000.0, 2500.0])
def test_a_constant_rate_against_its_closed_form(age):
    # With no ageing, a part is older than X at a stop exactly when it
    # has not failed (nor been replaced) in the X before it: exp(-lam X)
    # at each stop after the first X hours. Replaced, it is off line for
    # the stop's 12 hours, which takes them off its running.
    lam, threshold = 1.0 / 3000.0, 1500.0
    rbd = RepairableRBD(
        [("s", "seal"), ("seal", "t")],
        {
            "seal": {
                "reliability": surv.Exponential.from_params([lam]),
                "repairability": "instant",
                "repair_cost": 1000.0,
                "preventive": {
                    "shutdown": "quarterly",
                    "older_than": threshold,
                    "cost": 200.0,
                },
            }
        },
        shutdowns={"quarterly": stops()},
    )
    quarters = events_by_period(
        rbd,
        UsageForecast.calendar(START, END),
        period="quarter",
        unit="hours",
        state={"seal": NodeState(age=age)},
    )
    planned = quarters.events.node_preventive["seal"]
    replaced = math.exp(-lam * threshold)
    # The stop at the start itself is now, not ahead: the forecast starts
    # from each part's state then, so it replaces nothing, whatever the
    # part's age (as a task due at the start is not taken as done).
    want = np.array([0.0, replaced, replaced, replaced])
    np.testing.assert_allclose(planned, want, rtol=0.0, atol=2e-4)
    hours = np.diff([0.0, 2160.0, 4344.0, 6552.0, 8760.0])
    failures = quarters.events.node_failures["seal"]
    np.testing.assert_allclose(
        failures, lam * (hours - 12.0 * want), rtol=0.0, atol=2e-4
    )


def test_six_seals_of_different_ages_against_a_plain_simulation():
    ages = [0.0, 1500.0, 3000.0, 4500.0, 6000.0, 7500.0]
    names = [f"seal {k}" for k in range(len(ages))]
    seal = {
        "reliability": surv.Weibull.from_params([8000.0, 2.5]),
        "repairability": "instant",
        "repair_cost": 1000.0,
        "preventive": {
            "shutdown": "quarterly",
            "older_than": 4000.0,
            "cost": 200.0,
        },
    }
    edges = [("s", names[0])]
    edges += list(zip(names, names[1:])) + [(names[-1], "t")]
    rbd = RepairableRBD(
        edges,
        {name: dict(seal) for name in names},
        shutdowns={"quarterly": stops()},
    )
    state = {name: NodeState(age=age) for name, age in zip(names, ages)}
    months = events_by_period(
        rbd, UsageForecast.calendar(START, END), unit="hours", state=state
    )
    planned = sum(months.events.node_preventive[n] for n in names)
    failures = sum(months.events.node_failures[n] for n in names)
    # The planned replacements fall in the months the stops open.
    assert set(np.flatnonzero(planned > 0)) <= {0, 3, 6, 9}
    # Each seal on its own: the older it is, the likelier it goes first.
    own = [float(np.sum(months.events.node_preventive[n])) for n in names]
    assert own[-1] > own[0]
    expected = 1000.0 * failures.sum() + 200.0 * planned.sum()
    run = rbd.cost(
        8760.0, state=state, mc_samples=4000, seed=5, control_variate=False
    )
    assert abs(run.mean - expected) < 4.0 * run.mean_se
    exact = rbd.expected_cost(8760.0, state=state).mean
    assert exact == pytest.approx(expected, rel=1e-9)
