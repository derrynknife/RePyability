"""A part's cost in each of a run of windows from its state, and how often
each outcome happens (UC-9 of ``repyability/maintenance/README.md``):
``NonRepairable.cost_by_window``.
"""

import json

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import poisson

from repyability import NodeState, NonRepairable, WindowCosts
from repyability.maintenance._options import _unit

EDGES = [263.0, 629.0, 994.0, 1359.0]  # UC-9's four years from now
POLICIES = [
    {},
    {"every": 298.0, "resets": True},
    {"every": 180.0, "resets": False, "since_renewal": 159.0},
]


@pytest.fixture
def bearing():
    return NonRepairable(
        surv.Weibull.from_params([365.0, 2.0]),
        planned_cost=100.0,
        unplanned_cost=400.0,
    )


def pad(P, shape):
    out = np.zeros(shape)
    out[: P.shape[0], : P.shape[1]] = P
    return out


@pytest.mark.parametrize("age", [0.0, 40.0])
def test_an_exponential_life_has_independent_poisson_windows(age):
    # Run to failure with a constant rate: the failures in disjoint
    # windows are independent Poisson(lam w_k), whatever happened before.
    part = NonRepairable(
        surv.Exponential.from_params([0.01]), unplanned_cost=5.0
    )
    out = part.cost_by_window([100.0, 250.0, 300.0], age=age)
    for window, length in zip(out.windows, [100.0, 150.0, 50.0]):
        k = np.arange(len(window.failure_probabilities))
        error = np.abs(
            window.failure_probabilities - poisson.pmf(k, 0.01 * length)
        )
        assert error.max() <= window.precision
        assert window.mean == pytest.approx(5 * 0.01 * length, abs=1e-5)
    # The outcome classes: nothing, planned only (never), one, several.
    m = 0.01 * np.array([100.0, 150.0, 50.0])
    expected = np.stack(
        [np.exp(-m), 0 * m, m * np.exp(-m), 1 - np.exp(-m) * (1 + m)],
        axis=1,
    )
    assert out.outcome_probabilities == pytest.approx(expected, abs=2e-6)


def test_block_replacement_of_an_exponential_life_keeps_poisson_windows():
    # Memoryless lives: the planned replacements change nothing about the
    # failures, which stay Poisson by window; the planned ones are known.
    part = NonRepairable(
        surv.Exponential.from_params([0.01]),
        planned_cost=1.0,
        unplanned_cost=5.0,
    )
    out = part.cost_by_window(
        width=70.0, count=4, every=50.0, resets=False, since_renewal=20.0
    )
    # Planned at 30, 80, 130, 180, 230 (280 is the last window's end, so
    # the next window's): 1, 2, 1, 1 in the windows.
    planned = [int(np.argmax(w.planned_probabilities)) for w in out.windows]
    assert planned == [1, 2, 1, 1]
    for w in out.windows:
        assert w.planned_probabilities.max() == pytest.approx(1.0)
        k = np.arange(len(w.failure_probabilities))
        assert w.failure_probabilities == pytest.approx(
            poisson.pmf(k, 0.7), abs=1e-6
        )


@pytest.mark.parametrize("policy", POLICIES)
def test_the_windows_reconcile_with_the_whole_span(bearing, policy):
    out = bearing.cost_by_window(EDGES, age=224.0, **policy)
    whole = bearing.cost_distribution(1359.0, age=224.0, **policy)
    # Expectations add over windows; the total is the span's own.
    assert out.mean.sum() == pytest.approx(whole.mean, rel=2e-6)
    assert out.total.mean == pytest.approx(whole.mean)
    assert np.allclose(out.total.probabilities, whole.probabilities)
    for window in out.windows:
        assert window.probabilities.sum() == pytest.approx(
            1.0, abs=window.precision
        )
    assert out.outcome_probabilities.sum(axis=1) == pytest.approx(
        np.ones(4), abs=out.precision
    )
    # The first window is cost_distribution's own (on a grid sized for
    # the whole span, so within the two precisions).
    first = bearing.cost_distribution(263.0, age=224.0, **policy)
    shape = np.maximum(
        first.probabilities.shape, out.windows[0].probabilities.shape
    )
    gap = pad(out.windows[0].probabilities, shape) - pad(
        first.probabilities, shape
    )
    assert np.abs(gap).max() <= out.windows[0].precision + first.precision


@pytest.mark.parametrize("policy", POLICIES[1:])
def test_each_window_expects_the_work_the_module_expects(bearing, policy):
    # Each window's expected failures and planned replacements against the
    # difference of a one-component RepairableRBD's expected_events at its
    # ends (exact, by another method), through the windows' dependence.
    out = bearing.cost_by_window(EDGES, age=224.0, **policy)
    plan = {"interval": policy["every"], "cost": 100.0}
    state = NodeState(age=224.0)
    if not policy["resets"]:
        plan["policy"] = "block"
        state = NodeState(age=224.0, since_scheduled=159.0)
    unit = _unit(bearing, plan)
    events = unit.expected_events(np.array(EDGES), state={"unit": state})
    failures = np.diff(events.node_failures["unit"], prepend=0.0)
    planned = np.diff(events.node_preventive["unit"], prepend=0.0)
    got = [w.expected_failures for w in out.windows]
    assert got == pytest.approx(failures, abs=2e-6)
    got = [w.expected_planned for w in out.windows]
    assert got == pytest.approx(planned, abs=2e-6)


def test_the_first_window_from_an_aged_start_is_the_conditional_life(
    bearing,
):
    # P(nothing in [0, e1)) = R(a + e1) / R(a) when no planned
    # replacement can fall in it (age replacement at 600 from 224).
    life = bearing.reliability
    out = bearing.cost_by_window([100.0, 400.0], 600.0, resets=True, age=224.0)
    assert out.outcome_probabilities[0, 0] == pytest.approx(
        float(life.sf(324.0) / life.sf(224.0)), abs=1e-12
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_the_simulation_agrees_window_by_window(bearing, policy):
    n = 40_000
    exact = bearing.cost_by_window(EDGES, age=224.0, **policy)
    simulated = bearing.cost_by_window(
        EDGES, age=224.0, method="simulate", mc_samples=n, seed=11, **policy
    )
    assert simulated.method == "simulated" and simulated.precision is None
    for a, b in zip(
        exact.windows + [exact.total], simulated.windows + [simulated.total]
    ):
        shape = np.maximum(a.probabilities.shape, b.probabilities.shape)
        p, q = pad(a.probabilities, shape), pad(b.probabilities, shape)
        assert np.all(np.abs(p - q) <= 4 * np.sqrt(p * (1 - p) / n) + 1e-4)
        assert abs(a.mean - b.mean) < 4 * b.standard_error
    # The simulated windows are of the same histories as their total.
    assert simulated.mean.sum() == pytest.approx(simulated.total.mean)


def test_budgets_and_outcomes(bearing):
    out = bearing.cost_by_window(
        EDGES,
        180.0,
        resets=False,
        age=224.0,
        since_renewal=159.0,
        budget=500.0,
    )
    assert np.array_equal(
        out.probability_above_budget, out.probability_above(500.0)
    )
    # More than 500 a year means a failure beside the shuts' 200 to 300.
    for window, p in zip(out.windows, out.probability_above_budget):
        assert p == pytest.approx(window.probability_above(500.0))
    assert out.outcomes(1)[0] == out.windows[1].outcomes()[0]
    assert out.percentiles.shape == (4, 3)


@pytest.mark.parametrize("unit", [1e-6, 1e6])
def test_the_answer_does_not_depend_on_the_unit_of_time(unit):
    def part(scale):
        return NonRepairable(
            surv.Weibull.from_params([365.0 * scale, 2.0]),
            planned_cost=100.0,
            unplanned_cost=400.0,
        )

    days = part(1.0).cost_by_window(EDGES, 298.0, resets=True, age=224.0)
    other = part(unit).cost_by_window(
        np.array(EDGES) * unit, 298.0 * unit, resets=True, age=224.0 * unit
    )
    assert other.outcome_probabilities == pytest.approx(
        days.outcome_probabilities, abs=1e-9
    )


def test_the_result_goes_through_json(bearing):
    out = bearing.cost_by_window(width=200.0, count=3, age=50.0, budget=400.0)
    assert isinstance(out, WindowCosts)
    data = json.loads(json.dumps(out.to_dict()))
    assert data["outcome_classes"] == list(out.outcome_classes)
    assert len(data["windows"]) == 3
    assert data["total"]["mean"] == pytest.approx(out.total.mean)


@pytest.mark.parametrize(
    "kwargs, words",
    [
        ({}, "Give the windows' ends"),
        ({"ends": [100.0, 50.0]}, "ends must increase"),
        ({"ends": [100.0], "width": 10.0, "count": 2}, "not both"),
        ({"ends": [100.0], "every": 50.0}, "resets must be True"),
    ],
)
def test_windows_are_refused_in_plain_words(bearing, kwargs, words):
    with pytest.raises((ValueError, TypeError), match=words):
        bearing.cost_by_window(**kwargs)
