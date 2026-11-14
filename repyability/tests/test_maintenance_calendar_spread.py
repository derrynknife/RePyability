"""A part's cost over a window, and in each of a run of windows, from its
state under a calendar policy, with its spread (UC-8 and UC-9 of
``repyability/maintenance/README.md``): ``NonRepairable.cost_distribution``
and ``cost_by_window`` with ``calendar=``."""

import json

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import poisson

from repyability import (
    Calendar,
    CostDistribution,
    NodeState,
    NonRepairable,
    Opportunity,
    RepairableRBD,
)
from repyability.maintenance import _replacement

LIFE = surv.Weibull.from_params([365.0, 2.0])
# UC-7's plant: a shut every 180 days (the next in 21) at $100, no
# planned replacement between shuts, $400 at a failure.
SHUTS = Calendar(
    [Opportunity("shut", 100.0, period=180.0, offset=21.0)],
    failure=400.0,
    anytime=None,
)


def solve(part, calendar, **options):
    """The calendar policy on a grid of a given step (coarser than the
    public method's, for speed)."""
    return _replacement.optimal_calendar_policy(part, calendar, **options)


def bearing(life=LIFE):
    return NonRepairable(life, planned_cost=150.0, unplanned_cost=400.0)


@pytest.fixture(scope="module")
def best():
    return solve(bearing(), SHUTS, step=1.0)


def padded(a, b):
    rows = max(a.shape[0], b.shape[0])
    cols = max(a.shape[1], b.shape[1])
    out = np.zeros((2, rows, cols))
    out[0, : a.shape[0], : a.shape[1]] = a
    out[1, : b.shape[0], : b.shape[1]] = b
    return out


def agree(exact, simulated, n):
    """Every probability within four standard errors."""
    a, b = padded(exact, simulated)
    return np.all(np.abs(a - b) <= 4 * np.sqrt(a * (1 - a) / n) + 1e-4)


# -- closed forms -------------------------------------------------------------


def test_an_exponential_life_is_never_replaced_and_has_poisson_failures():
    # Memoryless: no planned replacement pays, so the policy never
    # replaces, and the failures in any window are Poisson(lam w), the
    # windows independent, whatever the age.
    part = NonRepairable(
        surv.Exponential.from_params([0.01]),
        planned_cost=1.0,
        unplanned_cost=5.0,
    )
    calendar = Calendar(
        [Opportunity("shut", 1.0, period=50.0)], failure=5.0, anytime=None
    )
    policy = solve(part, calendar, step=0.5)
    assert policy._next_planned(0.0, 0.0)[2] is None
    out = part.cost_distribution(300.0, calendar=policy, age=80.0)
    k = np.arange(len(out.failure_probabilities))
    assert out.probabilities.shape[1] == 1  # nothing planned
    error = np.abs(out.failure_probabilities - poisson.pmf(k, 3.0)).max()
    assert error <= out.precision < 1e-5
    assert out.mean == pytest.approx(15.0, abs=1e-4)
    windows = part.cost_by_window(
        [100.0, 250.0, 300.0], calendar=policy, age=80.0
    )
    for window, width in zip(windows.windows, (100.0, 150.0, 50.0)):
        k = np.arange(len(window.failure_probabilities))
        expected = poisson.pmf(k, 0.01 * width)
        error = np.abs(window.failure_probabilities - expected).max()
        assert error <= window.precision


def test_a_part_due_now_is_replaced_at_once(best):
    # 500 days old at the shut at 21: replaced there for certain, and
    # from then the policy runs from a new part.
    out = bearing().cost_distribution(21.0, calendar=best, age=500.0)
    # It fails before the shut with S(21) = R(521) / R(500) to go.
    survive = float(LIFE.sf(521.0) / LIFE.sf(500.0))
    assert out.probabilities[0, 0] == pytest.approx(survive, abs=1e-6)
    later = bearing().cost_distribution(22.0, calendar=best, age=500.0)
    assert later.probabilities[0, 1] == pytest.approx(
        survive * float(LIFE.sf(1.0)), abs=1e-4
    )


# -- the use case -------------------------------------------------------------


def test_the_uc8_figures_for_the_best_policy(best):
    # UC-8's answer by hand (20,000 simulations of the policy "at a shut if
    # at least 138 days old"): the means and the 10th, 50th and 90th
    # percentiles over each window.
    assert best.schedule()[0][1:3] == ("shut", 138.0)
    by_hand = {
        82.0: (133, [100, 100, 100]),
        447.0: (487, [300, 300, 900]),
        813.0: (846, [500, 800, 1400]),
        1086.0: (1090, [600, 1000, 1600]),
    }
    for window, (mean, percentiles) in by_hand.items():
        out = bearing().cost_distribution(window, calendar=best, age=224.0)
        assert out.mean == pytest.approx(mean, abs=2.0)
        assert list(out.percentiles) == percentiles
        assert out.precision < 1e-5
        assert out.policy == "calendar" and out.method == "numerical"
        assert out.kinds == ("shut",) and out.kind_prices == (100.0,)


def test_the_uc9_windows_for_the_best_policy(best):
    # UC-9's exact means by hand: $307, then $359 a full window.
    years = bearing().cost_by_window(
        [263.0, 629.0, 994.0, 1359.0], calendar=best, age=224.0, budget=600.0
    )
    assert np.round(years.mean) == pytest.approx([307, 359, 359, 359])
    assert years.outcome_probabilities.sum(axis=1) == pytest.approx(1.0)
    # The windows' means add up to the whole span's; the spreads do not.
    assert years.mean.sum() == pytest.approx(years.total.mean, abs=1e-4)
    assert years.percentiles[:, -1].sum() > years.total.percentiles[-1]
    # The first window is cost_distribution's.
    first = bearing().cost_distribution(263.0, calendar=best, age=224.0)
    a, b = padded(first.probabilities, years.windows[0].probabilities)
    assert np.abs(a - b).max() < 1e-12


# -- against simulations ------------------------------------------------------


def test_the_simulation_agrees_with_the_numerical_route(best):
    n = 20_000
    exact = bearing().cost_by_window([150.0, 400.0], calendar=best, age=224.0)
    simulated = bearing().cost_by_window(
        [150.0, 400.0],
        calendar=best,
        age=224.0,
        method="simulate",
        mc_samples=n,
        seed=3,
    )
    for e, s in zip(
        exact.windows + [exact.total], simulated.windows + [simulated.total]
    ):
        assert s.method == "simulated" and s.n_simulations == n
        assert abs(s.mean - e.mean) < 4 * s.standard_error
        assert agree(e.probabilities, s.probabilities, n)
    again = bearing().cost_by_window(
        [150.0, 400.0],
        calendar=best,
        age=224.0,
        method="simulate",
        mc_samples=n,
        seed=3,
    )
    assert np.array_equal(
        again.total.probabilities, simulated.total.probabilities
    )


def test_the_diagram_simulation_agrees(best):
    # Another implementation: a one-component RepairableRBD replaced on the
    # same policy, simulated by its event loop from the same state.
    rbd = RepairableRBD(
        [("s", "b"), ("b", "t")],
        {
            "b": {
                "reliability": LIFE,
                "repairability": "instant",
                "replace_cost": 400.0,
                "preventive": {"policy": "calendar", "calendar": best},
            }
        },
    )
    n = 4_000
    run = rbd.cost(
        400.0,
        mc_samples=n,
        seed=5,
        engine="python",
        state={"b": NodeState(age=224.0)},
    )
    exact = bearing().cost_distribution(400.0, calendar=best, age=224.0)
    costs = np.asarray(run.samples).ravel()
    assert abs(costs.mean() - exact.mean) < 4 * costs.std() / np.sqrt(n)
    for value, chance in zip(exact.costs, exact.cost_probabilities):
        share = np.mean(np.isclose(costs, value))
        assert abs(share - chance) <= 4 * np.sqrt(chance * (1 - chance) / n)


# -- several kinds of opportunity ---------------------------------------------


def test_outcomes_by_kind_price_each_kind():
    life = surv.Weibull.from_params([1000.0, 2.5])
    part = NonRepairable(life, planned_cost=4.0, unplanned_cost=20.0)
    calendar = Calendar(
        [
            Opportunity("quarterly", 3.0, period=91.0),
            Opportunity("annual", 1.0, period=364.0),
        ],
        failure=20.0,
        anytime=None,
    )
    policy = solve(part, calendar, step=1.0)
    out = part.cost_distribution(1500.0, calendar=policy, age=700.0)
    assert set(out.kinds) <= {"quarterly", "annual"}
    prices = dict(zip(out.kinds, out.kind_prices))
    assert prices["annual"] == 1.0
    total = 0.0
    for o in out.outcomes():
        spent = sum(prices[k] * n for k, n in o["planned"].items())
        assert o["cost"] == pytest.approx(20.0 * o["failures"] + spent)
        total += o["probability"]
    assert total == pytest.approx(1.0, abs=1e-9)
    assert out.cost_probabilities.sum() == pytest.approx(1.0, abs=1e-9)
    # The counts by kind add up to the counts of planned replacements.
    by_total = np.zeros(out.probabilities.shape)
    for o in out.outcomes():
        by_total[o["failures"], sum(o["planned"].values())] += o["probability"]
    assert by_total == pytest.approx(out.probabilities, abs=1e-11)


# -- the answer's shape, units, inputs ----------------------------------------


def test_the_answer_does_not_depend_on_the_unit_of_time(best):
    unit = 24.0
    life = surv.Weibull.from_params([365.0 * unit, 2.0])
    calendar = Calendar(
        [Opportunity("shut", 100.0, period=180.0 * unit, offset=21.0 * unit)],
        failure=400.0,
        anytime=None,
    )
    policy = solve(bearing(life), calendar, step=unit)
    days = bearing().cost_distribution(700.0, calendar=best, age=224.0)
    hours = bearing(life).cost_distribution(
        700.0 * unit, calendar=policy, age=224.0 * unit
    )
    assert hours.probabilities == pytest.approx(days.probabilities, abs=1e-9)


def test_the_result_goes_through_json(best):
    out = bearing().cost_distribution(
        400.0, calendar=best, age=50.0, budget=500.0
    )
    assert isinstance(out, CostDistribution)
    data = json.loads(json.dumps(out.to_dict()))
    assert np.array(data["probabilities"]) == pytest.approx(out.probabilities)
    assert data["kinds"] == ["shut"] and data["kind_prices"] == [100.0]
    assert data["kind_outcomes"] == out.outcomes()
    assert data["policy"] == "calendar" and data["every"] is None
    windows = bearing().cost_by_window([200.0, 400.0], calendar=best)
    assert json.loads(json.dumps(windows.to_dict()))["policy"] == "calendar"


def test_calendar_policies_are_refused_in_plain_words(best):
    part = bearing()
    with pytest.raises(ValueError, match="no every, resets"):
        part.cost_distribution(100.0, 180.0, resets=False, calendar=best)
    with pytest.raises(TypeError, match="CalendarPolicy"):
        part.cost_distribution(100.0, calendar="every shut")
    other = bearing(surv.Weibull.from_params([500.0, 2.0]))
    with pytest.raises(ValueError, match="another life"):
        other.cost_distribution(100.0, calendar=best)


def test_a_policy_worked_out_by_integration_is_simulated():
    part = bearing()
    policy = solve(part, SHUTS, method="integration")
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        part.cost_distribution(100.0, calendar=policy)
    out = part.cost_distribution(
        100.0, calendar=policy, method="simulate", mc_samples=200, seed=1
    )
    assert out.method == "simulated"
