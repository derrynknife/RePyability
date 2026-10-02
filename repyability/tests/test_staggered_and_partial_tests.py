"""Staggered tests and tests that can miss a failure (#136): an inspection's
``offset`` (the time of its first test) and ``coverage`` (the chance that a
test finds a failure), with ``full_test`` (every so many tests, one that
finds every failure). Exact long-run and from-new values against closed
forms and direct integration, and simulations against them."""

import math

import numpy as np
import pytest
from scipy import integrate
from surpyval import Exponential

from repyability import NodeState, RepairableRBD

E = Exponential.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SINGLE = [("s", "a"), ("a", "t")]


def hidden(rate, interval, **inspection):
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval, **inspection},
    }


def up(rate, interval, t, offset=0.0, coverage=1.0, full_test=None):
    """A tested unit's long-run probability of being up at ``t``: by its
    definition, a Markov chain over the tests since the last full one."""
    position = (t - offset) % (full_test or interval)
    k = math.floor(position / interval)
    u = position - k * interval
    rho = 1.0 - (1.0 - coverage) * (1.0 - math.exp(-rate * interval))
    return rho**k * math.exp(-rate * u)


# -- the spec -----------------------------------------------------------------


@pytest.mark.parametrize(
    "inspection, match",
    [
        ({"offset": -1.0}, "offset"),
        ({"offset": 100.0}, "offset"),
        ({"offset": "soon"}, "offset"),
        ({"coverage": 1.5}, "coverage"),
        ({"coverage": -0.1}, "coverage"),
        ({"coverage": 0.9}, "give its full_test"),
        ({"coverage": 0.9, "full_test": 250.0}, "whole multiple"),
        ({"coverage": 0.9, "full_test": 50.0}, "whole multiple"),
        ({"phase": 3.0}, "offset, coverage, full_test"),
    ],
)
def test_the_spec_is_checked(inspection, match):
    with pytest.raises(ValueError, match=match):
        RepairableRBD(SINGLE, {"a": hidden(1e-3, 100.0, **inspection)})


def test_a_full_test_is_a_whole_number_of_tests():
    rbd = RepairableRBD(
        SINGLE,
        {"a": hidden(1e-3, 0.1, coverage=0.5, full_test=0.30000000000000004)},
    )
    assert rbd._inspection["a"].per_full_test == 3


# -- the long run -------------------------------------------------------------


def test_one_unit_is_not_moved_by_its_offset():
    alone = RepairableRBD(SINGLE, {"a": hidden(1e-3, 100.0)})
    later = RepairableRBD(SINGLE, {"a": hidden(1e-3, 100.0, offset=37.0)})
    assert later.mean_availability() == pytest.approx(
        alone.mean_availability(), rel=1e-13
    )
    assert later.node_availability()["a"] == alone.node_availability()["a"]


@pytest.mark.parametrize("coverage", [0.0, 0.3, 0.9])
def test_one_unit_whose_tests_miss_failures(coverage):
    rate, interval, every = 1e-3, 100.0, 5
    rbd = RepairableRBD(
        SINGLE,
        {
            "a": hidden(
                rate,
                interval,
                coverage=coverage,
                full_test=every * interval,
                offset=20.0,
            )
        },
    )
    rho = 1.0 - (1.0 - coverage) * (1.0 - math.exp(-rate * interval))
    want = (1.0 - rho**every) / ((1.0 - coverage) * rate * every * interval)
    assert rbd.node_availability()["a"] == pytest.approx(want, rel=1e-12)
    assert rbd.mean_availability() == pytest.approx(want, rel=1e-12)
    # It fails at its rate while it is up.
    assert rbd.system_failure_frequency() == pytest.approx(
        rate * want, rel=1e-12
    )


def test_staggered_pairs_by_their_definition():
    # A 1oo2 pair, tested together, staggered by half an interval, and
    # with the second's tests missing failures.
    rate, interval = 2e-6, 8760.0
    for b, period in [
        ({}, interval),
        ({"offset": interval / 2}, interval),
        ({"offset": 1000.0, "coverage": 0.6, "full_test": 3 * interval}, 3),
    ]:
        if period == 3:
            period = 3 * interval
        rbd = RepairableRBD(
            PAIR,
            {"a": hidden(rate, interval), "b": hidden(rate, interval, **b)},
        )

        def both_down(t):
            return (1.0 - up(rate, interval, t)) * (
                1.0 - up(rate, interval, t, **b)
            )

        breaks = sorted(
            {k * interval for k in range(int(period / interval) + 1)}
            | {
                b.get("offset", 0.0) + k * interval
                for k in range(int(period / interval))
            }
        )
        want = sum(
            integrate.quad(both_down, lo, hi, epsabs=0, epsrel=1e-13)[0]
            for lo, hi in zip(breaks[:-1], breaks[1:])
        )
        want /= period
        assert rbd.mean_unavailability() == pytest.approx(want, rel=1e-10)
    # Tested together, (lambda tau)^2 / 3 to first order (the issue's
    # 1.00983e-4); staggered by half an interval, 5 (lambda tau)^2 / 24.
    together = RepairableRBD(
        PAIR, {"a": hidden(rate, interval), "b": hidden(rate, interval)}
    )
    staggered = RepairableRBD(
        PAIR,
        {
            "a": hidden(rate, interval),
            "b": hidden(rate, interval, offset=interval / 2),
        },
    )
    x = rate * interval
    assert together.mean_unavailability() == pytest.approx(
        1.00983e-4, rel=1e-5
    )
    assert staggered.mean_unavailability() == pytest.approx(
        5 * x**2 / 24, rel=0.02
    )
    assert staggered.mean_unavailability() < together.mean_unavailability()


# -- from new -----------------------------------------------------------------


@pytest.mark.parametrize(
    "inspection",
    [
        {"offset": 30.0},
        {"coverage": 0.6, "full_test": 500.0},
        {"coverage": 0.6, "full_test": 500.0, "offset": 40.0},
        {"coverage": 0.0, "full_test": 300.0, "offset": 70.0},
    ],
)
def test_from_new(inspection):
    rate, interval = 1e-3, 100.0
    rbd = RepairableRBD(SINGLE, {"a": hidden(rate, interval, **inspection)})
    offset = inspection.get("offset", 0.0)
    x = np.array([5.0, 29.0, 35.0, 99.0, 150.0, 333.0, 777.0, 1234.5])
    want = [
        (
            math.exp(-rate * t)
            if t < offset
            else up(
                rate,
                interval,
                t,
                offset,
                inspection.get("coverage", 1.0),
                inspection.get("full_test"),
            )
        )
        for t in x
    ]
    np.testing.assert_allclose(rbd.point_availability(x), want, rtol=1e-12)
    # Every failure is found and repaired, at most one waits: the failures
    # less the repairs by t are the chance it is down then.
    for t, a in zip(x, want):
        events = rbd.expected_events(float(t))
        waiting = events.node_failures["a"] - events.node_corrective["a"]
        assert waiting == pytest.approx(1.0 - a, rel=1e-9, abs=1e-12)
        tests_before = sum(
            1 for k in range(100) if 0 < offset + k * interval < t
        )
        assert events.node_inspections["a"] == tests_before


@pytest.mark.parametrize(
    "inspection",
    [
        {},
        {"offset": 30.0},
        {"coverage": 0.6, "full_test": 500.0},
        {"coverage": 0.0, "full_test": 300.0, "offset": 70.0},
    ],
)
def test_simulated_as_the_exact_values(inspection):
    spec = hidden(1e-3, 100.0, cost=1.0, **inspection)
    spec["repair_cost"] = 10.0
    rbd = RepairableRBD(SINGLE, {"a": spec})
    window = 1230.0
    result = rbd.availability(window, mc_samples=20000, seed=11)
    interval = result.mean_availability_interval()
    exact = rbd.mission_availability(window)
    assert abs(interval.estimate - exact) < 4 * interval.standard_error
    # The tests, including those that miss a failure, are charged.
    assert result.cost.by_category["inspection"] == pytest.approx(12.0)
    repairs = rbd.expected_cost(window).by_category["repair"]
    assert result.cost.by_category["repair"] == pytest.approx(repairs, 0.02)


def test_a_staggered_pair_simulated_as_the_exact_values():
    rate, interval = 1e-3, 300.0
    rbd = RepairableRBD(
        PAIR,
        {
            "a": hidden(rate, interval, coverage=0.7, full_test=1200.0),
            "b": hidden(
                rate, interval, offset=100.0, coverage=0.5, full_test=900.0
            ),
        },
    )
    window = 7200.0
    result = rbd.availability(window, mc_samples=4000, seed=5)
    interval_ = result.mean_availability_interval()
    exact = rbd.mission_availability(window)
    assert abs(interval_.estimate - exact) < 4 * interval_.standard_error


def test_a_tested_unit_whose_tests_miss_failures_starts_new():
    rbd = RepairableRBD(
        SINGLE,
        {"a": hidden(1e-3, 100.0, coverage=0.5, full_test=300.0)},
    )
    for state in (NodeState(age=20.0), NodeState(phase=10.0)):
        with pytest.raises(NotImplementedError, match="leave it out"):
            rbd.point_availability([50.0], state={"a": state})
        with pytest.raises(NotImplementedError, match="leave it out"):
            rbd.availability(100.0, mc_samples=10, state={"a": state})


def test_a_state_places_a_staggered_calendar():
    # Last tested 10 before 0: the offset is for a start from new.
    rate, interval = 1e-3, 100.0
    staggered = RepairableRBD(
        SINGLE, {"a": hidden(rate, interval, offset=60.0)}
    )
    plain = RepairableRBD(SINGLE, {"a": hidden(rate, interval)})
    state = {"a": NodeState(phase=10.0)}
    x = np.array([5.0, 50.0, 95.0, 250.0])
    np.testing.assert_allclose(
        staggered.point_availability(x, state=state),
        plain.point_availability(x, state=state),
        rtol=1e-14,
    )
    simulated = [
        rbd.availability(300.0, mc_samples=200, seed=1, state=state)
        .mean_availability_interval()
        .estimate
        for rbd in (staggered, plain)
    ]
    assert simulated[0] == simulated[1]
    # From a test at 0 too.
    state = {"a": NodeState(phase=0.0)}
    np.testing.assert_allclose(
        staggered.point_availability(x, state=state),
        plain.point_availability(x, state=state),
        rtol=1e-14,
    )


# -- choosing intervals, and saving ---------------------------------------------


def test_choosing_intervals_keeps_an_offsets_share():
    rate = 1e-4
    rbd = RepairableRBD(
        PAIR,
        {
            "a": hidden(rate, 1000.0, cost=10.0),
            "b": hidden(rate, 1000.0, offset=500.0, cost=10.0),
        },
    )
    plan = rbd._with_intervals(inspection={"a": 2000.0, "b": 2000.0})
    assert plan._inspection["b"].offset == 1000.0
    chosen = rbd.optimal_inspection_intervals(
        allowed={"a": [500.0, 1000.0, 2000.0], "b": [500.0, 1000.0, 2000.0]},
        min_availability=0.999,
    )
    assert set(chosen.intervals) == {"a", "b"}


def test_intervals_are_not_chosen_for_tests_that_miss_failures():
    rbd = RepairableRBD(
        SINGLE,
        {"a": hidden(1e-3, 100.0, coverage=0.5, full_test=300.0)},
    )
    with pytest.raises(NotImplementedError, match="not chosen here"):
        rbd.optimal_inspection_intervals(min_availability=0.9)
    route = rbd.analysis_routes()["optimal_inspection_intervals"]
    assert route.route == "refused"


def test_saved_and_loaded():
    rbd = RepairableRBD(
        PAIR,
        {
            "a": hidden(1e-3, 100.0),
            "b": hidden(
                1e-3, 100.0, offset=50.0, coverage=0.8, full_test=400.0
            ),
        },
    )
    again = RepairableRBD.from_json(rbd.to_json())
    assert again._inspection == rbd._inspection
    assert again.mean_availability() == rbd.mean_availability()
    # A schedule without them saves as before.
    saved = {c["node"]: c["component"] for c in rbd.to_dict()["components"]}
    assert saved["a"]["inspection"] == {
        "interval": 100.0,
        "duration": "instant",
    }
    assert saved["b"]["inspection"]["coverage"] == 0.8
