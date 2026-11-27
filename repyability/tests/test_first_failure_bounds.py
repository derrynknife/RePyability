"""Bounds on the first system failure of a repairable diagram
(``first_failure_bounds``, ``mean_time_to_first_failure_bound``), from its
unavailability and its expected failures.

Checked against the exact first failure on every catalogue diagram that
has one, the renewal function's closed form for one component, and seeded
simulations where the exact values refuse (Weibull lives, tests,
common-cause groups)."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    BetaFactor,
    CCFGroup,
    FirstFailureBounds,
    NodeState,
    RepairableRBD,
)
from repyability.rbd import routes
from repyability.tests.catalogue import repairable_kinds

E = surv.Exponential.from_params
W = surv.Weibull.from_params

PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
ONE = [("s", "a"), ("a", "t")]
TIMES = np.array([0.0, 1.0, 10.0, 50.0, 200.0, 1000.0, 5000.0])
#: The accuracy the two methods state: the expected failures to about 1e-7
#: (of a count of about 1 or less; relative past it), the unavailability to
#: about 4e-7, up to 4e-6 soon after the start.
W_RTOL, Q_ATOL = 1e-7, 4e-6


def unit(life=None, repair=None, **more):
    return {
        "reliability": life or E([0.01]),
        "repairability": repair or E([0.1]),
        **more,
    }


def exact_kinds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kinds = repairable_kinds()
        return sorted(
            name
            for name, rbd in kinds.items()
            if rbd.analysis_routes()["first_failure_sf"].route
            != routes.REFUSED
        )


@pytest.mark.parametrize("name", exact_kinds())
def test_the_exact_first_failure_lies_in_the_bracket(name):
    rbd = repairable_kinds()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bounds = rbd.first_failure_bounds(TIMES)
        exact = rbd.first_failure_sf(TIMES)
        mean = rbd.mean_time_to_first_failure()
        mean_bound = rbd.mean_time_to_first_failure_bound()
    slack = W_RTOL * np.maximum(bounds.expected_failures, 1.0)
    assert np.all(bounds.lower <= exact + slack)
    assert np.all(exact <= bounds.upper + Q_ATOL)
    assert mean_bound <= mean * (1.0 + 1e-6)


def test_one_component_is_bounded_by_its_renewal_function():
    # An exponential life (rate lam) and repair (rate mu): M(t) =
    # lam mu t / s + lam^2 / s^2 (1 - exp(-s t)), s = lam + mu; Q(t) =
    # lam / s (1 - exp(-s t)); the exact reliability exp(-lam t).
    lam, mu = 0.01, 0.1
    s = lam + mu
    rbd = RepairableRBD(ONE, {"a": unit(E([lam]), E([mu]))})
    bounds = rbd.first_failure_bounds(TIMES)
    renewals = lam * mu * TIMES / s + lam**2 / s**2 * -np.expm1(-s * TIMES)
    down = lam / s * -np.expm1(-s * TIMES)
    np.testing.assert_allclose(
        bounds.expected_failures, renewals, rtol=W_RTOL, atol=W_RTOL
    )
    np.testing.assert_allclose(
        bounds.lower, np.clip(1.0 - renewals, 0.0, 1.0), atol=W_RTOL
    )
    np.testing.assert_allclose(bounds.upper, 1.0 - down, atol=1e-12)
    exact = np.exp(-lam * TIMES)
    assert np.all(bounds.lower <= exact + W_RTOL)
    assert np.all(exact <= bounds.upper)
    # The mean's bound: the integral of 1 - M(t) to where M reaches 1.
    end = _root(
        lambda t: lam * mu * t / s + lam**2 / s**2 * -math.expm1(-s * t) - 1.0
    )
    integral = (
        end
        - lam * mu * end**2 / (2 * s)
        - lam**2 / s**2 * (end + math.expm1(-s * end) / s)
    )
    assert rbd.mean_time_to_first_failure_bound() == pytest.approx(
        integral, rel=1e-6
    )


def _root(f, low=0.0, high=1e6):
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if f(middle) < 0.0 else (low, middle)
    return 0.5 * (low + high)


def within(bounds, simulated, samples):
    """The simulated reliability in the bracket, within four standard
    errors."""
    p = np.clip(simulated, 1.0 / samples, 1.0 - 1.0 / samples)
    error = 4.0 * np.sqrt(p * (1.0 - p) / samples)
    return bool(
        np.all(bounds.lower <= simulated + error)
        and np.all(simulated <= bounds.upper + error)
    )


def test_a_weibull_pair_brackets_its_simulated_first_failure():
    rbd = RepairableRBD(PAIR, {n: unit(W([1000.0, 2.0])) for n in "ab"})
    with pytest.raises(NotImplementedError, match="first_failure_bounds"):
        rbd.first_failure_sf(100.0)
    x = np.array([200.0, 1000.0, 3000.0])
    bounds = rbd.first_failure_bounds(x)
    simulated = rbd.first_failure_sf(
        x, method="simulate", mc_samples=20_000, seed=11
    )
    assert within(bounds, simulated, 20_000)
    # Tight while the expected failures are small.
    assert bounds.upper[0] - bounds.lower[0] < 1e-3
    assert rbd.mean_time_to_first_failure_bound() > 0.0


def test_from_a_state():
    rbd = RepairableRBD(PAIR, {n: unit() for n in "ab"})
    state = {"a": NodeState(alive=False)}
    bounds = rbd.first_failure_bounds(TIMES, state=state)
    exact = rbd.first_failure_sf(TIMES, state=state)
    assert np.all(bounds.lower <= exact + W_RTOL)
    assert np.all(exact <= bounds.upper + Q_ATOL)
    assert rbd.mean_time_to_first_failure_bound(
        state=state
    ) <= rbd.mean_time_to_first_failure(state=state)
    # Worse than from new.
    assert (
        rbd.mean_time_to_first_failure_bound(state=state)
        < rbd.mean_time_to_first_failure_bound()
    )
    # Down at the start: failed already.
    both = {"a": NodeState(alive=False), "b": NodeState(alive=False)}
    down = rbd.first_failure_bounds(TIMES, state=both)
    assert down.lower.tolist() == [0.0] * TIMES.size
    assert down.upper.tolist() == [0.0] * TIMES.size
    assert rbd.mean_time_to_first_failure_bound(state=both) == 0.0


def test_started_stationary():
    rbd = RepairableRBD(PAIR, {n: unit() for n in "ab"})
    bounds = rbd.first_failure_bounds(TIMES, state="stationary")
    exact = rbd.first_failure_sf(TIMES, state="stationary")
    assert np.all(bounds.lower <= exact + W_RTOL)
    assert np.all(exact <= bounds.upper + Q_ATOL)


def test_tested_units_bracket_their_simulated_first_failure():
    # Hidden failures found by tests every 100 hours, repaired at once.
    tested = unit(W([1000.0, 1.5]), "instant", inspection={"interval": 100.0})
    rbd = RepairableRBD(PAIR, {n: tested for n in "ab"})
    x = np.array([150.0, 500.0, 1500.0])
    bounds = rbd.first_failure_bounds(x)
    assert not bounds.planned_outages
    simulated = rbd.first_failure_sf(
        x, method="simulate", mc_samples=10_000, seed=12
    )
    assert within(bounds, simulated, 10_000)


def test_a_common_cause_group_brackets_its_simulated_first_failure():
    rbd = RepairableRBD(
        PAIR,
        {n: unit(E([0.002]), E([0.5])) for n in "ab"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="first_failure_bounds"):
        rbd.first_failure_sf(100.0)
    x = np.array([100.0, 1000.0, 3000.0])
    bounds = rbd.first_failure_bounds(x)
    simulated = rbd.first_failure_sf(
        x, method="simulate", mc_samples=10_000, seed=13
    )
    assert within(bounds, simulated, 10_000)


def test_planned_outages_leave_no_upper_bound():
    # A test that takes time takes the system down without a failure.
    timed = unit(
        W([1000.0, 1.5]),
        "instant",
        inspection={"interval": 100.0, "duration": E([2.0])},
    )
    rbd = RepairableRBD(ONE, {"a": timed})
    bounds = rbd.first_failure_bounds([50.0, 500.0])
    assert bounds.planned_outages
    assert bounds.upper.tolist() == [1.0, 1.0]
    assert np.all(bounds.unavailability > 0.0)
    assert np.all(bounds.lower < 1.0)


def test_the_upper_bound_never_rises():
    # The unavailability of an exponential pair overshoots nothing, but a
    # Weibull unit's rises and falls: the bound keeps the highest so far.
    rbd = RepairableRBD(ONE, {"a": unit(W([100.0, 5.0]), E([0.05]))})
    x = np.linspace(0.0, 400.0, 81)
    bounds = rbd.first_failure_bounds(x[::-1])
    upper = bounds.upper[::-1]
    assert np.all(np.diff(upper) <= 0.0)
    assert np.all(upper <= 1.0 - bounds.unavailability[::-1] + 1e-15)


def test_where_the_expected_events_refuse_so_do_the_bounds():
    for name, rbd in repairable_kinds().items():
        report = rbd.analysis_routes()
        if report["expected_events"].route == routes.REFUSED:
            break
    else:
        pytest.skip("no catalogue diagram refuses the expected events")
    with pytest.raises((NotImplementedError, ValueError)) as error:
        rbd.expected_events(10.0)
    with pytest.raises(type(error.value)) as again:
        rbd.first_failure_bounds(10.0)
    assert str(again.value) == str(error.value), name


def test_a_scalar_time_and_json():
    rbd = RepairableRBD(PAIR, {n: unit() for n in "ab"})
    bounds = rbd.first_failure_bounds(100.0)
    assert isinstance(bounds, FirstFailureBounds)
    assert isinstance(bounds.lower, float) and isinstance(bounds.upper, float)
    out = json.loads(json.dumps(bounds.to_dict()))
    assert set(out) == {
        "time",
        "lower",
        "upper",
        "expected_failures",
        "unavailability",
        "planned_outages",
    }
    assert rbd.first_failure_bounds([[10.0, 20.0]]).lower.shape == (1, 2)
    with pytest.raises(ValueError, match="non-negative"):
        rbd.first_failure_bounds(-1.0)


def test_a_system_that_never_fails_has_an_infinite_mean_bound():
    rbd = RepairableRBD(
        PAIR,
        {n: unit(repair=surv.ExactEventTime.from_params([0.0])) for n in "ab"},
    )
    # Instant repairs of a pair: the system never goes down.
    assert rbd.mean_time_to_first_failure_bound() == math.inf
