"""Validation of the availability simulation against exact Markov results.

For a single component with exponential failure (rate lam) and repair (rate
mu), starting up, the transient availability has the exact closed form

    A(t) = mu/(lam+mu) + lam/(lam+mu) * exp(-(lam+mu) t).

Because every component in the simulation has its own independent repair
process, the exact *system* transient availability of any structure is the
structure function applied to these marginal availabilities (e.g. product for
series, 1 - product of complements for parallel). These tests hold the
simulated curve to the exact solution within Monte-Carlo sampling error.

All tests are seeded, so they are deterministic.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD

E = surv.Exponential.from_params


def _comp(lam, mu):
    return {"reliability": E([lam]), "repairability": E([mu])}


def _exact_marginal(lam, mu, t):
    t = np.asarray(t, dtype=float)
    s = lam + mu
    return mu / s + (lam / s) * np.exp(-s * t)


def _sim_at(result, t):
    return np.interp(t, result.timeline, result.availability)


def _assert_within_sampling_error(sim, exact, n, z=3.5, slack=0.006):
    # Binomial sampling error of the proportion at each point, plus a small
    # slack for interpolating the step-function timeline.
    se = np.sqrt(exact * (1.0 - exact) / n)
    np.testing.assert_array_less(np.abs(sim - exact), z * se + slack)


T_CHECK = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0])

# Shared single-component simulation (lam=0.2, mu=1.0, N=4000): run once,
# used by both the transient-accuracy and the confidence-band tests.
_SINGLE = {"lam": 0.2, "mu": 1.0, "n": 4000}


@pytest.fixture(scope="module")
def single_component_result():
    rbd = RepairableRBD(
        [("s", "A"), ("A", "t")],
        {"A": _comp(_SINGLE["lam"], _SINGLE["mu"])},
    )
    return rbd.availability(t_simulation=20.0, N=_SINGLE["n"], seed=42)


def test_transient_availability_matches_markov_single_component(
    single_component_result,
):
    exact = _exact_marginal(_SINGLE["lam"], _SINGLE["mu"], T_CHECK)
    _assert_within_sampling_error(
        _sim_at(single_component_result, T_CHECK), exact, _SINGLE["n"]
    )


def test_transient_availability_matches_markov_series():
    n = 3000
    rbd = RepairableRBD(
        [("s", "A"), ("A", "B"), ("B", "t")],
        {"A": _comp(0.2, 1.0), "B": _comp(0.4, 0.8)},
    )
    result = rbd.availability(t_simulation=20.0, N=n, seed=7)
    exact = _exact_marginal(0.2, 1.0, T_CHECK) * _exact_marginal(
        0.4, 0.8, T_CHECK
    )
    _assert_within_sampling_error(_sim_at(result, T_CHECK), exact, n)


def test_transient_availability_matches_markov_parallel():
    n = 3000
    rbd = RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {"A": _comp(0.5, 1.0), "B": _comp(0.8, 1.0)},
    )
    result = rbd.availability(t_simulation=20.0, N=n, seed=11)
    exact = 1.0 - (1.0 - _exact_marginal(0.5, 1.0, T_CHECK)) * (
        1.0 - _exact_marginal(0.8, 1.0, T_CHECK)
    )
    _assert_within_sampling_error(_sim_at(result, T_CHECK), exact, n)


# --- Uncertainty quantification ---------------------------------------------


def test_availability_interval_brackets_exact_solution(
    single_component_result,
):
    result = single_component_result
    lower, upper = result.availability_interval(confidence=0.99)
    exact = _exact_marginal(_SINGLE["lam"], _SINGLE["mu"], T_CHECK)
    lower_at = np.interp(T_CHECK, result.timeline, lower)
    upper_at = np.interp(T_CHECK, result.timeline, upper)
    # The 99% band should bracket the exact curve at these points (small
    # interpolation slack).
    assert np.all(lower_at - 0.005 <= exact)
    assert np.all(exact <= upper_at + 0.005)
    # Bands are ordered and within [0, 1].
    assert np.all(lower <= result.availability)
    assert np.all(result.availability <= upper)
    assert np.all((lower >= 0.0) & (upper <= 1.0))


def test_availability_se_and_interval_edge_cases():
    rbd = RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {"A": _comp(0.2, 1.0), "B": _comp(0.5, 1.0)},
    )
    # Forcing A working -> availability is exactly 1 everywhere.
    result = rbd.availability(
        t_simulation=10.0, N=200, working_nodes=["A"], seed=3
    )
    se = result.availability_se
    assert np.all(se == 0.0)  # p = 1 -> binomial SE is 0
    lower, upper = result.availability_interval()
    # The Wilson interval stays finite and sensible at p = 1: the upper bound
    # is 1 and the lower bound is strictly below 1 (unlike the plain normal
    # interval, which collapses to zero width).
    assert np.all(upper == 1.0)
    assert np.all(lower < 1.0)
    assert np.all(lower > 0.9)  # 200 sims all-up is strong evidence
    with pytest.raises(ValueError, match="confidence"):
        result.availability_interval(confidence=1.5)


def test_mttf_interval_contains_analytic_value():
    # For exponential components in series the system lifetime is exponential
    # with rate sum(lambda): MTTF = 1 / 0.7.
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (2, "t")],
        {1: E([0.2]), 2: E([0.5])},
    )
    interval = rbd.mean_time_to_failure_interval(
        mc_samples=20_000, confidence=0.99, seed=5
    )
    analytic = 1.0 / 0.7
    assert interval.lower <= analytic <= interval.upper
    assert interval.lower < interval.estimate < interval.upper
    assert interval.standard_error > 0.0
    assert interval.n_samples == 20_000
    # Consistent with the plain point estimate under the same seed.
    assert interval.estimate == pytest.approx(
        rbd.mean_time_to_failure(20_000, seed=5)
    )
    with pytest.raises(ValueError, match="confidence"):
        rbd.mean_time_to_failure_interval(mc_samples=10, confidence=0.0)
