"""
Tests the imperfect-repair (generalized-renewal / Kijima) support in the
``Repairable`` component: sourcing E[N(t)] from a simulation-backed
``GeneralizedRenewal`` model rather than an analytic ``cif``.

The exact anchors use the ``q = 1`` (minimal-repair) limit, where the
generalized-renewal process reduces to the baseline's cumulative hazard and the
optimal overhaul interval has a closed form.
"""

import warnings

import numpy as np
import pytest
import surpyval as surv
from surpyval.recurrent import CrowAMSAA, GeneralizedRenewal

from repyability.maintenance import FailureLimitPolicy
from repyability.repairable import (
    Repairable,
    minimal_repair_time_to_nth_failure,
)


def _gr(q, alpha=100.0, beta=2.0, kijima="ii"):
    """A generalized-renewal model, Weibull baseline, restoration factor q."""
    return GeneralizedRenewal.fit_from_parameters(
        [alpha, beta], q, kijima=kijima, dist=surv.Weibull
    )


def test_is_simulated_flag():
    # A generalized-renewal (mcf-only) model is simulation-backed...
    assert Repairable(_gr(0.5)).is_simulated is True
    # ...while an analytic cumulative-intensity model (cif) is not.
    analytic = Repairable(CrowAMSAA.from_params([0.5, 1.6]))
    assert analytic.is_simulated is False


def test_requires_cif_or_mcf():
    class NoCounting:
        pass

    with pytest.raises(ValueError, match="cif|mcf"):
        Repairable(NoCounting())


def test_q1_expected_failures_matches_cumulative_hazard():
    """At q=1 the process is minimal repair, so E[N(t)] equals the baseline
    Weibull cumulative hazard ``(t/alpha)**beta``."""
    rep = Repairable(_gr(1.0, alpha=100.0, beta=2.0))
    t = np.array([50.0, 100.0, 150.0])
    enf = rep._expected_failures(t, seed=3, n_simulations=8000)
    expected = (t / 100.0) ** 2.0
    np.testing.assert_allclose(enf, expected, rtol=0.05)


def test_cost_and_cost_rate_contract():
    rep = Repairable(_gr(0.5))
    rep.set_repair_and_overhaul_costs(1.0, 5.0)
    assert isinstance(rep.cost(100.0, seed=1, mc_samples=300), float)
    assert isinstance(rep.cost_rate(100.0, seed=1, mc_samples=300), float)
    arr = rep.cost_rate(np.array([100.0, 200.0]), seed=1, mc_samples=300)
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)


def test_optimal_interval_matches_analytic_at_q1():
    """For minimal repair (q=1) with a Weibull baseline the optimal overhaul
    interval has the closed form ``alpha * (co/(cr*(beta-1))) ** (1/beta)``;
    the simulated optimum must land near it."""
    alpha, beta, cr, co = 100.0, 2.0, 1.0, 5.0
    analytic = alpha * (co / (cr * (beta - 1.0))) ** (1.0 / beta)
    rep = Repairable(_gr(1.0, alpha=alpha, beta=beta))
    rep.set_repair_and_overhaul_costs(cr, co)
    interval = rep.find_optimal_overhaul_interval(
        seed=2, mc_samples=1200, max_interval=600.0
    )
    assert interval == pytest.approx(analytic, rel=0.12)


def test_default_horizon_finds_the_minimal_repair_optimum():
    """Near-minimal repair drives the simulated virtual age to where the
    baseline survival underflows; surpyval then ends those histories early and
    the simulated E[N(t)] stops growing. The search must not take that for a
    falling cost rate: it used to return its default horizon (1329) here."""
    alpha, beta, cr, co = 100.0, 2.0, 10.0, 50.0
    analytic = alpha * (co / (cr * (beta - 1.0))) ** (1.0 / beta)  # 223.6
    rep = Repairable(_gr(1.0, alpha=alpha, beta=beta, kijima="i"))
    rep.set_repair_and_overhaul_costs(cr, co)
    with warnings.catch_warnings():
        # Nothing may surface: neither surpyval's warnings about the stalled
        # attempts the search discarded, nor a "still falling" warning.
        warnings.simplefilter("error")
        policy = rep.optimal_overhaul_policy(seed=0, mc_samples=400)
    assert policy.interval == pytest.approx(analytic, rel=0.12)

    def exact_rate(t):
        return (cr * (t / alpha) ** beta + co) / t

    # The cost rate is flat near its minimum: the interval found costs
    # within 1% of the true optimum.
    assert exact_rate(policy.interval) == pytest.approx(
        exact_rate(analytic), rel=0.01
    )


class _CutShort:
    """A simulation-backed stand-in whose E[N(t)] is minimal repair of a
    Weibull(100, 2), but whose "simulator" is cut short beyond ``limit``
    (warning as surpyval does), so E[N(t)] stops growing there."""

    def __init__(self, limit, message):
        self.limit = limit
        self.message = message

    def mcf(self, t, items=1000, random_state=None):
        t = np.asarray(t, dtype=float)
        if t.max() > self.limit:
            warnings.warn(self.message)
        return (np.minimum(t, self.limit) / 100.0) ** 2.0


@pytest.mark.parametrize(
    "message",
    [
        "Some sequences produced a near-zero interarrival time (< tol) "
        "before reaching T, indicating a possible asymptote; they were "
        "terminated early at their last event.",
        "Some sequences reached max_events (10000) before T; increase "
        "max_events or check the model parameters.",
    ],
    ids=["stalled", "max events"],
)
def test_a_simulation_cut_short_shortens_the_search(message):
    """Beyond 600 the stand-in's E[N(t)] is flat, so over a horizon of 2000
    the cost rate would fall to the horizon. The search halves the horizon
    (2000, 1000, 500) until the simulation is no longer cut short, finds the
    true optimum, and drops the warnings of the attempts it discarded."""
    rep = Repairable(_CutShort(600.0, message))
    rep.set_repair_and_overhaul_costs(10.0, 50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        interval = rep.find_optimal_overhaul_interval(max_interval=2000.0)
    # 100 * sqrt(5), to the spacing of the search grid.
    assert interval == pytest.approx(100.0 * 5.0**0.5, rel=0.012)


def test_an_optimum_beyond_max_interval_warns():
    """A cost rate still falling at the horizon is not an optimum: the
    horizon is returned with a warning to search further."""
    rep = Repairable(_CutShort(np.inf, "unused"))
    rep.set_repair_and_overhaul_costs(10.0, 50_000.0)  # optimum 7071
    with pytest.warns(UserWarning, match="Raise max_interval"):
        interval = rep.find_optimal_overhaul_interval(max_interval=2000.0)
    assert interval == 2000.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        interval = rep.find_optimal_overhaul_interval(max_interval=20_000.0)
    assert interval == pytest.approx(100.0 * 5000.0**0.5, rel=0.012)


def test_an_optimum_beyond_the_horizon_warns():
    """With overhauls 1000 times dearer than repairs, the minimal-repair
    optimum (3162) lies beyond the default horizon (15 times the mean,
    1329.34). The search returns at most the horizon and warns, once, that
    the cost rate is still falling there, rather than returning a false
    optimum. A simulator that cannot resolve failures that far makes it
    shorten the horizon first, to where the baseline survival is 1e-10, and
    say so."""
    rep = Repairable(_gr(1.0, kijima="i"))
    rep.set_repair_and_overhaul_costs(1.0, 1000.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        interval = rep.find_optimal_overhaul_interval(seed=0, mc_samples=200)
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert "still falling at the search horizon" in messages[0]
    if "shortened from 1329.34" in messages[0]:
        assert interval == pytest.approx(100.0 * np.log(1e10) ** 0.5)
    else:
        assert "Raise max_interval" in messages[0]
        assert interval == pytest.approx(15.0 * 100.0 * np.sqrt(np.pi) / 2)


def test_stalls_the_search_cannot_avoid_are_reported():
    """When the simulation is cut short at every horizon the search tries
    (as it can be for q > 1, where the virtual age outruns the real age),
    the simulator's warning must reach the caller, not be
    swallowed with those of the discarded attempts."""
    rep = Repairable(
        _CutShort(
            1e-9,
            "Some sequences produced a near-zero interarrival time (< tol) "
            "before reaching T; they were ended early at their last event.",
        )
    )
    rep.set_repair_and_overhaul_costs(10.0, 50.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep.find_optimal_overhaul_interval(max_interval=2000.0)
    messages = [str(w.message) for w in caught]
    assert any("near-zero interarrival time" in m for m in messages)


def test_repairs_that_age_the_unit_bring_the_overhaul_forward():
    """With q > 1 each repair leaves the unit older than before the
    failure, so it wears out faster than under minimal repair and should be
    overhauled sooner than minimal repair's optimum, 100 * sqrt(5)."""
    rep = Repairable(_gr(1.5, kijima="i"))
    rep.set_repair_and_overhaul_costs(10.0, 50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the simulator may report stalls
        interval = rep.find_optimal_overhaul_interval(seed=0, mc_samples=100)
    assert 100.0 < interval < 100.0 * 5.0**0.5


def test_reproducible_with_seed():
    # Kijima I (the optimum, near 314, lies inside the horizon).
    rep = Repairable(_gr(0.5, kijima="i"))
    rep.set_repair_and_overhaul_costs(1.0, 5.0)
    p1 = rep.optimal_overhaul_policy(
        seed=7, mc_samples=400, max_interval=600.0
    )
    p2 = rep.optimal_overhaul_policy(
        seed=7, mc_samples=400, max_interval=600.0
    )
    assert p1.interval == p2.interval
    assert p1.cost_rate == p2.cost_rate


def test_costs_required_and_ordered():
    rep = Repairable(_gr(0.5))
    with pytest.raises(ValueError, match="costs not set"):
        rep.find_optimal_overhaul_interval(seed=1, mc_samples=200)
    with pytest.raises(ValueError, match="less than"):
        rep.set_repair_and_overhaul_costs(5.0, 1.0)


# -- Replace-at-N-th-failure policy ---------------------------------------


def test_minimal_repair_time_to_nth_failure_closed_form():
    # T_1 is just the baseline mean: alpha * Gamma(1 + 1/beta).
    from scipy.special import gamma

    assert minimal_repair_time_to_nth_failure(100.0, 2.0, 1) == pytest.approx(
        100.0 * gamma(1.5)
    )
    # A monotone increasing sequence.
    ts = [
        minimal_repair_time_to_nth_failure(100.0, 2.0, n) for n in range(1, 6)
    ]
    assert all(a < b for a, b in zip(ts, ts[1:]))
    with pytest.raises(ValueError):
        minimal_repair_time_to_nth_failure(100.0, 2.0, 0)


def test_expected_time_to_nth_failure_matches_closed_form_at_q1():
    rep = Repairable(_gr(1.0, alpha=100.0, beta=2.0))
    for n in (3, 6):
        sim = rep.expected_time_to_nth_failure(n, seed=1, mc_samples=5000)
        exact = minimal_repair_time_to_nth_failure(100.0, 2.0, n)
        assert sim == pytest.approx(exact, rel=0.05)


def test_expected_time_to_nth_failure_analytic_raises():
    analytic = Repairable(CrowAMSAA.from_params([0.5, 1.6]))
    with pytest.raises(ValueError, match="minimal_repair_time_to_nth_failure"):
        analytic.expected_time_to_nth_failure(3)


def test_expected_time_to_nth_failure_validation():
    rep = Repairable(_gr(0.5))
    with pytest.raises(ValueError, match="positive"):
        rep.expected_time_to_nth_failure(0, seed=1, mc_samples=200)


def test_optimal_failure_limit_policy():
    rep = Repairable(_gr(0.4, kijima="i"))
    rep.set_repair_and_overhaul_costs(1.0, 5.0)
    policy = rep.optimal_failure_limit_policy(
        seed=3, mc_samples=1200, max_failures=25
    )
    assert isinstance(policy, FailureLimitPolicy)
    assert policy.failure_count >= 1
    assert policy.cost_rate > 0
    # Reproducible for a fixed seed.
    again = rep.optimal_failure_limit_policy(
        seed=3, mc_samples=1200, max_failures=25
    )
    assert policy == again


def test_failure_limit_costs_required():
    rep = Repairable(_gr(0.5))
    with pytest.raises(ValueError, match="costs not set"):
        rep.find_optimal_replacement_failure_count(seed=1, mc_samples=200)


def test_the_simulations_seed_goes_by_random_state():
    """surpyval 0.21 renamed the simulations' ``seed`` to ``random_state``,
    and 0.22 removed ``seed``: the seed goes by the new name, so nothing
    warns, and it reaches the model."""
    rep = Repairable(_gr(0.5))
    rep.set_repair_and_overhaul_costs(1.0, 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rep.cost([50.0, 150.0], seed=3, mc_samples=100)
        rep.expected_time_to_nth_failure(2, seed=3, mc_samples=100)

    class Recording:
        """A model that records the random_state its simulations get."""

        def __init__(self):
            self.seen = []

        def mcf(self, x, items=1000, random_state=None):
            self.seen.append(("mcf", random_state))
            return np.asarray(x, dtype=float) / 100.0

        def count_terminated_simulation(
            self, events, items=1000, random_state=None
        ):
            self.seen.append(("count", random_state))
            return GeneralizedRenewal.fit_from_parameters(
                [100.0, 2.0], 0.5, kijima="ii", dist=surv.Weibull
            ).count_terminated_simulation(events, items=items)

    model = Recording()
    rep = Repairable(model)
    rep.set_repair_and_overhaul_costs(1.0, 10.0)
    rep.cost([50.0], seed=7, mc_samples=50)
    rep.expected_time_to_nth_failure(1, seed=7, mc_samples=50)
    assert ("mcf", 7) in model.seen and ("count", 7) in model.seen
