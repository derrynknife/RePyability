"""Tests for the Monte-Carlo options: simulating to a tolerance, antithetic
pairs, parallel runs and comparisons with common random numbers.

The references are exact values (the transient availability of independent
exponential components, which the structure function combines, and its
mean over the window; a system's MTTF, the integral of its reliability),
runs of the same size without the option, and properties the pairing and
the common random numbers guarantee sample by sample.
"""

import math
import os
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from scipy.stats import norm

from repyability import NonRepairableRBD, RepairableRBD, StandbyModel
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _streams, non_repairable_rbd
from repyability.rbd.non_repairable_rbd import _check_lifetimes

E = surv.Exponential.from_params
W = surv.Weibull.from_params
T = 50.0


def marginal(lam, mu, t):
    """An exponential component's availability at ``t``, starting up."""
    s = lam + mu
    return mu / s + lam / s * np.exp(-s * t)


def pump(mttr, cost=None):
    spec = {"reliability": E([0.1]), "repairability": E([1 / mttr])}
    if cost is not None:
        spec["repair_cost"] = cost
    return spec


def plant(mttr=1.0, cost=None):
    """Two pumps in parallel, then a valve."""
    valve = {"reliability": E([0.02]), "repairability": E([0.5])}
    if cost is not None:
        valve["repair_cost"] = cost
    return RepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")],
        {"p1": pump(mttr, cost), "p2": pump(mttr, cost), "v": valve},
    )


def plant_availability(mttr, t):
    pumps = 1 - (1 - marginal(0.1, 1 / mttr, t)) ** 2
    return pumps * marginal(0.02, 0.5, t)


def window_mean(f, window=T):
    return quad(f, 0, window)[0] / window


def half_width(values, antithetic=False, confidence=0.95):
    """The normal interval's half-width for the mean (of pairs' means)."""
    values = np.asarray(values, dtype=float)
    if antithetic:
        values = (values[0::2] + values[1::2]) / 2
    z = norm.ppf(0.5 + confidence / 2)
    return z * np.std(values, ddof=1) / np.sqrt(len(values))


def parallel(n, unit=W([100, 2])):
    names = [f"u{i}" for i in range(n)]
    edges = [("s", u) for u in names] + [(u, "t") for u in names]
    return NonRepairableRBD(edges, {u: unit for u in names})


def mttf(rbd):
    return quad(rbd.sf, 0, np.inf)[0]


def standby_pair():
    """A repairable unit whose life is a cold-standby pair: its draws cannot
    be replayed from uniforms of its own."""
    return {
        "reliability": StandbyModel([W([10, 2]), W([10, 2])]),
        "repairability": E([1.0]),
    }


class Drawn:
    """An exponential lifetime model that samples its own way, so its draws
    cannot be replayed from uniforms: ``random`` steps through events."""

    def __init__(self, scale):
        self.scale = scale

    def sf(self, x):
        return np.exp(-np.asarray(x, dtype=float) / self.scale)

    def ff(self, x):
        return 1 - self.sf(x)

    def random(self, size):
        return np.random.exponential(self.scale, size)


# -- simulating to a tolerance ----------------------------------------------


def test_availability_to_a_tolerance_is_a_run_of_its_size():
    rbd = plant()
    # Its own twin, it would take its exact values (#187): simulate.
    result = rbd.availability(
        T, mc_samples=200, seed=1, tolerance=0.004, control_variate=False
    )
    n = result.n_simulations
    assert n > 200 and n % 200 == 0
    fractions = result.uptimes / T
    assert half_width(fractions) <= 0.004
    # It stopped at the first check that passed.
    assert half_width(fractions[: n - 200]) > 0.004
    fixed = rbd.availability(T, mc_samples=n, seed=1)
    np.testing.assert_array_equal(fixed.uptimes, result.uptimes)
    np.testing.assert_array_equal(fixed.timeline, result.timeline)
    np.testing.assert_array_equal(fixed.availability, result.availability)
    assert fixed.node_uptime == result.node_uptime
    assert fixed.system_failures == result.system_failures
    interval = result.mean_availability_interval()
    assert interval.upper - interval.estimate <= 0.004
    assert interval.estimate == pytest.approx(
        result.system_uptime / (n * T), rel=1e-12
    )


def test_cost_to_a_tolerance():
    rbd = plant(cost=100.0)
    result = rbd.cost(
        T, mc_samples=100, seed=2, tolerance=15.0, control_variate=False
    )
    n = result.n_simulations
    assert n > 100 and n % 100 == 0
    assert half_width(result.samples) <= 15.0
    assert half_width(result.samples[: n - 100]) > 15.0
    fixed = rbd.cost(T, mc_samples=n, seed=2)
    np.testing.assert_array_equal(fixed.samples, result.samples)
    interval = result.mean_interval()
    assert interval.upper - interval.estimate <= 15.0


def test_a_run_that_does_not_converge_warns():
    rbd = plant()
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = rbd.availability(
            T,
            mc_samples=100,
            seed=3,
            tolerance=1e-6,
            max_samples=300,
            control_variate=False,
        )
    assert result.n_simulations == 300
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rbd.availability(
            T, mc_samples=100, seed=3, tolerance=0.5, control_variate=False
        )


@pytest.mark.parametrize(
    "options",
    [
        {"mc_samples": 0},
        {"mc_samples": 2.0},
        {"mc_samples": True},
        {"tolerance": 0.0},
        {"tolerance": -1.0},
        {"tolerance": math.inf},
        {"tolerance": "0.1"},
        {"tolerance": True},
        {"max_samples": 100},
        {"tolerance": 0.1, "max_samples": 50},
        {"tolerance": 0.1, "max_samples": 150.0},
        {"tolerance": 0.1, "confidence": 1.0},
        {"tolerance": 0.1, "confidence": 0.0},
        {"antithetic": True, "mc_samples": 101},
        {"antithetic": True, "tolerance": 0.1, "max_samples": 201},
        {"n_jobs": 0},
        {"n_jobs": -2},
        {"n_jobs": 1.5},
        {"n_jobs": True},
    ],
)
def test_invalid_options(options):
    options = {"mc_samples": 100, **options}
    with pytest.raises(ValueError):
        plant().availability(T, seed=0, **options)


# -- antithetic pairs ---------------------------------------------------------


def identity(u):
    return u


def uniform_streams(antithetic: bool) -> _streams.Run:
    """Streams of plain uniforms for components "a" and "b", in blocks of
    four simulations (or pairs) and chunks of eight rows."""
    specs = {
        ((node,), _streams.FAILURE): _streams.Spec(
            (node,), _streams.FAILURE, identity, 8, 4
        )
        for node in "ab"
    }
    return _streams.Run(_streams.Plan(7, antithetic, specs), reseed=False)


def test_each_component_pairs_its_own_draws():
    run = uniform_streams(antithetic=True)
    a = run.stream(("a",), _streams.FAILURE)
    b = run.stream(("b",), _streams.FAILURE)
    run.begin(0)
    first_a = [a.draw() for _ in range(70)]  # past several chunks
    first_b = [b.draw() for _ in range(3)]
    # The second of the pair: in another order, and b draws more, but
    # each component's k-th draw is one minus its k-th in the first.
    run.begin(1)
    second_b = [b.draw() for _ in range(5)]
    second_a = [a.draw() for _ in range(70)]
    np.testing.assert_allclose(np.add(first_a, second_a), 1.0, atol=1e-15)
    np.testing.assert_allclose(np.add(first_b, second_b[:3]), 1.0, atol=1e-15)
    assert len(set(first_a)) == 70 and set(first_a).isdisjoint(first_b)
    # The next pair draws afresh, in the same block and in the next.
    for replication in (2, 8):
        run.begin(replication)
        fresh = a.draw()
        assert all(abs(fresh - u) > 1e-12 for u in first_a + second_a)
    # Without pairs, every replication draws afresh: the first one what
    # the first pair did.
    plain = uniform_streams(antithetic=False)
    c = plain.stream(("a",), _streams.FAILURE)
    plain.begin(0)
    assert [c.draw() for _ in range(70)] == first_a
    plain.begin(1)
    assert abs(c.draw() - (1 - first_a[0])) > 1e-12


def test_antithetic_runs_are_reproducible():
    rbd = plant()
    seeded = rbd.availability(T, mc_samples=20, seed=5, antithetic=True)
    again = rbd.availability(T, mc_samples=20, seed=5, antithetic=True)
    np.testing.assert_array_equal(seeded.uptimes, again.uptimes)
    # Without a seed, from the global RNG as it stands: seed=5 seeds it.
    np.random.seed(5)
    unseeded = rbd.availability(T, mc_samples=20, antithetic=True)
    np.testing.assert_array_equal(unseeded.uptimes, seeded.uptimes)
    np.random.seed(6)
    other = rbd.availability(T, mc_samples=20, antithetic=True)
    assert not np.array_equal(other.uptimes, seeded.uptimes)


def test_antithetic_availability_is_unbiased_and_tighter():
    rbd = plant()
    exact = window_mean(lambda t: plant_availability(1.0, t))
    paired = rbd.availability(T, mc_samples=4000, seed=3, antithetic=True)
    interval = paired.mean_availability_interval()
    assert paired.antithetic
    assert abs(interval.estimate - exact) < 4 * interval.standard_error
    # The standard error is the pairs' means'.
    fractions = paired.uptimes / T
    pairs = (fractions[0::2] + fractions[1::2]) / 2
    assert interval.standard_error == pytest.approx(
        np.std(pairs, ddof=1) / np.sqrt(2000), rel=1e-12
    )
    independent = rbd.availability(T, mc_samples=4000, seed=3)
    assert not independent.antithetic
    assert (
        interval.standard_error
        < independent.mean_availability_interval().standard_error
    )


def test_antithetic_cost_is_unbiased_and_tighter():
    rbd = plant(cost=100.0)
    # A failure is charged 100; a unit fails at rate lambda while up.
    exact = (
        100.0
        * T
        * (
            2 * 0.1 * window_mean(lambda t: marginal(0.1, 1.0, t))
            + 0.02 * window_mean(lambda t: marginal(0.02, 0.5, t))
        )
    )
    paired = rbd.cost(T, mc_samples=4000, seed=4, antithetic=True)
    assert paired.antithetic
    assert abs(paired.mean - exact) < 4 * paired.mean_se
    pairs = (paired.samples[0::2] + paired.samples[1::2]) / 2
    assert paired.mean_se == pytest.approx(
        np.std(pairs, ddof=1) / np.sqrt(2000), rel=1e-12
    )
    independent = rbd.cost(T, mc_samples=4000, seed=4)
    assert paired.mean_se < 0.8 * independent.mean_se


def test_antithetic_needs_replayable_draws():
    rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": standby_pair()})
    with pytest.raises(NotImplementedError):
        rbd.availability(T, mc_samples=10, seed=0, antithetic=True)


# -- parallel runs ----------------------------------------------------------


def assert_same_results(a, b):
    np.testing.assert_array_equal(a.uptimes, b.uptimes)
    np.testing.assert_array_equal(a.timeline, b.timeline)
    np.testing.assert_array_equal(a.availability, b.availability)
    assert a.node_uptime == b.node_uptime
    assert a.system_failures == b.system_failures
    assert a.system_restorations == b.system_restorations
    if a.cost is not None:
        np.testing.assert_array_equal(a.cost.samples, b.cost.samples)
        assert a.cost.by_component == b.cost.by_component


def test_parallel_results_do_not_depend_on_the_processes():
    rbd = plant(cost=100.0)
    one = rbd.availability(T, mc_samples=600, seed=4, n_jobs=1)
    assert one.n_simulations == 600
    assert_same_results(
        one, rbd.availability(T, mc_samples=600, seed=4, n_jobs=2)
    )
    assert_same_results(
        one, rbd.availability(T, mc_samples=600, seed=4, n_jobs=-1)
    )
    exact = window_mean(lambda t: plant_availability(1.0, t))
    interval = one.mean_availability_interval()
    assert abs(interval.estimate - exact) < 4 * interval.standard_error
    # Nor on whether it runs in processes at all: each simulation is the
    # same however the run is cut up, so the first 250 are a run of 250.
    assert_same_results(one, rbd.availability(T, mc_samples=600, seed=4))
    first = rbd.availability(T, mc_samples=250, seed=4, n_jobs=2)
    np.testing.assert_array_equal(one.uptimes[:250], first.uptimes)
    assert not np.array_equal(one.uptimes[:250], one.uptimes[250:500])


def test_parallel_runs_to_a_tolerance_and_in_pairs():
    rbd = plant()
    runs = [
        rbd.availability(
            T,
            mc_samples=300,
            seed=5,
            tolerance=0.004,
            antithetic=True,
            n_jobs=jobs,
            control_variate=False,
        )
        for jobs in (1, 3)
    ]
    assert_same_results(*runs)
    assert runs[0].antithetic and runs[0].n_simulations > 300
    fractions = runs[0].uptimes / T
    assert half_width(fractions, antithetic=True) <= 0.004


# -- common random numbers ----------------------------------------------------


def test_a_system_compared_with_itself():
    rbd = plant(cost=100.0)
    for quantity in ("availability", "cost"):
        same = rbd.compare(rbd, T, mc_samples=200, seed=6, quantity=quantity)
        assert same.estimate == 0.0 and same.standard_error == 0.0


def test_compare_against_the_exact_difference():
    exact = window_mean(
        lambda t: plant_availability(1.0, t) - plant_availability(2.0, t)
    )
    gain = plant(1.0).compare(plant(2.0), T, mc_samples=2000, seed=5)
    assert abs(gain.estimate - exact) < 4 * gain.standard_error
    assert gain.lower < gain.estimate < gain.upper
    assert gain.n_samples == 2000
    # Two independent runs of the same size are far less precise.
    a = plant(1.0).availability(T, mc_samples=2000, seed=5)
    b = plant(2.0).availability(T, mc_samples=2000, seed=6)
    independent = math.hypot(
        a.mean_availability_interval().standard_error,
        b.mean_availability_interval().standard_error,
    )
    assert gain.standard_error < independent / 2


def test_compare_costs_against_the_exact_difference():
    def expected(mttr):
        return (
            100.0
            * T
            * 2
            * 0.1
            * window_mean(lambda t: marginal(0.1, 1 / mttr, t))
        )

    exact = expected(1.0) - expected(2.0)
    gain = plant(1.0, cost=100.0).compare(
        plant(2.0, cost=100.0), T, mc_samples=2000, seed=7, quantity="cost"
    )
    assert abs(gain.estimate - exact) < 4 * gain.standard_error


def keyed_uptimes(rbd, n, key, widths):
    tally = rbd._run(
        T,
        set(),
        set(),
        "p",
        n,
        False,
        None,
        entropy=key,
        widths=widths,
        common=True,
    )
    return np.asarray(tally.uptimes)


def common_widths(*rbds):
    """The narrowest width each stream has in any of ``rbds``, as
    ``compare`` gives two systems."""
    widths: dict = {}
    for rbd in rbds:
        for name, spec in rbd._stream_specs(T)[0].items():
            widths[name] = min(widths.get(name, spec.width), spec.width)
    return widths


def test_the_same_component_fails_and_is_repaired_alike():
    # A component in the same place draws the same numbers in both
    # systems: a spare pump never leaves the system up for less time, in
    # any simulation, and a faster repair of the same failures never
    # leaves the one pump up for less time either.
    one = RepairableRBD([("s", "p1"), ("p1", "t")], {"p1": pump(1.0)})
    two = RepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        {"p1": pump(1.0), "p2": pump(1.0)},
    )
    slow = RepairableRBD([("s", "p1"), ("p1", "t")], {"p1": pump(3.0)})
    widths = common_widths(one, two, slow)
    for key in (1, 2, 3):
        up = {
            name: keyed_uptimes(rbd, 300, key, widths)
            for name, rbd in (("one", one), ("two", two), ("slow", slow))
        }
        spare = up["two"] - up["one"]
        assert spare.min() >= -1e-9 and spare.max() > 0
        faster = up["one"] - up["slow"]
        assert faster.min() >= -1e-9 and faster.max() > 0
    # With independent draws some simulations would go the other way.
    np.random.seed(0)
    two_alone = two.availability(T, mc_samples=300, seed=1).uptimes
    one_alone = one.availability(T, mc_samples=300, seed=2).uptimes
    assert (two_alone - one_alone).min() < 0


def test_nested_components_are_matched_by_their_place():
    inner = RepairableRBD([("a", "x"), ("x", "b")], {"x": pump(1.0)})
    outer = RepairableRBD([("s", "n"), ("n", "t")], {"n": inner})
    flat = RepairableRBD([("s", "n"), ("n", "t")], {"n": pump(1.0)})
    same = outer.compare(outer, T, mc_samples=100, seed=8)
    assert same.estimate == 0.0
    # The nested component "x" is not the flat "n": they are not paired.
    assert outer.compare(flat, T, mc_samples=100, seed=8).standard_error > 0


@pytest.mark.parametrize(
    "options",
    [
        {"quantity": "uptime"},
        {"confidence": 1.5},
        {"mc_samples": 0},
        {"quantity": "cost"},  # neither system is priced
    ],
)
def test_invalid_comparisons(options):
    options = {"mc_samples": 10, **options}
    with pytest.raises(ValueError):
        plant().compare(plant(2.0), T, seed=0, **options)


def test_comparing_needs_replayable_draws():
    rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": standby_pair()})
    with pytest.raises(NotImplementedError):
        rbd.compare(rbd, T, mc_samples=10, seed=0)


# -- NonRepairableRBD: MTTF ---------------------------------------------------


def test_mttf_to_a_tolerance_is_a_run_of_its_size():
    rbd = parallel(2)
    interval = rbd.mean_time_to_failure_interval(
        mc_samples=1000, seed=1, tolerance=1.0
    )
    n = interval.n_samples
    assert n > 1000 and n % 1000 == 0
    assert interval.upper - interval.estimate <= 1.0
    lifetimes = rbd.random(n, seed=1)
    assert half_width(lifetimes[: n - 1000]) > 1.0
    assert interval == rbd.mean_time_to_failure_interval(mc_samples=n, seed=1)
    for mean in (rbd.mean, rbd.mean_time_to_failure):
        estimate = mean(1000, seed=1, method="simulate", tolerance=1.0)
        assert estimate == interval.estimate
    assert abs(interval.estimate - mttf(rbd)) < 4 * interval.standard_error


def test_mttf_that_does_not_converge_warns():
    rbd = parallel(2)
    with pytest.warns(RuntimeWarning, match="MTTF did not converge"):
        interval = rbd.mean_time_to_failure_interval(
            mc_samples=100, seed=2, tolerance=1e-3, max_samples=300
        )
    assert interval.n_samples == 300


def test_an_infinite_mttf_stops_at_once():
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t"), ("s", "t")], {"a": W([100, 2])}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        interval = rbd.mean_time_to_failure_interval(
            mc_samples=100, seed=0, tolerance=1.0
        )
    assert interval.n_samples == 100 and interval.estimate == math.inf


def test_antithetic_lifetimes_come_in_pairs():
    unit = W([100, 2])
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": unit})
    lifetimes = rbd.random(1000, seed=3, antithetic=True)
    np.testing.assert_allclose(
        unit.ff(lifetimes[0::2]) + unit.ff(lifetimes[1::2]), 1.0, rtol=1e-9
    )
    # Reproducible, and restoring the caller's generator.
    np.random.seed(11)
    before = np.random.random_sample()
    np.random.seed(11)
    again = rbd.random(1000, seed=3, antithetic=True)
    assert np.random.random_sample() == before
    np.testing.assert_array_equal(lifetimes, again)


def test_antithetic_mttf_is_unbiased_and_tighter():
    rbd = parallel(2)
    paired = rbd.mean_time_to_failure_interval(
        mc_samples=20_000, seed=4, antithetic=True
    )
    assert abs(paired.estimate - mttf(rbd)) < 4 * paired.standard_error
    lifetimes = rbd.random(20_000, seed=4, antithetic=True)
    pairs = (lifetimes[0::2] + lifetimes[1::2]) / 2
    assert paired.standard_error == pytest.approx(
        np.std(pairs, ddof=1) / np.sqrt(10_000), rel=1e-12
    )
    independent = rbd.mean_time_to_failure_interval(mc_samples=20_000, seed=4)
    assert paired.standard_error < 0.85 * independent.standard_error


def test_antithetic_mttf_to_a_tolerance():
    rbd = parallel(2)
    interval = rbd.mean_time_to_failure_interval(
        mc_samples=1000, seed=5, tolerance=1.0, antithetic=True
    )
    n = interval.n_samples
    lifetimes = rbd.random(n, seed=5, antithetic=True)
    assert half_width(lifetimes, antithetic=True) <= 1.0
    assert half_width(lifetimes[: n - 1000], antithetic=True) > 1.0
    assert interval.estimate == pytest.approx(np.mean(lifetimes), rel=1e-12)


def test_antithetic_draws_need_replayable_nodes():
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": Drawn(100.0)})
    with pytest.raises(NotImplementedError):
        rbd.random(10, seed=0, antithetic=True)
    with pytest.raises(ValueError, match="even"):
        parallel(2).random(11, seed=0, antithetic=True)


def test_a_nan_lifetime_cannot_be_paired():
    with pytest.raises(ValueError, match="NaN"):
        _check_lifetimes(np.array([1.0, np.nan]))
    _check_lifetimes(np.array([1.0, np.inf]))


def test_parallel_lifetimes_do_not_depend_on_the_processes(monkeypatch):
    monkeypatch.setattr(non_repairable_rbd, "RANDOM_BLOCK", 1000)
    rbd = parallel(2)
    one = rbd.random(3500, seed=6, n_jobs=1)
    np.testing.assert_array_equal(one, rbd.random(3500, seed=6, n_jobs=2))
    np.testing.assert_array_equal(one, rbd.random(3500, seed=6, n_jobs=-1))
    seeds = np.random.SeedSequence(6)
    first = rbd.random(1000, seed=montecarlo.block_seed(seeds))
    np.testing.assert_array_equal(one[:1000], first)
    paired = rbd.random(3500 - 1, seed=6, n_jobs=1, antithetic=False)
    assert len(paired) == 3499
    pairs = [
        rbd.random(3500 + 2, seed=7, n_jobs=j, antithetic=True) for j in (1, 3)
    ]
    np.testing.assert_array_equal(*pairs)
    unit = rbd.reliabilities["u0"]
    single = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": unit})
    lifetimes = single.random(3502, seed=7, n_jobs=2, antithetic=True)
    np.testing.assert_allclose(
        unit.ff(lifetimes[0::2]) + unit.ff(lifetimes[1::2]), 1.0, rtol=1e-9
    )
    intervals = [
        rbd.mean_time_to_failure_interval(
            mc_samples=1000, seed=8, tolerance=1.5, n_jobs=jobs
        )
        for jobs in (1, 2)
    ]
    assert intervals[0] == intervals[1]
    assert intervals[0].upper - intervals[0].estimate <= 1.5
    assert (
        abs(intervals[0].estimate - mttf(rbd))
        < 4 * intervals[0].standard_error
    )


def test_parallel_draws_step_through_events_when_they_must():
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": Drawn(100.0), "b": W([100, 2])},
    )
    assert rbd._row_sampler() is None
    one = rbd.random(300, seed=9, n_jobs=1)
    np.testing.assert_array_equal(one, rbd.random(300, seed=9, n_jobs=2))
    lifetimes = rbd.random(20_000, seed=9, n_jobs=2)
    exact = mttf(rbd)
    se = np.std(lifetimes, ddof=1) / np.sqrt(len(lifetimes))
    assert abs(np.mean(lifetimes) - exact) < 4 * se
    with pytest.raises(NotImplementedError):
        rbd.random(10, seed=0, antithetic=True)


@pytest.mark.parametrize(
    "options",
    [
        {"n_jobs": 0},
        {"n_jobs": "2"},
        {"tolerance": -1.0},
        {"max_samples": 1000},
        {"tolerance": 1.0, "max_samples": 10},
        {"antithetic": True, "mc_samples": 99},
        {"mc_samples": 0, "n_jobs": 1},
        {"tolerance": 1.0, "confidence": 2.0},
    ],
)
def test_invalid_mttf_options(options):
    options = {"mc_samples": 100, **options}
    with pytest.raises(ValueError):
        parallel(2).mean_time_to_failure_interval(seed=0, **options)


def test_invalid_draws():
    with pytest.raises(ValueError):
        parallel(2).random(0, seed=0, n_jobs=1)
    with pytest.raises(ValueError):
        parallel(2).random(10, seed=0, n_jobs=0)


# -- NonRepairableRBD: comparing MTTFs ----------------------------------------


def test_an_rbd_compared_with_itself():
    same = parallel(2).compare(parallel(2), mc_samples=500, seed=1)
    assert same.estimate == 0.0 and same.standard_error == 0.0


def test_compare_mttf_against_the_exact_difference():
    three, two = parallel(3), parallel(2)
    gain = three.compare(two, mc_samples=20_000, seed=2)
    exact = mttf(three) - mttf(two)
    assert abs(gain.estimate - exact) < 4 * gain.standard_error
    a = three.mean_time_to_failure_interval(mc_samples=20_000, seed=2)
    b = two.mean_time_to_failure_interval(mc_samples=20_000, seed=3)
    independent = math.hypot(a.standard_error, b.standard_error)
    assert gain.standard_error < 0.7 * independent


def test_compare_matches_nodes_by_name():
    # A third unit never shortens a sample's lifetime, and a longer-lived
    # unit in the same place never does either.
    better = W([120, 2])
    upgraded = NonRepairableRBD(
        [("s", "u0"), ("s", "u1"), ("u0", "t"), ("u1", "t")],
        {"u0": better, "u1": W([100, 2])},
    )
    for key in (1, 2):
        base = parallel(2)._keyed_lifetimes(2000, key)
        more = parallel(3)._keyed_lifetimes(2000, key) - base
        assert more.min() >= 0 and more.max() > 0
        up = upgraded._keyed_lifetimes(2000, key) - base
        assert up.min() >= 0 and up.max() > 0
    gain = upgraded.compare(parallel(2), mc_samples=20_000, seed=3)
    exact = mttf(upgraded) - mttf(parallel(2))
    assert abs(gain.estimate - exact) < 4 * gain.standard_error


def test_compare_with_repeated_nodes():
    fail = W([100, 2])
    shared = NonRepairableRBD(
        [
            ("s", "X"),
            ("X", "A"),
            ("A", "t"),
            ("s", "Y"),
            ("Y", "X2"),
            ("X2", "B"),
            ("B", "t"),
        ],
        {"X": fail, "X2": "X", "A": fail, "Y": fail, "B": fail},
    )
    series = NonRepairableRBD(
        [("s", "X"), ("X", "A"), ("A", "t")], {"X": fail, "A": fail}
    )
    lifetimes = shared._keyed_lifetimes(2000, 4)
    assert (lifetimes >= series._keyed_lifetimes(2000, 4)).all()
    gain = shared.compare(series, mc_samples=20_000, seed=4)
    exact = mttf(shared) - mttf(series)
    assert abs(gain.estimate - exact) < 4 * gain.standard_error


def test_compare_a_standby_node():
    # A node that draws several uniforms (a cold standby pair) against
    # the same two units in parallel.
    unit = W([100, 2])
    standby = NonRepairableRBD(
        [("s", "u0"), ("u0", "t")], {"u0": StandbyModel([unit, unit])}
    )
    gain = standby.compare(parallel(2), mc_samples=20_000, seed=5)
    exact = mttf(standby) - mttf(parallel(2))
    assert abs(gain.estimate - exact) < 4 * gain.standard_error


def test_each_uniform_of_a_node_has_its_own_stream():
    # A nested pair in parallel draws two uniforms per sample: the same
    # system as the flat pair, so the difference is zero (were the two
    # uniforms the same, the nested pair would be a single unit).
    nested = NonRepairableRBD([("s", "n"), ("n", "t")], {"n": parallel(2)})
    gain = nested.compare(parallel(2), mc_samples=20_000, seed=6)
    assert abs(gain.estimate) < 4 * gain.standard_error


def test_invalid_mttf_comparisons():
    with pytest.raises(ValueError):
        parallel(2).compare(parallel(3), mc_samples=0)
    with pytest.raises(ValueError):
        parallel(2).compare(parallel(3), mc_samples=10, confidence=0.0)
    drawn = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": Drawn(100.0)})
    with pytest.raises(NotImplementedError):
        drawn.compare(parallel(2), mc_samples=10, seed=0)


# -- the shared helpers -------------------------------------------------------


def test_helpers():
    assert montecarlo.blocks(0, 3) == []
    assert montecarlo.blocks(7, 3) == [3, 3, 1]
    assert montecarlo.jobs(-1) == (os.cpu_count() or 1)
    assert montecarlo.jobs(np.int64(2)) == 2
    assert math.isnan(montecarlo.standard_error([1.0], False))
    assert montecarlo.half_width([1.0], 0.95, False) == math.inf
    assert montecarlo.half_width([1.0, 2.0], 0.95, True) == math.inf
    assert montecarlo.sample_limit(10, None, None, False, ("N", "m")) is None
    assert montecarlo.sample_limit(10, 0.1, None, False, ("N", "m")) == 1000
    assert (
        montecarlo.more_samples(
            [1.0, math.inf], 10, 0.1, 0.95, 100, False, "x", "m"
        )
        == 0
    )
    assert (
        montecarlo.more_samples([1.0, 3.0], 10, 0.1, 0.95, 15, False, "x", "m")
        == 10
    )
    assert (
        montecarlo.more_samples(
            [1.0, 3.0] * 5, 10, 0.1, 0.95, 15, False, "x", "m"
        )
        == 5
    )
