"""Spares of block-replaced and tested components (#147): a block-replaced
component's replacements counted block interval by block interval, and a
tested one's on the lattice of its tests; checked against counts built from
the plain renewal count, binomial closed forms for a constant failure rate,
a Monte Carlo of the replacements, and the RBD's own simulation."""

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import binom

from repyability import RepairableRBD
from repyability.rbd import routes as r

E, W = surv.Exponential.from_params, surv.Weibull.from_params


def single(spec, **options):
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec}, **options)


def block(life, interval, repair="instant", **preventive):
    return {
        "reliability": life,
        "repairability": repair,
        "preventive": {"interval": interval, "policy": "block", **preventive},
    }


def inspected(life, interval, **inspection):
    return {
        "reliability": life,
        "repairability": "instant",
        "inspection": {"interval": interval, **inspection},
    }


def padded(*arrays):
    size = max(len(a) for a in arrays)
    return [np.pad(a, (0, size - len(a))) for a in arrays]


# -- block replacement --------------------------------------------------------


def test_block_intervals_with_instant_renewals_are_independent():
    # Each interval starts new, so the count is the plain renewal count of
    # each whole interval and of the part left, plus one per block time.
    life, interval, horizon = W([100.0, 2.0]), 60.0, 250.0
    demand = single(block(life, interval)).spares_demand(horizon)["c"]
    plain = single({"reliability": life, "repairability": "instant"})
    whole = plain.spares_demand(interval)["c"].probabilities
    rest = plain.spares_demand(horizon - 4 * interval)["c"].probabilities
    want = np.array([1.0])
    for _ in range(4):
        want = np.convolve(want, whole)
    want = np.concatenate([np.zeros(4), np.convolve(want, rest)])
    got, want = padded(demand.probabilities, want)
    np.testing.assert_allclose(got, want, atol=3e-6)


def test_a_block_interval_past_the_horizon_is_the_plain_count():
    life = W([100.0, 2.0])
    spec = block(life, 1000.0, repair=E([0.2]))
    plain = {"reliability": life, "repairability": E([0.2])}
    got, want = padded(
        single(spec).spares_demand(250.0)["c"].probabilities,
        single(plain).spares_demand(250.0)["c"].probabilities,
    )
    np.testing.assert_allclose(got, want, atol=3e-6)


def test_repairs_and_replacements_that_take_time_against_the_simulation():
    # A repair still going on at a block time skips that replacement, and
    # carries the next unit's start past it.
    rbd = single(
        block(W([100.0, 2.0]), 60.0, repair=W([15.0, 1.5]), duration=E([0.2]))
    )
    exact = rbd.spares_demand(250.0)["c"]
    simulated = rbd.spares_demand(
        250.0, method="simulate", mc_samples=40_000, seed=3
    )["c"]
    got, want = padded(exact.probabilities, simulated.probabilities)
    np.testing.assert_allclose(got, want, atol=0.01)
    assert exact.mean() == pytest.approx(simulated.mean(), rel=0.01)
    assert exact.probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_the_stock_of_a_block_replaced_component_is_refused():
    rbd = single(block(W([100.0, 2.0]), 60.0))
    with pytest.raises(NotImplementedError, match="#160") as error:
        rbd.spares_stock(30.0, fill_rate=0.9)
    report = rbd.analysis_routes()
    assert report["spares_stock"].route == r.REFUSED
    assert report["spares_stock"].reason == str(error.value)
    assert report["spares_demand"].route == r.NUMERICAL


# -- hidden failures found by tests -------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 7.0])
def test_a_constant_rate_finds_a_failure_at_each_test_alike(offset):
    # Memoryless: each test finds the unit failed with probability p, so
    # the count over the tests before the horizon is binomial.
    rate, interval, horizon = 0.02, 20.0, 250.0
    rbd = single(inspected(E([rate]), interval, offset=offset))
    p = -np.expm1(-rate * interval)
    tests = len(np.arange(offset or interval, horizon, interval))
    demand = rbd.spares_demand(horizon)["c"].probabilities
    # The first test, at the offset, finds a failure since 0.
    first = -np.expm1(-rate * (offset or interval))
    want = np.convolve(
        [1.0 - first, first], binom.pmf(np.arange(tests), tests - 1, p)
    )
    got, want = padded(demand, want)
    np.testing.assert_allclose(got, want, atol=1e-12)
    # In a lead time of 45 from a random time, 2 or 3 tests (3 with
    # probability 1/4); before a replacement, the 2 tests before it.
    stock = rbd.spares_stock(45.0, fill_rate=0.9)["c"]
    counts = np.arange(len(stock.on_order))
    random = 0.75 * binom.pmf(counts, 2, p) + 0.25 * binom.pmf(counts, 3, p)
    np.testing.assert_allclose(stock.on_order, random, atol=1e-12)
    counts = np.arange(len(stock.on_order_at_demand))
    np.testing.assert_allclose(
        stock.on_order_at_demand, binom.pmf(counts, 2, p), atol=1e-12
    )


def lattice(life, interval, first, size, rng):
    """Replacement times of a tested unit, by drawing its lives: each is
    replaced at the first test at or after it fails."""
    times, now, start = [], first, 0.0
    lives = life.qf(rng.uniform(size=size))
    k = 0
    while k < size:
        failure = start + lives[k]
        k += 1
        # The first test at or after the failure (``now`` is the next one).
        if failure > now:
            now += interval * np.ceil((failure - now) / interval)
        times.append(now)
        start, now = now, now + interval
    return np.array(times)


def test_a_wearing_life_against_a_monte_carlo_of_its_replacements():
    life, interval, first = W([100.0, 2.0]), 20.0, 7.0
    rbd = single(inspected(life, interval, offset=first))
    rng = np.random.default_rng(4)
    # From new over 250: 20,000 histories.
    counts = []
    for _ in range(20_000):
        times = lattice(life, interval, first, 12, rng)
        counts.append(int(np.sum(times < 250.0)))
    simulated = np.bincount(counts) / len(counts)
    got, want = padded(rbd.spares_demand(250.0)["c"].probabilities, simulated)
    np.testing.assert_allclose(got, want, atol=0.012)
    # In the long run: one long history.
    times = lattice(life, interval, first, 200_000, rng)
    starts = rng.uniform(times[100], times[-100], 200_000)
    on_order = np.searchsorted(times, starts + 45.0) - np.searchsorted(
        times, starts
    )
    stock = rbd.spares_stock(45.0, fill_rate=0.9)["c"]
    got, want = padded(stock.on_order, np.bincount(on_order) / len(on_order))
    np.testing.assert_allclose(got, want, atol=0.004)
    inner = times[100:-100]
    before = np.arange(100, len(times) - 100) - np.searchsorted(
        times, inner - 45.0, side="right"
    )
    got, want = padded(
        stock.on_order_at_demand, np.bincount(before) / len(before)
    )
    np.testing.assert_allclose(got, want, atol=0.004)


def test_frequent_tests_count_as_the_failures_do():
    life = W([100.0, 2.0])
    got, want = padded(
        single(inspected(life, 0.01)).spares_demand(250.0)["c"].probabilities,
        single({"reliability": life, "repairability": "instant"})
        .spares_demand(250.0)["c"]
        .probabilities,
    )
    np.testing.assert_allclose(got, want, atol=5e-4)


def test_the_simulation_agrees_for_a_tested_component():
    rbd = single(inspected(W([100.0, 2.0]), 20.0, offset=7.0))
    exact = rbd.spares_demand(250.0)["c"]
    simulated = rbd.spares_demand(
        250.0, method="simulate", mc_samples=40_000, seed=5
    )["c"]
    got, want = padded(exact.probabilities, simulated.probabilities)
    np.testing.assert_allclose(got, want, atol=0.01)
    assert exact.method == "exact"


def test_tests_that_take_time_or_miss_failures_are_refused():
    life = W([100.0, 2.0])
    for spec in (
        inspected(life, 20.0, duration=E([1.0])),
        inspected(life, 20.0, coverage=0.5, full_test=100.0),
        {**inspected(life, 20.0), "repairability": E([1.0])},
    ):
        rbd = single(spec)
        with pytest.raises(NotImplementedError, match="#159") as error:
            rbd.spares_demand(100.0)
        assert rbd.analysis_routes()["spares_demand"].reason == str(
            error.value
        )
    routes = single(inspected(life, 20.0)).analysis_routes()
    assert routes["spares_demand"].route == r.NUMERICAL
    assert routes["spares_stock"].route == r.NUMERICAL
