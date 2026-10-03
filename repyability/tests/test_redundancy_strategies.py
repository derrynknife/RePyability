"""Tests for redundancy allocation with k-out-of-n nodes and a choice of
redundancy strategy (``required``, ``strategy``, ``switching_probability``;
issue #78).

Node reliabilities are checked against closed forms -- binomial tails for
active k-out-of-n copies, Erlang and Poisson sums for exponential cold
standby -- and the allocations against an independent brute force over
every number of copies and every strategy.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import binom, gamma, poisson
from surpyval import FixedEventProbability

from repyability import ComponentOption, NonRepairableRBD, StandbyModel
from repyability.rbd.redundancy_allocation import (
    active_unreliability,
    node_designs,
)

MISSION = 1000.0


def exponential(mean):
    return surv.Exponential.from_params([1.0 / mean])


def series(means):
    names = [f"n{i}" for i in range(len(means))]
    edges = [("s", names[0])] + list(zip(names, names[1:]))
    rbd = NonRepairableRBD(
        edges + [(names[-1], "t")],
        {n: exponential(mean) for n, mean in zip(names, means)},
    )
    return names, rbd


def active(p, n, k):
    """k-out-of-n active copies of reliability p: a binomial tail."""
    return float(binom.sf(k - 1, n, p))


def cold(rate, n, k, t=MISSION):
    """k-out-of-n cold standby of identical exponential units: the
    arrangement fails at the (n - k + 1)-th failure, Erlang(n-k+1, k*rate).
    """
    return float(gamma.sf(t, n - k + 1, scale=1.0 / (k * rate)))


def node_reliability(mean, n, k, way, t=MISSION):
    p = math.exp(-t / mean)
    return active(p, n, k) if way == "active" else cold(1.0 / mean, n, k, t)


# -- node reliabilities: closed forms --------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("p", [0.3, 0.8, 0.97])
def test_active_k_out_of_n_is_a_binomial_tail(p, k):
    designs = node_designs([(p, 1.0)], 8, fewest=k)
    assert [d.counts[0] for d in designs] == list(range(k, 9))
    for d in designs:
        expected = active(p, d.counts[0], k)
        assert d.reliability == pytest.approx(expected, rel=1e-12)
        assert d.unreliability == pytest.approx(1 - expected, rel=1e-9)


def test_active_k_out_of_n_with_mixed_types():
    # Every outcome of every copy, enumerated.
    ps = [0.9, 0.7, 0.5]
    for counts in [(1, 1, 1), (2, 0, 1), (1, 2, 2), (0, 3, 1)]:
        copies = [p for p, c in zip(ps, counts) for _ in range(c)]
        for k in range(1, len(copies) + 1):
            failing = sum(
                math.prod(p if up else 1 - p for p, up in zip(copies, ups))
                for ups in itertools.product([0, 1], repeat=len(copies))
                if sum(ups) < k
            )
            got = active_unreliability([1 - p for p in ps], counts, k)
            assert got == pytest.approx(failing, abs=1e-15)


@pytest.mark.parametrize("k", [1, 2])
def test_cold_standby_node_is_erlang(k):
    # One costed node in series: the system reliability is the node's.
    names, rbd = series([2000.0])
    for budget in range(k, k + 5):
        result = rbd.allocate_redundancy(
            {"n0": 1.0}, budget=budget, t=MISSION, required=k, strategy="cold"
        )
        assert result.units == {"n0": budget}
        assert result.strategy == {"n0": "cold"}
        assert result.reliability == pytest.approx(
            cold(1 / 2000, budget, k), rel=1e-12
        )


def test_imperfect_switching_is_a_poisson_sum():
    # One unit required, exponential units, switching succeeding with
    # probability rho: the node survives j failures, j < n, only if all j
    # switches succeed, sum_j rho**j P(Poisson(rate t) = j). StandbyModel
    # computes this by numerical convolution (accurate to about 1e-3).
    names, rbd = series([2000.0])
    rho = 0.9
    for n in range(1, 5):
        result = rbd.allocate_redundancy(
            {"n0": 1.0},
            budget=n,
            t=MISSION,
            strategy="cold",
            switching_probability=rho,
        )
        expected = sum(
            rho**j * poisson.pmf(j, MISSION / 2000) for j in range(n)
        )
        assert result.reliability == pytest.approx(expected, rel=2e-3)


def test_weibull_cold_standby_uses_the_standby_model():
    model = surv.Weibull.from_params([1500, 2.0])
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": model})
    result = rbd.allocate_redundancy(
        {"a": 1.0}, budget=3, t=MISSION, strategy="cold"
    )
    direct = float(np.ravel(StandbyModel([model] * 3).sf(MISSION))[0])
    assert result.reliability == pytest.approx(direct, rel=1e-12)


# -- allocations: independent brute force ----------------------------------


def brute_force(rbd, names, means, costs, cap, fewest, ways, limit):
    """(reliability, cost, units, strategies) of every allocation within
    the budget, with the node reliabilities from the closed forms."""
    base = rbd._base_node_probabilities(np.atleast_1d(MISSION), set(), set())
    per_node = []
    for name, mean, k, node_ways in zip(names, means, fewest, ways):
        per_node.append(
            [
                (node_reliability(mean, n, k, way), n * costs[name], n, way)
                for n in range(k, cap + 1)
                for way in node_ways
            ]
        )
    for combo in itertools.product(*per_node):
        cost = sum(c[1] for c in combo)
        if cost > limit + 1e-9:
            continue
        probabilities = dict(base)
        for name, c in zip(names, combo):
            probabilities[name] = np.array([c[0]])
        reliability = float(np.ravel(rbd.system_probability(probabilities))[0])
        yield reliability, cost, tuple(c[2] for c in combo), combo


STRATEGIES = {
    "active": ("active",),
    "cold": ("cold",),
    "choose": ("active", "cold"),
}


@pytest.mark.parametrize("strategy", ["active", "cold", "choose"])
@pytest.mark.parametrize("seed", range(3))
def test_series_matches_brute_force(seed, strategy):
    rng = np.random.default_rng(seed)
    means = list(rng.uniform(800, 3000, 3))
    names, rbd = series(means)
    costs = dict(zip(names, map(float, rng.integers(1, 4, 3))))
    required = dict(zip(names, map(int, rng.integers(1, 3, 3))))
    fewest = [required[n] for n in names]
    ways = [STRATEGIES[strategy]] * 3
    limit = sum(required[n] * costs[n] for n in names) + 6.0
    every = list(brute_force(rbd, names, means, costs, 6, fewest, ways, limit))
    result = rbd.allocate_redundancy(
        costs,
        budget=limit,
        t=MISSION,
        max_units=6,
        required=required,
        strategy=strategy,
    )
    best = max(r for r, _, _, _ in every)
    assert result.reliability == pytest.approx(best, abs=1e-12)
    assert result.cost <= limit + 1e-9
    for name in names:
        assert result.units[name] >= required[name]
        assert result.strategy[name] in STRATEGIES[strategy]
    # The target form: the cheapest allocation reaching 90% of the best.
    target = 0.9 * best
    result = rbd.allocate_redundancy(
        costs,
        target=target,
        t=MISSION,
        max_units=6,
        required=required,
        strategy=strategy,
    )
    unlimited = brute_force(
        rbd, names, means, costs, 6, fewest, ways, math.inf
    )
    cheapest = min(c for r, c, _, _ in unlimited if r >= target)
    assert result.cost == pytest.approx(cheapest)
    assert result.reliability >= target


@pytest.mark.parametrize("strategy", ["active", "choose"])
def test_bridge_matches_brute_force(strategy):
    # Not a series of the costed nodes: the exhaustive search.
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "d"),
        ("c", "d"),
        ("b", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    means = {"a": 1500.0, "b": 1200.0, "c": 2500.0, "d": 900.0, "e": 1300.0}
    rbd = NonRepairableRBD(
        edges, {n: exponential(m) for n, m in means.items()}
    )
    costs = {"a": 3.0, "c": 4.0, "d": 1.5}
    names = list(costs)
    required = {"d": 2}
    fewest = [required.get(n, 1) for n in names]
    ways = [STRATEGIES[strategy]] * 3
    every = brute_force(
        rbd, names, [means[n] for n in names], costs, 4, fewest, ways, 16.0
    )
    best = max(r for r, _, _, _ in every)
    result = rbd.allocate_redundancy(
        costs,
        budget=16.0,
        t=MISSION,
        max_units=4,
        required=required,
        strategy=strategy,
    )
    assert result.reliability == pytest.approx(best, abs=1e-12)


def test_choosing_is_never_worse_than_either_strategy():
    rng = np.random.default_rng(11)
    for _ in range(4):
        means = list(rng.uniform(500, 2500, 3))
        names, rbd = series(means)
        costs = dict(zip(names, map(float, rng.integers(1, 4, 3))))
        budget = 2 * sum(costs.values()) + 5.0
        found = {
            strategy: rbd.allocate_redundancy(
                costs, budget=budget, t=MISSION, strategy=strategy, required=2
            ).reliability
            for strategy in ("active", "cold", "choose")
        }
        assert found["choose"] >= max(found["active"], found["cold"]) - 1e-15


def test_strategy_per_node_and_options():
    # A dict gives particular nodes a strategy; with options, the cold
    # spares are the chosen types.
    names, rbd = series([2000.0, 1500.0])
    options = [
        ComponentOption("own", exponential(1500.0), 1.0),
        ComponentOption("better", exponential(4000.0), 2.0),
    ]
    result = rbd.allocate_redundancy(
        {"n0": 1.0, "n1": options},
        budget=7,
        t=MISSION,
        strategy={"n1": "cold"},
    )
    assert result.strategy == {"n0": "active", "n1": "cold"}
    assert sum(result.mix["n1"].values()) == result.units["n1"]


@pytest.mark.parametrize("method", ["exact", "greedy"])
def test_greedy_and_exact_with_strategies(method):
    names, rbd = series([2000.0, 900.0, 1500.0])
    costs = {"n0": 1.0, "n1": 2.0, "n2": 1.5}
    result = rbd.allocate_redundancy(
        costs,
        budget=14,
        t=MISSION,
        strategy="choose",
        required={"n1": 2},
        method=method,
    )
    assert result.cost <= 14
    assert result.units["n1"] >= 2
    exact = rbd.allocate_redundancy(
        costs, budget=14, t=MISSION, strategy="choose", required={"n1": 2}
    )
    assert result.reliability <= exact.reliability + 1e-15


# -- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"required": 0}, "at least 1"),
        ({"required": 1.5}, "an integer"),
        ({"required": True}, "an integer"),
        ({"required": {"zz": 2}}, "not in costs"),
        ({"required": 3, "max_units": 2}, "fewer than the 3 copies"),
        ({"strategy": "warm"}, "'active', 'cold' or 'choose'"),
        ({"strategy": {"zz": "cold"}}, "not in costs"),
        (
            {"strategy": "cold", "switching_probability": 1.5},
            r"\[0, 1\]",
        ),
        ({"switching_probability": 0.9}, "which no node uses"),
        (
            {"strategy": {"n0": "cold"}, "switching_probability": {"n1": 0.9}},
            "not a cold standby node",
        ),
        (
            {"strategy": "cold", "required": 2, "switching_probability": 0.9},
            "one unit required",
        ),
    ],
)
def test_strategy_validation(kwargs, match):
    names, rbd = series([2000.0, 1500.0])
    with pytest.raises(ValueError, match=match):
        rbd.allocate_redundancy(
            {"n0": 1.0, "n1": 1.0}, budget=10, t=MISSION, **kwargs
        )


def test_cold_standby_needs_lifetime_models():
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t")], {"a": FixedEventProbability.from_params(0.1)}
    )
    with pytest.raises(ValueError, match="lifetime models"):
        rbd.allocate_redundancy({"a": 1.0}, budget=3, strategy="cold")
    options = [ComponentOption("fixed", 0.9, 1.0)]
    with pytest.raises(ValueError, match="lifetime models"):
        rbd.allocate_redundancy(
            {"a": options}, budget=3, strategy="choose", t=MISSION
        )


def test_budget_must_afford_the_required_copies():
    names, rbd = series([2000.0])
    with pytest.raises(ValueError, match="at least 3.*the required copies"):
        rbd.allocate_redundancy({"n0": 1.0}, budget=2, t=MISSION, required=3)


def test_choosing_with_imperfect_switching():
    # With perfect switching cold standby always wins; with unreliable
    # switching, active copies can win (Coit, 2003): here the long-lived
    # nodes stay active and the short-lived one gets cold spares. Checked
    # against every allocation, with the node reliabilities from binomial
    # tails and StandbyModel (a node without spares is the same either way).
    means = [8000.0, 700.0, 3000.0]
    names, rbd = series(means)
    costs = {"n0": 1.0, "n1": 1.0, "n2": 2.0}
    rho = 0.8
    per_node = []
    for mean in means:
        model = exponential(mean)
        p = math.exp(-MISSION / mean)
        designs = [(p, 1, "active")]
        for n in range(2, 6):
            designs.append((active(p, n, 1), n, "active"))
            standby = StandbyModel([model] * n, switching_probability=rho)
            designs.append(
                (float(np.ravel(standby.sf(MISSION))[0]), n, "cold")
            )
        per_node.append(designs)
    budget = 9.0
    best = max(
        math.prod(d[0] for d in combo)
        for combo in itertools.product(*per_node)
        if sum(d[1] * costs[n] for d, n in zip(combo, names)) <= budget
    )
    result = rbd.allocate_redundancy(
        costs,
        budget=budget,
        t=MISSION,
        max_units=5,
        strategy="choose",
        switching_probability=rho,
    )
    assert result.reliability == pytest.approx(best, rel=1e-12)
    assert result.strategy == {"n0": "active", "n1": "cold", "n2": "active"}
    # One copy has no spares: reported as active.
    single = rbd.allocate_redundancy(
        costs,
        budget=4,
        t=MISSION,
        strategy="choose",
        switching_probability=rho,
    )
    assert single.units == {"n0": 1, "n1": 1, "n2": 1}
    assert set(single.strategy.values()) == {"active"}


@pytest.mark.parametrize("mixing", [True, False])
def test_required_copies_with_types(mixing):
    # Two copies must work, of two types, as cold standby.
    names, rbd = series([2000.0])
    options = [
        ComponentOption("short", exponential(1000.0), 1.0),
        ComponentOption("long", exponential(3000.0), 2.0),
    ]
    result = rbd.allocate_redundancy(
        {"n0": options},
        budget=6,
        t=MISSION,
        required=2,
        strategy="cold",
        mixing=mixing,
        max_units=5,
    )
    counts = result.mix["n0"]
    assert sum(counts.values()) >= 2
    assert mixing or len(counts) == 1
    assert result.cost <= 6


def test_greedy_switches_strategy():
    # The greedy starts active; with perfect switching cold spares are
    # better, so it switches.
    names, rbd = series([2000.0])
    result = rbd.allocate_redundancy(
        {"n0": 1.0}, budget=3, t=MISSION, strategy="choose", method="greedy"
    )
    assert result.strategy == {"n0": "cold"}
    assert result.reliability == pytest.approx(cold(1 / 2000, 3, 1))


def test_cold_target_beyond_active_reach():
    # Three active copies of e^-1 reliability reach 0.747; three cold ones
    # reach 0.9197, so a target of 0.9 is reachable only as cold standby.
    names, rbd = series([1000.0])
    result = rbd.allocate_redundancy(
        {"n0": 1.0}, target=0.9, t=MISSION, strategy="cold", max_units=3
    )
    assert result.units == {"n0": 3}
    assert result.reliability == pytest.approx(cold(1 / 1000, 3, 1))
    with pytest.raises(ValueError, match="unreachable"):
        rbd.allocate_redundancy(
            {"n0": 1.0}, target=0.9, t=MISSION, max_units=3
        )
