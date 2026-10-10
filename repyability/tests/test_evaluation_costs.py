"""What the costly evaluations cost (#229), with the same values.

- An MTTF integrates the survival function on a few pieces a decade, by
  the Gauss-Kronrod (7, 15) rule, and doubles into the tail only until the
  curve is 0: a few hundred evaluations, where thousands made a large
  network's ``mean()`` take minutes.
- A network whose decision diagram is refused is refused at once after.
- The decision diagrams' computed table starts again past a limit, and a
  cofactor leaves the nodes after its variable alone: Fussell-Vesely's
  closures over a meshed core take half the memory.
- A common-cause group's chain keeps ``exp(G dt)`` for the steps it takes
  again, and a plan keeps its groups' tables for its other long-run
  values: an interval search's evaluations are ten times faster.
- A tested unit's walk convolves directly or by FFT without scipy's
  choosing each time.
- Of interval plans as good, the search keeps the first it tried, and of
  plans that cost the same the most available, whatever the last bits of
  their values: identical members' plans no longer turn on them.
"""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from scipy.signal import convolve, correlate

from repyability import MGL, CCFGroup, Network, NonRepairableRBD, RepairableRBD
from repyability import network as network_module
from repyability.rbd import (
    _ccf_chain,
    _ccf_groups,
    _hidden_tests,
    _intervals,
    _long_run,
    _ordered_bdd,
)
from repyability.rbd._mean_lifetime import (
    mean_lifetime,
    model_kinks,
    model_knots,
)

E, W = surv.Exponential.from_params, surv.Weibull.from_params


def counted(sf):
    """``sf``, and the number of times it has been evaluated at."""
    points = []

    def f(t):
        points.append(np.size(t))
        return sf(t)

    return f, points


CURVES = [
    # (name, survival function, models for the knots, exact mean)
    (
        "two series pairs in parallel",
        lambda t: 1 - (1 - np.exp(-0.1 * np.asarray(t)) ** 2) ** 2,
        [E([0.1])],
        7.5,
    ),
    ("weibull, beta 0.5", W([100.0, 0.5]).sf, [W([100.0, 0.5])], 200.0),
    (
        "weibull, beta 3",
        W([100.0, 3.0]).sf,
        [W([100.0, 3.0])],
        100 * math.gamma(1 + 1 / 3),
    ),
    (
        "weibull, beta 50",
        W([100.0, 50.0]).sf,
        [W([100.0, 50.0])],
        100 * math.gamma(1 + 1 / 50),
    ),
    (
        "weibull with an offset",
        W([100.0, 2.0], gamma=50.0).sf,
        [W([100.0, 2.0], gamma=50.0)],
        50 + 100 * math.gamma(1.5),
    ),
    (
        "lognormal, sigma 2",
        surv.LogNormal.from_params([3.0, 2.0]).sf,
        [surv.LogNormal.from_params([3.0, 2.0])],
        math.exp(5.0),
    ),
]


@pytest.mark.parametrize("name, sf, models, exact", CURVES)
def test_a_mean_in_a_few_hundred_points(name, sf, models, exact):
    f, points = counted(sf)
    mean = mean_lifetime(
        f, [model_knots(m) for m in models], [model_kinks(m) for m in models]
    )
    assert mean == pytest.approx(exact, rel=1e-12)
    assert sum(points) < 1200


def test_many_different_models_take_no_more():
    # Twenty distinct Weibulls in series: their knots made 63,100 points.
    models = [W([100.0 + 10 * i, 1.5 + 0.1 * i]) for i in range(20)]
    rbd = NonRepairableRBD(
        [("s", "n0")]
        + [(f"n{i}", f"n{i + 1}") for i in range(19)]
        + [("n19", "t")],
        {f"n{i}": m for i, m in enumerate(models)},
    )
    sf, points = counted(rbd.sf)
    rbd.sf = sf
    mean = rbd.mean()
    exact, _ = quad(
        lambda t: np.prod([m.sf(t) for m in models]),
        0,
        np.inf,
        epsabs=0,
        epsrel=1e-13,
        limit=500,
    )
    assert mean == pytest.approx(exact, rel=1e-11)
    assert sum(points) < 1200


def test_a_numerical_curve_s_grid_starts_pieces():
    from repyability import StandbyModel

    unit = W([100.0, 2.0])
    standby = StandbyModel([unit, unit], k=1)
    assert model_kinks(standby).size > 10  # its convolution's grid
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": standby})
    assert rbd.mean() == pytest.approx(2 * 100 * math.gamma(1.5), rel=1e-6)


def test_a_life_that_never_ends_has_no_mean():
    def sf(t):
        return 0.1 + 0.9 * np.exp(-np.asarray(t) / 10.0)

    assert mean_lifetime(sf, [], []) == np.inf


def grid(k, rate=0.1):
    links = {}
    for i in range(k):
        for j in range(k):
            if i + 1 < k:
                links[f"h{i}_{j}"] = ((i, j), (i + 1, j), E([rate]))
            if j + 1 < k:
                links[f"v{i}_{j}"] = ((i, j), (i, j + 1), E([rate]))
    return Network(links, source=(0, 0), target=(k - 1, k - 1))


def test_a_network_mean_against_its_simulation():
    net = grid(4)
    exact = net.mean()
    simulated = net.mean(method="simulate", mc_samples=40_000, seed=3)
    lives = net.random(40_000, seed=3)
    error = np.std(lives) / np.sqrt(len(lives))
    assert abs(exact - simulated) < 4 * error


def test_a_refused_network_is_refused_at_once_after(monkeypatch):
    monkeypatch.setattr(network_module, "MAX_STATES", 50)
    built = []
    real = network_module._frontier_plan

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(network_module, "_frontier_plan", counting)
    net = grid(5)
    messages = set()
    for call in (net.sf, net.birnbaum_importance, net.sf):
        with pytest.raises(NotImplementedError) as refused:
            call(1.0)
        messages.add(str(refused.value))
    with pytest.raises(NotImplementedError):
        net.mean()
    assert built == [1]
    assert len(messages) == 1


def meshed():
    """A geometric mesh of 50 nodes: 17,349 minimal cut sets."""
    rng = np.random.default_rng(3)
    points = rng.random((50, 2))
    points = points[np.argsort(points[:, 0])]
    names = [f"n{i}" for i in range(50)]
    edges = set()
    for i in range(50):
        later = np.arange(i + 1, 50)
        if not later.size:
            continue
        distance = np.linalg.norm(points[later] - points[i], axis=1)
        for j in later[np.argsort(distance)[:4]]:
            edges.add((names[i], names[j]))
    edges = sorted(edges)
    edges += [("s", names[k]) for k in range(3)]
    edges += [(names[-1 - k], "t") for k in range(3)]
    return NonRepairableRBD(
        edges, {n: E([0.01 * (1 + (i % 5))]) for i, n in enumerate(names)}
    )


def test_a_small_computed_table_gives_the_same_importances(monkeypatch):
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("a", "d")]
        + [("b", "d"), ("c", "t"), ("d", "t"), ("a", "e"), ("e", "t")],
        {n: E([0.01 * (i + 1)]) for i, n in enumerate("abcde")},
    )
    expected = rbd.fussell_vesely(30.0)
    monkeypatch.setattr(_ordered_bdd, "COMPUTED_LIMIT", 4)
    again = NonRepairableRBD(rbd._init_args["edges"], rbd.reliabilities)
    got = again.fussell_vesely(30.0)
    for node in expected:
        assert got[node] == pytest.approx(expected[node], rel=1e-13)


def test_fussell_vesely_on_a_mesh_agrees_with_its_cut_sets():
    rbd = meshed()
    fv = rbd.fussell_vesely(10.0)
    rare = rbd.fussell_vesely(10.0, method="rare_event")
    for node in fv:
        # The union is below the sum of its sets' probabilities.
        assert 0.0 <= fv[node] <= rare[node] * (1 + 1e-9)


def two_out_of_three():
    def unit():
        return {
            "reliability": E([1e-5]),
            "repairability": E([1 / 8.0]),
            "inspection": {"interval": 8760.0, "cost": 100.0},
            "repair_cost": 1000.0,
        }

    return RepairableRBD(
        [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
        {n: unit() for n in "abc"},
        k={"t": 2},
        ccf_groups=[CCFGroup(list("abc"), MGL(0.1, 0.3, basis="rate"))],
        downtime_cost_rate=10.0,
    )


def test_a_chain_s_kept_steps_give_what_its_series_gives():
    rbd = two_out_of_three()
    group = rbd.ccf_groups[0]
    times, _ = _long_run._long_run_grid(rbd)
    kept = _ccf_groups._group_states(rbd, group, times).probabilities
    # The same chain with each step's series summed for the vector alone.
    original = _ccf_chain._Hidden.evolve
    try:
        _ccf_chain._Hidden.evolve = _ccf_chain._Hidden._uniformized
        again = _ccf_groups._group_states(rbd, group, times).probabilities
    finally:
        _ccf_chain._Hidden.evolve = original
    np.testing.assert_allclose(kept, again, rtol=1e-11, atol=1e-15)


def test_a_plan_keeps_its_groups_tables_and_its_copies_do_not():
    rbd = two_out_of_three()
    plan = _intervals._with_intervals(rbd, inspection={"a": 4380.0})
    cost = plan.expected_cost_rate()
    assert "_ccf_tables" in plan.__dict__
    availability = plan.mean_availability()
    other = _intervals._with_intervals(plan, inspection={"b": 26280.0})
    assert "_ccf_tables" not in other.__dict__
    fresh = RepairableRBD(**{**rbd._init_args}).with_intervals({"a": 4380.0})
    assert cost == pytest.approx(fresh.expected_cost_rate(), rel=1e-12)
    assert availability == pytest.approx(fresh.mean_availability(), rel=1e-12)
    assert other.mean_availability() != pytest.approx(availability, rel=1e-6)


@pytest.mark.parametrize("n, m", [(3, 9), (64, 500), (65, 500), (400, 900)])
def test_the_walk_s_convolutions_are_scipy_s(n, m):
    rng = np.random.default_rng(n + m)
    a, b = rng.random(n), rng.random(m)
    np.testing.assert_allclose(
        _hidden_tests._convolve(a, b), convolve(a, b), rtol=1e-12, atol=1e-13
    )
    long, short = (a, b) if n >= m else (b, a)
    np.testing.assert_allclose(
        _hidden_tests._correlate(long, short),
        correlate(long, short, mode="valid"),
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.fixture(params=["every combination", "local search"])
def search(request, monkeypatch):
    if request.param == "local search":
        monkeypatch.setattr(_intervals, "_MAX_COMBINATIONS", 1)

    def choose(options, evaluate, **target):
        return _intervals._choose_from(
            list(options),
            options,
            evaluate,
            target.get("min_availability"),
            target.get("max_cost_rate"),
        )

    return choose


def test_identical_members_plans_do_not_turn_on_their_last_bits(search):
    # Two identical members, one tested at 1 and the other at 2 the best:
    # the two ways round cost the same but for the last bits.
    def evaluate_with(noise):
        def evaluate(intervals):
            a, b = intervals["a"], intervals["b"]
            cost = {2.0: 2.0, 3.0: 1.0, 4.0: 3.0}[a + b]
            return cost * (1 + noise * (a > b)), 0.99

        return evaluate

    options = {"a": [1.0, 2.0], "b": [1.0, 2.0]}
    chosen = [
        search(options, evaluate_with(noise)) for noise in (1e-15, 0.0, -1e-15)
    ]
    assert chosen[0] == chosen[1] == chosen[2]
    assert sorted(chosen[0].values()) == [1.0, 2.0]


@pytest.mark.parametrize("noise", [1e-15, 0.0, -1e-15])
def test_of_plans_that_cost_the_same_the_most_available(search, noise):
    def evaluate(intervals):
        if intervals["a"] == 1.0:
            return 5.0 * (1 + noise), 0.90
        return 5.0, 0.95

    assert search({"a": [1.0, 2.0]}, evaluate) == {"a": 2.0}
    # With a cost cap, the least unavailable, then the cheapest.
    capped = search({"a": [1.0, 2.0]}, evaluate, max_cost_rate=6.0)
    assert capped == {"a": 2.0}


def test_plans_apart_by_more_than_their_last_bits_are_not_as_good(search):
    def evaluate(intervals):
        return 5.0 * (1 + 1e-9 * intervals["a"]), 0.95

    assert search({"a": [2.0, 1.0]}, evaluate) == {"a": 1.0}
