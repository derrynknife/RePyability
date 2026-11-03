"""Reliability and mean time to the first system failure of a repairable
diagram (``first_failure_sf``, ``mean_time_to_first_failure``).

Checked against closed forms (a parallel pair's mean, birth-death chains
written out by hand and exponentiated with scipy's ``expm``, a series'
product), against ``NonRepairableRBD`` where no repair can matter, and
against seeded simulations."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import NodeState, NonRepairableRBD, RepairableRBD

E = surv.Exponential.from_params
W = surv.Weibull.from_params

LAM, MU = 0.01, 0.1
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
TRIPLE = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
TIMES = np.array([0.0, 30.0, 100.0, 300.0, 1000.0])


def unit(life=None, repair=None):
    return {
        "reliability": life or E([LAM]),
        "repairability": repair or E([MU]),
    }


def pair(**kwargs):
    return RepairableRBD(PAIR, {"a": unit(), "b": unit()}, **kwargs)


def survival(generator, start, times):
    """p(0) exp(T t) 1 for the up-states' sub-generator T."""
    return np.array([(start @ expm(generator * t)).sum() for t in times])


@pytest.mark.parametrize("crews", [None, 1, 2])
def test_a_parallel_pair_has_the_closed_form_mean(crews):
    # (3 lam + mu) / (2 lam^2): the same with one crew or two, as the
    # second unit is only ever down while the first is up.
    expected = (3 * LAM + MU) / (2 * LAM**2)
    assert pair(repair_crews=crews).mean_time_to_first_failure() == (
        pytest.approx(expected, rel=1e-12)
    )


@pytest.mark.parametrize("crews", [None, 1])
def test_a_parallel_pair_survival_is_the_matrix_exponential(crews):
    generator = np.array([[-2 * LAM, 2 * LAM], [MU, -(LAM + MU)]])
    expected = survival(generator, np.array([1.0, 0.0]), TIMES)
    got = pair(repair_crews=crews).first_failure_sf(TIMES)
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-14)


def test_a_pair_started_with_one_unit_down():
    generator = np.array([[-2 * LAM, 2 * LAM], [MU, -(LAM + MU)]])
    state = {"a": NodeState(alive=False)}
    got = pair().first_failure_sf(TIMES, state=state)
    np.testing.assert_allclose(
        got, survival(generator, np.array([0.0, 1.0]), TIMES), rtol=1e-11
    )
    # By hand: m1 = 1 / (lam + mu) + mu / (lam + mu) * (1 / (2 lam) + m1).
    assert pair().mean_time_to_first_failure(state=state) == pytest.approx(
        600.0, rel=1e-12
    )


def test_both_units_down_at_the_start_is_a_failure():
    state = {"a": NodeState(alive=False), "b": NodeState(alive=False)}
    assert pair().first_failure_sf([0.0, 10.0], state=state).tolist() == [
        0.0,
        0.0,
    ]
    assert pair().mean_time_to_first_failure(state=state) == 0.0


@pytest.mark.parametrize("crews, repair_rate", [(None, "scaled"), (1, "flat")])
def test_three_in_parallel_is_a_birth_death_chain(crews, repair_rate):
    # States 0, 1, 2 units down; the third failure is the system's.
    def mu(k):
        return k * MU if repair_rate == "scaled" else MU

    generator = np.array(
        [
            [-3 * LAM, 3 * LAM, 0.0],
            [mu(1), -(2 * LAM + mu(1)), 2 * LAM],
            [0.0, mu(2), -(LAM + mu(2))],
        ]
    )
    rbd = RepairableRBD(TRIPLE, {n: unit() for n in "abc"}, repair_crews=crews)
    start = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        rbd.first_failure_sf(TIMES),
        survival(generator, start, TIMES),
        rtol=1e-10,
        atol=1e-14,
    )
    mean = start @ np.linalg.solve(-generator, np.ones(3))
    assert rbd.mean_time_to_first_failure() == pytest.approx(mean, rel=1e-11)


def test_a_series_of_weibulls_is_the_non_repairable_reliability():
    lives = {"a": W([100.0, 2.0]), "b": W([150.0, 1.5]), "c": E([0.002])}
    edges = [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")]
    repairable = RepairableRBD(
        edges, {n: unit(life) for n, life in lives.items()}
    )
    plain = NonRepairableRBD(edges, lives)
    x = np.array([5.0, 40.0, 120.0, 300.0])
    np.testing.assert_allclose(
        repairable.first_failure_sf(x), plain.sf(x), rtol=1e-12
    )
    assert repairable.mean_time_to_first_failure() == pytest.approx(
        plain.mean(), rel=1e-9
    )


def test_a_series_conditions_on_its_components_ages():
    lives = {"a": W([100.0, 2.0]), "b": W([150.0, 1.5])}
    edges = [("s", "a"), ("a", "b"), ("b", "t")]
    rbd = RepairableRBD(edges, {n: unit(life) for n, life in lives.items()})
    x = np.array([10.0, 60.0])
    age = 50.0
    expected = lives["a"].sf(age + x) / lives["a"].sf(age) * lives["b"].sf(x)
    np.testing.assert_allclose(
        rbd.first_failure_sf(x, state={"a": NodeState(age=age)}),
        expected,
        rtol=1e-12,
    )


def test_a_series_of_exponentials_started_stationary():
    # Through the Markov chain: each unit is up with its long-run
    # availability, and fails at its rate.
    rates = {"a": 0.01, "b": 0.03}
    edges = [("s", "a"), ("a", "b"), ("b", "t")]
    rbd = RepairableRBD(edges, {n: unit(E([r])) for n, r in rates.items()})
    x = np.array([0.0, 20.0, 80.0])
    expected = np.ones(3)
    for rate in rates.values():
        expected = expected * (MU / (rate + MU)) * np.exp(-rate * x)
    got = rbd.first_failure_sf(x, state="stationary")
    np.testing.assert_allclose(got, expected, rtol=1e-11)
    assert rbd.mean_time_to_first_failure(state="stationary") == (
        pytest.approx(
            (MU / (0.01 + MU)) * (MU / (0.03 + MU)) / 0.04, rel=1e-11
        )
    )


def test_a_pair_that_is_repaired_at_once_never_fails():
    rbd = RepairableRBD(
        PAIR,
        {n: unit(repair=surv.ExactEventTime.from_params([0.0])) for n in "ab"},
    )
    assert rbd.first_failure_sf([0.0, 1e4]).tolist() == [1.0, 1.0]
    assert rbd.mean_time_to_first_failure() == math.inf


def test_repairs_help_a_redundant_pair_but_not_a_series():
    repaired, never = pair(), NonRepairableRBD(
        PAIR, {n: E([LAM]) for n in "ab"}
    )
    assert repaired.first_failure_sf(300.0) > never.sf(300.0)
    assert repaired.mean_time_to_first_failure() > never.mean()


def test_the_simulation_agrees_with_the_exact_values():
    exact = pair().first_failure_sf(TIMES[1:])
    simulated = pair().first_failure_sf(
        TIMES[1:],
        method="simulate",
        t_simulation=1000.0,
        mc_samples=20_000,
        seed=1,
    )
    error = np.sqrt(exact * (1 - exact) / 20_000)
    assert np.all(np.abs(simulated - exact) < 4 * error)


def test_the_simulated_mean_covers_the_exact_one():
    with pytest.warns(UserWarning, match="lower bound"):
        estimate = pair().mean_time_to_first_failure(
            method="simulate", t_simulation=3000.0, mc_samples=5000, seed=2
        )
    # The mean of min(T, 3000): a lower bound of 650, close to
    # the restricted mean worked out from the survival function.
    grid = np.linspace(0.0, 3000.0, 3001)
    values = pair().first_failure_sf(grid)
    restricted = float(np.sum((values[1:] + values[:-1]) / 2 * np.diff(grid)))
    assert estimate.lower < restricted < estimate.upper


def test_a_redundant_structure_with_weibull_lives_is_simulated_on_request():
    rbd = RepairableRBD(PAIR, {n: unit(W([100.0, 2.0])) for n in "ab"})
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        rbd.first_failure_sf(10.0)
    with pytest.raises(NotImplementedError, match="exponential lives"):
        rbd.mean_time_to_first_failure()
    value = rbd.first_failure_sf(
        [50.0, 150.0],
        method="simulate",
        t_simulation=150.0,
        mc_samples=2000,
        seed=3,
    )
    never = NonRepairableRBD(PAIR, {n: W([100.0, 2.0]) for n in "ab"})
    assert np.all(value >= never.sf([50.0, 150.0]) - 0.03)


def test_what_the_exact_values_do_not_cover_is_refused_with_a_reason():
    scheduled = {
        "reliability": E([LAM]),
        "repairability": E([MU]),
        "preventive": {"interval": 100.0},
    }
    rbd = RepairableRBD(PAIR, {"a": scheduled, "b": unit()})
    with pytest.raises(NotImplementedError, match="scheduled maintenance"):
        rbd.first_failure_sf(10.0)


def test_options_and_times_are_checked():
    with pytest.raises(ValueError, match="only with method='simulate'"):
        pair().first_failure_sf(10.0, mc_samples=10)
    with pytest.raises(ValueError, match="non-negative"):
        pair().first_failure_sf(-1.0)
    with pytest.raises(ValueError, match="method"):
        pair().first_failure_sf(1.0, method="guess")
    with pytest.raises(ValueError, match="t_simulation"):
        pair().mean_time_to_first_failure(method="simulate")


def test_a_scalar_time_gives_a_float():
    assert isinstance(pair().first_failure_sf(10.0), float)


def test_a_cross_linked_diagram_against_a_chain_written_out_by_hand():
    # Five independent units; paths a-c, b-d and a-e-d. The chain over
    # which units are down is built here from the paths alone.
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "d"),
        ("a", "e"),
        ("e", "d"),
        ("c", "t"),
        ("d", "t"),
    ]
    lam = {"a": 0.01, "b": 0.02, "c": 0.015, "d": 0.01, "e": 0.03}
    mu = {"a": 0.2, "b": 0.1, "c": 0.15, "d": 0.2, "e": 0.05}
    rbd = RepairableRBD(edges, {n: unit(E([lam[n]]), E([mu[n]])) for n in lam})
    names = sorted(lam)
    paths = [{"a", "c"}, {"b", "d"}, {"a", "e", "d"}]

    def up(down):
        return any(not (p & down) for p in paths)

    states = [
        frozenset(n for i, n in enumerate(names) if mask >> i & 1)
        for mask in range(2 ** len(names))
    ]
    working = [i for i, st in enumerate(states) if up(st)]
    index = {st: i for i, st in enumerate(states)}
    q = np.zeros((len(states), len(states)))
    for st, i in index.items():
        for n in names:
            j = index[st ^ {n}]
            q[i, j] += mu[n] if n in st else lam[n]
        q[i, i] = -q[i].sum()
    t = q[np.ix_(working, working)]
    start = np.zeros(len(working))
    start[working.index(index[frozenset()])] = 1.0
    x = np.array([10.0, 100.0, 500.0])
    np.testing.assert_allclose(
        rbd.first_failure_sf(x),
        survival(t, start, x),
        rtol=1e-10,
    )
    assert rbd.mean_time_to_first_failure() == pytest.approx(
        start @ np.linalg.solve(-t, np.ones(len(working))), rel=1e-10
    )
