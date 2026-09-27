"""
Tests allocation cases for the RBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import numpy as np
import pytest
from numpy.random import rand
from scipy.optimize import minimize

from repyability.rbd.rbd import RBD


# Test that allocation to a series system works.
def test_series_allocation():
    two_inputs_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
    ]
    rbd = RBD(two_inputs_edges)
    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        achieved = rbd.system_probability(allocated_probs)
        assert pytest.approx(achieved, rel=1e-3) == target

    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        assert pytest.approx(allocated_probs[1], rel=1e-3) == target ** (1 / 6)


# Test that allocation to a parallel system works.
def test_parallel_allocation():
    two_inputs_edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    rbd = RBD(two_inputs_edges)
    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        achieved = rbd.system_probability(allocated_probs)
        assert pytest.approx(achieved, rel=1e-3) == target

    for i in range(10):
        target = rand()
        allocated_probs = rbd.simple_allocation(target)
        assert pytest.approx(allocated_probs[1], rel=1e-3) == 1 - (
            1 - target
        ) ** (1 / 6)


def _series():
    return RBD([("s", 1), (1, 2), (2, 3), (3, "t")])


def test_improvement_allocation_below_the_current_reliability_stays_valid():
    # The current system probability is 0.9 * 0.8 * 0.7 = 0.504. A lower
    # target lowers the node probabilities, each capped at a failure
    # probability of 1; the old solver returned negative probabilities.
    rbd = _series()
    new = rbd.improvement_allocation(0.3, {1: 0.9, 2: 0.8, 3: 0.7})
    assert all(0.0 <= p <= 1.0 for p in new.values())
    assert rbd.system_probability(new).item() == pytest.approx(0.3)
    factors = {(1 - new[n]) / (1 - p) for n, p in {1: 0.9, 2: 0.8}.items()}
    assert max(factors) == pytest.approx(min(factors))


def test_improvement_allocation_raises_for_an_unreachable_target():
    # With nodes 1 and 2 fixed the system cannot beat 0.9 * 0.8 = 0.72.
    with pytest.raises(ValueError, match="cannot be reached.*0.72"):
        _series().improvement_allocation(
            0.9, {1: 0.9, 2: 0.8, 3: 0.7}, fixed=[1, 2]
        )


def test_allocation_to_a_target_of_one_makes_the_nodes_perfect():
    rbd = _series()
    assert rbd.equal_allocation(1.0) == {1: 1.0, 2: 1.0, 3: 1.0}
    assert rbd.res.x[0] == float("inf")
    assert rbd.equal_allocation(0.0) == {1: 0.0, 2: 0.0, 3: 0.0}


@pytest.mark.parametrize("value", [[0.9, 0.8], 1.2, -0.1])
def test_improvement_allocation_rejects_invalid_node_probabilities(value):
    with pytest.raises(ValueError, match="node_probabilities"):
        _series().improvement_allocation(0.9, {1: value})


def test_simple_allocation_raises_when_the_target_is_out_of_reach():
    # A node with weight 0 stays at 0.5, so a series system cannot beat
    # 0.5; the closest miss used to come back with res.success True.
    rbd = _series()
    with pytest.raises(ValueError, match="did not reach target 0.9"):
        rbd.simple_allocation(0.9, weights={1: 0.0, 2: 1.0, 3: 1.0})
    assert rbd.res is not None


def test_simple_allocation_raises_when_the_search_stalls():
    # Thirty nodes in parallel at 0.5 each work with probability
    # 1 - 0.5 ** 30, so near 1 that the search cannot move: it used to
    # return every node at 0.5 for a target of 0.5.
    parallel = RBD(
        [("s", i) for i in range(30)] + [(i, "t") for i in range(30)]
    )
    with pytest.raises(ValueError, match="improvement_allocation"):
        parallel.simple_allocation(0.5)
    # The exact method has no such limit.
    new = parallel.equal_allocation(0.5)
    assert parallel.system_probability(new).item() == pytest.approx(0.5)


@pytest.mark.parametrize("target", [0.0, 1e-6, 0.5, 0.999999, 1.0])
def test_simple_allocation_meets_reachable_targets(target):
    rbd = RBD([("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")])
    new = rbd.simple_allocation(target)
    achieved = rbd.system_probability(new).item()
    tail = min(target, 1 - target)
    assert abs(achieved - target) <= (1e-6 * tail if tail else 1e-6)


# -- Named methods: minimum effort (Albert) and cost-based (Mettas) --------


def _chain(names):
    edges = [("s", names[0])] + list(zip(names, names[1:]))
    return RBD(edges + [(names[-1], "t")])


def _pumps_and_valve():
    return RBD(
        [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")]
    )


def test_minimum_effort_raises_the_weakest_nodes_to_a_common_level():
    rbd = _chain("abcd")
    current = {"a": 0.7, "b": 0.8, "c": 0.9, "d": 0.95}
    new = rbd.minimum_effort_allocation(0.6, current)
    level = (0.6 / (0.9 * 0.95)) ** 0.5
    assert new == pytest.approx({"a": level, "b": level, "c": 0.9, "d": 0.95})
    assert rbd.system_probability(new).item() == pytest.approx(0.6)


@pytest.mark.parametrize(
    "effort",
    [
        lambda x, y: y - x,
        lambda x, y: np.log((1 - x) / (1 - y)),
    ],
    ids=["linear", "log unreliability"],
)
def test_minimum_effort_is_the_least_effort(effort):
    # Albert's result holds for any effort function meeting his
    # conditions: compare with a direct minimisation of two of them.
    names = "abcdef"
    rbd = _chain(names)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(0.6, 0.99, len(names))
        target = 0.5 * (np.prod(x) + 1.0)
        res = minimize(
            lambda y: sum(effort(x, y)),
            x,
            method="SLSQP",
            bounds=[(xi, 1 - 1e-12) for xi in x],
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda y: np.log(y).sum() - np.log(target),
                }
            ],
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        new = rbd.minimum_effort_allocation(target, dict(zip(names, x)))
        assert [new[n] for n in names] == pytest.approx(res.x, abs=1e-5)


def test_minimum_effort_edge_cases():
    rbd = _chain("abc")
    current = {"a": 0.9, "b": 0.8, "c": 0.95}
    assert rbd.minimum_effort_allocation(0.5, current) == current  # met
    assert rbd.minimum_effort_allocation(1.0, current) == {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
    }
    with pytest.raises(ValueError, match="series"):
        _pumps_and_valve().minimum_effort_allocation(
            0.99, {"p1": 0.9, "p2": 0.9, "v": 0.9}
        )
    with pytest.raises(ValueError, match="missing"):
        rbd.minimum_effort_allocation(0.9, {"a": 0.9, "b": 0.9})


def _mettas_cost(r, r_min, r_max, f):
    return np.exp((1 - f) * (r - r_min) / (r_max - r))


def _bridge():
    return RBD(
        [
            ("s", 1),
            ("s", 2),
            (1, 3),
            (2, 3),
            (1, 4),
            (3, 4),
            (2, 5),
            (3, 5),
            (4, "t"),
            (5, "t"),
        ],
        on_infeasible_rbd="ignore",
    )


_BRIDGE_CURRENT = {1: 0.8, 2: 0.7, 3: 0.6, 4: 0.85, 5: 0.75}
_BRIDGE_MAX = {1: 0.99, 2: 0.995, 3: 0.98, 4: 0.999, 5: 0.99}
_BRIDGE_EASE = {1: 0.9, 2: 0.5, 3: 0.2, 4: 0.7, 5: 0.4}


def test_cost_based_allocation_matches_a_direct_minimisation():
    # Mettas's problem solved directly in the node reliabilities.
    rbd = _bridge()
    nodes = list(rbd.nodes)
    r_min = np.array([_BRIDGE_CURRENT[n] for n in nodes])
    r_max = np.array([_BRIDGE_MAX[n] for n in nodes])
    f = np.array([_BRIDGE_EASE[n] for n in nodes])

    def system(r):
        return rbd.system_probability(dict(zip(nodes, r))).item()

    # The textbook form overflows near the maxima (the library works with
    # the log of the cost instead); the solver copes with the inf.
    with np.errstate(over="ignore"):
        direct = _direct_minimisation(system, r_min, r_max, f)
    new = rbd.cost_based_allocation(
        0.97, _BRIDGE_CURRENT, _BRIDGE_MAX, _BRIDGE_EASE
    )
    assert [new[n] for n in nodes] == pytest.approx(direct.x, abs=1e-5)
    assert rbd.system_probability(new).item() >= 0.97 - 1e-12


def _direct_minimisation(system, r_min, r_max, f):
    return minimize(
        lambda r: _mettas_cost(r, r_min, r_max, f).sum(),
        r_min,
        method="SLSQP",
        bounds=list(zip(r_min, r_max - 1e-9)),
        constraints=[{"type": "ineq", "fun": lambda r: system(r) - 0.97}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )


def test_cost_based_allocation_meets_the_optimality_conditions():
    # At the optimum every improved node buys system reliability at the
    # same marginal cost, c_i'(R_i) / I_B(i) = lambda, and a node left at
    # its current value would cost at least that much to improve.
    rbd = _bridge()
    new = rbd.cost_based_allocation(
        0.97, _BRIDGE_CURRENT, _BRIDGE_MAX, _BRIDGE_EASE
    )
    ratios, improved = [], []
    for n in rbd.nodes:
        r, r0, r1, f = (
            new[n],
            _BRIDGE_CURRENT[n],
            _BRIDGE_MAX[n],
            _BRIDGE_EASE[n],
        )
        marginal = (
            _mettas_cost(r, r0, r1, f) * (1 - f) * (r1 - r0) / (r1 - r) ** 2
        )
        up = rbd.system_probability({**new, n: 1.0}).item()
        down = rbd.system_probability({**new, n: 0.0}).item()
        ratios.append(marginal / (up - down))
        improved.append(r > _BRIDGE_CURRENT[n] + 1e-9)
    ratios = np.array(ratios)
    improved = np.array(improved)
    assert improved.sum() >= 3  # the hard middle node (3) stays put
    common = ratios[improved].mean()
    assert ratios[improved] == pytest.approx(common, rel=1e-4)
    assert np.all(ratios[~improved] >= common * (1 - 1e-4))


def test_cost_based_allocation_feasibility_and_symmetry():
    rbd = _chain("abc")
    current = {"a": 0.9, "b": 0.9, "c": 0.9}
    same = rbd.cost_based_allocation(0.95, current)
    assert same["a"] == pytest.approx(same["b"]) == pytest.approx(same["c"])
    assert rbd.system_probability(same).item() == pytest.approx(0.95)
    # A node that is harder to improve is improved less.
    harder = rbd.cost_based_allocation(0.95, current, feasibility={"c": 0.1})
    assert harder["c"] < harder["a"] == pytest.approx(harder["b"])


def test_cost_based_allocation_bounds():
    rbd = _pumps_and_valve()
    current = {"p1": 0.9, "p2": 0.9, "v": 0.9}
    # Already met: nothing changes.
    assert rbd.cost_based_allocation(0.5, current) == current
    # Maxima are respected, and a node held at its current value stays.
    new = rbd.cost_based_allocation(
        0.98, current, max_probabilities={"p1": 0.9, "v": 0.999}
    )
    assert new["p1"] == 0.9 and new["v"] < 0.999
    assert rbd.system_probability(new).item() == pytest.approx(0.98)
    # The valve caps the system: the reachable maximum is only approached.
    with pytest.raises(ValueError, match="cannot be reached"):
        rbd.cost_based_allocation(0.99, current, max_probabilities={"v": 0.99})
    with pytest.raises(ValueError, match="cannot be reached"):
        rbd.cost_based_allocation(1.0, current)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"feasibility": {"v": 1.0}}, "feasibility"),
        ({"max_probabilities": {"v": 0.8}}, "below"),
        ({"feasibility": {"x": 0.5}}, "not intermediate"),
    ],
    ids=["feasibility of 1", "maximum below current", "unknown node"],
)
def test_cost_based_allocation_validation(kwargs, message):
    rbd = _pumps_and_valve()
    with pytest.raises(ValueError, match=message):
        rbd.cost_based_allocation(
            0.95, {"p1": 0.9, "p2": 0.9, "v": 0.9}, **kwargs
        )


@pytest.mark.parametrize("shape", ["series", "parallel"])
def test_cost_based_allocation_on_large_systems(shape):
    # Where the least-squares heuristic stalls, the cost-based method is
    # exact: 100 nodes in series, or 30 in parallel.
    if shape == "series":
        n, start, target = 100, 0.99, 0.9
        rbd = _chain(list(range(n)))
    else:
        n, start, target = 30, 0.01, 0.9
        rbd = RBD([("s", i) for i in range(n)] + [(i, "t") for i in range(n)])
    new = rbd.cost_based_allocation(target, {i: start for i in range(n)})
    assert rbd.system_probability(new).item() == pytest.approx(target)
    assert max(new.values()) == pytest.approx(min(new.values()))
