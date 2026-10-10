"""Tests for the acquisition cost, the total cost of ownership and the
lowest-total-cost redundancy allocation of :class:`RepairableRBD`.

Every expectation is a hand calculation, a search over every design, or the
same RBD with the chosen copies drawn out as separate nodes. With Exponential
reliability (rate ``lam``) and repairability (rate ``mu``) a component is
down a fraction ``U = lam / (lam + mu)`` of the time and fails
``omega = lam * mu / (lam + mu)`` times per unit time, so ``n`` independent
copies in parallel are all down ``U ** n`` of the time and fail ``n * omega``
times per unit time.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import CostResult, RepairableRBD, TotalCostAllocation
from repyability.rbd import _costs, redundancy_allocation
from repyability.rbd.redundancy_allocation import lowest_total_cost

E = surv.Exponential.from_params
SERIES = [("s", "pump"), ("pump", "t")]


def spec(rate, repair_rate, **extra):
    out = {"reliability": E([rate]), "repairability": E([repair_rate])}
    out.update(extra)
    return out


def pump_rbd(**extra):
    # Fails every 1000 hours on average, repaired in 10; a pump costs 20,000
    # to buy and 500 per repair, and an hour without pumping costs 100.
    return RepairableRBD(
        SERIES,
        {
            "pump": spec(
                1e-3, 0.1, repair_cost=500.0, acquisition_cost=20000.0, **extra
            )
        },
        downtime_cost_rate=100.0,
    )


def down(rate, repair_rate):
    return rate / (rate + repair_rate)


def failures(rate, repair_rate):
    return rate * repair_rate / (rate + repair_rate)


# -- acquisition cost and the total cost -------------------------------------


def test_acquisition_cost_is_a_one_off_cost():
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": spec(0.1, 1.0, acquisition_cost=300.0, repair_cost=50.0),
            "b": spec(0.2, 1.0, acquisition_cost=200.0),
        },
        downtime_cost_rate=10.0,
    )
    assert rbd.acquisition_costs == {"a": 300.0, "b": 200.0}
    assert rbd.acquisition_cost == 500.0
    rate = rbd.expected_cost_rate()
    # Only the repairs of "a" and the system downtime are running costs.
    availability = (1 - down(0.1, 1.0)) * (1 - down(0.2, 1.0))
    assert rate == pytest.approx(
        50.0 * failures(0.1, 1.0) + 10.0 * (1 - availability), rel=1e-12
    )
    assert rbd.total_cost(1000.0) == pytest.approx(
        500.0 + 1000.0 * rate, rel=1e-12
    )
    assert rbd.total_cost(0.0) == 500.0
    assert rbd.total_cost(10, working_nodes=["a"]) == pytest.approx(
        500.0 + 10 * rbd.expected_cost_rate(working_nodes=["a"])
    )


def test_acquisition_cost_alone_prices_no_running_cost():
    rbd = RepairableRBD(
        SERIES, {"pump": spec(1e-3, 0.1, acquisition_cost=7.0)}
    )
    assert not rbd.has_costs
    assert rbd.expected_cost_rate() == 0.0
    # No running cost, exactly, and the acquisition beside it (#234), as
    # total_cost and expected_cost count it.
    for result in (
        rbd.cost(t_simulation=10.0, mc_samples=5, seed=0),
        rbd.availability(t_simulation=10.0, mc_samples=5, seed=0).cost,
        rbd.availability(
            t_simulation=10.0,
            mc_samples=5,
            seed=0,
            control_variate=False,
            conditional=False,
        ).cost,
    ):
        assert result.mean == 0.0 and result.acquisition_cost == 7.0
        assert result.mean_interval().method == "exact"
        assert list(result.samples) == [0.0] * 5
        assert result.by_category["repair"] == 0.0
    assert rbd.expected_cost(10.0).total == 7.0
    assert rbd.total_cost(1e6) == 7.0
    # With nothing priced at all, there is no cost to give.
    unpriced = RepairableRBD(SERIES, {"pump": spec(1e-3, 0.1)})
    assert unpriced.cost(t_simulation=10.0, mc_samples=5, seed=0) is None
    # A cost of 0 is left out.
    free = RepairableRBD(SERIES, {"pump": spec(1e-3, 0.1, acquisition_cost=0)})
    assert free.acquisition_costs == {}
    assert free.acquisition_cost == 0.0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "x"])
def test_invalid_acquisition_costs_are_rejected(bad):
    with pytest.raises(ValueError, match="acquisition_cost"):
        RepairableRBD(SERIES, {"pump": spec(1e-3, 0.1, acquisition_cost=bad)})


def test_acquisition_cost_cannot_be_a_distribution():
    with pytest.raises(ValueError, match="not a distribution"):
        RepairableRBD(
            SERIES, {"pump": spec(1e-3, 0.1, acquisition_cost=E([1.0]))}
        )


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "x", None])
def test_invalid_horizons_are_rejected(bad):
    rbd = pump_rbd()
    with pytest.raises(ValueError, match="horizon"):
        rbd.total_cost(bad)
    with pytest.raises(ValueError, match="horizon"):
        rbd.allocate_redundancy(bad)


def test_simulated_cost_reports_the_acquisition_cost_separately():
    # Buying the system is not a running cost: the samples are unchanged
    # and the acquisition cost is reported beside them.
    priced = pump_rbd().cost(t_simulation=2000.0, mc_samples=50, seed=1)
    unpriced = RepairableRBD(
        SERIES,
        {"pump": spec(1e-3, 0.1, repair_cost=500.0)},
        downtime_cost_rate=100.0,
    ).cost(t_simulation=2000.0, mc_samples=50, seed=1)
    assert priced.acquisition_cost == 20000.0
    assert unpriced.acquisition_cost == 0.0
    np.testing.assert_array_equal(priced.samples, unpriced.samples)
    assert priced.mean == unpriced.mean
    assert "acquisition_cost" in priced
    # The default for a hand-built result.
    assert CostResult(np.zeros(1), 1.0, 1, {}, {}).acquisition_cost == 0.0


def test_acquisition_cost_survives_a_json_round_trip():
    rbd = pump_rbd()
    restored = RepairableRBD.from_json(rbd.to_json())
    assert restored.acquisition_costs == {"pump": 20000.0}
    assert restored.total_cost(87600.0) == rbd.total_cost(87600.0)
    assert restored.allocate_redundancy(87600.0) == rbd.allocate_redundancy(
        87600.0
    )


# -- one pump: the hand calculation ------------------------------------------


def test_a_second_pump_pays_off_as_the_hand_calculation_says():
    a, c_r, D = 20000.0, 500.0, 100.0
    U, omega = down(1e-3, 0.1), failures(1e-3, 0.1)
    rbd = pump_rbd()

    def by_hand(n, H):
        return n * a + H * (n * c_r * omega + D * U**n)

    for H in (8760.0, 87600.0, 876000.0):
        assert rbd.total_cost(H) == pytest.approx(by_hand(1, H), rel=1e-12)
        best = rbd.allocate_redundancy(H)
        n = min(range(1, 10), key=lambda n: by_hand(n, H))
        assert best.units == {"pump": n}
        assert best.total_cost == pytest.approx(by_hand(n, H), rel=1e-12)
        assert best.acquisition_cost == n * a
        assert best.cost_rate == pytest.approx(
            n * c_r * omega + D * U**n, rel=1e-12
        )
        assert best.availability == pytest.approx(1 - U**n, abs=1e-15)
        assert best.horizon == H
        assert best.method == "exact"
    assert rbd.allocate_redundancy(8760.0).units == {"pump": 1}
    assert rbd.allocate_redundancy(87600.0).units == {"pump": 2}
    # A third pump never pays: it saves at most 100 * U**2 per hour, less
    # than its repairs cost.
    assert D * (U**2 - U**3) < c_r * omega
    assert rbd.allocate_redundancy(876000.0).units == {"pump": 2}

    # The second pump pays for itself from the horizon at which buying and
    # repairing it costs what it saves in downtime.
    break_even = a / (D * (U - U**2) - c_r * omega)
    assert rbd.allocate_redundancy(0.999 * break_even).units == {"pump": 1}
    assert rbd.allocate_redundancy(1.001 * break_even).units == {"pump": 2}


def test_the_result_is_a_typed_mapping():
    best = pump_rbd().allocate_redundancy(87600.0)
    assert isinstance(best, TotalCostAllocation)
    assert best["units"] is best.units
    assert set(best) == {
        "units",
        "total_cost",
        "acquisition_cost",
        "cost_rate",
        "availability",
        "horizon",
        "method",
        "trains",
        "discount_rate",
    }
    assert best.trains is None  # none considered
    assert best.discount_rate == 0.0
    assert best.total_cost == pytest.approx(
        best.acquisition_cost + best.horizon * best.cost_rate, rel=1e-15
    )


def test_a_minimum_availability_can_require_costlier_copies():
    rbd = pump_rbd()
    U = down(1e-3, 0.1)
    best = rbd.allocate_redundancy(87600.0, min_availability=0.9999999)
    # Three pumps are down U**3 ~ 9.7e-7 of the time: not enough.
    assert best.units == {"pump": 4}
    assert best.availability == pytest.approx(1 - U**4, abs=1e-15)
    assert best.availability >= 0.9999999
    greedy = rbd.allocate_redundancy(
        87600.0, min_availability=0.9999999, method="greedy"
    )
    assert greedy.units == {"pump": 4}
    # A limit the cheapest design already meets changes nothing.
    assert rbd.allocate_redundancy(
        87600.0, min_availability=0.99
    ) == rbd.allocate_redundancy(87600.0)


def test_max_units_caps_the_copies():
    rbd = pump_rbd()
    assert rbd.allocate_redundancy(876000.0, max_units=2).units == {"pump": 2}
    assert rbd.allocate_redundancy(876000.0, max_units={"pump": 1}).units == {
        "pump": 1
    }


# -- the design scored is the design drawn out -------------------------------

BRIDGE_EDGES = [
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
BRIDGE_PATHS = [{"a", "d"}, {"b", "c", "d"}, {"b", "e"}]


def drawn_out(edges, components, units, **kwargs):
    """The RBD with node ``x`` replaced by ``units[x]`` parallel copies."""

    def copies(node):
        n = units.get(node, 1)
        return [node] if n == 1 else [f"{node}#{j}" for j in range(n)]

    new_edges = [
        (u, v) for a, b in edges for u in copies(a) for v in copies(b)
    ]
    new_components = {
        copy: component
        for node, component in components.items()
        for copy in copies(node)
    }
    return RepairableRBD(new_edges, new_components, **kwargs)


def mixed_bridge_components():
    weibull = surv.Weibull.from_params([60.0, 2.5])
    return {
        # Plain repairable components, with every kind of running cost.
        "a": spec(
            0.02,
            0.5,
            acquisition_cost=800.0,
            repair_cost=40.0,
            replace_cost=15.0,
            downtime_cost=0.3,
        ),
        "b": spec(0.05, 0.4, acquisition_cost=500.0, repair_cost=25.0),
        # Aged, and replaced preventively at 30.
        "c": {
            "reliability": weibull,
            "repairability": E([0.2]),
            "repair_cost": 60.0,
            "acquisition_cost": 1200.0,
            "preventive": {"interval": 30.0, "cost": 10.0},
        },
        # Hidden failures, proof-tested every 20.
        "d": {
            "reliability": E([0.01]),
            "repairability": "instant",
            "repair_cost": 90.0,
            "acquisition_cost": 1500.0,
            "inspection": {"interval": 20.0, "cost": 5.0},
        },
        # Not bought: counted once, never copied by default.
        "e": spec(0.03, 0.5, repair_cost=10.0),
    }


@pytest.mark.parametrize(
    "horizon, downtime, target",
    [(2000.0, 50.0, None), (20000.0, 200.0, None), (5000.0, 80.0, 0.999)],
)
def test_allocation_scores_match_the_design_drawn_out(
    horizon, downtime, target
):
    components = mixed_bridge_components()
    rbd = RepairableRBD(BRIDGE_EDGES, components, downtime_cost_rate=downtime)
    best = rbd.allocate_redundancy(horizon, min_availability=target)
    assert set(best.units) == {"a", "b", "c", "d"}
    explicit = drawn_out(
        BRIDGE_EDGES, components, best.units, downtime_cost_rate=downtime
    )
    assert best.cost_rate == pytest.approx(
        explicit.expected_cost_rate(), rel=1e-12
    )
    assert best.availability == pytest.approx(
        explicit.mean_availability(), abs=1e-14
    )
    assert best.acquisition_cost == explicit.acquisition_cost
    assert best.total_cost == pytest.approx(
        explicit.total_cost(horizon), rel=1e-12
    )
    if target is not None:
        assert best.availability >= target


def test_every_design_near_the_optimum_scores_as_drawn_out():
    # Capping every node at its count in a design, with a huge downtime
    # cost, makes that design the optimum: so each design is scored.
    components = mixed_bridge_components()
    rbd = RepairableRBD(BRIDGE_EDGES, components, downtime_cost_rate=1e9)
    for counts in [(1, 1, 1, 1), (2, 1, 3, 1), (1, 2, 1, 2), (3, 2, 2, 2)]:
        units = dict(zip("abcd", counts))
        best = rbd.allocate_redundancy(100.0, max_units=units)
        assert best.units == units
        explicit = drawn_out(
            BRIDGE_EDGES, components, units, downtime_cost_rate=1e9
        )
        assert best.cost_rate == pytest.approx(
            explicit.expected_cost_rate(), rel=1e-12
        )
        assert best.availability == pytest.approx(
            explicit.mean_availability(), abs=1e-14
        )


def test_the_chosen_designs_simulated_cost_converges_to_its_rate():
    components = {
        "a": spec(0.1, 1.0, acquisition_cost=100.0, repair_cost=20.0),
        "b": spec(0.05, 0.5, acquisition_cost=150.0, repair_cost=30.0),
    }
    edges = [("s", "a"), ("a", "b"), ("b", "t")]
    rbd = RepairableRBD(edges, components, downtime_cost_rate=300.0)
    best = rbd.allocate_redundancy(1000.0)
    assert best.units == {"a": 3, "b": 3}
    explicit = drawn_out(
        edges, components, best.units, downtime_cost_rate=300.0
    )
    result = explicit.cost(t_simulation=2000.0, mc_samples=300, seed=5)
    assert result.acquisition_cost == best.acquisition_cost
    assert result.cost_rate == pytest.approx(best.cost_rate, rel=0.03)


# -- the exact search against every design -----------------------------------


def bridge_problem(rng):
    """A random priced bridge, and its total cost worked out by hand for
    any design (the structure function from the minimal path sets)."""
    nodes = "abcde"
    rate = {n: rng.uniform(1e-3, 2e-2) for n in nodes}
    repair = {n: rng.uniform(0.05, 0.5) for n in nodes}
    buy = {n: rng.uniform(500.0, 5000.0) for n in nodes}
    per_repair = {n: rng.uniform(0.0, 300.0) for n in nodes}
    own_downtime = {n: rng.choice([0.0, 2.0]) for n in nodes}
    D = rng.uniform(100.0, 2000.0)
    H = rng.uniform(1000.0, 20000.0)
    components = {
        n: spec(
            rate[n],
            repair[n],
            acquisition_cost=buy[n],
            repair_cost=per_repair[n],
            downtime_cost=own_downtime[n],
        )
        for n in nodes
    }
    rbd = RepairableRBD(BRIDGE_EDGES, components, downtime_cost_rate=D)
    U = {n: down(rate[n], repair[n]) for n in nodes}
    copy_cost = {
        n: buy[n]
        + H * (per_repair[n] * failures(rate[n], repair[n]))
        + H * own_downtime[n] * U[n]
        for n in nodes
    }

    def unavailability(units):
        works = 0.0
        for state in itertools.product((0, 1), repeat=len(nodes)):
            up = {n for n, s in zip(nodes, state) if s}
            if any(path <= up for path in BRIDGE_PATHS):
                p = 1.0
                for n, s in zip(nodes, state):
                    q = U[n] ** units[n]
                    p *= (1.0 - q) if s else q
                works += p
        return 1.0 - works

    def total(units):
        return sum(units[n] * copy_cost[n] for n in nodes) + H * D * (
            unavailability(units)
        )

    return rbd, H, total, unavailability, U, copy_cost, D


def useful_copies(U, c, HD):
    # The most copies of a node worth having, whatever the others: beyond
    # it a copy costs more than the most downtime it can save.
    k = 1
    while HD * U**k * (1 - U) > c:
        k += 1
    return k


@pytest.mark.parametrize("seed", range(12))
def test_exact_search_finds_the_best_of_every_design(seed):
    rng = np.random.default_rng(seed)
    rbd, H, total, unavailability, U, copy_cost, D = bridge_problem(rng)
    caps = {n: useful_copies(U[n], copy_cost[n], H * D) for n in "abcde"}
    assert max(caps.values()) <= 6  # the enumeration stays small

    designs = [
        dict(zip("abcde", counts))
        for counts in itertools.product(
            *[range(1, caps[n] + 1) for n in "abcde"]
        )
    ]
    scores = [total(units) for units in designs]
    best = rbd.allocate_redundancy(H)
    assert best.total_cost == pytest.approx(min(scores), rel=1e-9)
    assert total(best.units) == pytest.approx(min(scores), rel=1e-9)
    greedy = rbd.allocate_redundancy(H, method="greedy")
    assert greedy.method == "greedy"
    assert greedy.total_cost >= best.total_cost * (1 - 1e-12)

    # With a minimum availability, among the designs that meet it (within
    # caps that bound the search here).
    target = 1 - 0.1 * unavailability(best.units)
    feasible = [
        (score, units)
        for score, units in zip(scores, designs)
        if 1 - unavailability(units) >= target
    ]
    if feasible:
        limited = rbd.allocate_redundancy(
            H, min_availability=target, max_units=caps
        )
        assert limited.total_cost == pytest.approx(
            min(score for score, _ in feasible), rel=1e-9
        )
        assert limited.availability >= target


def test_the_exact_search_beats_greedy_where_greedy_stops_short():
    # Two nodes in series, each down half the time: a copy of one alone
    # saves 75 * 0.125 = 9.375 of downtime, less than the 10 it costs, so
    # greedy stops at one each; copies of both save 75 * 0.3125 = 23.4,
    # more than the 20 they cost.
    def unavailability(n):
        return 1 - (1 - 0.5 ** n[0]) * (1 - 0.5 ** n[1])

    def gain(k):
        return 0.5**k * 0.5

    def total(n):
        return 10.0 * sum(n) + 75.0 * unavailability(n)

    args = (unavailability, [10.0, 10.0], 75.0, [gain, gain], [3, 3])
    greedy = lowest_total_cost(*args, method="greedy")
    assert greedy[0] == (1, 1)
    assert greedy[1] + 75.0 * greedy[2] == total((1, 1)) == 76.25
    counts, copies, u = lowest_total_cost(*args)
    assert counts == (2, 2)
    assert copies + 75.0 * u == pytest.approx(total((2, 2)), rel=1e-15)
    assert total((2, 2)) == min(
        total(n) for n in itertools.product(range(1, 4), repeat=2)
    )


def test_the_exact_search_needs_every_copy_that_pays():
    # The same two nodes as an RBD: each component is down half the time
    # (failure and repair rates 1) and costs 10, and downtime costs 75 over
    # the horizon. The second copy of each pays only with the other's.
    def half_down():
        return spec(1.0, 1.0, acquisition_cost=10.0)

    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": half_down(), "b": half_down()},
        downtime_cost_rate=75.0,
    )
    best = rbd.allocate_redundancy(1.0)
    assert best.units == {"a": 2, "b": 2}
    assert best.total_cost == pytest.approx(40.0 + 75.0 * (1 - 0.75**2))
    greedy = rbd.allocate_redundancy(1.0, method="greedy")
    assert greedy.units == {"a": 1, "b": 1}
    assert greedy.total_cost == pytest.approx(76.25)


def two_half_down_nodes_in_series(n):
    return 1 - (1 - 0.5 ** n[0]) * (1 - 0.5 ** n[1])


def half_gain(k):
    return 0.5**k * 0.5


@pytest.mark.parametrize(
    "costs, limit, expected",
    [
        # Copies of the cheaper node lower the unavailability as much for a
        # third of the cost: greedy takes more of them.
        ((1.0, 3.0), 0.02, (8, 6)),
        # Greedy overshoots to (12, 11), then drops the copy it no longer
        # needs.
        ((1.0, 3.0), 0.001, (11, 11)),
    ],
)
def test_greedy_meets_a_limit_by_the_largest_fall_per_unit_cost(
    costs, limit, expected
):
    args = (two_half_down_nodes_in_series, costs, 0.0, [half_gain] * 2)
    greedy = lowest_total_cost(*args, [20, 20], limit, "greedy")
    exact = lowest_total_cost(*args, [20, 20], limit)
    assert greedy[0] == exact[0] == expected
    assert greedy[2] <= limit


def test_a_limit_can_need_copies_that_do_not_pay_for_themselves():
    # With no downtime cost no copy pays for itself, but meeting the limit
    # needs copies, and the cheapest way to meet it is not greedy's: nodes
    # down half the time, costing 1 and 2, unavailability at most 0.02.
    args = (two_half_down_nodes_in_series, (1.0, 2.0), 0.0, [half_gain] * 2)
    exact = lowest_total_cost(*args, [20, 20], 0.02)
    greedy = lowest_total_cost(*args, [20, 20], 0.02, "greedy")
    assert exact[:2] == ((8, 6), 20.0)
    assert greedy[:2] == ((7, 7), 21.0)
    cheapest = min(
        n[0] + 2 * n[1]
        for n in itertools.product(range(1, 21), repeat=2)
        if two_half_down_nodes_in_series(n) <= 0.02
    )
    assert cheapest == 20


def test_ties_go_to_fewer_copies():
    # A second copy costs 25 and saves 100 * 0.25 = 25: no better, no worse.
    args = (lambda n: 0.5 ** n[0], [25.0], 100.0, [half_gain], [math.inf])
    assert lowest_total_cost(*args)[0] == (1,)
    assert lowest_total_cost(*args, None, "greedy")[0] == (1,)


# -- the search on its own ---------------------------------------------------


def series_parallel_problem(rng, m):
    qs = rng.uniform(0.05, 0.6, m)
    costs = rng.uniform(1.0, 20.0, m)
    downtime = rng.uniform(50.0, 500.0)

    def unavailability(counts):
        works = 1.0
        for q, n in zip(qs, counts):
            works *= 1.0 - q**n
        return 1.0 - works

    gains = [lambda k, q=q: q**k * (1 - q) for q in qs]
    return unavailability, costs, downtime, gains


def own_unavailability(rng, m):
    # The same problem's nodes: each one's unavailability with n copies.
    qs = np.random.default_rng(rng).uniform(0.05, 0.6, m)
    return [lambda n, q=q: q**n for q in qs]


@pytest.mark.parametrize("seed", range(20))
def test_lowest_total_cost_against_every_design(seed):
    rng = np.random.default_rng(100 + seed)
    m = 3
    unavailability, costs, downtime, gains = series_parallel_problem(rng, m)
    caps = [int(c) for c in rng.integers(2, 6, m)]
    limit = None if seed % 2 else float(rng.uniform(0.001, 0.2))
    designs = list(itertools.product(*[range(1, c + 1) for c in caps]))
    scored = [
        (float(np.dot(n, costs)) + downtime * unavailability(n), n)
        for n in designs
        if limit is None or unavailability(n) <= limit
    ]
    if not scored:
        with pytest.raises(ValueError, match="cannot be brought down"):
            lowest_total_cost(
                unavailability, costs, downtime, gains, caps, limit
            )
        return
    counts, copies, u = lowest_total_cost(
        unavailability, costs, downtime, gains, caps, limit
    )
    assert copies + downtime * u == pytest.approx(min(scored)[0], rel=1e-12)
    assert copies == pytest.approx(float(np.dot(counts, costs)))
    assert u == unavailability(counts)
    assert all(1 <= n <= c for n, c in zip(counts, caps))
    if limit is not None:
        assert u <= limit
    greedy = lowest_total_cost(
        unavailability, costs, downtime, gains, caps, limit, "greedy"
    )
    assert greedy[1] + downtime * greedy[2] >= min(scored)[0] * (1 - 1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_uncapped_search_needs_no_caps(seed):
    # Without caps the useful-copies bound limits the search: the optimum
    # is the best design within those bounds.
    rng = np.random.default_rng(200 + seed)
    unavailability, costs, downtime, gains = series_parallel_problem(rng, 3)
    counts, copies, u = lowest_total_cost(
        unavailability, costs, downtime, gains, [math.inf] * 3
    )
    bounds = []
    for c, gain in zip(costs, gains):
        k = 1
        while downtime * gain(k) > c:
            k += 1
        bounds.append(k)
    best = min(
        float(np.dot(n, costs)) + downtime * unavailability(n)
        for n in itertools.product(*[range(1, k + 1) for k in bounds])
    )
    assert copies + downtime * u == pytest.approx(best, rel=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_the_series_program_agrees_with_the_branch_and_bound(seed):
    # The nodes are in series: the dynamic program over them finds the
    # same optimum as the search over every design, with or without a
    # limit on the unavailability.
    m = 4
    rng = np.random.default_rng(300 + seed)
    unavailability, costs, downtime, gains = series_parallel_problem(rng, m)
    series = own_unavailability(300 + seed, m)
    limit = None if seed % 2 else float(rng.uniform(0.001, 0.05))
    caps = [8] * m
    if limit is not None and unavailability(caps) > limit:
        for way in (None, series):
            with pytest.raises(ValueError, match="cannot be brought down"):
                lowest_total_cost(
                    unavailability,
                    costs,
                    downtime,
                    gains,
                    caps,
                    limit,
                    "exact",
                    way,
                )
        return
    searched = lowest_total_cost(
        unavailability, costs, downtime, gains, caps, limit
    )
    programmed = lowest_total_cost(
        unavailability, costs, downtime, gains, caps, limit, "exact", series
    )
    assert programmed[1] + downtime * programmed[2] == pytest.approx(
        searched[1] + downtime * searched[2], rel=1e-12
    )
    if limit is not None:
        assert programmed[2] <= limit
    # Without caps too (the useful copies bound the program's menus).
    if limit is None:
        uncapped = lowest_total_cost(
            unavailability,
            costs,
            downtime,
            gains,
            [math.inf] * m,
            None,
            "exact",
            series,
        )
        assert uncapped[1] + downtime * uncapped[2] == pytest.approx(
            searched[1] + downtime * searched[2], rel=1e-12
        )


def test_the_series_program_beats_greedy_to_a_limit():
    series = [lambda n: 0.5**n] * 2
    args = (two_half_down_nodes_in_series, (1.0, 2.0), 0.0, [half_gain] * 2)
    exact = lowest_total_cost(*args, [20, 20], 0.02, "exact", series)
    assert exact[:2] == ((8, 6), 20.0)
    uncapped = lowest_total_cost(*args, [math.inf] * 2, 0.02, "exact", series)
    assert uncapped[:2] == ((8, 6), 20.0)


def test_the_series_program_gives_up_when_too_large(monkeypatch):
    monkeypatch.setattr(redundancy_allocation, "SERIES_STATE_LIMIT", 5)
    series = [lambda n: 0.5**n] * 2
    args = (two_half_down_nodes_in_series, (1.0, 2.0), 0.0, [half_gain] * 2)
    with pytest.raises(ValueError, match="dynamic program held more"):
        lowest_total_cost(*args, [20, 20], 0.02, "exact", series)


def test_free_uncapped_copies_are_rejected():
    with pytest.raises(ValueError, match="costs nothing"):
        lowest_total_cost(
            lambda n: 0.5 ** n[0], [0.0], 10.0, [lambda k: 0.5**k], [math.inf]
        )
    # Capped, the free copies are all taken while they help.
    counts, copies, u = lowest_total_cost(
        lambda n: 0.5 ** n[0], [0.0], 10.0, [lambda k: 0.5**k], [4]
    )
    assert counts == (4,) and copies == 0.0 and u == 0.5**4


def test_a_limit_reached_only_in_the_limit_is_out_of_reach():
    with pytest.raises(ValueError, match="cannot be brought down"):
        lowest_total_cost(
            lambda n: 0.5 ** n[0],
            [1.0],
            10.0,
            [lambda k: 0.5**k],
            [math.inf],
            max_unavailability=0.0,
        )


def test_the_exact_search_gives_up_when_too_large(monkeypatch):
    monkeypatch.setattr(redundancy_allocation, "EXACT_SEARCH_LIMIT", 20)
    rng = np.random.default_rng(0)
    unavailability, costs, downtime, gains = series_parallel_problem(rng, 4)
    with pytest.raises(ValueError, match="without finishing"):
        lowest_total_cost(unavailability, costs, 1e9, gains, [8] * 4)
    # Greedy does not count its steps against the limit.
    lowest_total_cost(
        unavailability, costs, 1e9, gains, [8] * 4, None, "greedy"
    )


@pytest.mark.parametrize("target", [None, 0.9931])
def test_nodes_in_series_with_an_imperfect_rest(target):
    # "x" and "y" lie on every path, in series with a bridge that is not
    # bought: the system is up when both are and the bridge is. Scored
    # against every design, and against the design drawn out.
    edges = [("s", "x"), ("x", "a"), ("x", "b")] + [
        (u, v) for u, v in BRIDGE_EDGES if u != "s"
    ]
    edges = [(u, "y") if v == "t" else (u, v) for u, v in edges]
    edges.append(("y", "t"))
    bridge = {n: spec(0.02, 0.4) for n in "abcde"}
    components = {
        "x": spec(0.01, 0.2, acquisition_cost=900.0, repair_cost=40.0),
        "y": spec(0.03, 0.5, acquisition_cost=400.0, repair_cost=10.0),
        **bridge,
    }
    rbd = RepairableRBD(edges, components, downtime_cost_rate=500.0)
    H = 10000.0
    best = rbd.allocate_redundancy(H, min_availability=target)
    explicit = drawn_out(
        edges, components, best.units, downtime_cost_rate=500.0
    )
    assert best.total_cost == pytest.approx(explicit.total_cost(H), rel=1e-12)
    assert best.availability == pytest.approx(
        explicit.mean_availability(), abs=1e-14
    )

    rest = rbd.mean_availability(working_nodes=["x", "y"])

    def by_hand(nx, ny):
        Ux, Uy = down(0.01, 0.2), down(0.03, 0.5)
        availability = rest * (1 - Ux**nx) * (1 - Uy**ny)
        total = (
            nx * (900.0 + H * 40.0 * failures(0.01, 0.2))
            + ny * (400.0 + H * 10.0 * failures(0.03, 0.5))
            + H * 500.0 * (1 - availability)
            + H
            * _costs._node_cost_rate(rbd, "a", 1.0)  # the bridge costs nothing
        )
        return total, availability

    scored = [
        (by_hand(nx, ny), (nx, ny)) for nx in range(1, 8) for ny in range(1, 8)
    ]
    (total, availability), units = min(
        (s, u) for s, u in scored if target is None or s[1] >= target
    )
    assert (best.units["x"], best.units["y"]) == units
    assert best.total_cost == pytest.approx(total, rel=1e-12)
    assert best.availability == pytest.approx(availability, abs=1e-14)
    # The limit binds: without it three of each would do.
    assert (units == (3, 3)) == (target is None)


def test_nodes_in_series_with_inspected_nodes_elsewhere():
    # "x" is in series with a proof-tested channel "d": the rest of the
    # system varies over the test interval, "x" does not, so the series
    # program applies; the result is the design drawn out.
    components = {
        "x": spec(0.02, 0.5, acquisition_cost=300.0, repair_cost=20.0),
        "d": {
            "reliability": E([0.01]),
            "repairability": "instant",
            "inspection": {"interval": 10.0, "cost": 1.0},
        },
    }
    edges = [("s", "x"), ("x", "d"), ("d", "t")]
    rbd = RepairableRBD(edges, components, downtime_cost_rate=400.0)
    for target in (None, 0.95):
        best = rbd.allocate_redundancy(5000.0, min_availability=target)
        explicit = drawn_out(
            edges, components, best.units, downtime_cost_rate=400.0
        )
        assert best.units["x"] > 1
        assert best.cost_rate == pytest.approx(
            explicit.expected_cost_rate(), rel=1e-12
        )
        assert best.availability == pytest.approx(
            explicit.mean_availability(), abs=1e-14
        )


def test_the_series_program_is_used_exactly_when_it_applies(monkeypatch):
    from repyability.rbd import _repairable_allocation

    calls = []

    def recording(*args):
        calls.append(args[7])
        return lowest_total_cost(*args)

    monkeypatch.setattr(_repairable_allocation, "lowest_total_cost", recording)
    inspected = {
        "reliability": E([0.01]),
        "repairability": "instant",
        "inspection": {"interval": 10.0},
    }
    x = spec(0.02, 0.5, acquisition_cost=300.0)
    edges = [("s", "x"), ("x", "d"), ("d", "t")]

    # One pump, in series: each copy count's unavailability is U ** n.
    pump_rbd().allocate_redundancy(1000.0)
    U = down(1e-3, 0.1)
    assert [calls[-1][0](n) for n in (1, 2, 3)] == pytest.approx(
        [U, U**2, U**3], rel=1e-15
    )
    # "x" in series with an inspected node that is not bought.
    rbd = RepairableRBD(edges, {"x": x, "d": inspected}, downtime_cost_rate=9)
    rbd.allocate_redundancy(1000.0)
    assert calls[-1][0](2) == pytest.approx(down(0.02, 0.5) ** 2, rel=1e-15)
    # An inspected node's availability varies between tests: not separable.
    rbd.allocate_redundancy(1000.0, nodes=["x", "d"], max_units=2)
    assert calls[-1] is None
    # Not in series.
    RepairableRBD(
        BRIDGE_EDGES, mixed_bridge_components(), downtime_cost_rate=1.0
    ).allocate_redundancy(10.0, max_units=2)
    assert calls[-1] is None
    assert len(calls) == 4


# -- invalid requests --------------------------------------------------------


def test_invalid_requests_are_rejected():
    rbd = pump_rbd()
    with pytest.raises(ValueError, match="method"):
        rbd.allocate_redundancy(100.0, method="best")
    with pytest.raises(ValueError, match="at least one"):
        rbd.allocate_redundancy(100.0, nodes=[])
    with pytest.raises(ValueError, match="not a component"):
        rbd.allocate_redundancy(100.0, nodes=["valve"])
    with pytest.raises(ValueError, match="not a component"):
        rbd.allocate_redundancy(100.0, nodes=["s"])
    with pytest.raises(ValueError, match="at least 1"):
        rbd.allocate_redundancy(100.0, max_units=0)
    with pytest.raises(ValueError, match="integer"):
        rbd.allocate_redundancy(100.0, max_units=1.5)
    with pytest.raises(ValueError, match="not in nodes"):
        rbd.allocate_redundancy(100.0, max_units={"valve": 2})
    for bad in (0.0, 1.0, 1.5, -0.1, "x", float("nan")):
        with pytest.raises(ValueError, match="min_availability"):
            rbd.allocate_redundancy(100.0, min_availability=bad)
    with pytest.raises(ValueError, match="cannot be brought down"):
        rbd.allocate_redundancy(100.0, min_availability=0.999, max_units=1)


def test_nodes_default_to_the_components_with_a_price():
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": spec(0.1, 1.0), "b": spec(0.1, 1.0)},
        downtime_cost_rate=10.0,
    )
    with pytest.raises(ValueError, match="No component has an acquisition"):
        rbd.allocate_redundancy(100.0)
    # A component whose copies cost nothing needs a cap.
    with pytest.raises(ValueError, match="A copy of 'a' costs nothing"):
        rbd.allocate_redundancy(100.0, nodes=["a"])
    best = rbd.allocate_redundancy(100.0, nodes=["a"], max_units=3)
    assert best.units == {"a": 3}
    # Nodes are considered once, in the order given.
    best = rbd.allocate_redundancy(
        100.0, nodes=["b", "a", "b"], max_units={"a": 1, "b": 2}
    )
    assert list(best.units) == ["b", "a"]


def test_nested_rbds_cannot_be_copied():
    inner = pump_rbd()
    rbd = RepairableRBD(
        [("s", "sub"), ("sub", "x"), ("x", "t")],
        {"sub": inner, "x": spec(0.1, 1.0, acquisition_cost=5.0)},
        downtime_cost_rate=1.0,
    )
    with pytest.raises(ValueError, match="nested RepairableRBD"):
        rbd.allocate_redundancy(100.0, nodes=["sub"])
    # Its costs are not this RBD's: only "x" is bought here.
    assert rbd.acquisition_cost == 5.0
    assert set(rbd.allocate_redundancy(100.0).units) == {"x"}


def test_block_replacement_of_a_memoryless_pump_changes_nothing():
    # An exponential pump is as good as new at any age: replacing it at the
    # block times, in no time and at no cost, changes none of its costs.
    plain = pump_rbd()
    blocked = pump_rbd(preventive={"interval": 500.0, "policy": "block"})
    assert blocked.total_cost(1000.0) == pytest.approx(
        plain.total_cost(1000.0), rel=1e-6
    )
    assert (
        blocked.allocate_redundancy(1000.0).units
        == plain.allocate_redundancy(1000.0).units
    )
