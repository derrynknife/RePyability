"""Redundancy allocation of trains (#184): copies of a chain of components,
each another path alongside it into the node it feeds.

The reference is the design drawn by hand: the diagram with the copies as
components of their own, whose ``total_cost`` and ``mean_availability`` the
allocation must reproduce, and the cheapest of every design within the caps,
found by trying them all.
"""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import PerfectReliability, RepairableRBD

E = surv.Exponential.from_params
HORIZON = 87_600.0


def unit(mtbf, mttr, acquisition, repair=0.0, **extra):
    spec = {
        "reliability": E([1 / mtbf]),
        "repairability": E([1 / mttr]),
        "acquisition_cost": acquisition,
        **extra,
    }
    if repair:
        spec["repair_cost"] = repair
    return spec


PUMP = unit(1000.0, 10.0, 20_000.0, repair=500.0)
MOTOR = unit(5000.0, 20.0, 10_000.0)


def station(trains, k=2, rate=2000.0, valves=0, pump=PUMP, motor=MOTOR):
    """A station of ``trains`` pump trains (pump, then motor) needing ``k``,
    behind ``valves`` header valves in parallel (none: none)."""
    edges, components = [], {}
    feeds = ["s"]
    if valves:
        feeds = [f"valve {j}" for j in range(1, valves + 1)]
        edges += [("s", v) for v in feeds]
        components.update(
            {v: unit(20_000.0, 5.0, 3_000.0, repair=200.0) for v in feeds}
        )
    for i in range(1, trains + 1):
        edges += [(f, f"pump {i}") for f in feeds]
        edges += [(f"pump {i}", f"motor {i}"), (f"motor {i}", "t")]
        components[f"pump {i}"] = pump
        components[f"motor {i}"] = motor
    return RepairableRBD(
        edges, components, k={"t": k}, downtime_cost_rate=rate
    )


TRAIN = {"train 1": ["pump 1", "motor 1"]}


def same_design(best, drawn):
    assert best.total_cost == pytest.approx(
        drawn.total_cost(HORIZON), rel=1e-12
    )
    assert best.availability == pytest.approx(
        drawn.mean_availability(), rel=1e-12
    )
    assert best.acquisition_cost == pytest.approx(
        sum(drawn.acquisition_costs.values()), rel=1e-12
    )


@pytest.mark.parametrize("rate", [20.0, 2000.0, 200_000.0])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_trains_copies_join_its_vote(rate, k):
    base = station(3, k=k, rate=rate)
    best = base.allocate_redundancy(HORIZON, trains=TRAIN, max_units=5)
    # Every design within the cap, drawn by hand: the copies are trains of
    # their own, k of all of them needed.
    totals = {
        n: station(2 + n, k=k, rate=rate).total_cost(HORIZON)
        for n in range(1, 6)
    }
    cheapest = min(totals, key=totals.get)
    assert best.units == {"train 1": cheapest}
    assert best.trains == {"train 1": ["pump 1", "motor 1"]}
    same_design(best, station(2 + cheapest, k=k, rate=rate))
    # Unlimited, the exact search proves the same (no sixth train pays).
    unlimited = base.allocate_redundancy(HORIZON, trains=TRAIN)
    assert unlimited.units == best.units
    greedy = base.allocate_redundancy(HORIZON, trains=TRAIN, method="greedy")
    assert greedy.total_cost >= best.total_cost * (1 - 1e-12)
    same_design(greedy, station(2 + greedy.units["train 1"], k=k, rate=rate))


def test_copies_of_its_nodes_are_not_another_train():
    base = station(3)
    trained = base.allocate_redundancy(HORIZON, trains=TRAIN)
    alone = base.allocate_redundancy(HORIZON, nodes=["pump 1", "motor 1"])
    # Copies in parallel with train 1's own nodes stand in for its nodes
    # only; a fourth train stands in for any.
    assert trained.units == {"train 1": 2}
    assert alone.units == {"pump 1": 1, "motor 1": 2}
    assert trained.total_cost < alone.total_cost
    assert round(trained.total_cost) == 295306
    assert round(base.total_cost(HORIZON)) == 319927


@pytest.mark.parametrize("rate", [500.0, 5000.0, 50_000.0])
def test_nodes_and_trains_together(rate):
    # A header valve before the trains, copied in parallel, and the trains.
    base = station(3, rate=rate, valves=1)
    best = base.allocate_redundancy(
        HORIZON, nodes=["valve 1"], trains=TRAIN, max_units=4
    )
    totals = {
        (v, n): station(2 + n, rate=rate, valves=v).total_cost(HORIZON)
        for v, n in itertools.product(range(1, 5), range(1, 5))
    }
    v, n = min(totals, key=totals.get)
    assert best.units == {"valve 1": v, "train 1": n}
    same_design(best, station(2 + n, rate=rate, valves=v))


def test_a_target_availability():
    base = station(3, rate=10.0)
    assert base.allocate_redundancy(HORIZON, trains=TRAIN).units == {
        "train 1": 1
    }
    target = station(5, rate=10.0).mean_availability() * (1 - 1e-12)
    best = base.allocate_redundancy(
        HORIZON, trains=TRAIN, min_availability=target
    )
    # Four trains fall short of it: five are needed.
    assert station(4, rate=10.0).mean_availability() < target
    assert best.units == {"train 1": 3}
    same_design(best, station(5, rate=10.0))


def test_random_trains_find_the_cheapest_design():
    # Random votes, lives, repairs and costs: the exact search agrees with
    # trying every design.
    rng = np.random.default_rng(184)
    for _ in range(12):
        n_trains = int(rng.integers(1, 4))
        k = int(rng.integers(1, n_trains + 1))
        pump = unit(
            float(rng.uniform(200, 5000)),
            float(rng.uniform(1, 50)),
            float(rng.uniform(1000, 50_000)),
            repair=float(rng.uniform(0, 2000)),
        )
        motor = unit(
            float(rng.uniform(1000, 20_000)),
            float(rng.uniform(1, 50)),
            float(rng.uniform(1000, 20_000)),
        )
        rate = float(10 ** rng.uniform(1, 5))
        build = lambda n: station(  # noqa: E731
            n, k=k, rate=rate, pump=pump, motor=motor
        )
        best = build(n_trains).allocate_redundancy(
            HORIZON, trains=TRAIN, max_units=6
        )
        totals = {
            n: build(n_trains - 1 + n).total_cost(HORIZON) for n in range(1, 7)
        }
        assert best.total_cost == pytest.approx(
            min(totals.values()), rel=1e-12
        )
        same_design(best, build(n_trains - 1 + best.units["train 1"]))


def test_a_train_fed_by_several_nodes():
    # Two header valves in parallel each feed every train: a copy is fed by
    # both, as the train it copies.
    base = station(3, valves=2)
    best = base.allocate_redundancy(HORIZON, trains=TRAIN)
    same_design(best, station(2 + best.units["train 1"], valves=2))
    assert best.units == {"train 1": 2}


def test_a_train_of_one_node():
    pump = unit(1000.0, 10.0, 20_000.0, repair=500.0)

    def pumps(n):
        return RepairableRBD(
            [("s", f"p{i}") for i in range(1, n + 1)]
            + [(f"p{i}", "t") for i in range(1, n + 1)],
            {f"p{i}": pump for i in range(1, n + 1)},
            k={"t": 2},
            downtime_cost_rate=20_000.0,
        )

    best = pumps(3).allocate_redundancy(
        HORIZON, trains={"pumps": ["p1"]}, max_units=4
    )
    totals = {n: pumps(2 + n).total_cost(HORIZON) for n in range(1, 5)}
    assert best.units == {"pumps": min(totals, key=totals.get)} == {"pumps": 2}
    same_design(best, pumps(4))
    # The node's own copies, in parallel with it, are another design.
    alone = pumps(3).allocate_redundancy(HORIZON, nodes=["p1"])
    assert alone.total_cost > best.total_cost


def test_a_train_into_a_junction_vote():
    pump = unit(1000.0, 10.0, 20_000.0, repair=500.0)
    valve = unit(50_000.0, 5.0, 3_000.0)

    def drawn(n):
        return RepairableRBD(
            [("s", f"p{i}") for i in range(1, n + 1)]
            + [(f"p{i}", "vote") for i in range(1, n + 1)]
            + [("vote", "valve"), ("valve", "t")],
            {
                **{f"p{i}": pump for i in range(1, n + 1)},
                "vote": PerfectReliability,
                "valve": valve,
            },
            k={"vote": 2},
            downtime_cost_rate=2000.0,
        )

    best = drawn(3).allocate_redundancy(
        HORIZON, trains={"pumps": ["p1"]}, max_units=4
    )
    totals = {n: drawn(2 + n).total_cost(HORIZON) for n in range(1, 5)}
    assert best.units == {"pumps": min(totals, key=totals.get)}
    same_design(best, drawn(2 + best.units["pumps"]))


def test_trains_with_hidden_failures():
    # Copies of a tested train are tested with it, as copies of a tested
    # node are: the drawn design's tests are at the same times.
    sensor = {
        "reliability": E([1e-4]),
        "repairability": "instant",
        "inspection": {"interval": 2190.0, "cost": 50.0},
        "acquisition_cost": 5_000.0,
    }
    relay = unit(20_000.0, 4.0, 2_000.0)

    def drawn(n):
        return RepairableRBD(
            [("s", f"sensor {i}") for i in range(1, n + 1)]
            + [(f"sensor {i}", f"relay {i}") for i in range(1, n + 1)]
            + [(f"relay {i}", "t") for i in range(1, n + 1)],
            {
                **{f"sensor {i}": sensor for i in range(1, n + 1)},
                **{f"relay {i}": relay for i in range(1, n + 1)},
            },
            k={"t": 2},
            downtime_cost_rate=500.0,
        )

    best = drawn(2).allocate_redundancy(
        HORIZON, trains={"channel": ["sensor 1", "relay 1"]}, max_units=4
    )
    totals = {n: drawn(1 + n).total_cost(HORIZON) for n in range(1, 5)}
    assert best.total_cost == pytest.approx(min(totals.values()), rel=1e-9)
    same_design(best, drawn(1 + best.units["channel"]))


def test_the_bound_on_a_trains_gain_holds():
    # The n + 1-th train lowers the unavailability by at most its own
    # availability times the chance that at most k - 1 of the n work (just
    # that, in parallel).
    from scipy.stats import binom

    a = 1 / (1 + 10 / 1000) / (1 + 20 / 5000)
    for k in (1, 2, 3):
        u = [station(n, k=k).mean_unavailability() for n in range(k, 8)]
        for n, (before, after) in enumerate(zip(u, u[1:]), start=k):
            bound = a * binom.cdf(k - 1, n, a)
            assert before - after <= bound * (1 + 1e-9), (k, n)


@pytest.mark.parametrize(
    "trains, nodes, match",
    [
        ([("pump 1",)], None, "non-empty dict"),
        ({}, None, "non-empty dict"),
        ({"pump 2": ["pump 1"]}, None, "named as a node"),
        ({"x": "pump 1"}, None, "list of nodes"),
        ({"x": []}, None, "has no nodes"),
        ({"x": ["pump 1", "nope"]}, None, "not a component"),
        ({"x": ["pump 1"], "y": ["pump 1", "motor 1"]}, None, "one train"),
        (TRAIN, ["pump 1"], "on its own or with its train"),
        ({"x": ["pump 1", "motor 2"]}, None, "chain of nodes in series"),
        ({"x": ["motor 1", "pump 1"]}, None, "chain of nodes in series"),
    ],
)
def test_trains_are_checked(trains, nodes, match):
    with pytest.raises(ValueError, match=match):
        station(3).allocate_redundancy(HORIZON, trains=trains, nodes=nodes)


def test_a_train_ending_at_a_fork_and_free_copies_are_refused():
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("a", "c"), ("b", "t"), ("c", "t")],
        {n: unit(1000.0, 10.0, 1000.0) for n in "abc"},
        downtime_cost_rate=100.0,
    )
    with pytest.raises(ValueError, match="feeding one node"):
        rbd.allocate_redundancy(HORIZON, trains={"x": ["a"]})
    free = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": E([1e-3]), "repairability": E([0.1])}},
        downtime_cost_rate=100.0,
    )
    with pytest.raises(ValueError, match="copy of train 'x' costs nothing"):
        free.allocate_redundancy(HORIZON, trains={"x": ["a"]})
    capped = free.allocate_redundancy(
        HORIZON, trains={"x": ["a"]}, max_units={"x": 3}
    )
    assert capped.units == {"x": 3}
    with pytest.raises(ValueError, match="not in nodes or trains"):
        station(3).allocate_redundancy(
            HORIZON, trains=TRAIN, max_units={"train 2": 2}
        )


def test_without_trains_nothing_changes():
    # The search over nodes alone, as before, with no trains in the result.
    best = station(3).allocate_redundancy(HORIZON, nodes=["pump 1"])
    assert best.trains is None
    assert set(best) >= {"units", "total_cost", "trains"}
    assert math.isfinite(best.total_cost)
