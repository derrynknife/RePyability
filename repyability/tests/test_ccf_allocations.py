"""Common-cause groups in a RepairableRBD's allocations (#158). In the
redundancy allocation a member's copies join its group (a ``BetaFactor``'s
beta holds at any size), scored by a chain that counts each member's copies
at each level: checked against the chain of every copy, and the designs
against the systems with their copies drawn out. The availability
allocations hold the members, whose MTTF and MTTR are their group's: the
MTTFs and MTTRs they allocate are checked by rebuilding the system with
them."""

import itertools
import math

import numpy as np
import pytest
from scipy.special import comb
from surpyval import Exponential

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    PerfectReliability,
    RepairableRBD,
)
from repyability.rbd import _ccf_chain

E = Exponential.from_params


def revealed(rate, repair, **more):
    return {"reliability": E([rate]), "repairability": E([repair]), **more}


def inspected(rate, interval, offset=0.0, coverage=1.0, cost=0.0, **more):
    inspection = {"interval": interval, "offset": offset, "cost": cost}
    if coverage < 1.0:
        inspection.update(coverage=coverage, full_test=3 * interval)
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": inspection,
        **more,
    }


def drawn(copies, pump, valve, model, rate):
    """Pumps a and b in parallel, in a common-cause group, and a valve v
    after them, with ``copies[x]`` of each drawn out in parallel: a pump's
    copies (``pump(x)``) join its group."""
    names = {
        x: [x] + [f"{x} {j}" for j in range(2, copies.get(x, 1) + 1)]
        for x in "abv"
    }
    pumps = names["a"] + names["b"]
    edges = [("s", x) for x in pumps] + [(x, "m") for x in pumps]
    edges += [("m", x) for x in names["v"]] + [(x, "t") for x in names["v"]]
    components = {x: pump(x[0]) for x in pumps}
    components.update({x: valve for x in names["v"]}, m=PerfectReliability)
    return RepairableRBD(
        edges,
        components,
        ccf_groups=[CCFGroup(pumps, model)],
        downtime_cost_rate=rate,
    )


def revealed_pump(_):
    return revealed(0.01, 0.5, acquisition_cost=100.0, repair_cost=1.0)


def inspected_pump(base):
    offset = {"a": 0.0, "b": 50.0}[base]
    return inspected(0.002, 100.0, offset, 0.8, 1.0, acquisition_cost=80.0)


VALVE = revealed(0.002, 0.2, acquisition_cost=50.0)


def check_against_the_drawn_out_designs(pump, nodes, rate, most):
    def build(copies):
        return drawn(copies, pump, VALVE, BetaFactor(0.1), rate)

    rbd = build({})
    totals, systems = {}, {}
    for counts in itertools.product(range(1, most + 1), repeat=len(nodes)):
        systems[counts] = build(dict(zip(nodes, counts)))
        totals[counts] = systems[counts].total_cost(1000.0)
    for method in ("exact", "greedy"):
        best = rbd.allocate_redundancy(
            1000.0, nodes=nodes, max_units=most, method=method
        )
        design = tuple(best.units[x] for x in nodes)
        assert best.total_cost == pytest.approx(totals[design], rel=1e-12)
        assert best.availability == pytest.approx(
            systems[design].mean_availability(), rel=1e-12
        )
        if method == "exact":
            # Designs alike but for which member has the copies tie.
            assert best.total_cost == pytest.approx(
                min(totals.values()), rel=1e-12
            )


@pytest.mark.parametrize("rate", [5.0, 500.0, 5000.0])
def test_a_member_s_copies_join_its_group(rate):
    check_against_the_drawn_out_designs(revealed_pump, ["a", "v"], rate, 3)


@pytest.mark.parametrize("rate", [5.0, 50.0, 500.0])
def test_a_tested_member_s_copies_are_tested_with_it(rate):
    check_against_the_drawn_out_designs(
        inspected_pump, ["a", "b", "v"], rate, 2
    )


def schedule(offsets, interval=40.0):
    """Each member's tests in a period of three intervals, the last of each
    member's three a full test."""
    out = []
    for member, offset in enumerate(offsets):
        first = 0 if offset else 1
        for k in range(first, first + 3):
            out.append(
                _ccf_chain.ProofTest(offset + k * interval, member, k % 3 == 0)
            )
    return out


def collapsed(states, counts):
    """The states of every copy as those of the members' nodes: each down
    while all its copies are."""
    starts = np.cumsum([0, *counts[:-1]])
    down = np.stack(
        [
            states.down[:, s : s + c].all(axis=1)  # noqa: E203
            for s, c in zip(starts, counts)
        ],
        axis=1,
    )
    index = down.astype(int) @ (2 ** np.arange(len(counts)))
    return np.stack(
        [
            states.probabilities[:, index == j].sum(axis=1)
            for j in range(2 ** len(counts))
        ],
        axis=1,
    )


@pytest.mark.parametrize("counts", [(1, 1, 1), (2, 1, 1), (2, 3, 1), (3, 2)])
@pytest.mark.parametrize("coverage", [None, 1.0, 0.7])
def test_the_counted_chain_is_the_chain_of_every_copy(counts, coverage):
    model = BetaFactor(0.2)
    n = len(counts)
    members = [f"m{i}" for i in range(n)]
    every = [(m, j) for m, c in zip(members, counts) for j in range(c)]
    if coverage is None:
        counted = _ccf_chain.revealed(model, members, 0.01, 0.5, counts)
        full = _ccf_chain.revealed(model, every, 0.01, 0.5)
    else:
        offsets = [10.0 * i for i in range(n)]
        tests = schedule(offsets)
        position = np.cumsum([0, *counts[:-1]])
        each = [
            test._replace(member=int(position[test.member]) + j)
            for test in tests
            for j in range(counts[test.member])
        ]
        times = np.linspace(0.0, 119.0, 7)
        counted = _ccf_chain.hidden(
            model, members, 0.004, coverage, tests, 120.0, times, counts
        )
        full = _ccf_chain.hidden(
            model, every, 0.004, coverage, each, 120.0, times
        )
        assert (counted.probabilities.min()) < 1e-6  # small ones kept
    assert counted.members == tuple(members)
    assert counted.down.tolist() == [
        [bool(j >> i & 1) for i in range(n)] for j in range(2**n)
    ]
    np.testing.assert_allclose(
        counted.probabilities, collapsed(full, counts), rtol=1e-12, atol=0
    )
    if set(counts) == {1}:
        # One copy each: the group's own chain, to the last bit.
        assert np.array_equal(counted.probabilities, full.probabilities)


def test_many_copies_are_counted():
    # 22 copies of the pumps in one group would be 2 ** 22 states each up or
    # down; counted, they are as many as the pairs of counts. The pumps'
    # copies are all in parallel: they are all down a fraction E[p(A) **
    # N] of the time, A the time since the shared cause last struck
    # (exponential at its rate r), from which each copy is down with
    # p(a) = pi + (1 - pi) exp(-s a), at s its own failure and repair rates
    # together and pi their long-run ratio.
    rbd = drawn(
        {},
        lambda _: revealed(0.1, 1.0, acquisition_cost=100.0),
        revealed(0.02, 0.5, acquisition_cost=50.0),
        BetaFactor(0.01),
        50_000.0,
    )
    best = rbd.allocate_redundancy(1000.0)
    pumps = best.units["a"] + best.units["b"]
    assert pumps > 14
    own, r = 0.99 * 0.1, 0.01 * 0.1
    s = own + 1.0
    pi = own / s
    down = math.fsum(
        comb(pumps, j, exact=True)
        * pi ** (pumps - j)
        * (1.0 - pi) ** j
        * r
        / (r + j * s)
        for j in range(pumps + 1)
    )
    valve = 0.02 / 0.52
    assert best.availability == pytest.approx(
        (1.0 - down) * (1.0 - valve ** best.units["v"]), rel=1e-9
    )


def test_with_beta_zero_the_allocations_are_as_without_the_group():
    def pair(groups):
        return RepairableRBD(
            [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")],
            {
                "p1": revealed(0.1, 1.0, acquisition_cost=100.0),
                "p2": revealed(0.1, 1.0, acquisition_cost=100.0),
                "v": revealed(0.02, 0.5, acquisition_cost=50.0),
            },
            ccf_groups=groups,
            downtime_cost_rate=500.0,
        )

    grouped = pair([CCFGroup(["p1", "p2"], BetaFactor(0.0))])
    alone = pair(None)
    a = grouped.allocate_redundancy(1000.0, max_units=4)
    b = alone.allocate_redundancy(1000.0, max_units=4)
    assert a.units == b.units
    assert a.total_cost == pytest.approx(b.total_cost, rel=1e-12)
    assert a.availability == pytest.approx(b.availability, rel=1e-12)
    # The members are held, as fixed ones are.
    a = grouped.availability_allocation(0.99)
    b = alone.availability_allocation(0.99, fixed=["p1", "p2"])
    assert a.mttf == pytest.approx(b.mttf, rel=1e-9)
    assert a.system_availability == pytest.approx(
        b.system_availability, rel=1e-12
    )


def plant(v=(50.0, 2.0), c=(80.0, 1.0), model=None, hidden=False):
    """Pumps p1 and p2 in parallel and in a group, then a valve v and a
    controller c in series, each given as (MTTF, MTTR)."""

    def unit(mttf, mttr):
        return revealed(1.0 / mttf, 1.0 / mttr)

    pumps = {"p1": unit(10.0, 1.0), "p2": unit(10.0, 1.0)}
    if hidden:
        pumps = {
            "p1": inspected(0.01, 20.0),
            "p2": inspected(0.01, 20.0, offset=10.0),
        }
    return RepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "m"), ("p2", "m")]
        + [("m", "v"), ("v", "c"), ("c", "t")],
        {**pumps, "v": unit(*v), "c": unit(*c), "m": PerfectReliability},
        ccf_groups=[CCFGroup(["p1", "p2"], model or BetaFactor(0.1))],
    )


@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("model", [BetaFactor(0.1), MGL(0.2)])
@pytest.mark.parametrize("method", ["cost_based", "improvement", "equal"])
def test_the_members_keep_their_availability(method, model, hidden):
    rbd = plant(model=model, hidden=hidden)
    target = 0.96
    assert rbd.mean_availability() < target
    need = rbd.availability_allocation(target, method=method)
    assert set(need.mttf) == {"v", "c"}
    for member in ("p1", "p2"):
        assert need.availability[member] == rbd.node_availability()[member]
    again = plant((need.mttf["v"], 2.0), (need.mttf["c"], 1.0), model, hidden)
    assert again.mean_availability() == pytest.approx(
        need.system_availability, rel=1e-12
    )
    assert need.system_availability == pytest.approx(target, rel=1e-9)


@pytest.mark.parametrize("hidden", [False, True])
def test_mttfs_and_mttrs_around_a_group(hidden):
    rbd = plant(hidden=hidden)
    design = rbd.mttf_mttr_allocation(0.96)
    assert set(design.mttf) == set(design.mttr) == {"v", "c"}
    again = plant(
        (design.mttf["v"], design.mttr["v"]),
        (design.mttf["c"], design.mttr["c"]),
        hidden=hidden,
    )
    assert again.mean_availability() == pytest.approx(
        design.system_availability, rel=1e-12
    )
    assert design.system_availability == pytest.approx(0.96, rel=1e-6)


def test_the_least_effort_with_a_group_in_series():
    def series(v):
        return RepairableRBD(
            [("s", "p1"), ("p1", "p2"), ("p2", "v"), ("v", "t")],
            {
                "p1": revealed(0.01, 1.0),
                "p2": revealed(0.01, 1.0),
                "v": revealed(1.0 / v, 0.5),
            },
            ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.3))],
        )

    need = series(50.0).availability_allocation(0.95, method="minimum_effort")
    assert series(need.mttf["v"]).mean_availability() == pytest.approx(
        0.95, rel=1e-12
    )


def test_what_the_allocation_of_copies_refuses():
    rbd = drawn({}, revealed_pump, VALVE, MGL(0.2), 500.0)
    with pytest.raises(NotImplementedError, match="MGL model's letters"):
        rbd.allocate_redundancy(1000.0)
    # Its members left out, the valve takes copies.
    assert rbd.allocate_redundancy(1000.0, nodes=["v"]).units["v"] > 1
    rbd = drawn({}, revealed_pump, VALVE, BetaFactor(0.1), 500.0)
    with pytest.raises(NotImplementedError, match="in a common-cause group"):
        rbd.allocate_redundancy(1000.0, trains={"pump": ["a"]})


def test_the_routes_say_how():
    report = plant().analysis_routes()
    for name in (
        "allocate_redundancy",
        "availability_allocation",
        "mttf_mttr_allocation",
    ):
        assert report[name].route in ("exact", "numerical")
    assert "copies join its group" in report["allocate_redundancy"].reason
    assert "keep their availability" in (
        report["availability_allocation"].reason
    )
