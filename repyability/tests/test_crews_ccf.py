"""Common-cause groups in the repair crews' Markov chain (#251).

With limited repair crews a shared cause's failures queue for the crews
with the other components', so a group's own chain does not follow them:
the crews' chain takes the groups in. A shared cause strikes at its rate
and fails the members it names that are up, which join the queue one
after another in the cause's order, as the simulation's do
(``_event_loop._strike``). Here: the chain against states solved by hand,
against the chain without groups, against the groups' own chain where no
job waits, and against the simulation (which is given nothing of the
chain); the failure frequency against the chain's own flows; and what
still refuses."""

import math
from fractions import Fraction

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    MGL,
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    RepairableRBD,
)
from repyability.rbd import _crew_chain, _crews
from repyability.rbd import routes as r

E = surv.Exponential.from_params
LAM, MU, BETA = 0.02, 0.1, 0.3
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
#: ``a`` in series with ``b`` or ``z``: which of ``a`` and ``b`` a crew
#: takes first matters.
LADDER = [("s", "a"), ("a", "b"), ("a", "z"), ("b", "t"), ("z", "t")]


def unit(rate=LAM, repair=MU, **extra):
    return {"reliability": E([rate]), "repairability": E([repair]), **extra}


def pair(beta=BETA, crews=1, model=None, **options):
    """Two alike units in parallel in a common-cause group, one crew."""
    groups = None
    if beta is not None:
        groups = [
            CCFGroup(["a", "b"], model or BetaFactor(beta, basis="rate"))
        ]
    return RepairableRBD(
        PAIR,
        {"a": unit(), "b": unit()},
        repair_crews=crews,
        ccf_groups=groups,
        **options,
    )


def ladder(members="ab", crews=1, model=None, priority=None, **options):
    components = {
        m: unit(priority=(priority or {}).get(m, 0), repair_cost=3.0)
        for m in "ab"
    }
    components["z"] = unit(0.03, 0.2)
    return RepairableRBD(
        LADDER,
        components,
        repair_crews=crews,
        ccf_groups=[
            CCFGroup(list(members), model or BetaFactor(0.5, basis="rate"))
        ],
        downtime_cost_rate=5.0,
        **options,
    )


def flows(rbd):
    """The system's failures per unit time from the chain's own flows: over
    every transition that is a failure (a component's, or a strike), its
    rate where the system is up before it and down after."""
    chain = _crews._crew_chain(rbd)
    states = rbd.crew_states()
    up = np.array(
        [
            rbd.is_system_working(
                {n: bool(states.node_availability[n][s]) for n in rbd.nodes},
                "p",
            )
            for s in range(len(chain.probabilities))
        ],
        dtype=float,
    )
    sources, targets, rates, _ = chain.transitions
    return float(
        np.sum(
            chain.probabilities[sources]
            * rates
            * up[sources]
            * (1.0 - up[targets])
        )
    )


# -- the chain ----------------------------------------------------------------


def test_two_members_and_one_crew_are_the_chain_solved_by_hand():
    # By the number down (0, 1, 2), with rho = lam / mu: one fails at its
    # own cause (2 (1 - beta) lam), both at the shared one (beta lam), the
    # second at every cause it has (lam), and one is repaired at a time.
    # Balance at 0 and across the cut below 2:
    #   p1 = (2 - beta) rho p0,    p2 = rho (beta p0 + p1).
    rho = Fraction(LAM).limit_denominator() / Fraction(MU).limit_denominator()
    beta = Fraction(BETA).limit_denominator()
    p0 = Fraction(1)
    p1 = (2 - beta) * rho * p0
    p2 = rho * (beta * p0 + p1)
    total = p0 + p1 + p2
    rbd = pair()
    assert rbd.mean_unavailability() == pytest.approx(
        float(p2 / total), rel=1e-13
    )
    # Its failures: the shared cause from 0, the other unit from 1.
    lam = Fraction(LAM).limit_denominator()
    assert rbd.system_failure_frequency() == pytest.approx(
        float((beta * lam * p0 + lam * p1) / total), rel=1e-13
    )
    # A strike queues the members in the cause's order: a is repaired
    # first. The states with both down: a's repair with b waiting, from
    # the strike or b's failure; b's with a waiting, from a's failure.
    chain = _crews._crew_chain(rbd)
    p = dict(zip(chain.states, chain.probabilities))
    mu = float(Fraction(MU).limit_denominator())
    assert p[((0,), (1,))] * mu == pytest.approx(
        BETA * LAM * p[((), ())] + LAM * p[((0,), ())], rel=1e-12
    )
    assert p[((1,), (0,))] * mu == pytest.approx(
        LAM * p[((1,), ())], rel=1e-12
    )


def test_no_shared_part_is_the_chain_without_groups():
    plain, none = pair(None), pair(0.0)
    assert none.mean_availability() == plain.mean_availability()
    assert none.system_failure_frequency() == plain.system_failure_frequency()
    t = np.array([2.0, 20.0, 200.0])
    np.testing.assert_array_equal(
        none.point_availability(t), plain.point_availability(t)
    )
    np.testing.assert_array_equal(
        _crews._crew_chain(none).probabilities,
        _crews._crew_chain(plain).probabilities,
    )


@pytest.mark.parametrize(
    "model",
    [
        BetaFactor(0.2, basis="rate"),
        MGL(0.25, basis="rate"),
        AlphaFactor([0.8, 0.2], basis="rate"),
    ],
)
def test_a_crew_each_is_the_groups_own_long_run(model):
    # With a crew for every component no job waits: the chain with the
    # shared causes is then the group's own (which the diagram takes
    # where its crews do not limit it).
    free = ladder(crews=None, model=model)
    limited = ladder(crews=1, model=model)
    own, shared = _crews._crew_causes(limited, list("abz"))
    assert shared and own["a"] < LAM and own["z"] == 0.03
    chain = _crew_chain.solve(
        list("abz"),
        [own[n] for n in "abz"],
        [MU, MU, 0.2],
        [0.0] * 3,
        3,
        0,
        shared,
    )
    up = np.array(
        [
            free.is_system_working(dict(zip("abz", map(bool, row))), "p")
            for row in chain.up
        ],
        dtype=float,
    )
    assert float(chain.probabilities @ up) == pytest.approx(
        free.mean_availability(), rel=1e-12
    )
    for k, node in enumerate("abz"):
        assert chain.availability(node) == pytest.approx(
            free.node_availability()[node], rel=1e-12
        )


@pytest.mark.parametrize("members", ["ab", "ba"])
@pytest.mark.parametrize("crews", [1, 2])
def test_the_failure_frequency_is_the_chain_s_flows(members, crews):
    rbd = ladder(members, crews)
    frequency = rbd.system_failure_frequency()
    assert frequency == pytest.approx(flows(rbd), rel=1e-12)
    # Its terms: each component's own cause, and the group's shared ones.
    shares = rbd.barlow_proschan_importance()
    assert set(shares) == {"a", "b", "z", tuple(members)}
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-12)
    assert shares[tuple(members)] > 0.0
    # The up and down times follow.
    assert rbd.mean_up_time() == pytest.approx(
        rbd.mean_availability() / frequency, rel=1e-12
    )


def test_the_order_of_a_cause_s_members_is_the_queue_s():
    # Struck together with the crew free, the first in the cause's order
    # is repaired first, whatever the priorities (they order the jobs
    # that wait): a first keeps the system up more.
    first, last = ladder("ab"), ladder("ba")
    assert first.mean_availability() > last.mean_availability() + 1e-3
    urgent = ladder("ba", priority={"a": 1})
    assert urgent.mean_availability() > last.mean_availability()
    assert urgent.mean_availability() < first.mean_availability()


def test_a_member_held_is_struck_no_more_and_the_others_still_are():
    # Held working, a never fails; b still fails at its whole rate, its
    # own cause's and the shared one's: the same system with no group.
    grouped = ladder("ab")
    plain = RepairableRBD(
        LADDER,
        {"a": unit(), "b": unit(), "z": unit(0.03, 0.2)},
        repair_crews=1,
    )
    for held in ({"working_nodes": ["a"]}, {"broken_nodes": ["a"]}):
        assert grouped.mean_availability(**held) == pytest.approx(
            plain.mean_availability(**held), rel=1e-13
        )
    # Birnbaum's measure holds each node both ways.
    importance = grouped.birnbaum_importance()
    assert importance["a"] == pytest.approx(
        grouped.mean_availability(working_nodes=["a"])
        - grouped.mean_availability(broken_nodes=["a"]),
        rel=1e-12,
    )


# -- over time ----------------------------------------------------------------


def test_the_rate_s_parts_add_up_to_the_availability_s_slope():
    rbd = ladder("ab")
    t = np.array([3.0, 12.0, 40.0])
    rate = rbd.availability_rate(t)
    step = 1e-4
    slope = (
        rbd.point_availability(t + step) - rbd.point_availability(t - step)
    ) / (2 * step)
    np.testing.assert_allclose(rate.rate, slope, rtol=1e-6)
    assert ("a", "b") in rate.node_rate
    np.testing.assert_allclose(
        sum(np.asarray(v) for v in rate.node_rate.values()),
        rate.rate,
        rtol=1e-10,
    )
    # A strike only takes the system down.
    assert np.all(np.asarray(rate.node_rate[("a", "b")]) < 0.0)


def test_the_window_s_failures_are_split_by_cause():
    rbd = ladder("ab")
    T = 150.0
    events = rbd.expected_events(T)
    # The members fail at their whole rate while up: own and shared.
    for node in "ab":
        assert events.node_failures[node] == pytest.approx(
            LAM * (T - events.node_downtime[node]), rel=1e-10
        )
    # In the long run the window's failures grow at the frequency.
    long, longer = (
        rbd.expected_events(T).system_failures for T in (4000.0, 5000.0)
    )
    assert (longer - long) / 1000.0 == pytest.approx(
        rbd.system_failure_frequency(), rel=1e-9
    )


# -- the simulation -----------------------------------------------------------


def call_out():
    """Three members' worth of jobs for one crew and a call-out crew."""
    return RepairableRBD(
        [("s", n) for n in "abcd"] + [(n, "t") for n in "abcd"],
        {
            "a": unit(0.05, 0.15),
            "b": unit(0.05, 0.15, priority=1),
            "c": unit(0.04, 0.2, priority=2, repair_cost=4.0),
            "d": unit(0.04, 0.2, repair_cost=4.0),
        },
        k={"t": 3},
        repair_crews=1,
        call_out={"when_waiting": 2, "fee": 30.0, "rate": 4.0},
        ccf_groups=[
            CCFGroup(["a", "b"], MGL(0.4, basis="rate")),
            CCFGroup(["d", "c"], AlphaFactor([0.7, 0.3], basis="rate")),
        ],
        downtime_cost_rate=6.0,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: ladder("ab"),
        lambda: ladder("ba", crews=2, priority={"a": 1}),
        call_out,
        lambda: ladder("ab", model=BetaFactor(0.3)),
    ],
    ids=["one crew", "two crews", "two groups, a call-out crew", "by Q"],
)
def test_the_simulation_agrees_with_the_chain(build):
    T, n = 300.0, 6000
    rbd = build()
    run = rbd.availability(
        T, mc_samples=n, seed=20261121, control_variate=False
    )
    events = rbd.expected_events(T)
    down = T - np.asarray(run.uptimes)
    assert abs(down.mean() - events.system_downtime) < 3.5 * (
        down.std(ddof=1) / math.sqrt(n)
    )
    # The failures of a run are no fewer than Poisson's spread apart.
    failures = run.system_failures / n
    assert abs(failures - events.system_failures) < 3.5 * math.sqrt(
        2.0 * events.system_failures / n
    )
    for node in rbd.nodes:
        assert run.node_downtime[node] / n == pytest.approx(
            events.node_downtime[node], rel=0.03
        ), node
    cost = run.cost
    assert abs(cost.sample_mean - rbd.expected_cost(T).mean) < (
        3.5 * cost.mean_se
    )
    # The point availability, at the simulation's own times.
    at = np.array([5.0, 25.0, 100.0, 250.0])
    timeline = np.interp(at, run.timeline, run.availability)
    exact = rbd.point_availability(at)
    spread = np.sqrt(exact * (1.0 - exact) / n)
    assert np.all(np.abs(timeline - exact) < 3.5 * spread + 1e-3)


# -- what refuses, and what the report says -----------------------------------


def test_the_report_says_the_chain_has_the_groups():
    rbd = ladder("ab")
    report = rbd.analysis_routes()
    for method in (
        "mean_availability",
        "node_availability",
        "system_failure_frequency",
        "barlow_proschan_importance",
        "birnbaum_importance",
        "criticality_importance",
        "expected_cost_rate",
    ):
        assert report[method].route == r.EXACT, method
    for method in (
        "point_availability",
        "mission_availability",
        "expected_events",
        "expected_cost",
        "availability_rate",
    ):
        assert report[method].route == r.NUMERICAL, method


def test_a_cause_of_three_members_is_left_to_the_simulation():
    rbd = RepairableRBD(
        [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
        {n: unit() for n in "abc"},
        repair_crews=1,
        ccf_groups=[CCFGroup(list("abc"), BetaFactor(0.2, basis="rate"))],
    )
    with pytest.raises(NotImplementedError, match="one that strikes 3") as e:
        rbd.mean_availability()
    assert "availability() or cost()" in str(e.value)
    with pytest.raises(NotImplementedError, match="one that strikes 3"):
        rbd.point_availability(10.0)
    report = rbd.analysis_routes()
    assert report["mean_availability"].route == r.REFUSED
    assert report["availability"].route == r.SIMULATED
    rbd.availability(50.0, mc_samples=20, seed=1)
    # Three members whose causes strike two at most are taken.
    pairs = RepairableRBD(
        [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
        {n: unit() for n in "abc"},
        repair_crews=1,
        ccf_groups=[
            CCFGroup(list("abc"), AlphaFactor([0.7, 0.3, 0.0], basis="rate"))
        ],
    )
    assert 0.0 < pairs.mean_availability() < 1.0


def test_members_the_chain_has_no_place_for_refuse():
    instant = RepairableRBD(
        LADDER,
        {
            "a": {"reliability": E([LAM]), "repairability": "instant"},
            "b": {"reliability": E([LAM]), "repairability": "instant"},
            "z": unit(),
        },
        repair_crews=1,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2, basis="rate"))],
    )
    with pytest.raises(NotImplementedError, match="repaired instantly"):
        instant.mean_availability()
    tested = RepairableRBD(
        LADDER,
        {
            **{m: unit(inspection={"interval": 50.0}) for m in "ab"},
            "z": unit(),
        },
        repair_crews=1,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2, basis="rate"))],
    )
    with pytest.raises(NotImplementedError, match="inspections"):
        tested.mean_availability()
    assert tested.analysis_routes()["availability"].route == r.SIMULATED
