"""Common-cause groups in a RepairableRBD (#136): the long-run values of
groups of tested components (staggered or not, with tests that can miss a
failure) and of components whose failures are revealed, against their
definitions: Markov chains built by hand, and the nested windows of a
beta-factor's shared shock."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, linalg
from surpyval import Exponential, Weibull

from repyability import MGL, BetaFactor, CCFGroup, RepairableRBD

E = Exponential.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
VOTE = [("s", "a"), ("s", "b"), ("s", "c"), ("a", "t"), ("b", "t")]
VOTE += [("c", "t")]


def hidden(rate, interval, **inspection):
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval, **inspection},
    }


def revealed(rate, repair, **more):
    return {"reliability": E([rate]), "repairability": E([repair]), **more}


def beta_pair(rate, interval, beta, offset=0.0):
    return RepairableRBD(
        PAIR,
        {
            "a": hidden(rate, interval),
            "b": hidden(rate, interval, offset=offset),
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )


def both_down(rate, beta, ua, ub):
    """A beta-factor pair whose members were last tested ``ua`` and ``ub``
    ago: the chance both are down. The shared shock last struck ``V`` ago
    (``V`` exponential at ``beta * rate``): before both windows it fails
    both, inside one only the member with the longer window."""
    own = (1.0 - beta) * rate
    lo, hi = min(ua, ub), max(ua, ub)
    shock = beta * rate
    q_lo, q_hi = -math.expm1(-own * lo), -math.expm1(-own * hi)
    return (
        -math.expm1(-shock * lo)
        + (math.exp(-shock * lo) - math.exp(-shock * hi)) * q_lo
        + math.exp(-shock * hi) * q_lo * q_hi
    )


@pytest.mark.parametrize("offset", [0.0, 0.5, 0.3])
@pytest.mark.parametrize("rate", [2e-6, 2e-4])
def test_a_tested_pair_by_its_definition(rate, offset):
    interval, beta = 8760.0, 0.05
    rbd = beta_pair(rate, interval, beta, offset * interval)
    shift = offset * interval

    def down(t):
        return both_down(rate, beta, t % interval, (t - shift) % interval)

    breaks = sorted({0.0, shift, interval})
    want = sum(
        integrate.quad(down, lo, hi, epsabs=0, epsrel=1e-13)[0]
        for lo, hi in zip(breaks[:-1], breaks[1:])
    )
    want /= interval
    assert rbd.mean_unavailability() == pytest.approx(want, rel=1e-10)


def test_the_textbook_beta_terms():
    # IEC 61508's 1oo2: (1 - beta)^2 (lambda tau)^2 / 3 + beta lambda tau / 2
    # tested together; the shared term halves when staggered by half an
    # interval (the shock is found by whichever test comes first).
    rate, interval, beta = 2e-6, 8760.0, 0.05
    x = rate * interval
    together = beta_pair(rate, interval, beta).mean_unavailability()
    assert together == pytest.approx(
        (1 - beta) ** 2 * x**2 / 3 + beta * x / 2, rel=0.01
    )
    staggered = beta_pair(rate, interval, beta, interval / 2)
    assert staggered.mean_unavailability() == pytest.approx(
        5 * (1 - beta) ** 2 * x**2 / 24 + beta * x / 4, rel=0.01
    )
    # The issue's example: the shared term four times the independent.
    independent = RepairableRBD(
        PAIR, {"a": hidden(rate, interval), "b": hidden(rate, interval)}
    )
    assert together / independent.mean_unavailability() > 5


def test_small_probabilities_keep_their_precision():
    rate, interval, beta = 1e-9, 100.0, 0.1
    x = rate * interval
    got = beta_pair(rate, interval, beta).mean_unavailability()
    # beta x / 2 to first order; the next terms are of order x^2.
    assert got == pytest.approx(beta * x / 2, rel=1e-6)


def chain(states, rates, jumps):
    """By hand: a chain over ``states`` (tuples of member levels) with
    ``rates`` (a function of a state giving ``{target: rate}``) and test
    ``jumps`` (a function of a state and a test giving its target)."""
    index = {s: i for i, s in enumerate(states)}
    G = np.zeros((len(states), len(states)))
    for s in states:
        for target, rate in rates(s).items():
            if target != s:
                G[index[s], index[target]] += rate
    np.fill_diagonal(G, -G.sum(axis=1))
    return index, G


def test_tests_that_miss_failures_by_a_chain_built_by_hand():
    # A beta-factor pair, tested at different offsets with a coverage of
    # 0.7 and a full test every third test, by matrix exponentials.
    rate, interval, beta, coverage = 3e-3, 100.0, 0.2, 0.7
    offsets = (0.0, 40.0)
    full = 3
    rbd = RepairableRBD(
        PAIR,
        {
            node: hidden(
                rate,
                interval,
                offset=offset,
                coverage=coverage,
                full_test=full * interval,
            )
            for node, offset in zip("ab", offsets)
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    states = list(itertools.product(range(3), repeat=2))  # up, found, missed

    def rates(s):
        out = {}
        causes = [((0,), (1 - beta) * rate), ((1,), (1 - beta) * rate)]
        causes.append(((0, 1), beta * rate))
        for struck, r in causes:
            for level, share in ((1, coverage), (2, 1 - coverage)):
                t = list(s)
                for i in struck:
                    if t[i] == 0:
                        t[i] = level
                out[tuple(t)] = out.get(tuple(t), 0.0) + r * share
        return out

    index, G = chain(states, rates, None)
    period = full * interval
    tests = sorted(
        (offsets[i] + k * interval, i, k % full == 0)
        for i in range(2)
        for k in range(full + 1)
        if 0 < offsets[i] + k * interval <= period
    )

    def test_matrix(member, whole):
        J = np.zeros((len(states), len(states)))
        for s in states:
            t = list(s)
            if t[member] == 1 or (whole and t[member] == 2):
                t[member] = 0
            J[index[s], index[tuple(t)]] = 1.0
        return J

    # One period from all up reaches the long run; then average P(both
    # down) over a second, by quadrature between the tests.
    v = np.zeros(len(states))
    v[index[(0, 0)]] = 1.0
    now = 0.0
    for time, member, whole in tests:
        v = v @ linalg.expm(G * (time - now)) @ test_matrix(member, whole)
        now = time
    v = v @ linalg.expm(G * (period - now))
    down = np.array([s[0] != 0 and s[1] != 0 for s in states], dtype=float)
    total, now = 0.0, 0.0
    points, weights = np.polynomial.legendre.leggauss(30)
    for time, member, whole in tests + [(period, None, None)]:
        half = 0.5 * (time - now)
        for x, w in zip(points, weights):
            at = v @ linalg.expm(G * half * (x + 1.0))
            total += half * w * (at @ down)
        v = v @ linalg.expm(G * (time - now))
        if member is not None:
            v = v @ test_matrix(member, whole)
        now = time
    want = total / period
    assert rbd.mean_unavailability() == pytest.approx(want, rel=1e-9)


def revealed_chain(n, rate, repair, model):
    """A group of ``n`` members with revealed failures and independent
    exponential repairs, by hand: its states (which are down) and their
    long-run probabilities, and its causes."""
    members = [f"u{i}" for i in range(n)]
    own, shocks = model.decompose(members, 1.0)
    causes = [((i,), float(own[0]) * rate) for i in range(n)]
    causes += [
        (tuple(members.index(m) for m in struck), float(q[0]) * rate)
        for struck, q in shocks
    ]
    states = list(itertools.product((False, True), repeat=n))

    def rates(s):
        out = {}
        for struck, r in causes:
            t = list(s)
            for i in struck:
                t[i] = True
            out[tuple(t)] = out.get(tuple(t), 0.0) + r
        for i in range(n):
            if s[i]:
                t = list(s)
                t[i] = False
                out[tuple(t)] = out.get(tuple(t), 0.0) + repair
        return out

    index, G = chain(states, rates, None)
    A = np.vstack([G.T, np.ones(len(states))])
    b = np.zeros(len(states) + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return states, pi, causes


@pytest.mark.parametrize(
    "model", [BetaFactor(0.2), MGL(0.3, 0.4)], ids=["beta", "MGL"]
)
def test_a_revealed_two_out_of_three_by_a_chain_built_by_hand(model):
    rate, repair = 0.01, 0.1
    rbd = RepairableRBD(
        VOTE,
        {n: revealed(rate, repair) for n in "abc"},
        k={"t": 2},
        ccf_groups=[CCFGroup(["a", "b", "c"], model)],
    )
    states, pi, causes = revealed_chain(3, rate, repair, model)
    failed = np.array([sum(s) >= 2 for s in states])
    assert rbd.mean_unavailability() == pytest.approx(pi[failed].sum(), 1e-11)
    # Failures: the causes that take an up system down.
    frequency = 0.0
    for s, p in zip(states, pi):
        if sum(s) >= 2:
            continue
        for struck, r in causes:
            after = list(s)
            for i in struck:
                after[i] = True
            if sum(after) >= 2:
                frequency += p * r
    assert rbd.system_failure_frequency() == pytest.approx(frequency, 1e-10)
    assert rbd.mean_time_between_failures() == pytest.approx(1 / frequency)
    assert rbd.mean_up_time() == pytest.approx(
        pi[~failed].sum() / frequency, rel=1e-9
    )
    # Each member alone is as without the group.
    alone = RepairableRBD(
        VOTE, {n: revealed(rate, repair) for n in "abc"}, k={"t": 2}
    )
    assert rbd.node_availability() == alone.node_availability()
    assert rbd.mean_unavailability() > alone.mean_unavailability()


def test_the_cost_rate_prices_the_downtime_with_the_group():
    rbd = RepairableRBD(
        PAIR,
        {n: revealed(0.01, 0.1, repair_cost=5.0) for n in "ab"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
        downtime_cost_rate=100.0,
    )
    alone = RepairableRBD(
        PAIR,
        {n: revealed(0.01, 0.1, repair_cost=5.0) for n in "ab"},
        downtime_cost_rate=100.0,
    )
    extra = 100.0 * (rbd.mean_unavailability() - alone.mean_unavailability())
    assert rbd.expected_cost_rate() == pytest.approx(
        alone.expected_cost_rate() + extra, rel=1e-12
    )


def test_intervals_are_chosen_with_the_group():
    rate = 2e-6
    rbd = RepairableRBD(
        PAIR,
        {n: hidden(rate, 8760.0, cost=100.0) for n in "ab"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.05))],
    )
    allowed = {n: [2190.0, 4380.0, 8760.0, 17520.0] for n in "ab"}
    plan = rbd.optimal_inspection_intervals(
        allowed=allowed, min_availability=1 - 2.5e-4
    )
    # Yearly tests miss the target with the shared cause (5.3e-4); every
    # three months meet it.
    assert max(plan.intervals.values()) < 8760.0
    assert 1 - plan.availability <= 2.5e-4


def test_saved_and_loaded():
    rbd = RepairableRBD(
        VOTE,
        {
            "a": hidden(1e-4, 100.0),
            "b": hidden(1e-4, 100.0, offset=30.0),
            "c": hidden(1e-4, 100.0, offset=60.0),
        },
        k={"t": 2},
        ccf_groups=[CCFGroup(["a", "b", "c"], MGL(0.1, 0.3))],
    )
    again = RepairableRBD.from_json(rbd.to_json())
    assert again.mean_unavailability() == rbd.mean_unavailability()
    assert (
        "ccf_groups"
        not in RepairableRBD(
            PAIR, {n: revealed(0.01, 0.1) for n in "ab"}
        ).to_dict()
    )


def test_a_nested_rbd_with_a_group():
    inner = RepairableRBD(
        PAIR,
        {n: revealed(0.01, 0.1) for n in "ab"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
    )
    outer = RepairableRBD(
        [("s", "x"), ("x", "y"), ("y", "t")],
        {"x": inner, "y": revealed(0.001, 0.5)},
    )
    assert outer.mean_availability() == pytest.approx(
        inner.mean_availability() * outer.node_availability()["y"]
    )
    with pytest.raises(NotImplementedError, match="common-cause groups"):
        outer.availability(100.0, mc_samples=10)
    assert outer.analysis_routes()["availability"].route == "refused"


@pytest.mark.parametrize(
    "call",
    [
        lambda rbd: rbd.birnbaum_importance(),
        lambda rbd: rbd.fussell_vesely(),
        lambda rbd: rbd.point_availability([10.0]),
        lambda rbd: rbd.availability(100.0, mc_samples=10),
        lambda rbd: rbd.availability_allocation(0.999),
        lambda rbd: rbd.mean_availability(working_nodes=["a"]),
    ],
    ids=["importance", "FV", "over time", "simulation", "allocation", "held"],
)
def test_what_does_not_take_the_groups_in_refuses(call):
    rbd = RepairableRBD(
        PAIR,
        {n: revealed(0.01, 0.1) for n in "ab"},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
    )
    with pytest.raises(NotImplementedError, match="common-cause group"):
        call(rbd)


@pytest.mark.parametrize(
    "components, groups, match",
    [
        (
            {n: revealed(0.01, 0.1) for n in "ab"},
            [CCFGroup(["a", "z"], BetaFactor(0.1))],
            "not a component",
        ),
        (
            {"a": revealed(0.01, 0.1), "b": revealed(0.02, 0.1)},
            [CCFGroup(["a", "b"], BetaFactor(0.1))],
            "identical components",
        ),
        (
            {n: revealed(0.01, 0.1) for n in "ab"},
            [
                CCFGroup(["a", "b"], BetaFactor(0.1)),
                CCFGroup(["b", "a"], BetaFactor(0.1)),
            ],
            "more than one CCF group",
        ),
        (
            {n: revealed(0.01, 0.1) for n in "ab"},
            ["a", "b"],
            "CCFGroup instances",
        ),
    ],
)
def test_the_groups_are_checked(components, groups, match):
    with pytest.raises(ValueError, match=match):
        RepairableRBD(PAIR, components, ccf_groups=groups)


@pytest.mark.parametrize(
    "components, match",
    [
        (
            {
                n: {
                    "reliability": Weibull.from_params([100, 2]),
                    "repairability": E([0.1]),
                }
                for n in "ab"
            },
            "exponential lives",
        ),
        (
            {"a": hidden(0.01, 100.0), "b": revealed(0.01, 0.1)},
            "identical components",
        ),
        (
            {
                "a": hidden(0.01, 100.0),
                "b": hidden(0.01, 100.0, coverage=0.5, full_test=200.0),
            },
            "different coverages",
        ),
    ],
)
def test_what_the_chains_do_not_cover_is_refused(components, match):
    try:
        rbd = RepairableRBD(
            PAIR,
            components,
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
    except ValueError as error:
        assert match in str(error)
        return
    with pytest.raises(NotImplementedError, match=match):
        rbd.mean_availability()
    assert rbd.analysis_routes()["mean_availability"].route == "refused"


def test_the_capacity_distribution_with_the_group():
    rate, repair, beta = 0.01, 0.1, 0.2
    rbd = RepairableRBD(
        PAIR,
        {n: revealed(rate, repair) for n in "ab"},
        capacity={"a": 50.0, "b": 50.0},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    states, pi, _ = revealed_chain(2, rate, repair, BetaFactor(beta))
    down = [sum(s) for s in states]
    want = [sum(p for d, p in zip(down, pi) if d == k) for k in (2, 1, 0)]
    got = rbd.capacity_distribution()
    np.testing.assert_allclose(got.levels, [0.0, 50.0, 100.0])
    np.testing.assert_allclose(got.probabilities, want, rtol=1e-10)
