"""Common-cause groups in a RepairableRBD over time from new and in the
simulations (#158): each group's chain followed from every member up at 0,
the system worked out over its members' combinations, against the chains
built by hand, closed forms, the long run and the simulation; and the
simulation, in which each shared cause strikes as a Poisson process,
against the exact values."""

import warnings

import numpy as np
import pytest
from scipy import integrate, linalg
from surpyval import Exponential, Weibull

from repyability import MGL, BetaFactor, CCFGroup, NodeState, RepairableRBD
from repyability.rbd import routes as r

E, W = Exponential.from_params, Weibull.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
PAIR_THEN_C = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
THREE_THEN_C = [("s", "a"), ("s", "b"), ("s", "d")]
THREE_THEN_C += [("a", "c"), ("b", "c"), ("d", "c"), ("c", "t")]


def revealed(rate, repair, **more):
    return {"reliability": E([rate]), "repairability": E([repair]), **more}


def hidden(rate, interval, **inspection):
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval, **inspection},
    }


OTHER = {"reliability": W([500.0, 2.0]), "repairability": E([0.5])}


def pair_then_c(beta, member=None, b=None):
    member = member or revealed(0.01, 0.2)
    return RepairableRBD(
        PAIR_THEN_C,
        {"a": member, "b": b or member, "c": OTHER},
        ccf_groups=(
            None if beta is None else [CCFGroup(["a", "b"], BetaFactor(beta))]
        ),
    )


# -- over time, exact -------------------------------------------------------


@pytest.mark.parametrize(
    "member, b",
    [
        (revealed(0.01, 0.2), None),
        (hidden(1e-3, 300.0), hidden(1e-3, 300.0, offset=100.0)),
        (
            hidden(1e-3, 300.0, coverage=0.6, full_test=900.0),
            None,
        ),
    ],
    ids=["revealed", "staggered tests", "tests that miss"],
)
def test_no_shared_cause_is_the_independent_system(member, b):
    grouped, plain = pair_then_c(0.0, member, b), pair_then_c(None, member, b)
    x = np.array([0.0, 1.0, 50.0, 299.0, 301.0, 1500.0, 1e5])
    np.testing.assert_allclose(
        grouped.point_availability(x), plain.point_availability(x), atol=1e-6
    )
    assert grouped.mission_availability(3000.0) == pytest.approx(
        plain.mission_availability(3000.0), abs=1e-7
    )
    assert grouped.expected_failures(3000.0) == pytest.approx(
        plain.expected_failures(3000.0), rel=1e-6
    )


def pair_chain(rate, repair, beta):
    """The revealed pair's chain by hand: bit 0 member a down, bit 1 b."""
    own, shared = (1.0 - beta) * rate, beta * rate
    Q = np.zeros((4, 4))
    for state in range(4):
        for bit in (1, 2):
            if not state & bit:
                Q[state, state | bit] += own
            else:
                Q[state, state & ~bit] += repair
        if state != 3:
            Q[state, 3] += shared
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def test_a_revealed_pair_against_its_chain():
    rate, repair, beta = 0.01, 0.2, 0.1
    c_rate, c_repair = 0.002, 0.5
    other = revealed(c_rate, c_repair)
    rbd = RepairableRBD(
        PAIR_THEN_C,
        {"a": revealed(rate, repair), "b": revealed(rate, repair), "c": other},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    Q = pair_chain(rate, repair, beta)
    start = np.array([1.0, 0.0, 0.0, 0.0])

    def states(t):
        return start @ linalg.expm(Q * t)

    def c_up(t):
        total = c_rate + c_repair
        return c_repair / total + c_rate / total * np.exp(-total * t)

    def up(t):
        return (1.0 - states(t)[3]) * c_up(t)

    def failing(t):
        p = states(t)
        own = (1.0 - beta) * rate
        # The pair fails into both down while c is up; c fails while the
        # pair is up.
        pair = p[0] * beta * rate + (p[1] + p[2]) * (own + beta * rate)
        return pair * c_up(t) + (1.0 - p[3]) * c_up(t) * c_rate

    x = np.array([1.0, 5.0, 20.0, 100.0, 1000.0])
    # The other component's curve is on a grid (to about 1e-6 early on).
    np.testing.assert_allclose(
        rbd.point_availability(x), [up(t) for t in x], atol=2e-6
    )
    mission = integrate.quad(up, 0.0, 500.0, limit=500)[0] / 500.0
    assert rbd.mission_availability(500.0) == pytest.approx(mission, abs=1e-7)
    failures = integrate.quad(failing, 0.0, 500.0, limit=500)[0]
    assert rbd.expected_failures(500.0) == pytest.approx(failures, rel=1e-6)
    events = rbd.expected_events(500.0)
    assert events.system_failures == pytest.approx(failures, rel=1e-6)


def test_a_pair_tested_together_in_closed_form():
    # Both renewed at each test: since the last, u ago, both are down if
    # the shared cause struck, or each its own.
    rate, interval, beta = 1e-4, 1000.0, 0.1
    rbd = RepairableRBD(
        PAIR,
        {"a": hidden(rate, interval), "b": hidden(rate, interval)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    own, shared = (1.0 - beta) * rate, beta * rate

    def down(u):
        return 1.0 - np.exp(-shared * u) * (
            1.0 - (1.0 - np.exp(-own * u)) ** 2
        )

    def failing(u):
        both_up = np.exp(-(shared + 2.0 * own) * u)
        one_up = 2.0 * np.exp(-(shared + own) * u) * -np.expm1(-own * u)
        return both_up * shared + one_up * (own + shared)

    x = np.array([1.0, 400.0, 999.9, 1500.0, 2999.0])
    np.testing.assert_allclose(
        1.0 - rbd.point_availability(x),
        down(np.mod(x, interval)),
        rtol=1e-12,
        atol=1e-16,
    )
    pfd = integrate.quad(down, 0.0, interval)[0] / interval
    assert 1.0 - rbd.mission_availability(interval) == pytest.approx(
        pfd, rel=1e-10
    )
    assert rbd.mean_unavailability() == pytest.approx(pfd, rel=1e-10)
    assert rbd.expected_failures(interval) == pytest.approx(
        integrate.quad(failing, 0.0, interval)[0], rel=1e-10
    )


@pytest.mark.parametrize(
    "rbd",
    [
        pair_then_c(0.2),
        RepairableRBD(
            THREE_THEN_C,
            {n: revealed(0.01, 0.2) for n in "abd"} | {"c": OTHER},
            ccf_groups=[CCFGroup(["a", "b", "d"], MGL(0.2, 0.4))],
        ),
        RepairableRBD(
            PAIR,
            {
                "a": hidden(1e-4, 1000.0),
                "b": hidden(1e-4, 1000.0, offset=500.0),
            },
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        ),
    ],
    ids=["revealed", "MGL", "staggered tests"],
)
def test_it_settles_into_the_long_run(rbd):
    late = 1e6 + np.array([0.0, 125.0, 250.0, 625.0, 900.0])
    if rbd._inspection:
        # The long run's profile over a period, against its average.
        def down(t):
            return 1.0 - float(rbd.point_availability(t))

        period = sum(
            integrate.quad(down, a, a + 500.0, limit=200, epsabs=1e-16)[0]
            for a in (1e6, 1e6 + 500.0)
        )
        assert period / 1000.0 == pytest.approx(
            rbd.mean_unavailability(), rel=1e-9
        )
    else:
        np.testing.assert_allclose(
            rbd.point_availability(late), rbd.mean_availability(), rtol=1e-9
        )
    # A long window's failures, a period at a time once it has settled.
    window = 2e6
    assert rbd.expected_failures(window) / window == pytest.approx(
        rbd.system_failure_frequency(), rel=2e-3
    )


def test_the_capacity_over_time():
    def plant(beta):
        return RepairableRBD(
            [("s", n) for n in "abd"] + [(n, "t") for n in "abd"],
            {n: revealed(0.01, 0.2) for n in "abd"},
            capacity={n: 50.0 for n in "abd"},
            ccf_groups=(
                None
                if beta is None
                else [CCFGroup(["a", "b", "d"], BetaFactor(beta))]
            ),
        )

    x = np.array([0.0, 5.0, 50.0, 1000.0])
    grouped, plain = plant(0.0), plant(None)
    np.testing.assert_allclose(
        grouped.point_capacity(x).probabilities,
        plain.point_capacity(x).probabilities,
        atol=1e-6,
    )
    rbd = plant(0.1)
    capacity = rbd.point_capacity(x)
    # Up with any one member: its capacity reaches 50.
    np.testing.assert_allclose(
        capacity.meets(50.0), rbd.point_availability(x), atol=1e-14
    )
    assert rbd.point_capacity(1e5).meets(100.0) == pytest.approx(
        rbd.capacity_distribution().meets(100.0), rel=1e-12
    )
    assert rbd.mission_capacity(500.0).meets(50.0) == pytest.approx(
        rbd.mission_availability(500.0), abs=1e-9
    )


def test_a_nested_rbd_with_a_group_over_time():
    def inner():
        return RepairableRBD(
            PAIR,
            {n: revealed(0.01, 0.2) for n in "ab"},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
        )

    nested = RepairableRBD(
        [("s", "n"), ("n", "c"), ("c", "t")], {"n": inner(), "c": OTHER}
    )
    flat = pair_then_c(0.2)
    x = np.array([1.0, 50.0, 1000.0])
    np.testing.assert_allclose(
        nested.point_availability(x), flat.point_availability(x), rtol=1e-10
    )
    assert nested.expected_failures(2000.0) == pytest.approx(
        flat.expected_failures(2000.0), rel=1e-6
    )


# -- the simulation ---------------------------------------------------------


def simulated_cases():
    staggered = hidden(2e-4, 500.0, offset=250.0)
    missing = hidden(2e-4, 500.0, coverage=0.5, full_test=2000.0)
    return {
        "revealed": pair_then_c(0.2),
        "MGL": RepairableRBD(
            THREE_THEN_C,
            {n: revealed(0.01, 0.2) for n in "abd"} | {"c": OTHER},
            ccf_groups=[CCFGroup(["a", "b", "d"], MGL(0.2, 0.4))],
        ),
        "staggered tests": pair_then_c(0.2, hidden(2e-4, 500.0), staggered),
        "tests that miss": pair_then_c(0.3, missing),
        "every cause shared": pair_then_c(1.0),
        "nested": RepairableRBD(
            [("s", "n"), ("n", "c"), ("c", "t")],
            {
                "n": RepairableRBD(
                    PAIR,
                    {n: revealed(0.01, 0.2) for n in "ab"},
                    ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
                ),
                "c": OTHER,
            },
        ),
    }


@pytest.mark.parametrize("name", list(simulated_cases()))
def test_the_simulation_agrees(name):
    rbd = simulated_cases()[name]
    T, n = 3000.0, 6000
    result = rbd.availability(T, mc_samples=n, seed=3)
    interval = result.mean_availability_interval()
    exact = rbd.mission_availability(T)
    assert abs(interval.estimate - exact) < 4.0 * interval.standard_error
    failures = rbd.expected_failures(T)
    # A count's spread is about its square root.
    assert result.system_failures / n == pytest.approx(
        failures, abs=4.0 * np.sqrt(failures / n) + 1e-9
    )
    routes = rbd.analysis_routes()
    assert routes["availability"].route == r.SIMULATED
    assert routes["availability"].engine == "python"
    assert "shared causes strike" in routes["availability"].reason
    assert routes["point_availability"].route == r.NUMERICAL
    grouped = rbd.components.get("n", rbd)
    reason = grouped.analysis_routes()["point_availability"].reason
    assert "common-cause group" in reason


def test_a_shared_cause_fails_its_members_together():
    # Every cause is shared: a member fails only when the cause strikes,
    # and so does the other, unless it is down then.
    rbd = pair_then_c(1.0)
    runs = rbd.simulate_timelines(2000.0, mc_samples=20, seed=1)
    together = 0
    for a, b in zip(runs.components["a"], runs.components["b"]):
        fails_a = a.down_intervals[:, 0]
        fails_b = b.down_intervals[:, 0]
        for one, other, others in (
            (a, fails_a, fails_b),
            (b, fails_b, fails_a),
        ):
            for t in other:
                both = np.any(others == t)
                together += both
                assert both or not (b if one is a else a).state(t)
    assert together > 0


def test_simulated_timelines_are_the_availability_runs():
    rbd = simulated_cases()["staggered tests"]
    runs = rbd.simulate_timelines(3000.0, mc_samples=60, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rbd.availability(3000.0, mc_samples=60, seed=3)
    assert np.array_equal(runs.system.uptime, result.uptimes)
    assert runs.system.failures.sum() == result.system_failures
    for node, history in runs.components.items():
        assert history.uptime.sum() == pytest.approx(
            result.node_uptime[node], rel=1e-12
        )


def test_the_runs_are_reproduced():
    rbd = pair_then_c(0.2)
    one = rbd.availability(2000.0, mc_samples=300, seed=5)
    assert rbd.availability(2000.0, mc_samples=300, seed=5).system_uptime == (
        one.system_uptime
    )
    parallel = rbd.availability(2000.0, mc_samples=300, seed=5, n_jobs=2)
    assert np.array_equal(parallel.uptimes, one.uptimes)
    with pytest.raises(NotImplementedError, match="common-cause groups"):
        rbd.availability(2000.0, mc_samples=10, seed=5, engine="numba")


def test_the_control_variate_twin_keeps_the_groups():
    # The twin is the system itself: its exact values over the window.
    rbd = pair_then_c(0.2)
    result = rbd.availability(
        2000.0, mc_samples=200, seed=1, control_variate=True
    )
    interval = result.mean_availability_interval()
    assert interval.standard_error == 0.0
    assert interval.estimate == pytest.approx(
        rbd.mission_availability(2000.0), rel=1e-12
    )
    assert "the system itself" in rbd.analysis_routes()["availability"].twin


def test_event_stepping_with_a_group():
    rbd = pair_then_c(0.5)
    rbd.initialize_event_queue(3000.0)
    times, states = [], []
    while True:
        t, up = rbd.next_event()
        times.append(t)
        states.append(up)
        if t >= 3000.0:
            break
    assert np.all(np.diff(times) >= 0.0)
    # The system alternates down and up until the window's end.
    assert all(s != u for s, u in zip(states[:-2], states[1:-1]))


def test_what_is_refused():
    rbd = pair_then_c(0.2)
    with pytest.raises(NotImplementedError, match="cannot be held"):
        rbd.availability(10.0, mc_samples=5, working_nodes=["a"])
    with pytest.raises(NotImplementedError, match="cannot be held"):
        rbd.point_availability([1.0], broken_nodes=["b"])
    down = {"a": NodeState(alive=False, down_for=1.0)}
    with pytest.raises(NotImplementedError, match="current state"):
        rbd.point_availability([1.0], state=down)
    with pytest.raises(NotImplementedError, match="starts from a state"):
        rbd.availability(10.0, mc_samples=5, state=down)


def test_a_member_that_is_not_exponential_is_not_simulated():
    member = {"reliability": W([100.0, 2.0]), "repairability": E([0.1])}
    rbd = RepairableRBD(
        PAIR,
        {"a": member, "b": member},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="exponential lives"):
        rbd.availability(100.0, mc_samples=10)
    assert rbd.analysis_routes()["availability"].route == r.REFUSED


def inspected_for_a_while(offset):
    return {
        "reliability": E([0.01]),
        "repairability": E([0.5]),
        "inspection": {
            "interval": 50.0,
            "offset": offset,
            "duration": E([1.0]),
        },
    }


@pytest.mark.parametrize(
    "components",
    [
        {"a": inspected_for_a_while(0.0), "b": inspected_for_a_while(25.0)},
        {
            x: revealed(0.01, 0.5) | {"repairability": W([3.0, 2.0])}
            for x in "ab"
        },
    ],
    ids=["tests taking time", "Weibull repairs"],
)
def test_what_the_chains_need_not_the_simulation(components):
    # The groups' chains need tests and repairs in no time, or exponential
    # repairs, but the simulation draws the causes around any: with no
    # shared cause, it is the plain system's (numerical) long run.
    plain = RepairableRBD(PAIR, components)
    for beta in (0.0, 0.3):
        rbd = RepairableRBD(
            PAIR,
            components,
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
        )
        routes = rbd.analysis_routes()
        assert routes["mean_availability"].route == r.REFUSED
        assert routes["availability"].route == r.SIMULATED
        interval = rbd.availability(
            20000.0, mc_samples=400, seed=3
        ).mean_availability_interval()
        z = (interval.estimate - plain.mean_availability()) / (
            interval.standard_error
        )
        if beta == 0.0:
            assert abs(z) < 4.0
        else:
            assert z < -20.0  # the shared cause takes both down together
