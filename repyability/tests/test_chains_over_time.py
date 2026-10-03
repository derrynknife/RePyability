"""The availability over time of a system whose components wait for repair
crews, and of a standby group (#146): their Markov chains followed over
time by uniformization, checked against closed forms (with a crew for each
component, nothing waits), the matrix exponential of chains written out by
hand, the machine-repair model, the long-run values and the simulation; and
the importance measures with crews."""

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from scipy.linalg import expm

from repyability import NodeState, RepairableRBD
from repyability.rbd import _chain_transient
from repyability.rbd import routes as r

E = surv.Exponential.from_params


def unit(lam, mu, **spec):
    return {"reliability": E([lam]), "repairability": E([mu]), **spec}


def generator(rates: dict, size: int) -> np.ndarray:
    Q = np.zeros((size, size))
    for (i, j), rate in rates.items():
        Q[i, j] += rate
    return Q - np.diag(Q.sum(axis=1))


def at(Q, start, t) -> np.ndarray:
    """``p(t)`` for each of ``t`` (rows)."""
    return np.array([start @ expm(Q * x) for x in np.atleast_1d(t)])


def steady(Q) -> np.ndarray:
    """The long-run distribution of the chain with generator ``Q``."""
    system = Q.T.copy()
    system[0] = 1.0
    return np.linalg.solve(system, first(len(Q)))


def integral(Q, start, t) -> np.ndarray:
    """``integral_0^t p(s) ds`` for each of ``t`` (rows): ``pi t + (p(0) -
    p(t)) D``, with ``D = (1 pi - Q)^-1 - 1 pi`` the chain's deviation
    matrix, the integral of ``exp(Q s) - 1 pi`` over all ``s``."""
    pi = steady(Q)
    limit = np.outer(np.ones(len(Q)), pi)
    deviation = np.linalg.inv(limit - Q) - limit
    t = np.atleast_1d(t)
    return np.outer(t, pi) + (start - at(Q, start, t)) @ deviation


def first(size: int, state: int = 0) -> np.ndarray:
    start = np.zeros(size)
    start[state] = 1.0
    return start


T = np.array([0.0, 0.5, 3.0, 20.0, 100.0, 600.0, 1e4])

# -- crews: (a || b) then c, against the independent solution -----------------

RATES = {"a": (0.01, 0.2), "b": (0.02, 0.5), "c": (0.001, 0.1)}
BRIDGE = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]


def bridge(crews=None):
    return RepairableRBD(
        BRIDGE,
        {node: unit(*rates) for node, rates in RATES.items()},
        repair_crews=crews,
    )


def closed(t, lam, mu):
    """An exponential unit's point availability from new."""
    s = lam + mu
    return mu / s + lam / s * np.exp(-s * t)


def closed_uptime(t, lam, mu):
    s = lam + mu
    return mu / s * t + lam / s**2 * -np.expm1(-s * t)


def test_with_a_crew_for_each_component_nothing_waits(monkeypatch):
    # The chain with as many crews as components (followed, though the
    # components are then independent) is the closed form, exactly; the
    # renewal equation of #117 agrees to its own error.
    chain = bridge(crews=3)
    monkeypatch.setattr(chain, "_crews_couple", lambda: True)
    free = bridge()
    A, B, C = (closed(T, *RATES[node]) for node in "abc")
    exact = (1 - (1 - A) * (1 - B)) * C
    np.testing.assert_allclose(chain.point_availability(T), exact, atol=1e-13)
    np.testing.assert_allclose(free.point_availability(T), exact, atol=2e-6)

    def system(x):
        A, B, C = (closed(x, *RATES[node]) for node in "abc")
        return (1 - (1 - A) * (1 - B)) * C

    def failing(x):
        # Each component fails at its rate while it is up, and takes the
        # system down when it is critical.
        A, B, C = (closed(x, *RATES[node]) for node in "abc")
        (la, _), (lb, _), (lc, _) = RATES.values()
        return (
            la * A * (1 - B) * C
            + lb * B * (1 - A) * C
            + lc * C * (1 - (1 - A) * (1 - B))
        )

    def integrated(f, x):
        return quad(f, 0.0, x, epsabs=1e-13, epsrel=1e-13, limit=400)[0]

    windows = T[1:-1]
    mission = [integrated(system, x) / x for x in windows]
    np.testing.assert_allclose(
        chain.mission_availability(windows), mission, atol=1e-12
    )
    events = chain.expected_events(windows)
    failures = [integrated(failing, x) for x in windows]
    np.testing.assert_allclose(events.system_failures, failures, rtol=1e-10)
    np.testing.assert_allclose(
        free.expected_events(windows).system_failures, failures, atol=1e-8
    )
    for node, (lam, mu) in RATES.items():
        up = closed_uptime(windows, lam, mu)
        np.testing.assert_allclose(
            events.node_failures[node], lam * up, rtol=1e-11
        )
        np.testing.assert_allclose(
            events.node_corrective[node], lam * up, rtol=1e-11
        )
        np.testing.assert_allclose(
            events.node_downtime[node], windows - up, rtol=1e-10
        )
    # The importance measures are the independent ones.
    for name in (
        "birnbaum_importance",
        "improvement_potential",
        "risk_achievement_worth",
        "risk_reduction_worth",
        "criticality_importance",
        "fussell_vesely",
    ):
        got, want = getattr(chain, name)(), getattr(free, name)()
        for node in RATES:
            assert got[node] == pytest.approx(want[node], rel=1e-11)


# -- crews: two components and one crew, written out by hand ------------------

LA, MA, LB, MB = 0.05, 0.4, 0.02, 0.1
# Both up; a in repair; b in repair; a in repair and b waiting; b in repair
# and a waiting.
PAIR_CHAIN = generator(
    {
        (0, 1): LA,
        (0, 2): LB,
        (1, 0): MA,
        (1, 3): LB,
        (2, 0): MB,
        (2, 4): LA,
        (3, 2): MA,
        (4, 1): MB,
    },
    5,
)
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
# The system up in each state, and the rate of its failures there.
ARRANGEMENTS = {
    "parallel": (PARALLEL, [1, 1, 1, 0, 0], [0, LB, LA, 0, 0]),
    "series": (SERIES, [1, 0, 0, 0, 0], [LA + LB, 0, 0, 0, 0]),
}


def pair(edges, **options):
    return RepairableRBD(
        edges,
        {"a": unit(LA, MA), "b": unit(LB, MB)},
        repair_crews=1,
        **options,
    )


def pi_pair() -> np.ndarray:
    return steady(PAIR_CHAIN)


@pytest.mark.parametrize("arrangement", list(ARRANGEMENTS))
def test_a_shared_crew_against_the_matrix_exponential(arrangement):
    edges, up, failing = ARRANGEMENTS[arrangement]
    rbd = pair(edges)
    start = first(5)
    np.testing.assert_allclose(
        rbd.point_availability(T), at(PAIR_CHAIN, start, T) @ up, atol=1e-13
    )
    windows = T[1:]
    np.testing.assert_allclose(
        rbd.mission_availability(windows),
        integral(PAIR_CHAIN, start, windows) @ up / windows,
        atol=1e-13,
    )
    events = rbd.expected_events(windows)
    totals = integral(PAIR_CHAIN, start, windows)
    np.testing.assert_allclose(
        events.system_failures, totals @ failing, rtol=1e-11
    )
    # a is down in the states it is in repair or waiting.
    np.testing.assert_allclose(
        events.node_downtime["a"], totals @ [0, 1, 0, 1, 1], rtol=1e-11
    )
    np.testing.assert_allclose(
        events.node_failures["b"], LB * totals @ [1, 1, 0, 0, 0], rtol=1e-11
    )
    # From a in repair (however long it has been: its repair is
    # exponential), and from the long run.
    state = {"a": NodeState(alive=False, down_for=3.0), "b": NodeState(age=9)}
    np.testing.assert_allclose(
        rbd.point_availability(T, state=state),
        at(PAIR_CHAIN, first(5, 1), T) @ up,
        atol=1e-13,
    )
    level = rbd.mean_availability()
    assert level == pytest.approx(pi_pair() @ up, rel=1e-12)
    for value in rbd.point_availability(T, state="stationary"):
        assert value == pytest.approx(level, rel=1e-12)
    assert rbd.mission_availability(50.0, state="stationary") == (
        pytest.approx(level, rel=1e-12)
    )


def test_the_importance_measures_hold_each_node_in_the_chain():
    # In parallel: with a held working the system never fails, and with a
    # held failed b has the crew to itself.
    rbd = pair(PARALLEL)
    pi = pi_pair()
    U = pi[3] + pi[4]
    assert rbd.mean_unavailability() == pytest.approx(U, rel=1e-12)
    alone_b, alone_a = LB / (LB + MB), LA / (LA + MA)
    birnbaum = rbd.birnbaum_importance()
    assert birnbaum["a"] == pytest.approx(alone_b, rel=1e-12)
    assert birnbaum["b"] == pytest.approx(alone_a, rel=1e-12)
    raw = rbd.risk_achievement_worth()
    assert raw["a"] == pytest.approx(alone_b / U, rel=1e-12)
    assert rbd.improvement_potential()["a"] == pytest.approx(U, rel=1e-12)
    assert rbd.risk_reduction_worth()["a"] == np.inf
    # In series: a is down and critical when it is in repair and b is up;
    # a minimal cut set holding a is down whenever a is.
    rbd = pair(SERIES)
    U = 1.0 - pi[0]
    criticality = rbd.criticality_importance()
    assert criticality["a"] == pytest.approx(pi[1] / U, rel=1e-12)
    assert criticality["b"] == pytest.approx(pi[2] / U, rel=1e-12)
    fv = rbd.fussell_vesely()
    assert fv["a"] == pytest.approx((pi[1] + pi[3] + pi[4]) / U, rel=1e-12)
    report = rbd.analysis_routes()
    assert report["birnbaum_importance"].route == r.EXACT
    assert "hold each node working and failed" in (
        report["birnbaum_importance"].reason
    )


def test_states_that_do_not_say_where_the_chain_is_are_refused():
    rbd = pair(PARALLEL)
    down = NodeState(alive=False)
    with pytest.raises(ValueError, match="which jobs wait"):
        rbd.point_availability(1.0, state={"a": down, "b": down})
    with pytest.raises(NotImplementedError, match="tied together"):
        rbd.point_availability(1.0, state={"a": NodeState(stationary=True)})


# -- crews: identical units, the machine-repair model -------------------------


def k_of_n(n, lam, mu, crews, k, **options):
    edges = [("s", i) for i in range(n)] + [(i, "t") for i in range(n)]
    return RepairableRBD(
        edges,
        {i: unit(lam, mu) for i in range(n)},
        k={"t": k} if k > 1 else None,
        repair_crews=crews,
        **options,
    )


def machine_repair(n, lam, mu, crews) -> np.ndarray:
    """The generator of the number of ``n`` units down."""
    rates = {}
    for j in range(n):
        rates[(j, j + 1)] = (n - j) * lam
        rates[(j + 1, j)] = min(j + 1, crews) * mu
    return generator(rates, n + 1)


@pytest.mark.parametrize("n, k, crews", [(3, 1, 1), (4, 2, 2), (5, 3, 1)])
def test_identical_units_follow_the_machine_repair_model(n, k, crews):
    lam, mu = 0.1, 0.5
    Q = machine_repair(n, lam, mu, crews)
    up = (np.arange(n + 1) <= n - k).astype(float)
    # Up to down: one of the k units up fails with n - k down.
    failing = np.zeros(n + 1)
    failing[n - k] = k * lam
    start = first(n + 1)
    rbd = k_of_n(n, lam, mu, crews, k)
    np.testing.assert_allclose(
        rbd.point_availability(T), at(Q, start, T) @ up, atol=1e-13
    )
    windows = T[1:]
    totals = integral(Q, start, windows)
    np.testing.assert_allclose(
        rbd.mission_availability(windows), totals @ up / windows, atol=1e-13
    )
    np.testing.assert_allclose(
        rbd.expected_failures(windows), totals @ failing, rtol=1e-11
    )


def test_the_capacity_over_time_with_a_shared_crew():
    # Three pumps of 50, one crew: the capacity is 50 for each pump up.
    n, lam, mu = 3, 0.1, 0.5
    Q = machine_repair(n, lam, mu, 1)
    rbd = k_of_n(n, lam, mu, 1, 1, capacity={i: 50.0 for i in range(n)})
    point = rbd.point_capacity(T)
    assert point.levels.tolist() == [0.0, 50.0, 100.0, 150.0]
    # Level 50 (3 - j) with j down.
    want = at(Q, first(n + 1), T)[:, ::-1].T
    np.testing.assert_allclose(point.probabilities, want, atol=1e-13)
    windows = T[1:]
    mission = rbd.mission_capacity(windows)
    want = (integral(Q, first(n + 1), windows) / windows[:, None])[:, ::-1].T
    np.testing.assert_allclose(mission.probabilities, want, atol=1e-13)
    settled = rbd.point_capacity(1e6)
    np.testing.assert_allclose(
        settled.probabilities,
        rbd.capacity_distribution().probabilities,
        atol=1e-13,
    )


# -- crews: the long run and the simulation -----------------------------------


def four(crews=1, **options):
    return RepairableRBD(
        [("s", "a"), ("a", "b"), ("s", "c"), ("c", "d"), ("b", "t")]
        + [("d", "t")],
        {
            "a": unit(0.02, 0.3, repair_cost=10.0),
            "b": unit(0.01, 0.1, priority=1.0, repair_cost=40.0),
            "c": unit(0.03, 0.5, repair_cost=10.0),
            "d": unit(0.005, 0.05, downtime_cost=2.0),
        },
        repair_crews=crews,
        downtime_cost_rate=20.0,
        **options,
    )


def test_the_long_run_is_reached():
    rbd = four()
    late = np.array([5e3, 1e4, 1e6])
    np.testing.assert_allclose(
        rbd.point_availability(late), rbd.mean_availability(), rtol=1e-12
    )
    failures = rbd.expected_failures(late)
    assert (failures[2] - failures[1]) / (late[2] - late[1]) == (
        pytest.approx(rbd.system_failure_frequency(), rel=1e-11)
    )
    cost = rbd.expected_cost(late).mean
    assert (cost[2] - cost[1]) / (late[2] - late[1]) == pytest.approx(
        rbd.expected_cost_rate(), rel=1e-11
    )


def test_a_shared_crew_against_the_simulation():
    rbd = four()
    window, samples = 150.0, 4000
    result = rbd.availability(window, mc_samples=samples, seed=11)
    lower, upper = result.availability_interval(confidence=0.999)
    t = np.array([5.0, 20.0, 60.0, 140.0])
    index = np.searchsorted(result.timeline, t, side="right") - 1
    exact = rbd.point_availability(t)
    assert np.all((lower[index] <= exact) & (exact <= upper[index]))
    mean = result.mean_availability_interval(confidence=0.999)
    assert mean.lower <= rbd.mission_availability(window) <= mean.upper
    failures = rbd.expected_failures(window)
    spread = np.sqrt(result.system_failures) / samples
    assert abs(result.system_failures / samples - failures) < 4 * spread
    costs = rbd.cost(window, mc_samples=samples, seed=12)
    interval = costs.mean_interval(confidence=0.999)
    assert interval.lower <= rbd.expected_cost(window).mean <= interval.upper


# -- crews: nested RBDs -------------------------------------------------------


def inner():
    """A nested RBD of two independent units in parallel (crews of its
    own)."""
    return RepairableRBD(
        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
        {"x": unit(0.05, 0.2), "y": unit(0.04, 0.25)},
    )


def with_nested():
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "n"), ("b", "n"), ("n", "t")],
        {"a": unit(LA, MA), "b": unit(LB, MB), "n": inner()},
        repair_crews=1,
    )


def test_a_nested_rbd_beside_the_crews():
    # The nested RBD is independent of the crews' chain: in series, the
    # system's availability is the product.
    rbd = with_nested()
    alone = pair(PARALLEL)
    np.testing.assert_allclose(
        rbd.point_availability(T),
        alone.point_availability(T) * inner().point_availability(T),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rbd.point_availability(T, working_nodes=["n"]),
        alone.point_availability(T),
        atol=1e-13,
    )
    windows = np.array([10.0, 80.0, 400.0])
    result = rbd.availability(80.0, mc_samples=4000, seed=3)
    mean = result.mean_availability_interval(confidence=0.999)
    assert mean.lower <= rbd.mission_availability(80.0) <= mean.upper
    # The mission by quadrature, against the product's own.
    product = [
        quad(
            lambda x: alone.point_availability(x)
            * inner().point_availability(x),
            0.0,
            w,
            epsabs=1e-10,
            limit=200,
        )[0]
        / w
        for w in windows
    ]
    np.testing.assert_allclose(
        rbd.mission_availability(windows), product, atol=1e-8
    )
    report = rbd.analysis_routes()
    assert report["point_availability"].route == r.NUMERICAL
    assert "each pattern of the nested RBDs" in (
        report["point_availability"].reason
    )
    # The expected events (#162): in series and independent, the pair's
    # failures while the nested RBD is up, and its failures while the pair
    # is up.
    assert report["expected_failures"].route == r.NUMERICAL
    grids = [np.linspace(0.0, w, 2001) for w in windows]
    s = np.concatenate(grids)
    middle = np.concatenate([0.5 * (g[1:] + g[:-1]) for g in grids])
    counted = [alone.expected_failures(s), inner().expected_failures(s)]
    steps = [
        np.concatenate([np.diff(part) for part in np.split(c, 3)])
        for c in counted
    ]
    products = inner().point_availability(middle) * steps[0]
    products += alone.point_availability(middle) * steps[1]
    want = [part.sum() for part in np.split(products, 3)]
    np.testing.assert_allclose(rbd.expected_failures(windows), want, rtol=1e-5)
    # And the capacity: the pair's, when the nested RBD carries it.
    rated = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "n"), ("b", "n"), ("n", "t")],
        {"a": unit(LA, MA), "b": unit(LB, MB), "n": inner()},
        repair_crews=1,
        capacity={"a": 1.0, "b": 1.0, "n": 2.0},
    )
    pair_rated = RepairableRBD(
        PARALLEL,
        {"a": unit(LA, MA), "b": unit(LB, MB)},
        repair_crews=1,
        capacity={"a": 1.0, "b": 1.0},
    )
    capacity = rated.point_capacity(T)
    carried = pair_rated.point_capacity(T)
    up = inner().point_availability(T)
    assert capacity.levels.tolist() == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(
        capacity.probabilities[1:], carried.probabilities[1:] * up, atol=1e-12
    )
    assert rated.analysis_routes()["point_capacity"].route == r.NUMERICAL


def test_a_crew_rbd_nested_in_another():
    # (a || b) with one crew, in series with an independent unit u.
    outer = RepairableRBD(
        [("s", "p"), ("p", "u"), ("u", "t")],
        {"p": pair(PARALLEL), "u": unit(0.01, 0.5)},
    )
    np.testing.assert_allclose(
        outer.point_availability(T),
        pair(PARALLEL).point_availability(T) * closed(T, 0.01, 0.5),
        atol=2e-6,
    )
    report = outer.analysis_routes()
    assert report["point_availability"].route == r.NUMERICAL
    assert report["expected_failures"].route == r.NUMERICAL
    window, samples = 200.0, 4000
    result = outer.availability(window, mc_samples=samples, seed=8)
    failures = outer.expected_failures(window)
    spread = np.sqrt(result.system_failures) / samples
    assert abs(result.system_failures / samples - failures) < 4 * spread
    # A crew RBD with nested RBDs of its own gives its events to another
    # (#162): in series with u, its failures while u is up, and u's while
    # it is up.
    deeper = RepairableRBD(
        [("s", "w"), ("w", "u"), ("u", "t")],
        {"w": with_nested(), "u": unit(0.01, 0.5)},
    )
    s = np.linspace(0.0, window, 4001)
    middle = 0.5 * (s[1:] + s[:-1])
    own = np.diff(with_nested().expected_failures(s))
    want = closed(middle, 0.01, 0.5) @ own + 0.01 * (
        with_nested().point_availability(middle) * closed(middle, 0.01, 0.5)
    ) @ np.diff(s)
    assert deeper.expected_failures(window) == pytest.approx(want, rel=2e-6)
    assert deeper.analysis_routes()["expected_failures"].route == r.NUMERICAL
    result = deeper.availability(window, mc_samples=samples, seed=9)
    spread = np.sqrt(result.system_failures) / samples
    assert abs(result.system_failures / samples - want) < 4 * spread


def test_ample_crews_through_the_chain_are_the_independent_values():
    # With a crew for every component nothing waits, so the crews' chain
    # around a nested RBD (#162) gives what the independent nodes do.
    def plant(crews):
        return RepairableRBD(
            [("s", "a"), ("s", "b"), ("a", "n"), ("b", "n"), ("n", "t")],
            {
                "a": unit(LA, MA, repair_cost=7.0),
                "b": unit(LB, MB, repair_cost=3.0),
                "n": inner(),
            },
            repair_crews=crews,
            capacity={"a": 1.0, "b": 1.0, "n": 2.0},
        )

    independent, chained = plant(2), plant(2)
    chained._crews_couple = lambda: True  # type: ignore[method-assign]
    windows = np.array([5.0, 60.0, 400.0])
    for name in ("system_failures", "system_downtime"):
        np.testing.assert_allclose(
            chained.expected_events(windows)[name],
            independent.expected_events(windows)[name],
            rtol=1e-6,
            atol=1e-10,
        )
    np.testing.assert_allclose(
        chained.expected_cost(windows).total,
        independent.expected_cost(windows).total,
        rtol=1e-6,
    )
    for method in ("point_capacity", "mission_capacity"):
        ours = getattr(chained, method)(windows)
        theirs = getattr(independent, method)(windows)
        assert ours.levels.tolist() == theirs.levels.tolist()
        np.testing.assert_allclose(
            ours.probabilities, theirs.probabilities, atol=1e-6
        )


# -- standby groups -----------------------------------------------------------


def group(lam, mu, crews=None, **standby):
    return RepairableRBD(
        [("s", "g"), ("g", "t")],
        {"g": {**unit(lam, mu, repair_cost=5.0), "standby": standby}},
        repair_crews=crews,
    )


def test_a_cold_standby_pair_against_the_matrix_exponential():
    # Both ready; one operating and one in repair; both in repair.
    lam, mu = 0.02, 0.25
    Q = generator({(0, 1): lam, (1, 0): mu, (1, 2): lam, (2, 1): 2 * mu}, 3)
    up, failing, units = [1, 1, 0], [0, lam, 0], [lam, lam, 0]
    start = first(3)
    rbd = group(lam, mu)
    np.testing.assert_allclose(
        rbd.point_availability(T), at(Q, start, T) @ up, atol=1e-13
    )
    windows = T[1:]
    totals = integral(Q, start, windows)
    np.testing.assert_allclose(
        rbd.mission_availability(windows), totals @ up / windows, atol=1e-12
    )
    events = rbd.expected_events(windows)
    np.testing.assert_allclose(
        events.system_failures, totals @ failing, rtol=1e-10
    )
    np.testing.assert_allclose(
        events.node_failures["g"], totals @ failing, rtol=1e-10
    )
    # Each of its units' failures is a repair, paid for.
    np.testing.assert_allclose(
        events.node_corrective["g"], totals @ units, rtol=1e-10
    )
    np.testing.assert_allclose(
        rbd.expected_cost(windows).by_category["repair"],
        5.0 * (totals @ units),
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "crews, standby",
    [
        (1, {"units": 3, "k": 2, "dormancy_factor": 0.3}),
        (None, {"switching_probability": 0.9, "dormancy_factor": 1.0}),
    ],
)
def test_a_group_follows_its_chain(crews, standby):
    from repyability.rbd import _standby_chain

    lam, mu = 0.02, 0.25
    rbd = group(lam, mu, crews, **standby)
    arrangement = rbd._standby["g"]
    chain = _standby_chain.chain(
        arrangement.units,
        arrangement.k,
        lam,
        mu,
        arrangement.dormancy_factor,
        arrangement.switching_probability,
        crews,
    )
    start = first(len(chain.states))
    np.testing.assert_allclose(
        rbd.point_availability(T),
        at(chain.generator, start, T) @ chain.up,
        atol=1e-13,
    )
    windows = T[1:]
    np.testing.assert_allclose(
        rbd.expected_failures(windows),
        integral(chain.generator, start, windows) @ chain.failures,
        rtol=1e-10,
    )
    assert rbd.point_availability(1e7) == pytest.approx(
        rbd.mean_availability(), rel=1e-12
    )
    # In its long-run state from the start.
    for state in ({"g": NodeState(stationary=True)}, "stationary"):
        np.testing.assert_allclose(
            rbd.point_availability(T, state=state),
            rbd.mean_availability(),
            rtol=1e-12,
        )
    with pytest.raises(NotImplementedError, match="standby group"):
        rbd.point_availability(1.0, state={"g": NodeState(age=3.0)})


def test_a_group_in_a_system_against_the_simulation():
    rbd = RepairableRBD(
        [("s", "g"), ("g", "u"), ("u", "t")],
        {
            "g": {**unit(0.05, 0.2, repair_cost=5.0), "standby": {}},
            "u": unit(0.01, 0.5),
        },
    )
    report = rbd.analysis_routes()
    assert report["point_availability"].route == r.NUMERICAL
    assert report["expected_cost"].route == r.NUMERICAL
    window, samples = 150.0, 4000
    result = rbd.availability(window, mc_samples=samples, seed=21)
    lower, upper = result.availability_interval(confidence=0.999)
    t = np.array([5.0, 30.0, 90.0, 140.0])
    index = np.searchsorted(result.timeline, t, side="right") - 1
    exact = rbd.point_availability(t)
    assert np.all((lower[index] <= exact) & (exact <= upper[index]))
    failures = rbd.expected_failures(window)
    spread = np.sqrt(result.system_failures) / samples
    assert abs(result.system_failures / samples - failures) < 4 * spread
    costs = rbd.cost(window, mc_samples=samples, seed=22)
    interval = costs.mean_interval(confidence=0.999)
    assert interval.lower <= rbd.expected_cost(window).mean <= interval.upper


# -- what is too stiff to follow ----------------------------------------------


def test_a_chain_whose_rates_are_too_far_apart_is_refused(monkeypatch):
    monkeypatch.setattr(_chain_transient, "MAX_WORK", 1e5)
    rbd = pair(PARALLEL)
    with pytest.raises(NotImplementedError, match="rates are too far apart"):
        RepairableRBD(
            PARALLEL,
            {"a": unit(LA, 10.0), "b": unit(LB, 1e-3)},
            repair_crews=1,
        ).point_availability(1.0)
    rbd.point_availability(1.0)
