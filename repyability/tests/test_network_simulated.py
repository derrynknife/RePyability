"""A network of repaired elements simulated as a process, and the time to
the first loss of its connection.

The connection the simulation finds (a search along the working links)
is checked against the network's decision diagram on every state of
small networks; a network drawn as a block diagram is checked to draw the
same numbers as the search; and the simulated values against the exact
ones (within 4 standard errors, the seeds fixed), the first loss's exact
values against the Markov chain of the bridge's 32 states, written out.
"""

import itertools

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import Network, RepairableRBD
from repyability.network import _JOINED, _evaluate
from repyability.rbd.routes import Refused

E = surv.Exponential.from_params
W = surv.Weibull.from_params

BRIDGE = {
    "a": ("s", "x"),
    "b": ("s", "y"),
    "c": ("x", "y"),
    "d": ("x", "t"),
    "e": ("y", "t"),
}
BRIDGE_PATHS = [{"a", "d"}, {"b", "e"}, {"a", "c", "e"}, {"b", "c", "d"}]
RATES = {
    "a": (0.010, 0.50),
    "b": (0.020, 0.40),
    "c": (0.030, 0.30),
    "d": (0.015, 0.25),
    "e": (0.025, 0.60),
}


def spec(life, repair):
    return {"reliability": life, "repairability": repair}


def exponential_bridge(**options):
    return Network(
        {
            name: (u, v, spec(E([RATES[name][0]]), E([RATES[name][1]])))
            for name, (u, v) in BRIDGE.items()
        },
        "s",
        "t",
        **options,
    )


def structures():
    """Networks of every kind of connection, with failing nodes."""
    link = spec(E([0.01]), E([0.5]))
    links = {name: (u, v, link) for name, (u, v) in BRIDGE.items()}
    return {
        "bridge with nodes": Network(
            links, "s", "t", nodes={"x": link, "s": link}
        ),
        "three terminals": Network(
            links, terminals=["s", "t", "x"], nodes={"y": link}
        ),
        "all terminals": Network(links, terminals="all"),
        "directed with a cycle": Network(
            {**links, "back": ("y", "x", link)},
            "s",
            "t",
            nodes={"x": link},
            directed=True,
        ),
        "directed all terminals": Network(
            links, source="s", terminals="all", directed=True
        ),
        "doubled link": Network(
            {**links, "c2": ("y", "x", link)}, "s", "t", nodes={"t": link}
        ),
    }


@pytest.mark.parametrize("name", sorted(structures()))
def test_the_search_is_the_decision_diagram_on_every_state(name):
    net = structures()[name]
    joined = net._repaired._decomposition()
    works = joined.structure_function()
    plan = net._decomposition()
    elements = list(net.models)
    for state in itertools.product((False, True), repeat=len(elements)):
        status = dict(zip(elements, state))
        p = {n: float(status[n]) for n in plan.names}
        q = {n: 1.0 - v for n, v in p.items()}
        expected = bool(_evaluate(plan, p, q, (), _JOINED) > 0.5)
        assert works(status) is expected, status
        assert joined.works(status) is expected


def test_a_drawn_network_draws_what_the_search_draws():
    # Simulated as its diagram, or by the search over its connection,
    # the network gives the same histories for a seed.
    c = {
        "a": spec(E([0.01]), E([0.5])),
        "b": spec(W([80.0, 1.5]), E([0.4])),
        "c": spec(E([0.03]), W([3.0, 2.0])),
        "d": spec(W([120.0, 0.8]), W([4.0, 2.0])),
        "e": spec(E([0.002]), E([0.05])),
    }
    net = Network(
        {n: (u, v, c[n]) for n, (u, v) in BRIDGE.items()},
        "s",
        "t",
        nodes={"x": spec(W([300.0, 1.2]), E([0.2]))},
        directed=True,
    )
    drawn = net._simulated()
    assert isinstance(drawn, RepairableRBD) and drawn is not net._repaired
    for options in ({"control_variate": False}, {}):
        ours = drawn.availability(200.0, mc_samples=500, seed=3, **options)
        theirs = net._repaired.availability(
            200.0, mc_samples=500, seed=3, **options
        )
        np.testing.assert_array_equal(ours.timeline, theirs.timeline)
        np.testing.assert_array_equal(ours.availability, theirs.availability)
        assert ours.system_failures == theirs.system_failures
        assert ours.mean_availability == theirs.mean_availability
    public = net.availability(200.0, mc_samples=500, seed=3)
    np.testing.assert_array_equal(public.availability, ours.availability)
    first = dict(method="simulate", t_simulation=3000, mc_samples=300, seed=2)
    np.testing.assert_array_equal(
        net.first_failure_sf([10.0, 50.0], **first),
        net._repaired.first_failure_sf([10.0, 50.0], **first),
    )


def test_a_bridge_is_simulated_by_the_search():
    # Within 4 standard errors of the exact values: the unavailability at
    # a time, the losses of the connection by the window's end, and the
    # first loss.
    net = exponential_bridge()
    assert net._simulated() is net._repaired
    n, t = 4000, 200.0
    result = net.availability(t, mc_samples=n, seed=7)
    assert result.n_simulations == n
    assert result.mean_availability == pytest.approx(
        net.mission_availability(t), rel=1e-12
    )
    for x in (20.0, 100.0, 199.0):
        i = np.searchsorted(result.timeline, x, side="right") - 1
        q = float(net.point_unavailability(x))
        se = np.sqrt(q * (1 - q) / n)
        assert abs((1 - result.availability[i]) - q) < 4 * se, x
    # The losses: the count's variance is at most its mean's, a Poisson
    # count's bound (Bernoulli-like losses are under-dispersed), times
    # two for safety from clustering.
    w = float(net.expected_failures(t))
    se = np.sqrt(2 * w / n)
    assert abs(result.system_failures / n - w) < 4 * se
    x = np.array([20.0, 100.0])
    exact = net.first_failure_sf(x)
    simulated = net.first_failure_sf(
        x, method="simulate", t_simulation=t, mc_samples=n, seed=7
    )
    se = np.sqrt(exact * (1 - exact) / n)
    assert np.all(np.abs(simulated - exact) < 4 * se)
    mean = net.mean_time_to_first_failure(
        method="simulate", t_simulation=5000.0, mc_samples=n, seed=7
    )
    assert (
        abs(mean.estimate - net.mean_time_to_first_failure())
        < 4 * mean.standard_error
    )


def bridge_chain():
    """The Markov chain of the bridge's 32 states (a bit per link, set
    while it is down), the states with the terminals parted absorbing:
    its generator over the states with them joined, from all up."""
    names = list(BRIDGE)
    joined = [
        s
        for s in range(32)
        if any(
            all(not s >> names.index(n) & 1 for n in path)
            for path in BRIDGE_PATHS
        )
    ]
    index = {s: i for i, s in enumerate(joined)}
    q = np.zeros((len(joined), len(joined)))
    for s in joined:
        for k, name in enumerate(names):
            lam, mu = RATES[name]
            rate = mu if s >> k & 1 else lam
            q[index[s], index[s]] -= rate
            other = s ^ (1 << k)
            if other in index:
                q[index[s], index[other]] += rate
    return q, index[0]


def test_the_first_loss_is_the_bridge_chain():
    net = exponential_bridge()
    q, start = bridge_chain()
    x = np.array([0.0, 5.0, 50.0, 400.0])
    expected = [expm(q * t)[start].sum() for t in x]
    np.testing.assert_allclose(net.first_failure_sf(x), expected, rtol=1e-10)
    mean = np.linalg.solve(-q, np.ones(len(q)))[start]
    assert net.mean_time_to_first_failure() == pytest.approx(mean, rel=1e-10)
    routes = net.analysis_routes()
    assert routes["first_failure_sf"].route == "numerical"
    assert routes["availability"].route == "simulated"


def test_a_series_network_loses_its_connection_at_its_first_failure():
    # Links in series (one way through the nodes): no repair comes before
    # the first loss, so it is the product of the lives, whatever they are.
    lives = [W([100.0, 2.0]), W([80.0, 0.7]), E([0.01])]
    net = Network(
        {
            name: (i, i + 1, spec(life, E([0.5])))
            for i, (name, life) in enumerate(zip("abc", lives))
        },
        0,
        3,
    )
    x = np.array([1.0, 10.0, 100.0])
    expected = np.prod([life.sf(x) for life in lives], axis=0)
    np.testing.assert_allclose(net.first_failure_sf(x), expected, rtol=1e-12)


def test_a_network_too_large_to_work_out_is_simulated(monkeypatch):
    from repyability import network

    monkeypatch.setattr(network, "MAX_STATES", 3)
    net = exponential_bridge()
    routes = net.analysis_routes()
    assert routes["availability"].route == "simulated"
    result = net.availability(50.0, mc_samples=200, seed=1)
    assert result.n_simulations == 200
    with pytest.raises(NotImplementedError, match="availability") as error:
        net.point_availability(1.0)
    assert routes["point_availability"].reason == str(error.value)
    assert routes["first_failure_sf"].route == "refused"
    with pytest.raises(NotImplementedError) as error:
        net.first_failure_sf(1.0)
    assert routes["first_failure_sf"].reason == str(error.value)
    simulated = net.first_failure_sf(
        [20.0], method="simulate", t_simulation=100.0, mc_samples=200, seed=1
    )
    assert 0.0 < float(simulated[0]) < 1.0


def test_what_the_simulation_refuses():
    link = spec(E([0.01]), E([0.5]))
    mixed = Network(
        {
            "a": ("s", "t", link),
            "p": ("s", "t", 0.1),
            "l": ("s", "t", W([9, 2])),
        },
        "s",
        "t",
    )
    for call in (
        lambda net: net.availability(10.0, mc_samples=10, seed=1),
        lambda net: net.first_failure_sf(1.0),
        lambda net: net.mean_time_to_first_failure(),
    ):
        with pytest.raises(NotImplementedError, match=r"\['l', 'p'\]"):
            call(mixed)
        lives = Network({"a": ("s", "t", W([9, 2]))}, "s", "t")
        with pytest.raises(Refused, match="component specs"):
            call(lives)
