"""Two-terminal reliability of undirected networks with failing links and
nodes (#104)."""

import itertools

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import Network, NonRepairableRBD
from repyability import network as network_module

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = FixedEventProbability.from_params


def bridge(models, nodes=None):
    return Network(
        {
            "a": ("s", "x", models["a"]),
            "b": ("s", "y", models["b"]),
            "c": ("x", "y", models["c"]),
            "d": ("x", "t", models["d"]),
            "e": ("y", "t", models["e"]),
        },
        "s",
        "t",
        nodes=nodes,
    )


def test_the_bridge_in_closed_form():
    p = 0.9
    net = bridge({n: F(1 - p) for n in "abcde"})
    assert net.is_fixed
    assert net.sf() == pytest.approx(
        2 * p**2 + 2 * p**3 - 5 * p**4 + 2 * p**5, rel=1e-14
    )
    assert net.ff() == pytest.approx(1 - net.sf(), rel=1e-12)
    assert net.path_sets() == {
        frozenset("ad"),
        frozenset("be"),
        frozenset("ace"),
        frozenset("bcd"),
    }
    assert net.cut_sets() == {
        frozenset("ab"),
        frozenset("de"),
        frozenset("ace"),
        frozenset("bcd"),
    }
    # Through time: each link's reliability at t, in the same formula.
    models = {n: W([100.0 + 10 * i, 1.5]) for i, n in enumerate("abcde")}
    net = bridge(models)
    t = np.array([10.0, 50.0, 120.0])
    r = {n: models[n].sf(t) for n in models}
    expected = r["c"] * (1 - (1 - r["a"]) * (1 - r["b"])) * (
        1 - (1 - r["d"]) * (1 - r["e"])
    ) + (1 - r["c"]) * (1 - (1 - r["a"] * r["d"]) * (1 - r["b"] * r["e"]))
    np.testing.assert_allclose(net.sf(t), expected, rtol=1e-12)
    assert isinstance(net.sf(50.0), float)


def connected(links, up, nodes_up, source, target):
    """Whether the working links (through working nodes) join the
    terminals, by breadth-first search."""
    if not (nodes_up.get(source, True) and nodes_up.get(target, True)):
        return False
    seen, frontier = {source}, [source]
    while frontier:
        node = frontier.pop()
        for name, (u, v) in links.items():
            if not up[name] or node not in (u, v):
                continue
            other = v if node == u else u
            if other in seen or not nodes_up.get(other, True):
                continue
            if other == target:
                return True
            seen.add(other)
            frontier.append(other)
    return False


def test_small_networks_by_enumerating_their_states():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(40):
        size = int(rng.integers(3, 6))
        names = list(range(size))
        links = {}
        for j in range(int(rng.integers(size, size + 4))):
            u, v = rng.choice(size, 2, replace=False)
            links[f"L{j}"] = (int(u), int(v), F(float(rng.uniform(0.05, 0.6))))
        touched = {n for u, v, _ in links.values() for n in (u, v)}
        if not {0, size - 1} <= touched:
            continue
        nodes = {
            n: F(float(rng.uniform(0.05, 0.3)))
            for n in names
            if n in touched and rng.random() < 0.4
        }
        net = Network(links, 0, size - 1, nodes=nodes)
        elements = list(links) + list(nodes)
        q = {
            **{n: links[n][2].ff(1.0) for n in links},
            **{n: nodes[n].ff(1.0) for n in nodes},
        }
        expected = 0.0
        for states in itertools.product([True, False], repeat=len(elements)):
            state = dict(zip(elements, states))
            chance = np.prod(
                [1 - q[e] if state[e] else q[e] for e in elements]
            )
            if connected(
                {n: links[n][:2] for n in links},
                {n: state[n] for n in links},
                {n: state[n] for n in nodes},
                0,
                size - 1,
            ):
                expected += chance
        assert net.sf() == pytest.approx(expected, rel=1e-12, abs=1e-15)
        assert net.ff() == pytest.approx(1 - expected, rel=1e-10, abs=1e-15)
        checked += 1
    assert checked > 20


def test_series_and_parallel_networks_are_their_diagrams():
    models = [W([100.0, 2.0]), E([0.01]), W([200.0, 1.2])]
    t = np.array([5.0, 40.0, 150.0])
    chain = Network(
        {f"L{i}": (i, i + 1, m) for i, m in enumerate(models)}, 0, 3
    )
    series = NonRepairableRBD(
        [("s", "L0"), ("L0", "L1"), ("L1", "L2"), ("L2", "t")],
        {f"L{i}": m for i, m in enumerate(models)},
    )
    np.testing.assert_allclose(chain.sf(t), series.sf(t), rtol=1e-12)
    bundle = Network(
        {f"L{i}": ("s", "t", m) for i, m in enumerate(models)}, "s", "t"
    )
    parallel = NonRepairableRBD(
        [("s", f"L{i}") for i in range(3)]
        + [(f"L{i}", "t") for i in range(3)],
        {f"L{i}": m for i, m in enumerate(models)},
    )
    np.testing.assert_allclose(bundle.sf(t), parallel.sf(t), rtol=1e-12)
    # Means: 1 / sum of rates in series; the parallel diagram's otherwise.
    rates = [0.01, 0.02, 0.05]
    exp_chain = Network(
        {f"L{i}": (i, i + 1, E([r])) for i, r in enumerate(rates)}, 0, 3
    )
    assert exp_chain.mean() == pytest.approx(1 / sum(rates), rel=1e-9)
    assert bundle.mean() == pytest.approx(parallel.mean(), rel=1e-8)


def test_failing_nodes_and_terminals():
    # A node on every path is in series with the network; a terminal too.
    links = {n: F(0.1) for n in "abcde"}
    net = bridge(links, nodes={"x": F(0.2), "t": F(0.05)})
    plain = bridge(links)
    assert all("t" in path for path in net.path_sets())
    # Enumerate x's and t's states over the bridge without them.
    no_x = Network(
        {"b": ("s", "y", F(0.1)), "e": ("y", "t", F(0.1))}, "s", "t"
    )
    expected = 0.95 * (0.8 * plain.sf() + 0.2 * no_x.sf())
    assert net.sf() == pytest.approx(expected, rel=1e-12)


def test_birnbaum_importance_by_conditioning():
    models = {n: F(0.1 + 0.05 * i) for i, n in enumerate("abcde")}
    net = bridge(models)
    importance = net.birnbaum_importance()
    for name in "abcde":
        works = bridge({**models, name: F(0.0)}).sf()
        fails = bridge({**models, name: F(1.0)}).sf()
        assert importance[name] == pytest.approx(works - fails, rel=1e-12)


def test_the_simulation_agrees():
    models = {
        n: W([100.0 + 15 * i, 1.0 + 0.3 * i]) for i, n in enumerate("abcde")
    }
    net = bridge(models, nodes={"y": E([0.002])})
    t = np.array([20.0, 60.0, 150.0])
    exact = net.sf(t)
    simulated = net.sf(t, method="simulate", mc_samples=40_000, seed=3)
    np.testing.assert_allclose(simulated, exact, atol=0.01)
    assert net.mean(method="simulate", mc_samples=40_000, seed=3) == (
        pytest.approx(net.mean(), rel=0.02)
    )
    a = net.random(1_000, seed=7)
    assert np.array_equal(a, net.random(1_000, seed=7))
    # Each lifetime is the last time the terminals were joined.
    for life in a[:20]:
        assert life >= 0.0


def test_too_many_paths_refuses(monkeypatch):
    monkeypatch.setattr(network_module, "MAX_PATHS", 3)
    net = bridge({n: E([0.01]) for n in "abcde"})
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        net.sf(10.0)
    assert 0.0 < net.sf(10.0, method="simulate", seed=1) <= 1.0


@pytest.mark.parametrize(
    "links, source, target, nodes, message",
    [
        ({}, "s", "t", None, "non-empty dict"),
        ({"a": ("s", "t")}, "s", "t", None, r"\(node, node, model\)"),
        ({"a": ("s", "s", F(0.1))}, "s", "s", None, "to itself"),
        ({"a": ("s", "t", 0.1)}, "s", "t", None, "sf and ff"),
        ({"a": ("s", "t", F(0.1))}, "s", "s", None, "two nodes"),
        ({"a": ("s", "t", F(0.1))}, "s", "z", None, "in no link"),
        ({"a": ("s", "t", F(0.1))}, "s", "t", {"z": F(0.1)}, "in no link"),
        (
            {"a": ("s", "a", F(0.1)), "b": ("a", "t", F(0.1))},
            "s",
            "t",
            {"a": F(0.1)},
            "name of a link",
        ),
    ],
)
def test_the_network_is_checked(links, source, target, nodes, message):
    with pytest.raises(ValueError, match=message):
        Network(links, source, target, nodes=nodes)


def test_the_methods_are_checked():
    net = bridge({n: E([0.01]) for n in "abcde"})
    with pytest.raises(ValueError, match="x is required"):
        net.sf()
    with pytest.raises(ValueError, match="'exact' or 'simulate'"):
        net.sf(1.0, method="guess")
    with pytest.raises(ValueError, match="only to method='simulate'"):
        net.sf(1.0, mc_samples=10)
    with pytest.raises(ValueError, match="no lifetime"):
        bridge({n: F(0.1) for n in "abcde"}).mean()
    assert repr(net) == "Network(5 links, 0 failing nodes, 's' to 't')"
