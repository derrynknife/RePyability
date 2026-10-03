"""Two-terminal reliability of undirected networks with failing links and
nodes (#104)."""

import itertools

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import Network, NonRepairableRBD
from repyability import network as network_module
from repyability.rbd.shannon import _shannon_value_and_gradient

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


def test_too_many_paths_refuses_to_list_them(monkeypatch):
    monkeypatch.setattr(network_module, "MAX_PATHS", 3)
    net = bridge({n: E([0.01]) for n in "abcde"})
    with pytest.raises(NotImplementedError, match="too many to list"):
        net.path_sets()
    # The exact values come from the decision diagram.
    assert net.sf(10.0) == pytest.approx(
        bridge({n: E([0.01]) for n in "abcde"}).sf(10.0)
    )
    monkeypatch.setattr(network_module, "METHOD", "paths")
    listed = bridge({n: E([0.01]) for n in "abcde"})
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        listed.sf(10.0)
    assert 0.0 < listed.sf(10.0, method="simulate", seed=1) <= 1.0


def test_too_many_states_refuses(monkeypatch):
    monkeypatch.setattr(network_module, "MAX_STATES", 3)
    net = bridge({n: E([0.01]) for n in "abcde"})
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        net.sf(10.0)
    monkeypatch.setattr(network_module, "METHOD", "neither")
    with pytest.raises(ValueError, match="METHOD"):
        bridge({n: E([0.01]) for n in "abcde"}).sf(10.0)


# -- the decision diagram (#143) -------------------------------------------


def ladder(rungs):
    """Two rails joined by ``rungs`` rungs."""
    links = {}
    for i in range(rungs):
        links[f"r{i}"] = (f"a{i}", f"b{i}")
        if i + 1 < rungs:
            links[f"u{i}"] = (f"a{i}", f"a{i + 1}")
            links[f"l{i}"] = (f"b{i}", f"b{i + 1}")
    return links, "a0", f"b{rungs - 1}"


def lattice(rows, columns):
    """A grid, corner to corner."""
    links = {}
    for i in range(rows):
        for j in range(columns):
            if j + 1 < columns:
                links[f"h{i}.{j}"] = ((i, j), (i, j + 1))
            if i + 1 < rows:
                links[f"v{i}.{j}"] = ((i, j), (i + 1, j))
    return links, (0, 0), (rows - 1, columns - 1)


def built(shape, models, nodes=None):
    links, source, target = shape
    return Network(
        {name: (u, v, models(name)) for name, (u, v) in links.items()},
        source,
        target,
        nodes=nodes,
    )


def enumerated(net, p):
    """The probability that the terminals are joined, and each element's
    Birnbaum importance, by enumerating every element's state."""
    names = list(net.models)
    joined_with = {name: [0.0, 0.0] for name in names}
    total = 0.0
    for state in itertools.product([True, False], repeat=len(names)):
        works = dict(zip(names, state))
        weight = 1.0
        for name in names:
            weight *= p[name] if works[name] else 1.0 - p[name]
        parent = {}

        def find(a):
            while parent.get(a, a) != a:
                a = parent[a]
            return a

        for name, (u, v) in net.links.items():
            up = works[name] and all(
                works.get(w, True) for w in (u, v) if w in net.nodes
            )
            if up:
                parent[find(u)] = find(v)
        joined = all(
            works.get(w, True)
            for w in (net.source, net.target)
            if w in net.nodes
        ) and find(net.source) == find(net.target)
        if joined:
            total += weight
            for name in names:
                joined_with[name][0 if works[name] else 1] += weight / (
                    p[name] if works[name] else 1.0 - p[name]
                )
    return total, {n: w - f for n, (w, f) in joined_with.items()}


@pytest.mark.parametrize(
    "shape, nodes",
    [
        (ladder(4), None),
        (lattice(3, 3), None),
        (lattice(3, 3), {(1, 1): 0.1, (0, 0): 0.05, (2, 2): 0.02}),
        (
            (
                {
                    "a": ("s", "t"),
                    "b": ("s", "t"),
                    "c": ("t", "z"),
                    "d": ("q", "r"),
                    "e": ("r", "s"),
                },
                "s",
                "t",
            ),
            {"r": 0.3},
        ),
    ],
    ids=["ladder", "grid", "grid with failing nodes", "multi-links"],
)
def test_the_diagram_is_the_enumeration(shape, nodes):
    q = {name: 0.05 + 0.01 * i for i, name in enumerate(shape[0])}
    net = built(
        shape,
        lambda name: F(q[name]),
        nodes={n: F(v) for n, v in (nodes or {}).items()},
    )
    p = {
        name: 1.0 - float(np.ravel(model.ff(1.0))[0])
        for name, model in net.models.items()
    }
    joined, importance = enumerated(net, p)
    assert net.sf() == pytest.approx(joined, rel=1e-12)
    assert net.ff() == pytest.approx(1.0 - joined, rel=1e-10)
    got = net.birnbaum_importance()
    for name, value in importance.items():
        assert got[name] == pytest.approx(value, rel=1e-10, abs=1e-15)


@pytest.mark.parametrize(
    "shape, nodes",
    [
        (ladder(6), None),
        (lattice(3, 4), None),
        (lattice(4, 4), {(1, 1): 0.1, (2, 3): 0.05}),
        (
            (
                {
                    "a": ("s", "x"),
                    "b": ("s", "y"),
                    "c": ("x", "y"),
                    "d": ("x", "t"),
                    "e": ("y", "t"),
                },
                "s",
                "t",
            ),
            {"s": 0.01, "x": 0.1, "t": 0.02},
        ),
    ],
    ids=[
        "ladder",
        "grid",
        "grid with failing nodes",
        "bridge, terminals fail",
    ],
)
def test_the_diagram_is_the_paths_decomposition(shape, nodes, monkeypatch):
    def make():
        return built(
            shape,
            lambda name: W([100.0 + 5 * len(str(name)), 1.5]),
            nodes={n: E([v / 100]) for n, v in (nodes or {}).items()},
        )

    t = np.array([5.0, 40.0, 120.0])
    diagram = make()
    monkeypatch.setattr(network_module, "METHOD", "paths")
    paths = make()
    np.testing.assert_allclose(diagram.sf(t), paths.sf(t), rtol=1e-12)
    np.testing.assert_allclose(diagram.ff(t), paths.ff(t), rtol=1e-11)
    assert diagram.cut_sets() == paths.cut_sets()
    for name, value in paths.birnbaum_importance(t).items():
        np.testing.assert_allclose(
            diagram.birnbaum_importance(t)[name], value, rtol=1e-9, atol=1e-15
        )
    assert diagram.mean() == pytest.approx(paths.mean(), rel=1e-9)


def test_a_small_probability_keeps_its_precision():
    # Two parallel links in series with a third: ff = q^2 + q3 - q^2 q3.
    net = Network(
        {
            "a": ("s", "m", F(1e-9)),
            "b": ("s", "m", F(1e-9)),
            "c": ("m", "t", F(1e-12)),
        },
        "s",
        "t",
    )
    assert net.ff() == pytest.approx(1e-18 + 1e-12 - 1e-30, rel=1e-12)
    importance = net.birnbaum_importance()
    assert importance["a"] == pytest.approx(1e-9 * (1 - 1e-12), rel=1e-12)
    assert importance["c"] == pytest.approx(1 - 1e-18, rel=1e-15)


def test_a_grid_beyond_the_paths_is_exact():
    # Corner to corner, a 7 by 7 grid has about 575 million simple paths.
    net = built(lattice(7, 7), lambda name: E([0.01]))
    assert net.sf(10.0) == pytest.approx(
        net.sf(10.0, method="simulate", mc_samples=20_000, seed=3), abs=0.01
    )
    # Each link's importance, against its definition.
    importance = net.birnbaum_importance(10.0)
    plan = net._decomposition().steps()
    p, q = net._probabilities(np.array([10.0]), net.models)
    for name in ("h0.0", "v3.3", "h6.5"):
        works = _shannon_value_and_gradient(
            plan, {**p, name: 1.0}, {**q, name: 0.0}
        )[0]
        fails = _shannon_value_and_gradient(
            plan, {**p, name: 0.0}, {**q, name: 1.0}
        )[0]
        assert importance[name] == pytest.approx(
            float(np.ravel(works - fails)[0]), rel=1e-9
        )
    with pytest.raises(NotImplementedError, match="too many to list"):
        net.path_sets()


def random_network(rng):
    """A random network of a few nodes, some of which fail, and its
    elements' probabilities of working (some very near 1)."""
    size = int(rng.integers(3, 9))
    links = {}
    for j in range(int(rng.integers(2, 16))):
        u, v = rng.choice(size, 2, replace=False)
        links[f"L{j}"] = (int(u), int(v))
    touched = {n for link in links.values() for n in link}
    failing = {n for n in touched if rng.random() < 0.3}
    p = {name: float(rng.uniform(0.01, 0.99)) for name in [*links, *failing]}
    for name in p:
        if rng.random() < 0.1:
            p[name] = 1.0 - 10.0 ** -float(rng.integers(6, 14))
    return links, failing, p


def test_the_levels_are_evaluated_as_the_steps_are():
    # The diagram is built and evaluated a level at a time (#173): its
    # value is, to the last bit, that of evaluating its steps one by one,
    # and its gradient the same to rounding; it is reduced (no step
    # repeats another, or has its two branches equal).
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(300):
        links, failing, p = random_network(rng)
        touched = {n for link in links.values() for n in link}
        if not {0, 1} <= touched:
            continue
        net = Network(
            {name: (u, v, F(1.0 - p[name])) for name, (u, v) in links.items()},
            0,
            1,
            nodes={n: F(1.0 - p[n]) for n in failing},
        )
        plan = net._decomposition()
        steps = plan.steps()
        q = {name: 1.0 - value for name, value in p.items()}
        for terminals in ((0.0, 1.0), (1.0, 0.0)):
            value, gradient = _shannon_value_and_gradient(
                steps, p, q, terminals
            )
            assert network_module._evaluate(plan, p, q, (), terminals) == value
            levels, by_levels = network_module._value_and_gradient(
                plan, p, q, (), terminals
            )
            assert levels == value
            for name, change in gradient.items():
                assert by_levels[name] == pytest.approx(
                    change, rel=1e-12, abs=1e-300
                )
        assert len(set(steps[0])) == len(steps[0])
        assert all(active != inactive for _, active, inactive in steps[0])
        checked += 1
    assert checked > 200


def test_the_paths_plan_is_evaluated_by_levels(monkeypatch):
    monkeypatch.setattr(network_module, "METHOD", "paths")
    rng = np.random.default_rng(12)
    for _ in range(100):
        links, failing, p = random_network(rng)
        touched = {n for link in links.values() for n in link}
        if not {0, 1} <= touched:
            continue
        net = Network(
            {name: (u, v, F(1.0 - p[name])) for name, (u, v) in links.items()},
            0,
            1,
            nodes={n: F(1.0 - p[n]) for n in failing},
        )
        q = {name: 1.0 - value for name, value in p.items()}
        steps = network_module._shannon_plan(net._simple_paths())
        if not steps[0]:
            continue
        expected = _shannon_value_and_gradient(steps, p, q)[0]
        plan = net._decomposition()
        assert network_module._evaluate(plan, p, q, ()) == expected


def test_times_are_evaluated_in_turn(monkeypatch):
    # The values of many times are worked out a few at a time, when the
    # diagram is large: the same values.
    net = built(lattice(5, 5), lambda name: W([100.0, 1.5]))
    t = np.linspace(1.0, 200.0, 37)
    whole = net.sf(t), net.ff(t), net.birnbaum_importance(t)
    monkeypatch.setattr(network_module, "_EVALUATION_SIZE", 1)
    np.testing.assert_array_equal(net.sf(t), whole[0])
    np.testing.assert_array_equal(net.ff(t), whole[1])
    for name, value in net.birnbaum_importance(t).items():
        np.testing.assert_array_equal(value, whole[2][name])


def test_a_ten_by_ten_grid_is_exact():
    # Corner to corner, 1.9 million states before its decisions: refused
    # before #173, when its states were worked out one by one.
    net = built(lattice(10, 10), lambda name: F(0.1))
    reliability = net.sf()
    simulated = net.sf(method="simulate", mc_samples=20_000, seed=4)
    assert reliability == pytest.approx(simulated, abs=0.005)
    assert net.ff() == pytest.approx(1.0 - reliability, rel=1e-12)


def test_terminals_that_cannot_meet():
    net = Network({"a": ("s", "x", F(0.1)), "b": ("y", "t", F(0.1))}, "s", "t")
    assert net.sf() == 0.0
    assert net.ff() == 1.0
    assert net.cut_sets() == {frozenset()}


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
