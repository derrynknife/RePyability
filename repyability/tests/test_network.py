"""Two-terminal reliability of undirected networks with failing links and
nodes (#104)."""

import itertools
import re

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import Network, NonRepairableRBD
from repyability import network as network_module
from repyability.rbd.shannon import _shannon_value_and_gradient
from repyability.tests.catalogue import BRIDGE

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
        ({"a": ("s", "t", "0.1")}, "s", "t", None, "sf and ff"),
        ({"a": ("s", "t", 1.1)}, "s", "t", None, r"in \[0, 1\]"),
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


# -- the measures, lives, uncertainty, saving and routes (#315) ------------


def as_diagram(models, nodes=None):
    """The bridge as a block diagram: the links as nodes in the same
    roles (the catalogue's ``BRIDGE``)."""
    return NonRepairableRBD(BRIDGE, {**models, **(nodes or {})})


def each_agrees(a, b, rtol=1e-12):
    assert set(a) == set(b)
    for key in b:
        np.testing.assert_allclose(a[key], b[key], rtol=rtol, atol=1e-300)


#: The measures a network and a diagram define alike, with their options.
MEASURES = [
    ("improvement_potential", {}),
    ("risk_achievement_worth", {}),
    ("risk_reduction_worth", {}),
    ("criticality_importance", {}),
    ("criticality_importance", {"kind": "success"}),
    ("fussell_vesely", {}),
    ("fussell_vesely", {"fv_type": "p"}),
    ("fussell_vesely", {"method": "rare_event"}),
    ("fussell_vesely", {"fv_type": "p", "method": "rare_event"}),
    ("differential_importance", {}),
    ("differential_importance", {"change": "proportional"}),
    (
        "differential_importance",
        {"change": "proportional", "kind": "success"},
    ),
    ("differential_importance", {"groups": {"ends": "abde", "mid": "c"}}),
    ("joint_importance", {}),
]


@pytest.mark.parametrize(
    "models",
    [
        {n: F(0.1 + 0.05 * i) for i, n in enumerate("abcde")},
        {n: W([100.0 + 10 * i, 1.2 + 0.2 * i]) for i, n in enumerate("abcde")},
    ],
    ids=["fixed", "Weibull"],
)
def test_the_measures_are_the_diagram_s(models):
    # Each measure of the bridge network is the diagram's of the same
    # system, at each time: the measures are its definitions over the
    # same values (#315).
    net, rbd = bridge(models), as_diagram(models)
    fixed = net.is_fixed
    times = [None] if fixed else [10.0, np.array([10.0, 50.0, 120.0])]
    for x in times:
        for name, options in MEASURES:
            each_agrees(
                getattr(net, name)(x, **options),
                getattr(rbd, name)(x, **options),
            )
    each_agrees(net.structural_importance(), rbd.structural_importance())
    if fixed:
        return
    each_agrees(
        net.barlow_proschan_importance(), rbd.barlow_proschan_importance()
    )
    each_agrees(
        net.barlow_proschan_importance([20.0, 80.0]),
        rbd.barlow_proschan_importance([20.0, 80.0]),
    )
    assert net.time_to_reliability(0.5) == pytest.approx(
        rbd.time_to_reliability(0.5), rel=1e-12
    )
    np.testing.assert_allclose(
        net.bx_life([5, 50]), rbd.bx_life([5, 50]), rtol=1e-12
    )
    assert net.bx_life(10, upper_bound=200.0) == pytest.approx(
        net.time_to_reliability(0.9), rel=1e-12
    )


def test_failing_nodes_are_nodes_in_series_with_their_links():
    # A network whose nodes fail, against the diagram with those nodes
    # drawn in series with their links: a node between two links, and a
    # failing terminal.
    models = {
        "a": W([100.0, 1.5]),
        "b": W([120.0, 2.0]),
        "c": E([0.004]),
        "d": W([80.0, 1.2]),
    }
    nodes = {"m": E([0.003]), "t": W([300.0, 3.0])}
    net = Network(
        {
            "a": ("s", "m", models["a"]),
            "b": ("m", "t", models["b"]),
            "c": ("s", "t", models["c"]),
            "d": ("m", "t", models["d"]),
        },
        "s",
        "t",
        nodes=nodes,
    )
    rbd = NonRepairableRBD(
        [
            ("s", "a"),
            ("a", "m"),
            ("m", "b"),
            ("m", "d"),
            ("b", "t"),
            ("d", "t"),
            ("s", "c"),
            ("c", "t"),
            ("t", "end"),
        ],
        {**models, **nodes},
    )
    for x in (10.0, np.array([5.0, 60.0, 200.0])):
        np.testing.assert_allclose(net.sf(x), rbd.sf(x), rtol=1e-12)
        each_agrees(net.birnbaum_importance(x), rbd.birnbaum_importance(x))
        for name, options in MEASURES:
            if "groups" in options:
                continue
            each_agrees(
                getattr(net, name)(x, **options),
                getattr(rbd, name)(x, **options),
            )
    each_agrees(net.structural_importance(), rbd.structural_importance())
    each_agrees(
        net.barlow_proschan_importance(), rbd.barlow_proschan_importance()
    )
    assert net.bx_life(10) == pytest.approx(rbd.bx_life(10), rel=1e-12)


def test_small_measures_keep_their_precision():
    # Two parallel links in series with a third: ff = q^2 + q3 - q^2 q3,
    # and each measure in closed form from it.
    q, q3 = 1e-9, 1e-12
    net = Network(
        {
            "a": ("s", "m", F(q)),
            "b": ("s", "m", F(q)),
            "c": ("m", "t", F(q3)),
        },
        "s",
        "t",
    )
    ff = q**2 + q3 - q**2 * q3
    assert net.criticality_importance()["c"] == pytest.approx(
        (1 - q**2) * q3 / ff, rel=1e-12
    )
    assert net.criticality_importance()["a"] == pytest.approx(
        q * (1 - q3) * q / ff, rel=1e-12
    )
    assert net.fussell_vesely()["c"] == pytest.approx(q3 / ff, rel=1e-12)
    assert net.fussell_vesely()["a"] == pytest.approx(q**2 / ff, rel=1e-12)
    assert net.risk_achievement_worth()["c"] == pytest.approx(
        1 / ff, rel=1e-12
    )
    assert net.risk_reduction_worth()["c"] == pytest.approx(
        ff / q**2, rel=1e-12
    )
    assert net.risk_reduction_worth()["a"] == pytest.approx(ff / q3, rel=1e-12)
    assert net.improvement_potential()["c"] == pytest.approx(
        (1 - q**2) * q3, rel=1e-12
    )
    joint = net.joint_importance()
    assert joint["a", "b"] == pytest.approx(-(1 - q3), rel=1e-12)
    assert joint["a", "c"] == pytest.approx(q, rel=1e-12)


def fitted_bridge():
    """The bridge with fitted link models: two of one type (one fitted
    object), two of another, and one given."""
    rng = np.random.default_rng(0)
    first = surv.Weibull.fit(rng.weibull(1.5, 60) * 100)
    second = surv.Weibull.fit(rng.weibull(2.0, 40) * 120)
    return {
        "a": first,
        "b": first,
        "c": second,
        "d": W([90, 1.4]),
        "e": second,
    }


@pytest.mark.parametrize("sampling", ["random", "sobol"])
def test_the_uncertainty_is_drawn_as_the_diagram_s(sampling):
    # The draws of the links' fitted models are a diagram's with the same
    # seed and sources (the same machinery): the same samples, to the
    # rounding of the two engines (#315).
    models = fitted_bridge()
    net, rbd = bridge(models), as_diagram(models)
    assert net._uncertainty_sources(None) == [
        (("a", "b"), "fit"),
        (("c", "e"), "fit"),
    ]
    t = np.array([10.0, 50.0, 120.0])
    ours = net.sf_uncertainty(t, n_draws=100, seed=3, sampling=sampling)
    theirs = rbd.sf_uncertainty(t, n_draws=100, seed=3, sampling=sampling)
    assert ours.n_draws == 100 and ours.samples.shape == (100, 3)
    np.testing.assert_allclose(ours.samples, theirs.samples, rtol=1e-13)
    np.testing.assert_allclose(ours.nominal, theirs.nominal, rtol=1e-13)
    given = {("a", "b"): "fit", "d": {"alpha": st.uniform(80, 20)}}
    ours = net.sf_uncertainty(
        50.0, given, n_draws=50, seed=1, sampling=sampling
    )
    theirs = rbd.sf_uncertainty(
        50.0, given, n_draws=50, seed=1, sampling=sampling
    )
    assert isinstance(ours.nominal, float) and ours.samples.shape == (50,)
    np.testing.assert_allclose(ours.samples, theirs.samples, rtol=1e-13)
    ours = net.mean_uncertainty(given, n_draws=10, seed=1, sampling=sampling)
    theirs = rbd.mean_uncertainty(given, n_draws=10, seed=1, sampling=sampling)
    np.testing.assert_allclose(ours.samples, theirs.samples, rtol=1e-9)
    assert ours.nominal == pytest.approx(net.mean(), rel=1e-12)
    # Times of any shape, a draw's row in their shape.
    shaped = net.sf_uncertainty(t.reshape(3, 1), given, n_draws=4, seed=1)
    assert shaped.samples.shape == (4, 3, 1) and shaped.nominal.shape == (3, 1)


def test_the_uncertainty_of_a_fixed_network_needs_no_time():
    # Every draw a fixed probability: the network's reliability with the
    # drawn probabilities, each sample one of the combinations.
    models = {n: F(0.1 + 0.05 * i) for i, n in enumerate("abcde")}
    models["b"] = models["a"]
    net = bridge(models, nodes={"x": F(0.2)})
    given = {"x": [F(0.2), F(0.4)], ("a", "b"): [F(0.1), F(0.3)]}
    ours = net.sf_uncertainty(None, given, n_draws=30, seed=2)
    assert ours.samples.shape == (30,) and ours.nominal == net.sf()
    possible = {
        round(bridge({**models, "a": F(p), "b": F(p)}, {"x": F(q)}).sf(), 12)
        for p in (0.1, 0.3)
        for q in (0.2, 0.4)
    }
    assert set(np.round(ours.samples, 12)) == possible
    with pytest.raises(ValueError, match="x is required"):
        bridge({**models, "a": W([100, 2])}).sf_uncertainty(
            None, {"a": [W([80, 2])]}
        )


def test_the_uncertainty_is_checked():
    models = fitted_bridge()
    net = bridge(models)
    with pytest.warns(UserWarning, match="drawn without 'b'"):
        net.sf_uncertainty(50.0, {"a": "fit"}, n_draws=3, seed=1)
    for given, message in [
        ({}, "Give the uncertain elements"),
        ({"z": "fit"}, "not a link or a failing node"),
        ({"a": "fit", ("a", "b"): "fit"}, "given an uncertainty twice"),
        ({("a", "c"): "fit"}, "must have the same model"),
        ({"d": "fit"}, "parameter covariance"),
    ]:
        with pytest.raises(ValueError, match=message):
            net.sf_uncertainty(50.0, given, n_draws=3, seed=1)
    with pytest.raises(ValueError, match="n_draws must be"):
        net.sf_uncertainty(50.0, {"a": "fit"}, n_draws=0)
    with pytest.raises(ValueError, match="No link's or node's model"):
        bridge({n: W([100, 2]) for n in "abcde"}).sf_uncertainty(50.0)
    with pytest.raises(ValueError, match="no lifetime"):
        bridge({n: F(0.1) for n in "abcde"}).mean_uncertainty({"a": [F(0.2)]})


def test_the_network_saves_and_loads():
    models = fitted_bridge()
    net = bridge(models, nodes={"x": E([0.002]), "t": 0.01})
    t = np.array([10.0, 50.0, 120.0])
    # A fit's covariance round-trips, so its uncertainty can be drawn (the
    # links of one fitted object are separate objects once loaded, so they
    # are given together).
    given = {("a", "b"): "fit", ("c", "e"): "fit"}
    for clone in (
        Network.from_dict(net.to_dict()),
        Network.from_json(net.to_json(indent=2)),
    ):
        assert clone.links == net.links
        assert list(clone.nodes) == list(net.nodes)
        assert (clone.source, clone.target) == (net.source, net.target)
        np.testing.assert_array_equal(clone.sf(t), net.sf(t))
        np.testing.assert_array_equal(
            clone.sf_uncertainty(t, given, n_draws=5, seed=1).samples,
            net.sf_uncertainty(t, given, n_draws=5, seed=1).samples,
        )
    saved = net.to_dict()
    assert saved["type"] == "Network" and saved["source"] == "s"
    assert [link["link"] for link in saved["links"]] == list("abcde")
    assert saved["links"][0]["nodes"] == ["s", "x"]
    assert [node["node"] for node in saved["nodes"]] == ["x", "t"]
    assert saved["nodes"][1]["model"]["kind"] == "surpyval"
    # Tuple names, which JSON turns into lists, come back as tuples.
    grid = built(lattice(3, 3), lambda name: E([0.01]), nodes={(1, 1): F(0.1)})
    clone = Network.from_json(grid.to_json())
    assert clone.links == grid.links and set(clone.nodes) == {(1, 1)}
    assert clone.source == (0, 0) and clone.target == (2, 2)
    assert clone.sf(10.0) == grid.sf(10.0)
    with pytest.raises(ValueError, match="Not a network"):
        Network.from_dict({"type": "FaultTree"})

    class Own:
        def sf(self, x):
            return np.exp(-np.asarray(x) / 100)

        def ff(self, x):
            return 1 - self.sf(x)

    with pytest.raises(NotImplementedError, match="Cannot serialise"):
        Network({"a": ("s", "t", Own())}, "s", "t").to_dict()


#: A call of each analysis the routes report.
CALLS = {
    **{
        name: (lambda name: lambda net: getattr(net, name)(10.0))(name)
        for name in (
            "sf",
            "ff",
            "birnbaum_importance",
            "improvement_potential",
            "risk_achievement_worth",
            "risk_reduction_worth",
            "criticality_importance",
            "fussell_vesely",
            "differential_importance",
            "joint_importance",
            "barlow_proschan_importance",
        )
    },
    "structural_importance": lambda net: net.structural_importance(),
    "cut_sets": lambda net: net.cut_sets(),
    "path_sets": lambda net: net.path_sets(),
    "mean": lambda net: net.mean(),
    "time_to_reliability": lambda net: net.time_to_reliability(0.5),
    "bx_life": lambda net: net.bx_life(10),
    "random": lambda net: net.random(10, seed=1),
    "sf_uncertainty": lambda net: net.sf_uncertainty(
        10.0, {"a": [W([90, 2])]}, n_draws=2, seed=1
    ),
    "mean_uncertainty": lambda net: net.mean_uncertainty(
        {"a": [W([90, 2])]}, n_draws=2, seed=1
    ),
}


def test_the_routes_say_how_each_analysis_is_worked_out(monkeypatch):
    net = bridge({n: W([100.0, 1.5]) for n in "abcde"}, nodes={"x": E([0.01])})
    report = net.analysis_routes()
    public = {
        name
        for name in dir(Network)
        if not name.startswith("_") and callable(getattr(Network, name))
    }
    assert set(report) == set(CALLS)
    assert public - set(report) == {
        "analysis_routes",
        "from_dict",
        "from_json",
        "to_dict",
        "to_json",
    }
    assert report["sf"].route == report["fussell_vesely"].route == "exact"
    assert report["joint_importance"].route == "exact"
    assert report["mean"].route == report["bx_life"].route == "numerical"
    assert report["barlow_proschan_importance"].route == "numerical"
    assert report["random"].route == "simulated"
    assert report["sf_uncertainty"].route == "simulated"
    assert report["mean_uncertainty"].route == "simulated"
    fixed = bridge({n: F(0.1) for n in "abcde"})
    report = fixed.analysis_routes()
    assert report["sf"].route == "exact"
    for name in ("mean", "mean_uncertainty", "barlow_proschan_importance"):
        assert report[name].route == "refused"
        assert "no lifetime" in report[name].reason
    assert report["bx_life"].route == "refused"
    with pytest.raises(ValueError, match=re.escape(report["bx_life"].reason)):
        fixed.bx_life(10)
    # Too large a diagram: every exact value refused, with the message
    # the methods raise, and the simulation (and the paths) left.
    monkeypatch.setattr(network_module, "MAX_STATES", 3)
    large = bridge({n: W([100.0, 1.5]) for n in "abcde"})
    report = large.analysis_routes()
    for name, route in report.items():
        if name in ("random", "path_sets"):
            assert route.route != "refused"
            CALLS[name](large)
            continue
        assert route.route == "refused", name
        assert "method='simulate'" in route.reason
        with pytest.raises(NotImplementedError) as error:
            CALLS[name](large)
        assert str(error.value) == route.reason


def test_the_new_methods_are_checked():
    net = bridge({n: W([100.0, 1.5]) for n in "abcde"})
    with pytest.raises(ValueError, match="kind must be"):
        net.criticality_importance(10.0, kind="other")
    with pytest.raises(ValueError, match="fv_type must be"):
        net.fussell_vesely(10.0, fv_type="x")
    with pytest.raises(ValueError, match="'exact' or 'rare_event'"):
        net.fussell_vesely(10.0, method="guess")
    with pytest.raises(ValueError, match="change must be"):
        net.differential_importance(10.0, change="sideways")
    with pytest.raises(ValueError, match="names"):
        net.differential_importance(10.0, groups={"g": ["z"]})
    with pytest.raises(ValueError, match="percentage"):
        net.bx_life(100)
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        net.time_to_reliability(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        net.barlow_proschan_importance(-1.0)
    with pytest.raises(TypeError, match="must be numbers"):
        net.bx_life("10")
    with pytest.raises(ValueError, match="no lifetime"):
        bridge({n: F(0.1) for n in "abcde"}).barlow_proschan_importance()
    with pytest.raises(ValueError, match="does not vary"):
        bridge({n: F(0.1) for n in "abcde"}).time_to_reliability(0.5)
