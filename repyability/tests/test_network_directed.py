"""A directed network's decision diagram: every state enumerated in exact
arithmetic on small graphs with cycles, for two terminals, several and
all of them; two terminals against the paths' method; a network with no
cycle against its reliability block diagram; meshes with more paths than
can be listed; and what an undirected network's diagram is, unchanged."""

import itertools
import math
import random
from fractions import Fraction

import numpy as np
import pytest
import surpyval as surv

from repyability import Network
from repyability import network as network_module
from repyability.tests.test_network_terminals import (
    GRAPHS,
    cyclic,
    enumerated,
    grid,
    joined,
    nodes_of,
)

E, W = surv.Exponential.from_params, surv.Weibull.from_params


def both_ways(links):
    """Each link, and one back."""
    out = {}
    for name, (u, v) in links.items():
        out[name] = (u, v)
        out[("back", name)] = (v, u)
    return out


def drawn_graphs():
    """Twenty-four directed graphs of up to 6 nodes and 10 links, drawn
    from a fixed seed: most have cycles, some links are doubled and some
    nodes lead nowhere."""
    rng = random.Random(839)
    out = {}
    while len(out) < 24:
        nodes = rng.randint(3, 6)
        links = {
            f"l{i}": tuple(rng.sample(range(nodes), 2))
            for i in range(rng.randint(nodes, 10))
        }
        if len(nodes_of(links)) >= 3:
            out[f"drawn {len(out)}"] = links
    return out


#: Directed graphs: one with cycles and a link back into the source, the
#: undirected list's as drawn (one way) and with a link each way, and
#: the drawn ones.
DIRECTED = {
    "cyclic": cyclic(),
    "ring": {f"r{i}": (i, (i + 1) % 5) for i in range(5)},
    "bridge": GRAPHS["bridge"],
    "bridge both ways": both_ways(GRAPHS["bridge"]),
    "grid 2 by 3 both ways": {
        name: ends
        for name, ends in both_ways(GRAPHS["grid 2 by 3"]).items()
        if name not in [("back", "h0,0"), ("back", "v0,2")]
    },
    "tree": GRAPHS["tree"],
    "K4 both ways": both_ways(GRAPHS["K4"]),
    "doubled links": GRAPHS["doubled links"],
    **drawn_graphs(),
}


def cases():
    """Each graph with two terminals, three and all, from two sources,
    without and with nodes that fail (a terminal among them)."""
    for label, links in DIRECTED.items():
        nodes = nodes_of(links)
        for source in (nodes[0], nodes[-1]):
            rest = [node for node in nodes if node != source]
            for terminals in ([rest[0]], rest[:2], rest[-2:], "all"):
                for failing in ([], [source, rest[-1]], [rest[0]]):
                    yield pytest.param(
                        links,
                        source,
                        terminals,
                        failing,
                        id=f"{label}, {source} to {terminals}, {failing}",
                    )


def network_of(links, source, terminals, failing, q):
    return Network(
        {name: (u, v, float(q[name])) for name, (u, v) in links.items()},
        source=source,
        terminals=terminals,
        nodes={node: float(q[node]) for node in failing},
        directed=True,
    )


def chances(links, failing):
    """Each element's probability of failing, a Fraction, all different."""
    names = list(links) + list(failing)
    return {
        name: Fraction(2 * i + 3, 4 * len(names) + 9)
        for i, name in enumerate(names)
    }


@pytest.mark.parametrize("links, source, terminals, failing", list(cases()))
def test_every_state_enumerated(links, source, terminals, failing):
    q = chances(links, failing)
    net = network_of(links, source, terminals, failing, q)
    assert net.source == source and net.terminals[0] == source
    exact = enumerated(links, list(net.terminals), q, directed=True)
    assert float(net.sf()) == pytest.approx(float(exact), rel=1e-13, abs=0)
    assert float(net.ff()) == pytest.approx(float(1 - exact), rel=1e-13, abs=0)
    if len(net.terminals) == 2:
        # Two terminals, as the paths give them.
        paths = Network(**net._init_args)
        try:
            network_module.METHOD = "paths"
            assert float(paths.sf()) == pytest.approx(
                float(net.sf()), abs=1e-15
            )
            assert float(paths.ff()) == pytest.approx(
                float(net.ff()), abs=1e-15
            )
        finally:
            network_module.METHOD = "bdd"


def minimal(sets):
    return {s for s in sets if not any(other < s for other in sets)}


@pytest.mark.parametrize(
    "label", ["cyclic", "bridge both ways", "drawn 3", "drawn 7", "drawn 12"]
)
@pytest.mark.parametrize("terminals", ["two", "three", "all"])
def test_the_path_and_cut_sets_are_the_enumerated_ones(label, terminals):
    links = DIRECTED[label]
    nodes = nodes_of(links)
    failing = [nodes[1]]
    chosen = {"two": [nodes[-1]], "three": nodes[-2:], "all": "all"}
    net = network_of(
        links, nodes[0], chosen[terminals], failing, chances(links, failing)
    )
    names = list(net.models)
    works, fails = set(), set()
    for up in itertools.product([False, True], repeat=len(names)):
        state = dict(zip(names, up))
        if joined(links, list(net.terminals), state, directed=True):
            works.add(frozenset(n for n in names if state[n]))
        else:
            fails.add(frozenset(n for n in names if not state[n]))
    if not works:
        assert net.sf() == 0.0 and net.path_sets() == set()
        return
    assert net.path_sets() == minimal(works)
    assert net.cut_sets() == minimal(fails)
    assert set(net.minimal_path_sets()) == minimal(works)
    assert set(net.minimal_cut_sets()) == minimal(fails)
    # Structural importance: the share of the others' states in which
    # each element is critical.
    importance = net.structural_importance()
    for name in names:
        critical = sum(
            1
            for state in works
            if name in state and state - {name} not in works
        )
        assert importance[name] == pytest.approx(
            critical / 2 ** (len(names) - 1), abs=1e-15
        )


@pytest.mark.parametrize("terminals", [[(2, 2)], [(2, 2), (0, 2)], "all"])
def test_links_each_way_are_the_undirected_network(terminals):
    # A link that works each way independently, with one chance: what the
    # source reaches is what the undirected network joins to it.
    one_way = grid(3, 3)
    q = {name: 0.05 + 0.01 * i for i, name in enumerate(one_way)}
    undirected = Network(
        {name: (u, v, q[name]) for name, (u, v) in one_way.items()},
        terminals=(
            "all" if terminals == "all" else [(0, 0)] + list(terminals)
        ),
    )
    directed = Network(
        {
            name: (u, v, q[name[1] if isinstance(name, tuple) else name])
            for name, (u, v) in both_ways(one_way).items()
        },
        source=(0, 0),
        terminals=terminals,
        directed=True,
    )
    assert directed.ff() == pytest.approx(undirected.ff(), rel=1e-13)


def test_a_mesh_of_more_paths_than_can_be_listed():
    # An 11-by-11 grid of links to the right and down has 184,756 paths
    # corner to corner; its diagram draws it.
    links = {
        name: (u, v, W([100.0 + i % 7, 1.5]))
        for i, (name, (u, v)) in enumerate(grid(11, 11).items())
    }
    net = Network(links, (0, 0), (10, 10), directed=True)
    assert math.comb(20, 10) > network_module.MAX_PATHS
    x = np.array([5.0, 20.0, 60.0])
    rbd = net.to_rbd()
    assert net.ff(x) == pytest.approx(rbd.ff(x), rel=1e-11)
    assert net.sf(x) == pytest.approx(rbd.sf(x), rel=1e-11)
    with pytest.raises(NotImplementedError, match="too many to list"):
        net.path_sets()
    report = net.analysis_routes()
    assert report["sf"].route == "exact"
    assert report["minimal_path_sets"].route == "refused"


def test_a_mesh_with_cycles_and_more_paths_than_can_be_listed():
    # A 6-by-6 grid with a link each way between neighbours: 1,262,816
    # simple paths corner to corner, and the undirected grid's values.
    one_way = grid(6, 6)
    cable = E([0.05])
    net = Network(
        {name: (u, v, cable) for name, (u, v) in both_ways(one_way).items()},
        (0, 0),
        (5, 5),
        directed=True,
    )
    undirected = Network(
        {name: (u, v, cable) for name, (u, v) in one_way.items()},
        (0, 0),
        (5, 5),
    )
    x = np.array([0.5, 1.0, 4.0])
    assert net.ff(x) == pytest.approx(undirected.ff(x), rel=1e-12)
    assert net.mean() == pytest.approx(undirected.mean(), rel=1e-9)
    with pytest.raises(NotImplementedError, match="too many to list"):
        net.path_sets()
    # No link into the source or out of the target matters.
    ours = net.birnbaum_importance(1.0)
    assert set(ours) == set(net.models)
    for name, (u, v) in net.links.items():
        if v == (0, 0) or u == (5, 5):
            assert ours[name] == 0
        else:
            assert ours[name] >= 0


def unmerged(words):
    """``_distinct`` merging nothing: every row its own."""
    count = len(words[0])
    return np.arange(count), np.arange(count, dtype=np.int64)


def same_plan(one, other):
    return (
        one.names == other.names
        and one.root == other.root
        and all(
            np.array_equal(getattr(one, field), getattr(other, field))
            for field in ("pivots", "bounds", "active", "inactive")
        )
    )


def test_the_diagram_does_not_depend_on_which_states_merge(monkeypatch):
    # The reduced diagram is the one of the links' order, whichever
    # states were found equal on the way: an undirected network's (its
    # states in one word, and in several) and a directed one's.
    def plans():
        narrow = Network(
            {n: (u, v, 0.1) for n, (u, v) in grid(3, 4).items()},
            (0, 0),
            (2, 3),
        )
        wide = Network(
            {
                (a, b): (a, b, 0.1)
                for a, b in itertools.combinations(range(9), 2)
                if (a + b) % 3
            },
            terminals="all",
        )
        one_way = Network(
            {n: (u, v, 0.1) for n, (u, v) in both_ways(grid(3, 3)).items()},
            source=(0, 0),
            terminals=[(2, 2), (0, 2)],
            directed=True,
        )
        return [net._decomposition() for net in (narrow, wide, one_way)]

    merged = plans()
    monkeypatch.setattr(network_module, "_distinct", unmerged)
    monkeypatch.setattr(
        network_module,
        "_unique_rows",
        lambda rows: (rows, np.arange(len(rows), dtype=np.int64)),
    )
    for one, other in zip(merged, plans()):
        assert len(one.active) > 0
        assert same_plan(one, other)


def test_states_in_several_words_merge(monkeypatch):
    # A frontier of 14 nodes takes more than one 64-bit word a state:
    # equal states merge all the same, so the diagram stays small.
    links = {}
    for i in range(14):
        links[("in", i)] = ("s", i, 0.1)
        links[("out", i)] = (i, "t", 0.1)
    net = Network(links, "s", "t")
    monkeypatch.setattr(network_module, "MAX_STATES", 200_000)
    assert net.sf() == pytest.approx(1 - (1 - 0.81) ** 14, rel=1e-13)


def test_the_simulated_lifetime_reaches_every_terminal():
    links = {
        name: (u, v, W([100.0, 2.0])) for name, (u, v) in cyclic().items()
    }
    lives = dict(zip("abcdefghi", [5.0, 4.0, 9.0, 9.0, 1.0, 3.0, 9.0, 2.0, 9]))
    one = {name: np.array([life]) for name, life in lives.items()}
    # t is reached until 3 (s-a-1-b-2-c-3-f-t), 2 until 4 (s-a-1-b-2) and
    # 3 until 4 (through 2): both t and 2 until 3, and 2 and 3 until 4.
    net = Network(links, source="s", terminals=["t", 2], directed=True)
    assert net._lifetimes(one, 1).tolist() == [3.0]
    net = Network(links, source="s", terminals=[3, 2], directed=True)
    assert net._lifetimes(one, 1).tolist() == [4.0]
    net = Network(links, source="s", terminals="all", directed=True)
    assert net._lifetimes(one, 1).tolist() == [3.0]
    # From t: 2 by link i, and 3 through 2, both until 9.
    net = Network(links, source="t", terminals=[2, 3], directed=True)
    assert net._lifetimes(one, 1).tolist() == [9.0]


def test_what_a_directed_network_is_asked():
    links = {
        name: (u, v, 0.1 + 0.01 * i)
        for i, (name, (u, v)) in enumerate(cyclic().items())
    }
    # A list is reached from its first node; a source may be named, in
    # the list or out of it.
    listed = Network(links, terminals=["s", "t", 2], directed=True)
    named = Network(links, source="s", terminals=["t", 2], directed=True)
    within = Network(links, source="s", terminals=[2, "s", "t"], directed=True)
    assert listed.terminals == named.terminals == ("s", "t", 2)
    assert within.terminals == ("s", 2, "t")
    assert listed.sf() == named.sf() == within.sf()
    assert (named.source, named.target) == ("s", None)
    assert repr(named) == (
        "Network(9 directed links, 0 failing nodes, 3 terminals from 's')"
    )
    every = Network(links, source="t", terminals="all", directed=True)
    assert every.terminals[0] == "t" and len(every.terminals) == 5
    assert repr(every) == (
        "Network(9 directed links, 0 failing nodes, all terminals from 't')"
    )
    # Reached from another node, another network.
    assert Network(
        links, source=2, terminals=["t", "s"], directed=True
    ).sf() != (named.sf())
    # One terminal with a source is the source and target.
    assert Network(links, source="s", terminals=["t"], directed=True).sf() == (
        Network(links, "s", "t", directed=True).sf()
    )
    with pytest.raises(ValueError, match="say which with source="):
        Network(links, terminals="all", directed=True)
    with pytest.raises(ValueError, match="say which with source="):
        Network(links, terminals={"s", "t", 2}, directed=True)
    with pytest.raises(ValueError, match="'nowhere' is in no link"):
        Network(links, source="nowhere", terminals="all", directed=True)
    with pytest.raises(ValueError, match="names a node twice"):
        Network(links, source="s", terminals=["t", "t"], directed=True)
    with pytest.raises(ValueError, match="at least two nodes"):
        Network(links, source="s", terminals=["s"], directed=True)
    with pytest.raises(ValueError, match="not both"):
        Network(links, "s", "t", terminals=["s", "t", 2], directed=True)
    # An undirected network's terminals have no source.
    with pytest.raises(ValueError, match="not both"):
        Network(links, source="s", terminals=["t", 2])
    # A diagram has an input and an output; a capacity two terminals.
    with pytest.raises(ValueError, match="two terminals") as error:
        named.to_rbd()
    assert "own methods" in str(error.value)
    with pytest.raises(NotImplementedError, match="source= and target="):
        Network(
            links,
            source="s",
            terminals=["t", 2],
            directed=True,
            capacity={"a": 1.0},
        )
    # A terminal that nothing leads to is never reached.
    del links["g"]
    apart = Network(links, source="t", terminals=["s", 2], directed=True)
    assert apart.sf() == 0 and apart.ff() == 1
    assert apart.path_sets() == set()


def test_a_diagram_that_follows_too_many_nodes_says_so():
    # 70 nodes fed from the source, none leading on, all to reach: none
    # can leave before the last link.
    links = {i: ("s", i, 0.01) for i in range(70)}
    links.update({("on", i): (i, (i + 1) % 70, 0.5) for i in range(70)})
    net = Network(links, source="s", terminals="all", directed=True)
    with pytest.raises(NotImplementedError, match="more than 64 nodes") as e:
        net.sf()
    assert "method='simulate'" in str(e.value)
    report = net.analysis_routes()
    assert report["sf"].route == "refused"
    assert report["sf"].reason == str(e.value)
    assert report["random"].route == "simulated"
    assert 0 < net.sf(method="simulate", mc_samples=2000, seed=1) < 1
    # A chain of 200 is followed two nodes at a time.
    chain = Network(
        {i: (i, i + 1, 0.01) for i in range(200)},
        source=0,
        terminals="all",
        directed=True,
    )
    assert chain.sf() == pytest.approx(0.99**200, rel=1e-13)
    # Repaired elements are not simulated: the refusal does not say to.
    spec = {"reliability": E([0.01]), "repairability": E([1.0])}
    repaired = Network(
        {name: (u, v, spec) for name, (u, v, _) in links.items()},
        source="s",
        terminals="all",
        directed=True,
    )
    with pytest.raises(NotImplementedError, match="not simulated yet") as e:
        repaired.point_availability(1.0)
    assert "simulate it" not in str(e.value).lower().replace(
        "not simulated yet", ""
    )
