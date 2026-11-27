"""Networks of more than two terminals, and directed ones: K-terminal and
all-terminal reliability against every state enumerated in exact
arithmetic, closed forms, the two-terminal results kept, the simulation,
and a directed network against its reliability block diagram."""

import hashlib
import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv

from repyability import Network, NonRepairableRBD
from repyability import network as network_module

E, W = surv.Exponential.from_params, surv.Weibull.from_params


def grid(rows, columns):
    """A grid's links' ends, by name."""
    links = {}
    for i in range(rows):
        for j in range(columns):
            if j + 1 < columns:
                links[f"h{i},{j}"] = ((i, j), (i, j + 1))
            if i + 1 < rows:
                links[f"v{i},{j}"] = ((i, j), (i + 1, j))
    return links


def named(edges):
    return {f"l{i}": edge for i, edge in enumerate(edges)}


#: The fixed list of graphs, of up to 12 links each.
GRAPHS = {
    "tree": named([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]),
    "star": named([(0, i) for i in range(1, 6)]),
    "cycle of 5": named([(i, (i + 1) % 5) for i in range(5)]),
    "cycle of 8": named([(i, (i + 1) % 8) for i in range(8)]),
    "bridge": named(
        [("s", "x"), ("s", "y"), ("x", "y"), ("x", "t"), ("y", "t")]
    ),
    "grid 2 by 3": grid(2, 3),
    "grid 3 by 3": grid(3, 3),
    "grid 2 by 5": {
        name: ends for name, ends in grid(2, 5).items() if name != "h1,3"
    },
    "K4": named(itertools.combinations(range(4), 2)),
    "K5": named(itertools.combinations(range(5), 2)),
    "doubled links": named([(0, 1), (0, 1), (1, 2), (2, 0), (2, 3)]),
    "two parts": named([(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)]),
}


def nodes_of(links):
    return list(
        dict.fromkeys(node for ends in links.values() for node in ends)
    )


def states(q):
    """Every state of the elements (whether each works), with its
    probability, the probabilities of failing ``q`` being Fractions."""
    names = list(q)
    for up in itertools.product([False, True], repeat=len(names)):
        chance = Fraction(1)
        for name, works in zip(names, up):
            chance *= 1 - q[name] if works else q[name]
        yield dict(zip(names, up)), chance


def joined(links, terminals, up, directed=False):
    """Whether the terminals are joined (in a directed network, the second
    reached from the first) by the links and nodes that work."""
    if not all(up.get(node, True) for node in terminals):
        return False
    beside = {}
    for name, (u, v) in links.items():
        if up[name] and up.get(u, True) and up.get(v, True):
            beside.setdefault(u, []).append(v)
            if not directed:
                beside.setdefault(v, []).append(u)
    seen, stack = {terminals[0]}, [terminals[0]]
    while stack:
        for other in beside.get(stack.pop(), ()):
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return all(node in seen for node in terminals)


def enumerated(links, terminals, q, directed=False):
    """The probability that the terminals are joined, exactly."""
    return sum(
        (
            chance
            for up, chance in states(q)
            if joined(links, terminals, up, directed)
        ),
        Fraction(0),
    )


def cases():
    """Each graph with two, three and all terminals, without and with
    failing nodes (the first terminal among them), the terminals, the
    failing nodes and the probabilities chosen by a fixed generator."""
    rng = np.random.default_rng(815)
    for graph, links in GRAPHS.items():
        nodes = nodes_of(links)
        for k, with_nodes in itertools.product((2, 3, "all"), (False, True)):
            if k == "all":
                terminals = nodes
            else:
                terminals = [
                    nodes[i] for i in rng.choice(len(nodes), k, replace=False)
                ]
            failing = []
            if with_nodes:
                room = min(3, 14 - len(links))
                others = [n for n in nodes if n != terminals[0]]
                failing = [terminals[0]] + [
                    others[i]
                    for i in rng.choice(len(others), room - 1, replace=False)
                ]
            q = {
                name: Fraction(int(rng.integers(1, 10)), 10)
                for name in list(links) + failing
            }
            label = f"{graph}, {k} terminals" + (
                ", failing nodes" if with_nodes else ""
            )
            yield pytest.param(links, k, terminals, failing, q, id=label)


def network(links, k, terminals, failing, q, **options):
    return Network(
        {name: (u, v, float(q[name])) for name, (u, v) in links.items()},
        nodes={node: float(q[node]) for node in failing} or None,
        terminals="all" if k == "all" else terminals,
        **options,
    )


@pytest.mark.parametrize("links, k, terminals, failing, q", list(cases()))
def test_every_state_enumerated(links, k, terminals, failing, q):
    net = network(links, k, terminals, failing, q)
    exact = enumerated(links, terminals, q)
    assert float(net.sf()) == pytest.approx(float(exact), rel=1e-12, abs=1e-15)
    assert float(net.ff()) == pytest.approx(
        float(1 - exact), rel=1e-12, abs=1e-15
    )


@pytest.mark.parametrize(
    "graph, terminals, failing",
    [
        ("bridge", ["s", "t", "x"], ["x", "y"]),
        ("grid 2 by 3", "all", [(0, 1)]),
        ("K4", [0, 1, 2], [3]),
    ],
)
def test_the_importance_and_the_sets_enumerated(graph, terminals, failing):
    links = GRAPHS[graph]
    k = "all" if terminals == "all" else len(terminals)
    held = nodes_of(links) if terminals == "all" else terminals
    q = {
        name: Fraction(i + 2, 20)
        for i, name in enumerate(list(links) + failing)
    }
    net = network(links, k, held, failing, q)
    # Birnbaum: the reliability with the element working, less failed.
    birnbaum = net.birnbaum_importance()
    for name in q:
        works = enumerated(links, held, {**q, name: Fraction(0)})
        fails = enumerated(links, held, {**q, name: Fraction(1)})
        assert float(birnbaum[name]) == pytest.approx(
            float(works - fails), rel=1e-12, abs=1e-15
        )
    # The minimal path and cut sets: the smallest sets that join the
    # terminals working alone, and part them failed alone.
    names = list(q)
    paths, cuts = [], []
    for size in range(len(names) + 1):
        for chosen in map(frozenset, itertools.combinations(names, size)):
            up = {name: name in chosen for name in names}
            if joined(links, held, up) and not any(p <= chosen for p in paths):
                paths.append(chosen)
            down = {name: name not in chosen for name in names}
            if not joined(links, held, down) and not any(
                c <= chosen for c in cuts
            ):
                cuts.append(chosen)
    assert net.path_sets() == set(paths)
    assert net.cut_sets() == set(cuts)
    assert net.minimal_path_sets()[0] == min(
        paths, key=lambda s: (len(s), sorted(map(str, s)))
    )
    # Fussell-Vesely, exactly: a minimal cut set through the element has
    # failed, over the unreliability.
    fussell_vesely = net.fussell_vesely()
    unreliability = 1 - enumerated(links, held, q)
    for name in names:
        through = [cut for cut in cuts if name in cut]
        failed = sum(
            (
                chance
                for up, chance in states(q)
                if any(not any(up[e] for e in cut) for cut in through)
            ),
            Fraction(0),
        )
        assert float(fussell_vesely[name]) == pytest.approx(
            float(failed / unreliability), rel=1e-11, abs=1e-15
        )


def test_a_tree_is_the_product():
    q = {name: Fraction(i + 1, 10) for i, name in enumerate(GRAPHS["tree"])}
    net = network(GRAPHS["tree"], "all", None, [], q)
    assert float(net.sf()) == pytest.approx(
        float(math.prod(1 - x for x in q.values())), rel=1e-14
    )
    # Three leaves: the links of the tree that joins them.
    net = network(GRAPHS["tree"], 3, [0, 2, 4], [], q)
    on_tree = [(0, 1), (1, 2), (1, 3), (3, 4)]
    assert float(net.sf()) == pytest.approx(
        float(
            math.prod(
                1 - q[name]
                for name, ends in GRAPHS["tree"].items()
                if ends in on_tree
            )
        ),
        rel=1e-14,
    )
    assert net.path_sets() == {
        frozenset(n for n, ends in GRAPHS["tree"].items() if ends in on_tree)
    }


@pytest.mark.parametrize("n", [3, 5, 8, 30, 200])
@pytest.mark.parametrize("p", [0.5, 0.9, 0.999])
def test_a_cycle_all_terminal(n, p):
    # Every link works, or all but one.
    links = {i: (i, (i + 1) % n, 1 - p) for i in range(n)}
    net = Network(links, terminals="all")
    assert float(net.sf()) == pytest.approx(
        p**n + n * p ** (n - 1) * (1 - p), rel=1e-12
    )


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_a_complete_graph_all_terminal(n):
    # Gilbert's recursion: the node 1 is in a part of k nodes, joined
    # among themselves and to none of the other n - k.
    q = Fraction(3, 10)
    joined_up = {1: Fraction(1)}
    for m in range(2, n + 1):
        joined_up[m] = 1 - sum(
            math.comb(m - 1, k - 1) * joined_up[k] * q ** (k * (m - k))
            for k in range(1, m)
        )
    links = {
        (a, b): (a, b, float(q))
        for a, b in itertools.combinations(range(n), 2)
    }
    net = Network(links, terminals="all")
    assert float(net.sf()) == pytest.approx(float(joined_up[n]), rel=1e-12)
    assert float(net.ff()) == pytest.approx(float(1 - joined_up[n]), rel=1e-12)
    # Cayley: n ** (n - 2) spanning trees.
    assert len(net.path_sets()) == n ** (n - 2)


# ----------------------------------------------------------------------
# Two terminals: as before
# ----------------------------------------------------------------------


def plan_digest(plan):
    digest = hashlib.sha256()
    for field in ("pivots", "bounds", "active", "inactive"):
        digest.update(
            np.ascontiguousarray(getattr(plan, field), dtype="<i8").tobytes()
        )
    return digest.hexdigest()


def test_two_terminals_have_the_plan_they_had():
    # The plan of a grid of 36 nodes, corner to corner, as recorded before
    # a network took more than two terminals.
    links = {name: (u, v, 0.1) for name, (u, v) in grid(6, 6).items()}
    plan = Network(links, (0, 0), (5, 5))._decomposition()
    assert len(plan.active) == 4969 and plan.root == 4970
    assert plan_digest(plan) == (
        "fdbf0eee5f0a149b684fb5be0311fdb8be3960c3b3bf1b2b92844311bcf73652"
    )


@pytest.mark.parametrize("graph", sorted(GRAPHS))
def test_two_terminals_given_either_way_are_one_network(graph):
    nodes = nodes_of(GRAPHS[graph])
    links = {
        name: (u, v, W([100.0 + 7 * i, 1.5]))
        for i, (name, (u, v)) in enumerate(GRAPHS[graph].items())
    }
    failing = {nodes[1]: E([0.002])}
    s, t = nodes[0], nodes[-1]
    one = Network(links, s, t, nodes=failing)
    other = Network(links, nodes=failing, terminals=[s, t])
    assert (other.source, other.target) == (s, t) == one.terminals
    a, b = one._decomposition(), other._decomposition()
    assert a.names == b.names and a.root == b.root
    assert plan_digest(a) == plan_digest(b)
    x = np.array([10.0, 50.0, 200.0])
    assert np.array_equal(one.sf(x), other.sf(x))
    assert np.array_equal(one.random(50, seed=4), other.random(50, seed=4))
    assert one.path_sets() == other.path_sets()


def test_two_terminals_draw_the_lifetimes_they_drew():
    # Recorded before a network took more than two terminals.
    links = {
        name: (u, v, W([100 + 3 * i, 1.5]))
        for i, (name, (u, v)) in enumerate(grid(3, 3).items())
    }
    net = Network(links, (0, 0), (2, 2), nodes={(1, 1): W([150, 2.0])})
    assert net.random(5, seed=3) == pytest.approx(
        [52.01724041, 51.54707919, 51.14873464, 60.11940294, 80.42156784],
        rel=1e-9,
    )


def test_a_grid_stays_within_twice_two_terminals():
    # The diagram's size (its steps) for three corners, four, and every
    # node, against corner to corner.
    links = {name: (u, v, 0.1) for name, (u, v) in grid(6, 6).items()}
    two = len(Network(links, (0, 0), (5, 5))._decomposition().active)
    corners = [(0, 0), (5, 5), (0, 5), (5, 0)]
    for terminals in (corners[:3], corners, "all"):
        steps = len(
            Network(links, terminals=terminals)._decomposition().active
        )
        assert steps <= 2 * two, terminals
    # Every node a terminal: fewer states than two terminals have.
    assert steps < two


def test_a_grid_of_100_nodes_all_terminal(monkeypatch):
    # Within the states two terminals take (1.9 million): 0.8 million.
    monkeypatch.setattr(network_module, "MAX_STATES", 1_000_000)
    q = 1e-3
    links = {name: (u, v, q) for name, (u, v) in grid(10, 10).items()}
    net = Network(links, terminals="all")
    # The cuts of two and three links: each corner is cut off by its two,
    # each other node of the rim (32) by its three, and each corner with
    # a neighbour (8) by their three. The cuts of four are some hundreds,
    # so 1e-4 of these.
    assert float(net.ff()) == pytest.approx(4 * q**2 + 40 * q**3, rel=1e-3)
    assert float(net.ff()) > 4 * q**2 - 6 * q**4


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------


def lifetime_network(**options):
    links = {
        name: (u, v, W([100 + 3 * i, 1.5]))
        for i, (name, (u, v)) in enumerate(grid(3, 3).items())
    }
    return Network(links, nodes={(1, 1): W([150, 2.0])}, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"terminals": [(0, 0), (2, 2), (0, 2)]},
        {"terminals": "all"},
        {"source": (0, 0), "target": (2, 2), "directed": True},
        {
            "source": (0, 0),
            "terminals": [(2, 2), (0, 2)],
            "directed": True,
        },
        {"source": (0, 0), "terminals": "all", "directed": True},
    ],
    ids=["three", "all", "directed", "directed three", "directed all"],
)
def test_the_simulation_agrees_with_the_exact_values(options):
    net = lifetime_network(**options)
    n = 20_000
    lives = net.random(n, seed=11)
    for x in (20.0, 40.0, 80.0):
        exact = float(net.sf(x))
        standard_error = math.sqrt(exact * (1 - exact) / n)
        assert abs((lives > x).mean() - exact) < 4 * standard_error
    assert abs(lives.mean() - net.mean()) < 4 * lives.std() / math.sqrt(n)
    assert net.sf(40.0, method="simulate", mc_samples=n, seed=11) == (
        pytest.approx(float(net.sf(40.0)), abs=0.02)
    )


def test_a_lifetime_is_when_the_terminals_part():
    # A star: all its links are needed, so the first to fail ends it; with
    # two leaves as terminals, the first of their two links.
    links = {f"l{i}": (0, i, W([100.0, 2.0])) for i in range(1, 5)}
    lives = {f"l{i}": np.array([float(i), 9.0 - i]) for i in range(1, 5)}
    every = Network(links, terminals="all")
    assert every._lifetimes(lives, 2).tolist() == [1.0, 5.0]
    three = Network(links, terminals=[2, 3, 4])
    assert three._lifetimes(lives, 2).tolist() == [2.0, 5.0]
    # A failed terminal parts them, though the links still work.
    failing = Network(links, terminals=[2, 3, 4], nodes={3: W([50.0, 2.0])})
    lives[3] = np.array([4.0, 4.0])
    assert failing._lifetimes(lives, 2).tolist() == [2.0, 4.0]


# ----------------------------------------------------------------------
# Every method, saving and levers
# ----------------------------------------------------------------------


@pytest.mark.parametrize("terminals", [[(0, 0), (2, 2), (0, 2)], "all"])
def test_the_measures_follow_the_terminals(terminals):
    net = lifetime_network(terminals=terminals)
    x = np.array([20.0, 60.0])
    sf = net.sf(x)
    assert np.all(np.diff(sf) < 0) and np.all((0 < sf) & (sf < 1))
    assert net.Hf(x) == pytest.approx(-np.log(sf), rel=1e-12)
    assert net.hf(x) == pytest.approx(net.df(x) / sf, rel=1e-9)
    assert float(net.sf(net.bx_life(10))) == pytest.approx(0.9, rel=1e-9)
    table = net.importances(40.0)
    assert not table.refused
    assert set(table.measures["birnbaum_importance"]) == set(net.models)
    shares = net.barlow_proschan_importance()
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-6)
    # More terminals to keep joined: no more reliable.
    two = lifetime_network(source=(0, 0), target=(2, 2))
    assert np.all(sf <= two.sf(x))
    comparison = net.compare(two)
    assert comparison.estimate == pytest.approx(net.mean() - two.mean())
    levers = net.levers()
    assert len(levers) == 2 * len(net.models)
    longer = net.with_levers({levers[0]: 2 * levers[0].value})
    assert longer.terminals == net.terminals
    assert longer._all_terminals == net._all_terminals
    assert float(longer.sf(40.0)) > float(net.sf(40.0))
    sensitivity = net.parameter_sensitivity(40.0)
    assert sum(map(len, sensitivity.values())) == len(levers)
    band = net.bx_life_uncertainty(
        10, {"h0,0": {"alpha": st.uniform(90.0, 20.0)}}, n_draws=20, seed=2
    )
    assert band.nominal == net.bx_life(10)
    assert band.samples.min() < band.nominal < band.samples.max()


@pytest.mark.parametrize(
    "options",
    [
        {"terminals": [(0, 0), (2, 2), (0, 2)]},
        {"terminals": "all"},
        {"source": (0, 0), "target": (2, 2)},
        {"source": (0, 0), "target": (2, 2), "directed": True},
        {
            "source": (1, 1),
            "terminals": [(2, 2), (1, 2)],
            "directed": True,
        },
        {"source": (0, 0), "terminals": "all", "directed": True},
    ],
    ids=["three", "all", "two", "directed", "directed three", "directed all"],
)
def test_a_network_is_saved_with_its_terminals(options):
    net = lifetime_network(**options)
    saved = json.loads(net.to_json())
    back = Network.from_dict(saved)
    assert back.terminals == net.terminals
    assert (back.source, back.target) == (net.source, net.target)
    assert back.directed == net.directed
    assert back._all_terminals == net._all_terminals
    assert repr(back) == repr(net)
    assert back.sf(40.0) == net.sf(40.0)
    if len(net.terminals) == 2:
        # A save from before: the source and the target only.
        del saved["terminals"], saved["directed"]
        old = Network.from_dict(saved)
        assert old.terminals == net.terminals and not old.directed
        assert old.sf(40.0) == Network(
            net._init_args["links"], *net.terminals, nodes=net.nodes
        ).sf(40.0)


def test_the_reprs():
    assert repr(lifetime_network(terminals="all")) == (
        "Network(12 links, 1 failing nodes, all terminals)"
    )
    assert repr(lifetime_network(terminals=[(0, 0), (2, 2), (0, 2)])) == (
        "Network(12 links, 1 failing nodes, 3 terminals)"
    )
    assert repr(lifetime_network(source=(0, 0), target=(2, 2))) == (
        "Network(12 links, 1 failing nodes, (0, 0) to (2, 2))"
    )


def test_the_terminals_are_checked():
    links = {"a": (0, 1, 0.1), "b": (1, 2, 0.1)}
    with pytest.raises(ValueError, match="source= and target=, or"):
        Network(links)
    with pytest.raises(ValueError, match="not both"):
        Network(links, 0, 2, terminals=[0, 1, 2])
    with pytest.raises(ValueError, match="at least two"):
        Network(links, terminals=[0])
    with pytest.raises(ValueError, match="twice"):
        Network(links, terminals=[0, 1, 0])
    with pytest.raises(ValueError, match="'every'"):
        Network(links, terminals="every")
    with pytest.raises(ValueError, match="Terminal 5 is in no link"):
        Network(links, terminals=[0, 1, 5])
    with pytest.raises(TypeError, match="directed must be True or False"):
        Network(links, 0, 2, directed="yes")
    # Two nodes: all of them are a source and a target.
    pair = Network({"a": (0, 1, 0.1)}, terminals="all")
    assert (pair.source, pair.target) == (0, 1) and pair.sf() == 0.9


def test_what_more_than_two_terminals_refuse(monkeypatch):
    links = {
        (a, b): (a, b, 0.1) for a, b in itertools.combinations(range(6), 2)
    }
    net = Network(links, terminals="all")
    assert len(net.path_sets()) == 6**4
    # Too many trees to list: the reliability does not need them.
    monkeypatch.setattr(network_module, "MAX_PATHS", 1000)
    with pytest.raises(NotImplementedError, match="passes 1,000 sets") as e:
        net.path_sets()
    report = net.analysis_routes()
    assert report["minimal_path_sets"].route == "refused"
    assert report["minimal_path_sets"].reason == str(e.value)
    assert report["sf"].route == "exact"
    # The paths' method is two terminals'.
    monkeypatch.setattr(network_module, "METHOD", "paths")
    fresh = Network(links, terminals=[0, 1, 2])
    with pytest.raises(NotImplementedError, match="METHOD = 'bdd'"):
        fresh.sf()
    with pytest.raises(NotImplementedError, match="METHOD = 'bdd'"):
        Network(links, terminals=[0, 1, 2], directed=True).sf()


# ----------------------------------------------------------------------
# Directed networks
# ----------------------------------------------------------------------


@pytest.mark.parametrize("graph", ["bridge", "grid 2 by 3", "grid 3 by 3"])
def test_a_directed_network_is_its_diagram(graph):
    nodes = nodes_of(GRAPHS[graph])
    links = {
        name: (u, v, W([100.0 + 7 * i, 1.5]))
        for i, (name, (u, v)) in enumerate(GRAPHS[graph].items())
    }
    failing = {nodes[1]: E([0.002]), nodes[-1]: E([0.001])}
    net = Network(links, nodes[0], nodes[-1], nodes=failing, directed=True)
    rbd = net.to_rbd()
    assert isinstance(rbd, NonRepairableRBD)
    # Its nodes: the links and failing nodes, and the others as junctions.
    assert set(rbd.nodes) == set(net.models) | set(nodes)
    x = np.array([10.0, 50.0, 200.0])
    assert net.sf(x) == pytest.approx(rbd.sf(x), rel=1e-12)
    assert net.ff(x) == pytest.approx(rbd.ff(x), rel=1e-12)
    assert net.mean() == pytest.approx(rbd.mean(), rel=1e-9)
    assert net.path_sets() == set(rbd.minimal_path_sets())
    assert net.cut_sets() == set(rbd.minimal_cut_sets())
    ours, theirs = net.birnbaum_importance(x), rbd.birnbaum_importance(x)
    for name in net.models:
        assert ours[name] == pytest.approx(theirs[name], rel=1e-10, abs=1e-15)
    ours, theirs = net.fussell_vesely(50.0), rbd.fussell_vesely(50.0)
    for name in net.models:
        assert ours[name] == pytest.approx(theirs[name], rel=1e-10, abs=1e-15)
    # One way only: no more reliable than the links working both ways.
    both = Network(links, nodes[0], nodes[-1], nodes=failing)
    assert np.all(net.sf(x) <= both.sf(x))


def cyclic():
    """A directed network with cycles: a ring one way round with chords,
    and a link back into the source."""
    return {
        "a": ("s", 1),
        "b": (1, 2),
        "c": (2, 3),
        "d": (3, 1),
        "e": (2, "t"),
        "f": (3, "t"),
        "g": (1, "s"),
        "h": ("s", 3),
        "i": ("t", 2),
    }


def test_a_directed_network_with_cycles_enumerated():
    links = cyclic()
    failing = [2]
    q = {
        name: Fraction(i + 1, 12)
        for i, name in enumerate(list(links) + failing)
    }
    net = Network(
        {name: (u, v, float(q[name])) for name, (u, v) in links.items()},
        "s",
        "t",
        nodes={2: float(q[2])},
        directed=True,
    )
    exact = enumerated(links, ["s", "t"], q, directed=True)
    assert float(net.sf()) == pytest.approx(float(exact), rel=1e-12)
    assert float(net.ff()) == pytest.approx(float(1 - exact), rel=1e-12)
    # The other way: another network.
    back = Network(
        {name: (u, v, float(q[name])) for name, (u, v) in links.items()},
        "t",
        "s",
        nodes={2: float(q[2])},
        directed=True,
    )
    assert float(back.sf()) == pytest.approx(
        float(enumerated(links, ["t", "s"], q, directed=True)), rel=1e-12
    )
    # A diagram has no cycle.
    with pytest.raises(NotImplementedError, match="have a cycle") as error:
        net.to_rbd()
    assert "own methods" in str(error.value)


def test_a_directed_lifetime_follows_the_links():
    links = {
        name: (u, v, W([100.0, 2.0])) for name, (u, v) in cyclic().items()
    }
    net = Network(links, "s", "t", directed=True)
    lives = dict(zip("abcdefghi", [5.0, 4.0, 9.0, 9.0, 1.0, 3.0, 9.0, 2.0, 9]))
    # s-a-1-b-2-c-3-f-t lasts min(5, 4, 9, 3) = 3; s-h-3-f-t lasts 2 and
    # s-a-1-b-2-e-t lasts 1.
    one = {name: np.array([life]) for name, life in lives.items()}
    assert net._lifetimes(one, 1).tolist() == [3.0]


def test_what_to_rbd_draws_and_refuses():
    # Links off every way from the source to the target are left out; a
    # node with a link's name is drawn under another.
    net = Network(
        {
            "a": ("s", "x", 0.1),
            "x": ("x", "t", 0.1),
            "back": ("t", "s", 0.1),
            "dead end": ("x", "y", 0.1),
            "input": ("s", "t", 0.2),
        },
        "s",
        "t",
        nodes={"t": 0.05},
        directed=True,
    )
    rbd = net.to_rbd()
    assert set(rbd.nodes) == {"a", "input", "t", "x", "s", ("node", "x")}
    assert rbd.input_node == "input (network)" and rbd.output_node == "output"
    assert rbd.sf() == pytest.approx(net.sf(), rel=1e-14)
    assert net.sf() == pytest.approx(0.95 * (1 - 0.2 * (1 - 0.9 * 0.9)))
    undirected = Network({"a": ("s", "t", 0.1)}, "s", "t")
    with pytest.raises(ValueError, match="directed=True"):
        undirected.to_rbd()
    apart = Network(
        {"a": ("x", "s", 0.1), "b": ("x", "t", 0.1)}, "s", "t", directed=True
    )
    assert apart.sf() == 0.0 and apart.path_sets() == set()
    with pytest.raises(ValueError, match="No way leads"):
        apart.to_rbd()


def test_a_directed_network_of_too_many_paths_is_worked_out(monkeypatch):
    # Its values come from the decision diagram, which lists no path.
    monkeypatch.setattr(network_module, "MAX_PATHS", 3)
    links = {name: (u, v, 0.1) for name, (u, v) in grid(3, 3).items()}
    net = Network(links, (0, 0), (2, 2), directed=True)
    exact = float(
        enumerated(
            grid(3, 3),
            [(0, 0), (2, 2)],
            {name: Fraction(1, 10) for name in links},
            directed=True,
        )
    )
    assert net.sf() == pytest.approx(exact, rel=1e-14)
    assert net.to_rbd().sf() == pytest.approx(exact, rel=1e-12)
    with pytest.raises(NotImplementedError, match="too many to list") as error:
        net.path_sets()
    report = net.analysis_routes()
    assert report["sf"].route == "exact"
    assert report["path_sets"].route == "refused"
    assert report["path_sets"].reason == str(error.value)
    # The paths' method needs them.
    monkeypatch.setattr(network_module, "METHOD", "paths")
    with pytest.raises(NotImplementedError, match="METHOD = 'bdd'"):
        Network(links, (0, 0), (2, 2), directed=True).sf()
