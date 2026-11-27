"""A network's link capacities: the maximum flow between its terminals,
and its distribution at a time, in the long run and over a window.

The maximum flow is checked on every state of small networks against the
least capacity of a cut, every cut listed, in exact arithmetic; the
distributions against every state enumerated in ``Fraction``s, against a
``RepairableRBD``'s capacity analysis of the same series-parallel system
and against closed forms; the simulation, seeded, within four standard
errors.
"""

import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest
import surpyval as surv

from repyability import Network, NonRepairableRBD, RepairableRBD, network
from repyability.rbd import _network_capacity
from repyability.rbd.routes import EXACT, NUMERICAL, REFUSED

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params

BRIDGE = {
    "a": ("s", "x"),
    "b": ("s", "y"),
    "c": ("x", "y"),
    "d": ("x", "t"),
    "e": ("y", "t"),
}
# Each element's probability of having failed, as exact fractions.
Q = {
    "a": Fraction(1, 10),
    "b": Fraction(1, 5),
    "c": Fraction(3, 10),
    "d": Fraction(3, 20),
    "e": Fraction(1, 20),
    "f": Fraction(1, 4),
    "x": Fraction(1, 50),
    "s": Fraction(1, 25),
    "t": Fraction(3, 100),
}


def fixed(links, q=Q):
    return {name: (u, v, float(q[name])) for name, (u, v) in links.items()}


def cases():
    """Small networks of fixed probabilities with capacities: the bridge
    (with a failing node; one way; with links of no limit; with failing
    terminals), parallel links, and a mesh with a link the wrong way
    round for its flow."""
    whole = {"a": 3, "b": 2, "c": 1, "d": 2, "e": 3}
    mesh = {**BRIDGE, "f": ("s", "t")}
    return {
        "bridge": Network(
            fixed(BRIDGE), "s", "t", nodes={"x": float(Q["x"])}, capacity=whole
        ),
        "directed bridge": Network(
            fixed(BRIDGE), "s", "t", directed=True, capacity=whole
        ),
        "against the grain": Network(
            fixed({**BRIDGE, "c": ("y", "x")}),
            "s",
            "t",
            directed=True,
            capacity={"a": 1, "b": 4, "c": 2, "d": 3, "e": 1},
        ),
        "some unlimited": Network(
            fixed(BRIDGE), "s", "t", capacity={"a": 3, "c": 1.5, "e": 2}
        ),
        "an unlimited path": Network(
            fixed(mesh), "s", "t", capacity={"a": 3, "b": 2, "d": 2.5}
        ),
        "failing terminals": Network(
            fixed(mesh),
            "s",
            "t",
            nodes={n: float(Q[n]) for n in ("s", "t", "x")},
            capacity={"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.25, "e": 0.7},
        ),
        "parallel": Network(
            fixed({n: ("s", "t") for n in "abcd"}),
            "s",
            "t",
            capacity={"a": 1, "b": 2, "c": 4, "d": 8},
        ),
    }


def least_cut(net, works) -> Fraction:
    """The maximum flow of the state ``works`` (each element working or
    not) as the least capacity of a cut, every cut listed: each set of
    nodes holding the source and not the target, and the capacities of
    the working links out of it (None for no limit, ``inf``). A failed
    node's links carry nothing."""
    nodes = list(net._adjacent)
    alive = {n for n in nodes if works.get(n, True)}
    if net.source not in alive or net.target not in alive:
        return Fraction(0)
    links = [
        (u, v, net.capacity.get(name))
        for name, (u, v) in net.links.items()
        if works[name] and u in alive and v in alive
    ]
    others = [n for n in nodes if n not in (net.source, net.target)]
    best = None
    for size in range(len(others) + 1):
        for chosen in itertools.combinations(others, size):
            side = {net.source, *chosen}
            total: object = Fraction(0)
            for u, v, limit in links:
                out = u in side and v not in side
                if not net.directed:
                    out = out or (v in side and u not in side)
                if out:
                    if limit is None:
                        total = math.inf
                        break
                    # As the decimal it was given as: 0.1 + 0.2 is 0.3.
                    total += Fraction(str(limit))
            if best is None or total < best:
                best = total
    return best


def states(net):
    names = list(net.models)
    for values in itertools.product((True, False), repeat=len(names)):
        yield dict(zip(names, values))


def enumerated(net, q=Q) -> dict:
    """The capacity's exact distribution: every state of the elements,
    its probability a product of fractions, and its least cut."""
    out: dict = {}
    for works in states(net):
        p = Fraction(1)
        for name, up in works.items():
            p *= 1 - q[name] if up else q[name]
        level = least_cut(net, works)
        out[level] = out.get(level, Fraction(0)) + p
    return out


def close(distribution, exact: dict, tolerance: float) -> None:
    levels = sorted(level for level, p in exact.items() if p)
    assert distribution.levels.tolist() == [float(v) for v in levels]
    assert distribution.probabilities == pytest.approx(
        [float(exact[v]) for v in levels], rel=tolerance, abs=0.0
    )


# -- the maximum flow ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(cases()))
def test_the_maximum_flow_is_the_least_cut_on_every_state(name):
    net = cases()[name]
    flows = _network_capacity.graph(net)
    for works in states(net):
        found = flows.value([works[n] for n in net.models])
        assert found == float(least_cut(net, works)), works


# -- the exact distribution ---------------------------------------------------


@pytest.mark.parametrize("name", sorted(cases()))
def test_the_distribution_is_every_state_enumerated(name):
    net = cases()[name]
    close(net.capacity_distribution(), enumerated(net), 1e-13)


@pytest.mark.parametrize("name", sorted(cases()))
def test_the_capacity_is_positive_exactly_when_the_terminals_are_joined(name):
    net = cases()[name]
    capacity = net.capacity_distribution()
    least = capacity.levels[capacity.levels > 0].min()
    assert capacity.meets(least) == pytest.approx(net.sf(), rel=1e-13)
    assert capacity.probabilities[0] == pytest.approx(net.ff(), rel=1e-13)


def test_the_states_worked_out_are_far_fewer_than_every_state():
    # A chain of ten links ends at the first to fail: eleven ends, where
    # there are 1,024 states.
    chain = Network(
        {i: (i, i + 1, 0.1) for i in range(10)},
        0,
        10,
        capacity={i: 1 + i for i in range(10)},
    )
    capacity = chain.capacity_distribution()
    assert chain._capacity_tree.ends == 11
    assert capacity.levels.tolist() == [0.0, 1.0]
    assert capacity.probabilities[1] == pytest.approx(0.9**10, rel=1e-14)


def test_at_times_the_distribution_is_the_elements_reliabilities():
    models = {n: W([100 + 10 * i, 1.5]) for i, n in enumerate(BRIDGE)}
    net = Network(
        {n: (u, v, models[n]) for n, (u, v) in BRIDGE.items()},
        "s",
        "t",
        capacity={"a": 3, "b": 2, "c": 1, "d": 2, "e": 3},
    )
    x = np.array([[10.0, 50.0, 120.0], [1.0, 400.0, 80.0]])
    capacity = net.capacity_distribution(x)
    assert capacity.probabilities.shape == (len(capacity.levels), 2, 3)
    assert capacity.probabilities.sum(axis=0) == pytest.approx(1.0, abs=1e-14)
    for i, j in itertools.product(range(2), range(3)):
        t = float(x[i, j])
        # The elements' probabilities as the fractions the floats are.
        q = {n: Fraction(float(m.ff(t))) for n, m in models.items()}
        exact = enumerated(net, q)
        for level, row in zip(capacity.levels, capacity.probabilities):
            assert row[i, j] == pytest.approx(
                float(exact[Fraction(level)]), rel=1e-12, abs=1e-300
            )
        one = net.capacity_distribution(t)
        assert one.probabilities == pytest.approx(
            [float(exact[Fraction(level)]) for level in one.levels.tolist()],
            rel=1e-12,
        )
    assert capacity.meets(1)[0] == pytest.approx(net.sf(x[0]), rel=1e-13)


def test_a_directed_network_is_its_diagram():
    models = {n: W([100 + 10 * i, 1.5]) for i, n in enumerate(BRIDGE)}
    net = Network(
        {n: (u, v, models[n]) for n, (u, v) in BRIDGE.items()},
        "s",
        "t",
        nodes={"x": E([0.001])},
        directed=True,
        capacity={"a": 3, "b": 2, "c": 1, "d": 2, "e": 3},
    )
    rbd = net.to_rbd()
    assert isinstance(rbd, NonRepairableRBD)
    x = np.array([20.0, 90.0])
    ours, theirs = net.capacity_distribution(x), rbd.capacity_distribution(x)
    assert ours.levels.tolist() == theirs.levels.tolist()
    assert ours.probabilities == pytest.approx(theirs.probabilities, rel=1e-12)


# -- repaired elements --------------------------------------------------------


def series_parallel(life, repair, other):
    """A feeder into two lines side by side, as a network and as the
    diagram of the same three components, with the same capacities."""
    specs = {
        "feeder": {"reliability": life, "repairability": repair},
        "north": {"reliability": other, "repairability": repair},
        "south": {"reliability": life, "repairability": other},
    }
    capacity = {"feeder": 90, "north": 60, "south": 40}
    net = Network(
        {
            "feeder": ("plant", "hub", specs["feeder"]),
            "north": ("hub", "town", specs["north"]),
            "south": ("hub", "town", specs["south"]),
        },
        "plant",
        "town",
        capacity=capacity,
    )
    rbd = RepairableRBD(
        [
            ("in", "feeder"),
            ("feeder", "north"),
            ("feeder", "south"),
            ("north", "out"),
            ("south", "out"),
        ],
        specs,
        capacity=capacity,
    )
    return net, rbd


def same(ours, theirs, tolerance):
    assert ours.levels.tolist() == theirs.levels.tolist()
    assert ours.probabilities == pytest.approx(
        theirs.probabilities, rel=tolerance, abs=1e-300
    )


def test_a_series_parallel_network_is_its_diagram():
    # Lives and repairs with no closed form, so both take each element's
    # availability from the same curve.
    net, rbd = series_parallel(W([100.0, 1.5]), LN([1.0, 0.5]), W([60.0, 2.0]))
    same(net.capacity_distribution(), rbd.capacity_distribution(), 1e-12)
    x = np.array([0.0, 5.0, 40.0, 150.0])
    same(net.point_capacity(x), rbd.point_capacity(x), 1e-9)
    same(net.point_capacity(40.0), rbd.point_capacity(40.0), 1e-9)
    t = np.array([10.0, 80.0, 300.0])
    same(net.mission_capacity(t), rbd.mission_capacity(t), 1e-9)
    same(net.mission_capacity(80.0), rbd.mission_capacity(80.0), 1e-9)
    assert net.point_capacity(x).meets(1) == pytest.approx(
        net.point_availability(x), rel=1e-12
    )


def exponential_pair():
    line = {"reliability": E([0.01]), "repairability": E([0.5])}
    return Network(
        {"a": ("s", "t", line), "b": ("s", "t", line)},
        "s",
        "t",
        capacity={"a": 60, "b": 40},
    )


def test_exponential_links_have_closed_forms():
    net = exponential_pair()
    rate = 0.51

    def q(t):
        return (0.01 / rate) * -np.expm1(-rate * t)

    def levels(down):
        return [down**2, down * (1 - down), down * (1 - down), (1 - down) ** 2]

    x = np.array([0.5, 10.0, 200.0])
    at = net.point_capacity(x)
    assert at.levels.tolist() == [0.0, 40.0, 60.0, 100.0]
    assert at.probabilities == pytest.approx(np.array(levels(q(x))), rel=1e-12)
    long_run = net.capacity_distribution()
    assert long_run.probabilities == pytest.approx(
        levels(0.01 / rate), rel=1e-12
    )
    # Over a window: the mean of q and of q^2, in closed form.
    t = 30.0
    decay = -np.expm1(-rate * t) / (rate * t)
    twice = -np.expm1(-2 * rate * t) / (2 * rate * t)
    first = (0.01 / rate) * (1 - decay)
    second = (0.01 / rate) ** 2 * (1 - 2 * decay + twice)
    over = net.mission_capacity(t)
    assert over.probabilities == pytest.approx(
        [second, first - second, first - second, 1 - 2 * first + second],
        rel=1e-7,
    )
    assert over.mean == pytest.approx(100 * (1 - first), rel=1e-9)
    # At 0 every element is new, and a window of no length is that.
    assert net.point_capacity(0.0).levels.tolist() == [100.0]
    assert net.mission_capacity(0.0).probabilities.tolist() == [1.0]


def test_a_long_window_approaches_the_long_run():
    net, _ = series_parallel(W([100.0, 1.5]), LN([1.0, 0.5]), W([60.0, 2.0]))
    long_run = net.capacity_distribution()
    over = net.mission_capacity(1e6)
    assert over.probabilities == pytest.approx(
        long_run.probabilities, rel=1e-3
    )


# -- the simulation -----------------------------------------------------------


def within_four_standard_errors(estimate, exact, samples):
    levels = exact.levels.tolist()
    assert set(estimate.levels.tolist()) <= set(levels)
    found = dict(zip(estimate.levels.tolist(), estimate.probabilities))
    for level, p in zip(levels, exact.probabilities):
        error = np.sqrt(p * (1 - p) / samples)
        assert np.all(np.abs(found.get(level, 0.0) - p) <= 4 * error), level
    assert np.sum(estimate.probabilities, axis=0) == pytest.approx(1.0)


@pytest.mark.parametrize("name", sorted(cases()))
def test_the_simulated_distribution_is_within_four_standard_errors(name):
    net = cases()[name]
    estimate = net.capacity_distribution(
        method="simulate", mc_samples=20_000, seed=11
    )
    within_four_standard_errors(estimate, net.capacity_distribution(), 20_000)
    again = net.capacity_distribution(
        method="simulate", mc_samples=20_000, seed=11
    )
    assert again.probabilities.tolist() == estimate.probabilities.tolist()


def test_the_simulation_over_time_is_within_four_standard_errors():
    net, _ = series_parallel(W([100.0, 1.5]), LN([1.0, 0.5]), W([60.0, 2.0]))
    x = np.array([5.0, 40.0, 150.0])
    for name in ("point_capacity", "mission_capacity"):
        method = getattr(net, name)
        estimate = method(x, method="simulate", mc_samples=20_000, seed=5)
        assert estimate.probabilities.shape[1] == 3
        within_four_standard_errors(estimate, method(x), 20_000)
        one = method(40.0, method="simulate", mc_samples=20_000, seed=5)
        assert one.probabilities.ndim == 1
        within_four_standard_errors(one, method(40.0), 20_000)
    estimate = net.capacity_distribution(
        method="simulate", mc_samples=20_000, seed=5
    )
    within_four_standard_errors(estimate, net.capacity_distribution(), 20_000)


def test_lifetimes_simulate_at_each_time():
    models = {n: W([100 + 10 * i, 1.5]) for i, n in enumerate(BRIDGE)}
    net = Network(
        {n: (u, v, models[n]) for n, (u, v) in BRIDGE.items()},
        "s",
        "t",
        capacity={"a": 3, "b": 2, "c": 1, "d": 2, "e": 3},
    )
    x = np.array([30.0, 90.0])
    estimate = net.capacity_distribution(
        x, method="simulate", mc_samples=20_000, seed=2
    )
    within_four_standard_errors(estimate, net.capacity_distribution(x), 20_000)


# -- too many states ----------------------------------------------------------


def test_too_many_states_refuse_and_point_to_the_simulation(monkeypatch):
    assert network.MAX_CAPACITY_STATES == 100_000
    net = cases()["parallel"]  # sixteen levels, a state each
    monkeypatch.setattr(network, "MAX_CAPACITY_STATES", 15)
    with pytest.raises(NotImplementedError, match="more than 15 states") as e:
        net.capacity_distribution()
    assert "method='simulate'" in str(e.value)
    assert "repyability.network.MAX_CAPACITY_STATES" in str(e.value)
    route = net.analysis_routes()["capacity_distribution"]
    assert (route.route, route.reason) == (REFUSED, str(e.value))
    # The refusal is kept, and the simulation still runs.
    with pytest.raises(NotImplementedError, match="more than 15 states"):
        net.capacity_distribution()
    estimate = net.capacity_distribution(method="simulate", seed=1)
    assert estimate.mean == pytest.approx(15 * 0.8, rel=0.05)
    # The limit is read when the capacity is first worked out.
    monkeypatch.setattr(network, "MAX_CAPACITY_STATES", 16)
    assert len(cases()["parallel"].capacity_distribution().levels) == 16


# -- routes and refusals ------------------------------------------------------


def test_the_routes_say_how_the_capacity_is_worked_out():
    fixed_routes = cases()["bridge"].analysis_routes()
    assert fixed_routes["capacity_distribution"].route == EXACT
    for name in ("point_capacity", "mission_capacity"):
        assert fixed_routes[name].route == REFUSED
        assert "capacity_distribution(x)" in fixed_routes[name].reason
    repaired = exponential_pair().analysis_routes()
    assert repaired["capacity_distribution"].route == EXACT
    assert repaired["point_capacity"].route == EXACT
    assert repaired["mission_capacity"].route == NUMERICAL
    numerical, _ = series_parallel(
        W([100.0, 1.5]), LN([1.0, 0.5]), W([60.0, 2.0])
    )
    assert numerical.analysis_routes()["point_capacity"].route == NUMERICAL


def test_a_network_without_capacities_refuses_naming_the_argument():
    net = Network(fixed(BRIDGE), "s", "t")
    assert net.capacity == {}
    for name in (
        "capacity_distribution",
        "point_capacity",
        "mission_capacity",
    ):
        with pytest.raises(ValueError, match="capacity={link: amount}") as e:
            getattr(net, name)(1.0)
        assert net.analysis_routes()[name].reason == str(e.value)


def test_capacity_over_time_needs_repaired_elements():
    net = cases()["bridge"]
    for name in ("point_capacity", "mission_capacity"):
        with pytest.raises(ValueError, match=r"capacity_distribution\(x\)"):
            getattr(net, name)(1.0)


def test_a_repaired_network_takes_no_time_for_its_long_run():
    net = exponential_pair()
    with pytest.raises(ValueError, match=r"point_capacity\(x\)"):
        net.capacity_distribution(10.0)
    with pytest.raises(ValueError, match="0 or more|non-negative"):
        net.point_capacity(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        net.mission_capacity(-1.0)


def test_a_lifetime_network_needs_a_time():
    net = Network({"a": ("s", "t", W([100, 2]))}, "s", "t", capacity={"a": 5})
    with pytest.raises(ValueError, match="x is required"):
        net.capacity_distribution()


@pytest.mark.parametrize(
    "capacity, error, match",
    [
        ({"z": 1}, ValueError, "Unknown link 'z'"),
        ({"x": 1}, ValueError, "'x' in capacity is a node"),
        ({"a": 0}, ValueError, "must be positive"),
        ({"a": -1.0}, ValueError, "must be positive"),
        ({"a": float("nan")}, ValueError, "must be positive"),
        ({"a": "3"}, ValueError, "must be a number"),
        ({"a": True}, ValueError, "must be a number"),
        ([("a", 1)], ValueError, "dict of link: amount"),
    ],
)
def test_capacities_are_checked(capacity, error, match):
    with pytest.raises(error, match=match):
        Network(fixed(BRIDGE), "s", "t", nodes={"x": 0.02}, capacity=capacity)


def test_capacities_take_two_terminals():
    with pytest.raises(NotImplementedError, match="two terminals"):
        Network(fixed(BRIDGE), terminals="all", capacity={"a": 1})
    # Without capacities, as before.
    assert Network(fixed(BRIDGE), terminals="all", capacity={}).capacity == {}


def test_the_simulation_options_are_checked():
    net = cases()["bridge"]
    with pytest.raises(ValueError, match="only with method='simulate'"):
        net.capacity_distribution(seed=1)
    with pytest.raises(ValueError, match="only with method='simulate'"):
        exponential_pair().point_capacity(1.0, mc_samples=10)
    with pytest.raises(ValueError, match="method must be"):
        net.capacity_distribution(method="auto")
    with pytest.raises(ValueError, match="mc_samples must be a positive"):
        net.capacity_distribution(method="simulate", mc_samples=0)
    with pytest.raises(TypeError, match="seed must be"):
        net.capacity_distribution(method="simulate", seed=1.5)
    with pytest.raises(TypeError, match="mc_samples"):
        net.capacity_distribution(method="simulate", n_sims=10)


# -- saving, levers, results --------------------------------------------------


def test_capacities_are_saved_and_rebuilt():
    for net in (cases()["some unlimited"], exponential_pair()):
        again = Network.from_json(net.to_json())
        assert again.capacity == net.capacity
        kwargs = {} if net.is_repairable else {"x": None}
        same(
            again.capacity_distribution(**kwargs),
            net.capacity_distribution(**kwargs),
            1e-15,
        )
    grid = Network(
        {((0, 0), (0, 1)): ((0, 0), (0, 1), 0.1)},
        (0, 0),
        (0, 1),
        capacity={((0, 0), (0, 1)): math.inf},
    )
    again = Network.from_json(grid.to_json())
    assert again.capacity == grid.capacity
    assert again.capacity_distribution().levels.tolist() == [0.0, math.inf]
    assert "capacity" not in Network(fixed(BRIDGE), "s", "t").to_dict()


def test_a_network_with_levers_moved_keeps_its_capacities():
    net = cases()["bridge"]
    moved = net.with_levers({("a", "p"): 0.5})
    assert moved.capacity == net.capacity
    q = {**Q, "a": Fraction(1, 2)}
    close(moved.capacity_distribution(), enumerated(moved, q), 1e-13)


def test_the_result_goes_through_json():
    capacity = cases()["an unlimited path"].capacity_distribution()
    assert capacity.levels[-1] == math.inf
    assert capacity.mean == math.inf
    assert capacity.delivered_fraction(4) == pytest.approx(
        float(
            sum(
                min(level, 4) * p
                for level, p in enumerated(
                    cases()["an unlimited path"]
                ).items()
            )
            / 4
        ),
        rel=1e-13,
    )
    saved = json.loads(json.dumps(capacity.to_dict()))
    assert saved["probabilities"] == capacity.probabilities.tolist()


def test_the_capacity_needs_no_decision_diagram(monkeypatch):
    # A network too large for its decision diagram refuses sf, and still
    # gives its capacity: the maximum flow is worked out on the graph.
    monkeypatch.setattr(network, "MAX_STATES", 3)
    net = cases()["bridge"]
    with pytest.raises(NotImplementedError):
        net.sf()
    assert net.analysis_routes()["capacity_distribution"].route == EXACT
    close(net.capacity_distribution(), enumerated(net), 1e-13)


def test_the_capacity_does_not_depend_on_the_unit_of_time():
    def pair(unit):
        line = {
            "reliability": W([100.0 * unit, 1.5]),
            "repairability": LN([1.0 + math.log(unit), 0.5]),
        }
        return Network(
            {"a": ("s", "t", line), "b": ("s", "t", line)},
            "s",
            "t",
            capacity={"a": 60, "b": 40},
        )

    base = pair(1.0)
    for unit in (1e-6, 1e6):
        scaled = pair(unit)
        same(
            scaled.capacity_distribution(), base.capacity_distribution(), 1e-9
        )
        same(
            scaled.point_capacity(30.0 * unit), base.point_capacity(30.0), 1e-9
        )
        same(
            scaled.mission_capacity(30.0 * unit),
            base.mission_capacity(30.0),
            1e-9,
        )
