"""A network mixing repaired elements with fixed probabilities and lives
that are not repaired, and elements whose hidden failures are found by
tests.

Checked against closed forms, the fault tree of the same structure (the
OR of the network's minimal cut sets), every state of the elements
enumerated in ``Fraction``s at the elements' own probabilities, and the
``RepairableRBD`` of the same tested components.
"""

import itertools
import json
from fractions import Fraction

import numpy as np
import pytest
import surpyval as surv

from repyability import FaultTree, Network, RepairableRBD
from repyability.rbd.routes import EXACT, NUMERICAL, REFUSED

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params
LL = surv.LogLogistic.from_params

T = np.array([0.0, 0.5, 3.0, 20.0, 150.0])

BRIDGE = {
    "a": ("s", "x"),
    "b": ("s", "y"),
    "c": ("x", "y"),
    "d": ("x", "t"),
    "e": ("y", "t"),
}
PATHS = [{"a", "d"}, {"b", "e"}, {"a", "c", "e"}, {"b", "c", "d"}]
LONG_RUN = (
    "mean_availability",
    "mean_unavailability",
    "system_failure_frequency",
)


def spec(life, repair):
    return {"reliability": life, "repairability": repair}


def mixed_models(lives: bool = True):
    """The bridge's links: two repaired, a fixed probability, and (with
    ``lives``) two lives that are not repaired, else two more repaired."""
    return {
        "a": spec(E([0.01]), E([0.5])),
        "b": spec(W([80.0, 1.5]), LN([1.0, 0.5])),
        "c": 0.07,
        "d": W([120.0, 0.8]) if lives else spec(W([120.0, 0.8]), E([0.4])),
        "e": E([0.004]) if lives else spec(E([0.004]), W([4.0, 2.0])),
    }


def bridge(models, **kwargs):
    return Network(
        {n: (u, v, models[n]) for n, (u, v) in BRIDGE.items()},
        "s",
        "t",
        **kwargs,
    )


def as_tree(net, models):
    """The fault tree of the network's connection being lost: the OR of
    its minimal cut sets, each the AND of its elements."""
    cuts = [sorted(cut) for cut in net.minimal_cut_sets()]
    gates = {"top": ("or", [f"cut {i}" for i in range(len(cuts))])}
    for i, cut in enumerate(cuts):
        gates[f"cut {i}"] = ("and", cut)
    return FaultTree(gates, dict(models))


def parted(q: dict) -> Fraction:
    """The probability that the bridge's terminals are parted, every
    state of the links enumerated, each down with ``q`` exactly."""
    total = Fraction(0)
    for state in itertools.product((True, False), repeat=len(BRIDGE)):
        works = dict(zip(BRIDGE, state))
        if any(all(works[n] for n in path) for path in PATHS):
            continue
        p = Fraction(1)
        for name, up in works.items():
            p *= 1 - q[name] if up else q[name]
        total += p
    return total


# -- closed forms ----------


def test_a_repaired_line_beside_a_probability_and_a_life():
    # In series: the connection is lost when any of the three is down.
    rate, repair, p, life = 0.01, 0.5, 0.03, 0.002
    net = Network(
        {
            "line": ("s", "x", spec(E([rate]), E([repair]))),
            "breaker": ("x", "y", p),
            "cable": ("y", "t", E([life])),
        },
        "s",
        "t",
    )
    assert net.is_repairable and not net.is_fixed
    q = (rate / (rate + repair)) * -np.expm1(-(rate + repair) * T)
    survives = np.exp(-life * T)
    np.testing.assert_allclose(
        net.point_availability(T), (1 - q) * (1 - p) * survives, rtol=1e-12
    )
    np.testing.assert_allclose(
        net.point_unavailability(T),
        1 - (1 - q) * (1 - p) * survives,
        rtol=1e-12,
        atol=1e-300,
    )
    # The fixed probability never starts a loss over time; the line's
    # failures and the cable's one failure do, while the others are up.
    np.testing.assert_allclose(
        net.failure_frequency(T),
        (1 - p) * survives * (rate * (1 - q) + (1 - q) * life),
        rtol=1e-12,
    )
    # Two lines side by side, one repaired and one a fixed probability.
    pair = Network(
        {"a": ("s", "t", spec(E([rate]), E([repair]))), "b": ("s", "t", p)},
        "s",
        "t",
    )
    np.testing.assert_allclose(pair.point_unavailability(T), p * q, rtol=1e-12)
    down = rate / (rate + repair)
    assert pair.mean_unavailability() == pytest.approx(p * down, rel=1e-12)
    assert pair.mean_availability() == pytest.approx(1 - p * down, rel=1e-14)
    assert pair.system_failure_frequency() == pytest.approx(
        p * rate * (1 - down), rel=1e-12
    )


# -- the same structure as a fault tree ----------


@pytest.mark.parametrize("lives", [True, False])
def test_a_mixed_bridge_is_the_fault_tree_of_its_cut_sets(lives):
    models = mixed_models(lives)
    net = bridge(models)
    tree = as_tree(net, models)
    np.testing.assert_allclose(
        net.point_unavailability(T),
        tree.point_unavailability(T),
        rtol=1e-12,
        atol=1e-300,
    )
    np.testing.assert_allclose(
        net.mission_unavailability(T[1:]),
        tree.mean_unavailability(T[1:]),
        rtol=1e-9,
    )
    # At 0 too, where a life whose density starts infinite is not
    # critical.
    np.testing.assert_allclose(
        net.failure_frequency(T), tree.failure_frequency(T), rtol=1e-10
    )
    assert np.isfinite(net.failure_frequency(T)).all()
    np.testing.assert_allclose(
        net.expected_failures(T[1:]), tree.expected_failures(T[1:]), rtol=1e-9
    )
    if not lives:
        assert net.mean_unavailability() == pytest.approx(
            tree.mean_unavailability(), rel=1e-12
        )
        assert net.system_failure_frequency() == pytest.approx(
            tree.system_failure_frequency(), rel=1e-12
        )


def test_the_failure_frequency_at_0_is_the_initial_intensity():
    # New at 0, every element is up: the connection is lost at 0 only
    # through the repaired link a while the fixed probability beside it
    # is down, at a's density then, 1/50 (a log-logistic of shape 1). The
    # repaired d, whose density starts infinite, and the life e are not
    # critical at 0, each needing the other down.
    models = {
        "a": spec(LL([50.0, 1.0]), E([0.5])),
        "c": 0.3,
        "d": spec(W([120.0, 0.8]), E([0.4])),
        "e": E([0.004]),
    }
    links = {
        "a": ("s", "x"),
        "c": ("s", "x"),
        "d": ("x", "t"),
        "e": ("x", "t"),
    }
    net = Network(
        {n: (u, v, models[n]) for n, (u, v) in links.items()}, "s", "t"
    )
    assert net.failure_frequency(0.0) == pytest.approx(0.3 / 50.0, rel=1e-12)
    at = net.failure_frequency([0.0, 1e-7, 0.5])
    assert np.isfinite(at).all()
    # It is where the frequency starts from.
    assert at[1] == pytest.approx(at[0], rel=1e-6)
    np.testing.assert_allclose(
        at,
        as_tree(net, models).failure_frequency([0.0, 1e-7, 0.5]),
        rtol=1e-10,
    )


def test_a_mixed_bridge_is_every_state_at_the_elements_probabilities():
    models = mixed_models()
    net = bridge(models)
    # Each element alone between two terminals gives its own probability
    # of being down, exactly as the bridge takes it.
    alone = {
        name: Network({name: ("s", "t", model)}, "s", "t")
        for name, model in models.items()
        if isinstance(model, dict)
    }
    for t in (3.0, 20.0, 150.0):
        q = {
            "c": Fraction(0.07),
            "d": Fraction(float(models["d"].ff(t))),
            "e": Fraction(float(models["e"].ff(t))),
            **{
                name: Fraction(one.point_unavailability(t))
                for name, one in alone.items()
            },
        }
        assert net.point_unavailability(t) == pytest.approx(
            float(parted(q)), rel=1e-12
        )
        assert net.point_availability(t) == pytest.approx(
            float(1 - parted(q)), rel=1e-12
        )


def test_the_importance_measures_take_the_mixed_elements():
    models = mixed_models()
    net = bridge(models)
    t = 20.0
    whole = net.point_unavailability(t)
    birnbaum = net.birnbaum_importance(t)
    # By differences: the connection's unavailability with the element
    # down, less with it up, from networks with it fixed at 1 and at 0.
    for name in BRIDGE:
        down = bridge({**models, name: 1.0}).point_unavailability(t)
        up = bridge({**models, name: 0.0}).point_unavailability(t)
        assert birnbaum[name] == pytest.approx(down - up, rel=1e-9)
    assert set(net.importances(t).measures) >= {"birnbaum_importance"}
    assert whole > 0.0


# -- no long run ----------


def test_a_life_that_is_not_repaired_has_no_long_run():
    net = bridge(mixed_models())
    routes = net.analysis_routes()
    for name in LONG_RUN:
        with pytest.raises(NotImplementedError, match="no long run") as e:
            getattr(net, name)()
        assert "['d', 'e']" in str(e.value)
        assert "mission_unavailability(t)" in str(e.value)
        assert (routes[name].route, routes[name].reason) == (
            REFUSED,
            str(e.value),
        )
    assert routes["point_unavailability"].route == NUMERICAL
    assert routes["mission_unavailability"].route == NUMERICAL
    # Fixed probabilities have one.
    fixed = bridge(mixed_models(lives=False)).analysis_routes()
    assert {fixed[name].route for name in LONG_RUN} == {EXACT}


def test_a_mixed_network_has_no_lifetime_either():
    net = bridge(mixed_models())
    for name in ("sf", "ff", "mean"):
        with pytest.raises(NotImplementedError, match="point_availability"):
            getattr(net, name)(*(() if name == "mean" else (1.0,)))


# -- capacities ----------


def test_the_capacity_of_a_mixed_network():
    rate, repair, p, life = 0.01, 0.5, 0.03, 0.002
    net = Network(
        {
            "a": ("s", "t", spec(E([rate]), E([repair]))),
            "b": ("s", "t", p),
            "c": ("s", "t", E([life])),
        },
        "s",
        "t",
        capacity={"a": 1, "b": 2, "c": 4},
    )
    t = 30.0
    q = {
        "a": (rate / (rate + repair)) * -np.expm1(-(rate + repair) * t),
        "b": p,
        "c": -np.expm1(-life * t),
    }
    at = net.point_capacity(t)
    assert at.levels.tolist() == [float(i) for i in range(8)]
    for level, found in zip(range(8), at.probabilities):
        expected = 1.0
        for bit, name in zip((1, 2, 4), "abc"):
            expected *= 1 - q[name] if level & bit else q[name]
        assert found == pytest.approx(expected, rel=1e-12)
    # Over a window: each element's share of it down, the elements
    # independent, so the mean capacity is the sum of theirs.
    total = rate + repair
    mean_q = {
        "a": (rate / total) * (1 + np.expm1(-total * t) / (total * t)),
        "b": p,
        "c": 1 + np.expm1(-life * t) / (life * t),
    }
    assert net.mission_capacity(t).mean == pytest.approx(
        sum(c * (1 - mean_q[n]) for c, n in zip((1, 2, 4), "abc")), rel=1e-9
    )
    with pytest.raises(NotImplementedError, match="no long run") as e:
        net.capacity_distribution()
    assert "point_capacity(x) and mission_capacity(t)" in str(e.value)
    route = net.analysis_routes()["capacity_distribution"]
    assert (route.route, route.reason) == (REFUSED, str(e.value))
    # With the life repaired too, the long run takes the probability.
    repaired = Network(
        {
            "a": ("s", "t", spec(E([rate]), E([repair]))),
            "b": ("s", "t", p),
        },
        "s",
        "t",
        capacity={"a": 1, "b": 2},
    )
    down = rate / total
    assert repaired.capacity_distribution().probabilities == pytest.approx(
        [down * p, (1 - down) * p, down * (1 - p), (1 - down) * (1 - p)],
        rel=1e-12,
    )


# -- tested elements ----------


def hidden(rate, interval):
    return {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": interval},
    }


def test_a_tested_line_has_its_hidden_failures_closed_form():
    # A failure is found, and repaired at once, at the next test: at t
    # the line is down with 1 - exp(-rate (t mod interval)).
    rate, interval = 0.002, 50.0
    net = Network({"a": ("s", "t", hidden(rate, interval))}, "s", "t")
    t = np.array([10.0, 49.0, 60.0, 130.0])
    np.testing.assert_allclose(
        net.point_unavailability(t),
        -np.expm1(-rate * np.mod(t, interval)),
        rtol=1e-9,
    )
    # In the long run, the mean of that over an interval.
    assert net.mean_unavailability() == pytest.approx(
        1 + np.expm1(-rate * interval) / (rate * interval), rel=1e-9
    )


def test_tested_elements_are_the_diagram_of_the_same_components():
    specs = {
        "feeder": spec(W([100.0, 1.5]), LN([1.0, 0.5])),
        "north": hidden(0.002, 50.0),
        "south": {
            "reliability": W([300.0, 1.2]),
            "repairability": E([0.2]),
            "inspection": {"interval": 40.0},
        },
    }
    net = Network(
        {
            "feeder": ("plant", "hub", specs["feeder"]),
            "north": ("hub", "town", specs["north"]),
            "south": ("hub", "town", specs["south"]),
        },
        "plant",
        "town",
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
    )
    t = np.array([5.0, 45.0, 130.0, 400.0])
    for name in (
        "point_unavailability",
        "mission_unavailability",
        "expected_failures",
    ):
        np.testing.assert_allclose(
            getattr(net, name)(t), getattr(rbd, name)(t), rtol=1e-9
        )
    for name in LONG_RUN:
        assert getattr(net, name)() == pytest.approx(
            getattr(rbd, name)(), rel=1e-12
        )
    routes = net.analysis_routes()
    assert routes["point_unavailability"].route in (EXACT, NUMERICAL)
    assert routes["mean_unavailability"].route == EXACT


def test_an_element_takes_no_other_key():
    priced = {**spec(E([0.01]), E([0.5])), "cost": 10.0}
    with pytest.raises(ValueError, match="'inspection' only") as e:
        Network({"a": ("s", "t", priced)}, "s", "t")
    assert "RepairableRBD" in str(e.value)


# -- saving ----------


def test_a_mixed_or_tested_network_saves():
    models = {**mixed_models(), "a": hidden(0.002, 50.0)}
    net = bridge(models, nodes={"x": 0.01}, capacity={"a": 3, "e": 2})
    again = Network.from_dict(json.loads(json.dumps(net.to_dict())))
    np.testing.assert_array_equal(
        again.point_unavailability(T), net.point_unavailability(T)
    )
    assert (
        again.point_capacity(20.0).probabilities.tolist()
        == net.point_capacity(20.0).probabilities.tolist()
    )
