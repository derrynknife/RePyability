"""``compare`` on fault trees, networks and phased missions: the exact
difference where both values are worked out, and otherwise paired
simulations with common random numbers, as on the diagrams."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import FaultTree, Network, NonRepairableRBD, PhasedMission
from repyability.rbd import _montecarlo as montecarlo

E, W = surv.Exponential.from_params, surv.Weibull.from_params
OR = {"top": ("or", ["a", "pair"]), "pair": ("and", ["b", "c"])}


def tree(c_alpha=150.0):
    return FaultTree(
        OR,
        {"a": W([400.0, 2.0]), "b": W([100.0, 1.5]), "c": W([c_alpha, 1.5])},
    )


def dynamic(c_alpha=150.0):
    return FaultTree(
        {"top": ("or", ["a", "spare"]), "spare": ("spare", 0.2, ["b", "c"])},
        {"a": W([400.0, 2.0]), "b": W([100.0, 1.5]), "c": W([c_alpha, 1.5])},
    )


def network(c_rate=0.02):
    return Network(
        {
            "a": ("s", "m", E([0.01])),
            "b": ("m", "t", E([0.01])),
            "c": ("s", "t", E([c_rate])),
        },
        "s",
        "t",
    )


def mission(back=30.0):
    unit = W([100.0, 2.0])
    pair = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": unit, "b": unit},
    )
    one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
    return PhasedMission([("out", 40.0, pair), ("back", back, one)])


def independent_error(a: np.ndarray, b: np.ndarray) -> float:
    """The standard error of the difference of two independent means."""
    return math.hypot(
        montecarlo.standard_error(a, False),
        montecarlo.standard_error(b, False),
    )


@pytest.mark.parametrize(
    "build", [tree, dynamic, network, mission], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("method", [None, "simulate"])
def test_a_design_compared_with_itself_is_no_different(build, method):
    gain = build().compare(build(), mc_samples=2000, seed=1, method=method)
    assert gain.estimate == 0.0
    assert gain.lower == gain.upper == 0.0


def test_the_exact_differences_are_those_of_the_values():
    better, worse = tree(300.0), tree()
    gain = better.compare(worse)
    assert gain.method == "exact" and gain.standard_error == 0.0
    assert gain.estimate == pytest.approx(better.mean() - worse.mean())
    at = better.compare(worse, 50.0)
    assert at.estimate == pytest.approx(
        better.top_event_probability(50.0) - worse.top_event_probability(50.0)
    )
    nets = network(0.01).compare(network())
    assert nets.method == "exact"
    assert nets.estimate == pytest.approx(
        network(0.01).mean() - network().mean()
    )
    missions = mission(20.0).compare(mission())
    assert missions.method == "exact"
    assert missions.estimate == pytest.approx(
        mission(20.0).reliability() - mission().reliability()
    )


def test_a_tree_of_probabilities_compares_its_top_events():
    a = FaultTree(OR, {"a": 0.1, "b": 0.2, "c": 0.3})
    b = FaultTree(OR, {"a": 0.1, "b": 0.2, "c": 0.5})
    gain = a.compare(b)
    assert gain.estimate == pytest.approx(
        a.top_event_probability() - b.top_event_probability()
    )
    with pytest.raises(ValueError, match="compared exactly"):
        a.compare(b, method="simulate")


def test_simulated_differences_are_paired():
    # Seeded: each interval covers the exact difference, with a standard
    # error well below that of two independent runs of the same size.
    n = 20_000
    cases = [
        (tree(300.0), tree(), tree(300.0).mean() - tree().mean()),
        (network(0.01), network(), network(0.01).mean() - network().mean()),
    ]
    for better, worse, exact in cases:
        gain = better.compare(worse, mc_samples=n, seed=3, method="simulate")
        assert gain.method == "simulated" and gain.n_samples == n
        assert gain.lower < exact < gain.upper
        apart = independent_error(
            better.random(n, seed=4), worse.random(n, seed=5)
        )
        assert gain.standard_error < 0.6 * apart
    short, long = mission(20.0), mission()
    gain = short.compare(long, mc_samples=n, seed=3, method="simulate")
    exact = short.reliability() - long.reliability()
    assert gain.lower < exact < gain.upper
    apart = math.hypot(
        short.reliability_interval(n, seed=4).standard_error,
        long.reliability_interval(n, seed=5).standard_error,
    )
    assert gain.standard_error < 0.6 * apart


def test_a_dynamic_tree_is_simulated_on_its_own_draws():
    # The difference of the means, as the area between the trees'
    # (integrated) probabilities of the top event not having occurred.
    better, worse = dynamic(300.0), dynamic()
    gain = better.compare(worse, mc_samples=20_000, seed=3)
    assert gain.method == "simulated"
    exact, _ = integrate.quad(
        lambda t: float(better.sf(t)) - float(worse.sf(t)),
        0.0,
        2000.0,
        limit=200,
    )
    assert gain.lower < exact < gain.upper


def test_a_tree_compares_as_its_diagram_does():
    a, b = tree(300.0), tree()
    ours = a.compare(b, mc_samples=5000, seed=11, method="simulate")
    theirs = a.to_rbd().compare(
        b.to_rbd(), mc_samples=5000, seed=11, method="simulate"
    )
    assert ours.to_dict() == theirs.to_dict()


def test_elements_named_differently_are_drawn_apart():
    renamed = FaultTree(
        {"top": ("or", ["x", "pair"]), "pair": ("and", ["b", "c"])},
        {"x": W([400.0, 2.0]), "b": W([100.0, 1.5]), "c": W([150.0, 1.5])},
    )
    gain = renamed.compare(tree(), mc_samples=5000, seed=1, method="simulate")
    assert gain.lower < 0.0 < gain.upper
    other = Network({"x": ("s", "t", E([0.01]))}, "s", "t")
    assert other.compare(network(), mc_samples=2000, seed=1).method == "exact"
    other.compare(network(), mc_samples=2000, seed=1, method="simulate")


@pytest.mark.parametrize(
    "this, other, match",
    [
        (tree(), tree().to_rbd(), "FaultTree.from_rbd"),
        (network(), tree(), "another network"),
        (mission(), tree(), "another mission"),
    ],
)
def test_a_design_of_another_kind_is_refused(this, other, match):
    with pytest.raises(TypeError, match=match):
        this.compare(other)


def test_a_diagram_names_the_tree_conversion():
    with pytest.raises(TypeError, match="to_rbd"):
        tree().to_rbd().compare(tree())


def test_the_mean_takes_no_time():
    with pytest.raises(ValueError, match="leave t out"):
        tree().compare(tree(), 10.0, quantity="mean")
