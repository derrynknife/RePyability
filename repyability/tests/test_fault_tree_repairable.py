"""Fault trees of repaired and tested basic events: the top event's
unavailability over time ``Q(t)``, its average, its failure frequency
``w(t)`` and expected occurrences, against closed forms, enumeration of
the events' states, the repairable diagram of the same system and its
simulation."""

import itertools
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import FaultTree, RepairableRBD
from repyability.rbd.ccf import BetaFactor, CCFGroup
from repyability.rbd.routes import Refused
from repyability.tests.catalogue import fault_tree_kinds

E = surv.Exponential.from_params
W = surv.Weibull.from_params

T = np.array([0.0, 0.5, 3.0, 20.0, 150.0, 2000.0])


def unit(rate, repair):
    return {"reliability": E([rate]), "repairability": E([repair])}


def q_closed(rate, repair, t):
    """A unit's probability of being down at ``t``, new at 0."""
    return rate / (rate + repair) * -np.expm1(-(rate + repair) * t)


def w_closed(rate, repair, t):
    """A unit's failure intensity at ``t``: its rate while it works."""
    return rate * (1.0 - q_closed(rate, repair, t))


def failures_closed(rate, repair, t):
    """A unit's expected failures in ``[0, t)``."""
    s = rate + repair
    return rate * repair / s * t + (rate / s) ** 2 * -np.expm1(-s * t)


def mean_q_closed(rate, repair, t):
    """A unit's average probability of being down over ``[0, t]``."""
    s = rate + repair
    return rate / s * (1.0 - -np.expm1(-s * t) / (s * t))


RATES = {"a": (0.01, 0.5), "b": (0.02, 0.25), "c": (0.005, 0.1)}


def events(names="abc"):
    return {e: unit(*RATES[e]) for e in names}


# -- closed forms ------------------------------------------------------------


def test_one_event():
    tree = FaultTree({"top": ("or", ["a"])}, events("a"))
    rate, repair = RATES["a"]
    np.testing.assert_allclose(
        tree.point_unavailability(T), q_closed(rate, repair, T), rtol=1e-13
    )
    np.testing.assert_allclose(
        tree.failure_frequency(T), w_closed(rate, repair, T), rtol=1e-13
    )
    window = T[1:]
    np.testing.assert_allclose(
        tree.mean_unavailability(window),
        mean_q_closed(rate, repair, window),
        rtol=1e-9,
    )
    # Counted from the event's renewal equation on a grid, to about 1e-7
    # (see RepairableRBD.expected_failures).
    np.testing.assert_allclose(
        tree.expected_failures(window),
        failures_closed(rate, repair, window),
        atol=1e-7,
    )
    assert tree.mean_unavailability() == pytest.approx(
        rate / (rate + repair), rel=1e-13
    )
    assert tree.system_failure_frequency() == pytest.approx(
        rate * repair / (rate + repair), rel=1e-13
    )
    assert isinstance(tree.point_unavailability(3.0), float)
    assert tree.top_event_probability(3.0) == tree.point_unavailability(3.0)


def test_an_and_and_an_or_of_two():
    q = {e: q_closed(*RATES[e], T) for e in "ab"}
    w = {e: w_closed(*RATES[e], T) for e in "ab"}
    both = FaultTree({"top": ("and", ["a", "b"])}, events("ab"))
    np.testing.assert_allclose(
        both.point_unavailability(T), q["a"] * q["b"], rtol=1e-13
    )
    np.testing.assert_allclose(
        both.failure_frequency(T),
        q["a"] * w["b"] + q["b"] * w["a"],
        rtol=1e-13,
    )
    either = FaultTree({"top": ("or", ["a", "b"])}, events("ab"))
    np.testing.assert_allclose(
        either.point_unavailability(T),
        1.0 - (1.0 - q["a"]) * (1.0 - q["b"]),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        either.failure_frequency(T),
        (1.0 - q["b"]) * w["a"] + (1.0 - q["a"]) * w["b"],
        rtol=1e-13,
    )
    # The Birnbaum importance at a time is read from the events' q(t).
    np.testing.assert_allclose(
        both.birnbaum_importance(T)["a"], q["b"], rtol=1e-13
    )


def test_two_out_of_three():
    tree = FaultTree({"top": ("vote", 2, ["a", "b", "c"])}, events())
    q = {e: q_closed(*RATES[e], T) for e in "abc"}
    w = {e: w_closed(*RATES[e], T) for e in "abc"}
    a, b, c = q["a"], q["b"], q["c"]
    exact = a * b + a * c + b * c - 2.0 * a * b * c
    np.testing.assert_allclose(tree.point_unavailability(T), exact, rtol=1e-13)
    # An event is critical when exactly one of the other two is down.
    critical = {
        "a": b + c - 2.0 * b * c,
        "b": a + c - 2.0 * a * c,
        "c": a + b - 2.0 * a * b,
    }
    np.testing.assert_allclose(
        tree.failure_frequency(T),
        sum(critical[e] * w[e] for e in "abc"),
        rtol=1e-12,
    )


def test_a_tested_event_averages_over_its_interval():
    rate, tau = 0.001, 100.0
    sensor = {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": tau},
    }
    tree = FaultTree({"top": ("or", ["sensor"])}, {"sensor": sensor})
    exact = 1.0 - -math.expm1(-rate * tau) / (rate * tau)
    assert tree.mean_unavailability(tau) == pytest.approx(exact, rel=1e-8)
    assert tree.mean_unavailability(10 * tau) == pytest.approx(exact, rel=1e-8)
    assert tree.mean_unavailability() == pytest.approx(exact, rel=1e-12)
    # Its failures are found at the tests, so it fails while it works.
    x = np.array([10.0, 150.0, 260.0])
    since = x % tau
    q = -np.expm1(-rate * since)
    np.testing.assert_allclose(tree.point_unavailability(x), q, rtol=1e-6)
    np.testing.assert_allclose(
        tree.failure_frequency(x), rate * (1.0 - q), rtol=1e-6
    )


def test_a_small_unavailability_keeps_its_digits():
    # Three events each down about 1e-6 of the time, all needed: Q near
    # 1e-18, where one less an availability would be 0.
    rate, repair = 1e-6, 1.0
    tree = FaultTree(
        {"top": ("and", ["a", "b", "c"])},
        {e: unit(rate, repair) for e in "abc"},
    )
    x = np.array([0.1, 5.0, 1000.0])
    q = q_closed(rate, repair, x)
    np.testing.assert_allclose(tree.point_unavailability(x), q**3, rtol=1e-12)
    assert tree.mean_unavailability() == pytest.approx(
        (rate / (rate + repair)) ** 3, rel=1e-12
    )
    np.testing.assert_allclose(
        tree.failure_frequency(x),
        3.0 * q**2 * w_closed(rate, repair, x),
        rtol=1e-12,
    )


# -- against the diagram, enumeration and simulation -----------------------


@pytest.mark.parametrize("name", ["repaired", "tested"])
def test_a_tree_agrees_with_its_repairable_diagram(name):
    tree = fault_tree_kinds()[name]
    rbd = tree.to_repairable_rbd()
    assert isinstance(rbd, RepairableRBD)
    x = np.array([0.0, 7.0, 49.0, 120.0, 600.0])
    np.testing.assert_allclose(
        tree.point_unavailability(x),
        rbd.point_unavailability(x),
        rtol=1e-10,
        atol=1e-300,
    )
    np.testing.assert_allclose(
        tree.mean_unavailability(x[1:]),
        rbd.mission_unavailability(x[1:]),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        tree.expected_failures(x), rbd.expected_failures(x), rtol=1e-10
    )
    assert tree.mean_unavailability() == pytest.approx(
        rbd.mean_unavailability(), rel=1e-10
    )
    assert tree.system_failure_frequency() == pytest.approx(
        rbd.system_failure_frequency(), rel=1e-10
    )


@pytest.mark.parametrize("name", ["repaired", "tested"])
def test_the_failure_frequency_integrates_to_the_expected_failures(name):
    tree = fault_tree_kinds()[name]
    # Simpson's rule on a fine grid, between the tests and their ends
    # (where the frequency jumps: a test takes "a" off line for 1.0).
    edges = [0.0, 50.0, 51.0, 80.0, 100.0]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = np.linspace(lo, hi, 2001)
        # Its value at a test is after the jump: take it just before.
        x[-1] = np.nextafter(hi, -np.inf)
        w = tree.failure_frequency(x)
        h = x[1] - x[0]
        inner = 4.0 * w[1:-1:2].sum() + 2.0 * w[2:-1:2].sum()
        total += h / 3.0 * (w[0] + w[-1] + inner)
    assert total == pytest.approx(tree.expected_failures(100.0), rel=1e-6)


def test_a_repeated_event_against_every_state():
    tree = fault_tree_kinds()["repeated, repaired"]
    x = np.array([0.0, 4.0, 30.0, 400.0])
    singles = {
        e: FaultTree({"top": ("or", [e])}, {e: tree.events[e]})
        for e in tree.events
    }
    q = {e: one.point_unavailability(x) for e, one in singles.items()}
    w = {e: one.failure_frequency(x) for e, one in singles.items()}
    names = list(tree.events)

    def top(state: dict) -> bool:
        return (state["s"] or state["a"]) and (state["s"] or state["b"])

    total = np.zeros(len(x))
    frequency = np.zeros(len(x))
    for values in itertools.product([False, True], repeat=len(names)):
        state = dict(zip(names, values))
        weight = np.ones(len(x))
        for e in names:
            weight = weight * (q[e] if state[e] else 1.0 - q[e])
        total += weight * top(state)
        # An event up in this state, critical here, starts the top event
        # when it fails: its intensity given it is up, times the others'.
        for e in names:
            if state[e] or top(state) or not top({**state, e: True}):
                continue
            frequency += weight / (1.0 - q[e]) * w[e]
    np.testing.assert_allclose(tree.point_unavailability(x), total, rtol=1e-12)
    np.testing.assert_allclose(
        tree.failure_frequency(x), frequency, rtol=1e-12
    )
    with pytest.raises(NotImplementedError, match="feeds more than one gate"):
        tree.to_repairable_rbd()


def test_a_tree_agrees_with_the_simulation_of_its_diagram():
    tree = fault_tree_kinds()["repaired"]
    end = 200.0
    runs = tree.to_repairable_rbd().simulate_timelines(
        end, mc_samples=20_000, seed=809
    )
    n = len(runs.system)
    down = ~runs.system.state(end)
    for estimate, exact in (
        (down, tree.point_unavailability(end)),
        (runs.system.failures, tree.expected_failures(end)),
        (runs.system.downtime / end, tree.mean_unavailability(end)),
    ):
        values = np.asarray(estimate, dtype=float)
        error = values.std(ddof=1) / math.sqrt(n)
        assert abs(values.mean() - exact) < 4.0 * error, (
            values.mean(),
            exact,
            error,
        )


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_the_values_scale_with_the_unit_of_time(factor):
    def tree(f):
        return FaultTree(
            {"top": ("or", ["pair", "c"]), "pair": ("and", ["a", "b"])},
            {
                "a": {
                    "reliability": W([100.0 * f, 2.0]),
                    "repairability": E([0.5 / f]),
                },
                "b": unit(0.01 / f, 0.25 / f),
                "c": {
                    "reliability": E([0.002 / f]),
                    "repairability": "instant",
                    "inspection": {"interval": 80.0 * f},
                },
            },
        )

    base, scaled = tree(1.0), tree(factor)
    x = np.array([5.0, 60.0, 300.0])
    np.testing.assert_allclose(
        scaled.point_unavailability(x * factor),
        base.point_unavailability(x),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        scaled.failure_frequency(x * factor) * factor,
        base.failure_frequency(x),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        scaled.expected_failures(x * factor),
        base.expected_failures(x),
        rtol=1e-9,
    )


# -- what is refused, and saving ---------------------------------------------


def test_what_a_tree_of_repaired_events_refuses():
    tree = fault_tree_kinds()["repaired"]
    for call in (tree.sf, tree.ff, tree.Hf):
        with pytest.raises(NotImplementedError, match="point_unavailability"):
            call(10.0)
    with pytest.raises(NotImplementedError, match="to_repairable_rbd"):
        tree.to_rbd()
    with pytest.raises(NotImplementedError, match="to_repairable_rbd"):
        tree.levers()
    with pytest.raises(NotImplementedError, match="exponential lives"):
        tree.to_open_psa()
    with pytest.raises(ValueError, match="0 or more"):
        tree.point_unavailability(-1.0)
    with pytest.raises(ValueError, match="t is required"):
        tree.point_unavailability(None)
    not_gate = fault_tree_kinds()["repaired, not gate"]
    assert 0.0 < not_gate.point_unavailability(50.0) < 1.0
    with pytest.raises(NotImplementedError, match="repair intensity"):
        not_gate.failure_frequency(50.0)
    with pytest.raises(NotImplementedError, match="NOT gate"):
        not_gate.to_repairable_rbd()
    plain = fault_tree_kinds()["lifetimes"]
    with pytest.raises(Refused, match="component specs"):
        plain.point_unavailability(10.0)
    with pytest.raises(Refused, match="component specs"):
        plain.to_repairable_rbd()


def test_what_a_tree_does_not_take_with_repaired_events():
    with pytest.raises(ValueError, match="to_repairable_rbd"):
        FaultTree(
            {"top": ("or", ["a"])},
            {"a": {**unit(0.1, 1.0), "corrective_cost": 5.0}},
        )
    with pytest.raises(NotImplementedError, match="component spec"):
        FaultTree({"top": ("or", ["a", "b"])}, {"a": unit(0.1, 1.0), "b": 0.1})
    with pytest.raises(NotImplementedError, match="ccf_groups"):
        FaultTree(
            {"top": ("and", ["a", "b"])},
            {"a": unit(0.1, 1.0), "b": unit(0.1, 1.0)},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
    with pytest.raises(NotImplementedError, match="Dynamic gates"):
        FaultTree(
            {"top": ("pand", ["a", "b"])},
            {"a": unit(0.1, 1.0), "b": unit(0.1, 1.0)},
        )
    with pytest.raises(ValueError, match="repairability"):
        FaultTree({"top": ("or", ["a"])}, {"a": {"reliability": E([0.1])}})


@pytest.mark.parametrize("name", ["repaired", "tested", "repeated, repaired"])
def test_a_tree_of_repaired_events_saves(name):
    tree = fault_tree_kinds()[name]
    again = FaultTree.from_json(tree.to_json())
    assert again.is_repairable
    x = np.array([3.0, 90.0])
    np.testing.assert_array_equal(
        again.point_unavailability(x), tree.point_unavailability(x)
    )
