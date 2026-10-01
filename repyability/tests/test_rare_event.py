"""Small failure probabilities by simulation (#115): each method against
the exact unreliability, down to 1e-8, and the rules that choose one."""

import numpy as np
import pytest
import surpyval as surv
from scipy.optimize import brentq

from repyability import (
    BetaFactor,
    CCFGroup,
    NonRepairableRBD,
    PerfectReliability,
    PerfectUnreliability,
    StandbyModel,
)

UNIT = surv.Weibull.from_params([1000.0, 1.5])


def two_of_three():
    return NonRepairableRBD(
        [("s", n) for n in "abc"] + [(n, "v") for n in "abc"] + [("v", "t")],
        {**{n: UNIT for n in "abc"}, "v": PerfectReliability},
        k={"v": 2},
    )


def bridge():
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "d"),
        ("c", "d"),
        ("b", "e"),
        ("c", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    return NonRepairableRBD(edges, {n: UNIT for n in "abcde"})


def pairs(n):
    edges, reliabilities, last = [], {}, "s"
    for i in range(n):
        a, b, joint = f"a{i}", f"b{i}", f"j{i}"
        edges += [(last, a), (last, b), (a, joint), (b, joint)]
        reliabilities.update({a: UNIT, b: UNIT, joint: PerfectReliability})
        last = joint
    edges.append((last, "t"))
    return NonRepairableRBD(edges, reliabilities)


def standby():
    # A cold-standby group of three units, in series with a pair.
    return NonRepairableRBD(
        [("s", "g"), ("g", "a"), ("g", "b"), ("a", "t"), ("b", "t")],
        {"g": StandbyModel([UNIT] * 3, k=1), "a": UNIT, "b": UNIT},
    )


def at(rbd, p):
    """The time by which the system fails with probability ``p``."""
    return brentq(
        lambda t: np.log(max(float(rbd.ff(t)), 1e-300)) - np.log(p),
        1e-6,
        1e6,
    )


def close(estimate, exact, spread=4.0):
    """Whether an estimate is within ``spread`` standard errors of the
    exact value (and its interval is honest about it)."""
    return abs(estimate.estimate - exact) <= spread * estimate.standard_error


@pytest.mark.parametrize(
    "method", ["plain", "latin_hypercube", "sobol", "cross_entropy", "subset"]
)
def test_every_method_finds_a_moderate_probability(method):
    rbd = bridge()
    x = at(rbd, 1e-2)
    estimate = rbd.unreliability_interval(
        x, method=method, relative_tolerance=0.2, seed=1
    )
    assert estimate.method == method
    assert close(estimate, float(rbd.ff(x)))
    assert estimate.lower <= estimate.estimate <= estimate.upper
    assert estimate.n_samples > 0


@pytest.mark.parametrize("make", [two_of_three, bridge, standby])
@pytest.mark.parametrize("p", [1e-6, 1e-8])
def test_cross_entropy_and_subset_reach_rare_failures(make, p):
    rbd = make()
    x = at(rbd, p)
    exact = float(rbd.ff(x))
    for method in ("cross_entropy", "subset"):
        estimate = rbd.unreliability_interval(
            x, method=method, relative_tolerance=0.2, seed=2
        )
        assert close(estimate, exact), (method, estimate, exact)
        assert estimate.estimate == pytest.approx(exact, rel=0.5)


def test_subset_finds_many_ways_of_failing():
    # Ten pairs in series fail in ten ways: more than the cross-entropy
    # method's mixture follows, and no trouble for subset simulation.
    rbd = pairs(10)
    x = at(rbd, 1e-6)
    estimate = rbd.unreliability_interval(x, method="subset", seed=3)
    assert close(estimate, float(rbd.ff(x)))


@pytest.mark.parametrize(
    "make, p, method",
    [
        (bridge, 1e-2, "plain"),
        (two_of_three, 1e-6, "cross_entropy"),
        (bridge, 1e-6, "cross_entropy"),
        (lambda: pairs(10), 1e-6, "subset"),
        (standby, 1e-6, "subset"),
    ],
)
def test_auto_follows_the_rules(make, p, method):
    rbd = make()
    x = at(rbd, p)
    estimate = rbd.unreliability_interval(x, relative_tolerance=0.2, seed=4)
    assert estimate.method == method
    assert close(estimate, float(rbd.ff(x)))


def test_a_seed_reproduces_the_estimate():
    rbd = two_of_three()
    x = at(rbd, 1e-5)
    a = rbd.unreliability_interval(x, seed=5)
    assert a == rbd.unreliability_interval(x, seed=5)
    assert a != rbd.unreliability_interval(x, seed=6)


def test_a_fixed_lifetime_needs_no_samples():
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "t")], {"a": PerfectUnreliability}
    )
    estimate = rbd.unreliability_interval(1.0, seed=1)
    assert (estimate.estimate, estimate.standard_error) == (1.0, 0.0)


def test_running_out_of_samples_warns():
    rbd = two_of_three()
    with pytest.warns(RuntimeWarning, match="max_samples"):
        estimate = rbd.unreliability_interval(
            at(rbd, 1e-6), method="plain", max_samples=50_000, seed=1
        )
    assert estimate.n_samples <= 50_000


def test_what_cannot_be_simulated_is_refused():
    grouped = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": UNIT, "b": UNIT},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="common-cause"):
        grouped.unreliability_interval(10.0)
    fitted = NonRepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": surv.KaplanMeier.fit([100.0, 250.0, 400.0, 700.0])},
    )
    with pytest.raises(NotImplementedError, match="own random numbers"):
        fitted.unreliability_interval(10.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(x=-1.0), "at least 0"),
        (dict(x=np.inf), "at least 0"),
        (dict(x="soon"), "at least 0"),
        (dict(x=1.0, method="importance"), "method must be"),
        (dict(x=1.0, relative_tolerance=0.0), "relative_tolerance"),
        (dict(x=1.0, confidence=1.0), "confidence"),
        (dict(x=1.0, max_samples=0), "max_samples"),
    ],
)
def test_the_arguments_are_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bridge().unreliability_interval(**kwargs)
