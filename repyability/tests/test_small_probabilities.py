"""Small failure probabilities keep their precision (#148): ``ff``,
``unreliability``, ``Hf`` and ``df`` of a ``NonRepairableRBD``, and
``mean_unavailability`` of a ``RepairableRBD``, are worked out in their own
right rather than as one less a probability near 1, so they keep their full
relative precision down to 1e-15 and below. Each is checked against a
closed form evaluated exactly (in rational or 60-digit arithmetic)."""

from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    BetaFactor,
    CCFGroup,
    NonRepairable,
    NonRepairableRBD,
    RegressionNode,
    RepairableRBD,
    RepeatedNode,
)

#: Relative error allowed against an exact value: a few roundings.
RTOL = 1e-14

UNIT = surv.Weibull.from_params([1000.0, 1.5])
E = surv.Exponential.from_params
FIXED = surv.FixedEventProbability.from_params
EXACT = surv.ExactEventTime.from_params

#: Component failure probabilities, from 1e-15 to 1e-3 (and a large one).
QS = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 0.3]


def approx(value, rel=RTOL):
    """``pytest.approx`` to a relative tolerance only: its default absolute
    tolerance, 1e-12, would pass any two small probabilities."""
    return pytest.approx(value, rel=rel, abs=0.0)


def series(models):
    names = list(models)
    edges = [("s", names[0])]
    edges += list(zip(names[:-1], names[1:]))
    edges.append((names[-1], "t"))
    return NonRepairableRBD(edges, models)


def parallel(models):
    edges = [("s", n) for n in models] + [(n, "t") for n in models]
    return NonRepairableRBD(edges, models)


def two_of_three(models):
    edges = [("s", n) for n in models] + [(n, "t") for n in models]
    return NonRepairableRBD(edges, models, k={"t": 2})


def bridge(models):
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
    return NonRepairableRBD(edges, models)


# Each structure of identical components, and its unreliability as an exact
# function of the components' failure probability q.
STRUCTURES = {
    "series": (series, "abc", lambda q: 1 - (1 - q) ** 3),
    "parallel": (parallel, "abc", lambda q: q**3),
    "two_of_three": (two_of_three, "abc", lambda q: q**2 * (3 - 2 * q)),
    "bridge": (
        bridge,
        "abcde",
        lambda q: 2 * q**2 + 2 * q**3 - 5 * q**4 + 2 * q**5,
    ),
}


def exact(form, q) -> float:
    """``form`` evaluated exactly, at the float ``q`` taken as exact."""
    return float(form(Fraction(float(q))))


@pytest.mark.parametrize("name", STRUCTURES)
@pytest.mark.parametrize("q", QS)
def test_fixed_probabilities(name, q):
    build, nodes, form = STRUCTURES[name]
    rbd = build({n: FIXED(q) for n in nodes})
    assert rbd.ff() == approx(exact(form, q))
    assert rbd.unreliability() == rbd.ff()


@pytest.mark.parametrize("name", STRUCTURES)
def test_arrays_of_times(name):
    # The components' failure probabilities run from about 1e-15 to 0.6.
    build, nodes, form = STRUCTURES[name]
    rbd = build({n: UNIT for n in nodes})
    x = np.array([1e-7, 1e-5, 1e-3, 0.1, 10.0, 1000.0])
    got = rbd.ff(x)
    assert isinstance(got, np.ndarray) and got.shape == x.shape
    want = [exact(form, q) for q in UNIT.ff(x)]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)
    np.testing.assert_array_equal(rbd.unreliability(x), got)
    # Ordinary values agree with one less the reliability.
    np.testing.assert_allclose(got, 1 - rbd.sf(x), rtol=0.0, atol=1e-15)


def test_one_less_the_reliability_loses_what_ff_keeps():
    rbd = two_of_three({n: UNIT for n in "abc"})
    x = 0.01
    want = exact(STRUCTURES["two_of_three"][2], UNIT.ff(x))  # 3e-15
    assert rbd.ff(x) == approx(want)
    assert 1 - rbd.sf(x) != approx(want, 1e-5)


def test_conditioning_on_nodes():
    q = 1e-9
    rbd = two_of_three({n: FIXED(q) for n in "abc"})
    Q = Fraction(q)
    # With a failed, b or c failing fails the system; with a working, both.
    assert rbd.ff(broken_nodes=["a"]) == approx(float(1 - (1 - Q) ** 2))
    assert rbd.ff(working_nodes=["a"]) == approx(float(Q**2))
    assert rbd.ff(working_nodes=["a", "b"]) == 0.0
    assert rbd.ff(broken_nodes=["a", "b"]) == 1.0
    with pytest.raises(ValueError):
        rbd.ff(working_nodes=["a"], broken_nodes=["a"])
    with pytest.raises(ValueError):
        rbd.ff(method="x")


@pytest.mark.parametrize("q", QS)
def test_common_cause_groups(q):
    # A parallel pair fails if the shared cause strikes (beta * Q), or if
    # not, both fail independently ((1 - beta) * Q each).
    beta = 0.1
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": FIXED(q), "b": FIXED(q)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    Q, B = Fraction(q), Fraction(beta)
    want = B * Q + (1 - B * Q) * ((1 - B) * Q) ** 2
    assert rbd.ff() == approx(float(want))
    assert rbd.ff() == pytest.approx(1 - rbd.sf(), rel=0.0, abs=1e-15)


def test_common_cause_groups_over_time():
    beta = 0.05
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": UNIT, "b": UNIT},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    x = np.array([1e-6, 1e-3, 1.0, 100.0])
    B = Fraction(beta)
    want = [
        float(B * Q + (1 - B * Q) * ((1 - B) * Q) ** 2)
        for Q in map(Fraction, UNIT.ff(x))
    ]
    np.testing.assert_allclose(rbd.ff(x), want, rtol=RTOL, atol=0.0)


def test_cumulative_hazard_and_density():
    rbd = two_of_three({n: UNIT for n in "abc"})
    for x in [1e-4, 0.01, 1.0, 1000.0, 5000.0]:
        p, q = float(UNIT.sf(x)), float(UNIT.ff(x))
        f = float(UNIT.df(x))
        F = q * q * (3 - 2 * q)
        # -ln R, from F while it is small and from R = p**2 (1 + 2q) once
        # the system has almost surely failed.
        H = -np.log1p(-F) if F < 0.5 else -np.log(p * p * (1 + 2 * q))
        assert rbd.Hf(x) == approx(H, 1e-13)
        # The density, by a finite difference (of step 1e-6 below x = 1, so
        # from x = 0.01), to its truncation error.
        if x >= 0.01:
            density = 6 * q * p * f
            assert rbd.df(x) == approx(density, 1e-6)
            assert rbd.hf(x) == approx(density / (p * p * (1 + 2 * q)), 1e-6)
    x = np.array([1e-4, 5000.0])
    np.testing.assert_allclose(
        rbd.Hf(x), [rbd.Hf(1e-4), rbd.Hf(5000.0)], rtol=1e-15
    )


def test_repeated_node_in_series():
    node = RepeatedNode(UNIT, 3, "series")
    x = np.array([1e-7, 1e-3, 10.0, 1000.0])
    want = [float(1 - (1 - Fraction(q)) ** 3) for q in UNIT.ff(x)]
    np.testing.assert_allclose(node.ff(x), want, rtol=RTOL, atol=0.0)
    rbd = NonRepairableRBD([("s", "r"), ("r", "t")], {"r": node})
    np.testing.assert_allclose(rbd.ff(x), want, rtol=RTOL, atol=0.0)


@pytest.fixture(scope="module")
def regression_model():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 1))
    x = rng.weibull(1.8, size=400) * 80.0 * np.exp(-0.4 * Z[:, 0]) + 1e-3
    return surv.WeibullAFT.fit(x, Z=Z)


def test_regression_node(regression_model):
    from surpyval.univariate.regression import StepSchedule

    x = np.array([1e-6, 1e-3, 1.0, 50.0])
    fixed = RegressionNode(regression_model, covariates=[0.5])
    np.testing.assert_array_equal(
        fixed.ff(x),
        regression_model.ff(x, np.repeat([[0.5]], len(x), axis=0)),
    )
    schedule = StepSchedule.from_changepoints([0, 0.5], [[0.0], [0.8]])
    varying = RegressionNode(regression_model, schedule=schedule)
    H = regression_model.Hf_tvc(x, schedule)
    np.testing.assert_allclose(varying.ff(x), -np.expm1(-H), rtol=RTOL)
    for node in (fixed, varying):
        np.testing.assert_allclose(
            node.ff(x), 1 - node.sf(x), rtol=0.0, atol=1e-15
        )


@pytest.mark.parametrize("q", QS)
def test_importance_measures(q):
    # A 2-out-of-3 system of identical components, failing with
    # probability F = q**2 (3 - 2q): each one is critical when exactly one
    # of the other two works.
    rbd = two_of_three({n: FIXED(q) for n in "abc"})
    Q = Fraction(q)
    P = 1 - Q
    F = Q**2 * (3 - 2 * Q)
    birnbaum = 2 * P * Q
    want = {
        "birnbaum_importance": birnbaum,
        "improvement_potential": birnbaum * Q,
        "criticality_importance": birnbaum * Q / F,
        "fussell_vesely": 2 * Q**2 / F,
        "risk_achievement_worth": (1 - P**2) / F,
        "risk_reduction_worth": F / Q**2,
    }
    for name, value in want.items():
        assert getattr(rbd, name)()["a"] == approx(float(value)), name
    success = rbd.criticality_importance(kind="success")["a"]
    assert success == approx(float(birnbaum * P / (1 - F)))


@pytest.mark.parametrize("q", QS)
def test_birnbaum_importance_in_a_bridge(q):
    # The bridge's core is meshed: its importance comes through the Shannon
    # decomposition, not the modules' closed forms. With identical
    # components, the bridge element's is 2 q**2 p**2, and the five add up
    # to dR/dp, R = 2p**2 + 2p**3 - 5p**4 + 2p**5.
    rbd = bridge({n: FIXED(q) for n in "abcde"})
    Q = Fraction(q)
    P = 1 - Q
    middle = 2 * Q**2 * P**2
    total = 4 * P + 6 * P**2 - 20 * P**3 + 10 * P**4
    importance = rbd.birnbaum_importance()
    assert importance["c"] == approx(float(middle))
    for node in "abde":
        assert importance[node] == approx(float((total - middle) / 4))


def test_importance_over_time():
    rbd = two_of_three({n: UNIT for n in "abc"})
    x = np.array([1e-7, 1e-3, 10.0, 1000.0])
    want = [
        float(2 * Fraction(p) * Fraction(q))
        for p, q in zip(UNIT.sf(x), UNIT.ff(x))
    ]
    got = rbd.birnbaum_importance(x)["a"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)
    # Held working, "a" leaves "b" critical when "c" has failed.
    held = rbd.birnbaum_importance(x, working_nodes=["a"])["b"]
    np.testing.assert_allclose(held, UNIT.ff(x), rtol=RTOL, atol=0.0)


# -- RepairableRBD.mean_unavailability --------------------------------------


def repairable_pair(component, **kwargs):
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": component, "b": component},
        **kwargs,
    )


@pytest.mark.parametrize("mttf", [1e2, 1e5, 1e8, 1e11])
def test_a_parallel_pair_of_repairable_units(mttf):
    unit = {"reliability": E([1 / mttf]), "repairability": EXACT([1.0])}
    rbd = repairable_pair(unit)
    U = 1 / (Fraction(1 / (1 / mttf)) + 1)
    assert rbd.mean_unavailability() == approx(float(U**2), 1e-13)
    assert rbd.mean_unavailability(broken_nodes=["a"]) == approx(
        float(U), 1e-13
    )
    assert rbd.mean_unavailability(working_nodes=["a"]) == 0.0
    # Ordinary values agree with one less the availability.
    assert rbd.mean_unavailability() == pytest.approx(
        1 - rbd.mean_availability(), rel=0.0, abs=1e-15
    )


def test_a_two_of_three_of_repairable_units():
    unit = {"reliability": E([1e-7]), "repairability": EXACT([2.0])}
    rbd = RepairableRBD(
        [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
        {n: unit for n in "abc"},
        k={"t": 2},
    )
    U = 2 / (Fraction(1 / 1e-7) + 2)
    assert rbd.mean_unavailability() == approx(
        float(U**2 * (3 - 2 * U)), 1e-13
    )


def test_the_unit_unavailability_itself():
    unit = NonRepairable(E([1e-9]), EXACT([1.0]))
    want = float(1 / (Fraction(1 / 1e-9) + 1))
    assert unit.mean_unavailability() == approx(want)
    assert 1 - unit.mean_availability() != approx(want, 1e-9)
    # A unit some of whose units never fail ends up for good.
    cured = NonRepairable(E([0.01], p=0.9), E([0.5]))
    assert cured.mean_unavailability() == 0.0


@pytest.mark.parametrize("rate", [1e-5, 1e-8, 1e-11])
def test_a_pair_tested_together(rate):
    # PFDavg of a 1oo2 tested every tau: the average over the interval of
    # (1 - exp(-rate * t)) ** 2.
    tau = 8760.0
    component = {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": tau},
    }
    rbd = repairable_pair(component)
    getcontext().prec = 60
    x = Decimal(rate) * Decimal(tau)
    want = 1 - 2 * (1 - (-x).exp()) / x + (1 - (-2 * x).exp()) / (2 * x)
    assert rbd.mean_unavailability() == approx(float(want), 1e-13)
    alone = RepairableRBD([("s", "a"), ("a", "t")], {"a": component})
    assert alone.mean_unavailability() == approx(
        float(1 - (1 - (-x).exp()) / x), 1e-13
    )


def test_one_repair_crew():
    # The pair's chain: both up; one down (repaired); both down (one waits).
    lam, mu = 1e-5, 1.0
    unit = {"reliability": E([lam]), "repairability": E([mu])}
    rbd = repairable_pair(unit, repair_crews=1)
    r = Fraction(lam) / Fraction(mu)
    want = 2 * r**2 / (1 + 2 * r + 2 * r**2)
    assert rbd.mean_unavailability() == approx(float(want), 1e-12)
    assert rbd.mean_unavailability() == pytest.approx(
        1 - rbd.mean_availability(), rel=0.0, abs=1e-15
    )


def test_a_standby_group():
    # Two cold-standby units, one needed, each repaired at once: down while
    # both are, with probability (r**2 / 2) / (1 + r + r**2 / 2).
    lam, mu = 1e-5, 1.0
    group = {
        "reliability": E([lam]),
        "repairability": E([mu]),
        "standby": {"units": 2},
    }
    rbd = RepairableRBD([("s", "g"), ("g", "t")], {"g": group})
    r = Fraction(lam) / Fraction(mu)
    want = (r**2 / 2) / (1 + r + r**2 / 2)
    assert rbd.mean_unavailability() == approx(float(want), 1e-12)


def test_age_replacement():
    life = surv.Weibull.from_params([1000.0, 3.0])
    unit = {
        "reliability": life,
        "repairability": EXACT([1.0]),
        "preventive": {"interval": 1.0},
    }
    rbd = RepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
    up = NonRepairable(life).avg_replacement_time(1.0)
    fails = float(life.ff(1.0))  # 1e-9
    assert rbd.mean_unavailability() == approx(fails / (up + fails), 1e-12)


def test_a_nested_diagram():
    unit = {"reliability": E([1e-6]), "repairability": EXACT([1.0])}
    outer = RepairableRBD(
        [("s", "pair"), ("pair", "c"), ("c", "t")],
        {"pair": repairable_pair(unit), "c": unit},
    )
    U = 1 / (Fraction(1 / 1e-6) + 1)
    want = 1 - (1 - U**2) * (1 - U)
    assert outer.mean_unavailability() == approx(float(want), 1e-13)


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"inspection": {"interval": 50.0}},
        {"preventive": {"interval": 30.0, "policy": "block"}},
    ],
    ids=["plain", "inspected", "block"],
)
def test_ordinary_values_agree_with_the_availability(extra):
    unit = {
        "reliability": surv.Weibull.from_params([100.0, 1.0]),
        "repairability": "instant" if "inspection" in extra else E([0.5]),
        **extra,
    }
    other = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {"a": unit, "b": unit, "c": other},
    )
    for kwargs in [{}, {"broken_nodes": ["a"]}, {"working_nodes": ["c"]}]:
        assert rbd.mean_unavailability(**kwargs) == pytest.approx(
            1 - rbd.mean_availability(**kwargs), rel=0.0, abs=1e-12
        )
    with pytest.raises(ValueError):
        rbd.mean_unavailability(method="x")


# -- Birnbaum importance and the failure frequency of a RepairableRBD -------


@pytest.mark.parametrize("mttf", [1e2, 1e5, 1e8, 1e11])
def test_the_failure_frequency_of_a_parallel_pair(mttf):
    # Each unit is down a fraction U = 1 / (MTTF + 1) of the time, and fails
    # 1 / (MTTF + 1) times per unit time; the pair fails when one fails
    # while the other is down: 2 U**2 per unit time, for U**2 of the time.
    unit = {"reliability": E([1 / mttf]), "repairability": EXACT([1.0])}
    rbd = repairable_pair(unit)
    U = 1 / (Fraction(1 / (1 / mttf)) + 1)
    frequency = 2 * U**2
    assert rbd.birnbaum_importance()["a"] == approx(float(U), 1e-13)
    assert rbd.improvement_potential()["a"] == approx(float(U**2), 1e-13)
    assert rbd.risk_achievement_worth()["a"] == approx(float(1 / U), 1e-13)
    assert rbd.system_failure_frequency() == approx(float(frequency), 1e-13)
    assert rbd.mean_time_between_failures() == approx(
        float(1 / frequency), 1e-13
    )
    assert rbd.mean_up_time() == approx(float((1 - U**2) / frequency), 1e-13)
    assert rbd.mean_down_time() == approx(0.5, 1e-13)


def test_the_failure_frequency_with_one_repair_crew():
    # The pair fails when the working unit fails while the other is under
    # repair, and is down until that repair is done.
    lam, mu = 1e-6, 1.0
    unit = {"reliability": E([lam]), "repairability": E([mu])}
    rbd = repairable_pair(unit, repair_crews=1)
    r = Fraction(lam) / Fraction(mu)
    frequency = 2 * r * Fraction(lam) / (1 + 2 * r + 2 * r**2)
    assert rbd.system_failure_frequency() == approx(float(frequency), 1e-12)
    assert rbd.mean_down_time() == approx(1 / mu, 1e-12)


@pytest.mark.parametrize("rate", [1e-5, 1e-8, 1e-11])
def test_the_failure_frequency_of_a_pair_tested_together(rate):
    # Each unit fails at the constant rate while it is up, and the pair
    # when one does while the other has failed since the last test:
    # (1 / tau) * integral of 2 * rate * exp(-rate t) (1 - exp(-rate t)),
    # which is (1 - exp(-rate tau))**2 / tau.
    tau = 8760.0
    component = {
        "reliability": E([rate]),
        "repairability": "instant",
        "inspection": {"interval": tau},
    }
    rbd = repairable_pair(component)
    getcontext().prec = 60
    x = Decimal(rate) * Decimal(tau)
    want = (1 - (-x).exp()) ** 2 / Decimal(tau)
    assert rbd.system_failure_frequency() == approx(float(want), 1e-13)


def test_the_failure_frequency_under_age_replacement():
    # A unit replaced at age 1 fails first with probability F(1) = 1e-9.
    life = surv.Weibull.from_params([1000.0, 3.0])
    unit = {
        "reliability": life,
        "repairability": EXACT([1.0]),
        "preventive": {"interval": 1.0},
    }
    rbd = RepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
    fails = -np.expm1(-((1.0 / 1000.0) ** 3))
    cycle = NonRepairable(life).avg_replacement_time(1.0) + fails * 1.0
    assert rbd.system_failure_frequency() == approx(fails / cycle, 1e-12)


@pytest.mark.parametrize("mttf", [1e3, 1e9, 1e13])
def test_planned_outages_at_block_times(mttf):
    # "a" is replaced every 100 hours, taking 2: the pair goes down then
    # only if "b" is down, a fraction U of the time.
    a = {
        "reliability": surv.Weibull.from_params([1e6, 2.0]),
        "repairability": EXACT([1.0]),
        "preventive": {
            "interval": 100.0,
            "policy": "block",
            "duration": EXACT([2.0]),
        },
    }
    b = {"reliability": E([1 / mttf]), "repairability": EXACT([1.0])}
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")], {"a": a, "b": b}
    )
    U = 1 / (Fraction(1 / (1 / mttf)) + 1)
    up_before = Fraction(rbd._block_cycle("a").before)
    planned = rbd._outage_frequencies()[1]
    assert planned == approx(float(up_before * U / 100), 1e-12)
