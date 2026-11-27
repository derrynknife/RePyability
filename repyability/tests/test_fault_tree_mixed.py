"""Fault trees mixing repaired (or tested) events with fixed
probabilities and lives that are not repaired, and common-cause groups
over repaired events: against closed forms, numerical integrals, the
repairable diagram of the same system and seeded simulations."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import EventTree, FaultTree, NodeState
from repyability.rbd.ccf import BetaFactor, CCFGroup
from repyability.tests.catalogue import fault_tree_kinds

E = surv.Exponential.from_params
W = surv.Weibull.from_params

T = np.array([0.0, 0.5, 3.0, 20.0, 150.0, 2000.0])


def unit(rate, repair):
    return {"reliability": E([rate]), "repairability": E([repair])}


def q_closed(rate, repair, t):
    """A unit's probability of being down at ``t``, new at 0."""
    return rate / (rate + repair) * -np.expm1(-(rate + repair) * t)


def failures_closed(rate, repair, t):
    """A unit's expected failures in ``[0, t)``."""
    s = rate + repair
    return rate * repair / s * t + (rate / s) ** 2 * -np.expm1(-s * t)


def mean_q_closed(rate, repair, t):
    """A unit's average probability of being down over ``[0, t]``."""
    s = rate + repair
    return rate / s * (1.0 - -np.expm1(-s * t) / (s * t))


# -- closed forms ------------------------------------------------------------


def test_an_or_of_a_repaired_event_and_a_probability():
    rate, repair, p = 0.01, 0.5, 0.03
    tree = FaultTree(
        {"top": ("or", ["a", "h"])}, {"a": unit(rate, repair), "h": p}
    )
    q = q_closed(rate, repair, T)
    np.testing.assert_allclose(
        tree.point_unavailability(T), p + (1.0 - p) * q, rtol=1e-12
    )
    # A fixed probability never starts the top event over time.
    np.testing.assert_allclose(
        tree.failure_frequency(T),
        (1.0 - p) * rate * (1.0 - q),
        rtol=1e-12,
    )
    x = T[1:]
    np.testing.assert_allclose(
        tree.mean_unavailability(x),
        p + (1.0 - p) * mean_q_closed(rate, repair, x),
        rtol=1e-12,
    )
    # The event's expected failures come from its renewal equation on a
    # grid, to about 1e-7 (see RepairableRBD.expected_failures); the
    # probability leaves them as the event's alone gives them, times 1 - p.
    np.testing.assert_allclose(
        tree.expected_failures(x),
        (1.0 - p) * failures_closed(rate, repair, x),
        atol=1e-7,
    )
    alone = FaultTree({"top": ("or", ["a"])}, {"a": unit(rate, repair)})
    np.testing.assert_allclose(
        tree.expected_failures(x),
        (1.0 - p) * alone.expected_failures(x),
        rtol=1e-12,
    )
    steady = rate / (rate + repair)
    assert tree.mean_unavailability() == pytest.approx(
        p + (1.0 - p) * steady, rel=1e-12
    )
    assert tree.system_failure_frequency() == pytest.approx(
        (1.0 - p) * rate * (1.0 - steady), rel=1e-12
    )


def test_an_and_with_a_weibull_life_that_is_not_repaired():
    rate, repair = 0.01, 0.5
    life = W([200.0, 2.0])
    tree = FaultTree(
        {"top": ("and", ["a", "b"])}, {"a": unit(rate, repair), "b": life}
    )
    q = q_closed(rate, repair, T)
    F, f = life.ff(T), life.df(T)
    np.testing.assert_allclose(
        tree.point_unavailability(T), q * F, rtol=1e-12, atol=1e-300
    )
    # Either fails while the other is down: a's failures while b is down,
    # and b's (once) while a is.
    np.testing.assert_allclose(
        tree.failure_frequency(T),
        rate * (1.0 - q) * F + q * f,
        rtol=1e-12,
        atol=1e-300,
    )

    def point(s):
        return q_closed(rate, repair, s) * float(life.ff(s))

    def frequency(s):
        down = q_closed(rate, repair, s)
        return rate * (1.0 - down) * float(life.ff(s)) + down * float(
            life.df(s)
        )

    for end in (3.0, 20.0, 150.0, 2000.0):
        mean = integrate.quad(point, 0.0, end, epsabs=0.0, epsrel=1e-13)[0]
        count = integrate.quad(
            frequency, 0.0, end, epsabs=0.0, epsrel=1e-13, limit=200
        )[0]
        assert tree.mean_unavailability(end) == pytest.approx(
            mean / end, rel=1e-9
        )
        assert tree.expected_failures(end) == pytest.approx(count, rel=1e-7)
    # Once b has failed it stays: no long run.
    for call in (tree.mean_unavailability, tree.system_failure_frequency):
        with pytest.raises(NotImplementedError, match="no long run"):
            call()


def test_a_small_unavailability_with_a_probability_keeps_its_digits():
    tree = FaultTree(
        {"top": ("and", ["a", "h"])}, {"a": unit(1e-9, 1.0), "h": 1e-6}
    )
    x = np.array([1e-3, 1.0, 100.0])
    np.testing.assert_allclose(
        tree.point_unavailability(x),
        1e-6 * q_closed(1e-9, 1.0, x),
        rtol=1e-12,
    )


def test_a_fixed_probability_given_as_a_model():
    fixed = surv.FixedEventProbability.from_params([0.03])
    by_model = FaultTree(
        {"top": ("or", ["a", "h"])}, {"a": unit(0.01, 0.5), "h": fixed}
    )
    by_number = FaultTree(
        {"top": ("or", ["a", "h"])}, {"a": unit(0.01, 0.5), "h": 0.03}
    )
    np.testing.assert_array_equal(
        by_model.point_unavailability(T), by_number.point_unavailability(T)
    )
    assert by_model.mean_unavailability() == by_number.mean_unavailability()


# -- against the diagram -----------------------------------------------------


def test_a_fixed_probability_against_a_unit_in_its_steady_state():
    """A fixed probability is down as a repaired unit started in its
    steady state is, at every time: the same system as a diagram."""
    lam, mu = 0.002, 0.1
    tree = FaultTree(
        {"top": ("or", ["pair", "c"]), "pair": ("and", ["a", "b"])},
        {
            "a": {"reliability": W([100.0, 2.0]), "repairability": E([0.5])},
            "b": unit(0.01, 0.25),
            "c": lam / (lam + mu),
        },
    )
    drawn = FaultTree(
        {"top": ("or", ["pair", "c"]), "pair": ("and", ["a", "b"])},
        {**{e: tree.events[e] for e in "ab"}, "c": unit(lam, mu)},
    ).to_repairable_rbd()
    state = {"c": NodeState(stationary=True)}
    x = np.array([0.0, 7.0, 49.0, 120.0, 600.0])
    np.testing.assert_allclose(
        tree.point_unavailability(x),
        drawn.point_unavailability(x, state=state),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        tree.mean_unavailability(x[1:]),
        drawn.mission_unavailability(x[1:], state=state),
        rtol=1e-12,
    )
    assert tree.mean_unavailability() == pytest.approx(
        drawn.mean_unavailability(), rel=1e-12
    )


def common_cause(tested: bool = False):
    """Two alike repaired (or tested) events in a common-cause group under
    an AND gate, ORed with a third."""
    member = unit(0.002, 0.1)
    if tested:
        member = {
            "reliability": E([0.002]),
            "repairability": "instant",
            "inspection": {"interval": 80.0},
        }
    return FaultTree(
        {"top": ("or", ["pair", "c"]), "pair": ("and", ["a", "b"])},
        {"a": member, "b": member, "c": unit(0.0005, 0.25)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    )


@pytest.mark.parametrize("tested", [False, True])
def test_common_cause_against_the_diagram_with_the_same_groups(tested):
    tree = common_cause(tested)
    rbd = tree.to_repairable_rbd()
    assert [list(g.members) for g in rbd.ccf_groups] == [["a", "b"]]
    x = np.array([0.0, 7.0, 49.0, 120.0, 600.0])
    np.testing.assert_allclose(
        tree.point_unavailability(x),
        rbd.point_unavailability(x),
        rtol=1e-12,
        atol=1e-300,
    )
    np.testing.assert_allclose(
        tree.top_event_probability(x), rbd.point_unavailability(x), rtol=1e-12
    )
    np.testing.assert_allclose(
        tree.mean_unavailability(x[1:]),
        rbd.mission_unavailability(x[1:]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        tree.expected_failures(x[1:]),
        rbd.expected_failures(x[1:]),
        rtol=1e-12,
    )
    assert tree.mean_unavailability() == pytest.approx(
        rbd.mean_unavailability(), rel=1e-12
    )
    assert tree.system_failure_frequency() == pytest.approx(
        rbd.system_failure_frequency(), rel=1e-12
    )
    # The group's causes count: the independent pair is down less.
    alone = FaultTree(
        {"top": ("or", ["pair", "c"]), "pair": ("and", ["a", "b"])},
        dict(tree.events),
    )
    assert tree.mean_unavailability() > alone.mean_unavailability()


def test_common_cause_beside_a_probability():
    tree = common_cause()
    gates = {"top": ("or", ["pair", "c", "h"]), "pair": ("and", ["a", "b"])}
    p = 1e-3
    with_h = FaultTree(
        gates, {**tree.events, "h": p}, ccf_groups=tree.ccf_groups
    )
    x = np.array([5.0, 60.0, 400.0])
    np.testing.assert_allclose(
        with_h.point_unavailability(x),
        p + (1.0 - p) * tree.point_unavailability(x),
        rtol=1e-12,
    )
    assert with_h.mean_unavailability() == pytest.approx(
        p + (1.0 - p) * tree.mean_unavailability(), rel=1e-12
    )
    assert with_h.system_failure_frequency() == pytest.approx(
        (1.0 - p) * tree.system_failure_frequency(), rel=1e-12
    )


# -- seeded simulations ------------------------------------------------------


def _within(values, exact) -> None:
    values = np.asarray(values, dtype=float)
    error = values.std(ddof=1) / math.sqrt(len(values))
    assert abs(values.mean() - exact) < 4.0 * error, (
        values.mean(),
        exact,
        error,
    )


def test_a_mixed_tree_against_a_seeded_simulation():
    """Catalogue's tree of a repaired event ``a`` and a life ``b`` that is
    not repaired under an AND, ORed with a probability ``c``: each
    simulation draws ``a``'s ups and downs, ``b``'s life and whether ``c``
    is down, and counts the times ``a AND b`` starts with ``c`` up."""
    tree = fault_tree_kinds()["repaired, with a probability and a life"]
    a = tree.events["a"]
    life_b, p = tree.events["b"], tree.events["c"]
    end, n = 300.0, 20_000
    rng = np.random.default_rng(831)
    down_at_end = np.empty(n, dtype=bool)
    starts = np.zeros(n)
    downtime = np.zeros(n)

    def draw(model) -> float:
        return float(np.ravel(model.qf(rng.random()))[0])

    for k in range(n):
        tb = draw(life_b)
        c = rng.random() < p
        # a's down periods before the end.
        clock, periods = 0.0, []
        while clock < end:
            clock += draw(a["reliability"])
            if clock >= end:
                break
            back = clock + draw(a["repairability"])
            periods.append((clock, back))
            clock = back
        a_down_end = bool(periods) and periods[-1][1] > end
        down_at_end[k] = c or (a_down_end and tb < end)
        if c:
            downtime[k] = end
            continue
        for start, stop in periods:
            stop = min(stop, end)
            # Down together from max(start, tb) to stop.
            if tb < stop:
                downtime[k] += stop - max(start, tb)
                starts[k] += 1.0
    _within(down_at_end, tree.point_unavailability(end))
    _within(starts, tree.expected_failures(end))
    _within(downtime / end, tree.mean_unavailability(end))


def test_common_cause_against_a_seeded_simulation_of_its_diagram():
    tree = common_cause()
    end = 300.0
    runs = tree.to_repairable_rbd().simulate_timelines(
        end, mc_samples=20_000, seed=831
    )
    _within(~runs.system.state(end), tree.point_unavailability(end))
    _within(runs.system.failures, tree.expected_failures(end))
    _within(runs.system.downtime / end, tree.mean_unavailability(end))


# -- what is refused, saving and the other forms -----------------------------


def test_what_a_mixed_or_grouped_tree_refuses():
    mixed = fault_tree_kinds()["repaired, with a probability"]
    with pytest.raises(NotImplementedError, match="components are repaired"):
        mixed.to_repairable_rbd()
    grouped = common_cause()
    with pytest.raises(NotImplementedError, match="split their failure"):
        grouped.to_open_psa()
    with pytest.raises(NotImplementedError, match="split their failure"):
        grouped.to_open_psa(t=10.0)
    weibull = {"reliability": W([100.0, 2.0]), "repairability": E([0.5])}
    worn = FaultTree(
        {"top": ("and", ["a", "b"])},
        {"a": weibull, "b": weibull},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    )
    for name in ("point_unavailability", "failure_frequency"):
        with pytest.raises(NotImplementedError, match="exponential lives"):
            getattr(worn, name)(10.0)
    with pytest.raises(NotImplementedError, match="exponential lives"):
        worn.birnbaum_importance(10.0)
    routes = grouped.analysis_routes()
    for name in (
        "failure_frequency",
        "birnbaum_importance",
        "fussell_vesely",
        "ranked_cut_sets",
    ):
        assert routes[name].route != "refused"
        assert worn.analysis_routes()[name].route == "refused"


# -- common cause: failure frequency and the measures at a time ---------------

MEASURES = (
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "fussell_vesely",
)


@pytest.mark.parametrize("tested", [False, True])
def test_common_cause_measures_against_the_diagram(tested):
    """Each measure at times, against the repairable diagram's at the same
    times, which conditions on the groups' chains the same way."""
    tree = common_cause(tested)
    rbd = tree.to_repairable_rbd()
    x = np.array([7.0, 49.0, 120.0, 600.0])
    for name in MEASURES:
        mine, theirs = getattr(tree, name)(x), getattr(rbd, name)(x=x)
        assert set(mine) == set(theirs)
        for event in mine:
            np.testing.assert_allclose(
                mine[event], theirs[event], rtol=1e-12, err_msg=name
            )
    mine, theirs = tree.criticality_importance(x, kind="success"), (
        rbd.criticality_importance(x=x, kind="success")
    )
    for event in mine:
        np.testing.assert_allclose(mine[event], theirs[event], rtol=1e-12)
    mine, theirs = tree.fussell_vesely(x, fv_type="p"), rbd.fussell_vesely(
        x=x, fv_type="p"
    )
    for event in mine:
        np.testing.assert_allclose(mine[event], theirs[event], rtol=1e-12)
    joint, drawn = tree.joint_importance(x), rbd.joint_importance(x=x)
    for pair in [("a", "b"), ("a", "c"), ("b", "c")]:
        np.testing.assert_allclose(joint[pair], drawn[pair], rtol=1e-12)
    # A number gives floats, as without groups.
    assert isinstance(tree.birnbaum_importance(49.0)["a"], float)


def pair_chain(rate, repair, beta, t):
    """The joint states of two alike units in a beta-factor group (of the
    rate), each repaired on its own, new at 0: ``(both up, one given one
    down, both down)`` at each time, from the chain's matrix exponential
    (states: up-up, down-up, up-down, down-down)."""
    from scipy.linalg import expm

    own, shared = (1.0 - beta) * rate, beta * rate
    G = np.array(
        [
            [0.0, own, own, shared],
            [repair, 0.0, 0.0, rate],
            [repair, 0.0, 0.0, rate],
            [0.0, repair, repair, 0.0],
        ]
    )
    G -= np.diag(G.sum(axis=1))
    start = np.array([1.0, 0.0, 0.0, 0.0])
    states = np.array([start @ expm(G * x) for x in np.atleast_1d(t)])
    return states[:, 0], states[:, 1], states[:, 3]


def test_common_cause_failure_frequency_against_its_chain():
    """``or(and(a, b), c)``, ``a`` and ``b`` in a beta-factor group: the
    pair goes down together at ``rate`` from one down, and at the shared
    rate from both up, and ``c`` fails at its rate while up, so ``w =
    (1 - q_c) w_ab + (1 - q_ab) w_c``."""
    tree = common_cause()
    x = np.array([0.5, 7.0, 49.0, 120.0, 600.0, 5000.0])
    up, one, both = pair_chain(0.002, 0.1, 0.1, x)
    w_ab = 2.0 * one * 0.002 + up * 0.1 * 0.002
    q_c = q_closed(0.0005, 0.25, x)
    w_c = 0.0005 * (1.0 - q_c)
    np.testing.assert_allclose(
        tree.failure_frequency(x),
        (1.0 - q_c) * w_ab + (1.0 - both) * w_c,
        rtol=1e-12,
    )
    assert tree.failure_frequency(5000.0) == pytest.approx(
        tree.system_failure_frequency(), rel=1e-12
    )
    # The measures read the same chain: c's Birnbaum importance is the
    # pair's being up, the pair's members' conditioned on their states.
    np.testing.assert_allclose(
        tree.birnbaum_importance(x)["c"], 1.0 - both, rtol=1e-12
    )
    np.testing.assert_allclose(
        tree.conditional_probabilities(x).given_occurred["a"],
        q_c + (1.0 - q_c) * both / (one + both),
        rtol=1e-12,
    )
    ranked = dict(tree.ranked_cut_sets(49.0))
    assert ranked[frozenset({"a", "b"})] == pytest.approx(both[2], rel=1e-12)


def test_the_failure_frequency_at_0_is_the_initial_intensity():
    # New at 0, every event is up: the top event happens at 0 only
    # through a, at its density then, 1/50 (a log-logistic of shape 1);
    # b's density is 0 then, and d's, infinite, is not critical, as e is
    # up.
    tree = FaultTree(
        {
            "top": ("or", ["a", "cb", "de"]),
            "cb": ("and", ["c", "b"]),
            "de": ("and", ["d", "e"]),
        },
        {
            "a": {
                "reliability": surv.LogLogistic.from_params([50.0, 1.0]),
                "repairability": E([0.5]),
            },
            "b": {
                "reliability": W([80.0, 1.5]),
                "repairability": surv.LogNormal.from_params([1.0, 0.5]),
            },
            "c": 0.2,
            "d": {"reliability": W([120.0, 0.8]), "repairability": E([0.4])},
            "e": unit(0.004, 0.5),
        },
    )
    assert tree.failure_frequency(0.0) == pytest.approx(0.02, rel=1e-12)
    at = tree.failure_frequency([0.0, 1e-7, 0.5])
    assert np.isfinite(at).all()
    # With a common-cause group: the pair's shared cause, at a tenth of
    # a member's rate, and c's own failures, w_ab + w_c.
    assert common_cause().failure_frequency(0.0) == pytest.approx(
        0.1 * 0.002 + 0.0005, rel=1e-12
    )


@pytest.mark.parametrize("tested", [False, True])
def test_common_cause_failure_frequency_integrates_to_expected_failures(
    tested,
):
    tree = common_cause(tested)
    end = 300.0 if not tested else 160.0
    total, _ = integrate.quad(
        lambda u: tree.failure_frequency(u),
        0.0,
        end,
        points=[80.0] if tested else None,
        epsabs=0.0,
        epsrel=1e-10,
        limit=200,
    )
    assert total == pytest.approx(tree.expected_failures(end), rel=1e-6)


def test_common_cause_frequency_and_measures_against_a_seeded_simulation():
    """The diagram of the tree simulated: the top event's starts over a
    window against the integral of ``failure_frequency``, and the top
    event at a time given ``a`` down against
    ``conditional_probabilities``."""
    tree = common_cause()
    end = 300.0
    runs = tree.to_repairable_rbd().simulate_timelines(
        end, mc_samples=20_000, seed=850
    )
    total, _ = integrate.quad(
        lambda u: tree.failure_frequency(u), 0.0, end, epsrel=1e-10
    )
    _within(runs.system.failures, total)
    t = 250.0
    a_down = ~runs.components["a"].state(t)
    top = ~runs.system.state(t)
    _within(a_down, q_closed(0.002, 0.1, t))
    _within(
        top[a_down],
        tree.conditional_probabilities(t).given_occurred["a"],
    )
    _within(
        top[~a_down],
        tree.conditional_probabilities(t).given_not_occurred["a"],
    )


def test_trees_without_groups_keep_their_values():
    """The routes through the groups' chains leave a tree without groups
    as it was: its measures are those of its independent events."""
    tree = common_cause()
    alone = FaultTree(dict(tree.gates), dict(tree.events))
    x = np.array([7.0, 49.0])
    q = {
        "a": q_closed(0.002, 0.1, x),
        "b": q_closed(0.002, 0.1, x),
        "c": q_closed(0.0005, 0.25, x),
    }
    birnbaum = alone.birnbaum_importance(x)
    np.testing.assert_allclose(birnbaum["c"], 1.0 - q["a"] * q["b"])
    np.testing.assert_allclose(birnbaum["a"], (1.0 - q["c"]) * q["b"])
    w = (1.0 - q["c"]) * 2.0 * 0.002 * (1.0 - q["a"]) * q["b"] + (
        1.0 - q["a"] * q["b"]
    ) * 0.0005 * (1.0 - q["c"])
    np.testing.assert_allclose(alone.failure_frequency(x), w, rtol=1e-12)


def test_the_importance_measures_of_a_mixed_tree():
    """The measures read each event's probability of being down at t: a
    fixed probability's own, a life's ``F(t)``."""
    tree = fault_tree_kinds()["repaired, with a probability and a life"]
    t = 40.0
    a = FaultTree({"top": ("or", ["a"])}, {"a": tree.events["a"]})
    qa = a.point_unavailability(t)
    qb = float(tree.events["b"].ff(t))
    birnbaum = tree.birnbaum_importance(t)
    c = tree.events["c"]
    assert birnbaum["c"] == pytest.approx(1.0 - qa * qb, rel=1e-12)
    assert birnbaum["a"] == pytest.approx((1.0 - c) * qb, rel=1e-12)
    assert birnbaum["b"] == pytest.approx((1.0 - c) * qa, rel=1e-12)


@pytest.mark.parametrize(
    "name",
    [
        "repaired, with a probability",
        "repaired, with a probability and a life",
        "repaired, common cause",
    ],
)
def test_a_mixed_or_grouped_tree_saves(name):
    tree = fault_tree_kinds()[name]
    again = FaultTree.from_json(tree.to_json())
    x = np.array([3.0, 90.0])
    np.testing.assert_array_equal(
        again.point_unavailability(x), tree.point_unavailability(x)
    )


def test_a_mixed_tree_through_open_psa():
    tree = FaultTree(
        {"top": ("or", ["pair", "h"]), "pair": ("and", ["a", "b"])},
        {"a": unit(0.01, 0.5), "b": E([0.002]), "h": 0.001},
    )
    again = FaultTree.from_open_psa(tree.to_open_psa())
    x = np.array([3.0, 90.0])
    np.testing.assert_allclose(
        again.point_unavailability(x), tree.point_unavailability(x), rtol=1e-12
    )
    at = FaultTree.from_open_psa(tree.to_open_psa(t=90.0))
    assert at.top_event_probability() == pytest.approx(
        tree.point_unavailability(90.0), rel=1e-12
    )


def test_an_event_tree_with_a_repaired_heading_beside_a_layer():
    diesel, start, alarm = unit(0.02, 0.5), 0.01, 0.1
    tree = EventTree(
        0.2,
        [
            ("alarm", alarm),
            (
                "backup",
                FaultTree(
                    {"no backup": ("or", ["diesel", "start"])},
                    {"diesel": diesel, "start": start},
                ),
            ),
        ],
    )
    t = 30.0
    q = 1.0 - (1.0 - q_closed(0.02, 0.5, t)) * (1.0 - start)
    probabilities = tree.sequence_probabilities(t).probabilities
    assert probabilities[("failure", "failure")] == pytest.approx(
        alarm * q, rel=1e-12
    )
    assert probabilities[("success", "success")] == pytest.approx(
        (1.0 - alarm) * (1.0 - q), rel=1e-12
    )
