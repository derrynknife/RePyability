"""Blocks on standby in a phased mission: imperfect switching, warm
standby (dormancy), the order they are called on, and the exact values
of the Markov chain where every life has a constant rate, against closed
forms, the decision diagram and the simulation."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import (
    BlockSetting,
    MissionPhase,
    NonRepairableRBD,
    PhasedMission,
    RegressionNode,
)
from repyability.rbd import _mission_chain

E, W = surv.Exponential.from_params, surv.Weibull.from_params
P, B = MissionPhase, BlockSetting

PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
THREE = PAIR + [("s", "c"), ("c", "t")]
SERIES3 = [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")]
VOTE = [
    ("s", "a"),
    ("s", "b"),
    ("s", "c"),
    ("a", "v"),
    ("b", "v"),
    ("c", "v"),
    ("v", "t"),
]


def within(estimate, error, exact, z=4.0):
    return abs(estimate - exact) <= z * error + 1e-12


def binomial(p, n):
    return math.sqrt(max(p * (1 - p), 1e-12) / n)


def pair_mission(la, lb, t, **standby):
    rbd = NonRepairableRBD(PAIR, {"a": E([la]), "b": E([lb])})
    return PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"b": B(state="standby", **standby)})]
    )


# -- closed forms ------------------------------------------------------------


def test_cold_standby_pair_is_exact():
    lam, t = 0.01, 50.0
    mission = pair_mission(lam, lam, t)
    assert mission.analysis_routes()["reliability"].route == "exact"
    closed = math.exp(-lam * t) * (1 + lam * t)
    assert mission.reliability() == pytest.approx(closed, rel=1e-12)
    assert mission.unreliability() == pytest.approx(1 - closed, rel=1e-10)
    summary = mission.summary()
    assert summary.method == "exact"
    assert summary.lower is None and summary.upper is None
    assert summary.phases[0].takeovers["b"] == pytest.approx(
        1 - math.exp(-lam * t), rel=1e-12
    )


def test_cold_standby_pair_with_different_rates_is_exact():
    l1, l2, t = 0.02, 0.005, 60.0
    closed = math.exp(-l1 * t) + l1 / (l2 - l1) * (
        math.exp(-l1 * t) - math.exp(-l2 * t)
    )
    assert pair_mission(l1, l2, t).reliability() == pytest.approx(
        closed, rel=1e-12
    )


@pytest.mark.parametrize("p", [0.0, 0.7, 0.95])
def test_imperfect_switching(p):
    # The spare starts with probability p: R = e^-lt (1 + p l t).
    lam, t = 0.01, 50.0
    mission = pair_mission(lam, lam, t, switching_probability=p)
    closed = math.exp(-lam * t) * (1 + p * lam * t)
    assert mission.reliability() == pytest.approx(closed, rel=1e-12)
    interval = mission.reliability_interval(40_000, seed=3)
    assert within(interval.estimate, interval.standard_error, closed)


@pytest.mark.parametrize("f", [0.0, 0.25, 1.0])
def test_warm_standby(f):
    # Idle, the spare fails at f l: R = e^-lt + (e^-lt - e^-(1+f)lt) / f;
    # hot (f = 1) is the parallel pair, 1 - (1 - e^-lt)^2.
    lam, t = 0.01, 50.0
    mission = pair_mission(lam, lam, t, dormancy=f)
    if f == 0.0:
        closed = math.exp(-lam * t) * (1 + lam * t)
    else:
        closed = (
            math.exp(-lam * t)
            + (math.exp(-lam * t) - math.exp(-(1 + f) * lam * t)) / f
        )
    if f == 1.0:
        assert closed == pytest.approx(1 - (1 - math.exp(-lam * t)) ** 2)
    assert mission.reliability() == pytest.approx(closed, rel=1e-12)
    interval = mission.reliability_interval(40_000, seed=4)
    assert within(interval.estimate, interval.standard_error, closed)


def test_warm_standby_with_imperfect_switching():
    # Spare alive at the call with e^-f l s, then starting with p:
    # R = e^-lt + p int_0^t l e^-ls e^-fls e^-l(t-s) ds
    #   = e^-lt (1 + p (1 - e^-flt) / f).
    lam, t, f, p = 0.02, 40.0, 0.5, 0.9
    mission = pair_mission(lam, lam, t, dormancy=f, switching_probability=p)
    closed = math.exp(-lam * t) * (1 + p * (1 - math.exp(-f * lam * t)) / f)
    assert mission.reliability() == pytest.approx(closed, rel=1e-12)
    interval = mission.reliability_interval(40_000, seed=5)
    assert within(interval.estimate, interval.standard_error, closed)


def test_guaranteed_phase_then_standby():
    # a cannot fail in the first phase, so b is never called there; in the
    # second b is a cold spare: R = e^-l d2 (1 + l d2).
    lam, d1, d2 = 0.02, 30.0, 25.0
    rbd = NonRepairableRBD(PAIR, {"a": E([lam]), "b": E([lam])})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", d1, blocks={"a": "guaranteed", "b": "standby"}),
            P("two", d2, blocks={"b": "standby"}),
        ],
    )
    closed = math.exp(-lam * d2) * (1 + lam * d2)
    assert mission.reliability() == pytest.approx(closed, rel=1e-12)
    summary = mission.summary()
    assert summary.phases[0].takeovers == {"b": 0.0}
    assert summary.phases[0].failure_probability == 0.0


# -- the chain against the decision diagram ---------------------------------


@pytest.mark.parametrize(
    "edges, phases, k",
    [
        (
            SERIES3,
            [
                P("p", 10.0, blocks={"b": "guaranteed"}),
                P("q", 20.0, blocks={"c": "not_needed"}),
            ],
            {},
        ),
        (
            VOTE,
            [P("p", 5.0, k={"v": 3}), P("q", 30.0), P("r", 5.0, k={"v": 3})],
            {"v": 2},
        ),
        (
            THREE,
            [
                P("p", 15.0, blocks={"a": "not_needed"}),
                P("q", 0.0),
                P("r", 9.0),
            ],
            {},
        ),
    ],
    ids=["series", "vote", "parallel"],
)
def test_the_chain_agrees_with_the_decision_diagram(edges, phases, k):
    # Without standby the chain must give the decision diagram's values.
    models = {"a": E([0.01]), "b": E([0.03]), "c": E([0.02])}
    nodes = {n for e in edges for n in e} - {"s", "t", "v"}
    if "v" in {n for e in edges for n in e}:
        from repyability import PerfectReliability

        models["v"] = PerfectReliability
        nodes.add("v")
    rbd = NonRepairableRBD(edges, {n: models[n] for n in nodes}, k=k)
    mission = PhasedMission.from_diagram(rbd, phases)
    times = np.linspace(0.0, mission.duration, 13)
    chain = _mission_chain.run(mission, times=times)
    assert chain.failing == pytest.approx(
        list(mission.phase_failure_probabilities().values()),
        rel=1e-9,
        abs=1e-15,
    )
    assert chain.curve == pytest.approx(
        mission.reliability_over_time(times), rel=1e-10
    )
    for node in mission.components:
        assert _mission_chain.run(mission, held=node).failing == (
            pytest.approx(mission._phase_failures(node), rel=1e-9, abs=1e-15)
        )


# -- a whole mission: exact against one simulation ---------------------------


def plant():
    """Start-up, duty, standby: two pumps, one on standby with warm
    dormancy and an imperfect switch; a valve needed only on duty."""
    rbd = NonRepairableRBD(
        [
            ("s", "p1"),
            ("s", "p2"),
            ("p1", "valve"),
            ("p2", "valve"),
            ("valve", "t"),
        ],
        {"p1": E([0.004]), "p2": E([0.006]), "valve": E([0.001])},
    )
    spare = B(state="standby", dormancy=0.2, switching_probability=0.95)
    return PhasedMission.from_diagram(
        rbd,
        [
            P("start-up", 2.0, blocks={"p1": "guaranteed", "p2": spare}),
            P("duty", 100.0, blocks={"p2": spare}),
            P("standby", 50.0, blocks={"valve": "not_needed", "p1": spare}),
        ],
    )


def test_exact_and_simulated_agree_on_a_whole_mission():
    mission = plant()
    exact = mission.summary()
    n = 60_000
    simulated = mission.summary(method="simulate", mc_samples=n, seed=7)
    assert exact.method == "exact" and simulated.method == "simulated"
    assert within(
        simulated.reliability,
        simulated.interval.standard_error,
        exact.reliability,
    )
    for e, s in zip(exact.phases, simulated.phases):
        assert within(
            s.failure_probability,
            binomial(e.failure_probability, n),
            e.failure_probability,
        )
        assert within(
            s.reliability_at_end,
            binomial(e.reliability_at_end, n),
            e.reliability_at_end,
        )
        for node, share in e.takeovers.items():
            assert within(s.takeovers[node], binomial(share, n), share)
    errors = np.sqrt(
        np.maximum(
            exact.reliability_over_time * (1 - exact.reliability_over_time),
            1e-12,
        )
        / n
    )
    assert np.all(
        np.abs(simulated.reliability_over_time - exact.reliability_over_time)
        <= 4 * errors + 1e-12
    )
    # The band holds the simulated curve, and nearly always the exact one.
    assert np.all(simulated.lower <= simulated.reliability_over_time)
    assert np.all(simulated.reliability_over_time <= simulated.upper)
    inside = (simulated.lower <= exact.reliability_over_time) & (
        exact.reliability_over_time <= simulated.upper
    )
    assert inside.mean() > 0.8
    # The curve ends at the mission's reliability, falling throughout.
    assert exact.reliability_over_time[-1] == pytest.approx(exact.reliability)
    assert np.all(np.diff(exact.reliability_over_time) <= 1e-15)
    drivers = mission.failure_drivers()
    played = mission.failure_drivers("simulate", mc_samples=n, seed=7)
    for node, total in drivers.totals.items():
        assert within(played.totals[node], played.standard_errors[node], total)


def test_seeded_runs_repeat():
    mission = plant()
    one = mission.summary(method="simulate", mc_samples=5_000, seed=11)
    two = mission.summary(method="simulate", mc_samples=5_000, seed=11)
    assert json.dumps(one.to_dict()) == json.dumps(two.to_dict())
    other = mission.summary(method="simulate", mc_samples=5_000, seed=12)
    assert other.reliability != one.reliability
    payload = json.loads(json.dumps(one.to_dict()))
    assert len(payload["lower"]) == len(payload["times"])
    assert payload["phases"][0]["settings"]["p2"] == {
        "state": "standby",
        "covariates": None,
        "switching_probability": 0.95,
        "dormancy": 0.2,
    }


# -- covariates by phase, and ageing across phases ---------------------------


@pytest.fixture(scope="module")
def aft():
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    z = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    return surv.ExponentialAFT.fit(x, Z=z)


def test_covariates_by_phase_with_a_spare(aft):
    # a: the fitted exponential regression at load 1, then 2; b a cold
    # spare of rate mu. By hand, a lasting both phases, or failing at s
    # and b running from s to T.
    a = RegressionNode(aft, covariates=[1.0])
    l1 = 1.0 / float(a.mean())
    l2 = 1.0 / float(RegressionNode(aft, covariates=[2.0]).mean())
    mu, d1, d2 = 0.01, 20.0, 15.0
    total = d1 + d2
    rbd = NonRepairableRBD(PAIR, {"a": a, "b": E([mu])})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("light", d1, blocks={"b": "standby"}),
            P("heavy", d2, blocks={"a": B(covariates=[2.0]), "b": "standby"}),
        ],
    )
    first = integrate.quad(
        lambda s: l1 * math.exp(-l1 * s) * math.exp(-mu * (total - s)), 0, d1
    )[0]
    second = integrate.quad(
        lambda s: l2
        * math.exp(-l1 * d1 - l2 * (s - d1))
        * math.exp(-mu * (total - s)),
        d1,
        total,
    )[0]
    closed = math.exp(-l1 * d1 - l2 * d2) + first + second
    assert mission.analysis_routes()["reliability"].route == "exact"
    assert mission.reliability() == pytest.approx(closed, rel=1e-9)
    interval = mission.reliability_interval(40_000, seed=2)
    assert within(interval.estimate, interval.standard_error, closed)


def test_warm_weibull_spare_ages_while_idle():
    # a cannot fail in phase one, so b idles through it, ageing f d1; in
    # phase two a fails at once and b runs on from that age, so it must
    # have lasted f d1 idle and d2 running: R = S_b(f d1 + d2) (Weibull,
    # so simulated).
    life = W([40.0, 3.0])
    f, d1, d2 = 0.5, 30.0, 20.0
    rbd = NonRepairableRBD(PAIR, {"a": E([1e6]), "b": life})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P(
                "one",
                d1,
                blocks={
                    "a": "guaranteed",
                    "b": B(state="standby", dormancy=f),
                },
            ),
            P("two", d2),
        ],
    )
    assert mission.analysis_routes()["reliability"].route == "refused"
    closed = float(life.sf(f * d1 + d2))
    summary = mission.summary(mc_samples=20_000, seed=1)
    assert summary.method == "simulated"
    assert "constant failure rate" in summary.reason
    assert within(summary.reliability, summary.interval.standard_error, closed)
    # Cold, it would not have aged; hot, it would have aged d1.
    assert abs(closed - float(life.sf(d2))) > 0.05
    assert abs(closed - float(life.sf(d1 + d2))) > 0.05


def test_warm_spare_found_failed_when_called():
    # A spare that fails idle is found failed when a fails: R = e^-lt
    # (1 + (1 - e^-flt) / f) for exponentials; with a Weibull a, the
    # simulation against the integral.
    a_life = W([60.0, 2.0])
    lam, f, t = 0.01, 0.5, 50.0
    rbd = NonRepairableRBD(PAIR, {"a": a_life, "b": E([lam])})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"b": B(state="standby", dormancy=f)})]
    )
    closed = (
        float(a_life.sf(t))
        + integrate.quad(
            lambda s: float(a_life.df(s))
            * math.exp(-f * lam * s)
            * math.exp(-lam * (t - s)),
            0,
            t,
        )[0]
    )
    interval = mission.reliability_interval(40_000, seed=6)
    assert within(interval.estimate, interval.standard_error, closed)


# -- the rule ----------------------------------------------------------------


@pytest.mark.parametrize("method", ["exact", "simulate"])
def test_spares_are_called_in_the_order_the_phase_names_them(method):
    # c is named before b: it is called first, when a fails; b only when
    # c fails too.
    la, lb, lc, t = 0.02, 0.01, 0.03, 50.0
    rbd = NonRepairableRBD(THREE, {"a": E([la]), "b": E([lb]), "c": E([lc])})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"c": "standby", "b": "standby"})]
    )
    kwargs = {"mc_samples": 40_000, "seed": 1} if method == "simulate" else {}
    summary = mission.summary(method=method, **kwargs)
    c = 1 - math.exp(-la * t)
    # a then c failed by t: hypoexponential(la, lc).
    b = 1 - (la * math.exp(-lc * t) - lc * math.exp(-la * t)) / (la - lc)
    started = summary.phases[0].takeovers
    if method == "exact":
        assert started["c"] == pytest.approx(c, rel=1e-12)
        assert started["b"] == pytest.approx(b, rel=1e-12)
    else:
        assert within(started["c"], binomial(c, 40_000), c)
        assert within(started["b"], binomial(b, 40_000), b)


def test_settings_are_checked():
    with pytest.raises(ValueError, match="for a block on standby"):
        B(switching_probability=0.9)
    with pytest.raises(ValueError, match="for a block on standby"):
        B(state="guaranteed", dormancy=0.1)
    for bad in (-0.1, 1.5, "0.5", True):
        with pytest.raises(ValueError, match="from 0 to 1"):
            B(state="standby", dormancy=bad)
        with pytest.raises(ValueError, match="from 0 to 1"):
            B(state="standby", switching_probability=bad)
    assert B(state="standby", switching_probability=1, dormancy=0) == B(
        state="standby"
    )


def test_a_chain_too_large_is_simulated(monkeypatch):
    mission = pair_mission(0.01, 0.01, 10.0)
    monkeypatch.setattr(_mission_chain, "MAX_STATES", 1)
    report = mission.analysis_routes()
    assert report["reliability"].route == "refused"
    assert "MAX_STATES" in report["reliability"].reason
    assert report["summary"].route == "simulated"
    with pytest.raises(NotImplementedError, match="MAX_STATES"):
        mission.reliability()
    assert mission.summary(mc_samples=200, seed=1).method == "simulated"


def test_importance_measures_still_refuse_with_standby():
    mission = pair_mission(0.01, 0.01, 10.0)
    with pytest.raises(NotImplementedError, match="failure_drivers"):
        mission.birnbaum_importance()
    assert mission.failure_drivers().method == "exact"
