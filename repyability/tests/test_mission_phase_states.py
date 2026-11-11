"""A mission drawn as one diagram, each phase setting its blocks' states
(working, standby, guaranteed, not needed), covariates and vote counts
(``PhasedMission.from_diagram``), against closed forms with exponential
and Weibull lives, the per-phase-diagram missions they reduce to, and the
simulation."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import (
    BlockSetting,
    MissionDrivers,
    MissionPhase,
    MissionSummary,
    NonRepairableRBD,
    PhasedMission,
    RegressionNode,
)

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = surv.FixedEventProbability.from_params
P = MissionPhase

SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
# Two pairs in series: (a1 or b1) then (a2 or b2).
PAIRS = [
    ("s", "a1"),
    ("s", "b1"),
    ("a1", "m"),
    ("b1", "m"),
    ("m", "a2"),
    ("m", "b2"),
    ("a2", "t"),
    ("b2", "t"),
]
VOTE = [
    ("s", "a"),
    ("s", "b"),
    ("s", "c"),
    ("a", "v"),
    ("b", "v"),
    ("c", "v"),
    ("v", "t"),
]


def within(interval_or_estimate, error, exact, z=4.0):
    """Whether ``exact`` lies within ``z`` standard errors."""
    return abs(interval_or_estimate - exact) <= z * error


def loads():
    """An exponential life 100 at load 1 and 25 at load 2 (as fitted)."""
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    return x, np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])


@pytest.fixture(scope="module")
def aft():
    return surv.ExponentialAFT.fit(*loads()[:1], Z=loads()[1])


@pytest.fixture(scope="module")
def ph():
    return surv.ExponentialPH.fit(*loads()[:1], Z=loads()[1])


def weibull_regression(kind):
    rng = np.random.default_rng(3)
    load = np.repeat([1.0, 2.0], 60)
    life = 100.0 / load**1.5 * rng.weibull(2.0, load.size)
    return kind.fit(life, Z=load[:, None])


# -- all working: the per-phase-diagram mission ------------------------------


@pytest.mark.parametrize("edges", [SERIES, PAIR], ids=["series", "parallel"])
def test_all_working_is_the_mission_of_one_diagram_per_phase(edges):
    models = {"a": E([0.01]), "b": E([0.02])}
    rbd = NonRepairableRBD(edges, models)
    drawn = PhasedMission.from_diagram(rbd, [P("p", 10.0), P("q", 20.0)])
    plain = PhasedMission([("p", 10.0, rbd), ("q", 20.0, rbd)])
    assert drawn.reliability() == plain.reliability()
    assert drawn.unreliability() == plain.unreliability()
    assert (
        drawn.phase_failure_probabilities()
        == plain.phase_failure_probabilities()
    )
    assert drawn.birnbaum_importance() == plain.birnbaum_importance()
    # Series: exp(-(0.01 + 0.02) 30); parallel: 1 - (1 - e^-0.3)(1 - e^-0.6).
    if edges is SERIES:
        closed = math.exp(-0.03 * 30.0)
    else:
        closed = 1 - (1 - math.exp(-0.3)) * (1 - math.exp(-0.6))
    assert drawn.reliability() == pytest.approx(closed, rel=1e-13)
    # The simulation draws as before, to the last bit.
    assert drawn.reliability_interval(
        2000, seed=4
    ) == plain.reliability_interval(2000, seed=4)
    assert drawn.phase_failure_probabilities(
        "simulate", mc_samples=500, seed=9
    ) == plain.phase_failure_probabilities("simulate", mc_samples=500, seed=9)


# -- guaranteed --------------------------------------------------------------


def test_guaranteed_exponential_in_series():
    # R = exp(-l1 T) exp(-l2 (T - T1)), b unable to fail in the first phase.
    rbd = NonRepairableRBD(SERIES, {"a": E([0.01]), "b": E([0.02])})
    mission = PhasedMission.from_diagram(
        rbd,
        [P("one", 10.0, blocks={"b": "guaranteed"}), P("two", 20.0)],
    )
    closed = math.exp(-0.01 * 30.0) * math.exp(-0.02 * 20.0)
    assert mission.reliability() == pytest.approx(closed, rel=1e-13)
    assert mission.phase_failure_probabilities() == pytest.approx(
        {
            "one": 1 - math.exp(-0.1),
            "two": math.exp(-0.1) - closed,
        },
        rel=1e-12,
    )


def test_guaranteed_weibull_ages_through_the_phase():
    # Survival of phase two is S(T) / S(T1): the age advanced in phase one.
    life = W([50.0, 2.5])
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": life})
    mission = PhasedMission.from_diagram(
        rbd, [P("one", 30.0, blocks={"a": "guaranteed"}), P("two", 20.0)]
    )
    closed = float(life.sf(50.0)) / float(life.sf(30.0))
    assert mission.reliability() == pytest.approx(closed, rel=1e-13)
    assert mission.phase_failure_probabilities()["one"] == 0.0
    interval = mission.reliability_interval(20_000, seed=1)
    assert within(interval.estimate, interval.standard_error, closed)


# -- not needed --------------------------------------------------------------


def test_not_needed_fails_silently_and_fails_the_later_phase():
    # a is not needed in phase one: it can fail there unseen, and phase two
    # fails then at its start. P1 = 1 - exp(-lb D1); P2 = exp(-lb D1) -
    # exp(-(la + lb)(D1 + D2)).
    la, lb, d1, d2 = 0.03, 0.01, 10.0, 5.0
    rbd = NonRepairableRBD(SERIES, {"a": E([la]), "b": E([lb])})
    mission = PhasedMission.from_diagram(
        rbd, [P("one", d1, blocks={"a": "not_needed"}), P("two", d2)]
    )
    expected = {
        "one": 1 - math.exp(-lb * d1),
        "two": math.exp(-lb * d1) - math.exp(-(la + lb) * (d1 + d2)),
    }
    assert mission.phase_failure_probabilities() == pytest.approx(
        expected, rel=1e-12
    )
    # As the mission of one diagram per phase, a left out of phase one.
    without = NonRepairableRBD([("s", "b"), ("b", "t")], {"b": E([lb])})
    plain = PhasedMission([("one", d1, without), ("two", d2, rbd)])
    assert mission.reliability() == pytest.approx(
        plain.reliability(), rel=1e-14
    )
    # The step at phase two's start: a lost unseen in phase one.
    before, after = mission.reliability_over_time([d1 - 1e-9, d1])
    assert before - after == pytest.approx(
        math.exp(-lb * d1) * (1 - math.exp(-la * d1)), rel=1e-6
    )


def test_a_phase_vote_count():
    # 3 units, 2-out-of-3 in the diagram, 1 needed in phase one: the phase
    # is the diagram with k=1.
    unit = E([0.02])
    models = {"a": unit, "b": unit, "c": unit, "v": F(0.0)}
    vote = NonRepairableRBD(VOTE, models, k={"v": 2})
    mission = PhasedMission.from_diagram(
        vote, [P("one", 10.0, k={"v": 1}), P("two", 10.0)]
    )
    any_one = NonRepairableRBD(VOTE, models, k={"v": 1})
    plain = PhasedMission([("one", 10.0, any_one), ("two", 10.0, vote)])
    assert mission.reliability() == pytest.approx(
        plain.reliability(), rel=1e-14
    )


# -- standby -----------------------------------------------------------------


def test_cold_standby_pair_of_identical_exponentials():
    # R(t) = exp(-l t) (1 + l t).
    lam, t = 0.01, 50.0
    rbd = NonRepairableRBD(PAIR, {"a": E([lam]), "b": E([lam])})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"b": "standby"})]
    )
    closed = math.exp(-lam * t) * (1 + lam * t)
    interval = mission.reliability_interval(40_000, seed=1)
    assert within(interval.estimate, interval.standard_error, closed)
    # b starts exactly when a fails within the phase: 1 - exp(-l t).
    summary = mission.summary(mc_samples=40_000, seed=1)
    assert summary.method == "simulated"
    assert summary.phases[0].takeovers["b"] == pytest.approx(
        1 - math.exp(-lam * t), abs=0.01
    )


def test_cold_standby_pair_with_different_rates():
    # Hypoexponential: R = e^-l1t + l1 / (l2 - l1) (e^-l1t - e^-l2t).
    l1, l2, t = 0.02, 0.005, 60.0
    rbd = NonRepairableRBD(PAIR, {"a": E([l1]), "b": E([l2])})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"b": "standby"})]
    )
    closed = math.exp(-l1 * t) + l1 / (l2 - l1) * (
        math.exp(-l1 * t) - math.exp(-l2 * t)
    )
    interval = mission.reliability_interval(40_000, seed=3)
    assert within(interval.estimate, interval.standard_error, closed)


def test_a_standby_never_needed_never_ages():
    # a cannot fail in phase one (guaranteed), so b is never started there;
    # in phase two a fails at once (its rate is 1e6) and b, working, is
    # new: R = S_b(D2) (to about 1e-6), not S_b(D1 + D2) / S_b(D1).
    life = W([40.0, 3.0])
    rbd = NonRepairableRBD(PAIR, {"a": E([1e6]), "b": life})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", 30.0, blocks={"a": "guaranteed", "b": "standby"}),
            P("two", 25.0),
        ],
    )
    closed = float(life.sf(25.0))
    summary = mission.summary(mc_samples=20_000, seed=1)
    assert summary.phases[0].takeovers == {"b": 0.0}
    assert within(summary.reliability, summary.interval.standard_error, closed)
    assert abs(closed - float(life.sf(55.0) / life.sf(30.0))) > 0.1


def test_standby_in_one_phase_working_in_the_next():
    # b on standby in phase one, working (in parallel with a) in phase two:
    # a lasting phase one, then either through phase two; or a failing at
    # t < D1, b running from t to the end:
    # R = e^-la D1 [1 - (1 - e^-la D2)(1 - e^-lb D2)]
    #     + la e^-lb T (e^(lb - la) D1 - 1) / (lb - la).
    la, lb, d1, d2 = 0.02, 0.01, 20.0, 30.0
    total = d1 + d2
    rbd = NonRepairableRBD(PAIR, {"a": E([la]), "b": E([lb])})
    mission = PhasedMission.from_diagram(
        rbd, [P("one", d1, blocks={"b": "standby"}), P("two", d2)]
    )
    closed = math.exp(-la * d1) * (
        1 - (1 - math.exp(-la * d2)) * (1 - math.exp(-lb * d2))
    ) + la * math.exp(-lb * total) * (math.exp((lb - la) * d1) - 1) / (lb - la)
    interval = mission.reliability_interval(40_000, seed=5)
    assert within(interval.estimate, interval.standard_error, closed)


def test_only_the_needed_spare_starts():
    # Two independent cold-standby pairs in series: R = R1(T) R2(T), each
    # R_i(t) = e^-l_i t (1 + l_i t). Spare b1 starts only when a1 fails
    # while the mission is still going: P = int_0^T l1 e^-l1 t R2(t) dt.
    l1, l2, t = 0.02, 0.01, 40.0

    def pair(lam, x):
        return math.exp(-lam * x) * (1 + lam * x)

    models = {"a1": E([l1]), "b1": E([l1]), "a2": E([l2]), "b2": E([l2])}
    models["m"] = F(0.0)
    rbd = NonRepairableRBD(PAIRS, models)
    mission = PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"b1": "standby", "b2": "standby"})]
    )
    summary = mission.summary(mc_samples=40_000, seed=2)
    closed = pair(l1, t) * pair(l2, t)
    assert within(summary.reliability, summary.interval.standard_error, closed)
    started = summary.phases[0].takeovers
    first = integrate.quad(
        lambda x: l1 * math.exp(-l1 * x) * pair(l2, x), 0, t
    )[0]
    second = integrate.quad(
        lambda x: l2 * math.exp(-l2 * x) * pair(l1, x), 0, t
    )[0]
    assert started["b1"] == pytest.approx(first, abs=0.01)
    assert started["b2"] == pytest.approx(second, abs=0.01)


def test_standby_blocks_start_in_the_order_given():
    # Three in parallel, b and c on standby: when a fails, b (listed first)
    # starts, and c only when b fails too.
    lam, t = 0.02, 50.0
    edges = PAIR + [("s", "c"), ("c", "t")]
    rbd = NonRepairableRBD(edges, {n: E([lam]) for n in "abc"})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", t, blocks={"b": "standby", "c": "standby"})]
    )
    summary = mission.summary(mc_samples=20_000, seed=1)
    started = summary.phases[0].takeovers
    # b: a failed by t; c: a then b failed by t (Erlang 2).
    assert started["b"] == pytest.approx(1 - math.exp(-lam * t), abs=0.012)
    assert started["c"] == pytest.approx(
        1 - math.exp(-lam * t) * (1 + lam * t), abs=0.012
    )
    # R: Erlang 3.
    closed = math.exp(-lam * t) * (1 + lam * t + (lam * t) ** 2 / 2)
    assert within(summary.reliability, summary.interval.standard_error, closed)


def test_standby_missions_refuse_the_exact_analyses():
    rbd = NonRepairableRBD(PAIR, {"a": E([0.01]), "b": E([0.01])})
    mission = PhasedMission.from_diagram(
        rbd, [P("only", 10.0, blocks={"b": "standby"})]
    )
    report = mission.analysis_routes()
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        mission.reliability()
    assert report["reliability"].route == "refused"
    with pytest.raises(NotImplementedError, match="failure_drivers"):
        mission.birnbaum_importance()
    with pytest.raises(NotImplementedError, match="failure_drivers"):
        mission.minimal_cut_sets()
    assert report["birnbaum_importance"].route == "refused"
    for name in ("summary", "failure_drivers", "reliability_over_time"):
        assert report[name].route == "simulated"
        assert "standby" in report[name].reason
    assert mission.reliability("simulate", mc_samples=100, seed=1) > 0.9


# -- covariates --------------------------------------------------------------


def test_phase_covariates_are_the_step_schedule(aft):
    # The covariates of each phase along the block's clock: as a
    # RegressionNode with that step schedule.
    node = RegressionNode(aft, covariates=[1.0])
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": node})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", 30.0),
            P("two", 20.0, blocks={"a": BlockSetting(covariates=[2.0])}),
            P("three", 10.0, blocks={"a": BlockSetting(covariates=[1.5])}),
        ],
    )
    schedule = surv.StepSchedule.from_changepoints(
        [0, 30, 50], [[1.0], [2.0], [1.5]]
    )
    stepped = NonRepairableRBD(
        [("s", "a"), ("a", "t")], {"a": RegressionNode(aft, schedule=schedule)}
    )
    plain = PhasedMission(
        [
            ("one", 30.0, stepped),
            ("two", 20.0, stepped),
            ("three", 10.0, stepped),
        ]
    )
    assert mission.reliability() == pytest.approx(
        plain.reliability(), rel=1e-12
    )
    assert mission.phase_failure_probabilities() == pytest.approx(
        plain.phase_failure_probabilities(), rel=1e-12
    )


def test_proportional_hazards_exponential_by_phase(ph):
    # R = exp(-sum_k l phi(x_k) d_k), l phi(x) the model's hazard at x.
    node = RegressionNode(ph, covariates=[1.0])
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": node})
    durations, values = [30.0, 20.0, 10.0], [1.0, 2.0, 1.5]
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P(f"p{k}", d, blocks={"a": BlockSetting(covariates=[x])})
            for k, (d, x) in enumerate(zip(durations, values))
        ],
    )
    rates = [float(np.ravel(ph.Hf(1.0, np.array([[x]])))[0]) for x in values]
    closed = math.exp(-sum(r * d for r, d in zip(rates, durations)))
    assert mission.reliability() == pytest.approx(closed, rel=1e-10)
    interval = mission.reliability_interval(20_000, seed=2)
    assert within(interval.estimate, interval.standard_error, closed)


def test_covariates_and_standby_on_one_block(aft):
    # a (rate la) with b on standby throughout, at load 1 (rate r1 = 1/100)
    # in phase one and load 2 (r2 = 1/25) in phase two. a failing at t:
    # before D1, b runs at r1 to D1 then r2 to T; after, at r2 from t:
    # R = e^-la T + int_0^D1 la e^-la t e^-(r1 (D1 - t) + r2 D2) dt
    #     + int_D1^T la e^-la t e^-r2 (T - t) dt.
    la, d1, d2 = 0.02, 20.0, 15.0
    total = d1 + d2
    r1 = float(np.ravel(aft.Hf(1.0, np.array([[1.0]])))[0])
    r2 = float(np.ravel(aft.Hf(1.0, np.array([[2.0]])))[0])

    def between(rate, lo, hi):  # int_lo^hi e^(rate t) dt
        return (math.exp(rate * hi) - math.exp(rate * lo)) / rate

    closed = (
        math.exp(-la * total)
        + la * math.exp(-r1 * d1 - r2 * d2) * between(r1 - la, 0.0, d1)
        + la * math.exp(-r2 * total) * between(r2 - la, d1, total)
    )
    node = RegressionNode(aft, covariates=[1.0])
    rbd = NonRepairableRBD(PAIR, {"a": E([la]), "b": node})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", d1, blocks={"b": "standby"}),
            P(
                "two",
                d2,
                blocks={"b": BlockSetting(state="standby", covariates=[2.0])},
            ),
        ],
    )
    interval = mission.reliability_interval(40_000, seed=1)
    assert within(interval.estimate, interval.standard_error, closed)


@pytest.mark.parametrize("kind", ["WeibullAFT", "WeibullPH"])
def test_a_standby_started_at_once_follows_its_schedule(kind):
    # In series, a block on standby is needed at once, so runs all the
    # mission: its life is the step schedule's (exact), whose age the
    # per-phase models carry from phase to phase (simulated).
    model = weibull_regression(getattr(surv, kind))
    node = RegressionNode(model, covariates=[1.0])
    rbd = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": node})

    def phases(state):
        return [
            P("one", 20.0, blocks={"a": state}),
            P(
                "two",
                30.0,
                blocks={"a": BlockSetting(state=state, covariates=[2.0])},
            ),
        ]

    exact = PhasedMission.from_diagram(rbd, phases("working")).reliability()
    standby = PhasedMission.from_diagram(rbd, phases("standby"))
    interval = standby.reliability_interval(20_000, seed=4)
    assert within(interval.estimate, interval.standard_error, exact)


# -- the whole mission at once -----------------------------------------------


def flight():
    """Two engines and a nav unit: both engines to take off, either to
    cruise (nav needed), both to land; nav guaranteed in the take-off."""
    engine = W([1500.0, 1.4])
    edges = [
        ("s", "e1"),
        ("s", "e2"),
        ("e1", "v"),
        ("e2", "v"),
        ("v", "nav"),
        ("nav", "t"),
    ]
    models = {"e1": engine, "e2": engine, "v": F(0.0), "nav": E([2e-4])}
    rbd = NonRepairableRBD(edges, models, k={"v": 1})
    return PhasedMission.from_diagram(
        rbd,
        [
            P("take-off", 0.2, blocks={"nav": "guaranteed"}, k={"v": 2}),
            P("cruise", 10.0),
            P("landing", 0.3, blocks={"nav": "not_needed"}, k={"v": 2}),
        ],
    )


def test_reliability_over_time():
    mission = flight()
    ends = [phase.end for phase in mission.phases]
    over = mission.reliability_over_time(ends)
    through = [mission._exact(j + 1)[0] for j in range(3)]
    assert over[-1] == pytest.approx(mission.reliability(), rel=1e-14)
    # A phase's end: the earlier phases, each whole (no step at the next
    # phase's start here but at the landing's, the second engine needed).
    assert over[0] == pytest.approx(through[0], rel=1e-12)
    times = np.linspace(0.0, mission.duration, 401)
    curve = mission.reliability_over_time(times)
    assert np.all(np.diff(curve) <= 1e-15)
    assert curve[0] == 1.0
    # Inside a phase of one exponential: R(t) = exp(-l t).
    unit = E([0.01])
    one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
    simple = PhasedMission.from_diagram(one, [P("p", 10.0), P("q", 10.0)])
    at = np.array([3.0, 10.0, 17.5])
    assert simple.reliability_over_time(at) == pytest.approx(
        np.exp(-0.01 * at), rel=1e-13
    )
    assert simple.reliability_over_time(4.0) == pytest.approx(
        math.exp(-0.04), rel=1e-13
    )


def test_reliability_over_time_simulated_agrees():
    mission = flight()
    times = np.array([0.1, 5.0, 10.2, 10.5])
    exact = mission.reliability_over_time(times)
    simulated = mission.reliability_over_time(
        times, "simulate", mc_samples=40_000, seed=3
    )
    error = np.sqrt(exact * (1 - exact) / 40_000)
    assert np.all(np.abs(simulated - exact) <= 4 * error + 1e-12)


def test_failure_drivers_of_a_series_system():
    # Series of exponentials: without a, the mission fails with
    # 1 - exp(-lb T): the fall is exp(-lb T) - exp(-(la + lb) T); in phase
    # one, exp(-lb D1) - exp(-(la + lb) D1).
    la, lb, d1, d2 = 0.01, 0.02, 10.0, 20.0
    total = d1 + d2
    rbd = NonRepairableRBD(SERIES, {"a": E([la]), "b": E([lb])})
    mission = PhasedMission.from_diagram(rbd, [P("one", d1), P("two", d2)])
    drivers = mission.failure_drivers()
    assert isinstance(drivers, MissionDrivers)
    assert drivers.totals["a"] == pytest.approx(
        math.exp(-lb * total) - math.exp(-(la + lb) * total), rel=1e-12
    )
    assert drivers.falls["a"]["one"] == pytest.approx(
        math.exp(-lb * d1) - math.exp(-(la + lb) * d1), rel=1e-12
    )
    for block, falls in drivers.falls.items():
        assert sum(falls.values()) == pytest.approx(
            drivers.totals[block], rel=1e-12
        )
    potential = mission.improvement_potential()
    assert drivers.totals == pytest.approx(potential, rel=1e-10)
    assert drivers.top("two") == "b"


def test_failure_drivers_exact_and_simulated_agree():
    mission = flight()
    exact = mission.failure_drivers()
    simulated = mission.failure_drivers("simulate", mc_samples=40_000, seed=1)
    assert simulated.method == "simulated" and simulated.n_samples == 40_000
    for block, total in exact.totals.items():
        assert within(
            simulated.totals[block], simulated.standard_errors[block], total
        )


@pytest.mark.parametrize(
    "build",
    [
        "guaranteed",
        "not_needed",
        "vote",
        "covariates",
    ],
)
def test_simulated_and_exact_agree_without_standby(build, aft):
    if build == "covariates":
        node = RegressionNode(aft, covariates=[1.0])
        rbd = NonRepairableRBD(PAIR, {"a": node, "b": W([60.0, 2.0])})
        phases = [
            P("one", 20.0, blocks={"a": BlockSetting(covariates=[2.0])}),
            P("two", 20.0, blocks={"b": "guaranteed"}),
        ]
        mission = PhasedMission.from_diagram(rbd, phases)
    elif build == "vote":
        unit = W([40.0, 1.5])
        models = {"a": unit, "b": unit, "c": unit, "v": F(0.0)}
        rbd = NonRepairableRBD(VOTE, models, k={"v": 2})
        mission = PhasedMission.from_diagram(
            rbd,
            [
                P("one", 10.0, k={"v": 3}, blocks={"c": "guaranteed"}),
                P("two", 15.0, k={"v": 1}),
            ],
        )
    else:
        rbd = NonRepairableRBD(SERIES, {"a": W([50.0, 2.0]), "b": E([0.01])})
        mission = PhasedMission.from_diagram(
            rbd,
            [P("one", 15.0, blocks={"a": build}), P("two", 10.0)],
        )
    exact = mission.reliability()
    interval = mission.reliability_interval(40_000, seed=7)
    assert within(interval.estimate, interval.standard_error, exact)
    simulated = mission.phase_failure_probabilities(
        "simulate", mc_samples=40_000, seed=7
    )
    for name, value in mission.phase_failure_probabilities().items():
        error = math.sqrt(max(value * (1 - value), 1e-12) / 40_000)
        assert abs(simulated[name] - value) <= 4 * error + 1e-12


def test_summary_is_one_result():
    mission = flight()
    summary = mission.summary()
    assert isinstance(summary, MissionSummary)
    assert summary.method == "exact" and summary.reason is None
    assert summary.reliability == pytest.approx(mission.reliability())
    assert summary.unreliability == mission.unreliability()
    failing = mission.phase_failure_probabilities()
    assert [p.failure_probability for p in summary.phases] == pytest.approx(
        list(failing.values()), rel=1e-12
    )
    assert sum(p.share for p in summary.phases) == pytest.approx(1.0)
    assert summary.riskiest_phase == max(failing, key=failing.get)
    # Every boundary is on the grid, with 20 steps a phase.
    for phase in summary.phases:
        assert phase.start in summary.times and phase.end in summary.times
    assert summary.reliability_over_time[-1] == pytest.approx(
        summary.reliability
    )
    landing = summary.phases[2]
    assert landing.settings == {
        "nav": {"state": "not_needed", "covariates": None}
    }
    reached = summary.phases[1].reliability_at_end
    assert landing.conditional_failure == pytest.approx(
        landing.failure_probability / reached
    )
    assert landing.top_driver in ("e1", "e2")
    # Through JSON, with a short repr.
    restored = json.loads(json.dumps(summary.to_dict()))
    assert restored["phases"][2]["name"] == "landing"
    assert len(restored["times"]) == len(summary.times)
    assert "MissionSummary(exact" in repr(summary)
    assert len(repr(summary)) < 200


def test_simulated_summary_through_json():
    rbd = NonRepairableRBD(PAIR, {"a": E([0.01]), "b": E([0.01])})
    mission = PhasedMission.from_diagram(
        rbd, [P("one", 20.0, blocks={"b": "standby"}), P("two", 10.0)]
    )
    summary = mission.summary(mc_samples=2_000, seed=1)
    again = mission.summary(mc_samples=2_000, seed=1)
    assert summary.to_dict() == again.to_dict()
    assert summary.interval.n_samples == 2_000
    assert summary.reliability_over_time[-1] == summary.reliability
    restored = json.loads(json.dumps(summary.to_dict()))
    assert restored["method"] == "simulated"
    assert "standby" in restored["reason"]
    assert set(restored["phases"][0]["takeovers"]) == {"b"}
    assert restored["phases"][1]["takeovers"] == {}


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_results_do_not_depend_on_the_unit_of_time(scale):
    def mission(c):
        rbd = NonRepairableRBD(
            SERIES, {"a": W([50.0 * c, 2.0]), "b": E([0.01 / c])}
        )
        return PhasedMission.from_diagram(
            rbd,
            [
                P("one", 10.0 * c, blocks={"b": "guaranteed"}),
                P("two", 20.0 * c, blocks={"a": "not_needed"}),
            ],
        )

    base, scaled = mission(1.0), mission(scale)
    assert scaled.reliability() == pytest.approx(base.reliability(), rel=1e-9)
    one, other = base.summary(), scaled.summary()
    assert other.times == pytest.approx(one.times * scale, rel=1e-12)
    assert other.reliability_over_time == pytest.approx(
        one.reliability_over_time, rel=1e-9
    )
    assert (
        scaled.reliability_interval(500, seed=1).estimate
        == base.reliability_interval(500, seed=1).estimate
    )


# -- checks ------------------------------------------------------------------


def test_block_settings_are_checked_and_made_by_keyword():
    with pytest.raises(ValueError, match="state must be 'working'"):
        BlockSetting(state="asleep")
    with pytest.raises(ValueError, match="finite numbers"):
        BlockSetting(covariates=[1.0, math.inf])
    with pytest.raises(ValueError, match="numbers"):
        BlockSetting(covariates=["hot"])
    with pytest.raises(TypeError):
        BlockSetting("standby")  # type: ignore[misc]
    assert BlockSetting(covariates=3).covariates == (3.0,)
    assert BlockSetting(state="standby") == BlockSetting(state="standby")
    with pytest.raises(ValueError, match="duration must be a finite"):
        P("p", -1.0)
    with pytest.raises(ValueError, match="state name or a BlockSetting"):
        P("p", 1.0, blocks={"a": 3})
    with pytest.raises(ValueError, match="whole number of at least 1"):
        P("p", 1.0, k={"v": 0})


def test_from_diagram_refuses_in_plain_words(aft):
    rbd = NonRepairableRBD(SERIES, {"a": E([0.01]), "b": E([0.02])})
    with pytest.raises(ValueError, match="'zz', which is not a node"):
        PhasedMission.from_diagram(
            rbd, [P("p", 1.0, blocks={"zz": "standby"})]
        )
    with pytest.raises(ValueError, match="input or output node"):
        PhasedMission.from_diagram(rbd, [P("p", 1.0, blocks={"s": "standby"})])
    with pytest.raises(ValueError, match="vote count of 'zz'"):
        PhasedMission.from_diagram(rbd, [P("p", 1.0, k={"zz": 1})])
    with pytest.raises(ValueError, match="must be a MissionPhase"):
        PhasedMission.from_diagram(rbd, [("p", 1.0, rbd)])
    with pytest.raises(ValueError, match="at least one phase"):
        PhasedMission.from_diagram(rbd, [])
    with pytest.raises(ValueError, match="Two phases are named 'p'"):
        PhasedMission.from_diagram(rbd, [P("p", 1.0), P("p", 2.0)])
    with pytest.raises(ValueError, match="not a RegressionNode"):
        PhasedMission.from_diagram(
            rbd, [P("p", 1.0, blocks={"a": BlockSetting(covariates=[1.0])})]
        )
    node = RegressionNode(aft, covariates=[1.0])
    regression = NonRepairableRBD(SERIES, {"a": node, "b": E([0.02])})
    with pytest.raises(ValueError, match="its model takes 1"):
        PhasedMission.from_diagram(
            regression,
            [P("p", 1.0, blocks={"a": BlockSetting(covariates=[1.0, 2.0])})],
        )
    repeated = NonRepairableRBD(
        [("s", "a"), ("a", "a2"), ("a2", "t")], {"a": E([0.01]), "a2": "a"}
    )
    with pytest.raises(ValueError, match="repeat of 'a'"):
        PhasedMission.from_diagram(
            repeated, [P("p", 1.0, blocks={"a2": "standby"})]
        )
    with pytest.raises(ValueError, match="NonRepairableRBD"):
        PhasedMission.from_diagram("diagram", [P("p", 1.0)])
    mission = PhasedMission.from_diagram(rbd, [P("p", 1.0)])
    with pytest.raises(ValueError, match="from 0 to the mission's duration"):
        mission.reliability_over_time(2.0)
    with pytest.raises(ValueError, match="mc_samples"):
        mission.reliability_over_time(0.5, "exact", mc_samples=10)
