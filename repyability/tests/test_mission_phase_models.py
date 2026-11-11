"""A block's own life model in one phase of a mission drawn as one diagram
(``BlockSetting(model=...)``), against closed forms with exponential and
Weibull lives, the covariates it generalises, and the simulation."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    BlockSetting,
    MissionPhase,
    NonRepairableRBD,
    PhasedMission,
    RegressionNode,
)
from repyability.rbd.serialisation import deserialise_model

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = surv.FixedEventProbability.from_params
P = MissionPhase
M = BlockSetting

ONE = [("s", "a"), ("a", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def within(estimate, error, exact, z=4.0):
    """Whether ``exact`` lies within ``z`` standard errors."""
    return abs(estimate - exact) <= z * error


def H(model, x):
    return float(np.ravel(model.Hf(x))[0])


def loads():
    """An exponential life 100 at load 1 and 25 at load 2 (as fitted)."""
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    return x, np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])


@pytest.fixture(scope="module")
def aft():
    return surv.ExponentialAFT.fit(loads()[0], Z=loads()[1])


@pytest.fixture(scope="module")
def ph():
    return surv.ExponentialPH.fit(loads()[0], Z=loads()[1])


# -- by hand ------------------------------------------------------------------


def test_exponential_by_phase():
    # R = exp(-l1 d1 - l2 d2) = exp(-0.01 * 30 - 0.05 * 20) = exp(-1.3).
    rbd = NonRepairableRBD(ONE, {"a": E([0.002])})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", 30.0, blocks={"a": M(model=E([0.01]))}),
            P("two", 20.0, blocks={"a": M(model=E([0.05]))}),
        ],
    )
    assert mission.reliability() == pytest.approx(math.exp(-1.3), rel=1e-13)
    assert mission.phase_failure_probabilities() == pytest.approx(
        {"one": 1 - math.exp(-0.3), "two": math.exp(-0.3) - math.exp(-1.3)},
        rel=1e-12,
    )
    interval = mission.reliability_interval(40_000, seed=1)
    assert within(interval.estimate, interval.standard_error, math.exp(-1.3))


def test_weibull_by_phase_carries_the_age():
    # Entering phase two at age d1 on its own model:
    # R = exp(-[H1(d1) - H1(0)] - [H2(d1 + d2) - H2(d1)]).
    first, second = W([40.0, 0.7]), W([80.0, 3.0])
    d1, d2 = 10.0, 30.0
    closed = math.exp(
        -(H(first, d1) - H(first, 0.0)) - (H(second, d1 + d2) - H(second, d1))
    )
    rbd = NonRepairableRBD(ONE, {"a": second})
    mission = PhasedMission.from_diagram(
        rbd, [P("one", d1, blocks={"a": M(model=first)}), P("two", d2)]
    )
    assert mission.reliability() == pytest.approx(closed, rel=1e-12)
    # Not the life started afresh in phase two: exp(-H1(d1) - H2(d2)).
    fresh = math.exp(-H(first, d1) - H(second, d2))
    assert abs(mission.reliability() - fresh) > 1e-3
    over = mission.reliability_over_time([5.0, d1, 25.0])
    assert over == pytest.approx(
        [
            math.exp(-H(first, 5.0)),
            math.exp(-H(first, d1)),
            math.exp(-H(first, d1) - H(second, 25.0) + H(second, d1)),
        ],
        rel=1e-12,
    )
    interval = mission.reliability_interval(40_000, seed=2)
    assert within(interval.estimate, interval.standard_error, closed)


# -- identities ---------------------------------------------------------------


def test_the_node_s_own_model_is_no_setting():
    unit = W([50.0, 2.0])
    rbd = NonRepairableRBD(PAIR, {"a": unit, "b": E([0.01])})
    plain = PhasedMission.from_diagram(rbd, [P("p", 10.0), P("q", 20.0)])
    for model in (unit, W([50.0, 2.0])):
        same = PhasedMission.from_diagram(
            rbd, [P("p", 10.0, blocks={"a": M(model=model)}), P("q", 20.0)]
        )
        assert same.reliability() == plain.reliability()
        assert (
            same.phase_failure_probabilities()
            == plain.phase_failure_probabilities()
        )
        assert same.reliability_interval(
            2_000, seed=4
        ) == plain.reliability_interval(2_000, seed=4)


@pytest.mark.parametrize("fit", ["aft", "ph"])
@pytest.mark.parametrize("state", ["working", "standby"])
def test_a_regression_pinned_at_covariates_is_those_covariates(
    fit, state, request
):
    # model=RegressionNode(m, covariates=z) and covariates=z are one life.
    model = request.getfixturevalue(fit)
    node = RegressionNode(model, covariates=[1.0])
    rbd = NonRepairableRBD(PAIR, {"a": node, "b": E([0.02])})

    def mission(setting):
        return PhasedMission.from_diagram(
            rbd,
            [
                P("one", 20.0, blocks={"b": state}),
                P("two", 15.0, blocks={"a": setting, "b": state}),
                P("three", 10.0, blocks={"b": state}),
            ],
        )

    pinned = mission(M(model=RegressionNode(model, covariates=[2.0])))
    given = mission(M(covariates=[2.0]))
    both = mission(
        M(model=RegressionNode(model, covariates=[1.0]), covariates=[2.0])
    )
    if state == "working":
        assert pinned.reliability() == given.reliability()
        assert both.reliability() == given.reliability()
    for other in (pinned, both):
        assert other.reliability_interval(
            5_000, seed=3
        ) == given.reliability_interval(5_000, seed=3)


# -- redundancy, exact against simulated --------------------------------------


def test_a_redundant_pair_exact_against_the_simulation():
    rbd = NonRepairableRBD(PAIR, {"a": W([60.0, 2.0]), "b": E([0.01])})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("start", 5.0, blocks={"a": M(model=W([30.0, 0.8]))}),
            P("run", 40.0),
            P("hot", 10.0, blocks={"b": M(model=E([0.05]))}),
        ],
    )
    assert mission.analysis_routes()["reliability"].route == "exact"
    exact = mission.reliability()
    interval = mission.reliability_interval(100_000, seed=7)
    assert within(interval.estimate, interval.standard_error, exact)
    assert interval.upper - interval.lower < 0.007
    simulated = mission.phase_failure_probabilities(
        "simulate", mc_samples=100_000, seed=7
    )
    for name, value in mission.phase_failure_probabilities().items():
        error = math.sqrt(max(value * (1 - value), 1e-12) / 100_000)
        assert abs(simulated[name] - value) <= 4 * error
    drivers = mission.failure_drivers()
    played = mission.failure_drivers("simulate", mc_samples=100_000, seed=1)
    for block, total in drivers.totals.items():
        assert within(
            played.totals[block], played.standard_errors[block], total
        )


# -- states -------------------------------------------------------------------


def test_cold_standby_runs_on_the_phase_s_model():
    # a (rate la) with b on standby throughout, b at rate r1 in phase one
    # and r2 (its phase model) in two. a failing at t: before D1, b runs at
    # r1 to D1 then r2 to T; after, at r2 from t:
    # R = e^-la T + int_0^D1 la e^-la t e^-(r1 (D1 - t) + r2 D2) dt
    #     + int_D1^T la e^-la t e^-r2 (T - t) dt.
    la, r1, r2, d1, d2 = 0.02, 0.01, 0.04, 20.0, 15.0
    total = d1 + d2

    def between(rate, lo, hi):  # int_lo^hi e^(rate t) dt
        return (math.exp(rate * hi) - math.exp(rate * lo)) / rate

    closed = (
        math.exp(-la * total)
        + la * math.exp(-r1 * d1 - r2 * d2) * between(r1 - la, 0.0, d1)
        + la * math.exp(-r2 * total) * between(r2 - la, d1, total)
    )
    rbd = NonRepairableRBD(PAIR, {"a": E([la]), "b": E([r1])})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", d1, blocks={"b": "standby"}),
            P("two", d2, blocks={"b": M(state="standby", model=E([r2]))}),
        ],
    )
    assert mission.analysis_routes()["reliability_interval"].route == (
        "simulated"
    )
    interval = mission.reliability_interval(100_000, seed=1)
    assert within(interval.estimate, interval.standard_error, closed)
    summary = mission.summary(mc_samples=20_000, seed=1)
    assert summary.method == "simulated"


def test_standby_against_a_hand_written_simulation():
    # Weibull lives, so the age carried into the phase model matters: b
    # idle until a fails, then runs on its node's life in phase one and
    # on W(30, 3) in phase two, its age carried.
    la, d1, d2 = 0.03, 15.0, 20.0
    own, hot = W([40.0, 1.5]), W([30.0, 3.0])
    rbd = NonRepairableRBD(PAIR, {"a": E([la]), "b": own})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", d1, blocks={"b": "standby"}),
            P("two", d2, blocks={"b": M(state="standby", model=hot)}),
        ],
    )
    rng = np.random.default_rng(11)
    n = 200_000
    a_fails = rng.exponential(1 / la, n)
    budget = rng.exponential(1.0, n)
    start = np.minimum(a_fails, d1 + d2)
    # b's hazard used from start (age 0) to the mission's end.
    first = np.clip(d1 - start, 0, None)  # running time in phase one
    second = np.where(start < d1, d2, d1 + d2 - start)  # in phase two
    used = own.Hf(first) + hot.Hf(first + second) - hot.Hf(first)
    works = (a_fails >= d1 + d2) | (used < budget)
    reference = works.mean()
    interval = mission.reliability_interval(100_000, seed=5)
    error = math.hypot(
        interval.standard_error, math.sqrt(reference * (1 - reference) / n)
    )
    assert within(interval.estimate, error, reference)


def test_guaranteed_makes_the_model_moot():
    # Guaranteed in phase one, so its phase-one model spends nothing:
    # R = exp(-(H(d1 + d2) - H(d1))) on the node's own life.
    unit = W([50.0, 2.5])
    rbd = NonRepairableRBD(ONE, {"a": unit})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P(
                "one",
                30.0,
                blocks={"a": M(state="guaranteed", model=E([1.0]))},
            ),
            P("two", 20.0),
        ],
    )
    closed = math.exp(-(H(unit, 50.0) - H(unit, 30.0)))
    assert mission.reliability() == pytest.approx(closed, rel=1e-12)
    interval = mission.reliability_interval(20_000, seed=1)
    assert within(interval.estimate, interval.standard_error, closed)


def test_not_needed_fails_on_the_phase_s_model():
    # b is not needed in phase one, where it runs on rate m1, but a
    # failure there fails phase two: R = e^-(la T) e^-(m1 d1 + m2 d2).
    la, m1, m2, d1, d2 = 0.01, 0.03, 0.02, 10.0, 20.0
    rbd = NonRepairableRBD(SERIES, {"a": E([la]), "b": E([m2])})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P("one", d1, blocks={"b": M(state="not_needed", model=E([m1]))}),
            P("two", d2),
        ],
    )
    closed = math.exp(-la * (d1 + d2) - m1 * d1 - m2 * d2)
    assert mission.reliability() == pytest.approx(closed, rel=1e-13)
    assert mission.phase_failure_probabilities()["one"] == pytest.approx(
        1 - math.exp(-la * d1), rel=1e-12
    )


# -- units, saving ------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_results_do_not_depend_on_the_unit_of_time(scale):
    def mission(c):
        rbd = NonRepairableRBD(
            PAIR, {"a": W([50.0 * c, 2.0]), "b": E([0.01 / c])}
        )
        return PhasedMission.from_diagram(
            rbd,
            [
                P("one", 10.0 * c, blocks={"a": M(model=W([20.0 * c, 0.7]))}),
                P("two", 20.0 * c, blocks={"b": M(model=E([0.05 / c]))}),
            ],
        )

    base, scaled = mission(1.0), mission(scale)
    assert scaled.reliability() == pytest.approx(base.reliability(), rel=1e-9)
    assert scaled.reliability_over_time(
        [5.0 * scale, 25.0 * scale]
    ) == pytest.approx(base.reliability_over_time([5.0, 25.0]), rel=1e-9)
    assert (
        scaled.reliability_interval(500, seed=1).estimate
        == base.reliability_interval(500, seed=1).estimate
    )


class _Unsaved:
    """A life with no saved form: an exponential of rate 0.01."""

    def sf(self, x):
        return np.exp(-0.01 * np.asarray(x, float))

    def ff(self, x):
        return -np.expm1(-0.01 * np.asarray(x, float))

    def Hf(self, x):
        return 0.01 * np.asarray(x, float)

    def random(self, size):
        return np.random.exponential(100.0, size)

    def __repr__(self):
        return "_Unsaved()"


def test_the_summary_holds_the_model_saved():
    start_up = W([30.0, 0.8])
    rbd = NonRepairableRBD(PAIR, {"a": W([60.0, 2.0]), "b": E([0.01])})
    mission = PhasedMission.from_diagram(
        rbd,
        [
            P(
                "start",
                5.0,
                blocks={"a": M(model=start_up), "b": M(model=_Unsaved())},
            ),
            P("run", 40.0),
        ],
    )
    settings = mission.summary().phases[0].settings
    saved = json.loads(json.dumps(settings))["a"]
    assert saved["state"] == "working" and saved["covariates"] is None
    again = deserialise_model(saved["model"])
    assert np.allclose(again.sf([1.0, 10.0]), start_up.sf([1.0, 10.0]))
    assert settings["b"]["model"] == "_Unsaved()"
    # Not set: no model key, as before.
    assert "model" not in mission.summary().to_dict()["phases"][1]


# -- refusals -----------------------------------------------------------------


def test_block_setting_refuses_what_is_no_life():
    with pytest.raises(TypeError, match="Weibull distribution itself"):
        M(model=surv.Weibull)
    with pytest.raises(TypeError, match="no survival function"):
        M(model=0.3)
    with pytest.raises(ValueError, match="probability of failing, not a"):
        M(model=F(0.1))
    with pytest.raises(ValueError, match="not a RegressionNode"):
        M(model=E([0.1]), covariates=[1.0])


def test_with_covariates_the_model_is_a_regression_of_their_length(aft):
    with pytest.raises(ValueError, match="takes 1 covariate"):
        M(model=RegressionNode(aft, covariates=[1.0]), covariates=[1.0, 2.0])
    schedule = surv.StepSchedule.from_changepoints([0, 5], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="fixed covariates"):
        M(model=RegressionNode(aft, schedule=schedule), covariates=[1.0])


def test_from_diagram_checks_the_model_against_the_node():
    from repyability import StandbyModel

    pair = StandbyModel([E([0.01]), E([0.01])])
    rbd = NonRepairableRBD(SERIES, {"a": pair, "b": F(0.01)})
    with pytest.raises(ValueError, match="StandbyModel, which takes no"):
        PhasedMission.from_diagram(
            rbd, [P("p", 1.0, blocks={"a": M(model=E([0.1]))})]
        )
    with pytest.raises(ValueError, match="probability of failing, which"):
        PhasedMission.from_diagram(
            rbd, [P("p", 1.0, blocks={"b": M(model=E([0.1]))}), P("q", 1.0)]
        )


def test_a_regression_that_carries_no_age_is_refused():
    rng = np.random.default_rng(3)
    load = np.repeat([1.0, 2.0], 60)
    life = 100.0 / load**1.5 * rng.weibull(2.0, load.size)
    model = surv.WeibullDistributional.fit(life, Z=load[:, None])
    rbd = NonRepairableRBD(ONE, {"a": RegressionNode(model, covariates=[1.0])})
    pinned = RegressionNode(model, covariates=[2.0])
    for setting in (M(covariates=[2.0]), M(model=pinned)):
        with pytest.raises(ValueError, match="'Distributional'"):
            PhasedMission.from_diagram(
                rbd, [P("p", 1.0), P("q", 1.0, blocks={"a": setting})]
            )
