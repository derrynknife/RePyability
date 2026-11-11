"""A unit's own life in a state: the degradation level measured now
(``NodeState.level``) and a life model of the unit's own
(``NodeState.model``), in a ``NonRepairableRBD``'s condition-based
methods, a ``RepairableRBD``'s analyses from a state and its simulations.

Checked by hand (a gamma process's first passage at a whole shape is a
Poisson tail; a Wiener process's is the inverse Gaussian), by identities
(a level at the process's start is the unit new; a model that is the
node's own is the age alone, to the last bit, in the simulation too),
against the same diagram built with the unit's life as the node's where
only the first life matters, against the simulation and a hand-written
Monte Carlo of one unit, by the unit of time and of the level, and every
refusal with its message.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import norm
from surpyval.degradation import GammaProcessModel, WienerProcessModel

from repyability import (
    NodeState,
    NonRepairable,
    NonRepairableRBD,
    RegressionNode,
    RepairableRBD,
    SimulationChunk,
    StandbyModel,
)
from repyability.rbd import _runs
from repyability.rbd._overrides import FromLevel, checked_override
from repyability.tests.catalogue import PUMP, joint_model, joint_unit

ONE = [("s", "a"), ("a", "t")]
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
E = surv.Exponential.from_params
W = surv.Weibull.from_params


def gamma(y0=0.0, scale=1.0):
    """Wear accruing as a gamma process: over a time ``t`` a gamma of
    shape ``2 t`` and rate 4 (a mean of 0.5 a unit of time), failing at
    10; ``scale`` multiplies every level."""
    return GammaProcessModel(
        alpha=2.0, beta=4.0 / scale, threshold=10.0 * scale, y0=y0 * scale
    )


def wiener(y0=3.0):
    """A Wiener process drifting at 0.5 with diffusion 1, failing at 10."""
    return WienerProcessModel(mu=0.5, sigma=1.0, threshold=10.0, y0=y0)


def poisson_tail(k: int, m: float) -> float:
    """P(N >= k) for N Poisson with mean m: P(Gamma(k, 1) < m)."""
    return 1.0 - sum(math.exp(-m) * m**j / math.factorial(j) for j in range(k))


def inverse_gaussian_sf(t, d, mu=0.5, sigma=1.0):
    """The first passage of a Wiener process over a distance d: its
    survival, Phi((d - mu t) / (sigma sqrt t)) - exp(2 mu d / sigma^2)
    Phi(-(d + mu t) / (sigma sqrt t))."""
    t = np.asarray(t, dtype=float)
    root = sigma * np.sqrt(t)
    return norm.cdf((d - mu * t) / root) - np.exp(
        2.0 * mu * d / sigma**2
    ) * norm.cdf(-(d + mu * t) / root)


# -- NonRepairableRBD -------------------------------------------------------


def test_a_gamma_level_gives_the_poisson_tail():
    # From a level of 6 the wear left is 4, at rate 4: 16 in units of the
    # increments' scale. At t the wear is Gamma(2 t, 4); at t = 8 its
    # shape is 16, so the unit survives with P(Gamma(16, 1) < 16), the
    # Poisson tail P(N(16) >= 16); at t = 2.5, P(N(16) >= 5).
    rbd = NonRepairableRBD(ONE, {"a": gamma()})
    state = {"a": NodeState(age=3.0, level=6.0)}
    got = rbd.sf_given_state(np.array([2.5, 8.0]), state)
    expected = [poisson_tail(5, 16.0), poisson_tail(16, 16.0)]
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert expected[1] == pytest.approx(0.5332, abs=1e-4)
    # The age changes nothing given the level (independent increments).
    later = {"a": NodeState(age=30.0, level=6.0)}
    assert np.array_equal(rbd.sf_given_state(np.array([2.5, 8.0]), later), got)


def test_a_wiener_level_gives_the_inverse_gaussian():
    # From a level of 2, below the process's start of 3 (a Wiener path
    # can fall), the distance to the threshold is 8.
    rbd = NonRepairableRBD(ONE, {"a": wiener()})
    t = np.array([5.0, 16.0, 30.0])
    got = rbd.sf_given_state(t, {"a": NodeState(level=2.0)})
    np.testing.assert_allclose(got, inverse_gaussian_sf(t, 8.0), rtol=1e-10)


def test_a_parallel_pair_multiplies_by_hand():
    rbd = NonRepairableRBD(PARALLEL, {"a": gamma(), "b": wiener()})
    state = {"a": NodeState(level=6.0), "b": NodeState(level=2.0)}
    x = 8.0
    q_a = 1.0 - poisson_tail(16, 16.0)
    q_b = 1.0 - float(inverse_gaussian_sf(x, 8.0))
    assert rbd.sf_given_state(x, state) == pytest.approx(
        1.0 - q_a * q_b, rel=1e-10
    )
    # The importances from now: Birnbaum's of a is the chance b fails.
    imp = rbd.importances_given_state(x, state)
    assert imp["birnbaum"]["a"] == pytest.approx(q_b, rel=1e-10)


def test_a_level_at_the_start_is_the_unit_new():
    for model in (gamma(y0=1.0), wiener()):
        rbd = NonRepairableRBD(ONE, {"a": model})
        x = np.array([1.0, 10.0, 20.0])
        level = {"a": NodeState(level=float(model.y0))}
        np.testing.assert_allclose(
            rbd.sf_given_state(x, level), rbd.sf(x), rtol=1e-14
        )
        assert rbd.mean_residual_life(level) == pytest.approx(
            rbd.mean(), rel=1e-9
        )


def test_the_unit_own_model_is_its_age_alone_when_it_is_the_node_own():
    life = W([100.0, 2.0])
    rbd = NonRepairableRBD(PARALLEL, {"a": life, "b": W([80.0, 1.5])})
    x = np.array([0.0, 10.0, 40.0])
    aged = {"a": NodeState(age=50.0), "b": NodeState(age=20.0)}
    own = {"a": NodeState(age=50.0, model=life), "b": NodeState(age=20.0)}
    assert np.array_equal(
        rbd.sf_given_state(x, own), rbd.sf_given_state(x, aged)
    )
    assert rbd.remaining_life(0.5, own) == rbd.remaining_life(0.5, aged)


def test_the_unit_own_model_stands_in_for_the_node():
    # A pump refitted to its own history, worse than the fleet's.
    own = W([60.0, 3.0])
    rbd = NonRepairableRBD(PARALLEL, {"a": W([100.0, 2.0]), "b": E([0.01])})
    state = {"a": NodeState(age=20.0, model=own), "b": NodeState(age=5.0)}
    x = 15.0
    r_a = float(own.sf(35.0) / own.sf(20.0))
    r_b = math.exp(-0.01 * 15.0)
    assert rbd.sf_given_state(x, state) == pytest.approx(
        1.0 - (1.0 - r_a) * (1.0 - r_b), rel=1e-12
    )
    # The mean life left integrates the unit's own life.
    single = NonRepairableRBD(ONE, {"a": W([100.0, 2.0])})
    mine = {"a": NodeState(age=20.0, model=own)}
    by_hand = NonRepairableRBD(ONE, {"a": own}).mean_residual_life(
        {"a": NodeState(age=20.0)}
    )
    assert single.mean_residual_life(mine) == pytest.approx(by_hand, rel=1e-9)


def test_a_level_applies_to_the_unit_own_process():
    # The fleet's life is a Weibull; this unit is watched, its wear a
    # gamma process fitted to it.
    rbd = NonRepairableRBD(ONE, {"a": W([10.0, 2.0])})
    state = {"a": NodeState(age=4.0, level=6.0, model=gamma())}
    assert rbd.sf_given_state(8.0, state) == pytest.approx(
        poisson_tail(16, 16.0), rel=1e-12
    )


def test_a_joint_model_unit_stands_in_for_an_ordinary_node():
    # The pump's prediction from its readings, given as its own model.
    unit = joint_unit()
    rbd = NonRepairableRBD(ONE, {"a": W([100.0, 2.0])})
    x = np.array([5.0, 20.0])
    state = {"a": NodeState(age=40.0, model=unit)}
    np.testing.assert_allclose(
        rbd.sf_given_state(x, state), unit.sf(x), rtol=1e-14
    )
    # A prediction is taken as its JointNode.
    prediction = joint_model().predict(
        PUMP["Z"], PUMP["markers"], given=PUMP["age"]
    )
    as_given = {"a": NodeState(age=40.0, model=prediction)}
    np.testing.assert_allclose(
        rbd.sf_given_state(x, as_given), unit.sf(x), rtol=1e-14
    )
    with pytest.raises(ValueError, match="would condition it again"):
        rbd.sf_given_state(x, {"a": NodeState(age=10.0, model=unit)})


def test_scaling_the_level_changes_nothing():
    # Levels, threshold, start and the increments' scale times c: the same
    # unit measured in other units.
    x = np.array([2.5, 8.0, 20.0])
    base = NonRepairableRBD(ONE, {"a": gamma()}).sf_given_state(
        x, {"a": NodeState(level=6.0)}
    )
    for c in (1e-6, 1e-3, 1e3, 1e6):
        rbd = NonRepairableRBD(ONE, {"a": gamma(scale=c)})
        np.testing.assert_allclose(
            rbd.sf_given_state(x, {"a": NodeState(level=6.0 * c)}),
            base,
            rtol=1e-9,
        )
        drift = WienerProcessModel(0.5 * c, c, 10.0 * c, y0=3.0 * c)
        rbd = NonRepairableRBD(ONE, {"a": drift})
        np.testing.assert_allclose(
            rbd.sf_given_state(x, {"a": NodeState(level=2.0 * c)}),
            inverse_gaussian_sf(x, 8.0),
            rtol=1e-9,
        )


@pytest.mark.parametrize("k", [1e-6, 1e-3, 1e3, 1e6])
def test_results_scale_with_the_unit_of_time(k):
    # Time in units k times smaller: the gamma's shape accrues k times
    # slower, the Weibull's scale is k times larger.
    x, age = 8.0, 20.0
    base = NonRepairableRBD(PARALLEL, {"a": gamma(), "b": W([100.0, 2.0])})
    state = {
        "a": NodeState(age=3.0, level=6.0),
        "b": NodeState(age=age, model=W([60.0, 3.0])),
    }
    scaled = NonRepairableRBD(
        PARALLEL,
        {
            "a": GammaProcessModel(alpha=2.0 / k, beta=4.0, threshold=10.0),
            "b": W([100.0 * k, 2.0]),
        },
    )
    scaled_state = {
        "a": NodeState(age=3.0 * k, level=6.0),
        "b": NodeState(age=age * k, model=W([60.0 * k, 3.0])),
    }
    assert scaled.sf_given_state(x * k, scaled_state) == pytest.approx(
        base.sf_given_state(x, state), rel=1e-9
    )
    assert scaled.remaining_life(0.5, scaled_state) == pytest.approx(
        k * base.remaining_life(0.5, state), rel=1e-6
    )


# -- RepairableRBD: exact ---------------------------------------------------

#: A repair that never ends within the windows below: only the first
#: life matters.
NEVER = surv.ExactEventTime.from_params(1e6)


def unit(life, repair=NEVER, **spec):
    return RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": life,
                "repairability": repair,
                "replace_cost": 10.0,
                **spec,
            }
        },
    )


def test_the_first_life_from_a_level_without_repair():
    rbd = unit(gamma())
    x = np.array([0.0, 2.5, 8.0])
    state = {"a": NodeState(age=3.0, level=6.0)}
    expected = [1.0, poisson_tail(5, 16.0), poisson_tail(16, 16.0)]
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state), expected, atol=1e-7
    )
    assert rbd.expected_failures(8.0, state=state) == pytest.approx(
        1.0 - expected[2], abs=1e-6
    )
    cost = rbd.expected_cost(8.0, state=state)
    assert cost.mean == pytest.approx(10.0 * (1.0 - expected[2]), abs=1e-5)


def test_an_own_model_is_that_diagram_while_only_the_first_life_counts():
    own = W([60.0, 3.0])
    mine = unit(W([100.0, 2.0]))
    theirs = unit(own)
    x = np.array([0.0, 10.0, 30.0, 50.0])
    a = {"a": NodeState(age=20.0, model=own)}
    b = {"a": NodeState(age=20.0)}
    np.testing.assert_allclose(
        mine.point_availability(x, state=a),
        theirs.point_availability(x, state=b),
        rtol=0,
        atol=1e-12,
    )
    for method in ("expected_failures",):
        assert getattr(mine, method)(50.0, state=a) == pytest.approx(
            getattr(theirs, method)(50.0, state=b), abs=1e-12
        )
    npv = dict(discount_rate=0.05, value_rate=3.0)
    assert mine.expected_npv(50.0, state=a, **npv).npv == pytest.approx(
        theirs.expected_npv(50.0, state=b, **npv).npv, rel=1e-10
    )
    # The node's own model as the unit's: the age alone, to the last bit.
    same = {"a": NodeState(age=20.0, model=mine.components["a"].reliability)}
    assert np.array_equal(
        mine.point_availability(x, state=same),
        mine.point_availability(x, state=b),
    )


def test_age_replacement_falls_due_on_the_operating_time():
    # Replaced at age 5 (in no time): a unit 3 old, watched (its wear a
    # gamma process of its own, at a level of 6), runs its first life
    # from the level until 2 from now, then is replaced by a new unit of
    # the node's (Weibull) life.
    life = gamma()
    rbd = unit(W([100.0, 2.0]), preventive={"interval": 5.0, "cost": 1.0})
    state = {"a": NodeState(age=3.0, level=6.0, model=life)}
    up = rbd.point_availability(np.array([1.9]), state=state)
    assert up[0] == pytest.approx(float(life.sf(1.9, y0=6.0)), abs=1e-7)
    events = rbd.expected_events(2.5, state=state)
    # One replacement at 2 if it survived that long.
    assert events.node_preventive["a"] == pytest.approx(
        float(life.sf(2.0, y0=6.0)), abs=1e-6
    )


def test_a_level_with_repairs_against_a_hand_monte_carlo():
    # The unit from a level of 6, repaired in exponential times of mean
    # 0.5, its replacements new (at a level of 0): its expected failures
    # by T against first lives drawn from surpyval's first passage from
    # the level, then new lives, each repair after a failure.
    life, repair = gamma(), E([2.0])
    rbd = unit(life, repair)
    state = {"a": NodeState(age=3.0, level=6.0)}
    T = 30.0
    exact = rbd.expected_failures(T, state=state)
    g = np.random.default_rng(11)
    n = 20_000
    t = np.ravel(life.random(n, y0=6.0, random_state=1))
    count = np.zeros(n)
    while np.any(t < T):
        live = t < T
        count += live
        t = np.where(
            live,
            t
            + g.exponential(0.5, n)
            + np.ravel(life.random(n, random_state=int(g.integers(1e9)))),
            t,
        )
    se = count.std() / math.sqrt(n)
    assert abs(count.mean() - exact) < 4.0 * se, (count.mean(), exact, se)


def test_the_first_failure_of_a_series_from_a_level():
    rbd = unit(gamma(), E([2.0]))
    state = {"a": NodeState(level=6.0)}
    assert rbd.first_failure_sf(8.0, state=state) == pytest.approx(
        poisson_tail(16, 16.0), rel=1e-10
    )
    assert rbd.mean_time_to_first_failure(state=state) == pytest.approx(
        float(gamma().mean(y0=6.0)), rel=1e-6
    )


# -- RepairableRBD: simulation ----------------------------------------------


def test_the_simulation_from_a_level_agrees_with_the_exact_values():
    rbd = unit(gamma(), E([2.0]))
    state = {"a": NodeState(age=3.0, level=6.0)}
    run = rbd.availability(
        12.0, mc_samples=4000, seed=4, state=state, control_variate=False
    )
    interval = run.mean_availability_interval(0.9999)
    exact = rbd.mission_availability(12.0, state=state)
    assert interval.lower < exact < interval.upper
    assert interval.upper - interval.lower < 0.01
    cost = rbd.cost(
        12.0, mc_samples=4000, seed=4, state=state, control_variate=False
    ).mean_interval(0.9999)
    assert cost.lower < rbd.expected_cost(12.0, state=state).mean < cost.upper


def test_the_simulation_of_an_own_model_agrees_with_the_exact_values():
    rbd = unit(W([100.0, 2.0]), E([0.1]))
    state = {"a": NodeState(age=20.0, model=W([60.0, 3.0]))}
    run = rbd.availability(
        80.0, mc_samples=4000, seed=5, state=state, control_variate=False
    )
    interval = run.mean_availability_interval(0.9999)
    exact = rbd.mission_availability(80.0, state=state)
    assert interval.lower < exact < interval.upper


def test_the_node_own_model_simulates_as_the_age_alone():
    rbd = unit(W([100.0, 2.0]), E([0.1]))
    life = rbd.components["a"].reliability
    options = dict(mc_samples=300, seed=2, control_variate=False)
    aged = rbd.availability(80.0, state={"a": NodeState(age=20.0)}, **options)
    own = rbd.availability(
        80.0, state={"a": NodeState(age=20.0, model=life)}, **options
    )
    assert np.array_equal(aged.uptimes, own.uptimes)


def test_the_compiled_engine_refuses_an_own_life_as_any_start():
    # Every start from a state is the Python loop's (its START stream);
    # the refusal comes before numba is needed.
    rbd = unit(gamma(), E([2.0]))
    for state in (
        {"a": NodeState(age=3.0)},
        {"a": NodeState(age=3.0, level=6.0)},
    ):
        with pytest.raises(NotImplementedError, match="started from a state"):
            rbd.availability(
                5.0, mc_samples=10, seed=1, state=state, engine="numba"
            )
    options = dict(mc_samples=50, seed=1, control_variate=False)
    state = {"a": NodeState(age=3.0, level=6.0)}
    assert np.array_equal(
        rbd.availability(5.0, state=state, engine="auto", **options).uptimes,
        rbd.availability(5.0, state=state, engine="python", **options).uptimes,
    )


def test_chunks_carry_an_own_model():
    rbd = unit(W([100.0, 2.0]), E([0.1]))
    state = {"a": NodeState(age=20.0, level=None, model=W([60.0, 3.0]))}
    first = rbd.simulate_chunk(80.0, 0, 60, seed=3, state=state)
    rest = rbd.simulate_chunk(80.0, 60, 100, seed=3, state=state)
    merged = rbd.availability_from_chunks(
        [first, SimulationChunk.from_json(rest.to_json())]
    )
    whole = rbd.availability(80.0, mc_samples=100, seed=3, state=state)
    assert np.array_equal(merged.uptimes, whole.uptimes)
    # The key keeps a plain state as it was, and reads an own life back.
    plain = json.loads(_runs._state_key({"a": NodeState(age=2.0)}))
    assert set(plain[0][1]) == {
        "age",
        "alive",
        "down_for",
        "maintenance",
        "since_scheduled",
        "stationary",
        "virtual_age",
    }
    key = _runs._state_key({"a": NodeState(age=2.0, level=6.0, model=gamma())})
    back = _runs._states_from_key(rbd, key)["a"]
    assert back.level == 6.0 and back.model.params.tolist() == [2.0, 4.0]


def test_an_own_model_under_block_replacement_from_its_phase():
    # Replaced every 50 (in no time), last 20 ago: the unit with its own
    # model runs until 30 from now, then the block calendar renews it.
    rbd = unit(
        W([100.0, 2.0]),
        E([0.1]),
        preventive={"policy": "block", "interval": 50.0, "cost": 1.0},
    )
    own = W([60.0, 3.0])
    state = {"a": NodeState(age=20.0, since_scheduled=20.0, model=own)}
    run = rbd.availability(
        100.0, mc_samples=4000, seed=6, state=state, control_variate=False
    )
    interval = run.mean_availability_interval(0.9999)
    exact = rbd.mission_availability(100.0, state=state)
    assert interval.lower < exact < interval.upper
    # Unlike from the node's own life at that age.
    aged = rbd.mission_availability(
        100.0, state={"a": NodeState(age=20.0, since_scheduled=20.0)}
    )
    assert exact < aged


# -- refusals ---------------------------------------------------------------


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"level": float("nan")}, "finite number"),
        ({"level": "6"}, "finite number"),
        ({"level": True}, "finite number"),
        ({"level": 6.0, "alive": False}, "a unit that is down has neither"),
        ({"model": E([1.0]), "alive": False}, "a unit that is down"),
        ({"level": 6.0, "virtual_age": 1.0}, "give one or the other"),
        ({"model": E([1.0]), "stationary": True}, "long-run state"),
    ],
)
def test_a_state_refuses_what_has_no_meaning(fields, match):
    with pytest.raises(ValueError, match=match):
        NodeState(**fields)


def test_the_new_fields_show_and_are_not_new():
    state = NodeState(age=3.0, level=6.0)
    assert repr(state) == "NodeState(age=3.0, alive=True, level=6.0)"
    assert not state.new and not NodeState(model=E([1.0])).new
    assert "model=" in repr(NodeState(model=E([1.0])))


@pytest.mark.parametrize(
    "life, state, error, match",
    [
        (W([10.0, 2.0]), NodeState(level=1.0), ValueError, "Give its age"),
        (gamma(), NodeState(level=10.0), ValueError, "alive=False"),
        (gamma(), NodeState(level=12.0), ValueError, "has failed"),
        (gamma(y0=1.0), NodeState(level=0.5), ValueError, "only rises"),
        (gamma(), NodeState(model=surv.Weibull), TypeError, "itself"),
        (gamma(), NodeState(model=3.0), TypeError, "the number"),
        (
            gamma(),
            NodeState(model=StandbyModel([E([1.0]), E([1.0])])),
            ValueError,
            "takes no state",
        ),
    ],
)
def test_a_nonrepairable_diagram_refuses(life, state, error, match):
    rbd = NonRepairableRBD(ONE, {"a": life})
    with pytest.raises(error, match=match):
        rbd.sf_given_state(1.0, {"a": state})


def test_a_wiener_level_below_its_start_is_taken():
    rbd = NonRepairableRBD(ONE, {"a": wiener(y0=3.0)})
    assert 0.0 < rbd.sf_given_state(5.0, {"a": NodeState(level=-5.0)}) <= 1.0


def test_a_composite_node_takes_no_own_model():
    standby = StandbyModel([E([1.0]), E([1.0])])
    with pytest.raises(ValueError, match="takes no state"):
        checked_override("a", standby, E([1.0]))


def test_a_regression_node_is_an_own_model_of_a_nonrepairable_node():
    aft = surv.ExponentialAFT.fit(
        np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5]),
        Z=np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]]),
    )
    hot = RegressionNode(aft, covariates=[2.0])
    assert checked_override("a", E([0.01]), hot) is hot
    repairable = unit(E([0.01]), E([1.0]))
    with pytest.raises(ValueError, match="cannot be the life of a repaired"):
        repairable.point_availability(1.0, state={"a": NodeState(model=hot)})


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"inspection": {"interval": 10.0}}, "tested for hidden failures"),
        (
            {
                "preventive": {
                    "policy": "condition",
                    "interval": 5.0,
                    "level": 7.0,
                }
            },
            "measured degradation level",
        ),
        (
            {
                "preventive": {
                    "policy": "condition",
                    "interval": 5.0,
                    "threshold": 0.1,
                }
            },
            "replaced on condition",
        ),
    ],
)
def test_a_repairable_diagram_refuses_what_is_not_worked_out(spec, match):
    rbd = unit(gamma(), E([2.0]), **spec)
    with pytest.raises(NotImplementedError, match=match):
        rbd.point_availability(1.0, state={"a": NodeState(level=6.0)})


def test_a_unit_run_part_of_the_time_takes_no_own_life():
    rbd = unit(W([100.0, 2.0]), E([2.0]), duty=0.5)
    state = {"a": NodeState(age=4.0, model=W([60.0, 3.0]))}
    with pytest.raises(NotImplementedError, match="operates part of the"):
        rbd.point_availability(1.0, state=state)


def test_a_probability_is_no_own_life_of_a_repaired_unit():
    rbd = unit(E([0.01]), E([1.0]))
    probability = surv.FixedEventProbability.from_params(0.1)
    with pytest.raises(ValueError, match="probability of failing"):
        rbd.point_availability(1.0, state={"a": NodeState(model=probability)})


def test_the_crews_chain_refuses_an_own_life_and_the_simulation_takes_it():
    pair = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(PARALLEL, {"a": pair, "b": pair}, repair_crews=1)
    state = {"a": NodeState(age=5.0, model=W([60.0, 3.0]))}
    with pytest.raises(NotImplementedError, match="repair crews"):
        rbd.point_availability(1.0, state=state)
    run = rbd.availability(20.0, mc_samples=20, seed=1, state=state)
    assert 0.0 < run.mean_availability <= 1.0


def test_the_first_failure_chain_refuses_an_own_life():
    pair = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(PARALLEL, {"a": pair, "b": pair})
    with pytest.raises(NotImplementedError, match="Markov chain"):
        rbd.first_failure_sf(
            1.0, state={"a": NodeState(age=5.0, model=W([60.0, 3.0]))}
        )


def test_a_joint_unit_is_the_first_life_of_a_repaired_component():
    # Its prediction from its readings is the unit in service; the units
    # that replace it are new, with the node's life.
    joint = joint_unit()
    rbd = unit(W([100.0, 2.0]))
    x = np.array([5.0, 20.0])
    state = {"a": NodeState(age=40.0, model=joint)}
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state), joint.sf(x), atol=1e-7
    )
    with pytest.raises(ValueError, match="would condition it again"):
        rbd.point_availability(x, state={"a": NodeState(age=1.0, model=joint)})


def test_a_life_from_a_level_is_the_process_from_there():
    process = gamma()
    life = FromLevel(process, 6.0)
    t = np.array([1.0, 4.0, 9.0])
    for name in ("sf", "ff", "df", "hf", "Hf"):
        np.testing.assert_array_equal(
            getattr(life, name)(t), getattr(process, name)(t, y0=6.0)
        )
    p = np.array([0.1, 0.5])
    np.testing.assert_array_equal(life.qf(p), process.qf(p, y0=6.0))
    assert life.mean() == float(process.mean(y0=6.0))
    assert life.random(3).shape == (3,)
    assert "level=6.0" in repr(life)


def test_a_repaired_component_checks_its_own_model():
    standby = NonRepairable(StandbyModel([E([1.0]), E([1.0])]), E([1.0]))
    with pytest.raises(ValueError, match="units take no state"):
        checked_override("a", standby, E([1.0]))
    plain = NonRepairable(E([1.0]), E([1.0]))
    with pytest.raises(ValueError, match="takes no state"):
        checked_override("a", plain, StandbyModel([E([1.0]), E([1.0])]))
    with pytest.raises(TypeError, match="itself"):
        checked_override("a", plain, surv.Weibull)
    assert checked_override("a", plain, gamma()).threshold == 10.0


def test_a_joint_unit_simulates_as_its_exact_values():
    rbd = unit(W([100.0, 2.0]), E([0.1]))
    state = {"a": NodeState(age=40.0, model=joint_unit())}
    run = rbd.availability(
        60.0, mc_samples=2000, seed=8, state=state, control_variate=False
    )
    interval = run.mean_availability_interval(0.9999)
    exact = rbd.mission_availability(60.0, state=state)
    assert interval.lower < exact < interval.upper
