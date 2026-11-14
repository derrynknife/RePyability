"""Standby arrangements that had no exact answer (#339): warm standby with
two or more units operating, cold standby of different units with three or
more operating, and spares that may fail to switch in or to start, with
any dormancy.

Exponential units are exact (a Markov chain), checked against closed
forms; the numerical methods that now take spares that may fail to start
are checked against the chain on exponential units and against
simulation on others; and the draws are checked to be reproducible."""

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import gamma

from repyability import NonRepairableRBD, RepairableRBD, StandbyModel
from repyability.rbd._dependent_lifetimes import (
    ColdPairSurvival,
    MayNotStart,
    RenewalStandbySurvival,
    WarmStandbySurvival,
    served_probabilities,
)
from repyability.rbd._markov_standby import (
    MAX_STATES,
    MarkovStandbySurvival,
    TooManyStates,
)
from repyability.rbd._sampling import row_sampler
from repyability.rbd.numerical_convolution import ConvolvedSurvival
from repyability.rbd.routes import model_route
from repyability.rbd.serialisation import deserialise_model, serialise_model

E = surv.Exponential.from_params
W = surv.Weibull.from_params
RATE = 0.01
UNIT = E([RATE])
PUMP = W([100.0, 2.0])
T = np.array([1e-3, 5.0, 50.0, 150.0, 400.0, 1500.0])


def chain_of(model: StandbyModel) -> MarkovStandbySurvival:
    survival = model._sf_model
    assert isinstance(survival, MarkovStandbySurvival)
    return survival


# -- the Markov chain against closed forms ---------------------------------


def test_the_chain_of_identical_cold_units_is_erlang():
    # Four units, two operating: the failures come at 2 * rate, and the
    # third ends the arrangement. Both tails to full relative precision.
    chain = MarkovStandbySurvival([RATE] * 4, 2, 0.0, [1.0] * 2, [1.0] * 2)
    np.testing.assert_allclose(
        chain.sf(T), gamma.sf(T, a=3, scale=50.0), rtol=1e-12
    )
    np.testing.assert_allclose(
        chain.ff(T), gamma.cdf(T, a=3, scale=50.0), rtol=1e-12
    )
    assert chain.mean() == pytest.approx(150.0, rel=1e-14)


def test_the_chain_of_warm_units_is_hypoexponential():
    chain = MarkovStandbySurvival([RATE] * 4, 2, 0.3, [1.0] * 2, [1.0] * 2)
    stages = surv.Hypoexponential.from_params(
        [RATE * (2 + j * 0.3) for j in (2, 1, 0)]
    )
    np.testing.assert_allclose(chain.sf(T), stages.sf(T), rtol=1e-10)
    assert chain.mean() == pytest.approx(float(stages.mean()), rel=1e-12)


def test_different_rates_in_cold_standby_add_up():
    # One operating, two units of different rates: hypoexponential.
    chain = MarkovStandbySurvival([0.01, 0.03], 1, 0.0, [1.0], [1.0])
    expected = (0.03 * np.exp(-0.01 * T) - 0.01 * np.exp(-0.03 * T)) / 0.02
    np.testing.assert_allclose(chain.sf(T), expected, rtol=1e-12)
    assert chain.mean() == pytest.approx(100.0 + 100.0 / 3.0, rel=1e-14)


@pytest.mark.parametrize("dormancy", [0.0, 0.25, 1.0])
@pytest.mark.parametrize(
    "s, p", [(1.0, 1.0), (0.9, 1.0), (1.0, 0.7), (0.8, 0.6)]
)
def test_one_spare_that_may_fail_to_switch_or_start(dormancy, s, p):
    # The spare is alive when needed at u with exp(-kappa rate u), then
    # runs with s p: R = e^-rt (1 + s p (1 - e^-kappa r t) / kappa), which
    # is e^-rt (1 + s p r t) cold.
    model = StandbyModel(
        [UNIT, UNIT],
        dormancy_factor=dormancy,
        switching_probability=s,
        start_probability=p,
    )
    rt = RATE * T
    if dormancy:
        gain = (1.0 - np.exp(-dormancy * rt)) / dormancy
    else:
        gain = rt
    expected = np.exp(-rt) * (1.0 + s * p * gain)
    np.testing.assert_allclose(model.sf(T), expected, rtol=1e-11)
    assert not model.is_simulated
    mean = (1.0 + s * p / (1.0 + dormancy)) / RATE
    assert model.mean() == pytest.approx(mean, rel=1e-12)


def test_a_spare_that_fails_to_start_is_passed_for_the_next():
    # Cold, one operating, three units: the second adds its life if its
    # switch works and it starts; the third is reached if the second's
    # switch works (started or not) and adds its life with s p.
    s, p = 0.9, 0.7
    model = StandbyModel(
        [UNIT] * 3, switching_probability=s, start_probability=p
    )
    assert model.mean() == pytest.approx(
        (1.0 + s * p + s * s * p) / RATE, rel=1e-12
    )
    # One run spare has Erlang(2), two Erlang(3).
    one = s * p * (1.0 - s * p) + s * (1.0 - p) * s * p
    two = s * p * s * p
    expected = (
        (1.0 - one - two) * np.exp(-RATE * T)
        + one * gamma.sf(T, a=2, scale=100.0)
        + two * gamma.sf(T, a=3, scale=100.0)
    )
    np.testing.assert_allclose(model.sf(T), expected, rtol=1e-11)


def test_two_operating_with_a_spare_that_may_not_start():
    # Cold, three identical units, two operating: the failures by t are
    # Poisson(2 rate t), and the arrangement survives one if the spare
    # switches in and starts.
    s, p = 0.95, 0.8
    model = StandbyModel(
        [UNIT] * 3, k=2, switching_probability=s, start_probability=p
    )
    expected = np.exp(-2 * RATE * T) * (1.0 + s * p * 2 * RATE * T)
    np.testing.assert_allclose(model.sf(T), expected, rtol=1e-11)
    # A tiny failure probability keeps its precision.
    assert float(model.ff(1e-3)) == pytest.approx(1.0 - expected[0], rel=1e-6)


def test_a_tiny_dormancy_is_exact_and_near_cold():
    # The hypoexponential stage rates are too close to separate: the chain
    # takes it, and it is all but cold.
    nearly = StandbyModel([UNIT] * 3, k=2, dormancy_factor=1e-12)
    assert isinstance(nearly._sf_model, MarkovStandbySurvival)
    cold = StandbyModel([UNIT] * 3, k=2)
    np.testing.assert_allclose(nearly.sf(T), cold.sf(T), rtol=1e-9)


def test_cold_standby_of_three_different_exponentials_is_exact():
    units = [E([0.01]), E([0.02]), E([0.015]), E([0.03]), E([0.01])]
    model = StandbyModel(units, k=3, start_probability=0.9)
    chain = chain_of(model)
    assert model_route(model) == ("exact", chain.how)
    assert_draws_agree(model)


def test_identical_units_share_states():
    chain = MarkovStandbySurvival(
        [RATE] * 40, 3, 0.5, [0.99] * 37, [0.95] * 37
    )
    assert chain._states == 38


def test_a_chain_too_large_is_simulated_saying_why():
    rates = [0.01 * (1 + j / 10) for j in range(12)]
    with pytest.raises(TooManyStates, match=f"more than {MAX_STATES}"):
        MarkovStandbySurvival(rates, 2, 0.5, [1.0] * 10, [1.0] * 10)
    model = StandbyModel([E([r]) for r in rates], k=2, dormancy_factor=0.5)
    assert model.is_simulated
    with pytest.raises(NotImplementedError, match="200 states"):
        model.sf(10.0)


# -- the numerical methods against the chain -------------------------------


def test_the_warm_recursion_takes_switching_and_starts():
    units = [E([0.01]), E([0.02]), E([0.015])]
    warm = WarmStandbySurvival(
        units, 0.4, switching=[0.9, 0.8], starting=[0.7, 0.95]
    )
    chain = MarkovStandbySurvival(
        [0.01, 0.02, 0.015], 1, 0.4, [0.9, 0.8], [0.7, 0.95]
    )
    np.testing.assert_allclose(warm.sf(T), chain.sf(T), atol=2e-5)
    np.testing.assert_allclose(warm.ff(T), chain.ff(T), atol=2e-5)


def test_the_convolution_takes_spares_that_may_not_start():
    units = [E([0.01]), MayNotStart(E([0.02]), 0.7), MayNotStart(UNIT, 0.5)]
    conv = ConvolvedSurvival(units, switching_probability=[0.9, 0.8])
    chain = MarkovStandbySurvival(
        [0.01, 0.02, 0.01], 1, 0.0, [0.9, 0.8], [0.7, 0.5]
    )
    np.testing.assert_allclose(conv.sf(T), chain.sf(T), atol=1e-6)


def test_the_renewal_counts_take_spares_that_may_not_start():
    renewal = RenewalStandbySurvival(
        UNIT, 5, 2, [0.9, 1.0, 0.8], start_probs=[0.6, 0.9, 0.7]
    )
    chain = MarkovStandbySurvival(
        [RATE] * 5, 2, 0.0, [0.9, 1.0, 0.8], [0.6, 0.9, 0.7]
    )
    np.testing.assert_allclose(renewal.sf(T), chain.sf(T), atol=1e-6)


def test_the_pair_recursion_takes_spares_that_may_not_start():
    units = [E([0.01]), E([0.02]), MayNotStart(E([0.015]), 0.6), UNIT]
    pair = ColdPairSurvival(units, [0.9, 0.95])
    chain = MarkovStandbySurvival(
        [0.01, 0.02, 0.015, 0.01], 2, 0.0, [0.9, 0.95], [0.6, 1.0]
    )
    np.testing.assert_allclose(pair.sf(T), chain.sf(T), atol=1e-3)


def test_served_probabilities():
    # Perfect starts: the products of the switches.
    np.testing.assert_allclose(
        served_probabilities([0.9, 0.8], [1.0, 1.0]), [1.0, 0.9, 0.72]
    )
    # Perfect switches: at least m of the spares start.
    p = 0.7
    np.testing.assert_allclose(
        served_probabilities([1.0, 1.0], [p, p]),
        [1.0, 1.0 - (1.0 - p) ** 2, p * p],
    )


# -- exact or numerical against simulation ---------------------------------


def assert_draws_agree(model, n=100_000, seed=11):
    draws = model.random(n, seed=seed)
    x = np.quantile(draws, [0.1, 0.5, 0.9])
    empirical = (draws[:, None] > x[None, :]).mean(axis=0)
    spread = np.sqrt(empirical * (1.0 - empirical) / n)
    assert np.all(
        np.abs(np.ravel(model.sf(x)) - empirical) < 4.5 * spread + 1e-3
    )
    mean = draws.mean()
    assert model.mean() == pytest.approx(
        mean, abs=4.5 * draws.std() / np.sqrt(n)
    )


AGREEING = {
    "warm, two operating, exponential": StandbyModel(
        [E([0.01]), E([0.02]), E([0.015]), E([0.01])],
        k=2,
        dormancy_factor=0.3,
        switching_probability=0.9,
        start_probability=0.8,
    ),
    "hot, two operating, may not start": StandbyModel(
        [UNIT] * 4, k=2, dormancy_factor=1.0, start_probability=0.7
    ),
    "warm, one operating, Weibull": StandbyModel(
        [PUMP, W([80.0, 1.5]), PUMP],
        dormancy_factor=0.4,
        switching_probability=0.9,
        start_probability=[0.8, 0.6],
    ),
    "cold, one operating, Weibull": StandbyModel(
        [PUMP, W([80.0, 1.5]), W([90.0, 3.0])],
        switching_probability=0.9,
        start_probability=0.8,
    ),
    "cold, identical, two operating": StandbyModel(
        [PUMP] * 4, k=2, switching_probability=0.95, start_probability=0.8
    ),
    "cold, different, two operating": StandbyModel(
        [PUMP, W([80.0, 1.5]), W([90.0, 3.0]), W([90.0, 3.0])],
        k=2,
        switching_probability=0.9,
        start_probability=[0.8, 0.6],
    ),
}


@pytest.mark.parametrize("name", list(AGREEING))
def test_exact_and_numerical_agree_with_simulation(name):
    model = AGREEING[name]
    assert not model.is_simulated
    assert_draws_agree(model)


# -- simulated, saying why -------------------------------------------------


@pytest.mark.parametrize(
    "model, case, why",
    [
        (
            lambda: StandbyModel([PUMP] * 3, k=2, dormancy_factor=0.5),
            "warm standby with 2 units operating",
            "exact for exponential units",
        ),
        (
            lambda: StandbyModel(
                [PUMP] * 3, k=2, dormancy_factor=1.0, start_probability=0.9
            ),
            "hot standby with 2 units operating",
            "always switch in and start",
        ),
        (
            lambda: StandbyModel([PUMP, W([80, 1.5]), PUMP, W([90, 3])], k=3),
            "cold standby with 3 different units operating",
            "numerical for identical units or two operating",
        ),
    ],
)
def test_what_is_simulated_says_why(model, case, why):
    standby = model()
    assert standby.is_simulated
    with pytest.raises(NotImplementedError) as refused:
        standby.sf(10.0)
    assert case in str(refused.value) and why in str(refused.value)
    # It is drawn all the same, reproducibly.
    a = standby.random(1000, seed=4)
    np.testing.assert_array_equal(a, standby.random(1000, seed=4))
    assert np.all(np.isfinite(a)) and np.all(a > 0.0)


# -- reproducibility -------------------------------------------------------


@pytest.mark.parametrize("name", list(AGREEING))
def test_one_draw_at_a_time_replays_the_batch(name):
    # An RBD draws a node's lifetimes as a block of uniforms, row by row
    # (its row sampler): the same as one random(1) call after another.
    model = AGREEING[name]
    sampler = row_sampler(model)
    assert sampler is not None
    state = np.random.get_state()
    try:
        np.random.seed(7)
        one_by_one = np.array([model.random(1)[0] for _ in range(300)])
        np.random.seed(7)
        block = sampler.draw(np.random.random_sample((300, sampler.width)))
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(one_by_one, block)


def test_seeded_draws_repeat():
    model = AGREEING["warm, one operating, Weibull"]
    np.testing.assert_array_equal(
        model.random(500, seed=3), model.random(500, seed=3)
    )
    assert not np.array_equal(
        model.random(500, seed=3), model.random(500, seed=4)
    )


# -- in diagrams, saved ----------------------------------------------------


def test_a_diagram_with_an_exact_chain_is_exact():
    model = AGREEING["warm, two operating, exponential"]
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": model, "b": E([0.001])}
    )
    assert rbd.is_analytically_solvable()
    routes = rbd.analysis_routes()
    assert routes["sf"].route == "exact"
    np.testing.assert_allclose(
        rbd.sf(T), model.sf(T) * np.exp(-0.001 * T), rtol=1e-12
    )


def test_a_repairable_diagram_takes_them():
    # As a life replaced whole: exact where the arrangement is, simulated
    # (reproducibly) where it is not.
    repair = E([0.5])
    for model in (
        AGREEING["warm, two operating, exponential"],
        StandbyModel(
            [PUMP] * 3, k=2, dormancy_factor=0.5, start_probability=0.9
        ),
    ):
        rbd = RepairableRBD(
            [("s", "a"), ("a", "t")],
            {"a": {"reliability": model, "repairability": repair}},
        )
        first = rbd.availability(100.0, mc_samples=200, seed=5)
        again = rbd.availability(100.0, mc_samples=200, seed=5)
        assert first.mean_availability == again.mean_availability
        assert 0.0 < first.mean_availability < 1.0


@pytest.mark.parametrize("p", [0.8, [0.8, 0.6]])
def test_a_start_probability_is_saved(p):
    model = StandbyModel(
        [PUMP, PUMP, W([80.0, 1.5])], dormancy_factor=0.3, start_probability=p
    )
    saved = serialise_model(model)
    assert saved["start_probability"] == p
    loaded = deserialise_model(saved)
    np.testing.assert_array_equal(loaded.sf(T), model.sf(T))
    # Spares that always start save as before.
    assert "start_probability" not in serialise_model(StandbyModel([PUMP] * 2))
