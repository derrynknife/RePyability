"""Standby and load-sharing arrangements without simulation (#135, #138,
#139): hot standby exactly as k-out-of-n, warm standby with one unit
operating by a recursion over its switch-ins, cold standby of identical
units with several operating by renewal counts (with imperfect switching),
and load sharing of identical units by a recursion over their failures.

Each is checked against a closed form where there is one (parallel and
k-out-of-n units, Erlang and hypoexponential lives, a two-unit integral,
the expected order statistics) and against a million simulated lifetimes
from the model's own sampler.
"""

import math
import warnings

import numpy as np
import pytest
import scipy.integrate as si
import surpyval as surv
from scipy import stats

from repyability import LoadSharingModel, NonRepairableRBD, StandbyModel
from repyability.rbd import routes
from repyability.rbd._dependent_lifetimes import (
    KOutOfNSurvival,
    LoadSharingSurvival,
    WarmStandbySurvival,
)
from repyability.rbd._model_utils import lfp_extras
from repyability.rbd.numerical_convolution import ConvolvedSurvival
from repyability.rbd.serialisation import rbd_from_dict, rbd_to_dict

W, E = surv.Weibull.from_params, surv.Exponential.from_params
LN = surv.LogNormal.from_params
PUMP = W([100.0, 2.0])
MILLION = 1_000_000


def agrees_with_simulation(model, n=MILLION, seed=7, sigmas=4.5):
    """The model's sf at its lifetimes' percentiles, and its mean, agree
    with ``n`` lifetimes from its own sampler."""
    lives = model.random(n, seed=seed)
    t = np.quantile(lives, [0.01, 0.1, 0.5, 0.9, 0.99])
    found = (lives[:, None] > t).mean(axis=0)
    z = (np.asarray(model.sf(t), float) - found) / np.sqrt(
        found * (1 - found) / n
    )
    assert np.all(np.abs(z) < sigmas), z
    error = lives.std() / math.sqrt(n)
    assert abs(model.mean() - lives.mean()) < sigmas * error


def aft(baseline, seed=0):
    rng = np.random.default_rng(seed)
    load = rng.uniform(0.5, 3.0, size=2000)
    if baseline == "exponential":
        x = rng.exponential(100.0, size=2000) / load**1.2
        return surv.ExponentialAFT.fit(x, Z=load.reshape(-1, 1))
    x = rng.weibull(2.0, size=2000) * 200.0 / load**1.5
    return surv.WeibullAFT.fit(x, Z=load.reshape(-1, 1))


# -- hot standby: k-out-of-n, exactly --------------------------------------


def test_hot_standby_is_k_out_of_n_of_any_units():
    a, b, c = PUMP, W([80.0, 1.5]), E([0.01])
    t = np.array([0.0, 20.0, 50.0, 150.0, 300.0])
    parallel = StandbyModel([a, b], dormancy_factor=1.0)
    np.testing.assert_allclose(
        parallel.sf(t), 1 - a.ff(t) * b.ff(t), rtol=1e-12, atol=1e-300
    )
    p, q, r = a.sf(t), b.sf(t), c.sf(t)
    two = StandbyModel([a, b, c], k=2, dormancy_factor=1.0)
    np.testing.assert_allclose(
        two.sf(t),
        p * q * r + p * q * (1 - r) + p * (1 - q) * r + (1 - p) * q * r,
        rtol=1e-13,
    )
    assert not two.is_simulated
    assert routes.model_route(two)[0] == routes.EXACT
    # A small probability of failing keeps its precision.
    assert parallel.ff(1e-3) == pytest.approx(a.ff(1e-3) * b.ff(1e-3), 1e-12)
    # The mean is the area under the exact reliability.
    area = si.quad(lambda x: float(two.sf(x)), 0, np.inf, limit=200)[0]
    assert two.mean() == pytest.approx(area, rel=1e-8)
    assert routes.mean_route(two)[0] == routes.NUMERICAL
    agrees_with_simulation(two)


# -- warm standby, one unit operating ----------------------------------------


def two_unit_warm(first, second, kappa, x):
    """R(x) = S1(x) + integral of f1(u) S2(x - (1 - kappa) u) from 0 to x."""
    return (
        first.sf(x)
        + si.quad(
            lambda u: first.df(u) * second.sf(kappa * u + x - u),
            0,
            x,
            epsabs=1e-14,
            epsrel=1e-12,
            limit=500,
        )[0]
    )


@pytest.mark.parametrize(
    "first, second",
    [(PUMP, PUMP), (E([0.01]), E([0.01])), (PUMP, LN([4.0, 0.8]))],
    ids=["weibull", "exponential", "weibull then lognormal"],
)
def test_a_warm_pair_is_its_integral(first, second):
    x = np.array([5.0, 20.0, 50.0, 150.0, 300.0, 600.0])
    warm = WarmStandbySurvival([first, second], 0.3)
    expected = [two_unit_warm(first, second, 0.3, v) for v in x]
    np.testing.assert_allclose(warm.sf(x), expected, atol=1e-4)
    # Survival and failure are worked out apart, and add to one.
    np.testing.assert_allclose(warm._sf + warm._ff, 1.0, atol=1e-12)


def test_the_issues_numbers():
    # #135's example: the guide's warm.sf(150) was 0.4735, the exact 0.4815.
    t = np.array([50.0, 150.0, 300.0])
    warm = StandbyModel([PUMP, PUMP], dormancy_factor=0.3)
    np.testing.assert_allclose(warm.sf(t), [0.9830, 0.4815, 0.0098], atol=1e-4)
    assert not warm.is_simulated
    assert routes.model_route(warm)[0] == routes.NUMERICAL


def test_warm_standby_tends_to_cold_and_matches_exponential_stages():
    x = np.array([20.0, 80.0, 200.0])
    nearly_cold = WarmStandbySurvival([PUMP, W([80.0, 1.5])], 1e-12)
    cold = ConvolvedSurvival([PUMP, W([80.0, 1.5])])
    np.testing.assert_allclose(nearly_cold.sf(x), cold.sf(x), atol=5e-5)
    # Three identical exponential units: hypoexponential stages.
    three = WarmStandbySurvival([E([0.01])] * 3, 0.4)
    stages = surv.Hypoexponential.from_params(
        [0.01 * (1 + 2 * 0.4), 0.01 * (1 + 0.4), 0.01]
    )
    np.testing.assert_allclose(three.sf(x), stages.sf(x), atol=5e-5)
    assert three.mean() == pytest.approx(stages.mean(), rel=1e-4)


def test_warm_standby_of_different_units_agrees_with_simulation():
    model = StandbyModel(
        [PUMP, LN([4.2, 0.6]), E([0.012])], dormancy_factor=0.4
    )
    assert not model.is_simulated
    agrees_with_simulation(model)


def test_a_warm_spare_that_never_fails_keeps_the_arrangement_going():
    lasting = surv.Weibull.from_params([100.0, 2.0], **lfp_extras(0.8))
    model = StandbyModel([PUMP, lasting], dormancy_factor=0.5)
    # The spare is a never-failing unit with probability 0.2.
    assert model.sf(1e6) == pytest.approx(0.2, abs=1e-4)
    assert model.mean() == np.inf


# -- cold standby, several operating -----------------------------------------


def test_identical_cold_units_operating_together_are_renewal_counts():
    x = np.array([20.0, 80.0, 200.0, 400.0])
    # Exponential: two at a time, two spares, is Erlang(3, 2 * rate). The
    # closed form is used for those, so ask for the renewal counts directly.
    from repyability.rbd._dependent_lifetimes import RenewalStandbySurvival

    counts = RenewalStandbySurvival(E([0.01]), 4, 2)
    np.testing.assert_allclose(
        counts.sf(x), stats.gamma.sf(x, a=3, scale=50.0), atol=1e-6
    )
    # As many operating as units: a series of them.
    series = StandbyModel([PUMP, PUMP], k=2)
    np.testing.assert_allclose(series.sf(x), PUMP.sf(x) ** 2, atol=1e-8)
    model = StandbyModel([PUMP] * 4, k=2)
    assert not model.is_simulated
    assert routes.model_route(model)[0] == routes.NUMERICAL
    agrees_with_simulation(model)


@pytest.mark.parametrize(
    "units, k, switching",
    [([PUMP] * 4, 2, 0.9), ([LN([4.2, 0.6])] * 5, 3, [0.9, 0.8])],
    ids=["two of four", "three of five, one per spare"],
)
def test_cold_standby_with_imperfect_switching(units, k, switching):
    model = StandbyModel(units, k=k, switching_probability=switching)
    assert not model.is_simulated
    agrees_with_simulation(model)
    perfect = StandbyModel(units, k=k)
    t = np.array([50.0, 100.0])
    assert np.all(model.sf(t) < perfect.sf(t))


def test_the_switching_probabilities_are_one_per_spare():
    with pytest.raises(ValueError, match="length 2"):
        StandbyModel([PUMP] * 4, k=2, switching_probability=[0.9, 0.9, 0.9])
    with pytest.raises(NotImplementedError, match="cold standby"):
        StandbyModel(
            [PUMP] * 2, dormancy_factor=0.5, switching_probability=0.9
        )


def test_the_two_samplers_draw_alike_under_imperfect_switching(monkeypatch):
    model = StandbyModel([PUMP] * 4, k=2, switching_probability=0.8)
    batched = model.random(500, seed=3)
    monkeypatch.setattr(StandbyModel, "_row_sampler", lambda self: None)
    one_at_a_time = model.random(500, seed=3)
    np.testing.assert_array_equal(batched, one_at_a_time)


@pytest.mark.parametrize(
    "units, switching",
    [
        ([PUMP, LN([4.2, 0.6]), E([0.012]), PUMP], 1.0),
        ([PUMP, LN([4.2, 0.6]), E([0.012]), PUMP], [0.9, 0.7]),
        ([LN([4.2, 0.6]), PUMP, E([0.012])], 1.0),
    ],
    ids=["four", "four, switching", "three"],
)
def test_two_different_cold_units_operating(units, switching):
    model = StandbyModel(units, k=2, switching_probability=switching)
    assert not model.is_simulated
    assert routes.model_route(model)[0] == routes.NUMERICAL
    agrees_with_simulation(model)


def test_two_operating_of_one_kind_by_the_pair_recursion_too():
    # The recursion for different units, given identical ones, gives what
    # the renewal counts do; and two units operating with no spare are a
    # series pair.
    from repyability.rbd._dependent_lifetimes import (
        ColdPairSurvival,
        RenewalStandbySurvival,
    )

    x = np.array([20.0, 50.0, 100.0, 150.0, 250.0])
    pair = ColdPairSurvival([PUMP] * 4)
    np.testing.assert_allclose(
        pair.sf(x), RenewalStandbySurvival(PUMP, 4, 2).sf(x), atol=5e-5
    )
    exponential = ColdPairSurvival([E([0.01])] * 4)
    np.testing.assert_allclose(
        exponential.sf(x), stats.gamma.sf(x, a=3, scale=50.0), atol=2e-4
    )
    series = ColdPairSurvival([PUMP, E([0.01])])
    np.testing.assert_allclose(
        series.sf(x), PUMP.sf(x) * E([0.01]).sf(x), atol=2e-4
    )


def test_three_different_cold_units_operating_are_only_simulated():
    model = StandbyModel([PUMP, W([80.0, 1.5]), PUMP, W([90.0, 3.0])], k=3)
    assert model.is_simulated
    assert routes.model_route(model)[0] == routes.REFUSED
    with pytest.raises(NotImplementedError, match="3 different units"):
        model.sf(10.0)


# -- load sharing of identical units -----------------------------------------


@pytest.mark.parametrize(
    "baseline", [PUMP, E([0.01]), LN([4.0, 0.8]), W([100.0, 0.6])]
)
@pytest.mark.parametrize("n, k", [(3, 1), (4, 2)])
def test_without_a_load_effect_sharing_is_k_out_of_n(baseline, n, k):
    x = np.array([0.5, 10.0, 30.0, 60.0, 100.0, 150.0, 300.0])
    sharing = LoadSharingSurvival(baseline, [1.0] * n, n, k)
    exact = KOutOfNSurvival([baseline] * n, k)
    np.testing.assert_allclose(sharing.sf(x), exact.sf(x), atol=5e-4)


def test_exponential_units_sharing_a_load_are_hypoexponential():
    x = np.array([5.0, 20.0, 50.0, 100.0, 150.0, 300.0])
    sharing = LoadSharingSurvival(E([0.01]), [1.0, 2.0, 5.0], 3, 1)
    stages = surv.Hypoexponential.from_params([0.03, 0.04, 0.05])
    np.testing.assert_allclose(sharing.sf(x), stages.sf(x), atol=5e-4)
    assert sharing.mean() == pytest.approx(stages.mean(), rel=1e-3)


def test_weibull_units_sharing_a_load():
    unit = aft("weibull")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        group = LoadSharingModel([unit] * 4, load=4.0, k=2)
    assert not group.is_simulated
    assert routes.model_route(group)[0] == routes.NUMERICAL
    # The mean exactly: the failures' exposures are the baseline lives'
    # order statistics, each spacing run at its stage's rate.
    base = group._baselines[0]
    phis = [group._phi_table[0, s - 1] for s in (4, 3, 2)]

    def order_statistic_mean(i, n=4):
        c = math.factorial(n) / (math.factorial(i - 1) * math.factorial(n - i))
        return si.quad(
            lambda v: v
            * c
            * base.ff(v) ** (i - 1)
            * base.sf(v) ** (n - i)
            * base.df(v),
            0,
            np.inf,
            epsabs=1e-12,
            limit=500,
        )[0]

    means = [0.0] + [order_statistic_mean(i) for i in (1, 2, 3)]
    exact = sum((means[i] - means[i - 1]) / phis[i - 1] for i in (1, 2, 3))
    assert group.mean() == pytest.approx(exact, rel=5e-4)
    agrees_with_simulation(group)


def test_different_units_sharing_a_load_are_only_simulated():
    group = LoadSharingModel([aft("weibull"), aft("exponential")], load=2.0)
    assert group.is_simulated
    with pytest.raises(NotImplementedError, match="units that are different"):
        group.sf(10.0)


# -- in a diagram, and saved -------------------------------------------------


def test_they_reload_exactly_and_are_not_simulated_nodes():
    nodes = {
        "warm": StandbyModel([PUMP, PUMP], dormancy_factor=0.3),
        "hot": StandbyModel([PUMP, W([80.0, 1.5])], dormancy_factor=1.0),
        "cold": StandbyModel([PUMP] * 3, k=2, switching_probability=0.9),
    }
    rbd = NonRepairableRBD(
        [("s", "warm"), ("warm", "hot"), ("hot", "cold"), ("cold", "t")],
        nodes,
    )
    assert rbd.is_analytically_solvable()
    restored = rbd_from_dict(rbd_to_dict(rbd))
    x = np.array([10.0, 50.0, 100.0])
    np.testing.assert_array_equal(restored.sf(x), rbd.sf(x))
    assert rbd.analysis_routes()["sf"].route == routes.NUMERICAL
