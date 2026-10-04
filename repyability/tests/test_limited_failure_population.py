"""Tests for limited-failure-population and zero-inflated node models.

A surpyval model with ``p < 1`` describes a population in which a fraction
``1 - p`` of units never fail; one with ``f0 > 0``, a fraction ``f0`` dead
on arrival (failed at 0). Their ``random`` returns survival data rather
than lifetimes, so the simulations draw their lifetimes by the quantile
function: infinite for a unit that never fails, 0 for one dead on arrival.

The references are exact: the fractions that never fail or are dead on
arrival, closed forms for sums of such exponential lifetimes (cold
standby), the geometric number of failures of a repaired unit before a
replacement that never fails, and the absorption probabilities of a unit
that ends up, or down, for good.
"""

import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import (
    RBD,
    NonRepairable,
    NonRepairableRBD,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
)
from repyability.rbd._model_utils import (
    is_exponential,
    lfp_p,
    model_mean,
    never_fails,
)
from repyability.rbd._sampling import inverse_sampler
from repyability.rbd._streams import DURATION, FAILURE, REPAIR
from repyability.rbd.numerical_convolution import ConvolvedSurvival
from repyability.tests.keyed_draws import KeyedDraws

W = surv.Weibull.from_params
E = surv.Exponential.from_params

LFP = W([100, 2], lfp_p=0.9)
ZI = W([100, 2], f0=0.1)
BOTH = W([100, 2], lfp_p=0.9, f0=0.1)
MODELS = {"lfp": LFP, "zi": ZI, "both": BOTH}


def within(estimate, exact, n, z=4.0):
    """A proportion from ``n`` draws is within ``z`` standard errors."""
    se = math.sqrt(max(exact * (1 - exact), 1e-12) / n)
    return abs(estimate - exact) <= z * se + 1e-12


def erlang2_ff(rate, t):
    return 1 - math.exp(-rate * t) * (1 + rate * t)


# -- the helpers ------------------------------------------------------------


def test_model_helpers():
    assert W([100, 2]).extras == {}
    assert LFP.extras == {"lfp_p": 0.9}
    assert BOTH.extras == {"lfp_p": 0.9, "f0": 0.1}
    assert lfp_p(LFP) == 0.9 and lfp_p(W([100, 2])) == 1
    assert W([100, 2], gamma=5.0).extras == {"gamma": 5.0}
    assert never_fails(LFP) == pytest.approx(0.1)
    assert never_fails(ZI) == 0.0 and never_fails(E([0.1])) == 0.0
    assert model_mean(LFP) == math.inf
    base = W([100, 2]).mean()
    assert model_mean(ZI) == pytest.approx(0.9 * base, rel=1e-12)
    assert model_mean(W([100, 2])) == pytest.approx(base, rel=1e-12)
    assert is_exponential(E([0.1]))
    for model in (
        E([0.1], lfp_p=0.9),
        E([0.1], f0=0.1),
        E([0.1], gamma=1.0),
    ):
        assert not is_exponential(model)


@pytest.mark.parametrize("name", MODELS)
def test_draws_follow_the_model(name):
    model = MODELS[name]
    np.random.seed(1)
    x = model.random(50_000)
    p, f0 = float(lfp_p(model)), float(model.f0)
    assert within(np.mean(np.isinf(x)), 1 - p, len(x))
    assert within(np.mean(x == 0.0), f0, len(x))
    for t in (30.0, 100.0, 200.0):
        assert within(np.mean(x > t), float(model.sf(t)), len(x))
    # One global uniform per draw, as the batched path takes them.
    np.random.seed(2)
    model.random(7)
    after = np.random.random_sample()
    np.random.seed(2)
    np.random.random_sample(7)
    assert np.random.random_sample() == after
    np.random.seed(3)
    batched = inverse_sampler(model)(np.random.random_sample(7))
    np.random.seed(3)
    np.testing.assert_array_equal(batched, model.random(7))


def test_a_plain_model_draws_as_surpyval_does():
    model = W([100, 2])
    np.random.seed(4)
    ours = inverse_sampler(model)(np.random.random_sample(5))
    np.random.seed(4)
    np.testing.assert_array_equal(ours, model.random(5))


# -- NonRepairableRBD -------------------------------------------------------


def series(model):
    return NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": model, "b": W([300, 1.5])}
    )


def parallel(model):
    return NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": model, "b": W([300, 1.5])},
    )


@pytest.mark.parametrize("name", MODELS)
def test_simulated_lifetimes_match_the_reliability(name):
    for rbd in (series(MODELS[name]), parallel(MODELS[name])):
        assert rbd._row_sampler() is not None
        lifetimes = rbd.random(40_000, seed=5)
        n = len(lifetimes)
        for t in (0.0, 50.0, 150.0, 400.0):
            assert within(np.mean(lifetimes > t), float(rbd.sf(t)), n)
        # The lifetimes that never end: the system's reliability's floor.
        assert within(np.mean(np.isinf(lifetimes)), float(rbd.sf(1e12)), n)


@pytest.mark.parametrize("name", MODELS)
def test_the_event_loop_agrees(name, monkeypatch):
    rbd = series(MODELS[name])
    batched = rbd.random(2_000, seed=6)
    monkeypatch.setattr(rbd, "_row_sampler", lambda: None)
    np.testing.assert_array_equal(batched, rbd.random(2_000, seed=6))


@pytest.mark.parametrize("name", MODELS)
def test_mttf(name):
    model = MODELS[name]
    # In series with a unit that always fails, the system fails: the MTTF
    # is the integral of its reliability.
    rbd = series(model)
    interval = rbd.mean_time_to_failure_interval(mc_samples=40_000, seed=7)
    exact = quad(rbd.sf, 0, np.inf)[0]
    assert abs(interval.estimate - exact) < 4 * interval.standard_error
    # In parallel with units that may never fail, it may never fail: an
    # infinite MTTF, and a run to a tolerance stops at once.
    if lfp_p(model) < 1:
        assert parallel(model).mean() == math.inf
        simulated = parallel(model).mean(
            method="simulate", mc_samples=2_000, seed=8
        )
        assert simulated == math.inf
        stopped = parallel(model).mean_time_to_failure_interval(
            mc_samples=200, seed=8, tolerance=1.0
        )
        assert stopped.n_samples == 200


def test_node_mttf():
    # Not surpyval's defective mean, 0.9 * 88.6, for a unit that may never
    # fail; a unit dead on arrival has lifetime 0.
    assert series(LFP).node_mttf()["a"] == math.inf
    assert series(ZI).node_mttf()["a"] == pytest.approx(
        0.9 * W([100, 2]).mean(), rel=1e-12
    )
    assert series(BOTH).node_mttf()["a"] == math.inf


# -- composite nodes --------------------------------------------------------


@pytest.mark.parametrize(
    "p, f0",
    [(0.8, 0.0), (1.0, 0.2), (0.8, 0.2), (0.6, 0.3)],
)
def test_cold_standby_of_exponentials_with_extras(p, f0):
    # Each unit is 0 with probability f0, Exponential(rate) with p - f0,
    # and never fails otherwise; the pair's sum is 0, Exponential or
    # Erlang(2) as zero, one or two units are continuous, or infinite.
    rate = 0.1
    unit = E([rate], lfp_p=p, f0=f0) if f0 else E([rate], lfp_p=p)
    c = p - f0
    for pair in (StandbyModel([unit, unit]), RepeatedStandbyNode(unit, 2)):
        for t in (0.0, 5.0, 20.0, 60.0):
            ff = (
                f0**2
                + 2 * f0 * c * (1 - math.exp(-rate * t))
                + c**2 * erlang2_ff(rate, t)
            )
            assert float(np.ravel(pair.sf(t))[0]) == pytest.approx(
                1 - ff, abs=2e-4
            )
        assert float(np.ravel(pair.sf(1e9))[0]) == pytest.approx(
            1 - p**2, abs=1e-12
        )
        expected = math.inf if p < 1 else 2 * c / rate
        assert float(np.ravel(pair.mean())[0]) == pytest.approx(
            expected, rel=1e-3
        )


def test_the_convolution_keeps_the_dead_on_arrival_apart():
    # A unit dead on arrival adds nothing: the sum of two is the other's
    # lifetime (or 0) as often as one of them is dead on arrival. The
    # convolution takes them from the units' CDFs, which start at f0.
    unit = E([0.1], f0=0.2)
    assert float(unit.ff(0.0)) == pytest.approx(0.2)
    pair = ConvolvedSurvival([unit, unit])
    for t in (0.5, 5.0, 20.0, 60.0):
        both = 1 - erlang2_ff(0.1, t)
        one = math.exp(-0.1 * t)
        assert float(pair.sf(t)) == pytest.approx(
            0.8**2 * both + 2 * 0.2 * 0.8 * one, abs=1e-7
        )


def test_cold_standby_with_imperfect_switching():
    rate, p, s = 0.1, 0.8, 0.7
    unit = E([rate], lfp_p=p)
    pair = StandbyModel([unit, unit], switching_probability=s)
    for t in (5.0, 20.0, 60.0):
        one = 1 - p * (1 - math.exp(-rate * t))
        two = 1 - p**2 * erlang2_ff(rate, t)
        assert float(pair.sf(t)) == pytest.approx(
            (1 - s) * one + s * two, abs=2e-4
        )
    assert float(pair.sf(1e9)) == pytest.approx(
        (1 - s) * (1 - p) + s * (1 - p**2), abs=1e-12
    )


def test_standby_of_plain_exponentials_keeps_its_closed_form():
    unit = E([0.1])
    pair = StandbyModel([unit, unit])
    assert float(pair.sf(20.0)) == pytest.approx(
        1 - erlang2_ff(0.1, 20.0), rel=1e-12
    )


def test_warm_standby_that_may_never_fail():
    unit = W([100, 2], lfp_p=0.9)
    node = StandbyModel([unit, unit, unit], dormancy_factor=0.5)
    x = node.random(20_000, seed=9)
    assert not np.isnan(x).any()
    # It never fails if any of its units never fails.
    assert within(np.mean(np.isinf(x)), 1 - 0.9**3, len(x))
    assert node.mean() == math.inf
    # The rows with only finite budgets are played out all at once, as
    # before; the others one by one.
    np.random.seed(10)
    budgets = np.column_stack([unit.random(300) for _ in range(3)])
    finite = np.all(np.isfinite(budgets), axis=1)
    assert 0 < finite.sum() < 300
    together = node._warm_from_budgets(budgets)
    np.testing.assert_array_equal(
        together[finite], node._warm_lifetimes(budgets[finite])
    )
    np.testing.assert_array_equal(
        together, node._warm_lifetimes_by_sample(budgets)
    )


def test_cold_k_out_of_n_that_may_never_fail():
    # Two of three operating: it never fails if two units never fail.
    p = 0.9
    node = StandbyModel([W([100, 2], lfp_p=p)] * 3, k=2)
    never = 3 * (1 - p) ** 2 * p + (1 - p) ** 3
    x = node.random(40_000, seed=11)
    assert within(np.mean(np.isinf(x)), never, len(x))


def test_repeated_standby_draws():
    # The sum of two lifetimes is finite only if both are.
    node = RepeatedStandbyNode(W([100, 2], lfp_p=0.8), 2)
    x = node.random(40_000, seed=16)
    assert within(np.mean(np.isinf(x)), 1 - 0.8**2, len(x))
    for t in (100.0, 250.0):
        assert within(np.mean(x > t), float(node.sf(t)), len(x))


@pytest.mark.parametrize("kind", ["parallel", "series"])
def test_repeated_node(kind):
    p = 0.9
    node = RepeatedNode(W([100, 2], lfp_p=p), 3, kind)
    x = node.random(40_000, seed=12)
    never = 1 - p**3 if kind == "parallel" else (1 - p) ** 3
    assert within(np.mean(np.isinf(x)), never, len(x))
    for t in (50.0, 150.0):
        assert within(np.mean(x > t), float(node.sf(t)), len(x))


# -- repairable components ----------------------------------------------------


def repairable(reliability, repair=None):
    return RepairableRBD(
        [("s", "c"), ("c", "t")],
        {
            "c": {
                "reliability": reliability,
                "repairability": repair or E([1.0]),
            }
        },
    )


def test_a_unit_that_may_never_fail_ends_up_for_good():
    rbd = repairable(W([10, 2], lfp_p=0.9))
    assert rbd.mean_availability() == 1.0
    assert rbd.system_failure_frequency() == 0.0
    # Each replacement never fails with probability 0.1: the number of
    # failures is geometric, with mean 0.9 / 0.1.
    result = rbd.availability(2_000.0, mc_samples=2_000, seed=13)
    failures = result.system_failures / result.n_simulations
    se = math.sqrt(0.9 / 0.1**2 / result.n_simulations)
    assert abs(failures - 9.0) < 4 * se


def test_long_run_availability_with_absorbing_ends():
    # A unit never fails with probability u = 0.2; a replacement never
    # finishes with probability d = 0.3: the unit ends up for good with
    # probability u / (u + (1 - u) d).
    unit = NonRepairable(W([10, 2], lfp_p=0.8), E([1.0], lfp_p=0.7))
    exact = 0.2 / (0.2 + 0.8 * 0.3)
    assert unit.mean_availability() == pytest.approx(exact, rel=1e-12)
    assert unit.failure_frequency() == 0.0
    down = NonRepairable(W([10, 2]), E([1.0], lfp_p=0.7))
    assert down.mean_availability() == 0.0
    rbd = repairable(W([10, 2], lfp_p=0.8), E([1.0], lfp_p=0.7))
    assert rbd.mean_availability() == pytest.approx(exact, rel=1e-12)
    # Over a long window, the fraction of it up is close to that (the
    # time before the unit settles is short).
    result = rbd.availability(
        5_000.0, mc_samples=2_000, seed=14, control_variate=False
    )
    window = result.mean_availability_interval()
    assert abs(window.estimate - exact) < 4 * window.standard_error + 0.01


def test_repairable_draws_come_from_their_streams():
    # A limited-failure-population or zero-inflated lifetime (infinite for
    # a unit that never fails, 0 for one dead on arrival) comes from the
    # component's own stream like any other draw: each simulation's up time
    # is what its component's draws make it.
    rbd = repairable(W([10, 2], lfp_p=0.9, f0=0.05))
    window = 200.0
    result = rbd.availability(window, mc_samples=60, seed=15)
    draws = KeyedDraws(rbd, window, 15)
    never, dead = 0, 0
    for r in range(60):
        t, up, uptime = 0.0, True, 0.0
        while t < window:
            delay = draws.next(("c",), FAILURE if up else REPAIR, r)
            if up:
                uptime += min(t + delay, window) - t
                never += delay == np.inf
                dead += delay == 0.0
            t, up = t + delay, not up
        assert result.uptimes[r] == pytest.approx(uptime, rel=1e-12)
    assert never and dead
    # A maintenance time dead on arrival (done at once) has a stream too.
    maintained = RepairableRBD(
        [("s", "c"), ("c", "t")],
        {
            "c": {
                "reliability": W([70, 2]),
                "repairability": E([0.8]),
                "preventive": {
                    "interval": 30.0,
                    "duration": W([2, 1.5], f0=0.2),
                },
            }
        },
    )
    specs, complete = maintained._stream_specs(200.0)
    assert complete and (("c",), DURATION) in specs
    assert maintained.availability(
        200.0, mc_samples=100, seed=17
    ).n_simulations
    # So antithetic pairs and common random numbers work with them.
    assert rbd.compare(rbd, 200.0, mc_samples=50, seed=1).estimate == 0.0
    assert rbd.availability(
        200.0, mc_samples=20, seed=1, antithetic=True
    ).antithetic


# -- saving and sensitivity ---------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [LFP, ZI, BOTH, W([100, 2], gamma=5.0), E([0.01], lfp_p=0.7)],
)
def test_saving_keeps_the_extras(model):
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": model})
    back = RBD.from_json(rbd.to_json())
    t = np.array([0.0, 4.0, 50.0, 1e9])
    np.testing.assert_allclose(back.sf(t), rbd.sf(t), rtol=0, atol=1e-15)
    assert back.reliabilities["c"].extras == model.extras


def test_models_are_saved_in_surpyval_format():
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": BOTH})
    (entry,) = rbd.to_dict()["reliabilities"]
    assert entry["model"] == {"kind": "surpyval", "model": BOTH.to_dict()}


@pytest.mark.parametrize(
    "extras", [{}, {"gamma": 5.0, "p": 0.9, "f0": 0.1}], ids=["plain", "all"]
)
def test_files_saved_before_0_10_still_load(extras):
    saved = {"kind": "parametric", "dist": "Weibull", "params": [100.0, 2.0]}
    if extras:
        saved["extras"] = extras
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": W([100, 2])})
    data = rbd.to_dict()
    data["reliabilities"][0]["model"] = saved
    back = RBD.from_dict(data)
    t = np.array([0.0, 4.0, 50.0, 1e9])
    named = dict(extras)
    if "p" in named:
        named["lfp_p"] = named.pop("p")
    np.testing.assert_allclose(
        back.sf(t),
        NonRepairableRBD(
            [("s", "c"), ("c", "t")], {"c": W([100, 2], **named)}
        ).sf(t),
        rtol=0,
        atol=1e-15,
    )


def test_sensitivity_keeps_the_extras():
    rbd = NonRepairableRBD([("s", "c"), ("c", "t")], {"c": BOTH})
    sens = rbd.parameter_sensitivity(50.0)
    h = 1e-3

    def sf(alpha, beta):
        return float(W([alpha, beta], lfp_p=0.9, f0=0.1).sf(50.0))

    alpha = (sf(100 + h, 2) - sf(100 - h, 2)) / (2 * h)
    beta = (sf(100, 2 + h * 0.02) - sf(100, 2 - h * 0.02)) / (2 * h * 0.02)
    assert sens["c"]["alpha"] == pytest.approx(alpha, rel=1e-5)
    assert sens["c"]["beta"] == pytest.approx(beta, rel=1e-5)


# -- mean() is infinite when p < 1 (surpyval#404) ----------------------------
#
# Nothing here may take a time scale from it: these tests give the model an
# infinite mean() whatever surpyval does.


def _infinite_mean(model, monkeypatch):
    monkeypatch.setattr(model, "mean", lambda *a, **k: np.inf)
    return model


def test_failure_time_scale():
    from repyability.rbd._model_utils import failure_time_scale

    assert failure_time_scale(W([100, 2])) == pytest.approx(88.6227, rel=1e-5)
    # Some units never fail: the mean of those that fail, with the offset.
    lfp = W([100, 2], lfp_p=0.8, f0=0.1, gamma=5.0)
    assert failure_time_scale(lfp) == pytest.approx(93.6227, rel=1e-5)
    assert math.isnan(failure_time_scale(StandbyModel([lfp, lfp])))


def test_cold_standby_when_the_mean_is_infinite(monkeypatch):
    rate, p = 0.1, 0.8
    unit = _infinite_mean(E([rate], lfp_p=p), monkeypatch)
    pair = StandbyModel([unit, unit])
    # The pair fails only if both units do: sf = 1 - p^2 * Erlang(2) ff.
    for t in (5.0, 20.0, 60.0):
        expected = 1 - p**2 * erlang2_ff(rate, t)
        assert float(np.ravel(pair.sf(t))[0]) == pytest.approx(
            expected, abs=2e-4
        )


def test_no_replacement_pays_when_units_may_never_fail(monkeypatch):
    # In the long run a unit that never fails is kept for good, so running
    # to failure costs nothing per unit time, whatever the wear-out.
    unit = NonRepairable(
        _infinite_mean(W([1000, 2.5], lfp_p=0.9), monkeypatch)
    )
    unit.set_costs_planned_and_unplanned(cp=1, cu=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert unit.find_optimal_replacement() == np.inf
