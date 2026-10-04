"""Common-cause groups over a lifetime (#132): the warning when a group that
splits a failure probability is used beyond the small probabilities it is
meant for, and the groups that split the failure rate instead
(``basis="rate"``), which keep each member's own life distribution, hold
over the whole life, and enter the exact MTTF and the simulations.
"""

import itertools
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy import stats

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    NonRepairableRBD,
    PerfectReliability,
)
from repyability.rbd.serialisation import rbd_from_dict, rbd_to_dict
from repyability.tests.test_performance_equivalence import binomial_first

E = surv.Exponential.from_params([0.01])  # MTTF 100
W = surv.Weibull.from_params([100.0, 2.0])
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
VOTE = [
    ("s", "a"),
    ("s", "b"),
    ("s", "c"),
    ("a", "v"),
    ("b", "v"),
    ("c", "v"),
    ("v", "t"),
]
TIMES = np.array([10.0, 50.0, 100.0, 300.0, 10_000.0])


def pair(model, unit=E) -> NonRepairableRBD:
    return NonRepairableRBD(
        PAIR,
        {"a": unit, "b": unit},
        ccf_groups=[CCFGroup(["a", "b"], model)],
    )


def two_of_three(model, unit=W) -> NonRepairableRBD:
    return NonRepairableRBD(
        VOTE,
        {"a": unit, "b": unit, "c": unit, "v": PerfectReliability},
        k={"v": 2},
        ccf_groups=[CCFGroup(["a", "b", "c"], model)],
    )


def test_the_rate_split_answers_the_issue():
    independent = NonRepairableRBD(PAIR, {"a": E, "b": E})
    np.testing.assert_allclose(
        independent.sf(TIMES),
        [0.99094, 0.84518, 0.60042, 0.09710, 0.0],
        atol=5e-6,
    )
    with pytest.warns(UserWarning):
        probability = pair(BetaFactor(0.2)).sf(TIMES)
    # The probability split: more reliable than independence at the MTTF,
    # and 29% of systems never fail.
    np.testing.assert_allclose(
        probability, [0.97528, 0.83002, 0.65018, 0.34192, 0.288], atol=5e-6
    )
    rate = pair(BetaFactor(0.2, basis="rate")).sf(TIMES)
    np.testing.assert_allclose(
        rate, [0.97440, 0.80649, 0.57046, 0.09506, 0.0], atol=5e-6
    )
    # R^beta (2 R^(1-beta) - R^(2(1-beta))), exactly.
    R = np.exp(-0.01 * TIMES)
    np.testing.assert_allclose(
        rate, R**0.2 * (2 * R**0.8 - R**1.6), rtol=1e-12, atol=1e-300
    )
    assert np.all(rate <= independent.sf(TIMES) + 1e-15)


def test_the_rate_split_keeps_each_members_life():
    # Only "a" matters: it is in series with "b" in parallel with a node
    # that never fails.
    edges = [("s", "a"), ("a", "j"), ("j", "b"), ("j", "p"), ("b", "t")]
    edges += [("p", "t")]
    nodes = {"a": W, "b": W, "j": PerfectReliability, "p": PerfectReliability}
    t = np.array([20.0, 100.0, 250.0])

    def alone(model):
        group = [CCFGroup(["a", "b"], model)]
        return NonRepairableRBD(edges, nodes, ccf_groups=group).sf(t)

    np.testing.assert_allclose(
        alone(BetaFactor(0.3, basis="rate")), W.sf(t), rtol=1e-12
    )
    np.testing.assert_allclose(
        alone(MGL(0.3, basis="rate")), W.sf(t), rtol=1e-12
    )
    with pytest.warns(UserWarning):
        drifted = alone(BetaFactor(0.3))
    assert not np.allclose(drifted, W.sf(t), rtol=1e-3)
    # A series pair: both members' own causes and the shared shock.
    series = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": E, "b": E},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2, basis="rate"))],
    )
    assert series.sf(100.0) == pytest.approx(np.exp(-1.8), rel=1e-12)


def test_mgl_by_rate_is_independent_causes():
    """Each specific set of members has a cause of its own, striking
    independently: enumerate which causes have struck, and in each case
    count the members left that survive their own causes."""
    beta, gamma = 0.15, 0.4
    rbd = two_of_three(MGL(beta, gamma, basis="rate"))
    t = np.array([1.0, 20.0, 60.0, 100.0, 200.0])
    H = (t / 100.0) ** 2
    shared = [
        ({"a", "b"}, beta * (1 - gamma) / 2),
        ({"a", "c"}, beta * (1 - gamma) / 2),
        ({"b", "c"}, beta * (1 - gamma) / 2),
        ({"a", "b", "c"}, beta * gamma),
    ]
    expected = np.zeros_like(t)
    for struck in itertools.product([False, True], repeat=len(shared)):
        weight, down = np.ones_like(t), set()
        for hit, (members, c) in zip(struck, shared):
            weight = weight * (-np.expm1(-c * H) if hit else np.exp(-c * H))
            if hit:
                down |= members
        left = 3 - len(down)
        own = np.exp(-(1 - beta) * H)
        expected += weight * stats.binom.sf(1, left, own)
    np.testing.assert_allclose(rbd.sf(t), expected, rtol=1e-12)


def test_one_letter_mgl_is_the_beta_factor_by_either_basis():
    Q = np.array([1e-12, 1e-3, 0.3, 0.999])
    for basis in ("probability", "rate"):
        q_beta, shocks_beta = BetaFactor(0.2, basis=basis).decompose(
            ["a", "b"], Q
        )
        q_mgl, shocks_mgl = MGL(0.2, basis=basis).decompose(["a", "b"], Q)
        np.testing.assert_allclose(q_mgl, q_beta, rtol=1e-15)
        assert [m for m, _ in shocks_mgl] == [m for m, _ in shocks_beta]
        np.testing.assert_allclose(
            shocks_mgl[0][1], shocks_beta[0][1], rtol=1e-15
        )


def test_the_splits_agree_while_q_is_small():
    for model in (BetaFactor, lambda b, **k: MGL(b, 0.3, **k)):
        members = ["a", "b"] if model is BetaFactor else ["a", "b", "c"]
        for Q in (1e-6, 1e-3, 1e-2):
            q_p, shocks_p = model(0.1).decompose(members, Q)
            q_r, shocks_r = model(0.1, basis="rate").decompose(members, Q)
            assert q_r[0] == pytest.approx(q_p[0], rel=2 * Q)
            # The same sets, with probabilities agreeing to first order.
            assert {m for m, _ in shocks_r} == {m for m, _ in shocks_p}
            p = dict(shocks_p)
            for members_hit, prob in shocks_r:
                assert prob[0] == pytest.approx(p[members_hit][0], rel=2 * Q)


def test_small_probabilities_keep_their_precision():
    tiny = np.array([1e-15, 1e-9])
    q, shocks = BetaFactor(0.3, basis="rate").decompose(["a", "b"], tiny)
    # 1 - (1 - Q) ** 0.7, to its full precision.
    np.testing.assert_allclose(q, -np.expm1(0.7 * np.log1p(-tiny)), rtol=1e-14)
    np.testing.assert_allclose(
        shocks[0][1], -np.expm1(0.3 * np.log1p(-tiny)), rtol=1e-14
    )
    # The system's unreliability, dominated by the shared shock.
    rbd = pair(BetaFactor(0.2, basis="rate"))
    t = np.array([1e-9, 1e-6])
    np.testing.assert_allclose(rbd.ff(t), 0.2 * 0.01 * t, rtol=1e-6)
    # And the survival of a long life keeps its own: R^beta (2R^.8 - R^1.6).
    long = 5000.0
    R = np.exp(-0.01 * long)
    assert rbd.sf(long) == pytest.approx(
        R**0.2 * (2 * R**0.8 - R**1.6), rel=1e-9
    )


def test_a_probability_split_warns_once_beyond_its_range():
    rbd = pair(BetaFactor(0.2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rbd.sf(5.0)  # Q = 0.049
    with pytest.warns(UserWarning) as caught:
        rbd.sf([5.0, 100.0])
    (warning,) = caught
    message = str(warning.message)
    assert "['a', 'b']" in message and "Q = 0.632" in message
    assert "BetaFactor(beta=0.2, basis='rate')" in message
    assert warning.filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rbd.sf(300.0)  # told once for this diagram
        rbd.bx_life(50)
        pair(BetaFactor(0.2, basis="rate")).sf(TIMES)
        # A small fixed probability on demand is what the split is for.
        fixed = surv.FixedEventProbability.from_params(0.01)
        pair(BetaFactor(0.2), fixed).ff()
    with pytest.warns(UserWarning, match=r"MGL\(0.1, 0.3, basis='rate'\)"):
        two_of_three(MGL(0.1, 0.3)).ff(80.0)


def test_the_exact_mttf_takes_a_rate_split():
    rbd = pair(BetaFactor(0.2, basis="rate"))
    # The integral of 2R - R^(2 - beta).
    assert rbd.mean() == pytest.approx(200.0 - 100.0 / 1.8, rel=1e-9)
    assert rbd.mean_time_to_failure() == pytest.approx(rbd.mean())
    with pytest.raises(NotImplementedError, match="basis='rate'"):
        pair(BetaFactor(0.2)).mean()
    # Through a nested diagram too.
    outer = NonRepairableRBD(
        [("s", "inner"), ("inner", "c"), ("c", "t")],
        {"inner": rbd, "c": surv.Exponential.from_params([0.001])},
    )
    expected = outer.mean()
    t = np.linspace(0.0, 4000.0, 400_001)
    assert expected == pytest.approx(np.trapezoid(outer.sf(t), t), rel=1e-6)
    assert outer.analysis_routes()["mean"].route == "numerical"


def test_the_simulations_draw_the_shared_shocks():
    rbd = two_of_three(MGL(0.15, 0.4, basis="rate"))
    t = np.array([20.0, 60.0, 100.0, 150.0])
    lives = rbd.random(100_000, seed=3)
    exact = rbd.sf(t)
    found = (lives[:, None] > t).mean(axis=0)
    z = (found - exact) / np.sqrt(exact * (1 - exact) / len(lives))
    assert np.all(np.abs(z) < 4)
    mean = rbd.mean()
    interval = rbd.mean_time_to_failure_interval(50_000, seed=4)
    assert interval.lower < mean < interval.upper
    assert rbd.mean(method="simulate", seed=5, antithetic=True) == (
        pytest.approx(mean, rel=0.01)
    )
    # One at a time, when another node's draws cannot be batched.
    edges = VOTE[:-1] + [("v", "z"), ("z", "t")]
    nodes = {"a": W, "b": W, "c": W, "v": PerfectReliability}
    one_by_one = NonRepairableRBD(
        edges,
        {**nodes, "z": binomial_first([1e6, 1.0])},
        k={"v": 2},
        ccf_groups=[CCFGroup(["a", "b", "c"], MGL(0.15, 0.4, basis="rate"))],
    )
    assert one_by_one._row_sampler() is None
    events = one_by_one.random(20_000, seed=6)
    found = (events[:, None] > t).mean(axis=0)
    exact = one_by_one.sf(t)
    z = (found - exact) / np.sqrt(exact * (1 - exact) / len(events))
    assert np.all(np.abs(z) < 4)


def test_a_probability_split_is_still_left_out_of_simulations():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        split = pair(BetaFactor(0.2))
        independent = NonRepairableRBD(PAIR, {"a": E, "b": E})
        assert np.array_equal(
            split.random(1000, seed=1), independent.random(1000, seed=1)
        )
    assert "left out" in split.analysis_routes()["random"].reason


def test_compare_shares_each_members_own_causes():
    # With beta = 0 a group splitting the rate is independent members, and
    # each member draws its own causes from its own stream, as it would
    # outside a group: the two systems' lifetimes match.
    grouped = two_of_three(MGL(0.0, 0.0, basis="rate"))
    independent = NonRepairableRBD(
        VOTE,
        {"a": W, "b": W, "c": W, "v": PerfectReliability},
        k={"v": 2},
    )
    same = grouped.compare(
        independent, mc_samples=5000, seed=1, method="simulate"
    )
    assert abs(same.estimate) < 1e-9
    # A shared cause shortens a parallel group's life.
    names = ["a", "b", "c"]
    edges = [("s", n) for n in names] + [(n, "t") for n in names]
    alone = NonRepairableRBD(edges, {n: W for n in names})
    coupled = NonRepairableRBD(
        edges,
        {n: W for n in names},
        ccf_groups=[CCFGroup(names, MGL(0.2, 0.5, basis="rate"))],
    )
    worse = coupled.compare(
        alone, mc_samples=20_000, seed=1, method="simulate"
    )
    exact = coupled.mean() - alone.mean()
    assert exact < 0
    assert abs(worse.estimate - exact) < 4 * worse.standard_error


def test_a_group_that_cannot_be_drawn_is_refused():
    own_way = binomial_first([100.0, 2.0])
    rbd = pair(BetaFactor(0.2, basis="rate"), own_way)
    assert rbd.sf(50.0) == pytest.approx(
        pair(BetaFactor(0.2, basis="rate"), W).sf(50.0)
    )
    for call in (
        lambda: rbd.random(10, seed=1),
        lambda: rbd.mean(method="simulate", mc_samples=10),
        lambda: rbd.compare(rbd, mc_samples=10, method="simulate"),
        lambda: rbd.unreliability_interval(10.0),
    ):
        with pytest.raises(NotImplementedError, match="quantile function"):
            call()
    routes = rbd.analysis_routes()
    for name in ("random", "random_block", "mean_time_to_failure_interval"):
        assert routes[name].route == "refused"
    assert routes["mean"].route == "numerical"


def test_a_small_unreliability_with_a_rate_split():
    rbd = pair(BetaFactor(0.1, basis="rate"), W)
    x = 2.0
    result = rbd.unreliability_interval(x, seed=1, relative_tolerance=0.1)
    exact = rbd.ff(x)
    assert result.lower < exact < result.upper
    assert exact == pytest.approx(0.1 * W.ff(x), rel=0.01)
    with pytest.raises(NotImplementedError, match="split a failure prob"):
        pair(BetaFactor(0.1), W).unreliability_interval(x)


def test_the_basis_is_saved():
    for model in (BetaFactor(0.2, basis="rate"), MGL(0.1, 0.3, basis="rate")):
        rbd = (
            pair(model, W)
            if isinstance(model, BetaFactor)
            else two_of_three(model)
        )
        saved = rbd_to_dict(rbd)
        assert saved["ccf_groups"][0]["model"]["basis"] == "rate"
        restored = rbd_from_dict(saved)
        assert restored.ccf_groups[0].model == model
        assert restored.sf(80.0) == rbd.sf(80.0)
    # The default leaves the saved form as it was, and reads back.
    saved = rbd_to_dict(pair(BetaFactor(0.2), W))
    assert "basis" not in saved["ccf_groups"][0]["model"]
    assert rbd_from_dict(saved).ccf_groups[0].model.basis == "probability"


def test_the_models_say_their_basis():
    assert BetaFactor(0.2) != BetaFactor(0.2, basis="rate")
    assert BetaFactor(0.2, basis="rate") == BetaFactor(0.2, basis="rate")
    assert len({MGL(0.1, 0.3), MGL(0.1, 0.3, basis="rate")}) == 2
    assert repr(BetaFactor(0.2)) == "BetaFactor(beta=0.2)"
    assert repr(BetaFactor(0.2, basis="rate")) == (
        "BetaFactor(beta=0.2, basis='rate')"
    )
    assert repr(MGL(0.1)) == "MGL(0.1)"
    assert repr(MGL(0.1, 0.3, basis="rate")) == "MGL(0.1, 0.3, basis='rate')"
    for make in (
        lambda: BetaFactor(0.1, basis="frequency"),
        lambda: MGL(0.1, basis="Rate"),
    ):
        with pytest.raises(ValueError, match="'probability' or 'rate'"):
            make()


def test_the_routes_follow_the_basis():
    rate = pair(BetaFactor(0.2, basis="rate")).analysis_routes()
    assert rate["mean"].route == "numerical"
    assert "shared shocks" in rate["random"].reason
    assert rate["unreliability_interval"].route == "simulated"
    probability = pair(BetaFactor(0.2)).analysis_routes()
    assert probability["mean"].route == "refused"
    assert probability["unreliability_interval"].route == "refused"
