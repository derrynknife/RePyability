"""Tests for common-cause failures (CCF), issue #44: the beta-factor model on
``NonRepairableRBD``.

The exact anchors use a two-component parallel system, whose system
unreliability under the beta-factor decomposition (a shared cause of
probability ``beta*Q`` failing both, independent failures ``(1-beta)*Q``
otherwise) has the closed form the conditioning engine must reproduce, and
which collapses to the ordinary independent result at ``beta = 0``.
"""

import numpy as np
import pytest
from surpyval import FixedEventProbability, Weibull

from repyability import (
    MGL,
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    NodeState,
    NonRepairableRBD,
    PerfectReliability,
)

# The probability split's arithmetic is checked at large probabilities on
# purpose; its warning beyond them is tested in test_ccf_lifetime.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Common-cause group:UserWarning"
)

PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def _parallel(Q, ccf_groups=None):
    return NonRepairableRBD(
        PARALLEL,
        {
            "a": FixedEventProbability.from_params(Q),
            "b": FixedEventProbability.from_params(Q),
        },
        ccf_groups=ccf_groups,
    )


# -- beta = 0 reduces exactly to independence ------------------------------


@pytest.mark.parametrize("Q", [0.01, 0.1, 0.3])
def test_beta_zero_is_independent(Q):
    indep = _parallel(Q)
    with_ccf = _parallel(Q, [CCFGroup(["a", "b"], BetaFactor(0.0))])
    assert with_ccf.sf() == pytest.approx(indep.sf())


def test_beta_zero_kofn():
    # A 2-out-of-3 structure: beta=0 must match the plain k-of-n reliability.
    edges = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
    nodes = {n: FixedEventProbability.from_params(0.1) for n in "abc"}
    k = {"t": 2}
    indep = NonRepairableRBD(edges, nodes, k=k)
    ccf = NonRepairableRBD(
        edges, nodes, k=k, ccf_groups=[CCFGroup(list("abc"), BetaFactor(0.0))]
    )
    assert ccf.sf() == pytest.approx(indep.sf())


# -- the exact conditioning result -----------------------------------------


@pytest.mark.parametrize("Q,beta", [(0.01, 0.1), (0.1, 0.2), (0.2, 0.05)])
def test_parallel_matches_exact_conditioning(Q, beta):
    ccf = _parallel(Q, [CCFGroup(["a", "b"], BetaFactor(beta))])
    # Exact conditioning: shared cause fires (prob beta*Q -> both down) or not
    # (each fails independently with prob (1-beta)*Q).
    q_common = beta * Q
    r_independent_fail = (1 - beta) * Q
    expected = (1 - q_common) * (1 - r_independent_fail**2)
    assert float(ccf.sf()) == pytest.approx(expected)
    # ...and it is close to the (rare-event) textbook value
    # beta*Q + ((1-b)Q)^2 (they differ only at higher order).
    textbook = beta * Q + ((1 - beta) * Q) ** 2
    assert 1 - float(ccf.sf()) == pytest.approx(textbook, abs=1e-3)


def test_beta_one_kills_redundancy():
    # beta = 1: the pair always fails together, so a parallel pair is no better
    # than a single component.
    Q = 0.1
    ccf = _parallel(Q, [CCFGroup(["a", "b"], BetaFactor(1.0))])
    assert float(ccf.sf()) == pytest.approx(1 - Q)


def test_ccf_lowers_redundant_reliability():
    Q = 0.05
    indep = float(_parallel(Q).sf())
    for beta in (0.05, 0.2, 0.5):
        r = float(_parallel(Q, [CCFGroup(["a", "b"], BetaFactor(beta))]).sf())
        assert r < indep


def test_sf_monotone_decreasing_in_beta():
    Q = 0.1
    rs = [
        float(_parallel(Q, [CCFGroup(["a", "b"], BetaFactor(b))]).sf())
        for b in (0.0, 0.1, 0.3, 0.6, 1.0)
    ]
    assert all(a >= b for a, b in zip(rs, rs[1:]))


# -- multiple groups, time-varying -----------------------------------------


def test_multiple_disjoint_groups():
    # Two independent parallel pairs in series, each its own CCF group.
    edges = [
        ("s", "a1"),
        ("s", "a2"),
        ("a1", "m"),
        ("a2", "m"),
        ("m", "b1"),
        ("m", "b2"),
        ("b1", "t"),
        ("b2", "t"),
    ]
    Q = 0.1
    nodes = {
        n: FixedEventProbability.from_params(Q)
        for n in ("a1", "a2", "b1", "b2")
    }
    nodes["m"] = PerfectReliability  # perfect connector between the two stages
    groups = [
        CCFGroup(["a1", "a2"], BetaFactor(0.1)),
        CCFGroup(["b1", "b2"], BetaFactor(0.2)),
    ]
    rbd = NonRepairableRBD(edges, nodes, ccf_groups=groups)
    # Each stage is a parallel pair with CCF; the series system is their
    # product (the stages are independent).
    stage_a = (1 - 0.1 * Q) * (1 - (0.9 * Q) ** 2)
    stage_b = (1 - 0.2 * Q) * (1 - (0.8 * Q) ** 2)
    assert float(rbd.sf()) == pytest.approx(stage_a * stage_b)


def test_time_varying_array():
    rbd = NonRepairableRBD(
        PARALLEL,
        {
            "a": Weibull.from_params([100.0, 2.0]),
            "b": Weibull.from_params([100.0, 2.0]),
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    t = np.array([20.0, 60.0, 120.0])
    out = rbd.sf(t)
    assert out.shape == (3,)
    assert np.all(np.diff(out) < 0)  # decreasing in time
    # matches the per-time exact conditioning
    Q = 1 - np.exp(-((t / 100.0) ** 2.0))
    expected = (1 - 0.1 * Q) * (1 - (0.9 * Q) ** 2)
    assert np.allclose(out, expected)


# -- serialisation ---------------------------------------------------------


def test_serialisation_roundtrip():
    rbd = NonRepairableRBD(
        PARALLEL,
        {
            "a": Weibull.from_params([100.0, 2.0]),
            "b": Weibull.from_params([100.0, 2.0]),
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.12))],
    )
    restored = NonRepairableRBD.from_json(rbd.to_json())
    assert len(restored.ccf_groups) == 1
    assert restored.ccf_groups[0].model == BetaFactor(0.12)
    assert list(restored.ccf_groups[0].members) == ["a", "b"]
    t = np.array([30.0, 90.0])
    np.testing.assert_allclose(restored.sf(t), rbd.sf(t))


# -- unsupported combinations raise clearly --------------------------------


def test_the_analyses_given_ages_are_guarded():
    # They need members of different ages, which the models do not cover.
    rbd = NonRepairableRBD(
        PARALLEL,
        {
            "a": Weibull.from_params([100.0, 2.0]),
            "b": Weibull.from_params([100.0, 2.0]),
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    for call in (
        lambda: rbd.sf_given_state(50.0, {"a": NodeState(age=10)}),
        lambda: rbd.remaining_life(0.9, {"a": NodeState(age=10)}),
        lambda: rbd.importances_given_state(50.0, {"a": NodeState(age=10)}),
    ):
        with pytest.raises(NotImplementedError, match="CCF|common-cause"):
            call()


def test_structural_importance_allowed_with_ccf():
    # Structural importance is probability-free, so CCF does not change it.
    rbd = _parallel(0.1, [CCFGroup(["a", "b"], BetaFactor(0.3))])
    si = rbd.structural_importance()
    assert si == pytest.approx({"a": 0.5, "b": 0.5})


def test_forcing_ccf_member_raises():
    rbd = _parallel(0.1, [CCFGroup(["a", "b"], BetaFactor(0.1))])
    with pytest.raises(NotImplementedError, match="CCF|broken_nodes"):
        rbd.sf(broken_nodes=["a"])


# -- validation ------------------------------------------------------------


def test_betafactor_validation():
    with pytest.raises(ValueError, match="beta"):
        BetaFactor(-0.1)
    with pytest.raises(ValueError, match="beta"):
        BetaFactor(1.5)
    assert BetaFactor(0.2) == BetaFactor(0.2)
    assert BetaFactor(0.2) != BetaFactor(0.3)


def test_ccfgroup_validation():
    with pytest.raises(ValueError, match="at least 2"):
        CCFGroup(["a"], BetaFactor(0.1))
    with pytest.raises(ValueError, match="distinct"):
        CCFGroup(["a", "a"], BetaFactor(0.1))
    with pytest.raises(ValueError, match="BetaFactor"):
        CCFGroup(["a", "b"], object())


def test_group_validation_against_rbd():
    # Unknown member
    with pytest.raises(ValueError, match="not a node"):
        _parallel(0.1, [CCFGroup(["a", "z"], BetaFactor(0.1))])
    # Non-symmetric members
    with pytest.raises(ValueError, match="symmetric"):
        NonRepairableRBD(
            PARALLEL,
            {
                "a": Weibull.from_params([100.0, 2.0]),
                "b": Weibull.from_params([200.0, 2.0]),
            },
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
    # A node in two groups
    with pytest.raises(ValueError, match="more than one"):
        _parallel(
            0.1,
            [
                CCFGroup(["a", "b"], BetaFactor(0.1)),
                CCFGroup(["a", "b"], BetaFactor(0.2)),
            ],
        )


# -- Multiple Greek Letter (partial common cause) --------------------------


def test_mgl_two_members_equals_beta_factor():
    # MGL(beta) on a two-member group is exactly the beta-factor model.
    Q = 0.1
    bf = float(_parallel(Q, [CCFGroup(["a", "b"], BetaFactor(0.15))]).sf())
    mgl = float(_parallel(Q, [CCFGroup(["a", "b"], MGL(0.15))]).sf())
    assert mgl == pytest.approx(bf)


def test_mgl_1of3_matches_textbook():
    # A parallel triple (fails iff all three fail) with MGL(beta, gamma). The
    # all-three-down cut set has leading-order contributions Q3 + 3*Q2*Q1 +
    # Q1**3 from the MGL basic-event probabilities.
    Q, beta, gamma = 0.1, 0.1, 0.3
    edges = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
    nodes = {n: FixedEventProbability.from_params(Q) for n in "abc"}
    rbd = NonRepairableRBD(
        edges, nodes, ccf_groups=[CCFGroup(list("abc"), MGL(beta, gamma))]
    )
    q1 = (1 - beta) * Q
    q2 = beta * (1 - gamma) / 2 * Q
    q3 = beta * gamma * Q
    textbook = q3 + 3 * q2 * q1 + q1**3
    assert 1 - float(rbd.sf()) == pytest.approx(textbook, abs=1e-4)


def test_mgl_gamma_one_is_all_or_nothing():
    # gamma = 1 forces every >=2 common cause to escalate to all 3, i.e. the
    # group is all-or-nothing: identical to BetaFactor(beta) on the triple.
    Q, beta = 0.1, 0.2
    edges = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
    nodes = {n: FixedEventProbability.from_params(Q) for n in "abc"}
    mgl = NonRepairableRBD(
        edges, nodes, ccf_groups=[CCFGroup(list("abc"), MGL(beta, 1.0))]
    )
    beta_all = NonRepairableRBD(
        edges, nodes, ccf_groups=[CCFGroup(list("abc"), BetaFactor(beta))]
    )
    assert float(mgl.sf()) == pytest.approx(float(beta_all.sf()))


def test_mgl_serialisation_roundtrip():
    Q = 0.1
    edges = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
    nodes = {n: FixedEventProbability.from_params(Q) for n in "abc"}
    rbd = NonRepairableRBD(
        edges, nodes, ccf_groups=[CCFGroup(list("abc"), MGL(0.1, 0.3))]
    )
    restored = NonRepairableRBD.from_json(rbd.to_json())
    assert restored.ccf_groups[0].model == MGL(0.1, 0.3)
    assert float(restored.sf()) == pytest.approx(float(rbd.sf()))


def test_mgl_validation():
    with pytest.raises(ValueError, match="at least one"):
        MGL()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MGL(0.1, 1.5)
    assert MGL(0.1, 0.2).group_size == 3
    assert MGL(0.1) == MGL(0.1)
    assert MGL(0.1, 0.2) != MGL(0.1, 0.3)
    # An MGL model's parameter count must match the group size.
    with pytest.raises(ValueError, match="members"):
        CCFGroup(["a", "b"], MGL(0.1, 0.3))  # 2 letters -> needs 3 members


# -- Alpha-factor (#311) ------------------------------------------------------

# The alpha-factor model's basic-event probabilities (NUREG/CR-5485,
# Mosleh et al.), for a group of m members with alpha_1..alpha_m summing to
# 1 and each member's total Q: under staggered testing
#     Q_k = alpha_k / C(m-1, k-1) * Q,
# and under non-staggered testing
#     Q_k = k / C(m-1, k-1) * alpha_k / alpha_t * Q,  alpha_t = sum k alpha_k.
# A three-member group with alpha = (0.95, 0.04, 0.01) and Q = 1e-3 then has,
# by hand: staggered Q_1 = 9.5e-4, Q_2 = 0.04 / 2 * 1e-3 = 2e-5, Q_3 = 1e-5;
# non-staggered alpha_t = 0.95 + 0.08 + 0.03 = 1.06, Q_1 = 0.95 / 1.06 * 1e-3,
# Q_2 = 0.04 / 1.06 * 1e-3, Q_3 = 0.03 / 1.06 * 1e-3.
ALPHAS = (0.95, 0.04, 0.01)
BY_HAND = {
    "staggered": (9.5e-4, 2e-5, 1e-5),
    "non-staggered": (0.95e-3 / 1.06, 0.04e-3 / 1.06, 0.03e-3 / 1.06),
}


def _specific_sets(model, members, Q):
    """``Q_1, ..., Q_m`` from ``decompose``: the independent part, then
    one shock's probability for each size (every set of a size alike)."""
    q_independent, shocks = model.decompose(members, Q)
    by_size = {}
    for subset, p in shocks:
        by_size.setdefault(len(subset), set()).add(float(p[0]))
    assert all(len(values) == 1 for values in by_size.values())
    return (float(q_independent[0]),) + tuple(
        by_size[k].pop() for k in range(2, len(members) + 1)
    )


@pytest.mark.parametrize("testing", ["staggered", "non-staggered"])
def test_alpha_factor_specific_sets_by_hand(testing):
    model = AlphaFactor(ALPHAS, testing=testing)
    assert model.group_size == 3
    found = _specific_sets(model, list("abc"), 1e-3)
    assert found == pytest.approx(BY_HAND[testing], rel=1e-12)
    # The Q_k of the sets holding a member sum to its Q.
    Q1, Q2, Q3 = found
    assert Q1 + 2 * Q2 + Q3 == pytest.approx(1e-3, rel=1e-12)


def _mgl_letters(fractions):
    """The MGL letters of the specific-set fractions ``Q_k / Q``,
    ``k = 1..m``: ``rho_k`` is the share of a member's failures shared
    by at least ``k`` members among those shared by at least ``k - 1``."""
    from math import comb

    m = len(fractions)

    def at_least(k):
        return sum(
            comb(m - 1, j - 1) * fractions[j - 1] for j in range(k, m + 1)
        )

    return [at_least(k) / at_least(k - 1) for k in range(2, m + 1)]


@pytest.mark.parametrize("testing", ["staggered", "non-staggered"])
@pytest.mark.parametrize(
    "alphas", [(0.95, 0.04, 0.01), (0.9, 0.06, 0.03, 0.01), (0.8, 0.2)]
)
@pytest.mark.parametrize(
    "options", [{}, {"shocks": "independent"}, {"basis": "rate"}]
)
def test_alpha_factor_is_the_equivalent_mgl(testing, alphas, options):
    # Its outcomes, and so every analysis, are those of the MGL model with
    # the letters its specific-set probabilities give.
    model = AlphaFactor(alphas, testing=testing, **options)
    letters = _mgl_letters(
        _specific_sets(
            AlphaFactor(alphas, testing=testing),
            list("abcd")[: len(alphas)],
            1.0,
        )
    )
    mgl = MGL(*letters, **options)
    members = list("abcd")[: len(alphas)]
    Q = np.array([1e-4, 0.01, 0.1, 0.4])
    ours = model._split(members, Q, 1.0 - Q)
    theirs = mgl._split(members, Q, 1.0 - Q)
    np.testing.assert_allclose(ours[0], theirs[0], rtol=1e-12)
    np.testing.assert_allclose(ours[1], theirs[1], rtol=1e-12)
    assert [s for s, _ in ours[2]] == [s for s, _ in theirs[2]]
    for (_, p), (_, q) in zip(ours[2], theirs[2]):
        np.testing.assert_allclose(p, q, rtol=1e-12)
    own, causes = model._causes(members)
    own_mgl, causes_mgl = mgl._causes(members)
    assert own == pytest.approx(own_mgl, rel=1e-12)
    assert [s for s, _ in causes] == [s for s, _ in causes_mgl]
    assert [c for _, c in causes] == pytest.approx(
        [c for _, c in causes_mgl], rel=1e-12
    )
    # And a 2-out-of-m system agrees.
    edges = [("s", n) for n in members] + [(n, "t") for n in members]
    nodes = {n: Weibull.from_params([1000.0, 1.5]) for n in members}
    x = np.array([30.0, 300.0])
    assert NonRepairableRBD(
        edges, nodes, k={"t": 2}, ccf_groups=[CCFGroup(members, model)]
    ).ff(x) == pytest.approx(
        NonRepairableRBD(
            edges, nodes, k={"t": 2}, ccf_groups=[CCFGroup(members, mgl)]
        ).ff(x),
        rel=1e-12,
    )


def test_alpha_factor_on_a_pair_is_the_beta_factor():
    # Staggered, alpha_2 is the share of a member's failures that are
    # shared: the beta factor. Non-staggered, each shared event counts
    # twice, so beta = 2 alpha_2 / (alpha_1 + 2 alpha_2).
    Q, beta = 0.1, 0.15
    bf = float(_parallel(Q, [CCFGroup(["a", "b"], BetaFactor(beta))]).sf())
    staggered = AlphaFactor([1.0 - beta, beta])
    assert float(_parallel(Q, [CCFGroup(["a", "b"], staggered)]).sf()) == (
        pytest.approx(bf, rel=1e-12)
    )
    alpha_2 = beta / (2.0 - beta)
    non_staggered = AlphaFactor(
        [1.0 - alpha_2, alpha_2], testing="non-staggered"
    )
    assert float(
        _parallel(Q, [CCFGroup(["a", "b"], non_staggered)]).sf()
    ) == pytest.approx(bf, rel=1e-12)


def test_alpha_factor_serialisation_roundtrip():
    edges = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
    nodes = {n: FixedEventProbability.from_params(0.01) for n in "abc"}
    for model in (
        AlphaFactor(ALPHAS),
        AlphaFactor(ALPHAS, testing="non-staggered"),
        AlphaFactor(ALPHAS, shocks="independent"),
        AlphaFactor(ALPHAS, testing="non-staggered", basis="rate"),
    ):
        rbd = NonRepairableRBD(
            edges, nodes, ccf_groups=[CCFGroup(list("abc"), model)]
        )
        saved = rbd.to_dict()
        assert saved["ccf_groups"][0]["model"]["kind"] == "alpha_factor"
        assert saved["ccf_groups"][0]["model"]["testing"] == model.testing
        for restored in (
            NonRepairableRBD.from_dict(saved),
            NonRepairableRBD.from_json(rbd.to_json()),
        ):
            assert restored.ccf_groups[0].model == model
            assert repr(restored.ccf_groups[0].model) == repr(model)
            assert float(restored.sf()) == pytest.approx(
                float(rbd.sf()), rel=1e-12
            )


def test_alpha_factor_parameters_round_trip():
    from repyability.rbd.ccf import parameters, with_parameters

    model = AlphaFactor(ALPHAS, testing="non-staggered", shocks="independent")
    assert parameters(model) == {"alpha_2": 0.04, "alpha_3": 0.01}
    assert with_parameters(model, {}) == model
    moved = with_parameters(model, {"alpha_2": 0.1})
    assert moved == AlphaFactor(
        [0.89, 0.1, 0.01], testing="non-staggered", shocks="independent"
    )
    # alpha_1 is the remainder, and cannot go negative.
    with pytest.raises(ValueError, match="sum to at most 1"):
        with_parameters(model, {"alpha_2": 0.995})
    assert with_parameters(model, {"alpha_2": 0.99}).alphas == pytest.approx(
        (0.0, 0.99, 0.01)
    )
    with pytest.raises(ValueError, match="not parameters"):
        with_parameters(model, {"alpha_1": 0.5})


def test_alpha_factor_levers():
    edges = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
    nodes = {n: Weibull.from_params([100.0, 2.0]) for n in "abc"}
    rbd = NonRepairableRBD(
        edges,
        nodes,
        k={"t": 2},
        ccf_groups=[CCFGroup(list("abc"), AlphaFactor(ALPHAS))],
    )
    levers = {(lever.key, lever.name): lever for lever in rbd.levers()}
    assert set(levers) == {
        (("a", "b", "c"), "alpha"),
        (("a", "b", "c"), "beta"),
        (("a", "b", "c"), "ccf_alpha_2"),
        (("a", "b", "c"), "ccf_alpha_3"),
    }
    assert levers[(("a", "b", "c"), "ccf_alpha_2")].value == 0.04
    assert levers[(("a", "b", "c"), "ccf_alpha_3")].bounds == (0.0, 1.0)
    moved = rbd.with_levers({(("a", "b", "c"), "ccf_alpha_2"): 0.1})
    assert moved.ccf_groups[0].model == AlphaFactor([0.89, 0.1, 0.01])
    with pytest.raises(ValueError, match="between 0 and 1"):
        rbd.with_levers({(("a", "b", "c"), "ccf_alpha_2"): 1.5})
    with pytest.raises(ValueError, match="sum to at most 1"):
        rbd.with_levers({(("a", "b", "c"), "ccf_alpha_2"): 0.995})
    # The sensitivity moves each alpha_k with alpha_1 the remainder.
    x = 50.0
    report = rbd.parameter_sensitivity(x)
    step = 1e-5
    up = rbd.with_levers({(("a", "b", "c"), "ccf_alpha_3"): 0.01 + step})
    down = rbd.with_levers({(("a", "b", "c"), "ccf_alpha_3"): 0.01 - step})
    assert report[("a", "b", "c")]["ccf_alpha_3"] == pytest.approx(
        (float(up.sf(x)) - float(down.sf(x))) / (2 * step), rel=1e-5
    )


def test_alpha_factor_validation():
    with pytest.raises(ValueError, match="at least two"):
        AlphaFactor([1.0])
    with pytest.raises(ValueError, match="in a list"):
        AlphaFactor(0.9)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AlphaFactor([1.5, -0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AlphaFactor([0.9, "0.1"])
    with pytest.raises(ValueError, match="sum to 1"):
        AlphaFactor([0.9, 0.2])
    with pytest.raises(ValueError, match="testing must be"):
        AlphaFactor([0.9, 0.1], testing="weekly")
    with pytest.raises(ValueError, match="shocks must be"):
        AlphaFactor([0.9, 0.1], shocks="both")
    with pytest.raises(ValueError, match="strikes independently"):
        AlphaFactor([0.9, 0.1], basis="rate", shocks="exclusive")
    # A rounded table's alphas are scaled to sum to 1.
    rounded = AlphaFactor([0.958, 0.0263, 0.0086, 0.007])
    assert sum(rounded.alphas) == pytest.approx(1.0, abs=1e-15)
    assert rounded.alphas[1] / rounded.alphas[0] == pytest.approx(
        0.0263 / 0.958
    )
    assert AlphaFactor(ALPHAS) == AlphaFactor(list(ALPHAS))
    assert AlphaFactor(ALPHAS) != AlphaFactor(ALPHAS, testing="non-staggered")
    assert AlphaFactor(ALPHAS) != MGL(0.05, 0.2)
    assert len({AlphaFactor(ALPHAS), AlphaFactor(ALPHAS)}) == 1
    assert repr(AlphaFactor(ALPHAS)) == "AlphaFactor([0.95, 0.04, 0.01])"
    assert repr(
        AlphaFactor(ALPHAS, testing="non-staggered", basis="rate")
    ) == (
        "AlphaFactor([0.95, 0.04, 0.01], testing='non-staggered', "
        "basis='rate')"
    )
    assert repr(AlphaFactor(ALPHAS, shocks="independent")) == (
        "AlphaFactor([0.95, 0.04, 0.01], shocks='independent')"
    )
    assert AlphaFactor(ALPHAS).required_group_size() == 3
    with pytest.raises(ValueError, match="members"):
        CCFGroup(["a", "b"], AlphaFactor(ALPHAS))
    with pytest.raises(ValueError, match="BetaFactor, MGL or AlphaFactor"):
        CCFGroup(["a", "b"], "alpha")


def test_alpha_factor_warns_naming_its_rate_model():
    edges = [("s", n) for n in "abc"] + [(n, "t") for n in "abc"]
    nodes = {n: FixedEventProbability.from_params(0.3) for n in "abc"}
    rbd = NonRepairableRBD(
        edges,
        nodes,
        ccf_groups=[
            CCFGroup(list("abc"), AlphaFactor(ALPHAS, testing="non-staggered"))
        ],
    )
    with pytest.warns(
        UserWarning, match="testing='non-staggered', basis='rate'"
    ):
        rbd.sf()
