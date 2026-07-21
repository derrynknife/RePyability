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
    BetaFactor,
    CCFGroup,
    NodeState,
    NonRepairableRBD,
    PerfectReliability,
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


def test_importance_and_state_guarded():
    rbd = NonRepairableRBD(
        PARALLEL,
        {
            "a": Weibull.from_params([100.0, 2.0]),
            "b": Weibull.from_params([100.0, 2.0]),
        },
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    for call in (
        lambda: rbd.birnbaum_importance(50.0),
        lambda: rbd.criticality_importance(50.0),
        lambda: rbd.fussell_vesely(50.0),
        lambda: rbd.parameter_sensitivity(50.0),
        lambda: rbd.sf_given_state(50.0, {"a": NodeState(age=10)}),
        lambda: rbd.remaining_life(0.9, {"a": NodeState(age=10)}),
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
