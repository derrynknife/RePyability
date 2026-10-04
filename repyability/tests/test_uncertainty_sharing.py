"""Parameter uncertainty shared by the nodes holding one fitted model
(#214).

A fitted model is one population's: the nodes holding the same fitted
object, in the same role, are drawn alike in every draw, also when their
other models differ (a fleet's life fit, shared by pumps whose repairs
were recorded apart). A role-level spec says so, and a spec that draws
one of them and leaves the others at the fit is warned about.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, RepairableRBD
from repyability.rbd import _repairable_uncertainty as drawn


def fleet():
    """Two pumps in parallel: one life fit for both, a repair fit each."""
    g = np.random.default_rng(11)
    life = surv.Exponential.fit(g.exponential(100, 6))
    first = surv.Exponential.fit(g.exponential(5, 200))
    second = surv.Exponential.fit(g.exponential(5, 200))
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {"reliability": life, "repairability": first},
            "b": {"reliability": life, "repairability": second},
        },
    )


def lives(rbd, uncertainty, n=5, seed=0):
    """Each draw's life parameters of a and b."""
    inputs, groups = drawn.sources(rbd, uncertainty)
    models, group_models = drawn.draws(rbd, inputs, groups, n, seed)
    out = []
    for i in range(n):
        built = drawn.build(rbd, inputs, models, group_models, i)
        out.append(
            [
                float(np.ravel(built.components[node].reliability.params)[0])
                for node in "ab"
            ]
        )
    return np.array(out)


def test_a_shared_life_is_one_input_and_each_repair_its_own():
    assert drawn.fitted(fleet()) == {
        ("a", "b"): {"reliability": "fit"},
        "a": {"repairability": "fit"},
        "b": {"repairability": "fit"},
    }


def test_a_shared_life_is_drawn_once_a_draw():
    rbd = fleet()
    by_default = lives(rbd, None)
    np.testing.assert_array_equal(by_default[:, 0], by_default[:, 1])
    assert np.ptp(by_default[:, 0]) > 0.0


def test_the_role_level_spec_is_the_default():
    rbd = fleet()
    spec = {
        ("a", "b"): {"reliability": "fit"},
        "a": {"repairability": "fit"},
        "b": {"repairability": "fit"},
    }
    default = rbd.mean_availability_uncertainty(n_draws=200, seed=0)
    given = rbd.mean_availability_uncertainty(spec, n_draws=200, seed=0)
    np.testing.assert_array_equal(default.samples, given.samples)


def test_one_population_widens_the_interval():
    rbd = fleet()
    shared = rbd.mean_availability_uncertainty(n_draws=4000, seed=0)
    apart = rbd.mean_availability_uncertainty(
        {"a": "fit", "b": "fit"}, n_draws=4000, seed=0
    )
    # The issue's reference: independent draws from the same hess_inv give
    # a std of 0.00270, drawn once; 0.00157 per node.
    assert shared.std == pytest.approx(0.00270, rel=0.1)
    assert apart.std == pytest.approx(0.00157, rel=0.1)


def test_a_role_is_given_once():
    rbd = fleet()
    with pytest.raises(ValueError, match="reliability of node 'a'"):
        rbd.mean_availability_uncertainty(
            {("a", "b"): {"reliability": "fit"}, "a": "fit"}, n_draws=2
        )


def test_naming_one_holder_of_a_fit_warns():
    rbd = fleet()
    with pytest.warns(UserWarning, match="'b'"):
        rbd.mean_availability_uncertainty(
            {"a": {"reliability": "fit"}}, n_draws=2, seed=0
        )
    w = surv.Weibull.fit(np.random.default_rng(0).weibull(2, 15) * 100)
    pair = NonRepairableRBD(
        [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        {"p1": w, "p2": w},
    )
    with pytest.warns(UserWarning, match="'p2'"):
        pair.sf_uncertainty(80, {"p1": "fit"}, n_draws=2, seed=0)


def test_the_vega_splits_by_population():
    rbd = fleet()
    importance = rbd.uncertainty_importance()
    assert set(importance.first_order) == {("a", "b"), "a", "b"}
