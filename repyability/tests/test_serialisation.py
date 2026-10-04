"""Round-trip tests for RBD (de)serialisation to dict / JSON.

An RBD reconstructed from its own serialised form must be equivalent: same
structure (path sets, k values, repeated nodes) and same reliability /
availability.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD
from repyability.rbd.repeated_node import RepeatedNode
from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
from repyability.rbd.standby_node import StandbyModel
from repyability.utils.wrappers import numpy_seed

FEP = surv.FixedEventProbability
W = surv.Weibull.from_params
E = surv.Exponential.from_params


def _sf_matches(a, b, times=(1, 5, 20, 50, 100)):
    return all(np.isclose(float(a.sf(t)), float(b.sf(t))) for t in times)


def test_roundtrip_parallel_parametric_and_fixed_via_json():
    rbd = NonRepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {"A": W([100, 2]), "B": FEP.from_params(0.3)},
    )
    clone = NonRepairableRBD.from_json(rbd.to_json())
    assert _sf_matches(rbd, clone)
    assert rbd.get_min_path_sets() == clone.get_min_path_sets()


def test_roundtrip_k_out_of_n():
    rbd = NonRepairableRBD(
        [(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)],
        {2: W([50, 1.2]), 3: W([60, 1.1]), 4: W([70, 1.3])},
        k={5: 2},
    )
    clone = NonRepairableRBD.from_dict(rbd.to_dict())
    assert _sf_matches(rbd, clone)
    assert clone.G.nodes[5]["k"] == 2


def test_roundtrip_repeated_node_preserves_repeat():
    rbd = NonRepairableRBD(
        [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)],
        {
            2: W([100, 2]),
            3: FEP.from_params(0.1),
            4: FEP.from_params(0.15),
            5: 2,
        },
    )
    clone = NonRepairableRBD.from_json(rbd.to_json())
    assert _sf_matches(rbd, clone)
    assert clone.repeated == {5: 2}


def test_roundtrip_standby_and_repeated_wrappers():
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (2, 3), (3, "t")],
        {
            1: StandbyModel([W([5, 1.1]) for _ in range(3)]),
            2: RepeatedNode(W([10, 2]), 3, "parallel"),
            3: RepeatedStandbyNode(W([8, 1.2]), 2),
        },
    )
    clone = NonRepairableRBD.from_dict(rbd.to_dict())
    assert _sf_matches(rbd, clone)


def test_roundtrip_nested_rbd():
    sub = NonRepairableRBD(
        [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
        {"x": W([40, 2]), "y": W([40, 2])},
    )
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (2, "t")], {1: sub, 2: W([200, 1.5])}
    )
    clone = NonRepairableRBD.from_json(rbd.to_json())
    assert _sf_matches(rbd, clone)


def test_json_preserves_integer_node_names():
    # JSON object keys are strings; the list-of-entries encoding must keep
    # integer node identity so the reconstructed graph is the same.
    rbd = NonRepairableRBD(
        [("s", 1), (1, 2), (2, "t")], {1: W([10, 2]), 2: W([20, 2])}
    )
    clone = NonRepairableRBD.from_json(rbd.to_json())
    assert set(clone.nodes) == {1, 2}
    assert _sf_matches(rbd, clone)


def test_roundtrip_repairable_rbd():
    comps = {
        "A": {"reliability": E([0.1]), "repairability": E([1.0])},
        "B": {"reliability": E([0.2]), "repairability": E([0.5])},
    }
    rbd = RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")], comps
    )
    clone = RepairableRBD.from_json(rbd.to_json())
    assert np.isclose(rbd.mean_availability(), clone.mean_availability())
    assert np.isclose(
        rbd.mean_time_between_failures(), clone.mean_time_between_failures()
    )


def test_dict_carries_version_and_type():
    rbd = NonRepairableRBD([("s", 1), (1, "t")], {1: W([10, 2])})
    d = rbd.to_dict()
    assert d["type"] == "NonRepairableRBD"
    assert "repyability_version" in d


def test_from_dict_type_mismatch_raises():
    comps = {"A": {"reliability": E([0.1]), "repairability": E([1.0])}}
    rep = RepairableRBD([("s", "A"), ("A", "t")], comps)
    with pytest.raises(ValueError, match="from_dict got a"):
        NonRepairableRBD.from_dict(rep.to_dict())


def test_base_rbd_from_dict_dispatches():
    from repyability.rbd.rbd import RBD

    rbd = NonRepairableRBD([("s", 1), (1, "t")], {1: W([10, 2])})
    clone = RBD.from_dict(rbd.to_dict())
    assert isinstance(clone, NonRepairableRBD)


class _OwnModel:
    """A model surpyval does not know: only an ``sf``."""

    def sf(self, x):
        return np.exp(-np.asarray(x, dtype=float) / 100.0)


def test_unsupported_model_raises_not_implemented():
    rbd = NonRepairableRBD([("s", 1), (1, "t")], {1: _OwnModel()})
    with pytest.raises(NotImplementedError, match="Cannot serialise"):
        rbd.to_dict()


DISTRIBUTIONS = [
    ("Exponential", [0.01]),
    ("Weibull", [100, 2]),
    ("Normal", [100, 10]),
    ("LogNormal", [4, 0.5]),
    ("Gamma", [2, 0.05]),
    ("Gumbel", [100, 10]),
    ("Logistic", [100, 10]),
    ("LogLogistic", [100, 3]),
    ("ExpoWeibull", [100, 2, 1.5]),
    ("Beta", [2, 3]),
    ("Uniform", [0, 100]),
    ("FixedEventProbability", [0.1]),
    ("ExactEventTime", [50.0]),
    ("Hypoexponential", [0.01, 0.02]),
]


@pytest.mark.parametrize(
    "extras",
    [{}, {"gamma": 5.0}, {"lfp_p": 0.9}, {"f0": 0.1}],
    ids=["plain", "offset", "p", "f0"],
)
@pytest.mark.parametrize(
    "name, params", DISTRIBUTIONS, ids=[d[0] for d in DISTRIBUTIONS]
)
def test_every_surpyval_distribution_round_trips(name, params, extras):
    try:
        model = getattr(surv, name).from_params(params, **extras)
    except Exception:
        pytest.skip(f"surpyval builds no {name} with {extras}")
    rbd = NonRepairableRBD([("s", 1), (1, "t")], {1: model})
    back = NonRepairableRBD.from_json(rbd.to_json())
    t = np.array([0.0, 0.5, 20.0, 50.0, 90.0, 150.0])
    with np.errstate(all="ignore"):  # e.g. log(0) in a LogNormal's sf(0)
        np.testing.assert_allclose(back.sf(t), rbd.sf(t), rtol=0, atol=1e-14)


def test_a_fit_keeps_its_covariance():
    # So parameter uncertainty can still be propagated after loading.
    with numpy_seed(3):
        fitted = surv.Weibull.fit(W([100, 2]).random(40))
    rbd = NonRepairableRBD([("s", 1), (1, "t")], {1: fitted})
    back = NonRepairableRBD.from_json(rbd.to_json())
    np.testing.assert_allclose(
        np.asarray(back.reliabilities[1].hess_inv),
        np.asarray(fitted.hess_inv),
    )
    draws = {"x": 50.0, "uncertainty": {1: "fit"}, "n_draws": 200, "seed": 1}
    np.testing.assert_allclose(
        back.sf_uncertainty(**draws).samples,
        rbd.sf_uncertainty(**draws).samples,
    )


def test_tuple_node_names_survive_json():
    # JSON has no tuples: they come back as lists, which are not hashable,
    # so loading used to fail. Every place a node name can appear is
    # covered: edges, k, the input/output nodes, the model entries, a
    # repeated node's reference, and CCF group members.
    from surpyval import FixedEventProbability as FEP

    from repyability import BetaFactor, CCFGroup, PerfectReliability

    p1, p2, ps, ps2 = ("pump", 1), ("pump", 2), ("ps", 0), ("ps", 1)
    junction = ("v", ("nested", 0))
    unit = W([100, 2])
    rbd = NonRepairableRBD(
        [
            (("in",), ps),
            (ps, p1),
            (p1, junction),
            (("in",), ps2),
            (ps2, p2),
            (p2, junction),
            (junction, ("out",)),
        ],
        {
            ps: FEP.from_params(0.05),
            ps2: ps,
            p1: unit,
            p2: unit,
            junction: PerfectReliability,
        },
        k={junction: 1},
        input_node=("in",),
        output_node=("out",),
        ccf_groups=[CCFGroup([p1, p2], BetaFactor(0.1))],
    )
    clone = NonRepairableRBD.from_json(rbd.to_json())
    assert clone.sf(30) == rbd.sf(30)
    assert clone.repeated == {ps2: ps}
    assert (clone.input_node, clone.output_node) == (("in",), ("out",))
    assert tuple(clone.ccf_groups[0].members) == (p1, p2)

    repairable = RepairableRBD(
        [(("s",), p1), (p1, ("t",))],
        {p1: {"reliability": E([0.1]), "repairability": E([1.0])}},
    )
    repairable_clone = RepairableRBD.from_json(repairable.to_json())
    assert repairable_clone.nodes == [p1]
    assert (
        repairable_clone.mean_availability() == repairable.mean_availability()
    )
