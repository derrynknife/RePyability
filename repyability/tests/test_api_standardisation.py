"""Tests for the standardised API across RBD/NonRepairableRBD/RepairableRBD.

Covers the numpy-style return contract (scalar in -> float out, array in ->
array out), decorator introspection (docstrings/names preserved), constructor
parity, and importance-measure conditioning via working_nodes/broken_nodes.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD

FEP = surv.FixedEventProbability


def _weibull_series():
    return NonRepairableRBD(
        [("s", 1), (1, 2), (2, "t")],
        {
            1: surv.Weibull.from_params([20, 2]),
            2: surv.Weibull.from_params([100, 3]),
        },
    )


def _fixed_parallel():
    return NonRepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {"A": FEP.from_params(0.2), "B": FEP.from_params(0.4)},
    )


def _repairable_series():
    comps = {
        "A": {
            "reliability": surv.Exponential.from_params([0.2]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
        "B": {
            "reliability": surv.Exponential.from_params([0.5]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
    }
    return RepairableRBD([("s", "A"), ("A", "B"), ("B", "t")], comps)


# --- Scalar/array return contract ------------------------------------------


def test_scalar_in_float_out_array_in_array_out():
    rbd = _weibull_series()
    for method in ("sf", "ff", "df", "hf", "Hf"):
        scalar_result = getattr(rbd, method)(10.0)
        assert isinstance(scalar_result, float), method
        array_result = getattr(rbd, method)(np.array([10.0, 20.0, 30.0]))
        assert isinstance(array_result, np.ndarray), method
        assert array_result.shape == (3,), method
        assert array_result[0] == pytest.approx(scalar_result), method


def test_scalar_contract_for_dict_returns():
    rbd = _weibull_series()
    scalar_dict = rbd.node_sf(10.0)
    assert all(isinstance(v, float) for v in scalar_dict.values())
    array_dict = rbd.node_sf(np.array([10.0, 20.0]))
    assert all(
        isinstance(v, np.ndarray) and v.shape == (2,)
        for v in array_dict.values()
    )
    importance = rbd.birnbaum_importance(10.0)
    assert all(isinstance(v, float) for v in importance.values())


def test_cs_scalar_contract():
    rbd = _weibull_series()
    assert isinstance(rbd.cs(10.0, 20.0), float)
    assert isinstance(rbd.cs(np.array([10.0, 20.0]), 20.0), np.ndarray)


def test_fixed_rbd_honours_array_shape():
    rbd = _fixed_parallel()
    assert rbd.sf(1) == pytest.approx(0.92)
    assert isinstance(rbd.sf(1), float)
    # A fixed RBD is constant in time but still honours the input shape.
    out = rbd.sf(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, [0.92, 0.92, 0.92])
    # And x=None is allowed (time is irrelevant).
    assert rbd.sf() == pytest.approx(0.92)


def test_time_varying_rbd_requires_x():
    rbd = _weibull_series()
    with pytest.raises(ValueError, match="time-varying"):
        rbd.sf()


def test_mean_availability_returns_float():
    rbd = _repairable_series()
    assert isinstance(rbd.mean_availability(), float)
    assert isinstance(rbd.mean_unavailability(), float)


# --- Introspection (decorators preserve names/docstrings) -------------------


def test_decorated_methods_keep_name_and_docstring():
    for method in (
        NonRepairableRBD.sf,
        NonRepairableRBD.df,
        NonRepairableRBD.hf,
        NonRepairableRBD.Hf,
        NonRepairableRBD.birnbaum_importance,
        NonRepairableRBD.fussell_vesely,
    ):
        assert method.__name__ != "wrap"
        assert method.__doc__ is not None


# --- Constructor parity ------------------------------------------------------


def test_repairable_constructor_accepts_io_nodes_and_infeasible_mode():
    comps = {
        "A": {
            "reliability": surv.Exponential.from_params([0.2]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
    }
    rbd = RepairableRBD(
        [("s", "A"), ("A", "t")],
        comps,
        input_node="s",
        output_node="t",
        on_infeasible_rbd="raise",
    )
    assert rbd.input_node == "s"
    assert rbd.output_node == "t"


def test_repairable_missing_component_raises_clearly():
    comps = {
        "A": {
            "reliability": surv.Exponential.from_params([0.2]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
    }
    # Node "B" is in the graph but has no component definition.
    with pytest.raises(ValueError, match="no entry in"):
        RepairableRBD([("s", "A"), ("A", "B"), ("B", "t")], comps)


# --- Renames -----------------------------------------------------------------


def test_renamed_accessors_exist():
    rbd = _weibull_series()
    assert set(rbd.node_names()) == {1, 2}
    assert rbd.is_time_varying is True
    assert rbd.is_fixed is False
    fixed = _fixed_parallel()
    assert fixed.is_time_varying is False
    node_ff = fixed.node_ff(1)
    node_sf = fixed.node_sf(1)
    for node in node_sf:
        assert node_ff[node] == pytest.approx(1 - node_sf[node])


# --- Importance conditioning -------------------------------------------------


def test_nonrepairable_importance_conditioning():
    p = _fixed_parallel()
    # With B broken the parallel system reduces to A alone, whose Birnbaum
    # importance is then 1.
    bi = p.birnbaum_importance(1, broken_nodes=["B"])
    assert bi["A"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unknown node"):
        p.birnbaum_importance(1, working_nodes=["Z"])


def test_repairable_importances_return_floats_and_condition():
    rbd = _repairable_series()
    bi = rbd.birnbaum_importance()
    assert all(isinstance(v, float) for v in bi.values())
    # In series, conditioning on A always-working makes B's Birnbaum
    # importance exactly 1.
    bi_forced = rbd.birnbaum_importance(working_nodes=["A"])
    assert bi_forced["B"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unknown node"):
        rbd.fussell_vesely(working_nodes=["Z"])
