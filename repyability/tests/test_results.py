"""Tests for the typed result objects returned by RepairableRBD.availability().

They provide documented, typed attribute access while remaining backwards
compatible with the previous nested-dict API (subscript, keys/items/values,
``in``, ``dict()``).
"""

from collections.abc import Mapping

import numpy as np
import surpyval as surv

from repyability.rbd.repairable_rbd import RepairableRBD
from repyability.rbd.results import AvailabilityResult, Criticalities


def _result():
    comps = {
        "A": {
            "reliability": surv.Exponential.from_params([0.1]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
        "B": {
            "reliability": surv.Exponential.from_params([0.2]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
    }
    rbd = RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")], comps
    )
    return rbd.availability(t_simulation=15.0, N=100, seed=1)


def test_availability_returns_typed_result():
    r = _result()
    assert isinstance(r, AvailabilityResult)
    assert isinstance(r.criticalities, Criticalities)
    # Typed attribute access down the tree.
    assert isinstance(r.availability, np.ndarray)
    assert isinstance(r.criticalities.iou.up, dict)
    assert isinstance(
        r.criticalities.failure_criticality_index.per_system_failure, dict
    )


def test_result_backward_compatible_mapping_access():
    r = _result()
    # Old dict-style access still works and matches the typed attributes.
    assert r["availability"] is r.availability
    assert r["system_uptime"] == r.system_uptime
    assert r["criticalities"]["iou"]["up"] == r.criticalities.iou.up
    assert (
        r["criticalities"]["failure_criticality_index"]["per_system_failure"]
        == r.criticalities.failure_criticality_index.per_system_failure
    )
    # Mapping protocol.
    assert isinstance(r, Mapping)
    assert "criticalities" in r and "not_a_key" not in r
    assert set(r.keys()) == set(dict(r).keys())
    assert r.get("missing", "default") == "default"
