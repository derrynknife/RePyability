"""The misspelled ``fussel_vesely`` is retained as a deprecated alias of the
corrected ``fussell_vesely`` (Fussell-Vesely). These tests check the alias
warns and returns the same result as the current method.
"""

import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD


def test_nonrepairable_alias_warns_and_matches(rbd1: NonRepairableRBD):
    with pytest.warns(DeprecationWarning):
        deprecated = rbd1.fussel_vesely(2, fv_type="c")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the current name must NOT warn
        current = rbd1.fussell_vesely(2, fv_type="c")
    assert deprecated.keys() == current.keys()
    for node in current:
        np.testing.assert_allclose(deprecated[node], current[node])


def test_repairable_alias_warns_and_matches():
    rbd = RepairableRBD(
        edges=[("s", "a"), ("a", "t")],
        components={
            "a": {
                "reliability": surv.Exponential.from_params([0.1]),
                "repairability": surv.Exponential.from_params([0.5]),
            }
        },
    )
    with pytest.warns(DeprecationWarning):
        deprecated = rbd.fussel_vesely()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        current = rbd.fussell_vesely()
    assert deprecated.keys() == current.keys()
    for node in current:
        np.testing.assert_allclose(deprecated[node], current[node])
