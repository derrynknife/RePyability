"""
Tests miscellaneous cases for the NonRepairableRBD class.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""
import numpy as np

from repyability.rbd.helper_classes import PerfectReliability as pr
from repyability.rbd.helper_classes import PerfectUnreliability as pu


# Check components are correct lengths
def test_helper_classes():
    assert np.all(pr.sf(np.random.rand(10)) == 1)
    assert np.all(pr.ff(np.random.rand(10)) == 0)
    assert np.all(pr.random(100) == np.inf)

    assert np.all(pu.sf(np.random.rand(10)) == 0)
    assert np.all(pu.ff(np.random.rand(10)) == 1)
    assert np.all(pu.random(100) == 0)
