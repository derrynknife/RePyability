import surpyval as surv
import pytest
from repyability.rbd.rbd import RBD
import numpy as np


# Fixed probability distributions
# TODO: move this to SurPyval
class FixedProbability:
    def sf(self, x):
        return np.ones_like(x) * self.p

    def ff(self, x):
        return 1 - (np.ones_like(x) * self.p)


class FixedProbabilityFitter:
    @classmethod
    def from_params(cls, p):
        out = FixedProbability()
        out.p = p
        return out


# Test RBDs as pytest fixtures
@pytest.fixture
def rbd1():
    """Example 6.10 from Modarres & Kaminskiy."""
    qp = 0.03
    qv = 0.01
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [[1, 2], [1, 3], [2, 4], [3, 4], [4, 5]]
    components = {
        2: FixedProbabilityFitter.from_params(1 - qp),
        3: FixedProbabilityFitter.from_params(1 - qp),
        4: FixedProbabilityFitter.from_params(1 - qv),
    }

    return RBD(nodes=nodes, components=components, edges=edges)


@pytest.fixture
def rbd2():
    edges = [[1, 2], [2, 3], [2, 4], [4, 7], [3, 5], [5, 6], [6, 7], [7, 8]]
    nodes = {
        1: "input_node",
        8: "output_node",
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
    }
    components = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
        5: surv.Weibull.from_params([15, 1.2]),
        6: surv.Weibull.from_params([80, 10]),
        7: [
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
        ],
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd_series():
    """A simple RBD with three intermediate nodes in series."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [[1, 2], [2, 3], [3, 4], [4, 5]]
    components = {
        2: surv.Weibull.from_params([20, 2]),
        3: surv.Weibull.from_params([100, 3]),
        4: surv.Weibull.from_params([50, 20]),
    }
    return RBD(nodes, components, edges)


@pytest.fixture
def rbd_parallel():
    """A simple RBD with three intermediate nodes in parallel."""
    nodes = {1: "input_node", 2: 2, 3: 3, 4: 4, 5: "output_node"}
    edges = [[1, 2], [1, 3], [1, 4], [2, 5], [3, 5], [4, 5]]
    components = {
        2: FixedProbabilityFitter.from_params(0.8),
        3: FixedProbabilityFitter.from_params(0.9),
        4: FixedProbabilityFitter.from_params(0.85),
    }
    return RBD(nodes, components, edges)


# Tests
def test_rbd_components(rbd1, rbd2):
    # Check components are correct lengths
    assert len(rbd1.components) == 5
    assert len(rbd2.components) == 8


def test_rbd_all_path_sets(rbd1, rbd2):
    assert list(rbd1.all_path_sets()) == [[1, 2, 4, 5], [1, 3, 4, 5]]
    assert list(rbd2.all_path_sets()) == [
        [1, 2, 3, 5, 6, 7, 8],
        [1, 2, 4, 7, 8],
    ]


def test_rbd_sf_series(rbd_series):
    t = 5
    assert pytest.approx(
        rbd_series.components[2].sf(t)
        * rbd_series.components[3].sf(t)
        * rbd_series.components[4].sf(t)
    ) == rbd_series.sf(t)


def test_rbd_sf_parallel(rbd_parallel):
    t = 2
    assert pytest.approx(
        (1 - rbd_parallel.components[2].sf(t))
        * (1 - rbd_parallel.components[3].sf(t))
        * (1 - rbd_parallel.components[4].sf(t))
    ) == 1 - rbd_parallel.sf(t)


def test_rbd_sf_composite(rbd1):
    """Tests with an RBD with both parallel and series components."""
    t = 2
    assert pytest.approx(
        (1 - (1 - rbd1.components[2].sf(t)) * (1 - rbd1.components[3].sf(t)))
        * rbd1.components[4].sf(t)
    ) == rbd1.sf(t)


# TODO: Test importance calcs, need to fix survival function first though
# def test_rbd_fussel_vessely_path_set(rbd1):
#     # PS = Path Set
#     # PS1 = [1, 2, 4, 5]
#     # PS2 = [1, 3, 4, 5]
#     # Q(PS1) = Q(PS2) = Q23 * Q4
#     # Note Q2 = Q3 = Q23

#     Q_

#     # Path set unreliability
#     Q_PS = (1 - rbd1.components[2].sf(1)) * (1 - rbd1.components[4].sf(1))

#     # System unreliability
#     Q_sys =

#     # Fussel-Vessely importance of components 2 and 3
#     I_FV_23 = Q_PS / Q_sys

#     # Fussel-Vessely importance of comonent 4
#     rbd1.fussel_vessely()
