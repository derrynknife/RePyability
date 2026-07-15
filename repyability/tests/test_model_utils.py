"""Tests for the model-capability helpers, which also double as a surpyval
compatibility guard: they pin the distribution-name assumptions the RBD code
relies on, so an upstream surpyval rename fails here loudly rather than
silently changing reliability results.
"""

import surpyval as surv
from surpyval import FixedEventProbability

from repyability.rbd._model_utils import (
    distribution_name,
    is_exponential,
    is_fixed_probability,
)
from repyability.rbd.helper_classes import PerfectReliability
from repyability.rbd.standby_node import StandbyModel


def test_distribution_name_of_parametric():
    assert distribution_name(surv.Weibull.from_params([100, 2])) == "Weibull"
    assert (
        distribution_name(surv.Exponential.from_params([0.01]))
        == "Exponential"
    )
    assert (
        distribution_name(FixedEventProbability.from_params(0.1))
        == "FixedEventProbability"
    )


def test_distribution_name_none_for_non_distribution_models():
    # Models without a surpyval ``.dist`` return None rather than raising.
    assert distribution_name(PerfectReliability) is None
    standby = StandbyModel(
        [
            surv.Weibull.from_params([5, 1.1]),
            surv.Weibull.from_params([5, 1.1]),
        ]
    )
    assert distribution_name(standby) is None
    assert distribution_name(object()) is None


def test_is_fixed_probability():
    assert is_fixed_probability(FixedEventProbability.from_params(0.2))
    assert not is_fixed_probability(surv.Weibull.from_params([100, 2]))
    assert not is_fixed_probability(PerfectReliability)


def test_is_exponential():
    assert is_exponential(surv.Exponential.from_params([0.01]))
    assert not is_exponential(surv.Weibull.from_params([100, 2]))
    assert not is_exponential(PerfectReliability)
