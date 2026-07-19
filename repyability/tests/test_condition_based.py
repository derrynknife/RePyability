"""
Tests the condition-based ("digital twin") evaluation layer:
NodeState, sf_given_state, remaining_life and importances_given_state.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, NonRepairableRBD
from repyability.rbd.standby_node import StandbyModel
from repyability.utils.wrappers import conditional_survival


def _weibull_series() -> NonRepairableRBD:
    return NonRepairableRBD(
        [(1, 2), (2, 3), (3, 4), (4, 5)],
        {
            2: surv.Weibull.from_params([20, 2]),
            3: surv.Weibull.from_params([100, 3]),
            4: surv.Weibull.from_params([50, 20]),
        },
    )


def _weibull_parallel() -> NonRepairableRBD:
    return NonRepairableRBD(
        [(1, 2), (1, 3), (2, 4), (3, 4)],
        {
            2: surv.Weibull.from_params([50, 3]),
            3: surv.Weibull.from_params([50, 3]),
        },
    )


# -- NodeState ------------------------------------------------------------


def test_nodestate_defaults():
    ns = NodeState()
    assert ns.age == 0.0
    assert ns.alive is True


def test_nodestate_is_frozen():
    ns = NodeState(age=5)
    with pytest.raises(Exception):
        ns.age = 10  # type: ignore[misc]


def test_nodestate_negative_age_raises():
    with pytest.raises(ValueError, match="non-negative"):
        NodeState(age=-1.0)


# -- sf_given_state -------------------------------------------------------


def test_empty_state_matches_sf():
    rbd = _weibull_series()
    assert rbd.sf_given_state(10.0, {}) == pytest.approx(rbd.sf(10.0))
    # state omitted entirely behaves the same
    assert rbd.sf_given_state(10.0) == pytest.approx(rbd.sf(10.0))


def test_zero_horizon_is_one():
    rbd = _weibull_series()
    state = {2: NodeState(age=5), 3: NodeState(age=8)}
    assert rbd.sf_given_state(0.0, state) == pytest.approx(1.0)


def test_single_node_matches_conditional_survival():
    model = surv.Weibull.from_params([20, 2])
    rbd = NonRepairableRBD([(1, 2), (2, 3)], {2: model})
    expected = conditional_survival(model, 10.0, 5.0)
    assert rbd.sf_given_state(10.0, {2: NodeState(age=5)}) == pytest.approx(
        expected
    )


def test_scalar_and_array_contract():
    rbd = _weibull_series()
    state = {2: NodeState(age=5)}
    assert isinstance(rbd.sf_given_state(10.0, state), float)
    out = rbd.sf_given_state(np.array([5.0, 10.0, 15.0]), state)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_aging_reduces_reliability_for_wearing_component():
    """For an increasing-failure-rate (beta > 1) component, an aged unit is
    less reliable over the next x than a fresh one."""
    rbd = _weibull_series()
    fresh = rbd.sf_given_state(10.0, {})
    aged = rbd.sf_given_state(10.0, {2: NodeState(age=30)})
    assert aged < fresh


def test_failed_node_in_series_is_zero():
    rbd = NonRepairableRBD(
        [(1, 2), (2, 3), (3, 4)],
        {
            2: surv.Weibull.from_params([20, 2]),
            3: surv.Weibull.from_params([30, 2]),
        },
    )
    assert rbd.sf_given_state(
        5.0, {2: NodeState(alive=False)}
    ) == pytest.approx(0.0)


def test_failed_node_in_parallel_reduces_to_survivor():
    rbd = _weibull_parallel()
    expected = rbd.reliabilities[3].sf(5.0)
    assert rbd.sf_given_state(
        5.0, {2: NodeState(alive=False)}
    ) == pytest.approx(expected)


def test_cut_and_path_methods_agree():
    rbd = _weibull_parallel()
    state = {2: NodeState(age=20)}
    assert rbd.sf_given_state(5.0, state, method="c") == pytest.approx(
        rbd.sf_given_state(5.0, state, method="p")
    )


# -- remaining_life -------------------------------------------------------


def test_remaining_life_round_trips():
    rbd = _weibull_parallel()
    state = {2: NodeState(age=20)}
    rul = rbd.remaining_life(0.9, state)
    assert rbd.sf_given_state(rul, state) == pytest.approx(0.9)


def test_remaining_life_collapses_as_redundancy_worn():
    rbd = _weibull_parallel()
    fresh = rbd.remaining_life(0.9, {})
    one_worn = rbd.remaining_life(0.9, {2: NodeState(age=40)})
    both_worn = rbd.remaining_life(
        0.9, {2: NodeState(age=40), 3: NodeState(age=40)}
    )
    assert fresh > one_worn > both_worn


def test_remaining_life_exponential_is_memoryless():
    """An exponential component is memoryless, so ageing it leaves the RUL
    unchanged (equal to the as-new time_to_reliability)."""
    rbd = NonRepairableRBD(
        [(1, 2), (2, 3)], {2: surv.Exponential.from_params([0.1])}
    )
    as_new = rbd.time_to_reliability(0.5)
    aged = rbd.remaining_life(0.5, {2: NodeState(age=25)})
    assert aged == pytest.approx(as_new, rel=1e-6)


def test_remaining_life_unreachable_target_raises():
    """A failed node in series drops reliability to 0 at t=0, so no positive
    target is reachable."""
    rbd = NonRepairableRBD(
        [(1, 2), (2, 3), (3, 4)],
        {
            2: surv.Weibull.from_params([20, 2]),
            3: surv.Weibull.from_params([30, 2]),
        },
    )
    with pytest.raises(ValueError, match="exceeds the system reliability"):
        rbd.remaining_life(0.5, {2: NodeState(alive=False)})


def test_remaining_life_on_fixed_rbd_raises():
    from surpyval import FixedEventProbability

    rbd = NonRepairableRBD(
        [(1, 2), (2, 3)],
        {2: FixedEventProbability.from_params(1 - 0.9)},
    )
    with pytest.raises(ValueError, match="does not vary with time"):
        rbd.remaining_life(0.5, {})


# -- importances_given_state ---------------------------------------------


def test_importances_empty_state_matches_base_measures():
    rbd = _weibull_parallel()
    imp = rbd.importances_given_state(5.0, {})
    birnbaum = rbd.birnbaum_importance(5.0)
    criticality = rbd.criticality_importance(5.0)
    for node in birnbaum:
        assert imp["birnbaum"][node] == pytest.approx(birnbaum[node])
        assert imp["criticality"][node] == pytest.approx(criticality[node])


def test_importances_structure_and_state_dependence():
    rbd = _weibull_parallel()
    baseline = rbd.importances_given_state(5.0, {})
    conditioned = rbd.importances_given_state(5.0, {2: NodeState(age=40)})
    assert set(baseline) == {"birnbaum", "criticality"}
    assert set(baseline["birnbaum"]) == {2, 3}
    # ageing a component shifts the importance ranking
    assert baseline["birnbaum"][3] != pytest.approx(conditioned["birnbaum"][3])


def test_importances_scalar_and_array_contract():
    rbd = _weibull_parallel()
    state = {2: NodeState(age=20)}
    scalar = rbd.importances_given_state(5.0, state)
    assert all(isinstance(v, float) for v in scalar["birnbaum"].values())
    arr = rbd.importances_given_state(np.array([5.0, 10.0]), state)
    assert all(
        isinstance(v, np.ndarray) and v.shape == (2,)
        for v in arr["birnbaum"].values()
    )


# -- validation -----------------------------------------------------------


def test_unknown_node_in_state_raises():
    rbd = _weibull_series()
    with pytest.raises(ValueError, match="Unknown node"):
        rbd.sf_given_state(5.0, {"nope": NodeState(age=1)})


def test_state_value_must_be_nodestate():
    rbd = _weibull_series()
    with pytest.raises(TypeError, match="must be a NodeState"):
        rbd.sf_given_state(5.0, {2: 5.0})


def test_state_must_be_a_mapping():
    rbd = _weibull_series()
    with pytest.raises(TypeError, match="state must be a dict"):
        rbd.sf_given_state(5.0, [NodeState(age=1)])


def test_input_output_node_state_raises():
    rbd = _weibull_series()
    with pytest.raises(ValueError, match="input"):
        rbd.sf_given_state(5.0, {1: NodeState(age=1)})


def test_dynamic_node_state_raises():
    rbd = NonRepairableRBD(
        [(1, 2), (2, 3), (3, 4)],
        {
            2: surv.Weibull.from_params([20, 2]),
            3: StandbyModel(
                [
                    surv.Weibull.from_params([5, 1.1]),
                    surv.Weibull.from_params([5, 1.1]),
                ]
            ),
        },
    )
    with pytest.raises(ValueError, match="ordinary distribution"):
        rbd.sf_given_state(5.0, {3: NodeState(age=2)})
