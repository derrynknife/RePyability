"""What 0.13 deprecated is gone in 0.14: ``optimal_inspection_intervals``'
``offsets``, renamed ``offset_shares`` (#222); calling
``CapacityDistribution.mean()``, a property since 0.13 (#235); the
simulation options given to the exact ``mean`` of a ``StandbyModel``,
``LoadSharingModel`` or ``DegradingNode`` (#233); and ``rbd._sensitivity``'s
``levers`` and ``Lever``, the names callers used before the levers were
public (#244)."""

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    BetaFactor,
    CCFGroup,
    DegradingNode,
    LoadSharingModel,
    NonRepairableRBD,
    RepairableRBD,
    StandbyModel,
)
from repyability.rbd import _sensitivity

W, E = surv.Weibull.from_params, surv.Exponential.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def valve():
    return {
        "reliability": E([2e-6]),
        "repairability": "instant",
        "inspection": {"interval": 8760.0, "cost": 500.0},
    }


def test_offsets_is_refused_for_offset_shares():
    pair = RepairableRBD(
        [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        {"v1": valve(), "v2": valve()},
        ccf_groups=[CCFGroup(["v1", "v2"], BetaFactor(0.1))],
    )
    with pytest.raises(TypeError, match="offsets"):
        pair.optimal_inspection_intervals(allowed=[8760.0], offsets=[0.0, 0.5])
    plan = pair.optimal_inspection_intervals(
        allowed=[8760.0], offset_shares=[0.0, 0.5]
    )
    assert plan.offsets == {"v1": 0.0, "v2": 4380.0}


def test_the_capacity_mean_is_not_called():
    rbd = NonRepairableRBD(
        PAIR, {"a": W([100, 2]), "b": W([90, 2])}, capacity={"a": 1, "b": 2}
    )
    capacity = rbd.capacity_distribution(50.0)
    assert type(capacity.mean) is float
    with pytest.raises(TypeError, match="not callable"):
        capacity.mean()
    over_time = rbd.capacity_distribution([10.0, 50.0])
    assert type(over_time.mean) is np.ndarray
    with pytest.raises(TypeError, match="not callable"):
        over_time.mean()


def aft():
    x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    return surv.ExponentialAFT.fit(x, Z=load)


@pytest.mark.parametrize(
    "build",
    [
        lambda: StandbyModel([E([0.01]), E([0.01])]),
        lambda: LoadSharingModel([aft(), aft()], load=2.0, k=1),
        lambda: DegradingNode([(100, W([1000, 2])), (50, E([1 / 500]))]),
    ],
    ids=["standby", "load sharing", "degrading"],
)
@pytest.mark.parametrize("name", ["mc_samples", "seed"])
def test_an_exact_means_simulation_options_are_refused(build, name):
    node = build()
    with pytest.raises(TypeError, match=f"takes {name} only with"):
        node.mean(**{name: 10})
    assert node.mean(method="simulate", mc_samples=200, seed=1) > 0.0


def test_the_levers_private_names_are_gone():
    assert not hasattr(_sensitivity, "levers")
    assert not hasattr(_sensitivity, "Lever")
