"""The old argument names (#105) keep working until 1.0, with a
DeprecationWarning: the number of simulations is ``mc_samples`` everywhere,
and its cap ``max_samples``."""

import warnings

import pytest
import surpyval as surv

from repyability import (
    RBD,
    LoadSharingModel,
    RepairableRBD,
    RepeatedStandbyNode,
    StandbyModel,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params


def plant(mttf=10.0):
    component = {"reliability": E([1 / mttf]), "repairability": E([1.0])}
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": component, "b": component},
    )


@pytest.fixture
def unit_aft():
    import numpy as np

    rng = np.random.default_rng(1)
    load = rng.uniform(0.5, 2.0, size=300)
    lives = rng.weibull(2.0, size=300) * 80.0 / np.exp(0.4 * (load - 1))
    return surv.WeibullAFT.fit(lives + 1e-3, Z=load.reshape(-1, 1))


def test_the_old_simulation_counts_still_run_and_warn():
    new = plant().availability(50.0, mc_samples=40, seed=1)
    with pytest.warns(DeprecationWarning, match="N is deprecated"):
        old = plant().availability(50.0, N=40, seed=1)
    assert (
        old.mean_availability_interval().estimate
        == new.mean_availability_interval().estimate
    )
    with pytest.warns(DeprecationWarning, match="max_N is deprecated"):
        plant().availability(
            50.0, mc_samples=20, seed=1, tolerance=0.5, max_N=40
        )
    with pytest.warns(DeprecationWarning, match="N is deprecated"):
        plant().compare(plant(20.0), 50.0, N=20, seed=1)
    with pytest.raises(TypeError, match="not both"):
        plant().availability(50.0, mc_samples=10, N=10)


def test_the_old_node_model_names_still_work_and_warn(unit_aft):
    w = W([100, 2])
    with pytest.warns(DeprecationWarning, match="n_sims is deprecated"):
        old = StandbyModel([w] * 3, k=2, n_sims=500, seed=1)
    new = StandbyModel([w] * 3, k=2, mc_samples=500, seed=1)
    assert old.mean() == new.mean() and new.mc_samples == 500
    with pytest.warns(DeprecationWarning, match="n_sims is deprecated"):
        assert new.n_sims == 500
    with pytest.warns(DeprecationWarning, match="N is deprecated"):
        assert new.mean(N=200, seed=2) == new.mean(mc_samples=200, seed=2)
    with pytest.warns(DeprecationWarning, match="n_sims is deprecated"):
        LoadSharingModel([unit_aft] * 2, load=2.0, n_sims=200, seed=1)
    with pytest.warns(DeprecationWarning, match="ignores N"):
        RepeatedStandbyNode(w, 2, N=1000)
    with warnings.catch_warnings():
        # The new names don't warn. (A simulated StandbyModel's fit does,
        # with a FutureWarning: see test_deprecated_models.py.)
        warnings.simplefilter("error", DeprecationWarning)
        RepeatedStandbyNode(w, 2)
        StandbyModel([w] * 3, k=2, mc_samples=100, seed=1).mean()


def test_files_saved_with_n_sims_still_load():
    from repyability import NonRepairableRBD

    w = W([100, 2])
    group = StandbyModel([w] * 3, k=2, mc_samples=321, seed=1)
    saved = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": group}).to_dict()
    model = saved["reliabilities"][0]["model"]
    assert model["mc_samples"] == 321
    model["n_sims"] = model.pop("mc_samples")  # as 0.10 saved it
    assert RBD.from_dict(saved).reliabilities["a"].mc_samples == 321


def test_repairables_old_name_still_works_and_warns():
    from repyability import Repairable

    model = (
        surv.recurrent.CrowAMSAA.from_params([0.5, 1.5])
        if hasattr(surv, "recurrent")
        else None
    )
    if model is None:
        pytest.skip("surpyval has no recurrent models")
    unit = Repairable(model)
    unit.set_repair_and_overhaul_costs(1.0, 10.0)
    new = unit.cost(50.0, seed=1, mc_samples=50)
    with pytest.warns(DeprecationWarning, match="n_simulations is deprecated"):
        old = unit.cost(50.0, seed=1, n_simulations=50)
    assert old == new
