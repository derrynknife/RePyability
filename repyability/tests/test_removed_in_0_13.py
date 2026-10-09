"""What 0.12 deprecated is gone in 0.13: calling ``SparesDemand.mean()``
and ``std()``, properties since 0.12 (#184), and ``StandbyModel``'s and
``LoadSharingModel``'s ``mc_samples``, ``lower`` and ``seed``, which set a
fit to simulated lifetimes that 0.12 removed (#149)."""

import numpy as np
import pytest
import surpyval as surv

from repyability import LoadSharingModel, RepairableRBD, StandbyModel

W, E = surv.Weibull.from_params, surv.Exponential.from_params


def aft():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=400)
    x = rng.exponential(100.0 / np.exp(0.6 * (load - 1.0)), size=400)
    return surv.ExponentialAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))


def test_the_spares_demand_values_are_not_called():
    rbd = RepairableRBD(
        [("s", "pump"), ("pump", "t")],
        {"pump": {"reliability": E([0.01]), "repairability": "instant"}},
    )
    demand = rbd.spares_demand(1000.0)["pump"]
    assert type(demand.mean) is float and type(demand.std) is float
    for value in (demand.mean, demand.std):
        with pytest.raises(TypeError, match="not callable"):
            value()


@pytest.mark.parametrize("name", ["mc_samples", "lower", "seed"])
def test_the_fits_settings_are_refused(name):
    with pytest.raises(TypeError, match=name):
        StandbyModel([W([100, 2])] * 2, **{name: 1})
    with pytest.raises(TypeError, match=name):
        LoadSharingModel([aft()] * 2, load=2.0, **{name: 1})


def test_a_standby_models_options_are_given_by_name():
    # A switching probability or dormancy factor given by position was
    # after the removed settings, so it is refused rather than misread.
    units = [E([0.01])] * 2
    with pytest.raises(TypeError):
        StandbyModel(units, 1, None, None, 0.9)
    with pytest.raises(TypeError):
        StandbyModel(units, 1, 0.9)
    given = StandbyModel(units, 1, switching_probability=0.9)
    assert given.switching_probability == 0.9
    assert float(given.mean()) == pytest.approx(100.0 + 0.9 * 100.0)
