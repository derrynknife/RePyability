"""surpyval's fitted repairable-unit models as components (#269): a
Poisson process (CrowAMSAA, Duane, HPP) is minimal repair of the life
whose cumulative hazard is its cumulative intensity, and a generalised
renewal process (Kijima I or II) its life repaired by its Kijima model and
restoration factor."""

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD

EDGES = [("s", "a"), ("a", "t")]
TIMES = np.cumsum(np.random.default_rng(0).weibull(1.5, 60) * 10)


def one(life, **more):
    return RepairableRBD(
        EDGES, {"a": {"reliability": life, "repairability": "instant", **more}}
    )


@pytest.mark.parametrize("fitter", [surv.CrowAMSAA, surv.Duane, surv.HPP])
def test_a_poisson_process_has_its_intensity(fitter):
    # Minimal repair in no time: the expected number of failures by T is
    # the process's cumulative intensity, worked out exactly.
    process = fitter.fit(TIMES)
    rbd = one(process)
    life = rbd._init_args["components"]["a"]["reliability"]
    t = np.array([1.0, 10.0, 100.0, 500.0])
    np.testing.assert_allclose(life.Hf(t), process.cif(t), rtol=1e-12)
    for horizon in (100.0, 500.0):
        assert rbd.expected_failures(horizon) == pytest.approx(
            float(process.cif(horizon)), rel=1e-9
        )


@pytest.mark.parametrize("kijima", ["i", "ii"])
def test_a_generalised_renewal_process_is_its_life_and_kijima_repair(kijima):
    process = surv.GeneralizedRenewal.fit(TIMES, dist=surv.Weibull, kijima=kijima)
    repair = surv.Exponential.from_params([1.0])
    given = RepairableRBD(
        EDGES, {"a": {"reliability": process, "repairability": repair}}
    )
    spec = given._init_args["components"]["a"]
    assert spec["reliability"] is process.model
    assert spec["repair"] == {
        "model": f"kijima{len(kijima)}",
        "q": float(process.restoration),
    }
    # The same as the spec written out, simulation for simulation.
    written = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": process.model,
                "repairability": repair,
                "repair": spec["repair"],
            }
        },
    )
    runs = [
        rbd.availability(t_simulation=300.0, mc_samples=200, seed=3)
        for rbd in (given, written)
    ]
    assert runs[0].mean_availability == runs[1].mean_availability
    assert runs[0].system_failures == runs[1].system_failures


@pytest.mark.parametrize(
    "fitter", [surv.GeneralizedOneRenewal, surv.ARA], ids=["G1", "ARA"]
)
def test_renewal_models_that_are_not_kijimas_are_refused(fitter):
    process = fitter.fit(TIMES, dist=surv.Weibull)
    with pytest.raises(ValueError, match="not Kijima's.*SurPyval#833"):
        one(process)


def test_a_process_brings_its_own_repair():
    with pytest.raises(ValueError, match="brings its own repair"):
        one(surv.CrowAMSAA.fit(TIMES), repair={"model": "kijima1", "q": 0.5})


def test_a_process_is_saved_as_the_life_and_repair_it_is():
    rbd = one(surv.CrowAMSAA.fit(TIMES))
    back = RepairableRBD.from_dict(rbd.to_dict())
    assert back.expected_failures(200.0) == pytest.approx(
        rbd.expected_failures(200.0), rel=1e-12
    )
