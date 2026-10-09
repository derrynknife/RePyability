"""surpyval's fitted repairable-unit models as components (#269): a
Poisson process (CrowAMSAA, Duane, HPP) is minimal repair of the life
whose cumulative hazard is its cumulative intensity, and a generalised
renewal process (Kijima I or II) its life repaired by its Kijima model and
restoration factor."""

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD

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
    process = surv.GeneralizedRenewal.fit(
        TIMES, dist=surv.Weibull, kijima=kijima
    )
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


# -- starting from a virtual age (#269) --------------------------------------

LIFE = surv.Weibull.from_params([100.0, 2.5])


def H(t):
    return (np.asarray(t) / 100.0) ** 2.5


def kijima(model, q, **more):
    return RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": LIFE,
                "repairability": "instant",
                "repair": {"model": model, "q": q},
                **more,
            }
        },
    )


def reference_failures(model, q, virtual, age, horizon, n, seed):
    """The mean failures by ``horizon`` of a unit at virtual age
    ``virtual`` at its last repair and ``age`` since, repaired in no time,
    written out from Kijima's definition: each life ``X`` from virtual age
    ``v`` has ``H(v + X) = H(v) + E``; a repair after operating ``x``
    since the last takes ``v`` to ``v + q x`` (I) or ``q (v + x)`` (II)."""
    rng = np.random.default_rng(seed)
    total = 0
    for _ in range(n):
        v, since, t = virtual, age, 0.0
        while True:
            now = v + since
            left = 100.0 * (H(now) + rng.exponential()) ** (1 / 2.5) - now
            t += left
            if t > horizon:
                break
            total += 1
            x = since + left
            v = v + q * x if model == "kijima1" else q * (v + x)
            since = 0.0
    return total / n


@pytest.mark.parametrize("model", ["kijima1", "kijima2"])
@pytest.mark.parametrize(
    "virtual, age", [(0.0, 60.0), (150.0, 0.0), (90.0, 40.0)]
)
def test_a_unit_starts_from_its_virtual_age(model, virtual, age):
    n = 20_000
    run = kijima(model, 0.5).availability(
        t_simulation=200.0,
        mc_samples=n,
        seed=5,
        state={"a": NodeState(age=age, virtual_age=virtual or None)},
    )
    expected = reference_failures(model, 0.5, virtual, age, 200.0, n, 6)
    # Two estimates of a mean of about 3-6 failures (sd about 1.5), each
    # from 20,000 runs.
    assert run.system_failures / n == pytest.approx(expected, abs=0.06)


def test_minimal_repair_from_a_virtual_age_is_exact():
    # q = 1: the failures by T are H(v + a + T) - H(v + a).
    n = 20_000
    run = kijima("kijima1", 1.0).availability(
        t_simulation=50.0,
        mc_samples=n,
        seed=7,
        state={"a": NodeState(age=30.0, virtual_age=120.0)},
    )
    exact = float(H(200.0) - H(150.0))
    assert run.system_failures / n == pytest.approx(exact, abs=0.05)


def test_a_unit_down_returns_at_its_virtual_age():
    # Down, its repair all but over (a mean of 1e-6): then as a unit up at
    # its virtual age, age 0.
    rbd = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": LIFE,
                "repairability": surv.Exponential.from_params([1e6]),
                "repair": {"model": "kijima1", "q": 1.0},
            }
        },
    )
    n = 20_000
    run = rbd.availability(
        t_simulation=50.0,
        mc_samples=n,
        seed=8,
        state={"a": NodeState(alive=False, virtual_age=150.0)},
    )
    assert run.system_failures / n == pytest.approx(
        float(H(200.0) - H(150.0)), abs=0.05
    )


def test_a_fitted_units_state_from_unit_states():
    # surpyval's unit_states gives the virtual age now and the time since
    # the last failure: the state is age=since_failure,
    # virtual_age=virtual_age - since_failure.
    # The fitted unit, at the end of its history: its last failure just
    # repaired, at a virtual age above 0 (Kijima II, q above 0).
    process = surv.GeneralizedRenewal.fit(
        TIMES, dist=surv.Weibull, kijima="ii"
    )
    now = process.unit_states().iloc[0]
    since = float(now["since_failure"])
    state = NodeState(age=since, virtual_age=float(now["virtual_age"]) - since)
    assert state.virtual_age > 0.0
    rbd = one(process)
    started, new = (
        rbd.availability(
            t_simulation=100.0, mc_samples=4000, seed=1, state=given
        )
        for given in ({"a": state}, None)
    )
    # Older than new, it fails more often (its life is a wearing Weibull).
    assert started.system_failures > new.system_failures


def test_a_virtual_age_needs_imperfect_repair_and_a_simulation():
    with pytest.raises(ValueError, match="no virtual age"):
        one(LIFE).availability(
            t_simulation=10.0,
            mc_samples=2,
            state={"a": NodeState(age=5.0, virtual_age=3.0)},
        )
    with pytest.raises(NotImplementedError, match="taken by the simulations"):
        kijima("kijima1", 1.0).point_availability(
            10.0, state={"a": NodeState(age=5.0, virtual_age=3.0)}
        )
    with pytest.raises(NotImplementedError, match="since it was renewed"):
        kijima("kijima1", 0.5, replace_after=3).availability(
            t_simulation=10.0,
            mc_samples=2,
            state={"a": NodeState(age=5.0, virtual_age=3.0)},
        )
