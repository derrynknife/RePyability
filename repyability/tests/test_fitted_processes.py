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


# -- repairs that are not Kijima's: G1, ARA, ARI, stepped with surpyval --

N_RUNS = 4000


def failures_by(process, horizon, n=N_RUNS, seed=1, **more):
    """The mean failures by ``horizon`` of one unit given as ``process``,
    repaired in no time, simulated."""
    run = one(process, **more).availability(
        t_simulation=horizon, mc_samples=n, seed=seed
    )
    return run.system_failures / n


@pytest.mark.parametrize(
    "process",
    [
        surv.ARA.fit_from_parameters([100.0, 2.5], 0.0),
        surv.ARI.fit_from_parameters([100.0, 2.5], 0.0),
    ],
    ids=["ARA", "ARI"],
)
def test_no_reduction_is_minimal_repair(process):
    # ARA and ARI with rho = 0 take nothing away at a repair: minimal
    # repair, so the expected failures by T are H(T) = (T / 100) ** 2.5.
    expected = (300.0 / 100.0) ** 2.5  # 15.6
    # 4000 runs of a Poisson count of mean 15.6: sd of the mean 0.06.
    assert failures_by(process, 300.0) == pytest.approx(expected, abs=0.25)


def test_g1_with_no_stretch_is_a_renewal_process():
    # G1 with q = 0 draws every gap from the life itself: a renewal
    # process, and with an exponential life a Poisson one, lambda * T.
    process = surv.GeneralizedOneRenewal.fit_from_parameters(
        [0.05], 0.0, dist=surv.Exponential
    )
    assert failures_by(process, 200.0) == pytest.approx(10.0, abs=0.2)


@pytest.mark.parametrize(
    "process",
    [
        surv.GeneralizedOneRenewal.fit_from_parameters([100.0, 2.0], 0.3),
        surv.ARA.fit_from_parameters([100.0, 2.5], 0.6),
        surv.ARA.fit_from_parameters([100.0, 2.5], 0.4, m=2),
        surv.ARI.fit_from_parameters([100.0, 2.5], 0.5),
    ],
    ids=["G1", "ARA1", "ARA2", "ARI1"],
)
def test_each_family_gives_surpyvals_own_count(process):
    # surpyval's own simulation of the fitted model (mcf) is the oracle.
    own = float(
        np.ravel(process.mcf(np.array([300.0]), items=20_000, random_state=3))[
            0
        ]
    )
    # Two estimates of a mean of 3-8 failures (sd about 2), from 4000 and
    # 20,000 runs.
    assert failures_by(process, 300.0) == pytest.approx(own, abs=0.15)


def test_a_fitted_g1_ara_or_ari_unit_is_simulated_and_seeded():
    times = TIMES
    for fitter in (surv.GeneralizedOneRenewal, surv.ARA, surv.ARI):
        process = fitter.fit(times)
        spec = one(process)._init_args["components"]["a"]
        assert spec["repair"] == {"model": "process", "process": process}
        runs = [failures_by(process, 300.0, n=300, seed=s) for s in (5, 5, 6)]
        # The same seed, the same run; another seed, another.
        assert runs[0] == runs[1] != runs[2]


def test_a_stepped_unit_leaves_other_units_draws_as_they_were():
    # The unit's own streams: adding it to a diagram does not move the
    # draws of a plain unit beside it.
    life = surv.Weibull.from_params([100.0, 2.0])
    plain = {"reliability": life, "repairability": "instant"}
    edges = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
    alone = RepairableRBD(edges, {"a": plain, "b": plain})
    beside = RepairableRBD(
        edges,
        {
            "a": plain,
            "b": {
                "reliability": surv.ARA.fit_from_parameters([100.0, 2.5], 0.5),
                "repairability": "instant",
            },
        },
    )
    runs = [
        rbd.simulate_timelines(t_simulation=300.0, mc_samples=50, seed=2)
        for rbd in (alone, beside)
    ]
    first, second = (run.components["a"] for run in runs)
    np.testing.assert_array_equal(first.uptime, second.uptime)
    np.testing.assert_array_equal(first.failures, second.failures)


def test_a_stepped_unit_is_saved_as_its_fitted_model():
    process = surv.ARA.fit_from_parameters([100.0, 2.5], 0.6)
    rbd = one(process)
    saved = rbd.to_dict()
    back = RepairableRBD.from_dict(saved)
    spec = back._init_args["components"]["a"]
    assert spec["repair"]["model"] == "process"
    assert spec["repair"]["process"].restoration == pytest.approx(0.6)
    runs = [
        diagram.availability(t_simulation=300.0, mc_samples=200, seed=4)
        for diagram in (rbd, back)
    ]
    assert runs[0].system_failures == runs[1].system_failures


def test_a_stepped_unit_starts_from_its_age():
    # Run 80 since new without a failure: under ARA its next failure is
    # the life's from age 80, as for any unit; then the fitted repairs.
    process = surv.ARA.fit_from_parameters([100.0, 2.5], 0.0)
    run = one(process).availability(
        t_simulation=100.0,
        mc_samples=N_RUNS,
        seed=9,
        state={"a": NodeState(age=80.0)},
    )
    # Minimal repair: H(180) - H(80).
    exact = (1.8**2.5) - (0.8**2.5)
    assert run.system_failures / N_RUNS == pytest.approx(exact, abs=0.1)


def test_what_a_stepped_unit_refuses():
    process = surv.ARA.fit_from_parameters([100.0, 2.5], 0.5)
    with pytest.raises(NotImplementedError, match="not a Kijima virtual age"):
        one(process).availability(
            t_simulation=10.0,
            mc_samples=2,
            state={"a": NodeState(age=5.0, virtual_age=3.0)},
        )
    with pytest.raises(NotImplementedError, match="own rule"):
        one(process).point_availability(10.0)
    with pytest.raises(ValueError, match="restoring it partly"):
        one(
            process,
            preventive={
                "interval": 50.0,
                "restoration": {"model": "kijima1", "q": 0.5},
            },
        )
    with pytest.raises(ValueError, match="not the life from new"):
        RepairableRBD(
            EDGES,
            {
                "a": {
                    "reliability": surv.Weibull.from_params([50.0, 2.0]),
                    "repairability": "instant",
                    "repair": {"model": "process", "process": process},
                }
            },
        )


# -- Cox-Lewis: minimal repair of its own life --------------------------------


@pytest.mark.parametrize("beta", [0.01, 0.0, -0.01])
def test_a_cox_lewis_process_has_its_intensity(beta):
    process = surv.CoxLewis.from_params([-3.0, beta])
    rbd = one(process)
    for horizon in (50.0, 300.0):
        # Exact, as for any Poisson process.
        assert rbd.expected_failures(horizon) == pytest.approx(
            float(process.cif(horizon)), rel=1e-9
        )
    assert failures_by(process, 300.0) == pytest.approx(
        float(process.cif(300.0)), rel=0.03
    )
    back = RepairableRBD.from_dict(rbd.to_dict())
    assert back.expected_failures(300.0) == pytest.approx(
        rbd.expected_failures(300.0), rel=1e-12
    )


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


def test_a_seeded_run_from_a_virtual_age_is_its_own():
    # What is left of the life at the start is drawn from the unit's own
    # stream, not numpy's global random numbers.
    rbd = kijima("kijima1", 0.5)
    state = {"a": NodeState(age=40.0, virtual_age=90.0)}
    runs = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        run = rbd.availability(
            t_simulation=200.0, mc_samples=200, seed=5, state=state
        )
        runs.append(run.system_failures)
    assert runs[0] == runs[1]


def test_a_simulation_stepped_by_hand_starts_from_a_virtual_age():
    rbd = kijima("kijima1", 0.5)
    rbd.initialize_event_queue(
        200.0, state={"a": NodeState(age=40.0, virtual_age=90.0)}
    )
    t, up = rbd.next_event()
    assert 0.0 < t < 200.0 and not up


def test_a_regression_on_a_poisson_process_is_not_read_as_one():
    # A proportional-intensity model carries a Poisson process as its
    # baseline (its dist) but its intensity needs covariates: it is not
    # translated as the bare process.
    rng = np.random.default_rng(1)
    x = np.concatenate(
        [np.cumsum(rng.weibull(1.5, 5) * 10) for _ in range(20)]
    )
    i = np.repeat(np.arange(20), 5)
    Z = (i % 2).astype(float)[:, None]
    model = surv.ProportionalIntensityNHPP.fit(x, Z=Z, i=i)
    with pytest.raises(ValueError, match="ProportionalIntensityModel"):
        one(model)
