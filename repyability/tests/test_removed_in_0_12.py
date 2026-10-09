"""What 0.11 deprecated is gone in 0.12 (#149): the old simulation-count
names and ignored arguments raise, non-parametric RBD nodes are refused,
and the standby and load-sharing models with no exact or numerical
reliability refuse it, rather than fitting one to simulated lifetimes, and
are left to the simulations. Their constructors' simulation settings, which
set that fit, went in 0.13 (``test_removed_in_0_13``)."""

import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    DegradingNode,
    LoadSharingModel,
    NonRepairable,
    NonRepairableRBD,
    Repairable,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
)
from repyability.rbd import routes

W, E = surv.Weibull.from_params, surv.Exponential.from_params
ONE = [("s", "a"), ("a", "t")]


def km():
    return surv.KaplanMeier.fit([10.0, 30.0, 45.0, 60.0, 80.0])


def aft(baseline):
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=400)
    if baseline == "exponential":
        x = rng.exponential(100.0 / np.exp(0.6 * (load - 1.0)), size=400)
        return surv.ExponentialAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))
    x = rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1))
    return surv.WeibullAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))


def quiet(build):
    """Build, failing on any FutureWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return build()


# -- the old names and ignored arguments -------------------------------------


def unit():
    return {"reliability": W([100, 2]), "repairability": E([0.5])}


@pytest.mark.parametrize(
    "call",
    [
        lambda: RepairableRBD(ONE, {"a": unit()}).availability(10.0, N=5),
        lambda: RepairableRBD(ONE, {"a": unit()}).availability(
            10.0, tolerance=0.1, max_N=50
        ),
        lambda: RepairableRBD(ONE, {"a": unit()}).compare(
            RepairableRBD(ONE, {"a": unit()}), 10.0, N=5
        ),
        lambda: StandbyModel([W([100, 2])] * 2, n_sims=100),
        lambda: StandbyModel([W([100, 2])] * 2).mean(N=100),
        lambda: RepeatedNode(W([100, 2]), 2, "series").mean(N=100),
        lambda: NonRepairableRBD(ONE, {"a": W([100, 2])}).node_mttf(
            mc_samples=100
        ),
        lambda: RepeatedStandbyNode(W([100, 2]), 2, 10_000),
        lambda: NonRepairable(W([100, 2])).find_optimal_replacement(None),
    ],
    ids=[
        "N",
        "max_N",
        "compare's N",
        "n_sims",
        "a model's N",
        "a repeated node's N",
        "node_mttf's options",
        "a repeated standby node's N",
        "find_optimal_replacement's options",
    ],
)
def test_the_old_names_and_ignored_arguments_raise(call):
    with pytest.raises(TypeError):
        call()


def test_repairables_old_name_raises():
    from surpyval.recurrent import CrowAMSAA

    unit = Repairable(CrowAMSAA.from_params([100.0, 1.5]))
    unit.set_repair_and_overhaul_costs(10.0, 1000.0)
    with pytest.raises(TypeError):
        unit.cost(250.0, n_simulations=10)


def test_simulation_options_need_method_simulate():
    rbd = NonRepairableRBD(ONE, {"a": W([100, 2])})
    for method in ("mean", "mean_time_to_failure"):
        with pytest.raises(TypeError, match="only with method='simulate'"):
            getattr(rbd, method)(mc_samples=100)
        with pytest.raises(TypeError, match="seed"):
            getattr(rbd, method)(seed=1)
    assert rbd.mean(method="simulate", mc_samples=100, seed=1) > 0
    node = RepeatedNode(W([100, 2]), 2, "series")
    with pytest.raises(TypeError, match="only with method='simulate'"):
        node.mean(mc_samples=100)


def test_the_misspelt_alias_is_gone():
    assert not hasattr(NonRepairableRBD, "fussel_vesely")
    assert not hasattr(RepairableRBD, "fussel_vesely")


# -- non-parametric nodes ----------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        km,
        lambda: RepeatedNode(km(), 2, "parallel"),
        lambda: RepeatedStandbyNode(km(), 2),
        lambda: StandbyModel([km(), km()]),
        lambda: DegradingNode([(1.0, km()), (0.5, W([100, 2]))]),
    ],
    ids=["fit", "repeated", "repeated standby", "standby units", "stage"],
)
def test_a_non_parametric_node_is_refused(node):
    with pytest.raises(ValueError, match=r"\['a'\].*Fit a parametric"):
        NonRepairableRBD(ONE, {"a": node()})


@pytest.mark.parametrize(
    "spec",
    [
        lambda: {"reliability": km(), "repairability": E([0.5])},
        lambda: {"reliability": W([100, 2]), "repairability": km()},
        lambda: NonRepairable(km(), E([0.5])),
    ],
    ids=["life", "repair time", "NonRepairable"],
)
def test_a_non_parametric_component_is_refused(spec):
    with pytest.raises(ValueError, match=r"\['a'\].*non-parametric"):
        RepairableRBD(ONE, {"a": spec()})


def test_a_saved_diagram_with_one_is_refused_when_loaded():
    saved = NonRepairableRBD(ONE, {"a": W([100, 2])}).to_dict()
    saved["reliabilities"][0]["model"] = {
        "kind": "surpyval",
        "model": km().to_dict(),
    }
    with pytest.raises(ValueError, match="non-parametric"):
        NonRepairableRBD.from_dict(saved)


def test_a_standalone_non_repairable_keeps_its_fit():
    # Its maintenance policies take a Kaplan-Meier fit, searching ages
    # within the data.
    part = NonRepairable(km(), E([0.5]))
    part.set_costs_planned_and_unplanned(1.0, 5.0)
    assert 0.0 < part.find_optimal_replacement() <= 80.0


# -- models with no exact or numerical reliability ---------------------------

NO_RELIABILITY = [
    (
        lambda: StandbyModel([W([100, 2])] * 3, k=2, dormancy_factor=0.5),
        "warm standby with 2 units operating",
    ),
    (
        lambda: StandbyModel(
            [W([100, 2]), W([80, 1.5]), W([100, 2]), W([90, 3])], k=3
        ),
        "cold standby with 3 different units operating",
    ),
    (
        lambda: LoadSharingModel(
            [aft("weibull"), aft("exponential")], load=2.0
        ),
        "units that are different",
    ),
]
IDS = ["warm, two operating", "cold, three different", "load sharing"]


@pytest.mark.parametrize("build, case", NO_RELIABILITY, ids=IDS)
def test_a_model_with_no_reliability_refuses_it(build, case):
    model = quiet(build)
    assert model.is_simulated
    for call in (
        lambda: model.sf(10.0),
        lambda: model.ff(10.0),
        lambda: model.cs(5.0, 10.0),
    ):
        with pytest.raises(NotImplementedError, match=case):
            call()
    with pytest.raises(
        NotImplementedError, match=r"mean\(method='simulate', mc_samples="
    ):
        model.mean()
    # It still draws lifetimes, and estimates its mean from new ones.
    draws = model.random(2000, seed=1)
    assert np.all(draws > 0)
    estimate = model.mean(mc_samples=2000, seed=1)
    assert estimate == pytest.approx(float(np.mean(draws)), rel=1e-12)


@pytest.mark.parametrize("build, case", NO_RELIABILITY, ids=IDS)
def test_a_diagram_with_one_is_simulated(build, case):
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": build(), "b": W([90, 3])}
    )
    report = rbd.analysis_routes()
    assert rbd.get_non_analytic_nodes() == {
        "a": type(rbd.reliabilities["a"]).__name__
    }
    for name in ("sf", "ff", "mean", "birnbaum_importance", "node_mttf"):
        assert report[name].route == routes.REFUSED
        assert case in report[name].reason
    for name in ("random", "unreliability_interval", "compare"):
        assert report[name].route == routes.SIMULATED
    with pytest.raises(NotImplementedError, match=case):
        rbd.sf(10.0)
    assert rbd.mean(method="simulate", mc_samples=2000, seed=1) > 0
    # Held working, the node needs no reliability.
    assert rbd.sf(10.0, working_nodes=["a"]) == pytest.approx(
        float(W([90, 3]).sf(10.0))
    )


@pytest.mark.parametrize("build, case", NO_RELIABILITY[:2], ids=IDS[:2])
def test_a_repairable_component_with_one_is_simulated(build, case):
    rbd = RepairableRBD(
        ONE, {"a": {"reliability": build(), "repairability": E([0.5])}}
    )
    report = rbd.analysis_routes()
    for name in ("mean_availability", "point_availability", "spares_demand"):
        assert report[name].route == routes.REFUSED
        assert report[name].nodes == ("a",)
    with pytest.raises(
        NotImplementedError, match=r"mean\(method='simulate', mc_samples="
    ):
        rbd.mean_availability()
    with pytest.raises(NotImplementedError, match=case):
        rbd.point_availability([10.0])
    result = rbd.availability(200.0, mc_samples=50, seed=1)
    assert 0.0 < float(np.mean(result.availability)) <= 1.0


def test_exact_and_numerical_models_are_unchanged():
    for build in (
        lambda: StandbyModel([W([100, 2])] * 2),
        lambda: StandbyModel([E([0.01])] * 2, dormancy_factor=0.5),
        lambda: StandbyModel([E([0.01])] * 3, k=2),
        lambda: StandbyModel([W([100, 2])] * 2, dormancy_factor=0.5),
        lambda: StandbyModel([W([100, 2]), W([80, 1.5])], dormancy_factor=1),
        lambda: StandbyModel([W([100, 2])] * 3, k=2),
        lambda: StandbyModel([W([100, 2]), W([80, 1.5]), E([0.01])], k=2),
        lambda: LoadSharingModel([aft("exponential")] * 2, load=2.0),
        lambda: LoadSharingModel([aft("weibull")] * 2, load=2.0),
    ):
        model = quiet(build)
        assert not model.is_simulated
        assert 0.0 < float(model.sf(50.0)) < 1.0
        assert np.isfinite(model.mean())


def test_scoring_cold_standby_candidates_still_works():
    # allocate_redundancy scores cold standby that needs two copies working
    # with a StandbyModel of its own: identical copies, so numerical.
    rbd = NonRepairableRBD(ONE, {"a": W([100, 2])})
    result = quiet(
        lambda: rbd.allocate_redundancy(
            {"a": 1.0}, budget=3, t=50.0, required=2, strategy="cold"
        )
    )
    assert result.units == {"a": 3}
