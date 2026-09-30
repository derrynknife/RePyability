"""The report of how each analysis is computed (``analysis_routes``).

Every method must do what the report says, on diagrams of every kind:
refused, it raises the message the report gives; exact or numerical, it
gives the same result every time; simulated, it gives the same result for
the same seed. And the report covers every public analysis.
"""

import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    AnalysisRoute,
    BetaFactor,
    CCFGroup,
    DegradingNode,
    LoadSharingModel,
    NonRepairableRBD,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
)
from repyability.rbd import routes
from repyability.tests.test_performance_equivalence import binomial_first
from repyability.tests.test_simulation_engines import (
    identical,
    systems_of_every_kind,
)

W = surv.Weibull.from_params
E = surv.Exponential.from_params
L = surv.LogNormal.from_params
FIXED = surv.FixedEventProbability.from_params
EDGES = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]


def nonrepairable_rbds():
    unit = W([100, 2])
    rest = {"b": W([80, 1.5]), "c": E([0.002])}
    return {
        "plain": NonRepairableRBD(EDGES, {"a": unit, **rest}),
        "convolved standby": NonRepairableRBD(
            EDGES, {"a": StandbyModel([unit, unit]), **rest}
        ),
        "simulated standby": NonRepairableRBD(
            EDGES,
            {"a": StandbyModel([unit] * 3, k=2, n_sims=2000, seed=1), **rest},
        ),
        "repeated standby": NonRepairableRBD(
            EDGES, {"a": RepeatedStandbyNode(unit, 2), **rest}
        ),
        "repeated": NonRepairableRBD(
            EDGES, {"a": RepeatedNode(unit, 2, "parallel"), **rest}
        ),
        "common cause": NonRepairableRBD(
            EDGES,
            {"a": unit, "b": unit, "c": E([0.002])},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        ),
        "fixed": NonRepairableRBD(
            EDGES, {"a": FIXED(0.1), "b": FIXED(0.2), "c": FIXED(0.05)}
        ),
        "capacities": NonRepairableRBD(
            EDGES,
            {"a": unit, **rest},
            capacity={"a": 5.0, "b": 5.0, "c": 10.0},
        ),
        "unreplayable": NonRepairableRBD(
            EDGES, {"a": binomial_first([100, 2]), **rest}
        ),
        "nested": NonRepairableRBD(
            EDGES,
            {
                "a": NonRepairableRBD(
                    [("s", "x"), ("x", "t")],
                    {"x": StandbyModel([unit] * 3, k=2, n_sims=500, seed=2)},
                ),
                **rest,
            },
        ),
    }


def repairable_rbds():
    life, repair = W([500, 1.5]), E([0.5])

    def unit(**more):
        return {"reliability": life, "repairability": repair, **more}

    def system(a, **options):
        return RepairableRBD(
            EDGES, {"a": a, "b": unit(), "c": unit()}, **options
        )

    out = dict(systems_of_every_kind())
    out.update(
        {
            "age replacement, priced": system(
                unit(preventive={"interval": 300.0}, replace_cost=10.0),
            ),
            "block replacement": system(
                unit(preventive={"interval": 300.0, "policy": "block"})
            ),
            "tested, constant rate": system(
                {
                    "reliability": E([0.002]),
                    "repairability": "instant",
                    "inspection": {"interval": 100.0},
                }
            ),
            "tested, Weibull": system(
                unit(repairability="instant", inspection={"interval": 100.0})
            ),
            "simulated standby life": system(
                {
                    "reliability": StandbyModel(
                        [life] * 3, k=2, n_sims=2000, seed=1
                    ),
                    "repairability": repair,
                }
            ),
            "non-parametric life": system(
                {
                    "reliability": surv.KaplanMeier.fit(
                        [100.0, 250.0, 400.0, 700.0, 900.0]
                    ),
                    "repairability": repair,
                }
            ),
            "fixed probability": system(
                {"reliability": FIXED(0.1), "repairability": repair}
            ),
            "degrading capacity": system(
                {
                    "reliability": DegradingNode(
                        [(100.0, E([0.004])), (50.0, E([0.004]))]
                    ),
                    "repairability": repair,
                },
                capacity={"b": 100.0, "c": 100.0},
            ),
        }
    )
    return out


X = 30.0
NONREPAIRABLE_CALLS = {
    **{
        name: (lambda name: lambda rbd: getattr(rbd, name)(X))(name)
        for name in (
            "sf",
            "ff",
            "reliability",
            "unreliability",
            "Hf",
            "df",
            "hf",
            "node_sf",
            "node_ff",
            "birnbaum_importance",
            "improvement_potential",
            "risk_achievement_worth",
            "risk_reduction_worth",
            "criticality_importance",
            "fussell_vesely",
            "parameter_sensitivity",
            "capacity_distribution",
        )
    },
    "cs": lambda rbd: rbd.cs(X, 10.0),
    "time_to_reliability": lambda rbd: rbd.time_to_reliability(0.5),
    "bx_life": lambda rbd: rbd.bx_life(10),
    "remaining_life": lambda rbd: rbd.remaining_life(0.5, state={}),
    "sf_given_state": lambda rbd: rbd.sf_given_state(X, state={}),
    "importances_given_state": lambda rbd: rbd.importances_given_state(
        X, state={}
    ),
    "structural_importance": lambda rbd: rbd.structural_importance(),
    "random": lambda rbd: rbd.random(200, seed=1),
    "mean": lambda rbd: rbd.mean(2000, seed=1),
    "mean_time_to_failure": lambda rbd: rbd.mean_time_to_failure(2000, seed=1),
    "mean_time_to_failure_interval": (
        lambda rbd: rbd.mean_time_to_failure_interval(2000, seed=1)
    ),
    "compare": lambda rbd: rbd.compare(rbd, 2000, seed=1),
    "node_mttf": lambda rbd: rbd.node_mttf(2000, seed=1),
    "allocate_redundancy": lambda rbd: rbd.allocate_redundancy(
        {"b": 1.0}, budget=2, t=X
    ),
    "system_probability": lambda rbd: rbd.system_probability(
        {node: 0.9 for node in rbd.reliabilities}
    ),
}

REPAIRABLE_CALLS = {
    **{
        name: (lambda name: lambda rbd: getattr(rbd, name)())(name)
        for name in (
            "mean_availability",
            "mean_unavailability",
            "node_availability",
            "system_failure_frequency",
            "mean_time_between_failures",
            "mean_up_time",
            "mean_down_time",
            "birnbaum_importance",
            "improvement_potential",
            "risk_achievement_worth",
            "risk_reduction_worth",
            "criticality_importance",
            "fussell_vesely",
            "expected_cost_rate",
            "capacity_distribution",
            "optimal_replacement_intervals",
            "optimal_inspection_intervals",
            "structural_importance",
        )
    },
    "total_cost": lambda rbd: rbd.total_cost(1000.0),
    "point_availability": lambda rbd: rbd.point_availability([10.0, 200.0]),
    "mission_availability": lambda rbd: rbd.mission_availability(200.0),
    "availability": lambda rbd: rbd.availability(200.0, N=20, seed=1),
    "cost": lambda rbd: rbd.cost(200.0, N=20, seed=1),
    "compare": lambda rbd: rbd.compare(rbd, 200.0, N=20, seed=1),
}


def behaves_as_reported(route: AnalysisRoute, call) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if route.route == routes.REFUSED:
            with pytest.raises((NotImplementedError, ValueError)) as error:
                call()
            assert str(error.value) == route.reason
        else:
            identical(call(), call())


@pytest.mark.parametrize("name", sorted(nonrepairable_rbds()))
def test_every_nonrepairable_method_does_what_the_report_says(name):
    rbd = nonrepairable_rbds()[name]
    report = rbd.analysis_routes()
    for method, call in NONREPAIRABLE_CALLS.items():
        behaves_as_reported(report[method], lambda: call(rbd))


@pytest.mark.parametrize("name", sorted(repairable_rbds()))
def test_every_repairable_method_does_what_the_report_says(name):
    rbd = repairable_rbds()[name]
    report = rbd.analysis_routes()
    for method, call in REPAIRABLE_CALLS.items():
        behaves_as_reported(report[method], lambda: call(rbd))


@pytest.mark.parametrize("cls", [NonRepairableRBD, RepairableRBD])
def test_the_report_covers_every_public_analysis(cls):
    # Everything public but building, saving and the structure itself.
    not_analyses = {
        "analysis_routes",
        "from_dict",
        "from_json",
        "to_dict",
        "to_json",
        "find_irrelevant_components",
        "get_all_path_sets",
        "get_min_cut_sets",
        "get_min_path_sets",
        "get_non_analytic_nodes",
        "is_analytically_solvable",
        "is_system_working",
        "node_names",
    }
    public = {
        name
        for name, _ in inspect.getmembers(cls, callable)
        if not name.startswith("_")
    }
    rbd = next(
        iter(
            (
                nonrepairable_rbds()
                if cls is NonRepairableRBD
                else repairable_rbds()
            ).values()
        )
    )
    assert set(rbd.analysis_routes()) == public - not_analyses


def test_nodes_are_routed_by_how_their_reliability_is_found():
    unit = W([100, 2])
    cases = [
        (unit, routes.EXACT),
        (surv.KaplanMeier.fit([1.0, 2.0, 3.0]), routes.EXACT),
        (StandbyModel([E([0.01])] * 3, k=2), routes.EXACT),
        (StandbyModel([unit, unit]), routes.NUMERICAL),
        (StandbyModel([unit] * 3, k=2, n_sims=500, seed=1), routes.SIMULATED),
        (RepeatedStandbyNode(unit, 3), routes.NUMERICAL),
        (RepeatedNode(unit, 3, "series"), routes.EXACT),
        (DegradingNode([(1.0, unit), (0.5, unit)]), routes.NUMERICAL),
    ]
    for model, expected in cases:
        assert routes.model_route(model)[0] == expected, model


def test_a_simulated_node_is_named_and_the_rest_stay_exact():
    rbd = nonrepairable_rbds()["simulated standby"]
    report = rbd.analysis_routes()
    assert report["sf"].route == routes.SIMULATED
    assert report["sf"].nodes == ("a",)
    assert "2000 simulated lifetimes" in report["sf"].reason
    assert report["structural_importance"].route == routes.EXACT
    assert rbd.get_non_analytic_nodes() == {"a": "StandbyModel"}
    exact = nonrepairable_rbds()["convolved standby"]
    assert exact.analysis_routes()["sf"].route == routes.NUMERICAL
    assert exact.is_analytically_solvable()


def test_common_cause_groups_refuse_what_does_not_model_them():
    report = nonrepairable_rbds()["common cause"].analysis_routes()
    assert report["sf"].route == routes.EXACT
    assert report["birnbaum_importance"].route == routes.REFUSED
    assert report["allocate_redundancy"].route == routes.REFUSED
    assert "sampled independently" in report["mean"].reason


def test_the_repairable_report_names_the_refusing_component():
    report = repairable_rbds()["tested, Weibull"].analysis_routes()
    assert report["mean_availability"].route == routes.REFUSED
    assert report["mean_availability"].nodes == ("a",)
    assert report["availability"].route == routes.SIMULATED
    assert report["availability"].engine == "python"
    assert "inspections" in report["availability"].engine_reason


def test_maintenance_makes_the_long_run_numerical():
    for name in ("age replacement, priced", "block replacement"):
        report = repairable_rbds()[name].analysis_routes()
        assert report["mean_availability"].route == routes.NUMERICAL
        assert report["mean_availability"].nodes == ("a",)
    report = repairable_rbds()["tested, constant rate"].analysis_routes()
    assert report["mean_availability"].route == routes.EXACT


def test_a_simulated_life_makes_the_exact_values_simulated():
    report = repairable_rbds()["simulated standby life"].analysis_routes()
    assert report["mean_availability"].route == routes.SIMULATED
    assert report["mean_availability"].nodes == ("a",)


def test_the_engine_is_the_one_auto_would_run(monkeypatch):
    from repyability.rbd import _compiled

    plain = repairable_rbds()["koon"]
    monkeypatch.setattr(_compiled, "available", lambda: False)
    route = plain.analysis_routes()["availability"]
    assert route.engine == "python"
    assert "not installed" in route.engine_reason
    monkeypatch.setattr(_compiled, "available", lambda: True)
    assert plain.analysis_routes()["availability"].engine == "numba"
    capacities = repairable_rbds()["capacities"].analysis_routes()
    assert capacities["availability"].engine == "python"
    assert capacities["cost"].engine == "numba"


def test_a_load_sharing_group_is_simulated_only_without_a_closed_form():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=400)
    x = rng.exponential(scale=100.0 / np.exp(0.6 * (load - 1.0)), size=400)
    exponential = surv.ExponentialAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))
    weibull = surv.WeibullAFT.fit(
        rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1)) + 1e-3,
        Z=load.reshape(-1, 1),
    )
    closed = LoadSharingModel([exponential] * 2, load=2.0)
    fitted = LoadSharingModel([weibull] * 2, load=2.0, n_sims=500, seed=3)
    assert routes.model_route(closed)[0] == routes.EXACT
    assert routes.model_route(fitted)[0] == routes.SIMULATED


def test_a_route_reads_as_a_sentence():
    route = AnalysisRoute(
        routes.SIMULATED, "A simulation.", engine="python", engine_reason="x"
    )
    assert str(route) == "simulated: A simulation. Engine: python (x)."


def test_the_guide_s_table_agrees_with_the_report():
    # The saving guide's table gives each method's route on a diagram of
    # plain components: the report must give the same, for each class
    # the method is on.
    guide = Path(__file__).resolve().parents[2] / "docs/guide/saving.md"
    section = guide.read_text().split("## What is exact and what is simulated")
    table = section[1].split("\n## ")[0]
    life, repair = W([500, 1.5]), E([0.5])
    capacity = {"a": 5.0, "b": 5.0, "c": 10.0}
    plain = {
        NonRepairableRBD: nonrepairable_rbds()["capacities"],
        RepairableRBD: RepairableRBD(
            EDGES,
            {n: {"reliability": life, "repairability": repair} for n in "abc"},
            capacity=capacity,
        ),
    }
    reports = {cls: rbd.analysis_routes() for cls, rbd in plain.items()}
    checked = 0
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[1] not in routes._RANK:
            continue
        for name in re.findall(r"`(\w+)`", cells[0]):
            for cls, report in reports.items():
                if hasattr(cls, name):
                    assert report[name].route == cells[1], (cls, name)
                    checked += 1
    assert checked >= 30
