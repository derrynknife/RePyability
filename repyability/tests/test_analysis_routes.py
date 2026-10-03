"""The report of how each analysis is computed (``analysis_routes``).

Every method must do what the report says, on diagrams of every kind:
refused, it raises the message the report gives; exact or numerical, it
gives the same result every time; simulated, it gives the same result for
the same seed. And the report covers every public analysis.
"""

import inspect
import re
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    MGL,
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
    network,
)
from repyability.rbd import bdd, modular, phased_mission, routes
from repyability.tests.repository import source
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
BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "c"),
    ("a", "d"),
    ("c", "d"),
    ("b", "e"),
    ("c", "e"),
    ("d", "t"),
    ("e", "t"),
]


def too_meshed(build):
    """``build()``, its core given up on as too meshed to work out (see
    ``modular.GraphStructure``), as a far larger one would be (#172)."""
    limit, method = bdd.STEP_LIMIT, modular.CORE_METHOD
    bdd.STEP_LIMIT, modular.CORE_METHOD = 2, "bdd"
    try:
        rbd = build()
    finally:
        bdd.STEP_LIMIT, modular.CORE_METHOD = limit, method
    assert rbd.structure_check["is_too_meshed"]
    return rbd


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
            {
                "a": StandbyModel([unit] * 3, k=2, dormancy_factor=0.5),
                **rest,
            },
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
        "common cause by rate": NonRepairableRBD(
            EDGES,
            {"a": unit, "b": unit, "c": E([0.002])},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
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
                    {"x": StandbyModel([unit] * 3, k=2, dormancy_factor=0.5)},
                ),
                **rest,
            },
        ),
        "too meshed": too_meshed(
            lambda: NonRepairableRBD(
                BRIDGE,
                {"a": unit, "b": unit, "c": E([0.002]), "d": unit, "e": unit},
            )
        ),
        "too meshed, with capacities": too_meshed(
            lambda: NonRepairableRBD(
                BRIDGE,
                {"a": unit, "b": unit, "c": E([0.002]), "d": unit, "e": unit},
                capacity={"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0, "e": 5.0},
            )
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
            "block replacement, in no time": system(
                unit(
                    repairability="instant",
                    preventive={"interval": 300.0, "policy": "block"},
                )
            ),
            "replaced on condition": system(
                unit(
                    preventive={
                        "interval": 100.0,
                        "policy": "condition",
                        "threshold": 0.1,
                        "inspection_cost": 1.0,
                    },
                    replace_cost=10.0,
                )
            ),
            "opportunistic maintenance": RepairableRBD(
                EDGES,
                {
                    node: unit(
                        preventive={"interval": 300.0, "opportunity": 200.0},
                        group="train",
                        replace_cost=10.0,
                    )
                    for node in "abc"
                },
                maintenance_groups={
                    "train": {"setup_cost": 50.0, "system_down": True}
                },
            ),
            "grouped, no opportunities": RepairableRBD(
                EDGES,
                {
                    "a": unit(preventive={"interval": 300.0}, group="train"),
                    "b": unit(group="train"),
                    "c": unit(),
                },
                maintenance_groups={"train": {"setup_cost": 50.0}},
            ),
            "grouped block replacements": RepairableRBD(
                EDGES,
                {
                    node: unit(
                        preventive={"interval": 300.0, "policy": "block"},
                        group="train",
                    )
                    for node in "ab"
                }
                | {"c": unit()},
                maintenance_groups={"train": {"setup_cost": 50.0}},
            ),
            "imperfect repair": system(
                unit(
                    repair={"model": "kijima1", "q": 0.5},
                    repair_cost=1.0,
                    replace_cost=10.0,
                )
            ),
            "minimal repair in no time": system(
                {
                    "reliability": life,
                    "repairability": "instant",
                    "repair": {"model": "kijima1", "q": 1.0},
                    "repair_cost": 1.0,
                    "replace_cost": 10.0,
                }
            ),
            "imperfect repair, replaced and maintained": system(
                unit(
                    repair={"model": "kijima2", "q": 0.8},
                    replace_after=3,
                    preventive={"interval": 300.0},
                    replace_cost=10.0,
                )
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
            "tested, taking time": system(
                unit(
                    repairability="instant",
                    inspection={"interval": 100.0, "duration": E([2.0])},
                )
            ),
            "simulated standby life": system(
                {
                    "reliability": StandbyModel(
                        [life] * 3, k=2, dormancy_factor=0.5
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
            "one repair crew": system(
                unit(priority=1), repair_crews=1, downtime_cost_rate=5.0
            ),
            "enough repair crews": system(unit(), repair_crews=3),
            "too meshed": too_meshed(
                lambda: RepairableRBD(
                    BRIDGE, {n: unit(repair_cost=2.0) for n in "abcde"}
                )
            ),
            "too meshed, downtime priced": too_meshed(
                lambda: RepairableRBD(
                    BRIDGE,
                    {n: unit(repair_cost=2.0) for n in "abcde"},
                    downtime_cost_rate=5.0,
                    capacity={n: 5.0 for n in "abcde"},
                )
            ),
            "standby group": system(
                {
                    "reliability": E([0.002]),
                    "repairability": E([0.5]),
                    "standby": {"units": 3, "switching_probability": 0.95},
                    "repair_cost": 3.0,
                },
                downtime_cost_rate=5.0,
            ),
            "standby group, Weibull": system(
                unit(standby={"dormancy_factor": 0.5}, repair_cost=3.0),
                downtime_cost_rate=5.0,
            ),
            "common cause, tested": RepairableRBD(
                EDGES,
                {
                    "a": {
                        "reliability": E([0.002]),
                        "repairability": "instant",
                        "inspection": {
                            "interval": 100.0,
                            "coverage": 0.8,
                            "full_test": 300.0,
                        },
                    },
                    "b": {
                        "reliability": E([0.002]),
                        "repairability": "instant",
                        "inspection": {
                            "interval": 100.0,
                            "offset": 50.0,
                            "coverage": 0.8,
                            "full_test": 300.0,
                        },
                    },
                    "c": unit(repair_cost=2.0),
                },
                ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
                downtime_cost_rate=5.0,
            ),
            "common cause, revealed": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    }
                    for node in "ab"
                }
                | {
                    "c": unit(
                        preventive={"interval": 300.0, "policy": "block"}
                    )
                },
                ccf_groups=[CCFGroup(["a", "b"], MGL(0.2))],
            ),
            "common cause, Weibull": system(
                unit(), ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))]
            ),
            "common cause, timed block replacement": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                    }
                    for node in "ab"
                }
                | {
                    "c": unit(
                        preventive={
                            "interval": 300.0,
                            "policy": "block",
                            "duration": E([2.0]),
                        }
                    )
                },
                ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
            ),
            "one repair crew, exponential": RepairableRBD(
                EDGES,
                {
                    node: {
                        "reliability": E([0.002]),
                        "repairability": E([0.5]),
                        "priority": priority,
                        "repair_cost": 3.0,
                    }
                    for node, priority in zip("abc", (1, 0, 0))
                },
                repair_crews=1,
                downtime_cost_rate=5.0,
                capacity={"a": 5.0, "b": 5.0, "c": 10.0},
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
    "mean_residual_life": lambda rbd: rbd.mean_residual_life(state={}),
    "sf_given_state": lambda rbd: rbd.sf_given_state(X, state={}),
    "importances_given_state": lambda rbd: rbd.importances_given_state(
        X, state={}
    ),
    "structural_importance": lambda rbd: rbd.structural_importance(),
    "random": lambda rbd: rbd.random(200, seed=1),
    "random_block": lambda rbd: rbd.random_block(1, seed=1),
    "unreliability_interval": lambda rbd: rbd.unreliability_interval(
        X, relative_tolerance=0.5, max_samples=40_000, seed=1
    ),
    "mean": lambda rbd: rbd.mean(),
    "mean_time_to_failure": lambda rbd: rbd.mean_time_to_failure(),
    "mean_time_to_failure_interval": (
        lambda rbd: rbd.mean_time_to_failure_interval(2000, seed=1)
    ),
    "compare": lambda rbd: rbd.compare(rbd, 2000, seed=1),
    "node_mttf": lambda rbd: rbd.node_mttf(),
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
    "spares_demand": lambda rbd: rbd.spares_demand(200.0),
    "spares_stock": lambda rbd: rbd.spares_stock(50.0, fill_rate=0.9),
    "point_availability": lambda rbd: rbd.point_availability([10.0, 200.0]),
    "mission_availability": lambda rbd: rbd.mission_availability(200.0),
    "expected_failures": lambda rbd: rbd.expected_failures([10.0, 200.0]),
    "expected_events": lambda rbd: rbd.expected_events(200.0),
    "expected_cost": lambda rbd: rbd.expected_cost(200.0),
    "point_capacity": lambda rbd: rbd.point_capacity([10.0, 200.0]),
    "mission_capacity": lambda rbd: rbd.mission_capacity(200.0),
    "availability": lambda rbd: rbd.availability(200.0, mc_samples=20, seed=1),
    "simulate_chunk": lambda rbd: rbd.simulate_chunk(200.0, 10, 30, seed=1),
    "shards": lambda rbd: rbd.shards(200.0, 40, seed=1, size=8),
    "availability_from_chunks": lambda rbd: rbd.availability_from_chunks(
        rbd.simulate_chunk(200.0, 10, 30, seed=1), allow_gaps=True
    ),
    "simulate_timelines": lambda rbd: rbd.simulate_timelines(
        200.0, mc_samples=20, seed=1
    ),
    "cost": lambda rbd: rbd.cost(200.0, mc_samples=20, seed=1),
    "compare": lambda rbd: rbd.compare(rbd, 200.0, mc_samples=20, seed=1),
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
        "minimal_cut_sets",
        "minimal_path_sets",
        "get_non_analytic_nodes",
        "is_analytically_solvable",
        "is_system_working",
        "node_names",
        "system_timeline",
        "with_intervals",
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
        (StandbyModel([E([0.01])] * 3, k=2), routes.EXACT),
        (StandbyModel([unit, unit]), routes.NUMERICAL),
        (StandbyModel([unit] * 3, k=2), routes.NUMERICAL),
        (StandbyModel([unit] * 2, dormancy_factor=0.5), routes.NUMERICAL),
        (StandbyModel([unit] * 3, k=2, dormancy_factor=1.0), routes.EXACT),
        (StandbyModel([unit] * 3, k=2, dormancy_factor=0.5), routes.REFUSED),
        (RepeatedStandbyNode(unit, 3), routes.NUMERICAL),
        (RepeatedNode(unit, 3, "series"), routes.EXACT),
        (DegradingNode([(1.0, unit), (0.5, unit)]), routes.NUMERICAL),
    ]
    for model, expected in cases:
        assert routes.model_route(model)[0] == expected, model


def test_a_node_with_no_reliability_is_named_and_the_rest_stay_exact():
    rbd = nonrepairable_rbds()["simulated standby"]
    report = rbd.analysis_routes()
    assert report["sf"].route == routes.REFUSED
    assert report["sf"].nodes == ("a",)
    assert "no exact or numerical reliability" in report["sf"].reason
    assert report["random"].route == routes.SIMULATED
    assert report["unreliability_interval"].route == routes.SIMULATED
    assert report["structural_importance"].route == routes.EXACT
    assert rbd.get_non_analytic_nodes() == {"a": "StandbyModel"}
    exact = nonrepairable_rbds()["convolved standby"]
    assert exact.analysis_routes()["sf"].route == routes.NUMERICAL
    assert exact.is_analytically_solvable()


def test_common_cause_groups_refuse_what_does_not_model_them():
    report = nonrepairable_rbds()["common cause"].analysis_routes()
    assert report["sf"].route == routes.EXACT
    assert report["birnbaum_importance"].route == routes.EXACT
    assert report["allocate_redundancy"].route == routes.EXACT
    assert report["parameter_sensitivity"].route == routes.NUMERICAL
    assert report["sf_given_state"].route == routes.REFUSED
    assert report["mean"].route == routes.REFUSED
    assert (
        "sampled independently"
        in report["mean_time_to_failure_interval"].reason
    )


def test_the_repairable_report_names_the_refusing_component(monkeypatch):
    from repyability.rbd import _compiled

    monkeypatch.setattr(_compiled, "available", lambda: True)
    report = repairable_rbds()["tested, taking time"].analysis_routes()
    assert report["mean_availability"].route == routes.REFUSED
    assert report["mean_availability"].nodes == ("a",)
    assert report["availability"].route == routes.SIMULATED
    # numba's own loop simulates tests (#155), not imperfect repair.
    assert report["availability"].engine == "numba"
    imperfect = repairable_rbds()["imperfect repair"].analysis_routes()
    assert imperfect["availability"].engine == "python"
    assert "imperfect repair" in imperfect["availability"].engine_reason


def test_maintenance_makes_the_long_run_numerical():
    for name in (
        "age replacement, priced",
        "block replacement",
        "tested, Weibull",
    ):
        report = repairable_rbds()[name].analysis_routes()
        assert report["mean_availability"].route == routes.NUMERICAL
        assert report["mean_availability"].nodes == ("a",)
    report = repairable_rbds()["tested, constant rate"].analysis_routes()
    assert report["mean_availability"].route == routes.EXACT


def test_a_life_with_no_mean_refuses_the_long_run():
    report = repairable_rbds()["simulated standby life"].analysis_routes()
    assert report["mean_availability"].route == routes.REFUSED
    assert report["mean_availability"].nodes == ("a",)
    assert (
        "mean(mc_samples=..., seed=...)" in report["mean_availability"].reason
    )
    assert report["availability"].route == routes.SIMULATED


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
    # numba's own loop follows capacities (#155).
    assert capacities["availability"].engine == "numba"
    assert capacities["cost"].engine == "numba"


def test_a_load_sharing_group_is_simulated_only_with_different_units():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=400)
    x = rng.exponential(scale=100.0 / np.exp(0.6 * (load - 1.0)), size=400)
    exponential = surv.ExponentialAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))
    weibull = surv.WeibullAFT.fit(
        rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1)) + 1e-3,
        Z=load.reshape(-1, 1),
    )
    closed = LoadSharingModel([exponential] * 2, load=2.0)
    identical = LoadSharingModel([weibull] * 2, load=2.0)
    different = LoadSharingModel([weibull, exponential], load=2.0)
    assert routes.model_route(closed)[0] == routes.EXACT
    assert routes.model_route(identical)[0] == routes.NUMERICAL
    assert routes.model_route(different)[0] == routes.REFUSED


def test_a_route_reads_as_a_sentence():
    route = AnalysisRoute(
        routes.SIMULATED, "A simulation.", engine="python", engine_reason="x"
    )
    assert str(route) == "simulated: A simulation. Engine: python (x)."


def test_the_guide_s_table_agrees_with_the_report():
    # The saving guide's table gives each method's route on a diagram of
    # plain components: the report must give the same, for each class
    # the method is on.
    guide = source("docs/guide/saving.md")
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


def test_the_readme_says_what_is_simulated():
    # The README's table of what is simulated: each situation it lists is
    # routed as it says, on a diagram of that kind. When a route changes
    # (say, warm standby made exact), the README must change with it.
    readme = source("README.md").read_text()
    nonrepairable, repairable = nonrepairable_rbds(), repairable_rbds()

    def alone(node):
        return NonRepairableRBD([("s", "a"), ("a", "t")], {"a": node})

    unit = W([100, 2])
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=400)
    weibull = surv.WeibullAFT.fit(
        rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1)) + 1e-3,
        Z=load.reshape(-1, 1),
    )
    x = rng.exponential(scale=100.0 / np.exp(0.6 * (load - 1.0)), size=400)
    exponential = surv.ExponentialAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))
    sharing = LoadSharingModel([weibull, exponential], load=2.0)
    plain, ccf = nonrepairable["plain"], nonrepairable["common cause"]
    crew = repairable["one repair crew, exponential"]
    group = repairable["standby group"]
    minimal = repairable["minimal repair in no time"]
    claims = {
        "Sampled lifetimes or histories, and distributions or percentiles "
        "of an outcome over a window": [
            (plain, "random", "simulated"),
            (repairable["costed_pairs"], "availability", "simulated"),
            (repairable["costed_pairs"], "simulate_timelines", "simulated"),
            (repairable["maintained"], "simulate_timelines", "simulated"),
            (repairable["costed_pairs"], "expected_events", "numerical"),
            (repairable["costed_pairs"], "expected_cost", "numerical"),
            (repairable["capacities"], "mission_capacity", "numerical"),
        ],
        "Comparing two designs (`compare`)": [
            (plain, "compare", "simulated"),
            (repairable["costed_pairs"], "compare", "simulated"),
        ],
        "The uncertainty from fitted component parameters "
        "(`sf_uncertainty`, `mean_uncertainty`, `bx_life_uncertainty`, "
        "`time_to_reliability_uncertainty`)": [
            (plain, name, "simulated")
            for name in (
                "sf_uncertainty",
                "mean_uncertainty",
                "bx_life_uncertainty",
                "time_to_reliability_uncertainty",
            )
        ],
        "Small failure probabilities, with a node only simulations take": [
            (nonrepairable["simulated standby"], "ff", "refused"),
            (
                nonrepairable["simulated standby"],
                "unreliability_interval",
                "simulated",
            ),
            (plain, "ff", "exact"),
        ],
        "Warm standby with two or more units operating, of "
        "non-exponential units": [
            (nonrepairable["simulated standby"], "sf", "refused"),
            (nonrepairable["simulated standby"], "random", "simulated"),
            (
                alone(StandbyModel([unit] * 2, dormancy_factor=0.5)),
                "sf",
                "numerical",
            ),
            (
                alone(StandbyModel([unit] * 2, dormancy_factor=1.0)),
                "sf",
                "exact",
            ),
        ],
        "Cold standby with three or more different units operating, and "
        "load sharing of different units": [
            (
                alone(
                    StandbyModel([unit, W([80, 1.5]), unit, W([90, 3])], k=3)
                ),
                "sf",
                "refused",
            ),
            (alone(StandbyModel([unit] * 3, k=2)), "sf", "numerical"),
            (
                alone(StandbyModel([unit, W([80, 1.5]), unit], k=2)),
                "sf",
                "numerical",
            ),
            (alone(sharing), "sf", "refused"),
            (alone(sharing), "random", "simulated"),
        ],
        "Anything such a node is part of": [
            (nonrepairable["nested"], "sf", "refused"),
            (nonrepairable["nested"], "random", "simulated"),
        ],
        "Common-cause groups: analyses given ages, and the MTTF of a group "
        "splitting a failure probability": [
            (ccf, "sf", "exact"),
            (ccf, "birnbaum_importance", "exact"),
            (ccf, "fussell_vesely", "exact"),
            (ccf, "allocate_redundancy", "exact"),
            (ccf, "parameter_sensitivity", "numerical"),
            (ccf, "sf_uncertainty", "simulated"),
            (nonrepairable["common cause by rate"], "mean", "numerical"),
            (
                nonrepairable["common cause by rate"],
                "mean_uncertainty",
                "simulated",
            ),
            (nonrepairable["common cause by rate"], "random", "simulated"),
        ]
        + [
            (ccf, name, "refused")
            for name in ("mean", "mean_uncertainty", "sf_given_state")
        ],
        "Phased missions and networks too large for their decision "
        "diagrams": [],
        "Block diagrams too meshed for their decision diagrams": [
            (nonrepairable["too meshed"], "random", "simulated"),
            (
                nonrepairable["too meshed"],
                "mean_time_to_failure_interval",
                "simulated",
            ),
            (nonrepairable["too meshed"], "sf", "refused"),
            (repairable["too meshed"], "availability", "simulated"),
            (repairable["too meshed"], "cost", "simulated"),
            (repairable["too meshed"], "simulate_timelines", "simulated"),
            (repairable["too meshed"], "mean_availability", "refused"),
            (repairable["too meshed"], "point_availability", "refused"),
        ],
        "Hidden failures whose tests or repairs take time, or whose tests "
        "miss failures of a life that is not exponential": [
            (
                repairable["tested, taking time"],
                "mean_availability",
                "refused",
            ),
            (
                repairable["tested, taking time"],
                "availability",
                "simulated",
            ),
            (repairable["tested, Weibull"], "mean_availability", "numerical"),
            (
                repairable["tested, Weibull"],
                "point_availability",
                "numerical",
            ),
            (
                repairable["tested, constant rate"],
                "mean_availability",
                "exact",
            ),
        ],
        "Common-cause groups in a repairable diagram": [
            (
                repairable["common cause, tested"],
                "mean_availability",
                "exact",
            ),
            (
                repairable["common cause, tested"],
                "system_failure_frequency",
                "exact",
            ),
            (repairable["common cause, tested"], "availability", "refused"),
            (
                repairable["common cause, tested"],
                "simulate_timelines",
                "refused",
            ),
            (
                repairable["common cause, revealed"],
                "point_availability",
                "refused",
            ),
            (
                repairable["common cause, tested"],
                "birnbaum_importance",
                "exact",
            ),
            (
                repairable["common cause, revealed"],
                "availability_allocation",
                "refused",
            ),
        ],
        "Shared repair crews": [
            (crew, "mean_availability", "exact"),
            (crew, "birnbaum_importance", "exact"),
            (crew, "point_availability", "numerical"),
            (crew, "expected_cost", "numerical"),
            (crew, "point_capacity", "numerical"),
            (crew, "availability_allocation", "refused"),
            (repairable["one repair crew"], "mean_availability", "refused"),
        ],
        "Standby groups (a duty unit and its spares, repaired)": [
            (group, "mean_availability", "exact"),
            (group, "birnbaum_importance", "exact"),
            (group, "point_availability", "numerical"),
            (group, "expected_cost", "numerical"),
            (
                repairable["standby group, Weibull"],
                "mean_availability",
                "refused",
            ),
        ],
        "Opportunistic maintenance (renewals at a group's stops)": [
            (
                repairable["opportunistic maintenance"],
                "mean_availability",
                "refused",
            ),
        ],
        "Imperfect repair (Kijima), with or without replacement at the "
        "*N*-th failure": [
            (repairable["imperfect repair"], "mean_availability", "refused"),
            (repairable["imperfect repair"], "expected_failures", "refused"),
            (
                repairable["imperfect repair, replaced and maintained"],
                "mean_availability",
                "refused",
            ),
            (minimal, "expected_failures", "numerical"),
            (minimal, "point_availability", "numerical"),
            (minimal, "expected_cost", "numerical"),
            (minimal, "mean_availability", "refused"),
        ],
        "Spares of tested components whose tests or repairs take time, and "
        "the stock of block-replaced ones whose repairs or block replacements "
        "take time": [
            (repairable["tested, taking time"], "spares_demand", "refused"),
            (repairable["block replacement"], "spares_stock", "refused"),
            (repairable["block replacement"], "spares_demand", "numerical"),
            (
                repairable["block replacement, in no time"],
                "spares_stock",
                "numerical",
            ),
            (
                repairable["tested, constant rate"],
                "spares_demand",
                "numerical",
            ),
            (repairable["tested, Weibull"], "spares_stock", "numerical"),
        ],
    }
    rows = [
        line.split("|")[1].strip()
        for line in readme.split("## When is a simulation needed?")[1]
        .split("\n## ")[0]
        .splitlines()
        if line.startswith("| ") and not line.startswith("| **")
    ][1:]
    assert sorted(rows) == sorted(claims)
    for situation, checks in claims.items():
        for rbd, analysis, route in checks:
            found = rbd.analysis_routes()[analysis].route
            assert found == route, (situation, analysis, found)
    assert phased_mission.MAX_STATES == 200_000
    assert network.MAX_PATHS == 100_000
    assert network.MAX_STATES == 5_000_000
