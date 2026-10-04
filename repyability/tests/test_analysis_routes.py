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
import scipy.stats as st
import surpyval as surv

from repyability import (
    AnalysisRoute,
    DegradingNode,
    LoadSharingModel,
    NonRepairableRBD,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
    network,
)
from repyability.rbd import _repairable_uncertainty, phased_mission, routes
from repyability.rbd._model_utils import parametric_spec
from repyability.tests.catalogue import (
    EDGES,
    E,
    W,
    nonrepairable_kinds,
    repairable_kinds,
)
from repyability.tests.repository import source
from repyability.tests.test_simulation_engines import identical

X = 30.0


def outside_groups(rbd) -> set:
    """The nodes of no common-cause group."""
    members = {m for g in rbd.ccf_groups for m in g.members}
    return {node for node in rbd.nodes if node not in members}


def uncertain_node(rbd):
    """A node's life scale known to within 5%: the first node, outside
    the common-cause groups, whose model has parameters."""
    free = outside_groups(rbd)
    for node, model in rbd.reliabilities.items():
        spec = parametric_spec(model)
        if spec is not None and node in free:
            name, value = spec[2][0], spec[1][0]
            return {node: {name: st.uniform(0.95 * value, 0.1 * value)}}
    raise AssertionError("No node's model has parameters.")


def copied(rbd):
    """A node to give copies: 'b', unless only a BetaFactor group's member
    could be (an MGL group's refuses, its letters for a larger group being
    unknown)."""
    for group in rbd.ccf_groups:
        if "b" in group.members and type(group.model).__name__ != "BetaFactor":
            return "c"
    return "b"


def probabilities(rbd):
    """Each node's probability of working, for the methods that take
    them."""
    return {node: 0.9 for node in rbd.nodes}


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
            "differential_importance",
            "joint_importance",
            "reliability_rate",
            "barlow_proschan_importance",
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
        {copied(rbd): 1.0}, budget=2, t=X
    ),
    "system_probability": lambda rbd: rbd.system_probability(
        {node: 0.9 for node in rbd.reliabilities}
    ),
    "redundancy_front": lambda rbd: rbd.redundancy_front(
        {copied(rbd): 1.0}, budget=3, t=X
    ),
    "allocate_reliability_redundancy": (
        lambda rbd: rbd.allocate_reliability_redundancy(
            {"c": lambda r, n: n * (1.0 + 10.0 * r)},
            budget=30.0,
            bounds=(0.5, 0.99),
            t=X,
        )
    ),
    "sf_uncertainty": lambda rbd: rbd.sf_uncertainty(
        X, uncertain_node(rbd), n_draws=2, seed=1
    ),
    "mean_uncertainty": lambda rbd: rbd.mean_uncertainty(
        uncertain_node(rbd), n_draws=2, seed=1
    ),
    "time_to_reliability_uncertainty": (
        lambda rbd: rbd.time_to_reliability_uncertainty(
            0.5, uncertain_node(rbd), n_draws=2, seed=1
        )
    ),
    "bx_life_uncertainty": lambda rbd: rbd.bx_life_uncertainty(
        10, uncertain_node(rbd), n_draws=2, seed=1
    ),
    "uncertainty_importance": lambda rbd: rbd.uncertainty_importance(
        X, uncertain_node(rbd)
    ),
}


def uncertain(rbd):
    """A component's life scale known to within 5%: the first node, outside
    the common-cause groups, whose life has parameters."""
    members = {m for g in rbd.ccf_groups for m in g.members}
    for node in rbd.components:
        life = _repairable_uncertainty._models(rbd, node).get("reliability")
        spec = None if life is None else parametric_spec(life)
        if spec is not None and node not in members:
            name, value = spec[2][0], spec[1][0]
            spread = st.uniform(0.95 * value, 0.1 * value)
            return {node: {"reliability": {name: spread}}}
    raise AssertionError("No component's life has parameters.")


def held_back(rbd):
    """A component that may be given copies: of no common-cause group, and
    not a nested diagram, whose costs are its own."""
    free = outside_groups(rbd)
    return next(
        node
        for node, component in rbd.components.items()
        if node in free and not isinstance(component, RepairableRBD)
    )


def a_little_better(rbd):
    """A system availability a little above the current one."""
    try:
        now = rbd.mean_availability()
    except (NotImplementedError, ValueError):
        return 0.5  # the allocation refuses as the long run does
    return now + 0.01 * (1.0 - now)


def stepped(rbd):
    """One simulation stepped through by hand, from numpy's seeded global
    generator."""
    np.random.seed(1)
    rbd.initialize_event_queue(200.0)
    changes = [rbd.next_event()]
    while changes[-1][0] < 200.0:
        changes.append(rbd.next_event())
    return changes


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
            "differential_importance",
            "joint_importance",
            "barlow_proschan_importance",
            "parameter_sensitivity",
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
    "availability_rate": lambda rbd: rbd.availability_rate([10.0, 200.0]),
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
    "mean_availability_uncertainty": (
        lambda rbd: rbd.mean_availability_uncertainty(
            uncertain(rbd), n_draws=2, seed=1
        )
    ),
    "point_availability_uncertainty": (
        lambda rbd: rbd.point_availability_uncertainty(
            [10.0, 200.0], uncertain(rbd), n_draws=2, seed=1
        )
    ),
    "mission_availability_uncertainty": (
        lambda rbd: rbd.mission_availability_uncertainty(
            200.0, uncertain(rbd), n_draws=2, seed=1
        )
    ),
    "expected_cost_rate_uncertainty": (
        lambda rbd: rbd.expected_cost_rate_uncertainty(
            uncertain(rbd), n_draws=2, seed=1
        )
    ),
    "uncertainty_importance": lambda rbd: rbd.uncertainty_importance(
        uncertainty=uncertain(rbd)
    ),
    "system_probability": lambda rbd: rbd.system_probability(
        probabilities(rbd)
    ),
    "allocate_redundancy": lambda rbd: rbd.allocate_redundancy(
        1000.0, nodes=[held_back(rbd)], max_units=2
    ),
    "availability_allocation": lambda rbd: rbd.availability_allocation(
        a_little_better(rbd)
    ),
    "mttf_mttr_allocation": lambda rbd: rbd.mttf_mttr_allocation(
        a_little_better(rbd)
    ),
    "initialize_event_queue": stepped,
    "next_event": stepped,
}
# What the two classes share: the methods on node probabilities.
for calls in (NONREPAIRABLE_CALLS, REPAIRABLE_CALLS):
    calls.update(
        {
            "path_set_probabilities": lambda rbd: rbd.path_set_probabilities(
                probabilities(rbd)
            ),
            "system_capacity": lambda rbd: rbd.system_capacity(
                probabilities(rbd)
            ),
            "equal_allocation": lambda rbd: rbd.equal_allocation(0.95),
            "simple_allocation": lambda rbd: rbd.simple_allocation(0.95),
            **{
                name: (
                    lambda name: lambda rbd: getattr(rbd, name)(
                        0.95, probabilities(rbd)
                    )
                )(name)
                for name in (
                    "improvement_allocation",
                    "minimum_effort_allocation",
                    "cost_based_allocation",
                )
            },
        }
    )


def behaves_as_reported(route: AnalysisRoute, call) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if route.route == routes.REFUSED:
            with pytest.raises((NotImplementedError, ValueError)) as error:
                call()
            assert str(error.value) == route.reason
        else:
            identical(call(), call())


@pytest.mark.parametrize("name", sorted(nonrepairable_kinds()))
def test_every_nonrepairable_method_does_what_the_report_says(name):
    rbd = nonrepairable_kinds()[name]
    report = rbd.analysis_routes()
    for method, call in NONREPAIRABLE_CALLS.items():
        behaves_as_reported(report[method], lambda: call(rbd))


@pytest.mark.parametrize("name", sorted(repairable_kinds()))
def test_every_repairable_method_does_what_the_report_says(name):
    rbd = repairable_kinds()[name]
    report = rbd.analysis_routes()
    for method, call in REPAIRABLE_CALLS.items():
        behaves_as_reported(report[method], lambda: call(rbd))


@pytest.mark.parametrize(
    "kinds, calls",
    [
        (nonrepairable_kinds, NONREPAIRABLE_CALLS),
        (repairable_kinds, REPAIRABLE_CALLS),
    ],
)
def test_every_analysis_is_called(kinds, calls):
    # The checks above call every analysis the report gives a route (#239).
    report = next(iter(kinds().values())).analysis_routes()
    assert set(calls) == set(report)


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
                nonrepairable_kinds()
                if cls is NonRepairableRBD
                else repairable_kinds()
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
    rbd = nonrepairable_kinds()["simulated standby"]
    report = rbd.analysis_routes()
    assert report["sf"].route == routes.REFUSED
    assert report["sf"].nodes == ("a",)
    assert "no exact or numerical reliability" in report["sf"].reason
    assert report["random"].route == routes.SIMULATED
    assert report["unreliability_interval"].route == routes.SIMULATED
    assert report["structural_importance"].route == routes.EXACT
    assert rbd.get_non_analytic_nodes() == {"a": "StandbyModel"}
    exact = nonrepairable_kinds()["convolved standby"]
    assert exact.analysis_routes()["sf"].route == routes.NUMERICAL
    assert exact.is_analytically_solvable()


def test_common_cause_groups_refuse_what_does_not_model_them():
    report = nonrepairable_kinds()["common cause"].analysis_routes()
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
    report = repairable_kinds()[
        "tested, as long as the interval"
    ].analysis_routes()
    assert report["mean_availability"].route == routes.REFUSED
    assert report["mean_availability"].nodes == ("a",)
    assert report["availability"].route == routes.SIMULATED
    # numba's own loop simulates tests (#155), not imperfect repair.
    assert report["availability"].engine == "numba"
    imperfect = repairable_kinds()["imperfect repair"].analysis_routes()
    assert imperfect["availability"].engine == "python"
    assert "imperfect repair" in imperfect["availability"].engine_reason


def test_maintenance_makes_the_long_run_numerical():
    for name in (
        "age replacement, priced",
        "block replacement",
        "tested, Weibull",
    ):
        report = repairable_kinds()[name].analysis_routes()
        assert report["mean_availability"].route == routes.NUMERICAL
        assert report["mean_availability"].nodes == ("a",)
    report = repairable_kinds()["tested, constant rate"].analysis_routes()
    assert report["mean_availability"].route == routes.EXACT


def test_a_life_with_no_mean_refuses_the_long_run():
    report = repairable_kinds()["simulated standby life"].analysis_routes()
    assert report["mean_availability"].route == routes.REFUSED
    assert report["mean_availability"].nodes == ("a",)
    assert (
        "mean(mc_samples=..., seed=...)" in report["mean_availability"].reason
    )
    assert report["availability"].route == routes.SIMULATED


def test_the_engine_is_the_one_auto_would_run(monkeypatch):
    from repyability.rbd import _compiled

    plain = repairable_kinds()["koon"]
    monkeypatch.setattr(_compiled, "available", lambda: False)
    route = plain.analysis_routes()["availability"]
    assert route.engine == "python"
    assert "not installed" in route.engine_reason
    monkeypatch.setattr(_compiled, "available", lambda: True)
    assert plain.analysis_routes()["availability"].engine == "numba"
    capacities = repairable_kinds()["capacities"].analysis_routes()
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
        NonRepairableRBD: nonrepairable_kinds()["capacities"],
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
    nonrepairable, repairable = nonrepairable_kinds(), repairable_kinds()

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
    tested_group = repairable["common cause, tested"]
    revealed_group = repairable["common cause, revealed"]
    timed_group = repairable["common cause, timed tests and repairs"]
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
        "`time_to_reliability_uncertainty`; for a repairable system "
        "`mean_availability_uncertainty`, `point_availability_uncertainty`, "
        "`mission_availability_uncertainty`, "
        "`expected_cost_rate_uncertainty`)": [
            (plain, name, "simulated")
            for name in (
                "sf_uncertainty",
                "mean_uncertainty",
                "bx_life_uncertainty",
                "time_to_reliability_uncertainty",
            )
        ]
        + [
            (repairable["costed_pairs"], name, "simulated")
            for name in (
                "mean_availability_uncertainty",
                "point_availability_uncertainty",
                "mission_availability_uncertainty",
                "expected_cost_rate_uncertainty",
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
        "Common-cause groups in a repairable diagram": [
            (timed_group, "mean_availability", "numerical"),
            (timed_group, "birnbaum_importance", "numerical"),
            (timed_group, "point_availability", "numerical"),
            (timed_group, "system_failure_frequency", "refused"),
            (timed_group, "availability", "simulated"),
            (tested_group, "mean_availability", "exact"),
            (tested_group, "system_failure_frequency", "exact"),
            (tested_group, "birnbaum_importance", "exact"),
            (tested_group, "allocate_redundancy", "exact"),
            (tested_group, "availability_allocation", "exact"),
            (tested_group, "point_availability", "numerical"),
            (tested_group, "expected_cost", "numerical"),
            (tested_group, "availability", "simulated"),
            (tested_group, "simulate_timelines", "simulated"),
            (revealed_group, "mission_availability", "numerical"),
            (tested_group, "cost", "simulated"),
            (revealed_group, "availability", "simulated"),
            (
                repairable["common cause, Weibull"],
                "mean_availability",
                "refused",
            ),
            (repairable["common cause, Weibull"], "availability", "refused"),
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
        "Spares with repair crews, standby groups, opportunistic "
        "maintenance or imperfect repair": [
            (crew, "spares_stock", "refused"),
            (group, "spares_demand", "refused"),
            (
                repairable["opportunistic maintenance"],
                "spares_stock",
                "refused",
            ),
            (repairable["imperfect repair"], "spares_demand", "refused"),
            (repairable["block replacement"], "spares_stock", "numerical"),
            (repairable["tested, missing"], "spares_demand", "numerical"),
            (repairable["tested, missing"], "spares_stock", "numerical"),
            (repairable["tested, taking time"], "spares_demand", "numerical"),
            (repairable["tested, taking time"], "spares_stock", "numerical"),
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
