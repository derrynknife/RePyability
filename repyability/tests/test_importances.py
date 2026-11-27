"""``importances``: every importance measure in one table.

Each entry of the table is what the measure's own method returns, bit for
bit, on every diagram, tree, network and mission of the catalogue; what a
model refuses (or only simulates) is listed with the reason its route
gives; and the rows hold every component once.
"""

import json

import pytest
import surpyval as surv

from repyability import (
    FaultTree,
    ImportanceTable,
    NonRepairableRBD,
    PhasedMission,
    RepairableRBD,
)
from repyability.rbd import _importances, routes
from repyability.tests.catalogue import (
    fault_tree_kinds,
    mission_kinds,
    nonrepairable_kinds,
    repairable_kinds,
)
from repyability.tests.test_analysis_routes import X, networks
from repyability.tests.test_simulation_engines import identical

E = surv.Exponential.from_params


def single(model, name):
    """The measure's own method, called as ``importances`` calls it."""
    if isinstance(model, FaultTree):
        return (
            getattr(model, name)(X)
            if name != "structural_importance"
            else (model.structural_importance())
        )
    if isinstance(model, PhasedMission):
        return getattr(model, name)()
    if isinstance(model, RepairableRBD):
        return getattr(model, name)()
    return (
        getattr(model, name)(X)
        if name != "structural_importance"
        else model.structural_importance()
    )


def table_of(model):
    if isinstance(model, (PhasedMission, RepairableRBD)):
        return model.importances()
    return model.importances(X)


def check(model):
    report = model.analysis_routes()
    names = [n for n in _importances.MEASURES if n in report]
    if report["importances"].route == routes.REFUSED:
        with pytest.raises(NotImplementedError) as error:
            table_of(model)
        assert str(error.value) == report["importances"].reason
        return
    table = table_of(model)
    assert isinstance(table, ImportanceTable)
    refused = {
        n: report[n].reason for n in names if report[n].route == routes.REFUSED
    }
    refused.update(
        {
            n: _importances.SIMULATED_REASON
            for n in names
            if report[n].route == routes.SIMULATED
        }
    )
    assert table.refused == refused
    assert list(table.measures) == [n for n in names if n not in refused]
    for name, value in table.measures.items():
        identical(value, single(model, name))
    components = [row["component"] for row in table.rows()]
    assert len(components) == len(set(components))
    for name, value in table.measures.items():
        if name != "joint_importance":
            assert set(value) <= set(components)
    json.dumps(table.to_dict())


@pytest.mark.parametrize("name", sorted(nonrepairable_kinds()))
def test_a_diagram_table_is_its_methods(name):
    check(nonrepairable_kinds()[name])


@pytest.mark.parametrize("name", sorted(repairable_kinds()))
def test_a_repairable_table_is_its_methods(name):
    check(repairable_kinds()[name])


@pytest.mark.parametrize("name", sorted(fault_tree_kinds()))
def test_a_tree_table_is_its_methods(name):
    check(fault_tree_kinds()[name])


@pytest.mark.parametrize("name", sorted(networks()))
def test_a_network_table_is_its_methods(name):
    check(networks()[name])


@pytest.mark.parametrize("name", sorted(mission_kinds()))
def test_a_mission_table_is_its_methods(name):
    check(mission_kinds()[name])


def pair():
    return NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": E([0.01]), "b": E([0.02])},
    )


def test_the_rows_have_every_component_once_and_no_pairs():
    table = pair().importances(50)
    rows = table.rows()
    assert [row["component"] for row in rows] == ["a", "b"]
    assert "joint_importance" in table.measures
    assert all("joint_importance" not in row for row in rows)
    assert rows[0]["birnbaum_importance"] == (
        table.measures["birnbaum_importance"]["a"]
    )


def test_an_array_of_times_gives_arrays_that_are_lists_in_the_rows():
    table = pair().importances([10, 50])
    assert len(table.measures["birnbaum_importance"]["a"]) == 2
    assert isinstance(table.rows()[0]["birnbaum_importance"], list)
    json.dumps(table.to_dict())


def test_conditioning_arguments_reach_every_measure():
    rbd = pair()
    table = rbd.importances(50, working_nodes=["a"])
    identical(
        table.measures["birnbaum_importance"],
        rbd.birnbaum_importance(50, working_nodes=["a"]),
    )


def test_a_measure_the_model_refuses_is_named_with_its_reason():
    tree = FaultTree({"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2})
    table = tree.importances()
    assert "barlow_proschan_importance" in table.refused
    assert table.refused == {
        "barlow_proschan_importance": tree.analysis_routes()[
            "barlow_proschan_importance"
        ].reason
    }


def test_a_simulated_measure_runs_only_when_asked(monkeypatch):
    rbd = pair()
    real = rbd.analysis_routes

    def routed():
        report = real()
        report["birnbaum_importance"] = routes.AnalysisRoute(
            routes.SIMULATED, "drawn"
        )
        return report

    monkeypatch.setattr(rbd, "analysis_routes", routed)
    assert rbd.importances(50).refused == {
        "birnbaum_importance": "simulated: pass simulate=True"
    }
    assert "birnbaum_importance" in rbd.importances(50, simulate=True).measures
    with pytest.raises(ValueError, match="simulate must be"):
        rbd.importances(50, simulate="yes")


def test_a_table_goes_through_json_and_reads_as_a_mapping():
    table = pair().importances(50)
    again = json.loads(json.dumps(table.to_dict()))
    assert set(again) == {"measures", "refused"}
    assert set(table) == {"measures", "refused"}
    assert table["refused"] == table.refused


def test_the_route_is_the_worst_of_the_measures_that_run():
    report = pair().analysis_routes()
    run = [
        report[n].route
        for n in _importances.MEASURES
        if report[n].route in (routes.EXACT, routes.NUMERICAL)
    ]
    expected = routes.NUMERICAL if routes.NUMERICAL in run else routes.EXACT
    assert report["importances"].route == expected
