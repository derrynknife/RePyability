"""The catalogue of diagrams the analyses are checked on (#239).

``test_analysis_routes`` runs every analysis on every diagram of the
catalogue (``catalogue.py``), and checks it does what ``analysis_routes()``
says. Here:

- the catalogue has a diagram of every kind: every node class the package
  exports, every key of a repairable component's spec and every option of
  the diagrams, so that a new one cannot land without a diagram (and so
  without its routes checked);
- the two simulation engines agree on every repairable diagram the
  compiled one runs, and it refuses the others;
- where an analysis is exact or numerical, the simulation of the same
  quantity agrees with it, within its own error.
"""

import inspect
import math
import warnings

import numpy as np
import pytest

import repyability
from repyability import (
    DegradingNode,
    LoadSharingModel,
    NonRepairable,
    NonRepairableRBD,
    PerfectReliability,
    PerfectUnreliability,
    RegressionNode,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
)
from repyability.rbd import routes
from repyability.rbd._model_utils import is_fixed_probability, is_mixture
from repyability.rbd.results import _ResultMapping
from repyability.rbd.uncertainty import is_fit
from repyability.tests import catalogue
from repyability.tests.test_simulation_engines import identical, needs_numba

#: What the package exports that is not a node's model, so not asked of
#: the catalogue: a new export is either here or a node of the diagrams
#: that take it (below).
NOT_NODES = {
    "AnalysisRoute",
    "BetaFactor",
    "CCFGroup",
    "ComponentOption",
    "FailureLimitPolicy",
    "FaultTree",
    "MGL",
    "MaintenancePolicy",
    "Network",
    "NodeState",
    "PhasedMission",
    "RBD",
    # A component analysed alone: a diagram takes a NonRepairable.
    "Repairable",
    "SimulationChunk",
    "Timeline",
    "Timelines",
}
#: The node classes a NonRepairableRBD takes as a node's model.
NONREPAIRABLE_NODES = {
    DegradingNode,
    LoadSharingModel,
    NonRepairableRBD,
    PerfectReliability,
    PerfectUnreliability,
    RegressionNode,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
}
#: The node classes a RepairableRBD takes, as a component or its life.
REPAIRABLE_NODES = {
    DegradingNode,
    NonRepairable,
    PerfectReliability,
    RepairableRBD,
    StandbyModel,
}
#: The kinds of surpyval model every diagram class takes as a life.
SURPYVAL_KINDS = {
    "surpyval distribution",
    "surpyval fixed probability",
    "surpyval fit with a covariance",
}


def built(monkeypatch) -> list:
    """Each diagram the catalogue builds, as given to its class: the class,
    the diagram, its models (or components) and its other arguments.
    Nested diagrams are built as diagrams of their own; those the classes
    build for themselves are left out."""
    diagrams: list = []
    depth = [0]
    for cls in (NonRepairableRBD, RepairableRBD):
        original = cls.__init__
        signature = inspect.signature(original)

        def init(
            self, *args, _cls=cls, _init=original, _sig=signature, **kwargs
        ):
            outermost = depth[0] == 0
            depth[0] += 1
            try:
                _init(self, *args, **kwargs)
            finally:
                depth[0] -= 1
            if outermost:
                given = dict(_sig.bind(self, *args, **kwargs).arguments)
                given.pop("self")
                models = given.pop("reliabilities", None)
                if models is None:
                    models = given.pop("components")
                diagrams.append((_cls, self, dict(models), given))

        monkeypatch.setattr(cls, "__init__", init)
    catalogue.nonrepairable_kinds()
    catalogue.repairable_kinds()
    return diagrams


def model_kinds(model) -> set:
    """What a node's model is made of: its class, and those of the models
    inside it; for a surpyval model, what kind of model it is."""
    cls = model if isinstance(model, type) else type(model)
    if cls in (PerfectReliability, PerfectUnreliability):
        return {cls.__name__}
    if isinstance(model, DegradingNode):
        return {"DegradingNode"}.union(
            *(model_kinds(stage) for _, stage in model.stages)
        )
    if isinstance(model, StandbyModel):
        kinds = {"StandbyModel"}
        if model.is_simulated:
            kinds.add("StandbyModel, simulated")
        return kinds.union(*(model_kinds(m) for m in model.reliabilities))
    if isinstance(model, LoadSharingModel):
        return {"LoadSharingModel"}.union(
            *(model_kinds(m) for m in model.models)
        )
    if isinstance(model, (RepeatedNode, RepeatedStandbyNode, RegressionNode)):
        return {cls.__name__} | model_kinds(model.model)
    if isinstance(model, (NonRepairableRBD, RepairableRBD)):
        return {cls.__name__}
    if isinstance(model, NonRepairable):
        return {"NonRepairable"} | model_kinds(model.reliability)
    if is_mixture(model):
        return {"surpyval MixtureModel"}
    if "Regression" in cls.__name__:
        return {"surpyval regression model"}
    if is_fixed_probability(model):
        return {"surpyval fixed probability"}
    kinds = {"surpyval distribution"}
    if is_fit(model):
        kinds.add("surpyval fit with a covariance")
    return kinds


def common_causes(options) -> set:
    return {
        f"{type(group.model).__name__} common cause, "
        f"{group.model.basis} basis"
        for group in options.get("ccf_groups") or ()
    }


def nonrepairable_kinds(rbd, models, options) -> set:
    found = set().union(*(model_kinds(m) for m in models.values()))
    if options.get("k"):
        found.add("a k-out-of-n vote")
    if options.get("capacity"):
        found.add("capacities")
    if rbd.structure_check["is_too_meshed"]:
        found.add("a structure too meshed to work out")
    return found | common_causes(options)


def repairable_kinds(rbd, components, options) -> set:
    found = set()
    for component in components.values():
        if not isinstance(component, dict):
            found |= model_kinds(component)
            continue
        found |= {f"spec key {key!r}" for key in component}
        found |= model_kinds(component["reliability"])
        if isinstance(component["repairability"], str):
            found.add(f"repairability {component['repairability']!r}")
        for key in ("preventive", "inspection", "standby", "repair"):
            found |= {f"{key} key {sub!r}" for sub in component.get(key) or {}}
        preventive = component.get("preventive")
        if preventive:
            found.add(f"policy {preventive.get('policy', 'age')!r}")
        if component.get("repair"):
            found.add(f"repair model {component['repair']['model']!r}")
    for group in (options.get("maintenance_groups") or {}).values():
        found |= {f"maintenance group key {key!r}" for key in group}
    if options.get("k"):
        found.add("a k-out-of-n vote")
    if options.get("capacity"):
        found.add("capacities")
    if options.get("downtime_cost_rate"):
        found.add("downtime_cost_rate")
    if options.get("repair_crews") is not None:
        found.add("repair crews")
        if any(isinstance(c, RepairableRBD) for c in components.values()):
            found.add("repair crews around a nested RepairableRBD")
    groups = options.get("ccf_groups") or ()
    if any(
        isinstance(components[m], dict) and components[m].get("inspection")
        for group in groups
        for m in group.members
    ):
        found.add("common cause among tested components")
    if rbd.structure_check["is_too_meshed"]:
        found.add("a structure too meshed to work out")
    return found | common_causes(options)


def required_repairable() -> set:
    cls = RepairableRBD
    return (
        {f"spec key {key!r}" for key in cls.COMPONENT_SPEC_KEYS}
        | {f"preventive key {key!r}" for key in cls.PREVENTIVE_KEYS}
        | {f"policy {policy!r}" for policy in cls.PREVENTIVE_POLICIES}
        | {f"inspection key {key!r}" for key in cls.INSPECTION_KEYS}
        | {f"standby key {key!r}" for key in cls.STANDBY_KEYS}
        | {"repair key 'model'", "repair key 'q'"}
        | {f"repair model {model!r}" for model in cls.REPAIR_MODELS}
        | {f"maintenance group key {key!r}" for key in cls.GROUP_KEYS}
        | {"repairability 'instant'"}
        | {"a k-out-of-n vote", "capacities", "downtime_cost_rate"}
        | {"repair crews", "repair crews around a nested RepairableRBD"}
        | {
            "BetaFactor common cause, probability basis",
            "MGL common cause, probability basis",
            "common cause among tested components",
        }
        | {"a structure too meshed to work out", "StandbyModel, simulated"}
        | {node.__name__ for node in REPAIRABLE_NODES}
        | SURPYVAL_KINDS
    )


def required_nonrepairable() -> set:
    return (
        {node.__name__ for node in NONREPAIRABLE_NODES}
        | SURPYVAL_KINDS
        | {"surpyval MixtureModel", "surpyval regression model"}
        | {"a k-out-of-n vote", "capacities"}
        | {
            "BetaFactor common cause, probability basis",
            "BetaFactor common cause, rate basis",
            "MGL common cause, probability basis",
        }
        | {"a structure too meshed to work out", "StandbyModel, simulated"}
    )


def test_every_export_is_a_node_or_said_not_to_be():
    exported = {
        name
        for name in repyability.__all__
        if inspect.isclass(getattr(repyability, name))
        and not issubclass(getattr(repyability, name), _ResultMapping)
    }
    nodes = {cls.__name__ for cls in NONREPAIRABLE_NODES | REPAIRABLE_NODES}
    assert exported - NOT_NODES == nodes, (
        "Say which diagrams take each new export (NONREPAIRABLE_NODES, "
        "REPAIRABLE_NODES), or that it is not a node (NOT_NODES)."
    )


def test_the_catalogue_has_every_kind(monkeypatch):
    found = {NonRepairableRBD: set(), RepairableRBD: set()}
    for cls, rbd, models, options in built(monkeypatch):
        kinds = (
            nonrepairable_kinds
            if cls is NonRepairableRBD
            else repairable_kinds
        )
        found[cls] |= kinds(rbd, models, options)
    missing = {
        "NonRepairableRBD": sorted(
            required_nonrepairable() - found[NonRepairableRBD]
        ),
        "RepairableRBD": sorted(required_repairable() - found[RepairableRBD]),
    }
    assert missing == {"NonRepairableRBD": [], "RepairableRBD": []}, (
        "Add a diagram with each of these to catalogue.py: " f"{missing}"
    )


# -- the engines --------------------------------------------------------------


@needs_numba
@pytest.mark.parametrize("name", sorted(catalogue.repairable_kinds()))
def test_the_engines_agree_on_every_kind(name):
    rbd = catalogue.repairable_kinds()[name]
    route = rbd.analysis_routes()["availability"]
    if route.route == routes.REFUSED:
        pytest.skip(route.reason)
    options = dict(
        t_simulation=300.0, mc_samples=30, seed=5, control_variate=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if route.engine != "numba":
            with pytest.raises(NotImplementedError):
                rbd.availability(engine="numba", **options)
            return
        identical(
            rbd.availability(engine="python", **options),
            rbd.availability(engine="numba", **options),
        )
        if rbd.has_costs:
            identical(
                rbd.cost(engine="python", **options),
                rbd.cost(engine="numba", **options),
            )


# -- the exact and numerical routes against simulation ------------------------

#: Standard errors a simulated estimate may be from the exact value.
Z = 4.0


def computed(route) -> bool:
    return route.route in (routes.EXACT, routes.NUMERICAL)


def close(estimate: float, exact: float, error: float, n: int) -> bool:
    """Whether a simulated mean of ``n`` runs is within ``Z`` standard
    errors of the exact value, give or take what one run could add."""
    return abs(estimate - exact) <= Z * error + 1.0 / n


@pytest.mark.parametrize("name", sorted(catalogue.nonrepairable_kinds()))
def test_simulated_lifetimes_agree_with_the_exact_values(name):
    rbd = catalogue.nonrepairable_kinds()[name]
    report = rbd.analysis_routes()
    if report["random"].route == routes.REFUSED:
        pytest.skip(report["random"].reason)
    if any(g.model.basis == "probability" for g in rbd.ccf_groups):
        pytest.skip(
            "The simulation leaves a probability-basis common cause out, "
            "as its route says."
        )
    checked = 0
    n = 4000
    if computed(report["sf"]):
        lives = rbd.random(n, seed=7)
        for x in (10.0, 30.0, 100.0):
            p = float(rbd.sf(x))
            estimate = float(np.mean(lives > x))
            error = math.sqrt(p * (1.0 - p) / n)
            assert close(estimate, p, error, n), (x, estimate, p)
            checked += 1
    if computed(report["mean"]):
        exact = float(rbd.mean())
        interval = rbd.mean_time_to_failure_interval(n, seed=8)
        assert abs(interval.estimate - exact) <= Z * interval.standard_error
        checked += 1
    if not checked:
        pytest.skip("Nothing here is exact or numerical.")


#: The window each repairable diagram is simulated over: long enough for its
#: components to fail and be repaired a few times.
WINDOW = {"instrument air": 20000.0}


@pytest.mark.parametrize("name", sorted(catalogue.repairable_kinds()))
def test_the_simulation_agrees_with_the_exact_values(name):
    rbd = catalogue.repairable_kinds()[name]
    report = rbd.analysis_routes()
    window = WINDOW.get(name, 1000.0)
    n = 400
    # Plain runs: a control variate or a conditional run would take the
    # exact values in (CLAUDE.md).
    plain = dict(
        mc_samples=n, seed=13, control_variate=False, conditional=False
    )
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if (
            computed(report["mission_availability"])
            and report["availability"].route == routes.SIMULATED
        ):
            exact = rbd.mission_availability(window)
            interval = rbd.availability(
                window, **plain
            ).mean_availability_interval()
            assert close(
                interval.estimate, exact, interval.standard_error, n
            ), (interval, exact)
            checked += 1
        if (
            rbd.has_costs
            and computed(report["expected_cost"])
            and report["cost"].route == routes.SIMULATED
        ):
            exact = rbd.expected_cost(window).mean
            interval = rbd.cost(window, **plain).mean_interval()
            assert abs(interval.estimate - exact) <= (
                Z * interval.standard_error + 1e-9 * abs(exact)
            ), (interval, exact)
            checked += 1
    if not checked:
        pytest.skip("Nothing here is exact or numerical.")
