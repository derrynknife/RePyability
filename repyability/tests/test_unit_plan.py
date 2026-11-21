"""A unit's maintenance plan (UC-63 of the maintenance README): specs and
plans state one node, and a diagram of plans is the diagram of specs."""

import json
import warnings

import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.maintenance import (
    AgePolicy,
    AtShutdown,
    BlockPolicy,
    ConditionPolicy,
    ImperfectRepair,
    Inspection,
    Prices,
    Restoration,
    UnitPlan,
    Upgrade,
)
from repyability.rbd import modular
from repyability.tests.test_catalogue import built
from repyability.tests.test_simulation_engines import identical, needs_numba

#: The keys whose value at its default a plan leaves out of its spec.
_DEFAULTS = {"duration": "instant"}


def canonical(spec: dict, policy_default: str = "") -> dict:
    """``spec`` with the keys at their defaults left out (None, an
    ``"instant"`` duration, a task's policy where it is its default)."""
    out = {}
    for key, value in spec.items():
        if value is None or _DEFAULTS.get(key, object()) == value:
            continue
        if key in ("preventive", "inspection", "upgrade") and isinstance(
            value, dict
        ):
            value = canonical(value)
            if key == "preventive":
                default = (
                    "block"
                    if value.get("shutdown") is not None
                    or value.get("replace_at") is not None
                    else "age"
                )
                if value.get("policy") == default:
                    value.pop("policy")
        out[key] = value
    return out


def same(a, b) -> bool:
    """Whether ``a`` and ``b`` are one input: equal, or (a model built
    afresh from one given, such as a fitted process's life) saved alike."""
    if a is b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if hasattr(a, "to_dict") and type(a) is type(b):
        return a.to_dict() == b.to_dict()
    return bool(a == b)


def catalogue_specs(monkeypatch) -> list:
    """Every spec dict a catalogue diagram is given, by diagram and node."""
    return [
        (f"{type(rbd).__name__} {i}, node {name!r}", component)
        for i, (cls, rbd, components, _) in enumerate(built(monkeypatch))
        if cls is RepairableRBD
        for name, component in components.items()
        if isinstance(component, dict) and "reliability" in component
    ]


def test_every_catalogue_spec_reads_into_a_plan_and_back(monkeypatch):
    specs = catalogue_specs(monkeypatch)
    assert len(specs) > 50
    for where, spec in specs:
        plan = UnitPlan.from_spec(spec)
        assert plan.spec() == canonical(spec), where
        assert UnitPlan.from_spec(plan.spec()) == plan, where


def test_a_diagram_of_plans_is_the_diagram_of_specs(monkeypatch):
    diagrams = [
        (rbd, components, options)
        for cls, rbd, components, options in built(monkeypatch)
        if cls is RepairableRBD
        and any(isinstance(c, dict) for c in components.values())
    ]
    monkeypatch.undo()
    assert len(diagrams) > 30
    for rbd, components, options in diagrams:
        plans = {
            name: (
                UnitPlan.from_spec(c)
                if isinstance(c, dict) and "reliability" in c
                else c
            )
            for name, c in components.items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = RepairableRBD(components=plans, **options)
        for name, spec in rbd._init_args["components"].items():
            other = again._init_args["components"][name]
            if isinstance(spec, dict):
                spec, other = canonical(spec), canonical(other)
            assert same(spec, other), name
        if not isinstance(rbd._decomposition(), modular.GraphStructure):
            # (A meshed one is built under a lowered step limit.)
            assert again.analysis_routes() == rbd.analysis_routes()


def pump_spec() -> dict:
    return {
        "reliability": surv.Weibull.from_params([1000.0, 2.5]),
        "repairability": surv.Exponential.from_params([0.1]),
        "repair_cost": 400.0,
        "downtime_cost": 5.0,
        "duty": 0.6,
        "preventive": {
            "policy": "age",
            "interval": 4000.0,
            "basis": "operating",
            "cost": 20.0,
            "duration": surv.ExactEventTime.from_params([4.0]),
        },
    }


def pump_plan(spec: dict) -> UnitPlan:
    return UnitPlan(
        spec["reliability"],
        spec["repairability"],
        tasks=(
            AgePolicy(
                interval=4000.0,
                cost=20.0,
                duration=spec["preventive"]["duration"],
                basis="operating",
            ),
        ),
        prices=Prices(repair=400.0, downtime=5.0),
        duty=0.6,
    )


def test_the_pump_of_the_use_case():
    spec = pump_spec()
    plan = pump_plan(spec)
    assert plan.spec() == canonical(spec)
    assert UnitPlan.from_spec(spec) == plan
    assert plan.task("preventive").basis == "operating"
    assert plan.task("inspection") is None


def seeded_pair(engine: str):
    spec = pump_spec()
    edges = [("in", "pump"), ("pump", "valve"), ("valve", "out")]
    valve = {
        "reliability": surv.Exponential.from_params([1 / 3000.0]),
        "repairability": surv.Exponential.from_params([0.2]),
        "repair_cost": 50.0,
    }
    options = dict(
        t_simulation=8000.0,
        mc_samples=200,
        seed=4,
        engine=engine,
        control_variate=False,
    )
    of_specs = RepairableRBD(edges, {"pump": spec, "valve": valve})
    of_plans = RepairableRBD(
        edges, {"pump": pump_plan(spec), "valve": UnitPlan.from_spec(valve)}
    )
    return of_specs, of_plans, options


def test_seeded_runs_are_the_same_to_the_bit():
    of_specs, of_plans, options = seeded_pair("python")
    identical(
        of_specs.availability(**options), of_plans.availability(**options)
    )
    identical(of_specs.cost(**options), of_plans.cost(**options))


@needs_numba
def test_seeded_runs_are_the_same_on_the_compiled_engine():
    of_specs, of_plans, options = seeded_pair("numba")
    identical(
        of_specs.availability(**options), of_plans.availability(**options)
    )
    python = seeded_pair("python")
    identical(
        python[1].availability(**python[2]), of_plans.availability(**options)
    )


def test_plans_read_back_from_a_diagram():
    spec = pump_spec()
    rbd = RepairableRBD(
        [("in", "pump"), ("pump", "out")], {"pump": pump_plan(spec)}
    )
    assert rbd.plans == {"pump": pump_plan(spec)}


def test_a_plan_saves_and_loads():
    plan = UnitPlan(
        surv.Weibull.from_params([500.0, 1.5]),
        "instant",
        tasks=(
            BlockPolicy(
                interval=200.0,
                cost=3.0,
                restoration=Restoration(q=0.4, model="kijima1"),
            ),
        ),
        prices=Prices(
            repair=10.0, replace=surv.LogNormal.from_params([1.5, 0.2])
        ),
        imperfect_repair=ImperfectRepair(model="kijima2", q=0.5),
        priority=2.0,
        group="line",
    )
    loaded = UnitPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert loaded.spec().keys() == plan.spec().keys()
    assert loaded.task("preventive") == plan.task("preventive")
    assert loaded.imperfect_repair == plan.imperfect_repair
    assert loaded.prices.repair == 10.0
    assert loaded.prices.replace.params.tolist() == [1.5, 0.2]
    # The diagram saves a plan node as its spec, which loads as the node.
    rbd = RepairableRBD([("in", "a"), ("a", "out")], {"a": plan})
    assert RepairableRBD.from_dict(rbd.to_dict()).plans["a"].spec().keys() == (
        plan.spec().keys()
    )


@pytest.mark.parametrize(
    "task, spec",
    [
        (BlockPolicy(times=(10.0, 30.0)), {"replace_at": (10.0, 30.0)}),
        (
            ConditionPolicy(interval=50.0, threshold=0.1, inspection_cost=1.0),
            {
                "policy": "condition",
                "interval": 50.0,
                "threshold": 0.1,
                "inspection_cost": 1.0,
            },
        ),
        (
            AtShutdown(shutdown=["month", "year"], older_than=100.0),
            {"shutdown": ["month", "year"], "older_than": 100.0},
        ),
        (
            AtShutdown(shutdown="year", policy="condition", threshold=0.2),
            {"policy": "condition", "shutdown": "year", "threshold": 0.2},
        ),
        (
            AgePolicy(interval=100.0, restoration=Restoration(to_age=20.0)),
            {"interval": 100.0, "restore_to": 20.0},
        ),
        (
            AgePolicy(
                interval=100.0,
                restoration=Restoration(
                    q=1.1, replace_every=3, replacement_cost=9.0
                ),
            ),
            {
                "interval": 100.0,
                "restoration": {"model": "kijima2", "q": 1.1},
                "replace_every": 3,
                "replacement_cost": 9.0,
            },
        ),
    ],
)
def test_each_task_is_its_spec(task, spec):
    kind = "preventive"
    plan = UnitPlan("life", "instant", tasks=(task,))
    assert plan.spec()[kind] == spec
    assert UnitPlan.from_spec(plan.spec()) == plan


def test_inspection_and_upgrade_are_their_specs():
    plan = UnitPlan(
        "life",
        "instant",
        tasks=(
            Inspection(
                interval=720.0,
                offset=100.0,
                coverage=0.9,
                full_test=2160.0,
                cost=5.0,
            ),
            Upgrade(at=250.0, life="new life", cost=50.0),
        ),
    )
    assert plan.spec()["inspection"] == {
        "interval": 720.0,
        "offset": 100.0,
        "coverage": 0.9,
        "full_test": 2160.0,
        "cost": 5.0,
    }
    assert plan.spec()["upgrade"] == {
        "at": 250.0,
        "reliability": "new life",
        "cost": 50.0,
    }
    assert UnitPlan.from_spec(plan.spec()) == plan


def test_refusals_say_what_to_do():
    with pytest.raises(ValueError, match=r"AtShutdown\(shutdown=\[\.\.\.\]\)"):
        UnitPlan(
            "life",
            "instant",
            tasks=(AgePolicy(interval=1.0), BlockPolicy(interval=2.0)),
        )
    with pytest.raises(TypeError, match="must be one of AgePolicy"):
        UnitPlan("life", "instant", tasks=({"interval": 1.0},))
    with pytest.raises(ValueError, match="needs the unit's life"):
        UnitPlan(None)
    with pytest.raises(ValueError, match="one of q .* to_age .* process"):
        Restoration(q=0.5, to_age=10.0)
    with pytest.raises(ValueError, match=r"\['intervall'\].*spelling"):
        UnitPlan.from_spec(
            {"reliability": "life", "preventive": {"intervall": 3.0}}
        )
    with pytest.raises(ValueError, match=r"\['repair_costs'\]"):
        UnitPlan.from_spec({"reliability": "life", "repair_costs": 3.0})
    with pytest.raises(TypeError, match="prices are a Prices"):
        UnitPlan("life", prices=3.0)
    # A plan is checked by the diagram as its spec is, naming the node.
    plan = UnitPlan(
        surv.Weibull.from_params([500.0, 1.5]),
        "instant",
        tasks=(AgePolicy(interval=-1.0),),
    )
    with pytest.raises(ValueError, match="Component 'a': the preventive"):
        RepairableRBD([("in", "a"), ("a", "out")], {"a": plan})
