"""(De)serialisation of RBDs and their node models to plain dicts / JSON.

An RBD only existed as Python code; this module round-trips it to a
JSON-friendly structure so it can be saved, loaded, shared and version
controlled (e.g. by the Reliafy app).

Design notes
------------
- Node identity is preserved through JSON. Node names may be ints or strings,
  but JSON object keys are always strings, so the per-node collections
  (reliabilities, components, k, capacity) are serialised as *lists of
  entries* (``{"node": n, ...}``) rather than dicts keyed by node.
- The constructor inputs are captured verbatim at construction time and
  serialised, so ``from_dict(rbd.to_dict())`` simply reconstructs the RBD by
  calling its constructor again — faithful even for repeated nodes (whose
  graph is collapsed after construction).
- Node models are serialised structurally: surpyval's parametric models in
  its own format, ``model.to_dict()``, loaded with
  ``surpyval.from_dict``, so everything surpyval keeps (an offset, ``p``,
  ``f0``, a fit's covariance) round-trips; the RePyability wrappers
  (standby, degrading, repeated, NonRepairable) recursively; nested RBDs via
  their own
  ``to_dict``. Files from before 0.10.0, which saved a parametric model as
  ``(dist name, params, extras)``, still load.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from repyability._version import __version__
from repyability.non_repairable import NonRepairable
from repyability.rbd._model_utils import distribution_name, lfp_extras
from repyability.rbd.degrading_node import DegradingNode
from repyability.rbd.helper_classes import (
    PerfectReliability,
    PerfectUnreliability,
)
from repyability.rbd.load_sharing_node import LoadSharingModel
from repyability.rbd.rbd import RBD
from repyability.rbd.regression_node import RegressionNode
from repyability.rbd.repeated_node import PARALLEL, RepeatedNode
from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
from repyability.rbd.standby_node import StandbyModel

#: Whether every model must load back as the class it is (see
#: ``exactly``).
_EXACT: ContextVar[bool] = ContextVar("exact", default=False)

#: The class each kind of saved RePyability node wrapper loads as.
_WRAPPERS = {
    "degrading": DegradingNode,
    "standby": StandbyModel,
    "repeated_standby": RepeatedStandbyNode,
    "repeated_node": RepeatedNode,
    "non_repairable": NonRepairable,
    "regression_node": RegressionNode,
    "load_sharing": LoadSharingModel,
}


@contextmanager
def exactly():
    """Save so that every model loads back as the class it is: a model of
    a subclass (of a surpyval distribution, or of a RePyability node
    wrapper), which saves as the class it extends and would load without
    what the subclass changes, raises ``NotImplementedError``. A shard
    (see ``repyability.rbd.shards``), which a worker must simulate as the
    system itself, saves its system so."""
    token = _EXACT.set(True)
    try:
        yield
    finally:
        _EXACT.reset(token)


def _check_exact(model: Any, saved: dict) -> None:
    """Under ``exactly``: refuse a model that would load back as another
    class than its own."""
    import surpyval

    kind = saved["kind"]
    pairs = []
    if kind == "surpyval":
        pairs.append((model, type(surpyval.from_dict(saved["model"]))))
    elif kind in _WRAPPERS:
        pairs.append((model, _WRAPPERS[kind]))
    if kind == "load_sharing":
        pairs.extend(
            (unit, type(surpyval.from_dict(unit_saved)))
            for unit, unit_saved in zip(model.models, saved["models"])
        )
    for item, loads_as in pairs:
        if type(item) is not loads_as:
            package = loads_as.__module__.split(".")[0]
            raise NotImplementedError(
                f"a node model of type {type(item).__name__} saves as "
                f"{package}'s {loads_as.__name__}, and would load without "
                "what it changes"
            )


def serialise_model(model: Any) -> dict:
    """Serialise a node model to a JSON-friendly dict."""
    saved = _serialise_model(model)
    if _EXACT.get():
        _check_exact(model, saved)
    return saved


def _serialise_model(model: Any) -> dict:
    if model is PerfectReliability:
        return {"kind": "perfect_reliability"}
    if model is PerfectUnreliability:
        return {"kind": "perfect_unreliability"}
    if isinstance(model, RBD):
        return {"kind": "rbd", "rbd": rbd_to_dict(model)}
    if isinstance(model, DegradingNode):
        # Before StandbyModel, which it extends.
        return {
            "kind": "degrading",
            "stages": [
                {"capacity": capacity, "model": serialise_model(stage)}
                for capacity, stage in model.stages
            ],
        }
    if isinstance(model, StandbyModel):
        return {
            "kind": "standby",
            "reliabilities": [serialise_model(m) for m in model.reliabilities],
            "k": model.k,
            "switching_probability": model.switching_probability,
            "dormancy_factor": model.dormancy_factor,
        }
    if isinstance(model, RepeatedStandbyNode):
        return {
            "kind": "repeated_standby",
            "model": serialise_model(model.model),
            "repeats": model.repeats,
            "switching_probability": model.switching_probability,
        }
    if isinstance(model, RepeatedNode):
        return {
            "kind": "repeated_node",
            "model": serialise_model(model.model),
            "repeats": model.repeats,
            "repeated_kind": (
                "parallel" if model.kind == PARALLEL else "series"
            ),
        }
    if isinstance(model, NonRepairable):
        return {
            "kind": "non_repairable",
            "reliability": serialise_model(model.reliability),
            "time_to_replace": serialise_model(model.time_to_replace),
        }
    if isinstance(model, RegressionNode):
        return {"kind": "regression_node", **model.to_dict()}
    if isinstance(model, LoadSharingModel):
        return {
            "kind": "load_sharing",
            "models": [m.to_dict() for m in model.models],
            "load": model.load,
            "k": model.k,
        }
    if distribution_name(model) is not None:
        # surpyval's own format: everything surpyval keeps (an offset, p,
        # f0, a fit's covariance) round-trips, whatever it adds later.
        return {"kind": "surpyval", "model": model.to_dict()}
    raise NotImplementedError(
        f"Cannot serialise a node model of type {type(model).__name__}. "
        "Only surpyval's parametric models, the "
        "RePyability node wrappers (standby, degrading, repeated, "
        "NonRepairable, load-sharing, regression), perfect "
        "reliability/unreliability and nested RBDs are supported."
    )


def deserialise_model(d: dict) -> Any:
    """Reconstruct a node model from :func:`serialise_model`'s output."""
    import surpyval

    kind = d["kind"]
    if kind == "perfect_reliability":
        return PerfectReliability
    if kind == "perfect_unreliability":
        return PerfectUnreliability
    if kind == "surpyval":
        return surpyval.from_dict(d["model"])
    if kind == "degrading":
        return DegradingNode(
            [
                (stage["capacity"], deserialise_model(stage["model"]))
                for stage in d["stages"]
            ]
        )
    if kind == "parametric":
        # The format before 0.10.0: a distribution's name, parameters and
        # any offset, p and f0 ("extras", since 0.9.0). p is the
        # limited-failure proportion, lfp_p from surpyval 0.23.
        cls = getattr(surpyval, d["dist"])
        extras = dict(d.get("extras", {}))
        if "p" in extras:
            extras.update(lfp_extras(extras.pop("p")))
        return cls.from_params(d["params"], **extras)
    if kind == "rbd":
        return rbd_from_dict(d["rbd"])
    if kind == "standby":
        return StandbyModel(
            [deserialise_model(m) for m in d["reliabilities"]],
            k=d["k"],
            switching_probability=d.get("switching_probability", 1.0),
            dormancy_factor=d.get("dormancy_factor", 0.0),
        )
    if kind == "repeated_standby":
        return RepeatedStandbyNode(
            deserialise_model(d["model"]),
            d["repeats"],
            switching_probability=d.get("switching_probability", 1.0),
        )
    if kind == "repeated_node":
        return RepeatedNode(
            deserialise_model(d["model"]),
            d["repeats"],
            d["repeated_kind"],
        )
    if kind == "non_repairable":
        return NonRepairable(
            deserialise_model(d["reliability"]),
            deserialise_model(d["time_to_replace"]),
        )
    if kind == "regression_node":
        return RegressionNode.from_dict(d)
    if kind == "load_sharing":
        return LoadSharingModel(
            [surpyval.from_dict(md) for md in d["models"]],
            load=d["load"],
            k=d["k"],
        )
    raise ValueError(f"Unknown model kind {kind!r}.")


def _node_name(name: Any) -> Any:
    """A node name as read back from a document.

    JSON has no tuples, so a tuple node name comes back as a list. Node names
    are hashable and a list is not, so any list must have been a tuple.
    """
    if isinstance(name, list):
        return tuple(_node_name(part) for part in name)
    return name


def _serialise_reliability_value(node, value, all_nodes) -> dict:
    # A repeated node's value is the name of the node it repeats, not a model.
    if value in all_nodes:
        return {"kind": "repeat_of", "node": value}
    return serialise_model(value)


def _deserialise_reliability_value(d: dict) -> Any:
    if d.get("kind") == "repeat_of":
        return _node_name(d["node"])
    return deserialise_model(d)


def _serialise_component(value) -> dict:
    # RepairableRBD components: a {reliability, repairability} spec (which may
    # also carry cost fields), a NonRepairable, or a nested RepairableRBD.
    if isinstance(value, dict):
        from repyability.rbd.repairable_rbd import RepairableRBD

        repairability = value["repairability"]
        out: dict[str, Any] = {
            "kind": "component_spec",
            "reliability": serialise_model(value["reliability"]),
            # "instant" (repair in zero time) is a sentinel, not a model.
            "repairability": (
                "instant"
                if repairability == "instant"
                else serialise_model(repairability)
            ),
        }
        for key in RepairableRBD.COST_KEYS:
            cost = value.get(key)
            if hasattr(cost, "qf"):
                # A per-failure cost may be a distribution of the cost.
                out[key] = serialise_model(cost)
            elif cost:
                out[key] = float(cost)
        if value.get("acquisition_cost"):
            out["acquisition_cost"] = float(value["acquisition_cost"])
        if value.get("priority"):
            out["priority"] = float(value["priority"])
        if value.get("group") is not None:
            # A maintenance group's name, as a node's is.
            out["group"] = value["group"]
        if value.get("repair") is not None:
            # Imperfect repair: the model's name and its restoration factor.
            out["repair"] = {
                "model": value["repair"]["model"],
                "q": float(value["repair"]["q"]),
            }
        if value.get("replace_after") is not None:
            out["replace_after"] = int(value["replace_after"])
        for key in ("preventive", "inspection"):
            if value.get(key) is not None:
                out[key] = _serialise_schedule(value[key])
        if value.get("standby") is not None:
            # A standby group: numbers only.
            out["standby"] = {
                key: float(number) if isinstance(number, float) else number
                for key, number in value["standby"].items()
            }
        return out
    return serialise_model(value)


def _serialise_schedule(spec: dict) -> dict:
    # A component's preventive-maintenance or inspection schedule: the
    # interval (and policy, and a condition policy's threshold) as they are,
    # the duration ("instant" or a model) and the costs (each a number or a
    # distribution).
    out: dict[str, Any] = {"interval": float(spec["interval"])}
    if "policy" in spec:
        out["policy"] = spec["policy"]
    if spec.get("threshold") is not None:
        out["threshold"] = float(spec["threshold"])
    if spec.get("opportunity") is not None:
        out["opportunity"] = float(spec["opportunity"])
    # An inspection's offset, coverage and full tests, when given (so a
    # file without them is unchanged).
    for key in ("offset", "coverage", "full_test"):
        if spec.get(key) is not None:
            out[key] = float(spec[key])
    duration = spec.get("duration", "instant")
    out["duration"] = (
        "instant" if isinstance(duration, str) else serialise_model(duration)
    )
    for key in ("cost", "inspection_cost"):
        cost = spec.get(key)
        if hasattr(cost, "qf"):
            out[key] = serialise_model(cost)
        elif cost:
            out[key] = float(cost)
    return out


def _deserialise_schedule(d: dict) -> dict:
    out = dict(d)
    if isinstance(d["duration"], dict):
        out["duration"] = deserialise_model(d["duration"])
    for key in ("cost", "inspection_cost"):
        if isinstance(d.get(key), dict):
            out[key] = deserialise_model(d[key])
    return out


def _deserialise_component(d: dict) -> Any:
    if d.get("kind") == "component_spec":
        from repyability.rbd.repairable_rbd import RepairableRBD

        out: Any = {
            "reliability": deserialise_model(d["reliability"]),
            "repairability": (
                "instant"
                if d["repairability"] == "instant"
                else deserialise_model(d["repairability"])
            ),
        }
        for key in RepairableRBD.COST_KEYS:
            if key in d:
                cost = d[key]
                out[key] = (
                    deserialise_model(cost) if isinstance(cost, dict) else cost
                )
        if "acquisition_cost" in d:
            out["acquisition_cost"] = d["acquisition_cost"]
        if "priority" in d:
            out["priority"] = d["priority"]
        if "group" in d:
            out["group"] = _node_name(d["group"])
        if "repair" in d:
            out["repair"] = dict(d["repair"])
        if "replace_after" in d:
            out["replace_after"] = d["replace_after"]
        for key in ("preventive", "inspection"):
            if key in d:
                out[key] = _deserialise_schedule(d[key])
        if "standby" in d:
            out["standby"] = dict(d["standby"])
        return out
    return deserialise_model(d)


def _k_to_list(k):
    return None if not k else [{"node": n, "k": v} for n, v in k.items()]


def _k_from_list(k_list):
    if not k_list:
        return None
    return {_node_name(e["node"]): e["k"] for e in k_list}


def _group_options(options) -> dict:
    # A maintenance group's options: its set-up cost and whether a system
    # outage is a stop (numbers and booleans only).
    out: dict[str, Any] = {}
    options = options or {}
    if options.get("setup_cost"):
        out["setup_cost"] = float(options["setup_cost"])
    if options.get("system_down"):
        out["system_down"] = True
    return out


def _capacity_to_list(capacity):
    # An unlimited capacity is float("inf"), which json writes as Infinity.
    # A node working at several levels keeps them as [level, probability]
    # pairs: JSON object keys are strings.
    if not capacity:
        return None
    out = []
    for n, v in capacity.items():
        if isinstance(v, dict):
            levels = [[float(level), float(p)] for level, p in v.items()]
            out.append({"node": n, "levels": levels})
        else:
            out.append({"node": n, "capacity": float(v)})
    return out


def _capacity_from_list(capacity_list):
    if not capacity_list:
        return None
    return {
        _node_name(e["node"]): (
            {level: p for level, p in e["levels"]}
            if "levels" in e
            else e["capacity"]
        )
        for e in capacity_list
    }


def _ccf_to_list(ccf_groups):
    from repyability.rbd.ccf import MGL, BetaFactor

    if not ccf_groups:
        return None
    out = []
    for group in ccf_groups:
        if isinstance(group.model, BetaFactor):
            model = {"kind": "beta_factor", "beta": group.model.beta}
        elif isinstance(group.model, MGL):
            model = {"kind": "mgl", "letters": list(group.model.letters)}
        else:
            raise NotImplementedError(
                f"Cannot serialise CCF model {type(group.model).__name__}."
            )
        if group.model.basis != "probability":
            # Saved only when not the default, so older files read as
            # they always have.
            model["basis"] = group.model.basis
        elif group.model.shocks == "independent":
            model["shocks"] = "independent"
        out.append({"members": list(group.members), "model": model})
    return out


def _ccf_from_list(ccf_list):
    from repyability.rbd.ccf import MGL, BetaFactor, CCFGroup

    if not ccf_list:
        return None
    groups = []
    for entry in ccf_list:
        model_dict = entry["model"]
        kind = model_dict["kind"]
        basis = model_dict.get("basis", "probability")
        if kind == "beta_factor":
            model: object = BetaFactor(model_dict["beta"], basis=basis)
        elif kind == "mgl":
            model = MGL(
                *model_dict["letters"],
                basis=basis,
                shocks=model_dict.get("shocks"),
            )
        else:
            raise ValueError(f"Unknown CCF model kind {kind!r}.")
        members = [_node_name(m) for m in entry["members"]]
        groups.append(CCFGroup(members, model))
    return groups


def rbd_to_dict(rbd: RBD) -> dict:
    """Serialise an RBD (NonRepairableRBD or RepairableRBD) to a dict."""
    args = rbd._init_args
    out = {
        "repyability_version": __version__,
        "type": type(rbd).__name__,
        "edges": [list(e) for e in args["edges"]],
        "k": _k_to_list(args["k"]),
        "capacity": _capacity_to_list(args.get("capacity")),
        "input_node": args["input_node"],
        "output_node": args["output_node"],
        "on_infeasible_rbd": args["on_infeasible_rbd"],
    }
    if out["type"] == "RepairableRBD":
        out["components"] = [
            {"node": n, "component": _serialise_component(v)}
            for n, v in args["components"].items()
        ]
        out["downtime_cost_rate"] = args.get("downtime_cost_rate", 0.0)
        if args.get("repair_crews") is not None:
            out["repair_crews"] = int(args["repair_crews"])
        if args.get("maintenance_groups"):
            # Each group's options, by name (a name may not be a JSON key).
            out["maintenance_groups"] = [
                {"group": name, **_group_options(options)}
                for name, options in args["maintenance_groups"].items()
            ]
        if args.get("ccf_groups"):
            out["ccf_groups"] = _ccf_to_list(args["ccf_groups"])
    else:
        nodes = set(args["reliabilities"].keys())
        out["reliabilities"] = [
            {
                "node": n,
                "model": _serialise_reliability_value(n, v, nodes),
            }
            for n, v in args["reliabilities"].items()
        ]
        out["ccf_groups"] = _ccf_to_list(args.get("ccf_groups"))
    return out


def rbd_from_dict(d: dict) -> RBD:
    """Reconstruct an RBD from :func:`rbd_to_dict`'s output."""
    # Lazy imports to avoid an import cycle (these modules import this one).
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD
    from repyability.rbd.repairable_rbd import RepairableRBD

    rbd_type = d["type"]
    edges = [tuple(_node_name(n) for n in e) for e in d["edges"]]
    common = dict(
        k=_k_from_list(d.get("k")),
        capacity=_capacity_from_list(d.get("capacity")),
        input_node=_node_name(d.get("input_node")),
        output_node=_node_name(d.get("output_node")),
        on_infeasible_rbd=d.get("on_infeasible_rbd", "raise"),
    )
    if rbd_type == "RepairableRBD":
        components = {
            _node_name(e["node"]): _deserialise_component(e["component"])
            for e in d["components"]
        }
        groups = {
            _node_name(e["group"]): {
                key: value for key, value in e.items() if key != "group"
            }
            for e in d.get("maintenance_groups") or ()
        }
        return RepairableRBD(
            edges,
            components,
            downtime_cost_rate=d.get("downtime_cost_rate", 0.0),
            repair_crews=d.get("repair_crews"),
            maintenance_groups=groups or None,
            ccf_groups=_ccf_from_list(d.get("ccf_groups")),
            **common,
        )
    if rbd_type == "NonRepairableRBD":
        reliabilities = {
            _node_name(e["node"]): _deserialise_reliability_value(e["model"])
            for e in d["reliabilities"]
        }
        return NonRepairableRBD(
            edges,
            reliabilities,
            ccf_groups=_ccf_from_list(d.get("ccf_groups")),
            **common,
        )
    raise ValueError(f"Unknown RBD type {rbd_type!r}.")


def rbd_to_json(rbd: RBD, **json_kwargs) -> str:
    """Serialise an RBD to a JSON string (kwargs pass to ``json.dumps``)."""
    return json.dumps(rbd_to_dict(rbd), **json_kwargs)


def rbd_from_json(s: str) -> RBD:
    """Reconstruct an RBD from a JSON string."""
    return rbd_from_dict(json.loads(s))
