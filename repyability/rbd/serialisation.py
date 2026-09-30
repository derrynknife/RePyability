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
- Node models are serialised structurally: surpyval models (parametric and
  non-parametric) in surpyval's own format, ``model.to_dict()``, loaded with
  ``surpyval.from_dict``, so everything surpyval keeps (an offset, ``p``,
  ``f0``, a fit's covariance) round-trips; the RePyability wrappers
  (standby, degrading, repeated, NonRepairable) recursively; nested RBDs via
  their own
  ``to_dict``. Files from before 0.10.0, which saved a parametric model as
  ``(dist name, params, extras)``, still load.
"""

import json
from typing import Any

from surpyval import NonParametric

from repyability._version import __version__
from repyability.non_repairable import NonRepairable
from repyability.rbd._model_utils import distribution_name
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


def serialise_model(model: Any) -> dict:
    """Serialise a node model to a JSON-friendly dict."""
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
            "mc_samples": model.mc_samples,
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
            "mc_samples": model.mc_samples,
        }
    if distribution_name(model) is not None or isinstance(
        model, NonParametric
    ):
        # surpyval's own format: everything surpyval keeps (an offset, p,
        # f0, a fit's covariance) round-trips, whatever it adds later.
        return {"kind": "surpyval", "model": model.to_dict()}
    raise NotImplementedError(
        f"Cannot serialise a node model of type {type(model).__name__}. "
        "Only surpyval models (parametric and non-parametric), the "
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
        # any offset, p and f0 ("extras", since 0.9.0).
        cls = getattr(surpyval, d["dist"])
        return cls.from_params(d["params"], **d.get("extras", {}))
    if kind == "rbd":
        return rbd_from_dict(d["rbd"])
    if kind == "standby":
        return StandbyModel(
            [deserialise_model(m) for m in d["reliabilities"]],
            k=d["k"],
            mc_samples=d.get("mc_samples", d.get("n_sims", 10_000)),
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
            mc_samples=d.get("mc_samples", d.get("n_sims", 10_000)),
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
        for key in ("preventive", "inspection"):
            if value.get(key) is not None:
                out[key] = _serialise_schedule(value[key])
        return out
    return serialise_model(value)


def _serialise_schedule(spec: dict) -> dict:
    # A component's preventive-maintenance or inspection schedule: the
    # interval (and policy) as they are, the duration ("instant" or a model)
    # and the cost (a number or a distribution).
    out: dict[str, Any] = {"interval": float(spec["interval"])}
    if "policy" in spec:
        out["policy"] = spec["policy"]
    duration = spec.get("duration", "instant")
    out["duration"] = (
        "instant" if isinstance(duration, str) else serialise_model(duration)
    )
    cost = spec.get("cost")
    if hasattr(cost, "qf"):
        out["cost"] = serialise_model(cost)
    elif cost:
        out["cost"] = float(cost)
    return out


def _deserialise_schedule(d: dict) -> dict:
    out = dict(d)
    if isinstance(d["duration"], dict):
        out["duration"] = deserialise_model(d["duration"])
    if isinstance(d.get("cost"), dict):
        out["cost"] = deserialise_model(d["cost"])
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
        for key in ("preventive", "inspection"):
            if key in d:
                out[key] = _deserialise_schedule(d[key])
        return out
    return deserialise_model(d)


def _k_to_list(k):
    return None if not k else [{"node": n, "k": v} for n, v in k.items()]


def _k_from_list(k_list):
    if not k_list:
        return None
    return {_node_name(e["node"]): e["k"] for e in k_list}


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
        if kind == "beta_factor":
            model: object = BetaFactor(model_dict["beta"])
        elif kind == "mgl":
            model = MGL(*model_dict["letters"])
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
        return RepairableRBD(
            edges,
            components,
            downtime_cost_rate=d.get("downtime_cost_rate", 0.0),
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
