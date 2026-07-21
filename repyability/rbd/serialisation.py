"""(De)serialisation of RBDs and their node models to plain dicts / JSON.

An RBD only existed as Python code; this module round-trips it to a
JSON-friendly structure so it can be saved, loaded, shared and version
controlled (e.g. by the Reliafy app).

Design notes
------------
- Node identity is preserved through JSON. Node names may be ints or strings,
  but JSON object keys are always strings, so the per-node collections
  (reliabilities, components, k) are serialised as *lists of entries*
  (``{"node": n, ...}``) rather than dicts keyed by node.
- The constructor inputs are captured verbatim at construction time and
  serialised, so ``from_dict(rbd.to_dict())`` simply reconstructs the RBD by
  calling its constructor again — faithful even for repeated nodes (whose
  graph is collapsed after construction).
- Node models are serialised structurally: surpyval parametric distributions
  as ``(dist name, params)``; the RePyability wrappers (standby, repeated,
  NonRepairable) recursively; nested RBDs via their own ``to_dict``. Fitted
  non-parametric models are not serialisable (there is no public
  reconstruction API) and raise a clear error.
"""

import json
from typing import Any

import numpy as np

from repyability._version import __version__
from repyability.non_repairable import NonRepairable
from repyability.rbd._model_utils import distribution_name
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


def _params_list(model) -> list:
    return [float(p) for p in np.atleast_1d(model.params)]


def serialise_model(model: Any) -> dict:
    """Serialise a node model to a JSON-friendly dict."""
    if model is PerfectReliability:
        return {"kind": "perfect_reliability"}
    if model is PerfectUnreliability:
        return {"kind": "perfect_unreliability"}
    if isinstance(model, RBD):
        return {"kind": "rbd", "rbd": rbd_to_dict(model)}
    if isinstance(model, StandbyModel):
        return {
            "kind": "standby",
            "reliabilities": [serialise_model(m) for m in model.reliabilities],
            "k": model.k,
            "n_sims": model.n_sims,
            "switching_probability": model.switching_probability,
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
            "n_sims": model.n_sims,
        }
    dist = distribution_name(model)
    if dist is not None:
        return {
            "kind": "parametric",
            "dist": dist,
            "params": _params_list(model),
        }
    raise NotImplementedError(
        f"Cannot serialise a node model of type {type(model).__name__}. "
        "Only surpyval parametric distributions, the RePyability node "
        "wrappers (standby, repeated, NonRepairable), perfect "
        "reliability/unreliability and nested RBDs are supported (fitted "
        "non-parametric models have no reconstruction API)."
    )


def deserialise_model(d: dict) -> Any:
    """Reconstruct a node model from :func:`serialise_model`'s output."""
    import surpyval

    kind = d["kind"]
    if kind == "perfect_reliability":
        return PerfectReliability
    if kind == "perfect_unreliability":
        return PerfectUnreliability
    if kind == "parametric":
        cls = getattr(surpyval, d["dist"])
        return cls.from_params(d["params"])
    if kind == "rbd":
        return rbd_from_dict(d["rbd"])
    if kind == "standby":
        return StandbyModel(
            [deserialise_model(m) for m in d["reliabilities"]],
            k=d["k"],
            n_sims=d.get("n_sims", 10_000),
            switching_probability=d.get("switching_probability", 1.0),
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
            n_sims=d.get("n_sims", 10_000),
        )
    raise ValueError(f"Unknown model kind {kind!r}.")


def _serialise_reliability_value(node, value, all_nodes) -> dict:
    # A repeated node's value is the name of the node it repeats, not a model.
    if value in all_nodes:
        return {"kind": "repeat_of", "node": value}
    return serialise_model(value)


def _deserialise_reliability_value(d: dict) -> Any:
    if d.get("kind") == "repeat_of":
        return d["node"]
    return deserialise_model(d)


def _serialise_component(value) -> dict:
    # RepairableRBD components: a {reliability, repairability} spec, a
    # NonRepairable, or a nested RepairableRBD.
    if isinstance(value, dict):
        return {
            "kind": "component_spec",
            "reliability": serialise_model(value["reliability"]),
            "repairability": serialise_model(value["repairability"]),
        }
    return serialise_model(value)


def _deserialise_component(d: dict) -> Any:
    if d.get("kind") == "component_spec":
        return {
            "reliability": deserialise_model(d["reliability"]),
            "repairability": deserialise_model(d["repairability"]),
        }
    return deserialise_model(d)


def _k_to_list(k):
    return None if not k else [{"node": n, "k": v} for n, v in k.items()]


def _k_from_list(k_list):
    return None if not k_list else {e["node"]: e["k"] for e in k_list}


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
        groups.append(CCFGroup(entry["members"], model))
    return groups


def rbd_to_dict(rbd: RBD) -> dict:
    """Serialise an RBD (NonRepairableRBD or RepairableRBD) to a dict."""
    args = rbd._init_args
    out = {
        "repyability_version": __version__,
        "type": type(rbd).__name__,
        "edges": [list(e) for e in args["edges"]],
        "k": _k_to_list(args["k"]),
        "input_node": args["input_node"],
        "output_node": args["output_node"],
        "on_infeasible_rbd": args["on_infeasible_rbd"],
    }
    if out["type"] == "RepairableRBD":
        out["components"] = [
            {"node": n, "component": _serialise_component(v)}
            for n, v in args["components"].items()
        ]
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
    edges = [tuple(e) for e in d["edges"]]
    common = dict(
        k=_k_from_list(d.get("k")),
        input_node=d.get("input_node"),
        output_node=d.get("output_node"),
        on_infeasible_rbd=d.get("on_infeasible_rbd", "raise"),
    )
    if rbd_type == "RepairableRBD":
        components = {
            e["node"]: _deserialise_component(e["component"])
            for e in d["components"]
        }
        return RepairableRBD(edges, components, **common)
    if rbd_type == "NonRepairableRBD":
        reliabilities = {
            e["node"]: _deserialise_reliability_value(e["model"])
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
