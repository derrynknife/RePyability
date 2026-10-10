"""Parameter uncertainty in a repairable diagram (#200): the spread of its
availability (in the long run, at times or over a mission) and of its cost
rate over plausible models of its components, and each uncertain input's
share of that spread (its uncertainty importance, vega).

A component of a ``RepairableRBD`` has several models (its roles): its life
(``"reliability"``), its repair (``"repairability"``), and the durations of
its preventive maintenance and of its tests (``"preventive.duration"``,
``"inspection.duration"``), named as ``parameter_sensitivity``'s levers are.
Each may be uncertain, and is drawn as ``NonRepairableRBD.sf_uncertainty``
draws a node's model (see ``uncertainty``): from a fit's parameter
covariance, distributions over its parameters, or a list of models. A
common-cause group's model may be uncertain too. An input is a node, a
population of nodes sharing a model (a tuple), or a common-cause group;
a node may come under several inputs, a role under one (#214): a fleet's
life fit, shared by pumps whose repairs were recorded apart, is one input
of the pumps' lives, and each pump's repair one of its own.

Each draw is the diagram rebuilt from its constructor's arguments with the
drawn models, and its value worked out as for the diagram itself (exactly,
or numerically): one evaluation a draw, cheap in the long run, the
components' curves' cost over time.

The uncertainty importance takes the delta method (``parameter_
sensitivity``'s derivatives, against each input's parameter covariance) or
the Sobol indices from draws (Jansen's estimators), as
``NonRepairableRBD.uncertainty_importance`` does.
"""

import difflib
import warnings
from collections.abc import Mapping
from typing import Any, Dict, Hashable, List, NamedTuple, Optional, Tuple

import numpy as np

from repyability.utils.checks import seed as check_seed
from repyability.utils.wrappers import outside_level

from .uncertainty import (
    FIT,
    Counter,
    SobolPoints,
    check_sampling,
    draw_ccf_models,
    draw_models,
    is_fit,
    sobol_table,
    varied_ccf_parameters,
    varied_parameters,
)

#: A component's models that can be uncertain, by the names of
#: ``parameter_sensitivity``'s levers.
ROLES = (
    "reliability",
    "repairability",
    "preventive.duration",
    "inspection.duration",
)
#: The quantities, by the name of their method less ``_uncertainty``.
QUANTITIES = (
    "mean_availability",
    "point_availability",
    "mission_availability",
    "expected_cost_rate",
)
#: The quantities' other names: ``parameter_sensitivity``'s (#232).
QUANTITY_NAMES = {"cost_rate": "expected_cost_rate"}


def model_of(spec: dict, role: str):
    """A component's model in ``role`` (see ``ROLES``), from its spec dict;
    None where it has none (a repair in no time, say)."""
    if role in ("reliability", "repairability"):
        model = spec.get(role)
    else:
        options = spec.get(role.split(".")[0])
        model = options.get("duration") if isinstance(options, dict) else None
    return model if hasattr(model, "sf") else None


def with_model(spec: dict, role: str, model) -> dict:
    """A component's spec dict with its model in ``role`` replaced."""
    if role in ("reliability", "repairability"):
        return {**spec, role: model}
    key = role.split(".")[0]
    return {**spec, key: {**spec[key], "duration": model}}


class Source(NamedTuple):
    """An uncertain input of nodes: the key it was given under, its nodes,
    and the uncertainty of each of their uncertain models, by role."""

    key: Hashable
    members: tuple
    specs: Dict[str, Any]


def _spec(rbd, node) -> Optional[dict]:
    """A component's spec dict (see ``_sensitivity._as_spec``), or None
    for a node without models of its own (a nested RBD, a junction)."""
    from ._sensitivity import _as_spec

    return _as_spec(rbd._init_args["components"][node])


def _models(rbd, node) -> Dict[str, Any]:
    spec = _spec(rbd, node)
    return {} if spec is None else {r: model_of(spec, r) for r in ROLES}


def _label(members: tuple) -> str:
    return (
        f"Node {members[0]!r}"
        if len(members) == 1
        else f"Nodes {list(members)!r}"
    )


def _roles(spec: Any, models: dict, label: str) -> Dict[str, Any]:
    """The uncertainty of each of a component's uncertain models: ``spec``
    is ``"fit"`` (every one of its models that is a fit with a parameter
    covariance), ``{role: uncertainty}`` (a mapping whose keys are all
    roles), or one uncertainty, of its life."""
    if isinstance(spec, str) and spec == FIT:
        out = {
            role: FIT
            for role, model in models.items()
            if model is not None and is_fit(model)
        }
        if not out:
            raise ValueError(
                f"{label}: none of its models (its life, repair, or "
                "maintenance or test duration) is a surpyval fit with a "
                "parameter covariance, which 'fit' draws from. Give "
                "distributions over a model's parameters, or a list of "
                "models, by role: {'reliability': ..., 'repairability': "
                "...}."
            )
        return out
    if isinstance(spec, Mapping) and spec and all(k in ROLES for k in spec):
        for role in spec:
            if models.get(role) is None:
                raise ValueError(
                    f"{label} has no {role} model to draw (it is not a "
                    "model, or it has none)."
                )
        return dict(spec)
    if isinstance(spec, Mapping) and spec:
        _require_parameters_or_roles(spec, models.get("reliability"), label)
    if models.get("reliability") is None:
        raise ValueError(f"{label} has no life model to draw.")
    return {"reliability": spec}


def _require_parameters_or_roles(spec: Mapping, life, label: str) -> None:
    """Raise if ``spec``, distributions by the life's parameter names or
    uncertainties by role, has a key that is neither one of the life's
    parameters nor a role: saying which it is likelier meant, and what each
    may be (#232)."""
    dist = getattr(life, "dist", None)
    names = [str(name) for name in getattr(dist, "parameter_names", ())]
    roles = [key for key in spec if key in ROLES]
    unknown = [key for key in spec if key not in ROLES and key not in names]
    if roles and len(roles) < len(spec):
        raise ValueError(
            f"{label}: {sorted(map(str, set(spec) - set(roles)))} and "
            f"{roles} mix roles and parameters: give uncertainties by role "
            f"({', '.join(ROLES)}), or distributions over the life's "
            f"parameters ({', '.join(names) or 'none'}), not both."
        )
    if unknown:
        close = [
            match
            for key in unknown
            for match in difflib.get_close_matches(
                str(key), list(ROLES) + names, n=1
            )
        ]
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"{label}: {sorted(map(str, unknown))} are neither roles "
            f"({', '.join(ROLES)}) nor parameters of its life's model "
            f"({', '.join(names) or 'none'}).{hint}"
        )


def fitted(rbd) -> Dict[Hashable, Dict[str, str]]:
    """The uncertainty drawn when none is given: ``"fit"`` for every one
    of the components' models that is a surpyval fit with a parameter
    covariance, drawn once for all the nodes holding that fitted object in
    that role, and for one common-cause group's members together:
    ``{node or tuple of nodes: {role: "fit"}}``, by first node and role, a
    node under as many keys as it has populations."""
    order = {node: i for i, node in enumerate(rbd.components)}
    out: Dict[tuple, Dict[str, str]] = {}
    for role in ROLES:
        parent: Dict[Hashable, Hashable] = {}

        def root(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        first: Dict[int, Hashable] = {}
        for node in rbd.components:
            model = _models(rbd, node).get(role)
            if model is not None and is_fit(model):
                parent[node] = node
                parent[root(node)] = root(first.setdefault(id(model), node))
        for group in rbd.ccf_groups:
            inside = [m for m in group.members if m in parent]
            for member in inside[1:]:
                parent[root(member)] = root(inside[0])
        together: Dict[Hashable, list] = {}
        for node in parent:
            together.setdefault(root(node), []).append(node)
        for nodes in together.values():
            out.setdefault(tuple(nodes), {})[role] = FIT
    # By their first node, then role: a node's own inputs where it comes.
    return {
        nodes[0] if len(nodes) == 1 else nodes: roles
        for nodes, roles in sorted(
            out.items(),
            key=lambda item: (
                order[item[0][0]],
                ROLES.index(next(iter(item[1]))),
            ),
        )
    }


def half_named(label: str, others: list, what: str, together) -> None:
    """Warn that an input is drawn without ``others``, which hold the same
    fitted ``what`` and keep it as it is in every draw."""
    names = ", ".join(repr(n) for n in others)
    warnings.warn(
        f"{label} is drawn without {names}, which hold the same {what} "
        "object and keep it as fitted in every draw: if they are one "
        "population, its uncertainty is understated. Give them together, "
        f"{together!r}, to draw it once for all of them.",
        UserWarning,
        stacklevel=outside_level(),
    )


def sources(
    rbd, uncertainty
) -> Tuple[List[Source], Dict[int, Tuple[Any, Any]]]:
    """The uncertain inputs, checked: the nodes' (see ``Source``), and for
    each common-cause group whose model is uncertain, by its index, its key
    and its uncertainty. By default, every fitted model (``fitted``). A
    node may come under several inputs, a role of it under one; given, an
    input drawn without other nodes holding the same model object, which
    are left as they are, is warned about."""
    from .non_repairable_rbd import NonRepairableRBD

    given = uncertainty is not None
    if uncertainty is None:
        uncertainty = fitted(rbd)
        if not uncertainty:
            raise ValueError(
                "No component's model is a surpyval fit with a parameter "
                "covariance, which the uncertainty is drawn from by "
                "default: give the uncertain nodes, uncertainty={node: "
                "'fit'} or {node: {'repairability': {parameter: "
                "distribution}}}, for example."
            )
    if not uncertainty:
        raise ValueError(
            "Give the uncertain nodes: uncertainty={node: 'fit'}, for "
            "example."
        )
    out: List[Source] = []
    groups: Dict[int, Tuple[Any, Any]] = {}
    # Each node's roles given so far.
    seen: set = set()
    for key, spec in uncertainty.items():
        index = next(
            (i for i, g in enumerate(rbd.ccf_groups) if key is g), None
        )
        if index is not None:
            groups[index] = (key, spec)
            continue
        members = (
            key
            if isinstance(key, tuple) and key not in rbd.components
            else (key,)
        )
        for node in members:
            if node not in rbd.components:
                raise ValueError(
                    f"{node!r} in uncertainty is not a component node."
                )
            if _spec(rbd, node) is None:
                raise ValueError(
                    f"Node {node!r} has no models of its own to draw (a "
                    "nested RBD or a junction): give its own components' "
                    "uncertainty to the nested RBD."
                )
        label = _label(members)
        models = _models(rbd, members[0])
        roles = _roles(spec, models, label)
        for node in members:
            for role in roles:
                if (node, role) in seen:
                    raise ValueError(
                        f"The {role} of node {node!r} is given an "
                        "uncertainty twice."
                    )
                seen.add((node, role))
        for node in members[1:]:
            theirs = _models(rbd, node)
            for role in roles:
                if not NonRepairableRBD._same_model(
                    theirs.get(role), models[role]
                ):
                    raise ValueError(
                        f"Nodes {list(members)!r} share their uncertainty, "
                        f"so they must have the same {role} model."
                    )
        out.append(Source(key, tuple(members), roles))
    for group in rbd.ccf_groups:
        for role in ROLES:
            holders = [
                s.members
                for s in out
                if role in s.specs and set(s.members) & set(group.members)
            ]
            if holders and (
                len(holders) > 1 or not set(group.members) <= set(holders[0])
            ):
                raise ValueError(
                    f"Nodes {list(group.members)!r} are a common-cause "
                    "group, whose members carry one model: give their "
                    "uncertainty together, in one tuple of nodes (e.g. "
                    f"{{{tuple(group.members)!r}: 'fit'}})."
                )
    if given:
        for item in out:
            models = _models(rbd, item.members[0])
            for role in item.specs:
                others = [
                    node
                    for node in rbd.components
                    if node not in item.members
                    and (node, role) not in seen
                    and models[role] is not None
                    and _models(rbd, node).get(role) is models[role]
                ]
                if others:
                    who = (
                        f"node {item.members[0]!r}"
                        if len(item.members) == 1
                        else f"nodes {list(item.members)!r}"
                    )
                    half_named(
                        f"The {role} of {who}",
                        others,
                        f"{role} model",
                        {tuple(item.members) + tuple(others): {role: FIT}},
                    )
    return out, groups


def draws(
    rbd,
    inputs: List[Source],
    groups: Dict[int, Tuple[Any, Any]],
    n: int,
    seed,
    sampling: str = "random",
    paired: bool = False,
) -> Tuple[List[Dict[str, list]], list]:
    """``n`` plausible models of each input's uncertain roles (one dict of
    lists by role, for each input), and of each common-cause group's
    model (None for a certain one); ``paired``, two independent sets one
    after the other. With ``sampling="sobol"``, from a scrambled Sobol
    sequence's points, a pair's sets from dimensions of their own (as
    ``NonRepairableRBD._uncertain_draws``)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n_draws must be an integer, got {n!r}.")
    if n < 1:
        raise ValueError(f"n_draws must be at least 1, got {n}.")
    check_sampling(sampling)
    rng = np.random.default_rng(check_seed(seed))

    def draw(count: int, source) -> Tuple[List[Dict[str, list]], list]:
        drawn = []
        for item in inputs:
            models = _models(rbd, item.members[0])
            label = _label(item.members)
            drawn.append(
                {
                    role: draw_models(
                        models[role],
                        spec,
                        count,
                        source,
                        f"The {role} of {label[0].lower()}{label[1:]}",
                    )
                    for role, spec in item.specs.items()
                }
            )
        # The common-cause models after the nodes', as in a
        # NonRepairableRBD.
        drawn_groups = [
            (
                draw_ccf_models(group, groups[i][1], count, source)
                if i in groups
                else None
            )
            for i, group in enumerate(rbd.ccf_groups)
        ]
        return drawn, drawn_groups

    if sampling == "random":
        return draw(2 * n if paired else n, rng)
    counter = Counter()
    draw(1, counter)
    width = counter.dimensions
    table = sobol_table(n, 2 * width if paired else width, rng)
    first = draw(n, SobolPoints(table[:, :width]))
    if not paired:
        return first
    second = draw(n, SobolPoints(table[:, width:]))
    return (
        [
            {role: a[role] + b[role] for role in a}
            for a, b in zip(first[0], second[0])
        ],
        [None if a is None else a + b for a, b in zip(first[1], second[1])],
    )


def build(rbd, inputs, drawn, drawn_groups, i: int):
    """The diagram with the ``i``-th draw's models."""
    from .ccf import CCFGroup

    args = rbd._init_args
    components = dict(args["components"])
    # A node under several inputs takes each one's roles (#214).
    respecified: Dict[Hashable, Any] = {}
    for item, by_role in zip(inputs, drawn):
        for node in item.members:
            # A node with no models of its own is refused by ``sources``.
            spec: Any = respecified.get(node)
            if spec is None:
                spec = _spec(rbd, node)
            for role, models in by_role.items():
                spec = with_model(spec, role, models[i])
            respecified[node] = spec
    components.update(respecified)
    changed: Dict[str, Any] = {"components": components}
    if any(models is not None for models in drawn_groups):
        changed["ccf_groups"] = [
            group if models is None else CCFGroup(group.members, models[i])
            for group, models in zip(rbd.ccf_groups, drawn_groups)
        ]
    return type(rbd)(**{**args, **changed})


def value(rbd, of: str, x, state) -> np.ndarray:
    """The quantity ``of`` of a diagram, as a flat array (one value, or one
    for each of ``x``)."""
    if of == "mean_availability":
        out: Any = rbd.mean_availability()
    elif of == "expected_cost_rate":
        out = rbd.expected_cost_rate()
    elif of == "point_availability":
        out = rbd.point_availability(x, state=state)
    else:
        out = rbd.mission_availability(x, state=state)
    return np.atleast_1d(np.asarray(out, dtype=float)).ravel()


def samples(
    rbd, of: str, x, state, inputs, drawn, drawn_groups, n: int
) -> np.ndarray:
    """The quantity for each of ``n`` draws (a row each)."""
    return np.vstack(
        [
            value(build(rbd, inputs, drawn, drawn_groups, i), of, x, state)
            for i in range(n)
        ]
    )


def check(rbd, of: str, x, state) -> None:
    """Raise for a quantity, times or state the quantity does not take."""
    if of not in QUANTITIES:
        raise ValueError(f"of must be one of {list(QUANTITIES)}, got {of!r}.")
    if of in ("point_availability", "mission_availability"):
        if x is None:
            raise ValueError(
                "x is required: the times (or the missions' lengths) at "
                "which the availability is uncertain."
            )
        return
    if x is not None:
        raise ValueError(f"x is not taken for of={of!r}, a long-run value.")
    if state is not None:
        raise ValueError(
            f"state is where the components start from, which a long-run "
            f"value (of={of!r}) does not depend on."
        )


def _sensitivities(rbd, of: str, x, state, rel_step) -> dict:
    """``parameter_sensitivity``'s derivatives of the quantity ``of``."""
    kwargs: Dict[str, Any] = {"rel_step": rel_step}
    if of == "point_availability":
        kwargs.update(x=x, state=state)
    elif of == "mission_availability":
        if np.ndim(x) == 0:
            kwargs.update(window=x, state=state)
        else:
            # A sensitivity for each mission's length (#226), stacked.
            each = [
                rbd.parameter_sensitivity(
                    rel_step=rel_step, window=float(w), state=state
                )
                for w in np.asarray(x, dtype=float).ravel()
            ]
            return {
                key: {
                    lever: np.array(
                        [float(np.ravel(one[key][lever])[0]) for one in each]
                    )
                    for lever in levers
                }
                for key, levers in each[0].items()
            }
    if of == "expected_cost_rate":
        kwargs["of"] = "cost_rate"
    return rbd.parameter_sensitivity(**kwargs)


def delta(rbd, of: str, x, state, inputs, groups, rel_step) -> list:
    """Each input's part of the quantity's variance by the delta method:
    ``g^T Sigma g`` over its uncertain roles, ``g`` the quantity's
    derivatives in a role's parameters (``parameter_sensitivity``'s levers,
    summed over a population's nodes: they move together) and ``Sigma``
    their covariance (see ``uncertainty.varied_parameters``)."""
    from ._model_utils import parametric_spec

    derivatives = _sensitivities(rbd, of, x, state, rel_step)
    size = int(np.size(x)) if x is not None else 1

    def gradient(keys, names) -> np.ndarray:
        out = np.zeros((len(names), size))
        for k in keys:
            levers = derivatives.get(k, {})
            for j, name in enumerate(names):
                if name not in levers:
                    raise ValueError(
                        f"No derivative of the {of} in {name!r} of {k!r}: "
                        "its model's parameters are not levers of "
                        "parameter_sensitivity."
                    )
                out[j] += np.asarray(levers[name], float).ravel()
        return out

    members = {m for g in rbd.ccf_groups for m in g.members}
    parts = []
    for item in inputs:
        models = _models(rbd, item.members[0])
        label = _label(item.members)
        part = np.zeros(size)
        # A common-cause group's members' levers are under the group's
        # tuple, moved together.
        if set(item.members) & members:
            keys = [
                tuple(g.members)
                for g in rbd.ccf_groups
                if set(g.members) <= set(item.members)
            ] + [m for m in item.members if m not in members]
        else:
            keys = list(item.members)
        for role, spec in item.specs.items():
            positions, _, covariance, _ = varied_parameters(
                models[role],
                spec,
                f"The {role} of {label[0].lower()}{label[1:]}",
            )
            spec_of = parametric_spec(models[role])
            assert spec_of is not None  # a varied model is parametric
            names = spec_of.names
            g = gradient(keys, [f"{role}.{names[p]}" for p in positions])
            part = part + np.einsum("it,ij,jt->t", g, covariance, g)
        parts.append(part)
    for i in sorted(groups):
        group = rbd.ccf_groups[i]
        names, _, covariance, _ = varied_ccf_parameters(group, groups[i][1])
        g = gradient([tuple(group.members)], [f"ccf_{n}" for n in names])
        parts.append(np.einsum("it,ij,jt->t", g, covariance, g))
    return parts


def sobol(
    rbd, of, x, state, inputs, groups, n: int, seed, sampling: str
) -> Tuple[list, list, np.ndarray]:
    """Each input's first-order and total Sobol indices (Jansen's
    estimators), and the quantity's variance, from two independent sets of
    ``n`` draws and one more set for each input with it taken from the
    second, as ``NonRepairableRBD._sobol_indices``."""
    drawn, drawn_groups = draws(rbd, inputs, groups, n, seed, sampling, True)

    def half(models, second: bool):
        return (
            None if models is None else (models[n:] if second else models[:n])
        )

    a = [{r: half(m, False) for r, m in by.items()} for by in drawn]
    b = [{r: half(m, True) for r, m in by.items()} for by in drawn]
    a_groups = [half(m, False) for m in drawn_groups]
    b_groups = [half(m, True) for m in drawn_groups]

    def quantity(models, models_groups) -> np.ndarray:
        return samples(rbd, of, x, state, inputs, models, models_groups, n)

    y_a, y_b = quantity(a, a_groups), quantity(b, b_groups)
    variance = np.var(np.concatenate([y_a, y_b]), axis=0, ddof=1)
    first, total = [], []
    which = [("nodes", k) for k in range(len(inputs))] + [
        ("group", i) for i in sorted(groups)
    ]
    for kind, k in which:
        mixed, mixed_groups = list(a), list(a_groups)
        if kind == "nodes":
            mixed[k] = b[k]
        else:
            mixed_groups[k] = b_groups[k]
        y_ab = quantity(mixed, mixed_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            first.append(
                np.where(
                    variance > 0.0,
                    1.0 - 0.5 * np.mean((y_b - y_ab) ** 2, axis=0) / variance,
                    np.nan,
                )
            )
            total.append(
                np.where(
                    variance > 0.0,
                    0.5 * np.mean((y_a - y_ab) ** 2, axis=0) / variance,
                    np.nan,
                )
            )
    return first, total, variance
