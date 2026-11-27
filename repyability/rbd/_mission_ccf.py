"""Common-cause groups in a phased mission (see ``PhasedMission``): each
shared cause written into the phases' diagrams as a component of its own.

A mission runs through time, so a group in it needs a life for each of its
causes, which two kinds of model give. A ``Shocks`` model's causes strike at
rates of their own, whatever the members' ages: each is an exponential
component. A model that splits the failure *rate* (``basis="rate"``) gives
each cause a share ``c`` of the members' cumulative hazard ``H``: the cause
has survival ``exp(-c H(t))`` and a member's own failures ``exp(-c_1
H(t))``, the shares of the causes that strike a member and its own summing
to 1, so that it keeps its life. (A model that splits a failure
*probability* has no life to run through the phases, and is refused.)

A cause that has not struck by ``t`` is then a component in series with
every member it fails: here it is drawn after each of them (and after each
node that repeats one), in every phase's diagram that uses the member, one
component ``(name, members it fails)`` in all those places. A cause that
strikes in one phase so fails its members from then on, in every later
phase, as any component's failure does: the mission's decision diagram
takes the cause's segments like a component's, repeated across its
members, and the simulation draws one life per cause.
"""

from typing import Any, Dict, Hashable, List, Tuple

from surpyval import Parametric

from repyability.rbd.ccf import Shocks, checked_groups
from repyability.rbd.helper_classes import PerfectReliability
from repyability.rbd.serialisation import serialise_model


def cause_name(index: int, count: int) -> str:
    """The first item of the names of the causes of the ``index``-th of a
    mission's ``count`` groups: ``"cause"`` for an only group, else
    ``"cause 1"``, ``"cause 2"``, ..."""
    return "cause" if count == 1 else f"cause {index + 1}"


def checked(groups: list, given: Dict[Hashable, Any], phases) -> list:
    """The mission's common-cause ``groups``, checked against its
    components as given (``given``, by name) and its ``phases`` (each a
    ``(name, duration, rbd)``).

    Raises
    ------
    ValueError
        If a group is not a ``CCFGroup``, a member is not a component of
        the mission (or repeats another node), is in two groups, or the
        members of a group that splits the failure rate carry different
        models; or a group splits a failure probability.
    """
    repeats = {
        node: rbd.repeated[node]
        for _, _, rbd in phases
        for node in rbd.repeated
    }

    def check_member(member, group) -> None:
        if member in repeats:
            raise ValueError(
                f"CCF group member {member!r} repeats node "
                f"{repeats[member]!r}: name the node it repeats as the "
                "member, once."
            )
        if member not in given:
            raise ValueError(
                f"CCF group member {member!r} is not a component of the "
                "mission: a node with a life model in one of its phases' "
                "diagrams."
            )

    groups = checked_groups(
        groups,
        check_member,
        lambda member: serialise_model(given[member]),
        lambda group: (
            f"CCF group {list(group.members)} is not symmetric: its "
            "members must carry identical component models."
        ),
        noun="Component",
        shocks=True,
    )
    for group in groups:
        if isinstance(group.model, Shocks):
            continue
        if group.model.basis != "rate":
            raise ValueError(
                f"{group!r} splits its members' probability of failing, "
                "which has no time in it for a mission's phases to run "
                "through: split their failure rate instead (the same "
                "model with basis='rate'), or give the group shocks at a "
                "rate of their own (Shocks)."
            )
    return groups


def _share(model, share: float, group):
    """``model`` with the share ``share`` of its hazard, survival
    ``R(t) ** share``: surpyval's hazard ratio."""
    if not isinstance(model, Parametric):
        raise ValueError(
            f"{group!r} splits its members' failure rate, which a mission "
            "takes through surpyval's hazard ratio, so the members' model "
            "must be a surpyval parametric distribution, got "
            f"{type(model).__name__}. Shocks at a rate of their own "
            "(Shocks) take members of any model."
        )
    try:
        return model.with_params(model.params, hr=share * model.hr)
    except ValueError as error:
        raise ValueError(
            f"{group!r} splits its members' failure rate, which a mission "
            f"takes through surpyval's hazard ratio: {error} Shocks at a "
            "rate of their own (Shocks) take members of any model."
        ) from error


def causes(
    groups: list, given: Dict[Hashable, Any]
) -> Tuple[Dict[Hashable, Any], List[Tuple[Hashable, tuple, Any]]]:
    """What ``groups`` make of the mission's components: each member's
    model of its own failures (a member no cause leaves any to never
    fails on its own, ``PerfectReliability``), and each cause as ``(its
    component's name, the members it fails, its life model)``, in the
    groups' order."""
    import surpyval as surv

    own: Dict[Hashable, Any] = {}
    shared: List[Tuple[Hashable, tuple, Any]] = []
    for index, group in enumerate(groups):
        name = cause_name(index, len(groups))
        members = group.members
        if isinstance(group.model, Shocks):
            for struck, rate in group.model.causes(members):
                life = surv.Exponential.from_params([rate])
                shared.append(((name, struck), struck, life))
            continue
        model = given[members[0]]
        left, split = group.model._causes(members)
        for struck, share in split:
            if share > 0.0:
                struck = tuple(m for m in members if m in struck)
                shared.append(
                    ((name, struck), struck, _share(model, share, group))
                )
        for member in members:
            if left >= 1.0:
                continue  # no shared cause: its life as given
            own[member] = (
                _share(given[member], left, group)
                if left > 0.0
                else PerfectReliability
            )
    return own, shared


def written(phases: list, groups: list, given: Dict[Hashable, Any]) -> list:
    """The mission's ``phases`` (each a ``(name, duration, rbd)``) with
    the causes of ``groups`` written into their diagrams (see the module
    docstring); a diagram that uses no member is kept as it is, and one
    used by several phases is rewritten once.

    Raises
    ------
    ValueError
        If a cause's name is already a node of a diagram.
    """
    own, shared = causes(groups, given)
    made: Dict[int, Any] = {}
    out = []
    for name, duration, rbd in phases:
        if id(rbd) not in made:
            try:
                made[id(rbd)] = _with_causes(rbd, own, shared)
            except ValueError as error:
                raise ValueError(f"Phase {name!r}: {error}") from error
        out.append((name, duration, made[id(rbd)]))
    return out


def _with_causes(rbd, own: Dict[Hashable, Any], shared: list):
    """The diagram ``rbd`` with each member's model that of its own
    failures (``own``) and each of the causes ``shared`` after every
    place of each member it fails that the diagram uses: the first place
    the cause's component, the others repeating it."""
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD

    left_out = {rbd.input_node, rbd.output_node} | rbd._junctions()
    used = [node for node in rbd._components() if node not in left_out]
    struck_by: Dict[Hashable, list] = {}
    for component, struck, life in shared:
        for member in struck:
            if member in used:
                struck_by.setdefault(member, []).append((component, life))
    present = [node for node in used if node in own or node in struck_by]
    if not present:
        return rbd
    given = rbd._init_args
    reliabilities = dict(given["reliabilities"])
    edges = [tuple(edge) for edge in given["edges"]]
    for member in present:
        if member in own:
            reliabilities[member] = own[member]
        places = [member] + [
            node for node, of in rbd.repeated.items() if of == member
        ]
        for place in places:
            after = [edge for edge in edges if edge[0] == place]
            edges = [edge for edge in edges if edge[0] != place]
            last = place
            for component, life in struck_by.get(member, []):
                if component in reliabilities:
                    node = (*component, place)
                    model = component
                else:
                    node, model = component, life
                if node in reliabilities:
                    raise ValueError(
                        f"{node!r} is already a node of the diagram, and "
                        "is the name a common-cause group's shared cause "
                        "takes: rename the node."
                    )
                reliabilities[node] = model
                edges.append((last, node))
                last = node
            edges += [(last, to) for _, to in after]
    return NonRepairableRBD(
        **{**given, "edges": edges, "reliabilities": reliabilities}
    )
