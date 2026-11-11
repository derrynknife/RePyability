"""A unit's own life in a state: a measured degradation level, or a model
of its own standing in for the node's.

A :class:`~repyability.rbd.node_state.NodeState` says where the unit in
service now is. Its ``age`` conditions the node's life (``R(a + x) /
R(a)``). Two fields replace that life for this unit alone:

- ``level``, the degradation level measured now on a unit whose life is a
  surpyval Wiener or gamma process: its life from now is the first passage
  from the level to the failure threshold, which surpyval gives (``sf``,
  ``qf`` and ``mean`` take the level as ``y0``). Both processes have
  independent increments, so the level says all there is about the life
  left, and the age conditions nothing more (it stays the operating time,
  for an age replacement's due time).
- ``model``, a life of this unit's own, used instead of the node's, at the
  state's age. :func:`checked_override` decides whether a model may stand
  in for a node's, in both kinds of diagram (and for a phase's model of a
  phased mission): what the node could itself have been given as its
  life, and nothing else.

:func:`life_now` gives the life of the unit in service now, which the
analyses condition on in place of the node's.
"""

from typing import Any, Optional, Tuple

import numpy as np

from repyability.rbd._degradation import (
    failure_from,
    is_degradation,
)
from repyability.rbd._model_utils import is_fixed_probability
from repyability.utils.checks import no_distribution


def _repairable_life(model) -> bool:
    """Whether ``model`` is a component of a ``RepairableRBD`` (or a
    ``NonRepairable``), whose life is its ``reliability``."""
    from repyability.non_repairable import NonRepairable

    return isinstance(model, NonRepairable)


def checked_override(node, own_model, override):
    """``override``, checked as the life of ``node`` in place of
    ``own_model`` (the node's model as its diagram holds it: a
    ``NonRepairableRBD``'s node model, or a ``RepairableRBD``'s
    component), and as the diagram takes it (a joint model's prediction
    as its ``JointNode``).

    It must be what the node could itself have been given as its life in
    that kind of diagram, and the node a single unit (not a standby,
    load-sharing or repeated node, nor a nested diagram, which take no
    state). In a ``RepairableRBD`` it is the life of the unit in service
    now only, so a joint model's prediction from the unit's readings is
    taken too (its replacements are new, with the node's life); a
    probability of failing is not, as the unit's life must be one of
    times.

    Raises
    ------
    TypeError
        If ``override`` is not a model (as the diagram's own check says).
    ValueError
        If the node takes no state, or the model is not one the node could
        have been given, saying what to give.
    """
    from repyability.rbd.helper_classes import perfect_class
    from repyability.rbd.joint_node import JointNode, as_node
    from repyability.rbd.non_repairable_rbd import _check_model, _stateable

    label = f"The model given in the state of node {node!r}"
    override = perfect_class(as_node(override))
    if _repairable_life(own_model):
        if not _stateable(own_model.reliability):
            raise ValueError(
                f"Node {node!r} is a {type(own_model.reliability).__name__}, "
                "whose units take no state: it has no unit of its own to "
                "give a model."
            )
        if isinstance(override, JointNode):
            return override
        no_distribution(override, label)
        if is_fixed_probability(override):
            raise ValueError(
                f"{label} is a probability of failing, not a life: the unit "
                "in service of a repaired component lives for a time; give a "
                "distribution of its life."
            )
        from repyability.non_repairable import NonRepairable

        try:
            NonRepairable(override, "instant")
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"{label} cannot be the life of a repaired component: "
                f"{error}"
            ) from None
        if not _stateable(override):
            raise ValueError(
                f"{label} is a {type(override).__name__}, which takes no "
                "state: give the life of the unit itself."
            )
        return override
    if not _stateable(own_model):
        raise ValueError(
            f"Node {node!r} is a {type(own_model).__name__}, which takes no "
            "state: it has no unit of its own to give a model."
        )
    try:
        _check_model(node, override)
    except (TypeError, ValueError) as error:
        raise type(error)(
            f"{label} cannot stand in for its own: {error}"
        ) from None
    if not _stateable(override):
        raise ValueError(
            f"{label} is a {type(override).__name__}, which takes no state: "
            "give the life of the unit itself (a distribution, a "
            "RegressionNode, a JointNode, ...)."
        )
    return override


def checked_level(node, life, level: float) -> None:
    """Raise if ``level`` cannot be the measured level of the unit of
    ``node`` whose life is ``life`` (its own model, or the node's): the
    life must be a degradation process, and the level below its failure
    threshold and, for a gamma process (whose level only rises), at least
    its starting level."""
    if not is_degradation(life):
        raise ValueError(
            f"Node {node!r}: a level is the degradation level measured on "
            "a unit whose life is a surpyval Wiener or gamma degradation "
            f"process, and its life is a {type(life).__name__}. Give its "
            "age alone, or give it a degradation process fitted to its "
            "readings as its model."
        )
    threshold = float(life.threshold)
    if not float(level) < threshold:
        raise ValueError(
            f"Node {node!r}: its level, {float(level):g}, is at or past its "
            f"failure threshold, {threshold:g}, so the unit has failed: "
            "give alive=False."
        )
    from repyability.rbd._degradation import _kinds

    _, gamma = _kinds()
    if isinstance(life, gamma) and float(level) < float(life.y0):
        raise ValueError(
            f"Node {node!r}: its level, {float(level):g}, is below the "
            f"starting level of its gamma process, {float(life.y0):g}, and "
            "a gamma process only rises: check the level, or the "
            "process's y0."
        )


def unit_life(node, own_model, state):
    """The life of the unit of ``node`` in ``state`` as given (its own
    model, checked by :func:`checked_override`, or the node's), with its
    level, if any, checked against it (:func:`checked_level`)."""
    if state.model is None:
        life = own_model
    else:
        life = checked_override(node, own_model, state.model)
    if _repairable_life(life):
        life = life.reliability
    if state.level is not None:
        checked_level(node, life, state.level)
    return life


class FromLevel:
    """A degradation process's life from a measured level: the first
    passage from ``level`` to the threshold, from now. Its functions are
    the process's with ``y0=level``."""

    def __init__(self, process, level: float):
        self.process = process
        self.level = float(level)

    def sf(self, x):
        return self.process.sf(x, y0=self.level)

    def ff(self, x):
        return self.process.ff(x, y0=self.level)

    def df(self, x):
        return self.process.df(x, y0=self.level)

    def hf(self, x):
        return self.process.hf(x, y0=self.level)

    def Hf(self, x):
        return self.process.Hf(x, y0=self.level)

    def qf(self, p):
        return self.process.qf(p, y0=self.level)

    def mean(self) -> float:
        return float(self.process.mean(y0=self.level))

    def random(self, size: int) -> np.ndarray:
        return np.asarray(
            self.qf(np.random.uniform(size=size)), dtype=float
        ).reshape(size)

    def __repr__(self) -> str:
        return f"FromLevel({self.process!r}, level={self.level!r})"


def life_now(node, own_model, state) -> Tuple[Any, Optional[float]]:
    """The life of the unit of ``node`` in service now, in ``state``
    (alive), and its age in that life: the life from the measured level
    (:class:`FromLevel`), already from now (an age of None); a joint
    model's unit, from its prediction's age, also from now; otherwise the
    unit's own model, or the node's, at the state's age."""
    from repyability.rbd.joint_node import JointNode

    life = unit_life(node, own_model, state)
    if state.level is not None:
        return FromLevel(life, state.level), None
    if isinstance(life, JointNode):
        return life, None
    return life, float(state.age)


def life_left(node, own_model, state, u: float) -> float:
    """The life left to the unit of ``node`` in service now, in ``state``
    (with a level or a model of its own), from the uniform ``u``, for a
    simulation's start (see ``_event_loop._started``): from a level,
    surpyval's first-passage quantile from it (``_degradation.
    failure_from``, as a unit replaced on condition draws it); a joint
    model's unit, its quantile of the life left; any other, what is left
    of its life at its age, as a node's own is drawn
    (``_event_loop._aged_life``)."""
    from repyability.rbd._event_loop import _aged_life

    life, age = life_now(node, own_model, state)
    if isinstance(life, FromLevel):
        return failure_from(life.process, life.level, u)
    if age is None:
        return float(np.ravel(life.qf(np.array([u])))[0])
    return _aged_life(life, age, u)


def refuse_in_chain(states: dict) -> None:
    """Raise if a component in ``states`` (by node, those a Markov chain
    of exponential lives starts from) has a life of its own, which the
    chain, whose states are only up or down, cannot carry."""
    for node, state in states.items():
        if getattr(state, "own_life", False):
            raise NotImplementedError(
                f"Component {node!r} starts with a life of its own (a "
                "measured level, or its own model), which the Markov chain "
                "of the components' exponential lives does not take: "
                "simulate it (method='simulate', or availability(state=...))."
            )


def state_to_dict(state) -> dict:
    """A ``NodeState`` as plain data, for a run's settings (see
    ``_runs._state_key``): its fields, but a level or a model only when
    given (so that a state without them reads as it always has), and a
    model in its saved form (``serialisation.serialise_model``)."""
    import dataclasses

    from repyability.rbd.serialisation import serialise_model

    out = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name not in ("level", "model")
    }
    if state.level is not None:
        out["level"] = float(state.level)
    if state.model is not None:
        out["model"] = serialise_model(state.model)
    return out


def state_from_dict(value: dict):
    """The ``NodeState`` :func:`state_to_dict` made ``value`` of."""
    from repyability.rbd.node_state import NodeState
    from repyability.rbd.serialisation import deserialise_model

    value = dict(value)
    if value.get("model") is not None:
        value["model"] = deserialise_model(value["model"])
    return NodeState(**value)
