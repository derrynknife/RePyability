"""The unit of a diagram's models: read from their labels, checked to be
one, and carried on the diagram and its results.

A surpyval model can carry the unit its data were in (``model.unit``:
``"hours"``, ``"days"``, ``"km"``, ``"cycles"``), kept as typed. RePyability
converts nothing: a diagram adds and compares its models' times as they
are, so a block in hours beside a block in days gives a wrong answer. A
diagram whose models name different units is built with a warning that
names the blocks and their units; models with no unit say nothing. Labels
are compared with surpyval's ``same_unit``, so ``"h"``, ``"hrs"`` and
``"hours"`` agree. Before surpyval 0.28, which made it public, the labels
are compared as text (spaces aside), so ``"h"`` and ``"hours"`` are then
taken for different units.
"""

from __future__ import annotations

import warnings
from typing import Any, Hashable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import surpyval.utils as _surpyval_utils

from repyability.rbd.results import _ResultMapping
from repyability.utils.wrappers import outside_level

#: surpyval's ``same_unit``, public from 0.28 (None before it).
_surpyval_same_unit = getattr(_surpyval_utils, "same_unit", None)

#: The attributes through which RePyability's node models (standby,
#: repeated, load-sharing, regression and joint nodes, ``NonRepairable``,
#: a Bayesian fit's predictive, a degrading node's stages) hold the
#: models they are built from.
_PARTS = (
    "model",
    "models",
    "fit",
    "reliability",
    "reliabilities",
    "time_to_replace",
    "stages",
)

# Deep enough for any nesting of node models; a guard, not a limit.
_DEPTH = 12


def same_unit(a: Optional[str], b: Optional[str]) -> bool:
    """Whether the unit labels ``a`` and ``b`` name one unit: by
    surpyval's ``same_unit`` (pint reads them, so ``"h"`` and ``"hours"``
    agree), or, before surpyval 0.28, as text with runs of spaces made one.
    None (unknown) is the same only as None."""
    if _surpyval_same_unit is not None:
        return bool(_surpyval_same_unit(a, b))
    if a is None or b is None:
        return a is None and b is None
    return " ".join(a.split()) == " ".join(b.split())


def model_units(model: Any, _depth: int = 0) -> Iterator[str]:
    """The unit labels of ``model`` and of the models it is built from: a
    surpyval model's (or a nested diagram's) ``unit``, and those of a node
    model's parts or a component spec's values (its lives, its repairs).
    A model with no unit gives none."""
    if (
        _depth > _DEPTH
        or model is None
        or isinstance(model, (str, bytes, bool, int, float, np.number, type))
    ):
        return
    if hasattr(model, "unit"):
        # A surpyval model, or a nested diagram (which has checked its own
        # models): a leaf, its unit known or not.
        unit = model.unit
        if isinstance(unit, str):
            yield unit
        return
    if isinstance(model, Mapping):  # a RepairableRBD component spec
        for value in model.values():
            yield from model_units(value, _depth + 1)
        return
    if isinstance(model, (list, tuple)):
        for value in model:
            yield from model_units(value, _depth + 1)
        return
    for name in _PARTS:
        part = getattr(model, name, None)
        if part is not None and not callable(part):
            yield from model_units(part, _depth + 1)


def unit_groups(
    models: Mapping[Hashable, Any],
) -> List[Tuple[str, List[Hashable]]]:
    """The units ``models`` (by node) are in, each with its nodes, in the
    order first met: each unit by its first label (``"hrs"`` and
    ``"hours"`` are one unit). A node whose models are in two units (lives
    in hours, repairs in days) is under both."""
    groups: List[Tuple[str, List[Hashable]]] = []
    for node, model in models.items():
        for unit in dict.fromkeys(model_units(model)):
            for label, nodes in groups:
                if same_unit(label, unit):
                    if node not in nodes:
                        nodes.append(node)
                    break
            else:
                groups.append((unit, [node]))
    return groups


def mixed_units_message(
    groups: List[Tuple[str, List[Hashable]]], what: str = "diagram"
) -> str:
    """The warning for a ``what`` whose models are in the units of
    ``groups`` (two or more)."""
    listed = "; ".join(
        f"{label!r} for {', '.join(repr(n) for n in nodes)}"
        for label, nodes in groups
    )
    first, second = groups[0][0], groups[1][0]
    return (
        f"The {what}'s models are in different units: {listed}. "
        "RePyability converts nothing: it takes every time as it is, so it "
        f"would mix {first!r} with {second!r} and its answers would be "
        "wrong. Give every model in one unit: convert the times and refit "
        "(or rebuild) the others, with unit= naming that unit."
        + (
            ""
            if _surpyval_same_unit is not None
            else " (surpyval before 0.28 compares the labels as text; if "
            "these name one unit written two ways, such as 'h' and "
            "'hours', upgrade surpyval or write them the same.)"
        )
    )


def diagram_unit(
    models: Mapping[Hashable, Any], what: str = "diagram"
) -> Optional[str]:
    """The one unit ``models`` (by node) are in, by its first label; None
    when none has a unit, or (with a warning naming the nodes and their
    units) when they are in different units. Models with no unit are left
    out: they say nothing about it."""
    groups = unit_groups(models)
    if not groups:
        return None
    if len(groups) > 1:
        warnings.warn(
            mixed_units_message(groups, what),
            UserWarning,
            stacklevel=outside_level(),
        )
        return None
    return groups[0][0]


def stamp_unit(result: Any, unit: Optional[str]) -> Any:
    """``result``, carrying ``unit`` as its ``model_unit`` if it is a
    result (a ``_ResultMapping``) and ``unit`` is not None; anything else
    is returned as it is."""
    if unit is not None and isinstance(result, _ResultMapping):
        # The results are frozen dataclasses; the unit is no field of
        # theirs (it does not change what they compare equal to).
        object.__setattr__(result, "_model_unit", unit)
    return result
