"""A fault tree as an Open-PSA Model Exchange Format (MEF) document.

The MEF (version 2.0d) holds static fault trees: gates ``or``, ``and``,
``atleast`` (k-out-of-n) and ``not``, basic events with a probability or
a time-dependent expression of the mission time (``exponential``,
``Weibull``), and common-cause groups (here the beta-factor model). It has
no dynamic gates, so a tree with PAND, SPARE or SEQ gates is refused; an
FDEP gate is exported as its static equivalent (each event it forces an OR
of the event and its triggers), which is exact.

Names in the MEF are identifiers (letters, digits, ``-`` and ``_``,
starting with a letter): a name that is not one is made one, and the name
as given is kept as the element's ``label``.
"""

import re
from typing import Dict, Hashable, List, NamedTuple, Optional
from xml.sax.saxutils import quoteattr

import numpy as np


class _Group(NamedTuple):
    """A common-cause group's place among the tree's, to name it by."""

    place: int


def _identifiers(names: List[Hashable]) -> Dict[Hashable, str]:
    """An MEF identifier for each name, unique (case-insensitively, as
    some tools compare them)."""
    out: Dict[Hashable, str] = {}
    taken: set = set()
    for name in names:
        text = (
            f"ccf-group-{name.place + 1}"
            if isinstance(name, _Group)
            else str(name)
        )
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "x"
        if not base[0].isalpha():
            base = "x_" + base
        candidate, n = base, 1
        while candidate.lower() in taken:
            n += 1
            candidate = f"{base}-{n}"
        taken.add(candidate.lower())
        out[name] = candidate
    return out


def _number(x: float) -> str:
    return quoteattr(repr(float(x)))


def _expression(event: Hashable, model, t: Optional[float]) -> str:
    """The MEF expression of an event's probability."""
    from surpyval import Exponential, Weibull

    from repyability.rbd._model_utils import parametric_spec

    if isinstance(model, float):
        return f"<float value={_number(model)}/>"
    if t is not None:
        value = float(np.ravel(model.ff(t))[0])
        return f"<float value={_number(value)}/>"
    spec = parametric_spec(model)
    if spec is not None and spec.names == ["p"]:
        return f"<float value={_number(spec.params[0])}/>"
    if spec is not None and spec.cls is Exponential and not spec.extras:
        return (
            f"<exponential><float value={_number(spec.params[0])}/>"
            "<system-mission-time/></exponential>"
        )
    if (
        spec is not None
        and spec.cls is Weibull
        and set(spec.extras) <= {"gamma"}
    ):
        alpha, beta = spec.params
        t0 = spec.extras.get("gamma", 0.0)
        return (
            f"<Weibull><float value={_number(alpha)}/>"
            f"<float value={_number(beta)}/><float value={_number(t0)}/>"
            "<system-mission-time/></Weibull>"
        )
    raise NotImplementedError(
        f"The model of event {event!r} has no Open-PSA expression (the "
        "format writes probabilities, and exponential and Weibull lives "
        "over the mission time): pass t to export every event's "
        "probability at that time."
    )


def to_open_psa(tree, t: Optional[float], name: str) -> str:
    """The MEF document of ``tree`` (see ``FaultTree.to_open_psa``)."""
    from repyability.fault_tree import Forced
    from repyability.rbd.ccf import BetaFactor

    if tree.is_dynamic:
        kinds = sorted(
            {g.kind.upper() for g in tree._gates.values()}
            & {"PAND", "SPARE", "SEQ"}
        )
        raise NotImplementedError(
            "The Open-PSA Model Exchange Format has no dynamic gates, and "
            f"this tree has {', '.join(kinds)} gate(s): it cannot be "
            "exported. (An FDEP gate alone is, as its static equivalent.)"
        )
    members = {}
    for i, group in enumerate(tree.ccf_groups):
        model = group.model
        if not isinstance(model, BetaFactor) or model.basis != "probability":
            raise NotImplementedError(
                f"Common-cause group {list(group.members)}: only a "
                "BetaFactor that splits a probability (the default basis) "
                f"is exported, as the format's beta-factor model; got "
                f"{model!r}."
            )
        for m in group.members:
            members[m] = i
    gates = tree._gates
    groups = [_Group(i) for i in range(len(tree.ccf_groups))]
    ids = _identifiers([name] + list(gates) + list(tree.events) + groups)
    tree_id = ids[name]

    def label(x: Hashable, extra: str = "") -> str:
        text = extra or (str(x) if ids[x] != str(x) else "")
        return f"<label>{_escape(text)}</label>" if text else ""

    def reference(x: Hashable) -> str:
        kind = "gate" if x in gates else "basic-event"
        return f"<{kind} name={quoteattr(ids[x])}/>"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<opsa-mef>",
        f"  <define-fault-tree name={quoteattr(tree_id)}>",
    ]
    for g, gate in gates.items():
        inputs = "".join(reference(x) for x in gate.inputs)
        n = len(gate.inputs)
        if gate.kind == "not":
            formula = f"<not>{inputs}</not>"
        elif n == 1:
            formula = inputs
        elif gate.k == 1:
            formula = f"<or>{inputs}</or>"
        elif gate.k == n:
            formula = f"<and>{inputs}</and>"
        else:
            formula = f'<atleast min="{gate.k}">{inputs}</atleast>'
        note = (
            f"{g.event}, or a functional dependency's trigger forcing it"
            if isinstance(g, Forced)
            else ""
        )
        lines.append(
            f"    <define-gate name={quoteattr(ids[g])}>"
            f"{label(g, note)}{formula}</define-gate>"
        )
    for i, group in enumerate(tree.ccf_groups):
        first = group.members[0]
        refs = "".join(
            f"<basic-event name={quoteattr(ids[m])}/>" for m in group.members
        )
        lines.append(
            f"    <define-CCF-group name={quoteattr(ids[groups[i]])} "
            'model="beta-factor">'
            f"<members>{refs}</members>"
            f"<distribution>{_expression(first, tree.events[first], t)}"
            "</distribution>"
            f'<factor level="2"><float value={_number(group.model.beta)}/>'
            "</factor></define-CCF-group>"
        )
    lines.append("  </define-fault-tree>")
    lines.append("  <model-data>")
    for e, model in tree.events.items():
        if e in members:
            continue
        lines.append(
            f"    <define-basic-event name={quoteattr(ids[e])}>"
            f"{label(e)}{_expression(e, model, t)}</define-basic-event>"
        )
    lines.append("  </model-data>")
    lines.append("</opsa-mef>")
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
