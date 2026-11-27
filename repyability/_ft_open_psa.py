"""A fault tree as an Open-PSA Model Exchange Format (MEF) document.

The MEF (version 2.0d) holds static fault trees: gates ``or``, ``and``,
``atleast`` (k-out-of-n) and ``not``, basic events with a probability or
a time-dependent expression of the mission time (``exponential``,
``Weibull``) or of a repaired or tested component (``GLM``,
``periodic-test``), and common-cause groups (here the beta-factor model).
It has
no dynamic gates, so a tree with PAND, SPARE or SEQ gates is refused; an
FDEP gate is exported as its static equivalent (each event it forces an OR
of the event and its triggers), which is exact.

Names in the MEF are identifiers (letters, digits, ``-`` and ``_``,
starting with a letter): a name that is not one is made one, and the name
as given is kept as the element's ``label``.
"""

import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)
from xml.sax.saxutils import quoteattr

import numpy as np

#: The prefix of a common-cause group's attribute that keeps the name of
#: a member that had to be made an identifier.
_MEMBER_LABEL = "label-"


class _Group(NamedTuple):
    """A common-cause group's place among the tree's, to name it by."""

    place: int


def _identifiers(names: List[Hashable]) -> Dict[Hashable, str]:
    """An MEF identifier for each name, unique (case-insensitively, as
    some tools compare them)."""
    out: Dict[Hashable, str] = {}
    taken: set = set()
    for name in names:
        base = _base(
            f"ccf-group-{name.place + 1}"
            if isinstance(name, _Group)
            else str(name)
        )
        candidate, n = base, 1
        while candidate.lower() in taken:
            n += 1
            candidate = f"{base}-{n}"
        taken.add(candidate.lower())
        out[name] = candidate
    return out


def _base(text: str) -> str:
    """The MEF identifier made of ``text``, before it is made unique."""
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "x"
    return base if base[0].isalpha() else "x_" + base


def _number(x: float) -> str:
    return quoteattr(repr(float(x)))


def _expression(event: Hashable, model, t: Optional[float]) -> str:
    """The MEF expression of an event's probability (of a repaired or
    tested event, its unavailability)."""
    from surpyval import Exponential, Weibull

    from repyability.rbd._model_utils import parametric_spec

    if isinstance(model, float):
        return f"<float value={_number(model)}/>"
    if isinstance(model, dict):
        return _repaired_expression(event, model)
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


#: The keys of a tested event's ``"inspection"`` the format holds.
_TEST_KEYS = ("interval", "offset", "duration")


def _rate(event: Hashable, model, what: str) -> float:
    """The rate of an exponential ``model``, the event's ``what``; refused
    for any other model."""
    from surpyval import Exponential

    from repyability.rbd._model_utils import distribution_name, parametric_spec

    spec = parametric_spec(model)
    if spec is None or spec.cls is not Exponential or spec.extras:
        kind = distribution_name(model) or type(model).__name__
        raise NotImplementedError(
            f"Repaired event {event!r}: its {what} is a {kind}, and the "
            "format's repaired and tested events (GLM, periodic-test) "
            "have exponential lives and repairs: pass t to export each "
            "event's unavailability at that time."
        )
    return float(spec.params[0])


def _repaired_expression(event: Hashable, spec: dict) -> str:
    """The MEF expression of a repaired or tested event's unavailability
    over the mission time: ``GLM`` for one repaired as it fails, and
    ``periodic-test`` for a hidden failure found by tests (4 arguments
    with an instant repair, 5 with an exponential one, and 11 for tests
    that take a fixed time)."""
    lam = _rate(event, spec["reliability"], "life")
    repair = spec["repairability"]
    instant = isinstance(repair, str) and repair == "instant"
    mu = None if instant else _rate(event, repair, "repair time")
    test = spec.get("inspection")
    time = "<system-mission-time/>"

    def floats(*values: float) -> str:
        return "".join(f"<float value={_number(v)}/>" for v in values)

    if test is None:
        if mu is None:
            raise NotImplementedError(
                f"Repaired event {event!r} is repaired at once, so it is "
                "never down: the format's GLM takes a repair rate. Give "
                "it an exponential repair time, or leave it out."
            )
        return f"<GLM>{floats(0.0, lam, mu)}{time}</GLM>"
    from repyability.rbd._spec import _validate_inspection

    extra = sorted(map(str, set(test) - set(_TEST_KEYS)))
    if extra:
        raise NotImplementedError(
            f"Tested event {event!r}: its inspection has {extra}, which "
            "the format's periodic-test does not hold (it has the "
            f"interval, the first test and a fixed test time: "
            f"{', '.join(_TEST_KEYS)}). Pass t to export each event's "
            "unavailability at that time."
        )
    schedule, _ = _validate_inspection(event, test)
    tau = schedule.interval
    theta = schedule.offset or tau
    duration = test.get("duration")
    if schedule.duration is None:
        # The 4 arguments with an instant repair, else the 5.
        rates = (lam,) if mu is None else (lam, mu)
        return (
            f"<periodic-test>{floats(*rates, tau, theta)}{time}"
            "</periodic-test>"
        )
    if not isinstance(duration, (int, float, np.number)) or isinstance(
        duration, (bool, np.bool_)
    ):
        raise NotImplementedError(
            f"Tested event {event!r}: its test takes a time drawn from "
            f"{duration!r}, and the format's periodic-test takes a fixed "
            "test time. Pass t to export each event's unavailability at "
            "that time."
        )
    if mu is None:
        raise NotImplementedError(
            f"Tested event {event!r} is repaired at once after a test "
            "that takes time, and the format's periodic-test with a test "
            "time takes a repair rate: give the event an exponential "
            "repair time, or pass t to export each event's unavailability "
            "at that time."
        )
    # lambda, lambda* (no failure in a test: the unit does not age), mu,
    # tau, theta, gamma (no failure caused by the test), the test time,
    # x (off line in the test), sigma (every failure found), omega (no
    # bad restart) and the time.
    return (
        f"<periodic-test>{floats(lam, 0.0, mu, tau, theta, 0.0)}"
        f'{floats(float(duration))}<bool value="false"/>{floats(1.0, 0.0)}'
        f"{time}</periodic-test>"
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
    events = dict(tree.events)
    if tree.is_repairable and t is not None:
        from repyability.rbd._event_curves import down_at

        # Each event's unavailability at t, worked out in its own right.
        down = down_at(tree._repaired, np.array([t]))
        events = {e: float(np.ravel(down[e])[0]) for e in events}
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
        # A member is defined by its group, which has no labels for its
        # members: a name made an identifier is kept as an attribute.
        kept = "".join(
            f"<attribute name={quoteattr(_MEMBER_LABEL + ids[m])} "
            f"value={quoteattr(str(m))}/>"
            for m in group.members
            if ids[m] != str(m)
        )
        lines.append(
            f"    <define-CCF-group name={quoteattr(ids[groups[i]])} "
            'model="beta-factor">'
            + (f"<attributes>{kept}</attributes>" if kept else "")
            + f"<members>{refs}</members>"
            f"<distribution>{_expression(first, tree.events[first], t)}"
            "</distribution>"
            f'<factor level="2"><float value={_number(group.model.beta)}/>'
            "</factor></define-CCF-group>"
        )
    lines.append("  </define-fault-tree>")
    lines.append("  <model-data>")
    for e, model in events.items():
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


# -- reading ----------------------------------------------------------------

#: What the reader does not take, by element name, with the reason.
_REFUSED = {
    **dict.fromkeys(
        ("GLM", "periodic-test"),
        "it is read as a basic event's whole expression of the mission "
        "time (a repaired or tested event, its last argument "
        "<system-mission-time/> and the others constants), not inside "
        "another expression",
    ),
    **dict.fromkeys(
        ("MGL", "alpha-factor", "phi-factor"),
        "only the beta-factor model of common causes is read: build the "
        "group with repyability's CCFGroup and its MGL or AlphaFactor",
    ),
    **dict.fromkeys(
        ("iff", "imply", "cardinality"),
        "the reader takes the gates or, and, atleast, not, nand, nor and "
        "xor: write the formula with them",
    ),
    **dict.fromkeys(
        (
            "uniform-deviate",
            "normal-deviate",
            "lognormal-deviate",
            "gamma-deviate",
            "beta-deviate",
            "histogram",
        ),
        "the reader takes point values, not uncertainty deviates: give the "
        "value (the deviate's mean, say) as a float, and the uncertainty "
        "through a fitted model or sf_uncertainty",
    ),
    **dict.fromkeys(
        (
            "define-event-tree",
            "define-initiating-event",
            "define-initiating-event-group",
            "define-sequence",
            "define-branch",
            "define-consequence",
            "define-consequence-group",
            "define-alignment",
            "define-rule",
            "define-substitution",
        ),
        "a FaultTree is one fault tree, not an event tree's model: read "
        "each fault tree on its own (tree=)",
    ),
    **dict.fromkeys(
        ("define-extern-function", "extern-function"),
        "an external function is code the document cannot carry",
    ),
    "define-component": "a fault tree's components are not read: define "
    "the gates and events in the fault tree itself",
    "constant": "a constant in a formula is not read: use a house-event",
}

_GATES = ("or", "and", "atleast", "not", "nand", "nor", "xor")
_REFERENCES = ("gate", "basic-event", "house-event", "event")
_ARITHMETIC: Dict[str, Tuple[Optional[int], Callable[..., float]]] = {
    "neg": (1, lambda a: -a),
    "abs": (1, abs),
    "exp": (1, np.exp),
    "log": (1, np.log),
    "sqrt": (1, np.sqrt),
    "add": (None, lambda *a: float(np.sum(a))),
    "sub": (2, lambda a, b: a - b),
    "mul": (None, lambda *a: float(np.prod(a))),
    "div": (2, lambda a, b: a / b),
    "pow": (2, lambda a, b: a**b),
}


def _refuse(element: str, where: str) -> None:
    raise NotImplementedError(
        f"{where}: <{element}> is not read: {_REFUSED[element]}."
    )


class _Expression(NamedTuple):
    """An expression of the document: its value at a mission time, and
    whether it depends on it."""

    at: Any  # Callable[[Optional[float]], float]
    timed: bool


def _document(source) -> str:
    """The MEF document ``source`` gives: its text, a path to a file
    holding it, or a file opened for reading."""
    from pathlib import Path

    if hasattr(source, "read"):
        source = source.read()
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    if isinstance(source, str) and source.lstrip("﻿ \t\r\n").startswith("<"):
        return source
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"{str(source)!r} is neither an Open-PSA document (XML, "
                "starting with '<') nor a file."
            )
        return path.read_bytes().decode("utf-8")
    raise TypeError(
        "Give an Open-PSA document as its text, a path to a file, or a "
        f"file, not a {type(source).__name__}."
    )


def _parsed(text: str):
    """The document's root element, once it is known to declare no
    document type or entity (the input is untrusted: an entity can expand
    without bound or read a file)."""
    from xml.etree import ElementTree
    from xml.parsers import expat

    def declared(*args):
        raise ValueError(
            "The document declares a document type (<!DOCTYPE ...>) or an "
            "entity, which is not read, as an entity can expand without "
            "bound or read a file: remove the declaration (an Open-PSA "
            "document needs none)."
        )

    checker = expat.ParserCreate()
    checker.StartDoctypeDeclHandler = declared
    checker.EntityDeclHandler = declared
    try:
        checker.Parse(text.lstrip("﻿"), True)
        root = ElementTree.fromstring(text.lstrip("﻿"))
    except (expat.ExpatError, ElementTree.ParseError) as error:
        raise ValueError(f"Not a well-formed XML document: {error}.") from None
    if root.tag != "opsa-mef":
        raise ValueError(
            f"Not an Open-PSA document: its root is <{root.tag}>, not "
            "<opsa-mef>."
        )
    return root


def _restored(ids: Dict[str, Optional[str]]) -> Dict[str, Hashable]:
    """Each identifier's name: its label where the label is what the
    identifier was made of (as ``_identifiers`` makes them), else the
    identifier itself, and the identifier wherever two would be one."""
    names: Dict[str, Hashable] = {}
    for i, label in ids.items():
        base = None if label is None else _base(label)
        restores = base is not None and (
            i == base or re.fullmatch(re.escape(base) + r"-\d+", i)
        )
        names[i] = label if restores else i
    counts: Dict[Hashable, int] = {}
    for name in names.values():
        counts[name] = counts.get(name, 0) + 1
    return {i: (name if counts[name] == 1 else i) for i, name in names.items()}


class _Reader:
    """The definitions of a document, read into a ``FaultTree``'s gates,
    events and common-cause groups."""

    def __init__(self, root, tree: Optional[str], mission_time):
        self.mission_time = mission_time
        trees = []
        self.parameters: Dict[str, Any] = {}
        self.events: Dict[str, Any] = {}  # id: element (or a group's)
        self.labels: Dict[str, Optional[str]] = {}
        self.groups: list = []
        data = []
        for child in root:
            if child.tag == "define-fault-tree":
                trees.append(child)
            elif child.tag == "model-data":
                data.append(child)
            elif child.tag == "define-CCF-group":
                self._group(child)
            elif child.tag not in ("label", "attributes"):
                self._unknown(child.tag, "The document")
        names = [t.get("name") for t in trees]
        if not trees:
            raise ValueError("The document defines no fault tree.")
        if tree is None:
            if len(trees) > 1:
                raise ValueError(
                    f"The document defines {len(trees)} fault trees, "
                    f"{names}: name the one to read, tree=."
                )
            chosen = trees[0]
        elif tree in names:
            chosen = trees[names.index(tree)]
        else:
            raise ValueError(
                f"The document has no fault tree {tree!r}: it has {names}."
            )
        self.name = chosen.get("name")
        self.gates: Dict[str, Any] = {}
        # A gate is read from the chosen tree; events, parameters and
        # common-cause groups are the document's, whichever tree defines
        # them (the MEF's default, public).
        for owner, child in ((t, c) for t in trees for c in t):
            tag = child.tag
            if tag == "define-gate":
                if owner is chosen:
                    self._define(self.gates, child, "gate")
            elif tag == "define-CCF-group":
                self._group(child)
            elif tag in ("define-basic-event", "define-house-event"):
                self._define(self.events, child, "event")
            elif tag == "define-parameter":
                self._define(self.parameters, child, "parameter")
            elif tag not in ("label", "attributes"):
                self._unknown(tag, "A fault tree")
        for block in data:
            for child in block:
                tag = child.tag
                if tag in ("define-basic-event", "define-house-event"):
                    self._define(self.events, child, "event")
                elif tag == "define-parameter":
                    self._define(self.parameters, child, "parameter")
                elif tag not in ("label", "attributes"):
                    self._unknown(tag, "The model data")
        clash = set(self.gates) & set(self.events)
        if clash:
            raise ValueError(
                f"Name(s) {sorted(clash)} are defined as both gates and "
                "events: rename one of each in the document."
            )
        self.values: Dict[str, _Expression] = {}
        self._resolving: List[str] = []

    @staticmethod
    def _unknown(tag: str, where: str) -> None:
        if tag in _REFUSED:
            _refuse(tag, where)
        raise ValueError(f"{where} has an element <{tag}>, which is not MEF.")

    def _define(self, into: dict, element, kind: str) -> None:
        name = element.get("name")
        if not name:
            raise ValueError(
                f'A <{element.tag}> has no name: give it name="...".'
            )
        if name in into:
            raise ValueError(
                f"The {kind} {name!r} is defined twice: keep one definition."
            )
        into[name] = element
        if kind != "parameter":
            label = element.find("label")
            self.labels[name] = None if label is None else label.text

    def _group(self, element) -> None:
        name = element.get("name")
        model = element.get("model")
        where = f"Common-cause group {name!r}"
        if model in _REFUSED:
            _refuse(model, where)
        if model != "beta-factor":
            raise ValueError(
                f"{where} has model {model!r}, which is not an MEF "
                "common-cause model."
            )
        kept = {}
        for attribute in element.iterfind("attributes/attribute"):
            key = attribute.get("name") or ""
            if key.startswith(_MEMBER_LABEL):
                kept[key[len(_MEMBER_LABEL) :]] = attribute.get("value")
        members = [m.get("name") for m in element.iterfind("members/*")]
        distribution = element.find("distribution")
        factors = element.findall("factor") + element.findall("factors/factor")
        if len(members) < 2 or distribution is None or len(factors) != 1:
            raise ValueError(
                f"{where} needs <members> (two or more), a <distribution> "
                "and one <factor> (beta), as the beta-factor model has."
            )
        for m in members:
            if m in self.events:
                raise ValueError(
                    f"The event {m!r} is defined twice (as a member of "
                    f"{where.lower()} too): keep one definition."
                )
            self.events[m] = distribution
            self.labels[m] = kept.get(m)
        self.groups.append((members, factors[0], where))

    # -- expressions ------------------------------------------------------

    def _expression(self, element, where: str) -> _Expression:
        tag = element.tag
        if tag in _REFUSED:
            _refuse(tag, where)
        args = [
            self._expression(a, where)
            for a in element
            if a.tag not in ("label", "attributes")
        ]
        timed = any(a.timed for a in args)
        if tag in ("float", "int", "bool"):
            text = element.get("value", "")
            try:
                value = (
                    float({"true": 1, "false": 0}[text])
                    if tag == "bool"
                    else float(text)
                )
            except (KeyError, ValueError):
                raise ValueError(
                    f"{where}: <{tag} value={text!r}> is not a number."
                ) from None
            return _Expression(lambda t: value, False)
        if tag == "system-mission-time":
            return _Expression(lambda t: t, True)
        if tag == "parameter":
            return self._parameter(element.get("name"), where)
        lives: Dict[str, Tuple[Optional[int], Callable[..., float]]] = {
            "exponential": (2, _exponential),
            "Weibull": (4, _weibull),
        }
        n, f = {**lives, **_ARITHMETIC}.get(tag, (-1, _unknown_function))
        if f is _unknown_function:
            raise ValueError(
                f"{where}: <{tag}> is not an expression the reader takes "
                "(float, int, bool, parameter, system-mission-time, "
                "exponential, Weibull and arithmetic: "
                f"{', '.join(_ARITHMETIC)})."
            )
        if (n is None and not args) or (n is not None and len(args) != n):
            raise ValueError(
                f"{where}: <{tag}> takes {n or 'one or more'} argument(s), "
                f"got {len(args)}."
            )
        return _Expression(lambda t: f(*(a.at(t) for a in args)), timed)

    def _parameter(self, name: str, where: str) -> _Expression:
        if name in self.values:
            return self.values[name]
        if name not in self.parameters:
            raise ValueError(
                f"{where} uses the parameter {name!r}, which the document "
                "does not define: add its <define-parameter>."
            )
        if name in self._resolving:
            loop = self._resolving[self._resolving.index(name) :] + [name]
            raise ValueError(
                f"The parameters form a loop, {' -> '.join(loop)}: define "
                "one of them by a value."
            )
        self._resolving.append(name)
        body = [
            e
            for e in self.parameters[name]
            if e.tag not in ("label", "attributes")
        ]
        if len(body) != 1:
            raise ValueError(
                f"The parameter {name!r} needs one expression, has "
                f"{len(body)}."
            )
        value = self._expression(body[0], f"Parameter {name!r}")
        self._resolving.pop()
        self.values[name] = value
        return value

    def _constant(self, element, where: str) -> float:
        value = self._expression(element, where)
        if value.timed:
            raise ValueError(
                f"{where} depends on the mission time, where a constant "
                "is needed."
            )
        return float(value.at(None))

    def _through(self, element):
        """``element``, or the expression the parameter it is names."""
        seen = []
        while element.tag == "parameter":
            name = element.get("name")
            if name not in self.parameters or name in seen:
                return element
            seen.append(name)
            body = [
                e
                for e in self.parameters[name]
                if e.tag not in ("label", "attributes")
            ]
            if len(body) != 1:
                return element
            element = body[0]
        return element

    def model(self, event: str, name: Hashable):
        """The event's model: a probability, or a surpyval model for an
        exponential or Weibull life over the mission time."""
        import surpyval as surv

        element = self.events[event]
        where = f"Event {name!r}"
        if element.tag == "define-house-event":
            constant = element.find("constant")
            text = "false" if constant is None else constant.get("value")
            if text not in ("true", "false"):
                raise ValueError(
                    f"House event {name!r}: its constant must be true or "
                    f"false, got {text!r}."
                )
            return 1.0 if text == "true" else 0.0
        body = [e for e in element if e.tag not in ("label", "attributes")]
        if len(body) != 1:
            raise ValueError(
                f"{where} needs one expression for its probability, has "
                f'{len(body)}: give it, e.g. <float value="0.01"/>.'
            )
        expression = self._through(body[0])
        args = [
            self._through(a)
            for a in expression
            if a.tag not in ("label", "attributes")
        ]
        if expression.tag in ("GLM", "periodic-test"):
            return self._repaired(expression, args, where)
        life = {"exponential": 2, "Weibull": 4}.get(expression.tag)
        if (
            life == len(args)
            and args[-1].tag == "system-mission-time"
            and not any(self._expression(a, where).timed for a in args[:-1])
        ):
            params = [self._constant(a, where) for a in args[:-1]]
            try:
                if life == 2:
                    return surv.Exponential.from_params(params)
                alpha, beta, t0 = params
                extras = {"gamma": t0} if t0 != 0.0 else {}
                return surv.Weibull.from_params([alpha, beta], **extras)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"{where}: its {expression.tag} parameters {params} "
                    f"are not a valid life: {error}"
                ) from None
        value = self._expression(body[0], where)
        if value.timed and self.mission_time is None:
            raise ValueError(
                f"{where}: its probability is an expression of the mission "
                "time other than an exponential or Weibull life, which has "
                "no model: pass mission_time to read it at that time."
            )
        return float(value.at(self.mission_time))

    def _repaired(self, expression, args: list, where: str) -> dict:
        """The component spec of a repaired event (``GLM``) or a tested
        one (``periodic-test``), where the expression is one: refused,
        saying why, where it is not."""
        import surpyval as surv

        tag = expression.tag
        if not (
            args
            and args[-1].tag == "system-mission-time"
            and not any(self._expression(a, where).timed for a in args[:-1])
        ):
            _refuse(tag, where)
        values = [self._constant(a, where) for a in args[:-1]]
        counts = (3,) if tag == "GLM" else (3, 4, 10)
        if len(values) not in counts:
            raise ValueError(
                f"{where}: <{tag}> takes "
                f"{' or '.join(str(n + 1) for n in counts)} arguments, "
                f"got {len(args)}."
            )

        def rate(value: float, what: str):
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"{where}: the {what} of its <{tag}> must be a "
                    f"positive, finite rate, got {value!r}."
                )
            return surv.Exponential.from_params([value])

        if tag == "GLM":
            gamma, failure, repair = values
            if gamma != 0.0:
                raise NotImplementedError(
                    f"{where}: its <GLM> fails on demand at the start with "
                    f"probability gamma = {gamma!r}, and a repaired event "
                    "here is new (working) at 0, with no component spec "
                    "for a failure at the start: only gamma = 0 is read."
                )
            if repair == 0.0:
                # Never repaired: the exponential life itself.
                return rate(failure, "failure rate")
            return {
                "reliability": rate(failure, "failure rate"),
                "repairability": rate(repair, "repair rate"),
            }
        mu: Optional[float]
        if len(values) == 3:
            lam, tau, theta = values
            mu, duration = None, 0.0
        elif len(values) == 4:
            lam, mu, tau, theta = values
            duration = 0.0
        else:
            lam, lam_test, mu, tau, theta = values[:5]
            gamma, duration, x, sigma, omega = values[5:]
            fixed = {
                "lambda* (the failure rate in a test: a test here does "
                "not age the unit)": (lam_test, 0.0),
                "gamma (a failure caused by the test)": (gamma, 0.0),
                "x (the unit available in its test: a test here takes "
                "it off line)": (x, 0.0),
                "sigma (the chance that a test finds a failure: a "
                "test here that misses one leaves it to a full test, "
                "which the format has not)": (sigma, 1.0),
                "omega (a bad restart after a test or repair)": (
                    omega,
                    0.0,
                ),
            }
            differ = [
                f"{what} = {value!r}, not {wanted!r}"
                for what, (value, wanted) in fixed.items()
                if value != wanted
            ]
            if differ:
                raise NotImplementedError(
                    f"{where}: its 11-argument <periodic-test> is read "
                    "where it is a tested event's, and it has "
                    + "; ".join(differ)
                    + ". Give the event a component spec in the "
                    "FaultTree instead, or read the document at a "
                    "mission time after replacing the expression by its "
                    "value."
                )
            if not (np.isfinite(duration) and duration >= 0.0):
                raise ValueError(
                    f"{where}: the test time of its <periodic-test> must "
                    f"be a finite time, 0 or more, got {duration!r}."
                )
        if not (np.isfinite(tau) and tau > 0.0):
            raise ValueError(
                f"{where}: the time between tests of its <periodic-test> "
                f"must be positive and finite, got {tau!r}."
            )
        if not (np.isfinite(theta) and theta >= 0.0):
            raise ValueError(
                f"{where}: the time of the first test of its "
                f"<periodic-test> must be a finite time, 0 or more, got "
                f"{theta!r}."
            )
        if theta > tau or (theta == 0.0 and duration > 0.0):
            raise NotImplementedError(
                f"{where}: its <periodic-test> has its first test at "
                f"theta = {theta!r}, with tests every {tau!r}, and a "
                "tested event's first test falls after 0 and within its "
                "first interval (its inspection offset): give the event "
                "a component spec in the FaultTree instead."
            )
        inspection: Dict[str, float] = {"interval": tau}
        if 0.0 < theta < tau:
            inspection["offset"] = theta
        if duration > 0.0:
            inspection["duration"] = duration
        if mu == 0.0:
            raise NotImplementedError(
                f"{where}: its <periodic-test> has a repair rate of 0, so "
                "a failure found is never repaired, which a tested event "
                "here is: give it a repair rate."
            )
        return {
            "reliability": rate(lam, "failure rate"),
            "repairability": (
                "instant" if mu is None else rate(mu, "repair rate")
            ),
            "inspection": inspection,
        }

    # -- formulas ---------------------------------------------------------

    def tree(self, top: Optional[str]):
        """The gates, events, top and common-cause groups to build the
        ``FaultTree`` with."""
        from repyability.rbd.ccf import BetaFactor, CCFGroup

        names = _restored({**self.labels})
        taken = set(names.values())
        gates: Dict[Hashable, tuple] = {}
        used: List[str] = []

        def helper(owner: Hashable) -> Hashable:
            n = 1
            while f"{owner}/{n}" in taken:
                n += 1
            taken.add(f"{owner}/{n}")
            return f"{owner}/{n}"

        def reference(element, owner) -> Hashable:
            tag, name = element.tag, element.get("name")
            kind = self.events[name].tag if name in self.events else None
            if tag in ("gate", "event") and name in self.gates:
                return names[name]
            if (
                (tag == "event" and kind is not None)
                or (tag == "house-event" and kind == "define-house-event")
                or (
                    tag == "basic-event"
                    and kind in ("define-basic-event", "distribution")
                )
            ):
                if name not in used:
                    used.append(name)
                return names[name]
            raise ValueError(
                f"Gate {owner!r} has the input <{tag} name={name!r}>, which "
                "the document does not define as that: define it (a "
                "<define-gate>, <define-basic-event> or "
                "<define-house-event>), or correct the reference."
            )

        def formula(element, owner, name) -> None:
            tag = element.tag
            if tag in _REFUSED:
                _refuse(tag, f"Gate {owner!r}")
            if tag not in _GATES:
                raise ValueError(
                    f"Gate {owner!r} has the formula <{tag}>, which is not "
                    "a gate the reader takes: or, and, atleast, not, nand, "
                    "nor or xor."
                )
            inputs = [
                argument(a, owner)
                for a in element
                if a.tag not in ("label", "attributes")
            ]
            if tag == "or":
                gates[name] = ("or", inputs)
            elif tag == "and":
                gates[name] = ("and", inputs)
            elif tag == "atleast":
                k = element.get("min", "")
                if not k.isdigit():
                    raise ValueError(
                        f'Gate {owner!r}: <atleast> needs min="k", a '
                        f"whole number, got {k!r}."
                    )
                gates[name] = ("vote", int(k), inputs)
            elif tag == "not":
                gates[name] = ("not", inputs)
            elif tag in ("nand", "nor"):
                inner = helper(owner)
                gates[name] = ("not", [inner])
                gates[inner] = ("and" if tag == "nand" else "or", inputs)
            else:  # xor
                if len(inputs) != 2:
                    raise NotImplementedError(
                        f"Gate {owner!r}: <xor> of {len(inputs)} inputs is "
                        "not read, only of two: write it of xor gates of "
                        "two."
                    )
                a, b = inputs
                first, second = helper(owner), helper(owner)
                not_a, not_b = helper(owner), helper(owner)
                gates[name] = ("or", [first, second])
                gates[first] = ("and", [a, not_b])
                gates[second] = ("and", [not_a, b])
                gates[not_a] = ("not", [a])
                gates[not_b] = ("not", [b])

        def argument(element, owner) -> Hashable:
            if element.tag in _REFERENCES:
                return reference(element, owner)
            inner = helper(owner)
            formula(element, owner, inner)
            return inner

        for i, element in self.gates.items():
            body = [e for e in element if e.tag not in ("label", "attributes")]
            if len(body) != 1:
                raise ValueError(
                    f"Gate {names[i]!r} needs one formula, has {len(body)}."
                )
            if body[0].tag in _REFERENCES:
                gates[names[i]] = ("or", [reference(body[0], names[i])])
            else:
                formula(body[0], names[i], names[i])
        top_name = self._top(
            gates, None if top is None else names.get(top, top)
        )
        below = _below(gates, top_name)
        gates = {g: spec for g, spec in gates.items() if g in below}
        events = {
            names[e]: self.model(e, names[e])
            for e in self.events
            if names[e] in below and e in used
        }
        groups = []
        for members, factor, where in self.groups:
            kept = [names[m] for m in members if names[m] in events]
            if len(kept) > 1:
                beta = self._constant(
                    next(e for e in factor if e.tag != "label"), where
                )
                groups.append(CCFGroup(kept, BetaFactor(beta)))
        return gates, events, top_name, groups

    def _top(self, gates: dict, top) -> Hashable:
        """The top gate: ``top``, or the one gate no gate takes as an
        input; refused if the gates form a loop."""
        state: Dict[Hashable, int] = {}
        path: list = []

        def visit(g) -> None:
            state[g] = 1
            path.append(g)
            for x in gates[g][-1]:
                if x not in gates:
                    continue
                if state.get(x) == 1:
                    loop = path[path.index(x) :] + [x]
                    raise ValueError(
                        "The gates form a loop, "
                        f"{' -> '.join(map(str, loop))}: a fault tree's "
                        "gates must not; correct the document."
                    )
                if x not in state:
                    visit(x)
            path.pop()
            state[g] = 2

        for g in gates:
            if g not in state:
                visit(g)
        if top is not None:
            if top not in gates:
                raise ValueError(
                    f"top={top!r} is not a gate of fault tree "
                    f"{self.name!r}."
                )
            return top
        inputs = {x for spec in gates.values() for x in spec[-1]}
        roots = [g for g in gates if g not in inputs]
        if len(roots) != 1:
            raise ValueError(
                f"Fault tree {self.name!r} has {len(roots)} gates that are "
                f"no gate's input, {[str(r) for r in roots]}: name the top "
                "event, top=."
            )
        return roots[0]


def _below(gates: dict, top) -> set:
    seen = set()
    stack = [top]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        if x in gates:
            stack.extend(gates[x][-1])
    return seen


def _unknown_function(*args: float) -> float:  # pragma: no cover
    raise AssertionError("not an MEF function")


def _exponential(rate: float, t: float) -> float:
    return float(-np.expm1(-rate * t))


def _weibull(alpha: float, beta: float, t0: float, t: float) -> float:
    return 0.0 if t <= t0 else float(-np.expm1(-(((t - t0) / alpha) ** beta)))


def from_open_psa(cls, source, tree, mission_time, top):
    """The ``FaultTree`` of an MEF document (see
    ``FaultTree.from_open_psa``)."""
    if tree is not None and not isinstance(tree, str):
        raise TypeError(f"tree must be a fault tree's name, got {tree!r}.")
    if mission_time is not None:
        from repyability.utils.checks import number_or_nan

        value = number_or_nan(mission_time)
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(
                "mission_time must be a finite time, 0 or more, got "
                f"{mission_time!r}."
            )
        mission_time = value
    reader = _Reader(_parsed(_document(source)), tree, mission_time)
    gates, events, top, groups = reader.tree(top)
    return cls(gates, events, top=top, ccf_groups=groups or None)
