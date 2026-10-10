"""The units of a diagram's nodes, checked to agree.

A surpyval model is numbers: a Weibull fitted to lives in cycles and one
fitted to lives in hours are the same kind of object, and nothing in either
says which unit it is in. A diagram works in one unit throughout (its
times, intervals, rates and costs per time), so a node's model in another
unit gives wrong answers with no error. ``units`` says each node's unit, as
any text (``"hours"``, ``"cycles"``, ``"km"``), and the diagram refuses
nodes whose units differ. A nested diagram's unit is its nodes' one, and
takes part as its node's.
"""

from typing import Any, Dict, Iterable, Optional, Union

Units = Union[None, str, Dict[Any, str]]


def _text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{where} must be the name of a unit (any text, such as "
            f"'hours' or 'cycles'), got {value!r}."
        )
    return value.strip()


def checked(units: Units, nodes: Iterable, nested: Dict[Any, Any]) -> dict:
    """Each node's unit, ``{node: text}``, from ``units`` (one text for every
    node of ``nodes``, or a dict of some of them) and the units of the
    nested diagrams in ``nested`` (``{node: diagram}``); raise unless they
    all agree, ignoring case. A nested diagram whose node is given a unit
    must be in it."""
    nodes = list(nodes)
    if units is None:
        given: Dict[Any, str] = {}
    elif isinstance(units, str):
        text = _text(units, "units")
        given = dict.fromkeys(nodes, text)
    elif isinstance(units, dict):
        unknown = [node for node in units if node not in nodes]
        if unknown:
            raise ValueError(
                f"units names node(s) {unknown} that the diagram does not "
                "have."
            )
        given = {
            node: _text(value, f"The unit of node {node!r}")
            for node, value in units.items()
        }
    else:
        raise ValueError(
            "units must be the name of a unit for every node, or a dict of "
            f"each node's, got {units!r}."
        )
    out = dict(given)
    for node, diagram in nested.items():
        inner = getattr(diagram, "units", None)
        if inner is None:
            continue
        if node in given and given[node].casefold() != inner.casefold():
            raise ValueError(
                f"Node {node!r} is given the unit {given[node]!r}, but the "
                f"diagram it is works in {inner!r}."
            )
        out.setdefault(node, inner)
    groups: Dict[str, list] = {}
    for node, text in out.items():
        groups.setdefault(text.casefold(), []).append(node)
    if len(groups) > 1:
        listed = "; ".join(
            f"{out[members[0]]!r}: {members}" for members in groups.values()
        )
        raise ValueError(
            "The diagram's nodes are in different units, so their times "
            f"would be mixed: {listed}. Give every node's model in one "
            "unit (refit it in surpyval, or convert the data it was fitted "
            "to)."
        )
    return out


def common(units: dict) -> Optional[str]:
    """The diagram's unit: its nodes' one (as first given), or None if no
    node is given one."""
    return next(iter(units.values()), None)
