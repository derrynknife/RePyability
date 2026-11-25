"""A nested ``RepairableRBD``'s costs, counted in the diagram it is a node
of: its repairs, tasks, own downtime, crews, set-ups and capital, by
category, each labelled with where it falls.

A nested diagram runs on its own: nothing in the diagram it is a node of
changes what happens inside it (its components fail, are repaired and
maintained by its own rules, with crews of its own). So its costs are
its own cost methods' answers, added in:

- exactly, its expected costs over a window, discounted or not, and in
  the long run its cost rate (``window_parts``, ``rate``);
- simulated, its costs in each simulation, from a run of the nested
  diagram alone (``charge``) whose streams are named by its place in the
  system (``prefix``): every stream is keyed by its path, so simulation
  ``r`` of the nested diagram alone draws exactly what it draws inside
  simulation ``r`` of the system, on either engine;
- in the histories of a valuation of runs, its histories from the same
  run alone, priced as the system's are (``_pricing``).

A label is where a cost falls: a component of the nested diagram at
``name`` is ``(name, component)`` (and deeper, the path down), and what
the nested diagram itself pays (its own downtime, as a node down, its
crews and set-ups) is its node's, ``name``. A nested diagram's
``downtime_cost_rate`` is the cost of that node being down, so it is
the system's ``component_downtime``.

Held working or broken, a nested node is not run: held working it costs
nothing; held broken, its own downtime cost throughout. A nested diagram
tied to the system its costs could not be taken alone in (one with a
downtime cost in a shutdown's scope, whose holds the system sees and it
does not), or one
whose draws are not all streamed (which the simulation could not
repeat alone), is refused, with the reason (``refusal``).
"""

from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple

import numpy as np

#: A diagram's own costs that no component of it carries.
SYSTEM = ("system_downtime", "setup", "shutdown", "crews")


def as_parent(category: str) -> str:
    """A nested diagram's cost category as the system counts it: its
    system downtime is its node's downtime."""
    return "component_downtime" if category == "system_downtime" else category


def priced(rbd) -> List[Tuple[Hashable, Any]]:
    """The nested diagrams that carry running costs, by node, in the
    diagram's order."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return [
        (name, component)
        for name, component in getattr(rbd, "components", {}).items()
        if isinstance(component, RepairableRBD) and component.has_costs
    ]


def acquisitions(rbd) -> Dict[Hashable, float]:
    """The one-off costs of buying the diagram: its components' own
    ``"acquisition_cost"``, and each nested diagram's, all of it under
    its node."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    out = dict(rbd.acquisition_costs)
    for name, component in rbd.components.items():
        if isinstance(component, RepairableRBD) and component.acquisition_cost:
            out[name] = component.acquisition_cost
    return out


def relabel(nested, name, label):
    """A label of the nested diagram at ``name`` (one of its components,
    or a deeper label) as the system labels it."""
    if label in nested.components:
        return (name, label)
    return (name,) + tuple(label)


def system_priced(rbd) -> bool:
    """Whether the diagram pays anything no component of it carries: a
    system downtime cost, crews, a call-out crew, set-ups or shutdowns."""
    from repyability.rbd import _crews
    from repyability.rbd._common import _crew_salaries

    return bool(
        rbd.downtime_cost_rate
        or _crew_salaries(rbd)
        or _crews._call_out_priced(rbd)
        or any(group.setup_cost for group in rbd._maintenance.values())
        or any(shut.cost for shut in rbd._shutdowns.values())
    )


def owner(rbd, label):
    """The node of the diagram a cost's ``label`` falls under: the label,
    or for a nested diagram's, the nested node."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if label in rbd.components or not isinstance(label, tuple) or not label:
        return label
    head = label[0]
    if isinstance(rbd.components.get(head), RepairableRBD):
        return head
    return label


def refusal(
    rbd, held=frozenset(), simulated: bool = False, states=None
) -> Optional[str]:
    """Why a nested diagram's costs cannot be counted in this one, or
    None: one with a downtime cost in a shutdown's scope (the system
    holds it off line, which its own downtime cost does not see); and,
    ``simulated``, one some of whose draws are not streamed (its run
    alone could not repeat them). ``held`` are the nodes held working or
    broken, which are not run."""
    from repyability.rbd import _event_loop, _shutdowns

    states = {} if states is None else states
    for name, nested in priced(rbd):
        if name in held:
            continue
        holding = (
            _shutdowns.holding(rbd, name)
            if nested.downtime_cost_rate and getattr(rbd, "_shutdowns", None)
            else []
        )
        if holding:
            shut = holding[0][0]
            return (
                f"Node {name!r} is a nested RBD with a downtime cost, in "
                f"the scope of shutdown {shut.name!r}: the shut holds the "
                "nested system off line as the diagram sees it, which its "
                "own downtime cost (worked out alone, from its own state) "
                "does not take. Leave it out of the shutdown's scope, or "
                "price its downtime in this diagram's downtime_cost_rate."
            )
        if simulated:
            _, complete = _event_loop._stream_specs(
                nested, 1.0, (name,), states=states.get(name), priced=True
            )
            if not complete:
                return (
                    f"Node {name!r} is a nested RBD that carries costs, and "
                    "some of its draws do not come from a stream (a model "
                    "other than a surpyval parametric one), so its costs "
                    "cannot be simulated alone in the system's "
                    "simulations: give surpyval parametric models, or "
                    "price the nested diagram on its own."
                )
        inner = refusal(nested, simulated=simulated, states=states.get(name))
        if inner is not None:
            return inner
    return None


def require(rbd, held=frozenset(), simulated: bool = False, states=None):
    """Raise ``NotImplementedError`` with ``refusal``'s reason."""
    reason = refusal(rbd, held, simulated, states)
    if reason is not None:
        raise NotImplementedError(reason)


def refusal_unvalued(rbd) -> Optional[str]:
    """Why a valuation cannot take a nested diagram, or None: one whose
    units carry their own salvage, escalation or retirement (their plans'
    valuation terms), which the diagram they are nested in does not
    take."""
    for name, nested in priced(rbd):
        own = getattr(nested, "_valuation", {})
        if own:
            unit = next(iter(own))
            return (
                f"Component {unit!r} of the nested RBD at node {name!r} has "
                "valuation terms of its own (a salvage, escalation or "
                "retire_at), which a valuation of the diagram it is "
                "nested in does not take: give them for the nested node, "
                f"{name!r}, in the valuation's arguments."
            )
        inner = refusal_unvalued(nested)
        if inner is not None:
            return inner
    return None


def require_unvalued(rbd) -> None:
    """Raise ``NotImplementedError`` with ``refusal_unvalued``'s reason."""
    reason = refusal_unvalued(rbd)
    if reason is not None:
        raise NotImplementedError(reason)


# -- exact ---------------------------------------------------------------


def _held_down(nested, ends: np.ndarray, rate: float) -> np.ndarray:
    """A nested diagram held broken: its own downtime cost throughout,
    discounted at ``rate``, to each of ``ends``."""
    from repyability.value._rates import discounted_span

    return nested.downtime_cost_rate * discounted_span(0.0, ends, rate)


def window_parts(
    rbd, ends: np.ndarray, rate: float, working, broken, state
) -> Iterator[Tuple[str, Any, np.ndarray]]:
    """The nested diagrams' expected costs to each of ``ends``, from new
    or from their part of the (checked) ``state``, discounted at
    ``rate``: ``(category, label, values)``, each nested diagram's own
    and, deeper, its nested diagrams'."""
    from repyability.rbd import _windows

    states = state if isinstance(state, dict) else {}
    for name, nested in priced(rbd):
        if name in working:
            continue
        if name in broken:
            if nested.downtime_cost_rate:
                yield "component_downtime", name, _held_down(
                    nested, ends, rate
                )
            continue
        start = states.get(name)
        cells: dict = {}
        if rate:
            own = _windows._discounted_cost(
                nested, ends, rate, set(), set(), "p", start, cells
            )
        else:
            own = _windows._own_cost(
                nested,
                ends,
                set(),
                set(),
                "p",
                start,
                discount_rate=0.0,
                cells=cells,
            )
        for (node, category), values in cells.items():
            yield category, (name, node), np.broadcast_to(values, ends.shape)
        for category in SYSTEM:
            values = np.broadcast_to(own.by_category[category], ends.shape)
            if np.any(values):
                yield as_parent(category), name, values
        for category, label, values in window_parts(
            nested, ends, rate, (), (), start
        ):
            yield category, relabel(nested, name, label), values


def rate(rbd, working, broken) -> Tuple[float, Dict[Any, float]]:
    """The nested diagrams' long-run cost per unit time, in all and by
    nested node."""
    total = 0.0
    by_node: Dict[Any, float] = {}
    for name, nested in priced(rbd):
        if name in working:
            continue
        value = (
            float(nested.downtime_cost_rate)
            if name in broken
            else nested.expected_cost_rate()
        )
        by_node[name] = value
        total += value
    return total, by_node


# -- simulated -----------------------------------------------------------


def simulated(rbd, working, broken) -> List[Tuple[Hashable, Any]]:
    """The nested diagrams a run of ``rbd`` charges the costs of (the
    priced ones, held or not)."""
    return priced(rbd) if rbd.has_costs else []


def charge(
    tally,
    rbd,
    nested_nodes: List[Tuple[Hashable, Any]],
    start: int,
    stop: int,
    t_simulation: float,
    working,
    broken,
    entropy,
    antithetic: bool,
    states: Optional[dict],
    jobs: Optional[int],
    prefix: tuple,
) -> None:
    """Add to ``tally`` the costs of the nested diagrams in simulations
    ``start`` to ``stop - 1``: each nested diagram's own run alone, from
    its streams (named from ``prefix``, this diagram's place), or, held
    broken, its own downtime cost throughout."""
    from repyability.rbd import _event_loop
    from repyability.rbd._exact import ExactSum

    count = stop - start
    if count <= 0:
        return
    states = {} if states is None else states
    for name, nested in nested_nodes:
        if name in working:
            continue
        if name in broken:
            amount = nested.downtime_cost_rate * t_simulation
            if amount:
                charges = np.full(count, amount)
                total = ExactSum()
                total.add(charges)
                tally.add_nested(
                    charges,
                    {"component_downtime": total},
                    {name: ExactSum(total.partials)},
                )
            continue
        part = _event_loop._run_alone(
            nested,
            t_simulation,
            set(),
            set(),
            "p",
            count,
            False,
            None,
            antithetic,
            jobs=jobs,
            engine="python",
            entropy=entropy,
            first=start,
            states=states.get(name) or {},
            curve_points=1,
            prefix=prefix + (name,),
        )
        part._fold()
        categories: Dict[str, Any] = {}
        for category, amount in part.cost_by_category.items():
            key = as_parent(category)
            categories.setdefault(key, ExactSum()).add(amount)
        components: Dict[Any, Any] = {}
        own = ExactSum()
        for amount in part.cost_by_category.values():
            own.add(amount)
        for label, amount in part.cost_by_component.items():
            components[relabel(nested, name, label)] = amount
            own.add([-x for x in amount.partials])
        if system_priced(nested):
            components[name] = own
        tally.add_nested(part.cost_samples, categories, components)


# -- the histories of a valuation of runs --------------------------------


def price_runs(
    rbd,
    ledger,
    money,
    horizon: float,
    size: int,
    seed,
    first: int,
    states: Optional[dict],
    prefix: tuple = (),
) -> None:
    """Price into ``ledger`` the nested diagrams' costs in simulations
    ``first`` to ``first + size - 1`` of the run ``seed`` seeds: each
    nested diagram's histories from its own run alone (named from its
    place, so they are those it has inside the system's runs), priced
    as the system's are (``_pricing.priced``), labelled with its path;
    and, deeper, its nested diagrams'. ``states`` are the components'
    states at 0, checked."""
    from repyability.rbd import _pricing, _timeline_runs

    states = {} if states is None else states
    for name, nested in priced(rbd):
        path = prefix + (name,)
        start = states.get(name) or None
        runs = _timeline_runs.simulate(
            nested,
            horizon,
            size,
            seed,
            None,
            None,
            False,
            "python",
            None,
            first,
            start,
            prefix=path,
        )
        logs = {node: runs.components[node]._data for node in nested.costs}
        _pricing.priced(
            nested,
            logs,
            runs.system._data,
            ledger.ends,
            money,
            (),
            size,
            ledger=ledger,
            where=path,
        )
        price_runs(
            nested, ledger, money, horizon, size, seed, first, start, path
        )


# -- an outage log -------------------------------------------------------

#: Why a nested diagram's costs inside it are not compared with a log.
_INSIDE = (
    "A nested RBD's log is the nested system's: the events of its "
    "components, which are charged, are not in it."
)


def _logged(nested) -> Tuple[str, ...]:
    """The nested diagram's own costs its node's log determines: its
    downtime cost (the node down), and its crews' salaries (paid
    whatever happens), unless a call-out crew is priced."""
    from repyability.rbd import _crews

    out: Tuple[str, ...] = ()
    if nested.downtime_cost_rate:
        out += ("component_downtime",)
    if not _crews._call_out_priced(nested):
        out += ("crews",)
    return out


def unlogged(rbd, rows) -> List[Tuple[str, Any, str]]:
    """The cells of the cash-flow ``rows`` that fall inside a nested
    diagram and that a log of its node does not determine, ``(category,
    label, reason)``."""
    nested = dict(priced(rbd))
    out: List[Tuple[str, Any, str]] = []
    seen = set()
    for row in rows:
        label, category = row["component"], row["category"]
        if category == "acquisition":
            continue
        name = owner(rbd, label)
        if name not in nested or (category, label) in seen:
            continue
        if label == name and category in _logged(nested[name]):
            continue
        seen.add((category, label))
        out.append((category, label, _INSIDE))
    return out


def price_logs(rbd, logs, ledger, money) -> None:
    """Price into ``ledger`` what each nested diagram's log (its node's,
    ``logs``) determines of its own costs (see ``_logged``)."""
    from repyability.rbd import _pricing
    from repyability.rbd._common import _crew_salaries

    for name, nested in priced(rbd):
        kinds = _logged(nested)
        if "component_downtime" in kinds:
            ledger.spans(
                *_pricing.spans(logs[name]._data, up=False),
                nested.downtime_cost_rate,
                "component_downtime",
                name,
                money.growth(name, "component_downtime"),
            )
        if "crews" in kinds:
            ledger.spans(
                np.zeros(1, np.int64),
                np.zeros(1),
                np.full(1, ledger.ends[-1]),
                _crew_salaries(nested),
                "crews",
                name,
                money.growth(name, "crews"),
            )
