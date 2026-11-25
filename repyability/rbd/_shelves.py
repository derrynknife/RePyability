"""A shelf of spares in the simulation: a stock ``S``, reordered one for
one, each order arriving a lead time ``L`` after it was placed.

A shelf belongs to a component (or to a part that several share, or to a
node and its copies). Each *demand* on it takes a spare: by default each
replacement, as ``spares_demand`` counts them (a failure repaired as new,
or a standby, load-sharing or identical group's unit failure), while a
failure repaired imperfectly (the spec's ``"repair"``) is a repair, which
takes none, unless the shelf's ``repairs`` is True. A demand places an
order at once (one for one, ``(S - 1, S)``). With a spare on the shelf the
repair starts at once; with none, it waits for the first order still to
arrive that no earlier demand waits for. Orders arrive first in, first
out, as every one takes ``L``: the ``k``-th demand's spare is ready at
``max(t_k, t_{k - S} + L)`` (``t_k + L`` with no stock), so the repair,
and the unit's downtime, start late by the difference. A repair that
waits is a *stock-out*. Nothing else changes: the draws are those of the
run without a shelf, so a stock no run uses up gives that run bit for bit.

A repair that waits for its spare keeps the unit down until the spare is
in, and then (with repair crews) queues for a crew: the crew is not held
while the spare is on its way, so the job falls due when the spare
arrives (an ``Arrival``, taken at its instant among the failures). A
preventive task that replaces the unit, and finds the shelf empty, is
put off until the spare is in: the unit runs on as it is, and should it
fail first, the failure takes the spare ordered for the task (and the
task, its spare used, is not done). A test that finds a hidden failure
is a demand: the repair starts once the test is over and the spare is
in. A unit replaced on condition takes none of this yet.

A sub-system's (a nested RBD's) component draws on a shelf keyed
``(node, component)``, run in the sub-system's own loop, which takes its
events in time order (its parent asks it for its next change, and it
runs ahead to it): a replacement there takes a spare as at the top
level, its plain components and groups alike. So a shelf is either one
sub-system's or the top level's: shared across levels, or across
sub-systems, its demands would come out of time order, and it is
refused. A fleet's systems draw on one shelf as one diagram of their
copies (``fleet``).

The shelf is the Python event loop's alone (``_compiled.unsupported``
gives the reason to the compiled engine, which refuses rather than
differ). It is held on a copy of the diagram (``shelved``), which only the
simulated routes that take it build: ``spares_stock(method="simulate")``
and ``least_spend_for_downtime``'s ``spares`` lever. Refused, with the
reason: replacement on condition, maintenance groups, early renewals at
shutdowns and nested RBDs; with repair crews, tests of hidden failures
and a node a shutdown pauses.
"""

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, NamedTuple, Optional, Tuple

import numpy as np

from repyability.utils import checks


class Shelf(NamedTuple):
    """A shelf's settings: its ``stock``, its ``lead_time`` and whether a
    failure repaired imperfectly takes a spare too (``repairs``)."""

    stock: int
    lead_time: float
    repairs: bool = False


class Shelves(NamedTuple):
    """The shelves of a run: each shelf's settings by its name
    (``shelves``), and the shelf each node draws on (``of``)."""

    shelves: Dict[Hashable, Shelf]
    of: Dict[Hashable, Hashable]


def shelf(stock, lead_time, repairs=False, what: str = "the shelf") -> Shelf:
    """A shelf's settings, checked."""
    if not (checks.is_number(stock) and int(stock) == stock and stock >= 0):
        raise ValueError(
            f"The stock of {what} must be a whole number, at least 0; got "
            f"{stock!r}."
        )
    lead = checks.number_or_nan(lead_time)
    if not (math.isfinite(lead) and lead >= 0.0):
        raise ValueError(
            f"The lead time of {what} must be a number, at least 0; got "
            f"{lead_time!r}."
        )
    if not isinstance(repairs, (bool, np.bool_)):
        raise ValueError(
            f"repairs (whether a failure repaired imperfectly takes a spare "
            f"from {what}) must be True or False; got {repairs!r}."
        )
    return Shelf(int(stock), float(lead), bool(repairs))


def _inner_why_not(inner, node) -> Optional[str]:
    """Why component ``node`` of a sub-system ``inner`` cannot draw on a
    shelf in the sub-system's own loop, or None: it takes the plain
    replacements (and a group's units) alone."""
    from repyability.rbd import _crews
    from repyability.rbd.repairable_rbd import RepairableRBD

    if isinstance(inner.components[node], RepairableRBD):
        return "it is a nested RBD within a sub-system"
    if node in inner._preventive or node in inner._inspection:
        return (
            "it has a preventive task or tests, which a sub-system's own "
            "loop does not put off for a spare yet"
        )
    if node in inner._member_group or node in inner._outage_members:
        return "it is renewed early, which a shelf is not simulated for yet"
    if _crews._crews_limited(inner):
        return (
            "the sub-system has repair crews, which its own loop does not "
            "take with a shelf yet"
        )
    return None


def is_inner(rbd, key) -> bool:
    """Whether ``key`` names a sub-system's component, ``(node,
    component)``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    return (
        isinstance(key, tuple)
        and key not in rbd.components
        and len(key) == 2
        and isinstance(rbd.components.get(key[0]), RepairableRBD)
        and key[1] in rbd.components[key[0]].components
    )


def why_not(rbd, node) -> Optional[str]:
    """Why component ``node`` (or a sub-system's, ``(node, component)``)
    cannot draw on a shelf in the simulation, or None."""
    if is_inner(rbd, node):
        return _inner_why_not(rbd.components[node[0]], node[1])
    return _why_not(rbd, node)


def _why_not(rbd, node) -> Optional[str]:
    from repyability.rbd import _crews, _shutdowns
    from repyability.rbd.repairable_rbd import RepairableRBD

    if isinstance(rbd.components[node], RepairableRBD):
        return (
            "it is a nested RBD, whose components' spares are its own: "
            "give its components' shelves as (node, component)"
        )
    schedule = rbd._preventive.get(node)
    if schedule is not None and schedule.policy == "condition":
        return (
            "it is replaced on condition, whose replacement with the shelf "
            "empty is not simulated yet"
        )
    if node in rbd._member_group or node in rbd._outage_members:
        return (
            "it is renewed early (at its maintenance group's stops or at "
            "shutdowns), which a shelf is not simulated for yet"
        )
    if _crews._crews_limited(rbd):
        if node in rbd._inspection:
            return (
                "its failures are found by tests, and with repair crews a "
                "test's job waiting for its spare is not simulated yet"
            )
        if any(paused for _, paused in _shutdowns.holding(rbd, node)):
            return (
                "a shutdown pauses it, and with repair crews a job waiting "
                "for its spare through a pause is not simulated yet"
            )
    return None


def require(rbd, node, what: str) -> None:
    """Raise if component ``node`` cannot draw on a shelf (see
    ``why_not``); ``what`` names where the shelf was asked for."""
    from repyability.rbd import _requirements

    if not is_inner(rbd, node):
        _requirements._require_component(rbd, node, what)
    reason = why_not(rbd, node)
    if reason is not None:
        raise NotImplementedError(
            f"{what}: component {node!r} cannot draw on a shelf of spares "
            f"in the simulation, as {reason}."
        )


def _require_one_level(rbd, shelves: Shelves) -> None:
    """Refuse a shelf shared across levels, or across sub-systems (see
    the module's notes)."""
    level: Dict[Hashable, Any] = {}
    for key, name in shelves.of.items():
        here = key[0] if is_inner(rbd, key) else None  # type: ignore[index]
        if level.setdefault(name, here) != here:
            raise NotImplementedError(
                f"The shelf {name!r} is drawn on at two levels (the top "
                "level's components, or one sub-system's, each have their "
                "own loop, whose demands on one shelf would come out of "
                "time order): a shelf across levels is not simulated yet; "
                "give each level its own."
            )


def shelved(rbd, shelves: Shelves):
    """A copy of ``rbd`` whose simulations draw on ``shelves`` (none
    removes them). Its other methods are not for use: only the simulated
    routes that build it run it."""
    _require_one_level(rbd, shelves)
    out = copy.copy(rbd)
    out.__dict__.pop("_run_state", None)
    out._shelves = shelves if shelves.shelves else None
    return out


#: The diagram's settings a fleet's diagram of copies cannot carry: each
#: would couple its systems, or name their nodes.
_FLEET_REFUSES = (
    "repair_crews",
    "maintenance_groups",
    "ccf_groups",
    "shutdowns",
    "call_out",
)


def fleet_of(rbd, fleet: int):
    """One diagram of ``fleet`` copies of ``rbd``'s components, ``(j,
    node)`` for the ``j``-th system's, each between one input and one
    output (their structure does not change their events, as nothing
    couples a component to the others), so that a shelf can be drawn on
    across the fleet. Refused where something does couple them."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    args = rbd._init_args
    given = [name for name in _FLEET_REFUSES if args.get(name)]
    nested = [
        node
        for node, c in rbd.components.items()
        if isinstance(c, RepairableRBD)
    ]
    if given or nested:
        what = f"its {given[0]}" if given else f"the sub-system {nested[0]!r}"
        raise NotImplementedError(
            f"A fleet's shelf is simulated as one diagram of the systems' "
            f"components, which {what} would couple or nest: "
            "spares_stock(method='simulate') takes a fleet of diagrams "
            "with no repair crews, maintenance or common-cause groups, "
            "shutdowns, call-out crew or sub-systems. Give the fleet as "
            "one diagram yourself, with parts= pooling its shelves."
        )
    source, sink = ("fleet", "input"), ("fleet", "output")
    components, edges = {}, []
    for j in range(fleet):
        for node in rbd.components:
            given = args["components"][node]
            components[(j, node)] = (
                dict(given) if isinstance(given, dict) else copy.copy(given)
            )
            edges += [(source, (j, node)), ((j, node), sink)]
    return RepairableRBD(
        edges, components, input_node=source, output_node=sink
    )


def of(rbd) -> Optional[Shelves]:
    """The shelves a diagram's simulations draw on, or None."""
    return rbd.__dict__.get("_shelves")


class _Stock:
    """A shelf during one simulation: the arrival times of the last
    ``stock`` orders (fewer, early on, while spares are left on the
    shelf), and each demand's time and wait."""

    __slots__ = ("stock", "lead", "_orders", "times", "waits")

    def __init__(self, settings: Shelf):
        self.stock = settings.stock
        self.lead = settings.lead_time
        self._orders: deque = deque()
        self.times: List[float] = []
        self.waits: List[float] = []

    def take(self, t: float) -> float:
        """A demand at ``t``: when its spare is ready (``t``, with one on
        the shelf)."""
        orders = self._orders
        if self.stock == 0:
            ready = t + self.lead
        elif len(orders) < self.stock:
            ready = t
        else:
            ready = max(t, orders.popleft())
        if self.stock:
            orders.append(t + self.lead)
        self.times.append(t)
        self.waits.append(ready - t)
        return ready


@dataclass(frozen=True)
class Arrival:
    """The arrival of the spare a repair waited for, with repair crews
    (a dataclass, so equal to no node name): the job then queues for a
    crew. One for each node in a simulation."""

    node: Any


class Running(NamedTuple):
    """The shelves of one simulation: each by its name, and the one each
    node draws on, with whether its repairs take a spare; and as it runs,
    when the spare already taken for a node's next replacement is in
    (``ready``: a task put off for it, or a test's repair), when each
    shelved unit with a preventive task is due to fail (``ends``), the
    restoration each job waiting for its spare will end with once a crew
    takes it (``waiting``), and each node's ``Arrival``."""

    stocks: Dict[Hashable, _Stock]
    of: Dict[Hashable, Tuple[_Stock, bool]]
    ready: Dict[Hashable, float]
    ends: Dict[Hashable, float]
    waiting: Dict[Hashable, Any]
    arrivals: Dict[Hashable, Arrival]

    def arrival(self, node) -> Arrival:
        """``node``'s ``Arrival`` (the same one through the simulation, so
        that it keeps one place among events at one instant)."""
        found = self.arrivals.get(node)
        if found is None:
            found = self.arrivals[node] = Arrival(node)
        return found


def start(shelves: Optional[Shelves]) -> Optional[Running]:
    """The shelves at the start of a simulation, full."""
    if shelves is None:
        return None
    stocks = {name: _Stock(s) for name, s in shelves.shelves.items()}
    return Running(
        stocks,
        {
            node: (stocks[name], shelves.shelves[name].repairs)
            for node, name in shelves.of.items()
        },
        {},
        {},
        {},
        {},
    )


def inner(running: Optional[Running], node) -> Optional[Running]:
    """The shelves sub-system ``node``'s own loop draws on in a
    simulation: its components' (keyed ``(node, component)`` above), or
    None."""
    if running is None:
        return None
    of = {
        key[1]: held
        for key, held in running.of.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == node
    }
    if not of:
        return None
    return Running(running.stocks, of, {}, {}, {}, {})


class Record(NamedTuple):
    """One shelf over one simulation: its demands, those that waited
    (stock-outs), the time they waited in all, and the time it was out of
    stock (no spare on the shelf)."""

    demands: int
    stockouts: int
    waited: float
    out_of_stock: float


def _out_of_stock(stock: _Stock, end: float) -> float:
    """The time in ``[0, end)`` with no spare on the shelf: with ``S`` or
    more orders outstanding, each from its demand for the lead time."""
    if stock.stock == 0:
        return end
    if len(stock.times) < stock.stock:
        return 0.0
    placed = np.asarray(stock.times, dtype=float)
    edges = np.concatenate([placed, np.minimum(placed + stock.lead, end)])
    steps = np.concatenate(
        [np.ones(len(placed), int), -np.ones(len(placed), int)]
    )
    # At one time, arrivals before demands: an order that arrives as
    # another is placed leaves the count as it was.
    order = np.lexsort((steps, edges))
    edges, outstanding = edges[order], np.cumsum(steps[order])
    spans = np.diff(np.append(edges, end))
    return float(spans[outstanding >= stock.stock].sum())


def records(running: Optional[Running], end: float):
    """Each shelf's record of a simulation over ``[0, end)``."""
    if running is None:
        return None
    return {
        name: Record(
            len(stock.times),
            sum(1 for w in stock.waits if w > 0.0),
            float(sum(stock.waits)),
            _out_of_stock(stock, end),
        )
        for name, stock in running.stocks.items()
    }
