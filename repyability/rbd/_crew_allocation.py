"""Redundancy allocation with limited repair crews: the copies of the
components that cost least in total when the copies queue for the crews
with every other job (``RepairableRBD.allocate_redundancy``).

Without limited crews a copy changes only its own node, so a design's
values are products and the search has bounds of its own
(``redundancy_allocation.lowest_total_cost``). With them a copy is one
more job for the crews: it can make another component wait, each copy's
own running cost depends on the whole design, and a copy more can even
raise the system's unavailability. So each design is worked out whole,
from the crews' Markov chain of its components and their copies
(``_crew_chain``'s ``copies``: a copy alike its component in rates and
priority, the chain counting how many are up, a node up while any is),
and the search keeps to two bounds that still hold:

- crews never help by being fewer: a job that waits only starts later, so
  every component is up no more than with a crew of its own, and the
  system (coherent) is down no less. A design's unavailability is at
  least that of the same design with unlimited crews, its components
  independent (``Unlimited``, a product form that costs nothing to work
  out);
- every copy is bought, and no running cost is negative: a design costs
  at least its copies' acquisition, the crews' salaries and its downtime
  with unlimited crews.

``method="greedy"`` takes the single step (a copy more or fewer) that
lowers the total most until none does. ``method="exact"`` starts from
that design, lists every design that costs less to buy than it costs in
all (less the least downtime any design can have), each with the least it
can cost by the two bounds, and solves their chains from the most
promising on, until none left can beat the best found. A design it needs
whose chain is too large is refused, by name.
"""

import math
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from repyability.rbd import _costs, _crew_chain, _crews, _shutdowns
from repyability.rbd._common import _crew_salaries
from repyability.rbd.redundancy_allocation import _check_search_size, _slack

#: What the result says of how its designs were worked out.
BASIS = (
    "Each design's long-run values from the repair crews' Markov chain, "
    "its copies queueing for the crews with the other components."
)


class Design:
    """A design's long-run values (see ``Evaluator``): its unavailability,
    what its copies cost to buy, and its running cost per unit time."""

    __slots__ = ("unavailability", "acquisition", "cost_rate")

    def __init__(
        self, unavailability: float, acquisition: float, cost_rate: float
    ):
        self.unavailability = unavailability
        self.acquisition = acquisition
        self.cost_rate = cost_rate


class Evaluator:
    """The long-run values of ``rbd`` with ``counts[i]`` copies of
    ``chosen[i]`` (the original among them), each design from the crews'
    chain of its components and their copies, and kept."""

    def __init__(self, rbd, chosen: Sequence[Hashable]):
        self.rbd = rbd
        self.chosen = list(chosen)
        self.rates = _crews._crew_chain_rates(rbd)
        for node in self.chosen:
            if node not in self.rates:
                # (A nested RBD has been refused; a junction is no job.)
                raise ValueError(
                    f"Node {node!r} is no job of the repair crews, so its "
                    "copies cannot be worked out with them."
                )
        self.designs: Dict[Tuple[int, ...], Design] = {}

    def _copies(self, counts: Sequence[int]) -> List[int]:
        """How many copies each of the crews' jobs has."""
        units = dict(zip(self.chosen, counts))
        return [int(units.get(node, 1)) for node in self.rates]

    def states(self, counts: Sequence[int]) -> int:
        """How many states the design's chain has (see
        ``_crew_chain.state_count``)."""
        return _crew_chain.state_count(
            [self.rbd._priority.get(node, 0.0) for node in self.rates],
            [math.isinf(repair) for _, repair in self.rates.values()],
            self.rbd.repair_crews,
            _crews._call_when(self.rbd),
            self._copies(counts),
        )

    def __call__(self, counts: Sequence[int]) -> Design:
        key = tuple(int(n) for n in counts)
        if key not in self.designs:
            self.designs[key] = self._design(key)
        return self.designs[key]

    def _design(self, counts: Tuple[int, ...]) -> Design:
        rbd = self.rbd
        if self.states(counts) > _crew_chain.MAX_STATES:
            units = dict(zip(self.chosen, counts))
            raise NotImplementedError(
                f"The search needs the design {units}, whose Markov chain "
                "of the components' states and the repair queue has more "
                f"than the {_crew_chain.MAX_STATES:,} states it is solved "
                "for. Bound the copies with max_units, or compare designs "
                "by simulation, with compare()."
            )
        nodes = list(self.rates)
        chain = _crew_chain.solve(
            nodes,
            [self.rates[node][0] for node in nodes],
            [self.rates[node][1] for node in nodes],
            [rbd._priority.get(node, 0.0) for node in nodes],
            rbd.repair_crews,
            _crews._call_when(rbd),
            (),
            self._copies(counts),
        )
        weights = chain.probabilities
        states = len(weights)
        # A node is up while any of its copies is.
        up = {
            node: chain.up[:, k].astype(float) for k, node in enumerate(nodes)
        }
        down = {node: 1.0 - p for node, p in up.items()}
        for node, component in rbd.components.items():
            if node not in up:
                # A nested RBD, with crews of its own.
                up[node] = np.full(
                    states, float(component.mean_availability())
                )
                down[node] = np.full(
                    states, float(component.mean_unavailability())
                )
        for node in rbd.in_or_out:
            up[node], down[node] = np.ones(states), np.zeros(states)
        unavailability = float(weights @ rbd._system_unreliability(up, down))
        # Every copy is run: a node's cost at its copies' availability,
        # as many times (it is linear in the availability).
        rate = _shutdowns.cost_rate(rbd) + _crew_salaries(rbd)
        if _crews._call_out_priced(rbd):
            rate += float(weights @ _crews._call_out_vector(rbd, chain))
        rate += rbd.downtime_cost_rate * unavailability
        copies = self._copies(counts)
        for k, node in enumerate(nodes):
            if node in rbd.costs:
                each = float(weights @ chain.working[:, k]) / copies[k]
                rate += copies[k] * _costs._node_cost_rate(rbd, node, each)
        for node in rbd.costs:
            if node not in self.rates:
                rate += _costs._node_cost_rate(
                    rbd, node, float(rbd.components[node].mean_availability())
                )
        acquisition = math.fsum(
            n * rbd.acquisition_costs.get(node, 0.0)
            for node, n in self._units(counts).items()
        )
        return Design(unavailability, acquisition, rate)

    def _units(self, counts: Sequence[int]) -> Dict[Hashable, int]:
        """Every component's number of copies."""
        units = dict(zip(self.chosen, counts))
        return {node: int(units.get(node, 1)) for node in self.rbd.components}


class Unlimited:
    """The unavailability of a design with a crew for every job: its
    components independent, each down ``lambda / (lambda + mu)`` of the
    time, a node down while all its copies are. No design is down less
    with the crews it has (see the module's notes)."""

    def __init__(self, rbd, chosen: Sequence[Hashable]):
        self.rbd = rbd
        self.chosen = list(chosen)
        rates = _crews._crew_chain_rates(rbd)
        self.down: Dict[Hashable, float] = {}
        for node, component in rbd.components.items():
            if node in rates:
                life, repair = rates[node]
                self.down[node] = (
                    0.0 if math.isinf(repair) else life / (life + repair)
                )
            else:
                self.down[node] = float(component.mean_unavailability())

    def __call__(self, counts: Sequence[float]) -> float:
        """With ``counts[i]`` copies of ``chosen[i]`` (``inf``: perfect)."""
        down = dict(self.down)
        for node, n in zip(self.chosen, counts):
            down[node] = 0.0 if n == math.inf else down[node] ** n
        q = {node: np.array([value]) for node, value in down.items()}
        p = {node: 1.0 - value for node, value in q.items()}
        for node in self.rbd.in_or_out:
            p[node], q[node] = np.ones(1), np.zeros(1)
        return float(np.ravel(self.rbd._system_unreliability(p, q))[0])


def search(
    rbd,
    chosen: Sequence[Hashable],
    caps: Sequence[float],
    max_unavailability: Optional[float],
    method: str,
    present: float,
    worth: float,
) -> Tuple[Tuple[int, ...], Design]:
    """The copies of each node in ``chosen`` (at most ``caps``) with the
    lowest total cost over a horizon whose present value is ``present``,
    a unit time of the system up worth ``worth`` (see the module's
    notes), and the design's values. With ``max_unavailability``, among
    the designs within it."""
    evaluate = Evaluator(rbd, chosen)
    unlimited = Unlimited(rbd, chosen)
    prices = [rbd.acquisition_costs.get(node, 0.0) for node in chosen]
    for node, price, cap in zip(chosen, prices, caps):
        if price <= 0.0 and cap == math.inf:
            raise ValueError(
                f"A copy of {node!r} costs nothing to buy, so with "
                "limited repair crews the search, which is bounded by "
                "what the copies cost to buy, could add copies without "
                "end: give it an acquisition_cost or a max_units."
            )
    lowest = unlimited(caps)
    if max_unavailability is not None and lowest > max_unavailability:
        raise ValueError(
            "The unavailability cannot be brought down to "
            f"{max_unavailability!r}: the lowest it can reach within the "
            f"caps, even with a crew for every job, is {lowest!r}."
        )
    # What a unit of unavailability costs over the horizon, and what the
    # other components cost to buy and the crews are paid whatever the
    # design.
    downtime = present * (rbd.downtime_cost_rate + worth)
    fixed = present * _crew_salaries(rbd) + math.fsum(
        price
        for node, price in rbd.acquisition_costs.items()
        if node not in chosen
    )

    def total(design: Design) -> float:
        return design.acquisition + present * (
            design.cost_rate + worth * design.unavailability
        )

    def feasible(design: Design) -> bool:
        return (
            max_unavailability is None
            or design.unavailability <= max_unavailability
        )

    best: Optional[Tuple[float, Tuple[int, ...]]] = None
    try:
        counts, design = _greedy(evaluate, prices, caps, total, feasible)
        best = (total(design), counts)
    except ValueError:
        # No design within the limit by single steps.
        if method == "greedy" or math.inf in caps:
            raise
    if method == "greedy":
        assert best is not None
        return best[1], evaluate(best[1])
    # The designs that could beat it: those whose copies cost less to buy
    # than it costs in all, less the least downtime; each with the least
    # it can cost, its copies bought and its downtime with a crew for
    # every job.
    limit = math.inf if best is None else best[0]
    start = (1,) * len(chosen)
    bought = {start: math.fsum(prices)}
    frontier = [start]
    candidates = []
    while frontier:
        counts = frontier.pop()
        _check_search_size(len(bought), "max_units")
        u = unlimited(counts)
        least = fixed + bought[counts] + downtime * u
        if max_unavailability is None or u <= max_unavailability:
            candidates.append((least, counts))
        for i, price in enumerate(prices):
            if counts[i] < caps[i]:
                more = counts[:i] + (counts[i] + 1,) + counts[i + 1 :]
                cost = bought[counts] + price
                if more not in bought and (
                    fixed + cost + downtime * lowest < limit - _slack(limit)
                ):
                    bought[more] = cost
                    frontier.append(more)
    # The most promising first, until none left can beat the best found.
    for least, counts in sorted(candidates):
        if best is not None and least >= best[0] - _slack(best[0]):
            break
        design = evaluate(counts)
        if feasible(design) and (best is None or total(design) < best[0]):
            best = (total(design), counts)
    if best is None:
        raise ValueError(
            "The unavailability cannot be brought down to "
            f"{max_unavailability!r} within the caps with "
            f"{rbd.repair_crews} repair crew(s): every design waits too "
            "long for one. Give more crews, or a lower min_availability."
        )
    return best[1], evaluate(best[1])


def _greedy(
    evaluate: Evaluator,
    prices: Sequence[float],
    caps: Sequence[float],
    total: Callable[[Design], float],
    feasible: Callable[[Design], bool],
) -> Tuple[Tuple[int, ...], Design]:
    """From one of each: first, to a limit on the unavailability, the
    copy with the largest fall in it per unit bought, until it is met;
    then the single step (a copy more, or fewer while the limit holds)
    that lowers the total most, until none does (as
    ``lowest_total_cost``'s greedy method)."""
    m = len(prices)
    counts = (1,) * m

    def stepped(i: int, by: int) -> Tuple[int, ...]:
        return counts[:i] + (counts[i] + by,) + counts[i + 1 :]

    design = evaluate(counts)
    while not feasible(design):
        step, value = None, -math.inf
        for i in range(m):
            if counts[i] >= caps[i]:
                continue
            fall = (
                design.unavailability - evaluate(stepped(i, 1)).unavailability
            )
            if fall <= 0.0:
                continue
            worth = fall / prices[i] if prices[i] > 0.0 else math.inf
            if worth > value:
                step, value = stepped(i, 1), worth
        if step is None:
            raise ValueError(
                "The unavailability cannot be brought down to the limit: "
                "another copy no longer lowers it. Give more repair "
                "crews, or a lower min_availability."
            )
        counts, design = step, evaluate(step)
    while True:
        moves = [
            (total(evaluate(stepped(i, by))), stepped(i, by))
            for i in range(m)
            for by in (1, -1)
            if 1 <= counts[i] + by <= caps[i]
            and feasible(evaluate(stepped(i, by)))
        ]
        if not moves:
            break
        cost, move = min(moves)
        if cost >= total(design) - _slack(total(design)):
            break
        counts, design = move, evaluate(move)
    return counts, design
