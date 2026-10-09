"""The exact distribution of a system's capacity: how much it can deliver.

Each component has a capacity, the throughput it passes while it works; a
failed component passes nothing, and a component given no capacity limits
nothing (it passes whatever reaches it while it works). The system's
capacity is the most that can flow from the input to the output through
the working components, each passing at most its capacity: the diagram's
maximum flow. The edges carry any amount, so a series chain passes the
least of its members' capacities, and a parallel group the sum of theirs.
A k-out-of-n node keeps its meaning: it passes flow only while at least k
of its inputs are reached, as in the reliability analysis. So the system's
capacity is positive exactly when the system works.

The distribution is worked out on the diagram as the exact engine reduces
it (``modular.py``). A module's distribution has a closed form in its
members' -- the least of independent capacities for a series chain, their
sum for a parallel group, their sum while at least k work for a k-out-of-n
group -- which is the universal generating function of multi-state
systems. What is left (the core, e.g. a bridge) is worked out by
conditioning on its terms' capacities one at a time, in topological order.
By the max-flow min-cut theorem the flow there is the least total capacity
of a cut, so all that need be carried from one term to the next is the
running total of each cut not yet complete, the least complete total, and
whether each term still feeding one to come was reached. States that agree
on those are merged, so each distinct sub-problem is solved once.

A distribution is a pair of arrays: its levels, distinct and increasing,
and their probabilities, one row per level and one column per evaluation
(e.g. per time). Every probability is a sum of products of the components'
probabilities of working and of failing, with no differences, so each is
as precise as those it is built from, however close to 0 or 1. Levels are
rounded to 12 significant digits, so that totals that differ only by
rounding (``0.1 + 0.2`` and ``0.3``) are one level.
"""

import functools
import math
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np

from repyability.rbd.modular import (
    _SOURCE,
    KOON,
    NODE,
    PARALLEL,
    SERIES,
    FlowGraph,
)

Distribution = Tuple[np.ndarray, np.ndarray]

#: The state of the conditioning once some cut is known to carry nothing:
#: the capacity is then 0 whatever the rest.
_NOTHING = ("nothing",)


@functools.lru_cache(maxsize=1 << 16)
def tidy(value: float) -> float:
    """``value`` rounded to 12 significant digits (0 and infinity as
    they are)."""
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, 11 - math.floor(math.log10(abs(value))))


def merged(levels: np.ndarray, probabilities: np.ndarray) -> Distribution:
    """The distribution that puts ``probabilities`` (one row per entry of
    ``levels``) on ``levels``: equal levels (once tidied) merged, in
    increasing order, and levels that no evaluation can reach left out."""
    tidied = np.array([tidy(v) for v in np.ravel(levels)], dtype=float)
    unique, where = np.unique(tidied, return_inverse=True)
    # Each level's rows in their order (a stable sort), summed from its
    # first (#246): the sums ``np.add.at`` would make one row at a time.
    order = np.argsort(np.ravel(where), kind="stable")
    ranked = np.ravel(where)[order]
    first = np.flatnonzero(
        np.concatenate(([True], ranked[1:] != ranked[:-1]))
    )
    out = np.add.reduceat(probabilities[order], first, axis=0)
    reachable = np.any(out != 0.0, axis=1)
    if not reachable.all():
        unique, out = unique[reachable], out[reachable]
    return unique, out


def binary(
    capacity: float, works: np.ndarray, fails: np.ndarray
) -> Distribution:
    """A component that carries ``capacity`` while it works (with
    probabilities ``works``) and nothing once it has failed (``fails``)."""
    return merged(
        np.array([0.0, capacity]),
        np.vstack([np.ravel(fails), np.ravel(works)]),
    )


def node_distribution(
    capacity, works: np.ndarray, fails: np.ndarray
) -> Distribution:
    """A node's distribution from its capacity entry: a number, which it
    carries while it works, or a dict of the levels it works at and their
    probabilities given that it works."""
    if not isinstance(capacity, dict):
        return binary(capacity, works, fails)
    works = np.ravel(works)
    shares = np.array(list(capacity.values()), dtype=float)
    return merged(
        np.array([0.0] + list(capacity), dtype=float),
        np.vstack([np.ravel(fails), shares[:, None] * works[None, :]]),
    )


def working(distribution: Distribution) -> Distribution:
    """``distribution`` given that the node works: its levels above 0, in
    proportion. Where it cannot work at all, its highest level."""
    levels, probabilities = distribution
    up = levels > 0.0
    if not up.any():
        raise ValueError("A node that can never work cannot be forced to.")
    levels, probabilities = levels[up], probabilities[up]
    total = probabilities.sum(axis=0)
    top = np.zeros_like(probabilities)
    top[-1] = 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        given = np.where(total > 0.0, probabilities / total, top)
    return levels, given


def combine(a: Distribution, b: Distribution, how: Callable) -> Distribution:
    """The distribution of ``how(x, y)`` for independent ``x`` and ``y``
    distributed as ``a`` and ``b``."""
    levels = how(a[0][:, None], b[0][None, :])
    probabilities = (a[1][:, None, :] * b[1][None, :, :]).reshape(
        -1, a[1].shape[1]
    )
    return merged(levels, probabilities)


def _koon(members: List[Distribution], k: int) -> Distribution:
    """A k-out-of-n group's distribution: the sum of its members'
    capacities while at least ``k`` of them work, and 0 otherwise."""
    size = members[0][1].shape[1]
    # How many members work so far (up to k), and the distribution of
    # their total.
    totals: Dict[int, Distribution] = {0: (np.zeros(1), np.ones((1, size)))}
    for levels, probabilities in members:
        working = levels > 0.0
        updated: Dict[int, Distribution] = {}
        for count, total in totals.items():
            for works in (False, True):
                chosen = working == works
                if not chosen.any():
                    continue
                part = combine(
                    total, (levels[chosen], probabilities[chosen]), np.add
                )
                at = min(count + works, k)
                if at in updated:
                    both = updated[at]
                    part = merged(
                        np.concatenate([both[0], part[0]]),
                        np.vstack([both[1], part[1]]),
                    )
                updated[at] = part
        totals = updated
    below = [totals[c][1].sum(axis=0) for c in totals if c < k]
    levels, probabilities = totals.get(k, (np.zeros(0), np.zeros((0, size))))
    if below:
        levels = np.concatenate([[0.0], levels])
        probabilities = np.vstack([np.sum(below, axis=0), probabilities])
    return merged(levels, probabilities)


def _term_distributions(
    flow: FlowGraph, component: Callable[[Hashable], Distribution]
) -> Dict[int, Distribution]:
    """The distribution of every term the reduced diagram's vertices are
    built from, members before modules."""
    needed = set()
    stack = list(flow.vertices)
    while stack:
        i = stack.pop()
        if i not in needed:
            needed.add(i)
            if flow.terms[i][0] != NODE:
                stack.extend(flow.terms[i][1])
    out: Dict[int, Distribution] = {}
    # Each module comes after its members in the reduction's list.
    for i in sorted(needed):
        term = flow.terms[i]
        kind = term[0]
        if kind == NODE:
            out[i] = component(term[1])
            continue
        members = [out[c] for c in term[1]]
        if kind == KOON:
            out[i] = _koon(members, term[2])
            continue
        how = np.minimum if kind == SERIES else np.add
        assert kind in (SERIES, PARALLEL)
        value = members[0]
        for member in members[1:]:
            value = combine(value, member, how)
        out[i] = value
    return out


class _Step:
    """What conditioning on one vertex of the reduced diagram does to the
    carried state, worked out once from the structure alone."""

    def __init__(self):
        self.variable: Hashable = None
        # Where the capacity of the vertex's component is carried, if
        # another appearance of it came first (a repeated node); and for
        # each capacity carried on, where it was carried before (None for
        # this vertex's own).
        self.known: Optional[int] = None
        self.pending: tuple = ()
        # Whether the input feeds the vertex, where its other
        # predecessors' reach is carried, and its k; where each reach
        # carried on was carried before (the vertex's own is added last),
        # and whose they are.
        self.from_input = False
        self.pred_slots: tuple = ()
        self.k = 1
        self.reach_kept: tuple = ()
        self.reach: tuple = ()
        # The cuts: each open one after the step, as (where its total was
        # carried, or None if it opens here; whether the vertex is in it),
        # and where the totals of the cuts it completes were carried.
        self.carry: tuple = ()
        self.complete: tuple = ()


def _plan(flow: FlowGraph, gated: bool) -> List[_Step]:
    """The steps of the conditioning over the reduced diagram's vertices."""
    vertices = flow.vertices
    position = {v: i for i, v in enumerate(vertices)}
    end = len(vertices)
    variables = [
        ("node", flow.terms[v][1]) if flow.terms[v][0] == NODE else ("term", v)
        for v in vertices
    ]
    last_appearance: Dict[Hashable, int] = {}
    for i, variable in enumerate(variables):
        last_appearance[variable] = i
    # A vertex's reach is needed until its last successor.
    last_use = {v: -1 for v in vertices}
    for v, preds in zip(vertices, flow.preds):
        for u in preds:
            if u != _SOURCE:
                last_use[u] = max(last_use[u], position[v])
    for u in flow.sink_preds:
        if u != _SOURCE:
            last_use[u] = end
    cuts = flow.cuts()
    first = [position[cut[0]] for cut in cuts]
    last = [position[cut[-1]] for cut in cuts]
    containing: Dict[int, list] = {i: [] for i in range(end)}
    for c, cut in enumerate(cuts):
        for v in cut:
            containing[position[v]].append(c)

    steps = []
    pending: tuple = ()
    reach: tuple = ()
    open_cuts: tuple = ()
    for i, v in enumerate(vertices):
        step = _Step()
        variable = step.variable = variables[i]
        if variable in pending:
            step.known = pending.index(variable)
        after = [p for p in pending if last_appearance[p] > i]
        if variable not in pending and last_appearance[variable] > i:
            after.append(variable)
        step.pending = tuple(
            pending.index(p) if p in pending else None for p in after
        )
        pending = tuple(after)
        if gated:
            preds = flow.preds[i]
            step.from_input = _SOURCE in preds
            step.pred_slots = tuple(
                reach.index(u) for u in preds if u != _SOURCE
            )
            step.k = flow.k[i]
            kept = [u for u in reach if last_use[u] > i]
            step.reach_kept = tuple(reach.index(u) for u in kept)
            reach = step.reach = tuple(kept) + (v,)
        mine = set(containing[i])
        slot = {c: j for j, c in enumerate(open_cuts)}
        stays = [c for c in open_cuts if last[c] > i] + [
            c for c in sorted(mine) if first[c] == i and last[c] > i
        ]
        step.carry = tuple((slot.get(c), c in mine) for c in stays)
        step.complete = tuple(
            slot.get(c) for c in sorted(mine) if last[c] == i
        )
        open_cuts = tuple(stays)
        steps.append(step)
    return steps


def system_distribution(
    flow: FlowGraph,
    component: Callable[[Hashable], Distribution],
    size: int,
) -> Distribution:
    """The exact distribution of the capacity of the system whose reduced
    diagram is ``flow``, given each component's distribution (see the
    module docstring). ``size`` is the number of evaluations (columns)."""
    terms = _term_distributions(flow, component)
    gated = flow.sink_k > 1 or any(k > 1 for k in flow.k)
    steps = _plan(flow, gated)
    # A state: whether each vertex still feeding one to come was reached,
    # the capacities of the components still to appear again, each open
    # cut's running total, and the least complete total.
    states: Dict[tuple, np.ndarray] = {((), (), (), math.inf): np.ones(size)}
    for step, v in zip(steps, flow.vertices):
        levels, chances = terms[v]
        updated: Dict[tuple, np.ndarray] = {}
        for state, probability in states.items():
            if state is _NOTHING:
                _add(updated, _NOTHING, probability)
                continue
            reach, known, totals, least = state
            if step.known is None:
                choices = [
                    (level, probability * chance)
                    for level, chance in zip(levels.tolist(), chances)
                ]
            else:
                choices = [(known[step.known], probability)]
            for level, p in choices:
                reached = level > 0.0
                if gated and reached:
                    fed = step.from_input + sum(
                        reach[j] for j in step.pred_slots
                    )
                    reached = fed >= step.k
                flowing = level if reached else 0.0
                new_least = least
                for j in step.complete:
                    total = tidy((0.0 if j is None else totals[j]) + flowing)
                    if total < new_least:
                        new_least = total
                if new_least == 0.0:
                    _add(updated, _NOTHING, p)
                    continue
                new_totals = []
                for j, inside in step.carry:
                    total = 0.0 if j is None else totals[j]
                    if inside:
                        total = tidy(total + flowing)
                    # A cut already carrying at least the least complete
                    # total can no longer lower it.
                    new_totals.append(
                        math.inf if total >= new_least else total
                    )
                key = (
                    (
                        tuple(reach[j] for j in step.reach_kept) + (reached,)
                        if gated
                        else ()
                    ),
                    tuple(
                        level if j is None else known[j] for j in step.pending
                    ),
                    tuple(new_totals),
                    new_least,
                )
                _add(updated, key, p)
        states = updated

    final_reach = steps[-1].reach if steps else ()
    outcomes: list = []
    rows: list = []
    for state, probability in states.items():
        capacity = 0.0
        if state is not _NOTHING:
            reach, _, _, capacity = state
            if gated:
                fed = (_SOURCE in flow.sink_preds) + sum(
                    reach[final_reach.index(u)]
                    for u in flow.sink_preds
                    if u != _SOURCE
                )
                if fed < flow.sink_k:
                    capacity = 0.0
        outcomes.append(capacity)
        rows.append(probability)
    return merged(np.array(outcomes, dtype=float), np.array(rows))


def _add(states: Dict[tuple, np.ndarray], key: tuple, p: np.ndarray) -> None:
    if key in states:
        states[key] = states[key] + p
    else:
        states[key] = p
