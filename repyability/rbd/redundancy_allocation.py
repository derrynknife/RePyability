"""Redundancy allocation: how many copies of each node to fit.

The Redundancy Allocation Problem (RAP) chooses how many identical,
independent copies of each eligible node to fit in active parallel, to either

- **maximise** system reliability without the total cost exceeding a budget,
  or
- **minimise** total cost while meeting a system-reliability target.

A node with reliability ``p`` fitted as ``n`` active parallel copies has
reliability ``1 - (1 - p) ** n``; substituting that into the exact system
computation scores any allocation on any RBD structure, not only the textbook
series of subsystems. Adding a copy never lowers a coherent system's
reliability, which is what keeps the exact search small.

This is distinct from the reliability-allocation methods on ``RBD``, which
apportion a target reliability among existing components rather than choosing
integer numbers of units.

The search functions here work on plain sequences: ``evaluate`` maps a tuple
of unit counts (in ``costs`` order) to the system reliability, ``costs`` is
the per-copy cost of each node and ``caps`` the most copies allowed of each
(``math.inf`` for no limit). Each returns ``(reliability, cost, units)``.
"""

import math
from typing import Callable, Optional, Sequence, Tuple

# A search result: (system reliability, total cost, units per node).
Allocation = Tuple[float, float, Tuple[int, ...]]
# Maps a tuple of unit counts (in ``costs`` order; a count may be
# ``math.inf``, meaning unlimited copies) to the system reliability.
Evaluate = Callable[[Tuple[float, ...]], float]

#: The exact search gives up, with guidance, after examining this many
#: allocations, so an oversized problem fails in seconds instead of hanging.
EXACT_SEARCH_LIMIT = 500_000


def _slack(scale: float) -> float:
    # Absolute tolerance for comparing floating-point cost totals.
    return 1e-9 * max(1.0, abs(scale))


def _total_cost(costs: Sequence[float], units: Sequence[int]) -> float:
    return math.fsum(c * n for c, n in zip(costs, units))


def _check_search_size(examined: int) -> None:
    if examined > EXACT_SEARCH_LIMIT:
        raise ValueError(
            f"The exact search examined more than {EXACT_SEARCH_LIMIT:,} "
            "allocations without finishing. Use method='greedy', or bound "
            "the search with max_units (or a smaller budget)."
        )


def greedy(
    evaluate: Evaluate,
    costs: Sequence[float],
    caps: Sequence[float],
    budget: Optional[float] = None,
    target: Optional[float] = None,
) -> Allocation:
    """Greedy allocation: add the most cost-effective copy, one at a time.

    Starting from one unit of each node, repeatedly adds the copy with the
    largest gain in log-reliability per unit cost (the plain reliability
    gain per unit cost while the reliability is 0), skipping nodes at
    their cap and copies the budget cannot afford. It stops when the
    target is met, or when no affordable copy raises the reliability. Ties
    go to the node listed first. Fast and usually optimal or close to it,
    but not guaranteed optimal.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of unit counts (in ``costs`` order) to the system
        reliability.
    costs : Sequence[float]
        The cost of one copy of each node.
    caps : Sequence[float]
        The most units allowed of each node (``math.inf`` for no limit).
    budget : float, optional
        Never exceed this total cost (up to a tiny tolerance). The
        starting allocation, one of each, is not checked against it.
    target : float, optional
        Stop as soon as the reliability reaches this value.

    Returns
    -------
    tuple[float, float, tuple[int, ...]]
        ``(reliability, cost, units)`` of the allocation reached. With a
        ``target`` this may still fall short of it, if no copy helps any
        more; check the returned reliability.

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each:

    >>> import math
    >>> from repyability.rbd.redundancy_allocation import greedy
    >>> def evaluate(units):
    ...     return (1 - 0.1 ** units[0]) * (1 - 0.2 ** units[1])
    >>> reliability, cost, units = greedy(
    ...     evaluate, [1.0, 1.0], [math.inf, math.inf], budget=3
    ... )
    >>> units, round(reliability, 4), cost
    ((1, 2), 0.864, 3.0)
    """
    units = [1] * len(costs)
    reliability = evaluate(tuple(units))
    spent = math.fsum(costs)
    slack = _slack(budget) if budget is not None else 0.0
    while target is None or reliability < target:
        best = None  # (score, node index, reliability after the copy)
        for j, cost in enumerate(costs):
            if units[j] >= caps[j]:
                continue
            if budget is not None and spent + cost > budget + slack:
                continue
            units[j] += 1
            candidate = evaluate(tuple(units))
            units[j] -= 1
            if candidate <= reliability:
                continue
            if reliability > 0.0:
                score = (math.log(candidate) - math.log(reliability)) / cost
            else:
                score = candidate / cost
            if best is None or score > best[0]:
                best = (score, j, candidate)
        if best is None:
            break
        _, j, reliability = best
        units[j] += 1
        spent += costs[j]
    return reliability, _total_cost(costs, units), tuple(units)


def exact_max_reliability(
    evaluate: Evaluate,
    costs: Sequence[float],
    caps: Sequence[float],
    budget: float,
) -> Allocation:
    """The highest-reliability allocation costing at most ``budget``.

    An exhaustive search over the allocations within the budget and caps,
    of which only the *maximal* ones -- those that cannot take another copy
    of any node -- are evaluated: adding a copy never lowers a coherent
    system's reliability, so the optimum is always among them. Ties go to
    the cheaper allocation. Copies that add no reliability at all (of a
    node that is irrelevant to the system, or already perfect) are then
    removed, most expensive node first, so the result does not spend
    budget on nothing.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of unit counts (in ``costs`` order) to the system
        reliability.
    costs : Sequence[float]
        The cost of one copy of each node.
    caps : Sequence[float]
        The most units allowed of each node (``math.inf`` for no limit;
        the budget then bounds the search).
    budget : float
        The most the allocation may cost (up to a tiny tolerance). It must
        be at least ``sum(costs)``, the cost of one of each, which the
        caller checks.

    Returns
    -------
    tuple[float, float, tuple[int, ...]]
        ``(reliability, cost, units)`` of the optimal allocation.

    Raises
    ------
    ValueError
        If the search examines more than ``EXACT_SEARCH_LIMIT`` (500,000)
        allocations; use ``greedy``, tighter caps or a smaller budget.

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each:

    >>> import math
    >>> from repyability.rbd.redundancy_allocation import (
    ...     exact_max_reliability,
    ... )
    >>> def evaluate(units):
    ...     return (1 - 0.1 ** units[0]) * (1 - 0.2 ** units[1])
    >>> reliability, cost, units = exact_max_reliability(
    ...     evaluate, [1.0, 1.0], [math.inf, math.inf], budget=4
    ... )
    >>> units, round(reliability, 4), cost
    ((2, 2), 0.9504, 4.0)
    """
    k = len(costs)
    slack = _slack(budget)
    units = [1] * k
    best: Optional[Allocation] = None
    examined = 0

    def visit(i: int, remaining: float) -> None:
        nonlocal best, examined
        if i == k:
            examined += 1
            _check_search_size(examined)
            for j in range(k):
                if units[j] < caps[j] and costs[j] <= remaining + slack:
                    return
            reliability = evaluate(tuple(units))
            cost = _total_cost(costs, units)
            if (
                best is None
                or reliability > best[0]
                or (reliability == best[0] and cost < best[1])
            ):
                best = (reliability, cost, tuple(units))
            return
        most = min(caps[i], 1 + math.floor((remaining + slack) / costs[i]))
        for n in range(1, int(most) + 1):
            units[i] = n
            visit(i + 1, remaining - (n - 1) * costs[i])
        units[i] = 1

    visit(0, budget - math.fsum(costs))
    assert best is not None  # the all-ones allocation is always affordable
    return _drop_useless_copies(evaluate, costs, best)


def _drop_useless_copies(
    evaluate: Evaluate, costs: Sequence[float], best: Allocation
) -> Allocation:
    # Remove copies that add no reliability at all (of a node that is
    # irrelevant to the system, or already perfect), most expensive first,
    # so the reported allocation does not spend budget on nothing.
    reliability, _, found = best
    units = list(found)
    for j in sorted(range(len(costs)), key=lambda j: -costs[j]):
        while units[j] > 1:
            units[j] -= 1
            if evaluate(tuple(units)) < reliability:
                units[j] += 1
                break
    return reliability, _total_cost(costs, units), tuple(units)


def exact_min_cost(
    evaluate: Evaluate,
    costs: Sequence[float],
    caps: Sequence[float],
    target: float,
    start: Allocation,
) -> Allocation:
    """The cheapest allocation whose reliability is at least ``target``.

    ``start`` is any allocation already known to meet the target (e.g. the
    greedy solution): its cost bounds the search, which examines every
    allocation within the caps that costs no more than the best found so
    far. Ties in cost (up to a tiny tolerance) go to the more reliable
    allocation. If nothing better is found, ``start`` is returned.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of unit counts (in ``costs`` order) to the system
        reliability.
    costs : Sequence[float]
        The cost of one copy of each node.
    caps : Sequence[float]
        The most units allowed of each node (``math.inf`` for no limit;
        the cost of ``start`` then bounds the search).
    target : float
        The system reliability to reach.
    start : tuple[float, float, tuple[int, ...]]
        ``(reliability, cost, units)`` of an allocation that meets
        ``target``.

    Returns
    -------
    tuple[float, float, tuple[int, ...]]
        ``(reliability, cost, units)`` of the cheapest allocation found.

    Raises
    ------
    ValueError
        If the search examines more than ``EXACT_SEARCH_LIMIT`` (500,000)
        allocations; use ``greedy`` or tighter caps.

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each,
    starting from a known (but costlier) design that meets 90%:

    >>> import math
    >>> from repyability.rbd.redundancy_allocation import exact_min_cost
    >>> def evaluate(units):
    ...     return (1 - 0.1 ** units[0]) * (1 - 0.2 ** units[1])
    >>> start = (evaluate((3, 3)), 6.0, (3, 3))
    >>> reliability, cost, units = exact_min_cost(
    ...     evaluate, [1.0, 1.0], [math.inf, math.inf], 0.9, start
    ... )
    >>> units, round(reliability, 4), cost
    ((2, 2), 0.9504, 4.0)
    """
    k = len(costs)
    best = start
    slack = _slack(start[1])
    units = [1] * k
    # The cost of one copy of each node from position i onward.
    floor_from = [math.fsum(costs[i:]) for i in range(k + 1)]
    examined = 0

    def visit(i: int, spent: float) -> None:
        nonlocal best, examined
        if i == k:
            examined += 1
            _check_search_size(examined)
            reliability = evaluate(tuple(units))
            if reliability >= target and (
                spent < best[1] - slack
                or (spent <= best[1] + slack and reliability > best[0])
            ):
                best = (reliability, _total_cost(costs, units), tuple(units))
            return
        n = 1
        while n <= caps[i]:
            total = spent + n * costs[i]
            if total + floor_from[i + 1] > best[1] + slack:
                break
            units[i] = n
            visit(i + 1, total)
            n += 1
        units[i] = 1

    visit(0, 0.0)
    return best
