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

Allocation = Tuple[float, float, Tuple[int, ...]]
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
    """Add one copy at a time, always the copy with the largest gain in
    log-reliability per unit cost, until the target is met, or until no
    affordable copy improves the system.

    Fast and usually optimal or close to it, but not guaranteed optimal.
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

    Only *maximal* allocations -- ones that cannot take another copy of any
    node within the budget and caps -- are evaluated: adding a copy never
    lowers a coherent system's reliability, so the optimum is always among
    them. Ties go to the cheaper allocation.
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

    ``start`` is any allocation already known to meet the target (the greedy
    solution): its cost bounds the search, and every cheaper allocation is
    examined. Ties go to the more reliable allocation.
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
