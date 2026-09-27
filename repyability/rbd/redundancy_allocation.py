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
what one copy of each node uses, and ``caps`` the most copies allowed of each
(``math.inf`` for no limit). A copy may use one resource (a number) or
several (a tuple, one amount per resource, e.g. cost and weight: Fyffe, Hines
& Lee, 1968), and a budget is then one limit per resource (``math.inf`` for a
resource that is not limited). The objective, and the tie-break between
equally reliable allocations, use the ``primary`` resource (the first by
default). Each search returns ``(reliability, cost, units)``, ``cost`` being
the total of the primary resource.

When every costed node is in series with the rest of the system, the system
reliability is the product of the costed nodes' reliabilities (times the
reliability of the rest), and ``series_front`` solves the problem exactly by
dynamic programming over the nodes instead.
"""

import bisect
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

# A search result: (system reliability, total primary cost, units per node).
Allocation = Tuple[float, float, Tuple[int, ...]]
# What one copy of a node uses: one amount, or one per resource.
Amount = Union[float, Sequence[float]]
# Maps a tuple of unit counts (in ``costs`` order; a count may be
# ``math.inf``, meaning unlimited copies) to the system reliability.
Evaluate = Callable[[Tuple[float, ...]], float]

#: The exact search gives up, with guidance, after examining this many
#: allocations, so an oversized problem fails in seconds instead of hanging.
EXACT_SEARCH_LIMIT = 500_000


#: The dynamic program gives up, with guidance, after holding this many
#: partial allocations in total over its stages.
SERIES_STATE_LIMIT = 2_000_000


def _slack(scale: float) -> float:
    # Absolute tolerance for comparing floating-point cost totals.
    return 1e-9 * max(1.0, abs(scale)) if math.isfinite(scale) else 0.0


def _as_tuple(value: Amount) -> Tuple[float, ...]:
    """One amount, or one per resource, as a tuple."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value),)
    return tuple(float(a) for a in value)


def _as_limits(
    budget: Union[None, float, Sequence[float]], m: int
) -> Tuple[float, ...]:
    """A budget as one limit per resource (``math.inf`` for none)."""
    if budget is None:
        return (math.inf,) * m
    return _as_tuple(budget)


def _vectors(
    costs: Sequence[Amount], budget: Union[None, float, Sequence[float]]
) -> Tuple[List[Tuple[float, ...]], Tuple[float, ...]]:
    """Costs as one tuple per node and the budget as one limit per
    resource (``math.inf`` where a resource is not limited)."""
    amounts = [_as_tuple(c) for c in costs]
    return amounts, _as_limits(budget, len(amounts[0]) if amounts else 1)


def _total_cost(
    costs: Sequence[Tuple[float, ...]], units: Sequence[int], r: int = 0
) -> float:
    return math.fsum(c[r] * n for c, n in zip(costs, units))


def _fits(
    used: Sequence[float],
    extra: Sequence[float],
    limits: Sequence[float],
    slacks: Sequence[float],
) -> bool:
    return all(
        u + e <= limit + slack
        for u, e, limit, slack in zip(used, extra, limits, slacks)
    )


def _check_search_size(examined: int) -> None:
    if examined > EXACT_SEARCH_LIMIT:
        raise ValueError(
            f"The exact search examined more than {EXACT_SEARCH_LIMIT:,} "
            "allocations without finishing. Use method='greedy', or bound "
            "the search with max_units (or a smaller budget)."
        )


def greedy(
    evaluate: Evaluate,
    costs: Sequence[Amount],
    caps: Sequence[float],
    budget: Union[None, float, Sequence[float]] = None,
    target: Optional[float] = None,
    primary: int = 0,
) -> Allocation:
    """Greedy allocation: add the most cost-effective copy, one at a time.

    Starting from one unit of each node, repeatedly adds the copy with the
    largest gain in log-reliability per unit of what it uses (the plain
    reliability gain while the reliability is 0), skipping nodes at their
    cap and copies the budget cannot afford. What a copy uses is its cost
    with one resource, or with a ``target`` its use of the primary
    resource; to maximise reliability within several limits it is the sum
    of its shares of the limits (the marginal-gain rule of this family of
    heuristics); copies that use none of that come first. It stops when
    the target is met, or when no affordable copy raises the reliability.
    Ties go to the node listed first. Fast and usually optimal or close to
    it, but not guaranteed optimal.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of unit counts (in ``costs`` order) to the system
        reliability.
    costs : Sequence
        What one copy of each node uses: a number, or a tuple of one amount
        per resource.
    caps : Sequence[float]
        The most units allowed of each node (``math.inf`` for no limit).
    budget : float or Sequence[float], optional
        Never exceed this total (one limit per resource, ``math.inf`` for
        none), up to a tiny tolerance. The starting allocation, one of
        each, is not checked against it.
    target : float, optional
        Stop as soon as the reliability reaches this value.
    primary : int, optional
        The resource a copy's use is measured in with a ``target``, and the
        one whose total is returned (the first, by default).

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
    amounts, limits = _vectors(costs, budget)
    slacks = [_slack(limit) for limit in limits]
    limited = [r for r, limit in enumerate(limits) if math.isfinite(limit)]
    if target is None and len(amounts[0]) > 1:
        # Several limits: a copy's use is its total share of them.
        use = [math.fsum(a[r] / limits[r] for r in limited) for a in amounts]
    else:
        use = [a[primary] for a in amounts]
    units = [1] * len(amounts)
    reliability = evaluate(tuple(units))
    spent = [math.fsum(a[r] for a in amounts) for r in range(len(limits))]
    while target is None or reliability < target:
        best = None  # (score, node index, reliability after the copy)
        for j, cost in enumerate(use):
            if units[j] >= caps[j]:
                continue
            if not _fits(spent, amounts[j], limits, slacks):
                continue
            units[j] += 1
            candidate = evaluate(tuple(units))
            units[j] -= 1
            if candidate <= reliability:
                continue
            if reliability > 0.0:
                gain = math.log(candidate) - math.log(reliability)
            else:
                gain = candidate
            # A copy that uses nothing that counts comes first.
            score = (1, gain) if cost == 0.0 else (0, gain / cost)
            if best is None or score > best[0]:
                best = (score, j, candidate)
        if best is None:
            break
        _, j, reliability = best
        units[j] += 1
        spent = [u + a for u, a in zip(spent, amounts[j])]
    return reliability, _total_cost(amounts, units, primary), tuple(units)


def exact_max_reliability(
    evaluate: Evaluate,
    costs: Sequence[Amount],
    caps: Sequence[float],
    budget: Union[float, Sequence[float]],
    primary: int = 0,
) -> Allocation:
    """The highest-reliability allocation within ``budget``.

    An exhaustive search over the allocations within the budget and caps,
    of which only the *maximal* ones -- those that cannot take another copy
    of any node -- are evaluated: adding a copy never lowers a coherent
    system's reliability, so the optimum is always among them. Ties go to
    the allocation using less of the primary resource. Copies that add no
    reliability at all (of a node that is irrelevant to the system, or
    already perfect) are then removed, the node using most of the primary
    resource first, so the result does not spend budget on nothing.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of unit counts (in ``costs`` order) to the system
        reliability.
    costs : Sequence
        What one copy of each node uses: a number, or a tuple of one amount
        per resource.
    caps : Sequence[float]
        The most units allowed of each node (``math.inf`` for no limit;
        the budget then bounds the search, so every node must use some of
        a limited resource or have a cap, which the caller checks).
    budget : float or Sequence[float]
        The most the allocation may use (one limit per resource,
        ``math.inf`` for none), up to a tiny tolerance. It must afford one
        of each node, which the caller checks.
    primary : int, optional
        The resource that breaks ties and whose total is returned (the
        first, by default).

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
    amounts, limits = _vectors(costs, budget)
    m = len(limits)
    k = len(amounts)
    slacks = [_slack(limit) for limit in limits]
    units = [1] * k
    best: Optional[Allocation] = None
    examined = 0

    def visit(i: int, remaining: List[float]) -> None:
        nonlocal best, examined
        if i == k:
            examined += 1
            _check_search_size(examined)
            for j in range(k):
                if units[j] < caps[j] and all(
                    amounts[j][r] <= remaining[r] + slacks[r] for r in range(m)
                ):
                    return
            reliability = evaluate(tuple(units))
            cost = _total_cost(amounts, units, primary)
            if (
                best is None
                or reliability > best[0]
                or (reliability == best[0] and cost < best[1])
            ):
                best = (reliability, cost, tuple(units))
            return
        most = caps[i]
        for r in range(m):
            if amounts[i][r] > 0.0 and math.isfinite(remaining[r]):
                most = min(
                    most,
                    1 + math.floor((remaining[r] + slacks[r]) / amounts[i][r]),
                )
        for n in range(1, int(most) + 1):
            units[i] = n
            visit(
                i + 1,
                [remaining[r] - (n - 1) * amounts[i][r] for r in range(m)],
            )
        units[i] = 1

    visit(
        0,
        [limits[r] - math.fsum(a[r] for a in amounts) for r in range(m)],
    )
    assert best is not None  # the all-ones allocation is always affordable
    return _drop_useless_copies(evaluate, amounts, best, primary)


def _drop_useless_copies(
    evaluate: Evaluate,
    costs: Sequence[Tuple[float, ...]],
    best: Allocation,
    primary: int = 0,
) -> Allocation:
    # Remove copies that add no reliability at all (of a node that is
    # irrelevant to the system, or already perfect), most expensive first,
    # so the reported allocation does not spend budget on nothing.
    reliability, _, found = best
    units = list(found)
    for j in sorted(range(len(costs)), key=lambda j: -costs[j][primary]):
        while units[j] > 1:
            units[j] -= 1
            if evaluate(tuple(units)) < reliability:
                units[j] += 1
                break
    return reliability, _total_cost(costs, units, primary), tuple(units)


def exact_min_cost(
    evaluate: Evaluate,
    costs: Sequence[Amount],
    caps: Sequence[float],
    target: float,
    start: Optional[Allocation],
    budget: Union[None, float, Sequence[float]] = None,
    primary: int = 0,
) -> Optional[Allocation]:
    """The cheapest allocation whose reliability is at least ``target``.

    Cheapest in the primary resource. ``start``, if given, is an allocation
    already known to meet the target (e.g. the greedy solution): its cost
    bounds the search, which examines every allocation within the caps and
    ``budget`` that costs no more than the best found so far. Ties in cost
    (up to a tiny tolerance) go to the more reliable allocation. If nothing
    better is found, ``start`` is returned (``None`` if there was none and
    no allocation within the limits meets the target).

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of unit counts (in ``costs`` order) to the system
        reliability.
    costs : Sequence
        What one copy of each node uses: a number, or a tuple of one amount
        per resource.
    caps : Sequence[float]
        The most units allowed of each node (``math.inf`` for no limit;
        the cost of ``start`` or the budget then bounds the search).
    target : float
        The system reliability to reach.
    start : tuple[float, float, tuple[int, ...]] or None
        ``(reliability, cost, units)`` of an allocation that meets
        ``target``, or ``None`` (the budget must then bound the search).
    budget : float or Sequence[float], optional
        Limits on the resources (one per resource, ``math.inf`` for none).
    primary : int, optional
        The resource to minimise (the first, by default).

    Returns
    -------
    tuple[float, float, tuple[int, ...]] or None
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
    amounts, limits = _vectors(costs, budget)
    m = len(limits)
    k = len(amounts)
    slacks = [_slack(limit) for limit in limits]
    best = start
    bound = math.inf if start is None else start[1]
    slack = _slack(bound)
    units = [1] * k
    # The use of one copy of each node from position i onward.
    floor_from = [
        [math.fsum(a[r] for a in amounts[i:]) for r in range(m)]
        for i in range(k + 1)
    ]
    examined = 0

    def visit(i: int, spent: List[float]) -> None:
        nonlocal best, examined, bound, slack
        if i == k:
            examined += 1
            _check_search_size(examined)
            reliability = evaluate(tuple(units))
            cost = spent[primary]
            if reliability >= target and (
                best is None
                or cost < bound - slack
                or (cost <= bound + slack and reliability > best[0])
            ):
                best = (
                    reliability,
                    _total_cost(amounts, units, primary),
                    tuple(units),
                )
                bound, slack = best[1], _slack(best[1])
            return
        n = 1
        while n <= caps[i]:
            total = [spent[r] + n * amounts[i][r] for r in range(m)]
            rest = [total[r] + floor_from[i + 1][r] for r in range(m)]
            if rest[primary] > bound + slack:
                break
            if any(rest[r] > limits[r] + slacks[r] for r in range(m)):
                break
            units[i] = n
            visit(i + 1, total)
            n += 1
        units[i] = 1

    visit(0, [0.0] * m)
    return best


def _pareto_front(states: list) -> list:
    """The states no other state dominates: each is ``(use, value, ...)``
    with ``use`` a tuple (lower is better in every resource) and ``value``
    higher-is-better. Exact duplicates keep one representative."""
    if not states:
        return []
    m = len(states[0][0])
    kept = []
    if m == 1:
        states.sort(key=lambda s: (s[0][0], -s[1]))
        best = -math.inf
        for s in states:
            if s[1] > best:
                kept.append(s)
                best = s[1]
        return kept
    if m == 2:
        # Sweep by the first resource, keeping a staircase of the second
        # resource against value for the states already kept: a state is
        # dominated when a kept state uses no more of the second resource
        # and is at least as good.
        states.sort(key=lambda s: (s[0][0], s[0][1], -s[1]))
        stair_use: List[float] = []
        stair_value: List[float] = []
        for s in states:
            use, value = s[0][1], s[1]
            q = bisect.bisect_right(stair_use, use)
            if q and stair_value[q - 1] >= value:
                continue
            kept.append(s)
            lo = bisect.bisect_left(stair_use, use)
            hi = lo
            while hi < len(stair_use) and stair_value[hi] <= value:
                hi += 1
            stair_use[lo:hi] = [use]
            stair_value[lo:hi] = [value]
        return kept
    states.sort(key=lambda s: (-s[1], s[0]))
    kept_use = np.empty((0, m))
    pending: List[Tuple[float, ...]] = []
    for s in states:
        if pending:
            kept_use = np.vstack([kept_use, np.array(pending)])
            pending = []
        if len(kept_use) and np.any(np.all(kept_use <= np.array(s[0]), 1)):
            continue
        kept.append(s)
        pending.append(s[0])
    return kept


def series_front(
    choices: Sequence[Sequence[Tuple[Amount, float]]],
    budget: Union[None, float, Sequence[float]] = None,
    bound: Optional[float] = None,
    primary: int = 0,
) -> List[Tuple[Tuple[float, ...], float, Tuple[int, ...]]]:
    """Every non-dominated allocation of a series of costed nodes, by
    dynamic programming.

    When every costed node is in series with the rest of the system, the
    system reliability is the product of the costed nodes' reliabilities
    times that of the rest, so its logarithm adds up node by node. Taking
    the nodes in turn, the program keeps only the partial allocations that
    no other dominates (using no more of any resource while being at least
    as reliable): the best completion of a dominated one is never better,
    so the final front holds the optimum for every budget and every
    target, which the caller picks from. Partial allocations that cannot be
    completed within the budget (or the ``bound`` on the primary resource)
    are dropped on the way. Exact for any amounts, not only integers, with
    one, two or more resources (two is the fastest case after one).

    Parameters
    ----------
    choices : Sequence
        For each node, its alternatives: ``(use, log_reliability)`` with
        ``use`` a number or a tuple of one amount per resource, e.g. ``n``
        copies using ``n`` times a copy's amounts, with
        ``log(1 - (1 - p) ** n)``.
    budget : float or Sequence[float], optional
        Limits on the resources (one per resource, ``math.inf`` for none).
    bound : float, optional
        The most of the primary resource worth using (e.g. the cost of an
        allocation known to meet a target).
    primary : int, optional
        The resource ``bound`` applies to.

    Returns
    -------
    list of tuple
        ``(use, log_reliability, picks)`` for every non-dominated
        allocation, ``picks`` holding the index of each node's chosen
        alternative.

    Raises
    ------
    ValueError
        If the program holds more than ``SERIES_STATE_LIMIT`` (2,000,000)
        partial allocations in total; use ``greedy`` or fewer copies.

    Examples
    --------
    Two nodes in series, 90% and 80% reliable, one to three copies each at
    one cost unit per copy, within a budget of 4:

    >>> import math
    >>> from repyability.rbd.redundancy_allocation import series_front
    >>> choices = [
    ...     [(n, math.log(1 - q**n)) for n in (1, 2, 3)] for q in (0.1, 0.2)
    ... ]
    >>> front = series_front(choices, budget=4)
    >>> [(u, round(math.exp(value), 4), picks) for u, value, picks in front]
    [((2.0,), 0.72, (0, 0)), ((3.0,), 0.864, (0, 1)), ((4.0,), 0.9504, (1, 1))]
    """
    amounts = [[_as_tuple(use) for use, _ in node] for node in choices]
    m = len(amounts[0][0])
    limits = _as_limits(budget, m)
    slacks = [_slack(limit) for limit in limits]
    k = len(choices)
    # The least of each resource the nodes from position i onward can use.
    least = [[0.0] * m for _ in range(k + 1)]
    for i in reversed(range(k)):
        least[i] = [
            min(a[r] for a in amounts[i]) + least[i + 1][r] for r in range(m)
        ]
    cap = [limits[r] + slacks[r] for r in range(m)]
    if bound is not None:
        cap[primary] = min(cap[primary], bound + _slack(bound))
    # Each stage holds (use, value, parent index, alternative index).
    stages: list = []
    front: list = [((0.0,) * m, 0.0, -1, -1)]
    held = 0
    for i in range(k):
        reserve = least[i + 1]
        candidates = []
        for index, (use, value, _, _) in enumerate(front):
            for pick, (extra, (_, gain)) in enumerate(
                zip(amounts[i], choices[i])
            ):
                new = tuple(u + e for u, e in zip(use, extra))
                if any(new[r] + reserve[r] > cap[r] for r in range(m)):
                    continue
                candidates.append((new, value + gain, index, pick))
        front = _pareto_front(candidates)
        held += len(front)
        if held > SERIES_STATE_LIMIT:
            raise ValueError(
                f"The dynamic program held more than {SERIES_STATE_LIMIT:,} "
                "partial allocations without finishing. Use "
                "method='greedy', or bound the search with max_units (or a "
                "smaller budget)."
            )
        stages.append(front)
    result = []
    for use, value, parent, pick in front:
        picks = [pick]
        for i in range(k - 2, -1, -1):
            _, _, grandparent, earlier = stages[i][parent]
            picks.append(earlier)
            parent = grandparent
        result.append((use, value, tuple(reversed(picks))))
    result.sort(key=lambda r: (r[0][primary], -r[1]))
    return result
