"""Redundancy allocation: how many copies of each node to fit.

The Redundancy Allocation Problem (RAP) chooses how many independent copies
of each eligible node to fit in active parallel, to either

- **maximise** system reliability within a budget, or
- **minimise** the total cost while meeting a system-reliability target
  (and staying within a budget, if one is also given).

A node may have a choice of component types (*kinds*), each with its own
reliability and use of the resources (Fyffe, Hines & Lee, 1968). Its copies
are then all of one kind or, with mixing, any combination of kinds (Coit &
Smith, 1996). Copies of reliabilities ``p_1, ..., p_n`` in active parallel
have reliability ``1 - (1 - p_1) ... (1 - p_n)``, or, when ``k`` of them
must work, the probability that at least ``k`` do; other redundancy
strategies (such as cold standby) supply their own node reliability.
Substituting the node reliability into the exact system computation scores
any allocation on any RBD structure, not only the textbook series of
subsystems. A more reliable node never lowers a coherent system's
reliability, which is what keeps the exact search small.

This is distinct from the reliability-allocation methods on ``RBD``, which
apportion a target reliability among existing components rather than
choosing integer numbers of units.

The functions here work on plain sequences. What a copy uses is one amount
(a number) or one per resource (a tuple, e.g. cost and weight), and a budget
is one limit per resource (``math.inf`` for a resource that is not limited).
A node's kinds are ``(reliability, use)`` pairs, one copy of each; a node
without a choice has one kind. The combinations of copies a node may take
are its *designs* (``node_designs``). ``evaluate`` maps a tuple of node
reliabilities (in node order) to the system reliability. The objective, and
the tie-break between equally reliable allocations, use the ``primary``
resource (the first by default). Each search returns ``(reliability, cost,
designs)``: ``cost`` is the total of the primary resource, and ``designs``
holds the ``Design`` chosen for each node.

When every costed node is in series with the rest of the system, the system
reliability is the product of the costed nodes' reliabilities (times the
reliability of the rest), and ``series_front`` solves the problem exactly by
dynamic programming over the nodes instead.

For a repairable system, ``lowest_total_cost`` chooses the copies that
minimise a total cost of ownership instead: every copy costs money to buy
and to run, and the copies together save the cost of the system being down.
That total is not monotone in the number of copies, so its search bounds the
copies worth considering.
"""

import bisect
import functools
import math
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import minimize

# What one copy uses: one amount, or one per resource.
Amount = Union[float, Sequence[float]]
# One kind of copy of a node: (reliability of one copy, what it uses).
Kind = Tuple[float, Amount]
# A node's unreliability for a given number of copies of each kind.
NodeUnreliability = Callable[[Tuple[int, ...]], float]
# Maps a tuple of node reliabilities (in node order) to the system
# reliability.
Evaluate = Callable[[Tuple[float, ...]], float]

#: The exact search gives up, with guidance, after examining this many
#: allocations, so an oversized problem fails in seconds instead of hanging.
EXACT_SEARCH_LIMIT = 500_000

#: The dynamic program gives up, with guidance, after holding this many
#: partial allocations in total over its stages.
SERIES_STATE_LIMIT = 2_000_000

#: A node's designs are listed only up to this many, so a node with many
#: kinds and room for many copies fails fast with guidance.
DESIGN_LIMIT = 100_000


@dataclass(frozen=True)
class ComponentOption:
    """A candidate component type for a node, in redundancy allocation.

    Give a node a list of these in the ``costs`` of
    [`allocate_redundancy`][repyability.NonRepairableRBD.allocate_redundancy]
    to choose its copies among several types, e.g. a standard pump and a
    premium one, each with its own reliability and cost. The node's own
    model in the RBD is then not used: list it as an option too if it is a
    candidate.

    Parameters
    ----------
    name : Hashable
        What the result's ``mix`` calls this type. Distinct within a node.
    reliability : model or float
        The reliability of one copy: a model with an ``sf`` method (such
        as a fitted surpyval distribution), evaluated at the mission time,
        or a fixed probability of working, in [0, 1].
    cost : float or dict
        What one copy uses: a number (its cost) or a dict of resources,
        as for a node without options.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import ComponentOption
    >>> pumps = [
    ...     ComponentOption(
    ...         "standard", surv.Weibull.from_params([8000, 1.8]), cost=4000
    ...     ),
    ...     ComponentOption(
    ...         "premium", surv.Weibull.from_params([15000, 2.2]), cost=9000
    ...     ),
    ... ]
    >>> pumps[1].name, pumps[1].cost
    ('premium', 9000)
    """

    name: Hashable
    reliability: Any
    cost: Union[float, Dict[Hashable, float]]


class Design(NamedTuple):
    """One way to build a node: how many copies of each of its kinds."""

    #: What the copies use in all, one amount per resource.
    use: Tuple[float, ...]
    #: The node's reliability, ``1 - unreliability``.
    reliability: float
    #: The number of copies of each kind.
    counts: Tuple[int, ...]
    #: The probability that the node fails.
    unreliability: float
    #: The redundancy strategy, e.g. ``"active"`` or ``"cold"``.
    strategy: str = "active"


# A search result: (system reliability, total primary use, the design of
# each node).
Allocation = Tuple[float, float, Tuple[Design, ...]]


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


def redundancy_caps(
    nodes: Sequence[Hashable], max_units, named: str = "costs"
) -> List[float]:
    """Each node's most copies from ``max_units`` (an int for every node, a
    dict for some, or None), ``math.inf`` where unlimited; a ValueError for
    an invalid cap, or a dict naming nodes not among ``nodes`` (which the
    message says are ``named``)."""
    if max_units is None:
        return [math.inf] * len(nodes)
    if isinstance(max_units, dict):
        unknown = set(max_units) - set(nodes)
        if unknown:
            raise ValueError(
                f"max_units names node(s) not in {named}: "
                f"{sorted(map(str, unknown))}."
            )
        limits = {n: max_units.get(n, math.inf) for n in nodes}
    else:
        limits = {n: max_units for n in nodes}
    for node, limit in limits.items():
        if limit is math.inf:
            continue
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
            raise ValueError(
                f"max_units for node {node!r} must be an integer, got "
                f"{limit!r}."
            )
        if limit < 1:
            raise ValueError(
                f"max_units for node {node!r} must be at least 1, got "
                f"{limit!r}."
            )
    return [
        limits[n] if limits[n] is math.inf else int(limits[n]) for n in nodes
    ]


def _check_search_size(
    examined: int, bound: str = "max_units (or a smaller budget)"
) -> None:
    if examined > EXACT_SEARCH_LIMIT:
        raise ValueError(
            f"The exact search examined more than {EXACT_SEARCH_LIMIT:,} "
            "allocations without finishing. Use method='greedy', or bound "
            f"the search with {bound}."
        )


def _unreliability(qs: Sequence[float], counts: Sequence[int]) -> float:
    """The unreliability of copies in parallel, ``counts[j]`` of each kind
    of unreliability ``qs[j]``."""
    q = 1.0
    for q_kind, k in zip(qs, counts):
        if k:
            q *= q_kind**k
    return q


def active_unreliability(
    qs: Sequence[float], counts: Sequence[int], fewest: int = 1
) -> float:
    """The probability that fewer than ``fewest`` of the copies work, with
    ``counts[j]`` active copies of each kind of unreliability ``qs[j]``.

    Parameters
    ----------
    qs : Sequence[float]
        The unreliability of one copy of each kind.
    counts : Sequence[int]
        The number of copies of each kind.
    fewest : int, optional
        The number of copies that must work, by default 1.

    Returns
    -------
    float
        The node's unreliability.

    Examples
    --------
    Two of three copies of 90% reliability must work:

    >>> from repyability.rbd.redundancy_allocation import (
    ...     active_unreliability,
    ... )
    >>> round(active_unreliability([0.1], [3], fewest=2), 6)
    0.028
    """
    if fewest == 1:
        return _unreliability(qs, counts)
    # The distribution of the number of working copies, below ``fewest``.
    below = [1.0] + [0.0] * (fewest - 1)
    for q, k in zip(qs, counts):
        p = 1.0 - q
        for _ in range(k):
            for w in range(fewest - 1, 0, -1):
                below[w] = below[w] * q + below[w - 1] * p
            below[0] *= q
    return math.fsum(below)


def _useful_copies(q: float) -> int:
    """Copies of a kind of unreliability ``q`` past which more add nothing
    in double precision (one, for a perfect or a useless kind)."""
    if not 0.0 < q < 1.0:
        return 1
    # q ** n <= 2 ** -60 makes 1 - q ** n round to exactly 1.
    return math.ceil(60.0 * math.log(2.0) / -math.log(q)) + 1


def node_designs(
    kinds: Sequence[Kind],
    most: float,
    spare: Union[None, float, Sequence[float]] = None,
    mixing: bool = True,
    primary: int = 0,
    fewest: int = 1,
    unreliability: Optional[NodeUnreliability] = None,
    strategy: str = "active",
) -> List[Design]:
    """The designs of one node worth considering.

    Every combination of ``fewest`` to ``most`` copies -- all of one kind,
    or with ``mixing`` any mixture of kinds -- whose use fits within
    ``spare``, less those another design beats (using no more of any
    resource while being at least as reliable), sorted by use of the primary
    resource, the most reliable first among equals. Copies of a kind past
    the point where that many alone make the node perfectly reliable in
    double precision are not considered.

    Parameters
    ----------
    kinds : Sequence
        ``(reliability, use)`` of one copy of each kind, ``use`` a number or
        a tuple of one amount per resource.
    most : float
        The most copies in all (``math.inf`` for no limit).
    spare : float or Sequence[float], optional
        The most the node's copies may use of each resource (``math.inf``
        for no limit).
    mixing : bool, optional
        Whether copies of different kinds may be combined (the default), or
        must all be of one kind.
    primary : int, optional
        The resource to sort by (the first, by default).
    fewest : int, optional
        The fewest copies in all, by default 1.
    unreliability : callable, optional
        Maps the number of copies of each kind to the node's unreliability;
        it must not rise when a copy is added. By default the copies are
        active and ``fewest`` of them must work.
    strategy : str, optional
        The name recorded in each design, by default ``"active"``.

    Returns
    -------
    list of Design
        ``(use, reliability, counts, unreliability, strategy)`` for each
        design.

    Raises
    ------
    ValueError
        If there are more than ``DESIGN_LIMIT`` (100,000) combinations to
        consider; cap the copies (``most``) or turn off ``mixing``.

    Examples
    --------
    A standard part (90% reliable, cost 1) and a premium one (98%, cost 2),
    up to three copies for a spend of at most 5. Mixed designs are best at
    a spend of 4 and 5:

    >>> from repyability.rbd.redundancy_allocation import node_designs
    >>> for d in node_designs([(0.9, 1), (0.98, 2)], 3, spare=5):
    ...     print(d.use, d.counts, round(d.reliability, 5))
    (1.0,) (1, 0) 0.9
    (2.0,) (2, 0) 0.99
    (3.0,) (3, 0) 0.999
    (4.0,) (2, 1) 0.9998
    (5.0,) (1, 2) 0.99996

    Two of the copies must work (a 2-out-of-n node):

    >>> for d in node_designs([(0.9, 1)], 4, fewest=2):
    ...     print(d.counts, round(d.reliability, 4))
    (2,) 0.81
    (3,) 0.972
    (4,) 0.9963
    """
    amounts = [_as_tuple(a) for _, a in kinds]
    m = len(amounts[0])
    limits = _as_limits(spare, m)
    slacks = [_slack(limit) for limit in limits]
    qs = [1.0 - float(p) for p, _ in kinds]
    size = len(kinds)
    known: Dict[Tuple[int, ...], float] = {}

    def unreliable(counts: Tuple[int, ...]) -> float:
        if counts not in known:
            if unreliability is None:
                known[counts] = active_unreliability(qs, counts, fewest)
            else:
                known[counts] = float(unreliability(counts))
        return known[counts]

    def fits(n: int, j: int, used: Sequence[float]) -> bool:
        return all(
            used[r] + n * amounts[j][r] <= limits[r] + slacks[r]
            for r in range(m)
        )

    def alone(j: int, n: int) -> Tuple[int, ...]:
        return tuple(n if i == j else 0 for i in range(size))

    tops: List[float] = []
    if unreliability is None and fewest == 1:
        tops = [min(most, _useful_copies(q)) for q in qs]
    else:
        # The copies of each kind that alone make the node perfect.
        for j in range(size):
            n = fewest
            while n <= most and fits(n, j, [0.0] * m):
                if 1.0 - unreliable(alone(j, n)) == 1.0:
                    break
                n += 1
            tops.append(min(n, most))
    found: List[Design] = []
    counts = [0] * size

    def add() -> None:
        if len(found) >= DESIGN_LIMIT:
            raise ValueError(
                f"there are more than {DESIGN_LIMIT:,} designs to consider "
                f"({size} kinds, up to {most} copies). Give it a max_units"
                + (", or set mixing=False." if mixing and size > 1 else ".")
            )
        use = tuple(
            math.fsum(k * a[r] for k, a in zip(counts, amounts) if k)
            for r in range(m)
        )
        q = unreliable(tuple(counts))
        found.append(Design(use, 1.0 - q, tuple(counts), q, strategy))

    if mixing:

        def visit(j: int, total: int, used: List[float]) -> None:
            if j == size:
                if total >= fewest:
                    add()
                return
            n = 0
            while total + n <= most and n <= tops[j] and fits(n, j, used):
                counts[j] = n
                visit(
                    j + 1,
                    total + n,
                    [used[r] + n * amounts[j][r] for r in range(m)],
                )
                n += 1
            counts[j] = 0

        visit(0, 0, [0.0] * m)
    else:
        for j in range(size):
            n = fewest
            while n <= tops[j] and fits(n, j, [0.0] * m):
                counts[j] = n
                add()
                n += 1
            counts[j] = 0
    return best_designs(found, primary)


def best_designs(designs: List[Design], primary: int = 0) -> List[Design]:
    """The designs no other beats (using no more of any resource while
    being at least as reliable), sorted by use of the primary resource, the
    most reliable first among equals: e.g. to merge the designs of one node
    under several redundancy strategies.

    Parameters
    ----------
    designs : list of Design
        The candidate designs of one node.
    primary : int, optional
        The resource to sort by (the first, by default).

    Returns
    -------
    list of Design

    Examples
    --------
    >>> from repyability.rbd.redundancy_allocation import best_designs, Design
    >>> designs = [
    ...     Design((2.0,), 0.9, (2,), 0.1, "active"),
    ...     Design((2.0,), 0.95, (2,), 0.05, "cold"),
    ...     Design((1.0,), 0.8, (1,), 0.2, "active"),
    ... ]
    >>> [(d.use, d.strategy) for d in best_designs(designs)]
    [((1.0,), 'active'), ((2.0,), 'cold')]
    """
    front = _pareto_front(list(designs))
    front.sort(key=lambda d: (d.use[primary], -d.reliability))
    return front


def _bump(counts: Tuple[int, ...], j: int, by: int) -> Tuple[int, ...]:
    bumped = list(counts)
    bumped[j] += by
    return tuple(bumped)


def _moves(
    counts: Tuple[int, ...],
    amounts: Sequence[Tuple[float, ...]],
    most: float,
    mixing: bool,
) -> Iterator[Tuple[Tuple[int, ...], Tuple[float, ...]]]:
    """``(new counts, extra use)`` for each greedy step from a node design:
    one more copy of a kind (of its kind, without mixing), or the kind of
    one copy (with mixing) or of every copy (without) changed."""
    m = len(amounts[0])
    total = sum(counts)
    kinds = range(len(counts))
    if mixing:
        for j in kinds:
            if total < most:
                yield _bump(counts, j, 1), amounts[j]
        for i in kinds:
            for j in kinds:
                if counts[i] and j != i:
                    yield _bump(_bump(counts, i, -1), j, 1), tuple(
                        amounts[j][r] - amounts[i][r] for r in range(m)
                    )
        return
    i = next(j for j in kinds if counts[j])
    n = counts[i]
    if n < most:
        yield _bump(counts, i, 1), amounts[i]
    for j in kinds:
        if j != i:
            new = [0] * len(counts)
            new[j] = n
            yield tuple(new), tuple(
                n * amounts[j][r] - n * amounts[i][r] for r in range(m)
            )


def greedy(
    evaluate: Evaluate,
    kinds: Sequence[Sequence[Kind]],
    caps: Sequence[float],
    budget: Union[None, float, Sequence[float]] = None,
    target: Optional[float] = None,
    primary: int = 0,
    mixing: bool = True,
    fewest: Optional[Sequence[int]] = None,
    strategies: Optional[
        Sequence[Optional[Dict[str, NodeUnreliability]]]
    ] = None,
) -> Allocation:
    """Greedy allocation: take the most cost-effective step, one at a time.

    Starts from the fewest copies of each node (of its cheapest kind, under
    its first strategy), and repeatedly takes the step with the largest gain
    in log-reliability per unit of what it uses (the plain reliability gain
    while the reliability is 0), skipping nodes at their cap and steps the
    budget cannot afford. A step adds one copy to a node, changes the kind
    of one of its copies (with ``mixing``) or of all of them (without), or
    changes its redundancy strategy. What a step uses is measured in its
    cost with one resource, or with a ``target`` in the primary resource;
    to maximise reliability within several limits it is the sum of its
    shares of the limits (the marginal-gain rule of this family of
    heuristics); steps that use none of that come first. It stops when the
    target is met, or when no affordable step raises the reliability. Ties
    go to the node listed first. Fast and usually optimal or close to it,
    but not guaranteed optimal.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of node reliabilities (in node order) to the system
        reliability.
    kinds : Sequence
        For each node, its kinds: ``(reliability, use)`` of one copy, with
        ``use`` a number or a tuple of one amount per resource.
    caps : Sequence[float]
        The most copies allowed of each node (``math.inf`` for no limit).
    budget : float or Sequence[float], optional
        Never exceed this total (one limit per resource, ``math.inf`` for
        none), up to a tiny tolerance. The starting allocation is not
        checked against it.
    target : float, optional
        Stop as soon as the reliability reaches this value.
    primary : int, optional
        The resource a step's use is measured in with a ``target``, and the
        one whose total is returned (the first, by default).
    mixing : bool, optional
        Whether a node's copies may be of different kinds (the default).
    fewest : Sequence[int], optional
        The fewest copies of each node (such as the number that must work),
        by default one each.
    strategies : Sequence, optional
        For each node, ``None`` or its redundancy strategies: a dict of
        name to a function mapping the number of copies of each kind to the
        node's unreliability. By default every node is active, with
        ``fewest`` of its copies needed.

    Returns
    -------
    tuple[float, float, tuple of Design]
        ``(reliability, cost, designs)`` of the allocation reached. With a
        ``target`` this may still fall short of it, if no step helps any
        more; check the returned reliability.

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each:

    >>> import math
    >>> from repyability.rbd.redundancy_allocation import greedy
    >>> def evaluate(reliabilities):
    ...     return reliabilities[0] * reliabilities[1]
    >>> reliability, cost, designs = greedy(
    ...     evaluate, [[(0.9, 1)], [(0.8, 1)]], [math.inf, math.inf], budget=3
    ... )
    >>> [d.counts for d in designs], round(reliability, 4), cost
    ([(1,), (2,)], 0.864, 3.0)
    """
    amounts = [[_as_tuple(a) for _, a in node] for node in kinds]
    qs = [[1.0 - float(p) for p, _ in node] for node in kinds]
    size = len(kinds)
    least = [1] * size if fewest is None else list(fewest)
    ways: List[List[Tuple[str, NodeUnreliability]]] = []
    for i in range(size):
        given = None if strategies is None else strategies[i]
        if given is None:
            given = {
                "active": functools.partial(
                    active_unreliability, qs[i], fewest=least[i]
                )
            }
        ways.append(list(given.items()))
    m = len(amounts[0][0])
    limits = _as_limits(budget, m)
    slacks = [_slack(limit) for limit in limits]
    limited = [r for r, limit in enumerate(limits) if math.isfinite(limit)]
    shares = target is None and m > 1
    known: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}

    def measure(use: Sequence[float]) -> float:
        # Several limits: a step's use is its total share of them.
        if shares:
            return math.fsum(use[r] / limits[r] for r in limited)
        return use[primary]

    def unreliable(i: int, way: int, counts: Tuple[int, ...]) -> float:
        if (i, way, counts) not in known:
            known[i, way, counts] = float(ways[i][way][1](counts))
        return known[i, way, counts]

    counts: List[Tuple[int, ...]] = []
    for node_amounts, node_qs, n in zip(amounts, qs, least):
        first = min(
            range(len(node_qs)),
            key=lambda j: (measure(node_amounts[j]), node_qs[j]),
        )
        counts.append(
            tuple(n if j == first else 0 for j in range(len(node_qs)))
        )
    chosen = [0] * size
    unreliabilities = [unreliable(i, 0, counts[i]) for i in range(size)]
    reliabilities = [1.0 - q for q in unreliabilities]
    reliability = evaluate(tuple(reliabilities))
    spent = [
        math.fsum(
            a[r] * k
            for node_amounts, c in zip(amounts, counts)
            for a, k in zip(node_amounts, c)
            if k
        )
        for r in range(m)
    ]
    while target is None or reliability < target:
        # (score, node, strategy, counts, extra use, unreliability, system)
        best = None
        for i in range(size):
            steps = [
                (chosen[i], new, extra)
                for new, extra in _moves(
                    counts[i], amounts[i], caps[i], mixing
                )
            ] + [
                (way, counts[i], (0.0,) * m)
                for way in range(len(ways[i]))
                if way != chosen[i]
            ]
            for way, new, extra in steps:
                if not _fits(spent, extra, limits, slacks):
                    continue
                q = unreliable(i, way, new)
                if 1.0 - q <= reliabilities[i]:
                    continue
                trial = list(reliabilities)
                trial[i] = 1.0 - q
                candidate = evaluate(tuple(trial))
                if candidate <= reliability:
                    continue
                if reliability > 0.0:
                    gain = math.log(candidate) - math.log(reliability)
                else:
                    gain = candidate
                used = measure(extra)
                score = (1, gain) if used <= 0.0 else (0, gain / used)
                if best is None or score > best[0]:
                    best = (score, i, way, new, extra, q, candidate)
        if best is None:
            break
        _, i, chosen[i], counts[i], extra, q, reliability = best
        unreliabilities[i], reliabilities[i] = q, 1.0 - q
        spent = [u + e for u, e in zip(spent, extra)]
    designs = tuple(
        Design(
            tuple(
                math.fsum(k * a[r] for a, k in zip(amounts[i], counts[i]) if k)
                for r in range(m)
            ),
            reliabilities[i],
            counts[i],
            unreliabilities[i],
            ways[i][chosen[i]][0],
        )
        for i in range(size)
    )
    cost = math.fsum(d.use[primary] for d in designs)
    return reliability, cost, designs


def _reserve(menus: Sequence[Sequence[Design]], m: int) -> List[List[float]]:
    # The least of each resource the nodes from position i onward can use.
    reserve = [[0.0] * m for _ in range(len(menus) + 1)]
    for i in reversed(range(len(menus))):
        reserve[i] = [
            min(d.use[r] for d in menus[i]) + reserve[i + 1][r]
            for r in range(m)
        ]
    return reserve


def exact_max_reliability(
    evaluate: Evaluate,
    menus: Sequence[Sequence[Design]],
    budget: Union[float, Sequence[float]],
    primary: int = 0,
) -> Optional[Allocation]:
    """The highest-reliability allocation within ``budget``.

    An exhaustive search over the combinations of node designs within the
    budget, of which only the *maximal* ones -- those in which no node can
    afford a more reliable design -- are evaluated: a more reliable node
    never lowers a coherent system's reliability, so the optimum is always
    among them. Ties go to the allocation using less of the primary
    resource. Nodes whose designs add no reliability to the system (a node
    that is irrelevant to it) are then moved to cheaper designs, the node
    using most of the primary resource first, so the result does not spend
    budget on nothing.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of node reliabilities (in node order) to the system
        reliability.
    menus : Sequence
        Each node's designs, from ``node_designs``.
    budget : float or Sequence[float]
        The most the allocation may use (one limit per resource,
        ``math.inf`` for none), up to a tiny tolerance.
    primary : int, optional
        The resource that breaks ties and whose total is returned (the
        first, by default).

    Returns
    -------
    tuple[float, float, tuple of Design] or None
        ``(reliability, cost, designs)`` of the optimal allocation, or
        ``None`` if no combination of designs fits the budget.

    Raises
    ------
    ValueError
        If the search examines more than ``EXACT_SEARCH_LIMIT`` (500,000)
        allocations; use ``greedy``, tighter caps or a smaller budget.

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each:

    >>> from repyability.rbd.redundancy_allocation import (
    ...     exact_max_reliability,
    ...     node_designs,
    ... )
    >>> menus = [node_designs([(p, 1)], 3) for p in (0.9, 0.8)]
    >>> def evaluate(reliabilities):
    ...     return reliabilities[0] * reliabilities[1]
    >>> reliability, cost, designs = exact_max_reliability(
    ...     evaluate, menus, budget=4
    ... )
    >>> [d.counts for d in designs], round(reliability, 4), cost
    ([(2,), (2,)], 0.9504, 4.0)
    """
    k = len(menus)
    m = len(menus[0][0].use)
    limits = _as_limits(budget, m)
    slacks = [_slack(limit) for limit in limits]
    reserve = _reserve(menus, m)
    # For each design, the least extra use of a more reliable design of the
    # same node: if none of these fits, no more reliable design fits.
    steps = [
        [
            [
                step
                for step, _ in _pareto_front(
                    [
                        (tuple(b.use[r] - d.use[r] for r in range(m)), 0.0)
                        for b in menu
                        if b.reliability > d.reliability
                    ]
                )
            ]
            for d in menu
        ]
        for menu in menus
    ]
    picks = [0] * k
    best: Optional[Tuple[float, float, Tuple[int, ...]]] = None
    examined = 0

    def visit(i: int, remaining: List[float]) -> None:
        nonlocal best, examined
        if i == k:
            examined += 1
            _check_search_size(examined)
            for node in range(k):
                for step in steps[node][picks[node]]:
                    if all(
                        step[r] <= remaining[r] + slacks[r] for r in range(m)
                    ):
                        return
            designs = [menus[n][picks[n]] for n in range(k)]
            reliability = evaluate(tuple(d.reliability for d in designs))
            cost = math.fsum(d.use[primary] for d in designs)
            if (
                best is None
                or reliability > best[0]
                or (reliability == best[0] and cost < best[1])
            ):
                best = (reliability, cost, tuple(picks))
            return
        for j, design in enumerate(menus[i]):
            left = [remaining[r] - design.use[r] for r in range(m)]
            if any(left[r] + slacks[r] < reserve[i + 1][r] for r in range(m)):
                continue
            picks[i] = j
            visit(i + 1, left)
        picks[i] = 0

    visit(0, list(limits))
    if best is None:
        return None
    return _cheapen(evaluate, menus, best, limits, primary)


def _cheapen(
    evaluate: Evaluate,
    menus: Sequence[Sequence[Design]],
    best: Tuple[float, float, Tuple[int, ...]],
    limits: Sequence[float],
    primary: int = 0,
) -> Allocation:
    # Move nodes to cheaper designs that keep the system's reliability (a
    # node that is irrelevant to the system), the node using most of the
    # primary resource first, so the reported allocation does not spend
    # budget on nothing.
    reliability, _, found = best
    picks = list(found)
    k, m = len(menus), len(limits)
    slacks = [_slack(limit) for limit in limits]
    used = [
        math.fsum(menus[i][picks[i]].use[r] for i in range(k))
        for r in range(m)
    ]
    for i in sorted(range(k), key=lambda i: -menus[i][picks[i]].use[primary]):
        current = menus[i][picks[i]]
        for j, design in enumerate(menus[i]):
            if design.use[primary] >= current.use[primary]:
                break
            trial_use = [
                used[r] - current.use[r] + design.use[r] for r in range(m)
            ]
            if any(trial_use[r] > limits[r] + slacks[r] for r in range(m)):
                continue
            trial = [menus[n][picks[n]].reliability for n in range(k)]
            trial[i] = design.reliability
            if evaluate(tuple(trial)) >= reliability:
                picks[i], used = j, trial_use
                break
    designs = tuple(menus[i][picks[i]] for i in range(k))
    cost = math.fsum(d.use[primary] for d in designs)
    return reliability, cost, designs


def exact_min_cost(
    evaluate: Evaluate,
    menus: Sequence[Sequence[Design]],
    target: float,
    bound: float = math.inf,
    budget: Union[None, float, Sequence[float]] = None,
    primary: int = 0,
) -> Optional[Allocation]:
    """The cheapest allocation whose reliability is at least ``target``.

    Cheapest in the primary resource. The search examines every combination
    of node designs within the ``budget`` that costs no more than ``bound``
    (e.g. the cost of an allocation known to meet the target, such as the
    greedy solution) and than the best found so far. Ties in cost (up to a
    tiny tolerance) go to the more reliable allocation.

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of node reliabilities (in node order) to the system
        reliability.
    menus : Sequence
        Each node's designs, from ``node_designs``, sorted by use of the
        primary resource (as ``node_designs`` returns them).
    target : float
        The system reliability to reach.
    bound : float, optional
        The most of the primary resource worth using (by default no limit;
        the budget must then bound the search).
    budget : float or Sequence[float], optional
        Limits on the resources (one per resource, ``math.inf`` for none).
    primary : int, optional
        The resource to minimise (the first, by default).

    Returns
    -------
    tuple[float, float, tuple of Design] or None
        ``(reliability, cost, designs)`` of the cheapest allocation meeting
        the target, or ``None`` if none does within the bound and budget.

    Raises
    ------
    ValueError
        If the search examines more than ``EXACT_SEARCH_LIMIT`` (500,000)
        allocations; use ``greedy`` or tighter caps.

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each,
    with a known (but costlier) design that meets 90% costing 6:

    >>> from repyability.rbd.redundancy_allocation import (
    ...     exact_min_cost,
    ...     node_designs,
    ... )
    >>> menus = [node_designs([(p, 1)], 5) for p in (0.9, 0.8)]
    >>> def evaluate(reliabilities):
    ...     return reliabilities[0] * reliabilities[1]
    >>> reliability, cost, designs = exact_min_cost(
    ...     evaluate, menus, 0.9, bound=6
    ... )
    >>> [d.counts for d in designs], round(reliability, 4), cost
    ([(2,), (2,)], 0.9504, 4.0)
    """
    k = len(menus)
    m = len(menus[0][0].use)
    limits = _as_limits(budget, m)
    slacks = [_slack(limit) for limit in limits]
    reserve = _reserve(menus, m)
    slack = _slack(bound)
    picks = [0] * k
    best: Optional[Tuple[float, float, Tuple[int, ...]]] = None
    examined = 0

    def visit(i: int, spent: List[float]) -> None:
        nonlocal best, examined, bound, slack
        if i == k:
            examined += 1
            _check_search_size(examined)
            designs = [menus[n][picks[n]] for n in range(k)]
            reliability = evaluate(tuple(d.reliability for d in designs))
            cost = spent[primary]
            if (
                reliability >= target
                and cost <= bound + slack
                and (
                    best is None
                    or cost < bound - slack
                    or reliability > best[0]
                )
            ):
                total = math.fsum(d.use[primary] for d in designs)
                best = (reliability, total, tuple(picks))
                bound, slack = total, _slack(total)
            return
        for j, design in enumerate(menus[i]):
            used = [spent[r] + design.use[r] for r in range(m)]
            if used[primary] + reserve[i + 1][primary] > bound + slack:
                break
            if any(
                used[r] + reserve[i + 1][r] > limits[r] + slacks[r]
                for r in range(m)
            ):
                continue
            picks[i] = j
            visit(i + 1, used)
        picks[i] = 0

    visit(0, [0.0] * m)
    if best is None:
        return None
    reliability, cost, found = best
    return reliability, cost, tuple(menus[i][j] for i, j in enumerate(found))


def exact_front(
    evaluate: Evaluate,
    menus: Sequence[Sequence[Design]],
    budget: Union[None, float, Sequence[float]] = None,
    primary: int = 0,
) -> List[Allocation]:
    """Every non-dominated allocation within ``budget``: the trade-off
    between what the allocation uses and its reliability.

    An allocation is dominated when another uses no more of any resource
    and is at least as reliable; of allocations using the same and exactly
    as reliable, one is kept. Every combination of node designs within the
    budget is evaluated (``series_front`` is the fast alternative when the
    costed nodes are in series).

    Parameters
    ----------
    evaluate : callable
        Maps a tuple of node reliabilities (in node order) to the system
        reliability.
    menus : Sequence
        Each node's designs, from ``node_designs``.
    budget : float or Sequence[float], optional
        Limits on the resources (one per resource, ``math.inf`` for none).
    primary : int, optional
        The resource to sort by and whose total is returned (the first, by
        default).

    Returns
    -------
    list of tuple
        ``(reliability, cost, designs)`` for each non-dominated allocation,
        by increasing use of the primary resource (the most reliable first
        among equals).

    Raises
    ------
    ValueError
        If there are more than ``EXACT_SEARCH_LIMIT`` (500,000) allocations
        to examine; tighten the caps or the budget.

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each:

    >>> from repyability.rbd.redundancy_allocation import (
    ...     exact_front,
    ...     node_designs,
    ... )
    >>> menus = [node_designs([(p, 1)], 3) for p in (0.9, 0.8)]
    >>> def evaluate(reliabilities):
    ...     return reliabilities[0] * reliabilities[1]
    >>> for reliability, cost, designs in exact_front(evaluate, menus, 4):
    ...     print(cost, [d.counts for d in designs], round(reliability, 4))
    2.0 [(1,), (1,)] 0.72
    3.0 [(1,), (2,)] 0.864
    4.0 [(2,), (2,)] 0.9504
    """
    k = len(menus)
    m = len(menus[0][0].use)
    limits = _as_limits(budget, m)
    slacks = [_slack(limit) for limit in limits]
    reserve = _reserve(menus, m)
    picks = [0] * k
    found: list = []
    examined = 0

    def visit(i: int, remaining: List[float]) -> None:
        nonlocal examined
        if i == k:
            examined += 1
            _check_search_size(examined)
            designs = tuple(menus[n][picks[n]] for n in range(k))
            reliability = evaluate(tuple(d.reliability for d in designs))
            use = tuple(math.fsum(d.use[r] for d in designs) for r in range(m))
            found.append((use, reliability, designs))
            return
        for j, design in enumerate(menus[i]):
            left = [remaining[r] - design.use[r] for r in range(m)]
            if any(left[r] + slacks[r] < reserve[i + 1][r] for r in range(m)):
                continue
            picks[i] = j
            visit(i + 1, left)
        picks[i] = 0

    visit(0, list(limits))
    return _sorted_front(found, primary)


def _sorted_front(found: list, primary: int = 0) -> List[Allocation]:
    # The non-dominated (use, reliability, designs), as allocations by
    # increasing use of the primary resource.
    front = _pareto_front(found)
    front.sort(key=lambda f: (f[0][primary], -f[1]))
    return [
        (reliability, use[primary], designs)
        for use, reliability, designs in front
    ]


def _front_indices(use: np.ndarray, value: np.ndarray) -> np.ndarray:
    """The indices of the states that no other state dominates, in no
    particular order: state ``i`` uses ``use[i]`` (lower is better in every
    resource) and is worth ``value[i]`` (higher is better). Of exact
    duplicates, the first is kept."""
    n, m = use.shape
    if n == 0:
        return np.empty(0, dtype=int)
    index = np.arange(n)
    if m == 1:
        order = np.lexsort((index, -value, use[:, 0]))
        ordered = value[order]
        kept = np.empty(n, dtype=bool)
        kept[0] = True
        kept[1:] = ordered[1:] > np.maximum.accumulate(ordered)[:-1]
        return order[kept]
    if m == 2:
        # Sweep by the first resource, keeping a staircase of the second
        # resource against value for the states already kept: a state is
        # dominated when a kept state uses no more of the second resource
        # and is at least as good.
        order = np.lexsort((index, -value, use[:, 1], use[:, 0]))
        stair_use: List[float] = []
        stair_value: List[float] = []
        keep = []
        for position, (second, worth) in enumerate(
            zip(use[order, 1].tolist(), value[order].tolist())
        ):
            q = bisect.bisect_right(stair_use, second)
            if q and stair_value[q - 1] >= worth:
                continue
            keep.append(position)
            lo = bisect.bisect_left(stair_use, second)
            hi = lo
            while hi < len(stair_use) and stair_value[hi] <= worth:
                hi += 1
            stair_use[lo:hi] = [second]
            stair_value[lo:hi] = [worth]
        return order[np.array(keep, dtype=int)]
    keys = (index,) + tuple(use[:, r] for r in reversed(range(m)))
    order = np.lexsort(keys + (-value,))
    kept_use = np.empty((0, m))
    keep = []
    for position in order:
        if len(kept_use) and np.any(np.all(kept_use <= use[position], 1)):
            continue
        keep.append(position)
        kept_use = np.vstack([kept_use, use[position]])
    return np.array(keep, dtype=int)


def _pareto_front(states: list) -> list:
    """The states no other state dominates: each is ``(use, value, ...)``
    with ``use`` a tuple (lower is better in every resource) and ``value``
    higher-is-better. Exact duplicates keep the first."""
    if not states:
        return []
    use = np.array([s[0] for s in states], dtype=float)
    value = np.array([s[1] for s in states], dtype=float)
    return [states[i] for i in sorted(_front_indices(use, value))]


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
        ``use`` a number or a tuple of one amount per resource, e.g. one
        per design of the node from ``node_designs``.
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
    amounts = [
        np.array([_as_tuple(use) for use, _ in node], dtype=float)
        for node in choices
    ]
    gains = [np.array([g for _, g in node], dtype=float) for node in choices]
    m = amounts[0].shape[1]
    limits = _as_limits(budget, m)
    k = len(choices)
    # The least of each resource the nodes from position i onward can use.
    least = [np.zeros(m) for _ in range(k + 1)]
    for i in reversed(range(k)):
        least[i] = amounts[i].min(axis=0) + least[i + 1]
    cap = np.array([limit + _slack(limit) for limit in limits])
    if bound is not None:
        cap[primary] = min(cap[primary], bound + _slack(bound))
    uses = np.zeros((1, m))
    values = np.zeros(1)
    # Each stage holds, for each state kept, its parent in the previous
    # stage and the alternative it adds.
    stages = []
    held = 0
    for i in range(k):
        new_uses = uses[:, None, :] + amounts[i][None, :, :]
        new_values = values[:, None] + gains[i][None, :]
        fits = np.all(new_uses + least[i + 1] <= cap, axis=2)
        parents, picks = np.nonzero(fits)
        uses = new_uses[parents, picks]
        values = new_values[parents, picks]
        keep = _front_indices(uses, values)
        uses, values = uses[keep], values[keep]
        stages.append((parents[keep], picks[keep]))
        held += len(keep)
        if held > SERIES_STATE_LIMIT:
            raise ValueError(
                f"The dynamic program held more than {SERIES_STATE_LIMIT:,} "
                "partial allocations without finishing. Use "
                "method='greedy', or bound the search with max_units (or a "
                "smaller budget)."
            )
    result: List[Tuple[Tuple[float, ...], float, Tuple[int, ...]]] = []
    for final in range(len(values)):
        chosen = []
        state = final
        for parents, picks in reversed(stages):
            chosen.append(int(picks[state]))
            state = int(parents[state])
        result.append(
            (
                tuple(uses[final].tolist()),
                float(values[final]),
                tuple(reversed(chosen)),
            )
        )
    result.sort(key=lambda r: (r[0][primary], -r[1]))
    return result


# -- reliability-redundancy allocation -------------------------------------

# The system reliability, and its derivative with respect to each node's
# reliability, for a tuple of node reliabilities.
System = Callable[[Tuple[float, ...]], Tuple[float, Tuple[float, ...]]]
# What n copies of a node of component reliability r use: one amount per
# resource.
Use = Callable[[float, int], Tuple[float, ...]]


def reliability_redundancy(
    system: System,
    uses: Sequence[Use],
    budget: Sequence[float],
    bounds: Sequence[Tuple[float, float]],
    caps: Sequence[float],
) -> Tuple[float, Tuple[int, ...], Tuple[float, ...]]:
    """The reliability-redundancy allocation problem (RRAP): the number of
    copies of each node *and* the reliability of its components that
    maximise system reliability within the budget.

    A node of component reliability ``r`` fitted as ``n`` active copies has
    reliability ``1 - (1 - r) ** n``, and ``uses[i](r, n)`` is what its
    copies use (typically more for a higher ``r``, as a more reliable
    component costs more, and for more copies). This is a mixed-integer
    nonlinear problem (Tillman, Hwang & Kuo, 1977). It is solved exactly
    over the copies by branch and bound: every copy vector that fits the
    budget at the lowest reliabilities is bounded above by the system
    reliability with each node at the highest reliability it could afford
    alone, and the vectors are solved in decreasing order of that bound --
    each a continuous problem for the reliabilities, solved by SLSQP with
    the exact gradient from the lowest reliabilities -- until no bound
    beats the best found. The continuous solve finds a local optimum, which
    is the global one when the problem for fixed copies is convex, as for a
    series system with convex costs.

    The uses must not decrease with ``r`` or with ``n``: the bounds and the
    pruning rely on it.

    Parameters
    ----------
    system : callable
        Maps a tuple of node reliabilities to ``(reliability,
        derivatives)``: the system reliability and its derivative with
        respect to each node's reliability.
    uses : Sequence[callable]
        For each node, a function of ``(r, n)`` giving what ``n`` copies of
        component reliability ``r`` use: a tuple of one amount per resource.
    budget : Sequence[float]
        The most of each resource (``math.inf`` for no limit).
    bounds : Sequence[tuple[float, float]]
        The lowest and highest component reliability of each node, in
        [0, 1].
    caps : Sequence[float]
        The most copies of each node (``math.inf`` for no limit; the budget
        then bounds them).

    Returns
    -------
    tuple
        ``(reliability, units, component_reliabilities)``.

    Raises
    ------
    ValueError
        If even one copy of each node at its lowest reliability does not
        fit the budget, if a node's copies are not bounded, or if more than
        ``EXACT_SEARCH_LIMIT`` (500,000) copy vectors fit.

    Examples
    --------
    Two components in series. Each copy costs 10 to fit, plus
    ``(-1 / log(r)) ** 1.5`` for a component of reliability ``r``, which
    rises steeply as ``r`` nears 1; the budget is 100:

    >>> import math
    >>> from repyability.rbd.redundancy_allocation import (
    ...     reliability_redundancy,
    ... )
    >>> def system(p):
    ...     return p[0] * p[1], (p[1], p[0])
    >>> def use(r, n):
    ...     return (n * (10 + (-1 / math.log(r)) ** 1.5),)
    >>> reliability, units, r = reliability_redundancy(
    ...     system, [use, use], [100.0], [(0.5, 0.999)] * 2, [math.inf] * 2
    ... )
    >>> units, [round(x, 3) for x in r], round(reliability, 4)
    ((3, 3), [0.754, 0.754], 0.9705)
    """
    k = len(uses)
    m = len(budget)
    slacks = [_slack(limit) for limit in budget]
    lows = [float(lo) for lo, _ in bounds]
    highs = [float(hi) for _, hi in bounds]
    limited = [r for r, limit in enumerate(budget) if math.isfinite(limit)]

    def fits(total: Sequence[float]) -> bool:
        return all(total[r] <= budget[r] + slacks[r] for r in limited)

    # Every copy vector that fits at the lowest reliabilities (uses do not
    # decrease with the copies, so the search can stop at the first misfit).
    least = [uses[i](lows[i], 1) for i in range(k)]
    reserve = [[0.0] * m for _ in range(k + 1)]
    for i in reversed(range(k)):
        reserve[i] = [least[i][r] + reserve[i + 1][r] for r in range(m)]
    if not fits(reserve[0]):
        raise ValueError(
            "The budget cannot afford one copy of each node at its lowest "
            "reliability."
        )
    vectors: List[Tuple[Tuple[int, ...], List[Tuple[float, ...]]]] = []

    def visit(i: int, spent: List[float], chosen: list, low_uses: list):
        if i == k:
            if len(vectors) >= EXACT_SEARCH_LIMIT:
                raise ValueError(
                    f"More than {EXACT_SEARCH_LIMIT:,} copy vectors fit the "
                    "budget; bound the search with max_units."
                )
            vectors.append((tuple(chosen), list(low_uses)))
            return
        n = 1
        while n <= caps[i]:
            amounts = uses[i](lows[i], n)
            total = [spent[r] + amounts[r] for r in range(m)]
            if not fits([total[r] + reserve[i + 1][r] for r in range(m)]):
                break
            if n > 10_000:
                raise ValueError(
                    "A node could be copied without limit: more than 10,000 "
                    "copies fit the budget. Give it a max_units."
                )
            visit(i + 1, total, chosen + [n], low_uses + [amounts])
            n += 1

    visit(0, [0.0] * m, [], [])

    def node_reliabilities(r: Sequence[float], n: Sequence[int]) -> tuple:
        return tuple(1.0 - (1.0 - ri) ** ni for ri, ni in zip(r, n))

    def highest(i: int, n: int, others: Sequence[float]) -> float:
        # The highest reliability node i can afford with n copies, the
        # others using ``others`` (bisection: uses rise with r).
        def affordable(r: float) -> bool:
            amounts = uses[i](r, n)
            return fits([others[j] + amounts[j] for j in range(m)])

        if affordable(highs[i]):
            return highs[i]
        lo, hi = lows[i], highs[i]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if affordable(mid):
                lo = mid
            else:
                hi = mid
        return lo

    bounded = []
    for n, low_uses in vectors:
        total = [math.fsum(u[r] for u in low_uses) for r in range(m)]
        tops = [
            highest(i, n[i], [total[r] - low_uses[i][r] for r in range(m)])
            for i in range(k)
        ]
        bound = system(node_reliabilities(tops, n))[0]
        bounded.append((bound, n, tops))
    bounded.sort(key=lambda b: -b[0])

    def total_use(r: Sequence[float], n: Sequence[int]) -> List[float]:
        amounts = [uses[i](r[i], n[i]) for i in range(k)]
        return [math.fsum(a[j] for a in amounts) for j in range(m)]

    def solve(n: Tuple[int, ...]):
        def objective(r):
            reliability, derivatives = system(node_reliabilities(r, n))
            if reliability <= 0.0:
                return 1e300, np.zeros(k)
            gradient = [
                -derivatives[i]
                * n[i]
                * (1.0 - r[i]) ** (n[i] - 1)
                / reliability
                for i in range(k)
            ]
            return -math.log(reliability), np.array(gradient)

        def slack(r):
            used = total_use(r, n)
            return np.array([budget[j] - used[j] for j in limited])

        def slack_gradient(r):
            # Each node's use depends on its own reliability only, so one
            # finite difference per node (backward at the upper bound).
            columns = []
            for i in range(k):
                step = 1.5e-8 * max(1.0, abs(r[i]))
                if r[i] + step > highs[i]:
                    step = -step
                before = uses[i](r[i], n[i])
                after = uses[i](r[i] + step, n[i])
                columns.append(
                    [-(after[j] - before[j]) / step for j in limited]
                )
            return np.array(columns).T

        constraints = [{"type": "ineq", "fun": slack, "jac": slack_gradient}]
        # From the lowest reliabilities, which fit; if the solver ends
        # outside the budget, they are the answer.
        best = (system(node_reliabilities(lows, n))[0], tuple(lows))
        found = minimize(
            objective,
            np.array(lows, dtype=float),
            jac=True,
            bounds=list(zip(lows, highs)),
            constraints=constraints if limited else (),
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        r = tuple(float(x) for x in np.clip(found.x, lows, highs))
        if fits(total_use(r, n)):
            reliability = system(node_reliabilities(r, n))[0]
            if reliability > best[0]:
                best = (reliability, r)
        return best

    best: Optional[Tuple[float, Tuple[int, ...], Tuple[float, ...]]] = None
    for bound, n, tops in bounded:
        if best is not None and bound <= best[0]:
            break
        reliability, r = solve(n)
        if best is None or reliability > best[0]:
            best = (reliability, n, r)
    assert best is not None
    return best


# Maps a tuple of copy counts (in node order; ``math.inf`` for a node made
# perfect) to the system unavailability.
Unavailability = Callable[[Tuple[float, ...]], float]


def lowest_total_cost(
    unavailability: Unavailability,
    copy_costs: Sequence[float],
    downtime_cost: float,
    copy_gains: Sequence[Callable[[int], float]],
    max_units: Sequence[float],
    max_unavailability: Optional[float] = None,
    method: str = "exact",
    series: Optional[Sequence[Callable[[int], float]]] = None,
) -> Tuple[Tuple[int, ...], float, float]:
    """The numbers of copies with the lowest total cost.

    Node ``i`` fitted with ``n_i`` copies costs ``n_i * copy_costs[i]``
    (every copy is bought and run, the original included), and the system
    being down costs ``downtime_cost`` times its unavailability, so a
    design costs

    ```text
    total = sum_i n_i * copy_costs[i] + downtime_cost * unavailability(n)
    ```

    optionally subject to ``unavailability(n) <= max_unavailability``. More
    copies cost more and lower the unavailability, so the total is not
    monotone in them. Two bounds keep the exact search finite:

    - copies cost: a design no worse than a known one (the greedy solution)
      spends no more on copies than that design's total less the lowest
      possible downtime cost, which caps every node's copies;
    - useful copies (without ``max_unavailability``): the ``k + 1``-th copy
      of node ``i`` lowers the system unavailability by at most
      ``copy_gains[i](k)``, whatever the other nodes' copies. Those gains
      fall as ``k`` grows, so once ``downtime_cost * copy_gains[i](k)``
      is at most ``copy_costs[i]`` no further copy of node ``i`` lowers the
      total.

    ``method="exact"`` then searches every design within the caps by branch
    and bound, from the greedy solution: each partial design is bounded
    below by its copies' cost, one copy of each node still to choose, and
    the downtime cost with those nodes at their caps (the lowest they can
    make it), and is dropped if that cannot beat the best found (or, with
    ``max_unavailability``, if even that unavailability is too high). With
    ``series`` -- every node in series with the rest of the system, which
    then works when all of them and the rest do -- the system availability
    is the product of the nodes' and the rest's, and the best design is
    among those that no other beats by costing no more while being at
    least as available: ``series_front`` lists them by dynamic programming
    over the nodes, for any number of nodes.
    ``method="greedy"`` starts from one copy of each node, first (with
    ``max_unavailability``) adds the copy with the largest fall in
    unavailability per unit cost until the limit is met, then takes the
    single step (a copy more or, while the limit holds, a copy fewer) that
    lowers the total most, until none does.

    Parameters
    ----------
    unavailability : callable
        Maps a tuple of copy counts, one per node (``math.inf`` for a node
        made perfect), to the system unavailability. The system must be
        coherent: more copies never raise it.
    copy_costs : Sequence[float]
        What one copy of each node costs, finite and non-negative. A node
        whose copies cost nothing needs a finite ``max_units``.
    downtime_cost : float
        What the system costs down for the whole horizon, finite and
        non-negative.
    copy_gains : Sequence[callable]
        For each node, a function of ``k`` bounding above how much its
        ``k + 1``-th copy can lower the system unavailability, falling
        with ``k``: e.g. the node's own unavailability with ``k`` copies
        less that with ``k + 1``.
    max_units : Sequence[float]
        The most copies of each node (``math.inf`` for no limit).
    max_unavailability : float, optional
        The most system unavailability allowed, by default no limit.
    method : str, optional
        ``"exact"`` (the default) or ``"greedy"``.
    series : Sequence[callable], optional
        When every node is in series with the rest of the system, each
        node's unavailability as a function of its number of copies: the
        exact search is then a dynamic program.

    Returns
    -------
    tuple[tuple of int, float, float]
        ``(counts, copies_cost, unavailability)`` of the design found: its
        copies of each node, what the copies cost and its unavailability.

    Raises
    ------
    ValueError
        If a node's copies cost nothing and are not capped (so the search
        could add them without end), if ``max_unavailability`` cannot be
        met within the caps, or if the exact search examines more than
        ``EXACT_SEARCH_LIMIT`` (500,000) designs (with ``series``, holds
        more than ``SERIES_STATE_LIMIT``, 2,000,000, partial designs).

    Examples
    --------
    One node, a copy costing 10, available 90% of the time (so ``n``
    copies are down ``0.1 ** n`` of the time), with a downtime cost of
    1,000 over the horizon. A second copy saves 90 of downtime cost, a
    third 9:

    >>> import math
    >>> from repyability.rbd.redundancy_allocation import lowest_total_cost
    >>> counts, copies, u = lowest_total_cost(
    ...     lambda n: 0.1 ** n[0],
    ...     [10.0],
    ...     1000.0,
    ...     [lambda k: 0.1 ** k * 0.9],
    ...     [math.inf],
    ... )
    >>> counts, copies, round(copies + 1000.0 * u, 6)
    ((2,), 20.0, 30.0)
    """
    m = len(copy_costs)
    costs = [float(c) for c in copy_costs]
    caps = list(max_units)
    evaluations = 0

    def unavailable(counts: Sequence[float]) -> float:
        nonlocal evaluations
        evaluations += 1
        if method == "exact":
            _check_search_size(evaluations, "max_units")
        return float(unavailability(tuple(counts)))

    for i in range(m):
        if costs[i] <= 0.0 and caps[i] == math.inf:
            raise ValueError(
                f"A copy of node {i} costs nothing, so copies could be added "
                "without end: give it a cost or a cap."
            )
    feasible = (
        (lambda u: True)
        if max_unavailability is None
        else (lambda u: u <= max_unavailability)
    )
    best_possible = unavailable(caps)
    if not feasible(best_possible) or (
        max_unavailability is not None
        and best_possible == max_unavailability
        and math.inf in caps
    ):
        raise ValueError(
            f"The unavailability cannot be brought down to "
            f"{max_unavailability!r}: the lowest it can reach within the "
            f"caps is {best_possible!r}."
        )

    def copies_cost(counts: Sequence[int]) -> float:
        return math.fsum(n * c for n, c in zip(counts, costs))

    # The greedy solution: the answer for "greedy", and the incumbent the
    # exact search starts from.
    counts = [1] * m
    u = unavailable(counts)
    while not feasible(u):
        # The copy with the largest fall in unavailability per unit cost.
        step, step_u, step_value = -1, u, -math.inf
        for i in range(m):
            if counts[i] >= caps[i]:
                continue
            counts[i] += 1
            trial = unavailable(counts)
            counts[i] -= 1
            fall = u - trial
            if fall <= 0.0:
                continue
            value = fall / costs[i] if costs[i] > 0.0 else math.inf
            if value > step_value:
                step, step_u, step_value = i, trial, value
        if step < 0:
            raise ValueError(
                f"The unavailability cannot be brought down to "
                f"{max_unavailability!r}: another copy no longer lowers it."
            )
        counts[step] += 1
        u = step_u
    total = copies_cost(counts) + downtime_cost * u
    while True:
        # The single step (a copy more, or fewer) that lowers the total
        # most, keeping to the limit; ties go to the node listed first.
        moves = []
        for i in range(m):
            for by in (1, -1):
                if 1 <= counts[i] + by <= caps[i]:
                    design = list(counts)
                    design[i] += by
                    trial_u = unavailable(design)
                    if feasible(trial_u):
                        trial_total = (
                            copies_cost(design) + downtime_cost * trial_u
                        )
                        moves.append((trial_total, design, trial_u))
        if not moves:
            break
        move = min(moves, key=lambda move: move[0])
        if move[0] >= total - _slack(total):
            break
        total, counts, u = move
    if method == "greedy":
        return tuple(counts), copies_cost(counts), u

    # Caps for the exact search. Useful copies: without a limit on the
    # unavailability, a copy beyond the k-th of node i cannot pay for itself
    # once downtime_cost * copy_gains[i](k) <= copy_costs[i].
    if max_unavailability is None:
        for i in range(m):
            k = 1
            while k < caps[i] and downtime_cost * copy_gains[i](k) > costs[i]:
                k += 1
            caps[i] = min(caps[i], k)
    # Copies cost: a design no worse than the incumbent spends at most its
    # total, less the lowest downtime cost, on copies.
    floor = downtime_cost * best_possible
    one_each = math.fsum(costs)
    for i in range(m):
        if costs[i] > 0.0:
            spare = (total - floor - one_each) / costs[i]
            caps[i] = min(caps[i], 1 + math.floor(spare + 1e-9))
    best, best_total, best_u = list(counts), total, u
    slack = _slack(total)
    if series is not None:
        # The rest of the system's availability (every node perfect), and
        # the designs that no other beats on cost and availability at once:
        # the best is among them.
        others = 1.0 - unavailable([math.inf] * m)
        choices = []
        for i in range(m):
            node = []
            for n in range(1, int(caps[i]) + 1):
                q = series[i](n)
                node.append(
                    (n * costs[i], math.log1p(-q) if q < 1 else -math.inf)
                )
            choices.append(node)
        try:
            front = series_front(choices, bound=total - floor)
        except ValueError:
            raise ValueError(
                f"The dynamic program held more than {SERIES_STATE_LIMIT:,} "
                "partial designs without finishing. Use method='greedy', "
                "or bound the search with max_units."
            ) from None

        def approximate(point) -> float:
            # 1 - others * exp(value), keeping small values precise.
            use, value, _ = point
            u = (1.0 - others) - others * math.expm1(value)
            return use[0] + downtime_cost * u

        for point in sorted(front, key=approximate):
            if approximate(point) >= best_total + slack:
                break
            design = [j + 1 for j in point[2]]
            trial = unavailable(design)
            if not feasible(trial):
                continue
            trial_total = copies_cost(design) + downtime_cost * trial
            if trial_total < best_total - slack:
                best, best_total, best_u = design, trial_total, trial
        return tuple(best), copies_cost(best), best_u
    # The least the nodes from i on can cost: one copy each.
    rest = [math.fsum(costs[i:]) for i in range(m + 1)]
    picks = [1] * m

    def visit(i: int, spent: float) -> None:
        nonlocal best, best_total, best_u
        # The lowest downtime cost from here: nodes i on at their caps.
        low = downtime_cost * unavailable(picks[:i] + caps[i:])
        for n in range(1, int(caps[i]) + 1):
            cost = spent + n * costs[i]
            if cost + rest[i + 1] + low >= best_total - slack:
                break
            picks[i] = n
            later = i + 1
            trial = unavailable(picks[:later] + caps[later:])
            if not feasible(trial):
                continue
            bound = cost + rest[i + 1] + downtime_cost * trial
            if bound >= best_total - slack:
                continue
            if i + 1 == m:
                best, best_total, best_u = list(picks), bound, trial
            else:
                visit(i + 1, cost)
        picks[i] = 1

    visit(0, 0.0)
    return tuple(best), copies_cost(best), best_u
