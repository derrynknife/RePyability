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


def _check_search_size(examined: int) -> None:
    if examined > EXACT_SEARCH_LIMIT:
        raise ValueError(
            f"The exact search examined more than {EXACT_SEARCH_LIMIT:,} "
            "allocations without finishing. Use method='greedy', or bound "
            "the search with max_units (or a smaller budget)."
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
