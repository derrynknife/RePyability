"""Shocks that fail some of a diagram's nodes, written out as nodes.

A shock process strikes at a constant rate, whatever the age of what it
strikes, and each shock fails each of the members it reaches with a
probability (the binomial failure rate model of Vesely and of Atwood).
Shocks at rate ``rate`` that fail each of ``m`` members independently with
probability ``p`` are, by thinning the process, one independent cause for
each set of members: the cause that fails exactly a set of ``j`` of them
strikes at rate ``rate * p**j * (1 - p)**(m - j)``. A cause that has not
struck by ``t``, with probability ``exp(-its rate * t)``, is an
exponential component in series with every member it fails, so each is
drawn as one: a node after each of its members, the appearances after the
first naming the first (one component in several places).

The members keep their own lives, of any kind and each its own, and their
own ages: the causes have no age. Every analysis of the diagram then
takes the shocks as it takes any repeated component.
"""

from itertools import combinations
from typing import Any, Collection, Dict, Hashable, Iterable, List, Tuple

import surpyval as surv

from repyability.utils.checks import number_or_nan

#: The most members a shock process may reach: it has a cause for each
#: set of them, ``2**m - 1``.
MAX_MEMBERS = 10


def _rate(value, name: str) -> float:
    number = number_or_nan(value)
    if not 0.0 <= number < float("inf"):
        raise ValueError(
            f"{name} must be a finite number, 0 or more: shocks per unit of "
            f"the diagram's time. Got {value!r}."
        )
    return number


def shock_nodes(
    edges: Iterable[Tuple[Hashable, Hashable]],
    nodes: Dict[Any, Any],
    members: Collection[Hashable],
    rate: float,
    p: float = 1.0,
    lethal_rate: float = 0.0,
    name: Hashable = "shock",
) -> Tuple[List[Tuple[Hashable, Hashable]], Dict[Any, Any]]:
    """A diagram's edges and nodes with a shock process written in.

    Shocks arrive at ``rate``, whatever the members' ages, and each fails
    each of ``members`` independently with probability ``p`` (the binomial
    failure rate model); shocks of a second kind, at ``lethal_rate``, fail
    every member. The members keep their own lives (of any kind, and each
    its own) for their other failures. The result is passed to
    [`NonRepairableRBD`][repyability.NonRepairableRBD], whose every
    analysis then takes the shocks, given each member's own age too
    (``sf_given_state``): they have none.

    Each set of members that a shock can fail exactly is a cause of its
    own, at rate ``rate * p**j * (1 - p)**(m - j)`` for a set of ``j`` of
    the ``m`` members (plus ``lethal_rate`` for the set of all of them),
    independent of the others. It is drawn as an exponential node after
    each member it fails, one component in several places, named ``(name,
    its members, the member it follows)``. The importance measures
    therefore name each cause, and a member's own entry is that of its
    own failures.

    Parameters
    ----------
    edges : iterable of (node, node)
        The diagram's edges.
    nodes : dict
        The diagram's nodes, as ``NonRepairableRBD`` takes them.
    members : collection of hashable
        The nodes the shocks reach: two or more, at most ten.
    rate : float
        Shocks per unit of time; 0 or more.
    p : float, optional
        The probability that a shock fails a member it reaches, in
        (0, 1]. By default 1: every shock fails every member, which is
        one exponential node in series with them.
    lethal_rate : float, optional
        Shocks per unit of time that fail every member; by default 0.
    name : hashable, optional
        The first item of the cause nodes' names, by default ``"shock"``:
        give each process of a diagram its own.

    Returns
    -------
    edges : list of (node, node)
    nodes : dict
        The diagram with the causes written in. The arguments are not
        changed.

    Raises
    ------
    ValueError
        If a member is not a component of ``nodes`` (or repeats another
        node), there are fewer than two members or more than ten, a member
        is named twice, a rate is negative or not a number, ``p`` is not
        in (0, 1], or a cause's name is already a node.

    Notes
    -----
    The shocks are for a non-repairable diagram. In a ``RepairableRBD`` a
    node in series is down and repaired as one component, where a shock
    leaves each member it failed to be repaired on its own.

    Examples
    --------
    Two pumps of different ages in parallel; a shock every 50,000 hours
    on average fails each with probability 0.6:

    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD, shock_nodes
    >>> pump = surv.Weibull.from_params([8000, 2.5])
    >>> edges, nodes = shock_nodes(
    ...     [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
    ...     {"p1": pump, "p2": pump},
    ...     members=["p1", "p2"], rate=2e-5, p=0.6,
    ... )
    >>> rbd = NonRepairableRBD(edges, nodes)
    >>> round(rbd.sf_given_state(2000, {"p1": 6000, "p2": 1000}), 5)
    0.95111
    """
    members = list(members)
    nodes = dict(nodes)
    edges = [(a, b) for a, b in edges]
    if len(set(members)) != len(members):
        raise ValueError(f"members names a node twice: {members!r}.")
    if not 2 <= len(members) <= MAX_MEMBERS:
        raise ValueError(
            f"members must name 2 to {MAX_MEMBERS} nodes (a shock process "
            "has a cause for each set of them), got "
            f"{len(members)}. For a shock that fails one node, give the "
            "node a life in series with it; for more members, shocks that "
            "fail every member (p=1) are one exponential node in series "
            "with them."
        )
    for member in members:
        if member not in nodes:
            raise ValueError(
                f"Member {member!r} is not one of the nodes given."
            )
        if not hasattr(nodes[member], "sf"):
            raise ValueError(
                f"Member {member!r} repeats node {nodes[member]!r}: name "
                "the node it repeats as the member, once."
            )
    rate = _rate(rate, "rate")
    lethal_rate = _rate(lethal_rate, "lethal_rate")
    chance = number_or_nan(p)
    if not 0.0 < chance <= 1.0:
        raise ValueError(
            "p, the probability that a shock fails a member it reaches, "
            f"must be a number above 0 and at most 1, got {p!r}."
        )
    m = len(members)
    causes = []
    for size in range(1, m + 1):
        strikes = rate * chance**size * (1.0 - chance) ** (m - size)
        if size == m:
            strikes += lethal_rate
        if strikes > 0.0:
            causes += [
                (struck, strikes) for struck in combinations(members, size)
            ]
    for member in members:
        after = [edge for edge in edges if edge[0] == member]
        edges = [edge for edge in edges if edge[0] != member]
        last = member
        for struck, strikes in causes:
            if member not in struck:
                continue
            node, first = (name, struck, member), (name, struck, struck[0])
            if node in nodes:
                raise ValueError(
                    f"{node!r} is already a node: give this shock process "
                    "a name of its own (name=...)."
                )
            nodes[node] = (
                surv.Exponential.from_params([strikes])
                if node == first
                else first
            )
            edges.append((last, node))
            last = node
        edges += [(last, to) for _, to in after]
    return edges, nodes
