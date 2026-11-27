"""How much a network delivers between its two terminals: the maximum
flow through its working links, each carrying up to its capacity.

A link given a capacity carries at most that amount while it works (either
way in an undirected network, though not both ways at once; from its first
node to its second in a directed one), a link given none limits nothing,
and a failed link, or a link at a failed node, carries nothing. The
delivered amount is the maximum flow from the source to the target, found
by shortest augmenting paths (Edmonds and Karp) on the network's own
graph, for one state of the elements (each working or failed).

The exact distribution enumerates the states that matter (``_tree``): the
elements are decided one at a time, and a branch ends as soon as the flow
through the elements known to work equals the flow with the undecided ones
working too, since nothing still to be decided can then change it. The
element decided next is one the larger flow passes through, so its failure
is a different flow. Each branch's end is a set of elements working and a
set failed, with the capacity they give: its probability is the product of
theirs, so the distribution is a sum of products with no differences, as
precise as the elements' probabilities. The branches depend on the
structure and the capacities alone, so they are found once and evaluated
at any probabilities (at each time, or in the long run). A network of more
ends than ``network.MAX_CAPACITY_STATES`` refuses, pointing to the
simulation of states: each element drawn working or failed with its
probability, and the maximum flow of each distinct state drawn.

Levels are rounded to 12 significant digits (``capacity.tidy``), as a
diagram's are, and a flow below that share of the capacities' total is
taken as none.
"""

import math
from collections import deque
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd.capacity import tidy
from repyability.rbd.results import CapacityDistribution
from repyability.rbd.routes import Refused
from repyability.utils.checks import nonnegative_times, one_of
from repyability.utils.checks import seed as check_seed
from repyability.utils.checks import simulation_options

# An element's state while the states are enumerated.
_UNDECIDED, _WORKS, _FAILED = 0, 1, 2

#: The simulated states by default.
MC_SAMPLES = 10_000


class _Graph:
    """The network as a flow graph: its elements by position (the links,
    then the nodes that can fail, as in ``network.models``), each link's
    two nodes and capacity, and at each node the links that meet it."""

    def __init__(self, network):
        self.names = list(network.models)
        position = {name: i for i, name in enumerate(self.names)}
        node_id = {node: i for i, node in enumerate(network._adjacent)}
        self.directed = network.directed
        self.source = node_id[network.source]
        self.target = node_id[network.target]
        self.links = len(network.links)
        self.limit = [
            network.capacity.get(name, math.inf) for name in network.links
        ]
        # At each node: (link, the node at its other end, +1 if the node
        # is the link's first and -1 if its second).
        self.at: List[list] = [[] for _ in node_id]
        for i, (u, v) in enumerate(network.links.values()):
            self.at[node_id[u]].append((i, node_id[v], 1))
            self.at[node_id[v]].append((i, node_id[u], -1))
        # The element of each node that can fail (None: it always works).
        self.element: List[Optional[int]] = [None] * len(node_id)
        for node in network.nodes:
            self.element[node_id[node]] = position[node]
        finite = [c for c in self.limit if math.isfinite(c)]
        #: A flow no larger is none (see the module docstring).
        self.eps = 1e-12 * (sum(finite) if finite else 1.0)

    def _residual(self, link: int, sign: int, flow: list) -> float:
        """What more the link can carry out of the node it meets with
        ``sign``: an undirected link its capacity less what it carries
        that way (plus what it carries the other way); a directed link
        that from its first node, and from its second only what it
        carries, taken back."""
        if sign > 0 or not self.directed:
            return self.limit[link] - sign * flow[link]
        return flow[link]

    def _path(self, usable, flow: list) -> Optional[list]:
        """The fewest links from the source to the target that can each
        carry more, through usable elements: as (link, sign) from the
        target back to the source, or None if there is none."""
        if not (self._open(self.source, usable)):
            return None
        back: Dict[int, tuple] = {self.source: ()}
        queue = deque([self.source])
        while queue:
            node = queue.popleft()
            for link, other, sign in self.at[node]:
                if other in back or not usable[link]:
                    continue
                if not self._open(other, usable):
                    continue
                if self._residual(link, sign, flow) <= self.eps:
                    continue
                back[other] = (link, sign, node)
                if other == self.target:
                    path = []
                    while other != self.source:
                        link, sign, other = back[other]
                        path.append((link, sign))
                    return path
                queue.append(other)
        return None

    def _open(self, node: int, usable) -> bool:
        element = self.element[node]
        return element is None or bool(usable[element])

    def augment(self, usable, flow: list, value: float) -> float:
        """The maximum flow through the ``usable`` elements, reached from
        ``flow`` (each link's, from its first node to its second), which
        carries ``value`` and is updated in place: ``inf`` once a path of
        links with no limit joins the terminals (``flow`` is then left as
        it was)."""
        while True:
            path = self._path(usable, flow)
            if path is None:
                return value
            more = min(self._residual(link, s, flow) for link, s in path)
            if math.isinf(more):
                return math.inf
            for link, sign in path:
                flow[link] += sign * more
            value += more

    def value(self, usable) -> float:
        """The maximum flow through the ``usable`` elements (one state),
        tidied."""
        return tidy(self.augment(usable, [0.0] * self.links, 0.0))

    def decide(self, state: list) -> Tuple[float, Optional[int]]:
        """For the elements' ``state`` (each undecided, working or
        failed): the capacity and None, if the undecided ones cannot
        change it; otherwise the undecided element to decide next, one
        the flow with them all working passes through (a node before a
        link, then the first in the elements' order)."""
        flow = [0.0] * self.links
        works = [s == _WORKS for s in state]
        least = self.augment(works, flow, 0.0)
        if math.isinf(least):
            return least, None
        usable = [s != _FAILED for s in state]
        most = self.augment(usable, flow, least)
        if math.isinf(most):
            # The path with no limit, through some undecided element.
            path = self._path(usable, flow)
            assert path is not None
            carrying = {link for link, _ in path}
        elif most - least <= self.eps:
            return tidy(least), None
        else:
            carrying = {
                link for link, f in enumerate(flow) if abs(f) > self.eps
            }
        for node, element in enumerate(self.element):
            if element is not None and state[element] == _UNDECIDED:
                if any(link in carrying for link, _, _ in self.at[node]):
                    return math.nan, element
        return math.nan, min(
            link for link in carrying if state[link] == _UNDECIDED
        )


class _Tree:
    """The states that matter, as a binary tree in the order it is
    walked: at each place the ``element`` decided there (-1 at a branch's
    end), where the branch with it ``failed`` starts (the branch with it
    working starts at the next place), and at an end the position of its
    capacity in ``levels``."""

    def __init__(self, graph: _Graph, most: int):
        self.graph = graph
        self.element: List[int] = []
        self.failed: List[int] = []
        values: List[float] = []
        ends = 0
        # A state to work out, and the place whose failed branch it is.
        stack: List[Tuple[list, Optional[int]]] = [
            ([_UNDECIDED] * len(graph.names), None)
        ]
        while stack:
            state, parent = stack.pop()
            place = len(self.element)
            if parent is not None:
                self.failed[parent] = place
            value, element = graph.decide(state)
            self.failed.append(-1)
            values.append(value)
            if element is None:
                self.element.append(-1)
                ends += 1
                if ends > most:
                    raise _too_many(most)
                continue
            self.element.append(element)
            down, up = list(state), list(state)
            down[element], up[element] = _FAILED, _WORKS
            stack.append((down, place))
            stack.append((up, None))
        self.ends = ends
        self.levels = np.array(
            sorted({v for v, e in zip(values, self.element) if e < 0})
        )
        where = {level: i for i, level in enumerate(self.levels.tolist())}
        self.level = [
            where[v] if e < 0 else -1 for v, e in zip(values, self.element)
        ]

    def rows(self, p: list, q: list, size: int) -> np.ndarray:
        """The probability of each of ``levels`` (a row each, ``size``
        columns), from each element's probabilities of working ``p`` and
        of failing ``q``, by position: each end's is the product of those
        of the elements decided on the way to it."""
        out = np.zeros((len(self.levels), size))
        waiting = {0: np.ones(size)}
        for place, element in enumerate(self.element):
            reached = waiting.pop(place)
            if element < 0:
                out[self.level[place]] += reached
            else:
                waiting[self.failed[place]] = reached * q[element]
                waiting[place + 1] = reached * p[element]
        return out


def _too_many(most: int) -> NotImplementedError:
    return NotImplementedError(
        f"The network's capacity takes more than {most:,} states of its "
        "links and nodes to work out exactly (each a set of them working "
        "and failed that gives one maximum flow). Estimate it with "
        "method='simulate', which draws the states (mc_samples of them, "
        "seeded with seed), or raise the limit, "
        "repyability.network.MAX_CAPACITY_STATES."
    )


def require_capacity(net, name: str) -> None:
    """Raise unless the analysis ``name`` of the capacity applies to the
    ``net``work: some link has a capacity, and its elements are repaired
    for the capacity over time."""
    if not net.capacity:
        raise Refused(
            f"{name}() needs the links' capacities, and no link has one: "
            "give each link's throughput with capacity={link: amount} "
            "when building the network."
        )
    if name == "capacity_distribution" and net.is_repairable:
        net._refuse_unrepaired(name)
    if name != "capacity_distribution" and not net.is_repairable:
        raise Refused(
            f"{name}() works on repaired elements, and this network has "
            "none: capacity_distribution(x) gives the distribution of its "
            "capacity at x. Give the links, and the nodes that can fail, "
            "component specs ('reliability' and 'repairability') to have "
            "them repaired."
        )


def graph(net) -> _Graph:
    """The ``net``work's flow graph, built once."""
    if net._flow is None:
        net._flow = _Graph(net)
    return net._flow


def tree(net) -> _Tree:
    """The ``net``work's states that matter, found once; a refusal (too
    many) is kept, and raised again."""
    from repyability import network

    if net._capacity_refused is not None:
        raise NotImplementedError(net._capacity_refused)
    if net._capacity_tree is None:
        try:
            net._capacity_tree = _Tree(graph(net), network.MAX_CAPACITY_STATES)
        except NotImplementedError as refusal:
            net._capacity_refused = str(refusal)
            raise
    return net._capacity_tree


def _columns(net, values: Dict[Hashable, np.ndarray]) -> list:
    """Each element's ``values``, by position."""
    return [np.asarray(values[name], dtype=float) for name in net.models]


def _exact(
    net, p: dict, q: dict, size: int, every: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """The capacity's levels and their probabilities (a row each, ``size``
    columns) at the elements' probabilities of working ``p`` and of
    failing ``q``; the levels no column reaches left out, unless
    ``every``."""
    states = tree(net)
    rows = states.rows(_columns(net, p), _columns(net, q), size)
    if every:
        return states.levels, rows
    reached = np.any(rows != 0.0, axis=1)
    return states.levels[reached], rows[reached]


def _simulated(
    net, p: np.ndarray, u: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """The capacity's levels and their estimated probabilities, from the
    states drawn with the uniforms ``u`` (a row per sample, a column per
    element): in each column of ``p`` (elements by samples, or by one,
    by columns) an element works where its uniform is below its
    probability of working. Each distinct state's maximum flow is worked
    out once."""
    flows = graph(net)
    samples = len(u)
    known: Dict[bytes, float] = {}
    columns = []
    for j in range(p.shape[-1]):
        drawn = u < p[..., j].T
        states, counts = np.unique(drawn, axis=0, return_counts=True)
        shares: Dict[float, float] = {}
        for state, count in zip(states, counts.tolist()):
            key = state.tobytes()
            if key not in known:
                known[key] = flows.value(state.tolist())
            level = known[key]
            shares[level] = shares.get(level, 0) + count
        columns.append(shares)
    levels = np.array(sorted(set().union(*columns)))
    rows = np.array(
        [
            [shares.get(level, 0) / samples for shares in columns]
            for level in levels.tolist()
        ]
    ).reshape(len(levels), len(columns))
    return levels, rows


def _options(method: str, mc_samples, seed, name: str) -> Optional[int]:
    """The simulation's samples for the analysis ``name``, or None for
    the exact method, the options checked."""
    one_of("method", method, ("exact", "simulate"))
    if method == "exact":
        simulation_options(
            f"{name}()", {"mc_samples": mc_samples, "seed": seed}
        )
        return None
    check_seed(seed)
    samples = MC_SAMPLES if mc_samples is None else mc_samples
    montecarlo.check_count(samples, False, "mc_samples")
    return int(samples)


def _uniforms(net, samples: int, seed, extra: int = 0) -> np.ndarray:
    """The simulation's uniforms: a row per sample, a column per element
    and ``extra`` more."""
    rng = np.random.default_rng(check_seed(seed))
    return rng.random((samples, len(net.models) + extra))


def _result(levels, rows, scalar: bool) -> CapacityDistribution:
    return CapacityDistribution(levels, rows[:, 0] if scalar else rows)


def capacity_distribution(
    net, x, method: str, mc_samples, seed
) -> CapacityDistribution:
    """See ``Network.capacity_distribution``."""
    require_capacity(net, "capacity_distribution")
    samples = _options(method, mc_samples, seed, "capacity_distribution")
    if net.is_repairable:
        if x is not None:
            raise ValueError(
                "capacity_distribution() of a network of repaired elements "
                "is its long-run distribution and takes no time: "
                "point_capacity(x) gives the distribution at x, and "
                "mission_capacity(t) its average over [0, t]."
            )
        from repyability.rbd import _long_run

        _, q, weights = _long_run._long_run_unavailabilities(
            net._repaired, None, None
        )
        q = net._with_fixed(q, len(weights))
        p = {name: 1.0 - q[name] for name in net.models}
        if samples is None:
            levels, rows = _exact(net, p, q, len(weights))
            return CapacityDistribution(levels, rows @ weights)
        levels, rows = _simulated(
            net,
            np.array(_columns(net, p))[:, None, :],
            _uniforms(net, samples, seed),
        )
        return CapacityDistribution(levels, rows @ weights)
    from repyability.rbd.non_repairable_rbd import _times

    times, scalar = _times(net, x)
    shape = times.shape
    times = times.reshape(-1)
    p, q = net._probabilities(times, net.models)
    if samples is None:
        levels, rows = _exact(net, p, q, times.size)
    else:
        levels, rows = _simulated(
            net,
            np.array(_columns(net, p))[:, None, :],
            _uniforms(net, samples, seed),
        )
    if scalar:
        return CapacityDistribution(levels, rows[:, 0])
    return CapacityDistribution(levels, rows.reshape((len(levels),) + shape))


def point_capacity(net, x, method: str, mc_samples, seed):
    """See ``Network.point_capacity``."""
    require_capacity(net, "point_capacity")
    samples = _options(method, mc_samples, seed, "point_capacity")
    times = nonnegative_times(x)
    ends = times.ravel()
    p, q = net._availabilities(ends, net.models)
    if samples is None:
        levels, rows = _exact(net, p, q, ends.size)
    else:
        levels, rows = _simulated(
            net,
            np.array(_columns(net, p))[:, None, :],
            _uniforms(net, samples, seed),
        )
    return _result(levels, rows, np.ndim(x) == 0)


def mission_capacity(net, t, method: str, mc_samples, seed):
    """See ``Network.mission_capacity``."""
    from repyability.rbd import _curves, _event_curves, _repairable_capacity

    require_capacity(net, "mission_capacity")
    samples = _options(method, mc_samples, seed, "mission_capacity")
    windows = nonnegative_times(t)
    ends = windows.ravel()
    scalar = np.ndim(t) == 0
    if samples is not None:
        # Each sample a time drawn evenly over the window, and the
        # elements' states then.
        u = _uniforms(net, samples, seed, extra=1)
        moments = u[:, -1:] * ends[None, :]
        p, _ = net._availabilities(moments.ravel(), net.models)
        levels, rows = _simulated(
            net,
            np.array(_columns(net, p)).reshape(-1, samples, ends.size),
            u[:, :-1],
        )
        return _result(levels, rows, scalar)
    rbd = net._repaired
    horizon = float(ends.max()) if ends.size else 0.0
    curves, states = _event_curves._curves_to(rbd, ends)

    def rows_at(x: np.ndarray):
        q = net._with_fixed(
            _curves._nodes_down_at(rbd, curves, x, states), len(x)
        )
        p = {name: 1.0 - value for name, value in q.items()}
        return _exact(net, p, q, len(x), every=True)

    levels, rows = _repairable_capacity._mission_rows(
        rbd, list(curves.values()), rows_at, ends, horizon
    )
    return _result(levels, rows, scalar)
