"""Two-terminal reliability of undirected networks (#104).

Power, pipe and communication networks are undirected, and it is often
their links that fail. Their question is whether two terminals stay
connected: some path of working links (through working nodes, where nodes
can fail too) joins them. A reliability block diagram is directed, with
failing nodes, so a network has a model of its own.

The exact values come from a binary decision diagram built from the
network itself (#143), after Hardy, Lucet & Limnios (2007): the links are
decided in an order, and after each, what is left to decide depends only on
how the *frontier* (the nodes with links decided and links to come) is
joined up by the working links so far, and on which parts hold the
terminals. The states before each decision are worked out together, as
the rows of an array, and equal ones merged (#173), so the diagram grows
with the frontier's width, not with the number of paths, which multiply in
a meshed network: a grid of 100 nodes, corner to corner, takes a second and
a half (its widest decision has 42,000 states, of 1.9 million in all). A
node that can fail is decided as its first link is, and a failed one takes
its links out. The diagram is held as arrays, a level (one decision) at a
time, along which the probability, its complement and the gradient are
worked out; its steps are those of ``shannon.py``'s plans, which the cut
sets replay. ``METHOD = "paths"`` decides the network from its minimal
paths instead, as before: each simple path between the terminals, as the
set of its links and of the nodes on it that can fail, by the Shannon
decomposition of ``shannon.py``.

The simulation draws each element's lifetime and finds, in each sample,
the path that lasts longest: its lifetime is the network's.
"""

from collections import deque
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd._sampling import lifetime_sampler
from repyability.rbd.non_repairable_rbd import check_x
from repyability.rbd.shannon import _minimal_cut_sets, _shannon_plan
from repyability.utils.deprecation import refuse_removed_names
from repyability.utils.wrappers import numpy_seed

#: How the exact values are worked out: ``"bdd"`` (the default), by the
#: decision diagram built from the network, or ``"paths"``, by the Shannon
#: decomposition of its simple paths between the terminals (listed; slower
#: but for the smallest networks). Read when a network's values are first
#: worked out.
METHOD = "bdd"

#: The most simple paths listed (by ``path_sets`` and ``METHOD = "paths"``).
MAX_PATHS = 100_000

#: The most states the decision diagram is built from (each a way the
#: frontier can be joined up before a decision), about a second's work per
#: million: beyond, the exact values refuse, pointing to the simulation. A
#: square grid of 100 nodes has 1.9 million; one of 121, 7.7 million.
MAX_STATES = 5_000_000

# Value slots 0 and 1 of a plan: the terminals are parted, joined.
_FAIL, _WORK = 0, 1

# A decision's outcome that is no state: the terminals parted, joined.
_NONE_FAIL, _NONE_WORK = -1, -2


@refuse_removed_names
class Network:
    """An undirected network whose links, and optionally nodes, fail, and
    the reliability of the connection between two of its nodes.

    The network works while some path of working links, through working
    nodes, joins ``source`` to ``target``. A link works both ways. Nodes
    work for ever unless given a model in ``nodes``; the terminals can be
    given one too. Every element fails independently of the others.

    The exact values come from a binary decision diagram built from the
    network, link by link, whose size grows with the network's width
    rather than with its number of paths (see the module docstring): a
    grid of 100 nodes takes a second and a half. One of more than
    ``MAX_STATES`` (5,000,000) states refuses, pointing to the
    simulation.

    Parameters
    ----------
    links : dict
        Each link's name, mapped to ``(node, node, model)``: the two nodes
        it joins, which must differ, and its lifetime model (anything with
        ``sf`` and ``ff``, such as a fitted surpyval distribution), or the
        probability that it has failed (a number, taken as a
        ``FixedEventProbability``). Two links may join the same two
        nodes.
    source : Hashable
        One terminal: a node of the network.
    target : Hashable
        The other terminal, another node.
    nodes : dict, optional
        The nodes that can fail, each mapped to its model (or the
        probability that it has failed), by default none. Their names must
        differ from the links'.

    Attributes
    ----------
    links : dict
        Each link's two nodes, by name.
    models : dict
        Each element's model, by name: the links', then the nodes'.
    source, target : Hashable
        The terminals.
    is_fixed : bool
        Whether every model is a fixed probability (time is then
        irrelevant, and ``x`` may be left out).

    Raises
    ------
    ValueError
        If there is no link, a link is not a ``(node, node, model)`` of two
        different nodes, a model has no ``sf`` or ``ff``, a terminal or a
        node in ``nodes`` is in no link, the terminals are the same node,
        or a node has the name of a link.

    Examples
    --------
    The bridge network: links ``a`` and ``b`` out of the source, ``d`` and
    ``e`` into the target, and ``c`` across the middle, which works either
    way. With every link working with probability 0.9, the terminals stay
    connected with probability ``2p^2 + 2p^3 - 5p^4 + 2p^5``:

    >>> from surpyval import FixedEventProbability
    >>> from repyability import Network
    >>> link = FixedEventProbability.from_params(0.1)
    >>> bridge = Network(
    ...     {
    ...         "a": ("s", "x", link),
    ...         "b": ("s", "y", link),
    ...         "c": ("x", "y", link),
    ...         "d": ("x", "t", link),
    ...         "e": ("y", "t", link),
    ...     },
    ...     source="s",
    ...     target="t",
    ... )
    >>> round(bridge.sf(), 5)
    0.97848
    >>> sorted(sorted(p) for p in bridge.path_sets())
    [['a', 'c', 'e'], ['a', 'd'], ['b', 'c', 'd'], ['b', 'e']]
    """

    def __init__(
        self,
        links: Dict[Hashable, Tuple[Hashable, Hashable, Any]],
        source: Hashable,
        target: Hashable,
        nodes: Optional[Dict[Hashable, Any]] = None,
    ):
        if not isinstance(links, dict) or not links:
            raise ValueError(
                "links must be a non-empty dict of name: (node, node, model)."
            )
        self.links: Dict[Hashable, Tuple[Hashable, Hashable]] = {}
        self.models: Dict[Hashable, Any] = {}
        adjacent: Dict[Hashable, List[Tuple[Hashable, Hashable]]] = {}
        for name, link in links.items():
            if not (isinstance(link, (tuple, list)) and len(link) == 3):
                raise ValueError(
                    f"Link {name!r} must be a (node, node, model), got "
                    f"{link!r}."
                )
            u, v, model = link
            if u == v:
                raise ValueError(f"Link {name!r} joins node {u!r} to itself.")
            model = self._model(f"Link {name!r}", model)
            self.links[name] = (u, v)
            self.models[name] = model
            adjacent.setdefault(u, []).append((name, v))
            adjacent.setdefault(v, []).append((name, u))
        if source == target:
            raise ValueError("The source and target must be two nodes.")
        for terminal in (source, target):
            if terminal not in adjacent:
                raise ValueError(
                    f"Terminal {terminal!r} is in no link of the network."
                )
        self.nodes: Dict[Hashable, Any] = dict(nodes or {})
        for node, model in self.nodes.items():
            if node not in adjacent:
                raise ValueError(f"Node {node!r} is in no link.")
            if node in self.links:
                raise ValueError(
                    f"Node {node!r} has the name of a link: name them apart."
                )
            model = self._model(f"Node {node!r}", model)
            self.nodes[node] = self.models[node] = model
        self.source, self.target = source, target
        self._adjacent = adjacent
        self.is_fixed = all(
            is_fixed_probability(m) for m in self.models.values()
        )
        self._paths: Optional[List[frozenset]] = None
        self._plan: Optional[_Plan] = None
        # Why the exact values were refused, once found (#229): the search
        # that found it is not repeated.
        self._refused: Optional[str] = None

    @staticmethod
    def _model(what: str, model):
        """An element's model, checked: anything with ``sf`` and ``ff``,
        or a number, the probability that the element has failed, as a
        ``FixedEventProbability`` (as a fault tree takes it, #179)."""
        if isinstance(
            model, (int, float, np.integer, np.floating)
        ) and not isinstance(model, (bool, np.bool_)):
            if not 0.0 <= float(model) <= 1.0:
                raise ValueError(
                    f"{what}: a probability of failing must be in [0, 1], "
                    f"got {model!r}."
                )
            from surpyval import FixedEventProbability

            return FixedEventProbability.from_params(float(model))
        if not (hasattr(model, "sf") and hasattr(model, "ff")):
            raise ValueError(
                f"{what}: the model must have sf and ff (a lifetime "
                "distribution or a FixedEventProbability), or be the "
                f"probability that it has failed, got {model!r}."
            )
        return model

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def _simple_paths(self) -> List[frozenset]:
        """Every simple path between the terminals, as the set of its links
        and of its nodes that can fail.

        Raises
        ------
        NotImplementedError
            If there are more than ``MAX_PATHS``.
        """
        if self._paths is not None:
            return self._paths
        failing = set(self.nodes)
        paths: List[frozenset] = []
        start = [self.source] if self.source in failing else []
        # Depth first, without recursion: each entry is a node, the
        # elements of the path so far, the nodes on it and the next of the
        # node's links to try.
        stack: list = [(self.source, start, {self.source}, 0)]
        while stack:
            node, elements, visited, i = stack.pop()
            around = self._adjacent[node]
            if i >= len(around):
                continue
            stack.append((node, elements, visited, i + 1))
            name, other = around[i]
            if other in visited:
                continue
            step = elements + [name]
            if other in failing:
                step = step + [other]
            if other == self.target:
                paths.append(frozenset(step))
                if len(paths) > MAX_PATHS:
                    raise NotImplementedError(
                        f"The network has more than {MAX_PATHS:,} paths "
                        "between its terminals, too many to list. Its exact "
                        "values come from its decision diagram (network."
                        "METHOD = 'bdd', the default), or simulate it: "
                        "method='simulate'."
                    )
                continue
            stack.append((other, step, visited | {other}, 0))
        self._paths = paths
        return paths

    def path_sets(self) -> Set[frozenset]:
        """The minimal path sets: for each simple path between the
        terminals, the set of its links and of its nodes that can fail.

        Returns
        -------
        set[frozenset]
            The minimal path sets, each a frozenset of link and node names.

        Raises
        ------
        NotImplementedError
            If there are more than ``MAX_PATHS``.
        """
        return set(self._simple_paths())

    def cut_sets(self) -> Set[frozenset]:
        """The minimal cut sets: the smallest sets of links and nodes whose
        failure parts the terminals.

        Returns
        -------
        set[frozenset]
            The minimal cut sets, each a frozenset of link and node names.

        Raises
        ------
        NotImplementedError
            If the network's decision diagram would have more than
            ``MAX_STATES`` states (with ``METHOD = "paths"``, if there are
            more than ``MAX_PATHS`` paths).
        """
        return set(_minimal_cut_sets(self._decomposition().steps()))

    def _decomposition(self) -> "_Plan":
        """The plan of the probability that the terminals are joined (see
        ``shannon.py``), as a ``_Plan``: the decision diagram built from the
        network, or with ``METHOD = "paths"`` the Shannon decomposition of
        its simple paths."""
        if self._plan is None:
            if METHOD not in ("bdd", "paths"):
                raise ValueError(
                    f"network.METHOD must be 'bdd' or 'paths', got "
                    f"{METHOD!r}."
                )
            if self._refused is not None:
                raise NotImplementedError(self._refused)
            try:
                if METHOD == "paths":
                    paths = self._simple_paths()
                    # Nothing joins the terminals: they are always parted.
                    self._plan = (
                        _plan_from_steps(*_shannon_plan(paths))
                        if paths
                        else _constant(_FAIL)
                    )
                else:
                    order = _link_order(
                        self.links, self._adjacent, self.source, self.target
                    )
                    self._plan = _frontier_plan(
                        order,
                        self.links,
                        set(self.nodes),
                        self.source,
                        self.target,
                    )
            except NotImplementedError as refusal:
                self._refused = str(refusal)
                raise
        return self._plan

    # ------------------------------------------------------------------
    # Exact
    # ------------------------------------------------------------------

    def _probabilities(self, x: np.ndarray, names) -> Tuple[dict, dict]:
        """Each named element's probabilities of working and of failing at
        times ``x``, from its model's ``sf`` and ``ff``."""
        p: Dict[Hashable, np.ndarray] = {}
        q: Dict[Hashable, np.ndarray] = {}
        for name in names:
            model = self.models[name]
            with np.errstate(all="ignore"):
                p[name] = np.asarray(model.sf(x), dtype=float).reshape(x.shape)
                q[name] = np.asarray(model.ff(x), dtype=float).reshape(x.shape)
        return p, q

    def _value(self, x: np.ndarray, terminals=(0.0, 1.0)) -> np.ndarray:
        """The probability that the terminals are connected at times ``x``,
        or with ``terminals`` ``(1.0, 0.0)`` that they are not: a sum of
        products, so that a small one keeps its precision."""
        plan = self._decomposition()
        p, q = self._probabilities(x, plan.names)
        return _evaluate(plan, p, q, x.shape, terminals)

    def _method(self, method: str, mc_samples) -> str:
        if method not in ("exact", "simulate"):
            raise ValueError(
                f"method must be 'exact' or 'simulate', got {method!r}."
            )
        if method == "exact" and mc_samples is not None:
            raise ValueError("mc_samples applies only to method='simulate'.")
        return method

    @check_x
    def sf(
        self,
        x=None,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ):
        """The probability that the terminals are connected at time/s
        ``x``.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.
        method : str, optional
            ``"exact"`` (the default), from the minimal path sets, or
            ``"simulate"``: the fraction of ``mc_samples`` simulated
            networks still connected.
        mc_samples : int, optional
            The number of simulated networks, by default 10_000.
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float or numpy.ndarray
            The reliability: a float for a scalar ``x``.

        Raises
        ------
        ValueError
            If ``x`` is left out of a network whose models depend on time,
            or ``method`` or ``mc_samples`` is invalid.
        NotImplementedError
            With ``method="exact"``, if the network's decision diagram would
            have more than ``MAX_STATES`` states (with ``METHOD = "paths"``,
            if there are more than ``MAX_PATHS`` paths between the
            terminals).
        """
        if self._method(method, mc_samples) == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return np.mean(lives[:, None] > x[None, :], axis=0)
        return self._value(x)

    @check_x
    def ff(
        self,
        x=None,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ):
        """The probability that the terminals are parted at time/s ``x``,
        ``1 - sf(x)``, worked out to its own precision when it is small
        (see ``sf`` for the parameters).

        Parameters
        ----------
        x : float or array_like, optional
            The time/s.
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``.
        mc_samples : int, optional
            The number of simulated networks, by default 10_000.
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float or numpy.ndarray
            The unreliability.
        """
        if self._method(method, mc_samples) == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return np.mean(~(lives[:, None] > x[None, :]), axis=0)
        return self._value(x, (1.0, 0.0))

    @check_x
    def birnbaum_importance(self, x=None) -> Dict[Hashable, Any]:
        """Each element's Birnbaum importance at time/s ``x``: the
        reliability with it working less that with it failed.

        Every element's at once, as the derivative of the reliability, by
        one pass forward through the plan and one back, from whichever of
        the reliability and the unreliability is the smaller, so that a
        small importance keeps its precision.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s.

        Returns
        -------
        dict
            Per link and failing node, by name, its importance (0 for one
            that is in no path).
        """
        plan = self._decomposition()
        p, q = self._probabilities(x, plan.names)
        works, up = _value_and_gradient(plan, p, q, x.shape)
        fails, down = _value_and_gradient(plan, p, q, x.shape, (1.0, 0.0))
        zero = np.zeros(x.shape)
        out: Dict[Hashable, Any] = {}
        for name in self.models:
            out[name] = np.where(
                np.asarray(fails) <= np.asarray(works),
                -np.asarray(down.get(name, zero)),
                np.asarray(up.get(name, zero)),
            ) * np.ones(x.shape)
        return out

    def mean(
        self,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> float:
        """The mean time until the terminals are parted.

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default): the exact reliability integrated over
            time by quadrature (to about 1e-10). ``"simulate"``: the mean of
            ``mc_samples`` simulated lifetimes.
        mc_samples : int, optional
            The number of simulated networks, by default 10_000.
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float
            The mean lifetime: ``inf`` if some networks never part.

        Raises
        ------
        ValueError
            If a model is a fixed probability (it has no lifetime), or
            ``method`` or ``mc_samples`` is invalid.
        """
        if self.is_fixed or any(
            is_fixed_probability(m) for m in self.models.values()
        ):
            raise ValueError(
                "A fixed probability has no lifetime, so the network has no "
                "mean lifetime."
            )
        if self._method(method, mc_samples) == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return float(np.mean(lives))
        from repyability.rbd._mean_lifetime import (
            mean_lifetime,
            model_kinks,
            model_knots,
        )

        models = self.models.values()
        return mean_lifetime(
            lambda t: self._value(np.asarray(t, float)),
            [model_knots(m) for m in models],
            [model_kinks(m) for m in models],
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def random(self, size: int, seed=None) -> np.ndarray:
        """``size`` random lifetimes of the connection: in each, every
        element's lifetime is drawn, and the connection lasts as long as
        its longest-lasting path (whose life is that of its first element
        to fail).

        The path is found by adding links in order of their lives, longest
        first, until the terminals are joined: a link lasts no longer than
        the nodes it joins.

        Parameters
        ----------
        size : int
            The number of lifetimes.
        seed : int, optional
            Seeds the draws, by default None.

        Returns
        -------
        numpy.ndarray
            The lifetimes (``inf`` where the terminals never part).

        Raises
        ------
        ValueError
            If ``size`` is not a positive integer.
        """
        montecarlo.check_count(size, False, "size")
        names = list(self.models)
        lives: Dict[Hashable, np.ndarray] = {}
        with numpy_seed(seed):
            samplers = {n: lifetime_sampler(self.models[n]) for n in names}
            width = sum(s.width for s in samplers.values() if s is not None)
            u = np.random.random_sample((size, width))
            start = 0
            for name in names:
                sampler = samplers[name]
                if sampler is None:
                    lives[name] = np.asarray(
                        self.models[name].random(size), dtype=float
                    ).reshape(size)
                    continue
                block = u[:, start : start + sampler.width]
                start += sampler.width
                lives[name] = np.asarray(sampler.draw(block), dtype=float)
        # A link lasts no longer than the failing nodes it joins.
        index: Dict[Hashable, int] = {}
        for u_, v_ in self.links.values():
            index.setdefault(u_, len(index))
            index.setdefault(v_, len(index))
        ends = np.array(
            [[index[a], index[b]] for a, b in self.links.values()], dtype=int
        )
        last = np.column_stack([lives[name] for name in self.links])
        for node in self.nodes:
            touching = np.array(
                [node in pair for pair in self.links.values()], dtype=bool
            )
            last[:, touching] = np.minimum(
                last[:, touching], lives[node][:, None]
            )
        order = np.argsort(-last, axis=1, kind="stable")
        s, t = index[self.source], index[self.target]
        out = np.empty(size)
        for i in range(size):
            parent = list(range(len(index)))

            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            out[i] = -np.inf
            for j in order[i]:
                a, b = find(ends[j, 0]), find(ends[j, 1])
                if a != b:
                    parent[a] = b
                if find(s) == find(t):
                    out[i] = last[i, j]
                    break
        return np.maximum(out, 0.0)

    def __repr__(self) -> str:
        return (
            f"Network({len(self.links)} links, {len(self.nodes)} failing "
            f"nodes, {self.source!r} to {self.target!r})"
        )


def _link_order(
    links: Dict[Hashable, Tuple[Hashable, Hashable]],
    adjacent: Dict[Hashable, List[Tuple[Hashable, Hashable]]],
    source: Hashable,
    target: Hashable,
) -> List[Hashable]:
    """The order the links of the terminals' part of the network are
    decided in: by the breadth-first order of their later end, from either
    terminal, whichever keeps the frontiers narrower (the smaller sum over
    the links of two to the frontier's size). A link out of the
    terminals' part cannot join them, and is left out."""
    candidates = []
    for root in (source, target):
        position = {root: 0}
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for _, v in adjacent[u]:
                if v not in position:
                    position[v] = len(position)
                    queue.append(v)
        names = [
            name
            for name, (u, v) in links.items()
            if u in position and v in position
        ]
        names.sort(
            key=lambda n: (
                max(position[links[n][0]], position[links[n][1]]),
                min(position[links[n][0]], position[links[n][1]]),
            )
        )
        candidates.append(names)
    return min(candidates, key=lambda order: _order_cost(order, links))


def _order_cost(order: Sequence[Hashable], links) -> float:
    """The sum over ``order``'s links of two to the frontier's size after
    each: a measure of the decision diagram's size."""
    first: Dict[Hashable, int] = {}
    last: Dict[Hashable, int] = {}
    for i, name in enumerate(order):
        for w in links[name]:
            first.setdefault(w, i)
            last[w] = i
    opens = [0] * (len(order) + 1)
    for w in first:
        opens[first[w]] += 1
        opens[last[w]] -= 1
    cost, size = 0.0, 0
    for i in range(len(order)):
        size += opens[i]
        cost += 2.0 ** min(size, 1000)
    return cost


class _Plan(NamedTuple):
    """A decision diagram (a plan in the format of ``shannon.py``'s) held
    as arrays, level by level, so that it is evaluated a level at a time:
    the steps ``bounds[j]`` to ``bounds[j + 1]`` all branch on variable
    ``names[pivots[j]]``, step ``i`` filling value slot ``i + 2`` from its
    ``active`` and ``inactive`` slots, which earlier levels fill (or are
    slots 0 and 1, the terminals parted and joined). ``root`` is the slot
    of the whole."""

    names: List[Hashable]
    pivots: np.ndarray
    bounds: np.ndarray
    active: np.ndarray
    inactive: np.ndarray
    root: int

    def steps(self) -> tuple:
        """The plan in ``shannon.py``'s format, ``(steps, root)``, step
        ``i`` a ``(pivot, active, inactive)`` filling slot ``i + 2``."""
        steps: List[tuple] = []
        for j, pivot in enumerate(self.pivots.tolist()):
            start, end = self.bounds[j], self.bounds[j + 1]
            steps.extend(
                zip(
                    [self.names[pivot]] * int(end - start),
                    self.active[start:end].tolist(),
                    self.inactive[start:end].tolist(),
                )
            )
        return steps, self.root


def _constant(slot: int) -> _Plan:
    """The plan of a constant: the terminals always parted, or joined."""
    empty = np.zeros(0, dtype=np.int64)
    return _Plan([], empty, np.zeros(1, dtype=np.int64), empty, empty, slot)


def _plan_from_steps(steps: list, root: int) -> _Plan:
    """A plan in ``shannon.py``'s format, as a ``_Plan``: its steps grouped
    by their depth (the longest way down to a terminal) and pivot, so that
    each level's branches lead to earlier ones."""
    if not steps:
        return _constant(root)
    names = list(dict.fromkeys(pivot for pivot, _, _ in steps))
    index = {name: i for i, name in enumerate(names)}
    depth = [0, 0]
    for _, active, inactive in steps:
        depth.append(1 + max(depth[active], depth[inactive]))
    keys = [
        (depth[i + 2], index[pivot]) for i, (pivot, _, _) in enumerate(steps)
    ]
    order = sorted(range(len(steps)), key=keys.__getitem__)
    slot = np.arange(len(steps) + 2)
    slot[np.asarray(order) + 2] = np.arange(len(steps)) + 2
    active = np.array([slot[steps[i][1]] for i in order], dtype=np.int64)
    inactive = np.array([slot[steps[i][2]] for i in order], dtype=np.int64)
    pivots, bounds = [], []
    for position, i in enumerate(order):
        if not position or keys[i] != keys[order[position - 1]]:
            pivots.append(keys[i][1])
            bounds.append(position)
    bounds.append(len(order))
    return _Plan(
        names,
        np.array(pivots, dtype=np.int64),
        np.array(bounds, dtype=np.int64),
        active,
        inactive,
        int(slot[root]),
    )


#: The most values a plan's evaluation holds at once (its slots, times the
#: times evaluated together; 128 MB): more times are evaluated in turn. A
#: level's rows are gathered faster the more times each holds (#229): a
#: plan of 1.4 million slots evaluates a time in 12 ms with eleven at once,
#: 28 ms with two.
_EVALUATION_SIZE = 16_000_000


def _rows(plan: _Plan, values: dict, size: int) -> np.ndarray:
    """Each of the plan's variables' ``values`` (of ``size`` elements once
    broadcast), as the rows of one array."""
    out = np.empty((len(plan.names), size))
    for i, name in enumerate(plan.names):
        out[i] = np.broadcast_to(np.asarray(values[name], float), (size,))
    return out


def _forward(
    plan: _Plan, p: np.ndarray, q: np.ndarray, terminals
) -> np.ndarray:
    """Every slot's value, a row each, for the variables' probabilities
    ``p`` and complements ``q`` (a row per variable, a column per
    evaluation): a sum of products, so a small value keeps its precision."""
    values = np.empty((2 + len(plan.active), p.shape[1]))
    values[0], values[1] = terminals
    for j, pivot in enumerate(plan.pivots.tolist()):
        start, end = plan.bounds[j], plan.bounds[j + 1]
        values[2 + start : 2 + end] = (
            p[pivot] * values[plan.active[start:end]]
            + q[pivot] * values[plan.inactive[start:end]]
        )
    return values


def _chunks(plan: _Plan, size: int):
    """The ranges of evaluations worked out together."""
    step = max(1, _EVALUATION_SIZE // (2 + len(plan.active)))
    for start in range(0, size, step):
        yield start, min(size, start + step)


def _evaluate(
    plan: _Plan, p: dict, q: dict, shape: tuple, terminals=(0.0, 1.0)
) -> np.ndarray:
    """The plan's value for each element's probability ``p`` and its
    complement ``q`` (by name, each of ``shape``), level by level: the
    probability that the terminals are joined, or with ``terminals``
    ``(1.0, 0.0)`` that they are parted, each to its own precision."""
    size = int(np.prod(shape, dtype=np.int64))
    if plan.root < 2:
        return np.full(shape, terminals[plan.root], dtype=float)
    p_rows, q_rows = _rows(plan, p, size), _rows(plan, q, size)
    out = np.empty(size)
    for start, end in _chunks(plan, size):
        values = _forward(
            plan, p_rows[:, start:end], q_rows[:, start:end], terminals
        )
        out[start:end] = values[plan.root]
    return out.reshape(shape)


def _value_and_gradient(
    plan: _Plan, p: dict, q: dict, shape: tuple, terminals=(0.0, 1.0)
) -> Tuple[np.ndarray, Dict[Hashable, np.ndarray]]:
    """``_evaluate``'s value and its derivative with respect to each of the
    plan's variables' probabilities (its Birnbaum importance), by one pass
    up the levels and one back down."""
    size = int(np.prod(shape, dtype=np.int64))
    if plan.root < 2:
        return np.full(shape, terminals[plan.root], dtype=float), {}
    p_rows, q_rows = _rows(plan, p, size), _rows(plan, q, size)
    value = np.empty(size)
    gradient = np.zeros((len(plan.names), size))
    for start, end in _chunks(plan, size):
        p_, q_ = p_rows[:, start:end], q_rows[:, start:end]
        values = _forward(plan, p_, q_, terminals)
        adjoints = np.zeros_like(values)
        adjoints[plan.root] = 1.0
        for j in range(len(plan.pivots) - 1, -1, -1):
            pivot = int(plan.pivots[j])
            first, last = plan.bounds[j], plan.bounds[j + 1]
            adjoint = adjoints[2 + first : 2 + last]
            active = plan.active[first:last]
            inactive = plan.inactive[first:last]
            # Each evaluation's sum along a row of its own, so that it is
            # the same however many are worked out together.
            change = adjoint * (values[active] - values[inactive])
            gradient[pivot, start:end] += np.ascontiguousarray(change.T).sum(
                axis=1
            )
            np.add.at(adjoints, active, adjoint * p_[pivot])
            np.add.at(adjoints, inactive, adjoint * q_[pivot])
        value[start:end] = values[plan.root]
    return value.reshape(shape), {
        name: gradient[i].reshape(shape) for i, name in enumerate(plan.names)
    }


def _canonical(
    labels: np.ndarray, source: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frontier states (a row of labels each, -1 for a failed node, and the
    labels of the terminals' parts, -1 for one yet to enter) with their
    labels renumbered in order of first appearance, so that equal states
    are equal rows."""
    count, width = labels.shape
    if not count or not width:
        return labels, source, target
    size = int(labels.max()) + 1
    if size <= 0:
        return labels, source, target
    rows = np.arange(count)
    # Each row's new number for each label, given as the labels come.
    rank = np.full((count, size), -1, dtype=labels.dtype)
    seen = np.zeros(count, dtype=labels.dtype)
    for column in range(width):
        label = labels[:, column]
        at = rows[label >= 0]
        label = label[at]
        fresh = rank[at, label] < 0
        at, label = at[fresh], label[fresh]
        rank[at, label] = seen[at]
        seen[at] += 1

    def renamed(label: np.ndarray) -> np.ndarray:
        return np.where(
            label >= 0,
            rank[
                rows.reshape((-1,) + (1,) * (label.ndim - 1)),
                np.maximum(label, 0),
            ],
            label,
        ).astype(labels.dtype)

    return renamed(labels), renamed(source), renamed(target)


def _unique_rows(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The distinct rows of an array of small integers (none below -1),
    and each row's index among them: the rows are packed into 64-bit words
    first, which sort far faster than rows."""
    count, width = rows.shape
    if not count or not width:
        return rows[:1], np.zeros(count, dtype=np.int64)
    values = rows.astype(np.uint64) + np.uint64(1)
    bits = max(1, int(values.max()).bit_length())
    per_word = 64 // bits
    words = []
    for start in range(0, width, per_word):
        word = np.zeros(count, dtype=np.uint64)
        for column in range(start, min(width, start + per_word)):
            word = (word << np.uint64(bits)) | values[:, column]
        words.append(word)
    if len(words) == 1:
        _, first, inverse = np.unique(
            words[0], return_index=True, return_inverse=True
        )
        return rows[first], inverse.reshape(-1)
    order = np.lexsort(words[::-1])
    new = np.ones(count, dtype=bool)
    for word in words:
        ordered = word[order]
        new[1:] |= ordered[1:] != ordered[:-1]
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.cumsum(new) - 1
    return rows[order[new]], inverse


def _frontier_plan(
    order: Sequence[Hashable],
    links: Dict[Hashable, Tuple[Hashable, Hashable]],
    failing: Set[Hashable],
    source: Hashable,
    target: Hashable,
) -> _Plan:
    """The decision diagram of whether ``source`` and ``target`` are
    joined, its links decided in ``order`` (see the module docstring), as
    a ``_Plan``.

    The states before each decision are worked out together, as rows of
    an array (#173): each is the frontier's parts (a label for each
    frontier node, in the order they entered: the same label for nodes
    joined by working links, -1 for a failed node) and the parts that hold
    the source and the target (-1 for a terminal yet to enter). A node
    that can fail is decided before its first link. Each state's two
    outcomes are found at once, equal ones merged, and the diagram is then
    reduced from the last decision up (a state whose outcomes agree is
    dropped, and equal ones shared), which gives the same diagram as
    deciding the states one by one.

    Raises
    ------
    NotImplementedError
        If the diagram would have more than ``MAX_STATES`` states.
    """
    first: Dict[Hashable, int] = {}
    last: Dict[Hashable, int] = {}
    for i, name in enumerate(order):
        for w in links[name]:
            first.setdefault(w, i)
            last[w] = i
    if source not in first or target not in first:
        return _constant(_FAIL)  # no link joins the terminals' parts
    # The decisions: (variable, entering nodes, link ends, leaving nodes),
    # the ends None for a node's decision.
    decisions: List[tuple] = []
    for i, name in enumerate(order):
        u, v = links[name]
        ends = list(dict.fromkeys((u, v)))
        for w in ends:
            if first[w] == i and w in failing:
                decisions.append((w, (), None, ()))
        entering = tuple(w for w in ends if first[w] == i and w not in failing)
        leaving = tuple(w for w in ends if last[w] == i)
        decisions.append((name, entering, (u, v), leaving))
    # Each link's ends' positions once its entering nodes join the
    # frontier, and the positions that stay after its leaving ones go.
    shapes: List[Optional[tuple]] = []
    front: List[Hashable] = []
    for name, entering, ends, leaving in decisions:
        if ends is None:
            shapes.append(None)
            front.append(name)
            continue
        widened = front + list(entering)
        keep = tuple(j for j, w in enumerate(widened) if w not in leaving)
        shapes.append(
            (entering, widened.index(ends[0]), widened.index(ends[1]), keep)
        )
        front = [w for w in widened if w not in leaving]

    # The states before the next decision; each decision's outcomes, as
    # the index of the state after (or _NONE_FAIL, _NONE_WORK).
    labels = np.zeros((1, 0), dtype=np.int16)
    src = np.full(1, -1, dtype=np.int16)
    dst = np.full(1, -1, dtype=np.int16)
    outcomes: List[Tuple[np.ndarray, np.ndarray]] = []
    states = 1
    for k, (name, entering, ends, leaving) in enumerate(decisions):
        count = len(src)
        parts = (labels.max(axis=1, initial=-1) + 1).astype(np.int16)
        if ends is None:
            # A node that can fail, entering the frontier.
            works = (
                np.hstack([labels, parts[:, None]]),
                parts if name == source else src,
                parts if name == target else dst,
                None,
            )
            if name == source or name == target:
                fails = None  # the terminals are parted
            else:
                fails = (
                    np.hstack([labels, np.full((count, 1), -1, np.int16)]),
                    src,
                    dst,
                    None,
                )
            sides = [works, fails]
        else:
            _, iu, iv, keep = shapes[k]  # type: ignore[misc]
            grown, s, t = labels, src, dst
            for j, w in enumerate(entering):
                new = (parts + j).astype(np.int16)
                grown = np.hstack([grown, new[:, None]])
                if w == source:
                    s = new
                if w == target:
                    t = new
            a, b = grown[:, iu], grown[:, iv]
            merge = (a >= 0) & (b >= 0) & (a != b)
            joined = np.where(
                merge[:, None] & (grown == b[:, None]), a[:, None], grown
            )
            s_works = np.where(merge & (s == b), a, s)
            t_works = np.where(merge & (t == b), a, t)
            sides = [
                (
                    joined[:, list(keep)],
                    s_works,
                    t_works,
                    (s_works >= 0) & (s_works == t_works),
                ),
                (grown[:, list(keep)], s, t, None),
            ]
        # Each outcome: joined, parted (a terminal's part closed), or a
        # state after, renumbered.
        codes: List[np.ndarray] = []
        kept: List[Optional[tuple]] = []
        for side in sides:
            code = np.full(count, _NONE_FAIL, dtype=np.int32)
            if side is None:
                codes.append(code)
                kept.append(None)
                continue
            rows, s, t, work = side
            closed = ((s >= 0) & ~(rows == s[:, None]).any(axis=1)) | (
                (t >= 0) & ~(rows == t[:, None]).any(axis=1)
            )
            if work is not None:
                code[work] = _NONE_WORK
                closed &= ~work
                alive = ~closed & ~work
            else:
                alive = ~closed
            codes.append(code)
            kept.append((alive, *_canonical(rows[alive], s[alive], t[alive])))
        stacked = [
            np.hstack([rows, s[:, None], t[:, None]])
            for _, rows, s, t in filter(None, kept)
        ]
        unique, inverse = _unique_rows(np.vstack(stacked))
        offset = 0
        for code, side in zip(codes, kept):
            if side is None:
                continue
            alive = side[0]
            number = int(alive.sum())
            code[alive] = inverse[offset : offset + number]
            offset += number
        outcomes.append((codes[0], codes[1]))
        states += len(unique)
        if states > MAX_STATES:
            raise NotImplementedError(
                "The network's decision diagram has more than "
                f"{MAX_STATES:,} states (ways its frontier can be joined "
                "up), too many to work out exactly. Simulate it: "
                "method='simulate'."
            )
        width = unique.shape[1] - 2
        labels = unique[:, :width]
        src, dst = unique[:, width], unique[:, width + 1]

    # Reduce from the last decision up: each state's slot, the states after
    # the last decision all parted.
    names = [name for name, _, _, _ in decisions]
    slot = np.full(len(src), _FAIL, dtype=np.int64)
    pivots: List[int] = []
    sizes: List[int] = []
    actives: List[np.ndarray] = []
    inactives: List[np.ndarray] = []
    filled = 2
    for k in range(len(decisions) - 1, -1, -1):
        branches = []
        for code in outcomes[k]:
            branches.append(
                np.where(
                    code >= 0,
                    slot[np.maximum(code, 0)] if len(slot) else 0,
                    np.where(code == _NONE_WORK, _WORK, _FAIL),
                )
            )
        active, inactive = branches
        same = active == inactive
        out = np.empty(len(active), dtype=np.int64)
        out[same] = active[same]
        pairs = (active[~same] << 32) | inactive[~same]
        if len(pairs):
            unique_pairs, inverse = np.unique(pairs, return_inverse=True)
            out[~same] = filled + inverse.reshape(-1)
            pivots.append(k)
            sizes.append(len(unique_pairs))
            actives.append(unique_pairs >> 32)
            inactives.append(unique_pairs & 0xFFFFFFFF)
            filled += len(unique_pairs)
        slot = out
    bounds = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    return _Plan(
        names,
        np.array(pivots, dtype=np.int64),
        bounds.astype(np.int64),
        np.concatenate(actives) if actives else np.zeros(0, np.int64),
        np.concatenate(inactives) if inactives else np.zeros(0, np.int64),
        int(slot[0]),
    )
