"""Two-terminal reliability of undirected networks (#104).

Power, pipe and communication networks are undirected, and it is often
their links that fail. Their question is whether two terminals stay
connected: some path of working links (through working nodes, where nodes
can fail too) joins them. A reliability block diagram is directed, with
failing nodes, so a network has a model of its own.

The exact values come from the network's minimal paths: each simple path
between the terminals, as the set of its links and of the nodes on it that
can fail. No such set holds another (two simple paths with the same links
are the same path), so these are the minimal path sets, and the Shannon
decomposition of ``shannon.py`` works out the probability that one of them
works. The simulation draws each element's lifetime and finds, in each
sample, the path that lasts longest: its lifetime is the network's.
"""

from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import numpy as np

from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd._sampling import lifetime_sampler
from repyability.rbd.non_repairable_rbd import check_x
from repyability.rbd.shannon import _minimal_cut_sets, _shannon_plan
from repyability.utils.wrappers import numpy_seed

#: The most simple paths the exact values are worked out from; beyond, they
#: refuse, pointing to the simulation.
MAX_PATHS = 100_000


class Network:
    """An undirected network whose links, and optionally nodes, fail, and
    the reliability of the connection between two of its nodes.

    The network works while some path of working links, through working
    nodes, joins ``source`` to ``target``. A link works both ways. Nodes
    work for ever unless given a model in ``nodes``; the terminals can be
    given one too. Every element fails independently of the others.

    Parameters
    ----------
    links : dict
        Each link's name, mapped to ``(node, node, model)``: the two nodes
        it joins, which must differ, and its lifetime model (anything with
        ``sf`` and ``ff``, such as a fitted surpyval distribution, or a
        ``FixedEventProbability`` for a probability). Two links may join
        the same two nodes.
    source : Hashable
        One terminal: a node of the network.
    target : Hashable
        The other terminal, another node.
    nodes : dict, optional
        The nodes that can fail, each mapped to its model, by default none.
        Their names must differ from the links'.

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
            self._check_model(f"Link {name!r}", model)
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
            self._check_model(f"Node {node!r}", model)
            self.models[node] = model
        self.source, self.target = source, target
        self._adjacent = adjacent
        self.is_fixed = all(
            is_fixed_probability(m) for m in self.models.values()
        )
        self._paths: Optional[List[frozenset]] = None
        self._plan: Optional[tuple] = None

    @staticmethod
    def _check_model(what: str, model) -> None:
        if not (hasattr(model, "sf") and hasattr(model, "ff")):
            raise ValueError(
                f"{what}: the model must have sf and ff (a lifetime "
                f"distribution or a FixedEventProbability), got {model!r}."
            )

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
                        "between its terminals, too many to work out "
                        "exactly. Simulate it: method='simulate'."
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
            If there are more than ``MAX_PATHS`` paths.
        """
        return set(_minimal_cut_sets(self._decomposition()))

    def _decomposition(self) -> tuple:
        """The Shannon decomposition of the probability that some minimal
        path set works (see ``shannon.py``)."""
        if self._plan is None:
            paths = self._simple_paths()
            if not paths:
                self._plan = ([], 0)  # nothing joins the terminals
            else:
                self._plan = _shannon_plan(paths)
        return self._plan

    # ------------------------------------------------------------------
    # Exact
    # ------------------------------------------------------------------

    def _values(self, x: np.ndarray, forced: Optional[tuple] = None):
        """The probabilities that the terminals are connected and that they
        are not at times ``x``, each a sum of products (so a small one keeps
        its precision); ``forced`` holds one element as working (True) or
        failed (False)."""
        steps, root = self._decomposition()
        p: Dict[Hashable, np.ndarray] = {}
        q: Dict[Hashable, np.ndarray] = {}
        for name in {pivot for pivot, _, _ in steps}:
            model = self.models[name]
            with np.errstate(all="ignore"):
                p[name] = np.asarray(model.sf(x), dtype=float).reshape(x.shape)
                q[name] = np.asarray(model.ff(x), dtype=float).reshape(x.shape)
        if forced is not None:
            name, works = forced
            if name in p:
                p[name] = np.full(x.shape, 1.0 if works else 0.0)
                q[name] = 1.0 - p[name]
        out = []
        for terminals in ((0.0, 1.0), (1.0, 0.0)):
            values = [
                np.full(x.shape, terminals[0]),
                np.full(x.shape, terminals[1]),
            ]
            for pivot, active, inactive in steps:
                values.append(
                    p[pivot] * values[active] + q[pivot] * values[inactive]
                )
            out.append(values[root])
        return out[0], out[1]

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
            With ``method="exact"``, if there are more than ``MAX_PATHS``
            paths between the terminals.
        """
        if self._method(method, mc_samples) == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return np.mean(lives[:, None] > x[None, :], axis=0)
        return self._values(x)[0]

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
        return self._values(x)[1]

    @check_x
    def birnbaum_importance(self, x=None) -> Dict[Hashable, Any]:
        """Each element's Birnbaum importance at time/s ``x``: the
        reliability with it working less that with it failed.

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
        out: Dict[Hashable, Any] = {}
        for name in self.models:
            works, _ = self._values(x, (name, True))
            fails, _ = self._values(x, (name, False))
            out[name] = works - fails
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
        from repyability.rbd._mean_lifetime import mean_lifetime, model_knots

        knots = np.concatenate([model_knots(m) for m in self.models.values()])
        return mean_lifetime(
            lambda t: self._values(np.asarray(t, float))[0], knots
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
