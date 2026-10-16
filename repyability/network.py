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

The importance measures beyond Birnbaum's (#315) are the diagram's
formulas (``rbd/_measures.py``) over the network's values with an element
held working or failed, from the same diagram replayed; the exact
Fussell-Vesely numerators come from the decision diagrams of each
element's minimal cut sets (``modular._critical_closures``), as a
diagram's core's do. The B-lives invert ``sf``, and the parameter
uncertainty draws the elements' models as a diagram's
(``rbd/uncertainty.py``), each draw worked out exactly.
"""

import json
from collections import deque
from typing import (
    Any,
    Collection,
    Dict,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from repyability._version import __version__
from repyability.rbd import _measures
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd._differential import check as check_differential
from repyability.rbd._differential import shares
from repyability.rbd._mean_lifetime import (
    mean_lifetime,
    model_kinks,
    model_knots,
)
from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd._repairable_uncertainty import half_named
from repyability.rbd._sampling import lifetime_sampler
from repyability.rbd.modular import _critical_closures
from repyability.rbd.non_repairable_rbd import (
    NonRepairableRBD,
    _times,
    check_x,
)
from repyability.rbd.rbd import Pairs
from repyability.rbd.results import UncertaintyResult
from repyability.rbd.routes import AnalysisRoute, Refused
from repyability.rbd.shannon import _minimal_cut_sets, _shannon_plan
from repyability.rbd.uncertainty import (
    check_draws,
    draw_models,
    draw_with,
    fitted_together,
)
from repyability.utils.checks import one_of, real_array
from repyability.utils.deprecation import refuse_removed_names
from repyability.utils.json_io import read_json, write_json
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

# The values of those slots that give the probability that the terminals
# are joined, and that they are parted.
_JOINED, _PARTED = (0.0, 1.0), (1.0, 0.0)

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
    simulation. The importance measures are those of a
    [`NonRepairableRBD`][repyability.NonRepairableRBD], over the links and
    the failing nodes; [`analysis_routes`][repyability.Network.analysis_routes]
    says how each analysis is worked out, and
    [`to_dict`][repyability.Network.to_dict] saves the network.

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
        # The decision diagrams of each element's minimal cut sets, and
        # path sets, once built (see ``_failed_sets``).
        self._closures: Dict[str, tuple] = {}
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

    def _probabilities(
        self, x: np.ndarray, names, models: Optional[Mapping] = None
    ) -> Tuple[dict, dict]:
        """Each named element's probabilities of working and of failing at
        times ``x``, from its model's ``sf`` and ``ff`` (the ``models``
        given by name, else its own)."""
        models = self.models if models is None else models
        p: Dict[Hashable, np.ndarray] = {}
        q: Dict[Hashable, np.ndarray] = {}
        for name in names:
            model = models[name]
            with np.errstate(all="ignore"):
                p[name] = np.asarray(model.sf(x), dtype=float).reshape(x.shape)
                q[name] = np.asarray(model.ff(x), dtype=float).reshape(x.shape)
        return p, q

    def _inputs(
        self, x: np.ndarray, hold: Optional[Mapping[Hashable, bool]] = None
    ) -> Tuple["_Plan", dict, dict]:
        """The plan, and its elements' probabilities of working and of
        failing at times ``x``, those in ``hold`` held working (True: 1
        and 0) or failed."""
        plan = self._decomposition()
        p, q = self._probabilities(x, plan.names)
        for name, works in (hold or {}).items():
            p[name] = np.full(x.shape, 1.0 if works else 0.0)
            q[name] = np.full(x.shape, 0.0 if works else 1.0)
        return plan, p, q

    def _value(
        self,
        x: np.ndarray,
        terminals=_JOINED,
        hold: Optional[Mapping[Hashable, bool]] = None,
        models: Optional[Mapping] = None,
    ) -> np.ndarray:
        """The probability that the terminals are connected at times ``x``,
        or with ``terminals`` ``_PARTED`` that they are not: a sum of
        products, so that a small one keeps its precision. The elements
        in ``hold`` are held working (True) or failed; ``models`` stand in
        for the elements' own, by name (a draw's)."""
        if models is None:
            plan, p, q = self._inputs(x, hold)
        else:
            plan = self._decomposition()
            p, q = self._probabilities(x, plan.names, models)
        return _evaluate(plan, p, q, x.shape, terminals)

    def _birnbaum(
        self, x: np.ndarray, hold: Optional[Mapping[Hashable, bool]] = None
    ) -> Dict[Hashable, np.ndarray]:
        """Each element's Birnbaum importance at times ``x`` (see
        ``birnbaum_importance``), the elements in ``hold`` held working
        (True) or failed; 0 for one in no path."""
        plan, p, q = self._inputs(x, hold)
        works, up = _value_and_gradient(plan, p, q, x.shape)
        fails, down = _value_and_gradient(plan, p, q, x.shape, _PARTED)
        zero = np.zeros(x.shape)
        return {
            name: np.where(
                np.asarray(fails) <= np.asarray(works),
                -np.asarray(down.get(name, zero)),
                np.asarray(up.get(name, zero)),
            )
            * np.ones(x.shape)
            for name in self.models
        }

    def _require_time_varying(self) -> None:
        """Raise if the reliability does not vary with time: every model a
        fixed probability."""
        if self.is_fixed:
            raise Refused(
                "The reliability does not vary with time (every model is a "
                "fixed probability): the time to a reliability is undefined."
            )

    def _require_lifetimes(self, what: str) -> None:
        """Raise if an element is a fixed probability, which has no
        lifetime: the network then has no mean lifetime, and no failure
        density over time."""
        if any(is_fixed_probability(m) for m in self.models.values()):
            raise Refused(
                f"A fixed probability has no lifetime, so the network has "
                f"no {what}."
            )

    def _method(self, method: str, mc_samples) -> str:
        one_of("method", method, ("exact", "simulate"))
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
        return self._value(x, _PARTED)

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

        Raises
        ------
        ValueError, NotImplementedError
            As for ``sf``.
        """
        return self._birnbaum(x)

    @check_x
    def improvement_potential(self, x=None) -> Dict[Hashable, Any]:
        """Each element's improvement potential at time/s ``x``: how much
        the reliability would rise if it were made perfect.

        ``IP_i = R(i working) - R``, worked out as the Birnbaum importance
        times the element's probability of failing (its model's ``ff``),
        so that a small one keeps its precision; as
        [`NonRepairableRBD.improvement_potential`][repyability.NonRepairableRBD.improvement_potential]
        defines it.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.

        Returns
        -------
        dict
            Per link and failing node, by name, its improvement potential:
            floats for a scalar ``x``, arrays otherwise.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> round(bridge.improvement_potential()["a"], 5)
        0.01062
        """
        q = self._probabilities(x, self.models)[1]
        return _measures.improvement_potential(self._birnbaum(x), q)

    @check_x
    def risk_achievement_worth(self, x=None) -> Dict[Hashable, Any]:
        """Each element's risk achievement worth (RAW) at time/s ``x``: the
        factor by which the unreliability would grow if it were failed.

        ``RAW_i = (1 - R(i failed)) / (1 - R)``, each unreliability worked
        out in its own right from the decision diagram with the element
        held failed, as
        [`NonRepairableRBD.risk_achievement_worth`][repyability.NonRepairableRBD.risk_achievement_worth]
        defines it. It is at least 1; inf or nan, with a numpy
        RuntimeWarning, where the unreliability is 0.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.

        Returns
        -------
        dict
            Per link and failing node, by name, its RAW: floats for a
            scalar ``x``, arrays otherwise.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> round(bridge.risk_achievement_worth()["a"], 4)
        5.4414
        """
        return _measures.risk_achievement_worth(
            {
                name: self._value(x, _PARTED, {name: False})
                for name in self.models
            },
            self._value(x, _PARTED),
        )

    @check_x
    def risk_reduction_worth(self, x=None) -> Dict[Hashable, Any]:
        """Each element's risk reduction worth (RRW) at time/s ``x``: the
        factor by which the unreliability would shrink if it were made
        perfect.

        ``RRW_i = (1 - R) / (1 - R(i working))``, each unreliability worked
        out in its own right from the decision diagram with the element
        held working, as
        [`NonRepairableRBD.risk_reduction_worth`][repyability.NonRepairableRBD.risk_reduction_worth]
        defines it. It is at least 1, and inf where a perfect element makes
        the network perfect (a link that alone joins the terminals).

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.

        Returns
        -------
        dict
            Per link and failing node, by name, its RRW: floats for a
            scalar ``x``, arrays otherwise.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> round(bridge.risk_reduction_worth()["a"], 4)
        1.9743
        """
        return _measures.risk_reduction_worth(
            self._value(x, _PARTED),
            {
                name: self._value(x, _PARTED, {name: True})
                for name in self.models
            },
        )

    @check_x
    def criticality_importance(
        self, x=None, kind: str = "failure"
    ) -> Dict[Hashable, Any]:
        """Each element's criticality importance at time/s ``x``.

        With ``B_i`` the Birnbaum importance, ``R_i`` the element's
        reliability and ``R`` the network's, ``kind="failure"`` (the
        default) gives ``CI_i = B_i (1 - R_i) / (1 - R)``: the probability
        that the element has failed and is critical, given that the
        terminals are parted, the share of the network's failures it
        accounts for; ``kind="success"`` gives ``CI_i = B_i R_i / R``, the
        probability that it is working and critical, given that the
        terminals are joined. Each is worked out from the Birnbaum
        importance and the network's unreliability (or reliability) in
        their own right, so a small one keeps its precision; nan where the
        network cannot fail (or cannot work). As
        [`NonRepairableRBD.criticality_importance`][repyability.NonRepairableRBD.criticality_importance]
        defines it.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.
        kind : str, optional
            ``"failure"`` (the default) or ``"success"``.

        Returns
        -------
        dict
            Per link and failing node, by name, its criticality importance:
            floats for a scalar ``x``, arrays otherwise.

        Raises
        ------
        ValueError
            If ``kind`` is neither ``"failure"`` nor ``"success"``, or as
            for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> round(bridge.criticality_importance()["a"], 4)
        0.4935
        """
        one_of("kind", kind, ("failure", "success"))
        p, q = self._probabilities(x, self.models)
        if kind == "success":
            return _measures.criticality_importance(
                self._birnbaum(x), p, self._value(x)
            )
        return _measures.criticality_importance(
            self._birnbaum(x), q, self._value(x, _PARTED)
        )

    @check_x
    def fussell_vesely(
        self, x=None, fv_type: str = "c", method: str = "exact"
    ) -> Dict[Hashable, Any]:
        """Each element's Fussell-Vesely importance at time/s ``x``: the
        share of the network's failure probability that involves it.

        With ``fv_type="c"`` (the default), the probability that some
        minimal cut set containing the element has failed (every element
        in it), over the probability that the terminals are parted; with
        ``fv_type="p"``, the minimal path sets take the cut sets' place.
        ``method="exact"`` (the default) works out the union exactly, from
        a decision diagram of each element's sets (built once), as
        [`NonRepairableRBD.fussell_vesely`][repyability.NonRepairableRBD.fussell_vesely]
        does; ``"rare_event"`` sums the sets' probabilities instead, the
        rare-event approximation, which can exceed 1 when failures are not
        rare. Each probability is worked out in its own right, so a small
        one keeps its precision; nan where the network cannot fail.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.
        fv_type : str, optional
            ``"c"`` (the default) for the minimal cut sets and ``"p"`` for
            the minimal path sets.
        method : str, optional
            ``"exact"`` (the default) or ``"rare_event"``.

        Returns
        -------
        dict
            Per link and failing node, by name, its Fussell-Vesely
            importance: floats for a scalar ``x``, arrays otherwise.

        Raises
        ------
        ValueError
            If ``fv_type`` is not "c" or "p", ``method`` is not "exact" or
            "rare_event", or as for ``sf``.
        NotImplementedError
            As for ``sf`` (with ``method="rare_event"``, as for
            ``cut_sets`` or ``path_sets``).

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> round(bridge.fussell_vesely()["a"], 4)
        0.5065
        >>> round(bridge.fussell_vesely(method="rare_event")["a"], 4)
        0.5112
        """
        if fv_type not in ("c", "p"):
            raise ValueError(
                "fv_type must be either 'c' (cut-set) or 'p' (path-set), "
                f"fv_type={fv_type!r} was given."
            )
        one_of("method", method, ("exact", "rare_event"))
        p, q = self._probabilities(x, self.models)
        if method == "exact":
            numerators = self._failed_sets(fv_type, p, q, x.shape)
        else:
            sets = self.cut_sets() if fv_type == "c" else self.path_sets()
            numerators = {}
            for name in self.models:
                total = np.zeros(x.shape)
                for members in sets:
                    if name in members:
                        total = total + np.prod(
                            [q[m] for m in members], axis=0
                        )
                numerators[name] = total
        return _measures.fussell_vesely(numerators, self._value(x, _PARTED))

    def _failed_sets(
        self, fv_type: str, p: dict, q: dict, shape: tuple
    ) -> Dict[Hashable, np.ndarray]:
        """For each element, the probability that every element of some
        minimal cut set (``fv_type`` ``"c"``) or path set (``"p"``)
        containing it has failed, the numerator of its exact
        Fussell-Vesely importance: a union, from one decision diagram each
        of the sets through the element (see ``modular._critical_closures``:
        the path sets are the dual structure's cut sets), kept once built;
        0 for an element in no such set."""
        if fv_type not in self._closures:
            steps, root = self._decomposition().steps()
            if fv_type == "p":
                steps, root = _dual(steps, root)
            self._closures[fv_type] = (
                _critical_closures((steps, root))
                if root >= 2
                else ([], [], [])
            )
        variables, steps, roots = self._closures[fv_type]
        values = [np.zeros(shape), np.ones(shape)]
        for pivot, failed, works in steps:
            values.append(q[pivot] * values[failed] + p[pivot] * values[works])
        out = {name: np.zeros(shape) for name in self.models}
        for variable, root in zip(variables, roots):
            out[variable] = values[root] * q[variable]
        return out

    def structural_importance(self) -> Dict[Hashable, float]:
        """Each element's structural importance: the fraction of the other
        elements' states in which it is pivotal (the terminals joined when
        it works and parted when it fails), from the structure alone.

        It is the Birnbaum importance with every element's probability at
        1/2, as
        [`RBD.structural_importance`][repyability.RBD.structural_importance]
        works it out, and does not depend on the models.

        Returns
        -------
        dict
            Per link and failing node, by name, its structural importance,
            in ``[0, 1]``.

        Raises
        ------
        NotImplementedError
            If the network's decision diagram would have more than
            ``MAX_STATES`` states.

        Examples
        --------
        The bridge's middle link is pivotal in an eighth of the other
        links' states, each other link in three eighths:

        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> structural = bridge.structural_importance()
        >>> structural["c"], structural["a"]
        (0.125, 0.375)
        """
        plan = self._decomposition()
        half = {name: np.full(1, 0.5) for name in plan.names}
        _, gradient = _value_and_gradient(plan, half, half, (1,))
        return {
            name: float(np.ravel(gradient.get(name, 0.0))[0])
            for name in self.models
        }

    def differential_importance(
        self,
        x=None,
        *,
        change: str = "uniform",
        kind: str = "failure",
        groups: Optional[Mapping[Hashable, Collection[Hashable]]] = None,
    ) -> Dict[Hashable, Any]:
        """Each element's share of the change in the reliability when they
        all change together: the differential importance measure (DIM,
        Borgonovo & Apostolakis, 2001), as
        [`NonRepairableRBD.differential_importance`][repyability.NonRepairableRBD.differential_importance]
        defines it over the components.

        ``change="uniform"`` moves every element's probability by as much,
        and shares out the Birnbaum importance; ``"proportional"`` each by
        the same fraction of itself, and shares out the criticality
        importance of that ``kind``. The shares add up to 1, and a group's
        share is the sum of its members' (``groups``). NaN where there is
        nothing to share out.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.
        change : str, optional
            ``"uniform"`` (the default) or ``"proportional"``.
        kind : str, optional
            With a proportional change, which of each element's
            probabilities moves in proportion: ``"failure"`` (the default),
            of failing, or ``"success"``, of working.
        groups : dict, optional
            ``{name: elements}``: each group's share, the sum of its
            elements', instead of each element's.

        Returns
        -------
        dict
            The shares, by element or group name: floats for a scalar
            ``x``, arrays otherwise.

        Raises
        ------
        ValueError
            For an unknown ``change`` or ``kind``, a group naming an unknown
            element, or as for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> shares = bridge.differential_importance(
        ...     groups={"ends": ["a", "b", "d", "e"], "middle": ["c"]}
        ... )
        >>> {name: round(share, 4) for name, share in shares.items()}
        {'ends': 0.9633, 'middle': 0.0367}
        """
        check_differential("components", change, kind)
        if change == "uniform":
            values = self.birnbaum_importance(x)
        else:
            values = self.criticality_importance(x, kind)
        return shares(values, groups, scalar=x is None or np.ndim(x) == 0)

    def joint_importance(self, x=None) -> Dict[Tuple[Hashable, Hashable], Any]:
        """The joint (second-order) importance of each pair of elements:
        whether improving the two together is worth more than improving
        each (Hong & Lie, 1993), as
        [`NonRepairableRBD.joint_importance`][repyability.NonRepairableRBD.joint_importance]
        defines it.

        ``JRI(i, j) = R(1_i, 1_j) - R(1_i, 0_j) - R(0_i, 1_j) + R(0_i,
        0_j)``: how much element ``j``'s Birnbaum importance rises when
        element ``i`` goes from failed to working, worked out as that
        difference from the decision diagram with ``i`` held. Positive,
        the two are complements (links in a chain); negative, substitutes
        (links side by side); 0 where neither changes what the other is
        worth.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.

        Returns
        -------
        dict
            ``{(i, j): JRI}`` for each pair once, its names in order as
            text, and found either way round: floats for a scalar ``x``,
            arrays otherwise.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> joint = bridge.joint_importance()
        >>> round(joint["a", "d"], 4), round(joint["a", "b"], 4)
        (0.028, -0.972)
        """
        times, scalar = _times(self, x)
        names = list(self.models)
        up = {name: self._birnbaum(times, {name: True}) for name in names}
        down = {name: self._birnbaum(times, {name: False}) for name in names}
        out = Pairs()
        for k, i in enumerate(names):
            for j in names[k + 1 :]:  # noqa: E203
                value = up[i][j] - down[i][j]
                out[Pairs.oriented(i, j)] = (
                    float(np.ravel(value)[0]) if scalar else value
                )
        return out

    def barlow_proschan_importance(self, x=None) -> Dict[Hashable, Any]:
        """Each element's Barlow-Proschan importance: the probability that
        the network's failure is caused by the element's, given that it
        fails by ``x``, or over its whole life, as
        [`NonRepairableRBD.barlow_proschan_importance`][repyability.NonRepairableRBD.barlow_proschan_importance]
        defines it.

        An element's failure parts the terminals when the element is
        critical then, so the network has failed by ``x`` through element
        ``i`` with probability ``integral_0^x B_i(t) f_i(t) dt``, its
        Birnbaum importance times its failure density (its model's ``df``,
        or differences of its ``ff``), integrated by adaptive quadrature
        (to about ``1e-9``, relative); its share of the sum over the
        elements is its importance, and the shares add up to 1.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s by which the network has failed, by default its
            whole life.

        Returns
        -------
        dict
            Per link and failing node, by name, its share: floats for a
            scalar ``x`` or the whole life, arrays otherwise; nan where the
            network cannot have failed.

        Raises
        ------
        ValueError
            For a negative time, or if an element is a fixed probability
            (which has no lifetime).
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        Two links in a chain, with failure rates 1 and 3: the second causes
        three quarters of the failures.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> E = surv.Exponential.from_params
        >>> chain = Network(
        ...     {"a": (0, 1, E([1.0])), "b": (1, 2, E([3.0]))}, 0, 2
        ... )
        >>> shares = chain.barlow_proschan_importance()
        >>> {name: round(share, 6) for name, share in shares.items()}
        {'a': 0.25, 'b': 0.75}
        """
        self._require_lifetimes("failure density over time to share out")
        names = list(self.models)

        def parts(t: np.ndarray) -> np.ndarray:
            importance = self._birnbaum(t)
            columns = []
            for name in names:
                value = importance[name]
                with np.errstate(invalid="ignore"):
                    part = value * _measures.density(self.models[name], t)
                # An element that cannot matter, with an infinite density
                # (a Weibull life of shape below 1, at 0), takes no part.
                columns.append(np.where(value == 0.0, 0.0, part))
            return np.column_stack(columns)

        knots = np.concatenate(
            [np.empty(0)] + [model_knots(m) for m in self.models.values()]
        )
        knots = np.unique(knots[np.isfinite(knots) & (knots > 0.0)])
        if x is None:
            ends = np.array(
                [
                    _measures.life_end(
                        lambda t: self._value(np.asarray(t, float)), knots
                    )
                ]
            )
        else:
            ends = np.atleast_1d(real_array(x, "x")).ravel()
            if np.any(~np.isfinite(ends)) or np.any(ends < 0.0):
                raise ValueError(
                    f"Times must be finite and non-negative, got {x!r}."
                )
        shares_by_end = _measures.integrated_shares(parts, knots, ends)
        if x is None or np.ndim(x) == 0:
            return {
                name: float(shares_by_end[0, j])
                for j, name in enumerate(names)
            }
        return {
            name: shares_by_end[:, j].reshape(np.shape(x))
            for j, name in enumerate(names)
        }

    def time_to_reliability(self, target, upper_bound: Optional[float] = None):
        """The time at which the reliability falls to ``target``: the
        inverse of ``sf``, by root-finding, as
        [`NonRepairableRBD.time_to_reliability`][repyability.NonRepairableRBD.time_to_reliability]
        finds it.

        Parameters
        ----------
        target : float or array_like
            The reliability to solve for, in (0, 1), or several.
        upper_bound : float, optional
            A time at which the reliability is below ``target``; found by
            doubling from 1 if None.

        Returns
        -------
        float or numpy.ndarray
            The time at which ``sf(t) == target``: one for each target, if
            several are given.

        Raises
        ------
        ValueError
            If ``target`` is not in (0, 1), or exceeds the reliability at
            0; if every model is a fixed probability (the reliability does
            not vary with time); or if no upper bound is found.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> chain = Network(
        ...     {"a": (0, 1, surv.Weibull.from_params([100, 2]))}, 0, 1
        ... )
        >>> round(chain.time_to_reliability(0.9), 2)
        32.46
        """
        self._require_time_varying()
        return _measures.invert_reliability(
            lambda t: float(self.sf(t)), target, upper_bound
        )

    def bx_life(self, x, upper_bound: Optional[float] = None):
        """The Bx life: the time by which ``x`` percent of networks have
        failed, ``time_to_reliability(1 - x / 100)``.

        Parameters
        ----------
        x : float or array_like
            The percentage failed, in (0, 100), or several.
        upper_bound : float, optional
            As for ``time_to_reliability``.

        Returns
        -------
        float or numpy.ndarray
            The time by which ``x`` percent have failed: one for each ``x``,
            if several are given.

        Raises
        ------
        ValueError
            If ``x`` is not in (0, 100), or as for ``time_to_reliability``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> chain = Network(
        ...     {"a": (0, 1, surv.Weibull.from_params([100, 2]))}, 0, 1
        ... )
        >>> round(chain.bx_life(10), 2)
        32.46
        """
        x = real_array(x, "x")
        if not np.all((0.0 < x) & (x < 100.0)):
            raise ValueError("x must be a percentage in (0, 100).")
        return self.time_to_reliability(1.0 - x / 100.0, upper_bound)

    # ------------------------------------------------------------------
    # Parameter uncertainty
    # ------------------------------------------------------------------

    def sf_uncertainty(
        self,
        x=None,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
    ) -> UncertaintyResult:
        """The reliability over plausible models of the links and nodes
        (parameter uncertainty), as
        [`NonRepairableRBD.sf_uncertainty`][repyability.NonRepairableRBD.sf_uncertainty]
        gives a diagram's.

        Each of the ``n_draws`` draws gives every uncertain element a
        plausible model, and the reliability at ``x`` is worked out
        exactly for that draw. An element's uncertainty is ``"fit"`` (its
        parameters drawn from the normal approximation of its surpyval
        fit, through its covariance), ``{parameter name: distribution}``,
        or a list of models drawn with replacement; elements given together
        in a tuple share their draws, as links of one type fitted to one
        set of data should.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model, and every drawn
            model, is a fixed probability.
        uncertainty : dict, optional
            ``{element or tuple of elements: uncertainty}`` for the
            uncertain links and nodes; the others keep their models. By
            default, every element whose model is a surpyval fit with a
            parameter covariance is drawn as ``"fit"``, those sharing one
            model object together.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seeds the draws, by default None.
        sampling : str, optional
            ``"random"`` (the default), or ``"sobol"``: the points of a
            scrambled Sobol sequence, which cover the parameters more
            evenly.

        Returns
        -------
        UncertaintyResult
            The reliability of every draw (``samples``), the nominal value
            with the elements' own models, and summaries (``mean``,
            ``interval``, ...; see
            [`UncertaintyResult`][repyability.UncertaintyResult]).

        Raises
        ------
        ValueError
            If ``uncertainty`` is empty, names an unknown element or one
            twice, or a tuple's elements have different models; if an
            uncertainty cannot be drawn; if ``n_draws`` is not a positive
            integer; or if ``x`` is left out for time-varying models.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        A chain of two lines of one type, whose failure rate is known only
        to lie between 0.05 and 0.15 a year, drawn once for both:

        >>> import scipy.stats as st
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = surv.Exponential.from_params([0.1])
        >>> chain = Network({"a": (0, 1, line), "b": (1, 2, line)}, 0, 2)
        >>> rate = {"failure_rate": st.uniform(0.05, 0.1)}
        >>> result = chain.sf_uncertainty(
        ...     1.0, {("a", "b"): rate}, n_draws=10_000, seed=1
        ... )
        >>> round(result.nominal, 4)
        0.8187
        >>> lower, upper = result.interval(0.9)
        >>> round(lower, 3), round(upper, 3)
        (0.748, 0.896)
        """
        drawn = self._uncertain_draws(uncertainty, n_draws, seed, sampling)
        fixed = self.is_fixed and all(
            is_fixed_probability(m) for ms in drawn.values() for m in ms
        )
        if x is None:
            if not fixed:
                raise ValueError(
                    "x is required: an element's probability depends on "
                    "time."
                )
            x = 1.0
        times = np.atleast_1d(real_array(x, "x")).ravel()
        plan = self._decomposition()
        p: Dict[Hashable, np.ndarray] = {}
        q: Dict[Hashable, np.ndarray] = {}
        # Each element's probabilities in every draw, a draw's times
        # together: those of an element left as it is, repeated.
        for name in plan.names:
            models = drawn.get(name, [self.models[name]])
            rows = [
                self._probabilities(times, [name], {name: m}) for m in models
            ]
            p[name] = np.concatenate([row[0][name] for row in rows])
            q[name] = np.concatenate([row[1][name] for row in rows])
            if name not in drawn:
                p[name], q[name] = np.tile(p[name], n_draws), np.tile(
                    q[name], n_draws
                )
        samples = _evaluate(plan, p, q, (n_draws, len(times)))
        nominal = self._value(times)
        if np.ndim(x) == 0:
            return UncertaintyResult(
                samples=samples[:, 0],
                nominal=float(nominal[0]),
                n_draws=n_draws,
            )
        return UncertaintyResult(
            samples=samples.reshape((n_draws,) + np.shape(x)),
            nominal=nominal.reshape(np.shape(x)),
            n_draws=n_draws,
        )

    def mean_uncertainty(
        self,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
    ) -> UncertaintyResult:
        """The mean time until the terminals are parted over plausible
        models of the links and nodes (parameter uncertainty), as
        [`NonRepairableRBD.mean_uncertainty`][repyability.NonRepairableRBD.mean_uncertainty]
        gives a diagram's: each draw's mean worked out as ``mean`` works it
        out, the area under its exact reliability.

        Parameters
        ----------
        uncertainty : dict, optional
            As for ``sf_uncertainty``.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seeds the draws, by default None.
        sampling : str, optional
            ``"random"`` (the default) or ``"sobol"``, as for
            ``sf_uncertainty``.

        Returns
        -------
        UncertaintyResult
            The mean lifetime of every draw (``samples``; ``inf`` for one in
            which the terminals may never part), the exact mean with the
            elements' own models (``nominal``), and summaries.

        Raises
        ------
        ValueError
            As for ``sf_uncertainty``, or if an element is a fixed
            probability (which has no lifetime).
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> import scipy.stats as st
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = surv.Exponential.from_params([0.1])
        >>> chain = Network({"a": (0, 1, line), "b": (1, 2, line)}, 0, 2)
        >>> rate = {"failure_rate": st.uniform(0.05, 0.1)}
        >>> result = chain.mean_uncertainty({("a", "b"): rate}, seed=1)
        >>> round(result.nominal, 6)  # 1 / 0.2
        5.0
        >>> lower, upper = result.interval(0.9)
        >>> round(lower, 2), round(upper, 2)
        (3.43, 9.01)
        """
        self._require_lifetimes("mean lifetime")
        drawn = self._uncertain_draws(uncertainty, n_draws, seed, sampling)
        samples = np.empty(n_draws)
        for i in range(n_draws):
            samples[i] = self._mean_of(
                {
                    name: drawn[name][i] if name in drawn else model
                    for name, model in self.models.items()
                }
            )
        return UncertaintyResult(
            samples=samples,
            nominal=self._mean_of(self.models),
            n_draws=n_draws,
        )

    def _mean_of(self, models: Mapping) -> float:
        """The mean time until the terminals are parted with ``models``
        (by name) for the elements: the exact reliability integrated over
        time by quadrature (to about 1e-10)."""
        return mean_lifetime(
            lambda t: self._value(np.asarray(t, float), models=models),
            [model_knots(m) for m in models.values()],
            [model_kinks(m) for m in models.values()],
        )

    def _label(self, members: tuple) -> str:
        """An uncertain element, or elements, named for a message."""
        if len(members) > 1:
            return f"Elements {list(members)!r}"
        kind = "Link" if members[0] in self.links else "Node"
        return f"{kind} {members[0]!r}"

    def _uncertain_draws(
        self, uncertainty, n_draws: int, seed, sampling: str
    ) -> Dict[Hashable, list]:
        """``n_draws`` plausible models for each uncertain element, after
        checking ``uncertainty`` and ``n_draws`` (see ``sf_uncertainty``):
        the elements given together share their models. Drawn as a
        diagram's nodes' are (``uncertainty.draw_with``), so the same
        sources and seed give the same models."""
        check_draws(n_draws)
        sources = self._uncertainty_sources(uncertainty)

        def draw(n: int, source) -> Dict[Hashable, list]:
            drawn: Dict[Hashable, list] = {}
            for members, spec in sources:
                models = draw_models(
                    self.models[members[0]],
                    spec,
                    n,
                    source,
                    self._label(members),
                )
                for name in members:
                    drawn[name] = models
            return drawn

        return draw_with(draw, n_draws, seed, sampling)

    def _uncertainty_sources(self, uncertainty) -> List[Tuple[tuple, Any]]:
        """The uncertain inputs, checked (see ``sf_uncertainty``): for each
        element or tuple of elements given, its elements and its
        uncertainty. By default, every fitted element
        (``uncertainty.fitted_together``). Given, an element drawn without
        other elements holding the same model object, which are left as
        they are, is warned about."""
        given = uncertainty is not None
        if uncertainty is None:
            uncertainty = fitted_together(self.models)
            if not uncertainty:
                raise ValueError(
                    "No link's or node's model is a surpyval fit with a "
                    "parameter covariance, which the uncertainty is drawn "
                    "from by default: give the uncertain elements, "
                    "uncertainty={link: 'fit'} or {link: {parameter: "
                    "distribution}}, for example."
                )
        if not uncertainty:
            raise ValueError(
                "Give the uncertain elements: uncertainty={link: 'fit'}, for "
                "example."
            )
        sources: List[Tuple[tuple, Any]] = []
        seen: set = set()
        for key, spec in uncertainty.items():
            members = (
                (key,)
                if key in self.models or not isinstance(key, tuple)
                else key
            )
            for name in members:
                if name not in self.models:
                    raise ValueError(
                        f"{name!r} in uncertainty is not a link or a failing "
                        "node of the network."
                    )
                if name in seen:
                    raise ValueError(
                        f"{self._label((name,))} is given an uncertainty "
                        "twice."
                    )
                seen.add(name)
            model = self.models[members[0]]
            for name in members[1:]:
                if not NonRepairableRBD._same_model(self.models[name], model):
                    raise ValueError(
                        f"Elements {list(members)!r} share their "
                        "uncertainty, so they must have the same model."
                    )
            sources.append((tuple(members), spec))
        if given:
            for members, _ in sources:
                model = self.models[members[0]]
                others = [
                    name
                    for name in self.models
                    if name not in seen and self.models[name] is model
                ]
                if others:
                    half_named(
                        self._label(members),
                        others,
                        "model",
                        {tuple(members) + tuple(others): "fit"},
                    )
        return sources

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
        self._require_lifetimes("mean lifetime")
        if self._method(method, mc_samples) == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return float(np.mean(lives))
        return self._mean_of(self.models)

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

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """The network as a JSON-friendly dict (see ``from_dict``), as a
        diagram saves its nodes (see
        [`RBD.to_dict`][repyability.RBD.to_dict]).

        Returns
        -------
        dict
            The ``repyability_version`` that wrote it, the ``type``
            (``"Network"``), the terminals, and the links (each its name,
            the two nodes it joins and its model, serialised as a diagram's
            node models are: surpyval's in surpyval's own format, so a
            fit's covariance round-trips) and the failing nodes.

        Raises
        ------
        NotImplementedError
            If a model cannot be serialised: one that is not surpyval's
            (nor a RePyability node model).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> chain = Network(
        ...     {"a": (0, 1, surv.Weibull.from_params([100, 2]))}, 0, 1
        ... )
        >>> d = chain.to_dict()
        >>> d["type"], d["links"][0]["nodes"]
        ('Network', [0, 1])
        """
        from repyability.rbd.serialisation import serialise_model

        return {
            "repyability_version": __version__,
            "type": "Network",
            "source": self.source,
            "target": self.target,
            "links": [
                {
                    "link": name,
                    "nodes": [u, v],
                    "model": serialise_model(self.models[name]),
                }
                for name, (u, v) in self.links.items()
            ],
            "nodes": [
                {"node": name, "model": serialise_model(model)}
                for name, model in self.nodes.items()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Network":
        """Rebuild a network from ``to_dict``'s output, by calling the
        constructor again, so it is checked again. Tuple names (a grid's
        ``(i, j)``), which JSON turns into lists, are turned back.

        Parameters
        ----------
        d : dict
            As ``to_dict`` returns (possibly after a JSON round trip).

        Returns
        -------
        Network
            The network.

        Raises
        ------
        ValueError
            If ``d`` is not a network's dict, a model kind is unknown, or
            the constructor rejects the stored inputs.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> chain = Network(
        ...     {"a": (0, 1, surv.Weibull.from_params([100, 2]))}, 0, 1
        ... )
        >>> Network.from_dict(chain.to_dict()).sf(50) == chain.sf(50)
        True
        """
        from repyability.rbd.serialisation import (
            _node_name,
            deserialise_model,
        )

        if d.get("type") != "Network":
            raise ValueError(f"Not a network: type {d.get('type')!r}.")
        links = {
            _node_name(e["link"]): (
                _node_name(e["nodes"][0]),
                _node_name(e["nodes"][1]),
                deserialise_model(e["model"]),
            )
            for e in d["links"]
        }
        nodes = {
            _node_name(e["node"]): deserialise_model(e["model"])
            for e in d.get("nodes") or ()
        }
        return cls(
            links,
            _node_name(d["source"]),
            _node_name(d["target"]),
            nodes=nodes or None,
        )

    def to_json(self, fp=None, **json_kwargs) -> Optional[str]:
        """The network as a JSON document (see ``to_dict``): returned, or
        written to ``fp`` (as surpyval's models' ``to_json(fp)`` writes
        them).

        Parameters
        ----------
        fp : str, os.PathLike or file, optional
            A path, or a file opened for writing, to write the document to;
            by default None: it is returned.
        **json_kwargs
            Passed to ``json.dumps``, e.g. ``indent=2``.

        Returns
        -------
        str or None
            The JSON document, or None once written to ``fp``.

        Raises
        ------
        NotImplementedError
            If a model cannot be serialised (see ``to_dict``).
        TypeError
            If a name cannot be encoded as JSON.
        """
        return write_json(json.dumps(self.to_dict(), **json_kwargs), fp)

    @classmethod
    def from_json(cls, s) -> "Network":
        """Rebuild a network from ``to_json``'s output.

        Parameters
        ----------
        s : str, os.PathLike or file
            The JSON document: its text, a path to a file holding it, or a
            file opened for reading.

        Returns
        -------
        Network
            The network.

        Raises
        ------
        ValueError
            If ``s`` is not valid JSON, or for the reasons given in
            ``from_dict``.
        FileNotFoundError
            If ``s`` is neither a JSON document nor the path of a file.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> chain = Network(
        ...     {"a": (0, 1, surv.Weibull.from_params([100, 2]))}, 0, 1
        ... )
        >>> round(Network.from_json(chain.to_json()).sf(50), 4)
        0.7788
        """
        return cls.from_dict(json.loads(read_json(s)))

    # ------------------------------------------------------------------
    # The routes
    # ------------------------------------------------------------------

    def analysis_routes(self) -> Dict[str, AnalysisRoute]:
        """How each analysis of this network is computed, found without
        running it: exactly, numerically, by simulation, or not at all (as
        [`NonRepairableRBD.analysis_routes`][repyability.NonRepairableRBD.analysis_routes]
        reports a diagram's).

        The exact values come from the network's decision diagram, so a
        network of more than ``MAX_STATES`` states refuses every analysis
        but the simulation (and the paths listed), with the message the
        methods raise; an element whose model has no reliability of its
        own makes the analyses built on it refuse, as it does.

        Returns
        -------
        dict[str, AnalysisRoute]
            For each public analysis, by method name: its ``route``
            (``"exact"``, ``"numerical"``, ``"simulated"`` or
            ``"refused"``), the ``reason`` and the elements that decide it
            (see [`AnalysisRoute`][repyability.AnalysisRoute]).

        Examples
        --------
        >>> from repyability import Network
        >>> bridge = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("s", "y", 0.1),
        ...      "c": ("x", "y", 0.1), "d": ("x", "t", 0.1),
        ...      "e": ("y", "t", 0.1)},
        ...     "s", "t",
        ... )
        >>> routes = bridge.analysis_routes()
        >>> routes["fussell_vesely"].route, routes["random"].route
        ('exact', 'simulated')
        >>> routes["mean"].route  # a fixed probability has no lifetime
        'refused'
        """
        from repyability.rbd import routes as r

        elements = {name: r.model_route(m) for name, m in self.models.items()}
        out: Dict[str, AnalysisRoute] = {}

        def give(names, route) -> None:
            for name in names:
                out[name] = route

        def built(route, reason):
            return r.with_nodes(route, reason, elements, "reliabilities")

        give(
            ("sf", "ff"),
            built(
                r.EXACT,
                "The decision diagram over the elements' reliabilities, "
                "exactly.",
            ),
        )
        give(
            (
                "birnbaum_importance",
                "improvement_potential",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "criticality_importance",
            ),
            built(
                r.EXACT,
                "The exact reliability, and with each element held working "
                "and failed, from the decision diagram.",
            ),
        )
        out["fussell_vesely"] = built(
            r.EXACT,
            "The exact probability that a minimal cut set containing each "
            "element has failed, over the unreliability (method="
            "'rare_event' sums the cut sets' probabilities instead).",
        )
        out["joint_importance"] = built(
            r.EXACT,
            "Each element's exact Birnbaum importance with the other held "
            "working, less with it held failed.",
        )
        out["differential_importance"] = built(
            r.EXACT,
            "The exact Birnbaum importance of each element (or, for a "
            "proportional change, its criticality importance), shared out.",
        )
        out["structural_importance"] = AnalysisRoute(
            r.EXACT, "From the structure alone."
        )
        out["cut_sets"] = AnalysisRoute(
            r.EXACT, "Read off the decision diagram."
        )
        paths = r.refusal(self._simple_paths)
        out["path_sets"] = (
            r.refused(paths)
            if paths
            else AnalysisRoute(
                r.EXACT, "Every simple path between the terminals, listed."
            )
        )
        fixed = r.refusal(self._require_time_varying)
        give(
            ("time_to_reliability", "bx_life"),
            (
                r.refused(fixed)
                if fixed
                else built(
                    r.NUMERICAL,
                    "The exact reliability, inverted by root-finding.",
                )
            ),
        )
        lifetimes = r.refusal(lambda: self._require_lifetimes("mean lifetime"))
        out["mean"] = (
            r.refused(lifetimes)
            if lifetimes
            else built(
                r.NUMERICAL,
                "The exact reliability, integrated over time by adaptive "
                "Gauss-Kronrod quadrature (to about 1e-10, relative).",
            )
        )
        densities = r.refusal(
            lambda: self._require_lifetimes(
                "failure density over time to share out"
            )
        )
        out["barlow_proschan_importance"] = (
            r.refused(densities)
            if densities
            else built(
                r.NUMERICAL,
                "Each element's Birnbaum importance times its failure "
                "density, integrated over time by adaptive Gauss-Legendre "
                "quadrature (to about 1e-9).",
            )
        )
        out["random"] = AnalysisRoute(
            r.SIMULATED,
            "Monte-Carlo lifetimes: each element's drawn, and the "
            "longest-lasting path's taken.",
        )
        out["sf_uncertainty"] = built(
            r.SIMULATED,
            "The elements' parameters drawn from their uncertainty, and the "
            "exact reliability for each draw.",
        )
        out["mean_uncertainty"] = (
            r.refused(lifetimes)
            if lifetimes
            else built(
                r.SIMULATED,
                "The elements' parameters drawn from their uncertainty, and "
                "the exact mean lifetime for each draw (the area under its "
                "reliability).",
            )
        )
        # Every exact value needs the decision diagram: without it, only
        # the simulation (and the paths, listed on their own) is left.
        too_large = r.refusal(self._decomposition)
        if too_large:
            for name in out:
                if name not in ("random", "path_sets"):
                    out[name] = r.refused(too_large)
        return out

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


def _dual(steps: list, root: int) -> Tuple[list, int]:
    """The plan (``shannon.py``'s format) of the dual of the structure
    ``steps`` and ``root`` decide: the dual works unless every element of
    some minimal path set of the structure has failed, so its minimal cut
    sets are the structure's minimal path sets. Each step's branches are
    swapped (the elements' states complemented) and the terminals too (the
    value negated)."""

    def swapped(slot: int) -> int:
        return 1 - slot if slot < 2 else slot

    return [
        (pivot, swapped(inactive), swapped(active))
        for pivot, active, inactive in steps
    ], swapped(root)


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
