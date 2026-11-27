"""Reliability of networks: whether two terminals, several or all the
nodes stay connected (#104).

Power, pipe and communication networks are undirected, and it is often
their links that fail. Their question is whether two terminals stay
connected: some path of working links (through working nodes, where nodes
can fail too) joins them; or whether several sites, or every node, stay
connected to each other (K-terminal and all-terminal reliability). A
reliability block diagram is directed, with failing nodes, so a network
has a model of its own.

The exact values come from a binary decision diagram built from the
network itself (#143), after Hardy, Lucet & Limnios (2007): the links are
decided in an order, and after each, what is left to decide depends only on
how the *frontier* (the nodes with links decided and links to come) is
joined up by the working links so far, and on which parts hold a
terminal: a part that holds one and leaves the frontier parts the
terminals, unless it holds them all. The states before each decision are
worked out together, as
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
the link that joins the terminals when the links are added longest-lived
first (for two terminals, the path that lasts longest): its lifetime is
the network's.

A directed network (``directed=True``: each link works from its first
node to its second) takes two terminals and is decided from its simple
paths from the source to the target, as ``METHOD = "paths"`` decides an
undirected one; ``to_rbd`` gives the reliability block diagram of one
with no cycle. Whether more than two terminals are joined along one-way
links depends on the order they are reached in, which the frontier's
parts do not say, so that is refused.

The importance measures beyond Birnbaum's (#315) are the diagram's
formulas (``rbd/_measures.py``) over the network's values with an element
held working or failed, from the same diagram replayed; the exact
Fussell-Vesely numerators come from the decision diagrams of each
element's minimal cut sets (``modular._critical_closures``), as a
diagram's core's do. The B-lives invert ``sf``, and the parameter
uncertainty draws the elements' models as a diagram's
(``rbd/uncertainty.py``), each draw worked out exactly.

Links and nodes given a life and a repair (a component spec each) are
repaired: each one's curve comes from the repairable machinery
(``rbd/_event_curves.py``: the elements as the components of a private
``RepairableRBD`` evaluated over the network's plan, ``_Joined``), with
no renewal code here, and the connection's availability, unavailability
and failure frequency over time and in the long run follow from the same
diagram at the elements' probabilities. Such a network refuses what needs
a time at which its terminals part for good (``_refuse_repaired``). Its
other elements may be fixed probabilities and lives that are not repaired,
which the same diagram takes as inputs of their own (as a fault tree's,
see ``rbd/_event_curves.py``); with such a life it has no long run
(``_refuse_unrepaired``).

Links given a capacity carry up to it, and the network delivers the
maximum flow between its two terminals: its distribution, at a time, in
the long run and over a window, is worked out in
``rbd/_network_capacity.py`` from the states of the elements that change
the flow.
"""

import dataclasses
import functools
import json
import types
from collections import deque
from typing import (
    TYPE_CHECKING,
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
from numpy.typing import ArrayLike

from repyability._version import __version__
from repyability.rbd import _measures
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _paired
from repyability.rbd._differential import check as check_differential
from repyability.rbd._differential import shares
from repyability.rbd._importances import with_importances
from repyability.rbd._mean_lifetime import (
    mean_lifetime,
    model_kinks,
    model_knots,
)
from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd._repairable_uncertainty import half_named
from repyability.rbd._sampling import lifetime_sampler
from repyability.rbd._units import diagram_unit
from repyability.rbd.modular import _critical_closures
from repyability.rbd.non_repairable_rbd import (
    NonRepairableRBD,
    _times,
    check_x,
)
from repyability.rbd.rbd import Pairs, _set_order
from repyability.rbd.results import (
    CapacityDistribution,
    ConfidenceInterval,
    ImportanceTable,
    UncertaintyResult,
)
from repyability.rbd.routes import AnalysisRoute, Refused
from repyability.rbd.shannon import _minimal_cut_sets, _shannon_plan
from repyability.rbd.uncertainty import (
    band,
    carried,
    check_draws,
    draw_models,
    draw_with,
    fitted_together,
    input_band,
    shares_fit,
)
from repyability.utils.checks import (
    is_number,
    no_distribution,
    one_of,
    real_array,
    simulation_options,
)
from repyability.utils.deprecation import refuse_removed_names
from repyability.utils.json_io import read_json, write_json
from repyability.utils.wrappers import numpy_seed

if TYPE_CHECKING:  # pragma: no cover
    from repyability.rbd.results import Lever, TornadoResult

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

#: The most states the exact capacity distribution is worked out from
#: (each a set of links and nodes working and failed that gives one
#: maximum flow, see ``rbd/_network_capacity.py``): beyond, it refuses,
#: pointing to the simulation of states. Read when a network's capacity
#: is first worked out.
MAX_CAPACITY_STATES = 100_000

#: The most terminals the links' order is tried from (see ``_link_order``):
#: any order gives the same values, a narrower one a smaller diagram.
_ORDER_ROOTS = 8

# Value slots 0 and 1 of a plan: the terminals are parted, joined.
_FAIL, _WORK = 0, 1

# The values of those slots that give the probability that the terminals
# are joined, and that they are parted.
_JOINED, _PARTED = (0.0, 1.0), (1.0, 0.0)

#: The keys a repaired element's spec may have: its life, its repair and,
#: for a hidden failure found by periodic tests, its tests.
ELEMENT_SPEC_KEYS = ("reliability", "repairability", "inspection")

#: The long-run values, which a life that is not repaired has none of.
_LONG_RUN = frozenset(
    {
        "mean_availability",
        "mean_unavailability",
        "system_failure_frequency",
        "capacity_distribution",
    }
)

#: The analyses of a network of repaired elements alone: its values over
#: time and in the long run. A network of lifetimes or probabilities
#: refuses them.
_REPAIRED_OWN = frozenset(
    {
        "point_availability",
        "point_unavailability",
        "mission_availability",
        "mission_unavailability",
        "mean_availability",
        "mean_unavailability",
        "failure_frequency",
        "expected_failures",
        "system_failure_frequency",
    }
)
#: What a network of repaired elements gives besides: what is made of
#: its elements' probabilities of being down at a time, the structure,
#: and saving. Everything else public refuses it (see
#: ``Network._refuse_repaired``), a method added later too.
_REPAIRED_GIVES = _REPAIRED_OWN | frozenset(
    {
        "importances",
        "birnbaum_importance",
        "improvement_potential",
        "risk_achievement_worth",
        "risk_reduction_worth",
        "criticality_importance",
        "fussell_vesely",
        "joint_importance",
        "differential_importance",
        "structural_importance",
        "path_sets",
        "cut_sets",
        "minimal_path_sets",
        "minimal_cut_sets",
        "analysis_routes",
        "to_dict",
        "to_json",
        "capacity_distribution",
        "point_capacity",
        "mission_capacity",
    }
)
#: The analyses that need a time at which the terminals part for good.
_NEEDS_A_LIFETIME = frozenset(
    {
        "sf",
        "ff",
        "Hf",
        "df",
        "hf",
        "mean",
        "random",
        "compare",
        "time_to_reliability",
        "bx_life",
        "barlow_proschan_importance",
        "sf_uncertainty",
        "mean_uncertainty",
        "time_to_reliability_uncertainty",
        "bx_life_uncertainty",
    }
)

# A decision's outcome that is no state: the terminals parted, joined.
_NONE_FAIL, _NONE_WORK = -1, -2


class _TooLarge(NotImplementedError):
    """A network too large to work out exactly: ``what`` is too large,
    the ``limit`` (a name of this module) it passes, and the ``advice``
    for a network of lifetimes."""

    def __init__(self, what: str, limit: str, advice: str):
        super().__init__(f"{what} {advice}")
        self.what = what
        self.limit = limit


@refuse_removed_names
class Network:
    """A network whose links, and optionally nodes, fail, and the
    reliability of the connection between two of its nodes, several, or
    all of them.

    The network works while some path of working links, through working
    nodes, joins ``source`` to ``target``; given ``terminals``, while
    working links join every one of them to the others (K-terminal
    reliability; ``terminals="all"`` for every node, all-terminal
    reliability). A link works both ways, unless the network is
    ``directed``. Nodes work for ever unless given a model in ``nodes``;
    the terminals can be given one too, and a failed terminal parts them.
    Every element fails independently of the others.

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
        nodes. A link that is repaired takes a component spec in place of
        the model, ``{"reliability": life, "repairability": repair}``, as
        a [`RepairableRBD`][repyability.RepairableRBD]'s component (with
        an ``"inspection"`` for a hidden failure found by periodic
        tests): the network then gives the connection's availability and
        how often it is lost (see
        [`point_availability`][repyability.Network.point_availability]).
        The other elements may be probabilities (down with them at every
        time) and lives that are not repaired (down from their failure
        on, so that the network has no long run).
    source : Hashable, optional
        One terminal: a node of the network.
    target : Hashable, optional
        The other terminal, another node. Give both, or ``terminals``.
    nodes : dict, optional
        The nodes that can fail, each mapped to its model (or the
        probability that it has failed, or a repaired node's spec), by
        default none. Their names must differ from the links'.
    terminals : list or "all", optional
        The nodes that must stay connected to each other, two or more, in
        place of ``source`` and ``target``; ``"all"`` for every node of
        the network. Two terminals are the same network as ``source``
        and ``target``.
    directed : bool, optional
        Whether each link works one way only, from its first node to its
        second, by default False. A directed network works while a path
        along its links leads from ``source`` to ``target``; it takes two
        terminals. [`to_rbd`][repyability.Network.to_rbd] gives its
        reliability block diagram.
    capacity : dict, optional
        The most each link carries while it works, by link name, by
        default None: a positive number (``inf`` for no limit). A link
        given none limits nothing. The network's capacity is then the
        maximum flow from ``source`` to ``target`` (see
        [`capacity_distribution`][repyability.Network.capacity_distribution]);
        it takes two terminals.

    Attributes
    ----------
    links : dict
        Each link's two nodes, by name.
    models : dict
        Each element's model (or a repaired one's spec), by name: the
        links', then the nodes'.
    is_repairable : bool
        Whether some elements are repaired.
    terminals : tuple
        The terminals.
    source, target : Hashable
        The two terminals (None for a network of more than two).
    directed : bool
        Whether the links work one way only.
    capacity : dict
        The links' capacities, by name (empty if none was given).
    is_fixed : bool
        Whether every model is a fixed probability (time is then
        irrelevant, and ``x`` may be left out).

    Raises
    ------
    ValueError
        If there is no link, a link is not a ``(node, node, model)`` of two
        different nodes, a model has no ``sf`` or ``ff``, a terminal or a
        node in ``nodes`` is in no link, the terminals are the same node
        or fewer than two, both ``terminals`` and a source or target are
        given, a node has the name of a link, a repaired element's
        spec has a key other than its life, repair and tests, or a
        capacity is not a positive number or is not a link's.
    NotImplementedError
        If a directed network is given more than two terminals, or
        capacities are given with more than two terminals.

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

    All four nodes stay connected to each other with probability
    ``8p^3 - 11p^4 + 4p^5`` (each of the bridge's eight spanning trees,
    less what they share):

    >>> every = Network(
    ...     {name: (u, v, link) for name, (u, v) in bridge.links.items()},
    ...     terminals="all",
    ... )
    >>> round(every.sf(), 5)
    0.97686
    """

    # The kind of diagram, as the messages name it (see ``check_x``).
    _KIND = "network"

    def __init__(
        self,
        links: Dict[Hashable, Tuple[Hashable, Hashable, Any]],
        source: Optional[Hashable] = None,
        target: Optional[Hashable] = None,
        nodes: Optional[Dict[Hashable, Any]] = None,
        *,
        terminals: Any = None,
        directed: bool = False,
        capacity: Optional[Dict[Hashable, float]] = None,
    ):
        if not isinstance(directed, (bool, np.bool_)):
            raise TypeError(
                f"directed must be True or False, got {directed!r}."
            )
        self.directed = bool(directed)
        if not isinstance(links, dict) or not links:
            raise ValueError(
                "links must be a non-empty dict of name: (node, node, model)."
            )
        self.links: Dict[Hashable, Tuple[Hashable, Hashable]] = {}
        self.models: Dict[Hashable, Any] = {}
        adjacent: Dict[Hashable, List[Tuple[Hashable, Hashable]]] = {}
        out: Dict[Hashable, List[Tuple[Hashable, Hashable]]] = {}
        for name, link in links.items():
            if not (isinstance(link, (tuple, list)) and len(link) == 3):
                raise ValueError(
                    f"Link {name!r} must be a (node, node, model), got "
                    f"{link!r}."
                )
            u, v, model = link
            if u == v:
                raise ValueError(f"Link {name!r} joins node {u!r} to itself.")
            model = self._element(f"Link {name!r}", model)
            self.links[name] = (u, v)
            self.models[name] = model
            adjacent.setdefault(u, []).append((name, v))
            adjacent.setdefault(v, []).append((name, u))
            out.setdefault(u, []).append((name, v))
            out.setdefault(v, [])
        self.terminals, self._all_terminals = self._terminals(
            source, target, terminals, adjacent
        )
        if self.directed and len(self.terminals) > 2:
            raise NotImplementedError(
                "A directed network takes two terminals, a source and a "
                f"target: {len(self.terminals)} were given. Whether more "
                "than two are joined along one-way links depends on the "
                "order they are reached in, which the network's decision "
                "diagram does not follow. Give source= and target=, or "
                "leave directed=False if the links work both ways."
            )
        if len(self.terminals) == 2:
            source, target = self.terminals
        else:
            source = target = None
        self.nodes: Dict[Hashable, Any] = dict(nodes or {})
        for node, model in self.nodes.items():
            if node not in adjacent:
                raise ValueError(f"Node {node!r} is in no link.")
            if node in self.links:
                raise ValueError(
                    f"Node {node!r} has the name of a link: name them apart."
                )
            model = self._element(f"Node {node!r}", model)
            self.nodes[node] = self.models[node] = model
        self.source, self.target = source, target
        self._adjacent = adjacent
        # The links out of each node: every link, or in a directed
        # network those that start there.
        self._out = out if self.directed else adjacent
        # The unit the models are in (None if they name none); a warning
        # names the links and nodes when they name different units.
        specs = [e for e, m in self.models.items() if isinstance(m, dict)]
        #: Whether the elements are repaired components, whose connection
        #: comes and goes (see ``point_availability``).
        self.is_repairable = bool(specs)
        self.capacity = self._capacities(capacity)
        # The flow graph and the capacity's states, once worked out, or
        # why they were refused (see ``rbd/_network_capacity.py``).
        self._flow: Any = None
        self._capacity_tree: Any = None
        self._capacity_refused: Optional[str] = None
        self._paths: Optional[List[frozenset]] = None
        self._plan: Optional[_Plan] = None
        self._repaired: Any = None
        if self.is_repairable:
            from repyability.rbd._event_curves import Structured

            # The repaired elements as one diagram's components, over
            # the network's own structure, the others its fixed inputs
            # and lives of their own: their curves, and the values over
            # time.
            fixed, lives = {}, {}
            for name, model in self.models.items():
                if isinstance(model, dict):
                    continue
                if is_fixed_probability(model):
                    fixed[name] = float(
                        np.ravel(np.asarray(model.ff(1.0), dtype=float))[0]
                    )
                else:
                    lives[name] = model
            self._repaired = Structured(
                {name: self.models[name] for name in specs},
                _Joined(self),
                fixed=fixed,
                lives=lives,
            )
            self.unit: Optional[str] = self._repaired.unit
        else:
            self.unit = diagram_unit(self.models, "network")
        self.is_fixed = not specs and all(
            is_fixed_probability(m) for m in self.models.values()
        )
        # The decision diagrams of each element's minimal cut sets, and
        # path sets, once built (see ``_failed_sets``).
        self._closures: Dict[str, tuple] = {}
        # Why the exact values were refused, once found (#229): the search
        # that found it is not repeated.
        self._refused: Optional[str] = None

    @staticmethod
    def _terminals(source, target, terminals, adjacent) -> Tuple[tuple, bool]:
        """The terminals, checked, and whether they are every node
        (``terminals="all"``)."""
        if terminals is None:
            if source is None or target is None:
                raise ValueError(
                    "Give the network's terminals: source= and target=, or "
                    "terminals= (a list of nodes, or 'all' for every node)."
                )
            if source == target:
                raise ValueError("The source and target must be two nodes.")
            found: tuple = (source, target)
            every = False
        else:
            if source is not None or target is not None:
                raise ValueError(
                    "Give source= and target=, or terminals=, not both: "
                    "two terminals are terminals=[source, target]."
                )
            every = isinstance(terminals, str) and terminals == "all"
            if every:
                found = tuple(adjacent)
            elif isinstance(terminals, (str, bytes)) or not isinstance(
                terminals, (list, tuple, set, frozenset)
            ):
                raise ValueError(
                    "terminals must be a list of nodes, or 'all' for every "
                    f"node, got {terminals!r}."
                )
            else:
                found = tuple(dict.fromkeys(terminals))
                if len(found) != len(terminals):
                    raise ValueError(
                        f"terminals names a node twice: {terminals!r}."
                    )
            if len(found) < 2:
                raise ValueError(
                    "terminals must name at least two nodes, got "
                    f"{terminals!r}."
                )
        for terminal in found:
            if terminal not in adjacent:
                raise ValueError(
                    f"Terminal {terminal!r} is in no link of the network."
                )
        return found, every

    def _capacities(self, capacity) -> Dict[Hashable, float]:
        """The links' capacities, checked: each a positive number (``inf``
        for no limit), of a link, in a network of two terminals."""
        if capacity is None:
            return {}
        if not isinstance(capacity, Mapping):
            raise ValueError(
                "capacity must be a dict of link: amount, got "
                f"{capacity!r}."
            )
        if not capacity:
            return {}
        if len(self.terminals) != 2:
            raise NotImplementedError(
                "A network's capacity is the maximum flow between two "
                f"terminals: this one has {len(self.terminals)}. Give "
                "source= and target= with capacity=, or leave capacity "
                "out."
            )
        out: Dict[Hashable, float] = {}
        for link, value in capacity.items():
            if link in self.nodes and link not in self.links:
                raise ValueError(
                    f"{link!r} in capacity is a node: a network's "
                    "capacities are its links' (a node passes whatever "
                    "its links bring while it works). Give the capacity "
                    "to the links that meet it."
                )
            if link not in self.links:
                raise ValueError(
                    f"Unknown link {link!r} in capacity: it is not a link "
                    "of the network."
                )
            if isinstance(value, (bool, np.bool_)) or not is_number(value):
                raise ValueError(
                    f"The capacity of link {link!r} must be a number, got "
                    f"{value!r}."
                )
            if not float(value) > 0.0:
                raise ValueError(
                    f"The capacity of link {link!r} must be positive (inf "
                    f"for no limit), got {value!r}."
                )
            out[link] = float(value)
        return out

    @classmethod
    def _element(cls, what: str, model):
        """An element's model (see ``_model``), or the spec of a repaired
        one: a dict of its ``"reliability"``, its ``"repairability"`` and,
        for a hidden failure found by periodic tests, its
        ``"inspection"``, as a ``RepairableRBD``'s component's (its values
        are checked as a component's are, when the elements' diagram is
        built)."""
        if not isinstance(model, Mapping):
            return cls._model(what, model)
        unknown = set(model) - set(ELEMENT_SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"{what} has key(s) {sorted(map(str, unknown))}: a repaired "
                "element of a network takes 'reliability', 'repairability' "
                "and 'inspection' only. Preventive maintenance, crews, "
                "costs and the rest are not worked out for a network yet: "
                "a RepairableRBD takes them, for a network that can be "
                "drawn as a block diagram."
            )
        return dict(model)

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
        no_distribution(model, what)
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
        """Every simple path between the two terminals (in a directed
        network, from the source to the target along the links), as the
        set of its links and of its nodes that can fail.

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
            around = self._out[node]
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
                    raise _TooLarge(
                        f"The network has more than {MAX_PATHS:,} paths "
                        "between its terminals, too many to list.",
                        "MAX_PATHS",
                        (
                            "A directed network's exact values come from "
                            "its paths: simulate it (method='simulate'), "
                            "or, if it has no cycle, work it out as a "
                            "diagram: to_rbd()."
                            if self.directed
                            else "Its exact values come from its decision "
                            "diagram (network.METHOD = 'bdd', the "
                            "default), or simulate it: method='simulate'."
                        ),
                    )
                continue
            stack.append((other, step, visited | {other}, 0))
        self._paths = paths
        return paths

    def path_sets(self) -> Set[frozenset]:
        """The minimal path sets, as a set: for each simple path between
        two terminals, the set of its links and of its nodes that can fail;
        for more than two terminals, each smallest set of links and nodes
        that joins them all (a tree; a spanning tree for
        ``terminals="all"``), found from the decision diagram
        ([`minimal_path_sets`][repyability.Network.minimal_path_sets]
        lists them, smallest first).

        Returns
        -------
        set[frozenset]
            The minimal path sets, each a frozenset of link and node names.

        Raises
        ------
        NotImplementedError
            If there are more than ``MAX_PATHS``.
        """
        if len(self.terminals) == 2:
            return set(self._simple_paths())
        # A tree joining the terminals is a minimal cut set of the dual
        # structure, which the diagram with its branches swapped decides.
        try:
            return set(
                _minimal_cut_sets(
                    _dual(*self._decomposition().steps()), cap=MAX_PATHS
                )
            )
        except OverflowError:
            raise NotImplementedError(
                f"Listing the network's minimal path sets (the trees that "
                f"join its {len(self.terminals)} terminals) passes "
                f"{MAX_PATHS:,} sets, too many to list. The reliability, "
                "the importance measures and the cut sets do not need "
                "them."
            ) from None

    def cut_sets(self) -> Set[frozenset]:
        """The minimal cut sets, as a set: the smallest sets of links and
        nodes whose failure parts the terminals
        ([`minimal_cut_sets`][repyability.Network.minimal_cut_sets] lists
        them, smallest first).

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

    def minimal_path_sets(self) -> List[frozenset]:
        """The minimal path sets, listed smallest first (then by name), as
        a diagram's [`minimal_path_sets`][repyability.RBD.minimal_path_sets]
        and a tree's list them: the sets of ``path_sets()``.

        Returns
        -------
        list[frozenset]
            The minimal path sets, each a frozenset of link and node names.

        Raises
        ------
        NotImplementedError
            As for ``path_sets``.

        Examples
        --------
        >>> from repyability import Network
        >>> fork = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("x", "t", 0.1),
        ...      "c": ("s", "t", 0.2)},
        ...     "s", "t",
        ... )
        >>> [sorted(path) for path in fork.minimal_path_sets()]
        [['c'], ['a', 'b']]
        """
        return sorted(self.path_sets(), key=_set_order)

    def minimal_cut_sets(self) -> List[frozenset]:
        """The minimal cut sets, listed smallest first (then by name), as
        a diagram's [`minimal_cut_sets`][repyability.RBD.minimal_cut_sets]
        and a tree's list them: the sets of ``cut_sets()``.

        Returns
        -------
        list[frozenset]
            The minimal cut sets, each a frozenset of link and node names.

        Raises
        ------
        NotImplementedError
            As for ``cut_sets``.

        Examples
        --------
        >>> from repyability import Network
        >>> fork = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("x", "t", 0.1),
        ...      "c": ("s", "t", 0.2)},
        ...     "s", "t",
        ... )
        >>> [sorted(cut) for cut in fork.minimal_cut_sets()]
        [['a', 'c'], ['b', 'c']]
        """
        return sorted(self.cut_sets(), key=_set_order)

    def _decomposition(self) -> "_Plan":
        """The plan of the probability that the terminals are joined (see
        ``shannon.py``), as a ``_Plan``: the decision diagram built from the
        network, or with ``METHOD = "paths"``, and for a directed network,
        the Shannon decomposition of its simple paths."""
        if self._plan is None:
            if METHOD not in ("bdd", "paths"):
                raise ValueError(
                    f"network.METHOD must be 'bdd' or 'paths', got "
                    f"{METHOD!r}."
                )
            if self._refused is not None:
                raise NotImplementedError(self._refused)
            try:
                if METHOD == "paths" and len(self.terminals) > 2:
                    raise NotImplementedError(
                        "network.METHOD = 'paths' works out two terminals, "
                        f"from the paths between them: this network has "
                        f"{len(self.terminals)}. Use network.METHOD = "
                        "'bdd' (the default)."
                    )
                if METHOD == "paths" or self.directed:
                    paths = self._simple_paths()
                    # Nothing joins the terminals: they are always parted.
                    self._plan = (
                        _plan_from_steps(*_shannon_plan(paths))
                        if paths
                        else _constant(_FAIL)
                    )
                else:
                    order = _link_order(
                        self.links, self._adjacent, self.terminals
                    )
                    self._plan = _frontier_plan(
                        order, self.links, set(self.nodes), self.terminals
                    )
            except NotImplementedError as refusal:
                if self.is_repairable and isinstance(refusal, _TooLarge):
                    # The advice is to simulate, which is not there yet.
                    refusal = NotImplementedError(
                        f"{refusal.what} A network of repaired elements is "
                        "not simulated yet: the limit, "
                        f"repyability.network.{refusal.limit}, can be "
                        "raised."
                    )
                self._refused = str(refusal)
                raise refusal from None
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
        if self._repaired is not None:
            return self._availabilities(x, names)
        models = self.models if models is None else models
        p: Dict[Hashable, np.ndarray] = {}
        q: Dict[Hashable, np.ndarray] = {}
        # A model takes a row of times; the values are shaped as ``x`` is.
        times = x.ravel()
        for name in names:
            model = models[name]
            with np.errstate(all="ignore"):
                p[name] = np.asarray(model.sf(times), dtype=float).reshape(
                    x.shape
                )
                q[name] = np.asarray(model.ff(times), dtype=float).reshape(
                    x.shape
                )
        return p, q

    def _availabilities(self, x: np.ndarray, names) -> Tuple[dict, dict]:
        """Each named repaired element's probabilities of being up and of
        being down at times ``x``, new at 0, the second worked out in its
        own right (see ``rbd._event_curves.down_at``)."""
        from repyability.rbd._event_curves import down_at

        if (x < 0.0).any():
            raise ValueError(
                "x must be 0 or more: the repaired elements are new at 0."
            )
        down = self._with_fixed(down_at(self._repaired, x.ravel()), x.size)
        q = {name: down[name].reshape(x.shape) for name in names}
        return {name: 1.0 - value for name, value in q.items()}, q

    def _with_fixed(self, down: dict, size: int) -> dict:
        """The elements' probabilities of being ``down`` (``size`` of
        each), with those of the elements given as fixed probabilities:
        down with it at every time."""
        return {
            **down,
            **{
                name: np.full(size, value)
                for name, value in self._repaired._fixed.items()
            },
        }

    def _refuse_unrepaired(self, name: str) -> None:
        """Raise if the long-run analysis ``name`` is asked of a network
        whose repaired elements are mixed with lives that are not
        repaired, which has no long run."""
        lives = sorted(map(str, self._repaired._lives))
        if lives and name in _LONG_RUN:
            over = (
                "point_capacity(x) and mission_capacity(t)"
                if name == "capacity_distribution"
                else "mission_unavailability(t) and expected_failures(t)"
            )
            raise NotImplementedError(
                f"Element(s) {lives} are lives that are not repaired, so "
                "the network has no long run: once such an element has "
                f"failed, it stays failed. Ask for its values over time, "
                f"{over}, or give each a component spec ('reliability' and "
                "'repairability') to have it repaired."
            )

    def _refuse_repaired(self, name: str) -> None:
        """Raise if the analysis ``name`` does not apply to this network's
        elements: those of repaired elements (see ``point_availability``)
        refuse a network of probabilities or lifetimes, and a network of
        repaired elements refuses what needs a time at which its
        terminals part for good, or is not worked out for it yet."""
        if not self.is_repairable:
            if name not in _REPAIRED_OWN:
                return
            raise Refused(
                f"{name}() works on repaired elements, and this network "
                "has none: once its terminals are parted they stay parted "
                "(sf gives the probability that they are still joined at "
                "x). Give the links, and the nodes that can fail, component "
                "specs ('reliability' and 'repairability') to have them "
                "repaired."
            )
        if name in _REPAIRED_GIVES:
            self._refuse_unrepaired(name)
            return
        if name in _NEEDS_A_LIFETIME:
            raise NotImplementedError(
                f"{name}() needs a time at which the terminals part for "
                "good, which a network of repaired elements does not have: "
                "its connection comes and goes as they fail and are "
                "repaired. Ask for point_availability (the probability "
                "that the terminals are joined at x), mean_availability, "
                "failure_frequency or expected_failures."
            )
        raise NotImplementedError(
            f"{name}() is not available for a network of repaired elements "
            "yet. Its values over time and in the long run are "
            "(point_availability, mission_availability, mean_availability, "
            "failure_frequency, expected_failures, "
            "system_failure_frequency), and the importance measures at "
            "its elements' availabilities at a time (importances)."
        )

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
        # ``+ 0.0``: a negated zero (an element in no path) prints as 0.0.
        return {
            name: np.where(
                np.asarray(fails) <= np.asarray(works),
                -np.asarray(down.get(name, zero)),
                np.asarray(up.get(name, zero)),
            )
            * np.ones(x.shape)
            + 0.0
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
        """Raise if every element is a fixed probability: the network then
        parts at the start or never, so it has no lifetimes, no mean
        lifetime and no failure density over time (as a diagram of fixed
        nodes has none). A fixed element among others that age is a
        part that fails at the start or never: it has no density, so no
        part in the failures over time, and the mean is ``inf`` where it
        keeps the terminals joined for good."""
        if self.is_fixed:
            raise Refused(
                "The reliability does not vary with time (every model is a "
                f"fixed probability): the network has no lifetimes, so no "
                f"{what}."
            )

    def _method(self, method: str, mc_samples, seed, name: str) -> str:
        one_of("method", method, ("exact", "simulate"))
        if method == "exact":
            simulation_options(
                f"{name}()", {"mc_samples": mc_samples, "seed": seed}
            )
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
            or ``method`` is unknown, or ``mc_samples`` or ``seed`` is
            given with ``method="exact"``.
        NotImplementedError
            With ``method="exact"``, if the network's decision diagram would
            have more than ``MAX_STATES`` states (with ``METHOD = "paths"``,
            if there are more than ``MAX_PATHS`` paths between the
            terminals).
        """
        if self._method(method, mc_samples, seed, "sf") == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return np.mean(
                lives[:, None] > x.ravel()[None, :], axis=0
            ).reshape(x.shape)
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
        if self._method(method, mc_samples, seed, "ff") == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return np.mean(
                ~(lives[:, None] > x.ravel()[None, :]), axis=0
            ).reshape(x.shape)
        return self._value(x, _PARTED)

    def importances(
        self, x=None, *, simulate: bool = False
    ) -> ImportanceTable:
        """Every importance measure in one table.

        Calls each importance method of this model in turn (Birnbaum,
        improvement potential, risk achievement and reduction worth,
        criticality, Fussell-Vesely, differential, joint, Barlow-Proschan
        and structural, those the model has) with the arguments given, and
        gathers what they return. Each entry of ``measures`` is exactly
        the method's own result; nothing is computed a second way. A
        measure this model cannot work out is named under ``refused`` with
        the reason its method gives (as
        [`analysis_routes`][repyability.Network.analysis_routes] has it), and
        a measure only a simulation gives is listed there as simulated
        unless ``simulate=True``.

        Parameters
        ----------
        x : array_like, optional
            The time/s, as for each measure. May be left out only for a
            network of fixed probabilities.
        simulate : bool, optional
            Whether to run the measures that are simulated. By default
            False: they are listed under ``refused`` with ``"simulated:
            pass simulate=True"``.

        Returns
        -------
        ImportanceTable
            [`ImportanceTable`][repyability.ImportanceTable]: ``measures``
            (``{method name: {component: value}}``), ``refused``
            (``{method name: reason}``) and ``rows()``, one plain dict
            per component for a table. The joint importance, which is per
            pair, is in ``measures`` and not in ``rows()``.

        Raises
        ------
        NotImplementedError
            If no measure can be worked out here; the message gives each
            one's reason.
        ValueError
            If ``simulate`` is not a bool.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> unit = surv.Exponential.from_params([0.01])
        >>> net = Network(
        ...     {
        ...         "a": ("s", "t", unit),
        ...         "b": ("s", "t", unit),
        ...     },
        ...     source="s",
        ...     target="t",
        ... )
        >>> [row["component"] for row in net.importances(50).rows()]
        ['a', 'b']
        """
        from repyability.rbd import _importances

        calls = {
            name: (lambda name: lambda: getattr(self, name)(x))(name)
            for name in _importances.MEASURES
            if name != "structural_importance"
        }
        calls["structural_importance"] = self.structural_importance
        return _importances.table(self, calls, simulate)

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
            For a negative time, or if every element is a fixed probability
            (the network then has no lifetimes). A fixed element among
            others has no density, so no share: it has failed from the
            start, or never does.
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

    def _sf_and_ff(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The probabilities that the terminals are joined and that they
        are parted at times ``x``, each worked out in its own right (so a
        small one keeps its precision)."""
        return self._value(x), self._value(x, _PARTED)

    def _time_scale(self) -> float:
        """The shortest typical failure time over the elements' models (see
        ``NonRepairableRBD._time_scale``), to size a step at time 0 by;
        NaN if none has one."""
        scales = [_measures.model_scale(m) for m in self.models.values()]
        scales = [s for s in scales if np.isfinite(s) and s > 0.0]
        return min(scales) if scales else float("nan")

    @property
    def _init_args(self) -> dict:
        """The constructor's arguments, as ``with_levers`` rebuilds the
        network from them."""
        args: Dict[str, Any] = {
            "links": {
                name: (u, v, self.models[name])
                for name, (u, v) in self.links.items()
            },
            "nodes": dict(self.nodes),
            "directed": self.directed,
        }
        if self.capacity:
            args["capacity"] = dict(self.capacity)
        if self._all_terminals:
            args["terminals"] = "all"
        elif len(self.terminals) == 2:
            args["source"], args["target"] = self.terminals
        else:
            args["terminals"] = list(self.terminals)
        return args

    @check_x
    def Hf(self, x=None):
        """The cumulative hazard ``H(x) = -ln R(x)`` of the connection, ``R``
        its [`sf`][repyability.Network.sf]: exact, ``+inf`` where the
        terminals are certainly parted. While the connection is more
        likely to hold than not it is ``-log1p(-F(x))`` from the
        unreliability ``F``, so that a small one keeps its precision, as
        [`NonRepairableRBD.Hf`][repyability.NonRepairableRBD.Hf] has it.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.

        Returns
        -------
        float or numpy.ndarray
            The cumulative hazard: a float for a scalar ``x``.

        Raises
        ------
        ValueError
            If ``x`` is left out of a network whose models depend on time.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        One link with an exponential life: ``H(x)`` is its rate times
        ``x``.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> link = Network(
        ...     {"a": ("s", "t", surv.Exponential.from_params([0.01]))},
        ...     "s",
        ...     "t",
        ... )
        >>> round(link.Hf(50), 6)
        0.5
        """
        return _measures.cumulative_hazard(*self._sf_and_ff(x))

    @check_x
    def df(self, x=None, dx: float = 1e-6):
        """The density of the time to disconnection, ``f(x) = -dR/dx``.

        By a central finite difference of the exact
        [`sf`][repyability.Network.sf], with step ``dx * |x|``, relative
        to ``x`` (at 0, ``dx`` of the shortest of the elements' typical
        failure times), so that it does not depend on the unit of time, as
        [`NonRepairableRBD.df`][repyability.NonRepairableRBD.df] takes the
        diagram's: the difference is of the unreliability while the
        connection is more likely to hold than not, which keeps its
        precision, the lower point is clipped at 0, and a negative result
        (rounding) is clipped to 0. It assumes the reliability is smooth
        near ``x``, which does not hold where an element's model has a kink
        or a step; a network of probabilities alone has a density of 0.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.
        dx : float, optional
            The relative step of the difference, by default 1e-6.

        Returns
        -------
        float or numpy.ndarray
            The density: a float for a scalar ``x``.

        Raises
        ------
        ValueError
            If ``x`` is left out of a network whose models depend on time.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        One link with an exponential life: ``f(x) = rate * exp(-rate x)``.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> link = Network(
        ...     {"a": ("s", "t", surv.Exponential.from_params([0.01]))},
        ...     "s",
        ...     "t",
        ... )
        >>> round(link.df(50), 6)
        0.006065
        """
        return _measures.density_by_differences(
            self._sf_and_ff,
            x,
            _measures.difference_steps(x, dx, self._time_scale),
        )

    @check_x
    def hf(self, x=None, dx: float = 1e-6):
        """The hazard rate of the connection, ``h(x) = f(x) / R(x)``: the
        numerical density of [`df`][repyability.Network.df] over the exact
        [`sf`][repyability.Network.sf], ``+inf`` where the terminals are
        certainly parted.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.
        dx : float, optional
            The relative step of the difference, passed to ``df``, by
            default 1e-6.

        Returns
        -------
        float or numpy.ndarray
            The hazard rate: a float for a scalar ``x``.

        Raises
        ------
        ValueError
            If ``x`` is left out of a network whose models depend on time.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        One link with an exponential life has a constant hazard rate:

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> link = Network(
        ...     {"a": ("s", "t", surv.Exponential.from_params([0.01]))},
        ...     "s",
        ...     "t",
        ... )
        >>> round(link.hf(50), 6)
        0.01
        """
        sf = np.asarray(self.sf(x), dtype=float)
        density = np.asarray(self.df(x, dx=dx), dtype=float)
        return np.divide(
            density, sf, out=np.full_like(density, np.inf), where=sf > 0
        )

    def parameter_sensitivity(
        self, x=None, rel_step: float = 1e-5
    ) -> Dict[Hashable, Dict[str, Any]]:
        """The sensitivity of the probability that the terminals are joined
        (``sf``) to each parameter of the links' and nodes' models.

        In the guide's Greeks it is the levers' *deltas* (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        For element ``e`` with parameter ``theta``, ``d sf / d theta =
        I_B(e) * d sf_e(x; theta) / d theta``: its Birnbaum importance
        times how far its model's reliability moves with the parameter,
        the latter by a central difference with step ``rel_step *
        |theta|`` (``rel_step`` if ``theta`` is 0), the model rebuilt with
        ``from_params`` (one-sided where one side is not a valid value, nan
        where neither is), as
        [`NonRepairableRBD.parameter_sensitivity`][repyability.NonRepairableRBD.parameter_sensitivity]
        works out the diagram's. An element given as a probability has one
        parameter, ``p``. An element whose model is not a surpyval
        distribution with parameters to rebuild is left out.
        [`levers`][repyability.Network.levers] lists the parameters, with
        their values and ranges, and
        [`with_levers`][repyability.Network.with_levers] builds the network
        with them moved.

        Parameters
        ----------
        x : float or array_like, optional
            The time/s; may be left out if every model is a fixed
            probability.
        rel_step : float, optional
            The relative step of the differences, by default 1e-5.

        Returns
        -------
        dict
            ``{element: {parameter: sensitivity}}``, by surpyval's
            parameter names: floats for a scalar ``x``, arrays otherwise.

        Raises
        ------
        ValueError
            If ``x`` is left out of a network whose models depend on time.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        One link with an exponential life: ``R = exp(-rate x)``, so
        ``dR/d rate = -x exp(-rate x)``, about -9.048 at ``x = 10`` with
        rate 0.01:

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> link = Network(
        ...     {"a": ("s", "t", surv.Exponential.from_params([0.01]))},
        ...     "s",
        ...     "t",
        ... )
        >>> sens = link.parameter_sensitivity(10)
        >>> {name: round(v, 3) for name, v in sens["a"].items()}
        {'failure_rate': -9.048}
        """
        from repyability.rbd._model_utils import lever_spec
        from repyability.rbd.non_repairable_rbd import _dsf_dparam

        times, scalar = _times(self, x)
        birnbaum = self._birnbaum(times)
        out: Dict[Hashable, Dict[str, Any]] = {}
        for name, model in self.models.items():
            spec = lever_spec(model)
            if spec is None:
                continue
            values = {
                parameter: birnbaum[name]
                * _dsf_dparam(spec, j, times, rel_step)
                for j, parameter in enumerate(spec.names)
            }
            out[name] = {
                parameter: float(value[0]) if scalar else value
                for parameter, value in values.items()
            }
        return out

    def levers(self) -> List["Lever"]:
        """The parameters ``parameter_sensitivity`` moves, in the order it
        reports them: each link's and failing node's model's, by surpyval's
        names (``p`` for one given as a probability), each a
        [`Lever`][repyability.Lever] with its value and range, as a
        diagram's ``levers()`` lists them. Nothing is worked out;
        [`with_levers`][repyability.Network.with_levers] sets them. The
        levers are the models' own parameters: those of the units inside a
        composite model are not reached (see the guide's Greeks).

        Returns
        -------
        list of Lever
            The levers, in ``parameter_sensitivity``'s order.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> net = Network(
        ...     {"a": ("s", "t", surv.Weibull.from_params([100, 2]))},
        ...     "s",
        ...     "t",
        ...     nodes={"t": 0.05},
        ... )
        >>> for lever in net.levers():
        ...     print(lever.key, lever.name, lever.value, lever.bounds)
        a alpha 100.0 (0.0, inf)
        a beta 2.0 (0.0, inf)
        t p 0.05 (0.0, 1.0)
        """
        from repyability.rbd import _sensitivity

        return _sensitivity.public_levers(self)

    def with_levers(self, values) -> "Network":
        """A copy of this network with parameters at new values, each moved
        as ``parameter_sensitivity`` moves it: a link's or node's model
        rebuilt with the parameter changed (one given as a probability
        keeps being one). ``levers()`` lists the parameters.

        Parameters
        ----------
        values : dict
            Each parameter's new value, by its
            [`Lever`][repyability.Lever] (from ``levers()``) or by its
            ``(key, name)``, as ``parameter_sensitivity`` reports it.

        Returns
        -------
        Network
            The new network, built as this one was with the parameters
            moved; this one is unchanged.

        Raises
        ------
        ValueError
            For a parameter the network does not have (with the closest
            name), or a value its model refuses (one outside the lever's
            ``bounds``).
        TypeError
            If ``values`` is not a dict, a lever is neither a ``Lever``
            nor a ``(key, name)``, or a value is not a number.

        Examples
        --------
        Twice the Weibull scale: at 100 the link survives with probability
        ``exp(-(100 / 200) ** 2)``, about 0.7788, where it did with
        ``exp(-1)``:

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> net = Network(
        ...     {"a": ("s", "t", surv.Weibull.from_params([100, 2]))},
        ...     "s",
        ...     "t",
        ... )
        >>> longer = net.with_levers({("a", "alpha"): 200.0})
        >>> round(float(longer.sf(100)), 4)
        0.7788
        """
        from repyability.rbd import _sensitivity

        return _sensitivity.with_levers(self, values)

    def tornado(
        self,
        x: Optional[float] = None,
        *,
        quantity: str = "reliability",
        ranges: Optional[dict] = None,
        relative: float = 0.1,
    ) -> "TornadoResult":
        """The numbers of a tornado chart: the network's reliability at
        ``x`` with each parameter at its low and at its high, every other
        at its value, the parameters sorted by how far they swing it.

        As [`NonRepairableRBD.tornado`][repyability.NonRepairableRBD.tornado]
        gives a diagram's: where ``parameter_sensitivity`` gives each
        parameter's slope, this gives what a range of it does, each end
        the network rebuilt with the parameter there (as ``with_levers``
        builds it) and its reliability worked out exactly (``sf``). A
        parameter's range is ``ranges[lever]`` where given, and otherwise
        a share ``relative`` of its scale either way: of its value, or
        for a parameter on a log scale (a LogNormal's ``mu``) of 1. A
        range is cut to the parameter's range of values (``Lever.bounds``:
        a probability's to ``[0, 1]``), and its row says so; a parameter
        at 0 has no relative change, so without a range it is listed in
        ``skipped``, as is one whose network cannot be built or worked
        out at an end, with the reason.

        Parameters
        ----------
        x : float, optional
            One time. May be left out only if every model is a fixed
            probability.
        quantity : str, optional
            ``"reliability"``, the one quantity.
        ranges : dict, optional
            ``{lever: (low, high)}``, each lever by its
            [`Lever`][repyability.Lever] (from ``levers()``) or its
            ``(key, name)``.
        relative : float, optional
            The share of each parameter's scale it moves either way where
            no range is given, in ``(0, 1)``; by default 0.1.

        Returns
        -------
        TornadoResult
            The base reliability, a [`TornadoRow`][repyability.TornadoRow]
            for each parameter, the largest swing first, and those
            skipped.

        Raises
        ------
        ValueError
            If ``x`` is left out for a network that varies with time or is
            not one time, or a range or share is not valid.
        NotImplementedError
            Where ``sf`` refuses the network.

        Examples
        --------
        Two links in series, failing at 0.01 and 0.02 an hour: at 10
        hours the terminals are joined with probability ``exp(-0.3)``,
        and a tenth more or less on the second's rate swings it the more:

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> E = surv.Exponential.from_params
        >>> net = Network(
        ...     {"a": ("s", "x", E([0.01])), "b": ("x", "t", E([0.02]))},
        ...     "s",
        ...     "t",
        ... )
        >>> result = net.tornado(10)
        >>> round(result.base_value, 4)
        0.7408
        >>> for row in result.rows:
        ...     print(row.key, round(row.value_low, 4),
        ...           round(row.value_high, 4))
        b 0.7558 0.7261
        a 0.7483 0.7334
        """
        from repyability.rbd import _sensitivity

        one_of("quantity", quantity, ("reliability",))
        if x is not None and (np.ndim(x) != 0 or not is_number(x)):
            raise ValueError(
                "tornado takes one time x (its rows are sorted by the swing "
                f"there), got {x!r}: call it once for each time."
            )

        def evaluate(net) -> float:
            return float(np.ravel(net.sf(x))[0])

        return _sensitivity._tornado(
            self, evaluate, quantity, ranges, relative
        )

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
            The percentage failed (10 for the B10 life, not a fraction), in
            (0, 100), or several.
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
        drawn, kind = self._uncertain_draws(
            uncertainty, n_draws, seed, sampling
        )
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
                band=kind,
            )
        return UncertaintyResult(
            samples=samples.reshape((n_draws,) + np.shape(x)),
            nominal=nominal.reshape(np.shape(x)),
            n_draws=n_draws,
            band=kind,
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
            As for ``sf_uncertainty``, or if every element is a fixed
            probability (the network then has no lifetimes).
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
        drawn, kind = self._uncertain_draws(
            uncertainty, n_draws, seed, sampling
        )
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
            band=kind,
        )

    def time_to_reliability_uncertainty(
        self,
        target: ArrayLike,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
        upper_bound: Optional[float] = None,
    ) -> UncertaintyResult:
        """The time at which the reliability falls to ``target``, over
        plausible models of the links and nodes (parameter uncertainty), as
        [`NonRepairableRBD.time_to_reliability_uncertainty`][repyability.NonRepairableRBD.time_to_reliability_uncertainty]
        gives a diagram's.

        The models are drawn as ``sf_uncertainty`` draws them, and each
        draw's time is found as ``time_to_reliability`` finds it, from the
        draw's exact reliability.

        Parameters
        ----------
        target : float or array_like
            The reliability level, in (0, 1), or levels: each a row of
            draws, from the same parameter draws.
        uncertainty : dict, optional
            As for ``sf_uncertainty``.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seeds the draws, by default None.
        sampling : str, optional
            ``"random"`` (the default) or ``"sobol"``, as for
            ``sf_uncertainty``.
        upper_bound : float, optional
            An upper bound for each draw's search, at which its reliability
            is below ``target``; found by doubling if None.

        Returns
        -------
        UncertaintyResult
            The time of every draw (``samples``), the time with the
            elements' own models (``nominal``), and summaries such as
            ``interval``.

        Raises
        ------
        ValueError
            If ``target`` is not in (0, 1), every model is a fixed
            probability, a draw's reliability starts below ``target`` or
            cannot be bracketed, or as for ``sf_uncertainty``.

        Examples
        --------
        >>> import scipy.stats as st
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = surv.Exponential.from_params([0.1])
        >>> chain = Network({"a": (0, 1, line), "b": (1, 2, line)}, 0, 2)
        >>> rate = {"failure_rate": st.uniform(0.05, 0.1)}
        >>> result = chain.time_to_reliability_uncertainty(
        ...     0.9, {("a", "b"): rate}, n_draws=1000, seed=1
        ... )
        >>> round(result.nominal, 4)  # 0.1 * 2 per unit time: -ln(0.9) / 0.2
        0.5268
        """
        self._require_time_varying()
        targets = np.asarray(target, dtype=float)
        if not np.all((targets > 0.0) & (targets < 1.0)):
            raise ValueError(
                "target reliability must be in (0, 1) (or an array of "
                f"such), got {target!r}."
            )
        drawn, kind = self._uncertain_draws(
            uncertainty, n_draws, seed, sampling
        )
        flat = np.atleast_1d(targets).ravel()
        out = np.empty((n_draws, len(flat)))
        for i in range(n_draws):
            models = {
                name: drawn[name][i] if name in drawn else model
                for name, model in self.models.items()
            }

            def sf(t: float, models=models) -> float:
                return float(self._value(np.asarray([t]), models=models)[0])

            for j, level in enumerate(flat):
                out[i, j] = _measures.invert_reliability(
                    sf, float(level), upper_bound
                )
        # A row of draws for each target, in its shape (#226).
        samples = (
            out[:, 0]
            if targets.ndim == 0
            else out.reshape((n_draws,) + targets.shape)
        )
        return UncertaintyResult(
            samples=samples,
            nominal=self.time_to_reliability(target, upper_bound),
            n_draws=n_draws,
            band=kind,
        )

    def bx_life_uncertainty(
        self,
        x: ArrayLike,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
        upper_bound: Optional[float] = None,
    ) -> UncertaintyResult:
        """The Bx life, the time by which ``x`` percent of networks have
        failed (the B10 life for ``x = 10``), over plausible models of the
        links and nodes (parameter uncertainty):
        ``time_to_reliability_uncertainty(1 - x / 100, ...)``, as
        [`NonRepairableRBD.bx_life_uncertainty`][repyability.NonRepairableRBD.bx_life_uncertainty]
        gives a diagram's.

        Parameters
        ----------
        x : float or array_like
            The percentage failed, in (0, 100), or percentages: each a row
            of draws, from the same parameter draws.
        uncertainty : dict, optional
            As for ``sf_uncertainty``.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seeds the draws, by default None.
        sampling : str, optional
            ``"random"`` (the default) or ``"sobol"``.
        upper_bound : float, optional
            As for ``time_to_reliability_uncertainty``.

        Returns
        -------
        UncertaintyResult
            The Bx life of every draw, the nominal one, and their
            summaries.

        Raises
        ------
        ValueError
            If ``x`` is not in (0, 100), or as for
            ``time_to_reliability_uncertainty``.

        Examples
        --------
        >>> import scipy.stats as st
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = surv.Exponential.from_params([0.1])
        >>> chain = Network({"a": (0, 1, line), "b": (1, 2, line)}, 0, 2)
        >>> rate = {"failure_rate": st.uniform(0.05, 0.1)}
        >>> b10 = chain.bx_life_uncertainty(
        ...     10, {("a", "b"): rate}, n_draws=1000, seed=1
        ... )
        >>> round(b10.nominal, 4)
        0.5268
        """
        percent = real_array(x, "x")
        if not np.all((0.0 < percent) & (percent < 100.0)):
            raise ValueError("x must be a percentage in (0, 100).")
        return self.time_to_reliability_uncertainty(
            1.0 - percent / 100.0,
            uncertainty,
            n_draws=n_draws,
            seed=seed,
            sampling=sampling,
            upper_bound=upper_bound,
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
    ) -> Tuple[Dict[Hashable, list], str]:
        """``n_draws`` plausible models for each uncertain element, after
        checking ``uncertainty`` and ``n_draws`` (see ``sf_uncertainty``):
        the elements given together share their models. Drawn as a
        diagram's nodes' are (``uncertainty.draw_with``), so the same
        sources and seed give the same models. And what the band over
        them is (``uncertainty.band``)."""
        check_draws(n_draws)
        sources = self._uncertainty_sources(uncertainty)

        def draw(n: int, source) -> Dict[Hashable, list]:
            drawn: Dict[Hashable, list] = {}
            for members, spec in sources:
                first = self.models[members[0]]
                models = draw_models(
                    first, spec, n, source, self._label(members)
                )
                for name in members:
                    drawn[name] = carried(models, first, self.models[name])
            return drawn

        kind = band(
            input_band(self.models[members[0]], spec)
            for members, spec in sources
        )
        return draw_with(draw, n_draws, seed, sampling), kind

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
                theirs = self.models[name]
                if not (
                    NonRepairableRBD._same_model(theirs, model)
                    or shares_fit(theirs, model)
                ):
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
                    if name not in seen
                    and shares_fit(self.models[name], model)
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
            The mean lifetime: ``inf`` if some networks never part (a
            fixed probability that keeps the terminals joined, say, as a
            diagram's ``mean`` gives it).

        Raises
        ------
        ValueError
            If every model is a fixed probability (the network then has
            no lifetimes), ``method`` is unknown, or ``mc_samples`` or
            ``seed`` is given with ``method="exact"``.
        """
        self._require_lifetimes("mean lifetime")
        if self._method(method, mc_samples, seed, "mean") == "simulate":
            lives = self.random(
                10_000 if mc_samples is None else mc_samples, seed
            )
            return float(np.mean(lives))
        return self._mean_of(self.models)

    def compare(
        self,
        other: "Network",
        mc_samples: int = 100_000,
        seed=None,
        *,
        confidence: float = 0.95,
        method: Optional[str] = None,
    ) -> ConfidenceInterval:
        """How much longer (or shorter) this network keeps its terminals
        joined than ``other`` on average: exactly where ``mean`` works out
        both, and otherwise by simulation with common random numbers, as
        [`NonRepairableRBD.compare`][repyability.NonRepairableRBD.compare]
        compares two diagrams.

        Simulated, both networks' lifetimes are drawn ``mc_samples``
        times, a link or node of the same name in both drawing the same
        random numbers (the same quantile of its own model), so the
        differences come from how the networks differ, not from chance.
        The draws are keyed by name for the comparison alone: ``random``
        draws as it did.

        Parameters
        ----------
        other : Network
            The network to compare with.
        mc_samples : int, optional
            Simulated, the number of lifetimes of each, by default 100_000.
        seed : int, optional
            Simulated, the seed, by default None.
        confidence : float, optional
            The confidence level of the interval, by default 0.95.
        method : str, optional
            None (the default): exact where both means are, else simulated.
            ``"exact"``: refused where a mean is not exact.
            ``"simulate"``: simulated whatever.

        Returns
        -------
        ConfidenceInterval
            This network's mean lifetime minus ``other``'s: exact
            (``method="exact"``, with no error), or the mean of the paired
            differences (``"simulated"``), with its standard error and a
            normal interval.

        Raises
        ------
        TypeError
            If ``other`` is not a ``Network``.
        ValueError
            If ``mc_samples``, ``confidence`` or ``method`` is not valid,
            or a network has no lifetimes (every model a probability).
        NotImplementedError
            Simulated, if an element's draws cannot be replayed from
            uniforms.

        Examples
        --------
        A second link alongside the first, each failing at 0.01: the mean
        rises from 100 to 150.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> E = surv.Exponential.from_params
        >>> one = Network({"a": ("s", "t", E([0.01]))}, "s", "t")
        >>> two = Network(
        ...     {"a": ("s", "t", E([0.01])), "b": ("s", "t", E([0.01]))},
        ...     "s",
        ...     "t",
        ... )
        >>> gain = two.compare(one)
        >>> round(gain.estimate, 6), gain.method
        (50.0, 'exact')
        """
        _paired.check(method, mc_samples, confidence)
        _paired.same_kind(
            self,
            other,
            "a network compares with another network (a diagram's compare "
            "takes diagrams)",
        )
        for network in (self, other):
            network._require_lifetimes("mean lifetime")
        if method != "simulate":
            try:
                exact = float(self.mean() - other.mean())
            except (NotImplementedError, ValueError):
                if method == "exact":
                    raise
            else:
                return _paired.exact(exact, confidence)
        key = _paired.key(seed)
        differences = self._keyed_lifetimes(
            mc_samples, key
        ) - other._keyed_lifetimes(mc_samples, key)
        return _paired.simulated(differences, confidence)

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
        return self._lifetimes(lives, size)

    def _keyed_lifetimes(self, n: int, key: int) -> np.ndarray:
        """``n`` lifetimes in which each element draws its uniforms from a
        stream of its own, keyed by ``key`` and its name (common random
        numbers, see ``compare``)."""
        self._require_replayable()
        lives: Dict[Hashable, np.ndarray] = {}
        for name, model in self.models.items():
            sampler = lifetime_sampler(model)
            assert sampler is not None
            u = np.column_stack(
                [
                    _paired.element(key, name, j, n)
                    for j in range(sampler.width)
                ]
            )
            lives[name] = np.asarray(sampler.draw(u), dtype=float)
        return self._lifetimes(lives, n)

    def _require_replayable(self) -> None:
        """Raise unless every element's draws can be replayed from
        uniforms, as common random numbers (``compare``) need."""
        for name, model in self.models.items():
            if lifetime_sampler(model) is None:
                raise NotImplementedError(
                    "Common random numbers need every element's draws to be "
                    "replayable from uniforms (surpyval parametric "
                    "distributions and the composite models built from "
                    f"them); those of {name!r} are not."
                )

    def _lifetimes(
        self, lives: Dict[Hashable, np.ndarray], size: int
    ) -> np.ndarray:
        """The network's ``size`` lifetimes, each element's in ``lives``:
        the life of the link that joins the terminals, the links added
        longest-lived first (for two terminals, the longest-lasting
        path's)."""
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
        out = np.empty(size)
        if self.directed:
            s, t = index[self.source], index[self.target]
            for i in range(size):
                # The nodes reached from the source along the links so far.
                after: List[List[int]] = [[] for _ in index]
                reached = [False] * len(index)
                reached[s] = True
                out[i] = -np.inf
                for j in order[i]:
                    a, b = int(ends[j, 0]), int(ends[j, 1])
                    after[a].append(b)
                    if reached[a] and not reached[b]:
                        reached[b] = True
                        stack = [b]
                        while stack:
                            for c in after[stack.pop()]:
                                if not reached[c]:
                                    reached[c] = True
                                    stack.append(c)
                    if reached[t]:
                        out[i] = last[i, j]
                        break
            return np.maximum(out, 0.0)
        terminal = [False] * len(index)
        for node in self.terminals:
            terminal[index[node]] = True
        for i in range(size):
            parent = list(range(len(index)))

            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            # Whether each part holds a terminal, by its root, and how
            # many parts do: one, once the terminals are joined.
            holds = list(terminal)
            apart = len(self.terminals)
            out[i] = -np.inf
            for j in order[i]:
                a, b = find(ends[j, 0]), find(ends[j, 1])
                if a != b:
                    parent[a] = b
                    apart -= holds[a] and holds[b]
                    holds[b] = holds[a] or holds[b]
                if apart == 1:
                    out[i] = last[i, j]
                    break
        return np.maximum(out, 0.0)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Repaired elements
    # ------------------------------------------------------------------

    def _repaired_times(self, t, name: str) -> Tuple[np.ndarray, tuple]:
        """The times ``t`` of the analysis ``name`` of repaired elements
        as a row, and their shape (``()`` for a number), once the
        analysis applies."""
        self._refuse_repaired(name)
        times = real_array(t, "t")
        if (times < 0.0).any():
            raise ValueError(
                "t must be 0 or more: the repaired elements are new at 0."
            )
        return times.reshape(-1), times.shape

    @staticmethod
    def _shaped(values, shape: tuple):
        """``values`` (a row) in ``shape``: a float for a number."""
        values = np.asarray(values, dtype=float).reshape(shape)
        return float(values) if shape == () else values

    def point_availability(self, x: ArrayLike):
        """The probability that the terminals are connected at ``x``,
        ``A(x)``, the elements repaired and new at 0.

        Each element is down at ``x`` with its probability as a
        [`RepairableRBD`][repyability.RepairableRBD]'s component is (in
        closed form for an exponential life and repair, ``lambda /
        (lambda + mu) (1 - exp(-(lambda + mu) x))``, else from its
        renewal cycle's curve, to about 1e-7), independently of the
        others, and up with one less; the connection's probability
        follows from the network's decision diagram at those, exactly,
        so it is one less
        [`point_unavailability`][repyability.Network.point_unavailability]
        to rounding.

        Parameters
        ----------
        x : array_like
            Time/s, 0 or more: a number or an array.

        Returns
        -------
        float or numpy.ndarray
            ``A(x)``, a float for a number ``x`` and an array for an
            array.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``),
            or a time is negative.
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        Two lines between a plant and a town, each failing at rate 0.01
        and repaired at rate 0.5: either carries the load, so the town is
        cut off with probability ``q(x) ** 2``, ``q(x) = (1 / 51) (1 -
        exp(-0.51 x))``.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = {
        ...     "reliability": surv.Exponential.from_params([0.01]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> net = Network(
        ...     {"north": ("plant", "town", line),
        ...      "south": ("plant", "town", line)},
        ...     "plant", "town",
        ... )
        >>> round(net.point_availability(10.0), 6)
        0.99962
        >>> f"{net.point_unavailability(10.0):.6e}"
        '3.797938e-04'
        """
        times, shape = self._repaired_times(x, "point_availability")
        return self._shaped(self._value(times), shape)

    def point_unavailability(self, x: ArrayLike):
        """The probability that the terminals are parted at ``x``,
        ``Q(x) = 1 - A(x)``, the elements repaired and new at 0 (see
        [`point_availability`][repyability.Network.point_availability]).

        It is worked out as itself, from each element's probability of
        being down (its own closed form or curve) through the decision
        diagram's sum of products, not as one less the availability, so a
        small one keeps its digits.

        Parameters
        ----------
        x : array_like
            Time/s, 0 or more: a number or an array.

        Returns
        -------
        float or numpy.ndarray
            ``Q(x)``, a float for a number ``x`` and an array for an
            array.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``),
            or a time is negative.
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        Three lines in parallel, each down with probability about
        ``2e-6`` in the long run: the connection is lost with their
        product, far below what one less an availability could hold.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = {
        ...     "reliability": surv.Exponential.from_params([1e-6]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> net = Network(
        ...     {name: ("s", "t", line) for name in "abc"}, "s", "t"
        ... )
        >>> f"{net.point_unavailability(1000.0):.4e}"
        '8.0000e-18'
        """
        times, shape = self._repaired_times(x, "point_unavailability")
        return self._shaped(self._value(times, _PARTED), shape)

    def mission_availability(self, t: ArrayLike):
        """The average of the point availability over ``[0, t]``: the
        expected fraction of the window during which the terminals are
        connected, the elements repaired and new at 0.

        It is one less
        [`mission_unavailability`][repyability.Network.mission_unavailability],
        which is integrated to its own size, so the two agree to
        rounding.

        Parameters
        ----------
        t : array_like
            The window/s' ends, 0 or more: a number or an array.

        Returns
        -------
        float or numpy.ndarray
            The average, a float for a number ``t`` and an array for an
            array.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``),
            or a time is negative.
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = {
        ...     "reliability": surv.Exponential.from_params([0.01]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> net = Network({"a": ("s", "t", line)}, "s", "t")
        >>> round(net.mission_availability(100.0), 5)
        0.98078
        """
        times, shape = self._repaired_times(t, "mission_availability")
        parted = self._repaired.mission_unavailability(times)
        return self._shaped(1.0 - np.asarray(parted, dtype=float), shape)

    def mission_unavailability(self, t: ArrayLike):
        """The average of the point unavailability over ``[0, t]``: the
        expected fraction of the window during which the terminals are
        parted, the elements repaired and new at 0.

        ``Q(x)`` (see
        [`point_unavailability`][repyability.Network.point_unavailability])
        is integrated on the pieces between the elements' curves' bends,
        halved until the quadrature agrees with its halves', to its own
        size, so a small one keeps its digits (as
        [`mission_unavailability`][repyability.RepairableRBD.mission_unavailability]
        integrates a diagram's).

        Parameters
        ----------
        t : array_like
            The window/s' ends, 0 or more: a number or an array.

        Returns
        -------
        float or numpy.ndarray
            The average, a float for a number ``t`` and an array for an
            array.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``),
            or a time is negative.
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = {
        ...     "reliability": surv.Exponential.from_params([0.01]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> net = Network({"a": ("s", "t", line)}, "s", "t")
        >>> round(net.mission_unavailability(100.0), 5)
        0.01922
        """
        times, shape = self._repaired_times(t, "mission_unavailability")
        return self._shaped(
            self._repaired.mission_unavailability(times), shape
        )

    def mean_availability(self) -> float:
        """The long-run fraction of time the terminals are connected, the
        elements repaired: the network's decision diagram at each
        element's long-run availability, ``MTTF / (MTTF + MTTR)``,
        exactly (as
        [`mean_availability`][repyability.RepairableRBD.mean_availability]
        gives a diagram's).

        Returns
        -------
        float
            The long-run availability of the connection.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``).
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        The bridge network, every link with a Weibull life of mean 90.27
        and a mean repair of 10, so up ``p = 0.90027`` of the time in the
        long run: ``2p^2 + 2p^3 - 5p^4 + 2p^5``.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> link = {
        ...     "reliability": surv.Weibull.from_params([100.0, 1.5]),
        ...     "repairability": surv.Exponential.from_params([0.1]),
        ... }
        >>> bridge = Network(
        ...     {"a": ("s", "x", link), "b": ("s", "y", link),
        ...      "c": ("x", "y", link), "d": ("x", "t", link),
        ...      "e": ("y", "t", link)},
        ...     "s", "t",
        ... )
        >>> round(bridge.mean_availability(), 4)
        0.9786
        """
        self._refuse_repaired("mean_availability")
        return float(self._repaired.mean_availability())

    def mean_unavailability(self) -> float:
        """The long-run fraction of time the terminals are parted, the
        elements repaired: one less
        [`mean_availability`][repyability.Network.mean_availability],
        worked out in its own right from each element's long-run
        unavailability, ``MTTR / (MTTF + MTTR)``, so a small one keeps
        its digits.

        Returns
        -------
        float
            The long-run unavailability of the connection.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``).
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = {
        ...     "reliability": surv.Exponential.from_params([1e-6]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> net = Network(
        ...     {name: ("s", "t", line) for name in "abc"}, "s", "t"
        ... )
        >>> f"{net.mean_unavailability():.4e}"
        '8.0000e-18'
        """
        self._refuse_repaired("mean_unavailability")
        return float(self._repaired.mean_unavailability())

    def failure_frequency(self, t: ArrayLike):
        """How often the connection is lost at ``t``, ``w(t)``: the
        expected number of times per unit time that the terminals are
        parted then, the elements repaired and new at 0.

        An element's failure parts the terminals when the element is
        critical then: when they are joined with it up and parted with it
        down. The elements failing and being repaired independently,
        that is so with probability ``I_B^i(t)``, its Birnbaum importance
        at the elements' availabilities then, so

        ```text
        w(t) = sum_i I_B^i(t) w_i(t)
        ```

        ``w_i(t)`` the element's failure intensity: ``lambda A_i(t)`` for
        one of constant failure rate (it fails only while it works),
        else the rate of change of its expected failures, by differences
        on its renewal curve's grid. Its integral is
        [`expected_failures`][repyability.Network.expected_failures],
        and its long-run value
        [`system_failure_frequency`][repyability.Network.system_failure_frequency].

        Parameters
        ----------
        t : array_like
            Time/s, 0 or more: a number or an array.

        Returns
        -------
        float or numpy.ndarray
            ``w(t)``, a float for a number ``t`` and an array for an
            array.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``),
            or a time is negative.
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        Two links one after the other, failing at rates 0.01 and 0.02 and
        repaired at rate 0.5, once settled: ``A_2 w_1 + A_1 w_2``, ``w_i =
        lambda_i A_i``.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> E = surv.Exponential.from_params
        >>> net = Network(
        ...     {"a": ("s", "x", {"reliability": E([0.01]),
        ...                       "repairability": E([0.5])}),
        ...      "b": ("x", "t", {"reliability": E([0.02]),
        ...                       "repairability": E([0.5])})},
        ...     "s", "t",
        ... )
        >>> round(net.failure_frequency(1000.0), 6)
        0.028281
        >>> round(net.system_failure_frequency(), 6)
        0.028281
        """
        from repyability.rbd._event_curves import failure_frequency

        times, shape = self._repaired_times(t, "failure_frequency")
        return self._shaped(failure_frequency(self._repaired, times), shape)

    def expected_failures(self, t: ArrayLike):
        """The expected number of times the connection is lost in ``[0,
        t)``, the elements repaired and new at 0: the integral of
        [`failure_frequency`][repyability.Network.failure_frequency], as
        [`expected_failures`][repyability.RepairableRBD.expected_failures]
        counts a diagram's system failures (each element's expected
        failures from its renewal equation, weighed by its Birnbaum
        importance over time, and extended exactly once the elements'
        curves have settled).

        Parameters
        ----------
        t : array_like
            The window/s' ends, 0 or more: a number or an array.

        Returns
        -------
        float or numpy.ndarray
            The expected count, a float for a number ``t`` and an array
            for an array.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``),
            or a time is negative.
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        One link failing at rate 0.01, repaired at rate 0.5: ``(lambda mu
        / (lambda + mu)) t + (lambda / (lambda + mu)) ** 2 (1 -
        exp(-(lambda + mu) t))``.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> E = surv.Exponential.from_params
        >>> net = Network(
        ...     {"a": ("s", "t", {"reliability": E([0.01]),
        ...                       "repairability": E([0.5])})},
        ...     "s", "t",
        ... )
        >>> round(net.expected_failures(100.0), 5)
        0.98078
        """
        times, shape = self._repaired_times(t, "expected_failures")
        return self._shaped(self._repaired.expected_failures(times), shape)

    def system_failure_frequency(self) -> float:
        """How often the connection is lost in the long run: the expected
        number of times per unit time that the terminals are parted once
        the repaired elements have settled, ``sum_i I_B^i w_i`` at their
        long-run values (see
        [`failure_frequency`][repyability.Network.failure_frequency]),
        as
        [`system_failure_frequency`][repyability.RepairableRBD.system_failure_frequency]
        gives a diagram's. Its reciprocal is the mean time between losses
        of the connection.

        Returns
        -------
        float
            Expected losses of the connection per unit time, in the long
            run.

        Raises
        ------
        ValueError
            If the network's elements are not repaired (a ``Refused``).
        NotImplementedError
            If the network is too large to work out exactly.

        Examples
        --------
        Two lines side by side, failing at rates 0.01 and 0.02 and
        repaired at rate 0.5: ``q_1 w_2 + q_2 w_1``.

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> E = surv.Exponential.from_params
        >>> net = Network(
        ...     {"a": ("s", "t", {"reliability": E([0.01]),
        ...                       "repairability": E([0.5])}),
        ...      "b": ("s", "t", {"reliability": E([0.02]),
        ...                       "repairability": E([0.5])})},
        ...     "s", "t",
        ... )
        >>> f"{net.system_failure_frequency():.6e}"
        '7.541478e-04'
        """
        self._refuse_repaired("system_failure_frequency")
        return float(self._repaired.system_failure_frequency())

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def capacity_distribution(
        self,
        x: Optional[ArrayLike] = None,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> CapacityDistribution:
        """The distribution of what the network can deliver from the
        source to the target: at time/s ``x``, or, with repaired
        elements, in the long run.

        Each link carries up to its ``capacity`` (given when the network
        was built) while it works, either way unless the network is
        directed; a link given none limits nothing, and a failed link, or
        a link at a failed node, carries nothing. The delivered amount is
        the maximum flow between the two terminals, so it is positive
        exactly when they are connected.

        The exact distribution enumerates the states of the links and
        nodes that change the maximum flow, each with its probability (a
        product of the elements'), so it is as precise as they are. A
        network of more than ``MAX_CAPACITY_STATES`` (100,000) such
        states refuses, pointing to ``method="simulate"``, which draws
        the elements' states instead. See the guide's
        [Networks](../guide/networks.md).

        Parameters
        ----------
        x : array_like, optional
            Time/s, a number or an array, for a network of lifetimes; may
            be left out if every model is a fixed probability. A network
            of repaired elements takes none: this is its long-run
            distribution, and
            [`point_capacity`][repyability.Network.point_capacity] gives
            the distribution at a time.
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``: the shares of
            ``mc_samples`` states drawn from the elements' probabilities.
        mc_samples : int, optional
            The number of states drawn, by default 10_000.
        seed : int, optional
            Seeds the draws, by default None.

        Returns
        -------
        CapacityDistribution
            The capacities the network can have and their probabilities
            (one per level, or for an array ``x`` one row per level and
            one column per time): ``meets(demand)`` is the probability
            that a demand is met, ``mean`` the expected capacity and
            ``delivered_fraction(demand)`` the expected share of a demand
            delivered. In the long run, the fraction of time at each
            level.

        Raises
        ------
        ValueError
            If no link has a capacity (a ``Refused``), ``x`` is left out
            of a network whose models depend on time or given to a
            network of repaired elements, ``method`` is unknown, or
            ``mc_samples`` or ``seed`` is given with ``method="exact"``.
        NotImplementedError
            With ``method="exact"``, if the capacity takes more than
            ``MAX_CAPACITY_STATES`` states to work out.

        Examples
        --------
        Two pipelines from a field to a plant, of 60 and 40 units, each
        working with probability 0.9:

        >>> from repyability import Network
        >>> net = Network(
        ...     {"a": ("field", "plant", 0.1), "b": ("field", "plant", 0.1)},
        ...     "field", "plant",
        ...     capacity={"a": 60, "b": 40},
        ... )
        >>> capacity = net.capacity_distribution()
        >>> capacity.levels.tolist()
        [0.0, 40.0, 60.0, 100.0]
        >>> capacity.probabilities.round(4).tolist()
        [0.01, 0.09, 0.09, 0.81]
        >>> round(capacity.meets(50), 4), round(capacity.mean, 4)
        (0.9, 90.0)
        """
        from repyability.rbd import _network_capacity

        return _network_capacity.capacity_distribution(
            self, x, method, mc_samples, seed
        )

    def point_capacity(
        self,
        x: ArrayLike,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> CapacityDistribution:
        """The distribution of what the network can deliver at each time
        ``x``, the elements repaired and new at 0.

        It is
        [`capacity_distribution`][repyability.Network.capacity_distribution]
        at a time rather than in the long run: each element is up at
        ``x`` with its availability then (see
        [`point_availability`][repyability.Network.point_availability]),
        independently of the others, and the capacity's distribution
        follows from the maximum flow of each state that matters. The
        probability of a capacity above 0 is ``point_availability(x)``.

        Parameters
        ----------
        x : array_like
            Time/s, 0 or more: a number or an array.
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``: the shares of
            ``mc_samples`` states drawn from the elements' availabilities
            (the same uniforms at every time).
        mc_samples : int, optional
            The number of states drawn, by default 10_000.
        seed : int, optional
            Seeds the draws, by default None.

        Returns
        -------
        CapacityDistribution
            The levels and their probabilities at ``x``: one per level
            for a number, and one row per level and one column per time
            for an array.

        Raises
        ------
        ValueError
            If no link has a capacity or the elements are not repaired (a
            ``Refused``), a time is negative, or as for
            ``capacity_distribution``.
        NotImplementedError
            As for ``capacity_distribution``.

        Examples
        --------
        Two lines of 60 and 40 units, each failing at rate 0.01 and
        repaired at rate 0.5, so down at 10 with probability ``q = (1 /
        51) (1 - exp(-5.1))``:

        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = {
        ...     "reliability": surv.Exponential.from_params([0.01]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> net = Network(
        ...     {"a": ("s", "t", line), "b": ("s", "t", line)},
        ...     "s", "t",
        ...     capacity={"a": 60, "b": 40},
        ... )
        >>> capacity = net.point_capacity(10.0)
        >>> capacity.levels.tolist()
        [0.0, 40.0, 60.0, 100.0]
        >>> round(capacity.mean, 3)  # 100 (1 - q)
        98.051
        """
        from repyability.rbd import _network_capacity

        return _network_capacity.point_capacity(
            self, x, method, mc_samples, seed
        )

    def mission_capacity(
        self,
        t: ArrayLike,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> CapacityDistribution:
        """The distribution of what the network can deliver over ``[0,
        t]``, the elements repaired and new at 0: the expected fraction
        of the window spent at each capacity.

        It is the mean of
        [`point_capacity`][repyability.Network.point_capacity] over the
        window, integrated on pieces between the elements' curves' bends,
        halved until the quadrature agrees with its halves' (as
        [`mission_capacity`][repyability.RepairableRBD.mission_capacity]
        integrates a diagram's). Its ``delivered_fraction(demand)`` is
        the expected share of a demand delivered over the window, and as
        ``t`` grows it approaches ``capacity_distribution()``.

        Parameters
        ----------
        t : array_like
            The window/s' ends, 0 or more: a number or an array.
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``: each of
            ``mc_samples`` samples a time drawn evenly over the window
            and the elements' states then.
        mc_samples : int, optional
            The number of samples, by default 10_000.
        seed : int, optional
            Seeds the draws, by default None.

        Returns
        -------
        CapacityDistribution
            The levels and the expected fraction of each window at each:
            one per level for a number, and one row per level and one
            column per window for an array.

        Raises
        ------
        ValueError
            As for ``point_capacity``.
        NotImplementedError
            As for ``capacity_distribution``, or if the integral takes
            more pieces than the quadrature's limit.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import Network
        >>> line = {
        ...     "reliability": surv.Exponential.from_params([0.01]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> net = Network(
        ...     {"a": ("s", "t", line), "b": ("s", "t", line)},
        ...     "s", "t",
        ...     capacity={"a": 60, "b": 40},
        ... )
        >>> round(net.mission_capacity(100.0).mean, 3)
        98.078
        >>> round(net.capacity_distribution().mean, 3)  # 100 * 50 / 51
        98.039
        """
        from repyability.rbd import _network_capacity

        return _network_capacity.mission_capacity(
            self, t, method, mc_samples, seed
        )

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
        from repyability.rbd.serialisation import (
            _serialise_component,
            serialise_model,
        )

        def saved(model) -> dict:
            if isinstance(model, dict):
                return {"component": _serialise_component(model)}
            return {"model": serialise_model(model)}

        capacity = (
            {
                "capacity": [
                    {"link": link, "capacity": value}
                    for link, value in self.capacity.items()
                ]
            }
            if self.capacity
            else {}
        )
        return {
            "repyability_version": __version__,
            "type": "Network",
            "source": self.source,
            "target": self.target,
            "terminals": (
                "all" if self._all_terminals else list(self.terminals)
            ),
            "directed": self.directed,
            "links": [
                {
                    "link": name,
                    "nodes": [u, v],
                    **saved(self.models[name]),
                }
                for name, (u, v) in self.links.items()
            ],
            "nodes": [
                {"node": name, **saved(model)}
                for name, model in self.nodes.items()
            ],
            **capacity,
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
            _deserialise_component,
            _node_name,
            deserialise_model,
        )

        def loaded(e: dict):
            if "component" in e:
                return _deserialise_component(e["component"])
            return deserialise_model(e["model"])

        if d.get("type") != "Network":
            raise ValueError(f"Not a network: type {d.get('type')!r}.")
        links = {
            _node_name(e["link"]): (
                _node_name(e["nodes"][0]),
                _node_name(e["nodes"][1]),
                loaded(e),
            )
            for e in d["links"]
        }
        nodes = {
            _node_name(e["node"]): loaded(e) for e in d.get("nodes") or ()
        }
        directed = bool(d.get("directed", False))
        capacity = {
            _node_name(e["link"]): float(e["capacity"])
            for e in d.get("capacity") or ()
        } or None
        # A save from before a network took more than two terminals has
        # only its source and target.
        terminals = d.get("terminals")
        if terminals is None:
            return cls(
                links,
                _node_name(d["source"]),
                _node_name(d["target"]),
                nodes=nodes or None,
                directed=directed,
                capacity=capacity,
            )
        if terminals != "all":
            terminals = [_node_name(node) for node in terminals]
        return cls(
            links,
            nodes=nodes or None,
            terminals=terminals,
            directed=directed,
            capacity=capacity,
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
    # As a diagram
    # ------------------------------------------------------------------

    def to_rbd(self) -> NonRepairableRBD:
        """The reliability block diagram of a directed network: the same
        graph, each link and each node that can fail a node of the diagram
        with its model, and each other node a junction
        ([`PerfectReliability`][repyability.PerfectReliability]).

        The diagram has the links that lie on a way from the source to the
        target; the others cannot join them, and are left out. A diagram
        has no cycle, so a network with one among those links is not
        drawn: its own methods work it out.

        Returns
        -------
        NonRepairableRBD
            The diagram, from input node ``"input"`` to output node
            ``"output"`` (renamed if a link or node has that name). A
            node that cannot fail keeps its name, unless a link has it.

        Raises
        ------
        ValueError
            If the network is not directed (a link that works both ways
            is not an edge of a diagram), or no way leads from the source
            to the target.
        NotImplementedError
            If the links between the source and the target have a cycle.

        Examples
        --------
        >>> from repyability import Network
        >>> net = Network(
        ...     {"a": ("s", "x", 0.1), "b": ("x", "t", 0.1),
        ...      "c": ("s", "t", 0.2)},
        ...     "s", "t", directed=True,
        ... )
        >>> rbd = net.to_rbd()
        >>> round(rbd.sf(), 4), round(net.sf(), 4)
        (0.962, 0.962)
        """
        from repyability.rbd.helper_classes import PerfectReliability

        if not self.directed:
            raise Refused(
                "to_rbd() draws a directed network (directed=True): a "
                "link that works both ways is not an edge of a reliability "
                "block diagram, which is followed one way."
            )
        # The links on a way from the source to the target: none into the
        # source or out of the target, each from a node the source reaches
        # to one that reaches the target.
        arcs = {
            name: (u, v)
            for name, (u, v) in self.links.items()
            if v != self.source and u != self.target
        }

        def reached(start: Hashable, forward: bool) -> Set[Hashable]:
            beside: Dict[Hashable, List[Hashable]] = {}
            for u, v in arcs.values():
                a, b = (u, v) if forward else (v, u)
                beside.setdefault(a, []).append(b)
            seen, stack = {start}, [start]
            while stack:
                for other in beside.get(stack.pop(), ()):
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            return seen

        after, before = reached(self.source, True), reached(self.target, False)
        arcs = {
            name: (u, v)
            for name, (u, v) in arcs.items()
            if u in after and v in before
        }
        if not arcs:
            raise Refused(
                f"No way leads from the source {self.source!r} to the "
                f"target {self.target!r} along the links: there is no "
                "diagram to draw (the network never works)."
            )
        # A cycle: what is left once the nodes with no link into them
        # are taken away, one after another.
        into: Dict[Hashable, int] = {}
        for u, v in arcs.values():
            into.setdefault(u, 0)
            into[v] = into.get(v, 0) + 1
        free = [node for node, count in into.items() if not count]
        while free:
            node = free.pop()
            del into[node]
            for u, v in arcs.values():
                if u == node:
                    into[v] -= 1
                    if not into[v]:
                        free.append(v)
        if into:
            raise NotImplementedError(
                "The links between the source and the target have a cycle "
                f"(through {', '.join(sorted(map(repr, into)))}), and a "
                "reliability block diagram has none. The network's own "
                "methods work it out (sf, the importance measures, "
                "random)."
            )
        taken = set(self.models)
        drawn: Dict[Hashable, Hashable] = {}
        models: Dict[Hashable, Any] = {}
        for pair in arcs.values():
            for node in pair:
                if node in drawn:
                    continue
                if node in self.nodes:
                    drawn[node] = node
                    models[node] = self.nodes[node]
                    continue
                name = node
                while name in taken:
                    name = ("node", name)
                taken.add(name)
                drawn[node] = name
                models[name] = PerfectReliability
        ends = []
        for name in ("input", "output"):
            while name in taken:
                name += " (network)"
            taken.add(name)
            ends.append(name)
        edges = [(ends[0], drawn[self.source]), (drawn[self.target], ends[1])]
        for name, (u, v) in arcs.items():
            edges += [(drawn[u], name), (name, drawn[v])]
            models[name] = self.models[name]
        return NonRepairableRBD(
            edges,
            models,
            input_node=ends[0],
            output_node=ends[1],
            capacity={
                link: value
                for link, value in self.capacity.items()
                if link in arcs
            }
            or None,
        )

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
        but the simulation (and two terminals' paths, listed), with the
        message the methods raise, as a directed network of more than
        ``MAX_PATHS`` paths does; an element whose model has no
        reliability of its own makes the analyses built on it refuse, as
        it does. The routes are the same for two terminals, several and
        all of them.

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

        if self._repaired is not None:
            from repyability.rbd._event_curves import down_routes

            elements = down_routes(self._repaired)
            effect = "probabilities of being down"
        else:
            elements = {
                name: r.model_route(m) for name, m in self.models.items()
            }
            effect = "reliabilities"
        out: Dict[str, AnalysisRoute] = {}

        def give(names, route) -> None:
            for name in names:
                out[name] = route

        def built(route, reason):
            return r.with_nodes(route, reason, elements, effect)

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
        give(
            ("cut_sets", "minimal_cut_sets"),
            AnalysisRoute(r.EXACT, "Read off the decision diagram."),
        )
        paths = r.refusal(self.path_sets)
        give(
            ("path_sets", "minimal_path_sets"),
            (
                r.refused(paths)
                if paths
                else AnalysisRoute(
                    r.EXACT,
                    (
                        "Every simple path between the terminals, listed."
                        if len(self.terminals) == 2
                        else "Every smallest set of links and nodes that "
                        "joins the terminals, read off the decision "
                        "diagram."
                    ),
                )
            ),
        )
        out["Hf"] = built(
            r.EXACT,
            "The exact reliability and unreliability, as -log of one of "
            "them.",
        )
        give(
            ("df", "hf"),
            built(
                r.NUMERICAL,
                "The exact reliability, differentiated by central "
                "differences.",
            ),
        )
        out["parameter_sensitivity"] = built(
            r.NUMERICAL,
            "The exact Birnbaum importance times each parameter's "
            "derivative, by differences.",
        )
        out["tornado"] = built(
            r.EXACT,
            "The exact reliability with each parameter at its low and at "
            "its high, the network rebuilt.",
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
            (
                "Monte-Carlo lifetimes: each element's drawn, and the "
                "longest-lasting path's taken."
                if len(self.terminals) == 2
                else "Monte-Carlo lifetimes: each element's drawn, and the "
                "life taken of the link that joins the terminals, the "
                "links added longest-lived first."
            ),
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
        for name in ("time_to_reliability_uncertainty", "bx_life_uncertainty"):
            out[name] = (
                r.refused(fixed)
                if fixed
                else built(
                    r.SIMULATED,
                    "The elements' parameters drawn from their uncertainty, "
                    "and the time found for each draw by inverting its "
                    "exact reliability.",
                )
            )
        # Every exact value needs the decision diagram: without it, only
        # the simulation (and the paths, listed on their own) is left.
        too_large = r.refusal(self._decomposition)
        if too_large:
            for name in out:
                if name not in ("random", "path_sets", "minimal_path_sets"):
                    out[name] = r.refused(too_large)
        # (A repaired network's lifetime analyses are refused below.)
        replay = (
            None if self.is_repairable else r.refusal(self._require_replayable)
        )
        if lifetimes:
            out["compare"] = r.refused(lifetimes)
        elif out["mean"].route != r.REFUSED:
            out["compare"] = dataclasses.replace(
                out["mean"],
                reason="The difference of the two networks' mean lifetimes "
                "(see mean), where the other's is worked out too; otherwise, "
                "or with method='simulate', simulated lifetimes of both, "
                "with common random numbers.",
            )
        else:
            out["compare"] = (
                r.refused(replay)
                if replay
                else AnalysisRoute(
                    r.SIMULATED,
                    "Simulated lifetimes of both networks, a link or node "
                    "of the same name drawing the same random numbers in "
                    "both (common random numbers).",
                )
            )
        self._repaired_routes(out, built, too_large)
        self._capacity_routes(out, built)
        return with_importances(out)

    def _repaired_routes(self, out: dict, built, too_large) -> None:
        """The routes of the analyses of repaired elements, in ``out``:
        refused for a network without them; for a network of them, how
        each is worked out, and what the network refuses (in place of the
        routes of a network of lifetimes)."""
        from repyability.rbd import routes as r

        for name in _REPAIRED_OWN | set(out):
            reason = r.refusal(functools.partial(self._refuse_repaired, name))
            if reason:
                out[name] = r.refused(reason)
        if self._repaired is None:
            return
        if too_large:
            for name in _REPAIRED_OWN:
                out[name] = r.refused(too_large)
            return
        from repyability.rbd._event_curves import intensity_routes

        out["point_availability"] = built(
            r.EXACT,
            "The network's decision diagram over the elements' "
            "availabilities at x, each one less its probability of being "
            "down.",
        )
        out["point_unavailability"] = built(
            r.EXACT,
            "The network's decision diagram over the elements' "
            "probabilities of being down at x, each worked out in its own "
            "right.",
        )
        out["mission_availability"] = built(
            r.NUMERICAL,
            "One less the mission unavailability, which is integrated "
            "over the window to its own size.",
        )
        out["mission_unavailability"] = built(
            r.NUMERICAL,
            "The point unavailability integrated over the window, on "
            "pieces between the elements' curves' bends, to its own size.",
        )
        no_long_run = r.refusal(
            functools.partial(self._refuse_unrepaired, "mean_availability")
        )
        out["mean_availability"] = AnalysisRoute(
            r.EXACT,
            "The network's decision diagram over the elements' long-run "
            "availabilities, MTTF / (MTTF + MTTR).",
        )
        out["mean_unavailability"] = AnalysisRoute(
            r.EXACT,
            "The network's decision diagram over the elements' long-run "
            "unavailabilities, MTTR / (MTTF + MTTR), as a sum of products "
            "of its own.",
        )
        if no_long_run:
            for name in ("mean_availability", "mean_unavailability"):
                out[name] = r.refused(no_long_run)
        for name, route, reason in (
            (
                "failure_frequency",
                r.EXACT,
                "Each element's failure intensity times its Birnbaum "
                "importance from the network's decision diagram, summed.",
            ),
            (
                "expected_failures",
                r.NUMERICAL,
                "Each element's expected failures weighed by its Birnbaum "
                "importance over time, integrated on pieces between the "
                "elements' curves' bends (to about 1e-7), and extended "
                "exactly once they have settled.",
            ),
            (
                "system_failure_frequency",
                r.EXACT,
                "Each element's long-run failure frequency times its "
                "Birnbaum importance at the long-run values, summed.",
            ),
        ):
            out[name] = (
                r.refused(no_long_run)
                if no_long_run and name == "system_failure_frequency"
                else r.with_nodes(
                    route,
                    reason,
                    intensity_routes(self._repaired, name),
                    "failure intensities",
                )
            )

    def _capacity_routes(self, out: dict, built) -> None:
        """The routes of the capacity's analyses, in ``out``: refused
        without capacities, over time without repaired elements, and
        where the states that matter are too many; otherwise exact from
        the elements' probabilities (over a window, integrated)."""
        from repyability.rbd import _network_capacity
        from repyability.rbd import routes as r

        how = {
            "capacity_distribution": (
                AnalysisRoute(
                    r.EXACT,
                    "Every state of the links and nodes that changes the "
                    "maximum flow, at the elements' long-run "
                    "availabilities, MTTF / (MTTF + MTTR).",
                )
                if self.is_repairable
                else built(
                    r.EXACT,
                    "Every state of the links and nodes that changes the "
                    "maximum flow, each with the product of its "
                    "elements' probabilities at x (method='simulate' "
                    "draws the states instead).",
                )
            ),
            "point_capacity": built(
                r.EXACT,
                "Every state of the links and nodes that changes the "
                "maximum flow, at the elements' availabilities at x.",
            ),
            "mission_capacity": built(
                r.NUMERICAL,
                "The capacity's distribution at a time, integrated over "
                "the window on pieces between the elements' curves' "
                "bends.",
            ),
        }
        for name, route in how.items():
            reason = r.refusal(
                functools.partial(
                    _network_capacity.require_capacity, self, name
                )
            ) or r.refusal(lambda: _network_capacity.tree(self))
            out[name] = r.refused(reason) if reason else route

    def __repr__(self) -> str:
        unit = "" if self.unit is None else f", unit {self.unit!r}"
        if self._all_terminals:
            joined = "all terminals"
        elif len(self.terminals) == 2:
            joined = f"{self.source!r} to {self.target!r}"
        else:
            joined = f"{len(self.terminals)} terminals"
        links = "directed links" if self.directed else "links"
        return (
            f"Network({len(self.links)} {links}, {len(self.nodes)} failing "
            f"nodes, {joined}{unit})"
        )


def _refusing_repaired(name: str, method):
    """``method`` (the analysis ``name``), refusing a network of repaired
    elements first (see ``Network._refuse_repaired``)."""

    @functools.wraps(method)
    def wrap(self, *args, **kwargs):
        self._refuse_repaired(name)
        return method(self, *args, **kwargs)

    return wrap


# Every public method but those a network of repaired elements gives
# refuses one, a method added later too.
for _name, _method in list(vars(Network).items()):
    if (
        isinstance(_method, types.FunctionType)
        and not _name.startswith("_")
        and _name not in _REPAIRED_GIVES
    ):
        setattr(Network, _name, _refusing_repaired(_name, _method))
del _name, _method


def _link_order(
    links: Dict[Hashable, Tuple[Hashable, Hashable]],
    adjacent: Dict[Hashable, List[Tuple[Hashable, Hashable]]],
    terminals: Sequence[Hashable],
) -> List[Hashable]:
    """The order the links of the terminals' part of the network are
    decided in: by the breadth-first order of their later end, from a
    terminal (each of the first ``_ORDER_ROOTS`` is tried), whichever
    keeps the frontiers narrower (the smaller sum over the links of two to
    the frontier's size). A link out of the first terminal's part cannot
    join the terminals, and is left out."""
    candidates = []
    for root in terminals[:_ORDER_ROOTS]:
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


class _Joined:
    """A network's connection as the structure of a diagram of its
    elements (what ``rbd._event_curves.Structured`` evaluates in place of
    a diagram's ``modular.Decomposition``): the probabilities that the
    terminals are joined and parted, and their gradient, from the
    network's plan, which is built when first asked for."""

    #: No reduced diagram (the capacity analysis's).
    flow = None

    def __init__(self, network: "Network"):
        self._network = network

    @property
    def plan(self) -> "_Plan":
        return self._network._decomposition()

    @property
    def nodes(self) -> frozenset:
        """The elements the connection depends on."""
        return frozenset(self.plan.names)

    @property
    def always_works(self) -> bool:
        return self.plan.root == _WORK

    @staticmethod
    def _shape(shape) -> tuple:
        if shape is None:
            return ()
        return (shape,) if isinstance(shape, (int, np.integer)) else shape

    def probabilities(
        self,
        p: dict,
        q: Optional[dict] = None,
        shape=None,
        works: bool = True,
        fails: bool = True,
    ) -> tuple:
        """The probabilities that the terminals are joined and that they
        are parted (None for the one not asked for), from each element's
        probability of working ``p`` and of failing ``q`` (by default ``1
        - p``): each a sum of products of its own."""
        plan, size = self.plan, self._shape(shape)
        if q is None:
            q = {name: 1.0 - np.asarray(p[name], float) for name in plan.names}
        joined = _evaluate(plan, p, q, size, _JOINED) if works else None
        parted = _evaluate(plan, p, q, size, _PARTED) if fails else None
        if shape is None:
            return (
                None if joined is None else float(joined),
                None if parted is None else float(parted),
            )
        return joined, parted

    def value_and_gradient(self, p: dict, q: dict, shape=None) -> tuple:
        """The probabilities that the terminals are joined and parted,
        and the derivative of the first with respect to each element's
        probability of working (its Birnbaum importance), taken from
        whichever of the two is the smaller, so that it is not a
        difference of values near 1 (see ``Network._birnbaum``)."""
        plan, size = self.plan, self._shape(shape)
        joined, up = _value_and_gradient(plan, p, q, size, _JOINED)
        parted, down = _value_and_gradient(plan, p, q, size, _PARTED)
        from_parted = np.asarray(parted) <= np.asarray(joined)
        gradient = {
            name: np.where(from_parted, -down[name], up[name]) + 0.0
            for name in up
        }
        if shape is None:
            return (
                float(joined),
                float(parted),
                {name: float(value) for name, value in gradient.items()},
            )
        return joined, parted, gradient

    def works(self, status: Mapping, method: str = "p") -> bool:
        """Whether the terminals are joined, given whether each element
        works (truthy) or has failed (falsy)."""
        plan = self.plan
        p = {name: 1.0 if status[name] else 0.0 for name in plan.names}
        q = {name: 1.0 - value for name, value in p.items()}
        return bool(_evaluate(plan, p, q, (), _JOINED) > 0.5)


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


def _rows(plan: _Plan, values: dict, shape: tuple) -> np.ndarray:
    """Each of the plan's variables' ``values`` (of ``shape``, or as many
    in a row, or one for all), as the rows of one array, a column per
    evaluation."""
    size = int(np.prod(shape, dtype=np.int64))
    out = np.empty((len(plan.names), size))
    for i, name in enumerate(plan.names):
        value = np.asarray(values[name], float)
        out[i] = (
            value.reshape(-1)
            if value.size == size
            else np.broadcast_to(value, shape).ravel()
        )
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
    p_rows, q_rows = _rows(plan, p, shape), _rows(plan, q, shape)
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
    p_rows, q_rows = _rows(plan, p, shape), _rows(plan, q, shape)
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


def _canonical(labels: np.ndarray) -> np.ndarray:
    """Frontier states (a row of labels each, -1 for a failed node) with
    their labels renumbered in order of first appearance, so that equal
    states are equal rows."""
    count, width = labels.shape
    if not count or not width:
        return labels
    size = int(labels.max()) + 1
    if size <= 0:
        return labels
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
    return np.where(
        labels >= 0, rank[rows[:, None], np.maximum(labels, 0)], labels
    ).astype(labels.dtype)


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
    terminals: Collection[Hashable],
) -> _Plan:
    """The decision diagram of whether the ``terminals`` (two or more
    nodes) are all joined, its links decided in ``order`` (see the module
    docstring), as a ``_Plan``.

    The states before each decision are worked out together, as rows of
    an array (#173): each is the frontier's parts (a label for each
    frontier node, in the order they entered: the same label for nodes
    joined by working links, -1 for a failed node) and whether each part
    holds a terminal, kept as one number for each frontier node (twice
    its label, plus one if its part holds a terminal). A part that holds
    a terminal and leaves the frontier parts the terminals, unless it
    holds them all; the terminals are joined once every one has entered
    and one part holds them. A node that can fail is decided before its
    first link. Each state's two outcomes are found at once, equal ones
    merged, and the diagram is then reduced from the last decision up (a
    state whose outcomes agree is dropped, and equal ones shared), which
    gives the same diagram as deciding the states one by one.

    Raises
    ------
    NotImplementedError
        If the diagram would have more than ``MAX_STATES`` states.
    """
    terminals = set(terminals)
    first: Dict[Hashable, int] = {}
    last: Dict[Hashable, int] = {}
    for i, name in enumerate(order):
        for w in links[name]:
            first.setdefault(w, i)
            last[w] = i
    if not terminals <= first.keys():
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
    # frontier, the positions that stay after its leaving ones go and
    # those that go; and whether every terminal has entered by then.
    shapes: List[Optional[tuple]] = []
    complete: List[bool] = []
    front: List[Hashable] = []
    outside = len(terminals)
    for name, entering, ends, leaving in decisions:
        if ends is None:
            shapes.append(None)
            front.append(name)
            outside -= name in terminals
            complete.append(not outside)
            continue
        widened = front + list(entering)
        outside -= sum(w in terminals for w in entering)
        complete.append(not outside)
        keep = [j for j, w in enumerate(widened) if w not in leaving]
        gone = [j for j, w in enumerate(widened) if w in leaving]
        shapes.append(
            (widened.index(ends[0]), widened.index(ends[1]), keep, gone)
        )
        front = [w for w in widened if w not in leaving]

    # The states before the next decision; each decision's outcomes, as
    # the index of the state after (or _NONE_FAIL, _NONE_WORK).
    state = np.zeros((1, 0), dtype=np.int16)
    outcomes: List[Tuple[np.ndarray, np.ndarray]] = []
    states = 1
    for k, (name, entering, ends, leaving) in enumerate(decisions):
        count = len(state)
        labels = state >> 1  # -1 stays -1
        held = (state >= 0) & ((state & 1) == 1)
        parts = (labels.max(axis=1, initial=-1) + 1).astype(np.int16)

        def column(value) -> np.ndarray:
            return np.full((count, 1), value)

        # Each outcome: the frontier's labels and whether each node's part
        # holds a terminal, the states whose terminals it joins, and the
        # positions that leave (None: the terminals are parted).
        sides: List[Optional[tuple]]
        if ends is None:
            # A node that can fail, entering the frontier.
            is_terminal = name in terminals
            sides = [
                (
                    np.hstack([labels, parts[:, None]]),
                    np.hstack([held, column(is_terminal)]),
                    None,
                    [],
                ),
                (
                    None
                    if is_terminal
                    else (
                        np.hstack([labels, column(np.int16(-1))]),
                        np.hstack([held, column(False)]),
                        None,
                        [],
                    )
                ),
            ]
            keep = list(range(labels.shape[1] + 1))
        else:
            iu, iv, keep, gone = shapes[k]  # type: ignore[misc]
            grown, holds = labels, held
            for j, w in enumerate(entering):
                new = (parts + j).astype(np.int16)
                grown = np.hstack([grown, new[:, None]])
                holds = np.hstack([holds, column(w in terminals)])
            a, b = grown[:, iu], grown[:, iv]
            merge = (a >= 0) & (b >= 0) & (a != b)
            of_b = merge[:, None] & (grown == b[:, None])
            of_either = of_b | (merge[:, None] & (grown == a[:, None]))
            joined = np.where(of_b, a[:, None], grown)
            together = np.where(
                of_either, (holds[:, iu] | holds[:, iv])[:, None], holds
            )
            work = None
            if complete[k]:
                # One part holds every terminal: the labels that hold one
                # are all the same.
                top = np.iinfo(np.int16).max
                low = np.where(together, joined, top).min(axis=1, initial=top)
                work = low == np.where(together, joined, -1).max(
                    axis=1, initial=-1
                )
            sides = [
                (joined, together, work, gone),
                (grown, holds, None, gone),
            ]
        codes: List[np.ndarray] = []
        kept: List[Optional[tuple]] = []
        for side in sides:
            code = np.full(count, _NONE_FAIL, dtype=np.int32)
            if side is None:
                codes.append(code)
                kept.append(None)
                continue
            rows, flags, work, gone = side
            staying = rows[:, keep]
            # A part that holds a terminal closes: none of its nodes stay.
            closed = np.zeros(count, dtype=bool)
            for j in gone:
                closed |= flags[:, j] & ~(staying == rows[:, j : j + 1]).any(
                    axis=1
                )
            if work is not None:
                code[work] = _NONE_WORK
                alive = ~closed & ~work
            else:
                alive = ~closed
            codes.append(code)
            renamed = _canonical(staying[alive])
            kept.append(
                (
                    alive,
                    np.where(
                        renamed >= 0,
                        2 * renamed + flags[:, keep][alive],
                        -1,
                    ).astype(np.int16),
                )
            )
        unique, inverse = _unique_rows(
            np.vstack([rows for _, rows in filter(None, kept)])
        )
        offset = 0
        for code, after in zip(codes, kept):
            if after is None:
                continue
            alive = after[0]
            number = int(alive.sum())
            code[alive] = inverse[offset : offset + number]
            offset += number
        outcomes.append((codes[0], codes[1]))
        states += len(unique)
        if states > MAX_STATES:
            raise _TooLarge(
                "The network's decision diagram has more than "
                f"{MAX_STATES:,} states (ways its frontier can be joined "
                "up), too many to work out exactly.",
                "MAX_STATES",
                "Simulate it: method='simulate'.",
            )
        state = unique

    # Reduce from the last decision up: each state's slot, the states after
    # the last decision all parted.
    names = [name for name, _, _, _ in decisions]
    slot = np.full(len(state), _FAIL, dtype=np.int64)
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
