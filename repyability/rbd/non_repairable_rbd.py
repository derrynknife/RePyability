"""Reliability block diagrams of non-repairable components.

Defines ``NonRepairableRBD``, which computes the reliability of a system of
non-repairable components from its block diagram, and the small helpers it
uses (``check_x`` and the ``NodeFailure`` simulation event).
"""

import functools
import math
import warnings
import zlib
from copy import copy
from dataclasses import dataclass, field
from queue import PriorityQueue
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from surpyval import NonParametric

from repyability.utils.deprecation import (
    REMOVAL,
    ignored,
    nonparametric_nodes,
    warn_nonparametric,
)
from repyability.utils.wrappers import conditional_survival, numpy_seed

from . import _montecarlo as montecarlo
from . import capacity as _capacity
from . import redundancy_allocation
from ._mean_lifetime import mean_lifetime, model_knots
from ._model_utils import is_fixed_probability, model_mean, parametric_spec
from ._sampling import RowSampler, inverse_sampler, row_sampler
from .ccf import CCFGroup
from .degrading_node import DegradingNode
from .helper_classes import PerfectReliability, PerfectUnreliability
from .load_sharing_node import LoadSharingModel
from .node_state import NodeState
from .rbd import RBD, _check_on_infeasible_rbd, leaves_out_junctions
from .redundancy_allocation import ComponentOption, active_unreliability
from .repeated_node import RepeatedNode
from .repeated_standby_node import RepeatedStandbyNode
from .results import (
    CapacityDistribution,
    ConfidenceInterval,
    RedundancyAllocation,
    ReliabilityRedundancyAllocation,
    UncertaintyResult,
)
from .routes import AnalysisRoute
from .standby_node import StandbyModel
from .uncertainty import draw_models

# Event class for simulation
#: Lifetimes per block of a parallel draw (see ``NonRepairableRBD.random``):
#: each block is seeded by its position, so a parallel draw does not depend
#: on the number of processes.
RANDOM_BLOCK = 10_000
#: About how many uniforms a vectorised draw works on at once: a large draw
#: takes its uniforms in rows of this many, from the same stream, so its
#: lifetimes are the same as in one piece, but its arrays stay in cache.
DRAW_UNIFORMS = 2**20


def _random_block(task) -> np.ndarray:
    """One block of a parallel draw (in its own process)."""
    rbd, size, seed, antithetic = task
    with numpy_seed(seed):
        return rbd._draw(size, antithetic)


def _row_blocks(size: int, width: int):
    """``(first row, rows)`` blocks of ``size`` rows of ``width`` uniforms,
    about ``DRAW_UNIFORMS`` a block."""
    rows = max(1, DRAW_UNIFORMS // max(width, 1))
    for first in range(0, size, rows):
        yield first, min(rows, size - first)


def _check_lifetimes(lifetimes: np.ndarray) -> None:
    if np.isnan(lifetimes).any():
        raise ValueError(
            "A node drew a NaN lifetime, which antithetic and common random "
            "number draws cannot order; draw without them."
        )


@dataclass(order=True)
class NodeFailure:
    """A node failure scheduled in the event-driven lifetime simulation.

    Instances compare by ``time`` only (``node`` is excluded from
    comparisons), so a ``queue.PriorityQueue`` of them yields the failures
    in time order. Used by the one-sample-at-a-time path of
    [`NonRepairableRBD.random`][repyability.NonRepairableRBD.random].

    Parameters
    ----------
    time : float
        The time at which the node fails.
    node : Hashable
        The name of the node that fails.
    """

    time: float
    node: Hashable = field(compare=False)


def check_x(func):
    """Decorate an RBD method so it accepts any time input ``x``.

    The wrapper normalises ``x`` and shapes the result to match it. The
    wrapped method always receives ``x`` as a float numpy array of at least
    one dimension. The caller-facing contract is numpy-style: a scalar ``x``
    returns a float (or a dict of floats, for the per-node and importance
    methods), and an array ``x`` returns the method's array (or dict of
    arrays) unchanged.

    ``x=None`` is allowed only for a fixed-probability RBD, where time is
    irrelevant and ``x = 1.0`` is used; for a time-varying RBD the wrapper
    raises a ValueError rather than failing cryptically downstream.

    Parameters
    ----------
    func : callable
        A method with signature ``func(self, x, *args, **kwargs)`` whose
        ``self`` has an ``is_fixed`` attribute.

    Returns
    -------
    callable
        The wrapped method, with signature
        ``wrap(self, x=None, *args, **kwargs)``.
    """

    @functools.wraps(func)
    def wrap(obj, x=None, *args, **kwargs):
        x, scalar_in = _times(obj, x)
        result = func(obj, x, *args, **kwargs)
        if scalar_in:
            if isinstance(result, dict):
                return {k: np.asarray(v).item() for k, v in result.items()}
            return np.asarray(result).item()
        return result

    return wrap


def _times(obj, x) -> tuple:
    """The time argument of an RBD method as a float array of at least one
    dimension, and whether it was a scalar. ``x=None`` stands for any time
    (1.0) in a fixed-probability RBD, and is refused in a time-varying
    one."""
    if x is None:
        if obj.is_fixed:
            x = 1.0
        else:
            raise ValueError(
                "x is required: this RBD is time-varying (at least one "
                "node model's probability depends on time)."
            )
    scalar_in = np.ndim(x) == 0
    return np.atleast_1d(np.asarray(x, dtype=float)), scalar_in


def _dsf_dparam(cls, params, j, x_arr, rel_step, extras=None) -> np.ndarray:
    """Partial derivative of a distribution's ``sf`` at ``x_arr`` with respect
    to its ``j``-th parameter, by finite difference.

    ``cls.from_params`` rebuilds the distribution with a perturbed parameter
    (and the model's offset, limited-failure-population and zero-inflation
    ``extras``, kept as they are), so this works for any surpyval parametric
    distribution without hard-coding per-distribution derivative formulae.
    A central difference is used where both perturbations are valid; if one
    perturbation falls outside a parameter's admissible range (e.g. a
    probability leaving ``[0, 1]``) it falls back to a one-sided difference
    about the unperturbed value.
    """
    theta = params[j]
    h = rel_step * abs(theta) if theta != 0.0 else rel_step

    def perturbed(delta):
        trial = list(params)
        trial[j] = theta + delta
        try:
            model = cls.from_params(trial, **(extras or {}))
            return np.asarray(model.sf(x_arr), dtype=float)
        except Exception:
            return None

    up = perturbed(h)
    down = perturbed(-h)
    if up is not None and down is not None:
        return (up - down) / (2.0 * h)
    # A perturbation hit a parameter bound; fall back to a one-sided
    # difference about the unperturbed value.
    base = np.asarray(cls.from_params(list(params)).sf(x_arr), dtype=float)
    if up is not None:
        return (up - base) / h
    if down is not None:
        return (base - down) / h
    return np.full(x_arr.shape, np.nan)


def _check_model(node, model) -> None:
    """Raise if a node's model is not one (it has no ``sf``), saying what
    to give instead; a number most likely means a probability."""
    if callable(getattr(model, "sf", None)):
        return
    if isinstance(model, (int, float)) and not isinstance(model, bool):
        value = float(model)
        hint = ""
        if 0.0 <= value <= 1.0:
            hint = (
                f" (if {value:g} is its reliability, "
                f"FixedEventProbability.from_params({1.0 - value:g}))"
            )
        raise TypeError(
            f"The model of node {node!r} is the number {model!r}, not a "
            "model with a survival function (sf). For a node that fails "
            "with a fixed probability q, give "
            f"surpyval.FixedEventProbability.from_params(q), q being its "
            f"probability of failing{hint}; for one with a lifetime, a "
            "surpyval distribution such as "
            "surpyval.Weibull.from_params([alpha, beta])."
        )
    raise TypeError(
        f"The model of node {node!r} is {model!r}, which has no survival "
        "function (sf): give a surpyval model (e.g. "
        "surpyval.Weibull.from_params([alpha, beta]), or "
        "surpyval.FixedEventProbability.from_params(q) for a fixed "
        "probability of failing), a node class of RePyability's, or another "
        "NonRepairableRBD."
    )


class NonRepairableRBD(RBD):
    """A reliability block diagram (RBD) of non-repairable components.

    The diagram is a directed acyclic graph with exactly one input node (no
    predecessors) and one output node (no successors). Every other node is
    a component with a reliability model, and the system works while a
    path of working components joins the input to the output (``k`` makes
    a node k-out-of-n). Components fail independently, except within the
    common-cause groups given in ``ccf_groups``.

    The system reliability [`sf`][repyability.NonRepairableRBD.sf] is
    computed exactly from the node reliabilities, using the minimal path
    sets (or cut sets); the other analytic methods build on it. The mean
    time to failure and the lifetimes drawn by
    [`random`][repyability.NonRepairableRBD.random] are Monte-Carlo
    estimates.

    A node model can be:

    - a surpyval distribution: parametric (e.g.
      ``surpyval.Weibull.from_params([100, 2])`` or a fitted model),
      non-parametric (e.g. a ``surpyval.KaplanMeier`` fit) or a fixed
      per-demand probability (``surpyval.FixedEventProbability``);
    - a composite node: a [`StandbyModel`][repyability.StandbyModel],
      [`LoadSharingModel`][repyability.LoadSharingModel],
      [`RepeatedNode`][repyability.RepeatedNode],
      [`RepeatedStandbyNode`][repyability.RepeatedStandbyNode],
      [`RegressionNode`][repyability.RegressionNode] or another
      ``NonRepairableRBD`` nested as a single node;
    - [`PerfectReliability`][repyability.PerfectReliability] or
      [`PerfectUnreliability`][repyability.PerfectUnreliability].

    If every node is a fixed probability the RBD is fixed-probability (see
    [`is_fixed`][repyability.NonRepairableRBD.is_fixed]) and the time
    argument of its methods may be omitted.

    Parameters
    ----------
    edges : Iterable[tuple[Hashable, Hashable]]
        The directed edges ``(from_node, to_node)`` of the diagram, e.g.
        ``[("in", "a"), ("a", "out")]``.
    reliabilities : dict
        ``{node: model}`` for every component node (see above). A value
        that is the name of another node in this dict makes the key a
        *repeated* node: the same physical component drawn a second time.
        It stays where it is drawn, and every appearance is the one
        component: it works, or has failed, in all of them at once. That
        node cannot itself be a repeat. The input and output nodes need no
        entry: they are always perfectly reliable, and a model given for
        them is replaced.
    k : dict[Any, int], optional
        ``{node: k}`` for k-out-of-n nodes: a node with ``n`` predecessors
        is reached only when at least ``k`` of them are reached through
        working components (and it must work itself). By default ``k`` is
        1 for every node. For example, three parallel units feeding the
        output node ``"out"`` with ``k={"out": 2}`` form a 2-out-of-3
        system.
    input_node : Hashable, optional
        The input node. By default it is inferred as the unique node with
        no predecessors; naming it does not relax that requirement. A name
        not in the diagram, or a node with predecessors, raises a
        ValueError.
    output_node : Hashable, optional
        The output node. By default it is inferred as the unique node with
        no successors; naming it does not relax that requirement. A name
        not in the diagram, or a node with successors, raises a
        ValueError.
    on_infeasible_rbd : str, optional
        What to do if the diagram is invalid: ``"raise"`` (the default)
        raises a ValueError, while ``"warn"`` and ``"ignore"`` carry on,
        after emitting a UserWarning that lists the problems or silently.
        The results of an invalid RBD are not meaningful. Invalid diagrams
        include a cycle, more than one input or output node, a node
        without a model, and a ``k`` that is zero, exceeds the node's
        number of incoming branches or names an unknown node. The findings
        are kept in ``structure_check``.
    ccf_groups : Iterable[CCFGroup], optional
        Common-cause failure groups ([`CCFGroup`][repyability.CCFGroup]),
        by default none. Each member must be a component node (not the
        input or output node, nor a repeated node) in at most one group,
        and a group's members must have identical models (compared by
        their serialised form; the check is skipped for models that cannot
        be serialised). The groups are honoured by ``sf``/``ff`` and the
        methods computed from them (``reliability``, ``unreliability``,
        ``df``, ``hf``, ``Hf``, ``cs``, ``time_to_reliability``,
        ``bx_life``). ``random``, ``mean`` and the MTTF methods ignore
        them. The importance measures, ``parameter_sensitivity``, the
        condition-based methods and ``allocate_redundancy`` raise
        NotImplementedError.
    capacity : dict[Any, float or dict], optional
        Each node's capacity, keyed by node name, by default None: the
        throughput it passes while it works, a positive number in any unit
        (the same for every node), or a dict ``{level: probability}`` of the
        levels it works at and the probability of each while it works; for
        [`capacity_distribution`][repyability.NonRepairableRBD.capacity_distribution].
        A failed node passes nothing. A node with no capacity given limits
        nothing (``inf``), unless its model has capacities of its own: a
        [`DegradingNode`][repyability.DegradingNode]'s stages, or a nested
        ``NonRepairableRBD`` with capacities, whose distribution it then
        has. A repeated node has the capacity of the node it repeats,
        wherever it is drawn: give that node's.

    Attributes
    ----------
    reliabilities : dict
        ``{node: model}`` as used: it includes the input and output nodes
        (as [`PerfectReliability`][repyability.PerfectReliability]) and
        leaves out the repeated nodes.
    repeated : dict
        ``{repeated_node: node_it_repeats}``.
    ccf_groups : list[CCFGroup]
        The validated common-cause groups.
    capacity : dict
        The capacities given, keyed by node name, as floats.
    nodes : list
        The component nodes: every node except the input and output nodes
        and the repeated nodes (each is the component it repeats).
    input_node : Hashable
        The input node.
    output_node : Hashable
        The output node.
    structure_check : dict
        The findings of the structural validation, e.g. ``"is_valid"``,
        ``"has_cycles"``, ``"nodes_with_no_reliability_distribution"``,
        ``"all_distributions_fixed"`` and ``"non_analytic_nodes"``.

    Raises
    ------
    ValueError
        If ``on_infeasible_rbd`` is not an allowed value, a node's model is
        its own name, ``input_node`` or ``output_node`` is not in the
        diagram, the diagram is invalid (with ``on_infeasible_rbd="raise"``),
        ``ccf_groups`` is invalid, or a capacity is not a positive number or
        is for a node that cannot take one (the input or output node, a
        repeated node, or one not in the diagram).

    Examples
    --------
    Two pumps in parallel, feeding a valve in series:

    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD
    >>> pump = surv.Weibull.from_params([1000, 1.5])
    >>> valve = surv.Exponential.from_params([1e-4])
    >>> rbd = NonRepairableRBD(
    ...     [("in", "p1"), ("in", "p2"), ("p1", "v"), ("p2", "v"),
    ...      ("v", "out")],
    ...     {"p1": pump, "p2": pump, "v": valve},
    ... )
    >>> rbd.input_node, rbd.output_node, rbd.nodes
    ('in', 'out', ['p1', 'p2', 'v'])
    >>> round(rbd.sf(100), 4)
    0.9891

    A 2-out-of-3 system of fixed-probability units, so ``x`` may be
    omitted:

    >>> from surpyval import FixedEventProbability
    >>> unit = FixedEventProbability.from_params(0.1)
    >>> edges = [("in", u) for u in "abc"] + [(u, "out") for u in "abc"]
    >>> two_of_three = NonRepairableRBD(
    ...     edges, {u: unit for u in "abc"}, k={"out": 2}
    ... )
    >>> round(two_of_three.sf(), 4)
    0.972

    A repeated node: one power supply drawn in both branches (``"psu2"``
    repeats ``"psu"``), so its failure fails both:

    >>> shared = NonRepairableRBD(
    ...     [("in", "a"), ("a", "psu"), ("psu", "out"),
    ...      ("in", "b"), ("b", "psu2"), ("psu2", "out")],
    ...     {"a": unit, "b": unit, "psu": unit, "psu2": "psu"},
    ... )
    >>> shared.repeated
    {'psu2': 'psu'}
    >>> round(shared.sf(), 4)
    0.891

    A beta-factor common cause coupling two parallel units lowers the
    independent result of 0.99:

    >>> from repyability import BetaFactor, CCFGroup
    >>> coupled = NonRepairableRBD(
    ...     [("in", "a"), ("in", "b"), ("a", "out"), ("b", "out")],
    ...     {"a": unit, "b": unit},
    ...     ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    ... )
    >>> round(coupled.sf(), 4)
    0.982
    """

    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        reliabilities: dict[Any, Any],
        k: Optional[dict[Any, int]] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
        ccf_groups: Optional[Iterable[CCFGroup]] = None,
        capacity: Optional[dict[Any, float]] = None,
    ):
        _check_on_infeasible_rbd(on_infeasible_rbd)
        # Capture the constructor inputs verbatim (before any mutation) so the
        # RBD can be faithfully serialised via to_dict()/to_json().
        edges = list(edges)
        ccf_groups = list(ccf_groups) if ccf_groups else []
        self._init_args = {
            "edges": [tuple(e) for e in edges],
            "reliabilities": dict(reliabilities),
            "k": dict(k) if k else None,
            "input_node": input_node,
            "output_node": output_node,
            "on_infeasible_rbd": on_infeasible_rbd,
            "ccf_groups": ccf_groups,
            "capacity": dict(capacity) if capacity else None,
        }
        reliabilities = copy(reliabilities)
        for key, value in reliabilities.items():
            if key == value:
                raise ValueError(
                    "Reliability dict cannot point to a node to itself"
                )
        # repeated checks if something was referenced from another node
        repeated = {
            k: v for k, v in reliabilities.items() if v in reliabilities.keys()
        }

        reliabilities = {
            k: v
            for k, v in reliabilities.items()
            if v not in reliabilities.keys()
        }
        for node, model in reliabilities.items():
            _check_model(node, model)

        # A repeated node stays where it is drawn, and the exact engine
        # treats every appearance as the one component it repeats (joining
        # the nodes in the graph instead would add paths the diagram does
        # not have). Set before the base class works out the structure,
        # with the names given models, which it checks against the edges.
        self._aliases = dict(repeated)
        self._models_given = list(reliabilities) + list(repeated)
        super().__init__(
            edges,
            None,
            k,
            input_node,
            output_node,
            on_infeasible_rbd,
            capacity=capacity,
        )
        self.structure_check["has_repeated_node_in_cycle"] = False
        # A model for a name in no edge is not part of the diagram (the
        # structure check reports it).
        unused = set(self.structure_check["nodes_in_no_edge"])
        reliabilities = {
            n: m for n, m in reliabilities.items() if n not in unused
        }
        repeated = {n: m for n, m in repeated.items() if n not in unused}
        self._aliases = dict(repeated)

        # Check for repeated cycles or non-repeated cycles
        if self.structure_check["has_unique_input_node"]:
            reliabilities[self.input_node] = PerfectReliability
        if self.structure_check["has_unique_output_node"]:
            reliabilities[self.output_node] = PerfectReliability

        # The nodes in the edges with no model, as the base class found them.
        missing = list(self.structure_check["nodes_with_no_model"])
        self.structure_check["is_missing_distributions"] = bool(missing)
        self.structure_check["nodes_with_no_reliability_distribution"] = (
            missing
        )

        self.reliabilities = reliabilities
        warn_nonparametric(nonparametric_nodes(reliabilities))
        self.repeated = repeated
        self.ccf_groups = self._validate_ccf_groups(ccf_groups)

        self._fixed_probs: bool = all(
            self._model_is_fixed(model)
            for model in self.reliabilities.values()
        )
        self.structure_check["all_distributions_fixed"] = self._fixed_probs

        # Record whether the system reliability can be solved analytically
        # (equivalently with the BDD), or whether it requires simulation
        # because one or more nodes are simulation-based (e.g. standby nodes).
        non_analytic_nodes = self.get_non_analytic_nodes()
        self.structure_check["is_analytically_solvable"] = (
            len(non_analytic_nodes) == 0
        )
        self.structure_check["non_analytic_nodes"] = non_analytic_nodes

    def _repr_details(self) -> List[str]:
        """Repeated nodes, perfect junctions and common-cause groups, for
        ``repr``."""
        out = []
        repeated = getattr(self, "repeated", {})
        if repeated:
            out.append(f"{len(repeated)} repeated node(s)")
        junctions = self._junctions()
        if junctions:
            out.append(
                "junction(s) "
                + ", ".join(repr(n) for n in sorted(junctions, key=str))
            )
        groups = getattr(self, "ccf_groups", [])
        if groups:
            out.append(f"{len(groups)} common-cause group(s)")
        return out

    def _junctions(self) -> frozenset:
        """The ``PerfectReliability`` nodes: drawing devices, such as a
        k-out-of-n vote (see ``RBD._junctions``)."""
        reliabilities = getattr(self, "reliabilities", {})
        return frozenset(
            node
            for node in self.nodes
            if reliabilities.get(node) is PerfectReliability
        )

    def _validate_node_overrides(self, working_nodes, broken_nodes) -> None:
        """Extends the base check with the repeated-node rule: a repeated node
        has been collapsed into the node it repeats, so it cannot be
        independently forced working or broken (in either set)."""
        for label, nodes in (
            ("working_nodes", working_nodes),
            ("broken_nodes", broken_nodes),
        ):
            for node in nodes:
                if node in self.repeated:
                    raise ValueError(
                        f"Node {node}, given to {label}, is a repeat of node "
                        f"{self.repeated[node]}. Create a new RBD where it is "
                        "not a repeated node."
                    )
        super()._validate_node_overrides(working_nodes, broken_nodes)

    @check_x
    def sf(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> Union[float, np.ndarray]:
        """System reliability (survival function) at time/s ``x``.

        The probability that the system is still working at ``x``,
        computed exactly from each node's reliability ``sf(x)`` by a
        Shannon decomposition over the minimal path sets (``method="p"``)
        or cut sets (``method="c"``). Nodes are independent apart from any
        common-cause groups: with ``ccf_groups`` the result sums one exact
        evaluation per combination of the groups' shared-cause outcomes,
        so the cost grows with the number and size of the groups.

        ``working_nodes`` and ``broken_nodes`` condition on the state of
        some components by setting their reliability to 1 or 0, e.g. to
        see the system reliability once a component has failed.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD (see
            [`is_fixed`][repyability.NonRepairableRBD.is_fixed]).
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.
        method : str, optional
            ``"p"`` (the default) uses the minimal path sets, which avoids
            deriving the cut sets; ``"c"`` uses the minimal cut sets. Both
            are exact and give the same result.

        Returns
        -------
        float or numpy.ndarray
            The system reliability: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD, or a working/broken
            node is unknown, is the input or output node, is in both sets,
            or is a repeat of another node.
        NotImplementedError
            If a working/broken node is a member of a common-cause group.

        Examples
        --------
        Two components in parallel, each 90% reliable, so the system is
        ``1 - 0.1 * 0.1 = 0.99`` reliable:

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {
        ...         "a": FixedEventProbability.from_params(0.1),
        ...         "b": FixedEventProbability.from_params(0.1),
        ...     },
        ... )
        >>> round(rbd.sf(), 4)
        0.99

        Conditioning on node ``"a"`` having failed leaves only ``"b"``:

        >>> round(rbd.sf(broken_nodes=["a"]), 4)
        0.9
        """
        # Normalise the (optional) node overrides into sets for O(1) lookup.
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)

        node_probabilities = self._base_node_probabilities(
            x, working_nodes, broken_nodes
        )
        if self.ccf_groups:
            return self._ccf_system_probability(
                node_probabilities, working_nodes, broken_nodes, method
            )
        return self.system_probability(node_probabilities, method=method)

    def _base_node_probabilities(
        self, x, working_nodes, broken_nodes
    ) -> Dict[Any, np.ndarray]:
        """Per-node reliability at ``x``, honouring the working/broken
        overrides (perfectly reliable / perfectly unreliable)."""
        node_probabilities: dict[Any, np.ndarray] = {}
        for node_name in self.reliabilities.keys():
            if node_name in working_nodes:
                node_probabilities[node_name] = PerfectReliability.sf(x)
            elif node_name in broken_nodes:
                node_probabilities[node_name] = PerfectUnreliability.sf(x)
            else:
                node_probabilities[node_name] = self.reliabilities[
                    node_name
                ].sf(x)
        return node_probabilities

    def _base_node_failures(
        self, x, working_nodes, broken_nodes
    ) -> Dict[Any, np.ndarray]:
        """Per-node probability of failing by ``x``, from each model's own
        ``ff`` (which keeps a small one's precision, where one less its
        reliability would not), honouring the working/broken overrides."""
        node_failures: dict[Any, np.ndarray] = {}
        for node_name, model in self.reliabilities.items():
            if node_name in working_nodes:
                node_failures[node_name] = PerfectReliability.ff(x)
            elif node_name in broken_nodes:
                node_failures[node_name] = PerfectUnreliability.ff(x)
            elif hasattr(model, "ff"):
                node_failures[node_name] = model.ff(x)
            else:
                node_failures[node_name] = 1.0 - np.asarray(model.sf(x))
        return node_failures

    def _importance_inputs(
        self, x, working_nodes, broken_nodes
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        """The nodes' reliabilities and probabilities of failing at ``x``
        that the importance measures are worked out from, with the working
        and broken nodes held (after checking them): the second from each
        model's own ``ff`` (see ``_base_node_failures``), so that a small
        one keeps its precision."""
        node_probabilities = self._probabilities_with_overrides(
            self.node_sf(x), working_nodes, broken_nodes
        )
        node_failures = self._base_node_failures(
            x, set(working_nodes or ()), set(broken_nodes or ())
        )
        return node_probabilities, node_failures

    def _validate_ccf_groups(self, ccf_groups) -> list:
        """Validate common-cause groups against the RBD structure.

        Members must be real, non-repeated component nodes (not the input or
        output node), each node in at most one group, and each group symmetric
        (identical component models). Returns the validated list.
        """
        seen: set = set()
        for group in ccf_groups:
            if not isinstance(group, CCFGroup):
                raise ValueError(
                    "ccf_groups must contain CCFGroup instances, got "
                    f"{type(group).__name__}."
                )
            for member in group.members:
                if member not in self.reliabilities:
                    raise ValueError(
                        f"CCF group member {member!r} is not a node in the "
                        "RBD."
                    )
                if member in (self.input_node, self.output_node):
                    raise ValueError(
                        f"CCF group member {member!r} cannot be the input or "
                        "output node."
                    )
                if member in self.repeated:
                    raise ValueError(
                        f"CCF group member {member!r} cannot be a repeated "
                        "node."
                    )
                if member in seen:
                    raise ValueError(
                        f"Node {member!r} appears in more than one CCF group."
                    )
                seen.add(member)
            self._check_symmetric_group(group)
        return list(ccf_groups)

    def _check_symmetric_group(self, group) -> None:
        # Standard CCF theory is for symmetric groups, so the members must
        # carry identical component models. Compare via the serialised form
        # (exact); skip silently if a member is not serialisable.
        from repyability.rbd.serialisation import serialise_model

        try:
            specs = [
                serialise_model(self.reliabilities[m]) for m in group.members
            ]
        except Exception:
            return
        if any(spec != specs[0] for spec in specs[1:]):
            raise ValueError(
                f"CCF group {list(group.members)} is not symmetric: its "
                "members must carry identical component models."
            )

    def _capacity_models(self) -> dict:
        """``{node: model}`` for the nodes with no capacity entry whose model
        has a capacity distribution of its own: a ``DegradingNode``, or a
        nested ``NonRepairableRBD`` with capacities."""
        return {
            node: model
            for node, model in self.reliabilities.items()
            if node not in self.capacity
            and node not in self.in_or_out
            and (
                isinstance(model, DegradingNode)
                or (
                    isinstance(model, NonRepairableRBD)
                    and model._has_capacity()
                )
            )
        }

    def _ccf_system_probability(
        self, base_probabilities, working_nodes, broken_nodes, method
    ) -> np.ndarray:
        """Exact system reliability with common-cause groups, by conditioning
        on each group's shared-cause event.

        For each beta-factor group the shared cause either fires (probability
        ``beta * Q``, failing every member) or does not (each member fails only
        independently, reliability ``1 - (1 - beta) * Q``). Conditioning on the
        independent shared-cause events of every group and summing over the
        ``2 ** len(groups)`` combinations gives the exact result — each term a
        call to the ordinary independent engine. ``beta = 0`` recovers it
        exactly.
        """
        terms = [
            np.asarray(weight)
            * np.asarray(
                self.system_probability(node_probabilities, method=method)
            )
            for weight, node_probabilities in self._ccf_conditions(
                base_probabilities, working_nodes, broken_nodes
            )
        ]
        return np.sum(terms, axis=0)

    def _ccf_conditions(
        self, base_probabilities, working_nodes, broken_nodes
    ) -> Iterator[tuple]:
        """The combinations of the common-cause groups' shock outcomes (see
        ``_ccf_system_probability``): for each, its probability and the node
        reliabilities given it, under which the nodes are independent."""
        for weight, node_probabilities, _ in self._ccf_outcomes(
            base_probabilities, None, working_nodes, broken_nodes
        ):
            yield weight, node_probabilities

    def _ccf_outcomes(
        self, base_probabilities, base_failures, working_nodes, broken_nodes
    ) -> Iterator[tuple]:
        """``_ccf_conditions``, with the nodes' probabilities of failing
        given each outcome too when ``base_failures`` (each node's own
        ``ff``) is given, so that a small one keeps its precision (else
        None)."""
        from itertools import product

        forced = working_nodes | broken_nodes
        for group in self.ccf_groups:
            if forced.intersection(group.members):
                raise NotImplementedError(
                    "Forcing a CCF group member via working_nodes / "
                    "broken_nodes is not supported yet."
                )

        # Each group's mutually-exclusive shock outcomes: (weight, {member:
        # reliability}, {member: unreliability}) for every subset that can
        # fail together plus the no-shock case, from the model's
        # decomposition of Q(t) (taken from a representative member, since
        # groups are symmetric).
        group_outcomes = []
        for group in self.ccf_groups:
            first = group.members[0]
            Q = (
                1.0 - np.atleast_1d(base_probabilities[first])
                if base_failures is None
                else np.atleast_1d(np.asarray(base_failures[first], float))
            )
            q_independent, shocks = group.model.decompose(group.members, Q)
            r_independent = 1.0 - q_independent
            outcomes = []
            total_shock = np.zeros_like(Q)
            for subset, prob in shocks:
                total_shock = total_shock + prob
                outcomes.append(
                    (
                        prob,
                        {
                            member: (
                                np.zeros_like(Q)
                                if member in subset
                                else r_independent
                            )
                            for member in group.members
                        },
                        {
                            member: (
                                np.ones_like(Q)
                                if member in subset
                                else q_independent
                            )
                            for member in group.members
                        },
                    )
                )
            # No common-cause shock: every member fails only independently.
            outcomes.append(
                (
                    1.0 - total_shock,
                    {member: r_independent for member in group.members},
                    {member: q_independent for member in group.members},
                )
            )
            group_outcomes.append(outcomes)

        for combo in product(*group_outcomes):
            node_probabilities = dict(base_probabilities)
            node_failures = (
                None if base_failures is None else dict(base_failures)
            )
            weight: Any = 1.0
            for outcome_weight, member_probs, member_fails in combo:
                weight = weight * outcome_weight
                node_probabilities.update(member_probs)
                if node_failures is not None:
                    node_failures.update(member_fails)
            yield weight, node_probabilities, node_failures

    def _require_no_ccf_for_states(self) -> None:
        """Raise if the RBD has CCF groups, for the condition-based methods
        (``sf_given_state``, ``remaining_life``,
        ``importances_given_state``)."""
        if self.ccf_groups:
            raise NotImplementedError(
                "Condition-based evaluation (sf_given_state / remaining_life "
                "/ importances_given_state) does not yet account for "
                "common-cause (CCF) groups; use sf()/ff() for CCF system "
                "reliability."
            )

    def _require_no_ccf_for_allocation(self, reliabilities=False) -> None:
        """Raise if the RBD has CCF groups, for redundancy allocation (or,
        with ``reliabilities``, reliability-redundancy allocation)."""
        if not self.ccf_groups:
            return
        if reliabilities:
            raise NotImplementedError(
                "Reliability-redundancy allocation does not yet account for "
                "common-cause (CCF) groups."
            )
        raise NotImplementedError(
            "Redundancy allocation does not yet account for common-cause "
            "(CCF) groups: duplicating a group member would also have to "
            "extend its common-cause group."
        )

    def _require_time_varying(self) -> None:
        """Raise if the system reliability does not vary with time, so the
        time to a reliability is undefined."""
        if self.is_fixed:
            raise ValueError(
                "System reliability does not vary with time (all nodes are "
                "fixed-probability); the time to a reliability is undefined."
            )

    def _require_replayable(self) -> None:
        """Raise unless every node's draws can be replayed from uniforms,
        as common random numbers (``compare``) need."""
        for node in self._components():
            if row_sampler(self.reliabilities[node]) is None:
                raise NotImplementedError(
                    "Common random numbers need every node's draws to be "
                    "replayable from uniforms (surpyval parametric "
                    f"distributions and the composite nodes built from "
                    f"them); node {node!r}'s are not."
                )

    def _require_capacity_outside_groups(self) -> None:
        """Raise if a common-cause group member takes its capacity
        distribution from its own model, which the capacity analysis does
        not model."""
        grouped = {m for group in self.ccf_groups for m in group.members}
        own = grouped & set(self._capacity_models())
        if own:
            raise NotImplementedError(
                "The capacity analysis does not model a common-cause group "
                "whose members have capacity distributions of their own "
                f"(nodes {sorted(own, key=str)}): give them capacities "
                "instead."
            )

    def _require_no_ccf(self) -> None:
        """Raise if the RBD has CCF groups, for the probability-dependent
        importance / sensitivity measures that do not yet account for them.
        (Common-cause coupling is currently reflected only in ``sf()`` /
        ``ff()``; ``structural_importance`` is probability-free and so is
        unaffected.)"""
        if self.ccf_groups:
            raise NotImplementedError(
                "Importance and sensitivity measures do not yet account for "
                "common-cause (CCF) groups; CCF is currently supported by "
                "sf() / ff()."
            )

    @check_x
    def ff(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> Union[float, np.ndarray]:
        """System unreliability (failure probability) at time/s ``x``.

        The probability that the system has failed by ``x``: ``1 - sf(x)``,
        exact, and honouring common-cause groups and the working/broken
        nodes, as [`sf`][repyability.NonRepairableRBD.sf] does. It is
        worked out in its own right, from each node's probability of
        failing (its model's ``ff``), as a sum of products, so a small
        one keeps its full relative precision: a system that fails with
        probability ``1e-15`` gets ``1e-15``, where ``1 - sf(x)`` would
        lose it (``sf`` rounds to within ``1e-16`` of 1). A node's own
        precision limits it: a numerical one, such as a cold-standby group
        of non-exponential units (a convolution), is accurate to about
        ``1e-6``, so a smaller probability through it needs
        [`unreliability_interval`][repyability.NonRepairableRBD.unreliability_interval].

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (unreliability 0), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (unreliability 1), by default none.
        method : str, optional
            ``"p"`` or ``"c"``, as for ``sf``; both give the same result,
            computed the same way.

        Returns
        -------
        float or numpy.ndarray
            The system unreliability: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {
        ...         "a": FixedEventProbability.from_params(0.1),
        ...         "b": FixedEventProbability.from_params(0.1),
        ...     },
        ... )
        >>> round(rbd.ff(), 4)
        0.01

        A small probability keeps its precision: three components in
        parallel, each failing with probability ``1e-6``, fail together
        with probability ``1e-18``:

        >>> rbd = NonRepairableRBD(
        ...     [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
        ...     {n: FixedEventProbability.from_params(1e-6) for n in "abc"},
        ... )
        >>> f"{rbd.ff():.6g}"
        '1e-18'
        """
        return self._sf_and_ff(
            x, working_nodes, broken_nodes, method, sf=False
        )[1]

    def _sf_and_ff(
        self, x, working_nodes, broken_nodes, method: str = "p", sf=True
    ) -> Tuple[Any, np.ndarray]:
        """The system's reliability (unless ``sf`` is false: then None) and
        unreliability at ``x``, after checking the arguments as ``sf``
        does. Each is a sum of products of the nodes' own reliabilities and
        failure probabilities (their models' ``sf`` and ``ff``), so that a
        small one of either keeps its precision; with common-cause groups,
        summed over their outcomes as for ``sf`` (``method`` changes
        nothing here)."""
        if method not in ("p", "c"):
            raise ValueError("`method` must be either 'p' or 'c'")
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)

        node_probabilities = self._base_node_probabilities(
            x, working_nodes, broken_nodes
        )
        node_failures = self._base_node_failures(
            x, working_nodes, broken_nodes
        )
        if not self.ccf_groups:
            return self._system_probabilities(
                node_probabilities, node_failures, works=sf
            )
        up: Any = 0.0
        down: Any = 0.0
        for weight, probabilities, failures in self._ccf_outcomes(
            node_probabilities, node_failures, working_nodes, broken_nodes
        ):
            works, fails = self._system_probabilities(
                probabilities, failures, works=sf
            )
            weight = np.asarray(weight)
            up = up + weight * works if sf else None
            down = down + weight * fails
        return up, down

    def capacity_distribution(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> CapacityDistribution:
        """The exact distribution of the system's capacity at time/s ``x``.

        Each node carries its ``capacity`` (given when the RBD was built)
        while it works, at one level or several, and nothing once it has
        failed; the system's capacity is the most that can flow through the
        working nodes from the input to the output. A node given no
        capacity limits nothing, unless its model has capacities of its own
        (a ``DegradingNode``'s stages, or a nested RBD's capacities).
        With three pumps of half the demand each, one failure costs nothing
        and two cost half the output. The capacity is positive exactly when
        the system works, so the probability that it is positive is
        [`sf`][repyability.NonRepairableRBD.sf].

        The distribution is exact, from each node's reliability at ``x``,
        worked out as ``sf`` is (see
        [`RBD.system_capacity`][repyability.RBD.system_capacity]), and
        honours common-cause groups.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working, by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed, by default none.

        Returns
        -------
        CapacityDistribution
            The capacities the system can have, and their probabilities: one
            per level for a scalar ``x``, and one row per level and one
            column per time for an array. Its ``meets(demand)`` is the
            probability that the capacity meets a demand, and ``mean()`` the
            expected capacity.

        A node forced working is at the levels it can work at, in
        proportion to their probabilities at ``x``.

        Raises
        ------
        ValueError
            If no node has a capacity, ``x`` is omitted for a time-varying
            RBD, or a working/broken node is invalid (as for ``sf``).
        NotImplementedError
            If a working/broken node is a member of a common-cause group, or
            a member of one has a capacity distribution from its model.

        Examples
        --------
        Three pumps of 50 each in parallel, each 90% reliable at the time
        of interest, feeding a pipe that carries 120:

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> pump = FixedEventProbability.from_params(0.1)
        >>> pipe = FixedEventProbability.from_params(0.0)
        >>> plant = NonRepairableRBD(
        ...     [("in", "a"), ("in", "b"), ("in", "c"),
        ...      ("a", "pipe"), ("b", "pipe"), ("c", "pipe"),
        ...      ("pipe", "out")],
        ...     {"a": pump, "b": pump, "c": pump, "pipe": pipe},
        ...     capacity={"a": 50, "b": 50, "c": 50, "pipe": 120},
        ... )
        >>> capacity = plant.capacity_distribution()
        >>> capacity.levels.tolist()
        [0.0, 50.0, 100.0, 120.0]
        >>> round(capacity.meets(100), 4)  # two pumps or more
        0.972
        >>> round(capacity.mean(), 2)
        113.13
        """
        x, scalar = _times(self, x)
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        own = {}
        for node, model in self._capacity_models().items():
            distribution = model.capacity_distribution(x)
            own[node] = self._forced(
                (
                    distribution.levels,
                    np.atleast_2d(distribution.probabilities.T).T,
                ),
                node,
                working_nodes,
                broken_nodes,
            )
        self._require_capacity_outside_groups()
        base = self._base_node_probabilities(x, working_nodes, broken_nodes)
        conditions: Iterable[tuple] = (
            self._ccf_conditions(base, working_nodes, broken_nodes)
            if self.ccf_groups
            else [(1.0, base)]
        )
        levels, rows = [], []
        for weight, node_probabilities in conditions:
            arrays, size = self._node_arrays(node_probabilities)
            part = self._capacity_arrays(arrays, size, own)
            levels.append(part[0])
            rows.append(np.asarray(weight) * part[1])
        merged = _capacity.merged(np.concatenate(levels), np.vstack(rows))
        return CapacityDistribution(
            merged[0], merged[1][:, 0] if scalar else merged[1]
        )

    def sf_uncertainty(
        self,
        x: Optional[ArrayLike] = None,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
    ) -> UncertaintyResult:
        """System reliability over plausible node models (parameter
        uncertainty).

        A node's model is estimated from data, so its parameters are
        uncertain: this is *epistemic* uncertainty, about what the model
        is, as opposed to the aleatory variability the model describes.
        Each of the ``n_draws`` draws gives every uncertain node a
        plausible model, and the system reliability at ``x`` is computed
        exactly for that draw (all draws at once, by the vectorised exact
        engine). The spread of the results, and its percentiles
        (``interval``), say how well the system reliability is known.

        RePyability does not fit models: the draws use what the fit, done
        in surpyval, provides. A node's uncertainty is one of:

        - ``"fit"``: its parameters are drawn from the normal approximation
          of its maximum-likelihood fit (surpyval's ``hess_inv``), on the
          log scale for a parameter that must be positive and the logit
          scale for one in (0, 1), so that every draw is valid; an offset,
          zero-inflation or limited-failure-population parameter keeps its
          fitted value;
        - ``{parameter name: distribution}``: each named parameter is drawn
          from its distribution (anything with ``qf`` or ``ppf``, such as a
          surpyval or scipy.stats distribution), the others kept;
        - a list of models (e.g. refits to bootstrap resamples, or
          posterior draws, made in surpyval), drawn with replacement.

        Nodes of one population, such as identical pumps fitted to the
        same data, share their uncertainty: give them together as a tuple
        of node names, and each draw gives them the same model. Drawing
        them independently would understate the uncertainty, which is
        about the one population's parameters.

        Parameters
        ----------
        x : array_like, optional
            Time/s, a number or an array. May be left out when every node
            model, and every drawn model, is a fixed probability.
        uncertainty : dict
            ``{node or tuple of nodes: uncertainty}`` for the uncertain
            nodes (see above). The other nodes keep their models.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seed for the draws, for reproducible results.

        Returns
        -------
        UncertaintyResult
            The system reliability of every draw (``samples``), the nominal
            value with the nodes' own models, and summaries: ``mean``,
            ``median``, ``std``, ``percentile`` and ``interval`` (see
            [`UncertaintyResult`][repyability.UncertaintyResult]).

        Raises
        ------
        ValueError
            If ``uncertainty`` is empty, names an unknown, input, output or
            repeated node, or names a node twice; if a tuple's nodes have
            different models; if an uncertainty cannot be drawn (e.g.
            ``"fit"`` for a model with no fitted covariance, or an unknown
            parameter); if ``n_draws`` is not a positive integer; or if
            ``x`` is left out for time-varying models.
        NotImplementedError
            If the RBD has common-cause groups.

        Examples
        --------
        Two pumps in parallel, of one type whose exponential failure rate
        is uncertain (known only to lie between 1 and 3 per 1000 h), in
        series with a valve. Both pumps share the one uncertain rate:

        >>> import scipy.stats as st
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": surv.Exponential.from_params([0.002]),
        ...         "p2": surv.Exponential.from_params([0.002]),
        ...         "v": surv.FixedEventProbability.from_params(0.01),
        ...     },
        ... )
        >>> rate = {"failure_rate": st.uniform(0.001, 0.002)}
        >>> result = rbd.sf_uncertainty(
        ...     500, {("p1", "p2"): rate}, n_draws=10_000, seed=1
        ... )
        >>> round(result.nominal, 4)  # at the nominal rate, 2 per 1000 h
        0.5944
        >>> lower, upper = result.interval(0.9)
        >>> round(lower, 3), round(upper, 3)
        (0.409, 0.813)
        """
        self._require_no_ccf()
        if isinstance(n_draws, bool) or not isinstance(
            n_draws, (int, np.integer)
        ):
            raise ValueError(f"n_draws must be an integer, got {n_draws!r}.")
        if n_draws < 1:
            raise ValueError(f"n_draws must be at least 1, got {n_draws}.")
        if not uncertainty:
            raise ValueError(
                "Give the uncertain nodes: uncertainty={node: 'fit'}, for "
                "example."
            )
        components = set(self.nodes)
        groups = []
        seen: set = set()
        for key, spec in uncertainty.items():
            if key in components or not isinstance(key, tuple):
                members = (key,)
            else:
                members = key
            for node in members:
                if node in self.repeated:
                    raise ValueError(
                        f"Node {node!r} is a repeat of {self.repeated[node]!r}"
                        "; give the uncertainty of that node instead."
                    )
                if node not in components:
                    raise ValueError(
                        f"{node!r} in uncertainty is not a component node."
                    )
                if node in seen:
                    raise ValueError(
                        f"Node {node!r} is given an uncertainty twice."
                    )
                seen.add(node)
            model = self.reliabilities[members[0]]
            for node in members[1:]:
                if not self._same_model(self.reliabilities[node], model):
                    raise ValueError(
                        f"Nodes {list(members)!r} share their uncertainty, "
                        "so they must have the same model."
                    )
            groups.append((members, spec))

        rng = np.random.default_rng(seed)
        drawn: Dict[Hashable, list] = {}
        for members, spec in groups:
            label = (
                f"Node {members[0]!r}"
                if len(members) == 1
                else f"Nodes {list(members)!r}"
            )
            models = draw_models(
                self.reliabilities[members[0]], spec, n_draws, rng, label
            )
            for node in members:
                drawn[node] = models

        fixed = self.is_fixed and all(
            self._model_is_fixed(m) for ms in drawn.values() for m in ms
        )
        if x is None:
            if not fixed:
                raise ValueError(
                    "x is required: a node model's probability depends on "
                    "time."
                )
            x = 1.0
        scalar = np.ndim(x) == 0
        times = np.atleast_1d(np.asarray(x, dtype=float))
        probabilities: dict = {}
        for node in self.nodes:
            if node in drawn:
                rows = [
                    np.broadcast_to(
                        np.asarray(m.sf(times), dtype=float), times.shape
                    )
                    for m in drawn[node]
                ]
                probabilities[node] = np.concatenate(rows)
            else:
                values = np.broadcast_to(
                    np.asarray(self.reliabilities[node].sf(times), float),
                    times.shape,
                )
                probabilities[node] = np.tile(values, n_draws)
        samples = np.asarray(
            self.system_probability(probabilities), dtype=float
        ).reshape(n_draws, len(times))
        nominal = np.asarray(self.sf(times), dtype=float)
        if scalar:
            return UncertaintyResult(
                samples=samples[:, 0],
                nominal=float(nominal.reshape(-1)[0]),
                n_draws=n_draws,
            )
        return UncertaintyResult(
            samples=samples, nominal=nominal, n_draws=n_draws
        )

    @staticmethod
    def _same_model(a, b) -> bool:
        """Whether two node models are the same: one object, or the same
        distribution with the same parameters."""
        if a is b:
            return True
        name_a = getattr(getattr(a, "dist", None), "name", None)
        name_b = getattr(getattr(b, "dist", None), "name", None)
        if name_a is None or name_a != name_b:
            return False
        params_a = np.ravel(np.asarray(getattr(a, "params", []), float))
        params_b = np.ravel(np.asarray(getattr(b, "params", []), float))
        return (
            params_a.shape == params_b.shape
            and bool(np.all(params_a == params_b))
            and all(
                getattr(a, extra, None) == getattr(b, extra, None)
                for extra in ("gamma", "p", "f0")
            )
        )

    def allocate_redundancy(
        self,
        costs: Dict[
            Hashable,
            Union[float, Dict[Hashable, float], Sequence[ComponentOption]],
        ],
        *,
        budget: Union[float, Dict[Hashable, float], None] = None,
        target: Optional[float] = None,
        minimise: Optional[Hashable] = None,
        t: Optional[float] = None,
        max_units: Union[int, Dict[Hashable, int], None] = None,
        required: Union[int, Dict[Hashable, int], None] = None,
        strategy: Union[str, Dict[Hashable, str]] = "active",
        switching_probability: Union[float, Dict[Hashable, float]] = 1.0,
        mixing: bool = True,
        method: str = "exact",
    ) -> RedundancyAllocation:
        """Choose how many redundant copies of each node to fit.

        Solves the Redundancy Allocation Problem: pick how many independent
        copies of each node in ``costs`` to fit in active parallel, to either

        - maximise system reliability within the ``budget``, or
        - minimise total cost with system reliability at least ``target``
          (and within the ``budget``, if one is also given).

        A copy may use one resource (a number: its cost) or several (a dict
        such as ``{"cost": 4000, "weight": 12}``, as in Fyffe, Hines & Lee,
        1968), and the budget then limits each resource it names. A node
        may also be given a choice of component types, a list of
        [`ComponentOption`][repyability.ComponentOption] with their own
        reliabilities and costs: its copies are then all of one type or,
        with ``mixing``, any combination of types (Coit & Smith, 1996).
        Copies of reliabilities ``p_1, ..., p_n`` in active parallel have
        reliability ``1 - (1 - p_1) ... (1 - p_n)`` (``1 - (1 - p) ** n``
        for ``n`` identical copies); a node may instead need several of its
        copies working (``required``, k-out-of-n: Coit & Liu, 2000), and its
        spares may be cold standby rather than active (``strategy``: Coit,
        2001), or either, whichever is better (Coit, 2003). The system
        reliability of each candidate allocation is computed exactly from
        its nodes' reliabilities, so any RBD structure works; only a cold
        standby node's own reliability may be simulated (see
        ``strategy``). Nodes not in ``costs`` stay as they are.
        Reliability is evaluated at the single mission time ``t``.

        ``method="exact"`` returns a proven optimum. When every costed node
        is in series with the rest of the system (it lies on every path, as
        in the textbook series of subsystems), the system reliability is the
        product of the costed nodes' reliabilities and the rest's, and a
        dynamic program over the nodes finds the optimum however many
        costed nodes there are. On other structures an exhaustive search
        finds it, which suits a handful of costed nodes.

        Parameters
        ----------
        costs : dict
            ``{node: what one copy uses}`` for the nodes that may be
            duplicated: a number (its cost) or, for several resources, a dict
            of resource → amount, naming the same resources for every node.
            Or, for a choice of component types, a list of
            [`ComponentOption`][repyability.ComponentOption] (each giving
            what one copy of it uses in the same way); the node's own model
            is then not used. Every copy is counted, including the original,
            so the cheapest allocation (one of each) uses the sum of the
            amounts. A number must be finite and positive; the amounts in a
            dict finite and non-negative.
        budget : float or dict, optional
            The limit on the one resource (a number), or on each resource a
            dict names (resources it leaves out are not limited). With no
            ``target``, reliability is maximised within it, and it must
            afford one of each costed node. With a ``target`` it adds limits
            to the minimisation.
        target : float, optional
            Minimise the total of one resource subject to system reliability
            >= target, in (0, 1).
        minimise : Hashable, optional
            The resource a ``target`` minimises. Needed only with several
            resources, none of them called ``"cost"`` (which is minimised by
            default).
        t : float, optional
            The mission time at which reliability is evaluated, a single
            number. Required for a time-varying RBD (or time-varying
            options); not needed when every node and option is a fixed
            probability.
        max_units : int or dict, optional
            The most copies allowed (at least 1) of every costed node (an
            int) or of particular costed nodes (a dict; nodes it leaves out
            are unlimited), counting copies of every type, e.g. for space
            limits. By default unlimited; the budget or target bounds the
            search, so every node (every option) must use some of a limited
            (or, with a target, the minimised) resource, or have a cap.
        required : int or dict, optional
            The number of a node's copies that must work (k-out-of-n), for
            every costed node (an int) or particular ones (a dict; others
            need 1). By default 1. Every design has at least this many
            copies.
        strategy : str or dict, optional
            How a node's copies are arranged, for every costed node (a
            string) or particular ones (a dict; others are active):
            ``"active"`` (the default): all copies operate, and the node
            works while ``required`` of them do; ``"cold"``: ``required``
            copies operate and the others wait unpowered as cold spares,
            switched in, in the order of the node's options, as operating
            copies fail (a [`StandbyModel`][repyability.StandbyModel]: exact
            for identical Exponential units, a numerical convolution for one
            unit required, and otherwise simulated from 10 000 lifetimes,
            seeded, so reproducible); ``"choose"``: whichever of
            the two is better, chosen for each node by the optimiser. Cold
            standby needs lifetime models, not fixed probabilities.
        switching_probability : float or dict, optional
            For cold standby, the probability that switching onto a spare
            succeeds, for every cold node (a number) or particular ones (a
            dict). By default 1.0 (perfect switching). Imperfect switching
            is supported with one unit required.
        mixing : bool, optional
            For nodes with options: whether their copies may be of
            different types (the default), or must all be of one type.
        method : str, optional
            ``"exact"`` (the default) returns a proven optimum, by dynamic
            programming for costed nodes in series with the rest (giving up
            with an explanatory error beyond 2,000,000 partial allocations)
            and otherwise by an exhaustive search (giving up beyond 500,000
            allocations examined, which is a handful of costed nodes).
            ``"greedy"`` repeatedly takes the step (one more copy, or a
            change of type) with the best log-reliability gain per unit of
            what it uses (its cost, the minimised resource with a target,
            or with several limits its total share of them): fast for any
            size, usually optimal or close, but not guaranteed optimal.

        Returns
        -------
        RedundancyAllocation
            The chosen ``units`` per costed node (copies of every type),
            with the resulting ``reliability``, the total ``cost`` (of the
            resource minimised, ``"cost"``, or the first resource), the
            total of every resource in ``resources``, the ``method`` used,
            each node's ``strategy`` and, for nodes with options, how many
            of each type in ``mix`` (see
            [`RedundancyAllocation`][repyability.RedundancyAllocation]).

        Raises
        ------
        ValueError
            If neither ``budget`` nor ``target`` is given, or on any other
            invalid input: an unknown ``method``; empty ``costs``; a costed
            node that is not a component node or is a repeated node; a cost
            that is not finite and positive (an amount in a dict, not finite
            and non-negative); nodes naming different resources, or a
            mixture of numbers and dicts; invalid options (an empty list,
            duplicate names, or a reliability that is not a model or a
            probability); a ``budget`` naming an unknown resource, or a
            number with several resources; a node nothing limits (see
            ``max_units``); an invalid ``max_units``, ``minimise``,
            ``mixing``, ``required`` (or one above ``max_units``),
            ``strategy`` or ``switching_probability``; cold standby of a
            fixed probability; ``t`` missing for a time-varying RBD or not a
            single number; a budget that cannot afford the required copies
            of each costed node; a ``target`` outside (0, 1) or not
            reachable (within ``max_units`` and the budget); or an exact
            search that is too large.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        References
        ----------
        D. E. Fyffe, W. W. Hines and N. K. Lee, "System reliability
        allocation and a computational algorithm", IEEE Transactions on
        Reliability, 17(2), 64-69, 1968.

        J. D. Kettelle, "Least-cost allocations of reliability investment",
        Operations Research, 10(2), 249-265, 1962 (dominance in a dynamic
        program over the subsystems of a series system).

        D. W. Coit and A. E. Smith, "Reliability optimization of
        series-parallel systems using a genetic algorithm", IEEE
        Transactions on Reliability, 45(2), 254-260, 1996 (component
        mixing).

        D. W. Coit and J. Liu, "System reliability optimization with
        k-out-of-n subsystems", International Journal of Reliability,
        Quality and Safety Engineering, 7(2), 129-142, 2000.

        D. W. Coit, "Cold-standby redundancy optimization for nonrepairable
        systems", IIE Transactions, 33(6), 471-478, 2001.

        D. W. Coit, "Maximization of system reliability with a choice of
        redundancy strategies", IIE Transactions, 35(6), 535-543, 2003.

        Examples
        --------
        Two components in series, 90% and 80% reliable, one cost unit each.
        With a budget of 3 the extra copy goes to the weaker component:

        >>> from surpyval import FixedEventProbability
        >>> from repyability import ComponentOption, NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": FixedEventProbability.from_params(0.1),
        ...         "b": FixedEventProbability.from_params(0.2),
        ...     },
        ... )
        >>> best = rbd.allocate_redundancy({"a": 1.0, "b": 1.0}, budget=3)
        >>> best.units
        {'a': 1, 'b': 2}
        >>> round(best.reliability, 4)
        0.864

        The cheapest design that is at least 90% reliable needs two of each:

        >>> rbd.allocate_redundancy({"a": 1.0, "b": 1.0}, target=0.9).units
        {'a': 2, 'b': 2}

        With weight limited as well as cost, a second ``a`` (3 kg) no longer
        fits, and the budget goes on more copies of ``b``:

        >>> costs = {
        ...     "a": {"cost": 1, "weight": 3},
        ...     "b": {"cost": 1, "weight": 1},
        ... }
        >>> best = rbd.allocate_redundancy(
        ...     costs, budget={"cost": 4, "weight": 6}
        ... )
        >>> best.units, round(best.reliability, 4), best.resources
        ({'a': 1, 'b': 3}, 0.8928, {'cost': 4.0, 'weight': 6.0})

        ``b`` may instead be built from a standard part (80%, cost 1) or a
        premium one (98%, cost 2). Within a budget of 5 the best design
        mixes them; with every copy of ``b`` of one type it is less
        reliable:

        >>> choice = [
        ...     ComponentOption("standard", 0.8, cost=1),
        ...     ComponentOption("premium", 0.98, cost=2),
        ... ]
        >>> best = rbd.allocate_redundancy({"a": 1, "b": choice}, budget=5)
        >>> best.units, best.mix, round(best.reliability, 5)
        ({'a': 2, 'b': 2}, {'b': {'standard': 1, 'premium': 1}}, 0.98604)
        >>> same = rbd.allocate_redundancy(
        ...     {"a": 1, "b": choice}, budget=5, mixing=False
        ... )
        >>> same.mix, round(same.reliability, 5)
        ({'b': {'standard': 3}}, 0.98208)

        A pump that must deliver with two units running (2-out-of-n), as
        active copies or with cold spares, whichever is better. At 1000
        hours cold spares are better:

        >>> import surpyval as surv
        >>> station = NonRepairableRBD(
        ...     [("s", "pumps"), ("pumps", "t")],
        ...     {"pumps": surv.Exponential.from_params([1 / 2000])},
        ... )
        >>> best = station.allocate_redundancy(
        ...     {"pumps": 1}, budget=4, t=1000, required=2, strategy="choose"
        ... )
        >>> best.units, best.strategy, round(best.reliability, 4)
        ({'pumps': 4}, {'pumps': 'cold'}, 0.9197)
        """
        if budget is None and target is None:
            raise ValueError("Give a budget, a target, or both.")
        problem = self._redundancy_problem(
            costs,
            budget,
            target,
            minimise,
            t,
            max_units,
            required,
            strategy,
            switching_probability,
            mixing,
            method,
        )
        nodes, kinds, caps, limits = (
            problem.nodes,
            problem.kinds,
            problem.caps,
            problem.limits,
        )
        primary, fewest, ways = problem.primary, problem.fewest, problem.ways
        evaluate, within, menus = (
            problem.evaluate,
            problem.within,
            problem.menus,
        )
        greedy_strategies = problem.greedy_strategies
        found: Optional[tuple]
        if target is None:
            if method == "greedy":
                found = redundancy_allocation.greedy(
                    evaluate,
                    kinds,
                    caps,
                    budget=limits,
                    primary=primary,
                    mixing=mixing,
                    fewest=fewest,
                    strategies=greedy_strategies,
                )
                if not within(found[2]):
                    raise ValueError(
                        "The greedy search could not start within the "
                        "budget (one copy of each node, of its cheapest "
                        "option); use method='exact'."
                    )
            elif problem.in_series:
                found = self._series_redundancy(
                    nodes, menus(), limits, problem.base, evaluate, primary
                )
            else:
                found = redundancy_allocation.exact_max_reliability(
                    evaluate, menus(), limits, primary
                )
            if found is None:
                raise ValueError(
                    "No allocation fits within the budget (one copy of each "
                    "node, of some option, does not fit)."
                )
        else:
            target = float(target)
            if not (0.0 < target < 1.0):
                raise ValueError(f"target must be in (0, 1), got {target!r}.")
            # The best reliability any allocation can reach: every costed node
            # at its cap of its most reliable kind (unlimited active copies of
            # a kind that can work reach 1; cold spares are taken to).
            ceiling = evaluate(
                tuple(
                    (
                        1.0
                        if "cold" in ways[i]
                        else 1.0
                        - min(
                            (
                                float(p == 0.0)
                                if caps[i] == math.inf
                                else active_unreliability(
                                    [1.0 - p], [caps[i]], fewest[i]
                                )
                            )
                            for p, _ in kinds[i]
                        )
                    )
                    for i in range(len(nodes))
                )
            )
            if ceiling < target:
                raise ValueError(
                    f"target {target:g} is unreachable: the best achievable "
                    f"system reliability is {ceiling:.6g}"
                    + (
                        " within max_units."
                        if max_units is not None
                        else " even with unlimited copies of the costed "
                        "nodes."
                    )
                )
            limited = any(math.isfinite(limit) for limit in limits)
            amounts, m = problem.amounts, problem.m
            start: Optional[tuple] = redundancy_allocation.greedy(
                evaluate,
                kinds,
                caps,
                budget=limits,
                target=target,
                primary=primary,
                mixing=mixing,
                fewest=fewest,
                strategies=greedy_strategies,
            )
            assert start is not None
            if start[0] < target or not within(start[2]):
                if method == "greedy" or not limited:
                    raise ValueError(
                        f"target {target:g} could not be reached: adding "
                        "copies stopped improving the system at reliability "
                        f"{start[0]:.6g}"
                        + (" within the budget." if limited else ".")
                    )
                start = None
                for i, node in enumerate(nodes):
                    if caps[i] == math.inf and not all(
                        any(
                            a[r] > 0.0 and math.isfinite(limits[r])
                            for r in range(m)
                        )
                        for a in amounts[i]
                    ):
                        raise ValueError(
                            f"The greedy search could not reach target "
                            f"{target:g} within the budget, and node {node!r} "
                            "is not limited by it; give it a max_units."
                        )
            if method == "greedy":
                found = start
            else:
                bound = None if start is None else start[1]
                if problem.in_series:
                    found = self._series_redundancy(
                        nodes,
                        menus(bound),
                        limits,
                        problem.base,
                        evaluate,
                        primary,
                        target=target,
                        bound=bound,
                    )
                else:
                    found = redundancy_allocation.exact_min_cost(
                        evaluate,
                        menus(bound),
                        target,
                        bound=math.inf if bound is None else bound,
                        budget=limits,
                        primary=primary,
                    )
                if found is None:
                    found = start
            if found is None:
                raise ValueError(
                    f"target {target:g} is unreachable within the budget."
                )

        return self._redundancy_result(problem, found, method)

    def _redundancy_problem(
        self,
        costs,
        budget,
        target,
        minimise,
        t,
        max_units,
        required,
        strategy,
        switching_probability,
        mixing,
        method,
    ) -> SimpleNamespace:
        """Check the arguments of allocate_redundancy (and
        redundancy_front) and set up the problem the searches solve: the
        costed nodes, their kinds and designs, the limits and the exact
        system evaluation."""
        if method not in ("exact", "greedy"):
            raise ValueError(
                f"method must be 'exact' or 'greedy', got {method!r}."
            )
        if not isinstance(mixing, (bool, np.bool_)):
            raise ValueError(f"mixing must be True or False, got {mixing!r}.")
        mixing = bool(mixing)
        self._require_no_ccf_for_allocation()
        if not costs:
            raise ValueError("costs must name at least one node.")
        nodes = list(costs)
        for node in nodes:
            if node not in self.nodes:
                raise ValueError(
                    f"Node {node!r} in costs is not a component node of the "
                    "RBD (the input and output nodes cannot be duplicated)."
                )
            if node in self.repeated:
                raise ValueError(
                    f"Node {node!r} is a repeat of node "
                    f"{self.repeated[node]!r}; allocate redundancy to that "
                    "node instead."
                )
        options = self._redundancy_options(nodes, costs)
        labels, entries = [], []
        for node in nodes:
            if node in options:
                for option in options[node]:
                    labels.append(f"option {option.name!r} of node {node!r}")
                    entries.append(option.cost)
            else:
                labels.append(f"node {node!r}")
                entries.append(costs[node])
        resources, flat = self._redundancy_amounts(labels, entries)
        # What one copy of each kind uses, per node (one kind without
        # options).
        amounts = []
        for node in nodes:
            count = len(options.get(node, [None]))
            amounts.append(flat[:count])
            flat = flat[count:]
        limits = self._redundancy_limits(resources, budget)
        primary = self._redundancy_primary(resources, minimise, target)
        caps = redundancy_allocation.redundancy_caps(nodes, max_units)
        fewest = self._redundancy_required(nodes, required, caps)
        ways = self._redundancy_strategies(nodes, strategy)
        switching = self._redundancy_switching(
            nodes, switching_probability, ways, fewest
        )
        # The lifetime (or probability) model of one copy of each kind.
        models = [
            (
                [option.reliability for option in options[node]]
                if node in options
                else [self.reliabilities[node]]
            )
            for node in nodes
        ]
        for node, node_ways, node_models in zip(nodes, ways, models):
            if "cold" in node_ways and any(
                self._is_probability(model) or self._model_is_fixed(model)
                for model in node_models
            ):
                raise ValueError(
                    f"Cold standby needs lifetime models, but node {node!r} "
                    "(or one of its options) is a fixed probability."
                )
        m = len(resources)

        def limited_kind(amount) -> bool:
            return any(
                amount[r] > 0.0 and math.isfinite(limits[r]) for r in range(m)
            ) or (target is not None and amount[primary] > 0.0)

        for i, node in enumerate(nodes):
            if caps[i] == math.inf and not all(map(limited_kind, amounts[i])):
                raise ValueError(
                    f"Node {node!r} could be copied without limit: it uses "
                    "none of a limited resource. Give it a max_units, or "
                    "limit a resource it uses."
                )
        # The least one copy, and the required copies, of each node use of
        # each resource.
        cheapest = [
            [min(a[r] for a in node_amounts) for r in range(m)]
            for node_amounts in amounts
        ]
        least = [
            [n * amount for amount in row] for n, row in zip(fewest, cheapest)
        ]
        total_least = [math.fsum(row[r] for row in least) for r in range(m)]
        # Every limit given must afford one of each costed node.
        for r, limit in enumerate(limits):
            if isinstance(budget, dict):
                if resources[r] not in budget:
                    continue
            elif budget is None:
                continue
            need = total_least[r]
            if math.isfinite(limit) and limit >= need - 1e-9 * max(
                1.0, abs(need)
            ):
                continue
            what = (
                "one of each"
                if max(fewest) == 1
                else "the required copies of each"
            )
            if isinstance(budget, dict):
                raise ValueError(
                    f"budget[{resources[r]!r}] must be finite and at least "
                    f"{need:g}, the {resources[r]} of {what} costed node; "
                    f"got {budget[resources[r]]!r}."
                )
            raise ValueError(
                f"budget must be finite and at least {need:g}, the cost of "
                f"{what} costed node; got {budget!r}."
            )
        if t is not None and np.ndim(t) != 0:
            raise ValueError("t must be a single mission time.")
        if t is None:
            varying = [
                f"option {option.name!r} of node {node!r}"
                for node, node_options in options.items()
                for option in node_options
                if not self._is_probability(option.reliability)
                and not self._model_is_fixed(option.reliability)
            ]
            if self.is_time_varying or varying:
                raise ValueError(
                    "t (the mission time) is required: "
                    + (
                        "this RBD is time-varying"
                        if self.is_time_varying
                        else f"{varying[0]} is time-varying"
                    )
                    + ", so its reliability depends on when it is evaluated."
                )
            # A fixed-probability RBD does not depend on time (as in sf()).
            t = 1.0
        x = np.atleast_1d(np.asarray(t, dtype=float))

        base = {
            node: np.asarray(p, dtype=float)
            for node, p in self._base_node_probabilities(
                x, set(), set()
            ).items()
        }
        # Each node's kinds: (reliability of one copy, what it uses).
        kinds = []
        for node, node_amounts in zip(nodes, amounts):
            if node in options:
                reliabilities = [
                    self._option_reliability(node, option, x)
                    for option in options[node]
                ]
            else:
                reliabilities = [float(np.ravel(base[node])[0])]
            kinds.append(list(zip(reliabilities, node_amounts)))

        def cold(i: int) -> Callable[[tuple], float]:
            # The unreliability of a node's copies as cold standby (each
            # arrangement evaluated once).
            known: Dict[tuple, float] = {}

            def unreliability(counts: tuple) -> float:
                if sum(counts) == fewest[i]:
                    # No spares: the same as active copies (and exact).
                    return active(i)(counts)
                if counts not in known:
                    units = [
                        model
                        for model, n in zip(models[i], counts)
                        for _ in range(n)
                    ]
                    with warnings.catch_warnings():
                        # Scoring a candidate, not the user's model:
                        # its fit's deprecation is not theirs to act on.
                        warnings.filterwarnings(
                            "ignore", "This StandbyModel", FutureWarning
                        )
                        standby = StandbyModel(
                            units,
                            k=fewest[i],
                            switching_probability=switching[i],
                            seed=0,
                        )
                    known[counts] = float(np.ravel(standby.ff(x))[0])
                return known[counts]

            return unreliability

        def active(i: int) -> Callable[[tuple], float]:
            # The unreliability of a node's copies as active redundancy.
            return functools.partial(
                redundancy_allocation.active_unreliability,
                [1.0 - p for p, _ in kinds[i]],
                fewest=fewest[i],
            )

        # Each node's strategies: name -> the unreliability of its copies.
        strategies = [
            {
                way: (active(i) if way == "active" else cold(i))
                for way in node_ways
            }
            for i, node_ways in enumerate(ways)
        ]
        cache: Dict[tuple, float] = {}

        def evaluate(reliabilities: tuple) -> float:
            if reliabilities not in cache:
                probabilities = dict(base)
                for node, p in zip(nodes, reliabilities):
                    probabilities[node] = np.full_like(base[node], p)
                cache[reliabilities] = float(
                    np.ravel(self.system_probability(probabilities))[0]
                )
            return cache[reliabilities]

        def within(designs) -> bool:
            # Whether an allocation fits the budget.
            used = self._redundancy_totals(
                amounts, [d.counts for d in designs], m
            )
            return all(
                used[r] <= limit + 1e-9 * max(1.0, abs(limit))
                for r, limit in enumerate(limits)
            )

        def menus(bound=None) -> list:
            # Each node's designs worth considering within the budget (and
            # the bound on the primary resource).
            found = []
            for i, node in enumerate(nodes):
                most, spare = caps[i], []
                for r in range(m):
                    limit = limits[r]
                    if r == primary and bound is not None:
                        limit = min(limit, bound)
                    room = limit - (total_least[r] - least[i][r])
                    spare.append(room)
                    if math.isfinite(room) and cheapest[i][r] > 0.0:
                        most = min(
                            most,
                            math.floor(
                                (room + 1e-9 * max(1.0, abs(limit)))
                                / cheapest[i][r]
                            ),
                        )
                designs = []
                try:
                    for way, unreliability in strategies[i].items():
                        designs += redundancy_allocation.node_designs(
                            kinds[i],
                            most,
                            spare,
                            mixing,
                            primary,
                            fewest[i],
                            # Active copies: the built-in (and faster) form.
                            None if way == "active" else unreliability,
                            way,
                        )
                except ValueError as error:
                    raise ValueError(f"For node {node!r} {error}") from None
                designs = redundancy_allocation.best_designs(designs, primary)
                if not designs:
                    raise ValueError(
                        f"No design of node {node!r} fits within the budget."
                    )
                found.append(designs)
            return found

        # The greedy search's strategies (None: active copies only).
        greedy_strategies: List[Optional[Dict[str, Callable[[tuple], float]]]]
        greedy_strategies = [
            None if node_ways == ("active",) else strategies[i]
            for i, node_ways in enumerate(ways)
        ]
        # Whether every node is in series with the rest (alone a cut set).
        in_series = False
        if method == "exact":
            cut_sets = self.get_min_cut_sets()
            in_series = all(frozenset([node]) in cut_sets for node in nodes)
        return SimpleNamespace(
            nodes=nodes,
            options=options,
            resources=resources,
            amounts=amounts,
            m=m,
            limits=limits,
            primary=primary,
            caps=caps,
            fewest=fewest,
            ways=ways,
            kinds=kinds,
            base=base,
            evaluate=evaluate,
            within=within,
            menus=menus,
            greedy_strategies=greedy_strategies,
            in_series=in_series,
        )

    @staticmethod
    def _redundancy_result(problem, found, method) -> RedundancyAllocation:
        """The RedundancyAllocation of a search result."""
        reliability, _, designs = found
        counts = [d.counts for d in designs]
        used = NonRepairableRBD._redundancy_totals(
            problem.amounts, counts, problem.m
        )
        totals = dict(zip(problem.resources, used))
        return RedundancyAllocation(
            units={
                node: int(sum(c)) for node, c in zip(problem.nodes, counts)
            },
            reliability=reliability,
            cost=totals[problem.resources[problem.primary]],
            method=method,
            resources=totals,
            mix={
                node: {
                    option.name: int(k)
                    for option, k in zip(problem.options[node], c)
                    if k
                }
                for node, c in zip(problem.nodes, counts)
                if node in problem.options
            },
            strategy={
                node: d.strategy for node, d in zip(problem.nodes, designs)
            },
        )

    @staticmethod
    def _redundancy_totals(amounts, counts, m) -> List[float]:
        """The total of each resource an allocation uses."""
        return [
            math.fsum(
                a[r] * k
                for kinds, c in zip(amounts, counts)
                for a, k in zip(kinds, c)
                if k
            )
            for r in range(m)
        ]

    @staticmethod
    def _redundancy_required(nodes, required, caps) -> list:
        """The number of copies of each costed node that must work."""
        if isinstance(required, dict):
            unknown = set(required) - set(nodes)
            if unknown:
                raise ValueError(
                    "required names node(s) not in costs: "
                    f"{sorted(map(str, unknown))}."
                )
            given = {node: required.get(node, 1) for node in nodes}
        else:
            given = {
                node: 1 if required is None else required for node in nodes
            }
        fewest = []
        for node, cap in zip(nodes, caps):
            k = given[node]
            if (
                isinstance(k, (bool, np.bool_))
                or not isinstance(k, (int, np.integer))
                or k < 1
            ):
                raise ValueError(
                    f"required for node {node!r} must be an integer of at "
                    f"least 1, got {k!r}."
                )
            if k > cap:
                raise ValueError(
                    f"max_units for node {node!r} is {cap}, fewer than the "
                    f"{k} copies it requires."
                )
            fewest.append(int(k))
        return fewest

    @staticmethod
    def _redundancy_strategies(nodes, strategy) -> list:
        """The redundancy strategies each costed node may use."""
        choices = {
            "active": ("active",),
            "cold": ("cold",),
            "choose": ("active", "cold"),
        }
        if isinstance(strategy, dict):
            unknown = set(strategy) - set(nodes)
            if unknown:
                raise ValueError(
                    "strategy names node(s) not in costs: "
                    f"{sorted(map(str, unknown))}."
                )
            given = {node: strategy.get(node, "active") for node in nodes}
        else:
            given = {node: strategy for node in nodes}
        for node in nodes:
            if given[node] not in choices:
                raise ValueError(
                    f"strategy for node {node!r} must be 'active', 'cold' or "
                    f"'choose', got {given[node]!r}."
                )
        return [choices[given[node]] for node in nodes]

    @staticmethod
    def _redundancy_switching(nodes, switching_probability, ways, fewest):
        """The probability that switching onto each costed node's next cold
        spare succeeds."""
        cold = [node for node, way in zip(nodes, ways) if "cold" in way]
        if isinstance(switching_probability, dict):
            for node in switching_probability:
                if node not in cold:
                    raise ValueError(
                        "switching_probability applies to cold standby, and "
                        f"node {node!r} is not a cold standby node in costs."
                    )
            given = {
                node: switching_probability.get(node, 1.0) for node in nodes
            }
        else:
            if float(switching_probability) != 1.0 and not cold:
                raise ValueError(
                    "switching_probability applies to cold standby, which no "
                    "node uses (see strategy)."
                )
            given = {node: switching_probability for node in nodes}
        out = []
        for node, way, k in zip(nodes, ways, fewest):
            rho = float(given[node])
            if not 0.0 <= rho <= 1.0:
                raise ValueError(
                    f"switching_probability for node {node!r} must be in "
                    f"[0, 1], got {given[node]!r}."
                )
            if rho < 1.0 and "cold" in way and k > 1:
                raise ValueError(
                    "Imperfect switching is supported for cold standby with "
                    f"one unit required; node {node!r} requires {k}."
                )
            out.append(rho)
        return out

    @staticmethod
    def _redundancy_options(nodes, costs) -> dict:
        """The component options of the nodes given a list of them."""
        options = {}
        for node in nodes:
            entry = costs[node]
            if not isinstance(entry, (list, tuple)):
                continue
            if not entry:
                raise ValueError(
                    f"Node {node!r} has an empty list of options; give at "
                    "least one ComponentOption."
                )
            if not all(isinstance(o, ComponentOption) for o in entry):
                raise ValueError(
                    f"The options of node {node!r} must be ComponentOption "
                    "instances."
                )
            names = [o.name for o in entry]
            if len(set(names)) < len(names):
                raise ValueError(
                    f"The options of node {node!r} must have distinct "
                    f"names, got {names}."
                )
            for option in entry:
                value = option.reliability
                if NonRepairableRBD._is_probability(value):
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(
                            f"The reliability of option {option.name!r} of "
                            f"node {node!r} must be in [0, 1], got {value!r}."
                        )
                elif not callable(getattr(value, "sf", None)):
                    raise ValueError(
                        f"The reliability of option {option.name!r} of node "
                        f"{node!r} must be a model with an sf method or a "
                        f"probability, got {value!r}."
                    )
            options[node] = list(entry)
        return options

    @staticmethod
    def _is_probability(value) -> bool:
        return isinstance(
            value, (int, float, np.integer, np.floating)
        ) and not isinstance(value, (bool, np.bool_))

    def _option_reliability(self, node, option, x) -> float:
        """The reliability of one copy of a component option at ``x``."""
        value = option.reliability
        if self._is_probability(value):
            return float(value)
        p = float(np.ravel(value.sf(x))[0])
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"The reliability of option {option.name!r} of node "
                f"{node!r} must be in [0, 1], got {p!r}."
            )
        return p

    @staticmethod
    def _redundancy_amounts(labels, entries) -> tuple:
        """The resources, and what one copy uses of them for each entry of
        costs (a node, or one of its options)."""
        if all(isinstance(entry, dict) for entry in entries):
            resources = list(
                dict.fromkeys(r for entry in entries for r in entry)
            )
            if not resources:
                raise ValueError("costs must name at least one resource.")
            amounts = []
            for label, entry in zip(labels, entries):
                missing = [r for r in resources if r not in entry]
                if missing:
                    raise ValueError(
                        f"No amount of {missing} is given for {label}; "
                        "every node in costs must name the same resources."
                    )
                vector = []
                for r in resources:
                    amount = float(entry[r])
                    if not np.isfinite(amount) or amount < 0.0:
                        raise ValueError(
                            f"The {r} of {label} must be finite and "
                            f"non-negative, got {entry[r]!r}."
                        )
                    vector.append(amount)
                amounts.append(tuple(vector))
            return resources, amounts
        if any(isinstance(entry, dict) for entry in entries):
            raise ValueError(
                "costs must give every node (and option) a number (one "
                "resource) or every one a dict of resources, not a mixture."
            )
        amounts = []
        for label, entry in zip(labels, entries):
            cost = float(entry)
            if not np.isfinite(cost) or cost <= 0.0:
                raise ValueError(
                    f"The cost of {label} must be finite and positive, "
                    f"got {entry!r}."
                )
            amounts.append((cost,))
        return ["cost"], amounts

    @staticmethod
    def _redundancy_limits(resources, budget) -> tuple:
        """One limit per resource (inf where it is not limited)."""
        if budget is None:
            return (math.inf,) * len(resources)
        if isinstance(budget, dict):
            if not budget:
                raise ValueError("budget must limit at least one resource.")
            unknown = [r for r in budget if r not in resources]
            if unknown:
                raise ValueError(
                    f"budget names resource(s) {unknown} that costs does "
                    f"not use; the resources are {resources}."
                )
            return tuple(
                float(budget[r]) if r in budget else math.inf
                for r in resources
            )
        if len(resources) > 1:
            raise ValueError(
                "With several resources, budget must be a dict of limits, "
                f"e.g. {{{resources[0]!r}: ..., {resources[1]!r}: ...}}."
            )
        return (float(budget),)

    @staticmethod
    def _redundancy_primary(resources, minimise, target) -> int:
        """The index of the resource a target minimises (and that breaks
        ties): ``minimise``, else "cost", else the only (or first) one."""
        if minimise is not None:
            if target is None:
                raise ValueError(
                    "minimise names the resource a target minimises; give a "
                    "target too."
                )
            if minimise not in resources:
                raise ValueError(
                    f"minimise must name one of the resources {resources}, "
                    f"got {minimise!r}."
                )
            return resources.index(minimise)
        if "cost" in resources:
            return resources.index("cost")
        if len(resources) > 1 and target is not None:
            raise ValueError(
                "With several resources and none called 'cost', minimise "
                f"must name the one to minimise, from {resources}."
            )
        return 0

    def _series_redundancy(
        self,
        nodes,
        menus,
        limits,
        base,
        evaluate,
        primary,
        target=None,
        bound=None,
    ):
        """The exact optimum when every costed node is in series with the
        rest of the system, by the dynamic program of
        ``redundancy_allocation.series_front`` over each node's designs."""
        front = redundancy_allocation.series_front(
            self._design_choices(menus),
            budget=limits,
            bound=bound,
            primary=primary,
        )
        if not front:
            return None

        def chosen(picks):
            designs = tuple(menu[j] for menu, j in zip(menus, picks))
            return tuple(d.reliability for d in designs), designs

        if target is None:
            best = max(value for _, value, _ in front)
            ties = [f for f in front if f[1] >= best - 1e-12]
            use, _, picks = min(ties, key=lambda f: (f[0][primary], -f[1]))
            reliabilities, designs = chosen(picks)
            return evaluate(reliabilities), use[primary], designs
        # The system is the costed nodes times the rest, with the costed
        # nodes perfect.
        rest = dict(base)
        for node in nodes:
            rest[node] = np.ones_like(base[node])
        scale = float(np.ravel(self.system_probability(rest))[0])
        for use, value, picks in front:
            if scale * math.exp(value) < target * (1.0 - 1e-9):
                continue
            reliabilities, designs = chosen(picks)
            reliability = evaluate(reliabilities)
            if reliability >= target:
                return reliability, use[primary], designs
        return None

    @staticmethod
    def _design_choices(menus) -> list:
        """Each node's designs as (use, log-reliability) alternatives for
        the dynamic program."""
        return [
            [
                (
                    d.use,
                    (
                        math.log1p(-d.unreliability)
                        if d.unreliability < 1.0
                        else -math.inf
                    ),
                )
                for d in menu
            ]
            for menu in menus
        ]

    def redundancy_front(
        self,
        costs: Dict[
            Hashable,
            Union[float, Dict[Hashable, float], Sequence[ComponentOption]],
        ],
        *,
        budget: Union[float, Dict[Hashable, float], None] = None,
        t: Optional[float] = None,
        max_units: Union[int, Dict[Hashable, int], None] = None,
        required: Union[int, Dict[Hashable, int], None] = None,
        strategy: Union[str, Dict[Hashable, str]] = "active",
        switching_probability: Union[float, Dict[Hashable, float]] = 1.0,
        mixing: bool = True,
    ) -> List[RedundancyAllocation]:
        """The whole cost-reliability trade-off of redundancy allocation.

        Where ``allocate_redundancy`` returns the one best design for a
        budget or a target, this returns every design that no other beats
        by using no more of every resource while being at least as reliable
        (the Pareto front of multi-objective redundancy allocation, e.g.
        Taboada et al., 2007). With one resource
        it is the best reliability for each level of spending, so a design
        can be chosen by looking at the whole curve; the best design within
        any budget, and the cheapest meeting any target, are on it. With
        several resources it is the best reliability for each combination
        of them.

        The front is exact: when every costed node is in series with the
        rest of the system it comes from the dynamic program behind
        ``allocate_redundancy``, and otherwise from evaluating every
        combination of node designs within the budget (giving up with an
        explanatory error beyond 500,000 of them).

        Parameters
        ----------
        costs : dict
            What one copy of each costed node uses, as for
            ``allocate_redundancy``: a number, a dict of resources, or a
            list of [`ComponentOption`][repyability.ComponentOption].
        budget : float or dict, optional
            Limits on the resources, as for ``allocate_redundancy``. The
            budget or ``max_units`` must bound every node's copies.
        t : float, optional
            The mission time, as for ``allocate_redundancy``.
        max_units : int or dict, optional
            The most copies of each node, as for ``allocate_redundancy``.
        required : int or dict, optional
            The copies of each node that must work, as for
            ``allocate_redundancy``.
        strategy : str or dict, optional
            ``"active"``, ``"cold"`` or ``"choose"``, as for
            ``allocate_redundancy``.
        switching_probability : float or dict, optional
            For cold standby, as for ``allocate_redundancy``.
        mixing : bool, optional
            Whether a node's copies may mix types, as for
            ``allocate_redundancy``.

        Returns
        -------
        list of RedundancyAllocation
            One per non-dominated design, by increasing ``cost`` (the total
            of ``"cost"``, or of the first resource), the most reliable
            first among equals. With one resource the reliability rises
            along the list.

        Raises
        ------
        ValueError
            On invalid input, as for ``allocate_redundancy``, if a node's
            copies are not bounded by the budget or ``max_units``, or if
            there are too many designs to evaluate.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        References
        ----------
        H. A. Taboada, F. Baheranwala, D. W. Coit and N. Wattanapongsakorn,
        "Practical solutions for multi-objective optimization: an
        application to system reliability design problems", Reliability
        Engineering & System Safety, 92(3), 314-322, 2007.

        Examples
        --------
        Two components in series, 90% and 80% reliable, one cost unit each,
        spending up to 5:

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": FixedEventProbability.from_params(0.1),
        ...         "b": FixedEventProbability.from_params(0.2),
        ...     },
        ... )
        >>> front = rbd.redundancy_front({"a": 1.0, "b": 1.0}, budget=5)
        >>> for design in front:
        ...     print(design.cost, design.units, round(design.reliability, 4))
        2.0 {'a': 1, 'b': 1} 0.72
        3.0 {'a': 1, 'b': 2} 0.864
        4.0 {'a': 2, 'b': 2} 0.9504
        5.0 {'a': 2, 'b': 3} 0.9821
        """
        problem = self._redundancy_problem(
            costs,
            budget,
            None,
            None,
            t,
            max_units,
            required,
            strategy,
            switching_probability,
            mixing,
            "exact",
        )
        menus = problem.menus()
        if problem.in_series:
            found = []
            for use, _, picks in redundancy_allocation.series_front(
                self._design_choices(menus),
                budget=problem.limits,
                primary=problem.primary,
            ):
                designs = tuple(menu[j] for menu, j in zip(menus, picks))
                reliability = problem.evaluate(
                    tuple(d.reliability for d in designs)
                )
                found.append((use, reliability, designs))
            # The same front, ordered by the exact system reliabilities.
            front = redundancy_allocation._sorted_front(found, problem.primary)
        else:
            front = redundancy_allocation.exact_front(
                problem.evaluate, menus, problem.limits, problem.primary
            )
        return [
            self._redundancy_result(problem, allocation, "exact")
            for allocation in front
        ]

    def allocate_reliability_redundancy(
        self,
        uses: Dict[
            Hashable,
            Callable[[float, int], Union[float, Dict[Hashable, float]]],
        ],
        *,
        budget: Union[float, Dict[Hashable, float]],
        bounds: Union[
            Tuple[float, float], Dict[Hashable, Tuple[float, float]]
        ],
        t: Optional[float] = None,
        max_units: Union[int, Dict[Hashable, int], None] = None,
    ) -> ReliabilityRedundancyAllocation:
        """Choose each node's component reliability and number of copies
        together: the reliability-redundancy allocation problem (RRAP).

        Joins reliability allocation (what each component must achieve) to
        redundancy allocation (how many copies to fit): for each node in
        ``uses``, pick a component reliability ``r`` within its ``bounds``
        and a number of active copies ``n``, giving the node reliability
        ``1 - (1 - r) ** n``, to maximise system reliability within the
        budget. What the copies use is a function of both, typically rising
        steeply as ``r`` approaches 1 (Tillman, Hwang & Kuo, 1977). The
        node's own model in the RBD is not used; the other nodes stay as
        they are, evaluated at the mission time ``t``. Any structure works:
        the system reliability and its gradient come from the exact engine.

        It is solved exactly over the copies by branch and bound: each copy
        vector that fits the budget at the lowest reliabilities is bounded
        above by the system reliability with every node at the highest
        reliability it could afford alone, and the vectors are solved in
        decreasing order of that bound -- each a continuous problem for the
        reliabilities, solved by SLSQP with the exact gradient from the
        lowest reliabilities -- until no bound beats the best found. The
        continuous problem is solved to a local optimum, which for a series
        system with costs convex in the reliabilities is the global one. On
        the classic benchmarks (series, series-parallel, bridge and
        overspeed protection systems) it returns the best published
        solutions.

        Parameters
        ----------
        uses : dict
            ``{node: function}``: ``function(r, n)`` gives what ``n`` copies
            of component reliability ``r`` use, a number (their cost) or a
            dict of resource -> amount, naming the same resources for every
            node. It must not decrease as ``r`` or ``n`` grows.
        budget : float or dict
            The limit on the one resource (a number), or on each resource a
            dict names (resources it leaves out are not limited).
        bounds : tuple or dict
            The lowest and highest component reliability, ``(low, high)``
            in [0, 1], for every node (a tuple) or for each node (a dict
            naming every node in ``uses``). The lowest must be affordable.
        t : float, optional
            The mission time at which the other nodes' reliabilities are
            evaluated. Required for a time-varying RBD.
        max_units : int or dict, optional
            The most copies of every node (an int) or of particular nodes
            (a dict). By default the budget bounds them.

        Returns
        -------
        ReliabilityRedundancyAllocation
            The ``units`` and ``component_reliability`` of each node, the
            system ``reliability``, and the total of each resource used
            (``resources``; ``cost`` is the total of ``"cost"``, or of the
            first resource).

        Raises
        ------
        ValueError
            On invalid input: empty ``uses``, a node that is not a component
            node or is a repeated node, a use that is not a function or
            returns something other than finite, non-negative amounts of the
            same resources for every node, a ``budget`` naming an unknown
            resource, invalid ``bounds`` or ``max_units``, ``t`` missing for
            a time-varying RBD, a budget that cannot afford one copy of each
            node at its lowest reliability, a node whose copies nothing
            bounds, or more than 500,000 copy vectors to consider.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        References
        ----------
        F. A. Tillman, C.-L. Hwang and W. Kuo, "Determining component
        reliability and redundancy for optimum system reliability", IEEE
        Transactions on Reliability, 26(3), 162-165, 1977.

        Examples
        --------
        Two components in series. A component's reliability ``r`` is
        chosen in [0.5, 0.999]; each copy costs 10 to fit, plus
        ``(-1 / log(r)) ** 1.5``, which rises steeply as ``r`` nears 1.
        Within a budget of 100, three copies of each at ``r = 0.754`` is
        best:

        >>> import math
        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": FixedEventProbability.from_params(0.1),
        ...         "b": FixedEventProbability.from_params(0.2),
        ...     },
        ... )
        >>> def cost(r, n):
        ...     return n * (10 + (-1 / math.log(r)) ** 1.5)
        >>> best = rbd.allocate_reliability_redundancy(
        ...     {"a": cost, "b": cost}, budget=100, bounds=(0.5, 0.999)
        ... )
        >>> best.units
        {'a': 3, 'b': 3}
        >>> [round(r, 3) for r in best.component_reliability.values()]
        [0.754, 0.754]
        >>> round(best.reliability, 4)
        0.9705
        """
        self._require_no_ccf_for_allocation(reliabilities=True)
        if not uses:
            raise ValueError("uses must name at least one node.")
        nodes = list(uses)
        for node in nodes:
            if node not in self.nodes:
                raise ValueError(
                    f"Node {node!r} in uses is not a component node of the "
                    "RBD (the input and output nodes cannot be duplicated)."
                )
            if node in self.repeated:
                raise ValueError(
                    f"Node {node!r} is a repeat of node "
                    f"{self.repeated[node]!r}; allocate to that node instead."
                )
            if not callable(uses[node]):
                raise ValueError(
                    f"uses[{node!r}] must be a function of (r, n), got "
                    f"{uses[node]!r}."
                )
        if isinstance(bounds, dict):
            missing = [node for node in nodes if node not in bounds]
            unknown = [node for node in bounds if node not in uses]
            if missing or unknown:
                raise ValueError(
                    "bounds must give (low, high) for exactly the nodes in "
                    f"uses; missing {missing}, unknown {unknown}."
                )
            ranges = [bounds[node] for node in nodes]
        else:
            ranges = [bounds] * len(nodes)
        limits_of = []
        for node, pair in zip(nodes, ranges):
            try:
                low, high = (float(value) for value in pair)
            except (TypeError, ValueError):
                raise ValueError(
                    f"The bounds of node {node!r} must be a pair (low, "
                    f"high), got {pair!r}."
                ) from None
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(
                    f"The bounds of node {node!r} must satisfy 0 <= low <= "
                    f"high <= 1, got {pair!r}."
                )
            limits_of.append((low, high))
        # The resources, from what one copy of each node uses at its lowest
        # reliability.
        labels = [f"node {node!r}" for node in nodes]
        entries = [
            uses[node](low, 1) for node, (low, _) in zip(nodes, limits_of)
        ]
        if all(isinstance(entry, dict) for entry in entries):
            resources, _ = self._redundancy_amounts(labels, entries)
        else:
            resources = ["cost"]
            for label, entry in zip(labels, entries):
                if isinstance(entry, dict) or not np.isfinite(float(entry)):
                    raise ValueError(
                        f"The use of {label} must be a finite number or a "
                        f"dict of resources for every node, got {entry!r}."
                    )
        m = len(resources)

        def vector(node):
            # The node's use as a tuple, one amount per resource.
            function = uses[node]

            def use(r: float, n: int) -> Tuple[float, ...]:
                amount = function(r, n)
                if isinstance(amount, dict):
                    values = tuple(float(amount[x]) for x in resources)
                else:
                    values = (float(amount),)
                if len(values) != m or not all(
                    np.isfinite(v) and v >= 0.0 for v in values
                ):
                    raise ValueError(
                        f"The use of node {node!r} at r={r!r}, n={n!r} must "
                        "be finite and non-negative amounts of the resources "
                        f"{resources}, got {amount!r}."
                    )
                return values

            return use

        use_vectors = [vector(node) for node in nodes]
        limits = self._redundancy_limits(resources, budget)
        caps = redundancy_allocation.redundancy_caps(nodes, max_units)
        for i, node in enumerate(nodes):
            if caps[i] == math.inf:
                # Unbounded unless a limited resource grows with the copies.
                others = [
                    math.fsum(
                        use_vectors[j](limits_of[j][0], 1)[r]
                        for j in range(len(nodes))
                        if j != i
                    )
                    for r in range(m)
                ]
                n = 1
                while all(
                    others[r] + amount <= limits[r]
                    for r, amount in enumerate(
                        use_vectors[i](limits_of[i][0], n)
                    )
                ):
                    n += 1
                    if n > 10_000:
                        raise ValueError(
                            f"Node {node!r} could be copied without limit: "
                            "its copies do not use enough of a limited "
                            "resource. Give it a max_units."
                        )
        if t is not None and np.ndim(t) != 0:
            raise ValueError("t must be a single mission time.")
        if t is None:
            if self.is_time_varying:
                raise ValueError(
                    "t (the mission time) is required: this RBD is "
                    "time-varying, so its reliability depends on when it is "
                    "evaluated."
                )
            t = 1.0
        x = np.atleast_1d(np.asarray(t, dtype=float))
        base = {
            node: float(np.ravel(p)[0])
            for node, p in self._base_node_probabilities(
                x, set(), set()
            ).items()
        }
        structure = self._decomposition()

        def system(reliabilities):
            p = dict(base)
            q = {node: 1.0 - value for node, value in base.items()}
            for node, value in zip(nodes, reliabilities):
                p[node], q[node] = value, 1.0 - value
            value, _, gradient = structure.value_and_gradient(p, q)
            return value, tuple(gradient.get(node, 0.0) for node in nodes)

        reliability, units, components = (
            redundancy_allocation.reliability_redundancy(
                system, use_vectors, limits, limits_of, caps
            )
        )
        amounts = [
            use_vectors[i](components[i], units[i]) for i in range(len(nodes))
        ]
        totals = {
            resource: math.fsum(a[r] for a in amounts)
            for r, resource in enumerate(resources)
        }
        primary = resources.index("cost") if "cost" in resources else 0
        return ReliabilityRedundancyAllocation(
            units=dict(zip(nodes, map(int, units))),
            component_reliability=dict(zip(nodes, components)),
            reliability=reliability,
            cost=totals[resources[primary]],
            resources=totals,
        )

    def unreliability(self, x: Optional[ArrayLike] = None, *args, **kwargs):
        """System unreliability at time/s ``x``; the same as ``ff``.

        ``1 - sf(x)``, the probability that the system has failed by ``x``,
        worked out in its own right, so that a small one keeps its
        precision (see [`ff`][repyability.NonRepairableRBD.ff]).

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        *args
            Further positional arguments of ``ff`` (``working_nodes``,
            ``broken_nodes``, ``method``).
        **kwargs
            Keyword arguments of ``ff``.

        Returns
        -------
        float or numpy.ndarray
            The system unreliability: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.
        """
        return self.ff(x, *args, **kwargs)

    def reliability(self, x: Optional[ArrayLike] = None, *args, **kwargs):
        """System reliability at time/s ``x``; the same as ``sf``.

        The probability that the system is still working at ``x`` (see
        [`sf`][repyability.NonRepairableRBD.sf]).

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        *args
            Further positional arguments of ``sf`` (``working_nodes``,
            ``broken_nodes``, ``method``).
        **kwargs
            Keyword arguments of ``sf``.

        Returns
        -------
        float or numpy.ndarray
            The system reliability: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.
        """
        return self.sf(x, *args, **kwargs)

    def cs(self, x: ArrayLike, X: ArrayLike, *args, **kwargs) -> np.ndarray:
        """Conditional survival of the system.

        The probability that the system survives a *further* ``x`` given it
        has survived to age ``X``: ``R(x | X) = sf(X + x) / sf(X)``. The
        whole system is conditioned on having survived, not each component
        on its own age; for per-component ages use
        [`sf_given_state`][repyability.NonRepairableRBD.sf_given_state].

        The result is clipped to [0, 1] and is 0 where ``sf(X)`` is 0.
        ``x`` and ``X`` broadcast against each other.

        Parameters
        ----------
        x : array_like
            The further duration/s at which conditional survival is evaluated.
        X : array_like
            The age/s the system is known to have survived to.
        *args
            Further positional arguments of ``sf`` (``working_nodes``,
            ``broken_nodes``, ``method``).
        **kwargs
            Keyword arguments of ``sf``.

        Returns
        -------
        float or numpy.ndarray
            The conditional survival probability: a float if both ``x`` and
            ``X`` are scalars, otherwise an array.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        A component 50 hours old survives the next 10 hours with a higher
        probability than it had of surviving from new to 60 hours:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> round(rbd.cs(10, 50), 4)
        0.8958
        >>> round(rbd.sf(60), 4)
        0.6977
        """
        return conditional_survival(self, x, X, *args, **kwargs)

    @check_x
    @check_x
    def Hf(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> np.ndarray:
        """System cumulative hazard ``H(x) = -ln R(x)`` at time/s ``x``.

        Exact given the exact system reliability ``R(x)`` from
        [`sf`][repyability.NonRepairableRBD.sf] (so it honours
        common-cause groups); +inf wherever the system reliability has
        reached zero. While the system is more likely to work than not, it
        is ``-log1p(-F(x))``, from the unreliability ``F`` of
        [`ff`][repyability.NonRepairableRBD.ff], so that a small one keeps
        its precision.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working, by default none (as for ``sf``).
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed, by default none (as for ``sf``).
        method : str, optional
            ``"p"`` or ``"c"``, as for ``sf``.

        Returns
        -------
        float or numpy.ndarray
            The cumulative hazard: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        For a single exponential component ``H(x)`` is the failure rate
        times ``x``:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> round(rbd.Hf(50), 6)
        0.5
        """
        sf, ff = self._sf_and_ff(x, working_nodes, broken_nodes, method)
        with np.errstate(divide="ignore"):
            return np.where(
                ff < 0.5, -np.log1p(-np.minimum(ff, 0.5)), -np.log(sf)
            )

    @check_x
    def df(
        self,
        x: Optional[ArrayLike] = None,
        dx: float = 1e-6,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> np.ndarray:
        """System failure density ``f(x) = -dR/dx`` at time/s ``x``.

        Computed by a central finite difference of the system reliability
        from [`sf`][repyability.NonRepairableRBD.sf] (so it honours
        common-cause groups), with step ``h = dx * max(|x|, 1)``: relative
        to ``x`` for ``|x| >= 1``, absolute below. While the system is more
        likely to work than not, the difference is taken of its
        unreliability from [`ff`][repyability.NonRepairableRBD.ff]
        instead, which keeps its precision where the reliability's change
        would be lost to rounding. The lower point is clipped at 0, so the
        step never crosses into negative time (the difference is one-sided
        near 0), and negative results (numerical noise) are clipped to 0.
        It assumes ``R`` is smooth near ``x``, which does not hold for
        step-function node models such as a Kaplan-Meier fit.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD (whose density is 0).
        dx : float, optional
            Relative finite-difference step, by default 1e-6.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working, by default none (as for ``sf``).
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed, by default none (as for ``sf``).
        method : str, optional
            ``"p"`` or ``"c"``, as for ``sf``.

        Returns
        -------
        float or numpy.ndarray
            The failure density: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        For a single exponential component ``f(x) = rate * exp(-rate * x)``:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> round(rbd.df(50), 6)
        0.006065
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        h = dx * np.maximum(np.abs(x), 1.0)
        x_hi = x + h
        x_lo = np.maximum(x - h, 0.0)
        sf_lo, ff_lo = self._sf_and_ff(
            x_lo, working_nodes, broken_nodes, method
        )
        sf_hi, ff_hi = self._sf_and_ff(
            x_hi, working_nodes, broken_nodes, method
        )
        # The change in whichever of the two is the smaller keeps its
        # precision; that in the other would be lost to rounding.
        change = np.where(ff_hi <= 0.5, ff_hi - ff_lo, sf_lo - sf_hi)
        return np.clip(change / (x_hi - x_lo), 0.0, None)

    @check_x
    def hf(
        self, x: Optional[ArrayLike] = None, dx: float = 1e-6, **kwargs
    ) -> np.ndarray:
        """System hazard rate ``h(x) = f(x) / R(x)`` at time/s ``x``.

        The numerical failure density from
        [`df`][repyability.NonRepairableRBD.df] over the exact system
        reliability from [`sf`][repyability.NonRepairableRBD.sf] (both
        honour common-cause groups). It is +inf wherever the system
        reliability has reached zero.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        dx : float, optional
            Relative finite-difference step passed to ``df``, by default
            1e-6.
        **kwargs
            Keyword arguments of ``sf`` (``working_nodes``,
            ``broken_nodes``, ``method``).

        Returns
        -------
        float or numpy.ndarray
            The hazard rate: a float for scalar ``x``, an array for array
            ``x``.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        A single exponential component has a constant hazard rate:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> round(rbd.hf(50), 6)
        0.01
        """
        sf = np.asarray(self.sf(x, **kwargs), dtype=float)
        density = self.df(x, dx=dx, **kwargs)
        return np.divide(
            density,
            sf,
            out=np.full_like(density, np.inf),
            where=sf > 0,
        )

    @check_x
    def node_sf(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Dict[Any, Union[float, np.ndarray]]:
        """Reliability of each node at time/s ``x``.

        Each node model's own ``sf(x)``, including the input and output
        nodes (always 1). A repeated node appears only under the node it
        repeats.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        *args
            Accepted but ignored, so ``working_nodes``/``broken_nodes`` do
            not apply here.
        **kwargs
            Accepted but ignored.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: reliability}``: floats for scalar ``x``, arrays for
            array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> {k: round(v, 4) for k, v in sorted(rbd.node_sf(50).items())}
        {'c': 0.7788, 's': 1.0, 't': 1.0}
        """
        node_sf: Dict[Any, Union[float, np.ndarray]] = {}
        for node_name, node in self.reliabilities.items():
            node_sf[node_name] = node.sf(x)
        return node_sf

    @check_x
    def node_ff(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Dict[Any, Union[float, np.ndarray]]:
        """Unreliability of each node at time/s ``x``.

        Each node model's own ``ff(x)``, including the input and output
        nodes (always 0). A repeated node appears only under the node it
        repeats.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        *args
            Accepted but ignored, so ``working_nodes``/``broken_nodes`` do
            not apply here.
        **kwargs
            Accepted but ignored.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: unreliability}``: floats for scalar ``x``, arrays for
            array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD.
        """
        node_ff: Dict[Any, Union[float, np.ndarray]] = {}
        for node_name, node in self.reliabilities.items():
            node_ff[node_name] = node.ff(x)

        return node_ff

    @property
    def is_fixed(self) -> bool:
        """Whether the system reliability is constant in time.

        True when every node is a fixed (per-demand) probability, such as a
        ``surpyval.FixedEventProbability``, a
        [`RepeatedNode`][repyability.RepeatedNode] of one, or a nested
        fixed-probability RBD; perfect nodes do not count either way. Any
        other model (a lifetime distribution, a non-parametric fit, a
        standby or load-sharing node, ...) makes the RBD time-varying.

        For a fixed-probability RBD the time argument ``x`` of the
        reliability and importance methods may be omitted, and the
        methods that solve for a time (``time_to_reliability``,
        ``bx_life``, ``remaining_life``) raise a ValueError. Decided when
        the RBD is built.
        """
        return self._fixed_probs

    @staticmethod
    def _model_is_fixed(model) -> bool:
        """Whether a node model's reliability does not vary with time."""
        if isinstance(model, NonParametric):
            return False
        if isinstance(model, NonRepairableRBD):
            return model.is_fixed
        if model is PerfectReliability or model is PerfectUnreliability:
            return True
        if isinstance(model, (StandbyModel, LoadSharingModel)):
            return False
        if isinstance(model, RepeatedNode):
            return is_fixed_probability(model.model)
        return is_fixed_probability(model)

    @property
    def is_time_varying(self) -> bool:
        """Whether the system reliability varies with time.

        The complement of
        [`is_fixed`][repyability.NonRepairableRBD.is_fixed]: True when at
        least one node's reliability depends on time, in which case the
        time argument ``x`` must be given.
        """
        return not self._fixed_probs

    def _node_is_analytic(self, model) -> bool:
        """Whether a node's reliability is computed without simulation:
        exactly or numerically (see ``repyability.rbd.routes``)."""
        from repyability.rbd.routes import SIMULATED, model_route

        return model_route(model)[0] != SIMULATED

    def get_non_analytic_nodes(self) -> dict[Any, str]:
        """The nodes whose reliability is simulated.

        A node's reliability is simulated when it is a Kaplan-Meier fit to
        simulated lifetimes: a [`StandbyModel`][repyability.StandbyModel]
        or [`LoadSharingModel`][repyability.LoadSharingModel] with no
        closed form or convolution (see their ``is_simulated``), or a
        repeated node or nested RBD of one. A closed form or a numerical
        convolution is not simulated. ``analysis_routes`` says how each
        analysis of the RBD is computed.

        Returns
        -------
        dict[Any, str]
            ``{node: type name of its model}`` for every node whose
            reliability is simulated, e.g. ``{"a": "StandbyModel"}``. Empty
            if none is.
        """
        non_analytic: dict[Any, str] = {}
        for node_name, model in self.reliabilities.items():
            if not self._node_is_analytic(model):
                non_analytic[node_name] = type(model).__name__
        return non_analytic

    def is_analytically_solvable(self) -> bool:
        """Whether every node's reliability is computed without simulation.

        The exact system reliability (``sf`` and the methods built on it)
        is only as accurate as each node's own ``sf(t)``. That is a closed
        form or data for surpyval distributions, for
        [`RegressionNode`][repyability.RegressionNode] and the perfect
        nodes; a closed form or a numerical convolution for most standby
        and load-sharing arrangements; and a Kaplan-Meier fit to simulated
        lifetimes for the rest (see ``get_non_analytic_nodes``), which
        makes this False. ``sf`` still returns a value either way, carrying
        those nodes' Monte-Carlo error. ``analysis_routes`` says how each
        analysis is computed, including those that always simulate
        (``random``, ``mean``).

        The result is also stored at construction in
        ``structure_check["is_analytically_solvable"]``.

        Returns
        -------
        bool
            True if no node's reliability is simulated; False otherwise
            (``get_non_analytic_nodes`` lists which).

        Examples
        --------
        One cold spare for one unit is a numerical convolution; two units
        of three needed working, with one cold spare, is simulated:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, StandbyModel
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> pair = NonRepairableRBD(
        ...     [("s", "a"), ("a", "t")], {"a": StandbyModel([unit, unit])}
        ... )
        >>> pair.is_analytically_solvable()
        True
        >>> trio = NonRepairableRBD(
        ...     [("s", "a"), ("a", "t")],
        ...     {"a": StandbyModel([unit] * 3, k=2, mc_samples=2000, seed=1)},
        ... )
        >>> trio.is_analytically_solvable()
        False
        >>> trio.get_non_analytic_nodes()
        {'a': 'StandbyModel'}
        """
        return len(self.get_non_analytic_nodes()) == 0

    def analysis_routes(self) -> Dict[str, "AnalysisRoute"]:
        """How each analysis of this RBD is computed, found without running
        it: exactly, numerically, by simulation, or not at all.

        The route follows from the method, then from the nodes' models:
        the structure is always evaluated exactly, whatever the diagram, but
        a system value is only as exact as the node values it is made of.
        So a node whose reliability is a numerical convolution makes the
        analyses built on it numerical, and one whose reliability is fitted
        to simulated lifetimes makes them simulated (see
        ``get_non_analytic_nodes``). A refusal is found by the check the
        method itself runs, and its reason is the message it would raise.

        Returns
        -------
        dict[str, AnalysisRoute]
            For each public analysis, by method name: its ``route``
            (``"exact"``, ``"numerical"``, ``"simulated"`` or
            ``"refused"``), the ``reason``, and the ``nodes`` that decide it
            (see [`AnalysisRoute`][repyability.AnalysisRoute]).

        Examples
        --------
        Two units needed of three, with the third a cold spare, have no
        closed form, so the node's reliability is simulated, and so is
        everything built on it:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, StandbyModel
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": StandbyModel(
        ...             [unit] * 3, k=2, mc_samples=2000, seed=1
        ...         ),
        ...         "b": unit,
        ...     },
        ... )
        >>> routes = rbd.analysis_routes()
        >>> routes["sf"].route, routes["sf"].nodes
        ('simulated', ('a',))
        >>> routes["structural_importance"].route
        'exact'
        >>> routes["mean"].route
        'simulated'
        """
        from repyability.rbd import routes as r

        nodes = {n: r.model_route(m) for n, m in self.reliabilities.items()}
        out: Dict[str, r.AnalysisRoute] = {}

        def give(names, route) -> None:
            for name in names:
                out[name] = route

        def built(route, reason):
            return r.with_nodes(route, reason, nodes, "reliabilities")

        grouped = bool(self.ccf_groups)
        give(
            (
                "sf",
                "ff",
                "reliability",
                "unreliability",
                "Hf",
                "cs",
                "node_sf",
                "node_ff",
            ),
            built(
                r.EXACT,
                "The structure function over the node reliabilities, "
                "exactly"
                + (
                    ", summed over the common-cause groups' outcomes."
                    if grouped
                    else "."
                ),
            ),
        )
        give(
            ("df", "hf"),
            built(
                r.NUMERICAL,
                "The exact reliability, differentiated by central "
                "differences.",
            ),
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
        no_ccf = r.refusal(self._require_no_ccf)
        give(
            (
                "birnbaum_importance",
                "improvement_potential",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "criticality_importance",
                "fussell_vesely",
                "fussel_vesely",
            ),
            (
                r.refused(no_ccf)
                if no_ccf
                else built(
                    r.EXACT,
                    "The exact system reliability, with each node working and "
                    "failed.",
                )
            ),
        )
        out["parameter_sensitivity"] = (
            r.refused(no_ccf)
            if no_ccf
            else built(
                r.NUMERICAL,
                "The exact Birnbaum importance times each parameter's "
                "derivative, by differences (composite and non-parametric "
                "nodes are left out).",
            )
        )
        out["structural_importance"] = r.AnalysisRoute(
            r.EXACT, "From the structure alone."
        )
        stateless = [
            n
            for n, m in self.reliabilities.items()
            if not self._node_is_stateable(m)
        ]
        note = (
            f" A state cannot be given for {r._names(stateless)} (standby, "
            "load-sharing, repeated and nested nodes)."
            if stateless
            else ""
        )
        states = r.refusal(self._require_no_ccf_for_states)
        give(
            ("sf_given_state", "importances_given_state"),
            (
                r.refused(states)
                if states
                else built(
                    r.EXACT,
                    "The exact system reliability over each node's "
                    "reliability given its state." + note,
                )
            ),
        )
        out["remaining_life"] = (
            r.refused(states or fixed or "")
            if states or fixed
            else built(
                r.NUMERICAL,
                "The reliability given the state, inverted by root-finding."
                + note,
            )
        )
        capacity = r.refusal(self._require_capacity) or r.refusal(
            self._require_capacity_outside_groups
        )
        out["capacity_distribution"] = (
            r.refused(capacity)
            if capacity
            else built(
                r.EXACT,
                "The exact distribution of the system's capacity from the "
                "node reliabilities.",
            )
        )
        no_capacity = r.refusal(self._require_capacity)
        out["system_capacity"] = (
            r.refused(no_capacity)
            if no_capacity
            else r.AnalysisRoute(
                r.EXACT,
                "The exact distribution of the system's capacity from the "
                "node probabilities given.",
            )
        )
        out["system_probability"] = r.AnalysisRoute(
            r.EXACT,
            "The structure function over the node probabilities given.",
        )
        out["path_set_probabilities"] = r.AnalysisRoute(
            r.EXACT, "From the node probabilities given."
        )
        give(
            (
                "improvement_allocation",
                "equal_allocation",
                "simple_allocation",
                "minimum_effort_allocation",
                "cost_based_allocation",
            ),
            r.AnalysisRoute(
                r.NUMERICAL,
                "A solver over the exact system probability, from the node "
                "probabilities given (not the node models).",
            ),
        )
        out["sf_uncertainty"] = (
            r.refused(no_ccf)
            if no_ccf
            else r.AnalysisRoute(
                r.SIMULATED,
                "The node parameters drawn from their uncertainty, and the "
                "exact system reliability for each draw.",
            )
        )
        independent = (
            " The members of a common-cause group are sampled "
            "independently: the common cause is left out."
            if grouped
            else ""
        )
        batched = self._row_sampler() is not None
        out["random"] = r.AnalysisRoute(
            r.SIMULATED,
            (
                "Monte-Carlo lifetimes, drawn in batches."
                if batched
                else "Monte-Carlo lifetimes, one at a time, as some nodes' "
                "draws cannot be batched; antithetic sampling is refused."
            )
            + independent,
        )
        lifetimes = r.refusal(self._require_lifetimes)
        grouped_inside = tuple(
            n
            for n, m in self.reliabilities.items()
            if isinstance(m, NonRepairableRBD) and m._grouped()
        )
        give(
            ("mean", "mean_time_to_failure"),
            (
                r.refused(lifetimes, grouped_inside)
                if lifetimes
                else built(
                    r.NUMERICAL,
                    "The exact reliability, integrated over time by adaptive "
                    "Gauss-Legendre quadrature (to about 1e-10, relative).",
                )
            ),
        )
        message = r.refusal(self._rare_event_sampler)
        out["unreliability_interval"] = (
            r.refused(message)
            if message
            else r.AnalysisRoute(
                r.SIMULATED,
                "Simulated lifetimes, by the method chosen: plain sampling "
                "for a probability that is not small, subset simulation "
                "otherwise.",
            )
        )
        out["random_block"] = r.AnalysisRoute(
            r.SIMULATED,
            "A block of the Monte-Carlo lifetimes random draws with n_jobs."
            + independent,
        )
        out["mean_time_to_failure_interval"] = r.AnalysisRoute(
            r.SIMULATED,
            "The mean of Monte-Carlo lifetimes (see random), with its "
            "confidence interval." + independent,
        )
        replay = r.refusal(self._require_replayable)
        out["compare"] = (
            r.refused(replay)
            if replay
            else r.AnalysisRoute(
                r.SIMULATED,
                "Monte-Carlo lifetimes of both systems, with common random "
                "numbers." + independent,
            )
        )
        means = {n: r.mean_route(m) for n, m in self.reliabilities.items()}
        refusals = {
            n: how for n, (route, how) in means.items() if route == r.REFUSED
        }
        out["node_mttf"] = (
            r.refused(next(iter(refusals.values())), tuple(refusals))
            if refusals
            else r.with_nodes(
                r.EXACT, "Each node's own mean lifetime.", means, "means"
            )
        )
        allocation = r.refusal(self._require_no_ccf_for_allocation)
        give(
            ("allocate_redundancy", "redundancy_front"),
            (
                r.refused(allocation)
                if allocation
                else built(
                    r.EXACT,
                    "Each candidate's exact system reliability, searched "
                    "exactly or greedily. Cold standby copies (strategy "
                    "'cold' or 'choose') are scored as a StandbyModel, "
                    "simulated for a node that needs two or more copies "
                    "working, unless they are identical Exponential units.",
                )
            ),
        )
        rrap = r.refusal(
            functools.partial(
                self._require_no_ccf_for_allocation, reliabilities=True
            )
        )
        out["allocate_reliability_redundancy"] = (
            r.refused(rrap)
            if rrap
            else built(
                r.NUMERICAL,
                "Branch and bound over the copies, and SLSQP over the "
                "reliabilities, of the exact system reliability.",
            )
        )
        return dict(sorted(out.items()))

    def random(
        self,
        size,
        seed=None,
        *,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
    ):
        """Draw ``size`` random system lifetimes (Monte-Carlo).

        Each node's lifetime is drawn independently from its own model's
        ``random``, and the system fails when its last working minimal path
        set breaks: each sample is the maximum, over the minimal path sets,
        of the minimum lifetime of the path set's members (k-out-of-n and
        repeated nodes are accounted for). When every
        node's draws can be replayed as one block (e.g. surpyval parametric
        distributions, and composite nodes built from them), all samples
        are computed at once; otherwise they are simulated one at a time by
        failing nodes in time order until the system fails. Both paths draw
        the same random numbers in the same order, so they give identical
        results. Either way the system is worked out through the diagram's
        modules, without listing its path sets, so a large redundant diagram
        is sampled about as fast as a small one.

        Common-cause groups are ignored, without a warning: their
        basic-event model assumes a small failure probability, while a
        lifetime runs to ``Q = 1``. The same applies to
        ``mean(method="simulate")``, ``mean_time_to_failure_interval`` and
        ``compare``; the exact ``mean`` refuses them.
        Fixed-probability nodes have no lifetime (surpyval draws 0/1 event
        indicators for them), so the samples are not meaningful for an RBD
        containing any.

        Parameters
        ----------
        size : int
            Number of system lifetimes to draw.
        seed : int or None, optional
            If given, seeds numpy's global RNG for the duration of the draw so
            the result is reproducible (surpyval's ``.random`` uses the global
            RNG); the caller's RNG state is restored afterwards. By default
            None (non-reproducible). It cannot make surpyval non-parametric
            node models (e.g. a Kaplan-Meier fit) reproducible: surpyval
            draws those from a fresh, OS-seeded generator on every call.
        antithetic : bool, optional
            Draw the lifetimes in antithetic pairs, by default False: the
            second of each pair (samples ``2i`` and ``2i + 1``) is drawn
            from ``1 - u`` for every uniform ``u`` the first drew. Each
            lifetime is still a correct draw, but the two of a pair are
            negatively correlated (for a system whose lifetime rises with
            its nodes'), so their mean varies less than two independent
            draws'. The pairs, not the lifetimes, are independent: estimate
            a standard error from the pairs' means. ``size`` must be even,
            and every node's draws must be replayable (as for the batched
            sampling above), else ``NotImplementedError``.
        n_jobs : int, optional
            Draw in parallel, in blocks of ``RANDOM_BLOCK`` (10_000)
            lifetimes, each seeded in turn from ``seed``, over ``n_jobs``
            processes (-1: one per CPU). The blocks are the same however
            many processes draw them, so the lifetimes do not depend on
            ``n_jobs`` (any number, 1 included), but they differ from a
            draw without it. By default None: one draw in this process.

        Returns
        -------
        numpy.ndarray
            ``size`` system lifetimes; ``inf`` in a sample where the system
            never fails (e.g. through a perfectly reliable path).

        Raises
        ------
        ValueError
            If ``size`` is not a positive integer (even, with
            ``antithetic``) when ``antithetic`` or ``n_jobs`` is given, or
            ``n_jobs`` is not a positive integer or -1.
        NotImplementedError
            If ``antithetic`` is asked for and a node's draws cannot be
            replayed from uniforms (a node model sampled its own way).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> lifetimes = rbd.random(5, seed=1)
        >>> lifetimes.shape
        (5,)
        >>> bool((lifetimes == rbd.random(5, seed=1)).all())
        True
        """
        if not antithetic and n_jobs is None:
            with numpy_seed(seed):
                return self._draw(size)
        montecarlo.check_count(size, antithetic, "size")
        jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
        return self._simulate_lifetimes(size, seed, antithetic, jobs, None)

    def random_block(
        self, block: int, seed: int, *, antithetic: bool = False
    ) -> np.ndarray:
        """Block ``block`` of the lifetimes ``random(size, seed=seed,
        n_jobs=...)`` draws: lifetimes ``block * RANDOM_BLOCK`` to
        ``(block + 1) * RANDOM_BLOCK - 1`` of them, for any ``n_jobs``.

        With ``n_jobs``, ``random`` draws its lifetimes in blocks of
        ``RANDOM_BLOCK`` (10 000), each seeded from ``seed`` and its
        position alone. This draws one, so a large draw can be split across
        machines, or any block of it drawn again to check it. A last,
        shorter block of a draw is the start of the full block.

        Parameters
        ----------
        block : int
            The block's position, from 0.
        seed : int
            The draw's seed, required.
        antithetic : bool, optional
            Draw in antithetic pairs, as ``random`` does with
            ``antithetic=True``, by default False.

        Returns
        -------
        numpy.ndarray
            The block's ``RANDOM_BLOCK`` lifetimes.

        Raises
        ------
        ValueError
            If ``block`` is not a whole number of at least 0, or ``seed``
            is None.
        NotImplementedError
            As for ``random`` with ``antithetic``.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> whole = rbd.random(30_000, seed=5, n_jobs=1)
        >>> second = rbd.random_block(1, seed=5)
        >>> bool((second == whole[10_000:20_000]).all())
        True
        """
        if (
            isinstance(block, bool)
            or not isinstance(block, (int, np.integer))
            or block < 0
        ):
            raise ValueError(
                f"block must be a whole number of at least 0, got {block!r}."
            )
        if seed is None:
            raise ValueError(
                "A block needs the draw's seed: the blocks of a draw share it."
            )
        # The block-th of the seeds random spawns in turn (see block_seed).
        seeds = np.random.SeedSequence(seed, spawn_key=(int(block),))
        block_seed = int(seeds.generate_state(1, dtype=np.uint32)[0])
        return _random_block((self, RANDOM_BLOCK, block_seed, antithetic))

    def _draw(self, size, antithetic: bool = False) -> np.ndarray:
        """``size`` lifetimes from numpy's global RNG as it stands."""
        if antithetic:
            return self._random_antithetic(size)
        fast = self._random_vectorised(size)
        if fast is not None:
            return fast
        return self._random_by_events(size)

    def _random_antithetic(self, size) -> np.ndarray:
        """``size`` (even) lifetimes in antithetic pairs: the second of
        each pair drawn from ``1 - u`` for the first's uniforms ``u``."""
        sampler = self._row_sampler()
        if sampler is None:
            raise NotImplementedError(
                "Antithetic sampling needs every node's draws to be "
                "replayable from uniforms (surpyval parametric distributions "
                "and the composite nodes built from them)."
            )
        out = np.empty(size)
        for first, rows in _row_blocks(size // 2, sampler.width):
            u = np.random.random_sample((rows, sampler.width))
            out[2 * first : 2 * (first + rows) : 2] = sampler.draw(u)
            out[2 * first + 1 : 2 * (first + rows) : 2] = sampler.draw(1.0 - u)
        _check_lifetimes(out)
        return out

    def _simulate_lifetimes(
        self,
        n: int,
        seed,
        antithetic: bool,
        jobs: Optional[int],
        stop: Optional[Callable[[np.ndarray], int]],
    ) -> np.ndarray:
        """``n`` lifetimes, then more while ``stop`` (given those so far)
        asks for them: in this process from numpy's global RNG (seeded,
        then restored, with a seed), or, with ``jobs``, in seeded blocks
        over that many processes. Either way a run that stops at ``m``
        lifetimes draws those it would in a run of ``m`` from the start,
        in blocks of that run's first ``n``."""
        seeds = np.random.SeedSequence(seed) if jobs is not None else None
        executor = (
            montecarlo.process_pool(jobs)
            if jobs is not None and jobs > 1
            else None
        )
        parts: List[np.ndarray] = []
        try:
            with numpy_seed(seed if jobs is None else None):
                batch = n
                while batch:
                    if seeds is None:
                        parts.append(self._draw(batch, antithetic))
                    else:
                        tasks = [
                            (
                                self,
                                size,
                                montecarlo.block_seed(seeds),
                                antithetic,
                            )
                            for size in montecarlo.blocks(batch, RANDOM_BLOCK)
                        ]
                        if executor is None:
                            parts.extend(map(_random_block, tasks))
                        else:
                            parts.extend(executor.map(_random_block, tasks))
                    batch = 0 if stop is None else stop(np.concatenate(parts))
        finally:
            if executor is not None:
                executor.shutdown()
        return np.concatenate(parts)

    def _random_vectorised(self, size) -> Optional[np.ndarray]:
        """``random(size)`` without the per-sample event loop, when every
        node's draws can be replayed in one block (see :meth:`_row_sampler`);
        otherwise ``None`` (nothing is drawn)."""
        sampler = self._row_sampler()
        if sampler is None:
            return None
        state = np.random.get_state()
        out = np.empty(size)
        for first, rows in _row_blocks(size, sampler.width):
            out[first : first + rows] = sampler.draw(
                np.random.random_sample((rows, sampler.width))
            )
        if np.isnan(out).any():
            # The event loop's ordering of NaN times is not reproducible
            # here; rewind and let it run.
            np.random.set_state(state)
            return None
        return out

    def _row_sampler(self) -> Optional[RowSampler]:
        """This RBD's ``random(1)`` as a :class:`RowSampler`, when every
        node's draws can be replayed that way (which also lets an RBD nested
        as a node be batched); otherwise ``None``.

        A coherent system fails when its last intact path set breaks, so its
        lifetime is the max over minimal path sets of the min of their
        members' lifetimes: the same value the event loop finds. It is
        worked out through the diagram's modules, without listing the path
        sets: a series module fails at its members' first failure, a
        parallel one at their last, and a k-out-of-n one at the failure that
        leaves fewer than ``k`` working. A sample in which any node drew NaN
        comes out NaN, so that the caller falls back to the event loop,
        which orders NaN times its own way.
        """
        nodes = self._components()
        samplers: list[RowSampler] = []
        for node in nodes:
            node_sampler = row_sampler(self.reliabilities[node])
            if node_sampler is None:
                return None
            samplers.append(node_sampler)
        structure = self._decomposition()

        def sample(u):
            size = len(u)
            lifetimes, start = {}, 0
            for node, sampler in zip(nodes, samplers):
                end = start + sampler.width
                lifetimes[node] = sampler.draw(u[:, start:end])
                start = end
            out = np.array(structure.lifetime(lifetimes, size), dtype=float)
            for lifetime in lifetimes.values():
                out[np.isnan(lifetime)] = np.nan
            return out

        return RowSampler(sum(s.width for s in samplers), sample)

    def _components(self) -> list:
        """Every node of the diagram that is a component of its own (the
        input and output included), in the diagram's order: a repeated node
        is the component it repeats, which fails once for all its
        appearances."""
        return [n for n in self.G.nodes if n not in self.repeated]

    def _random_by_events(self, size) -> np.ndarray:
        """``random(size)`` by stepping through each sample's failures in
        time order until the system fails; works for any node model."""
        out = np.zeros(size)
        for i in range(size):
            event_queue: PriorityQueue = PriorityQueue()
            for node in self._components():
                # .random(1) returns a 1-element array; take the scalar so
                # the event time orders the PriorityQueue and assigns into
                # ``out`` (NumPy >= 2 rejects assigning a 1-element array to
                # a scalar).
                one = np.asarray(self.reliabilities[node].random(1))
                time = float(one.reshape(-1)[0])
                event_queue.put(NodeFailure(time, node))

            working_nodes = {k: True for k in self._components()}
            system_working = True
            while system_working:
                if event_queue.empty():
                    # Every node has failed and the system still works: an
                    # edge joins the input to the output directly, so it
                    # never fails (as the batched path finds too).
                    time = np.inf
                    break
                failure = event_queue.get()
                time = failure.time
                working_nodes[failure.node] = False
                system_working = self.is_system_working(
                    working_nodes, method="p"
                )
            out[i] = time
        return out

    def mean(
        self,
        mc_samples: Optional[int] = None,
        seed=None,
        *,
        method: str = "exact",
        tolerance: Optional[float] = None,
        confidence: Optional[float] = None,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
    ) -> float:
        """Mean time to failure (MTTF) of the system.

        Exact by default: the area under the system reliability,
        ``MTTF = integral from 0 to infinity of R(t) dt``, with ``R`` the
        exact [`sf`][repyability.NonRepairableRBD.sf], integrated by
        adaptive Gauss-Legendre quadrature to about ``1e-10``, relative. It
        is as exact as the node reliabilities it is made of: a node whose
        reliability is a numerical convolution, or a fit to simulated
        lifetimes, brings its own error (see ``analysis_routes``). A system
        that may never fail has an infinite MTTF, ``inf``: one that needs
        only nodes some of whose units never fail (a
        limited-failure-population model, ``PerfectReliability``). A
        fixed-probability node counts as working, or failed, from the start
        and for good, as ``sf`` takes it. This is also the MTTF this RBD
        brings to another it is nested in (see ``node_mttf``).

        ``method="simulate"`` estimates the MTTF instead, as the mean of
        ``mc_samples`` simulated lifetimes (see
        [`random`][repyability.NonRepairableRBD.random], which leaves out
        common-cause groups); the estimate's standard error is the
        lifetimes' standard deviation over ``sqrt(mc_samples)``.
        ``mean_time_to_failure_interval`` gives such an estimate with its
        confidence interval.

        Parameters
        ----------
        mc_samples : int, optional
            With ``method="simulate"``: the number of system lifetimes to
            simulate, by default 100_000.
        seed : int or None, optional
            With ``method="simulate"``: the seed for reproducibility (see
            ``random``), by default None.
        method : {"exact", "simulate"}, optional
            How to find the MTTF, by default ``"exact"``.
        tolerance : float, optional
            With ``method="simulate"``: simulate until the MTTF is known to
            within ``tolerance`` (in the lifetimes' units) either side, at
            ``confidence``. After the first ``mc_samples`` lifetimes, and
            each further ``mc_samples``, the run stops once the half-width of
            the confidence interval of the mean is at most ``tolerance``, or
            ``max_samples`` have been drawn (then with a RuntimeWarning). By
            default None: exactly ``mc_samples``. A run that stops at ``m``
            lifetimes gives the result of a run of ``m`` from the start
            (without ``n_jobs``).
        confidence : float, optional
            With ``method="simulate"``: the confidence level ``tolerance``
            is judged at, by default 0.95.
        max_samples : int, optional
            With ``method="simulate"``: the most lifetimes a run to
            ``tolerance`` draws, by default 100 times ``mc_samples``.
        antithetic : bool, optional
            With ``method="simulate"``: draw the lifetimes in antithetic
            pairs (see ``random``), by default False. ``mc_samples`` must
            then be even.
        n_jobs : int, optional
            With ``method="simulate"``: draw in parallel over ``n_jobs``
            processes (see ``random``), by default None.

        Returns
        -------
        float
            The MTTF (``inf`` if the system may never fail), or with
            ``method="simulate"`` its estimate.

        Raises
        ------
        ValueError
            If ``method`` is neither ``"exact"`` nor ``"simulate"``; for the
            exact MTTF, if every node is fixed-probability, so the system has
            no lifetimes; or if a simulation option is invalid (see
            ``random``; ``max_samples`` without a ``tolerance``, or smaller
            than ``mc_samples``).
        NotImplementedError
            For the exact MTTF, if the RBD (or an RBD nested in it) has
            common-cause groups. Their models split a failure probability
            they assume is small (see [`CCFGroup`][repyability.CCFGroup]),
            while over a whole lifetime it runs to 1.

        Warns
        -----
        FutureWarning
            If a simulation option is given without ``method="simulate"``:
            it is ignored.

        Examples
        --------
        A single exponential component with failure rate 0.01 has an MTTF
        of 100:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> round(rbd.mean(), 9)
        100.0
        >>> estimate = rbd.mean(method="simulate", mc_samples=10_000, seed=1)
        >>> print(f"{estimate:.1f}")
        98.9

        Two in parallel last 100 + 100 - 50 = 150 on average:

        >>> pair = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {n: surv.Exponential.from_params([0.01]) for n in "ab"},
        ... )
        >>> round(pair.mean(), 9)
        150.0
        """
        if method == "exact":
            ignored(
                "mean()",
                "the MTTF is exact unless method='simulate'.",
                {
                    "mc_samples": mc_samples,
                    "seed": seed,
                    "tolerance": tolerance,
                    "confidence": confidence,
                    "max_samples": max_samples,
                    "antithetic": antithetic,
                    "n_jobs": n_jobs,
                },
            )
            return self._exact_mean()
        if method != "simulate":
            raise ValueError(
                f"method must be 'exact' or 'simulate', got {method!r}."
            )
        samples = self._mttf_samples(
            100_000 if mc_samples is None else mc_samples,
            seed,
            tolerance,
            0.95 if confidence is None else confidence,
            max_samples,
            antithetic,
            n_jobs,
        )
        return samples.mean().item()

    def _exact_mean(self) -> float:
        """The exact MTTF: the area under the system reliability."""
        self._require_lifetimes()
        knots = [model_knots(m) for m in self.reliabilities.values()]
        return mean_lifetime(
            lambda t: self.sf(t), np.concatenate([np.empty(0), *knots])
        )

    def _require_lifetimes(self) -> None:
        """Raise unless the system's exact mean lifetime is defined: its
        reliability must vary with time, and hold over whole lifetimes,
        which common-cause groups' models do not."""
        if self.is_fixed:
            raise ValueError(
                "System reliability does not vary with time (all nodes are "
                "fixed-probability): the system has no lifetimes to average."
            )
        if self._grouped():
            raise NotImplementedError(
                "The exact MTTF does not account for common-cause (CCF) "
                "groups: their models split a failure probability they "
                "assume is small (see CCFGroup), while over a whole lifetime "
                "it runs to 1. method='simulate' estimates the MTTF with the "
                "common cause left out."
            )

    def _grouped(self) -> bool:
        """Whether this RBD, or one nested in it, has common-cause groups."""
        return bool(self.ccf_groups) or any(
            isinstance(model, NonRepairableRBD) and model._grouped()
            for model in self.reliabilities.values()
        )

    def mean_time_to_failure(
        self,
        mc_samples: Optional[int] = None,
        seed=None,
        *,
        method: str = "exact",
        tolerance: Optional[float] = None,
        confidence: Optional[float] = None,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
    ) -> float:
        """Mean time to failure (MTTF) of the system; the same as ``mean``.

        Exact by default, and with ``method="simulate"`` a Monte-Carlo
        estimate from ``mc_samples`` simulated lifetimes, which leaves out
        common-cause groups (see [`mean`][repyability.NonRepairableRBD.mean]).

        Parameters
        ----------
        mc_samples : int, optional
            With ``method="simulate"``: the number of system lifetimes to
            simulate, by default 100_000.
        seed : int or None, optional
            With ``method="simulate"``: the seed for reproducibility (see
            ``random``), by default None.
        method : {"exact", "simulate"}, optional
            How to find the MTTF, by default ``"exact"``.
        tolerance : float, optional
            With ``method="simulate"``: simulate until the MTTF is known to
            within ``tolerance`` either side (see ``mean``), by default None.
        confidence : float, optional
            With ``method="simulate"``: the confidence level ``tolerance``
            is judged at, by default 0.95.
        max_samples : int, optional
            With ``method="simulate"``: the most lifetimes a run to
            ``tolerance`` draws, by default 100 times ``mc_samples``.
        antithetic : bool, optional
            With ``method="simulate"``: draw the lifetimes in antithetic
            pairs (see ``random``), by default False.
        n_jobs : int, optional
            With ``method="simulate"``: draw in parallel over ``n_jobs``
            processes (see ``random``), by default None.

        Returns
        -------
        float
            The MTTF, or its estimate.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``mean``.

        Warns
        -----
        FutureWarning
            As for ``mean``.
        """
        if method == "exact":
            ignored(
                "mean_time_to_failure()",
                "the MTTF is exact unless method='simulate'.",
                {
                    "mc_samples": mc_samples,
                    "seed": seed,
                    "tolerance": tolerance,
                    "confidence": confidence,
                    "max_samples": max_samples,
                    "antithetic": antithetic,
                    "n_jobs": n_jobs,
                },
            )
            return self._exact_mean()
        return self.mean(
            mc_samples,
            seed=seed,
            method=method,
            tolerance=tolerance,
            confidence=confidence,
            max_samples=max_samples,
            antithetic=antithetic,
            n_jobs=n_jobs,
        )

    def unreliability_interval(
        self,
        x: float,
        *,
        method: str = "auto",
        relative_tolerance: float = 0.1,
        confidence: float = 0.95,
        max_samples: int = 10_000_000,
        seed=None,
    ) -> ConfidenceInterval:
        """The probability that the system has failed by ``x``, ``P(T <=
        x)``, estimated by simulation to a relative precision: for small
        probabilities, which plain sampling estimates slowly.

        The system's lifetime is drawn from a row of uniforms, as ``random``
        draws it (a standby or load-sharing node by its own logic), so the
        probability is an integral over the unit cube of uniforms, which
        these methods sample:

        - ``"plain"``: independent samples. For a probability ``p`` it
          takes about ``z**2 * (1 - p) / (p * relative_tolerance**2)``
          lifetimes: some 4e10 for ``p = 1e-8`` and 10 %.
        - ``"latin_hypercube"`` and ``"sobol"``: Latin hypercube samples,
          and scrambled Sobol points (randomised quasi-Monte Carlo), in
          independent replicates, the error from their spread.
        - ``"cross_entropy"``: importance sampling from a Gaussian, in the
          standard normal space of the uniforms, shifted towards failure by
          the cross-entropy method, the samples weighted by the likelihood
          ratio.
        - ``"subset"``: subset simulation: ``p`` as a product of
          conditional probabilities of failing by ever earlier times, each
          about 0.1, sampled by Markov chains, in independent runs.
        - ``"auto"`` (the default): plain sampling when a pilot of 20 000
          lifetimes sees at least 50 failures; else the cross-entropy
          method when the system fails in at most 8 ways (minimal cut
          sets), each of components that are distributions; else subset
          simulation (see the guide's measurements). The cross-entropy
          method's mixture follows each way of failing it is fitted to,
          and misses the others: asked for directly on a system with more,
          it underestimates, with an interval that does not show it.

        Each runs until the half-width of the interval is at most
        ``relative_tolerance`` times the estimate, or ``max_samples``
        lifetimes are spent (with a ``RuntimeWarning``). Where the exact
        unreliability is known, ``ff(x)`` gives it at once, to full
        precision however small it is; this method is
        for diagrams whose nodes are simulated (a standby or load-sharing
        node without a closed form), and for checking.

        Parameters
        ----------
        x : float
            The time, at least 0.
        method : str, optional
            ``"auto"``, ``"plain"``, ``"latin_hypercube"``, ``"sobol"``,
            ``"cross_entropy"`` or ``"subset"``, by default ``"auto"``.
        relative_tolerance : float, optional
            The half-width of the interval to reach, relative to the
            estimate, by default 0.1.
        confidence : float, optional
            The interval's confidence level, by default 0.95.
        max_samples : int, optional
            The most lifetimes to draw, by default 10 000 000.
        seed : int, optional
            Seeds the simulation, by default None (not reproducible).

        Returns
        -------
        ConfidenceInterval
            The estimate, its interval (within [0, 1]) and standard error,
            the lifetimes drawn (``n_samples``), and the method used
            (``method``).

        Raises
        ------
        ValueError
            If ``x`` is not a number of at least 0, ``method`` is unknown,
            or ``relative_tolerance``, ``confidence`` or ``max_samples`` is
            out of range.
        NotImplementedError
            If a node's draws cannot be replayed from uniforms (a model that
            draws its own random numbers), or the diagram has common-cause
            groups, which the simulation leaves out.

        Examples
        --------
        Three units, any two of which keep the system up, fail by 10 with
        probability about 3e-6:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PerfectReliability
        >>> unit = surv.Weibull.from_params([1000.0, 1.5])
        >>> rbd = NonRepairableRBD(
        ...     [("s", n) for n in "abc"] + [(n, "v") for n in "abc"]
        ...     + [("v", "t")],
        ...     {"a": unit, "b": unit, "c": unit, "v": PerfectReliability},
        ...     k={"v": 2},
        ... )
        >>> estimate = rbd.unreliability_interval(10.0, seed=1)
        >>> estimate.method
        'cross_entropy'
        >>> bool(abs(estimate.estimate / rbd.ff(10.0) - 1) < 0.2)
        True
        """
        from repyability.rbd import rare_event

        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float, np.number))
            or not 0.0 <= float(x) < math.inf
        ):
            raise ValueError(f"x must be a time of at least 0, got {x!r}.")
        if method not in ("auto",) + rare_event.METHODS:
            raise ValueError(
                "method must be 'auto', "
                + ", ".join(repr(m) for m in rare_event.METHODS)
                + f", got {method!r}."
            )
        if not 0.0 < relative_tolerance < 1.0:
            raise ValueError(
                "relative_tolerance must be in (0, 1), got "
                f"{relative_tolerance!r}."
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                f"confidence must be in (0, 1), got {confidence!r}."
            )
        montecarlo.check_count(max_samples, False, "max_samples")
        sampler = self._rare_event_sampler()
        result, used = rare_event.estimate(
            sampler.draw,
            sampler.width,
            float(x),
            method,
            float(relative_tolerance),
            float(confidence),
            int(max_samples),
            seed,
            self._few_failure_modes,
        )
        z = float(montecarlo.z_value(confidence))
        half = z * result.standard_error
        return ConfidenceInterval(
            estimate=result.p,
            lower=max(0.0, result.p - half),
            upper=min(1.0, result.p + half),
            confidence=confidence,
            standard_error=result.standard_error,
            n_samples=result.lifetimes,
            method=used,
        )

    def _few_failure_modes(self) -> bool:
        """Whether the system fails in few enough ways for the
        cross-entropy method (see ``rare_event.choose``): at most
        ``rare_event.MIXTURE`` minimal cut sets, of components that are
        each a distribution (or never or always fail), counted for a
        diagram of at most 40 components."""
        from repyability.rbd import rare_event

        components = [n for n in self._components() if n not in self.in_or_out]
        if len(components) > 40:
            return False
        for node in components:
            model = self.reliabilities[node]
            if model in (PerfectReliability, PerfectUnreliability):
                continue
            if inverse_sampler(model) is None:
                return False
        return len(self.get_min_cut_sets()) <= rare_event.MIXTURE

    def _rare_event_sampler(self) -> RowSampler:
        """The lifetimes as a function of rows of uniforms, for
        ``unreliability_interval``: raise if there are common-cause groups
        (which the simulation leaves out), or if a node's draws do not
        follow the uniforms (a model drawing its own random numbers)."""
        if self._grouped():
            raise NotImplementedError(
                "The simulated lifetimes leave common-cause groups out, so "
                "they do not estimate this diagram's unreliability; ff(x) "
                "includes the groups exactly."
            )
        sampler = self._row_sampler()
        replayable = False
        if sampler is not None:
            # The same uniforms must give the same lifetimes: a model that
            # draws its own random numbers does not.
            u = np.random.default_rng(0).random((16, sampler.width))
            state = np.random.get_state()
            try:
                replayable = np.array_equal(
                    sampler.draw(u), sampler.draw(u), equal_nan=True
                )
            finally:
                np.random.set_state(state)
        if sampler is None or not replayable:
            raise NotImplementedError(
                "Estimating a small unreliability needs every node's "
                "lifetime drawn from uniforms (surpyval parametric "
                "distributions, and the composite nodes built from them); "
                "a node here draws its own random numbers."
            )
        return sampler

    def mean_time_to_failure_interval(
        self,
        mc_samples: int = 100_000,
        confidence: float = 0.95,
        seed=None,
        *,
        tolerance: Optional[float] = None,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
    ) -> ConfidenceInterval:
        """Monte-Carlo MTTF estimate with a confidence interval.

        ``mean`` gives the MTTF exactly; this estimates it by simulation, as
        ``mean(method="simulate")`` does, and says how precise the estimate
        is. The MTTF is the mean of ``mc_samples`` simulated system lifetimes
        (see [`random`][repyability.NonRepairableRBD.random]; common-cause
        groups are ignored). By the central limit theorem its sampling
        error is normal with standard error
        ``sample std / sqrt(mc_samples)`` (of the antithetic pairs' means,
        over the square root of their number, with ``antithetic``), from
        which the two-sided interval ``estimate +/- z * standard_error`` is
        built; the lower bound is clipped at 0. The interval describes the
        simulation error only, not uncertainty in the node models.

        Parameters
        ----------
        mc_samples : int, optional
            Number of Monte-Carlo samples, by default 100_000.
        confidence : float, optional
            The confidence level, in (0, 1), by default 0.95.
        seed : int or None, optional
            Seed for reproducibility (see ``random``), by default None.
        tolerance : float, optional
            Simulate until the interval is at most ``tolerance`` either side
            of the estimate: after the first ``mc_samples`` lifetimes, and
            each further ``mc_samples``, the run stops once it is, or once
            ``max_samples`` have been drawn (then with a RuntimeWarning). By
            default None: exactly ``mc_samples``. Without ``n_jobs``, a run
            that stops at ``m`` lifetimes gives the result of a run of ``m``
            from the start.
        max_samples : int, optional
            The most lifetimes a run to ``tolerance`` draws, by default 100
            times ``mc_samples``.
        antithetic : bool, optional
            Draw the lifetimes in antithetic pairs (see ``random``), by
            default False: a narrower interval for the same number of
            lifetimes, when the system's lifetime rises with its nodes' (as
            a coherent system's does). ``mc_samples`` must then be even.
        n_jobs : int, optional
            Draw in parallel over ``n_jobs`` processes (-1: one per CPU; see
            ``random``), by default None. The result does not depend on the
            number of processes.

        Returns
        -------
        ConfidenceInterval
            The estimate, bounds, confidence level, standard error and
            sample count (see
            [`ConfidenceInterval`][repyability.ConfidenceInterval]).

        Raises
        ------
        ValueError
            If ``confidence`` is not in (0, 1), or another option is invalid
            (see ``mean``).
        NotImplementedError
            With ``antithetic``, if a node's draws cannot be replayed from
            uniforms.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> ci = rbd.mean_time_to_failure_interval(mc_samples=10_000, seed=1)
        >>> print(f"{ci.estimate:.1f} ({ci.lower:.1f}, {ci.upper:.1f})")
        98.9 (96.9, 100.8)

        Simulate until the MTTF is known to within 0.5 either side:

        >>> ci = rbd.mean_time_to_failure_interval(
        ...     mc_samples=10_000, seed=1, tolerance=0.5
        ... )
        >>> ci.n_samples, round(ci.upper - ci.estimate, 2)
        (160000, 0.49)
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        samples = self._mttf_samples(
            mc_samples,
            seed,
            tolerance,
            confidence,
            max_samples,
            antithetic,
            n_jobs,
        )
        estimate = float(samples.mean())
        standard_error = montecarlo.standard_error(samples, antithetic)
        z = montecarlo.z_value(confidence)
        return ConfidenceInterval(
            estimate=estimate,
            lower=max(0.0, estimate - z * standard_error),
            upper=estimate + z * standard_error,
            confidence=confidence,
            standard_error=standard_error,
            n_samples=len(samples),
        )

    def _mttf_samples(
        self,
        mc_samples,
        seed,
        tolerance,
        confidence,
        max_samples,
        antithetic: bool,
        n_jobs,
    ) -> np.ndarray:
        """The simulated lifetimes an MTTF estimate is the mean of."""
        if (
            tolerance is None
            and max_samples is None
            and not antithetic
            and n_jobs is None
        ):
            return self.random(mc_samples, seed=seed)
        montecarlo.check_count(mc_samples, antithetic, "mc_samples")
        montecarlo.check_confidence(confidence)
        limit = montecarlo.sample_limit(
            mc_samples,
            tolerance,
            max_samples,
            antithetic,
            ("mc_samples", "max_samples"),
        )
        jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)

        def more(values: np.ndarray) -> int:
            return montecarlo.more_samples(
                values,
                mc_samples,
                tolerance,
                confidence,
                limit,  # type: ignore[arg-type]
                antithetic,
                "MTTF",
                "max_samples",
            )

        stop = None if limit is None else more
        return self._simulate_lifetimes(
            mc_samples, seed, antithetic, jobs, stop
        )

    def compare(
        self,
        other: "NonRepairableRBD",
        mc_samples: int = 100_000,
        seed=None,
        *,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """How much longer (or shorter) this system's mean time to failure
        is than ``other``'s, by simulation with common random numbers.

        Both systems' lifetimes are simulated ``mc_samples`` times, and in
        each sample a component with the same name in both draws the same
        random numbers in both: the same lifetime where its model is the
        same, and a matching one (the same quantile of its own model) where
        it is not. The differences between the two systems' lifetimes then
        come from how the systems differ, not from chance, so their mean is
        a more precise estimate of the difference in MTTF than the
        difference of two independent estimates of the same size (the more
        the systems share, the more precise). The reliabilities themselves
        need no simulation: compare ``sf`` for those.

        Parameters
        ----------
        other : NonRepairableRBD
            The system to compare with.
        mc_samples : int, optional
            The number of lifetimes of each system, by default 100_000.
        seed : int, optional
            Seed for a reproducible comparison, by default None.
        confidence : float, optional
            The confidence level of the interval, by default 0.95.

        Returns
        -------
        ConfidenceInterval
            The mean difference (this system's lifetime minus ``other``'s)
            over the samples, with its standard error and a normal
            confidence interval (not clipped: the difference may be
            negative). Common-cause groups are ignored, as by ``random``.

        Raises
        ------
        ValueError
            If ``mc_samples`` or ``confidence`` is invalid.
        NotImplementedError
            If a node's draws cannot be replayed from uniforms (a node
            model sampled its own way). A non-parametric model (a
            Kaplan-Meier fit, say) draws its own random numbers, which the
            two systems do not share.

        Examples
        --------
        A third unit in parallel with two:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> def parallel(n):
        ...     names = [f"u{i}" for i in range(n)]
        ...     edges = [("s", u) for u in names] + [(u, "t") for u in names]
        ...     return NonRepairableRBD(edges, {u: unit for u in names})
        >>> gain = parallel(3).compare(parallel(2), mc_samples=20_000, seed=0)
        >>> round(gain.estimate, 1), round(gain.standard_error, 2)
        (14.8, 0.21)

        The exact difference, the integral of the difference in
        reliability, is 14.46. Two independent estimates from 20_000
        lifetimes each would give it with a standard error of about 0.42.
        """
        montecarlo.check_count(mc_samples, False, "mc_samples")
        montecarlo.check_confidence(confidence)
        key = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0])
        differences = self._keyed_lifetimes(
            mc_samples, key
        ) - other._keyed_lifetimes(mc_samples, key)
        estimate = float(np.mean(differences))
        standard_error = montecarlo.standard_error(differences, False)
        z = montecarlo.z_value(confidence)
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate - z * standard_error,
            upper=estimate + z * standard_error,
            confidence=confidence,
            standard_error=standard_error,
            n_samples=mc_samples,
        )

    def _keyed_lifetimes(self, n: int, key: int) -> np.ndarray:
        """``n`` lifetimes in which each component draws its uniforms from
        streams of its own, keyed by ``key``, its name and the uniform's
        place in its draw (common random numbers, see ``compare``)."""
        self._require_replayable()
        lifetimes = {}
        for node in self._components():
            sampler = cast(RowSampler, row_sampler(self.reliabilities[node]))
            name = zlib.crc32(repr(node).encode())
            u = np.empty((n, sampler.width))
            for j in range(sampler.width):
                u[:, j] = np.random.default_rng([key, name, j]).random(n)
            lifetimes[node] = sampler.draw(u)
        out = np.array(
            self._decomposition().lifetime(lifetimes, n), dtype=float
        )
        for lifetime in lifetimes.values():
            out[np.isnan(lifetime)] = np.nan
        _check_lifetimes(out)
        return out

    def time_to_reliability(
        self,
        target: float,
        upper_bound: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Time at which the system reliability falls to ``target``.

        Solves ``R(t) = target`` for ``t >= 0`` (the inverse of
        [`sf`][repyability.NonRepairableRBD.sf], so it honours common-cause
        groups). System reliability is non-increasing in time, so the
        crossing is bracketed between 0 and an upper bound and found with
        Brent's method (``scipy.optimize.brentq``). Unless ``upper_bound``
        is given, the bound is found by doubling from ``t = 1``.

        Parameters
        ----------
        target : float
            The reliability level to solve for, in (0, 1).
        upper_bound : float, optional
            An upper bound for the search, at which the reliability must be
            below ``target``; found automatically (by doubling) if None.
        **kwargs
            Keyword arguments of ``sf`` (``working_nodes``,
            ``broken_nodes``, ``method``).

        Returns
        -------
        float
            The time at which ``R(t) == target``.

        Raises
        ------
        ValueError
            If ``target`` is not in (0, 1); if the RBD is fixed-probability
            (reliability is constant in time); if ``target`` exceeds the
            system reliability at ``t = 0`` (so it is never reached); if no
            upper bound is found (after 1000 doublings); or if the
            reliability at ``upper_bound`` is still above ``target``. Also
            as for ``sf``.
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> round(rbd.time_to_reliability(0.9), 2)
        32.46
        """
        return self._invert_reliability(
            lambda t: float(self.sf(t, **kwargs)), target, upper_bound
        )

    def _invert_reliability(
        self,
        sf_func,
        target: float,
        upper_bound: Optional[float] = None,
    ) -> float:
        """Solve ``sf_func(t) == target`` for ``t >= 0``.

        ``sf_func`` maps a scalar time to a scalar system reliability and is
        monotonically non-increasing, so the solution is unique. Shared by
        :meth:`time_to_reliability` and :meth:`remaining_life`; the target is
        bracketed automatically (by doubling) unless ``upper_bound`` is given.
        """
        if not 0.0 < target < 1.0:
            raise ValueError("target reliability must be in (0, 1).")
        self._require_time_varying()

        r0 = float(sf_func(0.0))
        if target > r0:
            raise ValueError(
                f"target reliability {target} exceeds the system reliability "
                f"at t=0 ({r0:.6g}); it is never reached."
            )

        def f(t):
            return float(sf_func(t)) - target

        hi = upper_bound
        if hi is None:
            hi = 1.0
            for _ in range(1000):
                if f(hi) < 0.0:
                    break
                hi *= 2.0
            else:
                raise ValueError(
                    "Could not bracket the target reliability; pass an "
                    "explicit upper_bound."
                )
        return float(brentq(f, 0.0, hi))

    def bx_life(self, x: float, **kwargs) -> float:
        """Bx life: the time by which ``x`` percent of systems have failed.

        The time at which ``R(t) = 1 - x / 100``, found with
        ``time_to_reliability(1 - x / 100)``. For example ``bx_life(10)``
        is the B10 life (10% failed, 90% reliability).

        Parameters
        ----------
        x : float
            The percentage failed, in (0, 100).
        **kwargs
            Keyword arguments of ``time_to_reliability`` (``upper_bound``)
            and ``sf`` (``working_nodes``, ``broken_nodes``, ``method``).

        Returns
        -------
        float
            The time at which ``x`` percent of systems have failed.

        Raises
        ------
        ValueError
            If ``x`` is not in (0, 100), or as for ``time_to_reliability``
            (e.g. for a fixed-probability RBD).
        NotImplementedError
            As for ``sf``.

        Examples
        --------
        The B10 life (10% failed) of a single Weibull component:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> round(rbd.bx_life(10), 2)
        32.46
        """
        if not 0.0 < x < 100.0:
            raise ValueError("x must be a percentage in (0, 100).")
        return self.time_to_reliability(1.0 - x / 100.0, **kwargs)

    # -- Condition-based ("digital twin") evaluation -----------------------

    def _node_is_stateable(self, model) -> bool:
        """True if a single scalar age applies unambiguously to ``model``.

        Excludes the composite / dynamic node models -- a standby arrangement,
        a repeated node, and a nested RBD -- whose conditioning semantics are
        out of scope for condition-based evaluation in this release.
        """
        return not isinstance(
            model,
            (
                StandbyModel,
                RepeatedStandbyNode,
                LoadSharingModel,
                RepeatedNode,
                RBD,
            ),
        )

    def _validate_state(self, state) -> dict:
        """A condition-based ``state`` mapping, validated, with a plain
        number taken as the node's age (``NodeState(age=...)``).

        Raises rather than silently ignoring bad input: a non-mapping, a value
        that is neither a :class:`NodeState` nor an age, the input/output
        node, an unknown node, or a node whose model is composite/dynamic.
        """
        if not isinstance(state, dict):
            raise TypeError(
                "state must be a dict of {node: NodeState} (or {node: age}), "
                f"got {type(state).__name__}."
            )
        state = {
            node: (
                NodeState(age=float(value))
                if isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                else value
            )
            for node, value in state.items()
        }
        valid = set(self.nodes)
        for node, node_state in state.items():
            if not isinstance(node_state, NodeState):
                raise TypeError(
                    f"state[{node!r}] must be a NodeState (or a number, its "
                    f"age), got {type(node_state).__name__}."
                )
            if node in self.in_or_out:
                which = "input" if node == self.input_node else "output"
                raise ValueError(
                    f"Cannot set state for the {which} node {node!r}."
                )
            if node not in valid:
                raise ValueError(
                    f"Unknown node {node!r} in state; it is not a node of "
                    "this RBD."
                )
            if not self._node_is_stateable(self.reliabilities[node]):
                raise ValueError(
                    "Condition-based evaluation supports only ordinary "
                    f"distribution components in this release; node {node!r} "
                    f"is a {type(self.reliabilities[node]).__name__}."
                )
        return state

    def _state_node_probabilities(self, x, state) -> Dict[Any, ArrayLike]:
        """Per-node forward reliability at ``x`` given each node's ``state``.

        ``x`` is a 1-d array. Each node conditions on its own current life: a
        node absent from ``state`` uses its unconditioned reliability (age 0),
        a failed node contributes zero, and an alive node of age ``X``
        contributes ``conditional_survival(model, x, X) = sf(X + x) / sf(X)``.
        """
        self._require_no_ccf_for_states()
        state = self._validate_state(state)
        node_probabilities: Dict[Any, ArrayLike] = {}
        for node_name, model in self.reliabilities.items():
            node_state = state.get(node_name)
            if node_state is None:
                node_probabilities[node_name] = np.atleast_1d(model.sf(x))
            elif not node_state.alive:
                node_probabilities[node_name] = np.atleast_1d(
                    PerfectUnreliability.sf(x)
                )
            else:
                node_probabilities[node_name] = np.atleast_1d(
                    conditional_survival(model, x, node_state.age)
                )
        return node_probabilities

    @check_x
    def sf_given_state(
        self,
        x: Optional[ArrayLike] = None,
        state: Optional[Dict[Hashable, NodeState]] = None,
        method: str = "p",
    ) -> Union[float, np.ndarray]:
        """System reliability over the next ``x``, given each node's state.

        The condition-based ("digital twin") generalisation of
        [`sf`][repyability.NonRepairableRBD.sf]: instead of assuming every
        component is new, each component conditions on its own current
        [`NodeState`][repyability.NodeState] (e.g. streamed from sensors),
        and the conditioned node reliabilities are propagated exactly
        through the system:

            R_i(x | X_i)     = R_i(X_i + x) / R_i(X_i)
            R_sys(x | state) = system reliability from the R_i(x | X_i)

        A node that is alive at age ``X_i`` contributes ``R_i(x | X_i)``
        (0 where ``R_i(X_i)`` is 0), a failed node (``alive=False``)
        contributes 0, and a node left out of ``state`` contributes its
        ordinary reliability ``R_i(x)``, as if new.

        Parameters
        ----------
        x : array_like, optional
            The further duration/s at which reliability is evaluated
            (measured from *now*, so ``x = 0`` is the present). May be
            omitted only for a fixed-probability RBD.
        state : dict[Hashable, NodeState], optional
            ``{node: NodeState}``, the current state of some or all of the
            component nodes. By default empty, which reproduces ``sf(x)``.
        method : str, optional
            ``"p"`` (the default) uses the minimal path sets and ``"c"`` the
            minimal cut sets; both are exact.

        Returns
        -------
        float or numpy.ndarray
            System reliability given the state: a float for scalar ``x``, an
            array for array ``x``.

        Raises
        ------
        TypeError
            If ``state`` is not a dict, or one of its values is not a
            ``NodeState``.
        ValueError
            If ``x`` is omitted for a time-varying RBD, or ``state`` names
            the input or output node, an unknown node (including a repeated
            node), or a node whose model is a standby, load-sharing or
            repeated node or a nested RBD.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Notes
        -----
        Only lifetime (time-varying) distributions age. A fixed-probability
        component stated alive contributes reliability 1 going forward (its
        per-demand uncertainty is resolved by observing it alive), whereas
        left out of ``state`` it contributes its fixed probability.
        Composite / dynamic nodes (standby, load-sharing, repeated, nested
        RBD) are not supported here in this release: they may be left out
        of ``state`` (and are then unconditioned) but raise if given a
        state.

        Examples
        --------
        A component 40 hours into life is less reliable over the next 50 than
        a new one would be:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, NodeState
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> round(rbd.sf_given_state(50, {"c": NodeState(age=40)}), 4)
        0.522
        """
        if state is None:
            state = {}
        node_probabilities = self._state_node_probabilities(x, state)
        return self.system_probability(node_probabilities, method=method)

    def remaining_life(
        self,
        target: float,
        state: Optional[Dict[Hashable, NodeState]] = None,
        upper_bound: Optional[float] = None,
    ) -> float:
        """Remaining useful life: the time until reliability falls to target.

        The condition-based analogue of ``time_to_reliability``: it solves
        ``sf_given_state(t, state) == target`` for ``t``, by the same
        bracketing and Brent's method. Because
        [`sf_given_state`][repyability.NonRepairableRBD.sf_given_state] is
        measured from now, the result is the time *remaining* from the
        current state. ``remaining_life(1 - x / 100, state)`` is the
        conditional Bx life.

        Parameters
        ----------
        target : float
            The system reliability level to solve for, in (0, 1).
        state : dict[Hashable, NodeState], optional
            ``{node: NodeState}``, the current state of some or all of the
            component nodes (see ``sf_given_state``). By default empty (all
            nodes new).
        upper_bound : float, optional
            An upper bound for the search, at which the reliability must be
            below ``target``; found automatically (by doubling) if None.

        Returns
        -------
        float
            The remaining time until ``R_sys(t | state) == target``.

        Raises
        ------
        ValueError
            If ``target`` is not in (0, 1); if the RBD is fixed-probability;
            if ``target`` exceeds the current system reliability
            ``R_sys(0 | state)`` (e.g. because the state has already failed
            the system); if no upper bound is found; if the reliability at
            ``upper_bound`` is still above ``target``; or if ``state`` is
            invalid (as for ``sf_given_state``).
        TypeError
            If ``state`` is not a dict of ``NodeState`` values.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, NodeState
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Weibull.from_params([100, 2])},
        ... )
        >>> round(rbd.remaining_life(0.9, {"c": NodeState(age=40)}), 2)
        11.51
        """
        if isinstance(target, dict):
            raise TypeError(
                "remaining_life's first argument is the reliability target, a "
                "number in (0, 1), and the states come second: "
                "remaining_life(0.9, state) or remaining_life(0.9, "
                "state=state)."
            )
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float, np.integer, np.floating))
            or not 0.0 < float(target) < 1.0
        ):
            raise ValueError(
                "target is the reliability to fall to, a number strictly "
                f"between 0 and 1, got {target!r}."
            )
        if state is None:
            state = {}
        return self._invert_reliability(
            lambda t: float(self.sf_given_state(t, state)),
            target,
            upper_bound,
        )

    @leaves_out_junctions
    def importances_given_state(
        self,
        x: Optional[ArrayLike] = None,
        state: Optional[Dict[Hashable, NodeState]] = None,
        kind: str = "failure",
    ) -> Dict[str, Dict[Any, Union[float, np.ndarray]]]:
        """Birnbaum and criticality importance given each node's state.

        Shows how each component's importance over a forward horizon ``x``
        shifts once the current wear on every component is accounted for.
        The Birnbaum and criticality importance measures are evaluated at
        the conditioned node reliabilities ``R_i(x | X_i)`` (see
        [`sf_given_state`][repyability.NonRepairableRBD.sf_given_state])
        rather than at the as-new reliabilities, so the rankings reflect
        the current state. With ``F = 1 - R``:

            birnbaum_i    = R_sys(x | state, i working)
                            - R_sys(x | state, i failed)
            criticality_i = birnbaum_i * F_i(x | X_i) / F_sys(x | state)

        the failure-oriented criticality: the share of the system failures
        over the horizon that node ``i`` accounts for. ``kind="success"``
        gives the success-oriented form,
        ``birnbaum_i * R_i(x | X_i) / R_sys(x | state)``, instead. These
        are the conventions of ``birnbaum_importance`` and
        ``criticality_importance``, applied to the system as it is now. A
        failed node's success-oriented criticality is 0.

        Parameters
        ----------
        x : array_like, optional
            The forward horizon/s over which importance is evaluated (from
            now). May be omitted only for a fixed-probability RBD.
        state : dict[Hashable, NodeState], optional
            ``{node: NodeState}``, the current state of some or all of the
            component nodes (see ``sf_given_state``). By default empty (all
            nodes new).
        kind : str, optional
            The criticality's form: ``"failure"`` (the default) or
            ``"success"``, as for ``criticality_importance``.

        Returns
        -------
        dict[str, dict[Any, float | numpy.ndarray]]
            ``{"birnbaum": {node: value}, "criticality": {node: value}}``
            over the component nodes. Values are floats for scalar ``x``
            and arrays for array ``x``.

        Raises
        ------
        TypeError
            If ``state`` is not a dict of ``NodeState`` values.
        ValueError
            If ``x`` is omitted for a time-varying RBD, ``state`` is
            invalid (as for ``sf_given_state``), or ``kind`` is neither
            ``"failure"`` nor ``"success"``.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        Two identical units in parallel, one of them 60 hours old: over
        the next 20 hours the system now depends more on the new unit:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, NodeState
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> rbd = NonRepairableRBD(
        ...     [("s", "old"), ("s", "new"), ("old", "t"), ("new", "t")],
        ...     {"old": unit, "new": unit},
        ... )
        >>> imp = rbd.importances_given_state(20, {"old": NodeState(age=60)})
        >>> {k: round(v, 4) for k, v in sorted(imp["birnbaum"].items())}
        {'new': 0.2442, 'old': 0.0392}
        """
        if state is None:
            state = {}
        if x is None:
            if self.is_fixed:
                x = 1.0
            else:
                raise ValueError(
                    "x is required: this RBD is time-varying (at least one "
                    "node model's probability depends on time)."
                )
        scalar_in = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        node_probabilities = self._state_node_probabilities(x_arr, state)
        birnbaum = super()._birnbaum_importance(node_probabilities)
        criticality = super()._criticality_importance(node_probabilities, kind)

        def _squeeze(measure):
            return {
                node: (
                    float(np.asarray(value).reshape(-1)[0])
                    if scalar_in
                    else np.asarray(value)
                )
                for node, value in measure.items()
            }

        return {
            "birnbaum": _squeeze(birnbaum),
            "criticality": _squeeze(criticality),
        }

    def node_mttf(
        self, mc_samples: Optional[int] = None, seed=None
    ) -> dict[Any, float]:
        """Mean time to failure (MTTF) of each component node.

        Each node's MTTF comes from its own model, without simulating:

        - a nested ``NonRepairableRBD`` or a
          [`RepeatedNode`][repyability.RepeatedNode]: the area under its
          reliability (see [`mean`][repyability.NonRepairableRBD.mean]);
        - a [`StandbyModel`][repyability.StandbyModel] or
          [`LoadSharingModel`][repyability.LoadSharingModel]: its
          ``mean()``, exact where it has a closed form or convolution, and
          otherwise the mean of the lifetimes simulated when it was built,
          which its reliability is fitted to;
        - a fixed-probability node: 0.0, as it has no time dimension;
        - any other model: its own ``mean()``, e.g. the exact mean of a
          surpyval distribution, except that a limited-failure-population
          model's is infinite: some of its units never fail. (surpyval's
          ``mean()`` of one is the *defective* mean, its failing units' mean
          weighted by their fraction.)

        Common-cause groups do not affect a node's own MTTF.

        Parameters
        ----------
        mc_samples : int, optional
            Ignored and deprecated: no node's MTTF is simulated.
        seed : int or None, optional
            Ignored and deprecated.

        Returns
        -------
        dict[Any, float]
            ``{node: MTTF}`` over the component nodes (not the input or
            output node).

        Raises
        ------
        AttributeError
            If a node's model has no ``mean()`` method (e.g.
            [`PerfectReliability`][repyability.PerfectReliability]).
        NotImplementedError
            If a nested RBD has common-cause groups (see ``mean``).

        Warns
        -----
        FutureWarning
            If ``mc_samples`` or ``seed`` is given.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": surv.Exponential.from_params([0.01]),
        ...         "b": surv.FixedEventProbability.from_params(0.1),
        ...     },
        ... )
        >>> {k: round(v, 2) for k, v in sorted(rbd.node_mttf().items())}
        {'a': 100.0, 'b': 0.0}
        """
        ignored(
            "node_mttf()",
            "no node's MTTF is simulated.",
            {"mc_samples": mc_samples, "seed": seed},
        )
        out: dict[Any, float] = {}
        for node in self.nodes:
            model = self.reliabilities[node]
            if isinstance(
                model,
                (
                    StandbyModel,
                    LoadSharingModel,
                    NonRepairableRBD,
                    RepeatedNode,
                ),
            ):
                out[node] = float(np.ravel(model.mean())[0])
            elif is_fixed_probability(model):
                out[node] = 0.0
            else:
                out[node] = model_mean(model)
        return out

    # Importance measures
    # https://www.ntnu.edu/documents/624876/1277590549/chapt05.pdf/82cd565f-fa2f-43e4-a81a-095d95d39272
    @leaves_out_junctions
    @check_x
    def birnbaum_importance(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Birnbaum importance of each node at time/s ``x``.

        ``B_i = R_sys(i working) - R_sys(i failed)``: the rate at which the
        system reliability changes with node ``i``'s reliability, which is
        also the probability that node ``i`` is critical (the system works
        if ``i`` works and fails if ``i`` fails). It does not depend on node
        ``i``'s own reliability. Exact; it assumes independent nodes. It is
        worked out for every node at once, as the derivative of the system
        reliability: a sum of products of the nodes' reliabilities and
        probabilities of failing (their models' ``sf`` and ``ff``), so a
        small one keeps its full relative precision, where the difference
        above would cancel in a reliable system.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: importance}`` over the component nodes: floats for
            scalar ``x``, arrays for array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD, or a working/broken
            node is invalid (as for ``sf``).
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        In a two-component parallel system each node's Birnbaum importance is
        the probability the *other* node has failed:

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {
        ...         "a": FixedEventProbability.from_params(0.1),
        ...         "b": FixedEventProbability.from_params(0.1),
        ...     },
        ... )
        >>> bi = rbd.birnbaum_importance()
        >>> {k: round(v, 4) for k, v in sorted(bi.items())}
        {'a': 0.1, 'b': 0.1}

        Once ``"b"`` has failed, the system depends entirely on ``"a"``:

        >>> bi = rbd.birnbaum_importance(broken_nodes=["b"])
        >>> {k: round(v, 4) for k, v in sorted(bi.items())}
        {'a': 1.0, 'b': 0.1}
        """
        self._require_no_ccf()
        node_probabilities, node_failures = self._importance_inputs(
            x, working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._birnbaum_importance(
                node_probabilities, node_failures=node_failures
            ),
        )

    @leaves_out_junctions
    @check_x
    def improvement_potential(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Improvement potential of each node at time/s ``x``.

        ``IP_i = R_sys(i working) - R_sys``: how much the system reliability
        would rise if node ``i`` were made perfect. It equals
        ``B_i * (1 - R_i)``, with ``B_i`` the Birnbaum importance, and is
        worked out as that product (with ``1 - R_i`` from the node's model's
        ``ff``), so a small one keeps its precision. Exact; it assumes
        independent nodes.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: improvement potential}`` over the component nodes:
            floats for scalar ``x``, arrays for array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD, or a working/broken
            node is invalid (as for ``sf``).
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        Two pumps in parallel (each failing with probability 0.1) feeding a
        valve in series (0.05): a perfect valve gains the most.

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": FixedEventProbability.from_params(0.1),
        ...         "p2": FixedEventProbability.from_params(0.1),
        ...         "v": FixedEventProbability.from_params(0.05),
        ...     },
        ... )
        >>> ip = rbd.improvement_potential()
        >>> {k: round(v, 4) for k, v in sorted(ip.items())}
        {'p1': 0.0095, 'p2': 0.0095, 'v': 0.0495}
        """
        self._require_no_ccf()
        node_probabilities, node_failures = self._importance_inputs(
            x, working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._improvement_potential(
                node_probabilities, node_failures=node_failures
            ),
        )

    @leaves_out_junctions
    @check_x
    def risk_achievement_worth(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Risk achievement worth (RAW) of each node at time/s ``x``.

        ``RAW_i = (1 - R_sys(i failed)) / (1 - R_sys)``, per Modarres &
        Kaminskiy: the factor by which the system unreliability would grow
        if node ``i`` were failed. It is at least 1. Where the system
        unreliability is 0 the division gives inf or nan, with a numpy
        RuntimeWarning. Exact; it assumes independent nodes.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: RAW}`` over the component nodes: floats for scalar
            ``x``, arrays for array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD, or a working/broken
            node is invalid (as for ``sf``).
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        Two pumps in parallel (each failing with probability 0.1) feeding a
        valve in series (0.05): a failed valve fails the system outright.

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": FixedEventProbability.from_params(0.1),
        ...         "p2": FixedEventProbability.from_params(0.1),
        ...         "v": FixedEventProbability.from_params(0.05),
        ...     },
        ... )
        >>> raw = rbd.risk_achievement_worth()
        >>> {k: round(v, 4) for k, v in sorted(raw.items())}
        {'p1': 2.437, 'p2': 2.437, 'v': 16.8067}
        """
        self._require_no_ccf()
        node_probabilities, node_failures = self._importance_inputs(
            x, working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._risk_achievement_worth(
                node_probabilities, node_failures=node_failures
            ),
        )

    @leaves_out_junctions
    @check_x
    def risk_reduction_worth(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Risk reduction worth (RRW) of each node at time/s ``x``.

        ``RRW_i = (1 - R_sys) / (1 - R_sys(i working))``, per Modarres &
        Kaminskiy: the factor by which the system unreliability would
        shrink if node ``i`` were made perfect. It is at least 1, and inf
        (with a numpy RuntimeWarning) where a perfect node ``i`` makes the
        system perfect, e.g. a node that alone forms a path. Exact; it
        assumes independent nodes.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: RRW}`` over the component nodes: floats for scalar
            ``x``, arrays for array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD, or a working/broken
            node is invalid (as for ``sf``).
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        Two pumps in parallel (each failing with probability 0.1) feeding a
        valve in series (0.05):

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": FixedEventProbability.from_params(0.1),
        ...         "p2": FixedEventProbability.from_params(0.1),
        ...         "v": FixedEventProbability.from_params(0.05),
        ...     },
        ... )
        >>> rrw = rbd.risk_reduction_worth()
        >>> {k: round(v, 4) for k, v in sorted(rrw.items())}
        {'p1': 1.19, 'p2': 1.19, 'v': 5.95}
        """
        self._require_no_ccf()
        node_probabilities, node_failures = self._importance_inputs(
            x, working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._risk_reduction_worth(
                node_probabilities, node_failures=node_failures
            ),
        )

    @leaves_out_junctions
    @check_x
    def criticality_importance(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        kind: str = "failure",
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Criticality importance of each node at time/s ``x``.

        With ``B_i`` the Birnbaum importance, ``R_i`` the node's reliability
        and ``R_sys`` the system's:

        - ``kind="failure"`` (the default) gives the failure-oriented form
          (Rausand & Høyland), ``CI_i = B_i * (1 - R_i) / (1 - R_sys)``:
          the probability that node ``i`` has failed and is critical, given
          that the system has failed -- the share of system failures that
          node ``i`` accounts for. It ranks nodes in series by how
          unreliable they are. It is computed from the node
          unreliabilities, through the minimal cut sets, so the system
          unreliability is not lost to cancellation in ``1 - R_sys``
          however reliable the system is. It is ``nan`` where the system
          cannot fail (e.g. at ``x = 0``).
        - ``kind="success"`` gives the success-oriented form,
          ``CI_i = B_i * R_i / R_sys``: the probability that node ``i`` is
          working and critical, given that the system works. It is 1 for
          every node in series with the rest of the system, however
          unreliable, so it cannot rank them. It is ``nan`` where the
          system cannot work.

        Exact; it assumes independent nodes.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.
        kind : str, optional
            ``"failure"`` (the default) or ``"success"``.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: criticality importance}`` over the component nodes:
            floats for scalar ``x``, arrays for array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD, a working/broken
            node is invalid (as for ``sf``), or ``kind`` is neither
            ``"failure"`` nor ``"success"``.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        References
        ----------
        M. Rausand and A. Høyland, System Reliability Theory: Models,
        Statistical Methods, and Applications, 2nd edition, Wiley, 2004.

        Examples
        --------
        Two pumps in parallel (each failing with probability 0.1) feeding a
        valve in series (0.05): the valve accounts for 83% of the system's
        failures (the shares add to more than 1 here, as both pumps are
        critical in the same failures).

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": FixedEventProbability.from_params(0.1),
        ...         "p2": FixedEventProbability.from_params(0.1),
        ...         "v": FixedEventProbability.from_params(0.05),
        ...     },
        ... )
        >>> ci = rbd.criticality_importance()
        >>> {k: round(v, 4) for k, v in sorted(ci.items())}
        {'p1': 0.1597, 'p2': 0.1597, 'v': 0.8319}

        The success-oriented form gives the valve, like any node in series,
        exactly 1:

        >>> ci = rbd.criticality_importance(kind="success")
        >>> {k: round(v, 4) for k, v in sorted(ci.items())}
        {'p1': 0.0909, 'p2': 0.0909, 'v': 1.0}
        """
        self._require_no_ccf()
        node_probabilities, node_failures = self._importance_inputs(
            x, working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._criticality_importance(
                node_probabilities, kind, node_failures=node_failures
            ),
        )

    @leaves_out_junctions
    @check_x
    def fussell_vesely(
        self,
        x: Optional[ArrayLike] = None,
        fv_type: str = "c",
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Fussell-Vesely importance of each node at time/s ``x``.

        With ``fv_type="c"`` (the default) this is the usual cut-set
        measure: the probability that a minimal cut set containing node
        ``i`` has failed, given that the system has failed. The numerator
        uses the rare-event approximation, summing the probabilities of
        those cut sets instead of taking their union:

            FV_i = sum over minimal cut sets C containing i of
                   prod over j in C of (1 - R_j), divided by (1 - R_sys)

        The sum over-estimates the union, so values can exceed 1 when the
        failure probabilities are not small. Each ``1 - R`` is worked out in
        its own right (a node's from its model's ``ff``, the system's as
        [`ff`][repyability.NonRepairableRBD.ff] is), so small ones keep
        their precision.

        With ``fv_type="p"`` the same formula is applied to the minimal path
        sets instead: the numerator sums, over the minimal path sets
        containing ``i``, the probability that every member of the path set
        has failed. This value is not bounded by 1: for a node in a parallel
        pair it is ``(1 - R_i) / (1 - R_sys)``.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        fv_type : str, optional
            ``"c"`` (the default) sums over the minimal cut sets and ``"p"``
            over the minimal path sets.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            ``{node: Fussell-Vesely importance}`` over the component nodes:
            floats for scalar ``x``, arrays for array ``x``.

        Raises
        ------
        ValueError
            If ``fv_type`` is not "c" or "p", ``x`` is omitted for a
            time-varying RBD, or a working/broken node is invalid (as for
            ``sf``).
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        Two pumps in parallel (each failing with probability 0.1) feeding a
        valve in series (0.05): the valve alone is a cut set, and it
        contributes most of the system's failure probability.

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": FixedEventProbability.from_params(0.1),
        ...         "p2": FixedEventProbability.from_params(0.1),
        ...         "v": FixedEventProbability.from_params(0.05),
        ...     },
        ... )
        >>> fv = rbd.fussell_vesely()
        >>> {k: round(v, 4) for k, v in sorted(fv.items())}
        {'p1': 0.1681, 'p2': 0.1681, 'v': 0.8403}
        """
        self._require_no_ccf()
        node_probabilities, node_failures = self._importance_inputs(
            x, working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._fussell_vesely(
                node_probabilities, fv_type, node_failures=node_failures
            ),
        )

    def fussel_vesely(
        self, x: Optional[ArrayLike] = None, fv_type: str = "c"
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Deprecated misspelt alias of ``fussell_vesely``.

        Deprecated: use
        [`fussell_vesely`][repyability.NonRepairableRBD.fussell_vesely]
        instead; this alias will be removed in 0.12. It returns
        ``fussell_vesely(x, fv_type)`` and, unlike it, takes no
        ``working_nodes``/``broken_nodes``.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        fv_type : str, optional
            ``"c"`` (the default) sums over the minimal cut sets and ``"p"``
            over the minimal path sets.

        Returns
        -------
        dict[Any, float | numpy.ndarray]
            As for ``fussell_vesely``.

        Warns
        -----
        FutureWarning
            On every call.

        Raises
        ------
        ValueError
            As for ``fussell_vesely``.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.
        """
        warnings.warn(
            "fussel_vesely() is deprecated; use fussell_vesely() "
            f"(Fussell-Vesely). This alias will be removed in {REMOVAL}.",
            FutureWarning,
            stacklevel=2,
        )
        return self.fussell_vesely(x, fv_type)

    def parameter_sensitivity(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        rel_step: float = 1e-5,
    ) -> Dict[Any, Dict[str, Union[float, np.ndarray]]]:
        """Sensitivity of system reliability to each node's parameters.

        For node ``i`` with parameter ``theta``, the sensitivity at time/s
        ``x`` is

        ``d R_sys / d theta = B_i(x) * d sf_i(x; theta) / d theta``

        where ``B_i`` is the Birnbaum importance of node ``i`` (how much the
        system reliability moves per unit change in that node's reliability)
        and ``d sf_i / d theta`` is how much the node's reliability moves per
        unit change in the parameter. The parameter derivative is taken
        numerically by a central finite difference with step
        ``rel_step * |theta|`` (``rel_step`` if ``theta`` is 0), rebuilding
        the distribution with ``from_params``, so it applies to any
        parametric surpyval model without per-distribution formulae. If one
        perturbation is not a valid parameter value (e.g. a probability
        leaving [0, 1]) a one-sided difference is used; if neither is, the
        result is nan. It answers "which fitted parameter, if it were a
        little different, would move system reliability the most" -- e.g.
        to target data collection or to gauge the impact of estimation
        uncertainty.

        Only nodes with reconstructable surpyval distribution parameters
        are included, fixed-probability nodes among them (their parameter
        is the failure probability). Composite nodes (a nested RBD, a
        standby, load-sharing or repeated node, a regression node), fitted
        non-parametric models and the input and output nodes have no
        parameters to perturb and are omitted. A node forced via
        ``working_nodes``/``broken_nodes`` is pinned independently of its
        parameters, so its sensitivities are reported as zero.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        working_nodes : Collection[Hashable], optional
            Nodes to treat as working (reliability 1), by default none.
        broken_nodes : Collection[Hashable], optional
            Nodes to treat as failed (reliability 0), by default none.
        rel_step : float, optional
            Relative step used for the finite difference, by default ``1e-5``.

        Returns
        -------
        dict[Any, dict[str, float | numpy.ndarray]]
            ``{node: {parameter_name: sensitivity}}``, with surpyval's
            parameter names (e.g. ``"alpha"`` and ``"beta"`` for a Weibull;
            ``"param0"``, ``"param1"``, ... if it gives none). Sensitivities
            are floats for scalar ``x`` and arrays for array ``x``.

        Raises
        ------
        ValueError
            If ``x`` is omitted for a time-varying RBD, or a working/broken
            node is invalid (as for ``sf``).
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        For a single exponential component ``R = exp(-rate * x)``, so
        ``dR/d rate = -x * exp(-rate * x)``, about -9.048 at ``x = 10``
        with rate 0.01:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> sens = rbd.parameter_sensitivity(10)
        >>> {name: round(v, 3) for name, v in sens["c"].items()}
        {'failure_rate': -9.048}
        """
        if x is None:
            if self.is_fixed:
                x = 1.0
            else:
                raise ValueError(
                    "x is required: this RBD is time-varying (at least one "
                    "node model's probability depends on time)."
                )
        scalar_in = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        # Birnbaum importance validates the override sets and honours them.
        birnbaum = self.birnbaum_importance(x_arr, working_nodes, broken_nodes)
        forced = set(working_nodes or ()) | set(broken_nodes or ())

        def _out(value) -> Union[float, np.ndarray]:
            arr = np.asarray(value, dtype=float)
            return float(arr.reshape(-1)[0]) if scalar_in else arr

        sensitivities: Dict[Any, Dict[str, Union[float, np.ndarray]]] = {}
        for node_name, model in self.reliabilities.items():
            spec = parametric_spec(model)
            if spec is None:
                # Composite / non-parametric node: no parameters to perturb.
                continue
            cls, params, names, extras = spec
            node_out: Dict[str, Union[float, np.ndarray]] = {}
            if node_name in forced:
                # Pinned regardless of its parameters -> zero sensitivity.
                zero = np.zeros_like(x_arr)
                for name in names:
                    node_out[name] = _out(zero)
                sensitivities[node_name] = node_out
                continue
            b_i = np.asarray(birnbaum[node_name], dtype=float)
            for j, name in enumerate(names):
                dsf = _dsf_dparam(cls, params, j, x_arr, rel_step, extras)
                node_out[name] = _out(b_i * dsf)
            sensitivities[node_name] = node_out
        return sensitivities
