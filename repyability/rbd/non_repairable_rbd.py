"""Reliability block diagrams of non-repairable components.

Defines ``NonRepairableRBD``, which computes the reliability of a system of
non-repairable components from its block diagram, and the small helpers it
uses (``check_x`` and the ``NodeFailure`` simulation event).
"""

import functools
import math
import pprint
import warnings
from copy import copy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import (
    Any,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Optional,
    Union,
    cast,
)

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm
from surpyval import NonParametric

from repyability.utils.wrappers import conditional_survival, numpy_seed

from . import redundancy_allocation
from ._model_utils import is_fixed_probability, parametric_spec
from ._sampling import RowSampler, row_sampler
from .ccf import CCFGroup
from .helper_classes import PerfectReliability, PerfectUnreliability
from .load_sharing_node import LoadSharingModel
from .node_state import NodeState
from .rbd import RBD
from .repeated_node import RepeatedNode
from .repeated_standby_node import RepeatedStandbyNode
from .results import ConfidenceInterval, RedundancyAllocation
from .standby_node import StandbyModel


# Event class for simulation
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
        if x is None:
            if obj.is_fixed:
                x = 1.0
            else:
                raise ValueError(
                    "x is required: this RBD is time-varying (at least one "
                    "node model's probability depends on time)."
                )
        scalar_in = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = func(obj, x, *args, **kwargs)
        if scalar_in:
            if isinstance(result, dict):
                return {k: np.asarray(v).item() for k, v in result.items()}
            return np.asarray(result).item()
        return result

    return wrap


def _dsf_dparam(cls, params, j, x_arr, rel_step) -> np.ndarray:
    """Partial derivative of a distribution's ``sf`` at ``x_arr`` with respect
    to its ``j``-th parameter, by finite difference.

    ``cls.from_params`` rebuilds the distribution with a perturbed parameter,
    so this works for any surpyval parametric distribution without hard-coding
    per-distribution derivative formulae. A central difference is used where
    both perturbations are valid; if one perturbation falls outside a
    parameter's admissible range (e.g. a probability leaving ``[0, 1]``) it
    falls back to a one-sided difference about the unperturbed value.
    """
    theta = params[j]
    h = rel_step * abs(theta) if theta != 0.0 else rel_step

    def perturbed(delta):
        trial = list(params)
        trial[j] = theta + delta
        try:
            return np.asarray(cls.from_params(trial).sf(x_arr), dtype=float)
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
        Its edges are redirected to the node it repeats, so both places
        share one component. That node cannot itself be a repeat, and the
        redirected edges must not form a cycle. The input and output nodes
        need no entry: they are always perfectly reliable, and a model
        given for them is replaced.
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
        not in the diagram raises a ValueError.
    output_node : Hashable, optional
        The output node. By default it is inferred as the unique node with
        no successors; naming it does not relax that requirement. A name
        not in the diagram raises a ValueError.
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
    nodes : list
        The component nodes: every node except the input and output nodes
        (repeated nodes are merged into the node they repeat).
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
        diagram, the diagram is invalid (with ``on_infeasible_rbd="raise"``)
        or ``ccf_groups`` is invalid.

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
    ):
        if on_infeasible_rbd not in ["raise", "warn", "ignore"]:
            raise ValueError(
                "'on_infeasible_rbd' must be one of"
                + " {'raise', 'warn', 'ignore'}"
            )
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

        if repeated == {}:
            super().__init__(
                edges,
                set(reliabilities.keys()),
                k,
                input_node,
                output_node,
                on_infeasible_rbd,
            )
            self.structure_check["has_repeated_node_in_cycle"] = False
        else:
            new_edges = []
            for start, stop in edges:
                if start in repeated:
                    start = repeated[start]
                if stop in repeated:
                    stop = repeated[stop]
                new_edges.append((start, stop))
            super().__init__(
                new_edges,
                set(reliabilities.keys()),
                k,
                input_node,
                output_node,
                on_infeasible_rbd,
            )
            self.structure_check["has_repeated_node_in_cycle"] = False
            if self.structure_check["has_cycles"]:
                # Need to find if cycles are due to repeated components.
                G = nx.DiGraph()
                G.add_edges_from(edges)
                cycles = {
                    frozenset(cycle) for cycle in list(nx.simple_cycles(G))
                }
                non_repeated_node_cycles = copy(self.structure_check["cycles"])
                for cycle in self.structure_check["cycles"]:
                    if cycle not in cycles:
                        non_repeated_node_cycles.remove(cycle)
                        self.structure_check["has_repeated_node_in_cycle"] = (
                            True
                        )
                if len(non_repeated_node_cycles) == 0:
                    self.structure_check["has_cycles"] = False
                self.structure_check["cycles"] = non_repeated_node_cycles

        # Check for repeated cycles or non-repeated cycles
        if self.structure_check["has_unique_input_node"]:
            reliabilities[self.input_node] = PerfectReliability
        if self.structure_check["has_unique_output_node"]:
            reliabilities[self.output_node] = PerfectReliability

        # Check that all nodes in graph were in the reliabilities dict
        # Checking that all in the reliabilities dict are in the graph
        # is done in RBD initialisation since the RBD adds nodes from the
        # reliabilities dict and checks if they are connected.
        self.structure_check["is_missing_distributions"] = False
        self.structure_check["nodes_with_no_reliability_distribution"] = []
        for n in self.G.nodes:
            if n not in reliabilities:
                self.structure_check["is_valid"] = False
                self.structure_check["is_missing_distributions"] = True
                self.structure_check[
                    "nodes_with_no_reliability_distribution"
                ].append(n)

        if not self.structure_check["is_valid"]:
            if on_infeasible_rbd == "warn":
                warnings.warn(
                    "Strucutral Errors in RBD:\n"
                    + pprint.pformat(self.structure_check),
                    stacklevel=2,
                )
            elif on_infeasible_rbd == "raise":
                raise ValueError("RBD not correctly structured")
            elif on_infeasible_rbd == "ignore":
                pass

        self.reliabilities = reliabilities
        self.repeated = repeated
        self.ccf_groups = self._validate_ccf_groups(ccf_groups)

        fixed_flags = []
        for _, node in self.reliabilities.items():
            if isinstance(node, NonParametric):
                fixed_flags = [False]
                break
            elif isinstance(node, NonRepairableRBD):
                fixed_flags.append(node.is_fixed)
            elif node == PerfectReliability:
                continue
            elif node == PerfectUnreliability:
                continue
            else:
                # when node is a Parametric model
                if isinstance(node, (StandbyModel, LoadSharingModel)):
                    fixed_flags = [False]
                    break
                elif isinstance(node, RepeatedNode):
                    fixed_flags.append(is_fixed_probability(node.model))
                else:
                    fixed_flags.append(is_fixed_probability(node))

        self._fixed_probs: bool = all(fixed_flags)
        self.structure_check["all_distributions_fixed"] = self._fixed_probs

        # Record whether the system reliability can be solved analytically
        # (equivalently with the BDD), or whether it requires simulation
        # because one or more nodes are simulation-based (e.g. standby nodes).
        non_analytic_nodes = self.get_non_analytic_nodes()
        self.structure_check["is_analytically_solvable"] = (
            len(non_analytic_nodes) == 0
        )
        self.structure_check["non_analytic_nodes"] = non_analytic_nodes

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
        from itertools import product

        forced = working_nodes | broken_nodes
        for group in self.ccf_groups:
            if forced.intersection(group.members):
                raise NotImplementedError(
                    "Forcing a CCF group member via working_nodes / "
                    "broken_nodes is not supported yet."
                )

        # Each group's mutually-exclusive shock outcomes: (weight, {member:
        # reliability}) for every subset that can fail together plus the
        # no-shock case, from the model's decomposition of Q(t) (taken from a
        # representative member, since groups are symmetric).
        group_outcomes = []
        for group in self.ccf_groups:
            Q = 1.0 - np.atleast_1d(base_probabilities[group.members[0]])
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
                    )
                )
            # No common-cause shock: every member fails only independently.
            outcomes.append(
                (
                    1.0 - total_shock,
                    {member: r_independent for member in group.members},
                )
            )
            group_outcomes.append(outcomes)

        terms = []
        for combo in product(*group_outcomes):
            node_probabilities = dict(base_probabilities)
            weight: Any = 1.0
            for outcome_weight, member_probs in combo:
                weight = weight * outcome_weight
                node_probabilities.update(member_probs)
            terms.append(
                np.asarray(weight)
                * np.asarray(
                    self.system_probability(node_probabilities, method=method)
                )
            )
        return np.sum(terms, axis=0)

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

    def ff(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        """System unreliability (failure probability) at time/s ``x``.

        ``1 - sf(x)``: the probability that the system has failed by
        ``x``. Exact, and honours common-cause groups, as
        [`sf`][repyability.NonRepairableRBD.sf] does.

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
        """
        return 1 - self.sf(x, *args, **kwargs)

    def allocate_redundancy(
        self,
        costs: Dict[Hashable, float],
        *,
        budget: Optional[float] = None,
        target: Optional[float] = None,
        t: Optional[float] = None,
        max_units: Union[int, Dict[Hashable, int], None] = None,
        method: str = "exact",
    ) -> RedundancyAllocation:
        """Choose how many redundant copies of each node to fit.

        Solves the Redundancy Allocation Problem: pick how many identical,
        independent copies of each node in ``costs`` to fit in active
        parallel, to either

        - maximise system reliability with total cost at most ``budget``, or
        - minimise total cost with system reliability at least ``target``.

        Give exactly one of ``budget`` or ``target``. A node with reliability
        ``p`` fitted as ``n`` copies has reliability ``1 - (1 - p) ** n``, and
        the system reliability of each candidate allocation is computed
        exactly, so any RBD structure works. Nodes not in ``costs`` stay as
        they are. Reliability is evaluated at the single mission time ``t``.

        Parameters
        ----------
        costs : dict
            ``{node: cost per copy}`` for the nodes that may be duplicated.
            Every copy is costed, including the original, so the cheapest
            allocation (one of each) costs ``sum(costs.values())``. The
            "cost" can be any additive resource, e.g. weight or volume. Each
            cost must be finite and positive.
        budget : float, optional
            Maximise reliability subject to total cost <= budget. It must
            be at least ``sum(costs.values())``.
        target : float, optional
            Minimise cost subject to system reliability >= target, in (0, 1).
        t : float, optional
            The mission time at which reliability is evaluated, a single
            number. Required for a time-varying RBD; not needed when every
            node is a fixed probability.
        max_units : int or dict, optional
            The most copies allowed (at least 1) of every costed node (an
            int) or of particular costed nodes (a dict; nodes it leaves out
            are unlimited), e.g. for space limits. By default unlimited; the
            budget or target bounds the search.
        method : str, optional
            ``"exact"`` (the default) searches for a proven optimum and is
            fast for typical problems (a handful of nodes); it gives up
            with an explanatory error on problems too large to search
            (more than 500,000 allocations examined). ``"greedy"``
            repeatedly adds the copy with the best log-reliability gain per
            unit cost: fast for any size, usually optimal or close, but not
            guaranteed optimal.

        Returns
        -------
        RedundancyAllocation
            The chosen ``units`` per costed node, with the resulting
            ``reliability`` and total ``cost``, and the ``method`` used (see
            [`RedundancyAllocation`][repyability.RedundancyAllocation]).

        Raises
        ------
        ValueError
            If both or neither of ``budget`` and ``target`` are given, or on
            any other invalid input: an unknown ``method``; empty ``costs``;
            a costed node that is not a component node or is a repeated
            node; a cost that is not finite and positive; an invalid
            ``max_units``; ``t`` missing for a time-varying RBD or not a
            single number; a ``budget`` that cannot afford one of each
            costed node; a ``target`` outside (0, 1) or not reachable
            (within ``max_units``); or an exact search that is too large.
        NotImplementedError
            If the RBD has common-cause (CCF) groups.

        Examples
        --------
        Two components in series, 90% and 80% reliable, one cost unit each.
        With a budget of 3 the extra copy goes to the weaker component:

        >>> from surpyval import FixedEventProbability
        >>> from repyability import NonRepairableRBD
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
        """
        if (budget is None) == (target is None):
            raise ValueError("Give exactly one of budget or target.")
        if method not in ("exact", "greedy"):
            raise ValueError(
                f"method must be 'exact' or 'greedy', got {method!r}."
            )
        if self.ccf_groups:
            raise NotImplementedError(
                "Redundancy allocation does not yet account for common-cause "
                "(CCF) groups: duplicating a group member would also have to "
                "extend its common-cause group."
            )
        if not costs:
            raise ValueError("costs must name at least one node.")
        nodes = list(costs)
        unit_costs = []
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
            cost = float(costs[node])
            if not np.isfinite(cost) or cost <= 0.0:
                raise ValueError(
                    f"The cost of node {node!r} must be finite and positive, "
                    f"got {costs[node]!r}."
                )
            unit_costs.append(cost)

        caps = self._redundancy_caps(nodes, max_units)
        if t is not None and np.ndim(t) != 0:
            raise ValueError("t must be a single mission time.")
        if t is None:
            if self.is_time_varying:
                raise ValueError(
                    "t (the mission time) is required: this RBD is "
                    "time-varying, so its reliability depends on when it is "
                    "evaluated."
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
        cache: Dict[tuple, float] = {}

        def evaluate(units: tuple) -> float:
            if units not in cache:
                probabilities = dict(base)
                for node, n in zip(nodes, units):
                    probabilities[node] = 1.0 - (1.0 - base[node]) ** n
                cache[units] = float(
                    np.ravel(self.system_probability(probabilities))[0]
                )
            return cache[units]

        if budget is not None:
            budget = float(budget)
            minimum = math.fsum(unit_costs)
            if not np.isfinite(budget) or budget < minimum:
                raise ValueError(
                    f"budget must be finite and at least {minimum:g}, the "
                    "cost of one of each costed node; got "
                    f"{budget!r}."
                )
            if method == "exact":
                found = redundancy_allocation.exact_max_reliability(
                    evaluate, unit_costs, caps, budget
                )
            else:
                found = redundancy_allocation.greedy(
                    evaluate, unit_costs, caps, budget=budget
                )
        else:
            assert target is not None
            target = float(target)
            if not (0.0 < target < 1.0):
                raise ValueError(f"target must be in (0, 1), got {target!r}.")
            # The best reliability any allocation can reach: every costed node
            # at its cap, where 1 - (1 - p) ** inf gives the unlimited limit.
            ceiling = evaluate(tuple(caps))
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
            found = redundancy_allocation.greedy(
                evaluate, unit_costs, caps, target=target
            )
            if found[0] < target:
                raise ValueError(
                    f"target {target:g} could not be reached: adding copies "
                    f"stopped improving the system at reliability "
                    f"{found[0]:.6g}."
                )
            if method == "exact":
                found = redundancy_allocation.exact_min_cost(
                    evaluate, unit_costs, caps, target, found
                )

        reliability, total_cost, units = found
        return RedundancyAllocation(
            units={node: int(n) for node, n in zip(nodes, units)},
            reliability=reliability,
            cost=total_cost,
            method=method,
        )

    @staticmethod
    def _redundancy_caps(nodes, max_units) -> list:
        """Per-node copy limits for allocate_redundancy (inf = unlimited)."""
        if max_units is None:
            return [math.inf] * len(nodes)
        if isinstance(max_units, dict):
            unknown = set(max_units) - set(nodes)
            if unknown:
                raise ValueError(
                    "max_units names node(s) not in costs: "
                    f"{sorted(map(str, unknown))}."
                )
            limits = {n: max_units.get(n, math.inf) for n in nodes}
        else:
            limits = {n: max_units for n in nodes}
        for node, limit in limits.items():
            if limit is math.inf:
                continue
            if isinstance(limit, bool) or not isinstance(
                limit, (int, np.integer)
            ):
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
            limits[n] if limits[n] is math.inf else int(limits[n])
            for n in nodes
        ]

    def unreliability(self, x: Optional[ArrayLike] = None, *args, **kwargs):
        """System unreliability at time/s ``x``; the same as ``ff``.

        ``1 - sf(x)``, the probability that the system has failed by ``x``
        (see [`ff`][repyability.NonRepairableRBD.ff]).

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
            The system unreliability: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            As for ``sf``.
        NotImplementedError
            As for ``sf``.
        """
        return 1 - self.sf(x, *args, **kwargs)

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
    def Hf(self, x: Optional[ArrayLike] = None, **kwargs) -> np.ndarray:
        """System cumulative hazard ``H(x) = -ln R(x)`` at time/s ``x``.

        Exact given the exact system reliability ``R(x)`` from
        [`sf`][repyability.NonRepairableRBD.sf] (so it honours
        common-cause groups); +inf wherever the system reliability has
        reached zero.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD.
        **kwargs
            Keyword arguments of ``sf`` (``working_nodes``,
            ``broken_nodes``, ``method``).

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
        sf = np.asarray(self.sf(x, **kwargs), dtype=float)
        with np.errstate(divide="ignore"):
            return -np.log(sf)

    @check_x
    def df(
        self, x: Optional[ArrayLike] = None, dx: float = 1e-6, **kwargs
    ) -> np.ndarray:
        """System failure density ``f(x) = -dR/dx`` at time/s ``x``.

        Computed by a central finite difference of the system reliability
        from [`sf`][repyability.NonRepairableRBD.sf] (so it honours
        common-cause groups), with step ``h = dx * max(|x|, 1)``: relative
        to ``x`` for ``|x| >= 1``, absolute below. The lower point is
        clipped at 0, so the step never crosses into negative time (the
        difference is one-sided near 0), and negative results (numerical
        noise) are clipped to 0. It assumes ``R`` is smooth near ``x``,
        which does not hold for step-function node models such as a
        Kaplan-Meier fit.

        Parameters
        ----------
        x : array_like, optional
            Time/s as a number or an array. May be omitted only for a
            fixed-probability RBD (whose density is 0).
        dx : float, optional
            Relative finite-difference step, by default 1e-6.
        **kwargs
            Keyword arguments of ``sf`` (``working_nodes``,
            ``broken_nodes``, ``method``).

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
        density = (
            np.asarray(self.sf(x_lo, **kwargs), dtype=float)
            - np.asarray(self.sf(x_hi, **kwargs), dtype=float)
        ) / (x_hi - x_lo)
        return np.clip(density, 0.0, None)

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

    @property
    def is_time_varying(self) -> bool:
        """Whether the system reliability varies with time.

        The complement of
        [`is_fixed`][repyability.NonRepairableRBD.is_fixed]: True when at
        least one node's reliability depends on time, in which case the
        time argument ``x`` must be given.
        """
        return not self._fixed_probs

    # Dynamic (sequence-dependent) node model types. Depending on the node,
    # their sf(t) is a closed form (identical exponential units), a numerical
    # convolution or a Kaplan-Meier fit to simulated lifetimes; any of them is
    # treated as preventing a purely analytic / BDD solution of the system
    # (see is_analytically_solvable).
    _SIMULATION_NODE_TYPES = (
        StandbyModel,
        RepeatedStandbyNode,
        LoadSharingModel,
    )

    def _node_is_analytic(self, model) -> bool:
        """Returns True if a node's reliability is available without
        Monte-Carlo simulation (i.e. in closed form or from data), and so can
        be consumed directly by the analytic / BDD system probability.

        The check recurses through RepeatedNodes (analytic iff their underlying
        model is) and nested NonRepairableRBDs (analytic iff they are
        themselves analytically solvable).
        """
        # Perfect reliability / unreliability are constants
        if model is PerfectReliability or model is PerfectUnreliability:
            return True
        # Standby arrangements are simulation-based (KM fit) -> non-analytic
        if isinstance(model, self._SIMULATION_NODE_TYPES):
            return False
        # A repeated node is analytic iff its underlying model is
        if isinstance(model, RepeatedNode):
            return self._node_is_analytic(model.model)
        # A nested RBD is analytic iff it is itself analytically solvable
        if isinstance(model, NonRepairableRBD):
            return model.is_analytically_solvable()
        # Otherwise it is a surpyval parametric/non-parametric distribution
        # (incl. FixedEventProbability), all of which expose a usable sf(t)
        # without simulation.
        return True

    def get_non_analytic_nodes(self) -> dict[Any, str]:
        """The nodes that prevent a purely analytic solution.

        A node is non-analytic when its model is a
        [`StandbyModel`][repyability.StandbyModel],
        [`RepeatedStandbyNode`][repyability.RepeatedStandbyNode] or
        [`LoadSharingModel`][repyability.LoadSharingModel], or a
        [`RepeatedNode`][repyability.RepeatedNode] or nested
        ``NonRepairableRBD`` containing one (see
        ``is_analytically_solvable``).

        Returns
        -------
        dict[Any, str]
            ``{node: type name of its model}`` for every non-analytic node,
            e.g. ``{"a": "StandbyModel"}``. Empty if the RBD is
            analytically solvable.
        """
        non_analytic: dict[Any, str] = {}
        for node_name, model in self.reliabilities.items():
            if not self._node_is_analytic(model):
                non_analytic[node_name] = type(model).__name__
        return non_analytic

    def is_analytically_solvable(self) -> bool:
        """Whether every node is of an analytic (non-dynamic) model type.

        The exact system reliability (``sf`` and the methods built on it)
        is only as accurate as each node's own ``sf(t)``. That is a closed
        form or data for surpyval distributions (parametric,
        non-parametric or fixed-probability), for
        [`RegressionNode`][repyability.RegressionNode] and the perfect
        nodes, for repeated nodes of such models, and for nested RBDs that
        are themselves analytically solvable.

        Standby and load-sharing arrangements
        ([`StandbyModel`][repyability.StandbyModel],
        [`RepeatedStandbyNode`][repyability.RepeatedStandbyNode],
        [`LoadSharingModel`][repyability.LoadSharingModel]) are
        sequence-dependent, so their ``sf(t)`` is, depending on the node, a
        closed form (identical exponential units), a numerical convolution
        or a Kaplan-Meier fit to Monte-Carlo lifetimes (a step function
        bounded by the simulated support). This check is by type only: any
        such node, or a repeated node or nested RBD containing one, makes it
        False, even when the node's ``sf`` is a closed form. ``sf`` still
        returns a value either way; for such systems simulating the whole
        system (``random``, ``mean``) avoids the per-node approximation.

        The result is also stored at construction in
        ``structure_check["is_analytically_solvable"]``.

        Returns
        -------
        bool
            True if no node is of a standby or load-sharing type; False
            otherwise (``get_non_analytic_nodes`` lists which).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, StandbyModel
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> rbd = NonRepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {"a": StandbyModel([unit, unit]), "b": unit},
        ... )
        >>> rbd.is_analytically_solvable()
        False
        >>> rbd.get_non_analytic_nodes()
        {'a': 'StandbyModel'}
        """
        return len(self.get_non_analytic_nodes()) == 0

    def random(self, size, seed=None):
        """Draw ``size`` random system lifetimes (Monte-Carlo).

        Each node's lifetime is drawn independently from its own model's
        ``random``, and the system fails when its last working minimal path
        set breaks: each sample is the maximum, over the minimal path sets,
        of the minimum lifetime of the path set's members (k-out-of-n and
        repeated nodes are handled through the path sets). When every
        node's draws can be replayed as one block (e.g. surpyval parametric
        distributions, and composite nodes built from them), all samples
        are computed at once; otherwise they are simulated one at a time by
        failing nodes in time order until the system fails. Both paths draw
        the same random numbers in the same order, so they give identical
        results.

        Common-cause groups are ignored, without a warning: their
        basic-event model assumes a small failure probability, while a
        lifetime runs to ``Q = 1``. The same applies to ``mean``,
        ``mean_time_to_failure`` and ``mean_time_to_failure_interval``.
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

        Returns
        -------
        numpy.ndarray
            ``size`` system lifetimes; ``inf`` in a sample where the system
            never fails (e.g. through a perfectly reliable path).

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
        with numpy_seed(seed):
            fast = self._random_vectorised(size)
            if fast is not None:
                return fast
            return self._random_by_events(size)

    def _random_vectorised(self, size) -> Optional[np.ndarray]:
        """``random(size)`` without the per-sample event loop, when every
        node's draws can be replayed in one block (see :meth:`_row_sampler`);
        otherwise ``None`` (nothing is drawn)."""
        sampler = self._row_sampler()
        if sampler is None:
            return None
        state = np.random.get_state()
        out = sampler.draw(np.random.random_sample((size, sampler.width)))
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
        members' lifetimes: the same value the event loop finds. A sample in
        which any node drew NaN comes out NaN, so that the caller falls back
        to the event loop, which orders NaN times its own way.
        """
        nodes = list(self.G.nodes)
        samplers: list[RowSampler] = []
        for node in nodes:
            node_sampler = row_sampler(self.reliabilities[node])
            if node_sampler is None:
                return None
            samplers.append(node_sampler)
        path_sets = self.get_min_path_sets(include_in_out_nodes=False)

        def draw(u):
            size = len(u)
            lifetimes, start = {}, 0
            for node, sampler in zip(nodes, samplers):
                end = start + sampler.width
                lifetimes[node] = sampler.draw(u[:, start:end])
                start = end
            out = np.full(size, -np.inf)
            for path_set in path_sets:
                path_life = np.full(size, np.inf)
                for node in path_set:
                    path_life = np.minimum(path_life, lifetimes[node])
                out = np.maximum(out, path_life)
            for lifetime in lifetimes.values():
                out[np.isnan(lifetime)] = np.nan
            return out

        return RowSampler(sum(s.width for s in samplers), draw)

    def _random_by_events(self, size) -> np.ndarray:
        """``random(size)`` by stepping through each sample's failures in
        time order until the system fails; works for any node model."""
        out = np.zeros(size)
        for i in range(size):
            event_queue: PriorityQueue = PriorityQueue()
            for node in self.G.nodes:
                # .random(1) returns a 1-element array; take the scalar so
                # the event time orders the PriorityQueue and assigns into
                # ``out`` (NumPy >= 2 rejects assigning a 1-element array to
                # a scalar).
                draw = np.asarray(self.reliabilities[node].random(1))
                time = float(draw.reshape(-1)[0])
                event_queue.put(NodeFailure(time, node))

            working_nodes = {k: True for k in self.G.nodes}
            system_working = True
            while system_working:
                failure = event_queue.get()
                time = failure.time
                working_nodes[failure.node] = False
                system_working = self.is_system_working(
                    working_nodes, method="p"
                )
            out[i] = time
        return out

    def mean(self, mc_samples: int = 100_000, seed=None):
        """Mean time to failure (MTTF) of the system, by Monte-Carlo.

        The average of ``random(mc_samples, seed=seed)`` (see
        [`random`][repyability.NonRepairableRBD.random]), so it ignores
        common-cause groups and is not meaningful for fixed-probability
        nodes. The estimate's standard error is the lifetimes' standard
        deviation over ``sqrt(mc_samples)``; use
        ``mean_time_to_failure_interval`` to get it with a confidence
        interval. This is also the MTTF used for this RBD when it is
        nested as a node of another RBD (see ``node_mttf``).

        Parameters
        ----------
        mc_samples : int, optional
            Number of system lifetimes to simulate, by default 100_000.
        seed : int or None, optional
            Seed for reproducibility (see ``random``), by default None.

        Returns
        -------
        float
            The MTTF estimate.

        Examples
        --------
        A single exponential component with failure rate 0.01 has an
        exact MTTF of 100:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD
        >>> rbd = NonRepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": surv.Exponential.from_params([0.01])},
        ... )
        >>> print(f"{rbd.mean(mc_samples=10_000, seed=1):.1f}")
        98.9
        """
        return self.random(mc_samples, seed=seed).mean().item()

    def mean_time_to_failure(self, mc_samples: int = 100_000, seed=None):
        """Mean time to failure (MTTF) of the system; the same as ``mean``.

        A Monte-Carlo estimate from ``mc_samples`` simulated lifetimes (see
        [`mean`][repyability.NonRepairableRBD.mean]); common-cause groups
        are ignored.

        Parameters
        ----------
        mc_samples : int, optional
            Number of system lifetimes to simulate, by default 100_000.
        seed : int or None, optional
            Seed for reproducibility (see ``random``), by default None.

        Returns
        -------
        float
            The MTTF estimate.
        """
        return self.mean(mc_samples, seed=seed)

    def mean_time_to_failure_interval(
        self,
        mc_samples: int = 100_000,
        confidence: float = 0.95,
        seed=None,
    ) -> ConfidenceInterval:
        """Monte-Carlo MTTF estimate with a confidence interval.

        The MTTF is the mean of ``mc_samples`` simulated system lifetimes
        (see [`random`][repyability.NonRepairableRBD.random]; common-cause
        groups are ignored). By the central limit theorem its sampling
        error is normal with standard error
        ``sample std / sqrt(mc_samples)``, from which the two-sided
        interval ``estimate +/- z * standard_error`` is built; the lower
        bound is clipped at 0. The interval describes the simulation
        error only, not uncertainty in the node models.

        Parameters
        ----------
        mc_samples : int, optional
            Number of Monte-Carlo samples, by default 100_000.
        confidence : float, optional
            The confidence level, in (0, 1), by default 0.95.
        seed : int or None, optional
            Seed for reproducibility (see ``random``), by default None.

        Returns
        -------
        ConfidenceInterval
            The estimate, bounds, confidence level, standard error and
            sample count (see
            [`ConfidenceInterval`][repyability.ConfidenceInterval]).

        Raises
        ------
        ValueError
            If ``confidence`` is not in (0, 1).

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
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        samples = self.random(mc_samples, seed=seed)
        estimate = float(samples.mean())
        standard_error = float(samples.std(ddof=1) / np.sqrt(len(samples)))
        z = float(norm.ppf(0.5 + confidence / 2.0))
        return ConfidenceInterval(
            estimate=estimate,
            lower=max(0.0, estimate - z * standard_error),
            upper=estimate + z * standard_error,
            confidence=confidence,
            standard_error=standard_error,
            n_samples=mc_samples,
        )

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
        if self.is_fixed:
            raise ValueError(
                "System reliability does not vary with time (all nodes are "
                "fixed-probability); the time to a reliability is undefined."
            )

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

    def _validate_state(self, state) -> None:
        """Validate a condition-based ``state`` mapping.

        Raises rather than silently ignoring bad input: a non-mapping, a value
        that is not a :class:`NodeState`, the input/output node, an unknown
        node, or a node whose model is composite/dynamic.
        """
        if not isinstance(state, dict):
            raise TypeError(
                "state must be a dict of {node: NodeState}, got "
                f"{type(state).__name__}."
            )
        valid = set(self.nodes)
        for node, node_state in state.items():
            if not isinstance(node_state, NodeState):
                raise TypeError(
                    f"state[{node!r}] must be a NodeState, got "
                    f"{type(node_state).__name__}."
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

    def _state_node_probabilities(self, x, state) -> Dict[Any, ArrayLike]:
        """Per-node forward reliability at ``x`` given each node's ``state``.

        ``x`` is a 1-d array. Each node conditions on its own current life: a
        node absent from ``state`` uses its unconditioned reliability (age 0),
        a failed node contributes zero, and an alive node of age ``X``
        contributes ``conditional_survival(model, x, X) = sf(X + x) / sf(X)``.
        """
        if self.ccf_groups:
            raise NotImplementedError(
                "Condition-based evaluation (sf_given_state / remaining_life "
                "/ importances_given_state) does not yet account for "
                "common-cause (CCF) groups; use sf()/ff() for CCF system "
                "reliability."
            )
        self._validate_state(state)
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
        if state is None:
            state = {}
        return self._invert_reliability(
            lambda t: float(self.sf_given_state(t, state)),
            target,
            upper_bound,
        )

    def importances_given_state(
        self,
        x: Optional[ArrayLike] = None,
        state: Optional[Dict[Hashable, NodeState]] = None,
    ) -> Dict[str, Dict[Any, Union[float, np.ndarray]]]:
        """Birnbaum and criticality importance given each node's state.

        Shows how each component's importance over a forward horizon ``x``
        shifts once the current wear on every component is accounted for.
        The Birnbaum and criticality importance measures are evaluated at
        the conditioned node reliabilities ``R_i(x | X_i)`` (see
        [`sf_given_state`][repyability.NonRepairableRBD.sf_given_state])
        rather than at the as-new reliabilities, so the rankings reflect
        the current state:

            birnbaum_i    = R_sys(x | state, i working)
                            - R_sys(x | state, i failed)
            criticality_i = birnbaum_i * R_i(x | X_i) / R_sys(x | state)

        These are the conventions of ``birnbaum_importance`` and
        ``criticality_importance``: they measure how much the system
        reliability depends on each node *now*, not which node is most
        likely to have failed. A failed node's criticality is 0.

        Parameters
        ----------
        x : array_like, optional
            The forward horizon/s over which importance is evaluated (from
            now). May be omitted only for a fixed-probability RBD.
        state : dict[Hashable, NodeState], optional
            ``{node: NodeState}``, the current state of some or all of the
            component nodes (see ``sf_given_state``). By default empty (all
            nodes new).

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
            If ``x`` is omitted for a time-varying RBD, or ``state`` is
            invalid (as for ``sf_given_state``).
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
        criticality = super()._criticality_importance(node_probabilities)

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
        self, mc_samples: int = 100_000, seed=None
    ) -> dict[Any, float]:
        """Mean time to failure (MTTF) of each component node.

        Each node's MTTF comes from its own model:

        - a [`StandbyModel`][repyability.StandbyModel] or
          [`LoadSharingModel`][repyability.LoadSharingModel]: its
          ``mean(mc_samples)``, exact when the node has a closed form or
          convolution, otherwise a Monte-Carlo estimate from
          ``mc_samples`` lifetimes;
        - a nested ``NonRepairableRBD`` or a
          [`RepeatedNode`][repyability.RepeatedNode]: a Monte-Carlo
          estimate from ``mc_samples`` lifetimes;
        - a fixed-probability node: 0.0, as it has no time dimension;
        - any other model: its own ``mean()``, e.g. the exact mean of a
          surpyval distribution.

        Common-cause groups do not affect a node's own MTTF.

        Parameters
        ----------
        mc_samples : int, optional
            Number of Monte-Carlo samples for the nodes estimated by
            simulation, by default 100_000.
        seed : int or None, optional
            Seeds numpy's global RNG for the whole call, restoring the
            caller's state afterwards (see ``random``), by default None.

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
        out: dict[Any, float] = {}
        with numpy_seed(seed):
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
                    out[node] = float(np.atleast_1d(model.mean(mc_samples))[0])
                elif is_fixed_probability(model):
                    out[node] = 0.0
                else:
                    out[node] = float(np.atleast_1d(model.mean())[0])
        return out

    # Importance measures
    # https://www.ntnu.edu/documents/624876/1277590549/chapt05.pdf/82cd565f-fa2f-43e4-a81a-095d95d39272
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
        ``i``'s own reliability. Exact; it assumes independent nodes.

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
        node_probabilities = self._probabilities_with_overrides(
            self.node_sf(x), working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._birnbaum_importance(node_probabilities),
        )

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
        ``B_i * (1 - R_i)``, with ``B_i`` the Birnbaum importance. Exact; it
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
        node_probabilities = self._probabilities_with_overrides(
            self.node_sf(x), working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._improvement_potential(node_probabilities),
        )

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
        node_probabilities = self._probabilities_with_overrides(
            self.node_sf(x), working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._risk_achievement_worth(node_probabilities),
        )

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
        node_probabilities = self._probabilities_with_overrides(
            self.node_sf(x), working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._risk_reduction_worth(node_probabilities),
        )

    @check_x
    def criticality_importance(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Criticality importance of each node at time/s ``x``.

        ``CI_i = B_i * R_i / R_sys``, with ``B_i`` the Birnbaum importance:
        the probability that node ``i`` is working and critical, given that
        the system works. This is the success-oriented form; the
        failure-oriented criticality importance of some texts,
        ``B_i * (1 - R_i) / (1 - R_sys)``, is not what is computed. Exact;
        it assumes independent nodes.

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
            ``{node: criticality importance}`` over the component nodes:
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
        valve in series (0.05): whenever the system works, the valve is
        working and critical.

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
        {'p1': 0.0909, 'p2': 0.0909, 'v': 1.0}
        """
        self._require_no_ccf()
        node_probabilities = self._probabilities_with_overrides(
            self.node_sf(x), working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._criticality_importance(node_probabilities),
        )

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
        failure probabilities are not small.

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
        rel_dict = {}
        for node_name, node in self.reliabilities.items():
            rel_dict[node_name] = node.sf(x)
        rel_dict = self._probabilities_with_overrides(
            rel_dict, working_nodes, broken_nodes
        )
        return cast(
            Dict[Any, Union[float, np.ndarray]],
            super()._fussell_vesely(rel_dict, fv_type),
        )

    def fussel_vesely(
        self, x: Optional[ArrayLike] = None, fv_type: str = "c"
    ) -> dict[Any, Union[float, np.ndarray]]:
        """Deprecated misspelt alias of ``fussell_vesely``.

        Deprecated: use
        [`fussell_vesely`][repyability.NonRepairableRBD.fussell_vesely]
        instead; this alias will be removed in a future release. It returns
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
        DeprecationWarning
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
            "(Fussell-Vesely). This alias will be removed in a future "
            "release.",
            DeprecationWarning,
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
            cls, params, names = spec
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
                dsf = _dsf_dparam(cls, params, j, x_arr, rel_step)
                node_out[name] = _out(b_i * dsf)
            sensitivities[node_name] = node_out
        return sensitivities
