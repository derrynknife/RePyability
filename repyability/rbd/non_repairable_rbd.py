import functools
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

from ._model_utils import is_fixed_probability, parametric_spec
from .helper_classes import PerfectReliability, PerfectUnreliability
from .node_state import NodeState
from .rbd import RBD
from .repeated_node import RepeatedNode
from .repeated_standby_node import RepeatedStandbyNode
from .results import ConfidenceInterval
from .standby_node import StandbyModel


# Event class for simulation
@dataclass(order=True)
class NodeFailure:
    """Dataclass to hold an event's information. Comparisons are performed
    by time. status=False means the event is a node failure."""

    time: float
    node: Hashable = field(compare=False)


def check_x(func):
    """Normalises the time input ``x`` and enforces the return contract.

    The wrapped function always receives ``x`` as a 1-d numpy array. The
    caller-facing contract is numpy-style: a scalar ``x`` returns a float (or
    a dict of floats for the per-node/importance methods), an array ``x``
    returns a numpy array (or dict of arrays).

    ``x=None`` is allowed only for a fixed-probability RBD (where time is
    irrelevant); for a time-varying RBD it raises a ValueError rather than
    failing cryptically downstream.
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
    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        reliabilities: dict[Any, Any],
        k: Optional[dict[Any, int]] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
    ):
        if on_infeasible_rbd not in ["raise", "warn", "ignore"]:
            raise ValueError(
                "'on_infeasible_rbd' must be one of"
                + " {'raise', 'warn', 'ignore'}"
            )
        # Capture the constructor inputs verbatim (before any mutation) so the
        # RBD can be faithfully serialised via to_dict()/to_json().
        edges = list(edges)
        self._init_args = {
            "edges": [tuple(e) for e in edges],
            "reliabilities": dict(reliabilities),
            "k": dict(k) if k else None,
            "input_node": input_node,
            "output_node": output_node,
            "on_infeasible_rbd": on_infeasible_rbd,
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
                k,
                set(reliabilities.keys()),
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
                k,
                set(reliabilities.keys()),
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
                if isinstance(node, StandbyModel):
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
        """Returns the system reliability for time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Marks these nodes as perfectly reliable, by default None
        broken_nodes : Collection[Hashable], optional
            Marks these nodes as perfectly unreliable, by default None
        method: str, optional
            Input either "c" or "p" for the function to use the cut set or
            path set methods respectively. Both methods return the same
            (exact) result. By default the path set method ("p") is used, as
            it avoids deriving the cut sets.

        Returns
        -------
        float or np.ndarray
            The system reliability: a float for scalar ``x``, an array for
            array ``x``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input/output node, is
            in both sets, or is a repeat of another node.

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

        # Collect node probabilities to pass to RBD class
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

        return self.system_probability(node_probabilities, method=method)

    def ff(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        """Returns the system unreliability for time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        *args, **kwargs :
            Any sf() arguments

        Returns
        -------
        float or np.ndarray
            The system unreliability: a float for scalar ``x``, an array for
            array ``x``.

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

    def unreliability(self, x: Optional[ArrayLike] = None, *args, **kwargs):
        """Returns the system unreliability for time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        *args, **kwargs :
            Any sf() arguments

        Returns
        -------
        np.ndarray
            Unreliability values for all nodes at all times x
        """
        return 1 - self.sf(x, *args, **kwargs)

    def reliability(self, x: Optional[ArrayLike] = None, *args, **kwargs):
        """Returns the system reliability for time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        *args, **kwargs :
            Any sf() arguments

        Returns
        -------
        np.ndarray
            Reliability values for all nodes at all times x
        """
        return self.sf(x, *args, **kwargs)

    def cs(self, x: ArrayLike, X: ArrayLike, *args, **kwargs) -> np.ndarray:
        """Returns the conditional survival of the system.

        That is, the probability the system survives a *further* ``x`` given it
        has already survived to ``X``: ``R(x | X) = sf(X + x) / sf(X)``.

        Parameters
        ----------
        x : ArrayLike
            The further duration/s at which conditional survival is evaluated.
        X : ArrayLike
            The age/s the system is known to have survived to.
        *args, **kwargs :
            Any sf() arguments (e.g. working_nodes, broken_nodes, method).

        Returns
        -------
        np.ndarray
            The conditional survival probability/ies.
        """
        return conditional_survival(self, x, X, *args, **kwargs)

    @check_x
    def Hf(self, x: Optional[ArrayLike] = None, **kwargs) -> np.ndarray:
        """Returns the system cumulative hazard function H(x) = -ln R(x).

        This is exact given the (exact) system reliability R(x); it is +inf
        wherever the system reliability has reached zero.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable.
        **kwargs :
            Any sf() arguments (e.g. working_nodes, broken_nodes, method).
        """
        sf = np.asarray(self.sf(x, **kwargs), dtype=float)
        with np.errstate(divide="ignore"):
            return -np.log(sf)

    @check_x
    def df(
        self, x: Optional[ArrayLike] = None, dx: float = 1e-6, **kwargs
    ) -> np.ndarray:
        """Returns the system failure density f(x) = -dR/dx.

        Computed by central finite differences of the system reliability
        (which is composed of the nodes' reliabilities, so it is smooth in x);
        ``dx`` sets the relative step size. The step never crosses into
        negative time.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable.
        dx : float, optional
            Relative finite-difference step, by default 1e-6.
        **kwargs :
            Any sf() arguments (e.g. working_nodes, broken_nodes, method).
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
        """Returns the system hazard rate h(x) = f(x) / R(x).

        Uses the (numerical) failure density df() over the (exact) system
        reliability. It is +inf wherever the system reliability has reached
        zero.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable.
        dx : float, optional
            Relative finite-difference step passed to df(), by default 1e-6.
        **kwargs :
            Any sf() arguments (e.g. working_nodes, broken_nodes, method).
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
        """Returns each node's reliability at time/s x (a dict keyed by node
        name). Floats for scalar ``x``, arrays for array ``x``."""
        node_sf: Dict[Any, Union[float, np.ndarray]] = {}
        for node_name, node in self.reliabilities.items():
            node_sf[node_name] = node.sf(x)
        return node_sf

    @check_x
    def node_ff(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Dict[Any, Union[float, np.ndarray]]:
        """Returns each node's unreliability at time/s x (a dict keyed by node
        name). Floats for scalar ``x``, arrays for array ``x``."""
        node_ff: Dict[Any, Union[float, np.ndarray]] = {}
        for node_name, node in self.reliabilities.items():
            node_ff[node_name] = node.ff(x)

        return node_ff

    @property
    def is_fixed(self) -> bool:
        """True if every node's reliability is a fixed (time-invariant)
        probability, so the system reliability does not vary with time."""
        return self._fixed_probs

    @property
    def is_time_varying(self) -> bool:
        """True if any node's reliability varies with time (the complement of
        :attr:`is_fixed`)."""
        return not self._fixed_probs

    # Node model types whose reliability is obtained by Monte-Carlo simulation
    # (a Kaplan-Meier fit to simulated samples) rather than in closed form. A
    # standby arrangement is sequence-dependent (dynamic), so its sf(t) cannot
    # be expressed analytically and is instead estimated by simulation. Such
    # nodes therefore prevent a purely analytic / BDD solution of the system.
    _SIMULATION_NODE_TYPES = (StandbyModel, RepeatedStandbyNode)

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
        """Returns the nodes that prevent an analytic / BDD solution.

        Returns
        -------
        dict[Any, str]
            A mapping of node name -> the offending model's type name for
            every node whose reliability requires Monte-Carlo simulation (e.g.
            a StandbyModel). Empty if the RBD is analytically solvable.
        """
        non_analytic: dict[Any, str] = {}
        for node_name, model in self.reliabilities.items():
            if not self._node_is_analytic(model):
                non_analytic[node_name] = type(model).__name__
        return non_analytic

    def is_analytically_solvable(self) -> bool:
        """Returns whether the system reliability can be solved analytically.

        The analytic methods (the inclusion-exclusion in system_probability(),
        and equivalently a BDD evaluation) require every node to expose a
        reliability sf(t) that does not itself depend on Monte-Carlo
        simulation. This holds for parametric and non-parametric distributions,
        fixed-probability nodes, repeated nodes of such models, and nested RBDs
        that are themselves analytically solvable.

        It does NOT hold when any node is a standby arrangement (StandbyModel
        or RepeatedStandbyNode): a standby node is sequence-dependent and its
        sf(t) is estimated by simulation, so while sf()/system_probability()
        will still return a value, that value is only as good as the underlying
        Monte-Carlo + Kaplan-Meier fit (a step function bounded by the sampled
        support) rather than a closed-form result. Such systems are better
        evaluated by simulation (e.g. random()/mean()).

        Returns
        -------
        bool
            True if the RBD can be solved analytically / with a BDD, False if
            it requires simulation. Use get_non_analytic_nodes() to see which
            nodes are responsible.
        """
        return len(self.get_non_analytic_nodes()) == 0

    def random(self, size, seed=None):
        """Monte-Carlo simulate ``size`` system failure times.

        Parameters
        ----------
        size : int
            Number of system-lifetime samples to draw.
        seed : int or None, optional
            If given, seeds numpy's global RNG for the duration of the draw so
            the result is reproducible (surpyval's ``.random`` uses the global
            RNG); the caller's RNG state is restored afterwards. By default
            None (non-reproducible).
        """
        out = np.zeros(size)
        with numpy_seed(seed):
            for i in range(size):
                event_queue: PriorityQueue = PriorityQueue()
                for node in self.G.nodes:
                    # .random(1) returns a 1-element array; take the scalar so
                    # the event time orders the PriorityQueue and assigns into
                    # ``out`` (NumPy >= 2 rejects assigning a 1-element array
                    # to a scalar).
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
        """Returns the Mean Time To Failure of the RBD
        This is necessary for recursive calls which will only use the `mean`
        """
        return self.random(mc_samples, seed=seed).mean().item()

    def mean_time_to_failure(self, mc_samples: int = 100_000, seed=None):
        """
        User friendly way to get MTTF
        """
        return self.mean(mc_samples, seed=seed)

    def mean_time_to_failure_interval(
        self,
        mc_samples: int = 100_000,
        confidence: float = 0.95,
        seed=None,
    ) -> ConfidenceInterval:
        """Returns the Monte-Carlo MTTF estimate with its sampling
        uncertainty.

        The MTTF is the mean of ``mc_samples`` simulated system lifetimes; by
        the central limit theorem its sampling error is normal with standard
        error ``sample std / sqrt(mc_samples)``, from which the confidence
        interval is built.

        Parameters
        ----------
        mc_samples : int, optional
            Number of Monte-Carlo samples, by default 100_000.
        confidence : float, optional
            The confidence level, by default 0.95.
        seed : int or None, optional
            Seed for reproducibility (see :meth:`random`).

        Returns
        -------
        ConfidenceInterval
            The estimate, bounds, standard error and sample count.
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
        """Returns the time at which system reliability equals ``target``.

        Solves ``R(t) = target`` for ``t`` (the inverse of :meth:`sf`). System
        reliability is monotonically non-increasing in time, so the solution
        is unique.

        Parameters
        ----------
        target : float
            The reliability level to solve for, in (0, 1).
        upper_bound : float, optional
            An upper bound for the search; found automatically (by doubling)
            if None.
        **kwargs :
            Any sf() arguments (e.g. working_nodes, broken_nodes, method).

        Returns
        -------
        float
            The time at which ``R(t) == target``.

        Raises
        ------
        ValueError
            If ``target`` is not in (0, 1), if the RBD is fixed-probability
            (reliability is constant in time), or if ``target`` exceeds the
            system reliability at ``t = 0`` (so it is never reached).

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
        """Returns the B\\ :sub:`X` life: the time by which ``x`` percent of
        systems have failed, i.e. the time at which ``R(t) = 1 - x/100``.

        For example ``bx_life(10)`` is the B10 life (10% failed / 90%
        reliability).

        Parameters
        ----------
        x : float
            The percentage failed, in (0, 100).
        **kwargs :
            Any sf() arguments (e.g. working_nodes, broken_nodes, method).

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
            model, (StandbyModel, RepeatedStandbyNode, RepeatedNode, RBD)
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

    def _state_node_probabilities(self, x, state) -> Dict[Any, np.ndarray]:
        """Per-node forward reliability at ``x`` given each node's ``state``.

        ``x`` is a 1-d array. Each node conditions on its own current life: a
        node absent from ``state`` uses its unconditioned reliability (age 0),
        a failed node contributes zero, and an alive node of age ``X``
        contributes ``conditional_survival(model, x, X) = sf(X + x) / sf(X)``.
        """
        self._validate_state(state)
        node_probabilities: Dict[Any, np.ndarray] = {}
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
        """System reliability a further ``x`` into the future, given each
        node's current state.

        The condition-based ("digital twin") generalisation of :meth:`sf`:
        instead of assuming every component is new, each component conditions
        on its own current life ``X_i`` (streamed from sensors) and the
        conditioned per-node reliabilities are propagated exactly through the
        system::

            R_i(x | X_i)     = R_i(X_i + x) / R_i(X_i)
            R_sys(x | {X_i}) = system_probability({ R_i(x | X_i) })

        Parameters
        ----------
        x : ArrayLike
            The further duration/s at which reliability is evaluated (measured
            from *now*, so ``x = 0`` is the present).
        state : dict[Hashable, NodeState]
            The current state of each component. A node omitted from the
            mapping is treated as new (age 0), so an empty state reproduces
            :meth:`sf`.
        method : str, optional
            "p" (path-set, default) or "c" (cut-set); both are exact.

        Returns
        -------
        float or np.ndarray
            System reliability given the state: a float for scalar ``x``, an
            array for array ``x``.

        Notes
        -----
        Only lifetime (time-varying) distributions age; a fixed-probability
        component conditioned on being alive contributes reliability 1 going
        forward (its per-demand uncertainty is resolved by observing it
        alive). Composite / dynamic nodes (standby, repeated, nested RBD) are
        not supported here in this release and raise if given a state.

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
        """Remaining useful life (RUL): the further time until system
        reliability falls to ``target``, given each node's current state.

        The condition-based analogue of :meth:`time_to_reliability`: it solves
        ``sf_given_state(t, state) == target`` for ``t``. Because
        ``sf_given_state`` is measured from now, the result is the time
        *remaining* from the current state. ``remaining_life(1 - x/100,
        state)`` is the conditional B\\ :sub:`X` life.

        Parameters
        ----------
        target : float
            The system reliability level to solve for, in (0, 1).
        state : dict[Hashable, NodeState]
            The current state of each component (see :meth:`sf_given_state`).
        upper_bound : float, optional
            An upper bound for the search; found automatically if None.

        Returns
        -------
        float
            The remaining time until ``R_sys(t | state) == target``.

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
        """Live, state-dependent node importance over a forward horizon ``x``
        -- how each component's importance shifts once the current wear on
        every component is accounted for.

        Evaluates the Birnbaum and criticality importance measures at the
        conditioned per-node reliabilities ``R_i(x | X_i)`` (see
        :meth:`sf_given_state`) rather than at the as-new reliabilities, so the
        rankings reflect the current state. Both use the same conventions as
        :meth:`birnbaum_importance` and :meth:`criticality_importance`: they
        measure how much the system reliability depends on each node *now*, not
        which node is most likely to have failed.

        Parameters
        ----------
        x : ArrayLike
            The forward horizon/s over which importance is evaluated (from
            now).
        state : dict[Hashable, NodeState]
            The current state of each component (see :meth:`sf_given_state`).

        Returns
        -------
        dict[str, dict[Any, float | np.ndarray]]
            ``{"birnbaum": {node: value}, "criticality": {node: value}}``.
            Values are floats for scalar ``x`` and arrays for array ``x``.
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
        """Returns each node's mean time to failure (a dict keyed by node
        name). Simulation-based node models use ``mc_samples`` Monte-Carlo
        draws; fixed-probability nodes have no time dimension and return 0."""
        out: dict[Any, float] = {}
        with numpy_seed(seed):
            for node in self.nodes:
                model = self.reliabilities[node]
                if isinstance(
                    model, (StandbyModel, NonRepairableRBD, RepeatedNode)
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
        """Returns the Birnbaum measure of importance for all nodes.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being perfectly reliable, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes having failed, by default None

        Returns
        -------
        dict[Any, float | np.ndarray]
            Dictionary with node names as keys and Birnbaum importances as
            values (floats for scalar ``x``, arrays for array ``x``)

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
        """
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
        """Returns the improvement potential of all nodes.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being perfectly reliable, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes having failed, by default None

        Returns
        -------
        dict[Any, float | np.ndarray]
            Dictionary with node names as keys and improvement potentials as
            values (floats for scalar ``x``, arrays for array ``x``)
        """
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
        """Returns the RAW importance per Modarres & Kaminskiy. That is RAW_i =
        (unreliability of system given i failed) /
        (nominal system unreliability).

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being perfectly reliable, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes having failed, by default None

        Returns
        -------
        dict[Any, float | np.ndarray]
            Dictionary with node names as keys and RAW importances as values
            (floats for scalar ``x``, arrays for array ``x``)
        """
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
        """Returns the RRW importance per Modarres & Kaminskiy. That is RRW_i =
        (nominal unreliability of system) /
        (unreliability of system given i is working).

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being perfectly reliable, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes having failed, by default None

        Returns
        -------
        dict[Any, float | np.ndarray]
            Dictionary with node names as keys and RRW importances as values
            (floats for scalar ``x``, arrays for array ``x``)
        """
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
        """Returns the criticality importance of all nodes at time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being perfectly reliable, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes having failed, by default None

        Returns
        -------
        dict[Any, float | np.ndarray]
            Dictionary with node names as keys and criticality importances as
            values (floats for scalar ``x``, arrays for array ``x``)
        """
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
        """Calculate Fussell-Vesely importance of all nodes at time/s x.

        Briefly, the Fussell-Vesely importance measure for node i =
        (sum of probabilities of cut-sets including node i occuring/failing) /
        (the probability of the system failing).

        Typically this measure is implemented using cut-sets as mentioned
        above, although the measure can be implemented using path-sets. Both
        are implemented here.

        fv_type dictates the method:
            "c" - cut-set
            "p" - path-set

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        fv_type : str, optional
            Dictates the method of calculation, 'c' = cut-set and
            'p' = path-set, by default "c"
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being perfectly reliable, by default None
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes having failed, by default None

        Returns
        -------
        dict[Any, float | np.ndarray]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values (floats for scalar ``x``, arrays for array ``x``)

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set).
        """
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
        """Deprecated alias for :meth:`fussell_vesely` (corrected spelling)."""
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
        """Sensitivity of the system reliability to each node's distribution
        parameters at time/s ``x``.

        For node ``i`` with parameter ``theta``, the sensitivity is

        ``d R_sys / d theta = B_i(x) * d sf_i(x; theta) / d theta``

        where ``B_i`` is the Birnbaum importance of node ``i`` (how much the
        system reliability moves per unit change in that node's reliability)
        and ``d sf_i / d theta`` is how much the node's reliability moves per
        unit change in the parameter. The parameter derivative is taken
        numerically (a central finite difference, rebuilding the distribution
        with ``from_params``), so it applies to any parametric surpyval model
        without per-distribution formulae. It answers "which fitted parameter,
        if it were a little different, would move system reliability the most"
        -- e.g. to target data collection or to gauge the impact of estimation
        uncertainty.

        Only nodes with reconstructable distribution parameters are included;
        composite nodes (a nested RBD, a standby arrangement, a repeated node)
        and fitted non-parametric models have no parameters to perturb and are
        omitted. A node forced via ``working_nodes``/``broken_nodes`` is pinned
        independently of its parameters, so its sensitivities are reported as
        zero.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable.
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being perfectly reliable, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes having failed, by default None.
        rel_step : float, optional
            Relative step used for the finite difference, by default ``1e-5``.

        Returns
        -------
        dict[Any, dict[str, float | np.ndarray]]
            ``{node_name: {parameter_name: sensitivity}}``. Sensitivities are
            floats for scalar ``x`` and arrays for array ``x``.
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
