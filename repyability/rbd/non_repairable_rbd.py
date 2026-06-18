import pprint
import warnings
from copy import copy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, Collection, Dict, Hashable, Iterable, Optional

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from surpyval import NonParametric

from repyability.utils.wrappers import conditional_survival

from .helper_classes import PerfectReliability, PerfectUnreliability
from .rbd import RBD
from .repeated_node import RepeatedNode
from .repeated_standby_node import RepeatedStandbyNode
from .standby_node import StandbyModel


# Event class for simulation
@dataclass(order=True)
class NodeFailure:
    """Dataclass to hold an event's information. Comparisons are performed
    by time. status=False means the event is a node failure."""

    time: float
    node: Hashable = field(compare=False)


def check_x(func):
    """Handles a none or ArrayLike x value"""

    def wrap(obj, x=None, *args, **kwargs):
        if not obj.time_varying_rbd():
            x = 1.0
        # Make sure we are using a numpy array (of 1D)
        x = np.atleast_1d(x)
        result = func(obj, x, *args, **kwargs)
        return result

    return wrap


class NonRepairableRBD(RBD):
    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        reliabilities: dict[Any, Any],
        k: dict[Any, int] = {},
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
    ):
        if on_infeasible_rbd not in ["raise", "warn", "ignore"]:
            raise ValueError(
                "'on_infeasible_rbd' must be one of"
                + " {'raise', 'warn', 'ignore'}"
            )
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
            for (start, stop) in edges:
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
                        self.structure_check[
                            "has_repeated_node_in_cycle"
                        ] = True
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

        is_fixed = []
        for _, node in self.reliabilities.items():
            if isinstance(node, NonParametric):
                is_fixed = [False]
                break
            elif isinstance(node, NonRepairableRBD):
                is_fixed.append(node.__fixed_probs)
            elif node == PerfectReliability:
                continue
            elif node == PerfectUnreliability:
                continue
            else:
                # when node is a Parametric model
                if isinstance(node, StandbyModel):
                    is_fixed = [False]
                    break
                elif isinstance(node, RepeatedNode):
                    this_fixed = node.model.dist.name in [
                        "FixedEventProbability",
                        "Bernoulli",
                    ]
                    is_fixed.append(this_fixed)
                else:
                    this_fixed = node.dist.name in [
                        "FixedEventProbability",
                        "Bernoulli",
                    ]
                    is_fixed.append(this_fixed)

        self.__fixed_probs: bool
        if all(is_fixed):
            self.__fixed_probs = True
            self.structure_check["all_distributions_fixed"] = True
        else:
            self.__fixed_probs = False
            self.structure_check["all_distributions_fixed"] = False

        # Record whether the system reliability can be solved analytically
        # (equivalently with the BDD), or whether it requires simulation
        # because one or more nodes are simulation-based (e.g. standby nodes).
        non_analytic_nodes = self.get_non_analytic_nodes()
        self.structure_check["is_analytically_solvable"] = (
            len(non_analytic_nodes) == 0
        )
        self.structure_check["non_analytic_nodes"] = non_analytic_nodes

    @check_x
    def sf(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Collection[Hashable] = [],
        broken_nodes: Collection[Hashable] = [],
        method: str = "p",
    ) -> np.ndarray:
        """Returns the system reliability for time/s x.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable
        working_nodes : Collection[Hashable], optional
            Marks these nodes as perfectly reliable, by default []
        broken_nodes : Collection[Hashable], optional
            Marks these nodes as perfectly unreliable, by default []
        working_components : Collection[Hashable], optional
            Marks these components as perfectly reliable, by default []
        broken_components : Collection[Hashable], optional
            Marks these components as perfectly unreliable, by default []
        method: str, optional
            Input either "c" or "p" for the function to use the cut set or
            path set methods respectively. Both methods return the same
            (exact) result. By default the path set method ("p") is used, as
            it avoids deriving the cut sets.

        Returns
        -------
        np.ndarray
            Reliability values for all nodes at all times x

        Raises
        ------
        ValueError
            Working/broken node/component inconsistency (a component or node
            is supplied more than once to any of working_nodes, broken_nodes,
            working_components, broken_components)
        """
        # Check for any node/component argument inconsistency
        # check_sf_node_component_args_consistency(
        #     working_nodes,
        #     broken_nodes,
        #     working_components,
        #     broken_components,
        #     self.components_to_nodes,
        # )

        for node in working_nodes:
            if node in self.repeated:
                raise ValueError(
                    (
                        "Node {}, which has been set to working is a repeat "
                        + "of node {}. You need to create a new RBD where "
                        + "it is not a repeated node."
                    ).format(node, self.repeated[node])
                )

        # Turn node iterables into sets for O(1) lookup later
        working_nodes = set(working_nodes)
        broken_nodes = set(broken_nodes)

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

    def ff(self, x: Optional[ArrayLike] = None, *args, **kwargs) -> np.ndarray:
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
    def sf_by_node(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Dict[Any, np.ndarray]:

        # The return dict
        node_sf: Dict[Any, np.ndarray] = {}

        # Cache the component reliabilities for efficiency
        for node_name, node in self.reliabilities.items():
            node_sf[node_name] = node.sf(x)
        return node_sf

    @check_x
    def ff_by_node(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Dict[Any, np.ndarray]:
        node_ff: Dict[Any, np.ndarray] = {}

        # Cache the component reliabilities for efficiency
        for node_name, node in self.reliabilities.items():
            node_ff[node_name] = node.ff(x)

        return node_ff

    def time_varying_rbd(self):
        return not self.__fixed_probs

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

    def random(self, size):
        out = np.zeros(size)
        for i in range(size):
            event_queue: PriorityQueue = PriorityQueue()
            for node in self.G.nodes:
                # .random(1) returns a 1-element array; take the scalar so the
                # event time orders the PriorityQueue and assigns into ``out``
                # (NumPy >= 2 rejects assigning a 1-element array to a scalar).
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

    def mean(self, mc_samples: int = 100_000):
        """Returns the Mean Time To Failure of the RBD
        This is necessary for recursive calls which will only use the `mean`
        """
        return self.random(mc_samples).mean().item()

    def mean_time_to_failure(self, mc_samples: int = 100_000):
        """
        User friendly way to get MTTF
        """
        return self.mean(mc_samples)

    def node_mttf(self, mc_samples: int = 100_000):
        out: dict = {}
        for node in self.nodes:
            model = self.reliabilities[node]
            if isinstance(
                model, (StandbyModel, NonRepairableRBD, RepeatedNode)
            ):
                out[node] = model.mean(mc_samples)
            elif model.dist.name in ["FixedEventProbability", "Bernoulli"]:
                out[node] = 0
            else:
                out[node] = model.mean()
        return out

    # Importance measures
    # https://www.ntnu.edu/documents/624876/1277590549/chapt05.pdf/82cd565f-fa2f-43e4-a81a-095d95d39272
    @check_x
    def birnbaum_importance(
        self, x: Optional[ArrayLike] = None
    ) -> dict[Any, np.ndarray]:
        """Returns the Birnbaum measure of importance for all nodes.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent. If the RBD called on has two or more nodes associated
        with the same component then a UserWarning is raised.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Birnbaum importances as
            values
        """

        node_probabilities = self.sf_by_node(x)
        return super()._birnbaum_importance(node_probabilities)

    # TODO: update all importance measures to allow for component as well
    @check_x
    def improvement_potential(
        self, x: Optional[ArrayLike] = None
    ) -> dict[Any, np.ndarray]:
        """Returns the improvement potential of all nodes.

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and improvement potentials as
            values
        """
        node_probabilities = self.sf_by_node(x)
        return super()._improvement_potential(node_probabilities)

    @check_x
    def risk_achievement_worth(
        self, x: Optional[ArrayLike] = None
    ) -> dict[Any, np.ndarray]:
        """Returns the RAW importance per Modarres & Kaminskiy. That is RAW_i =
        (unreliability of system given i failed) /
        (nominal system unreliability).

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RAW importances as values
        """
        node_probabilities = self.sf_by_node(x)
        return super()._risk_achievement_worth(node_probabilities)

    @check_x
    def risk_reduction_worth(
        self, x: Optional[ArrayLike] = None
    ) -> dict[Any, np.ndarray]:
        """Returns the RRW importance per Modarres & Kaminskiy. That is RRW_i =
        (nominal unreliability of system) /
        (unreliability of system given i is working).

        Parameters
        ----------
        x : ArrayLike
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RRW importances as values
        """
        node_probabilities = self.sf_by_node(x)
        return super()._risk_reduction_worth(node_probabilities)

    @check_x
    def criticality_importance(
        self, x: Optional[ArrayLike] = None
    ) -> dict[Any, np.ndarray]:
        """Returns the criticality importance of all nodes at time/s x.

        Parameters
        ----------
        x : int | float | Iterable[int  |  float]
            Time/s as a number or iterable

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and criticality importances as
            values
        """
        node_probabilities = self.sf_by_node(x)
        return super()._criticality_importance(node_probabilities)

    @check_x
    def fussel_vesely(
        self, x: Optional[ArrayLike] = None, fv_type: str = "c"
    ) -> dict[Any, np.ndarray]:
        """Calculate Fussel-Vesely importance of all components at time/s x.

        Briefly, the Fussel-Vesely importance measure for node i =
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

        Returns
        -------
        dict[Any, np.ndarray]
            Dictionary with node names as keys and fussel-vessely importances
            as values

        Raises
        ------
        ValueError
            TODO
        NotImplementedError
            TODO
        """

        # Cache the component reliabilities for efficiency
        rel_dict = {}
        for node_name, node in self.reliabilities.items():
            # TODO: make log
            # Calculating reliability in the log-domain though so the
            # components' reliability can be added avoid possible underflow
            rel_dict[node_name] = node.sf(x)

        return super()._fussel_vesely(rel_dict, fv_type)
