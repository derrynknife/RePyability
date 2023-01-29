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

from .helper_classes import PerfectReliability, PerfectUnreliability
from .rbd import RBD
from .repeated_node import RepeatedNode
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
                print(non_repeated_node_cycles)
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

        if self.structure_check["is_valid"]:
            self.compile_bdd()

    @check_x
    def sf(
        self,
        x: Optional[ArrayLike] = None,
        working_nodes: Collection[Hashable] = [],
        broken_nodes: Collection[Hashable] = [],
        method: str = "c",
        approx: bool = False,
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
            path set methods respectively, by default "c". Both methods
            ultimately return the same results though.
        approx: bool, optional
            If true, only considers the first-order terms (w.r.t. the
            inclusion-exclusion principle), thereby reducing computation time.
            This approximation is only applicable to the cut set method
            (method="c"), a ValueError exception is raised if method="p" and
            approx=True. This approximation is typically sufficient for most
            use cases where reliabilities are close to 1. By default, False.

        Returns
        -------
        np.ndarray
            Reliability values for all nodes at all times x

        Raises
        ------
        ValueError
            - Working/broken node/component inconsistency (a component or node
              is supplied more than once to any of working_nodes, broken_nodes,
              working_components, broken_components)
            - The path set method must not be used with approx=True, see approx
              arg description above
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

        # Check that path set method and approximation are not used together
        # (The approximation is only applicable to the cutset method)
        if method == "p" and approx:
            raise ValueError(
                "The path set method must not be used with \
                approx=True, see approx arg description in docstring."
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

    def ff_by_node(
        self, x: Optional[ArrayLike] = None, *args, **kwargs
    ) -> Dict[Any, np.ndarray]:

        # The return dict
        node_ff: Dict[Any, np.ndarray] = {}

        # Cache the component reliabilities for efficiency
        for node_name, node in self.reliabilities.items():
            node_ff[node_name] = node.ff(x)

        return node_ff

    def time_varying_rbd(self):
        return not self.__fixed_probs

    def random(self, size):
        out = np.zeros(size)
        for i in range(size):
            event_queue: PriorityQueue = PriorityQueue()
            for node in self.G.nodes:
                time = self.reliabilities[node].random(1)
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
            if model.dist.name == "FixedEventProbability":
                out[node] = 0
            elif isinstance(
                model, (StandbyModel, NonRepairableRBD, RepeatedNode)
            ):
                out[node] = model.mean(mc_samples)
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
