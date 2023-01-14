from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from surpyval import NonParametric

from .rbd import RBD
from .standby_node import StandbyModel


class NonRepairableRBD(RBD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        is_fixed = []
        for component in self.components_to_nodes:
            this_component = self.reliability[component]
            if isinstance(this_component, NonParametric):
                is_fixed.append(False)
            elif isinstance(this_component, NonRepairableRBD):
                is_fixed.append(this_component.__fixed_probs)
            else:
                if isinstance(this_component, StandbyModel):
                    this_fixed = [False]
                    break
                else:
                    this_fixed = this_component.dist.name in [
                        "FixedEventProbability",
                        "Bernoulli",
                    ]
                    is_fixed.append(this_fixed)
        if all(is_fixed):
            self.__fixed_probs = True
        else:
            self.__fixed_probs = False

    def sf(self, x: ArrayLike = None, *args, **kwargs):
        if self.__fixed_probs:
            return super().sf(1.0, *args, **kwargs).item()
        else:
            return super().sf(x, *args, **kwargs)

    def ff(self, x: ArrayLike = None, *args, **kwargs):
        return 1 - self.sf(x, *args, **kwargs)

    def sf_by_node(self, x: ArrayLike = None, *args, **kwargs):
        if self.__fixed_probs:
            x = np.array([1.0])
        else:
            x = np.atleast_1d(x)
        return super().sf_by_node(x, *args, **kwargs)

    # Importance measures
    # https://www.ntnu.edu/documents/624876/1277590549/chapt05.pdf/82cd565f-fa2f-43e4-a81a-095d95d39272
    def birnbaum_importance(self, x: ArrayLike = None) -> dict[Any, float]:
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
        if self.__fixed_probs:
            x = np.array([1.0])
        else:
            x = np.atleast_1d(x)
        for component, node_set in self.components_to_nodes.items():
            if len(node_set) > 1:
                warn(
                    f"Birnbaum's measure of importance assumes nodes are \
                     dependent, but nodes {node_set} all depend on the same \
                     component '{component}."
                )

        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            failing = self.sf(x, broken_nodes=[node])
            node_importance[node] = working - failing
        return node_importance

    # TODO: update all importance measures to allow for component as well
    def improvement_potential(self, x: ArrayLike = None) -> dict[Any, float]:
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
        if self.__fixed_probs:
            x = np.array([1.0])
        else:
            x = np.atleast_1d(x)

        node_importance = {}
        for node in self.nodes.keys():
            working = self.sf(x, working_nodes=[node])
            as_is = self.sf(x)
            node_importance[node] = working - as_is
        return node_importance

    def risk_achievement_worth(self, x: ArrayLike = None) -> dict[Any, float]:
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
        if self.__fixed_probs:
            x = np.array([1.0])
        else:
            x = np.atleast_1d(x)

        node_importance = {}
        system_ff = self.ff(x)
        for node in self.nodes.keys():
            failing = self.ff(x, broken_nodes=[node])
            node_importance[node] = failing / system_ff
        return node_importance

    def risk_reduction_worth(self, x: ArrayLike = None) -> dict[Any, float]:
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
        if self.__fixed_probs:
            x = np.array([1.0])
        else:
            x = np.atleast_1d(x)

        node_importance = {}
        system_ff = self.ff(x)
        for node in self.nodes.keys():
            working = self.ff(x, working_nodes=[node])
            node_importance[node] = system_ff / working
        return node_importance

    def criticality_importance(self, x: ArrayLike = None) -> dict[Any, float]:
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
        if self.__fixed_probs:
            x = np.array([1.0])
        else:
            x = np.atleast_1d(x)

        bi = self.birnbaum_importance(x)
        node_importance = {}
        system_sf = self.sf(x)
        for node in self.nodes.keys():
            node_sf = self.reliability[self.nodes[node]].sf(x)
            node_importance[node] = bi[node] * node_sf / system_sf
        return node_importance

    def fussel_vesely(
        self, x: ArrayLike = None, fv_type: str = "c"
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
        if self.__fixed_probs:
            x = np.array([1.0])
        else:
            x = np.atleast_1d(x)

        # Get node-sets based on what method was requested
        if fv_type == "c":
            node_sets = self.get_min_cut_sets()
        elif fv_type == "p":
            node_sets = {
                frozenset(path_set)
                for path_set in self.get_min_path_sets(
                    include_in_out_nodes=False
                )
            }
        else:
            raise ValueError(
                f"fv_type must be either 'c' (cut-set) or 'p' (path-set), \
                fv_type={fv_type} was given."
            )

        # Ensure time is a numpy array
        x = np.atleast_1d(x)

        # Get system unreliability, this will be the denominator for all node
        # importance calcs
        system_unreliability = self.ff(x)

        # The return dict
        node_importance: dict[Any, np.ndarray] = {}

        # Cache the component reliabilities for efficiency
        rel_dict = {}
        for component in self.reliability.keys():
            # TODO: make log
            # Calculating reliability in the log-domain though so the
            # components' reliability can be added avoid possible underflow
            rel_dict[component] = self.reliability[component].ff(x)

        # For each node,
        for node in self.nodes.keys():
            # Sum up the probabilities of the node_sets containing the node
            # from failing
            node_fv_numerator = 0
            for node_set in node_sets:
                if node not in node_set:
                    continue
                node_set_fail_prob = 1
                # Take only the independent components in that node-set, i.e.
                # don't multiply the same component twice in a node-set
                components_in_node_set = {
                    self.nodes[fail_node] for fail_node in node_set
                }
                for component in components_in_node_set:
                    node_set_fail_prob *= rel_dict[component]
                node_fv_numerator += node_set_fail_prob

            node_importance[node] = node_fv_numerator / system_unreliability

        return node_importance
