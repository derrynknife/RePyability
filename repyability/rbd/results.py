"""Typed result objects for RBD analyses.

These dataclasses give the results of the simulation methods documented,
discoverable (IDE autocomplete) attributes instead of opaque nested dicts,
e.g. ``result.criticalities.iou.up`` rather than
``result["criticalities"]["iou"]["up"]``.

For backwards compatibility they also behave as read-only mappings, so the
previous dict-style access keeps working: ``result["availability"]``,
``result.keys()``, ``dict(result)``, ``"criticalities" in result``, and
iteration all still do what they used to. (They are ``collections.abc.Mapping``
instances, not ``dict`` subclasses, so ``isinstance(result, dict)`` is now
False; use ``isinstance(result, Mapping)`` if you need such a check.)
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Hashable, Tuple

import numpy as np
from scipy.stats import norm


class _ResultMapping(Mapping):
    """Read-only ``Mapping`` view over a dataclass's fields.

    Preserves the dict-style access the results used to have (``result[key]``,
    ``keys``/``items``/``values``, ``in``, ``dict(result)``, iteration) while
    the subclasses add typed, documented attributes.
    """

    def __getitem__(self, key):
        if key in self._field_names():
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self._field_names())

    def __len__(self):
        return len(self._field_names())

    def _field_names(self):
        return tuple(f.name for f in dataclasses.fields(self))


@dataclass
class ConfidenceInterval(_ResultMapping):
    """A Monte-Carlo estimate with its sampling uncertainty.

    Attributes
    ----------
    estimate : float
        The point estimate (the sample mean).
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.
    confidence : float
        The confidence level the bounds correspond to (e.g. 0.95).
    standard_error : float
        The standard error of the estimate.
    n_samples : int
        The number of Monte-Carlo samples the estimate was computed from.
    """

    estimate: float
    lower: float
    upper: float
    confidence: float
    standard_error: float
    n_samples: int


@dataclass
class UpDownImportance(_ResultMapping):
    """An importance measure split by system state.

    Attributes
    ----------
    up : dict
        Per-node measure while the system is up.
    down : dict
        Per-node measure while the system is down.
    """

    up: Dict[Hashable, float]
    down: Dict[Hashable, float]


@dataclass
class FailureCriticalityIndex(_ResultMapping):
    """Fractions relating a node's failures to system failures.

    Attributes
    ----------
    per_system_failure : dict
        For each node, the fraction of *system* failures that the node caused.
    per_component_failure : dict
        For each node, the fraction of the node's own failures that caused a
        system failure.
    """

    per_system_failure: Dict[Hashable, float]
    per_component_failure: Dict[Hashable, float]


@dataclass
class RestorationCriticalityIndex(_ResultMapping):
    """Fractions relating a node's restorations to system restorations.

    Attributes
    ----------
    by_system : dict
        For each node, the fraction of *system* restorations that the node
        caused.
    by_component : dict
        For each node, the fraction of the node's own restorations that
        restored the system.
    """

    by_system: Dict[Hashable, float]
    by_component: Dict[Hashable, float]


@dataclass
class Criticalities(_ResultMapping):
    """The importance/criticality measures from an availability simulation.

    Attributes
    ----------
    operational_criticality_index : UpDownImportance
        Time the system and a node were jointly up/down, over the system's
        up/down time.
    iou : UpDownImportance
        Intersection-over-union of a node's and the system's up/down intervals.
    failure_criticality_index : FailureCriticalityIndex
        How much each node drives system failures.
    restoration_criticality_index : RestorationCriticalityIndex
        How much each node drives system restorations.
    """

    operational_criticality_index: UpDownImportance
    iou: UpDownImportance
    failure_criticality_index: FailureCriticalityIndex
    restoration_criticality_index: RestorationCriticalityIndex


@dataclass
class AvailabilityResult(_ResultMapping):
    """The result of ``RepairableRBD.availability()``.

    The properties ``mean_up_time``, ``mean_down_time`` and
    ``failure_frequency`` are simulation *estimates* derived from the fields
    below; their exact steady-state counterparts are the
    ``RepairableRBD.mean_up_time()``, ``mean_down_time()`` and
    ``system_failure_frequency()`` methods.

    Attributes
    ----------
    timeline : numpy.ndarray
        Event times at which the (mean) system availability changes.
    availability : numpy.ndarray
        Mean system availability at each time in ``timeline``.
    system_uptime : float
        Total system uptime summed over all simulations.
    time_simulated_to : float
        The ``t_simulation`` the simulations were run to.
    criticalities : Criticalities
        The importance/criticality measures (see ``Criticalities``).
    node_uptime : dict
        Total uptime per node summed over all simulations.
    node_downtime : dict
        Total downtime per node summed over all simulations; a node's uptime
        plus downtime equals ``n_simulations * time_simulated_to``.
    system_downtime : float
        Total system downtime summed over all simulations.
    system_failures : int
        Number of system failures observed across all simulations.
    system_restorations : int
        Number of system restorations observed across all simulations.
    n_simulations : int
        The number of simulations run (``N``).
    """

    timeline: np.ndarray
    availability: np.ndarray
    system_uptime: float
    time_simulated_to: float
    criticalities: Criticalities
    node_uptime: Dict[Hashable, float]
    node_downtime: Dict[Hashable, float]
    system_downtime: float
    system_failures: int
    system_restorations: int
    n_simulations: int

    @property
    def availability_se(self) -> np.ndarray:
        """Pointwise standard error of the availability estimate.

        At each time in ``timeline`` the availability is the proportion of
        the ``n_simulations`` systems that were up, so its sampling standard
        error is the binomial ``sqrt(A (1 - A) / n)``.
        """
        p = np.asarray(self.availability, dtype=float)
        return np.sqrt(p * (1.0 - p) / self.n_simulations)

    def availability_interval(
        self, confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pointwise confidence band for the availability curve.

        Uses the Wilson score interval for a binomial proportion, which
        remains well-behaved when the estimated availability is at or near 0
        or 1 (where the plain normal interval collapses to zero width).

        Parameters
        ----------
        confidence : float, optional
            The confidence level, by default 0.95.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The lower and upper bounds at each time in ``timeline``.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        z = float(norm.ppf(0.5 + confidence / 2.0))
        p = np.asarray(self.availability, dtype=float)
        n = self.n_simulations
        denominator = 1.0 + z**2 / n
        centre = (p + z**2 / (2.0 * n)) / denominator
        half_width = (z / denominator) * np.sqrt(
            p * (1.0 - p) / n + z**2 / (4.0 * n**2)
        )
        lower = np.clip(centre - half_width, 0.0, 1.0)
        upper = np.clip(centre + half_width, 0.0, 1.0)
        return lower, upper

    @property
    def mean_up_time(self) -> float:
        """Simulation estimate of the Mean Up Time,
        ``system uptime / system failures``. Infinite if no failure was
        observed.

        Note: estimated from a finite window, so each simulation's final
        (unfinished) up period is censored; for windows that are short
        relative to the up-down cycle this biases the estimate. Prefer the
        exact ``RepairableRBD.mean_up_time()`` for steady-state values.
        """
        if self.system_failures > 0:
            return self.system_uptime / self.system_failures
        return float("inf") if self.system_uptime > 0 else 0.0

    @property
    def mean_down_time(self) -> float:
        """Simulation estimate of the Mean Down Time,
        ``system downtime / system restorations``. Infinite if downtime was
        observed but never restored.

        Note: estimated from a finite window (final unfinished down periods
        are censored), so short windows bias the estimate. Prefer the exact
        ``RepairableRBD.mean_down_time()`` for steady-state values.
        """
        if self.system_restorations > 0:
            return self.system_downtime / self.system_restorations
        return float("inf") if self.system_downtime > 0 else 0.0

    @property
    def failure_frequency(self) -> float:
        """Simulation estimate of the system failure frequency (failures per
        unit time), ``system failures / total simulated time``."""
        return self.system_failures / (
            self.n_simulations * self.time_simulated_to
        )
