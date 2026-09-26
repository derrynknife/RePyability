"""Typed result objects for RBD analyses.

These dataclasses give the results of the analysis methods documented,
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
from typing import Dict, Hashable, Optional, Tuple

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

    Returned by ``NonRepairableRBD.mean_time_to_failure_interval`` and
    [`CostResult.mean_interval`][repyability.CostResult.mean_interval].
    The estimate is a sample mean, which by the central limit theorem is
    approximately normal, so the interval is
    ``estimate +/- z * standard_error`` for the normal quantile ``z`` of the
    confidence level; both methods clip the lower bound at 0, as the MTTF
    and costs they estimate are non-negative. It describes how precisely
    the mean has been estimated, and narrows like ``1 / sqrt(n_samples)``.
    Like the other result types it is also a read-only mapping of its
    fields.

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

    Examples
    --------
    The MTTF of a single exponential component with mean 10, from 10,000
    simulated lifetimes:

    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD
    >>> rbd = NonRepairableRBD(
    ...     [("s", "c"), ("c", "t")],
    ...     {"c": surv.Exponential.from_params([0.1])},
    ... )
    >>> interval = rbd.mean_time_to_failure_interval(mc_samples=10_000, seed=0)
    >>> round(interval.estimate, 2), round(interval.standard_error, 2)
    (9.91, 0.1)
    >>> round(interval.lower, 2), round(interval.upper, 2)
    (9.71, 10.1)
    >>> interval["confidence"], interval.n_samples
    (0.95, 10000)
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

    Holds a per-node measure computed twice: once from the system's and the
    nodes' up (working) time, and once from their down (failed) time. Used
    for the operational criticality index and the intersection-over-union
    measures of [`Criticalities`][repyability.Criticalities], which defines
    them.

    Attributes
    ----------
    up : dict
        Node -> the measure computed from up time.
    down : dict
        Node -> the measure computed from down time.

    Examples
    --------
    A parallel pair is down only while both nodes are, so the share of the
    system's down time during which each node is down (the down-time
    operational criticality index) is 1:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> unit = {
    ...     "reliability": surv.Exponential.from_params([0.1]),
    ...     "repairability": surv.Exponential.from_params([1.0]),
    ... }
    >>> rbd = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": unit, "b": unit},
    ... )
    >>> result = rbd.availability(t_simulation=50, N=200, seed=0)
    >>> oci = result.criticalities.operational_criticality_index
    >>> {node: round(float(v), 4) for node, v in oci.down.items()}
    {'a': 1.0, 'b': 1.0}
    >>> oci["down"] is oci.down
    True
    """

    up: Dict[Hashable, float]
    down: Dict[Hashable, float]


@dataclass
class FailureCriticalityIndex(_ResultMapping):
    """Fractions relating a node's failures to system failures.

    A node causes a system failure when its own failure takes the system
    from up to down. The counts are summed over all the simulations of
    ``RepairableRBD.availability``, and only nodes that failed at least
    once appear (so not a node held working or broken).

    Attributes
    ----------
    per_system_failure : dict
        For each node, the fraction of *system* failures that the node
        caused. These sum to 1 over the nodes, or are all 0 if the system
        never failed.
    per_component_failure : dict
        For each node, the fraction of the node's own failures that caused a
        system failure.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> unit = {
    ...     "reliability": surv.Exponential.from_params([0.1]),
    ...     "repairability": surv.Exponential.from_params([1.0]),
    ... }
    >>> rbd = RepairableRBD(
    ...     [("s", "a"), ("a", "b"), ("b", "t")], {"a": unit, "b": unit}
    ... )
    >>> result = rbd.availability(t_simulation=50, N=200, seed=0)
    >>> fci = result.criticalities.failure_criticality_index
    >>> round(sum(fci.per_system_failure.values()), 4)
    1.0

    In series a node's failure brings the system down unless the other
    node is already down for repair:

    >>> share = fci.per_component_failure
    >>> {node: round(share[node], 2) for node in sorted(share)}
    {'a': 0.92, 'b': 0.91}
    """

    per_system_failure: Dict[Hashable, float]
    per_component_failure: Dict[Hashable, float]


@dataclass
class RestorationCriticalityIndex(_ResultMapping):
    """Fractions relating a node's restorations to system restorations.

    A node causes a system restoration when its own restoration takes the
    system from down to up. The counts are summed over all the simulations
    of ``RepairableRBD.availability``, and only nodes restored at least
    once appear (so not a node held working or broken).

    Attributes
    ----------
    by_system : dict
        For each node, the fraction of *system* restorations that the node
        caused. These sum to 1 over the nodes, or are all 0 if the system
        was never restored.
    by_component : dict
        For each node, the fraction of the node's own restorations that
        restored the system.

    Examples
    --------
    In a parallel pair the system comes back as soon as either node is
    repaired, so each node restores it about half the time:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> unit = {
    ...     "reliability": surv.Exponential.from_params([0.1]),
    ...     "repairability": surv.Exponential.from_params([1.0]),
    ... }
    >>> rbd = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": unit, "b": unit},
    ... )
    >>> result = rbd.availability(t_simulation=50, N=200, seed=0)
    >>> share = result.criticalities.restoration_criticality_index.by_system
    >>> {node: round(share[node], 1) for node in sorted(share)}
    {'a': 0.5, 'b': 0.5}
    """

    by_system: Dict[Hashable, float]
    by_component: Dict[Hashable, float]


@dataclass
class Criticalities(_ResultMapping):
    """The importance/criticality measures from an availability simulation.

    ``RepairableRBD.availability`` attributes the system's behaviour to its
    nodes in four ways, each keyed by node name. Every measure is a ratio
    of totals summed over all the simulations, and is 0 where its
    denominator is 0 (e.g. the down-time measures when the system was never
    down).

    - Operational criticality index: ``up[i]`` is the time node i and the
      system were both up over the system's total up time (the share of
      the system's up time during which node i was up); ``down[i]`` is the
      time both were down over the system's total down time.
    - Intersection over union: ``up[i]`` is the time both were up over the
      time either was up; ``down[i]`` is the time both were down over the
      time either was down. It is 1 when node i's up (or down) periods
      coincide with the system's.
    - Failure criticality index: how often node i's failures took the
      system down, per system failure and per failure of node i (see
      [`FailureCriticalityIndex`][repyability.FailureCriticalityIndex]).
    - Restoration criticality index: how often node i's restorations
      brought the system back, per system restoration and per restoration
      of node i (see
      [`RestorationCriticalityIndex`][repyability.RestorationCriticalityIndex]).

    The first two have an entry for every node; the last two only for the
    nodes that failed, or were restored, at least once.

    Attributes
    ----------
    operational_criticality_index : UpDownImportance
        Time the system and a node were jointly up/down, over the system's
        up/down time.
    iou : UpDownImportance
        Intersection-over-union of a node's and the system's up/down
        intervals.
    failure_criticality_index : FailureCriticalityIndex
        How much each node drives system failures.
    restoration_criticality_index : RestorationCriticalityIndex
        How much each node drives system restorations.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> unit = {
    ...     "reliability": surv.Exponential.from_params([0.1]),
    ...     "repairability": surv.Exponential.from_params([1.0]),
    ... }
    >>> rbd = RepairableRBD(
    ...     [("s", "a"), ("a", "b"), ("b", "t")], {"a": unit, "b": unit}
    ... )
    >>> crit = rbd.availability(t_simulation=50, N=200, seed=0).criticalities

    In series the system is up only while every node is up, but each node
    is down for only about half of the system's down time:

    >>> oci = crit.operational_criticality_index
    >>> {node: round(float(v), 4) for node, v in oci.up.items()}
    {'a': 1.0, 'b': 1.0}
    >>> {node: round(float(v), 2) for node, v in oci.down.items()}
    {'a': 0.53, 'b': 0.51}
    >>> crit["iou"] is crit.iou  # dict-style access also works
    True
    """

    operational_criticality_index: UpDownImportance
    iou: UpDownImportance
    failure_criticality_index: FailureCriticalityIndex
    restoration_criticality_index: RestorationCriticalityIndex


@dataclass
class CostResult(_ResultMapping):
    """The simulated cost of running the system over a window.

    Returned by ``RepairableRBD.cost`` and as the ``cost`` of the
    [`AvailabilityResult`][repyability.AvailabilityResult] of a priced RBD.
    Each of the ``n_simulations`` replications of the availability
    simulation yields one *total cost over the window*: repair/replace
    costs charged at each component failure, plus any per-component
    downtime cost, plus the system-downtime cost. ``samples`` is that
    distribution — the point of simulating rather than stopping at the
    exact ``RepairableRBD.expected_cost_rate()`` is the spread:
    ``percentile(90)`` answers "what could a bad window cost", which a mean
    cannot.

    Two different uncertainties live here. ``std`` and ``percentile``
    describe how much the cost of a window *varies*; that is a property of
    the system and does not shrink with more replications. ``mean_se`` and
    ``mean_interval`` describe how precisely the *expected* cost has been
    estimated; that shrinks like ``1 / sqrt(n_simulations)``.

    Attributes
    ----------
    samples : numpy.ndarray
        Total cost of each replication over the window (length
        ``n_simulations``).
    t_simulation : float
        The window length each replication was run for.
    n_simulations : int
        The number of replications.
    by_category : dict
        Mean per-replication cost split into ``"repair"`` and ``"replace"``
        (both charged per failure), ``"component_downtime"`` and
        ``"system_downtime"``. The four sum to ``mean``.
    by_component : dict
        Mean per-replication cost attributable to each costed component (its
        repair, replace and own downtime cost; the system-downtime cost is
        not attributed to components).

    Examples
    --------
    One component with MTTF 10 and MTTR 1, at 100 per repair, and 50 per
    hour of system downtime, over windows of 100 hours:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> rbd = RepairableRBD(
    ...     [("s", "c"), ("c", "t")],
    ...     {
    ...         "c": {
    ...             "reliability": surv.Exponential.from_params([0.1]),
    ...             "repairability": surv.Exponential.from_params([1.0]),
    ...             "repair_cost": 100.0,
    ...         }
    ...     },
    ...     downtime_cost_rate=50.0,
    ... )
    >>> result = rbd.cost(t_simulation=100.0, N=200, seed=0)
    >>> round(result.mean, 2), round(result.std, 2)
    (1351.33, 442.78)
    >>> round(result.by_category["repair"], 2)  # 100 per failure
    896.0
    >>> round(result.by_category["system_downtime"], 2)  # 50 per hour down
    455.33
    >>> interval = result.mean_interval(0.95)
    >>> round(interval.lower, 2), round(interval.upper, 2)
    (1289.96, 1412.69)
    """

    samples: np.ndarray
    t_simulation: float
    n_simulations: int
    by_category: Dict[str, float]
    by_component: Dict[Hashable, float]

    @property
    def mean(self) -> float:
        """Mean total cost over the window.

        The average of ``samples``: the simulated estimate of the expected
        cost of one window.

        Returns
        -------
        float
            The mean of ``samples``.
        """
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        """Sample standard deviation of the window's total cost.

        How much the cost of one window varies from window to window (with
        ``ddof=1``, so it is nan, with a numpy warning, for a single
        replication).

        Returns
        -------
        float
            The standard deviation of ``samples``.
        """
        return float(np.std(self.samples, ddof=1))

    @property
    def mean_se(self) -> float:
        """Standard error of ``mean``, ``std / sqrt(n_simulations)``: how far
        the simulated mean is likely to be from the true expected cost.

        Returns
        -------
        float
            The standard error of the mean.
        """
        return self.std / float(np.sqrt(len(self.samples)))

    def mean_interval(self, confidence: float = 0.95) -> ConfidenceInterval:
        """Confidence interval for the expected total cost over the window.

        By the central limit theorem the mean of the replications is normal
        with standard error ``mean_se``, from which the interval
        ``mean +/- z * mean_se`` is built, for the normal quantile ``z`` of
        ``confidence``; the lower bound is clipped at 0. Use it to judge
        whether ``N`` was large enough; for the range a single window's cost
        could fall in, use ``percentile`` instead.

        Parameters
        ----------
        confidence : float, optional
            The confidence level, strictly between 0 and 1, by default 0.95.

        Returns
        -------
        ConfidenceInterval
            The estimate (``mean``), bounds, standard error and sample count.

        Raises
        ------
        ValueError
            If ``confidence`` is not strictly between 0 and 1.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        estimate = self.mean
        standard_error = self.mean_se
        z = float(norm.ppf(0.5 + confidence / 2.0))
        return ConfidenceInterval(
            estimate=estimate,
            lower=max(0.0, estimate - z * standard_error),
            upper=estimate + z * standard_error,
            confidence=confidence,
            standard_error=standard_error,
            n_samples=len(self.samples),
        )

    @property
    def cost_rate(self) -> float:
        """Mean cost per unit time (``mean / t_simulation``); converges to
        the exact ``RepairableRBD.expected_cost_rate()`` as the window grows.

        Over a short window it differs from the long-run rate because every
        simulation starts with all components working.

        Returns
        -------
        float
            The mean cost per unit time over the window.
        """
        return self.mean / self.t_simulation

    def percentile(self, q) -> float:
        """The ``q``-th percentile of the window's total cost (e.g.
        ``percentile(90)`` for a planning-case budget).

        Computed from ``samples`` with ``numpy.percentile`` (linear
        interpolation between samples).

        Parameters
        ----------
        q : float
            The percentile, between 0 and 100.

        Returns
        -------
        float
            The cost below which about ``q`` percent of the simulated
            windows fall.

        Raises
        ------
        ValueError
            If ``q`` is outside ``[0, 100]``.
        """
        return float(np.percentile(self.samples, q))


@dataclass
class RedundancyAllocation(_ResultMapping):
    """The result of ``NonRepairableRBD.allocate_redundancy()``.

    The chosen number of copies of each costed node, with the system
    reliability and total cost that allocation gives. Like the other result
    types it is also a read-only mapping of its fields.

    Attributes
    ----------
    units : dict
        How many identical copies of each costed node to fit in active
        parallel. Always at least 1: the original unit.
    reliability : float
        The system reliability with that allocation (at the mission time
        ``t``, for a time-varying RBD).
    cost : float
        Total cost of the allocation, ``sum(costs[node] * units[node])``.
        Every copy is costed, including the original.
    method : str
        ``"exact"`` (a proven optimum) or ``"greedy"`` (a fast heuristic
        solution, usually but not always optimal).

    Examples
    --------
    Two components in series, 90% and 80% reliable, one cost unit each,
    with a budget of 3:

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
    >>> best.units, round(best.reliability, 4), best.cost, best.method
    ({'a': 1, 'b': 2}, 0.864, 3.0, 'exact')
    >>> best["units"] is best.units
    True
    """

    units: Dict[Hashable, int]
    reliability: float
    cost: float
    method: str


@dataclass
class AvailabilityResult(_ResultMapping):
    """The result of ``RepairableRBD.availability()``.

    Holds the simulated availability over time, the system's and every
    node's up and down totals, the system failure and restoration counts,
    and the criticality measures, all summed over the ``n_simulations``
    simulations of the window ``[0, time_simulated_to]``. The average
    availability over the window is
    ``system_uptime / (n_simulations * time_simulated_to)``.

    The properties ``mean_up_time``, ``mean_down_time`` and
    ``failure_frequency`` are simulation *estimates* derived from the fields
    below; their exact steady-state counterparts are the
    ``RepairableRBD.mean_up_time()``, ``mean_down_time()`` and
    ``system_failure_frequency()`` methods. ``availability_se`` and
    ``availability_interval`` give the sampling uncertainty of the
    availability curve.

    Like the other result types it is also a read-only mapping of its
    fields, so dict-style access (``result["availability"]``,
    ``result.keys()``, ``dict(result)``) keeps working.

    Attributes
    ----------
    timeline : numpy.ndarray
        Event times at which the (mean) system availability changes: 0,
        every time at which some simulated system changed state, and
        ``time_simulated_to``, in increasing order.
    availability : numpy.ndarray
        Mean system availability at each time in ``timeline``: the fraction
        of the simulated systems that are up from that time until the next
        (the estimated point availability). The last value, at
        ``time_simulated_to``, repeats the one before it.
    system_uptime : float
        Total system uptime summed over all simulations.
    time_simulated_to : float
        The ``t_simulation`` the simulations were run to.
    criticalities : Criticalities
        The importance/criticality measures (see
        [`Criticalities`][repyability.Criticalities]).
    node_uptime : dict
        Total uptime per node summed over all simulations.
    node_downtime : dict
        Total downtime per node summed over all simulations; a node's uptime
        plus downtime equals ``n_simulations * time_simulated_to``.
    system_downtime : float
        Total system downtime summed over all simulations.
    system_failures : int
        Number of system failures observed across all simulations (changes
        from up to down, including the zero-length outages an instantly
        repaired component causes).
    system_restorations : int
        Number of system restorations observed across all simulations.
    n_simulations : int
        The number of simulations run (``N``).
    cost : CostResult, optional
        The simulated cost distribution, when the RBD declares any costs;
        ``None`` when nothing is priced (no cost model to run).

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> rbd = RepairableRBD(
    ...     [("s", "c"), ("c", "t")],
    ...     {
    ...         "c": {
    ...             "reliability": surv.Exponential.from_params([0.1]),
    ...             "repairability": surv.Exponential.from_params([1.0]),
    ...         }
    ...     },
    ... )
    >>> result = rbd.availability(t_simulation=50, N=200, seed=0)
    >>> float(result.availability[0])  # every simulation starts up
    1.0
    >>> window = result.n_simulations * result.time_simulated_to
    >>> round(float(result.system_uptime) / window, 2)  # long run: 10 / 11
    0.91
    >>> lower, upper = result.availability_interval(0.95)
    >>> a = result.availability
    >>> bool(((lower <= a) & (a <= upper)).all())
    True
    >>> result["availability"] is result.availability
    True
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
    cost: Optional[CostResult] = None

    @property
    def availability_se(self) -> np.ndarray:
        """Pointwise standard error of the availability estimate.

        At each time in ``timeline`` the availability is the proportion of
        the ``n_simulations`` systems that were up, so its sampling standard
        error is the binomial ``sqrt(A (1 - A) / n)``. It is 0 where the
        estimate is 0 or 1; ``availability_interval`` stays informative
        there.

        Returns
        -------
        numpy.ndarray
            The standard error at each time in ``timeline``.
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
            The confidence level, strictly between 0 and 1, by default 0.95.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The lower and upper bounds at each time in ``timeline``, within
            ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``confidence`` is not strictly between 0 and 1.
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
        observed (0.0 if the system was never up).

        Note: estimated from a finite window, so each simulation's final
        (unfinished) up period is censored; for windows that are short
        relative to the up-down cycle this biases the estimate. Prefer the
        exact ``RepairableRBD.mean_up_time()`` for steady-state values.

        Returns
        -------
        float
            The estimated mean up time, possibly ``inf``.
        """
        if self.system_failures > 0:
            return self.system_uptime / self.system_failures
        return float("inf") if self.system_uptime > 0 else 0.0

    @property
    def mean_down_time(self) -> float:
        """Simulation estimate of the Mean Down Time,
        ``system downtime / system restorations``. Infinite if downtime was
        observed but never restored (0.0 if the system was never down).

        Note: estimated from a finite window (final unfinished down periods
        are censored), so short windows bias the estimate. Prefer the exact
        ``RepairableRBD.mean_down_time()`` for steady-state values.

        Returns
        -------
        float
            The estimated mean down time, possibly ``inf``.
        """
        if self.system_restorations > 0:
            return self.system_downtime / self.system_restorations
        return float("inf") if self.system_downtime > 0 else 0.0

    @property
    def failure_frequency(self) -> float:
        """Simulation estimate of the system failure frequency (failures per
        unit time), ``system failures / total simulated time``.

        The total simulated time is ``n_simulations * time_simulated_to``.
        The exact steady-state counterpart is
        ``RepairableRBD.system_failure_frequency()``; over a short window the
        estimate is biased, as every simulation starts with its components
        working.

        Returns
        -------
        float
            The estimated number of system failures per unit time.
        """
        return self.system_failures / (
            self.n_simulations * self.time_simulated_to
        )
