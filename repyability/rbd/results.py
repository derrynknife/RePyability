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
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np
from scipy.stats import norm

from repyability.rbd import _montecarlo as montecarlo


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
class UncertaintyResult(_ResultMapping):
    """The spread of a system quantity over plausible node models.

    Returned by ``NonRepairableRBD.sf_uncertainty``. Each of the
    ``n_draws`` draws gives every uncertain node a plausible model (its
    parameters drawn from what is known about them), and the system
    quantity is computed exactly for that draw. The samples therefore
    describe *epistemic* uncertainty, about what the models are, and not
    the aleatory variability that the models themselves describe. Their
    percentiles give uncertainty (credible) intervals; they do not narrow
    as ``n_draws`` grows, which only makes them more precise.

    Attributes
    ----------
    samples : numpy.ndarray
        One value per draw (``n_draws`` values) for a single time, or one
        row per draw and one column per time for an array of times.
    nominal : float or numpy.ndarray
        The value with every node's own model: the point estimate.
    n_draws : int
        The number of draws.

    Examples
    --------
    A pump whose Weibull model was fitted (by surpyval) to 50 failure
    times, in series with a valve that is 99% reliable. The fit's parameter
    covariance gives the draws:

    >>> import numpy as np
    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD
    >>> pump = surv.Weibull.fit(np.linspace(200, 1800, 50))
    >>> valve = surv.FixedEventProbability.from_params(0.01)
    >>> rbd = NonRepairableRBD(
    ...     [("s", "pump"), ("pump", "valve"), ("valve", "t")],
    ...     {"pump": pump, "valve": valve},
    ... )
    >>> result = rbd.sf_uncertainty(500, {"pump": "fit"}, n_draws=5000, seed=0)
    >>> round(result.nominal, 3), round(result.median, 3)
    (0.847, 0.846)
    >>> lower, upper = result.interval(0.9)
    >>> round(lower, 3), round(upper, 3)
    (0.777, 0.906)
    """

    samples: np.ndarray
    nominal: Any
    n_draws: int

    @property
    def mean(self) -> Any:
        """The mean over the draws (per time, for an array of times).

        Returns
        -------
        float or numpy.ndarray
            The mean of ``samples`` over the draws.
        """
        return self._per_time(np.mean(self.samples, axis=0))

    @property
    def median(self) -> Any:
        """The median over the draws (per time, for an array of times).

        Returns
        -------
        float or numpy.ndarray
            The median of ``samples`` over the draws.
        """
        return self._per_time(np.median(self.samples, axis=0))

    @property
    def std(self) -> Any:
        """The standard deviation over the draws (per time).

        Returns
        -------
        float or numpy.ndarray
            The sample standard deviation of ``samples`` over the draws.
        """
        return self._per_time(np.std(self.samples, axis=0, ddof=1))

    def percentile(self, q: float) -> Any:
        """The ``q``-th percentile over the draws (per time).

        Parameters
        ----------
        q : float
            The percentile, in [0, 100].

        Returns
        -------
        float or numpy.ndarray
            The percentile of ``samples`` over the draws.
        """
        return self._per_time(np.percentile(self.samples, q, axis=0))

    def interval(self, level: float = 0.9) -> tuple:
        """The equal-tailed uncertainty interval over the draws (per time).

        Parameters
        ----------
        level : float, optional
            The probability the interval holds, in (0, 1), by default 0.9:
            from the 5th to the 95th percentile.

        Returns
        -------
        tuple
            ``(lower, upper)``: floats, or arrays for an array of times.

        Raises
        ------
        ValueError
            If ``level`` is not in (0, 1).
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level!r}.")
        tail = 50.0 * (1.0 - level)
        return self.percentile(tail), self.percentile(100.0 - tail)

    def _per_time(self, values: np.ndarray) -> Any:
        return float(values) if np.ndim(values) == 0 else values


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
    costs charged at each component failure, plus the cost of each
    preventive replacement, plus any per-component downtime cost, plus the
    system-downtime cost. ``samples`` is that
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
        (both charged per failure; for a hidden failure, when an inspection
        finds it), ``"preventive"`` (charged per preventive replacement),
        ``"inspection"`` (charged per inspection), ``"component_downtime"``
        and ``"system_downtime"``. The six sum to ``mean``.
    by_component : dict
        Mean per-replication cost attributable to each costed component (its
        repair, replace, preventive, inspection and own downtime cost; the
        system-downtime cost is not attributed to components).
    acquisition_cost : float
        The one-off cost of buying the components (the sum of their
        ``"acquisition_cost"``), 0.0 if none is given. It is not a running
        cost, so it is *not* in ``samples`` or ``mean``: the cost of owning
        the system for one window from new is
        ``acquisition_cost + mean``.
    antithetic : bool
        Whether the replications ran in antithetic pairs (replications
        ``2i`` and ``2i + 1``; see ``RepairableRBD.availability``), by
        default False. Each sample is still a correct draw of a window's
        cost, but only the pairs are independent, so ``mean_se`` and
        ``mean_interval`` are worked out from the pairs' means.

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
    acquisition_cost: float = 0.0
    antithetic: bool = False

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
        the simulated mean is likely to be from the true expected cost. For
        an antithetic run, the standard deviation of the pairs' means over
        the square root of their number.

        Returns
        -------
        float
            The standard error of the mean.
        """
        if self.antithetic:
            return montecarlo.standard_error(self.samples, True)
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
class ReliabilityRedundancyAllocation(_ResultMapping):
    """The component reliability and number of copies chosen for each
    node by ``NonRepairableRBD.allocate_reliability_redundancy``.

    Attributes
    ----------
    units : dict
        The number of active copies of each node.
    component_reliability : dict
        The reliability chosen for each node's components (every copy of a
        node is the same).
    reliability : float
        The system reliability.
    cost : float
        The total of the ``"cost"`` resource (or of the first resource).
    resources : dict
        The total of each resource the copies use.

    Examples
    --------
    >>> import math
    >>> from surpyval import FixedEventProbability
    >>> from repyability import NonRepairableRBD
    >>> rbd = NonRepairableRBD(
    ...     [("s", "a"), ("a", "t")],
    ...     {"a": FixedEventProbability.from_params(0.1)},
    ... )
    >>> def cost(r, n):
    ...     return n * (10 + (-1 / math.log(r)) ** 1.5)
    >>> best = rbd.allocate_reliability_redundancy(
    ...     {"a": cost}, budget=50, bounds=(0.5, 0.999)
    ... )
    >>> best.units, round(best.component_reliability["a"], 3)
    ({'a': 3}, 0.754)
    >>> round(best.reliability, 4), round(best.cost, 4)
    (0.9851, 50.0)
    """

    units: Dict[Hashable, int]
    component_reliability: Dict[Hashable, float]
    reliability: float
    cost: float
    resources: Dict[Hashable, float]


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
        Every copy is costed, including the original. With several
        resources, the total of the one a target minimised, of
        ``"cost"``, or of the first resource.
    method : str
        ``"exact"`` (a proven optimum) or ``"greedy"`` (a fast heuristic
        solution, usually but not always optimal).
    resources : dict
        The total of every resource the copies use, keyed by resource
        (``{"cost": ...}`` when ``costs`` gave numbers).
    mix : dict
        For each node given a choice of component types, how many copies
        of each type it uses, ``{node: {type name: copies}}`` (types it
        does not use are left out).
    strategy : dict
        The redundancy strategy of each costed node, ``"active"`` or
        ``"cold"`` (standby).

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
    >>> best.resources
    {'cost': 3.0}
    >>> best["units"] is best.units
    True
    """

    units: Dict[Hashable, int]
    reliability: float
    cost: float
    method: str
    resources: Dict[Hashable, float] = field(default_factory=dict)
    mix: Dict[Hashable, Dict[Hashable, int]] = field(default_factory=dict)
    strategy: Dict[Hashable, str] = field(default_factory=dict)


@dataclass
class TotalCostAllocation(_ResultMapping):
    """The result of ``RepairableRBD.allocate_redundancy()``.

    How many copies of each node give a repairable system the lowest total
    cost of ownership over a horizon: buying the copies, running them
    (repairs, replacements, maintenance, inspections, their own downtime),
    and the cost of the system being down. Like the other result types it
    is also a read-only mapping of its fields.

    Attributes
    ----------
    units : dict
        How many identical copies of each node considered to fit in active
        parallel, each repaired independently. Always at least 1: the
        original unit.
    total_cost : float
        The total cost of owning the system for ``horizon`` with those
        copies, ``acquisition_cost + cost_rate * horizon``: what
        ``RepairableRBD.total_cost(horizon)`` gives for the system with the
        copies drawn out.
    acquisition_cost : float
        The one-off cost of buying every component, each copy included.
    cost_rate : float
        The long-run running cost per unit time: the ``expected_cost_rate``
        of the system with the copies drawn out.
    availability : float
        The system's long-run availability with those copies: its
        ``mean_availability``.
    horizon : float
        The horizon the total cost is taken over.
    method : str
        ``"exact"`` (a proven optimum) or ``"greedy"`` (a fast heuristic
        solution, usually but not always optimal).

    Examples
    --------
    A pump that fails on average every 1000 hours and takes 10 to repair,
    bought for 20,000 and repaired for 500, when an hour without pumping
    costs 100, over ten years (87,600 hours):

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> rbd = RepairableRBD(
    ...     [("s", "pump"), ("pump", "t")],
    ...     {
    ...         "pump": {
    ...             "reliability": surv.Exponential.from_params([1e-3]),
    ...             "repairability": surv.Exponential.from_params([0.1]),
    ...             "repair_cost": 500.0,
    ...             "acquisition_cost": 20000.0,
    ...         }
    ...     },
    ...     downtime_cost_rate=100.0,
    ... )
    >>> best = rbd.allocate_redundancy(87600.0)
    >>> best.units, round(best.total_cost), round(best.acquisition_cost)
    ({'pump': 2}, 127591, 40000)
    >>> round(best.availability, 6), best.method
    (0.999902, 'exact')
    """

    units: Dict[Hashable, int]
    total_cost: float
    acquisition_cost: float
    cost_rate: float
    availability: float
    horizon: float
    method: str


@dataclass
class MaintenancePlan(_ResultMapping):
    """The result of ``RepairableRBD.optimal_replacement_intervals()`` and
    ``RepairableRBD.optimal_inspection_intervals()``.

    The interval chosen for each component, and the system's exact
    long-run values with them. Like the other result types it is also a
    read-only mapping of its fields.

    Attributes
    ----------
    intervals : dict
        Node name -> its chosen interval: of age replacement (``inf`` to
        replace it only when it fails), or of inspection.
    cost_rate : float
        The system's long-run cost per unit time with those intervals: its
        ``expected_cost_rate``.
    availability : float
        The system's long-run availability with them: its
        ``mean_availability``. For a safety system, ``1 - availability`` is
        its average probability of failure on demand, PFDavg.
    """

    intervals: Dict[Hashable, float]
    cost_rate: float
    availability: float


def _meeting(levels: np.ndarray, demand: float) -> np.ndarray:
    """Which ``levels`` meet ``demand``. A level within rounding of it
    meets it: capacities that add up to the demand exactly (three units of
    ``1 / 3`` against a demand of 1) do."""
    if np.isinf(demand):
        return levels >= demand
    return levels >= demand - 1e-9 * abs(demand)


@dataclass
class CapacityDistribution(_ResultMapping):
    """The exact distribution of a system's capacity: how much it can
    deliver.

    Returned by ``NonRepairableRBD.capacity_distribution`` (at a time, or
    at each of several), ``RepairableRBD.capacity_distribution`` (in the
    long run) and ``RBD.system_capacity``. Each component carries its
    capacity while it works and nothing once it has failed, and the
    system's capacity is the most that can flow through the diagram from
    the input to the output. Like the other result types it is also a
    read-only mapping of its fields.

    Attributes
    ----------
    levels : numpy.ndarray
        The capacities the system can have, in increasing order: 0 when it
        is down, and each total its working components can carry. ``inf``
        when components with no capacity given can join the input to the
        output on their own.
    probabilities : numpy.ndarray
        The probability of each level: one per level, or for several times
        one row per level and one column per time. They sum to 1. In the
        long run, the fraction of time the system spends at each level.

    Examples
    --------
    Three pumps of 50 units each, in parallel, each available 90% of the
    time:

    >>> from repyability import RBD
    >>> pumps = RBD(
    ...     [("s", "a"), ("s", "b"), ("s", "c"),
    ...      ("a", "t"), ("b", "t"), ("c", "t")],
    ...     capacity={"a": 50, "b": 50, "c": 50},
    ... )
    >>> capacity = pumps.system_capacity({"a": 0.9, "b": 0.9, "c": 0.9})
    >>> capacity.levels.tolist()
    [0.0, 50.0, 100.0, 150.0]
    >>> capacity.probabilities.round(4).tolist()
    [0.001, 0.027, 0.243, 0.729]

    Two of the three meet a demand of 100:

    >>> round(capacity.meets(100), 4)
    0.972
    >>> round(capacity.mean(), 4)
    135.0
    >>> round(capacity.delivered_fraction(100), 4)
    0.9855
    """

    levels: np.ndarray
    probabilities: np.ndarray

    def meets(self, demand: float) -> Any:
        """The probability that the capacity meets a demand: that it is at
        least ``demand``.

        For a non-repairable system at a time, it is the system's
        reliability for that demand; in the long run, the fraction of time
        the system can meet it. With every component's capacity positive,
        ``meets`` of any demand above 0 but no more than the smallest level
        above 0 is the system's reliability (or availability): the
        capacity is positive exactly when the system works.

        Parameters
        ----------
        demand : float
            The demand, in the capacities' units. A capacity within
            rounding of it (a relative ``1e-9``) meets it.

        Returns
        -------
        float or numpy.ndarray
            The probability, per time for several times.

        Raises
        ------
        ValueError
            If ``demand`` is NaN.
        """
        demand = float(demand)
        if np.isnan(demand):
            raise ValueError("demand must be a number, not NaN.")
        met = _meeting(self.levels, demand)
        return self._per_time(np.sum(self.probabilities[met], axis=0))

    def mean(self) -> Any:
        """The expected capacity.

        In the long run, the average capacity over time. Infinite if the
        capacity can be infinite (see ``levels``).

        Returns
        -------
        float or numpy.ndarray
            The expected capacity, per time for several times.
        """
        levels = self.levels.reshape(
            (-1,) + (1,) * (self.probabilities.ndim - 1)
        )
        with np.errstate(invalid="ignore"):
            parts = np.where(
                self.probabilities > 0, levels * self.probabilities, 0.0
            )
        return self._per_time(np.sum(parts, axis=0))

    def delivered_fraction(self, demand: float) -> Any:
        """The expected fraction of a demand the system delivers:
        ``E[min(capacity, demand)] / demand``.

        A system with more capacity than the demand delivers the demand,
        and one with less delivers what it can. In the long run, this is the
        fraction of the demand met over time: the production availability.

        Parameters
        ----------
        demand : float
            The demand, a positive, finite number in the capacities' units.

        Returns
        -------
        float or numpy.ndarray
            The fraction, in ``[0, 1]``, per time for several times.

        Raises
        ------
        ValueError
            If ``demand`` is not a positive, finite number.
        """
        demand = float(demand)
        if not (np.isfinite(demand) and demand > 0.0):
            raise ValueError(
                f"demand must be a positive, finite number, got {demand!r}."
            )
        delivered = np.minimum(self.levels, demand) / demand
        delivered = delivered.reshape(
            (-1,) + (1,) * (self.probabilities.ndim - 1)
        )
        return self._per_time(np.sum(delivered * self.probabilities, axis=0))

    def _per_time(self, values: np.ndarray) -> Any:
        return float(values) if np.ndim(values) == 0 else values


@dataclass
class AvailabilityAllocation(_ResultMapping):
    """The result of ``RepairableRBD.availability_allocation()`` and
    ``RepairableRBD.mttf_mttr_allocation()``.

    What each component must achieve for the system to meet an availability
    target, and the system's exact long-run availability if they do. Like
    the other result types it is also a read-only mapping of its fields.

    Attributes
    ----------
    availability : dict
        Node name -> the long-run availability allocated to each component.
        The components that keep theirs (see ``availability_allocation``)
        are included, at their own.
    mttf : dict
        Node name -> an MTTF, for each component allocated an availability:
        from ``availability_allocation``, the MTTF that gives it that
        availability at its current MTTR; from ``mttf_mttr_allocation``,
        its MTTF in the cheapest design, together with the MTTR in
        ``mttr``.
    mttr : dict
        Node name -> an MTTR, for the same components: from
        ``availability_allocation``, the MTTR that gives the allocated
        availability at the current MTTF (so either this or the MTTF above
        will do); from ``mttf_mttr_allocation``, its MTTR in the cheapest
        design.
    system_availability : float
        The system's long-run availability with the allocated availabilities,
        as ``mean_availability`` gives it: the target, or more if the system
        already met it.
    """

    availability: Dict[Hashable, float]
    mttf: Dict[Hashable, float]
    mttr: Dict[Hashable, float]
    system_availability: float


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
    availability curve, and ``mean_availability_interval`` that of the
    mean availability over the window.

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
        from up to down caused by a failure, including the zero-length
        outages an instantly repaired component causes).
    system_restorations : int
        Number of system restorations observed across all simulations
        (changes from down to up, after a failure or a planned outage).
    n_simulations : int
        The number of simulations run (``N``).
    cost : CostResult, optional
        The simulated cost distribution, when the RBD declares any costs;
        ``None`` when nothing is priced (no cost model to run).
    system_planned_outages : int
        Number of planned outages of the system observed across all
        simulations: changes from up to down caused by preventive
        maintenance or an inspection that takes time. 0 without either.
    uptimes : numpy.ndarray, optional
        The system's up time in each simulation, in order (they sum to
        ``system_uptime``); None in a result built without them.
    antithetic : bool
        Whether the simulations ran in antithetic pairs (simulations ``2i``
        and ``2i + 1``; see ``RepairableRBD.availability``), by default
        False. ``mean_availability_interval`` then works from the pairs'
        means; the pointwise ``availability_se`` and
        ``availability_interval`` treat the simulations as independent.

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
    system_planned_outages: int = 0
    uptimes: Optional[np.ndarray] = None
    antithetic: bool = False

    def mean_availability_interval(
        self, confidence: float = 0.95
    ) -> ConfidenceInterval:
        """Confidence interval for the expected availability over the window.

        The estimate is the fraction of the window the system was up,
        ``system_uptime / (n_simulations * time_simulated_to)``: the mean,
        over the simulations, of each one's fraction up. By the central
        limit theorem that mean is normal with standard error
        ``std / sqrt(n)`` of the simulations' fractions (of antithetic
        pairs' means, for an antithetic run), from which the interval
        ``estimate +/- z * standard_error`` is built, clipped to [0, 1]. It
        describes the simulation error, and narrows like ``1 / sqrt(n)``;
        ``availability(tolerance=...)`` runs until it is narrow enough.

        Parameters
        ----------
        confidence : float, optional
            The confidence level, strictly between 0 and 1, by default 0.95.

        Returns
        -------
        ConfidenceInterval
            The estimate, bounds, standard error and number of simulations.

        Raises
        ------
        ValueError
            If ``confidence`` is not in (0, 1), or the result has no
            per-simulation up times (one built by hand without them).
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.uptimes is None:
            raise ValueError(
                "This result has no per-simulation up times to estimate "
                "the interval from."
            )
        fractions = np.asarray(self.uptimes, dtype=float) / (
            self.time_simulated_to
        )
        estimate = float(np.mean(fractions))
        se = montecarlo.standard_error(fractions, self.antithetic)
        z = montecarlo.z_value(confidence)
        return ConfidenceInterval(
            estimate=estimate,
            lower=max(0.0, estimate - z * se),
            upper=min(1.0, estimate + z * se),
            confidence=confidence,
            standard_error=se,
            n_samples=len(fractions),
        )

    @property
    def availability_se(self) -> np.ndarray:
        """Pointwise standard error of the availability estimate.

        At each time in ``timeline`` the availability is the proportion of
        the ``n_simulations`` systems that were up, so its sampling standard
        error is the binomial ``sqrt(A (1 - A) / n)``. It is 0 where the
        estimate is 0 or 1; ``availability_interval`` stays informative
        there. It treats the simulations as independent, which antithetic
        ones are not.

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
        ``system uptime / (system failures + planned outages)``: an up
        period ends at either. Infinite if the system was never observed to
        go down (0.0 if it was never up).

        Note: estimated from a finite window, so each simulation's final
        (unfinished) up period is censored; for windows that are short
        relative to the up-down cycle this biases the estimate. Prefer the
        exact ``RepairableRBD.mean_up_time()`` for steady-state values.

        Returns
        -------
        float
            The estimated mean up time, possibly ``inf``.
        """
        outages = self.system_failures + self.system_planned_outages
        if outages > 0:
            return self.system_uptime / outages
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
        unit time), ``system failures / total simulated time``. Planned
        outages are not failures.

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
