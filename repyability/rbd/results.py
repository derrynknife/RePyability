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

A result's values are attributes or properties, read without a call
(``mean``, ``std``, ``failure_frequency``), and what takes an argument is a
method (``mean_availability_interval(confidence)``, ``percentile(q)``,
``stock(probability)``), as on the RBDs, where what is given is an
attribute (``acquisition_cost``) and what is worked out from arguments a
method (``total_cost(t)``). ``SparesDemand``'s ``mean`` and ``std`` were
methods until 0.12 (#184), and ``CapacityDistribution``'s ``mean`` until
0.13 (#235).
"""

import dataclasses
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Tuple, cast

import numpy as np
from scipy.special import ndtri

from repyability.rbd import _montecarlo as montecarlo
from repyability.utils.wrappers import outside_level


def _on_percent_scale(q, name: str) -> None:
    """Warn when ``q``, a percentile on numpy's scale of 0 to 100, lies
    strictly between 0 and 1: more likely a fraction meant as a percent
    (#233)."""
    values = np.atleast_1d(np.asarray(q, dtype=float))
    if np.any((values > 0.0) & (values < 1.0)):
        warnings.warn(
            f"{name}({q!r}) is a percentile on the scale of 0 to 100, as "
            "numpy's is: for the 5th percentile give 5, not 0.05 (interval "
            "takes a level in (0, 1)).",
            UserWarning,
            stacklevel=outside_level(),
        )


def _json_key(key) -> Any:
    """A mapping's key as JSON holds it: as it is if JSON takes it (text, a
    number, a bool or None), else as its text, such as a tuple node name's
    ``"('a', 1)"`` (#235)."""
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    if isinstance(key, np.generic):
        return key.item()
    return str(key)


def plain(value) -> Any:
    """``value`` as plain data, for JSON: a result (or anything
    with a ``to_dict``) as its ``to_dict()``, a dataclass as a dict of its
    fields, an array or a tuple as a list, a numpy number as a Python one,
    a mapping with its keys as JSON holds them (see ``_json_key``), and
    anything else as it is."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(getattr(value, "to_dict", None)) and not isinstance(
        value, type
    ):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: plain(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {_json_key(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    return value


class _ResultMapping(Mapping):
    """Read-only ``Mapping`` view over a dataclass's fields.

    Preserves the dict-style access the results used to have (``result[key]``,
    ``keys``/``items``/``values``, ``in``, ``dict(result)``, iteration) while
    the subclasses add typed, documented attributes.

    A result of a diagram whose models carry a unit (surpyval's
    ``unit=``) carries it as ``model_unit``, and in ``to_dict``.
    """

    @property
    def model_unit(self) -> Optional[str]:
        """The unit of the models of the diagram that gave this result
        (``"hours"``, ``"km"``, ``"cycles"``), as their labels name it:
        the unit its times, rates and lives are in. None when the models
        name no unit, or name different ones (the diagram warned of
        that when it was built)."""
        return self.__dict__.get("_model_unit")

    def to_dict(self) -> dict:
        """The result as plain data, ready for ``json.dumps`` (#235): each
        field by name, arrays as lists, numpy numbers as Python ones, the
        results it holds as their own ``to_dict()``, and keys JSON cannot
        hold (tuple node names, say) as their text.

        Returns
        -------
        dict
            The fields, as plain data. Infinite and undefined values stay
            floats (``inf``, ``nan``), which Python's ``json`` writes as
            ``Infinity`` and ``NaN``. A result whose diagram's models
            carry a unit has it as ``"model_unit"`` too.
        """
        out = {name: plain(getattr(self, name)) for name in self}
        if self.model_unit is not None:
            out["model_unit"] = self.model_unit
        return out

    def __getitem__(self, key):
        if key in self._field_names():
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self._field_names())

    def __len__(self):
        return len(self._field_names())

    def _field_names(self):
        # Every result class is a dataclass (the base is not).
        return tuple(f.name for f in dataclasses.fields(cast(Any, self)))


@dataclass(frozen=True)
class StructureSummary(_ResultMapping):
    """How big a diagram's structure is, found without listing its path or
    cut sets.

    Returned by [`RBD.structure_summary`][repyability.RBD.structure_summary].
    A diagram is first reduced to *modules*: series chains, parallel
    groups and k-out-of-n groups, each independent of the rest. A
    series-parallel diagram reduces to one module. Otherwise something is
    left over, a *core* (a bridge is the smallest), over the modules and
    single nodes around it, and only the core needs the heavier methods.
    The summary says how much of the diagram is each, and how many path
    and cut sets there are, which can be far too many to list. It is an
    immutable dataclass, read as a mapping too, with ``to_dict()`` for
    JSON.

    Attributes
    ----------
    n_components : int
        The components that can decide whether the system works (the
        relevant nodes), leaving out the input and output nodes, any
        junction (a node that never fails) and the irrelevant nodes.
    n_irrelevant : int
        The components in no minimal path set, whose state never changes
        whether the system works (see
        [`find_irrelevant_components`][repyability.RBD.find_irrelevant_components]).
    always_works : bool
        True when the input is joined to the output by an edge that never
        fails, so there is nothing to fail.
    series_parallel : bool
        True when the whole diagram reduces to modules, with no core left.
    n_modules : int
        The series, parallel and k-out-of-n groups the reduction found
        (nested ones counted each), not counting single nodes.
    module_depth : int
        How deeply the modules nest: 0 for a single component, 1 for one
        group of components, and one more for each group held in another.
        A core counts as one more level over the terms it joins.
    core_size : int
        The modules and single nodes the core joins, 0 for a
        series-parallel diagram.
    core_path_sets : int
        The core's minimal path sets (over those modules and nodes), 0 for
        a series-parallel diagram.
    n_path_sets : int or None
        The minimal path sets of the whole diagram. ``None`` when the
        diagram is too meshed to work out exactly.
    n_cut_sets : int or None
        The minimal cut sets of the whole diagram. ``None`` when the
        diagram is too meshed to work out exactly.
    counts_exact : bool
        True when ``n_path_sets`` and ``n_cut_sets`` are exact. A
        component drawn in more than one place (a repeated node) can make
        two counted sets the same set, or one hold another, so then they
        are upper bounds. False also when they are ``None``.
    too_meshed : str or None
        Why the diagram could not be worked out exactly, if it could not;
        ``None`` otherwise.

    Examples
    --------
    A diagram with a cross-link that is not series-parallel: the core joins
    three paths, and there are three minimal path sets.

    >>> from repyability import RBD
    >>> rbd = RBD(
    ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "d"),
    ...      ("a", "e"), ("e", "d"), ("c", "t"), ("d", "t")]
    ... )
    >>> info = rbd.structure_summary()
    >>> info.n_components, info.series_parallel, info.n_path_sets
    (5, False, 3)
    """

    n_components: int
    n_irrelevant: int
    always_works: bool
    series_parallel: bool
    n_modules: int
    module_depth: int
    core_size: int
    core_path_sets: int
    n_path_sets: Optional[int]
    n_cut_sets: Optional[int]
    counts_exact: bool
    too_meshed: Optional[str]


@dataclass
class EventsByWindow(_ResultMapping):
    """What a system is expected to do in each of a run of consecutive
    windows from now: returned by
    [`RepairableRBD.expected_events_by_window`][repyability.RepairableRBD.expected_events_by_window].

    The windows are ``[starts[i], ends[i])``, the first starting at 0
    (now, or the state given). Every count is exact (see
    [`ExpectedEvents`][repyability.ExpectedEvents]) and is the window's
    own, not a running total: the difference of the expected counts at its
    two ends, which add up over windows. A value that numerics leave a
    hair below 0 for a window with no events is shown as 0.

    Attributes
    ----------
    starts : numpy.ndarray
        Each window's start, in the models' time unit.
    ends : numpy.ndarray
        Each window's end.
    system_failures : numpy.ndarray
        The expected system failures in each window.
    system_planned_outages : numpy.ndarray
        The expected planned outages of the system in each window.
    system_downtime : numpy.ndarray
        The expected time the system is down in each window.
    node_failures : dict
        Each component's expected failures in each window (an array per
        component).
    node_corrective : dict
        Each component's expected corrective actions in each window.
    node_preventive : dict
        Each component's expected preventive replacements in each window.
    node_inspections : dict
        Each component's expected tests in each window.
    node_downtime : dict
        Each component's expected down time in each window.
    shutdowns : numpy.ndarray
        The shuts of the diagram's ``shutdowns`` that start in each
        window (planned, so exact counts; zeros without shutdowns).
    system_planned_downtime : numpy.ndarray, optional
        The part of ``system_downtime`` in each window that is planned,
        and ``system_unplanned_downtime`` the rest (see
        [`ExpectedEvents`][repyability.ExpectedEvents]); None where the
        exact methods cannot tell them apart.
    system_unplanned_downtime : numpy.ndarray, optional
        The system's unplanned downtime in each window.
    node_planned_downtime : dict, optional
        Each component's planned downtime in each window.
    node_unplanned_downtime : dict, optional
        Each component's time failed in each window.

    Examples
    --------
    One component with a block replacement every 100 hours, over three
    windows of 100 hours from 40 hours into its interval: its replacement
    is due in 60 hours, and then every 100:

    >>> import surpyval as surv
    >>> from repyability import NodeState, RepairableRBD
    >>> rbd = RepairableRBD(
    ...     [("s", "c"), ("c", "t")],
    ...     {"c": {
    ...         "reliability": surv.Exponential.from_params([1e-9]),
    ...         "repairability": "instant",
    ...         "preventive": {"interval": 100.0, "policy": "block"},
    ...     }},
    ... )
    >>> state = {"c": NodeState(since_scheduled=40.0)}
    >>> windows = rbd.expected_events_by_window(
    ...     width=100.0, count=3, state=state
    ... )
    >>> windows.node_preventive["c"].round(3).tolist()
    [1.0, 1.0, 1.0]
    """

    starts: np.ndarray
    ends: np.ndarray
    system_failures: np.ndarray
    system_planned_outages: np.ndarray
    system_downtime: np.ndarray
    node_failures: Dict[Hashable, np.ndarray]
    node_corrective: Dict[Hashable, np.ndarray]
    node_preventive: Dict[Hashable, np.ndarray]
    node_inspections: Dict[Hashable, np.ndarray]
    node_downtime: Dict[Hashable, np.ndarray]
    shutdowns: Any = None
    system_planned_downtime: Optional[np.ndarray] = None
    system_unplanned_downtime: Optional[np.ndarray] = None
    node_planned_downtime: Optional[Dict[Hashable, np.ndarray]] = None
    node_unplanned_downtime: Optional[Dict[Hashable, np.ndarray]] = None


@dataclass
class CostByWindow(_ResultMapping):
    """What a system costs in each of a run of consecutive windows from
    now, and how much that can vary: returned by
    [`RepairableRBD.cost_by_window`][repyability.RepairableRBD.cost_by_window].

    The windows are ``[starts[i], ends[i])``. ``mean`` is exact. The spread
    comes from one set of simulated histories cut at each window's end, so
    every window's costs are of the same histories (a window's cost is
    the cost to its end less the cost to its start). A cost that comes
    in steps (a replacement at a fixed price, say) has few distinct
    values, and percentiles hide that: ``outcomes`` gives each cost and
    its share, and ``probability_above`` the chance of overspending a
    budget. Ranges do not add over windows: use the ``total`` row for a
    run's total.

    Attributes
    ----------
    starts : numpy.ndarray
        Each window's start.
    ends : numpy.ndarray
        Each window's end.
    mean : numpy.ndarray
        The exact expected cost of each window.
    simulated_mean : numpy.ndarray
        The simulations' own mean of each window's cost (it agrees with
        ``mean`` to within ``standard_error``).
    standard_error : numpy.ndarray
        The standard error of ``simulated_mean``.
    samples : numpy.ndarray
        Each simulation's cost in each window, shape ``(windows,
        simulations)``; none is below 0.
    n_simulations : int
        The number of simulations.
    levels : tuple of float
        The percentiles of ``percentiles``.
    percentiles : numpy.ndarray
        Each window's cost percentiles at ``levels``, shape ``(windows,
        len(levels))``.
    """

    starts: np.ndarray
    ends: np.ndarray
    mean: np.ndarray
    simulated_mean: np.ndarray
    standard_error: np.ndarray
    samples: np.ndarray
    n_simulations: int
    levels: Tuple[float, ...]
    percentiles: np.ndarray

    def percentile(self, q: float) -> np.ndarray:
        """Each window's ``q``-th percentile of cost.

        Parameters
        ----------
        q : float
            The percentile, from 0 to 100.
        """
        return np.percentile(self.samples, q, axis=1)

    def total_percentile(self, q: float) -> float:
        """The ``q``-th percentile of the cost of all the windows
        together: not the sum of the windows' (ranges do not add).

        Parameters
        ----------
        q : float
            The percentile, from 0 to 100.
        """
        return float(np.percentile(self.samples.sum(axis=0), q))

    def probability_above(self, budget: float) -> np.ndarray:
        """Each window's chance of costing more than ``budget``.

        Parameters
        ----------
        budget : float
            The cost to exceed.
        """
        return np.mean(self.samples > budget, axis=1)

    def outcomes(self, window: int, decimals: int = 6) -> list:
        """The distinct costs of a window (counted from 0) and how often
        each happened, cheapest first: ``[{"cost": ..., "share": ...}]``
        (costs rounded to ``decimals`` places).

        Parameters
        ----------
        window : int
            The window, counted from 0.
        decimals : int, optional
            The places costs are rounded to, by default 6.
        """
        costs, counts = np.unique(
            np.round(self.samples[window], decimals), return_counts=True
        )
        return [
            {"cost": float(c), "share": float(n) / self.n_simulations}
            for c, n in zip(costs, counts)
        ]


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
    method : str, optional
        How the estimate was found. For a ``RepairableRBD`` run's mean
        (``mean_availability_interval``, the cost's ``mean_interval``):
        ``"simulated"``, the mean of the simulations' own values;
        ``"control_variate"``, of their values controlled by an exact twin
        (see [`ControlVariate`][repyability.ControlVariate]); ``"exact"``,
        a run controlled by an exact twin that is the system itself, whose
        estimate is then its exact value; ``"conditional"``,
        the mean of each simulation's expected values given its modules'
        histories (see
        [`ConditionalRun`][repyability.ConditionalRun]). Where the method
        chooses how to simulate (``NonRepairableRBD.unreliability_interval``),
        the way it chose; ``"simulated"`` for the other simulated intervals
        (``mean_time_to_failure_interval``,
        ``PhasedMission.reliability_interval``), and ``"exact"`` where
        ``compare`` works the difference out exactly.

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
    method: Optional[str] = None


@dataclass
class ControlVariate(_ResultMapping):
    """The exact twin a simulation run was controlled by, with
    ``RepairableRBD.availability``'s or ``cost``'s ``control_variate``; by
    default, the system itself where the exact methods take it (see
    ``itself``).

    The twin is the system with its components failing and repaired
    independently: the same diagram, components and models, without what
    ties them together (a limit on repair crews, maintenance groups) or
    what the exact methods over time do not take (a standby group's
    switching, its units then operating together; imperfect repair;
    replacement on condition; inspections they do not take), so that its
    expected values over the window are exact. It is simulated alongside
    the system with common random numbers (each of its streams draws what
    the system's stream of the same name draws), so its values move with
    the system's, and its error against its exact value shows how far the
    system's own mean is off. Each simulation's controlled value is ``x -
    coefficient * (twin - exact)``: their mean estimates the system's mean
    without bias (but for the coefficient's coming from the same run, an
    error of order ``1 / n``), with ``1 - correlation**2`` times the
    variance of the plain mean.

    Attributes
    ----------
    twin : numpy.ndarray
        The twin's value in each simulation, in order: its fraction of the
        window up, or its cost.
    exact : float
        The twin's exact expected value over the window
        (``mission_availability``, or ``expected_cost``'s mean).
    coefficient : float
        The multiple of the twin's error taken off: ``cov(x, twin) /
        var(twin)`` over the run (over antithetic pairs' means, for an
        antithetic run), the one that leaves the least variance.
    correlation : float
        The correlation of the system's values with the twin's, over the
        same.
    itself : bool
        Whether the twin is the system itself, whose expected values the
        exact methods work out (independent components, or tied together
        only as the exact methods take them: crews, standby groups or
        common-cause groups their chains follow): its exact value is then
        the system's, which the intervals give, with no error. A run takes
        it by default.

    Examples
    --------
    Four simulations whose twin was up 96.5% of the time on average,
    where its exact mean is 95%: the system's own mean, 95.25%, is taken
    down by about as much.

    >>> import numpy as np
    >>> from repyability import ControlVariate
    >>> x = np.array([0.90, 0.95, 0.99, 0.97])
    >>> twin = np.array([0.92, 0.96, 1.00, 0.98])
    >>> control = ControlVariate.of(x, twin, exact=0.95)
    >>> round(control.coefficient, 4), round(control.correlation, 4)
    (1.1286, 0.9981)
    >>> round(float(control.controlled(x).mean()), 4)
    0.9356
    >>> round(control.variance_reduction)
    261
    """

    twin: np.ndarray
    exact: float
    coefficient: float
    correlation: float
    itself: bool = False

    def __repr__(self) -> str:
        # A summary (#235): each simulation's twin value is in ``twin``.
        return (
            f"ControlVariate(exact={self.exact:.6g}, "
            f"coefficient={self.coefficient:.6g}, "
            f"correlation={self.correlation:.6g}, itself={self.itself}, "
            f"n_simulations={len(np.atleast_1d(self.twin))})"
        )

    @classmethod
    def of(
        cls,
        values,
        twin,
        exact: float,
        antithetic: bool = False,
        itself: bool = False,
    ) -> "ControlVariate":
        """The control of a run whose simulations gave ``values``, by a
        twin that gave ``twin`` (in the same order) and whose exact mean is
        ``exact``.

        Parameters
        ----------
        values : array_like
            The system's value in each simulation.
        twin : array_like
            The twin's value in each simulation.
        exact : float
            The twin's exact expected value.
        antithetic : bool, optional
            Whether the simulations come in antithetic pairs, by default
            False: the coefficient is then worked out from the pairs'
            means.
        itself : bool, optional
            Whether the twin is the system itself, by default False.

        Returns
        -------
        ControlVariate
            The twin's values, its exact value, the coefficient and the
            correlation.
        """
        x = np.asarray(values, dtype=float)
        y = np.asarray(twin, dtype=float)
        if antithetic and len(x) % 2 == 0:
            x = x.reshape(-1, 2).mean(axis=1)
            y_fit = y.reshape(-1, 2).mean(axis=1)
        else:
            y_fit = y
        coefficient = correlation = 0.0
        if len(x) > 1:
            covariance = np.cov(x, y_fit)
            if covariance[1, 1] > 0.0:
                coefficient = float(covariance[0, 1] / covariance[1, 1])
                if covariance[0, 0] > 0.0:
                    correlation = float(
                        covariance[0, 1]
                        / np.sqrt(covariance[0, 0] * covariance[1, 1])
                    )
        return cls(
            twin=y,
            exact=float(exact),
            coefficient=coefficient,
            correlation=correlation,
            itself=bool(itself),
        )

    def _exactly(self, confidence: float, samples: int):
        """The interval of a run controlled by the system itself (see
        ``itself``): its exact value, with no error."""
        return ConfidenceInterval(
            estimate=self.exact,
            lower=self.exact,
            upper=self.exact,
            confidence=confidence,
            standard_error=0.0,
            n_samples=samples,
            method="exact",
        )

    def controlled(self, values) -> np.ndarray:
        """The controlled values of the simulations whose own ``values``
        (in the same order as the twin's) these were.

        Parameters
        ----------
        values : array_like
            The system's value in each simulation.

        Returns
        -------
        numpy.ndarray
            ``values - coefficient * (twin - exact)``.
        """
        return np.asarray(values, dtype=float) - self.coefficient * (
            self.twin - self.exact
        )

    @property
    def variance_reduction(self) -> float:
        """How many times as many plain simulations the controlled estimate
        is worth: ``1 / (1 - correlation**2)``, the plain mean's variance
        over the controlled mean's (infinite for a twin that is the system
        itself).

        Returns
        -------
        float
            The factor.
        """
        left = 1.0 - self.correlation**2
        return 1.0 / left if left > 0.0 else float("inf")


@dataclass
class UncertaintyResult(_ResultMapping):
    """The spread of a system quantity over plausible node models.

    Returned by ``NonRepairableRBD.sf_uncertainty``. Each of the
    ``n_draws`` draws gives every uncertain node a plausible model (its
    parameters drawn from what is known about them), and the system
    quantity is computed exactly for that draw. The samples therefore
    describe *epistemic* uncertainty, about what the models are, and not
    the aleatory variability that the models themselves describe. Their
    percentiles give uncertainty intervals; they do not narrow as
    ``n_draws`` grows, which only makes them more precise.

    What the interval is depends on where the draws came from, which
    ``band`` says: a *confidence* band where every uncertain input is a
    maximum-likelihood fit drawn from its parameter covariance (the
    sampling uncertainty of the estimates); a *credible* band where every
    one is a Bayesian fit drawn from its posterior (how sure the fit is,
    given its prior and data); ``"given"`` where the inputs were drawn from
    distributions or lists of models given; and ``"mixed"`` where the
    inputs are of more than one of these kinds, a band over all of them at
    once (each draw takes every input's), neither a confidence nor a
    credible band alone.

    Its repr summarises the draws: the nominal value, the median and the
    90% interval.

    Attributes
    ----------
    samples : numpy.ndarray
        One value per draw (``n_draws`` values) for a single time, or one
        row per draw and one column per time for an array of times.
    nominal : float or numpy.ndarray
        The value with every node's own model: the point estimate.
    n_draws : int
        The number of draws.
    band : str or None
        What the band over the draws is (see above): ``"confidence"``,
        ``"credible"``, ``"given"`` or ``"mixed"``; None for a result made
        by hand.

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
    >>> result.band  # from the fit's parameter covariance
    'confidence'
    """

    samples: np.ndarray
    nominal: Any
    n_draws: int
    band: Optional[str] = None

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
            The percentile, in [0, 100], as numpy takes it: ``q`` strictly
            between 0 and 1 warns, as it is more likely a fraction meant.

        Returns
        -------
        float or numpy.ndarray
            The percentile of ``samples`` over the draws.
        """
        _on_percent_scale(q, "percentile")
        return self._percentile(q)

    def _percentile(self, q: float) -> Any:
        samples = np.asarray(self.samples, dtype=float)
        if np.isfinite(samples).all():
            return self._per_time(np.percentile(samples, q, axis=0))
        # Infinite draws (a limited failure population's mean, say, #226):
        # numpy's interpolation between two takes inf - inf; the same
        # linear interpolation, but where both ends are one value, it.
        ordered = np.sort(samples, axis=0)
        h = (len(ordered) - 1) * float(q) / 100.0
        low, high = ordered[int(np.floor(h))], ordered[int(np.ceil(h))]
        with np.errstate(invalid="ignore"):
            values = np.where(
                low == high, low, low + (high - low) * (h - np.floor(h))
            )
        return self._per_time(values)

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
        return self._percentile(tail), self._percentile(100.0 - tail)

    def _per_time(self, values: np.ndarray) -> Any:
        return float(values) if np.ndim(values) == 0 else values

    def __repr__(self) -> str:
        # A summary (#184): the samples are in ``samples``.
        lower, upper = self.interval(0.9)

        def shown(value) -> str:
            if np.ndim(value) == 0:
                return f"{float(value):.6g}"
            return np.array2string(
                np.asarray(value, dtype=float),
                precision=6,
                threshold=8,
                separator=", ",
            )

        return (
            f"{type(self).__name__}(nominal={shown(self.nominal)}, "
            f"median={shown(self.median)}, "
            f"interval_90=({shown(lower)}, {shown(upper)}), "
            f"n_draws={self.n_draws}"
            + ("" if self.band is None else f", band={self.band!r}")
            + ")"
        )


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
    >>> result = rbd.availability(t_simulation=50, mc_samples=200, seed=0)
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
    >>> result = rbd.availability(t_simulation=50, mc_samples=200, seed=0)
    >>> fci = result.criticalities.failure_criticality_index
    >>> round(sum(fci.per_system_failure.values()), 4)
    1.0

    In series a node's failure brings the system down unless the other
    node is already down for repair:

    >>> share = fci.per_component_failure
    >>> {node: round(share[node], 2) for node in sorted(share)}
    {'a': 0.92, 'b': 0.89}
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
    >>> result = rbd.availability(t_simulation=50, mc_samples=200, seed=0)
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
    >>> result = rbd.availability(t_simulation=50, mc_samples=200, seed=0)
    >>> crit = result.criticalities

    In series the system is up only while every node is up, but each node
    is down for only about half of the system's down time:

    >>> oci = crit.operational_criticality_index
    >>> {node: round(float(v), 4) for node, v in oci.up.items()}
    {'a': 1.0, 'b': 1.0}
    >>> {node: round(float(v), 2) for node, v in oci.down.items()}
    {'a': 0.5, 'b': 0.55}
    >>> crit["iou"] is crit.iou  # dict-style access also works
    True
    """

    operational_criticality_index: UpDownImportance
    iou: UpDownImportance
    failure_criticality_index: FailureCriticalityIndex
    restoration_criticality_index: RestorationCriticalityIndex


@dataclass
class ConditionalRun(_ResultMapping):
    """How a run's expected values were taken given its modules' histories:
    ``RepairableRBD.availability`` or ``cost`` on a system with
    dependent modules, by default or with ``conditional=True``.

    The modules are the nodes whose values over time the exact methods do
    not work out (a standby group of other lives, a nested RBD that needs
    simulating, imperfect repair, a maintenance group's members, the nodes
    limited repair crews serve). Every other node is independent of them,
    so each simulation's *expected* values given its modules' histories
    are worked out exactly (the system given each joint state of the
    modules, once). Their mean, and its interval
    (``mean_availability_interval``, the cost's ``mean_interval``), estimate
    the window's expected values without bias, and more precisely than the
    simulations' own values, whose randomness from the other nodes is gone.

    By default (``whole``) the whole system is simulated, as a plain run
    simulates it, and only the means (``mean_availability``, the cost's
    ``mean`` and breakdowns) and their intervals take the simulations'
    expected values given their modules (``uptimes``, ``costs``):
    everything else in the result, the spread included, is the
    simulations' own (``sample_mean_availability``, the cost's
    ``sample_mean``). With ``conditional=True`` only the modules are
    simulated, which costs only their events: the result's own values are
    then the expected ones, whose spread is less than a window's own, so
    it gives no percentiles.

    Attributes
    ----------
    modules : tuple
        The nodes simulated as modules, in the order of the RBD's
        components.
    states : int
        The joint states of the modules (each up or down) the simulations
        met: the system was worked out exactly given each. Just one, and
        the modules never changed state in the run: their outages were not
        sampled, and every simulation's expected values given them are the
        same, which says nothing of the error. The means and their
        intervals are then the simulations' own (``method="simulated"``)
        for a whole run, and have no error to give (``nan``) for a run of
        the modules alone, which a ``tolerance`` does not stop: run more
        simulations.
    availability_square : numpy.ndarray, optional
        With ``conditional=True``, at each time of the run's curve, the
        mean over the simulations of the square of each one's chance of
        being up then, for the curve's standard error
        (``AvailabilityResult.availability_se``); None otherwise.
    whole : bool
        Whether the whole system was simulated, by default False: True for
        the default run of a system with modules, whose means alone take
        the values below; False with ``conditional=True``.
    uptimes : numpy.ndarray, optional
        With ``whole``, each simulation's expected up time given its
        modules' histories, in order; None otherwise (with
        ``conditional=True`` they are the result's own ``uptimes``).
    costs : numpy.ndarray, optional
        With ``whole``, each simulation's expected cost given them, in
        order, if anything is priced; None otherwise.
    delivered : numpy.ndarray, optional
        With ``whole``, each simulation's expected fraction of the demand
        delivered given them, in order, if a demand was given; None
        otherwise (with ``conditional=True`` they are the result's own
        ``delivered``).
    """

    modules: Tuple[Hashable, ...]
    states: int
    availability_square: Optional[np.ndarray] = None
    whole: bool = False
    uptimes: Optional[np.ndarray] = None
    costs: Optional[np.ndarray] = None
    delivered: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        # A summary (#235): each simulation's values are in the arrays.
        values = self.uptimes if self.uptimes is not None else self.costs
        parts = [
            f"modules={self.modules!r}",
            f"states={self.states}",
            f"whole={self.whole}",
        ]
        if values is not None:
            parts.append(f"n_simulations={len(np.atleast_1d(values))}")
        return f"ConditionalRun({', '.join(parts)})"

    @property
    def informative(self) -> bool:
        """Whether the modules changed state in the run, so that the
        spread of the simulations' expected values given them estimates
        the error, or there are none, and the values are exact."""
        return self.states > 1 or not self.modules

    @property
    def unknown(self) -> bool:
        """Whether the error of the run's means is unknown: a run of the
        modules alone in which they never changed state."""
        return not self.whole and not self.informative


def _no_spread(conditional: Optional["ConditionalRun"], what: str) -> None:
    """Raise if the values of a run of the modules alone are asked for
    their spread (see ``ConditionalRun``)."""
    if conditional is not None and not conditional.whole:
        raise ValueError(
            f"A conditional run's values are each simulation's expected "
            f"value given its modules' histories (see ConditionalRun), whose "
            f"spread is less than a window's own: {what} needs the whole "
            "system simulated (leave out conditional=True)."
        )


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

    One expected value: ``mean`` is the run's estimate of the
    expected cost of a window, the one ``mean_interval`` gives an interval
    for, and ``cost_rate`` and the breakdowns follow it. By default it is
    exact where the exact methods work it out (``method="exact"``, see
    [`ControlVariate`][repyability.ControlVariate]), and otherwise taken
    given the histories of the system's modules where they apply
    (``"conditional"``, see
    [`ConditionalRun`][repyability.ConditionalRun]); else it is the
    simulations' own (``"simulated"``). The simulations' own mean, the
    average of ``samples``, is ``sample_mean`` whatever the method: with
    ``control_variate=False`` and ``conditional=False`` the two are the
    same.

    Two different uncertainties live here. ``std`` and ``percentile``
    describe how much the cost of a window *varies*; that is a property of
    the system and does not shrink with more replications. ``mean_se`` and
    ``mean_interval`` describe how precisely the *expected* cost has been
    estimated: 0 for an exact one, and otherwise a standard error that
    shrinks like ``1 / sqrt(n_simulations)``.

    Its repr summarises the run: the estimate, its standard error and
    method, the simulations' own mean and spread.

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
        The expected cost of a window split into ``"repair"`` and
        ``"replace"`` (both charged per failure; for a hidden failure, when
        an inspection finds it), ``"preventive"`` (charged per preventive
        replacement), ``"inspection"`` (charged per inspection),
        ``"component_downtime"``, ``"system_downtime"`` and ``"setup"`` (a
        maintenance group's set-up cost, charged once per stop), found as
        ``mean`` is: exact for an exact ``mean``, given the modules'
        histories for a conditional one, and otherwise the simulations' own
        means. The seven sum to ``mean``, but for a run controlled by a
        twin that is not the system itself (``control_variate=True``),
        whose breakdowns are the simulations' own and sum to
        ``sample_mean``.
    by_component : dict
        The expected cost attributable to each costed component (its
        repair, replace, preventive, inspection and own downtime cost; the
        system-downtime cost is not attributed to components), found as
        ``by_category`` is.
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
        cost, but only the pairs are independent, so ``mean_se``,
        ``sample_se`` and ``mean_interval`` are worked out from the pairs'
        means.
    control_variate : ControlVariate, optional
        The exact twin the run was controlled by (see
        [`ControlVariate`][repyability.ControlVariate]): by default the
        system itself, where the exact methods work out its expected cost,
        whose ``mean`` is then that cost; with
        ``control_variate=True``, the system without what ties its
        components together, whose ``mean`` is the controlled estimate.
        ``samples`` and ``sample_mean`` stay the simulations' own. None
        otherwise.
    conditional : ConditionalRun, optional
        For a system with dependent modules (see
        [`ConditionalRun`][repyability.ConditionalRun]): by default
        (``whole``), ``mean`` is the mean of each simulation's expected
        cost given its modules' histories, while ``samples`` and
        ``sample_mean`` stay the simulations' own. With
        ``conditional=True``, ``samples`` are those expected costs, so
        ``mean`` is ``sample_mean``, and ``std`` and ``percentile`` refuse.
        None otherwise.
    lost_production_planned : float, optional
        The lost production (``downtime_cost_rate`` times the system's
        downtime) of the system's *planned* downtime, the simulations' own
        mean per window: while it is down only because blocks are in
        planned work (preventive maintenance or a test of a working unit
        that takes time, or a shut). See
        [`RepairableRBD.maintenance_effect`][repyability.RepairableRBD.maintenance_effect]
        for the definition, and its exact value. None for a result built
        without it (a run with ``conditional=True``).
    lost_production_unplanned : float, optional
        The same for the system's *unplanned* downtime: while it would
        still be down with every planned task, test and shut in progress
        finished. The two add up to the simulations' own mean of the
        ``"system_downtime"`` cost.

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
    >>> result = rbd.cost(t_simulation=100.0, mc_samples=200, seed=0)

    One component's expected cost over a window is exact, so by default
    ``mean`` is that cost, with no error, and the breakdowns split it;
    the simulations' own mean and spread are beside it:

    >>> round(result.mean, 2), result.mean_se
    (1360.33, 0.0)
    >>> round(result.by_category["repair"], 2)  # 100 per failure
    909.92
    >>> round(result.by_category["system_downtime"], 2)  # 50 per hour down
    450.41
    >>> round(result.sample_mean, 2), round(result.std, 2)
    (1387.42, 436.54)
    >>> result.mean_interval(0.95).method
    'exact'

    The simulations' own estimate, and its interval, are a
    ``control_variate=False`` away:

    >>> own = rbd.cost(
    ...     t_simulation=100.0, mc_samples=200, seed=0, control_variate=False
    ... )
    >>> round(own.mean, 2), round(own.by_category["repair"], 2)
    (1387.42, 927.0)
    >>> interval = own.mean_interval(0.95)
    >>> interval.method, round(interval.lower, 2), round(interval.upper, 2)
    ('simulated', 1326.92, 1447.92)
    """

    samples: np.ndarray
    t_simulation: float
    n_simulations: int
    by_category: Dict[str, float]
    by_component: Dict[Hashable, float]
    acquisition_cost: float = 0.0
    antithetic: bool = False
    control_variate: Optional[ControlVariate] = None
    conditional: Optional[ConditionalRun] = None
    lost_production_planned: Optional[float] = None
    lost_production_unplanned: Optional[float] = None

    def _estimate(self) -> Tuple[float, float, str]:
        """The run's estimate of the expected cost of a window, its
        standard error and how it was found (see ``mean_interval``)."""
        conditional = self.conditional
        control = self.control_variate
        if control is not None:
            if control.itself:
                return control.exact, 0.0, "exact"
            values = control.controlled(self.samples)
            method = "control_variate"
        elif (
            conditional is not None
            and conditional.whole
            and conditional.informative
        ):
            assert conditional.costs is not None
            values = np.asarray(conditional.costs, dtype=float)
            method = "conditional"
        else:
            values = np.asarray(self.samples, dtype=float)
            # A run of the modules alone: its own values are theirs.
            method = (
                "conditional"
                if conditional is not None and not conditional.whole
                else "simulated"
            )
        error = montecarlo.standard_error(values, self.antithetic)
        if conditional is not None and conditional.unknown:
            # The modules never changed state (#215).
            error = math.nan
        return float(np.mean(values)), error, method

    @property
    def mean(self) -> float:
        """The run's estimate of the expected cost of a window: the
        estimate ``mean_interval`` gives, exact where the exact methods
        work it out, given the modules' histories where those apply, and
        otherwise the simulations' own, ``sample_mean`` (see the class's
        notes and ``mean_interval().method``).

        Returns
        -------
        float
            The estimated expected cost over the window.
        """
        return self._estimate()[0]

    @property
    def sample_mean(self) -> float:
        """The simulations' own mean cost over the window: the average of
        ``samples``, whatever ``mean`` is.

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

        Raises
        ------
        ValueError
            For a conditional run (see ``conditional``).
        """
        _no_spread(self.conditional, "the cost's standard deviation")
        return float(np.std(self.samples, ddof=1))

    @property
    def mean_se(self) -> float:
        """The standard error of ``mean``: how far it is likely to be from
        the true expected cost. 0.0 for an exact ``mean``; otherwise that of
        the values it is the mean of, ``std / sqrt(n_simulations)`` of the
        simulations' own (``sample_se``), controlled or conditional ones;
        for an antithetic run, of the pairs' means. nan where it cannot be
        known (fewer than two simulations, or a conditional run whose
        modules never changed state).

        Returns
        -------
        float
            The standard error of the mean.
        """
        return self._estimate()[1]

    @property
    def sample_se(self) -> float:
        """The standard error of ``sample_mean``, ``std / sqrt(
        n_simulations)`` (for an antithetic run, of the pairs' means; nan
        for fewer than two).

        Returns
        -------
        float
            The standard error of the simulations' own mean.
        """
        return montecarlo.standard_error(self.samples, self.antithetic)

    def mean_interval(self, confidence: float = 0.95) -> ConfidenceInterval:
        """Confidence interval for the expected total cost over the window.

        Around ``mean``, with standard error ``mean_se``: by the central
        limit theorem a mean of simulations' values is normal, so the
        interval is ``mean +/- z * mean_se`` for the normal quantile ``z``
        of ``confidence``, its lower bound clipped at 0; an exact ``mean``
        has no error, and the interval is that value. Use it to judge
        whether ``mc_samples`` was large enough; for the range a single
        window's cost could fall in, use ``percentile`` instead. Its
        ``method`` says how ``mean`` was found: ``"exact"``, by the exact
        methods (a ``control_variate`` that is the system itself, see
        [`ControlVariate`][repyability.ControlVariate]);
        ``"control_variate"``, the simulations' values controlled by an
        exact twin; ``"conditional"``, each simulation's expected cost
        given its modules' histories (see
        [`ConditionalRun`][repyability.ConditionalRun]); ``"simulated"``,
        the simulations' own.

        Parameters
        ----------
        confidence : float, optional
            The confidence level, strictly between 0 and 1, by default 0.95.

        Returns
        -------
        ConfidenceInterval
            The estimate (``mean``), bounds, standard error, sample count
            and method.

        Raises
        ------
        ValueError
            If ``confidence`` is not strictly between 0 and 1.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.control_variate is not None and self.control_variate.itself:
            return self.control_variate._exactly(confidence, len(self.samples))
        estimate, standard_error, method = self._estimate()
        z = float(ndtri(0.5 + confidence / 2.0))
        unknown = np.isnan(standard_error)
        return ConfidenceInterval(
            estimate=estimate,
            lower=(
                np.nan if unknown else max(0.0, estimate - z * standard_error)
            ),
            upper=np.nan if unknown else estimate + z * standard_error,
            confidence=confidence,
            standard_error=standard_error,
            n_samples=len(self.samples),
            method=method,
        )

    @property
    def cost_rate(self) -> float:
        """The expected cost per unit time over the window, ``mean /
        t_simulation``; it converges to the exact
        ``RepairableRBD.expected_cost_rate()`` as the window grows.

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
            The percentile, between 0 and 100: ``q`` strictly between 0
            and 1 warns, as it is more likely a fraction meant.

        Returns
        -------
        float
            The cost below which about ``q`` percent of the simulated
            windows fall.

        Raises
        ------
        ValueError
            If ``q`` is outside ``[0, 100]``, or for a conditional run (see
            ``conditional``).
        """
        _no_spread(self.conditional, "a percentile of the cost")
        _on_percent_scale(q, "percentile")
        return float(np.percentile(self.samples, q))

    def __repr__(self) -> str:
        # A summary (#223, #235): the samples are in ``samples``.
        estimate, error, method = self._estimate()
        parts = [
            f"mean={estimate:.6g}",
            f"mean_se={error:.3g}",
            f"method={method!r}",
            f"sample_mean={self.sample_mean:.6g}",
        ]
        if self.conditional is None or self.conditional.whole:
            if len(self.samples) > 1:
                parts.append(f"std={float(np.std(self.samples, ddof=1)):.6g}")
        parts += [
            f"n_simulations={self.n_simulations}",
            f"t_simulation={self.t_simulation:g}",
        ]
        if self.acquisition_cost:
            parts.append(f"acquisition_cost={self.acquisition_cost:g}")
        return f"{type(self).__name__}({', '.join(parts)})"


@dataclass
class ExpectedEvents(_ResultMapping):
    """What a system is expected to do over a window from new: returned by
    ``RepairableRBD.expected_events``.

    Every value is exact (numerical, with no simulation): the mean of what
    ``RepairableRBD.availability`` counts in each simulation of the window,
    and divides by ``n_simulations``. Events at the window's end itself
    fall outside it, as in the simulation. Each value is a float for one
    window, or an array in the shape of the windows given.

    Attributes
    ----------
    window : float or numpy.ndarray
        The window's length, from new.
    system_failures : float or numpy.ndarray
        The expected number of system failures in the window (changes from
        up to down caused by a failure, including the zero-length outages
        an instantly repaired component causes).
    system_planned_outages : float or numpy.ndarray
        The expected number of planned outages of the system: changes from
        up to down caused by preventive maintenance that takes time.
    system_downtime : float or numpy.ndarray
        The expected time the system is down in the window,
        ``window * (1 - mission_availability(window))``.
    node_failures : dict
        Each component's expected number of failures (a nested RBD's: its
        system failures).
    node_corrective : dict
        Each component's expected number of corrective actions, at each of
        which ``repair_cost`` and ``replace_cost`` are charged: its failures,
        or for hidden failures, those found by a test in the window (a
        nested RBD's: its system failures).
    node_preventive : dict
        Each component's expected number of preventive replacements (age or
        block), at each of which its preventive cost is charged (a nested
        RBD's: its planned outages, as the simulation counts them for the
        node; its own components' maintenance is counted by its own
        ``expected_events``).
    node_inspections : dict
        Each component's expected number of tests (of hidden failures).
    node_downtime : dict
        Each component's expected time down in the window (with a
        shutdown, the time it holds the component off line included).
    shutdowns : float or numpy.ndarray
        The number of shuts of the diagram's ``shutdowns`` that start in
        the window (0 without shutdowns). A shut is down time of the
        system (``system_downtime``) and, where it takes the system down
        from up, one of its planned outages.
    system_planned_downtime : float or numpy.ndarray, optional
        The part of ``system_downtime`` that is *planned*: the system down
        only because blocks are in planned work (preventive maintenance
        or a test of a working unit that takes time, or a shut). See
        [`RepairableRBD.maintenance_effect`][repyability.RepairableRBD.maintenance_effect]
        for the definition. None where the exact methods cannot tell the
        two apart (planned outages with limited repair crews or
        common-cause groups): ``availability`` simulates them.
    system_unplanned_downtime : float or numpy.ndarray, optional
        The rest of ``system_downtime``, *unplanned*: the system down
        because of failures, so that it would still be down with every
        planned task, test and shut in progress finished. The two add up
        to ``system_downtime``.
    node_planned_downtime : dict, optional
        Each component's part of its ``node_downtime`` in planned work
        (maintenance or a test that takes time, or held off line by a
        shut while it has not failed; a nested RBD's, its own planned
        downtime).
    node_unplanned_downtime : dict, optional
        Each component's time failed: in a repair, waiting for one, or
        failed unseen until a test finds it. The two add up to its
        ``node_downtime``.

    Examples
    --------
    One component with failure rate 0.1 and repair rate 1, over 10 time
    units, fails ``0.1 * (10 / 1.1 + 0.1 / 1.1 ** 2 * (1 - exp(-11)))``
    times on average:

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
    >>> window = rbd.expected_events(10.0)
    >>> round(window.system_failures, 4)
    0.9174
    >>> round(window.node_downtime["c"], 3)
    0.826
    """

    window: Any
    system_failures: Any
    system_planned_outages: Any
    system_downtime: Any
    node_failures: Dict[Hashable, Any]
    node_corrective: Dict[Hashable, Any]
    node_preventive: Dict[Hashable, Any]
    node_inspections: Dict[Hashable, Any]
    node_downtime: Dict[Hashable, Any]
    shutdowns: Any = 0.0
    system_planned_downtime: Any = None
    system_unplanned_downtime: Any = None
    node_planned_downtime: Optional[Dict[Hashable, Any]] = None
    node_unplanned_downtime: Optional[Dict[Hashable, Any]] = None


@dataclass
class RateBreakdown(_ResultMapping):
    """How fast a system's availability (or reliability) is changing at
    each time, and each node's part in it: returned by
    ``RepairableRBD.availability_rate`` and
    ``NonRepairableRBD.reliability_rate``.

    With independent nodes the system's value is multilinear in theirs, so
    its rate is the sum, over the nodes, of each one's Birnbaum importance
    times its own rate: ``rate`` is ``sum(node_rate.values())``, and a
    node's part says how much it is pulling the system down (negative) or
    up then. Where a scheduled event makes a node's availability jump (a
    block replacement or test that takes it off line), the system's may
    jump too: ``jumps`` holds the system's jumps, split among the nodes in
    ``node_jumps``, whose parts add up to them.

    Attributes
    ----------
    x : float or numpy.ndarray
        The times.
    rate : float or numpy.ndarray
        The system's rate of change at each time, per unit time: after
        anything that happens at it.
    node_rate : dict
        Each node's part in the rate.
    jump_times : numpy.ndarray
        The times, after 0 and up to the last of ``x``, at which a node's
        value jumps (and so the system's may).
    jumps : numpy.ndarray
        The system's jump at each.
    node_jumps : dict
        Each node's part in the jumps.

    Examples
    --------
    One unit, failing at rate 0.1 and repaired at rate 1, from new: its
    availability ``A(t) = (1 + 0.1 exp(-1.1 t)) / 1.1`` falls at
    ``0.1 exp(-1.1 t)``:

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
    >>> round(rbd.availability_rate(1.0).rate, 5)  # -0.1 * exp(-1.1)
    -0.03329
    """

    x: Any
    rate: Any
    node_rate: Dict[Hashable, Any]
    jump_times: np.ndarray
    jumps: np.ndarray
    node_jumps: Dict[Hashable, np.ndarray]


@dataclass
class UncertaintyImportance(_ResultMapping):
    """Each uncertain input's part in the uncertainty of a system quantity:
    returned by ``NonRepairableRBD.uncertainty_importance``.

    The inputs are the uncertainties given (a node, a tuple of nodes of one
    population, a common-cause group's model), by the keys they were given
    under. Which input's uncertainty widens the interval most is where more
    data would narrow it most.

    Attributes
    ----------
    method : str
        ``"delta"`` (the delta method) or ``"sobol"`` (variance-based, from
        draws).
    variance : float or numpy.ndarray
        The quantity's variance over the inputs' uncertainty: the delta
        method's approximation, the sum of its parts, or the draws'.
    first_order : dict
        Each input's first-order share of the variance. The delta method's
        part, ``g^T Sigma g`` over the variance, ``g`` the quantity's
        gradient in the input's parameters and ``Sigma`` their covariance:
        the shares add up to 1. Sobol's first-order index,
        ``Var(E[Q | input]) / Var(Q)``: the share of the variance that
        knowing the input exactly would remove.
    total : dict
        Each input's total share: Sobol's total index,
        ``E[Var(Q | the others)] / Var(Q)``, which counts its interactions
        with the others too (a total well above the first-order share says
        the input matters through them). The delta method's, its part again
        (a linear approximation has no interactions).

    Examples
    --------
    >>> import surpyval as surv
    >>> import scipy.stats as st
    >>> from repyability import NonRepairableRBD
    >>> rbd = NonRepairableRBD(
    ...     [("s", "a"), ("a", "b"), ("b", "t")],
    ...     {
    ...         "a": surv.Exponential.from_params([0.01]),
    ...         "b": surv.Exponential.from_params([0.01]),
    ...     },
    ... )
    >>> rates = {
    ...     "a": {"failure_rate": st.uniform(0.005, 0.01)},
    ...     "b": {"failure_rate": st.uniform(0.008, 0.004)},
    ... }
    >>> parts = rbd.uncertainty_importance(10.0, rates)
    >>> {node: round(share, 3) for node, share in parts.first_order.items()}
    {'a': 0.862, 'b': 0.138}
    """

    method: str
    variance: Any
    first_order: Dict[Hashable, Any]
    total: Dict[Hashable, Any]


@dataclass
class ExpectedCost(_ResultMapping):
    """The expected cost of running a system over a window from new:
    returned by ``RepairableRBD.expected_cost``.

    Exact (numerical, with no simulation): the mean that
    ``RepairableRBD.cost`` estimates, by the same categories, from each
    category's expected events (see
    [`ExpectedEvents`][repyability.ExpectedEvents]) times its mean cost,
    and the expected downtimes times their rates. Each value is a float for
    one window, or an array in the shape of the windows given.

    Attributes
    ----------
    window : float or numpy.ndarray
        The window's length, from new.
    mean : float or numpy.ndarray
        The expected running cost of the window: the sum of
        ``by_category``.
    by_category : dict
        The expected cost of ``"repair"`` and ``"replace"`` (per corrective
        action), ``"preventive"`` (per preventive replacement),
        ``"inspection"`` (per test), ``"component_downtime"``,
        ``"system_downtime"`` and ``"setup"`` (once per stop of a
        maintenance group), as in ``CostResult``.
    by_component : dict
        The expected cost attributable to each costed component (its
        repair, replace, preventive, inspection and own downtime cost).
    acquisition_cost : float
        The one-off cost of buying the components, not in ``mean``.
    discount_rate : float
        The continuous rate the costs were discounted at: with one
        above 0, every value but ``acquisition_cost`` (paid at the start)
        is a present value. By default 0.

    Examples
    --------
    One component with MTTF 10 and MTTR 1, at 100 per repair, and 50 per
    unit time of system downtime, over 100 time units:

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
    >>> cost = rbd.expected_cost(100.0)
    >>> round(cost.mean, 2)  # cost() simulates about this, with its spread
    1360.33
    >>> round(cost.by_category["repair"], 2)  # 100 per failure
    909.92
    >>> round(cost.by_category["system_downtime"], 2)  # 50 per unit down
    450.41
    >>> round(cost.cost_rate, 2), round(rbd.expected_cost_rate(), 2)
    (13.6, 13.64)
    """

    window: Any
    mean: Any
    by_category: Dict[str, Any]
    by_component: Dict[Hashable, Any]
    acquisition_cost: float = 0.0
    discount_rate: float = 0.0

    @property
    def total(self) -> Any:
        """The cost of owning the system for the window from new:
        ``acquisition_cost + mean``.

        Returns
        -------
        float or numpy.ndarray
            The total cost for each window.
        """
        return self.acquisition_cost + self.mean

    @property
    def cost_rate(self) -> Any:
        """The mean cost per unit time over the window, ``mean / window``
        (nan for a window of 0); it approaches
        ``RepairableRBD.expected_cost_rate()`` as the window grows (but
        for a discounted cost, whose rate falls away).

        Returns
        -------
        float or numpy.ndarray
            The cost rate for each window.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.divide(self.mean, self.window)
        return float(rate) if np.ndim(rate) == 0 else rate


@dataclass
class MaintenanceEffect(_ResultMapping):
    """What a diagram's planned work does: how much of its downtime and
    lost production is planned and how much is from failures, and the
    failures, downtime and cost it saves against running to failure.
    Returned by
    [`RepairableRBD.maintenance_effect`][repyability.RepairableRBD.maintenance_effect].

    **Planned and unplanned.** The system is down *unplanned* at an
    instant if it would still be down with every planned task, test and
    shut in progress finished: if its structure, with every block in
    planned work counted as up, is down. Its *planned* downtime is the
    rest: down only because blocks are in planned work (preventive
    maintenance or a test of a working unit that takes time, or held off
    line by a shut). A block that has failed (in a repair, waiting for a
    crew, or failed unseen until a test finds it) is down unplanned, also
    while a shut holds it. Lost production (the system's
    ``downtime_cost_rate``) splits the same way, and a block's own
    downtime cost by its own state.

    **Against running to failure.** The same diagram with no planned work
    (no ``"preventive"`` task, no shutdowns, no maintenance groups; its
    tests of hidden failures kept, as a test does not avoid failures):
    ``*_without`` its values, and ``*_avoided`` or ``*_saved`` the plan's
    less them.

    Over a ``window`` each value is the expected total in it (from the
    state given); in the long run (``window`` None) each is per unit
    time: a downtime is a share of the time (an unavailability), failures
    and costs are rates.

    Attributes
    ----------
    window : float or None
        The window's length, or None for the long run.
    method : str
        ``"exact"`` (numerical, no simulation) or ``"simulated"``.
    system_downtime : float
        The system's expected downtime with the plan.
    system_planned_downtime : float
        Its planned part.
    system_unplanned_downtime : float
        Its unplanned part; the two add up to ``system_downtime``.
    lost_production : float
        ``downtime_cost_rate`` times ``system_downtime``.
    lost_production_planned : float
        Of the planned downtime.
    lost_production_unplanned : float
        Of the unplanned downtime.
    component_downtime_cost_planned : float
        The blocks' own downtime costs (their ``downtime_cost``) while in
        planned work; None when simulated.
    component_downtime_cost_unplanned : float
        While failed; None when simulated.
    node_planned_downtime : dict
        Each component's downtime in planned work; None when simulated.
    node_unplanned_downtime : dict
        Each component's time failed; None when simulated.
    system_failures : float
        The system's expected failures with the plan.
    system_failures_without : float
        Running to failure.
    system_failures_avoided : float
        ``system_failures_without - system_failures``.
    node_failures : dict
        Each component's expected failures with the plan (its corrective
        actions in the long run).
    node_failures_without : dict
        Running to failure.
    node_failures_avoided : dict
        Without less with.
    system_downtime_without : float
        The system's downtime running to failure.
    downtime_saved : float
        ``system_downtime_without - system_downtime`` (negative when the
        plan's own downtime is more than the failures' it saves).
    cost : float
        The expected running cost with the plan (a rate in the long run;
        0 with nothing priced).
    cost_without : float
        Running to failure.
    cost_saved : float
        ``cost_without - cost`` (negative when the plan costs more than it
        saves).
    mc_samples : int or None
        Simulated: the simulations of each diagram; None when exact.
    standard_errors : dict or None
        Simulated: the standard errors of ``downtime_saved`` and
        ``cost_saved``, from the differences of paired simulations (the
        two diagrams run with the same seed, so each simulation of the
        plan draws the lives the one without it draws, as far as their
        events agree); ``system_failures_avoided`` is the difference of
        the two runs' means. None when exact.

    Examples
    --------
    A pump under age replacement every 600 hours, each taking 8 hours,
    repaired in about 20 hours when it fails, at 100 an hour of lost
    production:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> pump = {
    ...     "reliability": surv.Weibull.from_params([1000.0, 3.0]),
    ...     "repairability": surv.Exponential.from_params([1 / 20.0]),
    ...     "preventive": {
    ...         "interval": 600.0,
    ...         "duration": surv.ExactEventTime.from_params([8.0]),
    ...     },
    ... }
    >>> rbd = RepairableRBD(
    ...     [("s", "p"), ("p", "t")], {"p": pump}, downtime_cost_rate=100.0
    ... )
    >>> effect = rbd.maintenance_effect()  # the long run
    >>> round(effect.system_planned_downtime, 5)  # share of the time
    0.01112
    >>> round(effect.system_unplanned_downtime, 5)
    0.0067
    >>> year = 8760.0
    >>> round(effect.system_failures * year, 2)  # failures a year
    2.93
    >>> round(effect.system_failures_avoided * year, 2)  # the plan avoids
    6.66
    >>> round(effect.lost_production_unplanned * year)  # a year
    5870
    """

    window: Optional[float]
    method: str
    system_downtime: float
    system_planned_downtime: float
    system_unplanned_downtime: float
    lost_production: float
    lost_production_planned: float
    lost_production_unplanned: float
    component_downtime_cost_planned: Optional[float]
    component_downtime_cost_unplanned: Optional[float]
    node_planned_downtime: Optional[Dict[Hashable, float]]
    node_unplanned_downtime: Optional[Dict[Hashable, float]]
    system_failures: float
    system_failures_without: float
    system_failures_avoided: float
    node_failures: Dict[Hashable, float]
    node_failures_without: Dict[Hashable, float]
    node_failures_avoided: Dict[Hashable, float]
    system_downtime_without: float
    downtime_saved: float
    cost: float
    cost_without: float
    cost_saved: float
    mc_samples: Optional[int] = None
    standard_errors: Optional[Dict[str, float]] = None


@dataclass
class ExpectedNPV(_ResultMapping):
    """What running a system is worth, over a horizon from new or from a
    state: returned by ``RepairableRBD.expected_npv``.

    In plain terms: the net present value (NPV) is what the system earns
    while it works, less what it costs to own and run, each amount counted
    at what it is worth now (an amount ``t`` from now is worth
    ``exp(-r t)`` of itself). It is the average over every way the
    failures could fall, worked out exactly (no simulation) from the
    expected number of events and the expected time up: the average of
    the present values is the present value of the averages. It says
    nothing of the spread; a bad run can lose what the average gains.

    The equivalent annual cost (``eac``) is the steady cost per unit time
    over the horizon that is worth the same as all the costs: the figure to
    compare plans of different lengths or lives. With no discounting it
    is the total cost over the horizon divided by its length.

    Attributes
    ----------
    horizon : float
        The length of time valued, in the models' unit.
    discount_rate : float
        The continuous rate per unit time the amounts were discounted at
        (0: not discounted).
    value_rate : float
        The value earned per unit time while the system works.
    npv : float
        The expected net present value: ``value - cost``.
    value : float
        The present value of what the system earns while it works.
    cost : float
        The present value of every cost: the running costs and the
        acquisition cost (paid at the start).
    eac : float
        The equivalent annual cost: ``cost`` divided by the
        ``annuity_factor``, per unit of the models' time.
    undiscounted_npv : float
        The same sum with no discounting.
    undiscounted_value : float
        The expected time up times ``value_rate``, undiscounted.
    undiscounted_cost : float
        The expected cost over the horizon with no discounting, with the
        acquisition cost.
    cost_by_category : dict
        The present value of the cost in each category (``"repair"``,
        ``"replace"``, ``"preventive"``, ``"inspection"``,
        ``"component_downtime"``, ``"system_downtime"``, ``"setup"``, and
        ``"acquisition"``), as positive amounts.
    cost_by_component : dict
        The present value of the cost attributable to each costed
        component, with its acquisition cost.
    period_ends : list
        Where each period of the cash-flow table ends, in the models'
        time: the last is the horizon.
    cash_flows : list
        The cash-flow table: one dict per period, category and component
        with something in it, with ``period`` (from 1), ``start``, ``end``,
        ``category`` (a cost category, or ``"value"``), ``component`` (None
        for value, the system's downtime and a group's set-up),
        ``present_value`` and ``undiscounted``, both signed: costs
        negative, value positive. The ``present_value`` of all rows add up
        to ``npv``. Costs given by period and category are the exact
        increments of the expected cost; the table is for applying your
        own tax, depreciation and financial-year conventions outside the
        package.

    Examples
    --------
    A pump earning 10 per hour while it works, failing on average every
    1000 hours, repaired in 10 at 500 per repair, bought for 20,000,
    valued over 5 years of 8,760 hours at 8% a year:

    >>> import math
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> rbd = RepairableRBD(
    ...     [("s", "p"), ("p", "t")],
    ...     {
    ...         "p": {
    ...             "reliability": surv.Exponential.from_params([1e-3]),
    ...             "repairability": surv.Exponential.from_params([0.1]),
    ...             "repair_cost": 500.0,
    ...             "acquisition_cost": 20000.0,
    ...         }
    ...     },
    ... )
    >>> r = math.log(1.08) / 8760  # per hour
    >>> result = rbd.expected_npv(
    ...     5 * 8760.0, discount_rate=r, value_rate=10.0, periods=5
    ... )
    >>> round(result.npv), round(result.value), round(result.cost)
    (321975, 359974, 37999)
    >>> round(result.eac * 8760)  # per year
    9155
    """

    horizon: float
    discount_rate: float
    value_rate: float
    npv: float
    value: float
    cost: float
    eac: float
    undiscounted_npv: float
    undiscounted_value: float
    undiscounted_cost: float
    cost_by_category: Dict[str, float]
    cost_by_component: Dict[Hashable, float]
    period_ends: List[float]
    cash_flows: List[Dict[str, Any]]

    @property
    def annuity_factor(self) -> float:
        """What one unit of cost per unit time, steady over the horizon,
        is worth now: ``(1 - exp(-r H)) / r`` for the ``discount_rate``
        ``r`` and the ``horizon`` ``H``, and ``H`` with none.

        Returns
        -------
        float
            The factor ``eac`` divides the cost by.
        """
        if not self.discount_rate:
            return self.horizon
        return float(
            -np.expm1(-self.discount_rate * self.horizon) / self.discount_rate
        )


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

    How many copies of each node (or train of nodes) give a repairable
    system the lowest total cost of ownership over a horizon: buying the
    copies, running them (repairs, replacements, maintenance, inspections,
    their own downtime), and the cost of the system being down. Like the
    other result types it is also a read-only mapping of its fields.

    Attributes
    ----------
    units : dict
        How many identical copies of each node considered to fit in active
        parallel, each repaired independently, and of each train to have
        alongside it (under the train's name). Always at least 1: the
        original unit or train.
    total_cost : float
        The total cost of owning the system for ``horizon`` with those
        copies, ``acquisition_cost + cost_rate * horizon`` (with the
        horizon's present value for it when discounted): what
        ``RepairableRBD.total_cost(horizon, discount_rate=...)`` gives for
        the system with the copies drawn out.
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
    trains : dict or None
        The trains considered (``allocate_redundancy``'s ``trains``), each
        name with its nodes in order; None without.
    discount_rate : float
        The continuous discount rate the running costs were discounted at
        (0: undiscounted).

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
    trains: Optional[Dict[Hashable, list]] = None
    discount_rate: float = 0.0


@dataclass
class DemonstrationPlan(_ResultMapping):
    """The result of ``demonstration_plan`` and ``mtbf_demonstration_plan``.

    A demonstration test that keeps both of its risks: a design no better
    than the target passes it with a chance of at most the consumer's
    risk, and a good design fails it with a chance of at most the
    producer's. Like the other result types it is also a read-only mapping
    of its fields.

    Attributes
    ----------
    n : int or None
        The number of units to test (an attribute test); None for a test
        of a constant failure rate.
    test_multiple : float or None
        How many missions long each unit's test is (an attribute test);
        None for a test of a constant failure rate.
    test_time : float or None
        The total unit time to test (a test of a constant failure rate);
        None for an attribute test.
    failures : int
        The failures the test allows: it passes with at most so many.
    consumer_risk : float
        The chance that a design at the target passes.
    producer_risk : float
        The chance that a design at the good level fails.
    """

    n: Optional[int]
    test_multiple: Optional[float]
    test_time: Optional[float]
    failures: int
    consumer_risk: float
    producer_risk: float


@dataclass
class FailureForecast(_ResultMapping):
    """The failures of a part replaced when it fails, over a window from
    its current unit's age: returned by
    [`NonRepairable.failures`][repyability.NonRepairable.failures].

    Attributes
    ----------
    window : float or numpy.ndarray
        The window's length, in the models' time unit.
    age : float
        The current unit's age at the window's start.
    probability : float or numpy.ndarray
        The chance the current unit fails in the window.
    expected_failures : float or numpy.ndarray
        The expected number of failures in the window, each a replacement:
        the current unit's, and its replacements'.
    expected_cost : float, numpy.ndarray or None
        Their expected cost: the unplanned cost times
        ``expected_failures``; None if the part has no unplanned cost.
    """

    window: Any
    age: float
    probability: Any
    expected_failures: Any
    expected_cost: Any


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
    offsets : dict or None
        For ``optimal_inspection_intervals``, node name -> the time of its
        first test (from 0 to less than its interval): those chosen with
        ``offset_shares``, or else each keeping its share of the interval,
        so that plans compare as they are. ``with_intervals`` takes
        them as they are. None for ``optimal_replacement_intervals``.
    """

    intervals: Dict[Hashable, float]
    cost_rate: float
    availability: float
    offsets: Optional[Dict[Hashable, float]] = None


@dataclass(frozen=True)
class Lever(_ResultMapping):
    """One lever of a diagram: a value that its ``parameter_sensitivity``
    moves.

    [`RepairableRBD.levers`][repyability.RepairableRBD.levers] and
    [`NonRepairableRBD.levers`][repyability.NonRepairableRBD.levers] list
    them, in the order ``parameter_sensitivity`` reports them, and
    ``with_levers`` builds the diagram with them at other values. Like the
    other result types it is also a read-only mapping of its fields.

    Attributes
    ----------
    key : Hashable
        Whose lever it is, as ``parameter_sensitivity`` keys its result: a
        node; the tuple of a common-cause group's members, for the
        parameters of the model they share and the group's own; or None,
        for the repair crews.
    name : str
        The lever, as ``parameter_sensitivity`` names it: in a
        ``RepairableRBD`` ``"reliability.alpha"``,
        ``"inspection.interval"``, ``"standby.units"``, ``"ccf_beta"``,
        ``"repair_crews"``, ...; in a ``NonRepairableRBD`` the parameter's
        own name (``"alpha"``) or the group's (``"ccf_beta"``), or a
        regression node's covariate (``"covariate.load"``, its feature
        name, or ``"covariate.0"``, its place).
    value : float
        Its value in the diagram.
    discrete : bool
        Whether it counts something (standby units, repair crews), its
        sensitivity the change one more makes; else its sensitivity is a
        derivative.
    bounds : tuple of float
        ``(lower, upper)``, the range of its values: a value outside it is
        refused (``-inf`` and ``inf`` where there is no bound). An end may
        itself be a valid value or not: a coverage of 1 is, a rate of 0 is
        not.
    calendar : bool
        Whether it moves the period of a calendar that its component
        shares with others (a block replacement's or a test's interval,
        with other components' block replacements or tests): the long-run
        values jump as it leaves the common calendar, so its long-run
        sensitivity takes its schedule apart from the others' (see
        ``RepairableRBD.parameter_sensitivity``). Always False in a
        ``NonRepairableRBD``.
    """

    key: Hashable
    name: str
    value: float
    discrete: bool
    bounds: Tuple[float, float]
    calendar: bool = False


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
    >>> round(capacity.mean, 4)
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

    @property
    def mean(self) -> Any:
        """The expected capacity.

        In the long run, the average capacity over time. Infinite if the
        capacity can be infinite (see ``levels``). A property, as the other
        results' values are (a method until 0.13, #235).

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
    simulations of the window ``[0, time_simulated_to]``.

    One expected value: ``mean_availability`` is the run's estimate
    of the expected availability over the window, the one
    ``mean_availability_interval`` gives an interval for. By default it is
    exact where the exact methods work it out (``method="exact"``, see
    [`ControlVariate`][repyability.ControlVariate]), and otherwise taken
    given the histories of the system's modules where they apply
    (``"conditional"``, see
    [`ConditionalRun`][repyability.ConditionalRun]); else it is the
    simulations' own (``"simulated"``). The simulations' own average,
    ``system_uptime / (n_simulations * time_simulated_to)``, is
    ``sample_mean_availability`` whatever the method, as the curve, the
    totals and the counts are theirs. Against a demand, the
    ``delivered_fraction`` is taken the same way (exact where
    ``mission_capacity`` works it out, given the modules' histories, or
    the simulations' own, ``sample_delivered_fraction``), and
    ``delivered_fraction_interval`` says which. Its repr summarises the
    run.

    The properties ``mean_up_time``, ``mean_down_time`` and
    ``failure_frequency`` are simulation *estimates* derived from the fields
    below; their exact steady-state counterparts are the
    ``RepairableRBD.mean_up_time()``, ``mean_down_time()`` and
    ``system_failure_frequency()`` methods. ``availability_se`` and
    ``availability_interval`` give the sampling uncertainty of the
    availability curve, and ``mean_availability_interval`` that of
    ``mean_availability``.

    Like the other result types it is also a read-only mapping of its
    fields, so dict-style access (``result["availability"]``,
    ``result.keys()``, ``dict(result)``) keeps working.

    Attributes
    ----------
    timeline : numpy.ndarray
        Event times at which the (mean) system availability changes: 0,
        every time at which some simulated system changed state, and
        ``time_simulated_to``, in increasing order. With ``curve_points``
        (see ``RepairableRBD.availability``), the grid's times instead:
        ``time_simulated_to * k / curve_points`` for ``k`` from 0.
    availability : numpy.ndarray
        Mean system availability at each time in ``timeline``: the fraction
        of the simulated systems that are up from that time until the next
        (the estimated point availability). The last value, at
        ``time_simulated_to``, repeats the one before it. On a grid
        (``curve_points``), the fraction up at each grid time (from just
        after any change there), which the full curve takes there too; it
        may change between grid times.
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
    system_failures : int or float
        Number of system failures observed across all simulations (changes
        from up to down caused by a failure, including the zero-length
        outages an instantly repaired component causes); with
        ``conditional=True``, their expected number.
    system_restorations : int or float
        Number of system restorations observed across all simulations
        (changes from down to up, after a failure or a planned outage);
        with ``conditional=True``, their expected number.
    n_simulations : int
        The number of simulations run (``mc_samples``).
    cost : CostResult, optional
        The simulated cost distribution, when the RBD declares any costs;
        ``None`` when nothing is priced (no cost model to run).
    system_planned_outages : int or float
        Number of planned outages of the system observed across all
        simulations: changes from up to down caused by preventive
        maintenance or an inspection that takes time. 0 without either;
        with ``conditional=True``, their expected number.
    uptimes : numpy.ndarray, optional
        The system's up time in each simulation, in order (they sum to
        ``system_uptime``); None in a result built without them.
    antithetic : bool
        Whether the simulations ran in antithetic pairs (simulations ``2i``
        and ``2i + 1``; see ``RepairableRBD.availability``), by default
        False. ``mean_availability_interval`` then works from the pairs'
        means; the pointwise ``availability_se`` and
        ``availability_interval`` treat the simulations as independent.
    capacity_timeline : numpy.ndarray, optional
        With capacities (the ``capacity`` of the ``RepairableRBD``): 0, each
        time at which the capacity of some simulated system changed, and
        ``time_simulated_to``, in increasing order. None without
        capacities, as are the other capacity fields.
    capacity : numpy.ndarray, optional
        The simulated systems' mean capacity at each time in
        ``capacity_timeline``, from that time until the next: the capacity
        curve. ``inf`` while one of them can carry an unlimited amount.
    capacity_time : dict, optional
        Capacity -> the time spent at it, summed over the simulations: they
        add up to ``n_simulations * time_simulated_to``. A node working at
        several levels counts at each in proportion to its probability.
    demand : float, optional
        The demand the delivered fraction is measured against (see
        ``RepairableRBD.availability``); None without one.
    delivered : numpy.ndarray, optional
        Each simulation's delivered fraction, in order: the integral of
        ``min(capacity, demand)`` over the window, over ``demand`` times
        its length. None without a demand.
    delivered_control : ControlVariate, optional
        With a demand, the control of the delivered fractions by the
        system itself where ``mission_capacity`` works out the expected
        fraction delivered (as ``control_variate`` controls the
        availability's): ``delivered_fraction`` is then that exact value.
        None otherwise.
    opportunistic_renewals : dict, optional
        With maintenance groups (see ``RepairableRBD``): each component's
        early renewals at another's stop, summed over the simulations.
        None without groups.
    control_variate : ControlVariate, optional
        The exact twin the run was controlled by (see
        [`ControlVariate`][repyability.ControlVariate]): by default the
        system itself, where it is its own exact twin, or with
        ``control_variate=True`` the system without what ties its
        components together. ``mean_availability_interval`` is then the
        controlled estimate's (for the system itself, its exact value),
        while everything else (``uptimes``, the curve, the totals) stays
        the simulations' own. None otherwise.
    conditional : ConditionalRun, optional
        For a system with dependent modules (see
        [`ConditionalRun`][repyability.ConditionalRun]): by default
        (``whole``), the whole system was simulated, and
        ``mean_availability_interval`` is that of each simulation's
        expected fraction up given its modules' histories, while everything
        else stays the simulations' own. With ``conditional=True``, the run
        simulated only the modules, and its values are each simulation's
        expected values given their histories: ``uptimes``, the totals and
        counts (floats), the curve (on a grid, its chance of being up at
        each time) and the cost's ``samples``; ``criticalities`` is then
        None. None otherwise.
    shutdowns : int
        The number of shuts of the diagram's ``shutdowns`` that start in
        the window, the same in every simulation (they are planned); 0
        without shutdowns. A shut that takes the system down from up is
        one of its ``system_planned_outages``.
    system_planned_downtime : float, optional
        The part of ``system_downtime`` (summed over all simulations, as
        it is) that is *planned*: the system down only because blocks are
        in planned work (preventive maintenance or a test of a working
        unit that takes time, or a shut). See
        [`RepairableRBD.maintenance_effect`][repyability.RepairableRBD.maintenance_effect]
        for the definition. None for a result built without it (a run
        with ``conditional=True``).
    system_unplanned_downtime : float, optional
        The rest of ``system_downtime``, *unplanned*: the system down
        because of failures, so that it would still be down if every
        planned task, test and shut in progress were finished. The two
        add up to ``system_downtime``.

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
    >>> result = rbd.availability(t_simulation=50, mc_samples=200, seed=0)
    >>> float(result.availability[0])  # every simulation starts up
    1.0
    >>> round(result.mean_availability, 4)  # exact; long run: 10 / 11
    0.9107
    >>> round(result.sample_mean_availability, 4)  # the simulations' own
    0.9117
    >>> result.mean_availability_interval().method
    'exact'
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
    criticalities: Optional[Criticalities]
    node_uptime: Dict[Hashable, float]
    node_downtime: Dict[Hashable, float]
    system_downtime: float
    system_failures: float
    system_restorations: float
    n_simulations: int
    cost: Optional[CostResult] = None
    system_planned_outages: float = 0
    uptimes: Optional[np.ndarray] = None
    antithetic: bool = False
    capacity_timeline: Optional[np.ndarray] = None
    capacity: Optional[np.ndarray] = None
    capacity_time: Optional[Dict[float, float]] = None
    demand: Optional[float] = None
    delivered: Optional[np.ndarray] = None
    opportunistic_renewals: Optional[Dict[Hashable, int]] = None
    control_variate: Optional[ControlVariate] = None
    conditional: Optional[ConditionalRun] = None
    delivered_control: Optional[ControlVariate] = None
    shutdowns: int = 0
    system_planned_downtime: Optional[float] = None
    system_unplanned_downtime: Optional[float] = None

    @property
    def mean_capacity(self) -> Optional[float]:
        """Simulation estimate of the average capacity over the window:
        each capacity times the time spent at it, over
        ``n_simulations * time_simulated_to``. Over a long window it
        approaches the exact long-run ``capacity_distribution().mean``.

        Returns
        -------
        float or None
            The average capacity (``inf`` if some time was spent at an
            unlimited one), or None without capacities.
        """
        if self.capacity_time is None:
            return None
        total = sum(
            level * time for level, time in self.capacity_time.items() if time
        )
        return float(total) / (self.n_simulations * self.time_simulated_to)

    @property
    def delivered_fraction(self) -> Optional[float]:
        """The run's estimate of the expected fraction of the demand
        delivered over the window: the production availability, the
        estimate ``delivered_fraction_interval`` gives, exact where
        ``mission_capacity`` works it out, given the modules' histories
        where those apply, and otherwise the simulations' own,
        ``sample_delivered_fraction`` (as ``mean_availability`` is taken,
        see the class's notes). Over a long window it approaches the
        exact long-run ``capacity_distribution().delivered_fraction(demand)``.

        Returns
        -------
        float or None
            The delivered fraction, in ``[0, 1]``, or None without a
            demand.
        """
        if self.delivered is None:
            return None
        return self._delivered_estimate()[0]

    @property
    def sample_delivered_fraction(self) -> Optional[float]:
        """The simulations' own delivered fraction over the window, the
        mean of ``delivered``, whatever ``delivered_fraction`` is.

        Returns
        -------
        float or None
            The fraction of the demand the simulated systems delivered,
            or None without a demand.
        """
        if self.delivered is None:
            return None
        return float(np.mean(self.delivered))

    def _delivered_estimate(self) -> Tuple[float, float, str]:
        """The run's estimate of the expected delivered fraction over the
        window, its standard error and how it was found (see
        ``delivered_fraction_interval``), as ``_estimate`` finds the
        availability's."""
        control, conditional = self.delivered_control, self.conditional
        if control is not None and control.itself:
            return control.exact, 0.0, "exact"
        fractions = np.asarray(self.delivered, dtype=float)
        method = "simulated"
        if conditional is not None and (
            conditional.informative or not conditional.whole
        ):
            if not conditional.whole:
                # A run of the modules alone: its own values are theirs.
                method = "conditional"
            elif conditional.delivered is not None:
                method = "conditional"
                fractions = np.asarray(conditional.delivered, dtype=float)
        error = (
            math.nan
            if conditional is not None and conditional.unknown
            else montecarlo.standard_error(fractions, self.antithetic)
        )
        return float(np.mean(fractions)), error, method

    def delivered_fraction_interval(
        self, confidence: float = 0.95
    ) -> ConfidenceInterval:
        """Confidence interval for the expected delivered fraction over the
        window.

        Around ``delivered_fraction``, as ``mean_availability_interval``
        is around the availability: an exact one has no error, and the
        interval is that value; otherwise it is the mean of the
        simulations' delivered fractions (the conditional ones, given the
        modules' histories, where those apply), with standard error ``std
        / sqrt(n)`` (of antithetic pairs' means, for an antithetic run),
        and the interval, clipped to [0, 1], describes the simulation
        error. Its ``method`` says how the estimate was found:
        ``"exact"`` (``mission_capacity`` works it out, see
        ``delivered_control``), ``"conditional"`` (see
        [`ConditionalRun`][repyability.ConditionalRun]) or
        ``"simulated"``, the simulations' own.

        Parameters
        ----------
        confidence : float, optional
            The confidence level, strictly between 0 and 1, by default 0.95.

        Returns
        -------
        ConfidenceInterval
            The estimate, bounds, standard error, number of simulations and
            method.

        Raises
        ------
        ValueError
            If ``confidence`` is not in (0, 1), or the result has no
            delivered fractions (no capacities, or no demand).
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.delivered is None:
            raise ValueError(
                "This result has no delivered fractions: the RBD had no "
                "capacities, or no demand to measure them against."
            )
        estimate, se, method = self._delivered_estimate()
        z = montecarlo.z_value(confidence)
        return ConfidenceInterval(
            estimate=estimate,
            lower=max(0.0, estimate - z * se),
            upper=min(1.0, estimate + z * se),
            confidence=confidence,
            standard_error=se,
            n_samples=len(self.delivered),
            method=method,
        )

    def _estimate(self) -> Tuple[float, float, str]:
        """The run's estimate of the expected availability over the window,
        its standard error and how it was found (see
        ``mean_availability_interval``)."""
        control, conditional = self.control_variate, self.conditional
        if control is not None and control.itself:
            return control.exact, 0.0, "exact"
        if self.uptimes is None:
            raise ValueError(
                "This result has no per-simulation up times to estimate "
                "the interval from."
            )
        T = self.time_simulated_to
        fractions = np.asarray(self.uptimes, dtype=float) / T
        method = "simulated"
        if control is not None:
            fractions = control.controlled(fractions)
            method = "control_variate"
        elif conditional is not None and (
            conditional.informative or not conditional.whole
        ):
            # A run of the modules alone: its own values are theirs.
            method = "conditional"
            if conditional.whole:
                assert conditional.uptimes is not None
                fractions = np.asarray(conditional.uptimes, dtype=float) / T
        error = (
            math.nan
            if conditional is not None and conditional.unknown
            else montecarlo.standard_error(fractions, self.antithetic)
        )
        return float(np.mean(fractions)), error, method

    @property
    def mean_availability(self) -> float:
        """The run's estimate of the expected availability over the window:
        the estimate ``mean_availability_interval`` gives, exact
        where the exact methods work it out, given the modules' histories
        where those apply, and otherwise the simulations' own,
        ``sample_mean_availability`` (see the class's notes). The
        long-run value is ``RepairableRBD.mean_availability()``.

        Returns
        -------
        float
            The estimated mean availability over the window.
        """
        exact = (
            self.control_variate is not None and self.control_variate.itself
        )
        if self.uptimes is None and not exact:
            # A result built without each simulation's up time.
            return self.sample_mean_availability
        return self._estimate()[0]

    @property
    def sample_mean_availability(self) -> float:
        """The simulations' own mean availability over the window,
        ``system_uptime / (n_simulations * time_simulated_to)``, whatever
        ``mean_availability`` is.

        Returns
        -------
        float
            The fraction of the simulated time the system was up.
        """
        return float(self.system_uptime) / (
            self.n_simulations * self.time_simulated_to
        )

    def mean_availability_interval(
        self, confidence: float = 0.95
    ) -> ConfidenceInterval:
        """Confidence interval for the expected availability over the window.

        Around ``mean_availability``: an exact one has no error, and the
        interval is that value; otherwise it is the mean of the
        simulations' values (their fractions of the window up,
        ``uptimes / time_simulated_to``, controlled or conditional ones),
        which by the central limit theorem is normal with standard error
        ``std / sqrt(n)`` of those values (of antithetic pairs' means, for
        an antithetic run), from which the interval ``estimate +/- z *
        standard_error`` is built, clipped to [0, 1]. It describes the
        simulation error, and narrows like ``1 / sqrt(n)``;
        ``availability(tolerance=...)`` runs until it is narrow enough. Its
        ``method`` says how the estimate was found: ``"exact"``, by the
        exact methods (a ``control_variate`` that is the system itself, see
        [`ControlVariate`][repyability.ControlVariate]);
        ``"control_variate"``, the simulations' fractions controlled by an
        exact twin; ``"conditional"``, each simulation's expected fraction
        up given its modules' histories (see
        [`ConditionalRun`][repyability.ConditionalRun]); ``"simulated"``,
        the simulations' own.

        Parameters
        ----------
        confidence : float, optional
            The confidence level, strictly between 0 and 1, by default 0.95.

        Returns
        -------
        ConfidenceInterval
            The estimate, bounds, standard error, number of simulations and
            method.

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
        if self.control_variate is not None and self.control_variate.itself:
            return self.control_variate._exactly(confidence, len(self.uptimes))
        estimate, se, method = self._estimate()
        z = montecarlo.z_value(confidence)
        unknown = np.isnan(se)
        return ConfidenceInterval(
            estimate=estimate,
            lower=np.nan if unknown else max(0.0, estimate - z * se),
            upper=np.nan if unknown else min(1.0, estimate + z * se),
            confidence=confidence,
            standard_error=se,
            n_samples=len(self.uptimes),
            method=method,
        )

    @property
    def availability_se(self) -> np.ndarray:
        """Pointwise standard error of the availability estimate.

        At each time in ``timeline`` the availability is the proportion of
        the ``n_simulations`` systems that were up, so its sampling standard
        error is the binomial ``sqrt(A (1 - A) / n)``. It is 0 where the
        estimate is 0 or 1; ``availability_interval`` stays informative
        there. It treats the simulations as independent, which antithetic
        ones are not. For a run of the modules alone (``conditional=True``,
        see ``conditional``), each simulation's value is its chance of being
        up then, and the error is that of their mean, from their mean
        square.

        Returns
        -------
        numpy.ndarray
            The standard error at each time in ``timeline``.
        """
        p = np.asarray(self.availability, dtype=float)
        if self.conditional is not None and not self.conditional.whole:
            if self.conditional.unknown:
                # The modules never changed state (#215).
                return np.full_like(p, np.nan)
            square = self.conditional.availability_square
            assert square is not None
            n = self.n_simulations
            if n < 2:
                return np.zeros_like(p)
            variance = np.maximum(square - p * p, 0.0) * n / (n - 1)
            return np.sqrt(variance / n)
        return np.sqrt(p * (1.0 - p) / self.n_simulations)

    def availability_interval(
        self, confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pointwise confidence band for the availability curve.

        Uses the Wilson score interval for a binomial proportion, which
        remains well-behaved when the estimated availability is at or near 0
        or 1 (where the plain normal interval collapses to zero width). For
        a run of the modules alone (``conditional=True``, see
        ``conditional``), whose simulations' values are chances rather than
        0 or 1, the normal interval of their mean (``availability_se``).

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
        z = float(ndtri(0.5 + confidence / 2.0))
        p = np.asarray(self.availability, dtype=float)
        if self.conditional is not None and not self.conditional.whole:
            se = self.availability_se
            return (
                np.clip(p - z * se, 0.0, 1.0),
                np.clip(p + z * se, 0.0, 1.0),
            )
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

    def __repr__(self) -> str:
        # A summary (#223, #235): the curve and totals are in the fields.
        parts = []
        exact = (
            self.control_variate is not None and self.control_variate.itself
        )
        if self.uptimes is not None or exact:
            estimate, error, method = self._estimate()
            parts += [
                f"mean_availability={estimate:.6g}",
                f"standard_error={error:.3g}",
                f"method={method!r}",
            ]
        parts += [
            f"sample_mean_availability={self.sample_mean_availability:.6g}",
            f"system_failures={self.system_failures:g}",
            f"n_simulations={self.n_simulations}",
            f"time_simulated_to={self.time_simulated_to:g}",
        ]
        if self.cost is not None:
            parts.append(f"cost={self.cost!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


@dataclass
class SparesDemand(_ResultMapping):
    """How many spares a component uses over a horizon: the distribution of
    its replacements, from new, for one system or a fleet of them.

    Returned (one per component) by ``RepairableRBD.spares_demand``. A
    component uses a spare at each failure and each preventive replacement
    (a standby group, at each of its units' failures). Like the other result
    types it is also a read-only mapping of its fields.

    Attributes
    ----------
    probabilities : numpy.ndarray
        The probability of each number of replacements, ``0, 1, 2, ...``;
        they sum to 1 (up to a tail below about 1e-12).
    horizon : float
        The time they are counted over, from new.
    fleet : int
        How many systems use them, each from new.
    method : str
        ``"exact"``, worked out to about 1e-6, or ``"simulate"``, the
        fractions of simulations.
    members : tuple or None
        The components whose spares a part pools (see ``spares_demand``'s
        ``parts``), or None for one component's.

    Examples
    --------
    A component with a constant failure rate of 0.01, replaced in no time,
    uses a Poisson number of spares, 10 on average in 1,000 hours:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> rbd = RepairableRBD(
    ...     [("s", "pump"), ("pump", "t")],
    ...     {
    ...         "pump": {
    ...             "reliability": surv.Exponential.from_params([0.01]),
    ...             "repairability": "instant",
    ...         }
    ...     },
    ... )
    >>> demand = rbd.spares_demand(1000.0)["pump"]
    >>> round(demand.mean, 4)
    10.0
    >>> demand.stock(0.95)  # covers the 1,000 hours 95% of the time
    15
    """

    probabilities: np.ndarray
    horizon: float
    fleet: int
    method: str
    members: Optional[tuple] = None

    def __repr__(self) -> str:
        # A summary (#235): the distribution is in ``probabilities``.
        parts = [
            f"mean={float(self.mean):.6g}",
            f"std={float(self.std):.6g}",
            f"probabilities=<{len(self.probabilities)} values>",
            f"horizon={self.horizon:g}",
            f"fleet={self.fleet}",
            f"method={self.method!r}",
        ]
        if self.members is not None:
            parts.append(f"members={self.members!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @property
    def mean(self) -> float:
        """The expected number of spares used (a property, as the other
        results' values are).

        Returns
        -------
        float
            The mean number of replacements.
        """
        return float(self.probabilities @ np.arange(len(self.probabilities)))

    @property
    def std(self) -> float:
        """The standard deviation of the number of spares used (a
        property, as ``mean`` is).

        Returns
        -------
        float
            The standard deviation of the number of replacements.
        """
        counts = np.arange(len(self.probabilities))
        return float(np.sqrt(self.probabilities @ (counts - self.mean) ** 2))

    def covered(self, stock: int) -> float:
        """The probability that ``stock`` spares cover the horizon's
        demand: that it is ``stock`` or fewer.

        Parameters
        ----------
        stock : int
            The spares held, none replenished.

        Returns
        -------
        float
            The probability.
        """
        if stock < 0:
            return 0.0
        return float(min(self.probabilities[: int(stock) + 1].sum(), 1.0))

    def stock(self, probability: float) -> int:
        """The fewest spares that cover the horizon's demand with at least
        ``probability``, none replenished.

        Parameters
        ----------
        probability : float
            The chance of not running out, in ``[0, 1)``.

        Returns
        -------
        int
            The stock.

        Raises
        ------
        ValueError
            If ``probability`` is not in ``[0, 1)``.
        """
        if not 0.0 <= probability < 1.0:
            raise ValueError(
                f"probability must be in [0, 1), got {probability!r}."
            )
        cumulative = np.cumsum(self.probabilities)
        return int(np.searchsorted(cumulative, probability - 1e-12))


@dataclass
class SparesStock(_ResultMapping):
    """The stock of a component's spares that meets a target when each
    spare used is reordered at once and arrives a lead time later
    (one-for-one, or ``(S - 1, S)``, replenishment), in the long run.

    Returned (one per component) by ``RepairableRBD.spares_stock``. With a
    stock of ``S``, a spare is on the shelf when fewer than ``S`` are on
    order: those used in the last lead time. Like the other result types
    it is also a read-only mapping of its fields.

    Attributes
    ----------
    stock : int
        The fewest spares that meet the targets.
    fill_rate : float
        With that stock, the fraction of demands met from the shelf.
    stockout_probability : float
        With that stock, the fraction of time none is on the shelf.
    lead_time : float
        The time a spare ordered takes to arrive.
    fleet : int
        How many systems draw on the stock.
    on_order : numpy.ndarray
        The distribution of how many spares are on order at a random time:
        the demand in a lead time.
    on_order_at_demand : numpy.ndarray
        The same, as a demand finds it (not counting itself).
    members : tuple or None
        The components whose spares a part pools (see ``spares_stock``'s
        ``parts``), or None for one component's.

    Examples
    --------
    A component with a constant failure rate of 0.01, replaced in no time,
    and a lead time of 300 hours: the spares on order are Poisson, 3 on
    average, and 7 on the shelf meet 96.6% of demands:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> rbd = RepairableRBD(
    ...     [("s", "pump"), ("pump", "t")],
    ...     {
    ...         "pump": {
    ...             "reliability": surv.Exponential.from_params([0.01]),
    ...             "repairability": "instant",
    ...         }
    ...     },
    ... )
    >>> stock = rbd.spares_stock(300.0, fill_rate=0.95)["pump"]
    >>> stock.stock
    7
    >>> round(stock.fill_rate, 4)
    0.9665
    """

    stock: int
    fill_rate: float
    stockout_probability: float
    lead_time: float
    fleet: int
    on_order: np.ndarray
    on_order_at_demand: np.ndarray
    members: Optional[tuple] = None

    def fill_rate_for(self, stock: int) -> float:
        """The fraction of demands a stock of ``stock`` meets from the
        shelf: that a demand finds fewer than ``stock`` on order.

        Parameters
        ----------
        stock : int
            The spares held.

        Returns
        -------
        float
            The fill rate.
        """
        if stock <= 0:
            return 0.0
        return float(min(self.on_order_at_demand[: int(stock)].sum(), 1.0))

    def stockout_probability_for(self, stock: int) -> float:
        """The fraction of time a stock of ``stock`` leaves the shelf
        empty: that ``stock`` or more are on order.

        Parameters
        ----------
        stock : int
            The spares held.

        Returns
        -------
        float
            The stock-out probability.
        """
        if stock <= 0:
            return 1.0
        return float(max(1.0 - self.on_order[: int(stock)].sum(), 0.0))


@dataclass
class TimelineSimulation(_ResultMapping):
    """Simulated up/down histories of a repairable system and its
    components, one per simulation, as timelines.

    Returned by
    [`RepairableRBD.simulate_timelines`][repyability.RepairableRBD.simulate_timelines].
    Each [`Timelines`][repyability.Timelines] works out its measures for
    every simulation at once (``uptime``, ``failures``, ``first_failure``,
    ``availability_curve()``, ...), indexes to one simulation's
    [`Timeline`][repyability.Timeline], and merges with others. Like the
    other result types it is also a read-only mapping of its fields.

    Attributes
    ----------
    system : Timelines
        The system's histories, as the event loop has them: each change's
        cause is the component whose change made it.
    components : dict
        Node -> its [`Timelines`][repyability.Timelines]: each component's
        histories (a nested RBD's, its own system's; a standby group's, the
        group's).
    time_simulated_to : float
        The window's end.
    n_simulations : int
        How many histories each holds.
    antithetic : bool
        Whether the simulations came in antithetic pairs.
    engine : str
        The engine that made them: ``"python"`` or ``"numba"``.
    method : str
        How they were made: ``"event loop"``, recorded by the event loop as
        it ran them, the simulations ``availability`` runs with the same
        seed. (Until 0.13 independent components' histories could be
        ``"streams"``, drawn from their streams, the same histories.)
    start : int
        The run's simulation the first history is (see ``join``).

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> unit = {
    ...     "reliability": surv.Weibull.from_params([100, 1.5]),
    ...     "repairability": surv.Exponential.from_params([0.5]),
    ... }
    >>> rbd = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": unit, "b": unit},
    ... )
    >>> runs = rbd.simulate_timelines(
    ...     1000.0, mc_samples=500, seed=1, engine="python"
    ... )
    >>> runs.method, len(runs.system)
    ('event loop', 500)
    >>> pair, pump = runs.system.failures, runs.components["a"].failures
    >>> bool(pair.mean() < pump.mean())
    True

    Two halves of the run, made apart and joined, are the run:

    >>> halves = [
    ...     rbd.simulate_timelines(1000.0, mc_samples=250, seed=1, start=s)
    ...     for s in (0, 250)
    ... ]
    >>> TimelineSimulation.join(halves).system == runs.system
    True
    """

    system: Any
    components: Dict[Hashable, Any]
    time_simulated_to: float
    n_simulations: int
    antithetic: bool = False
    engine: str = "python"
    method: str = "event loop"
    start: int = 0

    @classmethod
    def join(cls, parts) -> "TimelineSimulation":
        """Consecutive ranges of one run's simulations, made apart (with
        ``simulate_timelines``' ``start``, in other processes or on other
        machines, say), as the one result of them all.

        Parameters
        ----------
        parts : iterable of TimelineSimulation
            The ranges, in any order: together, simulations ``start`` to
            ``start + n - 1`` of one run, with no gap.

        Returns
        -------
        TimelineSimulation
            Their histories one after another, from the first range's
            ``start``. Its ``engine`` and ``method`` are the parts', joined
            by commas where they differ.

        Raises
        ------
        ValueError
            If there are no parts, or they are not consecutive ranges of a
            run over one window, with the same components and pairing.
        """
        from repyability.timelines import Timelines, _stacked

        ranges = sorted(parts, key=lambda part: part.start)
        if not ranges:
            raise ValueError("Give at least one TimelineSimulation to join.")
        first = ranges[0]
        nodes = list(first.components)
        for before, after in zip(ranges, ranges[1:]):
            if after.start != before.start + before.n_simulations:
                raise ValueError(
                    f"The parts are not consecutive: simulations "
                    f"{before.start} to "
                    f"{before.start + before.n_simulations - 1} are followed "
                    f"by {after.start}."
                )
        for part in ranges:
            if (
                part.time_simulated_to != first.time_simulated_to
                or part.antithetic != first.antithetic
                or list(part.components) != nodes
            ):
                raise ValueError(
                    "The parts are not of one run: they differ in their "
                    "window, components or antithetic pairing."
                )

        def joined(timelines: list) -> Any:
            data = [timeline._data for timeline in timelines]
            return Timelines._from_data(
                _stacked(data, data[0].leaves, first.time_simulated_to),
                timelines[0].name,
            )

        def names(field: str) -> str:
            return ", ".join(
                dict.fromkeys(getattr(part, field) for part in ranges)
            )

        return cls(
            system=joined([part.system for part in ranges]),
            components={
                node: joined([part.components[node] for part in ranges])
                for node in nodes
            },
            time_simulated_to=first.time_simulated_to,
            n_simulations=sum(part.n_simulations for part in ranges),
            antithetic=first.antithetic,
            engine=names("engine"),
            method=names("method"),
            start=first.start,
        )


@dataclass
class StressStrengthResult(_ResultMapping):
    """The result of [`stress_strength`][repyability.stress_strength]: the
    chance that a strength exceeds the stress (or the largest of ``n_loads``
    stresses), ``R = P(strength > stress)``. Like the other result types it
    is also a read-only mapping of its fields.

    Attributes
    ----------
    reliability : float
        ``R``, from the models as given.
    failure_probability : float
        ``1 - R``, the chance the stress reaches the strength. Worked out
        from the strength's ``ff`` (or a closed form) rather than as
        ``1 - R``, so that it is not lost to rounding when ``R`` is near
        1; quadrature is accurate to about 1e-13 absolute.
    reliability_index : float
        The normal quantile of ``R`` (the "safety index"): the number of
        standard deviations of a normal margin that would give ``R``. It
        is infinite when ``R`` is 1 and ``-inf`` when it is 0.
    n_loads : int
        The number of independent stress applications the strength must
        survive (1 for a single stress).
    method : str
        How ``R`` was found: ``"exact"`` (a closed form), ``"numerical"``
        (quadrature) or ``"simulated"``.
    standard_error : float or None
        The Monte-Carlo standard error of ``R``; None unless simulated.
    n_samples : int or None
        The number of simulated strength and stress draws; None unless
        simulated.
    lower : float or None
        The lower bound of ``R`` from the models' parameter uncertainty;
        None unless ``uncertainty`` was asked for.
    upper : float or None
        The upper bound of ``R`` from the models' parameter uncertainty;
        None unless ``uncertainty`` was asked for.
    confidence : float or None
        The confidence level of ``lower`` and ``upper``.
    n_draws : int or None
        The number of draws of the models' parameters behind the bounds.
    """

    reliability: float
    failure_probability: float
    reliability_index: float
    n_loads: int
    method: str
    standard_error: Optional[float] = None
    n_samples: Optional[int] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    confidence: Optional[float] = None
    n_draws: Optional[int] = None


@dataclass
class StressStrengthCurve(_ResultMapping):
    """The result of
    [`stress_strength_over_time`][repyability.stress_strength_over_time]:
    the interference reliability of a strength that changes with age.

    Attributes
    ----------
    times : numpy.ndarray
        The ages, in the unit the models share.
    reliability : numpy.ndarray
        ``R(t) = P(strength at age t > stress)`` at each age.
    failure_probability : numpy.ndarray
        ``1 - R(t)``, worked out directly.
    n_loads : int
        The number of independent stress applications at each age.
    method : str
        ``"exact"`` if every age used a closed form, else ``"numerical"``.
    """

    times: Any
    reliability: Any
    failure_probability: Any
    n_loads: int
    method: str


@dataclass
class PlanVsActual(_ResultMapping):
    """A plan against what happened, and what it is worth now: returned by
    ``RepairableRBD.plan_vs_actual``.

    In plain terms: the plan's expected cash flows over the time so far
    (from ``expected_npv``) stand beside the cash flows the outage log
    implies, priced with the diagram's own costs; the variance is the
    actual less the expected. Amounts are signed as in ``ExpectedNPV``
    (costs negative, value positive), so a **positive variance is in your
    favour**: less spent, or more earned, than the plan expected. The plan
    is then valued again from the components' states now for the rest of
    the horizon, and "what happened so far plus what is expected from
    here" stands beside the original plan's value.

    Costs the log does not determine (inspections that found nothing, a
    group's set-up costs, purchases, ...) are compared on neither side:
    they are listed in ``not_compared`` with the reason, and carried at
    the plan's expectation (``as_planned_so_far``) in the re-valued plan.

    Attributes
    ----------
    now : float
        The end of the log, in the models' time from the start of the
        plan.
    horizon : float
        The plan's horizon.
    discount_rate : float
        The continuous discount rate per unit time (0: undiscounted).
    value_rate : float
        The value earned per unit time while the system works.
    plan_npv : float
        The original plan's expected net present value over the horizon.
    revalued_npv : float
        ``actual_so_far + as_planned_so_far + rest_npv``: the plan's value
        knowing what happened so far, as of the start.
    change : float
        ``revalued_npv - plan_npv``: ``variance_so_far`` plus the change in
        the expected value of the rest (``rest_npv - plan_rest_npv``).
    expected_so_far : float
        The present value of the plan's expected cash flows to ``now``, in
        the categories compared.
    actual_so_far : float
        The present value of the cash flows the log implies, in the same
        categories.
    variance_so_far : float
        ``actual_so_far - expected_so_far``.
    undiscounted_expected_so_far : float
        ``expected_so_far`` with no discounting.
    undiscounted_actual_so_far : float
        ``actual_so_far`` with no discounting.
    as_planned_so_far : float
        The present value of the plan's expected cash flows to ``now`` in
        the cells not compared, carried as planned.
    plan_rest_npv : float
        The original plan's expected value of the time from ``now`` to the
        horizon, as of the start.
    rest_npv : float
        The expected value of the time from ``now`` to the horizon from the
        states now, as of the start (``rest.npv`` is it as of ``now``).
    variance_by_category : dict
        The variance's present value by category (``"value"`` included).
    variance_by_component : dict
        The variance's present value by component.
    period_ends : list
        The ends of the periods, the horizon last; one is at ``now``.
    expected_cash_flows : list
        The plan's cash-flow rows to ``now`` that are compared, as in
        ``ExpectedNPV.cash_flows``.
    actual_cash_flows : list
        The rows the log implies, in the same form.
    variance_cash_flows : list
        Actual less expected, row for row (by period, category and
        component).
    rest_cash_flows : list
        The rest's expected rows from the states now, on the plan's clock
        and periods, present values as of the start; no acquisition.
    not_compared : list
        A dict for each priced cell the log does not determine:
        ``category``, ``component`` (None for the system's) and
        ``reason``.
    states : dict
        The components' states now that the rest was valued from.
    derived_states : list
        The components whose states came from their logs (the others were
        given).
    plan : ExpectedNPV
        The original plan over the horizon, from its start.
    rest : ExpectedNPV or None
        The rest valued as of ``now`` (its own clock from 0), with no
        acquisition cost; None if ``now`` is the horizon.
    """

    now: float
    horizon: float
    discount_rate: float
    value_rate: float
    plan_npv: float
    revalued_npv: float
    change: float
    expected_so_far: float
    actual_so_far: float
    variance_so_far: float
    undiscounted_expected_so_far: float
    undiscounted_actual_so_far: float
    as_planned_so_far: float
    plan_rest_npv: float
    rest_npv: float
    variance_by_category: Dict[str, float]
    variance_by_component: Dict[Hashable, float]
    period_ends: List[float]
    expected_cash_flows: List[Dict[str, Any]]
    actual_cash_flows: List[Dict[str, Any]]
    variance_cash_flows: List[Dict[str, Any]]
    rest_cash_flows: List[Dict[str, Any]]
    not_compared: List[Dict[str, Any]]
    states: Dict[Hashable, Any]
    derived_states: List[Hashable]
    plan: ExpectedNPV
    rest: Optional[ExpectedNPV]

    def __repr__(self) -> str:
        return (
            f"PlanVsActual(now={self.now:g}, horizon={self.horizon:g}, "
            f"plan_npv={self.plan_npv:.6g}, "
            f"revalued_npv={self.revalued_npv:.6g}, "
            f"variance_so_far={self.variance_so_far:.6g}, "
            f"not_compared={len(self.not_compared)})"
        )


@dataclass(frozen=True)
class NodeActionRates(_ResultMapping):
    """A component's actions per unit time in the long run: returned, one
    for each component, by
    [`RepairableRBD.node_action_rates`][repyability.RepairableRBD.node_action_rates].

    A long-run rate is the expected number of actions in a long window
    over its length. Multiplied by a price per action, it is that
    action's cost per unit time, as
    [`expected_cost_rate`][repyability.RepairableRBD.expected_cost_rate]
    counts them.

    Attributes
    ----------
    corrective : float
        Failures per unit time, each one a repair (or replacement): of a
        standby group, its units' failures.
    preventive : float
        Preventive replacements per unit time, under age or block
        replacement or on condition; 0 for a component that has none.

    Examples
    --------
    >>> from repyability import NodeActionRates
    >>> rates = NodeActionRates(corrective=0.002, preventive=0.01)
    >>> rates.to_dict()
    {'corrective': 0.002, 'preventive': 0.01}
    """

    corrective: float
    preventive: float


@dataclass(frozen=True)
class CrewJob(_ResultMapping):
    """A job the repair crews work on: returned by
    [`RepairableRBD.crew_jobs`][repyability.RepairableRBD.crew_jobs].

    A plain component is one job. Each unit of a standby group is a job of
    its own, so a group of three units can keep three crews busy.

    Attributes
    ----------
    node : Hashable
        The component (or standby group) the job is for.
    unit : int or None
        ``None`` for a plain component; for a standby group, the unit's
        number in the group, from 0.

    Examples
    --------
    >>> from repyability import CrewJob
    >>> CrewJob("pumps", 1).to_dict()
    {'node': 'pumps', 'unit': 1}
    """

    node: Hashable
    unit: Optional[int]


@dataclass(frozen=True)
class CrewStates(_ResultMapping):
    """The repair crews' Markov chain in the long run: returned by
    [`RepairableRBD.crew_states`][repyability.RepairableRBD.crew_states].

    With fewer repair crews than jobs, a failed component may wait for a
    crew, so the components no longer fail and recover independently. The
    long-run values then come from a Markov chain whose states say which
    components are up and which jobs are waiting. In each state each
    component the crews serve is either up (1) or down (0). A long-run
    value is the average of its value in each state, weighed by the
    states' ``probabilities``: a node's long-run availability is
    ``probabilities @ node_availability[node]``.

    Attributes
    ----------
    node_availability : dict
        ``{node: array}``: each node's availability in each state. 1 or 0
        for a component the crews serve, and for a node held working or
        broken; a nested RBD's own long-run availability in every state
        (it has crews of its own); 1 for the input and output nodes.
    probabilities : numpy.ndarray
        Each state's long-run probability, adding up to 1.

    Examples
    --------
    >>> import numpy as np
    >>> from repyability import CrewStates
    >>> states = CrewStates(
    ...     {"a": np.array([1.0, 0.0])}, np.array([0.9, 0.1])
    ... )
    >>> float(states.probabilities @ states.node_availability["a"])
    0.9
    """

    node_availability: Dict[Hashable, np.ndarray]
    probabilities: np.ndarray

    def __repr__(self) -> str:
        # A summary (#235): the states' values are in the fields.
        return (
            f"CrewStates(n_states={len(self.probabilities)}, "
            f"nodes={len(self.node_availability)})"
        )


@dataclass(frozen=True)
class ReplacementCycle(_ResultMapping):
    """The renewal cycle of a component under block replacement or
    replaced on condition, in the long run: returned by
    [`RepairableRBD.replacement_cycle`][repyability.RepairableRBD.replacement_cycle].

    Under block replacement every ``T`` a unit is replaced at each multiple
    of ``T`` at which it is up, and repaired at each failure in between;
    replaced on condition, it is inspected every ``T`` and replaced when
    an inspection finds it worn. Its long-run values repeat over each
    interval, which ``phase`` cuts into cells. Each mean over the cycle's
    ``length`` is a long-run rate: ``up / length`` is the component's
    long-run availability, and ``failures / length`` its failures per
    unit time.

    Attributes
    ----------
    up : float
        The mean time up in a cycle.
    length : float
        The mean length of a cycle.
    failures : float
        The mean number of failures in a cycle.
    phase : numpy.ndarray
        A grid over one interval, from 0 to ``T``: the cells' edges.
    availability : numpy.ndarray
        The long-run probability that the component is up in each cell
        (at the cell's middle), one fewer than ``phase``.
    failure_rate : numpy.ndarray
        The long-run failure intensity in each cell.
    before : float
        The probability that the component is up just before a block time
        (or an inspection).
    after : float
        The probability that it is up just after one.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> spec = {
    ...     "reliability": surv.Weibull.from_params([1000.0, 3.0]),
    ...     "repairability": surv.Exponential.from_params([0.5]),
    ...     "preventive": {"interval": 400.0, "policy": "block"},
    ... }
    >>> rbd = RepairableRBD([("s", "c"), ("c", "t")], {"c": spec})
    >>> cycle = rbd.replacement_cycle("c")
    >>> round(cycle.length, 1), round(cycle.up / cycle.length, 4)
    (400.4, 0.9997)

    The cycle is a little longer than the interval: a replacement due
    while the pump is down is skipped.
    """

    up: float
    length: float
    failures: float
    phase: np.ndarray
    availability: np.ndarray
    failure_rate: np.ndarray
    before: float
    after: float

    def __repr__(self) -> str:
        # A summary (#235): the profiles over the interval are in the
        # fields.
        return (
            f"ReplacementCycle(up={self.up:.6g}, length={self.length:.6g}, "
            f"failures={self.failures:.6g}, before={self.before:.6g}, "
            f"after={self.after:.6g}, cells={len(self.availability)})"
        )


@dataclass(frozen=True)
class SimulationSamples(_ResultMapping):
    """Each simulation's values from a run: returned by
    [`RepairableRBD.simulation_samples`][repyability.RepairableRBD.simulation_samples].

    The run's draws are common random numbers: each comes from a stream
    named by the component it is for (its node name, down through nested
    RBDs) and seeded from the run's entropy. So two diagrams run with the
    same entropy draw the same numbers for a component in the same place
    in both, and the differences between their samples come from how they
    differ rather than from chance.

    Attributes
    ----------
    availability : numpy.ndarray
        Each simulation's fraction of the window the system was up.
    cost : numpy.ndarray or None
        Each simulation's running cost over the window (as
        [`cost`][repyability.RepairableRBD.cost] counts it, without the
        acquisition cost); ``None`` for a diagram with no running costs.

    Examples
    --------
    >>> import numpy as np
    >>> from repyability import SimulationSamples
    >>> samples = SimulationSamples(np.array([1.0, 0.98]), None)
    >>> samples
    SimulationSamples(n_simulations=2, mean_availability=0.99, cost=None)
    """

    availability: np.ndarray
    cost: Optional[np.ndarray]

    def __repr__(self) -> str:
        # A summary (#235): each simulation's values are in the fields.
        cost = (
            "None"
            if self.cost is None
            else f"{float(np.mean(self.cost)):.6g} (mean)"
        )
        return (
            f"SimulationSamples(n_simulations={len(self.availability)}, "
            "mean_availability="
            f"{float(np.mean(self.availability)):.6g}, cost={cost})"
        )


@dataclass(frozen=True)
class CoreStructure(_ResultMapping):
    """The part of a diagram that does not reduce to modules (its core):
    returned by
    [`RBD.core_structure`][repyability.RBD.core_structure].

    A diagram is reduced to modules (series chains, parallel groups and
    k-out-of-n groups, each independent of the rest) by replacing each
    with a single vertex. What is left when nothing more reduces is the
    core: a graph from the input to the output whose vertices are modules
    and single nodes. Its minimal path sets are what the exact methods
    must search. A component drawn in more than one place (a repeated
    node) stays out of every module, as the modules' members must be
    independent of the rest, so it is left in the core.

    Attributes
    ----------
    graph : networkx.DiGraph
        The core. Its vertices are numbers; each has ``k`` (how many of
        its predecessors must work for it to be reached) and
        ``components``, the frozenset of the diagram's nodes it stands
        for (empty for the input and output). The input vertex is the
        one with no predecessors.
    output : int
        The output vertex.

    Examples
    --------
    >>> from repyability import RBD
    >>> bridge = RBD(
    ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("a", "d"),
    ...      ("c", "d"), ("b", "e"), ("c", "e"), ("d", "t"), ("e", "t")]
    ... )
    >>> core = bridge.core_structure()
    >>> core.graph.in_degree(core.output)
    2
    """

    graph: Any
    output: int

    def to_dict(self) -> dict:
        """The core as plain data, ready for ``json.dumps``: its vertices,
        each with its ``k`` and the components it stands for, its edges,
        and the output vertex.

        Returns
        -------
        dict
            ``{"vertices": [...], "edges": [[u, v], ...], "output": v}``.
        """
        return {
            "vertices": [
                {
                    "vertex": int(v),
                    "k": int(data["k"]),
                    "components": [
                        _json_key(n)
                        for n in sorted(data["components"], key=str)
                    ],
                }
                for v, data in self.graph.nodes(data=True)
            ],
            "edges": [[int(u), int(v)] for u, v in self.graph.edges],
            "output": int(self.output),
        }

    def __repr__(self) -> str:
        return (
            f"CoreStructure(vertices={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, output={self.output})"
        )


@dataclass(frozen=True)
class ConditionalProbabilities(_ResultMapping):
    """A fault tree's top event probability, alone and given each basic
    event: returned by
    [`FaultTree.conditional_probabilities`][repyability.FaultTree.conditional_probabilities].

    Every importance measure of a basic event is made of these three
    numbers and the event's own probability: Birnbaum's is
    ``given_occurred[e] - given_not_occurred[e]``, the risk achievement
    worth ``given_occurred[e] / top`` and the risk reduction worth
    ``top / given_not_occurred[e]``.

    Attributes
    ----------
    top : float or numpy.ndarray
        The top event's probability: a float for a number ``t``, an array
        for an array.
    given_occurred : dict
        ``{event: probability}``: the top event's probability given that
        the event has occurred.
    given_not_occurred : dict
        ``{event: probability}``: the top event's probability given that
        it has not.

    Examples
    --------
    >>> from repyability import FaultTree
    >>> tree = FaultTree(
    ...     {"top": ("or", ["valve", "flow"]),
    ...      "flow": ("and", ["pump 1", "pump 2"])},
    ...     {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
    ... )
    >>> given = tree.conditional_probabilities()
    >>> round(given.top, 4), given.given_occurred["valve"]
    (0.0595, 1.0)
    """

    top: Any
    given_occurred: Dict[Hashable, Any]
    given_not_occurred: Dict[Hashable, Any]


@dataclass(frozen=True)
class ModelParameters(_ResultMapping):
    """A parametric model's parameters, to rebuild it with them moved:
    returned by [`model_parameters`][repyability.model_parameters].

    Attributes
    ----------
    cls : object
        The surpyval distribution the model is of (``surpyval.Weibull``,
        say), whose ``from_params`` rebuilds it.
    params : list of float
        The parameters' values: the distribution's, then the share that
        ever fails (``lfp_p``, a limited failure population) and the share
        dead on arrival (``f0``, zero inflation) where it has them, the
        order of surpyval's ``covariance()``.
    names : list of str
        The parameters' names, in the order of ``params``.
    extras : dict
        surpyval's ``model.extras`` (the offset ``gamma`` among them),
        which the rebuilt model keeps.
    bounds : list of tuple
        Each parameter's range as its distribution bounds it, ``(low,
        high)``, ``None`` where it does not.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import model_parameters
    >>> found = model_parameters(surv.Weibull.from_params([100.0, 2.0]))
    >>> found.names, found.params
    (['alpha', 'beta'], [100.0, 2.0])
    >>> round(float(found.build([200.0, 2.0]).sf(200.0)), 4)
    0.3679
    """

    cls: Any
    params: List[float]
    names: List[str]
    extras: Dict[str, Any]
    bounds: List[Tuple[Any, Any]]

    def build(self, values) -> Any:
        """The model with its parameters at ``values``.

        Parameters
        ----------
        values : sequence of float
            A value for each parameter, in the order of ``params``.

        Returns
        -------
        object
            The surpyval model, with ``extras`` as the original's and the
            shares (``lfp_p``, ``f0``) at their given values.
        """
        from repyability.rbd._model_utils import ParametricSpec

        return ParametricSpec(
            self.cls, self.params, self.names, self.extras, self.bounds
        ).build(values)

    def to_dict(self) -> dict:
        """The parameters as plain data, ready for ``json.dumps``: the
        distribution by its name.

        Returns
        -------
        dict
            The fields, ``cls`` as the distribution's name.
        """
        out = super().to_dict()
        out["cls"] = getattr(self.cls, "name", str(self.cls))
        return out
