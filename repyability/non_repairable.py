import warnings

import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.optimize import minimize, minimize_scalar
from surpyval import ExactEventTime, NonParametric, Parametric

from repyability.maintenance import MaintenancePolicy
from repyability.rbd._model_utils import (
    distribution_name,
    is_exponential,
    model_mean,
)
from repyability.rbd.standby_node import StandbyModel

FAILURE = 1
REPLACE = 0


def _scalar_sf(model):
    """``model.sf`` as a function of one age returning a float (a
    ``StandbyModel``'s ``sf`` returns an array even for a scalar)."""

    def sf(t):
        return float(np.ravel(model.sf(np.atleast_1d(float(t))))[0])

    return sf


def _piecewise_linear_sf(model):
    """A non-parametric estimate's survival function, and its knots.

    Linear between the estimate's time points, starting from 1 at age 0,
    and held at its last value beyond the last one (as surpyval's own
    step function is), so it stays a probability.
    """
    ages = np.asarray(model.x, dtype=float)
    survival = np.clip(1.0 - np.asarray(model.F, dtype=float), 0.0, 1.0)
    if ages[0] > 0.0:
        ages = np.concatenate([[0.0], ages])
        survival = np.concatenate([[1.0], survival])

    def sf(t):
        return np.interp(t, ages, survival)

    return sf, ages


def _cost_rates(cp, cu, ages, R):
    """The age-replacement cost rate at each of ``ages`` (increasing, from
    0), the cycle length integrated by the cumulative trapezoidal rule. The
    rate at age 0 is ``inf``."""
    cycle = np.concatenate(
        [[0.0], np.cumsum(np.diff(ages) * (R[1:] + R[:-1]) / 2.0)]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = (cp * R + cu * (1.0 - R)) / cycle
    rates[0] = np.inf
    return rates


class NonRepairable:
    """A component renewed by replacement ("as good as new").

    Pairs a lifetime (reliability) model with a time-to-replace model for
    a unit that cannot be repaired in place: every failure or planned
    replacement fits a new unit, so each cycle is a statistical renewal.

    It plays two roles:

    - Standalone, it prices the classic *age-replacement* policy: replace
      preventively at age ``t`` (planned, cost ``cp``) or on failure
      (unplanned, cost ``cu > cp``), whichever comes first, via
      ``cost_rate``, ``find_optimal_replacement()`` and
      ``optimal_replacement_policy()``. The cost calculations treat
      replacement as instantaneous (they do not use ``time_to_replace``).
    - Inside a [`RepairableRBD`][repyability.RepairableRBD], it is the
      per-component model (the RBD builds one from each
      ``{"reliability": ..., "repairability": ...}`` component spec): the
      unit alternates between a time to failure drawn from
      ``reliability`` and a replacement time drawn from
      ``time_to_replace`` (see ``next_event()``), and
      ``mean_availability()`` and ``failure_frequency()`` give its exact
      long-run behaviour. The same object can be given for several nodes:
      each node gets its own copy.

    Contrast with [`Repairable`][repyability.Repairable], which models
    minimal ("as bad as old") or imperfect repair and the overhaul
    policy.

    Parameters
    ----------
    reliability : surpyval model or StandbyModel
        The unit's lifetime distribution, one of:

        - a surpyval parametric distribution, fitted or built with
          ``from_params``;
        - a surpyval non-parametric estimate (e.g. a ``KaplanMeier``
          fit). For the cost calculations its survival function is taken
          as linear between its time points, starting from 1 at age 0,
          and held at its last value beyond the last one; the
          availability methods reject it, and its draws in
          ``next_event()`` cannot be seeded;
        - a [`StandbyModel`][repyability.StandbyModel], in any of its
          forms (a closed form, a convolution or a simulation), through its
          survival function ``sf``.
    time_to_replace : surpyval model, optional
        The distribution of the time taken to replace the unit after a
        failure, used by the availability and event methods. Default
        ``ExactEventTime.from_params(0)``: instantaneous replacement.

    Attributes
    ----------
    reliability : object
        The lifetime model given.
    time_to_replace : object
        The time-to-replace model given.
    cost_rate : callable
        ``cost_rate(t)``, the long-run expected cost per unit time of
        replacing at age ``t`` or on failure:
        ``(cp * R(t) + cu * (1 - R(t))) / avg_replacement_time(t)``, with
        ``R`` the survival function of ``reliability``. Vectorised over
        ``t``; returns a numpy array (0-d for a scalar ``t``). Needs the
        costs set by ``set_costs_planned_and_unplanned()`` (without them
        it raises ValueError).

    Raises
    ------
    ValueError
        If ``reliability`` is not a surpyval parametric or non-parametric
        model or a ``StandbyModel``.

    Examples
    --------
    Age replacement of a wearing-out unit:

    >>> import surpyval as surv
    >>> from repyability import NonRepairable
    >>> unit = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
    >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
    >>> round(float(unit.find_optimal_replacement()), 1)
    493.0
    >>> [round(float(g), 5) for g in unit.cost_rate([250, 493, 1000])]
    [0.00453, 0.00346, 0.00452]

    As an RBD component with a mean time to failure of 100 and a mean
    time to replace of 2:

    >>> from repyability import RepairableRBD
    >>> pump = NonRepairable(
    ...     surv.Exponential.from_params([0.01]),
    ...     surv.Exponential.from_params([0.5]),
    ... )
    >>> round(pump.mean_availability(), 4)
    0.9804
    >>> rbd = RepairableRBD([("s", "pump"), ("pump", "t")], {"pump": pump})
    >>> round(rbd.mean_availability(), 4)
    0.9804
    """

    def __init__(
        self, reliability, time_to_replace=ExactEventTime.from_params(0)
    ):
        if isinstance(reliability, Parametric):
            self.model_parameterization = "parametric"
            self.reliability_function = reliability.sf
        elif isinstance(reliability, NonParametric):
            self.model_parameterization = "non-parametric"
            self.reliability_function, self._knots = _piecewise_linear_sf(
                reliability
            )
        elif isinstance(reliability, StandbyModel):
            # Whatever the arrangement's survival function is (a closed
            # form, a convolution or a Kaplan-Meier fit to simulated
            # lifetimes), its sf gives it.
            self.model_parameterization = "standby"
            self.reliability_function = _scalar_sf(reliability)
        else:
            raise ValueError("Unknown reliability function")

        self.reliability = reliability
        self.time_to_replace = time_to_replace
        self.cost_rate = np.vectorize(
            self._cost_rate,
            doc=(
                "Long-run cost per unit time of an age-replacement policy "
                "with replacement age t:\n"
                "(cp * R(t) + cu * F(t)) / integral_0^t R(u) du.\n"
                "Vectorised over t."
            ),
        )
        self.__next_event_type = FAILURE

    def set_costs_planned_and_unplanned(self, cp, cu):
        """Set the planned and unplanned replacement costs.

        Required before any cost calculation (``cost_rate``,
        ``find_optimal_replacement()``, ``optimal_replacement_policy()``).
        The costs are stored as the ``cp`` and ``cu`` attributes.

        Parameters
        ----------
        cp : float
            Cost of a planned (preventive) replacement, made when the unit
            reaches the replacement age without failing.
        cu : float
            Cost of an unplanned (corrective) replacement, made on failure.
            Must exceed ``cp``, otherwise preventive replacement never
            pays.

        Raises
        ------
        ValueError
            If a cost is negative or not finite, or if ``cp >= cu``.
        """
        if not (np.isfinite(cp) and np.isfinite(cu)):
            raise ValueError(f"costs must be finite, got cp={cp}, cu={cu}.")
        if cp < 0:
            raise ValueError(f"costs must be non-negative, got cp={cp}.")
        if cp >= cu:
            raise ValueError("Planned costs must be less than unplanned costs")
        self.cp = cp
        self.cu = cu

    def _require_costs(self) -> None:
        if not hasattr(self, "cp"):
            raise ValueError(
                "costs not set: call set_costs_planned_and_unplanned"
                "(cp, cu) first"
            )

    def avg_replacement_time(self, t):
        """Expected time between replacements when replacing at age ``t``.

        The expected cycle length of the age-replacement policy,
        ``E[min(T, t)] = integral_0^t R(u) du``: a cycle ends at failure
        or at age ``t``, whichever comes first. It is the denominator of
        ``cost_rate``.

        For a parametric model the integral is computed by quadrature, and
        for a ``StandbyModel`` by the trapezoidal rule on 4,001 ages (its
        survival function may be a step function). For a non-parametric
        model it is exact for the survival function described in the
        class docstring (linear between the estimate's time points, from 1
        at age 0, and held beyond the last one).

        Parameters
        ----------
        t : float
            The replacement age (a scalar).

        Returns
        -------
        float
            The expected cycle length, in the lifetime model's time units.

        Examples
        --------
        For an exponential lifetime with rate 0.01 the integral is
        ``(1 - exp(-0.01 * t)) / 0.01``:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> unit = NonRepairable(surv.Exponential.from_params([0.01]))
        >>> round(unit.avg_replacement_time(100), 2)
        63.21
        """
        if self.model_parameterization == "parametric":
            out = quad(self.reliability_function, 0, t)[0]
        elif self.model_parameterization == "standby":
            # A simulated arrangement's survival function is a Kaplan-Meier
            # step function, which quadrature handles poorly: integrate on a
            # fine grid instead.
            ages = np.linspace(0.0, float(t), 4001)
            R = np.ravel(np.asarray(self.reliability.sf(ages), dtype=float))
            out = float(trapezoid(np.clip(R, 0.0, 1.0), ages))
        else:
            # The survival function is linear between its knots, so the
            # trapezoidal rule over the knots below t, and t, is exact.
            t = float(t)
            if t <= 0.0:
                return 0.0
            inner = self._knots[(self._knots > 0.0) & (self._knots < t)]
            ages = np.concatenate([[0.0], inner, [t]])
            out = float(trapezoid(self.reliability_function(ages), ages))

        return out

    def _cost_rate(self, t):
        # surpyval >= 0.11 returns 1-element arrays from sf/mean/qf, and t can
        # arrive as a 1-element array from scipy.optimize.minimize. Coerce to a
        # plain float so the quadrature bound in avg_replacement_time() is a
        # scalar (scipy.integrate.quad rejects array bounds).
        self._require_costs()
        t = float(np.asarray(t).item())
        planned_costs = self.reliability_function(t) * self.cp
        unplanned_costs = (1 - self.reliability_function(t)) * self.cu
        avg_repl_time = self.avg_replacement_time(t)
        return (planned_costs + unplanned_costs) / avg_repl_time

    def _log_cost_rate(self, t):
        return np.log(self._cost_rate(t))

    def mean_unavailability(self) -> float:
        """Long-run unavailability, ``1 - mean_availability()``.

        Returns
        -------
        float
            The long-run fraction of time the unit is down (being
            replaced).

        Raises
        ------
        ValueError
            If ``reliability`` is a non-parametric model.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> unit = NonRepairable(
        ...     surv.Exponential.from_params([0.01]),
        ...     surv.Exponential.from_params([0.5]),
        ... )
        >>> round(unit.mean_unavailability(), 4)
        0.0196
        """
        return 1 - self.mean_availability()

    def mean_availability(self) -> float:
        """Long-run availability, ``MTTF / (MTTF + MTTR)``.

        The steady-state fraction of time the unit is up when it is run to
        failure and then replaced, so up and down times alternate (an
        alternating renewal process). ``MTTF`` is the mean of
        ``reliability`` and ``MTTR`` the mean of ``time_to_replace``.
        For a simulated ``StandbyModel`` lifetime the MTTF is a Monte Carlo
        estimate drawn from numpy's global RNG, so it varies slightly
        between calls.

        Returns
        -------
        float
            The long-run availability, between 0 and 1.

        Raises
        ------
        ValueError
            If ``reliability`` is a non-parametric model.

        Examples
        --------
        A unit with MTTF 100 and a mean time to replace of 2:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> unit = NonRepairable(
        ...     surv.Exponential.from_params([0.01]),
        ...     surv.Exponential.from_params([0.5]),
        ... )
        >>> round(unit.mean_availability(), 4)
        0.9804
        """
        if isinstance(self.reliability, NonParametric):
            raise ValueError(
                "Mean Availability requires a parametric reliability model"
            )
        mttf = model_mean(self.reliability)
        mttr = model_mean(self.time_to_replace)
        return mttf / (mttr + mttf)

    def failure_frequency(self) -> float:
        """Long-run failure frequency (failures per unit time).

        For an alternating renewal process (fail, replace, fail, ...) this
        is ``1 / (MTTF + MTTR)``: one failure per mean up-down cycle, with
        ``MTTF`` the mean of ``reliability`` and ``MTTR`` the mean of
        ``time_to_replace``.

        Returns
        -------
        float
            The expected number of failures per unit time in the long run.

        Raises
        ------
        ValueError
            If ``reliability`` is a non-parametric model.

        Examples
        --------
        One failure per ``100 + 2`` time units:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> unit = NonRepairable(
        ...     surv.Exponential.from_params([0.01]),
        ...     surv.Exponential.from_params([0.5]),
        ... )
        >>> round(unit.failure_frequency(), 5)
        0.0098
        """
        if isinstance(self.reliability, NonParametric):
            raise ValueError(
                "Failure frequency requires a parametric reliability model"
            )
        mttf = model_mean(self.reliability)
        mttr = model_mean(self.time_to_replace)
        return 1.0 / (mttf + mttr)

    def _cost_rate_with_log_x(self, x):
        return self._cost_rate(np.exp(x))

    def find_optimal_replacement(self, options=None):
        """The replacement age that minimises the long-run cost rate.

        Minimises the age-replacement cost rate (see ``cost_rate``)
        ``(cp * R(t) + cu * (1 - R(t))) / integral_0^t R(u) du`` over the
        replacement age ``t``:

        - Parametric model: an exponential lifetime, or a Weibull with
          shape ``beta <= 1``, does not wear out, so preventive
          replacement never pays and ``inf`` is returned (unless the model
          has an offset ``gamma``, zero-inflation ``f0`` or a limited
          failure population ``p``). Otherwise ``scipy.optimize.minimize``
          (BFGS) searches over ``log(t)``, starting from the mean life,
          and its result is kept in the ``optimisation_results``
          attribute.
        - Non-parametric model: the cost rate is evaluated exactly at
          10,001 evenly spaced ages from 0 to the estimate's last time
          point, and at its time points, and the cheapest age is
          returned (the estimate says nothing about later ages).
        - ``StandbyModel``: its survival function may be a step function
          (a simulated arrangement), so the cost rate is evaluated on a
          grid of 2,001 ages up to where survival falls to 1e-6, and the
          cheapest grid age is refined with a bounded search between its
          neighbours. If survival never falls that far, ``inf`` is
          returned.

        Only the two cases above give ``inf``. For another lifetime
        without wear-out (e.g. a Gamma with shape below 1) the cost rate
        keeps falling towards the run-to-failure rate ``cu / MTTF``, and
        the search stops at a large, finite age where the curve has
        flattened; if in doubt, compare ``cost_rate`` over a range of
        ages.

        Parameters
        ----------
        options : object, optional
            Deprecated and ignored; it will be removed in a future release.

        Returns
        -------
        float
            The optimal replacement age, in the lifetime model's time
            units, or ``inf`` if preventive replacement never pays.

        Raises
        ------
        ValueError
            If the costs have not been set with
            ``set_costs_planned_and_unplanned()`` (the parametric ``inf``
            cases above return without them).

        Warns
        -----
        DeprecationWarning
            If ``options`` is given.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> unit = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
        >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
        >>> round(float(unit.find_optimal_replacement()), 1)
        493.0

        A memoryless (exponential) unit is never worth replacing early:

        >>> unit = NonRepairable(surv.Exponential.from_params([0.001]))
        >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
        >>> unit.find_optimal_replacement()
        inf
        """
        if options is not None:
            warnings.warn(
                "find_optimal_replacement()'s options argument is ignored "
                "and deprecated; it will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.model_parameterization == "parametric":
            if is_exponential(self.reliability) and not (
                getattr(self.reliability, "offset", False)
                or getattr(self.reliability, "zi", False)
                or getattr(self.reliability, "lfp", False)
            ):
                # Memoryless lifetime: an old unit is statistically as good
                # as a new one, so preventive replacement never pays.
                return np.inf
            if distribution_name(self.reliability) == "Weibull":
                if self.reliability.offset:
                    pass
                elif self.reliability.zi:
                    pass
                elif self.reliability.lfp:
                    pass
                elif self.reliability.params[1] <= 1:
                    return np.inf
        # The cases above need no costs; every search below does.
        self._require_costs()
        if self.model_parameterization == "parametric":
            # When using a parametric distribution the optimisation is
            # straight forward. Simply find the point in the support where
            # the cost rate is minimised. Uses quadrature to integrate!
            mean = self.reliability.mean()
            old_err_state = np.seterr(all="ignore")
            res = minimize(self._cost_rate_with_log_x, np.log(mean), tol=1e-10)
            np.seterr(**old_err_state)
            self.optimisation_results = res
            optimal = np.exp(res.x[0])
        elif self.model_parameterization == "standby":
            optimal = self._optimal_standby_replacement()
        else:
            # Scan the estimate's range on a grid that includes its knots,
            # on which the cumulative trapezoidal rule is exact.
            ages = np.union1d(
                np.linspace(0.0, self._knots[-1], 10001), self._knots
            )
            rates = _cost_rates(
                self.cp, self.cu, ages, self.reliability_function(ages)
            )
            optimal = float(ages[int(np.nanargmin(rates))])
        return optimal

    def _optimal_standby_replacement(self) -> float:
        """The cheapest replacement age for a standby arrangement.

        Its survival function may be a Kaplan-Meier step function, on which
        a gradient search stalls, so the cost rate is scanned on a grid of
        ages up to where the arrangement has all but surely failed, and the
        cheapest grid age refined with a bounded search between its
        neighbours.
        """
        # Bracket the age by which survival has fallen to 1e-6.
        upper = 1.0
        for _ in range(200):
            if self.reliability_function(upper) <= 1e-6:
                break
            upper *= 2.0
        else:
            # It may never fail (e.g. a limited failure population): the
            # cost rate keeps falling as the age grows, so never replace.
            return np.inf
        while upper > 1e-12 and self.reliability_function(upper / 2) <= 1e-6:
            upper /= 2.0
        ages = np.linspace(0.0, upper, 2001)
        R = np.clip(
            np.ravel(np.asarray(self.reliability.sf(ages), dtype=float)),
            0.0,
            1.0,
        )
        rates = _cost_rates(self.cp, self.cu, ages, R)
        i = int(np.nanargmin(rates))
        lo, hi = ages[max(i - 1, 1)], ages[min(i + 1, len(ages) - 1)]
        if hi <= lo:
            return float(ages[i])
        refined = minimize_scalar(
            self._cost_rate, bounds=(lo, hi), method="bounded"
        )
        return float(refined.x)

    def optimal_replacement_policy(self) -> MaintenancePolicy:
        """The optimal age-replacement policy as a typed result.

        Finds the replacement age with ``find_optimal_replacement()`` (see
        there for the search and its limits) and evaluates ``cost_rate``
        at it. When preventive replacement never pays (no wear-out: an
        exponential lifetime, or a Weibull with shape ``<= 1``) the
        interval is ``inf`` and the cost rate is the run-to-failure rate
        ``cu / MTTF``.

        Returns
        -------
        MaintenancePolicy
            ``interval`` is the cost-optimal replacement age and
            ``cost_rate`` the long-run cost per unit time under it.

        Raises
        ------
        ValueError
            If the costs have not been set (see
            ``set_costs_planned_and_unplanned()``).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> unit = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
        >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
        >>> policy = unit.optimal_replacement_policy()
        >>> round(policy.interval, 1), round(policy.cost_rate, 6)
        (493.0, 0.003462)

        Run to failure: ``cu / MTTF = 5 / 100``.

        >>> unit = NonRepairable(surv.Exponential.from_params([0.01]))
        >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
        >>> policy = unit.optimal_replacement_policy()
        >>> policy.interval, round(policy.cost_rate, 4)
        (inf, 0.05)
        """
        self._require_costs()
        interval = self.find_optimal_replacement()
        if np.isinf(interval):
            rate = self.cu / model_mean(self.reliability)
        else:
            rate = float(self._cost_rate(interval))
        return MaintenancePolicy(interval=float(interval), cost_rate=rate)

    def reset(self):
        """Restart the event sequence with a working, as-new unit.

        The next ``next_event()`` call then draws a time to failure. A
        [`RepairableRBD`][repyability.RepairableRBD] simulation resets each
        component this way at the start of every simulated history.
        """
        self.__next_event_type = FAILURE

    def next_event(self):
        """Draw the time to the unit's next state change.

        The event API a [`RepairableRBD`][repyability.RepairableRBD]
        simulation steps each component through. Calls alternate, starting
        (after construction or ``reset()``) with a failure:

        - a time to failure drawn from ``reliability``, returned as
          ``(time, False)``: the unit is now down;
        - a replacement time drawn from ``time_to_replace``, returned as
          ``(time, True)``: the unit is working again, as new.

        Each time is a duration measured from the previous event, not an
        absolute time. Each draw is one ``random(1)`` call on the model.
        surpyval parametric models draw from numpy's global RNG, so seed
        it (e.g. with ``np.random.seed``, which
        ``RepairableRBD.availability(seed=...)`` does) for reproducible
        draws; surpyval non-parametric models cannot be seeded this way.
        (For most parametric models the RBD simulation replays these same
        draws in batches instead of calling this method, with identical
        results.)

        Returns
        -------
        tuple of (float, bool)
            The time to the event, and whether the unit is working after
            it.

        Examples
        --------
        With fixed times, a failure 10 time units after each replacement,
        and replacements that take 2:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> unit = NonRepairable(
        ...     surv.ExactEventTime.from_params(10),
        ...     surv.ExactEventTime.from_params(2),
        ... )
        >>> unit.next_event()
        (10.0, False)
        >>> unit.next_event()
        (2.0, True)
        >>> unit.next_event()
        (10.0, False)
        >>> unit.reset()
        >>> unit.next_event()
        (10.0, False)
        """
        if self.__next_event_type == FAILURE:
            self.__next_event_type = REPLACE
            return self.reliability.random(1).item(), False
        elif self.__next_event_type == REPLACE:
            self.__next_event_type = FAILURE
            return self.time_to_replace.random(1).item(), True
