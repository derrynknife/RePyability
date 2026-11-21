import numpy as np
from scipy.optimize import minimize, minimize_scalar
from surpyval import ExactEventTime, NonParametric, Parametric

from repyability.maintenance import MaintenancePolicy, SkipValue
from repyability.rbd._bayesian_life import PosteriorPredictive, as_life
from repyability.rbd._covariate_schedule import is_scheduled
from repyability.rbd._degradation import check_life, is_degradation
from repyability.rbd._model_utils import (
    by_functions,
    distribution_name,
    is_exponential,
    is_mixture,
    life_functions,
    model_mean,
    never_fails,
)
from repyability.rbd._usage import first_clock_life
from repyability.rbd.helper_classes import PerfectReliability, perfect_class
from repyability.rbd.joint_node import NOT_REPAIRABLE, is_joint
from repyability.rbd.results import FailureForecast
from repyability.rbd.standby_node import StandbyModel
from repyability.utils.checks import is_number, no_distribution, real_array

FAILURE = 1
REPLACE = 0


#: The lives whose survival is a smooth curve, integrated by quadrature and
#: searched by a gradient method: a distribution, and a degradation process
#: (its first-passage time).
_SMOOTH = ("parametric", "degradation")


def _why_not(model) -> str:
    """What a surpyval model given as a life or a repair lacks of a life's
    functions, for a refusal's message ("." for another object)."""
    if not type(model).__module__.startswith("surpyval."):
        return "."
    missing = life_functions(model)
    if not missing:
        return "."
    hint = ""
    if callable(getattr(model, "at", None)):
        hint = (
            " It is a regression, whose functions need covariates: give "
            "the life of a unit at its covariates, model.at(Z) (or a "
            "RegressionNode in a NonRepairableRBD); for covariates that "
            "change over time in a RepairableRBD, give the model with the "
            "spec's 'covariates' (a StepSchedule)."
        )
    reasons = "; ".join(f"{name}: {why}" for name, why in missing.items())
    return f", which cannot work out {', '.join(missing)} ({reasons}).{hint}"


def _check_functions(model, what: str) -> None:
    """Refuse a surpyval model, given as a time to replace, that lacks a
    life's functions (a regression without its covariates, a curve that
    ends at its data), saying what it lacks: the repairs are drawn, and
    the exact methods integrate them, through those functions."""
    if not type(model).__module__.startswith("surpyval.") or isinstance(
        model, (Parametric, NonParametric)
    ):
        return
    if is_mixture(model) or isinstance(model, PosteriorPredictive):
        return
    if is_degradation(model) or by_functions(model):
        return
    raise ValueError(f"{what} is a {type(model).__name__}" + _why_not(model))


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
        - a surpyval ``MixtureModel`` (a population of several modes,
          such as infant mortality and wear-out), which has no quantile
          function: the simulations invert its distribution function;
        - a surpyval non-parametric estimate (e.g. a ``KaplanMeier``
          fit). For the cost calculations its survival function is taken
          as linear between its time points, starting from 1 at age 0,
          and held at its last value beyond the last one; the
          availability methods reject it, and its draws in
          ``next_event()`` cannot be seeded;
        - a [`StandbyModel`][repyability.StandbyModel], in any of its
          forms (a closed form, a convolution or a simulation), through its
          survival function ``sf``.
    time_to_replace : surpyval model or "instant", optional
        The distribution of the time taken to replace the unit after a
        failure, used by the availability and event methods. By default
        None, or ``"instant"`` as a ``RepairableRBD`` spec takes it:
        instantaneous replacement (``ExactEventTime.from_params(0)``).
    planned_cost : float, optional
        What a planned (preventive) replacement costs, at least 0, by name;
        as ``set_costs_planned_and_unplanned`` sets it (``cp``). By default
        None: not given.
    unplanned_cost : float, optional
        What a replacement at a failure costs, at least 0, by name (``cu``):
        with ``planned_cost``, more than it. Either may be given alone:
        ``failures`` needs only this one. By default None: not given.
    opportunity_cost : float, optional
        What a planned replacement at an opportunity (a stop or shut,
        [`Opportunity`][repyability.Opportunity]) costs, at least 0. By
        default None: an opportunity is priced as any other day, at
        ``planned_cost``.
    job_duration : float or str, optional
        How long replacing the part takes: an opportunity shorter than it
        is not used for this part (the policy's ``prices`` say so). On
        real dates text (``"6h"``) or a number of hours; in model time a
        number in the life's unit. By default None: every opportunity is
        used.
    opportunity_costs : dict, optional
        Prices at particular opportunities, overriding
        ``opportunity_cost``: keyed by the opportunity's name or the
        ``Opportunity`` itself. A policy refuses a key that names no
        opportunity of its collection, listing the names it has.

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
        If ``reliability`` is not a surpyval parametric, mixture or
        non-parametric model or a ``StandbyModel``, or ``time_to_replace``
        is text other than ``"instant"``.
    TypeError
        If either is a surpyval distribution itself (``surv.Weibull``)
        rather than a model of it, or ``time_to_replace`` is not a model.

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
        self,
        reliability,
        time_to_replace=None,
        *,
        planned_cost=None,
        unplanned_cost=None,
        opportunity_cost=None,
        job_duration=None,
        opportunity_costs=None,
    ):
        # A Bayesian fit is its posterior predictive.
        # A life at a usage rate is the distribution it is in hours.
        reliability = first_clock_life(
            "a NonRepairable", perfect_class(as_life(reliability))
        )
        time_to_replace = as_life(time_to_replace)
        no_distribution(reliability, "A NonRepairable's life")
        no_distribution(time_to_replace, "A NonRepairable's time_to_replace")
        if isinstance(time_to_replace, str):
            if time_to_replace != "instant":
                raise ValueError(
                    f"time_to_replace {time_to_replace!r} is no model: give "
                    "a time-to-replace model, or None or 'instant' for "
                    "replacement in no time."
                )
            time_to_replace = None
        if time_to_replace is None:
            # Replaced in no time.
            time_to_replace = ExactEventTime.from_params(0)
        elif not callable(getattr(time_to_replace, "mean", None)):
            raise TypeError(
                "time_to_replace must be a model of the time to replace (a "
                "surpyval distribution, such as "
                "surv.Exponential.from_params([rate])), or None or 'instant' "
                f"for replacement in no time, got {time_to_replace!r}."
            )
        else:
            _check_functions(
                time_to_replace, "A NonRepairable's time_to_replace"
            )
        if (
            isinstance(reliability, Parametric)
            or is_mixture(reliability)
            or isinstance(reliability, PosteriorPredictive)
        ):
            # A surpyval MixtureModel (#227) is no Parametric, but has the
            # survival function, distribution, density and mean the
            # analyses take; the simulations invert its distribution
            # function (it has no quantile function, see _sampling).
            self.model_parameterization = "parametric"
            self.reliability_function = reliability.sf
        elif isinstance(reliability, NonParametric):
            self.model_parameterization = "non-parametric"
            self.reliability_function, self._knots = _piecewise_linear_sf(
                reliability
            )
        elif by_functions(reliability):
            # Any other surpyval model with a life's functions (#93): a
            # Royston-Parmar fit, a regression at a unit's covariates
            # (regression.at(Z)), a destructive degradation model; or a
            # NonRepairableRBD, a subassembly replaced whole. Its
            # survival is a smooth curve, as a distribution's.
            self.model_parameterization = "parametric"
            self.reliability_function = reliability.sf
        elif is_degradation(reliability):
            # A fitted Wiener or gamma degradation process (#271): its life
            # is the time its level first reaches its threshold.
            check_life(reliability, "A NonRepairable's life")
            self.model_parameterization = "degradation"
            self.reliability_function = reliability.sf
        elif isinstance(reliability, StandbyModel):
            # Whatever the arrangement's survival function is (a closed
            # form or a numerical method), its sf gives it; one with neither
            # is only simulated, and its sf refuses.
            self.model_parameterization = "standby"
            self.reliability_function = _scalar_sf(reliability)
        elif is_scheduled(reliability):
            # A life under a covariate schedule on a RepairableRBD's clock
            # (a spec's "covariates"): only its simulations take it, and
            # its survival function refuses, saying so.
            self.model_parameterization = "covariate schedule"
            self.reliability_function = reliability.sf
        elif is_joint(reliability):
            raise ValueError(NOT_REPAIRABLE)
        elif reliability is PerfectReliability:
            raise ValueError(
                "A NonRepairable's life must end, and PerfectReliability "
                "never does. In a RepairableRBD, give PerfectReliability "
                "itself as the node, for a junction (such as a k-out-of-n "
                "vote point) or a part that never fails."
            )
        else:
            raise ValueError(
                "A NonRepairable's life must be a surpyval model with a "
                "life's functions of time (sf, ff, df, hf, Hf, qf and "
                "mean: a distribution, a MixtureModel, a Bayesian fit, a "
                "Royston-Parmar fit, a regression at one unit's covariates "
                "with regression.at(Z), a degradation process), a surpyval "
                "non-parametric estimate, a StandbyModel or a "
                "NonRepairableRBD (a subassembly replaced whole); got a "
                f"{type(reliability).__name__}" + _why_not(reliability)
            )

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
        self._set_costs(planned_cost, unplanned_cost)
        self._set_opportunity_prices(
            opportunity_cost=opportunity_cost,
            job_duration=job_duration,
            opportunity_costs=opportunity_costs,
        )

    def _set_opportunity_prices(
        self, *, opportunity_cost, job_duration, opportunity_costs
    ) -> None:
        """Keep the part's prices at opportunities, checked (the keys of
        ``opportunity_costs`` are checked against the opportunities when a
        policy runs)."""
        from repyability.maintenance._opportunities import (
            check_part_prices,
        )

        (
            self.opportunity_cost,
            self.job_duration,
            self.opportunity_costs,
        ) = check_part_prices(
            opportunity_cost, job_duration, opportunity_costs
        )

    def _set_costs(self, planned, unplanned) -> None:
        """Keep the costs given to the constructor (``cp``, ``cu``): each
        a finite number at least 0, and with both, the planned less."""
        for name, cost in (
            ("planned_cost", planned),
            ("unplanned_cost", unplanned),
        ):
            if cost is not None and not (
                is_number(cost) and 0.0 <= float(cost) < np.inf
            ):
                raise ValueError(
                    f"{name} must be a number, at least 0; got {cost!r}."
                )
        if planned is not None and unplanned is not None:
            self.set_costs_planned_and_unplanned(
                float(planned), float(unplanned)
            )
        elif planned is not None:
            self.cp = float(planned)
        elif unplanned is not None:
            self.cu = float(unplanned)

    def __repr__(self) -> str:
        """A short summary: the life model's distribution, how a
        replacement takes its time, and the costs given."""
        parts = [
            distribution_name(self.reliability)
            or type(self.reliability).__name__
        ]
        if self.time_to_replace is not None:
            replace = distribution_name(self.time_to_replace)
            parts.append(
                f"replaced in {replace or type(self.time_to_replace).__name__}"
            )
        if hasattr(self, "cp"):
            parts.append(f"planned_cost={self.cp:g}")
        if hasattr(self, "cu"):
            parts.append(f"unplanned_cost={self.cu:g}")
        if getattr(self, "opportunity_cost", None) is not None:
            parts.append(f"opportunity_cost={self.opportunity_cost:g}")
        return f"NonRepairable({', '.join(parts)})"

    def failures(self, t, age: float = 0.0) -> FailureForecast:
        """The failures over a window ``t`` from now, with the current unit
        ``age`` old and working, each failure replaced by a new unit.

        The chance the current unit fails in the window is its life
        conditioned on the age reached, ``1 - R(age + t) / R(age)``. The
        expected number of failures counts its replacements' too: the
        renewal count of the part as a
        [`RepairableRBD`][repyability.RepairableRBD]'s component started
        from that age (its ``expected_events``, exact, with no
        simulation), each replacement taking its ``time_to_replace``.

        Parameters
        ----------
        t : float or array_like
            The window's length (or lengths), at least 0, in the models'
            time unit.
        age : float, optional
            The current unit's age, at least 0, by default 0 (new).

        Returns
        -------
        FailureForecast
            The chance the current unit fails, the expected failures, and
            their expected cost (with an ``unplanned_cost``).

        Examples
        --------
        A bearing with a Weibull life (scale 365 days, shape 2), 224 days
        old, over the next 82 days, at 400 a failure:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]), unplanned_cost=400.0
        ... )
        >>> forecast = bearing.failures(82.0, age=224.0)
        >>> round(forecast.probability, 4)
        0.2784
        >>> round(forecast.expected_failures, 4)
        0.283
        >>> round(forecast.expected_cost, 2)
        113.2
        """
        from repyability.rbd.node_state import NodeState
        from repyability.rbd.repairable_rbd import RepairableRBD

        window = real_array(t, "t")
        if not np.all(np.isfinite(window) & (window >= 0.0)):
            raise ValueError(f"t must be at least 0, and finite; got {t!r}.")
        if not (is_number(age) and 0.0 <= float(age) < np.inf):
            raise ValueError(f"age must be a number, at least 0; got {age!r}.")
        age = float(age)
        surviving = float(np.ravel(self.reliability_function(age))[0])
        if not surviving > 0.0:
            raise ValueError(
                f"The life gives no chance of a unit working at age {age:g}."
            )
        reached = np.asarray(self.reliability_function(age + window), float)
        probability = 1.0 - reached.reshape(window.shape) / surviving
        unit = RepairableRBD([("s", "unit"), ("unit", "t")], {"unit": self})
        events = unit.expected_events(
            window if window.ndim else float(window),
            state={"unit": NodeState(age=age)},
        )
        failures = np.asarray(events.node_failures["unit"], float)
        cost = None if not hasattr(self, "cu") else failures * self.cu

        def shaped(values):
            return float(values) if np.ndim(values) == 0 else values

        return FailureForecast(
            shaped(window),
            age,
            shaped(probability),
            shaped(failures),
            None if cost is None else shaped(cost),
        )

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
        if not (hasattr(self, "cp") and hasattr(self, "cu")):
            raise ValueError(
                "costs not set: give planned_cost and unplanned_cost, or "
                "call set_costs_planned_and_unplanned(cp, cu) first"
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
        from scipy.integrate import quad, trapezoid

        if self.model_parameterization in _SMOOTH:
            # surpyval evaluates an offset model below its offset through a
            # fractional power of a negative number before masking it, which
            # warns although the survival (1) is right.
            # (Through a float: some models' sf gives an array for a scalar
            # age, which quad cannot take.)
            with np.errstate(invalid="ignore"):
                out = quad(_scalar_sf(self.reliability), 0, t)[0]
        elif self.model_parameterization == "standby":
            # A numerical survival function is interpolated on its grid,
            # with kinks quadrature handles poorly: integrate on a fine grid
            # instead.
            ages = np.linspace(0.0, float(t), 4001)
            R = np.ravel(np.asarray(self.reliability.sf(ages), dtype=float))
            out = float(trapezoid(np.clip(R, 0.0, 1.0), ages))
        elif self.model_parameterization == "covariate schedule":
            raise self.reliability._no_reliability()
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

        ``MTTR / (MTTF + MTTR)``, worked out in its own right (not as one
        less the availability), so that a small one keeps its precision;
        with units that never fail or replacements that never finish, the
        probability that the unit ends down for good (see
        ``mean_availability``).

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
        if isinstance(self.reliability, NonParametric):
            raise ValueError(
                "Mean Availability requires a parametric reliability model"
            )
        return self._long_run_unavailability()

    def mean_availability(self) -> float:
        """Long-run availability, ``MTTF / (MTTF + MTTR)``.

        The steady-state fraction of time the unit is up when it is run to
        failure and then replaced, so up and down times alternate (an
        alternating renewal process). ``MTTF`` is the mean of
        ``reliability`` and ``MTTR`` the mean of ``time_to_replace``.
        For a simulated ``StandbyModel`` lifetime the MTTF is the mean of
        the lifetimes simulated when it was built (see its ``mean``), so it
        is the same on every call.

        If some units never fail (a limited-failure-population model,
        ``p < 1``), sooner or later a replacement is one of them, and the
        unit is then up for good: the long-run availability is 1. Likewise,
        if some replacements never finish, the unit ends down for good.
        With both, it ends up for good with probability
        ``u / (u + (1 - u) * d)``, for ``u`` the fraction of units that
        never fail and ``d`` the fraction of replacements that never finish,
        and that is its long-run availability.

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

        One in ten units never fails (``lfp_p``, the share that ever
        fails, is 0.9), so in the long run the unit is up.

        >>> cured = NonRepairable(
        ...     surv.Exponential.from_params([0.01], lfp_p=0.9),
        ...     surv.Exponential.from_params([0.5]),
        ... )
        >>> cured.mean_availability()
        1.0
        >>> cured.failure_frequency()
        0.0
        """
        if isinstance(self.reliability, NonParametric):
            raise ValueError(
                "Mean Availability requires a parametric reliability model"
            )
        return self._long_run()[0]

    def _long_run(self) -> tuple:
        """The long-run availability and failure frequency (see
        ``mean_availability`` and ``failure_frequency``)."""
        up_for_good = never_fails(self.reliability)
        down_for_good = never_fails(self.time_to_replace)
        if up_for_good or down_for_good:
            # Each cycle ends up for good, down for good, or in another
            # cycle: after finitely many failures the unit stays put.
            fails = 1.0 - up_for_good
            return up_for_good / (up_for_good + fails * down_for_good), 0.0
        mttf = model_mean(self.reliability)
        mttr = model_mean(self.time_to_replace)
        return mttf / (mttr + mttf), 1.0 / (mttf + mttr)

    def _long_run_unavailability(self) -> float:
        """The long-run unavailability (see ``mean_unavailability``), as
        ``_long_run`` gives the availability."""
        up_for_good = never_fails(self.reliability)
        down_for_good = never_fails(self.time_to_replace)
        if up_for_good or down_for_good:
            ends_down = (1.0 - up_for_good) * down_for_good
            return ends_down / (up_for_good + ends_down)
        mttf = model_mean(self.reliability)
        mttr = model_mean(self.time_to_replace)
        return mttr / (mttr + mttf)

    def failure_frequency(self) -> float:
        """Long-run failure frequency (failures per unit time).

        For an alternating renewal process (fail, replace, fail, ...) this
        is ``1 / (MTTF + MTTR)``: one failure per mean up-down cycle, with
        ``MTTF`` the mean of ``reliability`` and ``MTTR`` the mean of
        ``time_to_replace``. It is 0 if some units never fail or some
        replacements never finish: the unit then fails only finitely often
        (see ``mean_availability``).

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
        return self._long_run()[1]

    def _cost_rate_with_log_x(self, x):
        return self._cost_rate(np.exp(x))

    def find_optimal_replacement(self, *, discount_rate=0.0):
        """The replacement age that minimises the long-run cost rate, or,
        with a discount rate, the equivalent annual cost.

        Minimises the age-replacement cost rate (see ``cost_rate``)
        ``(cp * R(t) + cu * (1 - R(t))) / integral_0^t R(u) du`` over the
        replacement age ``t``:

        - Parametric model: an exponential lifetime, or a Weibull with
          shape ``beta <= 1``, does not wear out, so preventive
          replacement never pays and ``inf`` is returned at once (unless
          the model has an offset ``gamma`` or zero-inflation ``f0``). So
          does a limited failure population (``p < 1``), whatever its
          shape: running to failure then costs nothing in the long run
          (see below). Otherwise
          ``scipy.optimize.minimize`` (BFGS) searches over ``log(t)``,
          starting from the mean life, and its result is kept in the
          ``optimisation_results`` attribute. With an offset ``gamma`` no
          unit can fail before that age, so the cost rate falls as
          ``cp / t`` up to it and replacing at ``gamma`` itself is also
          considered. If the best age found is no cheaper than running to
          failure (see below), ``inf`` is returned.
        - A degradation process: searched as a parametric model is, from
          the mean of its first-passage time.
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

        Preventive replacement must beat running to failure, whose long-run
        cost rate is ``cu / MTTF`` -- or 0 when some units never fail (a
        limited failure population): the units that never fail are then
        kept. Without wear-out (e.g. a Weibull or Gamma with shape below 1,
        with or without an offset, zero-inflation or a limited failure
        population) the cost rate only falls towards that rate as the age
        grows, so ``inf`` is returned rather than wherever the search
        stopped on the flattening curve. A replacement age must save more
        than one part in a million to count.

        **With a discount rate** money spent later is worth less now, so
        the age is chosen to make the present value of every replacement,
        forever, the least (Fox, 1966); ``discount_rate`` times that value
        is the equivalent annual cost (EAC), a steady cost per unit time
        worth the same. With ``r`` the rate, ``R`` the reliability and
        ``M(t)`` the integral of ``exp(-r u) R(u)`` from 0 to ``t``::

            EAC(t) = [cu * (1 - exp(-r t) R(t) - r M(t))
                      + cp * exp(-r t) R(t)] / M(t)

        which is the cost rate above at ``r = 0``. Discounting puts off
        the planned replacement's cost, so for a part that wears out the
        best age is usually older than undiscounted. The cost is scanned
        on a grid of ages (the trapezoidal rule) up to where the part has
        all but surely failed or a cost is discounted to ``exp(-40)``, and
        the cheapest age refined by a bounded search on the cost by
        quadrature (a non-parametric estimate: its own grid, as above).
        An exponential life, or a Weibull of shape ``<= 1``, still returns
        ``inf`` at once; a limited failure population is searched, as
        running to failure is no longer free when the failures to come
        are discounted. ``inf`` is returned where no age beats running to
        failure, whose EAC is ``cu * (1 - r M) / M`` with ``M`` the
        integral to infinity.

        Parameters
        ----------
        discount_rate : float, optional
            The continuous discount rate per unit of the lifetime model's
            time, by default 0 (undiscounted, the long-run cost rate). For
            a yearly rate of 7% with a life in hours, ``math.log(1.07) /
            8760``.

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

        Discounted at 10% a year, with the life in hours, the best age is
        a little older:

        >>> import math
        >>> unit = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
        >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
        >>> a_year = math.log(1.10) / 8760
        >>> best = unit.find_optimal_replacement(discount_rate=a_year)
        >>> round(float(best), 1)
        493.4
        """
        from repyability.maintenance._options import (
            discount_rate as _discount_rate,
        )

        rate = _discount_rate(discount_rate)
        if rate > 0.0:
            return self._discounted_optimum(rate)[0]
        if self.model_parameterization == "parametric":
            if never_fails(self.reliability) > 0.0:
                # Some units never fail: in the long run one of them is in
                # place for good, so running to failure costs nothing per
                # unit time and any replacement age costs more.
                return np.inf
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
        if self.model_parameterization in _SMOOTH:
            # When using a parametric distribution the optimisation is
            # straight forward. Simply find the point in the support where
            # the cost rate is minimised. Uses quadrature to integrate!
            mean = model_mean(self.reliability)
            old_err_state = np.seterr(all="ignore")
            res = minimize(self._cost_rate_with_log_x, np.log(mean), tol=1e-10)
            self.optimisation_results = res
            optimal = float(np.exp(res.x[0]))
            rate = self._cost_rate(optimal) if np.isfinite(optimal) else np.inf
            # No unit fails before an offset, so up to it the cost rate is
            # cp / t, lowest at the offset itself; the search can stop short
            # of that kink.
            offset = float(getattr(self.reliability, "gamma", 0) or 0)
            if getattr(self.reliability, "offset", False) and offset > 0:
                at_offset = self._cost_rate(offset)
                if at_offset < rate:
                    optimal, rate = offset, at_offset
            np.seterr(**old_err_state)
            # Replacing must beat running to failure; without wear-out the
            # cost rate only falls towards that rate, and the search stops
            # wherever the curve has flattened.
            if not rate < self._run_to_failure_rate() * (1 - 1e-6):
                return np.inf
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

    def _memoryless(self) -> bool:
        """Whether the life is one preventive replacement never improves,
        discounted or not: an exponential, or a Weibull of shape at most
        1, with no offset, zero-inflation or limited failure
        population."""
        if self.model_parameterization != "parametric":
            return False
        model = self.reliability
        if (
            getattr(model, "offset", False)
            or getattr(model, "zi", False)
            or getattr(model, "lfp", False)
        ):
            return False
        if is_exponential(model):
            return True
        return distribution_name(model) == "Weibull" and model.params[1] <= 1

    def _discounted_optimum(self, rate: float):
        """The age-replacement age with the least equivalent annual cost at
        ``rate`` (more than 0), and that cost (see
        ``find_optimal_replacement``)."""
        from repyability.maintenance._options import (
            discounted_optimum,
            discounted_run_to_failure,
        )

        if self._memoryless():
            if self.cu is None:
                return np.inf, np.nan
            return np.inf, discounted_run_to_failure(self, rate, self.cu)
        self._require_costs()
        return discounted_optimum(self, rate, self.cp, self.cu)

    def _run_to_failure_rate(self) -> float:
        """The long-run cost rate with no preventive replacement,
        ``cu / MTTF``: 0 when some units never fail (a limited failure
        population), since those are then kept for good."""
        with np.errstate(all="ignore"):
            never_fail = float(self.reliability_function(np.inf))
        if never_fail > 0.0:
            return 0.0
        return self.cu / model_mean(self.reliability)

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

    def optimal_replacement_policy(
        self, *, discount_rate=0.0
    ) -> MaintenancePolicy:
        """The optimal age-replacement policy as a typed result.

        Finds the replacement age with ``find_optimal_replacement()`` (see
        there for the search and its limits) and evaluates ``cost_rate``
        at it. When preventive replacement never pays (no wear-out, e.g.
        an exponential lifetime or a Weibull with shape ``<= 1``) the
        interval is ``inf`` and the cost rate is the run-to-failure rate
        ``cu / MTTF``, or 0 when some units never fail.

        With a discount rate the age is the one with the least equivalent
        annual cost (EAC; Fox, 1966): the present value of every
        replacement, forever, spread as a steady cost per unit time. The
        result then carries that present value (``npv``) and ``eac``, and
        ``cost_rate`` is still the undiscounted long-run cost rate of the
        age chosen (of running to failure, where ``interval`` is ``inf``).

        Parameters
        ----------
        discount_rate : float, optional
            The continuous discount rate per unit of the lifetime model's
            time, by default 0 (undiscounted). For a yearly rate of 7%
            with a life in hours, ``math.log(1.07) / 8760``.

        Returns
        -------
        MaintenancePolicy
            ``interval`` is the cost-optimal replacement age and
            ``cost_rate`` the long-run cost per unit time under it; with a
            discount rate, ``discount_rate``, ``npv`` and ``eac`` too.

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

        Discounted at 10% a year, the life in hours:

        >>> import math
        >>> unit = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
        >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
        >>> rate = math.log(1.1) / 8760
        >>> policy = unit.optimal_replacement_policy(discount_rate=rate)
        >>> round(policy.interval, 1), round(policy.eac * 8760, 2)
        (493.4, 30.27)
        """
        from repyability.maintenance._options import (
            discount_rate as _discount_rate,
        )

        self._require_costs()
        discount = _discount_rate(discount_rate)
        if discount > 0.0:
            interval, eac = self._discounted_optimum(discount)
        else:
            interval = self.find_optimal_replacement()
        if np.isinf(interval):
            rate = self._run_to_failure_rate()
        else:
            rate = float(self._cost_rate(interval))
        if discount == 0.0:
            return MaintenancePolicy(interval=float(interval), cost_rate=rate)
        return MaintenancePolicy(
            interval=float(interval),
            cost_rate=rate,
            discount_rate=discount,
            npv=float(eac) / discount,
            eac=float(eac),
        )

    def optimal_next_replacement(
        self,
        age,
        *,
        now_cost=None,
        scheduled_outages=(),
        policy=None,
        site=None,
        plan=None,
        now=None,
        readings=None,
        unit=None,
    ):
        """When to replace the part in service next: now, at one of the
        scheduled outages, at a planned replacement between them, or at
        its failure.

        [`optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy]
        is the rule for a part in general; this is the next replacement of
        the one in service, from its age and the outages scheduled (this
        package's use case UC-12). Each choice's cost is its expected cost
        until the part is replaced (at the failure price if it fails
        first), less the time the part serves until then, valued at the
        baseline: the long-run cost a unit of time of how the position is
        run, ``policy.cost_rate``. A policy chosen at a discount values it
        at its equivalent annual cost instead, and each choice is
        discounted at the policy's rate (its present value and the
        policy's after it, Fox, 1966; ``discount_rate`` on the result).
        By default that is the age
        replacement's (``optimal_replacement_policy().cost_rate``); where
        cheap shuts keep coming, give the calendar's policy
        ([`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy]),
        whose rate is lower, so a part is kept longer. The cheapest
        choice is the best. With no outages and the default baseline,
        replacing now is the best from the optimal replacement age.

        A planned replacement is one later than now, at its best time.
        Where no later time beats replacing at once (a part past its
        optimal age), replacing at once is replacing now, which costs
        ``now_cost``: the planned choice is then replacing now (time 0,
        at ``now_cost``), ``now`` is True exactly when the best choice
        is at time 0, and ``replace_from_age`` follows. A planned
        replacement with a lead time is a scheduled outage at that time.

        Parameters
        ----------
        age : float
            The part's age now, at least 0, in the life's time unit.
        now_cost : float, optional
            What replacing it now costs, by default the planned cost.
        scheduled_outages : list of (float, float), optional
            The scheduled outages (shuts, turnarounds) as (time from now,
            price) pairs: when each is, in the life's time unit, more than
            0, and what replacing the part at it costs, at least 0.
        policy : CalendarPolicy, MaintenancePolicy or ShutdownPolicy, optional
            The policy whose ``cost_rate`` is the baseline, by default
            ``optimal_replacement_policy()`` (with a site, the site's
            policy, worked out by its route).
        site : Opportunities, optional
            On real dates: the site's opportunities, priced for this part
            as
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy]
            prices them (a deprecated ``Shutdowns`` too). The choices are
            then now, each of the next two shuts of each kind, a planned
            replacement on any other day (where the site allows one) and
            running to failure, on the real dates, with the time beyond
            valued at the policy's figure (its cost a year, or for a
            horizon its net present cost spread evenly over its years).
            ``age`` may be text in the part's unit (``"3,100 h"``).
        plan : optional
            As for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        now : optional
            As for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        readings : optional
            As for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        unit : optional
            As for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].

        Returns
        -------
        NextReplacement or NextReplacementOnDates
            Every choice with its cost and break-even age (the youngest at
            which replacing now costs no more), the best, and the age from
            which replacing now is the best; with a site, each choice with
            its date and the best in words.

        Raises
        ------
        ValueError
            If the costs have not been set, or an argument is out of its
            range.
        TypeError
            If ``policy`` is not a policy (its cost rate alone is not one).
        NotImplementedError
            For a life with no survival function in closed form (a
            non-parametric estimate or a ``StandbyModel``).

        Examples
        --------
        A bearing with a Weibull life (scale 365 days, shape 2), 200 days
        old, $150 for a planned replacement, $400 at a failure, and
        outages scheduled in 150 and 330 days, at which replacing it costs
        $100:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> found = bearing.optimal_next_replacement(
        ...     200.0, scheduled_outages=[(150.0, 100.0), (330.0, 100.0)]
        ... )
        >>> found.now, found.best.kind, found.best.time
        (False, 'outage', 150.0)
        >>> round(found.replace_from_age, 1)
        297.6

        With shuts every 180 days at $100 kept coming, the calendar's
        policy values a day of service at 0.990 rather than 1.117, and
        replacing now beats the outage in 150 days from a younger age:

        >>> from repyability import Opportunities, Opportunity
        >>> at_shuts = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ...     opportunity_cost=100.0,
        ... )
        >>> shuts = at_shuts.optimal_opportunity_policy(
        ...     Opportunities([Opportunity("shut", period=180.0)])
        ... )
        >>> found = bearing.optimal_next_replacement(
        ...     200.0,
        ...     scheduled_outages=[(150.0, 100.0), (330.0, 100.0)],
        ...     policy=shuts,
        ... )
        >>> outage = found.choices[1]
        >>> round(found.cost_rate, 3), round(outage.break_even_age, 1)
        (0.99, 256.1)
        """
        from repyability.maintenance._next import optimal_next_replacement
        from repyability.maintenance._site_policy import next_on_dates

        if site is not None:
            if now_cost is not None or tuple(scheduled_outages):
                raise ValueError(
                    "With a site, the shuts and the price on any other day "
                    "come from it: give no now_cost or scheduled_outages."
                )
            from repyability.maintenance._opportunities import resolve

            site, _ = resolve(self, site, on_dates=True)
            return next_on_dates(
                self,
                age,
                site,
                plan=plan,
                now=now,
                readings=readings,
                unit=unit,
                policy=policy,
            )
        _calendar_only(plan=plan, now=now, readings=readings, unit=unit)
        return optimal_next_replacement(
            self,
            age,
            now_cost=now_cost,
            scheduled_outages=scheduled_outages,
            policy=policy,
        )

    def skip_next_replacement(
        self, every, *, resets, time_to_next=None, discount_rate=0.0
    ) -> SkipValue:
        """Whether to skip the next planned replacement, right after a
        failure (a new unit just in): the present-value cost of doing it
        against that of skipping it, and the NPV of skipping.

        The part is replaced when it fails (at its ``unplanned_cost``) and
        on a plan (at its ``planned_cost``), in no time either way (this
        package's use case UC-6). The plan either resets at each
        replacement, planned or not (age replacement: the next planned
        replacement is ``every`` after the failure), or keeps to fixed
        times (block replacement: the next is ``time_to_next`` after the
        failure, and the following one ``every`` later). Both costs are
        exact, with no simulation; what each covers is in
        [`SkipValue`][repyability.SkipValue].
        [`optimal_next_replacement`][repyability.NonRepairable.optimal_next_replacement]
        is the undiscounted, long-run question of a part in service with
        no plan in force.

        Parameters
        ----------
        every : float
            The plan's interval, more than 0, in the life's time unit.
        resets : bool
            Whether the plan resets at each replacement: True, the next
            planned replacement is ``every`` after the last replacement,
            planned or not; False, planned replacements fall on fixed
            times, ``every`` apart, whatever happens.
        time_to_next : float, optional
            For a plan that does not reset: the time from the failure to
            the next planned replacement, more than 0 and at most
            ``every``. A plan that resets has its next planned replacement
            ``every`` after the failure, and takes none.
        discount_rate : float, optional
            The continuous discount rate per unit time of the life, by
            default 0 (undiscounted): for an annual rate of 8% with a life
            in days, ``math.log(1.08) / 365.25``.

        Returns
        -------
        SkipValue
            The present-value cost of doing the replacement (``replace``)
            and of skipping it (``skip``), their difference (``npv``:
            positive, skipping pays), the ``best`` of the two, and what
            they were worked out for. On a plan that resets each is the
            whole future's, endless (``inf``) with no discounting, where
            only ``npv`` is finite. A discount rate that discounts the
            costs away (a rate a year given with a life in days) is
            warned of.

        Raises
        ------
        ValueError
            If the costs have not been set, or an argument is out of its
            range, or ``time_to_next`` is given with a plan that resets or
            missing from one that does not.
        NotImplementedError
            For a life with no survival function in closed form (a
            non-parametric estimate or a ``StandbyModel``).

        Examples
        --------
        A bearing with a Weibull life (scale 365 days, shape 2), replaced
        every 180 days for $150, and at a failure for $400; right after a
        failure, at 8% a year, skipping the next planned replacement
        saves:

        >>> import math
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> a_day = math.log(1.08) / 365.25
        >>> value = bearing.skip_next_replacement(
        ...     180.0, resets=True, discount_rate=a_day
        ... )
        >>> round(value.npv, 2), value.best
        (26.56, 'skip')

        On fixed times, it depends on how soon the next planned
        replacement is:

        >>> value = bearing.skip_next_replacement(
        ...     180.0, resets=False, time_to_next=30.0, discount_rate=a_day
        ... )
        >>> round(value.npv, 2)
        123.08
        """
        from repyability.maintenance._skip import skip_next_replacement

        return skip_next_replacement(
            self,
            every,
            resets=resets,
            time_to_next=time_to_next,
            discount_rate=discount_rate,
        )

    def compare_replacement_options(
        self,
        age,
        due_in,
        every,
        *,
        defer_by=(),
        shuts=None,
        discount_rate=0.0,
    ):
        """Compare the options for the part's next planned replacement as
        present-value costs, from its age now: replace as planned, replace
        now, defer it, move it to a shut, or skip it.

        In plain terms: the plan replaces the part on fixed times, the next
        ``due_in`` from now and the following one ``every`` later; a
        failure is replaced at once. Each option is costed to the
        following planned replacement, where every option has a new part
        in, so they are compared like for like: the failures each lets
        happen, and the price each replacement is done at. It extends
        [`skip_next_replacement`][repyability.NonRepairable.skip_next_replacement]
        (which it agrees with for a new part) to a part of any age and to
        deferring and moving the work. Exact, with no simulation.

        Parameters
        ----------
        age : float
            The part's age now, at least 0.
        due_in : float
            The time from now to the next planned replacement, more than 0
            and at most ``every``.
        every : float
            The plan's interval.
        defer_by : float or array_like, optional
            Delays to try: replace that long after it is due.
        shuts : mapping or list, optional
            Other times to do the work, from now, each with its own price:
            ``{name: (time, price)}`` or a list of ``(time, price)``.
        discount_rate : float, optional
            The continuous discount rate per unit of time, by default 0.

        Returns
        -------
        ReplacementComparison
            Every option's cost and what it saves against the plan, and
            the cheapest.

        Raises
        ------
        ValueError
            If an argument is out of range or the costs are not set.

        Examples
        --------
        A bearing (Weibull, scale 365 days, shape 2), 200 days old, its
        planned replacement ($150) due in 30 days and the next 180 days
        later; a failure costs $400, and a shut in 90 days replaces it for
        $100:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> result = bearing.compare_replacement_options(
        ...     200.0, 30.0, 180.0, defer_by=[30.0],
        ...     shuts={"shut": (90.0, 100.0)},
        ... )
        >>> [round(o.cost) for o in result.options]
        [277, 269, 289, 257, 274]
        >>> [o.name for o in result.options]
        ['replace', 'now', 'defer by 30', 'shut', 'skip']
        >>> result.best.name
        'shut'
        """
        from repyability.maintenance._options import (
            compare_replacement_options,
        )

        return compare_replacement_options(
            self,
            age,
            due_in,
            every,
            defer_by=defer_by,
            shuts=shuts,
            discount_rate=discount_rate,
        )

    def delay_effect(
        self, age, planned_in, done_in, *, weight=1.0, discount_rate=0.0
    ):
        """What delaying a planned replacement adds: the extra chance the
        part fails first, and the extra expected cost, from its age now.

        The replacement planned ``planned_in`` from now is done ``done_in``
        from now instead (a missed visit, a postponed shut). The part is
        replaced at its planned cost if it lasts, at its unplanned cost if
        it fails first; the delay adds the difference of the two cycles.
        Exact. ``weight`` is the chance the delay happens (a visit is only
        sometimes missed), for the expected added risk and cost.

        Parameters
        ----------
        age : float
            The part's age now.
        planned_in : float
            When the replacement was planned and when it is done, from
            now; ``done_in`` at least ``planned_in``.
        done_in : float
            The companion of ``planned_in`` above.
        weight : float, optional
            The chance the delay happens, by default 1.
        discount_rate : float, optional
            The continuous discount rate, by default 0.

        Returns
        -------
        DelayEffect
            The risk and cost planned and delayed, and what the delay
            adds.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> late = bearing.delay_effect(200.0, 10.0, 40.0)
        >>> round(late.added_failure_probability, 4)
        0.0934
        >>> round(late.added_cost, 2)
        23.36
        """
        from repyability.maintenance._options import delay_effect

        return delay_effect(
            self,
            age,
            planned_in,
            done_in,
            weight=weight,
            discount_rate=discount_rate,
        )

    def cost_of_waiting_for_visit(
        self, age, due_in, visits, *, discount_rate=0.0
    ):
        """What it costs to round a due replacement up to the next visit:
        the work comes due ``due_in`` from now, but can only be done at the
        next of ``visits``.

        ``visits`` is an [`Opportunity`][repyability.Opportunity] (visits
        every so often, or on given times) or a list of times from now. The
        replacement is done at the first visit at or after the due time;
        the result is that of
        [`delay_effect`][repyability.NonRepairable.delay_effect] from the
        due time to that visit.

        Parameters
        ----------
        age : float
            The part's age now.
        due_in : float
            The time from now at which the replacement comes due.
        visits : Opportunity or array_like
            The visits, in the part's time unit.
        discount_rate : float, optional
            The continuous discount rate, by default 0.

        Returns
        -------
        DelayEffect
            With ``done_in`` the visit's time.

        Raises
        ------
        ValueError
            If no visit comes at or after the due time.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable, Opportunity
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> weekly = Opportunity("weekly", period=7.0)
        >>> result = bearing.cost_of_waiting_for_visit(200.0, 10.0, weekly)
        >>> result.done_in, round(result.added_failure_probability, 4)
        (14.0, 0.0123)
        """
        from repyability.maintenance._options import (
            delay_effect,
            next_visit,
        )

        due = float(due_in)
        return delay_effect(
            self,
            age,
            due,
            next_visit(visits, due),
            discount_rate=discount_rate,
        )

    def cost_of_missed_visit(
        self,
        age,
        visit_in,
        next_visit_in,
        *,
        miss_probability=1.0,
        discount_rate=0.0,
    ):
        """The risk and cost added when a visit is missed and its work
        waits for the next one.

        The work planned for the visit ``visit_in`` from now is done at the
        next, ``next_visit_in``. With ``miss_probability`` below 1 the
        added risk and cost are also given as expected values over the
        chance of a miss (``expected_added_*``). It is
        [`delay_effect`][repyability.NonRepairable.delay_effect] with the
        chance a visit is missed as its weight.

        Parameters
        ----------
        age : float
            The part's age now.
        visit_in : float
            The times from now of the visit and of the next one.
        next_visit_in : float
            The companion of ``visit_in`` above.
        miss_probability : float, optional
            The chance the visit is missed, by default 1.
        discount_rate : float, optional
            The continuous discount rate, by default 0.

        Returns
        -------
        DelayEffect

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> r = bearing.cost_of_missed_visit(
        ...     200.0, 10.0, 40.0, miss_probability=0.1
        ... )
        >>> round(r.expected_added_cost, 2)
        2.34
        """
        from repyability.maintenance._options import delay_effect

        return delay_effect(
            self,
            age,
            visit_in,
            next_visit_in,
            weight=miss_probability,
            discount_rate=discount_rate,
        )

    def slack_effect(self, interval, slack):
        """The cost and risk of allowing a replacement to slip: it is due
        at age ``interval`` and may be done up to ``slack`` late.

        In plain terms: how much more often does the part fail first, and
        how much more does it cost in the long run, if the work is done
        at the latest age allowed, ``interval + slack``, instead of on
        time? Age replacement with the part's planned and unplanned costs,
        exact (the long-run cost rate is
        [`optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy]'s).
        The mean over a uniform lateness is given as well.

        Parameters
        ----------
        interval : float
            The age at which the replacement is due, more than 0.
        slack : float
            How late it may be done, at least 0.

        Returns
        -------
        SlackEffect

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=150.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> effect = bearing.slack_effect(300.0, 60.0)
        >>> round(effect.added_risk, 4), round(effect.added_cost_rate, 5)
        (0.1309, 0.01136)
        """
        from repyability.maintenance._options import slack_effect

        return slack_effect(self, interval, slack)

    def age_replacement_value(self, age, *, discount_rate=0.0):
        """What replacing the part at ``age``, or when it fails, costs for
        good: its net present value (Fox, 1966) and equivalent annual
        cost. Exact, with no simulation.

        In plain terms: the part is replaced by a new one at ``age`` (cost
        ``cp``) or at a failure (cost ``cu``), again and again. Counting
        each replacement at what it is worth now, the sum of them all is
        finite when discounted. The equivalent annual cost is the steady
        cost per unit time worth the same: the discounted counterpart of
        ``cost_rate(age)``, to which it comes as the discount rate goes to
        0. For a whole plant, with value earned and a cash-flow table, see
        [`RepairableRBD.expected_npv`][repyability.RepairableRBD.expected_npv].

        With ``R`` the part's reliability, ``f`` its density and ``r`` the
        discount rate, the present value is::

            [cu * integral_0^T exp(-r t) f(t) dt + cp * exp(-r T) R(T)]
              / [1 - integral_0^T exp(-r t) f(t) dt - exp(-r T) R(T)]

        Parameters
        ----------
        age : float
            The replacement age, more than 0 and finite.
        discount_rate : float, optional
            The continuous discount rate per unit of time, by default 0
            (undiscounted: ``npv`` is then ``inf`` and ``eac`` the
            long-run cost rate).

        Returns
        -------
        AgeReplacementValue
            The net present value, equivalent annual cost and first
            cycle's cost.

        Raises
        ------
        ValueError
            If an argument is out of range or the costs are not set.

        Examples
        --------
        A bearing (Weibull, scale 1000 hours, shape 2.5), replaced at 500
        hours for 1, or at a failure for 5, at 8% a year (in hours):

        >>> import math
        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([1000.0, 2.5]),
        ...     planned_cost=1.0,
        ...     unplanned_cost=5.0,
        ... )
        >>> a_year = math.log(1.08) / 8760
        >>> value = bearing.age_replacement_value(500.0, discount_rate=a_year)
        >>> round(value.npv, 1), round(value.eac * 8760, 2)
        (393.5, 30.28)
        """
        from repyability.maintenance._options import age_replacement_value

        return age_replacement_value(self, age, discount_rate=discount_rate)

    def renewal_forecast(
        self, window, every, *, resets, age=0.0, since_renewal=None
    ):
        """A part renewed at every service and when it fails: the chance
        it fails before each service, the expected failures, renewals and
        cost over a window, and whether the plan is worth it.

        Exact, from the part's age and how far it is into its service
        interval. A plan that does not reset (``resets=False``, block
        replacement) services on fixed times, the first ``every -
        since_renewal`` from now whatever happens; one that resets starts
        each interval at the last renewal, planned or not. It is
        compared with running to failure and with the part's best age
        replacement.

        Parameters
        ----------
        window : float
            The time ahead to forecast, more than 0.
        every : float
            The service interval.
        resets : bool
            Whether each interval starts at the last renewal.
        age : float, optional
            The part's age now, by default 0.
        since_renewal : float, optional
            For a plan that does not reset: how far into the interval it
            is, less than ``every``, by default the age.

        Returns
        -------
        RenewalForecast

        Examples
        --------
        A seal (Weibull, scale 6000 h, shape 2.2) renewed every 500 running
        hours for $200 and at a failure for $900; 340 h old, 340 h into the
        interval, over the next 944 h:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> seal = NonRepairable(
        ...     surv.Weibull.from_params([6000.0, 2.2]),
        ...     planned_cost=200.0,
        ...     unplanned_cost=900.0,
        ... )
        >>> plan = seal.renewal_forecast(
        ...     944.0, 500.0, resets=False, age=340.0
        ... )
        >>> [round(r, 4) for r in plan.risk_before_service]
        [0.0024, 0.0042]
        >>> round(plan.expected_cost, 2), plan.worth_it
        (407.06, False)
        """
        from repyability.maintenance._options import renewal_forecast

        return renewal_forecast(
            self,
            window,
            every,
            resets=resets,
            age=age,
            since_renewal=since_renewal,
        )

    def cost_distribution(
        self,
        window=None,
        every=None,
        *,
        resets=None,
        age=0.0,
        since_renewal=None,
        calendar=None,
        policy=None,
        site=None,
        plan=None,
        now=None,
        readings=None,
        until=None,
        budget=None,
        levels=(10.0, 50.0, 90.0),
        method="exact",
        mc_samples=None,
        seed=None,
    ):
        """What the part's maintenance costs over a window from its state
        now, and how widely that can vary.

        A budget starts from the part as it is, and carries a range, not
        only a figure: this gives the chance of every number of failures
        and of planned replacements in the window, and so of every cost
        (failures at the unplanned cost, planned replacements at the
        planned cost), its mean, standard deviation, percentiles and the
        chance of exceeding a budget. The policy is run to failure (no
        ``every``), age replacement (``every`` with ``resets=True``) or
        block replacement (``resets=False``), as
        [`renewal_forecast`][repyability.NonRepairable.renewal_forecast]
        takes it, or a calendar policy (``calendar``, from
        [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy]:
        at each opportunity, replace the part if it is at least that
        time's threshold age, at that opportunity's price); replacements
        take no time. Numerical by default (to the ``precision``
        reported, about 1e-6), simulated with ``method="simulate"``. How
        it is worked out: the [maintenance guide](../guide/maintenance.md).

        Parameters
        ----------
        window : float
            The window's length from now, more than 0. An event at its
            end falls outside it.
        every : float, optional
            The policy's interval; by default None, run to failure.
        resets : bool, optional
            With ``every``: True for age replacement (each interval starts
            at the last replacement, planned or not), False for block
            replacement (fixed times).
        age : float, optional
            The part's age now, by default 0. Under age replacement a part
            at or past ``every`` is replaced at once.
        since_renewal : float, optional
            For block replacement: how far into the interval it is, less
            than ``every``, by default the age.
        policy : ShutdownPolicy, optional
            On real dates: a site's policy (or give it as ``calendar``).
            The window starts at the policy's ``now``, with the part at the
            policy's ``age`` unless ``age`` is given.
        until : date or str, optional
            With ``policy``: the last day the window runs to, in place of
            ``window``.
        site : optional
            With ``policy``: where given, must be the policy's.
        plan : optional
            With ``policy``: where given, must be the policy's.
        now : optional
            With ``policy``: where given, must be the policy's.
        readings : optional
            With ``policy``: where given, must be the policy's.

        calendar : CalendarPolicy, optional
            Instead of ``every``: a calendar policy worked out for this
            part, its calendar's times counted from now. Failures cost the
            part's ``unplanned_cost``; each planned replacement, the price
            of its opportunity (``kinds`` and ``kind_prices`` in the
            result, and the outcomes by kind).
        budget : float, optional
            A budget, for the chance the cost exceeds it.
        levels : array_like, optional
            The percentiles to report, from 0 to 100, by default 10, 50
            and 90.
        method : str, optional
            ``"exact"`` (the default: numerical, to the precision
            reported) or ``"simulate"``.
        mc_samples : int, optional
            With ``method="simulate"``, the number of simulations, by
            default 10,000.
        seed : int, optional
            With ``method="simulate"``, the random seed.

        Returns
        -------
        CostDistribution

        Raises
        ------
        ValueError
            If an argument is out of range, a cost the policy needs is not
            given, or the part takes time to replace.
        NotImplementedError
            If the life has no survival function in closed form, or, by
            default, for a calendar policy worked out by integration
            (simulate it).

        Examples
        --------
        A bearing (Weibull, scale 365 days, shape 2), 224 days old,
        replaced at every shut (every 180 days, the next in 21) for $100
        and at a failure for $400, over the next 1,086 days:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=100.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> spend = bearing.cost_distribution(
        ...     1086.0, 180.0, resets=False, age=224.0, since_renewal=159.0
        ... )
        >>> round(spend.mean, 2), [float(c) for c in spend.percentiles]
        (1155.04, [600.0, 1000.0, 1800.0])
        >>> round(spend.probability_above(1500.0), 3)
        0.151
        """
        from repyability.maintenance._site_policy import on_dates
        from repyability.maintenance._window_cost import cost_distribution

        calendar, window, age = on_dates(
            calendar,
            policy,
            window,
            until,
            age,
            site=site,
            plan=plan,
            now=now,
            readings=readings,
        )
        return cost_distribution(
            self,
            window,
            every,
            resets=resets,
            age=age,
            since_renewal=since_renewal,
            calendar=calendar,
            budget=budget,
            levels=levels,
            method=method,
            mc_samples=mc_samples,
            seed=seed,
        )

    def cost_by_window(
        self,
        ends=None,
        every=None,
        *,
        width=None,
        count=None,
        resets=None,
        age=0.0,
        since_renewal=None,
        calendar=None,
        policy=None,
        site=None,
        plan=None,
        now=None,
        readings=None,
        budget=None,
        levels=(10.0, 50.0, 90.0),
        method="exact",
        mc_samples=None,
        seed=None,
    ):
        """What the part's maintenance costs in each of a run of
        consecutive windows from its state now, and how often each outcome
        happens in each.

        A budget passed on by period needs each window's own spend, not
        the running total, and what could happen in it. Each window is
        worked out through every earlier one (a failure in one renews the
        part for the next), as
        [`cost_distribution`][repyability.NonRepairable.cost_distribution]
        works out one window from now, with the same policies and
        arguments: each window's chance of every count and cost, its mean,
        spread and percentiles, the chance of each outcome (nothing,
        planned work only, one failure, several failures) and of exceeding
        a budget, and the whole span's own distribution (ranges do not add
        over windows). Give the windows' ``ends`` (uneven windows are
        fine), or a ``width`` and a ``count``; the first starts now.

        Parameters
        ----------
        ends : array_like, optional
            The windows' end times from now, increasing.
        every : float, optional
            The policy's interval; by default None, run to failure.
        width : float, optional
            Instead of ``ends``: the length of each window.
        count : int, optional
            Instead of ``ends``: how many windows of ``width``.
        resets : bool, optional
            As for `cost_distribution`.
        age : float, optional
            The part's age now, by default 0.
        since_renewal : float, optional
            As for `cost_distribution`.
        policy : ShutdownPolicy, optional
            On real dates: a site's policy (or give it as ``calendar``);
            ``ends`` may then be dates, each window's last day. The windows
            start at the policy's ``now``, with the part at the policy's
            ``age`` unless ``age`` is given.
        site : optional
            With ``policy``: where given, must be the policy's.
        plan : optional
            With ``policy``: where given, must be the policy's.
        now : optional
            With ``policy``: where given, must be the policy's.
        readings : optional
            With ``policy``: where given, must be the policy's.

        calendar : CalendarPolicy, optional
            As for `cost_distribution`.
        budget : float, optional
            A budget for each window, for the chance it is exceeded.
        levels : array_like, optional
            The percentiles to report, by default 10, 50 and 90.
        method : str, optional
            ``"exact"`` (the default: numerical) or ``"simulate"``.
        mc_samples : int, optional
            With ``method="simulate"``, the number of simulations.
        seed : int, optional
            With ``method="simulate"``, the random seed.

        Returns
        -------
        WindowCosts

        Raises
        ------
        ValueError, NotImplementedError
            As for `cost_distribution`, or if the windows are not given
            properly.

        Examples
        --------
        The bearing of `cost_distribution`, replaced at every shut, over
        four windows of a year:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable
        >>> bearing = NonRepairable(
        ...     surv.Weibull.from_params([365.0, 2.0]),
        ...     planned_cost=100.0,
        ...     unplanned_cost=400.0,
        ... )
        >>> years = bearing.cost_by_window(
        ...     [263.0, 629.0, 994.0, 1359.0], 180.0, resets=False,
        ...     age=224.0, since_renewal=159.0,
        ... )
        >>> [round(float(m), 1) for m in years.mean]
        [330.0, 382.3, 382.1, 382.2]
        >>> years.outcome_classes
        ('nothing', 'planned only', 'one failure', 'several failures')
        >>> [round(float(p), 3) for p in years.outcome_probabilities[1]]
        [0.0, 0.613, 0.324, 0.063]
        """
        from repyability.maintenance._site_policy import on_dates
        from repyability.maintenance._window_outcomes import cost_by_window

        calendar, ends, age = on_dates(
            calendar,
            policy,
            ends,
            None,
            age,
            site=site,
            plan=plan,
            now=now,
            readings=readings,
            many=True,
        )
        return cost_by_window(
            self,
            ends,
            every,
            width=width,
            count=count,
            resets=resets,
            age=age,
            since_renewal=since_renewal,
            calendar=calendar,
            budget=budget,
            levels=levels,
            method=method,
            mc_samples=mc_samples,
            seed=seed,
        )

    def optimal_opportunity_policy(
        self,
        opportunities,
        *,
        where=None,
        plan=None,
        now=None,
        readings=None,
        age=0.0,
        unit=None,
        method=None,
        horizon=None,
        discount=None,
        failure_timing="half_step",
        max_age=None,
        tail="repeat",
        mc_samples=64,
        seed=None,
        dated_until=None,
        discount_rate=0.0,
    ):
        """The long-run optimal replacement policy of this part over
        opportunities to replace it: stops and shuts at the part's price
        there, a planned replacement on any day at a dearer one (or none),
        and a failure at the dearest.

        **The prices are the part's** (one
        [`Opportunities`][repyability.Opportunities] serves every part): at
        an opportunity, the part's ``opportunity_costs`` for it (by name or
        object; a key naming no opportunity here is refused), else its
        ``opportunity_cost``, else its ``planned_cost`` (an opportunity
        priced as any other day); on any day (where the opportunities
        allow it, ``any_day=True``) its ``planned_cost``; at a failure its
        ``unplanned_cost``. An opportunity shorter than the part's
        ``job_duration``, or (with ``where=``) one that does not take that
        label down, is not used. The policy's ``prices``
        ([`OpportunityPrices`][repyability.OpportunityPrices]) give each
        price and where it came from, and each opportunity not used and
        why. (The deprecated ``Calendar`` and ``Shutdowns``, with prices on
        the opportunities, are taken as before: those prices are the
        part's for those opportunities.)

        **Opportunities on real dates.** Given opportunities on real dates
        (rules, dates, plain periods or uses), it returns a
        [`ShutdownPolicy`][repyability.ShutdownPolicy], worked out by the
        route the shuts allow, which the result names in words
        (``route``, ``route_text``;
        [`Opportunities.route`][repyability.Opportunities.route] says it
        beforehand):

        1. **exact**: the shuts repeat in the part's own clock on a short
           cycle (about a minute's work or less): the exact long-run
           average cost a year;
        2. a **long cycle**: give ``method="exact"`` (slow, said
           beforehand) or ``method="sampling"`` (an estimate with its
           standard error);
        3. shuts that **don't repeat** (rules by month or year, a use on
           another meter): give ``method="averaged"`` (the renewal-reward
           formula, each shut spread evenly within its usual gaps, said on
           the result) or
        4. ``horizon=`` and ``discount=``: the lowest net present cost on
           the real dates to the horizon.

        **Fixed dates** (an ``Opportunity`` with ``dates=``, such as
        turnarounds) are never averaged. Without a horizon the policy is
        ``"dated_then_long_run"``: a dated window from now to the last
        fixed date (or to ``dated_until``), in which every shut, the
        recurring kinds' too, is on its real dates; after it only the
        recurring shuts remain, and their long-run average (by the route
        above that they take on their own, ``method`` applying to them)
        values everything beyond. The rule is given in two phases, and
        ``cost`` is the long run's a year (``dated_cost`` the expected
        cost to the window's end). A kind given ``then_every`` recurs at
        that gap after its last date, joining the recurring shuts.

        Nothing is chosen silently: a route that needs a choice is refused
        with what to choose. A failure costs the part's
        ``unplanned_cost``. ``failure_timing`` is ``"half_step"`` (a grid,
        each failure's replacement counted half a step old) or ``"exact"``
        (quadrature between the shuts, failures at their instant; a
        ``SlowSolverWarning`` where it takes over about 5 s). The guide's
        [shutdown policies](../guide/maintenance.md) has the examples.

        **Opportunities in model time** (periods, or dates as numbers)
        are worked out on the part's model time, returning a
        [`CalendarPolicy`][repyability.CalendarPolicy]:

        Age replacement
        ([`optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy])
        has one planned price at every time, so the state is the age
        alone. Here the price depends on the time, so the best policy is a
        threshold age for each time, ``a*(t)``: replace if the part is at
        least that old. It minimises the long-run average cost a unit of
        time (opportunity-based age replacement, Dekker et al.). The
        methods (this package's use case UC-13):

        - ``"dp"``: dynamic programming on a grid, its step sized from
          the life (about the mean life over 1,000) and put on the
          opportunities' own times. Over a periodic calendar whose cycle
          has at most 20,000 steps, by relative value iteration: the cost
          rate exact for the grid, which is within about the square of
          the step of deciding at any moment. Over a longer one, by a
          long finite horizon, the cost rate differenced.
        - ``"integration"``: between opportunities in continuous time, on
          quadrature nodes; only with no replacement on any day.
        - ``"sampling"``: for a cycle too long to sweep (co-prime
          periods), the cost rate is the mean over ``mc_samples`` sampled
          times of the cycle of a window's differenced rate, with its
          standard error (``cost_rate_se``); each window is an exact
          solve that may take a second or more, so ``mc_samples`` sets
          the cost. The rates by kind (``replacement_rates``,
          ``cost_rates``) and the schedule's chances are then estimates
          from a new unit run in, with no standard error, and move with
          the seed; the threshold at a time is a solve over a window
          from it (deterministic).
        - ``"auto"``: ``"dp"``, or ``"sampling"`` for a periodic calendar
          whose cycle is too long to sweep, with a ``FutureWarning``: from
          0.21 that refuses, naming a ready ``horizon=`` (exact by
          default, simulation on request: give ``horizon=`` or
          ``method="sampling"``).

        **Exact by default; refused where it can't be; a horizon never
        refused.** Without ``horizon=`` the answer is the exact repeating
        cycle's, with or without a discount. Where that can't be worked
        out exactly (a cycle too long to sweep, given dates, shuts that
        drift on real dates, a duty plan's clock), it is refused, and the
        refusal names a ready horizon worked out from the inputs: with a
        discount, the time past which costs count for under one part in a
        billion; with none, the full cycle and three of the part's life
        spans. ``horizon=`` (a model time here, a date on real dates)
        sweeps backward from it over every opportunity as it falls: the
        lowest net present cost to then (``npv``) and its equivalent
        annual cost (``eac``), or with no discount the total to then.

        The grid's step is not an argument: it is a numerical setting,
        sized from the models (never in absolute time), not a choice of
        when decisions are made; a calendar says that, by its
        opportunities. It is a whole fraction of the periods, so a
        periodic calendar keeps its period on the grid whatever its
        offsets; an offset or date off the grid (a few hours on a period
        of days, as dates turned into model time give) is laid at the
        grid time nearest it, with a warning.

        Without a horizon, a calendar with given dates (a plan) is
        extended past its last date by ``tail``: ``"repeat"`` repeats the
        plan (from 0 to its last date and its mean gap), ``"anytime"``
        leaves only replacement
        on any day. Both are solved, and the policy says the first time
        they differ (``determined_until``); its cost rate is the repeated
        plan's.

        Parameters
        ----------
        opportunities : Opportunities
            The opportunities, on real dates or in model time (a list of
            ``Opportunity`` is taken as ``Opportunities`` of them; the
            deprecated ``Calendar`` and ``Shutdowns`` too).
        where : str, optional
            A label (a line, a unit): only the opportunities that take it
            down are used, those with no ``takes_down`` taking the whole
            site down. By default None: every opportunity.
        plan : DutyPlan, optional
            With a site: how the equipment runs, turning dates into the
            part's clock; without, the part's clock is calendar time.
        now : date or str, optional
            With a site: the day answers count from (needed for shuts on
            dates, uses and horizons).
        readings : dict, optional
            With a site: the meters at ``now``, placing shuts by use.
        age : float or str, optional
            With a site: the age of the part in service at ``now``, by
            default 0 (just fitted), for ``next_replacement``.
        unit : str, optional
            With a site: the part's unit (``"days"``, ``"hours"``, or a meter
            of the plan such as ``"km"``), by default its model's label.
        method : str, optional
            With a site: None (exact where the shuts allow), ``"exact"``,
            ``"sampling"`` or ``"averaged"``. With a calendar: ``"auto"``
            (as None), ``"dp"``, ``"integration"`` or ``"sampling"``.
        horizon, discount : optional
            With a site: the last day costs count to, and the yearly
            discount rate as a fraction (0.07 for 7% a year; required with
            a horizon, 0 for the plain total). With opportunities in model
            time, ``horizon`` is a time in their own time (a number), with
            ``discount_rate`` (0 for the plain total), and the policy is
            the lowest net present cost to it (``basis="horizon"``).
            ``discount`` without a horizon discounts the long run: the
            policy minimises the present value of every replacement from
            now on, forever, and
            ``cost`` is its equivalent annual cost (EAC), the steady cost
            a year worth the same, ``ln(1 + discount)`` times the present
            value. A dated window is then discounted to its end, and the
            recurring shuts' discounted long run values everything after
            it. Only the routes that repeat exactly are discounted (the
            exact routes, the dated window, and plain age replacement on
            any day), with no duty plan (whose clock does not keep a steady
            pace with the calendar); the others are refused with what to
            give instead.
        failure_timing : str, optional
            With a site: ``"half_step"`` (the default) or ``"exact"``.
        dated_until : date or str, optional
            With a site: the last day of the dated window, to extend it
            past the last fixed date (a shut schedule planned for the next
            three years); by default the window ends at the last fixed
            date, and with no fixed dates there is none.
        max_age : float, optional
            The oldest age on the grid, by default the age a unit outlives
            with a chance of 1e-9.
        method : str, optional
            ``"auto"`` (the default), ``"dp"``, ``"integration"`` or
            ``"sampling"``.
        tail : str, optional
            How a plan of given dates goes on after its last date:
            ``"repeat"`` (the default) or ``"anytime"``.
        mc_samples : int, optional
            The times sampled by ``"sampling"``, by default 64; the other
            methods take none.
        seed : int, optional
            Seeds the sampled times; the other methods draw nothing.
        discount_rate : float, optional
            With a calendar in model time: the continuous discount rate
            per unit of the calendar's time, by default 0 (the long-run
            average). Given, the thresholds minimise the present value at
            time 0 of every replacement from a new part, forever (Fox,
            1966; discounted value iteration over the cycle), and the
            policy carries that value (``npv``) and ``discount_rate`` times
            it, the equivalent annual cost (``eac``); ``cost_rate`` stays
            the long-run (undiscounted) cost rate of the policy chosen. For
            a yearly rate of 7% with a calendar in days,
            ``math.log(1.07) / 365.25``. A periodic calendar of a short
            cycle only (``"dp"`` or ``"integration"``): a long cycle, a
            plan of given dates and ``"sampling"`` are refused with what to
            give instead (a ready ``horizon=`` among it). A site takes
            ``discount`` (a yearly rate) instead.

        Returns
        -------
        ShutdownPolicy or CalendarPolicy
            With a site, the policy on real dates. With a calendar, the
            long-run cost rate and what it is made of by kind, each
            shut's threshold and chance of a replacement (``schedule()``),
            a unit's next replacement (``next_replacement()``), how it
            was worked out, and the life it was worked out for.

        Raises
        ------
        ValueError
            If a price the calendar takes from the part is not set, the
            calendar allows no planned replacement at all (no
            opportunities, and none on any day), or an argument is out
            of its range.
        NotImplementedError
            For ``method="integration"`` with replacement on any day, or
            ``method="sampling"`` with given dates.

        Examples
        --------
        With no opportunities and a planned replacement on any day at the
        planned cost, it is age replacement:

        >>> import surpyval as surv
        >>> from repyability import NonRepairable, Opportunities
        >>> unit = NonRepairable(
        ...     surv.Weibull.from_params([1000, 2.5]),
        ...     planned_cost=1.0,
        ...     unplanned_cost=5.0,
        ... )
        >>> policy = unit.optimal_opportunity_policy(Opportunities())
        >>> round(policy.cost_rate, 6)
        0.003462
        >>> round(policy.next_replacement(age=0.0).time, 1)
        493.3
        >>> round(unit.optimal_replacement_policy().interval, 1)
        493.0

        One record of stops for two parts, each with its own prices:

        >>> from repyability import Opportunity
        >>> stops = Opportunities([Opportunity("stop", period=28.0)])
        >>> seal = NonRepairable(
        ...     surv.Weibull.from_params([1000, 2.5]),
        ...     planned_cost=4.0, unplanned_cost=20.0, opportunity_cost=3.0,
        ... )
        >>> policy = seal.optimal_opportunity_policy(stops)
        >>> policy.prices.price("stop"), policy.prices.row("stop").source
        (3.0, 'opportunity_cost')
        """
        from repyability.maintenance._opportunities import (
            resolve,
            site_given,
            with_prices,
        )
        from repyability.maintenance._replacement import (
            optimal_calendar_policy,
        )
        from repyability.maintenance._shutdowns import Shutdowns
        from repyability.maintenance._site_policy import find_policy

        calendar, prices = resolve(
            self,
            opportunities,
            where=where,
            on_dates=site_given(
                plan=plan,
                now=now,
                readings=readings,
                unit=unit,
                horizon=None if is_number(horizon) else horizon,
                discount=discount,
                dated_until=dated_until,
                age=age,
                failure_timing=failure_timing,
            ),
        )
        if isinstance(calendar, Shutdowns):
            if max_age is not None or tail != "repeat":
                raise ValueError(
                    "max_age and tail are for a Calendar in model time; a "
                    "site's shuts take neither."
                )
            if discount_rate != 0.0:
                raise ValueError(
                    "discount_rate is for a Calendar in model time (a "
                    "continuous rate a unit of its time); a site takes "
                    "discount=, a yearly rate such as 0.07."
                )
            return with_prices(
                find_policy(
                    self,
                    calendar,
                    plan=plan,
                    now=now,
                    readings=readings,
                    age=age,
                    unit=unit,
                    method=method,
                    horizon=horizon,
                    discount=discount,
                    failure_timing=failure_timing,
                    mc_samples=mc_samples,
                    seed=seed,
                    dated_until=dated_until,
                ),
                prices,
            )
        _model_horizon(horizon)
        _calendar_only(
            plan=plan,
            now=now,
            readings=readings,
            unit=unit,
            discount=discount,
            dated_until=dated_until,
        )
        if age != 0.0:
            raise ValueError(
                "age is for a site's shuts (Shutdowns); a Calendar's policy "
                "gives next_replacement(age, now) instead."
            )
        if failure_timing != "half_step":
            raise ValueError(
                "failure_timing is for a site's shuts (Shutdowns); a Calendar "
                "takes method='dp' or 'integration'."
            )
        return with_prices(
            optimal_calendar_policy(
                self,
                calendar,
                max_age=max_age,
                method="auto" if method is None else method,
                tail=tail,
                mc_samples=mc_samples,
                seed=seed,
                discount_rate=discount_rate,
                horizon=horizon,
            ),
            prices,
        )

    def optimal_calendar_policy(
        self,
        calendar,
        *,
        plan=None,
        now=None,
        readings=None,
        age=0.0,
        unit=None,
        method=None,
        horizon=None,
        discount=None,
        failure_timing="half_step",
        max_age=None,
        tail="repeat",
        mc_samples=64,
        seed=None,
        dated_until=None,
        discount_rate=0.0,
    ):
        """**Deprecated** (removed in 0.21): renamed
        [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy],
        which takes the same arguments and gives the same policy.

        Parameters
        ----------
        calendar : Opportunities, Calendar or Shutdowns
            The opportunities.
        plan : optional
            As for ``optimal_opportunity_policy``.
        now : optional
            As for ``optimal_opportunity_policy``.
        readings : optional
            As for ``optimal_opportunity_policy``.
        age : optional
            As for ``optimal_opportunity_policy``.
        unit : optional
            As for ``optimal_opportunity_policy``.
        method : optional
            As for ``optimal_opportunity_policy``.
        horizon : optional
            As for ``optimal_opportunity_policy``.
        discount : optional
            As for ``optimal_opportunity_policy``.
        failure_timing : optional
            As for ``optimal_opportunity_policy``.
        max_age : optional
            As for ``optimal_opportunity_policy``.
        tail : optional
            As for ``optimal_opportunity_policy``.
        mc_samples : optional
            As for ``optimal_opportunity_policy``.
        seed : optional
            As for ``optimal_opportunity_policy``.
        dated_until : optional
            As for ``optimal_opportunity_policy``.
        discount_rate : optional
            As for ``optimal_opportunity_policy``.

        Returns
        -------
        ShutdownPolicy or CalendarPolicy
            As ``optimal_opportunity_policy`` returns.
        """
        from repyability.utils.deprecation import replaced

        replaced(
            "NonRepairable.optimal_calendar_policy",
            "NonRepairable.optimal_opportunity_policy",
            "the opportunities are one record for every part, on real "
            "dates or in model time",
        )
        return self.optimal_opportunity_policy(
            calendar,
            plan=plan,
            now=now,
            readings=readings,
            age=age,
            unit=unit,
            method=method,
            horizon=horizon,
            discount=discount,
            failure_timing=failure_timing,
            max_age=max_age,
            tail=tail,
            mc_samples=mc_samples,
            seed=seed,
            dated_until=dated_until,
            discount_rate=discount_rate,
        )

    def opportunity_policy_cost(
        self,
        opportunities,
        rule,
        *,
        where=None,
        max_age=None,
        plan=None,
        now=None,
        readings=None,
        unit=None,
        method=None,
        horizon=None,
        discount=None,
        failure_timing="half_step",
        mc_samples=64,
        seed=None,
        dated_until=None,
        discount_rate=0.0,
    ):
        """The long-run average cost a unit of time of a given replacement
        rule over a calendar of opportunities: to price simple policies
        against
        [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].

        The prices are the part's, laid on the opportunities as
        [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy]
        lays them (``where`` as there). On real dates, the rule
        names the kinds of opportunity (and ``"any other day"``), it is
        priced by the same route as the policy would be (``plan``,
        ``now``, ``readings``, ``unit``, ``method``, ``horizon``,
        ``discount``, ``failure_timing``, ``dated_until`` as there), and a
        [`RuleCost`][repyability.RuleCost] comes back: the cost, its
        measure, the route and words, and the replacements by kind. With
        ``discount=`` and no horizon, the rule's discounted long run is
        priced exactly where the shuts repeat (its equivalent annual cost,
        as the policy's own is worked out, with the decisions fixed), and
        refused with a ready horizon where they don't. Given opportunities
        in model time it returns a float: the rate; with
        ``discount_rate``, the equivalent cost a unit of time of the
        rule's discounted long run (a periodic calendar of a short cycle;
        otherwise refused with a ready ``horizon=``); with ``horizon`` (a
        model time), the rule's net present cost to it.

        The rule gives, for each kind of replacement it uses (an
        opportunity's name, or ``"anytime"`` for any day), the age from
        which it replaces the part; the kinds it leaves out never replace.
        Where kinds coincide, the one with the youngest age replaces (the
        cheapest, at a tie). Worked out as the optimal policy's cost rate
        is, on the same grid, with the decisions fixed (a plan of given
        dates repeated past its last date).

        Parameters
        ----------
        opportunities : Opportunities
            The opportunities.
        rule : dict
            The age, at least 0 (``inf`` for never), from which each kind
            replaces: ``{"annual": 0.0}`` replaces at every annual shut,
            ``{"anytime": 493.0}`` at age 493 on any day, ``{}`` runs to
            failure.
        where : str, optional
            As for ``optimal_opportunity_policy``.
        max_age : float, optional
            The oldest age on the grid, as for ``optimal_calendar_policy``.
        plan : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        now : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        readings : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        unit : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        method : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        horizon : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        discount : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        failure_timing : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        mc_samples : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        seed : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        dated_until : optional
            With a site: as for
            [`optimal_opportunity_policy`][repyability.NonRepairable.optimal_opportunity_policy].
        discount_rate : float, optional
            With opportunities in model time: the continuous discount rate
            a unit of their time, by default 0.

        Returns
        -------
        float or RuleCost
            In model time the rule's long-run cost a unit of time (its
            EAC with ``discount_rate``, its net present cost with
            ``horizon``); on real dates a ``RuleCost``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairable, Opportunities
        >>> unit = NonRepairable(
        ...     surv.Weibull.from_params([1000, 2.5]),
        ...     planned_cost=1.0,
        ...     unplanned_cost=5.0,
        ... )
        >>> rate = unit.opportunity_policy_cost(
        ...     Opportunities(), {"anytime": 493.0}
        ... )
        >>> round(rate, 6)
        0.003462
        """
        from repyability.maintenance._opportunities import (
            resolve,
            site_given,
            with_prices,
        )
        from repyability.maintenance._replacement import (
            calendar_policy_cost,
        )
        from repyability.maintenance._shutdowns import Shutdowns
        from repyability.maintenance._site_policy import price_rule

        calendar, prices = resolve(
            self,
            opportunities,
            where=where,
            on_dates=site_given(
                plan=plan,
                now=now,
                readings=readings,
                unit=unit,
                horizon=None if is_number(horizon) else horizon,
                discount=discount,
                dated_until=dated_until,
                failure_timing=failure_timing,
                method=method,
            ),
        )
        if isinstance(calendar, Shutdowns):
            if max_age is not None:
                raise ValueError("max_age is for a Calendar in model time.")
            if discount_rate != 0.0:
                raise ValueError(
                    "discount_rate is for opportunities in model time (a "
                    "continuous rate a unit of their time); a site takes "
                    "discount=, a yearly rate such as 0.07."
                )
            return with_prices(
                price_rule(
                    self,
                    calendar,
                    rule,
                    plan=plan,
                    now=now,
                    readings=readings,
                    unit=unit,
                    method=method,
                    horizon=horizon,
                    discount=discount,
                    failure_timing=failure_timing,
                    mc_samples=mc_samples,
                    seed=seed,
                    dated_until=dated_until,
                ),
                prices,
            )
        _model_horizon(horizon)
        _calendar_only(
            plan=plan,
            now=now,
            readings=readings,
            unit=unit,
            discount=discount,
            method=method,
            dated_until=dated_until,
        )
        if failure_timing != "half_step":
            raise ValueError(
                "failure_timing is for a site's shuts (Shutdowns)."
            )
        return calendar_policy_cost(
            self,
            calendar,
            rule,
            max_age=max_age,
            discount_rate=discount_rate,
            horizon=horizon,
        )

    def calendar_policy_cost(
        self,
        calendar,
        rule,
        *,
        max_age=None,
        plan=None,
        now=None,
        readings=None,
        unit=None,
        method=None,
        horizon=None,
        discount=None,
        failure_timing="half_step",
        mc_samples=64,
        seed=None,
        dated_until=None,
        discount_rate=0.0,
    ):
        """**Deprecated** (removed in 0.21): renamed
        [`opportunity_policy_cost`][repyability.NonRepairable.opportunity_policy_cost],
        which takes the same arguments and gives the same cost.

        Parameters
        ----------
        calendar : Opportunities, Calendar or Shutdowns
            The opportunities.
        rule : dict
            As for ``opportunity_policy_cost``.
        max_age : optional
            As for ``opportunity_policy_cost``.
        plan : optional
            As for ``opportunity_policy_cost``.
        now : optional
            As for ``opportunity_policy_cost``.
        readings : optional
            As for ``opportunity_policy_cost``.
        unit : optional
            As for ``opportunity_policy_cost``.
        method : optional
            As for ``opportunity_policy_cost``.
        horizon : optional
            As for ``opportunity_policy_cost``.
        discount : optional
            As for ``opportunity_policy_cost``.
        failure_timing : optional
            As for ``opportunity_policy_cost``.
        mc_samples : optional
            As for ``opportunity_policy_cost``.
        seed : optional
            As for ``opportunity_policy_cost``.
        dated_until : optional
            As for ``opportunity_policy_cost``.
        discount_rate : optional
            As for ``opportunity_policy_cost``.

        Returns
        -------
        float or RuleCost
            As ``opportunity_policy_cost`` returns.
        """
        from repyability.utils.deprecation import replaced

        replaced(
            "NonRepairable.calendar_policy_cost",
            "NonRepairable.opportunity_policy_cost",
            "the opportunities are one record for every part, on real "
            "dates or in model time",
        )
        return self.opportunity_policy_cost(
            calendar,
            rule,
            max_age=max_age,
            plan=plan,
            now=now,
            readings=readings,
            unit=unit,
            method=method,
            horizon=horizon,
            discount=discount,
            failure_timing=failure_timing,
            mc_samples=mc_samples,
            seed=seed,
            dated_until=dated_until,
            discount_rate=discount_rate,
        )

    @staticmethod
    def opportunity_prices(
        opportunities,
        *,
        replacement_cost,
        replacement_duration,
        failure_cost,
        failure_duration,
        downtime_cost,
    ):
        """A part's prices worked out from what a replacement costs and how
        long the plant is down for it: one row per opportunity, then any
        other day and a failure, each with its price and why.

        - At an opportunity the job fits in: ``replacement_cost`` (the
          plant is down anyway).
        - On any other day: ``replacement_cost + downtime_cost *
          replacement_duration``.
        - At a failure: ``failure_cost + downtime_cost *
          failure_duration``.
        - At an opportunity shorter than ``replacement_duration``: not
          used.

        ``downtime_cost`` is the system's (what an hour of the plant down
        costs), so it is not kept on the part: the result's ``apply(part)``
        gives the part the prices it works out (``planned_cost``,
        ``unplanned_cost``, ``opportunity_cost`` and ``job_duration``).

        Parameters
        ----------
        opportunities : Opportunities
            The opportunities.
        replacement_cost : float
            What the replacement itself costs (parts and labour).
        replacement_duration : float or str
            How long it takes: on real dates text (``"6h"``) or hours, the
            downtime cost a unit of it being an hour's; in model time a
            number in the life's unit.
        failure_cost : float
            What a replacement at a failure costs itself.
        failure_duration : float or str
            How long the plant is down at a failure, as
            ``replacement_duration``.
        downtime_cost : float
            What the plant down costs a unit of duration.

        Returns
        -------
        OpportunityPrices
            The rows, and the part's prices to ``apply``.

        Examples
        --------
        >>> from repyability import NonRepairable, Opportunities, Opportunity
        >>> from repyability.schedule import Recurrence
        >>> site = Opportunities([
        ...     Opportunity("weekly stop", Recurrence.weekly("Sat"),
        ...                 duration="8h"),
        ...     Opportunity("shift change", Recurrence.weekly("Mon"),
        ...                 duration="4h"),
        ... ])
        >>> prices = NonRepairable.opportunity_prices(
        ...     site, replacement_cost=1_200, replacement_duration="6h",
        ...     failure_cost=4_200, failure_duration="18h",
        ...     downtime_cost=2_500,
        ... )
        >>> print(prices.describe())  # doctest: +NORMALIZE_WHITESPACE
        weekly stop: $1,200 (replacement_cost alone: the plant is down
        anyway)
        shift change: not used (the 6 h job doesn't fit in the 4 h shift
        change)
        any day: $16,200 (1200 + 2500 x 6 h)
        failure: $49,200 (4200 + 2500 x 18 h)
        """
        from repyability.maintenance._opportunities import (
            opportunity_prices,
        )

        return opportunity_prices(
            opportunities,
            replacement_cost=replacement_cost,
            replacement_duration=replacement_duration,
            failure_cost=failure_cost,
            failure_duration=failure_duration,
            downtime_cost=downtime_cost,
        )

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


def _model_horizon(horizon) -> None:
    """Refuse a horizon on real dates given with opportunities in model
    time, which take one in their own time."""
    if horizon is not None and not is_number(horizon):
        raise ValueError(
            "These opportunities are in model time, so a horizon is a time "
            f"in their own time (a number, such as 36500); got {horizon!r}. "
            "A date is for opportunities on real dates."
        )


def _calendar_only(**given) -> None:
    """Refuse a site's options given with a ``Calendar`` in model time."""
    named = sorted(k for k, v in given.items() if v is not None)
    if named:
        raise ValueError(
            f"{', '.join(named)} {'is' if len(named) == 1 else 'are'} for a "
            "site's shuts on real dates (Shutdowns); a Calendar is in model "
            "time and takes none."
        )
