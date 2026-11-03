"""The numbers behind each maintenance decision for a failure mode: run to
failure, scheduled replacement, failure finding (testing a hidden
failure) and on-condition maintenance.

Each is a small calculator in model time, in the life's time unit, from
the part's life and the prices of what is done: it gives the numbers that
compare the decisions (a cost rate, the failures per unit time, the
chance of a failure first) and, where there is one, the best interval.
Records, workflow and dates are the caller's.

**Run to failure.** The part is replaced when it fails: a failure every
``MTTF`` plus the replacement time, at the unplanned cost.

**Scheduled replacement** at age ``T`` (age replacement): one cycle is the
part's life cut off at ``T``, of mean ``M(T) = integral of R`` to ``T``,
so the failures per unit time are ``(1 - R(T)) / M(T)``, the planned
replacements ``R(T) / M(T)``, and the cost rate ``(planned R(T) +
unplanned (1 - R(T))) / M(T)``.

**Failure finding.** A protective device fails without notice and the
failure is found only at a test, every ``tau``, when it is repaired as
new; a device that has not failed carries its age through the test. A
cycle from new ends at the test that finds the failure, the ``j``-th with
probability ``R((j - 1) tau) - R(j tau)``, so the mean cycle is ``tau
sum_{j>=0} R(j tau)``, and the mean unavailability (the average chance
the device is failed when a demand comes) is::

    U = 1 - MTTF / (tau sum_{j>=0} R(j tau)),

which for an exponential life is ``1 - (1 - exp(-lambda tau)) / (lambda
tau)``, close to ``lambda tau / 2``. Demands come at a rate ``d``, so the
multiple failures (a demand met by a failed device) come at ``d U``.

**On condition.** A failure is preceded by a detectable warning (a
potential failure) a time ``X`` before it (the P-F interval). Inspections
every ``tau`` find the warning with probability ``p`` each time. With a
random phase an interval ``x`` holds ``floor(x / tau)`` inspections, and
one more with the chance of its fractional part, so a warning of length
``x`` is missed with probability ``q^floor(x/tau) (1 - p frac(x/tau))``,
``q = 1 - p``; the chance it is found is one less its mean over ``X``.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: Terms of a failure-finding sum, and the survival at which it ends.
_MAX_TESTS = 20_000_000
_TAIL = 1e-17


def _number(value, name: str, low: float = 0.0, strict: bool = False) -> float:
    ok = (
        checks.is_number(value)
        and math.isfinite(float(value))
        and (float(value) > low if strict else float(value) >= low)
    )
    if not ok:
        bound = "more than" if strict else "at least"
        raise ValueError(
            f"{name} must be a number, {bound} {low:g}; got {value!r}."
        )
    return float(value)


def _survival_at(model, times) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return np.asarray(model.sf(times), dtype=float).reshape(times.shape)


def _mean_life(model) -> float:
    from repyability.rbd._model_utils import model_mean

    return model_mean(model)


def _replacement_time(part) -> float:
    ttr = getattr(part, "time_to_replace", None)
    if ttr is None or isinstance(ttr, str):
        return 0.0
    from repyability.rbd._model_utils import model_mean

    return model_mean(ttr)


def _life_integral(part, until: float) -> float:
    """``M(until)``: the integral of the survival function to ``until``."""
    from repyability.rbd._mean_lifetime import _integrate, model_knots

    knots = model_knots(part.reliability)
    inside = knots[(knots > 0.0) & (knots < until)]
    return float(
        _integrate(
            lambda x: _survival_at(part.reliability, x).ravel(),
            np.concatenate(([0.0], inside, [until])),
        )
    )


# -- run to failure ----------------------------------------------------------


@dataclass
class RunToFailure(_ResultMapping):
    """Running a part to failure: returned by
    [`run_to_failure`][repyability.run_to_failure].

    Attributes
    ----------
    mean_life : float
        The part's mean life.
    mean_replacement_time : float
        How long a replacement takes (0 for instant).
    failure_rate : float
        Failures per unit time in the long run, ``1 / (mean life + mean
        replacement time)``.
    availability : float
        The long-run share of time the part is working.
    cost_rate : float or None
        The long-run cost a unit of time (the unplanned cost times the
        failure rate); None without an unplanned cost.
    """

    mean_life: float
    mean_replacement_time: float
    failure_rate: float
    availability: float
    cost_rate: Optional[float]


def run_to_failure(part) -> RunToFailure:
    """The long-run numbers for replacing a part only when it fails.

    Parameters
    ----------
    part : NonRepairable
        The part: its life, the time a replacement takes
        (``time_to_replace``) and, for a cost, its ``unplanned_cost``.

    Returns
    -------
    RunToFailure

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import NonRepairable, run_to_failure
    >>> pump = NonRepairable(
    ...     surv.Exponential.from_params([0.002]), unplanned_cost=900.0
    ... )
    >>> numbers = run_to_failure(pump)
    >>> numbers.mean_life, numbers.cost_rate
    (500.0, 1.8)
    """
    mean = _mean_life(part.reliability)
    ttr = _replacement_time(part)
    cycle = mean + ttr
    cost = getattr(part, "cu", None)
    return RunToFailure(
        mean,
        ttr,
        1.0 / cycle,
        mean / cycle,
        None if cost is None else cost / cycle,
    )


# -- scheduled replacement ---------------------------------------------------


@dataclass
class ScheduledReplacement(_ResultMapping):
    """Replacing a part at a fixed age: returned by
    [`scheduled_replacement`][repyability.scheduled_replacement].

    Attributes
    ----------
    interval : float
        The replacement age.
    optimal : bool
        Whether it was chosen as the cheapest.
    failure_probability : float
        The chance the part fails before it is replaced, ``1 - R(T)``.
    mean_cycle : float
        The mean time between replacements, planned or not.
    failure_rate : float
        Failures and planned replacements per unit time.
    replacement_rate : float
        The companion of ``failure_rate`` above.
    cost_rate : float
        The long-run cost a unit of time.
    run_to_failure_cost_rate : float
        The cost rate with no scheduled replacement.
    saving_rate : float
        ``run_to_failure_cost_rate - cost_rate``: positive, the schedule
        pays.
    worth_it : bool
        Whether the schedule costs less than running to failure.
    """

    interval: float
    optimal: bool
    failure_probability: float
    mean_cycle: float
    failure_rate: float
    replacement_rate: float
    cost_rate: float
    run_to_failure_cost_rate: float
    saving_rate: float
    worth_it: bool


def scheduled_replacement(part, interval=None) -> ScheduledReplacement:
    """The numbers for replacing a part at a fixed age, planned and
    unplanned, against running it to failure.

    In plain terms: replace the part when it reaches age ``interval``, or
    sooner if it fails. How often does it fail, how often is it replaced
    on schedule, what does it cost a unit of time, and is that less than
    waiting for failures? Without an ``interval`` the cheapest age is used
    (see
    [`optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy]).
    Exact. Replacements take no time.

    Parameters
    ----------
    part : NonRepairable
        The part, with planned and unplanned costs set.
    interval : float, optional
        The replacement age, more than 0. By default the cheapest.

    Returns
    -------
    ScheduledReplacement

    Raises
    ------
    ValueError
        If the costs are not set, the interval is out of range, or
        replacing never pays and no interval is given (``interval`` is
        then ``inf`` in the result of ``optimal_replacement_policy``).

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import NonRepairable, scheduled_replacement
    >>> bearing = NonRepairable(
    ...     surv.Weibull.from_params([365.0, 2.0]),
    ...     planned_cost=150.0,
    ...     unplanned_cost=400.0,
    ... )
    >>> plan = scheduled_replacement(bearing, 250.0)
    >>> round(plan.failure_probability, 4), round(plan.cost_rate, 4)
    (0.3745, 1.1286)
    >>> best = scheduled_replacement(bearing)
    >>> round(best.interval), best.optimal, best.worth_it
    (298, True, True)
    """
    part._require_costs()
    optimal = interval is None
    if optimal:
        policy = part.optimal_replacement_policy()
        if not math.isfinite(policy.interval):
            raise ValueError(
                "Replacing this part on a schedule never pays (its cost "
                "rate is lowest running it to failure): give an interval "
                "to cost one anyway."
            )
        interval = policy.interval
    interval = _number(interval, "interval", strict=True)
    survival = float(_survival_at(part.reliability, interval))
    cycle = _life_integral(part, interval)
    failures = 1.0 - survival
    cost = (part.cp * survival + part.cu * failures) / cycle
    rtf = float(part._run_to_failure_rate())
    return ScheduledReplacement(
        interval,
        optimal,
        failures,
        cycle,
        failures / cycle,
        survival / cycle,
        cost,
        rtf,
        rtf - cost,
        bool(cost < rtf),
    )


# -- failure finding ---------------------------------------------------------


@dataclass
class FailureFinding(_ResultMapping):
    """Testing a hidden failure: returned by
    [`failure_finding`][repyability.failure_finding] and
    [`optimal_failure_finding_interval`][repyability.optimal_failure_finding_interval].

    Attributes
    ----------
    interval : float
        The time between tests.
    basis : str
        ``"given"``, ``"target"`` (the longest interval meeting the
        target) or ``"cost"`` (the cheapest).
    mean_unavailability : float
        The average chance the device is failed when it is needed.
    mean_time_to_find : float
        The mean time a failure stays hidden before a test finds it.
    failures_found_rate : float
        Failures found per unit time.
    tests_rate : float
        Tests per unit time.
    multiple_failure_rate : float or None
        Demands met by a failed device per unit time (the demand rate
        times the mean unavailability); None without a demand rate.
    cost_rate : float or None
        Tests, repairs and the consequence of multiple failures a unit of
        time; None without prices.
    """

    interval: float
    basis: str
    mean_unavailability: float
    mean_time_to_find: float
    failures_found_rate: float
    tests_rate: float
    multiple_failure_rate: Optional[float]
    cost_rate: Optional[float]


def _unavailability(part, tau: float) -> tuple:
    """``(U, mean cycle)`` of a device tested every ``tau``: see the
    module's notes."""
    from repyability.rbd._common import _constant_rate

    mean = _mean_life(part.reliability)
    rate = _constant_rate(part.reliability)
    if rate is not None:
        decay = -math.expm1(-rate * tau)
        return 1.0 - decay / (rate * tau), tau / decay
    chunk, total, j = 4096, 0.0, 0
    while True:
        times = (j + np.arange(chunk)) * tau
        values = _survival_at(part.reliability, times)
        total += float(values.sum())
        j += chunk
        if values[-1] < _TAIL * max(total, 1.0):
            break
        if j >= _MAX_TESTS:
            raise NotImplementedError(
                "The tests are so much shorter than the device's life "
                f"({tau:g} against a mean of {mean:g}) that the sum over "
                "them is too long: use a longer interval."
            )
        chunk = min(2 * chunk, 4_000_000)
    cycle = tau * total
    return 1.0 - mean / cycle, cycle


def _check_protective(part) -> None:
    if part.model_parameterization not in ("parametric", "degradation"):
        raise NotImplementedError(
            "failure finding needs a life with a survival function in "
            "closed form (a surpyval parametric distribution or mixture); "
            f"this part's is {part.model_parameterization}."
        )


def _prices(test_cost, repair_cost, consequence_cost, demand_rate):
    test = _number(test_cost, "test_cost")
    repair = _number(repair_cost, "repair_cost")
    consequence = _number(consequence_cost, "consequence_cost")
    demand = (
        None if demand_rate is None else _number(demand_rate, "demand_rate")
    )
    if consequence > 0.0 and demand is None:
        raise ValueError(
            "A consequence_cost needs the demand_rate: how often the "
            "device is needed."
        )
    return test, repair, consequence, demand


def _finding(part, tau, basis, test, repair, consequence, demand):
    unavailability, cycle = _unavailability(part, tau)
    mean = _mean_life(part.reliability)
    found = 1.0 / cycle
    multiple = None if demand is None else demand * unavailability
    cost = test / tau + repair * found
    if multiple is not None:
        cost += consequence * multiple
    priced = test > 0.0 or repair > 0.0 or consequence > 0.0
    return FailureFinding(
        tau,
        basis,
        unavailability,
        cycle - mean,
        found,
        1.0 / tau,
        multiple,
        cost if priced else None,
    )


def failure_finding(
    protective,
    test_interval,
    *,
    test_cost=0.0,
    repair_cost=0.0,
    demand_rate=None,
    consequence_cost=0.0,
) -> FailureFinding:
    """The numbers for testing a hidden failure every ``test_interval``:
    how often the device is found failed when it is needed, and what the
    testing costs.

    In plain terms: a protective device (a relief valve, a trip) fails
    without anyone knowing, and the failure shows only when it is tested.
    If the protected event comes at random, how likely is it to find the
    device failed, how often does that happen, and what does it cost to
    test? A test finds and repairs a failed device as new; one that has
    not failed carries on at its age. Tests take no time. Exact (see the
    module's notes).

    Parameters
    ----------
    protective : NonRepairable
        The device: its life (a failure is hidden) as a part. Its own
        costs are not used.
    test_interval : float
        The time between tests, more than 0.
    test_cost : float, optional
        What a test costs, by default 0.
    repair_cost : float, optional
        What repairing a failure found costs, by default 0.
    demand_rate : float, optional
        How often the protected event comes, per unit time.
    consequence_cost : float, optional
        What a demand met by a failed device costs (needs
        ``demand_rate``), by default 0.

    Returns
    -------
    FailureFinding

    Examples
    --------
    A relief valve failing hidden at 0.001 a year, tested every 2 years, the
    protected event coming once in 50 years:

    >>> import surpyval as surv
    >>> from repyability import NonRepairable, failure_finding
    >>> valve = NonRepairable(surv.Exponential.from_params([0.001]))
    >>> result = failure_finding(valve, 2.0, demand_rate=0.02)
    >>> round(result.mean_unavailability, 6)
    0.000999
    >>> round(result.multiple_failure_rate * 1e5, 3)
    1.999
    """
    _check_protective(protective)
    tau = _number(test_interval, "test_interval", strict=True)
    prices = _prices(test_cost, repair_cost, consequence_cost, demand_rate)
    return _finding(protective, tau, "given", *prices)


def optimal_failure_finding_interval(
    protective,
    *,
    target_unavailability=None,
    test_cost=0.0,
    repair_cost=0.0,
    demand_rate=None,
    consequence_cost=0.0,
) -> FailureFinding:
    """The failure-finding interval: the longest that keeps the mean
    unavailability within a target, or else the cheapest.

    With a ``target_unavailability`` it is the longest interval whose mean
    unavailability does not exceed it (for an exponential life about
    ``2 * target / failure rate``). Without one it minimises the cost rate
    of tests, repairs and the consequence of multiple failures, which
    needs a ``test_cost``, a ``demand_rate`` and a ``consequence_cost``.

    Parameters
    ----------
    protective : NonRepairable
        The device, as for
        [`failure_finding`][repyability.failure_finding].
    target_unavailability : float, optional
        The mean unavailability not to exceed, between 0 and 1.
    test_cost : float, optional
        As for [`failure_finding`][repyability.failure_finding].
    repair_cost : float, optional
        The companion of ``test_cost`` above.
    demand_rate : float, optional
        The companion of ``test_cost`` above.
    consequence_cost : float, optional
        The companion of ``test_cost`` above.

    Returns
    -------
    FailureFinding
        At the interval chosen, with ``basis`` ``"target"`` or ``"cost"``.

    Raises
    ------
    ValueError
        If neither a target nor all the prices a cost needs are given, or
        the target is out of range.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import NonRepairable, optimal_failure_finding_interval
    >>> valve = NonRepairable(surv.Exponential.from_params([0.001]))
    >>> result = optimal_failure_finding_interval(
    ...     valve, target_unavailability=0.001
    ... )
    >>> round(result.interval, 1), result.basis
    (2.0, 'target')
    """
    _check_protective(protective)
    prices = _prices(test_cost, repair_cost, consequence_cost, demand_rate)
    test, _, consequence, demand = prices
    scale = _mean_life(protective.reliability)
    if target_unavailability is not None:
        target = _number(
            target_unavailability, "target_unavailability", strict=True
        )
        if target >= 1.0:
            raise ValueError(
                "target_unavailability is a probability, less than 1; got "
                f"{target_unavailability!r}."
            )
        high = 50.0 * scale

        def excess(log_tau: float) -> float:
            return _unavailability(protective, math.exp(log_tau))[0] - target

        low = math.log(1e-8 * scale)
        if excess(math.log(high)) <= 0.0:
            raise ValueError(
                "The device stays within the target at any test interval "
                "up to 50 mean lives: it needs no testing for it."
            )
        tau = math.exp(brentq(excess, low, math.log(high), xtol=1e-13))
        return _finding(protective, tau, "target", *prices)
    if not (test > 0.0 and consequence > 0.0 and demand):
        raise ValueError(
            "Give a target_unavailability, or the test_cost, "
            "demand_rate and consequence_cost the cheapest interval is "
            "worked out from."
        )

    def rate(log_tau: float) -> float:
        return _finding(
            protective, math.exp(log_tau), "cost", *prices
        ).cost_rate  # type: ignore[return-value]

    grid = np.linspace(math.log(1e-4 * scale), math.log(20.0 * scale), 200)
    values = np.array([rate(g) for g in grid])
    i = int(np.argmin(values))
    low, high = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    refined = minimize_scalar(
        rate, bounds=(low, high), method="bounded", options={"xatol": 1e-12}
    )
    best = refined.x if refined.fun <= values[i] else grid[i]
    return _finding(protective, math.exp(best), "cost", *prices)


# -- on condition ------------------------------------------------------------


@dataclass
class OnCondition(_ResultMapping):
    """Inspecting for a warning of failure: returned by
    [`on_condition`][repyability.on_condition] and
    [`optimal_on_condition_interval`][repyability.optimal_on_condition_interval].

    Attributes
    ----------
    interval : float
        The time between inspections.
    basis : str
        ``"given"`` or ``"cost"``.
    detection_probability : float
        The chance a warning is found before it becomes a failure.
    failure_rate : float
        Failures (warnings missed) per unit time.
    found_rate : float
        Warnings found, and acted on, per unit time.
    inspection_rate : float
        Inspections per unit time.
    cost_rate : float or None
        Inspections, work on warnings found and failures a unit of time;
        None without prices.
    """

    interval: float
    basis: str
    detection_probability: float
    failure_rate: float
    found_rate: float
    inspection_rate: float
    cost_rate: Optional[float]


def _warning_model(warning):
    """``(sf, mean)`` of the warning time: a number is a fixed warning."""
    if checks.is_number(warning):
        x = _number(warning, "pf_interval", strict=True)
        return (lambda t: np.where(np.asarray(t) < x, 1.0, 0.0)), x, x
    if not hasattr(warning, "sf"):
        raise TypeError(
            "pf_interval must be a time (a fixed warning) or a surpyval "
            f"model of the warning time; got {warning!r}."
        )
    return (
        lambda t: _survival_at(warning, t),
        _mean_life(warning),
        None,
    )


def _detection(warning, tau: float, p: float) -> float:
    """The chance an inspection every ``tau``, finding a warning with
    probability ``p`` each time, finds it before the failure."""
    q = 1.0 - p
    sf, mean, fixed = _warning_model(warning)
    if fixed is not None:
        whole = math.floor(fixed / tau)
        fraction = fixed / tau - whole
        miss = q**whole * (1.0 - p * fraction)
        return 1.0 - miss
    # Miss probability: sum over the inspection intervals [k tau,
    # (k + 1) tau) of q^k (P(X in it) - p E[(X - k tau) / tau; X in it]),
    # with E[(X - a); X in [a, b)] = tau G(b) - integral of G over it.
    miss, k = 0.0, 0
    while True:
        a, b = k * tau, (k + 1) * tau
        sa, sb = float(sf(a)), float(sf(b))
        survival_integral = quad(
            lambda x: float(sf(x)), a, b, epsabs=0.0, epsrel=1e-12
        )[0]
        g_integral = tau - survival_integral
        first_moment = (tau * (1.0 - sb) - g_integral) / tau
        miss += q**k * ((sa - sb) - p * first_moment)
        k += 1
        if sb < 1e-16 or q**k < 1e-16 or k > 200_000:
            break
    return 1.0 - miss


def _on_condition(
    warning, tau, basis, p, onset, inspection, found, failed
) -> OnCondition:
    detect = _detection(warning, tau, p)
    priced = inspection > 0.0 or found > 0.0 or failed > 0.0
    cost = inspection / tau + onset * (detect * found + (1 - detect) * failed)
    return OnCondition(
        tau,
        basis,
        detect,
        onset * (1.0 - detect),
        onset * detect,
        1.0 / tau,
        cost if priced else None,
    )


def _on_condition_inputs(
    detection_probability,
    onset_rate,
    inspection_cost,
    found_cost,
    failure_cost,
):
    p = _number(detection_probability, "detection_probability")
    if not 0.0 < p <= 1.0:
        raise ValueError(
            "detection_probability is a probability, more than 0 and at "
            f"most 1; got {detection_probability!r}."
        )
    onset = _number(onset_rate, "onset_rate", strict=True)
    return (
        p,
        onset,
        _number(inspection_cost, "inspection_cost"),
        _number(found_cost, "found_cost"),
        _number(failure_cost, "failure_cost"),
    )


def on_condition(
    pf_interval,
    inspection_interval,
    *,
    onset_rate,
    detection_probability=1.0,
    inspection_cost=0.0,
    found_cost=0.0,
    failure_cost=0.0,
) -> OnCondition:
    """The numbers for inspecting every ``inspection_interval`` for a
    warning that precedes a failure.

    In plain terms: a part shows a detectable sign of trouble some time
    (the P-F interval) before it fails. If you inspect every so often,
    what share of warnings do you catch in time, how many failures are
    left, and what does it cost? The inspections' phase against the
    warnings is random. Exact (see the module's notes).

    Parameters
    ----------
    pf_interval : float or surpyval model
        The time from a warning to the failure: a number for a fixed
        warning, or a fitted model of it.
    inspection_interval : float
        The time between inspections, more than 0.
    onset_rate : float
        Warnings per unit time, in the long run (for a part replaced at
        each, one over its mean life).
    detection_probability : float, optional
        The chance an inspection finds a warning that is there, by
        default 1.
    inspection_cost : float, optional
        What an inspection costs; acting on a warning found (replacing
        the part in time); and a failure the warning ended in. By default
        0.
    found_cost : float, optional
        The companion of ``inspection_cost`` above.
    failure_cost : float, optional
        The companion of ``inspection_cost`` above.

    Returns
    -------
    OnCondition

    Examples
    --------
    A warning always 30 days before failure, inspected every 20 days, the
    part failing once in 400 days:

    >>> from repyability import on_condition
    >>> result = on_condition(30.0, 20.0, onset_rate=1 / 400)
    >>> result.detection_probability
    1.0
    >>> on_condition(30.0, 60.0, onset_rate=1 / 400).detection_probability
    0.5
    """
    tau = _number(inspection_interval, "inspection_interval", strict=True)
    inputs = _on_condition_inputs(
        detection_probability,
        onset_rate,
        inspection_cost,
        found_cost,
        failure_cost,
    )
    return _on_condition(pf_interval, tau, "given", *inputs)


def optimal_on_condition_interval(
    pf_interval,
    *,
    onset_rate,
    inspection_cost,
    found_cost,
    failure_cost,
    detection_probability=1.0,
) -> OnCondition:
    """The cheapest interval between inspections for a warning.

    Parameters
    ----------
    pf_interval : float
        As for [`on_condition`][repyability.on_condition].
    onset_rate : float
        As for [`on_condition`][repyability.on_condition].
    detection_probability : float, optional
        As for [`on_condition`][repyability.on_condition].
    inspection_cost : float
        The prices, as for [`on_condition`][repyability.on_condition]; the
        inspection and the failure must cost something, and a failure more
        than acting on a warning.
    found_cost : float
        The companion of ``inspection_cost`` above.
    failure_cost : float
        The companion of ``inspection_cost`` above.

    Returns
    -------
    OnCondition
        At the cheapest interval (``basis="cost"``); ``interval`` is
        ``inf`` if inspecting never pays.

    Examples
    --------
    >>> from repyability import optimal_on_condition_interval
    >>> best = optimal_on_condition_interval(
    ...     30.0, onset_rate=1 / 400, inspection_cost=20.0,
    ...     found_cost=300.0, failure_cost=5000.0,
    ... )
    >>> round(best.interval, 1), round(best.detection_probability, 6)
    (30.0, 1.0)
    """
    inputs = _on_condition_inputs(
        detection_probability,
        onset_rate,
        inspection_cost,
        found_cost,
        failure_cost,
    )
    _, onset, inspection, found, failed = inputs
    if not (inspection > 0.0 and failed > found):
        raise ValueError(
            "The cheapest interval needs an inspection_cost, and a "
            "failure_cost above the found_cost."
        )
    _, mean, fixed = _warning_model(pf_interval)
    scale = mean

    def rate(log_tau: float) -> float:
        return _on_condition(
            pf_interval, math.exp(log_tau), "cost", *inputs
        ).cost_rate  # type: ignore[return-value]

    grid = np.linspace(math.log(1e-3 * scale), math.log(30.0 * scale), 240)
    values = np.array([rate(g) for g in grid])
    i = int(np.argmin(values))
    if i == len(grid) - 1:
        return OnCondition(
            math.inf, "cost", 0.0, onset, 0.0, 0.0, onset * failed
        )
    low, high = grid[max(i - 1, 0)], grid[i + 1]
    refined = minimize_scalar(
        rate, bounds=(low, high), method="bounded", options={"xatol": 1e-12}
    )
    best = refined.x if refined.fun <= values[i] else grid[i]
    # A fixed warning makes the rate piecewise smooth, with its best at
    # the edge of a step: also try the warning's length and its fractions.
    if fixed is not None:
        candidates = [fixed / n for n in range(1, 21)]
        best = math.log(
            min(
                candidates + [math.exp(best)],
                key=lambda tau: rate(math.log(tau)),
            )
        )
    return _on_condition(pf_interval, math.exp(best), "cost", *inputs)
