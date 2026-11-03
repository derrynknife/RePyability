"""Choices about one part's replacement, compared as costs from the part's
state now (UC-6 and UC-11 of this package's ``README.md``, and the
small calculators for a visit schedule).

Everything is in model time: the part's age now, and the times from now
to the next planned replacement, to a shut or to a visit. Dates, calendars
and the work orders behind them are for the caller to turn into these.

**Comparing options** (``compare_replacement_options``). A plan replaces
the part on fixed times, the next ``due_in`` from now and the following
one ``every`` later (block replacement); a failure is replaced at once at
the unplanned cost. The options differ in when the planned replacement is
done, or whether: each is costed to the following planned replacement, at
``horizon = due_in + every``, where every option has a new part in. A
replacement at time ``u`` for price ``p`` costs, in present value at the
discount rate ``r``::

    F_age(u) + exp(-r u) (p + F_0(horizon - u))

with ``F_a(t)`` the expected discounted cost of the failures in ``t`` of a
part ``a`` old, each replaced by a new one at once (exact: a
[`RepairableRBD`][repyability.RepairableRBD] component's ``expected_cost``).
Skipping it costs ``F_age(horizon)``. The replacement at the horizon is the
same in every option and left out. The planned replacement is made whether
or not the part failed since: a plan on fixed times, as for
[`skip_next_replacement`][repyability.NonRepairable.skip_next_replacement],
whose answer this agrees with.

**A delay** (``delay_effect``). A replacement planned ``planned_in`` from
now is done ``done_in`` from now instead. Taken as one cycle to the
replacement, the part survives to ``u`` with ``S(u) = R(age + u) / R(age)``
and the replacement costs::

    cost(u) = planned exp(-r u) S(u) + unplanned (1 - exp(-r u) S(u) - r M(u)),
    M(u) = integral of exp(-r x) S(x) from 0 to u.

The delay adds ``cost(done_in) - cost(planned_in)`` and
``S(planned_in) - S(done_in)`` to the chance the part fails first.

**A tolerance** (``slack_effect``). A task due at age ``T`` that may be
done up to ``slack`` late: the age replacement's chance of a failure first
and its long-run cost rate, at ``T`` and at ``T + slack``.

**A part renewed at each service** (``renewal_forecast``). From the part's
age and how far it is into the service interval, the chance it fails
before each service, and the expected failures, renewals and cost over a
window: a one-component [`RepairableRBD`][repyability.RepairableRBD]
(exact), against the part's best age replacement.
"""

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional

import numpy as np

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks


def _number(value, name: str, low: float = 0.0, strict: bool = False):
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


def _require_closed_form(part, what: str) -> None:
    part._require_costs()
    if part.model_parameterization not in ("parametric", "degradation"):
        raise NotImplementedError(
            f"{what} needs a life with a survival function in closed form "
            "(a surpyval parametric distribution, mixture or degradation "
            f"process); this part's is {part.model_parameterization}."
        )


def _survival(part, age: float):
    """``S(u) = R(age + u) / R(age)`` as a function of arrays of times."""
    sf = part.reliability_function
    base = float(np.ravel(sf(age))[0])
    if not base > 0.0:
        raise ValueError(
            f"The life gives no chance of a unit working at age {age:g}."
        )

    def conditional(u):
        u = np.asarray(u, dtype=float)
        return np.asarray(sf(age + u), dtype=float).reshape(u.shape) / base

    return conditional


def _discounted_length(part, age: float, until: float, rate: float) -> float:
    """``M(until)``: the integral of ``exp(-r x) S(x)`` to ``until``."""
    from repyability.rbd._mean_lifetime import _integrate, model_knots

    survival = _survival(part, age)

    def discounted(x: np.ndarray) -> np.ndarray:
        return np.exp(-rate * x) * np.ravel(survival(x))

    if until <= 0.0:
        return 0.0
    knots = model_knots(part.reliability) - age
    inside = knots[(knots > 0.0) & (knots < until)]
    return float(
        _integrate(discounted, np.concatenate(([0.0], inside, [until])))
    )


def _single_cycle_cost(part, age: float, until: float, rate: float) -> float:
    """The present value of one cycle: a planned replacement at ``until``
    (at its planned cost) unless the part fails first (at its unplanned
    cost)."""
    survival = float(np.ravel(_survival(part, age)(until))[0])
    kept = math.exp(-rate * until) * survival
    failed = 1.0 - kept - rate * _discounted_length(part, age, until, rate)
    return part.cu * max(failed, 0.0) + part.cp * kept


def _unit(part, plan: Optional[dict] = None):
    """A one-component diagram of the part, replaced in no time."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    spec: dict = {
        "reliability": part.reliability,
        "repairability": "instant",
        "replace_cost": part.cu,
    }
    if plan is not None:
        spec["preventive"] = plan
    return RepairableRBD([("s", "unit"), ("unit", "t")], {"unit": spec})


def _failure_costs(part, age: float, times, rate: float) -> np.ndarray:
    """``F_age(t)``: the expected discounted cost of the failures in each
    ``t`` of a part ``age`` old, each replaced by a new one at once."""
    from repyability.rbd.node_state import NodeState

    times = np.asarray(times, dtype=float)
    out = np.zeros(times.shape)
    positive = times > 0.0
    if positive.any():
        state = {"unit": NodeState(age=age)}
        out[positive] = np.asarray(
            _unit(part)
            .expected_cost(times[positive], state=state, discount_rate=rate)
            .mean,
            dtype=float,
        ).ravel()
    return out


# -- comparing options (UC-6) ------------------------------------------------


@dataclass
class OptionValue(_ResultMapping):
    """One option of
    [`ReplacementComparison`][repyability.ReplacementComparison].

    Attributes
    ----------
    kind : str
        ``"replace"`` (as planned), ``"now"``, ``"defer"``, ``"shut"`` (move
        to a shut) or ``"skip"``.
    name : str
        The option's name: its kind, or the shut's name.
    time : float or None
        When the planned replacement is done, from now; None for skipping.
    price : float or None
        What the planned replacement costs; None for skipping.
    cost : float
        The present-value cost to the following planned replacement.
    saving : float
        What it saves against replacing as planned: that cost less this
        one (positive: cheaper than the plan).
    """

    kind: str
    name: str
    time: Optional[float]
    price: Optional[float]
    cost: float
    saving: float


@dataclass
class ReplacementComparison(_ResultMapping):
    """The options for a part's next planned replacement, compared as
    present-value costs from its age now: returned by
    [`NonRepairable.compare_replacement_options`][repyability.NonRepairable.compare_replacement_options].

    Attributes
    ----------
    age : float
        The part's age now.
    due_in : float
        The time from now to its next planned replacement.
    every : float
        The plan's interval: the following planned replacement is this
        long after the next.
    horizon : float
        ``due_in + every``, to which every option is costed.
    discount_rate : float
        The continuous discount rate.
    options : list of OptionValue
        Every option, replacing as planned first.
    best : OptionValue
        The cheapest option (on a tie, the earlier).
    """

    age: float
    due_in: float
    every: float
    horizon: float
    discount_rate: float
    options: List[OptionValue]
    best: OptionValue


def _shuts(shuts) -> list:
    if shuts is None:
        return []
    items = shuts.items() if isinstance(shuts, Mapping) else shuts
    out = []
    for i, item in enumerate(items):
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            name, (time, price) = item[0], item[1]
        else:
            name, (time, price) = f"shut {i + 1}", item
        out.append((name, time, price))
    return out


def compare_replacement_options(
    part,
    age,
    due_in,
    every,
    *,
    defer_by=(),
    shuts=None,
    discount_rate=0.0,
) -> ReplacementComparison:
    """The work of ``NonRepairable.compare_replacement_options``."""
    _require_closed_form(part, "compare_replacement_options")
    age = _number(age, "age")
    every = _number(every, "every", strict=True)
    due_in = _number(due_in, "due_in", strict=True)
    if due_in > every:
        raise ValueError(
            f"due_in ({due_in:g}) must be at most every ({every:g}): the "
            "next planned replacement is within one interval."
        )
    rate = _number(discount_rate, "discount_rate")
    horizon = due_in + every
    from repyability.maintenance._skip import _warn_discounted_away

    _warn_discounted_away(part, rate, [horizon])
    plans = [("replace", "replace", due_in, part.cp)]
    if due_in > 0.0:
        plans.append(("now", "now", 0.0, part.cp))
    for delay in np.atleast_1d(checks.real_array(defer_by, "defer_by")):
        if not (0.0 < delay and due_in + delay <= horizon):
            raise ValueError(
                "Each defer_by must be more than 0 and at most every "
                f"({every:g}), so that the replacement is done before the "
                f"following one; got {delay!r}."
            )
        plans.append(
            ("defer", f"defer by {delay:g}", due_in + float(delay), part.cp)
        )
    for name, time, price in _shuts(shuts):
        time = _number(time, f"The time of {name!r}")
        price = _number(price, f"The price of {name!r}")
        if time > horizon:
            raise ValueError(
                f"{name!r} is at {time:g}, past the following planned "
                f"replacement ({horizon:g}): move it before it."
            )
        plans.append(("shut", name, time, price))
    times = np.array([p[2] for p in plans])
    own = _failure_costs(part, age, np.append(times, horizon), rate)
    fresh = _failure_costs(part, 0.0, horizon - times, rate)
    costs = [
        own[i] + math.exp(-rate * t) * (p[3] + fresh[i])
        for i, (p, t) in enumerate(zip(plans, times))
    ]
    planned_cost = costs[0]
    options = [
        OptionValue(kind, name, time, price, cost, planned_cost - cost)
        for (kind, name, time, price), cost in zip(plans, costs)
    ]
    skip = float(own[-1])
    options.append(
        OptionValue("skip", "skip", None, None, skip, planned_cost - skip)
    )
    best = min(options, key=lambda option: option.cost)
    return ReplacementComparison(
        age, due_in, every, horizon, rate, options, best
    )


# -- a delay, and a tolerance ------------------------------------------------


@dataclass
class DelayEffect(_ResultMapping):
    """What delaying a planned replacement adds: returned by
    [`NonRepairable.delay_effect`][repyability.NonRepairable.delay_effect],
    [`cost_of_waiting_for_visit`][repyability.NonRepairable.cost_of_waiting_for_visit]
    and
    [`cost_of_missed_visit`][repyability.NonRepairable.cost_of_missed_visit].

    Attributes
    ----------
    age : float
        The part's age now.
    planned_in, done_in : float
        When the replacement was planned and when it is done, from now.
    delay : float
        ``done_in - planned_in``.
    failure_probability_planned, failure_probability_done : float
        The chance the part fails before the replacement, planned and
        delayed.
    added_failure_probability : float
        Their difference: the risk the delay adds.
    cost_planned, cost_done : float
        The present-value cost of the cycle to the replacement (the part's
        planned cost if it survives, its unplanned cost if it fails first).
    added_cost : float
        Their difference: what the delay adds in expectation.
    weight : float
        The chance the delay happens (1, or the chance a visit is missed).
    expected_added_failure_probability, expected_added_cost : float
        The added risk and cost times ``weight``.
    """

    age: float
    planned_in: float
    done_in: float
    delay: float
    failure_probability_planned: float
    failure_probability_done: float
    added_failure_probability: float
    cost_planned: float
    cost_done: float
    added_cost: float
    weight: float
    expected_added_failure_probability: float
    expected_added_cost: float


def delay_effect(
    part, age, planned_in, done_in, *, weight=1.0, discount_rate=0.0
) -> DelayEffect:
    """The work of ``NonRepairable.delay_effect`` (see there)."""
    _require_closed_form(part, "delay_effect")
    age = _number(age, "age")
    planned_in = _number(planned_in, "planned_in")
    done_in = _number(done_in, "done_in")
    if done_in < planned_in:
        raise ValueError(
            f"done_in ({done_in:g}) must not be before planned_in "
            f"({planned_in:g}): a delay moves the work later."
        )
    weight = _number(weight, "the weight")
    if weight > 1.0:
        raise ValueError(f"The weight is a probability; got {weight!r}.")
    rate = _number(discount_rate, "discount_rate")
    survival = _survival(part, age)
    fail_planned = 1.0 - float(np.ravel(survival(planned_in))[0])
    fail_done = 1.0 - float(np.ravel(survival(done_in))[0])
    cost_planned = _single_cycle_cost(part, age, planned_in, rate)
    cost_done = _single_cycle_cost(part, age, done_in, rate)
    added_risk = fail_done - fail_planned
    added_cost = cost_done - cost_planned
    return DelayEffect(
        age,
        planned_in,
        done_in,
        done_in - planned_in,
        fail_planned,
        fail_done,
        added_risk,
        cost_planned,
        cost_done,
        added_cost,
        weight,
        weight * added_risk,
        weight * added_cost,
    )


def next_visit(visits, after: float) -> float:
    """The first of the visits at or after ``after``: an
    [`Opportunity`][repyability.Opportunity], or a list of times."""
    from repyability.maintenance._calendar import Opportunity

    if isinstance(visits, Opportunity):
        end = (
            after + 2.0 * visits.period
            if visits.period is not None
            else math.inf
        )
        found = visits.times(after, end)
    else:
        times = np.sort(checks.real_array(visits, "visits").ravel())
        found = times[times >= after]
    if len(found) == 0:
        raise ValueError(
            f"There is no visit at or after {after:g}: the visits end "
            "before it."
        )
    return float(found[0])


@dataclass
class SlackEffect(_ResultMapping):
    """What allowing a task to be done late costs: returned by
    [`NonRepairable.slack_effect`][repyability.NonRepairable.slack_effect].

    Attributes
    ----------
    interval : float
        The age at which the task (a replacement) is due.
    slack : float
        How late it may be done.
    risk_on_time, risk_at_limit : float
        The chance the part fails before the replacement, done at the due
        age and at the latest age allowed.
    added_risk : float
        Their difference.
    cost_rate_on_time, cost_rate_at_limit : float
        The long-run cost a unit of time of replacing at each age.
    added_cost_rate : float
        Their difference: the most the tolerance costs, if the work is
        always as late as allowed.
    mean_cost_rate_late : float
        The long-run cost rate if the work is done a uniformly spread time
        late, between 0 and the slack.
    """

    interval: float
    slack: float
    risk_on_time: float
    risk_at_limit: float
    added_risk: float
    cost_rate_on_time: float
    cost_rate_at_limit: float
    added_cost_rate: float
    mean_cost_rate_late: float


def slack_effect(part, interval, slack) -> SlackEffect:
    """The work of ``NonRepairable.slack_effect`` (see there)."""
    _require_closed_form(part, "slack_effect")
    interval = _number(interval, "interval", strict=True)
    slack = _number(slack, "slack")
    sf = part.reliability_function
    risk_on = 1.0 - float(np.ravel(sf(interval))[0])
    risk_at = 1.0 - float(np.ravel(sf(interval + slack))[0])
    rate_on = float(part._cost_rate(interval))
    rate_at = float(part._cost_rate(interval + slack))
    if slack > 0.0:
        from scipy.integrate import quad

        mean_late = (
            quad(
                lambda late: float(part._cost_rate(interval + late)),
                0.0,
                slack,
                epsabs=0.0,
                epsrel=1e-10,
            )[0]
            / slack
        )
    else:
        mean_late = rate_on
    return SlackEffect(
        interval,
        slack,
        risk_on,
        risk_at,
        risk_at - risk_on,
        rate_on,
        rate_at,
        rate_at - rate_on,
        float(mean_late),
    )


# -- a part renewed at each service (UC-11) ----------------------------------


@dataclass
class RenewalForecast(_ResultMapping):
    """A part renewed on a plan, from its state now: returned by
    [`NonRepairable.renewal_forecast`][repyability.NonRepairable.renewal_forecast].

    Attributes
    ----------
    window : float
        The time ahead the forecast covers.
    age : float
        The part's age now.
    every : float
        The service interval.
    resets : bool
        Whether the services keep to fixed times (False) or each interval
        starts at the last renewal, planned or not (True).
    service_times : list of float
        The times from now of the services in the window.
    risk_before_service : list of float
        The chance of a failure before each of them: the first from the
        part's age now; each later one from a part new at the service
        before.
    expected_failures : float
        The expected number of failures in the window.
    expected_renewals : float
        The expected number of planned renewals in the window.
    expected_cost : float
        The expected cost in the window: failures at the unplanned cost,
        renewals at the planned cost.
    failure_probability_without_renewals : float
        The chance the part fails in the window with no planned renewal.
    expected_failures_without_renewals : float
        The expected failures in the window with no planned renewal.
    cost_rate : float
        The plan's long-run cost a unit of time.
    run_to_failure_cost_rate : float
        The long-run cost rate with no planned renewal.
    best_interval, best_cost_rate : float
        The part's best age replacement
        ([`optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy])
        and its cost rate; ``inf`` for an interval when renewing never
        pays.
    worth_it : bool
        Whether the plan costs less than running to failure.
    """

    window: float
    age: float
    every: float
    resets: bool
    service_times: List[float]
    risk_before_service: List[float]
    expected_failures: float
    expected_renewals: float
    expected_cost: float
    failure_probability_without_renewals: float
    expected_failures_without_renewals: float
    cost_rate: float
    run_to_failure_cost_rate: float
    best_interval: float
    best_cost_rate: float
    worth_it: bool


def renewal_forecast(
    part, window, every, *, resets, age=0.0, since_renewal=None
) -> RenewalForecast:
    """The work of ``NonRepairable.renewal_forecast`` (see there)."""
    from repyability.rbd.node_state import NodeState

    _require_closed_form(part, "renewal_forecast")
    window = _number(window, "window", strict=True)
    every = _number(every, "every", strict=True)
    age = _number(age, "age")
    if not isinstance(resets, bool):
        raise TypeError(
            "resets must be True (each interval starts at the last "
            "renewal, planned or not) or False (the services keep to "
            f"fixed times); got {resets!r}."
        )
    if resets:
        if since_renewal is not None:
            raise ValueError(
                "A plan that resets has its services every interval after "
                "the last renewal, which is the part's age: give no "
                "since_renewal."
            )
        since = age
    else:
        since = age if since_renewal is None else since_renewal
        since = _number(since, "since_renewal")
    if since >= every:
        raise ValueError(
            f"The part is {since:g} into an interval of {every:g}: a "
            "service is due now or past. Give a time into the interval "
            "less than every."
        )
    first = every - since
    services = np.arange(first, window, every)
    survival = _survival(part, age)
    risks = (
        [1.0 - float(np.ravel(survival(first))[0])] if len(services) else []
    )
    risks += [1.0 - float(np.ravel(part.reliability_function(every))[0])] * (
        max(len(services) - 1, 0)
    )
    plan = {"interval": every, "cost": part.cp}
    if not resets:
        plan["policy"] = "block"
    unit = _unit(part, plan)
    state = {
        "unit": (
            NodeState(age=age)
            if resets
            else NodeState(age=age, since_scheduled=since)
        )
    }
    events = unit.expected_events(window, state=state)
    failures = float(events.node_failures["unit"])
    renewals = float(events.node_preventive["unit"])
    cost = float(unit.expected_cost(window, state=state).mean)
    bare = part.failures(window, age=age)
    policy = part.optimal_replacement_policy()
    if resets:
        plan_rate = float(part._cost_rate(every))
    else:
        plan_rate = (
            part.cp + part.cu * float(part.failures(every).expected_failures)
        ) / every
    rtf = float(part._run_to_failure_rate())
    return RenewalForecast(
        window,
        age,
        every,
        resets,
        [float(s) for s in services],
        risks,
        failures,
        renewals,
        cost,
        float(bare.probability),
        float(bare.expected_failures),
        plan_rate,
        rtf,
        float(policy.interval),
        float(policy.cost_rate),
        bool(plan_rate < rtf),
    )
