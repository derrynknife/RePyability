"""Maintenance: the single-unit optimisers' result types, and the choices
about a part's next replacement (#319).

RePyability is about timelines: the calendar side of maintenance planning
(tasks on dates, operating schedules, meters read by date) is Reliafy's.
``README.md`` in this package records the use cases, the question each
answers and how it is solved."""

from dataclasses import dataclass

from repyability.rbd.results import _ResultMapping


@dataclass(frozen=True)
class MaintenancePolicy(_ResultMapping):
    """An optimal preventive-maintenance policy for a single component.

    Returned by the ``optimal_replacement_policy()`` method of
    [`NonRepairable`][repyability.NonRepairable] (age replacement: replace
    at age ``interval`` or on failure, whichever comes first) and by the
    ``optimal_overhaul_policy()`` method of
    [`Repairable`][repyability.Repairable] (periodic overhaul: repair each
    failure and renew the unit every ``interval``). It is an immutable
    (frozen) dataclass, read as a mapping too, with ``to_dict()`` for
    JSON.

    Attributes
    ----------
    interval : float
        The optimal preventive interval, in the time units of the
        component's models: the replacement age for age replacement, or
        the overhaul interval for a repairable component. ``inf`` when
        preventive action never pays (run to failure / never overhaul).
    cost_rate : float
        The long-run expected cost per unit time under the policy. When
        ``interval`` is ``inf`` this is the limiting cost rate of running
        without preventive action.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import NonRepairable
    >>> unit = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
    >>> unit.set_costs_planned_and_unplanned(cp=1, cu=5)
    >>> policy = unit.optimal_replacement_policy()
    >>> round(policy.interval, 1), round(policy.cost_rate, 6)
    (493.0, 0.003462)
    """

    interval: float
    cost_rate: float


@dataclass(frozen=True)
class FailureLimitPolicy(_ResultMapping):
    """An optimal failure-limit (replace-at-N-th-failure) policy.

    Returned by the ``optimal_failure_limit_policy()`` method of
    [`Repairable`][repyability.Repairable]: rather than renewing at a
    fixed *age*, the unit is repaired on each failure and replaced at the
    ``failure_count``-th failure. It is an immutable (frozen) dataclass,
    read as a mapping too, with ``to_dict()`` for JSON.

    Attributes
    ----------
    failure_count : int
        The optimal number of failures per replacement cycle: repair the
        first ``failure_count - 1`` failures, then replace on the last.
        ``1`` means replace at every failure.
    cost_rate : float
        The long-run expected cost per unit time under the policy,
        ``(cr * (n - 1) + co) / E[T_n]`` at ``n = failure_count``, where
        ``E[T_n]`` is the expected time to the ``n``-th failure.

    Examples
    --------
    >>> import surpyval as surv
    >>> from surpyval.recurrent import GeneralizedRenewal
    >>> from repyability import Repairable
    >>> grp = GeneralizedRenewal.fit_from_parameters(
    ...     [100.0, 2.0], 0.4, kijima="i", dist=surv.Weibull
    ... )
    >>> unit = Repairable(grp)
    >>> unit.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
    >>> policy = unit.optimal_failure_limit_policy(
    ...     seed=1, mc_samples=200, max_failures=15
    ... )
    >>> policy.failure_count, round(policy.cost_rate, 3)
    (7, 0.031)
    """

    failure_count: int
    cost_rate: float


@dataclass(frozen=True)
class OverhaulReplacePolicy(_ResultMapping):
    """An optimal repair, overhaul and replace policy for a single component.

    Returned by the ``optimal_overhaul_replace_policy()`` method of
    [`Repairable`][repyability.Repairable]: failures are repaired minimally,
    the unit is overhauled every ``interval`` (an overhaul takes off part of
    its age) and replaced by a new unit after ``cycles`` such periods. It is
    an immutable (frozen) dataclass, read as a mapping too, with
    ``to_dict()`` for JSON.

    Attributes
    ----------
    interval : float
        The time between overhauls, in the time units of the component's
        model. ``inf`` when overhauling never pays.
    cycles : int
        The number of periods of ``interval`` between replacements: the
        unit is overhauled ``cycles - 1`` times, and replaced at the end of
        the ``cycles``-th period. ``1`` means no overhaul: replace every
        ``interval``.
    cost_rate : float
        The long-run expected cost per unit time under the policy.

    Examples
    --------
    >>> from surpyval.recurrent import CrowAMSAA
    >>> from repyability import Repairable
    >>> unit = Repairable(CrowAMSAA.from_params([100.0, 2.5]))
    >>> unit.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
    >>> policy = unit.optimal_overhaul_replace_policy(
    ...     overhaul_factor=0.6, replacement_cost=20.0, max_cycles=5
    ... )
    >>> policy.cycles, round(policy.interval, 1), round(policy.cost_rate, 4)
    (2, 177.7, 0.1172)
    """

    interval: float
    cycles: int
    cost_rate: float


# After the policies, which the modules it imports may import from here.
from repyability.maintenance._calendar import (  # noqa: E402
    Calendar,
    Opportunity,
)
from repyability.maintenance._decisions import (  # noqa: E402
    FailureFinding,
    OnCondition,
    RunToFailure,
    ScheduledReplacement,
    failure_finding,
    on_condition,
    optimal_failure_finding_interval,
    optimal_on_condition_interval,
    run_to_failure,
    scheduled_replacement,
)
from repyability.maintenance._next import (  # noqa: E402
    NextReplacement,
    ReplacementChoice,
)
from repyability.maintenance._options import (  # noqa: E402
    DelayEffect,
    OptionValue,
    RenewalForecast,
    ReplacementComparison,
    SlackEffect,
)
from repyability.maintenance._replacement import (  # noqa: E402
    CalendarPolicy,
    PlannedReplacement,
)
from repyability.maintenance._skip import SkipValue  # noqa: E402
from repyability.maintenance._visits import (  # noqa: E402
    LayeredVisits,
    RatioChoice,
    layered_cost_rate,
    optimal_layered_visits,
)
