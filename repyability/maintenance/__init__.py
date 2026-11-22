"""Maintenance: the single-unit optimisers' result types, and the choices
about a part's next replacement (#319).

These work in model time; the calendar side (tasks on dates, recurrence
rules, operating schedules) is ``repyability.schedule``.
``README.md`` in this package records the use cases, the question each
answers and how it is solved."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

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
        The long-run expected cost per unit time under the policy,
        undiscounted (also when the policy was chosen with a discount
        rate). When ``interval`` is ``inf`` this is the limiting cost rate
        of running without preventive action.
    discount_rate : float
        The continuous discount rate per unit of time the policy was
        chosen at, 0 for none.
    npv : float
        The present value of every replacement from a new unit, forever,
        under the policy (Fox, 1966); ``inf`` with no discounting.
    eac : float
        The equivalent annual cost: the steady cost per unit time worth
        the same as ``npv``, ``discount_rate * npv``. With no discounting
        it is ``cost_rate``.

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
    discount_rate: float = 0.0
    npv: float = math.inf
    eac: Optional[float] = None

    def __post_init__(self):
        if self.eac is None:
            object.__setattr__(self, "eac", self.cost_rate)


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
    >>> grp = GeneralizedRenewal.from_params(
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
    and the cheapest of four plans is kept: overhaul every ``interval``
    (an overhaul takes off part of the unit's age) and replace by a new
    unit after ``cycles`` such periods; overhaul only, never replacing;
    replace only, with no overhauls; or replace at the
    ``failure_count``-th failure. It is an immutable (frozen) dataclass,
    read as a mapping too, with ``to_dict()`` for JSON.

    Attributes
    ----------
    interval : float
        The time between overhauls (between replacements for ``"replace
        only"``), in the time units of the component's model. ``inf`` when
        there is no planned overhaul or replacement (``"failure limit"``
        and ``"run to failure"``).
    cycles : int or float
        The number of periods of ``interval`` between replacements: the
        unit is overhauled ``cycles - 1`` times, and replaced at the end of
        the ``cycles``-th period. ``1`` means no overhaul: replace every
        ``interval``. ``inf`` for ``"overhaul only"``: never replace.
    cost_rate : float
        The long-run expected cost per unit time under the policy.
    plan : str
        The plan that won: ``"overhaul and replace"``, ``"overhaul
        only"``, ``"replace only"``, ``"failure limit"``, or ``"run to
        failure"`` when the unit does not wear out, so that nothing pays.
    failure_count : int or None
        For ``"failure limit"``, the failure at which the unit is replaced
        (the earlier ones are repaired); otherwise None.
    alternatives : dict
        Each plan compared, by name, with its cost rate and its best
        ``interval``, ``cycles`` or ``failure_count``.
    saving : float
        How much less the winner costs per unit time than the next best
        plan; 0 when nothing else was compared.
    cycles_limit_reached : bool
        True when the best ``"overhaul and replace"`` plan used the most
        periods searched (``max_cycles``): more might pay, and they tend
        towards ``"overhaul only"``.

    Examples
    --------
    >>> from surpyval.recurrent import CrowAMSAA
    >>> from repyability import Repairable
    >>> unit = Repairable(CrowAMSAA.from_params([100.0, 2.5]))
    >>> unit.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
    >>> policy = unit.optimal_overhaul_replace_policy(
    ...     overhaul_factor=0.6, replacement_cost=20.0, max_cycles=5
    ... )
    >>> policy.plan, policy.cycles, round(policy.interval, 1)
    ('overhaul only', inf, 73.8)
    >>> sorted(policy.alternatives)  # doctest: +NORMALIZE_WHITESPACE
    ['failure limit', 'overhaul and replace', 'overhaul only',
     'replace only']
    """

    interval: float
    cycles: Union[int, float]
    cost_rate: float
    plan: str = "overhaul and replace"
    failure_count: Optional[int] = None
    alternatives: Optional[Mapping[str, Mapping[str, Any]]] = None
    saving: float = 0.0
    cycles_limit_reached: bool = False


# After the policies, which the modules it imports may import from here.
from repyability.maintenance._calendar import Opportunity  # noqa: E402
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
from repyability.maintenance._finding_window import (  # noqa: E402
    FailureFindingWindow,
    failure_finding_window,
)
from repyability.maintenance._next import (  # noqa: E402
    NextReplacement,
    ReplacementChoice,
)
from repyability.maintenance._operating import (  # noqa: E402
    ScheduleValue,
    SwitchTime,
    best_switch_time,
    operating_schedule_value,
)
from repyability.maintenance._opportunities import (  # noqa: E402
    Calendar,
    Opportunities,
    OpportunityPrice,
    OpportunityPrices,
    Shutdown,
    Shutdowns,
)
from repyability.maintenance._options import (  # noqa: E402
    AgeReplacementValue,
    DelayEffect,
    OptionValue,
    RenewalForecast,
    ReplacementComparison,
    SlackEffect,
)
from repyability.maintenance._plan import (  # noqa: E402
    AgePolicy,
    AtShutdown,
    BlockPolicy,
    CalendarReplacement,
    ConditionPolicy,
    ImperfectRepair,
    Inspection,
    Prices,
    Restoration,
    UnitPlan,
    Upgrade,
)
from repyability.maintenance._repair_or_replace import (  # noqa: E402
    RepairOrReplace,
    RepairOrReplacePolicy,
    UnitDecision,
)
from repyability.maintenance._replacement import (  # noqa: E402
    CalendarPolicy,
    PlannedReplacement,
)
from repyability.maintenance._shutdowns import (  # noqa: E402
    ShutdownRoute,
    SlowSolverWarning,
)
from repyability.maintenance._site_policy import (  # noqa: E402
    DatedChoice,
    DatedReplacement,
    NextReplacementOnDates,
    ReplacementRule,
    RuleCost,
    ShutdownPolicy,
)
from repyability.maintenance._skip import SkipValue  # noqa: E402
from repyability.maintenance._visits import (  # noqa: E402
    LayeredVisits,
    RatioChoice,
    layered_cost_rate,
    optimal_layered_visits,
)
from repyability.maintenance._window_cost import CostDistribution  # noqa: E402
from repyability.maintenance._window_outcomes import WindowCosts  # noqa: E402
