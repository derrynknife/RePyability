"""Shared result types for component maintenance-policy optimisations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenancePolicy:
    """An optimal preventive-maintenance policy for a single component.

    Returned by the ``optimal_replacement_policy()`` method of
    [`NonRepairable`][repyability.NonRepairable] (age replacement: replace
    at age ``interval`` or on failure, whichever comes first) and by the
    ``optimal_overhaul_policy()`` method of
    [`Repairable`][repyability.Repairable] (periodic overhaul: repair each
    failure and renew the unit every ``interval``). It is an immutable
    (frozen) dataclass.

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
class FailureLimitPolicy:
    """An optimal failure-limit (replace-at-N-th-failure) policy.

    Returned by the ``optimal_failure_limit_policy()`` method of
    [`Repairable`][repyability.Repairable]: rather than renewing at a
    fixed *age*, the unit is repaired on each failure and replaced at the
    ``failure_count``-th failure. It is an immutable (frozen) dataclass.

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
    ...     seed=1, n_simulations=200, max_failures=15
    ... )
    >>> policy.failure_count, round(policy.cost_rate, 3)
    (7, 0.031)
    """

    failure_count: int
    cost_rate: float
