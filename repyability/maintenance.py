"""Shared result types for component maintenance-policy optimisations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenancePolicy:
    """An optimal preventive-maintenance policy for a single component.

    Returned by ``NonRepairable.optimal_replacement_policy()`` (age
    replacement) and ``Repairable.optimal_overhaul_policy()`` (overhaul
    under minimal repair).

    Attributes
    ----------
    interval : float
        The optimal preventive interval: the replacement age for an
        age-replacement policy, or the overhaul interval for a
        minimal-repair component. ``inf`` when preventive action never
        pays (run to failure / never overhaul).
    cost_rate : float
        The long-run cost per unit time under the policy. When
        ``interval`` is ``inf`` this is the limiting cost rate of running
        without preventive action.
    """

    interval: float
    cost_rate: float


@dataclass(frozen=True)
class FailureLimitPolicy:
    """An optimal failure-limit (replace-at-N-th-failure) policy.

    Returned by ``Repairable.optimal_failure_limit_policy()``: rather than
    renewing at a fixed *age*, the unit is repaired on each failure and
    replaced at the ``failure_count``-th failure.

    Attributes
    ----------
    failure_count : int
        The optimal number of failures per replacement cycle (repairs on the
        first ``failure_count - 1`` failures, then replace on the last).
    cost_rate : float
        The long-run cost per unit time under the policy.
    """

    failure_count: int
    cost_rate: float
