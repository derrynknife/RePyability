# API reference

Generated from the docstrings of the public API: everything exported from the
top-level `repyability` package (`from repyability import ...`). The
installed version is `repyability.__version__`. For task-oriented
explanations and runnable examples, see the [user guide](guide/index.md).

## System models

::: repyability.RBD

::: repyability.NonRepairableRBD

::: repyability.RepairableRBD

## Node models

Anything exposing `sf`/`ff` can be a node; these are the composite and helper
models provided here (see [Building an RBD](guide/building.md) and
[Redundancy models](guide/redundancy-models.md)).

::: repyability.StandbyModel

::: repyability.RepeatedNode

::: repyability.RepeatedStandbyNode

::: repyability.LoadSharingModel

::: repyability.RegressionNode

::: repyability.PerfectReliability

::: repyability.PerfectUnreliability

## Common-cause failures

::: repyability.CCFGroup

::: repyability.BetaFactor

::: repyability.MGL

## Condition-based evaluation

::: repyability.NodeState

## Components and maintenance

::: repyability.NonRepairable

::: repyability.Repairable

::: repyability.minimal_repair_time_to_nth_failure

## Result types

::: repyability.AvailabilityResult

::: repyability.Criticalities

::: repyability.UpDownImportance

::: repyability.FailureCriticalityIndex

::: repyability.RestorationCriticalityIndex

::: repyability.CostResult

::: repyability.ConfidenceInterval

::: repyability.RedundancyAllocation

::: repyability.MaintenancePolicy

::: repyability.FailureLimitPolicy
