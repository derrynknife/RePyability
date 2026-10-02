# API reference

Generated from the docstrings of the public API: everything exported from the
top-level `repyability` package (`from repyability import ...`). The
installed version is `repyability.__version__`. For task-oriented
explanations and runnable examples, see the [user guide](guide/index.md).

## System models

::: repyability.RBD

::: repyability.NonRepairableRBD

::: repyability.RepairableRBD

::: repyability.FaultTree

::: repyability.PhasedMission

::: repyability.Network

## Node models

Anything exposing `sf`/`ff` can be a node; these are the composite and helper
models provided here (see [Building an RBD](guide/building.md) and
[Redundancy models](guide/redundancy-models.md)).

::: repyability.StandbyModel

::: repyability.DegradingNode

::: repyability.RepeatedNode

::: repyability.RepeatedStandbyNode

::: repyability.LoadSharingModel

::: repyability.RegressionNode

::: repyability.PerfectReliability

::: repyability.PerfectUnreliability

## Design inputs

::: repyability.ComponentOption

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

## Demonstration test planning

::: repyability.demonstration_sample_size

::: repyability.demonstrated_reliability

::: repyability.demonstration_test_multiple

::: repyability.demonstration_pass_probability

::: repyability.mtbf_test_time

::: repyability.demonstrated_mtbf

::: repyability.mtbf_pass_probability

## Result types

::: repyability.AnalysisRoute

::: repyability.AvailabilityResult

::: repyability.CapacityDistribution

::: repyability.Criticalities

::: repyability.UpDownImportance

::: repyability.FailureCriticalityIndex

::: repyability.RestorationCriticalityIndex

::: repyability.CostResult

::: repyability.ExpectedEvents

::: repyability.ExpectedCost

::: repyability.ConfidenceInterval

::: repyability.ControlVariate

::: repyability.UncertaintyResult

::: repyability.RedundancyAllocation

::: repyability.ReliabilityRedundancyAllocation

::: repyability.TotalCostAllocation

::: repyability.MaintenancePlan

::: repyability.AvailabilityAllocation

::: repyability.SparesDemand

::: repyability.SparesStock

::: repyability.SimulationChunk

::: repyability.run_shard

::: repyability.MaintenancePolicy

::: repyability.FailureLimitPolicy
