# Unit plans

One unit's maintenance plan as plain data: its life and repair models, its
tasks, each in its own clock, and its prices. A
[`RepairableRBD`](repairable.md) node takes a `UnitPlan` as it takes a spec
dict (the plan's shorthand), and a plan is valued alone through the diagram
of its one node. See [Maintaining a system](../guide/system-maintenance.md)
for the tasks and [Costs](../guide/costs.md) for the prices. They are
exported at the top level and from `repyability.maintenance`:
`from repyability import UnitPlan`.

::: repyability.UnitPlan

::: repyability.Prices

::: repyability.AgePolicy

::: repyability.BlockPolicy

::: repyability.ConditionPolicy

::: repyability.CalendarReplacement

::: repyability.AtShutdown

::: repyability.Inspection

::: repyability.Upgrade

::: repyability.Restoration

::: repyability.ImperfectRepair
