"""Money: one home for how RePyability values what it computes.

Both routes use it: a diagram's valuations (``RepairableRBD``'s
``expected_npv``, ``npv_distribution``, ``plan_vs_actual``, its discounted
costs) and the single part's and the calendar's (``repyability.maintenance``
and ``repyability.schedule``).

- ``_rates``: discounting, with the one convention (continuous, at a rate
  per unit of the clock the money falls on) and the one place a yearly
  rate is converted to it.
- ``_escalation``: growth rates by stream, and inflation for real rates.
- ``value_schedule``: ``ValueSchedule``, a value earned in steps.
- ``_categories``: the cost categories.
- ``_flows``: the cash-flow table (``Ledger``) and its rows.
"""

from repyability.value.value_schedule import ValueSchedule

__all__ = ["ValueSchedule"]
