# Maintenance: use cases

This is the working record of RePyability's maintenance questions (#319),
built **one use case at a time**: each use case is a question someone
actually asks, written down before any code, with how it is answered and
what was learned.

**RePyability computes; Reliafy records** (the scope note in
`CLAUDE.md`). Every use case here is asked in model time, from a state:
ages, how far into an interval, windows of time from now. The use cases
on the calendar (tasks on dates, monthly and yearly rules, operating
schedules and meters read by date) are recorded in
`repyability/schedule/README.md`, under the same numbers (UC-2, UC-3,
UC-4 and UC-10 there). The caller (the Reliafy app, or anyone) holds the
records and decides what "today" is.

**Money has one home**, `repyability.value`: discounting (continuous, at
a rate per unit of the clock the money falls on; a yearly rate converted
to it in one place), escalation and real rates, `ValueSchedule`, the
cost categories and the cash-flow table. The use cases here that value a
plan (UC-19, UC-21, UC-44, UC-46, UC-57 and the others with a
`discount_rate`) and the calendar's in `repyability/schedule/README.md`
all go through it.

## Layers

Which module answers which question (the layering in `CLAUDE.md`,
Architecture):

| Question | Where | Use cases |
|---|---|---|
| What is an amount at a time worth now; how do prices grow; what is the cash-flow table? | `repyability.value` | UC-19, UC-57 |
| What is one unit's plan (life, repairs, tasks, prices, salvage, growth, retirement)? | `repyability.maintenance.UnitPlan` | UC-63 |
| What is that plan worth alone, its cash flows, spread, plan against actual, long-run cost and best interval? | `UnitPlan`'s methods, each through its one-node diagram | UC-65 |
| A single part's policies (age, block, calendar, opportunities, next replacement, slack, layered visits) | `NonRepairable` and the functions of this package | UC-5 to UC-18, UC-44, UC-45 |
| The discounted cost per year of age replacement | `_options.cycle_eac`, one definition for a part and a diagram's component | UC-44 |
| A unit whose life is a subassembly | a `NonRepairableRBD` as the life | UC-66 |
| What happens on dates: rules, duty plans, each unit's plan on the calendar | `repyability.schedule` (its README) | UC-10, UC-39, UC-64 there |
| What belongs to the system: structure, system downtime and value, crews, spares, site shutdowns, common-cause groups | `RepairableRBD` | UC-21, UC-23, UC-46, UC-59 |
| How bad a bad run is (a quantile or tail mean of a priced outcome, with its bound) | `repyability.value._risk` | UC-68 |

Not yet single:
- the calendar's present values (`failures_by_period`) still bracket their discounting where the diagram's are exact;
- a nested `RepairableRBD`'s costs are not counted in its parent.

## The use cases, by number

| # | The question here, in model time | State |
|---|---|---|
| UC-1 | A system's expected maintenance work in each of a run of windows from its state | **Built**: `RepairableRBD.expected_events_by_window` (#375) |
| UC-2, UC-3, UC-4 | Tasks on dates and monthly rules | **Built** in `repyability.schedule` (its README) |
| UC-5 | A part's chance of failing, expected failures and their cost over a window, from its age | **Built**: `NonRepairable.failures` |
| UC-6 | Right after a failure, the value of skipping the next planned replacement; and, from any state, replace, defer, move to a shut or skip compared | **Built**: `NonRepairable.skip_next_replacement`, and `NonRepairable.compare_replacement_options` |
| UC-7 | The cheapest replacement policy when replacing at a shut is cheaper than between shuts | Built as `replacement_policies`, then removed: UC-13 answers it |
| UC-8 | Each policy's cost over a window from a state, and its spread | **Built** for a part: `NonRepairable.cost_distribution` (run to failure, age and block replacement, and a calendar policy, `calendar=`); a system's simulated by `RepairableRBD.cost_by_window` |
| UC-9 | The cost in each of a run of windows from a state, and how often each outcome happens | **Built** for a part: `NonRepairable.cost_by_window` (the same policies); a system's simulated by `RepairableRBD.cost_by_window` (#375) |
| UC-10 | Tasks by use on an operating schedule | In `repyability.schedule` (its README) |
| UC-11 | A part renewed at every service and at failure: its risk before each service, from its state, in running hours | **Built**: `NonRepairable.renewal_forecast` |
| UC-14 | The best visit interval and depth ratio for a layered package of visits | **Built**: `optimal_layered_visits` |
| UC-15 | The cost and risk of letting a task slip by a tolerance | **Built**: `NonRepairable.slack_effect` |
| UC-16 | The extra risk and cost of rounding a due replacement up to the next visit | **Built**: `NonRepairable.cost_of_waiting_for_visit` |
| UC-18 | The numbers behind each maintenance decision for a failure mode: run to failure, scheduled replacement, failure finding, on condition | **Built**: `run_to_failure`, `scheduled_replacement`, `failure_finding`, `on_condition` |
| UC-17 | The risk added when a visit is missed and its work waits | **Built**: `NonRepairable.cost_of_missed_visit` (and `delay_effect`) |
| UC-12 | When to replace this part next: now, at a scheduled outage, at a planned replacement, or at its failure? | **Built**: `NonRepairable.optimal_next_replacement` |
| UC-13 | The long-run optimal replacement policy over a calendar of shuts with prices of their own | **Built**: `NonRepairable.optimal_opportunity_policy` (0.14's `optimal_calendar_policy`, deprecated), the prices on the part since UC-45 |
| UC-19 | What a plan is worth over a horizon at a discount rate, and as a yearly cost | **Built**: `RepairableRBD.expected_npv`, `NonRepairable.age_replacement_value` (value over time and escalation: UC-57) |
| UC-20 | One unit run on an operating schedule (covariates and value by stretch, in model time): the value earned, the failures' cost and the NPV, and the best time to switch from one way of running to another | **Built** (one unit, numerical): `operating_schedule_value`, `best_switch_time`; (a diagram, simulated): `RepairableRBD.best_switch_time` |
| UC-21 | A plan against its outage log: expected and actual cash flows so far, the variance, and the plan re-valued from the states now | **Built**: `RepairableRBD.plan_vs_actual` |
| UC-46 | The spread of a plan's net present value: each simulated run's NPV, its percentiles, the chance of missing a hurdle and the average of the worst runs | **Built**: `RepairableRBD.npv_distribution` (with `contract=`), `compare_npv` |
| UC-22 | Overhauls that restore a unit only partly (a Kijima restoration factor), every Nth replacing it: availability, failures and cost per year against overhauls that renew it, and the cheapest interval | **Built**: a `"preventive"` spec's `"restoration"`, `"replace_every"` and `"replacement_cost"`, simulated (#398) |
| UC-23 | The line is shut 3 days every 12 months, every block gets its PM then and the bearing is replaced if older than X: availability, the shut's share of the downtime and cost, failures between shuts, and whether a 9- or 18-month shut is cheaper | **Built**: `RepairableRBD(shutdowns=...)`, tasks bound with `"shutdown"` and `"older_than"`; exact and numerical; the shuts given as the site's `Opportunities` in model time (each opportunity a shutdown, `shut_costs` their costs) or as dicts, its shorthand |
| UC-24 | Outages on a timeline: shuts at given times with a duration and a scope (renew some blocks, restore others), the plan's cost by window and its NPV | **Built**: the same `shutdowns` at `"times"`; over a horizon, exact or numerical where every job renews its block (`expected_npv`, `expected_cost`, `expected_events_by_window`), simulated where a job restores part-way; no long run |
| UC-51 | A part replaced at a list of times in model time (8,000, 14,000 and 26,000 hours) rather than on one interval: its plan's availability, failures, cost and NPV over a horizon | **Built**: a `"preventive"` spec's `"replace_at"`; numerical over a horizon (`expected_npv`, `expected_cost`, `expected_events`, `mission_availability`, `point_availability`), simulated too; the long run refused |
| UC-52 | Is a mid-life upgrade worth its capital cost, should the system be retired before the horizon, and what is it worth with a salvage value at the end? | **Built**: a component spec's `"upgrade"` (numerical over a horizon, and simulated: `npv_distribution`, `compare_npv`), and `expected_npv(retire_at=, salvage=)` |
| UC-53 | A protective device tested for a hidden failure at given times (not one interval), from its age, over a window: what each test finds, the time it is failed, the demands that meet it failed, and the plan's cost and present value; and the same never tested | **Built**: `failure_finding_window` (exact at the tests where only a test finds a failure, numerical where a demand shows it; to a stated precision) |
| UC-25 | How much of the downtime and lost production is planned and how much is from failures, and how many failures the plan avoids against running to failure | **Built**: `RepairableRBD.maintenance_effect`, and the split in `ExpectedEvents`, `EventsByWindow`, `AvailabilityResult` and `CostResult`; exact where the blocks fail independently, simulated otherwise |
| UC-26 | A pump run part of the time with its overhaul counted in running hours: availability and cost, and at another duty | **Built**: a `"preventive"` or `"inspection"` spec's `"basis": "operating"`, put on the calendar through the block's `"duty"` |
| UC-43 | A site's shuts on real dates (rules, one-offs, plain periods, a use on a meter) in one record, the best rule for a part there, worked out by a route that is always named and never chosen silently, given back as dates; the unit in service, its cost's spread, and a diagram's outages from the same record | **Built**: `Opportunities` of `Opportunity` on real dates (0.19's `Shutdowns` and `Shutdown`, deprecated), `optimal_opportunity_policy(site, ...)` returning a `ShutdownPolicy`, `Opportunities.route`, `opportunity_policy_cost(site, ...)`, `optimal_next_replacement(..., site=)`, `cost_distribution`/`cost_by_window(policy=)`, `Opportunities.on_diagram` |
| UC-44 | The best replacement policy in the long run at a discount rate: age replacement, a calendar of shuts in model time and a site on real dates, by the least equivalent annual cost | **Built**: `discount_rate=` of `find_optimal_replacement`, `optimal_replacement_policy` and `optimal_opportunity_policy` (`discount_rate=` in model time, and `discount=` without a horizon on real dates) |
| UC-45 | One record of a site's opportunities (stops and shuts) for every part, each part bringing its own prices (at an opportunity, on any other day, at a failure, and per opportunity) and the length of its job, so an opportunity too short for the job is not used; and the prices worked out from a replacement's cost and the plant's downtime | **Built**: `Opportunity`, `Opportunities`, `NonRepairable(opportunity_cost=, opportunity_costs=, job_duration=)`, the policy's `prices` (`OpportunityPrices`), `NonRepairable.opportunity_prices` |
| UC-55 | How sure an NPV and a choice of intervals are: the NPV's change with the discount rate, its spread over the fitted models' uncertainty with each input's share, and how often the chosen replacement intervals stay the best | **Built**: `expected_npv(rate_sensitivity=True)`, `RepairableRBD.expected_npv_uncertainty`, `optimal_replacement_intervals(robust=True or "reoptimise")` |
| UC-56 | At each planned outage, also replace the parts close enough to their own replacement, since the plant is down anyway: what the scope adds, by outage, and what the plan then costs | **Built** (simulated): a shutdown's `"opportunistic"`, the count by outage in `AvailabilityResult.outage_jobs`; the exact methods refuse, naming the simulation |
| UC-57 | What a plan is worth when the price it earns steps up, its costs escalate at their own rates, and the rate is real or nominal; and the same escalation on dates | **Built**: `expected_npv(value_rate=ValueSchedule(...), escalation=, rates=, inflation=)`; `escalation=` beside `discount=` in `repyability.schedule` |
| UC-58 | An outage's overhaul that leaves a part as if a given age, not by a Kijima factor | **Built** (simulated): a `"preventive"` spec's `"restore_to"`; `restore_to=0` is a renewal (exact where renewals are) |
| UC-62 | A service that does harm: a fitted restoration factor above 1, each service leaving the unit older than it was. What does it do to the failures, and is the service worth stopping? | **Built** (simulated): a `"preventive"` `"restoration"` with `q` above 1; `compare_npv` against no service |
| UC-63 | One pump's plan: its life and repairs, its tasks each in its own clock, and its prices, held as one object that a diagram's node takes | **Built**: `UnitPlan` with `AgePolicy`, `BlockPolicy`, `ConditionPolicy`, `CalendarReplacement`, `AtShutdown`, `Inspection`, `Upgrade`, `Restoration`, `ImperfectRepair` and `Prices`; a spec dict is its shorthand (`UnitPlan.from_spec`, `plan.spec()`) |
| UC-65 | One pump's plan valued alone: its NPV over eight years at 7%, its cash flows by year, its spread, its plan against what happened, its long-run cost and best interval; with its salvage, its prices' growth and its retirement on the plan | **Built**: `UnitPlan.expected_npv`, `cash_flows`, `npv_distribution`, `plan_vs_actual`, `expected_cost`, `expected_cost_rate`, `events_by_period`, `failures_by_period`, `optimal_replacement_interval`, `optimal_inspection_interval` and `analysis_routes`, each its one-node diagram's (`UnitPlan.diagram`), on a `CalendarContext` if given; `UnitPlan(salvage=, escalation=, retire_at=)` and the spec keys of the same names |
| UC-59 | A part renewed at a short stop every month and again at the long yearly overhaul, each renewal as long as its shut | **Built**: a `"preventive"` spec's `"shutdown"` as a list of shutdowns, exact over a horizon |
| UC-60 | A part upgraded in year 3 and block-replaced before and after, the new model on the same interval | **Built**: an `"upgrade"` with a `"preventive"` block policy, numerical over a horizon and simulated |
| UC-72 | A part upgraded in year 3 that is age-replaced or tested, swapped at the upgrade or kept in service until it next comes out | **Built**: an `"upgrade"` with a `"preventive"` age policy or an `"inspection"`, numerical over a horizon (tests and repairs in no time) and simulated; `"replace_in_service": False` simulated |
| UC-66 | A pump whose seal is a redundant pair, replaced whole when both have failed: its life is a diagram | **Built**: a `NonRepairableRBD` as a life wherever a life is taken (`UnitPlan`, a spec's `"reliability"`, `NonRepairable`), by its functions; `NonRepairableRBD.qf` inverts its `ff` to near machine precision |
| UC-64 | What a part costs per year once upgraded: the long run of the diagram with the new life, the time before the upgrade ignored | **Built**: the long-run methods of a diagram with an `"upgrade"` give the values of the diagram after it |
| UC-68 | "95% sure": the least spend that keeps a bad year's downtime (hours or cost) under a limit, with the frontier and both planned and unplanned downtime | **Built** (simulated): `RepairableRBD.least_spend_for_downtime` and `UnitPlan.least_spend_for_downtime`, with redundancy, interval, test-interval, upgrade, crew and call-out levers, and parameter uncertainty (`uncertainty=`); risk measures in `repyability.value._risk` |
| UC-27 | A fleet whose units differ (lemons fail more and are written off sooner): the cheapest rule for a new unit (keep repairing, replace at an age, replace at a count of failures), and for one unit that has just failed, repair it again or replace it | **Built**: `RepairOrReplace` (`fleet_policy`, `unit_decision`, `break_even_frailty`), from a surpyval joint frailty model; numerical, bands from parameter draws |
| UC-73 | A unit that fails on demand, called on at set times (planned starts, a recurrence rule) or at a rate that changes over the year: its availability, failures and cost over a window | **Built**: a spec's (and a `UnitPlan`'s) `"demand_times"`, exact over a window and simulated; `"demand_plan"`, exact over a window where repairs take no time and simulated otherwise; `"demand_basis"` for demands counted in operating time |

## How a use case is added

1. **Write it down first**, using the template below, with the status
   *proposed*. Say who asks, in their own words, what they know and what
   they want back.
2. **Answer it by hand with what RePyability has today**, if it can be
   answered at all. Record the script and its output under *Today*. This
   shows exactly what is missing, and gives the numbers the new code must
   reproduce.
3. **Decide how it is solved**: the method, and whether it is exact,
   numerical or simulated (exact by default, simulation on request, as
   everywhere in RePyability). Note the choices made and why.
4. **Build the smallest thing that answers it**, with tests that check it
   against the hand-worked answer and an independent check (a closed
   form, the simulation, or a sum that must hold). Mark it *built*, with
   the pull request.

Keep to the project's rules (`CLAUDE.md`):
- Models come in already fitted; fitting stays in surpyval.
- Results are plain data (`results._ResultMapping`, with `to_dict()`),
  never plots: the Reliafy app draws them.
- Model time, not dates: dates go through `repyability.schedule`, whose
  use cases are in its own README.
- Inputs are checked where they are given (`utils.checks`).

## Template

```markdown
### UC-n: <the question, in the asker's words>

- **Status:** proposed | answered by hand | built (#PR)
- **Who asks, and why:**
- **What they know:** (the diagram, the components' states now, the plan, costs, ...)
- **What they want back:** (the shape of the answer: a table by window, a number)
- **How it is solved:** (the method; exact, numerical or simulated; the existing code it builds on)
- **Today:** (how close RePyability gets now: a script and its output)
- **Checks:** (how we know the answer is right)
- **Learned / open questions:**
```

## Use cases

### UC-1: "What maintenance work is coming on this system, in each of the next twelve windows of 730 hours, from its state now?"

(First asked month by month; months, as dates, are Reliafy's, and a
window of 730 hours stands for a month here.)

- **Status:** built: `RepairableRBD.expected_events_by_window(ends, state=)`
  and `EventsByWindow` (#375), each window's counts in one call (the
  differences below, a hair below 0 shown as 0). The record below is
  the hand-worked answer it was checked against.
- **Who asks, and why:** a planner building the next year's work and
  budget for one system already in service.
- **What they know:** the system's diagram, each component's fitted life
  and repair models and its maintenance (age replacement, block
  replacement, proof tests of hidden failures), and each component's
  state now: its age, and how far it is into its block or test interval.
- **What they want back:** in each window, the expected number of
  preventive replacements, corrective repairs and tests, by component,
  and of planned system outages.
- **How it is solved:** `RepairableRBD.expected_events(t, state=...)`
  gives each count over `[0, t)` from the state, exactly (no simulation),
  counting the events due at exact times (an age, a block replacement)
  exactly. A window's expected count is the difference of the counts at
  its two ends, as expectations add over windows.
- **Today (by hand):**

  ```python
  import numpy as np
  import surpyval as surv
  from repyability import NodeState, RepairableRBD

  window = 730.0  # hours
  fixed = surv.ExactEventTime.from_params
  rbd = RepairableRBD(
      [("s", "pump"), ("pump", "valve"), ("valve", "sensor"), ("sensor", "t")],
      {
          "pump": {   # replaced at age 6000 h, 4 h of work
              "reliability": surv.Weibull.from_params([9000.0, 2.5]),
              "repairability": surv.LogNormal.from_params([np.log(8.0), 0.5]),
              "preventive": {"interval": 6000.0, "duration": fixed([4.0])},
          },
          "valve": {  # replaced every 8760 h whatever happens, 6 h of work
              "reliability": surv.Weibull.from_params([20000.0, 1.5]),
              "repairability": surv.Exponential.from_params([1 / 12.0]),
              "preventive": {"interval": 8760.0, "policy": "block",
                             "duration": fixed([6.0])},
          },
          "sensor": { # hidden failures, proof-tested every 4380 h
              "reliability": surv.Exponential.from_params([1 / 50000.0]),
              "repairability": "instant",
              "inspection": {"interval": 4380.0},
          },
      },
  )
  state = {
      "pump": NodeState(age=5200.0),
      "valve": NodeState(age=7000.0, since_scheduled=7000.0),
      "sensor": NodeState(age=3000.0, since_scheduled=3000.0),
  }
  ends = window * np.arange(1, 13)
  events = rbd.expected_events(ends, state=state)
  by_window = lambda cumulative: np.diff(cumulative, prepend=0.0)
  by_window(events.node_preventive["pump"])
  ```

  Expected counts in each window of 730 hours (rounded):

  | Window | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Pump replaced (age 6000 h) | 0 | 0.90 | 0 | 0 | 0 | 0 | 0 | 0 | 0.05 | 0.65 | 0.00 | 0.01 |
  | Valve replaced (block) | 0 | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | Sensor tested | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
  | Pump repaired | 0.09 | 0.01 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.07 | 0.07 | 0.03 | 0.01 | 0.02 |
  | Valve repaired | 0.03 | 0.03 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
  | Sensor failure found at a test | 0 | 0.08 | 0 | 0 | 0 | 0 | 0 | 0.08 | 0 | 0 | 0 | 0 |
  | System planned outages | 0 | 0.83 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0.05 | 0.63 | 0.00 | 0.01 |

  How to read it:
  - **The pump** reaches age 6000 h in window 2 and is replaced then,
    unless it failed first: 0.90 is the chance it survives the 800 h from
    5200 h to 6000 h. Its replacement then reaches 6000 h in windows 9
    to 12, depending on when it was put in.
  - **The valve** is replaced every 8760 hours whatever happens (block
    replacement), so its replacement is certain, in window 3.
  - **The sensor** is tested in windows 2 and 8.
- **Checks:** the windows add up to `expected_events` over the whole
  span. The block replacements and tests land where their phases put
  them, worked out by hand. The age replacement's first chance matches
  `R(6000) / R(5200)`.
- **Learned / open questions:**
  - **A task and a system outage are different counts.** In window 2 the
    pump's replacement is 0.90 likely, but the system's planned outages
    are 0.83. The sensor's failures are hidden, so with probability about
    0.07 it has already failed unseen, and the series system is already
    down when the pump work starts: that work causes no new outage. The
    forecast must say which it reports, and explain this.
  - **Differencing a numerical count can give tiny negatives**
    (`-1e-12`) in a window with no events, to be shown as 0, without
    changing the sums.
  - By hand it means differencing and picking the node dictionaries
    apart. What RePyability could give is the counts in consecutive
    windows from a state in one call; months, as dates, are Reliafy's.


### UC-5: "My bearing is 224 days old and is replaced when it fails. What's the chance it fails in the next 82 days, how many replacements should I expect, and at $400 each, what will they cost?"

(First asked with dates: installed on 1 March 2026, from 11 October to
the end of 2026. Turning dates into an age and a window is Reliafy's; the
question here is the model-time one it becomes.)

- **Status:** built: `NonRepairable(..., unplanned_cost=)` and its
  `failures(t, age=)`, giving a `FailureForecast`.
- **Who asks, and why:** a planner with a non-repairable part (a bearing)
  run to failure and replaced, budgeting ahead.
- **What they know:** the bearing's life, a Weibull with a scale of 365
  days and a shape of 2 (fitted in surpyval); the current one is 224
  days old and working; a failed one is replaced with a new one, for
  $400.
- **The answer:** over the next 82 days, **a 27.8% chance it fails, 0.283
  replacements expected (a new bearing can fail too), $113.20.**
- **How it is solved:** in model time:
  - the chance it fails: the life conditioned on the age reached,
    `1 - R(age + t) / R(age)`, exact;
  - the expected replacements: the renewal count from that age, as
    `RepairableRBD.expected_events` works it out for the part as a
    component started from its state (exact, to about 1e-7; each
    replacement taking the part's `time_to_replace`).
- **As built:**

  ```python
  bearing = NonRepairable(surv.Weibull.from_params([365.0, 2.0]), unplanned_cost=400.0)
  forecast = bearing.failures(82.0, age=224.0)
  forecast.probability, forecast.expected_failures, forecast.expected_cost
  # 0.2784, 0.2830, 113.20
  ```

  First built with dates, as a `Component` (a name, a life, an installed
  date and a cost each failure); that overlapped `NonRepairable`, which
  is the part (its life, replacement time and costs), and dates are
  Reliafy's, so it became a method of the part in model time, and
  `NonRepairable` takes `planned_cost` and `unplanned_cost` by name.
- **Checks:** `test_non_repairable_failures.py`: the UC-5 answers against
  the closed form and a plain simulation; an exponential life (its rate
  times the window, whatever its age); Weibull units of several ages and
  windows, and replacements that take time, against the simulation;
  several windows at once; the input checks and the costs.

### UC-6: "Right after a failure, what's the NPV of skipping the next planned replacement?"

- **Status:** built: `NonRepairable.skip_next_replacement` and
  `SkipValue` (`repyability/maintenance/_skip.py`, #350).
- **Who asks, and why:** the planner of UC-5, now replacing the bearing on
  a plan as well as when it fails. A planned replacement costs less than
  an unplanned one, but replaces a working bearing. Right after a
  failure, the new bearing makes the next planned replacement look
  wasteful: is it worth skipping?
- **What they know (the maintainer chose these numbers):** the bearing of
  UC-5 (Weibull, scale 365 days, shape 2); a plan every 180 days; $150
  for a planned replacement and $400 for one at a failure; a discount
  rate of 8% a year. It is asked right after a failure, a new bearing
  just in.
- **What they want back:** the two options compared, each a
  present-value cost, and the NPV of skipping, as what it saves
  (positive: skip it), net of the failures it adds.
- **The option the maintainer named: does the plan reset?**
  - **The plan resets** (age replacement): the next planned replacement
    is an interval after the last replacement, planned or not, so right
    after a failure it is 180 days away. Skipping it, the bearing runs
    to a failure or to 360 days. Every replacement starts the same
    process afresh, so the two choices are the same process from the
    bearing's replacement on, but at different times: the NPV needs the
    discounted value of the plan from new.
  - **The plan does not reset** (fixed intervals, block replacement):
    planned replacements fall every 180 days whatever happens. Right
    after a failure, the next may be any number of days away, up to 180;
    a failure just before it leaves a nearly new bearing to be replaced.
    Skipping it, the bearing runs on to a failure or the following one.
    Both choices replace at the following one, so they differ only
    before it.
- **The answer:**
  - **Resetting plan: skipping saves $26.56.**
  - **Plan on fixed intervals:** it depends on the days to the next
    planned replacement:

    | Days to the next planned replacement | 1 | 10 | 30 | 60 | 90 | 120 | 150 | 180 |
    |---|---|---|---|---|---|---|---|---|
    | NPV of skipping ($) | 149.06 | 140.73 | 123.08 | 99.06 | 78.21 | 60.68 | 46.42 | 35.26 |

  Skipping pays every time: a 180-day plan is too frequent for this
  bearing. `NonRepairable.optimal_replacement_policy()` puts the best
  age at about 298 days (undiscounted). On a plan every 365 days the
  answer changes sign: skipping pays when the next is less than
  about 261 days away (the failure came at least about 104 days after
  the last planned replacement), and costs up to about $2 otherwise.
- **How it is solved:** exactly, with no simulation:
  - **Resetting:** renewal-reward with discounting. With `M(s)` the
    expected discounted length of a cycle cut off at age `s` (the
    integral of `exp(-r x) R(x)` to `s`) and `A(s)` its expected
    discounted cost, the plan from new is worth `A(T) / (r M(T))`, and
    `NPV = A(T) - A(2T) + A(T) (M(2T) - M(T)) / M(T)`. The discount rate
    cancels from the last term, so it holds undiscounted too (the
    classic comparison: the cycle cost saved, less the cost rate over
    the time added). `M` is integrated by the Gauss-Kronrod rule that
    the MTTF uses (`_mean_lifetime._integrate`).
  - **Fixed intervals:** `NPV = planned exp(-r d) + exp(-r d) F(T) - (F(d + T) -
    F(d))`, `d` the days to the next planned replacement and `F(t)` the expected
    discounted cost of a new bearing's failures (and its replacements')
    in `[0, t)`: `RepairableRBD.expected_cost(t, discount_rate=r)`.
  - **The two costs compared.** On fixed intervals, each choice's cost
    to the following planned replacement: done, `planned exp(-r d) +
    exp(-r d) F(T) + F(d)`; skipped, `F(d + T)`. On a resetting plan with
    a discount rate, the present value of the whole future under each:
    done, the plan from new, `A(T) / (r M(T))`; skipped, the longer first
    cycle and then the plan from new, `A(2T) + (1 - r M(2T)) A(T) / (r
    M(T))`. Undiscounted those are infinite, and `replace` and `skip`
    are `inf` (one meaning for the fields whatever the rate: the whole
    future's present value; from the 0.14 persona review, where they
    had shown the first cycles' comparison terms, 0 and -26.33); only
    the NPV is finite, the classic comparison: each first cycle's cost
    less its length at the plan's cost rate `g = A(T) / M(T)`, `A(T) -
    g M(T)` (0, the plan's own) against `A(2T) - g M(2T)`. Either way
    the difference is the NPV above. A rate that discounts the costs
    away (0.08 given as a rate a day) is warned of on both plans, by
    the check a `RepairableRBD`'s costs make (#231).
  - **The discount rate** is continuous, per unit of the life's time, as
    `RepairableRBD.expected_cost` takes it (0 by default): for 8% a year
    with a life in days, `r = log(1.08) / 365.25`, so a cost a year away
    is worth `1 / 1.08`. Reliafy turns a rate a year into it.
- **Today (before this use case):** the fixed-interval plan by hand from three
  `expected_cost` calls and the formula above; the resetting plan not
  at all (`NonRepairable` prices age replacement only in the long run,
  undiscounted).
- **Design, as built:** a method of the part, in the life's time unit,
  with the part's own costs:

  ```python
  import math
  from repyability import NonRepairable

  bearing = NonRepairable(surv.Weibull.from_params([365.0, 2.0]), planned_cost=150, unplanned_cost=400)
  a_day = math.log(1.08) / 365.25
  value = bearing.skip_next_replacement(180, resets=True, discount_rate=a_day)
  value.replace, value.skip, value.npv, value.best      # 5735.39, 5708.84, 26.56, "skip"

  value = bearing.skip_next_replacement(180, resets=False, time_to_next=30, discount_rate=a_day)
  value.npv                                             # 123.08
  ```

  `resets` has no default: the two plans answer differently, so it is
  always said. A plan on fixed intervals needs `time_to_next`; a
  resetting one refuses it (its next planned replacement is always
  `every` away). `SkipValue` carries `resets`, `every`, `time_to_next`,
  `discount_rate`, `replace`, `skip`, `npv = replace - skip` and `best`.
  It was first built as a `ReplacementPlan(life, every, planned_cost,
  unplanned_cost, resets=)` with a `skip_next(discount_rate,
  days_to_next=)` in days and a rate a year, and reworked before release
  (#350): UC-12's `optimal_next_replacement` answers the long-run,
  undiscounted question of a part with no plan in force, and UC-6 the
  discounted one about a plan in force, so each keeps a method of the
  part.
- **Checks:** `test_maintenance_skip.py` checks:
  - the answers against plain simulations of both plans (the resetting
    one over 300 years, each choice's present value too);
  - the two costs of a fixed plan against the part's failures over the
    windows, and of a resetting plan against an exponential life's closed
    forms;
  - an exponential life against both plans' closed forms, at 0%, 8% and
    50% a year;
  - the resetting plan with no discounting against the classic
    undiscounted comparison, and a tiny rate next to it;
  - the 180-day plan's NPVs falling with the time to the next one, and
    the 365-day plan's change of sign;
  - the same answer with the life in hours and the rate an hour;
  - the input checks, and that the costs are the part's own.
- **Learned / open questions:**
  - **The plan itself may be the question.** With these costs a 180-day
    plan is too frequent, so skipping always pays. Comparing the plan
    with the best one (discounted), or the latest day at which skipping
    a fixed-interval replacement still pays, are natural next use cases.
  - **Model time, not dates.** "Right after a failure" needs only the
    time to the next planned replacement; dates, and a rate a year, are
    Reliafy's.
  - **Replacements take no time.**
  - **The API** (#350): the maintainer did not like `skip_next` on a plan
    object of its own. It is now a comparison of the two options, each a
    present-value cost, from an explicit state (right after a failure),
    on the part (`NonRepairable`). Deferring, or moving to a shut, are
    UC-12's and UC-13's.

- **Extended (compared from any state):**
  `NonRepairable.compare_replacement_options(age, due_in, every, *,
  defer_by=, shuts=, discount_rate=)` replaces the one-purpose
  comparison with every option side by side, as present-value costs from
  the part's age now, on a plan of fixed times: replace as planned,
  replace now, defer by given amounts, move to a shut (a time and its own
  price), or skip. Each is costed to the following planned replacement,
  where every option has a new part in: `F_age(u) + exp(-r u) (price +
  F_0(horizon - u))`, with `F_a(t)` a part `a` old's failures' discounted
  cost (exact, a `RepairableRBD` component). For a new part it agrees with
  `skip_next_replacement` to 1e-9 (the table above), and every option
  matches a seeded simulation of that plan within four standard errors.
  Replacements that reset (age replacement) are not compared here: the
  options shift the later replacements, which `skip_next_replacement`
  (`resets=True`) works out. This is the first RCM calculator
  (`NonRepairable.compare_replacement_options` and `scheduled_replacement`
  share its numbers).

### UC-7: "Replacing the bearing at a shut (every 180 days) costs $100, a planned replacement between shuts $150, and one at a failure $400. Which replacement policy is cheapest in the long run?"

- **Status:** built as `replacement_policies(life, shut_every=,
  shut_cost=, planned_cost=, failure_cost=)` (#339), then removed before
  release: UC-13's `NonRepairable.optimal_calendar_policy` answers it (one
  shut type, a planned replacement on any day, a failure), finding the best
  policy of any form rather than the best of four families. The record
  below is kept for what it found.
- **Who asks, and why:** the planner of UC-6. The plant shuts every 180
  days; replacing in a shut is cheaper than a planned replacement that
  needs its own outage, which is cheaper than a failure.
- **What they know:** the bearing of UC-5 (Weibull, scale 365 days,
  shape 2), the shut interval and the three prices.
- **What they want back:** each policy's long-run cost a day, and the
  best.
- **The policies** (the last holds the others):
  1. run to failure;
  2. age replacement between shuts at age τ (the plan resets);
  3. at every shut, or every second shut (block replacement);
  4. at each shut, only if the bearing is at least *a* days old;
  5. both: at a shut if at least *a* old, or between shuts at age τ.
- **The answer:**

  | Policy | Cost a day |
  |---|---|
  | Run to failure | 1.237 |
  | Age replacement between shuts, best τ about 300 days | 1.117 |
  | Every second shut | 1.098 |
  | Every shut | 1.056 |
  | **At each shut, if at least about 138 days old** | **0.990** |

  Any *a* from 120 to 180 days costs within 0.1% of the best. A
  replacement between shuts never pays at these prices (policy 5's best
  τ is "never"): the shuts come often enough.
- **How it is solved:**
  - policies 1 to 3: a one-component `RepairableRBD`'s
    `expected_cost_rate()`, exact;
  - policy 4: exactly too, through replacement on condition, which
    replaces at an inspection a unit more likely than a threshold to fail
    before the next. For a hazard that rises (as this Weibull's), that is
    a unit at least *a* old, at the threshold `1 - R(a + 180) / R(a)`;
    the best *a* by a one-dimensional search;
  - policy 5: simulated by hand (RePyability's policies do not combine a
    condition at shuts with an age between them).
- **Checks:** every cost rate against a plain simulation of the policy
  (which runs about 0.001 a day low, from stopping at 150,000 days: run
  to failure 1.2354 against the exact 1.2366).
- **Learned:**
  - **It agrees with UC-6.** With the shut price, UC-6's value of
    skipping the next shut right after a failure changes sign when the
    shut is about 145 days away (discounted at 0.021% a day, 8% a year;
    142 undiscounted), next to the best *a*.
  - **An age, not a condition.** A rule "at least *a* old at a shut"
    should be given as an age for any life: where the hazard does not
    rise, an age and a condition threshold differ.
  - Policy 5 exactly needs a Markov chain on how far into the shut
    interval each new bearing went in, on a grid, if a case needs it.
- **As built:** the maintainer asked for the tools to find the best policy
  of four families: run to failure, every *d* days between shuts, every
  *z*-th shut, and every *x*-th shut as long as older than *y*.

  ```python
  import surpyval as surv
  from repyability import replacement_policies

  found = replacement_policies(
      surv.Weibull.from_params([365.0, 2.0]),
      shut_every=180.0, shut_cost=100.0, planned_cost=150.0, failure_cost=400.0,
  )
  found.run_to_failure      # 1.2366 a day
  found.between_shuts       # at age 297.6, 1.1170
  found.every_shut          # every shut (z = 1), 1.0555
  found.at_a_shut_if_older  # every shut (x = 1) if at least 137.9 old, 0.9903
  found.best                # the last; found.saving, 0.2463 a day
  ```

  - **One calculation for the shut families:** replacement on condition at
    inspections every *x* shuts, its rule an age (`older_than`, new to
    `_condition_replacement.condition_cycle`; 0 for every shut), exact for
    any life, not only one whose hazard rises. Between shuts:
    `NonRepairable.optimal_replacement_policy` at the planned price.
  - **The search:** every number of shuts, and for each the age on a grid
    of 8 a shut interval, refined around the best, up to the age a unit
    outlives with a chance of 0.1%. About 10 seconds for the bearing.
  - **Learned while building:** a rule that seldom replaces (every sixth
    shut, if older than 1,080 days) has cycles of many thousands of
    failures, and its cost a day is good only to a few parts in 10,000.
    With a shut dearer than a failure, such a rule "beat" running to
    failure by 0.01%, which cannot be (a planned replacement dearer than
    a failure never pays). So a planned policy is chosen only if it saves
    more than 0.1% against running to failure.
  - **Checks** (`test_maintenance_shuts.py`): the UC-7 figures; every shut
    against RePyability's block replacement (to 1e-6); three shut
    policies against a plain simulation; the age rule against the
    threshold rule for a rising hazard; an exponential life (no planned
    replacement pays); a shut dearer than a failure; the answer the same
    in days and in hours; JSON; the input checks.

### UC-8: "From the bearing's state now, what will each policy cost over the next *t* days, and how much could that vary?"

(First asked over a timeline of dates; the windows here are the days
from now to the end of each quarter, as Reliafy would turn them.)

- **Status:** built for one part under run to failure, age replacement
  and block replacement: `NonRepairable.cost_distribution(window, every,
  *, resets, age=, since_renewal=)` and `CostDistribution` (numerical by
  default, `method="simulate"` as the check); and under a calendar policy
  such as UC-7's best (at a shut if at least 138 days old), the
  `CalendarPolicy` of `optimal_calendar_policy`: `cost_distribution(window,
  calendar=policy, age=)`, numerical too. The answer by hand below is the
  check.
- **Who asks, and why:** the planner of UC-7: a long-run cost a day says
  which policy is cheapest, but a budget starts from the bearing as it
  is, and needs to know how far the spend can stray from the mean.
- **What they know:** UC-7's bearing, prices and policies; its state: 224
  days old, the next shut in 21 days (159 days into the 180-day shut
  interval).
- **What they want back:** the cost over windows of 82, 447, 813 and 1,086
  days from now: its mean and its spread (10th, 50th, 90th percentiles).
- **The answer by hand** (dollars, 20,000 simulations):

  **The best policy** (at a shut if at least 138 days old):

  | Window (days) | Mean | 10% | 50% | 90% |
  |---|---|---|---|---|
  | 82 | 133 | 100 | 100 | 100 |
  | 447 | 487 | 300 | 300 | 900 |
  | 813 | 846 | 500 | 800 | 1,400 |
  | 1,086 | 1,090 | 600 | 1,000 | 1,600 |

  **Over 1,086 days:**

  | Policy | Mean | 10% to 90% | Long run, a year |
  |---|---|---|---|
  | At a shut if at least 138 days old | 1,090 | 600 to 1,600 | 362 |
  | Every shut | 1,155 | 600 to 1,800 | 386 |
  | Run to failure | 1,373 | 800 to 2,000 | 452 |

  - **The first window costs more than the long run** (about $590 a year
    over the first 82 days, against $362): the bearing is old, and is
    almost sure to be replaced at the shut in 21 days.
  - **The spend comes in steps** ($100 at a shut, $400 at a failure), so
    the percentiles jump.
  - **The best policy lowers the mean more than the spread.**
- **How it was solved by hand:**
  - **the mean:** `RepairableRBD.expected_cost(t, state=...)`, exact, from
    `NodeState(age=224, since_scheduled=159)`;
  - **the spread:** simulated: `RepairableRBD.cost(t, state=...,
    seed=...)` for each window with one seed. Simulation *r* draws the
    same lives whatever the window, so the runs to each window are the
    same histories cut off there.
- **How it is solved, as built:** the cost over the window is `cu N_f +
  cp N_p`, the failures and the planned replacements in it, so the joint
  distribution of the two counts gives everything: the mean, the standard
  deviation, every percentile, each outcome's chance and the chance of
  exceeding a budget.
  - **Age replacement and run to failure.** The part's history is a run
    of cycles, each ending in a failure (before age *T*) or a planned
    replacement (at exactly *T*); the part in now has a first cycle of its
    own, from its age. A planned cycle has a fixed length, so *i* failure
    cycles and *j* planned ones take the failure cycles' total plus *jT*
    whatever their order: the chance of *(i, j)* is a binomial count of
    orders times the chance that the failure cycles' total, shifted by
    *jT*, ends before the window's end with the next cycle running past
    it. The totals' distributions are convolutions of the failure cycle's,
    worked out on a grid (exact cell increments, the trapezoidal rule in
    the measure, error falling as the step squared), sized from the
    lives' spread, never in absolute time. Run to failure is *T*
    infinite.
  - **Block replacement.** The planned replacements fall at fixed times,
    so their number is known, and between them the part runs to failure
    from new: the failures are a sum of independent counts, one for each
    stretch (the first from the part's age), and their distribution is the
    stretches' convolved.
  - **Precision.** Worked out again on a grid twice as coarse; the
    largest change of any probability, plus the mass the counts leave
    out, is reported as `precision` (1e-9 to 3e-6 for the bearing; about
    three times the true error, which the exponential checks measure).
  - **Simulated** (`method="simulate"`, seeded): the part's cycles drawn
    from its life, each count's share of the simulations.
- **As built** (the bearing, 224 days old, the next shut in 21 days):

  ```python
  bearing = NonRepairable(surv.Weibull.from_params([365.0, 2.0]),
                          planned_cost=100.0, unplanned_cost=400.0)
  shuts = bearing.cost_distribution(1086.0, 180.0, resets=False,
                                    age=224.0, since_renewal=159.0, budget=1500.0)
  shuts.mean, shuts.percentiles, shuts.probability_above_budget
  # 1155.0, [600, 1000, 1800], 0.151
  ```

  | Window (days) | Every shut: mean | 10% | 50% | 90% | Run to failure: mean | 10% | 50% | 90% |
  |---|---|---|---|---|---|---|---|---|
  | 82 | 139.6 | 100 | 100 | 100 | 113.2 | 0 | 0 | 400 |
  | 447 | 521.5 | 300 | 300 | 700 | 583.6 | 400 | 400 | 800 |
  | 813 | 903.9 | 500 | 900 | 1,300 | 1,035.3 | 800 | 1,200 | 1,600 |
  | 1,086 | 1,155.0 | 600 | 1,000 | 1,800 | 1,372.9 | 800 | 1,200 | 2,000 |

  Over 1,086 days, the shut policy makes six planned replacements for
  certain, and 0, 1, 2 or 3 failures with chances 22%, 36%, 26% and 11%;
  the chance of spending more than $1,500 is 15% with the shuts and 44%
  run to failure.
- **Checks:** by hand, the simulated means agreed with the exact ones
  within $2 (20,000 simulations each). As built
  (`test_maintenance_cost_spread.py`): the figures by hand over 1,086 days
  (the means and percentiles of every shut and of run to failure);
  an exponential life run to failure (failures Poisson(λW) from any age:
  each probability within the precision reported, the mean, variance, P90
  and the chance of exceeding a budget); an exponential life under age
  replacement with no chance of a planned replacement in the window
  (Poisson) and with exactly one (`P(k, 1) = exp(-λτ) pois(k, λ(W - τ))`,
  `P(k, 0) = pois(k, λW) - P(k, 1)`); a life close to fixed (Weibull of
  shape 60: two failures nearly surely); from an aged start, the chance of
  a failure is the conditional life's, and `P(0, 0) = R(a + W) / R(a)`
  under each policy; the mean against the module's expected cost
  (`failures` and a one-component `RepairableRBD`'s `expected_cost`) for
  each policy, at three ages and two windows (to 2e-6); the simulation
  against the numerical route (every probability within four standard
  errors); the same probabilities with time in units from 1e-6 to 1e6;
  JSON; the input checks.
- **Learned:**
  - **The steps matter.** Every shut's 90th percentile over 447 days is
    $700, three planned replacements and one failure, while the median
    is three replacements alone: a budget range from the percentiles
    alone hides the jump, which the outcomes show.
  - **By hand,** each window's spread had to come from the same
    simulations, or the 90th percentiles of longer windows need not even
    rise; the seeded streams gave that for free. As built, each window is
    worked out exactly on its own, so the percentiles rise with the
    window by construction.
- **The best policy, as built** (UC-7's: a calendar policy). The asker
  has the policy from `optimal_calendar_policy` and wants the range its
  budget should carry, as for the simpler policies.
  - **Why the other policies' method does not carry over.** Under age
    replacement every cycle of a new part has one distribution, so the
    counts are convolutions of it. Under a calendar policy a part put in
    at time `s` is next replaced as planned at a time `pi(s)` the policy
    fixes (the first opportunity at which it is at least the threshold),
    so cycles differ by when they start.
  - **How it is solved:** on a grid of half the policy's own step, the
    expected renewals at each grid time are kept by the counts so far:
    failures, and planned replacements by kind of opportunity (their
    prices differ). The renewals of a count pass a planned replacement to
    the count with one more of its kind, and their failures to the count
    with one more failure, so the counts are taken in order, none needing
    a sweep in time: a count's failures by cell are one convolution (FFT)
    of its renewals for each value `pi(s)` takes, cut there. Before the
    window nothing is counted, and the renewals are followed forward in
    time (the renewal measure). A failure in a cell renews half at each
    end of it (the trapezoidal rule), and a renewal at a grid time takes
    half of each neighbouring cell's `pi`, so a threshold on a grid time
    splits it; the error falls as the step squared. Worked out again on
    the policy's step, the largest change of a probability (plus any
    mass left out) is `precision`. Numerical; simulated with
    `method="simulate"` (each cycle's planned time asked of the policy,
    as a `RepairableRBD` asks it).
  - **The answer** (the policy worked out by `optimal_calendar_policy` for
    a shut every 180 days, the next in 21, at $100, a failure at $400):

    ```python
    plant = Calendar([Opportunity("shut", 100.0, period=180.0, offset=21.0)],
                     anytime=None)
    best = bearing.optimal_calendar_policy(plant)     # at a shut if >= 138 days
    spend = bearing.cost_distribution(1086.0, calendar=best, age=224.0,
                                      budget=1500.0)
    spend.mean, spend.percentiles, spend.probability_above_budget
    # 1090.0, [600, 1000, 1600], 0.150
    ```

    | Window (days) | Mean | 10% | 50% | 90% | By hand: mean |
    |---|---|---|---|---|---|
    | 82 | 132.7 | 100 | 100 | 100 | 133 |
    | 447 | 486.5 | 300 | 300 | 900 | 487 |
    | 813 | 845.8 | 500 | 800 | 1,400 | 846 |
    | 1,086 | 1,090.0 | 600 | 1,000 | 1,600 | 1,090 |

    Every percentile is the one by hand. Over 1,086 days the most likely
    outcomes are one failure and five shut replacements ($900, 26%) and
    six shut replacements alone ($600, 22%); `outcomes()` lists them by
    kind of opportunity, `{"failures": 1, "planned": {"shut": 5}, ...}`.
    About half a second a window.
  - **Checks** (`test_maintenance_calendar_spread.py`): the figures by
    hand; an exponential life (no planned replacement pays, so the policy
    never replaces: Poisson failures in every window, within the
    precision); a part due at the next shut (`S(21)` of no event, then a
    replacement there); the method's own simulation, and a one-component
    `RepairableRBD` on the same policy simulated by its event loop (every
    probability within four standard errors; seeds repeat); a calendar
    of two kinds (each outcome priced by kind, the counts by kind adding
    up to the counts); the same probabilities in hours; JSON; refusals (a
    policy with `every`, one for another life; a policy worked out by
    integration has no grid, so only `method="simulate"`).
  - **Learned:** the calendar policy's spread is about as wide as every
    shut's (10% to 90%, $600 to $1,600 against $600 to $1,800) for a mean
    $65 lower; and, as with every shut, the steps decide it: a 90th
    percentile of $900 over 447 days is one failure on top of the planned
    shuts.

### UC-9: "What will the bearing cost in each of a run of consecutive windows from now, and how often does each outcome happen in each?"

(First asked by financial year, July to June, from 11 October 2026; the
windows here, 0 to 263, 263 to 629, 629 to 994 and 994 to 1,359 days
from now, are those years as Reliafy would turn them. Summing by
financial year is Reliafy's.)

- **Status:** built for one part under run to failure, age replacement,
  block replacement and a calendar policy such as UC-7's best:
  `NonRepairable.cost_by_window(ends, every, *, resets, age=,
  since_renewal=, calendar=)` and `WindowCosts`, numerical by default
  (`method="simulate"` as the check). A system's windows are simulated:
  `RepairableRBD.cost_by_window` and `CostByWindow` (#375) give each
  window's exact mean and its simulated spread, outcomes and chance of
  exceeding a budget from one run. The answer by hand below (the best
  policy, simulated) is the check of the calendar policy's.
- **Who asks, and why:** the planner of UC-8, passing a budget on: each
  window's own spend, not the running total, and what could happen in
  it.
- **What they know:** UC-8's bearing, its state and the best policy of
  UC-7.
- **The answer:** the share of simulations with each outcome in each
  window (1,000,000 plain simulations; RePyability's own agree within
  about 0.3 points):

  | What happens in the window | Cost | 0–263 | 263–629 | 629–994 | 994–1,359 |
  |---|---|---|---|---|---|
  | No failure, one shut replacement | $100 | – | 0.9% | 1.2% | 1.5% |
  | No failure, two shut replacements | $200 | 70.7% | 59.1% | 58.6% | 58.2% |
  | One failure, no shut replacement | $400 | – | 0.5% | 0.7% | 0.8% |
  | One failure, one shut replacement | $500 | 22.1% | 27.2% | 26.9% | 26.4% |
  | One failure, two shut replacements | $600 | 3.0% | 2.9% | 3.2% | 3.5% |
  | Two failures, no shut replacement | $800 | 1.4% | 4.8% | 4.6% | 4.5% |
  | Two failures, one shut replacement | $900 | 2.5% | 3.5% | 3.6% | 3.8% |
  | Three or more failures, or two and both shuts | $1,000+ | 0.3% | 1.2% | 1.2% | 1.2% |
  | Mean (exact) | | $307 | $359 | $359 | $359 |

  The chance of more than $200 (any failure) is 29% in the first window
  and 40% in each full one; of more than $600, 4% and 9%.
- **How it was solved by hand:** as UC-8, at each window's end: the means by
  differencing `expected_cost(t, state=...)`, exact; each simulation's
  cost in a window as its cost to the window's end less its cost to its
  start, from one seed (`cost(...).samples`), so every window's outcomes
  are of the same histories.
- **Checks by hand:** no simulation spends less than $0 in a window; the
  simulated means agree with the exact ones within $4.
- **Learned:**
  - **Costs in steps need their outcomes, not percentiles.** Here the
    10th, 50th and 90th percentiles read $200, $200, $600, and hide that
    60% of full windows cost exactly $200: give each outcome and its
    share, and the chance of exceeding a budget.
  - **Ranges do not add.** The four windows' 90th percentiles sum to
    $2,300; the 90th percentile of their total is $2,000.
  - What RePyability could give: each simulation's cost in each of a run
    of windows, from one run (by hand it took a run per window's end).
- **How it is solved, as built:** each window from the process through
  every earlier one, exactly: the windows are dependent (a failure in one
  renews the part for the next), so a window's distribution is not that
  of a part started afresh at its start.
  - **Run to failure and age replacement.** What happens in a window
    depends on the past only through its first event: after it a new
    part is in, and the rest is UC-8's cycles of new parts. The first
    event is the part's in service at the window's start, put in at the
    last renewal before it: the part in now (never renewed), the last of
    the planned replacements' chain from now (no failure yet: an atom of
    known time and chance), or a renewal after a failure, spread by the
    renewal measure (the expected number of renewals by each time, the
    same sums of cycle totals as UC-8's counts). UC-8's formulas then
    give the window's counts with that first cycle.
  - **Block replacement.** The planned replacements in each window are
    known; the failures are the stretch run to failure from the last
    planned replacement before the window, seen from its start as above,
    then each stretch after a planned replacement in it, from new,
    independent.
  - **Outcome classes:** nothing; planned work only (no failure: known
    work at a known price); one failure; several failures (each whatever
    the planned work beside it). They are what a planner acts on (the
    failures are the unplanned spend and the crew call-outs), and the
    full table of counts is there beside them (`outcomes(window)`).
  - The whole span's own distribution is given too (`total`, UC-8's),
    since ranges do not add.
- **As built** (the bearing of UC-8 replaced at every shut, and run to
  failure, over the four windows):

  ```python
  years = bearing.cost_by_window([263.0, 629.0, 994.0, 1359.0], 180.0,
                                 resets=False, age=224.0, since_renewal=159.0,
                                 budget=600.0)
  years.mean, years.outcome_probabilities, years.probability_above_budget
  ```

  | Every shut | 0–263 | 263–629 | 629–994 | 994–1,359 |
  |---|---|---|---|---|
  | Planned only (two shuts, $200) | 70.8% | 61.3% | 61.3% | 61.3% |
  | One failure ($600) | 26.2% | 32.4% | 32.3% | 32.3% |
  | Several failures ($1,000+) | 3.1% | 6.3% | 6.3% | 6.4% |
  | Mean | $330.0 | $382.3 | $382.1 | $382.2 |
  | 10%, 50%, 90% | 200, 200, 600 | 200, 200, 600 | 200, 200, 600 | 200, 200, 600 |

  | Run to failure | 0–263 | 263–629 | 629–994 | 994–1,359 |
  |---|---|---|---|---|
  | Nothing | 24.6% | 15.9% | 15.7% | 15.7% |
  | One failure ($400) | 62.6% | 58.2% | 58.6% | 58.6% |
  | Several failures ($800+) | 12.8% | 25.9% | 25.7% | 25.7% |
  | Mean | $355.7 | $451.9 | $451.6 | $451.3 |

  Over the whole span, every shut costs $1,476.5 on average, 10% to 90%
  from $800 to $2,000, where the windows' 90th percentiles sum to $2,400;
  run to failure $1,710.5, $1,200 to $2,400 (the windows' sum: $3,200).
- **Checks** (`test_maintenance_window_outcomes.py`): an exponential life
  run to failure (independent Poisson(λ w) windows from any age, and the
  outcome classes' closed forms), and under block replacement (Poisson
  failures, the planned ones known, one at the end of a window falling in
  the next); for each policy, the windows' means adding up to the whole
  span's (to 2e-6), the total being UC-8's, each window's and each
  outcome row's chances adding up to 1, and the first window being UC-8's
  within the precisions; each window's expected failures and planned
  replacements against a one-component `RepairableRBD`'s
  `expected_events` differenced (another method, through the windows'
  dependence); the first window from an aged start, `R(a + e_1) / R(a)`;
  the simulation, window by window and in total, within four standard
  errors; units from 1e-6 to 1e6; JSON; the windows' input checks.
- **The best policy, as built** (`calendar=`, UC-8's method window by
  window: each window's counts start from the renewals the process
  brings to its start, uncounted before it):

  ```python
  years = bearing.cost_by_window([263.0, 629.0, 994.0, 1359.0],
                                 calendar=best, age=224.0, budget=600.0)
  ```

  | What happens in the window | Cost | 0–263 | 263–629 | 629–994 | 994–1,359 |
  |---|---|---|---|---|---|
  | No failure, one shut replacement | $100 | – | 0.9% | 1.2% | 1.5% |
  | No failure, two shut replacements | $200 | 70.8% | 59.0% | 58.6% | 58.2% |
  | One failure, no shut replacement | $400 | – | 0.5% | 0.7% | 0.8% |
  | One failure, one shut replacement | $500 | 22.1% | 27.2% | 26.9% | 26.4% |
  | One failure, two shut replacements | $600 | 3.1% | 2.9% | 3.2% | 3.5% |
  | Two failures, no shut replacement | $800 | 1.4% | 4.8% | 4.6% | 4.5% |
  | Two failures, one shut replacement | $900 | 2.4% | 3.5% | 3.6% | 3.7% |
  | Mean | | $306.9 | $358.7 | $358.9 | $359.0 |

  Every share is within 0.1 points of the million simulations by hand,
  and every mean within $0.3 of the exact means by hand. The chance of
  more than $600 is 4.1% in the first window and 9.5% in each full one;
  over the whole span the mean is $1,383.5 and its 90th percentile
  $2,000, where the windows' 90th percentiles sum to $2,300, as by hand.
  Checks (`test_maintenance_calendar_spread.py`): the means by hand; the
  windows' means adding up to the span's; the first window being
  `cost_distribution`'s; the outcome classes adding up to 1; the
  simulation, window by window and in total; the exponential's
  independent Poisson windows.

### UC-11: "My pump's seal is replaced at every 500-hour service and whenever it fails. It has run 340 hours, and the next service is in 160. What's the chance it fails before each service, what will failures and replacements cost over the next 944 running hours, and is replacing it at every service worth it?"

(It comes from Reliafy's side: the pump's on and off times give its
running hours, so the seal's age (340 h, since it went in at the last
service), how far the pump is into its service interval (340 of 500 h)
and, through the operating schedule, how many running hours lie ahead
(944 h to the end of the year, at 16 hours a weekday). Turning the
answers back into dates is Reliafy's too.)

- **Status:** built: `NonRepairable.renewal_forecast(window, every, *,
  resets, age=, since_renewal=)` and `RenewalForecast` (see *As built*).
- **Who asks, and why:** a planner whose part is renewed by the
  maintenance as well as at failures, so its risk starts again at each
  service: what is the risk before the next one, and is the plan right?
- **What they know:**
  - the seal's life in running hours, fitted in surpyval from the pump's
    history: a Weibull with a scale of 6,000 h and a shape of 2.2;
  - the plan: a new seal at every service, every 500 running hours on
    the pump's meter (block replacement: the services do not move when a
    seal fails), for $200; a new seal at a failure, at once, for $900;
  - the state: the seal is 340 h old, and the next service is in 160 h.
- **What they want back:** the chance it fails before each service, the
  expected failures, replacements and cost over the hours ahead, and
  whether the plan is worth it.
- **The answer:**
  - **Before the next service** (160 h): a **0.24%** chance it fails,
    `1 - R(500) / R(340)`.
  - **Before each later service:** a new seal goes in at each, so the
    chance is the same in every interval, **0.42%**, `1 - R(500)`.
  - **Over the next 944 running hours** (two services): **0.0078 failures
    expected, 2 seals at services, $407** ($7 of failures, $400 of
    planned seals).
  - **Without the replacements at services**, the same seal from 340 h
    would fail in those 944 h with a chance of 3.1% ($28 expected); over
    5,000 h, 54%.
  - **The plan is not worth it.** Renewing at every service costs $0.41 a
    running hour; the best age replacement (`NonRepairable(...,
    planned_cost=200, unplanned_cost=900).optimal_replacement_policy()`)
    replaces at about 3,180 h, for $0.12 a running hour. With services
    every 500 h, the natural policy is UC-7's: renew at a service only if
    the seal is older than some age.
- **How it is solved:** in running hours, by today's methods:
  - the chances: the life conditioned on the age, exact;
  - the expected failures, planned replacements and cost: a
    one-component `RepairableRBD` with a block replacement every 500 h
    (`"preventive": {"interval": 500, "policy": "block", "cost": 200}`),
    `replace_cost` 900, started from `NodeState(age=340, since_scheduled=340)`:
    `expected_events(944, state=...)` and `expected_cost(944, state=...)`,
    exact;
  - the long-run cost: its `expected_cost_rate()`.
- **Today (by hand):**

  ```python
  import surpyval as surv
  from repyability import NodeState, RepairableRBD

  life = surv.Weibull.from_params([6000.0, 2.2])            # running hours
  rbd = RepairableRBD(
      [("s", "seal"), ("seal", "t")],
      {"seal": {"reliability": life, "repairability": "instant",
                "replace_cost": 900.0,
                "preventive": {"interval": 500.0, "policy": "block", "cost": 200.0}}},
  )
  state = {"seal": NodeState(age=340.0, since_scheduled=340.0)}
  rbd.expected_events(944.0, state=state).node_failures      # {'seal': 0.0078}
  rbd.expected_cost(944.0, state=state).mean                 # 407.06
  1 - life.sf(500) / life.sf(340)                            # 0.0024, before the next service
  ```

- **Checks:** the expected failures over 944 h against a plain simulation
  of the seal and its services (0.0080 ± 0.0001, against 0.0078).
- **Proposed design (to agree with the maintainer):** a part's risk and
  costs under a plan that renews it, on the part, in model time, beside
  `failures`: the plan (renew every so many hours, keeping to the
  interval or resetting at each renewal) and the state (the part's age,
  how far into the interval), giving the chance of a failure before each
  of the next renewals, and the expected failures, renewals and cost
  over a window. It shares its open questions with UC-6's rework (the
  plan as arguments or an object, the state given explicitly).
- **Learned:**
  - **Renewal at each service resets the risk:** the chance before a
    service is small and the same each time, where without it the risk
    grows with the seal's age.
  - **The answer can show the plan is wrong:** here a renewal at every
    service costs over three times the best age replacement, which is
    the question to ask next (UC-7's shape, at services).
  - **History in, risk out:** Reliafy's history gives the age and the
    time since the last service in running hours, RePyability the risk
    and cost in running hours, and the schedule turns them into dates.

- **As built (UC-11):** `NonRepairable.renewal_forecast` takes the
  window, the service interval, `resets` (said every time, as for UC-6:
  `False` keeps the services to fixed times, block replacement; `True`
  starts each interval at the last renewal, planned or not), the age and,
  for fixed times, how far into the interval the part is. It gives the
  chance of a failure before each service (the first from the age, each
  later one from a part new at the service before: `1 - R(every)`), the
  expected failures, renewals and cost over the window (a one-component
  `RepairableRBD`, exact), the same without renewals, the plan's long-run
  cost rate against running to failure and against the best age
  replacement, and `worth_it`. The seal's numbers above are the test:
  0.0024 and 0.0042, 0.00785 failures, 2 renewals, $407.06, 0.4076 a
  hour against 0.1198 at the best age, 3,179 h. Its expected failures
  match a seeded simulation of both plans (fixed and resetting) within
  four standard errors.

### UC-12: "The bearing is 200 days old. Replacing it now costs $150; a planned replacement any time costs $150 too, and one at a failure $400. Outages are scheduled in 150 and 330 days, at which replacing it costs $100. Do I replace it now?"

- **Status:** built: `NonRepairable.optimal_next_replacement(age,
  now_cost=, scheduled_outages=, policy=)` (see *As built*); `policy=`,
  the baseline as a `CalendarPolicy`, added by #348.
- **Who asks, and why:** a planner deciding on one part, today, from its
  age and the outages scheduled. The maintainer asked for this first, as
  the clean starting point for the shut questions (UC-7's policies are
  rules for every part; this is one decision for this part).
- **What they know:**
  - the part's life (UC-5's bearing: Weibull, scale 365 days, shape 2)
    and its age now;
  - the price of replacing it now (an outage today, an opportunity, or a
    planned job), of a planned replacement at any time, and of one at a
    failure;
  - the scheduled outages (shuts, turnarounds), each as a time from now
    and what replacing the part at it costs (Reliafy turns the dates
    into times).
- **What they want back:** replace now or not; what each choice costs;
  and the age at which replacing now becomes the best choice.
- **The choices**, for this part only: replace it now; keep it to one
  of the scheduled outages and replace it there; keep it to a planned
  replacement at a time between (the best time, found continuously);
  or run it to failure. In each, a failure first means a replacement at
  the failure price.
- **How it is solved:** each choice's expected cost until this part is
  replaced, less the value of the days it serves:

  `cost(u, price) = price · S(u) + failure · (1 - S(u)) - g · ∫₀ᵘ S`,

  where `S(u) = R(age + u) / R(age)` is the chance it is still working
  `u` from now, and `g`, the **baseline**, is what a day of the part's
  service is worth: the long-run cost a day of how the position is run.
  By default the age replacement's
  (`optimal_replacement_policy().cost_rate`, 1.117 a day for the
  bearing); given `policy=`, a `CalendarPolicy` (UC-13) or a
  `MaintenancePolicy`, its `cost_rate` (#348). Replacing now costs its
  price, and serves no days of this part. The best choice is the
  smallest.
- **Why the days are valued:** keeping the part means a new one, and
  every part after it, starts later; the days the old one serves are days
  a new one would otherwise serve, at the baseline's cost a day. Without
  `g`, a bearing 6 days old would be replaced: kept to the outage in 150
  days, 83% reach it ($100) and 17% fail ($400), $150 on average, as
  replacing now costs. With it, the 141 days it serves on average are
  worth 141 × 1.117 = $157.5, so keeping it costs -$7.5 against $150.
- **With no outages it is the textbook age replacement.** The best later
  replacement of a part of age *a* is at 297.6 - *a* days from now, and
  replacing now is the best from 297.6 days old: the optimal interval
  exactly. The baseline is that calculation's own cost rate: the optimal
  interval is the age at which a day's added risk, (400 - 150) × hazard,
  reaches it.
- **The answer:**

  | Choice | Age 100 | Age 200 | Age 300 |
  |---|---|---|---|
  | Replace now ($150) | 150.0 | 150.0 | **150.0** |
  | At the outage in 150 days ($100) | **55.3** | **110.0** | 154.7 |
  | At the outage in 330 days ($100) | 84.5 | 156.5 | 206.4 |
  | Planned, at the best time ($150) | 86.3 (in 198 days) | 133.9 (in 98) | 150.0 (now) |
  | Run to failure | 128.0 | 186.1 | 226.0 |

  At 200 days old: **keep it to the outage in 150 days**.

  Each later choice has its own break-even age, past which replacing now
  beats it; replacing now is the best once it beats them all:

  | Replacing now beats... | from age |
  |---|---|
  | running to failure | 133.3 days |
  | the outage in 330 days | 189.3 days |
  | the outage in 150 days | 288.4 days |
  | a planned replacement later | 297.6 days |

  So **replace now from 297.6 days old**, the optimal age (from 288.4,
  a planned replacement within days is cheaper by very little).
- **When cheap shuts keep coming, the baseline is the calendar policy's
  (#348).** With a shut every 180 days at $100 (UC-13's
  `optimal_calendar_policy`), a day of the bearing's service is worth
  0.990 rather than 1.117, so keeping it is dearer at every age and
  replacing now beats each later choice from a younger age. At 200 days
  old the outage in 150 days is still the best (124.6, against 150.0
  now, 178.5 at the outage in 330 days, 142.8 planned in 64 days and
  210.4 run to failure); the break-even ages:

  | Replacing now beats... | from age, age replacement's rate | with the shuts' |
  |---|---|---|
  | running to failure | 133.3 days | 86.4 days |
  | the outage in 330 days | 189.3 days | 153.0 days |
  | the outage in 150 days | 288.4 days | 256.1 days |
  | a planned replacement later | 297.6 days | 263.9 days |

  So with the shuts kept coming, **replace now from 263.9 days old**.
  The policy of a calendar with no opportunities (replacement on any
  day at the planned price) is age replacement, and gives the answer
  above: its rate is exact for its grid, and the break-even ages agree
  to 1e-4 days.
- **As built:**

  ```python
  import surpyval as surv
  from repyability import NonRepairable

  bearing = NonRepairable(
      surv.Weibull.from_params([365.0, 2.0]),
      planned_cost=150.0, unplanned_cost=400.0,
  )
  found = bearing.optimal_next_replacement(
      200.0, scheduled_outages=[(150.0, 100.0), (330.0, 100.0)]
  )
  found.best              # the outage in 150 days, 110.0
  found.now               # False
  found.choices           # now, each outage, planned (in 97.6 days), failure
  found.replace_from_age  # 297.6: replacing now beats every later choice
  found.cost_rate         # 1.117 a day: the baseline

  from repyability import Calendar, Opportunity

  shuts = bearing.optimal_calendar_policy(
      Calendar([Opportunity("shut", 100.0, period=180.0)])
  )
  found = bearing.optimal_next_replacement(
      200.0, scheduled_outages=[(150.0, 100.0), (330.0, 100.0)],
      policy=shuts,
  )
  found.cost_rate         # 0.990 a day: the shuts' policy's
  found.replace_from_age  # 263.9
  ```

  - **The maintainer's choices:** the name (`optimal_replacement_policy`
    is the rule for a part in general, a hypothetical one, and
    `optimal_next_replacement` the next replacement of the one in
    service); the outages as `scheduled_outages`, (time from now, price)
    pairs and nothing else; and the baseline the optimal policy: the
    replacement policy's, or the policy given (`policy=`, a
    `CalendarPolicy` or a `MaintenancePolicy`; its cost rate alone is
    refused, naming the policy to give; #348). A first version also
    took the regular shuts (`shut_every=`, `shut_cost=`) to value a day
    at `replacement_policies`' best (0.990 a day with shuts every 180
    days, which moves the break-even against the outage in 150 days to
    256 days): that doubled up on the outages, and assumed what comes
    after the listed ones; UC-13's calendar policy now says what the
    shuts are worth, and the outages listed stay the ones scheduled.
  - `now_cost` is the planned price unless given (an outage today, or a
    chance while the machine is apart, is cheaper). **A planned
    replacement is one later than now** (from the 0.14 persona review):
    where no later time beats replacing at once (a part past its
    optimal age), replacing at once is replacing now, at `now_cost`, so
    the planned choice is then replacing now (time 0, at `now_cost`,
    never the planned price at time 0 that `now_cost` contradicts),
    `now` is True exactly when the best choice is at time 0, and
    `replace_from_age` follows. A part 400 days old with `now_cost=500`
    runs to failure (254.5, against 500 now, from no age); with
    `now_cost=100` it is replaced now. A planned replacement with a
    lead time is a scheduled outage at that time.
  - **Each choice's break-even age** (`ReplacementChoice.break_even_age`)
    is the youngest at which replacing now costs no more than it, on a
    grid of 64 ages up to the one a new part reaches with a chance of
    0.1%, then halved down to; `replace_from_age` is the largest. A
    planned replacement's best time is found on a grid of 2,001 times up
    to the time the part outlives with a chance of 0.1%, refined; if none
    beats running to failure, its time is `inf`.
  - **Lives:** those with a survival function in closed form (surpyval's
    parametric distributions, mixtures, degradation processes); a
    non-parametric estimate or a `StandbyModel` is refused.
  - **Checks** (`test_maintenance_next.py`): the figures above; with no
    outages, the best planned time is the optimal age less the part's,
    and replacing now the best from the optimal age (to 1e-4); every
    choice against a plain simulation of it, at either baseline; the
    policy of a calendar with no opportunities gives the default's
    answer (to 1e-3), the shuts' policy the figures above, every choice
    dearer and beaten younger, and a break-even age is where the choice
    costs what replacing now does; the age replacement policy given is
    the default; a cost rate, a calendar or anything but a policy is
    refused; an exponential life (each
    outage's cost `price · exp(-λu)`, running to failure 0 and best);
    a planned replacement at once is replacing now, at `now_cost` above
    and below the planned price, and `now` is True exactly when the
    best choice is at time 0; a
    cheaper chance now breaks even younger; the answer the same in days
    and in hours; JSON; the input checks.
- **Learned / open questions:**
  - It is a break-even age for each later choice, as the maintainer
    framed it: the decision is "replace now" once the part is older than
    the largest.
  - **Still to check:** the decision against the timeline calculation
    (the best decision for every age and day, worked back from past the
    last scheduled outage), to measure what valuing a new part the same
    whenever it is fitted leaves out.

### UC-13: "Replacing the bearing costs 1 in the annual shut, 2 in a quarterly shut, 3 at a weekly stop, 4 on any other day and 20 at a failure. What is the best replacement policy in the long run?"

- **Status:** built (the maintainer's brief, "long-run optimal replacement
  policy over an outage calendar"): `NonRepairable.optimal_calendar_policy`,
  `calendar_policy_cost`, `Opportunity`, `Calendar`, `CalendarPolicy`.
  Since UC-45 (0.20) the prices come with the part, and the names are
  `optimal_opportunity_policy`, `opportunity_policy_cost` and
  `Opportunities` (the old ones deprecated, with the same numbers).
  Decisions to take or verify: `repyability/maintenance/CALENDAR_DECISIONS.md`.
- **In the 0.20 API:**

  ```python
  plant = Opportunities([
      Opportunity("weekly", period=7.0),
      Opportunity("quarterly", period=91.0),
      Opportunity("annual", period=364.0),
  ])                                         # any_day=True: 4 on any other day
  bearing = NonRepairable(
      surv.Weibull.from_params([1000.0, 2.5]),
      planned_cost=4.0, unplanned_cost=20.0,
      opportunity_costs={"weekly": 3.0, "quarterly": 2.0, "annual": 1.0},
  )
  policy = bearing.optimal_opportunity_policy(plant)   # 0.006943 a day
  bearing.opportunity_policy_cost(plant, {"annual": 0.0})
  ```
- **Who asks, and why:** a reliability engineer setting a part's
  replacement rule in a plant whose shuts make replacing cheap at some
  times and dear at others. Age replacement assumes one planned price at
  every moment; here the price depends on the time.
- **What they know:** the part's life (Weibull, scale 1000 days, shape
  2.5); the shuts, each with its period, offset (or dates) and price; the
  price on any other day (or none); the price at a failure.
- **What they want back:** the long-run cost a day and what it is made
  of (replacements and cost by kind of shut, and failures), which shuts
  the policy uses and the age from which each replaces, when the unit in
  service will be replaced, and what simpler rules cost.
- **How it is solved:** the state is the time and the age; dynamic
  programming on a grid (relative value iteration over a periodic
  calendar's cycle, exact for the grid; a long finite horizon otherwise),
  integration between opportunities in continuous time (no replacement on
  any day), and, for a cycle too long to sweep, the cost rate sampled over
  times of the cycle, each a window solved exactly.
- **The answer:** 0.006943 a day (25.27 over ten years); the annual shut
  replaces a unit of 88 days or more, the quarterly shuts one of 222, 324
  and 574 days (the last a quarter before the annual), the weekly stops
  older ones still, and no unit kept reaches the age the dearer day would
  replace it at. Every annual shut costs 25.48 over ten years, age
  replacement on any day 50.41, running to failure 82.05.
- **Checks** (`test_maintenance_calendar.py`,
  `test_maintenance_calendar_methods.py`): age replacement with no
  opportunities (to 1e-5); the cycle's rate against a long horizon's
  (to 1e-6); the policy simulated plainly (within 2 standard errors); the
  phase effect; skipping a young unit; the policy over 10, 20 and 40
  years (the same) and 5 (within 10 days); the simple policies; a life
  whose replacing ages are no threshold (flagged); repeated plans against
  the periodic calendar; integration against the grid, the daily limit
  (age replacement to 1e-6) and long gaps; sampling against the exact
  rate, a co-prime calendar, its determinism; the routes; units.
- **Learned:**
  - The brief's recursion renews a failed unit at the next step, low by
    about half a step a failure (1.7e-4 for this bearing at a step of a
    day). Renewing it half a step on average (as new or a step old, half
    each) makes the error second order (1.3e-7).
  - At prices like these the annual shut does nearly all the work: the
    quarterly and weekly stops matter only for a unit a failure renewed
    early in the year, so a calendar with a dearer failure (40) is the
    one that tells the methods apart.
  - **What the policy does, rather than a threshold for every time** (the
    maintainer, after seeing it used): `threshold(t)` and `replace(t,
    age)` were dropped as of no use (most times' thresholds, the weekly
    stops' and the dearer day's, are never reached). In their place:
    `replacement_rates` and `cost_rates` by kind (the long-run chance of
    each step's replacement, forward from a new unit until the ages at a
    cycle's start settle; exact for the grid, the costs summing to the
    cost rate to 1e-11, and for the quadrature with integration);
    `schedule()` gives each shut's long-run chance of a replacement (0 for
    a shut never used); and `next_replacement(age, now)` says when the
    unit in service will be replaced, at which shut and price. For the
    plant: 0.962 replacements a year at the annual shut, 0.078 failures,
    no other kind used.
  - **From the 0.14 persona review** (a planner using the bearing of
    UC-12, with shuts every 180 days at 100):
    - **The step is sized from the periods, not the offsets.** A shut at
      an offset of 0.1234 days (three hours into the day, what a date
      turned into model time gives) had the step sized from the gcd of
      the period and the offset, 1/5000, far below the life's step, so
      the life-sized step was taken, which does not divide 180: the
      calendar lost its period, `"auto"` sampled it (57 s for 0.5 s,
      the cost rate 0.04% off, no period) and the schedule drifted
      (0.0, 180.17, 360.03). The step is now a whole fraction of the
      periods (and a plan's repeat), the offsets and dates on it too
      only where that keeps it near the life's; an offset off the grid
      is laid at the grid time nearest it (within half a step, with the
      warning), and the policy is the aligned calendar's: the same step,
      cost rate, thresholds and chances, `dp`, exact, period 180.
    - **`"auto"` says when it samples** (the cycle has more than 20,000
      steps; `mc_samples` sets the cost), and the docs say that a
      sampled policy's rates by kind and the schedule's chances are
      estimates from a unit run in with no standard error (they move by
      some 10% between seeds at 8 samples); only `cost_rate_se` has one.
    - **A calendar allowing no planned replacement is refused**
      (`Calendar()` on a part with no `planned_cost`, or
      `Calendar(anytime=None)`), as `optimal_replacement_policy` refuses
      with no costs, rather than return running to failure.
    - **A unit past the oldest age worked out is replaced at the next
      opportunity** (`next_replacement(5000.0, now=0.0)` gives the shut
      at 0, not `inf`): it is past every threshold.
    - **The policy carries the life it was worked out for** (`life`, its
      saved form), and a `RepairableRBD` component with another life is
      refused, naming both: a policy worked out for the Weibull(1000,
      2.5) bearing given to the Weibull(365, 2) one simulated at a cost
      rate 19% off the one worked out for it, silently.


### UC-43: "Our shuts are on real dates: every Saturday, the first Tuesday of November, two turnarounds, and the truck's service every 15,000 km. What is the best rule for this part, on those dates, and how was it worked out?"

- **In the 0.20 API** (the prices on the part, UC-45): the plant of
  the answers below is

  ```python
  plant = Opportunities(
      [
          Opportunity("weekly stop", Recurrence.weekly("Sat"), duration="8h"),
          Opportunity("quarterly shut", Recurrence.monthly(
              weekday="Sunday", nth=1, every=3, anchor="2026-12-06"),
              duration="1d"),
          Opportunity("annual shut", Recurrence.yearly(
              month="November", weekday="Tuesday", nth=1), duration="3d"),
          Opportunity("turnaround", dates=["2027-03-03", "2028-09-09"],
                      duration="14d"),
      ],
      time_zone="Australia/Brisbane",
  )
  pump_bearing = NonRepairable(
      surv.Weibull.from_params([20_000, 2.5]),
      planned_cost=4.0, unplanned_cost=40.0,
      opportunity_costs={"weekly stop": 3.0, "quarterly shut": 2.0,
                         "annual shut": 1.0, "turnaround": 0.5},
  )
  pump_bearing.optimal_opportunity_policy(
      plant, method="averaged", plan=shifts, now="2026-11-16",
      unit="hours", age=3100,
  )
  ```

  and Example 6 is `optimal_opportunity_policy(Opportunities(),
  unit="days")` for a part with `planned_cost=1`.
- **Status:** built: `Shutdowns([Shutdown(...)], any_day=, time_zone=)`,
  `NonRepairable.optimal_calendar_policy(site, plan=, now=, readings=,
  age=, unit=, method=, horizon=, discount=, failure_timing=)` returning a
  `ShutdownPolicy`, `Shutdowns.route(...)` (a `ShutdownRoute`),
  `calendar_policy_cost(site, rule, ...)` (a `RuleCost`),
  `optimal_next_replacement(age, site=, ...)` (a
  `NextReplacementOnDates`), `cost_distribution(until=, policy=)` and
  `cost_by_window(dates, policy=)`, `Shutdowns.on_diagram(start, end)`
  for a `RepairableRBD`'s outages, and a `ShutdownPolicy` as a diagram
  component's `"calendar"` policy.
- **Who asks, and why:** UC-13's engineer, whose shuts are not periods
  in the part's clock but rules on dates ("every Saturday", "the first
  Tuesday of November"), one-off dates, plain periods, or a use ("every
  15,000 km or 6 months"); and the planner who keeps the site's shuts as
  one record that gives the diagrams their outages too. UC-40 laid one
  rule's dates on the part's clock as given times; that answered the
  policy but said nothing about the assumption a plan of given dates
  makes past its end (it was repeated), and kept the diagram's outages
  (`shutdown_spec`, UC-41) a separate record.
- **What they know:** the part's life (in calendar time, operating hours,
  km, landings: one clock); its failure price (`unplanned_cost`, always:
  the site prices only planned replacements); each shut's rule and price
  (and duration, for diagrams); the price of a planned replacement on any
  other day, or none; how the equipment runs (a `DutyPlan`), the meters
  now, the day answers count from.
- **What they want back:** the best rule ("replace at the annual shut if
  older than ..."), its cost a year (or its net present cost to a
  horizon), how it was worked out in words, the next replacement of the
  part in service as a date, the schedule of planned replacements by
  date, the replacements a year by kind, and what simpler rules cost.
- **How it is solved:** the shuts are laid on the part's clock (calendar
  time since `now`'s midnight in the life's unit, or a meter of the plan's
  forecast), over 28 years ahead (after which weekdays fall on the same
  dates again). Whether they *truly repeat in that clock* is checked on
  those model times (the fewest shuts `p` after which each is a fixed
  length on, within 1e-9, seen at least twice over), not assumed from the
  rule. Four routes, the result naming its route, solver and measure:
  1. **exact**: the shuts repeat on a cycle short enough to work out in
     about a minute (the estimate: relative value iteration's steps at
     about 12 µs a step and 5.5 ns an age): UC-13's calendar policy over
     the shuts as a repeating calendar (a kind with several shuts a
     cycle, as Saturdays on weekday shifts are not, is laid as one
     opportunity a shut), on the grid (`failure_timing="half_step"`) or
     by UC-13's quadrature (`"exact"`). A use on the life's own meter is
     exact whatever the duty: its shuts fall every 15,000 km in the
     gearbox's km.
  2. **a long cycle**: refused without `method="exact"` (said
     beforehand, with the time) or `method="sampling"` (UC-13's sampled
     windows, with the standard error).
  3. **averaged** (`method="averaged"`): the renewal-reward formula with
     each kind's timing spread evenly within its usual gaps: with
     thresholds `a_k` a kind and `a_0` for any other day, the planned
     replacement comes at `P = min(a_0, min_k a_k + U_k)`, `U_k` the
     equilibrium residual of kind `k`'s gaps (density `Pr(G_k > x) /
     E[G_k]`, the gaps those the rule gives in the part's clock over 28
     years), independent. Fixed dates are never averaged (5).
     Then a cycle lasts `∫ R Q`, kind `k` replaces with `∫ R q_k`, any
     other day with `R(a_0) Q(a_0)`, a failure the rest: integrals of `R`
     times piecewise polynomials, by Gauss-Legendre on the pieces (the
     closed form). The thresholds minimise the rate (each in turn over a
     scan and Brent's method, against never; then Nelder-Mead). After
     Dekker & Dijkstra (1992), NRL 39(2):175-190, and Dekker & Smeitink
     (1991), EJOR 53(1):46-63. With no shuts it is plain age replacement,
     route 1.
  4. **horizon** (`horizon=`, `discount=`, both required, 0% allowed;
     never refused, and named with a ready value by every refusal, UC-44):
     the backward dynamic programme over the real shuts to the horizon,
     each grid step discounted by the calendar time it spans (or UC-13's
     quadrature, each gap discounted at its own rate); the policy is
     worked out again to a horizon at least 5 years (or 3 life spans)
     later, and the first decision that differs is named: from about then
     on, decisions depend on where the horizon ends.
  5. **dated then long run** (`"dated_then_long_run"`, the default where
     a kind has fixed `dates=` from now on, or `dated_until=` is given;
     a decision of 15 Nov 2026): a dated window from now to the last
     fixed date (or `dated_until`), in which every shut, the recurring
     kinds' too, is on its real dates, so fixed dates and rules combine
     freely; after it only the recurring shuts remain (a kind with
     `then_every="4 years"` recurring at that gap from its last date),
     and their long-run average, by the route they take on their own
     (1-3, `method` applying to them, a choice asked for as there; with
     none recurring and no other day, running to failure), values
     everything beyond. Undiscounted, the long run's cost a year adds the
     same to every choice in the window, so the window's decisions turn
     on its relative values: the backward programme of 4 at 0%, over the
     recurring shuts on their real dates past the window's end, run on 5
     years (or 3 life spans) and doubled until two runs give the window
     the same decisions (to a grid step and 0.1%). After the window a
     part is replaced by the long run's own rule. `cost` is the long
     run's a year, `dated_cost` the expected cost to the window's end,
     and the rule is in two phases (`rule.dated`: each fixed date's own
     threshold). Fixed dates all before now are left out. With
    `discount=` and no horizon (UC-44) the window and its run-on are
    discounted (the programme of 4 at that rate), so what lies past the
    window is valued by the recurring shuts' discounted long run, not by
    undiscounted relative values; `cost` is then the long run's
    equivalent annual cost and `dated_cost` the present value to the
    window's end.

  Results are dated through the layout: a shut's own date, a planned
  replacement on any other day at the first moment the part's clock
  reaches its time.
- **The answers, by hand** (the part Weibull(1000, 2.5) days, planned 1,
  failure 5, unless said):
  - Example 1, no site: 493.05 days, 0.003462 a day (the textbook;
    `optimal_replacement_policy`). Example 6, `Shutdowns([], any_day=1)`:
    route 1, closed form, **493.05 days and 0.0034620 a day**, $1.26 a
    year.
  - Example 2, a monthly stop every 30 days, no other day: the same
    as UC-13's `Calendar([Opportunity("m", 1, period=30)],
    anytime=None)`: **0.0034635 a day** ($1.27 a year), replace at a stop
    once **479.1 days** old. Averaged (one kind, equal gaps): 0.0034628,
    within 0.02% (see *Learned*).
  - UC-13's quarterly (91 days, 3) and annual (364 days, 1) shuts at a
    failure price of 20, laid from 16 November 2026: route 1,
    **0.0069430 a day**, UC-13's own figure; by quadrature 0.0069430
    (1.6e-6 apart). To 2046 and 2066 at 0%: net present costs 49.84 and
    100.68, so 0.006959 a day between them (0.2% from route 1: the two
    horizons end at different points of the 364-day cycle), and decisions
    in the first 18 years are the same either way.
  - The 7, 93 and 400-day shuts at 3, 2 and 1, any other day 4, failure
    40, the part Weibull(1000, 5): one cycle is 260,400 days; exactly
    (method="exact", 15 s) **1.2801 a year**; sampled, 48 windows, 1.2766
    ± 0.0192 and 1.2588 ± 0.0176 (seeds 2 and 3).
  - The plant (Mon-Fri 06:00-22:00 in Brisbane, the bearing Weibull(20,000
    h, 2.5), failure 40, aged 3,100 h on 16 November 2026; a weekly stop on Saturday at 3, a
    quarterly shut on the first Sunday of every third month at 2, the
    first Tuesday of November at 1, turnarounds on 3 March 2027 and 9
    September 2028 at 0.50, any other day 4): the route check gives the
    dated window to 9 September 2028 and refuses to choose the long run
    (the quarterly shut falls 84 to 98 days apart, the annual 364 or 371).
    Averaged long run: **$1.90 a year** after the window (the recurring
    shuts' own figure); in the window replace at the turnaround of 3
    March 2027 if older than 1,544 h and at that of 9 September 2028 if
    older than 798.5 h; after it at the annual shut if older than 1,949
    h; the part in service next at the turnaround of 3 March 2027, 4,332
    h old; $3.31 expected to 9 September 2028. To 16 November 2036 at 7%:
    $12.30, the turnaround of 3 March 2027 next, then the annual shuts,
    decisions from about 29 May 2034 depending on the horizon. (Before 15
    Nov 2026 the averaged route took the two turnarounds as one every 18
    months: $1.50 a year, all at turnarounds that stop after 2028.)
  - The truck (Example 4: 400 km a day, the gearbox Weibull(180,000 km,
    2.2), failure 40,000, the service every 15,000 km or 6 months at 9,000,
    any other day 14,000, the odometer at 212,000 km): route 1 whatever
    the 6-month limit, which never bites at 37.5 days a service; replace
    at a service once 89,000 km old, $26,113 a year; services on 18
    December 2026, 25 January, 3 March, 10 April, 17 May 2027 (every 37.5
    days). At 50 km a day the 6-month limit always bites, and the
    services repeat every 9,131 km in the gearbox's clock: still route 1.
- **Checks** (`test_maintenance_shutdowns.py`): route 1 against the
  textbook (no shuts; a shut every day; one kind against UC-13's
  `Calendar`); sampling within its error of exact on 7/93/400; averaged
  against exact on one kind with equal gaps, its statement listing every
  kind with its gaps, and its formula against a plain simulation of its
  own assumption; the horizon to a long one at 0% against route 1's rate,
  its early decisions the same for 10 and 20 years, grid against
  quadrature; one record moving both a diagram's outages and the policy's
  opportunities; every refusal with its wording; saving and loading; the
  old model-time API unchanged. The dated window
  (`test_maintenance_shutdowns_dated.py`): the plant's turnarounds before
  September 2028 and annual shuts after; the long-run figure the
  recurring shuts' alone; the window's decisions those of a horizon 40
  years out at 0%, and by quadrature close to the grid's; a monthly rule
  with two fixed dates the same as all of them fixed dates; `then_every`
  the same as adding the recurring kind; the route check's words; the
  part in service seeing the turnarounds ahead; `dated_until`.
- **Learned:**
  - **"Averaged equals exact when the gaps are equal" holds closely for
    one kind, not exactly, and not for several.** The averaged formula
    takes each part as meeting the shuts at a random phase; on a truly
    repeating calendar a part fitted at a shut meets the next at a fixed
    phase, and kinds that coincide (a quarterly shut on the annual) are
    not independent. One kind every 30 days: 0.0034628 against 0.0034635
    (0.02%); UC-13's quarterly and annual shuts: 0.0071122 against
    0.0069430 (2.4%). So the averaged route says its assumption on every
    result, and is offered only by request.
  - **Averaging one-off dates over their gaps was too strong an
    assumption, so it went.** Two turnarounds 18 months apart were
    averaged as a turnaround every 18 months; at $0.50 the averaged rule
    replaced only at turnarounds, which stop after the second. Since 15
    Nov 2026 fixed dates take the dated window (route 5): the turnarounds
    as they come, then the recurring shuts' long run, with no horizon or
    discount to choose.
  - **The short-cycle limit is about work, not years.** 7/93/400 days is
    a 713-year cycle, but on a day's grid it is about 50 s of work for the
    textbook part (route 1, by the one-minute limit) and longer for a
    part needing a finer grid (route 2). The route check gives the
    estimate either way.
  - **A time zone with daylight saving breaks "repeats" for a life in
    calendar time:** a weekly stop at midnight is 167 or 169 hours apart
    across the changes. The route says so; `time_zone=None` takes clock
    times as they are, if that is what is meant.


### UC-44: "Money spent later is worth less to us (7% a year). What replacement age, or rule at our shuts, is cheapest in the long run at that rate?"

- **Status:** built.
- **In the 0.20 API:**

  ```python
  stops = Opportunities([Opportunity("stop", period=28.0)], any_day=False)
  part = NonRepairable(surv.Weibull.from_params([1000.0, 2.5]),
                       unplanned_cost=20.0, opportunity_cost=3.0)
  part.optimal_opportunity_policy(stops, discount_rate=math.log(1.07) / 365.25)
  site = Opportunities([Opportunity("stop", Recurrence.every(4, "weeks", "2026-11-21"))],
                       any_day=False)
  part.optimal_opportunity_policy(site, unit="days", now="2026-11-16", discount=0.07)
  # Where nothing repeats exactly, a horizon (the refusal names one):
  part.optimal_opportunity_policy(stops, discount_rate=math.log(1.07) / 365.25,
                                  horizon=112000)
  part.optimal_opportunity_policy(site, unit="days", now="2026-11-16",
                                  discount=0.07, horizon="2333-11-16")
  ```
- **Who asks, and why:** a reliability or maintenance engineer whose
  finance team discounts costs: the long-run cost rate (UC-13, age
  replacement) treats a dollar in ten years as a dollar now, which
  replaces earlier than their own figures would.
- **What they know:** the part's life and prices, as for age replacement,
  UC-13 or UC-43, and a discount rate: continuous, per unit of the
  life's time (`discount_rate=`) in model time; yearly (`discount=`,
  0.07 for 7%) on a site's real dates, as a horizon takes it.
- **What they want back:** the age, or the rule at each shut, with the
  least equivalent annual cost (EAC), that cost, and the present value
  of every replacement from now on, forever (`npv`); the undiscounted
  long-run figures of the rule chosen (`cost_rate`, the rates by kind)
  stay beside them.
- **How it is solved:**
  - **Age replacement** (Fox, 1966): `EAC(T) = [cu (1 - exp(-r T) R(T) -
    r M(T)) + cp exp(-r T) R(T)] / M(T)`, `M` the integral of `exp(-r t)
    R(t)` to `T` (the cost rate at `r = 0`), scanned on a log grid of
    ages by the trapezoidal rule up to where the part has all but surely
    failed or a cost is discounted to `exp(-40)`, the best refined by a
    bounded search on the cost by quadrature (`age_replacement_value`'s);
    against running to failure, `cu (1 - r M) / M` with `M` to infinity.
    A non-parametric estimate is scanned on its own grid.
  - **A calendar in model time**, periodic, short cycle: value iteration
    over the cycle, each grid step discounted by `exp(-r step)` (or each
    gap of the quadrature at `r`), from the last sweep's values less a new
    part's, stopped by MacQueen's bounds: every path through a cycle is
    discounted by the same `exp(-r L)`, so the fixed point lies in a known
    band of the last sweep. `npv` is a new part's at time 0. The rule so
    chosen is then priced undiscounted by relative value iteration with
    its decisions fixed.
  - **A site:** `discount=` with no horizon takes the exact routes
    discounted (the yearly rate as `ln(1 + d)` over the part's units a
    year), plain age replacement on any day by Fox's formula, and the
    dated window discounted with the recurring shuts' discounted long run
    after it (UC-43, route 5). `cost` is the EAC a year, `ln(1 + d)`
    times `npv`.
  - **The rule (the maintainer, 16 Nov 2026): "Exact by default; refuse
    what can't be done exactly; `horizon=` never refuses."**
    - **Exact by default:** with no `horizon=`, the exact repeating
      cycle's answer, discounted or not, wherever the opportunities
      repeat within `MAX_CYCLE` steps (the routes above, unchanged).
    - **Refused, with a ready answer:** a cycle longer than `MAX_CYCLE`,
      given dates in model time, shuts that drift on real dates, a duty
      plan, sampling or averaged gaps with a discount, and pricing a rule
      where nothing repeats. Each refusal names a horizon worked out from
      the inputs: with a discount, where costs are discounted to under
      one part in a billion (`ln(1e9) / ln(1 + d)` years, 307 at 7%: from
      16 Nov 2026, `horizon='2333-11-16'`; in model time `ln(1e9) / r`,
      rounded up to three figures: 112,000 days at 7% a year); with none,
      the full cycle and three of the part's life spans (at most two
      million grid steps). Where sampling applies (undiscounted) it names
      `method='sampling'` too.
    - **`horizon=` never refuses for those reasons:** a backward sweep
      from the horizon (`_horizon.backward`, `_replacement._sweep` with
      each step discounted), on the real dates (through a duty plan's
      forecast) or in model time (`horizon=` a number, with
      `discount_rate=`; given dates as they are, nothing assumed after
      them). It gives the net present cost (`npv`, or a site's `cost`)
      and the equivalent annual cost (`eac`: `r NPC / (1 - exp(-r H))`);
      undiscounted, the total to the horizon (and it over `H`).
    - **A given rule** (`opportunity_policy_cost(..., discount_rate=)`
      in model time, `discount=` on a site, a site's `compare()`) is
      priced over the exact discounted long run as the optimiser's is,
      with the decisions fixed (`_discounted_values(fixed=)`, or the
      quadrature's); where nothing repeats, the rules above.
    - **`method="auto"`** in model time still samples a cycle too long
      to sweep, undiscounted, as released, but with a `FutureWarning`:
      from 0.21 it refuses, naming the ready horizon and
      `method='sampling'`.
    - **`optimal_next_replacement`** with a discounted policy discounts
      its own choices at the policy's rate, its baseline the policy's
      EAC (`_next`): each choice's present value and the policy's after
      it, less the policy's from now, `price e^(-ru) S(u) + cu (1 -
      e^(-ru) S(u) - r M(u)) - EAC M(u)`, `M(u) = ∫₀ᵘ e^(-rt) S(t) dt`;
      at rate 0, the undiscounted cost as before.
  - **The five cases once refused, before and after** (7% a year;
    `A_DAY = ln(1.07) / 365.25`):
    - (a) shuts every 7, 93 and 401 days at 3, 2 and 1, any day 4,
      failure 40, Weibull(1000, 5), `discount_rate=A_DAY`: refused ("give
      opportunities whose periods share a shorter cycle, or ... a site");
      now refused with `horizon=112000`, and with it NPC 18.31, 1.239 a
      year, the next replacement at the 401-day shut (4.7 s).
    - (b) given model-time dates [100, 400] at 1, any day 4, failure 20,
      Weibull(1000, 2.5): refused ("give its shuts as a site"); now
      `horizon=112000`: NPC 69.84, 4.73 a year.
    - (c) an annual shut on the first Tuesday of November (364 or 371
      days apart) at 1, failure 20, `discount=0.07`: refused ("give
      horizon= with discount="); now with `horizon='2333-11-16'`: $36.69
      to then, $2.48 a year, replace at the shut if older than 89.6 to
      91.4 days, next on 2 November 2027.
    - (d) a stop every 4 weeks on `DutyPlan(Shifts("Mon-Fri", "06:00",
      "22:00"))`, the bearing Weibull(20,000 h, 2.5), failure 40:
      refused (steady pace); now `horizon='2333-11-16'`: $25.60, $1.73 a
      year, at a stop from 3,815 to 3,833 hours.
    - (e) pricing `{"annual shut": 400}` at (c)'s site: refused ("not yet
      at a discount over the long run"); now refused only because the
      shut drifts, with `horizon='2333-11-16'`: $61.03, $4.13 a year. On
      the 28-day stop, which repeats, it is priced exactly: `{"stop":
      419}` at $4.22 a year, the optimiser's own EAC to 1e-9.
- **Horizon against exact** (the 28-day stop below, model time): the
  horizon's EAC a year 4.0275 at 10 years, 4.1521 at 20, 4.2100 at 50,
  4.21625 at 100, 4.2164614 at 200 and 4.2164617 at 307 (the ready
  horizon), against the exact 4.2164617 (gaps 4.5%, 1.5%, 0.15%, 5e-5,
  6e-8, 3e-11); the threshold at the first stop 419.125 days from 50
  years on (410.4 at 10 years, where the end is a part's life away).
- **Today:** the Weibull(1000 h, 2.5) bearing at `cp = 1`, `cu = 5`:
  493.0 h undiscounted; 493.2 h at 5% a year and 493.4 h at 10%, the
  life in hours (8,760 a year), where a cycle is a few weeks and
  discounting hardly matters; with the life in days, 497.7 and 502.3
  days. A shut every 28 days at 3, a failure at 20, Weibull(1000 days,
  2.5): the stop's threshold 414 days undiscounted, 419 at 7% and 433.5 at
  30% a year.
- **Checks:** `test_discounted_long_run.py`: rate 0 the undiscounted
  answers unchanged; the age against Fox's cost from the density by
  quadrature, and its present value against a simulation of the
  discounted renewal costs; a later age as the rate rises for a part
  that wears out; the rate going to 0 coming to the cost-rate optimum;
  running to failure winning at a high rate; a calendar's present value
  against a plain backward sweep over many cycles, grid against
  quadrature; a site's against the horizon route's net present cost to
  50 years, and the dated window's decisions against a long discounted
  horizon's.
- **Checks** (`test_exact_or_horizon.py`): rate 0 and the exact routes
  bit for bit as before; the horizon's EAC coming to the exact one as it
  lengthens, its first decisions within a step; each of (a)-(e) refused
  with its ready horizon, which works and gives the same to 1e-8 doubled;
  a fixed rule's discounted price against a plain backward sweep over
  many cycles and a seeded simulation of its discounted costs, and the
  optimiser's rule at its own EAC (grid and quadrature); the next
  replacement's discounted choices against present values from the
  density; the `FutureWarning` with its ready horizon.
- **Learned / open questions:** the exact EAC of the 28-day stop is
  4.2165 a year, not the 4.202 once quoted; its threshold 419.1 days. A
  site's horizon lays the shuts on a grid sized from the life alone, the
  exact route's on the shuts, so the two differ by about 1.5e-5 there.
  Not yet discounted: the repairable unit's overhaul, failure-limit and
  overhaul-replace policies, failure finding and on-condition intervals,
  and `RepairableRBD.optimal_replacement_intervals`. A rule's undiscounted
  price over a long cycle in model time is still differenced over a long
  finite horizon, not refused.


### UC-45: "One list of our stops and shuts serves every part. Each part's replacement costs differently, and some jobs don't fit in a short stop. What is each part's best rule, and what did it price each stop at?"

- **Status:** built (the maintainer's decision, 16 Nov 2026; see
  `CALENDAR_DECISIONS.md`, G): `Opportunity`, `Opportunities`,
  `NonRepairable(opportunity_cost=, opportunity_costs=, job_duration=)`,
  `optimal_opportunity_policy` and `opportunity_policy_cost` (the policy's
  `prices`, an `OpportunityPrices`), and `NonRepairable.opportunity_prices`.
- **Who asks, and why:** a planner who keeps one record of the site's
  stops and shuts, and an engineer setting rules for many parts against
  it. Until 0.20 the price was on the opportunity (`Opportunity(name,
  cost)`, `Shutdown(replace_cost=)`), so each part needed its own copy of
  the site; and a model-time `Calendar` and a real-date `Shutdowns` were
  two ways to say the same thing.
- **What they know:** the site's opportunities (when, how long, what each
  takes down); for each part its replacement's price at an opportunity,
  on any other day and at a failure (or a replacement's cost, its
  duration and the plant's downtime cost an hour, from which those
  follow), any opportunity priced differently, and how long the job takes.
- **What they want back:** each part's policy, as UC-13 and UC-43, with
  the price each opportunity was given and where it came from, and the
  opportunities not used and why ("the 6 h job doesn't fit in the 4 h
  stop").
- **How it is solved:** no new method. The part's prices are laid on the
  opportunities (`_opportunities.resolve`): at each opportunity the part's
  `opportunity_costs` for it (by name, or the `Opportunity` itself), else
  its `opportunity_cost`, else its `planned_cost` (an opportunity priced
  as any other day); on any other day its `planned_cost` where
  `any_day=True`; at a failure its `unplanned_cost`. An opportunity shorter
  than `job_duration` (either length unsaid: used), or with `where=` one
  whose `takes_down` leaves that label out, is dropped. The result is the
  solvers' own record (a model-time `Calendar`, or a site), solved as
  UC-13 and UC-43 solve it. A key of `opportunity_costs` naming no
  opportunity of the collection is refused, listing those it has. The
  0.14 and 0.19 forms are read the same way, their prices taken as the
  part's overrides for those opportunities. `opportunity_prices` works
  out the prices: at a fitting opportunity the replacement alone, on any
  day the replacement and its downtime, at a failure the failure's cost
  and its downtime; `apply(part)` gives the part those prices (the
  downtime cost is the system's, not kept on the part).
- **The answers, by hand:** a replacement of 1,200 taking 6 h, the plant
  down at 2,500 an hour, a failure of 4,200 with 18 h down: 1,200 at a
  stop of 8 h; not at a stop of 4 h; 1,200 + 2,500 x 6 = 16,200 on any
  other day; 4,200 + 2,500 x 18 = 49,200 at a failure. UC-13's plant as a
  part with `planned_cost=4`, `unplanned_cost=20` and overrides 3, 2 and
  1: 0.006943 a day, the `Calendar`'s figure bit for bit.
- **Checks** (`test_opportunities.py`): the old forms against the new,
  bit for bit, in model time, on a site with and without fixed dates, and
  discounted both ways; overrides by name and by object; a misspelt name
  refused with the names; a stop too short skipped, said, and the same as
  the site without it; the price with no `opportunity_cost`; the helper
  against the hand-worked prices, and applied; one record serving two
  parts, each its own calendar's policy; `where=`; saving and loading the
  opportunities, a saved `Shutdowns`, and the part's prices; each old
  form's warning.
- **Learned / open:** `takes_down` is recorded and filters a policy by
  `where=`; a diagram does not read it yet (a shut's `scope` is still
  given to `on_diagram`). The results keep the names `ShutdownPolicy`
  and `ShutdownRoute`.


### UC-14: "A cheap task every visit and a deeper one every few visits: how often should we visit, and how deep?"

- **Status:** built: `optimal_layered_visits(light, deep, *, visit_cost=,
  max_ratio=)` and `layered_cost_rate`.
- **Who asks, and why:** a planner bundling the tasks of several parts
  into visits that each cost something: more visits mean fewer failures
  but more visit cost, and a part that lives long should be renewed at
  every few visits only.
- **What they know:** two parts with lives and prices, what a visit costs
  by itself.
- **What they want back:** the visit interval, the depth ratio `k` (the
  deep part at every `k`-th visit), the cost rate and its parts.
- **How it is solved:** the cost rate of a part renewed every `P` is
  `(planned + unplanned M(P)) / P` with `M` the renewal count (exact,
  `NonRepairable.failures`); the package is `visit_cost / T +
  rate_light(T) + rate_deep(k T)`; for each `k` the best `T` is found on
  a grid and refined. One layer without a visit cost is the optimal
  block-replacement interval.
- **Checks:** the cost rate against the closed form for exponential
  lives and a seeded simulation of the plan; the optimum and each
  ratio's best rate against a dense scan using a renewal function solved
  here by a separate discretisation (within 0.4%).
- **Learned:** with a cheap failure, ratio 1 wins; the ratio only pays
  when the light part's failures are dear (a failure of 1,000 against a
  renewal of 20 gives a ratio of 10 at every 30 days). Layers are
  independent here: a failure replaces its part but does not move the
  visits.

### UC-15: "This task is due every 300 days but may slip by up to 60. What does the slip cost?"

- **Status:** built: `NonRepairable.slack_effect(interval, slack)`.
- **How it is solved:** age replacement at `T` and at `T + slack`:
  the chance of a failure first and the long-run cost rate
  (`(planned R + unplanned (1 - R)) / integral of R`), their
  differences, and the rate averaged over a uniform lateness.
- **Checks:** both rates against a seeded simulation of the cycles;
  the risk against the survival function.
- **Learned:** below the optimal age a slip can lower the cost rate
  (a negative `added_cost_rate`): the tolerance is then free.

### UC-16: "The replacement comes due in 10 days but the crew visits weekly. What does waiting for the visit cost?"

- **Status:** built: `NonRepairable.cost_of_waiting_for_visit(age, due_in,
  visits)`, with `visits` an `Opportunity` or a list of times.
- **How it is solved:** the first visit at or after the due time, then
  `delay_effect` from the due time to it: one cycle to the replacement,
  `cost(u) = planned exp(-r u) S(u) + unplanned (1 - exp(-r u) S(u) - r
  M(u))`, `S(u) = R(age + u) / R(age)`.
- **Checks:** the exponential closed form (`added_cost = (unplanned -
  planned) (exp(-lam d1) - exp(-lam d2))`) and a seeded simulation of
  the cycle, discounted and not.

### UC-17: "If we miss the visit, how much risk does the delay add?"

- **Status:** built: `NonRepairable.cost_of_missed_visit(age, visit_in,
  next_visit_in, miss_probability=)` and `delay_effect`.
- **How it is solved:** as UC-16 with the next visit as the new time,
  weighted by the chance the visit is missed for the expected added risk
  and cost. Skipping and deferring a replacement for a whole interval
  are UC-6's; this is one cycle.


### UC-18: "For this failure mode, what do the four ways of managing it each cost and prevent?"

- **Status:** built: `run_to_failure(part)`, `scheduled_replacement(part,
  interval=None)`, `failure_finding(device, test_interval, ...)` with
  `optimal_failure_finding_interval`, and `on_condition(pf_interval,
  inspection_interval, ...)` with `optimal_on_condition_interval`. The
  option compared at a planned replacement is UC-6's
  `compare_replacement_options`.
- **Who asks, and why:** someone choosing the task for one failure mode
  who needs the numbers to compare: how often it fails, what it costs a
  unit of time, and how much risk is left.
- **How each is solved (exact, model time):**
  - *Run to failure:* `1 / (MTTF + replacement time)` failures a unit of
    time, at the unplanned cost.
  - *Scheduled replacement at age `T`:* the cycle is the life cut off at
    `T`: failures `(1 - R(T)) / M(T)`, planned replacements `R(T) /
    M(T)`, cost `(planned R + unplanned (1 - R)) / M(T)`, `M` the
    integral of `R`; the cheapest `T` is `optimal_replacement_policy`'s.
  - *Failure finding:* a hidden failure found at a test, every `tau`, and
    repaired as new; a working device carries its age on. A cycle ends at
    the test that finds the failure, so the mean unavailability is `1 -
    MTTF / (tau sum R(j tau))`, which for an exponential life is `1 - (1 -
    exp(-lambda tau)) / (lambda tau)`, near `lambda tau / 2`. Demands at
    a rate `d` meet a failed device at `d U`. The interval comes from a
    target on `U` (the longest meeting it) or from the cost rate of tests,
    repairs and the consequence of multiple failures.
  - *On condition:* a warning precedes the failure by the P-F interval
    `X`; inspections every `tau` at a random phase find it with
    probability `p` each. A warning of length `x` is missed with
    probability `q^floor(x/tau) (1 - p frac(x/tau))`, `q = 1 - p`; the
    chance it is found is one less its mean over `X`. A fixed warning is
    inspected best as often as it is long.
- **Checks:** closed forms for an exponential life (each), and seeded
  simulations of the plans themselves (cycles with tests; warnings with
  random phase) that do not use the formulas, within four standard
  errors; the intervals against dense scans.
- **Learned / left:** repair and test times are taken as nil; one
  device's hidden failure is independent of the demand; the detection
  probability does not change as the warning grows. The system-level
  consequence of a failure is a diagram's, not here.


### UC-19: "What is this plan worth over ten years at 8%, and what is it as a yearly cost?"

- **Status:** built (the first part of a net present value valuation):
  `RepairableRBD.expected_npv(horizon, discount_rate=, value_rate=,
  periods=)` and `NonRepairable.age_replacement_value(age,
  discount_rate=)`.
- **Who asks, and why:** someone who must justify a plan (an interval, a
  spare, a repair crew, a redundant unit) at the cost of capital: what the
  plant earns while it works, less what it costs to buy and run, each
  amount counted at what it is worth now, and the same cost as a steady
  yearly figure to compare plans of different lives.
- **How it is solved (exact, model time):** by the linearity of
  expectation, the expected present value of a cost charged at events is
  its price times the discounted expected number of events, and of a value
  earned per unit time the discounted expected time up: `integral exp(-r t)
  A(t) dt`. The costs are `expected_cost(discount_rate=)`, which discounts
  each event at its own time (a block replacement or a test at its exact
  time); the value is that integral of the system's point availability,
  summed on the same pieces as the mission availability, with the settled
  level or period after it in closed form. The acquisition cost is paid at
  the start. The equivalent annual cost is the cost over the annuity
  factor `(1 - exp(-r H)) / r` (`H / 1` with no discounting), which tends
  to the cost rate over a long horizon. For one part replaced at an age or
  at a failure for ever, Fox (1966): `PV = [cu A + cp exp(-r T) R(T)] /
  (r M(T))`, `A = 1 - exp(-r T) R(T) - r M(T)` the discounted chance of
  failing first, `M` the integral of `exp(-r t) R(t)` to `T`.
- **The answer:** the net present value, its value and cost, the
  equivalent annual cost, the undiscounted totals, and a cash-flow table
  by period, category and component, in model time (which period is which
  financial year is the caller's).
- **Checks:** `test_expected_npv.py`: an exponential pump's value against
  the closed form of its availability, discounted; its repair cost as `cost
  x lambda x` the time up; no discounting against `mission_availability` and
  `expected_cost`; block replacements at exact times; an age replacement
  over a long horizon against Fox's formula integrated independently; the
  table adding up to the total; the cost rate as the rate goes to 0.
- **Not yet:** value that varies in time, escalation, plans of actions at
  given times, crews as cash flows, and NPV as
  the objective of the optimisers other than age replacement and the
  calendar policies (UC-44).


### UC-20: "If we run at 70 °C instead of 30 °C we make more, but the bearings wear out faster. Does it pay?"

- **Status:** built for one unit: `operating_schedule_value(model, times,
  covariates, value_rates, horizon, failure_cost=, down_time=,
  discount_rate=, age=)` and `best_switch_time(model, covariates,
  value_rates, horizon, ...)`. Built for a diagram, simulated:
  `RepairableRBD.best_switch_time(covariates, value_rates, horizon, *,
  discount_rate=, mc_samples=, seed=, scan=, engine=, state=)`, whose
  switching blocks run on covariate schedules (a spec's `"covariates"`);
  each candidate switch time is the diagram with those schedules,
  simulated from one seed (common random numbers) and priced as
  `npv_distribution` prices its runs, the value split at the switch.
- **Who asks, and why:** an operations manager who can raise output at
  the cost of faster wear, and wants to know whether the extra output
  pays for the extra failures, and when to change.
- **What they know:** the unit's life against its covariates (a fitted
  surpyval regression model: accelerated failure time, proportional
  hazards or odds, additive hazards); the operating schedule in model
  time from 0 (when each stretch starts, its covariates, and what the unit
  earns per unit of time up in it: the calendar is the caller's, who turns
  it into model time); the cost of a failure and how long a replacement
  takes; the discount rate; the horizon.
- **What they want back:** the NPV of each schedule with the value earned
  and the failures' cost apart, the expected failures and time up, a table
  by stretch, and the switch time that maximises the NPV, with the NPV of
  never switching and of switching at once beside it.
- **How it is solved (numerical, model time):** the unit is replaced by a
  new one at each failure; a unit put in at time `v` runs along the rest
  of the schedule, so its life is the model's `sf_tvc` along the schedule
  shifted by `v`, and its failures and replacements form a renewal process
  whose life depends on when each unit was put in. Its expected failures
  and chance of being up solve a renewal equation in the plant's time, on
  a grid sized from the lives with the schedule's changes at cell edges:
  within the stretch a unit was put in, its life is that stretch's
  fixed-covariate life averaged exactly over the cell; past it, the life
  along the path of a unit put in at the cell's middle; a replacement's
  end follows from the down time's distribution integrated exactly over
  the cell. The value is the discounted integral of the value rate times
  the chance of being up, the cost the discounted failures times their
  cost. The grid is halved until every total agrees with the halved
  grid's to a tolerance (1e-6 relative by default). The best switch time
  is searched on 41 switch times across the horizon (the NPV need not have
  one peak), then polished by Brent's method between the best one's
  neighbours.
- **Today:** a constant schedule is a one-component `RepairableRBD` whose
  life is the model's at the fixed covariates (a Weibull of scale
  `alpha / exp(beta z)` for the guide's Weibull AFT bearing), which
  `expected_npv` values exactly. Over 20,000 hours at 8% a year, 150,000
  a failure, 24 hours down and value 30 or 70 an hour: cool (30 °C), NPV
  177,051 (value 548,462, cost 371,411); hot (70 °C), NPV 391,649 (value
  1,274,025, cost 882,376). `operating_schedule_value` gives the same to
  the unit. The guide's schedule (cool for 2000 hours, then hot) has no
  such model: a `RepairableRBD` refuses a `RegressionNode` as a life, so
  it could be answered only by simulating the process by hand, as the
  checks below do. It is NPV 368,081 (value 1,195,407, cost 827,326).
- **Checks:** `test_operating_schedule.py`: exponential lives and repairs
  against the two-state Markov chain stretch by stretch; an accelerated
  life with instant replacement against the baseline's renewal function
  at the accumulated exposure (from `NonRepairable.failures`); a constant
  schedule against a one-component `RepairableRBD`'s `expected_npv` and
  `expected_events`; proportional hazards (where time does not rescale)
  against a hand-written seeded simulation drawing each life along its
  shifted path, with a unit aged at 0 and log-normal down times; the best
  switch time against a fine scan, with an end best each way; results
  scaling with the unit of time from 1e-6 to 1e6 and with the
  covariate's unit; the refusals, JSON and `help()`.
  `test_switch_time_diagram.py`: a one-block diagram against the one-unit
  answer (its choice, each end's NPV within its interval, the gain over
  switching at once within the interval on the difference); a block the
  load does not touch in series, which leaves the switch time; seeded
  searches repeat and their intervals narrow with the runs; the value
  rates split at the switch against a closed form; the refusals.
- **Learned / left:** each unit put in before the last change has its own
  life, so the work grows with the cells before it (one evaluation of the
  model along a path each); `best_switch_time` values each switch time to
  a looser tolerance by default (1e-4) for that reason. A diagram's search
  simulates each candidate; drawing a life along a schedule calls the
  model's cumulative hazard a few times per draw, which is most of its
  time. Left: a schedule
  that varies continuously (a `CovariatePath`), and value that depends on
  more than the stretch.

### UC-21: "We've had two years on this plan. How does what happened compare with what we expected, and what's the plan worth now?"

- **Status:** built: `RepairableRBD.plan_vs_actual(timelines, now,
  horizon, discount_rate=, value_rate=, periods=, state=)`.
- **Who asks, and why:** an engineer (or an agent) reviewing a plan part
  way through: where the plan's assumptions were wrong, and what the plan
  is worth from today, knowing what happened.
- **What they know:** the diagram with its costs (the plan), the outage
  log of each component (`Timeline`s, planned outages flagged) on the
  diagram's clock from the start of the plan to `now`, and, where the log
  does not give it, a component's state now.
- **How it is solved (exact, model time):** the expected side is
  `expected_npv`'s cash-flow table, its periods cut at `now`. The actual
  side prices the log with the same costs: each failure its repair and
  replace costs at its instant, each planned outage its preventive cost,
  each down span its downtime rate, the system's down and up spans (merged
  from the components' through the structure, as `system_timeline` does)
  its downtime rate and its value, each discounted exactly (`exp(-r t)` at
  an event, `(exp(-r a) - exp(-r b)) / r` over a span). A cost the log
  does not determine (tests that find nothing, a hidden failure's repair,
  a standby group's units, replacements that take no time, calendar
  prices, set-up and acquisition costs) is compared on neither side and
  listed with its reason. The states now are read off the logs where the
  diagram's rules let it (a unit renewed at each repair, or replaced at an
  age, on a block calendar or on condition by maintenance that takes time:
  its age since its last restoration, or how long it has been down and
  why, and where it is on its calendar); elsewhere it refuses and asks for
  `state=`. The rest is `expected_npv` from those states, discounted back
  to the start, without the purchase.
- **The answer:** the expected, actual and variance tables to `now` by
  period, category and component (a positive variance in the plan's
  favour), what is not compared and why, the states used, and the plan's
  value now (what happened so far, the uncompared costs as planned, and
  the rest from today) beside its original value.
- **Checks:** `test_plan_vs_actual.py`: a pump's log priced by hand
  (each discount factor written out) and its variance against
  `expected_npv`'s table; the states read off logs by hand, and each
  refusal; simulated histories (`simulate_timelines`) priced as logs
  averaging to the plan's exact table, cell by cell, for a redundant pair
  and for age and block replacement; and the tower property, the mean of
  what happened so far plus the rest valued from each history's state
  equal to the plan's value, for the same diagrams.
- **Learned / open questions:** the actual prices paid are not used (the
  log is priced at the plan's mean prices; a cost drawn from a
  distribution at its mean); the states the log does not give are asked
  for; and the spread of the expected side (is the actual inside what the
  plan allowed for?) is not given for a diagram yet: the comparison is
  against the plan's expectation.

### UC-46: "The expected NPV of these two plans is close. How bad could a run of each be, and how likely are we to miss our hurdle?"

- **Status:** built: `RepairableRBD.npv_distribution(horizon,
  discount_rate=, value_rate=, mc_samples=, seed=, hurdle=, tail=,
  contract=)` and `compare_npv(other, horizon, ...)`.
- **Who asks, and why:** someone choosing between plans whose expected
  NPVs (UC-19) are close: the mean hides the risk, and a plan with a
  slightly lower mean and a much better worst case is often the one to
  choose.
- **What they know:** the diagram with its costs and value per unit time
  while it works, the horizon and the discount rate; optionally a hurdle
  the NPV must reach and the share of worst runs to average.
- **How it is solved (simulated, model time):** the runs are the
  histories `simulate_timelines` makes (the event loop's, on either
  engine, by seed). Each is priced as UC-21 prices a log, with the
  diagram's costs at the times they fell (`rbd/_pricing.py`, which
  `plan_vs_actual` shares): each failure its repair and replace costs,
  each planned outage its preventive cost, each down span its downtime
  rate, the system's up spans their value, each discounted exactly; the
  acquisition costs are paid at 0 in every run. Only the runs' present
  values are kept, made in parts of 2,000. A cost the histories do not
  record (a standby group's units' failures, tests, replacements that
  take no time, calendar prices, set-up costs, a partial restoration's
  replacement) refuses the diagram, naming the cost and
  what to run instead, rather than being left out. The summaries are
  plain: the mean with its normal interval, percentiles as order
  statistics (the `ceil(p n)`-th lowest) with binomial intervals, the
  share below the hurdle, and the CVaR as the mean of the `ceil(tail n)`
  lowest.
- **The answer:** each run's NPV, value and cost; the mean, its interval
  and the split of value and cost by category and component as in
  `ExpectedNPV`; P10, P50 and P90; the chance below the hurdle; the CVaR.
- **Contracts and two plans (part 2):** an `AvailabilityContract` pays,
  in each period, `bonus_per_point` for each percentage point of the
  run's availability above `target` and charges `penalty_per_point` for
  each below, at most `cap` either way, at the period's end; a period's
  availability is the share of it covered by the system's up spans in
  that run. `compare_npv` prices both plans' runs from one seed; each
  component draws from its own named streams, so one in the same place
  in both plans draws the same numbers (as `compare`), and each run's
  difference comes from how the plans differ. A plan with a component
  whose draws are not streamed is refused, naming it.
- **Checks:** `test_npv_distribution.py`: the mean against `expected_npv`
  (an exponential pump, an age-replaced Weibull unit, a crew-limited
  pair); with no discounting, each run against the simulation's own cost
  (`cost().samples`) and uptime, also for crews with Weibull lives, which
  the exact methods refuse; the engines run for run; a run made in parts
  equal to one made whole; the percentiles, CVaR and a binomial interval
  by hand. `test_npv_contracts.py`: a target of 0 pays nothing; a
  system that never fails paid in closed form (a capped bonus each
  month); each run's payments against its own availability by period,
  and the periods' uptime against the run's; identical plans differ by
  exactly 0, with and without a contract; the difference's variance
  below the sum of the plans' (about half the independent standard
  error with a contract, whose thresholds pay on small differences in a
  period's uptime the plans do not share).
- **Learned / open questions:** a cost given as a distribution is charged
  at its mean (`priced_at_mean`), so the spread is that of the events
  alone; the costs the histories do not record could come from the event
  loop's own records, which would change both engines.

### UC-55: "The plan's NPV rests on a discount rate we argue about and on lives fitted to a few failures. How much does each move it, and would the intervals we chose still be the best?"

- **Status:** built: `expected_npv(..., rate_sensitivity=True)`,
  `RepairableRBD.expected_npv_uncertainty(horizon, uncertainty, ...)`
  and `optimal_replacement_intervals(..., robust=True | "reoptimise",
  uncertainty=, n_draws=, seed=)`.
- **Who asks, and why:** an owner signing off a plan valued by UC-19 and
  chosen by `optimal_replacement_intervals`. The finance team's rate is
  a range, not a number, and the lives came from a few dozen failures:
  a choice made on point estimates can be the wrong one once that is
  counted, and they want to know how often, and what it would cost.
- **What they know:** the diagram, its costs and value rate, the horizon
  and the rate; which models are fitted (their covariance, or a Bayesian
  posterior), or distributions over parameters given by hand.
- **What they want back:** `d NPV / d r` with its error; an interval on
  the NPV, which inputs make it wide, and what was held fixed; for a
  choice of intervals, the share of plausible models under which it is
  still the best and the expected regret where it is not.
- **How it is solved:**
  - *The rate (numerical):* the NPV is analytic in `r`, each amount at
    `t` weighed by `exp(-r t)`. Its derivative is Richardson's
    extrapolation of differences on steps `h = 0.01 / T` and `2h`
    (`T` the time valued, so the steps follow the unit of time), central
    where the rate stays at least 0 and one-sided from it otherwise; a
    third of the two steps' difference is the stated error. Weighting
    each cash flow by `-t exp(-r t)` through every integral would be
    exact, but would add a second weighted path to each of the cost
    integrals; four more valuations do it with the code that values the
    plan.
  - *The models (sampled):* the draws are those of
    `expected_cost_rate_uncertainty` (`_repairable_uncertainty.draws`
    and `build`), each draw valued by `expected_npv` exactly. The shares
    are the delta method's, as `uncertainty_importance(method="delta")`
    gives them: the NPV's derivatives in each drawn parameter (a central
    difference of 1% of the lever, the step `parameter_sensitivity` takes
    over time) against the parameters' covariance. Models no input draws
    are listed as held fixed.
  - *The choice (sampled):* under each draw, the choice and its
    neighbours (each interval one step either way: the next allowed
    interval, or a quarter longer or shorter in a continuous search) are
    scored by the plan's own objective and target, all under the same
    draw, so their differences are not blurred by draws they do not
    share; the share in which the choice wins and its mean regret
    follow. Choosing again under each draw (`"reoptimise"`) is offered,
    slower; its share of draws with the same choice is given only for
    allowed intervals, as a continuous search never lands on the same
    point twice.
- **Today:** `expected_npv` at two rates by hand gave a difference
  quotient; the uncertainty methods gave the availability and the cost
  rate, not the NPV; nothing scored a choice under other models.
- **Checks:** `test_npv_uncertainty.py`: the rate sensitivity against a
  closed form for an exponential pump at 8% a year, at 0 and just above
  it (to 1e-6, inside the stated error), against a small change of rate,
  and a salvage's part against `-T exp(-r T) S`; each draw's NPV against
  the closed form at its drawn rate; the delta method's variance against
  the closed form's derivative and the fit's `lambda^2 / n`, and the
  draws' variance against it; seeded draws repeat; a choice built to flip
  under one of two lives is the best in exactly the draws of the other,
  with the regret the other life's own choice saves, and the same by
  re-optimising.
- **Learned / open questions:** the neighbours' regret is a lower bound
  on the regret against each draw's own optimum; the two agree where the
  optimum moves at most one step. The delta method's shares assume the
  NPV close to linear over the spread: the draws' variance beside
  `delta_variance` says whether it is.

### UC-56: "We have outages planned at 800, 1,500 and 2,500 hours. The pump is replaced every 1,200 hours anyway; at an outage, we'd rather change it if it's past 500. How many pumps does each outage take, and what does that save?"

- **Status:** built (simulated): a shutdown's `"opportunistic": {node:
  age}` (or one age for every block in its scope under age
  replacement).
- **Who asks, and why:** the planner setting an outage's scope. The
  plant is down anyway, so a part close to its own replacement is
  changed then rather than on its own downtime later: that is how real
  shutdown scopes are set, and it changes what the plan costs and is
  worth.
- **What they know:** the diagram; the part's own age replacement (its
  interval, cost and duration); the outages (UC-24: times or a period, a
  duration, a scope); for each part, the age past which it is changed at
  an outage.
- **What they want back:** for each outage, how many parts it is
  expected to take, with a confidence interval; the plan's availability
  and cost with the rule, against without it.
- **How it is solved (simulated):** at each start of the shut, the
  Python event loop renews each listed part that is working and at
  least that old (its time since it was put in), as a stop of a
  maintenance group renews a member early (`_event_loop._stop`): the
  failure or replacement it had ahead is cancelled, the renewal takes
  the part's own preventive cost and duration, and its next age
  replacement counts from then. A part in such a shut's scope is worked
  on there, so a shut that stands idle blocks still does not stand it
  still. The runs count each outage's renewals (and their squares), so
  `outage_jobs` gives each outage's mean count, standard error and 95%
  interval; `opportunistic_renewals` gives each part's total, and the
  renewals are charged as the part's preventive work. An age of `inf`
  renews nothing, so the rule is then the plan without it, seed for
  seed. A shut in no time with the rule is a moment that renews parts
  and holds nothing.
- **Exact routes:** refused: a part's renewal at an outage depends on
  its age then, which ties its renewals to the outages' times in a way
  the curves over time (each part's renewal process on its own
  calendar) do not follow. The message names `availability()` and
  `cost()`. numba's loop does not run it: `_compiled.unsupported` names
  the shutdowns, and the Python loop runs them (as for any shut that
  takes time, costs or falls at given times).
- **Today:** `"older_than"` on a task bound to a shutdown renews a part
  with no schedule of its own at a shut once that old; a part with its
  own age replacement could not be renewed at an outage.
- **Checks:** `test_outage_opportunistic.py`: an age of `inf` gives the
  plan without the rule to the last bit (and the exact methods the same
  value); with the age of the first outage's time, the count there is
  the chance the part never failed, `R(800)`, within its interval, with
  the binomial standard error; the counts by outage add up to the
  part's renewals; the renewals cut the cost and are charged as
  preventive work; a single age is the dict of every age-replaced part
  in scope; chunks merged give the run's counts; saving; the refusals
  and the checks of the rule. The catalogue's "outages renewing
  opportunistically" runs every analysis against its route.
- **Learned / open questions:** the renewals are charged in the
  `"preventive"` category, as a group's opportunistic renewals are,
  rather than a category of their own (which would widen every cost
  table and both engines' arrays); their cost is the count times the
  part's preventive cost. `npv_distribution` prices them (UC-61).

### UC-57: "Our selling price steps up next year, labour rises 4% a year and parts 2%, and finance quotes a real rate of 5% with 3% inflation. What is the plan worth, and does it matter whether we work in real or nominal money?"

- **Status:** built: `expected_npv(..., value_rate=ValueSchedule(times,
  rates), escalation={stream: g}, rates="nominal" | "real",
  inflation=)`; on dates, `escalation=` beside `discount=` in
  `repyability.schedule` (`Task.due`, `Task.long_run_value`,
  `work_by_period`, `failures_by_period`, `events_by_period`,
  `task_value`).
- **Who asks, and why:** the analyst valuing a plan over ten years. A
  constant price and constant costs (UC-19) overstate a plan whose
  costs rise faster than what it earns, and a rate quoted in real terms
  applied to nominal prices (or the other way round) is the commonest
  error in an NPV.
- **What they know:** the diagram, its costs (`repair_cost`, ...,
  `downtime_cost_rate`, the lost production) and what it earns while it
  works (`value_rate`), each in today's money; how each stream grows; the
  rate and whether it is real or nominal, and inflation.
- **What they want back:** the NPV, its cash-flow table with every
  stream grown (nominal, or real), and the basis stated on the result.
- **How it is solved (as exact as `expected_npv` is):**
  - *What the plant earns:* `value_rate` is a number or a
    `ValueSchedule(times, rates)`, steps in model time (plain data, so a
    valuation can be saved and made again; a function of time is
    refused). A step `v` on `[a, b)` with growth `g` adds `v * integral
    exp(-(r - g) t) A(t) dt`, the uptime integral `expected_npv` already
    takes at the rate `r - g`, differenced at the steps' ends.
  - *Escalation:* each stream (a cost category: `repair`, `replace`,
    `preventive`, `inspection`, `component_downtime`, `system_downtime`,
    `setup`, `shutdown`, `crews`; a component, which takes precedence
    over its categories; `"value"`; `"salvage"`) takes a continuous
    growth `g`: an amount at `t` is its price times `exp(g t)`, so its
    present value is the cost discounted at `r - g`, worked out once for
    each distinct `r - g` by the existing discounted cost
    (`_windows.expected_cost`, or `_windows._discounted_cost` for a
    rate below 0, which is valid over a horizon); the undiscounted
    column is at `-g`. The downtime cost rate and the value rate are the
    streams `system_downtime`/`component_downtime` and `value`: no
    second notion of either. The acquisition is paid at 0 and does not
    grow.
  - *Real or nominal:* `rates="nominal"` (the default) takes the rate
    and growths as given. `rates="real"` takes them in real terms; with
    `inflation=i`, every amount is made nominal (the rate `r + i`, every
    stream's growth `g + i`, a stream with none `i`), which leaves every
    present value as it is and makes the undiscounted columns nominal.
    The result states `rate_basis`, `inflation`, `escalation` and
    `value_schedule`, and its `discount_rate` is the rate given, so the
    equivalent annual cost is the level annuity in the basis given: a
    level real annuity with real rates.
  - *On dates:* `escalation=` is an effective yearly rate (0.03 for 3% a
    year) beside the yearly `discount=`: a cost `y` years after the start
    is worth `((1 + e) / (1 + d)) ** y` of its price today, so every
    present value (and the equivalent annual cost) is taken at the net
    yearly rate `(1 + d) / (1 + e) - 1`, which may be below 0 over a
    window; for ever (`Task.long_run_value`) it must be above 0. The
    expected costs by period stay in today's prices, as the note says.
- **Today:** a constant `value_rate` and no growth; a schedule had to be
  valued as runs over sub-horizons and added up by hand.
- **Checks:** `test_npv_escalation.py`: an exponential unit with a
  value growing at `g` against the closed form; a schedule of two steps
  against the sum of the steps valued apart; a stream's escalation
  against the cost discounted at `r - g`, also with `g > r`; the
  real/nominal identity (every stream lifted by inflation leaves the NPV
  unchanged); no growth and a number for the value give every field
  unchanged to the last bit; a component's growth taking precedence over
  its category's; a salvage grown at its own rate; the refusals. On
  dates (the same file): a task's present value with
  escalation against the factor worked by hand, `events_by_period`
  against `expected_npv` with the matching continuous rates, and for
  ever refused when the escalation reaches the rate.
- **The spread and the log (part 2):** `npv_distribution`,
  `compare_npv` and `plan_vs_actual` take the same `value_rate`,
  `escalation`, `rates` and `inflation` (and `best_switch_time` the
  last three, its value being its two rates). A run or a log is priced
  as UC-46 and UC-21 price it (`rbd/_pricing.py`), each amount at `t`
  its price times `exp(g t)` for its stream, so discounted at `r - g`:
  an event `p exp(-(r - g) t)`, a rate over a span `v * integral
  exp(-(r - g) t) dt` in closed form, and an up span cut at each step of
  the schedule, each piece at its step's rate. A contract's payments
  grow only with the inflation of real rates. `plan_vs_actual`'s plan is
  `expected_npv` with the same arguments; the rest from now is valued on
  its own clock with the schedule's steps from now (the step in force
  then from 0), and each of its rows grown to now (`exp(g now)`) and
  discounted to the start (`exp(-r now)`), so a stream's present value
  is `exp(-(r - g) now)` of the rest's. Checks in
  `test_npv_value_runs.py`: each run's value and cost against a
  reference written out from its history (to 1e-12); the runs' mean
  against `expected_npv` with the same schedule and growths (4,000
  runs, a 99.99% interval); one step and no growth equal to the rate
  alone to the last bit; inflation leaving the present values (1e-12)
  and lifting the undiscounted; two plans with an escalation that names
  a component of only one; a pump that all but never fails, whose rest
  from now is the plan after now row by row (1e-6), and whose log is
  priced by hand; so far plus the rest averaging to the plan with a
  step inside the rest (the tower property, 60 histories); the switch
  time's present values unchanged by inflation; a JSON request.
- **Learned / open questions:** a contract's payments are no stream of
  `escalation` (they grow with inflation alone): terms that index them
  would be a contract's own option.

### UC-58: "At the outage the pump is overhauled, and leaves as if 3,000 hours old whatever it was before. What does that do to the plan?"

- **Status:** built (simulated): a `"preventive"` spec's `"restore_to":
  age`, for a task at a shutdown (UC-24) or on the part's own calendar.
- **Who asks, and why:** the planner whose overhaul contractor states
  the result as an age ("as good as 3,000 hours"), not as a share of the
  age taken away, which is what a Kijima factor (`"restoration"`) says.
- **What they know:** the diagram, the outages and their scope, the part's
  life, and the age the overhaul leaves it at.
- **What they want back:** the plan's availability, failures, costs and
  their spread, as for any simulated plan.
- **How it is solved (simulated):** the task sets the unit's virtual age
  to the age given, whatever it was (`_ImperfectComponent.maintained`,
  beside Kijima I and II, `_Restoration` with `kijima="to_age"`); the
  unit's next life is drawn given that age from the end of the task
  (`_aged_life`, the virtual age's own stream), as a Kijima restoration
  draws it. `restore_to=0` is a renewal: it is taken as no restoration at
  all, so it is exact wherever a renewal is and the same simulation, seed
  for seed. A part restored to an age is refused by the exact methods, as
  a Kijima restoration is ("restored only partly ... to a virtual age of
  3000"), naming `availability()` and `cost()`; numba's loop leaves it to
  Python (`_compiled.unsupported`: a restoration, and a dated shut).
  `"restore_to"` and `"restoration"` are one or the other. The age is a
  number; an age that depends on the age before is a Kijima factor
  already.
- **Today:** a Kijima factor only, whose age left depends on the age
  before.
- **Checks:** `test_restore_to.py`: `restore_to=0` the renewal's run to
  the last bit, and its exact values; a memoryless part unchanged by any
  age (against the renewal's exact availability, within the run's
  error); a Weibull part whose repair outlasts the window (so it fails at
  most once) against its survival from the age given, `R(TAU) R(a + t -
  TAU - D) / R(a)`, integrated, at two ages, and far from the renewal's;
  the refusals; saving. The catalogue's "outages restoring to an age"
  runs every analysis against its route.
- **Learned / open questions:** an age that is a function of the age at
  the outage was left out; Kijima II is the proportional one.

### UC-62: "Our 250-hour service is invasive: the fit of the pump's history says each one leaves it older than it was (Kijima II, q = 1.1). What does that do to the failures, and is the service worth stopping?"

**Who asks.** A reliability engineer whose fitted restoration factor
(surpyval's `GeneralizedRenewal` on the pump's repair and service history)
came out above 1: services that disturb a working unit (a seal reseated,
a bearing reassembled) and make an early failure more likely.

**What they know.** The pump's life (Weibull, scale 1,000 h, shape 2.5),
replaced at a failure for 400; the service every 250 h, 4 h long, 20 a
time; the fitted factor, Kijima II, `q = 1.1`.

**What they want back.** The failures with and without the service, and
the value of the service against not doing it, at 7% a year over a year.

**How it is solved.** A `"preventive"` task with a `"restoration"` whose
`q` is above 1: Kijima's definitions hold for any `q` from 0, a service
after an operating time `x` since the last action taking the virtual age
`v` to `v + q x` (Kijima I) or `q (v + x)` (Kijima II). Below 1 the task
takes age away; at 1 it leaves the unit as old as it was; above 1 it adds
age. Only a task may do that: a repair's `q` stays in [0, 1], as a
repair restores a failed unit. The simulation takes it as it takes any
restoration (the compiled engine hands a restoring task to the Python
loop, as before), and `compare_npv` values the plan with the service
against the plan without it, run by run on the same random numbers.

**The hand-worked answer.** With no failure, tasks at 250, 500 and 750 h
leave the virtual age at 275, 577.5 and 910.25 h under Kijima II with
`q = 1.1` (275, 550 and 825 h under Kijima I), against 250, 500 and 750 at
`q = 1`. The chance of no failure by 1,000 h is `exp(-sum of H over the
four pieces)`, `H(t) = (t / 1000)^2.5`: 0.295142 under Kijima II and
0.326487 under Kijima I with `q = 1.1`, and `exp(-1) = 0.367879` at
`q = 1`, which is the pump with no service at all (each piece of `H`
continues where the last stopped). At `q = 0.5` it is 0.703789 (II) and
0.618741 (I): a good service.

Simulated (4,000 runs, seed 8, instant repairs, services in no time),
the failures over 3,000 h are 2.974 at `q = 1` (the renewal function
without service: 2.973), and under Kijima II 3.233 at 1.1 and 3.488 at
1.2 (standard errors 0.013). With the 400 failure cost, the 20 service
and 4 h service outages, over a year at 7% (4,000 runs, seed 3), the
service at `q = 1.1` is worth -918.97 (standard error 11.67) against no
service, worse in 87% of the runs: worth stopping. At `q = 0.5` it is
worth +1,528.85 (standard error 13.00).

**Checks.** At `q = 1` the chance of surviving the pieces equals the
pump without service, exactly, and Kijima I and II give the same runs to
the bit (`v + 1 x` and `1 (v + x)` are the same number); the survival
chance with `q = 1.1` against the closed form above; failures and cost
rise with `q` above 1; a repair's `q` above 1 is refused, saying only a
task's may be.

**What was learned.** Nothing new was needed in the engines: Kijima's
arithmetic was already right for any `q`; only the check on the spec
held it to [0, 1]. A factor above 1 is how harmful or invasive
maintenance is modelled, and the NPV comparison is how a plan shows that
a task is worth stopping.

### UC-63: "One pump's plan: what are its tasks and prices, and how does the diagram take it?"

**Who asks.** Anyone who writes a unit's maintenance down once and uses
it in several places: on a diagram today, valued alone or on the
calendar later. And a caller (the Reliafy app) that keeps a unit's plan
as a record and hands it over, rather than assembling a spec dict.

**What they know.** The pump's life (Weibull, scale 1,000 h, shape 2.5)
and its repair time (exponential, mean 10 h); a service every 4,000
operating hours that renews it, 20 a time, taking a fixed 4 h; a repair
costs 400 and its own downtime 5 an hour. It runs 60% of the time.

**What they want back.** One object that says all of that, each task in
its own clock (4,000 operating hours is kept as operating hours, a
task at the site's shutdowns as a binding to them by name, a calendar
policy as the policy), that a `RepairableRBD` node takes as it takes
the spec dict, and that saves and loads.

**How it is solved.** `UnitPlan(life, repair, tasks=(...),
prices=Prices(...))`, frozen plain data. Each task is its own object:
`AgePolicy`, `BlockPolicy` (on an interval or at given times),
`ConditionPolicy` (a threshold or a measured level), `CalendarReplacement`
(a `CalendarPolicy` or `ShutdownPolicy`), `AtShutdown` (bound to the
site's shutdowns by name), `Inspection` and `Upgrade`; a task that
restores partly carries a `Restoration` (Kijima, to an age, or a fitted
process; every so many tasks a replacement), and imperfect repairs an
`ImperfectRepair`. A task's own price is the task's (`cost`), as each
task has its own; the unit's prices (repair, replace, its own downtime,
acquisition) are `Prices`. A task counted in operating time says so
(`basis="operating"`). What belongs to the system stays on the diagram:
structure, system downtime and value, crews, spares, the site's
shutdowns and common-cause groups (a plan only names its maintenance
group and its place in the crews' queue).

The spec dict is the plan's shorthand: `UnitPlan.from_spec(spec)`
reads one, and `plan.spec()` writes the spec the diagram's engines read.
A node given a plan is the node given its spec, so every analysis, the
routes, the levers and saving take it unchanged; `RepairableRBD.plans`
reads each spec node back as a plan.

**The hand-worked answer.** The plan above and the spec

```
{"reliability": Weibull(1000, 2.5), "repairability": Exponential(0.1),
 "repair_cost": 400, "downtime_cost": 5, "duty": 0.6,
 "preventive": {"policy": "age", "interval": 4000, "basis": "operating",
                "cost": 20, "duration": ExactEventTime(4)}}
```

are one node: `plan.spec()` is that dict, and `from_spec` of it is the
plan. Seeded, a diagram of plans and the same diagram of specs give the
same runs to the last bit, on both engines; their routes are the same.

**Checks.** Every spec of the test catalogue reads into a plan and back
to the same spec (up to keys at their defaults, such as an `"age"`
policy or an `"instant"` duration); a reader that met a key it does not
place refuses, so a new spec key cannot be dropped quietly. Diagrams of
plans against diagrams of specs, seeded, to the bit, on both engines; a
plan node in the catalogue; a plan saves and loads; a plan with two
preventive tasks is refused, saying a spec holds one (bind it to several
shutdowns instead).

**What was learned.** The spec's keys already were a unit's plan; what
the plan adds is a name for each task and its clock, and one place a
unit's maintenance is written down. Salvage, a unit's own escalation and
retirement were the analyses' arguments (`expected_npv(salvage=,
retire_at=, escalation=)`); they joined the plan when a plan was valued
alone (UC-65).

### UC-65: "What is this pump's plan worth alone over eight years at 7%, and its cash flows by year?"

**Who asks.** The planner of UC-63, who has written the pump's plan down
once and wants it valued without drawing a diagram around it: its net
present value, a yearly table to put beside the budget, how bad a run
could be, how the plan has gone so far, and its cost per year in the
long run. The caller (the Reliafy app) values a unit's plan alone as it
values a system.

**What they know.** The pump's life (Weibull, scale 20,000 h, shape
2.5) and repairs (exponential, mean 24 h, 6,000 each); a renewal at
12,000 h of age for 1,500; bought for 40,000 and worth 5,000 when it
goes; eight years of 8,760 h at 7% a year.

**What they want back.** The NPV, the cash flows by year and category,
and the other valuations a diagram has, with the salvage (and, where
given, the growth of its prices and its retirement) held by the plan.

**How it is solved.** A plan alone is the diagram of its one node,
`UnitPlan.diagram()` (the node named `UnitPlan.NODE`, `"unit"`), and
each valuation of the plan is that diagram's method with the same
arguments: `expected_npv`, `cash_flows` (its table), `npv_distribution`,
`plan_vs_actual` (one `Timeline`), `expected_cost`,
`expected_cost_rate`, `events_by_period` and `failures_by_period` (the
diagram's events on the calendar), the optimisers and `analysis_routes`.
There is no second route: the diagram's exact methods are already the
single unit's (its renewal curves, age and block replacement, the tested
unit's model), so a plan alone and a diagram holding it agree by
construction, exact where the diagram is exact and simulated where it
simulates. A `calendar` (a `schedule.CalendarContext` whose duty names
`"unit"`) puts the diagram on it first, as `CalendarContext.diagram`
does.

The unit's own money moves onto the plan: `salvage` (a number or a
function of the age of the unit in service), `escalation` (one growth
rate for all its prices) and `retire_at`. They are spec keys of the same
names, so a diagram holding the plan takes them too: `expected_npv`
takes each as its argument of the same name for the node, an argument
given for the node taking precedence (a system's salvage, a number, is
added to the units' own). A unit retiring is the system's retirement
only where it is the diagram's only unit; in a diagram of several it
retires alone (`expected_npv(retire_at={node: t}, retired=)`): from `t`
it is held failed (its paths lost, the default, which never invents
availability) or as a junction, its own costs stop, the system's go on
(its value and downtime as the diagram held so), and its salvage is
credited at `t`. Each stretch between retirements is the diagram with
the units retired so far held, which holds because the units are
independent: a unit tied to the others (shared crews, a maintenance
group, a common cause, a shutdown) is refused. The runs and
`plan_vs_actual` refuse a unit retiring alone, pointing to
`expected_npv`. The valuations of
runs (`npv_distribution`, `compare_npv`) end at a plan's retirement and
credit every run its salvage at the end (a number; a salvage by age as
`worth(age)` of each run's own unit in service at the end, if it is up:
a unit that keeps a virtual age (imperfect repair, a fitted process, a
task that restores it partly) at its virtual age; one tested for hidden
failures credited nothing if failed unseen, else at its age since it
was put in, less its tests off line; a group's units each at its own
age, a spare's the life it used up waiting; a nested diagram's
components each by their own salvage; a life on a duty plan that
changes at its operating age; only a fitted ARI process, which keeps no
virtual age, is refused),
as `expected_npv` does, and so do `best_switch_time` (every
candidate credited alike, the candidates up to the retirement) and a
service's `task_value` (each side's runs to the plan's retirement, its
salvage in the period it falls in, no service after it).
`plan_vs_actual` values the plan and the rest from today to the
retirement, the salvage credited there as `expected_npv` credits it
(by age, from the unit's state today); a log past the retirement is
refused.

**The hand-worked answer, today.** The diagram of the one pump,
`RepairableRBD([("s", "p"), ("p", "t")], {"p": spec})`, valued by
`expected_npv(8 * 8760, discount_rate=log(1.07) / 8760, salvage={"p":
5000}, periods=8)`: an NPV of -48,719 (the purchase 40,000, the repairs
and renewals, the salvage's 5,000 worth 2,910 now), an equivalent
annual cost of 8,357 a year, and eight yearly rows by category. The
plan alone gives the same numbers to the last bit.

**Checks.** `test_plan_alone.py`: over every plan of the catalogue that
stands alone (no maintenance group, site shutdown or standby group), the
plan's `expected_npv`, `expected_cost_rate` and `npv_distribution` equal
a diagram holding it, drawn and named otherwise, to 1e-12 (or both
refuse alike), and their routes agree; on a calendar of one share and
of weekday shifts, the plan's runs, expected NPV and events by week
equal its diagram on that calendar; the plan's salvage (a number and by
age), escalation and retirement equal the arguments for the node, the
arguments win, a retirement past the horizon is the horizon, a unit's
retirement in a diagram of several retires it alone (and
`test_retire_units.py`: a pair's NPV with one retired against the
closed form of the exponential units' availabilities, to 1e-6, failed
and as a junction; at the horizon nothing changes; at 0 it is the
diagram with it held broken; a salvage at its retirement is worth
`value * exp(-r t)`), the runs refuse a
salvage by age, and the terms save and load; each run of a plan with a
salvage or retirement is the run of the plan without, to the
retirement, plus the salvage `expected_npv` credits
(`test_npv_runs_own_end.py`); the exact NPV is the mean of
4,000 seeded runs within four standard errors.

**What was learned.** The diagram is already the single unit's engine;
what a plan alone needed was a name for its one-node diagram and the
unit's money on the plan. The simulated runs then took a salvage given
as a number and a retirement, then a salvage by age, each run's unit
in service at the end (`test_salvage_by_age_runs.py`: a memoryless unit
against `E[min(X, H)]`, a plain one against `expected_npv`, within four
standard errors; then the units whose age is more than the time since
they were put in, `test_salvage_by_age_cases.py`: each age by hand from
its run's histories, and closed forms for hidden failures, standby
pairs, identical units, nested units and a duty plan), and
`expected_npv` one unit's retirement within a
system (`test_retire_units.py`); the switch times, logs and services' values took the plan's
own end next (`test_npv_runs_own_end.py`).

### UC-59: "A stop every month renews the bearing; a long overhaul every year renews the pump and the bearing. What is the plan worth over two years, and how available is the line?"

- **Status:** built: a `"preventive"` spec's `"shutdown"` given as a
  list of the diagram's shutdowns, valued over a horizon by
  `point_availability`, `mission_availability`, `expected_events`,
  `expected_cost`, `expected_npv` and the windows, with no simulation.
- **Who asks, and why:** the planner of a line with more than one kind
  of outage (a short stop on a period, a long overhaul, outages at given
  times), where a part is renewed at each and the renewal takes as long
  as the outage it is done in. A task naming one shutdown could not say
  this.
- **What they know:** a line of a pump (Weibull 20,000 h and 2.5,
  repairs lognormal, 6,000 each, renewed only at the overhaul for 2,500)
  and a bearing (repairs exponential, 12 h mean, 1,500 each, renewed at
  both for 200); 400 an hour down; a stop every 730 h for 4 h (300 each)
  and an overhaul every 8,760 h from 365 for 48 h (5,000 each).
- **What they want back:** the plan's cost and value over two years at
  8% a year, the failures and renewals to expect, and the availability.
- **How it is solved:** numerically, as a plan of replacements is
  (UC-51): the part's renewal times are its shutdowns' starts merged, up
  to the first after the horizon, and the part is followed stretch by
  stretch between them (`_block_replacement.plan_availability`), each
  stretch's grid taking the length of the renewal that starts it (its
  own shut's) and the shutdowns' windows holding it off line
  (`MaskedCurve`). The simulation (Python only) renews it at the first
  start of any of its shutdowns, for that shut's length
  (`_Preventive.due`, `duration_at`). The renewals repeat on no interval
  of the part's own, so the long-run methods refuse and name the values
  over a horizon; a long-run state, or a state in a renewal (whose shut
  and time left are unknown), is refused. Two of the shutdowns may not
  start at one time (which renewal is done there would not be said),
  and a task kept unless due or restoring part-way is at one shutdown.
- **Today (by hand, before it was built):** only by simulation, the
  bearing drawn as a block renewed at each of two shuts not being
  something a task could say. By hand for an exponential bearing alone
  (rate 1/3,000, repair rate 1/12), its two-state chain between
  renewals of their own lengths (4 h, 48 h) gives its availability at
  1,000 h and 9,200 h as 0.996016 and 0.996423; the built curve gives the
  same to 1e-6. The line, as built:

  ```python
  plan = line(bearing_life)   # the diagram above
  value = plan.expected_npv(2 * 8760.0, discount_rate=np.log(1.08) / 8760.0)
  # exponential bearing (mean 3,000 h): 141,458 undiscounted, 132,815
  #   discounted; availability 0.98508; bearing failures 5.75, renewals 24.90
  # Weibull bearing (3,000 h, 2.0): 113,611 and 106,994; availability
  #   0.98810; bearing failures 1.31, renewals 24.96
  ```

  The simulation (`cost(2 * 8760, mc_samples=4000, seed=1,
  control_variate=False, conditional=False)`) gives 141,834 ± 318 and
  113,493 ± 175.
- **Checks:** `test_shutdowns_several.py`: a list of one shutdown its
  shutdown's values to 1e-10 (on a period and at given times); two
  shutdowns of one length one shutdown at the merged times, to 1e-10; an
  exponential block against its chain between renewals of two lengths
  written out by hand; the simulation from new, from an age and from a
  repair going on, 20,000 seeded runs within four standard errors; the
  refusals; saving. The catalogue's "a block renewed at several
  shutdowns" runs every analysis against its route.
- **Learned / open questions:**
  - A block kept unless due, or tested, at shutdowns at given times is
    still simulated: its recursion keeps the units by age on a grid of
    one step from one inspection to the next, which stretches of
    different lengths do not share.
  - The same block drawn as two blocks in series is a different system
    (each renewed at only its own shut), not this one.

### UC-22: "A pump's 6-monthly overhaul restores it only partly (a fitted Kijima restoration factor), and every third overhaul replaces it. What availability, failures and cost per year, against overhauls that renew it, and which overhaul interval is cheapest?"

- **Status:** built (#398): a `"preventive"` spec's `"restoration"` (Kijima's
  model and `q`, or a fitted `GeneralizedRenewal`), `"replace_every"` and
  `"replacement_cost"`, simulated by `availability`, `cost`,
  `simulate_timelines`, `cost_by_window` and
  `spares_demand(method="simulate")`. The record below is the
  hand-worked answer it was checked against.
- **Who asks, and why:** a reliability engineer who has fitted a
  generalised renewal process (Kijima I, restoration factor `q = 0.3`)
  to the overhaul history of a fleet of pumps, and finds an overhaul
  takes away only 70% of the age the pump gained since its last one. The
  plan has always been priced as if each overhaul renewed the pump.
- **What they know:** the pump's life in operating hours (Weibull,
  `alpha = 9000`, `beta = 2.5`); a failure is repaired by replacing the
  pump (as new; exponential repair, mean 24 h; 20,000 a failure, lost
  production included); an overhaul every 6 months (4380 h, on the
  calendar, skipped while the pump is down), 24 h long, 2,500 each;
  every third overhaul replaces the pump instead, at 9,000; a horizon of
  10 years (87,600 h), from new. Model time only: hours, no dates.
- **What they want back:** per year, the expected failures, overhauls
  that restore and overhauls that replace, the cost, and the
  availability; the same with overhauls that renew (today's plan); and
  the cost per year at overhaul intervals of 3, 4, 6, 9 and 12 months.
- **How it is solved:** simulated (a partial restoration does not renew
  the unit, so its failures are not a renewal process's and the exact
  methods do not apply). The overhaul keeps the unit's virtual age
  (Kijima I: `v <- v + q x`; Kijima II: `v <- q (v + x)`, `x` the
  operating time since the last action), and the next life is drawn
  given it, as an imperfect repair's is; every Nth overhaul since the
  unit was last put in as new replaces it. The event loop's imperfect
  repair stand-in keeps the virtual age, so it is extended to the
  overhaul; `availability`, `cost`, `simulate_timelines` and
  `cost_by_window` then answer. The interval is chosen by simulating each
  candidate with the same seed.
- **Today (by hand):** RePyability takes an overhaul only as a renewal.
  The overhauls that renew are exact; the partial restoration is
  simulated directly, written out from Kijima's definition in numpy
  (`direct` below, 4,000 runs a case).

  ```python
  import numpy as np
  import surpyval as surv
  from repyability import RepairableRBD

  ALPHA, BETA, YEAR = 9000.0, 2.5, 8760.0
  HORIZON = 10 * YEAR
  COSTS = dict(failure=20000.0, overhaul=2500.0, replacement=9000.0)

  def as_new(interval):  # today: the overhaul renews the pump
      return RepairableRBD(
          [("s", "pump"), ("pump", "t")],
          {"pump": {
              "reliability": surv.Weibull.from_params([ALPHA, BETA]),
              "repairability": surv.Exponential.from_params([1 / 24.0]),
              "repair_cost": COSTS["failure"],
              "preventive": {
                  "interval": interval, "policy": "block",
                  "duration": surv.ExactEventTime.from_params([24.0]),
                  "cost": COSTS["overhaul"],
              },
          }},
      )

  def direct(interval, q, kijima, every, n, seed):
      """Per year: failures, restoring and replacing overhauls, cost;
      and the availability, simulated from Kijima's definition."""
      rng = np.random.default_rng(seed)
      out = np.zeros((n, 5))
      for r in range(n):
          t = v = up = 0.0
          count = fails = restored = replaced = 0
          while t < HORIZON:
              # A life from virtual age v: H(v + X) = H(v) + E.
              life = ALPHA * ((v / ALPHA) ** BETA + rng.exponential()) ** (1 / BETA) - v
              due = (np.floor(t / interval) + 1.0) * interval
              if t + life <= due:  # fails first: repaired as new
                  if t + life >= HORIZON:
                      up += HORIZON - t
                      break
                  up += life
                  fails += 1
                  t += life + rng.exponential(24.0)
                  v, count = 0.0, 0
                  continue
              if due >= HORIZON:
                  up += HORIZON - t
                  break
              x = due - t  # operated since the last action
              up += x
              if every is not None and count + 1 >= every:
                  v, count, replaced = 0.0, 0, replaced + 1
              else:
                  v = v + q * x if kijima == "kijima1" else q * (v + x)
                  count, restored = count + 1, restored + 1
              t = due + 24.0  # down, not ageing, for the overhaul
          cost = (fails * COSTS["failure"] + restored * COSTS["overhaul"]
                  + replaced * COSTS["replacement"])
          out[r] = (fails, restored, replaced, cost, up / HORIZON)
      return out.mean(axis=0), out.std(axis=0, ddof=1) / np.sqrt(n)
  ```

  Per year (± one standard error of the simulated ones):

  | Overhaul every 6 months | Failures | Restoring | Replacing | Cost | Availability |
  |---|---|---|---|---|---|
  | Renews (today, exact) | 0.3055 | 1.896 | 0 | 10,851 | 0.99397 |
  | Renews (`direct`, `q = 0`) | 0.3015 ± 0.0026 | 1.896 | 0 | 10,771 ± 52 | 0.99398 |
  | `q = 0.3`, never replaced | 0.6819 ± 0.0024 | 1.895 | 0 | 18,375 ± 47 | 0.99296 |
  | `q = 0.3`, every third replaces | 0.4976 ± 0.0029 | 1.498 | 0.397 | 17,273 ± 51 | 0.99345 |
  | `q = 1` (as bad as old) | 1.0495 ± 0.0023 | 1.894 | 0 | 25,726 ± 46 | 0.99194 |

  The interval, with `q = 0.3` and every third overhaul replacing (seed
  2), against overhauls that renew (exact):

  | Interval | 3 months | 4 months | 6 months | 9 months | 12 months |
  |---|---|---|---|---|---|
  | Cost per year, `q = 0.3`, every third | 21,530 ± 37 | 18,577 ± 45 | **17,259 ± 51** | 18,117 ± 52 | 19,046 ± 50 |
  | Availability | 0.98876 | 0.99123 | 0.99346 | 0.99453 | 0.99526 |
  | Cost per year, renewing | 11,991 | **10,685** | 10,851 | 13,166 | 15,565 |

  Priced as renewals the plan costs about 10,900 a year; with the fitted
  restoration it costs about 17,300, and the failures more than double.
  Replacing at every third overhaul pays (17,300 against 18,400 for
  never replacing), and 6 months stays the cheapest interval, where the
  renewal model put it at 4.

- **Built:** the overhaul's `"restoration"` and every third replacing,
  simulated (4,000 runs, seed 1):

  ```python
  def restored(interval, q, every=None):  # as as_new, the overhaul partial
      preventive = {
          "interval": interval, "policy": "block",
          "duration": surv.ExactEventTime.from_params([24.0]),
          "cost": COSTS["overhaul"],
          "restoration": {"model": "kijima1", "q": q},  # or the fitted
      }                                                 # GeneralizedRenewal
      if every is not None:
          preventive["replace_every"] = every
          preventive["replacement_cost"] = COSTS["replacement"]
      return RepairableRBD(
          [("s", "pump"), ("pump", "t")],
          {"pump": {
              "reliability": surv.Weibull.from_params([ALPHA, BETA]),
              "repairability": surv.Exponential.from_params([1 / 24.0]),
              "repair_cost": COSTS["failure"],
              "preventive": preventive,
          }},
      )

  run = restored(YEAR / 2, 0.3, every=3).availability(
      HORIZON, mc_samples=4000, seed=1, control_variate=False
  )
  run.cost.by_category["repair"] / COSTS["failure"] / 10   # failures a year
  run.cost.mean / 10, run.mean_availability
  ```

  | Overhaul every 6 months | Failures a year | Cost a year | Availability | `direct` |
  |---|---|---|---|---|
  | `q = 0.3`, never replaced | 0.6785 | 18,308 | 0.99294 | 0.6819 ± 0.0024, 18,375 ± 47 |
  | `q = 0.3`, every third replaces | 0.4960 | 17,247 | 0.99344 | 0.4976 ± 0.0029, 17,273 ± 51 |
  | `q = 1` | 1.0454 | 25,642 | 0.99194 | 1.0495 ± 0.0023, 25,726 ± 46 |

  The interval, as the guide's example runs it (2,000 runs, seed 1):
  21,550, 18,574, **17,292**, 18,176 and 18,961 a year at 3, 4, 6, 9 and
  12 months, against `direct`'s 21,530, 18,577, 17,259, 18,117 and
  19,046 (± about 50).
- **Checks:** `test_pm_restoration.py`: the simulation against a direct
  simulation written in the test from Kijima's definition (failures,
  tasks that restore and that replace, up time) for Kijima I and II,
  the block, age and condition policies, every third replacing, and
  tasks and repairs that take time, within four standard errors; from a
  virtual age; with minimal repairs and Kijima II overhauls in no time,
  every second replacing, the cost over whole cycles is
  `Repairable.overhaul_replace_cost_rate`'s, which is exact; `q = 1` with
  instant tasks fails as often as with no task at all (the renewal
  function, exact); `q = 0` is today's diagram to the last bit, with its
  exact routes; the replacement's cost and spares; the refusals.
- **Learned / open questions:**
  - Priced as renewals the plan understates the failures by more than
    half and the cost by 60%, and puts the cheapest interval at 4 months
    instead of 6.
  - Kijima I with `q = 0` is a renewal only while the unit's repairs
    renew it: after imperfect repairs the unit is older than new at its
    task, and Kijima I takes away only the age since the last action.
    Only the exact renewals are taken as no restoration.
  - From a state, a unit restored partly takes its virtual age at its
    last action and its age since, and the age policy's clock is taken
    to start at that action (true when its repairs renew it; with
    imperfect repairs too, a state is refused, as for imperfect repair
    alone). With `"replace_every"` a state is refused: it does not say
    how many tasks the unit has had since it was renewed.
  - Not built: with a calendar policy (worked out for a unit put in as
    new), a degradation level (not a virtual age) or an opportunity (an
    early renewal at a group's stop); imperfect repair under the
    condition policy stays refused (the same code may take it, but it
    needs checks of its own).
  - Exact routes for a restoration that is clearly a renewal or clearly
    nothing are open: `q = 1` with instant tasks is no task at all (the
    unit's own renewal process), which the exact methods could take.

### UC-23: "The line is shut 3 days every 12 months (in hours); every block gets its PM then, and the bearing is replaced if it is older than 18 months. What availability, how much of the downtime and cost is the shut's, how many failures between shuts, and is a 9- or 18-month shut cheaper?"

- **Status:** built: `RepairableRBD(..., shutdowns={"line": {"period",
  "offset", "duration", "scope", "cost", "idle_ageing"}})`, and the work
  given on each block (`"preventive": {"shutdown": "line"}`, with
  `"older_than"`; `"inspection": {"shutdown": "line"}`). Exact and
  numerical by default, simulated on request.
- **Who asks, and why:** a maintenance planner whose line stops for an
  annual shutdown: every block's PM is done in it, the bearing is changed
  only if it has run long enough, and the shutdown itself costs (crews,
  scaffolding) and loses production. They want to know what the shut
  costs against what it buys, and whether to shut more or less often.
- **What they know:** three blocks in series: a pump (Weibull 30,000 h,
  shape 2), a motor (Weibull 60,000 h, shape 1.5) and a bearing (Weibull
  20,000 h, shape 3), each repaired in about 24 h (exponential) at 20,000
  a failure; PM at the shut 3,000, 2,000 and 1,500; the shut every 8,760 h
  for 72 h, 10,000 each; lost production 500 an hour; the bearing
  replaced at a shut once older than 13,140 h (18 months). Model time
  only: hours from new, no dates.
- **What they want back:** the long-run availability and cost a year;
  how much of the downtime and of the cost is the shut's; the failures a
  year (between shuts); the same at 9- and 18-month shuts; and over 10
  years the cost by category, with the shut's own.
- **How it is solved:** a task bound to a periodic shut *is* today's
  schedule aligned with it: a block renewed at every shut is block
  replacement with the shut's period, its offset (the time of the first
  shut, a new field of `_Preventive`, which numba's loop takes too) and
  a fixed duration; a test at every shut an inspection likewise (which
  has its offset already). They are translated so at construction, and
  every exact method takes them as it is. The shut's windows are a
  deterministic mask: the blocks in scope held at 0 inside them (the
  long run's grid and the integrals' pieces get the windows' edges), each
  start a planned outage at an exact time, and their Birnbaum importance
  0 while held. "Older than X" is replacement on condition with the
  decision on the age (`_condition_replacement`'s recursion, the rule
  generalised; numerical). The simulation has the windows as events of
  their own, the scope masked from the structure while the blocks' own
  processes go on (`"idle_ageing"`), or a paused block's pending event
  shifted by the shut's length.
- **Today (by hand):** equal block schedules approximate the shut: each
  block replaced every 8,760 h for 72 h (the bearing every 17,520 h,
  whatever its age), and the shut's cost added by hand:

  ```python
  def today(months, bearing_every=2):
      fixed = surv.ExactEventTime.from_params([72.0])
      blocks = {}
      for name, life in LIVES.items():
          every = bearing_every if name == "bearing" else 1
          blocks[name] = {
              "reliability": life,
              "repairability": E([1 / 24.0]),
              "repair_cost": 20000.0,
              "preventive": {
                  "interval": every * months * YEAR / 12, "policy": "block",
                  "duration": fixed, "cost": PM[name],
              },
          }
      return RepairableRBD(EDGES, blocks, downtime_cost_rate=500.0)

  rbd = today(12)
  rbd.mean_availability()                                  # 0.99053
  rbd.expected_cost_rate() * YEAR + 10000                  # 64,991 a year
  rbd.system_failure_frequency() * YEAR                    # 0.3869
  ```

  | Shut every | 9 months | 12 months | 18 months |
  |---|---|---|---|
  | Availability | 0.98826 | 0.99053 | 0.99292 |
  | Cost a year (shut's cost by hand) | 77,893 | 64,991 | 52,235 |
  | Failures a year | 0.2726 | 0.3869 | 0.5344 |

  Its shortcomings: the bearing is replaced every second block time
  whatever its age (one renewed by a repair a week before is changed
  anyway), so "older than X" cannot be said; the shut's cost is outside
  the model, and its downtime is the blocks' own (a block in scope with
  no PM of its own is not held down, and its failures in the shut count
  against the system); the shut's share of the downtime cannot be read
  off; shuts at given times cannot be said at all.
- **Built:**

  ```python
  def line(months, older_than=1.5 * YEAR):
      blocks = {}
      for name, life in LIVES.items():
          preventive = {"shutdown": "line", "cost": PM[name]}
          if name == "bearing":
              preventive["older_than"] = older_than
          blocks[name] = {"reliability": life, "repairability": E([1 / 24.0]),
                          "repair_cost": 20000.0, "preventive": preventive}
      return RepairableRBD(
          EDGES, blocks,
          shutdowns={"line": {"period": months * YEAR / 12, "duration": 72.0,
                              "cost": 10000.0}},
          downtime_cost_rate=500.0,
      )
  ```

  | Shut every | 9 months | 12 months | 18 months |
  |---|---|---|---|
  | Availability | 0.98791 | 0.99042 | 0.99301 |
  | Cost a year | 81,448 | 65,899 | **51,866** |
  | Failures a year (between shuts) | 0.4004 | 0.4237 | 0.5464 |

  At 12 months the unavailability is 0.00958, of which the shut's 72 h a
  year (0.00822) is 86%; of 65,899 a year, the shut's own cost and its
  lost production (10,000 + 36,000) are 70%. Over 10 years from new
  (`expected_cost(87600)`): 594,092, of which repairs 82,334, PM 48,672,
  lost production 373,086 and the shuts 90,000; 9 shuts, 8.98 planned
  outages, 4.10 failures. Simulated (2,000 runs, seed 1): 592,899 ±
  1,331, availability 0.991480 against the exact 0.991482, 4.04 failures
  and 8.99 planned outages. The 18-month shut is the cheapest: the
  shuts' downtime and cost fall faster than the failures between them
  rise.
- **Checks:** `test_shutdowns.py`: closed forms for one exponential
  block shut every period (long-run availability `(1 - D/P) mu/(lam +
  mu)`, the point availability in and out of a shut, the planned outages
  `sum A(s-)`, the failures, the costs and the NPV's discounting); a
  block renewed at a shut in no time against block replacement from the
  matching state, and a test at a shut against an inspection with its
  offset; a block renewed at each shut (instant repair) against
  `lam (1 - D/P)`; a paused block against its own curve shifted by the
  shuts' time; "older than" between renewing at every shut and never;
  the simulation against all of these, and the catalogue's diagrams
  exact against simulated (`test_catalogue.py`); both engines to the last
  bit on tasks at a shut in no time (`test_simulation_engines.py`);
  units (`test_units.py`); saving; the refusals.
- **Learned / open questions:**
  - A block's age at a shut is its age since it was put in service: one
    renewed at the last shut is `period - duration` old at the next, so
    "older than 12 months" at 12-month shuts replaces it every *other*
    shut.
  - In this example the 18-month shut is cheapest, though the failures
    between shuts rise by a third; the "today" approximation agreed on
    the ranking but not the failures (it changes the bearing on a fixed
    count of shuts, whatever its age).
  - Not built: the masked windows in numba's loop (it runs shuts in no
    time that cost nothing, which are their tasks alone); shuts of
    different calendars that overlap; standing still a block with a
    calendar of its own, inspected and kept at a shut, nested, in a
    standby or common-cause group, a maintenance group or behind limited
    repair crews (refused); importance measures, capacities, the first
    failure, spares, allocations, parameter uncertainty, the interval
    searches, `plan_vs_actual` and the crews' chain with shutdowns
    (refused, in plain words); adding the shut's period to
    `optimal_replacement_intervals`.

### UC-24: "Our outages are planned at given times, each with a duration and a scope: some blocks are renewed in them, others restored. What does the plan cost by window, and what is it worth?"

- **Status:** built: the same `shutdowns`, at `"times"` with a
  `"scope"`, and the work on each block (`"preventive": {"shutdown":
  ..., "restoration": ...}`). Where every job renews its block,
  `expected_npv`, `expected_cost`, `expected_events_by_window` and the
  other values over a horizon are worked out (numerical, as block
  replacement is); a job that restores part-way (UC-22) is simulated by
  `availability` and `cost`, and the exact methods say so. The outages
  end, so the long run is refused.
- **Who asks, and why:** a planner laying out the next four years of
  outages, irregularly spaced, who wants the plan's cash flows.
- **What they know:** the line of UC-23; outages at 3,000, 12,000 and
  26,000 h, 72 h each, holding the pump and motor off line, 10,000 each;
  the pump renewed in each, the motor restored (Kijima I, `q = 0.5`).
- **What they want back:** the cost in each year, and its present value.
- **How it is solved:** a block renewed at the outages is a block
  replaced at their times (UC-51's stretches, each replacement taking
  the outage's length), the blocks in scope are held off line by the
  same mask as on a period, at the given windows, and each outage's
  cost is an event at its start. Simulated: the windows at their times,
  the tasks bound to them at those times (`_Preventive.due` and
  `_Inspection.due` read the times). A list of different outages is
  several shutdowns, each with its times, duration, scope and cost; no
  second record (an `Outage` class) was added beside `shutdowns` and
  `Opportunities.on_diagram`, which writes them.
- **Today (by hand, and as built):** the cost to each year's end by
  `cost(..., seed=1)`, differenced (2,000 runs each):

  ```python
  ends = YEAR * np.arange(1, 5)
  runs = np.vstack([rbd.cost(float(t), mc_samples=2000, seed=1).samples
                    for t in ends])
  np.diff(runs, axis=0, prepend=0.0).mean(axis=1)
  # [56,733, 68,541, 78,070, 21,466]  (± 317, 489, 577, 525)
  ```

  Discounted at 8% a year at each year's middle, about 195,400. The
  shutdowns' own cost is 10,000 at each of the three, and each run's
  `shutdowns` is 3.
- **As built, renewing both blocks** (the motor renewed too, at 2,000):

  ```python
  value = rbd.expected_npv(4 * YEAR, discount_rate=np.log(1.08) / YEAR,
                           periods=4)
  # by year: 56,310, 69,046, 76,488, 19,447; in all 221,291.9
  value.cost                              # 193,053.7 at 8% a year
  value.cost_by_category["shutdown"]      # 26,697.2 (10,000 at each)
  rbd.mission_availability(4 * YEAR)      # 0.99238
  rbd.cost(4 * YEAR, mc_samples=4000, seed=1, control_variate=False)
  # 220,643 ± 551, simulated
  ```

  `maintenance_effect(4 * YEAR)` splits the 267.0 h down into 215.8
  planned and 51.3 unplanned, exactly (215.8 and 50.8 simulated).
- **Checks:** `test_shutdowns.py`: outages at a period's times against
  the shutdown on that period (to 1e-10); one outage renewing a unit
  against its replacement at that time and the window by hand; the
  simulation, from new and from a state; the outages' count and cost;
  the long run's refusal and what is left to the simulation.
- **Learned / open questions:**
  - A shut at time 0 holds its blocks but does no work on them (a
    task falls due after 0, as in the simulation).
  - A block is worked on at one shutdown's times: one renewed in
    outages of different lengths or scopes (two shutdowns) is not said
    yet. Nor is a restoration to a given virtual age: a job restores by
    a Kijima factor (UC-22).
  - A block kept unless it is due (`"older_than"`, on condition) at
    outages at given times is numerical too: its recursion runs stretch
    by stretch, the kept units' ages moved onto each stretch's grid
    keeping their mass and mean age. Equally spaced times give the
    period's values (within 1.1e-8), a threshold of 0 renewing at every
    outage, and an age past the horizon no task, to the grid's
    first-order error (2.2e-7 at 2,000 steps a stretch, halving as the
    steps double). One tested at them is simulated: its recursion is on
    a period.
  - `cost_by_window` and `expected_npv` could take the simulated mean
    where the exact one refuses, as `cost_by_window` does for its spread.

### UC-51: "The pump is replaced at 8,000, 14,000 and 26,000 hours, not on one interval. What will that plan cost over four years, what is it worth at 8%, and how available is the line?"

- **Status:** built: a `"preventive"` spec's `"replace_at"`, valued over
  a horizon by `expected_npv`, `expected_cost`, `expected_events`,
  `mission_availability` and `point_availability`, with no simulation.
- **Who asks, and why:** a planner whose jobs are a list of times (the
  outages there are, the budget's years), not one interval; every later
  question about a plan on a timeline (outages with a scope, upgrades,
  re-planning from today) starts from this one.
- **What they know:** a line of a pump (Weibull life, 12,000 h and shape
  2.5; repairs lognormal, 9,000 each) and a motor (Weibull 40,000 h and
  1.5; 15,000 each), 400 an hour while the line is down; the pump
  replaced at 8,000, 14,000 and 26,000 h if it is up, at 3,000 and about
  12 hours each.
- **What they want back:** the cost in each year and its present value,
  the replacements and failures to expect, and the availability.
- **How it is solved:** numerically, as block replacement is, which it is
  at any list of times: the unit is followed stretch by stretch, from
  one time to the next, each on a grid of its own length
  (`_block_replacement.plan_availability`), the repairs and replacements
  still going on at a time carried onto the next stretch's grid; each
  replacement is an event at its exact time, counted and discounted
  there. After the last time the unit is replaced no more. From a state
  the times are counted from now.
- **Today (by hand, before it was built):** only by simulation (the
  times given to a shutdown of no length, the pump's task bound to it),
  or, for equal gaps, block replacement on a period. As built (its own
  simulation, `cost(4 * 8760, mc_samples=4000, seed=1,
  control_variate=False)`, gives 91,867 ± 657):

  ```python
  plan = line({"replace_at": [8_000.0, 14_000.0, 26_000.0],
               "duration": LN([2.5, 0.3]), "cost": 3_000.0})
  value = plan.expected_npv(4 * 8760.0, discount_rate=np.log(1.08) / 8760.0,
                            periods=4)
  value.undiscounted_cost, value.cost   # 91,650.6 and 77,679.1
  # by year: 19,886, 20,116, 32,597, 19,052
  plan.expected_events(4 * 8760.0)      # pump: 1.586 failures, 2.990 replacements
  plan.mission_availability(4 * 8760.0) # 0.99588
  ```

  Block replacement every 8,000 h costs 78,144.8 at the same rate.
- **Checks:** `test_replace_at.py`: equally spaced times against block
  replacement on that period (to 1e-10); an exponential unit against its
  two-state chain between replacements that take a fixed time, and a
  Weibull unit that fails once at most, in closed form; replacements
  discounted at their times; uneven stretches against grids four times
  as fine; the simulation, from new and from a state; the refusals.
- **Learned / open questions:**
  - A plan ends, so there is no long run to ask for: the long-run
    methods refuse and say to ask for a horizon.
  - The stretch after the last time is as long as the horizon needs, on
    the other stretches' step, up to 80,000 steps: a horizon tens of
    times the plan's longest gap past its last time is followed more
    coarsely.
  - A stretch far shorter than the repairs cannot be followed on its
    grid (as a block interval cannot) and refuses.
  - `plan_vs_actual` values the rest from today on the same diagram,
    where a plan's times would be counted again: refused for now; the
    rest is `expected_npv(state=)` on a diagram with the times still to
    come.
  - The compiled engine leaves a plan to the Python loop.

### UC-52: "The pump is worn. Is upgrading it at 2,000 hours for 30,000 worth it, or should we retire it at 6,000 hours and sell it for 8,000?"

- **Status:** built: a component spec's `"upgrade"`, and `retire_at=`
  and `salvage=` on `expected_npv`.
- **Who asks, and why:** whoever holds the capital budget: an upgrade, a
  retirement and a resale are what a net present value is for.
- **What they know:** a pump failing every 500 h on average, repaired in
  50 at 2,000 a repair, earning 40 an hour while it works; an upgrade at
  2,000 h, for 30,000, to a pump failing every 2,000 h; or its sale for
  8,000 at 6,000 h; 8% a year, over 8,000 h.
- **What they want back:** each plan's net present value, with the
  capital and the salvage in the cash flows.
- **How it is solved:** an upgrade is a replacement at its time by a unit
  of another life (UC-51's stretches, the stretches from the upgrade on
  with the new life), the capital cost an event at that time. A
  retirement cuts the valuation at its time. A salvage is credited at
  the end and discounted from there; by age, its expected value over the
  age of the unit in service (never failed, or put in at the end of a
  repair and up since), for a plain component from new.
- **Today (by hand, before it was built):** the two-state chains either
  side of the upgrade, integrated against `exp(-r t)` in closed form
  (the test's `worth`): 252,977.9 kept as it is, 253,936.5 upgraded,
  199,015.8 retired at 6,000 h with the sale. As built:

  ```python
  kept = pump().expected_npv(8000.0, discount_rate=r, value_rate=40.0)
  upgraded = pump(upgrade={"at": 2000.0, "reliability": E([1 / 2000.0]),
                           "cost": 30000.0}).expected_npv(
      8000.0, discount_rate=r, value_rate=40.0)
  retired = pump().expected_npv(8000.0, discount_rate=r, value_rate=40.0,
                                retire_at=6000.0, salvage=8000.0)
  kept.npv, upgraded.npv, retired.npv  # 252,977.9, 253,936.5, 199,015.8
  ```
- **Checks:** `test_upgrade_retire_salvage.py`: an upgrade to the same
  life at no cost against a replacement at that time (to 1e-10);
  memoryless units either side of an upgrade, and a Weibull unit that
  fails once at most, in closed form; the capital cost at its time;
  retirement at the horizon (nothing changes) and before it (the shorter
  valuation); a constant salvage, `value * exp(-r T)`; a salvage by age
  of a memoryless unit against its age's closed form.
- **Learned / open questions:**
  - Simulated too (`availability`, `cost`, `npv_distribution`,
    `compare_npv`, in Python): a life that starts at or after the
    upgrade's time is the upgrade's, turned from the same uniform of the
    component's `FAILURE` stream as the old life would have been
    (`_event_loop._UpgradedComponent`), so an upgrade to the same life is
    a replacement at its time draw for draw, and `compare_npv` against
    keeping the old pump matches the runs. The capital cost is charged
    at its time in every run that reaches it. For the pump above,
    `compare_npv(kept, 8000, mc_samples=10000, seed=1)` gives the
    upgrade 864 ± 107 more (958.6 exact), better in 51% of the runs; its
    own runs' NPV has p10 240,386 and p90 266,604.
  - An upgraded component takes no maintenance or tests of its own yet,
    and the long run, the spares counts and `plan_vs_actual` refuse it.
  - A salvage by age is a plain component's from new; any other takes a
    number.
  - A retirement is the system's; a component's is not said yet.

### UC-60: "The pump is block-replaced every 4,000 hours. In year 3 we fit a better model and keep replacing it on the same interval. What is that worth over eight years?"

- **Status:** built: a component spec's `"upgrade"` with a
  `"preventive"` of `"policy": "block"` on its own interval.
- **Who asks, and why:** the capital planner whose upgrade does not end
  the maintenance plan: the new model is serviced as the old one was.
- **What they know:** a pump (Weibull 6,000 h and 2.0; repairs
  exponential, 24 h mean, 5,000 each), replaced every 4,000 h in 12 h
  for 2,500; 400 an hour down; an upgrade at 26,280 h (year 3) to a
  Weibull of 12,000 h and 2.0, for 40,000; 8% a year over 8 years.
- **What they want back:** the plan's cost and value with and without
  the upgrade, and how sure the difference is.
- **How it is solved:** the upgrade is one more replacement of the
  block policy, at its time, at the policy's cost and in its time, beside
  the capital cost; the units put into service from it have the new
  life. Over a horizon its block times and the upgrade are a plan's
  times (UC-51's stretches, the lives switching at the upgrade); the
  simulation draws each life from the life in service when it starts
  (UC-52). An upgrade at a block time is that block replacement. The
  calendar counts from now (a `since_scheduled` is refused), and there
  is no long run (refused, naming the values over a horizon).
- **Today (by hand, before it was built):** an upgrade refused any task
  of its own; a plan of `replace_at` times with a second diagram for the
  new life, joined by hand. A memoryless unit's chain between
  replacements, its rate changing at the upgrade, is the test's
  reference. As built: kept, 220,736.8 undiscounted and 163,830.7
  discounted, availability 0.99482; upgraded, 223,084.5 and 171,592.5,
  0.99571. `compare_npv(kept, mc_samples=4000, seed=1)` gives the
  upgrade -7,251 ± 308 (-7,761.8 exact): it does not pay over 8 years.
- **Checks:** `test_upgrade_maintenance.py`: an upgrade to the same life
  at a block time is block replacement (curve and events to 1e-10; its
  NPV equals the same plan of block times to 1e-10, and block
  replacement's to 2.5e-10, integrated on other pieces); between block
  times, one more replacement (a plan at the merged times, to 1e-10); a
  memoryless unit against its chain; the simulation from new and from
  an age (20,000 runs, four standard errors); the refusals. The
  catalogue's "upgraded at a time, block-replaced".
- **Learned / open questions:** age replacement and tests on an upgraded
  part (built: UC-72), its long run after the upgrade, its spares and
  `plan_vs_actual`,
  and retiring a single part (`retire_at={node: t}`) are left for a
  later use case.

### UC-66: "Our pump's seals are a redundant pair, and the cartridge is replaced whole once both have failed. What is its plan worth?"

**Who asks.** The planner of UC-65, whose unit is a subassembly: a
seal cartridge with two seals in parallel, each with its own fitted
life, replaced as one part when it fails.

**What they know.** Each seal's life (exponential, 0.01 a day); the
cartridge's repair (exponential, mean 2 days) and prices.

**What they want back.** The plan's values with the cartridge's life
being the diagram of its seals, worked out exactly where the diagram's
functions allow and simulated otherwise, as for any life.

**How it is solved.** A `NonRepairableRBD` whose reliability changes
with time is taken by its functions, as any surpyval model with a
life's functions is (`_model_utils.by_functions`, `is_diagram_life`):
its `sf`, `ff`, `df`, `hf`, `Hf` and `mean` are the diagram's own, and
`qf` (new) inverts `ff` for a share up to a half and `sf` above, on
logarithms, by false position kept to a bracket, so a draw is seeded
from the streams as a model's is. So `UnitPlan(life=diagram)`, a spec's
`"reliability"` and `NonRepairable(diagram)` take it, saving takes it,
and every route treats it as a smooth life.

**The hand-worked answer.** The pair's life is `(1 - exp(-0.01 t))^2`
failed, a mean of `1/0.01 + 1/0.02 = 150` days, so repaired in a mean
of 2 days the cartridge is up `150 / 152 = 0.98684` of the time, which
`UnitPlan(pair, repair).diagram().mean_availability()` gives.

**Checks.** `test_diagram_life.py`: the parallel pair's `qf` against
its closed form, and a series pair of Weibulls of shape 2 against the
one Weibull it is, each to 1e-13 over shares from 1e-14 to 1 - 1e-9; a
unit whose life is the series pair against the unit of that one
Weibull, exact (long run, over time, NPV) and simulated (seeded
availability and NPV runs), to 1e-12; the parallel pair's availability
against `150 / 152`; a diagram taken by `NonRepairable` and a spec,
saved and loaded. A catalogue kind holds a diagram as a life.

**What was learned.** Nothing but the quantile was missing: a diagram
already had a life's functions. A nested `RepairableRBD`'s costs
counted in its parent are left for later.

### UC-64: "Once the pump is upgraded, what does it cost per year?"

- **Status:** built: the long-run methods of a diagram with an
  `"upgrade"` (`mean_availability`, `expected_cost_rate`, the importance
  measures by default, `replacement_cycle`, ...) give the long run once
  upgraded.
- **Who asks, what they know, what they want back:** the planner of
  UC-60, who has the upgrade over a horizon and wants the steady state
  after it: the availability and cost rate the plant settles to with
  the new model, to set a budget per year.
- **How it is solved:** after its upgrade a component has the new life
  for ever, so the long run is that of the diagram built with the new
  life (its block replacement kept), the time before the upgrade and
  its capital cost ignored. The long-run methods are asked of that
  diagram, which the diagram builds once. Asked over time (`x`,
  `window` or `state`), the measures follow the upgrade as before.
  `total_cost(horizon)`, the long-run rate times a horizon from 0,
  stays refused: the horizon starts before the upgrade.
- **Today (by hand, before it was built):** the long run was refused;
  the second diagram, built by hand with the new life, gave it.
- **Checks:** `test_upgrade_long_run.py`: each long-run method equals
  the diagram built with the new life (to 1e-12), with and without
  block replacement; a memoryless unit's `mu / (lambda' + mu)`; the
  routes say so; values over time still follow the upgrade.
- **Learned / open questions:** age replacement and tests on an
  upgraded part, its spares counts and `plan_vs_actual` across the
  upgrade, and retiring a single part are still open.

### UC-68: "I want to be 95% sure I don't lose more than x on downtime this year. What is the least I need to pay to make that happen?"

**Who asks.** An owner whose contracts, budgets and insurance are set on
a bad year, not the average one. Every optimiser so far takes expected
values; a bad year matters as much.

**What they know.** The system's diagram (or one unit's plan), its
repairs and tasks; the levers they could pull, each at its own price:
copies of a unit in parallel (redundancy), and a unit's maintenance
interval; a limit `x` on the year's downtime, in hours or in cost; how
sure (95%).

**What they want back.** The cheapest design whose bad year (95th
percentile) stays under `x`; the frontier of spend against the bad
year's downtime, so they see what each extra dollar buys; both the
planned and the unplanned downtime of every design (the mean and the bad
year's, in hours and cost), so the trade-off shows; and, when the levers
cannot reach `x`, which floor stopped them.

**Decisions (the owner's, 21 Nov 2026).**
- The target is **downtime only**: the system's hours down, or their cost
  (the diagram's `downtime_cost_rate` and any `value_rate` lost). Not
  total cost: spending to cut downtime raises the maintenance part.
- Spend is the decision and has no bound unless `budget=` is given.
- By default all system downtime counts; `downtime="unplanned"` counts
  failures only (the split of UC-25: down unplanned when it would still
  be down with every planned task finished).
- The risk measure is the downtime's quantile at `level` (0.95), or the
  mean of the worst share beyond it (`objective="cvar"`). The randomness
  is that of failures and repairs, the models taken as known (their
uncertainty is an option, `uncertainty=`, added after).
- "Sure" despite simulation noise: a design meets `x` only when an upper
  confidence bound on its percentile is under it (an order statistic,
  the binomial method, exact; the tail mean's bound is normal, and says
  so).

**How it is solved.** The risk measures sit in the money module
(`repyability.value._risk`), on priced outcomes. Each design is run by
`availability` on one seed (common random numbers), and each run's
downtime is `horizon - uptime`, its unplanned part the engines' own
(kept per run), the rest planned. The search is the allocations'
marginal analysis: from the diagram as given, each step takes the move
that lowers the bad year's downtime most per extra spend: a copy of a
node (drawn as `allocate_redundancy` draws a train of one node), or a
node's interval moved to a later candidate (any later one, as a
quantile can stay put over one step); its spend is the copies' prices
plus the change in the expected preventive cost. It stops at the first
design that meets `x`, or when no move lowers the downtime. The
frontier is the designs evaluated that no other beats on both spend and
downtime. Floors: the budget, the levers' caps, or planned task downtime
at or over the limit alone (shorter intervals add to it, and copies with
the same tasks do not remove it). A plan alone takes its one-node
diagram's method (`UnitPlan.least_spend_for_downtime`).

**The hand-worked answer.** A pump with exponential lives (mean 1,000 h)
and repairs (mean 50 h), up at the start, over 8,760 h: its downtime is
an alternating renewal process's occupation time, whose distribution is
exact by uniformisation (given `n` jumps at rate `L = max(λ, μ)`, the
`n + 1` spacings are Dirichlet(1, ..., 1), so `k` spent down sum to
`T × Beta(k, n + 1 - k)`). Its 95th percentile is **766.35 h** (median
393.08 h; no downtime at all with chance `exp(-8.76) = 0.000157`). 4,000
runs give 761.7 h (bound 783.3). A second pump at 5,000 brings the bad
year under 100 h (80.6 h, bound 90.2, from 1,000 runs); limits of 200,
50, 10 and 1 h are each met by copies, at a spend that rises as the
limit falls.

The trade-off: a wearing unit (Weibull, scale 4,000 h, shape 3; repairs
of mean 72 h) with an 8 h task every 8,000 h. To be 95% sure of under 10
h of *unplanned* downtime the interval must come down to 500 h (1,000
runs): the bad year's unplanned downtime falls from 347 h to 0, and
planned downtime rises from none to 136 h. With all downtime counted,
10 h cannot be reached by intervals: the planned task downtime is the
floor, and the answer says so.

**Checks.** `test_downtime_risk.py`: the single pump's percentile against
the exact distribution (within four of the quantile's standard errors,
seed fixed beforehand), and the distribution's ends; the quantile bound
against the binomial definition; level 0.5 on a symmetric loss is its
median; redundancy meets every limit at a rising spend; the frontier
never rises with spend; along the unplanned frontier planned downtime
rises; the planned floor and the budget floor named; seeded repeats and
cost as hours times the rate; a plan alone equal to its diagram; the
refusals. The bound's coverage at its stated rate is
`scripts/downtime_bound_coverage.py`, a study with fixed seeds: run on
25 November 2026, the 95% bound covered the exact 95th percentile
(766.35 h) in 970 of 1,000 repeats of 500 runs, 0.970 (99% interval
0.953 to 0.982), so it passes, a little conservative, as an order
statistic's bound is.

**What was learned.** The engines already split each run's downtime into
planned and unplanned, and then summed it; keeping it per run was all
the risk needed. Spares, crews and call-out, upgrades and parameter
uncertainty are levers still to add (filed).

**Crews and call-out (added after).** Who asks: the same owner, whose
downtime is jobs waiting for a crew. What they know: the crews they have
(`repair_crews`), what a crew costs (its salary at the `crew_cost_rate`,
and a price besides), and call-out contracts on offer (a fee per call,
a rate while engaged). Solved in the same search: `crews={"price": p,
"max": m}` adds a crew a step, at `p` and its salary; `call_out=[...]`
tries the contracts in order, each priced by its expected fees and time
engaged from its own runs. A design's spend now counts the change in
its expected inspection and crew costs, as well as its preventive cost.
Hand-worked answer: two pumps in parallel, each failing every 50 h on
average and repaired in 50 h, one crew. Both are down 0.4 of the time
(the chain on the pumps down has shares 1 : 2 : 2): 3,504 h a year. A
second crew, or a call-out at the first job waiting, makes the pumps
independent: both down a quarter of the time, 2,190 h; a third crew does
nothing more, and the search stops there. A group of units (standby,
load-sharing, identical) is one node to the levers: a copy is a second
group in parallel, and its units' repairs are the crews' jobs. Checks:
`test_downtime_risk.py`, the crew and call-out designs against the
diagram built with them on the same seed, their mean downtime against
the chain, the spend against the runs' crew cost, each kind of group,
and the refusals.

**Test intervals (added after).** For a unit whose failures hide until
a test finds them, `inspections={node: [intervals]}` moves its test
interval through the candidates, each at its tests' expected cost.
Hand-worked answer: a failure every 1,000 h on average, 20 h repairs,
a test at 50 every 2,000 h: 4,784 h down in a year (the exact value
over the horizon); every 1,000, 500, 250 and 100 h, 3,210, 1,925,
1,127 and 573 h, at 200, 650, 1,550 and about 4,150 more a year. The
simulated designs agree within four standard errors. A bad year stays
over 300 h even at 100 h: the levers' cap is named.

**Upgrades (added after).** "A better pump costs 3,000 more: is it the
cheapest way to stay under 400 h?" `upgrades={node: {"reliability":
better_life, "cost": price}}` (or a list of offers, tried in order)
gives the node's units the better life from the start, at the price
for each unit: a copy of an upgraded node is an upgraded unit too, and
pays it. Hand-worked answer: the pump of the first answer, its failures
every 4,000 h instead of 1,000, has its bad year's downtime exact by the
same uniformisation; 4,000 runs agree within four standard errors. An
upgrade at a later time (the spec's own `"upgrade"`) is not a lever
yet: a node that has one is refused. A load-sharing group's better life
must take the load, as its own does; one that cannot is refused before
the search.

**Parameter uncertainty (added after).** "The pump's life is fitted to
30 failures: how sure is 95% sure?" `uncertainty=True` (or the inputs
as `expected_npv_uncertainty` takes them) runs each design over
`n_draws` draws of the fitted models, the same draws and seeds for
every design (common random numbers still), the `mc_samples` runs
shared among the draws and pooled: the bad year is that of the
predictive distribution, and the result's `uncertainty` says what was
drawn, what was held as given (an upgrade's life among them), and that
the bound is approximate, as runs sharing a draw are not independent.
Hand-worked answer: the pump of the first answer, its rate fitted to 30
failures (1.2e-3 an hour); over 50 draws its downtime is the mixture of
each drawn rate's exact distribution, whose 95th percentile is 914.9 h;
4,000 runs give 917.2 h, against 846 h with the fitted rate taken as
known.

**Spares on the shelf (added after).** "Spares take a month to come:
how many do we hold to be 95% sure a bad year loses under 300 h, and how
many to be 90% sure no repair waits for one?" Who asks: the same owner,
and the stores planner. What they know: each part's lead time, what a
spare costs to hold, the spares held now. The analytic `spares_stock`
gave the long run with the demand taken as going on while a repair waits
(since corrected: a unit waiting for a spare makes no demand, a finite
source, exact for a constant rate replaced in no time); neither the stock's effect on downtime nor the chance of getting through
a horizon with no stock-out was answered. Solved with a shelf in the
Python event loop (`rbd/_shelves.py`): a stock `S`, each spare used
reordered at once and arriving `L` later, first in first out, so the
`k`-th demand's spare is ready at `max(t_k, t_{k-S} + L)` and its repair
starts then. A replacement takes a spare (as `spares_demand` counts
them: a failure repaired as new, a group's unit failures); a failure
repaired imperfectly takes one only if the shelf says so (`repairs`).
The draws are the run's without a shelf, so an unused stock changes
nothing. The compiled engine refuses a shelf (`_compiled.unsupported`),
as it must not differ. `least_spend_for_downtime(spares={node:
{"price": p, "lead_time": L, "max": m, "stock": s}})` adds spares a step
at `p` each, every design run with the shelf (the node's copies draw on
it); `spares_stock(method="simulate", horizon=, no_stockout=)` runs
stocks 0, 1, ... on one seed until the chance of no stock-out over the
horizon has its lower Clopper-Pearson bound over the target.
Hand-worked answers: (1) no stock and a lead time of 100 h on a unit
failing every 1,000 h with 10 h repairs is the unit with repairs of
`100 + Exp(0.1)`, exact over 10,000 h at 0.901396; 4,000 runs give
0.901192 (-0.4 standard errors). (2) With nothing ordered arriving in
the window, a bad window's downtime is under `Y` while the `(S + 1)`-th
failure is after `T - Y`: `P(Poisson(λ(T - Y)) <= S)`, 0.815 at 4 and
0.916 at 5 for a mean of 3, and at 90% the lever finds 5; the chance of
no stock-out is `P(N(T) <= S)`, which the simulated stock matches at
every stock. (3) One spare, 300 h lead time, a unit failing every
1,000 h repaired in no time: the long-run fill rate is `exp(-0.3) =
0.7408`, but a repair that waits keeps the unit down, so fewer demands
find the shelf empty: the chain of the waits, `w' = max(0, L - w - U)`,
gives 0.76923, and the simulation 0.76926. At three spares, where
stock-outs are rare, the long-run fill rate and stock-out probability
(0.99820 and 0.00229) and the simulation (0.99836, 0.00228) agree within
two standard errors. Checks: `test_spares_shelf.py`, these, a stock no
run uses up equal to no shelf bit for bit on each kind of node (plain,
imperfect repair, standby and identical groups, common cause, beside a
sub-system), the demands against `spares_demand`'s count, the bound's
coverage (under the true chance in 184 of 200 seeded repeats at 90%),
the refusals (repair crews, preventive tasks, tests, a sub-system's
components, a fleet). Still to add: a shelf with crews, tasks and tests,
and for a sub-system's components.

**The shelf with crews, tasks and tests (added after).** "Our pumps have
a 1,000 h task, a crew of one and spares that take 300 h: is the stock
still enough?" Rules, each the simplest that keeps the unit's own
process as it was: (a) with repair crews, a repair waiting for its spare
falls due as a crew's job only once the spare is in (an `Arrival` taken
among the failures at its instant), so no crew is held while a spare is
on its way; (b) a task that replaces the unit takes a spare, and with
the shelf empty is put off until it is in, the unit running on as it is:
should it fail first (its life was drawn when it was put in, so the loop
knows when), the failure takes the task's spare and waits for it, and
the task, its spare used, is not done; (c) a test that finds a hidden
failure takes a spare, the repair starting once the test is over and
the spare is in. Hand-worked answers: (1) age replacement at 500 h in
no time, failures at 1/1,000 h replaced in no time, no stock and 300 h
lead time: from each renewal, a failure before 500 h is down 300 h in a
cycle of `X + 300`, and otherwise the cycle is 800 h long, down
`(800 - X)+`: the long-run availability is 0.79408, and 400 runs of
200,000 h give 0.79536 (+1.8 standard errors); taken down to wait, it
would be 0.567. (2) Tests every 200 h in no time, the same lead: a cycle
is `200 / (1 - exp(-0.2)) + 200` long, up 1,000 h of it, 0.76726
against 0.76832 simulated (+1.4 standard errors). (3) A unit `p`
repaired in no time but waiting 200 h for its spare, and a unit `q`
with 20 h repairs, one crew: `q` never waits for `p`, so its uptime is
the run with no crews' (to 1e-12), and `p` is down `200 + π_q / μ` per
failure, `π_q = 2λ / (2λ + μ)` being `q`'s share under repair when the
spare comes. Checks: `test_spares_shelf.py`. Still refused, with the
reason: replacement on condition, maintenance groups, early renewals at
shutdowns, and with crews, tested units and units a shutdown pauses.

**The shelf in the diagram (added after).** "Our pump holds two spares
with a 12-week lead time: what is its availability and cost over the
year, and its timelines?" Who asks: the owner who has a shelf, not one
to size. What they know: the stock held and the lead time. Before, the
shelf lived only in `spares_stock(method="simulate")` and the `spares`
lever, so every other simulation took a spare as always at hand. Solved
as a spec key, `"spares": {"stock", "lead_time", "repairs"}`, held as
the diagram's shelves (each node's own, a sub-system's component's in
the sub-system's loop) and drawn on by every simulation through the
same loop; the routes that try their own shelves put theirs over the
spec's. The risk is an exact method that ignores the shelf: one check,
`_requirements._require_no_shelf`, sits in the checks of a node's
models that the exact routes make, in each unit's curve over time, and
(for every node at once) in the long-run grid and the crews' and
common-cause groups' chains, so all refuse but those of the given
probabilities or the structure alone. Building it showed the checks of
the models were not enough alone: a standby group's or a tested unit's
values over time, a block-replaced unit's long run and the chains do
not make them, and refused nothing until the check was put there too;
a test now runs every method on each kind of node a shelf takes. What the shelf is not
simulated for is refused when the diagram is built, with what to do; a
unit a shutdown pauses with crews is refused by the shutdowns' own
check first. Hand-worked answer: the spec's shelf is the shelf given to
the loop, bit for bit, on every kind of node it takes, so (1) of the
first shelf answer holds for it (0.901396 exact). Checks:
`test_spares_shelf.py`, and the catalogue's two diagrams with a shelf
(`test_analysis_routes.py`, `test_catalogue.py`). Still not simulated:
a shelf across levels, and the compiled engine's shelf.

### UC-61: "Our plan has two outages a year that cost 300 each, and at each we renew the pump if it is past 500 hours. The expected NPV is fine; how bad could a run be?"

- **Status:** built: `npv_distribution`, `compare_npv` and
  `best_switch_time` take a diagram with `shutdowns`.
- **Who asks, and why:** the planner of UC-46 whose plan has planned
  outages (UC-23, UC-24) or renews parts at them (UC-56): the spread of
  the NPV was refused, though `cost()` gave the undiscounted spread and
  `expected_npv` the mean where it is exact.
- **What they know:** the diagram, its shutdowns (times or a period, a
  duration, a scope, a cost, opportunistic renewals), the value rate,
  the horizon and the rate.
- **What they want back:** UC-46's answer: each run's NPV, its
  percentiles, the chance below a hurdle, the CVaR; two plans compared
  run for run.
- **How it is solved (simulated):** the runs are priced as UC-46's,
  with three things the histories alone got wrong. A shut's cost is
  deterministic: charged at each of its starts in the window, in every
  run, discounted exactly. A component a shut holds off line shows a
  planned change down in its history, which is no task: the Python loop
  (the only engine that holds anything; numba's runs shuts in no time
  that cost nothing) marks each hold, and it is not charged. And the
  changes a hold hides (a failure while idle in the shut, a renewal made
  in it: a task at the shut that comes after the hold, an opportunistic
  renewal) are recorded beside the histories, each with its time and
  whether planned, and charged as a recorded change is. The public
  timelines are unchanged: what the system and a component's downtime
  saw is still the held view.
- **Today (by hand, before it was built):** refused, naming
  `expected_npv` and `cost()`. Pricing the histories as they were, the
  outage diagram of `test_shutdowns.py` (800 h, 300 runs) charged 393 a run of
  preventive work for 128 done (each hold taken as a task) and missed
  failures in the shuts; with the holds and hidden changes, each run's
  undiscounted cost is `cost().samples`' to 1e-10. Building it showed
  that a shut in no time with a cost was not charged by the simulation at
  all (0 against `expected_cost`'s 45 over 10 shuts); it is now.
- **Checks:** `test_npv_shutdowns.py`: at no discounting, each run
  against the simulation's own cost and uptime, and each category
  against `cost()`, for outages at given times (one standing a part
  still), opportunistic renewals in long shuts, a shut in no time and a
  part with its own tasks held by shuts that do no work on it; that
  those cases have holds and hidden changes to price; the mean against
  `expected_npv` (4,000 runs, a 99.99% interval) and the shuts' cost
  against its calendar sum to 1e-12; identical plans differ by exactly
  0; the shut in no time charged by `cost()`.
- **Learned / open questions:** the hidden changes could be the
  component's own history (its process under the hold) recorded in
  full; only the charged changes are needed, so only those are kept.
  `plan_vs_actual` still refuses shutdowns: a log given by hand would
  need its holds marked.

### UC-72: "The pump is age-replaced at 4,000 hours, and the trip is tested monthly. In year 3 each gets a better model. Do we swap the unit in service then, or fit the new one when the old comes out?"

- **Status:** built: a component spec's `"upgrade"` with a
  `"preventive"` of `"policy": "age"`, or with an `"inspection"`; and the
  upgrade's `"replace_in_service"` (True by default; False keeps the unit
  in service).
- **Who asks, and why:** the capital planner whose part is maintained by
  its age or found failed by tests, not on a calendar (UC-60's): the
  upgrade has to fit the plan the part is already on, and the new part
  may be bought now and fitted when the old one comes out.
- **What they know:** a pump (Weibull 6,000 h and 2.0; repairs
  exponential, 24 h mean, 5,000 each), replaced at 4,000 h of age in 12 h
  for 2,500; 400 an hour down; an upgrade at 26,280 h (year 3) to a
  Weibull of 12,000 h and 2.0, for 40,000; 8% a year over 8 years. A
  trip tested every month, found failed only by its tests.
- **What they want back:** the plan's availability, cost and value with
  the upgrade, swapped at its time or fitted when the old unit comes out,
  and how sure the difference is.
- **How it is solved:** by default the unit in service at the upgrade
  is replaced then. Under age replacement the unit up then is replaced
  as at its age (at the task's cost and in its time, beside the capital
  cost; a replacement due at that time is the same one), and every unit
  put in from then has the new life, its age counted from when it was
  put in; a unit in a repair or a replacement then is back with the new
  life once it is over. Over a horizon, the old life's renewal equation
  is solved up to the upgrade (the diagram before its upgrades), and from
  there the new life's from the states the unit may be in then: up, and
  so replaced; in a repair, with what is left of it; or in a replacement
  (`_curves._upgraded_curve`). What is left of the repairs and
  replacements going on at the upgrade comes from the failures and
  replacements before it, at their rate on cells back from it and at
  their exact times (`_curves._ongoing`). With tests, the unit in
  service, working or failed unseen, is replaced in no time (a test due
  then is not done), and the new unit is tested on the same calendar:
  the old life's tests' model up to the upgrade and a new unit's from
  it (tests and repairs in no time, finding every failure; tests that
  take time, or miss, are simulated). With `"replace_in_service": False`
  the unit stays until its next failure or due replacement (its age,
  its block time, or a test that finds it failed), and the unit put in
  then has the new life: simulated, the values over a horizon refused
  with the simulation to run. The simulation draws each life from the
  life in service when it starts (UC-52); with tests, the unit put in
  before the upgrade is replaced at the upgrade or, off line for a test
  then, once the test is over (`_event_loop._swapped_at`). The long run
  is the diagram once upgraded (UC-64); the calendar of an age
  replacement counts from now (a `since_scheduled` is refused), a
  tested part's calendar is its own.
- **Today (by hand, before it was built):** an upgrade refused age
  replacement and tests, so the analyst ran two diagrams, one each side
  of the upgrade, and joined them by hand, guessing the state at the
  join. As built, the pump: 0.99613 available over eight years for
  204,836.6 (age replacement alone: 0.99543 for 194,980.0); the
  simulation (4,000 runs) 0.996132 ± 0.000013 and 204,898 ± 428.
  Fitting the new pump when the old one comes out instead,
  `compare_npv(kept, mc_samples=2000, seed=1)` gives swapping -1,258
  ± 114 (at 8%): fitting it when the old one comes out is worth some
  1,260 more.
- **Checks:** `test_upgrade_age_tests.py`: a memoryless unit upgraded
  to its own life and replaced in no time is age replacement (to 1e-6),
  its replacements those before the upgrade, the one at it and from
  there a unit's new or in a repair (to 1e-6 relative); a unit kept in
  service upgraded to its own life is the component without its upgrade,
  run for run (age replacement, tests, no task); a memoryless unit
  against its two-state chain, its rate changing at the upgrade (1e-6),
  and tested against its closed form (1e-9); the simulation from new and
  from states (up at an age, in a repair, in a replacement, on its tests'
  calendar), 20,000 runs within four standard errors: age-replaced
  0.96988 exact against 0.96979 ± 0.00015, 109.98 against 109.97 ± 0.03;
  tested 0.96449 against 0.96431 ± 0.00019; a unit kept in service fails
  as its own life says (lives of exactly 300 and, from the upgrade at
  400, 1,000: two failures kept, one swapped); the refusals name the
  simulation. The catalogue's "upgraded at a time, age-replaced",
  "upgraded at a time, tested" and "upgraded at a time, the unit in
  service kept".
- **Learned / open questions:**
  - The default is Derryn's: the unit in service at the upgrade is
    replaced then; the option keeps it.
  - A unit kept in service is not worked out over a horizon yet: after
    the upgrade it is the old unit from its age then, which a renewal
    equation started from the distribution of its age would follow.
  - Tests that take time, or miss failures, on an upgraded part are
    simulated: `_hidden_tests` could be followed on each side of the
    upgrade, as tests in no time are.

### UC-53: "The trip is tested at 6 and 12 months and then at 2 years, not on one interval. Over the next 3 years, what will each test find, how long is the trip failed, and what does the plan cost against never testing it?"

- **Status:** built: `failure_finding_window(device, test_times, window,
  age=, test_cost=, repair_cost=, demand_rate=, consequence_cost=,
  discount_rate=, demands_reveal=True)` returning a
  `FailureFindingWindow`.
- **Who asks:** UC-18's engineer, whose tests are at times of their own
  (dates laid on the device's clock, UC-54 of the schedule README), from
  a device that is not new, over a budget's window, and who wants the
  side "never tested" to set against it.
- **What they know:** a trip failing hidden at a constant 0.1 a year,
  working now; tests at 0.5, 1 and 2 years, $200 each; a failure repaired
  as new for $1,500; a demand once in 2 years (0.5 a year), which costs
  $50,000 if it meets the trip failed, and which shows the failure, so
  the trip is repaired then.
- **Today:** `failure_finding(device, tau, ...)` gives the long run of
  one even interval from new, undiscounted, with a failure found only by
  a test: no list of times, no window, no age, no "never tested" (its
  unavailability is 1 in the long run), no discount.
- **The answer, by hand:** between two tests the trip is a two-state
  chain: working, it fails at 0.1; failed, a demand shows it at 0.5. From
  a test (or the start) the chance it is failed after `s` years is `0.1 /
  0.6 (1 - exp(-0.6 s))`. So the tests find it failed with chance
  **0.0432** (after 0.5 years), **0.0432** and **0.0752** (after 1 year):
  0.1616 failures found. The time failed is `0.1 / 0.6 (s - (1 - exp(-0.6
  s)) / 0.6)` over each stretch (0.5, 0.5, 1 and 1 years): 0.01133 +
  0.01133 + 0.04134 + 0.04134 = **0.10535 years** (a mean unavailability
  of 0.0351), so 0.0527 demands meet it failed. Cost: 3 tests $600;
  repairs $1,500 x (0.1616 + 0.0527) = $321.40; consequences $50,000 x
  0.0527 = $2,633.75: **$3,555.15**. *Never tested:* one stretch of 3
  years: failed for 0.26814 years (0.0894), 0.1341 demands on a failed
  trip, $201.10 of repairs and $6,703.46 of consequences: **$6,904.57**.
  The three tests save $3,349.
- **How it is solved:** the device is working at an age since it was
  last made new, or failed unseen. A test of a working device leaves its
  age; one that finds it failed makes it new, in no time. `A(t)`, the
  chance it is working, is the sum over the times it was made new of the
  chance of that and of surviving since: `A(t) = R(age + t) / R(age) +
  sum_k q_k R(t - t_k) + d integral_0^t U(s) R(t - s) ds`, `U = 1 - A`,
  with `q_k = U(t_k)` just before test `k` (what it finds) and the last
  term the devices a demand showed failed. Where only a test finds a
  failure the last term is not there and `A` is exact at any time (a
  table of the tests so far). With it, this is a Volterra equation of
  the second kind, solved by the trapezoid rule on a grid of each stretch
  between tests. The failed time (and its discounted integral) is the
  trapezoid rule's on the same grid either way; the whole is worked out
  on one grid and on one twice as fine, each number taken from the two
  by Richardson's step, and the grid halved until the two agree to
  `tolerance` (1e-6 of the failed time and of the failures found, by
  default). Their difference, in money, is `precision`. A test and the
  repair of what it finds are discounted from the test's time; the
  demands' cost inside the integral.
- **Decisions (defaults):**
  - **A demand that meets the device failed shows the failure**, which is
    repaired as new then, at `repair_cost` (`demands_reveal=True`). This
    is what makes "never tested" a finite number. `demands_reveal=False`
    is `failure_finding`'s convention: only a test finds a failure, and
    every demand until then costs the consequence.
  - Tests and repairs take no time, and a test never misses a failure,
    as `failure_finding` has it.
  - The device is working at the window's start (a test at time 0 only
    costs). `test_times=()` is "never tested".
  - Demands are a Poisson process at `demand_rate`, independent of the
    device.
  - `discount_rate` is per unit of the device's time, continuously
    compounded, as `expected_npv`'s; dates and a yearly rate are the
    schedule module's (UC-54).
  - At most 2,000 tests in a window (the work grows with their square),
    refused naming `failure_finding` for the long run.
- **Checks** (`test_failure_finding_window.py`):
  `test_uc53_trip_tested_at_uneven_times` (the numbers above),
  `test_exponential_closed_form` (the two-state chain's closed form,
  with and without demands revealing, with and without a discount, and
  the curve), `test_even_tests_reproduce_the_long_run` (even tests, no
  discount, only tests finding: `failure_finding`'s unavailability, cost
  rate and failures found, for a constant rate on any whole number of
  intervals and for a Weibull life over 400 mean lives),
  `test_weibull_against_a_simulation` (a wearing device 4 old, tested at
  uneven times and never, with and without demands revealing, within
  four standard errors of a seeded simulation of the plan),
  `test_never_tested_without_reveal_is_the_life_itself`,
  `test_a_test_at_the_start_finds_nothing`,
  `test_the_unit_of_time_does_not_matter`, `test_refusals`.
- **Learned / left:** the precision given is the two grids' difference,
  which is several times the error of the extrapolated numbers (1e-15
  against 4e-7 of the cost for the trip above); a life with an infinite
  density at 0 (a Weibull shape under 1) converges more slowly and takes
  the finest grid. A test that can miss a failure, and tests and repairs
  that take time (which `RepairableRBD`'s inspected blocks have, on a
  period), are not here. The best times for the tests are not searched
  for.

**A unit that fails on demand, in the diagram (added after).** "Our
standby pump fails to start on 2% of demands, and it is called on about
once every two days: how available is the line, and what does its repair
cost a year?" Who asks: the reliability engineer with OREDA's
fail-to-start probability. What they know: the probability per demand,
how often demands come, and the repair time. Before, a `RepairableRBD`
took the probability only in its simulations and refused the exact
methods, as a probability says nothing about time. Solved as a spec key
(and a `UnitPlan`'s `demand_rate`): demands a Poisson process at
`demand_rate`, as `failure_finding` takes them, each failing a working
unit with the probability, so its failures are an exponential life of
rate `demand_rate * p`, built once when the diagram is, which every
route then takes. Hand-worked: one pump repaired in 2 days on average is
available `0.5 / (0.5 + 0.5 * 0.02) = 0.98039`. Checks
(`test_demand_rate.py`): every method against the diagram given that
life, to the bit, seeded runs included; the closed form; the levers,
saving, units and refusals. Left: a unit that wears out and may also
fail on a demand (two components in series for now), and demands at
fixed times.

### UC-73: "The standby generator is started at three planned outages next year, and it fails to start on 10% of starts. How available is it, how many failed starts do we expect, and what do the repairs cost? And the fire pump, called on more in summer than in winter?"

- **Status:** built: a component spec's `"demand_times"` (a list of
  times, such as `CalendarContext.rule_times(rule)` of a recurrence
  rule) or `"demand_plan"` (demands per unit of time from given times),
  on a `"reliability"` that is a `surpyval.FixedEventProbability`, with
  `"demand_basis"` saying whether the plan is on the diagram's clock
  (the default) or in the block's operating time; a `UnitPlan`'s fields
  of the same names.
- **Who asks:** the reliability engineer of the previous paragraph, whose
  demands do not come at random at a steady rate: a standby unit started
  at planned outages or proof runs, on dates the plan fixes, or a pump
  called on by a season (more fires in summer, fewer starts at
  weekends).
- **What they know:** the probability `p` of failing per demand, the
  dates of the demands (or the expected demands per day through the
  year), and the repair time.
- **Today:** a `"demand_rate"` makes the demands a Poisson process at
  one rate, so a demand plan has to be averaged into a rate, which puts
  the risk at the wrong times: a window between two outages sees
  failures it cannot have.
- **The answer, by hand:** demands at 10, 20 and 30 days, `p = 0.1`, a
  failed start repaired in exactly 15 days. A demand meets the generator
  working unless a repair is under way. The first fails it with chance
  `f1 = 0.1`; at day 20 the repair from day 10 (to day 25) is under way
  if there was one, so `f2 = 0.1 (1 - 0.1) = 0.09`; at day 30 that one
  is over and only day 20's (to day 35) can be under way, so `f3 = 0.1
  (1 - 0.09) = 0.091`. **0.281** failed starts; over days 0 to 50 the
  generator is down for `15 x 0.281 = 4.215` days, available **0.9157**;
  at day 22 it is down if day 10's or day 20's start failed, available
  `1 - 0.19 = 0.81`.
- **How it is solved:** demands at set times are a life that can end
  only at them, which does not restart with a repair: the plan is on the
  calendar. A unit is up for demand `k` unless one of its earlier
  failures is still being repaired, and at most one is, so the chance
  that demand `k` fails it is `f_k = p (1 - sum_{j<k} f_j P(R >= t_k -
  t_j))`, `R` the repair time, and the point availability `A(t) = 1 -
  sum_{t_k <= t} f_k P(R > t - t_k)`: exact at every time, the expected
  failures `sum f_k` over a window, and a window's integrals on pieces
  broken at the demands and the repair's quantiles. A unit repaired at
  `t0` meets the demands after `t0` (one repaired at a demand's instant
  is not called on by it). The simulation draws, for a unit put into
  service, the number of demands it survives, geometric, from its own
  failure stream, and takes the demand that many on. A rate that changes
  is the duty plans' machinery: the expected demands `Λ(t)` by time `t`,
  piecewise linear, and the life `Exponential(p)` counted in demands,
  put on the diagram's clock by `Λ⁻¹` (survival `exp(-p Λ(t))` from
  new); a constant rate is `"demand_rate"` itself.
- **The clock:** a demand plan is on the diagram's clock by default, and
  then a `"duty"` is refused (it would stretch the demands as though
  they came only while the block runs). With `"demand_basis":
  "operating"`, the plan is in the block's operating time, and the duty
  puts it on the calendar: the times `t_k` at `D⁻¹(t_k)`, a rate
  `λ(D(t)) D'(t)`, a `"demand_rate"` `d` times the share. This settles
  the refusal of a demand rate with a duty.
- **Checks** (`test_demand_plans.py`): the hand-worked generator to
  1e-12 (point availability, failures and availability over windows),
  and a seeded simulation (its design fixed before it ran) within four
  standard errors; demands equally spaced at `h` with `p = λ h`
  converging on the exponential life of rate `λ` as `h` shrinks; a rate
  that changes against `exp(-p Λ(t))`, simulated with no repair and
  exactly in a repair of no time; seeded runs of a diagram's other nodes
  unchanged by a plan on one; the routes, catalogue kinds, levers,
  saving and refusals.
- **Learned / left:** the long-run analyses are refused (a list of
  times ends, so its long run is that of a unit never called on), as are
  imperfect repair and tasks (the curve has no place for them), which the
  simulations take. A rate that changes is exact only where its repairs
  and tasks take no time, as duty plans are; else simulated. A unit with
  a life and a demand mode in one component is a separate use case.


### UC-25: "How much of our downtime and lost production is planned and how much is from failures, and how many failures does the plan avoid each year against running to failure?"

- **Status:** built: `RepairableRBD.maintenance_effect(t=None,
  state=..., method=None)` returning a `MaintenanceEffect`; the split
  natively in `ExpectedEvents` and `EventsByWindow`
  (`system_planned_downtime`, `system_unplanned_downtime`,
  `node_planned_downtime`, `node_unplanned_downtime`),
  `AvailabilityResult` (`system_planned_downtime`,
  `system_unplanned_downtime`) and `CostResult`
  (`lost_production_planned`, `lost_production_unplanned`).
- **Who asks, and why:** the planner of UC-23, reporting to operations:
  of the hours the line is down, how many are the plan's own (which they
  scheduled and can move) and how many are breakdowns; and whether the
  plan earns its keep, the failures it avoids against what it costs.
- **What they know:** the line of UC-23 (pump, motor and bearing in
  series, shut 72 h every 12 months at 10,000 a shut, every block's PM
  done in it, the bearing changed only once older than 18 months, 500 an
  hour of lost production). Model time only.
- **What they want back:** a year's downtime split into planned and
  unplanned hours, and the lost production likewise; the failures a year
  with the plan and running to failure, and the difference; the cost a
  year either way.
- **How it is solved:** the definition, in plain words: the system is
  down *unplanned* if it would still be down with every planned task,
  test and shut in progress finished; *planned* otherwise. A block in a
  repair, waiting for a crew or failed unseen is failed (unplanned, also
  while a shut holds it); one in maintenance or a test of a working unit
  that takes time, or held by a shut, is in planned work. With
  independent blocks, `P(down unplanned at t) = 1 - phi(P(block not
  failed at t))`, so each exact route that tells a block's failed state
  from its planned work gives the split: the renewal grid of age
  replacement keeps its repairs' part apart (`GridCurve.failed_at`), a
  block-replaced unit's replacement in progress is the share replaced at
  the block time not yet back (`BlockCurve.failed_at`), a tested unit's
  test of a working unit is its rows' `tested` share not yet over
  (`TestedCurve.failed_at`), a shut's mask is planned, the block's own
  process underneath (`MaskedCurve.failed_at`), and a nested RBD's is its
  own structure's (`SystemCurve.failed_at`); the long run over the long
  run's grid likewise. Running to failure is the diagram rebuilt without
  `"preventive"`, shutdowns and maintenance groups (its tests kept: a test
  avoids no failure). The simulation follows each block's failed state
  path by path, the system's unplanned downtime added up as it changes,
  on both engines (`_unplanned`, `_kernel`); a nested RBD running ahead
  of its parent passes up whether it is down unplanned, and each change
  of that while it is down, taken in time order. Crews and common-cause
  groups with planned outages (whose chains do not tell planned work
  from a repair) are simulated.
- **Today (by hand):** the twin with zero-length planned work that some
  callers use: the same line with a shut that takes no time (so its
  tasks too), its unavailability read as the unplanned part:

  ```python
  today = shut_line(12, duration=0.0)
  today.mean_unavailability() * 8760          # 10.29 h a year
  annual.mean_unavailability() * 8760         # 83.92 h a year in all
  ```

  so 73.63 h planned and 10.29 unplanned. It is not the same thing: with
  no shut the blocks age on another timeline (renewed 72 h earlier each
  year), and the failures that start in a shut, or are still being
  repaired when it starts, are not there to count. Failures avoided
  needed a second diagram drawn by hand without the tasks and shut.
- **Built:**

  ```python
  effect = annual.maintenance_effect()        # per hour, the long run
  ```

  | A year | With the plan | Running to failure | The plan's effect |
  |---|---|---|---|
  | Downtime | 83.92 h: 70.55 planned, 13.37 unplanned | 23.52 h | 60.40 h more |
  | Lost production | 41,960: 35,274 planned, 6,687 unplanned | 11,759 | |
  | Failures | 0.4237 | 0.9791 | 0.5554 avoided |
  | Cost | 65,899 | 31,372 | 34,526 more |

  The shut's 72 h a year are 70.55 h planned: the rest is failures that
  start or carry on in it (the blocks keep ageing through it). Over 10
  years from new (`maintenance_effect(87600.0)`): 746.2 h down, 647.4
  planned and 98.7 unplanned; 4.10 failures against 8.73, 4.62 avoided;
  simulated (2,000 runs, seed 1) 647.4 and 98.9 h, 4.70 avoided, and the
  cost saved -312,190 ± 1,440 against the exact -314,499. At a 9-month
  shut 96.3 h planned and 9.7 unplanned a year, at 18 months 48.1 and
  13.1.
- **Checks:** `test_planned_split.py`: one block under age replacement
  (planned `E[PM time] / E[cycle]`, unplanned `E[repair time] /
  E[cycle]`), a parallel pair (unplanned `f^2`, planned `(f + m)^2 -
  f^2`), one unit in maintenance and the other failed (planned), a
  periodic shut of the whole system (unplanned `q`, planned `D / P (1 -
  q)`, ageing or standing still), a timed test of a hidden failure
  (planned `w D / T`); exact against simulated on a nested diagram and on
  the catalogue's (`test_catalogue.py`); both engines to the last bit
  (`test_simulation_engines.py` compares every field); the parts adding up
  to the downtime in every result; failures avoided against two runs
  differenced and two diagrams' exact values; units; saving to JSON.
- **Learned / open questions:**
  - By the definition, a parallel pair with one unit in maintenance and
    the other failed is down *planned* (it would be up with the
    maintenance over), though the change that took it down was a
    failure: `system_failures` and `system_planned_outages` count by that
    change, the split by the state.
  - Here the plan avoids more than half the failures but costs more than
    it saves: the shut's lost production dominates. The split shows where
    to look (moving the shut, or shortening it).
  - Not built: the split through the crews' and common-cause groups'
    chains (they could keep a block in planned work apart from one in a
    repair), and shuts at given times exactly (as UC-24).

### UC-26: "Our pump runs 60% of the time and its overhaul is every 4,000 running hours: what does that do to availability and cost, and what if it ran 80%?"

- **Status:** built: a `"preventive"` or `"inspection"` spec's `"basis":
  "operating"`, with the block's `"duty"`.
- **Who asks, and why:** a reliability engineer whose maintenance plan
  counts the pump's tasks by its hour meter, as the maker's manual does,
  while the plant is planned on the calendar. Production wants to run
  the pump harder (80% of the time): what does that cost?
- **What they know:** the pump's life fitted in running hours (a Weibull
  of scale 5,000 h and shape 2.5), its share of the time running (60%),
  the overhaul every 4,000 running hours (24 h, 8,000; it renews the
  pump), a failure's repair (48 h) and cost (30,000). Model time only.
  (Not UC-10: there the duty follows an operating schedule on dates,
  Reliafy's; here it is one average share.)
- **What they want back:** the availability, the failures and the cost a
  year, the downtime split into planned and unplanned, at 60% and at
  80%; and the best overhaul interval, in running hours.
- **How it is solved:** at construction. A block of duty `d` ages only
  while it runs, so its life on the diagram's clock is `R(d t)` (the
  `"duty"` already did this); a task counted in running hours falls due
  every `N` of them, every `N / d` hours on the clock, so its times
  (`interval`, `opportunity`, `older_than`; a test's `interval`,
  `offset`, `full_test`) are divided by `d`. Every exact route, both
  engines and the planned split then take the converted schedule as
  they take any other: no new code in the analyses. The spec as given
  keeps the running hours, for saving, the levers, `with_intervals` and
  the interval choices (`optimal_replacement_intervals` takes and gives
  the intervals in running hours). Refused: a task in running hours with
  no duty (say what share of the time the block runs, or give the
  interval on the calendar), at a shut (on the calendar), or under a
  `"calendar"` policy (whose thresholds are its own).
- **Today (by hand):** the life and the interval converted by hand,
  `5000 / 0.6` and `4000 / 0.6` calendar hours. The trap is to convert
  the life and not the task (an overhaul every 4,000 calendar hours,
  about 2,400 running hours):

  | A year | By running hours (6,667 h on the calendar) | Every 4,000 calendar hours |
  |---|---|---|
  | Availability | 0.99403 | 0.99285 |
  | Failures | 0.6616 | 0.3354 |
  | Cost | 26,699 | 25,566 |
  | Downtime, planned + unplanned | 20.55 h + 31.76 h | 46.51 h + 16.10 h |

- **Built:**

  ```python
  pump = {
      "reliability": W([5000.0, 2.5]),        # running hours
      "repairability": FIXED([48.0]),
      "replace_cost": 30000.0,
      "duty": 0.6,
      "preventive": {
          "interval": 4000.0, "basis": "operating",
          "duration": FIXED([24.0]), "cost": 8000.0,
      },
  }
  ```

  | A year | 60% of the time | 80% of the time |
  |---|---|---|
  | Availability | 0.99403 | 0.99205 |
  | Failures | 0.6616 | 0.8804 |
  | Cost | 26,699 | 35,529 |
  | Downtime, planned + unplanned | 20.55 h + 31.76 h | 27.35 h + 42.26 h |
  | Best overhaul interval | 2,870 running hours | 2,868 running hours |

  The 60% figures are the hand conversion's to the last bit (the same
  objects). Running harder costs a third more a year, almost all of it
  the pump's own running: per running hour nothing changed. The best
  interval in running hours hardly moves with the duty (only the repair
  and overhaul times, on the calendar, weigh differently).
- **Checks:** `test_duty_tasks.py`: every task in running hours against
  the same block given on the calendar by hand (age, block and condition
  policies, an opportunity, tests with an offset and full tests, a
  restoring task, a shut with the block ageing or standing still), exact
  values and seeded simulations equal, both engines to the last bit; the
  planned split exact against simulated; the levers and the interval
  choices in running hours (a sensitivity per running hour is the
  calendar's over `d`); saving keeps the running hours and the basis;
  the refusals. `test_units.py`: tasks in running hours scale with the
  unit.
- **Learned / open questions:**
  - A shut is on the calendar, so a task bound to one is refused in
    running hours, `"older_than"` with it too. "Replace at the shut if it
    has run more than X hours" would read naturally; it is `older_than`
    over `d` on the calendar, which a caller can give today.
  - A starting state (`NodeState`) stays on the diagram's clock: an age
    and a time since the last task in calendar hours.

### UC-27: "Our pumps that fail often also seem to get scrapped sooner. This one has just failed again: do we repair it, or replace it? And for a new pump, when should we stop repairing?"

- **Status:** built: `RepairOrReplace(model, repair_cost=,
  replacement_cost=, failure_cost=, write_off_cost=, discount_rate=)`,
  with `fleet_policy`, `unit_decision`, `unit_decisions` and
  `break_even_frailty`.
- **Who asks, and why:** a reliability or fleet engineer whose repair
  records show that some units are lemons: they fail more often and are
  written off (beyond economic repair) sooner. Each failure is a choice:
  pay for another repair, or buy a new unit.
- **What they know:** a joint frailty model fitted in surpyval
  (`surpyval.JointFrailty.fit`) to the fleet's failures and write-offs,
  with each unit's frailty (`model.frailties`); the unit's age now and
  whether it has just failed; a repair's cost, a failure's downtime cost,
  a new unit's cost and a write-off's cost; optionally a discount rate.
  Model time only (ages in the model's unit).
- **What they want back:** for a new unit, the cost rate of keeping on
  repairing it until it is written off, of replacing it at the best age,
  and of replacing it at the best count of failures, with the cheapest;
  for one unit, which of these to follow from now (or to replace it now),
  what that saves, and the frailty at which the answer turns. Each cost
  rate with its 90% two-sided band.
- **How it is solved (numerical):** given its frailty `u` a unit's
  failures are minimal repairs at the rate `u r(t)` and its write-off has
  the hazard `u**alpha h(t)`, independent of each other. A rule's cost
  rate is the expected cost of a cycle (new until replaced or written
  off) over its expected length (each discounted at a continuous rate,
  the equivalent cost rate, as in UC-19). The cycle's integrals are over
  age (Gauss-Legendre on pieces even in the logarithm of age) and, for a
  new unit, over the frailty (Gauss-Hermite on its normal score), of the
  model's `iif`, `cif`, `hf` and `Hf` at each frailty; a count of failures
  through the Poisson probabilities of the expected failures. For one
  unit, the frailty is taken as known (its estimate), and each rule is
  valued against replacing now with what follows a replacement priced at
  the fleet's cost rate (the relative value: the rule's cost less the
  fleet's cost rate times its time, less the replacement's cost); a
  failure just had adds its repair to every rule that keeps the unit.
  The bands draw the fitted parameters from their covariance (on the log
  scale for the positive ones) and recompute each rule's cost rate at its
  fitted best age or count.
- **Today:** `Repairable` refuses the model (#404), as its `cif` counts
  repairs as if no unit were ever written off, which overstates them.
- **Checks:** `test_repair_or_replace.py`: with no frailty variance, every
  rule against direct quadrature of minimal repair and an independent
  write-off (to 1e-8), keep repairing against its closed form for
  shapes alike, constant rates against `(cr + cf) / a + cw / b` and no age
  rule paying; a small variance tends to none, a small discount rate to
  none; the frailty nodes against the model's population `sf`; an age rule
  against the model's `mcf`, `ff` and `sf`, and against its `random`; a
  count rule against a simulation written from the model's definition; a
  new unit like the fleet saving exactly the replacement; a unit's
  numbers against quadrature from its age; break-even frailty: units
  above it replaced now, below it kept, and it falls with age; every cost
  rate rising with each cost; the answers scaling with the unit of time;
  covariates as rescaled baselines; seeded bands; the refusals.
- **Learned / open questions:**
  - Counting failures pays for two reasons: a unit with many failures is
    likely a lemon, and replacing at a failure saves that failure's
    repair. The second holds with no frailty variance at all.
  - A unit's frailty is a plug-in (surpyval gives each unit's posterior
    mean, not its distribution); its failures from now would change the
    estimate, which the decision does not foresee. Decide again with a
    refitted model.
  - The gap time scale (each failure renews the rate) is refused:
    surpyval 0.27 has no expected count of failures for it and cannot
    simulate a unit of a given frailty.
  - surpyval's joint frailty model is experimental (its design decisions
    are proposed), with Weibull baselines and no bootstrap: the bands are
    Wald draws.
