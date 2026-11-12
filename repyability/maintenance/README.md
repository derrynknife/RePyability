# Maintenance: use cases

This is the working record of RePyability's maintenance questions (#319),
built **one use case at a time**: each use case is a question someone
actually asks, written down before any code, with how it is answered and
what was learned.

**RePyability is about timelines; Reliafy is about calendars** (the scope
note in `CLAUDE.md`). Every use case here is asked in model time, from a
state: ages, how far into an interval, windows of time from now. Dates,
calendars, operating schedules, meters read by date and financial years
are Reliafy's, which turns them into model time, calls RePyability, and
turns the answers back into dates. Reliafy's record
(`docs/maintenance-scheduling.md`) holds the calendar side of the same
use cases, under the same numbers.

## The use cases, by number

| # | The question here, in model time | State |
|---|---|---|
| UC-1 | A system's expected maintenance work in each of a run of windows from its state | **Built**: `RepairableRBD.expected_events_by_window` (#375) |
| UC-2, UC-3, UC-4 | Tasks on dates and monthly rules: calendar only | Reliafy's |
| UC-5 | A part's chance of failing, expected failures and their cost over a window, from its age | **Built**: `NonRepairable.failures` |
| UC-6 | Right after a failure, the value of skipping the next planned replacement; and, from any state, replace, defer, move to a shut or skip compared | **Built**: `NonRepairable.skip_next_replacement`, and `NonRepairable.compare_replacement_options` |
| UC-7 | The cheapest replacement policy when replacing at a shut is cheaper than between shuts | Built as `replacement_policies`, then removed: UC-13 answers it |
| UC-8 | Each policy's cost over a window from a state, and its spread | **Built** for a part: `NonRepairable.cost_distribution` (run to failure, age and block replacement); a calendar policy simulated by `RepairableRBD.cost_by_window` |
| UC-9 | The cost in each of a run of windows from a state, and how often each outcome happens | **Built** for a part: `NonRepairable.cost_by_window` (run to failure, age and block replacement); a system's simulated by `RepairableRBD.cost_by_window` (#375) |
| UC-10 | Tasks by use on an operating schedule: calendar only | Reliafy's |
| UC-11 | A part renewed at every service and at failure: its risk before each service, from its state, in running hours | **Built**: `NonRepairable.renewal_forecast` |
| UC-14 | The best visit interval and depth ratio for a layered package of visits | **Built**: `optimal_layered_visits` |
| UC-15 | The cost and risk of letting a task slip by a tolerance | **Built**: `NonRepairable.slack_effect` |
| UC-16 | The extra risk and cost of rounding a due replacement up to the next visit | **Built**: `NonRepairable.cost_of_waiting_for_visit` |
| UC-18 | The numbers behind each maintenance decision for a failure mode: run to failure, scheduled replacement, failure finding, on condition | **Built**: `run_to_failure`, `scheduled_replacement`, `failure_finding`, `on_condition` |
| UC-17 | The risk added when a visit is missed and its work waits | **Built**: `NonRepairable.cost_of_missed_visit` (and `delay_effect`) |
| UC-12 | When to replace this part next: now, at a scheduled outage, at a planned replacement, or at its failure? | **Built**: `NonRepairable.optimal_next_replacement` |
| UC-13 | The long-run optimal replacement policy over a calendar of shuts with prices of their own | **Built**: `NonRepairable.optimal_calendar_policy` |
| UC-19 | What a plan is worth over a horizon at a discount rate, and as a yearly cost | **Built**: `RepairableRBD.expected_npv`, `NonRepairable.age_replacement_value` |
| UC-20 | One unit run on an operating schedule (covariates and value by stretch, in model time): the value earned, the failures' cost and the NPV, and the best time to switch from one way of running to another | **Built** (one unit): `operating_schedule_value`, `best_switch_time` |
| UC-21 | A plan against its outage log: expected and actual cash flows so far, the variance, and the plan re-valued from the states now | **Built**: `RepairableRBD.plan_vs_actual` |
| UC-22 | Overhauls that restore a unit only partly (a Kijima restoration factor), every Nth replacing it: availability, failures and cost per year against overhauls that renew it, and the cheapest interval | **Built**: a `"preventive"` spec's `"restoration"`, `"replace_every"` and `"replacement_cost"`, simulated (#398) |

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
- Model time, not dates: Reliafy turns dates into ages and windows.
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
- **How it is solved:** in model time (RePyability is about timelines):
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
  default, `method="simulate"` as the check). UC-7's best policy (at a
  shut if at least 138 days old) is a calendar policy, still simulated
  (`RepairableRBD.cost_by_window` with the policy as the part's
  preventive maintenance). The answer by hand below is the check.
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

### UC-9: "What will the bearing cost in each of a run of consecutive windows from now, and how often does each outcome happen in each?"

(First asked by financial year, July to June, from 11 October 2026; the
windows here, 0 to 263, 263 to 629, 629 to 994 and 994 to 1,359 days
from now, are those years as Reliafy would turn them. Summing by
financial year is Reliafy's.)

- **Status:** built for one part under run to failure, age replacement
  and block replacement: `NonRepairable.cost_by_window(ends, every, *,
  resets, age=, since_renewal=)` and `WindowCosts`, numerical by default
  (`method="simulate"` as the check). A system's windows, and a calendar
  policy such as UC-7's best, are simulated: `RepairableRBD.cost_by_window`
  and `CostByWindow` (#375) give each window's exact mean and its
  simulated spread, outcomes and chance of exceeding a budget from one
  run. The answer by hand below (the best policy, simulated) is kept as
  the record.
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
  Decisions to take or verify: `repyability/maintenance/CALENDAR_DECISIONS.md`.
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
  given times, the simulated distribution, crews as cash flows, and NPV as
  the optimisers' objective.


### UC-20: "If we run at 70 °C instead of 30 °C we make more, but the bearings wear out faster. Does it pay?"

- **Status:** built for one unit: `operating_schedule_value(model, times,
  covariates, value_rates, horizon, failure_cost=, down_time=,
  discount_rate=, age=)` and `best_switch_time(model, covariates,
  value_rates, horizon, ...)`. A whole diagram on an operating schedule is
  not built: a `RepairableRBD` component cannot yet take a covariate-
  dependent life.
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
- **Learned / left:** each unit put in before the last change has its own
  life, so the work grows with the cells before it (one evaluation of the
  model along a path each); `best_switch_time` values each switch time to
  a looser tolerance by default (1e-4) for that reason. Left: a diagram on
  a schedule (covariate-dependent lives in a `RepairableRBD`), a schedule
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
