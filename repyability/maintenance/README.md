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
| UC-1 | A system's expected maintenance work in each of a run of windows from its state | Answered by hand |
| UC-2, UC-3, UC-4 | Tasks on dates and monthly rules: calendar only | Reliafy's |
| UC-5 | A part's chance of failing, expected failures and their cost over a window, from its age | **Built**: `NonRepairable.failures` |
| UC-6 | Right after a failure, the value of skipping the next planned replacement | **Built**: `ReplacementPlan.skip_next`, to be reworked |
| UC-7 | The cheapest replacement policy when replacing at a shut is cheaper than between shuts | Built as `replacement_policies`, then removed: UC-13 answers it |
| UC-8 | Each policy's cost over a window from a state, and its spread | Answered by hand |
| UC-9 | The cost in each of a run of windows from a state, and how often each outcome happens | Answered by hand |
| UC-10 | Tasks by use on an operating schedule: calendar only | Reliafy's |
| UC-11 | A part renewed at every service and at failure: its risk before each service, from its state, in running hours | Answered by hand |
| UC-12 | When to replace this part next: now, at a scheduled outage, at a planned replacement, or at its failure? | **Built**: `NonRepairable.optimal_next_replacement` |
| UC-13 | The long-run optimal replacement policy over a calendar of shuts with prices of their own | **Built**: `NonRepairable.optimal_calendar_policy` |

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

- **Status:** answered by hand with today's RePyability; not built.
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
      "valve": NodeState(age=7000.0, phase=7000.0),
      "sensor": NodeState(age=3000.0, phase=3000.0),
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

- **Status:** built: `ReplacementPlan` and `SkipValue`
  (`repyability/maintenance/_plans.py`).
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
- **What they want back:** the NPV of skipping, as what it saves
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
  - **The discount rate** is a rate a year, `i`; costs are discounted
    continuously at `r = log(1 + i) / 365.25` a day, so a cost a year
    away is worth `1 / (1 + i)`.
- **Today (before this use case):** the fixed-interval plan by hand from three
  `expected_cost` calls and the formula above; the resetting plan not
  at all (`NonRepairable` prices age replacement only in the long run,
  undiscounted).
- **Design, as built:**

  ```python
  from repyability.maintenance import ReplacementPlan

  life = surv.Weibull.from_params([365.0, 2.0])
  resetting = ReplacementPlan(life, every=180, planned_cost=150, unplanned_cost=400, resets=True)
  resetting.skip_next(discount_rate=0.08).npv                    # 26.56

  fixed = ReplacementPlan(life, every=180, planned_cost=150, unplanned_cost=400, resets=False)
  fixed.skip_next(discount_rate=0.08, days_to_next=30).npv    # 123.08
  ```

  `resets` has no default: the two plans answer differently, so it is
  always said. A plan on fixed intervals needs `days_to_next`; a resetting one
  refuses it (its next planned replacement is always `every` away).
- **Checks:** `test_maintenance_plans.py` checks:
  - the answers against plain simulations of both plans (the resetting
    one over 300 years);
  - an exponential life against both plans' closed forms, at 0%, 8% and
    50% a year;
  - the resetting plan with no discounting against the classic
    undiscounted comparison, and a tiny rate next to it;
  - the 180-day plan's NPVs falling with the days to the next one, and
    the 365-day plan's change of sign;
  - the input checks.
- **Learned / open questions:**
  - **The plan itself may be the question.** With these costs a 180-day
    plan is too frequent, so skipping always pays. Comparing the plan
    with the best one (discounted), or the latest day at which skipping
    a fixed-interval replacement still pays, are natural next use cases.
  - **Days, not dates.** "Right after a failure" needs only the days to
    the next planned replacement; dates are Reliafy's.
  - **Replacements take no time.**
  - **The API is being reworked** (the maintainer does not like
    `skip_next`): a comparison of the options at a planned replacement
    (replace, skip, defer, move to a shut), each a present-value cost,
    from an explicit state, on the part (`NonRepairable`), as the first of
    the RCM calculators.

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

- **Status:** answered by hand with today's RePyability; not built.
- **Who asks, and why:** the planner of UC-7: a long-run cost a day says
  which policy is cheapest, but a budget starts from the bearing as it
  is, and needs to know how far the spend can stray from the mean.
- **What they know:** UC-7's bearing, prices and policies; its state: 224
  days old, the next shut in 21 days (159 days into the 180-day shut
  interval).
- **What they want back:** the cost over windows of 82, 447, 813 and 1,086
  days from now: its mean and its spread (10th, 50th, 90th percentiles).
- **The answer** (dollars):

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
- **How it is solved:**
  - **the mean:** `RepairableRBD.expected_cost(t, state=...)`, exact, from
    `NodeState(age=224, phase=159)`;
  - **the spread:** simulated, as a spread must be:
    `RepairableRBD.cost(t, state=..., seed=...)` for each window with one
    seed. Simulation *r* draws the same lives whatever the window, so the
    runs to each window are the same histories cut off there.
- **Checks:** the simulated means agree with the exact ones within $2
  (20,000 simulations each).
- **Learned:** each window's spread must come from the same simulations,
  or the 90th percentiles of longer windows need not even rise; the
  seeded streams give that for free. What RePyability could give in one
  call: a policy's cost over several windows from a state, its mean exact
  and its distribution from one run.

### UC-9: "What will the bearing cost in each of a run of consecutive windows from now, and how often does each outcome happen in each?"

(First asked by financial year, July to June, from 11 October 2026; the
windows here, 0 to 263, 263 to 629, 629 to 994 and 994 to 1,359 days
from now, are those years as Reliafy would turn them. Summing by
financial year is Reliafy's.)

- **Status:** answered by hand with today's RePyability; not built.
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
- **How it is solved:** as UC-8, at each window's end: the means by
  differencing `expected_cost(t, state=...)`, exact; each simulation's
  cost in a window as its cost to the window's end less its cost to its
  start, from one seed (`cost(...).samples`), so every window's outcomes
  are of the same histories.
- **Checks:** no simulation spends less than $0 in a window; the
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

### UC-11: "My pump's seal is replaced at every 500-hour service and whenever it fails. It has run 340 hours, and the next service is in 160. What's the chance it fails before each service, what will failures and replacements cost over the next 944 running hours, and is replacing it at every service worth it?"

(It comes from Reliafy's side: the pump's on and off times give its
running hours, so the seal's age (340 h, since it went in at the last
service), how far the pump is into its service interval (340 of 500 h)
and, through the operating schedule, how many running hours lie ahead
(944 h to the end of the year, at 16 hours a weekday). Turning the
answers back into dates is Reliafy's too.)

- **Status:** answered by hand with today's RePyability; design to agree.
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
    `replace_cost` 900, started from `NodeState(age=340, phase=340)`:
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
  state = {"seal": NodeState(age=340.0, phase=340.0)}
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
    phase in running hours, RePyability the risk and cost in running
    hours, and the schedule turns them into dates.

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
    chance while the machine is apart, is cheaper).
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
    replacing now dearer than a planned replacement is never best; a
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

