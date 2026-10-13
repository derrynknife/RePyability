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
| UC-7 | The cheapest replacement policy when replacing at a shut is cheaper than between shuts | Answered by hand |
| UC-8 | Each policy's cost over a window from a state, and its spread | Answered by hand |
| UC-9 | The cost in each of a run of windows from a state, and how often each outcome happens | Answered by hand |
| UC-10 | Tasks by use on an operating schedule: calendar only | Reliafy's |

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

- **Status:** answered by hand with today's RePyability; not built. It
  would be one of the RCM calculators.
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
