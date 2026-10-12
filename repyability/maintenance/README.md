# Maintenance: use cases

This is the working record of the maintenance pieces RePyability keeps
(#319), built **one use case at a time**: each use case is a question
someone actually asks, written down before any code, with how it is
answered and what was learned.

**RePyability is about timelines; Reliafy is about calendars** (the scope
note in `CLAUDE.md`). The calendar side, tasks on dates and monthly rules,
operating schedules and meters read by date, moved to Reliafy
(`backend/services/maintenance_schedule.py`, Reliafy/reliafy#389), with
the record of its use cases (UC-1 to UC-4 and UC-10, and UC-7 to UC-9,
the forecaster's requirements, in Reliafy's `docs/maintenance-scheduling.md`).
What stays here works in model time, from a state: a part's failures from
its age (UC-5) and the choice at a planned replacement (UC-6), the first
of RCM's quantitative questions.

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
- **What they want back:** (the shape of the answer: a table by period, a calendar, a number)
- **How it is solved:** (the method; exact, numerical or simulated; the existing code it builds on)
- **Today:** (how close RePyability gets now: a script and its output)
- **Checks:** (how we know the answer is right)
- **Learned / open questions:**
```

## Use cases

### UC-5: "My bearing was installed on 1 March 2026 and is replaced when it fails. What's the chance it fails before the end of 2026, how many replacements should I expect, and at $400 each, what will they cost?"

- **Status:** built: `NonRepairable(..., unplanned_cost=)` and its
  `failures(t, age=)`, giving a `FailureForecast`.
- **Who asks, and why:** a planner with a non-repairable part (a bearing)
  run to failure and replaced, budgeting for the rest of the year.
- **What they know:** the bearing's life, a Weibull with a scale of 365
  days and a shape of 2 (fitted in surpyval); the current one was
  installed on 1 March 2026 and is working today (11 October 2026); a
  failed one is replaced with a new one, for $400.
- **The answer:** the bearing is 224 days old, and 82 days are left to the
  end of 31 December: **a 27.8% chance it fails, 0.283 replacements
  expected (a new bearing can fail too), $113.20.**
- **How it is solved:** in model time (RePyability is about timelines:
  the dates are turned into an age and a window by Reliafy, or by hand):
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
  - **The plan does not reset** (calendar, block replacement): planned
    replacements fall on fixed dates. Right after a failure, the next
    date may be any number of days away, up to 180; a failure just
    before it leaves a nearly new bearing to be replaced. Skipping it,
    the bearing runs on to a failure or the following date. Both choices
    replace at the following date, so they differ only before it.
- **The answer:**
  - **Resetting plan: skipping saves $26.56.**
  - **Calendar plan:** it depends on the days to the next date:

    | Days to the next planned date | 1 | 10 | 30 | 60 | 90 | 120 | 150 | 180 |
    |---|---|---|---|---|---|---|---|---|
    | NPV of skipping ($) | 149.06 | 140.73 | 123.08 | 99.06 | 78.21 | 60.68 | 46.42 | 35.26 |

  Skipping pays every time: a 180-day plan is too frequent for this
  bearing. `NonRepairable.optimal_replacement_policy()` puts the best
  age at about 298 days (undiscounted). On a 365-day calendar plan the
  answer changes sign: skipping pays when the next date is less than
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
  - **Calendar:** `NPV = planned exp(-r d) + exp(-r d) F(T) - (F(d + T) -
    F(d))`, `d` the days to the next date and `F(t)` the expected
    discounted cost of a new bearing's failures (and its replacements')
    in `[0, t)`: `RepairableRBD.expected_cost(t, discount_rate=r)`.
  - **The discount rate** is a rate a year, `i`; costs are discounted
    continuously at `r = log(1 + i) / 365.25` a day, so a cost a year
    away is worth `1 / (1 + i)`.
- **Today (before this use case):** the calendar plan by hand from three
  `expected_cost` calls and the formula above; the resetting plan not
  at all (`NonRepairable` prices age replacement only in the long run,
  undiscounted).
- **Design, as built:**

  ```python
  from repyability.maintenance import ReplacementPlan

  life = surv.Weibull.from_params([365.0, 2.0])
  resetting = ReplacementPlan(life, every=180, planned_cost=150, unplanned_cost=400, resets=True)
  resetting.skip_next(discount_rate=0.08).npv                    # 26.56

  calendar = ReplacementPlan(life, every=180, planned_cost=150, unplanned_cost=400, resets=False)
  calendar.skip_next(discount_rate=0.08, days_to_next=30).npv    # 123.08
  ```

  `resets` has no default: the two plans answer differently, so it is
  always said. A calendar plan needs `days_to_next`; a resetting one
  refuses it (its next planned replacement is always `every` away).
- **Checks:** `test_maintenance_plans.py` checks:
  - the answers against plain simulations of both plans (the resetting
    one over 300 years);
  - an exponential life against both plans' closed forms, at 0%, 8% and
    50% a year;
  - the resetting plan with no discounting against the classic
    undiscounted comparison, and a tiny rate next to it;
  - the 180-day plan's NPVs falling with the days to the next date, and
    the 365-day plan's change of sign;
  - the input checks.
- **Learned / open questions:**
  - **The plan itself may be the question.** With these costs a 180-day
    plan is too frequent, so skipping always pays. Comparing the plan
    with the best one (discounted), or the latest day at which skipping
    a calendar replacement still pays, are natural next use cases.
  - **Days, not dates.** "Right after a failure" needs only the days to
    the next planned date, and dates are Reliafy's.
  - **Replacements take no time.**
  - **The API is being reworked** (the maintainer does not like
    `skip_next`): a comparison of the options at a planned replacement
    (replace, skip, defer, move to a shut), each a present-value cost,
    from an explicit state, on the part (`NonRepairable`), as the first of
    the RCM calculators.
