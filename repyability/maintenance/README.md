# The maintenance forecaster: use cases

This is the working record for the planned-maintenance forecaster (#319).
It is built **one use case at a time**. Each use case is a question someone
actually asks, written down before any code. The record says how the
question is answered, and what was learned along the way. The forecaster's
design follows from the use cases; nothing goes in that no use case needs.

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
- Results are plain data (`results._ResultMapping`, with `to_dict()`), never plots: the Reliafy app draws them.
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

### UC-1: "What maintenance work is coming on this system over the next year, month by month?"

- **Status:** answered by hand. The forecaster is not built yet.
- **Who asks, and why:** a maintenance planner, building next year's
  work plan and budget for one system already in service.
- **What they know:**
  - the system's diagram, with each component's fitted life and repair
    models and its maintenance (age replacement, block replacement on a
    calendar, proof tests of hidden failures);
  - each component's state now: its age, and where it is in its calendar.
- **What they want back:** for each month, the expected number of:
  - preventive replacements, by component;
  - corrective repairs, by component;
  - tests, by component;
  - planned system outages.
- **How it is solved:** `RepairableRBD.expected_events(t, state=...)`
  already gives each count over `[0, t)` from the state now, exactly (no
  simulation). It also counts the events due at exact times (an age, a
  block date) exactly. A month's expected count is the difference of the
  counts at its two ends, since expectations add over windows. So the
  forecast is `expected_events` at the month boundaries, differenced.
- **Today:** this works by hand:

  ```python
  import numpy as np
  import surpyval as surv
  from repyability import NodeState, RepairableRBD

  month = 730.0  # hours
  fixed = surv.ExactEventTime.from_params
  rbd = RepairableRBD(
      [("s", "pump"), ("pump", "valve"), ("valve", "sensor"), ("sensor", "t")],
      {
          "pump": {   # replaced at age 6000 h, 4 h of work
              "reliability": surv.Weibull.from_params([9000.0, 2.5]),
              "repairability": surv.LogNormal.from_params([np.log(8.0), 0.5]),
              "preventive": {"interval": 6000.0, "duration": fixed([4.0])},
          },
          "valve": {  # replaced every 8760 h on the calendar, 6 h of work
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
  ends = month * np.arange(1, 13)
  events = rbd.expected_events(ends, state=state)
  by_month = lambda cumulative: np.diff(cumulative, prepend=0.0)
  by_month(events.node_preventive["pump"])
  ```

  Expected counts, month by month (rounded):

  | Month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Pump replaced (age 6000 h) | 0 | 0.90 | 0 | 0 | 0 | 0 | 0 | 0 | 0.05 | 0.65 | 0.00 | 0.01 |
  | Valve replaced (calendar) | 0 | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | Sensor tested | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
  | Pump repaired | 0.09 | 0.01 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.07 | 0.07 | 0.03 | 0.01 | 0.02 |
  | Valve repaired | 0.03 | 0.03 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
  | Sensor failure found at a test | 0 | 0.08 | 0 | 0 | 0 | 0 | 0 | 0.08 | 0 | 0 | 0 | 0 |
  | System planned outages | 0 | 0.83 | 0.99 | 0 | 0 | 0 | 0 | 0 | 0.05 | 0.63 | 0.00 | 0.01 |

  How to read it:
  - **The pump** reaches age 6000 h in month 2 and is replaced then,
    unless it failed first: 0.90 is the chance it survives the 800 h from
    5200 h to 6000 h. Its replacement then reaches 6000 h in months 9 to
    12, depending on when it was put in.
  - **The valve** is on a calendar, so its replacement is certain, in
    month 3.
  - **The sensor** is tested in months 2 and 8.
- **Checks:** the months add up to `expected_events` over the year.
  Calendar tasks land on their dates, worked out by hand. The age
  replacement's first chance matches `R(6000) / R(5200)`.
- **Learned / open questions:**
  - **A task and a system outage are different counts.** In month 2 the
    pump's replacement is 0.90 likely, but the system's planned outages
    are 0.83. The sensor's failures are hidden, so with probability about
    0.07 it has already failed unseen, and the series system is already
    down when the pump work starts: that work causes no new outage. The
    forecast must say which it reports, and explain this.
  - **Differencing a numerical count can give tiny negatives**
    (`-1e-12`) in a month with no events. The forecaster should present
    them as 0, without changing the sums.
  - Doing this by hand means knowing to difference, picking the node
    dictionaries apart, and getting no calendar of *when* in the month,
    no hours, parts or cost, and no spread. Those are the next use
    cases' questions.

### UC-2: "I've got a weekly task, started 15 July 2026. How many times will it happen through the rest of 2026?"

- **Status:** built: `Task` and `TaskDates`
  (`repyability/maintenance/_tasks.py`).
- **Who asks, and why:** anyone with a recurring task on a fixed
  calendar (an inspection, a lubrication round, a test), planning the
  rest of the year.
- **What they know:** the task's first date (Wednesday 15 July 2026), its
  interval (a week), and the period asked about: from today (Sunday 11
  October 2026) to 31 December 2026, both days included.
- **What they want back:** how many times, and ideally the dates.
- **The answer:** **12 times**: every Wednesday from 14 October to 30
  December 2026 (14, 21, 28 Oct; 4, 11, 18, 25 Nov; 2, 9, 16, 23, 30
  Dec). Counting the 13 already done since 15 July, 25 in all of 2026.
- **How it is solved:** this needs no probability at all. A calendar
  task happens on its dates whatever the equipment does, so the dates are
  the first date plus whole intervals, kept to those inside the period.
  It is exact by date arithmetic.
- **Today:** RePyability has no dates and no tasks of their own, only
  components with maintenance. The count can be had by dressing the task
  up as a block-replaced component and converting the dates to day
  numbers by hand:

  ```python
  import surpyval as surv
  from repyability import NodeState, RepairableRBD

  rbd = RepairableRBD(
      [("s", "task"), ("task", "t")],
      {
          "task": {
              # a life far longer than the window: PerfectReliability is
              # refused ("never fails, so it takes no preventive")
              "reliability": surv.Exponential.from_params([1e-12]),
              "repairability": "instant",
              "preventive": {"interval": 7.0, "policy": "block"},
          }
      },
  )
  # 82 days from 11 Oct to 1 Jan (the window [0, t) leaves its end out);
  # phase: 4 days since the last time, 7 Oct
  rbd.expected_events(82.0, state={"task": NodeState(phase=4.0)}).node_preventive
  # -> {'task': 12.0}
  ```

- **Checks:** the dates by hand (above). The dates one interval apart,
  with the first on or after the period's start and the last on or before
  its end. A whole year of a weekly task from 1 January gives 52 or 53.
- **Learned / open questions:**
  - **Dates are new.** Everything in RePyability is a number of time
    units from 0, in the models' own unit. The forecaster has to take
    dates and turn them into model time with an origin date and a unit
    (days, hours). The conversion belongs at the edge: no result inside
    may depend on the unit (`CLAUDE.md`).
  - **A task is not a component.** A plain recurring task has nothing
    to fail, and RePyability has nowhere to put it.
  - **The ends of the period.** People say "through 31 December" and
    mean the day included. `expected_events` counts `[0, t)`, so the
    window has to end at 1 January. The forecaster should take dates
    with both ends included and do the conversion itself.
  - **Later, intervals that aren't a fixed length.** "Monthly" or "on
    the first Monday" aren't a number of days. They are not needed here,
    but the design shouldn't rule them out.
- **Design, as built (the smallest thing that answers it):** a `Task` in
  `repyability.maintenance`: a name, a first date (a `datetime.date`, or
  text such as `"2026-07-15"`) and an interval (a `datetime.timedelta`,
  or a number of days). Its `due(start, end)` returns its dates within
  the period, both ends included, as a `TaskDates` result (the dates and
  their `count`, with `to_dict()`). It uses the standard
  library's dates (no new dependency). Two changes from the first
  proposal:
  - The count is `due.count`, not `len(due)`: a result is a read-only
    mapping of its fields, so its `len` is the number of fields.
  - `to_dict()` (every result's, through `results.plain`) now writes a
    date as ISO text, as JSON has no dates.

  ```python
  from datetime import timedelta
  from repyability.maintenance import Task

  conveyor = Task("inspect conveyor", first="2026-07-15", every=timedelta(weeks=1))
  due = conveyor.due("2026-10-11", "2026-12-31")
  due.count   # 12
  due.dates   # 2026-10-14, 2026-10-21, ..., 2026-12-30
  ```

  `test_maintenance_tasks.py` checks the UC-2 answer, and the dates
  against the calendar walked a day at a time on random tasks and
  periods.
- **Next, noted by the maintainer:** a task will hold a collection of
  tasks (a weekly round made of several jobs, say). Not built yet: it
  waits for a use case that needs it.

### UC-3: "How much will this cost?" (the weekly task of UC-2)

- **Status:** built: `Task(..., cost=)` and `TaskDates.cost`.
- **Who asks, and why:** the same planner, budgeting for the task over
  the rest of the year.
- **What they know:** the task of UC-2, and that it costs $250 each
  time, the same every time.
- **What they want back:** just the total, not a split by month.
- **The answer:** **$3,000**: 12 times at $250.
- **How it is solved:** the task's cost each time, times the number of
  times it falls due in the period. It is exact, with no probability, as
  the dates are.
- **Today (before this use case):** `due.count * 250`, by hand. A
  `RepairableRBD` could also give it, through the block-replaced
  component of UC-2 with a preventive cost, but only with the same
  workarounds.
- **Design, as built:** `Task` takes a `cost` each time (a number, at
  least 0, in the user's currency; by default None, not given), and
  `TaskDates.cost` is the total, `count * cost`. It is None when the task
  has no cost, so a cost never given is not mistaken for a free task.

  ```python
  conveyor = Task("inspect conveyor", first="2026-07-15",
                  every=timedelta(weeks=1), cost=250.0)
  conveyor.due("2026-10-11", "2026-12-31").cost   # 3000.0
  ```

  `test_maintenance_tasks.py` checks the UC-3 answer, the cost as the
  count times the cost each time, and the input checks.
- **Learned / open questions:**
  - **One price, no breakdown.** The planner gave one figure. Hours,
    parts and rates (and so the labour plan) wait for a use case that
    needs them.
  - **The same price every time.** A price that changes (from 1 January,
    say) would make the cost depend on the date, not just the count.
  - **No currency.** A cost is a number in whatever currency the user
    works in, as `RepairableRBD`'s costs are.

### UC-4: "What about the first Tuesday of the month? And on the 30th of every month?"

- **Status:** built: `Monthly`, on python-dateutil.
- **Who asks, and why:** the same planner, whose tasks are not all a
  fixed number of days apart: many are set by the calendar's months.
- **What they know:** the rule (the first Tuesday; the 30th), and the
  period: from today (11 October 2026) to 31 December 2026.
- **What they want back:** how many times each falls due, and when.
- **The answer:**
  - **First Tuesday: 2 times**, on 3 November and 1 December. October's
    was the 6th, already past.
  - **The 30th: 3 times**, on 30 October, 30 November and 30 December.
- **Decisions (the maintainer's):**
  - **Short months are an option, defaulting to the last day.** February
    has no 30th. `short_months="last"` (the default) makes the task fall
    due on the month's last day instead (28 February, or the 29th in a
    leap year); `"skip"` leaves the month without it (the iCalendar
    standard's own behaviour).
  - **python-dateutil does the calendar arithmetic, behind RePyability's
    own rules.** Calendars get complicated (every other month, the last
    weekday, a set number of times, exclusions), and dateutil's `rrule`
    is the iCalendar standard's recurrence, used for twenty years. Every
    install already had it, through surpyval's pandas and matplotlib; it
    is now a dependency of its own (2.8.2 or later). Users write readable
    rules (`Monthly(30)`, `Monthly(weekday="Tuesday", nth=1)`), not
    iCalendar text.
- **How it is solved:** every schedule is a dateutil rule from the task's
  first date: a fixed interval is a daily rule every *n* days, and a
  monthly rule names a day or a weekday of the month. The last day for a
  short month is the latest of the 28th up to the day that the month has
  (the standard's `BYSETPOS=-1`). A period's dates are the rule's between
  its ends, both included. It is exact, with no probability.
- **Design, as built:**

  ```python
  from repyability.maintenance import Monthly, Task

  Task("check pumps", "2026-07-15", every=Monthly(weekday="Tuesday", nth=1))
  Task("meter reading", "2026-07-15", every=Monthly(30))                        # February: the 28th
  Task("meter reading", "2026-07-15", every=Monthly(30, short_months="skip"))   # February: none
  Task("month end", "2026-07-15", every=Monthly(weekday="Friday", nth=-1))      # the last Friday
  ```

  With a monthly rule, `first` is when the schedule starts: the task falls
  due on the rule's dates from then on. `nth` may be 1 to 4 or -1 to -4,
  which every month has; a fifth Tuesday is not in every month, and would
  raise the short-month question again.
- **Checks:** `test_maintenance_tasks.py` checks the UC-4 answers;
  February under each option, in a leap year too; the 31st (12 times a
  year, or the 7 long months); and every rule's dates against the
  calendar walked a day at a time (on 200 random rules, tasks and periods,
  over several years). The weekly tasks of UC-2 now go through dateutil
  too, and still match their walk. The tests pass with dateutil 2.8.2,
  the floor.
- **Learned / open questions:**
  - **Holidays and working days.** The last Friday of December 2026 is
    Christmas Day. dateutil can leave out listed dates, but has no
    holiday calendars, and nothing moves a date to the working day
    before or after. **Set aside by the maintainer:** public holidays
    are not to be implemented for now.
  - **Other rules.** Every other month, quarterly, yearly, a set number
    of times: dateutil has them all. Each gets a readable form here when
    a use case needs it. Accepting a dateutil `rrule` (or iCalendar text)
    directly, for anything not covered yet, is possible but not done.

### UC-5: "My bearing was installed on 1 March 2026 and is replaced when it fails. What's the chance it fails before the end of 2026, how many replacements should I expect, and at $400 each, what will they cost?"

- **Status:** built: `Component` and `FailureForecast`
  (`repyability/maintenance/_components.py`).
- **Who asks, and why:** a planner with a non-repairable part (a bearing)
  that is run to failure and replaced, budgeting for the rest of the year.
- **What they know:** the bearing's life, a Weibull with a scale of 365
  days and a shape of 2 (fitted in surpyval); that the current one was
  installed on 1 March 2026 and is still working today (11 October 2026);
  that a failed one is replaced with a new one, for $400.
- **What they want back:** for 11 October to 31 December 2026, the chance
  it fails, the expected number of replacements, and their expected cost.
- **The answer:** the bearing is 224 days old, and the period is 82 days
  (both ends included):
  - **chance it fails: 27.8%**, `1 - R(306) / R(224)`;
  - **expected replacements: 0.283**, a little more than the chance of a
    failure, because a new bearing can fail before the year is out too;
  - **expected cost: $113.20**, 0.283 times $400.
- **How it is solved:**
  - **The chance it fails:** the life conditioned on the age reached,
    `1 - R(age + days) / R(age)`, exact.
  - **The expected replacements:** the renewal count from that age,
    which `RepairableRBD.expected_events` already works out exactly (no
    simulation, to about 1e-7) for a component replaced in no time,
    started from its state. The forecaster calls it rather than having a
    second renewal calculation of its own.
- **Today (before this use case):** a one-component diagram, with the
  dates turned into days by hand:

  ```python
  rbd = RepairableRBD(
      [("s", "bearing"), ("bearing", "t")],
      {"bearing": {"reliability": life, "repairability": "instant",
                   "replace_cost": 400.0}},
  )
  state = {"bearing": NodeState(age=224.0)}
  rbd.expected_events(82.0, state=state).node_failures   # {'bearing': 0.2830}
  rbd.expected_cost(82.0, state=state).mean              # 113.20
  ```

  The chance of a failure is not among its results. A component with
  redundancy around it is another question (#313).
- **Design, as built:** a `Component` in `repyability.maintenance`, beside
  `Task`: a name, a life (a surpyval model, in days), the date the
  current unit was installed, and a cost each replacement (optional, as
  for a task). Its `failures(start, end)` gives a `FailureForecast`: the
  `probability`, `expected_failures` and `expected_cost`.

  ```python
  bearing = Component("bearing", surv.Weibull.from_params([365.0, 2.0]),
                      installed="2026-03-01", cost=400.0)
  forecast = bearing.failures("2026-10-11", "2026-12-31")
  forecast.probability         # 0.2784
  forecast.expected_failures   # 0.2830
  forecast.expected_cost       # 113.20
  ```

- **Checks:** `test_maintenance_components.py` checks:
  - the UC-5 answers, against the closed form and a plain simulation;
  - an exponential life, whose expected failures are its rate times the
    days whatever its age;
  - Weibull units of several ages and periods, against the simulation;
  - a one-day period, where the expected failures are the chance of one;
  - the input checks.
- **Learned / open questions:**
  - **Days only.** A life in running hours needs the hours a day it runs
    to put it on the calendar. That waits for a use case.
  - **The current unit is taken to be working at the start** of the
    period, and the period may not start before it was installed.
  - **Replacements take no time.** A replacement that takes days, or
    waits for a part, waits for a use case.

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
  - **Days, not dates, so far.** "Right after a failure" needs only the
    days to the next planned date. Tying the plan to a `Component` (its
    installed date) and a `Task` (the planned dates) waits for a use
    case.
  - **Replacements take no time**, as in UC-5.

### UC-7: "Replacing at a scheduled shut is cheaper than a planned replacement off the schedule. What's the best way to replace the bearing?"

- **Status:** answered by hand; design proposed, not built.
- **Who asks, and why:** the planner of UC-6. The plant has planned shuts
  every 180 days (the schedule). Replacing the bearing in a shut is
  cheaper and easier than a planned replacement that needs its own
  outage, which is cheaper than a failure. The maintainer's framing:
  "just another part of the optimisation problem".
- **What they know:** the bearing of UC-5 (Weibull, scale 365 days,
  shape 2); shuts every 180 days; **$100 at a shut, $150 planned off the
  schedule, $400 at a failure** (the maintainer chose these).
- **What they want back:** the cheapest way to replace it, in the long
  run.
- **How it was framed (proposed by Claude, as the maintainer had not
  pinned it down):** five policies, the last containing the others:
  1. **run to failure** ($400 each);
  2. **age replacement off the schedule** at age τ (the plan resets),
     $150 each;
  3. **every shut**, or **every second shut** (a calendar plan), $100
     each;
  4. **at each shut by age:** replace in a shut only if the bearing is
     at least *a* days old, else leave it to the next shut. This is
     UC-6's skip decision, taken at every shut;
  5. **both:** in a shut if at least *a* days old, or off the schedule
     at age τ if that comes first.
- **The answer** (long-run cost a day, undiscounted):

  | Policy | Cost a day |
  |---|---|
  | Run to failure | 1.237 |
  | Age replacement off the schedule, best τ about 300 days | 1.117 |
  | Every second shut | 1.098 |
  | Every shut | 1.056 |
  | **At each shut, if at least about 138 days old** | **0.990** |

  - **The best policy:** replace in a shut only if the bearing is at
    least about 138 days old. Right after a failure, skip the next shut
    if it is less than about 138 days away, and replace at the one after.
    Anywhere from 120 to 180 days costs within 0.1% of it.
  - **An off-schedule replacement adds nothing at these prices.** The
    shuts come often enough that the best τ for policy 5 is "never".
    Dearer failures, or shuts further apart, would change that.
  - **It agrees with UC-6.** With the shut price, UC-6's NPV of skipping
    the next shut (right after a failure, calendar plan) changes sign
    when the shut is about 145 days away at 8% a year (142 undiscounted):
    the one-off decision and the long-run policy say the same thing.
- **How it is solved:**
  - **Policies 1 to 3:** a one-component `RepairableRBD` with
    `expected_cost_rate()` (a replacement on failure at $400, and
    preventive age or block replacement), exact.
  - **Policy 4:** exactly too, through replacement on condition (#145),
    which inspects on a calendar and replaces a unit more likely than a
    threshold to fail before the next inspection. For a life whose hazard
    rises (as this Weibull's), that is a unit at least *a* days old: the
    threshold is `1 - R(a + 180) / R(a)`. The best *a* then comes from a
    one-dimensional search.
  - **Policy 5:** simulated by hand here (RePyability's policies do not
    combine a condition at inspections with an age off them). With these
    prices it is policy 4.
- **Today:** all by hand, as above. The thresholds are found by turning
  an age into a condition threshold, which only holds for a rising
  hazard.
- **Checks (by hand):** the exact cost rates against a plain simulation
  of every policy (which runs about 0.001 a day low, from stopping at
  150,000 days: its run to failure gives 1.2354, the exact 1.2366); every
  shut against the calendar plan's exact value; and UC-6's break-even
  against the best threshold.
- **Proposed design (to agree with the maintainer):** a way to say "the
  shuts" once (every 180 days, $100), and to compare the policies with
  the three prices, giving each policy's cost a day and the best
  threshold. Open:
  - **Undiscounted, or discounted?** UC-6 used 8% a year; the long-run
    comparison here is undiscounted.
  - **Ages rather than conditions.** A policy "at least *a* days old at a
    shut" should be said as an age, worked out for any life (where the
    hazard does not rise, an age and a condition threshold differ).
  - **Policy 5 exactly**, if a case needs it: a unit's phase between
    shuts makes it a Markov chain on the phase, to be worked out on a
    grid.
- **Learned:** the cheaper shut price is what makes skipping a shut
  worthwhile only for young bearings. With the off-schedule price ($150)
  at the shuts, as in UC-6, skipping always paid.

### UC-8: "Those cost rates are good, but what does that actually look like over a timeline, and what is the range of those costs over time?"

- **Status:** answered by hand; design proposed, not built.
- **Who asks, and why:** the planner of UC-7. A long-run cost a day says
  which policy is cheapest, but a budget is for real dates, from the
  bearing as it is now, and needs to know how far the actual spend can
  stray from the average.
- **What they know:** UC-7's bearing, prices and policies; the bearing
  in now is 224 days old today (11 October 2026, as in UC-5); the next
  shut is on **1 November 2026** (21 days away; Claude's assumption),
  then every 180 days.
- **What they want back:** for each policy, the cost from today to each
  date ahead, its mean and its range.
- **The answer** (the cost from 11 October 2026 to the end of each
  quarter shown, in dollars; the range from the 10th to the 90th
  percentile):

  **The best policy** (replace at a shut if at least 138 days old):

  | To the end of | Mean | 10% | 50% | 90% |
  |---|---|---|---|---|
  | Dec 2026 | 133 | 100 | 100 | 100 |
  | Dec 2027 | 487 | 300 | 300 | 900 |
  | Dec 2028 | 846 | 500 | 800 | 1,400 |
  | Sep 2029 | 1,090 | 600 | 1,000 | 1,600 |

  **Three years on** (to the end of September 2029):

  | Policy | Mean | 10% to 90% | Long run, a year |
  |---|---|---|---|
  | Best: at a shut if at least 138 days old | 1,090 | 600 to 1,600 | 362 |
  | Every shut | 1,155 | 600 to 1,800 | 386 |
  | Run to failure | 1,373 | 800 to 2,000 | 452 |

  - **The first months cost more than the long run** (about $590 a year
    to December 2026, against $362): the bearing in now is old, and is
    almost certain to be replaced at the November shut.
  - **The spend comes in steps**, $100 at a shut and $400 at a failure,
    so the percentiles jump: half the time the best policy spends $300
    by the end of 2027, but one time in ten $900 or more (a failure or
    two).
  - **The best policy lowers the mean more than the range.** Its 90th
    percentile is $200 under every shut's after three years, its 10th
    the same.
- **How it is solved:**
  - **The mean:** `RepairableRBD.expected_cost(t, state=...)` at each
    date, exact, from the bearing's state today (its age, and the days
    since the last shut: `NodeState(age=224, phase=159)`). The policies
    are UC-7's: replacement on condition at the shuts for the best one,
    block replacement for every shut.
  - **The range:** simulated, as a range must be (the README: "the answer
    is a sample"). `RepairableRBD.cost(t, state=..., seed=...)` at each
    date with one seed: simulation *r* draws the same lives whatever the
    window, so the runs to each date are the same histories cut off
    there, and each date's percentiles are of one set of timelines.
    Their means agree with the exact ones at every date (within $2,
    20,000 simulations each).
- **Today:** all of it by hand, as above: a one-component diagram per
  policy, the dates turned into days, and a simulation per date.
- **Proposed design (to agree with the maintainer):** a forecast over
  dates for a component under a plan, the mean exact and the range
  simulated, from its state today. Open:
  - **Cumulative, or each period's own?** The tables above are the
    cost to each date. A quarter's own spend is lumpy (mostly $0 or
    $100, $400 with a failure); both could be given.
  - **Which range?** 10% to 90% here; or any percentiles asked for.
  - **Dates ahead:** month ends, quarter ends, or each shut.
  - **The plan as an input:** UC-7's policies need the shuts (a `Task`)
    and the rule at them (an age), tied to a `Component` (its life,
    installed date and prices). UC-6 and UC-7 left that open too.
- **Learned:** each date's range must come from the same simulations, cut
  off there, or the 90th percentiles of successive dates need not even
  rise. The seeded streams give that for free.

### UC-9: "Accountants need this in financial year sums."

- **Status:** answered by hand; design proposed, not built.
- **Who asks, and why:** the planner of UC-8, passing the forecast to the
  accountants, who budget each financial year's own spend: not the
  running total, and not calendar quarters.
- **What they know:** UC-8's bearing, policies, prices and shuts. The
  financial year runs **July to June** (Claude's assumption, as in
  Australia: FY2028 is 1 July 2027 to 30 June 2028); the forecast starts
  today, 11 October 2026, part way through FY2027.
- **What they want back:** each financial year's cost, its mean and its
  range.
- **The answer** (dollars; the range from the 10th to the 90th
  percentile):

  | Financial year | Best policy: mean | 10% / 50% / 90% | Every shut: mean | Run to failure: mean |
  |---|---|---|---|---|
  | FY2027 (from 11 Oct 2026) | 307 | 200 / 200 / 500 | 330 | 356 |
  | FY2028 | 359 | 200 / 200 / 600 | 382 | 452 |
  | FY2029 | 359 | 200 / 200 / 600 | 382 | 452 |
  | FY2030 | 359 | 200 / 200 / 600 | 382 | 451 |
  | **All four** | **1,384** | **800 / 1,400 / 2,000** | 1,477 | 1,711 |

  - **A typical year is two shuts and no failure** ($200, the median);
    the mean ($359) carries the chance of a failure ($400) on top. A
    full year (FY2028) costs $200 in 60% of the simulations, $500 (a
    failure, and one shut replacement) in 27%, and more in the rest, so
    a budget of the mean is right on average but short in about two
    years in five (40%).
  - **Yearly 90th percentiles do not add up.** For the best policy they
    sum to $2,300 over the four years, but the four years' own 90th
    percentile is $2,000: bad years seldom come together. A contingency
    for the whole period should be set on the period, not by adding the
    years'.
  - The part year, FY2027, has both of its shuts in it (1 November 2026
    and 30 April 2027), so it costs nearly a full year's.
- **The outcomes of each year** (asked for after the percentiles above
  misled: with the costs in steps, 60% of years at exactly $200 make the
  10th and 50th percentiles both $200). The share of years with each
  outcome, from 1,000,000 plain simulations; RePyability's 20,000 agree
  with every row within about 0.3 points:

  | What happens in the year | Cost | FY2027 (from 11 Oct) | FY2028 | FY2029 | FY2030 |
  |---|---|---|---|---|---|
  | No failure, one shut replacement | $100 | – | 0.9% | 1.2% | 1.5% |
  | No failure, two shut replacements | $200 | 70.7% | 59.1% | 58.6% | 58.2% |
  | One failure, no shut replacement | $400 | – | 0.5% | 0.7% | 0.8% |
  | One failure, one shut replacement | $500 | 22.1% | 27.2% | 26.9% | 26.4% |
  | One failure, two shut replacements | $600 | 3.0% | 2.9% | 3.2% | 3.5% |
  | Two failures, no shut replacement | $800 | 1.4% | 4.8% | 4.6% | 4.5% |
  | Two failures, one shut replacement | $900 | 2.5% | 3.5% | 3.6% | 3.8% |
  | Three or more failures, or two and both shuts | $1,000+ | 0.3% | 1.2% | 1.2% | 1.2% |
  | Mean | | $307 | $359 | $359 | $359 |

  The chance of spending more than $200 (any failure) is 29% in FY2027
  and 40% a full year; more than $600, 4% and 9%; more than $900, 0.3%
  and 1.2%. A $100 or $400 year comes when a failure late in the year
  before leaves a bearing too young to replace at the year's first shut.
- **How it is solved:** as UC-8, at each financial year's end:
  - **The mean:** the differences of `expected_cost(t, state=...)` at the
    years' ends, exact.
  - **The range:** each simulation's cost in a year is its cost to the
    year's end less its cost to the year's start, from the same seed
    (`cost(...).samples`), so each year's percentiles are of the same
    histories. Checked: no simulation spends less than $0 in a year, and
    the simulated means agree with the exact ones within $4 (20,000
    simulations).
- **Today:** by hand, as UC-8.
- **Proposed design (to agree with the maintainer):** UC-8's forecast,
  summed by period: a financial year with its starting month (July here;
  April or October elsewhere), calendar years, quarters or months; and
  over the whole horizon, whose range is its own (not the years' added).
  Open: the starting month, the percentiles, and how many years ahead.
- **Learned:**
  - **Lumpy costs need their outcomes, not percentiles.** When costs
    come in a few steps, percentiles hide the shape (here 10% / 50% /
    90% = $200 / $200 / $600), and read as if no year were cheap. The
    forecaster should give each period's outcomes (what happened: the
    failures and planned replacements, and its cost, with its share) and
    the chance of exceeding a budget, as above.
  - **The per-period range** needs each simulation's cost per
    period, not only its total: `cost()` gives the total of one window,
    so by hand it took a run per year's end. The forecaster should sum
    each simulation's costs into the periods in one run.

## Candidates

Questions to take up next, one at a time, in whatever order proves useful.
Each becomes a full use case above when it is taken up.

- **Calendar:** "Which tasks are due, on which dates?" A list of
  scheduled tasks (component, kind, due date, the probability it happens
  then), not just counts per month.
- **Workload and budget:** "How many hours, which parts and how much money
  does each month need?" Each task's hours, parts and cost summed per
  period.
- **Planning margin:** "How busy could a month get?" The spread of each
  month's work (the probability of more than *k* tasks, a 90% plan), not
  just its mean.
- **Fleet:** "What's coming across all our assets?" Many systems, each in
  its own state, summed.
- **Grouping:** "Which tasks could be done together?" Tasks due close
  together on one asset, and the outages saved by combining them.
- **What if:** "What happens to the plan if we stretch the pump interval
  to 7000 h?" The forecast under a changed plan, beside the current one.
- **Plan against actual:** "Are we doing the work we forecast?" The
  forecast against the work recorded (from outage logs, as timelines).
