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
    before or after. That waits for a use case (perhaps with the
    `holidays` package).
  - **Other rules.** Every other month, quarterly, yearly, a set number
    of times: dateutil has them all. Each gets a readable form here when
    a use case needs it. Accepting a dateutil `rrule` (or iCalendar text)
    directly, for anything not covered yet, is possible but not done.

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
