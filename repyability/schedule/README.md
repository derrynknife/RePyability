# Schedules: use cases

The working record of `repyability.schedule`, built **one use case at a
time**, as `repyability/maintenance/README.md` is: each use case is a
question someone actually asks, written down before any code, with who
asks, what they know, the answer worked by hand, how it is solved, and the
checks.

**RePyability computes; Reliafy records** (the scope note in `CLAUDE.md`).
RePyability's models work in model time, and this module connects model
time to the calendar: recurrence rules (which dates), duty plans (how many
operating hours by when) and forecasts through them. It takes dates, rules,
plans and readings as plain data and returns plain data; it never reads the
clock, stores anything or knows a workspace. The caller (the Reliafy app,
or anyone) holds the records, decides what "today" is, works out the state
from its history (when a task was last done, a meter's reading), and turns
the answers into screens and work orders.

Use cases UC-2, UC-3, UC-4 and UC-10 were first answered for this package,
moved out with the calendar side in October 2026, and return here under
their own numbers with the answers recorded for them. The use cases in
model time (UC-1, UC-5 to UC-27) stay in `repyability/maintenance/README.md`.

## Layers

This module is the calendar layer (`CLAUDE.md`, Architecture). It turns
dates into model time and back, and it says what each unit's duty is.
It leaves money to `repyability.value`, a unit's plan to
`repyability.maintenance` and the system to `RepairableRBD`:

| Question | Where | Use cases |
|---|---|---|
| Which dates does a rule give? | `Recurrence` (its `dates`) | UC-2, UC-3, UC-4 |
| How many operating hours by when? | `DutyPlan`, `UsageForecast` | UC-10 |
| The calendar an analysis runs on, and each unit's duty plan | `CalendarContext` (taken by `events_by_period` and by `UnitPlan`'s methods as `calendar=`) | UC-64 |
| A part's failures and costs by month | `failures_by_period` (a `NonRepairable`), or `UnitPlan.failures_by_period` (the diagram's engine) | UC-48 |
| A system's events and costs by month | `events_by_period` (a `RepairableRBD`) | UC-39, UC-64 |

The duty plan belongs to the analysis: the same diagram or plan runs on
another site's calendar with other plans.

## Conventions

- **Dates are whole local dates** (a `datetime.date`, or ISO text such as
  `"2026-11-14"`). A rule such as "the first Saturday" has no time of day,
  so no time zone or daylight saving can move it; which day "today" is
  belongs to the caller.
- **A period includes both of its ends.** People say "through 31
  December" and mean the day included; `dates("2026-10-11",
  "2026-12-31")` includes both.
- **python-dateutil's `rrule` does the date arithmetic** (the iCalendar
  standard's recurrence, RFC 5545, tested for twenty years), behind
  RePyability's own types, which are what is saved. Two choices differ from
  iCalendar on purpose: a day a short month lacks (the 30th in February)
  falls on its last day by default (`short_months="last"`; `"skip"` is
  iCalendar's), and months or years counted from an anchor keep the
  anchor's day (31 January, 28 February, 31 March).
- **Results are plain data** (`_ResultMapping`s with `to_dict()`, dates as
  ISO text); rules and tasks have `to_dict()` and `from_dict()` with a
  `"kind"` and a `"version"`.
- **`discount=` is the effective yearly rate** (0.07 is 7% a year), the
  one convention of the dated replacement policies
  (`repyability.maintenance`): an amount `y` years after the date it is
  discounted to is worth `(1 + discount)^(-y)`, a year being 365.25
  days; the continuous rate a year is `ln(1 + discount)`. Money is
  discounted to the forecast's first date (the caller's "today"). Every
  present value here goes through `_discount.py`, whose arithmetic is
  the money module's (`repyability.value`): the one convention of
  RePyability, continuous discounting at a rate per unit of the clock
  the money falls on, with a yearly rate converted to it there and
  nowhere else. With no `discount` nothing is discounted and every result is
  as it was.

## The use cases, by number

| # | The question | State |
|---|---|---|
| UC-2 | A weekly task first done on 15 July: how many times through the rest of the year? | **Built**: `Recurrence.every`, `Task.due` |
| UC-3 | What does it cost? | **Built**: `Task(cost=)`, `TaskDue.cost` |
| UC-4 | The first Tuesday of the month, and the 30th | **Built**: `Recurrence.monthly` |
| UC-10 | A pump on weekday shifts, serviced every 500 running hours: when is each service due? | **Built** (every run going ahead): `DutyPlan`, `Usage`, `Task.due(usage=)`; the dates with 90% of runs going ahead not yet |
| UC-28 | The first Tuesday of November every year | **Built**: `Recurrence.yearly` |
| UC-29 | The first, or the fifth, Saturday of each month, and the months without one | **Built**: `Recurrence.monthly(nth=)`, `Recurrence.notes` |
| UC-30 | Two limits, whichever comes first, and a task waiting for the crew's visit | **Built**: `Task(also=, visits=, tolerance=, last_done=)` |
| UC-31 | The work in each month: how many tasks, hours and cost | **Built**: `work_by_period` |
| UC-47 | $250 every 30 days at 7% a year: what is it worth this year, and for ever? | **Built**: `Task.due(discount=)`, `work_by_period(discount=)`, `Task.long_run_value` |
| UC-48 | The wear ring's expected failure costs by month, as present values | **Built**: `failures_by_period(discount=)` (to a stated precision; a window, not the long run) |
| UC-49 | A system's expected costs by month, as present values | **Built**: `events_by_period(cost=True, discount=)` (exact) |
| UC-50 | Is replacing the wear ring at every 6-monthly service worth it, at 7% a year? | **Built**: `task_value(task, Replaces(part), usage, discount=)` |
| UC-54 | Is the 6-monthly trip test worth it, at 7% a year? | **Built**: `task_value` with `Finds(device, demand_rate=, consequence_cost=, repair_cost=)` |
| UC-67 | The gearbox is serviced every 3 months: what is the service worth at 7%, with no effect on its life (the default), as new, or with the fit's restoration factor? | **Built**: `task_value` with `Services(plan, effect=)`: no effect exact; a `Restoration`, or a fitted `Condition`, simulated on common random numbers with a 90% interval |
| UC-57 | The same present values with costs escalating at a yearly rate | **Built**: `escalation=` beside `discount=` (the maintenance README's UC-57; the costs by period stay at today's prices) |
| UC-32 | A pump on weekday shifts: its hours by month, and its 2,000-hour service on dates | **Built**: `DutyPlan`, `UsageForecast` |
| UC-33 | A concrete run: when is the 500-hour check due? | **Built**: `Runs`, a crossing's stops |
| UC-34 | About 3 h a day, or 2 to 4: when are the 250-hour inspections? | **Built**: `Rate`, ranges |
| UC-35 | 250 hours or 12 months at 15 h a month: which limit sets the date? | **Built**: `Task(every=Usage(...), also=Recurrence...)` |
| UC-36 | Exact runs for now, a rate after, with no duty made at the handover | **Built**: `background`, `specific`, `fill` |
| UC-37 | A night shift the nights the clocks change | **Built**: `time_zone=` |
| UC-38 | The wear ring on the weekday pump: expected failures and cost by month, and the range | **Built**: `failures_by_period` |
| UC-39 | A system's events by calendar month | **Built**: `events_by_period` |
| UC-40 | Shuts on dates as opportunities on the part's clock | **Built**: `Opportunity.from_recurrence` |
| UC-41 | The annual overhaul on the first Tuesday of November, as a diagram's shutdown | **Built**: `shutdown_spec` (worked out over a horizon where the blocks worked on are renewed, and simulated; no long run) |
| UC-43 | A site's shuts on real dates, one record for policies and diagrams | **Built** in `repyability.maintenance` (`Shutdowns`; its README's UC-43), laid on the part's clock through this module's rules, plans and forecasts |
| UC-42 | Starts per operating hour for a life on two clocks | **Built**: `UsageForecast.life_at_usage` (needs surpyval 0.28) |

## Use cases

### UC-2: "I've got a weekly task, started 15 July 2026. How many times will it happen through the rest of 2026?"

- **Status:** built: `Recurrence.every(1, "weeks", "2026-07-15")`, and
  `Task(...).due(start, end)` returning a `TaskDue`.
- **Who asks, and why:** anyone with a recurring task on a fixed calendar
  (an inspection, a lubrication round, a test), planning the rest of the
  year.
- **What they know:** the first date (Wednesday 15 July 2026), the
  interval (a week), and the period: from today (Sunday 11 October 2026)
  to 31 December 2026, both days included.
- **The answer, by hand:** **12 times**: every Wednesday from 14 October
  to 30 December (14, 21, 28 Oct; 4, 11, 18, 25 Nov; 2, 9, 16, 23, 30
  Dec). With the 13 done since 15 July, 25 in 2026.
- **How it is solved:** no probability: a calendar task happens on its
  dates whatever the equipment does, so its dates are the first date plus
  whole intervals, kept to the period. Exact.
- **Checks:** the dates by hand; every rule against its definition walked
  a day at a time, on random rules and periods; a weekly task from 1
  January gives 53 in 2026 (52 weeks and a day).

  ```python
  from repyability.schedule import Recurrence, Task

  conveyor = Task("inspect conveyor", every=Recurrence.every(1, "weeks", "2026-07-15"))
  due = conveyor.due("2026-10-11", "2026-12-31")
  due.count     # 12
  due.dates     # 2026-10-14, 2026-10-21, ..., 2026-12-30
  ```

- **Learned:** a task is not a component (nothing fails), so it is its own
  type; dates are new to RePyability and stay at the edge: nothing in the
  model-time methods depends on them.

### UC-3: "How much will this cost?" (the weekly task of UC-2)

- **Status:** built: `Task(cost=)` and `TaskDue.cost`.
- **Who asks:** the same planner, budgeting the rest of the year.
- **What they know:** $250 each time, the same every time.
- **The answer, by hand:** **$3,000**: 12 times at $250.
- **How it is solved:** the cost each time times the count, exact. A cost
  never given is `None`, so it is not taken for a free task.
- **Checks:** the cost is the count times the cost each time, for any
  period; `None` and 0 kept apart.

### UC-4: "What about the first Tuesday of the month? And on the 30th of every month?"

- **Status:** built: `Recurrence.monthly(weekday="Tuesday", nth=1)` and
  `Recurrence.monthly(day=30)`.
- **What they know:** the rule and the period, from 11 October to 31
  December 2026.
- **The answer, by hand:** the first Tuesday: **3 November and 1
  December** (October's, the 6th, is past); the 30th: **30 October, 30
  November and 30 December**.
- **Decisions (the maintainer's, recorded with UC-4):**
  - A short month falls on its last day by default (28 February, or the
    29th in a leap year); `short_months="skip"` leaves it without one.
  - python-dateutil does the arithmetic, behind readable rules. It came
    with every install already (through surpyval's pandas) and is now a
    dependency of its own (2.8.2 or later).
- **Checks:** February under each option, a leap year, the 31st (12 times
  a year, or the 7 long months), and random rules against the calendar
  walked a day at a time.

### UC-10: "My pump runs 06:00 to 22:00 on weekdays and is serviced every 500 running hours. When is each service due?"

- **Status:** built, with every run going ahead: `DutyPlan(Shifts(...))`,
  `Usage(500, "hours")` and `Task.due(usage=)` reproduce the three dates
  (`test_uc10_pump_every_500_hours`). The 90% dates (whole runs lost) are
  not built yet: `utilisation=` counts each planned hour as its share, an
  expected value (see UC-32 to UC-37).
- **Who asks:** a planner whose maintenance is set by use (running hours,
  starts), with an operating schedule.
- **What they know:** the pump runs Monday to Friday 06:00 to 22:00 (16
  hours a run); its meter read 12,340 hours before Monday 12 October
  2026's start; the service is every 500 hours, last done at 12,000.
- **The answer, by hand (every run going ahead):** 12,500 h at **Fri 23
  Oct 22:00** (the stop); 13,000 h at **Tue 8 Dec 10:00** (mid-run);
  13,500 h at **Wed 20 Jan 2027 14:00**. Two in the rest of 2026; at a
  flat 16 hours every day it would have been three (21 Oct, 22 Nov, 23
  Dec), so the schedule matters. With 90% of runs going ahead, the 10% /
  50% / 90% dates of the second are Thu 10 / Mon 14 / Fri 18 Dec, 10:00
  (the nth run to go ahead is negative binomial; a plain simulation of
  200,000 runs agrees).
- **Decisions recorded:** with no time zone, clock times with no daylight
  saving; utilisation is whole runs that go ahead; the forecast reports
  when the meter crosses (and the stops either side).

### UC-28: "Our shutdown overhaul is on the first Tuesday of November, every year. When are the next three?"

- **Status:** built: `Recurrence.yearly(month=11, weekday="Tuesday",
  nth=1)`.
- **Who asks, and why:** a planner booking an annual shut on a weekday
  pattern; a yearly rule kept to the anchor's day of the month cannot say
  it.
- **What they know:** the rule, and today, 14 November 2026.
- **The answer, by hand:** **2 November 2027, 7 November 2028, 6 November
  2029** (1 November 2027 is a Monday; 1 November 2028 a Wednesday, so the
  first Tuesday is the 7th; 1 November 2029 a Thursday, so the 6th). The
  2026 one, 3 November, is past.
- **How it is solved:** dateutil's yearly rule with the month and the
  weekday's position in it (`BYMONTH=11;BYDAY=+1TU`). Exact.
- **Checks:** the dates; the gaps are 364 or 371 days, never a fixed
  period (so routes that need a period take these as given times); the
  rule's `describe()` says "the first Tuesday of November, every year";
  its iCalendar text, read back by dateutil, gives the same dates.

### UC-29: "Our weekend crew works the first Saturday of each month. And the fifth Saturday, when there is one?"

- **Status:** built: `Recurrence.monthly(weekday="Saturday", nth=1)` and
  `nth=5`, with `Recurrence.notes` listing the months without one.
- **Who asks:** a planner whose crew rosters on weekday patterns.
- **What they know:** the rule; from today, 14 November 2026.
- **The answer, by hand:** the first Saturday: **5 December, 2 January, 6
  February, 6 March** (to 31 March 2027; 7 November is past). The fifth
  Saturdays of 2027: **30 January, 29 May, 31 July, 30 October**, four;
  the other **eight months have none** and are skipped (a month has a
  fifth Saturday only when it has 29 or more days and starts close enough
  before a Saturday: 1 January 2027 is a Friday, so January's Saturdays
  are the 2nd, 9th, 16th, 23rd and 30th).
- **Decisions:** a fifth weekday is allowed (it was refused before), and
  the only sensible meaning for a month without one is to skip it; the
  result says how many and which (`notes(...).missing_months`, and a line
  in `TaskDue.notes`). The same for the 31st with `short_months="skip"`.
- **Checks:** the dates; the missing months are exactly the months on the
  rule's cycle with no date, on random rules.

### UC-30: "The oil change is every 12 months or on the 5-monthly service, whichever comes first, and the crew only visits on Tuesdays. When is it done, and how late?"

- **Status:** built: `Task(every=, also=, last_done=, visits=,
  tolerance=, resets=)`.
- **Who asks:** a planner with tasks set by two limits (the usual one is
  "250 hours or 12 months"; the hours limit plugs in with tasks by use),
  and work done only on crew visits.
- **What they know:** each limit; when the task was last done; when the
  crew visits; how many days late still counts as on time.
- **The answer, by hand:**
  - *Two limits:* every 30 days or on the 1st of the month, last done on
    10 January 2027: due 1 Feb (the 1st, before 9 Feb), 1 Mar (before 3
    Mar), 31 Mar (30 days from 1 Mar, before 1 Apr), 1 Apr; the result
    says which limit set each.
  - *Visits:* every 3 months, last done 20 September 2026, visits on
    Tuesdays, 3 days' tolerance: due 20 Dec, done Tue 22 Dec (2 days late,
    on time); counting again from the visit, due 22 Mar 2027, done Tue 23
    Mar (1 day); due 23 Jun, done Tue 29 Jun (6 days, late); due 29 Sep,
    done Tue 5 Oct (6 days, late).
  - *Overdue:* every month, last done 10 March, asked from 1 June: it was
    due on 10 April, so it is overdue, assumed done on 1 June, and then due
    on 1 July and 1 August.
- **Decisions (each said in the result):**
  - A forecast assumes each time is done on its due date, or at the first
    visit on or after it (never early).
  - An interval (`Recurrence.every`) counts again from when the task is
    done; a pattern (weekly, monthly, yearly, given dates) keeps its own
    dates, its next being the first after the task was done. `resets=`
    overrides both, and `TaskDue.resets` says which applied. Done on its
    own due date, an interval keeps its anchor's day (31 January, 28
    February, 31 March), rather than drifting to the 28th.
  - Once a task is done, a resetting limit counts again from then, whichever
    limit set the date.
  - An overdue task appears once, with its first missed due date, done at
    the period's start; the later missed dates are not listed (it is done
    once).
  - A visit that does the task covers any of a fixed rule's dates before
    it (a weekly task with monthly visits is done monthly).
- **Checks:** the dates above; with no last done and no visits, a task's
  dates are its rule's, on random rules; the task through JSON and back;
  a stand-in limit by use through the hook, mixed with a calendar limit.

### UC-31: "How much work is coming each month: how many tasks, how many hours, what cost?"

- **Status:** built: `work_by_period(tasks, period, start, end)` returning
  a `WorkByPeriod`.
- **Who asks:** a planner building next year's labour plan and budget
  (UC-1's table, without the probabilities).
- **What they know:** the tasks, each with its hours and cost where known;
  the periods (day, week, month, quarter, year); the dates.
- **The answer, by hand:** from 16 November 2026 to 15 November 2027, an
  inspection every Monday (1.5 h, $120), a service every 3 months from 20
  December (4 h, $480) and the overhaul on the first Tuesday of November
  (40 h, no cost given): **53 inspections** (3 in the rest of November,
  then 4, 4, 4, 5, 4, 5, 4, 4, 5, 4, 4, and 3 in 1 to 15 November 2027),
  **4 services** (December, March, June, September) and the overhaul on 2
  November 2027. December's work: 4 inspections and a service, 10 hours,
  $960.
- **Decisions:** each task counts in the period it is done in (after its
  visit); the first and last periods are cut to the dates asked; weeks run
  Monday to Sunday; tasks with no hours or no cost are named
  (`no_hours`, `no_cost`) rather than counted as 0. Financial years wait
  for the periods module.
- **Checks:** the counts above; the periods add up to each task's whole;
  the periods cover the dates asked with no gap.

### UC-47: "I have a $250 task every 30 days from today. At 7% a year, what is it worth this year, and if it goes on for ever?"

- **Status:** built: `Task.due(start, end, discount=)` (`TaskDue.npv`,
  `eac`), `work_by_period(..., discount=)` (`WorkByPeriod.npv`,
  `total_npv`, `eac`) and `Task.long_run_value(start, discount)`
  returning a `LongRunValue`.
- **Who asks:** the owner of a maintenance budget comparing a recurring
  task with a one-off spend (a modification that removes the task), who
  has the NPV of replacement policies in model time and asks whether the
  scheduled work ties into it.
- **What they know:** the task, its rule and its cost each time; today's
  date (20 November 2026 here); the yearly discount rate.
- **The answer, by hand:** from 20 November 2026 to 19 November 2027 the
  task is done on days 0, 30, 60, ..., 360: **13 times, $3,250**. Each is
  worth `250 / 1.07^(days / 365.25)` today: 250.00, 248.61, 247.24,
  245.87, 244.50, 243.15, 241.80, 240.46, 239.13, 237.80, 236.49, 235.18
  and 233.87, **$3,144.10** in all. By quarter (the first and last cut to
  the dates): 2, 3, 3, 3 and 2 tasks; $498.61, $737.61, $725.41, $713.42
  and $469.05. Repeating for ever it is a geometric sum, `250 / (1 -
  1.07^(-30 / 365.25))` = `250 / 0.0055417` = **$45,112**. The equivalent
  annual cost of the perpetuity is `45,112 × ln(1.07)` = $3,052 a year
  (a little more than 12.175 tasks a year at $250, $3,044, because each
  interval's task is paid at its start).
- **How it is solved:** the dates a task is done on are known, so a
  period's present value is a finite sum, each cost discounted from the
  date the task is done (after its visit), in the period it is counted
  in. For ever: a plain interval of days or weeks (no second limit,
  visits, season or excluded dates) repeats exactly from the second time
  it is done, so the first is discounted on its own and the rest are the
  geometric sum above (`method="closed form"`). Any other task (the first
  Tuesday of the month, every 3 months, a task waiting for visits, two
  limits) is summed date by date (`method="summed"`): a task is done at
  most once a day, so the dates after day `T` are worth less than `cost ×
  v^(T+1) / (1 - v)` with `v = 1.07^(-1 / 365.25)`, and the sum stops
  where that bound is under `precision` (1e-9 by default) of what is
  summed, reporting the bound. At 7% that is some 360 years of dates. The
  first Tuesday of every month at $250 is worth **$44,350.51** (12 a
  year, the first on 1 December).
- **Decisions (defaults):**
  - `discount` is the yearly effective rate, discounted to `start`.
  - `eac` is the steady cost **a year** over the period (both dates
    included) worth the present value, as the dated policies' `eac`; at a
    discount of 0 it is the cost over the period's years.
  - Tasks with no cost stay named in `no_cost`, and a note says the
    present values leave them out; a task with no cost has no `npv`.
  - **Refused, saying what to run:** a long-run value at `discount=0`
    (the costs add up without end); a task on given dates, or done at
    visits on given dates (they end: `due` to the last date is the whole
    value); a limit by use (`Usage` follows a usage forecast, which
    ends); a rate so low that the sum does not settle to `precision` by
    the year 9999, where dates stop (under about 0.3% a year at 1e-9).
  - A date far from a rule's anchor is now found without walking to it
    from the anchor (`Recurrence._counted_from`): the same dates, in a
    time that no longer grows with the distance.
- **Checks:** `test_uc47_thirty_day_task_this_year_and_for_ever` (the
  numbers above to the cent), `test_long_run_against_100000_dates`,
  `test_a_pattern_for_ever_against_2000_years_of_dates` (the first
  Tuesdays worked out without `Recurrence`),
  `test_no_discount_leaves_results_as_they_were`,
  `test_discount_of_zero_is_the_plain_total`,
  `test_periods_present_values_add_up`,
  `test_long_run_with_visits_and_overdue`,
  `test_what_has_no_long_run_value_is_refused` and
  `test_far_dates_are_the_anchors_own`.

## Duty plans and the date ↔ usage map

These use cases are answered by `DutyPlan`, `UsageForecast` and the
`Usage` limit on a `Task`. Every answer was worked by hand from the
clock times and the calendar before the code, and each is a test in
`test_schedule_usage.py` or `test_schedule_duty.py`. UC-10 (tasks by use
on an operating schedule) is answered by the same code: its three dates
(Fri 23 Oct 22:00, Tue 8 Dec 10:00, Wed 20 Jan 2027 14:00) are
`test_uc10_pump_every_500_hours`.

### UC-32: "My pump runs Monday to Friday, 06:00 to 22:00. Its meter read 12,340 h at the start of Monday 16 November 2026. How many hours will it run each month, and when does its 2,000-hour service fall due?"

- **Status:** built: `DutyPlan(Shifts(...)).forecast(...)`, its `at`,
  `when`, `by_period` and `due`, and `Task(every=Usage(2000, "hours"))`.
- **Who asks, and why:** a planner putting a task counted in running
  hours on the calendar, and budgeting hours by month.
- **What they know:** the shift pattern, one meter reading and when it
  was taken, and the reading when the service was last done (12,000 h).
- **The answer:** 176 h in the rest of November (11 weekdays × 16 h),
  then 368, 336, 320, 368, 352, 336, 352, 352, 352, 352, 336 and 352 h
  (each month's weekdays × 16 h); the meter reads 16,692 h on 1 December
  2027. The service is due at 14,000 h, **Thu 8 April 2027 18:00**
  (1,660 h on: 20 weeks to Mon 5 April, then 16 + 16 + 16 h and 12 h into
  Thursday's run), and at 16,000 h, **Thu 30 September 2027 18:00** (25
  weeks later). The pump stops at 22:00 those days, when the work is
  done. At a flat 80 h a week the first date would show as Saturday 10
  April, a day the pump does not run.
- **How it is solved:** exact. The plan's runs are laid out between real
  instants, and each meter is a running total: straight lines while it
  runs, flat while it doesn't, stepping up at a run's start for
  `"starts"`. A crossing is found by searching the totals; the inverse,
  `model_time` and `moment`, is operating time since the start, the
  model's clock.
- **Checks:** `test_pump_hours_by_date_and_month`,
  `test_pump_service_due_mid_run`, `test_map_both_ways`,
  `test_task_by_use`.

### UC-33: "The generator is booked to run from 12:31 on Saturday to 03:27 on Sunday. Its meter reads 486.2 h. When is its 500-hour check due?"

- **Status:** built: `Runs`, and a crossing's stops.
- **The answer:** the run is 14 h 56 min; the meter reads 501.13 h after
  it. 500 h is 13.8 h (13 h 48 min) into the run: **02:19 on Sunday 15
  November**, with the stop after it at 03:27.
- **Checks:** `test_generator_concrete_run`.

### UC-34: "The conveyor runs about 3 hours a day, between 2 and 4 if I'm honest. The meter read 1,180 h on 1 December 2026 and its idlers are inspected every 250 h, last at 1,000 h. When are the next inspections?"

- **Status:** built: `Rate`, a range of rates, and `earliest`/`latest`.
- **The answer:** at 1,250 h, 70/3 = 23⅓ days on: **24 December 2026
  08:00**; at 1,500 h, 106⅔ days on: **17 March 2027 16:00**; then every
  83⅓ days. At 2 to 4 h a day the first is between **18 December 12:00**
  (4 h a day) and **5 January 2027 00:00** (2 h a day).
- **How it is solved:** each calendar period (here each day) gets the
  rate's amount spread evenly over its real length; a range gives a low,
  a middle and a high map, the middle answering, the others beside it.
  A rate is an assumption, and every answer from it says so (`exact` is
  False).
- **Checks:** `test_conveyor_general_rate_and_range`.

### UC-35: "Our standby generator runs about 15 hours a month on test runs. Its oil change is every 250 hours or 12 months, whichever comes first. Which limit actually sets the date?"

- **Status:** built: a `Task` with `every=Usage(250)` and
  `also=Recurrence.every(12, "months", ...)`; `TaskDue.notes` says when
  a limit set none of the dates, and how far behind it was.
- **The answer:** 250 h at 15 h a month takes 16⅔ months, so **the
  12-month limit sets every date** (14 November 2027, 14 November 2028;
  `limits` says `"also"` for each); the hours would not be reached until
  4 April 2028, about 4.7 months after the first calendar date. The
  hours limit is doing nothing, and the result says so in `notes`: "The
  limit "every 250 hours" set none of the dates: when the task was first
  due (2027-11-14, by the other limit), it would not have been reached
  until 2028-04-04, about 4.7 months later." (A month here is 30.44
  days; a gap under 60 days is given in days; a forecast too short to
  reach it says it would not be reached within the forecast.)
- **Checks:** `test_two_limits_the_calendar_first`, and
  `test_two_limits_the_hours_first` for the reverse on the pump (there
  the 12-month limit sets nothing, 6.3 months behind).

### UC-36: "I know exactly when it runs today and tomorrow; after that, about 3 hours a day. Can the forecast use both, and not count the handover twice?"

- **Status:** built: a plan's `background` and `specific` parts, and
  `fill`.
- **The answer:** "exact runs to 16 Nov 03:30, then about 3 h a day".
  The runs are used where they cover and the background the moment they
  run out. **No duty is made at the handover:** the background applies
  only to the time the specific parts leave. On Monday 16 November the
  runs give 3.5 h to 03:30 and the background 3 × 20.5/24 = 2.5625 h
  for the rest of the day, 6.06 h in all, never 3.5 + 3 h. A shift or
  roster background starts again at its next clock time, counting only
  the part of a run no specific part covers. For a rate that is a
  budget ("15 h a month"), `fill="top_up"` gives the month in which the
  runs end what they left of its 15 h (never less than nothing); the
  months after get 15 h.
- **Rules:** two parts of one layer that would count the same time on a
  meter are refused, naming both; a background part with no end ends
  where the next starts; a specific part covers from its `from_` to its
  `until` (runs: their first start to their last stop by default).
- **Checks:** `test_layers_say_where_the_hours_come_from`,
  `test_pro_rata_handover_adds_no_duty`,
  `test_aligned_handovers_stay_within_the_parts`,
  `test_a_shift_partly_covered_counts_only_its_uncovered_part`,
  `test_handover_across_daylight_saving`,
  `test_top_up_fills_the_budget_and_never_adds`,
  `test_top_up_across_a_month_end`, and
  `test_days_add_up_and_never_pass_24_hours` (every kind of background
  and both fills: the days add up to the change in the reading to 1e-9,
  and no day passes its own length).

### UC-37: "Our Sydney plant runs a night shift, 22:00 to 06:00. How long is it the nights the clocks change?"

- **Status:** built: `time_zone=` on a plan.
- **The answer:** **7 hours** over the night of 3–4 October 2026 (the
  clocks go forward at 02:00) and **9 hours** over 3–4 April 2027 (back
  at 03:00); 8 hours both nights in Brisbane, and with no zone. A run
  starting at 02:30 on 4 October 2026, a time that does not exist,
  starts at 03:00, and the notes say so; a time that happens twice means
  the first.
- **How it is solved:** wall-clock times at the site are turned into
  real instants (the standard library's `zoneinfo`), and every hour is
  counted between real instants; answers come back as the site's
  wall-clock times with the zone. With no zone, clock times are taken as
  they are.
- **Checks:** `test_a_night_shift_across_daylight_saving`,
  `test_times_that_do_not_exist_or_happen_twice`.

## Failures, events and shuts on the calendar

These use cases ask RePyability's model-time methods from dates and label
their answers with dates (`_forecast.py`). **No probability is new
here**: each turns dates into model time (operating time through the
`UsageForecast`, or calendar time since a start), calls an existing
method, and puts the dates back. Ranges are the 5th to 95th percentile:
90% of outcomes, two-sided, stated on every result. Each answer is a test
in `test_schedule_forecast.py`.

### UC-38: "The pump's wear ring has a fitted life in operating hours. On the pump's weekday shifts, how many failures should I expect each month, what will they cost, and how far could that be out?"

- **Status:** built: `failures_by_period(part, usage, period, age=)`
  returning a `FailuresByPeriod`.
- **Who asks:** a planner or reliability engineer budgeting one position
  by month (and, summed over many positions, a site).
- **What they know:** the pump of UC-32 (Mon–Fri 06:00–22:00, from Monday
  16 November 2026, forecast to 1 December 2027); the ring's life
  (Weibull, scale 9,000 h, shape 2.5) and its age now (5,200 h); it is
  replaced when it fails, at $4,800.
- **The answer, by hand:** the operating hours by month are UC-32's (176,
  368, 336, 320, 368, 352, 336, 352, 352, 352, 352, 336, 352; 4,352 h in
  all). The chance the ring fails by 1 December 2027 is
  `1 - R(5,200 + 4,352) / R(5,200)`, with `R(t) = exp(-(t / 9,000)^2.5)`:
  (5,200 / 9,000)^2.5 = 0.25375 and (9,552 / 9,000)^2.5 = 1.16046, so
  `1 - exp(-0.90671)` = **59.6%**. Counting the replacement ring's own
  failures too (a renewal count, `NonRepairable.failures`), **0.622
  failures are expected, $2,988**. By month: 0.022, 0.047, 0.045, 0.045,
  0.053, 0.051, 0.050, 0.053, 0.053, 0.053, 0.052, 0.049, 0.051 failures
  ($105, $227, $217, $214, $252, $247, $239, $252, $253, $252, $250, $236,
  $244). The chance of no failure is 40.4%, of one 57.0%, of two or more
  2.6%, so the range over the whole span is **0 to 1 failure, $0 to
  $4,800** (90% of outcomes). In a single month the 95th percentile is
  $4,800 where that month's chance of a failure is 5% or more, else $0.
- **How it is solved:** each period's end becomes operating time since
  the forecast's start (`usage.model_time`); the chance of at least one
  failure by each end is `NonRepairable.failures`' at those times (exact,
  run to failure); each period's expected failures, cost and range, and
  any policy (`every=` with `resets=`, or a `calendar=` from UC-40), come
  from `NonRepairable.cost_by_window` (numerical, to about 1e-6), which
  works each window out through the earlier ones (a failure in one renews
  the ring for the next). Periods with no running (a weekend, by day)
  have nothing in them. A part with no cost, or one that takes time to
  replace (which `cost_by_window` refuses), gets `NonRepairable.failures`'
  exact expected failures and no ranges, with the method's own reason in
  the notes; for the second, the dates take the replacement as taking no
  calendar time.
- **Decisions (defaults):**
  - The part's `age` is at the forecast's start.
  - The model's unit is read from its label (surpyval's `unit`: `"h"`,
    `"hrs"`, `"days"`, ...), else given as `unit=`, else hours, with a
    note. A label that is not a unit of time (`"km"`) is refused, as is
    a `unit=` that contradicts the label.
  - The range is fixed at the 5th to 95th percentile (90%, two-sided),
    in counts and in cost.
- **Checks:** `test_uc38_wear_ring_by_month` (the numbers above, each
  month against `NonRepairable.failures` at the operating-hour ends, the
  months adding up to the whole), `test_uc38_against_a_plain_simulation`
  (200,000 rings renewed plainly),
  `test_periods_with_no_running_have_nothing`,
  `test_the_unit_is_read_from_the_model_or_given` (the same answers with
  the life in days), `test_a_policy_by_period` (age replacement against
  `cost_distribution`), and
  `test_a_part_with_no_cost_or_a_replacement_taking_time`.

### UC-39: "What is coming on this system each calendar month?"

- **Status:** built: `events_by_period(rbd, usage, period, state=)`
  returning an `EventsByPeriod`.
- **Who asks:** UC-1's planner, who asked month by month and was given
  windows of 730 hours.
- **What they know:** a `RepairableRBD` and its state now; the months.
- **The answer, by hand:** a pump whose failures come at a constant rate
  (one per 2,000 h), repaired at once: **0.372 failures in January 2027**
  (744 h), 0.336 in February (672 h), 4.38 in the year (8,760 h). In
  Sydney, October 2026 has **743 hours** (the clocks go forward on the
  4th): 0.3715. UC-1's diagram (pump and valve) by the months of 2027
  adds up to `expected_events(8760)`, and the valve's block replacement,
  1,760 h on, falls in March (1,416 h to the end of February, 2,160 h to
  the end of March).
- **How it is solved:** a diagram's clock is calendar time (blocks'
  `"duty"` and `"basis"` put running hours on it), so each period's end
  is the real time from the forecast's start, in the diagram's unit;
  `expected_events_by_window` answers (exact), and `cost=True` adds
  `cost_by_window` (each period's mean exact, its range simulated, said
  in the notes).
- **Decisions (defaults):** the diagram's unit is read from its models'
  label where the diagram carries one, else `unit=`, else hours with a
  note; cost is off by default (its range is simulated).
- **Checks:** `test_uc39_constant_rate_by_calendar_month`,
  `test_calendar_months_in_a_zone_are_real_hours`,
  `test_cost_by_month_is_exact_on_average`,
  `test_uc1_diagram_by_calendar_month_sums_to_the_year`.

### UC-48: "What are the wear ring's expected failure costs worth today, month by month?"

- **Status:** built: `failures_by_period(..., discount=, npv_tolerance=)`
  (`FailuresByPeriod.expected_npv`, `npv`, `npv_precision`, `eac`).
- **Who asks:** UC-38's planner, whose budget is in present values.
- **What they know:** UC-38's ring and pump; 7% a year.
- **The answer, by hand:** UC-38's months cost $105, $227, $217, $214,
  $252, $247, $239, $252, $253, $252, $250, $236 and $244, $2,988 in
  all. A month's cost falls somewhere inside it, so it is worth at most
  its cost discounted from the month's start and at least from its end
  (the months end 0.041, 0.126, 0.211, 0.287, 0.372, 0.455, 0.539,
  0.622, 0.706, 0.791, 0.873, 0.958 and 1.040 years on): **between
  $2,874.75 and $2,890.71**, so $2,882.73 give or take $7.98. Each
  failure's cost discounted from when it is expected to fall (the
  integral below), the value is **$2,882.74**, to within $0.016 at the
  default tolerance (and $2,882.741 to within $0.0007 at
  `npv_tolerance=1e-5`); by month $104.40, $226.20, $214.53, $210.05,
  $246.83, $239.74, $230.71, $242.24, $241.48, $239.51, $236.52,
  $222.18 and $228.36. The equivalent annual cost over the 1.04 years
  is $2,869.51. (Until #604, short windows each discounted at their
  middle gave $2,882.77 give or take $2.47, and an EAC of $2,869.53.)
- **How it is solved:** an expected cost accrues as the part runs, so
  discounting it needs the dates inside each period. With `C(s)` the
  expected cost by the calendar time `s` (the part's expected cost to
  the operating time the duty plan has run by `s`), the present value
  to `T` is `exp(-r T) C(T) + r ∫ exp(-r s) C(s) ds`, each cost
  discounted from when it is expected to fall. It is the diagram's own
  discounting (`_common._discounted_accrual`, the one design of an
  accrued cost's present value): the integral by the Gauss-Kronrod (7,
  15) rule on pieces from the start (32 equal ones, cut again at the
  periods' ends), the pieces whose two rules differ most halved until
  the differences, times `r`, are within `npv_tolerance` of the value.
  `npv_precision` is that error estimate plus what the error of `C`
  itself can move the value: run to failure, `failures`' count is
  worked out at a grid's points and taken linearly between them, off
  by at most an eighth of the grid's largest second difference, and at
  most that times `exp(-r T) + |1 - exp(-r T)|` in the present value.
  `C` comes from `failures` run to failure, and from `cost_by_window`'s means under a policy
  (every policy it takes, with no new renewal maths). Days with no
  running add nothing.
- **Decisions (defaults):**
  - `npv_tolerance` is 1e-3 (ten cents in a hundred dollars). The
    rules' difference is a cautious estimate: the ring's value at 1e-3
    is within $0.002 of the one at 1e-5. A duty plan's shifts bend `C`
    at every start and stop, which the halving closes in on; at most
    60 rounds are worked out, and a forecast that falls short of the
    tolerance says so in its notes, with what it reached.
  - The ranges (5th to 95th percentile) stay undiscounted, and the notes
    say so: a discounted distribution of a part's cost is a later item.
  - At `discount=0` the present values are the expected costs
    themselves, with a precision of 0.
  - A part with no cost has no present value (a note); a part whose
    replacement takes time (run to failure with `NonRepairable.failures`)
    is discounted the same way from `failures`' expected counts.
  - A window only: the value of the part for ever on dates is not built
    (see below).
- **Checks:** `test_uc48_wear_ring_present_value_by_month` (the numbers
  above, inside the hand bracket by month),
  `test_constant_rate_in_calendar_time_against_the_closed_form` (rate ×
  cost × the integral of the discount factor, inside the bracket),
  `test_the_bracket_narrows_and_holds_a_simulation` (tolerances from 5e-3 to 1e-4;
  400,000 seeded rings on the duty plan, within four standard errors
  and the bracket), `test_nothing_accrues_while_idle`,
  `test_a_policy_as_a_present_value`,
  `test_a_part_with_no_cost_has_no_present_value`,
  `test_a_forecast_short_of_the_tolerance_says_so`.

### UC-49: "And the system's expected costs, month by month, as present values?"

- **Status:** built: `events_by_period(..., cost=True, discount=)`
  (`EventsByPeriod.expected_npv`, `npv`, `eac`).
- **Who asks:** UC-39's planner.
- **What they know:** UC-39's pump (a failure per 2,000 h on average,
  repaired at once), $1,000 a repair; the months of 2027; 7% a year.
- **The answer, by hand:** costs come at $0.50 an hour, so a span to `T`
  hours is worth `0.5 × (1 - e^(-r T)) / r` with `r = ln(1.07) / 8,766`
  an hour: January (744 h) **$370.93**, to the end of February (1,416 h)
  $704.15, so February $333.21, and the year (8,760 h) **$4,235.21**
  against $4,380 undiscounted. The equivalent annual cost is $4,383 a
  year of 365.25 days: the cost rate itself, as it should be for costs
  at a constant rate.
- **How it is solved:** a diagram's clock is calendar time, so the
  yearly rate is one continuous rate per unit of it (`ln(1 + discount)`
  over the units in a year of 365.25 days), which
  `RepairableRBD.expected_cost(t, discount_rate=)` takes: each period's
  present value is its difference at the period's ends. Exact, as
  `expected_cost` is, with each event discounted at its own time (a
  block replacement at its instant, not the middle of a month).
- **Decisions (defaults):** `discount=` needs `cost=True` (refused
  otherwise, saying so); the range stays undiscounted (a diagram's
  discounted spread is `npv_distribution`'s, in model time); an
  acquisition cost, paid at the start, is not in the periods, as it is
  not in `cost_by_window`'s.
- **Checks:** `test_uc49_constant_rate_by_month_as_present_values`,
  `test_events_present_value_is_expected_npvs_cost`,
  `test_events_discount_needs_cost`.

### UC-50: "Is replacing the wear ring at every 6-monthly service worth it, at 7% a year?"

- **Status:** built: `task_value(task, Replaces(part), usage, age=,
  discount=, period=)` returning a `TaskValue`.
- **Who asks:** UC-48's planner, who has the ring's failure costs as
  present values and a task that costs money each time: does the task pay
  for itself?
- **What they know:** UC-38's ring (Weibull, scale 9,000 operating hours,
  shape 2.5), 5,200 h old on Monday 16 November 2026, $4,800 when it
  fails; the pump of UC-32 (Mon–Fri 06:00–22:00, forecast to 1 December
  2027); the 6-monthly service (`Recurrence.every(6, "months",
  "2026-12-01")`), at which a new ring costs $600; 7% a year.
- **The answer, by hand:** the service is done on 1 December 2026 and 1
  June 2027 (1 December 2027 is the forecast's end, not in it). At
  midnight those are **176 h and 2,256 h** of running from the start
  (UC-32's months), and the forecast is 4,352 h. *With the task* the ring
  is new at each, whatever its age, and a failure in between is replaced
  too without moving the dates: the old ring runs 176 h from 5,200 h
  (renewal count 0.02178), a new one 2,080 h (0.02541) and another 2,096
  h (0.02590), **0.0731 failures, $350.82**, and two planned rings,
  $1,200: **$1,550.82**. *Without the task* (run to failure from the same
  age over the same dates) it is UC-38's **0.6224 failures, $2,987.56**.
  The task saves **$1,436.74** and avoids **0.549 failures**. By month,
  with the task: 0.0218 failures in November (the old ring's 176 h), then
  0.0003, 0.0014, 0.0026, 0.0050, 0.0070 and 0.0090 (the new ring ageing),
  0.0003 again in June, ... At 7% a year, the planned rings are worth
  $600 × 1.07^(-15/365.25) + $600 × 1.07^(-197/365.25) = $598.34 +
  $578.50 = **$1,176.83**. Each month's failure cost discounted at its
  start and at its end brackets the rest, as in UC-48: with the task the
  failures are worth $339.39 to $341.00, so **$1,516.22 to $1,517.84** in
  all; without, UC-48's **$2,874.75 to $2,890.71**; the task's present
  value is **between $1,356.92 and $1,374.49**. The service pays, mostly
  by putting a new ring in for an old one at once.
- **How it is solved:** the task's dates come from `Task.due` over the
  forecast; each is done at midnight of its date (a limit by use: when the
  meter reaches its reading), which `usage.model_time` puts on the part's
  clock. Between two of them the ring is a renewal process from new (from
  `age` before the first), so its expected failures to any time are a
  sum of `NonRepairable.failures` over the stretches (its renewal
  count, to about 1e-6; no new renewal maths). Without the task they are `failures` from `age`. Each
  side's failure cost is discounted as UC-48's (from when it is
  expected to fall, the integral's error estimate the precision); a
  planned replacement is discounted from its own instant.
  `value = without − with`, by period and in all.
- **Decisions (defaults):**
  - **Block replacement on dates:** a failure between two services does
    not move the next; the task's dates are its own (`Task.due`, with its
    visits, second limit and `resets`).
  - **"Without" is run to failure** from the same age over the same
    dates, not the best other policy (`NonRepairable.renewal_forecast`
    and the optimal policies answer that in model time).
  - **One price:** the task's `cost` is the planned ring's. A task with
    no cost takes the part's `planned_cost`; given both, they must agree,
    or it refuses naming the two numbers. The part needs an
    `unplanned_cost` (without it the task saves nothing to set against its
    cost).
  - A planned replacement takes no time on the part's clock (its
    `time_to_replace` is a failure's, which `failures` counts).
  - Only a replacement is valued here (`Replaces`); a test of a hidden
    failure is UC-54's (`Finds`), and a service UC-67's (`Services`).
  - The `calendar=` of `cost_by_window` is an optimised policy's
    thresholds, so it does not say "replace at every date, whatever the
    age"; `every=` says it only for even gaps in model time, which
    months on a duty plan are not. The stretches between the dates say
    it for any dates.
- **Checks:** `test_uc50_wear_ring_six_monthly_service` (the numbers
  above, inside the hand bracket by month),
  `test_even_dates_match_renewal_forecast` (a discount of 0 against
  `renewal_forecast`'s plan and run-to-failure costs),
  `test_constant_rate_task_only_costs` (exponential: the value is minus
  the task's own present value), `test_wear_out_against_a_simulation`
  (both plans simulated from a seed, within four standard errors),
  `test_periods_add_up`, `test_one_price`,
  `test_what_is_not_a_replacement_is_refused`,
  `test_a_task_by_use_matches_block_replacement_in_model_time`.

### UC-54: "Is the 6-monthly trip test worth it, at 7% a year?"

- **Status:** built: `task_value(task, Finds(device, demand_rate=,
  consequence_cost=, repair_cost=), usage, age=, discount=, period=)`
  returning a `TaskValue`.
- **Who asks:** UC-50's planner, about a task that replaces nothing: a
  test of a protective device (a trip, a relief valve, a standby pump's
  start). A test prevents no failure; what it is worth is the time the
  device would otherwise sit failed with nobody knowing.
- **What they know:** a trip failing hidden at a constant 0.1 a year,
  working on Monday 16 November 2026; a test on 1 January and 1 July
  (`Recurrence.every(6, "months", "2027-01-01")`), $400 each; a failure
  repaired as new for $3,000; the protected event once in 2 years (0.5 a
  year), which costs $250,000 if it meets the trip failed and shows the
  failure, repaired then; three years, to 16 November 2029; 7% a year.
- **The answer, by hand:** the trip's life is in calendar time, so its
  clock is the calendar's: the six tests are 46, 227, 411, 593, 777 and
  958 days from the start (0.126, 0.621, 1.125, 1.624, 2.127 and 2.623
  years), and the forecast 1,096 days (3.0007 years). As UC-53 of the
  maintenance README: from a test (or the start) the chance the trip is
  failed after `s` years is `0.1 / 0.6 (1 - exp(-0.6 s))`. *With the
  task* the tests find it failed with chances 0.0121, 0.0429, 0.0435,
  0.0431, 0.0435 and 0.0429 (**0.2279 failures found**); it is failed
  for 0.0640 years (a mean unavailability of **0.0213**), so **0.0320
  demands** meet it failed: tests $2,400, repairs $3,000 x (0.2279 +
  0.0320) = $779.60, consequences $250,000 x 0.0320 = $7,995.30:
  **$11,174.90**. *Without the task* (never tested; a failure stays
  until a demand shows it) it is failed for 0.2682 years (**0.0894**),
  **0.1341 demands** meet it failed: $253,000 x 0.1341 = **$33,931.58**.
  The tests save **$22,756.68**. At 7% a year (the calendar's clock is
  the trip's, so the closed form discounts exactly, at `ln 1.07` a
  year): the tests are worth $400 x sum 1.07^(-t) = $2,190.54, with the
  task **$10,087.23**, without **$29,936.76**: the task's present value
  is **$19,849.53**.
- **How it is solved:** the task's dates come from `Task.due` and are
  laid on the device's clock as UC-50's are. With them as the tests'
  times, the device's chance of being failed over time is UC-53's
  (`maintenance._finding_window`), from `age`, working at the start;
  without the task it is the same with no tests. Each side's demands on
  a failed device accrue with the device's operating time (the demand
  rate times the failed time), so they go by period and are discounted
  exactly as UC-50's failures are (`_accrued_npv`'s integral, its error
  estimate in `precision`, with the grid's own error);
  a test's cost and the repair of what it finds are discounted from the
  instant the task is done. `value = without - with`. The result is
  UC-50's `TaskValue`: its "failures" are the demands that meet the
  device failed, and it adds `found` (the failures the tests find) and
  `unavailability_with` and `unavailability_without` (the device's mean
  unavailability in each period, and over the forecast in `total`).
- **Decisions (defaults):**
  - **A demand that meets the device failed shows the failure**, which
    is repaired then at `repair_cost`, with the task and without it
    (`demands_reveal=False` on `Finds` turns it off on both sides: only
    a test finds a failure, as `failure_finding` has it, and never
    tested a failure is for good).
  - **Tests and repairs take no time and a test never misses**, as
    `failure_finding` assumes.
  - **The caller gives the demand rate and the consequence's cost**; a
    `Task` does not carry them, and a diagram's system-level consequence
    is not worked out here. The demand rate is per unit of the device's
    own time, so demands come only while it runs.
  - The task's `cost` is one test's; a task with no cost is refused
    (0.0 for a free test). The device's own costs are not used.
  - Tests at one time on the device's clock (two dates with no running
    between them) are one test of the device: each costs, and the first
    finds what there is to find.
  - "Without" is never tested, not the best other interval
    (`optimal_failure_finding_interval` answers that in the long run).
- **Checks** (`test_schedule_finds.py`):
  `test_uc54_six_monthly_trip_test` (the numbers above, undiscounted to
  1e-9 and discounted within `precision`),
  `test_only_tests_find_a_failure`,
  `test_matches_the_model_time_calculation` (a Weibull device 1,000 h
  old tested every 30 days against `failure_finding_window`, with and
  without a discount), `test_uneven_dates_under_a_duty_plan` (the first
  Tuesday of each month on a pump on weekday shifts, against the closed
  form on the pump's clock, each day's cost discounted at its start and
  its end by hand), `test_tests_while_the_device_does_not_run`,
  `test_the_value_follows_the_prices` (rising with the consequence,
  falling with the test's cost, zero at the break-even price),
  `test_periods_add_up`, `test_refusals`.

### UC-67: "The gearbox is serviced every 3 months, and the fit says a service takes off 40% of its age. What is the service worth at 7%?"

- **Status:** built: `task_value(task, Services(plan, effect=), usage,
  age=, discount=, period=, mc_samples=, seed=, confidence=)` returning
  a `TaskValue` whose `method` says `"exact"` or `"simulated"`.
- **Who asks:** UC-50's planner, about a task that neither replaces
  nor tests: a service (lubricate, adjust, clean, overhaul). Most of a
  plan's recurring tasks are services, and what a service does to the
  failure rate is not known from the task.
- **What they know:** the gearbox's plan (`UnitPlan`): its life
  (Weibull, scale 8,760 h, shape 3), each failure repaired as new in no
  time at 12,000; it runs around the clock (`UsageForecast.calendar`),
  4,000 h old on 1 January 2027; the service every 3 months from 1 April
  (`Recurrence.every(3, "months", "2027-04-01")`), 800 each; a year, to
  1 January 2028; 7% a year. What a service does, if they say: the fit
  of the gearbox's history (surpyval's `GeneralizedRenewal`) gives a
  Kijima II factor of 0.6 (a service leaves 60% of its virtual age).
- **What they want back:** the service's value against not doing it,
  by quarter and in all, and how sure that is.
- **Decisions (the owner's, 21 Nov 2026):**
  - **By default a service has no effect on the life**: it is an
    accepted cost. RePyability never invents a benefit. Its value is
    then minus the present value of its own cost, exactly, and the
    answer says the effect was taken as none.
  - A benefit is credited only through an effect the caller states:
    a **restoration factor** (`maintenance.Restoration`, Kijima I or
    II, the factor from 0 up: 0 is as new under Kijima II, 1 no effect,
    above 1 a service that does harm, UC-62), a virtual age to restore
    to, a fitted surpyval process, or a **fitted condition**
    (`Condition(life, fresh=, stale=, lasts=)`: a regression life fitted
    with a "serviced or overdue" covariate, at `fresh` for `lasts` after
    each service and at `stale` otherwise, as a covariate schedule on
    the diagram's clock, UC-41's `ScheduledLife`).
- **The answer, by hand:** the services are on 1 April, 1 July and 1
  October, 2,160, 4,344 and 6,552 h from the start (the forecast is
  8,760 h). Their present value is 800 × (1.07^(-90/365.25) +
  1.07^(-181/365.25) + 1.07^(-273/365.25)) = 786.77 + 773.62 + 760.55 =
  **2,320.95** (2,400 undiscounted).
  - *No effect (the default):* the value is **-2,320.95** (-786.77,
    -773.62 and -760.55 in the second, third and fourth quarters, 0 in
    the first), exactly. A factor of 1 (Kijima I or II) leaves the
    gearbox as old as it was, which is the same statement: -2,320.95,
    exactly.
  - *As new (Kijima II, factor 0):* each service renews the gearbox,
    and a failure in between is renewed too, so the plan is UC-50's
    block replacement on the dates: from `NonRepairable.failures`, 0.271
    failures with the service against 1.149 without (renewal from 4,000
    h), 0.878 avoided, a value of **8,135.54** undiscounted (with
    5,652.28 against without 13,787.83) and **7,794.48** at 7% (within
    15.81, UC-50's bracket).
  - *Factor 0.6:* no closed form: the simulation's value lies between
    the two (checked: the value falls as the factor rises).
- **How it is solved:** the service's dates come from `Task.due` and
  are laid on the diagram's clock (real time from the forecast's start,
  in the plan's unit), as UC-50's are. With no effect, the value is the
  task's cost at each instant, discounted through `repyability.value`,
  and nothing else is worked out: the unit's own costs are the same with
  and without the service and are left out of both sides. With an
  effect, the plan's one-node diagram (`UnitPlan.diagram`, on a
  `CalendarContext` giving the node its duty) runs twice from one seed:
  with the service as a task of no duration at the dated times
  (`BlockPolicy(times=, restoration=)`, or the covariate schedule its
  dates give) and without it. The node is named alike in both, so it
  draws the same random numbers in both (`compare_npv`'s common random
  numbers) and each run's difference comes from the service. Each run's
  history is priced with the plan's own prices by
  `_npv_runs.priced_chunk` (the money module's `Ledger`), by calendar
  period, discounted at the yearly rate per unit of the diagram's
  clock; the service's own cost is the exact one of the default, added
  to the with side. The value by period and in all is the mean
  difference, with a two-sided interval (90% by default) from the runs'
  spread.
- **As JSON:** `value_plan`'s `"task_value"` analysis takes the task
  (`Task.to_dict()`), the forecast (its start, end and duty plan), the
  effect as plain data (none, `{"kind": "restoration", ...}` or
  `{"kind": "condition", ...}`, `schedule._service.effect_to_dict`) and
  the unit's plan as the diagram, so a caller that records the choice of
  effect per failure mode (RCM's) can send it as it is stored. A
  restoration by a fitted process has no plain form: its Kijima model
  and factor are sent instead. `test_value_plan_services` checks each
  answer equals `task_value` called directly.
- **Decisions (defaults):**
  - A service takes no time, as UC-50's replacement and UC-54's test.
  - A service due at the forecast's start is done then: its effect is
    the start's (the gearbox's age taken as having had no action since
    new, so a Kijima factor `q` leaves `q × age`).
  - The plan holds no preventive task of its own (it holds one at most,
    which the service would be); a plan that does is refused, saying to
    value the service on the plan without it.
  - 4,000 runs by default; a seed is drawn and kept when none is given.
  - What the engine does not follow is refused with its own reason: a
    virtual age on a duty plan that changes over time, a covariate
    schedule with a duty, a state with imperfect repair.
- **Checks** (`test_schedule_services.py`): the default against the hand
  numbers to the cent; factor 1 equal to the default exactly; as new
  (factor 0) against UC-50's `Replaces` values within four standard
  errors (bound fixed before the run); as new with minimal repairs
  from new against the overhaul's closed form, the failures with
  services every `T` being `H(T)` a stretch; seeded runs repeat; the
  interval narrows with more runs; the value falls as the factor rises
  (0, 0.6, 1.2); an exponential life gains nothing at any factor (the
  interval holds minus the task's cost); a fitted condition's value
  follows its covariate's effect; refusals.
- **Learned:** valuing a service needed no new engine: a dated task
  with a restoration was already a plan's task, and the runs' pricing
  already the money module's. The default of no effect is the honest
  one: it states that the service is a cost until someone says what it
  does.

### UC-40: "Replacing the part is cheap at the annual overhaul, the first Tuesday of November. Can the calendar policy use the real dates?"

- **Status:** built: `Opportunity.from_recurrence(name, cost, rule,
  start, end, unit=, usage=, at=, time_zone=)`.
- **Who asks:** UC-13's engineer, whose shuts are on dates, not periods.
- **The answer, by hand:** for a life in days, from 14 November 2026:
  **353, 724 and 1,088 days** (2 November 2027, 7 November 2028, 6
  November 2029: 353 days, then 371 across a leap year, then 364). For
  the weekday pump's ring, in operating hours from 16 November 2026: the
  overhaul at 00:00 on 2 November 2027 is at **4,016 h** (4,000 h to the
  end of October and Monday 1 November's 16 h). Shuts on Saturday and
  Sunday fall at one operating time (80 h, 160 h, ...) and are one
  opportunity each weekend.
- **How it is solved:** the rule's dates, at a clock time (midnight by
  default), become `usage.model_time` (operating time) or the real time
  since `start` (calendar time): the opportunity's `dates`. The policy
  maths is untouched; "the first Tuesday of November" has no fixed
  period, so it goes in as given times.
- **Checks:** `test_uc40_opportunity_in_calendar_days`,
  `test_uc40_opportunity_in_operating_hours`.

### UC-41: "Put our annual overhaul (the first Tuesday of November, 48 hours from 06:00) on the system's diagram."

- **Status:** built: `shutdown_spec(rule, start, end, duration=, unit=,
  at=, time_zone=, cost=, scope=, idle_ageing=)`, the spec
  `RepairableRBD(shutdowns={name: spec})` takes.
- **The answer, by hand:** with time 0 at 14 November 2026: **8,478,
  17,382 and 26,118 hours** (353, 724 and 1,088 days, each plus 6 h). In
  Sydney the first is the same 8,478 h: both ends are in daylight saving.
- **How it is solved:** as UC-40, on the diagram's clock (calendar
  time), as the shutdown's `"times"`.
- **Over a horizon:** `expected_events`, `expected_cost` and
  `expected_npv` (and so `events_by_period`) take shuts at given times
  where the blocks worked on in them are renewed; the simulations
  (`cost()`, `availability()`) take any. They end, so the long-run
  methods refuse such a diagram with a message that says so.
- **Checks:** `test_uc41_shutdown_spec_on_dates`,
  `test_uc41_a_diagram_takes_the_dated_shuts` (two shuts in two years,
  $40,000, simulated).

### UC-42: "The pump's life was fitted on two clocks, running hours and starts. What rate do I give it?"

- **Status:** built: `UsageForecast.life_at_usage(model, meter="starts",
  per="hours")`, which hands `rate(meter, per=per)` to the model's
  `at_usage` (surpyval 0.28's `EquivalenceModel`; with an earlier
  surpyval it refuses, saying so). The block that takes the life is
  separate work.
- **The answer, by hand:** on the weekday pump, one start a run of 16 h:
  **0.0625 starts an operating hour**. For a life of Weibull scale 2,000,
  shape 2 on the combined clock `T = hours + 10 starts`, the life in
  hours is that of `T / 1.625`: Weibull scale 1,230.8 h, shape 2, mean
  1,090.7 h.
- **Checks:** `test_uc42_rate_for_a_life_on_two_clocks` (a stand-in
  model); `test_uc42_with_surpyval_equivalence`, skipped before surpyval
  0.28 (run by hand against surpyval's `dev`: it passes).

### UC-64: "Our line has two pumps in parallel: one runs weekday shifts, the other around the clock. The first has its seal changed every 4,000 running hours, the second at a check on the first Monday of each quarter. What failures, services and costs come each month?"

- **Status:** built: `CalendarContext(usage, duty={node: forecast or
  share}, unit=)`, taken by `events_by_period`; a block's `"duty"`
  takes a share or a `DutySchedule` (`UsageForecast.schedule()`); a
  task on a rule is replaced at `CalendarContext.rule_times(rule, at)`
  (its `"replace_at"`).
- **Who asks:** UC-39's planner, whose pumps do not run alike: one on
  Monday to Friday 06:00 to 22:00 (80 h a week), one 24/7.
- **What they know:** one diagram (the two pumps, each life in operating
  hours, Weibull scale 3,000 h and shape 2, repairs of about 8 h); the
  shift pump's 4-hour seal change every 4,000 operating hours, the 24/7
  pump's at the check on the first Monday of each quarter (06:00); each
  pump's duty plan; the months of 2027.
- **The answer, by hand:** each pump runs its own clock `D(t)`, the
  operating hours from 1 January to `t`. The shift pump runs 4,176 h in
  2027 and reaches 4,000 h at the end of the shift on **Thursday 16
  December, 22:00** (`usage.moment(4000)`), so its one seal change falls
  in December (the plan's hours, whatever repairs the pump has: the
  service is on the plan). The 24/7 pump's seal
  is changed on **4 January, 5 April, 5 July and 4 October** at 06:00
  (the rule's dates). With repairs in no time, a pump is its operating-time
  process seen through `D`: its expected failures by `t` are `M(D(t))`,
  `M` the renewal count (with its block service) in operating time, from
  the one-node diagram with no duty; the simulation on the calendar
  agrees within its error (`test_a_unit_is_its_operating_process_seen_through_its_plan`).
- **How it is solved:** the engine always runs on calendar time. The
  analysis, not the plan, says each unit's duty: `CalendarContext.diagram`
  gives each node its plan as its `"duty"`. A plan of one share is the
  constant share it always was (the model rescaled, exact). A plan that
  changes is a life on the diagram's clock (`_duty.PlannedLife`): a unit
  put in at `t0` fails at `D⁻¹(D(t0) + L)`, `L` drawn from the model's own
  quantile function in operating hours, and a task counted in operating
  time falls when `D` reaches it (`_Preventive.clock`). Its reliability
  from new, `R(D(t))`, is the model's own `sf`; a unit put in later has
  another, so it is no renewal process, and the Python simulation counts
  it (the compiled engine hands it over). `events_by_period` then counts
  each month's events in the histories and takes the cost's mean and
  range from the same runs, each cost discounted from when it falls.
- **Decisions (defaults):** a plan belongs to the analysis
  (`CalendarContext.duty`), not to the unit's plan; a task in operating
  time on a block with no duty counts the diagram's clock as its
  operating time (as a model without a calendar runs from 0); a plan
  holds its last share past its end (the calendar refuses a forecast
  that ends first); on a plan that changes, a unit starts new (a state
  other than new, a virtual age, replacement on condition, tests counted
  in operating time and standby groups are refused, saying so).
- **Checks:** `test_duty_plans.py`: a plan of one share is the duty share
  to the bit; the simulation against `M(D(t))` for run to failure, block
  and age replacement; a usage task's dates equal the forecast's
  `moment`s; a rule's dates; saving; the refusals. A rule as a spec of
  its own (resolved by the calendar) is not built: the diagram must
  stand without a calendar, and a rule's dates need one.

### What is not built yet

- **A part's expected failure costs for ever, on dates.**
  `failures_by_period(discount=)` values a forecast's window. "The dated
  window, then the discounted long run" (as `maintenance/_dated.py` does
  for the policies on dates, with `discounted_run_to_failure` and
  `age_replacement_value` where the usage keeps a steady pace) is not
  joined to it yet; a task's long run is (`Task.long_run_value`).
- **A service on what the engine does not follow yet.** UC-67's
  simulated effects keep the engine's limits: a virtual age is not
  followed on a duty plan that changes, nor from a state with imperfect
  repairs; a covariate of the time since the service (a ramp, not a
  step) is not a schedule the engine takes.
- A test that can miss a failure or that takes the device off line, and
  an inspection that finds a warning before the failure (on condition),
  are not valued on dates yet either.
- **A task's value for ever.** `task_value` values the forecast's window;
  the long run beyond it is not joined to it (as for the part's own
  failure costs above).
- **Discounted ranges.** The 5th to 95th percentiles stay undiscounted.
- **Utilisation as whole runs lost.** `utilisation=0.9` counts each
  planned hour and start as 0.9, an expected value. The dates when 90%
  of runs go ahead (the nth run to go ahead follows a negative binomial)
  are not worked out yet.
- **Exact values on a duty plan that changes.** A unit on such a plan
  is simulated (UC-64). Where its repairs and tasks take no time it is
  its operating-time process seen through `D`, so its exact curves could
  be the operating-time ones at `D(t)`; not built yet. Nor are tests
  counted in operating time on such a plan, or a start other than new.

## Candidates

- The dates when only some runs go ahead (UC-10's 90% dates).
- Financial years and quarters as periods.
- A calendar feed (.ics) from `to_rrule()`.
