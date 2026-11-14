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

## The use cases, by number

| # | The question | State |
|---|---|---|
| UC-2 | A weekly task first done on 15 July: how many times through the rest of the year? | **Built**: `Recurrence.every`, `Task.due` |
| UC-3 | What does it cost? | **Built**: `Task(cost=)`, `TaskDue.cost` |
| UC-4 | The first Tuesday of the month, and the 30th | **Built**: `Recurrence.monthly` |
| UC-10 | A pump on weekday shifts, serviced every 500 running hours: when is each service due? | Answered by hand (below); built with the duty plans |
| UC-28 | The first Tuesday of November every year | **Built**: `Recurrence.yearly` |
| UC-29 | The first, or the fifth, Saturday of each month, and the months without one | **Built**: `Recurrence.monthly(nth=)`, `Recurrence.notes` |
| UC-30 | Two limits, whichever comes first, and a task waiting for the crew's visit | **Built**: `Task(also=, visits=, tolerance=, last_done=)` |
| UC-31 | The work in each month: how many tasks, hours and cost | **Built**: `work_by_period` |

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

- **Status:** answered by hand; to be built with the duty plans and tasks
  by use (`DutyPlan`, `Usage`), which plug into `Task` through its limits
  (see `_task._Limit`).
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

## Candidates

- Tasks by use on a duty plan (UC-10), and "250 hours or 12 months".
- Failures and costs by calendar month through the map between dates and
  operating hours.
- Shuts on dates laid on model time (`Opportunity` from a `Recurrence`).
- Financial years and quarters as periods.
- A calendar feed (.ics) from `to_rrule()`.
