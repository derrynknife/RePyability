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
| UC-10 | A pump on weekday shifts, serviced every 500 running hours: when is each service due? | **Built** (every run going ahead): `DutyPlan`, `Usage`, `Task.due(usage=)`; the dates with 90% of runs going ahead not yet |
| UC-28 | The first Tuesday of November every year | **Built**: `Recurrence.yearly` |
| UC-29 | The first, or the fifth, Saturday of each month, and the months without one | **Built**: `Recurrence.monthly(nth=)`, `Recurrence.notes` |
| UC-30 | Two limits, whichever comes first, and a task waiting for the crew's visit | **Built**: `Task(also=, visits=, tolerance=, last_done=)` |
| UC-31 | The work in each month: how many tasks, hours and cost | **Built**: `work_by_period` |
| UC-32 | A pump on weekday shifts: its hours by month, and its 2,000-hour service on dates | **Built**: `DutyPlan`, `UsageForecast` |
| UC-33 | A concrete run: when is the 500-hour check due? | **Built**: `Runs`, a crossing's stops |
| UC-34 | About 3 h a day, or 2 to 4: when are the 250-hour inspections? | **Built**: `Rate`, ranges |
| UC-35 | 250 hours or 12 months at 15 h a month: which limit sets the date? | **Built**: `Task(every=Usage(...), also=Recurrence...)` |
| UC-36 | Exact runs for now, a rate after, with no duty made at the handover | **Built**: `background`, `specific`, `fill` |
| UC-37 | A night shift the nights the clocks change | **Built**: `time_zone=` |
| UC-38 | The wear ring on the weekday pump: expected failures and cost by month, and the range | **Built**: `failures_by_period` |
| UC-39 | A system's events by calendar month | **Built**: `events_by_period` |
| UC-40 | Shuts on dates as opportunities on the part's clock | **Built**: `Opportunity.from_recurrence` |
| UC-41 | The annual overhaul on the first Tuesday of November, as a diagram's shutdown | **Built**: `shutdown_spec` (simulated; the exact methods take shuts on a period) |
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
- **Not built:** the exact methods (`expected_events`, `expected_cost`,
  and so `events_by_period`) take shuts on a period only; a diagram with
  shuts at given times is simulated (`cost()`, `availability()`), and
  `events_by_period` refuses it with the diagram's own message.
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

### What is not built yet

- **Utilisation as whole runs lost.** `utilisation=0.9` counts each
  planned hour and start as 0.9, an expected value. The dates when 90%
  of runs go ahead (the nth run to go ahead follows a negative binomial)
  are not worked out yet.
- **A duty that changes inside a `RepairableRBD`.** `schedule()` gives
  the duty as steps over calendar time, for that later work; today a
  block takes one average share, `duty()`.
- **Shuts at given times in the exact methods.** `shutdown_spec` puts
  dated shuts on a diagram, which the simulations take; the exact
  `expected_events` and `expected_cost` (and so `events_by_period`) take
  shuts on a period only.

## Candidates

- The dates when only some runs go ahead (UC-10's 90% dates).
- Financial years and quarters as periods.
- A calendar feed (.ics) from `to_rrule()`.
