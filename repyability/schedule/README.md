# Schedules: use cases

<!-- Duty plans and the date <-> usage map (UC-32 to UC-37). The use
cases of recurrence rules and tasks on dates come first in this file. -->

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
  `also=Recurrence.every(12, "months", ...)`.
- **The answer:** 250 h at 15 h a month takes 16⅔ months, so **the
  12-month limit sets every date** (14 November 2027, 14 November 2028;
  `limits` says `"also"` for each); the hours would not be reached until
  4 April 2028, about 4.7 months after the first calendar date. The
  hours limit is doing nothing.
- **Checks:** `test_two_limits_the_calendar_first`, and
  `test_two_limits_the_hours_first` for the reverse on the pump.

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

### What is not built yet

- **Utilisation as whole runs lost.** `utilisation=0.9` counts each
  planned hour and start as 0.9, an expected value. The dates when 90%
  of runs go ahead (the nth run to go ahead follows a negative binomial)
  are not worked out yet.
- **A duty that changes inside a `RepairableRBD`.** `schedule()` gives
  the duty as steps over calendar time, for that later work; today a
  block takes one average share, `duty()`.
